//! `dsyevd_device`: the hybrid host/device `DSYEVD`.
//!
//! One persistent device matrix `M` (`lda × n`) holds A during
//! tridiagonalisation and is reused as Z by `DSTEDC` (identity refill) and
//! `DORMTR`; Z is then read in column chunks of at most
//! [`READ_CHUNK_BYTES`] straight into the caller's `a`. Under
//! [`Exactness::Reference`] and a backend without contraction the result is
//! bit-identical to [`crate::dsyevd_alloc`] (and so to Reference-LAPACK
//! called with `LWORK`/`LIWORK` from [`crate::syevd_opt_work_len`]).
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::flavor::DeviceBlas;
use super::ops::Lanes;
use super::workspace::{Counters, Phase, TransferCounters, read_range};
use super::{geometry, ormtr, probe, stedc, sytrd};
use crate::aux::{PREC, SAFMIN, SclType, dlansy_m, dlascl};
use crate::flavor::RefBlas;
use crate::ptr::V;
use crate::sytrd::{ORM_NB, SYTRD_NB, TSIZE};
use crate::{Job, LapackError, Uplo};

/// Smallest merge rank `k` a lean merge is used for (below it the staged
/// `k × k` upload is negligible and bit-exact).
pub const LEAN_MIN_K: usize = 3;

/// Largest single read of the final Z (host transient is one chunk).
pub const READ_CHUNK_BYTES: usize = 64 << 20;

/// Device memory the solver may use, in bytes. Picks the merge `U` mode (for
/// [`MergeU::Auto`]) and the merge row chunk `R`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceBudget {
    pub bytes: usize,
}

impl DeviceBudget {
    /// No limit.
    pub const UNLIMITED: DeviceBudget = DeviceBudget { bytes: usize::MAX };
}

/// How the `DSTEDC` merge obtains the secular eigenvector matrix `U`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeU {
    /// Upload the `k × k` matrix per merge (fastest; bit-exact order).
    Staged,
    /// Rebuild `U` on the device from O(k) vectors (tolerance only).
    Lean,
    /// `Staged` when it fits the budget, else `Lean`.
    Auto,
}

/// Arithmetic contract of the device path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Exactness {
    /// Reference operation order everywhere (staged `U`, `DLARFB`'s own
    /// sequence). Bit-identical to Reference-LAPACK on a non-contracting
    /// backend.
    #[default]
    Reference,
    /// Allows reordered operations; results are tolerance-gated.
    Fast,
}

/// Persistent device state reused across calls (e.g. SCF iterations): the
/// matrix buffer `M`, the K1 output `y`, the lane table and the traffic
/// counters. Grows only when a call needs a larger `lda × n`.
pub struct DeviceWorkspace<R: Runtime> {
    pub(super) client: ComputeClient<R>,
    pub(super) lanes: Lanes,
    pub(super) counters: Counters,
    /// `M` (the real part of `M` for `zheevd_device`).
    pub(super) m: Handle,
    pub(super) m_reals: usize,
    /// 1 for a real `M`, 2 when `m_im` is live.
    m_parts: usize,
    /// Imaginary part of `M` (`zheevd_device` only; 1 real otherwise).
    pub(super) m_im: Handle,
    pub(super) y: Handle,
    pub(super) y_im: Handle,
    pub(super) y_reals: usize,
    budget: DeviceBudget,
    u: MergeU,
    pub(super) exactness: Exactness,
    chunk_rows: usize,
    pub(super) work: Vec<f64>,
    pub(super) iwork: Vec<usize>,
}

/// Bytes the device path needs at size `n` beyond the budget-independent
/// minimum, split as `(resident, staged_merge_peak, per_chunk_row)`.
fn device_bytes(lda: usize, n: usize, width: usize) -> (usize, usize, usize) {
    let kpad = ORM_NB.div_ceil(width) * width;
    let resident = 8 * (lda * n + lda);
    let sytrd = 8 * 2 * SYTRD_NB * lda + 8 * lda;
    let ormtr = 8 * (2 * n * kpad + 2 * n * kpad + 3 * kpad * kpad + 2 * kpad);
    // Staged U: s_top + s_bot ≤ 2k² with k ≤ n, plus three permutations.
    let staged = 8 * 2 * n * n + 4 * 3 * n;
    (resident + sytrd.max(ormtr), staged, 8 * n)
}

impl<R: Runtime> DeviceWorkspace<R> {
    /// A workspace for problems up to `n` (it grows on demand). `u` and the
    /// budget pick the merge mode and row chunk. Exactness defaults to
    /// [`Exactness::Reference`].
    pub fn new(client: &ComputeClient<R>, n: usize, job: Job, budget: DeviceBudget, u: MergeU) -> Self {
        let width = geometry::vector_width(client, 8);
        let lanes = Lanes::new(client, width);
        let mut ws = DeviceWorkspace {
            client: client.clone(),
            lanes,
            counters: Counters::default(),
            m: client.empty(8),
            m_reals: 0,
            m_parts: 1,
            m_im: client.empty(8),
            y: client.empty(8),
            y_im: client.empty(8),
            y_reals: 0,
            budget,
            u,
            exactness: Exactness::Reference,
            chunk_rows: 0,
            work: Vec::new(),
            iwork: Vec::new(),
        };
        let _ = job;
        ws.reserve(n);
        ws
    }

    /// Select the arithmetic contract.
    pub fn with_exactness(mut self, exactness: Exactness) -> Self {
        self.exactness = exactness;
        self
    }

    /// Override the merge row chunk `R` (0 = pick from the budget).
    pub fn with_chunk_rows(mut self, rows: usize) -> Self {
        self.chunk_rows = rows;
        self
    }

    /// Upload/read/launch counters accumulated since the last reset.
    pub fn counters(&self) -> TransferCounters {
        self.counters.snapshot()
    }

    /// Reset the counters (and the phase times).
    pub fn reset_counters(&self) {
        self.counters.reset();
    }

    /// Wall time per phase since the last reset; zero unless
    /// [`super::workspace::set_phase_profiling`] is on.
    pub fn phase_times(&self) -> super::workspace::PhaseTimes {
        self.counters.phase_times()
    }

    /// Return the runtime pool's free transient pages to the device
    /// (`client.memory_cleanup()`). Between calls of a repeating size the
    /// pool would otherwise reuse them, so call this only when the device
    /// memory is needed elsewhere.
    pub fn release_transients(&self) {
        self.client.memory_cleanup();
    }

    /// Device bytes held persistently.
    pub fn resident_bytes(&self) -> usize {
        8 * self.m_parts * self.m_reals + 16 * self.y_reals + 4 * self.lanes.width
    }

    pub(super) fn reserve(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        let lda = geometry::lda_for(&self.client, n);
        if lda > self.y_reals {
            let client = &self.client;
            (self.y, self.y_im) = client
                .memory_persistent_allocation((), |_| (client.empty(lda * 8), client.empty(lda * 8)))
                .expect("persistent allocation");
            self.y_reals = lda;
        }
    }

    /// Make `staged` the resident `M`. cubecl 0.10 has no partial write into
    /// an existing handle, so the one-copy image becomes a new persistent
    /// buffer; the previous `M` is released first, so the pool hands back
    /// its page when the size repeats and at most one `M` is ever alive.
    pub(super) fn stage(&mut self, staged: Vec<f64>) {
        let len = staged.len();
        self.counters.upload(len * 8);
        self.m = self.client.empty(8);
        self.m_im = self.client.empty(8);
        let client = &self.client;
        self.m = client
            .memory_persistent_allocation(staged, |v| client.create(cubecl::bytes::Bytes::from_elems(v)))
            .expect("persistent allocation");
        self.m_reals = len;
        self.m_parts = 1;
    }

    /// [`Self::stage`] without the host expansion: upload the caller's
    /// `n × n` triangle as it stands and mirror it into `M` with the staging
    /// kernel. `stage_symmetric` is an O(n²) loop on the driver thread and
    /// was 104 ms of a 1.24 s n=2048 solve (476 ms at n=4096) on the 2-vCPU
    /// measurement host; the kernel is a pure copy, so the image is
    /// bit-identical either way. It also uploads `n²` reals instead of
    /// `lda·n`.
    pub(super) fn stage_from_host(&mut self, a: &[f64], n: usize, lda: usize, upper: bool) {
        let src = super::workspace::upload(&self.client, &a[..n * n]);
        self.counters.upload(n * n * 8);
        self.stage_from_device(&src, n, lda, upper);
    }

    /// [`Self::stage`] from an `n × n` device buffer (two-triangle staging
    /// kernel; no host transfer).
    pub(super) fn stage_from_device(&mut self, src: &Handle, n: usize, lda: usize, upper: bool) {
        self.m = self.client.empty(8);
        self.m_im = self.client.empty(8);
        let client = &self.client;
        self.m = client.memory_persistent_allocation((), |_| client.empty(lda * n * 8)).expect("persistent allocation");
        super::ops::stage_sym::<R, f64>(client, src, &self.m, n, lda, upper, false);
        self.counters.launch();
        self.m_reals = lda * n;
        self.m_parts = 1;
    }

    /// [`Self::stage_from_host`] for planar complex input: upload the two
    /// `n × n` planes and mirror them with the Hermitian staging kernel
    /// (`herm_im` negates the mirrored imaginary part and zeroes the
    /// diagonal), so `stage_hermitian`'s O(n²) host loop is not run.
    pub(super) fn stage_complex_from_host(&mut self, a_re: &[f64], a_im: &[f64], n: usize, lda: usize, upper: bool) {
        let client = self.client.clone();
        let sre = super::workspace::upload(&client, &a_re[..n * n]);
        let sim = super::workspace::upload(&client, &a_im[..n * n]);
        self.counters.upload(n * n * 8);
        self.counters.upload(n * n * 8);
        self.stage_complex_from_device(&sre, &sim, n, lda, upper);
    }

    /// [`Self::stage_from_device`] for planar complex input (Hermitian staging).
    pub(super) fn stage_complex_from_device(&mut self, re: &Handle, im: &Handle, n: usize, lda: usize, upper: bool) {
        self.m = self.client.empty(8);
        self.m_im = self.client.empty(8);
        let client = &self.client;
        (self.m, self.m_im) = client
            .memory_persistent_allocation((), |_| (client.empty(lda * n * 8), client.empty(lda * n * 8)))
            .expect("persistent allocation");
        super::ops::stage_sym::<R, f64>(client, re, &self.m, n, lda, upper, false);
        super::ops::stage_sym::<R, f64>(client, im, &self.m_im, n, lda, upper, true);
        self.counters.launch();
        self.counters.launch();
        self.m_reals = lda * n;
        self.m_parts = 2;
    }

    /// [`Self::stage`] for a planar complex image.
    pub(super) fn stage_complex(&mut self, re: Vec<f64>, im: Vec<f64>) {
        let len = re.len();
        self.counters.upload(len * 8);
        self.counters.upload(len * 8);
        self.m = self.client.empty(8);
        self.m_im = self.client.empty(8);
        let client = &self.client;
        (self.m, self.m_im) = client
            .memory_persistent_allocation((re, im), |(re, im)| {
                (
                    client.create(cubecl::bytes::Bytes::from_elems(re)),
                    client.create(cubecl::bytes::Bytes::from_elems(im)),
                )
            })
            .expect("persistent allocation");
        self.m_reals = len;
        self.m_parts = 2;
    }

    /// `(merge mode is lean, R)` for size `n`, or `WorkspaceTooSmall`.
    /// `parts` is 2 for the planar complex path (every matrix-sized buffer
    /// doubles; the real divide-and-conquer merge does not).
    pub(super) fn plan(&self, n: usize, wantz: bool, parts: usize) -> Result<(bool, usize), LapackError> {
        let lda = geometry::lda_for(&self.client, n);
        let w = self.lanes.width;
        let (fixed, staged, per_row) = device_bytes(lda, n, w);
        let fixed = fixed * parts;
        let budget = self.budget.bytes;
        if !wantz {
            return if fixed <= budget {
                Ok((false, 0))
            } else {
                Err(LapackError::WorkspaceTooSmall { need: fixed, got: budget })
            };
        }
        let min_rows = w;
        let lean = match (self.exactness, self.u) {
            (Exactness::Reference, MergeU::Lean) => {
                return Err(LapackError::InvalidArgument {
                    arg: "u",
                    reason: "MergeU::Lean is never bit-exact; use Exactness::Fast".into(),
                });
            }
            (_, MergeU::Staged) | (Exactness::Reference, MergeU::Auto) => false,
            (Exactness::Fast, MergeU::Lean) => true,
            (Exactness::Fast, MergeU::Auto) => fixed.saturating_add(staged).saturating_add(per_row * min_rows) > budget,
        };
        // Lean: O(n) merge vectors (perms, secular indices, ẑ, Λ, o, τ, ν).
        let base = fixed.saturating_add(if lean { 8 * 8 * n } else { staged });
        if base.saturating_add(per_row * min_rows) > budget {
            return Err(LapackError::WorkspaceTooSmall { need: base + per_row * min_rows, got: budget });
        }
        let r = if self.chunk_rows > 0 {
            self.chunk_rows
        } else {
            let fit = (budget - base) / per_row;
            n.div_ceil(4).max(min_rows).min(fit.max(min_rows))
        };
        Ok((lean, r))
    }
}

/// `DSYEVD` on the device: `a` is column-major `n × n` (valid in its `uplo`
/// triangle); on return `w` holds ascending eigenvalues and, for
/// [`Job::Vectors`], `a` holds the eigenvectors. Bit-identical to
/// [`crate::dsyevd_alloc`] under [`Exactness::Reference`] on a
/// non-contracting backend (see [`probe::contraction_probe`]).
pub fn dsyevd_device<R: Runtime>(
    client: &ComputeClient<R>,
    job: Job,
    uplo: Uplo,
    n: usize,
    a: &mut [f64],
    w: &mut [f64],
    ws: &mut DeviceWorkspace<R>,
) -> Result<(), LapackError> {
    dsyevd_device_impl::<R, RefBlas>(client, job, uplo, n, a, w, ws, None, false).map(|_| ())
}

/// A column-major `lda × n` f64 matrix held on the device (rows `n..lda` are
/// padding). [`DeviceMatrix::compact`] gives the plain `n × n` buffer.
#[derive(Clone, Debug)]
pub struct DeviceMatrix {
    pub handle: Handle,
    pub lda: usize,
    pub n: usize,
}

impl DeviceMatrix {
    /// A new `n × n` column-major buffer holding the leading block.
    pub fn compact<R: Runtime>(&self, client: &ComputeClient<R>) -> Handle {
        super::ops::compact::<R, f64>(client, &self.handle, self.n, self.lda)
    }
}

/// Device-resident [`dsyevd_device`]: `a` is an `n × n` column-major f64
/// device buffer (logical length `n²`, valid in its `uplo` triangle, left
/// unchanged). The eigenvectors stay on the device and are returned for
/// [`Job::Vectors`]; `w` receives the eigenvalues. `a` is read once (the
/// hybrid algorithm keeps the Householder vectors on the host) and `M` is
/// staged from it on the device; Z is never read back. Bit-identical to
/// [`dsyevd_device`].
pub fn dsyevd_device_resident<R: Runtime>(
    client: &ComputeClient<R>,
    job: Job,
    uplo: Uplo,
    n: usize,
    a: &Handle,
    w: &mut [f64],
    ws: &mut DeviceWorkspace<R>,
) -> Result<Option<DeviceMatrix>, LapackError> {
    probe::require_f64(client)?;
    let mut host = vec![0.0f64; n * n];
    read_flat_chunked(client, &ws.counters, a, n * n, &mut host);
    dsyevd_device_impl::<R, RefBlas>(client, job, uplo, n, &mut host, w, ws, Some(a), true)
}

/// Read `out.len()` reals from the start of `h` (logical length `len`) in
/// chunks of at most [`READ_CHUNK_BYTES`].
pub(super) fn read_flat_chunked<R: Runtime>(
    client: &ComputeClient<R>,
    counters: &Counters,
    h: &Handle,
    len: usize,
    out: &mut [f64],
) {
    let chunk = READ_CHUNK_BYTES / 8;
    let mut i = 0;
    while i < len {
        let l = chunk.min(len - i);
        read_range(client, h, len, i, l, &mut out[i..i + l]);
        counters.read(l * 8);
        i += l;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dsyevd_device_impl<R: Runtime, B: DeviceBlas>(
    client: &ComputeClient<R>,
    job: Job,
    uplo: Uplo,
    n: usize,
    a: &mut [f64],
    w: &mut [f64],
    ws: &mut DeviceWorkspace<R>,
    source: Option<&Handle>,
    resident: bool,
) -> Result<Option<DeviceMatrix>, LapackError> {
    probe::require_f64(client)?;
    if a.len() < n * n {
        return Err(LapackError::InvalidArgument {
            arg: "a",
            reason: format!("need n*n = {} entries, got {}", n * n, a.len()),
        });
    }
    if w.len() < n {
        return Err(LapackError::InvalidArgument {
            arg: "w",
            reason: format!("need n = {n} entries, got {}", w.len()),
        });
    }
    let wantz = job == Job::Vectors;
    if n == 0 {
        return Ok(None);
    }
    if n == 1 {
        w[0] = a[0];
        if !wantz {
            return Ok(None);
        }
        a[0] = 1.0;
        if !resident {
            return Ok(None);
        }
        let lda = geometry::lda_for(client, 1);
        let mut img = vec![0.0f64; lda];
        img[0] = 1.0;
        ws.stage(img);
        return Ok(Some(DeviceMatrix { handle: ws.m.clone(), lda, n }));
    }
    let (lean, chunk_rows) = ws.plan(n, wantz, 1)?;
    let lean = lean && B::FAST_OK;
    let mut t = std::time::Instant::now();
    ws.reserve(n);
    let upper = uplo == Uplo::Upper;
    let lda = geometry::lda_for(client, n);

    // DSYEVD scaling (host, O(n²) reads of the caller's triangle).
    let smlnum = SAFMIN / PREC;
    let bignum = 1.0 / smlnum;
    let (rmin, rmax) = (smlnum.sqrt(), bignum.sqrt());
    // SAFETY: a has n*n entries (checked).
    let anrm = unsafe { dlansy_m(upper, n, V::new(a), n) };
    let sigma = if anrm > 0.0 && anrm < rmin {
        Some(rmin / anrm)
    } else if anrm > rmax {
        Some(rmax / anrm)
    } else {
        None
    };
    if let Some(s) = sigma {
        // SAFETY: as above.
        unsafe { dlascl(if upper { SclType::U } else { SclType::L }, 1.0, s, n, n, V::new(a), n) };
    }

    // Host O(n) and panel workspace.
    let mut e = vec![0.0f64; n - 1];
    let mut tau = vec![0.0f64; n - 1];
    let lwork = (n * SYTRD_NB).max(n * ORM_NB + TSIZE).max(B::orm_lwork(n));
    if ws.work.len() < lwork {
        ws.work.resize(lwork, 0.0);
    }

    // Tridiagonalisation: stage A (both triangles) as the resident M, from
    // the caller's device buffer when it holds the (unscaled) input.
    match source {
        Some(src) if sigma.is_none() => ws.stage_from_device(src, n, lda, upper),
        _ => ws.stage_from_host(a, n, lda, upper),
    }
    ws.counters.mark(client, Phase::Stage, &mut t);
    {
        let DeviceWorkspace { lanes, counters, m, y, y_reals, work, .. } = ws;
        let mut eng = sytrd::DeviceTrd::<'_, R, B> {
            client,
            lanes,
            counters,
            m: m.clone(),
            lda,
            n,
            y: y.clone(),
            y_len: *y_reals,
            buf: Vec::new(),
            flavor: std::marker::PhantomData,
        };
        // SAFETY: a/w/e/tau/work lengths checked or allocated; the engine
        // reads and writes device buffers sized `lda × n` (≤ m_reals).
        unsafe {
            crate::sytrd::dsytrd_with(
                &mut eng,
                upper,
                n,
                V::new(a),
                n,
                V::new(w),
                V::new(&mut e),
                V::new(&mut tau),
                V::new(work),
                n * SYTRD_NB,
            );
        }
    }
    ws.counters.mark(client, Phase::Trd, &mut t);

    let info;
    if !wantz {
        // SAFETY: w has n entries, e has n-1.
        info = unsafe { crate::steqr::dsterf_raw(n, V::new(w), V::new(&mut e)) };
        ws.counters.mark(client, Phase::Stedc, &mut t);
    } else {
        let (lw, liw) = crate::stedc::stedc_work_len(n);
        let mut dwork = vec![0.0f64; lw.max(1)];
        if ws.iwork.len() < liw.max(1) {
            ws.iwork.resize(liw.max(1), 0);
        }
        let mut dummy = [0.0f64; 1];
        {
            let DeviceWorkspace { lanes, counters, m, iwork, .. } = ws;
            let mut eng =
                stedc::DeviceDc::<'_, R, B>::new_flavor(client, lanes, counters, m.clone(), lda, n, chunk_rows);
            eng.lean_min_k = if lean { LEAN_MIN_K } else { usize::MAX };
            // SAFETY: w/e/dwork/iwork lengths allocated above; the device
            // engine never touches the host Z view (a 1-element dummy).
            info = unsafe {
                crate::stedc::dstedc_i_with(
                    &mut eng,
                    n,
                    V::new(w),
                    V::new(&mut e),
                    V::new(&mut dummy),
                    n,
                    V::new(&mut dwork),
                    V::new(iwork),
                )
            };
        }
        drop(dwork);
        ws.counters.mark(client, Phase::Stedc, &mut t);
        let lwork_orm = B::orm_lwork(n);
        let DeviceWorkspace { lanes, counters, m, work, exactness, .. } = ws;
        let blocked = crate::sytrd::orm_blocking(n, n - 1, lwork_orm).is_some();
        if B::ORM_ON_DEVICE == 2 && blocked {
            let mut eng = ormtr::DeviceOrmOb { client, lanes, counters, z: m.clone(), lda, ncols: n };
            // SAFETY: as the Reference engine below.
            unsafe {
                crate::sytrd::dormtr_ln_with(
                    &mut eng,
                    upper,
                    n,
                    n,
                    V::new(a),
                    n,
                    V::new(&mut tau),
                    V::new(&mut dummy),
                    n,
                    V::new(work),
                    lwork_orm,
                );
            }
            counters.mark(client, Phase::Orm, &mut t);
            if !resident {
                read_columns(client, counters, m, lda, n, a);
            }
            counters.mark(client, Phase::Readback, &mut t);
        } else if B::ORM_ON_DEVICE == 1 && blocked {
            let mut eng = ormtr::DeviceOrm {
                client,
                lanes,
                counters,
                z: m.clone(),
                lda,
                ncols: n,
                fast: B::FAST_OK && *exactness == Exactness::Fast,
            };
            // SAFETY: a (reflectors), tau and work lengths checked/allocated;
            // the engine never touches the host C view on the blocked path.
            unsafe {
                crate::sytrd::dormtr_ln_with(
                    &mut eng,
                    upper,
                    n,
                    n,
                    V::new(a),
                    n,
                    V::new(&mut tau),
                    V::new(&mut dummy),
                    n,
                    V::new(work),
                    lwork_orm,
                );
            }
            counters.mark(client, Phase::Orm, &mut t);
            if !resident {
                read_columns(client, counters, m, lda, n, a);
            }
            counters.mark(client, Phase::Readback, &mut t);
        } else {
            // Unblocked DORM2L/DORM2R (n ≤ 33 on the Reference track), or a
            // flavour whose DLARFB has no device kernels: apply on the host.
            let mut z = vec![0.0f64; n * n];
            read_columns(client, counters, m, lda, n, &mut z);
            // SAFETY: as above, with a host C view of n*n entries.
            unsafe {
                crate::sytrd::dormtr_ln_with(
                    &mut crate::sytrd::HostOrm::<B>::new(),
                    upper,
                    n,
                    n,
                    V::new(a),
                    n,
                    V::new(&mut tau),
                    V::new(&mut z),
                    n,
                    V::new(work),
                    lwork_orm,
                );
            }
            a[..n * n].copy_from_slice(&z);
            if resident {
                let mut img = vec![0.0f64; lda * n];
                for j in 0..n {
                    img[j * lda..j * lda + n].copy_from_slice(&a[j * n..(j + 1) * n]);
                }
                ws.stage(img);
            }
        }
    }
    if let Some(s) = sigma {
        // SAFETY: w has n entries.
        unsafe { B::dscal(n, 1.0 / s, V::new(w), 1) };
    }
    if info != 0 {
        return Err(LapackError::NoConvergence { routine: if wantz { "dstedc" } else { "dsterf" }, info });
    }
    Ok(if resident && wantz { Some(DeviceMatrix { handle: ws.m.clone(), lda, n }) } else { None })
}

/// Read columns `0..n` (rows `0..n`) of `M` into column-major `out` in chunks
/// of at most [`READ_CHUNK_BYTES`].
pub(super) fn read_columns<R: Runtime>(
    client: &ComputeClient<R>,
    counters: &Counters,
    m: &Handle,
    lda: usize,
    n: usize,
    out: &mut [f64],
) {
    let cols = (READ_CHUNK_BYTES / (8 * lda)).max(1);
    let mut buf = vec![0.0f64; cols.min(n) * lda];
    let mut j0 = 0;
    while j0 < n {
        let nc = cols.min(n - j0);
        let len = (nc - 1) * lda + n;
        read_range(client, m, lda * n, j0 * lda, len, &mut buf[..len]);
        counters.read(len * 8);
        for j in 0..nc {
            out[(j0 + j) * n..(j0 + j + 1) * n].copy_from_slice(&buf[j * lda..j * lda + n]);
        }
        j0 += nc;
    }
}

/// **OpenBLAS track.** [`dsyevd_device`] with the OpenBLAS-order kernels:
/// bit-identical to [`crate::openblas::dsyevd_host`] (scipy's OpenBLAS
/// 0.3.30, SkylakeX core, one thread, its own workspace query).
pub fn dsyevd_device_openblas<R: Runtime>(
    client: &ComputeClient<R>,
    job: Job,
    uplo: Uplo,
    n: usize,
    a: &mut [f64],
    w: &mut [f64],
    ws: &mut DeviceWorkspace<R>,
) -> Result<(), LapackError> {
    crate::openblas::require_fma();
    dsyevd_device_impl::<R, crate::openblas::OpenBlas>(client, job, uplo, n, a, w, ws, None, false).map(|_| ())
}
