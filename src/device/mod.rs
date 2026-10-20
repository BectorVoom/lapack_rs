//! On-device path (feature `device`): CubeCL kernels generic over the float
//! type and the runtime, driven by hybrid host/device drivers.

pub(crate) mod complex;
pub(crate) mod flavor;
pub mod geometry;
pub mod kernels;
pub mod ops;
pub(crate) mod ormtr;
pub mod probe;
pub mod stedc;
pub mod syevd;
pub(crate) mod sytrd;
pub mod workspace;

pub use complex::{DeviceZMatrix, zheevd_device, zheevd_device_openblas, zheevd_device_resident};
pub use syevd::{
    DeviceBudget, DeviceMatrix, DeviceWorkspace, Exactness, MergeU, dsyevd_device, dsyevd_device_openblas,
    dsyevd_device_resident,
};
pub use workspace::TransferCounters;

use cubecl::prelude::*;

use crate::ptr::V;
use crate::{LapackError, Uplo};

/// Hybrid `DSYTRD` (blocked, `NB = NX = 32`): bit-identical to the host
/// [`crate::blocks::dsytrd`] with `LWORK = 32·n`. `a`, `d`, `e`, `tau` are as
/// in LAPACK (`LDA = n`). Returns the transfer counters.
pub fn dsytrd_device<R: Runtime>(
    client: &ComputeClient<R>,
    uplo: Uplo,
    n: usize,
    a: &mut [f64],
    d: &mut [f64],
    e: &mut [f64],
    tau: &mut [f64],
) -> Result<workspace::TransferCounters, LapackError> {
    probe::require_f64(client)?;
    let chk = |name: &'static str, len: usize, need: usize| {
        if len < need {
            Err(LapackError::InvalidArgument { arg: name, reason: format!("need {need} entries, got {len}") })
        } else {
            Ok(())
        }
    };
    chk("a", a.len(), n * n)?;
    chk("d", d.len(), n)?;
    chk("e", e.len(), n.saturating_sub(1).max(1))?;
    chk("tau", tau.len(), n.saturating_sub(1).max(1))?;
    let counters = workspace::Counters::default();
    if n == 0 {
        return Ok(counters.snapshot());
    }
    let upper = uplo == Uplo::Upper;
    let width = geometry::vector_width(client, 8);
    let lda = geometry::lda_for(client, n);
    let lanes = ops::Lanes::new(client, width);
    let staged = sytrd::stage_symmetric(a, n, lda, upper);
    counters.upload(staged.len() * 8);
    let m = workspace::upload_vec(client, staged);
    let y = client.empty(lda * 8);
    let lwork = 32 * n;
    let mut work = vec![0.0f64; lwork];
    let mut eng = sytrd::DeviceTrd::<'_, R, crate::flavor::RefBlas> {
        client,
        lanes: &lanes,
        counters: &counters,
        m,
        lda,
        n,
        y,
        y_len: lda,
        buf: Vec::new(),
        flavor: std::marker::PhantomData,
    };
    // SAFETY: lengths checked above; the host views cover distinct slices and
    // the engine only reads/writes device buffers sized from `lda`/`n`.
    unsafe {
        crate::sytrd::dsytrd_with(
            &mut eng,
            upper,
            n,
            V::new(a),
            n,
            V::new(d),
            V::new(e),
            V::new(tau),
            V::new(&mut work),
            lwork,
        );
    }
    Ok(counters.snapshot())
}

/// Hybrid `DSTEDC('I')`: `d`, `e` on the host, eigenvectors written to `z`
/// (`n × n`, column-major) at the end. Bit-identical to the host `dstedc`.
/// `chunk_rows` is the merge row chunk `R` (0 = `⌈n/4⌉`).
pub fn dstedc_device<R: Runtime>(
    client: &ComputeClient<R>,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    z: &mut [f64],
    chunk_rows: usize,
) -> Result<workspace::TransferCounters, LapackError> {
    probe::require_f64(client)?;
    if d.len() < n || e.len() < n.saturating_sub(1).max(1) || z.len() < n * n {
        return Err(LapackError::InvalidArgument { arg: "d/e/z", reason: format!("need n={n}, n-1 and n*n entries") });
    }
    let counters = workspace::Counters::default();
    if n == 0 {
        return Ok(counters.snapshot());
    }
    let width = geometry::vector_width(client, 8);
    let lda = geometry::lda_for(client, n);
    let lanes = ops::Lanes::new(client, width);
    let zh = client.empty(lda * n * 8);
    let (lw, liw) = crate::stedc::stedc_work_len(n);
    let mut work = vec![0.0f64; lw.max(1)];
    let mut iwork = vec![0usize; liw.max(1)];
    let r = if chunk_rows == 0 { n.div_ceil(4) } else { chunk_rows };
    let mut eng = stedc::DeviceDc::new(client, &lanes, &counters, zh.clone(), lda, n, r);
    let mut dummy = [0.0f64; 1];
    // SAFETY: d/e/work/iwork lengths checked or allocated above; the device
    // engine never touches the host `z` view (a 1-element dummy).
    let info = unsafe {
        crate::stedc::dstedc_i_with(
            &mut eng,
            n,
            V::new(d),
            V::new(e),
            V::new(&mut dummy),
            n,
            V::new(&mut work),
            V::new(&mut iwork),
        )
    };
    drop(eng);
    let mut img = vec![0.0f64; lda * n];
    workspace::read_range(client, &zh, lda * n, 0, lda * n, &mut img);
    counters.read(lda * n * 8);
    for j in 0..n {
        z[j * n..j * n + n].copy_from_slice(&img[j * lda..j * lda + n]);
    }
    if info != 0 {
        return Err(LapackError::NoConvergence { routine: "dstedc", info });
    }
    Ok(counters.snapshot())
}

/// Hybrid `DORMTR('L', UPLO, 'N', N, N, A, N, TAU, C, N)` with `LWORK =
/// N·32 + TSIZE` (the blocked path when `n > 33`), `C` uploaded, updated on
/// the device and read back. Bit-identical to the host `dormtr`.
pub fn dormtr_device<R: Runtime>(
    client: &ComputeClient<R>,
    uplo: Uplo,
    n: usize,
    a: &mut [f64],
    tau: &mut [f64],
    c: &mut [f64],
) -> Result<workspace::TransferCounters, LapackError> {
    probe::require_f64(client)?;
    if a.len() < n * n || c.len() < n * n || tau.len() < n.saturating_sub(1).max(1) {
        return Err(LapackError::InvalidArgument {
            arg: "a/tau/c",
            reason: format!("need n*n, n-1, n*n entries (n={n})"),
        });
    }
    let counters = workspace::Counters::default();
    if n <= 1 {
        return Ok(counters.snapshot());
    }
    let lwork = n * crate::sytrd::SYTRD_NB + crate::sytrd::TSIZE;
    let mut work = vec![0.0f64; lwork];
    if crate::sytrd::orm_blocking(n, n - 1, lwork).is_none() {
        // SAFETY: lengths checked; host views only.
        unsafe {
            crate::sytrd::dormtr_ln(
                uplo == Uplo::Upper,
                n,
                n,
                V::new(a),
                n,
                V::new(tau),
                V::new(c),
                n,
                V::new(&mut work),
                lwork,
            );
        }
        return Ok(counters.snapshot());
    }
    let width = geometry::vector_width(client, 8);
    let lda = geometry::lda_for(client, n);
    let lanes = ops::Lanes::new(client, width);
    let mut img = vec![0.0f64; lda * n];
    for j in 0..n {
        img[j * lda..j * lda + n].copy_from_slice(&c[j * n..j * n + n]);
    }
    counters.upload(img.len() * 8);
    let zh = workspace::upload_vec(client, img);
    let mut eng =
        ormtr::DeviceOrm { client, lanes: &lanes, counters: &counters, z: zh.clone(), lda, ncols: n, fast: false };
    let mut dummy = [0.0f64; 1];
    // SAFETY: a/tau/work lengths checked or allocated; the device engine never
    // touches the host `c` view on the blocked path.
    unsafe {
        crate::sytrd::dormtr_ln_with(
            &mut eng,
            uplo == Uplo::Upper,
            n,
            n,
            V::new(a),
            n,
            V::new(tau),
            V::new(&mut dummy),
            n,
            V::new(&mut work),
            lwork,
        );
    }
    let mut out = vec![0.0f64; lda * n];
    workspace::read_range(client, &zh, lda * n, 0, lda * n, &mut out);
    counters.read(out.len() * 8);
    for j in 0..n {
        c[j * n..j * n + n].copy_from_slice(&out[j * lda..j * lda + n]);
    }
    Ok(counters.snapshot())
}
