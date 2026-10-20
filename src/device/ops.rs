//! Launchers: host-validated shapes, all `ArrayArg::from_raw_parts` inside
//! `unsafe` blocks with a `SAFETY:` note, confined to this module.

use std::sync::atomic::{AtomicBool, Ordering};

/// Launch every kernel through the bounds-checked `launch` instead of
/// `launch_unchecked` (the launch-parity test; results must be bit-identical).
static CHECKED_LAUNCH: AtomicBool = AtomicBool::new(false);

/// Row accumulators per unit of the row-tiled GPU variants (K2r, K3r).
pub const ROWS: usize = 8;

/// K1 layout override: 0 = default, 1 = lane-vector rows (K1), 2 = scalar
/// column reads (K1col), 3 = lane-vector column reads (K1colv).
static SYMV_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Force the K1 layout (benchmarks and parity tests; 0 = default, 1 = K1,
/// 2 = K1col, 3 = K1colv). Results must not depend on it.
#[doc(hidden)]
pub fn set_symv_mode(mode: u8) {
    SYMV_MODE.store(mode, Ordering::Relaxed);
}

/// Whether K1/K1c read rows as columns (K1col; see [`set_symv_mode`]).
fn symv_col<R: Runtime>(client: &ComputeClient<R>) -> bool {
    match SYMV_MODE.load(Ordering::Relaxed) {
        1 => false,
        2 | 3 => true,
        _ => super::geometry::has_planes(client),
    }
}

/// Whether K1col reads `M` through the lane-vector view (K1colv).
///
/// Off by default: on a Tesla T4 (f64, width 2) K1colv is **2.2x slower**
/// than the scalar column reads at every trailing size
/// (`examples/device_kernel_bench`, interleaved, min of 3: m=512 0.181 vs
/// 0.092 ms, m=1024 0.447 vs 0.207 ms, m=2047 0.902 vs 0.424 ms), even
/// though it issues half the load instructions. It stays available as a
/// tuning knob (mode 3) and as parity coverage, like K2r/K3r.
fn symv_col_vec<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let _ = client;
    SYMV_MODE.load(Ordering::Relaxed) == 3
}

/// Kernel-variant override: 0 = pick from the device, 1 = simple, 2 = row-tiled.
static VARIANT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(u8::MAX);

/// Kernel variant of K2/K3 (a tuning knob; both are bit-identical).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The measured default (currently the simple kernels).
    Auto,
    Simple,
    RowTiled,
}

/// Force a K2/K3 variant (tests; otherwise `LAPACK_RS_KERNEL=simple|row_tiled`
/// or the device-derived choice).
#[doc(hidden)]
pub fn set_variant(v: Variant) {
    VARIANT.store(
        match v {
            Variant::Auto => 0,
            Variant::Simple => 1,
            Variant::RowTiled => 2,
        },
        Ordering::Relaxed,
    );
}

fn row_tiled<R: Runtime>(client: &ComputeClient<R>, tiled_lanes: usize) -> bool {
    let mut v = VARIANT.load(Ordering::Relaxed);
    if v == u8::MAX {
        v = match std::env::var("LAPACK_RS_KERNEL").as_deref() {
            Ok("simple") => 1,
            Ok("row_tiled") => 2,
            _ => 0,
        };
        let _ = VARIANT.compare_exchange(u8::MAX, v, Ordering::Relaxed, Ordering::Relaxed);
    }
    // Auto = simple: measured on a Tesla T4 (f64, width 2, 32-lane warps),
    // K2r and K3r are 1.1-1.8x slower than K2/K3 at every trailing size
    // (`examples/device_kernel_bench.rs`); the row-tiled kernels stay
    // available as a tuning knob.
    let _ = (client, tiled_lanes);
    v == 2
}

/// Select bounds-checked launches for all subsequent kernel launches.
#[doc(hidden)]
pub fn set_checked_launch(on: bool) {
    CHECKED_LAUNCH.store(on, Ordering::Relaxed);
}

/// `kernel::launch_unchecked::<F, R>(args)`, or `launch` when
/// [`set_checked_launch`] is on. Must be expanded inside an `unsafe` block.
macro_rules! kl {
    ($($k:ident)::+; $($arg:expr),* $(,)?) => {
        if CHECKED_LAUNCH.load(Ordering::Relaxed) {
            $($k)::+::launch::<F, R>($($arg),*)
        } else {
            $($k)::+::launch_unchecked::<F, R>($($arg),*)
        }
    };
}

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::geometry::{launch_1d, launch_2d};
use super::kernels::{symv, syr2k};

/// Per-client constants every launch needs: the lane width `N` and the lane
/// offset vector `[0, 1, …, N-1]`.
pub struct Lanes {
    pub width: usize,
    pub lane_ofs: Handle,
}

impl Lanes {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, width: usize) -> Self {
        let ofs: Vec<u32> = (0..width as u32).collect();
        Lanes { width, lane_ofs: super::workspace::upload(client, &ofs) }
    }
}

/// K1: `y(off..off+m) = alpha · M[off.., off..] · x` (`BETA = 0`), reference
/// `DSYMV` order. `m_buf` is `lda × ncols` (`lda % width == 0`), `x` has `m`
/// entries, `y` has `lda` entries (rows outside the block are garbage).
#[allow(clippy::too_many_arguments)]
pub fn symv<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    m_buf: &Handle,
    lda: usize,
    ncols: usize,
    off: usize,
    m: usize,
    x: &Handle,
    y: &Handle,
    alpha: F,
    upper: bool,
) {
    let w = lanes.width;
    assert!(lda % w == 0 && off + m <= lda && off + m <= ncols && m > 0);
    // Default: K1col on hardware planes, where consecutive column steps of K1
    // jump a whole column and K1col measured 1.7-1.9x faster (Tesla T4, f64,
    // m >= 512); K1 on the CPU runtime.
    if symv_col(client) && symv_col_vec(client) {
        let (count, dim) = launch_1d(client, m, m);
        // SAFETY: as the scalar K1col below; the vector view indexes
        // (r*lda + off + q)/w with the same bound, and `lda % w == 0` keeps
        // every lane-vector inside one column.
        unsafe {
            kl!(symv::symv_col_vec_kernel;
                client,
                count,
                dim,
                w,
                ArrayArg::from_raw_parts(m_buf.clone(), lda * ncols),
                ArrayArg::from_raw_parts(m_buf.clone(), lda * ncols),
                ArrayArg::from_raw_parts(x.clone(), m),
                ArrayArg::from_raw_parts(y.clone(), lda),
                lda as u32,
                off as u32,
                m as u32,
                alpha,
                upper,
                alpha == F::new(1.0));
        }
        return;
    }
    if symv_col(client) {
        let (count, dim) = launch_1d(client, m, m);
        // SAFETY: m_buf holds lda*ncols scalars and reads r*lda + p with
        // r, p < off+m <= min(lda, ncols); x holds m scalars indexed q < m;
        // y holds lda scalars indexed r < lda.
        unsafe {
            kl!(symv::symv_col_kernel;
                client,
                count,
                dim,
                ArrayArg::from_raw_parts(m_buf.clone(), lda * ncols),
                ArrayArg::from_raw_parts(x.clone(), m),
                ArrayArg::from_raw_parts(y.clone(), lda),
                lda as u32,
                off as u32,
                m as u32,
                alpha,
                upper,
                alpha == F::new(1.0));
        }
        return;
    }
    let rv0 = off / w;
    let units = (off + m).div_ceil(w) - rv0;
    let (count, dim) = launch_1d(client, units, m * w);
    // SAFETY: m_buf holds lda*ncols scalars and every column read is
    // (p*lda/w + rv) vectors with p < ncols, rv < lda/w; x holds m scalars
    // indexed q < m; y holds lda scalars indexed rv < lda/w; lane_ofs holds w.
    unsafe {
        kl!(symv::symv_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(m_buf.clone(), lda * ncols),
            ArrayArg::from_raw_parts(x.clone(), m),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            ArrayArg::from_raw_parts(y.clone(), lda),
            (lda / w) as u32,
            off as u32,
            m as u32,
            rv0 as u32,
            units as u32,
            alpha,
            upper,);
    }
}

/// K2: `M[off..off+m, off..off+m] += α(V·Wᵀ + W·Vᵀ)` on both triangles in
/// reference `DSYR2K` order. `vw` packs `nb` V columns then `nb` W columns,
/// each `lda` rows aligned with `M`'s rows.
#[allow(clippy::too_many_arguments)]
pub fn syr2k<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    c_buf: &Handle,
    lda: usize,
    ncols: usize,
    off: usize,
    m: usize,
    vw: &Handle,
    nb: usize,
    alpha: F,
    upper: bool,
) {
    let w = lanes.width;
    assert!(lda % w == 0 && off + m <= lda && off + m <= ncols && m > 0 && nb > 0);
    let rv0 = off / w;
    let units_x = (off + m).div_ceil(w) - rv0;
    let tiles = m.div_ceil(ROWS);
    if row_tiled(client, units_x * tiles * w) {
        let (count, dim, units) = launch_2d(client, units_x, tiles, 2 * nb * w * ROWS);
        // SAFETY: as below; tile columns are clamped into [off, off+m).
        unsafe {
            kl!(syr2k::syr2k_rt_kernel;
                client,
                count,
                dim,
                w,
                ArrayArg::from_raw_parts(c_buf.clone(), lda * ncols),
                ArrayArg::from_raw_parts(vw.clone(), 2 * nb * lda),
                ArrayArg::from_raw_parts(vw.clone(), 2 * nb * lda),
                ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
                lda as u32,
                off as u32,
                m as u32,
                rv0 as u32,
                units_x as u32,
                units as u32,
                alpha,
                nb as u32,
                upper,
                ROWS as u32);
        }
        return;
    }
    let (count, dim, units) = launch_2d(client, units_x, m, 2 * nb * w);
    // SAFETY: c_buf holds lda*ncols scalars, indexed (j*lda/w + rv) with
    // j < off+m <= ncols; vw holds 2*nb*lda scalars (vector view (l*lda/w + rv),
    // scalar view (l*lda + j) with j < lda); lane_ofs holds w u32.
    unsafe {
        kl!(syr2k::syr2k_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(c_buf.clone(), lda * ncols),
            ArrayArg::from_raw_parts(vw.clone(), 2 * nb * lda),
            ArrayArg::from_raw_parts(vw.clone(), 2 * nb * lda),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            lda as u32,
            off as u32,
            m as u32,
            rv0 as u32,
            units_x as u32,
            units as u32,
            alpha,
            nb as u32,
            upper,);
    }
}

use super::kernels::{copy, merge};

fn aligned_units(w: usize, lo: usize, hi: usize) -> (usize, usize) {
    let rv0 = lo / w;
    (rv0, hi.div_ceil(w) - rv0)
}

/// K7: `Z` columns `g0..g0+m` := identity block of order `m` (all `lda` rows).
pub fn fill_identity<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    z: &Handle,
    lda: usize,
    ncols: usize,
    g0: usize,
    m: usize,
) {
    assert!(g0 + m <= ncols && g0 + m <= lda);
    let (count, dim) = launch_1d(client, lda * m, 1);
    // SAFETY: writes (g0+c)*lda + r with c < m, r < lda: inside lda*ncols.
    unsafe {
        kl!(copy::fill_identity_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            lda as u32,
            lda as u32,
            g0 as u32,
            m as u32,);
    }
}

/// K7: `Z[g0.., g0..] := src` (`m × m`).
pub fn scatter_block<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    z: &Handle,
    lda: usize,
    ncols: usize,
    src: &Handle,
    g0: usize,
    m: usize,
) {
    assert!(g0 + m <= ncols && g0 + m <= lda);
    let (count, dim) = launch_1d(client, m * m, 1);
    // SAFETY: src holds m*m scalars; writes (g0+j)*lda + g0+i inside lda*ncols.
    unsafe {
        kl!(copy::scatter_block_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(src.clone(), m * m),
            lda as u32,
            g0 as u32,
            m as u32,);
    }
}

/// K7: `out[q] := Z[row, col0 + q]`, `q < len`.
#[allow(clippy::too_many_arguments)]
pub fn gather_row<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    z: &Handle,
    lda: usize,
    ncols: usize,
    out: &Handle,
    row: usize,
    col0: usize,
    len: usize,
) {
    assert!(row < lda && col0 + len <= ncols);
    let (count, dim) = launch_1d(client, len, 1);
    // SAFETY: reads (col0+q)*lda + row inside lda*ncols; out holds len.
    unsafe {
        kl!(copy::gather_row_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(out.clone(), len),
            lda as u32,
            row as u32,
            col0 as u32,
            len as u32,);
    }
}

/// K7: Givens rotations in list order on rows `[lo, hi)`. `cols` holds
/// `2·nrot` global column indices `(x, y)`, `cs` holds `2·nrot` `(c, s)`.
#[allow(clippy::too_many_arguments)]
pub fn rotations<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &Handle,
    lda: usize,
    ncols: usize,
    cols: &Handle,
    cs: &Handle,
    nrot: usize,
    lo: usize,
    hi: usize,
) {
    let w = lanes.width;
    assert!(lda % w == 0 && hi <= lda && lo < hi && nrot > 0);
    let (rv0, units) = aligned_units(w, lo, hi);
    let (count, dim) = launch_1d(client, units, 6 * nrot * w);
    // SAFETY: column indices < ncols (host-validated by the caller's block
    // bounds), row vectors rv < lda/w; cols/cs hold 2*nrot.
    unsafe {
        kl!(copy::rotations_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(cols.clone(), 2 * nrot),
            ArrayArg::from_raw_parts(cs.clone(), 2 * nrot),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            (lda / w) as u32,
            rv0 as u32,
            units as u32,
            nrot as u32,
            lo as u32,
            hi as u32,);
    }
}

/// K7: swap columns `ca`, `cb` over all rows.
pub fn swap_cols<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &Handle,
    lda: usize,
    ncols: usize,
    ca: usize,
    cb: usize,
) {
    let w = lanes.width;
    assert!(lda % w == 0 && ca < ncols && cb < ncols);
    let (count, dim) = launch_1d(client, lda / w, 2 * w);
    // SAFETY: indices c*lda/w + rv with c < ncols, rv < lda/w.
    unsafe {
        kl!(copy::swap_cols_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            (lda / w) as u32,
            ca as u32,
            cb as u32,);
    }
}

/// Row-chunk scratch geometry for the merge: lane-vectors `rv0..rv0+rvecs`.
#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    pub rv0: usize,
    pub rvecs: usize,
}

/// K3m (staged U), one row range `[lo, hi)` inside `chunk`: output columns
/// `j < n` of the block into `dst` (`chunk.rvecs·w × n`). `perm` (global
/// columns, `nterms` entries) and `s` (`lds × k`, the terms at rows
/// `s_row0..s_row0 + nterms`) drive `j < k`; `perm4`
/// (global columns, `n − k` entries) the gathered `j ≥ k`. `nterms == 0`
/// writes zeros for `j < k` (`DLASET`).
#[allow(clippy::too_many_arguments)]
pub fn merge_rows<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &Handle,
    lda: usize,
    ncols: usize,
    dst: &Handle,
    chunk: Chunk,
    n: usize,
    k: usize,
    perm: &Handle,
    nterms: usize,
    perm4: &Handle,
    s: &Handle,
    lds: usize,
    s_row0: usize,
    lo: usize,
    hi: usize,
) {
    let w = lanes.width;
    assert!(lda % w == 0 && lo < hi && hi <= lda && k <= n);
    assert!(nterms == 0 || s_row0 + nterms <= lds);
    assert!(lo >= chunk.rv0 * w && hi <= (chunk.rv0 + chunk.rvecs) * w);
    let (rv0, units_x) = aligned_units(w, lo, hi);
    let dst_len = chunk.rvecs * w * n;
    if nterms == 0 && k > 0 {
        // Zero columns j < k.
        let (count, dim, units) = launch_2d(client, units_x, k, w);
        // SAFETY: dst index j*rvecs + (rv - chunk.rv0) < rvecs*k <= dst_len/w.
        unsafe {
            kl!(merge::merge_zero_kernel;
                client,
                count,
                dim,
                w,
                ArrayArg::from_raw_parts(dst.clone(), dst_len),
                ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
                (lda / w) as u32,
                rv0 as u32,
                units_x as u32,
                units as u32,
                lo as u32,
                hi as u32,
                chunk.rvecs as u32,
                chunk.rv0 as u32,);
        }
    }
    // Columns j < k with terms (when nterms > 0) and every j >= k (gather);
    // the kernel branches on j only.
    let j0 = if nterms == 0 { k } else { 0 };
    if j0 >= n {
        return;
    }
    let (count, dim, units) = launch_2d(client, units_x, n - j0, w * nterms.max(1));
    // SAFETY: z reads perm[l]*lda/w + rv with perm < ncols; dst writes
    // j*rvecs + rv - chunk.rv0 < rvecs*n; s reads j*lds + s_row0 + l <
    // lds*k (asserted s_row0 + nterms <= lds); perm4
    // reads j - k < n - k.
    unsafe {
        kl!(merge::merge_staged_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(dst.clone(), dst_len),
            ArrayArg::from_raw_parts(perm.clone(), nterms.max(1)),
            ArrayArg::from_raw_parts(perm4.clone(), (n - k).max(1)),
            ArrayArg::from_raw_parts(s.clone(), (lds * k).max(1)),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            (lda / w) as u32,
            rv0 as u32,
            units_x as u32,
            units as u32,
            lo as u32,
            hi as u32,
            k as u32,
            nterms as u32,
            lds.max(1) as u32,
            s_row0 as u32,
            chunk.rvecs as u32,
            chunk.rv0 as u32,
            j0 as u32,);
    }
}

use super::kernels::secular;

/// `DLAMCH` / `dnrm2.f90` constants the K4 kernels take as arguments.
pub struct SecularConsts<F> {
    pub eps: F,
    /// `DLAED6`'s `SMALL1 = BASE**INT(LOG(SAFMIN)/LOG(BASE)/3)`.
    pub small1: F,
    pub tsml: F,
    pub tbig: F,
    pub ssml: F,
    pub sbig: F,
    pub maxn: F,
}

/// K4: `DLAED4` for all `k >= 3` roots: `delta` (`k × k`), `dlam` (`k`) and
/// `info` (`k` u32) are written; `d`, `z` hold `k` entries.
#[allow(clippy::too_many_arguments)]
pub fn secular_roots<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    d: &Handle,
    z: &Handle,
    delta: &Handle,
    dlam: &Handle,
    info: &Handle,
    k: usize,
    rho: F,
    c: &SecularConsts<F>,
) {
    assert!(k >= 3);
    let (count, dim) = launch_1d(client, k, 16 * k);
    // SAFETY: d, z, dlam, info hold k entries indexed < k; delta holds k*k,
    // unit j writes/reads j*k + (0..k).
    unsafe {
        kl!(secular::secular_roots_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(d.clone(), k),
            ArrayArg::from_raw_parts(z.clone(), k),
            ArrayArg::from_raw_parts(delta.clone(), k * k),
            ArrayArg::from_raw_parts(dlam.clone(), k),
            ArrayArg::from_raw_parts(info.clone(), k),
            k as u32,
            rho,
            c.eps,
            c.small1);
    }
}

/// K4: `DLAED3`'s `W` (`k` entries) from `delta`; `neg[i]` = sign bit of the
/// original `W(i)`.
pub fn secular_w<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    delta: &Handle,
    d: &Handle,
    neg: &Handle,
    w: &Handle,
    k: usize,
) {
    let (count, dim) = launch_1d(client, k, 2 * k);
    // SAFETY: delta holds k*k (reads j*k + i, i, j < k); d, neg, w hold k.
    unsafe {
        kl!(secular::secular_w_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(delta.clone(), k * k),
            ArrayArg::from_raw_parts(d.clone(), k),
            ArrayArg::from_raw_parts(neg.clone(), k),
            ArrayArg::from_raw_parts(w.clone(), k),
            k as u32);
    }
}

/// K4: `DLAED3`'s normalised, row-permuted eigenvector columns into `u`
/// (`k × k`); `delta` is overwritten; `indx` holds 0-based rows `< k`.
#[allow(clippy::too_many_arguments)]
pub fn secular_u<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    w: &Handle,
    indx: &Handle,
    delta: &Handle,
    u: &Handle,
    k: usize,
    c: &SecularConsts<F>,
) {
    let (count, dim) = launch_1d(client, k, 4 * k);
    // SAFETY: w, indx hold k; every indx value is < k (asserted against the
    // 1-based DLAED2 permutation in `device::stedc::DeviceDc::secular`, which
    // is the only caller); delta and u hold k*k, unit j touches j*k + (0..k).
    unsafe {
        kl!(secular::secular_u_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(w.clone(), k),
            ArrayArg::from_raw_parts(indx.clone(), k),
            ArrayArg::from_raw_parts(delta.clone(), k * k),
            ArrayArg::from_raw_parts(u.clone(), k * k),
            k as u32,
            c.tsml,
            c.tbig,
            c.ssml,
            c.sbig,
            c.maxn);
    }
}

/// The O(k) vectors of a lean merge (see `merge_lean_kernel`).
pub struct LeanU<'a> {
    /// Secular index per Q2 term, `nterms` entries.
    pub sec: &'a Handle,
    pub zhat: &'a Handle,
    pub dlam: &'a Handle,
    pub orig: &'a Handle,
    pub tau: &'a Handle,
    pub norm: &'a Handle,
}

/// K3m lean: [`merge_rows`] with `U` evaluated from [`LeanU`] per flop.
#[allow(clippy::too_many_arguments)]
pub fn merge_rows_lean<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &Handle,
    lda: usize,
    ncols: usize,
    dst: &Handle,
    chunk: Chunk,
    n: usize,
    k: usize,
    perm: &Handle,
    nterms: usize,
    perm4: &Handle,
    lean: &LeanU<'_>,
    lo: usize,
    hi: usize,
) {
    let w = lanes.width;
    assert!(lda % w == 0 && lo < hi && hi <= lda && k <= n && k > 0);
    assert!(lo >= chunk.rv0 * w && hi <= (chunk.rv0 + chunk.rvecs) * w);
    if nterms == 0 {
        // No terms: the staged launcher's zero/gather path is identical.
        let dummy = super::workspace::upload(client, &[0.0f64]);
        merge_rows::<R, F>(client, lanes, z, lda, ncols, dst, chunk, n, k, perm, 0, perm4, &dummy, 1, 0, lo, hi);
        return;
    }
    let (rv0, units_x) = aligned_units(w, lo, hi);
    let dst_len = chunk.rvecs * w * n;
    let (count, dim, units) = launch_2d(client, units_x, n, w * nterms);
    // SAFETY: as `merge_rows`; sec holds nterms indices < k; zhat, dlam, tau,
    // norm hold k scalars; orig holds k indices < k; perm4 reads j - k < n - k.
    unsafe {
        kl!(merge::merge_lean_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(dst.clone(), dst_len),
            ArrayArg::from_raw_parts(perm.clone(), nterms),
            ArrayArg::from_raw_parts(lean.sec.clone(), nterms),
            ArrayArg::from_raw_parts(perm4.clone(), (n - k).max(1)),
            ArrayArg::from_raw_parts(lean.zhat.clone(), k),
            ArrayArg::from_raw_parts(lean.dlam.clone(), k),
            ArrayArg::from_raw_parts(lean.orig.clone(), k),
            ArrayArg::from_raw_parts(lean.tau.clone(), k),
            ArrayArg::from_raw_parts(lean.norm.clone(), k),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            (lda / w) as u32,
            rv0 as u32,
            units_x as u32,
            units as u32,
            lo as u32,
            hi as u32,
            k as u32,
            nterms as u32,
            chunk.rvecs as u32,
            chunk.rv0 as u32,
            0u32);
    }
}

/// K7: copy `dst` chunk columns `0..n` back into `Z` columns `c0..c0+n`,
/// rows `[lo, hi)`.
#[allow(clippy::too_many_arguments)]
pub fn copy_back<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &Handle,
    lda: usize,
    ncols: usize,
    dst: &Handle,
    chunk: Chunk,
    c0: usize,
    n: usize,
    lo: usize,
    hi: usize,
) {
    let w = lanes.width;
    assert!(c0 + n <= ncols && lo >= chunk.rv0 * w && hi <= (chunk.rv0 + chunk.rvecs) * w);
    let (rv0, units_x) = aligned_units(w, lo, hi);
    let (count, dim, units) = launch_2d(client, units_x, n, w);
    // SAFETY: src index j*rvecs + rv - chunk.rv0 < rvecs*n; Z index
    // (c0+j)*lda/w + rv inside lda*ncols.
    unsafe {
        kl!(copy::copy_cols_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(dst.clone(), chunk.rvecs * w * n),
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            (lda / w) as u32,
            c0 as u32,
            rv0 as u32,
            units_x as u32,
            units as u32,
            lo as u32,
            hi as u32,
            chunk.rvecs as u32,
            chunk.rv0 as u32,);
    }
}

use super::kernels::gemm;

/// Right-`DTRMM` chain variant of a `DLARFB` step.
#[derive(Clone, Copy, Debug)]
pub struct Chain {
    pub above: bool,
    pub asc: bool,
    pub scale: bool,
}

/// K3: `w_out(i,j) = [diag_j·] w_in(i,j) + Σ coef(j,kk)·w_in(i,kk)` over the
/// variant's range (row-major `n × kpad`).
#[allow(clippy::too_many_arguments)]
pub fn trmm_chain<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    w_in: &Handle,
    w_out: &Handle,
    coef: &Handle,
    diag: &Handle,
    n: usize,
    k: usize,
    kpad: usize,
    chain: Chain,
) {
    let w = lanes.width;
    assert!(kpad % w == 0 && k <= kpad && n > 0 && k > 0);
    let units = n * (kpad / w);
    let (count, dim) = launch_1d(client, units, k * w);
    // SAFETY: w_in/w_out hold n*kpad scalars (vector index i*kpad/w + jv,
    // scalar index i*kpad + kk with kk < k <= kpad); coef holds kpad*kpad;
    // diag holds kpad; lane_ofs holds w.
    unsafe {
        kl!(gemm::trmm_chain_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(w_in.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_in.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_out.clone(), n * kpad),
            ArrayArg::from_raw_parts(coef.clone(), kpad * kpad),
            ArrayArg::from_raw_parts(diag.clone(), kpad),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            kpad as u32,
            k as u32,
            units as u32,
            chain.above,
            chain.asc,
            chain.scale,);
    }
}

/// K3: `w_out(i,j) = Σ_{l<cnt} Z[c_row0+l, i]·V_rm[vrow0+l, j] + w_in(i,j)`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_t<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &Handle,
    lda: usize,
    ncols: usize,
    v_rm: &Handle,
    v_rows: usize,
    w_in: &Handle,
    w_out: &Handle,
    n: usize,
    kpad: usize,
    c_row0: usize,
    vrow0: usize,
    cnt: usize,
) {
    let w = lanes.width;
    assert!(n <= ncols && c_row0 + cnt <= lda && vrow0 + cnt <= v_rows && kpad % w == 0);
    let tiled_units = n.div_ceil(ROWS) * (kpad / w);
    if row_tiled(client, tiled_units * w) {
        let (count, dim) = launch_1d(client, tiled_units, cnt * w * ROWS);
        // SAFETY: as below; tile rows are clamped into [0, n).
        unsafe {
            kl!(gemm::gemm_t_rt_kernel;
                client,
                count,
                dim,
                w,
                ArrayArg::from_raw_parts(z.clone(), lda * ncols),
                ArrayArg::from_raw_parts(v_rm.clone(), v_rows * kpad),
                ArrayArg::from_raw_parts(w_in.clone(), n * kpad),
                ArrayArg::from_raw_parts(w_out.clone(), n * kpad),
                lda as u32,
                kpad as u32,
                c_row0 as u32,
                vrow0 as u32,
                cnt as u32,
                n as u32,
                tiled_units as u32,
                ROWS as u32);
        }
        return;
    }
    let units = n * (kpad / w);
    let (count, dim) = launch_1d(client, units, cnt * w);
    // SAFETY: z index i*lda + c_row0 + l < lda*ncols; v_rm holds v_rows*kpad;
    // w buffers hold n*kpad.
    unsafe {
        kl!(gemm::gemm_t_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(v_rm.clone(), v_rows * kpad),
            ArrayArg::from_raw_parts(w_in.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_out.clone(), n * kpad),
            lda as u32,
            kpad as u32,
            c_row0 as u32,
            vrow0 as u32,
            cnt as u32,
            units as u32,);
    }
}

/// K3: `Z[c_row0+r, i] += Σ_{l<k} (−W(i,l))·V[pad + r + l·ldv]`, `r < cnt`,
/// with `pad = c_row0 % width` and `ldv` a multiple of the width (units are
/// sector-aligned rows; see the kernel).
#[allow(clippy::too_many_arguments)]
pub fn gemm_n<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    width: usize,
    z: &Handle,
    lda: usize,
    ncols: usize,
    v: &Handle,
    ldv: usize,
    wbuf: &Handle,
    n: usize,
    kpad: usize,
    k: usize,
    c_row0: usize,
    cnt: usize,
) {
    let pad = c_row0 % width;
    assert!(n <= ncols && c_row0 + cnt <= lda && ldv % width == 0 && pad + cnt <= ldv && c_row0 - pad + ldv <= lda);
    let units = n * ldv;
    let (count, dim) = launch_1d(client, units, k);
    // SAFETY: z index i*lda + c_row0 - pad + r with r < pad + cnt stays inside
    // column i (< lda*ncols); v index r + l*ldv < k*ldv (the upload size);
    // w index i*kpad + l < n*kpad.
    unsafe {
        kl!(
            gemm::gemm_n_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(v.clone(), k * ldv),
            ArrayArg::from_raw_parts(wbuf.clone(), n * kpad),
            lda as u32,
            ldv as u32,
            kpad as u32,
            c_row0 as u32,
            pad as u32,
            cnt as u32,
            k as u32,
            units as u32
        )
    }
}

/// K7: `W(i,j) := Z[c_row0 + j, i]` (row-major `n × kpad`, zero padding).
#[allow(clippy::too_many_arguments)]
pub fn rows_to_w<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    z: &Handle,
    lda: usize,
    ncols: usize,
    wbuf: &Handle,
    n: usize,
    kpad: usize,
    k: usize,
    c_row0: usize,
) {
    assert!(n <= ncols && c_row0 + k <= lda);
    let (count, dim) = launch_1d(client, n * kpad, 1);
    // SAFETY: z index i*lda + c_row0 + j inside lda*ncols; w holds n*kpad.
    unsafe {
        kl!(copy::rows_to_w_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(wbuf.clone(), n * kpad),
            lda as u32,
            kpad as u32,
            k as u32,
            c_row0 as u32,
            n as u32,);
    }
}

/// K7: `Z[c_row0 + j, i] -= W(i,j)`.
#[allow(clippy::too_many_arguments)]
pub fn sub_w_from_rows<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    z: &Handle,
    lda: usize,
    ncols: usize,
    wbuf: &Handle,
    n: usize,
    kpad: usize,
    k: usize,
    c_row0: usize,
) {
    assert!(n <= ncols && c_row0 + k <= lda);
    let (count, dim) = launch_1d(client, n * kpad, 1);
    // SAFETY: as `rows_to_w`.
    unsafe {
        kl!(copy::sub_w_from_rows_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(wbuf.clone(), n * kpad),
            lda as u32,
            kpad as u32,
            k as u32,
            c_row0 as u32,
            n as u32,);
    }
}

use super::kernels::zkernels;

/// A planar complex device buffer: real and imaginary parts with identical
/// layout.
#[derive(Clone, Debug)]
pub struct ZHandle {
    pub re: Handle,
    pub im: Handle,
}

/// K1c: complex `ZHEMV` (`BETA = 0`); layout as [`symv`].
#[allow(clippy::too_many_arguments)]
pub fn hemv<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    m_buf: &ZHandle,
    lda: usize,
    ncols: usize,
    off: usize,
    m: usize,
    x: &ZHandle,
    y: &ZHandle,
    alpha: (F, F),
    upper: bool,
) {
    let w = lanes.width;
    assert!(lda % w == 0 && off + m <= lda && off + m <= ncols && m > 0);
    if symv_col(client) {
        let (count, dim) = launch_1d(client, m, 2 * m);
        // SAFETY: as the K1col branch of `symv`, for each planar part.
        unsafe {
            kl!(zkernels::hemv_col_kernel;
                client,
                count,
                dim,
                ArrayArg::from_raw_parts(m_buf.re.clone(), lda * ncols),
                ArrayArg::from_raw_parts(m_buf.im.clone(), lda * ncols),
                ArrayArg::from_raw_parts(x.re.clone(), m),
                ArrayArg::from_raw_parts(x.im.clone(), m),
                ArrayArg::from_raw_parts(y.re.clone(), lda),
                ArrayArg::from_raw_parts(y.im.clone(), lda),
                lda as u32,
                off as u32,
                m as u32,
                alpha.0,
                alpha.1,
                upper);
        }
        return;
    }
    let rv0 = off / w;
    let units = (off + m).div_ceil(w) - rv0;
    let (count, dim) = launch_1d(client, units, 2 * m * w);
    // SAFETY: as `symv`, for each of the two planar parts.
    unsafe {
        kl!(zkernels::hemv_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(m_buf.re.clone(), lda * ncols),
            ArrayArg::from_raw_parts(m_buf.im.clone(), lda * ncols),
            ArrayArg::from_raw_parts(x.re.clone(), m),
            ArrayArg::from_raw_parts(x.im.clone(), m),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            ArrayArg::from_raw_parts(y.re.clone(), lda),
            ArrayArg::from_raw_parts(y.im.clone(), lda),
            (lda / w) as u32,
            off as u32,
            m as u32,
            rv0 as u32,
            units as u32,
            alpha.0,
            alpha.1,
            upper);
    }
}

/// K2c: complex `ZHER2K('N')` with `BETA = ONE` on both triangles; layout as
/// [`syr2k`].
#[allow(clippy::too_many_arguments)]
pub fn her2k<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    c_buf: &ZHandle,
    lda: usize,
    ncols: usize,
    off: usize,
    m: usize,
    vw: &ZHandle,
    nb: usize,
    alpha: (F, F),
    upper: bool,
) {
    let w = lanes.width;
    assert!(lda % w == 0 && off + m <= lda && off + m <= ncols && m > 0 && nb > 0);
    let rv0 = off / w;
    let units_x = (off + m).div_ceil(w) - rv0;
    let (count, dim, units) = launch_2d(client, units_x, m, 4 * nb * w);
    // SAFETY: as `syr2k`, for each of the two planar parts.
    unsafe {
        kl!(zkernels::her2k_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(c_buf.re.clone(), lda * ncols),
            ArrayArg::from_raw_parts(c_buf.im.clone(), lda * ncols),
            ArrayArg::from_raw_parts(vw.re.clone(), 2 * nb * lda),
            ArrayArg::from_raw_parts(vw.im.clone(), 2 * nb * lda),
            ArrayArg::from_raw_parts(vw.re.clone(), 2 * nb * lda),
            ArrayArg::from_raw_parts(vw.im.clone(), 2 * nb * lda),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            lda as u32,
            off as u32,
            m as u32,
            rv0 as u32,
            units_x as u32,
            units as u32,
            alpha.0,
            alpha.1,
            nb as u32,
            upper);
    }
}

/// K3c: complex right-side `ZTRMM` chain; `coef` is `kpad × kpad` per part,
/// `diag` packs `[re | im | flag]` (`3·kpad`).
#[allow(clippy::too_many_arguments)]
pub fn ztrmm_chain<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    w_in: &ZHandle,
    w_out: &ZHandle,
    coef: &ZHandle,
    diag: &Handle,
    n: usize,
    k: usize,
    kpad: usize,
    above: bool,
    asc: bool,
) {
    let w = lanes.width;
    assert!(kpad % w == 0 && k <= kpad && n > 0 && k > 0);
    let units = n * (kpad / w);
    let (count, dim) = launch_1d(client, units, 2 * k * w);
    // SAFETY: as `trmm_chain`; diag holds 3*kpad scalars.
    unsafe {
        kl!(zkernels::ztrmm_chain_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(w_in.re.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_in.im.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_in.re.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_in.im.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_out.re.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_out.im.clone(), n * kpad),
            ArrayArg::from_raw_parts(coef.re.clone(), kpad * kpad),
            ArrayArg::from_raw_parts(coef.im.clone(), kpad * kpad),
            ArrayArg::from_raw_parts(diag.clone(), 3 * kpad),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            kpad as u32,
            k as u32,
            units as u32,
            above,
            asc);
    }
}

/// K3c: `W_out = α·(C₂ᴴ V₂) + β·W_in` (row-major `n × kpad`).
#[allow(clippy::too_many_arguments)]
pub fn zgemm_c<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &ZHandle,
    lda: usize,
    ncols: usize,
    v_rm: &ZHandle,
    v_rows: usize,
    w_in: &ZHandle,
    w_out: &ZHandle,
    n: usize,
    kpad: usize,
    c_row0: usize,
    cnt: usize,
    alpha: (F, F),
    beta: (F, F),
) {
    let w = lanes.width;
    assert!(n <= ncols && c_row0 + cnt <= lda && cnt <= v_rows && kpad % w == 0);
    let units = n * (kpad / w);
    let (count, dim) = launch_1d(client, units, 2 * cnt * w);
    // SAFETY: as `gemm_t`, for each of the two planar parts.
    unsafe {
        kl!(zkernels::zgemm_c_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(z.re.clone(), lda * ncols),
            ArrayArg::from_raw_parts(z.im.clone(), lda * ncols),
            ArrayArg::from_raw_parts(v_rm.re.clone(), v_rows * kpad),
            ArrayArg::from_raw_parts(v_rm.im.clone(), v_rows * kpad),
            ArrayArg::from_raw_parts(w_in.re.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_in.im.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_out.re.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_out.im.clone(), n * kpad),
            lda as u32,
            kpad as u32,
            c_row0 as u32,
            0u32,
            cnt as u32,
            units as u32,
            alpha.0,
            alpha.1,
            beta.0,
            beta.1);
    }
}

/// K3c: `C₂ += V₂·(α·Wᴴ)`, sector-aligned scalar units as [`gemm_n`].
#[allow(clippy::too_many_arguments)]
pub fn zgemm_n<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    width: usize,
    z: &ZHandle,
    lda: usize,
    ncols: usize,
    v: &ZHandle,
    ldv: usize,
    wbuf: &ZHandle,
    n: usize,
    kpad: usize,
    k: usize,
    c_row0: usize,
    cnt: usize,
    alpha: (F, F),
) {
    let pad = c_row0 % width;
    assert!(n <= ncols && c_row0 + cnt <= lda && ldv % width == 0 && pad + cnt <= ldv && c_row0 - pad + ldv <= lda);
    let units = n * ldv;
    let (count, dim) = launch_1d(client, units, 2 * k);
    // SAFETY: as `gemm_n`, for each of the two planar parts.
    unsafe {
        kl!(zkernels::zgemm_n_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.re.clone(), lda * ncols),
            ArrayArg::from_raw_parts(z.im.clone(), lda * ncols),
            ArrayArg::from_raw_parts(v.re.clone(), k * ldv),
            ArrayArg::from_raw_parts(v.im.clone(), k * ldv),
            ArrayArg::from_raw_parts(wbuf.re.clone(), n * kpad),
            ArrayArg::from_raw_parts(wbuf.im.clone(), n * kpad),
            lda as u32,
            ldv as u32,
            kpad as u32,
            c_row0 as u32,
            pad as u32,
            cnt as u32,
            k as u32,
            units as u32,
            alpha.0,
            alpha.1);
    }
}

/// K7c: `W(i,j) := conj(Z[c_row0 + j, i])`, zero padding.
#[allow(clippy::too_many_arguments)]
pub fn zrows_to_w<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    z: &ZHandle,
    lda: usize,
    ncols: usize,
    wbuf: &ZHandle,
    n: usize,
    kpad: usize,
    k: usize,
    c_row0: usize,
) {
    assert!(n <= ncols && c_row0 + k <= lda);
    let (count, dim) = launch_1d(client, n * kpad, 2);
    // SAFETY: as `rows_to_w`, for each of the two planar parts.
    unsafe {
        kl!(zkernels::zrows_to_w_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.re.clone(), lda * ncols),
            ArrayArg::from_raw_parts(z.im.clone(), lda * ncols),
            ArrayArg::from_raw_parts(wbuf.re.clone(), n * kpad),
            ArrayArg::from_raw_parts(wbuf.im.clone(), n * kpad),
            lda as u32,
            kpad as u32,
            k as u32,
            c_row0 as u32,
            n as u32);
    }
}

/// K7c: `Z[c_row0 + j, i] -= conj(W(i,j))`.
#[allow(clippy::too_many_arguments)]
pub fn zsub_w_from_rows<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    z: &ZHandle,
    lda: usize,
    ncols: usize,
    wbuf: &ZHandle,
    n: usize,
    kpad: usize,
    k: usize,
    c_row0: usize,
) {
    assert!(n <= ncols && c_row0 + k <= lda);
    let (count, dim) = launch_1d(client, n * kpad, 2);
    // SAFETY: as `sub_w_from_rows`, for each of the two planar parts.
    unsafe {
        kl!(zkernels::zsub_w_from_rows_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.re.clone(), lda * ncols),
            ArrayArg::from_raw_parts(z.im.clone(), lda * ncols),
            ArrayArg::from_raw_parts(wbuf.re.clone(), n * kpad),
            ArrayArg::from_raw_parts(wbuf.im.clone(), n * kpad),
            lda as u32,
            kpad as u32,
            k as u32,
            c_row0 as u32,
            n as u32);
    }
}

/// K7: `buf[0..len] := 0`.
pub fn fill_zero<R: Runtime, F: Float + CubeElement>(client: &ComputeClient<R>, buf: &Handle, len: usize) {
    let (count, dim) = launch_1d(client, len, 1);
    // SAFETY: writes indices < len inside a buffer of at least len scalars
    // (checked by the caller's allocation).
    unsafe {
        kl!(zkernels::fill_zero_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(buf.clone(), len),
            len as u32);
    }
}

/// K7: stage an `n × n` device matrix as the `lda × n` two-triangle image
/// (see the kernel). `dst` must hold `lda·n` reals.
#[allow(clippy::too_many_arguments)]
pub fn stage_sym<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    src: &Handle,
    dst: &Handle,
    n: usize,
    lda: usize,
    upper: bool,
    herm_im: bool,
) {
    assert!(lda >= n);
    let (count, dim) = launch_1d(client, lda * n, 1);
    // SAFETY: reads src[(i or j) + (j or i)*n] with i, j < n (< n*n); writes
    // dst[u] with u < lda*n.
    unsafe {
        kl!(copy::stage_sym_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(src.clone(), n * n),
            ArrayArg::from_raw_parts(dst.clone(), lda * n),
            n as u32,
            lda as u32,
            upper,
            herm_im);
    }
}

/// K7: the leading `n × n` block of an `lda × n` matrix as a new `n × n`
/// buffer.
pub fn compact<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    src: &Handle,
    n: usize,
    lda: usize,
) -> Handle {
    assert!(lda >= n);
    let dst = client.empty(n * n * core::mem::size_of::<F>());
    let (count, dim) = launch_1d(client, n * n, 1);
    // SAFETY: reads src[i + j*lda] < lda*n, writes dst[u] < n*n.
    unsafe {
        kl!(copy::compact_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(src.clone(), lda * n),
            ArrayArg::from_raw_parts(dst.clone(), n * n),
            n as u32,
            lda as u32);
    }
    dst
}

use super::kernels::stemr;

/// `DLAMCH`-derived constants the K8 (MRRR) kernels take as arguments.
///
/// Path R fills these from Reference-LAPACK (`EPS = DLAMCH('P')`,
/// `PIVMIN = SAFMIN*MAX(1, EMAX**2)`, `RELTOL = SQRT(EPS)`); Path C fills them
/// from libcint 6.1.3 (`eps = 2*DBL_EPSILON`, `reltol = 2.1e-8`, no `PIVMIN`
/// floor). One kernel body, two oracles — never a literal in the kernel.
pub struct StemrConsts<F> {
    /// `RELTOL` of `DLARRK`, `GAPTOL` of `DLAR1V`.
    pub reltol: F,
    pub pivmin: F,
    pub eps: F,
}

/// K8: batched `DLAEV2`. `abc` holds `3 * nprob` reals `(a, b, c)`, `out`
/// receives `4 * nprob` reals `(RT1, RT2, CS1, SN1)`.
pub fn laev2_batch<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    abc: &Handle,
    out: &Handle,
    nprob: usize,
) {
    let (count, dim) = launch_1d(client, nprob, 32);
    // SAFETY: `abc` holds 3*nprob and unit b reads 3b..3b+3 < 3*nprob;
    // `out` holds 4*nprob and unit b writes 4b..4b+4 < 4*nprob. The kernel
    // bounds-checks `b < nprob`.
    unsafe {
        kl!(stemr::laev2_batch_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(abc.clone(), 3 * nprob),
            ArrayArg::from_raw_parts(out.clone(), 4 * nprob),
            nprob as u32);
    }
}

/// K8: batched `DLARRK`. `nptr` is `nprob + 1` monotone offsets into `d`/`e2`
/// (`dlen` total); `nw` indices are launched per problem and the kernel skips
/// `k >= n_b`. `w`/`werr`/`info` hold `nprob * nw`.
#[allow(clippy::too_many_arguments)]
pub fn larrk_batch<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    d: &Handle,
    e2: &Handle,
    nptr: &Handle,
    gl: &Handle,
    gu: &Handle,
    consts: &Handle,
    itmax: &Handle,
    w: &Handle,
    werr: &Handle,
    info: &Handle,
    nprob: usize,
    nw: usize,
    dlen: usize,
) {
    let (count, dim) = launch_1d(client, nprob * nw, 64 * nw);
    // SAFETY: `nptr` holds nprob+1 entries, is monotone and ends at `dlen`
    // (asserted by the caller in `device::stemr`), so `base + n <= dlen` for
    // every unit and every `d`/`e2` read is < dlen; `gl`, `gu`, `itmax` hold
    // nprob and are indexed by `b < nprob`; `w`, `werr`, `info` hold
    // nprob*nw and are indexed by `b*nw + k` with `k < nw`; `consts` holds 3.
    // The kernel bounds-checks `ABSOLUTE_POS < nprob*nw`.
    unsafe {
        kl!(stemr::larrk_batch_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(d.clone(), dlen),
            ArrayArg::from_raw_parts(e2.clone(), dlen),
            ArrayArg::from_raw_parts(nptr.clone(), nprob + 1),
            ArrayArg::from_raw_parts(gl.clone(), nprob),
            ArrayArg::from_raw_parts(gu.clone(), nprob),
            ArrayArg::from_raw_parts(consts.clone(), 3),
            ArrayArg::from_raw_parts(itmax.clone(), nprob),
            ArrayArg::from_raw_parts(w.clone(), nprob * nw),
            ArrayArg::from_raw_parts(werr.clone(), nprob * nw),
            ArrayArg::from_raw_parts(info.clone(), nprob * nw),
            nprob as u32,
            nw as u32);
    }
}

/// K8: batched `DLANEG`, one problem per unit, twist index `r[b]`.
#[allow(clippy::too_many_arguments)]
pub fn laneg_batch<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    d: &Handle,
    lld: &Handle,
    nptr: &Handle,
    sigma: &Handle,
    r: &Handle,
    negcnt: &Handle,
    nprob: usize,
    dlen: usize,
) {
    let (count, dim) = launch_1d(client, nprob, 8);
    // SAFETY: `nptr` is monotone with nprob+1 entries ending at `dlen`, so
    // every `d`/`lld` read is < dlen; `sigma`, `r`, `negcnt` hold nprob and
    // are indexed by `b < nprob`; `1 <= r[b] <= n_b` is asserted by the
    // caller. The kernel bounds-checks `b < nprob`.
    unsafe {
        kl!(stemr::laneg_batch_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(d.clone(), dlen),
            ArrayArg::from_raw_parts(lld.clone(), dlen),
            ArrayArg::from_raw_parts(nptr.clone(), nprob + 1),
            ArrayArg::from_raw_parts(sigma.clone(), nprob),
            ArrayArg::from_raw_parts(r.clone(), nprob),
            ArrayArg::from_raw_parts(negcnt.clone(), nprob),
            nprob as u32,
            128u32);
    }
}

/// K8: batched `DLAR1V`, one problem per unit. `z` shares `d`'s offsets;
/// `work` is the `4n` `lplus|uminus|s|p` carve at `4 * nptr[b]`.
#[allow(clippy::too_many_arguments)]
pub fn lar1v_batch<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    b: &Lar1vBatch<'_>,
    nprob: usize,
    dlen: usize,
) {
    let (count, dim) = launch_1d(client, nprob, 16);
    // SAFETY: `nptr` is monotone with nprob+1 entries ending at `dlen`, so
    // every `d`/`l`/`ld`/`lld`/`z` access is < dlen and every `work` access is
    // < 4*dlen; `lambda`, `r_in`, `wantnc` hold nprob; `isuppz`/`iout` hold
    // 2*nprob (unit b writes 2b, 2b+1) and `out` holds 4*nprob (unit b writes
    // 4b..4b+4); `consts` holds 3. The kernel bounds-checks `b < nprob`.
    unsafe {
        kl!(stemr::lar1v_batch_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(b.d.clone(), dlen),
            ArrayArg::from_raw_parts(b.l.clone(), dlen),
            ArrayArg::from_raw_parts(b.ld.clone(), dlen),
            ArrayArg::from_raw_parts(b.lld.clone(), dlen),
            ArrayArg::from_raw_parts(b.nptr.clone(), nprob + 1),
            ArrayArg::from_raw_parts(b.lambda.clone(), nprob),
            ArrayArg::from_raw_parts(b.r_in.clone(), nprob),
            ArrayArg::from_raw_parts(b.consts.clone(), 3),
            ArrayArg::from_raw_parts(b.wantnc.clone(), nprob),
            ArrayArg::from_raw_parts(b.z.clone(), dlen),
            ArrayArg::from_raw_parts(b.work.clone(), 4 * dlen),
            ArrayArg::from_raw_parts(b.isuppz.clone(), 2 * nprob),
            ArrayArg::from_raw_parts(b.out.clone(), 4 * nprob),
            ArrayArg::from_raw_parts(b.iout.clone(), 2 * nprob),
            nprob as u32);
    }
}

/// The handles [`lar1v_batch`] launches with; grouped so the argument list
/// stays readable and every length is checked in one place.
pub struct Lar1vBatch<'a> {
    pub d: &'a Handle,
    pub l: &'a Handle,
    pub ld: &'a Handle,
    pub lld: &'a Handle,
    pub nptr: &'a Handle,
    pub lambda: &'a Handle,
    pub r_in: &'a Handle,
    pub consts: &'a Handle,
    pub wantnc: &'a Handle,
    pub z: &'a Handle,
    pub work: &'a Handle,
    pub isuppz: &'a Handle,
    pub out: &'a Handle,
    pub iout: &'a Handle,
}

/// K8: `dlarrk.f:193`'s `ITMAX` on the device, one problem per unit. The
/// integer gate of `tests/device_stemr_cpu.rs`: a backend `log` that differs
/// from glibc's across an integer boundary must fail here, not as a one-ULP
/// eigenvalue.
pub fn larrk_itmax_batch<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    tnorm: &Handle,
    pivmin: &Handle,
    log_two: F,
    out: &Handle,
    nprob: usize,
) {
    let (count, dim) = launch_1d(client, nprob, 4);
    // SAFETY: `tnorm`, `pivmin` and `out` hold nprob entries and unit b
    // touches only index b. The kernel bounds-checks `b < nprob`.
    unsafe {
        kl!(stemr::larrk_itmax_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(tnorm.clone(), nprob),
            ArrayArg::from_raw_parts(pivmin.clone(), nprob),
            log_two,
            ArrayArg::from_raw_parts(out.clone(), nprob),
            nprob as u32);
    }
}

use super::kernels::cint_stemr as cintk;

/// The handles [`cint_stemr_batch`] launches with. Grouped so the argument
/// list stays readable and every length is checked in one place.
pub struct CintStemrBatch<'a> {
    /// Concatenated diagonals, `dlen` reals; **overwritten**.
    pub diag: &'a Handle,
    /// Concatenated off-diagonals, `dlen` reals; **overwritten**.
    pub offd: &'a Handle,
    /// `nprob + 1` monotone offsets into `diag`/`offd`, ending at `dlen`.
    pub nptr: &'a Handle,
    /// `nprob + 1` offsets into `vec` (problem `b` owns `n_b * n_b` reals).
    pub vptr: &'a Handle,
    /// `dlen` reals: the ascending eigenvalues, problem-major.
    pub eig: &'a Handle,
    /// `vlen` reals: `vec[i*n + j]`, libcint's convention.
    pub vec: &'a Handle,
    /// `nprob * (9 * n_max + 1)` reals.
    pub work: &'a Handle,
    /// `nprob * 5 * n_max` u32.
    pub iwork: &'a Handle,
    /// [`cintk::CC_LEN`] reals: libcint's non-dyadic constants, which cannot
    /// be literals in the kernel (`F::new` takes an `f32`). Build it with
    /// [`cint_stemr_consts`].
    pub cc: &'a Handle,
    /// `nprob * SC_LEN` reals.
    pub sc: &'a Handle,
    /// `nprob * ICNT_LEN` u32.
    pub icnt: &'a Handle,
    /// `nprob` u32: libcint's `INFO`, sign dropped.
    pub info: &'a Handle,
}

/// **K9, Path C.** Batched `cint_stemr` (libcint 6.1.3's `_CINTdiagonalize`),
/// one problem per unit.
///
/// `eps` is C's `DBL_EPSILON` and `big` is libcint's `bstres = 1e307`; both
/// are arguments, never literals, so the same kernel body with
/// Reference-LAPACK's constants is [`stemr`]'s.
#[allow(clippy::too_many_arguments)]
pub fn cint_stemr_batch<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    b: &CintStemrBatch<'_>,
    nprob: usize,
    dlen: usize,
    vlen: usize,
    n_max: u32,
    eps: F,
    big: F,
) {
    assert!(n_max as usize <= crate::cint_stemr::MXRYSROOTS, "n_max above libcint's MXRYSROOTS");
    let wper = 9 * n_max as usize + 1;
    let iwper = 5 * n_max as usize;
    let (count, dim) = launch_1d(client, nprob, 64 * n_max as usize);
    // SAFETY: `nptr` holds nprob+1 monotone offsets ending at `dlen` (asserted
    // by the caller), so every `diag`/`offd`/`eig` access is < dlen; `vptr`
    // likewise bounds every `vec` access by `vlen`; `work` and `iwork` are
    // carved per unit at `b * wper` / `b * iwper` with `n_b <= n_max`, and
    // `sc`/`icnt` at `b * SC_LEN` / `b * ICNT_LEN`, so units never overlap;
    // `info` holds nprob. The kernel bounds-checks `b < nprob`.
    unsafe {
        kl!(cintk::cint_stemr_batch_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(b.diag.clone(), dlen),
            ArrayArg::from_raw_parts(b.offd.clone(), dlen),
            ArrayArg::from_raw_parts(b.nptr.clone(), nprob + 1),
            ArrayArg::from_raw_parts(b.vptr.clone(), nprob + 1),
            ArrayArg::from_raw_parts(b.eig.clone(), dlen),
            ArrayArg::from_raw_parts(b.vec.clone(), vlen),
            ArrayArg::from_raw_parts(b.work.clone(), nprob * wper),
            ArrayArg::from_raw_parts(b.iwork.clone(), nprob * iwper),
            ArrayArg::from_raw_parts(b.cc.clone(), cintk::CC_LEN as usize),
            ArrayArg::from_raw_parts(b.sc.clone(), nprob * cintk::SC_LEN as usize),
            ArrayArg::from_raw_parts(b.icnt.clone(), nprob * cintk::ICNT_LEN as usize),
            ArrayArg::from_raw_parts(b.info.clone(), nprob),
            eps,
            big,
            nprob as u32,
            n_max);
    }
}

/// **K9, Path C, parity/debug only.** One problem, `CubeDim::new_1d(1)`.
/// Never a production path: `tests/device_cint_stemr_cpu.rs` uses it to pin
/// the inline callee against the host transcription.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn cint_stemr_one<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    diag: &Handle,
    offd: &Handle,
    eig: &Handle,
    vec: &Handle,
    work: &Handle,
    iwork: &Handle,
    cc: &Handle,
    sc: &Handle,
    icnt: &Handle,
    info: &Handle,
    n: usize,
    eps: F,
    big: F,
) {
    assert!(n >= 1 && n <= crate::cint_stemr::MXRYSROOTS, "n outside 1..=MXRYSROOTS");
    let (wlen, iwlen) = crate::cint_stemr::cint_stemr_work_len(n);
    // SAFETY: every length below is the allocation's own length and matches
    // `cint_stemr_work_len(n)`; the single unit indexes `diag`/`offd`/`eig`
    // by `< n`, `vec` by `< n*n`, `work` by `< 9n+1`, `iwork` by `< 5n`,
    // `cc` by `< CC_LEN`, `sc`/`icnt` by `< SC_LEN`/`< ICNT_LEN`, `info` by 0.
    unsafe {
        kl!(cintk::cint_stemr_kernel;
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(diag.clone(), n),
            ArrayArg::from_raw_parts(offd.clone(), n),
            ArrayArg::from_raw_parts(eig.clone(), n),
            ArrayArg::from_raw_parts(vec.clone(), n * n),
            ArrayArg::from_raw_parts(work.clone(), wlen),
            ArrayArg::from_raw_parts(iwork.clone(), iwlen.max(1)),
            ArrayArg::from_raw_parts(cc.clone(), cintk::CC_LEN as usize),
            ArrayArg::from_raw_parts(sc.clone(), cintk::SC_LEN as usize),
            ArrayArg::from_raw_parts(icnt.clone(), cintk::ICNT_LEN as usize),
            ArrayArg::from_raw_parts(info.clone(), 1),
            eps,
            big,
            n as u32);
    }
}

/// libcint 6.1.3's non-dyadic constants, in the slot order
/// `src/device/kernels/cint_stemr.rs`'s `CC_*` names give.
///
/// They cannot be written as literals inside a `#[cube]` body: `F::new` takes
/// an **`f32`**, so `F::new(0.333)` is the `f32` nearest `0.333` widened back
/// to `f64` — about 1e-8 away from libcint's `double 0.333`, which shows up
/// immediately in `_dlasq4`'s shift. Dyadic values (`0.25`, `0.5`, `64.0`,
/// `100.0`, `1.5`) survive the round trip and are built from `F::from_int` in
/// the kernel instead.
pub fn cint_stemr_consts() -> [f64; cintk::CC_LEN as usize] {
    let mut c = [0.0f64; cintk::CC_LEN as usize];
    c[cintk::CC_THIRD as usize] = 0.333;
    c[cintk::CC_CNST1 as usize] = 0.563;
    c[cintk::CC_HUNDREDTH as usize] = 0.01;
    c[cintk::CC_CNST3 as usize] = 1.05;
    c[cintk::CC_CNST2 as usize] = 1.01;
    c[cintk::CC_RTL as usize] = 2.1e-8;
    c[cintk::CC_MINGAP as usize] = 0.001;
    c[cintk::CC_FAIL as usize] = 1e16;
    c[cintk::CC_BIG as usize] = 1e307;
    c[cintk::CC_EPS as usize] = f64::EPSILON;
    c
}

// ---------------------------------------------------------------------------
// OpenBLAS track (bit-exact to scipy's OpenBLAS 0.3.30 SkylakeX; see
// `crate::openblas`). Never mixed with the Reference kernels above.
// ---------------------------------------------------------------------------

/// K1ob: `y(off..off+m) = DSYMV(UPLO, m, alpha, M[off.., off..], x, beta = 0)`
/// in OpenBLAS `dsymv_{L,U}_SKYLAKEX` order. Shapes as [`symv`].
#[allow(clippy::too_many_arguments)]
pub fn ob_symv<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    m_buf: &Handle,
    lda: usize,
    ncols: usize,
    off: usize,
    m: usize,
    x: &Handle,
    y: &Handle,
    alpha: F,
    upper: bool,
) {
    assert!(off + m <= lda && off + m <= ncols && m > 0);
    let (count, dim) = launch_1d(client, m, m);
    // SAFETY: m_buf holds lda*ncols scalars and reads (off+i)*lda + off + r
    // with i, r < m, off+m <= min(lda, ncols); x holds m scalars indexed < m;
    // y holds lda scalars indexed off+i < lda.
    unsafe {
        kl!(super::kernels::ob_symv::ob_symv_row_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(m_buf.clone(), lda * ncols),
            ArrayArg::from_raw_parts(x.clone(), m),
            ArrayArg::from_raw_parts(y.clone(), lda),
            lda as u32,
            off as u32,
            m as u32,
            alpha,
            upper);
    }
}

/// K2ob: `M[off..off+m, off..off+m] += alpha(V·Wᵀ + W·Vᵀ)` on both triangles
/// in OpenBLAS `dsyr2k_{U,L}N` order (`beta = 1`, one `k` block). `vw` packs
/// `nb` V columns then `nb` W columns, each `lda` rows aligned with `M`.
///
/// With lane-vectors (`lanes.width > 1`) the plain-chain elements run as
/// `(lane-vector, column)` units and the special elements
/// ([`ob_syr2k_special_intervals`], widened to whole lane-vectors) as a
/// scalar element list; with width 1 one scalar unit per column.
#[allow(clippy::too_many_arguments)]
pub fn ob_syr2k<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    c_buf: &Handle,
    lda: usize,
    ncols: usize,
    off: usize,
    m: usize,
    vw: &Handle,
    nb: usize,
    alpha: F,
    upper: bool,
) {
    assert!(off + m <= lda && off + m <= ncols && m > 0 && nb > 0 && nb < 2 * 384);
    let w = lanes.width;
    if w == 1 || lda % w != 0 {
        let (count, dim) = launch_1d(client, m, m * nb);
        // SAFETY: c_buf holds lda*ncols scalars, indexed (g0+j)*lda + g0+i and
        // (g0+i)*lda + g0+j with i, j < m, off+m <= min(lda, ncols); vw holds
        // 2*nb*lda scalars indexed l*lda + g0 + i with l < 2*nb, g0+i < lda.
        unsafe {
            kl!(super::kernels::ob_syr2k::ob_syr2k_col_kernel;
                client,
                count,
                dim,
                ArrayArg::from_raw_parts(c_buf.clone(), lda * ncols),
                ArrayArg::from_raw_parts(vw.clone(), 2 * nb * lda),
                lda as u32,
                off as u32,
                m as u32,
                nb as u32,
                alpha,
                upper);
        }
        return;
    }
    let (ioff, ivals) = ob_syr2k_special_intervals(m, upper);
    let (rv0, units_x) = aligned_units(w, off, off + m);
    let h_ioff = super::workspace::upload(client, &ioff);
    let h_ivals = super::workspace::upload(client, if ivals.is_empty() { &[0u32, 0u32][..] } else { &ivals[..] });
    let units = units_x * m;
    let (count, dim) = launch_1d(client, units, w * nb);
    // SAFETY: c reads/writes (g0+j)*lda/w + rv with rv < lda/w, g0+j < ncols;
    // vw vector view (l*lda/w + rv), scalar view l*lda + g0 + j with l < 2*nb;
    // ioff holds m+1, ivals holds 2*ioff[m] (>= 2); lane_ofs holds w.
    unsafe {
        kl!(super::kernels::ob_syr2k::ob_syr2k_vec_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(c_buf.clone(), lda * ncols),
            ArrayArg::from_raw_parts(vw.clone(), 2 * nb * lda),
            ArrayArg::from_raw_parts(vw.clone(), 2 * nb * lda),
            ArrayArg::from_raw_parts(h_ioff, m + 1),
            ArrayArg::from_raw_parts(h_ivals, ivals.len().max(2)),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            lda as u32,
            off as u32,
            m as u32,
            rv0 as u32,
            units_x as u32,
            units as u32,
            nb as u32,
            alpha,
            upper);
    }
    // Special elements, widened to the lane-vectors the vector pass skipped.
    let mut elems: Vec<u32> = Vec::new();
    let mut rows: Vec<usize> = Vec::new();
    for j in 0..m {
        rows.clear();
        for t in ioff[j] as usize..ioff[j + 1] as usize {
            let (lo, hi) = (ivals[2 * t] as usize + off, ivals[2 * t + 1] as usize + off);
            let (a, b) = ((lo / w) * w, hi.div_ceil(w) * w);
            rows.extend(a.max(off)..b.min(off + m));
        }
        rows.sort_unstable();
        rows.dedup();
        for &r in &rows {
            elems.push((r - off) as u32);
            elems.push(j as u32);
        }
    }
    if elems.is_empty() {
        return;
    }
    let cnt = elems.len() / 2;
    let h_el = super::workspace::upload(client, &elems);
    let (count, dim) = launch_1d(client, cnt, 2 * nb);
    // SAFETY: elems holds 2*cnt block-relative indices < m; c_buf index
    // (g0+j)*lda + g0+i < lda*ncols; vw as above.
    unsafe {
        kl!(super::kernels::ob_syr2k::ob_syr2k_list_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(c_buf.clone(), lda * ncols),
            ArrayArg::from_raw_parts(vw.clone(), 2 * nb * lda),
            ArrayArg::from_raw_parts(h_el, 2 * cnt),
            lda as u32,
            off as u32,
            m as u32,
            nb as u32,
            cnt as u32,
            alpha,
            upper);
    }
}

/// K7rot-ob: apply `nrot` Givens rotations `(cols[2t], cols[2t+1], c, s)` in
/// order to rows `[lo, hi)` in OpenBLAS `drot_k` order.
#[allow(clippy::too_many_arguments)]
pub fn ob_rotations<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &Handle,
    lda: usize,
    ncols: usize,
    cols: &Handle,
    cs: &Handle,
    nrot: usize,
    lo: usize,
    hi: usize,
) {
    let w = lanes.width;
    assert!(lda % w == 0 && hi <= lda && lo < hi && nrot > 0);
    let (rv0, units) = aligned_units(w, lo, hi);
    let (count, dim) = launch_1d(client, units, 6 * nrot * w);
    // SAFETY: column indices < ncols (host-validated by the caller's block
    // bounds), row vectors rv < lda/w; cols/cs hold 2*nrot.
    unsafe {
        kl!(super::kernels::ob_stedc::ob_rotations_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(cols.clone(), 2 * nrot),
            ArrayArg::from_raw_parts(cs.clone(), 2 * nrot),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            (lda / w) as u32,
            rv0 as u32,
            units as u32,
            nrot as u32,
            lo as u32,
            hi as u32);
    }
}

/// K3m-ob: rows `[lo, hi)` of the merge block rows `[blk_r0, blk_r0+blk_m)`,
/// output columns `0..n`: `j < k` is row `r` of `DGEMM('N','N', blk_m, k,
/// nterms, 1, Q2, S, 0)` in OpenBLAS order (`Q2(r, l) = Z[r, perm[l]]`,
/// `S(l, j) = s[j·lds + s_row0 + l]`), `j >= k` gathers `Z[r, perm4[j-k]]`.
/// `dst` holds `dst_rows` rows per column from global row `dst_r0`
/// (lane-vector aligned when `width > 1`). Rows on the plain fused chain
/// run as lane-vectors; the few rows in 4/2/1 row tiles of a later `GEMM_P`
/// block, or in the small kernel's dot-style tail, run as scalar units.
#[allow(clippy::too_many_arguments)]
pub fn ob_merge_rows<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &Handle,
    lda: usize,
    ncols: usize,
    dst: &Handle,
    dst_rows: usize,
    dst_r0: usize,
    n: usize,
    k: usize,
    perm: &Handle,
    nterms: usize,
    perm4: &Handle,
    s: &Handle,
    lds: usize,
    s_row0: usize,
    lo: usize,
    hi: usize,
    blk_r0: usize,
    blk_m: usize,
) {
    assert!(lo < hi && hi <= lda && k <= n && lo >= dst_r0 && hi <= dst_r0 + dst_rows);
    assert!(lo >= blk_r0 && hi <= blk_r0 + blk_m);
    assert!(nterms == 0 || s_row0 + nterms <= lds);
    let small = (blk_m as f64) * (k as f64) * (nterms as f64) <= 1.0e6;
    // Rows (relative to the block) whose chain is not the plain fused one.
    let special = |i: usize| -> bool {
        if small {
            let m8 = blk_m & !7;
            let dot_from = if blk_m - m8 > 4 || nterms < 16 { blk_m } else { m8 };
            i >= dot_from
        } else {
            let rule = |rem: usize| {
                if rem >= 384 {
                    192
                } else if rem > 192 {
                    (rem / 2).div_ceil(16) * 16
                } else {
                    rem
                }
            };
            let first = rule(blk_m);
            if i < first {
                return false;
            }
            let (mut is, mut mi) = (first, rule(blk_m - first));
            while i >= is + mi {
                is += mi;
                mi = rule(blk_m - is);
            }
            let full16 = (mi / 16) * 16;
            i - is >= full16 + ((mi - full16) & 8)
        }
    };
    let w = lanes.width;
    let (rv0, units_x) = aligned_units(w, lo, hi);
    let mut vplain = vec![1u32; units_x];
    let mut scalar_rows = Vec::new();
    for (q, vp) in vplain.iter_mut().enumerate() {
        let (r0, r1) = (((rv0 + q) * w).max(lo), ((rv0 + q + 1) * w).min(hi));
        if (r0..r1).any(|r| k > 0 && special(r - blk_r0)) {
            *vp = 0;
            scalar_rows.extend((r0..r1).map(|r| r as u32));
        }
    }
    if w > 1 && dst_r0 % w == 0 && dst_rows % w == 0 && vplain.iter().any(|&v| v != 0) {
        let h_vplain = super::workspace::upload(client, &vplain);
        let (count, dim) = launch_1d(client, units_x * n, w * nterms.max(1));
        // SAFETY: z reads perm*lda/w + rv with perm < ncols, rv < lda/w; dst
        // writes j*dst_rvecs + rv - dst_rv0 < dst_rvecs*n (rv inside the dst
        // rows, asserted); s reads j*lds + s_row0 + l < lds*k; perm4 reads
        // j - k < n - k; vplain holds units_x; lane_ofs holds w.
        unsafe {
            kl!(super::kernels::ob_stedc::ob_merge_vec_kernel;
                client,
                count,
                dim,
                w,
                ArrayArg::from_raw_parts(z.clone(), lda * ncols),
                ArrayArg::from_raw_parts(dst.clone(), dst_rows * n),
                ArrayArg::from_raw_parts(perm.clone(), nterms.max(1)),
                ArrayArg::from_raw_parts(perm4.clone(), (n - k).max(1)),
                ArrayArg::from_raw_parts(s.clone(), (lds * k).max(1)),
                ArrayArg::from_raw_parts(h_vplain, units_x),
                ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
                (lda / w) as u32,
                rv0 as u32,
                units_x as u32,
                (units_x * n) as u32,
                lo as u32,
                hi as u32,
                k as u32,
                nterms as u32,
                lds as u32,
                s_row0 as u32,
                (dst_rows / w) as u32,
                (dst_r0 / w) as u32,
                small as u32);
        }
    } else {
        scalar_rows = (lo..hi).map(|r| r as u32).collect();
    }
    if scalar_rows.is_empty() {
        return;
    }
    let nrows = scalar_rows.len();
    let h_rows = super::workspace::upload(client, &scalar_rows);
    let (count, dim) = launch_1d(client, nrows * n, nterms.max(1));
    // SAFETY: z reads perm[l]*lda + r and perm4[j-k]*lda + r with perm < ncols
    // and r < hi <= lda; dst writes j*dst_rows + r - dst_r0 < dst_rows*n; s
    // reads j*lds + s_row0 + l < lds*k; perm4 reads j - k < n - k; rows holds
    // nrows.
    unsafe {
        kl!(super::kernels::ob_stedc::ob_merge_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(dst.clone(), dst_rows * n),
            ArrayArg::from_raw_parts(perm.clone(), nterms.max(1)),
            ArrayArg::from_raw_parts(perm4.clone(), (n - k).max(1)),
            ArrayArg::from_raw_parts(s.clone(), (lds * k).max(1)),
            ArrayArg::from_raw_parts(h_rows, nrows),
            lda as u32,
            nrows as u32,
            (nrows * n) as u32,
            blk_r0 as u32,
            blk_m as u32,
            k as u32,
            nterms as u32,
            lds as u32,
            s_row0 as u32,
            dst_rows as u32,
            dst_r0 as u32,
            0u32,
            small as u32);
    }
}

/// Kob-trmm: `W_out = DTRMM('R', UPLO, TRANS, DIAG, n, k, ONE, A, k, W_in)` in
/// OpenBLAS order (`a` host column-major `k × k`, `W` row-major `n × kpad`).
/// The packed triangle and the row/column tile tables of `trmm_R.c` are built
/// here; the kernel only walks its windows.
#[allow(clippy::too_many_arguments)]
pub fn ob_trmm_right<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    w_in: &Handle,
    w_out: &Handle,
    a: &[f64],
    n: usize,
    k: usize,
    kpad: usize,
    upper: bool,
    trans: bool,
    nounit: bool,
) {
    assert!(n > 0 && k > 0 && k <= kpad && k <= 384 && a.len() >= k * k);
    const GEMM_P: usize = 192;
    let rt = upper == trans;
    // Packed triangle: RT packs the lower triangle of op(A), RN the upper.
    let opa = |r: usize, c: usize| if trans { a[c + r * k] } else { a[r + c * k] };
    let mut tri = vec![0.0f64; k * k];
    for j in 0..k {
        for l in 0..k {
            let inside = if rt { j < l } else { j > l };
            tri[l + j * k] = if l == j {
                if nounit { opa(l, l) } else { 1.0 }
            } else if inside {
                opa(l, j)
            } else {
                0.0
            };
        }
    }
    let row_tile = |r: usize, m: usize| -> u32 {
        let full = (m / 16) * 16;
        if r < full {
            return 16;
        }
        let (rem, t) = (m - full, r - full);
        let mut start = 0;
        for d in [8usize, 4, 2, 1] {
            if rem & d != 0 {
                if t < start + d {
                    return d as u32;
                }
                start += d;
            }
        }
        unreachable!("row tile")
    };
    let first = n.min(GEMM_P);
    let rows: Vec<u32> = (0..n)
        .map(|i| {
            if i < first {
                row_tile(i, first) | 256
            } else {
                let is = (i / GEMM_P) * GEMM_P;
                row_tile(i - is, (n - is).min(GEMM_P))
            }
        })
        .collect();
    // (start, size) of column tiles inside `cnt` columns from `base`.
    let tiles = |base: usize, cnt: usize, out: &mut [u32]| {
        let (mut j, mut nc) = (0usize, cnt);
        for (d, gt) in [(12usize, 11usize), (10, 9), (8, 7), (6, 5), (4, 3), (2, 1)] {
            while nc > gt {
                for q in j..j + d {
                    out[2 * (base + q)] = (base + j) as u32;
                    out[2 * (base + q) + 1] = d as u32;
                }
                j += d;
                nc -= d;
            }
        }
        if nc > 0 {
            out[2 * (base + j)] = (base + j) as u32;
            out[2 * (base + j) + 1] = 1;
        }
    };
    let mut cols = vec![0u32; 4 * k];
    let mut jjs = 0;
    while jjs < k {
        let rem = k - jjs;
        let mj = if rem >= 6 {
            6
        } else if rem > 2 {
            2
        } else {
            rem
        };
        tiles(jjs, mj, &mut cols[..2 * k]);
        jjs += mj;
    }
    tiles(0, k, &mut cols[2 * k..]);
    let h_tri = super::workspace::upload(client, &tri);
    let h_rows = super::workspace::upload(client, &rows);
    let h_cols = super::workspace::upload(client, &cols);
    let (count, dim) = launch_1d(client, n * k, k);
    // SAFETY: w buffers hold n*kpad scalars indexed i*kpad + l (l < k); tri
    // holds k*k indexed j*k + l; rows holds n; cols holds 4k indexed < 4k.
    unsafe {
        kl!(super::kernels::ob_ormtr::ob_trmm_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(w_in.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_out.clone(), n * kpad),
            ArrayArg::from_raw_parts(h_tri, k * k),
            ArrayArg::from_raw_parts(h_rows, n),
            ArrayArg::from_raw_parts(h_cols, 4 * k),
            n as u32,
            k as u32,
            kpad as u32,
            rt);
    }
}

/// Rows (0-based inside `t` rows) of the blocked `level3.c` driver whose row
/// tile is 4/2/1 inside a later `GEMM_P` row block: `[lo, hi)` pairs.
fn gemm_tail_rows(t: usize) -> Vec<u32> {
    let rule = |rem: usize| {
        if rem >= 384 {
            192
        } else if rem > 192 {
            (rem / 2).div_ceil(16) * 16
        } else {
            rem
        }
    };
    let mut out = Vec::new();
    let mut is = rule(t);
    while is < t {
        let mi = rule(t - is);
        let full = (mi / 16) * 16;
        let lo = is + full + ((mi - full) & 8);
        if lo < is + mi {
            out.push(lo as u32);
            out.push((is + mi) as u32);
        }
        is += mi;
    }
    out
}

/// `12·q12(n)`: columns below it sit in 12-wide column tiles.
fn tile12_limit(n: usize) -> usize {
    if n > 11 { 12 * ((n - 12) / 12 + 1) } else { 0 }
}

/// Kob-gemm-tn: `W_out = W_in + C₂ᵀ·V₂` (`DGEMM('T','N', n, k, t, ONE, C₂, V₂,
/// ONE, W)`) in OpenBLAS order; `C₂(l, i) = Z[c_row0 + l, i]`, `V₂(l, j) =
/// v_rm[(vrow0 + l)·kpad + j]`. Lane-vectors over the columns of `W` for the
/// blocked driver; the small `tn` path and the units whose lanes mix modes
/// run as scalar elements.
#[allow(clippy::too_many_arguments)]
pub fn ob_gemm_tn<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &Handle,
    lda: usize,
    ncols: usize,
    v_rm: &Handle,
    v_rows: usize,
    w_in: &Handle,
    w_out: &Handle,
    n: usize,
    k: usize,
    kpad: usize,
    c_row0: usize,
    vrow0: usize,
    t: usize,
) {
    assert!(n <= ncols && c_row0 + t <= lda && vrow0 + t <= v_rows && k <= kpad && n > 0 && k > 0 && t > 0);
    // dgemm_small_matrix_permit for TN: MNK <= 1e6, M·N <= 1200, K >= 32.
    let small = (n as f64) * (k as f64) * (t as f64) <= 1.0e6 && n * k <= 1200 && t >= 32;
    let w = lanes.width;
    let mut elems: Vec<u32> = Vec::new();
    if small || w == 1 || kpad % w != 0 {
        for i in 0..n {
            for j in 0..k {
                elems.push(i as u32);
                elems.push(j as u32);
            }
        }
    } else {
        let kv = kpad / w;
        let (count, dim) = launch_1d(client, n * kv, t * w);
        // SAFETY: z reads i*lda + c_row0 + l < lda*ncols; v_rm vector view reads
        // (vrow0 + l)*kv + jv < v_rows*kv; w buffers hold n*kv vectors.
        unsafe {
            kl!(super::kernels::ob_ormtr::ob_gemm_tn_vec_kernel;
                client,
                count,
                dim,
                w,
                ArrayArg::from_raw_parts(z.clone(), lda * ncols),
                ArrayArg::from_raw_parts(v_rm.clone(), v_rows * kpad),
                ArrayArg::from_raw_parts(w_in.clone(), n * kpad),
                ArrayArg::from_raw_parts(w_out.clone(), n * kpad),
                lda as u32,
                kpad as u32,
                c_row0 as u32,
                vrow0 as u32,
                t as u32,
                n as u32,
                k as u32);
        }
        // Mixed units: a tail row whose lane-vector straddles the 12-tile limit.
        let thr = tile12_limit(k);
        if thr % w != 0 && thr < k {
            let jf = (thr / w) * w;
            let tails = gemm_tail_rows(n);
            for pair in tails.chunks(2) {
                for i in pair[0] as usize..pair[1] as usize {
                    for j in jf..(jf + w).min(k) {
                        elems.push(i as u32);
                        elems.push(j as u32);
                    }
                }
            }
        }
    }
    if elems.is_empty() {
        return;
    }
    let cnt = elems.len() / 2;
    let h_el = super::workspace::upload(client, &elems);
    let (count, dim) = launch_1d(client, cnt, t);
    // SAFETY: elems hold (i < n, j < k); z reads i*lda + c_row0 + l < lda*ncols;
    // v_rm reads (vrow0 + l)*kpad + j < v_rows*kpad; w buffers hold n*kpad.
    unsafe {
        kl!(super::kernels::ob_ormtr::ob_gemm_tn_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(v_rm.clone(), v_rows * kpad),
            ArrayArg::from_raw_parts(w_in.clone(), n * kpad),
            ArrayArg::from_raw_parts(w_out.clone(), n * kpad),
            ArrayArg::from_raw_parts(h_el, 2 * cnt),
            lda as u32,
            kpad as u32,
            c_row0 as u32,
            vrow0 as u32,
            t as u32,
            n as u32,
            k as u32,
            small as u32,
            cnt as u32);
    }
}

/// Kob-gemm-nt: `C₂ -= V₂·Wᵀ` in place (`DGEMM('N','T', t, n, k, −ONE, V₂, W,
/// ONE, C₂)`) in OpenBLAS order; `C₂(r, i) = Z[c_row0 + r, i]`, `V₂(r, l) =
/// v_rm[(vrow0 + r)·kpad + l]` (and, with lane-vectors, the same values in the
/// column-major `v_cm` with `ldv` rows from global row `c_row0 − pad`),
/// `W(i, l) = w[i·kpad + l]`. Lane-vectors over rows for the plain chain;
/// the 4/2/1 row tiles of later `GEMM_P` blocks in columns inside 12-tiles
/// run as scalar elements.
#[allow(clippy::too_many_arguments)]
pub fn ob_gemm_nt<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &Handle,
    lda: usize,
    ncols: usize,
    v_rm: &Handle,
    v_rows: usize,
    v_cm: Option<(&Handle, usize)>,
    w: &Handle,
    n: usize,
    k: usize,
    kpad: usize,
    c_row0: usize,
    vrow0: usize,
    t: usize,
) {
    assert!(n <= ncols && c_row0 + t <= lda && vrow0 + t <= v_rows && k <= kpad && k <= 384 && n > 0 && t > 0);
    let small = (t as f64) * (n as f64) * (k as f64) <= 1.0e6;
    let width = lanes.width;
    let mut elems: Vec<u32> = Vec::new();
    match v_cm {
        Some((h_vcm, ldv)) if width > 1 && lda % width == 0 && vrow0 == 0 => {
            let pad = c_row0 % width;
            assert!(ldv % width == 0 && pad + t <= ldv && c_row0 - pad + ldv <= lda);
            let sp = if small { Vec::new() } else { gemm_tail_rows(t) };
            let thr = if small { 0 } else { tile12_limit(n) };
            let (rv0, units_x) = aligned_units(width, c_row0, c_row0 + t);
            let h_sp = super::workspace::upload(client, if sp.is_empty() { &[0u32, 0u32][..] } else { &sp[..] });
            let units = units_x * n;
            let (count, dim) = launch_1d(client, units, k * width);
            // SAFETY: z vector view i*lda/w + rv < lda*ncols/w; v_cm vector view
            // l*ldv/w + rv - vbase with rv - vbase < ldv/w (asserted pad + t <=
            // ldv); w reads i*kpad + l < n*kpad; sp holds 2*nsp (>= 2).
            unsafe {
                kl!(super::kernels::ob_ormtr::ob_gemm_nt_vec_kernel;
                    client,
                    count,
                    dim,
                    width,
                    ArrayArg::from_raw_parts(z.clone(), lda * ncols),
                    ArrayArg::from_raw_parts(h_vcm.clone(), k * ldv),
                    ArrayArg::from_raw_parts(w.clone(), n * kpad),
                    ArrayArg::from_raw_parts(h_sp, sp.len().max(2)),
                    ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), width),
                    (lda / width) as u32,
                    (ldv / width) as u32,
                    ((c_row0 - pad) / width) as u32,
                    rv0 as u32,
                    units_x as u32,
                    units as u32,
                    c_row0 as u32,
                    t as u32,
                    kpad as u32,
                    k as u32,
                    thr as u32,
                    (sp.len() / 2) as u32);
            }
            // Rows of the skipped lane-vectors, columns below the 12-tile limit.
            let mut rows: Vec<usize> = Vec::new();
            for pair in sp.chunks(2) {
                let (lo, hi) = (pair[0] as usize + c_row0, pair[1] as usize + c_row0);
                let (a, b) = ((lo / width) * width, hi.div_ceil(width) * width);
                rows.extend(a.max(c_row0)..b.min(c_row0 + t));
            }
            rows.sort_unstable();
            rows.dedup();
            for i in 0..thr.min(n) {
                for &r in &rows {
                    elems.push((r - c_row0) as u32);
                    elems.push(i as u32);
                }
            }
        }
        _ => {
            for i in 0..n {
                for r in 0..t {
                    elems.push(r as u32);
                    elems.push(i as u32);
                }
            }
        }
    }
    if elems.is_empty() {
        return;
    }
    let cnt = elems.len() / 2;
    let h_el = super::workspace::upload(client, &elems);
    let (count, dim) = launch_1d(client, cnt, k);
    // SAFETY: elems hold (r < t, i < n); z writes i*lda + c_row0 + r < lda*ncols;
    // v_rm reads (vrow0 + r)*kpad + l < v_rows*kpad; w reads i*kpad + l.
    unsafe {
        kl!(super::kernels::ob_ormtr::ob_gemm_nt_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.clone(), lda * ncols),
            ArrayArg::from_raw_parts(v_rm.clone(), v_rows * kpad),
            ArrayArg::from_raw_parts(w.clone(), n * kpad),
            ArrayArg::from_raw_parts(h_el, 2 * cnt),
            lda as u32,
            kpad as u32,
            c_row0 as u32,
            vrow0 as u32,
            t as u32,
            n as u32,
            k as u32,
            small as u32,
            cnt as u32);
    }
}

/// K2ob element classes. For the `m × m` block of an OpenBLAS `dsyr2k_{U,L}N`
/// call (one `k` block), the triangle element `(p, q)` is *special* when its
/// micro-kernel accumulation is not the plain fused chain: a diagonal
/// `32 × 32` scratch piece, or a two/four-lane element (row tile 4/2/1 ×
/// column tile 12). Pure function of the block rules; see
/// [`crate::device::kernels::ob_syr2k`].
pub fn ob_syr2k_special_elem(m: usize, upper: bool, p: usize, q: usize) -> bool {
    const P: usize = 192;
    const MN: usize = 32;
    let prule = |rem: usize| {
        if rem >= 2 * P {
            P
        } else if rem > P {
            (rem / 2).div_ceil(MN) * MN
        } else {
            rem
        }
    };
    let q12 = |n: usize| if n > 11 { (n - 12) / 12 + 1 } else { 0 };
    let tail = |r: usize, mm: usize| {
        let full = (mm / 16) * 16;
        r >= full + ((mm - full) & 8)
    };
    let (mut s, mut e) = (0, prule(m));
    while e <= p {
        s = e;
        e = s + prule(m - s);
    }
    let mb = e - s;
    if upper {
        if q >= e {
            if s == 0 {
                let jjs = e + ((q - e) / MN) * MN;
                let cn = (m - jjs).min(MN);
                q - jjs < 12 * q12(cn) && tail(p, e)
            } else {
                q - e < 12 * q12(m - e) && tail(p - s, mb)
            }
        } else {
            let lp = ((q - s) / MN) * MN;
            let nn = (mb - lp).min(MN);
            p - s >= lp || (q - s - lp < 12 * q12(nn) && tail(p - s, lp))
        }
    } else if q < s {
        q < 12 * q12(s) && tail(p - s, mb)
    } else {
        let lp = ((q - s) / MN) * MN;
        let nn = (mb - lp).min(MN);
        p < s + lp + nn || (q - s - lp < 12 * q12(nn) && tail(p - s - lp - nn, mb - lp - nn))
    }
}

/// For every block column `j`, the block-row intervals `[lo, hi)` whose
/// element `(i, j)` (either triangle, mirrored to the triangle element) is
/// K2ob-special, in closed form. `offsets[j]..offsets[j+1]` index pairs in
/// `ivals`. Equal to scanning [`ob_syr2k_special_elem`] (gated by
/// `tests/device_openblas_cpu.rs`).
pub fn ob_syr2k_special_intervals(m: usize, upper: bool) -> (Vec<u32>, Vec<u32>) {
    const P: usize = 192;
    const MN: usize = 32;
    let prule = |rem: usize| {
        if rem >= 2 * P {
            P
        } else if rem > P {
            (rem / 2).div_ceil(MN) * MN
        } else {
            rem
        }
    };
    let q12 = |n: usize| if n > 11 { (n - 12) / 12 + 1 } else { 0 };
    let tt = |mm: usize| {
        let full = (mm / 16) * 16;
        full + ((mm - full) & 8)
    };
    let tail = |r: usize, mm: usize| r >= tt(mm);
    let mut blocks = Vec::new();
    let mut s = 0;
    while s < m {
        let e = s + prule(m - s);
        blocks.push((s, e));
        s = e;
    }
    let block_of = |p: usize| *blocks.iter().find(|&&(s, e)| p >= s && p < e).expect("row block");
    let mut offsets = Vec::with_capacity(m + 1);
    let mut ivals = Vec::new();
    for j in 0..m {
        offsets.push((ivals.len() / 2) as u32);
        let mut push = |lo: usize, hi: usize| {
            if lo < hi {
                ivals.push(lo as u32);
                ivals.push(hi as u32);
            }
        };
        if !upper {
            // Own rows i >= j: p = i, q = j.
            for &(sb, eb) in &blocks {
                if eb <= j {
                    continue;
                }
                let mb = eb - sb;
                if j < sb {
                    if j < 12 * q12(sb) {
                        push(sb + tt(mb), eb);
                    }
                } else {
                    let lp = ((j - sb) / MN) * MN;
                    let nn = (mb - lp).min(MN);
                    push(j, sb + lp + nn);
                    if j - sb - lp < 12 * q12(nn) {
                        push(sb + lp + nn + tt(mb - lp - nn), eb);
                    }
                }
            }
            // Mirror rows i < j: p = j, q = i.
            let (s, e) = block_of(j);
            let mb = e - s;
            if tail(j - s, mb) {
                push(0, s.min(12 * q12(s)));
            }
            let lpj = ((j - s) / MN) * MN;
            push(s + lpj, j);
            let mut lp = 0;
            while lp < lpj {
                let nn = (mb - lp).min(MN);
                if tail(j - s - lp - nn, mb - lp - nn) {
                    push(s + lp, s + lp + nn.min(12 * q12(nn)));
                }
                lp += MN;
            }
        } else {
            // Own rows i <= j: p = i, q = j.
            for &(sb, eb) in &blocks {
                if sb > j {
                    break;
                }
                let mb = eb - sb;
                if j >= eb {
                    if sb == 0 {
                        let jjs = eb + ((j - eb) / MN) * MN;
                        let cn = (m - jjs).min(MN);
                        if j - jjs < 12 * q12(cn) {
                            push(tt(eb), eb);
                        }
                    } else if j - eb < 12 * q12(m - eb) {
                        push(sb + tt(mb), eb);
                    }
                } else {
                    let lp = ((j - sb) / MN) * MN;
                    let nn = (mb - lp).min(MN);
                    if j - sb - lp < 12 * q12(nn) {
                        push(sb + tt(lp), sb + lp);
                    }
                    push(sb + lp, j + 1);
                }
            }
            // Mirror rows i > j: p = j, q = i.
            let (s, e) = block_of(j);
            let mb = e - s;
            if s == 0 {
                if tail(j, e) {
                    let mut jjs = e;
                    while jjs < m {
                        let cn = (m - jjs).min(MN);
                        push(jjs, jjs + cn.min(12 * q12(cn)));
                        jjs += MN;
                    }
                }
            } else if tail(j - s, mb) {
                push(e, e + (m - e).min(12 * q12(m - e)));
            }
            let lpj = ((j - s) / MN) * MN;
            let nnj = (mb - lpj).min(MN);
            push(j + 1, s + lpj + nnj);
            let mut lp = lpj + MN;
            while lp < mb {
                let nn = (mb - lp).min(MN);
                if tail(j - s, lp) {
                    push(s + lp, s + lp + nn.min(12 * q12(nn)));
                }
                lp += MN;
            }
        }
    }
    offsets.push((ivals.len() / 2) as u32);
    (offsets, ivals)
}

/// **OpenBLAS track.** The legacy `ZLARF('L', m, n, V, 1, TAU, C, LDC, WORK)`
/// of OpenBLAS 0.3.30's `ZUNM2L`/`ZUNM2R` on the device matrix `z` (`C(1,1)`
/// at global row `crow`, columns `0..n`): `LASTV` scanned on the host `v`,
/// `ILAZLC` on the device into a one-element buffer that `ZGEMV('C')` (into a
/// device `WORK`) and `ZGERC` (in place) read, so consecutive reflectors never
/// synchronise with the host. `neg_tau` is `−TAU` (the `ZGERC` alpha, complex
/// negation done by the caller).
#[allow(clippy::too_many_arguments)]
pub fn ob_zlarf_left<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    z: &ZHandle,
    lda: usize,
    ncols: usize,
    crow: usize,
    m: usize,
    n: usize,
    v_re: &[f64],
    v_im: &[f64],
    tau: (f64, f64),
    neg_tau: (F, F),
) {
    use super::kernels::ob_zormtr;
    if tau.0 == 0.0 && tau.1 == 0.0 {
        return;
    }
    let mut lastv = m;
    while lastv > 0 && v_re[lastv - 1] == 0.0 && v_im[lastv - 1] == 0.0 {
        lastv -= 1;
    }
    if lastv == 0 || n == 0 {
        return;
    }
    assert!(crow + m <= lda && n <= ncols && v_re.len() >= m && v_im.len() >= m);
    let lastc = n;
    let lastc_buf = client.empty(4);
    // SAFETY: reads c*lda + crow + r with c < n <= ncols, crow + r < lda;
    // out holds one u32.
    unsafe {
        kl!(ob_zormtr::ob_zlastc_kernel;
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(z.re.clone(), lda * ncols),
            ArrayArg::from_raw_parts(z.im.clone(), lda * ncols),
            ArrayArg::from_raw_parts(lastc_buf.clone(), 1),
            lda as u32,
            crow as u32,
            lastv as u32,
            n as u32);
    }
    let w = lanes.width;
    let pad = crow % w;
    let vlen = (pad + lastv).div_ceil(w) * w;
    let (mut vr, mut vi) = (vec![0.0f64; vlen], vec![0.0f64; vlen]);
    vr[pad..pad + lastv].copy_from_slice(&v_re[..lastv]);
    vi[pad..pad + lastv].copy_from_slice(&v_im[..lastv]);
    let (h_vr, h_vi) = (super::workspace::upload(client, &vr), super::workspace::upload(client, &vi));
    let (h_wr, h_wi) = (client.empty(lastc * 8), client.empty(lastc * 8));
    let (count, dim) = launch_1d(client, lastc, lastv);
    // SAFETY: z reads j*lda + crow + r < lda*ncols (j < lastc <= ncols, crow +
    // r < lda); v reads pad + r < vlen; work writes j < LASTC <= n; lastc_buf
    // holds one u32.
    unsafe {
        kl!(ob_zormtr::ob_zgemv_c_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(z.re.clone(), lda * ncols),
            ArrayArg::from_raw_parts(z.im.clone(), lda * ncols),
            ArrayArg::from_raw_parts(h_vr.clone(), vlen),
            ArrayArg::from_raw_parts(h_vi.clone(), vlen),
            ArrayArg::from_raw_parts(h_wr.clone(), lastc),
            ArrayArg::from_raw_parts(h_wi.clone(), lastc),
            ArrayArg::from_raw_parts(lastc_buf.clone(), 1),
            lda as u32,
            crow as u32,
            lastv as u32,
            pad as u32);
    }
    assert!(lda % w == 0);
    let (rv0, units_x) = aligned_units(w, crow, crow + lastv);
    let units = units_x * lastc;
    let (count, dim) = launch_1d(client, units, w);
    // SAFETY: c vector view j*lda/w + rv < lda*ncols/w; x vector view rv -
    // (crow - pad)/w < vlen/w; work reads j < lastc; lane_ofs holds w.
    unsafe {
        kl!(ob_zormtr::ob_zgerc_vec_kernel;
            client,
            count,
            dim,
            w,
            ArrayArg::from_raw_parts(z.re.clone(), lda * ncols),
            ArrayArg::from_raw_parts(z.im.clone(), lda * ncols),
            ArrayArg::from_raw_parts(h_vr, vlen),
            ArrayArg::from_raw_parts(h_vi, vlen),
            ArrayArg::from_raw_parts(h_wr, lastc),
            ArrayArg::from_raw_parts(h_wi, lastc),
            ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
            (lda / w) as u32,
            rv0 as u32,
            units_x as u32,
            units as u32,
            crow as u32,
            lastv as u32,
            ((crow - pad) / w) as u32,
            neg_tau.0,
            neg_tau.1,
            ArrayArg::from_raw_parts(lastc_buf, 1));
    }
}

/// `T1 = alpha·x` exactly as `zsymv_[UL]_sse2.S` builds its `NEW_X` buffer
/// (`x_r·a_r + x_i·(−a_i)`, `x_r·a_i + x_i·a_r`, unfused): the input K1c-ob
/// reads in place of `x`.
pub fn ob_hemv_t1(xr: &[f64], xi: &[f64], alpha: (f64, f64)) -> (Vec<f64>, Vec<f64>) {
    let ai_neg = -alpha.1;
    xr.iter().zip(xi).map(|(&r, &i)| ((r * alpha.0) + (i * ai_neg), (r * alpha.1) + (i * alpha.0))).unzip()
}

/// K1c-ob: `y(off..off+m) = ZHEMV(UPLO, m, alpha, M[off.., off..], x, 0)` in
/// OpenBLAS `zhemv_{L,U}_SKYLAKEX` order (planar, `M` fully Hermitian). `x`
/// holds `T1 = alpha·x` from [`ob_hemv_t1`].
#[allow(clippy::too_many_arguments)]
pub fn ob_hemv<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    m_buf: &ZHandle,
    lda: usize,
    ncols: usize,
    off: usize,
    m: usize,
    x: &ZHandle,
    y: &ZHandle,
    alpha: (F, F),
    upper: bool,
) {
    assert!(off + m <= lda && off + m <= ncols && m > 0);
    let (count, dim) = launch_1d(client, m, 2 * m);
    // SAFETY: m_buf planes hold lda*ncols scalars, read at (off+c)*lda + off
    // + q with c, q < m; `x` planes hold `T1 = alpha·x` (m scalars, see
    // [`ob_hemv_t1`]); y planes hold lda, written at off + r < lda.
    unsafe {
        kl!(super::kernels::ob_zhetrd::ob_hemv_row_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(m_buf.re.clone(), lda * ncols),
            ArrayArg::from_raw_parts(m_buf.im.clone(), lda * ncols),
            ArrayArg::from_raw_parts(x.re.clone(), m),
            ArrayArg::from_raw_parts(x.im.clone(), m),
            ArrayArg::from_raw_parts(y.re.clone(), lda),
            ArrayArg::from_raw_parts(y.im.clone(), lda),
            lda as u32,
            off as u32,
            m as u32,
            alpha.0,
            alpha.1,
            upper);
    }
}

/// K2c-ob element classes: the triangle element `(p, q)` of an `m × m`
/// OpenBLAS `zher2k_{U,L}N` block (one `k` block) is *special* when it is in a
/// diagonal `4 × 4` scratch piece or in the `m % 4` tail rows of its kernel
/// call (the two-chain `m2`/`m1` accumulation). See
/// [`crate::device::kernels::ob_zhetrd`].
pub fn ob_her2k_special_elem(m: usize, upper: bool, p: usize, q: usize) -> bool {
    const P: usize = 256;
    const MN: usize = 4;
    let prule = |rem: usize| {
        if rem >= 2 * P {
            P
        } else if rem > P {
            (rem / 2).div_ceil(MN) * MN
        } else {
            rem
        }
    };
    let (mut s, mut e) = (0, prule(m));
    while e <= p {
        s = e;
        e = s + prule(m - s);
    }
    let mb = e - s;
    let m21 = |rr: usize, cm: usize| rr >= cm - cm % 4;
    if upper {
        if q >= e {
            if s == 0 { m21(p, e) } else { m21(p - s, mb) }
        } else {
            let lp = ((q - s) / MN) * MN;
            p - s >= lp || m21(p - s, lp)
        }
    } else if q < s {
        m21(p - s, mb)
    } else {
        let lp = ((q - s) / MN) * MN;
        let nn = (mb - lp).min(MN);
        p < s + lp + nn || m21(p - s - lp - nn, mb - lp - nn)
    }
}

/// Per block column `j`, the block-row intervals of K2c-ob special elements
/// (either triangle), in closed form (CSR: `offsets[j]..offsets[j+1]` pairs).
/// Equal to scanning [`ob_her2k_special_elem`] (gated by
/// `tests/device_openblas_cpu.rs`).
pub fn ob_her2k_special_intervals(m: usize, upper: bool) -> (Vec<u32>, Vec<u32>) {
    const P: usize = 256;
    const MN: usize = 4;
    let prule = |rem: usize| {
        if rem >= 2 * P {
            P
        } else if rem > P {
            (rem / 2).div_ceil(MN) * MN
        } else {
            rem
        }
    };
    let mut blocks = Vec::new();
    let mut s = 0;
    while s < m {
        let e = s + prule(m - s);
        blocks.push((s, e));
        s = e;
    }
    let block_of = |p: usize| *blocks.iter().find(|&&(s, e)| p >= s && p < e).expect("row block");
    let tail0 = |sb: usize, mb: usize| sb + mb - mb % 4;
    let mut offsets = Vec::with_capacity(m + 1);
    let mut ivals = Vec::new();
    for j in 0..m {
        offsets.push((ivals.len() / 2) as u32);
        let mut push = |lo: usize, hi: usize| {
            if lo < hi {
                ivals.push(lo as u32);
                ivals.push(hi as u32);
            }
        };
        if !upper {
            for &(sb, eb) in &blocks {
                if eb <= j {
                    continue;
                }
                let mb = eb - sb;
                if j < sb {
                    push(tail0(sb, mb), eb);
                } else {
                    let lp = ((j - sb) / MN) * MN;
                    let nn = (mb - lp).min(MN);
                    push(j, sb + lp + nn);
                    let cm = mb - lp - nn;
                    push(sb + lp + nn + cm - cm % 4, eb);
                }
            }
            let (s, e) = block_of(j);
            let mb = e - s;
            let lpj = ((j - s) / MN) * MN;
            if j >= tail0(s, mb) {
                push(0, s);
                push(s, s + lpj);
            }
            push(s + lpj, j);
        } else {
            for &(sb, eb) in &blocks {
                if sb > j {
                    break;
                }
                let mb = eb - sb;
                if j >= eb {
                    push(tail0(sb, mb), eb);
                } else {
                    let lp = ((j - sb) / MN) * MN;
                    push(sb + lp, j + 1);
                    if lp % 4 != 0 {
                        push(sb + lp - lp % 4, sb + lp);
                    }
                }
            }
            let (s, e) = block_of(j);
            let mb = e - s;
            if j >= tail0(s, mb) {
                push(e, m);
            }
            let lpj = ((j - s) / MN) * MN;
            let nnj = (mb - lpj).min(MN);
            push(j + 1, s + lpj + nnj);
        }
    }
    offsets.push((ivals.len() / 2) as u32);
    (offsets, ivals)
}

/// K2c-ob: `M[off.., off..] += alpha·V·Wᴴ + conj(alpha)·W·Vᴴ` on both
/// triangles in OpenBLAS `zher2k_{U,L}N` order (`beta = 1`, one `k` block);
/// `vw` planes pack `nb` V then `nb` W columns aligned with `M`. Plain
/// elements run as lane-vector units, the special ones
/// ([`ob_her2k_special_intervals`], widened to lane-vectors) as a scalar
/// element list.
#[allow(clippy::too_many_arguments)]
pub fn ob_her2k<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    lanes: &Lanes,
    c_buf: &ZHandle,
    lda: usize,
    ncols: usize,
    off: usize,
    m: usize,
    vw: &ZHandle,
    nb: usize,
    alpha: (F, F),
    upper: bool,
) {
    assert!(off + m <= lda && off + m <= ncols && m > 0 && nb > 0 && nb < 2 * 128);
    let w = lanes.width;
    let (ioff, ivals) = ob_her2k_special_intervals(m, upper);
    let mut elems: Vec<u32> = Vec::new();
    if w > 1 && lda % w == 0 {
        let (rv0, units_x) = aligned_units(w, off, off + m);
        let h_ioff = super::workspace::upload(client, &ioff);
        let h_ivals = super::workspace::upload(client, if ivals.is_empty() { &[0u32, 0u32][..] } else { &ivals[..] });
        let units = units_x * m;
        let (count, dim) = launch_1d(client, units, 4 * w * nb);
        // SAFETY: c planes vector view (g0+j)*lda/w + rv < lda*ncols/w; vw planes
        // vector view l*lda/w + rv and scalar view l*lda + g0 + j with l < 2*nb;
        // ioff holds m+1, ivals >= 2; lane_ofs holds w.
        unsafe {
            kl!(super::kernels::ob_zhetrd::ob_her2k_vec_kernel;
                client,
                count,
                dim,
                w,
                ArrayArg::from_raw_parts(c_buf.re.clone(), lda * ncols),
                ArrayArg::from_raw_parts(c_buf.im.clone(), lda * ncols),
                ArrayArg::from_raw_parts(vw.re.clone(), 2 * nb * lda),
                ArrayArg::from_raw_parts(vw.im.clone(), 2 * nb * lda),
                ArrayArg::from_raw_parts(vw.re.clone(), 2 * nb * lda),
                ArrayArg::from_raw_parts(vw.im.clone(), 2 * nb * lda),
                ArrayArg::from_raw_parts(h_ioff, m + 1),
                ArrayArg::from_raw_parts(h_ivals, ivals.len().max(2)),
                ArrayArg::from_raw_parts(lanes.lane_ofs.clone(), w),
                lda as u32,
                off as u32,
                m as u32,
                rv0 as u32,
                units_x as u32,
                units as u32,
                nb as u32,
                alpha.0,
                alpha.1,
                upper);
        }
        let mut rows: Vec<usize> = Vec::new();
        for j in 0..m {
            rows.clear();
            for t in ioff[j] as usize..ioff[j + 1] as usize {
                let (lo, hi) = (ivals[2 * t] as usize + off, ivals[2 * t + 1] as usize + off);
                rows.extend(((lo / w) * w).max(off)..(hi.div_ceil(w) * w).min(off + m));
            }
            rows.sort_unstable();
            rows.dedup();
            for &r in &rows {
                elems.push((r - off) as u32);
                elems.push(j as u32);
            }
        }
    } else {
        for j in 0..m {
            for i in 0..m {
                elems.push(i as u32);
                elems.push(j as u32);
            }
        }
    }
    if elems.is_empty() {
        return;
    }
    let cnt = elems.len() / 2;
    let h_el = super::workspace::upload(client, &elems);
    let (count, dim) = launch_1d(client, cnt, 8 * nb);
    // SAFETY: elems hold (i, j) < m; c planes index (g0+j)*lda + g0+i <
    // lda*ncols; vw planes index l*lda + g0 + i with l < 2*nb.
    unsafe {
        kl!(super::kernels::ob_zhetrd::ob_her2k_list_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(c_buf.re.clone(), lda * ncols),
            ArrayArg::from_raw_parts(c_buf.im.clone(), lda * ncols),
            ArrayArg::from_raw_parts(vw.re.clone(), 2 * nb * lda),
            ArrayArg::from_raw_parts(vw.im.clone(), 2 * nb * lda),
            ArrayArg::from_raw_parts(h_el, 2 * cnt),
            lda as u32,
            off as u32,
            m as u32,
            nb as u32,
            cnt as u32,
            alpha.0,
            alpha.1,
            upper);
    }
}
