//! Device side of the BLAS flavours ([`crate::flavor`]): which kernels the
//! hybrid drivers launch for the matrix-sized operations. The Reference
//! flavour launches the reference-order kernels (K1, K2, K3m, K7rot); the
//! OpenBLAS flavour the OpenBLAS-order ones (K1ob, K2ob, K3m-ob, K7rot-ob).
//! Pure moves (gathers, scatters, swaps, identity fills) are shared.
#![allow(clippy::too_many_arguments)]

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ops::{self, Chunk, Lanes, ZHandle};
use crate::flavor::{Blas, RefBlas};
use crate::openblas::OpenBlas;

/// The device operations whose bits depend on the BLAS flavour.
pub(crate) trait DeviceBlas: Blas {
    /// `DLAED3` for large `K` may run on the device (K4). Only for flavours
    /// whose `DNRM2` K4 reproduces.
    const SECULAR_ON_DEVICE: bool;
    /// `ZHEMV`'s `x` upload is `T1 = alpha·x` (OpenBLAS `NEW_X`) rather than `x`.
    const HEMV_T1: bool;
    /// `DORMTR`'s blocked `DLARFB` runs on the device: 0 = host, 1 = the
    /// Reference engine (`ormtr::DeviceOrm`), 2 = the OpenBLAS engine
    /// (`ormtr::DeviceOrmOb`).
    const ORM_ON_DEVICE: u8;
    /// `Exactness::Fast` (lean merge, fused `DLARFB`) is available; it is
    /// defined against the Reference kernels only.
    const FAST_OK: bool;
    /// `DORMTR`'s `LWORK` inside `DSYEVD` for order `n`: the Reference track
    /// reproduces [`crate::syevd_opt_work_len`] (blocked), the OpenBLAS track
    /// scipy's query value [`crate::syevd_work_len`] (`LLWRK2 = 1 + 4n + n²`).
    fn orm_lwork(n: usize) -> usize;
    /// `ZHETRD`'s matrix-sized operations run on the device (K1c/K2c).
    const ZTRD_ON_DEVICE: bool;
    /// `ZUNMTR` on the device: 1 = the Reference blocked engine
    /// (`complex::DeviceZOrm`, `n > 33`), 2 = the OpenBLAS unblocked engine
    /// (`complex::DeviceZOrmOb`, every `n`).
    const ZORM_KIND: u8;
    /// `ZUNMTR`'s `LWORK` inside `ZHEEVD` (`LLWRK2`), as [`Self::orm_lwork`].
    fn zorm_lwork(n: usize) -> usize;

    /// `y(off..off+m) = DSYMV(UPLO, m, alpha, M[off.., off..], x, 0)`.
    fn symv<R: Runtime>(
        client: &ComputeClient<R>,
        lanes: &Lanes,
        m_buf: &Handle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        x: &Handle,
        y: &Handle,
        alpha: f64,
        upper: bool,
    );
    /// `M[off.., off..] += alpha(V·Wᵀ + W·Vᵀ)` (`DSYR2K`, `beta = 1`).
    fn syr2k<R: Runtime>(
        client: &ComputeClient<R>,
        lanes: &Lanes,
        c_buf: &Handle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        vw: &Handle,
        nb: usize,
        alpha: f64,
        upper: bool,
    );
    /// `y(off..off+m) = ZHEMV(UPLO, m, CONE, M[off.., off..], x, CZERO)`.
    fn dev_zhemv<R: Runtime>(
        client: &ComputeClient<R>,
        lanes: &Lanes,
        m_buf: &ZHandle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        x: &ZHandle,
        y: &ZHandle,
        upper: bool,
    );
    /// `M[off.., off..] += −V·Wᴴ − W·Vᴴ` (`ZHER2K`, `alpha = −CONE`, `beta = 1`).
    fn dev_zher2k<R: Runtime>(
        client: &ComputeClient<R>,
        lanes: &Lanes,
        c_buf: &ZHandle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        vw: &ZHandle,
        nb: usize,
        upper: bool,
    );
    /// `DLAED2`'s batched rotations on rows `[lo, hi)`.
    fn rotations<R: Runtime>(
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
    );
    /// K3m over rows `[lo, hi)` of the merge block rows `[blk_r0,
    /// blk_r0 + blk_m)` (see [`ops::merge_rows`]).
    fn merge_rows<R: Runtime>(
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
        blk_r0: usize,
        blk_m: usize,
    );
}

impl DeviceBlas for RefBlas {
    fn dev_zhemv<R: Runtime>(
        client: &ComputeClient<R>,
        lanes: &Lanes,
        m_buf: &ZHandle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        x: &ZHandle,
        y: &ZHandle,
        upper: bool,
    ) {
        ops::hemv::<R, f64>(client, lanes, m_buf, lda, ncols, off, m, x, y, (1.0, 0.0), upper)
    }
    fn dev_zher2k<R: Runtime>(
        client: &ComputeClient<R>,
        lanes: &Lanes,
        c_buf: &ZHandle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        vw: &ZHandle,
        nb: usize,
        upper: bool,
    ) {
        ops::her2k::<R, f64>(client, lanes, c_buf, lda, ncols, off, m, vw, nb, (-1.0, -0.0), upper)
    }
    const SECULAR_ON_DEVICE: bool = true;
    const HEMV_T1: bool = false;
    const ORM_ON_DEVICE: u8 = 1;
    const FAST_OK: bool = true;
    const ZTRD_ON_DEVICE: bool = true;
    const ZORM_KIND: u8 = 1;
    fn orm_lwork(n: usize) -> usize {
        n * crate::sytrd::ORM_NB + crate::sytrd::TSIZE
    }
    fn zorm_lwork(n: usize) -> usize {
        n * crate::sytrd::ORM_NB + crate::sytrd::TSIZE
    }
    fn symv<R: Runtime>(
        client: &ComputeClient<R>,
        lanes: &Lanes,
        m_buf: &Handle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        x: &Handle,
        y: &Handle,
        alpha: f64,
        upper: bool,
    ) {
        ops::symv::<R, f64>(client, lanes, m_buf, lda, ncols, off, m, x, y, alpha, upper)
    }
    fn syr2k<R: Runtime>(
        client: &ComputeClient<R>,
        lanes: &Lanes,
        c_buf: &Handle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        vw: &Handle,
        nb: usize,
        alpha: f64,
        upper: bool,
    ) {
        ops::syr2k::<R, f64>(client, lanes, c_buf, lda, ncols, off, m, vw, nb, alpha, upper)
    }
    fn rotations<R: Runtime>(
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
        ops::rotations::<R, f64>(client, lanes, z, lda, ncols, cols, cs, nrot, lo, hi)
    }
    fn merge_rows<R: Runtime>(
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
        _blk_r0: usize,
        _blk_m: usize,
    ) {
        ops::merge_rows::<R, f64>(
            client, lanes, z, lda, ncols, dst, chunk, n, k, perm, nterms, perm4, s, lds, s_row0, lo, hi,
        )
    }
}

impl DeviceBlas for OpenBlas {
    fn dev_zhemv<R: Runtime>(
        client: &ComputeClient<R>,
        _lanes: &Lanes,
        m_buf: &ZHandle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        x: &ZHandle,
        y: &ZHandle,
        upper: bool,
    ) {
        // `x` already holds T1 = CONE·x (the engine builds it with `ops::ob_hemv_t1`).
        ops::ob_hemv::<R, f64>(client, m_buf, lda, ncols, off, m, x, y, (1.0, 0.0), upper)
    }
    fn dev_zher2k<R: Runtime>(
        client: &ComputeClient<R>,
        lanes: &Lanes,
        c_buf: &ZHandle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        vw: &ZHandle,
        nb: usize,
        upper: bool,
    ) {
        ops::ob_her2k::<R, f64>(client, lanes, c_buf, lda, ncols, off, m, vw, nb, (-1.0, -0.0), upper)
    }
    // K4's `U` columns are normalised with the reference `dnrm2.f90`;
    // OpenBLAS's is the x87 `nrm2.S`, so `DLAED3` stays on the host.
    const SECULAR_ON_DEVICE: bool = false;
    const HEMV_T1: bool = true;
    const ORM_ON_DEVICE: u8 = 2;
    const FAST_OK: bool = false;
    const ZTRD_ON_DEVICE: bool = true;
    const ZORM_KIND: u8 = 2;
    fn orm_lwork(n: usize) -> usize {
        1 + 4 * n + n * n
    }
    fn zorm_lwork(n: usize) -> usize {
        crate::heevd_work_len(n, crate::Job::Vectors).0 - n - n * n
    }
    fn symv<R: Runtime>(
        client: &ComputeClient<R>,
        _lanes: &Lanes,
        m_buf: &Handle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        x: &Handle,
        y: &Handle,
        alpha: f64,
        upper: bool,
    ) {
        ops::ob_symv::<R, f64>(client, m_buf, lda, ncols, off, m, x, y, alpha, upper)
    }
    fn syr2k<R: Runtime>(
        client: &ComputeClient<R>,
        lanes: &Lanes,
        c_buf: &Handle,
        lda: usize,
        ncols: usize,
        off: usize,
        m: usize,
        vw: &Handle,
        nb: usize,
        alpha: f64,
        upper: bool,
    ) {
        ops::ob_syr2k::<R, f64>(client, lanes, c_buf, lda, ncols, off, m, vw, nb, alpha, upper)
    }
    fn rotations<R: Runtime>(
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
        ops::ob_rotations::<R, f64>(client, lanes, z, lda, ncols, cols, cs, nrot, lo, hi)
    }
    fn merge_rows<R: Runtime>(
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
        blk_r0: usize,
        blk_m: usize,
    ) {
        let w = lanes.width;
        ops::ob_merge_rows::<R, f64>(
            client,
            lanes,
            z,
            lda,
            ncols,
            dst,
            chunk.rvecs * w,
            chunk.rv0 * w,
            n,
            k,
            perm,
            nterms,
            perm4,
            s,
            lds,
            s_row0,
            lo,
            hi,
            blk_r0,
            blk_m,
        )
    }
}
