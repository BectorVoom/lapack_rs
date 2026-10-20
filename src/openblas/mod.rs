//! **OpenBLAS oracle (third track).** Bit-exact to scipy's bundled OpenBLAS
//! 0.3.30 (`libscipy_openblas-6cdc3b4a.so`, `DYNAMIC_ARCH` build dispatched to
//! the `SkylakeX` core, `OPENBLAS_NUM_THREADS=1`), never to Reference-LAPACK
//! or libcint, and never asserted equal to either. Its oracle is
//! `crates/openblas_ref`; its gates are `tests/openblas_*_bitexact.rs`.
//!
//! OpenBLAS vendors Reference-LAPACK 3.12.1 unchanged except `DORM2L`/
//! `DORM2R` (`ZUNM2L`/`ZUNM2R`), which still call `DLARF`; it compiles that
//! Fortran without FMA. Every difference is therefore in the BLAS kernels,
//! transcribed here from the binary (with the C/assembly source for
//! structure). Unlike the Reference track these kernels **do** fuse
//! multiply-adds (`f64::mul_add`), exactly where gcc emitted `vfmadd*`, and
//! `DNRM2` runs in x87 extended precision (emulated in [`level1`]).
//!
//! [`OpenBlas`] plugs them into the shared LAPACK transcription through
//! [`crate::flavor::Blas`].
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments, clippy::missing_safety_doc)]

#[doc(hidden)]
pub mod level1;
#[doc(hidden)]
pub mod level2;
#[doc(hidden)]
pub mod level3;
#[doc(hidden)]
pub mod zlevel1;
#[doc(hidden)]
pub mod zlevel2;
#[doc(hidden)]
pub mod zlevel3;

use crate::flavor::Blas;

/// The host transcriptions are compiled with `target_feature(enable = "fma")`
/// so that `mul_add` is one `vfmadd` instruction instead of a libm call (the
/// value is the same either way: both are the exactly rounded fused
/// multiply-add). Calling them on a CPU without FMA would be undefined, so
/// every public entry point checks first. The oracle this track reproduces
/// (the `SkylakeX` core) implies AVX-512, hence FMA.
pub(crate) fn require_fma() {
    #[cfg(target_arch = "x86_64")]
    assert!(
        std::arch::is_x86_feature_detected!("fma"),
        "lapack_rs::openblas needs a CPU with FMA (the OpenBLAS SkylakeX track)"
    );
    #[cfg(not(target_arch = "x86_64"))]
    panic!("lapack_rs::openblas reproduces an x86_64 OpenBLAS build");
}
use crate::ptr::D;
use crate::zblas::{Tr, Z, ZV};

/// scipy's OpenBLAS 0.3.30, `SkylakeX` core, one thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenBlas;

impl Blas for OpenBlas {
    /// OpenBLAS 0.3.30 vendors `DORM2L`/`DORM2R` calling `DLARF`.
    const LEGACY_DLARF: bool = true;

    unsafe fn ddot(n: usize, x: D, incx: usize, y: D, incy: usize) -> f64 {
        level1::ddot(n, x, incx, y, incy)
    }
    unsafe fn dnrm2(n: usize, x: D, incx: usize) -> f64 {
        level1::dnrm2(n, x, incx)
    }
    unsafe fn daxpy(n: usize, da: f64, x: D, incx: usize, y: D, incy: usize) {
        level1::daxpy(n, da, x, incx, y, incy)
    }
    unsafe fn dscal(n: usize, da: f64, x: D, incx: usize) {
        level1::dscal(n, da, x, incx)
    }
    unsafe fn drot(n: usize, x: D, incx: usize, y: D, incy: usize, c: f64, s: f64) {
        level1::drot(n, x, incx, y, incy, c, s)
    }
    unsafe fn idamax(n: usize, x: D, incx: usize) -> usize {
        level1::idamax(n, x, incx)
    }
    unsafe fn dsymv(upper: bool, n: usize, alpha: f64, a: D, lda: usize, x: D, beta: f64, y: D) {
        level2::dsymv(upper, n, alpha, a, lda, x, beta, y)
    }
    unsafe fn dsyr2(upper: bool, n: usize, alpha: f64, x: D, y: D, a: D, lda: usize) {
        level2::dsyr2(upper, n, alpha, x, y, a, lda)
    }
    unsafe fn dgemv(
        trans: bool,
        m: usize,
        n: usize,
        alpha: f64,
        a: D,
        lda: usize,
        x: D,
        incx: usize,
        beta: f64,
        y: D,
        incy: usize,
    ) {
        level2::dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
    unsafe fn dger(m: usize, n: usize, alpha: f64, x: D, incx: usize, y: D, incy: usize, a: D, lda: usize) {
        level2::dger(m, n, alpha, x, incx, y, incy, a, lda)
    }
    unsafe fn dsyr2k_n(
        upper: bool,
        n: usize,
        k: usize,
        alpha: f64,
        a: D,
        lda: usize,
        b: D,
        ldb: usize,
        beta: f64,
        c: D,
        ldc: usize,
    ) {
        level3::dsyr2k_n(upper, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
    unsafe fn dgemm(
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: D,
        lda: usize,
        b: D,
        ldb: usize,
        beta: f64,
        c: D,
        ldc: usize,
    ) {
        level3::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
    unsafe fn dtrmm(
        left: bool,
        upper: bool,
        trans: bool,
        nounit: bool,
        m: usize,
        n: usize,
        alpha: f64,
        a: D,
        lda: usize,
        b: D,
        ldb: usize,
    ) {
        level3::dtrmm(left, upper, trans, nounit, m, n, alpha, a, lda, b, ldb)
    }
    unsafe fn zdotc(n: usize, x: ZV, incx: usize, y: ZV, incy: usize) -> Z {
        zlevel1::zdotc(n, x, incx, y, incy)
    }
    unsafe fn dznrm2(n: usize, x: ZV, incx: usize) -> f64 {
        zlevel1::dznrm2(n, x, incx)
    }
    unsafe fn zaxpy(n: usize, za: Z, x: ZV, incx: usize, y: ZV, incy: usize) {
        zlevel1::zaxpy(n, za, x, incx, y, incy)
    }
    unsafe fn zscal(n: usize, za: Z, x: ZV, incx: usize) {
        zlevel1::zscal(n, za, x, incx)
    }
    unsafe fn zdscal(n: usize, da: f64, x: ZV, incx: usize) {
        zlevel1::zdscal(n, da, x, incx)
    }
    unsafe fn zgemv(
        trans: Tr,
        m: usize,
        n: usize,
        alpha: Z,
        a: ZV,
        lda: usize,
        x: ZV,
        incx: usize,
        beta: Z,
        y: ZV,
        incy: usize,
    ) {
        zlevel2::zgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
    unsafe fn zgerc(m: usize, n: usize, alpha: Z, x: ZV, incx: usize, y: ZV, incy: usize, a: ZV, lda: usize) {
        zlevel2::zgerc(m, n, alpha, x, incx, y, incy, a, lda)
    }
    unsafe fn zhemv(upper: bool, n: usize, alpha: Z, a: ZV, lda: usize, x: ZV, beta: Z, y: ZV) {
        zlevel2::zhemv(upper, n, alpha, a, lda, x, beta, y)
    }
    unsafe fn zher2(upper: bool, n: usize, alpha: Z, x: ZV, y: ZV, a: ZV, lda: usize) {
        zlevel2::zher2(upper, n, alpha, x, y, a, lda)
    }
    unsafe fn zher2k_n(
        upper: bool,
        n: usize,
        k: usize,
        alpha: Z,
        a: ZV,
        lda: usize,
        b: ZV,
        ldb: usize,
        beta: f64,
        c: ZV,
        ldc: usize,
    ) {
        zlevel3::zher2k_n(upper, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
    unsafe fn zgemm(
        transa: Tr,
        transb: Tr,
        m: usize,
        n: usize,
        k: usize,
        alpha: Z,
        a: ZV,
        lda: usize,
        b: ZV,
        ldb: usize,
        beta: Z,
        c: ZV,
        ldc: usize,
    ) {
        zlevel3::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
    unsafe fn ztrmm(
        left: bool,
        upper: bool,
        trans: Tr,
        nounit: bool,
        m: usize,
        n: usize,
        alpha: Z,
        a: ZV,
        lda: usize,
        b: ZV,
        ldb: usize,
    ) {
        zlevel3::ztrmm(left, upper, trans, nounit, m, n, alpha, a, lda, b, ldb)
    }
}

/// `DSYEVD` bit-exact to scipy's OpenBLAS (`scipy_dsyevd_` called with its own
/// workspace query, i.e. [`crate::syevd_work_len`]): the sequential host
/// transcription of the OpenBLAS track.
pub fn dsyevd_host(
    job: crate::Job,
    uplo: crate::Uplo,
    n: usize,
    a: &mut [f64],
    w: &mut [f64],
) -> Result<(), crate::LapackError> {
    require_fma();
    use crate::ptr::V;
    if a.len() < n * n || w.len() < n {
        return Err(crate::LapackError::InvalidArgument {
            arg: "a/w",
            reason: format!("need n*n and n entries (n={n})"),
        });
    }
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        w[0] = a[0];
        if job == crate::Job::Vectors {
            a[0] = 1.0;
        }
        return Ok(());
    }
    let (lwork, liwork) = crate::syevd_work_len(n, job);
    let mut work = vec![0.0f64; lwork.max(1)];
    let mut iwork = vec![0usize; liwork.max(1)];
    let wantz = job == crate::Job::Vectors;
    // SAFETY: lengths checked/allocated at the query sizes.
    let info = unsafe {
        crate::syevd::dsyevd_raw::<OpenBlas>(
            wantz,
            uplo == crate::Uplo::Lower,
            n,
            V::new(a),
            V::new(w),
            V::new(&mut work),
            lwork,
            V::new(&mut iwork),
            liwork,
        )
    };
    if info != 0 {
        return Err(crate::LapackError::NoConvergence { routine: if wantz { "dstedc" } else { "dsterf" }, info });
    }
    Ok(())
}

/// `ZHEEVD` bit-exact to scipy's OpenBLAS (`scipy_zheevd_` with its own
/// workspace query, i.e. [`crate::heevd_work_len`]) on planar storage: the
/// sequential host transcription of the OpenBLAS track.
pub fn zheevd_host(
    job: crate::Job,
    uplo: crate::Uplo,
    n: usize,
    a_re: &mut [f64],
    a_im: &mut [f64],
    w: &mut [f64],
) -> Result<(), crate::LapackError> {
    require_fma();
    use crate::ptr::V;
    if a_re.len() < n * n || a_im.len() < n * n || w.len() < n {
        return Err(crate::LapackError::InvalidArgument {
            arg: "a/w",
            reason: format!("need n*n and n entries (n={n})"),
        });
    }
    let wantz = job == crate::Job::Vectors;
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        w[0] = a_re[0];
        if wantz {
            a_re[0] = 1.0;
            a_im[0] = 0.0;
        }
        return Ok(());
    }
    let (lwork, lrwork, liwork) = crate::heevd_work_len(n, job);
    let (mut wre, mut wim) = (vec![0.0f64; lwork], vec![0.0f64; lwork]);
    let mut rwork = vec![0.0f64; lrwork];
    let mut iwork = vec![0usize; liwork];
    // SAFETY: lengths checked/allocated at the query sizes; distinct slices.
    let info = unsafe {
        crate::heevd::zheevd_raw::<OpenBlas>(
            wantz,
            uplo == crate::Uplo::Lower,
            n,
            ZV::new(V::new(a_re), V::new(a_im)),
            V::new(w),
            ZV::new(V::new(&mut wre), V::new(&mut wim)),
            lwork,
            V::new(&mut rwork),
            lrwork,
            V::new(&mut iwork),
        )
    };
    if info != 0 {
        return Err(crate::LapackError::NoConvergence { routine: if wantz { "zstedc" } else { "dsterf" }, info });
    }
    Ok(())
}

/// `DSYEVD` bit-exact to scipy's OpenBLAS on the default CubeCL runtime
/// ([`crate::runtime`]): the hybrid device driver with the OpenBLAS-order
/// kernels (K1ob `DSYMV`, K2ob `DSYR2K`, K7rot-ob, K3m-ob). Bit-identical to
/// [`dsyevd_host`], which is gated against the library itself.
pub fn dsyevd(
    job: crate::Job,
    uplo: crate::Uplo,
    n: usize,
    a: &mut [f64],
    w: &mut [f64],
) -> Result<(), crate::LapackError> {
    require_fma();
    crate::runtime::with_workspace(n, |client, ws| {
        crate::device::syevd::dsyevd_device_impl::<_, OpenBlas>(client, job, uplo, n, a, w, ws, None, false).map(|_| ())
    })
}

/// `ZHEEVD` bit-exact to scipy's OpenBLAS on the default CubeCL runtime
/// ([`crate::runtime`]), planar storage: bit-identical to [`zheevd_host`].
pub fn zheevd(
    job: crate::Job,
    uplo: crate::Uplo,
    n: usize,
    a_re: &mut [f64],
    a_im: &mut [f64],
    w: &mut [f64],
) -> Result<(), crate::LapackError> {
    crate::runtime::with_workspace(n, |client, ws| {
        crate::device::zheevd_device_openblas(client, job, uplo, n, a_re, a_im, w, ws)
    })
}
