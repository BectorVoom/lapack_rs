//! **OpenBLAS oracle (third track).** Test-only FFI to scipy's bundled OpenBLAS
//! 0.3.30 (`libscipy_openblas-*.so`, symbols prefixed `scipy_`). Never
//! compared with, or asserted equal to, the Reference-LAPACK oracle
//! (`lapack_ref`) or the libcint oracle (`cint_ref`).
//!
//! OpenBLAS is a `DYNAMIC_ARCH` build: the kernel set, and so the bits, depend
//! on the core it dispatches to and on the thread count. [`require`] asserts
//! the pinned configuration ([`PINNED_CONFIG`], [`PINNED_CORE`], one thread)
//! before any comparison is trusted.
//!
//! gfortran ABI: every argument by reference, `INTEGER` = `i32`, hidden
//! `CHARACTER` lengths (`usize`) appended after the explicit arguments.
//! Functions are exposed raw (`unsafe`, pointer arguments) under the same
//! names as the Fortran routines without the prefix.

#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

use std::ffi::CStr;

/// Whether the library was linked (`LAPACK_RS_OPENBLAS_DIR` at build time).
pub const AVAILABLE: bool = cfg!(openblas_ref);

/// `scipy_openblas_get_config()` of the build every gate was measured on.
pub const PINNED_CONFIG: &str = "OpenBLAS 0.3.30 DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=64";
/// `scipy_openblas_get_corename()` the transcriptions follow.
pub const PINNED_CORE: &str = "SkylakeX";

/// `COMPLEX*16` as stored.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

/// Path of the linked `.so` (only when [`AVAILABLE`]).
pub fn library_path() -> Option<&'static str> {
    option_env!("LAPACK_RS_OPENBLAS_LIB")
}

macro_rules! ffi_decls {
    ($( fn $name:ident = $sym:ident ( $($a:ident : $t:ty),* $(,)? ) $(-> $r:ty)? ; )*) => {
        #[cfg(openblas_ref)]
        mod sys {
            #[allow(unused_imports)]
            use super::{C64, Ch, Cf, Cz, I, Mf, Mz};
            unsafe extern "C" {
                $( pub fn $sym($($a: $t),*) $(-> $r)?; )*
            }
        }
        $(
            #[inline]
            pub unsafe fn $name($($a: $t),*) $(-> $r)? {
                #[cfg(openblas_ref)]
                {
                    // SAFETY: forwarded; the caller upholds the routine's contract.
                    unsafe { sys::$sym($($a),*) }
                }
                #[cfg(not(openblas_ref))]
                {
                    let _ = ($($a),*);
                    panic!(concat!(stringify!($name), ": OpenBLAS oracle not linked (set LAPACK_RS_OPENBLAS_DIR)"))
                }
            }
        )*
    };
}

type I = *const i32;
type Cf = *const f64;
type Mf = *mut f64;
type Cz = *const C64;
type Mz = *mut C64;
type Ch = *const u8;

ffi_decls! {
    fn openblas_get_config = scipy_openblas_get_config() -> *const std::ffi::c_char;
    fn openblas_get_corename = scipy_openblas_get_corename() -> *const std::ffi::c_char;
    fn openblas_get_parallel = scipy_openblas_get_parallel() -> i32;
    fn openblas_get_num_threads = scipy_openblas_get_num_threads() -> i32;
    fn openblas_set_num_threads = scipy_openblas_set_num_threads(n: i32);

    // LAPACK drivers and the routines they call.
    fn dsyevd = scipy_dsyevd_(jobz: Ch, uplo: Ch, n: I, a: Mf, lda: I, w: Mf, work: Mf, lwork: I, iwork: *mut i32, liwork: I, info: *mut i32, l1: usize, l2: usize);
    fn zheevd = scipy_zheevd_(jobz: Ch, uplo: Ch, n: I, a: Mz, lda: I, w: Mf, work: Mz, lwork: I, rwork: Mf, lrwork: I, iwork: *mut i32, liwork: I, info: *mut i32, l1: usize, l2: usize);
    fn dsytrd = scipy_dsytrd_(uplo: Ch, n: I, a: Mf, lda: I, d: Mf, e: Mf, tau: Mf, work: Mf, lwork: I, info: *mut i32, l1: usize);
    fn zhetrd = scipy_zhetrd_(uplo: Ch, n: I, a: Mz, lda: I, d: Mf, e: Mf, tau: Mz, work: Mz, lwork: I, info: *mut i32, l1: usize);
    fn dsytd2 = scipy_dsytd2_(uplo: Ch, n: I, a: Mf, lda: I, d: Mf, e: Mf, tau: Mf, info: *mut i32, l1: usize);
    fn dlatrd = scipy_dlatrd_(uplo: Ch, n: I, nb: I, a: Mf, lda: I, e: Mf, tau: Mf, w: Mf, ldw: I, l1: usize);
    fn dormtr = scipy_dormtr_(side: Ch, uplo: Ch, trans: Ch, m: I, n: I, a: Mf, lda: I, tau: Mf, c: Mf, ldc: I, work: Mf, lwork: I, info: *mut i32, l1: usize, l2: usize, l3: usize);
    fn zunmtr = scipy_zunmtr_(side: Ch, uplo: Ch, trans: Ch, m: I, n: I, a: Mz, lda: I, tau: Mz, c: Mz, ldc: I, work: Mz, lwork: I, info: *mut i32, l1: usize, l2: usize, l3: usize);
    fn dstedc = scipy_dstedc_(compz: Ch, n: I, d: Mf, e: Mf, z: Mf, ldz: I, work: Mf, lwork: I, iwork: *mut i32, liwork: I, info: *mut i32, l1: usize);
    fn dsteqr = scipy_dsteqr_(compz: Ch, n: I, d: Mf, e: Mf, z: Mf, ldz: I, work: Mf, info: *mut i32, l1: usize);
    fn dsterf = scipy_dsterf_(n: I, d: Mf, e: Mf, info: *mut i32);
    fn dlaed4 = scipy_dlaed4_(n: I, i: I, d: Cf, z: Cf, delta: Mf, rho: Cf, dlam: Mf, info: *mut i32);
    fn dlarfg = scipy_dlarfg_(n: I, alpha: Mf, x: Mf, incx: I, tau: Mf);
    fn zlarfg = scipy_zlarfg_(n: I, alpha: Mz, x: Mz, incx: I, tau: Mz);
    fn dlarfb = scipy_dlarfb_(side: Ch, trans: Ch, direct: Ch, storev: Ch, m: I, n: I, k: I, v: Cf, ldv: I, t: Cf, ldt: I, c: Mf, ldc: I, work: Mf, ldwork: I, l1: usize, l2: usize, l3: usize, l4: usize);
    fn dlarft = scipy_dlarft_(direct: Ch, storev: Ch, n: I, k: I, v: Mf, ldv: I, tau: Cf, t: Mf, ldt: I, l1: usize, l2: usize);
    fn dlarf = scipy_dlarf_(side: Ch, m: I, n: I, v: Cf, incv: I, tau: Cf, c: Mf, ldc: I, work: Mf, l1: usize);
    fn dlacpy = scipy_dlacpy_(uplo: Ch, m: I, n: I, a: Cf, lda: I, b: Mf, ldb: I, l1: usize);
    fn dlapy2 = scipy_dlapy2_(x: Cf, y: Cf) -> f64;
    fn dlamch = scipy_dlamch_(c: Ch, l1: usize) -> f64;
    fn dlanst = scipy_dlanst_(norm: Ch, n: I, d: Cf, e: Cf, l1: usize) -> f64;
    fn dlansy = scipy_dlansy_(norm: Ch, uplo: Ch, n: I, a: Cf, lda: I, work: Mf, l1: usize, l2: usize) -> f64;
    fn dlascl = scipy_dlascl_(ty: Ch, kl: I, ku: I, cfrom: Cf, cto: Cf, m: I, n: I, a: Mf, lda: I, info: *mut i32, l1: usize);
    fn ilaenv = scipy_ilaenv_(ispec: I, name: Ch, opts: Ch, n1: I, n2: I, n3: I, n4: I, l1: usize, l2: usize) -> i32;

    // BLAS level 1.
    fn ddot = scipy_ddot_(n: I, x: Cf, incx: I, y: Cf, incy: I) -> f64;
    fn daxpy = scipy_daxpy_(n: I, alpha: Cf, x: Cf, incx: I, y: Mf, incy: I);
    fn dscal = scipy_dscal_(n: I, alpha: Cf, x: Mf, incx: I);
    fn dnrm2 = scipy_dnrm2_(n: I, x: Cf, incx: I) -> f64;
    fn dcopy = scipy_dcopy_(n: I, x: Cf, incx: I, y: Mf, incy: I);
    fn dswap = scipy_dswap_(n: I, x: Mf, incx: I, y: Mf, incy: I);
    fn drot = scipy_drot_(n: I, x: Mf, incx: I, y: Mf, incy: I, c: Cf, s: Cf);
    fn idamax = scipy_idamax_(n: I, x: Cf, incx: I) -> i32;
    fn dznrm2 = scipy_dznrm2_(n: I, x: Cz, incx: I) -> f64;
    fn zdscal = scipy_zdscal_(n: I, alpha: Cf, x: Mz, incx: I);
    fn zscal = scipy_zscal_(n: I, alpha: Cz, x: Mz, incx: I);
    fn zaxpy = scipy_zaxpy_(n: I, alpha: Cz, x: Cz, incx: I, y: Mz, incy: I);
    // `scipy_zdotc_` is OpenBLAS `interface/zdot.c` built without
    // `RETURN_BY_STACK`/`RETURN_BY_STRUCT`: it returns the `double _Complex`
    // by value, i.e. in xmm0:xmm1 (disassembly: `vmovsd 0x10(%rsp),%xmm0;
    // vmovsd 0x18(%rsp),%xmm1; ret` in `zdotc_k_SKYLAKEX`, tail-jumped to
    // from the interface), which is exactly the SysV return of `#[repr(C)]
    // C64` (two `f64` fields, class SSE,SSE). No hidden result pointer.
    fn zdotc = scipy_zdotc_(n: I, x: Cz, incx: I, y: Cz, incy: I) -> C64;
    fn zcopy = scipy_zcopy_(n: I, x: Cz, incx: I, y: Mz, incy: I);
    fn zswap = scipy_zswap_(n: I, x: Mz, incx: I, y: Mz, incy: I);

    // BLAS level 2.
    fn dgemv = scipy_dgemv_(trans: Ch, m: I, n: I, alpha: Cf, a: Cf, lda: I, x: Cf, incx: I, beta: Cf, y: Mf, incy: I, l1: usize);
    fn dsymv = scipy_dsymv_(uplo: Ch, n: I, alpha: Cf, a: Cf, lda: I, x: Cf, incx: I, beta: Cf, y: Mf, incy: I, l1: usize);
    fn dsyr2 = scipy_dsyr2_(uplo: Ch, n: I, alpha: Cf, x: Cf, incx: I, y: Cf, incy: I, a: Mf, lda: I, l1: usize);
    fn dger = scipy_dger_(m: I, n: I, alpha: Cf, x: Cf, incx: I, y: Cf, incy: I, a: Mf, lda: I);
    fn dtrmv = scipy_dtrmv_(uplo: Ch, trans: Ch, diag: Ch, n: I, a: Cf, lda: I, x: Mf, incx: I, l1: usize, l2: usize, l3: usize);
    fn zhemv = scipy_zhemv_(uplo: Ch, n: I, alpha: Cz, a: Cz, lda: I, x: Cz, incx: I, beta: Cz, y: Mz, incy: I, l1: usize);
    fn zgemv = scipy_zgemv_(trans: Ch, m: I, n: I, alpha: Cz, a: Cz, lda: I, x: Cz, incx: I, beta: Cz, y: Mz, incy: I, l1: usize);
    fn zher2 = scipy_zher2_(uplo: Ch, n: I, alpha: Cz, x: Cz, incx: I, y: Cz, incy: I, a: Mz, lda: I, l1: usize);
    fn zgerc = scipy_zgerc_(m: I, n: I, alpha: Cz, x: Cz, incx: I, y: Cz, incy: I, a: Mz, lda: I);

    // BLAS level 3.
    fn dgemm = scipy_dgemm_(ta: Ch, tb: Ch, m: I, n: I, k: I, alpha: Cf, a: Cf, lda: I, b: Cf, ldb: I, beta: Cf, c: Mf, ldc: I, l1: usize, l2: usize);
    fn dsyr2k = scipy_dsyr2k_(uplo: Ch, trans: Ch, n: I, k: I, alpha: Cf, a: Cf, lda: I, b: Cf, ldb: I, beta: Cf, c: Mf, ldc: I, l1: usize, l2: usize);
    fn dtrmm = scipy_dtrmm_(side: Ch, uplo: Ch, transa: Ch, diag: Ch, m: I, n: I, alpha: Cf, a: Cf, lda: I, b: Mf, ldb: I, l1: usize, l2: usize, l3: usize, l4: usize);
    fn zgemm = scipy_zgemm_(ta: Ch, tb: Ch, m: I, n: I, k: I, alpha: Cz, a: Cz, lda: I, b: Cz, ldb: I, beta: Cz, c: Mz, ldc: I, l1: usize, l2: usize);
    fn zher2k = scipy_zher2k_(uplo: Ch, trans: Ch, n: I, k: I, alpha: Cz, a: Cz, lda: I, b: Cz, ldb: I, beta: Cf, c: Mz, ldc: I, l1: usize, l2: usize);
    fn ztrmm = scipy_ztrmm_(side: Ch, uplo: Ch, transa: Ch, diag: Ch, m: I, n: I, alpha: Cz, a: Cz, lda: I, b: Mz, ldb: I, l1: usize, l2: usize, l3: usize, l4: usize);
}

fn cstr(p: *const std::ffi::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: OpenBLAS returns a static NUL-terminated string.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().trim().to_string()
}

/// `(config, corename, parallel, threads)` as the library reports them,
/// queried once: `openblas_get_config` rebuilds its answer in a static buffer
/// with `strcat` on every call, so concurrent test threads would race on it.
pub fn configuration() -> Option<(String, String, i32, i32)> {
    static CONFIG: std::sync::OnceLock<(String, String, i32, i32)> = std::sync::OnceLock::new();
    if !AVAILABLE {
        return None;
    }
    // SAFETY: argument-less queries, serialised by the `OnceLock`.
    Some(
        CONFIG
            .get_or_init(|| unsafe {
                (
                    cstr(openblas_get_config()),
                    cstr(openblas_get_corename()),
                    openblas_get_parallel(),
                    openblas_get_num_threads(),
                )
            })
            .clone(),
    )
}

/// Standard gate for OpenBLAS oracle tests. Returns `true` when the oracle is
/// linked **and** pinned: config string [`PINNED_CONFIG`], core
/// [`PINNED_CORE`] (set `OPENBLAS_CORETYPE=SkylakeX`) and one thread (forced
/// here, and `OPENBLAS_NUM_THREADS=1` is required so the pool never starts).
/// A mismatched pin panics: a comparison against a different kernel set is
/// not a result. Without the library the test prints `SKIP`, unless
/// `OPENBLAS_REQUIRE_REF=1`, which panics.
pub fn require(test: &str) -> bool {
    let Some((config, core, _parallel, _threads)) = configuration() else {
        if std::env::var("OPENBLAS_REQUIRE_REF").as_deref() == Ok("1") {
            panic!("{test}: OpenBLAS oracle not linked (set LAPACK_RS_OPENBLAS_DIR) but OPENBLAS_REQUIRE_REF=1");
        }
        eprintln!("SKIP {test}: LAPACK_RS_OPENBLAS_DIR not set at build time");
        return false;
    };
    assert_eq!(config, PINNED_CONFIG, "{test}: OpenBLAS build differs from the pinned one");
    assert_eq!(core, PINNED_CORE, "{test}: OpenBLAS dispatched a different core; set OPENBLAS_CORETYPE={PINNED_CORE}");
    assert_eq!(
        std::env::var("OPENBLAS_NUM_THREADS").as_deref(),
        Ok("1"),
        "{test}: set OPENBLAS_NUM_THREADS=1 (threaded reductions are not reproducible)"
    );
    // SAFETY: argument-less setter.
    unsafe { openblas_set_num_threads(1) };
    true
}

/// Fortran `CHARACTER*1` argument.
pub fn ch(c: char) -> u8 {
    c as u8
}
