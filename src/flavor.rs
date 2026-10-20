//! BLAS flavours: which compiled BLAS the LAPACK transcriptions reproduce.
//!
//! The LAPACK control flow of `DSYEVD`/`ZHEEVD` is one transcription of
//! Reference-LAPACK 3.12.1. What differs between oracles is the BLAS under it
//! (and, for OpenBLAS's vendored LAPACK, one routine choice — see
//! [`Blas::LEGACY_DLARF`]). Every arithmetic BLAS call of that control flow
//! goes through this trait, so one transcription serves two oracles that are
//! never mixed:
//!
//! * [`RefBlas`] — the reference BLAS of Reference-LAPACK 3.12.1
//!   (`crate::blas`, `crate::zblas`); the crate's default contract.
//! * `crate::openblas::OpenBlas` — scipy's OpenBLAS 0.3.30, `SkylakeX` core,
//!   one thread (the third oracle track).
//!
//! Pure moves (`DCOPY`, `DSWAP`, `ZCOPY`, `ZSWAP`) compute nothing and are
//! identical in every flavour, so they are called directly.
#![allow(clippy::too_many_arguments)]

use crate::ptr::D;
use crate::zblas::{Tr, Z, ZV};

macro_rules! blas_trait {
    ($( fn $name:ident ( $($a:ident : $t:ty),* $(,)? ) $(-> $r:ty)? ; )*) => {
        /// The arithmetic BLAS routines `DSYEVD`/`ZHEEVD` reach, with the
        /// signatures of [`crate::blas`] / [`crate::zblas`].
        ///
        /// # Safety
        /// Every method has the contract of the reference routine of the same
        /// name: views in bounds for the shapes passed, Fortran aliasing.
        pub(crate) trait Blas: Send + Sync + 'static {
            /// `DORM2L`/`DORM2R` (`ZUNM2L`/`ZUNM2R`) apply each reflector
            /// through `DLARF` with `A(i,i)` temporarily set to one (LAPACK
            /// before 3.12.1, as vendored by OpenBLAS 0.3.30) instead of
            /// 3.12.1's `DLARF1L`/`DLARF1F`.
            const LEGACY_DLARF: bool;
            $( unsafe fn $name($($a: $t),*) $(-> $r)?; )*
        }

        /// The reference BLAS of Reference-LAPACK 3.12.1.
        #[derive(Clone, Copy, Debug, Default)]
        pub(crate) struct RefBlas;

        impl Blas for RefBlas {
            const LEGACY_DLARF: bool = false;
            $(
                #[inline(always)]
                unsafe fn $name($($a: $t),*) $(-> $r)? {
                    unsafe { blas_trait!(@path $name)($($a),*) }
                }
            )*
        }
    };
    (@path ddot) => { crate::blas::ddot };
    (@path dnrm2) => { crate::blas::dnrm2 };
    (@path daxpy) => { crate::blas::daxpy };
    (@path dscal) => { crate::blas::dscal };
    (@path drot) => { crate::blas::drot };
    (@path idamax) => { crate::blas::idamax };
    (@path dsymv) => { crate::blas::dsymv };
    (@path dsyr2) => { crate::blas::dsyr2 };
    (@path dgemv) => { crate::blas::dgemv };
    (@path dger) => { crate::blas::dger };
    (@path dsyr2k_n) => { crate::blas::dsyr2k_n };
    (@path dgemm) => { crate::blas::dgemm };
    (@path dtrmm) => { crate::blas::dtrmm };
    (@path $name:ident) => { crate::zblas::$name };
}

blas_trait! {
    fn ddot(n: usize, x: D, incx: usize, y: D, incy: usize) -> f64;
    fn dnrm2(n: usize, x: D, incx: usize) -> f64;
    fn daxpy(n: usize, da: f64, x: D, incx: usize, y: D, incy: usize);
    fn dscal(n: usize, da: f64, x: D, incx: usize);
    fn drot(n: usize, x: D, incx: usize, y: D, incy: usize, c: f64, s: f64);
    fn idamax(n: usize, x: D, incx: usize) -> usize;
    fn dsymv(upper: bool, n: usize, alpha: f64, a: D, lda: usize, x: D, beta: f64, y: D);
    fn dsyr2(upper: bool, n: usize, alpha: f64, x: D, y: D, a: D, lda: usize);
    fn dgemv(trans: bool, m: usize, n: usize, alpha: f64, a: D, lda: usize, x: D, incx: usize, beta: f64, y: D, incy: usize);
    fn dger(m: usize, n: usize, alpha: f64, x: D, incx: usize, y: D, incy: usize, a: D, lda: usize);
    fn dsyr2k_n(upper: bool, n: usize, k: usize, alpha: f64, a: D, lda: usize, b: D, ldb: usize, beta: f64, c: D, ldc: usize);
    fn dgemm(transa: bool, transb: bool, m: usize, n: usize, k: usize, alpha: f64, a: D, lda: usize, b: D, ldb: usize, beta: f64, c: D, ldc: usize);
    fn dtrmm(left: bool, upper: bool, trans: bool, nounit: bool, m: usize, n: usize, alpha: f64, a: D, lda: usize, b: D, ldb: usize);
    fn zdotc(n: usize, x: ZV, incx: usize, y: ZV, incy: usize) -> Z;
    fn dznrm2(n: usize, x: ZV, incx: usize) -> f64;
    fn zaxpy(n: usize, za: Z, x: ZV, incx: usize, y: ZV, incy: usize);
    fn zscal(n: usize, za: Z, x: ZV, incx: usize);
    fn zdscal(n: usize, da: f64, x: ZV, incx: usize);
    fn zgemv(trans: Tr, m: usize, n: usize, alpha: Z, a: ZV, lda: usize, x: ZV, incx: usize, beta: Z, y: ZV, incy: usize);
    fn zgerc(m: usize, n: usize, alpha: Z, x: ZV, incx: usize, y: ZV, incy: usize, a: ZV, lda: usize);
    fn zhemv(upper: bool, n: usize, alpha: Z, a: ZV, lda: usize, x: ZV, beta: Z, y: ZV);
    fn zher2(upper: bool, n: usize, alpha: Z, x: ZV, y: ZV, a: ZV, lda: usize);
    fn zher2k_n(upper: bool, n: usize, k: usize, alpha: Z, a: ZV, lda: usize, b: ZV, ldb: usize, beta: f64, c: ZV, ldc: usize);
    fn zgemm(transa: Tr, transb: Tr, m: usize, n: usize, k: usize, alpha: Z, a: ZV, lda: usize, b: ZV, ldb: usize, beta: Z, c: ZV, ldc: usize);
    fn ztrmm(left: bool, upper: bool, trans: Tr, nounit: bool, m: usize, n: usize, alpha: Z, a: ZV, lda: usize, b: ZV, ldb: usize);
}
