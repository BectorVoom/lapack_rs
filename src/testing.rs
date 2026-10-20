//! Internal transcriptions exposed for the component oracle tests
//! (`tests/ref_blas_bitexact.rs`, `tests/ref_lapack_components.rs`).
//!
//! Not part of the supported API. Every function is `unsafe`: the caller
//! guarantees the Fortran shape contract (array lengths, leading dimensions,
//! positive increments). Debug builds bounds-check each element access.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

pub use crate::ptr::V;

/// The OpenBLAS 0.3.30 SkylakeX BLAS transcriptions (third oracle track),
/// for `tests/openblas_*_bitexact.rs`.
pub mod openblas {
    // Every wrapper below calls FMA-compiled transcriptions; the oracle
    // tests run on the OpenBLAS SkylakeX host (AVX-512, hence FMA).
    pub use crate::openblas::level1;
    pub use crate::openblas::level2;
    pub use crate::openblas::level3;
    pub use crate::openblas::zlevel1;
    pub use crate::openblas::zlevel2;
    pub use crate::openblas::zlevel3;

    use crate::openblas::OpenBlas;

    /// `DSTEDC('I')` in the OpenBLAS flavour.
    pub unsafe fn dstedc_i(
        n: usize,
        dd: &mut [f64],
        e: &mut [f64],
        z: &mut [f64],
        work: &mut [f64],
        iwork: &mut [usize],
    ) -> usize {
        let mut eng = crate::stedc::HostDc::<OpenBlas>::new();
        unsafe {
            crate::stedc::dstedc_i_with(
                &mut eng,
                n,
                super::V::new(dd),
                super::V::new(e),
                super::V::new(z),
                n,
                super::V::new(work),
                super::V::new(iwork),
            )
        }
    }
    /// `DSYTRD` in the OpenBLAS flavour, `LWORK = work.len()`.
    pub unsafe fn dsytrd(
        upper: bool,
        n: usize,
        a: &mut [f64],
        dd: &mut [f64],
        e: &mut [f64],
        tau: &mut [f64],
        work: &mut [f64],
    ) {
        let lwork = work.len();
        let mut eng = crate::sytrd::HostTrd::<OpenBlas>::new();
        unsafe {
            crate::sytrd::dsytrd_with(
                &mut eng,
                upper,
                n,
                super::V::new(a),
                n,
                super::V::new(dd),
                super::V::new(e),
                super::V::new(tau),
                super::V::new(work),
                lwork,
            )
        }
    }
    /// `DORMTR('L', UPLO, 'N')` in the OpenBLAS flavour, `LWORK = work.len()`.
    pub unsafe fn dormtr_ln(upper: bool, n: usize, a: &mut [f64], tau: &mut [f64], c: &mut [f64], work: &mut [f64]) {
        let lwork = work.len();
        let mut eng = crate::sytrd::HostOrm::<OpenBlas>::new();
        unsafe {
            crate::sytrd::dormtr_ln_with(
                &mut eng,
                upper,
                n,
                n,
                super::V::new(a),
                n,
                super::V::new(tau),
                super::V::new(c),
                n,
                super::V::new(work),
                lwork,
            )
        }
    }

    /// `openblas::level3::dsyr2k_n` on slices.
    pub unsafe fn dsyr2k_n(
        upper: bool,
        n: usize,
        k: usize,
        alpha: f64,
        a: &mut [f64],
        lda: usize,
        b: &mut [f64],
        ldb: usize,
        beta: f64,
        c: &mut [f64],
        ldc: usize,
    ) {
        unsafe {
            crate::openblas::level3::dsyr2k_n(
                upper,
                n,
                k,
                alpha,
                super::V::new(a),
                lda,
                super::V::new(b),
                ldb,
                beta,
                super::V::new(c),
                ldc,
            )
        }
    }
    /// `openblas::level3::dgemm` on slices.
    pub unsafe fn dgemm(
        ta: bool,
        tb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &mut [f64],
        lda: usize,
        b: &mut [f64],
        ldb: usize,
        beta: f64,
        c: &mut [f64],
        ldc: usize,
    ) {
        unsafe {
            crate::openblas::level3::dgemm(
                ta,
                tb,
                m,
                n,
                k,
                alpha,
                super::V::new(a),
                lda,
                super::V::new(b),
                ldb,
                beta,
                super::V::new(c),
                ldc,
            )
        }
    }
    /// `openblas::level1::drot` on slices (unit increments).
    pub unsafe fn drot(n: usize, x: &mut [f64], y: &mut [f64], c: f64, s: f64) {
        unsafe { crate::openblas::level1::drot(n, super::V::new(x), 1, super::V::new(y), 1, c, s) }
    }
    /// `openblas::level3::dtrmm` on slices.
    pub unsafe fn dtrmm(
        left: bool,
        upper: bool,
        trans: bool,
        nounit: bool,
        m: usize,
        n: usize,
        alpha: f64,
        a: &mut [f64],
        lda: usize,
        b: &mut [f64],
        ldb: usize,
    ) {
        unsafe {
            crate::openblas::level3::dtrmm(
                left,
                upper,
                trans,
                nounit,
                m,
                n,
                alpha,
                super::V::new(a),
                lda,
                super::V::new(b),
                ldb,
            )
        }
    }
    /// Planar `ZHETRD` in the OpenBLAS flavour (`work` holds `2*LWORK`).
    pub unsafe fn zhetrd(
        upper: bool,
        n: usize,
        a_re: &mut [f64],
        a_im: &mut [f64],
        dd: &mut [f64],
        e: &mut [f64],
        tau_re: &mut [f64],
        tau_im: &mut [f64],
        work: &mut [f64],
    ) {
        let lwork = work.len() / 2;
        let (wr, wi) = work.split_at_mut(lwork);
        let z = |r: &mut [f64], i: &mut [f64]| crate::zblas::ZV::new(super::V::new(r), super::V::new(i));
        let mut eng = crate::hetrd::HostZTrd::<OpenBlas>::new();
        unsafe {
            crate::hetrd::zhetrd_with(
                &mut eng,
                upper,
                n,
                z(a_re, a_im),
                n,
                super::V::new(dd),
                super::V::new(e),
                z(tau_re, tau_im),
                z(wr, wi),
                lwork,
            )
        }
    }
    /// Planar `ZUNMTR('L', UPLO, 'N')` in the OpenBLAS flavour (`work` holds `2*LWORK`).
    pub unsafe fn zunmtr_ln(
        upper: bool,
        n: usize,
        a_re: &mut [f64],
        a_im: &mut [f64],
        tau_re: &mut [f64],
        tau_im: &mut [f64],
        c_re: &mut [f64],
        c_im: &mut [f64],
        work: &mut [f64],
    ) {
        let lwork = work.len() / 2;
        let (wr, wi) = work.split_at_mut(lwork);
        let z = |r: &mut [f64], i: &mut [f64]| crate::zblas::ZV::new(super::V::new(r), super::V::new(i));
        let mut eng = crate::hetrd::HostZOrm::<OpenBlas>::new();
        unsafe {
            crate::hetrd::zunmtr_ln_with(
                &mut eng,
                upper,
                n,
                n,
                z(a_re, a_im),
                n,
                z(tau_re, tau_im),
                z(c_re, c_im),
                n,
                z(wr, wi),
                lwork,
            )
        }
    }

    /// `openblas::zlevel2::zhemv` on planar slices.
    pub unsafe fn zhemv(
        upper: bool,
        n: usize,
        alpha: (f64, f64),
        a_re: &mut [f64],
        a_im: &mut [f64],
        lda: usize,
        x_re: &mut [f64],
        x_im: &mut [f64],
        beta: (f64, f64),
        y_re: &mut [f64],
        y_im: &mut [f64],
    ) {
        let z = |r: &mut [f64], i: &mut [f64]| crate::zblas::ZV::new(super::V::new(r), super::V::new(i));
        unsafe {
            crate::openblas::zlevel2::zhemv(
                upper,
                n,
                super::Z::new(alpha.0, alpha.1),
                z(a_re, a_im),
                lda,
                z(x_re, x_im),
                super::Z::new(beta.0, beta.1),
                z(y_re, y_im),
            )
        }
    }
    /// `openblas::zlevel3::zher2k_n` on planar slices.
    pub unsafe fn zher2k_n(
        upper: bool,
        n: usize,
        k: usize,
        alpha: (f64, f64),
        a_re: &mut [f64],
        a_im: &mut [f64],
        lda: usize,
        b_re: &mut [f64],
        b_im: &mut [f64],
        ldb: usize,
        beta: f64,
        c_re: &mut [f64],
        c_im: &mut [f64],
        ldc: usize,
    ) {
        let z = |r: &mut [f64], i: &mut [f64]| crate::zblas::ZV::new(super::V::new(r), super::V::new(i));
        unsafe {
            crate::openblas::zlevel3::zher2k_n(
                upper,
                n,
                k,
                super::Z::new(alpha.0, alpha.1),
                z(a_re, a_im),
                lda,
                z(b_re, b_im),
                ldb,
                beta,
                z(c_re, c_im),
                ldc,
            )
        }
    }
    /// `openblas::level2::dsymv` on slices.
    pub unsafe fn dsymv(
        upper: bool,
        n: usize,
        alpha: f64,
        a: &mut [f64],
        lda: usize,
        x: &mut [f64],
        beta: f64,
        y: &mut [f64],
    ) {
        unsafe {
            crate::openblas::level2::dsymv(
                upper,
                n,
                alpha,
                super::V::new(a),
                lda,
                super::V::new(x),
                beta,
                super::V::new(y),
            )
        }
    }
}
pub use crate::zblas::{Tr, Z, ZV};

/// `DLAMCH` values the transcription uses: `(cmach, value)`.
pub const DLAMCH: &[(char, f64)] = &[
    ('E', crate::aux::EPS),
    ('S', crate::aux::SAFMIN),
    ('P', crate::aux::PREC),
    ('O', crate::aux::HUGEVAL),
    ('B', crate::aux::BASE),
];

/// `DNRM2`'s (3.12.1 `dnrm2.f90`) Blue-style scaling thresholds, and
/// `DLAED4`/`DLAED6`'s `EPS`/`SMALL1` — the constants the device `K4` kernels
/// (`src/device/kernels/secular.rs`) take as launch arguments, exposed here
/// so an oracle test can build the same `SecularConsts` the host uses
/// without duplicating the formulas.
pub fn secular_eps() -> f64 {
    crate::aux::EPS
}
pub fn secular_small1() -> f64 {
    crate::laed4::dlaed6_small1()
}
pub fn secular_tsml() -> f64 {
    crate::blas::TSML
}
pub fn secular_tbig() -> f64 {
    crate::blas::TBIG
}
pub fn secular_ssml() -> f64 {
    crate::blas::SSML
}
pub fn secular_sbig() -> f64 {
    crate::blas::SBIG
}

fn d(s: &mut [f64]) -> V<f64> {
    V::new(s)
}

pub unsafe fn ddot(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize) -> f64 {
    unsafe { crate::blas::ddot(n, d(x), incx, d(y), incy) }
}
pub unsafe fn dnrm2(n: usize, x: &mut [f64], incx: usize) -> f64 {
    unsafe { crate::blas::dnrm2(n, d(x), incx) }
}
pub unsafe fn dgemv(
    trans: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: &mut [f64],
    lda: usize,
    x: &mut [f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
) {
    unsafe { crate::blas::dgemv(trans, m, n, alpha, d(a), lda, d(x), incx, beta, d(y), 1) }
}
pub unsafe fn dsymv(
    upper: bool,
    n: usize,
    alpha: f64,
    a: &mut [f64],
    lda: usize,
    x: &mut [f64],
    beta: f64,
    y: &mut [f64],
) {
    unsafe { crate::blas::dsymv(upper, n, alpha, d(a), lda, d(x), beta, d(y)) }
}
pub unsafe fn dsyr2(upper: bool, n: usize, alpha: f64, x: &mut [f64], y: &mut [f64], a: &mut [f64], lda: usize) {
    unsafe { crate::blas::dsyr2(upper, n, alpha, d(x), d(y), d(a), lda) }
}
pub unsafe fn dsyr2k_n(
    upper: bool,
    n: usize,
    k: usize,
    alpha: f64,
    a: &mut [f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    unsafe { crate::blas::dsyr2k_n(upper, n, k, alpha, d(a), lda, d(b), ldb, beta, d(c), ldc) }
}
pub unsafe fn dgemm(
    ta: bool,
    tb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &mut [f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    unsafe { crate::blas::dgemm(ta, tb, m, n, k, alpha, d(a), lda, d(b), ldb, beta, d(c), ldc) }
}
pub unsafe fn dtrmm(
    left: bool,
    upper: bool,
    trans: bool,
    nounit: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: &mut [f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
) {
    unsafe { crate::blas::dtrmm(left, upper, trans, nounit, m, n, alpha, d(a), lda, d(b), ldb) }
}
/// `ty`: `'G'`, `'L'` or `'U'`.
pub unsafe fn dlascl(ty: char, cfrom: f64, cto: f64, m: usize, n: usize, a: &mut [f64], lda: usize) {
    let t = match ty {
        'L' => crate::aux::SclType::L,
        'U' => crate::aux::SclType::U,
        _ => crate::aux::SclType::G,
    };
    unsafe { crate::aux::dlascl(t, cfrom, cto, m, n, d(a), lda) }
}
pub fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    crate::aux::dlartg(f, g)
}
pub fn dlapy2(x: f64, y: f64) -> f64 {
    crate::aux::dlapy2(x, y)
}
pub unsafe fn dlarfg(n: usize, alpha: f64, x: &mut [f64]) -> (f64, f64) {
    unsafe { crate::sytrd::dlarfg::<crate::flavor::RefBlas>(n, alpha, d(x)) }
}
pub fn powi(x: f64, n: i32) -> f64 {
    crate::aux::powidf2(x, n)
}
pub fn fmax(a: f64, b: f64) -> f64 {
    crate::aux::fmax(a, b)
}
pub fn fmin(a: f64, b: f64) -> f64 {
    crate::aux::fmin(a, b)
}
pub fn sign(a: f64, b: f64) -> f64 {
    crate::aux::sign(a, b)
}
/// `(re, im)` pairs for the complex helpers.
pub fn zladiv(x: (f64, f64), y: (f64, f64)) -> (f64, f64) {
    let z = crate::hetrd::zladiv(Z::new(x.0, x.1), Z::new(y.0, y.1));
    (z.re, z.im)
}
pub fn zdiv_gfortran(x: (f64, f64), y: (f64, f64)) -> (f64, f64) {
    let z = crate::zblas::zdiv_gfortran(Z::new(x.0, x.1), Z::new(y.0, y.1));
    (z.re, z.im)
}
pub fn zmul(x: (f64, f64), y: (f64, f64)) -> (f64, f64) {
    let z = Z::new(x.0, x.1) * Z::new(y.0, y.1);
    (z.re, z.im)
}
pub fn rzmul(r: f64, y: (f64, f64)) -> (f64, f64) {
    let z = Z::r(r) * Z::new(y.0, y.1);
    (z.re, z.im)
}
pub fn zabs(x: (f64, f64)) -> f64 {
    Z::new(x.0, x.1).abs()
}
pub unsafe fn zlarfg(n: usize, alpha: (f64, f64), x_re: &mut [f64], x_im: &mut [f64]) -> ((f64, f64), (f64, f64)) {
    let (a, t) = unsafe {
        crate::hetrd::zlarfg::<crate::flavor::RefBlas>(n, Z::new(alpha.0, alpha.1), ZV::new(d(x_re), d(x_im)))
    };
    ((a.re, a.im), (t.re, t.im))
}
pub unsafe fn dsteqr_i(n: usize, dd: &mut [f64], e: &mut [f64], z: &mut [f64], work: &mut [f64]) -> usize {
    unsafe { crate::steqr::dsteqr_raw(crate::Compz::I, n, d(dd), d(e), d(z), n, d(work)) }
}
pub unsafe fn dsterf(n: usize, dd: &mut [f64], e: &mut [f64]) -> usize {
    unsafe { crate::steqr::dsterf_raw(n, d(dd), d(e)) }
}
pub unsafe fn dlaed4(n: usize, i: usize, dd: &mut [f64], z: &mut [f64], delta: &mut [f64], rho: f64) -> (f64, usize) {
    unsafe { crate::laed4::dlaed4(n, i, d(dd), d(z), d(delta), rho) }
}
pub unsafe fn dstedc_i(
    n: usize,
    dd: &mut [f64],
    e: &mut [f64],
    z: &mut [f64],
    work: &mut [f64],
    iwork: &mut [usize],
) -> usize {
    unsafe { crate::stedc::dstedc_i(n, d(dd), d(e), d(z), n, d(work), V::new(iwork)) }
}
pub unsafe fn dsytrd(
    upper: bool,
    n: usize,
    a: &mut [f64],
    dd: &mut [f64],
    e: &mut [f64],
    tau: &mut [f64],
    work: &mut [f64],
) {
    let lwork = work.len();
    unsafe { crate::sytrd::dsytrd(upper, n, d(a), n, d(dd), d(e), d(tau), d(work), lwork) }
}
pub unsafe fn dormtr_ln(upper: bool, n: usize, a: &mut [f64], tau: &mut [f64], c: &mut [f64], work: &mut [f64]) {
    let lwork = work.len();
    unsafe { crate::sytrd::dormtr_ln(upper, n, n, d(a), n, d(tau), d(c), n, d(work), lwork) }
}
/// Planar `ZHETRD`; `work` holds `2*LWORK` reals (real parts, then imaginary).
pub unsafe fn zhetrd(
    upper: bool,
    n: usize,
    a_re: &mut [f64],
    a_im: &mut [f64],
    dd: &mut [f64],
    e: &mut [f64],
    tau_re: &mut [f64],
    tau_im: &mut [f64],
    work: &mut [f64],
) {
    let lwork = work.len() / 2;
    let (wr, wi) = work.split_at_mut(lwork);
    unsafe {
        crate::hetrd::zhetrd(
            upper,
            n,
            ZV::new(d(a_re), d(a_im)),
            n,
            d(dd),
            d(e),
            ZV::new(d(tau_re), d(tau_im)),
            ZV::new(d(wr), d(wi)),
            lwork,
        )
    }
}
/// Planar `ZUNMTR( 'L', UPLO, 'N', ... )`; `work` holds `2*LWORK` reals.
pub unsafe fn zunmtr_ln(
    upper: bool,
    n: usize,
    a_re: &mut [f64],
    a_im: &mut [f64],
    tau_re: &mut [f64],
    tau_im: &mut [f64],
    c_re: &mut [f64],
    c_im: &mut [f64],
    work: &mut [f64],
) {
    let lwork = work.len() / 2;
    let (wr, wi) = work.split_at_mut(lwork);
    unsafe {
        crate::hetrd::zunmtr_ln(
            upper,
            n,
            n,
            ZV::new(d(a_re), d(a_im)),
            n,
            ZV::new(d(tau_re), d(tau_im)),
            ZV::new(d(c_re), d(c_im)),
            n,
            ZV::new(d(wr), d(wi)),
            lwork,
        )
    }
}

// ---------------------------------------------------------------------------
// MRRR components (`src/larr.rs`, `src/lasq.rs`, `src/larre.rs`,
// `src/larrv.rs`, `src/stemr.rs`).
// ---------------------------------------------------------------------------

/// `DLAEV2( A, B, C, RT1, RT2, CS1, SN1 )`.
pub fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    crate::aux::dlaev2(a, b, c)
}

/// `DLAE2( A, B, C, RT1, RT2 )`.
pub fn dlae2(a: f64, b: f64, c: f64) -> (f64, f64) {
    crate::aux::dlae2(a, b, c)
}

/// `DLARRC`; `matt` is `LSAME( JOBT, 'T' )`.
pub unsafe fn dlarrc(matt: bool, n: usize, vl: f64, vu: f64, dd: &mut [f64], e: &mut [f64]) -> (i32, i32, i32, i32) {
    unsafe { crate::larr::dlarrc(matt, n, vl, vu, d(dd), d(e)) }
}

/// `DLARRK`. Returns `(w, werr, info)`.
pub unsafe fn dlarrk(
    n: usize,
    iw: i32,
    gl: f64,
    gu: f64,
    dd: &mut [f64],
    e2: &mut [f64],
    pivmin: f64,
    reltol: f64,
) -> (f64, f64, i32) {
    unsafe { crate::larr::dlarrk(n, iw, gl, gu, d(dd), d(e2), pivmin, reltol) }
}

/// `dlarrk.f:193`'s `ITMAX`, as an integer (the device gate compares this).
pub fn larrk_itmax(tnorm: f64, pivmin: f64) -> i32 {
    crate::larr::larrk_itmax(tnorm, pivmin)
}

/// `DLANEG`.
pub unsafe fn dlaneg(n: usize, dd: &mut [f64], lld: &mut [f64], sigma: f64, pivmin: f64, r: usize) -> i32 {
    unsafe { crate::larr::dlaneg(n, d(dd), d(lld), sigma, pivmin, r) }
}

/// `DLAR1V`. Returns
/// `(negcnt, ztz, mingma, r, isuppz1, isuppz2, nrminv, resid, rqcorr)`.
pub unsafe fn dlar1v(
    n: usize,
    b1: usize,
    bn: usize,
    lambda: f64,
    dd: &mut [f64],
    l: &mut [f64],
    ld: &mut [f64],
    lld: &mut [f64],
    pivmin: f64,
    gaptol: f64,
    z: &mut [f64],
    wantnc: bool,
    r_in: usize,
    work: &mut [f64],
) -> (i32, f64, f64, usize, usize, usize, f64, f64, f64) {
    unsafe {
        crate::larr::dlar1v(n, b1, bn, lambda, d(dd), d(l), d(ld), d(lld), pivmin, gaptol, d(z), wantnc, r_in, d(work))
    }
}

/// `DLASQ2( N, Z, INFO )`; `ieee` is `ILAENV( 10, 'DLASQ2', ... ) == 1`.
pub unsafe fn dlasq2(n: usize, z: &mut [f64], ieee: bool) -> i32 {
    unsafe { crate::lasq::dlasq2(n, d(z), ieee) }
}

/// `DLASQ6( I0, N0, Z, PP, .. )`. Returns
/// `(dmin, dmin1, dmin2, dn, dn1, dn2)`.
pub unsafe fn dlasq6(i0: usize, n0: usize, z: &mut [f64], pp: usize) -> (f64, f64, f64, f64, f64, f64) {
    let mut s = crate::lasq::Dq::default();
    unsafe { crate::lasq::dlasq6(i0, n0, d(z), pp, &mut s) };
    (s.dmin, s.dmin1, s.dmin2, s.dn, s.dn1, s.dn2)
}

/// `DLASQ5( I0, N0, Z, PP, TAU, SIGMA, .., IEEE, EPS )`. Returns
/// `(tau, dmin, dmin1, dmin2, dn, dn1, dn2)`.
pub unsafe fn dlasq5(
    i0: usize,
    n0: usize,
    z: &mut [f64],
    pp: usize,
    tau: f64,
    sigma: f64,
    ieee: bool,
    eps: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let mut s = crate::lasq::Dq { tau, sigma, ..crate::lasq::Dq::default() };
    unsafe { crate::lasq::dlasq5(i0, n0, d(z), pp, &mut s, ieee, eps) };
    (s.tau, s.dmin, s.dmin1, s.dmin2, s.dn, s.dn1, s.dn2)
}

/// `DLASQ4( I0, N0, Z, PP, N0IN, .. , TAU, TTYPE, G )`. The tuple is the
/// reference's `(DMIN, DMIN1, DMIN2, DN, DN1, DN2, TAU, G, TTYPE)` in and out.
pub unsafe fn dlasq4(
    i0: usize,
    n0: usize,
    z: &mut [f64],
    pp: usize,
    n0in: usize,
    state: (f64, f64, f64, f64, f64, f64, f64, f64, i32),
) -> (f64, f64, i32) {
    let mut s = crate::lasq::Dq {
        dmin: state.0,
        dmin1: state.1,
        dmin2: state.2,
        dn: state.3,
        dn1: state.4,
        dn2: state.5,
        tau: state.6,
        g: state.7,
        ttype: state.8,
        ..crate::lasq::Dq::default()
    };
    unsafe { crate::lasq::dlasq4(i0, n0, d(z), pp, n0in, &mut s) };
    (s.tau, s.g, s.ttype)
}

fn j(s: &mut [i32]) -> V<i32> {
    V::new(s)
}

/// `DLARUV( ISEED, N, X )`.
pub unsafe fn dlaruv(iseed: &mut [i32; 4], n: usize, x: &mut [f64]) {
    unsafe { crate::larr2::dlaruv(iseed, n, d(x)) }
}

/// `DLARNV( IDIST, ISEED, N, X )`.
pub unsafe fn dlarnv(idist: i32, iseed: &mut [i32; 4], n: usize, x: &mut [f64]) {
    unsafe { crate::larr2::dlarnv(idist, iseed, n, d(x)) }
}

/// `DLARRA( .. )`. Returns `(nsplit, info)`.
pub unsafe fn dlarra(
    n: usize,
    dd: &mut [f64],
    e: &mut [f64],
    e2: &mut [f64],
    spltol: f64,
    tnrm: f64,
    isplit: &mut [i32],
) -> (i32, i32) {
    unsafe { crate::larr2::dlarra(n, d(dd), d(e), d(e2), spltol, tnrm, j(isplit)) }
}

/// `DLARRR( N, D, E, INFO )`.
pub unsafe fn dlarrr(n: usize, dd: &mut [f64], e: &mut [f64]) -> i32 {
    unsafe { crate::larr2::dlarrr(n, d(dd), d(e)) }
}

/// `DLARRB( .. )`.
pub unsafe fn dlarrb(
    n: usize,
    dd: &mut [f64],
    lld: &mut [f64],
    ifirst: usize,
    ilast: usize,
    rtol1: f64,
    rtol2: f64,
    offset: usize,
    w: &mut [f64],
    wgap: &mut [f64],
    werr: &mut [f64],
    work: &mut [f64],
    iwork: &mut [i32],
    pivmin: f64,
    spdiam: f64,
    twist: i32,
) -> i32 {
    unsafe {
        crate::larr2::dlarrb(
            n,
            d(dd),
            d(lld),
            ifirst,
            ilast,
            rtol1,
            rtol2,
            offset,
            d(w),
            d(wgap),
            d(werr),
            d(work),
            j(iwork),
            pivmin,
            spdiam,
            twist,
        )
    }
}

/// `DLARRJ( .. )`.
pub unsafe fn dlarrj(
    n: usize,
    dd: &mut [f64],
    e2: &mut [f64],
    ifirst: usize,
    ilast: usize,
    rtol: f64,
    offset: usize,
    w: &mut [f64],
    werr: &mut [f64],
    work: &mut [f64],
    iwork: &mut [i32],
    pivmin: f64,
    spdiam: f64,
) -> i32 {
    unsafe {
        crate::larr2::dlarrj(
            n,
            d(dd),
            d(e2),
            ifirst,
            ilast,
            rtol,
            offset,
            d(w),
            d(werr),
            d(work),
            j(iwork),
            pivmin,
            spdiam,
        )
    }
}

/// `DLARRF( .. )`. Returns `(sigma, info)`.
pub unsafe fn dlarrf(
    n: usize,
    dd: &mut [f64],
    l: &mut [f64],
    ld: &mut [f64],
    clstrt: usize,
    clend: usize,
    w: &mut [f64],
    wgap: &mut [f64],
    werr: &mut [f64],
    spdiam: f64,
    clgapl: f64,
    clgapr: f64,
    pivmin: f64,
    dplus: &mut [f64],
    lplus: &mut [f64],
    work: &mut [f64],
) -> (f64, i32) {
    unsafe {
        crate::larr2::dlarrf(
            n,
            d(dd),
            d(l),
            d(ld),
            clstrt,
            clend,
            d(w),
            d(wgap),
            d(werr),
            spdiam,
            clgapl,
            clgapr,
            pivmin,
            d(dplus),
            d(lplus),
            d(work),
        )
    }
}

/// `DLARRE( .. )` with `RANGE = 'A'`. Returns
/// `(vl, vu, nsplit, m, pivmin, info)`.
pub unsafe fn dlarre_all(
    n: usize,
    vl: f64,
    vu: f64,
    dd: &mut [f64],
    e: &mut [f64],
    e2: &mut [f64],
    rtol1: f64,
    rtol2: f64,
    spltol: f64,
    isplit: &mut [i32],
    w: &mut [f64],
    werr: &mut [f64],
    wgap: &mut [f64],
    iblock: &mut [i32],
    indexw: &mut [i32],
    gers: &mut [f64],
    work: &mut [f64],
    iwork: &mut [i32],
) -> (f64, f64, i32, i32, f64, i32) {
    unsafe {
        crate::larre::dlarre(
            crate::larre::Range::All,
            n,
            vl,
            vu,
            1,
            n,
            d(dd),
            d(e),
            d(e2),
            rtol1,
            rtol2,
            spltol,
            j(isplit),
            d(w),
            d(werr),
            d(wgap),
            j(iblock),
            j(indexw),
            d(gers),
            d(work),
            j(iwork),
        )
    }
}

// ---------------------------------------------------------------------------
// Path C leaves (`src/cint_stemr.rs`). The oracle for these is **libcint
// 6.1.3**, not Reference-LAPACK; see the module header there.
// ---------------------------------------------------------------------------

/// libcint `_dlarrk`. Returns `(w, werr, info)`.
pub unsafe fn cint_dlarrk(
    n: usize,
    iw: i32,
    gl: f64,
    gu: f64,
    dd: &mut [f64],
    e2: &mut [f64],
    reltol: f64,
) -> (f64, f64, i32) {
    unsafe { crate::cint_stemr::testing::dlarrk(n, iw, gl, gu, d(dd), d(e2), reltol) }
}

/// libcint `_dlarrc`. Returns `(lcnt, rcnt)`.
pub unsafe fn cint_dlarrc(n: usize, vl: f64, vu: f64, dd: &mut [f64], e2: &mut [f64]) -> (i32, i32) {
    unsafe { crate::cint_stemr::testing::dlarrc(n, vl, vu, d(dd), d(e2)) }
}

/// libcint `_dlaneg`.
pub unsafe fn cint_dlaneg(n: usize, dd: &mut [f64], lld: &mut [f64], sigma: f64, twist: i32) -> i32 {
    unsafe { crate::cint_stemr::testing::dlaneg(n, d(dd), d(lld), sigma, twist) }
}

/// libcint `_dlarrb`; `w`, `wgap` (with `wgap[-1]` at `wgap_m1[0]`) and
/// `werr` are updated in place.
pub unsafe fn cint_dlarrb(
    n: usize,
    dd: &mut [f64],
    lld: &mut [f64],
    ifirst: usize,
    ilast: usize,
    rtol1: f64,
    rtol2: f64,
    w: &mut [f64],
    wgap_m1: &mut [f64],
    werr: &mut [f64],
    twist: i32,
) {
    unsafe {
        crate::cint_stemr::testing::dlarrb(
            n,
            d(dd),
            d(lld),
            ifirst,
            ilast,
            rtol1,
            rtol2,
            d(w),
            d(wgap_m1),
            d(werr),
            twist,
        )
    }
}

/// libcint `_dlarrf`. Returns `(sigma, info)`; `dplus`/`lplus` are written.
pub unsafe fn cint_dlarrf(
    n: usize,
    dd: &mut [f64],
    offd: &mut [f64],
    ld: &mut [f64],
    clstrt: usize,
    w: &mut [f64],
    wgap_m1: &mut [f64],
    werr: &mut [f64],
    clgapl: f64,
    dplus: &mut [f64],
    lplus: &mut [f64],
) -> (f64, i32) {
    unsafe {
        crate::cint_stemr::testing::dlarrf(
            n,
            d(dd),
            d(offd),
            d(ld),
            clstrt,
            d(w),
            d(wgap_m1),
            d(werr),
            clgapl,
            d(dplus),
            d(lplus),
        )
    }
}

/// libcint `_dlar1v`. Returns `(negcnt, twist, resid, rqcorr)`; `vec` and
/// `work` (`3 * n`) are written.
pub unsafe fn cint_dlar1v(
    n: usize,
    lambda: f64,
    dd: &mut [f64],
    offd: &mut [f64],
    ld: &mut [f64],
    lld: &mut [f64],
    vec: &mut [f64],
    twist_in: i32,
    work: &mut [f64],
) -> (i32, i32, f64, f64) {
    unsafe { crate::cint_stemr::testing::dlar1v(n, lambda, d(dd), d(offd), d(ld), d(lld), d(vec), twist_in, d(work)) }
}

/// libcint `_dlaev2`; writes `eig[0..2]` and `vec[0..4]`.
pub unsafe fn cint_dlaev2(eig: &mut [f64], vec: &mut [f64], dd: &mut [f64], offd: &mut [f64]) -> i32 {
    unsafe { crate::cint_stemr::testing::dlaev2(d(eig), d(vec), d(dd), d(offd)) }
}

/// libcint `_dlasq4`. `state` is `(dmin[0..3], dn[0..3], tau)`; returns the
/// updated `tau`.
pub unsafe fn cint_dlasq4(
    work: &mut [f64],
    qp: usize,
    q1p: usize,
    ep: usize,
    e1p: usize,
    i0: usize,
    n0: usize,
    n0init: usize,
    dmin: [f64; 3],
    dn: [f64; 3],
    tau_in: f64,
) -> f64 {
    unsafe { crate::cint_stemr::testing::dlasq4(d(work), qp, q1p, ep, e1p, i0, n0, n0init, dmin, dn, tau_in) }
}

/// libcint `_dlasq5`. Returns `(dmin[0..3], dn[0..3])`.
pub unsafe fn cint_dlasq5(
    work: &mut [f64],
    qp: usize,
    q1p: usize,
    ep: usize,
    e1p: usize,
    i0: usize,
    n0: usize,
    tau: f64,
    tol: f64,
) -> ([f64; 3], [f64; 3]) {
    unsafe { crate::cint_stemr::testing::dlasq5(d(work), qp, q1p, ep, e1p, i0, n0, tau, tol) }
}
