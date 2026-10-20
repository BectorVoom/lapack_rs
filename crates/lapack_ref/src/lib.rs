//! Test-only FFI to the bit-exact oracle: Reference-LAPACK 3.12.1 plus the
//! reference BLAS of the same tree, built by `cargo xtask build-ref-lapack`.
//!
//! gfortran ABI: every argument by reference, `INTEGER` = `i32`, hidden
//! `CHARACTER` lengths (`usize`) appended after the explicit arguments,
//! `COMPLEX*16` function results returned like C `double _Complex`.
//!
//! The safe wrappers take column-major slices with `LDA = n` and `i32`
//! `IWORK`, and return `INFO`.

#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

/// Whether the oracle archives were linked (`LAPACK_RS_REF_DIR` was set at
/// build time).
pub const AVAILABLE: bool = cfg!(lapack_ref);

/// Path of `ref-lapack.toml` (only when [`AVAILABLE`]).
pub fn manifest_path() -> Option<&'static str> {
    option_env!("LAPACK_RS_REF_MANIFEST")
}

/// Standard gate for oracle tests: `true` when the oracle can run. Without it
/// the test prints `SKIP`, unless `LAPACK_RS_REQUIRE_REF=1`, which panics so a
/// gate run can never pass vacuously.
pub fn require(test: &str) -> bool {
    if AVAILABLE {
        return true;
    }
    if std::env::var("LAPACK_RS_REQUIRE_REF").as_deref() == Ok("1") {
        panic!("{test}: oracle not linked (set LAPACK_RS_REF_DIR at build time) but LAPACK_RS_REQUIRE_REF=1");
    }
    eprintln!("SKIP {test}: LAPACK_RS_REF_DIR not set at build time");
    false
}

/// `COMPLEX*16` as returned by value.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

#[cfg(lapack_ref_extras)]
mod ffi_extras {
    use super::C64;
    unsafe extern "C" {
        pub fn xzdiv_(a: *const C64, b: *const C64, c: *mut C64);
        pub fn xzmul_(a: *const C64, b: *const C64, c: *mut C64);
        pub fn xrzmul_(r: *const f64, a: *const C64, c: *mut C64);
        pub fn xpowi_(x: *const f64, n: *const i32, y: *mut f64);
        pub fn xcabs_(a: *const C64, r: *mut f64);
        pub fn xmax_(a: *const f64, b: *const f64, r: *mut f64);
        pub fn xmin_(a: *const f64, b: *const f64, r: *mut f64);
        pub fn xsign_(a: *const f64, b: *const f64, r: *mut f64);
    }
}

/// gfortran lowering probes (`liblapackrs_extras.a`), compiled with the
/// oracle's flags. `None` when the probes were not built.
pub mod probes {
    use super::C64;
    pub const AVAILABLE: bool = cfg!(lapack_ref_extras);
    macro_rules! probe {
        ($name:ident, $sym:ident, ($($a:ident: $t:ty),*) -> $r:ty, $init:expr) => {
            pub fn $name($($a: $t),*) -> Option<$r> {
                #[cfg(lapack_ref_extras)]
                {
                    let mut out: $r = $init;
                    // SAFETY: scalars by reference.
                    unsafe { super::ffi_extras::$sym($(&$a),*, &mut out) };
                    Some(out)
                }
                #[cfg(not(lapack_ref_extras))]
                {
                    let _ = ($($a),*);
                    None
                }
            }
        };
    }
    probe!(zdiv, xzdiv_, (a: C64, b: C64) -> C64, C64::default());
    probe!(zmul, xzmul_, (a: C64, b: C64) -> C64, C64::default());
    probe!(rzmul, xrzmul_, (r: f64, a: C64) -> C64, C64::default());
    probe!(powi, xpowi_, (x: f64, n: i32) -> f64, 0.0);
    probe!(cabs, xcabs_, (a: C64) -> f64, 0.0);
    probe!(max, xmax_, (a: f64, b: f64) -> f64, 0.0);
    probe!(min, xmin_, (a: f64, b: f64) -> f64, 0.0);
    probe!(sign, xsign_, (a: f64, b: f64) -> f64, 0.0);
}

#[cfg(lapack_ref)]
mod ffi {
    use super::C64;
    unsafe extern "C" {
        pub fn dsyevd_(
            jobz: *const u8,
            uplo: *const u8,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            w: *mut f64,
            work: *mut f64,
            lwork: *const i32,
            iwork: *mut i32,
            liwork: *const i32,
            info: *mut i32,
            l1: usize,
            l2: usize,
        );
        pub fn zheevd_(
            jobz: *const u8,
            uplo: *const u8,
            n: *const i32,
            a: *mut C64,
            lda: *const i32,
            w: *mut f64,
            work: *mut C64,
            lwork: *const i32,
            rwork: *mut f64,
            lrwork: *const i32,
            iwork: *mut i32,
            liwork: *const i32,
            info: *mut i32,
            l1: usize,
            l2: usize,
        );
        pub fn dsytrd_(
            uplo: *const u8,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            d: *mut f64,
            e: *mut f64,
            tau: *mut f64,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32,
            l1: usize,
        );
        pub fn zhetrd_(
            uplo: *const u8,
            n: *const i32,
            a: *mut C64,
            lda: *const i32,
            d: *mut f64,
            e: *mut f64,
            tau: *mut C64,
            work: *mut C64,
            lwork: *const i32,
            info: *mut i32,
            l1: usize,
        );
        pub fn dormtr_(
            side: *const u8,
            uplo: *const u8,
            trans: *const u8,
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            tau: *mut f64,
            c: *mut f64,
            ldc: *const i32,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32,
            l1: usize,
            l2: usize,
            l3: usize,
        );
        pub fn zunmtr_(
            side: *const u8,
            uplo: *const u8,
            trans: *const u8,
            m: *const i32,
            n: *const i32,
            a: *mut C64,
            lda: *const i32,
            tau: *mut C64,
            c: *mut C64,
            ldc: *const i32,
            work: *mut C64,
            lwork: *const i32,
            info: *mut i32,
            l1: usize,
            l2: usize,
            l3: usize,
        );
        pub fn dstedc_(
            compz: *const u8,
            n: *const i32,
            d: *mut f64,
            e: *mut f64,
            z: *mut f64,
            ldz: *const i32,
            work: *mut f64,
            lwork: *const i32,
            iwork: *mut i32,
            liwork: *const i32,
            info: *mut i32,
            l1: usize,
        );
        pub fn dsteqr_(
            compz: *const u8,
            n: *const i32,
            d: *mut f64,
            e: *mut f64,
            z: *mut f64,
            ldz: *const i32,
            work: *mut f64,
            info: *mut i32,
            l1: usize,
        );
        pub fn dsterf_(n: *const i32, d: *mut f64, e: *mut f64, info: *mut i32);
        pub fn dlaed4_(
            n: *const i32,
            i: *const i32,
            d: *const f64,
            z: *const f64,
            delta: *mut f64,
            rho: *const f64,
            dlam: *mut f64,
            info: *mut i32,
        );
        pub fn dlarfg_(n: *const i32, alpha: *mut f64, x: *mut f64, incx: *const i32, tau: *mut f64);
        pub fn dnrm2_(n: *const i32, x: *const f64, incx: *const i32) -> f64;
        pub fn dlartg_(f: *const f64, g: *const f64, c: *mut f64, s: *mut f64, r: *mut f64);
        pub fn dlapy2_(x: *const f64, y: *const f64) -> f64;
        pub fn dgemm_(
            ta: *const u8,
            tb: *const u8,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            alpha: *const f64,
            a: *const f64,
            lda: *const i32,
            b: *const f64,
            ldb: *const i32,
            beta: *const f64,
            c: *mut f64,
            ldc: *const i32,
            l1: usize,
            l2: usize,
        );
        pub fn zladiv_(x: *const C64, y: *const C64) -> C64;
        pub fn zlarfg_(n: *const i32, alpha: *mut C64, x: *mut C64, incx: *const i32, tau: *mut C64);
        pub fn ddot_(n: *const i32, x: *const f64, incx: *const i32, y: *const f64, incy: *const i32) -> f64;
        pub fn dgemv_(
            t: *const u8,
            m: *const i32,
            n: *const i32,
            alpha: *const f64,
            a: *const f64,
            lda: *const i32,
            x: *const f64,
            incx: *const i32,
            beta: *const f64,
            y: *mut f64,
            incy: *const i32,
            l1: usize,
        );
        pub fn dsymv_(
            u: *const u8,
            n: *const i32,
            alpha: *const f64,
            a: *const f64,
            lda: *const i32,
            x: *const f64,
            incx: *const i32,
            beta: *const f64,
            y: *mut f64,
            incy: *const i32,
            l1: usize,
        );
        pub fn dsyr2_(
            u: *const u8,
            n: *const i32,
            alpha: *const f64,
            x: *const f64,
            incx: *const i32,
            y: *const f64,
            incy: *const i32,
            a: *mut f64,
            lda: *const i32,
            l1: usize,
        );
        pub fn dsyr2k_(
            u: *const u8,
            t: *const u8,
            n: *const i32,
            k: *const i32,
            alpha: *const f64,
            a: *const f64,
            lda: *const i32,
            b: *const f64,
            ldb: *const i32,
            beta: *const f64,
            c: *mut f64,
            ldc: *const i32,
            l1: usize,
            l2: usize,
        );
        pub fn dtrmm_(
            side: *const u8,
            uplo: *const u8,
            ta: *const u8,
            diag: *const u8,
            m: *const i32,
            n: *const i32,
            alpha: *const f64,
            a: *const f64,
            lda: *const i32,
            b: *mut f64,
            ldb: *const i32,
            l1: usize,
            l2: usize,
            l3: usize,
            l4: usize,
        );
        pub fn dlascl_(
            ty: *const u8,
            kl: *const i32,
            ku: *const i32,
            cfrom: *const f64,
            cto: *const f64,
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            info: *mut i32,
            l1: usize,
        );
        pub fn dlamch_(c: *const u8, l1: usize) -> f64;
        pub fn dlaev2_(
            a: *const f64,
            b: *const f64,
            c: *const f64,
            rt1: *mut f64,
            rt2: *mut f64,
            cs1: *mut f64,
            sn1: *mut f64,
        );
        pub fn dlae2_(a: *const f64, b: *const f64, c: *const f64, rt1: *mut f64, rt2: *mut f64);
        pub fn zgemv_(
            trans: *const u8,
            m: *const i32,
            n: *const i32,
            alpha: *const C64,
            a: *const C64,
            lda: *const i32,
            x: *const C64,
            incx: *const i32,
            beta: *const C64,
            y: *mut C64,
            incy: *const i32,
            l1: usize,
        );
        // ---- MRRR (DSTEMR) component closure -------------------------------
        pub fn dlarrc_(
            jobt: *const u8,
            n: *const i32,
            vl: *const f64,
            vu: *const f64,
            d: *const f64,
            e: *const f64,
            pivmin: *const f64,
            eigcnt: *mut i32,
            lcnt: *mut i32,
            rcnt: *mut i32,
            info: *mut i32,
            l1: usize,
        );
        pub fn dlarrk_(
            n: *const i32,
            iw: *const i32,
            gl: *const f64,
            gu: *const f64,
            d: *const f64,
            e2: *const f64,
            pivmin: *const f64,
            reltol: *const f64,
            w: *mut f64,
            werr: *mut f64,
            info: *mut i32,
        );
        pub fn dlaneg_(
            n: *const i32,
            d: *const f64,
            lld: *const f64,
            sigma: *const f64,
            pivmin: *const f64,
            r: *const i32,
        ) -> i32;
        pub fn dlar1v_(
            n: *const i32,
            b1: *const i32,
            bn: *const i32,
            lambda: *const f64,
            d: *const f64,
            l: *const f64,
            ld: *const f64,
            lld: *const f64,
            pivmin: *const f64,
            gaptol: *const f64,
            z: *mut f64,
            wantnc: *const i32,
            negcnt: *mut i32,
            ztz: *mut f64,
            mingma: *mut f64,
            r: *mut i32,
            isuppz: *mut i32,
            nrminv: *mut f64,
            resid: *mut f64,
            rqcorr: *mut f64,
            work: *mut f64,
        );
        pub fn dlasq2_(n: *const i32, z: *mut f64, info: *mut i32);
        pub fn dlarra_(
            n: *const i32,
            d: *const f64,
            e: *mut f64,
            e2: *mut f64,
            spltol: *const f64,
            tnrm: *const f64,
            nsplit: *mut i32,
            isplit: *mut i32,
            info: *mut i32,
        );
        pub fn dlarrr_(n: *const i32, d: *const f64, e: *mut f64, info: *mut i32);
        pub fn dlarrb_(
            n: *const i32,
            d: *const f64,
            lld: *const f64,
            ifirst: *const i32,
            ilast: *const i32,
            rtol1: *const f64,
            rtol2: *const f64,
            offset: *const i32,
            w: *mut f64,
            wgap: *mut f64,
            werr: *mut f64,
            work: *mut f64,
            iwork: *mut i32,
            pivmin: *const f64,
            spdiam: *const f64,
            twist: *const i32,
            info: *mut i32,
        );
        pub fn dlarrf_(
            n: *const i32,
            d: *const f64,
            l: *const f64,
            ld: *const f64,
            clstrt: *const i32,
            clend: *const i32,
            w: *const f64,
            wgap: *mut f64,
            werr: *const f64,
            spdiam: *const f64,
            clgapl: *const f64,
            clgapr: *const f64,
            pivmin: *const f64,
            sigma: *mut f64,
            dplus: *mut f64,
            lplus: *mut f64,
            work: *mut f64,
            info: *mut i32,
        );
        pub fn dlarrj_(
            n: *const i32,
            d: *const f64,
            e2: *const f64,
            ifirst: *const i32,
            ilast: *const i32,
            rtol: *const f64,
            offset: *const i32,
            w: *mut f64,
            werr: *mut f64,
            work: *mut f64,
            iwork: *mut i32,
            pivmin: *const f64,
            spdiam: *const f64,
            info: *mut i32,
        );
        pub fn dlarre_(
            range: *const u8,
            n: *const i32,
            vl: *mut f64,
            vu: *mut f64,
            il: *const i32,
            iu: *const i32,
            d: *mut f64,
            e: *mut f64,
            e2: *mut f64,
            rtol1: *const f64,
            rtol2: *const f64,
            spltol: *const f64,
            nsplit: *mut i32,
            isplit: *mut i32,
            m: *mut i32,
            w: *mut f64,
            werr: *mut f64,
            wgap: *mut f64,
            iblock: *mut i32,
            indexw: *mut i32,
            gers: *mut f64,
            pivmin: *mut f64,
            work: *mut f64,
            iwork: *mut i32,
            info: *mut i32,
            l1: usize,
        );
        pub fn dlarrv_(
            n: *const i32,
            vl: *const f64,
            vu: *const f64,
            d: *mut f64,
            l: *mut f64,
            pivmin: *const f64,
            isplit: *const i32,
            m: *const i32,
            dol: *const i32,
            dou: *const i32,
            minrgp: *const f64,
            rtol1: *const f64,
            rtol2: *const f64,
            w: *mut f64,
            werr: *mut f64,
            wgap: *mut f64,
            iblock: *const i32,
            indexw: *const i32,
            gers: *const f64,
            z: *mut f64,
            ldz: *const i32,
            isuppz: *mut i32,
            work: *mut f64,
            iwork: *mut i32,
            info: *mut i32,
        );
        pub fn dstemr_(
            jobz: *const u8,
            range: *const u8,
            n: *const i32,
            d: *mut f64,
            e: *mut f64,
            vl: *const f64,
            vu: *const f64,
            il: *const i32,
            iu: *const i32,
            m: *mut i32,
            w: *mut f64,
            z: *mut f64,
            ldz: *const i32,
            nzc: *const i32,
            isuppz: *mut i32,
            tryrac: *mut i32,
            work: *mut f64,
            lwork: *const i32,
            iwork: *mut i32,
            liwork: *const i32,
            info: *mut i32,
            l1: usize,
            l2: usize,
        );
        pub fn dlarnv_(idist: *const i32, iseed: *mut i32, n: *const i32, x: *mut f64);
        pub fn dlasq6_(
            i0: *const i32,
            n0: *const i32,
            z: *mut f64,
            pp: *const i32,
            dmin: *mut f64,
            dmin1: *mut f64,
            dmin2: *mut f64,
            dn: *mut f64,
            dnm1: *mut f64,
            dnm2: *mut f64,
        );
        pub fn dlasq5_(
            i0: *const i32,
            n0: *const i32,
            z: *mut f64,
            pp: *const i32,
            tau: *mut f64,
            sigma: *const f64,
            dmin: *mut f64,
            dmin1: *mut f64,
            dmin2: *mut f64,
            dn: *mut f64,
            dnm1: *mut f64,
            dnm2: *mut f64,
            ieee: *const i32,
            eps: *const f64,
        );
        pub fn dlasq4_(
            i0: *const i32,
            n0: *const i32,
            z: *const f64,
            pp: *const i32,
            n0in: *const i32,
            dmin: *const f64,
            dmin1: *const f64,
            dmin2: *const f64,
            dn: *const f64,
            dn1: *const f64,
            dn2: *const f64,
            tau: *mut f64,
            ttype: *mut i32,
            g: *mut f64,
        );
        pub fn ilaenv_(
            ispec: *const i32,
            name: *const u8,
            opts: *const u8,
            n1: *const i32,
            n2: *const i32,
            n3: *const i32,
            n4: *const i32,
            l1: usize,
            l2: usize,
        ) -> i32;
    }
}

#[allow(unused_macros)]
macro_rules! unavailable {
    () => {
        panic!("lapack_ref oracle not linked; set LAPACK_RS_REF_DIR at build time")
    };
}

fn ch(c: char) -> u8 {
    c as u8
}

/// `DSYEVD`. `lwork`/`liwork` are the lengths of `work`/`iwork`.
pub fn dsyevd(
    jobz: char,
    uplo: char,
    n: usize,
    a: &mut [f64],
    w: &mut [f64],
    work: &mut [f64],
    iwork: &mut [i32],
) -> i32 {
    #[cfg(lapack_ref)]
    {
        let (jz, ul) = (ch(jobz), ch(uplo));
        let (n_, lda) = (n as i32, n.max(1) as i32);
        let (lw, liw) = (work.len() as i32, iwork.len() as i32);
        let mut info = 0;
        // SAFETY: lengths are the caller's; the reference reads/writes within LWORK/LIWORK.
        unsafe {
            ffi::dsyevd_(
                &jz,
                &ul,
                &n_,
                a.as_mut_ptr(),
                &lda,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lw,
                iwork.as_mut_ptr(),
                &liw,
                &mut info,
                1,
                1,
            )
        };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (jobz, uplo, n, a, w, work, iwork, ch('x'));
        unavailable!()
    }
}

/// `ZHEEVD` on interleaved `COMPLEX*16` storage.
pub fn zheevd(
    jobz: char,
    uplo: char,
    n: usize,
    a: &mut [C64],
    w: &mut [f64],
    work: &mut [C64],
    rwork: &mut [f64],
    iwork: &mut [i32],
) -> i32 {
    #[cfg(lapack_ref)]
    {
        let (jz, ul) = (ch(jobz), ch(uplo));
        let (n_, lda) = (n as i32, n.max(1) as i32);
        let (lw, lrw, liw) = (work.len() as i32, rwork.len() as i32, iwork.len() as i32);
        let mut info = 0;
        // SAFETY: as for dsyevd.
        unsafe {
            ffi::zheevd_(
                &jz,
                &ul,
                &n_,
                a.as_mut_ptr(),
                &lda,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lw,
                rwork.as_mut_ptr(),
                &lrw,
                iwork.as_mut_ptr(),
                &liw,
                &mut info,
                1,
                1,
            )
        };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (jobz, uplo, n, a, w, work, rwork, iwork);
        unavailable!()
    }
}

/// `DSYTRD`.
pub fn dsytrd(
    uplo: char,
    n: usize,
    a: &mut [f64],
    d: &mut [f64],
    e: &mut [f64],
    tau: &mut [f64],
    work: &mut [f64],
) -> i32 {
    #[cfg(lapack_ref)]
    {
        let ul = ch(uplo);
        let (n_, lda, lw) = (n as i32, n.max(1) as i32, work.len() as i32);
        let mut info = 0;
        // SAFETY: caller-sized buffers.
        unsafe {
            ffi::dsytrd_(
                &ul,
                &n_,
                a.as_mut_ptr(),
                &lda,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                tau.as_mut_ptr(),
                work.as_mut_ptr(),
                &lw,
                &mut info,
                1,
            )
        };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (uplo, n, a, d, e, tau, work);
        unavailable!()
    }
}

/// `ZHETRD` on interleaved storage.
pub fn zhetrd(
    uplo: char,
    n: usize,
    a: &mut [C64],
    d: &mut [f64],
    e: &mut [f64],
    tau: &mut [C64],
    work: &mut [C64],
) -> i32 {
    #[cfg(lapack_ref)]
    {
        let ul = ch(uplo);
        let (n_, lda, lw) = (n as i32, n.max(1) as i32, work.len() as i32);
        let mut info = 0;
        // SAFETY: caller-sized buffers.
        unsafe {
            ffi::zhetrd_(
                &ul,
                &n_,
                a.as_mut_ptr(),
                &lda,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                tau.as_mut_ptr(),
                work.as_mut_ptr(),
                &lw,
                &mut info,
                1,
            )
        };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (uplo, n, a, d, e, tau, work);
        unavailable!()
    }
}

/// `DORMTR( 'L', UPLO, 'N', N, N, A, N, TAU, C, N, WORK, LWORK )`.
pub fn dormtr_ln(uplo: char, n: usize, a: &mut [f64], tau: &mut [f64], c: &mut [f64], work: &mut [f64]) -> i32 {
    #[cfg(lapack_ref)]
    {
        let (s, ul, t) = (b'L', ch(uplo), b'N');
        let (n_, lda, lw) = (n as i32, n.max(1) as i32, work.len() as i32);
        let mut info = 0;
        // SAFETY: caller-sized buffers.
        unsafe {
            ffi::dormtr_(
                &s,
                &ul,
                &t,
                &n_,
                &n_,
                a.as_mut_ptr(),
                &lda,
                tau.as_mut_ptr(),
                c.as_mut_ptr(),
                &lda,
                work.as_mut_ptr(),
                &lw,
                &mut info,
                1,
                1,
                1,
            )
        };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (uplo, n, a, tau, c, work);
        unavailable!()
    }
}

/// `ZUNMTR( 'L', UPLO, 'N', N, N, A, N, TAU, C, N, WORK, LWORK )`.
pub fn zunmtr_ln(uplo: char, n: usize, a: &mut [C64], tau: &mut [C64], c: &mut [C64], work: &mut [C64]) -> i32 {
    #[cfg(lapack_ref)]
    {
        let (s, ul, t) = (b'L', ch(uplo), b'N');
        let (n_, lda, lw) = (n as i32, n.max(1) as i32, work.len() as i32);
        let mut info = 0;
        // SAFETY: caller-sized buffers.
        unsafe {
            ffi::zunmtr_(
                &s,
                &ul,
                &t,
                &n_,
                &n_,
                a.as_mut_ptr(),
                &lda,
                tau.as_mut_ptr(),
                c.as_mut_ptr(),
                &lda,
                work.as_mut_ptr(),
                &lw,
                &mut info,
                1,
                1,
                1,
            )
        };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (uplo, n, a, tau, c, work);
        unavailable!()
    }
}

/// `DSTEDC( COMPZ, N, D, E, Z, N, WORK, LWORK, IWORK, LIWORK )`.
pub fn dstedc(
    compz: char,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    z: &mut [f64],
    work: &mut [f64],
    iwork: &mut [i32],
) -> i32 {
    #[cfg(lapack_ref)]
    {
        let cz = ch(compz);
        let (n_, ldz) = (n as i32, n.max(1) as i32);
        let (lw, liw) = (work.len() as i32, iwork.len() as i32);
        let mut info = 0;
        // SAFETY: caller-sized buffers.
        unsafe {
            ffi::dstedc_(
                &cz,
                &n_,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                z.as_mut_ptr(),
                &ldz,
                work.as_mut_ptr(),
                &lw,
                iwork.as_mut_ptr(),
                &liw,
                &mut info,
                1,
            )
        };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (compz, n, d, e, z, work, iwork);
        unavailable!()
    }
}

/// `DSTEQR( COMPZ, N, D, E, Z, N, WORK )`.
pub fn dsteqr(compz: char, n: usize, d: &mut [f64], e: &mut [f64], z: &mut [f64], work: &mut [f64]) -> i32 {
    #[cfg(lapack_ref)]
    {
        let cz = ch(compz);
        let (n_, ldz) = (n as i32, n.max(1) as i32);
        let mut info = 0;
        // SAFETY: caller-sized buffers.
        unsafe {
            ffi::dsteqr_(
                &cz,
                &n_,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                z.as_mut_ptr(),
                &ldz,
                work.as_mut_ptr(),
                &mut info,
                1,
            )
        };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (compz, n, d, e, z, work);
        unavailable!()
    }
}

/// `DSTERF( N, D, E )`.
pub fn dsterf(n: usize, d: &mut [f64], e: &mut [f64]) -> i32 {
    #[cfg(lapack_ref)]
    {
        let n_ = n as i32;
        let mut info = 0;
        // SAFETY: caller-sized buffers.
        unsafe { ffi::dsterf_(&n_, d.as_mut_ptr(), e.as_mut_ptr(), &mut info) };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, d, e);
        unavailable!()
    }
}

/// `DLAED4( N, I, D, Z, DELTA, RHO, DLAM, INFO )`. Returns `(dlam, info)`.
pub fn dlaed4(n: usize, i: usize, d: &[f64], z: &[f64], delta: &mut [f64], rho: f64) -> (f64, i32) {
    #[cfg(lapack_ref)]
    {
        let (n_, i_) = (n as i32, i as i32);
        let (mut dlam, mut info) = (0.0, 0);
        // SAFETY: caller-sized buffers of length n.
        unsafe { ffi::dlaed4_(&n_, &i_, d.as_ptr(), z.as_ptr(), delta.as_mut_ptr(), &rho, &mut dlam, &mut info) };
        (dlam, info)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, i, d, z, delta, rho);
        unavailable!()
    }
}

/// `DLARFG( N, ALPHA, X, 1, TAU )`. Returns `(alpha, tau)`.
pub fn dlarfg(n: usize, alpha: f64, x: &mut [f64]) -> (f64, f64) {
    #[cfg(lapack_ref)]
    {
        let (n_, inc) = (n as i32, 1i32);
        let (mut al, mut tau) = (alpha, 0.0);
        let mut dummy = [0.0f64; 1];
        let p = if x.is_empty() { dummy.as_mut_ptr() } else { x.as_mut_ptr() };
        // SAFETY: x has n-1 entries.
        unsafe { ffi::dlarfg_(&n_, &mut al, p, &inc, &mut tau) };
        (al, tau)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, alpha, x);
        unavailable!()
    }
}

/// `DNRM2( N, X, 1 )`.
pub fn dnrm2(x: &[f64]) -> f64 {
    #[cfg(lapack_ref)]
    {
        let (n_, inc) = (x.len() as i32, 1i32);
        // SAFETY: n = x.len().
        unsafe { ffi::dnrm2_(&n_, x.as_ptr(), &inc) }
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = x;
        unavailable!()
    }
}

/// `DLARTG( F, G, C, S, R )`. Returns `(c, s, r)`.
pub fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    #[cfg(lapack_ref)]
    {
        let (mut c, mut s, mut r) = (0.0, 0.0, 0.0);
        // SAFETY: scalars by reference.
        unsafe { ffi::dlartg_(&f, &g, &mut c, &mut s, &mut r) };
        (c, s, r)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (f, g);
        unavailable!()
    }
}

/// `DLAPY2( X, Y )`.
pub fn dlapy2(x: f64, y: f64) -> f64 {
    #[cfg(lapack_ref)]
    {
        // SAFETY: scalars by reference.
        unsafe { ffi::dlapy2_(&x, &y) }
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (x, y);
        unavailable!()
    }
}

/// `DGEMM` with explicit leading dimensions.
pub fn dgemm(
    ta: char,
    tb: char,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    #[cfg(lapack_ref)]
    {
        let (ta_, tb_) = (ch(ta), ch(tb));
        let (m_, n_, k_) = (m as i32, n as i32, k as i32);
        let (lda_, ldb_, ldc_) = (lda as i32, ldb as i32, ldc as i32);
        // SAFETY: caller-sized buffers.
        unsafe {
            ffi::dgemm_(
                &ta_,
                &tb_,
                &m_,
                &n_,
                &k_,
                &alpha,
                a.as_ptr(),
                &lda_,
                b.as_ptr(),
                &ldb_,
                &beta,
                c.as_mut_ptr(),
                &ldc_,
                1,
                1,
            )
        };
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        unavailable!()
    }
}

/// `ZLADIV( X, Y )`.
pub fn zladiv(x: C64, y: C64) -> C64 {
    #[cfg(lapack_ref)]
    {
        // SAFETY: scalars by reference, complex result by value.
        unsafe { ffi::zladiv_(&x, &y) }
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (x, y);
        unavailable!()
    }
}

/// `ZLARFG( N, ALPHA, X, 1, TAU )`. Returns `(alpha, tau)`.
pub fn zlarfg(n: usize, alpha: C64, x: &mut [C64]) -> (C64, C64) {
    #[cfg(lapack_ref)]
    {
        let (n_, inc) = (n as i32, 1i32);
        let (mut al, mut tau) = (alpha, C64::default());
        let mut dummy = [C64::default(); 1];
        let p = if x.is_empty() { dummy.as_mut_ptr() } else { x.as_mut_ptr() };
        // SAFETY: x has n-1 entries.
        unsafe { ffi::zlarfg_(&n_, &mut al, p, &inc, &mut tau) };
        (al, tau)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, alpha, x);
        unavailable!()
    }
}

macro_rules! avail {
    ($body:block, $($unused:expr),*) => {{
        #[cfg(lapack_ref)]
        $body
        #[cfg(not(lapack_ref))]
        {
            let _ = ($($unused),*);
            unavailable!()
        }
    }};
}

/// `DDOT( N, X, INCX, Y, INCY )` over the given strided slices.
pub fn ddot(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize) -> f64 {
    avail!(
        {
            let (n_, ix, iy) = (n as i32, incx as i32, incy as i32);
            // SAFETY: caller-sized buffers.
            unsafe { ffi::ddot_(&n_, x.as_ptr(), &ix, y.as_ptr(), &iy) }
        },
        n,
        x,
        incx,
        y,
        incy
    )
}

/// `DGEMV` (unit `INCY`).
pub fn dgemv(
    t: char,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    avail!(
        {
            let (t_, m_, n_, lda_, ix, iy) = (ch(t), m as i32, n as i32, lda as i32, incx as i32, incy as i32);
            // SAFETY: caller-sized buffers.
            unsafe {
                ffi::dgemv_(&t_, &m_, &n_, &alpha, a.as_ptr(), &lda_, x.as_ptr(), &ix, &beta, y.as_mut_ptr(), &iy, 1)
            }
        },
        t,
        m,
        n,
        alpha,
        a,
        lda,
        x,
        incx,
        beta,
        y
    )
}

/// `DSYMV` (unit increments).
pub fn dsymv(u: char, n: usize, alpha: f64, a: &[f64], lda: usize, x: &[f64], beta: f64, y: &mut [f64]) {
    avail!(
        {
            let (u_, n_, lda_, one) = (ch(u), n as i32, lda as i32, 1i32);
            // SAFETY: caller-sized buffers.
            unsafe {
                ffi::dsymv_(&u_, &n_, &alpha, a.as_ptr(), &lda_, x.as_ptr(), &one, &beta, y.as_mut_ptr(), &one, 1)
            }
        },
        u,
        n,
        alpha,
        a,
        lda,
        x,
        beta,
        y
    )
}

/// `DSYR2` (unit increments).
pub fn dsyr2(u: char, n: usize, alpha: f64, x: &[f64], y: &[f64], a: &mut [f64], lda: usize) {
    avail!(
        {
            let (u_, n_, lda_, one) = (ch(u), n as i32, lda as i32, 1i32);
            // SAFETY: caller-sized buffers.
            unsafe { ffi::dsyr2_(&u_, &n_, &alpha, x.as_ptr(), &one, y.as_ptr(), &one, a.as_mut_ptr(), &lda_, 1) }
        },
        u,
        n,
        alpha,
        x,
        y,
        a,
        lda
    )
}

/// `DSYR2K`.
pub fn dsyr2k(
    u: char,
    t: char,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    avail!(
        {
            let (u_, t_, n_, k_) = (ch(u), ch(t), n as i32, k as i32);
            let (lda_, ldb_, ldc_) = (lda as i32, ldb as i32, ldc as i32);
            // SAFETY: caller-sized buffers.
            unsafe {
                ffi::dsyr2k_(
                    &u_,
                    &t_,
                    &n_,
                    &k_,
                    &alpha,
                    a.as_ptr(),
                    &lda_,
                    b.as_ptr(),
                    &ldb_,
                    &beta,
                    c.as_mut_ptr(),
                    &ldc_,
                    1,
                    1,
                )
            }
        },
        u,
        t,
        n,
        k,
        alpha,
        a,
        lda,
        b,
        ldb,
        beta,
        c,
        ldc
    )
}

/// `DTRMM`.
pub fn dtrmm(
    side: char,
    uplo: char,
    ta: char,
    diag: char,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
) {
    avail!(
        {
            let (s_, u_, t_, d_) = (ch(side), ch(uplo), ch(ta), ch(diag));
            let (m_, n_, lda_, ldb_) = (m as i32, n as i32, lda as i32, ldb as i32);
            // SAFETY: caller-sized buffers.
            unsafe {
                ffi::dtrmm_(&s_, &u_, &t_, &d_, &m_, &n_, &alpha, a.as_ptr(), &lda_, b.as_mut_ptr(), &ldb_, 1, 1, 1, 1)
            }
        },
        side,
        uplo,
        ta,
        diag,
        m,
        n,
        alpha,
        a,
        lda,
        b,
        ldb
    )
}

/// `DLASCL( TYPE, 0, 0, CFROM, CTO, M, N, A, LDA, INFO )`.
pub fn dlascl(ty: char, cfrom: f64, cto: f64, m: usize, n: usize, a: &mut [f64], lda: usize) -> i32 {
    avail!(
        {
            let (t_, z, m_, n_, lda_) = (ch(ty), 0i32, m as i32, n as i32, lda as i32);
            let mut info = 0;
            // SAFETY: caller-sized buffers.
            unsafe { ffi::dlascl_(&t_, &z, &z, &cfrom, &cto, &m_, &n_, a.as_mut_ptr(), &lda_, &mut info, 1) };
            info
        },
        ty,
        cfrom,
        cto,
        m,
        n,
        a,
        lda
    )
}

/// `DLAMCH( CMACH )`.
pub fn dlamch(c: char) -> f64 {
    avail!(
        {
            let c_ = ch(c);
            // SAFETY: one character by reference.
            unsafe { ffi::dlamch_(&c_, 1) }
        },
        c
    )
}

// ---------------------------------------------------------------------------
// MRRR (DSTEMR) component wrappers.
//
// Argument order and mutability are `SRC/*.f`'s; every buffer is the caller's
// and is sized by the reference's documented length, so a short slice is a
// test bug, not an oracle one.
// ---------------------------------------------------------------------------

/// `DLAEV2( A, B, C, RT1, RT2, CS1, SN1 )`.
pub fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    #[cfg(lapack_ref)]
    {
        let (mut rt1, mut rt2, mut cs1, mut sn1) = (0.0, 0.0, 0.0, 0.0);
        // SAFETY: seven scalars by reference.
        unsafe { ffi::dlaev2_(&a, &b, &c, &mut rt1, &mut rt2, &mut cs1, &mut sn1) };
        (rt1, rt2, cs1, sn1)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (a, b, c);
        unavailable!()
    }
}

/// `DLAE2( A, B, C, RT1, RT2 )`.
pub fn dlae2(a: f64, b: f64, c: f64) -> (f64, f64) {
    #[cfg(lapack_ref)]
    {
        let (mut rt1, mut rt2) = (0.0, 0.0);
        // SAFETY: five scalars by reference.
        unsafe { ffi::dlae2_(&a, &b, &c, &mut rt1, &mut rt2) };
        (rt1, rt2)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (a, b, c);
        unavailable!()
    }
}

/// `ZGEMV( TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY )`.
pub fn zgemv(
    trans: char,
    m: usize,
    n: usize,
    alpha: C64,
    a: &[C64],
    lda: usize,
    x: &[C64],
    incx: usize,
    beta: C64,
    y: &mut [C64],
    incy: usize,
) {
    #[cfg(lapack_ref)]
    {
        let t = ch(trans);
        let (m_, n_, lda_, incx_, incy_) = (m as i32, n as i32, lda as i32, incx as i32, incy as i32);
        // SAFETY: caller-sized buffers.
        unsafe {
            ffi::zgemv_(&t, &m_, &n_, &alpha, a.as_ptr(), &lda_, x.as_ptr(), &incx_, &beta, y.as_mut_ptr(), &incy_, 1)
        };
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
        unavailable!()
    }
}

/// `DLARRC( JOBT, N, VL, VU, D, E, PIVMIN, EIGCNT, LCNT, RCNT, INFO )`.
/// Returns `(eigcnt, lcnt, rcnt, info)`.
pub fn dlarrc(jobt: char, n: usize, vl: f64, vu: f64, d: &[f64], e: &[f64], pivmin: f64) -> (i32, i32, i32, i32) {
    #[cfg(lapack_ref)]
    {
        let j = ch(jobt);
        let n_ = n as i32;
        let (mut eigcnt, mut lcnt, mut rcnt, mut info) = (0, 0, 0, 0);
        // SAFETY: `d` has n entries, `e` at least n-1.
        unsafe {
            ffi::dlarrc_(
                &j,
                &n_,
                &vl,
                &vu,
                d.as_ptr(),
                e.as_ptr(),
                &pivmin,
                &mut eigcnt,
                &mut lcnt,
                &mut rcnt,
                &mut info,
                1,
            )
        };
        (eigcnt, lcnt, rcnt, info)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (jobt, n, vl, vu, d, e, pivmin);
        unavailable!()
    }
}

/// `DLARRK( N, IW, GL, GU, D, E2, PIVMIN, RELTOL, W, WERR, INFO )`.
/// Returns `(w, werr, info)`.
pub fn dlarrk(
    n: usize,
    iw: usize,
    gl: f64,
    gu: f64,
    d: &[f64],
    e2: &[f64],
    pivmin: f64,
    reltol: f64,
) -> (f64, f64, i32) {
    #[cfg(lapack_ref)]
    {
        let (n_, iw_) = (n as i32, iw as i32);
        let (mut w, mut werr, mut info) = (0.0, 0.0, 0);
        // SAFETY: `d` has n entries, `e2` at least n-1.
        unsafe {
            ffi::dlarrk_(&n_, &iw_, &gl, &gu, d.as_ptr(), e2.as_ptr(), &pivmin, &reltol, &mut w, &mut werr, &mut info)
        };
        (w, werr, info)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, iw, gl, gu, d, e2, pivmin, reltol);
        unavailable!()
    }
}

/// `DLANEG( N, D, LLD, SIGMA, PIVMIN, R )`.
pub fn dlaneg(n: usize, d: &[f64], lld: &[f64], sigma: f64, pivmin: f64, r: usize) -> i32 {
    #[cfg(lapack_ref)]
    {
        let (n_, r_) = (n as i32, r as i32);
        // SAFETY: `d` has n entries, `lld` at least n-1.
        unsafe { ffi::dlaneg_(&n_, d.as_ptr(), lld.as_ptr(), &sigma, &pivmin, &r_) }
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, d, lld, sigma, pivmin, r);
        unavailable!()
    }
}

/// `DLAR1V( .. )`. Returns `(negcnt, ztz, mingma, r, isuppz, nrminv, resid, rqcorr)`;
/// `z` and `work` are written in place.
#[allow(clippy::type_complexity)]
pub fn dlar1v(
    n: usize,
    b1: usize,
    bn: usize,
    lambda: f64,
    d: &[f64],
    l: &[f64],
    ld: &[f64],
    lld: &[f64],
    pivmin: f64,
    gaptol: f64,
    z: &mut [f64],
    wantnc: bool,
    r_in: usize,
    work: &mut [f64],
) -> (i32, f64, f64, i32, [i32; 2], f64, f64, f64) {
    #[cfg(lapack_ref)]
    {
        let (n_, b1_, bn_) = (n as i32, b1 as i32, bn as i32);
        let wnc: i32 = if wantnc { 1 } else { 0 };
        let mut r = r_in as i32;
        let mut isuppz = [0i32; 2];
        let (mut negcnt, mut ztz, mut mingma) = (0, 0.0, 0.0);
        let (mut nrminv, mut resid, mut rqcorr) = (0.0, 0.0, 0.0);
        // SAFETY: caller-sized buffers (`work` is 4n).
        unsafe {
            ffi::dlar1v_(
                &n_,
                &b1_,
                &bn_,
                &lambda,
                d.as_ptr(),
                l.as_ptr(),
                ld.as_ptr(),
                lld.as_ptr(),
                &pivmin,
                &gaptol,
                z.as_mut_ptr(),
                &wnc,
                &mut negcnt,
                &mut ztz,
                &mut mingma,
                &mut r,
                isuppz.as_mut_ptr(),
                &mut nrminv,
                &mut resid,
                &mut rqcorr,
                work.as_mut_ptr(),
            )
        };
        (negcnt, ztz, mingma, r, isuppz, nrminv, resid, rqcorr)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, b1, bn, lambda, d, l, ld, lld, pivmin, gaptol, z, wantnc, r_in, work);
        unavailable!()
    }
}

/// `DLASQ2( N, Z, INFO )`. `z` is the reference's `4*N` workspace.
pub fn dlasq2(n: usize, z: &mut [f64]) -> i32 {
    #[cfg(lapack_ref)]
    {
        let n_ = n as i32;
        let mut info = 0;
        // SAFETY: `z` has at least 4n entries.
        unsafe { ffi::dlasq2_(&n_, z.as_mut_ptr(), &mut info) };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, z);
        unavailable!()
    }
}

/// `DLARRA( N, D, E, E2, SPLTOL, TNRM, NSPLIT, ISPLIT, INFO )`.
pub fn dlarra(
    n: usize,
    d: &[f64],
    e: &mut [f64],
    e2: &mut [f64],
    spltol: f64,
    tnrm: f64,
    isplit: &mut [i32],
) -> (i32, i32) {
    #[cfg(lapack_ref)]
    {
        let n_ = n as i32;
        let (mut nsplit, mut info) = (0, 0);
        // SAFETY: caller-sized buffers.
        unsafe {
            ffi::dlarra_(
                &n_,
                d.as_ptr(),
                e.as_mut_ptr(),
                e2.as_mut_ptr(),
                &spltol,
                &tnrm,
                &mut nsplit,
                isplit.as_mut_ptr(),
                &mut info,
            )
        };
        (nsplit, info)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, d, e, e2, spltol, tnrm, isplit);
        unavailable!()
    }
}

/// `DLARRR( N, D, E, INFO )`.
pub fn dlarrr(n: usize, d: &[f64], e: &mut [f64]) -> i32 {
    #[cfg(lapack_ref)]
    {
        let n_ = n as i32;
        let mut info = 0;
        // SAFETY: caller-sized buffers.
        unsafe { ffi::dlarrr_(&n_, d.as_ptr(), e.as_mut_ptr(), &mut info) };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, d, e);
        unavailable!()
    }
}

/// `DLARRB( .. )`.
pub fn dlarrb(
    n: usize,
    d: &[f64],
    lld: &[f64],
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
    #[cfg(lapack_ref)]
    {
        let (n_, if_, il_, off_) = (n as i32, ifirst as i32, ilast as i32, offset as i32);
        let mut info = 0;
        // SAFETY: caller-sized buffers (`work` 2n, `iwork` 2n).
        unsafe {
            ffi::dlarrb_(
                &n_,
                d.as_ptr(),
                lld.as_ptr(),
                &if_,
                &il_,
                &rtol1,
                &rtol2,
                &off_,
                w.as_mut_ptr(),
                wgap.as_mut_ptr(),
                werr.as_mut_ptr(),
                work.as_mut_ptr(),
                iwork.as_mut_ptr(),
                &pivmin,
                &spdiam,
                &twist,
                &mut info,
            )
        };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, d, lld, ifirst, ilast, rtol1, rtol2, offset, w, wgap, werr, work, iwork, pivmin, spdiam, twist);
        unavailable!()
    }
}

/// `DLARRF( .. )`. Returns `(sigma, info)`.
pub fn dlarrf(
    n: usize,
    d: &[f64],
    l: &[f64],
    ld: &[f64],
    clstrt: usize,
    clend: usize,
    w: &[f64],
    wgap: &mut [f64],
    werr: &[f64],
    spdiam: f64,
    clgapl: f64,
    clgapr: f64,
    pivmin: f64,
    dplus: &mut [f64],
    lplus: &mut [f64],
    work: &mut [f64],
) -> (f64, i32) {
    #[cfg(lapack_ref)]
    {
        let (n_, cs_, ce_) = (n as i32, clstrt as i32, clend as i32);
        let (mut sigma, mut info) = (0.0, 0);
        // SAFETY: caller-sized buffers (`work` 2n).
        unsafe {
            ffi::dlarrf_(
                &n_,
                d.as_ptr(),
                l.as_ptr(),
                ld.as_ptr(),
                &cs_,
                &ce_,
                w.as_ptr(),
                wgap.as_mut_ptr(),
                werr.as_ptr(),
                &spdiam,
                &clgapl,
                &clgapr,
                &pivmin,
                &mut sigma,
                dplus.as_mut_ptr(),
                lplus.as_mut_ptr(),
                work.as_mut_ptr(),
                &mut info,
            )
        };
        (sigma, info)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, d, l, ld, clstrt, clend, w, wgap, werr, spdiam, clgapl, clgapr, pivmin, dplus, lplus, work);
        unavailable!()
    }
}

/// `DLARRJ( .. )`.
pub fn dlarrj(
    n: usize,
    d: &[f64],
    e2: &[f64],
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
    #[cfg(lapack_ref)]
    {
        let (n_, if_, il_, off_) = (n as i32, ifirst as i32, ilast as i32, offset as i32);
        let mut info = 0;
        // SAFETY: caller-sized buffers (`work` 2n, `iwork` 2n).
        unsafe {
            ffi::dlarrj_(
                &n_,
                d.as_ptr(),
                e2.as_ptr(),
                &if_,
                &il_,
                &rtol,
                &off_,
                w.as_mut_ptr(),
                werr.as_mut_ptr(),
                work.as_mut_ptr(),
                iwork.as_mut_ptr(),
                &pivmin,
                &spdiam,
                &mut info,
            )
        };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (n, d, e2, ifirst, ilast, rtol, offset, w, werr, work, iwork, pivmin, spdiam);
        unavailable!()
    }
}

/// `DLARRE( .. )`. Returns `(vl, vu, nsplit, m, pivmin, info)`; the arrays are
/// written in place.
#[allow(clippy::type_complexity)]
pub fn dlarre(
    range: char,
    n: usize,
    vl_in: f64,
    vu_in: f64,
    il: usize,
    iu: usize,
    d: &mut [f64],
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
    #[cfg(lapack_ref)]
    {
        let r = ch(range);
        let (n_, il_, iu_) = (n as i32, il as i32, iu as i32);
        let (mut vl, mut vu) = (vl_in, vu_in);
        let (mut nsplit, mut m, mut pivmin, mut info) = (0, 0, 0.0, 0);
        // SAFETY: caller-sized buffers (`work` 6n, `iwork` 5n).
        unsafe {
            ffi::dlarre_(
                &r,
                &n_,
                &mut vl,
                &mut vu,
                &il_,
                &iu_,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                e2.as_mut_ptr(),
                &rtol1,
                &rtol2,
                &spltol,
                &mut nsplit,
                isplit.as_mut_ptr(),
                &mut m,
                w.as_mut_ptr(),
                werr.as_mut_ptr(),
                wgap.as_mut_ptr(),
                iblock.as_mut_ptr(),
                indexw.as_mut_ptr(),
                gers.as_mut_ptr(),
                &mut pivmin,
                work.as_mut_ptr(),
                iwork.as_mut_ptr(),
                &mut info,
                1,
            )
        };
        (vl, vu, nsplit, m, pivmin, info)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (
            range, n, vl_in, vu_in, il, iu, d, e, e2, rtol1, rtol2, spltol, isplit, w, werr, wgap, iblock, indexw,
            gers, work, iwork,
        );
        unavailable!()
    }
}

/// `DLARRV( .. )`.
pub fn dlarrv(
    n: usize,
    vl: f64,
    vu: f64,
    d: &mut [f64],
    l: &mut [f64],
    pivmin: f64,
    isplit: &[i32],
    m: usize,
    dol: usize,
    dou: usize,
    minrgp: f64,
    rtol1: f64,
    rtol2: f64,
    w: &mut [f64],
    werr: &mut [f64],
    wgap: &mut [f64],
    iblock: &[i32],
    indexw: &[i32],
    gers: &[f64],
    z: &mut [f64],
    ldz: usize,
    isuppz: &mut [i32],
    work: &mut [f64],
    iwork: &mut [i32],
) -> i32 {
    #[cfg(lapack_ref)]
    {
        let (n_, m_, dol_, dou_, ldz_) = (n as i32, m as i32, dol as i32, dou as i32, ldz as i32);
        let mut info = 0;
        // SAFETY: caller-sized buffers (`work` 12n, `iwork` 7n).
        unsafe {
            ffi::dlarrv_(
                &n_,
                &vl,
                &vu,
                d.as_mut_ptr(),
                l.as_mut_ptr(),
                &pivmin,
                isplit.as_ptr(),
                &m_,
                &dol_,
                &dou_,
                &minrgp,
                &rtol1,
                &rtol2,
                w.as_mut_ptr(),
                werr.as_mut_ptr(),
                wgap.as_mut_ptr(),
                iblock.as_ptr(),
                indexw.as_ptr(),
                gers.as_ptr(),
                z.as_mut_ptr(),
                &ldz_,
                isuppz.as_mut_ptr(),
                work.as_mut_ptr(),
                iwork.as_mut_ptr(),
                &mut info,
            )
        };
        info
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (
            n, vl, vu, d, l, pivmin, isplit, m, dol, dou, minrgp, rtol1, rtol2, w, werr, wgap, iblock, indexw, gers, z,
            ldz, isuppz, work, iwork,
        );
        unavailable!()
    }
}

/// `DSTEMR( JOBZ, RANGE, N, D, E, VL, VU, IL, IU, M, W, Z, LDZ, NZC, ISUPPZ,
/// TRYRAC, WORK, LWORK, IWORK, LIWORK, INFO )`. Returns `(m, tryrac, info)`.
pub fn dstemr(
    jobz: char,
    range: char,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    vl: f64,
    vu: f64,
    il: usize,
    iu: usize,
    w: &mut [f64],
    z: &mut [f64],
    ldz: usize,
    nzc: i32,
    isuppz: &mut [i32],
    tryrac_in: bool,
    work: &mut [f64],
    iwork: &mut [i32],
) -> (i32, bool, i32) {
    #[cfg(lapack_ref)]
    {
        let (jz, rg) = (ch(jobz), ch(range));
        let (n_, il_, iu_, ldz_) = (n as i32, il as i32, iu as i32, ldz as i32);
        let (lw, liw) = (work.len() as i32, iwork.len() as i32);
        let mut tryrac: i32 = if tryrac_in { 1 } else { 0 };
        let (mut m, mut info) = (0, 0);
        // SAFETY: caller-sized buffers.
        unsafe {
            ffi::dstemr_(
                &jz,
                &rg,
                &n_,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                &vl,
                &vu,
                &il_,
                &iu_,
                &mut m,
                w.as_mut_ptr(),
                z.as_mut_ptr(),
                &ldz_,
                &nzc,
                isuppz.as_mut_ptr(),
                &mut tryrac,
                work.as_mut_ptr(),
                &lw,
                iwork.as_mut_ptr(),
                &liw,
                &mut info,
                1,
                1,
            )
        };
        (m, tryrac != 0, info)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (jobz, range, n, d, e, vl, vu, il, iu, w, z, ldz, nzc, isuppz, tryrac_in, work, iwork);
        unavailable!()
    }
}

/// `DLARNV( IDIST, ISEED, N, X )`.
pub fn dlarnv(idist: i32, iseed: &mut [i32; 4], n: usize, x: &mut [f64]) {
    #[cfg(lapack_ref)]
    {
        let n_ = n as i32;
        // SAFETY: `x` has at least n entries; `iseed` is 4 entries.
        unsafe { ffi::dlarnv_(&idist, iseed.as_mut_ptr(), &n_, x.as_mut_ptr()) };
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (idist, iseed, n, x);
        unavailable!()
    }
}

/// `DLASQ6( I0, N0, Z, PP, DMIN, DMIN1, DMIN2, DN, DNM1, DNM2 )`.
pub fn dlasq6(i0: usize, n0: usize, z: &mut [f64], pp: usize) -> (f64, f64, f64, f64, f64, f64) {
    #[cfg(lapack_ref)]
    {
        let (i, n, p) = (i0 as i32, n0 as i32, pp as i32);
        let (mut dmin, mut dmin1, mut dmin2, mut dn, mut dnm1, mut dnm2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        // SAFETY: `z` has at least 4*n0 entries.
        unsafe {
            ffi::dlasq6_(&i, &n, z.as_mut_ptr(), &p, &mut dmin, &mut dmin1, &mut dmin2, &mut dn, &mut dnm1, &mut dnm2)
        };
        (dmin, dmin1, dmin2, dn, dnm1, dnm2)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (i0, n0, z, pp);
        unavailable!()
    }
}

/// `DLASQ5( .. )`. Returns `(tau, dmin, dmin1, dmin2, dn, dnm1, dnm2)`.
pub fn dlasq5(
    i0: usize,
    n0: usize,
    z: &mut [f64],
    pp: usize,
    tau_in: f64,
    sigma: f64,
    ieee: bool,
    eps: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    #[cfg(lapack_ref)]
    {
        let (i, n, p) = (i0 as i32, n0 as i32, pp as i32);
        let ie: i32 = if ieee { 1 } else { 0 };
        let mut tau = tau_in;
        let (mut dmin, mut dmin1, mut dmin2, mut dn, mut dnm1, mut dnm2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        // SAFETY: `z` has at least 4*n0 entries.
        unsafe {
            ffi::dlasq5_(
                &i,
                &n,
                z.as_mut_ptr(),
                &p,
                &mut tau,
                &sigma,
                &mut dmin,
                &mut dmin1,
                &mut dmin2,
                &mut dn,
                &mut dnm1,
                &mut dnm2,
                &ie,
                &eps,
            )
        };
        (tau, dmin, dmin1, dmin2, dn, dnm1, dnm2)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (i0, n0, z, pp, tau_in, sigma, ieee, eps);
        unavailable!()
    }
}

/// `DLASQ4( .. )`. `state` is `(DMIN, DMIN1, DMIN2, DN, DN1, DN2, TAU, G,
/// TTYPE)`; returns the updated `(TAU, G, TTYPE)`.
pub fn dlasq4(
    i0: usize,
    n0: usize,
    z: &[f64],
    pp: usize,
    n0in: usize,
    state: (f64, f64, f64, f64, f64, f64, f64, f64, i32),
) -> (f64, f64, i32) {
    #[cfg(lapack_ref)]
    {
        let (i, n, p, ni) = (i0 as i32, n0 as i32, pp as i32, n0in as i32);
        let (mut tau, mut g, mut ttype) = (state.6, state.7, state.8);
        // SAFETY: `z` has at least 4*n0+pp entries.
        unsafe {
            ffi::dlasq4_(
                &i,
                &n,
                z.as_ptr(),
                &p,
                &ni,
                &state.0,
                &state.1,
                &state.2,
                &state.3,
                &state.4,
                &state.5,
                &mut tau,
                &mut ttype,
                &mut g,
            )
        };
        (tau, g, ttype)
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (i0, n0, z, pp, n0in, state);
        unavailable!()
    }
}

/// `ILAENV( ISPEC, NAME, OPTS, N1, N2, N3, N4 )`.
pub fn ilaenv(ispec: i32, name: &str, opts: &str, n1: i32, n2: i32, n3: i32, n4: i32) -> i32 {
    #[cfg(lapack_ref)]
    {
        // SAFETY: hidden lengths match the byte slices.
        unsafe { ffi::ilaenv_(&ispec, name.as_ptr(), opts.as_ptr(), &n1, &n2, &n3, &n4, name.len(), opts.len()) }
    }
    #[cfg(not(lapack_ref))]
    {
        let _ = (ispec, name, opts, n1, n2, n3, n4);
        unavailable!()
    }
}
