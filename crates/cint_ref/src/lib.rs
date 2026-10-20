//! Test-only FFI to **libcint 6.1.3**'s vendored tridiagonal eigensolver:
//! `src/eigh.c`'s `#else` (no-`LAPACK_FOUND`) branch, which is what cintx's
//! oracle build uses and therefore cintx's ground truth.
//!
//! This is **not** Reference-LAPACK. `docs/cintx-lapack-blas-adoption-plan.md`
//! §4.2 lists six constant-level divergences between the two, each sufficient
//! on its own to make `max_ulp = 0` unreachable, so the two oracles must never
//! be mixed: `ref_*_bitexact` covers Reference-LAPACK, `cint_*_bitexact`
//! covers this one.
//!
//! Set `CINTX_LIBCINT_DIR` to the pinned tree (cintx's `libcint-master/`) at
//! build time. Without it every entry point reports unavailable and the tests
//! print `SKIP` — unless `CINT_REF_REQUIRE=1`, which panics so a gate run can
//! never pass vacuously.

#![allow(clippy::missing_safety_doc)]

/// Whether libcint's `eigh.c` was compiled and linked.
pub const AVAILABLE: bool = cfg!(cint_ref);

/// The pinned libcint tree (only when [`AVAILABLE`]).
pub fn source_dir() -> Option<&'static str> {
    option_env!("CINT_REF_DIR")
}

/// `MXRYSROOTS`, libcint's cap on the problem size.
pub const MXRYSROOTS: usize = 13;

/// Standard gate for the Path C oracle tests.
pub fn require(test: &str) -> bool {
    if AVAILABLE {
        return true;
    }
    if std::env::var("CINT_REF_REQUIRE").as_deref() == Ok("1") {
        panic!("{test}: libcint oracle not linked (set CINTX_LIBCINT_DIR at build time) but CINT_REF_REQUIRE=1");
    }
    eprintln!("SKIP {test}: CINTX_LIBCINT_DIR not set at build time");
    false
}

#[cfg(cint_ref)]
mod ffi {
    unsafe extern "C" {
        pub fn cintref_diagonalize(n: i32, diag: *mut f64, diag_off1: *mut f64, eig: *mut f64, vec: *mut f64) -> i32;
        pub fn cintref_record_begin(cap: i32, ns: *mut i32, diag: *mut f64, off: *mut f64);
        pub fn cintref_record_end() -> i32;
        pub fn CINTrys_jacobi(n: i32, x: f64, lower: f64, roots: *mut f64, weights: *mut f64) -> i32;
        pub fn CINTrys_laguerre(n: i32, x: f64, lower: f64, roots: *mut f64, weights: *mut f64) -> i32;
        pub fn CINTlrys_jacobi(n: i32, x: f64, lower: f64, roots: *mut f64, weights: *mut f64) -> i32;
        pub fn CINTlrys_laguerre(n: i32, x: f64, lower: f64, roots: *mut f64, weights: *mut f64) -> i32;
    }
}

#[allow(unused_macros)]
macro_rules! unavailable {
    () => {
        panic!("libcint oracle not linked; set CINTX_LIBCINT_DIR at build time")
    };
}

/// `_CINTdiagonalize( n, diag, diag_off1, eig, vec )` — libcint 6.1.3's
/// vendored solver, bypassing the corpus recorder.
///
/// `diag[0..n]` is overwritten; `diag_off1[0..n]` holds the off-diagonal in
/// `[0..n-1]` with `[n-1]` as scratch; `eig[0..n]` receives ascending
/// eigenvalues and `vec[0..n*n]` the eigenvectors with `vec[i*n + j]` the
/// `j`-th component of the `i`-th. Returns `INFO`.
pub fn diagonalize(n: usize, diag: &mut [f64], diag_off1: &mut [f64], eig: &mut [f64], vec: &mut [f64]) -> i32 {
    #[cfg(cint_ref)]
    {
        assert!(diag.len() >= n && diag_off1.len() >= n && eig.len() >= n && vec.len() >= n * n);
        // SAFETY: lengths asserted; the four slices are distinct.
        unsafe {
            ffi::cintref_diagonalize(
                n as i32,
                diag.as_mut_ptr(),
                diag_off1.as_mut_ptr(),
                eig.as_mut_ptr(),
                vec.as_mut_ptr(),
            )
        }
    }
    #[cfg(not(cint_ref))]
    {
        let _ = (n, diag, diag_off1, eig, vec);
        unavailable!()
    }
}

/// One recorded Wheeler problem: the tridiagonal libcint handed its own
/// eigensolver.
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    pub n: usize,
    /// `diag[0..n]`, the Wheeler `a` coefficients.
    pub diag: Vec<f64>,
    /// `diag_off1[0..n]`: `b[1..n]` in `[0..n-1]`, `[n-1]` scratch.
    pub off: Vec<f64>,
}

/// Which libcint Wheeler entry point a sweep drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    /// `CINTrys_jacobi` (double moments).
    Jacobi,
    /// `CINTrys_laguerre` (double moments, `lower > 0`).
    Laguerre,
    /// `CINTlrys_jacobi` (long-double moments).
    LJacobi,
    /// `CINTlrys_laguerre`.
    LLaguerre,
}

/// Drive libcint's own Wheeler path over `(n, x, lower)` and return every
/// tridiagonal it handed `_CINTdiagonalize` — the A0 Wheeler corpus, recorded
/// from the vendor rather than reconstructed.
pub fn wheeler_corpus(cases: &[(Entry, usize, f64, f64)], cap: usize) -> Vec<Problem> {
    #[cfg(cint_ref)]
    {
        let mut ns = vec![0i32; cap];
        let mut diag = vec![0.0f64; cap * MXRYSROOTS];
        let mut off = vec![0.0f64; cap * MXRYSROOTS];
        // SAFETY: the three buffers hold `cap` slots of MXRYSROOTS entries,
        // which is the shim's contract, and the recorder stops at `cap`.
        unsafe { ffi::cintref_record_begin(cap as i32, ns.as_mut_ptr(), diag.as_mut_ptr(), off.as_mut_ptr()) };
        for &(entry, n, x, lower) in cases {
            let mut roots = vec![0.0f64; MXRYSROOTS];
            let mut weights = vec![0.0f64; MXRYSROOTS];
            // SAFETY: `roots`/`weights` hold MXRYSROOTS >= n entries.
            unsafe {
                let f = match entry {
                    Entry::Jacobi => ffi::CINTrys_jacobi,
                    Entry::Laguerre => ffi::CINTrys_laguerre,
                    Entry::LJacobi => ffi::CINTlrys_jacobi,
                    Entry::LLaguerre => ffi::CINTlrys_laguerre,
                };
                f(n as i32, x, lower, roots.as_mut_ptr(), weights.as_mut_ptr());
            }
        }
        // SAFETY: recording was begun above.
        let got = unsafe { ffi::cintref_record_end() } as usize;
        (0..got)
            .map(|k| Problem {
                n: ns[k] as usize,
                diag: diag[k * MXRYSROOTS..k * MXRYSROOTS + MXRYSROOTS].to_vec(),
                off: off[k * MXRYSROOTS..k * MXRYSROOTS + MXRYSROOTS].to_vec(),
            })
            .collect()
    }
    #[cfg(not(cint_ref))]
    {
        let _ = (cases, cap);
        unavailable!()
    }
}
