//! `lapack_rs`: a divide-and-conquer symmetric/Hermitian eigensolver
//! transcribed from Reference-LAPACK v3.12.1 (and the reference BLAS of the
//! same tree), bit for bit.
//!
//! Every floating-point value is the one Reference-LAPACK computes when built
//! with `-O2 -march=x86-64 -ffp-contract=off` (see `xtask` and
//! `tests/ref_*`). Speed comes only from reorderings that cannot change a
//! value: element-wise kernels split across units, and independent
//! divide-and-conquer subproblems, secular roots and eigenvector columns run
//! in parallel. Results are identical at any launch geometry.
//!
//! There is one kernel set and no host parallel path. [`dsyevd_alloc`] and
//! [`zheevd_alloc`] run the CubeCL drivers on the runtime selected at compile
//! time (see [`runtime`]); [`dsyevd`] and [`zheevd`], which take the caller's
//! `LWORK`, stay the sequential reference transcription because the
//! reference's block sizes — and so its bits — depend on those lengths.
//!
//! Conventions follow LAPACK: column-major storage with `LDA = n`, [`Uplo`],
//! workspace queries ([`syevd_work_len`], [`heevd_work_len`]) and
//! caller-supplied workspace, so the drivers allocate nothing. Complex
//! matrices are passed as planar real/imaginary slices.
//!
//! The device path is also exposed directly in [`device`] for callers that
//! hold their own client and workspace (a device-resident SCF loop).
//!
//! # Two oracles
//!
//! Everything above is bit-exact to Reference-LAPACK. [`cint_stemr`] is the
//! one exception: it is the tridiagonal eigensolver **libcint 6.1.3** ships in
//! `src/eigh.c`'s no-`LAPACK_FOUND` branch, bit-exact to *that*, for callers
//! (cintx) whose ground truth is libcint rather than LAPACK. The two are
//! different MRRR variants with different constants and cannot be satisfied by
//! one routine — see `docs/a0-wheeler-differential.md`, which measures up to
//! 4.25e9 ULP between them. [`dstemr`] is the Reference-LAPACK MRRR entry
//! point; [`cint_stemr`] is libcint's. Do not substitute one for the other.
//!
//! A third, opt-in track, [`openblas`], reproduces scipy's bundled OpenBLAS
//! 0.3.30 (`SkylakeX` core, one thread) bit for bit instead: the same LAPACK
//! transcription over an OpenBLAS BLAS transcribed from the binary, with
//! OpenBLAS-order CubeCL kernels (`docs/openblas-bitexact-result.md`). It is
//! never compared with the other two.

#![warn(clippy::unwrap_used)]
// The transcriptions keep the reference's index arithmetic and operand order
// verbatim (`(N-NX+NB-1)/NB`, `n % 2 != 0`, `x = t * x`, `!(a < b)` with
// NaN semantics, indexed DO loops): rewriting them idiomatically is exactly
// the kind of edit that silently changes a bit.
#![allow(
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::int_plus_one,
    clippy::assign_op_pattern,
    clippy::too_many_arguments
)]

mod aux;
mod blas;
pub(crate) mod cint_stemr;
mod error;
mod flavor;
mod heevd;
mod hetrd;
mod laed4;
mod larr;
mod larr2;
mod larre;
mod larrv;
mod lasq;
pub mod openblas;
mod ptr;
pub mod runtime;
mod stedc;
mod stemr;
mod steqr;
mod syevd;
mod sytrd;
#[doc(hidden)]
pub mod testing;
mod zblas;

pub mod device;

pub use cint_stemr::{MXRYSROOTS, cint_stemr, cint_stemr_work_len};
pub use error::LapackError;
pub use heevd::{heevd_min_work_len, heevd_opt_work_len, heevd_work_len, zheevd, zheevd_alloc, zheevd_alloc_host};
#[cfg(feature = "stats")]
pub use stedc::stats;
pub use stedc::{SMLSIZ, stedc_work_len};
pub use stemr::{dstemr, dstemr_min_work_len, dstemr_work_len};
pub use steqr::{Compz, dsteqr, dsterf};
pub use syevd::{dsyevd, dsyevd_alloc, dsyevd_alloc_host, syevd_min_work_len, syevd_opt_work_len, syevd_work_len};

/// Which triangle of the input matrix holds the data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Uplo {
    Upper,
    Lower,
}

/// Whether eigenvectors are computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Job {
    ValuesOnly,
    Vectors,
}

/// BLAS `TRANS` for the level-2 kernels exposed in [`blocks`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Trans {
    /// `'N'`: `y := alpha*A*x + beta*y`.
    NoTrans,
    /// `'T'`: `y := alpha*A**T*x + beta*y`.
    Trans,
    /// `'C'`: `y := alpha*A**H*x + beta*y` (same as `'T'` for real `A`).
    ConjTrans,
}

/// Building blocks, exposed for component-level oracle tests. Each function
/// mirrors the reference routine of the same name on column-major slices with
/// `LDA = n` (or the stated leading dimension).
pub mod blocks {
    use crate::ptr::V;
    use crate::{Job, LapackError, Trans, Uplo};

    fn check(name: &'static str, s: usize, need: usize) -> Result<(), LapackError> {
        if s < need {
            Err(LapackError::InvalidArgument { arg: name, reason: format!("need {need} entries, got {s}") })
        } else {
            Ok(())
        }
    }

    /// `DSYTRD( UPLO, N, A, N, D, E, TAU, WORK, LWORK )` with `LWORK = work.len()`.
    pub fn dsytrd(
        uplo: Uplo,
        n: usize,
        a: &mut [f64],
        d: &mut [f64],
        e: &mut [f64],
        tau: &mut [f64],
        work: &mut [f64],
    ) -> Result<(), LapackError> {
        check("a", a.len(), n * n)?;
        check("d", d.len(), n)?;
        check("e", e.len(), n.saturating_sub(1).max(1))?;
        check("tau", tau.len(), n.saturating_sub(1).max(1))?;
        check("work", work.len(), 1)?;
        if n == 0 {
            return Ok(());
        }
        let lwork = work.len();
        // SAFETY: lengths checked; distinct slices. dsytrd writes E/TAU(1..N-1).
        unsafe {
            crate::sytrd::dsytrd(
                matches!(uplo, Uplo::Upper),
                n,
                V::new(a),
                n,
                V::new(d),
                V::new(e),
                V::new(tau),
                V::new(work),
                lwork,
            );
        }
        Ok(())
    }

    /// `DORMTR( 'L', UPLO, 'N', N, N, A, N, TAU, C, N, WORK, LWORK )`.
    pub fn dormtr_ln(
        uplo: Uplo,
        n: usize,
        a: &mut [f64],
        tau: &mut [f64],
        c: &mut [f64],
        work: &mut [f64],
    ) -> Result<(), LapackError> {
        check("a", a.len(), n * n)?;
        check("c", c.len(), n * n)?;
        check("tau", tau.len(), n.saturating_sub(1).max(1))?;
        check("work", work.len(), n.max(1))?;
        if n == 0 {
            return Ok(());
        }
        let lwork = work.len();
        // SAFETY: lengths checked; distinct slices.
        unsafe {
            crate::sytrd::dormtr_ln(
                matches!(uplo, Uplo::Upper),
                n,
                n,
                V::new(a),
                n,
                V::new(tau),
                V::new(c),
                n,
                V::new(work),
                lwork,
            );
        }
        Ok(())
    }

    /// `DSTEDC( 'I', N, D, E, Z, N, WORK, LWORK, IWORK, LIWORK )`.
    pub fn dstedc(
        n: usize,
        d: &mut [f64],
        e: &mut [f64],
        z: &mut [f64],
        work: &mut [f64],
        iwork: &mut [usize],
    ) -> Result<(), LapackError> {
        let (lw, liw) = crate::stedc::stedc_work_len(n);
        check("d", d.len(), n)?;
        check("e", e.len(), n.saturating_sub(1).max(1))?;
        check("z", z.len(), n * n)?;
        if work.len() < lw {
            return Err(LapackError::WorkspaceTooSmall { need: lw, got: work.len() });
        }
        if iwork.len() < liw {
            return Err(LapackError::WorkspaceTooSmall { need: liw, got: iwork.len() });
        }
        if n == 0 {
            return Ok(());
        }
        // SAFETY: lengths checked; distinct slices.
        let info =
            unsafe { crate::stedc::dstedc_i(n, V::new(d), V::new(e), V::new(z), n, V::new(work), V::new(iwork)) };
        if info != 0 { Err(LapackError::NoConvergence { routine: "dstedc", info }) } else { Ok(()) }
    }

    /// `DLAED4( N, I, D, Z, DELTA, RHO, DLAM, INFO )` (`i` 1-based). Returns `DLAM`.
    pub fn dlaed4(
        n: usize,
        i: usize,
        d: &mut [f64],
        z: &mut [f64],
        delta: &mut [f64],
        rho: f64,
    ) -> Result<f64, LapackError> {
        check("d", d.len(), n)?;
        check("z", z.len(), n)?;
        check("delta", delta.len(), n)?;
        if i == 0 || i > n {
            return Err(LapackError::InvalidArgument { arg: "i", reason: format!("need 1 <= i <= n = {n}") });
        }
        // SAFETY: lengths checked; d/z only read, delta written.
        let (dlam, info) = unsafe { crate::laed4::dlaed4(n, i, V::new(d), V::new(z), V::new(delta), rho) };
        if info != 0 { Err(LapackError::NoConvergence { routine: "dlaed4", info }) } else { Ok(dlam) }
    }

    /// `DLARFG( N, ALPHA, X, 1, TAU )`. Returns `(beta, tau)`.
    pub fn dlarfg(n: usize, alpha: f64, x: &mut [f64]) -> Result<(f64, f64), LapackError> {
        check("x", x.len(), n.saturating_sub(1))?;
        let mut dummy = [0.0f64; 1];
        let v = if x.is_empty() { V::new(&mut dummy) } else { V::new(x) };
        // SAFETY: n-1 entries checked.
        Ok(unsafe { crate::sytrd::dlarfg::<crate::flavor::RefBlas>(n, alpha, v) })
    }

    /// `DNRM2( N, X, 1 )`.
    pub fn dnrm2(x: &mut [f64]) -> f64 {
        let n = x.len();
        if n == 0 {
            return 0.0;
        }
        // SAFETY: n = x.len().
        unsafe { crate::blas::dnrm2(n, V::new(x), 1) }
    }

    /// `DLARTG( F, G, C, S, R )`. Returns `(c, s, r)`.
    pub fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
        crate::aux::dlartg(f, g)
    }

    /// `DLAPY2( X, Y )`.
    pub fn dlapy2(x: f64, y: f64) -> f64 {
        crate::aux::dlapy2(x, y)
    }

    /// libgcc `__powidf2` (Fortran `x**n`, variable integer `n`).
    pub fn powi(x: f64, n: i32) -> f64 {
        crate::aux::powidf2(x, n)
    }

    /// `DGEMM( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )`.
    #[allow(clippy::too_many_arguments)]
    pub fn dgemm(
        transa: bool,
        transb: bool,
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
        if m == 0 || n == 0 {
            return;
        }
        let (ra, ca) = if transa { (k, m) } else { (m, k) };
        let (rb, cb) = if transb { (n, k) } else { (k, n) };
        assert!(lda >= ra.max(1) && a.len() >= lda * ca.saturating_sub(1) + ra.min(if ca == 0 { 0 } else { ra }));
        assert!(ldb >= rb.max(1) && b.len() >= ldb * cb.saturating_sub(1) + if cb == 0 { 0 } else { rb });
        assert!(ldc >= m && c.len() >= ldc * (n - 1) + m);
        let mut da = [0.0f64; 1];
        let mut db = [0.0f64; 1];
        let av = if a.is_empty() { V::new(&mut da) } else { V::new(a) };
        let bv = if b.is_empty() { V::new(&mut db) } else { V::new(b) };
        // SAFETY: shapes asserted above; the three slices are distinct.
        unsafe { crate::blas::dgemm(transa, transb, m, n, k, alpha, av, lda, bv, ldb, beta, V::new(c), ldc) }
    }

    /// `DLAEV2( A, B, C, RT1, RT2, CS1, SN1 )`: the eigen-decomposition of
    /// the symmetric `[[a, b], [b, c]]`. Returns `(rt1, rt2, cs1, sn1)` with
    /// `|rt1| >= |rt2|`; `[cs1, sn1]` is the eigenvector for `rt1`.
    pub fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
        crate::aux::dlaev2(a, b, c)
    }

    /// `DLAE2( A, B, C, RT1, RT2 )`: eigenvalues only of `[[a, b], [b, c]]`.
    pub fn dlae2(a: f64, b: f64, c: f64) -> (f64, f64) {
        crate::aux::dlae2(a, b, c)
    }

    /// `DGEMV( TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY )`.
    ///
    /// `a` is column-major with leading dimension `lda >= max(1, m)`.
    pub fn dgemv(
        trans: Trans,
        m: usize,
        n: usize,
        alpha: f64,
        a: &mut [f64],
        lda: usize,
        x: &mut [f64],
        incx: usize,
        beta: f64,
        y: &mut [f64],
        incy: usize,
    ) -> Result<(), LapackError> {
        if m == 0 || n == 0 {
            return Ok(());
        }
        if lda < m.max(1) {
            return Err(LapackError::InvalidArgument { arg: "lda", reason: format!("need lda >= m = {m}") });
        }
        if incx == 0 || incy == 0 {
            return Err(LapackError::InvalidArgument { arg: "incx/incy", reason: "must be positive".into() });
        }
        check("a", a.len(), lda * (n - 1) + m)?;
        let tr = !matches!(trans, Trans::NoTrans);
        let (lenx, leny) = if tr { (m, n) } else { (n, m) };
        check("x", x.len(), (lenx - 1) * incx + 1)?;
        check("y", y.len(), (leny - 1) * incy + 1)?;
        // SAFETY: shapes checked above; the three slices are distinct.
        unsafe {
            crate::blas::dgemv(tr, m, n, alpha, V::new(a), lda, V::new(x), incx, beta, V::new(y), incy);
        }
        Ok(())
    }

    /// `ZGEMV( TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY )` on
    /// planar complex slices (separate real and imaginary parts, the
    /// convention [`crate::zheevd`] uses).
    pub fn zgemv(
        trans: Trans,
        m: usize,
        n: usize,
        alpha: (f64, f64),
        a_re: &mut [f64],
        a_im: &mut [f64],
        lda: usize,
        x_re: &mut [f64],
        x_im: &mut [f64],
        incx: usize,
        beta: (f64, f64),
        y_re: &mut [f64],
        y_im: &mut [f64],
        incy: usize,
    ) -> Result<(), LapackError> {
        use crate::zblas::{Tr, Z, ZV};
        if m == 0 || n == 0 {
            return Ok(());
        }
        if lda < m.max(1) {
            return Err(LapackError::InvalidArgument { arg: "lda", reason: format!("need lda >= m = {m}") });
        }
        if incx == 0 || incy == 0 {
            return Err(LapackError::InvalidArgument { arg: "incx/incy", reason: "must be positive".into() });
        }
        let tr = match trans {
            Trans::NoTrans => Tr::N,
            Trans::Trans => Tr::T,
            Trans::ConjTrans => Tr::C,
        };
        let (lenx, leny) = if tr == Tr::N { (n, m) } else { (m, n) };
        check("a.re", a_re.len(), lda * (n - 1) + m)?;
        check("a.im", a_im.len(), lda * (n - 1) + m)?;
        check("x.re", x_re.len(), (lenx - 1) * incx + 1)?;
        check("x.im", x_im.len(), (lenx - 1) * incx + 1)?;
        check("y.re", y_re.len(), (leny - 1) * incy + 1)?;
        check("y.im", y_im.len(), (leny - 1) * incy + 1)?;
        let av = ZV::new(V::new(a_re), V::new(a_im));
        let xv = ZV::new(V::new(x_re), V::new(x_im));
        let yv = ZV::new(V::new(y_re), V::new(y_im));
        let al = Z { re: alpha.0, im: alpha.1 };
        let be = Z { re: beta.0, im: beta.1 };
        // SAFETY: shapes checked above; the six slices are distinct.
        unsafe { crate::zblas::zgemv(tr, m, n, al, av, lda, xv, incx, be, yv, incy) }
        Ok(())
    }

    /// `DSYEVD` / `DSTERF` building-block access by job (values only uses
    /// `DSTERF`).
    pub fn dsterf(n: usize, d: &mut [f64], e: &mut [f64]) -> Result<(), LapackError> {
        let _ = Job::ValuesOnly;
        crate::steqr::dsterf(n, d, e)
    }
}
