//! Householder tridiagonalisation and its back-transformation.
//!
//! Transcribed from Reference-LAPACK v3.12.1 `SRC/dlarfg.f`, `SRC/dsytd2.f`,
//! `SRC/dlatrd.f`, `SRC/dsytrd.f`, `SRC/iladlc.f`, `SRC/dlarf1f.f`,
//! `SRC/dlarf1l.f`, `SRC/dorm2l.f`, `SRC/dorm2r.f`, `SRC/dlarft.f`
//! (recursive, 3.12.1), `SRC/dlarfb.f`, `SRC/dormql.f`, `SRC/dormqr.f` and
//! `SRC/dormtr.f`. The multiplication routines are transcribed for
//! `SIDE = 'L'`, `TRANS = 'N'`, the only combination `DSYEVD` uses.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::aux::{EPS, SAFMIN, dlapy2, sign};
use crate::blas::dcopy;
use crate::flavor::{Blas, RefBlas};
use crate::ptr::{D, ix};
use std::marker::PhantomData;

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;
const HALF: f64 = 0.5;

/// `ILAENV( 1, 'DSYTRD', ... )`.
pub(crate) const SYTRD_NB: usize = 32;
/// `ILAENV( 3, 'DSYTRD', ... )`.
pub(crate) const SYTRD_NX: usize = 32;
/// `ILAENV( 2, 'DSYTRD', ... )`.
pub(crate) const SYTRD_NBMIN: usize = 2;
/// `ILAENV( 1, 'DORMQL'/'DORMQR', ... )`.
pub(crate) const ORM_NB: usize = 32;
/// `NBMAX`, `LDT`, `TSIZE` of `DORMQL`/`DORMQR`.
pub(crate) const NBMAX: usize = 64;
pub(crate) const LDT: usize = NBMAX + 1;
pub(crate) const TSIZE: usize = LDT * NBMAX;

/// `DLARFG` with `INCX = 1`. Returns `(alpha, tau)`.
pub(crate) unsafe fn dlarfg<B: Blas>(n: usize, alpha: f64, x: D) -> (f64, f64) {
    let mut alpha = alpha;
    if n <= 1 {
        return (alpha, ZERO);
    }
    let mut xnorm = B::dnrm2(n - 1, x, 1);
    if xnorm == ZERO {
        return (alpha, ZERO);
    }
    let mut beta = -sign(dlapy2(alpha, xnorm), alpha);
    let safmin = SAFMIN / EPS;
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = ONE / safmin;
        loop {
            knt += 1;
            B::dscal(n - 1, rsafmn, x, 1);
            beta *= rsafmn;
            alpha *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnorm = B::dnrm2(n - 1, x, 1);
        beta = -sign(dlapy2(alpha, xnorm), alpha);
    }
    let tau = (beta - alpha) / beta;
    B::dscal(n - 1, ONE / (alpha - beta), x, 1);
    for _ in 0..knt {
        beta *= safmin;
    }
    (beta, tau)
}

/// `DSYTD2`.
pub(crate) unsafe fn dsytd2<B: Blas>(upper: bool, n: usize, a: D, lda: usize, d: D, e: D, tau: D) {
    if n == 0 {
        return;
    }
    let at = |i: usize, j: usize| ix(i, j, lda);
    if upper {
        for i in (1..n).rev() {
            let (alpha, taui) = dlarfg::<B>(i, a.at(at(i, i + 1)), a.off(at(1, i + 1)));
            a.set(at(i, i + 1), alpha);
            e.set(i - 1, a.at(at(i, i + 1)));
            if taui != ZERO {
                a.set(at(i, i + 1), ONE);
                B::dsymv(true, i, taui, a, lda, a.off(at(1, i + 1)), ZERO, tau);
                let alpha = -HALF * taui * B::ddot(i, tau, 1, a.off(at(1, i + 1)), 1);
                B::daxpy(i, alpha, a.off(at(1, i + 1)), 1, tau, 1);
                B::dsyr2(true, i, -ONE, a.off(at(1, i + 1)), tau, a, lda);
                a.set(at(i, i + 1), e.at(i - 1));
            }
            d.set(i, a.at(at(i + 1, i + 1)));
            tau.set(i - 1, taui);
        }
        d.set(0, a.at(at(1, 1)));
    } else {
        for i in 1..n {
            let (alpha, taui) = dlarfg::<B>(n - i, a.at(at(i + 1, i)), a.off(at((i + 2).min(n), i)));
            a.set(at(i + 1, i), alpha);
            e.set(i - 1, a.at(at(i + 1, i)));
            if taui != ZERO {
                a.set(at(i + 1, i), ONE);
                B::dsymv(false, n - i, taui, a.off(at(i + 1, i + 1)), lda, a.off(at(i + 1, i)), ZERO, tau.off(i - 1));
                let alpha = -HALF * taui * B::ddot(n - i, tau.off(i - 1), 1, a.off(at(i + 1, i)), 1);
                B::daxpy(n - i, alpha, a.off(at(i + 1, i)), 1, tau.off(i - 1), 1);
                B::dsyr2(false, n - i, -ONE, a.off(at(i + 1, i)), tau.off(i - 1), a.off(at(i + 1, i + 1)), lda);
                a.set(at(i + 1, i), e.at(i - 1));
            }
            d.set(i - 1, a.at(at(i, i)));
            tau.set(i - 1, taui);
        }
        d.set(n - 1, a.at(at(n, n)));
    }
}

/// Where the matrix-sized operations of `DSYTRD` run: the host reference BLAS
/// ([`HostTrd`]) or a device (`device::sytrd`). Everything else — `DLARFG`, the
/// `DGEMV` panel corrections, `DSYTD2` on the tail — stays on the host, so a
/// single transcription drives both paths.
pub(crate) trait TrdEngine {
    /// The BLAS flavour of the host-side calls (`DLARFG`, `DGEMV`, ...).
    type B: Blas;
    /// Before `DLATRD` on the panel starting at global column `i` (1-based).
    unsafe fn panel(&mut self, _upper: bool, _n: usize, _i: usize, _nb: usize, _a: D, _lda: usize) {}
    /// `DSYMV( UPLO, M, ALPHA, A(G0+1,G0+1), LDA, X, 1, ZERO, Y, 1 )`, where
    /// `blk` is the host view of that block and `g0` its 0-based global offset.
    unsafe fn symv(&mut self, upper: bool, m: usize, alpha: f64, g0: usize, blk: D, lda: usize, x: D, y: D);
    /// `DSYR2K( UPLO, 'N', M, NB, -ONE, V, LDA, W, LDW, ONE, C, LDA )` with
    /// `C = A(G0+1, G0+1)`.
    unsafe fn syr2k(&mut self, upper: bool, m: usize, nb: usize, g0: usize, v: D, lda: usize, w: D, ldw: usize, c: D);
    /// Before `DSYTD2` on the tail block `A(G0+1, G0+1)` of order `mt`.
    unsafe fn tail(&mut self, _upper: bool, _n: usize, _g0: usize, _mt: usize, _a: D, _lda: usize) {}
}

/// The BLAS flavour `B` on host memory.
pub(crate) struct HostTrd<B>(pub(crate) PhantomData<B>);

impl<B: Blas> HostTrd<B> {
    pub(crate) const fn new() -> Self {
        HostTrd(PhantomData)
    }
}

impl<B: Blas> TrdEngine for HostTrd<B> {
    type B = B;
    #[inline(always)]
    unsafe fn symv(&mut self, upper: bool, m: usize, alpha: f64, _g0: usize, blk: D, lda: usize, x: D, y: D) {
        B::dsymv(upper, m, alpha, blk, lda, x, ZERO, y);
    }
    #[inline(always)]
    unsafe fn syr2k(&mut self, upper: bool, m: usize, nb: usize, _g0: usize, v: D, lda: usize, w: D, ldw: usize, c: D) {
        B::dsyr2k_n(upper, m, nb, -ONE, v, lda, w, ldw, ONE, c, lda);
    }
}

/// `DLATRD`. `base` is the 0-based global offset of this call's `A(1,1)`.
pub(crate) unsafe fn dlatrd<E: TrdEngine>(
    eng: &mut E,
    base: usize,
    upper: bool,
    n: usize,
    nb: usize,
    a: D,
    lda: usize,
    e: D,
    tau: D,
    w: D,
    ldw: usize,
) {
    if n == 0 {
        return;
    }
    let at = |i: usize, j: usize| ix(i, j, lda);
    let wt = |i: usize, j: usize| ix(i, j, ldw);
    if upper {
        for i in ((n - nb + 1)..=n).rev() {
            let iw = i + nb - n;
            if i < n {
                E::B::dgemv(
                    false,
                    i,
                    n - i,
                    -ONE,
                    a.off(at(1, i + 1)),
                    lda,
                    w.off(wt(i, iw + 1)),
                    ldw,
                    ONE,
                    a.off(at(1, i)),
                    1,
                );
                E::B::dgemv(
                    false,
                    i,
                    n - i,
                    -ONE,
                    w.off(wt(1, iw + 1)),
                    ldw,
                    a.off(at(i, i + 1)),
                    lda,
                    ONE,
                    a.off(at(1, i)),
                    1,
                );
            }
            if i > 1 {
                let (alpha, t) = dlarfg::<E::B>(i - 1, a.at(at(i - 1, i)), a.off(at(1, i)));
                a.set(at(i - 1, i), alpha);
                tau.set(i - 2, t);
                e.set(i - 2, a.at(at(i - 1, i)));
                a.set(at(i - 1, i), ONE);
                eng.symv(true, i - 1, ONE, base, a, lda, a.off(at(1, i)), w.off(wt(1, iw)));
                if i < n {
                    E::B::dgemv(
                        true,
                        i - 1,
                        n - i,
                        ONE,
                        w.off(wt(1, iw + 1)),
                        ldw,
                        a.off(at(1, i)),
                        1,
                        ZERO,
                        w.off(wt(i + 1, iw)),
                        1,
                    );
                    E::B::dgemv(
                        false,
                        i - 1,
                        n - i,
                        -ONE,
                        a.off(at(1, i + 1)),
                        lda,
                        w.off(wt(i + 1, iw)),
                        1,
                        ONE,
                        w.off(wt(1, iw)),
                        1,
                    );
                    E::B::dgemv(
                        true,
                        i - 1,
                        n - i,
                        ONE,
                        a.off(at(1, i + 1)),
                        lda,
                        a.off(at(1, i)),
                        1,
                        ZERO,
                        w.off(wt(i + 1, iw)),
                        1,
                    );
                    E::B::dgemv(
                        false,
                        i - 1,
                        n - i,
                        -ONE,
                        w.off(wt(1, iw + 1)),
                        ldw,
                        w.off(wt(i + 1, iw)),
                        1,
                        ONE,
                        w.off(wt(1, iw)),
                        1,
                    );
                }
                E::B::dscal(i - 1, tau.at(i - 2), w.off(wt(1, iw)), 1);
                let alpha = -HALF * tau.at(i - 2) * E::B::ddot(i - 1, w.off(wt(1, iw)), 1, a.off(at(1, i)), 1);
                E::B::daxpy(i - 1, alpha, a.off(at(1, i)), 1, w.off(wt(1, iw)), 1);
            }
        }
    } else {
        for i in 1..=nb {
            E::B::dgemv(
                false,
                n - i + 1,
                i - 1,
                -ONE,
                a.off(at(i, 1)),
                lda,
                w.off(wt(i, 1)),
                ldw,
                ONE,
                a.off(at(i, i)),
                1,
            );
            E::B::dgemv(
                false,
                n - i + 1,
                i - 1,
                -ONE,
                w.off(wt(i, 1)),
                ldw,
                a.off(at(i, 1)),
                lda,
                ONE,
                a.off(at(i, i)),
                1,
            );
            if i < n {
                let (alpha, t) = dlarfg::<E::B>(n - i, a.at(at(i + 1, i)), a.off(at((i + 2).min(n), i)));
                a.set(at(i + 1, i), alpha);
                tau.set(i - 1, t);
                e.set(i - 1, a.at(at(i + 1, i)));
                a.set(at(i + 1, i), ONE);
                eng.symv(
                    false,
                    n - i,
                    ONE,
                    base + i,
                    a.off(at(i + 1, i + 1)),
                    lda,
                    a.off(at(i + 1, i)),
                    w.off(wt(i + 1, i)),
                );
                E::B::dgemv(
                    true,
                    n - i,
                    i - 1,
                    ONE,
                    w.off(wt(i + 1, 1)),
                    ldw,
                    a.off(at(i + 1, i)),
                    1,
                    ZERO,
                    w.off(wt(1, i)),
                    1,
                );
                E::B::dgemv(
                    false,
                    n - i,
                    i - 1,
                    -ONE,
                    a.off(at(i + 1, 1)),
                    lda,
                    w.off(wt(1, i)),
                    1,
                    ONE,
                    w.off(wt(i + 1, i)),
                    1,
                );
                E::B::dgemv(
                    true,
                    n - i,
                    i - 1,
                    ONE,
                    a.off(at(i + 1, 1)),
                    lda,
                    a.off(at(i + 1, i)),
                    1,
                    ZERO,
                    w.off(wt(1, i)),
                    1,
                );
                E::B::dgemv(
                    false,
                    n - i,
                    i - 1,
                    -ONE,
                    w.off(wt(i + 1, 1)),
                    ldw,
                    w.off(wt(1, i)),
                    1,
                    ONE,
                    w.off(wt(i + 1, i)),
                    1,
                );
                E::B::dscal(n - i, tau.at(i - 1), w.off(wt(i + 1, i)), 1);
                let alpha = -HALF * tau.at(i - 1) * E::B::ddot(n - i, w.off(wt(i + 1, i)), 1, a.off(at(i + 1, i)), 1);
                E::B::daxpy(n - i, alpha, a.off(at(i + 1, i)), 1, w.off(wt(i + 1, i)), 1);
            }
        }
    }
}

/// `DSYTRD` blocking decision for a given `LWORK`: `(nb, nx, ldwork)`.
pub(crate) fn sytrd_blocking(n: usize, lwork: usize) -> (usize, usize) {
    let mut nb = SYTRD_NB;
    let mut nx = n;
    if nb > 1 && nb < n {
        nx = nb.max(SYTRD_NX);
        if nx < n {
            let ldwork = n;
            let iws = ldwork * nb;
            if lwork < iws {
                nb = (lwork / ldwork).max(1);
                if nb < SYTRD_NBMIN {
                    nx = n;
                }
            }
        } else {
            nx = n;
        }
    } else {
        nb = 1;
    }
    (nb, nx)
}

/// `DSYTRD`. `work` is `WORK(1:LWORK)`; `lwork >= 1`.
pub(crate) unsafe fn dsytrd(upper: bool, n: usize, a: D, lda: usize, d: D, e: D, tau: D, work: D, lwork: usize) {
    dsytrd_with(&mut HostTrd::<RefBlas>::new(), upper, n, a, lda, d, e, tau, work, lwork);
}

/// `DSYTRD` with its matrix-sized operations routed through `eng`.
pub(crate) unsafe fn dsytrd_with<E: TrdEngine>(
    eng: &mut E,
    upper: bool,
    n: usize,
    a: D,
    lda: usize,
    d: D,
    e: D,
    tau: D,
    work: D,
    lwork: usize,
) {
    if n == 0 {
        return;
    }
    let (nb, nx) = sytrd_blocking(n, lwork);
    let ldwork = n;
    if upper {
        let kk = n - ((n - nx + nb - 1) / nb) * nb;
        // DO 20 I = N - NB + 1, KK + 1, -NB
        let start = n as isize - nb as isize + 1;
        let stop = kk as isize + 1;
        let mut i = start;
        while i >= stop {
            let iu = i as usize;
            eng.panel(true, n, iu, nb, a, lda);
            dlatrd(eng, 0, true, iu + nb - 1, nb, a, lda, e, tau, work, ldwork);
            eng.syr2k(true, iu - 1, nb, 0, a.off(ix(1, iu, lda)), lda, work, ldwork, a);
            for j in iu..=(iu + nb - 1) {
                a.set(ix(j - 1, j, lda), e.at(j - 2));
                d.set(j - 1, a.at(ix(j, j, lda)));
            }
            i -= nb as isize;
        }
        eng.tail(true, n, 0, kk, a, lda);
        dsytd2::<E::B>(true, kk, a, lda, d, e, tau);
    } else {
        // DO 40 I = 1, N - NX, NB
        let mut i = 1usize;
        if n > nx {
            while i <= n - nx {
                eng.panel(false, n, i, nb, a, lda);
                dlatrd(
                    eng,
                    i - 1,
                    false,
                    n - i + 1,
                    nb,
                    a.off(ix(i, i, lda)),
                    lda,
                    e.off(i - 1),
                    tau.off(i - 1),
                    work,
                    ldwork,
                );
                eng.syr2k(
                    false,
                    n - i - nb + 1,
                    nb,
                    i + nb - 1,
                    a.off(ix(i + nb, i, lda)),
                    lda,
                    work.off(nb),
                    ldwork,
                    a.off(ix(i + nb, i + nb, lda)),
                );
                for j in i..=(i + nb - 1) {
                    a.set(ix(j + 1, j, lda), e.at(j - 1));
                    d.set(j - 1, a.at(ix(j, j, lda)));
                }
                i += nb;
            }
        }
        eng.tail(false, n, i - 1, n - i + 1, a, lda);
        dsytd2::<E::B>(false, n - i + 1, a.off(ix(i, i, lda)), lda, d.off(i - 1), e.off(i - 1), tau.off(i - 1));
    }
}

/// `ILADLC`.
unsafe fn iladlc(m: usize, n: usize, a: D, lda: usize) -> usize {
    if n == 0 {
        return n;
    }
    if a.at(ix(1, n, lda)) != ZERO || a.at(ix(m, n, lda)) != ZERO {
        return n;
    }
    for c in (1..=n).rev() {
        for i in 1..=m {
            if a.at(ix(i, c, lda)) != ZERO {
                return c;
            }
        }
    }
    0
}

/// `DLARF1F( 'L', M, N, V, 1, TAU, C, LDC, WORK )`: `V(1)` is implicitly 1.
unsafe fn dlarf1f_left<B: Blas>(m: usize, n: usize, v: D, tau: f64, c: D, ldc: usize, work: D) {
    let mut lastv = 1usize;
    let mut lastc = 0usize;
    if tau != ZERO {
        lastv = m;
        let mut i = lastv;
        while lastv > 1 && v.at(i - 1) == ZERO {
            lastv -= 1;
            i -= 1;
        }
        lastc = iladlc(lastv, n, c, ldc);
    }
    if lastc == 0 {
        return;
    }
    if lastv == 1 {
        B::dscal(lastc, ONE - tau, c, ldc);
    } else {
        B::dgemv(true, lastv - 1, lastc, ONE, c.off(1), ldc, v.off(1), 1, ZERO, work, 1);
        B::daxpy(lastc, ONE, c, ldc, work, 1);
        B::daxpy(lastc, -tau, work, 1, c, ldc);
        B::dger(lastv - 1, lastc, -tau, v.off(1), 1, work, 1, c.off(1), ldc);
    }
}

/// `DLARF1L( 'L', M, N, V, 1, TAU, C, LDC, WORK )`: `V(M)` is implicitly 1.
unsafe fn dlarf1l_left<B: Blas>(m: usize, n: usize, v: D, tau: f64, c: D, ldc: usize, work: D) {
    let mut firstv = 1usize;
    let mut lastc = 0usize;
    let mut lastv = 0usize;
    let mut i = 1usize;
    if tau != ZERO {
        lastv = m;
        while lastv > firstv && v.at(i - 1) == ZERO {
            firstv += 1;
            i += 1;
        }
        lastc = iladlc(lastv, n, c, ldc);
    }
    if lastc == 0 {
        return;
    }
    if lastv > 0 {
        if lastv == firstv {
            B::dscal(lastc, ONE - tau, c.off(firstv - 1), ldc);
        } else {
            B::dgemv(true, lastv - firstv, lastc, ONE, c.off(firstv - 1), ldc, v.off(i - 1), 1, ZERO, work, 1);
            B::daxpy(lastc, ONE, c.off(lastv - 1), ldc, work, 1);
            B::daxpy(lastc, -tau, work, 1, c.off(lastv - 1), ldc);
            B::dger(lastv - firstv, lastc, -tau, v.off(i - 1), 1, work, 1, c.off(firstv - 1), ldc);
        }
    }
}

/// `DLARF( 'L', M, N, V, 1, TAU, C, LDC, WORK )` as LAPACK 3.12.0 (and
/// OpenBLAS 0.3.30's vendored copy) writes it: `V` explicit, `LASTV` scanned
/// from the end, then `DGEMV` + `DGER`. Used when `B::LEGACY_DLARF`.
unsafe fn dlarf_left<B: Blas>(m: usize, n: usize, v: D, tau: f64, c: D, ldc: usize, work: D) {
    let mut lastv = 0usize;
    let mut lastc = 0usize;
    if tau != ZERO {
        lastv = m;
        while lastv > 0 && v.at(lastv - 1) == ZERO {
            lastv -= 1;
        }
        lastc = iladlc(lastv, n, c, ldc);
    }
    if lastv > 0 {
        B::dgemv(true, lastv, lastc, ONE, c, ldc, v, 1, ZERO, work, 1);
        B::dger(lastv, lastc, -tau, v, 1, work, 1, c, ldc);
    }
}

/// `DORM2L( 'L', 'N', ... )`.
unsafe fn dorm2l_ln<B: Blas>(m: usize, n: usize, k: usize, a: D, lda: usize, tau: D, c: D, ldc: usize, work: D) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    for i in 1..=k {
        let mi = m - k + i;
        if B::LEGACY_DLARF {
            // AII = A( NQ-K+I, I ); A( NQ-K+I, I ) = ONE; DLARF; restore.
            let p = ix(m - k + i, i, lda);
            let aii = a.at(p);
            a.set(p, ONE);
            dlarf_left::<B>(mi, n, a.off(ix(1, i, lda)), tau.at(i - 1), c, ldc, work);
            a.set(p, aii);
        } else {
            dlarf1l_left::<B>(mi, n, a.off(ix(1, i, lda)), tau.at(i - 1), c, ldc, work);
        }
    }
}

/// `DORM2R( 'L', 'N', ... )`.
unsafe fn dorm2r_ln<B: Blas>(m: usize, n: usize, k: usize, a: D, lda: usize, tau: D, c: D, ldc: usize, work: D) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    for i in (1..=k).rev() {
        let mi = m - i + 1;
        if B::LEGACY_DLARF {
            // AII = A( I, I ); A( I, I ) = ONE; DLARF; restore.
            let p = ix(i, i, lda);
            let aii = a.at(p);
            a.set(p, ONE);
            dlarf_left::<B>(mi, n, a.off(p), tau.at(i - 1), c.off(ix(i, 1, ldc)), ldc, work);
            a.set(p, aii);
        } else {
            dlarf1f_left::<B>(mi, n, a.off(ix(i, i, lda)), tau.at(i - 1), c.off(ix(i, 1, ldc)), ldc, work);
        }
    }
}

/// `DLARFT( DIRECT, 'Columnwise', ... )` (recursive, 3.12.1).
/// `forward = true` is `'F'` (QR), `false` is `'B'` (QL).
pub(crate) unsafe fn dlarft_c<B: Blas>(forward: bool, n: usize, k: usize, v: D, ldv: usize, tau: D, t: D, ldt: usize) {
    if n == 0 || k == 0 {
        return;
    }
    if n == 1 || k == 1 {
        t.set(0, tau.at(0));
        return;
    }
    let l = k / 2;
    let vi = |i: usize, j: usize| ix(i, j, ldv);
    let ti = |i: usize, j: usize| ix(i, j, ldt);
    if forward {
        dlarft_c::<B>(true, n, l, v, ldv, tau, t, ldt);
        dlarft_c::<B>(true, n - l, k - l, v.off(vi(l + 1, l + 1)), ldv, tau.off(l), t.off(ti(l + 1, l + 1)), ldt);
        for j in 1..=l {
            for i in 1..=(k - l) {
                t.set(ti(j, l + i), v.at(vi(l + i, j)));
            }
        }
        B::dtrmm(false, false, false, false, l, k - l, ONE, v.off(vi(l + 1, l + 1)), ldv, t.off(ti(1, l + 1)), ldt);
        B::dgemm(
            true,
            false,
            l,
            k - l,
            n - k,
            ONE,
            v.off(vi(k + 1, 1)),
            ldv,
            v.off(vi(k + 1, l + 1)),
            ldv,
            ONE,
            t.off(ti(1, l + 1)),
            ldt,
        );
        B::dtrmm(true, true, false, true, l, k - l, -ONE, t, ldt, t.off(ti(1, l + 1)), ldt);
        B::dtrmm(false, true, false, true, l, k - l, ONE, t.off(ti(l + 1, l + 1)), ldt, t.off(ti(1, l + 1)), ldt);
    } else {
        dlarft_c::<B>(false, n - l, k - l, v, ldv, tau, t, ldt);
        dlarft_c::<B>(false, n, l, v.off(vi(1, k - l + 1)), ldv, tau.off(k - l), t.off(ti(k - l + 1, k - l + 1)), ldt);
        for j in 1..=(k - l) {
            for i in 1..=l {
                t.set(ti(k - l + i, j), v.at(vi(n - k + j, k - l + i)));
            }
        }
        B::dtrmm(false, true, false, false, l, k - l, ONE, v.off(vi(n - k + 1, 1)), ldv, t.off(ti(k - l + 1, 1)), ldt);
        B::dgemm(
            true,
            false,
            l,
            k - l,
            n - k,
            ONE,
            v.off(vi(1, k - l + 1)),
            ldv,
            v,
            ldv,
            ONE,
            t.off(ti(k - l + 1, 1)),
            ldt,
        );
        B::dtrmm(
            true,
            false,
            false,
            true,
            l,
            k - l,
            -ONE,
            t.off(ti(k - l + 1, k - l + 1)),
            ldt,
            t.off(ti(k - l + 1, 1)),
            ldt,
        );
        B::dtrmm(false, false, false, true, l, k - l, ONE, t, ldt, t.off(ti(k - l + 1, 1)), ldt);
    }
}

/// `DLARFB( 'L', 'N', DIRECT, 'Columnwise', ... )`.
pub(crate) unsafe fn dlarfb_lnc<B: Blas>(
    forward: bool,
    m: usize,
    n: usize,
    k: usize,
    v: D,
    ldv: usize,
    t: D,
    ldt: usize,
    c: D,
    ldc: usize,
    work: D,
    ldwork: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    // TRANS = 'N' so TRANST = 'T'.
    let wk = |i: usize, j: usize| ix(i, j, ldwork);
    if forward {
        for j in 1..=k {
            dcopy(n, c.off(ix(j, 1, ldc)), ldc, work.off(wk(1, j)), 1);
        }
        B::dtrmm(false, false, false, false, n, k, ONE, v, ldv, work, ldwork);
        if m > k {
            B::dgemm(
                true,
                false,
                n,
                k,
                m - k,
                ONE,
                c.off(ix(k + 1, 1, ldc)),
                ldc,
                v.off(ix(k + 1, 1, ldv)),
                ldv,
                ONE,
                work,
                ldwork,
            );
        }
        B::dtrmm(false, true, true, true, n, k, ONE, t, ldt, work, ldwork);
        if m > k {
            B::dgemm(
                false,
                true,
                m - k,
                n,
                k,
                -ONE,
                v.off(ix(k + 1, 1, ldv)),
                ldv,
                work,
                ldwork,
                ONE,
                c.off(ix(k + 1, 1, ldc)),
                ldc,
            );
        }
        B::dtrmm(false, false, true, false, n, k, ONE, v, ldv, work, ldwork);
        for j in 1..=k {
            for i in 1..=n {
                let p = ix(j, i, ldc);
                c.set(p, c.at(p) - work.at(wk(i, j)));
            }
        }
    } else {
        for j in 1..=k {
            dcopy(n, c.off(ix(m - k + j, 1, ldc)), ldc, work.off(wk(1, j)), 1);
        }
        B::dtrmm(false, true, false, false, n, k, ONE, v.off(ix(m - k + 1, 1, ldv)), ldv, work, ldwork);
        if m > k {
            B::dgemm(true, false, n, k, m - k, ONE, c, ldc, v, ldv, ONE, work, ldwork);
        }
        B::dtrmm(false, false, true, true, n, k, ONE, t, ldt, work, ldwork);
        if m > k {
            B::dgemm(false, true, m - k, n, k, -ONE, v, ldv, work, ldwork, ONE, c, ldc);
        }
        B::dtrmm(false, true, true, false, n, k, ONE, v.off(ix(m - k + 1, 1, ldv)), ldv, work, ldwork);
        for j in 1..=k {
            for i in 1..=n {
                let p = ix(m - k + j, i, ldc);
                c.set(p, c.at(p) - work.at(wk(i, j)));
            }
        }
    }
}

/// Integer division truncating toward zero, as Fortran does.
fn fdiv(a: isize, b: isize) -> isize {
    a / b
}

/// Effective block size of `DORMQL`/`DORMQR` (`'L'`, `'N'`) for `LWORK`, or
/// `None` for the unblocked path.
pub(crate) fn orm_blocking(n: usize, k: usize, lwork: usize) -> Option<usize> {
    let nw = n.max(1);
    let mut nb = ORM_NB.min(NBMAX) as isize;
    let lwkopt = nw * ORM_NB + TSIZE;
    let nbmin = 2isize;
    if nb > 1 && (nb as usize) < k && lwork < lwkopt {
        nb = fdiv(lwork as isize - TSIZE as isize, nw as isize);
    }
    if nb < nbmin || nb as usize >= k { None } else { Some(nb as usize) }
}

/// Where `DLARFB` runs inside `DORMQL`/`DORMQR` (the `T` factor from
/// `DLARFT` is always built on the host).
pub(crate) trait OrmEngine {
    /// The BLAS flavour of `DLARFT` and the unblocked path.
    type B: Blas;
    /// `DLARFB( 'L', 'N', DIRECT, 'C', M, N, K, V, LDV, T, LDT, C, LDC, WORK, LDWORK )`
    /// with `C(1,1)` at 0-based global row `crow` of the full `C`.
    unsafe fn larfb(
        &mut self,
        forward: bool,
        m: usize,
        n: usize,
        k: usize,
        v: D,
        ldv: usize,
        t: D,
        ldt: usize,
        c: D,
        crow: usize,
        ldc: usize,
        work: D,
        ldwork: usize,
    );
}

/// The BLAS flavour `B` on host memory.
pub(crate) struct HostOrm<B>(pub(crate) PhantomData<B>);

impl<B: Blas> HostOrm<B> {
    pub(crate) const fn new() -> Self {
        HostOrm(PhantomData)
    }
}

impl<B: Blas> OrmEngine for HostOrm<B> {
    type B = B;
    #[inline(always)]
    unsafe fn larfb(
        &mut self,
        forward: bool,
        m: usize,
        n: usize,
        k: usize,
        v: D,
        ldv: usize,
        t: D,
        ldt: usize,
        c: D,
        _crow: usize,
        ldc: usize,
        work: D,
        ldwork: usize,
    ) {
        dlarfb_lnc::<B>(forward, m, n, k, v, ldv, t, ldt, c, ldc, work, ldwork);
    }
}

/// `DORMQL( 'L', 'N', ... )`; `crow` is the global row of `C(1,1)`.
unsafe fn dormql_ln<E: OrmEngine>(
    eng: &mut E,
    m: usize,
    n: usize,
    k: usize,
    a: D,
    lda: usize,
    tau: D,
    c: D,
    crow: usize,
    ldc: usize,
    work: D,
    lwork: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let nq = m;
    let nw = n.max(1);
    match orm_blocking(n, k, lwork) {
        None => dorm2l_ln::<E::B>(m, n, k, a, lda, tau, c, ldc, work),
        Some(nb) => {
            let iwt = nw * nb; // 0-based WORK(IWT)
            let mut i = 1usize;
            while i <= k {
                let ib = nb.min(k - i + 1);
                dlarft_c::<E::B>(
                    false,
                    nq - k + i + ib - 1,
                    ib,
                    a.off(ix(1, i, lda)),
                    lda,
                    tau.off(i - 1),
                    work.off(iwt),
                    LDT,
                );
                let mi = m - k + i + ib - 1;
                eng.larfb(false, mi, n, ib, a.off(ix(1, i, lda)), lda, work.off(iwt), LDT, c, crow, ldc, work, nw);
                i += nb;
            }
        }
    }
}

/// `DORMQR( 'L', 'N', ... )`; `crow` is the global row of `C(1,1)`.
unsafe fn dormqr_ln<E: OrmEngine>(
    eng: &mut E,
    m: usize,
    n: usize,
    k: usize,
    a: D,
    lda: usize,
    tau: D,
    c: D,
    crow: usize,
    ldc: usize,
    work: D,
    lwork: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let nq = m;
    let nw = n.max(1);
    match orm_blocking(n, k, lwork) {
        None => dorm2r_ln::<E::B>(m, n, k, a, lda, tau, c, ldc, work),
        Some(nb) => {
            let iwt = nw * nb;
            let mut i = ((k - 1) / nb) * nb + 1;
            loop {
                let ib = nb.min(k - i + 1);
                dlarft_c::<E::B>(true, nq - i + 1, ib, a.off(ix(i, i, lda)), lda, tau.off(i - 1), work.off(iwt), LDT);
                let mi = m - i + 1;
                eng.larfb(
                    true,
                    mi,
                    n,
                    ib,
                    a.off(ix(i, i, lda)),
                    lda,
                    work.off(iwt),
                    LDT,
                    c.off(ix(i, 1, ldc)),
                    crow + i - 1,
                    ldc,
                    work,
                    nw,
                );
                if i <= nb {
                    break;
                }
                i -= nb;
            }
        }
    }
}

/// `DORMTR( 'L', UPLO, 'N', M, N, ... )`. `lwork >= max(1, n)`.
pub(crate) unsafe fn dormtr_ln(
    upper: bool,
    m: usize,
    n: usize,
    a: D,
    lda: usize,
    tau: D,
    c: D,
    ldc: usize,
    work: D,
    lwork: usize,
) {
    dormtr_ln_with(&mut HostOrm::<RefBlas>::new(), upper, m, n, a, lda, tau, c, ldc, work, lwork);
}

/// `DORMTR( 'L', UPLO, 'N', ... )` with `DLARFB` routed through `eng`. The
/// unblocked path (`orm_blocking` returns `None`) always runs on the host
/// views.
pub(crate) unsafe fn dormtr_ln_with<E: OrmEngine>(
    eng: &mut E,
    upper: bool,
    m: usize,
    n: usize,
    a: D,
    lda: usize,
    tau: D,
    c: D,
    ldc: usize,
    work: D,
    lwork: usize,
) {
    let nq = m;
    if m == 0 || n == 0 || nq == 1 {
        return;
    }
    let mi = m - 1;
    let ni = n;
    if upper {
        dormql_ln(eng, mi, ni, nq - 1, a.off(ix(1, 2, lda)), lda, tau, c, 0, ldc, work, lwork);
    } else {
        dormqr_ln(eng, mi, ni, nq - 1, a.off(ix(2, 1, lda)), lda, tau, c.off(ix(2, 1, ldc)), 1, ldc, work, lwork);
    }
}
