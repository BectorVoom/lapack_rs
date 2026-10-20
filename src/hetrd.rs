//! Hermitian tridiagonalisation and its back-transformation, planar storage.
//!
//! Transcribed from Reference-LAPACK v3.12.1 `SRC/dlapy3.f`, `SRC/dladiv.f`,
//! `SRC/zladiv.f`, `SRC/zlarfg.f`, `SRC/zhetd2.f`, `SRC/zlatrd.f`,
//! `SRC/zhetrd.f`, `SRC/ilazlc.f`, `SRC/zlarf1f.f`, `SRC/zlarf1l.f`,
//! `SRC/zunm2l.f`, `SRC/zunm2r.f`, `SRC/zlarft.f`, `SRC/zlarfb.f`,
//! `SRC/zunmql.f`, `SRC/zunmqr.f` and `SRC/zunmtr.f` (multiplication
//! routines for `SIDE = 'L'`, `TRANS = 'N'`).
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::aux::{EPS, HUGEVAL, SAFMIN, dlapy2, fmax, sign};
use crate::flavor::{Blas, RefBlas};
use crate::ptr::{D, ix};
use crate::sytrd::{LDT, orm_blocking, sytrd_blocking};
use crate::zblas::{CONE, CZERO, Tr, Z, ZV, zcopy, zlacgv};
use std::marker::PhantomData;

const HALF: Z = Z::new(0.5, 0.0);

/// Where the matrix-sized operations of `ZHETRD` run: the host reference BLAS
/// ([`HostZTrd`]) or a device (`device::complex`); the complex counterpart of
/// [`crate::sytrd::TrdEngine`].
pub(crate) trait ZTrdEngine {
    /// The BLAS flavour of the host-side calls.
    type B: Blas;
    /// Before `ZLATRD` on the panel starting at global column `i` (1-based).
    unsafe fn panel(&mut self, _upper: bool, _n: usize, _i: usize, _nb: usize, _a: ZV, _lda: usize) {}
    /// `ZHEMV( UPLO, M, CONE, A(G0+1,G0+1), LDA, X, 1, CZERO, Y, 1 )`.
    unsafe fn hemv(&mut self, upper: bool, m: usize, g0: usize, blk: ZV, lda: usize, x: ZV, y: ZV);
    /// `ZHER2K( UPLO, 'N', M, NB, -CONE, V, LDA, W, LDW, ONE, C, LDA )` with
    /// `C = A(G0+1, G0+1)`.
    unsafe fn her2k(
        &mut self,
        upper: bool,
        m: usize,
        nb: usize,
        g0: usize,
        v: ZV,
        lda: usize,
        w: ZV,
        ldw: usize,
        c: ZV,
    );
    /// Before `ZHETD2` on the tail block `A(G0+1, G0+1)` of order `mt`.
    unsafe fn tail(&mut self, _upper: bool, _n: usize, _g0: usize, _mt: usize, _a: ZV, _lda: usize) {}
}

/// The BLAS flavour `B` on host memory.
pub(crate) struct HostZTrd<B>(pub(crate) PhantomData<B>);

impl<B: Blas> HostZTrd<B> {
    pub(crate) const fn new() -> Self {
        HostZTrd(PhantomData)
    }
}

impl<B: Blas> ZTrdEngine for HostZTrd<B> {
    type B = B;
    #[inline(always)]
    unsafe fn hemv(&mut self, upper: bool, m: usize, _g0: usize, blk: ZV, lda: usize, x: ZV, y: ZV) {
        B::zhemv(upper, m, CONE, blk, lda, x, CZERO, y);
    }
    #[inline(always)]
    unsafe fn her2k(
        &mut self,
        upper: bool,
        m: usize,
        nb: usize,
        _g0: usize,
        v: ZV,
        lda: usize,
        w: ZV,
        ldw: usize,
        c: ZV,
    ) {
        B::zher2k_n(upper, m, nb, -CONE, v, lda, w, ldw, 1.0, c, lda);
    }
}

/// Where `ZLARFB('L','N',*,'C')` runs; see [`crate::sytrd::OrmEngine`].
/// `crow` is the 0-based global row of `C(1, ·)`.
pub(crate) trait ZOrmEngine {
    /// The BLAS flavour of `ZLARFT` and the unblocked path.
    type B: Blas;
    /// The legacy unblocked `ZLARF('L', M, N, V, 1, TAU, C, LDC, WORK)`
    /// (only reached when `B::LEGACY_DLARF`) with `C(1, 1)` at 0-based
    /// global row `crow` of the full `C`, run by the engine; `false` means
    /// the host transcription runs it on the host views.
    unsafe fn larf_left(&mut self, _m: usize, _n: usize, _v: ZV, _tau: Z, _crow: usize) -> bool {
        false
    }
    unsafe fn larfb(
        &mut self,
        forward: bool,
        m: usize,
        n: usize,
        k: usize,
        v: ZV,
        ldv: usize,
        t: ZV,
        ldt: usize,
        c: ZV,
        crow: usize,
        ldc: usize,
        work: ZV,
        ldwork: usize,
    );
}

/// `ZLARFB` in the BLAS flavour `B` on host memory.
pub(crate) struct HostZOrm<B>(pub(crate) PhantomData<B>);

impl<B: Blas> HostZOrm<B> {
    pub(crate) const fn new() -> Self {
        HostZOrm(PhantomData)
    }
}

impl<B: Blas> ZOrmEngine for HostZOrm<B> {
    type B = B;
    #[inline(always)]
    unsafe fn larfb(
        &mut self,
        forward: bool,
        m: usize,
        n: usize,
        k: usize,
        v: ZV,
        ldv: usize,
        t: ZV,
        ldt: usize,
        c: ZV,
        _crow: usize,
        ldc: usize,
        work: ZV,
        ldwork: usize,
    ) {
        zlarfb_lnc::<B>(forward, m, n, k, v, ldv, t, ldt, c, ldc, work, ldwork);
    }
}

/// `DLAPY3`.
pub(crate) fn dlapy3(x: f64, y: f64, z: f64) -> f64 {
    let xabs = x.abs();
    let yabs = y.abs();
    let zabs = z.abs();
    let w = fmax(fmax(xabs, yabs), zabs);
    if w == 0.0 || w > HUGEVAL {
        xabs + yabs + zabs
    } else {
        let (a, b, c) = (xabs / w, yabs / w, zabs / w);
        w * (a * a + b * b + c * c).sqrt()
    }
}

fn dladiv2(a: f64, b: f64, c: f64, d: f64, r: f64, t: f64) -> f64 {
    if r != 0.0 {
        let br = b * r;
        if br != 0.0 { (a + br) * t } else { a * t + (b * t) * r }
    } else {
        (a + d * (b / c)) * t
    }
}

fn dladiv1(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let r = d / c;
    let t = 1.0 / (c + d * r);
    let p = dladiv2(a, b, c, d, r, t);
    let a = -a;
    let q = dladiv2(b, a, c, d, r, t);
    (p, q)
}

/// `DLADIV( A, B, C, D, P, Q )`: robust `(a + ib) / (c + id)`.
pub(crate) fn dladiv(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    const BS: f64 = 2.0;
    let (mut aa, mut bb, mut cc, mut dd) = (a, b, c, d);
    let ab = fmax(a.abs(), b.abs());
    let cd = fmax(c.abs(), d.abs());
    let mut s = 1.0;
    let ov = HUGEVAL;
    let un = SAFMIN;
    let eps = EPS;
    let be = BS / (eps * eps);
    if ab >= 0.5 * ov {
        aa *= 0.5;
        bb *= 0.5;
        s *= 2.0;
    }
    if cd >= 0.5 * ov {
        cc *= 0.5;
        dd *= 0.5;
        s *= 0.5;
    }
    if ab <= un * BS / eps {
        aa *= be;
        bb *= be;
        s /= be;
    }
    if cd <= un * BS / eps {
        cc *= be;
        dd *= be;
        s *= be;
    }
    let (p, mut q);
    if d.abs() <= c.abs() {
        (p, q) = dladiv1(aa, bb, cc, dd);
    } else {
        (p, q) = dladiv1(bb, aa, dd, cc);
        q = -q;
    }
    (p * s, q * s)
}

/// `ZLADIV( X, Y )`.
pub(crate) fn zladiv(x: Z, y: Z) -> Z {
    let (zr, zi) = dladiv(x.re, x.im, y.re, y.im);
    Z::new(zr, zi)
}

/// `ZLARFG( N, ALPHA, X, 1, TAU )`. Returns `(alpha, tau)`.
pub(crate) unsafe fn zlarfg<B: Blas>(n: usize, alpha: Z, x: ZV) -> (Z, Z) {
    let mut alpha = alpha;
    if n == 0 {
        return (alpha, CZERO);
    }
    let mut xnorm = B::dznrm2(n - 1, x, 1);
    let mut alphr = alpha.re;
    let mut alphi = alpha.im;
    if xnorm == 0.0 && alphi == 0.0 {
        return (alpha, CZERO);
    }
    let mut beta = -sign(dlapy3(alphr, alphi, xnorm), alphr);
    let safmin = SAFMIN / EPS;
    let rsafmn = 1.0 / safmin;
    let mut knt = 0;
    if beta.abs() < safmin {
        loop {
            knt += 1;
            B::zdscal(n - 1, rsafmn, x, 1);
            beta *= rsafmn;
            alphi *= rsafmn;
            alphr *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnorm = B::dznrm2(n - 1, x, 1);
        alpha = Z::new(alphr, alphi);
        beta = -sign(dlapy3(alphr, alphi, xnorm), alphr);
    }
    let tau = Z::new((beta - alphr) / beta, -alphi / beta);
    alpha = zladiv(Z::r(1.0), alpha - Z::r(beta));
    B::zscal(n - 1, alpha, x, 1);
    for _ in 0..knt {
        beta *= safmin;
    }
    let _ = dlapy2;
    (Z::r(beta), tau)
}

/// `ZHETD2`.
pub(crate) unsafe fn zhetd2<B: Blas>(upper: bool, n: usize, a: ZV, lda: usize, d: D, e: D, tau: ZV) {
    if n == 0 {
        return;
    }
    let at = |i: usize, j: usize| ix(i, j, lda);
    if upper {
        a.set(at(n, n), Z::r(a.at(at(n, n)).re));
        for i in (1..n).rev() {
            let alpha = a.at(at(i, i + 1));
            let (alpha, taui) = zlarfg::<B>(i, alpha, a.off(at(1, i + 1)));
            e.set(i - 1, alpha.re);
            if taui.nz() {
                a.set(at(i, i + 1), CONE);
                B::zhemv(true, i, taui, a, lda, a.off(at(1, i + 1)), CZERO, tau);
                let alpha = -HALF * taui * B::zdotc(i, tau, 1, a.off(at(1, i + 1)), 1);
                B::zaxpy(i, alpha, a.off(at(1, i + 1)), 1, tau, 1);
                B::zher2(true, i, -CONE, a.off(at(1, i + 1)), tau, a, lda);
            } else {
                a.set(at(i, i), Z::r(a.at(at(i, i)).re));
            }
            a.set(at(i, i + 1), Z::r(e.at(i - 1)));
            d.set(i, a.at(at(i + 1, i + 1)).re);
            tau.set(i - 1, taui);
        }
        d.set(0, a.at(at(1, 1)).re);
    } else {
        a.set(at(1, 1), Z::r(a.at(at(1, 1)).re));
        for i in 1..n {
            let alpha = a.at(at(i + 1, i));
            let (alpha, taui) = zlarfg::<B>(n - i, alpha, a.off(at((i + 2).min(n), i)));
            e.set(i - 1, alpha.re);
            if taui.nz() {
                a.set(at(i + 1, i), CONE);
                B::zhemv(false, n - i, taui, a.off(at(i + 1, i + 1)), lda, a.off(at(i + 1, i)), CZERO, tau.off(i - 1));
                let alpha = -HALF * taui * B::zdotc(n - i, tau.off(i - 1), 1, a.off(at(i + 1, i)), 1);
                B::zaxpy(n - i, alpha, a.off(at(i + 1, i)), 1, tau.off(i - 1), 1);
                B::zher2(false, n - i, -CONE, a.off(at(i + 1, i)), tau.off(i - 1), a.off(at(i + 1, i + 1)), lda);
            } else {
                a.set(at(i + 1, i + 1), Z::r(a.at(at(i + 1, i + 1)).re));
            }
            a.set(at(i + 1, i), Z::r(e.at(i - 1)));
            d.set(i - 1, a.at(at(i, i)).re);
            tau.set(i - 1, taui);
        }
        d.set(n - 1, a.at(at(n, n)).re);
    }
}

/// `ZLATRD`. `base` is the 0-based global offset of this call's `A(1,1)`.
pub(crate) unsafe fn zlatrd<E: ZTrdEngine>(
    eng: &mut E,
    base: usize,
    upper: bool,
    n: usize,
    nb: usize,
    a: ZV,
    lda: usize,
    e: D,
    tau: ZV,
    w: ZV,
    ldw: usize,
) {
    if n == 0 {
        return;
    }
    let at = |i: usize, j: usize| ix(i, j, lda);
    let wt = |i: usize, j: usize| ix(i, j, ldw);
    let one = CONE;
    let mone = -CONE;
    if upper {
        for i in ((n - nb + 1)..=n).rev() {
            let iw = i + nb - n;
            if i < n {
                a.set(at(i, i), Z::r(a.at(at(i, i)).re));
                zlacgv(n - i, w.off(wt(i, iw + 1)), ldw);
                E::B::zgemv(
                    Tr::N,
                    i,
                    n - i,
                    mone,
                    a.off(at(1, i + 1)),
                    lda,
                    w.off(wt(i, iw + 1)),
                    ldw,
                    one,
                    a.off(at(1, i)),
                    1,
                );
                zlacgv(n - i, w.off(wt(i, iw + 1)), ldw);
                zlacgv(n - i, a.off(at(i, i + 1)), lda);
                E::B::zgemv(
                    Tr::N,
                    i,
                    n - i,
                    mone,
                    w.off(wt(1, iw + 1)),
                    ldw,
                    a.off(at(i, i + 1)),
                    lda,
                    one,
                    a.off(at(1, i)),
                    1,
                );
                zlacgv(n - i, a.off(at(i, i + 1)), lda);
                a.set(at(i, i), Z::r(a.at(at(i, i)).re));
            }
            if i > 1 {
                let alpha = a.at(at(i - 1, i));
                let (alpha, t) = zlarfg::<E::B>(i - 1, alpha, a.off(at(1, i)));
                tau.set(i - 2, t);
                e.set(i - 2, alpha.re);
                a.set(at(i - 1, i), CONE);
                eng.hemv(true, i - 1, base, a, lda, a.off(at(1, i)), w.off(wt(1, iw)));
                if i < n {
                    E::B::zgemv(
                        Tr::C,
                        i - 1,
                        n - i,
                        one,
                        w.off(wt(1, iw + 1)),
                        ldw,
                        a.off(at(1, i)),
                        1,
                        CZERO,
                        w.off(wt(i + 1, iw)),
                        1,
                    );
                    E::B::zgemv(
                        Tr::N,
                        i - 1,
                        n - i,
                        mone,
                        a.off(at(1, i + 1)),
                        lda,
                        w.off(wt(i + 1, iw)),
                        1,
                        one,
                        w.off(wt(1, iw)),
                        1,
                    );
                    E::B::zgemv(
                        Tr::C,
                        i - 1,
                        n - i,
                        one,
                        a.off(at(1, i + 1)),
                        lda,
                        a.off(at(1, i)),
                        1,
                        CZERO,
                        w.off(wt(i + 1, iw)),
                        1,
                    );
                    E::B::zgemv(
                        Tr::N,
                        i - 1,
                        n - i,
                        mone,
                        w.off(wt(1, iw + 1)),
                        ldw,
                        w.off(wt(i + 1, iw)),
                        1,
                        one,
                        w.off(wt(1, iw)),
                        1,
                    );
                }
                E::B::zscal(i - 1, tau.at(i - 2), w.off(wt(1, iw)), 1);
                let alpha = -HALF * tau.at(i - 2) * E::B::zdotc(i - 1, w.off(wt(1, iw)), 1, a.off(at(1, i)), 1);
                E::B::zaxpy(i - 1, alpha, a.off(at(1, i)), 1, w.off(wt(1, iw)), 1);
            }
        }
    } else {
        for i in 1..=nb {
            a.set(at(i, i), Z::r(a.at(at(i, i)).re));
            zlacgv(i - 1, w.off(wt(i, 1)), ldw);
            E::B::zgemv(
                Tr::N,
                n - i + 1,
                i - 1,
                mone,
                a.off(at(i, 1)),
                lda,
                w.off(wt(i, 1)),
                ldw,
                one,
                a.off(at(i, i)),
                1,
            );
            zlacgv(i - 1, w.off(wt(i, 1)), ldw);
            zlacgv(i - 1, a.off(at(i, 1)), lda);
            E::B::zgemv(
                Tr::N,
                n - i + 1,
                i - 1,
                mone,
                w.off(wt(i, 1)),
                ldw,
                a.off(at(i, 1)),
                lda,
                one,
                a.off(at(i, i)),
                1,
            );
            zlacgv(i - 1, a.off(at(i, 1)), lda);
            a.set(at(i, i), Z::r(a.at(at(i, i)).re));
            if i < n {
                let alpha = a.at(at(i + 1, i));
                let (alpha, t) = zlarfg::<E::B>(n - i, alpha, a.off(at((i + 2).min(n), i)));
                tau.set(i - 1, t);
                e.set(i - 1, alpha.re);
                a.set(at(i + 1, i), CONE);
                eng.hemv(
                    false,
                    n - i,
                    base + i,
                    a.off(at(i + 1, i + 1)),
                    lda,
                    a.off(at(i + 1, i)),
                    w.off(wt(i + 1, i)),
                );
                E::B::zgemv(
                    Tr::C,
                    n - i,
                    i - 1,
                    one,
                    w.off(wt(i + 1, 1)),
                    ldw,
                    a.off(at(i + 1, i)),
                    1,
                    CZERO,
                    w.off(wt(1, i)),
                    1,
                );
                E::B::zgemv(
                    Tr::N,
                    n - i,
                    i - 1,
                    mone,
                    a.off(at(i + 1, 1)),
                    lda,
                    w.off(wt(1, i)),
                    1,
                    one,
                    w.off(wt(i + 1, i)),
                    1,
                );
                E::B::zgemv(
                    Tr::C,
                    n - i,
                    i - 1,
                    one,
                    a.off(at(i + 1, 1)),
                    lda,
                    a.off(at(i + 1, i)),
                    1,
                    CZERO,
                    w.off(wt(1, i)),
                    1,
                );
                E::B::zgemv(
                    Tr::N,
                    n - i,
                    i - 1,
                    mone,
                    w.off(wt(i + 1, 1)),
                    ldw,
                    w.off(wt(1, i)),
                    1,
                    one,
                    w.off(wt(i + 1, i)),
                    1,
                );
                E::B::zscal(n - i, tau.at(i - 1), w.off(wt(i + 1, i)), 1);
                let alpha = -HALF * tau.at(i - 1) * E::B::zdotc(n - i, w.off(wt(i + 1, i)), 1, a.off(at(i + 1, i)), 1);
                E::B::zaxpy(n - i, alpha, a.off(at(i + 1, i)), 1, w.off(wt(i + 1, i)), 1);
            }
        }
    }
}

/// `ZHETRD`. `work` is `WORK(1:LWORK)`.
pub(crate) unsafe fn zhetrd(upper: bool, n: usize, a: ZV, lda: usize, d: D, e: D, tau: ZV, work: ZV, lwork: usize) {
    zhetrd_with(&mut HostZTrd::<RefBlas>::new(), upper, n, a, lda, d, e, tau, work, lwork);
}

/// [`zhetrd`] with its matrix-sized operations on `eng`.
pub(crate) unsafe fn zhetrd_with<E: ZTrdEngine>(
    eng: &mut E,
    upper: bool,
    n: usize,
    a: ZV,
    lda: usize,
    d: D,
    e: D,
    tau: ZV,
    work: ZV,
    lwork: usize,
) {
    if n == 0 {
        return;
    }
    // ILAENV for ZHETRD returns the same NB/NX/NBMIN as for DSYTRD.
    let (nb, nx) = sytrd_blocking(n, lwork);
    let ldwork = n;
    if upper {
        let kk = n - ((n - nx + nb - 1) / nb) * nb;
        let mut i = n as isize - nb as isize + 1;
        while i >= kk as isize + 1 {
            let iu = i as usize;
            eng.panel(true, n, iu, nb, a, lda);
            zlatrd(eng, 0, true, iu + nb - 1, nb, a, lda, e, tau, work, ldwork);
            eng.her2k(true, iu - 1, nb, 0, a.off(ix(1, iu, lda)), lda, work, ldwork, a);
            for j in iu..=(iu + nb - 1) {
                a.set(ix(j - 1, j, lda), Z::r(e.at(j - 2)));
                d.set(j - 1, a.at(ix(j, j, lda)).re);
            }
            i -= nb as isize;
        }
        eng.tail(true, n, 0, kk, a, lda);
        zhetd2::<E::B>(true, kk, a, lda, d, e, tau);
    } else {
        let mut i = 1usize;
        if n > nx {
            while i <= n - nx {
                eng.panel(false, n, i, nb, a, lda);
                zlatrd(
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
                eng.her2k(
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
                    a.set(ix(j + 1, j, lda), Z::r(e.at(j - 1)));
                    d.set(j - 1, a.at(ix(j, j, lda)).re);
                }
                i += nb;
            }
        }
        eng.tail(false, n, i - 1, n - i + 1, a, lda);
        zhetd2::<E::B>(false, n - i + 1, a.off(ix(i, i, lda)), lda, d.off(i - 1), e.off(i - 1), tau.off(i - 1));
    }
}

unsafe fn ilazlc(m: usize, n: usize, a: ZV, lda: usize) -> usize {
    if n == 0 {
        return n;
    }
    if a.at(ix(1, n, lda)).nz() || a.at(ix(m, n, lda)).nz() {
        return n;
    }
    for c in (1..=n).rev() {
        for i in 1..=m {
            if a.at(ix(i, c, lda)).nz() {
                return c;
            }
        }
    }
    0
}

/// `ZLARF1F( 'L', ... )`.
unsafe fn zlarf1f_left<B: Blas>(m: usize, n: usize, v: ZV, tau: Z, c: ZV, ldc: usize, work: ZV) {
    let mut lastv = 1usize;
    let mut lastc = 0usize;
    if tau.nz() {
        lastv = m;
        let mut i = lastv;
        while lastv > 1 && !v.at(i - 1).nz() {
            lastv -= 1;
            i -= 1;
        }
        lastc = ilazlc(lastv, n, c, ldc);
    }
    if lastc == 0 {
        return;
    }
    if lastv == 1 {
        B::zscal(lastc, CONE - tau, c, ldc);
    } else {
        B::zgemv(Tr::C, lastv - 1, lastc, CONE, c.off(1), ldc, v.off(1), 1, CZERO, work, 1);
        for i in 1..=lastc {
            work.set(i - 1, work.at(i - 1) + c.at(ix(1, i, ldc)).conj());
        }
        for i in 1..=lastc {
            let p = ix(1, i, ldc);
            c.set(p, c.at(p) - tau * work.at(i - 1).conj());
        }
        B::zgerc(lastv - 1, lastc, -tau, v.off(1), 1, work, 1, c.off(1), ldc);
    }
}

/// `ZLARF( 'L', M, N, V, 1, TAU, C, LDC, WORK )` as LAPACK 3.12.0 (and
/// OpenBLAS 0.3.30's vendored copy) writes it: `V` explicit, `LASTV` scanned
/// from the end, then `ZGEMV('C')` + `ZGERC`. Used when `B::LEGACY_DLARF`.
unsafe fn zlarf_left<B: Blas>(m: usize, n: usize, v: ZV, tau: Z, c: ZV, ldc: usize, work: ZV) {
    let mut lastv = 0usize;
    let mut lastc = 0usize;
    if tau.nz() {
        lastv = m;
        while lastv > 0 && !v.at(lastv - 1).nz() {
            lastv -= 1;
        }
        lastc = ilazlc(lastv, n, c, ldc);
    }
    if lastv > 0 {
        B::zgemv(Tr::C, lastv, lastc, CONE, c, ldc, v, 1, CZERO, work, 1);
        B::zgerc(lastv, lastc, -tau, v, 1, work, 1, c, ldc);
    }
}

/// `ZLARF1L( 'L', ... )`.
unsafe fn zlarf1l_left<B: Blas>(m: usize, n: usize, v: ZV, tau: Z, c: ZV, ldc: usize, work: ZV) {
    let mut firstv = 1usize;
    let mut lastc = 0usize;
    let mut lastv = 0usize;
    let mut i = 1usize;
    if tau.nz() {
        lastv = m;
        while lastv > firstv && !v.at(i - 1).nz() {
            firstv += 1;
            i += 1;
        }
        lastc = ilazlc(lastv, n, c, ldc);
    }
    if lastc == 0 {
        return;
    }
    if lastv == firstv {
        B::zscal(lastc, CONE - tau, c.off(lastv - 1), ldc);
    } else {
        B::zgemv(Tr::C, lastv - firstv, lastc, CONE, c.off(firstv - 1), ldc, v.off(i - 1), 1, CZERO, work, 1);
        for j in 1..=lastc {
            work.set(j - 1, work.at(j - 1) + c.at(ix(lastv, j, ldc)).conj());
        }
        for j in 1..=lastc {
            let p = ix(lastv, j, ldc);
            c.set(p, c.at(p) - tau * work.at(j - 1).conj());
        }
        B::zgerc(lastv - firstv, lastc, -tau, v.off(i - 1), 1, work, 1, c.off(firstv - 1), ldc);
    }
}

/// `ZLARFT( DIRECT, 'Columnwise', ... )` (recursive, 3.12.1).
unsafe fn zlarft_c<B: Blas>(forward: bool, n: usize, k: usize, v: ZV, ldv: usize, tau: ZV, t: ZV, ldt: usize) {
    // PARAMETER( ONE=1.0D+0, ZERO=0.0D+0, NEG_ONE=-1.0D+0 ) as COMPLEX*16.
    let one = CONE;
    let neg_one = Z::new(-1.0, 0.0);
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
        zlarft_c::<B>(true, n, l, v, ldv, tau, t, ldt);
        zlarft_c::<B>(true, n - l, k - l, v.off(vi(l + 1, l + 1)), ldv, tau.off(l), t.off(ti(l + 1, l + 1)), ldt);
        for j in 1..=l {
            for i in 1..=(k - l) {
                t.set(ti(j, l + i), v.at(vi(l + i, j)).conj());
            }
        }
        B::ztrmm(false, false, Tr::N, false, l, k - l, one, v.off(vi(l + 1, l + 1)), ldv, t.off(ti(1, l + 1)), ldt);
        B::zgemm(
            Tr::C,
            Tr::N,
            l,
            k - l,
            n - k,
            one,
            v.off(vi(k + 1, 1)),
            ldv,
            v.off(vi(k + 1, l + 1)),
            ldv,
            one,
            t.off(ti(1, l + 1)),
            ldt,
        );
        B::ztrmm(true, true, Tr::N, true, l, k - l, neg_one, t, ldt, t.off(ti(1, l + 1)), ldt);
        B::ztrmm(false, true, Tr::N, true, l, k - l, one, t.off(ti(l + 1, l + 1)), ldt, t.off(ti(1, l + 1)), ldt);
    } else {
        zlarft_c::<B>(false, n - l, k - l, v, ldv, tau, t, ldt);
        zlarft_c::<B>(false, n, l, v.off(vi(1, k - l + 1)), ldv, tau.off(k - l), t.off(ti(k - l + 1, k - l + 1)), ldt);
        for j in 1..=(k - l) {
            for i in 1..=l {
                t.set(ti(k - l + i, j), v.at(vi(n - k + j, k - l + i)).conj());
            }
        }
        B::ztrmm(false, true, Tr::N, false, l, k - l, one, v.off(vi(n - k + 1, 1)), ldv, t.off(ti(k - l + 1, 1)), ldt);
        B::zgemm(
            Tr::C,
            Tr::N,
            l,
            k - l,
            n - k,
            one,
            v.off(vi(1, k - l + 1)),
            ldv,
            v,
            ldv,
            one,
            t.off(ti(k - l + 1, 1)),
            ldt,
        );
        B::ztrmm(
            true,
            false,
            Tr::N,
            true,
            l,
            k - l,
            neg_one,
            t.off(ti(k - l + 1, k - l + 1)),
            ldt,
            t.off(ti(k - l + 1, 1)),
            ldt,
        );
        B::ztrmm(false, false, Tr::N, true, l, k - l, one, t, ldt, t.off(ti(k - l + 1, 1)), ldt);
    }
}

/// `ZLARFB( 'L', 'N', DIRECT, 'Columnwise', ... )`.
unsafe fn zlarfb_lnc<B: Blas>(
    forward: bool,
    m: usize,
    n: usize,
    k: usize,
    v: ZV,
    ldv: usize,
    t: ZV,
    ldt: usize,
    c: ZV,
    ldc: usize,
    work: ZV,
    ldwork: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let one = CONE;
    let mone = -CONE;
    let wk = |i: usize, j: usize| ix(i, j, ldwork);
    if forward {
        for j in 1..=k {
            zcopy(n, c.off(ix(j, 1, ldc)), ldc, work.off(wk(1, j)), 1);
            zlacgv(n, work.off(wk(1, j)), 1);
        }
        B::ztrmm(false, false, Tr::N, false, n, k, one, v, ldv, work, ldwork);
        if m > k {
            B::zgemm(
                Tr::C,
                Tr::N,
                n,
                k,
                m - k,
                one,
                c.off(ix(k + 1, 1, ldc)),
                ldc,
                v.off(ix(k + 1, 1, ldv)),
                ldv,
                one,
                work,
                ldwork,
            );
        }
        B::ztrmm(false, true, Tr::C, true, n, k, one, t, ldt, work, ldwork);
        if m > k {
            B::zgemm(
                Tr::N,
                Tr::C,
                m - k,
                n,
                k,
                mone,
                v.off(ix(k + 1, 1, ldv)),
                ldv,
                work,
                ldwork,
                one,
                c.off(ix(k + 1, 1, ldc)),
                ldc,
            );
        }
        B::ztrmm(false, false, Tr::C, false, n, k, one, v, ldv, work, ldwork);
        for j in 1..=k {
            for i in 1..=n {
                let p = ix(j, i, ldc);
                c.set(p, c.at(p) - work.at(wk(i, j)).conj());
            }
        }
    } else {
        for j in 1..=k {
            zcopy(n, c.off(ix(m - k + j, 1, ldc)), ldc, work.off(wk(1, j)), 1);
            zlacgv(n, work.off(wk(1, j)), 1);
        }
        B::ztrmm(false, true, Tr::N, false, n, k, one, v.off(ix(m - k + 1, 1, ldv)), ldv, work, ldwork);
        if m > k {
            B::zgemm(Tr::C, Tr::N, n, k, m - k, one, c, ldc, v, ldv, one, work, ldwork);
        }
        B::ztrmm(false, false, Tr::C, true, n, k, one, t, ldt, work, ldwork);
        if m > k {
            B::zgemm(Tr::N, Tr::C, m - k, n, k, mone, v, ldv, work, ldwork, one, c, ldc);
        }
        B::ztrmm(false, true, Tr::C, false, n, k, one, v.off(ix(m - k + 1, 1, ldv)), ldv, work, ldwork);
        for j in 1..=k {
            for i in 1..=n {
                let p = ix(m - k + j, i, ldc);
                c.set(p, c.at(p) - work.at(wk(i, j)).conj());
            }
        }
    }
}

unsafe fn zunmql_ln<E: ZOrmEngine>(
    eng: &mut E,
    m: usize,
    n: usize,
    k: usize,
    a: ZV,
    lda: usize,
    tau: ZV,
    c: ZV,
    crow: usize,
    ldc: usize,
    work: ZV,
    lwork: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let nq = m;
    let nw = n.max(1);
    match orm_blocking(n, k, lwork) {
        None => {
            if k == 0 {
                return;
            }
            for i in 1..=k {
                if E::B::LEGACY_DLARF {
                    // AII = A( NQ-K+I, I ); A( NQ-K+I, I ) = ONE; ZLARF; restore.
                    let p = ix(nq - k + i, i, lda);
                    let aii = a.at(p);
                    a.set(p, CONE);
                    if !eng.larf_left(m - k + i, n, a.off(ix(1, i, lda)), tau.at(i - 1), crow) {
                        zlarf_left::<E::B>(m - k + i, n, a.off(ix(1, i, lda)), tau.at(i - 1), c, ldc, work);
                    }
                    a.set(p, aii);
                } else {
                    zlarf1l_left::<E::B>(m - k + i, n, a.off(ix(1, i, lda)), tau.at(i - 1), c, ldc, work);
                }
            }
        }
        Some(nb) => {
            let iwt = nw * nb;
            let mut i = 1usize;
            while i <= k {
                let ib = nb.min(k - i + 1);
                zlarft_c::<E::B>(
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

unsafe fn zunmqr_ln<E: ZOrmEngine>(
    eng: &mut E,
    m: usize,
    n: usize,
    k: usize,
    a: ZV,
    lda: usize,
    tau: ZV,
    c: ZV,
    crow: usize,
    ldc: usize,
    work: ZV,
    lwork: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let nq = m;
    let nw = n.max(1);
    match orm_blocking(n, k, lwork) {
        None => {
            for i in (1..=k).rev() {
                if E::B::LEGACY_DLARF {
                    // AII = A( I, I ); A( I, I ) = ONE; ZLARF; restore.
                    let p = ix(i, i, lda);
                    let aii = a.at(p);
                    a.set(p, CONE);
                    if !eng.larf_left(m - i + 1, n, a.off(p), tau.at(i - 1), crow + i - 1) {
                        zlarf_left::<E::B>(m - i + 1, n, a.off(p), tau.at(i - 1), c.off(ix(i, 1, ldc)), ldc, work);
                    }
                    a.set(p, aii);
                } else {
                    zlarf1f_left::<E::B>(
                        m - i + 1,
                        n,
                        a.off(ix(i, i, lda)),
                        tau.at(i - 1),
                        c.off(ix(i, 1, ldc)),
                        ldc,
                        work,
                    );
                }
            }
        }
        Some(nb) => {
            let iwt = nw * nb;
            let mut i = ((k - 1) / nb) * nb + 1;
            loop {
                let ib = nb.min(k - i + 1);
                zlarft_c::<E::B>(true, nq - i + 1, ib, a.off(ix(i, i, lda)), lda, tau.off(i - 1), work.off(iwt), LDT);
                eng.larfb(
                    true,
                    m - i + 1,
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

/// `ZUNMTR( 'L', UPLO, 'N', M, N, ... )`.
pub(crate) unsafe fn zunmtr_ln(
    upper: bool,
    m: usize,
    n: usize,
    a: ZV,
    lda: usize,
    tau: ZV,
    c: ZV,
    ldc: usize,
    work: ZV,
    lwork: usize,
) {
    zunmtr_ln_with(&mut HostZOrm::<RefBlas>::new(), upper, m, n, a, lda, tau, c, ldc, work, lwork);
}

/// [`zunmtr_ln`] with the blocked `ZLARFB` on `eng`.
pub(crate) unsafe fn zunmtr_ln_with<E: ZOrmEngine>(
    eng: &mut E,
    upper: bool,
    m: usize,
    n: usize,
    a: ZV,
    lda: usize,
    tau: ZV,
    c: ZV,
    ldc: usize,
    work: ZV,
    lwork: usize,
) {
    let nq = m;
    if m == 0 || n == 0 || nq == 1 {
        return;
    }
    if upper {
        zunmql_ln(eng, m - 1, n, nq - 1, a.off(ix(1, 2, lda)), lda, tau, c, 0, ldc, work, lwork);
    } else {
        zunmqr_ln(eng, m - 1, n, nq - 1, a.off(ix(2, 1, lda)), lda, tau, c.off(ix(2, 1, ldc)), 1, ldc, work, lwork);
    }
}
