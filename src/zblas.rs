//! Complex double precision reference BLAS (Reference-LAPACK v3.12.1
//! `BLAS/SRC`) on planar storage.
//!
//! Transcribed from `zdotc.f`, `dznrm2.f90`, `zaxpy.f`, `zscal.f`,
//! `zdscal.f`, `zcopy.f`, `zswap.f`, `zgemv.f`, `zgerc.f`, `zhemv.f`,
//! `zher2.f`, `zher2k.f`, `zgemm.f` and `ztrmm.f`.
//!
//! Complex arithmetic follows gfortran's lowering at `-O2` with the default
//! `-fsigned-zeros`: GCC's complex-lowering pass never treats a constant zero
//! part as "known zero" when signed zeros are honoured, so every product is
//! the full `(ar*br - ai*bi, ar*bi + ai*br)`, including mixed REAL*COMPLEX
//! operands (the REAL is promoted to `(x, 0.0)`). This was verified on the
//! oracle's own `zlascl.f.o` (`A(I,J)*MUL` computes `ar*mul - ai*0`).
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::blas::{SBIG, SSML, TBIG, TSML};
use crate::ptr::{D, ix};
use std::ops::{Add, Mul, Neg, Sub};

/// `COMPLEX*16` value with gfortran's inline arithmetic.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Z {
    pub re: f64,
    pub im: f64,
}

pub(crate) const CZERO: Z = Z { re: 0.0, im: 0.0 };
pub(crate) const CONE: Z = Z { re: 1.0, im: 0.0 };

impl Z {
    #[inline(always)]
    pub const fn new(re: f64, im: f64) -> Z {
        Z { re, im }
    }
    /// `DCMPLX( X )`, or the promotion of a REAL operand.
    #[inline(always)]
    pub(crate) const fn r(re: f64) -> Z {
        Z { re, im: 0.0 }
    }
    /// `DCONJG`.
    #[inline(always)]
    pub(crate) fn conj(self) -> Z {
        Z { re: self.re, im: -self.im }
    }
    /// `Z.NE.ZERO`.
    #[inline(always)]
    pub(crate) fn nz(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
    /// `Z.EQ.W`.
    #[inline(always)]
    pub(crate) fn eq(self, w: Z) -> bool {
        self.re == w.re && self.im == w.im
    }
    /// `DCABS1`.
    #[inline(always)]
    pub(crate) fn cabs1(self) -> f64 {
        self.re.abs() + self.im.abs()
    }
    /// `ABS( Z )`: gfortran calls libm `cabs`.
    #[inline(always)]
    pub(crate) fn abs(self) -> f64 {
        // SAFETY: plain libm call with a by-value complex (two doubles in
        // xmm0/xmm1, the SysV ABI of `double _Complex`).
        unsafe { cabs(self) }
    }
}

unsafe extern "C" {
    /// glibc `cabs`, the symbol gfortran's `ABS( COMPLEX*16 )` lowers to.
    fn cabs(z: Z) -> f64;
}

impl Add for Z {
    type Output = Z;
    #[inline(always)]
    fn add(self, b: Z) -> Z {
        Z { re: self.re + b.re, im: self.im + b.im }
    }
}
impl Sub for Z {
    type Output = Z;
    #[inline(always)]
    fn sub(self, b: Z) -> Z {
        Z { re: self.re - b.re, im: self.im - b.im }
    }
}
impl Mul for Z {
    type Output = Z;
    #[inline(always)]
    fn mul(self, b: Z) -> Z {
        Z { re: self.re * b.re - self.im * b.im, im: self.re * b.im + self.im * b.re }
    }
}
impl Neg for Z {
    type Output = Z;
    #[inline(always)]
    fn neg(self) -> Z {
        Z { re: -self.re, im: -self.im }
    }
}

/// Planar complex view: real and imaginary parts in two real arrays.
#[derive(Clone, Copy)]
pub struct ZV {
    pub re: D,
    pub im: D,
}

impl ZV {
    #[inline(always)]
    pub fn new(re: D, im: D) -> ZV {
        ZV { re, im }
    }
    #[inline(always)]
    pub(crate) fn off(self, k: usize) -> ZV {
        ZV { re: self.re.off(k), im: self.im.off(k) }
    }
    /// # Safety
    /// `i` in bounds.
    #[inline(always)]
    pub(crate) unsafe fn at(self, i: usize) -> Z {
        Z { re: self.re.at(i), im: self.im.at(i) }
    }
    /// # Safety
    /// `i` in bounds, unaliased for writes.
    #[inline(always)]
    pub(crate) unsafe fn set(self, i: usize, v: Z) {
        self.re.set(i, v.re);
        self.im.set(i, v.im);
    }
}

/// Transposition mode of the complex level-2/3 routines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tr {
    N,
    T,
    C,
}

/// `ZDOTC`: `ZTEMP = ZTEMP + DCONJG(ZX(I))*ZY(I)`, a sequential chain.
pub(crate) unsafe fn zdotc(n: usize, x: ZV, incx: usize, y: ZV, incy: usize) -> Z {
    let mut t = CZERO;
    let (mut ixx, mut iy) = (0, 0);
    for _ in 0..n {
        t = t + x.at(ixx).conj() * y.at(iy);
        ixx += incx;
        iy += incy;
    }
    t
}

/// `DZNRM2` (3.10+ f90): real then imaginary part of each entry.
pub(crate) unsafe fn dznrm2(n: usize, x: ZV, incx: usize) -> f64 {
    const MAXN: f64 = f64::MAX;
    if n == 0 {
        return 0.0;
    }
    let mut notbig = true;
    let (mut asml, mut amed, mut abig) = (0.0f64, 0.0f64, 0.0f64);
    let mut ixx = 0;
    for _ in 0..n {
        for ax in [x.re.at(ixx).abs(), x.im.at(ixx).abs()] {
            if ax > TBIG {
                let t = ax * SBIG;
                abig += t * t;
                notbig = false;
            } else if ax < TSML {
                if notbig {
                    let t = ax * SSML;
                    asml += t * t;
                }
            } else {
                amed += ax * ax;
            }
        }
        ixx += incx;
    }
    let (scl, sumsq);
    if abig > 0.0 {
        if amed > 0.0 || amed > MAXN || amed.is_nan() {
            abig += (amed * SBIG) * SBIG;
        }
        scl = 1.0 / SBIG;
        sumsq = abig;
    } else if asml > 0.0 {
        if amed > 0.0 || amed > MAXN || amed.is_nan() {
            let amed = amed.sqrt();
            let asml = asml.sqrt() / SSML;
            let (ymin, ymax) = if asml > amed { (amed, asml) } else { (asml, amed) };
            scl = 1.0;
            let r = ymin / ymax;
            sumsq = (ymax * ymax) * (1.0 + r * r);
        } else {
            scl = 1.0 / SSML;
            sumsq = asml;
        }
    } else {
        scl = 1.0;
        sumsq = amed;
    }
    scl * sumsq.sqrt()
}

/// `ZAXPY`.
pub(crate) unsafe fn zaxpy(n: usize, za: Z, x: ZV, incx: usize, y: ZV, incy: usize) {
    if n == 0 || za.cabs1() == 0.0 {
        return;
    }
    let (mut ixx, mut iy) = (0, 0);
    for _ in 0..n {
        y.set(iy, y.at(iy) + za * x.at(ixx));
        ixx += incx;
        iy += incy;
    }
}

/// `ZSCAL`.
pub(crate) unsafe fn zscal(n: usize, za: Z, x: ZV, incx: usize) {
    if n == 0 || incx == 0 || za.eq(CONE) {
        return;
    }
    let mut i = 0;
    for _ in 0..n {
        x.set(i, za * x.at(i));
        i += incx;
    }
}

/// `ZDSCAL`: `DCMPLX( DA*DBLE(ZX(I)), DA*DIMAG(ZX(I)) )`.
pub(crate) unsafe fn zdscal(n: usize, da: f64, x: ZV, incx: usize) {
    if n == 0 || incx == 0 || da == 1.0 {
        return;
    }
    let mut i = 0;
    for _ in 0..n {
        let v = x.at(i);
        x.set(i, Z::new(da * v.re, da * v.im));
        i += incx;
    }
}

/// `ZCOPY`.
pub(crate) unsafe fn zcopy(n: usize, x: ZV, incx: usize, y: ZV, incy: usize) {
    let (mut ixx, mut iy) = (0, 0);
    for _ in 0..n {
        y.set(iy, x.at(ixx));
        ixx += incx;
        iy += incy;
    }
}

/// `ZSWAP`.
pub(crate) unsafe fn zswap(n: usize, x: ZV, incx: usize, y: ZV, incy: usize) {
    let (mut ixx, mut iy) = (0, 0);
    for _ in 0..n {
        let t = x.at(ixx);
        x.set(ixx, y.at(iy));
        y.set(iy, t);
        ixx += incx;
        iy += incy;
    }
}

/// `ZLACGV`.
pub(crate) unsafe fn zlacgv(n: usize, x: ZV, incx: usize) {
    let mut i = 0;
    for _ in 0..n {
        x.set(i, x.at(i).conj());
        i += incx;
    }
}

/// Lanes advanced together by the complex reduction kernels (two real
/// chains per complex element).
const ZLANES: usize = 4;

impl ZV {
    /// `(re, im)` slices of the next `len` elements.
    ///
    /// # Safety
    /// As [`crate::ptr::V::slice`].
    #[inline(always)]
    pub(crate) unsafe fn zs<'a>(self, len: usize) -> (&'a [f64], &'a [f64]) {
        unsafe { (self.re.slice(len), self.im.slice(len)) }
    }
    /// Mutable `(re, im)` slices of the next `len` elements.
    ///
    /// # Safety
    /// As [`crate::ptr::V::slice_mut`].
    #[inline(always)]
    pub(crate) unsafe fn zs_mut<'a>(self, len: usize) -> (&'a mut [f64], &'a mut [f64]) {
        unsafe { (self.re.slice_mut(len), self.im.slice_mut(len)) }
    }
}

/// `Y(I) = Y(I) + T*A(I)` on planar slices.
#[inline(always)]
fn zaxpy_s(t: Z, a: (&[f64], &[f64]), y: (&mut [f64], &mut [f64])) {
    let n = y.0.len();
    let (ar, ai, yr, yi) = (&a.0[..n], &a.1[..n], y.0, &mut y.1[..n]);
    for i in 0..n {
        let pr = t.re * ar[i] - t.im * ai[i];
        let pi = t.re * ai[i] + t.im * ar[i];
        yr[i] += pr;
        yi[i] += pi;
    }
}

/// `C(I) = T*C(I)` on planar slices.
#[inline(always)]
fn zscal_s(t: Z, c: (&mut [f64], &mut [f64])) {
    let n = c.0.len();
    let (cr, ci) = (c.0, &mut c.1[..n]);
    for i in 0..n {
        let (r, im) = (cr[i], ci[i]);
        cr[i] = t.re * r - t.im * im;
        ci[i] = t.re * im + t.im * r;
    }
}

/// `acc_q = acc_q + op(A_q(l)) * X(l)`, l ascending, for `ZLANES` columns.
/// `conj` applies `DCONJG` to `A`.
#[inline(always)]
fn zdotl(conj: bool, a: [(&[f64], &[f64]); ZLANES], x: (&[f64], &[f64]), acc: &mut [Z; ZLANES]) {
    let n = x.0.len();
    let xr = x.0;
    let xi = &x.1[..n];
    let ar: [&[f64]; ZLANES] = std::array::from_fn(|q| &a[q].0[..n]);
    let ai: [&[f64]; ZLANES] = std::array::from_fn(|q| &a[q].1[..n]);
    let mut sr: [f64; ZLANES] = std::array::from_fn(|q| acc[q].re);
    let mut si: [f64; ZLANES] = std::array::from_fn(|q| acc[q].im);
    let sg = if conj { -1.0 } else { 1.0 };
    for l in 0..n {
        let (xlr, xli) = (xr[l], xi[l]);
        for q in 0..ZLANES {
            let (pr, pi) = (ar[q][l], sg * ai[q][l]);
            sr[q] += pr * xlr - pi * xli;
            si[q] += pr * xli + pi * xlr;
        }
    }
    *acc = std::array::from_fn(|q| Z::new(sr[q], si[q]));
}

/// One chain of [`zdotl`].
#[inline(always)]
fn zdot1(conj: bool, a: (&[f64], &[f64]), x: (&[f64], &[f64]), acc: Z) -> Z {
    let n = x.0.len();
    let (ar, ai, xr, xi) = (&a.0[..n], &a.1[..n], x.0, &x.1[..n]);
    let (mut sr, mut si) = (acc.re, acc.im);
    let sg = if conj { -1.0 } else { 1.0 };
    for l in 0..n {
        let (pr, pi) = (ar[l], sg * ai[l]);
        sr += pr * xr[l] - pi * xi[l];
        si += pr * xi[l] + pi * xr[l];
    }
    Z::new(sr, si)
}

/// `ZGEMV`. `'N'` is element-wise per row, `'T'`/`'C'` per column (lanes of
/// column chains advance together).
pub(crate) unsafe fn zgemv(
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
    if m == 0 || n == 0 || (alpha.eq(CZERO) && beta.eq(CONE)) {
        return;
    }
    let leny = if trans == Tr::N { m } else { n };
    if !beta.eq(CONE) {
        let mut iy = 0;
        for _ in 0..leny {
            if beta.eq(CZERO) {
                y.set(iy, CZERO);
            } else {
                y.set(iy, beta * y.at(iy));
            }
            iy += incy;
        }
    }
    if alpha.eq(CZERO) {
        return;
    }
    if trans == Tr::N {
        {
            let (lo, hi) = (0usize, m);

            let mut jx = 0;
            for j in 1..=n {
                let temp = alpha * x.at(jx);
                let aj = a.off((j - 1) * lda + lo);
                if incy == 1 {
                    zaxpy_s(temp, aj.zs(hi - lo), y.off(lo).zs_mut(hi - lo));
                } else {
                    for i in lo..hi {
                        let iy = i * incy;
                        y.set(iy, y.at(iy) + temp * aj.at(i - lo));
                    }
                }
                jx += incx;
            }
        }
    } else {
        let conj = trans == Tr::C;
        {
            let (lo, hi) = (0usize, n);

            let mut j = lo + 1;
            if incx == 1 {
                let xs = x.zs(m);
                while j + ZLANES - 1 <= hi {
                    let mut acc = [CZERO; ZLANES];
                    zdotl(conj, std::array::from_fn(|q| a.off((j - 1 + q) * lda).zs(m)), xs, &mut acc);
                    for (q, t) in acc.iter().enumerate() {
                        let jy = (j - 1 + q) * incy;
                        y.set(jy, y.at(jy) + alpha * *t);
                    }
                    j += ZLANES;
                }
            }
            while j <= hi {
                let mut temp = CZERO;
                let mut ixx = 0;
                for i in 1..=m {
                    let aij = a.at(ix(i, j, lda));
                    let aij = if conj { aij.conj() } else { aij };
                    temp = temp + aij * x.at(ixx);
                    ixx += incx;
                }
                let jy = (j - 1) * incy;
                y.set(jy, y.at(jy) + alpha * temp);
                j += 1;
            }
        }
    }
}

/// `ZGERC`. Element-wise per column.
pub(crate) unsafe fn zgerc(m: usize, n: usize, alpha: Z, x: ZV, incx: usize, y: ZV, incy: usize, a: ZV, lda: usize) {
    if m == 0 || n == 0 || alpha.eq(CZERO) {
        return;
    }
    {
        let (lo, hi) = (0usize, n);

        for j in (lo + 1)..=hi {
            let yj = y.at((j - 1) * incy);
            if yj.nz() {
                let temp = alpha * yj.conj();
                if incx == 1 {
                    // A(I,J) + X(I)*TEMP; complex products commute bitwise.
                    zaxpy_s(temp, x.zs(m), a.off((j - 1) * lda).zs_mut(m));
                } else {
                    let mut ixx = 0;
                    for i in 1..=m {
                        let p = ix(i, j, lda);
                        a.set(p, a.at(p) + x.at(ixx) * temp);
                        ixx += incx;
                    }
                }
            }
        }
    }
}

/// `ZHEMV` with unit increments, element-wise per row as `dsymv`.
pub(crate) unsafe fn zhemv(upper: bool, n: usize, alpha: Z, a: ZV, lda: usize, x: ZV, beta: Z, y: ZV) {
    if n == 0 || (alpha.eq(CZERO) && beta.eq(CONE)) {
        return;
    }
    if !beta.eq(CONE) {
        for i in 0..n {
            if beta.eq(CZERO) {
                y.set(i, CZERO);
            } else {
                y.set(i, beta * y.at(i));
            }
        }
    }
    if alpha.eq(CZERO) {
        return;
    }
    let (xr, xi) = x.zs(n);
    let col = |j: usize| a.off((j - 1) * lda);
    {
        let (lo, hi) = (0usize, n);

        let mut t2 = [CZERO; ZLANES];
        let mut t2_first = usize::MAX;
        if upper {
            for j in (lo + 1)..=n {
                let temp1 = alpha * x.at(j - 1);
                let cj = col(j);
                let top = (j - 1).min(hi);
                if top > lo {
                    zaxpy_s(temp1, cj.off(lo).zs(top - lo), y.off(lo).zs_mut(top - lo));
                }
                if j <= hi {
                    if t2_first == usize::MAX || j >= t2_first + ZLANES {
                        t2_first = j;
                        let mut acc = [CZERO; ZLANES];
                        let q_n = ZLANES.min(hi + 1 - j);
                        if q_n == ZLANES {
                            let c: [ZV; ZLANES] = std::array::from_fn(|q| col(j + q));
                            zdotl(
                                true,
                                std::array::from_fn(|q| c[q].zs(j - 1)),
                                (&xr[..j - 1], &xi[..j - 1]),
                                &mut acc,
                            );
                            for q in 1..ZLANES {
                                let (cr, ci) = c[q].zs(j - 1 + q);
                                acc[q] = zdot1(
                                    true,
                                    (&cr[j - 1..], &ci[j - 1..]),
                                    (&xr[j - 1..j - 1 + q], &xi[j - 1..j - 1 + q]),
                                    acc[q],
                                );
                            }
                        } else {
                            for q in 0..q_n {
                                acc[q] =
                                    zdot1(true, col(j + q).zs(j - 1 + q), (&xr[..j - 1 + q], &xi[..j - 1 + q]), CZERO);
                            }
                        }
                        t2 = acc;
                    }
                    let temp2 = t2[j - t2_first];
                    y.set(j - 1, y.at(j - 1) + temp1 * Z::r(cj.at(j - 1).re) + alpha * temp2);
                }
            }
        } else {
            for j in 1..=hi {
                let temp1 = alpha * x.at(j - 1);
                let cj = col(j);
                let own = j > lo;
                if own {
                    y.set(j - 1, y.at(j - 1) + temp1 * Z::r(cj.at(j - 1).re));
                    if t2_first == usize::MAX || j >= t2_first + ZLANES {
                        t2_first = j;
                        let mut acc = [CZERO; ZLANES];
                        let q_n = ZLANES.min(hi + 1 - j).min(n + 1 - j);
                        if q_n == ZLANES && j + ZLANES <= n {
                            let start = j + ZLANES;
                            let c: [ZV; ZLANES] = std::array::from_fn(|q| col(j + q));
                            for q in 0..ZLANES {
                                let (cr, ci) = c[q].zs(n);
                                acc[q] = zdot1(
                                    true,
                                    (&cr[j + q..start - 1], &ci[j + q..start - 1]),
                                    (&xr[j + q..start - 1], &xi[j + q..start - 1]),
                                    CZERO,
                                );
                            }
                            let len = n + 1 - start;
                            zdotl(
                                true,
                                std::array::from_fn(|q| c[q].off(start - 1).zs(len)),
                                (&xr[start - 1..], &xi[start - 1..]),
                                &mut acc,
                            );
                        } else {
                            for q in 0..q_n {
                                let jq = j + q;
                                let (cr, ci) = col(jq).zs(n);
                                acc[q] = zdot1(true, (&cr[jq..], &ci[jq..]), (&xr[jq..], &xi[jq..]), CZERO);
                            }
                        }
                        t2 = acc;
                    }
                }
                let first = (j + 1).max(lo + 1);
                if first <= hi {
                    let len = hi + 1 - first;
                    zaxpy_s(temp1, cj.off(first - 1).zs(len), y.off(first - 1).zs_mut(len));
                }
                if own {
                    y.set(j - 1, y.at(j - 1) + alpha * t2[j - t2_first]);
                }
            }
        }
    }
}

/// `ZHER2` with unit increments (the `zhetd2` tail only).
pub(crate) unsafe fn zher2(upper: bool, n: usize, alpha: Z, x: ZV, y: ZV, a: ZV, lda: usize) {
    if n == 0 || alpha.eq(CZERO) {
        return;
    }
    for j in 1..=n {
        let xj = x.at(j - 1);
        let yj = y.at(j - 1);
        let pjj = ix(j, j, lda);
        if xj.nz() || yj.nz() {
            let temp1 = alpha * yj.conj();
            let temp2 = (alpha * xj).conj();
            let (i0, i1) = if upper { (1, j - 1) } else { (j + 1, n) };
            if upper {
                for i in i0..=i1 {
                    let p = ix(i, j, lda);
                    a.set(p, a.at(p) + x.at(i - 1) * temp1 + y.at(i - 1) * temp2);
                }
                let dj = a.at(pjj).re + (xj * temp1 + yj * temp2).re;
                a.set(pjj, Z::r(dj));
            } else {
                let dj = a.at(pjj).re + (xj * temp1 + yj * temp2).re;
                a.set(pjj, Z::r(dj));
                for i in i0..=i1 {
                    let p = ix(i, j, lda);
                    a.set(p, a.at(p) + x.at(i - 1) * temp1 + y.at(i - 1) * temp2);
                }
            }
        } else {
            a.set(pjj, Z::r(a.at(pjj).re));
        }
    }
}

/// `ZHER2K` with `TRANS = 'N'`; `beta` is real. Element-wise per column.
pub(crate) unsafe fn zher2k_n(
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
    if n == 0 || ((alpha.eq(CZERO) || k == 0) && beta == 1.0) {
        return;
    }
    let rb = Z::r(beta);
    if alpha.eq(CZERO) {
        for j in 1..=n {
            let pjj = ix(j, j, ldc);
            if beta == 0.0 {
                let (i0, i1) = if upper { (1, j) } else { (j, n) };
                for i in i0..=i1 {
                    c.set(ix(i, j, ldc), CZERO);
                }
            } else if upper {
                for i in 1..j {
                    let p = ix(i, j, ldc);
                    c.set(p, rb * c.at(p));
                }
                c.set(pjj, Z::r(beta * c.at(pjj).re));
            } else {
                c.set(pjj, Z::r(beta * c.at(pjj).re));
                for i in (j + 1)..=n {
                    let p = ix(i, j, ldc);
                    c.set(p, rb * c.at(p));
                }
            }
        }
        return;
    }
    {
        let (lo, hi) = (0usize, n);

        for j in (lo + 1)..=hi {
            let pjj = ix(j, j, ldc);
            let (i0, i1) = if upper { (1, j - 1) } else { (j + 1, n) };
            let len = (i1 + 1).saturating_sub(i0);
            if beta == 0.0 {
                let (z0, z1) = if upper { (1, j) } else { (j, n) };
                for i in z0..=z1 {
                    c.set(ix(i, j, ldc), CZERO);
                }
            } else if beta != 1.0 {
                if len > 0 {
                    zscal_s(rb, c.off(ix(i0, j, ldc)).zs_mut(len));
                }
                c.set(pjj, Z::r(beta * c.at(pjj).re));
            } else {
                c.set(pjj, Z::r(c.at(pjj).re));
            }
            for l in 1..=k {
                let ajl = a.at(ix(j, l, lda));
                let bjl = b.at(ix(j, l, ldb));
                if ajl.nz() || bjl.nz() {
                    let t1 = alpha * bjl.conj();
                    let t2 = (alpha * ajl).conj();
                    if len > 0 {
                        let (cr, ci) = c.off(ix(i0, j, ldc)).zs_mut(len);
                        let (ar, ai) = a.off(ix(i0, l, lda)).zs(len);
                        let (br, bi) = b.off(ix(i0, l, ldb)).zs(len);
                        for i in 0..len {
                            // (C + A*TEMP1) + B*TEMP2, complex products in full.
                            let r = (cr[i] + (ar[i] * t1.re - ai[i] * t1.im)) + (br[i] * t2.re - bi[i] * t2.im);
                            let m = (ci[i] + (ar[i] * t1.im + ai[i] * t1.re)) + (br[i] * t2.im + bi[i] * t2.re);
                            cr[i] = r;
                            ci[i] = m;
                        }
                    }
                    let dj = c.at(pjj).re + (ajl * t1 + bjl * t2).re;
                    c.set(pjj, Z::r(dj));
                }
            }
        }
    }
}

/// `ZGEMM`. Element-wise per column of C in every branch.
pub(crate) unsafe fn zgemm(
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
    if m == 0 || n == 0 || ((alpha.eq(CZERO) || k == 0) && beta.eq(CONE)) {
        return;
    }
    if alpha.eq(CZERO) {
        for j in 1..=n {
            for i in 1..=m {
                let p = ix(i, j, ldc);
                c.set(p, if beta.eq(CZERO) { CZERO } else { beta * c.at(p) });
            }
        }
        return;
    }
    let bget = |l: usize, j: usize| -> Z {
        match transb {
            Tr::N => b.at(ix(l, j, ldb)),
            Tr::T => b.at(ix(j, l, ldb)),
            Tr::C => b.at(ix(j, l, ldb)).conj(),
        }
    };
    {
        let (lo, hi) = (0usize, n);

        for j in (lo + 1)..=hi {
            let cj = c.off((j - 1) * ldc);
            if transa == Tr::N {
                if beta.eq(CZERO) {
                    let (cr, ci) = cj.zs_mut(m);
                    cr.fill(0.0);
                    ci.fill(0.0);
                } else if !beta.eq(CONE) {
                    zscal_s(beta, cj.zs_mut(m));
                }
                for l in 1..=k {
                    let temp = alpha * bget(l, j);
                    zaxpy_s(temp, a.off((l - 1) * lda).zs(m), cj.zs_mut(m));
                }
            } else {
                let conj = transa == Tr::C;
                let mut i = 0usize;
                if transb == Tr::N {
                    let bj = b.off((j - 1) * ldb).zs(k);
                    while i + ZLANES <= m {
                        let mut acc = [CZERO; ZLANES];
                        zdotl(conj, std::array::from_fn(|q| a.off((i + q) * lda).zs(k)), bj, &mut acc);
                        for q in 0..ZLANES {
                            let p = i + q;
                            let v = if beta.eq(CZERO) { alpha * acc[q] } else { alpha * acc[q] + beta * cj.at(p) };
                            cj.set(p, v);
                        }
                        i += ZLANES;
                    }
                }
                while i < m {
                    let mut temp = CZERO;
                    for l in 1..=k {
                        let ali = a.at(ix(l, i + 1, lda));
                        let ali = if conj { ali.conj() } else { ali };
                        temp = temp + ali * bget(l, j);
                    }
                    let v = if beta.eq(CZERO) { alpha * temp } else { alpha * temp + beta * cj.at(i) };
                    cj.set(i, v);
                    i += 1;
                }
            }
        }
    }
}

/// `ZTRMM`. Left side element-wise per column of B, right side per row.
pub(crate) unsafe fn ztrmm(
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
    if m == 0 || n == 0 {
        return;
    }
    if alpha.eq(CZERO) {
        for j in 1..=n {
            for i in 1..=m {
                b.set(ix(i, j, ldb), CZERO);
            }
        }
        return;
    }
    let noconj = trans == Tr::T;
    let a_ = |i: usize, j: usize| a.at(ix(i, j, lda));
    let ac = |i: usize, j: usize| if noconj { a_(i, j) } else { a_(i, j).conj() };
    let bi = |i: usize, j: usize| ix(i, j, ldb);
    if left {
        {
            let (lo, hi) = (0usize, n);

            for j in (lo + 1)..=hi {
                match (trans == Tr::N, upper) {
                    (true, true) => {
                        for k in 1..=m {
                            if b.at(bi(k, j)).nz() {
                                let mut temp = alpha * b.at(bi(k, j));
                                if k > 1 {
                                    zaxpy_s(temp, a.off(ix(1, k, lda)).zs(k - 1), b.off(bi(1, j)).zs_mut(k - 1));
                                }
                                if nounit {
                                    temp = temp * a_(k, k);
                                }
                                b.set(bi(k, j), temp);
                            }
                        }
                    }
                    (true, false) => {
                        for k in (1..=m).rev() {
                            if b.at(bi(k, j)).nz() {
                                let temp = alpha * b.at(bi(k, j));
                                b.set(bi(k, j), temp);
                                if nounit {
                                    b.set(bi(k, j), b.at(bi(k, j)) * a_(k, k));
                                }
                                if k < m {
                                    zaxpy_s(
                                        temp,
                                        a.off(ix(k + 1, k, lda)).zs(m - k),
                                        b.off(bi(k + 1, j)).zs_mut(m - k),
                                    );
                                }
                            }
                        }
                    }
                    (false, true) => {
                        for i in (1..=m).rev() {
                            let mut temp = b.at(bi(i, j));
                            if nounit {
                                temp = temp * ac(i, i);
                            }
                            for k in 1..i {
                                temp = temp + ac(k, i) * b.at(bi(k, j));
                            }
                            b.set(bi(i, j), alpha * temp);
                        }
                    }
                    (false, false) => {
                        for i in 1..=m {
                            let mut temp = b.at(bi(i, j));
                            if nounit {
                                temp = temp * ac(i, i);
                            }
                            for k in (i + 1)..=m {
                                temp = temp + ac(k, i) * b.at(bi(k, j));
                            }
                            b.set(bi(i, j), alpha * temp);
                        }
                    }
                }
            }
        }
    } else {
        {
            let (lo, hi) = (0usize, m);

            let len = hi - lo;
            let bc = |j: usize| b.off(ix(lo + 1, j, ldb));
            let add = |j: usize, k: usize, temp: Z| zaxpy_s(temp, bc(k).zs(len), bc(j).zs_mut(len));
            let scale = |j: usize, temp: Z| zscal_s(temp, bc(j).zs_mut(len));
            match (trans == Tr::N, upper) {
                (true, true) => {
                    for j in (1..=n).rev() {
                        let mut temp = alpha;
                        if nounit {
                            temp = temp * a_(j, j);
                        }
                        scale(j, temp);
                        for k in 1..j {
                            if a_(k, j).nz() {
                                add(j, k, alpha * a_(k, j));
                            }
                        }
                    }
                }
                (true, false) => {
                    for j in 1..=n {
                        let mut temp = alpha;
                        if nounit {
                            temp = temp * a_(j, j);
                        }
                        scale(j, temp);
                        for k in (j + 1)..=n {
                            if a_(k, j).nz() {
                                add(j, k, alpha * a_(k, j));
                            }
                        }
                    }
                }
                (false, true) => {
                    for k in 1..=n {
                        for j in 1..k {
                            if a_(j, k).nz() {
                                add(j, k, alpha * ac(j, k));
                            }
                        }
                        let mut temp = alpha;
                        if nounit {
                            temp = temp * ac(k, k);
                        }
                        if !temp.eq(CONE) {
                            scale(k, temp);
                        }
                    }
                }
                (false, false) => {
                    for k in (1..=n).rev() {
                        for j in (k + 1)..=n {
                            if a_(j, k).nz() {
                                add(j, k, alpha * ac(j, k));
                            }
                        }
                        let mut temp = alpha;
                        if nounit {
                            temp = temp * ac(k, k);
                        }
                        if !temp.eq(CONE) {
                            scale(k, temp);
                        }
                    }
                }
            }
        }
    }
}

/// gfortran's inline `A / B` for `COMPLEX*16` (`-fcx-fortran-rules`, GCC
/// `expand_complex_div_wide`). Not reached by the eigensolver call tree
/// (`ZLARFG` divides through `ZLADIV`); kept with a component test so a
/// future transcription that needs it starts from verified code.
#[allow(dead_code)]
pub(crate) fn zdiv_gfortran(a: Z, b: Z) -> Z {
    let (ar, ai, br, bi) = (a.re, a.im, b.re, b.im);
    if br.abs() < bi.abs() {
        let ratio = br / bi;
        let div = (br * ratio) + bi;
        let tr = (ar * ratio) + ai;
        let ti = (ai * ratio) - ar;
        Z::new(tr / div, ti / div)
    } else {
        let ratio = bi / br;
        let div = (bi * ratio) + br;
        let tr = (ai * ratio) + ar;
        let ti = ai - (ar * ratio);
        Z::new(tr / div, ti / div)
    }
}
