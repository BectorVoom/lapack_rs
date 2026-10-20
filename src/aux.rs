//! LAPACK auxiliary routines (Reference-LAPACK v3.12.1).
//!
//! Transcribed from `INSTALL/dlamch.f`, `SRC/dlapy2.f`, `SRC/dlartg.f90`,
//! `SRC/dlae2.f`, `SRC/dlaev2.f`, `SRC/dlasr.f`, `SRC/dlasrt.f`,
//! `SRC/dlascl.f`, `SRC/dlanst.f`, `SRC/dlansy.f`, `SRC/dlaset.f`,
//! `SRC/dlacpy.f`, `SRC/dlamrg.f`, plus libgcc's `__powidf2` (the lowering of
//! `x**n` for a variable default-kind integer `n`).
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::blas::pow2;
use crate::ptr::{D, I, ix};

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;
const TWO: f64 = 2.0;
const HALF: f64 = 0.5;

/// `DLAMCH('E')`: `EPSILON(0d0)*0.5` (rounding mode is to-nearest).
pub(crate) const EPS: f64 = pow2(-53);
/// `DLAMCH('S')`: `TINY(0d0)` (`1/HUGE` is smaller).
pub(crate) const SAFMIN: f64 = pow2(-1022);
/// `DLAMCH('P')`: `EPS*RADIX`.
pub(crate) const PREC: f64 = pow2(-52);
/// `DLAMCH('O')`: `HUGE(0d0)`.
pub(crate) const HUGEVAL: f64 = f64::MAX;
/// `DLAMCH('B')`.
pub(crate) const BASE: f64 = 2.0;

/// libgcc `__powidf2`: `x**m` by binary exponentiation, in libgcc's step order.
pub(crate) fn powidf2(mut x: f64, m: i32) -> f64 {
    let mut n = m.unsigned_abs();
    let mut y = if n % 2 != 0 { x } else { 1.0 };
    loop {
        n >>= 1;
        if n == 0 {
            break;
        }
        x *= x;
        if n % 2 != 0 {
            y *= x;
        }
    }
    if m < 0 { 1.0 / y } else { y }
}

/// Fortran `SIGN(A, B)` for reals as gfortran lowers it (`copysign`).
#[inline(always)]
pub(crate) fn sign(a: f64, b: f64) -> f64 {
    a.copysign(b)
}

/// gfortran `MAX(A, B)` lowered to `MAX_EXPR` (x86 `maxsd`: second operand
/// on equality or NaN).
#[inline(always)]
pub(crate) fn fmax(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// gfortran `MIN(A, B)` lowered to `MIN_EXPR` (x86 `minsd`).
#[inline(always)]
pub(crate) fn fmin(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// `DLAPY2`.
pub(crate) fn dlapy2(x: f64, y: f64) -> f64 {
    let x_is_nan = x.is_nan();
    let y_is_nan = y.is_nan();
    let mut r = 0.0;
    if x_is_nan {
        r = x;
    }
    if y_is_nan {
        r = y;
    }
    if !(x_is_nan || y_is_nan) {
        let xabs = x.abs();
        let yabs = y.abs();
        let w = fmax(xabs, yabs);
        let z = fmin(xabs, yabs);
        if z == ZERO || w > HUGEVAL {
            r = w;
        } else {
            let q = z / w;
            r = w * (ONE + q * q).sqrt();
        }
    }
    r
}

/// `DLARTG` (3.10+ f90). Returns `(c, s, r)`.
pub(crate) fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    let safmin = SAFMIN;
    let safmax = ONE / SAFMIN;
    let rtmin = safmin.sqrt();
    let rtmax = (safmax / 2.0).sqrt();
    let f1 = f.abs();
    let g1 = g.abs();
    if g == ZERO {
        (ONE, ZERO, f)
    } else if f == ZERO {
        (ZERO, sign(ONE, g), g1)
    } else if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = (f * f + g * g).sqrt();
        let c = f1 / d;
        let r = sign(d, f);
        let s = g / r;
        (c, s, r)
    } else {
        let u = fmin(safmax, fmax(fmax(safmin, f1), g1));
        let fs = f / u;
        let gs = g / u;
        let d = (fs * fs + gs * gs).sqrt();
        let c = fs.abs() / d;
        let r = sign(d, f);
        let s = gs / r;
        (c, s, r * u)
    }
}

/// `DLAE2`. Returns `(rt1, rt2)`.
pub(crate) fn dlae2(a: f64, b: f64, c: f64) -> (f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        let q = ab / adf;
        adf * (ONE + q * q).sqrt()
    } else if adf < ab {
        let q = adf / ab;
        ab * (ONE + q * q).sqrt()
    } else {
        ab * TWO.sqrt()
    };
    if sm < ZERO {
        let rt1 = HALF * (sm - rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b)
    } else if sm > ZERO {
        let rt1 = HALF * (sm + rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b)
    } else {
        (HALF * rt, -HALF * rt)
    }
}

/// `DLAEV2`. Returns `(rt1, rt2, cs1, sn1)`.
pub(crate) fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        let q = ab / adf;
        adf * (ONE + q * q).sqrt()
    } else if adf < ab {
        let q = adf / ab;
        ab * (ONE + q * q).sqrt()
    } else {
        ab * TWO.sqrt()
    };
    let (rt1, rt2, sgn1);
    if sm < ZERO {
        rt1 = HALF * (sm - rt);
        sgn1 = -1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if sm > ZERO {
        rt1 = HALF * (sm + rt);
        sgn1 = 1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else {
        rt1 = HALF * rt;
        rt2 = -HALF * rt;
        sgn1 = 1;
    }
    let (cs, sgn2) = if df >= ZERO { (df + rt, 1) } else { (df - rt, -1) };
    let acs = cs.abs();
    let (mut cs1, mut sn1);
    if acs > ab {
        let ct = -tb / cs;
        sn1 = ONE / (ONE + ct * ct).sqrt();
        cs1 = ct * sn1;
    } else if ab == ZERO {
        cs1 = ONE;
        sn1 = ZERO;
    } else {
        let tn = -cs / tb;
        cs1 = ONE / (ONE + tn * tn).sqrt();
        sn1 = tn * cs1;
    }
    if sgn1 == sgn2 {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    (rt1, rt2, cs1, sn1)
}

/// `DLASR` with `SIDE = 'R'`, `PIVOT = 'V'`; `forward` selects `DIRECT = 'F'`.
pub(crate) unsafe fn dlasr_rv(forward: bool, m: usize, n: usize, c: D, s: D, a: D, lda: usize) {
    if m == 0 || n == 0 {
        return;
    }
    let step = |j: usize| {
        let ctemp = c.at(j - 1);
        let stemp = s.at(j - 1);
        if ctemp != ONE || stemp != ZERO {
            for i in 1..=m {
                let p1 = ix(i, j + 1, lda);
                let p0 = ix(i, j, lda);
                let temp = a.at(p1);
                a.set(p1, ctemp * temp - stemp * a.at(p0));
                a.set(p0, stemp * temp + ctemp * a.at(p0));
            }
        }
    };
    if forward {
        for j in 1..n {
            step(j);
        }
    } else {
        for j in (1..n).rev() {
            step(j);
        }
    }
}

/// `DLASRT('D', ...)`: decreasing order, the reference's quicksort with the
/// same median-of-three pivot, the same partition scan and the same
/// largest-part-first stack order, so equal and signed-zero elements land in
/// the reference's positions.
pub(crate) unsafe fn dlasrt_dec(n: usize, d: D) {
    const SELECT: usize = 20;
    if n <= 1 {
        return;
    }
    let dd = |i: usize| d.at(i - 1);
    let mut stack = [[0usize; 2]; 32];
    let mut stkpnt = 1usize;
    stack[0] = [1, n];
    loop {
        let start = stack[stkpnt - 1][0];
        let endd = stack[stkpnt - 1][1];
        stkpnt -= 1;
        if endd - start <= SELECT && endd > start {
            // Insertion sort (DIR = 0).
            for i in (start + 1)..=endd {
                let mut j = i;
                while j > start {
                    if dd(j) > dd(j - 1) {
                        let dmnmx = dd(j);
                        d.set(j - 1, dd(j - 1));
                        d.set(j - 2, dmnmx);
                    } else {
                        break;
                    }
                    j -= 1;
                }
            }
        } else if endd - start > SELECT {
            let d1 = dd(start);
            let d2 = dd(endd);
            let i = (start + endd) / 2;
            let d3 = dd(i);
            let dmnmx = if d1 < d2 {
                if d3 < d1 {
                    d1
                } else if d3 < d2 {
                    d3
                } else {
                    d2
                }
            } else if d3 < d2 {
                d2
            } else if d3 < d1 {
                d3
            } else {
                d1
            };
            let mut i = start as isize - 1;
            let mut j = endd as isize + 1;
            loop {
                loop {
                    j -= 1;
                    if !(dd(j as usize) < dmnmx) {
                        break;
                    }
                }
                loop {
                    i += 1;
                    if !(dd(i as usize) > dmnmx) {
                        break;
                    }
                }
                if i < j {
                    let tmp = dd(i as usize);
                    d.set(i as usize - 1, dd(j as usize));
                    d.set(j as usize - 1, tmp);
                } else {
                    break;
                }
            }
            let j = j as usize;
            if j - start > endd - j - 1 {
                stkpnt += 1;
                stack[stkpnt - 1] = [start, j];
                stkpnt += 1;
                stack[stkpnt - 1] = [j + 1, endd];
            } else {
                stkpnt += 1;
                stack[stkpnt - 1] = [j + 1, endd];
                stkpnt += 1;
                stack[stkpnt - 1] = [start, j];
            }
        }
        if stkpnt == 0 {
            break;
        }
    }
}

/// `DLASRT('I', ...)`.
pub(crate) unsafe fn dlasrt_inc(n: usize, d: D) {
    const SELECT: usize = 20;
    if n <= 1 {
        return;
    }
    let dd = |i: usize| d.at(i - 1);
    let mut stack = [[0usize; 2]; 32];
    let mut stkpnt = 1usize;
    stack[0] = [1, n];
    loop {
        let start = stack[stkpnt - 1][0];
        let endd = stack[stkpnt - 1][1];
        stkpnt -= 1;
        if endd - start <= SELECT && endd > start {
            // Insertion sort (DIR = 1).
            for i in (start + 1)..=endd {
                let mut j = i;
                while j > start {
                    if dd(j) < dd(j - 1) {
                        let dmnmx = dd(j);
                        d.set(j - 1, dd(j - 1));
                        d.set(j - 2, dmnmx);
                    } else {
                        break;
                    }
                    j -= 1;
                }
            }
        } else if endd - start > SELECT {
            let d1 = dd(start);
            let d2 = dd(endd);
            let i = (start + endd) / 2;
            let d3 = dd(i);
            let dmnmx = if d1 < d2 {
                if d3 < d1 {
                    d1
                } else if d3 < d2 {
                    d3
                } else {
                    d2
                }
            } else if d3 < d2 {
                d2
            } else if d3 < d1 {
                d3
            } else {
                d1
            };
            // Signed indices: I starts at START-1, J at ENDD+1.
            let mut i = start as isize - 1;
            let mut j = endd as isize + 1;
            loop {
                loop {
                    j -= 1;
                    if !(dd(j as usize) > dmnmx) {
                        break;
                    }
                }
                loop {
                    i += 1;
                    if !(dd(i as usize) < dmnmx) {
                        break;
                    }
                }
                if i < j {
                    let tmp = dd(i as usize);
                    d.set(i as usize - 1, dd(j as usize));
                    d.set(j as usize - 1, tmp);
                } else {
                    break;
                }
            }
            let j = j as usize;
            if j - start > endd - j - 1 {
                stkpnt += 1;
                stack[stkpnt - 1] = [start, j];
                stkpnt += 1;
                stack[stkpnt - 1] = [j + 1, endd];
            } else {
                stkpnt += 1;
                stack[stkpnt - 1] = [j + 1, endd];
                stkpnt += 1;
                stack[stkpnt - 1] = [start, j];
            }
        }
        if stkpnt == 0 {
            break;
        }
    }
}

/// `DLASCL` matrix types used by the drivers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SclType {
    G,
    L,
    U,
}

/// `DLASCL` for types `'G'`, `'L'`, `'U'`.
pub(crate) unsafe fn dlascl(ty: SclType, cfrom: f64, cto: f64, m: usize, n: usize, a: D, lda: usize) {
    if n == 0 || m == 0 {
        return;
    }
    let smlnum = SAFMIN;
    let bignum = ONE / smlnum;
    let mut cfromc = cfrom;
    let mut ctoc = cto;
    loop {
        let cfrom1 = cfromc * smlnum;
        let mul;
        let done;
        if cfrom1 == cfromc {
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                mul = ctoc;
                done = true;
                cfromc = ONE;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != ZERO {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
                if mul == ONE {
                    return;
                }
            }
        }
        for j in 1..=n {
            let (i0, i1) = match ty {
                SclType::G => (1, m),
                SclType::L => (j, m),
                SclType::U => (1, j.min(m)),
            };
            for i in i0..=i1 {
                let p = ix(i, j, lda);
                a.set(p, a.at(p) * mul);
            }
        }
        if done {
            return;
        }
    }
}

/// `DLANST('M', ...)`.
pub(crate) unsafe fn dlanst_m(n: usize, d: D, e: D) -> f64 {
    if n == 0 {
        return ZERO;
    }
    let mut anorm = d.at(n - 1).abs();
    for i in 1..n {
        let sum = d.at(i - 1).abs();
        if anorm < sum || sum.is_nan() {
            anorm = sum;
        }
        let sum = e.at(i - 1).abs();
        if anorm < sum || sum.is_nan() {
            anorm = sum;
        }
    }
    anorm
}

/// `DLANSY('M', UPLO, ...)`.
pub(crate) unsafe fn dlansy_m(upper: bool, n: usize, a: D, lda: usize) -> f64 {
    let mut value = ZERO;
    for j in 1..=n {
        let (i0, i1) = if upper { (1, j) } else { (j, n) };
        for i in i0..=i1 {
            let sum = a.at(ix(i, j, lda)).abs();
            if value < sum || sum.is_nan() {
                value = sum;
            }
        }
    }
    value
}

/// `DLASET('Full', M, N, ALPHA, BETA, A, LDA)`.
pub(crate) unsafe fn dlaset_full(m: usize, n: usize, alpha: f64, beta: f64, a: D, lda: usize) {
    for j in 1..=n {
        for i in 1..=m {
            a.set(ix(i, j, lda), alpha);
        }
    }
    for i in 1..=m.min(n) {
        a.set(ix(i, i, lda), beta);
    }
}

/// `DLACPY('A', M, N, A, LDA, B, LDB)`.
pub(crate) unsafe fn dlacpy_all(m: usize, n: usize, a: D, lda: usize, b: D, ldb: usize) {
    for j in 1..=n {
        for i in 1..=m {
            b.set(ix(i, j, ldb), a.at(ix(i, j, lda)));
        }
    }
}

/// `DLAMRG`. `dtrd1`/`dtrd2` are `+1` or `-1`; `index` receives 1-based
/// positions.
pub(crate) unsafe fn dlamrg(n1: usize, n2: usize, a: D, dtrd1: isize, dtrd2: isize, index: I) {
    let mut n1sv = n1;
    let mut n2sv = n2;
    let mut ind1: isize = if dtrd1 > 0 { 1 } else { n1 as isize };
    let mut ind2: isize = if dtrd2 > 0 { 1 + n1 as isize } else { (n1 + n2) as isize };
    let mut i = 1usize;
    while n1sv > 0 && n2sv > 0 {
        if a.at(ind1 as usize - 1) <= a.at(ind2 as usize - 1) {
            index.set(i - 1, ind1 as usize);
            i += 1;
            ind1 += dtrd1;
            n1sv -= 1;
        } else {
            index.set(i - 1, ind2 as usize);
            i += 1;
            ind2 += dtrd2;
            n2sv -= 1;
        }
    }
    if n1sv == 0 {
        for _ in 1..=n2sv {
            index.set(i - 1, ind2 as usize);
            i += 1;
            ind2 += dtrd2;
        }
    } else {
        for _ in 1..=n1sv {
            index.set(i - 1, ind1 as usize);
            i += 1;
            ind1 += dtrd1;
        }
    }
}
