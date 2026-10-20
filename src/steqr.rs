//! Implicit QL/QR on a symmetric tridiagonal matrix.
//!
//! Transcribed from Reference-LAPACK v3.12.1 `SRC/dsteqr.f` and `SRC/dsterf.f`.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::aux::{
    EPS, SAFMIN, SclType, dlae2, dlaev2, dlanst_m, dlapy2, dlartg, dlascl, dlaset_full, dlasr_rv, dlasrt_inc, sign,
};
use crate::blas::dswap;
use crate::ptr::{D, V, ix};

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;
const TWO: f64 = 2.0;
const THREE: f64 = 3.0;
const MAXIT: usize = 30;

/// `COMPZ` of `DSTEQR`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Compz {
    /// `'N'`: eigenvalues only.
    N,
    /// `'V'`: multiply the input `Z` by the eigenvectors of the tridiagonal.
    V,
    /// `'I'`: `Z` is initialised to the identity.
    I,
}

/// `DSTEQR` on raw views. Returns `INFO` (0 or the number of unconverged
/// off-diagonals). `work` needs `max(1, 2*(n-1))` entries when `compz != N`.
pub(crate) unsafe fn dsteqr_raw(compz: Compz, n: usize, d: D, e: D, z: D, ldz: usize, work: D) -> usize {
    let icompz = match compz {
        Compz::N => 0,
        Compz::V => 1,
        Compz::I => 2,
    };
    let mut info = 0usize;
    if n == 0 {
        return 0;
    }
    if n == 1 {
        if icompz == 2 {
            z.set(0, ONE);
        }
        return 0;
    }
    let dd = |i: usize| d.at(i - 1);
    let ee = |i: usize| e.at(i - 1);
    let eps = EPS;
    let eps2 = eps * eps;
    let safmin = SAFMIN;
    let safmax = ONE / safmin;
    let ssfmax = safmax.sqrt() / THREE;
    let ssfmin = safmin.sqrt() / eps2;

    if icompz == 2 {
        dlaset_full(n, n, ZERO, ONE, z, ldz);
    }
    let nmaxit = n * MAXIT;
    let mut jtot = 0usize;
    let mut l1 = 1usize;
    let nm1 = n - 1;

    'l10: loop {
        if l1 > n {
            break 'l10;
        }
        if l1 > 1 {
            e.set(l1 - 2, ZERO);
        }
        let mut m = n;
        if l1 <= nm1 {
            for mm in l1..=nm1 {
                let tst = ee(mm).abs();
                if tst == ZERO {
                    m = mm;
                    break;
                }
                if tst <= (dd(mm).abs().sqrt() * dd(mm + 1).abs().sqrt()) * eps {
                    e.set(mm - 1, ZERO);
                    m = mm;
                    break;
                }
            }
        }
        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue 'l10;
        }
        let anorm = dlanst_m(lend - l + 1, d.off(l - 1), e.off(l - 1));
        let mut iscale = 0;
        if anorm == ZERO {
            continue 'l10;
        }
        if anorm > ssfmax {
            iscale = 1;
            dlascl(SclType::G, anorm, ssfmax, lend - l + 1, 1, d.off(l - 1), n);
            dlascl(SclType::G, anorm, ssfmax, lend - l, 1, e.off(l - 1), n);
        } else if anorm < ssfmin {
            iscale = 2;
            dlascl(SclType::G, anorm, ssfmin, lend - l + 1, 1, d.off(l - 1), n);
            dlascl(SclType::G, anorm, ssfmin, lend - l, 1, e.off(l - 1), n);
        }
        if dd(lend).abs() < dd(l).abs() {
            lend = lsv;
            l = lendsv;
        }

        if lend > l {
            // QL iteration.
            'l40: loop {
                let mut m = lend;
                if l != lend {
                    for mm in l..lend {
                        let t = ee(mm).abs();
                        let tst = t * t;
                        if tst <= (eps2 * dd(mm).abs()) * dd(mm + 1).abs() + safmin {
                            m = mm;
                            break;
                        }
                    }
                }
                if m < lend {
                    e.set(m - 1, ZERO);
                }
                let mut p = dd(l);
                if m == l {
                    // 80
                    d.set(l - 1, p);
                    l += 1;
                    if l <= lend {
                        continue 'l40;
                    }
                    break 'l40;
                }
                if m == l + 1 {
                    let (rt1, rt2);
                    if icompz > 0 {
                        let (r1, r2, c, s) = dlaev2(dd(l), ee(l), dd(l + 1));
                        rt1 = r1;
                        rt2 = r2;
                        work.set(l - 1, c);
                        work.set(n - 1 + l - 1, s);
                        dlasr_rv(false, n, 2, work.off(l - 1), work.off(n - 1 + l - 1), z.off((l - 1) * ldz), ldz);
                    } else {
                        (rt1, rt2) = dlae2(dd(l), ee(l), dd(l + 1));
                    }
                    d.set(l - 1, rt1);
                    d.set(l, rt2);
                    e.set(l - 1, ZERO);
                    l += 2;
                    if l <= lend {
                        continue 'l40;
                    }
                    break 'l40;
                }
                if jtot == nmaxit {
                    break 'l40;
                }
                jtot += 1;
                let mut g = (dd(l + 1) - p) / (TWO * ee(l));
                let mut r = dlapy2(g, ONE);
                g = dd(m) - p + (ee(l) / (g + sign(r, g)));
                let mut s = ONE;
                let mut c = ONE;
                p = ZERO;
                let mm1 = m - 1;
                for i in (l..=mm1).rev() {
                    let f = s * ee(i);
                    let b = c * ee(i);
                    (c, s, r) = dlartg(g, f);
                    if i != m - 1 {
                        e.set(i, r);
                    }
                    g = dd(i + 1) - p;
                    r = (dd(i) - g) * s + TWO * c * b;
                    p = s * r;
                    d.set(i, g + p);
                    g = c * r - b;
                    if icompz > 0 {
                        work.set(i - 1, c);
                        work.set(n - 1 + i - 1, -s);
                    }
                }
                if icompz > 0 {
                    let mm = m - l + 1;
                    dlasr_rv(false, n, mm, work.off(l - 1), work.off(n - 1 + l - 1), z.off((l - 1) * ldz), ldz);
                }
                d.set(l - 1, dd(l) - p);
                e.set(l - 1, g);
            }
        } else {
            // QR iteration.
            'l90: loop {
                let mut m = lend;
                if l != lend {
                    for mm in ((lend + 1)..=l).rev() {
                        let t = ee(mm - 1).abs();
                        let tst = t * t;
                        if tst <= (eps2 * dd(mm).abs()) * dd(mm - 1).abs() + safmin {
                            m = mm;
                            break;
                        }
                    }
                }
                if m > lend {
                    e.set(m - 2, ZERO);
                }
                let mut p = dd(l);
                if m == l {
                    // 130
                    d.set(l - 1, p);
                    if l == lend {
                        break 'l90;
                    }
                    l -= 1;
                    if l >= lend {
                        continue 'l90;
                    }
                    break 'l90;
                }
                if m + 1 == l {
                    let (rt1, rt2);
                    if icompz > 0 {
                        let (r1, r2, c, s) = dlaev2(dd(l - 1), ee(l - 1), dd(l));
                        rt1 = r1;
                        rt2 = r2;
                        work.set(m - 1, c);
                        work.set(n - 1 + m - 1, s);
                        dlasr_rv(true, n, 2, work.off(m - 1), work.off(n - 1 + m - 1), z.off((l - 2) * ldz), ldz);
                    } else {
                        (rt1, rt2) = dlae2(dd(l - 1), ee(l - 1), dd(l));
                    }
                    d.set(l - 2, rt1);
                    d.set(l - 1, rt2);
                    e.set(l - 2, ZERO);
                    // L = L - 2; IF( L.GE.LEND ) GO TO 90
                    if l < lend + 2 {
                        break 'l90;
                    }
                    l -= 2;
                    continue 'l90;
                }
                if jtot == nmaxit {
                    break 'l90;
                }
                jtot += 1;
                let mut g = (dd(l - 1) - p) / (TWO * ee(l - 1));
                let mut r = dlapy2(g, ONE);
                g = dd(m) - p + (ee(l - 1) / (g + sign(r, g)));
                let mut s = ONE;
                let mut c = ONE;
                p = ZERO;
                let lm1 = l - 1;
                for i in m..=lm1 {
                    let f = s * ee(i);
                    let b = c * ee(i);
                    (c, s, r) = dlartg(g, f);
                    if i != m {
                        e.set(i - 2, r);
                    }
                    g = dd(i) - p;
                    r = (dd(i + 1) - g) * s + TWO * c * b;
                    p = s * r;
                    d.set(i - 1, g + p);
                    g = c * r - b;
                    if icompz > 0 {
                        work.set(i - 1, c);
                        work.set(n - 1 + i - 1, s);
                    }
                }
                if icompz > 0 {
                    let mm = l - m + 1;
                    dlasr_rv(true, n, mm, work.off(m - 1), work.off(n - 1 + m - 1), z.off((m - 1) * ldz), ldz);
                }
                d.set(l - 1, dd(l) - p);
                e.set(lm1 - 1, g);
            }
        }

        // 140: undo scaling.
        if iscale == 1 {
            dlascl(SclType::G, ssfmax, anorm, lendsv - lsv + 1, 1, d.off(lsv - 1), n);
            dlascl(SclType::G, ssfmax, anorm, lendsv - lsv, 1, e.off(lsv - 1), n);
        } else if iscale == 2 {
            dlascl(SclType::G, ssfmin, anorm, lendsv - lsv + 1, 1, d.off(lsv - 1), n);
            dlascl(SclType::G, ssfmin, anorm, lendsv - lsv, 1, e.off(lsv - 1), n);
        }
        if jtot < nmaxit {
            continue 'l10;
        }
        for i in 1..n {
            if ee(i) != ZERO {
                info += 1;
            }
        }
        return info;
    }

    // 160: sort eigenvalues (and vectors) in increasing order.
    if icompz == 0 {
        dlasrt_inc(n, d);
    } else {
        for ii in 2..=n {
            let i = ii - 1;
            let mut k = i;
            let mut p = dd(i);
            for j in ii..=n {
                if dd(j) < p {
                    k = j;
                    p = dd(j);
                }
            }
            if k != i {
                d.set(k - 1, dd(i));
                d.set(i - 1, p);
                dswap(n, z.off(ix(1, i, ldz)), 1, z.off(ix(1, k, ldz)), 1);
            }
        }
    }
    info
}

/// `DSTERF` on raw views. Returns `INFO`.
pub(crate) unsafe fn dsterf_raw(n: usize, d: D, e: D) -> usize {
    let mut info = 0usize;
    if n <= 1 {
        return 0;
    }
    let dd = |i: usize| d.at(i - 1);
    let ee = |i: usize| e.at(i - 1);
    let eps = EPS;
    let eps2 = eps * eps;
    let safmin = SAFMIN;
    let safmax = ONE / safmin;
    let ssfmax = safmax.sqrt() / THREE;
    let ssfmin = safmin.sqrt() / eps2;
    let nmaxit = n * MAXIT;
    let mut jtot = 0usize;
    let mut l1 = 1usize;

    'l10: loop {
        if l1 > n {
            break 'l10;
        }
        if l1 > 1 {
            e.set(l1 - 2, ZERO);
        }
        let mut m = n;
        for mm in l1..n {
            if ee(mm).abs() <= (dd(mm).abs().sqrt() * dd(mm + 1).abs().sqrt()) * eps {
                e.set(mm - 1, ZERO);
                m = mm;
                break;
            }
        }
        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue 'l10;
        }
        let anorm = dlanst_m(lend - l + 1, d.off(l - 1), e.off(l - 1));
        let mut iscale = 0;
        if anorm == ZERO {
            continue 'l10;
        }
        if anorm > ssfmax {
            iscale = 1;
            dlascl(SclType::G, anorm, ssfmax, lend - l + 1, 1, d.off(l - 1), n);
            dlascl(SclType::G, anorm, ssfmax, lend - l, 1, e.off(l - 1), n);
        } else if anorm < ssfmin {
            iscale = 2;
            dlascl(SclType::G, anorm, ssfmin, lend - l + 1, 1, d.off(l - 1), n);
            dlascl(SclType::G, anorm, ssfmin, lend - l, 1, e.off(l - 1), n);
        }
        for i in l..lend {
            let t = ee(i);
            e.set(i - 1, t * t);
        }
        if dd(lend).abs() < dd(l).abs() {
            lend = lsv;
            l = lendsv;
        }

        if lend >= l {
            // QL iteration.
            'l50: loop {
                let mut m = lend;
                if l != lend {
                    for mm in l..lend {
                        if ee(mm).abs() <= eps2 * (dd(mm) * dd(mm + 1)).abs() {
                            m = mm;
                            break;
                        }
                    }
                }
                if m < lend {
                    e.set(m - 1, ZERO);
                }
                let p = dd(l);
                if m == l {
                    d.set(l - 1, p);
                    l += 1;
                    if l <= lend {
                        continue 'l50;
                    }
                    break 'l50;
                }
                if m == l + 1 {
                    let rte = ee(l).sqrt();
                    let (rt1, rt2) = dlae2(dd(l), rte, dd(l + 1));
                    d.set(l - 1, rt1);
                    d.set(l, rt2);
                    e.set(l - 1, ZERO);
                    l += 2;
                    if l <= lend {
                        continue 'l50;
                    }
                    break 'l50;
                }
                if jtot == nmaxit {
                    break 'l50;
                }
                jtot += 1;
                let rte = ee(l).sqrt();
                let mut sigma = (dd(l + 1) - p) / (TWO * rte);
                let r = dlapy2(sigma, ONE);
                sigma = p - (rte / (sigma + sign(r, sigma)));
                let mut c = ONE;
                let mut s = ZERO;
                let mut gamma = dd(m) - sigma;
                let mut p = gamma * gamma;
                for i in (l..m).rev() {
                    let bb = ee(i);
                    let r = p + bb;
                    if i != m - 1 {
                        e.set(i, s * r);
                    }
                    let oldc = c;
                    c = p / r;
                    s = bb / r;
                    let oldgam = gamma;
                    let alpha = dd(i);
                    gamma = c * (alpha - sigma) - s * oldgam;
                    d.set(i, oldgam + (alpha - gamma));
                    if c != ZERO {
                        p = (gamma * gamma) / c;
                    } else {
                        p = oldc * bb;
                    }
                }
                e.set(l - 1, s * p);
                d.set(l - 1, sigma + gamma);
            }
        } else {
            // QR iteration.
            'l100: loop {
                let mut m = lend;
                for mm in ((lend + 1)..=l).rev() {
                    if ee(mm - 1).abs() <= eps2 * (dd(mm) * dd(mm - 1)).abs() {
                        m = mm;
                        break;
                    }
                }
                if m > lend {
                    e.set(m - 2, ZERO);
                }
                let p = dd(l);
                if m == l {
                    d.set(l - 1, p);
                    if l == lend {
                        break 'l100;
                    }
                    l -= 1;
                    if l >= lend {
                        continue 'l100;
                    }
                    break 'l100;
                }
                if m + 1 == l {
                    let rte = ee(l - 1).sqrt();
                    let (rt1, rt2) = dlae2(dd(l), rte, dd(l - 1));
                    d.set(l - 1, rt1);
                    d.set(l - 2, rt2);
                    e.set(l - 2, ZERO);
                    if l < lend + 2 {
                        break 'l100;
                    }
                    l -= 2;
                    continue 'l100;
                }
                if jtot == nmaxit {
                    break 'l100;
                }
                jtot += 1;
                let rte = ee(l - 1).sqrt();
                let mut sigma = (dd(l - 1) - p) / (TWO * rte);
                let r = dlapy2(sigma, ONE);
                sigma = p - (rte / (sigma + sign(r, sigma)));
                let mut c = ONE;
                let mut s = ZERO;
                let mut gamma = dd(m) - sigma;
                let mut p = gamma * gamma;
                for i in m..l {
                    let bb = ee(i);
                    let r = p + bb;
                    if i != m {
                        e.set(i - 2, s * r);
                    }
                    let oldc = c;
                    c = p / r;
                    s = bb / r;
                    let oldgam = gamma;
                    let alpha = dd(i + 1);
                    gamma = c * (alpha - sigma) - s * oldgam;
                    d.set(i - 1, oldgam + (alpha - gamma));
                    if c != ZERO {
                        p = (gamma * gamma) / c;
                    } else {
                        p = oldc * bb;
                    }
                }
                e.set(l - 2, s * p);
                d.set(l - 1, sigma + gamma);
            }
        }

        // 150: undo scaling.
        if iscale == 1 {
            dlascl(SclType::G, ssfmax, anorm, lendsv - lsv + 1, 1, d.off(lsv - 1), n);
        }
        if iscale == 2 {
            dlascl(SclType::G, ssfmin, anorm, lendsv - lsv + 1, 1, d.off(lsv - 1), n);
        }
        if jtot < nmaxit {
            continue 'l10;
        }
        for i in 1..n {
            if ee(i) != ZERO {
                info += 1;
            }
        }
        return info;
    }
    dlasrt_inc(n, d);
    info
}

/// Safe `DSTEQR`: `d` (n), `e` (n-1, may be longer), `z` (ldz*n when
/// `compz != N`), `work` (`max(1, 2*(n-1))` when `compz != N`).
pub fn dsteqr(
    compz: Compz,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    z: &mut [f64],
    ldz: usize,
    work: &mut [f64],
) -> Result<(), crate::LapackError> {
    use crate::LapackError::{InvalidArgument, NoConvergence};
    if d.len() < n || e.len() < n.saturating_sub(1) {
        return Err(InvalidArgument { arg: "d/e", reason: format!("need n={n} and n-1 entries") });
    }
    if compz != Compz::N {
        if ldz < n.max(1) || z.len() < ldz * n {
            return Err(InvalidArgument { arg: "z", reason: format!("need ldz>=n and ldz*n entries (n={n})") });
        }
        if work.len() < (2 * n.saturating_sub(1)).max(1) {
            return Err(crate::LapackError::WorkspaceTooSmall {
                need: (2 * n.saturating_sub(1)).max(1),
                got: work.len(),
            });
        }
    }
    if n == 0 {
        return Ok(());
    }
    let mut ezero = [0.0f64; 1];
    let ev = if e.is_empty() { V::new(&mut ezero) } else { V::new(e) };
    let mut zzero = [0.0f64; 1];
    let zv = if z.is_empty() { V::new(&mut zzero) } else { V::new(z) };
    let mut wzero = [0.0f64; 1];
    let wv = if work.is_empty() { V::new(&mut wzero) } else { V::new(work) };
    // SAFETY: lengths checked above; d, e, z, work are distinct slices.
    let info = unsafe { dsteqr_raw(compz, n, V::new(d), ev, zv, ldz.max(1), wv) };
    if info != 0 { Err(NoConvergence { routine: "dsteqr", info }) } else { Ok(()) }
}

/// Safe `DSTERF`.
pub fn dsterf(n: usize, d: &mut [f64], e: &mut [f64]) -> Result<(), crate::LapackError> {
    use crate::LapackError::{InvalidArgument, NoConvergence};
    if d.len() < n || e.len() < n.saturating_sub(1) {
        return Err(InvalidArgument { arg: "d/e", reason: format!("need n={n} and n-1 entries") });
    }
    if n <= 1 {
        return Ok(());
    }
    // SAFETY: lengths checked above.
    let info = unsafe { dsterf_raw(n, V::new(d), V::new(e)) };
    if info != 0 { Err(NoConvergence { routine: "dsterf", info }) } else { Ok(()) }
}
