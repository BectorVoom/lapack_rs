//! K4: `DLAED3` on the device — the secular roots (`DLAED4`/`DLAED6`), the
//! Löwner products `W` and the normalised eigenvector columns `U`.
//!
//! Line-by-line mirrors of [`crate::laed4`] (Reference-LAPACK 3.12.1
//! `dlaed4.f`, `dlaed6.f`) and of the `K > 2` branch of `dlaed3.f`, with the
//! same operand order, branches and gfortran lowerings (`MAX(a,b)` =
//! `a > b ? a : b`, `MIN(a,b)` = `a < b ? a : b`, `SIGN` from a host sign
//! flag). Every root, row and column is an independent unit, so the results
//! are bit-identical to the host at any launch geometry.
//!
//! Layout: `delta` is `K × K` column-major (`DELTA` of root `j` in column
//! `j`), `d` = `DLAMBDA`, `z` = `W` (the deflated z-vector).

// `let mut x = zero;` before branch assignment is how a #[cube] kernel
// declares a variable; rustc sees the initial value as unused.
#![allow(unused_assignments)]

use cubecl::prelude::*;

// One lowering of MAX/MIN in the crate (`kernels/gf.rs`).
use super::gf::{fmax_c, fmin_c};

/// `DLAED6` (`d`, `z` passed as scalars). Returns `TAU`; sets `info[slot]`
/// to 1 on non-convergence.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn dlaed6_c<F: Float>(
    kniter: u32,
    orgati: bool,
    rho: F,
    d0: F,
    d1: F,
    d2: F,
    z0: F,
    z1: F,
    z2: F,
    finit: F,
    eps: F,
    small1: F,
    info: &mut Array<u32>,
    slot: usize,
) -> F {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let two = F::from_int(2);
    let four = F::from_int(4);
    let eight = F::from_int(8);
    let mut lbd = d0;
    let mut ubd = d1;
    if orgati {
        lbd = d1;
        ubd = d2;
    }
    if finit < zero {
        lbd = zero;
    } else {
        ubd = zero;
    }
    let mut tau = zero;
    if kniter == 2 {
        let mut a = zero;
        let mut b = zero;
        let mut c = zero;
        if orgati {
            let temp = (d2 - d1) / two;
            c = rho + z0 / ((d0 - d1) - temp);
            a = c * (d1 + d2) + z1 + z2;
            b = c * d1 * d2 + z1 * d2 + z2 * d1;
        } else {
            let temp = (d0 - d1) / two;
            c = rho + z2 / ((d2 - d1) - temp);
            a = c * (d0 + d1) + z0 + z1;
            b = c * d0 * d1 + z0 * d1 + z1 * d0;
        }
        let temp = fmax_c::<F>(fmax_c::<F>(F::abs(a), F::abs(b)), F::abs(c));
        a = a / temp;
        b = b / temp;
        c = c / temp;
        if c == zero {
            tau = b / a;
        } else if a <= zero {
            tau = (a - F::sqrt(F::abs(a * a - four * b * c))) / (two * c);
        } else {
            tau = two * b / (a + F::sqrt(F::abs(a * a - four * b * c)));
        }
        if tau < lbd || tau > ubd {
            tau = (lbd + ubd) / two;
        }
        if d0 == tau || d1 == tau || d2 == tau {
            tau = zero;
        } else {
            let temp =
                finit + tau * z0 / (d0 * (d0 - tau)) + tau * z1 / (d1 * (d1 - tau)) + tau * z2 / (d2 * (d2 - tau));
            if temp <= zero {
                lbd = tau;
            } else {
                ubd = tau;
            }
            if F::abs(finit) <= F::abs(temp) {
                tau = zero;
            }
        }
    }

    let sminv1 = one / small1;
    let small2 = small1 * small1;
    let sminv2 = sminv1 * sminv1;

    let mut temp = fmin_c::<F>(F::abs(d0 - tau), F::abs(d1 - tau));
    if orgati {
        temp = fmin_c::<F>(F::abs(d1 - tau), F::abs(d2 - tau));
    }
    let mut scale: bool = false;
    let mut sclinv = one;
    // DSCALE / ZSCALE as scalars (local arrays in this helper, inlined at
    // two call sites, trip cubecl-opt 0.10's GVN pass).
    let mut ds0 = d0;
    let mut ds1 = d1;
    let mut ds2 = d2;
    let mut zs0 = z0;
    let mut zs1 = z1;
    let mut zs2 = z2;
    if temp <= small1 {
        scale = true;
        let mut sclfac = sminv1;
        sclinv = small1;
        if temp <= small2 {
            sclfac = sminv2;
            sclinv = small2;
        }
        ds0 = d0 * sclfac;
        zs0 = z0 * sclfac;
        ds1 = d1 * sclfac;
        zs1 = z1 * sclfac;
        ds2 = d2 * sclfac;
        zs2 = z2 * sclfac;
        tau = tau * sclfac;
        lbd = lbd * sclfac;
        ubd = ubd * sclfac;
    }

    let mut fc = zero;
    let mut df = zero;
    let mut ddf = zero;
    {
        let temp = one / (ds0 - tau);
        let temp1 = zs0 * temp;
        let temp2 = temp1 * temp;
        let temp3 = temp2 * temp;
        fc = fc + temp1 / ds0;
        df = df + temp2;
        ddf = ddf + temp3;
    }
    {
        let temp = one / (ds1 - tau);
        let temp1 = zs1 * temp;
        let temp2 = temp1 * temp;
        let temp3 = temp2 * temp;
        fc = fc + temp1 / ds1;
        df = df + temp2;
        ddf = ddf + temp3;
    }
    {
        let temp = one / (ds2 - tau);
        let temp1 = zs2 * temp;
        let temp2 = temp1 * temp;
        let temp3 = temp2 * temp;
        fc = fc + temp1 / ds2;
        df = df + temp2;
        ddf = ddf + temp3;
    }
    let mut f = finit + tau * fc;

    let mut done: bool = F::abs(f) <= zero;
    if !done {
        if f <= zero {
            lbd = tau;
        } else {
            ubd = tau;
        }
        // NITER = 1 before the loop: iterations 2..=MAXIT.
        let mut it: u32 = 2u32;
        while it <= 40u32 && !done {
            let mut temp1 = ds0 - tau;
            let mut temp2 = ds1 - tau;
            if orgati {
                temp1 = ds1 - tau;
                temp2 = ds2 - tau;
            }
            let mut a = (temp1 + temp2) * f - temp1 * temp2 * df;
            let mut b = temp1 * temp2 * f;
            let mut c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
            let temp = fmax_c::<F>(fmax_c::<F>(F::abs(a), F::abs(b)), F::abs(c));
            a = a / temp;
            b = b / temp;
            c = c / temp;
            let mut eta = zero;
            if c == zero {
                eta = b / a;
            } else if a <= zero {
                eta = (a - F::sqrt(F::abs(a * a - four * b * c))) / (two * c);
            } else {
                eta = two * b / (a + F::sqrt(F::abs(a * a - four * b * c)));
            }
            if f * eta >= zero {
                eta = -f / df;
            }
            tau = tau + eta;
            if tau < lbd || tau > ubd {
                tau = (lbd + ubd) / two;
            }
            fc = zero;
            let mut erretm = zero;
            df = zero;
            ddf = zero;
            // The reference loop over I = 1..3 exits (GOTO 60) at the first
            // DSCALE(I) = TAU, after the earlier terms were accumulated.
            let hit0: bool = (ds0 - tau) == zero;
            let hit1: bool = (ds1 - tau) == zero;
            let hit2: bool = (ds2 - tau) == zero;
            if !hit0 {
                let temp = one / (ds0 - tau);
                let temp1 = zs0 * temp;
                let temp2 = temp1 * temp;
                let temp3 = temp2 * temp;
                let temp4 = temp1 / ds0;
                fc = fc + temp4;
                erretm = erretm + F::abs(temp4);
                df = df + temp2;
                ddf = ddf + temp3;
                if !hit1 {
                    let temp = one / (ds1 - tau);
                    let temp1 = zs1 * temp;
                    let temp2 = temp1 * temp;
                    let temp3 = temp2 * temp;
                    let temp4 = temp1 / ds1;
                    fc = fc + temp4;
                    erretm = erretm + F::abs(temp4);
                    df = df + temp2;
                    ddf = ddf + temp3;
                    if !hit2 {
                        let temp = one / (ds2 - tau);
                        let temp1 = zs2 * temp;
                        let temp2 = temp1 * temp;
                        let temp3 = temp2 * temp;
                        let temp4 = temp1 / ds2;
                        fc = fc + temp4;
                        erretm = erretm + F::abs(temp4);
                        df = df + temp2;
                        ddf = ddf + temp3;
                    }
                }
            }
            if hit0 || hit1 || hit2 {
                done = true;
            } else {
                f = finit + tau * fc;
                erretm = eight * (F::abs(finit) + F::abs(tau) * erretm) + F::abs(tau) * df;
                if (F::abs(f) <= four * eps * erretm) || ((ubd - lbd) <= four * eps * F::abs(tau)) {
                    done = true;
                } else {
                    if f <= zero {
                        lbd = tau;
                    } else {
                        ubd = tau;
                    }
                    it += 1u32;
                }
            }
        }
        if !done {
            info[slot] = 1u32;
        }
    }
    if scale {
        tau = tau * sclinv;
    }
    tau
}

/// `DLAED4` for the last root `I = N` (`N >= 3`): column `base` of `delta`
/// is `DELTA`; returns `DLAM`, sets `info[slot]` on non-convergence.
#[cube]
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn dlaed4_last<F: Float>(
    d: &Array<F>,
    z: &Array<F>,
    delta: &mut Array<F>,
    info: &mut Array<u32>,
    slot: usize,
    n: u32,
    base: u32,
    rho: F,
    eps: F,
) -> F {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let two = F::from_int(2);
    let three = F::from_int(3);
    let four = F::from_int(4);
    let eight = F::from_int(8);
    let ten = F::from_int(10);
    let rhoinv = one / rho;
    let i = n;
    // The last root.
    let ii = n - 1;
    let mut niter: u32 = 1u32;
    let midpt = rho / two;
    for j in 1..n + 1 {
        delta[(base + j - 1) as usize] = (d[(j - 1) as usize] - d[(i - 1) as usize]) - midpt;
    }
    let mut psi = zero;
    for j in 1..n - 1 {
        psi = psi + z[(j - 1) as usize] * z[(j - 1) as usize] / delta[(base + j - 1) as usize];
    }
    let mut c = rhoinv + psi;
    let zn1 = z[(n - 2) as usize];
    let zn = z[(n - 1) as usize];
    let mut w = c + zn1 * zn1 / delta[(base + ii - 1) as usize] + zn * zn / delta[(base + n - 1) as usize];
    let mut tau = zero;
    let mut dltlb = zero;
    let mut dltub = midpt;
    if w <= zero {
        let temp = zn1 * zn1 / (d[(n - 1) as usize] - d[(n - 2) as usize] + rho) + zn * zn / rho;
        if c <= temp {
            tau = rho;
        } else {
            let del = d[(n - 1) as usize] - d[(n - 2) as usize];
            let a = -c * del + zn1 * zn1 + zn * zn;
            let b = zn * zn * del;
            if a < zero {
                tau = two * b / (F::sqrt(a * a + four * b * c) - a);
            } else {
                tau = (a + F::sqrt(a * a + four * b * c)) / (two * c);
            }
        }
        dltlb = midpt;
        dltub = rho;
    } else {
        let del = d[(n - 1) as usize] - d[(n - 2) as usize];
        let a = -c * del + zn1 * zn1 + zn * zn;
        let b = zn * zn * del;
        if a < zero {
            tau = two * b / (F::sqrt(a * a + four * b * c) - a);
        } else {
            tau = (a + F::sqrt(a * a + four * b * c)) / (two * c);
        }
    }
    for j in 1..n + 1 {
        delta[(base + j - 1) as usize] = (d[(j - 1) as usize] - d[(i - 1) as usize]) - tau;
    }
    let mut dpsi = zero;
    psi = zero;
    let mut erretm = zero;
    for j in 1..ii + 1 {
        let temp = z[(j - 1) as usize] / delta[(base + j - 1) as usize];
        psi = psi + z[(j - 1) as usize] * temp;
        dpsi = dpsi + temp * temp;
        erretm = erretm + psi;
    }
    erretm = F::abs(erretm);
    let temp = zn / delta[(base + n - 1) as usize];
    let mut phi = zn * temp;
    let mut dphi = temp * temp;
    erretm = eight * (-phi - psi) + erretm - phi + rhoinv + F::abs(tau) * (dpsi + dphi);
    w = rhoinv + phi + psi;
    let mut done = F::abs(w) <= eps * erretm;
    if !done {
        if w <= zero {
            dltlb = fmax_c::<F>(dltlb, tau);
        } else {
            dltub = fmin_c::<F>(dltub, tau);
        }
        niter += 1u32;
        let dn1 = delta[(base + n - 2) as usize];
        let dn = delta[(base + n - 1) as usize];
        c = w - dn1 * dpsi - dn * dphi;
        let a = (dn1 + dn) * w - dn1 * dn * (dpsi + dphi);
        let b = dn1 * dn * w;
        if c < zero {
            c = F::abs(c);
        }
        let mut eta = zero;
        if c == zero {
            eta = -w / (dpsi + dphi);
        } else if a >= zero {
            eta = (a + F::sqrt(F::abs(a * a - four * b * c))) / (two * c);
        } else {
            eta = two * b / (a - F::sqrt(F::abs(a * a - four * b * c)));
        }
        if w * eta > zero {
            eta = -w / (dpsi + dphi);
        }
        let temp = tau + eta;
        if temp > dltub || temp < dltlb {
            if w < zero {
                eta = (dltub - tau) / two;
            } else {
                eta = (dltlb - tau) / two;
            }
        }
        for j in 1..n + 1 {
            delta[(base + j - 1) as usize] = delta[(base + j - 1) as usize] - eta;
        }
        tau = tau + eta;
        dpsi = zero;
        psi = zero;
        erretm = zero;
        for j in 1..ii + 1 {
            let temp = z[(j - 1) as usize] / delta[(base + j - 1) as usize];
            psi = psi + z[(j - 1) as usize] * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = F::abs(erretm);
        let temp = zn / delta[(base + n - 1) as usize];
        phi = zn * temp;
        dphi = temp * temp;
        erretm = eight * (-phi - psi) + erretm - phi + rhoinv + F::abs(tau) * (dpsi + dphi);
        w = rhoinv + phi + psi;

        let mut it: u32 = niter + 1u32;
        while it <= 30u32 && !done {
            if F::abs(w) <= eps * erretm {
                done = true;
            } else {
                if w <= zero {
                    dltlb = fmax_c::<F>(dltlb, tau);
                } else {
                    dltub = fmin_c::<F>(dltub, tau);
                }
                let dn1 = delta[(base + n - 2) as usize];
                let dn = delta[(base + n - 1) as usize];
                let c = w - dn1 * dpsi - dn * dphi;
                let a = (dn1 + dn) * w - dn1 * dn * (dpsi + dphi);
                let b = dn1 * dn * w;
                let mut eta = two * b / (a - F::sqrt(F::abs(a * a - four * b * c)));
                if a >= zero {
                    eta = (a + F::sqrt(F::abs(a * a - four * b * c))) / (two * c);
                }
                if w * eta > zero {
                    eta = -w / (dpsi + dphi);
                }
                let temp = tau + eta;
                if temp > dltub || temp < dltlb {
                    if w < zero {
                        eta = (dltub - tau) / two;
                    } else {
                        eta = (dltlb - tau) / two;
                    }
                }
                for j in 1..n + 1 {
                    delta[(base + j - 1) as usize] = delta[(base + j - 1) as usize] - eta;
                }
                tau = tau + eta;
                dpsi = zero;
                psi = zero;
                erretm = zero;
                for j in 1..ii + 1 {
                    let temp = z[(j - 1) as usize] / delta[(base + j - 1) as usize];
                    psi = psi + z[(j - 1) as usize] * temp;
                    dpsi = dpsi + temp * temp;
                    erretm = erretm + psi;
                }
                erretm = F::abs(erretm);
                let temp = zn / delta[(base + n - 1) as usize];
                phi = zn * temp;
                dphi = temp * temp;
                erretm = eight * (-phi - psi) + erretm - phi + rhoinv + F::abs(tau) * (dpsi + dphi);
                w = rhoinv + phi + psi;
                it += 1u32;
            }
        }
        if !done {
            info[slot] = 1u32;
        }
    }
    d[(i - 1) as usize] + tau
}

/// `DLAED4` for an interior root `0 < I < N` (`N >= 3`); as [`dlaed4_last`].
#[cube]
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn dlaed4_mid<F: Float>(
    d: &Array<F>,
    z: &Array<F>,
    delta: &mut Array<F>,
    info: &mut Array<u32>,
    slot: usize,
    n: u32,
    i: u32,
    base: u32,
    rho: F,
    eps: F,
    small1: F,
) -> F {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let two = F::from_int(2);
    let three = F::from_int(3);
    let four = F::from_int(4);
    let eight = F::from_int(8);
    let ten = F::from_int(10);
    let rhoinv = one / rho;
    // 0 < I < N.
    let mut niter: u32 = 1u32;
    let ip1 = i + 1;
    let di = d[(i - 1) as usize];
    let dip1 = d[(ip1 - 1) as usize];
    let zi = z[(i - 1) as usize];
    let zip1 = z[(ip1 - 1) as usize];
    let del = dip1 - di;
    let midpt = del / two;
    for j in 1..n + 1 {
        delta[(base + j - 1) as usize] = (d[(j - 1) as usize] - di) - midpt;
    }
    let mut psi = zero;
    for j in 1..i {
        psi = psi + z[(j - 1) as usize] * z[(j - 1) as usize] / delta[(base + j - 1) as usize];
    }
    let mut phi = zero;
    // Descending `while jr1 >= i + 2 { …; jr1 -= 1 }` rewritten as an
    // ascending `for` over the identical visit order (n, n-1, …, i+2):
    // cubecl-cpu 0.10 cannot lower a `while` loop with a decremented
    // counter ("operation with block successors must terminate its parent
    // block", see `tests/cubecl_cpu_limits.rs`). `t` ranges ascending over
    // `i+2..=n`; `jr1 = n + (i+2) - t` walks `n, n-1, …, i+2`, so every term
    // is added to `phi` in exactly the original order — this cannot change
    // a floating-point value.
    for t in i + 2..n + 1 {
        let jr1 = n + (i + 2) - t;
        phi = phi + z[(jr1 - 1) as usize] * z[(jr1 - 1) as usize] / delta[(base + jr1 - 1) as usize];
    }
    let c = rhoinv + psi + phi;
    let mut w = c + zi * zi / delta[(base + i - 1) as usize] + zip1 * zip1 / delta[(base + ip1 - 1) as usize];
    let orgati = w > zero;
    let mut tau = zero;
    let mut dltlb = zero;
    let mut dltub = midpt;
    if orgati {
        let a = c * del + zi * zi + zip1 * zip1;
        let b = zi * zi * del;
        if a > zero {
            tau = two * b / (a + F::sqrt(F::abs(a * a - four * b * c)));
        } else {
            tau = (a - F::sqrt(F::abs(a * a - four * b * c))) / (two * c);
        }
    } else {
        let a = c * del - zi * zi - zip1 * zip1;
        let b = zip1 * zip1 * del;
        if a < zero {
            tau = two * b / (a - F::sqrt(F::abs(a * a + four * b * c)));
        } else {
            tau = -(a + F::sqrt(F::abs(a * a + four * b * c))) / (two * c);
        }
        dltlb = -midpt;
        dltub = zero;
    }
    if orgati {
        for j in 1..n + 1 {
            delta[(base + j - 1) as usize] = (d[(j - 1) as usize] - di) - tau;
        }
    } else {
        for j in 1..n + 1 {
            delta[(base + j - 1) as usize] = (d[(j - 1) as usize] - dip1) - tau;
        }
    }
    let mut ii = i + 1;
    if orgati {
        ii = i;
    }
    // The reference's II-1 and II+1: loop BOUNDS (the psi and phi sums below
    // run 1..=iim1 and iip1..=n, and are empty exactly when ii == 1 or
    // ii == n), so they must not be clamped.
    let iim1 = ii - 1;
    let iip1 = ii + 1;
    // The same indices as ARRAY SUBSCRIPTS, used only inside the `swtch3`
    // arms. `swtch3` is forced false for ii == 1 and ii == n, so these arms
    // never run there and the clamp changes no value; it keeps a speculated
    // untaken arm from reading out of bounds under `launch_unchecked`
    // (ii - 1 underflows at ii == 1, ii + 1 is past `d` at ii == n).
    let iim1x = select(ii > 1, iim1, 1u32);
    let iip1x = select(ii < n, iip1, n);

    let mut dpsi = zero;
    psi = zero;
    let mut erretm = zero;
    for j in 1..iim1 + 1 {
        let temp = z[(j - 1) as usize] / delta[(base + j - 1) as usize];
        psi = psi + z[(j - 1) as usize] * temp;
        dpsi = dpsi + temp * temp;
        erretm = erretm + psi;
    }
    erretm = F::abs(erretm);
    let mut dphi = zero;
    phi = zero;
    // Same rewrite as `jr1` above: ascending `for` over `iip1..=n`, visiting
    // `jr2 = n, n-1, …, iip1` in the original order, so `phi`/`dphi`/`erretm`
    // accumulate identically.
    for t in iip1..n + 1 {
        let jr2 = n + iip1 - t;
        let temp = z[(jr2 - 1) as usize] / delta[(base + jr2 - 1) as usize];
        phi = phi + z[(jr2 - 1) as usize] * temp;
        dphi = dphi + temp * temp;
        erretm = erretm + phi;
    }
    w = rhoinv + phi + psi;

    let mut swtch3: bool = false;
    if orgati {
        if w < zero {
            swtch3 = true;
        }
    } else if w > zero {
        swtch3 = true;
    }
    if ii == 1 || ii == n {
        swtch3 = false;
    }

    let zii = z[(ii - 1) as usize];
    let temp = zii / delta[(base + ii - 1) as usize];
    let mut dw = dpsi + dphi + temp * temp;
    let temp = zii * temp;
    w = w + temp;
    erretm = eight * (phi - psi) + erretm + two * rhoinv + three * F::abs(temp) + F::abs(tau) * dw;

    let mut done = F::abs(w) <= eps * erretm;
    let mut failed: bool = false;
    if !done {
        if w <= zero {
            dltlb = fmax_c::<F>(dltlb, tau);
        } else {
            dltub = fmin_c::<F>(dltub, tau);
        }

        niter += 1u32;
        let mut eta = zero;
        let dli = delta[(base + i - 1) as usize];
        let dlip1 = delta[(base + ip1 - 1) as usize];
        if !swtch3 {
            let mut c = zero;
            if orgati {
                let t = zi / dli;
                c = w - dlip1 * dw - (di - dip1) * (t * t);
            } else {
                let t = zip1 / dlip1;
                c = w - dli * dw - (dip1 - di) * (t * t);
            }
            let mut a = (dli + dlip1) * w - dli * dlip1 * dw;
            let b = dli * dlip1 * w;
            if c == zero {
                if a == zero {
                    if orgati {
                        a = zi * zi + dlip1 * dlip1 * (dpsi + dphi);
                    } else {
                        a = zip1 * zip1 + dli * dli * (dpsi + dphi);
                    }
                }
                eta = b / a;
            } else if a <= zero {
                eta = (a - F::sqrt(F::abs(a * a - four * b * c))) / (two * c);
            } else {
                eta = two * b / (a + F::sqrt(F::abs(a * a - four * b * c)));
            }
        } else {
            let temp = rhoinv + psi + phi;
            let dlim1 = delta[(base + iim1x - 1) as usize];
            let dlii = delta[(base + ii - 1) as usize];
            let dlip = delta[(base + iip1x - 1) as usize];
            let mut c = zero;
            let mut z1 = zero;
            let mut z3 = zero;
            if orgati {
                let mut temp1 = z[(iim1x - 1) as usize] / dlim1;
                temp1 = temp1 * temp1;
                c = temp - dlip * (dpsi + dphi) - (d[(iim1x - 1) as usize] - d[(iip1x - 1) as usize]) * temp1;
                z1 = z[(iim1x - 1) as usize] * z[(iim1x - 1) as usize];
                z3 = dlip * dlip * ((dpsi - temp1) + dphi);
            } else {
                let mut temp1 = z[(iip1x - 1) as usize] / dlip;
                temp1 = temp1 * temp1;
                c = temp - dlim1 * (dpsi + dphi) - (d[(iip1x - 1) as usize] - d[(iim1x - 1) as usize]) * temp1;
                z1 = dlim1 * dlim1 * (dpsi + (dphi - temp1));
                z3 = z[(iip1x - 1) as usize] * z[(iip1x - 1) as usize];
            }
            let z2 = zii * zii;
            eta = dlaed6_c::<F>(niter, orgati, c, dlim1, dlii, dlip, z1, z2, z3, w, eps, small1, info, slot);
            if info[slot] != 0u32 {
                failed = true;
            }
        }

        if !failed {
            if w * eta >= zero {
                eta = -w / dw;
            }
            let temp = tau + eta;
            if temp > dltub || temp < dltlb {
                if w < zero {
                    eta = (dltub - tau) / two;
                } else {
                    eta = (dltlb - tau) / two;
                }
            }

            let mut prew = w;

            for j in 1..n + 1 {
                delta[(base + j - 1) as usize] = delta[(base + j - 1) as usize] - eta;
            }

            dpsi = zero;
            psi = zero;
            erretm = zero;
            for j in 1..iim1 + 1 {
                let temp = z[(j - 1) as usize] / delta[(base + j - 1) as usize];
                psi = psi + z[(j - 1) as usize] * temp;
                dpsi = dpsi + temp * temp;
                erretm = erretm + psi;
            }
            erretm = F::abs(erretm);
            dphi = zero;
            phi = zero;
            // Same rewrite as `jr1`/`jr2`: ascending `for` visiting
            // `jr3 = n, n-1, …, iip1` in the original order.
            for t in iip1..n + 1 {
                let jr3 = n + iip1 - t;
                let temp = z[(jr3 - 1) as usize] / delta[(base + jr3 - 1) as usize];
                phi = phi + z[(jr3 - 1) as usize] * temp;
                dphi = dphi + temp * temp;
                erretm = erretm + phi;
            }
            let temp = zii / delta[(base + ii - 1) as usize];
            dw = dpsi + dphi + temp * temp;
            let temp = zii * temp;
            w = rhoinv + phi + psi + temp;
            erretm = eight * (phi - psi) + erretm + two * rhoinv + three * F::abs(temp) + F::abs(tau + eta) * dw;

            let mut swtch: bool = false;
            if orgati {
                if -w > F::abs(prew) / ten {
                    swtch = true;
                }
            } else if w > F::abs(prew) / ten {
                swtch = true;
            }

            tau = tau + eta;

            let mut it: u32 = niter + 1u32;
            while it <= 30u32 && !done && !failed {
                if F::abs(w) <= eps * erretm {
                    done = true;
                } else {
                    if w <= zero {
                        dltlb = fmax_c::<F>(dltlb, tau);
                    } else {
                        dltub = fmin_c::<F>(dltub, tau);
                    }
                    let mut eta = zero;
                    let dli = delta[(base + i - 1) as usize];
                    let dlip1 = delta[(base + ip1 - 1) as usize];
                    if !swtch3 {
                        let mut c = zero;
                        if !swtch {
                            if orgati {
                                let t = zi / dli;
                                c = w - dlip1 * dw - (di - dip1) * (t * t);
                            } else {
                                let t = zip1 / dlip1;
                                c = w - dli * dw - (dip1 - di) * (t * t);
                            }
                        } else {
                            let temp = zii / delta[(base + ii - 1) as usize];
                            if orgati {
                                dpsi = dpsi + temp * temp;
                            } else {
                                dphi = dphi + temp * temp;
                            }
                            c = w - dli * dpsi - dlip1 * dphi;
                        }
                        let mut a = (dli + dlip1) * w - dli * dlip1 * dw;
                        let b = dli * dlip1 * w;
                        if c == zero {
                            if a == zero {
                                if !swtch {
                                    if orgati {
                                        a = zi * zi + dlip1 * dlip1 * (dpsi + dphi);
                                    } else {
                                        a = zip1 * zip1 + dli * dli * (dpsi + dphi);
                                    }
                                } else {
                                    a = dli * dli * dpsi + dlip1 * dlip1 * dphi;
                                }
                            }
                            eta = b / a;
                        } else if a <= zero {
                            eta = (a - F::sqrt(F::abs(a * a - four * b * c))) / (two * c);
                        } else {
                            eta = two * b / (a + F::sqrt(F::abs(a * a - four * b * c)));
                        }
                    } else {
                        let temp = rhoinv + psi + phi;
                        let dlim1 = delta[(base + iim1x - 1) as usize];
                        let dlii = delta[(base + ii - 1) as usize];
                        let dlip = delta[(base + iip1x - 1) as usize];
                        let mut c = zero;
                        let mut z1 = zero;
                        let mut z3 = zero;
                        if swtch {
                            c = temp - dlim1 * dpsi - dlip * dphi;
                            z1 = dlim1 * dlim1 * dpsi;
                            z3 = dlip * dlip * dphi;
                        } else if orgati {
                            let mut temp1 = z[(iim1x - 1) as usize] / dlim1;
                            temp1 = temp1 * temp1;
                            c = temp
                                - dlip * (dpsi + dphi)
                                - (d[(iim1x - 1) as usize] - d[(iip1x - 1) as usize]) * temp1;
                            z1 = z[(iim1x - 1) as usize] * z[(iim1x - 1) as usize];
                            z3 = dlip * dlip * ((dpsi - temp1) + dphi);
                        } else {
                            let mut temp1 = z[(iip1x - 1) as usize] / dlip;
                            temp1 = temp1 * temp1;
                            c = temp
                                - dlim1 * (dpsi + dphi)
                                - (d[(iip1x - 1) as usize] - d[(iim1x - 1) as usize]) * temp1;
                            z1 = dlim1 * dlim1 * (dpsi + (dphi - temp1));
                            z3 = z[(iip1x - 1) as usize] * z[(iip1x - 1) as usize];
                        }
                        let z2 = zii * zii;
                        eta = dlaed6_c::<F>(it, orgati, c, dlim1, dlii, dlip, z1, z2, z3, w, eps, small1, info, slot);
                        if info[slot] != 0u32 {
                            failed = true;
                        }
                    }

                    if !failed {
                        if w * eta >= zero {
                            eta = -w / dw;
                        }
                        let temp = tau + eta;
                        if temp > dltub || temp < dltlb {
                            if w < zero {
                                eta = (dltub - tau) / two;
                            } else {
                                eta = (dltlb - tau) / two;
                            }
                        }

                        for j in 1..n + 1 {
                            delta[(base + j - 1) as usize] = delta[(base + j - 1) as usize] - eta;
                        }

                        tau = tau + eta;
                        prew = w;

                        dpsi = zero;
                        psi = zero;
                        erretm = zero;
                        for j in 1..iim1 + 1 {
                            let temp = z[(j - 1) as usize] / delta[(base + j - 1) as usize];
                            psi = psi + z[(j - 1) as usize] * temp;
                            dpsi = dpsi + temp * temp;
                            erretm = erretm + psi;
                        }
                        erretm = F::abs(erretm);
                        dphi = zero;
                        phi = zero;
                        // Same rewrite as `jr1`/`jr2`/`jr3`: ascending `for`
                        // visiting `jr4 = n, n-1, …, iip1` in the original
                        // order.
                        for t in iip1..n + 1 {
                            let jr4 = n + iip1 - t;
                            let temp = z[(jr4 - 1) as usize] / delta[(base + jr4 - 1) as usize];
                            phi = phi + z[(jr4 - 1) as usize] * temp;
                            dphi = dphi + temp * temp;
                            erretm = erretm + phi;
                        }
                        let temp = zii / delta[(base + ii - 1) as usize];
                        dw = dpsi + dphi + temp * temp;
                        let temp = zii * temp;
                        w = rhoinv + phi + psi + temp;
                        erretm = eight * (phi - psi) + erretm + two * rhoinv + three * F::abs(temp) + F::abs(tau) * dw;
                        if w * prew > zero && F::abs(w) > F::abs(prew) / ten {
                            swtch = !swtch;
                        }
                        it += 1u32;
                    }
                }
            }
            if !done && !failed {
                info[slot] = 1u32;
            }
        }
    }
    let mut out = zero;
    if !failed {
        out = dip1 + tau;
        if orgati {
            out = di + tau;
        }
    }
    out
}

/// `DLAED4( K, J, D, Z, DELTA(1,J), RHO, DLAM, INFO )` for every root `J`
/// (`K >= 3`): unit `J − 1` writes column `J − 1` of `delta`, `dlam[J − 1]`
/// and `info[J − 1]` (0 = converged).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn secular_roots_kernel<F: Float + CubeElement>(
    d: &Array<F>,
    z: &Array<F>,
    delta: &mut Array<F>,
    dlam: &mut Array<F>,
    info: &mut Array<u32>,
    n: u32,
    rho: F,
    eps: F,
    small1: F,
) {
    if ABSOLUTE_POS < n as usize {
        let slot = ABSOLUTE_POS;
        let i = ABSOLUTE_POS as u32 + 1;
        info[slot] = 0u32;
        if i == n {
            dlam[slot] = dlaed4_last::<F>(d, z, delta, info, slot, n, (i - 1) * n, rho, eps);
        } else {
            dlam[slot] = dlaed4_mid::<F>(d, z, delta, info, slot, n, i, (i - 1) * n, rho, eps, small1);
        }
    }
}

/// `DLAED3` loop 60 and the `W(I) = SIGN(SQRT(-W(I)), S(I))` step: unit `I`
/// starts from `DELTA(I, I)` and multiplies `DELTA(I,J)/(DLAMBDA(I) −
/// DLAMBDA(J))` over `J ≠ I` ascending. `neg[I]` is the sign bit of `S(I)`.
#[cube(launch, launch_unchecked)]
pub fn secular_w_kernel<F: Float + CubeElement>(
    delta: &Array<F>,
    d: &Array<F>,
    neg: &Array<u32>,
    w: &mut Array<F>,
    n: u32,
) {
    if ABSOLUTE_POS < n as usize {
        let i = ABSOLUTE_POS as u32;
        let mut wi = delta[(i * n + i) as usize];
        let li = d[i as usize];
        for j in 0..i {
            wi = wi * (delta[(j * n + i) as usize] / (li - d[j as usize]));
        }
        for j in i + 1..n {
            wi = wi * (delta[(j * n + i) as usize] / (li - d[j as usize]));
        }
        let r = F::abs(F::sqrt(-wi));
        w[i as usize] = select(neg[i as usize] != 0u32, -r, r);
    }
}

/// `DLAED3` loop 110 for column `J`: `S(I) = W(I)/DELTA(I,J)`, `TEMP =
/// DNRM2(K, S)` (3.12.1 `dnrm2.f90`, Blue's scaled sums), `U(I,J) =
/// S(INDX(I))/TEMP`. `delta` is overwritten with `S`; `u` is `K × K`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn secular_u_kernel<F: Float + CubeElement>(
    w: &Array<F>,
    indx: &Array<u32>,
    delta: &mut Array<F>,
    u: &mut Array<F>,
    n: u32,
    tsml: F,
    tbig: F,
    ssml: F,
    sbig: F,
    maxn: F,
) {
    if ABSOLUTE_POS < n as usize {
        let zero = F::from_int(0);
        let one = F::from_int(1);
        let j = ABSOLUTE_POS as u32;
        let base = j * n;
        for i in 0..n {
            delta[(base + i) as usize] = w[i as usize] / delta[(base + i) as usize];
        }
        let mut notbig = true;
        let mut asml = zero;
        let mut amed = zero;
        let mut abig = zero;
        for i in 0..n {
            let ax = F::abs(delta[(base + i) as usize]);
            if ax > tbig {
                let t = ax * sbig;
                abig = abig + t * t;
                notbig = false;
            } else if ax < tsml {
                if notbig {
                    let t = ax * ssml;
                    asml = asml + t * t;
                }
            } else {
                amed = amed + ax * ax;
            }
        }
        let mut scl = one;
        let mut sumsq = amed;
        let amed_any = (amed > zero) || (amed > maxn) || (amed != amed);
        if abig > zero {
            if amed_any {
                abig = abig + (amed * sbig) * sbig;
            }
            scl = one / sbig;
            sumsq = abig;
        } else if asml > zero {
            if amed_any {
                let amed_r = F::sqrt(amed);
                let asml_r = F::sqrt(asml) / ssml;
                let mut ymin = asml_r;
                let mut ymax = amed_r;
                if asml_r > amed_r {
                    ymin = amed_r;
                    ymax = asml_r;
                }
                let r = ymin / ymax;
                sumsq = (ymax * ymax) * (one + r * r);
            } else {
                scl = one / ssml;
                sumsq = asml;
            }
        }
        let temp = scl * F::sqrt(sumsq);
        for i in 0..n {
            u[(base + i) as usize] = delta[(base + indx[i as usize]) as usize] / temp;
        }
    }
}
