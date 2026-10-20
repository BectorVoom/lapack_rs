//! Secular equation solvers.
//!
//! Transcribed from Reference-LAPACK v3.12.1 `SRC/dlaed4.f`, `SRC/dlaed5.f`
//! and `SRC/dlaed6.f`. Each call is independent per root, so `dlaed3` runs
//! them in parallel across roots without changing a bit.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments, clippy::neg_multiply)]

use crate::aux::{BASE, EPS, SAFMIN, fmax, fmin, powidf2};
use crate::ptr::D;

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;
const TWO: f64 = 2.0;
const THREE: f64 = 3.0;
const FOUR: f64 = 4.0;
const EIGHT: f64 = 8.0;
const TEN: f64 = 10.0;

/// `DLAED5`. `d`, `z`, `delta` have 2 entries. Returns `DLAM`.
pub(crate) unsafe fn dlaed5(i: usize, d: D, z: D, delta: D, rho: f64) -> f64 {
    let d1 = d.at(0);
    let d2 = d.at(1);
    let z1 = z.at(0);
    let z2 = z.at(1);
    let del = d2 - d1;
    let dlam;
    if i == 1 {
        let w = ONE + TWO * rho * (z2 * z2 - z1 * z1) / del;
        if w > ZERO {
            let b = del + rho * (z1 * z1 + z2 * z2);
            let c = rho * z1 * z1 * del;
            let tau = TWO * c / (b + (b * b - FOUR * c).abs().sqrt());
            dlam = d1 + tau;
            delta.set(0, -z1 / tau);
            delta.set(1, z2 / (del - tau));
        } else {
            let b = -del + rho * (z1 * z1 + z2 * z2);
            let c = rho * z2 * z2 * del;
            let tau = if b > ZERO {
                -TWO * c / (b + (b * b + FOUR * c).sqrt())
            } else {
                (b - (b * b + FOUR * c).sqrt()) / TWO
            };
            dlam = d2 + tau;
            delta.set(0, -z1 / (del + tau));
            delta.set(1, -z2 / tau);
        }
        let temp = (delta.at(0) * delta.at(0) + delta.at(1) * delta.at(1)).sqrt();
        delta.set(0, delta.at(0) / temp);
        delta.set(1, delta.at(1) / temp);
    } else {
        let b = -del + rho * (z1 * z1 + z2 * z2);
        let c = rho * z2 * z2 * del;
        let tau =
            if b > ZERO { (b + (b * b + FOUR * c).sqrt()) / TWO } else { TWO * c / (-b + (b * b + FOUR * c).sqrt()) };
        dlam = d2 + tau;
        delta.set(0, -z1 / (del + tau));
        delta.set(1, -z2 / tau);
        let temp = (delta.at(0) * delta.at(0) + delta.at(1) * delta.at(1)).sqrt();
        delta.set(0, delta.at(0) / temp);
        delta.set(1, delta.at(1) / temp);
    }
    dlam
}

/// `SMALL1 = BASE**( INT( LOG( SAFMIN ) / LOG( BASE ) / THREE ) )`, evaluated
/// at run time as the Fortran does (glibc `log`, libgcc `__powidf2`).
pub(crate) fn dlaed6_small1() -> f64 {
    let safmin = std::hint::black_box(SAFMIN);
    let base = std::hint::black_box(BASE);
    let e = (safmin.ln() / base.ln() / THREE) as i32;
    powidf2(base, e)
}

/// `DLAED6`. `d`, `z` have 3 entries. Returns `(tau, info)`.
pub(crate) fn dlaed6(kniter: usize, orgati: bool, rho: f64, d: [f64; 3], z: [f64; 3], finit: f64) -> (f64, usize) {
    const MAXIT: usize = 40;
    let mut info = 0;
    let (mut lbd, mut ubd) = if orgati { (d[1], d[2]) } else { (d[0], d[1]) };
    if finit < ZERO {
        lbd = ZERO;
    } else {
        ubd = ZERO;
    }
    let niter = 1;
    let mut tau = ZERO;
    if kniter == 2 {
        let (mut a, mut b, mut c);
        if orgati {
            let temp = (d[2] - d[1]) / TWO;
            c = rho + z[0] / ((d[0] - d[1]) - temp);
            a = c * (d[1] + d[2]) + z[1] + z[2];
            b = c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1];
        } else {
            let temp = (d[0] - d[1]) / TWO;
            c = rho + z[2] / ((d[2] - d[1]) - temp);
            a = c * (d[0] + d[1]) + z[0] + z[1];
            b = c * d[0] * d[1] + z[0] * d[1] + z[1] * d[0];
        }
        let temp = fmax(fmax(a.abs(), b.abs()), c.abs());
        a /= temp;
        b /= temp;
        c /= temp;
        if c == ZERO {
            tau = b / a;
        } else if a <= ZERO {
            tau = (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c);
        } else {
            tau = TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt());
        }
        if tau < lbd || tau > ubd {
            tau = (lbd + ubd) / TWO;
        }
        if d[0] == tau || d[1] == tau || d[2] == tau {
            tau = ZERO;
        } else {
            let temp = finit
                + tau * z[0] / (d[0] * (d[0] - tau))
                + tau * z[1] / (d[1] * (d[1] - tau))
                + tau * z[2] / (d[2] * (d[2] - tau));
            if temp <= ZERO {
                lbd = tau;
            } else {
                ubd = tau;
            }
            if finit.abs() <= temp.abs() {
                tau = ZERO;
            }
        }
    }

    let eps = EPS;
    let small1 = dlaed6_small1();
    let sminv1 = ONE / small1;
    let small2 = small1 * small1;
    let sminv2 = sminv1 * sminv1;

    let temp = if orgati {
        fmin((d[1] - tau).abs(), (d[2] - tau).abs())
    } else {
        fmin((d[0] - tau).abs(), (d[1] - tau).abs())
    };
    let mut scale = false;
    let mut sclinv = ONE;
    let mut dscale = [0.0f64; 3];
    let mut zscale = [0.0f64; 3];
    if temp <= small1 {
        scale = true;
        let sclfac;
        if temp <= small2 {
            sclfac = sminv2;
            sclinv = small2;
        } else {
            sclfac = sminv1;
            sclinv = small1;
        }
        for i in 0..3 {
            dscale[i] = d[i] * sclfac;
            zscale[i] = z[i] * sclfac;
        }
        tau *= sclfac;
        lbd *= sclfac;
        ubd *= sclfac;
    } else {
        dscale = d;
        zscale = z;
    }

    let mut fc = ZERO;
    let mut df = ZERO;
    let mut ddf = ZERO;
    for i in 0..3 {
        let temp = ONE / (dscale[i] - tau);
        let temp1 = zscale[i] * temp;
        let temp2 = temp1 * temp;
        let temp3 = temp2 * temp;
        fc += temp1 / dscale[i];
        df += temp2;
        ddf += temp3;
    }
    let mut f = finit + tau * fc;

    'done: {
        if f.abs() <= ZERO {
            break 'done;
        }
        if f <= ZERO {
            lbd = tau;
        } else {
            ubd = tau;
        }
        let iter = niter + 1;
        for _niter in iter..=MAXIT {
            let (temp1, temp2) =
                if orgati { (dscale[1] - tau, dscale[2] - tau) } else { (dscale[0] - tau, dscale[1] - tau) };
            let mut a = (temp1 + temp2) * f - temp1 * temp2 * df;
            let mut b = temp1 * temp2 * f;
            let mut c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
            let temp = fmax(fmax(a.abs(), b.abs()), c.abs());
            a /= temp;
            b /= temp;
            c /= temp;
            let mut eta = if c == ZERO {
                b / a
            } else if a <= ZERO {
                (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c)
            } else {
                TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt())
            };
            if f * eta >= ZERO {
                eta = -f / df;
            }
            tau += eta;
            if tau < lbd || tau > ubd {
                tau = (lbd + ubd) / TWO;
            }
            fc = ZERO;
            let mut erretm = ZERO;
            df = ZERO;
            ddf = ZERO;
            for i in 0..3 {
                if (dscale[i] - tau) != ZERO {
                    let temp = ONE / (dscale[i] - tau);
                    let temp1 = zscale[i] * temp;
                    let temp2 = temp1 * temp;
                    let temp3 = temp2 * temp;
                    let temp4 = temp1 / dscale[i];
                    fc += temp4;
                    erretm += temp4.abs();
                    df += temp2;
                    ddf += temp3;
                } else {
                    break 'done;
                }
            }
            f = finit + tau * fc;
            erretm = EIGHT * (finit.abs() + tau.abs() * erretm) + tau.abs() * df;
            if (f.abs() <= FOUR * eps * erretm) || ((ubd - lbd) <= FOUR * eps * tau.abs()) {
                break 'done;
            }
            if f <= ZERO {
                lbd = tau;
            } else {
                ubd = tau;
            }
        }
        info = 1;
    }
    if scale {
        tau *= sclinv;
    }
    (tau, info)
}

/// `DLAED4( N, I, D, Z, DELTA, RHO, DLAM, INFO )`. `i` is 1-based. Returns
/// `(dlam, info)`.
pub(crate) unsafe fn dlaed4(n: usize, i: usize, d: D, z: D, delta: D, rho: f64) -> (f64, usize) {
    const MAXIT: usize = 30;
    let dd = |j: usize| d.at(j - 1);
    let zz = |j: usize| z.at(j - 1);
    let dl = |j: usize| delta.at(j - 1);
    let sdl = |j: usize, v: f64| delta.set(j - 1, v);

    if n == 1 {
        let dlam = dd(1) + rho * zz(1) * zz(1);
        sdl(1, ONE);
        return (dlam, 0);
    }
    if n == 2 {
        return (dlaed5(i, d, z, delta, rho), 0);
    }

    let eps = EPS;
    let rhoinv = ONE / rho;
    let info;

    if i == n {
        // The last root.
        let ii = n - 1;
        let mut niter = 1;
        let midpt = rho / TWO;
        for j in 1..=n {
            sdl(j, (dd(j) - dd(i)) - midpt);
        }
        let mut psi = ZERO;
        for j in 1..=(n - 2) {
            psi += zz(j) * zz(j) / dl(j);
        }
        let mut c = rhoinv + psi;
        let mut w = c + zz(ii) * zz(ii) / dl(ii) + zz(n) * zz(n) / dl(n);
        let mut tau;
        let (mut dltlb, mut dltub);
        if w <= ZERO {
            let temp = zz(n - 1) * zz(n - 1) / (dd(n) - dd(n - 1) + rho) + zz(n) * zz(n) / rho;
            if c <= temp {
                tau = rho;
            } else {
                let del = dd(n) - dd(n - 1);
                let a = -c * del + zz(n - 1) * zz(n - 1) + zz(n) * zz(n);
                let b = zz(n) * zz(n) * del;
                if a < ZERO {
                    tau = TWO * b / ((a * a + FOUR * b * c).sqrt() - a);
                } else {
                    tau = (a + (a * a + FOUR * b * c).sqrt()) / (TWO * c);
                }
            }
            dltlb = midpt;
            dltub = rho;
        } else {
            let del = dd(n) - dd(n - 1);
            let a = -c * del + zz(n - 1) * zz(n - 1) + zz(n) * zz(n);
            let b = zz(n) * zz(n) * del;
            if a < ZERO {
                tau = TWO * b / ((a * a + FOUR * b * c).sqrt() - a);
            } else {
                tau = (a + (a * a + FOUR * b * c).sqrt()) / (TWO * c);
            }
            dltlb = ZERO;
            dltub = midpt;
        }
        for j in 1..=n {
            sdl(j, (dd(j) - dd(i)) - tau);
        }
        let mut dpsi = ZERO;
        psi = ZERO;
        let mut erretm = ZERO;
        for j in 1..=ii {
            let temp = zz(j) / dl(j);
            psi += zz(j) * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = erretm.abs();
        let temp = zz(n) / dl(n);
        let mut phi = zz(n) * temp;
        let mut dphi = temp * temp;
        erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv + tau.abs() * (dpsi + dphi);
        w = rhoinv + phi + psi;
        if w.abs() <= eps * erretm {
            return (dd(i) + tau, 0);
        }
        if w <= ZERO {
            dltlb = fmax(dltlb, tau);
        } else {
            dltub = fmin(dltub, tau);
        }
        niter += 1;
        c = w - dl(n - 1) * dpsi - dl(n) * dphi;
        let a = (dl(n - 1) + dl(n)) * w - dl(n - 1) * dl(n) * (dpsi + dphi);
        let b = dl(n - 1) * dl(n) * w;
        if c < ZERO {
            c = c.abs();
        }
        let mut eta;
        if c == ZERO {
            eta = -w / (dpsi + dphi);
        } else if a >= ZERO {
            eta = (a + (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c);
        } else {
            eta = TWO * b / (a - (a * a - FOUR * b * c).abs().sqrt());
        }
        if w * eta > ZERO {
            eta = -w / (dpsi + dphi);
        }
        let temp = tau + eta;
        if temp > dltub || temp < dltlb {
            if w < ZERO {
                eta = (dltub - tau) / TWO;
            } else {
                eta = (dltlb - tau) / TWO;
            }
        }
        for j in 1..=n {
            sdl(j, dl(j) - eta);
        }
        tau += eta;
        dpsi = ZERO;
        psi = ZERO;
        erretm = ZERO;
        for j in 1..=ii {
            let temp = zz(j) / dl(j);
            psi += zz(j) * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = erretm.abs();
        let temp = zz(n) / dl(n);
        phi = zz(n) * temp;
        dphi = temp * temp;
        erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv + tau.abs() * (dpsi + dphi);
        w = rhoinv + phi + psi;

        let iter = niter + 1;
        for _niter in iter..=MAXIT {
            if w.abs() <= eps * erretm {
                return (dd(i) + tau, 0);
            }
            if w <= ZERO {
                dltlb = fmax(dltlb, tau);
            } else {
                dltub = fmin(dltub, tau);
            }
            let c = w - dl(n - 1) * dpsi - dl(n) * dphi;
            let a = (dl(n - 1) + dl(n)) * w - dl(n - 1) * dl(n) * (dpsi + dphi);
            let b = dl(n - 1) * dl(n) * w;
            let mut eta = if a >= ZERO {
                (a + (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c)
            } else {
                TWO * b / (a - (a * a - FOUR * b * c).abs().sqrt())
            };
            if w * eta > ZERO {
                eta = -w / (dpsi + dphi);
            }
            let temp = tau + eta;
            if temp > dltub || temp < dltlb {
                if w < ZERO {
                    eta = (dltub - tau) / TWO;
                } else {
                    eta = (dltlb - tau) / TWO;
                }
            }
            for j in 1..=n {
                sdl(j, dl(j) - eta);
            }
            tau += eta;
            dpsi = ZERO;
            psi = ZERO;
            erretm = ZERO;
            for j in 1..=ii {
                let temp = zz(j) / dl(j);
                psi += zz(j) * temp;
                dpsi += temp * temp;
                erretm += psi;
            }
            erretm = erretm.abs();
            let temp = zz(n) / dl(n);
            phi = zz(n) * temp;
            dphi = temp * temp;
            erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv + tau.abs() * (dpsi + dphi);
            w = rhoinv + phi + psi;
        }
        info = 1;
        return (dd(i) + tau, info);
    }

    // 0 < I < N.
    let mut niter = 1;
    let ip1 = i + 1;
    let del = dd(ip1) - dd(i);
    let midpt = del / TWO;
    for j in 1..=n {
        sdl(j, (dd(j) - dd(i)) - midpt);
    }
    let mut psi = ZERO;
    for j in 1..i {
        psi += zz(j) * zz(j) / dl(j);
    }
    let mut phi = ZERO;
    for j in ((i + 2)..=n).rev() {
        phi += zz(j) * zz(j) / dl(j);
    }
    let c = rhoinv + psi + phi;
    let mut w = c + zz(i) * zz(i) / dl(i) + zz(ip1) * zz(ip1) / dl(ip1);
    let orgati;
    let mut tau;
    let (mut dltlb, mut dltub);
    if w > ZERO {
        orgati = true;
        let a = c * del + zz(i) * zz(i) + zz(ip1) * zz(ip1);
        let b = zz(i) * zz(i) * del;
        if a > ZERO {
            tau = TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt());
        } else {
            tau = (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c);
        }
        dltlb = ZERO;
        dltub = midpt;
    } else {
        orgati = false;
        let a = c * del - zz(i) * zz(i) - zz(ip1) * zz(ip1);
        let b = zz(ip1) * zz(ip1) * del;
        if a < ZERO {
            tau = TWO * b / (a - (a * a + FOUR * b * c).abs().sqrt());
        } else {
            tau = -(a + (a * a + FOUR * b * c).abs().sqrt()) / (TWO * c);
        }
        dltlb = -midpt;
        dltub = ZERO;
    }
    if orgati {
        for j in 1..=n {
            sdl(j, (dd(j) - dd(i)) - tau);
        }
    } else {
        for j in 1..=n {
            sdl(j, (dd(j) - dd(ip1)) - tau);
        }
    }
    let ii = if orgati { i } else { i + 1 };
    let iim1 = ii - 1;
    let iip1 = ii + 1;

    let mut dpsi = ZERO;
    psi = ZERO;
    let mut erretm = ZERO;
    for j in 1..=iim1 {
        let temp = zz(j) / dl(j);
        psi += zz(j) * temp;
        dpsi += temp * temp;
        erretm += psi;
    }
    erretm = erretm.abs();
    let mut dphi = ZERO;
    phi = ZERO;
    for j in (iip1..=n).rev() {
        let temp = zz(j) / dl(j);
        phi += zz(j) * temp;
        dphi += temp * temp;
        erretm += phi;
    }
    w = rhoinv + phi + psi;

    let mut swtch3 = false;
    if orgati {
        if w < ZERO {
            swtch3 = true;
        }
    } else if w > ZERO {
        swtch3 = true;
    }
    if ii == 1 || ii == n {
        swtch3 = false;
    }

    let temp = zz(ii) / dl(ii);
    let mut dw = dpsi + dphi + temp * temp;
    let temp = zz(ii) * temp;
    w += temp;
    erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv + THREE * temp.abs() + tau.abs() * dw;

    let dlam_of = |tau: f64| if orgati { dd(i) + tau } else { dd(ip1) + tau };

    if w.abs() <= eps * erretm {
        return (dlam_of(tau), 0);
    }
    if w <= ZERO {
        dltlb = fmax(dltlb, tau);
    } else {
        dltub = fmin(dltub, tau);
    }

    niter += 1;
    let mut eta;
    if !swtch3 {
        let c = if orgati {
            let t = zz(i) / dl(i);
            w - dl(ip1) * dw - (dd(i) - dd(ip1)) * (t * t)
        } else {
            let t = zz(ip1) / dl(ip1);
            w - dl(i) * dw - (dd(ip1) - dd(i)) * (t * t)
        };
        let mut a = (dl(i) + dl(ip1)) * w - dl(i) * dl(ip1) * dw;
        let b = dl(i) * dl(ip1) * w;
        if c == ZERO {
            if a == ZERO {
                if orgati {
                    a = zz(i) * zz(i) + dl(ip1) * dl(ip1) * (dpsi + dphi);
                } else {
                    a = zz(ip1) * zz(ip1) + dl(i) * dl(i) * (dpsi + dphi);
                }
            }
            eta = b / a;
        } else if a <= ZERO {
            eta = (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c);
        } else {
            eta = TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt());
        }
    } else {
        let temp = rhoinv + psi + phi;
        let (c, z1, z3);
        if orgati {
            let mut temp1 = zz(iim1) / dl(iim1);
            temp1 *= temp1;
            c = temp - dl(iip1) * (dpsi + dphi) - (dd(iim1) - dd(iip1)) * temp1;
            z1 = zz(iim1) * zz(iim1);
            z3 = dl(iip1) * dl(iip1) * ((dpsi - temp1) + dphi);
        } else {
            let mut temp1 = zz(iip1) / dl(iip1);
            temp1 *= temp1;
            c = temp - dl(iim1) * (dpsi + dphi) - (dd(iip1) - dd(iim1)) * temp1;
            z1 = dl(iim1) * dl(iim1) * (dpsi + (dphi - temp1));
            z3 = zz(iip1) * zz(iip1);
        }
        let z2 = zz(ii) * zz(ii);
        let (e, inf) = dlaed6(niter, orgati, c, [dl(iim1), dl(ii), dl(iip1)], [z1, z2, z3], w);
        eta = e;
        if inf != 0 {
            return (0.0, inf);
        }
    }

    if w * eta >= ZERO {
        eta = -w / dw;
    }
    let temp = tau + eta;
    if temp > dltub || temp < dltlb {
        if w < ZERO {
            eta = (dltub - tau) / TWO;
        } else {
            eta = (dltlb - tau) / TWO;
        }
    }

    let mut prew = w;

    for j in 1..=n {
        sdl(j, dl(j) - eta);
    }

    dpsi = ZERO;
    psi = ZERO;
    erretm = ZERO;
    for j in 1..=iim1 {
        let temp = zz(j) / dl(j);
        psi += zz(j) * temp;
        dpsi += temp * temp;
        erretm += psi;
    }
    erretm = erretm.abs();
    dphi = ZERO;
    phi = ZERO;
    for j in (iip1..=n).rev() {
        let temp = zz(j) / dl(j);
        phi += zz(j) * temp;
        dphi += temp * temp;
        erretm += phi;
    }
    let temp = zz(ii) / dl(ii);
    dw = dpsi + dphi + temp * temp;
    let temp = zz(ii) * temp;
    w = rhoinv + phi + psi + temp;
    erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv + THREE * temp.abs() + (tau + eta).abs() * dw;

    let mut swtch = false;
    if orgati {
        if -w > prew.abs() / TEN {
            swtch = true;
        }
    } else if w > prew.abs() / TEN {
        swtch = true;
    }

    tau += eta;

    let iter = niter + 1;
    for niter in iter..=MAXIT {
        if w.abs() <= eps * erretm {
            return (dlam_of(tau), 0);
        }
        if w <= ZERO {
            dltlb = fmax(dltlb, tau);
        } else {
            dltub = fmin(dltub, tau);
        }
        let mut eta;
        if !swtch3 {
            let c = if !swtch {
                if orgati {
                    let t = zz(i) / dl(i);
                    w - dl(ip1) * dw - (dd(i) - dd(ip1)) * (t * t)
                } else {
                    let t = zz(ip1) / dl(ip1);
                    w - dl(i) * dw - (dd(ip1) - dd(i)) * (t * t)
                }
            } else {
                let temp = zz(ii) / dl(ii);
                if orgati {
                    dpsi += temp * temp;
                } else {
                    dphi += temp * temp;
                }
                w - dl(i) * dpsi - dl(ip1) * dphi
            };
            let mut a = (dl(i) + dl(ip1)) * w - dl(i) * dl(ip1) * dw;
            let b = dl(i) * dl(ip1) * w;
            if c == ZERO {
                if a == ZERO {
                    if !swtch {
                        if orgati {
                            a = zz(i) * zz(i) + dl(ip1) * dl(ip1) * (dpsi + dphi);
                        } else {
                            a = zz(ip1) * zz(ip1) + dl(i) * dl(i) * (dpsi + dphi);
                        }
                    } else {
                        a = dl(i) * dl(i) * dpsi + dl(ip1) * dl(ip1) * dphi;
                    }
                }
                eta = b / a;
            } else if a <= ZERO {
                eta = (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c);
            } else {
                eta = TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt());
            }
        } else {
            let temp = rhoinv + psi + phi;
            let (c, z1, z3);
            if swtch {
                c = temp - dl(iim1) * dpsi - dl(iip1) * dphi;
                z1 = dl(iim1) * dl(iim1) * dpsi;
                z3 = dl(iip1) * dl(iip1) * dphi;
            } else if orgati {
                let mut temp1 = zz(iim1) / dl(iim1);
                temp1 *= temp1;
                c = temp - dl(iip1) * (dpsi + dphi) - (dd(iim1) - dd(iip1)) * temp1;
                z1 = zz(iim1) * zz(iim1);
                z3 = dl(iip1) * dl(iip1) * ((dpsi - temp1) + dphi);
            } else {
                let mut temp1 = zz(iip1) / dl(iip1);
                temp1 *= temp1;
                c = temp - dl(iim1) * (dpsi + dphi) - (dd(iip1) - dd(iim1)) * temp1;
                z1 = dl(iim1) * dl(iim1) * (dpsi + (dphi - temp1));
                z3 = zz(iip1) * zz(iip1);
            }
            let z2 = zz(ii) * zz(ii);
            let (e, inf) = dlaed6(niter, orgati, c, [dl(iim1), dl(ii), dl(iip1)], [z1, z2, z3], w);
            eta = e;
            if inf != 0 {
                return (0.0, inf);
            }
        }

        if w * eta >= ZERO {
            eta = -w / dw;
        }
        let temp = tau + eta;
        if temp > dltub || temp < dltlb {
            if w < ZERO {
                eta = (dltub - tau) / TWO;
            } else {
                eta = (dltlb - tau) / TWO;
            }
        }

        for j in 1..=n {
            sdl(j, dl(j) - eta);
        }

        tau += eta;
        prew = w;

        dpsi = ZERO;
        psi = ZERO;
        erretm = ZERO;
        for j in 1..=iim1 {
            let temp = zz(j) / dl(j);
            psi += zz(j) * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = erretm.abs();
        dphi = ZERO;
        phi = ZERO;
        for j in (iip1..=n).rev() {
            let temp = zz(j) / dl(j);
            phi += zz(j) * temp;
            dphi += temp * temp;
            erretm += phi;
        }
        let temp = zz(ii) / dl(ii);
        dw = dpsi + dphi + temp * temp;
        let temp = zz(ii) * temp;
        w = rhoinv + phi + psi + temp;
        erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv + THREE * temp.abs() + tau.abs() * dw;
        if w * prew > ZERO && w.abs() > prew.abs() / TEN {
            swtch = !swtch;
        }
    }
    info = 1;
    (dlam_of(tau), info)
}
