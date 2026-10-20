//! **Path C. The oracle of this module is libcint 6.1.3, NOT Reference-LAPACK
//! 3.12.1, and `ref_*_bitexact` does not and must not cover it.**
//!
//! Pinned source: `cintx/libcint-master/src/eigh.c`, SHA-256
//! `8e04d0ccdf735518e6e045632238680ab9e36936c5d1d475ac14b192d6a3d1a3`, the
//! `#else` (no-`LAPACK_FOUND`) branch — the twelve `static` routines
//! `_dlarrk` (62), `_dlarrc` (118), `_dlasq4` (149), `_dlasq5` (388),
//! `_dlasq2` (431), `_compute_eigenvalues` (629), `_dlarrf` (853),
//! `_dlaneg` (897), `_dlarrb` (934), `_dlar1v` (992),
//! `_compute_eigenvectors` (1095), `_dlaev2` (1381) — plus the
//! `_CINTdiagonalize` glue at 1450.
//!
//! This is a **different MRRR from Reference-LAPACK's**, not a differently
//! rounded one. `docs/cintx-lapack-blas-adoption-plan.md` §4.2 lists six
//! constant-level divergences (no `-4·PIVMIN` bracket term, no pivot floor in
//! the Sturm count, `ITMAX 1000` instead of a computed cap, a different
//! convergence test, `rtl = 2.1e-8` instead of `SQRT(EPS)`, and
//! `eps = 2·DBL_EPSILON` instead of `DLAMCH('P')`), and
//! `docs/a0-wheeler-differential.md` measures the consequence: 23.1% of
//! eigenvalues agree bit for bit on the Wheeler corpus and the worst
//! disagreement is 4.25e9 ULP. Mixing the two contracts in one routine is
//! exactly what §4 rules out — so this module never calls [`crate::stemr`],
//! [`crate::larr`], [`crate::larr2`], [`crate::larre`] or [`crate::lasq`],
//! and they never call it.
//!
//! Gate: `tests/cint_stemr_bitexact.rs` against `crates/cint_ref`
//! (`CINTX_LIBCINT_DIR`), `max_ulp = 0`.
//!
//! Transcription notes:
//!
//! * `MAX(x, y)` / `MIN(x, y)` are the C macros `((x) > (y) ? (x) : (y))` /
//!   `((x) < (y) ? (x) : (y))` — [`cmax`] / [`cmin`] here, never `f64::max`,
//!   which differs on `NaN` and `-0.0`.
//! * `DBL_EPSILON` is `f64::EPSILON` (`2^-52`), i.e. **twice** the
//!   `DLAMCH('E')` this crate uses elsewhere. `eps = 2 * DBL_EPSILON` in
//!   `_compute_eigenvalues` is therefore `2^-51`.
//! * `wgap[-1]` is a real, load-bearing negative index (`eigh.c:830` writes it,
//!   `eigh.c:1220` reads it): `wgap` points at `work + 2n + 1`, so `wgap[-1]`
//!   is `work[2n]`. [`Wgap`] keeps that addressing explicit instead of
//!   silently dropping the slot.
//! * `_dlasq2` swaps the `qvecp`/`qvec1p`/`evecp`/`evec1p` pointers each
//!   `iwhilb` step while still writing converged eigenvalues into the *base*
//!   `qvec`; both are reproduced.
//! * `_dlarrb` carries `mid`/`width` across its `i` loop (they are C
//!   function-scope locals, not re-initialised per `i`), and `continue`
//!   leaves them untouched.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::ptr::{D, J};

/// libcint's `MXRYSROOTS`.
pub const MXRYSROOTS: usize = 13;

const MAXTRY: usize = 6;
const MAXRQITER: usize = 6;
const ITMAX: usize = 1000;
const FAIL_THRESHOLD: f64 = 1e16;
// `RTOL1`/`RTOL2` (`eigh.c:57-58`) are only referenced by the `ndepth > 0`
// limited-bisection block, which is COMMENTED OUT upstream
// (`eigh.c:1188-1205`). Kept because they are part of the pinned source's
// constant set and a future libcint that re-enables that block would need
// them unchanged.
#[allow(dead_code)]
const RTOL1: f64 = 1.5e-8;
#[allow(dead_code)]
const RTOL2: f64 = 7.45e-11;
const MINGAP: f64 = 0.001;
/// C's `DBL_EPSILON`.
const DBL_EPSILON: f64 = f64::EPSILON;

/// `#define MAX(x, y) ((x) > (y) ? (x) : (y))`.
#[inline(always)]
fn cmax(x: f64, y: f64) -> f64 {
    if x > y { x } else { y }
}

/// `#define MIN(x, y) ((x) < (y) ? (x) : (y))`.
#[inline(always)]
fn cmin(x: f64, y: f64) -> f64 {
    if x < y { x } else { y }
}

/// `wgap`, which libcint indexes from `-1` (`eigh.c:830`, `eigh.c:1220`).
/// The view is based one slot *before* the C pointer.
#[derive(Clone, Copy)]
struct Wgap(D);

impl Wgap {
    /// `wgap[i]`, `i >= -1`.
    #[inline(always)]
    unsafe fn at(self, i: isize) -> f64 {
        self.0.at((i + 1) as usize)
    }
    /// `wgap[i] = v`, `i >= -1`.
    #[inline(always)]
    unsafe fn set(self, i: isize, v: f64) {
        self.0.set((i + 1) as usize, v)
    }
}

/// `_dlarrk( n, iw, gl, gu, diag, e2, reltol, w, werr )` (`eigh.c:62`).
///
/// Returns `(w, werr, info)`. Note what is **absent** relative to
/// `dlarrk.f`: no `-FUDGE*TWO*PIVMIN` bracket term, no `PIVMIN` pivot floor,
/// a fixed `ITMAX = 1000`, and a `tmp1 <= 0 || tmp1 < reltol*MAX(..)`
/// convergence test.
unsafe fn dlarrk(n: usize, iw: i32, gl: f64, gu: f64, diag: D, e2: D, reltol: f64) -> (f64, f64, i32) {
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    let tnorm = cmax(gl.abs(), gu.abs());
    let mut info: i32 = -1;
    let mut left = gl - tnorm * 2. * DBL_EPSILON * n as f64;
    let mut right = gu + tnorm * 2. * DBL_EPSILON * n as f64;

    for _it in 0..ITMAX {
        let tmp1 = (right - left).abs();
        if tmp1 <= 0.0 || tmp1 < reltol * cmax(right.abs(), left.abs()) {
            info = 0;
            break;
        }

        // Number of negative pivots at the mid-point.
        let mid = (left + right) * 0.5;
        let mut negcnt: i32 = 0;
        let mut tmp1 = diag.at(0) - mid;
        if tmp1 <= 0.0 {
            negcnt += 1;
        }
        for i in 1..n {
            tmp1 = diag.at(i) - e2.at(i - 1) / tmp1 - mid;
            if tmp1 <= 0.0 {
                negcnt += 1;
            }
        }
        if negcnt >= iw {
            right = mid;
        } else {
            left = mid;
        }
    }

    ((left + right) * 0.5, (right - left).abs() * 0.5, info)
}

/// `_dlarrc( n, vl, vu, diag, e2, lcnt, rcnt )` (`eigh.c:118`). Returns
/// `(lcnt, rcnt)`.
unsafe fn dlarrc(n: usize, vl: f64, vu: f64, diag: D, e2: D) -> (i32, i32) {
    let mut left_count: i32 = 0;
    let mut right_count: i32 = 0;
    let mut lpivot = diag.at(0) - vl;
    let mut rpivot = diag.at(0) - vu;
    if lpivot <= 0.0 {
        left_count += 1;
    }
    if rpivot <= 0.0 {
        right_count += 1;
    }
    for i in 0..n - 1 {
        let tmp = e2.at(i);
        lpivot = diag.at(i + 1) - vl - tmp / lpivot;
        rpivot = diag.at(i + 1) - vu - tmp / rpivot;
        if lpivot <= 0.0 {
            left_count += 1;
        }
        if rpivot <= 0.0 {
            right_count += 1;
        }
    }
    (left_count, right_count)
}

/// `_dlasq4( i0, n0, n0init, qvecp, qvec1p, evecp, evec1p, dmin, dn, tau )`
/// (`eigh.c:149`): the dqds shift.
///
/// Every `return 0` before `*tau = s` leaves `tau` at its incoming value, as
/// in the C.
unsafe fn dlasq4(
    i0: usize,
    n0: usize,
    n0init: usize,
    qvecp: D,
    qvec1p: D,
    evecp: D,
    evec1p: D,
    dmin: &[f64; 3],
    dn: &[f64; 3],
    tau: &mut f64,
) -> i32 {
    let mut s = 0.0f64;
    // The `for (i = ..; i >= i0; i--)` loops can start at -1; keep them signed.
    let i0s = i0 as isize;
    let n0s = n0 as isize;
    let q = |i: isize| qvecp.at(i as usize);
    let q1 = |i: isize| qvec1p.at(i as usize);
    let e = |i: isize| evecp.at(i as usize);
    let e1 = |i: isize| evec1p.at(i as usize);

    if n0init == n0 {
        // No eigenvalues deflated.
        if dmin[0] == dn[0] {
            if dmin[1] == dn[1] {
                // Cases 2 and 3.
                let b1 = (q(n0s - 1) * e(n0s - 2)).sqrt();
                let b2 = (q(n0s - 2) * e(n0s - 3)).sqrt();
                let a2 = q(n0s - 2) + e(n0s - 2);
                let gap2 = dmin[2] - a2 - dmin[2] * 0.25;
                let gap1 = if gap2 > b2 { a2 - dn[0] - b2 / gap2 * b2 } else { a2 - dn[0] - (b1 + b2) };
                if gap1 > b1 {
                    s = cmax(dn[0] - b1 / gap1 * b1, dmin[0] * 0.5);
                } else {
                    s = 0.0;
                    if dn[0] > b1 {
                        s = dn[0] - b1;
                    }
                    if a2 > b1 + b2 {
                        s = cmin(s, a2 - (b1 + b2));
                    }
                    s = cmax(s, dmin[0] * 0.333);
                }
            } else {
                // Case 4.
                if e(n0s - 2) > q(n0s - 2) {
                    return 0;
                }
                let mut b2 = e(n0s - 2) / q(n0s - 2);
                let mut a2 = b2;

                // Approximate contribution to norm squared from I < NN-1.
                let mut i = n0s - 3;
                while i >= i0s {
                    let b1 = b2;
                    if e(i) > q(i) {
                        return 0;
                    }
                    b2 *= e(i) / q(i);
                    a2 += b2;
                    if 0.563 < a2 || cmax(b1, b2) < a2 * 0.01 {
                        break;
                    }
                    i -= 1;
                }
                a2 *= 1.05;

                // Rayleigh quotient residual bound.
                s = if a2 < 0.563 { dn[0] * (1. - a2.sqrt()) / (a2 + 1.) } else { dmin[0] * 0.25 };
            }
        } else if dmin[0] == dn[1] {
            // Case 4.
            if e1(n0s - 2) > q1(n0s - 1) || e(n0s - 3) > q(n0s - 3) {
                return 0;
            }
            let mut a2 = e1(n0s - 2) / q1(n0s - 1);
            let mut b2 = e(n0s - 3) / q(n0s - 3);

            a2 += b2;
            let mut i = n0s - 4;
            while i >= i0s {
                if b2 == 0.0 {
                    break;
                }
                let b1 = b2;
                if e(i) > q(i) {
                    return 0;
                }
                b2 *= e(i) / q(i);
                a2 += b2;
                if cmax(b2, b1) * 100. < a2 || 0.563 < a2 {
                    break;
                }
                i -= 1;
            }
            a2 *= 1.05;

            s = if a2 < 0.563 { dn[1] * (1. - a2.sqrt()) / (a2 + 1.) } else { dmin[0] * 0.25 };
        } else if dmin[0] == dn[2] {
            // Case 5.
            if e1(n0s - 3) > q1(n0s - 2) || e1(n0s - 2) > q1(n0s - 1) {
                return 0;
            }
            let mut a2 = e1(n0s - 3) / q1(n0s - 2) * (e1(n0s - 2) / q1(n0s - 1) + 1.);

            // Approximate contribution to norm squared from I < NN-2.
            if n0 - i0 > 3 {
                let mut b2 = e(n0s - 4) / q(n0s - 4);
                a2 += b2;
                let mut i = n0s - 5;
                while i >= i0s {
                    let b1 = b2;
                    if e(i) > q(i) {
                        return 0;
                    }
                    b2 *= e(i) / q(i);
                    a2 += b2;
                    if 0.563 < a2 || cmax(b2, b1) < a2 * 0.01 {
                        break;
                    }
                    i -= 1;
                }
                a2 *= 1.05;
            }

            s = if a2 < 0.563 { dn[2] * (1. - a2.sqrt()) / (a2 + 1.) } else { dmin[0] * 0.25 };
        } else {
            // Case 6, no information to guide us.
            s = dmin[0] * 0.25;
        }
    } else if n0init == n0 + 1 {
        // One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN.
        if dmin[1] == dn[1] && dmin[2] == dn[2] {
            if e(n0s - 2) > q(n0s - 2) {
                return 0;
            }
            s = dmin[1] * 0.333;
            let mut b1 = e(n0s - 2) / q(n0s - 2);
            let mut b2 = b1;
            if b2 != 0.0 {
                let mut i = n0s - 3;
                while i >= i0s {
                    let a2 = b1;
                    if e(i) > q(i) {
                        return 0;
                    }
                    b1 *= e(i) / q(i);
                    b2 += b1;
                    if cmax(b1, a2) * 100. < b2 {
                        break;
                    }
                    i -= 1;
                }
            }
            let a2 = dmin[1] / (b2 * 1.05 + 1.);
            let b2 = (b2 * 1.05).sqrt();
            let gap2 = dmin[2] * 0.5 - a2;
            if gap2 > 0.0 && gap2 > b2 * a2 {
                s = cmax(s, a2 * (1. - a2 * 1.01 * (b2 / gap2) * b2));
            } else {
                s = cmax(s, a2 * (1. - b2 * 1.01));
            }
        } else if dmin[1] == dn[1] {
            s = dmin[1] * 0.5;
        } else {
            s = dmin[1] * 0.25;
        }
    } else if n0init == n0 + 2 {
        // Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.
        if dmin[2] == dn[2] && e(n0s - 2) * 2. < q(n0s - 2) {
            if e(n0s - 2) > q(n0s - 2) {
                return 0;
            }
            let mut b1 = e(n0s - 2) / q(n0s - 2);
            let mut b2 = b1;
            if b2 != 0.0 {
                let mut i = n0s - 2;
                while i > i0s {
                    if e(i - 1) > q(i - 1) {
                        return 0;
                    }
                    b1 *= e(i - 1) / q(i - 1);
                    b2 += b1;
                    if b1 * 100. < b2 {
                        break;
                    }
                    i -= 1;
                }
            }
            s = dmin[2] * 0.333;
            let a2 = dmin[2] / (b2 * 1.05 + 1.);
            let b2 = (b2 * 1.05).sqrt();
            let gap2 = q(n0s - 2) + e(n0s - 3) - (q(n0s - 3) * e(n0s - 3)).sqrt() - a2;
            if gap2 > 0.0 && gap2 > b2 * a2 {
                s = cmax(s, a2 * (1. - a2 * 1.01 * (b2 / gap2) * b2));
            } else {
                s = cmax(s, a2 * (1. - b2 * 1.01));
            }
        } else {
            s = dmin[2] * 0.25;
        }
    } else if n0init > n0 + 2 {
        // Case 12, more than two eigenvalues deflated. No information.
        s = 0.0;
    }

    *tau = s;
    0
}

/// `_dlasq5( i0, n0, qvecp, qvec1p, evecp, evec1p, tau, tol, dmin, dn )`
/// (`eigh.c:388`): one dqds step.
unsafe fn dlasq5(
    i0: usize,
    n0: usize,
    qvecp: D,
    qvec1p: D,
    evecp: D,
    evec1p: D,
    tau: f64,
    tol: f64,
    dmin: &mut [f64; 3],
    dn: &mut [f64; 3],
) {
    let mut diag = qvecp.at(i0) - tau;
    let mut diag_min = diag;

    let mut j = i0;
    while j + 3 < n0 {
        qvec1p.set(j, diag + evecp.at(j));
        let temp = qvecp.at(j + 1) / qvec1p.at(j);
        diag = diag * temp - tau;
        if diag < tol {
            diag = 0.0;
        }
        diag_min = cmin(diag_min, diag);
        evec1p.set(j, evecp.at(j) * temp);
        j += 1;
    }
    dn[2] = diag;

    let j = n0 - 3;
    qvec1p.set(j, diag + evecp.at(j));
    let temp = qvecp.at(j + 1) / qvec1p.at(j);
    evec1p.set(j, evecp.at(j) * temp);
    diag = diag * temp - tau;
    dn[1] = diag;

    let j = n0 - 2;
    qvec1p.set(j, diag + evecp.at(j));
    let temp = qvecp.at(j + 1) / qvec1p.at(j);
    evec1p.set(j, evecp.at(j) * temp);
    diag = diag * temp - tau;
    dn[0] = diag;

    qvec1p.set(n0 - 1, diag);

    dmin[2] = diag_min;
    dmin[1] = cmin(dmin[2], dn[1]);
    dmin[0] = cmin(dmin[1], dn[0]);
}

/// `_dlasq2( n, work, diag, diag_off )` (`eigh.c:431`).
///
/// `work` is `4 * n` carved as `qvec | qvec1 | evec | evec1`; the eigenvalues
/// come back in `work[0..n]` (the base `qvec`).
unsafe fn dlasq2(n: usize, work: D, diag: D, diag_off: D) -> i32 {
    debug_assert!(n > 2);
    let qvec = work;
    let qvec1 = work.off(n);
    let evec = work.off(n * 2);
    let evec1 = work.off(n * 3);

    let mut j = n - 1;
    for i in 0..n - 1 {
        let temp = diag.at(i).abs();
        qvec1.set(i, 0.0);
        evec1.set(i, 0.0);
        evec.set(j - 1, diag_off.at(i) * diag_off.at(i) * temp);
        qvec.set(j, temp);
        j -= 1;
    }
    qvec.set(0, diag.at(n - 1).abs());
    qvec1.set(n - 1, 0.0);
    evec.set(n - 1, 0.0);
    evec1.set(n - 1, 0.0);

    // Reverse the qd-array, if warranted.
    if qvec.at(0) < qvec.at(n - 1) * 1.5 {
        let mut j = n - 1;
        for i in 0..n / 2 {
            let temp = qvec.at(i);
            qvec.set(i, qvec.at(j));
            qvec.set(j, temp);

            let temp = evec.at(i);
            evec.set(i, evec.at(j - 1));
            evec.set(j - 1, temp);
            j -= 1;
        }
    }

    // dqd maps Z to ZZ plus Li's test.
    let mut diag_sum = qvec.at(0);
    for i in 0..n - 1 {
        let temp = diag_sum + evec.at(i);
        qvec1.set(i, temp);
        evec1.set(i, qvec.at(i + 1) * (evec.at(i) / temp));
        diag_sum = qvec.at(i + 1) * (diag_sum / temp);
    }
    qvec1.set(n - 1, diag_sum);
    let mut diag_sum = qvec1.at(0);
    for i in 0..n - 1 {
        let temp = diag_sum + evec1.at(i);
        qvec.set(i, temp);
        evec.set(i, qvec1.at(i + 1) * (evec1.at(i) / temp));
        diag_sum = qvec1.at(i + 1) * (diag_sum / temp);
    }
    qvec.set(n - 1, diag_sum);

    let mut n0 = n;
    let mut tau = 0.0f64;
    let mut dmin = [0.0f64; 3];
    let mut dn = [0.0f64; 3];

    let mut itry = 0usize;
    while n0 > 0 {
        if itry >= n {
            return 3;
        }
        itry += 1;

        // E(N0) holds the (negated) SIGMA at which I0:N0 split off.
        let mut sigma = -evec.at(n0 - 1);
        if sigma < 0.0 {
            return 1;
        }

        // Last unreduced submatrix's top index I0, QMAX and EMIN.
        let mut emax = 0.0f64;
        let mut qmin = qvec.at(n0 - 1);
        let mut i = n0 - 1;
        while i > 0 {
            if evec.at(i - 1) <= 0.0 {
                break;
            }
            if qmin >= emax * 4. {
                qmin = cmin(qmin, qvec.at(i));
                emax = cmax(emax, evec.at(i - 1));
            }
            i -= 1;
        }

        let i0 = i;
        let mut qvecp = qvec;
        let mut qvec1p = qvec1;
        let mut evecp = evec;
        let mut evec1p = evec1;

        if qmin < 0.0 || emax < 0.0 {
            return 1;
        }

        // Put -(initial shift) into DMIN.
        dmin[0] = -cmax(0.0, qmin - 2. * (qmin * emax).sqrt());

        // Now I0:N0 is unreduced.
        let nbig = (n0 - i0) * 10;
        let mut iwhilb = 0usize;
        let mut broke = false;
        while iwhilb < nbig {
            let n0init = n0;
            let tol = DBL_EPSILON * 100.;
            let tol2 = tol * tol;

            while n0 > i0 {
                let n1 = n0 - 1;
                // `eigh.c` computes `n2 = n0 - 2` unconditionally as a signed
                // `int` (`-1` when `n0 == 1`, e.g. the last 1x1 deflation with
                // `i0 == 0`); it is never read on that path because
                // `n1 == i0` short-circuits the `||` chain. A `usize` cannot
                // hold that `-1`, so the guard is spelled out: reaching `n2`
                // requires `n0 >= 2`, which `n1 != i0` implies
                // (`i0 <= n0 - 1`). Reads evaluate in the same order on every
                // path, so the values are unchanged.
                if n1 == i0 {
                    qvec.set(n1, qvecp.at(n1) + sigma);
                    n0 -= 1;
                    continue;
                }
                let n2 = n0 - 2;
                if evecp.at(n2) < tol2 * (sigma + qvecp.at(n1)) || evec1p.at(n2) < tol2 * qvecp.at(n2) {
                    qvec.set(n1, qvecp.at(n1) + sigma);
                    n0 -= 1;
                    continue;
                }
                if n2 == i0 || evecp.at(n1 - 2) < tol2 * sigma || evec1p.at(n1 - 2) < tol2 * qvecp.at(n1 - 2) {
                    if qvecp.at(n1) > qvecp.at(n2) {
                        let s = qvecp.at(n1);
                        qvecp.set(n1, qvecp.at(n2));
                        qvecp.set(n2, s);
                    }
                    let mut t = (qvecp.at(n2) - qvecp.at(n1) + evecp.at(n2)) * 0.5;
                    if evecp.at(n2) > qvecp.at(n1) * tol2 && t != 0.0 {
                        let s = qvecp.at(n1) * (evecp.at(n2) / t);
                        let s = qvecp.at(n1) * (evecp.at(n2) / (t + (t * (t + s)).sqrt()));
                        t = qvecp.at(n2) + (s + evecp.at(n2));
                        qvecp.set(n1, qvecp.at(n1) * (qvecp.at(n2) / t));
                        qvecp.set(n2, t);
                    }
                    qvec.set(n2, qvecp.at(n2) + sigma);
                    qvec.set(n1, qvecp.at(n1) + sigma);
                    n0 -= 2;
                    continue;
                }
                break;
            }
            if n0 <= i0 {
                broke = true;
                break;
            }

            if dmin[0] <= 0.0 {
                tau = -dmin[0];
            } else {
                dlasq4(i0, n0, n0init, qvecp, qvec1p, evecp, evec1p, &dmin, &dn, &mut tau);
            }

            // Call dqds until DMIN > 0.
            let tol = DBL_EPSILON * sigma;
            for _iter in 0..3 {
                dlasq5(i0, n0, qvecp, qvec1p, evecp, evec1p, tau, tol, &mut dmin, &mut dn);
                if dmin[0] >= 0.0 {
                    break;
                } else if dmin[1] > 0.0 {
                    tau += dmin[0];
                } else {
                    tau *= 0.25;
                }
            }
            if dmin[0] < 0.0 {
                tau = 0.0;
                dlasq5(i0, n0, qvecp, qvec1p, evecp, evec1p, tau, tol, &mut dmin, &mut dn);
            }

            sigma += tau;

            // Swap vec / vec1.
            std::mem::swap(&mut qvecp, &mut qvec1p);
            std::mem::swap(&mut evecp, &mut evec1p);
            iwhilb += 1;
        }

        if !broke && iwhilb == nbig {
            // Not converged.
            return 2;
        }
    }
    0
}

/// `_compute_eigenvalues( n, diag, diag_off1, w, werr, wgap, work )`
/// (`eigh.c:629`).
///
/// `work` needs `5 * n`; `wgap` is a [`Wgap`] because `eigh.c:830` writes
/// `wgap[-1]`.
unsafe fn compute_eigenvalues(n: usize, diag: D, diag_off1: D, w: D, werr: D, wgap: Wgap, work: D) -> i32 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        w.set(0, diag.at(0));
        werr.set(0, 0.0);
        wgap.set(0, 0.0);
        // Store the shift for the initial RRR, which is zero in this case.
        diag_off1.set(0, 0.0);
        return 0;
    }

    let eps = 2. * DBL_EPSILON;
    let rtl = 2.1e-8; // "sqrt(eps)", hard-coded
    let rtol = 16. * DBL_EPSILON;

    // Init WERR, WGAP; Gerschgorin intervals and spectral diameter.
    let mut gl = diag.at(0);
    let mut gu = diag.at(0);
    let mut eold = 0.0f64;
    diag_off1.set(n - 1, 0.0);
    for i in 0..n {
        werr.set(i, 0.0);
        wgap.set(i as isize, 0.0);
        let eabs = diag_off1.at(i).abs();
        let tmp1 = eabs + eold;
        gl = cmin(gl, diag.at(i) - tmp1);
        gu = cmax(gu, diag.at(i) + tmp1);
        eold = eabs;
    }
    // Overwritten by `isrght - isleft` below before it is read, as in the C.
    #[allow(unused_assignments)]
    let mut spectral_diameter = gu - gl;
    // Will hold the shift for the initial RRR; zero for now.
    diag_off1.set(n - 1, 0.0);

    let e2 = work;
    let work = work.off(n);
    for i in 0..n - 1 {
        e2.set(i, diag_off1.at(i) * diag_off1.at(i));
    }

    // Case of DQDS: approximations to the extremal eigenvalues of the block.
    let (tmp, tmp1, iinfo) = dlarrk(n, 1, gl, gu, diag, e2, rtl);
    if iinfo != 0 {
        return -1;
    }
    let isleft = cmax(gl, tmp - tmp1 - eps * 100. * (tmp - tmp1).abs());
    let (tmp, tmp1, iinfo) = dlarrk(n, n as i32, gl, gu, diag, e2, rtl);
    if iinfo != 0 {
        return -1;
    }
    let isrght = cmin(gu, tmp + tmp1 + eps * 100. * (tmp + tmp1).abs());
    // Improve the estimate of the spectral diameter.
    spectral_diameter = isrght - isleft;

    // Negcount at the 1/4 and 3/4 points.
    let (lcnt, rcnt) = dlarrc(n, isleft + spectral_diameter * 0.25, isrght - spectral_diameter * 0.25, diag, e2);

    let mut sigma;
    let sgndef;
    if lcnt - 1 >= n as i32 - rcnt {
        sigma = cmax(isleft, gl);
        sgndef = 1.0f64;
    } else {
        sigma = cmin(isrght, gu);
        sgndef = -1.0f64;
    }

    // The shift increment TAU, in case SIGMA needs refining.
    let mut tau = cmax(spectral_diameter * n as f64, 2. * sigma.abs()) * eps;

    for idum in 0..MAXTRY {
        // L D L^T of T - sigma I: D in work[0..n], L in work[n..2n],
        // reciprocal pivots in work[2n..3n].
        let mut dpivot = diag.at(0) - sigma;
        work.set(0, dpivot);
        let mut dmax = work.at(0).abs();
        for i in 0..n - 1 {
            work.set(n * 2 + i, 1. / work.at(i));
            let tmp = diag_off1.at(i) * work.at(n * 2 + i);
            work.set(n + i, tmp);
            dpivot = diag.at(i + 1) - sigma - tmp * diag_off1.at(i);
            work.set(i + 1, dpivot);
            dmax = cmax(dmax, dpivot.abs());
        }
        // Element growth.
        let mut norep = dmax > spectral_diameter * 64.;
        if !norep {
            // Every entry of D (of L D L^T) must have the same sign.
            for i in 0..n {
                let tmp = sgndef * work.at(i);
                if tmp < 0.0 {
                    norep = true;
                    break;
                }
            }
        }
        if norep {
            if idum == MAXTRY - 1 {
                // The fudged Gerschgorin shift should succeed.
                if sgndef == 1.0 {
                    sigma = gl - spectral_diameter * 2. * eps * n as f64;
                } else {
                    sigma = gu + spectral_diameter * 2. * eps * n as f64;
                }
            } else {
                sigma -= sgndef * tau;
                tau *= 2.;
            }
        } else {
            break;
        }
    }

    // An initial base representation was found. Store the shift, D and L.
    diag_off1.set(n - 1, sigma);
    for ip in 0..n {
        diag.set(ip, work.at(ip));
    }
    for ip in 0..n - 1 {
        diag_off1.set(ip, work.at(n + ip));
    }

    // dqds for all eigenvalues of the block.
    let iinfo = dlasq2(n, work, diag, diag_off1);
    if iinfo != 0 {
        return -5;
    }
    for i in 0..n {
        if work.at(i) < 0.0 {
            return -6;
        }
    }
    if sgndef > 0.0 {
        for i in 0..n {
            w.set(i, work.at(n - 1 - i));
        }
    } else {
        for i in 0..n {
            w.set(i, -work.at(i));
        }
    }

    for i in 0..n {
        werr.set(i, rtol * w.at(i).abs());
    }

    // The right gap between the intervals.
    for i in 0..n - 1 {
        wgap.set(i as isize, cmax(0.0, w.at(i + 1) - werr.at(i + 1) - (w.at(i) + werr.at(i))));
    }
    // `wgap[-1]` (`eigh.c:830`): the distance to GL, read back by
    // `_compute_eigenvectors` when the leftmost cluster starts at index 0.
    wgap.set(-1, cmax(0.0, w.at(0) - werr.at(0) - gl));
    wgap.set(n as isize - 1, cmax(0.0, gu - sigma - (w.at(n - 1) + werr.at(n - 1))));

    0
}

/// `_dlarrf( n, diag, diag_off1, ld, clstrt, w, wgap, werr, clgapl, sigma,
/// dplus, lplus )` (`eigh.c:853`). Returns `(sigma, info)`.
unsafe fn dlarrf(
    n: usize,
    diag: D,
    diag_off1: D,
    ld: D,
    clstrt: usize,
    w: D,
    wgap: Wgap,
    werr: D,
    clgapl: f64,
    dplus: D,
    lplus: D,
) -> (f64, i32) {
    // A small fudge, to really shift to the outside.
    let mut lsigma = w.at(clstrt) - werr.at(clstrt);
    lsigma -= lsigma.abs() * 4. * DBL_EPSILON;

    let growthbound = diag.at(0) * 8.;
    let mut sigma = 0.0f64;
    let mut max1 = 0.0f64;

    for _ktry in 0..2 {
        // Element growth when shifting to the left end of the cluster.
        let mut s = -lsigma;
        dplus.set(0, diag.at(0) + s);
        max1 = dplus.at(0).abs();
        for i in 0..n - 1 {
            let tmp = ld.at(i) / dplus.at(i);
            lplus.set(i, tmp);
            s = s * tmp * diag_off1.at(i) - lsigma;
            dplus.set(i + 1, diag.at(i + 1) + s);
            max1 = cmax(max1, dplus.at(i + 1).abs());
        }
        sigma = lsigma;
        if max1 <= growthbound {
            return (sigma, 0);
        }

        // The shift failed the RRR test; back off to the outside.
        lsigma -= cmin(clgapl * 0.25, wgap.at(clstrt as isize));
    }

    if max1 > FAIL_THRESHOLD {
        return (sigma, 1);
    }
    (sigma, 0)
}

/// `_dlaneg( n, diag, lld, sigma, twist_index )` (`eigh.c:897`).
///
/// No `BLKLEN` blocking and no NaN rerun: libcint's is the plain two-sweep
/// form.
unsafe fn dlaneg(n: usize, diag: D, lld: D, sigma: f64, twist_index: i32) -> i32 {
    let mut negcnt: i32 = 0;
    // I) upper part.
    let mut t = -sigma;
    let mut j: i32 = 0;
    while j < twist_index - 1 {
        let dplus = diag.at(j as usize) + t;
        if dplus < 0.0 {
            negcnt += 1;
        }
        t = t / dplus * lld.at(j as usize) - sigma;
        j += 1;
    }

    // II) lower part.
    let mut p = diag.at(n - 1) - sigma;
    let mut j: i32 = n as i32 - 2;
    while j >= twist_index - 1 {
        let dminus = lld.at(j as usize) + p;
        if dminus < 0.0 {
            negcnt += 1;
        }
        p = p / dminus * diag.at(j as usize) - sigma;
        j -= 1;
    }

    // III) twist index. T was shifted by SIGMA initially.
    if t + sigma + p < 0.0 {
        negcnt += 1;
    }
    negcnt
}

/// `_dlarrb( n, diag, lld, ifirst, ilast, rtol1, rtol2, w, wgap, werr,
/// twist_index )` (`eigh.c:934`).
///
/// `mid`/`width` are C function-scope locals carried across the `i` loop: an
/// `i` that `continue`s leaves them at the previous `i`'s values.
unsafe fn dlarrb(
    n: usize,
    diag: D,
    lld: D,
    ifirst: usize,
    ilast: usize,
    rtol1: f64,
    rtol2: f64,
    w: D,
    wgap: Wgap,
    werr: D,
    twist_index: i32,
) -> i32 {
    let cvrgd = cmax(rtol1 * wgap.at(ifirst as isize), rtol2 * cmax(w.at(ifirst).abs(), w.at(ilast - 1).abs()));

    let mut mid = 0.0f64;
    let mut width = 0.0f64;
    for i in ifirst..ilast {
        if werr.at(i) < cvrgd {
            continue;
        }

        // Make sure [LEFT, RIGHT] contains the desired eigenvalue.
        let mut left = w.at(i);
        let mut back = werr.at(i);
        let mut negcnt = ilast as i32;
        while negcnt > i as i32 {
            left -= back;
            back *= 2.;
            negcnt = dlaneg(n, diag, lld, left, twist_index);
        }

        let mut right = w.at(i);
        let mut back = werr.at(i);
        let mut negcnt = ifirst as i32;
        while negcnt <= i as i32 {
            right += back;
            back *= 2.;
            negcnt = dlaneg(n, diag, lld, right, twist_index);
        }

        for _iter in 0..ITMAX {
            // One bisection step.
            mid = (left + right) * 0.5;
            width = right - mid;
            if width < cvrgd {
                break;
            }
            let negcnt = dlaneg(n, diag, lld, mid, twist_index);
            if negcnt <= i as i32 {
                left = mid;
            } else {
                right = mid;
            }
        }
        w.set(i, mid);
        werr.set(i, width);
    }

    for i in ifirst..ilast - 1 {
        wgap.set(i as isize, cmax(0.0, w.at(i + 1) - werr.at(i + 1) - w.at(i) - werr.at(i)));
    }
    0
}

/// `_dlar1v( n, lambda, diag, diag_off1, ld, lld, gaptol, vec, negcnt,
/// twist_index, resid, rqcorr, work )` (`eigh.c:992`).
///
/// `work` is `3 * n`: `lplus | uminus | work_p`. Unlike `dlar1v.f` there is no
/// `PIVMIN` floor, no `SAWNAN` rerun and no `ISUPPZ`; the vector is normalised
/// in place at the end.
unsafe fn dlar1v(
    n: usize,
    lambda: f64,
    diag: D,
    diag_off1: D,
    ld: D,
    lld: D,
    _gaptol: f64,
    vec: D,
    twist_index: &mut i32,
    work: D,
) -> (i32, f64, f64) {
    let lplus = work;
    let uminus = work.off(n);
    let work_p = work.off(n * 2);

    let (r1, r2) = if *twist_index == -1 {
        *twist_index = 0;
        (0usize, n)
    } else {
        (*twist_index as usize, *twist_index as usize + 1)
    };

    // Progressive transform (differential form) until the index R1.
    let mut neg2: i32 = 0;
    let mut s = diag.at(n - 1) - lambda;
    work_p.set(n - 1, s);
    let mut i = n as isize - 2;
    while i >= r1 as isize {
        let iu = i as usize;
        let dminus = lld.at(iu) + s;
        if dminus < 0.0 {
            neg2 += 1;
        }
        let tmp = diag.at(iu) / dminus;
        uminus.set(iu, diag_off1.at(iu) * tmp);
        s = s * tmp - lambda;
        work_p.set(iu, s);
        i -= 1;
    }

    // Stationary transform (differential form) until the index R2.
    let mut neg1: i32 = 0;
    let mut s = -lambda;
    for i in 0..r1 {
        let dplus = diag.at(i) + s;
        if dplus < 0.0 {
            neg1 += 1;
        }
        lplus.set(i, ld.at(i) / dplus);
        s = s * lplus.at(i) * diag_off1.at(i) - lambda;
    }
    let mut mingma = s + lambda + work_p.at(r1);
    if mingma < 0.0 {
        neg1 += 1;
    }

    let negcnt = neg1 + neg2;

    // Index (from R1 to R2) of the largest (in magnitude) diagonal element of
    // the inverse.
    for i in r1..r2 - 1 {
        let dplus = diag.at(i) + s;
        lplus.set(i, ld.at(i) / dplus);
        let tmp = s * lplus.at(i) * diag_off1.at(i);
        s = tmp - lambda;

        let tmp = tmp + work_p.at(i + 1);
        if tmp.abs() <= mingma.abs() {
            mingma = tmp;
            *twist_index = i as i32 + 1;
        }
    }

    // FP vector: solve N^T v = e_r.
    vec.set(*twist_index as usize, 1.0);
    let mut ztz = 1.0f64;

    let mut i = *twist_index - 1;
    while i >= 0 {
        let iu = i as usize;
        let tmp = -(lplus.at(iu) * vec.at(iu + 1));
        ztz += tmp * tmp;
        vec.set(iu, tmp);
        i -= 1;
    }

    for i in *twist_index as usize..n - 1 {
        let tmp = -(uminus.at(i) * vec.at(i));
        ztz += tmp * tmp;
        vec.set(i + 1, tmp);
    }

    let tmp = 1. / ztz;
    let nrminv = tmp.sqrt();
    for i in 0..n {
        vec.set(i, vec.at(i) * nrminv);
    }
    (negcnt, mingma.abs() * nrminv, mingma * tmp)
}

/// `_compute_eigenvectors( n, diag, diag_off1, w, werr, wgap, vec, work,
/// iwork )` (`eigh.c:1095`).
///
/// `work` needs `6 * n`, `iwork` needs `5 * n`.
unsafe fn compute_eigenvectors(
    n: usize,
    diag: D,
    diag_off1: D,
    w: D,
    werr: D,
    wgap: Wgap,
    vec: D,
    work: D,
    iwork: J,
) -> i32 {
    if n == 0 {
        return 0;
    }
    let buf_w = work;
    let buf_ld = work.off(n);
    let buf_lld = work.off(n * 2);
    let buf_wrk = work.off(n * 3);
    for i in 0..n * 6 {
        work.set(i, 0.0);
    }
    let twist_indices = iwork;
    for i in 0..n {
        twist_indices.set(i, 0);
    }
    let mut old_cluster_range = iwork.off(n);
    let mut new_cluster_range = iwork.off(n * 3);

    let eps = DBL_EPSILON;
    let rqtol = DBL_EPSILON * 2.;
    let tol = DBL_EPSILON * 8.;

    let mut sigma = diag_off1.at(n - 1);
    if n == 1 {
        vec.set(0, 1.0);
        w.set(0, w.at(0) + sigma);
        return 0;
    }

    for i in 0..n {
        buf_w.set(i, w.at(i));
        w.set(i, w.at(i) + sigma);
    }
    let mut ncluster = 1usize;
    old_cluster_range.set(0, 0);
    old_cluster_range.set(1, n as i32);
    let mut idone = 0usize;

    for ndepth in 0..n {
        if idone == n {
            break;
        }
        let ncluster1 = ncluster;
        ncluster = 0;
        // Process the clusters on the current level.
        for icls in 0..ncluster1 {
            let oldfst = old_cluster_range.at(icls * 2) as usize;
            let oldlst = old_cluster_range.at(icls * 2 + 1) as usize;
            if ndepth > 0 {
                // Retrieve the RRR of the cluster from the previous level.
                for i in 0..n {
                    diag.set(i, vec.at(oldfst * n + i));
                }
                for i in 0..n - 1 {
                    diag_off1.set(i, vec.at((oldfst + 1) * n + i));
                }
                sigma = vec.at((oldfst + 2) * n - 1);
            }
            // DL and DLL of the current RRR.
            for j in 0..n - 1 {
                let tmp = diag.at(j) * diag_off1.at(j);
                buf_ld.set(j, tmp);
                buf_lld.set(j, tmp * diag_off1.at(j));
            }
            // `eigh.c:1188-1205` has the `ndepth > 0` limited-bisection block
            // COMMENTED OUT upstream; it is left out here for the same reason
            // it is left out there — including it would change every bit.

            // Process the current node.
            let mut newfst = oldfst;
            for newlst in oldfst + 1..=oldlst {
                if newlst < oldlst && wgap.at(newlst as isize - 1) < MINGAP * buf_w.at(newlst - 1).abs() {
                    continue;
                }

                if newlst - newfst > 1 {
                    // Left- and rightmost eigenvalue of the child to high
                    // precision, so the shift can be as close as possible.
                    dlarrb(n, diag, buf_lld, newfst, newfst + 1, rqtol, rqtol, buf_w, wgap, werr, n as i32);
                    dlarrb(n, diag, buf_lld, newlst - 1, newlst, rqtol, rqtol, buf_w, wgap, werr, n as i32);

                    // RRR of the child cluster; the new RRR is stored in `vec`.
                    let (tau, iinfo) = dlarrf(
                        n,
                        diag,
                        diag_off1,
                        buf_ld,
                        newfst,
                        buf_w,
                        wgap,
                        werr,
                        wgap.at(newfst as isize - 1),
                        vec.off(newfst * n),
                        vec.off((newfst + 1) * n),
                    );
                    if iinfo != 0 {
                        return -2;
                    }
                    // A new RRR was found: update and store the shift.
                    vec.set((newfst + 2) * n - 1, sigma + tau);

                    for k in newfst..newlst {
                        let mut fudge = eps * 3. * buf_w.at(k).abs();
                        buf_w.set(k, buf_w.at(k) - tau);
                        fudge += eps * 4. * buf_w.at(k).abs();
                        // Fudge errors. Gaps are deliberately not fudged.
                        werr.set(k, werr.at(k) + fudge);
                    }
                    new_cluster_range.set(ncluster * 2, newfst as i32);
                    new_cluster_range.set(ncluster * 2 + 1, newlst as i32);
                    ncluster += 1;
                } else {
                    // Eigenvector of a singleton.
                    let mut lambda = buf_w.at(newfst);
                    let mut left = lambda - werr.at(newfst);
                    let mut right = lambda + werr.at(newfst);
                    let gap = wgap.at(newfst as isize);
                    let gaptol = if newfst == 0 || newfst + 1 == n { 0.0 } else { gap * eps };
                    let savgap = gap;
                    let resid_tol = tol * gap;
                    let rqcorr_tol = rqtol * lambda.abs();

                    let mut needbs = false;
                    let mut bstres = 1e307f64;
                    let mut bstw = 0.0f64;
                    twist_indices.set(newfst, -1);
                    for _iter in 0..MAXRQITER {
                        let mut ti = twist_indices.at(newfst);
                        let (negcnt, resid, rqcorr) = dlar1v(
                            n,
                            lambda,
                            diag,
                            diag_off1,
                            buf_ld,
                            buf_lld,
                            gaptol,
                            vec.off(newfst * n),
                            &mut ti,
                            buf_wrk,
                        );
                        twist_indices.set(newfst, ti);

                        // Keep track of the best lambda.
                        if resid < bstres {
                            bstres = resid;
                            bstw = lambda;
                        }

                        // Convergence test for the Rayleigh-Quotient iteration.
                        if resid < resid_tol || rqcorr.abs() < rqcorr_tol {
                            break;
                        }

                        // Only use the RQCORR if it improves the iterate.
                        if lambda + rqcorr > right || lambda + rqcorr < left {
                            needbs = true;
                            break;
                        }

                        // The RQCORR update must not move the eigenvalue
                        // toward a neighbour.
                        if (newfst as i32) < negcnt {
                            // The wanted eigenvalue lies to the left.
                            if rqcorr > 0.0 {
                                needbs = true;
                                break;
                            }
                            right = lambda;
                        } else {
                            // The wanted eigenvalue lies to the right.
                            if rqcorr < 0.0 {
                                needbs = true;
                                break;
                            }
                            left = lambda;
                        }

                        buf_w.set(newfst, (right + left) * 0.5);
                        lambda += rqcorr;
                        werr.set(newfst, (right - left) * 0.5);

                        if right - left < rqcorr_tol {
                            if bstres < resid {
                                lambda = bstw;
                                let mut ti = twist_indices.at(newfst);
                                dlar1v(
                                    n,
                                    lambda,
                                    diag,
                                    diag_off1,
                                    buf_ld,
                                    buf_lld,
                                    gaptol,
                                    vec.off(newfst * n),
                                    &mut ti,
                                    buf_wrk,
                                );
                                twist_indices.set(newfst, ti);
                            }
                            break;
                        }
                    }

                    if needbs {
                        dlarrb(
                            n,
                            diag,
                            buf_lld,
                            newfst,
                            newfst + 1,
                            0.0,
                            eps * 2.,
                            buf_w,
                            wgap,
                            werr,
                            twist_indices.at(newfst) + 1,
                        );
                        lambda = buf_w.at(newfst);

                        // Reset the twist index so the true MINGMA is
                        // recomputed from an inaccurate LAMBDA.
                        twist_indices.set(newfst, -1);
                        let mut ti = -1i32;
                        dlar1v(
                            n,
                            lambda,
                            diag,
                            diag_off1,
                            buf_ld,
                            buf_lld,
                            gaptol,
                            vec.off(newfst * n),
                            &mut ti,
                            buf_wrk,
                        );
                        twist_indices.set(newfst, ti);
                    }

                    w.set(newfst, lambda + sigma);

                    // Recompute the gaps, allowing them only to grow.
                    if newfst > 0 {
                        wgap.set(
                            newfst as isize - 1,
                            cmax(
                                wgap.at(newfst as isize - 1),
                                w.at(newfst) - werr.at(newfst) - w.at(newfst - 1) - werr.at(newfst - 1),
                            ),
                        );
                    }
                    if newfst < n - 1 {
                        wgap.set(
                            newfst as isize,
                            cmax(savgap, w.at(newfst + 1) - werr.at(newfst + 1) - w.at(newfst) - werr.at(newfst)),
                        );
                    }

                    idone += 1;
                }

                newfst = newlst;
            }
        }

        std::mem::swap(&mut old_cluster_range, &mut new_cluster_range);
    }
    if idone < n {
        return -2;
    }
    0
}

/// `_dlaev2( eig, vec, diag, diag_off1 )` (`eigh.c:1381`): libcint's 2x2
/// closed form.
///
/// **Not** `dlaev2.f`: `RT = SQRT(TB*TB + DF*DF)` instead of the scaled
/// `ADF*SQRT(1+(AB/ADF)**2)` form, and `RT2 = (a*c - b*b)/RT1` instead of the
/// reference's `(ACMX/RT1)*ACMN - (B/RT1)*B`. Both differ in the last places
/// and the first can overflow where the reference does not; that is libcint's
/// choice, and this module reproduces it.
unsafe fn dlaev2(eig: D, vec: D, diag: D, diag_off1: D) -> i32 {
    let a = diag.at(0);
    let b = diag_off1.at(0);
    let c = diag.at(1);

    let sm = a + c;
    let df = a - c;
    let tb = b + b;

    let rt = (tb * tb + df * df).sqrt();

    let rt1;
    let rt2;
    let sgn1: i32;
    if sm > 0.0 {
        rt1 = (sm + rt) * 0.5;
        sgn1 = 1;
        rt2 = (a * c - b * b) / rt1;
    } else if sm < 0.0 {
        rt1 = (sm - rt) * 0.5;
        sgn1 = -1;
        rt2 = (a * c - b * b) / rt1;
    } else {
        rt1 = rt * 0.5;
        rt2 = rt * -0.5;
        sgn1 = 1;
    }

    // The eigenvector.
    let cs;
    let sgn2: i32;
    if df >= 0.0 {
        cs = df + rt;
        sgn2 = 1;
    } else {
        cs = df - rt;
        sgn2 = -1;
    }

    let mut cs1;
    let mut sn1;
    if cs.abs() > tb.abs() {
        let ct = -tb / cs;
        sn1 = 1. / (ct * ct + 1.).sqrt();
        cs1 = ct * sn1;
    } else if b == 0.0 {
        cs1 = 1.0;
        sn1 = 0.0;
    } else {
        let tn = -cs / tb;
        cs1 = 1. / (tn * tn + 1.).sqrt();
        sn1 = tn * cs1;
    }

    if sgn1 == sgn2 {
        let tmp = cs1;
        cs1 = -sn1;
        sn1 = tmp;
    }

    eig.set(0, rt2);
    eig.set(1, rt1);
    vec.set(0, -sn1);
    vec.set(1, cs1);
    vec.set(2, cs1);
    vec.set(3, sn1);
    0
}

/// `(work, iwork)` lengths for [`cint_stemr`], from `eigh.c:1462`:
/// `double work[MXRYSROOTS*9 + 1]`, `int iwork[MXRYSROOTS*5]`, sized at `n`.
pub fn cint_stemr_work_len(n: usize) -> (usize, usize) {
    (n * 9 + 1, n * 5)
}

/// `_CINTdiagonalize( n, diag, diag_off1, eig, vec )` (`eigh.c:1450`):
/// eigenvalues and eigenvectors of a symmetric tridiagonal, **bit-exact to
/// libcint 6.1.3**, in libcint's own Wheeler convention.
///
/// * `diag[0..n]` is the diagonal and is **overwritten**.
/// * `diag_off1[0..n]` is the off-diagonal in `[0..n-1]`; `[n-1]` is scratch,
///   and is also overwritten.
/// * `eig[0..n]` receives the ascending eigenvalues.
/// * `vec[0..n*n]` receives the eigenvectors: `vec[i*n + j]` is the `j`-th
///   component of the `i`-th. (Wheeler reads `vec[i*n]`.)
/// * `work` / `iwork` are caller-supplied, at least
///   [`cint_stemr_work_len`]; the driver allocates nothing.
///
/// Returns libcint's `INFO` (0 on success).
///
/// # Panics
/// If any slice is shorter than its documented length, or `n > MXRYSROOTS`.
pub fn cint_stemr(
    n: usize,
    diag: &mut [f64],
    diag_off1: &mut [f64],
    eig: &mut [f64],
    vec: &mut [f64],
    work: &mut [f64],
    iwork: &mut [i32],
) -> i32 {
    if n == 0 {
        return 0;
    }
    assert!(n <= MXRYSROOTS, "cint_stemr: n = {n} > MXRYSROOTS = {MXRYSROOTS}");
    let (lw, liw) = cint_stemr_work_len(n);
    assert!(diag.len() >= n, "cint_stemr: diag needs {n} entries");
    assert!(diag_off1.len() >= n, "cint_stemr: diag_off1 needs {n} entries");
    assert!(eig.len() >= n, "cint_stemr: eig needs {n} entries");
    assert!(vec.len() >= n * n, "cint_stemr: vec needs {} entries", n * n);
    assert!(work.len() >= lw, "cint_stemr: work needs {lw} entries");
    assert!(iwork.len() >= liw, "cint_stemr: iwork needs {liw} entries");

    if n == 1 {
        eig[0] = diag[0];
        vec[0] = 1.0;
        return 0;
    }
    // SAFETY: every length is asserted above; the six slices are distinct.
    unsafe {
        let d = D::new(diag);
        let e = D::new(diag_off1);
        let ei = D::new(eig);
        let v = D::new(vec);
        let wk = D::new(work);
        let iw = J::new(iwork);
        if n == 2 {
            return dlaev2(ei, v, d, e);
        }
        // `eigh.c:1463-1466`: buf_err = work + n, buf_gp = work + 2n + 1,
        // buf_wrk = work + 3n + 1. `wgap` is `buf_gp`, and `Wgap` is based one
        // slot earlier so `wgap[-1]` (= work[2n]) is addressable.
        let buf_err = wk.off(n);
        let wgap = Wgap(wk.off(n * 2));
        let buf_wrk = wk.off(n * 3 + 1);

        let mut info = compute_eigenvalues(n, d, e, ei, buf_err, wgap, buf_wrk);
        if info == 0 {
            info = compute_eigenvectors(n, d, e, ei, buf_err, wgap, v, buf_wrk, iw);
        }
        info
    }
}

/// The `static` routines of `eigh.c`'s `#else` branch, exposed for the
/// component-level oracle tests (`tests/cint_stemr_bitexact.rs`) and for the
/// device parity gate (`tests/device_cint_stemr_cpu.rs`), which pins each
/// `#[cube]` leaf against the routine it transcribes.
///
/// Not part of the supported API: every function is `unsafe` and takes the
/// C's raw shape contract. The oracle is **libcint 6.1.3**, not
/// Reference-LAPACK.
#[doc(hidden)]
pub(crate) mod testing {
    use super::*;

    /// `_dlarrk`. Returns `(w, werr, info)`.
    pub(crate) unsafe fn dlarrk(n: usize, iw: i32, gl: f64, gu: f64, diag: D, e2: D, reltol: f64) -> (f64, f64, i32) {
        super::dlarrk(n, iw, gl, gu, diag, e2, reltol)
    }

    /// `_dlarrc`. Returns `(lcnt, rcnt)`.
    pub(crate) unsafe fn dlarrc(n: usize, vl: f64, vu: f64, diag: D, e2: D) -> (i32, i32) {
        super::dlarrc(n, vl, vu, diag, e2)
    }

    /// `_dlaneg`.
    pub(crate) unsafe fn dlaneg(n: usize, diag: D, lld: D, sigma: f64, twist: i32) -> i32 {
        super::dlaneg(n, diag, lld, sigma, twist)
    }

    /// `_dlarrb`; `wgap_m1` is the `-1`-based `wgap` view.
    pub(crate) unsafe fn dlarrb(
        n: usize,
        diag: D,
        lld: D,
        ifirst: usize,
        ilast: usize,
        rtol1: f64,
        rtol2: f64,
        w: D,
        wgap_m1: D,
        werr: D,
        twist: i32,
    ) {
        super::dlarrb(n, diag, lld, ifirst, ilast, rtol1, rtol2, w, Wgap(wgap_m1), werr, twist);
    }

    /// `_dlarrf`. Returns `(sigma, info)`.
    pub(crate) unsafe fn dlarrf(
        n: usize,
        diag: D,
        offd: D,
        ld: D,
        clstrt: usize,
        w: D,
        wgap_m1: D,
        werr: D,
        clgapl: f64,
        dplus: D,
        lplus: D,
    ) -> (f64, i32) {
        super::dlarrf(n, diag, offd, ld, clstrt, w, Wgap(wgap_m1), werr, clgapl, dplus, lplus)
    }

    /// `_dlar1v`. Returns `(negcnt, twist, resid, rqcorr)`.
    pub(crate) unsafe fn dlar1v(
        n: usize,
        lambda: f64,
        diag: D,
        offd: D,
        ld: D,
        lld: D,
        vec: D,
        twist_in: i32,
        work: D,
    ) -> (i32, i32, f64, f64) {
        let mut twist = twist_in;
        let (negcnt, resid, rqcorr) = super::dlar1v(n, lambda, diag, offd, ld, lld, 0.0, vec, &mut twist, work);
        (negcnt, twist, resid, rqcorr)
    }

    /// `_dlaev2`.
    pub(crate) unsafe fn dlaev2(eig: D, vec: D, diag: D, offd: D) -> i32 {
        super::dlaev2(eig, vec, diag, offd)
    }

    /// `_dlasq4`. Returns the updated `TAU`.
    pub(crate) unsafe fn dlasq4(
        work: D,
        qp: usize,
        q1p: usize,
        ep: usize,
        e1p: usize,
        i0: usize,
        n0: usize,
        n0init: usize,
        dmin: [f64; 3],
        dn: [f64; 3],
        tau_in: f64,
    ) -> f64 {
        let mut tau = tau_in;
        super::dlasq4(i0, n0, n0init, work.off(qp), work.off(q1p), work.off(ep), work.off(e1p), &dmin, &dn, &mut tau);
        tau
    }

    /// `_dlasq5`. Returns `(dmin, dn)`.
    pub(crate) unsafe fn dlasq5(
        work: D,
        qp: usize,
        q1p: usize,
        ep: usize,
        e1p: usize,
        i0: usize,
        n0: usize,
        tau: f64,
        tol: f64,
    ) -> ([f64; 3], [f64; 3]) {
        let mut dmin = [0.0f64; 3];
        let mut dn = [0.0f64; 3];
        super::dlasq5(i0, n0, work.off(qp), work.off(q1p), work.off(ep), work.off(e1p), tau, tol, &mut dmin, &mut dn);
        (dmin, dn)
    }
}
