//! The dqds eigenvalue engine (Reference-LAPACK v3.12.1).
//!
//! Transcribed from `SRC/dlasq2.f`, `SRC/dlasq3.f`, `SRC/dlasq4.f`,
//! `SRC/dlasq5.f`, `SRC/dlasq6.f`.
//!
//! `Z` is the reference's `4*N` qd workspace, indexed 1-based throughout
//! (`z.at(k - 1)` is Fortran `Z( K )`). `IEEE` is `ILAENV( 10, ... ) == 1`,
//! which is `.TRUE.` on every target this crate builds for; the non-IEEE
//! branches are transcribed anyway so the two read alike and a future
//! `IEEE = .FALSE.` gate has something to compare against.
//!
//! **EMIN note.** `MIN_EXPR` is commutative in GIMPLE and gcc canonicalises
//! its operand order, so a Fortran `MIN( A, B )` does not always become
//! `minsd src=B, dst=A`. In `dlasq5.f` every `EMIN = MIN( Z( J4 ), EMIN )`
//! is emitted as `minsd %xmm(Z), %xmm(EMIN)`, i.e. `EMIN < Z ? EMIN : Z` —
//! the *opposite* of the source order, which matters exactly when one side is
//! `NaN` or a signed zero, both of which the dqds fixtures produce. The
//! transcription follows the emitted order ([`fmin`] is `a < b ? a : b`), and
//! `tests/ref_mrrr_components.rs` pins it: the `kind = 1`, `TAU = 1` fixture
//! drives `Inf`, `NaN` and `-0.0` through this very `MIN`.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::aux::{PREC, SAFMIN, dlasrt_dec, fmax, fmin};
use crate::ptr::D;

const ZERO: f64 = 0.0;
const QURTR: f64 = 0.250;
const HALF: f64 = 0.5;
const ONE: f64 = 1.0;
const TWO: f64 = 2.0;
const FOUR: f64 = 4.0;
const HUNDRD: f64 = 100.0;
const CBIAS: f64 = 1.50;

/// Mutable state `DLASQ3` threads through `DLASQ4`/`DLASQ5`/`DLASQ6`.
///
/// The reference passes these as individual `INOUT` scalars; grouping them
/// keeps the argument lists readable without changing any value.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Dq {
    pub dmin: f64,
    pub dmin1: f64,
    pub dmin2: f64,
    pub dn: f64,
    pub dn1: f64,
    pub dn2: f64,
    pub tau: f64,
    pub g: f64,
    pub ttype: i32,
    pub sigma: f64,
    pub desig: f64,
    pub qmax: f64,
    pub nfail: i32,
    pub iter: i32,
    pub ndiv: i32,
}

/// `DLASQ6( I0, N0, Z, PP, DMIN, DMIN1, DMIN2, DN, DNM1, DNM2 )`:
/// one dqd step (shift zero), in IEEE-safe form.
///
/// # Safety
/// `z` has at least `4 * n0` entries.
pub(crate) unsafe fn dlasq6(i0: usize, n0: usize, z: D, pp: usize, s: &mut Dq) {
    if n0 < i0 + 2 {
        return; // ( N0-I0-1 ).LE.0
    }
    let safmin = SAFMIN;
    let zz = |k: usize| z.at(k - 1);
    let sz = |k: usize, v: f64| z.set(k - 1, v);

    let j4 = 4 * i0 + pp - 3;
    let mut emin = zz(j4 + 4);
    let mut d = zz(j4);
    s.dmin = d;

    if pp == 0 {
        let mut j4 = 4 * i0;
        while j4 <= 4 * (n0 - 3) {
            sz(j4 - 2, d + zz(j4 - 1));
            if zz(j4 - 2) == ZERO {
                sz(j4, ZERO);
                d = zz(j4 + 1);
                s.dmin = d;
                emin = ZERO;
            } else if safmin * zz(j4 + 1) < zz(j4 - 2) && safmin * zz(j4 - 2) < zz(j4 + 1) {
                let temp = zz(j4 + 1) / zz(j4 - 2);
                sz(j4, zz(j4 - 1) * temp);
                d = d * temp;
            } else {
                sz(j4, zz(j4 + 1) * (zz(j4 - 1) / zz(j4 - 2)));
                d = zz(j4 + 1) * (d / zz(j4 - 2));
            }
            s.dmin = fmin(s.dmin, d);
            emin = fmin(emin, zz(j4));
            j4 += 4;
        }
    } else {
        let mut j4 = 4 * i0;
        while j4 <= 4 * (n0 - 3) {
            sz(j4 - 3, d + zz(j4));
            if zz(j4 - 3) == ZERO {
                sz(j4 - 1, ZERO);
                d = zz(j4 + 2);
                s.dmin = d;
                emin = ZERO;
            } else if safmin * zz(j4 + 2) < zz(j4 - 3) && safmin * zz(j4 - 3) < zz(j4 + 2) {
                let temp = zz(j4 + 2) / zz(j4 - 3);
                sz(j4 - 1, zz(j4) * temp);
                d = d * temp;
            } else {
                sz(j4 - 1, zz(j4 + 2) * (zz(j4) / zz(j4 - 3)));
                d = zz(j4 + 2) * (d / zz(j4 - 3));
            }
            s.dmin = fmin(s.dmin, d);
            emin = fmin(emin, zz(j4 - 1));
            j4 += 4;
        }
    }

    // Unroll last two steps.
    s.dn2 = d;
    s.dmin2 = s.dmin;
    let mut j4 = 4 * (n0 - 2) - pp;
    let mut j4p2 = j4 + 2 * pp - 1;
    sz(j4 - 2, s.dn2 + zz(j4p2));
    if zz(j4 - 2) == ZERO {
        sz(j4, ZERO);
        s.dn1 = zz(j4p2 + 2);
        s.dmin = s.dn1;
        emin = ZERO;
    } else if safmin * zz(j4p2 + 2) < zz(j4 - 2) && safmin * zz(j4 - 2) < zz(j4p2 + 2) {
        let temp = zz(j4p2 + 2) / zz(j4 - 2);
        sz(j4, zz(j4p2) * temp);
        s.dn1 = s.dn2 * temp;
    } else {
        sz(j4, zz(j4p2 + 2) * (zz(j4p2) / zz(j4 - 2)));
        s.dn1 = zz(j4p2 + 2) * (s.dn2 / zz(j4 - 2));
    }
    s.dmin = fmin(s.dmin, s.dn1);

    s.dmin1 = s.dmin;
    j4 += 4;
    j4p2 = j4 + 2 * pp - 1;
    sz(j4 - 2, s.dn1 + zz(j4p2));
    if zz(j4 - 2) == ZERO {
        sz(j4, ZERO);
        s.dn = zz(j4p2 + 2);
        s.dmin = s.dn;
        emin = ZERO;
    } else if safmin * zz(j4p2 + 2) < zz(j4 - 2) && safmin * zz(j4 - 2) < zz(j4p2 + 2) {
        let temp = zz(j4p2 + 2) / zz(j4 - 2);
        sz(j4, zz(j4p2) * temp);
        s.dn = s.dn1 * temp;
    } else {
        sz(j4, zz(j4p2 + 2) * (zz(j4p2) / zz(j4 - 2)));
        s.dn = zz(j4p2 + 2) * (s.dn1 / zz(j4 - 2));
    }
    s.dmin = fmin(s.dmin, s.dn);

    sz(j4 + 2, s.dn);
    sz(4 * n0 - pp, emin);
}

/// `DLASQ5( I0, N0, Z, PP, TAU, SIGMA, DMIN, DMIN1, DMIN2, DN, DNM1, DNM2,
/// IEEE, EPS )`: one dqds step with shift `TAU`.
///
/// The reference's non-IEEE branch `RETURN`s early on a negative `D`, leaving
/// `DN`/`Z( J4+2 )`/`Z( 4*N0-PP )` unwritten; that is reproduced exactly.
///
/// # Safety
/// `z` has at least `4 * n0` entries.
pub(crate) unsafe fn dlasq5(i0: usize, n0: usize, z: D, pp: usize, s: &mut Dq, ieee: bool, eps: f64) {
    if n0 < i0 + 2 {
        return; // ( N0-I0-1 ).LE.0
    }
    let zz = |k: usize| z.at(k - 1);
    let sz = |k: usize, v: f64| z.set(k - 1, v);

    let dthresh = eps * (s.sigma + s.tau);
    if s.tau < dthresh * HALF {
        s.tau = ZERO;
    }
    let mut emin;
    let mut d;
    let mut j4;
    let mut j4p2;

    if s.tau != ZERO {
        j4 = 4 * i0 + pp - 3;
        emin = zz(j4 + 4);
        d = zz(j4) - s.tau;
        s.dmin = d;
        s.dmin1 = -zz(j4);

        if ieee {
            if pp == 0 {
                let mut j = 4 * i0;
                while j <= 4 * (n0 - 3) {
                    sz(j - 2, d + zz(j - 1));
                    let temp = zz(j + 1) / zz(j - 2);
                    d = d * temp - s.tau;
                    s.dmin = fmin(s.dmin, d);
                    sz(j, zz(j - 1) * temp);
                    emin = fmin(emin, zz(j)); // MIN( Z(J4), EMIN ): see the EMIN note
                    j += 4;
                }
            } else {
                let mut j = 4 * i0;
                while j <= 4 * (n0 - 3) {
                    sz(j - 3, d + zz(j));
                    let temp = zz(j + 2) / zz(j - 3);
                    d = d * temp - s.tau;
                    s.dmin = fmin(s.dmin, d);
                    sz(j - 1, zz(j) * temp);
                    emin = fmin(emin, zz(j - 1)); // MIN( Z(J4-1), EMIN ): see the EMIN note
                    j += 4;
                }
            }

            // Unroll last two steps.
            s.dn2 = d;
            s.dmin2 = s.dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            sz(j4 - 2, s.dn2 + zz(j4p2));
            sz(j4, zz(j4p2 + 2) * (zz(j4p2) / zz(j4 - 2)));
            s.dn1 = zz(j4p2 + 2) * (s.dn2 / zz(j4 - 2)) - s.tau;
            s.dmin = fmin(s.dmin, s.dn1);

            s.dmin1 = s.dmin;
            j4 += 4;
            j4p2 = j4 + 2 * pp - 1;
            sz(j4 - 2, s.dn1 + zz(j4p2));
            sz(j4, zz(j4p2 + 2) * (zz(j4p2) / zz(j4 - 2)));
            s.dn = zz(j4p2 + 2) * (s.dn1 / zz(j4 - 2)) - s.tau;
            s.dmin = fmin(s.dmin, s.dn);
        } else {
            if pp == 0 {
                let mut j = 4 * i0;
                while j <= 4 * (n0 - 3) {
                    sz(j - 2, d + zz(j - 1));
                    if d < ZERO {
                        return;
                    }
                    sz(j, zz(j + 1) * (zz(j - 1) / zz(j - 2)));
                    d = zz(j + 1) * (d / zz(j - 2)) - s.tau;
                    s.dmin = fmin(s.dmin, d);
                    emin = fmin(emin, zz(j));
                    j += 4;
                }
            } else {
                let mut j = 4 * i0;
                while j <= 4 * (n0 - 3) {
                    sz(j - 3, d + zz(j));
                    if d < ZERO {
                        return;
                    }
                    sz(j - 1, zz(j + 2) * (zz(j) / zz(j - 3)));
                    d = zz(j + 2) * (d / zz(j - 3)) - s.tau;
                    s.dmin = fmin(s.dmin, d);
                    emin = fmin(emin, zz(j - 1));
                    j += 4;
                }
            }

            s.dn2 = d;
            s.dmin2 = s.dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            sz(j4 - 2, s.dn2 + zz(j4p2));
            if s.dn2 < ZERO {
                return;
            }
            sz(j4, zz(j4p2 + 2) * (zz(j4p2) / zz(j4 - 2)));
            s.dn1 = zz(j4p2 + 2) * (s.dn2 / zz(j4 - 2)) - s.tau;
            s.dmin = fmin(s.dmin, s.dn1);

            s.dmin1 = s.dmin;
            j4 += 4;
            j4p2 = j4 + 2 * pp - 1;
            sz(j4 - 2, s.dn1 + zz(j4p2));
            if s.dn1 < ZERO {
                return;
            }
            sz(j4, zz(j4p2 + 2) * (zz(j4p2) / zz(j4 - 2)));
            s.dn = zz(j4p2 + 2) * (s.dn1 / zz(j4 - 2)) - s.tau;
            s.dmin = fmin(s.dmin, s.dn);
        }
    } else {
        // Version that sets d's to zero if they are small enough.
        j4 = 4 * i0 + pp - 3;
        emin = zz(j4 + 4);
        d = zz(j4) - s.tau;
        s.dmin = d;
        s.dmin1 = -zz(j4);
        if ieee {
            if pp == 0 {
                let mut j = 4 * i0;
                while j <= 4 * (n0 - 3) {
                    sz(j - 2, d + zz(j - 1));
                    let temp = zz(j + 1) / zz(j - 2);
                    d = d * temp - s.tau;
                    if d < dthresh {
                        d = ZERO;
                    }
                    s.dmin = fmin(s.dmin, d);
                    sz(j, zz(j - 1) * temp);
                    emin = fmin(emin, zz(j)); // MIN( Z(J4), EMIN ): see the EMIN note
                    j += 4;
                }
            } else {
                let mut j = 4 * i0;
                while j <= 4 * (n0 - 3) {
                    sz(j - 3, d + zz(j));
                    let temp = zz(j + 2) / zz(j - 3);
                    d = d * temp - s.tau;
                    if d < dthresh {
                        d = ZERO;
                    }
                    s.dmin = fmin(s.dmin, d);
                    sz(j - 1, zz(j) * temp);
                    emin = fmin(emin, zz(j - 1)); // MIN( Z(J4-1), EMIN ): see the EMIN note
                    j += 4;
                }
            }

            s.dn2 = d;
            s.dmin2 = s.dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            sz(j4 - 2, s.dn2 + zz(j4p2));
            sz(j4, zz(j4p2 + 2) * (zz(j4p2) / zz(j4 - 2)));
            s.dn1 = zz(j4p2 + 2) * (s.dn2 / zz(j4 - 2)) - s.tau;
            s.dmin = fmin(s.dmin, s.dn1);

            s.dmin1 = s.dmin;
            j4 += 4;
            j4p2 = j4 + 2 * pp - 1;
            sz(j4 - 2, s.dn1 + zz(j4p2));
            sz(j4, zz(j4p2 + 2) * (zz(j4p2) / zz(j4 - 2)));
            s.dn = zz(j4p2 + 2) * (s.dn1 / zz(j4 - 2)) - s.tau;
            s.dmin = fmin(s.dmin, s.dn);
        } else {
            if pp == 0 {
                let mut j = 4 * i0;
                while j <= 4 * (n0 - 3) {
                    sz(j - 2, d + zz(j - 1));
                    if d < ZERO {
                        return;
                    }
                    sz(j, zz(j + 1) * (zz(j - 1) / zz(j - 2)));
                    d = zz(j + 1) * (d / zz(j - 2)) - s.tau;
                    if d < dthresh {
                        d = ZERO;
                    }
                    s.dmin = fmin(s.dmin, d);
                    emin = fmin(emin, zz(j));
                    j += 4;
                }
            } else {
                let mut j = 4 * i0;
                while j <= 4 * (n0 - 3) {
                    sz(j - 3, d + zz(j));
                    if d < ZERO {
                        return;
                    }
                    sz(j - 1, zz(j + 2) * (zz(j) / zz(j - 3)));
                    d = zz(j + 2) * (d / zz(j - 3)) - s.tau;
                    if d < dthresh {
                        d = ZERO;
                    }
                    s.dmin = fmin(s.dmin, d);
                    emin = fmin(emin, zz(j - 1));
                    j += 4;
                }
            }

            s.dn2 = d;
            s.dmin2 = s.dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            sz(j4 - 2, s.dn2 + zz(j4p2));
            if s.dn2 < ZERO {
                return;
            }
            sz(j4, zz(j4p2 + 2) * (zz(j4p2) / zz(j4 - 2)));
            s.dn1 = zz(j4p2 + 2) * (s.dn2 / zz(j4 - 2)) - s.tau;
            s.dmin = fmin(s.dmin, s.dn1);

            s.dmin1 = s.dmin;
            j4 += 4;
            j4p2 = j4 + 2 * pp - 1;
            sz(j4 - 2, s.dn1 + zz(j4p2));
            if s.dn1 < ZERO {
                return;
            }
            sz(j4, zz(j4p2 + 2) * (zz(j4p2) / zz(j4 - 2)));
            s.dn = zz(j4p2 + 2) * (s.dn1 / zz(j4 - 2)) - s.tau;
            s.dmin = fmin(s.dmin, s.dn);
        }
    }

    sz(j4 + 2, s.dn);
    sz(4 * n0 - pp, emin);
}

/// `DLASQ4( I0, N0, Z, PP, N0IN, DMIN, DMIN1, DMIN2, DN, DN1, DN2, TAU,
/// TTYPE, G )`: the shift.
///
/// Several reference branches `RETURN` before `TAU = S`, leaving `TAU` at its
/// incoming value with `TTYPE` already updated; those early exits are kept.
///
/// # Safety
/// `z` has at least `4 * n0 + pp` entries.
pub(crate) unsafe fn dlasq4(i0: usize, n0: usize, z: D, pp: usize, n0in: usize, s: &mut Dq) {
    const CNST1: f64 = 0.5630;
    const CNST2: f64 = 1.010;
    const CNST3: f64 = 1.050;
    const THIRD: f64 = 0.3330;
    let zz = |k: usize| z.at(k - 1);
    // `DO .. , -4` loop bounds go negative for small blocks (`NN-17` is -1 at
    // N0 = 4, PP = 0), so the descending counters below are `isize` and the
    // reads convert back only inside the loop body.
    let zi = |k: isize| z.at((k - 1) as usize);

    // A negative DMIN forces the shift to take that absolute value.
    if s.dmin <= ZERO {
        s.tau = -s.dmin;
        s.ttype = -1;
        return;
    }

    let nn = 4 * n0 + pp;
    let mut sh: f64;
    if n0in == n0 {
        // No eigenvalues deflated.
        if s.dmin == s.dn || s.dmin == s.dn1 {
            let mut b1 = zz(nn - 3).sqrt() * zz(nn - 5).sqrt();
            let mut b2 = zz(nn - 7).sqrt() * zz(nn - 9).sqrt();
            let mut a2 = zz(nn - 7) + zz(nn - 5);

            // Cases 2 and 3.
            if s.dmin == s.dn && s.dmin1 == s.dn1 {
                let gap2 = s.dmin2 - a2 - s.dmin2 * QURTR;
                let gap1 = if gap2 > ZERO && gap2 > b2 { a2 - s.dn - (b2 / gap2) * b2 } else { a2 - s.dn - (b1 + b2) };
                if gap1 > ZERO && gap1 > b1 {
                    sh = fmax(s.dn - (b1 / gap1) * b1, HALF * s.dmin);
                    s.ttype = -2;
                } else {
                    sh = ZERO;
                    if s.dn > b1 {
                        sh = s.dn - b1;
                    }
                    if a2 > (b1 + b2) {
                        sh = fmin(sh, a2 - (b1 + b2));
                    }
                    sh = fmax(sh, THIRD * s.dmin);
                    s.ttype = -3;
                }
            } else {
                // Case 4.
                s.ttype = -4;
                sh = QURTR * s.dmin;
                let gam;
                let np: isize;
                if s.dmin == s.dn {
                    gam = s.dn;
                    a2 = ZERO;
                    if zz(nn - 5) > zz(nn - 7) {
                        return;
                    }
                    b2 = zz(nn - 5) / zz(nn - 7);
                    np = nn as isize - 9;
                } else {
                    let np0 = nn - 2 * pp;
                    gam = s.dn1;
                    if zz(np0 - 4) > zz(np0 - 2) {
                        return;
                    }
                    a2 = zz(np0 - 4) / zz(np0 - 2);
                    if zz(nn - 9) > zz(nn - 11) {
                        return;
                    }
                    b2 = zz(nn - 9) / zz(nn - 11);
                    np = nn as isize - 13;
                }

                // Approximate contribution to norm squared from I < NN-1.
                a2 = a2 + b2;
                // `DO 10 I4 = NP, 4*I0-1+PP, -4`.
                let lo = 4 * i0 as isize - 1 + pp as isize;
                let mut i4 = np;
                while i4 >= lo {
                    if b2 == ZERO {
                        break;
                    }
                    b1 = b2;
                    if zi(i4) > zi(i4 - 2) {
                        return;
                    }
                    b2 = b2 * (zi(i4) / zi(i4 - 2));
                    a2 = a2 + b2;
                    if HUNDRD * fmax(b2, b1) < a2 || CNST1 < a2 {
                        break;
                    }
                    i4 -= 4;
                }
                a2 = CNST3 * a2;

                // Rayleigh quotient residual bound.
                if a2 < CNST1 {
                    sh = gam * (ONE - a2.sqrt()) / (ONE + a2);
                }
            }
        } else if s.dmin == s.dn2 {
            // Case 5.
            s.ttype = -5;
            sh = QURTR * s.dmin;

            // Contribution to norm squared from I > NN-2.
            let np = nn - 2 * pp;
            let mut b1 = zz(np - 2);
            let mut b2 = zz(np - 6);
            let gam = s.dn2;
            if zz(np - 8) > b2 || zz(np - 4) > b1 {
                return;
            }
            let mut a2 = (zz(np - 8) / b2) * (ONE + zz(np - 4) / b1);

            // Approximate contribution to norm squared from I < NN-2.
            if n0 - i0 > 2 {
                b2 = zz(nn - 13) / zz(nn - 15);
                a2 = a2 + b2;
                let lo = 4 * i0 as isize - 1 + pp as isize;
                let mut i4 = nn as isize - 17;
                while i4 >= lo {
                    if b2 == ZERO {
                        break;
                    }
                    b1 = b2;
                    if zi(i4) > zi(i4 - 2) {
                        return;
                    }
                    b2 = b2 * (zi(i4) / zi(i4 - 2));
                    a2 = a2 + b2;
                    if HUNDRD * fmax(b2, b1) < a2 || CNST1 < a2 {
                        break;
                    }
                    i4 -= 4;
                }
                a2 = CNST3 * a2;
            }

            if a2 < CNST1 {
                sh = gam * (ONE - a2.sqrt()) / (ONE + a2);
            }
        } else {
            // Case 6, no information to guide us.
            if s.ttype == -6 {
                s.g = s.g + THIRD * (ONE - s.g);
            } else if s.ttype == -18 {
                s.g = QURTR * THIRD;
            } else {
                s.g = QURTR;
            }
            sh = s.g * s.dmin;
            s.ttype = -6;
        }
    } else if n0in == n0 + 1 {
        // One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN.
        if s.dmin1 == s.dn1 && s.dmin2 == s.dn2 {
            // Cases 7 and 8.
            s.ttype = -7;
            sh = THIRD * s.dmin1;
            if zz(nn - 5) > zz(nn - 7) {
                return;
            }
            let mut b1 = zz(nn - 5) / zz(nn - 7);
            let mut b2 = b1;
            if b2 != ZERO {
                let lo = 4 * i0 as isize - 1 + pp as isize;
                let mut i4 = 4 * n0 as isize - 9 + pp as isize;
                while i4 >= lo {
                    let a2 = b1;
                    if zi(i4) > zi(i4 - 2) {
                        return;
                    }
                    b1 = b1 * (zi(i4) / zi(i4 - 2));
                    b2 = b2 + b1;
                    if HUNDRD * fmax(b1, a2) < b2 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            b2 = (CNST3 * b2).sqrt();
            let a2 = s.dmin1 / (ONE + b2 * b2);
            let gap2 = HALF * s.dmin2 - a2;
            if gap2 > ZERO && gap2 > b2 * a2 {
                sh = fmax(sh, a2 * (ONE - CNST2 * a2 * (b2 / gap2) * b2));
            } else {
                sh = fmax(sh, a2 * (ONE - CNST2 * b2));
                s.ttype = -8;
            }
        } else {
            // Case 9.
            sh = QURTR * s.dmin1;
            if s.dmin1 == s.dn1 {
                sh = HALF * s.dmin1;
            }
            s.ttype = -9;
        }
    } else if n0in == n0 + 2 {
        // Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.
        // Cases 10 and 11.
        if s.dmin2 == s.dn2 && TWO * zz(nn - 5) < zz(nn - 7) {
            s.ttype = -10;
            sh = THIRD * s.dmin2;
            if zz(nn - 5) > zz(nn - 7) {
                return;
            }
            let mut b1 = zz(nn - 5) / zz(nn - 7);
            let mut b2 = b1;
            if b2 != ZERO {
                let lo = 4 * i0 as isize - 1 + pp as isize;
                let mut i4 = 4 * n0 as isize - 9 + pp as isize;
                while i4 >= lo {
                    if zi(i4) > zi(i4 - 2) {
                        return;
                    }
                    b1 = b1 * (zi(i4) / zi(i4 - 2));
                    b2 = b2 + b1;
                    if HUNDRD * b1 < b2 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            b2 = (CNST3 * b2).sqrt();
            let a2 = s.dmin2 / (ONE + b2 * b2);
            let gap2 = zz(nn - 7) + zz(nn - 9) - zz(nn - 11).sqrt() * zz(nn - 9).sqrt() - a2;
            if gap2 > ZERO && gap2 > b2 * a2 {
                sh = fmax(sh, a2 * (ONE - CNST2 * a2 * (b2 / gap2) * b2));
            } else {
                sh = fmax(sh, a2 * (ONE - CNST2 * b2));
            }
        } else {
            sh = QURTR * s.dmin2;
            s.ttype = -11;
        }
    } else if n0in > n0 + 2 {
        // Case 12, more than two eigenvalues deflated. No information.
        sh = ZERO;
        s.ttype = -12;
    } else {
        // Unreachable for the reference's callers; `TAU` keeps its value.
        return;
    }

    s.tau = sh;
}

/// `DLASQ3( I0, N0, Z, PP, DMIN, SIGMA, DESIG, QMAX, NFAIL, ITER, NDIV, IEEE,
/// TTYPE, DMIN1, DMIN2, DN, DN1, DN2, G, TAU )`: one dqds transform.
///
/// `i0`, `n0` and `pp` are `INOUT` in the reference and come back updated.
///
/// # Safety
/// `z` has at least `4 * n0` entries.
pub(crate) unsafe fn dlasq3(i0: usize, n0: &mut usize, z: D, pp: &mut usize, s: &mut Dq, ieee: bool) {
    let zz = |k: usize| z.at(k - 1);
    let sz = |k: usize, v: f64| z.set(k - 1, v);

    let n0in = *n0;
    let eps = PREC;
    let tol = eps * HUNDRD;
    let tol2 = tol * tol;

    // Check for deflation (label 10).
    'deflate: loop {
        if *n0 < i0 {
            return;
        }
        let mut two_eig = false;
        if *n0 == i0 {
            // GO TO 20: one eigenvalue.
            sz(4 * *n0 - 3, zz(4 * *n0 + *pp - 3) + s.sigma);
            *n0 -= 1;
            continue 'deflate;
        }
        let nn = 4 * *n0 + *pp;
        if *n0 == i0 + 1 {
            two_eig = true; // GO TO 40
        }
        if !two_eig {
            // Whether E(N0-1) is negligible, 1 eigenvalue.
            if !(zz(nn - 5) > tol2 * (s.sigma + zz(nn - 3)) && zz(nn - 2 * *pp - 4) > tol2 * zz(nn - 7)) {
                // GO TO 20
                sz(4 * *n0 - 3, zz(4 * *n0 + *pp - 3) + s.sigma);
                *n0 -= 1;
                continue 'deflate;
            }
            // Label 30: whether E(N0-2) is negligible, 2 eigenvalues.
            if zz(nn - 9) > tol2 * s.sigma && zz(nn - 2 * *pp - 8) > tol2 * zz(nn - 11) {
                break 'deflate; // GO TO 50
            }
            two_eig = true;
        }
        if two_eig {
            // Label 40.
            if zz(nn - 3) > zz(nn - 7) {
                let sw = zz(nn - 3);
                sz(nn - 3, zz(nn - 7));
                sz(nn - 7, sw);
            }
            let mut t = HALF * ((zz(nn - 7) - zz(nn - 3)) + zz(nn - 5));
            if zz(nn - 5) > zz(nn - 3) * tol2 && t != ZERO {
                let mut sv = zz(nn - 3) * (zz(nn - 5) / t);
                if sv <= t {
                    sv = zz(nn - 3) * (zz(nn - 5) / (t * (ONE + (ONE + sv / t).sqrt())));
                } else {
                    sv = zz(nn - 3) * (zz(nn - 5) / (t + t.sqrt() * (t + sv).sqrt()));
                }
                t = zz(nn - 7) + (sv + zz(nn - 5));
                sz(nn - 3, zz(nn - 3) * (zz(nn - 7) / t));
                sz(nn - 7, t);
            }
            sz(4 * *n0 - 7, zz(nn - 7) + s.sigma);
            sz(4 * *n0 - 3, zz(nn - 3) + s.sigma);
            *n0 -= 2;
            continue 'deflate;
        }
    }

    // Label 50.
    if *pp == 2 {
        *pp = 0;
    }

    // Reverse the qd-array, if warranted.
    if s.dmin <= ZERO || *n0 < n0in {
        if CBIAS * zz(4 * i0 + *pp - 3) < zz(4 * *n0 + *pp - 3) {
            let ipn4 = 4 * (i0 + *n0);
            let mut j4 = 4 * i0;
            while j4 <= 2 * (i0 + *n0 - 1) {
                let temp = zz(j4 - 3);
                sz(j4 - 3, zz(ipn4 - j4 - 3));
                sz(ipn4 - j4 - 3, temp);
                let temp = zz(j4 - 2);
                sz(j4 - 2, zz(ipn4 - j4 - 2));
                sz(ipn4 - j4 - 2, temp);
                let temp = zz(j4 - 1);
                sz(j4 - 1, zz(ipn4 - j4 - 5));
                sz(ipn4 - j4 - 5, temp);
                let temp = zz(j4);
                sz(j4, zz(ipn4 - j4 - 4));
                sz(ipn4 - j4 - 4, temp);
                j4 += 4;
            }
            if *n0 - i0 <= 4 {
                sz(4 * *n0 + *pp - 1, zz(4 * i0 + *pp - 1));
                sz(4 * *n0 - *pp, zz(4 * i0 - *pp));
            }
            s.dmin2 = fmin(s.dmin2, zz(4 * *n0 + *pp - 1));
            sz(4 * *n0 + *pp - 1, fmin(fmin(zz(4 * *n0 + *pp - 1), zz(4 * i0 + *pp - 1)), zz(4 * i0 + *pp + 3)));
            sz(4 * *n0 - *pp, fmin(fmin(zz(4 * *n0 - *pp), zz(4 * i0 - *pp)), zz(4 * i0 - *pp + 4)));
            s.qmax = fmax(fmax(s.qmax, zz(4 * i0 + *pp - 3)), zz(4 * i0 + *pp + 1));
            s.dmin = -ZERO;
        }
    }

    // Choose a shift.
    dlasq4(i0, *n0, z, *pp, n0in, s);

    // Call dqds until DMIN > 0 (label 70).
    let mut underflow = false;
    loop {
        dlasq5(i0, *n0, z, *pp, s, ieee, eps);

        s.ndiv += (*n0 - i0 + 2) as i32;
        s.iter += 1;

        if s.dmin >= ZERO && s.dmin1 >= ZERO {
            break; // Success (GO TO 90).
        } else if s.dmin < ZERO
            && s.dmin1 > ZERO
            && zz(4 * (*n0 - 1) - *pp) < tol * (s.sigma + s.dn1)
            && s.dn.abs() < tol * s.sigma
        {
            // Convergence hidden by negative DN.
            sz(4 * (*n0 - 1) - *pp + 2, ZERO);
            s.dmin = ZERO;
            break; // GO TO 90
        } else if s.dmin < ZERO {
            // TAU too big. Select new TAU and try again.
            s.nfail += 1;
            if s.ttype < -22 {
                // Failed twice. Play it safe.
                s.tau = ZERO;
            } else if s.dmin1 > ZERO {
                // Late failure. Gives excellent shift.
                s.tau = (s.tau + s.dmin) * (ONE - TWO * eps);
                s.ttype -= 11;
            } else {
                // Early failure. Divide by 4.
                s.tau = QURTR * s.tau;
                s.ttype -= 12;
            }
            continue; // GO TO 70
        } else if s.dmin.is_nan() {
            if s.tau == ZERO {
                underflow = true;
                break; // GO TO 80
            } else {
                s.tau = ZERO;
                continue; // GO TO 70
            }
        } else {
            // Possible underflow. Play it safe.
            underflow = true;
            break; // GO TO 80
        }
    }

    if underflow {
        // Label 80: risk of underflow.
        dlasq6(i0, *n0, z, *pp, s);
        s.ndiv += (*n0 - i0 + 2) as i32;
        s.iter += 1;
        s.tau = ZERO;
    }

    // Label 90.
    let t;
    if s.tau < s.sigma {
        s.desig = s.desig + s.tau;
        t = s.sigma + s.desig;
        s.desig = s.desig - (t - s.sigma);
    } else {
        t = s.sigma + s.tau;
        s.desig = s.sigma - (t - s.tau) + s.desig;
    }
    s.sigma = t;
}

/// `DLASQ2( N, Z, INFO )`: all eigenvalues of the symmetric positive definite
/// tridiagonal with `Z( 2*I-1 )` the diagonal and `Z( 2*I )` the off-diagonal
/// squares, in decreasing order in `Z( 1..N )`.
///
/// `ieee` is `ILAENV( 10, 'DLASQ2', ... ) == 1`.
///
/// # Safety
/// `z` has at least `4 * n` entries.
pub(crate) unsafe fn dlasq2(n: usize, z: D, ieee: bool) -> i32 {
    let zz = |k: usize| z.at(k - 1);
    let sz = |k: usize, v: f64| z.set(k - 1, v);

    let mut info: i32 = 0;
    let eps = PREC;
    let safmin = SAFMIN;
    let tol = eps * HUNDRD;
    let tol2 = tol * tol;

    if n == 0 {
        return info;
    } else if n == 1 {
        // 1-by-1 case.
        if zz(1) < ZERO {
            info = -201;
        }
        return info;
    } else if n == 2 {
        // 2-by-2 case.
        if zz(1) < ZERO {
            return -201;
        } else if zz(2) < ZERO {
            return -202;
        } else if zz(3) < ZERO {
            return -203;
        } else if zz(3) > zz(1) {
            let d = zz(3);
            sz(3, zz(1));
            sz(1, d);
        }
        sz(5, zz(1) + zz(2) + zz(3));
        if zz(2) > zz(3) * tol2 {
            let mut t = HALF * ((zz(1) - zz(3)) + zz(2));
            let mut s = zz(3) * (zz(2) / t);
            if s <= t {
                s = zz(3) * (zz(2) / (t * (ONE + (ONE + s / t).sqrt())));
            } else {
                s = zz(3) * (zz(2) / (t + t.sqrt() * (t + s).sqrt()));
            }
            t = zz(1) + (s + zz(2));
            sz(3, zz(3) * (zz(1) / t));
            sz(1, t);
        }
        sz(2, zz(3));
        sz(6, zz(2) + zz(1));
        return info;
    }

    // Check for negative data and compute sums of q's and e's.
    sz(2 * n, ZERO);
    let mut emin = zz(2);
    let mut qmax = ZERO;
    let mut zmax = ZERO;
    let mut d = ZERO;
    let mut e = ZERO;

    let mut k = 1usize;
    while k <= 2 * (n - 1) {
        if zz(k) < ZERO {
            return -((200 + k) as i32);
        } else if zz(k + 1) < ZERO {
            return -((200 + k + 1) as i32);
        }
        d = d + zz(k);
        e = e + zz(k + 1);
        qmax = fmax(qmax, zz(k));
        emin = fmin(emin, zz(k + 1));
        zmax = fmax(fmax(qmax, zmax), zz(k + 1));
        k += 2;
    }
    if zz(2 * n - 1) < ZERO {
        return -((200 + 2 * n - 1) as i32);
    }
    d = d + zz(2 * n - 1);
    qmax = fmax(qmax, zz(2 * n - 1));
    let _ = fmax(qmax, zmax); // ZMAX is dead after this point in the reference.

    // Check for diagonality.
    if e == ZERO {
        for k in 2..=n {
            sz(k, zz(2 * k - 1));
        }
        dlasrt_dec(n, z);
        sz(2 * n - 1, d);
        return info;
    }

    let trace = d + e;

    // Check for zero data.
    if trace == ZERO {
        sz(2 * n - 1, ZERO);
        return info;
    }

    // Rearrange data for locality: Z=(q1,qq1,e1,ee1,q2,qq2,e2,ee2,...).
    let mut k = 2 * n;
    while k >= 2 {
        sz(2 * k, ZERO);
        sz(2 * k - 1, zz(k));
        sz(2 * k - 2, ZERO);
        sz(2 * k - 3, zz(k - 1));
        k -= 2;
    }

    let mut i0 = 1usize;
    let mut n0 = n;

    // Reverse the qd-array, if warranted.
    if CBIAS * zz(4 * i0 - 3) < zz(4 * n0 - 3) {
        let ipn4 = 4 * (i0 + n0);
        let mut i4 = 4 * i0;
        while i4 <= 2 * (i0 + n0 - 1) {
            let temp = zz(i4 - 3);
            sz(i4 - 3, zz(ipn4 - i4 - 3));
            sz(ipn4 - i4 - 3, temp);
            let temp = zz(i4 - 1);
            sz(i4 - 1, zz(ipn4 - i4 - 5));
            sz(ipn4 - i4 - 5, temp);
            i4 += 4;
        }
    }

    // Initial split checking via dqd and Li's test.
    let mut pp = 0usize;

    for _k in 1..=2 {
        let mut d = zz(4 * n0 + pp - 3);
        // `DO 50 I4 = 4*(N0-1)+PP, 4*I0+PP, -4`.
        let lo = 4 * i0 + pp;
        let mut i4 = 4 * (n0 - 1) + pp;
        while i4 >= lo {
            if zz(i4 - 1) <= tol2 * d {
                sz(i4 - 1, -ZERO);
                d = zz(i4 - 3);
            } else {
                d = zz(i4 - 3) * (d / (d + zz(i4 - 1)));
            }
            if i4 < 4 {
                break;
            }
            i4 -= 4;
        }

        // dqd maps Z to ZZ plus Li's test.
        let mut emin = zz(4 * i0 + pp + 1);
        let mut d = zz(4 * i0 + pp - 3);
        let mut i4 = 4 * i0 + pp;
        while i4 <= 4 * (n0 - 1) + pp {
            sz(i4 - 2 * pp - 2, d + zz(i4 - 1));
            if zz(i4 - 1) <= tol2 * d {
                sz(i4 - 1, -ZERO);
                sz(i4 - 2 * pp - 2, d);
                sz(i4 - 2 * pp, ZERO);
                d = zz(i4 + 1);
            } else if safmin * zz(i4 + 1) < zz(i4 - 2 * pp - 2) && safmin * zz(i4 - 2 * pp - 2) < zz(i4 + 1) {
                let temp = zz(i4 + 1) / zz(i4 - 2 * pp - 2);
                sz(i4 - 2 * pp, zz(i4 - 1) * temp);
                d = d * temp;
            } else {
                sz(i4 - 2 * pp, zz(i4 + 1) * (zz(i4 - 1) / zz(i4 - 2 * pp - 2)));
                d = zz(i4 + 1) * (d / zz(i4 - 2 * pp - 2));
            }
            emin = fmin(emin, zz(i4 - 2 * pp));
            i4 += 4;
        }
        sz(4 * n0 - pp - 2, d);

        // Now find qmax.
        qmax = zz(4 * i0 - pp - 2);
        let mut i4 = 4 * i0 - pp + 2;
        while i4 <= 4 * n0 - pp - 2 {
            qmax = fmax(qmax, zz(i4));
            i4 += 4;
        }

        // Prepare for the next iteration on K.
        pp = 1 - pp;
    }

    // Initialise variables to pass to DLASQ3.
    let mut s = Dq {
        ttype: 0,
        dmin1: ZERO,
        dmin2: ZERO,
        dn: ZERO,
        dn1: ZERO,
        dn2: ZERO,
        g: ZERO,
        tau: ZERO,
        iter: 2,
        nfail: 0,
        ndiv: 2 * (n0 - i0) as i32,
        qmax,
        ..Dq::default()
    };

    let mut finished = false;
    for _iwhila in 1..=n + 1 {
        if n0 < 1 {
            finished = true;
            break; // GO TO 170
        }

        // E(N0) holds the (negated) SIGMA at which I0:N0 split off.
        s.desig = ZERO;
        s.sigma = if n0 == n { ZERO } else { -zz(4 * n0 - 1) };
        if s.sigma < ZERO {
            return 1;
        }

        // Last unreduced submatrix's top index I0, QMAX and EMIN;
        // Gershgorin-type bound if q's are much greater than e's.
        let mut emax = ZERO;
        let mut emin = if n0 > i0 { zz(4 * n0 - 5).abs() } else { ZERO };
        let mut qmin = zz(4 * n0 - 3);
        s.qmax = qmin;
        let mut i4 = 4 * n0;
        let mut hit = false;
        while i4 >= 8 {
            if zz(i4 - 5) <= ZERO {
                hit = true;
                break; // GO TO 100
            }
            if qmin >= FOUR * emax {
                qmin = fmin(qmin, zz(i4 - 3));
                emax = fmax(emax, zz(i4 - 5));
            }
            s.qmax = fmax(s.qmax, zz(i4 - 7) + zz(i4 - 5));
            emin = fmin(emin, zz(i4 - 5));
            i4 -= 4;
        }
        if !hit {
            i4 = 4;
        }

        // Label 100.
        i0 = i4 / 4;
        pp = 0;

        if n0 > i0 + 1 {
            let mut dee = zz(4 * i0 - 3);
            let mut deemin = dee;
            let mut kmin = i0;
            let mut i4 = 4 * i0 + 1;
            while i4 <= 4 * n0 - 3 {
                dee = zz(i4) * (dee / (dee + zz(i4 - 2)));
                if dee <= deemin {
                    deemin = dee;
                    kmin = (i4 + 3) / 4;
                }
                i4 += 4;
            }
            if (kmin - i0) * 2 < n0 - kmin && deemin <= HALF * zz(4 * n0 - 3) {
                let ipn4 = 4 * (i0 + n0);
                pp = 2;
                let mut i4 = 4 * i0;
                while i4 <= 2 * (i0 + n0 - 1) {
                    let temp = zz(i4 - 3);
                    sz(i4 - 3, zz(ipn4 - i4 - 3));
                    sz(ipn4 - i4 - 3, temp);
                    let temp = zz(i4 - 2);
                    sz(i4 - 2, zz(ipn4 - i4 - 2));
                    sz(ipn4 - i4 - 2, temp);
                    let temp = zz(i4 - 1);
                    sz(i4 - 1, zz(ipn4 - i4 - 5));
                    sz(ipn4 - i4 - 5, temp);
                    let temp = zz(i4);
                    sz(i4, zz(ipn4 - i4 - 4));
                    sz(ipn4 - i4 - 4, temp);
                    i4 += 4;
                }
            }
        }

        // Put -(initial shift) into DMIN.
        s.dmin = -fmax(ZERO, qmin - TWO * qmin.sqrt() * emax.sqrt());

        // I0:N0 is unreduced. PP = 0 ping, 1 pong, 2 = flipped on entry.
        let nbig = 100 * (n0 - i0 + 1);
        let mut submatrix_done = false;
        for _iwhilb in 1..=nbig {
            if i0 > n0 {
                submatrix_done = true;
                break; // GO TO 150
            }

            dlasq3(i0, &mut n0, z, &mut pp, &mut s, ieee);

            pp = 1 - pp;

            // When EMIN is very small check for splits.
            if pp == 0 && n0 >= i0 + 3 {
                if zz(4 * n0) <= tol2 * s.qmax || zz(4 * n0 - 1) <= tol2 * s.sigma {
                    let mut splt = i0 - 1;
                    s.qmax = zz(4 * i0 - 3);
                    let mut emin2 = zz(4 * i0 - 1);
                    let mut oldemn = zz(4 * i0);
                    let mut i4 = 4 * i0;
                    while i4 <= 4 * (n0 - 3) {
                        if zz(i4) <= tol2 * zz(i4 - 3) || zz(i4 - 1) <= tol2 * s.sigma {
                            sz(i4 - 1, -s.sigma);
                            splt = i4 / 4;
                            s.qmax = ZERO;
                            emin2 = zz(i4 + 3);
                            oldemn = zz(i4 + 4);
                        } else {
                            s.qmax = fmax(s.qmax, zz(i4 + 1));
                            emin2 = fmin(emin2, zz(i4 - 1));
                            oldemn = fmin(oldemn, zz(i4));
                        }
                        i4 += 4;
                    }
                    sz(4 * n0 - 1, emin2);
                    sz(4 * n0, oldemn);
                    i0 = splt + 1;
                }
            }
        }

        if submatrix_done {
            continue; // label 150, end IWHILB -> next IWHILA
        }

        // Maximum number of iterations exceeded: restore SIGMA and place the
        // new d's and e's in a qd array (possibly for several blocks).
        info = 2;
        let mut i1 = i0;
        let mut n1;
        loop {
            let mut tempq = zz(4 * i0 - 3);
            sz(4 * i0 - 3, zz(4 * i0 - 3) + s.sigma);
            for k in i0 + 1..=n0 {
                let tempe = zz(4 * k - 5);
                sz(4 * k - 5, zz(4 * k - 5) * (tempq / zz(4 * k - 7)));
                tempq = zz(4 * k - 3);
                sz(4 * k - 3, zz(4 * k - 3) + s.sigma + tempe - zz(4 * k - 5));
            }

            // Previous block, if there is one.
            if i1 > 1 {
                n1 = i1 - 1;
                while i1 >= 2 && zz(4 * i1 - 5) >= ZERO {
                    i1 -= 1;
                }
                s.sigma = -zz(4 * n1 - 1);
            } else {
                break;
            }
        }

        for k in 1..=n {
            sz(2 * k - 1, zz(4 * k - 3));
            // Only the block 1..N0 is unfinished; the rest of the e's must be
            // essentially zero, though other data is sometimes stored there.
            if k < n0 {
                sz(2 * k, zz(4 * k - 1));
            } else {
                sz(2 * k, 0.0);
            }
        }
        return info;
    }

    if !finished {
        return 3;
    }

    // Label 170: move q's to the front.
    for k in 2..=n {
        sz(k, zz(4 * k - 3));
    }

    // Sort and compute sum of eigenvalues.
    dlasrt_dec(n, z);

    let mut e = ZERO;
    for k in (1..=n).rev() {
        e = e + zz(k);
    }

    // Store trace, sum(eigenvalues) and information on performance.
    sz(2 * n + 1, trace);
    sz(2 * n + 2, e);
    sz(2 * n + 3, s.iter as f64);
    sz(2 * n + 4, s.ndiv as f64 / ((n * n) as f64));
    sz(2 * n + 5, HUNDRD * s.nfail as f64 / (s.iter as f64));
    info
}
