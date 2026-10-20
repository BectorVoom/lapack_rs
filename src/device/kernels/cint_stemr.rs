//! **K9, Path C. The oracle of this module is libcint 6.1.3, NOT
//! Reference-LAPACK 3.12.1.**
//!
//! Device transcription of [`crate::cint_stemr`] — libcint 6.1.3
//! `src/eigh.c`'s `#else` branch, SHA-256
//! `8e04d0ccdf735518e6e045632238680ab9e36936c5d1d475ac14b192d6a3d1a3` — kept
//! line for line against the host module so the two can be read side by side.
//! `docs/cintx-lapack-blas-adoption-plan.md` §6.2 sets the three surfaces:
//!
//! 1. [`cint_stemr_dev`] — **production**, a plain `#[cube]` callee with no
//!    `launch`, so a foreign crate's kernel can call it *inside its own
//!    launch*. For an `n <= 13` problem solved once per shell quartet, a
//!    launch plus a blocking read-back is ~10⁵× the arithmetic (plan §6.1),
//!    so the launcher form is the slow shape and this is the one cintx links
//!    against. Every buffer is caller-owned.
//! 2. [`cint_stemr_batch_kernel`] — **standalone**, one *problem* per unit.
//!    `B` problems cost one dispatch and one batched read-back.
//! 3. [`cint_stemr_kernel`] — **parity/debug only**, one problem, used by
//!    `tests/device_cint_stemr_cpu.rs` to pin surface 1 against the host.
//!
//! Surfaces 2 and 3 are one-line wrappers over 1, so there is one body to keep
//! bit-exact, not three.
//!
//! Two representation changes, neither of which is a value change:
//!
//! * **Twist indices are `u32` with [`TWIST_NONE`] for the C's `-1`.**
//!   `#[cube]` has no signed sentinel that survives all three backends
//!   identically; the arithmetic is otherwise the C's.
//! * **Pointer swaps are base-offset swaps.** `_dlasq2` swaps
//!   `qvecp`/`qvec1p`/`evecp`/`evec1p` each step; a `#[cube]` body cannot
//!   swap `&mut Array` references, so the four `u32` base offsets into `work`
//!   are swapped instead. The *base* `qvec` (offset 0) still receives the
//!   converged eigenvalues, exactly as in the C.
//!
//! Every descending reference loop is rewritten as `for t in 0..count { let
//! j = top - t; .. }` over the identical visit order (cubecl-cpu 0.10 cannot
//! lower a decremented counter; `tests/cubecl_cpu_limits.rs`), and every
//! index expression stays in `u32` with the underflow guarded where the C
//! relies on signed comparison.

// `let mut x = zero;` before branch assignment is how a #[cube] kernel
// declares a variable; rustc sees the initial value as unused.
#![allow(unused_assignments)]

use cubecl::prelude::*;

use super::gf::{fmax_c, fmin_c};

/// libcint's `MXRYSROOTS`.
pub const MXRYSROOTS: u32 = 13;

/// The `u32` stand-in for libcint's `twist_index == -1`.
pub const TWIST_NONE: u32 = u32::MAX;

const MAXTRY: u32 = 6;
const MAXRQITER: u32 = 6;
const ITMAX: u32 = 1000;

/// C's `DBL_EPSILON` as an argument, never a literal: Path C passes
/// `f64::EPSILON`, and the same body with Reference-LAPACK's constants is
/// [`super::stemr`].
#[cube]
fn dbl_eps<F: Float>(eps: F) -> F {
    eps
}

// ---------------------------------------------------------------------------
// Buffer layout.
//
// libcint's solver runs out of ONE `double work[MXRYSROOTS*9 + 1]` and ONE
// `int iwork[MXRYSROOTS*5]`, with `werr`, `wgap` and every inner scratch
// pointing into it. A `#[cube]` body cannot hold four `&mut Array` borrows of
// the same buffer, so every helper here takes the single `work` array plus the
// base offsets it needs — which is also closer to the C than four names would
// be. Two small caller-owned arrays carry the values the C returns through
// pointers:
//
// `sc` (>= `SC_LEN` reals)   0..3 dmin, 3..6 dn, 6 tau, 7 w, 8 werr,
//                            9 sigma, 10 resid, 11 rqcorr
// `icnt` (>= `ICNT_LEN` u32) 0 lcnt, 1 rcnt
// ---------------------------------------------------------------------------

/// Length of the caller-owned real scratch every entry point needs.
pub const SC_LEN: u32 = 16;
/// Length of the caller-owned integer scratch.
pub const ICNT_LEN: u32 = 4;

// ---------------------------------------------------------------------------
// libcint's constants, as a caller-owned table.
//
// `F::new` takes an **`f32`**, so `cc[(CC_THIRD) as usize]` is the f32 nearest 0.333
// widened to f64 — about 1e-8 off libcint's `double` 0.333, which shows up
// immediately in `_dlasq4`'s shift. Only dyadic values (`0.25`, `0.5`, `2.0`,
// `4.0`, `8.0`, `64.0`, `100.0`, `1.5`) survive that round trip, and those are
// built from `F::from_int` here anyway. Everything else is passed in through
// `cc`, which is also what the adoption plan §5 requires: "every tolerance is
// an argument, never a literal", so one kernel body can serve two oracles.
// ---------------------------------------------------------------------------

/// Length of the caller-owned constant table.
pub const CC_LEN: u32 = 12;

/// `0.333` (`_dlasq4`'s `dmin * .333`).
pub const CC_THIRD: u32 = 0;
/// `0.563` (`_dlasq4`'s `CNST1`).
pub const CC_CNST1: u32 = 1;
/// `0.01` (`_dlasq4`'s `a2 * .01`).
pub const CC_HUNDREDTH: u32 = 2;
/// `1.05` (`_dlasq4`'s `CNST3`).
pub const CC_CNST3: u32 = 3;
/// `1.01` (`_dlasq4`'s `CNST2`).
pub const CC_CNST2: u32 = 4;
/// `2.1e-8` (`_compute_eigenvalues`'s hard-coded `rtl`).
pub const CC_RTL: u32 = 5;
/// `0.001` (`MINGAP`).
pub const CC_MINGAP: u32 = 6;
/// `1e16` (`FAIL_THRESHOLD`).
pub const CC_FAIL: u32 = 7;
/// `1e307` (`_compute_eigenvectors`'s initial `bstres`).
pub const CC_BIG: u32 = 8;
/// C's `DBL_EPSILON`.
pub const CC_EPS: u32 = 9;

const SC_DMIN: u32 = 0;
const SC_DN: u32 = 3;
const SC_TAU: u32 = 6;
const SC_W: u32 = 7;
const SC_WERR: u32 = 8;
const SC_SIGMA: u32 = 9;
const SC_RESID: u32 = 10;
const SC_RQCORR: u32 = 11;

/// `_dlarrk` (`eigh.c:62`). Writes `w`, `werr` to `sc[SC_W]`, `sc[SC_WERR]`;
/// returns `INFO` (0 converged, 1 = the C's `-1`).
///
/// Note what is **absent** relative to `dlarrk.f`: no `-FUDGE*TWO*PIVMIN`
/// bracket term, no `PIVMIN` pivot floor, a fixed `ITMAX = 1000`, and a
/// `tmp1 <= 0 || tmp1 < reltol*MAX(..)` convergence test. Those are four of
/// the six divergences of the adoption plan §4.2.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_dlarrk_c<F: Float>(
    diag: &Array<F>,
    dbase: u32,
    work: &Array<F>,
    e2base: u32,
    n: u32,
    iw: u32,
    gl: F,
    gu: F,
    reltol: F,
    eps: F,
    sc: &mut Array<F>,
    scb: u32,
) -> u32 {
    let zero = F::from_int(0);
    let two = F::from_int(2);
    let half = F::from_int(1) / two;

    let mut info: u32 = 1; // the C's INFO = -1
    if n > 0 {
        let tnorm = fmax_c::<F>(F::abs(gl), F::abs(gu));
        // `gl - tnorm * 2. * DBL_EPSILON * n`, left to right, `n` last.
        let pert = ((tnorm * two) * eps) * F::cast_from(n);
        let mut left = gl - pert;
        let mut right = gu + pert;

        let itmax = ITMAX;
        let mut it: u32 = 0;
        while it < itmax {
            let tmp1 = F::abs(right - left);
            let mut done: bool = false;
            if tmp1 <= zero {
                done = true;
            }
            if tmp1 < reltol * fmax_c::<F>(F::abs(right), F::abs(left)) {
                done = true;
            }
            if done {
                info = 0;
                it = itmax;
            } else {
                let mid = (left + right) * half;
                let mut negcnt: u32 = 0;
                let mut t = diag[dbase as usize] - mid;
                if t <= zero {
                    negcnt += 1;
                }
                let mut i: u32 = 1;
                while i < n {
                    t = diag[(dbase + i) as usize] - work[(e2base + i - 1u32) as usize] / t - mid;
                    if t <= zero {
                        negcnt += 1;
                    }
                    i += 1;
                }
                if negcnt >= iw {
                    right = mid;
                } else {
                    left = mid;
                }
                it += 1;
            }
        }

        sc[(scb + SC_W) as usize] = (left + right) * half;
        sc[(scb + SC_WERR) as usize] = F::abs(right - left) * half;
    } else {
        info = 0;
        sc[(scb + SC_W) as usize] = zero;
        sc[(scb + SC_WERR) as usize] = zero;
    }
    info
}

/// `_dlarrc` (`eigh.c:118`). Writes `lcnt`, `rcnt` to `icnt[0]`, `icnt[1]`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_dlarrc_c<F: Float>(
    n: u32,
    vl: F,
    vu: F,
    diag: &Array<F>,
    dbase: u32,
    work: &Array<F>,
    e2base: u32,
    icnt: &mut Array<u32>,
    icb: u32,
) {
    let zero = F::from_int(0);
    let mut left_count: u32 = 0;
    let mut right_count: u32 = 0;
    let mut lpivot = diag[dbase as usize] - vl;
    let mut rpivot = diag[dbase as usize] - vu;
    if lpivot <= zero {
        left_count += 1;
    }
    if rpivot <= zero {
        right_count += 1;
    }
    let mut i: u32 = 0;
    while i + 1u32 < n {
        let tmp = work[(e2base + i) as usize];
        lpivot = diag[(dbase + i + 1u32) as usize] - vl - tmp / lpivot;
        rpivot = diag[(dbase + i + 1u32) as usize] - vu - tmp / rpivot;
        if lpivot <= zero {
            left_count += 1;
        }
        if rpivot <= zero {
            right_count += 1;
        }
        i += 1;
    }
    icnt[icb as usize] = left_count;
    icnt[(icb + 1u32) as usize] = right_count;
}

/// `_dlaneg` (`eigh.c:897`): the plain two-sweep form, with neither `dlaneg.f`'s
/// `BLKLEN` blocking nor its NaN reruns. `twist_index >= 1` at every call site.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_dlaneg_c<F: Float>(
    diag: &Array<F>,
    dbase: u32,
    work: &Array<F>,
    lldbase: u32,
    n: u32,
    sigma: F,
    twist_index: u32,
) -> u32 {
    let zero = F::from_int(0);
    let mut negcnt: u32 = 0;

    // I) upper part (ascending already).
    let mut t = -sigma;
    let mut j: u32 = 0;
    while j + 1u32 < twist_index {
        let dplus = diag[(dbase + j) as usize] + t;
        if dplus < zero {
            negcnt += 1;
        }
        t = t / dplus * work[(lldbase + j) as usize] - sigma;
        j += 1;
    }

    // II) lower part. The C counter DECREMENTS
    // (`for (j = n-2; j >= twist_index-1; --j)`); same visit order from an
    // ascending counter, `j = (n-2) - s`.
    let mut p = diag[(dbase + n - 1u32) as usize] - sigma;
    let mut cnt: u32 = 0;
    if n >= twist_index {
        cnt = n - twist_index;
    }
    let mut s: u32 = 0;
    while s < cnt {
        let j = (n - 2u32) - s;
        let dminus = work[(lldbase + j) as usize] + p;
        if dminus < zero {
            negcnt += 1;
        }
        p = p / dminus * diag[(dbase + j) as usize] - sigma;
        s += 1;
    }

    // III) twist index. T was shifted by SIGMA initially.
    if t + sigma + p < zero {
        negcnt += 1;
    }
    negcnt
}

/// `_dlarrb` (`eigh.c:934`).
///
/// `mid` and `width` are C function-scope locals carried across the `i` loop:
/// an `i` that `continue`s leaves them at the previous `i`'s values, which is
/// reproduced here.
///
/// `wgbase` is the **`-1`-based** `wgap` view (`wgap[k]` is
/// `work[wgbase + k + 1]`), because `eigh.c` indexes `wgap` from `-1`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_dlarrb_c<F: Float>(
    diag: &Array<F>,
    dbase: u32,
    work: &mut Array<F>,
    lldbase: u32,
    wbase: u32,
    wgbase: u32,
    ebase: u32,
    n: u32,
    ifirst: u32,
    ilast: u32,
    rtol1: F,
    rtol2: F,
    twist_index: u32,
) {
    let zero = F::from_int(0);
    let two = F::from_int(2);
    let half = F::from_int(1) / two;

    let cvrgd = fmax_c::<F>(
        rtol1 * work[(wgbase + ifirst + 1u32) as usize],
        rtol2 * fmax_c::<F>(F::abs(work[(wbase + ifirst) as usize]), F::abs(work[(wbase + ilast - 1u32) as usize])),
    );

    let itmax = ITMAX;
    let mut mid = zero;
    let mut width = zero;
    let mut i = ifirst;
    while i < ilast {
        if work[(ebase + i) as usize] >= cvrgd {
            // [LEFT, RIGHT] must contain the desired eigenvalue.
            let mut left = work[(wbase + i) as usize];
            let mut back = work[(ebase + i) as usize];
            let mut negcnt = ilast;
            while negcnt > i {
                left = left - back;
                back = back * two;
                negcnt = cint_dlaneg_c::<F>(diag, dbase, work, lldbase, n, left, twist_index);
            }

            let mut right = work[(wbase + i) as usize];
            let mut back = work[(ebase + i) as usize];
            // The C's `for (negcnt = ifirst; negcnt <= i;)` always runs once
            // (`ifirst <= i`), so the flag reproduces a do-while.
            let mut negcnt = ifirst;
            let mut first: bool = true;
            while first || negcnt <= i {
                first = false;
                right = right + back;
                back = back * two;
                negcnt = cint_dlaneg_c::<F>(diag, dbase, work, lldbase, n, right, twist_index);
            }

            let mut iter: u32 = 0;
            while iter < itmax {
                mid = (left + right) * half;
                width = right - mid;
                if width < cvrgd {
                    iter = itmax;
                } else {
                    let negcnt = cint_dlaneg_c::<F>(diag, dbase, work, lldbase, n, mid, twist_index);
                    if negcnt <= i {
                        left = mid;
                    } else {
                        right = mid;
                    }
                    iter += 1;
                }
            }
            work[(wbase + i) as usize] = mid;
            work[(ebase + i) as usize] = width;
        }
        i += 1;
    }

    let mut i = ifirst;
    while i + 1u32 < ilast {
        work[(wgbase + i + 1u32) as usize] = fmax_c::<F>(
            zero,
            work[(wbase + i + 1u32) as usize]
                - work[(ebase + i + 1u32) as usize]
                - work[(wbase + i) as usize]
                - work[(ebase + i) as usize],
        );
        i += 1;
    }
}

/// `_dlarrf` (`eigh.c:853`). Writes `SIGMA` to `sc[SC_SIGMA]`; the new `DPLUS`
/// and `LPLUS` go into `vec` at `dplus_base` / `lplus_base`. Returns `INFO`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_dlarrf_c<F: Float>(
    diag: &Array<F>,
    dbase: u32,
    offd: &Array<F>,
    ebase_o: u32,
    work: &Array<F>,
    ldbase: u32,
    wbase: u32,
    wgbase: u32,
    werrbase: u32,
    n: u32,
    clstrt: u32,
    clgapl: F,
    eps: F,
    vec: &mut Array<F>,
    dplus_base: u32,
    lplus_base: u32,
    cc: &Array<F>,
    sc: &mut Array<F>,
    scb: u32,
) -> u32 {
    let zero = F::from_int(0);
    let four = F::from_int(4);
    let eight = F::from_int(8);
    let quarter = F::from_int(1) / four;

    // A small fudge, to really shift to the outside.
    let mut lsigma = work[(wbase + clstrt) as usize] - work[(werrbase + clstrt) as usize];
    lsigma = lsigma - F::abs(lsigma) * four * eps;

    let growthbound = diag[dbase as usize] * eight;
    let mut sigma = zero;
    let mut max1 = zero;
    // `#[cube]`'s inference distinguishes the two `let` forms: an
    // UNANNOTATED integer literal (`2u32`) is a comptime constant, while
    // `let mut x: u32 = 0` is a runtime variable. A mutable counter therefore
    // needs the annotation and a constant bound must not have one — and
    // assigning a bare literal to the counter to leave a loop does not
    // survive the expansion at all, hence the named bound.
    let ktry_max = 2u32;
    let mut ktry: u32 = 0;
    let mut found: bool = false;
    while ktry < ktry_max {
        let mut s = -lsigma;
        vec[dplus_base as usize] = diag[dbase as usize] + s;
        max1 = F::abs(vec[dplus_base as usize]);
        let mut i: u32 = 0;
        while i + 1u32 < n {
            let tmp = work[(ldbase + i) as usize] / vec[(dplus_base + i) as usize];
            vec[(lplus_base + i) as usize] = tmp;
            s = s * tmp * offd[(ebase_o + i) as usize] - lsigma;
            vec[(dplus_base + i + 1u32) as usize] = diag[(dbase + i + 1u32) as usize] + s;
            max1 = fmax_c::<F>(max1, F::abs(vec[(dplus_base + i + 1u32) as usize]));
            i += 1;
        }
        sigma = lsigma;
        if max1 <= growthbound {
            found = true;
            ktry = ktry_max;
        } else {
            // The shift failed the RRR test; back off to the outside.
            lsigma = lsigma - fmin_c::<F>(clgapl * quarter, work[(wgbase + clstrt + 1u32) as usize]);
            ktry += 1;
        }
    }

    sc[(scb + SC_SIGMA) as usize] = sigma;
    let mut info: u32 = 0;
    let overflowed = max1 > cc[(CC_FAIL) as usize];
    if !found && overflowed {
        info = 1;
    }
    info
}

/// `_dlar1v` (`eigh.c:992`). `wrkbase` is the `3 * n` `lplus | uminus |
/// work_p` scratch. Writes `RESID`, `RQCORR` to `sc[SC_RESID]`,
/// `sc[SC_RQCORR]`; returns `NEGCNT`. The twist index is read and written at
/// `iwork[tbase]`, with [`TWIST_NONE`] for the C's `-1`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_dlar1v_c<F: Float>(
    n: u32,
    lambda: F,
    diag: &Array<F>,
    dbase: u32,
    offd: &Array<F>,
    ebase_o: u32,
    work: &mut Array<F>,
    ldbase: u32,
    lldbase: u32,
    wrkbase: u32,
    vec: &mut Array<F>,
    vbase: u32,
    iwork: &mut Array<u32>,
    tbase: u32,
    sc: &mut Array<F>,
    scb: u32,
) -> u32 {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let lplus = wrkbase;
    let uminus = wrkbase + n;
    let work_p = wrkbase + 2u32 * n;

    let mut r1: u32 = 0;
    let mut r2 = n;
    if iwork[tbase as usize] == TWIST_NONE {
        iwork[tbase as usize] = 0;
    } else {
        r1 = iwork[tbase as usize];
        r2 = iwork[tbase as usize] + 1u32;
    }

    // Progressive transform (differential form) until the index R1.
    // C: `for (i = n-2; i >= r1; --i)`; ascending counter, same order.
    let mut neg2: u32 = 0;
    let mut s = diag[(dbase + n - 1u32) as usize] - lambda;
    work[(work_p + n - 1u32) as usize] = s;
    let mut cnt: u32 = 0;
    if n >= r1 + 1u32 {
        cnt = n - 1u32 - r1;
    }
    let mut t: u32 = 0;
    while t < cnt {
        let i = (n - 2u32) - t;
        let dminus = work[(lldbase + i) as usize] + s;
        if dminus < zero {
            neg2 += 1;
        }
        let tmp = diag[(dbase + i) as usize] / dminus;
        work[(uminus + i) as usize] = offd[(ebase_o + i) as usize] * tmp;
        s = s * tmp - lambda;
        work[(work_p + i) as usize] = s;
        t += 1;
    }

    // Stationary transform (differential form) until the index R2.
    let mut neg1: u32 = 0;
    let mut s = -lambda;
    let mut i: u32 = 0;
    while i < r1 {
        let dplus = diag[(dbase + i) as usize] + s;
        if dplus < zero {
            neg1 += 1;
        }
        work[(lplus + i) as usize] = work[(ldbase + i) as usize] / dplus;
        s = s * work[(lplus + i) as usize] * offd[(ebase_o + i) as usize] - lambda;
        i += 1;
    }
    let mut mingma = s + lambda + work[(work_p + r1) as usize];
    if mingma < zero {
        neg1 += 1;
    }

    let negcnt = neg1 + neg2;

    // Index (from R1 to R2) of the largest (in magnitude) diagonal element of
    // the inverse.
    let mut i = r1;
    while i + 1u32 < r2 {
        let dplus = diag[(dbase + i) as usize] + s;
        work[(lplus + i) as usize] = work[(ldbase + i) as usize] / dplus;
        let tmp = s * work[(lplus + i) as usize] * offd[(ebase_o + i) as usize];
        s = tmp - lambda;

        let tmp = tmp + work[(work_p + i + 1u32) as usize];
        if F::abs(tmp) <= F::abs(mingma) {
            mingma = tmp;
            iwork[tbase as usize] = i + 1u32;
        }
        i += 1;
    }

    // FP vector: solve N^T v = e_r.
    let r = iwork[tbase as usize];
    vec[(vbase + r) as usize] = one;
    let mut ztz = one;

    // C: `for (i = r-1; i >= 0; --i)`; ascending counter, same order.
    let mut t: u32 = 0;
    while t < r {
        let i = (r - 1u32) - t;
        let tmp = -(work[(lplus + i) as usize] * vec[(vbase + i + 1u32) as usize]);
        ztz = ztz + tmp * tmp;
        vec[(vbase + i) as usize] = tmp;
        t += 1;
    }

    let mut i = r;
    while i + 1u32 < n {
        let tmp = -(work[(uminus + i) as usize] * vec[(vbase + i) as usize]);
        ztz = ztz + tmp * tmp;
        vec[(vbase + i + 1u32) as usize] = tmp;
        i += 1;
    }

    let tmp = one / ztz;
    let nrminv = F::sqrt(tmp);
    let mut i: u32 = 0;
    while i < n {
        vec[(vbase + i) as usize] = vec[(vbase + i) as usize] * nrminv;
        i += 1;
    }
    sc[(scb + SC_RESID) as usize] = F::abs(mingma) * nrminv;
    sc[(scb + SC_RQCORR) as usize] = mingma * tmp;
    negcnt
}

/// `_dlaev2` (`eigh.c:1381`): libcint's 2x2 closed form.
///
/// **Not** `dlaev2.f`: `RT = SQRT(TB*TB + DF*DF)` rather than the reference's
/// scaled `ADF*SQRT(1+(AB/ADF)**2)` form, and `RT2 = (a*c - b*b)/RT1` rather
/// than `(ACMX/RT1)*ACMN - (B/RT1)*B`. Both differ in the last places and the
/// first can overflow where the reference does not; that is libcint's choice.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_dlaev2_c<F: Float>(
    diag: &Array<F>,
    dbase: u32,
    offd: &Array<F>,
    ebase_o: u32,
    eig: &mut Array<F>,
    eigbase: u32,
    vec: &mut Array<F>,
    vbase: u32,
) {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let two = F::from_int(2);
    let half = one / two;

    let a = diag[dbase as usize];
    let b = offd[ebase_o as usize];
    let c = diag[(dbase + 1u32) as usize];

    let sm = a + c;
    let df = a - c;
    let tb = b + b;

    let rt = F::sqrt(tb * tb + df * df);

    let mut rt1 = rt * half;
    let mut rt2 = -(rt * half);
    let mut sgn1_pos: bool = true;
    if sm > zero {
        rt1 = (sm + rt) * half;
        sgn1_pos = true;
        rt2 = (a * c - b * b) / rt1;
    } else if sm < zero {
        rt1 = (sm - rt) * half;
        sgn1_pos = false;
        rt2 = (a * c - b * b) / rt1;
    }

    let mut cs = df - rt;
    let mut sgn2_pos: bool = false;
    if df >= zero {
        cs = df + rt;
        sgn2_pos = true;
    }

    let mut cs1 = one;
    let mut sn1 = zero;
    if F::abs(cs) > F::abs(tb) {
        let ct = -tb / cs;
        sn1 = one / F::sqrt(ct * ct + one);
        cs1 = ct * sn1;
    } else if b != zero {
        let tn = -cs / tb;
        cs1 = one / F::sqrt(tn * tn + one);
        sn1 = tn * cs1;
    }

    if sgn1_pos == sgn2_pos {
        let tmp = cs1;
        cs1 = -sn1;
        sn1 = tmp;
    }

    eig[eigbase as usize] = rt2;
    eig[(eigbase + 1u32) as usize] = rt1;
    vec[vbase as usize] = -sn1;
    vec[(vbase + 1u32) as usize] = cs1;
    vec[(vbase + 2u32) as usize] = cs1;
    vec[(vbase + 3u32) as usize] = sn1;
}

/// `_dlasq4` (`eigh.c:149`): the dqds shift, into `sc[SC_TAU]`.
///
/// Every `return 0` before `*tau = s` leaves `tau` at its incoming value, as
/// in the C; the `done` flag reproduces that without an early return.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_dlasq4_c<F: Float>(
    work: &Array<F>,
    qp: u32,
    q1p: u32,
    ep: u32,
    e1p: u32,
    i0: u32,
    n0: u32,
    n0init: u32,
    cc: &Array<F>,
    sc: &mut Array<F>,
    scb: u32,
) {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let d0 = sc[(scb + SC_DMIN) as usize];
    let d1 = sc[(scb + SC_DMIN + 1u32) as usize];
    let d2 = sc[(scb + SC_DMIN + 2u32) as usize];
    let n_0 = sc[(scb + SC_DN) as usize];
    let n_1 = sc[(scb + SC_DN + 1u32) as usize];
    let n_2 = sc[(scb + SC_DN + 2u32) as usize];

    let mut s = zero;
    // `bail` is the C's `return 0` before `*tau = s`.
    let mut bail: bool = false;

    if n0init == n0 {
        // No eigenvalues deflated.
        if d0 == n_0 {
            if d1 == n_1 {
                // Cases 2 and 3.
                let b1 = F::sqrt(work[(qp + n0 - 1u32) as usize] * work[(ep + n0 - 2u32) as usize]);
                let b2 = F::sqrt(work[(qp + n0 - 2u32) as usize] * work[(ep + n0 - 3u32) as usize]);
                let a2 = work[(qp + n0 - 2u32) as usize] + work[(ep + n0 - 2u32) as usize];
                let gap2 = d2 - a2 - d2 * (F::from_int(1) / F::from_int(4));
                let mut gap1 = a2 - n_0 - (b1 + b2);
                if gap2 > b2 {
                    gap1 = a2 - n_0 - b2 / gap2 * b2;
                }
                if gap1 > b1 {
                    s = fmax_c::<F>(n_0 - b1 / gap1 * b1, d0 * (F::from_int(1) / F::from_int(2)));
                } else {
                    s = zero;
                    if n_0 > b1 {
                        s = n_0 - b1;
                    }
                    if a2 > b1 + b2 {
                        s = fmin_c::<F>(s, a2 - (b1 + b2));
                    }
                    s = fmax_c::<F>(s, d0 * cc[(CC_THIRD) as usize]);
                }
            } else {
                // Case 4.
                if work[(ep + n0 - 2u32) as usize] > work[(qp + n0 - 2u32) as usize] {
                    bail = true;
                }
                if !bail {
                    let mut b2 = work[(ep + n0 - 2u32) as usize] / work[(qp + n0 - 2u32) as usize];
                    let mut a2 = b2;

                    // `for (i = n0-3; i >= i0; i--)`; ascending counter.
                    let mut cnt: u32 = 0;
                    if n0 >= i0 + 3u32 {
                        cnt = n0 - i0 - 2u32;
                    }
                    let mut t: u32 = 0;
                    while t < cnt {
                        let i = (n0 - 3u32) - t;
                        let b1 = b2;
                        if work[(ep + i) as usize] > work[(qp + i) as usize] {
                            bail = true;
                            t = cnt;
                        } else {
                            b2 = b2 * (work[(ep + i) as usize] / work[(qp + i) as usize]);
                            a2 = a2 + b2;
                            if cc[(CC_CNST1) as usize] < a2 || fmax_c::<F>(b1, b2) < a2 * cc[(CC_HUNDREDTH) as usize] {
                                t = cnt;
                            } else {
                                t += 1;
                            }
                        }
                    }
                    if !bail {
                        a2 = a2 * cc[(CC_CNST3) as usize];
                        // Rayleigh quotient residual bound.
                        s = d0 * (F::from_int(1) / F::from_int(4));
                        if a2 < cc[(CC_CNST1) as usize] {
                            s = n_0 * (one - F::sqrt(a2)) / (a2 + one);
                        }
                    }
                }
            }
        } else if d0 == n_1 {
            // Case 4.
            if work[(e1p + n0 - 2u32) as usize] > work[(q1p + n0 - 1u32) as usize]
                || work[(ep + n0 - 3u32) as usize] > work[(qp + n0 - 3u32) as usize]
            {
                bail = true;
            }
            if !bail {
                let mut a2 = work[(e1p + n0 - 2u32) as usize] / work[(q1p + n0 - 1u32) as usize];
                let mut b2 = work[(ep + n0 - 3u32) as usize] / work[(qp + n0 - 3u32) as usize];

                a2 = a2 + b2;
                // `for (i = n0-4; i >= i0; i--)`.
                let mut cnt: u32 = 0;
                if n0 >= i0 + 4u32 {
                    cnt = n0 - i0 - 3u32;
                }
                let mut t: u32 = 0;
                while t < cnt {
                    let i = (n0 - 4u32) - t;
                    if b2 == zero {
                        t = cnt;
                    } else {
                        let b1 = b2;
                        if work[(ep + i) as usize] > work[(qp + i) as usize] {
                            bail = true;
                            t = cnt;
                        } else {
                            b2 = b2 * (work[(ep + i) as usize] / work[(qp + i) as usize]);
                            a2 = a2 + b2;
                            if fmax_c::<F>(b2, b1) * F::from_int(100) < a2 || cc[(CC_CNST1) as usize] < a2 {
                                t = cnt;
                            } else {
                                t += 1;
                            }
                        }
                    }
                }
                if !bail {
                    a2 = a2 * cc[(CC_CNST3) as usize];
                    s = d0 * (F::from_int(1) / F::from_int(4));
                    if a2 < cc[(CC_CNST1) as usize] {
                        s = n_1 * (one - F::sqrt(a2)) / (a2 + one);
                    }
                }
            }
        } else if d0 == n_2 {
            // Case 5.
            if work[(e1p + n0 - 3u32) as usize] > work[(q1p + n0 - 2u32) as usize]
                || work[(e1p + n0 - 2u32) as usize] > work[(q1p + n0 - 1u32) as usize]
            {
                bail = true;
            }
            if !bail {
                let mut a2 = work[(e1p + n0 - 3u32) as usize] / work[(q1p + n0 - 2u32) as usize]
                    * (work[(e1p + n0 - 2u32) as usize] / work[(q1p + n0 - 1u32) as usize] + one);

                if n0 > i0 + 3u32 {
                    let mut b2 = work[(ep + n0 - 4u32) as usize] / work[(qp + n0 - 4u32) as usize];
                    a2 = a2 + b2;
                    // `for (i = n0-5; i >= i0; i--)`.
                    let mut cnt: u32 = 0;
                    if n0 >= i0 + 5u32 {
                        cnt = n0 - i0 - 4u32;
                    }
                    let mut t: u32 = 0;
                    while t < cnt {
                        let i = (n0 - 5u32) - t;
                        let b1 = b2;
                        if work[(ep + i) as usize] > work[(qp + i) as usize] {
                            bail = true;
                            t = cnt;
                        } else {
                            b2 = b2 * (work[(ep + i) as usize] / work[(qp + i) as usize]);
                            a2 = a2 + b2;
                            if cc[(CC_CNST1) as usize] < a2 || fmax_c::<F>(b2, b1) < a2 * cc[(CC_HUNDREDTH) as usize] {
                                t = cnt;
                            } else {
                                t += 1;
                            }
                        }
                    }
                    if !bail {
                        a2 = a2 * cc[(CC_CNST3) as usize];
                    }
                }

                if !bail {
                    s = d0 * (F::from_int(1) / F::from_int(4));
                    if a2 < cc[(CC_CNST1) as usize] {
                        s = n_2 * (one - F::sqrt(a2)) / (a2 + one);
                    }
                }
            }
        } else {
            // Case 6, no information to guide us.
            s = d0 * (F::from_int(1) / F::from_int(4));
        }
    } else if n0init == n0 + 1u32 {
        // One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN.
        if d1 == n_1 && d2 == n_2 {
            if work[(ep + n0 - 2u32) as usize] > work[(qp + n0 - 2u32) as usize] {
                bail = true;
            }
            if !bail {
                s = d1 * cc[(CC_THIRD) as usize];
                let mut b1 = work[(ep + n0 - 2u32) as usize] / work[(qp + n0 - 2u32) as usize];
                let mut b2 = b1;
                if b2 != zero {
                    // `for (i = n0-3; i >= i0; i--)`.
                    let mut cnt: u32 = 0;
                    if n0 >= i0 + 3u32 {
                        cnt = n0 - i0 - 2u32;
                    }
                    let mut t: u32 = 0;
                    while t < cnt {
                        let i = (n0 - 3u32) - t;
                        let a2 = b1;
                        if work[(ep + i) as usize] > work[(qp + i) as usize] {
                            bail = true;
                            t = cnt;
                        } else {
                            b1 = b1 * (work[(ep + i) as usize] / work[(qp + i) as usize]);
                            b2 = b2 + b1;
                            if fmax_c::<F>(b1, a2) * F::from_int(100) < b2 {
                                t = cnt;
                            } else {
                                t += 1;
                            }
                        }
                    }
                }
                if !bail {
                    let a2 = d1 / (b2 * cc[(CC_CNST3) as usize] + one);
                    let b2s = F::sqrt(b2 * cc[(CC_CNST3) as usize]);
                    let gap2 = d2 * (F::from_int(1) / F::from_int(2)) - a2;
                    if gap2 > zero && gap2 > b2s * a2 {
                        s = fmax_c::<F>(s, a2 * (one - a2 * cc[(CC_CNST2) as usize] * (b2s / gap2) * b2s));
                    } else {
                        s = fmax_c::<F>(s, a2 * (one - b2s * cc[(CC_CNST2) as usize]));
                    }
                }
            }
        } else if d1 == n_1 {
            s = d1 * (F::from_int(1) / F::from_int(2));
        } else {
            s = d1 * (F::from_int(1) / F::from_int(4));
        }
    } else if n0init == n0 + 2u32 {
        // Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.
        let two = F::from_int(2);
        if d2 == n_2 && work[(ep + n0 - 2u32) as usize] * two < work[(qp + n0 - 2u32) as usize] {
            if work[(ep + n0 - 2u32) as usize] > work[(qp + n0 - 2u32) as usize] {
                bail = true;
            }
            if !bail {
                let mut b1 = work[(ep + n0 - 2u32) as usize] / work[(qp + n0 - 2u32) as usize];
                let mut b2 = b1;
                if b2 != zero {
                    // `for (i = n0-2; i > i0; i--)`, reading index `i - 1`.
                    let mut cnt: u32 = 0;
                    if n0 >= i0 + 3u32 {
                        cnt = n0 - 2u32 - i0;
                    }
                    let mut t: u32 = 0;
                    while t < cnt {
                        let i = (n0 - 2u32) - t;
                        if work[(ep + i - 1u32) as usize] > work[(qp + i - 1u32) as usize] {
                            bail = true;
                            t = cnt;
                        } else {
                            b1 = b1 * (work[(ep + i - 1u32) as usize] / work[(qp + i - 1u32) as usize]);
                            b2 = b2 + b1;
                            if b1 * F::from_int(100) < b2 {
                                t = cnt;
                            } else {
                                t += 1;
                            }
                        }
                    }
                }
                if !bail {
                    s = d2 * cc[(CC_THIRD) as usize];
                    let a2 = d2 / (b2 * cc[(CC_CNST3) as usize] + one);
                    let b2s = F::sqrt(b2 * cc[(CC_CNST3) as usize]);
                    let gap2 = work[(qp + n0 - 2u32) as usize] + work[(ep + n0 - 3u32) as usize]
                        - F::sqrt(work[(qp + n0 - 3u32) as usize] * work[(ep + n0 - 3u32) as usize])
                        - a2;
                    if gap2 > zero && gap2 > b2s * a2 {
                        s = fmax_c::<F>(s, a2 * (one - a2 * cc[(CC_CNST2) as usize] * (b2s / gap2) * b2s));
                    } else {
                        s = fmax_c::<F>(s, a2 * (one - b2s * cc[(CC_CNST2) as usize]));
                    }
                }
            }
        } else {
            s = d2 * (F::from_int(1) / F::from_int(4));
        }
    } else if n0init > n0 + 2u32 {
        // Case 12, more than two eigenvalues deflated. No information.
        s = zero;
    }

    if !bail {
        sc[(scb + SC_TAU) as usize] = s;
    }
}

/// `_dlasq5` (`eigh.c:388`): one dqds step.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_dlasq5_c<F: Float>(
    work: &mut Array<F>,
    qp: u32,
    q1p: u32,
    ep: u32,
    e1p: u32,
    i0: u32,
    n0: u32,
    tau: F,
    tol: F,
    sc: &mut Array<F>,
    scb: u32,
) {
    let zero = F::from_int(0);
    let mut diag = work[(qp + i0) as usize] - tau;
    let mut diag_min = diag;

    let mut j = i0;
    while j + 3u32 < n0 {
        work[(q1p + j) as usize] = diag + work[(ep + j) as usize];
        let temp = work[(qp + j + 1u32) as usize] / work[(q1p + j) as usize];
        diag = diag * temp - tau;
        if diag < tol {
            diag = zero;
        }
        diag_min = fmin_c::<F>(diag_min, diag);
        work[(e1p + j) as usize] = work[(ep + j) as usize] * temp;
        j += 1;
    }
    sc[(scb + SC_DN + 2u32) as usize] = diag;

    // The two unrolled tail steps use fresh names (`ja`, `jb`, `ta`, `tb`)
    // rather than shadowing the loop's `j`/`temp`: a `#[cube]` body that
    // shadows a `let mut` with a `let` does not lower on cubecl-cpu 0.10 (the
    // launch silently no-ops; `docs/cubecl-cpu-lowering-limit.md`).
    let ja = n0 - 3u32;
    work[(q1p + ja) as usize] = diag + work[(ep + ja) as usize];
    let ta = work[(qp + ja + 1u32) as usize] / work[(q1p + ja) as usize];
    work[(e1p + ja) as usize] = work[(ep + ja) as usize] * ta;
    diag = diag * ta - tau;
    sc[(scb + SC_DN + 1u32) as usize] = diag;

    let jb = n0 - 2u32;
    work[(q1p + jb) as usize] = diag + work[(ep + jb) as usize];
    let tb = work[(qp + jb + 1u32) as usize] / work[(q1p + jb) as usize];
    work[(e1p + jb) as usize] = work[(ep + jb) as usize] * tb;
    diag = diag * tb - tau;
    sc[(scb + SC_DN) as usize] = diag;

    work[(q1p + n0 - 1u32) as usize] = diag;

    sc[(scb + SC_DMIN + 2u32) as usize] = diag_min;
    sc[(scb + SC_DMIN + 1u32) as usize] =
        fmin_c::<F>(sc[(scb + SC_DMIN + 2u32) as usize], sc[(scb + SC_DN + 1u32) as usize]);
    sc[(scb + SC_DMIN) as usize] = fmin_c::<F>(sc[(scb + SC_DMIN + 1u32) as usize], sc[(scb + SC_DN) as usize]);
}

/// `_dlasq2` (`eigh.c:431`). `work[qb .. qb + 4n]` is
/// `qvec | qvec1 | evec | evec1`; the eigenvalues come back in the base
/// `qvec`. Returns `INFO`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_dlasq2_c<F: Float>(
    n: u32,
    work: &mut Array<F>,
    qb: u32,
    diag: &Array<F>,
    dbase: u32,
    offd: &Array<F>,
    ebase_o: u32,
    eps: F,
    cc: &Array<F>,
    sc: &mut Array<F>,
    scb: u32,
) -> u32 {
    let zero = F::from_int(0);
    let two = F::from_int(2);
    let four = F::from_int(4);
    let half = F::from_int(1) / two;

    let qvec = qb;
    let qvec1 = qb + n;
    let evec = qb + n * 2u32;
    let evec1 = qb + n * 3u32;

    // The C walks `for (i = 0, j = n-1; i < n-1; ++i, --j)`. `j` is written
    // as an expression of `i` rather than decremented: cubecl-cpu 0.10 cannot
    // lower a decremented counter (AGENTS.md, `tests/cubecl_cpu_limits.rs`),
    // and the visit order and operands are identical.
    let mut i: u32 = 0;
    while i + 1u32 < n {
        let j = (n - 1u32) - i;
        let temp = F::abs(diag[(dbase + i) as usize]);
        work[(qvec1 + i) as usize] = zero;
        work[(evec1 + i) as usize] = zero;
        work[(evec + j - 1u32) as usize] = offd[(ebase_o + i) as usize] * offd[(ebase_o + i) as usize] * temp;
        work[(qvec + j) as usize] = temp;
        i += 1;
    }
    work[qvec as usize] = F::abs(diag[(dbase + n - 1u32) as usize]);
    work[(qvec1 + n - 1u32) as usize] = zero;
    work[(evec + n - 1u32) as usize] = zero;
    work[(evec1 + n - 1u32) as usize] = zero;

    // Reverse the qd-array, if warranted.
    if work[qvec as usize] < work[(qvec + n - 1u32) as usize] * (F::from_int(3) / F::from_int(2)) {
        let mut i: u32 = 0;
        while i < n / 2u32 {
            let j = (n - 1u32) - i;
            let temp = work[(qvec + i) as usize];
            work[(qvec + i) as usize] = work[(qvec + j) as usize];
            work[(qvec + j) as usize] = temp;

            let temp = work[(evec + i) as usize];
            work[(evec + i) as usize] = work[(evec + j - 1u32) as usize];
            work[(evec + j - 1u32) as usize] = temp;
            i += 1;
        }
    }

    // dqd maps Z to ZZ plus Li's test.
    let mut diag_sum = work[qvec as usize];
    let mut i: u32 = 0;
    while i + 1u32 < n {
        let temp = diag_sum + work[(evec + i) as usize];
        work[(qvec1 + i) as usize] = temp;
        work[(evec1 + i) as usize] = work[(qvec + i + 1u32) as usize] * (work[(evec + i) as usize] / temp);
        diag_sum = work[(qvec + i + 1u32) as usize] * (diag_sum / temp);
        i += 1;
    }
    work[(qvec1 + n - 1u32) as usize] = diag_sum;
    let mut diag_sum = work[qvec1 as usize];
    let mut i: u32 = 0;
    while i + 1u32 < n {
        let temp = diag_sum + work[(evec1 + i) as usize];
        work[(qvec + i) as usize] = temp;
        work[(evec + i) as usize] = work[(qvec1 + i + 1u32) as usize] * (work[(evec1 + i) as usize] / temp);
        diag_sum = work[(qvec1 + i + 1u32) as usize] * (diag_sum / temp);
        i += 1;
    }
    work[(qvec + n - 1u32) as usize] = diag_sum;

    // `n0` DESCENDS in the C (`--n0`, `n0 += -2`), and both the outer
    // `for (itry = 0; n0 > 0; ++itry)` and the inner `while (n0 > i0)` are
    // driven by it. cubecl-cpu 0.10 cannot lower a loop whose counter is
    // decremented (AGENTS.md, `tests/cubecl_cpu_limits.rs`,
    // `docs/cubecl-cpu-lowering-limit.md`), so the ASCENDING `ndef = n - n0`
    // drives the loops instead and `n0` is kept in sync from it. Same visit
    // order, same operands, same values.
    let mut ndef: u32 = 0;
    let mut n0 = n;
    let mut tau = zero;
    sc[(scb + SC_DMIN) as usize] = zero;
    sc[(scb + SC_DMIN + 1u32) as usize] = zero;
    sc[(scb + SC_DMIN + 2u32) as usize] = zero;
    sc[(scb + SC_DN) as usize] = zero;
    sc[(scb + SC_DN + 1u32) as usize] = zero;
    sc[(scb + SC_DN + 2u32) as usize] = zero;

    let mut ret: u32 = 0;
    let mut stop: bool = false;
    let mut itry: u32 = 0;
    while ndef < n && !stop {
        if itry >= n {
            ret = 3;
            stop = true;
        } else {
            itry += 1;

            // E(N0) holds the (negated) SIGMA at which I0:N0 split off.
            let mut sigma = -work[(evec + n0 - 1u32) as usize];
            if sigma < zero {
                ret = 1;
                stop = true;
            } else {
                // Last unreduced submatrix's top index I0, QMAX and EMIN.
                // The C walks `for (i = n0-1; i > 0; i--)` and keeps the final
                // `i`; ascending counter, same order, same exit index.
                let mut emax = zero;
                let mut qmin = work[(qvec + n0 - 1u32) as usize];
                let mut i0 = n0 - 1u32;
                let mut t: u32 = 0;
                while t + 1u32 < n0 {
                    let i = (n0 - 1u32) - t;
                    if work[(evec + i - 1u32) as usize] <= zero {
                        i0 = i;
                        t = n0; // leave the loop with i0 = i
                    } else {
                        if qmin >= emax * four {
                            qmin = fmin_c::<F>(qmin, work[(qvec + i) as usize]);
                            emax = fmax_c::<F>(emax, work[(evec + i - 1u32) as usize]);
                        }
                        i0 = i - 1u32;
                        t += 1;
                    }
                }
                if n0 == 1u32 {
                    i0 = 0;
                }

                let mut qp = qvec;
                let mut q1p = qvec1;
                let mut ep = evec;
                let mut e1p = evec1;

                if qmin < zero || emax < zero {
                    ret = 1;
                    stop = true;
                } else {
                    // Put -(initial shift) into DMIN.
                    sc[(scb + SC_DMIN) as usize] = -fmax_c::<F>(zero, qmin - two * F::sqrt(qmin * emax));

                    let nbig = (n0 - i0) * 10u32;
                    let mut iwhilb: u32 = 0;
                    let mut broke: bool = false;
                    while iwhilb < nbig && !broke {
                        let n0init = n0;
                        let tol = eps * F::from_int(100);
                        let tol2 = tol * tol;

                        // Index expressions are hoisted into `u32` locals
                        // here, which is pure address arithmetic and cannot
                        // change a value. It is also what keeps this function
                        // under cubecl-cpu 0.10's per-function lowering limit
                        // (`tests/cubecl_cpu_limits.rs`,
                        // `docs/cubecl-cpu-lowering-limit.md`).
                        let mut deflating: bool = true;
                        while n - ndef > i0 && deflating {
                            let n1 = n0 - 1u32;
                            let n2 = n0 - 2u32;
                            let qn1 = (qp + n1) as usize;
                            let qn2 = (qp + n2) as usize;
                            let en2 = (ep + n2) as usize;
                            let mut one_eig: bool = false;
                            if n1 == i0 {
                                one_eig = true;
                            }
                            if work[en2] < tol2 * (sigma + work[qn1]) {
                                one_eig = true;
                            }
                            if work[(e1p + n2) as usize] < tol2 * work[qn2] {
                                one_eig = true;
                            }
                            if one_eig {
                                work[(qvec + n1) as usize] = work[qn1] + sigma;
                                ndef += 1;
                                n0 = n - ndef;
                            } else {
                                let n3 = n1 - 2u32;
                                let mut two_eig: bool = false;
                                if n2 == i0 {
                                    two_eig = true;
                                }
                                if work[(ep + n3) as usize] < tol2 * sigma {
                                    two_eig = true;
                                }
                                if work[(e1p + n3) as usize] < tol2 * work[(qp + n3) as usize] {
                                    two_eig = true;
                                }
                                if two_eig {
                                    if work[qn1] > work[qn2] {
                                        let sw = work[qn1];
                                        work[qn1] = work[qn2];
                                        work[qn2] = sw;
                                    }
                                    let mut t = (work[qn2] - work[qn1] + work[en2]) * half;
                                    if work[en2] > work[qn1] * tol2 && t != zero {
                                        let s = work[qn1] * (work[en2] / t);
                                        let s = work[qn1] * (work[en2] / (t + F::sqrt(t * (t + s))));
                                        t = work[qn2] + (s + work[en2]);
                                        work[qn1] = work[qn1] * (work[qn2] / t);
                                        work[qn2] = t;
                                    }
                                    work[(qvec + n2) as usize] = work[qn2] + sigma;
                                    work[(qvec + n1) as usize] = work[qn1] + sigma;
                                    ndef += 2;
                                    n0 = n - ndef;
                                } else {
                                    deflating = false;
                                }
                            }
                        }

                        if n0 <= i0 {
                            broke = true;
                        } else {
                            if sc[(scb + SC_DMIN) as usize] <= zero {
                                tau = -sc[(scb + SC_DMIN) as usize];
                            } else {
                                sc[(scb + SC_TAU) as usize] = tau;
                                cint_dlasq4_c::<F>(work, qp, q1p, ep, e1p, i0, n0, n0init, cc, sc, scb);
                                tau = sc[(scb + SC_TAU) as usize];
                            }

                            // Call dqds until DMIN > 0.
                            let tol = eps * sigma;
                            let mut iter: u32 = 0;
                            while iter < 3u32 {
                                cint_dlasq5_c::<F>(work, qp, q1p, ep, e1p, i0, n0, tau, tol, sc, scb);
                                if sc[(scb + SC_DMIN) as usize] >= zero {
                                    iter = 3u32;
                                } else {
                                    if sc[(scb + SC_DMIN + 1u32) as usize] > zero {
                                        tau = tau + sc[(scb + SC_DMIN) as usize];
                                    } else {
                                        tau = tau * (F::from_int(1) / F::from_int(4));
                                    }
                                    iter += 1;
                                }
                            }
                            if sc[(scb + SC_DMIN) as usize] < zero {
                                tau = zero;
                                cint_dlasq5_c::<F>(work, qp, q1p, ep, e1p, i0, n0, tau, tol, sc, scb);
                            }

                            sigma = sigma + tau;

                            // Swap the base offsets, not the pointers.
                            let sw = qp;
                            qp = q1p;
                            q1p = sw;
                            let sw = ep;
                            ep = e1p;
                            e1p = sw;
                            iwhilb += 1;
                        }
                    }

                    if !broke && iwhilb == nbig {
                        ret = 2;
                        stop = true;
                    }
                }
            }
        }
    }
    ret
}

/// `_compute_eigenvalues` (`eigh.c:629`).
///
/// `work` is the driver's single buffer. `wgbase` is the **`-1`-based**
/// `wgap` view; `wrkbase` is the C's `buf_wrk`, which needs `5 * n`
/// (`e2` then the `3n` factorisation scratch that `_dlasq2` reuses as its
/// `4n` qd array). Returns `INFO`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_compute_eigenvalues_c<F: Float>(
    n: u32,
    diag: &mut Array<F>,
    dbase: u32,
    offd: &mut Array<F>,
    ebase_o: u32,
    eig: &mut Array<F>,
    eigbase: u32,
    work: &mut Array<F>,
    werrbase: u32,
    wgbase: u32,
    wrkbase: u32,
    eps_c: F,
    cc: &Array<F>,
    sc: &mut Array<F>,
    scb: u32,
    icnt: &mut Array<u32>,
    icb: u32,
) -> u32 {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let two = F::from_int(2);
    let four = F::from_int(4);
    let sixteen = F::from_int(16);
    let quarter = one / four;

    let eps = two * eps_c;
    let rtl = cc[(CC_RTL) as usize]; // "sqrt(eps)", hard-coded in libcint
    let rtol = sixteen * eps_c;

    // Init WERR, WGAP; Gerschgorin intervals and spectral diameter.
    let mut gl = diag[dbase as usize];
    let mut gu = diag[dbase as usize];
    let mut eold = zero;
    offd[(ebase_o + n - 1u32) as usize] = zero;
    let mut i: u32 = 0;
    while i < n {
        work[(werrbase + i) as usize] = zero;
        work[(wgbase + i + 1u32) as usize] = zero;
        let eabs = F::abs(offd[(ebase_o + i) as usize]);
        let tmp1 = eabs + eold;
        gl = fmin_c::<F>(gl, diag[(dbase + i) as usize] - tmp1);
        gu = fmax_c::<F>(gu, diag[(dbase + i) as usize] + tmp1);
        eold = eabs;
        i += 1;
    }
    let mut spectral_diameter = gu - gl;
    offd[(ebase_o + n - 1u32) as usize] = zero;

    // `e2 = work; work += n;`
    let e2base = wrkbase;
    let wb = wrkbase + n;
    let mut i: u32 = 0;
    while i + 1u32 < n {
        work[(e2base + i) as usize] = offd[(ebase_o + i) as usize] * offd[(ebase_o + i) as usize];
        i += 1;
    }

    // Case of DQDS: approximations to the extremal eigenvalues of the block.
    let mut info: u32 = 0;
    let mut bad: bool = false;
    let iinfo = cint_dlarrk_c::<F>(diag, dbase, work, e2base, n, 1u32, gl, gu, rtl, eps_c, sc, scb);
    if iinfo != 0 {
        bad = true;
        info = 1; // the C's -1
    }
    let isleft = fmax_c::<F>(
        gl,
        sc[(scb + SC_W) as usize]
            - sc[(scb + SC_WERR) as usize]
            - eps * F::from_int(100) * F::abs(sc[(scb + SC_W) as usize] - sc[(scb + SC_WERR) as usize]),
    );
    let iinfo = cint_dlarrk_c::<F>(diag, dbase, work, e2base, n, n, gl, gu, rtl, eps_c, sc, scb);
    if iinfo != 0 {
        bad = true;
        info = 1;
    }
    let isrght = fmin_c::<F>(
        gu,
        sc[(scb + SC_W) as usize]
            + sc[(scb + SC_WERR) as usize]
            + eps * F::from_int(100) * F::abs(sc[(scb + SC_W) as usize] + sc[(scb + SC_WERR) as usize]),
    );

    if !bad {
        // Improve the estimate of the spectral diameter.
        spectral_diameter = isrght - isleft;

        // Negcount at the 1/4 and 3/4 points.
        cint_dlarrc_c::<F>(
            n,
            isleft + spectral_diameter * quarter,
            isrght - spectral_diameter * quarter,
            diag,
            dbase,
            work,
            e2base,
            icnt,
            icb,
        );
        let lcnt = icnt[icb as usize];
        let rcnt = icnt[(icb + 1u32) as usize];

        let mut sigma = fmin_c::<F>(isrght, gu);
        let mut sgndef = -one;
        // `lcnt - 1 >= n - rcnt` in C's `int` arithmetic; `lcnt >= 1` is not
        // guaranteed, so compare with the subtraction moved to the other side.
        if lcnt + rcnt >= n + 1u32 {
            sigma = fmax_c::<F>(isleft, gl);
            sgndef = one;
        }

        // The shift increment TAU, in case SIGMA needs refining.
        let mut tau = fmax_c::<F>(spectral_diameter * F::cast_from(n), two * F::abs(sigma)) * eps;

        let maxtry = MAXTRY;
        let mut idum: u32 = 0;
        while idum < maxtry {
            // L D L^T of T - sigma I: D in work[0..n], L in work[n..2n],
            // reciprocal pivots in work[2n..3n].
            let mut dpivot = diag[dbase as usize] - sigma;
            work[wb as usize] = dpivot;
            let mut dmax = F::abs(work[wb as usize]);
            let mut i: u32 = 0;
            while i + 1u32 < n {
                work[(wb + n * 2u32 + i) as usize] = one / work[(wb + i) as usize];
                let tmp = offd[(ebase_o + i) as usize] * work[(wb + n * 2u32 + i) as usize];
                work[(wb + n + i) as usize] = tmp;
                dpivot = diag[(dbase + i + 1u32) as usize] - sigma - tmp * offd[(ebase_o + i) as usize];
                work[(wb + i + 1u32) as usize] = dpivot;
                dmax = fmax_c::<F>(dmax, F::abs(dpivot));
                i += 1;
            }
            // Element growth.
            let mut norep: bool = dmax > spectral_diameter * F::from_int(64);
            if !norep {
                // Every entry of D (of L D L^T) must have the same sign.
                let mut i: u32 = 0;
                while i < n {
                    let tmp = sgndef * work[(wb + i) as usize];
                    if tmp < zero {
                        norep = true;
                        i = n;
                    } else {
                        i += 1;
                    }
                }
            }
            if norep {
                if idum == maxtry - 1u32 {
                    // The fudged Gerschgorin shift should succeed.
                    if sgndef == one {
                        sigma = gl - spectral_diameter * two * eps * F::cast_from(n);
                    } else {
                        sigma = gu + spectral_diameter * two * eps * F::cast_from(n);
                    }
                } else {
                    sigma = sigma - sgndef * tau;
                    tau = tau * two;
                }
                idum += 1;
            } else {
                idum = maxtry;
            }
        }

        // Store the shift, D and L.
        offd[(ebase_o + n - 1u32) as usize] = sigma;
        let mut ip: u32 = 0;
        while ip < n {
            diag[(dbase + ip) as usize] = work[(wb + ip) as usize];
            ip += 1;
        }
        let mut ip: u32 = 0;
        while ip + 1u32 < n {
            offd[(ebase_o + ip) as usize] = work[(wb + n + ip) as usize];
            ip += 1;
        }

        // dqds for all eigenvalues of the block.
        let iinfo = cint_dlasq2_c::<F>(n, work, wb, diag, dbase, offd, ebase_o, eps_c, cc, sc, scb);
        if iinfo != 0 {
            bad = true;
            info = 5; // the C's -5
        }
        if !bad {
            let mut i: u32 = 0;
            while i < n {
                if work[(wb + i) as usize] < zero {
                    bad = true;
                    info = 6; // the C's -6
                    i = n;
                } else {
                    i += 1;
                }
            }
        }
        if !bad {
            if sgndef > zero {
                let mut i: u32 = 0;
                while i < n {
                    eig[(eigbase + i) as usize] = work[(wb + n - 1u32 - i) as usize];
                    i += 1;
                }
            } else {
                let mut i: u32 = 0;
                while i < n {
                    eig[(eigbase + i) as usize] = -work[(wb + i) as usize];
                    i += 1;
                }
            }

            let mut i: u32 = 0;
            while i < n {
                work[(werrbase + i) as usize] = rtol * F::abs(eig[(eigbase + i) as usize]);
                i += 1;
            }

            // The right gap between the intervals.
            let mut i: u32 = 0;
            while i + 1u32 < n {
                work[(wgbase + i + 1u32) as usize] = fmax_c::<F>(
                    zero,
                    eig[(eigbase + i + 1u32) as usize]
                        - work[(werrbase + i + 1u32) as usize]
                        - (eig[(eigbase + i) as usize] + work[(werrbase + i) as usize]),
                );
                i += 1;
            }
            // `wgap[-1]` (`eigh.c:830`), read back by the eigenvector pass.
            work[wgbase as usize] = fmax_c::<F>(zero, eig[eigbase as usize] - work[werrbase as usize] - gl);
            work[(wgbase + n) as usize] = fmax_c::<F>(
                zero,
                gu - sigma - (eig[(eigbase + n - 1u32) as usize] + work[(werrbase + n - 1u32) as usize]),
            );
        }
    }
    info
}

/// `_compute_eigenvectors` (`eigh.c:1095`).
///
/// `wrkbase` is the C's `work` (the same `buf_wrk`), carved as
/// `buf_w | buf_ld | buf_lld | buf_wrk2` and needing `6 * n`. `iwork` holds
/// the twist indices at `0..n` and the two cluster-range tables at `n` and
/// `3n`. Returns `INFO`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_compute_eigenvectors_c<F: Float>(
    n: u32,
    diag: &mut Array<F>,
    dbase: u32,
    offd: &mut Array<F>,
    ebase_o: u32,
    eig: &mut Array<F>,
    eigbase: u32,
    work: &mut Array<F>,
    werrbase: u32,
    wgbase: u32,
    wrkbase: u32,
    vec: &mut Array<F>,
    vbase: u32,
    iwork: &mut Array<u32>,
    iwbase: u32,
    eps_c: F,
    big: F,
    cc: &Array<F>,
    sc: &mut Array<F>,
    scb: u32,
) -> u32 {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let two = F::from_int(2);
    let three = F::from_int(3);
    let four = F::from_int(4);
    let eight = F::from_int(8);
    let half = one / two;

    let buf_w = wrkbase;
    let buf_ld = wrkbase + n;
    let buf_lld = wrkbase + n * 2u32;
    let buf_wrk = wrkbase + n * 3u32;
    let mut i: u32 = 0;
    while i < n * 6u32 {
        work[(wrkbase + i) as usize] = zero;
        i += 1;
    }
    let mut i: u32 = 0;
    while i < n {
        iwork[(iwbase + i) as usize] = 0;
        i += 1;
    }
    // Every `iwork` index is relative to this unit's block: the batched kernel
    // gives each unit `5 * n_max` slots at `b * 5 * n_max`, and a missing base
    // here is a silent data race between units, not a bounds error.
    let mut old_cluster = iwbase + n;
    let mut new_cluster = iwbase + n * 3u32;

    let eps = eps_c;
    let rqtol = eps_c * two;
    let tol = eps_c * eight;

    let mut sigma = offd[(ebase_o + n - 1u32) as usize];

    let mut i: u32 = 0;
    while i < n {
        work[(buf_w + i) as usize] = eig[(eigbase + i) as usize];
        eig[(eigbase + i) as usize] = eig[(eigbase + i) as usize] + sigma;
        i += 1;
    }
    let mut ncluster: u32 = 1;
    iwork[old_cluster as usize] = 0;
    iwork[(old_cluster + 1u32) as usize] = n;
    let mut idone: u32 = 0;

    let mut ndepth: u32 = 0;
    let mut fail: bool = false;
    while ndepth < n {
        if idone == n {
            ndepth = n;
        } else {
            let ncluster1 = ncluster;
            ncluster = 0;
            // Process the clusters on the current level.
            let mut icls: u32 = 0;
            while icls < ncluster1 {
                let oldfst = iwork[(old_cluster + icls * 2u32) as usize];
                let oldlst = iwork[(old_cluster + icls * 2u32 + 1u32) as usize];
                if ndepth > 0u32 {
                    // Retrieve the RRR of the cluster from the previous level.
                    let mut i: u32 = 0;
                    while i < n {
                        diag[(dbase + i) as usize] = vec[(vbase + oldfst * n + i) as usize];
                        i += 1;
                    }
                    let mut i: u32 = 0;
                    while i + 1u32 < n {
                        offd[(ebase_o + i) as usize] = vec[(vbase + (oldfst + 1u32) * n + i) as usize];
                        i += 1;
                    }
                    sigma = vec[(vbase + (oldfst + 2u32) * n - 1u32) as usize];
                }
                // DL and DLL of the current RRR.
                let mut j: u32 = 0;
                while j + 1u32 < n {
                    let tmp = diag[(dbase + j) as usize] * offd[(ebase_o + j) as usize];
                    work[(buf_ld + j) as usize] = tmp;
                    work[(buf_lld + j) as usize] = tmp * offd[(ebase_o + j) as usize];
                    j += 1;
                }
                // `eigh.c:1188-1205`'s `ndepth > 0` limited-bisection block is
                // COMMENTED OUT upstream and is left out here for the same
                // reason: including it would change every bit.

                // Process the current node.
                let mut newfst = oldfst;
                let mut newlst = oldfst + 1u32;
                while newlst <= oldlst {
                    let mut skip: bool = false;
                    if newlst < oldlst {
                        if work[(wgbase + newlst) as usize]
                            < cc[(CC_MINGAP) as usize] * F::abs(work[(buf_w + newlst - 1u32) as usize])
                        {
                            skip = true;
                        }
                    }
                    if !skip {
                        if newlst - newfst > 1u32 {
                            // Left- and rightmost eigenvalue of the child to
                            // high precision.
                            cint_dlarrb_c::<F>(
                                diag,
                                dbase,
                                work,
                                buf_lld,
                                buf_w,
                                wgbase,
                                werrbase,
                                n,
                                newfst,
                                newfst + 1u32,
                                rqtol,
                                rqtol,
                                n,
                            );
                            cint_dlarrb_c::<F>(
                                diag,
                                dbase,
                                work,
                                buf_lld,
                                buf_w,
                                wgbase,
                                werrbase,
                                n,
                                newlst - 1u32,
                                newlst,
                                rqtol,
                                rqtol,
                                n,
                            );

                            // RRR of the child cluster, stored in `vec`.
                            let clgapl = work[(wgbase + newfst) as usize];
                            let iinfo = cint_dlarrf_c::<F>(
                                diag,
                                dbase,
                                offd,
                                ebase_o,
                                work,
                                buf_ld,
                                buf_w,
                                wgbase,
                                werrbase,
                                n,
                                newfst,
                                clgapl,
                                eps_c,
                                vec,
                                vbase + newfst * n,
                                vbase + (newfst + 1u32) * n,
                                cc,
                                sc,
                                scb,
                            );
                            if iinfo != 0 {
                                fail = true;
                            }
                            let tau = sc[(scb + SC_SIGMA) as usize];
                            vec[(vbase + (newfst + 2u32) * n - 1u32) as usize] = sigma + tau;

                            let mut k = newfst;
                            while k < newlst {
                                let mut fudge = eps * three * F::abs(work[(buf_w + k) as usize]);
                                work[(buf_w + k) as usize] = work[(buf_w + k) as usize] - tau;
                                fudge = fudge + eps * four * F::abs(work[(buf_w + k) as usize]);
                                // Fudge errors; gaps are deliberately not fudged.
                                work[(werrbase + k) as usize] = work[(werrbase + k) as usize] + fudge;
                                k += 1;
                            }
                            iwork[(new_cluster + ncluster * 2u32) as usize] = newfst;
                            iwork[(new_cluster + ncluster * 2u32 + 1u32) as usize] = newlst;
                            ncluster += 1;
                        } else {
                            // Eigenvector of a singleton.
                            let mut lambda = work[(buf_w + newfst) as usize];
                            let mut left = lambda - work[(werrbase + newfst) as usize];
                            let mut right = lambda + work[(werrbase + newfst) as usize];
                            let gap = work[(wgbase + newfst + 1u32) as usize];
                            // `gaptol` is computed exactly as `eigh.c:1254`
                            // does and passed nowhere: libcint's `_dlar1v`
                            // takes a `gaptol` parameter and never reads it
                            // (unlike `dlar1v.f`, which uses it to truncate
                            // the FP vector). Kept so the two read alike.
                            let mut _gaptol = gap * eps;
                            if newfst == 0u32 || newfst + 1u32 == n {
                                _gaptol = zero;
                            }
                            let savgap = gap;
                            let resid_tol = tol * gap;
                            let rqcorr_tol = rqtol * F::abs(lambda);

                            let mut needbs: bool = false;
                            // libcint's `bstres = 1e307`: an argument, because
                            // `F::new` takes an `f32` and 1e307 is out of its range,
                            // and because no constant in this module is a literal.
                            let mut bstres = big;
                            let mut bstw = zero;
                            iwork[(iwbase + newfst) as usize] = TWIST_NONE;
                            let maxrq = MAXRQITER;
                            let mut iter: u32 = 0;
                            while iter < maxrq {
                                let negcnt = cint_dlar1v_c::<F>(
                                    n,
                                    lambda,
                                    diag,
                                    dbase,
                                    offd,
                                    ebase_o,
                                    work,
                                    buf_ld,
                                    buf_lld,
                                    buf_wrk,
                                    vec,
                                    vbase + newfst * n,
                                    iwork,
                                    iwbase + newfst,
                                    sc,
                                    scb,
                                );
                                let resid = sc[(scb + SC_RESID) as usize];
                                let rqcorr = sc[(scb + SC_RQCORR) as usize];

                                // Keep track of the best lambda.
                                if resid < bstres {
                                    bstres = resid;
                                    bstw = lambda;
                                }

                                let mut leave: bool = false;
                                // Convergence test for the RQ iteration.
                                if resid < resid_tol || F::abs(rqcorr) < rqcorr_tol {
                                    leave = true;
                                }
                                if !leave {
                                    // Only use the RQCORR if it improves the
                                    // iterate.
                                    if lambda + rqcorr > right || lambda + rqcorr < left {
                                        needbs = true;
                                        leave = true;
                                    }
                                }
                                if !leave {
                                    // The RQCORR update must not move the
                                    // eigenvalue toward a neighbour.
                                    if newfst < negcnt {
                                        // The wanted eigenvalue lies left.
                                        if rqcorr > zero {
                                            needbs = true;
                                            leave = true;
                                        } else {
                                            right = lambda;
                                        }
                                    } else {
                                        // The wanted eigenvalue lies right.
                                        if rqcorr < zero {
                                            needbs = true;
                                            leave = true;
                                        } else {
                                            left = lambda;
                                        }
                                    }
                                }
                                if leave {
                                    iter = maxrq;
                                } else {
                                    work[(buf_w + newfst) as usize] = (right + left) * half;
                                    lambda = lambda + rqcorr;
                                    work[(werrbase + newfst) as usize] = (right - left) * half;

                                    if right - left < rqcorr_tol {
                                        if bstres < resid {
                                            lambda = bstw;
                                            cint_dlar1v_c::<F>(
                                                n,
                                                lambda,
                                                diag,
                                                dbase,
                                                offd,
                                                ebase_o,
                                                work,
                                                buf_ld,
                                                buf_lld,
                                                buf_wrk,
                                                vec,
                                                vbase + newfst * n,
                                                iwork,
                                                iwbase + newfst,
                                                sc,
                                                scb,
                                            );
                                        }
                                        iter = maxrq;
                                    } else {
                                        iter += 1;
                                    }
                                }
                            }

                            if needbs {
                                let tw = iwork[(iwbase + newfst) as usize] + 1u32;
                                cint_dlarrb_c::<F>(
                                    diag,
                                    dbase,
                                    work,
                                    buf_lld,
                                    buf_w,
                                    wgbase,
                                    werrbase,
                                    n,
                                    newfst,
                                    newfst + 1u32,
                                    zero,
                                    eps * two,
                                    tw,
                                );
                                lambda = work[(buf_w + newfst) as usize];

                                // Reset the twist index so the true MINGMA is
                                // recomputed from an inaccurate LAMBDA.
                                iwork[(iwbase + newfst) as usize] = TWIST_NONE;
                                cint_dlar1v_c::<F>(
                                    n,
                                    lambda,
                                    diag,
                                    dbase,
                                    offd,
                                    ebase_o,
                                    work,
                                    buf_ld,
                                    buf_lld,
                                    buf_wrk,
                                    vec,
                                    vbase + newfst * n,
                                    iwork,
                                    iwbase + newfst,
                                    sc,
                                    scb,
                                );
                            }

                            eig[(eigbase + newfst) as usize] = lambda + sigma;

                            // Recompute the gaps, allowing them only to grow.
                            if newfst > 0u32 {
                                work[(wgbase + newfst) as usize] = fmax_c::<F>(
                                    work[(wgbase + newfst) as usize],
                                    eig[(eigbase + newfst) as usize]
                                        - work[(werrbase + newfst) as usize]
                                        - eig[(eigbase + newfst - 1u32) as usize]
                                        - work[(werrbase + newfst - 1u32) as usize],
                                );
                            }
                            if newfst + 1u32 < n {
                                work[(wgbase + newfst + 1u32) as usize] = fmax_c::<F>(
                                    savgap,
                                    eig[(eigbase + newfst + 1u32) as usize]
                                        - work[(werrbase + newfst + 1u32) as usize]
                                        - eig[(eigbase + newfst) as usize]
                                        - work[(werrbase + newfst) as usize],
                                );
                            }

                            idone += 1;
                        }

                        newfst = newlst;
                    }
                    newlst += 1;
                }
                icls += 1;
            }

            let sw = old_cluster;
            old_cluster = new_cluster;
            new_cluster = sw;
            ndepth += 1;
        }
    }
    let mut info: u32 = 0;
    if idone < n || fail {
        info = 2; // the C's -2
    }
    info
}

/// **Surface 1 (production).** `_CINTdiagonalize` (`eigh.c:1450`) as a plain
/// `#[cube]` callee: no `launch`, so a foreign crate's kernel calls it inside
/// its own launch instead of paying a round trip per shell quartet.
///
/// Buffers, all caller-owned, at the given bases:
///
/// * `diag[dbase .. dbase + n]` — the diagonal, **overwritten**.
/// * `offd[ebase_o .. ebase_o + n]` — the off-diagonal in `[0 .. n-1]`,
///   `[n-1]` scratch, **overwritten**.
/// * `eig[eigbase .. eigbase + n]` — the ascending eigenvalues.
/// * `vec[vbase .. vbase + n*n]` — `vec[i*n + j]` is the `j`-th component of
///   the `i`-th eigenvector, libcint's convention.
/// * `work` — `9*n + 1` reals from `wbase`; `iwork` — `5*n` u32.
/// * `sc` — [`SC_LEN`] reals; `icnt` — [`ICNT_LEN`] u32.
/// * `eps` — C's `DBL_EPSILON`; an argument, never a literal, so the same
///   body with Reference-LAPACK's constants stays [`super::stemr`]'s.
///
/// `info[ibase]` receives libcint's `INFO` with the sign dropped (the C's
/// `-1`, `-2`, `-5`, `-6` arrive as `1`, `2`, `5`, `6`), because `#[cube]`
/// has no signed sentinel that survives all three backends identically.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cint_stemr_dev<F: Float>(
    diag: &mut Array<F>,
    dbase: u32,
    offd: &mut Array<F>,
    ebase_o: u32,
    eig: &mut Array<F>,
    eigbase: u32,
    vec: &mut Array<F>,
    vbase: u32,
    work: &mut Array<F>,
    wbase: u32,
    iwork: &mut Array<u32>,
    iwbase: u32,
    cc: &Array<F>,
    sc: &mut Array<F>,
    scb: u32,
    icnt: &mut Array<u32>,
    icb: u32,
    eps: F,
    big: F,
    info: &mut Array<u32>,
    ibase: u32,
    n: u32,
) {
    let one = F::from_int(1);
    info[ibase as usize] = 0;
    if n == 1u32 {
        eig[eigbase as usize] = diag[dbase as usize];
        vec[vbase as usize] = one;
    } else if n == 2u32 {
        cint_dlaev2_c::<F>(diag, dbase, offd, ebase_o, eig, eigbase, vec, vbase);
    } else if n > 2u32 {
        // `eigh.c:1463-1466`: buf_err = work + n, buf_gp = work + 2n + 1,
        // buf_wrk = work + 3n + 1. `wgbase` is one slot before `buf_gp`, so
        // `wgap[-1]` is addressable.
        let werrbase = wbase + n;
        let wgbase = wbase + n * 2u32;
        let wrkbase = wbase + n * 3u32 + 1u32;

        let mut rc = cint_compute_eigenvalues_c::<F>(
            n, diag, dbase, offd, ebase_o, eig, eigbase, work, werrbase, wgbase, wrkbase, eps, cc, sc, scb, icnt, icb,
        );
        if rc == 0u32 {
            rc = cint_compute_eigenvectors_c::<F>(
                n, diag, dbase, offd, ebase_o, eig, eigbase, work, werrbase, wgbase, wrkbase, vec, vbase, iwork,
                iwbase, eps, big, cc, sc, scb,
            );
        }
        info[ibase as usize] = rc;
    }
}

/// **Surface 3 (parity/debug only).** One problem, `CubeDim::new_1d(1)`.
/// Never a production path: `tests/device_cint_stemr_cpu.rs` uses it to pin
/// [`cint_stemr_dev`] against the host transcription.
#[doc(hidden)]
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cint_stemr_kernel<F: Float + CubeElement>(
    diag: &mut Array<F>,
    offd: &mut Array<F>,
    eig: &mut Array<F>,
    vec: &mut Array<F>,
    work: &mut Array<F>,
    iwork: &mut Array<u32>,
    cc: &Array<F>,
    sc: &mut Array<F>,
    icnt: &mut Array<u32>,
    info: &mut Array<u32>,
    eps: F,
    big: F,
    n: u32,
) {
    if ABSOLUTE_POS < 1usize {
        cint_stemr_dev::<F>(
            diag, 0u32, offd, 0u32, eig, 0u32, vec, 0u32, work, 0u32, iwork, 0u32, cc, sc, 0u32, icnt, 0u32, eps, big,
            info, 0u32, n,
        );
    }
}

/// **Surface 2 (standalone).** One *problem* per unit: `ABSOLUTE_POS` selects
/// it, `nptr[b] .. nptr[b+1]` is its block in the concatenated `diag`/`offd`,
/// and `vptr[b]` its block in `vec`. `B` problems cost one dispatch and one
/// batched read-back, not `B` of each.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cint_stemr_batch_kernel<F: Float + CubeElement>(
    diag: &mut Array<F>,
    offd: &mut Array<F>,
    nptr: &Array<u32>,
    vptr: &Array<u32>,
    eig: &mut Array<F>,
    vec: &mut Array<F>,
    work: &mut Array<F>,
    iwork: &mut Array<u32>,
    cc: &Array<F>,
    sc: &mut Array<F>,
    icnt: &mut Array<u32>,
    info: &mut Array<u32>,
    eps: F,
    big: F,
    nprob: u32,
    #[comptime] n_max: u32,
) {
    if ABSOLUTE_POS < nprob as usize {
        let b = ABSOLUTE_POS as u32;
        let base = nptr[b as usize];
        let n = nptr[(b + 1u32) as usize] - base;
        cint_stemr_dev::<F>(
            diag,
            base,
            offd,
            base,
            eig,
            base,
            vec,
            vptr[b as usize],
            work,
            b * (9u32 * n_max + 1u32),
            iwork,
            b * (5u32 * n_max),
            cc,
            sc,
            b * SC_LEN,
            icnt,
            b * ICNT_LEN,
            eps,
            big,
            info,
            b,
            n,
        );
    }
}
