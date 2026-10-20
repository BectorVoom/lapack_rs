//! K8: the MRRR leaf kernels — `DLAEV2`, `DLARRK`, `DLANEG` and `DLAR1V`.
//!
//! Device transcriptions of Reference-LAPACK 3.12.1 `dlaev2.f`, `dlarrk.f`,
//! `dlaneg.f`, `dlar1v.f`, and, with different constants, of the matching
//! `#else`-branch routines of libcint 6.1.3 `src/eigh.c`. **Same bodies,
//! different constants: every tolerance is an argument**, never a literal, so
//! one body serves both oracles (`docs/cintx-lapack-blas-adoption-plan.md`
//! §5 and §7).
//!
//! Bit contract: one unit owns one (problem, index) pair, so each unit
//! replays the host chain in the reference's operand order and the result is
//! independent of geometry, vector width, planes and checked launches.
//!
//! Two backend constraints are designed in rather than discovered:
//!
//! * **Ascending `while` counters only.** cubecl-cpu 0.10 cannot lower a loop
//!   whose counter is decremented (`tests/cubecl_cpu_limits.rs`), and
//!   `dlaneg.f`'s part II and three of `dlar1v.f`'s loops descend. Each is
//!   rewritten as `for t in 0..count { let j = top - t; .. }`, which visits
//!   the same indices in the same order with the same operands and so cannot
//!   change a value. The reference loop label is named at each site.
//! * **`DISNAN` is `x /= x`.** `SAWNAN = DISNAN( X )` is written `x != x`,
//!   which is what `dlaisnan.f` computes and what the host transcription
//!   does; `F::is_nan` returns cubecl's vectorised `WithScalar<bool>`, not a
//!   `bool`, and comparing a value with itself lowers to the backend
//!   predicate on every target including signalling NaNs.
//! * **Scalar unary minus only.** `-(half * rt)` and friends are scalar and
//!   exact; AGENTS.md's ban is on *vector* unary minus (HIP's C rejects it)
//!   and on the `0 - x` workaround, which loses the sign of zero.

// `let mut x = zero;` before branch assignment is how a #[cube] kernel
// declares a variable; rustc sees the initial value as unused.
#![allow(unused_assignments)]

use cubecl::prelude::*;

use super::gf::fmax_c;

/// `DLAEV2` (`dlaev2.f`): eigenvalues and the first eigenvector of
/// `[[a, b], [b, c]]`. Writes `RT1, RT2, CS1, SN1` to `out[slot..slot+4]`.
///
/// `|RT1| >= |RT2|`; `[CS1, SN1]` is the eigenvector for `RT1`. The
/// reference's "Order of execution important" on `RT2` (`dlaev2.f:179`) is
/// load-bearing: `(ACMX/RT1)*ACMN - (B/RT1)*B` keeps its parenthesisation.
#[cube]
pub fn dlaev2_c<F: Float>(a: F, b: F, c: F, out: &mut Array<F>, slot: u32) {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let two = F::from_int(2);
    // `HALF = 0.5D0` built exactly from integers: `F::new` takes an `f32`
    // in cubecl 0.10, and `one / two` is exact in every IEEE format.
    let half = F::from_int(1) / two;

    let sm = a + c;
    let df = a - c;
    let adf = F::abs(df);
    let tb = b + b;
    let ab = F::abs(tb);

    // IF( ABS(A).GT.ABS(C) ) THEN ACMX = A; ACMN = C ELSE ACMX = C; ACMN = A
    let mut acmx = c;
    let mut acmn = a;
    if F::abs(a) > F::abs(c) {
        acmx = a;
        acmn = c;
    }

    // RT = ADF*SQRT(1+(AB/ADF)**2) / AB*SQRT(1+(ADF/AB)**2) / AB*SQRT(2)
    let mut rt = ab * F::sqrt(two);
    if adf > ab {
        let t = ab / adf;
        rt = adf * F::sqrt(one + t * t);
    } else if adf < ab {
        let t = adf / ab;
        rt = ab * F::sqrt(one + t * t);
    }

    let mut rt1 = half * rt;
    let mut rt2 = -(half * rt);
    // SGN1 is a flag, not an integer: its only use is `SGN1.EQ.SGN2`.
    let mut sgn1_pos: bool = true;
    if sm < zero {
        rt1 = half * (sm - rt);
        sgn1_pos = false;
        // Order of execution important (dlaev2.f:179).
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if sm > zero {
        rt1 = half * (sm + rt);
        sgn1_pos = true;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    }

    // IF( DF.GE.ZERO ) THEN CS = DF + RT; SGN2 = 1 ELSE CS = DF - RT; SGN2 = -1
    let mut cs = df - rt;
    let mut sgn2_pos: bool = false;
    if df >= zero {
        cs = df + rt;
        sgn2_pos = true;
    }

    let acs = F::abs(cs);
    // The `AB.EQ.ZERO` branch's assignment, as the initialise-then-overwrite
    // form the Conditionals manual recommends.
    let mut cs1 = one;
    let mut sn1 = zero;
    if acs > ab {
        let ct = -tb / cs;
        sn1 = one / F::sqrt(one + ct * ct);
        cs1 = ct * sn1;
    } else if ab != zero {
        let tn = -cs / tb;
        cs1 = one / F::sqrt(one + tn * tn);
        sn1 = tn * cs1;
    }

    if sgn1_pos == sgn2_pos {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }

    out[slot as usize] = rt1;
    out[(slot + 1) as usize] = rt2;
    out[(slot + 2) as usize] = cs1;
    out[(slot + 3) as usize] = sn1;
}

/// `dlarrk.f:193`: `ITMAX = INT((LOG(TNORM+PIVMIN)-LOG(PIVMIN))/LOG(TWO))+2`.
///
/// `log_two` is passed in because gfortran constant-folds `LOG(TWO)`, so only
/// the two data-dependent logarithms run on the device.
///
/// **Named risk.** `F::ln` lowers to the backend's `log`, which is not
/// required to agree with the glibc `log` gfortran calls; a disagreement that
/// crosses an integer boundary changes the halving count and so changes `W`.
/// `tests/device_stemr_cpu.rs` compares this against the host's value **as a
/// `u32`, for every problem in the corpus**, so a libm difference fails
/// loudly instead of showing up as a mysterious one-ULP eigenvalue. Callers
/// on the host pass the host's `itmax` and skip this entirely; only a fused
/// inline callee with device-side `gl`/`gu` needs it.
#[cube]
pub fn larrk_itmax<F: Float>(tnorm: F, pivmin: F, log_two: F) -> u32 {
    let q = (F::ln(tnorm + pivmin) - F::ln(pivmin)) / log_two;
    u32::cast_from(F::floor(q)) + 2u32
}

/// `DLARRK` (`dlarrk.f`): bisection for the `iw`-th eigenvalue (1-based) of
/// the tridiagonal `(d[base..base+n], e2[base..base+n-1])` inside `[gl, gu]`.
///
/// Returns `W`, writes `WERR` to `werr[slot]` and `INFO` (0 converged,
/// 1 = the reference's `-1`) to `info[slot]`.
///
/// `itmax`, `eps`, `pivmin` and `reltol` are arguments: Path R passes
/// Reference-LAPACK's, Path C passes libcint's. `FUDGE = TWO` and
/// `HALF = 0.5D0` are the reference's own `PARAMETER`s and stay literal.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn dlarrk_c<F: Float>(
    d: &Array<F>,
    e2: &Array<F>,
    base: u32,
    n: u32,
    iw: u32,
    gl: F,
    gu: F,
    reltol: F,
    pivmin: F,
    eps: F,
    itmax: u32,
    werr: &mut Array<F>,
    info: &mut Array<u32>,
    slot: u32,
) -> F {
    let zero = F::from_int(0);
    let two = F::from_int(2);
    // `HALF = 0.5D0` built exactly from integers: `F::new` takes an `f32`
    // in cubecl 0.10, and `one / two` is exact in every IEEE format.
    let half = F::from_int(1) / two;
    let fudge = two;

    // Quick return: N <= 0 -> INFO = 0. The caller never passes n == 0; the
    // guard is the reference's and is kept so the branch counts match.
    let mut w = zero;
    if n > 0 {
        let tnorm = fmax_c::<F>(F::abs(gl), F::abs(gu));
        let rtoli = reltol;
        let atoli = fudge * two * pivmin;
        let mut inf = 1u32; // INFO = -1

        // LEFT  = GL - FUDGE*TNORM*EPS*N - FUDGE*TWO*PIVMIN
        // RIGHT = GU + FUDGE*TNORM*EPS*N + FUDGE*TWO*PIVMIN
        // Parenthesised left to right as gfortran evaluates it, with N
        // converted to double last: getting this wrong is a silent one-ULP
        // bracket shift.
        let pert = ((fudge * tnorm) * eps) * F::cast_from(n);
        let ftp = fudge * two * pivmin;
        let mut left = gl - pert - ftp;
        let mut right = gu + pert + ftp;

        let mut it = 0u32;
        let mut step = 0u32;
        let bound = itmax + 2u32; // the reference exits on IT > ITMAX
        while step < bound {
            let tmp1 = F::abs(right - left);
            let tmp2 = fmax_c::<F>(F::abs(right), F::abs(left));
            // IF( TMP1.LT.MAX( ATOLI, PIVMIN, RTOLI*TMP2 ) ) -> converged
            let tol = fmax_c::<F>(fmax_c::<F>(atoli, pivmin), rtoli * tmp2);
            let mut done: bool = false;
            if tmp1 < tol {
                inf = 0u32;
                done = true;
            }
            if it > itmax {
                done = true;
            }
            if done {
                step = bound; // leave the bounded loop
            } else {
                it += 1u32;
                let mid = half * (left + right);
                let mut negcnt = 0u32;
                let mut t = d[base as usize] - mid;
                if F::abs(t) < pivmin {
                    t = -pivmin;
                }
                if t <= zero {
                    negcnt += 1u32;
                }
                let mut i = 1u32;
                while i < n {
                    t = d[(base + i) as usize] - e2[(base + i - 1u32) as usize] / t - mid;
                    if F::abs(t) < pivmin {
                        t = -pivmin;
                    }
                    if t <= zero {
                        negcnt += 1u32;
                    }
                    i += 1u32;
                }
                if negcnt >= iw {
                    right = mid;
                } else {
                    left = mid;
                }
                step += 1u32;
            }
        }

        w = half * (left + right);
        werr[slot as usize] = half * F::abs(right - left);
        info[slot as usize] = inf;
    } else {
        info[slot as usize] = 0u32;
    }
    w
}

/// `DLANEG` (`dlaneg.f`): negcount of `L D L^T - sigma I` with twist index
/// `r` (1-based, `1 <= r <= n`).
///
/// `d`, `lld` are this problem's block at `base`. `blklen` is the reference's
/// `BLKLEN = 128`, `#[comptime]` so the `NaN`-rerun blocks fold.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn dlaneg_c<F: Float>(
    d: &Array<F>,
    lld: &Array<F>,
    base: u32,
    n: u32,
    sigma: F,
    r: u32,
    #[comptime] blklen: u32,
) -> u32 {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let mut negcnt = 0u32;

    // I) upper part: DO 210 BJ = 1, R-1, BLKLEN (ascending already).
    let mut t = -sigma;
    let mut bj = 1u32;
    while bj + 1u32 <= r {
        let mut hi = bj + blklen - 1u32;
        if hi + 1u32 > r {
            hi = r - 1u32;
        }
        let mut neg1 = 0u32;
        let bsav = t;
        let mut j = bj;
        while j <= hi {
            let dplus = d[(base + j - 1u32) as usize] + t;
            if dplus < zero {
                neg1 += 1u32;
            }
            let tmp = t / dplus;
            t = tmp * lld[(base + j - 1u32) as usize] - sigma;
            j += 1u32;
        }
        // A NaN can only follow a zero pivot after an infinite pivot; the
        // reference reruns the block substituting 1 for Inf/Inf. These reruns
        // are not dead code on a nearly-degenerate tridiagonal.
        if t != t {
            neg1 = 0u32;
            t = bsav;
            let mut j2 = bj;
            while j2 <= hi {
                let dplus = d[(base + j2 - 1u32) as usize] + t;
                if dplus < zero {
                    neg1 += 1u32;
                }
                let mut tmp = t / dplus;
                if tmp != tmp {
                    tmp = one;
                }
                t = tmp * lld[(base + j2 - 1u32) as usize] - sigma;
                j2 += 1u32;
            }
        }
        negcnt += neg1;
        bj += blklen;
    }

    // II) lower part. The reference counter DECREMENTS
    // (`DO 230 BJ = N-1, R, -BLKLEN`); same visit order from an ASCENDING
    // counter: BJ = (N-1) - b*BLKLEN, J = BJ - s.
    let mut p = d[(base + n - 1u32) as usize] - sigma;
    if n > r {
        let nblk = (n - 1u32 - r) / blklen + 1u32;
        let mut b = 0u32;
        while b < nblk {
            let top = (n - 1u32) - b * blklen; // BJ
            let mut lo = r; // MAX(BJ-BLKLEN+1, R)
            if top + 1u32 > blklen {
                let cand = top + 1u32 - blklen;
                if cand > r {
                    lo = cand;
                }
            }
            let cnt = top - lo + 1u32;
            let mut neg2 = 0u32;
            let bsav = p;
            let mut s = 0u32;
            while s < cnt {
                let j = top - s;
                let dminus = lld[(base + j - 1u32) as usize] + p;
                if dminus < zero {
                    neg2 += 1u32;
                }
                let tmp = p / dminus;
                p = tmp * d[(base + j - 1u32) as usize] - sigma;
                s += 1u32;
            }
            if p != p {
                neg2 = 0u32;
                p = bsav;
                let mut s2 = 0u32;
                while s2 < cnt {
                    let j = top - s2;
                    let dminus = lld[(base + j - 1u32) as usize] + p;
                    if dminus < zero {
                        neg2 += 1u32;
                    }
                    let mut tmp = p / dminus;
                    if tmp != tmp {
                        tmp = one;
                    }
                    p = tmp * d[(base + j - 1u32) as usize] - sigma;
                    s2 += 1u32;
                }
            }
            negcnt += neg2;
            b += 1u32;
        }
    }

    // III) twist index. T was shifted by SIGMA initially.
    let gamma = (t + sigma) + p;
    if gamma < zero {
        negcnt += 1u32;
    }
    negcnt
}

/// `DLAR1V` (`dlar1v.f`): the FP vector `z` for `lambda`, the twist index
/// `r`, `MINGMA`, `NEGCNT`, `NRMINV`, `RESID` and `RQCORR`.
///
/// `work` is the reference's `WORK` carved as `lplus = work[0..n]`,
/// `uminus = work[n..2n]`, `s = work[2n..3n]`, `p = work[3n..4n]`, all
/// relative to `wbase`; the carve is kept so the transcription reads against
/// the source. Scalars come back in `out[slot..slot+4]` as
/// `[MINGMA, NRMINV, RESID, RQCORR]`; `isuppz[slot2..slot2+2]` gets `[B1, BN]`
/// and `iout[slot2..slot2+2]` the twist index and `NEGCNT`.
///
/// `NEGCNT = -1` when `WANTNC` is false: the `u32` here carries `0` plus the
/// flag, and the host driver maps it back, because `#[cube]` has no signed
/// sentinel that survives all three backends identically. `iout[slot2 + 1]`
/// is only read when `wantnc`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn dlar1v_c<F: Float>(
    d: &Array<F>,
    l: &Array<F>,
    ld: &Array<F>,
    lld: &Array<F>,
    base: u32,
    n: u32,
    b1: u32,
    bn: u32,
    lambda: F,
    r_in: u32,
    gaptol: F,
    pivmin: F,
    eps: F,
    wantnc: bool,
    z: &mut Array<F>,
    zbase: u32,
    work: &mut Array<F>,
    wbase: u32,
    isuppz: &mut Array<u32>,
    out: &mut Array<F>,
    iout: &mut Array<u32>,
    slot: u32,
    slot2: u32,
) {
    let zero = F::from_int(0);
    let one = F::from_int(1);

    // The reference's `WORK` carve, kept 1-based exactly as `dlar1v.f`
    // writes it (`INDLPL = 0`, `INDUMN = N`, `INDS = 2*N+1`, `INDP = 3*N+1`)
    // and as [`crate::larr::dlar1v`] transcribes it, so both sides index by
    // the same expression. `WORK( INDS+I )` is `work[inds + i - 1]`, while
    // `WORK( INDS+B1-1 )` is `work[inds + b1 - 2]` — the two forms differ by
    // one and mixing them silently aliases `S(1)` onto `WORK( INDS )`.
    let indlpl = wbase;
    let indumn = wbase + n;
    let inds = wbase + 2u32 * n + 1u32;
    let indp = wbase + 3u32 * n + 1u32;

    let mut r1 = r_in;
    let mut r2 = r_in;
    if r_in == 0u32 {
        r1 = b1;
        r2 = bn;
    }

    if b1 == 1u32 {
        work[(inds - 1u32) as usize] = zero;
    } else {
        work[(inds + b1 - 2u32) as usize] = lld[(base + b1 - 2u32) as usize];
    }

    // Stationary transform (differential form) up to R2.
    let mut neg1 = 0u32;
    let mut s = work[(inds + b1 - 2u32) as usize] - lambda;
    let mut i = b1;
    while i < r1 {
        let dplus = d[(base + i - 1u32) as usize] + s;
        work[(indlpl + i - 1u32) as usize] = ld[(base + i - 1u32) as usize] / dplus;
        if dplus < zero {
            neg1 += 1u32;
        }
        work[(inds + i - 1u32) as usize] = s * work[(indlpl + i - 1u32) as usize] * l[(base + i - 1u32) as usize];
        s = work[(inds + i - 1u32) as usize] - lambda;
        i += 1u32;
    }
    let mut sawnan1: bool = s != s;
    if !sawnan1 {
        let mut i2 = r1;
        while i2 < r2 {
            let dplus = d[(base + i2 - 1u32) as usize] + s;
            work[(indlpl + i2 - 1u32) as usize] = ld[(base + i2 - 1u32) as usize] / dplus;
            work[(inds + i2 - 1u32) as usize] =
                s * work[(indlpl + i2 - 1u32) as usize] * l[(base + i2 - 1u32) as usize];
            s = work[(inds + i2 - 1u32) as usize] - lambda;
            i2 += 1u32;
        }
        sawnan1 = s != s;
    }
    if sawnan1 {
        // Slower version: floor the pivot, substitute LLD for a zero LPLUS.
        neg1 = 0u32;
        s = work[(inds + b1 - 2u32) as usize] - lambda;
        let mut i3 = b1;
        while i3 < r1 {
            let mut dplus = d[(base + i3 - 1u32) as usize] + s;
            if F::abs(dplus) < pivmin {
                dplus = -pivmin;
            }
            work[(indlpl + i3 - 1u32) as usize] = ld[(base + i3 - 1u32) as usize] / dplus;
            if dplus < zero {
                neg1 += 1u32;
            }
            work[(inds + i3 - 1u32) as usize] =
                s * work[(indlpl + i3 - 1u32) as usize] * l[(base + i3 - 1u32) as usize];
            if work[(indlpl + i3 - 1u32) as usize] == zero {
                work[(inds + i3 - 1u32) as usize] = lld[(base + i3 - 1u32) as usize];
            }
            s = work[(inds + i3 - 1u32) as usize] - lambda;
            i3 += 1u32;
        }
        let mut i4 = r1;
        while i4 < r2 {
            let mut dplus = d[(base + i4 - 1u32) as usize] + s;
            if F::abs(dplus) < pivmin {
                dplus = -pivmin;
            }
            work[(indlpl + i4 - 1u32) as usize] = ld[(base + i4 - 1u32) as usize] / dplus;
            work[(inds + i4 - 1u32) as usize] =
                s * work[(indlpl + i4 - 1u32) as usize] * l[(base + i4 - 1u32) as usize];
            if work[(indlpl + i4 - 1u32) as usize] == zero {
                work[(inds + i4 - 1u32) as usize] = lld[(base + i4 - 1u32) as usize];
            }
            s = work[(inds + i4 - 1u32) as usize] - lambda;
            i4 += 1u32;
        }
    }

    // Progressive transform down to R1. Reference: DO 80 I = BN-1, R1, -1.
    // Ascending counter, identical visit order.
    let mut neg2 = 0u32;
    work[(indp + bn - 2u32) as usize] = d[(base + bn - 1u32) as usize] - lambda;
    let mut cnt = 0u32;
    if bn > r1 {
        cnt = bn - r1;
    }
    let mut t = 0u32;
    while t < cnt {
        let i5 = (bn - 1u32) - t;
        let dminus = lld[(base + i5 - 1u32) as usize] + work[(indp + i5 - 1u32) as usize];
        let tmp = d[(base + i5 - 1u32) as usize] / dminus;
        if dminus < zero {
            neg2 += 1u32;
        }
        work[(indumn + i5 - 1u32) as usize] = l[(base + i5 - 1u32) as usize] * tmp;
        work[(indp + i5 - 2u32) as usize] = work[(indp + i5 - 1u32) as usize] * tmp - lambda;
        t += 1u32;
    }
    let pr1 = work[(indp + r1 - 2u32) as usize];
    let sawnan2: bool = pr1 != pr1;
    if sawnan2 {
        neg2 = 0u32;
        let mut t2 = 0u32;
        while t2 < cnt {
            let i6 = (bn - 1u32) - t2;
            let mut dminus = lld[(base + i6 - 1u32) as usize] + work[(indp + i6 - 1u32) as usize];
            if F::abs(dminus) < pivmin {
                dminus = -pivmin;
            }
            let tmp = d[(base + i6 - 1u32) as usize] / dminus;
            if dminus < zero {
                neg2 += 1u32;
            }
            work[(indumn + i6 - 1u32) as usize] = l[(base + i6 - 1u32) as usize] * tmp;
            work[(indp + i6 - 2u32) as usize] = work[(indp + i6 - 1u32) as usize] * tmp - lambda;
            if tmp == zero {
                work[(indp + i6 - 2u32) as usize] = d[(base + i6 - 1u32) as usize] - lambda;
            }
            t2 += 1u32;
        }
    }

    // Largest (in magnitude) diagonal element of the inverse, R1..R2.
    let mut mingma = work[(inds + r1 - 2u32) as usize] + work[(indp + r1 - 2u32) as usize];
    if mingma < zero {
        neg1 += 1u32;
    }
    let mut negcnt = 0u32;
    if wantnc {
        negcnt = neg1 + neg2;
    }
    if F::abs(mingma) == zero {
        mingma = eps * work[(inds + r1 - 2u32) as usize];
    }
    let mut r = r1;
    let mut i7 = r1;
    while i7 < r2 {
        let mut tmp = work[(inds + i7 - 1u32) as usize] + work[(indp + i7 - 1u32) as usize];
        if tmp == zero {
            tmp = eps * work[(inds + i7 - 1u32) as usize];
        }
        if F::abs(tmp) <= F::abs(mingma) {
            mingma = tmp;
            r = i7 + 1u32;
        }
        i7 += 1u32;
    }

    // Solve N^T v = e_r.
    let mut lo = b1;
    let mut hi = bn;
    z[(zbase + r - 1u32) as usize] = one;
    let mut ztz = one;

    // Upwards from R. Reference: DO 210/230 I = R-1, B1, -1. Ascending.
    // The early exit sets ISUPPZ and STOPS accumulating ZTZ: the reference
    // jumps out of the loop, it does not zero the tail. An `if` that merely
    // skipped the store would keep accumulating and change NRMINV.
    let mut ucnt = 0u32;
    if r > b1 {
        ucnt = r - b1;
    }
    let mut u = 0u32;
    let mut done: bool = false;
    while u < ucnt {
        if !done {
            let i8 = (r - 1u32) - u;
            let mut zi = -(work[(indlpl + i8 - 1u32) as usize] * z[(zbase + i8) as usize]);
            if sawnan1 || sawnan2 {
                // Slower loop: Z(I+1) == 0 -> Z(I) = -(LD(I+1)/LD(I))*Z(I+2)
                if z[(zbase + i8) as usize] == zero {
                    zi =
                        -(ld[(base + i8) as usize] / ld[(base + i8 - 1u32) as usize]) * z[(zbase + i8 + 1u32) as usize];
                }
            }
            z[(zbase + i8 - 1u32) as usize] = zi;
            if (F::abs(zi) + F::abs(z[(zbase + i8) as usize])) * F::abs(ld[(base + i8 - 1u32) as usize]) < gaptol {
                z[(zbase + i8 - 1u32) as usize] = zero;
                lo = i8 + 1u32;
                done = true;
            } else {
                ztz = ztz + zi * zi;
            }
        }
        u += 1u32;
    }

    // Downwards from R. Reference: DO 250/270 I = R, BN-1 (already ascending).
    let mut i9 = r;
    let mut done2: bool = false;
    while i9 < bn {
        if !done2 {
            let mut zn = -(work[(indumn + i9 - 1u32) as usize] * z[(zbase + i9 - 1u32) as usize]);
            if sawnan1 || sawnan2 {
                if z[(zbase + i9 - 1u32) as usize] == zero {
                    zn = -(ld[(base + i9 - 2u32) as usize] / ld[(base + i9 - 1u32) as usize])
                        * z[(zbase + i9 - 2u32) as usize];
                }
            }
            z[(zbase + i9) as usize] = zn;
            if (F::abs(z[(zbase + i9 - 1u32) as usize]) + F::abs(zn)) * F::abs(ld[(base + i9 - 1u32) as usize]) < gaptol
            {
                z[(zbase + i9) as usize] = zero;
                hi = i9;
                done2 = true;
            } else {
                ztz = ztz + zn * zn;
            }
        }
        i9 += 1u32;
    }

    let tmp = one / ztz;
    let nrminv = F::sqrt(tmp);
    out[slot as usize] = mingma;
    out[(slot + 1u32) as usize] = nrminv;
    out[(slot + 2u32) as usize] = F::abs(mingma) * nrminv;
    out[(slot + 3u32) as usize] = mingma * tmp;
    isuppz[slot2 as usize] = lo;
    isuppz[(slot2 + 1u32) as usize] = hi;
    iout[slot2 as usize] = r;
    // The Conditionals manual's statement form, not an `if` expression.
    let mut nc = 0u32;
    if wantnc {
        nc = negcnt;
    }
    iout[(slot2 + 1u32) as usize] = nc;
}

// ---------------------------------------------------------------------------
// Batched launchers. One *problem* per unit; `ABSOLUTE_POS` selects it and
// `nptr[b]..nptr[b+1]` is its block in the concatenated buffers, so `B`
// problems cost one dispatch and one batched read-back, not `B` of each.
//
// The inner loops are `O(n)` long and each unit owns a whole problem, so
// problem-major (`base = nptr[b]`, consecutive `j`) is the layout that wins —
// the deliberate exception to the stride-1 rule that
// `tests/device_sector_model.rs` records.
// ---------------------------------------------------------------------------

/// Batched `DLAEV2`: unit `b` reads `abc[3b..3b+3]` and writes
/// `out[4b..4b+4]`.
#[cube(launch, launch_unchecked)]
pub fn laev2_batch_kernel<F: Float + CubeElement>(abc: &Array<F>, out: &mut Array<F>, nprob: u32) {
    if ABSOLUTE_POS < nprob as usize {
        let b = ABSOLUTE_POS as u32;
        dlaev2_c::<F>(
            abc[(3u32 * b) as usize],
            abc[(3u32 * b + 1u32) as usize],
            abc[(3u32 * b + 2u32) as usize],
            out,
            4u32 * b,
        );
    }
}

/// Batched `DLARRK`: unit `u` decodes `(problem b, index iw)` from
/// `ABSOLUTE_POS`, with `nw` indices per problem.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn larrk_batch_kernel<F: Float + CubeElement>(
    d: &Array<F>,
    e2: &Array<F>,
    nptr: &Array<u32>,
    gl: &Array<F>,
    gu: &Array<F>,
    consts: &Array<F>,
    itmax: &Array<u32>,
    w: &mut Array<F>,
    werr: &mut Array<F>,
    info: &mut Array<u32>,
    nprob: u32,
    nw: u32,
) {
    if ABSOLUTE_POS < (nprob * nw) as usize {
        let u = ABSOLUTE_POS as u32;
        let b = u / nw;
        let k = u % nw;
        let base = nptr[b as usize];
        let n = nptr[(b + 1u32) as usize] - base;
        if k < n {
            let slot = b * nw + k;
            w[slot as usize] = dlarrk_c::<F>(
                d,
                e2,
                base,
                n,
                k + 1u32,
                gl[b as usize],
                gu[b as usize],
                consts[0],
                consts[1],
                consts[2],
                itmax[b as usize],
                werr,
                info,
                slot,
            );
        }
    }
}

/// Batched `DLANEG`: unit `b` runs one problem with twist index `r[b]`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn laneg_batch_kernel<F: Float + CubeElement>(
    d: &Array<F>,
    lld: &Array<F>,
    nptr: &Array<u32>,
    sigma: &Array<F>,
    r: &Array<u32>,
    negcnt: &mut Array<u32>,
    nprob: u32,
    #[comptime] blklen: u32,
) {
    if ABSOLUTE_POS < nprob as usize {
        let b = ABSOLUTE_POS as u32;
        let base = nptr[b as usize];
        let n = nptr[(b + 1u32) as usize] - base;
        negcnt[b as usize] = dlaneg_c::<F>(d, lld, base, n, sigma[b as usize], r[b as usize], blklen);
    }
}

/// Batched `DLAR1V`: unit `b` runs one problem into `z[zptr[b]..]` with
/// scratch `work[4 * nptr[b]..]`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn lar1v_batch_kernel<F: Float + CubeElement>(
    d: &Array<F>,
    l: &Array<F>,
    ld: &Array<F>,
    lld: &Array<F>,
    nptr: &Array<u32>,
    lambda: &Array<F>,
    r_in: &Array<u32>,
    consts: &Array<F>,
    wantnc: &Array<u32>,
    z: &mut Array<F>,
    work: &mut Array<F>,
    isuppz: &mut Array<u32>,
    out: &mut Array<F>,
    iout: &mut Array<u32>,
    nprob: u32,
) {
    if ABSOLUTE_POS < nprob as usize {
        let b = ABSOLUTE_POS as u32;
        let base = nptr[b as usize];
        let n = nptr[(b + 1u32) as usize] - base;
        dlar1v_c::<F>(
            d,
            l,
            ld,
            lld,
            base,
            n,
            1u32,
            n,
            lambda[b as usize],
            r_in[b as usize],
            consts[0],
            consts[1],
            consts[2],
            wantnc[b as usize] != 0u32,
            z,
            base,
            work,
            4u32 * base,
            isuppz,
            out,
            iout,
            4u32 * b,
            2u32 * b,
        );
    }
}

/// `larrk_itmax` on the device, one problem per unit: the integer gate of
/// [`larrk_itmax`].
#[cube(launch, launch_unchecked)]
pub fn larrk_itmax_kernel<F: Float + CubeElement>(
    tnorm: &Array<F>,
    pivmin: &Array<F>,
    log_two: F,
    out: &mut Array<u32>,
    nprob: u32,
) {
    if ABSOLUTE_POS < nprob as usize {
        let b = ABSOLUTE_POS as u32;
        out[b as usize] = larrk_itmax::<F>(tnorm[b as usize], pivmin[b as usize], log_two);
    }
}
