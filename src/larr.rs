//! MRRR leaf routines (Reference-LAPACK v3.12.1).
//!
//! Transcribed from `SRC/dlarrc.f`, `SRC/dlarrk.f`, `SRC/dlaneg.f`,
//! `SRC/dlar1v.f`, `SRC/dlarra.f`, `SRC/dlarrr.f`, `SRC/dlarrb.f`,
//! `SRC/dlarrf.f`, `SRC/dlarrj.f`.
//!
//! Every value is the one gfortran computes for the pinned sources: operand
//! order, parenthesisation, `MAX`/`MIN`/`ABS` lowering and the reference's
//! zero-skips and NaN reruns are kept verbatim. `DISNAN(X)` is `X /= X`,
//! i.e. [`f64::is_nan`].
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::aux::{PREC, fmax};
use crate::ptr::D;

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;
const TWO: f64 = 2.0;
const HALF: f64 = 0.5;

/// `DLARRC( JOBT, N, VL, VU, D, E, PIVMIN, EIGCNT, LCNT, RCNT, INFO )`.
///
/// `matt` is `LSAME( JOBT, 'T' )`. Returns `(eigcnt, lcnt, rcnt, info)`.
///
/// # Safety
/// `d` has `n` entries, `e` at least `n - 1`.
pub(crate) unsafe fn dlarrc(matt: bool, n: usize, vl: f64, vu: f64, d: D, e: D) -> (i32, i32, i32, i32) {
    let info = 0;
    let mut lcnt: i32 = 0;
    let mut rcnt: i32 = 0;
    if n == 0 {
        return (0, 0, 0, info);
    }
    if matt {
        // Sturm sequence count on T.
        let mut lpivot = d.at(0) - vl;
        let mut rpivot = d.at(0) - vu;
        if lpivot <= ZERO {
            lcnt += 1;
        }
        if rpivot <= ZERO {
            rcnt += 1;
        }
        for i in 1..=n - 1 {
            let tmp = e.at(i - 1) * e.at(i - 1); // E(I)**2
            lpivot = (d.at(i) - vl) - tmp / lpivot;
            rpivot = (d.at(i) - vu) - tmp / rpivot;
            if lpivot <= ZERO {
                lcnt += 1;
            }
            if rpivot <= ZERO {
                rcnt += 1;
            }
        }
    } else {
        // Sturm sequence count on L D L^T.
        let mut sl = -vl;
        let mut su = -vu;
        for i in 1..=n - 1 {
            let lpivot = d.at(i - 1) + sl;
            let rpivot = d.at(i - 1) + su;
            if lpivot <= ZERO {
                lcnt += 1;
            }
            if rpivot <= ZERO {
                rcnt += 1;
            }
            let tmp = e.at(i - 1) * d.at(i - 1) * e.at(i - 1);

            let tmp2 = tmp / lpivot;
            if tmp2 == ZERO {
                sl = tmp - vl;
            } else {
                sl = sl * tmp2 - vl;
            }

            let tmp2 = tmp / rpivot;
            if tmp2 == ZERO {
                su = tmp - vu;
            } else {
                su = su * tmp2 - vu;
            }
        }
        let lpivot = d.at(n - 1) + sl;
        let rpivot = d.at(n - 1) + su;
        if lpivot <= ZERO {
            lcnt += 1;
        }
        if rpivot <= ZERO {
            rcnt += 1;
        }
    }
    (rcnt - lcnt, lcnt, rcnt, info)
}

/// `dlarrk.f:193`: `ITMAX = INT((LOG(TNORM+PIVMIN)-LOG(PIVMIN))/LOG(TWO))+2`.
///
/// Fortran `INT` truncates toward zero; the quotient is non-negative for every
/// caller (`TNORM >= 0`, `PIVMIN > 0`), so `trunc` and `floor` agree, but the
/// reference's `INT` is written out.
pub(crate) fn larrk_itmax(tnorm: f64, pivmin: f64) -> i32 {
    (((tnorm + pivmin).ln() - pivmin.ln()) / TWO.ln()) as i32 + 2
}

/// `DLARRK( N, IW, GL, GU, D, E2, PIVMIN, RELTOL, W, WERR, INFO )`.
///
/// Returns `(w, werr, info)`.
///
/// # Safety
/// `d` has `n` entries, `e2` at least `n - 1`.
pub(crate) unsafe fn dlarrk(
    n: usize,
    iw: i32,
    gl: f64,
    gu: f64,
    d: D,
    e2: D,
    pivmin: f64,
    reltol: f64,
) -> (f64, f64, i32) {
    // Quick return if possible.
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    const FUDGE: f64 = TWO;
    let eps = PREC; // DLAMCH( 'P' )
    let tnorm = fmax(gl.abs(), gu.abs());
    let rtoli = reltol;
    let atoli = FUDGE * TWO * pivmin;

    let itmax = larrk_itmax(tnorm, pivmin);

    let mut info: i32 = -1;

    let mut left = gl - FUDGE * tnorm * eps * n as f64 - FUDGE * TWO * pivmin;
    let mut right = gu + FUDGE * tnorm * eps * n as f64 + FUDGE * TWO * pivmin;
    let mut it: i32 = 0;

    loop {
        // Check if interval converged or maximum number of iterations reached.
        let tmp1 = (right - left).abs();
        let tmp2 = fmax(right.abs(), left.abs());
        if tmp1 < fmax(fmax(atoli, pivmin), rtoli * tmp2) {
            info = 0;
            break;
        }
        if it > itmax {
            break;
        }

        // Count number of negative pivots for mid-point.
        it += 1;
        let mid = HALF * (left + right);
        let mut negcnt: i32 = 0;
        let mut tmp1 = d.at(0) - mid;
        if tmp1.abs() < pivmin {
            tmp1 = -pivmin;
        }
        if tmp1 <= ZERO {
            negcnt += 1;
        }

        for i in 2..=n {
            tmp1 = d.at(i - 1) - e2.at(i - 2) / tmp1 - mid;
            if tmp1.abs() < pivmin {
                tmp1 = -pivmin;
            }
            if tmp1 <= ZERO {
                negcnt += 1;
            }
        }

        if negcnt >= iw {
            right = mid;
        } else {
            left = mid;
        }
    }

    // Converged or maximum number of iterations reached.
    (HALF * (left + right), HALF * (right - left).abs(), info)
}

/// `DLANEG( N, D, LLD, SIGMA, PIVMIN, R )`: the negcount of
/// `L D L^T - sigma I` with twist index `r` (1-based).
///
/// # Safety
/// `d` has `n` entries, `lld` at least `n - 1`.
pub(crate) unsafe fn dlaneg(n: usize, d: D, lld: D, sigma: f64, _pivmin: f64, r: usize) -> i32 {
    // `dlaneg.f:142`. Not a tuning parameter: it bounds how far a NaN can
    // propagate before the slower rerun catches it.
    const BLKLEN: usize = 128;
    let mut negcnt: i32 = 0;

    // I) upper part: L D L^T - SIGMA I = L+ D+ L+^T
    let mut t = -sigma;
    let mut bj = 1usize;
    while bj <= r.saturating_sub(1) {
        let hi = (bj + BLKLEN - 1).min(r - 1);
        let mut neg1: i32 = 0;
        let bsav = t;
        for jj in bj..=hi {
            let dplus = d.at(jj - 1) + t;
            if dplus < ZERO {
                neg1 += 1;
            }
            let tmp = t / dplus;
            t = tmp * lld.at(jj - 1) - sigma;
        }
        // A NaN occurs only with a zero pivot after an infinite pivot;
        // substituting 1 for T/DPLUS is the correct limit.
        if t.is_nan() {
            neg1 = 0;
            t = bsav;
            for jj in bj..=hi {
                let dplus = d.at(jj - 1) + t;
                if dplus < ZERO {
                    neg1 += 1;
                }
                let mut tmp = t / dplus;
                if tmp.is_nan() {
                    tmp = ONE;
                }
                t = tmp * lld.at(jj - 1) - sigma;
            }
        }
        negcnt += neg1;
        bj += BLKLEN;
    }

    // II) lower part: L D L^T - SIGMA I = U- D- U-^T
    // `DO 230 BJ = N-1, R, -BLKLEN` visited as an ascending block counter
    // over the identical visit order (the device kernels need this shape; the
    // host keeps it so the two read alike).
    let mut p = d.at(n - 1) - sigma;
    if n >= r + 1 {
        let nblk = (n - 1 - r) / BLKLEN + 1;
        for b in 0..nblk {
            let top = (n - 1) - b * BLKLEN; // BJ
            let lo = if top + 1 > BLKLEN { (top + 1 - BLKLEN).max(r) } else { r };
            let mut neg2: i32 = 0;
            let bsav = p;
            for s in 0..=(top - lo) {
                let jj = top - s;
                let dminus = lld.at(jj - 1) + p;
                if dminus < ZERO {
                    neg2 += 1;
                }
                let tmp = p / dminus;
                p = tmp * d.at(jj - 1) - sigma;
            }
            if p.is_nan() {
                neg2 = 0;
                p = bsav;
                for s in 0..=(top - lo) {
                    let jj = top - s;
                    let dminus = lld.at(jj - 1) + p;
                    if dminus < ZERO {
                        neg2 += 1;
                    }
                    let mut tmp = p / dminus;
                    if tmp.is_nan() {
                        tmp = ONE;
                    }
                    p = tmp * d.at(jj - 1) - sigma;
                }
            }
            negcnt += neg2;
        }
    }

    // III) twist index. T was shifted by SIGMA initially.
    let gamma = (t + sigma) + p;
    if gamma < ZERO {
        negcnt += 1;
    }
    negcnt
}

/// `DLAR1V( N, B1, BN, LAMBDA, D, L, LD, LLD, PIVMIN, GAPTOL, Z, WANTNC,
/// NEGCNT, ZTZ, MINGMA, R, ISUPPZ, NRMINV, RESID, RQCORR, WORK )`.
///
/// `r_in` is the input `R` (0 selects the twist index); the chosen `R` comes
/// back in the tuple. Returns
/// `(negcnt, ztz, mingma, r, isuppz1, isuppz2, nrminv, resid, rqcorr)`.
///
/// # Safety
/// `d`, `l`, `ld`, `lld`, `z` have `n` entries; `work` has `4 * n`.
pub(crate) unsafe fn dlar1v(
    n: usize,
    b1: usize,
    bn: usize,
    lambda: f64,
    d: D,
    l: D,
    ld: D,
    lld: D,
    pivmin: f64,
    gaptol: f64,
    z: D,
    wantnc: bool,
    r_in: usize,
    work: D,
) -> (i32, f64, f64, usize, usize, usize, f64, f64, f64) {
    let eps = PREC; // DLAMCH( 'Precision' )

    let (r1, r2) = if r_in == 0 { (b1, bn) } else { (r_in, r_in) };

    // WORK carve: LPLUS | UMINUS | S | P (`dlar1v.f:282-286`).
    let indlpl = 0usize;
    let indumn = n;
    let inds = 2 * n + 1;
    let indp = 3 * n + 1;

    if b1 == 1 {
        work.set(inds - 1, ZERO);
    } else {
        work.set(inds + b1 - 1 - 1, lld.at(b1 - 2));
    }

    // Stationary transform (differential form) until the index R2.
    let mut neg1: i32 = 0;
    let mut s = work.at(inds + b1 - 1 - 1) - lambda;
    for i in b1..r1 {
        let dplus = d.at(i - 1) + s;
        work.set(indlpl + i - 1, ld.at(i - 1) / dplus);
        if dplus < ZERO {
            neg1 += 1;
        }
        work.set(inds + i - 1, s * work.at(indlpl + i - 1) * l.at(i - 1));
        s = work.at(inds + i - 1) - lambda;
    }
    let mut sawnan1 = s.is_nan();
    if !sawnan1 {
        for i in r1..r2 {
            let dplus = d.at(i - 1) + s;
            work.set(indlpl + i - 1, ld.at(i - 1) / dplus);
            work.set(inds + i - 1, s * work.at(indlpl + i - 1) * l.at(i - 1));
            s = work.at(inds + i - 1) - lambda;
        }
        sawnan1 = s.is_nan();
    }

    if sawnan1 {
        // Slower version of the above loop.
        neg1 = 0;
        s = work.at(inds + b1 - 1 - 1) - lambda;
        for i in b1..r1 {
            let mut dplus = d.at(i - 1) + s;
            if dplus.abs() < pivmin {
                dplus = -pivmin;
            }
            work.set(indlpl + i - 1, ld.at(i - 1) / dplus);
            if dplus < ZERO {
                neg1 += 1;
            }
            work.set(inds + i - 1, s * work.at(indlpl + i - 1) * l.at(i - 1));
            if work.at(indlpl + i - 1) == ZERO {
                work.set(inds + i - 1, lld.at(i - 1));
            }
            s = work.at(inds + i - 1) - lambda;
        }
        for i in r1..r2 {
            let mut dplus = d.at(i - 1) + s;
            if dplus.abs() < pivmin {
                dplus = -pivmin;
            }
            work.set(indlpl + i - 1, ld.at(i - 1) / dplus);
            work.set(inds + i - 1, s * work.at(indlpl + i - 1) * l.at(i - 1));
            if work.at(indlpl + i - 1) == ZERO {
                work.set(inds + i - 1, lld.at(i - 1));
            }
            s = work.at(inds + i - 1) - lambda;
        }
    }

    // Progressive transform (differential form) until the index R1.
    // `DO 80 I = BN-1, R1, -1` as an ascending counter over the same order.
    let mut neg2: i32 = 0;
    work.set(indp + bn - 1 - 1, d.at(bn - 1) - lambda);
    let cnt = bn.saturating_sub(r1);
    for t in 0..cnt {
        let i = (bn - 1) - t;
        let dminus = lld.at(i - 1) + work.at(indp + i - 1);
        let tmp = d.at(i - 1) / dminus;
        if dminus < ZERO {
            neg2 += 1;
        }
        work.set(indumn + i - 1, l.at(i - 1) * tmp);
        work.set(indp + i - 1 - 1, work.at(indp + i - 1) * tmp - lambda);
    }
    let sawnan2 = work.at(indp + r1 - 1 - 1).is_nan();

    if sawnan2 {
        // Slower version of the above loop.
        neg2 = 0;
        for t in 0..cnt {
            let i = (bn - 1) - t;
            let mut dminus = lld.at(i - 1) + work.at(indp + i - 1);
            if dminus.abs() < pivmin {
                dminus = -pivmin;
            }
            let tmp = d.at(i - 1) / dminus;
            if dminus < ZERO {
                neg2 += 1;
            }
            work.set(indumn + i - 1, l.at(i - 1) * tmp);
            work.set(indp + i - 1 - 1, work.at(indp + i - 1) * tmp - lambda);
            if tmp == ZERO {
                work.set(indp + i - 1 - 1, d.at(i - 1) - lambda);
            }
        }
    }

    // Index (from R1 to R2) of the largest (in magnitude) diagonal element of
    // the inverse.
    let mut mingma = work.at(inds + r1 - 1 - 1) + work.at(indp + r1 - 1 - 1);
    if mingma < ZERO {
        neg1 += 1;
    }
    let negcnt: i32 = if wantnc { neg1 + neg2 } else { -1 };
    if mingma.abs() == ZERO {
        mingma = eps * work.at(inds + r1 - 1 - 1);
    }
    let mut r = r1;
    for i in r1..r2 {
        let mut tmp = work.at(inds + i - 1) + work.at(indp + i - 1);
        if tmp == ZERO {
            tmp = eps * work.at(inds + i - 1);
        }
        if tmp.abs() <= mingma.abs() {
            mingma = tmp;
            r = i + 1;
        }
    }

    // FP vector: solve N^T v = e_r.
    let mut isuppz1 = b1;
    let mut isuppz2 = bn;
    z.set(r - 1, ONE);
    let mut ztz = ONE;

    // Upwards from R (`DO 210` / `DO 230`), ascending counter.
    let ucnt = r.saturating_sub(b1);
    if !sawnan1 && !sawnan2 {
        for u in 0..ucnt {
            let i = (r - 1) - u;
            z.set(i - 1, -(work.at(indlpl + i - 1) * z.at(i)));
            if (z.at(i - 1).abs() + z.at(i).abs()) * ld.at(i - 1).abs() < gaptol {
                z.set(i - 1, ZERO);
                isuppz1 = i + 1;
                break;
            }
            ztz = ztz + z.at(i - 1) * z.at(i - 1);
        }
    } else {
        for u in 0..ucnt {
            let i = (r - 1) - u;
            if z.at(i) == ZERO {
                z.set(i - 1, -(ld.at(i) / ld.at(i - 1)) * z.at(i + 1));
            } else {
                z.set(i - 1, -(work.at(indlpl + i - 1) * z.at(i)));
            }
            if (z.at(i - 1).abs() + z.at(i).abs()) * ld.at(i - 1).abs() < gaptol {
                z.set(i - 1, ZERO);
                isuppz1 = i + 1;
                break;
            }
            ztz = ztz + z.at(i - 1) * z.at(i - 1);
        }
    }

    // Downwards from R (`DO 250` / `DO 270`), already ascending.
    if !sawnan1 && !sawnan2 {
        for i in r..bn {
            z.set(i, -(work.at(indumn + i - 1) * z.at(i - 1)));
            if (z.at(i - 1).abs() + z.at(i).abs()) * ld.at(i - 1).abs() < gaptol {
                z.set(i, ZERO);
                isuppz2 = i;
                break;
            }
            ztz = ztz + z.at(i) * z.at(i);
        }
    } else {
        for i in r..bn {
            if z.at(i - 1) == ZERO {
                z.set(i, -(ld.at(i - 2) / ld.at(i - 1)) * z.at(i - 2));
            } else {
                z.set(i, -(work.at(indumn + i - 1) * z.at(i - 1)));
            }
            if (z.at(i - 1).abs() + z.at(i).abs()) * ld.at(i - 1).abs() < gaptol {
                z.set(i, ZERO);
                isuppz2 = i;
                break;
            }
            ztz = ztz + z.at(i) * z.at(i);
        }
    }

    // Quantities for the convergence test.
    let tmp = ONE / ztz;
    let nrminv = tmp.sqrt();
    let resid = mingma.abs() * nrminv;
    let rqcorr = mingma * tmp;

    (negcnt, ztz, mingma, r, isuppz1, isuppz2, nrminv, resid, rqcorr)
}
