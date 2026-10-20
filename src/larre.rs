//! `DLARRE` (Reference-LAPACK v3.12.1 `SRC/dlarre.f`): splitting, the initial
//! relatively robust representations and the eigenvalue approximations.
//!
//! **`RANGE` coverage.** Only `RANGE = 'A'` (`IRANGE = ALLRNG`) is
//! transcribed. `'V'` and `'I'` go through `DLARRD`/`DLAEBZ`
//! (`dlarre.f:470`, ~1 500 reference lines) which nothing in this crate's
//! surface reaches: [`crate::dstemr`] exposes `RANGE = 'A'` only. Those two
//! branches return `INFO = -1`, and the structure of the routine is otherwise
//! kept verbatim so the missing call is visible where the reference makes it.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::aux::{PREC, SAFMIN, fmax, fmin};
use crate::blas::dcopy;
use crate::larr::{dlarrc, dlarrk};
use crate::larr2::{dlarnv, dlarra, dlarrb};
use crate::lasq::dlasq2;
use crate::ptr::{D, J};

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;
const TWO: f64 = 2.0;
const FOUR: f64 = 4.0;
const HNDRD: f64 = 100.0;
const PERT: f64 = 8.0;
const HALF: f64 = ONE / TWO;
const FOURTH: f64 = ONE / FOUR;
const FAC: f64 = HALF;
const MAXGROWTH: f64 = 64.0;
const FUDGE: f64 = 2.0;
const MAXTRY: usize = 6;

/// `RANGE` of [`dlarre`], decoded as the reference's `IRANGE`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Range {
    /// `'A'` — `ALLRNG`.
    All,
    /// `'I'` — `INDRNG` (not transcribed; see the module header).
    Index,
    /// `'V'` — `VALRNG` (not transcribed; see the module header).
    Value,
}

/// `DLARRE( RANGE, N, VL, VU, IL, IU, D, E, E2, RTOL1, RTOL2, SPLTOL, NSPLIT,
/// ISPLIT, M, W, WERR, WGAP, IBLOCK, INDEXW, GERS, PIVMIN, WORK, IWORK,
/// INFO )`.
///
/// Returns `(vl, vu, nsplit, m, pivmin, info)`.
///
/// # Safety
/// `d` has `n` entries; `e`, `e2` have `n` (the routine writes `E( N )`);
/// `isplit`, `iblock`, `indexw` have `n`; `w`, `werr`, `wgap` have `n`;
/// `gers` has `2 * n`; `work` has `6 * n`; `iwork` has `5 * n`.
pub(crate) unsafe fn dlarre(
    range: Range,
    n: usize,
    vl_in: f64,
    vu_in: f64,
    il: usize,
    iu: usize,
    d: D,
    e: D,
    e2: D,
    rtol1: f64,
    rtol2: f64,
    spltol: f64,
    isplit: J,
    w: D,
    werr: D,
    wgap: D,
    iblock: J,
    indexw: J,
    gers: D,
    work: D,
    iwork: J,
) -> (f64, f64, i32, i32, f64, i32) {
    let mut info: i32 = 0;
    let mut nsplit: i32 = 0;
    let mut m: usize = 0;
    let (mut vl, mut vu) = (vl_in, vu_in);
    let mut pivmin = ZERO;

    if n == 0 {
        return (vl, vu, nsplit, m as i32, pivmin, info);
    }

    let irange = range;

    // Machine constants.
    let safmin = SAFMIN;
    let eps = PREC;

    // Parameters.
    let rtl = eps.sqrt();
    let spltol_bs = eps.sqrt(); // BSRTOL, only used on the DLARRD path.
    let _ = spltol_bs;

    // 1x1 matrix: quick return.
    if n == 1 {
        if irange == Range::All
            || (irange == Range::Value && d.at(0) > vl && d.at(0) <= vu)
            || (irange == Range::Index && il == 1 && iu == 1)
        {
            m = 1;
            w.set(0, d.at(0));
            // The computation error of the eigenvalue is zero.
            werr.set(0, ZERO);
            wgap.set(0, ZERO);
            iblock.set(0, 1);
            indexw.set(0, 1);
            gers.set(0, d.at(0));
            gers.set(1, d.at(0));
        }
        // E( 1 ) holds the shift for the initial RRR, zero in this case.
        e.set(0, ZERO);
        return (vl, vu, nsplit, m as i32, pivmin, info);
    }

    // General case. Init WERR, WGAP; Gerschgorin intervals, spectral diameter,
    // maximum off-diagonal entry and PIVMIN.
    let mut gl = d.at(0);
    let mut gu = d.at(0);
    let mut eold = ZERO;
    let mut emax = ZERO;
    e.set(n - 1, ZERO);
    for i in 1..=n {
        werr.set(i - 1, ZERO);
        wgap.set(i - 1, ZERO);
        let eabs = e.at(i - 1).abs();
        if eabs >= emax {
            emax = eabs;
        }
        let tmp1 = eabs + eold;
        gers.set(2 * i - 1 - 1, d.at(i - 1) - tmp1);
        gl = fmin(gl, gers.at(2 * i - 1 - 1));
        gers.set(2 * i - 1, d.at(i - 1) + tmp1);
        gu = fmax(gu, gers.at(2 * i - 1));
        eold = eabs;
    }
    // The minimum pivot allowed in the Sturm sequence for T.
    pivmin = safmin * fmax(ONE, emax * emax);
    // Spectral diameter (the Gerschgorin bounds are off by at most sqrt(2)).
    let mut spdiam = gu - gl;

    // Splitting points.
    let (ns, _iinfo) = dlarra(n, d, e, e2, spltol, spdiam, isplit);
    nsplit = ns;

    // Bisection instead of the faster dqds can be forced; left in for future
    // multisection work.
    let forceb = false;

    let mut usedqd = irange == Range::All && !forceb;

    let mm: usize = 0;
    if irange == Range::All && !forceb {
        // Interval [VL, VU] that contains all eigenvalues.
        vl = gl;
        vu = gu;
    } else {
        // `CALL DLARRD( RANGE, 'B', .. )` — not transcribed; see the module
        // header. Nothing in this crate's surface reaches it.
        return (vl, vu, nsplit, 0, pivmin, -1);
    }

    // Loop over unreduced blocks.
    let mut ibegin = 1usize;
    let mut wbegin = 1usize;
    for jblk in 1..=nsplit as usize {
        let iend = isplit.at(jblk - 1) as usize;
        let inn = iend - ibegin + 1;

        // 1 x 1 block.
        if inn == 1 {
            if irange == Range::All
                || (irange == Range::Value && d.at(ibegin - 1) > vl && d.at(ibegin - 1) <= vu)
                || (irange == Range::Index && iblock.at(wbegin - 1) == jblk as i32)
            {
                m += 1;
                w.set(m - 1, d.at(ibegin - 1));
                werr.set(m - 1, ZERO);
                // The gap of a single block does not matter later.
                wgap.set(m - 1, ZERO);
                iblock.set(m - 1, jblk as i32);
                indexw.set(m - 1, 1);
                wbegin += 1;
            }
            // E( IEND ) holds the shift for the initial RRR.
            e.set(iend - 1, ZERO);
            ibegin = iend + 1;
            continue;
        }

        // Blocks larger than 1x1. E( IEND ) will hold the shift; zero for now.
        e.set(iend - 1, ZERO);

        // Local outer bounds GL, GU for the block.
        let mut gl = d.at(ibegin - 1);
        let mut gu = d.at(ibegin - 1);
        for i in ibegin..=iend {
            gl = fmin(gers.at(2 * i - 1 - 1), gl);
            gu = fmax(gers.at(2 * i - 1), gu);
        }
        spdiam = gu - gl;

        let mut mb: usize;
        let mut wend = wbegin;
        let mut indl: usize;
        let mut indu: usize;
        let mut sigma;

        if !(irange == Range::All && !forceb) {
            // Count the eigenvalues in the current block (DLARRD path).
            mb = 0;
            for i in wbegin..=mm {
                if iblock.at(i - 1) == jblk as i32 {
                    mb += 1;
                } else {
                    break;
                }
            }
            if mb == 0 {
                e.set(iend - 1, ZERO);
                ibegin = iend + 1;
                continue;
            }
            usedqd = (mb as f64) > FAC * (inn as f64) && !forceb;
            wend = wbegin + mb - 1;
            sigma = ZERO;
            for i in wbegin..wend {
                wgap.set(i - 1, fmax(ZERO, w.at(i) - werr.at(i) - (w.at(i - 1) + werr.at(i - 1))));
            }
            wgap.set(wend - 1, fmax(ZERO, vu - sigma - (w.at(wend - 1) + werr.at(wend - 1))));
            indl = indexw.at(wbegin - 1) as usize;
            indu = indexw.at(wend - 1) as usize;
        } else {
            mb = 0;
            indl = 0;
            indu = 0;
        }

        let isleft;
        let isrght;
        if (irange == Range::All && !forceb) || usedqd {
            // dqds: approximations to the extremal eigenvalues of the block.
            let (tmp, tmp1, iinfo) = dlarrk(inn, 1, gl, gu, d.off(ibegin - 1), e2.off(ibegin - 1), pivmin, rtl);
            if iinfo != 0 {
                return (vl, vu, nsplit, m as i32, pivmin, -1);
            }
            isleft = fmax(gl, tmp - tmp1 - HNDRD * eps * (tmp - tmp1).abs());

            let (tmp, tmp1, iinfo) =
                dlarrk(inn, inn as i32, gl, gu, d.off(ibegin - 1), e2.off(ibegin - 1), pivmin, rtl);
            if iinfo != 0 {
                return (vl, vu, nsplit, m as i32, pivmin, -1);
            }
            isrght = fmin(gu, tmp + tmp1 + HNDRD * eps * (tmp + tmp1).abs());
            // Improve the estimate of the spectral diameter.
            spdiam = isrght - isleft;
        } else {
            // Bisection: approximations to the wanted extremal eigenvalues.
            isleft = fmax(
                gl,
                w.at(wbegin - 1) - werr.at(wbegin - 1) - HNDRD * eps * (w.at(wbegin - 1) - werr.at(wbegin - 1)).abs(),
            );
            isrght =
                fmin(gu, w.at(wend - 1) + werr.at(wend - 1) + HNDRD * eps * (w.at(wend - 1) + werr.at(wend - 1)).abs());
        }

        // Which end the base representation should be shifted to, and whether
        // dqds or bisection computes the eigenvalue approximations.
        let s1;
        let s2;
        if irange == Range::All && !forceb {
            usedqd = true;
            indl = 1;
            indu = inn;
            mb = inn;
            wend = wbegin + mb - 1;
            // 1/4 and 3/4 points of the spectrum.
            s1 = isleft + FOURTH * spdiam;
            s2 = isrght - FOURTH * spdiam;
        } else if usedqd {
            s1 = isleft + FOURTH * spdiam;
            s2 = isrght - FOURTH * spdiam;
        } else {
            let tmp = fmin(isrght, vu) - fmax(isleft, vl);
            s1 = fmax(isleft, vl) + FOURTH * tmp;
            s2 = fmin(isrght, vu) - FOURTH * tmp;
        }

        // Negcount at the 1/4 and 3/4 points.
        let (mut cnt1, mut cnt2) = (0i32, 0i32);
        if mb > 1 {
            let (_cnt, c1, c2, _ii) = dlarrc(true, inn, s1, s2, d.off(ibegin - 1), e.off(ibegin - 1));
            cnt1 = c1;
            cnt2 = c2;
        }

        let sgndef;
        if mb == 1 {
            sigma = gl;
            sgndef = ONE;
        } else if cnt1 - indl as i32 >= indu as i32 - cnt2 {
            if irange == Range::All && !forceb {
                sigma = fmax(isleft, gl);
            } else if usedqd {
                sigma = isleft;
            } else {
                sigma = fmax(isleft, vl);
            }
            sgndef = ONE;
        } else {
            if irange == Range::All && !forceb {
                sigma = fmin(isrght, gu);
            } else if usedqd {
                sigma = isrght;
            } else {
                sigma = fmin(isrght, vu);
            }
            sgndef = -ONE;
        }

        // The shift increment TAU, in case SIGMA needs refining.
        let mut tau;
        if usedqd {
            tau = spdiam * eps * n as f64 + TWO * pivmin;
            tau = fmax(tau, TWO * eps * sigma.abs());
        } else if mb > 1 {
            let clwdth = w.at(wend - 1) + werr.at(wend - 1) - w.at(wbegin - 1) - werr.at(wbegin - 1);
            let avgap = (clwdth / ((wend - wbegin) as f64)).abs();
            if sgndef == ONE {
                tau = HALF * fmax(wgap.at(wbegin - 1), avgap);
                tau = fmax(tau, werr.at(wbegin - 1));
            } else {
                tau = HALF * fmax(wgap.at(wend - 1 - 1), avgap);
                tau = fmax(tau, werr.at(wend - 1));
            }
        } else {
            tau = werr.at(wbegin - 1);
        }

        let mut found = false;
        for idum in 1..=MAXTRY {
            // L D L^T of T - sigma I: D in WORK(1:IN), L in WORK(IN+1:2*IN),
            // reciprocal pivots in WORK(2*IN+1:3*IN).
            let mut dpivot = d.at(ibegin - 1) - sigma;
            work.set(0, dpivot);
            let mut dmax = work.at(0).abs();
            let mut jj = ibegin;
            for i in 1..=inn - 1 {
                work.set(2 * inn + i - 1, ONE / work.at(i - 1));
                let tmp = e.at(jj - 1) * work.at(2 * inn + i - 1);
                work.set(inn + i - 1, tmp);
                dpivot = (d.at(jj) - sigma) - tmp * e.at(jj - 1);
                work.set(i, dpivot);
                dmax = fmax(dmax, dpivot.abs());
                jj += 1;
            }
            // Element growth.
            let mut norep = dmax > MAXGROWTH * spdiam;
            if usedqd && !norep {
                // Definiteness: every entry of D (of L D L^T) has one sign.
                for i in 1..=inn {
                    let tmp = sgndef * work.at(i - 1);
                    if tmp < ZERO {
                        norep = true;
                    }
                }
            }
            if norep {
                // With IRANGE = ALLRNG the Gerschgorin shift makes the matrix
                // definite, so this is really only reached for VALRNG/INDRNG.
                if idum == MAXTRY - 1 {
                    if sgndef == ONE {
                        sigma = gl - FUDGE * spdiam * eps * n as f64 - FUDGE * TWO * pivmin;
                    } else {
                        sigma = gu + FUDGE * spdiam * eps * n as f64 + FUDGE * TWO * pivmin;
                    }
                } else {
                    sigma = sigma - sgndef * tau;
                    tau = TWO * tau;
                }
            } else {
                // An initial RRR is found.
                found = true;
                break;
            }
        }
        if !found {
            // No base representation in MAXTRY iterations.
            return (vl, vu, nsplit, m as i32, pivmin, 2);
        }

        // Label 83: store the shift, D and L.
        e.set(iend - 1, sigma);
        dcopy(inn, work, 1, d.off(ibegin - 1), 1);
        dcopy(inn - 1, work.off(inn), 1, e.off(ibegin - 1), 1);

        if mb > 1 {
            // Perturb each entry of the base representation by a small but
            // random relative amount, to overcome glued matrices.
            let mut iseed = [1i32; 4];
            dlarnv(2, &mut iseed, 2 * inn - 1, work);
            for i in 1..=inn - 1 {
                d.set(ibegin + i - 1 - 1, d.at(ibegin + i - 1 - 1) * (ONE + eps * PERT * work.at(i - 1)));
                e.set(ibegin + i - 1 - 1, e.at(ibegin + i - 1 - 1) * (ONE + eps * PERT * work.at(inn + i - 1)));
            }
            d.set(iend - 1, d.at(iend - 1) * (ONE + eps * FOUR * work.at(inn - 1)));
        }

        // The Gerschgorin intervals are deliberately not updated; W is.
        if !usedqd {
            // Shift the DLARRD approximations to their representation.
            for jj in wbegin..=wend {
                w.set(jj - 1, w.at(jj - 1) - sigma);
                werr.set(jj - 1, werr.at(jj - 1) + w.at(jj - 1).abs() * eps);
            }
            for i in ibegin..=iend - 1 {
                work.set(i - 1, d.at(i - 1) * e.at(i - 1) * e.at(i - 1));
            }
            let iinfo = dlarrb(
                inn,
                d.off(ibegin - 1),
                work.off(ibegin - 1),
                indl,
                indu,
                rtol1,
                rtol2,
                indl - 1,
                w.off(wbegin - 1),
                wgap.off(wbegin - 1),
                werr.off(wbegin - 1),
                work.off(2 * n),
                iwork,
                pivmin,
                spdiam,
                inn as i32,
            );
            if iinfo != 0 {
                return (vl, vu, nsplit, m as i32, pivmin, -4);
            }
            // DLARRB computes all gaps correctly except the last one.
            wgap.set(wend - 1, fmax(ZERO, (vu - sigma) - (w.at(wend - 1) + werr.at(wend - 1))));
            for i in indl..=indu {
                m += 1;
                iblock.set(m - 1, jblk as i32);
                indexw.set(m - 1, i as i32);
            }
        } else {
            // dqds for all eigenvalues of the block. RTOL is the order of the
            // tolerance DLASQ2 uses (an estimate; the worst case is 4*N*EPS).
            let rtol = (inn as f64).ln() * FOUR * eps;
            let mut jj = ibegin;
            for i in 1..=inn - 1 {
                work.set(2 * i - 1 - 1, d.at(jj - 1).abs());
                work.set(2 * i - 1, e.at(jj - 1) * e.at(jj - 1) * work.at(2 * i - 1 - 1));
                jj += 1;
            }
            work.set(2 * inn - 1 - 1, d.at(iend - 1).abs());
            work.set(2 * inn - 1, ZERO);
            let iinfo = dlasq2(inn, work, true);
            if iinfo != 0 {
                // IINFO = -5: an index is part of a tight cluster.
                return (vl, vu, nsplit, m as i32, pivmin, -5);
            }
            for i in 1..=inn {
                if work.at(i - 1) < ZERO {
                    return (vl, vu, nsplit, m as i32, pivmin, -6);
                }
            }
            if sgndef > ZERO {
                for i in indl..=indu {
                    m += 1;
                    w.set(m - 1, work.at(inn - i + 1 - 1));
                    iblock.set(m - 1, jblk as i32);
                    indexw.set(m - 1, i as i32);
                }
            } else {
                for i in indl..=indu {
                    m += 1;
                    w.set(m - 1, -work.at(i - 1));
                    iblock.set(m - 1, jblk as i32);
                    indexw.set(m - 1, i as i32);
                }
            }

            for i in m - mb + 1..=m {
                werr.set(i - 1, rtol * w.at(i - 1).abs());
            }
            for i in m - mb + 1..=m - 1 {
                wgap.set(i - 1, fmax(ZERO, w.at(i) - werr.at(i) - (w.at(i - 1) + werr.at(i - 1))));
            }
            wgap.set(m - 1, fmax(ZERO, (vu - sigma) - (w.at(m - 1) + werr.at(m - 1))));
        }
        // Next block.
        ibegin = iend + 1;
        wbegin = wend + 1;
    }

    let _ = mm;
    info = 0;
    (vl, vu, nsplit, m as i32, pivmin, info)
}
