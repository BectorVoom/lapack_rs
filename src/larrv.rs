//! `DLARRV` (Reference-LAPACK v3.12.1 `SRC/dlarrv.f`): eigenvectors of
//! `L D L^T` from the MRRR representation tree.
//!
//! `Z` is column-major with leading dimension `ldz`; `IWORK` is carved as the
//! reference carves it (`IINDR`, `IINDC1`, `IINDC2`, `IINDWK`) and `WORK` as
//! (`eigenvalues`, `INDLD`, `INDLLD`, `INDWRK`).
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::aux::{PREC, dlaset_full, fmax, fmin};
use crate::blas::{dcopy, dscal};
use crate::larr::dlar1v;
use crate::larr2::{dlarrb, dlarrf};
use crate::ptr::{D, J, ix};

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;
const TWO: f64 = 2.0;
const THREE: f64 = 3.0;
const FOUR: f64 = 4.0;
const HALF: f64 = 0.5;
const MAXITR: i32 = 10;

/// `DLARRV( N, VL, VU, D, L, PIVMIN, ISPLIT, M, DOL, DOU, MINRGP, RTOL1,
/// RTOL2, W, WERR, WGAP, IBLOCK, INDEXW, GERS, Z, LDZ, ISUPPZ, WORK, IWORK,
/// INFO )`.
///
/// # Safety
/// `d`, `l` have `n` entries; `w`, `werr`, `wgap`, `iblock`, `indexw` have
/// `m`; `gers` has `2 * n`; `z` has `ldz * max(1, m)`; `isuppz` has `2 * m`;
/// `work` has `12 * n`; `iwork` has `7 * n`.
pub(crate) unsafe fn dlarrv(
    n: usize,
    vl: f64,
    _vu: f64,
    d: D,
    l: D,
    pivmin: f64,
    isplit: J,
    m: usize,
    dol: usize,
    dou: usize,
    minrgp: f64,
    rtol1_in: f64,
    rtol2_in: f64,
    w: D,
    werr: D,
    wgap: D,
    iblock: J,
    indexw: J,
    gers: D,
    z: D,
    ldz: usize,
    isuppz: J,
    work: D,
    iwork: J,
) -> i32 {
    let mut rtol1 = rtol1_in;
    let mut rtol2 = rtol2_in;
    if n == 0 || m == 0 {
        return 0;
    }

    // The first N entries of WORK are reserved for the eigenvalues.
    let indld = n + 1;
    let indlld = 2 * n + 1;
    let indwrk = 3 * n + 1;
    let minwsize = 12 * n;
    for i in 1..=minwsize {
        work.set(i - 1, ZERO);
    }

    // IWORK(IINDR+1:IINDR+N) holds the twist indices R.
    let iindr = 0usize;
    let iindc1 = n;
    let iindc2 = 2 * n;
    let iindwk = 3 * n + 1;

    let miniwsize = 7 * n;
    for i in 1..=miniwsize {
        iwork.set(i - 1, 0);
    }

    let mut zusedl = 1usize;
    if dol > 1 {
        zusedl = dol - 1;
    }
    let mut zusedu = m;
    if dou < m {
        zusedu = dou + 1;
    }
    let zusedw = zusedu - zusedl + 1;

    dlaset_full(n, zusedw, ZERO, ZERO, z.off(ix(1, zusedl, ldz)), ldz);

    let eps = PREC;
    let rqtol = TWO * eps;

    // Expert flags for standard code.
    let tryrqc = true;

    if dol == 1 && dou == m {
    } else {
        // Only selected eigenpairs: bisection has to reach full accuracy.
        rtol1 = FOUR * eps;
        rtol2 = FOUR * eps;
    }

    // DONE is the number of eigenvectors already computed.
    let mut _done = 0usize;
    let mut ibegin = 1usize;
    let mut wbegin = 1usize;
    for jblk in 1..=iblock.at(m - 1) as usize {
        let iend = isplit.at(jblk - 1) as usize;
        let mut sigma = l.at(iend - 1);
        // Eigenvectors of the submatrix IBEGIN..IEND.
        let mut wend = wbegin - 1;
        while wend < m && iblock.at(wend) == jblk as i32 {
            wend += 1;
        }
        if wend < wbegin {
            ibegin = iend + 1;
            continue;
        } else if wend < dol || wbegin > dou {
            ibegin = iend + 1;
            wbegin = wend + 1;
            continue;
        }

        // Local spectral diameter of the block.
        let mut gl = gers.at(2 * ibegin - 1 - 1);
        let mut gu = gers.at(2 * ibegin - 1);
        for i in ibegin + 1..=iend {
            gl = fmin(gers.at(2 * i - 1 - 1), gl);
            gu = fmax(gers.at(2 * i - 1), gu);
        }
        let spdiam = gu - gl;

        // OLDIEN is the last index of the previous block.
        let oldien = ibegin - 1;
        let inn = iend - ibegin + 1;
        let im = wend - wbegin + 1;

        // 1x1 block.
        if ibegin == iend {
            _done += 1;
            z.set(ix(ibegin, wbegin, ldz), ONE);
            isuppz.set(2 * wbegin - 1 - 1, ibegin as i32);
            isuppz.set(2 * wbegin - 1, ibegin as i32);
            w.set(wbegin - 1, w.at(wbegin - 1) + sigma);
            work.set(wbegin - 1, w.at(wbegin - 1));
            ibegin = iend + 1;
            wbegin += 1;
            continue;
        }

        // The desired (shifted) eigenvalues are in W(WBEGIN:WEND).
        dcopy(im, w.off(wbegin - 1), 1, work.off(wbegin - 1), 1);

        // W holds the eigenvalue approximations w.r.t. the original T.
        for i in 1..=im {
            w.set(wbegin + i - 1 - 1, w.at(wbegin + i - 1 - 1) + sigma);
        }

        let mut ndepth: usize = 0;
        let mut parity: i32 = 1;
        let mut nclus: usize = 1;
        iwork.set(iindc1 + 1 - 1, 1);
        iwork.set(iindc1 + 2 - 1, im as i32);

        let mut idone: usize = 0;
        // Generate the representation tree and compute the eigenvectors.
        while idone < im {
            // Crude protection against infinitely deep trees.
            if ndepth > m {
                return -2;
            }
            let oldncl = nclus;
            nclus = 0;

            parity = 1 - parity;
            let (oldcls, newcls) = if parity == 0 { (iindc1, iindc2) } else { (iindc2, iindc1) };

            // Process the clusters on the current level.
            for i in 1..=oldncl {
                let jj = oldcls + 2 * i;
                // OLDFST, OLDLST: first, last index of the current cluster,
                // relative to WBEGIN.
                let oldfst = iwork.at(jj - 1 - 1) as usize;
                let oldlst = iwork.at(jj - 1) as usize;
                if ndepth > 0 {
                    // Retrieve the RRR of the cluster from the previous level.
                    let jz = if dol == 1 && dou == m {
                        wbegin + oldfst - 1
                    } else if wbegin + oldfst - 1 < dol {
                        dol - 1
                    } else if wbegin + oldfst - 1 > dou {
                        dou
                    } else {
                        wbegin + oldfst - 1
                    };
                    dcopy(inn, z.off(ix(ibegin, jz, ldz)), 1, d.off(ibegin - 1), 1);
                    dcopy(inn - 1, z.off(ix(ibegin, jz + 1, ldz)), 1, l.off(ibegin - 1), 1);
                    sigma = z.at(ix(iend, jz + 1, ldz));

                    // Zero the corresponding entries of Z.
                    dlaset_full(inn, 2, ZERO, ZERO, z.off(ix(ibegin, jz, ldz)), ldz);
                }

                // DL and DLL of the current RRR.
                for jj in ibegin..=iend - 1 {
                    let tmp = d.at(jj - 1) * l.at(jj - 1);
                    work.set(indld - 1 + jj - 1, tmp);
                    work.set(indlld - 1 + jj - 1, tmp * l.at(jj - 1));
                }

                if ndepth > 0 {
                    // P, Q: first and last eigenvalue to compute in the block.
                    let p = indexw.at(wbegin - 1 + oldfst - 1) as usize;
                    let q = indexw.at(wbegin - 1 + oldlst - 1) as usize;
                    let offset = indexw.at(wbegin - 1) as usize - 1;
                    // Limited bisection to the precision needed.
                    let iinfo = dlarrb(
                        inn,
                        d.off(ibegin - 1),
                        work.off(indlld + ibegin - 1 - 1),
                        p,
                        q,
                        rtol1,
                        rtol2,
                        offset,
                        work.off(wbegin - 1),
                        wgap.off(wbegin - 1),
                        werr.off(wbegin - 1),
                        work.off(indwrk - 1),
                        iwork.off(iindwk - 1),
                        pivmin,
                        spdiam,
                        inn as i32,
                    );
                    if iinfo != 0 {
                        return -1;
                    }
                    // Recompute the extremal gaps from W (which holds the
                    // unshifted eigenvalues); only allow them to grow.
                    if oldfst > 1 {
                        wgap.set(
                            wbegin + oldfst - 2 - 1,
                            fmax(
                                wgap.at(wbegin + oldfst - 2 - 1),
                                w.at(wbegin + oldfst - 1 - 1)
                                    - werr.at(wbegin + oldfst - 1 - 1)
                                    - w.at(wbegin + oldfst - 2 - 1)
                                    - werr.at(wbegin + oldfst - 2 - 1),
                            ),
                        );
                    }
                    if wbegin + oldlst - 1 < wend {
                        wgap.set(
                            wbegin + oldlst - 1 - 1,
                            fmax(
                                wgap.at(wbegin + oldlst - 1 - 1),
                                w.at(wbegin + oldlst - 1)
                                    - werr.at(wbegin + oldlst - 1)
                                    - w.at(wbegin + oldlst - 1 - 1)
                                    - werr.at(wbegin + oldlst - 1 - 1),
                            ),
                        );
                    }
                    // Store the refined approximation with all shifts in W.
                    for jj in oldfst..=oldlst {
                        w.set(wbegin + jj - 1 - 1, work.at(wbegin + jj - 1 - 1) + sigma);
                    }
                }

                // Process the current node.
                let mut newfst = oldfst;
                for jj in oldfst..=oldlst {
                    let newlst;
                    if jj == oldlst {
                        // Right end of the cluster, also the child boundary.
                        newlst = jj;
                    } else if wgap.at(wbegin + jj - 1 - 1) >= minrgp * work.at(wbegin + jj - 1 - 1).abs() {
                        // The right relative gap is big enough.
                        newlst = jj;
                    } else {
                        // Inside a child cluster; the relative gap is small.
                        continue;
                    }

                    let newsiz = newlst - newfst + 1;

                    // NEWFTT: where the new RRR or eigenvector goes in Z.
                    let newftt = if dol == 1 && dou == m {
                        wbegin + newfst - 1
                    } else if wbegin + newfst - 1 < dol {
                        dol - 1
                    } else if wbegin + newfst - 1 > dou {
                        dou
                    } else {
                        wbegin + newfst - 1
                    };

                    if newsiz > 1 {
                        // The child is a cluster: compute and store its RRR.
                        let lgap = if newfst == 1 {
                            fmax(ZERO, w.at(wbegin - 1) - werr.at(wbegin - 1) - vl)
                        } else {
                            wgap.at(wbegin + newfst - 2 - 1)
                        };
                        let rgap = wgap.at(wbegin + newlst - 1 - 1);

                        // Left- and rightmost eigenvalue of the child to high
                        // precision, so the shift can be as close as possible.
                        for k in 1..=2 {
                            let p = if k == 1 {
                                indexw.at(wbegin - 1 + newfst - 1) as usize
                            } else {
                                indexw.at(wbegin - 1 + newlst - 1) as usize
                            };
                            let offset = indexw.at(wbegin - 1) as usize - 1;
                            dlarrb(
                                inn,
                                d.off(ibegin - 1),
                                work.off(indlld + ibegin - 1 - 1),
                                p,
                                p,
                                rqtol,
                                rqtol,
                                offset,
                                work.off(wbegin - 1),
                                wgap.off(wbegin - 1),
                                werr.off(wbegin - 1),
                                work.off(indwrk - 1),
                                iwork.off(iindwk - 1),
                                pivmin,
                                spdiam,
                                inn as i32,
                            );
                        }

                        if wbegin + newlst - 1 < dol || wbegin + newfst - 1 > dou {
                            // The cluster holds no desired eigenvalue: skip
                            // this branch. (Skipping earlier would change the
                            // representation tree.)
                            idone += newlst - newfst + 1;
                            newfst = jj + 1;
                            continue;
                        }

                        // RRR of the child cluster; DLARRF needs LWORK = 2*N.
                        let (tau, iinfo) = dlarrf(
                            inn,
                            d.off(ibegin - 1),
                            l.off(ibegin - 1),
                            work.off(indld + ibegin - 1 - 1),
                            newfst,
                            newlst,
                            work.off(wbegin - 1),
                            wgap.off(wbegin - 1),
                            werr.off(wbegin - 1),
                            spdiam,
                            lgap,
                            rgap,
                            pivmin,
                            z.off(ix(ibegin, newftt, ldz)),
                            z.off(ix(ibegin, newftt + 1, ldz)),
                            work.off(indwrk - 1),
                        );
                        if iinfo == 0 {
                            // A new RRR was found: update and store the shift.
                            let ssigma = sigma + tau;
                            z.set(ix(iend, newftt + 1, ldz), ssigma);
                            // WORK are the midpoints, WERR the semi-widths;
                            // the entries in W are unchanged.
                            for k in newfst..=newlst {
                                let mut fudge = THREE * eps * work.at(wbegin + k - 1 - 1).abs();
                                work.set(wbegin + k - 1 - 1, work.at(wbegin + k - 1 - 1) - tau);
                                fudge = fudge + FOUR * eps * work.at(wbegin + k - 1 - 1).abs();
                                werr.set(wbegin + k - 1 - 1, werr.at(wbegin + k - 1 - 1) + fudge);
                                // Gaps are deliberately not fudged.
                            }

                            nclus += 1;
                            let k = newcls + 2 * nclus;
                            iwork.set(k - 1 - 1, newfst as i32);
                            iwork.set(k - 1, newlst as i32);
                        } else {
                            return -2;
                        }
                    } else {
                        // Eigenvector of a singleton.
                        let mut iter: i32 = 0;
                        let tol = FOUR * (inn as f64).ln() * eps;

                        let k = newfst;
                        let windex = wbegin + k - 1;
                        let windmn = (windex - 1).max(1);
                        let windpl = (windex + 1).min(m);
                        let mut lambda = work.at(windex - 1);
                        _done += 1;
                        // Whether the eigenvector computation is skipped.
                        let eskip = windex < dol || windex > dou;
                        let mut savgap = ZERO;
                        if !eskip {
                            let mut left = work.at(windex - 1) - werr.at(windex - 1);
                            let mut right = work.at(windex - 1) + werr.at(windex - 1);
                            let indeig = indexw.at(windex - 1);

                            // The gaps come from WORK (same shift for a child).
                            let lgap = if k == 1 {
                                // Force a small left gap; see `dlarrv.f:795`.
                                eps * fmax(left.abs(), right.abs())
                            } else {
                                wgap.at(windmn - 1)
                            };
                            let rgap = if k == im { eps * fmax(left.abs(), right.abs()) } else { wgap.at(windex - 1) };
                            let gap = fmin(lgap, rgap);
                            let gaptol = if k == 1 || k == im {
                                // The support can be cut off by a large GAPTOL.
                                ZERO
                            } else {
                                gap * eps
                            };
                            let mut isupmn = inn;
                            let mut isupmx = 1usize;
                            // WGAP temporarily holds the minimum gap, so that
                            // bisection refines to the required precision.
                            savgap = wgap.at(windex - 1);
                            wgap.set(windex - 1, gap);

                            let mut usedbs = false;
                            let mut usedrq = false;
                            let mut needbs = !tryrqc;
                            let mut bstres = ZERO;
                            let mut bstw = ZERO;
                            let mut nrminv;
                            let mut resid;

                            // Label 120.
                            loop {
                                if needbs {
                                    // Take the bisection as the new iterate.
                                    usedbs = true;
                                    let itmp1 = iwork.at(iindr + windex - 1);
                                    let offset = indexw.at(wbegin - 1) as usize - 1;
                                    let iinfo = dlarrb(
                                        inn,
                                        d.off(ibegin - 1),
                                        work.off(indlld + ibegin - 1 - 1),
                                        indeig as usize,
                                        indeig as usize,
                                        ZERO,
                                        TWO * eps,
                                        offset,
                                        work.off(wbegin - 1),
                                        wgap.off(wbegin - 1),
                                        werr.off(wbegin - 1),
                                        work.off(indwrk - 1),
                                        iwork.off(iindwk - 1),
                                        pivmin,
                                        spdiam,
                                        itmp1,
                                    );
                                    if iinfo != 0 {
                                        return -3;
                                    }
                                    lambda = work.at(windex - 1);
                                    // Reset the twist index so the true MINGMA
                                    // is recomputed from an inaccurate LAMBDA.
                                    iwork.set(iindr + windex - 1, 0);
                                }
                                // Given LAMBDA, compute the eigenvector.
                                let r_in = iwork.at(iindr + windex - 1) as usize;
                                let (negcnt, _ztz, _mingma, r_out, isz1, isz2, nrm, res, rqcorr) = dlar1v(
                                    inn,
                                    1,
                                    inn,
                                    lambda,
                                    d.off(ibegin - 1),
                                    l.off(ibegin - 1),
                                    work.off(indld + ibegin - 1 - 1),
                                    work.off(indlld + ibegin - 1 - 1),
                                    pivmin,
                                    gaptol,
                                    z.off(ix(ibegin, windex, ldz)),
                                    !usedbs,
                                    r_in,
                                    work.off(indwrk - 1),
                                );
                                iwork.set(iindr + windex - 1, r_out as i32);
                                isuppz.set(2 * windex - 1 - 1, isz1 as i32);
                                isuppz.set(2 * windex - 1, isz2 as i32);
                                nrminv = nrm;
                                resid = res;
                                if iter == 0 {
                                    bstres = resid;
                                    bstw = lambda;
                                } else if resid < bstres {
                                    bstres = resid;
                                    bstw = lambda;
                                }
                                isupmn = isupmn.min(isuppz.at(2 * windex - 1 - 1) as usize);
                                isupmx = isupmx.max(isuppz.at(2 * windex - 1) as usize);
                                iter += 1;

                                // Convergence test for the Rayleigh-Quotient
                                // iteration (skipped when bisection was used).
                                if resid > tol * gap && rqcorr.abs() > rqtol * lambda.abs() && !usedbs {
                                    // The RQCORR update must not move the
                                    // eigenvalue toward a neighbour.
                                    let sgndef = if indeig <= negcnt { -ONE } else { ONE };
                                    if rqcorr * sgndef >= ZERO && lambda + rqcorr <= right && lambda + rqcorr >= left {
                                        usedrq = true;
                                        if sgndef == ONE {
                                            left = lambda;
                                        } else {
                                            right = lambda;
                                        }
                                        work.set(windex - 1, HALF * (right + left));
                                        lambda = lambda + rqcorr;
                                        werr.set(windex - 1, HALF * (right - left));
                                    } else {
                                        needbs = true;
                                    }
                                    if right - left < rqtol * lambda.abs() {
                                        // Bisection accuracy reached.
                                        usedbs = true;
                                        continue;
                                    } else if iter < MAXITR {
                                        continue;
                                    } else if iter == MAXITR {
                                        needbs = true;
                                        continue;
                                    } else {
                                        return 5;
                                    }
                                } else {
                                    let mut stp2ii = false;
                                    if usedrq && usedbs && bstres <= resid {
                                        lambda = bstw;
                                        stp2ii = true;
                                    }
                                    if stp2ii {
                                        // Improve the error angle by a second step.
                                        let r_in = iwork.at(iindr + windex - 1) as usize;
                                        let (_nc, _zt, _mg, r_out, isz1, isz2, nrm, res, _rq) = dlar1v(
                                            inn,
                                            1,
                                            inn,
                                            lambda,
                                            d.off(ibegin - 1),
                                            l.off(ibegin - 1),
                                            work.off(indld + ibegin - 1 - 1),
                                            work.off(indlld + ibegin - 1 - 1),
                                            pivmin,
                                            gaptol,
                                            z.off(ix(ibegin, windex, ldz)),
                                            !usedbs,
                                            r_in,
                                            work.off(indwrk - 1),
                                        );
                                        iwork.set(iindr + windex - 1, r_out as i32);
                                        isuppz.set(2 * windex - 1 - 1, isz1 as i32);
                                        isuppz.set(2 * windex - 1, isz2 as i32);
                                        nrminv = nrm;
                                        let _ = res;
                                    }
                                    work.set(windex - 1, lambda);
                                    break;
                                }
                            }

                            // FP-vector support w.r.t. the whole matrix.
                            isuppz.set(2 * windex - 1 - 1, isuppz.at(2 * windex - 1 - 1) + oldien as i32);
                            isuppz.set(2 * windex - 1, isuppz.at(2 * windex - 1) + oldien as i32);
                            let zfrom = isuppz.at(2 * windex - 1 - 1) as usize;
                            let zto = isuppz.at(2 * windex - 1) as usize;
                            let isupmn = isupmn + oldien;
                            let isupmx = isupmx + oldien;
                            // The support may have changed during the RQI.
                            if isupmn < zfrom {
                                for ii in isupmn..=zfrom - 1 {
                                    z.set(ix(ii, windex, ldz), ZERO);
                                }
                            }
                            if isupmx > zto {
                                for ii in zto + 1..=isupmx {
                                    z.set(ix(ii, windex, ldz), ZERO);
                                }
                            }
                            dscal(zto - zfrom + 1, nrminv, z.off(ix(zfrom, windex, ldz)), 1);
                        }

                        // Label 125: update W.
                        w.set(windex - 1, lambda + sigma);
                        // Recompute the gaps, allowing them only to grow.
                        if !eskip {
                            if k > 1 {
                                wgap.set(
                                    windmn - 1,
                                    fmax(
                                        wgap.at(windmn - 1),
                                        w.at(windex - 1) - werr.at(windex - 1) - w.at(windmn - 1) - werr.at(windmn - 1),
                                    ),
                                );
                            }
                            if windex < wend {
                                wgap.set(
                                    windex - 1,
                                    fmax(
                                        savgap,
                                        w.at(windpl - 1) - werr.at(windpl - 1) - w.at(windex - 1) - werr.at(windex - 1),
                                    ),
                                );
                            }
                        }
                        idone += 1;
                    }
                    // Proceed to any remaining child nodes.
                    newfst = jj + 1;
                }
            }
            ndepth += 1;
        }
        ibegin = iend + 1;
        wbegin = wend + 1;
    }

    0
}
