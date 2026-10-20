//! MRRR block routines (Reference-LAPACK v3.12.1): splitting, bisection
//! refinement and the shifted-RRR search.
//!
//! Transcribed from `SRC/dlarra.f`, `SRC/dlarrr.f`, `SRC/dlarrb.f`,
//! `SRC/dlarrj.f`, `SRC/dlarrf.f`, plus `SRC/dlarnv.f` and `SRC/dlaruv.f`
//! (the perturbation `DLARRE` applies to a failing RRR).
//!
//! Integer workspaces are `i32`: `DLARRB` and `DLARRJ` store `-1` sentinels
//! in their unconverged-interval linked lists.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::aux::{PREC, SAFMIN, fmax, fmin};
use crate::blas::dcopy;
use crate::larr::dlaneg;
use crate::ptr::{D, J};

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;
const TWO: f64 = 2.0;
const FOUR: f64 = 4.0;
const HALF: f64 = 0.5;
const QUART: f64 = 0.25;

/// `DLARUV`'s `MM` table (`dlaruv.f` `DATA` statements), row-major
/// `MM( I, 1..4 )`.
#[rustfmt::skip]
const MM: [[i32; 4]; 128] = [
    [494, 322, 2508, 2549], [2637, 789, 3754, 1145], [255, 1440, 1766, 2253], [2008, 752, 3572, 305],
    [1253, 2859, 2893, 3301], [3344, 123, 307, 1065], [4084, 1848, 1297, 3133], [1739, 643, 3966, 2913],
    [3143, 2405, 758, 3285], [3468, 2638, 2598, 1241], [688, 2344, 3406, 1197], [1657, 46, 2922, 3729],
    [1238, 3814, 1038, 2501], [3166, 913, 2934, 1673], [1292, 3649, 2091, 541], [3422, 339, 2451, 2753],
    [1270, 3808, 1580, 949], [2016, 822, 1958, 2361], [154, 2832, 2055, 1165], [2862, 3078, 1507, 4081],
    [697, 3633, 1078, 2725], [1706, 2970, 3273, 3305], [491, 637, 17, 3069], [931, 2249, 854, 3617],
    [1444, 2081, 2916, 3733], [444, 4019, 3971, 409], [3577, 1478, 2889, 2157], [3944, 242, 3831, 1361],
    [2184, 481, 2621, 3973], [1661, 2075, 1541, 1865], [3482, 4058, 893, 2525], [657, 622, 736, 1409],
    [3023, 3376, 3992, 3445], [3618, 812, 787, 3577], [1267, 234, 2125, 77], [1828, 641, 2364, 3761],
    [164, 4005, 2460, 2149], [3798, 1122, 257, 1449], [3087, 3135, 1574, 3005], [2400, 2640, 3912, 225],
    [2870, 2302, 1216, 85], [3876, 40, 3248, 3673], [1905, 1832, 3401, 3117], [1593, 2247, 2124, 3089],
    [1797, 2034, 2762, 1349], [1234, 2637, 149, 2057], [3460, 1287, 2245, 413], [328, 1691, 166, 65],
    [2861, 496, 466, 1845], [1950, 1597, 4018, 697], [617, 2394, 1399, 3085], [2070, 2584, 190, 3441],
    [3331, 1843, 2879, 1573], [769, 336, 153, 3689], [1558, 1472, 2320, 2941], [2412, 2407, 18, 929],
    [2800, 433, 712, 533], [189, 2096, 2159, 2841], [287, 1761, 2318, 4077], [2045, 2810, 2091, 721],
    [1227, 566, 3443, 2821], [2838, 442, 1510, 2249], [209, 41, 449, 2397], [2770, 1238, 1956, 2817],
    [3654, 1086, 2201, 245], [3993, 603, 3137, 1913], [192, 840, 3399, 1997], [2253, 3168, 1321, 3121],
    [3491, 1499, 2271, 997], [2889, 1084, 3667, 1833], [2857, 3438, 2703, 2877], [2094, 2408, 629, 1633],
    [1818, 1589, 2365, 981], [688, 2391, 2431, 2009], [1407, 288, 1113, 941], [634, 26, 3922, 2449],
    [3231, 512, 2554, 197], [815, 1456, 184, 2441], [3524, 171, 2099, 285], [1914, 1677, 3228, 1473],
    [516, 2657, 4012, 2741], [164, 2270, 1921, 3129], [303, 2587, 3452, 909], [2144, 2961, 3901, 2801],
    [3480, 1970, 572, 421], [119, 1817, 3309, 4073], [3357, 676, 3171, 2813], [837, 1410, 817, 2337],
    [2826, 3723, 3039, 1429], [2332, 2803, 1696, 1177], [2089, 3185, 1256, 1901], [3780, 184, 3715, 81],
    [1700, 663, 2077, 1669], [3712, 499, 3019, 2633], [150, 3784, 1497, 2269], [2000, 1631, 1101, 129],
    [3375, 1925, 717, 1141], [1621, 3912, 51, 249], [3090, 1398, 981, 3917], [3765, 1349, 1978, 2481],
    [1149, 1441, 1813, 3941], [3146, 2224, 3881, 2217], [33, 2411, 76, 2749], [3082, 1907, 3846, 3041],
    [2741, 3192, 3694, 1877], [359, 2786, 1682, 345], [3316, 382, 124, 2861], [1749, 37, 1660, 1809],
    [185, 759, 3997, 3141], [2784, 2948, 479, 2825], [2202, 1862, 1141, 157], [2199, 3802, 886, 2881],
    [1364, 2423, 3514, 3637], [1244, 2051, 1301, 1465], [2020, 2295, 3604, 2829], [3160, 1332, 1888, 2161],
    [2785, 1832, 1836, 3365], [2772, 2405, 1990, 361], [1217, 3638, 2058, 2685], [1822, 3661, 692, 3745],
    [1245, 327, 1194, 2325], [2252, 3660, 20, 3609], [3904, 716, 3285, 3821], [2774, 1842, 2046, 3537],
    [997, 3987, 2107, 517], [2573, 1368, 3508, 3017], [1148, 1848, 3525, 2141], [545, 2366, 3801, 1537],
];

/// `DLARUV( ISEED, N, X )`: `n <= 128` uniform `(0, 1)` reals from the
/// multiplicative congruential generator, and the advanced seed.
///
/// # Safety
/// `x` has at least `n` entries.
pub(crate) unsafe fn dlaruv(iseed: &mut [i32; 4], n: usize, x: D) {
    const IPW2: i32 = 4096;
    const R: f64 = ONE / (IPW2 as f64);
    const LV: usize = 128;
    if n < 1 {
        return;
    }
    let mut i1 = iseed[0];
    let mut i2 = iseed[1];
    let mut i3 = iseed[2];
    let mut i4 = iseed[3];
    let (mut it1, mut it2, mut it3, mut it4) = (0i32, 0i32, 0i32, 0i32);

    for i in 1..=n.min(LV) {
        let m = MM[i - 1];
        loop {
            // Multiply the seed by the i-th power of the multiplier mod 2**48.
            it4 = i4 * m[3];
            it3 = it4 / IPW2;
            it4 = it4 - IPW2 * it3;
            it3 = it3 + i3 * m[3] + i4 * m[2];
            it2 = it3 / IPW2;
            it3 = it3 - IPW2 * it2;
            it2 = it2 + i2 * m[3] + i3 * m[2] + i4 * m[1];
            it1 = it2 / IPW2;
            it2 = it2 - IPW2 * it1;
            it1 = it1 + i1 * m[3] + i2 * m[2] + i3 * m[1] + i4 * m[0];
            it1 = it1 % IPW2;

            // Convert the 48-bit integer to a real in (0, 1).
            let v = R * (it1 as f64 + R * (it2 as f64 + R * (it3 as f64 + R * it4 as f64)));
            x.set(i - 1, v);
            if v == 1.0 {
                // All of the leading bits happened to be 1; iterate again.
                i1 += 2;
                i2 += 2;
                i3 += 2;
                i4 += 2;
                continue;
            }
            break;
        }
    }

    iseed[0] = it1;
    iseed[1] = it2;
    iseed[2] = it3;
    iseed[3] = it4;
}

/// `DLARNV( IDIST, ISEED, N, X )`.
///
/// `IDIST = 1` and `2` are bit-exact against the oracle
/// (`tests/ref_mrrr_components.rs::larnv_and_laruv`). `IDIST = 3` goes
/// through `LOG` and `COS`, where Rust's `f64::cos` and the glibc `cos`
/// gfortran calls differ in the last place, so it is **not** bit-exact and is
/// not gated. Nothing in the MRRR path reaches it: `DLARRE` calls
/// `DLARNV( 2, .. )`.
///
/// # Safety
/// `x` has at least `n` entries.
pub(crate) unsafe fn dlarnv(idist: i32, iseed: &mut [i32; 4], n: usize, x: D) {
    const LV: usize = 128;
    const TWOPI: f64 = 6.283_185_307_179_586_5;
    let mut u = [0.0f64; LV];
    let uv = D::new(&mut u);
    let mut iv = 1usize;
    while iv <= n {
        let il = (LV / 2).min(n - iv + 1);
        let il2 = if idist == 3 { 2 * il } else { il };
        dlaruv(iseed, il2, uv);
        if idist == 1 {
            for i in 1..=il {
                x.set(iv + i - 1 - 1, uv.at(i - 1));
            }
        } else if idist == 2 {
            for i in 1..=il {
                x.set(iv + i - 1 - 1, TWO * uv.at(i - 1) - ONE);
            }
        } else if idist == 3 {
            for i in 1..=il {
                x.set(iv + i - 1 - 1, (-TWO * uv.at(2 * i - 2).ln()).sqrt() * (TWOPI * uv.at(2 * i - 1)).cos());
            }
        }
        iv += LV / 2;
    }
}

/// `DLARRA( N, D, E, E2, SPLTOL, TNRM, NSPLIT, ISPLIT, INFO )`.
///
/// Returns `(nsplit, info)`.
///
/// # Safety
/// `d` has `n` entries, `e`/`e2` at least `n - 1`, `isplit` at least `n`.
pub(crate) unsafe fn dlarra(n: usize, d: D, e: D, e2: D, spltol: f64, tnrm: f64, isplit: J) -> (i32, i32) {
    let info = 0;
    let mut nsplit: usize = 1;
    if n == 0 {
        return (nsplit as i32, info);
    }
    if spltol < ZERO {
        // Criterion based on absolute off-diagonal value.
        let tmp1 = spltol.abs() * tnrm;
        for i in 1..=n - 1 {
            let eabs = e.at(i - 1).abs();
            if eabs <= tmp1 {
                e.set(i - 1, ZERO);
                e2.set(i - 1, ZERO);
                isplit.set(nsplit - 1, i as i32);
                nsplit += 1;
            }
        }
    } else {
        // Criterion that guarantees relative accuracy.
        for i in 1..=n - 1 {
            let eabs = e.at(i - 1).abs();
            if eabs <= spltol * d.at(i - 1).abs().sqrt() * d.at(i).abs().sqrt() {
                e.set(i - 1, ZERO);
                e2.set(i - 1, ZERO);
                isplit.set(nsplit - 1, i as i32);
                nsplit += 1;
            }
        }
    }
    isplit.set(nsplit - 1, n as i32);
    (nsplit as i32, info)
}

/// `DLARRR( N, D, E, INFO )`: whether relative-accuracy computations are
/// promised (`INFO = 0`) or not (`INFO = 1`).
///
/// # Safety
/// `d` has `n` entries, `e` at least `n - 1`.
pub(crate) unsafe fn dlarrr(n: usize, d: D, e: D) -> i32 {
    const RELCOND: f64 = 0.999;
    if n == 0 {
        return 0;
    }
    // Default: do NOT go for relative-accuracy preserving computations.
    let safmin = SAFMIN;
    let eps = PREC;
    let smlnum = safmin / eps;
    let rmin = smlnum.sqrt();

    // Test for scaled diagonal dominance.
    let mut yesrel = true;
    let mut offdig = ZERO;
    let mut tmp = d.at(0).abs().sqrt();
    if tmp < rmin {
        yesrel = false;
    }
    if yesrel {
        for i in 2..=n {
            let tmp2 = d.at(i - 1).abs().sqrt();
            if tmp2 < rmin {
                yesrel = false;
            }
            if !yesrel {
                break;
            }
            let offdig2 = e.at(i - 2).abs() / (tmp * tmp2);
            if offdig + offdig2 >= RELCOND {
                yesrel = false;
            }
            if !yesrel {
                break;
            }
            tmp = tmp2;
            offdig = offdig2;
        }
    }

    if yesrel { 0 } else { 1 }
}

/// `dlarrb.f:243` / `dlarrj.f:210`:
/// `MAXITR = INT((LOG(SPDIAM+PIVMIN)-LOG(PIVMIN))/LOG(TWO))+2`.
fn maxitr(spdiam: f64, pivmin: f64) -> i32 {
    (((spdiam + pivmin).ln() - pivmin.ln()) / TWO.ln()) as i32 + 2
}

/// `DLARRB( N, D, LLD, IFIRST, ILAST, RTOL1, RTOL2, OFFSET, W, WGAP, WERR,
/// WORK, IWORK, PIVMIN, SPDIAM, TWIST, INFO )`.
///
/// # Safety
/// `d` has `n` entries, `lld` at least `n - 1`; `w`, `wgap`, `werr` are
/// indexed `I - OFFSET`; `work` has `2 * n`, `iwork` has `2 * n`.
pub(crate) unsafe fn dlarrb(
    n: usize,
    d: D,
    lld: D,
    ifirst: usize,
    ilast: usize,
    rtol1: f64,
    rtol2: f64,
    offset: usize,
    w: D,
    wgap: D,
    werr: D,
    work: D,
    iwork: J,
    pivmin: f64,
    spdiam: f64,
    twist: i32,
) -> i32 {
    let info = 0;
    if n == 0 {
        return info;
    }
    let maxitr = maxitr(spdiam, pivmin);
    let mnwdth = TWO * pivmin;

    let mut r = twist as usize;
    if twist < 1 || twist > n as i32 {
        r = n;
    }

    // Unconverged intervals in [ WORK(2*I-1), WORK(2*I) ] as a linked list.
    let mut i1 = ifirst;
    let mut nint: i32 = 0;
    let mut prev: usize = 0;

    let mut rgap = wgap.at(i1 - offset - 1);
    for i in i1..=ilast {
        let k = 2 * i;
        let ii = i - offset;
        let mut left = w.at(ii - 1) - werr.at(ii - 1);
        let mut right = w.at(ii - 1) + werr.at(ii - 1);
        let lgap = rgap;
        rgap = wgap.at(ii - 1);
        let gap = fmin(lgap, rgap);

        // Do while( NEGCNT(LEFT).GT.I-1 ).
        let mut back = werr.at(ii - 1);
        loop {
            let negcnt = dlaneg(n, d, lld, left, pivmin, r);
            if negcnt > i as i32 - 1 {
                left = left - back;
                back = TWO * back;
            } else {
                break;
            }
        }

        // Do while( NEGCNT(RIGHT).LT.I ).
        let mut back = werr.at(ii - 1);
        let negcnt;
        loop {
            let nc = dlaneg(n, d, lld, right, pivmin, r);
            if nc < i as i32 {
                right = right + back;
                back = TWO * back;
            } else {
                negcnt = nc;
                break;
            }
        }
        let width = HALF * (left - right).abs();
        let tmp = fmax(left.abs(), right.abs());
        let cvrgd = fmax(rtol1 * gap, rtol2 * tmp);
        if width <= cvrgd || width <= mnwdth {
            // Already converged; remove it from the list.
            iwork.set(k - 1 - 1, -1);
            if i == i1 && i < ilast {
                i1 = i + 1;
            }
            if prev >= i1 && i <= ilast {
                iwork.set(2 * prev - 1 - 1, i as i32 + 1);
            }
        } else {
            prev = i;
            nint += 1;
            iwork.set(k - 1 - 1, i as i32 + 1);
            iwork.set(k - 1, negcnt);
        }
        work.set(k - 1 - 1, left);
        work.set(k - 1, right);
    }

    // Do while( NINT.GT.0 ) and ITER <= MAXITR.
    let mut iter: i32 = 0;
    loop {
        let mut prev = i1 as i32 - 1;
        let mut i = i1;
        let olnint = nint;

        for _ip in 1..=olnint {
            let k = 2 * i;
            let ii = i - offset;
            let rgap = wgap.at(ii - 1);
            let mut lgap = rgap;
            if ii > 1 {
                lgap = wgap.at(ii - 2);
            }
            let gap = fmin(lgap, rgap);
            let next = iwork.at(k - 1 - 1);
            let left = work.at(k - 1 - 1);
            let right = work.at(k - 1);
            let mid = HALF * (left + right);

            let width = right - mid;
            let tmp = fmax(left.abs(), right.abs());
            let cvrgd = fmax(rtol1 * gap, rtol2 * tmp);
            if width <= cvrgd || width <= mnwdth || iter == maxitr {
                nint -= 1;
                iwork.set(k - 1 - 1, 0);
                if i1 == i {
                    i1 = next as usize;
                } else if prev >= i1 as i32 {
                    iwork.set(2 * prev as usize - 1 - 1, next);
                }
                i = next as usize;
                continue;
            }
            prev = i as i32;

            // One bisection step.
            let negcnt = dlaneg(n, d, lld, mid, pivmin, r);
            if negcnt <= i as i32 - 1 {
                work.set(k - 1 - 1, mid);
            } else {
                work.set(k - 1, mid);
            }
            i = next as usize;
        }
        iter += 1;
        if !(nint > 0 && iter <= maxitr) {
            break;
        }
    }

    // All intervals have converged.
    for i in ifirst..=ilast {
        let k = 2 * i;
        let ii = i - offset;
        if iwork.at(k - 1 - 1) == 0 {
            w.set(ii - 1, HALF * (work.at(k - 1 - 1) + work.at(k - 1)));
            werr.set(ii - 1, work.at(k - 1) - w.at(ii - 1));
        }
    }

    for i in ifirst + 1..=ilast {
        let ii = i - offset;
        wgap.set(ii - 1 - 1, fmax(ZERO, w.at(ii - 1) - werr.at(ii - 1) - w.at(ii - 2) - werr.at(ii - 2)));
    }

    info
}

/// `DLARRJ( N, D, E2, IFIRST, ILAST, RTOL, OFFSET, W, WERR, WORK, IWORK,
/// PIVMIN, SPDIAM, INFO )`.
///
/// # Safety
/// `d` has `n` entries, `e2` at least `n - 1`; `work`/`iwork` have `2 * n`.
pub(crate) unsafe fn dlarrj(
    n: usize,
    d: D,
    e2: D,
    ifirst: usize,
    ilast: usize,
    rtol: f64,
    offset: usize,
    w: D,
    werr: D,
    work: D,
    iwork: J,
    pivmin: f64,
    spdiam: f64,
) -> i32 {
    let info = 0;
    if n == 0 {
        return info;
    }
    let maxitr = maxitr(spdiam, pivmin);

    let mut i1 = ifirst;
    let i2 = ilast;
    let mut nint: i32 = 0;
    let mut prev: usize = 0;

    for i in i1..=i2 {
        let k = 2 * i;
        let ii = i - offset;
        let mut left = w.at(ii - 1) - werr.at(ii - 1);
        let mid = w.at(ii - 1);
        let mut right = w.at(ii - 1) + werr.at(ii - 1);
        let width = right - mid;
        let tmp = fmax(left.abs(), right.abs());

        if width < rtol * tmp {
            // Already converged; remove it from the list.
            iwork.set(k - 1 - 1, -1);
            if i == i1 && i < i2 {
                i1 = i + 1;
            }
            if prev >= i1 && i <= i2 {
                iwork.set(2 * prev - 1 - 1, i as i32 + 1);
            }
        } else {
            prev = i;
            // Do while( CNT(LEFT).GT.I-1 ).
            let mut fac = ONE;
            loop {
                let mut cnt: i32 = 0;
                let s = left;
                let mut dplus = d.at(0) - s;
                if dplus < ZERO {
                    cnt += 1;
                }
                for j in 2..=n {
                    dplus = d.at(j - 1) - s - e2.at(j - 2) / dplus;
                    if dplus < ZERO {
                        cnt += 1;
                    }
                }
                if cnt > i as i32 - 1 {
                    left = left - werr.at(ii - 1) * fac;
                    fac = TWO * fac;
                } else {
                    break;
                }
            }

            // Do while( CNT(RIGHT).LT.I ).
            let mut fac = ONE;
            let cnt_final;
            loop {
                let mut cnt: i32 = 0;
                let s = right;
                let mut dplus = d.at(0) - s;
                if dplus < ZERO {
                    cnt += 1;
                }
                for j in 2..=n {
                    dplus = d.at(j - 1) - s - e2.at(j - 2) / dplus;
                    if dplus < ZERO {
                        cnt += 1;
                    }
                }
                if cnt < i as i32 {
                    right = right + werr.at(ii - 1) * fac;
                    fac = TWO * fac;
                } else {
                    cnt_final = cnt;
                    break;
                }
            }
            nint += 1;
            iwork.set(k - 1 - 1, i as i32 + 1);
            iwork.set(k - 1, cnt_final);
        }
        work.set(k - 1 - 1, left);
        work.set(k - 1, right);
    }

    let savi1 = i1;

    let mut iter: i32 = 0;
    loop {
        let mut prev = i1 as i32 - 1;
        let mut i = i1;
        let olnint = nint;

        for _p in 1..=olnint {
            let k = 2 * i;
            let next = iwork.at(k - 1 - 1);
            let left = work.at(k - 1 - 1);
            let right = work.at(k - 1);
            let mid = HALF * (left + right);

            let width = right - mid;
            let tmp = fmax(left.abs(), right.abs());

            if width < rtol * tmp || iter == maxitr {
                nint -= 1;
                iwork.set(k - 1 - 1, 0);
                if i1 == i {
                    i1 = next as usize;
                } else if prev >= i1 as i32 {
                    iwork.set(2 * prev as usize - 1 - 1, next);
                }
                i = next as usize;
                continue;
            }
            prev = i as i32;

            // One bisection step.
            let mut cnt: i32 = 0;
            let s = mid;
            let mut dplus = d.at(0) - s;
            if dplus < ZERO {
                cnt += 1;
            }
            for j in 2..=n {
                dplus = d.at(j - 1) - s - e2.at(j - 2) / dplus;
                if dplus < ZERO {
                    cnt += 1;
                }
            }
            if cnt <= i as i32 - 1 {
                work.set(k - 1 - 1, mid);
            } else {
                work.set(k - 1, mid);
            }
            i = next as usize;
        }
        iter += 1;
        if !(nint > 0 && iter <= maxitr) {
            break;
        }
    }

    // All intervals have converged.
    for i in savi1..=ilast {
        let k = 2 * i;
        let ii = i - offset;
        if iwork.at(k - 1 - 1) == 0 {
            w.set(ii - 1, HALF * (work.at(k - 1 - 1) + work.at(k - 1)));
            werr.set(ii - 1, work.at(k - 1) - w.at(ii - 1));
        }
    }

    info
}

/// `DLARRF( N, D, L, LD, CLSTRT, CLEND, W, WGAP, WERR, SPDIAM, CLGAPL,
/// CLGAPR, PIVMIN, SIGMA, DPLUS, LPLUS, WORK, INFO )`.
///
/// Returns `(sigma, info)`.
///
/// # Safety
/// `d`, `dplus` have `n` entries; `l`, `ld`, `lplus` at least `n - 1`;
/// `work` has `2 * n`.
pub(crate) unsafe fn dlarrf(
    n: usize,
    d: D,
    l: D,
    ld: D,
    clstrt: usize,
    clend: usize,
    w: D,
    wgap: D,
    werr: D,
    spdiam: f64,
    clgapl: f64,
    clgapr: f64,
    pivmin: f64,
    dplus: D,
    lplus: D,
    work: D,
) -> (f64, i32) {
    const MAXGROWTH1: f64 = 8.0;
    const MAXGROWTH2: f64 = 8.0;
    const KTRYMAX: i32 = 1;
    const SLEFT: i32 = 1;
    const SRIGHT: i32 = 2;

    let mut sigma = ZERO;
    if n == 0 {
        return (sigma, 0);
    }

    let fact = (1i32 << KTRYMAX) as f64;
    let eps = PREC;
    // `SHIFT = 0` (`dlarrf.f:248`) is the reference's initial value; every
    // path out of the loop sets it to SLEFT or SRIGHT before it is read.
    #[allow(unused_assignments)]
    let mut shift: i32 = 0;
    let mut forcer = false;

    // Setting NOFAIL to .FALSE. is the quick fix for LAPACK bug 113.
    let nofail = false;

    // Average gap length of the cluster.
    let clwdth = (w.at(clend - 1) - w.at(clstrt - 1)).abs() + werr.at(clend - 1) + werr.at(clstrt - 1);
    let avgap = clwdth / ((clend - clstrt) as f64);
    let mingap = fmin(clgapl, clgapr);
    // Initial values for shifts to both ends of the cluster.
    let mut lsigma = fmin(w.at(clstrt - 1), w.at(clend - 1)) - werr.at(clstrt - 1);
    let mut rsigma = fmax(w.at(clstrt - 1), w.at(clend - 1)) + werr.at(clend - 1);

    // A small fudge, to really shift to the outside.
    lsigma = lsigma - lsigma.abs() * FOUR * eps;
    rsigma = rsigma + rsigma.abs() * FOUR * eps;

    // Upper bounds for how much to back off the initial shifts.
    let ldmax = QUART * mingap + TWO * pivmin;
    let rdmax = QUART * mingap + TWO * pivmin;

    let mut ldelta = fmax(avgap, wgap.at(clstrt - 1)) / fact;
    let mut rdelta = fmax(avgap, wgap.at(clend - 1 - 1)) / fact;

    // Record of the best representation found.
    let s0 = SAFMIN;
    let mut smlgrowth = ONE / s0;
    let fail = ((n - 1) as f64) * mingap / (spdiam * eps);
    let fail2 = ((n - 1) as f64) * mingap / (spdiam * eps.sqrt());
    let mut bestshift = lsigma;

    let mut ktry: i32 = 0;
    let growthbound = MAXGROWTH1 * spdiam;
    let mut indx: i32 = 0;

    'outer: loop {
        // Label 5.
        let mut sawnan1 = false;
        let mut sawnan2 = false;
        // Do not back off too much of the initial shifts.
        ldelta = fmin(ldmax, ldelta);
        rdelta = fmin(rdmax, rdelta);

        // Left end.
        let mut s = -lsigma;
        dplus.set(0, d.at(0) + s);
        if dplus.at(0).abs() < pivmin {
            dplus.set(0, -pivmin);
            sawnan1 = true;
        }
        let mut max1 = dplus.at(0).abs();
        for i in 1..=n - 1 {
            lplus.set(i - 1, ld.at(i - 1) / dplus.at(i - 1));
            s = s * lplus.at(i - 1) * l.at(i - 1) - lsigma;
            dplus.set(i, d.at(i) + s);
            if dplus.at(i).abs() < pivmin {
                dplus.set(i, -pivmin);
                sawnan1 = true;
            }
            max1 = fmax(max1, dplus.at(i).abs());
        }
        sawnan1 = sawnan1 || max1.is_nan();

        if forcer || (max1 <= growthbound && !sawnan1) {
            sigma = lsigma;
            shift = SLEFT;
            break 'outer; // GOTO 100
        }

        // Right end.
        let mut s = -rsigma;
        work.set(0, d.at(0) + s);
        if work.at(0).abs() < pivmin {
            work.set(0, -pivmin);
            sawnan2 = true;
        }
        let mut max2 = work.at(0).abs();
        for i in 1..=n - 1 {
            work.set(n + i - 1, ld.at(i - 1) / work.at(i - 1));
            s = s * work.at(n + i - 1) * l.at(i - 1) - rsigma;
            work.set(i, d.at(i) + s);
            if work.at(i).abs() < pivmin {
                work.set(i, -pivmin);
                sawnan2 = true;
            }
            max2 = fmax(max2, work.at(i).abs());
        }
        sawnan2 = sawnan2 || max2.is_nan();

        if forcer || (max2 <= growthbound && !sawnan2) {
            sigma = rsigma;
            shift = SRIGHT;
            break 'outer; // GOTO 100
        }

        // Both shifts led to too much element growth.
        let mut goto50 = false;
        if sawnan1 && sawnan2 {
            goto50 = true;
        } else {
            if !sawnan1 {
                indx = 1;
                if max1 <= smlgrowth {
                    smlgrowth = max1;
                    bestshift = lsigma;
                }
            }
            if !sawnan2 {
                if sawnan1 || max2 <= max1 {
                    indx = 2;
                }
                if max2 <= smlgrowth {
                    smlgrowth = max2;
                    bestshift = rsigma;
                }
            }
        }

        if !goto50 {
            // The refined RRR test, for isolated clusters and no NaN.
            let dorrr1 = clwdth < mingap / (128.0f64) && fmin(max1, max2) < fail2 && !sawnan1 && !sawnan2;
            let tryrrr1 = true;
            if tryrrr1 && dorrr1 {
                if indx == 1 {
                    let mut tmp = dplus.at(n - 1).abs();
                    let mut znm2 = ONE;
                    let mut prod = ONE;
                    let mut oldp = ONE;
                    // `DO 15 I = N-1, 1, -1`, ascending counter, same order.
                    for t in 0..n - 1 {
                        let i = (n - 1) - t;
                        if prod <= eps {
                            prod = ((dplus.at(i) * work.at(n + i)) / (dplus.at(i - 1) * work.at(n + i - 1))) * oldp;
                        } else {
                            prod = prod * work.at(n + i - 1).abs();
                        }
                        oldp = prod;
                        znm2 = znm2 + prod * prod;
                        tmp = fmax(tmp, (dplus.at(i - 1) * prod).abs());
                    }
                    let rrr1 = tmp / (spdiam * znm2.sqrt());
                    if rrr1 <= MAXGROWTH2 {
                        sigma = lsigma;
                        shift = SLEFT;
                        break 'outer; // GOTO 100
                    }
                } else if indx == 2 {
                    let mut tmp = work.at(n - 1).abs();
                    let mut znm2 = ONE;
                    let mut prod = ONE;
                    let mut oldp = ONE;
                    for t in 0..n - 1 {
                        let i = (n - 1) - t;
                        if prod <= eps {
                            prod = ((work.at(i) * lplus.at(i)) / (work.at(i - 1) * lplus.at(i - 1))) * oldp;
                        } else {
                            prod = prod * lplus.at(i - 1).abs();
                        }
                        oldp = prod;
                        znm2 = znm2 + prod * prod;
                        tmp = fmax(tmp, (work.at(i - 1) * prod).abs());
                    }
                    let rrr2 = tmp / (spdiam * znm2.sqrt());
                    if rrr2 <= MAXGROWTH2 {
                        sigma = rsigma;
                        shift = SRIGHT;
                        break 'outer; // GOTO 100
                    }
                }
            }
        }

        // Label 50.
        if ktry < KTRYMAX {
            // Both shifts failed the RRR test too; back off to the outside.
            lsigma = fmax(lsigma - ldelta, lsigma - ldmax);
            rsigma = fmin(rsigma + rdelta, rsigma + rdmax);
            ldelta = TWO * ldelta;
            rdelta = TWO * rdelta;
            ktry += 1;
            continue 'outer; // GOTO 5
        } else if smlgrowth < fail || nofail {
            // Take the best representation found.
            lsigma = bestshift;
            rsigma = bestshift;
            forcer = true;
            continue 'outer; // GOTO 5
        } else {
            return (sigma, 1);
        }
    }

    // Label 100.
    if shift == SRIGHT {
        // Store the new L and D back into DPLUS, LPLUS.
        dcopy(n, work, 1, dplus, 1);
        dcopy(n - 1, work.off(n), 1, lplus, 1);
    }

    (sigma, 0)
}
