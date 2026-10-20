//! `DSTEMR` (Reference-LAPACK v3.12.1 `SRC/dstemr.f`): the MRRR eigensolver
//! for a symmetric tridiagonal matrix.
//!
//! **`RANGE` coverage.** The public entry point exposes `RANGE = 'A'` only —
//! `'V'` and `'I'` need `DLARRD`/`DLAEBZ`, which are not transcribed (see
//! [`crate::larre`]). The driver keeps the reference's structure so the two
//! read alike; `ALLEIG` is the only branch reached.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::aux::{PREC, SAFMIN, dlae2, dlaev2, dlanst_m, dlasrt_inc};
use crate::blas::{dcopy, dscal, dswap};
use crate::larr2::dlarrj;
use crate::larre::{Range, dlarre};
use crate::larrv::dlarrv;
use crate::ptr::{D, J, ix};
use crate::{Job, LapackError};

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;
const FOUR: f64 = 4.0;
const MINRGP: f64 = 1.0e-3;

/// `(LWORK, LIWORK)` for [`dstemr`] (`dstemr.f:389`): `18*n`/`10*n` with
/// eigenvectors, `12*n`/`8*n` without.
pub fn dstemr_min_work_len(n: usize, job: Job) -> (usize, usize) {
    if matches!(job, Job::Vectors) { (18 * n, 10 * n) } else { (12 * n, 8 * n) }
}

/// The same as [`dstemr_min_work_len`]: `DSTEMR`'s workspace is not blocked,
/// so the optimal and minimum lengths coincide. Kept for symmetry with
/// [`crate::syevd_work_len`].
pub fn dstemr_work_len(n: usize, job: Job) -> (usize, usize) {
    dstemr_min_work_len(n, job)
}

/// `DSTEMR( JOBZ, 'A', N, D, E, .., M, W, Z, LDZ, NZC, ISUPPZ, TRYRAC, WORK,
/// LWORK, IWORK, LIWORK, INFO )`, the internal form.
///
/// Returns `(m, tryrac, info)`. `d` and `e` are destroyed; `e` must have `n`
/// entries (`DLARRE` writes `E( N )`).
///
/// # Safety
/// `d`, `e` have `n` entries; `w` has `n`; `z` has `ldz * n` when
/// `wantz`; `isuppz` has `2 * n`; `work`/`iwork` are at least the lengths
/// [`dstemr_min_work_len`] reports.
pub(crate) unsafe fn dstemr_a(
    wantz: bool,
    n: usize,
    d: D,
    e: D,
    w: D,
    z: D,
    ldz: usize,
    isuppz: J,
    tryrac_in: bool,
    work: D,
    iwork: J,
) -> (usize, bool, i32) {
    let mut tryrac = tryrac_in;
    let mut laeswap = false;

    // Machine constants.
    let safmin = SAFMIN;
    let eps = PREC;
    let smlnum = safmin / eps;
    let bignum = ONE / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = (bignum.sqrt()).min(ONE / safmin.sqrt().sqrt());

    // N = 0, 1, 2.
    let mut m: usize = 0;
    if n == 0 {
        return (0, tryrac, 0);
    }

    if n == 1 {
        m = 1;
        w.set(0, d.at(0));
        if wantz {
            z.set(ix(1, 1, ldz), ONE);
            isuppz.set(0, 1);
            isuppz.set(1, 1);
        }
        return (m, tryrac, 0);
    }

    let mut nsplit: i32 = 0;

    if n == 2 {
        // `CS`/`SN` stay zero on the `JOBZ = 'N'` path, as in the reference.
        #[allow(unused_assignments)]
        let (mut r1, mut r2, mut cs, mut sn) = (ZERO, ZERO, ZERO, ZERO);
        if !wantz {
            let (a, b) = dlae2(d.at(0), e.at(0), d.at(1));
            r1 = a;
            r2 = b;
        } else {
            let (a, b, c, s) = dlaev2(d.at(0), e.at(0), d.at(1));
            r1 = a;
            r2 = b;
            cs = c;
            sn = s;
        }
        // DLAE2/DLAEV2 satisfy |R1| >= |R2|; the code below needs R1 >= R2.
        if r1 < r2 {
            e.set(1, r1);
            r1 = r2;
            r2 = e.at(1);
            laeswap = true;
        }
        m += 1;
        w.set(m - 1, r2);
        if wantz {
            if laeswap {
                z.set(ix(1, m, ldz), cs);
                z.set(ix(2, m, ldz), sn);
            } else {
                z.set(ix(1, m, ldz), -sn);
                z.set(ix(2, m, ldz), cs);
            }
            // At most one of SN and CS can be zero.
            if sn != ZERO {
                if cs != ZERO {
                    isuppz.set(2 * m - 1 - 1, 1);
                    isuppz.set(2 * m - 1, 2);
                } else {
                    isuppz.set(2 * m - 1 - 1, 1);
                    isuppz.set(2 * m - 1, 1);
                }
            } else {
                isuppz.set(2 * m - 1 - 1, 2);
                isuppz.set(2 * m - 1, 2);
            }
        }
        m += 1;
        w.set(m - 1, r1);
        if wantz {
            if laeswap {
                z.set(ix(1, m, ldz), -sn);
                z.set(ix(2, m, ldz), cs);
            } else {
                z.set(ix(1, m, ldz), cs);
                z.set(ix(2, m, ldz), sn);
            }
            if sn != ZERO {
                if cs != ZERO {
                    isuppz.set(2 * m - 1 - 1, 1);
                    isuppz.set(2 * m - 1, 2);
                } else {
                    isuppz.set(2 * m - 1 - 1, 1);
                    isuppz.set(2 * m - 1, 1);
                }
            } else {
                isuppz.set(2 * m - 1 - 1, 2);
                isuppz.set(2 * m - 1, 2);
            }
        }
    } else {
        // General N.
        let indgrs = 1usize;
        let inderr = 2 * n + 1;
        let indgp = 3 * n + 1;
        let indd = 4 * n + 1;
        let inde2 = 5 * n + 1;
        let indwrk = 6 * n + 1;

        let iinspl = 1usize;
        let iindbl = n + 1;
        let iindw = 2 * n + 1;
        let iindwk = 3 * n + 1;

        // Scale the matrix to the allowable range, if necessary.
        let mut scale = ONE;
        let mut tnrm = dlanst_m(n, d, e);
        if tnrm > ZERO && tnrm < rmin {
            scale = rmin / tnrm;
        } else if tnrm > rmax {
            scale = rmax / tnrm;
        }
        if scale != ONE {
            dscal(n, scale, d, 1);
            dscal(n - 1, scale, e, 1);
            tnrm = tnrm * scale;
        }

        // THRESH is DLARRE's splitting parameter: negative forces the
        // off-diagonal-size criterion, positive the relative-accuracy one.
        let iinfo = if tryrac { crate::larr2::dlarrr(n, d, e) } else { -1 };
        let thresh = if iinfo == 0 {
            eps
        } else {
            // Relative accuracy is wanted but T does not guarantee it.
            tryrac = false;
            -eps
        };

        if tryrac {
            // The original diagonal, needed for relative accuracy.
            dcopy(n, d, 1, work.off(indd - 1), 1);
        }
        // Squares of the off-diagonal values of T.
        for jj in 1..=n - 1 {
            work.set(inde2 + jj - 1 - 1, e.at(jj - 1) * e.at(jj - 1));
        }

        // Bisection tolerances.
        let (rtol1, mut rtol2) = if !wantz {
            // DLARRE computes the eigenvalues to full precision.
            (FOUR * eps, FOUR * eps)
        } else {
            // DLARRV will refine them, so DLARRE can be less accurate.
            (eps.sqrt(), (eps.sqrt() * 5.0e-3).max(FOUR * eps))
        };

        let (wl, _wu, ns, mm, pivmin, iinfo) = dlarre(
            Range::All,
            n,
            ZERO,
            ZERO,
            0,
            0,
            d,
            e,
            work.off(inde2 - 1),
            rtol1,
            rtol2,
            thresh,
            iwork.off(iinspl - 1),
            w,
            work.off(inderr - 1),
            work.off(indgp - 1),
            iwork.off(iindbl - 1),
            iwork.off(iindw - 1),
            work.off(indgrs - 1),
            work.off(indwrk - 1),
            iwork.off(iindwk - 1),
        );
        if iinfo != 0 {
            return (0, tryrac, 10 + iinfo.abs());
        }
        nsplit = ns;
        m = mm as usize;

        if wantz {
            let iinfo = dlarrv(
                n,
                wl,
                _wu,
                d,
                e,
                pivmin,
                iwork.off(iinspl - 1),
                m,
                1,
                m,
                MINRGP,
                rtol1,
                rtol2,
                w,
                work.off(inderr - 1),
                work.off(indgp - 1),
                iwork.off(iindbl - 1),
                iwork.off(iindw - 1),
                work.off(indgrs - 1),
                z,
                ldz,
                isuppz,
                work.off(indwrk - 1),
                iwork.off(iindwk - 1),
            );
            if iinfo != 0 {
                return (m, tryrac, 20 + iinfo.abs());
            }
        } else {
            // DLARRE computed the eigenvalues of the shifted root
            // representation; apply the shifts to get those of the original.
            for jj in 1..=m {
                let itmp = iwork.at(iindbl + jj - 1 - 1) as usize;
                w.set(jj - 1, w.at(jj - 1) + e.at(iwork.at(iinspl + itmp - 1 - 1) as usize - 1));
            }
        }

        if tryrac {
            // Refine the eigenvalues to be relatively accurate w.r.t. T.
            let mut ibegin = 1usize;
            let mut wbegin = 1usize;
            for jblk in 1..=iwork.at(iindbl + m - 1 - 1) as usize {
                let iend = iwork.at(iinspl + jblk - 1 - 1) as usize;
                let inn = iend - ibegin + 1;
                let mut wend = wbegin - 1;
                while wend < m && iwork.at(iindbl + wend - 1) == jblk as i32 {
                    wend += 1;
                }
                if wend < wbegin {
                    ibegin = iend + 1;
                    continue;
                }

                let offset = iwork.at(iindw + wbegin - 1 - 1) as usize - 1;
                let ifirst = iwork.at(iindw + wbegin - 1 - 1) as usize;
                let ilast = iwork.at(iindw + wend - 1 - 1) as usize;
                rtol2 = FOUR * eps;
                dlarrj(
                    inn,
                    work.off(indd + ibegin - 1 - 1),
                    work.off(inde2 + ibegin - 1 - 1),
                    ifirst,
                    ilast,
                    rtol2,
                    offset,
                    w.off(wbegin - 1),
                    work.off(inderr + wbegin - 1 - 1),
                    work.off(indwrk - 1),
                    iwork.off(iindwk - 1),
                    pivmin,
                    tnrm,
                );
                ibegin = iend + 1;
                wbegin = wend + 1;
            }
        }

        // Rescale the eigenvalues if the matrix was scaled.
        if scale != ONE {
            dscal(m, ONE / scale, w, 1);
        }
    }

    // Sort into increasing order, with the eigenvectors when present.
    if nsplit > 1 || n == 2 {
        if !wantz {
            dlasrt_inc(m, w);
        } else {
            for jj in 1..=m.saturating_sub(1) {
                let mut i = 0usize;
                let mut tmp = w.at(jj - 1);
                for k in jj + 1..=m {
                    if w.at(k - 1) < tmp {
                        i = k;
                        tmp = w.at(k - 1);
                    }
                }
                if i != 0 {
                    w.set(i - 1, w.at(jj - 1));
                    w.set(jj - 1, tmp);
                    dswap(n, z.off(ix(1, i, ldz)), 1, z.off(ix(1, jj, ldz)), 1);
                    let itmp = isuppz.at(2 * i - 1 - 1);
                    isuppz.set(2 * i - 1 - 1, isuppz.at(2 * jj - 1 - 1));
                    isuppz.set(2 * jj - 1 - 1, itmp);
                    let itmp = isuppz.at(2 * i - 1);
                    isuppz.set(2 * i - 1, isuppz.at(2 * jj - 1));
                    isuppz.set(2 * jj - 1, itmp);
                }
            }
        }
    }

    (m, tryrac, 0)
}

/// `DSTEMR( JOBZ, 'A', N, D, E, .., M, W, Z, N, N, ISUPPZ, TRYRAC, WORK,
/// LWORK, IWORK, LIWORK, INFO )`: all eigenvalues, and optionally all
/// eigenvectors, of a real symmetric tridiagonal matrix by MRRR.
///
/// `d` (length `n`) and `e` (length `n`; only `e[..n-1]` is read, `e[n-1]` is
/// scratch) are **destroyed**. `w` receives the `m` eigenvalues in ascending
/// order; with [`Job::Vectors`], `z` is column-major with `LDZ = n` and
/// `isuppz[2*i]`, `isuppz[2*i+1]` bracket the support of column `i`.
///
/// `tryrac` on input asks for eigenvalues that are relatively accurate with
/// respect to `T`; it comes back `false` when `T` does not warrant it. The
/// caller supplies `work`/`iwork` of at least [`dstemr_min_work_len`]; the
/// driver allocates nothing.
///
/// Returns `m`.
pub fn dstemr(
    job: Job,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    w: &mut [f64],
    z: &mut [f64],
    isuppz: &mut [i32],
    tryrac: &mut bool,
    work: &mut [f64],
    iwork: &mut [i32],
) -> Result<usize, LapackError> {
    let wantz = matches!(job, Job::Vectors);
    let (lw, liw) = dstemr_min_work_len(n, job);
    let need = |name: &'static str, got: usize, want: usize| -> Result<(), LapackError> {
        if got < want {
            Err(LapackError::InvalidArgument { arg: name, reason: format!("need {want} entries, got {got}") })
        } else {
            Ok(())
        }
    };
    need("d", d.len(), n)?;
    need("e", e.len(), n)?;
    need("w", w.len(), n)?;
    if wantz {
        need("z", z.len(), n * n)?;
        need("isuppz", isuppz.len(), 2 * n)?;
    }
    if work.len() < lw {
        return Err(LapackError::WorkspaceTooSmall { need: lw, got: work.len() });
    }
    if iwork.len() < liw {
        return Err(LapackError::WorkspaceTooSmall { need: liw, got: iwork.len() });
    }
    if n == 0 {
        return Ok(0);
    }
    let mut dummy = [0.0f64; 1];
    let mut dummyi = [0i32; 2];
    let zv = if wantz { D::new(z) } else { D::new(&mut dummy) };
    let sv = if wantz { J::new(isuppz) } else { J::new(&mut dummyi) };
    // SAFETY: every length is checked above; the slices are distinct.
    let (m, rac, info) = unsafe {
        dstemr_a(wantz, n, D::new(d), D::new(e), D::new(w), zv, n.max(1), sv, *tryrac, D::new(work), J::new(iwork))
    };
    *tryrac = rac;
    if info != 0 {
        return Err(LapackError::NoConvergence { routine: "dstemr", info: info as usize });
    }
    Ok(m)
}
