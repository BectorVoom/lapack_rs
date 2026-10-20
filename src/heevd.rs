//! `ZHEEVD` on planar storage. Transcribed from Reference-LAPACK v3.12.1
//! `SRC/zheevd.f`, `SRC/zstedc.f` (`COMPZ = 'I'`), `SRC/zsteqr.f`,
//! `SRC/zlasr.f`, `SRC/zlanhe.f`, `SRC/zlascl.f` and `SRC/dlanst.f`
//! (`NORM = 'I'`).
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::aux::{EPS, PREC, SAFMIN, SclType, dlae2, dlaev2, dlanst_m, dlapy2, dlartg, dlascl, dlaset_full, sign};
use crate::flavor::{Blas, RefBlas};
use crate::hetrd::{HostZOrm, HostZTrd, zhetrd_with, zunmtr_ln_with};
use crate::ptr::{D, I, V, ix};
use crate::stedc::{HostDc, SMLSIZ, dstedc_i_with};
use crate::steqr::dsterf_raw;
use crate::sytrd::SYTRD_NB;
use crate::zblas::{CONE, CZERO, Z, ZV, zswap};
use crate::{Job, LapackError, Uplo};

/// `DLANST( 'I', ... )` (= `'O'`, the 1-norm of the tridiagonal).
unsafe fn dlanst_i(n: usize, d: D, e: D) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return d.at(0).abs();
    }
    let mut anorm = d.at(0).abs() + e.at(0).abs();
    let sum = e.at(n - 2).abs() + d.at(n - 1).abs();
    if anorm < sum || sum.is_nan() {
        anorm = sum;
    }
    for i in 2..n {
        let sum = d.at(i - 1).abs() + e.at(i - 1).abs() + e.at(i - 2).abs();
        if anorm < sum || sum.is_nan() {
            anorm = sum;
        }
    }
    anorm
}

/// `ZLASR( 'R', 'V', DIRECT, M, N, C, S, A, LDA )` with real rotations.
unsafe fn zlasr_rv(forward: bool, m: usize, n: usize, c: D, s: D, a: ZV, lda: usize) {
    if m == 0 || n == 0 {
        return;
    }
    let step = |j: usize| {
        let ctemp = c.at(j - 1);
        let stemp = s.at(j - 1);
        if ctemp != 1.0 || stemp != 0.0 {
            let (zc, zs) = (Z::r(ctemp), Z::r(stemp));
            for i in 1..=m {
                let p1 = ix(i, j + 1, lda);
                let p0 = ix(i, j, lda);
                let temp = a.at(p1);
                a.set(p1, zc * temp - zs * a.at(p0));
                a.set(p0, zs * temp + zc * a.at(p0));
            }
        }
    };
    if forward {
        for j in 1..n {
            step(j);
        }
    } else {
        for j in (1..n).rev() {
            step(j);
        }
    }
}

/// `ZSTEQR( COMPZ = 'I', ... )`. `work` has `2*(n-1)` reals. Returns `INFO`.
pub(crate) unsafe fn zsteqr_i(n: usize, d: D, e: D, z: ZV, ldz: usize, work: D) -> usize {
    let mut info = 0usize;
    if n == 0 {
        return 0;
    }
    if n == 1 {
        z.set(0, CONE);
        return 0;
    }
    let dd = |i: usize| d.at(i - 1);
    let ee = |i: usize| e.at(i - 1);
    let eps = EPS;
    let eps2 = eps * eps;
    let safmin = SAFMIN;
    let safmax = 1.0 / safmin;
    let ssfmax = safmax.sqrt() / 3.0;
    let ssfmin = safmin.sqrt() / eps2;
    // ZLASET( 'Full', N, N, CZERO, CONE, Z, LDZ )
    for j in 1..=n {
        for i in 1..=n {
            z.set(ix(i, j, ldz), CZERO);
        }
    }
    for i in 1..=n {
        z.set(ix(i, i, ldz), CONE);
    }
    let nmaxit = n * 30;
    let mut jtot = 0usize;
    let mut l1 = 1usize;
    let nm1 = n - 1;

    'l10: loop {
        if l1 > n {
            break 'l10;
        }
        if l1 > 1 {
            e.set(l1 - 2, 0.0);
        }
        let mut m = n;
        if l1 <= nm1 {
            for mm in l1..=nm1 {
                let tst = ee(mm).abs();
                if tst == 0.0 {
                    m = mm;
                    break;
                }
                if tst <= (dd(mm).abs().sqrt() * dd(mm + 1).abs().sqrt()) * eps {
                    e.set(mm - 1, 0.0);
                    m = mm;
                    break;
                }
            }
        }
        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue 'l10;
        }
        let anorm = dlanst_i(lend - l + 1, d.off(l - 1), e.off(l - 1));
        let mut iscale = 0;
        if anorm == 0.0 {
            continue 'l10;
        }
        if anorm > ssfmax {
            iscale = 1;
            dlascl(SclType::G, anorm, ssfmax, lend - l + 1, 1, d.off(l - 1), n);
            dlascl(SclType::G, anorm, ssfmax, lend - l, 1, e.off(l - 1), n);
        } else if anorm < ssfmin {
            iscale = 2;
            dlascl(SclType::G, anorm, ssfmin, lend - l + 1, 1, d.off(l - 1), n);
            dlascl(SclType::G, anorm, ssfmin, lend - l, 1, e.off(l - 1), n);
        }
        if dd(lend).abs() < dd(l).abs() {
            lend = lsv;
            l = lendsv;
        }
        if lend > l {
            'l40: loop {
                let mut m = lend;
                if l != lend {
                    for mm in l..lend {
                        let t = ee(mm).abs();
                        if t * t <= (eps2 * dd(mm).abs()) * dd(mm + 1).abs() + safmin {
                            m = mm;
                            break;
                        }
                    }
                }
                if m < lend {
                    e.set(m - 1, 0.0);
                }
                let mut p = dd(l);
                if m == l {
                    d.set(l - 1, p);
                    l += 1;
                    if l <= lend {
                        continue 'l40;
                    }
                    break 'l40;
                }
                if m == l + 1 {
                    let (rt1, rt2, c, s) = dlaev2(dd(l), ee(l), dd(l + 1));
                    work.set(l - 1, c);
                    work.set(n - 1 + l - 1, s);
                    zlasr_rv(false, n, 2, work.off(l - 1), work.off(n - 1 + l - 1), z.off((l - 1) * ldz), ldz);
                    d.set(l - 1, rt1);
                    d.set(l, rt2);
                    e.set(l - 1, 0.0);
                    l += 2;
                    if l <= lend {
                        continue 'l40;
                    }
                    break 'l40;
                }
                if jtot == nmaxit {
                    break 'l40;
                }
                jtot += 1;
                let mut g = (dd(l + 1) - p) / (2.0 * ee(l));
                let mut r = dlapy2(g, 1.0);
                g = dd(m) - p + (ee(l) / (g + sign(r, g)));
                let (mut s, mut c) = (1.0, 1.0);
                p = 0.0;
                for i in (l..m).rev() {
                    let f = s * ee(i);
                    let b = c * ee(i);
                    (c, s, r) = dlartg(g, f);
                    if i != m - 1 {
                        e.set(i, r);
                    }
                    g = dd(i + 1) - p;
                    r = (dd(i) - g) * s + 2.0 * c * b;
                    p = s * r;
                    d.set(i, g + p);
                    g = c * r - b;
                    work.set(i - 1, c);
                    work.set(n - 1 + i - 1, -s);
                }
                zlasr_rv(false, n, m - l + 1, work.off(l - 1), work.off(n - 1 + l - 1), z.off((l - 1) * ldz), ldz);
                d.set(l - 1, dd(l) - p);
                e.set(l - 1, g);
            }
        } else {
            'l90: loop {
                let mut m = lend;
                if l != lend {
                    for mm in ((lend + 1)..=l).rev() {
                        let t = ee(mm - 1).abs();
                        if t * t <= (eps2 * dd(mm).abs()) * dd(mm - 1).abs() + safmin {
                            m = mm;
                            break;
                        }
                    }
                }
                if m > lend {
                    e.set(m - 2, 0.0);
                }
                let mut p = dd(l);
                if m == l {
                    d.set(l - 1, p);
                    if l == lend {
                        break 'l90;
                    }
                    l -= 1;
                    if l >= lend {
                        continue 'l90;
                    }
                    break 'l90;
                }
                if m + 1 == l {
                    let (rt1, rt2, c, s) = dlaev2(dd(l - 1), ee(l - 1), dd(l));
                    work.set(m - 1, c);
                    work.set(n - 1 + m - 1, s);
                    zlasr_rv(true, n, 2, work.off(m - 1), work.off(n - 1 + m - 1), z.off((l - 2) * ldz), ldz);
                    d.set(l - 2, rt1);
                    d.set(l - 1, rt2);
                    e.set(l - 2, 0.0);
                    if l < lend + 2 {
                        break 'l90;
                    }
                    l -= 2;
                    continue 'l90;
                }
                if jtot == nmaxit {
                    break 'l90;
                }
                jtot += 1;
                let mut g = (dd(l - 1) - p) / (2.0 * ee(l - 1));
                let mut r = dlapy2(g, 1.0);
                g = dd(m) - p + (ee(l - 1) / (g + sign(r, g)));
                let (mut s, mut c) = (1.0, 1.0);
                p = 0.0;
                let lm1 = l - 1;
                for i in m..=lm1 {
                    let f = s * ee(i);
                    let b = c * ee(i);
                    (c, s, r) = dlartg(g, f);
                    if i != m {
                        e.set(i - 2, r);
                    }
                    g = dd(i) - p;
                    r = (dd(i + 1) - g) * s + 2.0 * c * b;
                    p = s * r;
                    d.set(i - 1, g + p);
                    g = c * r - b;
                    work.set(i - 1, c);
                    work.set(n - 1 + i - 1, s);
                }
                zlasr_rv(true, n, l - m + 1, work.off(m - 1), work.off(n - 1 + m - 1), z.off((m - 1) * ldz), ldz);
                d.set(l - 1, dd(l) - p);
                e.set(lm1 - 1, g);
            }
        }
        if iscale == 1 {
            dlascl(SclType::G, ssfmax, anorm, lendsv - lsv + 1, 1, d.off(lsv - 1), n);
            dlascl(SclType::G, ssfmax, anorm, lendsv - lsv, 1, e.off(lsv - 1), n);
        } else if iscale == 2 {
            dlascl(SclType::G, ssfmin, anorm, lendsv - lsv + 1, 1, d.off(lsv - 1), n);
            dlascl(SclType::G, ssfmin, anorm, lendsv - lsv, 1, e.off(lsv - 1), n);
        }
        if jtot == nmaxit {
            for i in 1..n {
                if ee(i) != 0.0 {
                    info += 1;
                }
            }
            return info;
        }
    }
    let _ = crate::aux::dlasrt_inc;
    for ii in 2..=n {
        let i = ii - 1;
        let mut k = i;
        let mut p = dd(i);
        for j in ii..=n {
            if dd(j) < p {
                k = j;
                p = dd(j);
            }
        }
        if k != i {
            d.set(k - 1, dd(i));
            d.set(i - 1, p);
            zswap(n, z.off(ix(1, i, ldz)), 1, z.off(ix(1, k, ldz)), 1);
        }
    }
    let _ = dlae2;
    info
}

/// `ZLANHE( 'M', UPLO, ... )`.
pub(crate) unsafe fn zlanhe_m(upper: bool, n: usize, a: ZV, lda: usize) -> f64 {
    let mut value = 0.0f64;
    let mut upd = |sum: f64| {
        if value < sum || sum.is_nan() {
            value = sum;
        }
    };
    for j in 1..=n {
        if upper {
            for i in 1..j {
                upd(a.at(ix(i, j, lda)).abs());
            }
            upd(a.at(ix(j, j, lda)).re.abs());
        } else {
            upd(a.at(ix(j, j, lda)).re.abs());
            for i in (j + 1)..=n {
                upd(a.at(ix(i, j, lda)).abs());
            }
        }
    }
    value
}

/// `ZLASCL( 'L'/'U', 0, 0, CFROM, CTO, N, N, A, LDA )`: `A(I,J)*MUL`.
pub(crate) unsafe fn zlascl_tri(lower: bool, cfrom: f64, cto: f64, n: usize, a: ZV, lda: usize) {
    if n == 0 {
        return;
    }
    let smlnum = SAFMIN;
    let bignum = 1.0 / smlnum;
    let (mut cfromc, mut ctoc) = (cfrom, cto);
    loop {
        let cfrom1 = cfromc * smlnum;
        let (mul, done);
        if cfrom1 == cfromc {
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
                if mul == 1.0 {
                    return;
                }
            }
        }
        let zm = Z::r(mul);
        for j in 1..=n {
            let (i0, i1) = if lower { (j, n) } else { (1, j) };
            for i in i0..=i1 {
                let p = ix(i, j, lda);
                a.set(p, a.at(p) * zm);
            }
        }
        if done {
            return;
        }
    }
}

/// `(LWMIN, LRWMIN, LIWMIN)` of `ZHEEVD`.
pub fn heevd_min_work_len(n: usize, job: Job) -> (usize, usize, usize) {
    if n <= 1 {
        (1, 1, 1)
    } else {
        match job {
            Job::Vectors => (2 * n + n * n, 1 + 5 * n + 2 * n * n, 3 + 5 * n),
            Job::ValuesOnly => (n + 1, n, 1),
        }
    }
}

/// LAPACK's workspace query of `ZHEEVD`: `(LOPT, LROPT, LIOPT)` with
/// `LOPT = MAX(LWMIN, N + N*NB)`.
pub fn heevd_work_len(n: usize, job: Job) -> (usize, usize, usize) {
    let (lw, lrw, liw) = heevd_min_work_len(n, job);
    if n <= 1 { (lw, lrw, liw) } else { (lw.max(n + n * SYTRD_NB), lrw, liw) }
}

/// Workspace at which every stage of `ZHEEVD` runs blocked: `(LWORK, LRWORK,
/// LIWORK)`.
///
/// LAPACK's query value leaves `ZUNMTR` only `N` entries (`LLWRK2 = LWORK -
/// N - N^2`), which forces the unblocked `ZUNM2R`/`ZUNM2L` back-transform.
/// This size adds `N*NB + TSIZE` so the blocked `ZLARFB` path runs. Results
/// stay bit-identical to the reference called with the same `LWORK`.
pub fn heevd_opt_work_len(n: usize, job: Job) -> (usize, usize, usize) {
    let (lw, lrw, liw) = heevd_work_len(n, job);
    if n <= 1 || job == Job::ValuesOnly {
        (lw, lrw, liw)
    } else {
        (n + n * n + n * SYTRD_NB + crate::sytrd::TSIZE, lrw, liw)
    }
}

/// `ZHEEVD( JOBZ, UPLO, N, A, N, W, WORK, LWORK, RWORK, LRWORK, IWORK, LIWORK )`
/// on planar storage.
///
/// `a_re`/`a_im` are the real and imaginary parts of the column-major `n x n`
/// matrix. `work` holds the complex `WORK` planar (`LWORK = work.len()/2`:
/// real parts first, then imaginary parts), `rwork.len()` is `LRWORK` and
/// `iwork.len()` is `LIWORK`. Allocates nothing.
pub fn zheevd(
    job: Job,
    uplo: Uplo,
    n: usize,
    a_re: &mut [f64],
    a_im: &mut [f64],
    w: &mut [f64],
    work: &mut [f64],
    rwork: &mut [f64],
    iwork: &mut [usize],
) -> Result<(), LapackError> {
    let wantz = job == Job::Vectors;
    let lower = uplo == Uplo::Lower;
    if a_re.len() < n * n || a_im.len() < n * n {
        return Err(LapackError::InvalidArgument {
            arg: "a_re/a_im",
            reason: format!("need n*n = {} entries each", n * n),
        });
    }
    if w.len() < n {
        return Err(LapackError::InvalidArgument { arg: "w", reason: format!("need n = {n} entries") });
    }
    let (lwmin, lrwmin, liwmin) = heevd_min_work_len(n, job);
    let lwork = work.len() / 2;
    if lwork < lwmin {
        return Err(LapackError::WorkspaceTooSmall { need: 2 * lwmin, got: work.len() });
    }
    if rwork.len() < lrwmin {
        return Err(LapackError::WorkspaceTooSmall { need: lrwmin, got: rwork.len() });
    }
    if iwork.len() < liwmin {
        return Err(LapackError::WorkspaceTooSmall { need: liwmin, got: iwork.len() });
    }
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        w[0] = a_re[0];
        if wantz {
            a_re[0] = 1.0;
            a_im[0] = 0.0;
        }
        return Ok(());
    }
    let (wre, wim) = work.split_at_mut(lwork);
    let lrwork = rwork.len();
    // SAFETY: lengths checked above; all slices are distinct.
    let info = unsafe {
        zheevd_raw::<RefBlas>(
            wantz,
            lower,
            n,
            ZV::new(V::new(a_re), V::new(a_im)),
            V::new(w),
            ZV::new(V::new(wre), V::new(wim)),
            lwork,
            V::new(rwork),
            lrwork,
            V::new(iwork),
        )
    };
    if info != 0 {
        return Err(LapackError::NoConvergence { routine: if wantz { "zstedc" } else { "dsterf" }, info });
    }
    Ok(())
}

pub(crate) unsafe fn zheevd_raw<B: Blas>(
    wantz: bool,
    lower: bool,
    n: usize,
    a: ZV,
    w: D,
    work: ZV,
    lwork: usize,
    rwork: D,
    lrwork: usize,
    iwork: I,
) -> usize {
    let lda = n;
    let upper = !lower;
    let safmin = SAFMIN;
    let eps = PREC;
    let smlnum = safmin / eps;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = bignum.sqrt();
    let anrm = zlanhe_m(upper, n, a, lda);
    let mut iscale = false;
    let mut sigma = 0.0;
    if anrm > 0.0 && anrm < rmin {
        iscale = true;
        sigma = rmin / anrm;
    } else if anrm > rmax {
        iscale = true;
        sigma = rmax / anrm;
    }
    if iscale {
        zlascl_tri(lower, 1.0, sigma, n, a, lda);
    }
    let inde = 0usize;
    let indtau = 0usize;
    let indwrk = indtau + n;
    let indrwk = inde + n;
    let indwk2 = indwrk + n * n;
    let llwork = lwork - indwrk;
    let llrwk = lrwork - indrwk;

    zhetrd_with(
        &mut HostZTrd::<B>::new(),
        upper,
        n,
        a,
        lda,
        w,
        rwork.off(inde),
        work.off(indtau),
        work.off(indwrk),
        llwork,
    );
    let info;
    if !wantz {
        info = dsterf_raw(n, w, rwork.off(inde));
    } else {
        let llwrk2 = lwork - indwk2;
        info = zstedc_i::<B>(n, w, rwork.off(inde), work.off(indwrk), n, rwork.off(indrwk), llrwk, iwork);
        zunmtr_ln_with(
            &mut HostZOrm::<B>::new(),
            upper,
            n,
            n,
            a,
            lda,
            work.off(indtau),
            work.off(indwrk),
            n,
            work.off(indwk2),
            llwrk2,
        );
        for j in 1..=n {
            for i in 1..=n {
                a.set(ix(i, j, lda), work.off(indwrk).at(ix(i, j, n)));
            }
        }
    }
    if iscale {
        let imax = if info == 0 { n } else { (info - 1).min(n) };
        B::dscal(imax, 1.0 / sigma, w, 1);
    }
    info
}

/// `ZSTEDC( 'I', N, D, E, Z, LDZ, WORK, LWORK, RWORK, LRWORK, IWORK, LIWORK )`.
unsafe fn zstedc_i<B: Blas>(n: usize, d: D, e: D, z: ZV, ldz: usize, rwork: D, lrwork: usize, iwork: I) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        z.set(0, CONE);
        return 0;
    }
    if n <= SMLSIZ {
        return zsteqr_i(n, d, e, z, ldz, rwork);
    }
    dlaset_full(n, n, 0.0, 1.0, rwork, n);
    let ll = n * n;
    let _ = (lrwork, dlanst_m);
    let info = dstedc_i_with(&mut HostDc::<B>::new(), n, d, e, rwork, n, rwork.off(ll), iwork);
    for j in 1..=n {
        for i in 1..=n {
            z.set(ix(i, j, ldz), Z::r(rwork.at((j - 1) * n + i - 1)));
        }
    }
    info
}

/// `ZHEEVD` on the default CubeCL runtime (see [`crate::runtime`]).
///
/// Bit-identical to [`zheevd`] called with [`heevd_opt_work_len`], and so to
/// Reference-LAPACK 3.12.1 called with the same lengths. The workspace is
/// cached per thread (see [`crate::dsyevd_alloc`]).
pub fn zheevd_alloc(
    job: Job,
    uplo: Uplo,
    n: usize,
    a_re: &mut [f64],
    a_im: &mut [f64],
    w: &mut [f64],
) -> Result<(), LapackError> {
    crate::runtime::with_workspace(n, |client, ws| {
        crate::device::zheevd_device(client, job, uplo, n, a_re, a_im, w, ws)
    })
}

/// [`zheevd`] with its workspace allocated once, at [`heevd_opt_work_len`]:
/// the sequential reference transcription, with no device runtime involved.
pub fn zheevd_alloc_host(
    job: Job,
    uplo: Uplo,
    n: usize,
    a_re: &mut [f64],
    a_im: &mut [f64],
    w: &mut [f64],
) -> Result<(), LapackError> {
    let (lw, lrw, liw) = heevd_opt_work_len(n, job);
    let mut work = vec![0.0f64; 2 * lw];
    let mut rwork = vec![0.0f64; lrw];
    let mut iwork = vec![0usize; liw];
    zheevd(job, uplo, n, a_re, a_im, w, &mut work, &mut rwork, &mut iwork)
}
