//! `DSYEVD`: eigenvalues and optionally eigenvectors of a real symmetric
//! matrix. Transcribed from Reference-LAPACK v3.12.1 `SRC/dsyevd.f`.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::aux::{PREC, SAFMIN, SclType, dlacpy_all, dlansy_m, dlascl};
use crate::flavor::{Blas, RefBlas};
use crate::ptr::{I, V};
use crate::stedc::{HostDc, SMLSIZ, dstedc_i_with};
use crate::steqr::dsterf_raw;
use crate::sytrd::{HostOrm, HostTrd, SYTRD_NB, dormtr_ln_with, dsytrd_with};
use crate::{Job, LapackError, Uplo};

/// `(LWMIN, LIWMIN)` of `DSYEVD`: the smallest accepted `work`/`iwork`.
pub fn syevd_min_work_len(n: usize, job: Job) -> (usize, usize) {
    if n <= 1 {
        (1, 1)
    } else {
        match job {
            Job::Vectors => (1 + 6 * n + 2 * n * n, 3 + 5 * n),
            Job::ValuesOnly => (2 * n + 1, 1),
        }
    }
}

/// LAPACK's workspace-query result of `DSYEVD` (`WORK(1)`, `IWORK(1)` after
/// `LWORK = -1`): `LOPT = MAX(LWMIN, 2N + N*NB)` with `NB = 32`.
pub fn syevd_work_len(n: usize, job: Job) -> (usize, usize) {
    let (lwmin, liwmin) = syevd_min_work_len(n, job);
    if n <= 1 { (lwmin, liwmin) } else { (lwmin.max(2 * n + n * SYTRD_NB), liwmin) }
}

/// Workspace at which every stage of `DSYEVD` runs blocked: `(LWORK, LIWORK)`.
///
/// For small `n` LAPACK's query value leaves `DORMTR` less than
/// `N*NB + TSIZE`, which selects the unblocked `DORM2R`/`DORM2L`. This size
/// guarantees the blocked path; results stay bit-identical to the reference
/// called with the same `LWORK`.
pub fn syevd_opt_work_len(n: usize, job: Job) -> (usize, usize) {
    let (lw, liw) = syevd_work_len(n, job);
    if n <= 1 || job == Job::ValuesOnly {
        (lw, liw)
    } else {
        (lw.max(1 + 2 * n + n * n + n * SYTRD_NB + crate::sytrd::TSIZE), liw)
    }
}

/// `DSYEVD( JOBZ, UPLO, N, A, N, W, WORK, LWORK, IWORK, LIWORK, INFO )`.
///
/// `a` is column-major `n x n` (`LDA = n`); on return it holds the
/// orthonormal eigenvectors (`Job::Vectors`) or is destroyed. `w` receives
/// the eigenvalues in ascending order. `work.len()` is `LWORK` and
/// `iwork.len()` is `LIWORK`; the block sizes the reference derives from them
/// are reproduced, so results are bit-identical to the reference called with
/// the same lengths. Allocates nothing.
pub fn dsyevd(
    job: Job,
    uplo: Uplo,
    n: usize,
    a: &mut [f64],
    w: &mut [f64],
    work: &mut [f64],
    iwork: &mut [usize],
) -> Result<(), LapackError> {
    let wantz = matches!(job, Job::Vectors);
    let lower = matches!(uplo, Uplo::Lower);
    if a.len() < n * n {
        return Err(LapackError::InvalidArgument {
            arg: "a",
            reason: format!("need n*n = {} entries, got {}", n * n, a.len()),
        });
    }
    if w.len() < n {
        return Err(LapackError::InvalidArgument {
            arg: "w",
            reason: format!("need n = {n} entries, got {}", w.len()),
        });
    }
    let (lwmin, liwmin) = syevd_min_work_len(n, job);
    if work.len() < lwmin {
        return Err(LapackError::WorkspaceTooSmall { need: lwmin, got: work.len() });
    }
    if iwork.len() < liwmin {
        return Err(LapackError::WorkspaceTooSmall { need: liwmin, got: iwork.len() });
    }
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        w[0] = a[0];
        if wantz {
            a[0] = 1.0;
        }
        return Ok(());
    }
    let lwork = work.len();
    let liwork = iwork.len();
    // SAFETY: every length used below was checked above: a >= n*n, w >= n,
    // work >= LWMIN (which covers each sub-range the reference addresses),
    // iwork >= LIWMIN. The slices are distinct.
    let info = unsafe {
        dsyevd_raw::<RefBlas>(wantz, lower, n, V::new(a), V::new(w), V::new(work), lwork, V::new(iwork), liwork)
    };
    if info != 0 {
        return Err(LapackError::NoConvergence { routine: if wantz { "dstedc" } else { "dsterf" }, info });
    }
    Ok(())
}

/// `DSYEVD` body on raw views, BLAS flavour `B`. Returns `INFO`.
pub(crate) unsafe fn dsyevd_raw<B: Blas>(
    wantz: bool,
    lower: bool,
    n: usize,
    a: V<f64>,
    w: V<f64>,
    work: V<f64>,
    lwork: usize,
    iwork: I,
    liwork: usize,
) -> usize {
    let lda = n;
    let upper = !lower;
    let safmin = SAFMIN;
    let eps = PREC;
    let smlnum = safmin / eps;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = bignum.sqrt();

    let anrm = dlansy_m(upper, n, a, lda);
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
        dlascl(if lower { SclType::L } else { SclType::U }, 1.0, sigma, n, n, a, lda);
    }

    let inde = 0usize;
    let indtau = inde + n;
    let indwrk = indtau + n;
    let llwork = lwork - indwrk;
    let indwk2 = indwrk + n * n;

    dsytrd_with(
        &mut HostTrd::<B>::new(),
        upper,
        n,
        a,
        lda,
        w,
        work.off(inde),
        work.off(indtau),
        work.off(indwrk),
        llwork,
    );
    let info;
    if !wantz {
        info = dsterf_raw(n, w, work.off(inde));
    } else {
        let llwrk2 = lwork - indwk2;
        let _ = (liwork, SMLSIZ);
        info =
            dstedc_i_with(&mut HostDc::<B>::new(), n, w, work.off(inde), work.off(indwrk), n, work.off(indwk2), iwork);
        dormtr_ln_with(
            &mut HostOrm::<B>::new(),
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
        dlacpy_all(n, n, work.off(indwrk), n, a, lda);
    }
    if iscale {
        B::dscal(n, 1.0 / sigma, w, 1);
    }
    info
}

/// `DSYEVD` on the default CubeCL runtime (see [`crate::runtime`]).
///
/// Bit-identical to [`dsyevd`] called with [`syevd_opt_work_len`], and so to
/// Reference-LAPACK 3.12.1 called with the same `LWORK`. The workspace — host
/// and device — is cached per thread and grown on demand, so a loop of solves
/// pays for it once; [`crate::runtime::release_workspace`] gives it back.
pub fn dsyevd_alloc(job: Job, uplo: Uplo, n: usize, a: &mut [f64], w: &mut [f64]) -> Result<(), LapackError> {
    crate::runtime::with_workspace(n, |client, ws| crate::device::dsyevd_device(client, job, uplo, n, a, w, ws))
}

/// [`dsyevd`] with its workspace allocated once, at [`syevd_opt_work_len`]:
/// the sequential reference transcription, with no device runtime involved.
/// Used by the oracle gates as the value [`dsyevd_alloc`] must reproduce.
pub fn dsyevd_alloc_host(job: Job, uplo: Uplo, n: usize, a: &mut [f64], w: &mut [f64]) -> Result<(), LapackError> {
    let (lwork, liwork) = syevd_opt_work_len(n, job);
    let mut work = vec![0.0f64; lwork];
    let mut iwork = vec![0usize; liwork];
    dsyevd(job, uplo, n, a, w, &mut work, &mut iwork)
}
