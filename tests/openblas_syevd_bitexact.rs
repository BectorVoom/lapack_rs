//! **OpenBLAS track.** Whole `DSYEVD` against scipy's OpenBLAS 0.3.30
//! (`scipy_dsyevd_` with its own workspace query), bit for bit: `w`, all n²
//! entries of Z and INFO, both `UPLO`, both `JOBZ`. Never compared with the
//! Reference-LAPACK oracle.

mod common;
use common::*;
use lapack_rs::{Job, Uplo};
use openblas_ref as ob;

fn ob_dsyevd(jobz: char, uplo: char, n: usize, a: &mut [f64], w: &mut [f64]) -> i32 {
    let (jz, ul) = (ob::ch(jobz), ob::ch(uplo));
    let (n_, lda) = (n as i32, n.max(1) as i32);
    let mut info = 0;
    let (mut wq, mut iwq, m1, mut dw) = (0.0f64, 0i32, -1i32, 0.0f64);
    // SAFETY: workspace query.
    unsafe { ob::dsyevd(&jz, &ul, &n_, a.as_mut_ptr(), &lda, &mut dw, &mut wq, &m1, &mut iwq, &m1, &mut info, 1, 1) };
    let (lw, liw) = ((wq as i32).max(1), iwq.max(1));
    let mut work = vec![0.0; lw as usize];
    let mut iwork = vec![0i32; liw as usize];
    // SAFETY: lengths from the query.
    unsafe {
        ob::dsyevd(
            &jz,
            &ul,
            &n_,
            a.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lw,
            iwork.as_mut_ptr(),
            &liw,
            &mut info,
            1,
            1,
        )
    };
    info
}

fn run(tag: &str, n: usize, a0: &[f64]) {
    for job in [Job::Vectors, Job::ValuesOnly] {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            let mut a = a0.to_vec();
            let mut w = vec![0.0; n];
            let res = lapack_rs::openblas::dsyevd_host(job, uplo, n, &mut a, &mut w);
            let mut ar = a0.to_vec();
            let mut wr = vec![0.0; n];
            let info = ob_dsyevd(
                if job == Job::Vectors { 'V' } else { 'N' },
                if uplo == Uplo::Upper { 'U' } else { 'L' },
                n,
                &mut ar,
                &mut wr,
            );
            let tag = format!("{tag} n={n} {job:?} {uplo:?}");
            assert_eq!(res.is_ok(), info == 0, "{tag}: INFO ours={res:?} openblas={info}");
            assert_bits(&format!("{tag} w"), &w, &wr);
            if job == Job::Vectors {
                assert_bits(&format!("{tag} Z"), &a, &ar);
            }
        }
    }
}

#[test]
fn openblas_syevd_random_grid() {
    if !ob::require("openblas_syevd_random_grid") {
        return;
    }
    let max_n: usize = std::env::var("LAPACK_RS_MAX_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    for &n in N_GRID.iter().filter(|&&n| n <= max_n) {
        run("random", n, &random_sym(n, 1000 + n as u64));
    }
}

#[test]
fn openblas_syevd_fixtures() {
    if !ob::require("openblas_syevd_fixtures") {
        return;
    }
    for &n in &[4usize, 33, 64, 129] {
        for (name, a) in dense_fixtures(n) {
            run(name, n, &a);
        }
    }
}
