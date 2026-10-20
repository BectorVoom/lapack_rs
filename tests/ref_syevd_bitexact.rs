//! Whole `dsyevd` against Reference-LAPACK 3.12.1, bit for bit: `w`, all n^2
//! entries of Z and INFO, for both `uplo`, both `jobz`, and `LWORK` equal to
//! the query value and to the minimum.

mod common;
use common::*;
use lapack_rs::{Job, Uplo, dsyevd, syevd_min_work_len, syevd_opt_work_len, syevd_work_len};

fn run_case(n: usize, a0: &[f64], job: Job, uplo: Uplo, minimal: bool, tag: &str) {
    let (lw, liw) = if minimal {
        syevd_min_work_len(n, job)
    } else if tag.starts_with("opt") {
        syevd_opt_work_len(n, job)
    } else {
        syevd_work_len(n, job)
    };
    let mut a = a0.to_vec();
    let mut w = vec![0.0; n];
    let mut work = vec![0.0; lw];
    let mut iwork = vec![0usize; liw];
    let threads = if tag.ends_with("@1") { 1 } else { 8 };
    let res = with_threads(threads, || dsyevd(job, uplo, n, &mut a, &mut w, &mut work, &mut iwork));

    let mut ar = a0.to_vec();
    let mut wr = vec![0.0; n];
    let mut workr = vec![0.0; lw];
    let mut iworkr = vec![0i32; liw];
    let jz = if job == Job::Vectors { 'V' } else { 'N' };
    let ul = if uplo == Uplo::Upper { 'U' } else { 'L' };
    let info = lapack_ref::dsyevd(jz, ul, n, &mut ar, &mut wr, &mut workr, &mut iworkr);
    assert_eq!(res.is_ok(), info == 0, "{tag}: INFO ours={res:?} ref={info}");
    let tag = format!("{tag} n={n} job={job:?} uplo={uplo:?} minimal={minimal}");
    assert_bits(&format!("{tag} w"), &w, &wr);
    if job == Job::Vectors {
        assert_bits(&format!("{tag} Z"), &a, &ar);
    }
}

#[test]
fn syevd_random_grid() {
    if !lapack_ref::require("syevd_random_grid") {
        return;
    }
    let max_n: usize = std::env::var("LAPACK_RS_MAX_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    for &n in N_GRID.iter().filter(|&&n| n <= max_n) {
        let a = random_sym(n, 1000 + n as u64);
        for job in [Job::Vectors, Job::ValuesOnly] {
            for uplo in [Uplo::Upper, Uplo::Lower] {
                for minimal in [false, true] {
                    run_case(n, &a, job, uplo, minimal, "random");
                    if !minimal {
                        run_case(n, &a, job, uplo, false, "opt_random");
                    }
                }
            }
        }
    }
}

#[test]
fn syevd_fixtures_and_threads() {
    if !lapack_ref::require("syevd_fixtures_and_threads") {
        return;
    }
    let max_n: usize = std::env::var("LAPACK_RS_MAX_N").ok().and_then(|s| s.parse().ok()).unwrap_or(513);
    for &n in [2usize, 25, 26, 33, 65, 129, 257, 513].iter().filter(|&&n| n <= max_n) {
        for (name, a) in dense_fixtures(n) {
            for job in [Job::Vectors, Job::ValuesOnly] {
                for uplo in [Uplo::Upper, Uplo::Lower] {
                    let mut a = a.clone();
                    poison_other_triangle(&mut a, n, uplo == Uplo::Upper);
                    run_case(n, &a, job, uplo, false, &format!("{name}@8"));
                    run_case(n, &a, job, uplo, true, &format!("{name}@1"));
                }
            }
        }
    }
}
