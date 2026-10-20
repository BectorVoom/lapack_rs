//! Whole `zheevd` against Reference-LAPACK 3.12.1, bit for bit: `w`, both
//! planar parts of Z and INFO, for both `uplo`, both `jobz`, and workspace
//! lengths equal to the query value and to the minimum.

mod common;
use common::*;
use lapack_rs::{Job, Uplo, heevd_min_work_len, heevd_opt_work_len, heevd_work_len, zheevd};

pub fn run_case(n: usize, re0: &[f64], im0: &[f64], job: Job, uplo: Uplo, minimal: bool, tag: &str) {
    let (lw, lrw, liw) = if minimal {
        heevd_min_work_len(n, job)
    } else if tag.starts_with("opt") {
        heevd_opt_work_len(n, job)
    } else {
        heevd_work_len(n, job)
    };
    let (mut re, mut im) = (re0.to_vec(), im0.to_vec());
    let mut w = vec![0.0; n];
    let mut work = vec![0.0; 2 * lw];
    let mut rwork = vec![0.0; lrw];
    let mut iwork = vec![0usize; liw];
    let threads = if tag.ends_with("@1") { 1 } else { 8 };
    let res =
        with_threads(threads, || zheevd(job, uplo, n, &mut re, &mut im, &mut w, &mut work, &mut rwork, &mut iwork));

    let mut ar = to_c64(re0, im0);
    let mut wr = vec![0.0; n];
    let mut workr = vec![lapack_ref::C64::default(); lw];
    let mut rworkr = vec![0.0; lrw];
    let mut iworkr = vec![0i32; liw];
    let jz = if job == Job::Vectors { 'V' } else { 'N' };
    let ul = if uplo == Uplo::Upper { 'U' } else { 'L' };
    let info = lapack_ref::zheevd(jz, ul, n, &mut ar, &mut wr, &mut workr, &mut rworkr, &mut iworkr);
    let tag = format!("{tag} n={n} job={job:?} uplo={uplo:?} minimal={minimal}");
    assert_eq!(res.is_ok(), info == 0, "{tag}: INFO ours={res:?} ref={info}");
    assert_bits(&format!("{tag} w"), &w, &wr);
    if job == Job::Vectors {
        let (rr, ri) = from_c64(&ar);
        assert_bits(&format!("{tag} Z re"), &re, &rr);
        assert_bits(&format!("{tag} Z im"), &im, &ri);
    }
}

#[test]
fn heevd_random_grid() {
    if !lapack_ref::require("heevd_random_grid") {
        return;
    }
    let max_n: usize = std::env::var("LAPACK_RS_MAX_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    for &n in N_GRID.iter().filter(|&&n| n <= max_n) {
        let (re, im) = random_herm(n, 2000 + n as u64);
        for job in [Job::Vectors, Job::ValuesOnly] {
            for uplo in [Uplo::Upper, Uplo::Lower] {
                for minimal in [false, true] {
                    run_case(n, &re, &im, job, uplo, minimal, "random");
                    if !minimal {
                        run_case(n, &re, &im, job, uplo, false, "opt_random");
                    }
                }
            }
        }
    }
}

fn herm_fixtures(n: usize) -> Vec<(&'static str, Vec<f64>, Vec<f64>)> {
    let mut out = Vec::new();
    let (re, im) = random_herm(n, 7000 + n as u64);
    out.push(("random", re, im));
    // Pure-imaginary off-diagonals, real diagonal.
    let (mut re, im) = random_herm(n, 7100 + n as u64);
    for j in 0..n {
        for i in 0..n {
            if i != j {
                re[i + j * n] = 0.0;
            }
        }
    }
    out.push(("pure_imaginary_offdiag", re, im));
    // Real symmetric data in complex storage (all imaginary parts zero).
    let a = random_sym(n, 7200 + n as u64);
    out.push(("real_valued", a, vec![0.0; n * n]));
    let mut id = vec![0.0; n * n];
    for i in 0..n {
        id[i + i * n] = 1.0;
    }
    out.push(("identity", id, vec![0.0; n * n]));
    let (re, im) = random_herm(n, 7300 + n as u64);
    out.push(("scaled_tiny", re.iter().map(|v| v * 1e-160).collect(), im.iter().map(|v| v * 1e-160).collect()));
    let (re, im) = random_herm(n, 7400 + n as u64);
    out.push(("scaled_huge", re.iter().map(|v| v * 1e160).collect(), im.iter().map(|v| v * 1e160).collect()));
    out
}

#[test]
fn heevd_fixtures_and_threads() {
    if !lapack_ref::require("heevd_fixtures_and_threads") {
        return;
    }
    let max_n: usize = std::env::var("LAPACK_RS_MAX_N").ok().and_then(|s| s.parse().ok()).unwrap_or(257);
    for &n in [2usize, 25, 26, 33, 65, 129, 257].iter().filter(|&&n| n <= max_n) {
        for (name, re, im) in herm_fixtures(n) {
            for job in [Job::Vectors, Job::ValuesOnly] {
                for uplo in [Uplo::Upper, Uplo::Lower] {
                    let (mut re, mut im) = (re.clone(), im.clone());
                    poison_other_triangle(&mut re, n, uplo == Uplo::Upper);
                    poison_other_triangle(&mut im, n, uplo == Uplo::Upper);
                    run_case(n, &re, &im, job, uplo, false, &format!("{name}@8"));
                    run_case(n, &re, &im, job, uplo, true, &format!("{name}@1"));
                }
            }
        }
    }
}
