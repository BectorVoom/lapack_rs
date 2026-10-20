//! **OpenBLAS track.** Whole `ZHEEVD` against scipy's OpenBLAS 0.3.30
//! (`scipy_zheevd_` with its own workspace query), bit for bit: `w`, all n²
//! entries of Z (both parts) and INFO, both `UPLO`, both `JOBZ`. Never
//! compared with the Reference-LAPACK oracle.

mod common;
use common::*;
use lapack_rs::{Job, Uplo};
use openblas_ref::{self as ob, C64};

fn ob_zheevd(jobz: char, uplo: char, n: usize, a: &mut [C64], w: &mut [f64]) -> i32 {
    let (jz, ul) = (ob::ch(jobz), ob::ch(uplo));
    let (n_, lda) = (n as i32, n.max(1) as i32);
    let mut info = 0;
    let (mut wq, mut rwq, mut iwq, m1, mut dw) = (C64::default(), 0.0f64, 0i32, -1i32, 0.0f64);
    // SAFETY: workspace query.
    unsafe {
        ob::zheevd(
            &jz,
            &ul,
            &n_,
            a.as_mut_ptr(),
            &lda,
            &mut dw,
            &mut wq,
            &m1,
            &mut rwq,
            &m1,
            &mut iwq,
            &m1,
            &mut info,
            1,
            1,
        )
    };
    let (lw, lrw, liw) = ((wq.re as i32).max(1), (rwq as i32).max(1), iwq.max(1));
    let mut work = vec![C64::default(); lw as usize];
    let mut rwork = vec![0.0; lrw as usize];
    let mut iwork = vec![0i32; liw as usize];
    // SAFETY: lengths from the query.
    unsafe {
        ob::zheevd(
            &jz,
            &ul,
            &n_,
            a.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lw,
            rwork.as_mut_ptr(),
            &lrw,
            iwork.as_mut_ptr(),
            &liw,
            &mut info,
            1,
            1,
        )
    };
    info
}

fn run(tag: &str, n: usize, re: &[f64], im: &[f64]) {
    for job in [Job::Vectors, Job::ValuesOnly] {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            let (mut ar, mut ai, mut w) = (re.to_vec(), im.to_vec(), vec![0.0; n]);
            let res = lapack_rs::openblas::zheevd_host(job, uplo, n, &mut ar, &mut ai, &mut w);
            let mut z: Vec<C64> = re.iter().zip(im).map(|(&re, &im)| C64 { re, im }).collect();
            let mut wr = vec![0.0; n];
            let info = ob_zheevd(
                if job == Job::Vectors { 'V' } else { 'N' },
                if uplo == Uplo::Upper { 'U' } else { 'L' },
                n,
                &mut z,
                &mut wr,
            );
            let tag = format!("{tag} n={n} {job:?} {uplo:?}");
            assert_eq!(res.is_ok(), info == 0, "{tag}: INFO ours={res:?} openblas={info}");
            assert_bits(&format!("{tag} w"), &w, &wr);
            if job == Job::Vectors {
                let zr: Vec<f64> = z.iter().map(|c| c.re).collect();
                let zi: Vec<f64> = z.iter().map(|c| c.im).collect();
                assert_bits(&format!("{tag} Z re"), &ar, &zr);
                assert_bits(&format!("{tag} Z im"), &ai, &zi);
            }
        }
    }
}

#[test]
fn openblas_heevd_random_grid() {
    if !ob::require("openblas_heevd_random_grid") {
        return;
    }
    let max_n: usize = std::env::var("LAPACK_RS_MAX_N").ok().and_then(|s| s.parse().ok()).unwrap_or(513);
    for &n in N_GRID.iter().filter(|&&n| n <= max_n) {
        let (re, im) = random_herm(n, 2000 + n as u64);
        run("random", n, &re, &im);
    }
}
