//! **OpenBLAS track.** `openblas::dsyevd` (hybrid driver, OpenBLAS-order
//! device kernels on the default CubeCL runtime) against the sequential host
//! transcription `openblas::dsyevd_host`, bit for bit. The host path is gated
//! against scipy's OpenBLAS by `tests/openblas_syevd_bitexact.rs`, so this
//! needs no oracle library.

mod common;
use common::*;
use lapack_rs::{Job, Uplo, openblas};

fn run(tag: &str, n: usize, a0: &[f64]) {
    for job in [Job::Vectors, Job::ValuesOnly] {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            let (mut a, mut w) = (a0.to_vec(), vec![0.0; n]);
            let r1 = openblas::dsyevd(job, uplo, n, &mut a, &mut w);
            let (mut ah, mut wh) = (a0.to_vec(), vec![0.0; n]);
            let r2 = openblas::dsyevd_host(job, uplo, n, &mut ah, &mut wh);
            let tag = format!("{tag} n={n} {job:?} {uplo:?}");
            assert_eq!(r1.is_ok(), r2.is_ok(), "{tag}: {r1:?} vs {r2:?}");
            assert_bits(&format!("{tag} w"), &w, &wh);
            if job == Job::Vectors {
                assert_bits(&format!("{tag} Z"), &a, &ah);
            }
        }
    }
}

#[test]
fn openblas_device_syevd_matches_host() {
    let max_n: usize = std::env::var("LAPACK_RS_MAX_N").ok().and_then(|s| s.parse().ok()).unwrap_or(513);
    for &n in N_GRID.iter().filter(|&&n| n <= max_n) {
        run("random", n, &random_sym(n, 1000 + n as u64));
    }
    for &n in &[33usize, 64, 129] {
        for (name, a) in dense_fixtures(n) {
            run(name, n, &a);
        }
    }
}
