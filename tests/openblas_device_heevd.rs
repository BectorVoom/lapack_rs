//! **OpenBLAS track.** `openblas::zheevd` (hybrid driver, OpenBLAS-order
//! device kernels on the default CubeCL runtime) against the sequential host
//! transcription `openblas::zheevd_host`, bit for bit (the host path is gated
//! against scipy's OpenBLAS by `tests/openblas_heevd_bitexact.rs`).

mod common;
use common::*;
use lapack_rs::{Job, Uplo, openblas};

#[test]
fn openblas_device_heevd_matches_host() {
    let max_n: usize = std::env::var("LAPACK_RS_MAX_N").ok().and_then(|s| s.parse().ok()).unwrap_or(257);
    for &n in N_GRID.iter().filter(|&&n| n <= max_n) {
        let (re, im) = random_herm(n, 3000 + n as u64);
        for job in [Job::Vectors, Job::ValuesOnly] {
            for uplo in [Uplo::Lower, Uplo::Upper] {
                let (mut ar, mut ai, mut w) = (re.clone(), im.clone(), vec![0.0; n]);
                let r1 = openblas::zheevd(job, uplo, n, &mut ar, &mut ai, &mut w);
                let (mut hr, mut hi, mut wh) = (re.clone(), im.clone(), vec![0.0; n]);
                let r2 = openblas::zheevd_host(job, uplo, n, &mut hr, &mut hi, &mut wh);
                let tag = format!("n={n} {job:?} {uplo:?}");
                assert_eq!(r1.is_ok(), r2.is_ok(), "{tag}: {r1:?} vs {r2:?}");
                assert_bits(&format!("{tag} w"), &w, &wh);
                if job == Job::Vectors {
                    assert_bits(&format!("{tag} Z re"), &ar, &hr);
                    assert_bits(&format!("{tag} Z im"), &ai, &hi);
                }
            }
        }
    }
}
