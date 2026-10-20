//! `dsyevd`/`zheevd` identities (no external oracle): ‖A − UΛUᴴ‖∞ ≤ 1e-9,
//! ‖UᴴU − I‖∞ ≤ 1e-9 (1e-10 Hermitian orthogonality), eigenvalues ascending,
//! values-only equal to the vectors run, and faer eigenvalues within 1e-10
//! relative as an independent cross-check.

mod common;
use common::*;
use faer::{Mat, Side};
use lapack_rs::{Job, Uplo, dsyevd_alloc, zheevd_alloc};

#[test]
fn real_identities_and_faer() {
    for &n in &[1usize, 2, 17, 26, 64, 129, 200] {
        for (name, a0) in dense_fixtures(n) {
            if name.starts_with("scaled") {
                continue;
            }
            for uplo in [Uplo::Upper, Uplo::Lower] {
                let (mut z, mut w) = (a0.clone(), vec![0.0; n]);
                dsyevd_alloc(Job::Vectors, uplo, n, &mut z, &mut w).expect("dsyevd");
                assert!(w.windows(2).all(|p| p[0] <= p[1]), "{name} n={n}: not ascending");
                let (rec, orth) = sym_residuals(n, &a0, &z, &w);
                assert!(rec <= 1e-9 && orth <= 1e-9, "{name} n={n} {uplo:?}: rec={rec:e} orth={orth:e}");
                let (mut a2, mut w2) = (a0.clone(), vec![0.0; n]);
                dsyevd_alloc(Job::ValuesOnly, uplo, n, &mut a2, &mut w2).expect("dsyevd N");
                for (a, b) in w.iter().zip(&w2) {
                    assert!((a - b).abs() <= 1e-10 * a.abs().max(1.0), "{name} n={n}: values-only {a} vs {b}");
                }
                let fa = Mat::<f64>::from_fn(n, n, |i, j| a0[i + j * n]);
                let fw = fa.self_adjoint_eigenvalues(Side::Lower).expect("faer");
                for (a, b) in w.iter().zip(&fw) {
                    assert!((a - b).abs() <= 1e-10 * a.abs().max(1.0), "{name} n={n}: faer {b} vs {a}");
                }
            }
        }
    }
}

#[test]
fn complex_identities_and_faer() {
    use faer::c64;
    for &n in &[1usize, 2, 17, 26, 64, 129] {
        let (re0, im0) = random_herm(n, 55 + n as u64);
        for uplo in [Uplo::Upper, Uplo::Lower] {
            let (mut zr, mut zi, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
            zheevd_alloc(Job::Vectors, uplo, n, &mut zr, &mut zi, &mut w).expect("zheevd");
            let (rec, orth) = herm_residuals(n, &re0, &im0, &zr, &zi, &w);
            assert!(rec <= 1e-9 && orth <= 1e-10, "n={n} {uplo:?}: rec={rec:e} orth={orth:e}");
            let fa = Mat::<c64>::from_fn(n, n, |i, j| c64::new(re0[i + j * n], im0[i + j * n]));
            let fw = fa.self_adjoint_eigenvalues(Side::Lower).expect("faer");
            for (a, b) in w.iter().zip(&fw) {
                assert!((a - b).abs() <= 1e-10 * a.abs().max(1.0), "n={n}: faer {b} vs {a}");
            }
        }
    }
}
