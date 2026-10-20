//! `dsteqr`/`dsterf` against closed forms (no external oracle needed):
//! the 1-2-1 tridiagonal has eigenvalues 2 - 2 cos(k pi / (n+1)).

mod common;
use common::*;
use lapack_rs::{Compz, dsteqr, dsterf};

fn closed_form(n: usize) -> Vec<f64> {
    let mut v: Vec<f64> =
        (1..=n).map(|k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos()).collect();
    v.sort_by(f64::total_cmp);
    v
}

#[test]
fn laplacian_eigenvalues() {
    for &n in &[1usize, 2, 3, 10, 25, 64, 200] {
        let want = closed_form(n);
        let mut d = vec![2.0; n];
        let mut e = vec![-1.0; n.saturating_sub(1)];
        let mut z = vec![0.0; n * n];
        let mut work = vec![0.0; (2 * n).max(1)];
        dsteqr(Compz::I, n, &mut d, &mut e, &mut z, n.max(1), &mut work).expect("dsteqr");
        for (g, w) in d.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-12, "dsteqr n={n}: {g} vs {w}");
        }
        // Orthonormal eigenvectors and T z = lambda z.
        for k in 0..n {
            for i in 0..n {
                let tz = 2.0 * z[i + k * n]
                    - if i > 0 { z[i - 1 + k * n] } else { 0.0 }
                    - if i + 1 < n { z[i + 1 + k * n] } else { 0.0 };
                assert!((tz - d[k] * z[i + k * n]).abs() <= 1e-12, "Tz != lz at n={n}");
            }
        }
        let mut d2 = vec![2.0; n];
        let mut e2 = vec![-1.0; n.saturating_sub(1)];
        dsterf(n, &mut d2, &mut e2).expect("dsterf");
        for (g, w) in d2.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-12, "dsterf n={n}: {g} vs {w}");
        }
    }
}

#[test]
fn random_tridiagonal_identities() {
    for &n in &[5usize, 40, 150] {
        let (d0, e0) = random_tridiag(n, 99 + n as u64);
        let (mut d, mut e) = (d0.clone(), e0.clone());
        let mut z = vec![0.0; n * n];
        let mut work = vec![0.0; 2 * n];
        dsteqr(Compz::I, n, &mut d, &mut e, &mut z, n, &mut work).expect("dsteqr");
        let mut t = vec![0.0; n * n];
        for i in 0..n {
            t[i + i * n] = d0[i];
            if i + 1 < n {
                t[i + 1 + i * n] = e0[i];
                t[i + (i + 1) * n] = e0[i];
            }
        }
        let (rec, orth) = sym_residuals(n, &t, &z, &d);
        assert!(rec <= 1e-12 && orth <= 1e-12, "n={n}: rec={rec:e} orth={orth:e}");
    }
}
