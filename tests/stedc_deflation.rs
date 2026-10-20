//! Adversarial deflation fixtures with branch counters (`--features stats`):
//! every deflation branch of `dlaed2` and the K = 1 / K = 2 shortcuts of
//! `dlaed3` must run somewhere in the fixture set, and every fixture keeps
//! ‖T − ZΛZᵀ‖ and ‖ZᵀZ − I‖ small.

#![cfg(feature = "stats")]

mod common;
use common::*;
use lapack_rs::stats;

#[test]
fn deflation_branches_run_and_identities_hold() {
    stats::reset();
    let mut seen = Vec::new();
    for &n in &[26usize, 64, 100, 257] {
        for (name, d0, e0) in tridiag_fixtures(n) {
            let before = stats::snapshot();
            let (lw, liw) = lapack_rs::stedc_work_len(n);
            let (mut d, mut e) = (d0.clone(), e0.clone());
            e.push(0.0);
            let mut z = vec![0.0; n * n];
            let (mut work, mut iwork) = (vec![0.0; lw], vec![0usize; liw]);
            lapack_rs::blocks::dstedc(n, &mut d, &mut e, &mut z, &mut work, &mut iwork).expect("dstedc");
            let mut t = vec![0.0; n * n];
            for i in 0..n {
                t[i + i * n] = d0[i];
                if i + 1 < n {
                    t[i + 1 + i * n] = e0[i];
                    t[i + (i + 1) * n] = e0[i];
                }
            }
            let (rec, orth) = sym_residuals(n, &t, &z, &d);
            let scale = d0.iter().chain(&e0).fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
            assert!(
                rec <= 1e-12 * scale * n as f64 && orth <= 1e-12 * n as f64,
                "{name} n={n}: rec={rec:e} orth={orth:e}"
            );
            let after = stats::snapshot();
            seen.push((
                name,
                n,
                after.merges - before.merges,
                after.small_z_deflation - before.small_z_deflation,
                after.rotation_deflation - before.rotation_deflation,
            ));
        }
    }
    let s = stats::snapshot();
    eprintln!("{s:?}");
    for (name, n, merges, small_z, rot) in &seen {
        eprintln!("{name:>18} n={n:<4} merges={merges:<4} small_z={small_z:<6} rotation={rot}");
    }
    assert!(s.merges > 0, "no merge ran");
    assert!(s.splits > 0, "no D&C split ran");
    assert!(s.small_z_deflation > 0, "small-z deflation never ran");
    assert!(s.rotation_deflation > 0, "Givens-rotation deflation never ran");
    assert!(s.secular_roots > 0, "secular solver never ran");
    assert!(s.full_deflation > 0, "the rho*|z| <= tol full-deflation shortcut never ran");
}
