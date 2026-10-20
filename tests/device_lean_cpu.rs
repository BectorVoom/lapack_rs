//! Lean merge U (`Exactness::Fast` + `MergeU::Lean`) on the CPU runtime:
//! tolerance-gated against the bit-exact staged path — eigenvalues ≤ 1e-10
//! relative, eigenvector residual and
//! orthogonality ≤ 1e-9 (1e-10 Hermitian), including the deflation-heavy
//! tridiagonal fixtures; lower device peak memory than staged; `Auto` picks
//! lean only when staged does not fit the budget.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::workspace::set_memory_sampling;
use lapack_rs::device::{DeviceBudget, DeviceWorkspace, Exactness, MergeU, dsyevd_device, zheevd_device};
use lapack_rs::{Job, Uplo};

fn ws(
    c: &ComputeClient<CpuRuntime>,
    n: usize,
    u: MergeU,
    exact: Exactness,
    budget: usize,
) -> DeviceWorkspace<CpuRuntime> {
    DeviceWorkspace::new(c, n, Job::Vectors, DeviceBudget { bytes: budget }, u).with_exactness(exact)
}

#[test]
fn lean_matches_staged_to_tolerance() {
    let c = CpuRuntime::client(&CpuDevice);
    set_memory_sampling(true);
    for &n in &[40usize, 150, 300] {
        let mut mats = vec![("random".to_string(), random_sym(n, 900 + n as u64))];
        for (name, d, e) in tridiag_fixtures(n) {
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                a[i + i * n] = d[i];
                if i + 1 < n {
                    a[i + 1 + i * n] = e[i];
                    a[i + (i + 1) * n] = e[i];
                }
            }
            mats.push((format!("tridiag {name}"), a));
        }
        for (name, a0) in mats {
            let mut staged = ws(&c, n, MergeU::Staged, Exactness::Reference, usize::MAX);
            let (mut as_, mut w_s) = (a0.clone(), vec![0.0; n]);
            dsyevd_device(&c, Job::Vectors, Uplo::Lower, n, &mut as_, &mut w_s, &mut staged).expect("staged");
            // Device memory is client-global: release the staged workspace first.
            let staged_peak = staged.counters().peak_device_bytes;
            drop(staged);
            let mut lean = ws(&c, n, MergeU::Lean, Exactness::Fast, usize::MAX);
            let (mut al, mut w_l) = (a0.clone(), vec![0.0; n]);
            dsyevd_device(&c, Job::Vectors, Uplo::Lower, n, &mut al, &mut w_l, &mut lean).expect("lean");
            let tag = format!("{name} n={n}");
            // Child merges feed the parent's z vector, so eigenvalues agree to
            // roundoff, not bits, beyond a single merge.
            for i in 0..n {
                assert!(
                    (w_l[i] - w_s[i]).abs() <= 1e-10 * w_s[i].abs().max(1.0),
                    "{tag}: w[{i}] {} vs {}",
                    w_l[i],
                    w_s[i]
                );
            }
            let (res, orth) = sym_residuals(n, &a0, &al, &w_l);
            let scale = w_l.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            println!(
                "{tag}: lean residual {res:e} orth {orth:e} (scale {scale:e}); peak staged {staged_peak} lean {}",
                lean.counters().peak_device_bytes
            );
            assert!(res <= 1e-9 * scale && orth <= 1e-9, "{tag}: {res:e} {orth:e}");
            if n >= 150 && name == "random" {
                assert!(lean.counters().peak_device_bytes < staged_peak, "{tag}: lean not smaller");
            }
        }
    }
    // Hermitian.
    let n = 120;
    let (re0, im0) = random_herm(n, 5);
    let mut lean = ws(&c, n, MergeU::Lean, Exactness::Fast, usize::MAX);
    let (mut ar, mut ai, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
    zheevd_device(&c, Job::Vectors, Uplo::Upper, n, &mut ar, &mut ai, &mut w, &mut lean).expect("lean z");
    let (res, orth) = herm_residuals(n, &re0, &im0, &ar, &ai, &w);
    assert!(res <= 1e-10 && orth <= 1e-10, "herm lean: {res:e} {orth:e}");
}

#[test]
fn auto_picks_lean_only_when_staged_does_not_fit() {
    let c = CpuRuntime::client(&CpuDevice);
    let n = 200;
    let a0 = random_sym(n, 17);
    let (mut ar, mut wr) = (a0.clone(), vec![0.0; n]);
    lapack_rs::dsyevd_alloc(Job::Vectors, Uplo::Lower, n, &mut ar, &mut wr).expect("host");
    // Unlimited: Auto under Fast stays staged. The fused back-transform leaves
    // the eigenvalues alone, so staged merges keep them bit-exact (a lean merge
    // would not); the vectors are tolerance-gated.
    let mut big = ws(&c, n, MergeU::Auto, Exactness::Fast, usize::MAX);
    let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
    dsyevd_device(&c, Job::Vectors, Uplo::Lower, n, &mut a, &mut w, &mut big).expect("auto big");
    assert_bits("auto unlimited w (staged)", &w, &wr);
    let dz = a.iter().zip(&ar).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max);
    assert!(dz <= 1e-10, "auto unlimited z: {dz:e}");
    // A budget below the staged need but above lean's: Auto goes lean and succeeds;
    // Reference refuses with WorkspaceTooSmall.
    let lda = lapack_rs::device::geometry::lda_for(&c, n);
    let budget = 8 * (lda * n + 2 * 32 * lda + 8 * n * 32 + 16 * n) + 8 * n * n / 2;
    let mut small = ws(&c, n, MergeU::Auto, Exactness::Fast, budget);
    let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
    dsyevd_device(&c, Job::Vectors, Uplo::Lower, n, &mut a, &mut w, &mut small).expect("auto small → lean");
    let (res, orth) = sym_residuals(n, &a0, &a, &w);
    assert!(res <= 1e-9 * 40.0 && orth <= 1e-9, "{res:e} {orth:e}");
    let mut refuse = ws(&c, n, MergeU::Auto, Exactness::Reference, budget);
    let r = dsyevd_device(&c, Job::Vectors, Uplo::Lower, n, &mut a0.clone(), &mut vec![0.0; n], &mut refuse);
    assert!(matches!(r, Err(lapack_rs::LapackError::WorkspaceTooSmall { .. })), "{r:?}");
}

#[test]
fn fast_back_transform_matches_to_tolerance() {
    let c = CpuRuntime::client(&CpuDevice);
    for &n in &[34usize, 100, 257] {
        let a0 = random_sym(n, 300 + n as u64);
        for uplo in [Uplo::Lower, Uplo::Upper] {
            let (mut ar, mut wr) = (a0.clone(), vec![0.0; n]);
            lapack_rs::dsyevd_alloc(Job::Vectors, uplo, n, &mut ar, &mut wr).expect("host");
            let mut fast = ws(&c, n, MergeU::Staged, Exactness::Fast, usize::MAX);
            let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
            dsyevd_device(&c, Job::Vectors, uplo, n, &mut a, &mut w, &mut fast).expect("fast");
            // Staged merge + fused back-transform: eigenvalues are untouched by
            // DORMTR, so they stay bit-exact; vectors are tolerance-gated.
            assert_bits(&format!("fast w n={n} {uplo:?}"), &w, &wr);
            let dz = a.iter().zip(&ar).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max);
            let (res, orth) = sym_residuals(n, &a0, &a, &w);
            println!("fast n={n} {uplo:?}: max |Δz| {dz:e}, residual {res:e}, orth {orth:e}");
            assert!(dz <= 1e-10 && res <= 1e-9 * 40.0 && orth <= 1e-9, "fast n={n} {uplo:?}");
        }
        let (re0, im0) = random_herm(n, 400 + n as u64);
        for uplo in [Uplo::Lower, Uplo::Upper] {
            let (mut hr, mut hi, mut wh) = (re0.clone(), im0.clone(), vec![0.0; n]);
            lapack_rs::zheevd_alloc(Job::Vectors, uplo, n, &mut hr, &mut hi, &mut wh).expect("host");
            let mut fast = ws(&c, n, MergeU::Lean, Exactness::Fast, usize::MAX);
            let (mut ar, mut ai, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
            zheevd_device(&c, Job::Vectors, uplo, n, &mut ar, &mut ai, &mut w, &mut fast).expect("fast z");
            let (res, orth) = herm_residuals(n, &re0, &im0, &ar, &ai, &w);
            println!("fast+lean herm n={n} {uplo:?}: residual {res:e}, orth {orth:e}");
            assert!(res <= 1e-10 * 40.0 && orth <= 1e-10, "herm fast n={n} {uplo:?}");
            for i in 0..n {
                assert!((w[i] - wh[i]).abs() <= 1e-10 * wh[i].abs().max(1.0));
            }
        }
    }
}
