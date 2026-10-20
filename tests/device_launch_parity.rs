//! On the CPU runtime `dsyevd_device` is bit-identical across the bounds
//! checked `launch` and `launch_unchecked`, across launch geometries (cube
//! dimensions), kernel variants (row-tiled K2r/K3r) and K1 layouts (K1, the
//! column-read K1col and its lane-vector form K1colv).
//! Every element is a reference-ordered chain, so any difference is an
//! indexing bug, not rounding. The switches are process-global; tests that
//! run concurrently may observe each other's settings, which is harmless
//! because no result depends on them.

#![cfg(feature = "cpu")]

mod common;
use common::*;

/// Restores a process-global test switch even if an assertion panics, so one
/// failure cannot change what the other tests in this binary exercise.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        ops::set_symv_mode(0);
        ops::set_variant(ops::Variant::Auto);
        ops::set_checked_launch(false);
        geometry::set_cpu_dim_override(0);
    }
}
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::{DeviceBudget, DeviceWorkspace, MergeU, dsyevd_device, geometry, ops};
use lapack_rs::{Job, Uplo};

#[test]
fn checked_unchecked_and_geometry_parity() {
    let _restore = Restore;
    let client = CpuRuntime::client(&CpuDevice);
    for &(n, uplo) in &[(70usize, Uplo::Lower), (131, Uplo::Upper)] {
        let a0 = random_sym(n, 4242 + n as u64);
        let run = |tag: &str| {
            let mut ws = DeviceWorkspace::new(&client, n, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Staged)
                .with_chunk_rows(9);
            let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
            dsyevd_device(&client, Job::Vectors, uplo, n, &mut a, &mut w, &mut ws)
                .unwrap_or_else(|e| panic!("{tag}: {e}"));
            (a, w)
        };
        let (a_ref, w_ref) = run("unchecked");
        ops::set_checked_launch(true);
        let (a_chk, w_chk) = run("checked");
        ops::set_checked_launch(false);
        assert_bits(&format!("checked w n={n}"), &w_chk, &w_ref);
        assert_bits(&format!("checked z n={n}"), &a_chk, &a_ref);
        // Kernel variants: row-tiled K2r/K3r (register locals) against the
        // simple K2/K3, forced on the CPU runtime at small n.
        ops::set_variant(ops::Variant::RowTiled);
        let (a_rt, w_rt) = run("row-tiled");
        ops::set_variant(ops::Variant::Auto);
        assert_bits(&format!("row-tiled w n={n}"), &w_rt, &w_ref);
        assert_bits(&format!("row-tiled z n={n}"), &a_rt, &a_ref);
        for dim in [1u32, 3, 16] {
            geometry::set_cpu_dim_override(dim);
            let (a_g, w_g) = run("geometry");
            geometry::set_cpu_dim_override(0);
            assert_bits(&format!("dim={dim} w n={n}"), &w_g, &w_ref);
            assert_bits(&format!("dim={dim} z n={n}"), &a_g, &a_ref);
        }
    }
}

/// K1col (column reads, the GPU default) against K1 on the CPU runtime, both
/// UPLO, vectors on: identical bits because both triangles of the trailing
/// block are bitwise equal.
#[test]
fn symv_column_layout_parity() {
    let _restore = Restore;
    let client = CpuRuntime::client(&CpuDevice);
    for &(n, uplo) in &[(70usize, Uplo::Lower), (131, Uplo::Upper), (131, Uplo::Lower), (70, Uplo::Upper)] {
        let a0 = random_sym(n, 5151 + n as u64);
        let run = |mode: u8| {
            ops::set_symv_mode(mode);
            let mut ws = DeviceWorkspace::new(&client, n, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Staged);
            let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
            dsyevd_device(&client, Job::Vectors, uplo, n, &mut a, &mut w, &mut ws).expect("device");
            ops::set_symv_mode(0);
            (a, w)
        };
        let (a1, w1) = run(1);
        let (a2, w2) = run(2);
        assert_bits(&format!("K1col w n={n} {uplo:?}"), &w2, &w1);
        assert_bits(&format!("K1col z n={n} {uplo:?}"), &a2, &a1);
        // K1colv: the same column reads through the lane-vector view, with a
        // scalar prologue/tail wherever `off + q` is not lane-aligned.
        let (a3, w3) = run(3);
        assert_bits(&format!("K1colv w n={n} {uplo:?}"), &w3, &w1);
        assert_bits(&format!("K1colv z n={n} {uplo:?}"), &a3, &a1);
        // Complex: K1c against its column-read form (conjugated off-diagonal).
        let (re0, im0) = random_herm(n, 6161 + n as u64);
        let zrun = |mode: u8| {
            ops::set_symv_mode(mode);
            let mut ws = DeviceWorkspace::new(&client, n, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Staged);
            let (mut ar, mut ai, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
            lapack_rs::device::zheevd_device(&client, Job::Vectors, uplo, n, &mut ar, &mut ai, &mut w, &mut ws)
                .expect("zheevd_device");
            ops::set_symv_mode(0);
            (ar, ai, w)
        };
        let (r1, i1, zw1) = zrun(1);
        let (r2, i2, zw2) = zrun(2);
        assert_bits(&format!("K1c-col w n={n} {uplo:?}"), &zw2, &zw1);
        assert_bits(&format!("K1c-col z re n={n} {uplo:?}"), &r2, &r1);
        assert_bits(&format!("K1c-col z im n={n} {uplo:?}"), &i2, &i1);
    }
}

/// **K8 and K9 (the MRRR leaves) under checked vs unchecked launches.**
///
/// `kl!` picks `launch` or `launch_unchecked` from a process-global switch, so
/// every kernel has to be pinned across it (AGENTS.md, device rules item 9).
/// Only the leaves cubecl-cpu 0.10 can lower are here; the rest are covered on
/// the GPU (`docs/cubecl-cpu-lowering-limit.md`).
#[test]
fn mrrr_leaves_checked_unchecked_parity() {
    let _restore = Restore;
    let client = CpuRuntime::client(&CpuDevice);
    use lapack_rs::device::workspace::{read_range, upload};

    let n = 13usize;
    let mut r = Rng::new(88_001);
    let d0: Vec<f64> = (0..n).map(|_| r.uniform()).collect();
    let e0: Vec<f64> = (0..n).map(|i| if i + 1 < n { r.uniform() } else { 0.0 }).collect();
    let e2: Vec<f64> = e0.iter().map(|v| v * v).collect();
    let lld: Vec<f64> = (0..n)
        .map(|i| {
            let den = if d0[i].abs() < 1e-3 { 1e-3 } else { d0[i] };
            let l = e0[i] / den;
            l * l * d0[i]
        })
        .collect();
    let (gl, gu) = (-4.0f64, 4.0f64);
    let pivmin = f64::MIN_POSITIVE;
    let reltol = f64::EPSILON.sqrt();
    let eps = lapack_rs::testing::DLAMCH.iter().find(|(c, _)| *c == 'P').expect("P").1;
    let itmax: Vec<u32> = vec![lapack_rs::testing::larrk_itmax(gu.abs(), pivmin) as u32; 1];

    // K8: DLAEV2, DLARRK and DLANEG under both launch modes.
    let run = || -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
        let abc = upload(&client, &[1.0f64, 0.5, -2.0]);
        let out = upload(&client, &vec![0.0f64; 4]);
        ops::laev2_batch::<CpuRuntime, f64>(&client, &abc, &out, 1);
        let mut laev2 = vec![0.0f64; 4];
        read_range(&client, &out, 4, 0, 4, &mut laev2);

        let hd = upload(&client, &d0);
        let he2 = upload(&client, &e2);
        let hnptr = upload(&client, &[0u32, n as u32]);
        let hgl = upload(&client, &[gl]);
        let hgu = upload(&client, &[gu]);
        let hc = upload(&client, &[reltol, pivmin, eps]);
        let hit = upload(&client, &itmax);
        let hw = upload(&client, &vec![0.0f64; n]);
        let hwe = upload(&client, &vec![0.0f64; n]);
        let hi = upload(&client, &vec![0u32; n]);
        ops::larrk_batch::<CpuRuntime, f64>(&client, &hd, &he2, &hnptr, &hgl, &hgu, &hc, &hit, &hw, &hwe, &hi, 1, n, n);
        let mut w = vec![0.0f64; n];
        let mut info = vec![0u32; n];
        read_range(&client, &hw, n, 0, n, &mut w);
        read_range(&client, &hi, n, 0, n, &mut info);

        let hlld = upload(&client, &lld);
        let hs = upload(&client, &[0.5f64]);
        let hr = upload(&client, &[(n as u32 + 1) / 2]);
        let hn = upload(&client, &[0u32]);
        ops::laneg_batch::<CpuRuntime, f64>(&client, &hd, &hlld, &hnptr, &hs, &hr, &hn, 1, n);
        let mut neg = vec![0u32; 1];
        read_range(&client, &hn, 1, 0, 1, &mut neg);

        (laev2, w, info, neg)
    };

    ops::set_checked_launch(false);
    let unchecked = run();
    ops::set_checked_launch(true);
    let checked = run();
    ops::set_checked_launch(false);

    assert_bits("K8 dlaev2 checked vs unchecked", &checked.0, &unchecked.0);
    assert_bits("K8 dlarrk W checked vs unchecked", &checked.1, &unchecked.1);
    assert_eq!(checked.2, unchecked.2, "K8 dlarrk INFO checked vs unchecked");
    assert_eq!(checked.3, unchecked.3, "K8 dlaneg checked vs unchecked");
    // The kernels ran: a silent no-op would leave the uploaded zeros.
    assert!(unchecked.1.iter().any(|&v| v != 0.0), "K8 dlarrk wrote nothing");
}
