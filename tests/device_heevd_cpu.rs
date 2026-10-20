//! `zheevd_device` on the CPU runtime (`Exactness::Reference`): bit-identical
//! to the host `zheevd_alloc` (itself bit-identical to Reference-LAPACK) for
//! both `uplo` and `job`, the unblocked (`n ≤ 33`) and `ZSTEQR` (`n ≤ 25`)
//! paths, a garbage imaginary diagonal and workspace reuse; plus residuals.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::{DeviceBudget, DeviceWorkspace, MergeU, dsyevd_device, zheevd_device};
use lapack_rs::{Job, Uplo};

fn host(job: Job, uplo: Uplo, n: usize, re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut ar, mut ai, mut w) = (re.to_vec(), im.to_vec(), vec![0.0; n]);
    lapack_rs::zheevd_alloc(job, uplo, n, &mut ar, &mut ai, &mut w).expect("host");
    (ar, ai, w)
}

#[test]
fn zheevd_device_matches_host_bitwise() {
    let client = CpuRuntime::client(&CpuDevice);
    let mut ws = DeviceWorkspace::new(&client, 8, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
    for &n in &[1usize, 2, 7, 25, 26, 33, 34, 50, 65, 97, 140] {
        let (re0, mut im0) = random_herm(n, 1300 + n as u64);
        // ZHEEVD ignores the imaginary diagonal; poison it.
        for i in 0..n {
            im0[i + i * n] = 0.37 * (i as f64 + 1.0);
        }
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for job in [Job::ValuesOnly, Job::Vectors] {
                let (hr, hi, hw) = host(job, uplo, n, &re0, &im0);
                let (mut ar, mut ai, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
                zheevd_device(&client, job, uplo, n, &mut ar, &mut ai, &mut w, &mut ws).expect("device");
                let tag = format!("n={n} {uplo:?} {job:?}");
                assert_bits(&format!("w {tag}"), &w, &hw);
                if job == Job::Vectors {
                    assert_bits(&format!("z re {tag}"), &ar, &hr);
                    assert_bits(&format!("z im {tag}"), &ai, &hi);
                }
            }
        }
    }
    // The same workspace still serves the real path afterwards.
    let n = 40;
    let a0 = random_sym(n, 3);
    let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
    dsyevd_device(&client, Job::Vectors, Uplo::Lower, n, &mut a, &mut w, &mut ws).expect("real after complex");
    let (mut ah, mut wh) = (a0.clone(), vec![0.0; n]);
    lapack_rs::dsyevd_alloc(Job::Vectors, Uplo::Lower, n, &mut ah, &mut wh).expect("host");
    assert_bits("real w", &w, &wh);
    assert_bits("real z", &a, &ah);
}

#[test]
fn zheevd_device_residuals() {
    let client = CpuRuntime::client(&CpuDevice);
    let n = 120;
    let (re0, im0) = random_herm(n, 21);
    let mut ws =
        DeviceWorkspace::new(&client, n, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Staged).with_chunk_rows(7);
    let (mut ar, mut ai, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
    zheevd_device(&client, Job::Vectors, Uplo::Upper, n, &mut ar, &mut ai, &mut w, &mut ws).expect("device");
    let (res, orth) = herm_residuals(n, &re0, &im0, &ar, &ai, &w);
    assert!(res <= 1e-10 && orth <= 1e-10, "{res:e} {orth:e}");
}
