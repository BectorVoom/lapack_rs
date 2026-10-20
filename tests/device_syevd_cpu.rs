//! `dsyevd_device` on the CPU runtime (`Exactness::Reference`, staged U):
//! bit-identical to the host `dsyevd_alloc` (itself bit-identical to
//! Reference-LAPACK), across `uplo`, `job`, merge chunk `R ∈ {1, 7, n}` and
//! workspace reuse; plus residual sanity and budget errors.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::{DeviceBudget, DeviceWorkspace, Exactness, MergeU, dsyevd_device};
use lapack_rs::{Job, LapackError, Uplo};

fn client() -> ComputeClient<CpuRuntime> {
    CpuRuntime::client(&CpuDevice)
}

fn host(job: Job, uplo: Uplo, n: usize, a0: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (mut a, mut w) = (a0.to_vec(), vec![0.0; n]);
    lapack_rs::dsyevd_alloc(job, uplo, n, &mut a, &mut w).expect("host");
    (a, w)
}

#[test]
fn dsyevd_device_matches_host_bitwise() {
    let client = client();
    // One workspace reused across every size (grows on demand).
    let mut ws = DeviceWorkspace::new(&client, 8, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
    for &n in &[1usize, 2, 5, 26, 33, 34, 51, 64, 97, 130, 200] {
        let a0 = random_sym(n, 900 + n as u64);
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for job in [Job::ValuesOnly, Job::Vectors] {
                let (ah, wh) = host(job, uplo, n, &a0);
                let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
                dsyevd_device(&client, job, uplo, n, &mut a, &mut w, &mut ws).expect("device");
                let tag = format!("n={n} {uplo:?} {job:?}");
                assert_bits(&format!("w {tag}"), &w, &wh);
                if job == Job::Vectors {
                    assert_bits(&format!("z {tag}"), &a, &ah);
                }
            }
        }
    }
    println!("counters: {:?}", ws.counters());
}

#[test]
fn dsyevd_device_chunk_rows_and_fixtures() {
    let client = client();
    let n = 120;
    for (name, a0) in dense_fixtures(n) {
        let (ah, wh) = host(Job::Vectors, Uplo::Lower, n, &a0);
        for r in [1usize, 7, n] {
            let mut ws = DeviceWorkspace::new(&client, n, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Staged)
                .with_chunk_rows(r);
            let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
            dsyevd_device(&client, Job::Vectors, Uplo::Lower, n, &mut a, &mut w, &mut ws).expect("device");
            assert_bits(&format!("w {name} R={r}"), &w, &wh);
            assert_bits(&format!("z {name} R={r}"), &a, &ah);
        }
    }
}

#[test]
fn dsyevd_device_residuals_and_determinism() {
    let client = client();
    let n = 150;
    let a0 = random_sym(n, 77);
    let mut ws = DeviceWorkspace::new(&client, n, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
    let (mut a1, mut w1) = (a0.clone(), vec![0.0; n]);
    dsyevd_device(&client, Job::Vectors, Uplo::Upper, n, &mut a1, &mut w1, &mut ws).expect("run 1");
    let (mut a2, mut w2) = (a0.clone(), vec![0.0; n]);
    dsyevd_device(&client, Job::Vectors, Uplo::Upper, n, &mut a2, &mut w2, &mut ws).expect("run 2");
    assert_bits("run-to-run w", &w1, &w2);
    assert_bits("run-to-run z", &a1, &a2);
    let (res, orth) = sym_residuals(n, &a0, &a1, &w1);
    assert!(res <= 1e-9 && orth <= 1e-9, "residual {res:e}, orthogonality {orth:e}");
}

#[test]
fn dsyevd_device_budget_errors() {
    let client = client();
    let n = 64;
    let a0 = random_sym(n, 5);
    let mut tiny = DeviceWorkspace::new(&client, n, Job::Vectors, DeviceBudget { bytes: 1024 }, MergeU::Staged);
    let r = dsyevd_device(&client, Job::Vectors, Uplo::Lower, n, &mut a0.clone(), &mut vec![0.0; n], &mut tiny);
    assert!(matches!(r, Err(LapackError::WorkspaceTooSmall { .. })), "{r:?}");
    let mut lean = DeviceWorkspace::new(&client, n, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Lean);
    let r = dsyevd_device(&client, Job::Vectors, Uplo::Lower, n, &mut a0.clone(), &mut vec![0.0; n], &mut lean);
    assert!(matches!(r, Err(LapackError::InvalidArgument { .. })), "{r:?}");
    let _ = Exactness::Fast;
}
