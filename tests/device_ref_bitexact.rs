//! `dsyevd_device` (`Exactness::Reference`) against Reference-LAPACK 3.12.1
//! directly, bit for bit, when the CPU runtime's contraction probe reports no
//! contraction. The oracle runs with `LWORK`/`LIWORK` from
//! `syevd_opt_work_len`.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::probe::contraction_probe;
use lapack_rs::device::{DeviceBudget, DeviceWorkspace, MergeU, dsyevd_device};
use lapack_rs::{Job, Uplo, syevd_opt_work_len};

#[test]
fn dsyevd_device_matches_reference_lapack() {
    if !lapack_ref::require("dsyevd_device_matches_reference_lapack") {
        return;
    }
    let client = CpuRuntime::client(&CpuDevice);
    let probe = contraction_probe::<CpuRuntime, f64>(&client, f64::EPSILON, 1.0, 2.0);
    if !probe.no_contraction() {
        println!("SKIP: CPU runtime contracts ({probe:?}); bit-exactness not claimed");
        return;
    }
    let mut ws = DeviceWorkspace::new(&client, 64, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
    let mut cases: Vec<(String, usize, Vec<f64>)> = Vec::new();
    for &n in &[2usize, 17, 33, 34, 64, 100, 257] {
        cases.push((format!("random n={n}"), n, random_sym(n, 7000 + n as u64)));
    }
    for (name, a) in dense_fixtures(90) {
        cases.push((name.to_string(), 90, a));
    }
    for (tag, n, a0) in &cases {
        let n = *n;
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for job in [Job::ValuesOnly, Job::Vectors] {
                let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
                dsyevd_device(&client, job, uplo, n, &mut a, &mut w, &mut ws).expect("device");
                let (lw, liw) = syevd_opt_work_len(n, job);
                let (mut ar, mut wr) = (a0.clone(), vec![0.0; n]);
                let (mut work, mut iwork) = (vec![0.0; lw], vec![0i32; liw]);
                let jz = if job == Job::Vectors { 'V' } else { 'N' };
                let ul = if uplo == Uplo::Upper { 'U' } else { 'L' };
                let info = lapack_ref::dsyevd(jz, ul, n, &mut ar, &mut wr, &mut work, &mut iwork);
                assert_eq!(info, 0, "{tag}");
                let t = format!("{tag} {uplo:?} {job:?}");
                assert_bits(&format!("w {t}"), &w, &wr);
                if job == Job::Vectors {
                    assert_bits(&format!("z {t}"), &a, &ar);
                }
            }
        }
    }
}

#[test]
fn zheevd_device_matches_reference_lapack() {
    if !lapack_ref::require("zheevd_device_matches_reference_lapack") {
        return;
    }
    let client = CpuRuntime::client(&CpuDevice);
    let probe = contraction_probe::<CpuRuntime, f64>(&client, f64::EPSILON, 1.0, 2.0);
    if !probe.no_contraction() {
        println!("SKIP: CPU runtime contracts ({probe:?}); bit-exactness not claimed");
        return;
    }
    let mut ws = DeviceWorkspace::new(&client, 64, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
    for &n in &[2usize, 20, 33, 34, 64, 131] {
        let (re0, mut im0) = random_herm(n, 8100 + n as u64);
        if n % 2 == 0 {
            for i in 0..n {
                im0[i + i * n] = -1.5 + i as f64; // ignored by ZHEEVD
            }
        }
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for job in [Job::ValuesOnly, Job::Vectors] {
                let (mut ar, mut ai, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
                lapack_rs::device::zheevd_device(&client, job, uplo, n, &mut ar, &mut ai, &mut w, &mut ws)
                    .expect("device");
                let (lw, lrw, liw) = lapack_rs::heevd_opt_work_len(n, job);
                let mut a = to_c64(&re0, &im0);
                let mut wr = vec![0.0; n];
                let mut work = vec![lapack_ref::C64 { re: 0.0, im: 0.0 }; lw];
                let (mut rwork, mut iwork) = (vec![0.0; lrw], vec![0i32; liw]);
                let jz = if job == Job::Vectors { 'V' } else { 'N' };
                let ul = if uplo == Uplo::Upper { 'U' } else { 'L' };
                let info = lapack_ref::zheevd(jz, ul, n, &mut a, &mut wr, &mut work, &mut rwork, &mut iwork);
                assert_eq!(info, 0);
                let t = format!("n={n} {uplo:?} {job:?}");
                assert_bits(&format!("w {t}"), &w, &wr);
                if job == Job::Vectors {
                    let (rr, ri) = from_c64(&a);
                    assert_bits(&format!("z re {t}"), &ar, &rr);
                    assert_bits(&format!("z im {t}"), &ai, &ri);
                }
            }
        }
    }
}
