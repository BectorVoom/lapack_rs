//! Device-resident entry points (`*_device_resident`): the input stays a
//! device buffer, the eigenvectors come back as a device matrix. Bit-identical
//! to the host-slice device API (and so to Reference-LAPACK), including the
//! scaled (`DLASCL`) and unblocked (`n ≤ 33`) branches; the input buffer is
//! unchanged and the final Z is never read back.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::workspace::{read_range, upload};
use lapack_rs::device::{
    DeviceBudget, DeviceWorkspace, MergeU, dsyevd_device, dsyevd_device_resident, zheevd_device, zheevd_device_resident,
};
use lapack_rs::{Job, Uplo};

fn read_all(c: &ComputeClient<CpuRuntime>, h: &cubecl::server::Handle, len: usize) -> Vec<f64> {
    let mut v = vec![0.0; len];
    read_range(c, h, len, 0, len, &mut v);
    v
}

#[test]
fn dsyevd_resident_matches_slice_api() {
    let c = CpuRuntime::client(&CpuDevice);
    let mut ws = DeviceWorkspace::new(&c, 8, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
    for &n in &[1usize, 2, 20, 33, 34, 70, 130] {
        let mut cases = vec![("random", random_sym(n, 50 + n as u64))];
        let mut tiny = random_sym(n, 60 + n as u64);
        tiny.iter_mut().for_each(|v| *v *= 1e-160);
        cases.push(("tiny", tiny));
        for (name, a0) in cases {
            for uplo in [Uplo::Lower, Uplo::Upper] {
                for job in [Job::ValuesOnly, Job::Vectors] {
                    let (mut ar, mut wr) = (a0.clone(), vec![0.0; n]);
                    dsyevd_device(&c, job, uplo, n, &mut ar, &mut wr, &mut ws).expect("slice");
                    let h = upload(&c, &a0);
                    let mut w = vec![0.0; n];
                    ws.reset_counters();
                    let z = dsyevd_device_resident(&c, job, uplo, n, &h, &mut w, &mut ws).expect("resident");
                    let reads = ws.counters().read_bytes;
                    let tag = format!("{name} n={n} {uplo:?} {job:?}");
                    assert_bits(&format!("w {tag}"), &w, &wr);
                    assert_bits(&format!("input unchanged {tag}"), &read_all(&c, &h, n * n), &a0);
                    match (job, z) {
                        (Job::Vectors, Some(z)) => {
                            let zc = read_all(&c, &z.compact(&c), n * n);
                            assert_bits(&format!("z {tag}"), &zc, &ar);
                            if n > 33 {
                                // Input read + panel/column reads, but no n² Z read at the end.
                                let (mut a2, mut w2) = (a0.clone(), vec![0.0; n]);
                                ws.reset_counters();
                                dsyevd_device(&c, job, uplo, n, &mut a2, &mut w2, &mut ws).expect("slice");
                                assert!(reads <= ws.counters().read_bytes, "{tag}: resident read more");
                            }
                        }
                        (Job::ValuesOnly, None) => {}
                        (j, z) => panic!("{tag}: {j:?} returned {:?}", z.is_some()),
                    }
                }
            }
        }
    }
}

#[test]
fn zheevd_resident_matches_slice_api() {
    let c = CpuRuntime::client(&CpuDevice);
    let mut ws = DeviceWorkspace::new(&c, 8, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
    for &n in &[1usize, 3, 26, 34, 90] {
        let (re0, im0) = random_herm(n, 70 + n as u64);
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for job in [Job::ValuesOnly, Job::Vectors] {
                let (mut sr, mut si, mut wr) = (re0.clone(), im0.clone(), vec![0.0; n]);
                zheevd_device(&c, job, uplo, n, &mut sr, &mut si, &mut wr, &mut ws).expect("slice");
                let (hr, hi) = (upload(&c, &re0), upload(&c, &im0));
                let mut w = vec![0.0; n];
                let z = zheevd_device_resident(&c, job, uplo, n, &hr, &hi, &mut w, &mut ws).expect("resident");
                let tag = format!("n={n} {uplo:?} {job:?}");
                assert_bits(&format!("w {tag}"), &w, &wr);
                if job == Job::Vectors {
                    let z = z.expect("vectors");
                    let (cr, ci) = z.compact(&c);
                    assert_bits(&format!("z re {tag}"), &read_all(&c, &cr, n * n), &sr);
                    assert_bits(&format!("z im {tag}"), &read_all(&c, &ci, n * n), &si);
                } else {
                    assert!(z.is_none());
                }
            }
        }
    }
}
