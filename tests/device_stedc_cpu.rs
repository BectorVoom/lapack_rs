//! Hybrid `dstedc_device` on the CPU runtime: bit-identical to the host
//! `dstedc` on the deflation fixtures, for several merge row chunks.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;

#[test]
fn dstedc_device_matches_host_bitwise() {
    let client = CpuRuntime::client(&CpuDevice);
    for &n in &[1usize, 2, 20, 26, 45, 64, 100, 200] {
        for (name, d0, e0) in tridiag_fixtures(n) {
            let (lw, liw) = lapack_rs::stedc_work_len(n);
            let (mut dh, mut eh) = (d0.clone(), e0.clone());
            eh.push(0.0);
            let mut zh = vec![0.0; n * n];
            let (mut w, mut iw) = (vec![0.0; lw.max(1)], vec![0usize; liw.max(1)]);
            lapack_rs::blocks::dstedc(n, &mut dh, &mut eh, &mut zh, &mut w, &mut iw).expect("host");
            for r in [0usize, 1, 7, n] {
                let (mut d, mut e) = (d0.clone(), e0.clone());
                e.push(0.0);
                let mut z = vec![0.0; n * n];
                let c = lapack_rs::device::dstedc_device(&client, n, &mut d, &mut e, &mut z, r).expect("device");
                let tag = format!("{name} n={n} R={r}");
                assert_bits(&format!("d {tag}"), &d, &dh);
                assert_bits(&format!("Z {tag}"), &z, &zh);
                if r == 0 && n == 200 {
                    println!("{tag}: {c:?}");
                }
            }
        }
    }
}
