//! Hybrid `dormtr_device` (Exactness::Reference) on the CPU runtime:
//! bit-identical to the host `dormtr` for both `uplo`, blocked sizes with
//! full and short last blocks.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::Uplo;

#[test]
fn dormtr_device_matches_host_bitwise() {
    let client = CpuRuntime::client(&CpuDevice);
    for &n in &[2usize, 20, 34, 45, 65, 97, 130] {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            let mut a = random_sym(n, 400 + n as u64);
            let m1 = n.max(2) - 1;
            let (mut d, mut e, mut tau) = (vec![0.0; n], vec![0.0; m1], vec![0.0; m1]);
            let mut work = vec![0.0; 32 * n];
            lapack_rs::blocks::dsytrd(uplo, n, &mut a, &mut d, &mut e, &mut tau, &mut work).expect("dsytrd");
            let c0 = random_sym(n, 500 + n as u64);
            let mut ch = c0.clone();
            let mut wk = vec![0.0; 32 * n + 4160];
            lapack_rs::blocks::dormtr_ln(uplo, n, &mut a.clone(), &mut tau.clone(), &mut ch, &mut wk).expect("host");
            let mut c = c0.clone();
            let cnt = lapack_rs::device::dormtr_device(&client, uplo, n, &mut a.clone(), &mut tau.clone(), &mut c)
                .expect("device");
            assert_bits(&format!("dormtr n={n} {uplo:?}"), &c, &ch);
            println!("n={n} {uplo:?}: {cnt:?}");
        }
    }
}
