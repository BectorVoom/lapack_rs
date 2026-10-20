//! Hybrid `dsytrd_device` on the CPU runtime: bit-identical to the host
//! `dsytrd` (and so to Reference-LAPACK) with `LWORK = 32·n`.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::Uplo;

#[test]
fn dsytrd_device_matches_host_bitwise() {
    let client = CpuRuntime::client(&CpuDevice);
    for &n in &[1usize, 2, 16, 33, 45, 65, 100, 130] {
        let a0 = random_sym(n, 300 + n as u64);
        for uplo in [Uplo::Lower, Uplo::Upper] {
            let m1 = n.max(2) - 1;
            let (mut a, mut d, mut e, mut tau) = (a0.clone(), vec![0.0; n], vec![0.0; m1], vec![0.0; m1]);
            let c =
                lapack_rs::device::dsytrd_device(&client, uplo, n, &mut a, &mut d, &mut e, &mut tau).expect("device");
            let (mut ah, mut dh, mut eh, mut tauh) = (a0.clone(), vec![0.0; n], vec![0.0; m1], vec![0.0; m1]);
            let mut work = vec![0.0; (32 * n).max(1)];
            lapack_rs::blocks::dsytrd(uplo, n, &mut ah, &mut dh, &mut eh, &mut tauh, &mut work).expect("host");
            let tag = format!("n={n} {uplo:?}");
            assert_bits(&format!("d {tag}"), &d, &dh);
            assert_bits(&format!("e {tag}"), &e, &eh);
            assert_bits(&format!("tau {tag}"), &tau, &tauh);
            for j in 0..n {
                for i in 0..n {
                    if (uplo == Uplo::Lower && i >= j) || (uplo == Uplo::Upper && i <= j) {
                        assert_eq!(a[i + j * n].to_bits(), ah[i + j * n].to_bits(), "A({i},{j}) {tag}");
                    }
                }
            }
            println!("{tag}: {c:?}");
        }
    }
}
