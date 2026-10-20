//! Device memory and wall time of `dsyevd_device` on the CPU runtime.
//! `cargo run --release --features device,cpu --example device_memory -- 1024`
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::{DeviceBudget, DeviceWorkspace, MergeU, dsyevd_device};
use lapack_rs::{Job, Uplo};

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let client = CpuRuntime::client(&CpuDevice);
    let mut s = 12345u64;
    let mut a0 = vec![0.0f64; n * n];
    for j in 0..n {
        for i in j..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            a0[i + j * n] = v;
            a0[j + i * n] = v;
        }
    }
    lapack_rs::device::workspace::set_memory_sampling(true);
    let before = client.memory_usage().unwrap();
    let mut ws = DeviceWorkspace::new(&client, n, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Staged);
    for job in [Job::ValuesOnly, Job::Vectors] {
        let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
        ws.reset_counters();
        let t = std::time::Instant::now();
        dsyevd_device(&client, job, Uplo::Lower, n, &mut a, &mut w, &mut ws).unwrap();
        let dt = t.elapsed().as_secs_f64();
        let after = client.memory_usage().unwrap();
        println!("n={n} {job:?}: {dt:.3}s  {:?}", ws.counters());
        println!(
            "  peak live {:.1} MiB, reserved {:.1} MiB (before {:.1}), n² = {:.1} MiB",
            ws.counters().peak_device_bytes as f64 / 1048576.0,
            after.bytes_reserved as f64 / 1048576.0,
            before.bytes_reserved as f64 / 1048576.0,
            (8 * n * n) as f64 / 1048576.0
        );
    }
    ws.release_transients();
    let _ = client.memory_usage();
    println!(
        "after release_transients: reserved {:.1} MiB",
        client.memory_usage().unwrap().bytes_reserved as f64 / 1048576.0
    );
    let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
    let t = std::time::Instant::now();
    lapack_rs::dsyevd_alloc(Job::Vectors, Uplo::Lower, n, &mut a, &mut w).unwrap();
    println!("host dsyevd_alloc: {:.3}s", t.elapsed().as_secs_f64());
}
