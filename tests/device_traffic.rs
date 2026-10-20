//! Transfer and memory gates of `dsyevd_device` on the CPU runtime:
//! per-phase read/upload counts follow the design (one staging upload and one
//! staging launch; one `v` upload, one K1 launch and one `w` read per column;
//! one panel read and one `[V|W]` upload per panel; chunked final read), and
//! device memory stays within the table
//! bound (peak live bytes ≤ 2.25 n² staged plus O(NB·lda) scratch; the pool's
//! reserved pages are reported, not gated). One test: memory is
//! process-global.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::{DeviceBudget, DeviceWorkspace, MergeU, dsyevd_device};
use lapack_rs::{Job, Uplo};

#[test]
fn traffic_and_memory() {
    let client = CpuRuntime::client(&CpuDevice);
    let n = 256usize;
    let a0 = random_sym(n, 99);
    let lda = lapack_rs::device::geometry::lda_for(&client, n);

    // Values only: sytrd traffic.
    let mut ws = DeviceWorkspace::new(&client, n, Job::ValuesOnly, DeviceBudget::UNLIMITED, MergeU::Staged);
    let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
    dsyevd_device(&client, Job::ValuesOnly, Uplo::Lower, n, &mut a, &mut w, &mut ws).expect("values");
    let c = ws.counters();
    println!("values-only n={n}: {c:?}");
    let nb = 32;
    let panels = (n - nb).div_ceil(nb); // blocked panels while n - i + 1 > NX
    let cols = panels * nb;
    // uploads: staging + per column v + per panel [V|W]
    assert!(c.uploads <= 1 + cols + panels, "uploads {c:?}");
    // reads: per panel columns + per column w + the DSYTD2 tail
    assert!(c.reads <= panels + cols + 1, "reads {c:?}");
    // launches: staging (the symmetric image is mirrored on the device, not
    // expanded on the host) + per column K1 + per panel K2
    assert!(c.launches <= 1 + cols + panels, "launches {c:?}");
    // Panel reads are NB·lda reals each, w reads ≤ n, the tail ≤ lda·NX.
    let read_bound = 8 * (panels * nb * lda + cols * n + (nb + 1) * lda);
    assert!(c.read_bytes <= read_bound, "read bytes {c:?} > {read_bound}");

    // Vectors: the final read is chunked, the total read is bounded.
    ws.reset_counters();
    lapack_rs::device::workspace::set_memory_sampling(true);
    let before = client.memory_usage().expect("usage");
    let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
    dsyevd_device(&client, Job::Vectors, Uplo::Lower, n, &mut a, &mut w, &mut ws).expect("vectors");
    let c = ws.counters();
    let after = client.memory_usage().expect("usage");
    println!("vectors n={n}: {c:?}");
    println!("memory before {before:?}\nmemory after  {after:?}");
    // staging lda·n + per-column v (≤ n·lda) + [V|W] (≤ 2n·lda) + staged U
    // over the tree (≤ 4n², geometric) + back-transform blocks (≤ 2n·kpad
    // per block, n/NB blocks → ≈ 2n² + coefficients).
    let up_bound = 8 * (4 * lda * n + 6 * n * n + 64 * lda);
    assert!(c.upload_bytes <= up_bound, "upload bytes {c:?} > {up_bound}");
    // Live device bytes at each phase peak: M plus the staged merge
    // (≤ 1.25 n² over M on the top merge) plus fixed O(NB·lda) scratch.
    let peak = c.peak_device_bytes as f64;
    let bound = 8.0 * (2.25 * (lda * n) as f64 + (2 * 32 * lda + 4 * n * 32 + 64 * lda) as f64) + 65536.0;
    println!("peak live {peak} vs bound {bound} (reserved {})", after.bytes_reserved);
    assert!(peak > 0.0 && peak <= bound, "device memory {peak} > {bound}");
    let (res, orth) = sym_residuals(n, &a0, &a, &w);
    assert!(res <= 1e-9 && orth <= 1e-9, "{res:e} {orth:e}");
}
