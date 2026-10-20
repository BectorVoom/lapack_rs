//! f64 capability and contraction probes (manual 15 §3.3).

#![cfg(feature = "cpu")]

use cubecl::Runtime;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use lapack_rs::device::probe::{contraction_probe, require_f64, supports_f64};

#[test]
fn cpu_runtime_reports_f64_and_contraction() {
    let client = CpuRuntime::client(&CpuDevice);
    let hw = &client.properties().hardware;
    println!(
        "cpu runtime: plane_size_max={} cores={:?} load_width={} max_vector_size={} f64 widths={:?}",
        hw.plane_size_max,
        hw.num_cpu_cores,
        hw.load_width,
        hw.max_vector_size,
        client.io_optimized_vector_sizes(8).collect::<Vec<_>>()
    );
    assert!(supports_f64(&client));
    require_f64(&client).expect("f64");
    let p64 = contraction_probe::<CpuRuntime, f64>(&client, f64::EPSILON, 1.0, 2.0);
    let p32 = contraction_probe::<CpuRuntime, f32>(&client, f32::EPSILON, 1.0, 2.0);
    println!("contraction probe f64: {p64:?}  f32: {p32:?}");
}
