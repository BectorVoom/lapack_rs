//! Sub-range reads and sub-range kernel bindings through offset handles.

#![cfg(feature = "cpu")]

use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::workspace::{read_range, upload};

#[cube(launch)]
fn write_index<F: Float + CubeElement>(out: &mut Array<F>, base: u32) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = F::cast_from(base) + F::cast_from(ABSOLUTE_POS as u32);
    }
}

#[test]
fn subrange_read_returns_exactly_the_range() {
    let client = CpuRuntime::client(&CpuDevice);
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.5).collect();
    let h = upload(&client, &data);
    for (start, len) in [(0usize, 1usize), (3, 17), (512, 256), (999, 1), (0, 1000)] {
        let mut out = vec![0.0f64; len];
        read_range(&client, &h, data.len(), start, len, &mut out);
        assert_eq!(&out[..], &data[start..start + len], "range {start}+{len}");
    }
}

#[test]
fn subrange_binding_indexes_from_zero() {
    let client = CpuRuntime::client(&CpuDevice);
    let n = 64usize;
    let h = upload(&client, &vec![-1.0f64; n]);
    let (start, len) = (16usize, 24usize);
    let sub = h.clone().offset_start((start * 8) as u64).offset_end(((n - start - len) * 8) as u64);
    // SAFETY: the sub-range holds `len` f64 values.
    unsafe {
        write_index::launch::<f64, CpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(len as u32),
            ArrayArg::from_raw_parts(sub, len),
            1000,
        );
    }
    let mut all = vec![0.0f64; n];
    read_range(&client, &h, n, 0, n, &mut all);
    let honoured = (0..n).all(|i| {
        let want = if i >= start && i < start + len { 1000.0 + (i - start) as f64 } else { -1.0 };
        all[i] == want
    });
    println!("cpu runtime honours offset bindings: {honoured}");
    assert!(honoured, "offset binding not honoured; launchers must use base-offset indexing: {:?}", &all[..]);
}
