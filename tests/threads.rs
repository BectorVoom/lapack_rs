//! The default-runtime entry points reproduce the sequential reference
//! transcription bit for bit, at any launch geometry.
//!
//! `dsyevd_alloc`/`zheevd_alloc` run the CubeCL drivers on
//! `runtime::DefaultRuntime`; `dsyevd_alloc_host`/`zheevd_alloc_host` are the
//! sequential host transcription at `*_opt_work_len`, which is the value the
//! reference produces. Every unit of every kernel is an independent
//! reference-ordered chain, so neither the number of units per cube (the CPU
//! runtime's "thread count") nor the cubes-per-SM target may move a bit.

mod common;
use common::*;
use lapack_rs::device::geometry;
use lapack_rs::{Job, Uplo, dsyevd_alloc, dsyevd_alloc_host, zheevd_alloc, zheevd_alloc_host};

/// Restore the process-global geometry knobs even if an assertion panics.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        geometry::set_cpu_dim_override(0);
        geometry::set_cubes_per_sm(u32::MAX);
    }
}

#[test]
fn syevd_default_runtime_matches_host_at_any_geometry() {
    let _restore = Restore;
    for &n in &[26usize, 129, 400] {
        let a0 = random_sym(n, 21 + n as u64);
        let (mut ah, mut wh) = (a0.clone(), vec![0.0; n]);
        dsyevd_alloc_host(Job::Vectors, Uplo::Upper, n, &mut ah, &mut wh).expect("host");
        let mut run = |tag: &str| {
            let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
            dsyevd_alloc(Job::Vectors, Uplo::Upper, n, &mut a, &mut w).expect("dsyevd_alloc");
            assert_bits(&format!("w n={n} {tag}"), &w, &wh);
            assert_bits(&format!("Z n={n} {tag}"), &a, &ah);
        };
        run("default");
        for dim in [1u32, 3, 16] {
            geometry::set_cpu_dim_override(dim);
            run(&format!("cpu_dim={dim}"));
            geometry::set_cpu_dim_override(0);
        }
        for per_sm in [0u32, 1, 16] {
            geometry::set_cubes_per_sm(per_sm);
            run(&format!("cubes_per_sm={per_sm}"));
            geometry::set_cubes_per_sm(u32::MAX);
        }
    }
}

#[test]
fn heevd_default_runtime_matches_host_at_any_geometry() {
    let _restore = Restore;
    for &n in &[26usize, 129, 300] {
        let (re0, im0) = random_herm(n, 31 + n as u64);
        let (mut rh, mut ih, mut wh) = (re0.clone(), im0.clone(), vec![0.0; n]);
        zheevd_alloc_host(Job::Vectors, Uplo::Lower, n, &mut rh, &mut ih, &mut wh).expect("host");
        let mut run = |tag: &str| {
            let (mut re, mut im, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
            zheevd_alloc(Job::Vectors, Uplo::Lower, n, &mut re, &mut im, &mut w).expect("zheevd_alloc");
            assert_bits(&format!("w n={n} {tag}"), &w, &wh);
            assert_bits(&format!("Z re n={n} {tag}"), &re, &rh);
            assert_bits(&format!("Z im n={n} {tag}"), &im, &ih);
        };
        run("default");
        for dim in [1u32, 16] {
            geometry::set_cpu_dim_override(dim);
            run(&format!("cpu_dim={dim}"));
            geometry::set_cpu_dim_override(0);
        }
    }
}
