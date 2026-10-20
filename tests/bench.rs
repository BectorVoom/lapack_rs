//! Speed and peak-heap baseline against faer 0.24 `SelfAdjointEigen`
//! (`#[ignore]`; run with `cargo test --release --test bench -- --ignored
//! --nocapture --test-threads=1`).
//!
//! Candidates are interleaved within each pass after a warm-up, the minimum
//! over passes is reported, and peak heap is measured with a counting global
//! allocator from a reset baseline (inputs excluded, outputs and workspace
//! included for both).

#[path = "common/alloc.rs"]
mod alloc;
mod common;

use common::*;
use faer::{Mat, Side};
use lapack_rs::{Job, Uplo};
use std::time::Instant;

#[global_allocator]
static GLOBAL: alloc::Counting = alloc::Counting;

fn ns() -> Vec<usize> {
    std::env::var("LAPACK_RS_BENCH_N")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.parse().ok()).collect())
        .unwrap_or_else(|| vec![16, 64, 256, 512, 1024, 2048])
}

fn passes() -> usize {
    std::env::var("LAPACK_RS_BENCH_PASSES").ok().and_then(|s| s.parse().ok()).unwrap_or(3)
}

fn time<R>(f: impl FnOnce() -> R) -> (f64, usize, R) {
    alloc::reset_peak();
    let base = alloc::LIVE.load(std::sync::atomic::Ordering::Relaxed);
    let t = Instant::now();
    let r = f();
    let dt = t.elapsed().as_secs_f64();
    (dt, alloc::peak() - base, r)
}

#[test]
#[ignore]
fn bench_real_vs_faer() {
    println!(
        "threads={} (rayon), timing=min over {} interleaved passes after warm-up",
        rayon::current_num_threads(),
        passes()
    );
    println!(
        "{:>6} {:>12} {:>12} {:>8} {:>14} {:>14} {:>8}",
        "n", "lapack_rs s", "faer s", "ratio", "ours peak MiB", "faer peak MiB", "mem"
    );
    for n in ns() {
        let a0 = random_sym(n, 42 + n as u64);
        let fa = Mat::<f64>::from_fn(n, n, |i, j| a0[i + j * n]);
        let ours = || {
            let mut a = a0.clone();
            let mut w = vec![0.0; n];
            lapack_rs::dsyevd_alloc(Job::Vectors, Uplo::Lower, n, &mut a, &mut w).expect("dsyevd");
            (a, w)
        };
        let theirs = || fa.self_adjoint_eigen(Side::Lower).expect("faer evd");
        let reference = || {
            let (lw, liw) = lapack_rs::syevd_work_len(n, Job::Vectors);
            let mut a = a0.clone();
            let mut w = vec![0.0; n];
            let mut work = vec![0.0; lw];
            let mut iwork = vec![0i32; liw];
            lapack_ref::dsyevd('V', 'L', n, &mut a, &mut w, &mut work, &mut iwork)
        };
        let _ = ours();
        let _ = theirs();
        let (mut to, mut tf, mut tr, mut po, mut pf) = (f64::MAX, f64::MAX, f64::MAX, 0, 0);
        for _ in 0..passes() {
            let (t, p, _) = time(ours);
            to = to.min(t);
            po = po.max(p);
            let (t, p, _) = time(theirs);
            tf = tf.min(t);
            pf = pf.max(p);
            if lapack_ref::AVAILABLE && n <= 2048 {
                let (t, _, _) = time(reference);
                tr = tr.min(t);
            }
        }
        let mib = |b: usize| b as f64 / (1 << 20) as f64;
        println!(
            "{n:>6} {to:>12.4} {tf:>12.4} {:>8.3} {:>14.2} {:>14.2} {:>8.3}   ref-lapack(1 thread) {:.4}s",
            to / tf,
            mib(po),
            mib(pf),
            po as f64 / pf.max(1) as f64,
            if tr == f64::MAX { f64::NAN } else { tr }
        );
    }
}

#[test]
#[ignore]
fn bench_complex_vs_faer() {
    use faer::c64;
    println!("threads={} (rayon)", rayon::current_num_threads());
    println!(
        "{:>6} {:>12} {:>12} {:>8} {:>14} {:>14} {:>8}",
        "n", "lapack_rs s", "faer s", "ratio", "ours peak MiB", "faer peak MiB", "mem"
    );
    for n in ns() {
        let (re0, im0) = random_herm(n, 43 + n as u64);
        let fa = Mat::<c64>::from_fn(n, n, |i, j| c64::new(re0[i + j * n], im0[i + j * n]));
        let ours = || {
            let (mut re, mut im) = (re0.clone(), im0.clone());
            let mut w = vec![0.0; n];
            lapack_rs::zheevd_alloc(Job::Vectors, Uplo::Lower, n, &mut re, &mut im, &mut w).expect("zheevd");
            (re, im, w)
        };
        let theirs = || fa.self_adjoint_eigen(Side::Lower).expect("faer evd");
        let _ = ours();
        let _ = theirs();
        let (mut to, mut tf, mut po, mut pf) = (f64::MAX, f64::MAX, 0, 0);
        for _ in 0..passes() {
            let (t, p, _) = time(ours);
            to = to.min(t);
            po = po.max(p);
            let (t, p, _) = time(theirs);
            tf = tf.min(t);
            pf = pf.max(p);
        }
        let mib = |b: usize| b as f64 / (1 << 20) as f64;
        println!(
            "{n:>6} {to:>12.4} {tf:>12.4} {:>8.3} {:>14.2} {:>14.2} {:>8.3}",
            to / tf,
            mib(po),
            mib(pf),
            po as f64 / pf.max(1) as f64
        );
    }
}
