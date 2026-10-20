//! The drivers allocate nothing: with caller-supplied workspace, `dsyevd` and
//! `zheevd` perform zero heap allocations (counting global allocator).
//!
//! The calls are measured on a rayon worker thread. Called from a thread
//! outside the pool, rayon's global job injector (crossbeam `Injector`)
//! allocates one ~1.5 KiB block per 63 injected jobs; that is rayon's queue,
//! not a driver buffer, and a worker thread pushes to its own deque instead.
//! Also records the extra-memory budget.

#[path = "common/alloc.rs"]
mod alloc;
mod common;

use common::*;
use lapack_rs::{Job, Uplo, dsyevd, heevd_opt_work_len, syevd_opt_work_len, zheevd};

#[global_allocator]
static GLOBAL: alloc::Counting = alloc::Counting;

fn warm_pool() {
    use rayon::prelude::*;
    let v: Vec<u64> = (0..100_000u64).into_par_iter().map(|x| x ^ 1).collect();
    assert_eq!(v.len(), 100_000);
}

/// Allocations performed while `f` runs on a pool worker. The calling
/// thread blocks in `install` on rayon's thread-local lock latch, which does
/// not allocate once warmed.
fn allocs_on_worker(f: impl FnOnce() + Send) -> usize {
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(8).build().expect("pool"));
    let pool = POOL.get().expect("pool");
    pool.install(|| {});
    pool.install(|| {
        let before = alloc::allocs();
        f();
        alloc::allocs() - before
    })
}

static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

/// One test function on purpose: the counting allocator is process-global,
/// and a second concurrently running test (or libtest recording its result)
/// would be counted against the driver.
#[test]
fn drivers_do_not_allocate_and_budget() {
    extra_memory_budget();
    warm_pool();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(8).build().expect("pool")).install(warm_pool);
    for &n in &[1usize, 2, 30, 100, 300] {
        for job in [Job::Vectors, Job::ValuesOnly] {
            let a0 = random_sym(n, 11 + n as u64);
            let (lw, liw) = syevd_opt_work_len(n, job);
            let (mut a, mut w, mut work, mut iwork) = (a0.clone(), vec![0.0; n], vec![0.0; lw], vec![0usize; liw]);
            // Warm the dispatch path once at this size (first-use lazies).
            dsyevd(job, Uplo::Lower, n, &mut a, &mut w, &mut work, &mut iwork).expect("dsyevd");
            a.copy_from_slice(&a0);
            let count = allocs_on_worker(|| {
                dsyevd(job, Uplo::Lower, n, &mut a, &mut w, &mut work, &mut iwork).expect("dsyevd");
            });
            assert_eq!(count, 0, "dsyevd n={n} {job:?} allocated {count} times");

            let (re0, im0) = random_herm(n, 13 + n as u64);
            let (lw, lrw, liw) = heevd_opt_work_len(n, job);
            let (mut re, mut im) = (re0.clone(), im0.clone());
            let (mut w, mut work, mut rwork, mut iwork) =
                (vec![0.0; n], vec![0.0; 2 * lw], vec![0.0; lrw], vec![0usize; liw]);
            zheevd(job, Uplo::Upper, n, &mut re, &mut im, &mut w, &mut work, &mut rwork, &mut iwork).expect("zheevd");
            re.copy_from_slice(&re0);
            im.copy_from_slice(&im0);
            let count = allocs_on_worker(|| {
                zheevd(job, Uplo::Upper, n, &mut re, &mut im, &mut w, &mut work, &mut rwork, &mut iwork)
                    .expect("zheevd");
            });
            assert_eq!(count, 0, "zheevd n={n} {job:?} allocated {count} times");
        }
    }
}

fn extra_memory_budget() {
    // Beyond the caller's n^2 input (reals).
    for &n in &[64usize, 1000] {
        let nn = n * n;
        let (lw, liw) = lapack_rs::syevd_min_work_len(n, Job::Vectors);
        let extra = lw + liw.div_ceil(8) + n;
        assert!(extra <= 2 * nn + 12 * n, "dsyevd Vectors extra {extra} > 2n^2 + O(n)");
        let (lw, liw) = lapack_rs::syevd_min_work_len(n, Job::ValuesOnly);
        assert!(lw + liw <= 4 * n, "dsyevd ValuesOnly workspace must be O(n)");
        let (lw, lrw, liw) = lapack_rs::heevd_min_work_len(n, Job::Vectors);
        let extra_reals = 2 * lw + lrw + liw.div_ceil(8) + n;
        // Reference-LAPACK's own minimum is 4n^2 + O(n) reals for ZHEEVD
        // (complex Z plus the real D&C Z and its workspace); the bit-exact
        // LWORK semantics keep it. Recorded, not the plan's 3n^2 target.
        assert!(extra_reals <= 4 * nn + 16 * n, "zheevd Vectors extra {extra_reals} > 4n^2 + O(n)");
        let (lw, lrw, liw) = lapack_rs::heevd_min_work_len(n, Job::ValuesOnly);
        assert!(2 * lw + lrw + liw <= 4 * n + 4, "zheevd ValuesOnly workspace must be O(n)");
    }
}
