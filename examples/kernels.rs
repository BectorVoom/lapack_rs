//! Kernel micro-timings (diagnostic).
use lapack_rs::testing as t;
use std::time::Instant;

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let mut s = 99u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut a: Vec<f64> = (0..n * n).map(|_| rnd()).collect();
    let mut x: Vec<f64> = (0..n).map(|_| rnd()).collect();
    let mut y = vec![0.0; n];
    let reps = 20;
    let t0 = Instant::now();
    for _ in 0..reps {
        unsafe { t::dsymv(false, n, 1.0, &mut a, n, &mut x, 0.0, &mut y) };
    }
    println!("dsymv n={n}: {:.3} ms/call", t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
    let k = 32;
    let mut v: Vec<f64> = (0..n * k).map(|_| rnd()).collect();
    let mut w: Vec<f64> = (0..n * k).map(|_| rnd()).collect();
    let t0 = Instant::now();
    unsafe { t::dsyr2k_n(false, n, k, -1.0, &mut v, n, &mut w, n, 1.0, &mut a, n) };
    println!("dsyr2k n={n} k={k}: {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);
    let mut c = vec![0.0; n * k];
    let t0 = Instant::now();
    unsafe { t::dgemm(true, false, n, k, n, 1.0, &mut a, n, &mut v, n, 1.0, &mut c, n) };
    println!("dgemm T,N {n}x{k}x{n}: {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);
    let mut c2 = a.clone();
    let t0 = Instant::now();
    unsafe { t::dgemm(false, true, n, n, k, -1.0, &mut v, n, &mut w, n, 1.0, &mut c2, n) };
    println!("dgemm N,T {n}x{n}x{k}: {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);
    let mut yy = vec![0.0; k];
    let t0 = Instant::now();
    for _ in 0..reps {
        unsafe { t::dgemv(true, n, k, 1.0, &mut v, n, &mut x, 1, 0.0, &mut yy) };
    }
    println!("dgemv T {n}x{k}: {:.3} ms/call", t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
}
