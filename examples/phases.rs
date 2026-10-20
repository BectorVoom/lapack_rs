//! Phase timing of the host dsyevd pipeline (diagnostic).
use lapack_rs::testing as t;
use std::time::Instant;

fn main() {
    if std::env::var("COMPLEX").is_ok() {
        let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
        complex_phases(n);
        return;
    }
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let mut s = 12345u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            let v = rnd();
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
    }
    for _rep in 0..2 {
        let mut aa = a.clone();
        let (mut d, mut e, mut tau) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut work = vec![0.0; 32 * n];
        let t0 = Instant::now();
        unsafe { t::dsytrd(false, n, &mut aa, &mut d, &mut e, &mut tau, &mut work) };
        let t1 = t0.elapsed().as_secs_f64();
        let (lw, liw) = lapack_rs::stedc_work_len(n);
        let mut z = vec![0.0; n * n];
        let mut w2 = vec![0.0; lw];
        let mut iw = vec![0usize; liw];
        let t0 = Instant::now();
        unsafe { t::dstedc_i(n, &mut d, &mut e, &mut z, &mut w2, &mut iw) };
        let t2 = t0.elapsed().as_secs_f64();
        let mut w3 = vec![0.0; 32 * n + 4160];
        let t0 = Instant::now();
        unsafe { t::dormtr_ln(false, n, &mut aa, &mut tau, &mut z, &mut w3) };
        let t3 = t0.elapsed().as_secs_f64();
        println!("n={n} threads={} sytrd={t1:.4}s stedc={t2:.4}s ormtr={t3:.4}s", rayon::current_num_threads());
    }
}

#[allow(dead_code)]
pub fn complex_phases(n: usize) {
    let mut s = 777u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let (mut re, mut im) = (vec![0.0; n * n], vec![0.0; n * n]);
    for j in 0..n {
        for i in j..n {
            let (a, b) = (rnd(), if i == j { 0.0 } else { rnd() });
            re[i + j * n] = a;
            re[j + i * n] = a;
            im[i + j * n] = b;
            im[j + i * n] = -b;
        }
    }
    let (mut d, mut e) = (vec![0.0; n], vec![0.0; n]);
    let (mut tr, mut ti) = (vec![0.0; n], vec![0.0; n]);
    let mut work = vec![0.0; 2 * 32 * n];
    let t0 = Instant::now();
    unsafe { t::zhetrd(false, n, &mut re, &mut im, &mut d, &mut e, &mut tr, &mut ti, &mut work) };
    let t1 = t0.elapsed().as_secs_f64();
    let (lw, liw) = lapack_rs::stedc_work_len(n);
    let mut z = vec![0.0; n * n];
    let mut w2 = vec![0.0; lw];
    let mut iw = vec![0usize; liw];
    let t0 = Instant::now();
    unsafe { t::dstedc_i(n, &mut d, &mut e, &mut z, &mut w2, &mut iw) };
    let t2 = t0.elapsed().as_secs_f64();
    let mut zi = vec![0.0; n * n];
    let mut w3 = vec![0.0; 2 * (32 * n + 4160)];
    let t0 = Instant::now();
    unsafe { t::zunmtr_ln(false, n, &mut re, &mut im, &mut tr, &mut ti, &mut z, &mut zi, &mut w3) };
    let t3 = t0.elapsed().as_secs_f64();
    println!("complex n={n} threads={} hetrd={t1:.4}s stedc={t2:.4}s unmtr={t3:.4}s", rayon::current_num_threads());
}
