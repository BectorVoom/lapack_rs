//! Whole-solve timing: `dsyevd_alloc` (Reference track, CubeCL runtime),
//! `openblas::dsyevd_host` and `openblas::dsyevd` (OpenBLAS track), and
//! scipy's OpenBLAS `dsyevd` (one thread). Diagnostic only.
use openblas_ref as ob;
use std::time::Instant;

fn sym(n: usize) -> Vec<f64> {
    let mut s = 12345u64;
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
    }
    a
}

fn best(passes: usize, mut f: impl FnMut()) -> f64 {
    let mut b = f64::MAX;
    for _ in 0..passes {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn main() {
    let ob_ok = ob::require("openblas_vs_device");
    let host = std::env::var("HOST").is_ok();
    let ns: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    for n in if ns.is_empty() { vec![256, 512, 1024] } else { ns } {
        let a = sym(n);
        let job = lapack_rs::Job::Vectors;
        let lo = lapack_rs::Uplo::Lower;
        // warm-up (JIT)
        let (mut x, mut w) = (a.clone(), vec![0.0; n]);
        lapack_rs::dsyevd_alloc(job, lo, n, &mut x, &mut w).expect("ref");
        let t_ref = best(3, || {
            let (mut x, mut w) = (a.clone(), vec![0.0; n]);
            lapack_rs::dsyevd_alloc(job, lo, n, &mut x, &mut w).expect("ref");
        });
        let t_obdev = {
            let (mut x, mut w) = (a.clone(), vec![0.0; n]);
            lapack_rs::openblas::dsyevd(job, lo, n, &mut x, &mut w).expect("ob dev");
            best(3, || {
                let (mut x, mut w) = (a.clone(), vec![0.0; n]);
                lapack_rs::openblas::dsyevd(job, lo, n, &mut x, &mut w).expect("ob dev");
            })
        };
        let t_obhost = if host {
            best(1, || {
                let (mut x, mut w) = (a.clone(), vec![0.0; n]);
                lapack_rs::openblas::dsyevd_host(job, lo, n, &mut x, &mut w).expect("ob host");
            })
        } else {
            f64::NAN
        };
        let t_scipy = if ob_ok {
            let (lw, liw) = lapack_rs::syevd_work_len(n, job);
            best(3, || {
                let (mut x, mut w, mut wk, mut iw) = (a.clone(), vec![0.0; n], vec![0.0; lw], vec![0i32; liw]);
                let (n_, lw_, liw_, mut info) = (n as i32, lw as i32, liw as i32, 0);
                unsafe {
                    ob::dsyevd(
                        &b'V',
                        &b'L',
                        &n_,
                        x.as_mut_ptr(),
                        &n_,
                        w.as_mut_ptr(),
                        wk.as_mut_ptr(),
                        &lw_,
                        iw.as_mut_ptr(),
                        &liw_,
                        &mut info,
                        1,
                        1,
                    )
                };
            })
        } else {
            f64::NAN
        };
        println!(
            "n={n:5} ref-device {t_ref:.4}s | ob-device {t_obdev:.4}s | ob-host {t_obhost:.4}s | scipy-openblas(1t) {t_scipy:.4}s"
        );
    }
}
