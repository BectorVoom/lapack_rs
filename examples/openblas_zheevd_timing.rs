//! **OpenBLAS track.** `zheevd` timing: `zheevd_alloc` (Reference, device),
//! `openblas::zheevd_host`, `openblas::zheevd` (when present) and scipy's
//! OpenBLAS `zheevd` (one thread). Diagnostic only.
use openblas_ref::{self as ob, C64};
use std::time::Instant;

fn herm(n: usize) -> (Vec<f64>, Vec<f64>) {
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
            im[i + j * n] = b;
            re[j + i * n] = a;
            im[j + i * n] = -b;
        }
    }
    (re, im)
}

fn best(p: usize, mut f: impl FnMut()) -> f64 {
    (0..p)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::MAX, f64::min)
}

fn main() {
    if std::env::var("PHASES").is_ok() {
        for n in std::env::args().skip(1).filter_map(|s| s.parse().ok()) {
            phases(n);
        }
        return;
    }
    let ob_ok = ob::require("openblas_zheevd_timing");
    let ns: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let (job, lo) = (lapack_rs::Job::Vectors, lapack_rs::Uplo::Lower);
    for n in if ns.is_empty() { vec![256, 512] } else { ns } {
        let (re, im) = herm(n);
        let run_ref = || {
            let (mut a, mut b, mut w) = (re.clone(), im.clone(), vec![0.0; n]);
            lapack_rs::zheevd_alloc(job, lo, n, &mut a, &mut b, &mut w).expect("ref");
        };
        run_ref();
        let t_ref = best(3, run_ref);
        let t_host = best(1, || {
            let (mut a, mut b, mut w) = (re.clone(), im.clone(), vec![0.0; n]);
            lapack_rs::openblas::zheevd_host(job, lo, n, &mut a, &mut b, &mut w).expect("ob host");
        });
        let run_obdev = || {
            let (mut a, mut b, mut w) = (re.clone(), im.clone(), vec![0.0; n]);
            lapack_rs::openblas::zheevd(job, lo, n, &mut a, &mut b, &mut w).expect("ob dev");
        };
        run_obdev();
        let t_obdev = best(3, run_obdev);
        let t_scipy = if ob_ok {
            let (lw, lrw, liw) = lapack_rs::heevd_work_len(n, job);
            best(3, || {
                let mut a: Vec<C64> = re.iter().zip(&im).map(|(&re, &im)| C64 { re, im }).collect();
                let (mut w, mut wk, mut rw, mut iw) =
                    (vec![0.0; n], vec![C64::default(); lw], vec![0.0; lrw], vec![0i32; liw]);
                let (n_, lw_, lrw_, liw_, mut info) = (n as i32, lw as i32, lrw as i32, liw as i32, 0);
                unsafe {
                    ob::zheevd(
                        &b'V',
                        &b'L',
                        &n_,
                        a.as_mut_ptr(),
                        &n_,
                        w.as_mut_ptr(),
                        wk.as_mut_ptr(),
                        &lw_,
                        rw.as_mut_ptr(),
                        &lrw_,
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
            "n={n:5} ref-device {t_ref:.4}s | ob-device {t_obdev:.4}s | ob-host {t_host:.4}s | scipy-openblas(1t) {t_scipy:.4}s"
        );
    }
}

/// Host phase split of the OpenBLAS-flavour `ZHEEVD` pieces (`PHASES=1`).
#[allow(dead_code)]
pub fn phases(n: usize) {
    use lapack_rs::testing::openblas as tob;
    let (re, im) = herm(n);
    let (mut ar, mut ai) = (re.clone(), im.clone());
    let (mut d, mut e, mut tr, mut ti) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut work = vec![0.0; 2 * (n + 32 * n)];
    let t = Instant::now();
    unsafe { tob::zhetrd(false, n, &mut ar, &mut ai, &mut d, &mut e, &mut tr, &mut ti, &mut work) };
    let t_trd = t.elapsed().as_secs_f64();
    let (lw, liw) = lapack_rs::stedc_work_len(n);
    let (mut z, mut w2, mut iw) = (vec![0.0; n * n], vec![0.0; lw], vec![0usize; liw]);
    let t = Instant::now();
    unsafe { tob::dstedc_i(n, &mut d, &mut e, &mut z, &mut w2, &mut iw) };
    let t_dc = t.elapsed().as_secs_f64();
    let (mut cr, mut ci) = (z.clone(), vec![0.0; n * n]);
    let mut w3 = vec![0.0; 2 * n];
    let t = Instant::now();
    unsafe { tob::zunmtr_ln(false, n, &mut ar, &mut ai, &mut tr, &mut ti, &mut cr, &mut ci, &mut w3) };
    let t_orm = t.elapsed().as_secs_f64();
    println!("n={n} openblas host: zhetrd {t_trd:.3}s dstedc {t_dc:.3}s zunmtr(unblocked) {t_orm:.3}s");
}
