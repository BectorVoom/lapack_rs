//! **OpenBLAS track.** Phase timing of `DSYEVD` in the OpenBLAS flavour: the
//! host transcription (`lapack_rs::testing::openblas`) against scipy's
//! OpenBLAS itself (one thread), per phase and whole. Diagnostic only.
//!
//! `OPENBLAS_CORETYPE=SkylakeX OPENBLAS_NUM_THREADS=1 LAPACK_RS_OPENBLAS_DIR=...
//! cargo run --release --example openblas_phases -- 256 512 1024`
use lapack_rs::testing::openblas as tob;
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

fn main() {
    assert!(ob::require("openblas_phases"));
    let ns: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    for n in if ns.is_empty() { vec![512] } else { ns } {
        let a = sym(n);
        let n_ = n as i32;
        let lw_trd = 32 * n;
        let (lwd, liwd) = lapack_rs::stedc_work_len(n);
        let lw_orm = 1 + 4 * n + n * n;
        let mut best = [f64::MAX; 6];
        for _ in 0..2 {
            let mut aa = a.clone();
            let (mut d, mut e, mut tau, mut work) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; lw_trd]);
            let t0 = Instant::now();
            unsafe { tob::dsytrd(false, n, &mut aa, &mut d, &mut e, &mut tau, &mut work) };
            best[0] = best[0].min(t0.elapsed().as_secs_f64());
            let (mut z, mut w2, mut iw) = (vec![0.0; n * n], vec![0.0; lwd], vec![0usize; liwd]);
            let (d0, e0) = (d.clone(), e.clone());
            let t0 = Instant::now();
            unsafe { tob::dstedc_i(n, &mut d, &mut e, &mut z, &mut w2, &mut iw) };
            best[1] = best[1].min(t0.elapsed().as_secs_f64());
            let z0 = z.clone();
            let mut w3 = vec![0.0; lw_orm];
            let t0 = Instant::now();
            unsafe { tob::dormtr_ln(false, n, &mut aa.clone(), &mut tau.clone(), &mut z, &mut w3) };
            best[2] = best[2].min(t0.elapsed().as_secs_f64());

            // scipy's OpenBLAS, same inputs.
            let mut ab = a.clone();
            let (mut db, mut eb, mut taub, mut workb) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; lw_trd]);
            let (lw, mut info) = (lw_trd as i32, 0);
            let t0 = Instant::now();
            unsafe {
                ob::dsytrd(
                    &b'L',
                    &n_,
                    ab.as_mut_ptr(),
                    &n_,
                    db.as_mut_ptr(),
                    eb.as_mut_ptr(),
                    taub.as_mut_ptr(),
                    workb.as_mut_ptr(),
                    &lw,
                    &mut info,
                    1,
                )
            };
            best[3] = best[3].min(t0.elapsed().as_secs_f64());
            let (mut dd, mut ee, mut zz, mut ww, mut ii) =
                (d0.clone(), e0.clone(), vec![0.0; n * n], vec![0.0; lwd], vec![0i32; liwd]);
            let (lwd_, liwd_) = (lwd as i32, liwd as i32);
            let t0 = Instant::now();
            unsafe {
                ob::dstedc(
                    &b'I',
                    &n_,
                    dd.as_mut_ptr(),
                    ee.as_mut_ptr(),
                    zz.as_mut_ptr(),
                    &n_,
                    ww.as_mut_ptr(),
                    &lwd_,
                    ii.as_mut_ptr(),
                    &liwd_,
                    &mut info,
                    1,
                )
            };
            best[4] = best[4].min(t0.elapsed().as_secs_f64());
            let (mut zc, mut w4, lwo) = (z0.clone(), vec![0.0; lw_orm], lw_orm as i32);
            let t0 = Instant::now();
            unsafe {
                ob::dormtr(
                    &b'L',
                    &b'L',
                    &b'N',
                    &n_,
                    &n_,
                    aa.as_mut_ptr(),
                    &n_,
                    tau.as_mut_ptr(),
                    zc.as_mut_ptr(),
                    &n_,
                    w4.as_mut_ptr(),
                    &lwo,
                    &mut info,
                    1,
                    1,
                    1,
                )
            };
            best[5] = best[5].min(t0.elapsed().as_secs_f64());
        }
        println!(
            "n={n:5}  ours: sytrd {:.4}s stedc {:.4}s ormtr {:.4}s | openblas: sytrd {:.4}s stedc {:.4}s ormtr {:.4}s | ratio {:.2}x",
            best[0],
            best[1],
            best[2],
            best[3],
            best[4],
            best[5],
            (best[0] + best[1] + best[2]) / (best[3] + best[4] + best[5])
        );
    }
}
