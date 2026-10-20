//! **Task O0** of `docs/openblas-bitexact-plan.md`: measure, before building
//! anything, how far this crate's Reference-LAPACK transcription is from
//! scipy's OpenBLAS 0.3.30 (`crates/openblas_ref`), and which BLAS routines
//! the gap comes from.
//!
//! **O0 reports; it does not gate.** Two tables:
//!
//! 1. whole `DSYEVD` / `ZHEEVD` (`JOBZ = 'V'`, both `UPLO`, the `LWORK` query
//!    value of OpenBLAS itself, which is what `scipy.linalg.eigh` passes):
//!    lapack_rs's host path vs OpenBLAS — % bit-identical `w` and `Z`, max ULP;
//! 2. every BLAS routine `DSYEVD` reaches, on the shapes it calls them with:
//!    Reference BLAS (`crates/lapack_ref`) vs OpenBLAS — which kernels change
//!    bits at all.
//!
//! Run with `OPENBLAS_CORETYPE=SkylakeX OPENBLAS_NUM_THREADS=1` and both
//! `LAPACK_RS_OPENBLAS_DIR` and `LAPACK_RS_REF_DIR`:
//! `cargo test --release --test openblas_o0_differential -- --nocapture --test-threads=1`.

mod common;
use common::*;
use lapack_rs::{Job, Uplo, dsyevd, zheevd};
use openblas_ref as ob;

fn ulps(a: f64, b: f64) -> u64 {
    if a.to_bits() == b.to_bits() {
        return 0;
    }
    if !a.is_finite() || !b.is_finite() {
        return u64::MAX;
    }
    let key = |x: f64| -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    (key(a) - key(b)).unsigned_abs()
}

#[derive(Default)]
struct Cmp {
    total: usize,
    same: usize,
    max_ulp: u64,
    max_abs: f64,
}

impl Cmp {
    fn add(&mut self, got: &[f64], want: &[f64]) {
        for (g, w) in got.iter().zip(want) {
            self.total += 1;
            let u = ulps(*g, *w);
            if u == 0 {
                self.same += 1;
            }
            self.max_ulp = self.max_ulp.max(u);
            self.max_abs = self.max_abs.max((g - w).abs());
        }
    }
    fn pct(&self) -> f64 {
        if self.total == 0 { 100.0 } else { 100.0 * self.same as f64 / self.total as f64 }
    }
}

fn ob_dsyevd(uplo: char, n: usize, a: &mut [f64], w: &mut [f64]) -> (i32, usize) {
    let (jz, ul) = (ob::ch('V'), ob::ch(uplo));
    let n_ = n as i32;
    let lda = n.max(1) as i32;
    let mut info = 0;
    let (mut wq, mut iwq) = (0.0f64, 0i32);
    let (m1, dummy) = (-1i32, 0.0f64);
    let mut dw = dummy;
    // SAFETY: workspace query.
    unsafe {
        ob::dsyevd(&jz, &ul, &n_, a.as_mut_ptr(), &lda, &mut dw, &mut wq, &m1, &mut iwq, &m1, &mut info, 1, 1);
    }
    let lwork = wq as usize;
    let mut work = vec![0.0; lwork.max(1)];
    let mut iwork = vec![0i32; (iwq as usize).max(1)];
    let (lw, liw) = (lwork as i32, iwq);
    // SAFETY: lengths from the query.
    unsafe {
        ob::dsyevd(
            &jz,
            &ul,
            &n_,
            a.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lw,
            iwork.as_mut_ptr(),
            &liw,
            &mut info,
            1,
            1,
        );
    }
    (info, lwork)
}

fn ob_zheevd(uplo: char, n: usize, a: &mut [ob::C64], w: &mut [f64]) -> (i32, usize, usize, usize) {
    let (jz, ul) = (ob::ch('V'), ob::ch(uplo));
    let n_ = n as i32;
    let lda = n.max(1) as i32;
    let mut info = 0;
    let (mut wq, mut rwq, mut iwq) = (ob::C64::default(), 0.0f64, 0i32);
    let m1 = -1i32;
    let mut dw = 0.0f64;
    // SAFETY: workspace query.
    unsafe {
        ob::zheevd(
            &jz,
            &ul,
            &n_,
            a.as_mut_ptr(),
            &lda,
            &mut dw,
            &mut wq,
            &m1,
            &mut rwq,
            &m1,
            &mut iwq,
            &m1,
            &mut info,
            1,
            1,
        );
    }
    let (lw, lrw, liw) = (wq.re as usize, rwq as usize, iwq as usize);
    let mut work = vec![ob::C64::default(); lw.max(1)];
    let mut rwork = vec![0.0; lrw.max(1)];
    let mut iwork = vec![0i32; liw.max(1)];
    let (lw_, lrw_, liw_) = (lw as i32, lrw as i32, liw as i32);
    // SAFETY: lengths from the query.
    unsafe {
        ob::zheevd(
            &jz,
            &ul,
            &n_,
            a.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lw_,
            rwork.as_mut_ptr(),
            &lrw_,
            iwork.as_mut_ptr(),
            &liw_,
            &mut info,
            1,
            1,
        );
    }
    (info, lw, lrw, liw)
}

const NS: &[usize] = &[1, 2, 3, 4, 5, 8, 10, 16, 17, 25, 26, 31, 32, 33, 48, 63, 64, 65, 100, 128, 129, 200, 257];

#[test]
fn o0_whole_drivers() {
    if !ob::require("o0_whole_drivers") {
        return;
    }
    println!("OpenBLAS: {:?} ({:?})", ob::configuration(), ob::library_path());
    println!(
        "{:>5} {:>4} {:>8} {:>9} {:>12} {:>11} {:>9} {:>12} {:>11}",
        "n", "kind", "lwork", "w same%", "w maxULP", "w max|d|", "Z same%", "Z maxULP", "Z max|d|"
    );
    for &n in NS {
        for uplo in ['L', 'U'] {
            let a0 = random_sym(n, 4000 + n as u64);
            let mut ao = a0.clone();
            let mut wo = vec![0.0; n];
            let (info, lwork) = ob_dsyevd(uplo, n, &mut ao, &mut wo);
            assert_eq!(info, 0);
            let mut a = a0.clone();
            let mut w = vec![0.0; n];
            let mut work = vec![0.0; lwork.max(1)];
            let (_, liw) = lapack_rs::syevd_opt_work_len(n, Job::Vectors);
            let mut iwork = vec![0usize; liw.max(1)];
            let u = if uplo == 'L' { Uplo::Lower } else { Uplo::Upper };
            dsyevd(Job::Vectors, u, n, &mut a, &mut w, &mut work, &mut iwork).expect("dsyevd");
            let (mut cw, mut cz) = (Cmp::default(), Cmp::default());
            cw.add(&w, &wo);
            cz.add(&a, &ao);
            println!(
                "{n:>5} {:>4} {lwork:>8} {:>8.1}% {:>12} {:>11.2e} {:>8.1}% {:>12} {:>11.2e}",
                format!("d{uplo}"),
                cw.pct(),
                cw.max_ulp,
                cw.max_abs,
                cz.pct(),
                cz.max_ulp,
                cz.max_abs
            );
        }
    }
    for &n in NS {
        for uplo in ['L', 'U'] {
            let (re, im) = random_herm(n, 5000 + n as u64);
            let mut ao: Vec<ob::C64> = re.iter().zip(&im).map(|(&re, &im)| ob::C64 { re, im }).collect();
            let mut wo = vec![0.0; n];
            let (info, lw, lrw, liw) = ob_zheevd(uplo, n, &mut ao, &mut wo);
            assert_eq!(info, 0);
            let (mut ar, mut ai) = (re.clone(), im.clone());
            let mut w = vec![0.0; n];
            let mut work = vec![0.0; 2 * lw.max(1)];
            let mut rwork = vec![0.0; lrw.max(1)];
            let mut iwork = vec![0usize; liw.max(1)];
            let u = if uplo == 'L' { Uplo::Lower } else { Uplo::Upper };
            zheevd(Job::Vectors, u, n, &mut ar, &mut ai, &mut w, &mut work, &mut rwork, &mut iwork).expect("zheevd");
            let zo_re: Vec<f64> = ao.iter().map(|c| c.re).collect();
            let zo_im: Vec<f64> = ao.iter().map(|c| c.im).collect();
            let (mut cw, mut cz) = (Cmp::default(), Cmp::default());
            cw.add(&w, &wo);
            cz.add(&ar, &zo_re);
            cz.add(&ai, &zo_im);
            println!(
                "{n:>5} {:>4} {lw:>8} {:>8.1}% {:>12} {:>11.2e} {:>8.1}% {:>12} {:>11.2e}",
                format!("z{uplo}"),
                cw.pct(),
                cw.max_ulp,
                cw.max_abs,
                cz.pct(),
                cz.max_ulp,
                cz.max_abs
            );
        }
    }
}

fn vecr(n: usize, seed: u64) -> Vec<f64> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.uniform()).collect()
}

fn row(name: &str, shape: &str, c: &Cmp) {
    println!("{name:>10} {shape:>18} {:>8.1}% {:>12} {:>11.2e}", c.pct(), c.max_ulp, c.max_abs);
}

#[test]
fn o0_blas_components() {
    if !ob::require("o0_blas_components") || !lapack_ref::require("o0_blas_components") {
        return;
    }
    let one = 1i32;
    println!("{:>10} {:>18} {:>9} {:>12} {:>11}", "routine", "shape", "same%", "maxULP", "max|d|");
    for &n in &[1usize, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 129, 257] {
        let x = vecr(n, 1 + n as u64);
        let y = vecr(n, 2 + n as u64);
        let a = random_sym(n, 3 + n as u64);
        let n_ = n as i32;

        // ddot
        let mut c = Cmp::default();
        // SAFETY: n entries each.
        let d_ob = unsafe { ob::ddot(&n_, x.as_ptr(), &one, y.as_ptr(), &one) };
        c.add(&[lapack_ref::ddot(n, &x, 1, &y, 1)], &[d_ob]);
        row("ddot", &format!("n={n}"), &c);

        // dnrm2
        let mut c = Cmp::default();
        // SAFETY: n entries.
        let v_ob = unsafe { ob::dnrm2(&n_, x.as_ptr(), &one) };
        c.add(&[lapack_ref::dnrm2(&x)], &[v_ob]);
        row("dnrm2", &format!("n={n}"), &c);

        // dsymv (alpha = 1, beta = 0 as in DLATRD; and alpha = tau as in DSYTD2)
        for (uplo, alpha) in [('L', 1.0), ('U', 1.0), ('L', 0.37), ('U', -0.61)] {
            let mut yr = vec![0.0; n];
            lapack_ref::dsymv(uplo, n, alpha, &a, n, &x, 0.0, &mut yr);
            let mut yo = vec![0.0; n];
            let (u, beta) = (ob::ch(uplo), 0.0f64);
            // SAFETY: n×n A, n-vectors.
            unsafe { ob::dsymv(&u, &n_, &alpha, a.as_ptr(), &n_, x.as_ptr(), &one, &beta, yo.as_mut_ptr(), &one, 1) };
            let mut c = Cmp::default();
            c.add(&yr, &yo);
            row("dsymv", &format!("{uplo} a={alpha} n={n}"), &c);
        }

        // dgemv (DLATRD: y = -1·A·x + 1·y, TRANS='N' on (n-i)×(i-1); and 'T')
        for t in ['N', 'T'] {
            let k = (n / 2).max(1);
            let g = vecr(n * k, 9 + n as u64);
            let xs = vecr(if t == 'N' { k } else { n }, 11 + n as u64);
            let ylen = if t == 'N' { n } else { k };
            let mut yr = vecr(ylen, 12 + n as u64);
            let mut yo = yr.clone();
            lapack_ref::dgemv(t, n, k, -1.0, &g, n, &xs, 1, 1.0, &mut yr, 1);
            let (tt, k_, al, be) = (ob::ch(t), k as i32, -1.0f64, 1.0f64);
            // SAFETY: n×k A.
            unsafe { ob::dgemv(&tt, &n_, &k_, &al, g.as_ptr(), &n_, xs.as_ptr(), &one, &be, yo.as_mut_ptr(), &one, 1) };
            let mut c = Cmp::default();
            c.add(&yr, &yo);
            row("dgemv", &format!("{t} {n}x{k}"), &c);
        }

        // daxpy / dscal
        {
            let alpha = -0.4321;
            let mut yr = y.clone();
            let mut yo = y.clone();
            for i in 0..n {
                yr[i] += alpha * x[i];
            }
            // SAFETY: n entries.
            unsafe { ob::daxpy(&n_, &alpha, x.as_ptr(), &one, yo.as_mut_ptr(), &one) };
            let mut c = Cmp::default();
            c.add(&yr, &yo);
            row("daxpy", &format!("n={n}"), &c);
            let mut xr = x.clone();
            let mut xo = x.clone();
            for v in xr.iter_mut() {
                *v *= alpha;
            }
            // SAFETY: n entries.
            unsafe { ob::dscal(&n_, &alpha, xo.as_mut_ptr(), &one) };
            let mut c = Cmp::default();
            c.add(&xr, &xo);
            row("dscal", &format!("n={n}"), &c);
        }

        // dsyr2 (DSYTD2)
        for uplo in ['L', 'U'] {
            let mut ar = a.clone();
            let mut ao = a.clone();
            lapack_ref::dsyr2(uplo, n, -1.0, &x, &y, &mut ar, n);
            let (u, al) = (ob::ch(uplo), -1.0f64);
            // SAFETY: n×n A.
            unsafe { ob::dsyr2(&u, &n_, &al, x.as_ptr(), &one, y.as_ptr(), &one, ao.as_mut_ptr(), &n_, 1) };
            let mut c = Cmp::default();
            c.add(&ar, &ao);
            row("dsyr2", &format!("{uplo} n={n}"), &c);
        }

        // dsyr2k (DSYTRD: alpha = -1, beta = 1, TRANS='N', k = NB)
        for uplo in ['L', 'U'] {
            let k = n.min(32);
            let v = vecr(n * k, 21 + n as u64);
            let w = vecr(n * k, 22 + n as u64);
            let mut ar = a.clone();
            let mut ao = a.clone();
            lapack_ref::dsyr2k(uplo, 'N', n, k, -1.0, &v, n, &w, n, 1.0, &mut ar, n);
            let (u, t, k_, al, be) = (ob::ch(uplo), ob::ch('N'), k as i32, -1.0f64, 1.0f64);
            // SAFETY: n×k V/W, n×n C.
            unsafe {
                ob::dsyr2k(&u, &t, &n_, &k_, &al, v.as_ptr(), &n_, w.as_ptr(), &n_, &be, ao.as_mut_ptr(), &n_, 1, 1)
            };
            let mut c = Cmp::default();
            c.add(&ar, &ao);
            row("dsyr2k", &format!("{uplo} n={n} k={k}"), &c);
        }

        // dgemm (DLAED3/DLAED1 merges and DLARFB: alpha = ±1, beta = 0/1)
        for (ta, tb, alpha, beta) in [('N', 'N', 1.0, 0.0), ('N', 'T', -1.0, 1.0), ('T', 'N', 1.0, 0.0)] {
            let k = (n / 3).max(1);
            let (ar_, ac, br, bc) = match (ta, tb) {
                ('N', 'N') => (n, k, k, n),
                ('N', 'T') => (n, k, n, k),
                _ => (k, n, k, n),
            };
            let ga = vecr(ar_ * ac, 31 + n as u64);
            let gb = vecr(br * bc, 32 + n as u64);
            let (m, nn) = match (ta, tb) {
                ('N', 'N') => (n, n),
                ('N', 'T') => (n, n),
                _ => (n, n),
            };
            let mut cr = vecr(m * nn, 33 + n as u64);
            let mut co = cr.clone();
            lapack_ref::dgemm(ta, tb, m, nn, k, alpha, &ga, ar_, &gb, br, beta, &mut cr, m);
            let (ta_, tb_, m_, nn_, k_, lda, ldb) =
                (ob::ch(ta), ob::ch(tb), m as i32, nn as i32, k as i32, ar_ as i32, br as i32);
            // SAFETY: shapes as above.
            unsafe {
                ob::dgemm(
                    &ta_,
                    &tb_,
                    &m_,
                    &nn_,
                    &k_,
                    &alpha,
                    ga.as_ptr(),
                    &lda,
                    gb.as_ptr(),
                    &ldb,
                    &beta,
                    co.as_mut_ptr(),
                    &m_,
                    1,
                    1,
                )
            };
            let mut c = Cmp::default();
            c.add(&cr, &co);
            row("dgemm", &format!("{ta}{tb} {m}x{nn} k={k}"), &c);
        }

        // dtrmm (DLARFB: 'R','L'/'U','N'/'T','U')
        for (side, uplo, ta) in [('R', 'L', 'N'), ('R', 'U', 'N'), ('R', 'L', 'T'), ('R', 'U', 'T')] {
            let k = (n / 2).max(1);
            let t = vecr(k * k, 41 + n as u64);
            let mut br = vecr(n * k, 42 + n as u64);
            let mut bo = br.clone();
            lapack_ref::dtrmm(side, uplo, ta, 'N', n, k, 1.0, &t, k, &mut br, n);
            let (s, u, t_, d, m_, k_, al) =
                (ob::ch(side), ob::ch(uplo), ob::ch(ta), ob::ch('N'), n as i32, k as i32, 1.0f64);
            // SAFETY: k×k T, n×k B.
            unsafe { ob::dtrmm(&s, &u, &t_, &d, &m_, &k_, &al, t.as_ptr(), &k_, bo.as_mut_ptr(), &m_, 1, 1, 1, 1) };
            let mut c = Cmp::default();
            c.add(&br, &bo);
            row("dtrmm", &format!("{side}{uplo}{ta}N {n}x{k}"), &c);
        }

        // dlarfg
        {
            let mut xr = x.clone();
            let mut xo = x.clone();
            let alpha = 0.3;
            let (ar, tr) = lapack_ref::dlarfg(n + 1, alpha, &mut xr);
            let (mut al, mut tau, np1) = (alpha, 0.0f64, (n + 1) as i32);
            // SAFETY: n entries.
            unsafe { ob::dlarfg(&np1, &mut al, xo.as_mut_ptr(), &one, &mut tau) };
            let mut c = Cmp::default();
            c.add(&[ar, tr], &[al, tau]);
            c.add(&xr, &xo);
            row("dlarfg", &format!("n={}", n + 1), &c);
        }
    }
}
