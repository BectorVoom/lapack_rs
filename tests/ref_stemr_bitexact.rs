//! `DSTEMR` (MRRR) against Reference-LAPACK 3.12.1, bit for bit: eigenvalues,
//! eigenvectors, `ISUPPZ`, `M`, `TRYRAC` and `INFO`, for `RANGE = 'A'` and
//! both `JOBZ` values.
//!
//! A mismatch here is bisected to one routine with
//! `tests/ref_mrrr_components.rs`.

mod common;
use common::*;
use lapack_rs::{Job, blocks, dstemr, dstemr_min_work_len, dstemr_work_len};

/// The tridiagonal fixture set: the shared `tridiag_fixtures` plus the MRRR
/// stress cases (split blocks, tight clusters, graded and scaled spectra).
fn fixtures(n: usize) -> Vec<(String, Vec<f64>, Vec<f64>)> {
    let mut v: Vec<(String, Vec<f64>, Vec<f64>)> =
        tridiag_fixtures(n).into_iter().map(|(name, d, e)| (name.to_string(), d, e)).collect();
    let m1 = n.max(2) - 1;
    let mut r = Rng::new(20_000 + n as u64);

    // Split into blocks: every third off-diagonal is exactly zero.
    let d: Vec<f64> = (0..n).map(|_| r.uniform()).collect();
    let e: Vec<f64> = (0..m1).map(|i| if i % 3 == 2 { 0.0 } else { r.uniform() }).collect();
    v.push(("split3".into(), d, e));

    // Tight clusters of nearly equal eigenvalues.
    let d: Vec<f64> = (0..n).map(|i| 1.0 + 1e-13 * (i / 3) as f64).collect();
    let e: Vec<f64> = (0..m1).map(|_| 1e-14).collect();
    v.push(("cluster".into(), d, e));

    // Graded spectrum over many exponents (exercises the DSTEMR scaling).
    let d: Vec<f64> = (0..n).map(|i| 10f64.powi(-(i as i32 % 30))).collect();
    let e: Vec<f64> = (0..m1).map(|i| 10f64.powi(-(i as i32 % 30) - 1)).collect();
    v.push(("graded".into(), d, e));

    // Denormal-adjacent and huge scalings: DSTEMR's RMIN/RMAX paths.
    let d: Vec<f64> = (0..n).map(|_| r.uniform() * 1e-300).collect();
    let e: Vec<f64> = (0..m1).map(|_| r.uniform() * 1e-300).collect();
    v.push(("tiny".into(), d, e));
    let d: Vec<f64> = (0..n).map(|_| r.uniform() * 1e300).collect();
    let e: Vec<f64> = (0..m1).map(|_| r.uniform() * 1e300).collect();
    v.push(("huge".into(), d, e));

    // Glued matrices: two identical blocks joined by a tiny off-diagonal.
    let d: Vec<f64> = (0..n).map(|i| (i % (n / 2).max(1)) as f64).collect();
    let e: Vec<f64> = (0..m1).map(|i| if i + 1 == n / 2 { 1e-300 } else { 1.0 }).collect();
    v.push(("glued".into(), d, e));

    // Wilkinson W+: the classic pathologically-close eigenvalue pairs.
    let half = (n / 2) as i64;
    let d: Vec<f64> = (0..n).map(|i| (i as i64 - half).abs() as f64).collect();
    let e: Vec<f64> = (0..m1).map(|_| 1.0).collect();
    v.push(("wilkinson".into(), d, e));

    // One long cluster: identical diagonal, off-diagonal at the bisection
    // floor, which is what drives DLARRV's representation tree deepest.
    let d: Vec<f64> = (0..n).map(|_| 1.0).collect();
    let e: Vec<f64> = (0..m1).map(|_| 1e-15).collect();
    v.push(("one_cluster".into(), d, e));

    // Element growth: a near-singular L D L^T that pushes DLARRE through its
    // MAXTRY shift retries (and, on some n, into its INFO paths).
    let d: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1e-14 } else { 1e14 }).collect();
    let e: Vec<f64> = (0..m1).map(|_| 1.0).collect();
    v.push(("growth".into(), d, e));

    // Alternating sign and magnitude across the whole double range.
    let d: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1e-290 } else { -1e290 }).collect();
    let e: Vec<f64> = (0..m1).map(|i| if i % 3 == 0 { 1e-290 } else { 1e150 }).collect();
    v.push(("straddle".into(), d, e));

    v
}

#[test]
fn stemr_values_and_vectors() {
    if !lapack_ref::require("stemr_values_and_vectors") {
        return;
    }
    for &n in &[0usize, 1, 2, 3, 4, 5, 13, 16, 17, 31, 32, 33, 64, 65, 127, 128, 200, 257] {
        for (name, d0, e0) in fixtures(n) {
            // DSTEMR reads E(1..N-1) and uses E(N) as scratch.
            let mut e_full = vec![0.0; n.max(1)];
            e_full[..e0.len().min(n.max(1))].copy_from_slice(&e0[..e0.len().min(n.max(1))]);
            for job in [Job::ValuesOnly, Job::Vectors] {
                for tryrac0 in [true, false] {
                    let jz = if matches!(job, Job::Vectors) { 'V' } else { 'N' };
                    let tag = format!("n={n} {name} jobz={jz} tryrac={tryrac0}");
                    let (lw, liw) = dstemr_min_work_len(n, job);
                    let wantz = matches!(job, Job::Vectors);

                    let (mut d, mut e) = (d0.clone(), e_full.clone());
                    let mut w = vec![0.0; n.max(1)];
                    let mut z = vec![0.0; (n * n).max(1)];
                    let mut isuppz = vec![0i32; (2 * n).max(2)];
                    let mut tryrac = tryrac0;
                    let (mut work, mut iwork) = (vec![0.0; lw.max(1)], vec![0i32; liw.max(1)]);
                    // `INFO` is part of the contract: the "huge" fixture is a
                    // matrix DSTEMR genuinely fails on (element growth in
                    // DLARRF), and the transcription must fail the same way.
                    let (m, info) = match dstemr(
                        job,
                        n,
                        &mut d,
                        &mut e,
                        &mut w,
                        &mut z,
                        &mut isuppz,
                        &mut tryrac,
                        &mut work,
                        &mut iwork,
                    ) {
                        Ok(m) => (m, 0i32),
                        Err(lapack_rs::LapackError::NoConvergence { info, .. }) => (0, info as i32),
                        Err(err) => panic!("dstemr {tag}: {err:?}"),
                    };

                    let (mut dr, mut er) = (d0.clone(), e_full.clone());
                    let mut wr = vec![0.0; n.max(1)];
                    let mut zr = vec![0.0; (n * n).max(1)];
                    let mut isuppzr = vec![0i32; (2 * n).max(2)];
                    let (mut workr, mut iworkr) = (vec![0.0; lw.max(1)], vec![0i32; liw.max(1)]);
                    let (mr, tryracr, infor) = lapack_ref::dstemr(
                        jz,
                        'A',
                        n,
                        &mut dr,
                        &mut er,
                        0.0,
                        0.0,
                        1,
                        n,
                        &mut wr,
                        &mut zr,
                        n.max(1),
                        if wantz { n as i32 } else { 0 },
                        &mut isuppzr,
                        tryrac0,
                        &mut workr,
                        &mut iworkr,
                    );
                    assert_eq!(info, infor, "dstemr INFO {tag}");
                    assert_eq!(tryrac, tryracr, "dstemr TRYRAC {tag}");
                    if info != 0 {
                        continue;
                    }
                    assert_eq!(m as i32, mr, "dstemr M {tag}");
                    assert_bits(&format!("dstemr W {tag}"), &w[..m], &wr[..m]);
                    if wantz && n > 0 {
                        assert_bits(&format!("dstemr Z {tag}"), &z[..n * m], &zr[..n * m]);
                        assert_eq!(&isuppz[..2 * m], &isuppzr[..2 * m], "dstemr ISUPPZ {tag}");
                    }
                }
            }
        }
    }
}

#[test]
fn stemr_work_len_and_errors() {
    // The workspace query is `dstemr.f:389`'s 18*n / 10*n and 12*n / 8*n.
    for n in [0usize, 1, 7, 100] {
        assert_eq!(dstemr_min_work_len(n, Job::Vectors), (18 * n, 10 * n));
        assert_eq!(dstemr_min_work_len(n, Job::ValuesOnly), (12 * n, 8 * n));
        assert_eq!(dstemr_work_len(n, Job::Vectors), dstemr_min_work_len(n, Job::Vectors));
    }
    // A short workspace is an error, not a panic, and nothing is allocated.
    let n = 8;
    let (mut d, mut e) = (vec![1.0; n], vec![0.5; n]);
    let mut w = vec![0.0; n];
    let mut z = vec![0.0; n * n];
    let mut isuppz = vec![0i32; 2 * n];
    let mut tryrac = true;
    let mut work = vec![0.0; 3];
    let mut iwork = vec![0i32; 3];
    assert!(
        dstemr(Job::Vectors, n, &mut d, &mut e, &mut w, &mut z, &mut isuppz, &mut tryrac, &mut work, &mut iwork)
            .is_err()
    );
    // `n = 0` returns `m = 0` without touching anything.
    let mut e0: [f64; 0] = [];
    let mut e1: [f64; 0] = [];
    let mut e2: [f64; 0] = [];
    let mut e3: [f64; 0] = [];
    let mut e4: [f64; 0] = [];
    let mut i0: [i32; 0] = [];
    let mut i1: [i32; 0] = [];
    assert_eq!(
        dstemr(Job::Vectors, 0, &mut e0, &mut e1, &mut e2, &mut e3, &mut i0, &mut tryrac, &mut e4, &mut i1)
            .expect("n=0"),
        0
    );
    // DLAEV2/DLAE2 are the n = 2 fast path and are part of the public blocks.
    let (rt1, rt2, cs1, sn1) = blocks::dlaev2(3.0, 1.0, 2.0);
    let (a1, a2) = blocks::dlae2(3.0, 1.0, 2.0);
    assert_eq!((rt1.to_bits(), rt2.to_bits()), (a1.to_bits(), a2.to_bits()));
    assert!((cs1 * cs1 + sn1 * sn1 - 1.0).abs() < 1e-15);
}
