//! **OpenBLAS track.** The OpenBLAS-order device kernels (`K*ob`) on the
//! CubeCL CPU runtime (f64) against the host OpenBLAS transcriptions
//! (`crate::openblas`, themselves gated bit for bit against scipy's OpenBLAS
//! by `tests/openblas_*_bitexact.rs`). No oracle library is needed here.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::ops;
use lapack_rs::device::workspace::{read_range, upload};
use lapack_rs::testing as t;

fn client() -> ComputeClient<CpuRuntime> {
    CpuRuntime::client(&CpuDevice)
}

#[test]
fn k1ob_symv_matches_host_openblas_dsymv() {
    let c = client();
    let mut sizes: Vec<usize> = (1..=40).collect();
    sizes.extend([47, 48, 49, 63, 64, 65, 100, 129, 200, 257]);
    for &m in &sizes {
        for &off in &[0usize, 3] {
            let n = m + off;
            let lda = n + 5;
            let a = random_sym(n, 300 + n as u64);
            let mut img = vec![0.0; lda * n];
            for j in 0..n {
                for i in 0..n {
                    img[i + j * lda] = a[i + j * n];
                }
            }
            let x = edge(m, 11 + m as u64);
            let h_m = upload(&c, &img);
            let h_x = upload(&c, &x);
            for upper in [false, true] {
                for alpha in [1.0, -0.731] {
                    let h_y = upload(&c, &vec![7.0; lda]);
                    ops::ob_symv::<CpuRuntime, f64>(&c, &h_m, lda, n, off, m, &h_x, &h_y, alpha, upper);
                    let mut got = vec![0.0; m];
                    read_range(&c, &h_y, lda, off, m, &mut got);
                    let mut blk = vec![0.0; m * m];
                    for j in 0..m {
                        for i in 0..m {
                            blk[i + j * m] = a[off + i + (off + j) * n];
                        }
                    }
                    let mut xx = x.clone();
                    let mut want = vec![7.0; m];
                    // SAFETY: m×m block, m-vectors.
                    unsafe { t::openblas::dsymv(upper, m, alpha, &mut blk, m, &mut xx, 0.0, &mut want) };
                    assert_bits(&format!("K1ob m={m} off={off} upper={upper} alpha={alpha}"), &got, &want);
                }
            }
        }
    }
}

#[test]
fn k2ob_syr2k_matches_host_openblas_dsyr2k() {
    let c = client();
    let mut sizes: Vec<usize> =
        vec![1, 2, 5, 17, 31, 32, 33, 47, 63, 64, 65, 100, 191, 192, 193, 200, 255, 383, 384, 385, 450, 600];
    sizes.sort();
    for &m in &sizes {
        for &(off, nb) in &[(0usize, 32usize), (7, 32), (0, 5), (3, 13)] {
            let n = m + off;
            let lda = (n + 3).div_ceil(8) * 8;
            let a = random_sym(n, 400 + n as u64);
            let mut img = vec![0.0; lda * n];
            for j in 0..n {
                for i in 0..n {
                    img[i + j * lda] = a[i + j * n];
                }
            }
            let mut r = Rng::new(9 + m as u64);
            let v: Vec<f64> = (0..m * nb).map(|_| r.uniform()).collect();
            let w: Vec<f64> = (0..m * nb).map(|_| r.uniform()).collect();
            let mut packed = vec![0.0; 2 * nb * lda];
            for l in 0..nb {
                for i in 0..m {
                    packed[l * lda + off + i] = v[i + l * m];
                    packed[(nb + l) * lda + off + i] = w[i + l * m];
                }
            }
            let h_vw = upload(&c, &packed);
            for (upper, width) in [(false, 1usize), (true, 1), (false, 8), (true, 8)] {
                let lanes = ops::Lanes::new(&c, width);
                let h_m = upload(&c, &img);
                ops::ob_syr2k::<CpuRuntime, f64>(&c, &lanes, &h_m, lda, n, off, m, &h_vw, nb, -1.0, upper);
                let mut got = vec![0.0; lda * n];
                read_range(&c, &h_m, lda * n, 0, lda * n, &mut got);
                let mut blk = vec![0.0; m * m];
                for j in 0..m {
                    for i in 0..m {
                        blk[i + j * m] = a[off + i + (off + j) * n];
                    }
                }
                let (mut vv, mut ww) = (v.clone(), w.clone());
                // SAFETY: m×nb V/W, m×m C.
                unsafe { t::openblas::dsyr2k_n(upper, m, nb, -1.0, &mut vv, m, &mut ww, m, 1.0, &mut blk, m) };
                let mut want = vec![0.0; m * m];
                let mut have = vec![0.0; m * m];
                for j in 0..m {
                    for i in 0..m {
                        let (ti, tj) = if (upper && i <= j) || (!upper && i >= j) { (i, j) } else { (j, i) };
                        want[i + j * m] = blk[ti + tj * m];
                        have[i + j * m] = got[off + i + (off + j) * lda];
                    }
                }
                assert_bits(&format!("K2ob m={m} off={off} nb={nb} upper={upper} width={width}"), &have, &want);
            }
        }
    }
}

#[test]
fn k3m_ob_merge_matches_host_openblas_dgemm() {
    let c = client();
    // (block rows m, output columns n, k, terms t): small and blocked paths,
    // blocked row blocks (> 192, > 384 rows) and l blocks (> 384 terms).
    let cases: &[(usize, usize, usize, usize)] = &[
        (5, 7, 5, 3),
        (13, 20, 17, 16),
        (12, 12, 12, 0),
        (40, 50, 45, 40),
        (100, 120, 100, 100),
        (200, 210, 200, 150),
        (400, 420, 400, 390),
        (130, 800, 790, 777),
        (211, 40, 40, 400),
        (212, 40, 37, 400),
        (212, 26, 26, 800),
    ];
    for &(m, n, k, t) in cases {
        for width in [1usize, 8] {
            let lda = (m + 3).div_ceil(8) * 8;
            let ncols = n + t + 5;
            let mut r = Rng::new(31 + (m * n) as u64);
            let zimg: Vec<f64> = (0..lda * ncols).map(|_| r.uniform()).collect();
            let perm: Vec<u32> = (0..t.max(1)).map(|l| ((l * 7 + 3) % ncols) as u32).collect();
            let perm4: Vec<u32> = (0..(n - k).max(1)).map(|l| ((l * 5 + 1) % ncols) as u32).collect();
            let lds = t + 2;
            let s: Vec<f64> = (0..(lds * k).max(1)).map(|_| r.uniform()).collect();
            let (hz, hp, hp4, hs) = (upload(&c, &zimg), upload(&c, &perm), upload(&c, &perm4), upload(&c, &s));
            let blk_r0 = 2;
            let blk_m = m.min(lda - blk_r0);
            // dst covers all lda rows (lane-aligned), sentinel outside the block.
            let dst = upload(&c, &vec![9.0f64; lda * n]);
            let lanes = ops::Lanes::new(&c, width);
            ops::ob_merge_rows::<CpuRuntime, f64>(
                &c,
                &lanes,
                &hz,
                lda,
                ncols,
                &dst,
                lda,
                0,
                n,
                k,
                &hp,
                t,
                &hp4,
                &hs,
                lds,
                1,
                blk_r0,
                blk_r0 + blk_m,
                blk_r0,
                blk_m,
            );
            let mut full = vec![0.0; lda * n];
            read_range(&c, &dst, lda * n, 0, lda * n, &mut full);
            let mut got = vec![0.0; blk_m * n];
            for j in 0..n {
                for i in 0..blk_m {
                    got[i + j * blk_m] = full[j * lda + blk_r0 + i];
                }
                for i in (0..blk_r0).chain(blk_r0 + blk_m..lda) {
                    assert_eq!(full[j * lda + i], 9.0, "row {i} outside the block was written");
                }
            }
            let mut q2 = vec![0.0; blk_m * t.max(1)];
            for l in 0..t {
                for i in 0..blk_m {
                    q2[i + l * blk_m] = zimg[perm[l] as usize * lda + blk_r0 + i];
                }
            }
            let mut sb = vec![0.0; t.max(1) * k];
            for j in 0..k {
                for l in 0..t {
                    sb[l + j * t.max(1)] = s[j * lds + 1 + l];
                }
            }
            let mut want = vec![7.0; blk_m * n];
            if t > 0 {
                // SAFETY: shapes as allocated.
                unsafe {
                    t::openblas::dgemm(
                        false, false, blk_m, k, t, 1.0, &mut q2, blk_m, &mut sb, t, 0.0, &mut want, blk_m,
                    )
                };
            } else {
                want[..blk_m * k].iter_mut().for_each(|v| *v = 0.0);
            }
            for j in k..n {
                for i in 0..blk_m {
                    want[i + j * blk_m] = zimg[perm4[j - k] as usize * lda + blk_r0 + i];
                }
            }
            assert_bits(&format!("K3m-ob m={blk_m} n={n} k={k} t={t} width={width}"), &got, &want);
        }
    }
}

#[test]
fn k7rot_ob_matches_host_openblas_drot() {
    let c = client();
    let lanes = ops::Lanes::new(&c, 8);
    let (lda, ncols) = (40usize, 9usize);
    let mut r = Rng::new(77);
    let img: Vec<f64> = (0..lda * ncols).map(|_| r.uniform()).collect();
    let rots: Vec<(u32, u32, f64, f64)> = vec![(1, 4, 0.6, 0.8), (4, 7, -0.28, 0.96), (2, 1, 0.123, -0.9923)];
    let h = upload(&c, &img);
    let cols: Vec<u32> = rots.iter().flat_map(|&(a, b, _, _)| [a, b]).collect();
    let cs: Vec<f64> = rots.iter().flat_map(|&(_, _, c, s)| [c, s]).collect();
    let (hc, hs) = (upload(&c, &cols), upload(&c, &cs));
    ops::ob_rotations::<CpuRuntime, f64>(&c, &lanes, &h, lda, ncols, &hc, &hs, rots.len(), 3, 37);
    let mut got = vec![0.0; lda * ncols];
    read_range(&c, &h, lda * ncols, 0, lda * ncols, &mut got);
    let mut want = img.clone();
    for &(a, b, cc, ss) in &rots {
        let (a, b) = (a as usize, b as usize);
        let mut x: Vec<f64> = (3..37).map(|i| want[a * lda + i]).collect();
        let mut y: Vec<f64> = (3..37).map(|i| want[b * lda + i]).collect();
        // SAFETY: equal lengths.
        unsafe { t::openblas::drot(x.len(), &mut x, &mut y, cc, ss) };
        for (q, i) in (3..37).enumerate() {
            want[a * lda + i] = x[q];
            want[b * lda + i] = y[q];
        }
    }
    assert_bits("K7rot-ob", &got, &want);
}

/// Row-major `n × kpad` image of a column-major `n × k` matrix.
fn to_rm(b: &[f64], n: usize, k: usize, kpad: usize) -> Vec<f64> {
    let mut w = vec![0.0; n * kpad];
    for i in 0..n {
        for j in 0..k {
            w[i * kpad + j] = b[i + j * n];
        }
    }
    w
}

#[test]
fn kob_trmm_right_matches_host_openblas_dtrmm() {
    let c = client();
    for &n in &[1usize, 3, 17, 40, 191, 192, 193, 250, 400, 513] {
        for &k in &[1usize, 2, 5, 7, 13, 24, 31, 32] {
            let kpad = k.div_ceil(8) * 8;
            let mut r = Rng::new((n * 100 + k) as u64);
            let a: Vec<f64> = (0..k * k).map(|_| r.uniform()).collect();
            let b: Vec<f64> = (0..n * k).map(|_| r.uniform()).collect();
            let hw = upload(&c, &to_rm(&b, n, k, kpad));
            for upper in [false, true] {
                for trans in [false, true] {
                    for nounit in [false, true] {
                        let ho = upload(&c, &vec![0.0f64; n * kpad]);
                        ops::ob_trmm_right::<CpuRuntime, f64>(&c, &hw, &ho, &a, n, k, kpad, upper, trans, nounit);
                        let mut got_rm = vec![0.0; n * kpad];
                        read_range(&c, &ho, n * kpad, 0, n * kpad, &mut got_rm);
                        let (mut aa, mut bb) = (a.clone(), b.clone());
                        // SAFETY: k×k A, n×k B.
                        unsafe { t::openblas::dtrmm(false, upper, trans, nounit, n, k, 1.0, &mut aa, k, &mut bb, n) };
                        let got: Vec<f64> = (0..n * k).map(|p| got_rm[(p % n) * kpad + p / n]).collect();
                        assert_bits(
                            &format!("Kob-trmm n={n} k={k} upper={upper} trans={trans} nounit={nounit}"),
                            &got,
                            &bb,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn kob_gemm_tn_nt_match_host_openblas_dgemm() {
    let c = client();
    // (n columns of C, k, t rows of the rectangular part)
    for &(n, k, t) in &[
        (10usize, 32usize, 40usize),
        (37, 32, 50),
        (38, 32, 33),
        (100, 32, 60),
        (250, 32, 300),
        (513, 32, 800),
        (600, 13, 70),
        (30, 5, 20),
        (420, 29, 450),
        (300, 17, 211),
    ] {
        for width in [1usize, 8] {
            let kpad = k.div_ceil(8) * 8;
            let crow = 5usize;
            let lda = (crow + t + 7).div_ceil(8) * 8;
            let mut r = Rng::new((n * 31 + t) as u64);
            let z: Vec<f64> = (0..lda * n).map(|_| r.uniform()).collect();
            let v: Vec<f64> = (0..t * k).map(|_| r.uniform()).collect(); // column-major t × k
            let w: Vec<f64> = (0..n * k).map(|_| r.uniform()).collect(); // column-major n × k
            let mut v_rm = vec![0.0; t * kpad];
            let pad = crow % 8;
            let ldvp = (pad + t).div_ceil(8) * 8;
            let mut v_cm = vec![0.0; ldvp * k];
            for l in 0..t {
                for j in 0..k {
                    v_rm[l * kpad + j] = v[l + j * t];
                    v_cm[pad + l + j * ldvp] = v[l + j * t];
                }
            }
            let lanes = ops::Lanes::new(&c, width);
            let (hz, hv, hvc, hw) =
                (upload(&c, &z), upload(&c, &v_rm), upload(&c, &v_cm), upload(&c, &to_rm(&w, n, k, kpad)));
            // TN: W += C2ᵀ V2 with C2 = Z rows crow..crow+t.
            let ho = upload(&c, &vec![0.0f64; n * kpad]);
            ops::ob_gemm_tn::<CpuRuntime, f64>(&c, &lanes, &hz, lda, n, &hv, t, &hw, &ho, n, k, kpad, crow, 0, t);
            let mut got_rm = vec![0.0; n * kpad];
            read_range(&c, &ho, n * kpad, 0, n * kpad, &mut got_rm);
            let got: Vec<f64> = (0..n * k).map(|p| got_rm[(p % n) * kpad + p / n]).collect();
            let (mut c2, mut vv, mut want) = (vec![0.0; t * n], v.clone(), w.clone());
            for i in 0..n {
                for l in 0..t {
                    c2[l + i * t] = z[i * lda + crow + l];
                }
            }
            // SAFETY: shapes as allocated.
            unsafe { t::openblas::dgemm(true, false, n, k, t, 1.0, &mut c2, t, &mut vv, t, 1.0, &mut want, n) };
            assert_bits(&format!("Kob-gemm-tn n={n} k={k} t={t} width={width}"), &got, &want);
            // NT: C2 -= V2 Wᵀ.
            ops::ob_gemm_nt::<CpuRuntime, f64>(
                &c,
                &lanes,
                &hz,
                lda,
                n,
                &hv,
                t,
                Some((&hvc, ldvp)),
                &hw,
                n,
                k,
                kpad,
                crow,
                0,
                t,
            );
            let mut zz = vec![0.0f64; lda * n];
            read_range(&c, &hz, lda * n, 0, lda * n, &mut zz);
            let (mut vv, mut ww) = (v.clone(), w.clone());
            let mut c2w = vec![0.0; t * n];
            for i in 0..n {
                for l in 0..t {
                    c2w[l + i * t] = z[i * lda + crow + l];
                }
            }
            // SAFETY: shapes as allocated.
            unsafe { t::openblas::dgemm(false, true, t, n, k, -1.0, &mut vv, t, &mut ww, n, 1.0, &mut c2w, t) };
            let mut got2 = vec![0.0; t * n];
            for i in 0..n {
                for l in 0..t {
                    got2[l + i * t] = zz[i * lda + crow + l];
                }
                for l in (0..crow).chain(crow + t..lda) {
                    assert_eq!(zz[i * lda + l].to_bits(), z[i * lda + l].to_bits(), "row {l} outside C2 was written");
                }
            }
            assert_bits(&format!("Kob-gemm-nt n={n} k={k} t={t} width={width}"), &got2, &c2w);
        }
    }
}

#[test]
fn k2ob_special_intervals_match_elementwise_rule() {
    for &m in &[1usize, 2, 17, 31, 32, 33, 64, 100, 191, 192, 193, 200, 257, 300, 383, 384, 385, 450, 520, 700] {
        for upper in [false, true] {
            let (off, iv) = ops::ob_syr2k_special_intervals(m, upper);
            for j in 0..m {
                let mut from_iv = vec![false; m];
                for t in off[j] as usize..off[j + 1] as usize {
                    for i in iv[2 * t] as usize..iv[2 * t + 1] as usize {
                        from_iv[i] = true;
                    }
                }
                for i in 0..m {
                    let (p, q) = if upper { (i.min(j), i.max(j)) } else { (i.max(j), i.min(j)) };
                    let want = ops::ob_syr2k_special_elem(m, upper, p, q);
                    assert_eq!(from_iv[i], want, "m={m} upper={upper} (i={i}, j={j})");
                }
            }
        }
    }
}

#[test]
fn k1c_ob_hemv_matches_host_openblas_zhemv() {
    let c = client();
    let mut sizes: Vec<usize> = (1..=20).collect();
    sizes.extend([31, 32, 33, 64, 65, 100, 129, 200]);
    for &m in &sizes {
        for &off in &[0usize, 3] {
            let n = m + off;
            let lda = n + 5;
            let (re, im) = random_herm(n, 600 + n as u64);
            let (mut ir, mut ii) = (vec![0.0; lda * n], vec![0.0; lda * n]);
            for j in 0..n {
                for i in 0..n {
                    ir[i + j * lda] = re[i + j * n];
                    ii[i + j * lda] = im[i + j * n];
                }
            }
            let (xr, xi) = (edge(m, 21 + m as u64), edge(m, 22 + m as u64));
            let hm = lapack_rs::device::ops::ZHandle { re: upload(&c, &ir), im: upload(&c, &ii) };
            for upper in [false, true] {
                for alpha in [(1.0, 0.0), (-0.37, 0.81)] {
                    let (t1r, t1i) = ops::ob_hemv_t1(&xr, &xi, alpha);
                    let hx = lapack_rs::device::ops::ZHandle { re: upload(&c, &t1r), im: upload(&c, &t1i) };
                    let hy = lapack_rs::device::ops::ZHandle {
                        re: upload(&c, &vec![7.0; lda]),
                        im: upload(&c, &vec![7.0; lda]),
                    };
                    ops::ob_hemv::<CpuRuntime, f64>(&c, &hm, lda, n, off, m, &hx, &hy, alpha, upper);
                    let (mut gr, mut gi) = (vec![0.0; m], vec![0.0; m]);
                    read_range(&c, &hy.re, lda, off, m, &mut gr);
                    read_range(&c, &hy.im, lda, off, m, &mut gi);
                    let (mut br, mut bi) = (vec![0.0; m * m], vec![0.0; m * m]);
                    for j in 0..m {
                        for i in 0..m {
                            br[i + j * m] = re[off + i + (off + j) * n];
                            bi[i + j * m] = im[off + i + (off + j) * n];
                        }
                    }
                    let (mut xxr, mut xxi) = (xr.clone(), xi.clone());
                    let (mut wr, mut wi) = (vec![7.0; m], vec![7.0; m]);
                    // SAFETY: m×m block, m-vectors.
                    unsafe {
                        t::openblas::zhemv(
                            upper,
                            m,
                            alpha,
                            &mut br,
                            &mut bi,
                            m,
                            &mut xxr,
                            &mut xxi,
                            (0.0, 0.0),
                            &mut wr,
                            &mut wi,
                        )
                    };
                    let tag = format!("K1c-ob m={m} off={off} upper={upper} alpha={alpha:?}");
                    assert_bits(&format!("{tag} re"), &gr, &wr);
                    assert_bits(&format!("{tag} im"), &gi, &wi);
                }
            }
        }
    }
}

#[test]
fn k2c_ob_her2k_matches_host_openblas_zher2k() {
    let c = client();
    for &m in &[1usize, 2, 3, 4, 5, 7, 8, 17, 31, 32, 33, 64, 100, 255, 256, 257, 300, 511, 512, 513, 600] {
        for &(off, nb) in &[(0usize, 32usize), (5, 32), (0, 7), (2, 3)] {
            let n = m + off;
            let lda = (n + 3).div_ceil(8) * 8;
            let (re, im) = random_herm(n, 800 + n as u64);
            let (mut ir, mut ii) = (vec![0.0; lda * n], vec![0.0; lda * n]);
            for j in 0..n {
                for i in 0..n {
                    ir[i + j * lda] = re[i + j * n];
                    ii[i + j * lda] = im[i + j * n];
                }
            }
            let mut r = Rng::new(19 + m as u64);
            let mut rv = |len: usize| -> Vec<f64> { (0..len).map(|_| r.uniform()).collect() };
            let (v_re, v_im, w_re, w_im) = (rv(m * nb), rv(m * nb), rv(m * nb), rv(m * nb));
            let (mut pr, mut pi) = (vec![0.0; 2 * nb * lda], vec![0.0; 2 * nb * lda]);
            for l in 0..nb {
                for i in 0..m {
                    pr[l * lda + off + i] = v_re[i + l * m];
                    pi[l * lda + off + i] = v_im[i + l * m];
                    pr[(nb + l) * lda + off + i] = w_re[i + l * m];
                    pi[(nb + l) * lda + off + i] = w_im[i + l * m];
                }
            }
            let hvw = lapack_rs::device::ops::ZHandle { re: upload(&c, &pr), im: upload(&c, &pi) };
            for (upper, width) in [(false, 1usize), (true, 1), (false, 8), (true, 8)] {
                let alpha = (-1.0, -0.0);
                let lanes = ops::Lanes::new(&c, width);
                let hm = lapack_rs::device::ops::ZHandle { re: upload(&c, &ir), im: upload(&c, &ii) };
                ops::ob_her2k::<CpuRuntime, f64>(&c, &lanes, &hm, lda, n, off, m, &hvw, nb, alpha, upper);
                let (mut gr, mut gi) = (vec![0.0; lda * n], vec![0.0; lda * n]);
                read_range(&c, &hm.re, lda * n, 0, lda * n, &mut gr);
                read_range(&c, &hm.im, lda * n, 0, lda * n, &mut gi);
                let (mut br, mut bi) = (vec![0.0; m * m], vec![0.0; m * m]);
                for j in 0..m {
                    for i in 0..m {
                        br[i + j * m] = re[off + i + (off + j) * n];
                        bi[i + j * m] = im[off + i + (off + j) * n];
                    }
                }
                let (mut a1, mut a2, mut b1, mut b2) = (v_re.clone(), v_im.clone(), w_re.clone(), w_im.clone());
                // SAFETY: m×nb V/W, m×m C.
                unsafe {
                    t::openblas::zher2k_n(
                        upper, m, nb, alpha, &mut a1, &mut a2, m, &mut b1, &mut b2, m, 1.0, &mut br, &mut bi, m,
                    )
                };
                let (mut wr, mut wi, mut hr, mut hi) =
                    (vec![0.0; m * m], vec![0.0; m * m], vec![0.0; m * m], vec![0.0; m * m]);
                for j in 0..m {
                    for i in 0..m {
                        let own = (upper && i <= j) || (!upper && i >= j);
                        let (ti, tj) = if own { (i, j) } else { (j, i) };
                        wr[i + j * m] = br[ti + tj * m];
                        wi[i + j * m] = if own || i == j { bi[ti + tj * m] } else { -bi[ti + tj * m] };
                        hr[i + j * m] = gr[off + i + (off + j) * lda];
                        hi[i + j * m] = gi[off + i + (off + j) * lda];
                    }
                }
                let tag = format!("K2c-ob m={m} off={off} nb={nb} upper={upper} width={width}");
                assert_bits(&format!("{tag} re"), &hr, &wr);
                assert_bits(&format!("{tag} im"), &hi, &wi);
            }
        }
    }
}

#[test]
fn k2c_ob_special_intervals_match_elementwise_rule() {
    for &m in &[1usize, 2, 3, 4, 5, 7, 17, 255, 256, 257, 300, 511, 512, 513, 700, 1030] {
        for upper in [false, true] {
            let (off, iv) = ops::ob_her2k_special_intervals(m, upper);
            for j in 0..m {
                let mut from_iv = vec![false; m];
                for t in off[j] as usize..off[j + 1] as usize {
                    for i in iv[2 * t] as usize..iv[2 * t + 1] as usize {
                        from_iv[i] = true;
                    }
                }
                for i in 0..m {
                    let (p, q) = if upper { (i.min(j), i.max(j)) } else { (i.max(j), i.min(j)) };
                    assert_eq!(
                        from_iv[i],
                        ops::ob_her2k_special_elem(m, upper, p, q),
                        "m={m} upper={upper} (i={i}, j={j})"
                    );
                }
            }
        }
    }
}
