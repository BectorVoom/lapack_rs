//! Every device kernel on the CubeCL CPU runtime (f64) against the host
//! reference transcription on the same inputs: bit for bit (the CPU runtime's
//! contraction probe reports no contraction), with irregular shapes.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::geometry::{lda_for, vector_width};
use lapack_rs::device::ops::{self, Lanes};
use lapack_rs::device::workspace::{read_range, upload};
use lapack_rs::testing as t;

fn client() -> ComputeClient<CpuRuntime> {
    CpuRuntime::client(&CpuDevice)
}

/// Column-major `lda × n` device image of a symmetric `n × n` matrix.
fn stage_sym(a: &[f64], n: usize, lda: usize) -> Vec<f64> {
    let mut s = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..n {
            s[i + j * lda] = a[i + j * n];
        }
    }
    s
}

#[test]
fn k1_symv_matches_host_dsymv() {
    let c = client();
    for width in [1usize, 2, 4, 8] {
        let lanes = Lanes::new(&c, width);
        for &(n, off) in &[(1usize, 0usize), (5, 0), (37, 0), (45, 3), (45, 44), (64, 17), (70, 9)] {
            let lda = n.div_ceil(width) * width;
            let a = random_sym(n, 100 + n as u64);
            let m = n - off;
            let x = edge(m, 7 + off as u64);
            let h_m = upload(&c, &stage_sym(&a, n, lda));
            let h_x = upload(&c, &x);
            for upper in [false, true] {
                for alpha in [1.0, -0.75] {
                    let h_y = c.empty(lda * 8);
                    ops::symv::<CpuRuntime, f64>(&c, &lanes, &h_m, lda, n, off, m, &h_x, &h_y, alpha, upper);
                    let mut got = vec![0.0; m];
                    read_range(&c, &h_y, lda, off, m, &mut got);
                    // Host reference on the trailing block (column-major m × m).
                    let mut blk = vec![0.0; m * m];
                    for j in 0..m {
                        for i in 0..m {
                            blk[i + j * m] = a[(off + i) + (off + j) * n];
                        }
                    }
                    let (mut xx, mut want) = (x.clone(), vec![0.0; m]);
                    unsafe { t::dsymv(upper, m, alpha, &mut blk, m, &mut xx, 0.0, &mut want) };
                    assert_bits(&format!("K1 w={width} n={n} off={off} upper={upper} a={alpha}"), &got, &want);
                }
            }
        }
    }
}

#[test]
fn k2_syr2k_matches_host_on_both_triangles() {
    let c = client();
    for width in [1usize, 4, 8] {
        let lanes = Lanes::new(&c, width);
        for &(n, off, nb) in &[(9usize, 2usize, 3usize), (45, 12, 5), (37, 0, 8), (64, 30, 4)] {
            let lda = n.div_ceil(width) * width;
            let a = random_sym(n, 200 + n as u64);
            let m = n - off;
            let mut v = edge(m * nb, 11);
            let mut wv = edge(m * nb, 13);
            // Exact zeros in whole rows to exercise the zero-skip branches.
            for l in 0..nb {
                v[1 + l * m] = 0.0;
                wv[1 + l * m] = 0.0;
            }
            let mut packed = vec![0.0; 2 * nb * lda];
            for l in 0..nb {
                for i in 0..m {
                    packed[(off + i) + l * lda] = v[i + l * m];
                    packed[(off + i) + (nb + l) * lda] = wv[i + l * m];
                }
            }
            for upper in [false, true] {
                let h_c = upload(&c, &stage_sym(&a, n, lda));
                let h_vw = upload(&c, &packed);
                ops::syr2k::<CpuRuntime, f64>(&c, &lanes, &h_c, lda, n, off, m, &h_vw, nb, -1.0, upper);
                let mut got = vec![0.0f64; lda * n];
                read_range(&c, &h_c, lda * n, 0, lda * n, &mut got);
                let mut blk = vec![0.0; m * m];
                for j in 0..m {
                    for i in 0..m {
                        blk[i + j * m] = a[(off + i) + (off + j) * n];
                    }
                }
                let (mut v2, mut w2) = (v.clone(), wv.clone());
                unsafe { t::dsyr2k_n(upper, m, nb, -1.0, &mut v2, m, &mut w2, m, 1.0, &mut blk, m) };
                for j in 0..m {
                    for i in 0..m {
                        let own = if upper { i <= j } else { i >= j };
                        let (ri, rj) = if own { (i, j) } else { (j, i) };
                        let want = blk[ri + rj * m];
                        let g = got[(off + i) + (off + j) * lda];
                        assert_eq!(
                            g.to_bits(),
                            want.to_bits(),
                            "K2 w={width} n={n} off={off} nb={nb} upper={upper} ({i},{j}) {g:e} vs {want:e}"
                        );
                    }
                }
                // Outside the block nothing changed.
                let orig = stage_sym(&a, n, lda);
                for j in 0..n {
                    for i in 0..lda {
                        if i < off || j < off || i >= n {
                            assert_eq!(
                                got[i + j * lda].to_bits(),
                                orig[i + j * lda].to_bits(),
                                "K2 wrote outside block at ({i},{j})"
                            );
                        }
                    }
                }
            }
        }
    }
    let _ = (lda_for::<CpuRuntime>, vector_width::<CpuRuntime>);
}
