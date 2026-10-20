//! Bit-exact oracle tests for the OpenBLAS 0.3.30 SkylakeX level-2 transcriptions
//! (`lapack_rs::testing::openblas::level2`) against `scipy_dgemv_`, `scipy_dsymv_`, `scipy_dsyr2_`, `scipy_dger_`.
//! Every output element is compared, including untouched ones (other triangle, `lda` padding
//! rows, increment gaps).
mod common;
use common::ob::*;
use lapack_rs::testing::V;
use lapack_rs::testing::openblas::level2::{dgemv, dger, dsymv, dsyr2};
use openblas_ref as ob;

fn scalars(r: &mut Rng, one: bool) -> Vec<f64> {
    let mut v = vec![0.0, -1.0, 0.7 * r.uniform() - 1.3];
    if one {
        v.push(1.0);
    }
    v
}

fn fill(r: &mut Rng, n: usize, trial: usize) -> Vec<f64> {
    if trial % 2 == 0 { r.vec(n) } else { r.vec_edge(n) }
}

// ------------------------------------------------------------------------------------------
// DGEMV
// ------------------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn gemv_case(
    r: &mut Rng,
    trans: bool,
    m: usize,
    n: usize,
    ldpad: usize,
    incx: usize,
    incy: usize,
    alphas: &[f64],
    betas: &[f64],
    trial: usize,
) {
    let lda = m.max(1) + ldpad;
    let (lenx, leny) = if trans { (m, n) } else { (n, m) };
    let a = fill(r, lda * n.max(1), trial);
    let x = fill(r, lenx * incx + 1, trial + 1);
    let y0 = fill(r, leny * incy + 1, trial);
    for &alpha in alphas {
        for &beta in betas {
            let mut ya = a.clone();
            let mut xa = x.clone();
            let mut want = y0.clone();
            let mut got = y0.clone();
            let (m_, n_, lda_, ix, iy) = (m as i32, n as i32, lda as i32, incx as i32, incy as i32);
            let tc = ob::ch(if trans { 'T' } else { 'N' });
            // SAFETY: arrays sized for (m, n, lda, incx, incy).
            unsafe {
                ob::dgemv(&tc, &m_, &n_, &alpha, a.as_ptr(), &lda_, x.as_ptr(), &ix, &beta, want.as_mut_ptr(), &iy, 1)
            };
            // SAFETY: as above.
            unsafe {
                dgemv(trans, m, n, alpha, V::new(&mut ya), lda, V::new(&mut xa), incx, beta, V::new(&mut got), incy)
            };
            assert_bits(
                &format!(
                    "dgemv trans={trans} m={m} n={n} lda={lda} inc=({incx},{incy}) alpha={alpha:e} beta={beta:e} trial={trial}"
                ),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn dgemv_small_grid_bitexact() {
    if !ob::require("dgemv_small_grid_bitexact") {
        return;
    }
    let mut r = Rng::new(0x6e11);
    let s = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 20, 23, 24, 25, 31, 32, 33];
    for trans in [false, true] {
        for &m in &s {
            for &n in &s {
                for (incx, incy) in [(1usize, 1usize), (3, 1), (1, 2), (2, 3)] {
                    for ldpad in [0usize, 3] {
                        let trial = (m + n + incx) % 2;
                        let al = scalars(&mut r, true);
                        let be = scalars(&mut r, true);
                        gemv_case(&mut r, trans, m, n, ldpad, incx, incy, &al, &be, trial);
                    }
                }
            }
        }
    }
}

#[test]
fn dgemv_rectangular_bitexact() {
    if !ob::require("dgemv_rectangular_bitexact") {
        return;
    }
    let mut r = Rng::new(0x6e12);
    let small = [1usize, 2, 3, 5, 8, 13];
    let large = [63usize, 64, 65, 100, 127, 128, 129, 257, 300, 511, 512, 513, 700, 1023, 1024, 1025, 1100];
    for trans in [false, true] {
        for &b in &large {
            for &s in &small {
                for (m, n) in [(b, s), (s, b)] {
                    for (incx, incy) in [(1usize, 1usize), (3, 1), (1, 2)] {
                        for ldpad in [0usize, 5] {
                            let al = scalars(&mut r, true);
                            let be = scalars(&mut r, true);
                            gemv_case(&mut r, trans, m, n, ldpad, incx, incy, &al, &be, (m ^ n) % 2);
                        }
                    }
                }
            }
        }
        // Big x big, tall and wide, around the lane/tail boundaries (m & 3, n & 3).
        for (m, n) in [
            (300, 257),
            (257, 300),
            (511, 129),
            (129, 511),
            (700, 513),
            (513, 702),
            (1100, 1025),
            (1025, 1099),
            (1024, 1024),
            (1101, 1102),
        ] {
            for (incx, incy) in [(1usize, 1usize), (7, 1), (1, 3)] {
                for ldpad in [0usize, 2] {
                    let al = [1.0, -1.0, r.uniform()];
                    let be = [0.0, 1.0, r.uniform()];
                    gemv_case(&mut r, trans, m, n, ldpad, incx, incy, &al, &be, 1);
                }
            }
        }
    }
}

#[test]
fn dgemv_nbmax_blocks_bitexact() {
    if !ob::require("dgemv_nbmax_blocks_bitexact") {
        return;
    }
    // NBMAX = 2048 row blocks ('N' and 'T') and the n0 column blocks of 'T'.
    let mut r = Rng::new(0x6e13);
    for trans in [false, true] {
        for (m, n) in [
            (2048, 3),
            (2049, 2),
            (2051, 5),
            (2052, 1),
            (4099, 3),
            (4096, 2),
            (6147, 1),
            (3, 2048),
            (5, 2051),
            (2, 4097),
            (7, 6150),
            (2051, 2053),
        ] {
            for (incx, incy) in [(1usize, 1usize), (2, 1), (1, 2)] {
                let al = [-1.0, r.uniform()];
                let be = [0.0, 1.0, r.uniform()];
                gemv_case(&mut r, trans, m, n, 1, incx, incy, &al, &be, 1);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// DSYMV
// ------------------------------------------------------------------------------------------

#[test]
fn dsymv_bitexact() {
    if !ob::require("dsymv_bitexact") {
        return;
    }
    let mut r = Rng::new(0x5e11);
    let mut ns = sizes(1100);
    ns.extend([767, 768, 769, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1100]);
    for n in ns {
        for upper in [true, false] {
            for ldpad in [0usize, 3] {
                let lda = n.max(1) + ldpad;
                let trial = (n + ldpad) % 3;
                let a = fill(&mut r, lda * n.max(1), trial);
                let x = fill(&mut r, n + 1, trial + 1);
                let y0 = fill(&mut r, n + 1, trial);
                for alpha in scalars(&mut r, true) {
                    for beta in scalars(&mut r, true) {
                        let mut aa = a.clone();
                        let mut xa = x.clone();
                        let mut want = y0.clone();
                        let mut got = y0.clone();
                        let (n_, lda_, one) = (n as i32, lda as i32, 1i32);
                        let uc = ob::ch(if upper { 'U' } else { 'L' });
                        // SAFETY: arrays sized for (n, lda), unit increments.
                        unsafe {
                            ob::dsymv(
                                &uc,
                                &n_,
                                &alpha,
                                a.as_ptr(),
                                &lda_,
                                x.as_ptr(),
                                &one,
                                &beta,
                                want.as_mut_ptr(),
                                &one,
                                1,
                            )
                        };
                        // SAFETY: as above.
                        unsafe {
                            dsymv(upper, n, alpha, V::new(&mut aa), lda, V::new(&mut xa), beta, V::new(&mut got))
                        };
                        assert_bits(
                            &format!("dsymv upper={upper} n={n} lda={lda} alpha={alpha:e} beta={beta:e} trial={trial}"),
                            &got,
                            &want,
                        );
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// DSYR2
// ------------------------------------------------------------------------------------------

#[test]
fn dsyr2_bitexact() {
    if !ob::require("dsyr2_bitexact") {
        return;
    }
    let mut r = Rng::new(0x5e12);
    let mut ns = sizes(1100);
    ns.extend([98, 99, 100, 101, 1023, 1024, 1025, 1100]);
    for n in ns {
        for upper in [true, false] {
            for ldpad in [0usize, 2] {
                let lda = n.max(1) + ldpad;
                let trial = (n + ldpad) % 3;
                let a0 = fill(&mut r, lda * n.max(1), trial);
                let x = fill(&mut r, n + 1, trial + 1);
                let y = fill(&mut r, n + 1, trial);
                for alpha in scalars(&mut r, true) {
                    let mut want = a0.clone();
                    let mut got = a0.clone();
                    let mut xa = x.clone();
                    let mut ya = y.clone();
                    let (n_, lda_, one) = (n as i32, lda as i32, 1i32);
                    let uc = ob::ch(if upper { 'U' } else { 'L' });
                    // SAFETY: arrays sized for (n, lda), unit increments.
                    unsafe {
                        ob::dsyr2(&uc, &n_, &alpha, x.as_ptr(), &one, y.as_ptr(), &one, want.as_mut_ptr(), &lda_, 1)
                    };
                    // SAFETY: as above.
                    unsafe { dsyr2(upper, n, alpha, V::new(&mut xa), V::new(&mut ya), V::new(&mut got), lda) };
                    assert_bits(
                        &format!("dsyr2 upper={upper} n={n} lda={lda} alpha={alpha:e} trial={trial}"),
                        &got,
                        &want,
                    );
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// DGER
// ------------------------------------------------------------------------------------------

fn ger_case(r: &mut Rng, m: usize, n: usize, ldpad: usize, incx: usize, incy: usize, trial: usize) {
    let lda = m.max(1) + ldpad;
    let a0 = fill(r, lda * n.max(1), trial);
    let x = fill(r, m * incx + 1, trial + 1);
    let y = fill(r, n * incy + 1, trial);
    let tau = r.uniform() + 1.0;
    for alpha in [-tau, -1.0, 0.0, 1.0] {
        let mut want = a0.clone();
        let mut got = a0.clone();
        let mut xa = x.clone();
        let mut ya = y.clone();
        let (m_, n_, lda_, ix, iy) = (m as i32, n as i32, lda as i32, incx as i32, incy as i32);
        // SAFETY: arrays sized for (m, n, lda, incx, incy).
        unsafe { ob::dger(&m_, &n_, &alpha, x.as_ptr(), &ix, y.as_ptr(), &iy, want.as_mut_ptr(), &lda_) };
        // SAFETY: as above.
        unsafe { dger(m, n, alpha, V::new(&mut xa), incx, V::new(&mut ya), incy, V::new(&mut got), lda) };
        assert_bits(
            &format!("dger m={m} n={n} lda={lda} inc=({incx},{incy}) alpha={alpha:e} trial={trial}"),
            &got,
            &want,
        );
    }
}

#[test]
fn dger_bitexact() {
    if !ob::require("dger_bitexact") {
        return;
    }
    let mut r = Rng::new(0x9e11);
    let s = [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65];
    for &m in &s {
        for &n in &s {
            for (incx, incy) in [(1usize, 1usize), (2, 1), (1, 3)] {
                for ldpad in [0usize, 3] {
                    ger_case(&mut r, m, n, ldpad, incx, incy, (m + n) % 2);
                }
            }
        }
    }
    // Around the m*n <= 8192 no-buffer threshold and the STACK_ALLOC(m) buffer, tall/wide/large.
    for (m, n) in [
        (128, 64),
        (128, 65),
        (8192, 1),
        (8193, 1),
        (1, 8193),
        (91, 90),
        (91, 91),
        (257, 300),
        (700, 3),
        (3, 700),
        (1100, 1025),
        (1025, 1100),
    ] {
        for (incx, incy) in [(1usize, 1usize), (2, 1)] {
            for ldpad in [0usize, 4] {
                ger_case(&mut r, m, n, ldpad, incx, incy, 1);
            }
        }
    }
}
