//! **OpenBLAS track.** K1ob: `DSYMV` (`BETA = 0`) bit-exact to OpenBLAS 0.3.30
//! `dsymv_L_SKYLAKEX` / `dsymv_U_SKYLAKEX` (the host transcription is
//! [`crate::openblas::level2`]), on a fully symmetric device matrix.
//!
//! OpenBLAS walks columns in groups of four and fuses every accumulation
//! (`fma`). Although its loops are column-major, each output row's value only
//! depends on `A` and `x`, never on another row's `y`: row `i` is a chain
//! over the column groups in ascending order, and the column sum `tmp2` of
//! column `i` is a fixed set of lane chains. So one unit owns one row and
//! reads only column `i` of the block (the row read `A(i, j)` is `M[j, i]`,
//! identical bits because both triangles are kept equal), exactly the access
//! pattern of K1col. The row-region boundaries (kernel rows vs scalar rows,
//! `j1 = (j/8)·8` for upper, `n - (j+1) >= 12` and `m2` for lower) decide the
//! per-row formula and are reproduced here.
//!
//! `y = s + y` and the lane folds are unfused adds; `fma(a, b, c)` is one
//! fused multiply-add (CubeCL `fma`, lowered to `llvm.fma` on the CPU
//! runtime). The operands of the product commute exactly, so only the
//! fused/unfused structure and the sum order matter.

use cubecl::prelude::*;

/// `A(r, i)` of the block: column `i` read at row `r`.
#[cube]
fn at<F: Float + CubeElement>(m_buf: &Array<F>, base: u32, r: u32) -> F {
    m_buf[(base + r) as usize]
}

/// `((l0 + l2) + (l1 + l3))` of four lane chains `acc[r & 3] = fma(A(r,i), x_r, acc)`
/// over `r in lo..hi` with `hi - lo` a multiple of 4 and lane = `(r - lo) & 3`.
#[cube]
fn fold4<F: Float + CubeElement>(m_buf: &Array<F>, x: &Array<F>, base: u32, lo: u32, hi: u32) -> F {
    let mut l0 = F::from_int(0);
    let mut l1 = F::from_int(0);
    let mut l2 = F::from_int(0);
    let mut l3 = F::from_int(0);
    let mut r = lo;
    while r < hi {
        l0 = fma(at::<F>(m_buf, base, r), x[r as usize], l0);
        l1 = fma(at::<F>(m_buf, base, r + 1), x[(r + 1) as usize], l1);
        l2 = fma(at::<F>(m_buf, base, r + 2), x[(r + 2) as usize], l2);
        l3 = fma(at::<F>(m_buf, base, r + 3), x[(r + 3) as usize], l3);
        r += 4u32;
    }
    (l0 + l2) + (l1 + l3)
}

/// `acc = fma(A(r,i), x_r, acc)` for `r in lo..hi` (one ordered chain).
#[cube]
fn chain<F: Float + CubeElement>(m_buf: &Array<F>, x: &Array<F>, base: u32, lo: u32, hi: u32, acc0: F) -> F {
    let mut acc = acc0;
    for r in lo..hi {
        acc = fma(at::<F>(m_buf, base, r), x[r as usize], acc);
    }
    acc
}

/// `symv_l_kernel`'s column sum over rows `from..to` (`to - from` a multiple
/// of 4): 8 lanes over the AVX-512 part, folded `l + l+4`, then 4 lanes, then
/// `(l0 + l2) + (l1 + l3)`.
#[cube]
fn fold8_4<F: Float + CubeElement>(m_buf: &Array<F>, x: &Array<F>, base: u32, from: u32, to: u32) -> F {
    let delta = (to - from) & !7u32;
    let to2 = from + delta;
    let mut e0 = F::from_int(0);
    let mut e1 = F::from_int(0);
    let mut e2 = F::from_int(0);
    let mut e3 = F::from_int(0);
    let mut e4 = F::from_int(0);
    let mut e5 = F::from_int(0);
    let mut e6 = F::from_int(0);
    let mut e7 = F::from_int(0);
    let mut r = from;
    while r < to2 {
        e0 = fma(at::<F>(m_buf, base, r), x[r as usize], e0);
        e1 = fma(at::<F>(m_buf, base, r + 1), x[(r + 1) as usize], e1);
        e2 = fma(at::<F>(m_buf, base, r + 2), x[(r + 2) as usize], e2);
        e3 = fma(at::<F>(m_buf, base, r + 3), x[(r + 3) as usize], e3);
        e4 = fma(at::<F>(m_buf, base, r + 4), x[(r + 4) as usize], e4);
        e5 = fma(at::<F>(m_buf, base, r + 5), x[(r + 5) as usize], e5);
        e6 = fma(at::<F>(m_buf, base, r + 6), x[(r + 6) as usize], e6);
        e7 = fma(at::<F>(m_buf, base, r + 7), x[(r + 7) as usize], e7);
        r += 8u32;
    }
    let mut l0 = e4 + e0;
    let mut l1 = e5 + e1;
    let mut l2 = e6 + e2;
    let mut l3 = e7 + e3;
    while r < to {
        l0 = fma(at::<F>(m_buf, base, r), x[r as usize], l0);
        l1 = fma(at::<F>(m_buf, base, r + 1), x[(r + 1) as usize], l1);
        l2 = fma(at::<F>(m_buf, base, r + 2), x[(r + 2) as usize], l2);
        l3 = fma(at::<F>(m_buf, base, r + 3), x[(r + 3) as usize], l3);
        r += 4u32;
    }
    (l0 + l2) + (l1 + l3)
}

/// K1ob: `y(off + i) = row i of DSYMV(UPLO, m, alpha, M[off.., off..], x, beta = 0)`,
/// one unit per row `i < m`. `m_buf` is `lda × ncols`, `x` has `m` entries,
/// `y` has `lda` (rows outside the block are untouched).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_symv_row_kernel<F: Float + CubeElement>(
    m_buf: &Array<F>,
    x: &Array<F>,
    y: &mut Array<F>,
    lda: u32,
    off: u32,
    m: u32,
    alpha: F,
    #[comptime] upper: bool,
) {
    if ABSOLUTE_POS < m as usize {
        let i = ABSOLUTE_POS as u32;
        let n = m;
        let base = (off + i) * lda + off;
        let zero = F::from_int(0);
        let mut yi = zero;
        if upper {
            // `n < 16` is `dsymv_U.c`'s plain column loop: every column is a
            // tail column (m2 = 0).
            let mut m2 = n - (n % 4u32);
            if n < 16u32 {
                m2 = 0u32;
            }
            let groups = m2 / 4u32;
            for gi in 0..groups {
                let j = gi * 4u32;
                let j1 = (j / 8u32) * 8u32;
                if i < j1 {
                    // dsymv_kernel_4x4: y = fma(a_k, t_k, y), k = 0..3.
                    for k in 0..4u32 {
                        let tk = alpha * x[(j + k) as usize];
                        yi = fma(at::<F>(m_buf, base, j + k), tk, yi);
                    }
                } else if i < j {
                    // Inlined dsymv_kernel_1x4.
                    let t0 = alpha * x[j as usize];
                    let t1 = alpha * x[(j + 1) as usize];
                    let t2 = alpha * x[(j + 2) as usize];
                    let t3 = alpha * x[(j + 3) as usize];
                    let mut s = t1 * at::<F>(m_buf, base, j + 1);
                    s = fma(at::<F>(m_buf, base, j), t0, s);
                    s = fma(at::<F>(m_buf, base, j + 2), t2, s);
                    s = fma(at::<F>(m_buf, base, j + 3), t3, s);
                    yi = s + yi;
                } else if i < j + 4u32 {
                    // Own group: diagonal step (with the column sum), then the
                    // later columns of the group.
                    let kk = i - j;
                    let mut tmp2 = zero;
                    if j1 != 0u32 {
                        tmp2 = fold4::<F>(m_buf, x, base, 0u32, j1) + tmp2;
                    }
                    if j1 < j {
                        tmp2 = tmp2 + chain::<F>(m_buf, x, base, j1, j, zero);
                    }
                    let t2 = chain::<F>(m_buf, x, base, j, i, tmp2);
                    let ti = alpha * x[i as usize];
                    yi = fma(ti, at::<F>(m_buf, base, i), alpha * t2) + yi;
                    for k in kk + 1..4u32 {
                        let tk = alpha * x[(j + k) as usize];
                        yi = fma(tk, at::<F>(m_buf, base, j + k), yi);
                    }
                }
            }
            // Tail columns j in m2..n: rows above, then the diagonal.
            let lo = select(i > m2, i, m2);
            if i >= m2 {
                let t2 = chain::<F>(m_buf, x, base, 0u32, i, zero);
                let ti = alpha * x[i as usize];
                yi = fma(ti, at::<F>(m_buf, base, i), alpha * t2) + yi;
            }
            for j in lo + select(i >= m2, 1u32, 0u32)..n {
                let tj = alpha * x[j as usize];
                yi = fma(at::<F>(m_buf, base, j), tj, yi);
            }
        } else {
            let m2 = (n / 4u32) * 4u32;
            // Groups strictly before row i's own group.
            let mut before = m2 / 4u32;
            if i < m2 {
                before = i / 4u32;
            }
            for gi in 0..before {
                let j = gi * 4u32;
                let t0 = alpha * x[j as usize];
                let t1 = alpha * x[(j + 1) as usize];
                let t2 = alpha * x[(j + 2) as usize];
                let t3 = alpha * x[(j + 3) as usize];
                let a0 = at::<F>(m_buf, base, j);
                let a1 = at::<F>(m_buf, base, j + 1);
                let a2 = at::<F>(m_buf, base, j + 2);
                let a3 = at::<F>(m_buf, base, j + 3);
                if j + 13u32 <= n && i < m2 {
                    // symv_l_kernel row form.
                    let mut s = t1 * a1;
                    s = fma(a0, t0, s);
                    s = fma(a2, t2, s);
                    s = fma(a3, t3, s);
                    yi = s + yi;
                } else {
                    yi = fma(t0, a0, yi);
                    yi = fma(t1, a1, yi);
                    yi = fma(t2, a2, yi);
                    yi = fma(t3, a3, yi);
                }
            }
            if i < m2 {
                let g = (i / 4u32) * 4u32;
                let k = i - g;
                yi = fma(alpha * x[i as usize], at::<F>(m_buf, base, i), yi);
                for kp in 0..k {
                    yi = fma(alpha * x[(g + kp) as usize], at::<F>(m_buf, base, g + kp), yi);
                }
                let mut tmp2 = chain::<F>(m_buf, x, base, i + 1, g + 4u32, zero);
                if g + 13u32 <= n {
                    if m2 > g + 4u32 {
                        tmp2 = tmp2 + fold8_4::<F>(m_buf, x, base, g + 4u32, m2);
                    }
                    tmp2 = chain::<F>(m_buf, x, base, m2, n, tmp2);
                } else {
                    tmp2 = chain::<F>(m_buf, x, base, g + 4u32, n, tmp2);
                }
                yi = fma(tmp2, alpha, yi);
            } else {
                for j in m2..i {
                    yi = fma(alpha * x[j as usize], at::<F>(m_buf, base, j), yi);
                }
                yi = fma(alpha * x[i as usize], at::<F>(m_buf, base, i), yi);
                let t2 = chain::<F>(m_buf, x, base, i + 1, n, zero);
                yi = fma(t2, alpha, yi);
            }
        }
        y[(off + i) as usize] = yi;
    }
}
