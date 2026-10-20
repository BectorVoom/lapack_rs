//! **OpenBLAS track.** The `DSTEDC` device kernels in OpenBLAS 0.3.30
//! SkylakeX order (host transcriptions: [`crate::openblas::level1::drot`],
//! [`crate::openblas::level3::dgemm`]).
//!
//! * K7rot-ob: `DLAED2`'s Givens rotations, `x' = fma(c, x, s·y)`,
//!   `y' = fma(y, c, −(s·x))` per element (every `drot_k` path computes
//!   this). The negation is a multiply by `−1` (exact; no vector unary minus).
//! * K3m-ob: the `DLAED2` gather plus `DLAED3`'s `DGEMM('N','N', rows, K, T,
//!   ONE, Q2, S, ZERO, Q)`, one scalar unit per (row, output column). The
//!   element's bits depend on the whole call's shape (`rows × K × T`): the
//!   small-matrix path (`rows·K·T <= 1e6`, `B0` build) chains or reduces
//!   eight lanes by row position; the blocked driver adds one fused kernel
//!   store per `GEMM_Q` block of `l`, with the lane mode of the row/column
//!   tile inside its `GEMM_P` row block. Both are recomputed here from the
//!   element's row inside its block and its column.

use cubecl::prelude::*;

use super::ob_syr2k::acc_mode_pub as acc_mode;

/// K7rot-ob. Rows of the lane-vectors `rv0..rv0+units` inside `[lo, hi)`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_rotations_kernel<F: Float + CubeElement, N: Size>(
    z: &mut Array<Vector<F, N>>,
    cols: &Array<u32>,
    cs: &Array<F>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda_vecs: u32,
    rv0: u32,
    units: u32,
    nrot: u32,
    lo: u32,
    hi: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let rv = rv0 + ABSOLUTE_POS as u32;
        let rows = Vector::<u32, N>::new(rv * N::value() as u32) + lane_ofs[0];
        let keep = rows.greater_equal(Vector::<u32, N>::new(lo)).and(rows.less_than(Vector::<u32, N>::new(hi)));
        let neg = Vector::<F, N>::new(F::from_int(-1));
        for t in 0..nrot {
            let px = (cols[(2 * t) as usize] * lda_vecs + rv) as usize;
            let py = (cols[(2 * t + 1) as usize] * lda_vecs + rv) as usize;
            let c = Vector::<F, N>::new(cs[(2 * t) as usize]);
            let s = Vector::<F, N>::new(cs[(2 * t + 1) as usize]);
            let x = z[px];
            let y = z[py];
            let sy = s * y;
            let sx = s * x;
            let xn = fma(c, x, sy);
            let yn = fma(y, c, sx * neg);
            z[py] = select_many(keep, yn, y);
            z[px] = select_many(keep, xn, x);
        }
    }
}

/// `GEMM_P`, `GEMM_Q`, `GEMM_UNROLL_M` of the pinned library.
const GEMM_P: u32 = 192;
const GEMM_Q: u32 = 384;
const UNROLL_M: u32 = 16;

/// Block size of `level3.c` for `rem` left along `m` (`GEMM_P`) or `k`
/// (`GEMM_Q`).
#[cube]
fn gemm_block(rem: u32, p: u32) -> u32 {
    let mut r = rem;
    if rem >= p * 2u32 {
        r = p;
    } else if rem > p {
        r = ((rem / 2u32) + UNROLL_M - 1u32) / UNROLL_M * UNROLL_M;
    }
    r
}

/// K3m-ob. See the module header. `z`, `dst` are scalar views; `dst` holds
/// `dst_rows` rows per output column from global row `dst_r0`. Units are
/// `(row in rows[..units_x]) × (column j0..n)`, flattened row-major within a
/// column.
/// `small != 0` selects the small-matrix path (`rows·K·T <= 1e6`, decided on
/// the host).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_merge_kernel<F: Float + CubeElement>(
    z: &Array<F>,
    dst: &mut Array<F>,
    perm: &Array<u32>,
    perm4: &Array<u32>,
    s: &Array<F>,
    rows: &Array<u32>,
    lda: u32,
    units_x: u32,
    units: u32,
    blk_r0: u32,
    blk_m: u32,
    k: u32,
    nterms: u32,
    lds: u32,
    s_row0: u32,
    dst_rows: u32,
    dst_r0: u32,
    j0: u32,
    small: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let r = rows[(u % units_x) as usize];
        let j = j0 + u / units_x;
        let di = (j * dst_rows + r - dst_r0) as usize;
        let zero = F::from_int(0);
        let mut val = zero;
        if j < k {
            let i = r - blk_r0;
            let m = blk_m;
            let t = nterms;
            let sj = j * lds + s_row0;
            if small != 0u32 {
                // Small kernel nn, B0 build (alpha = 1: `sum*alpha` exact).
                let m8 = m & !7u32;
                let mut dot_from = m8;
                if m - m8 > 4u32 || t < 16u32 {
                    dot_from = m;
                }
                if i < dot_from {
                    let mut acc = zero;
                    for l in 0..t {
                        acc = fma(z[(perm[l as usize] * lda + r) as usize], s[(sj + l) as usize], acc);
                    }
                    val = acc;
                } else {
                    let mut e0 = zero;
                    let mut e1 = zero;
                    let mut e2 = zero;
                    let mut e3 = zero;
                    let mut e4 = zero;
                    let mut e5 = zero;
                    let mut e6 = zero;
                    let mut e7 = zero;
                    let t8 = t & !7u32;
                    let mut l = 0u32;
                    while l < t8 {
                        e0 = fma(z[(perm[l as usize] * lda + r) as usize], s[(sj + l) as usize], e0);
                        e1 = fma(z[(perm[(l + 1) as usize] * lda + r) as usize], s[(sj + l + 1) as usize], e1);
                        e2 = fma(z[(perm[(l + 2) as usize] * lda + r) as usize], s[(sj + l + 2) as usize], e2);
                        e3 = fma(z[(perm[(l + 3) as usize] * lda + r) as usize], s[(sj + l + 3) as usize], e3);
                        e4 = fma(z[(perm[(l + 4) as usize] * lda + r) as usize], s[(sj + l + 4) as usize], e4);
                        e5 = fma(z[(perm[(l + 5) as usize] * lda + r) as usize], s[(sj + l + 5) as usize], e5);
                        e6 = fma(z[(perm[(l + 6) as usize] * lda + r) as usize], s[(sj + l + 6) as usize], e6);
                        e7 = fma(z[(perm[(l + 7) as usize] * lda + r) as usize], s[(sj + l + 7) as usize], e7);
                        l += 8u32;
                    }
                    let rem = t - t8;
                    if rem != 0u32 {
                        // Masked load: lanes >= rem add 0*0.
                        let mut lane = 0u32;
                        while lane < 8u32 {
                            let mut a = zero;
                            let mut b = zero;
                            if lane < rem {
                                a = z[(perm[(t8 + lane) as usize] * lda + r) as usize];
                                b = s[(sj + t8 + lane) as usize];
                            }
                            if lane == 0u32 {
                                e0 = fma(a, b, e0);
                            } else if lane == 1u32 {
                                e1 = fma(a, b, e1);
                            } else if lane == 2u32 {
                                e2 = fma(a, b, e2);
                            } else if lane == 3u32 {
                                e3 = fma(a, b, e3);
                            } else if lane == 4u32 {
                                e4 = fma(a, b, e4);
                            } else if lane == 5u32 {
                                e5 = fma(a, b, e5);
                            } else if lane == 6u32 {
                                e6 = fma(a, b, e6);
                            } else {
                                e7 = fma(a, b, e7);
                            }
                            lane += 1u32;
                        }
                    }
                    let m4 = m & !3u32;
                    let n4 = k & !3u32;
                    if i < m4 || j < n4 {
                        val = ((e0 + e1) + (e2 + e3)) + ((e4 + e5) + (e6 + e7));
                    } else {
                        val = ((e0 + e4) + (e2 + e6)) + ((e1 + e5) + (e3 + e7));
                    }
                }
            } else {
                // Blocked driver: beta = 0 zero fill, then one fused store per
                // `l` block. First row block: column chunks of 6/2 -> plain.
                let first = gemm_block(m, GEMM_P);
                let mut mode = 0u32;
                if i >= first {
                    let mut is = first;
                    let mut mi = gemm_block(m - is, GEMM_P);
                    while i >= is + mi {
                        is += mi;
                        mi = gemm_block(m - is, GEMM_P);
                    }
                    mode = acc_mode(i - is, mi, j, k);
                }
                let mut ls = 0u32;
                while ls < t {
                    let ml = gemm_block(t - ls, GEMM_Q);
                    let mut acc = zero;
                    let mut l = ls;
                    let end = ls + ml;
                    if mode == 1u32 && ml >= 2u32 {
                        let mut lo2 = zero;
                        let mut hi2 = zero;
                        while end - l >= 2u32 {
                            lo2 = fma(z[(perm[l as usize] * lda + r) as usize], s[(sj + l) as usize], lo2);
                            hi2 = fma(z[(perm[(l + 1) as usize] * lda + r) as usize], s[(sj + l + 1) as usize], hi2);
                            l += 2u32;
                        }
                        acc = hi2 + lo2;
                    } else if mode == 2u32 && ml >= 4u32 {
                        let mut s0 = zero;
                        let mut s1 = zero;
                        let mut s2 = zero;
                        let mut s3 = zero;
                        while end - l >= 4u32 {
                            s0 = fma(z[(perm[l as usize] * lda + r) as usize], s[(sj + l) as usize], s0);
                            s1 = fma(z[(perm[(l + 1) as usize] * lda + r) as usize], s[(sj + l + 1) as usize], s1);
                            s2 = fma(z[(perm[(l + 2) as usize] * lda + r) as usize], s[(sj + l + 2) as usize], s2);
                            s3 = fma(z[(perm[(l + 3) as usize] * lda + r) as usize], s[(sj + l + 3) as usize], s3);
                            l += 4u32;
                        }
                        acc = (s3 + s2) + (s1 + s0);
                    }
                    while l < end {
                        acc = fma(z[(perm[l as usize] * lda + r) as usize], s[(sj + l) as usize], acc);
                        l += 1u32;
                    }
                    val = fma(acc, F::from_int(1), val);
                    ls += ml;
                }
            }
        } else {
            val = z[(perm4[(j - k) as usize] * lda + r) as usize];
        }
        dst[di] = val;
    }
}

/// K3m-ob, lane-vector units for rows whose accumulation is the plain fused
/// chain (every row of the small path below `dot_from`, every row of the
/// blocked driver outside the 4/2/1 row tiles of a later `GEMM_P` block):
/// the same per-row chain as [`ob_merge_kernel`], lane-wise. Units are
/// `(lane-vector, column)`; lane-vectors with `vplain == 0` and rows outside
/// `[lo, hi)` keep their `dst` value (the scalar kernel covers them).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_merge_vec_kernel<F: Float + CubeElement, N: Size>(
    z: &Array<Vector<F, N>>,
    dst: &mut Array<Vector<F, N>>,
    perm: &Array<u32>,
    perm4: &Array<u32>,
    s: &Array<F>,
    vplain: &Array<u32>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda_vecs: u32,
    rv0: u32,
    units_x: u32,
    units: u32,
    lo: u32,
    hi: u32,
    k: u32,
    nterms: u32,
    lds: u32,
    s_row0: u32,
    dst_rvecs: u32,
    dst_rv0: u32,
    small: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let rv = rv0 + u % units_x;
        let j = u / units_x;
        if vplain[(u % units_x) as usize] != 0u32 {
            let rows = Vector::<u32, N>::new(rv * N::value() as u32) + lane_ofs[0];
            let keep = rows.greater_equal(Vector::<u32, N>::new(lo)).and(rows.less_than(Vector::<u32, N>::new(hi)));
            let di = (j * dst_rvecs + rv - dst_rv0) as usize;
            let zero = Vector::<F, N>::new(F::from_int(0));
            let mut val = zero;
            if j < k {
                let sj = j * lds + s_row0;
                if small != 0u32 {
                    for l in 0..nterms {
                        val = fma(
                            z[(perm[l as usize] * lda_vecs + rv) as usize],
                            Vector::<F, N>::new(s[(sj + l) as usize]),
                            val,
                        );
                    }
                } else {
                    let one = Vector::<F, N>::new(F::from_int(1));
                    let mut ls = 0u32;
                    while ls < nterms {
                        let ml = gemm_block(nterms - ls, GEMM_Q);
                        let mut acc = zero;
                        for l in ls..ls + ml {
                            acc = fma(
                                z[(perm[l as usize] * lda_vecs + rv) as usize],
                                Vector::<F, N>::new(s[(sj + l) as usize]),
                                acc,
                            );
                        }
                        val = fma(acc, one, val);
                        ls += ml;
                    }
                }
            } else {
                val = z[(perm4[(j - k) as usize] * lda_vecs + rv) as usize];
            }
            dst[di] = select_many(keep, val, dst[di]);
        }
    }
}
