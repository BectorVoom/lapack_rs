//! OpenBLAS 0.3.30 SkylakeX track (scipy's `libscipy_openblas-6cdc3b4a.so`,
//! `OPENBLAS_CORETYPE=SkylakeX`, `OPENBLAS_NUM_THREADS=1`): complex double level-3 BLAS
//! `ZGEMM`, `ZHER2K` (`TRANS = 'N'`) and `ZTRMM`, bit for bit.
//!
//! Transcribed (interface -> driver -> kernel symbol -> source):
//! * `ZGEMM`: `interface/gemm.c` (`scipy_zgemm_`; `zgemm_small_matrix_permit_SKYLAKEX`
//!   returns 0, so there is no small-matrix path; one thread) -> `zgemm_{nn,nt,nc,tn,tt,tc,cn,
//!   ct,cc}` (`driver/level3/level3.c`, built without `SKYLAKEX`, so `min_jj` is capped at
//!   `3*GEMM_UNROLL_N`) -> `zgemm_beta_SKYLAKEX` (`kernel/x86_64/zgemm_beta.S`),
//!   `zgemm_itcopy/incopy_SKYLAKEX` (`kernel/generic/zgemm_tcopy_4.c`/`zgemm_ncopy_4.c`),
//!   `zgemm_oncopy/otcopy_SKYLAKEX` (`zgemm_ncopy_2.c`/`zgemm_tcopy_2.c`),
//!   `zgemm_kernel_{n,l,r,b}_SKYLAKEX` (`kernel/x86_64/zgemm_kernel_4x2_skylakex.c`, inline
//!   AVX-512 asm, `ZGEMM_SKX_MODE` 0..3).
//! * `ZHER2K`: `interface/syr2k.c` (`HER2K`, `scipy_zher2k_`) -> `zher2k_UN`/`zher2k_LN`
//!   (`driver/level3/zher2k_k.c` + `level3_syr2k.c`; `syrk_beta` -> `dscal_k_SKYLAKEX`,
//!   `kernel/x86_64/dscal.c`) -> `zher2k_kernel_UN`/`zher2k_kernel_LN`
//!   (`driver/level3/zher2k_kernel.c`) -> `zgemm_kernel_r_SKYLAKEX`, `zgemm_beta_SKYLAKEX`.
//! * `ZTRMM`: `interface/trsm.c` (`TRMM`, `scipy_ztrmm_`) -> `ztrmm_{L,R}{N,T,C}{U,L}{U,N}`
//!   (`driver/level3/trmm_L.c`, `trmm_R.c`) -> `zgemm_beta_SKYLAKEX`,
//!   `zgemm_kernel_{n,l,r}_SKYLAKEX`, `ztrmm_kernel_{LN,LT,LC,RN,RT,RC}_SKYLAKEX`
//!   (`kernel/x86_64/zgemm_kernel_4x2_haswell.S` with `TRMMKERNEL`, AVX2 asm),
//!   `ztrmm_{iun,iln,iut,ilt}{u,n}copy_SKYLAKEX` (`kernel/generic/ztrmm_{un,ln,ut,lt}copy_4.c`)
//!   and `ztrmm_{oun,oln,out,olt}{u,n}copy_SKYLAKEX` (`..._2.c`).
//!
//! Every kernel is hand-written assembly (inline strings or `.S`), so the fused operations are
//! exactly the ones written there: `x.mul_add(y, acc)` is one `vfmadd*`/`vfmaddsub*`/
//! `vfmsubadd*` lane (one rounding), `a*b` followed by `+`/`-` is `vmulpd` then an add/sub.
//! Lane parities measured on the oracle machine: `vfmaddsub*` subtracts in even lanes (real
//! parts) and adds in odd lanes (imaginary parts), `vfmsubadd*` the opposite; `vaddsubpd`
//! (VEX `66 0F D0`) is `src1 - src2` in even lanes and `src1 + src2` in odd lanes.
//!
//! Each driver is written as pack -> micro-kernel -> unpack on explicit interleaved
//! (`re, im`) buffers laid out like OpenBLAS's `sa`/`sb`, so the later CubeCL port sees which
//! chains are per output element and in which order they run.
use crate::zblas::{Tr, Z, ZV};

/// `gotoblas->zgemm_p` (offset `0xd90` of `gotoblas_t`), live value of the pinned library
/// (`driver/others/parameter.c` / `param.h` SKYLAKEX); asserted by `tests/zlevel3.rs`.
pub const ZGEMM_P: usize = 256;
/// `gotoblas->zgemm_q` (offset `0xd94`).
pub const ZGEMM_Q: usize = 128;
/// `gotoblas->zgemm_r` (offset `0xd98`), computed at load time from `BUFFER_SIZE`.
pub const ZGEMM_R: usize = 16112;
/// `gotoblas->zgemm_unroll_m` (offset `0xd9c`).
pub const ZGEMM_UNROLL_M: usize = 4;
/// `gotoblas->zgemm_unroll_n` (offset `0xda0`).
pub const ZGEMM_UNROLL_N: usize = 2;
/// `gotoblas->zgemm_unroll_mn` (offset `0xda4`).
pub const ZGEMM_UNROLL_MN: usize = 4;

// =============================================================================================
// zgemm_beta_SKYLAKEX (kernel/x86_64/zgemm_beta.S)
// =============================================================================================

/// `ZGEMM_BETA(m, n, 0, beta_r, beta_i, NULL, 0, NULL, 0, c, ldc)`.
///
/// `ucomisd`/`jne` treats NaN as equal to zero, so the fill path is taken when both parts
/// compare "equal" to 0; it stores `beta_r` itself (a signed zero). Otherwise every element is
/// `re = cr*br - ci*bi`, `im = cr*bi + ci*br` with separate `mulsd` and one add/sub: element-wise.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zgemm_beta(m: usize, n: usize, beta: Z, c: ZV, ldc: usize) {
    if m == 0 || n == 0 {
        return;
    }
    let zero_r = beta.re == 0.0 || beta.re.is_nan();
    let zero_i = beta.im == 0.0 || beta.im.is_nan();
    for j in 0..n {
        let col = c.off(j * ldc);
        for i in 0..m {
            if zero_r && zero_i {
                col.set(i, Z::new(beta.re, beta.re));
            } else {
                let (cr, ci) = (col.re.at(i), col.im.at(i));
                let re = cr * beta.re - ci * beta.im;
                let im = cr * beta.im + ci * beta.re;
                col.set(i, Z::new(re, im));
            }
        }
    }
}

// =============================================================================================
// Packing: zgemm_{it,in}copy (4 rows) and zgemm_{on,ot}copy (2 columns)
// =============================================================================================

/// Row-group sizes of a `GEMM_UNROLL_M = 4` packing (and of the kernel's `M` loop):
/// `4, 4, .., 4`, then `2` if `rows & 2`, then `1` if `rows & 1`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn row_groups(rows: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut v = Vec::new();
    let mut r = 0;
    while r + 4 <= rows {
        v.push((r, 4));
        r += 4;
    }
    if rows - r >= 2 {
        v.push((r, 2));
        r += 2;
    }
    if rows - r == 1 {
        v.push((r, 1));
    }
    v.into_iter()
}

/// `zgemm_tcopy_4.c` / `zgemm_ncopy_4.c` layout: for each row group (see [`row_groups`]),
/// for `l in 0..cols`, the group's rows `(re, im)`. `get(q, l)` is `op(A)[row q, col l]`
/// (transposition only; the copies never conjugate). Pure data movement.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn pack_rows4(cols: usize, rows: usize, get: impl Fn(usize, usize) -> Z, buf: &mut [f64]) {
    let mut p = 0;
    for (r0, g) in row_groups(rows) {
        for l in 0..cols {
            for q in r0..r0 + g {
                let z = get(q, l);
                buf[p] = z.re;
                buf[p + 1] = z.im;
                p += 2;
            }
        }
    }
}

/// `zgemm_ncopy_2.c` / `zgemm_tcopy_2.c` layout: for each column pair, for `l in 0..rows`,
/// `(col j, col j+1)`; then the odd column: for `l`, `col`. `get(l, t)` is `op(B)[row l, col t]`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn pack_cols2(rows: usize, cols: usize, get: impl Fn(usize, usize) -> Z, buf: &mut [f64]) {
    let mut p = 0;
    let mut t = 0;
    while t + 2 <= cols {
        for l in 0..rows {
            for u in t..t + 2 {
                let z = get(l, u);
                buf[p] = z.re;
                buf[p + 1] = z.im;
                p += 2;
            }
        }
        t += 2;
    }
    if t < cols {
        for l in 0..rows {
            let z = get(l, t);
            buf[p] = z.re;
            buf[p + 1] = z.im;
            p += 2;
        }
    }
}

/// `GEMM_ITCOPY(m, n, A + (i0 + l0*lda), lda, buf)`: rows `i0..` of `A`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn gemm_itcopy(m: usize, n: usize, a: ZV, off: usize, lda: usize, buf: &mut [f64]) {
    pack_rows4(m, n, |q, l| a.at(off + q + l * lda), buf);
}
/// `GEMM_INCOPY(m, n, A + (l0 + i0*lda), lda, buf)`: columns `i0..` of `A` as rows.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn gemm_incopy(m: usize, n: usize, a: ZV, off: usize, lda: usize, buf: &mut [f64]) {
    pack_rows4(m, n, |q, l| a.at(off + l + q * lda), buf);
}
/// `GEMM_ONCOPY(m, n, B + (l0 + j0*ldb), ldb, buf)`: columns `j0..` of `B`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn gemm_oncopy(m: usize, n: usize, b: ZV, off: usize, ldb: usize, buf: &mut [f64]) {
    pack_cols2(m, n, |l, t| b.at(off + l + t * ldb), buf);
}
/// `GEMM_OTCOPY(m, n, B + (j0 + l0*ldb), ldb, buf)`: rows `j0..` of `B` as columns.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn gemm_otcopy(m: usize, n: usize, b: ZV, off: usize, ldb: usize, buf: &mut [f64]) {
    pack_cols2(m, n, |l, t| b.at(off + t + l * ldb), buf);
}

/// Component entry points of the copy routines (packed layouts), for component tests.
pub mod pack {
    use super::*;
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    pub unsafe fn itcopy(m: usize, n: usize, a: ZV, lda: usize, buf: &mut [f64]) {
        gemm_itcopy(m, n, a, 0, lda, buf)
    }
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    pub unsafe fn incopy(m: usize, n: usize, a: ZV, lda: usize, buf: &mut [f64]) {
        gemm_incopy(m, n, a, 0, lda, buf)
    }
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    pub unsafe fn oncopy(m: usize, n: usize, b: ZV, ldb: usize, buf: &mut [f64]) {
        gemm_oncopy(m, n, b, 0, ldb, buf)
    }
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    pub unsafe fn otcopy(m: usize, n: usize, b: ZV, ldb: usize, buf: &mut [f64]) {
        gemm_otcopy(m, n, b, 0, ldb, buf)
    }
}

// =============================================================================================
// zgemm_kernel_{n,l,r,b}_SKYLAKEX (kernel/x86_64/zgemm_kernel_4x2_skylakex.c)
// =============================================================================================

/// `ZGEMM_SKX_MODE`: which packed operand the kernel conjugates. `n` = mode 0 (NN..TT),
/// `l` = mode 1 (conjugate `a`: RN, CN, ..), `r` = mode 2 (conjugate `b`: NR, NC, ..),
/// `b` = mode 3 (both).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Conj {
    pub a: bool,
    pub b: bool,
}

/// Offset in the packed `b` block of column `t` of an `ndim`-column group, entry `l`.
#[inline(always)]
fn b_index(ndim: usize, k: usize, t: usize, l: usize) -> usize {
    if ndim == 1 { l * 2 } else { (t / 2) * k * 4 + l * 4 + (t % 2) * 2 }
}

/// `CNAME(m, n, k, alpha_r, alpha_i, A, B, C, LDC)`: `C[i,j] = alpha*sum_l A[i,l]*B[l,j] + C[i,j]`
/// over the packed blocks, with this per-element arithmetic:
///
/// * The `n` loop takes column groups of 12, 10, 8, 6, 4, 2, 1 (`COMPUTE(ndim)`); within each
///   the `m` loop takes row groups of 4 (`COMPUTE_m4`, zmm), one of 2 (`COMPUTE_m2`, ymm) and
///   one of 1 (`COMPUTE_m1`). The grouping never mixes elements: every `(i, j)` owns its
///   accumulators, all zeroed (`vpxor`, `+0`), and runs `l = 0..k` in order.
/// * Rows of an m4 group keep FOUR independent fused chains per element
///   (`sum r*br`, `sum r*bi`, `sum i*br`, `sum i*bi`) and combine them only in `SAVE`.
///   Rows of an m2/m1 group keep TWO chains (re, im) that fold the complex product into the
///   accumulator every step (two nested fused ops per part).
/// * `SAVE` adds `alpha*S` to `C` with two nested fused ops per part.
///
/// So the result is element-wise (any partition of the `(i, j)` set is safe), but each element
/// is a fixed-order chain over `l`, and which chain shape it gets depends on the element's row
/// position inside the kernel call's `M` (`M - M%4` rows are m4 rows).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zgemm_kernel(
    conj: Conj,
    m: usize,
    n: usize,
    k: usize,
    alpha: Z,
    a: &[f64],
    b: &[f64],
    c: ZV,
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let (ar, ai) = (alpha.re, alpha.im);
    let mut n_count = n;
    let mut b_base = 0usize;
    let mut c_col = 0usize; // column index of the group's first column
    for ndim in [12usize, 10, 8, 6, 4, 2, 1] {
        // `for(;n_count>ndim-1;n_count-=ndim)`, and `if(n_count>0) COMPUTE(1)` once.
        while n_count >= ndim {
            let bb = &b[b_base..];
            let mut a_base = 0usize;
            for (r0, g) in row_groups(m) {
                let ab = &a[a_base..];
                for t in 0..ndim {
                    let cc = c.off((c_col + t) * ldc + r0);
                    match g {
                        4 => kernel_m4(conj, k, ar, ai, ab, bb, ndim, t, cc),
                        _ => kernel_m21(conj, g, k, ar, ai, ab, bb, ndim, t, cc),
                    }
                }
                a_base += g * k * 2;
            }
            b_base += ndim * k * 2;
            c_col += ndim;
            n_count -= ndim;
        }
    }
}

/// One column `t` of `COMPUTE_m4(ndim)`: rows `q = 0..4` of the group (`KERNEL_t_k1m4n*` +
/// `SAVE_m4n*`).
#[inline(always)]
unsafe fn kernel_m4(conj: Conj, k: usize, ar: f64, ai: f64, a: &[f64], b: &[f64], ndim: usize, t: usize, c: ZV) {
    // zmm8.. (c_le): lanes (2q, 2q+1) = (sum r_q*br, sum r_q*bi);
    // zmm9.. (c_ri): lanes (2q, 2q+1) = (sum i_q*br, sum i_q*bi), negated terms if conj a.
    let mut le = [[0.0f64; 2]; 4];
    let mut ri = [[0.0f64; 2]; 4];
    for l in 0..k {
        let bo = b_index(ndim, k, t, l);
        let (br, bi) = (b[bo], b[bo + 1]);
        for q in 0..4 {
            let (r, i) = (a[l * 8 + 2 * q], a[l * 8 + 2 * q + 1]);
            // vmovddup: zmm4 = (r_q, r_q), zmm5 = (i_q, i_q); zmm3 = (br, bi) broadcast.
            // vfmadd231pd zmm4, zmm3, c_le
            le[q][0] = br.mul_add(r, le[q][0]);
            le[q][1] = bi.mul_add(r, le[q][1]);
            if !conj.a {
                // vfmadd231pd zmm5, zmm3, c_ri
                ri[q][0] = br.mul_add(i, ri[q][0]);
                ri[q][1] = bi.mul_add(i, ri[q][1]);
            } else {
                // vfnmadd231pd zmm5, zmm3, c_ri
                ri[q][0] = (-br).mul_add(i, ri[q][0]);
                ri[q][1] = (-bi).mul_add(i, ri[q][1]);
            }
        }
    }
    for q in 0..4 {
        let cz = c.at(q);
        // vpermilpd $85, c_ri, c_ri
        let sw = [ri[q][1], ri[q][0]];
        let (re, im);
        if !conj.b {
            // vfmadd231pd c_ri, zmm2(-1,1), c_le
            let s_re = sw[0].mul_add(-1.0, le[q][0]);
            let s_im = sw[1].mul_add(1.0, le[q][1]);
            // vpermilpd $85, c_le, zmm4; vfmaddsub213pd (C), zmm1, zmm4
            let z_re = ai.mul_add(s_im, -cz.re);
            let z_im = ai.mul_add(s_re, cz.im);
            // vfmaddsub213pd zmm4, zmm0, c_le
            re = ar.mul_add(s_re, -z_re);
            im = ar.mul_add(s_im, z_im);
        } else {
            // vfnmadd231pd c_ri, zmm2(-1,1), c_le
            let s_re = (-sw[0]).mul_add(-1.0, le[q][0]);
            let s_im = (-sw[1]).mul_add(1.0, le[q][1]);
            let (z_re, z_im) = (s_im, s_re); // vpermilpd $85, c_le, zmm4
            // vfmsubadd213pd (C), zmm0, c_le
            let x_re = ar.mul_add(s_re, cz.re);
            let x_im = ar.mul_add(s_im, -cz.im);
            // vfmsubadd231pd zmm4, zmm1, c_le
            re = ai.mul_add(z_re, x_re);
            im = ai.mul_add(z_im, -x_im);
        }
        c.set(q, Z::new(re, im));
    }
}

/// One column `t` of `COMPUTE_m2(ndim)` (`g = 2`) or `COMPUTE_m1(ndim)` (`g = 1`).
#[inline(always)]
unsafe fn kernel_m21(
    conj: Conj,
    g: usize,
    k: usize,
    ar: f64,
    ai: f64,
    a: &[f64],
    b: &[f64],
    ndim: usize,
    t: usize,
    c: ZV,
) {
    let mut acc = [[0.0f64; 2]; 2];
    for l in 0..k {
        let bo = b_index(ndim, k, t, l);
        let (br, bi) = (b[bo], b[bo + 1]);
        for q in 0..g {
            let (r, i) = (a[l * 2 * g + 2 * q], a[l * 2 * g + 2 * q + 1]);
            let s = &mut acc[q];
            // a0 = (r, i), ap = vpermilpd $5 = (i, r); first with bi, then with br.
            if conj.a == conj.b {
                // vfmaddsub231pd ap, bi, c; vfmaddsub231pd a0, br, c
                let t_re = bi.mul_add(i, -s[0]);
                let t_im = bi.mul_add(r, s[1]);
                s[0] = br.mul_add(r, -t_re);
                s[1] = br.mul_add(i, t_im);
            } else {
                // vfmsubadd231pd ap, bi, c; vfmsubadd231pd a0, br, c
                let t_re = bi.mul_add(i, s[0]);
                let t_im = bi.mul_add(r, -s[1]);
                s[0] = br.mul_add(r, t_re);
                s[1] = br.mul_add(i, -t_im);
            }
        }
    }
    for (q, s) in acc.iter().enumerate().take(g) {
        let cz = c.at(q);
        let (re, im);
        if !conj.a {
            // vpermilpd $5, c, y3; vfmaddsub213pd (C), alpha_i, y3; vfmaddsub213pd y3, alpha_r, c
            let y_re = ai.mul_add(s[1], -cz.re);
            let y_im = ai.mul_add(s[0], cz.im);
            re = ar.mul_add(s[0], -y_re);
            im = ar.mul_add(s[1], y_im);
        } else {
            // vpermilpd $5, c, y3; vfmsubadd213pd (C), alpha_r, c; vfmsubadd231pd y3, alpha_i, c
            let x_re = ar.mul_add(s[0], cz.re);
            let x_im = ar.mul_add(s[1], -cz.im);
            re = ai.mul_add(s[1], x_re);
            im = ai.mul_add(s[0], -x_im);
        }
        c.set(q, Z::new(re, im));
    }
}

// =============================================================================================
// ZGEMM: interface/gemm.c + driver/level3/level3.c
// =============================================================================================

/// `min_i`/`min_l` block rule of `level3.c`: `>= 2*P -> P`, `> P -> ((x/2 + U - 1)/U)*U`.
#[inline(always)]
fn half_block(x: usize, p: usize, u: usize) -> usize {
    if x >= 2 * p {
        p
    } else if x > p {
        (x / 2).div_ceil(u) * u
    } else {
        x
    }
}

/// `ZGEMM` through the Fortran interface (`OPENBLAS_NUM_THREADS=1`).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zgemm(
    transa: Tr,
    transb: Tr,
    m: usize,
    n: usize,
    k: usize,
    alpha: Z,
    a: ZV,
    lda: usize,
    b: ZV,
    ldb: usize,
    beta: Z,
    c: ZV,
    ldc: usize,
) {
    // interface/gemm.c: `if ((args.m == 0) || (args.n == 0)) return;`
    if m == 0 || n == 0 {
        return;
    }
    // level3.c: BETA_OPERATION over the whole of C.
    if beta.re != 1.0 || beta.im != 0.0 {
        zgemm_beta(m, n, beta, c, ldc);
    }
    if k == 0 {
        return;
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }
    let conj = Conj { a: transa == Tr::C, b: transb == Tr::C };
    let a_trans = transa != Tr::N;
    let b_trans = transb != Tr::N;
    let (p, q, r, um, un) = (ZGEMM_P, ZGEMM_Q, ZGEMM_R, ZGEMM_UNROLL_M, ZGEMM_UNROLL_N);

    let mut sa = vec![0.0f64; 2 * p.min(m.max(1)) * q.min(k)];
    let mut sb = vec![0.0f64; 2 * r.min(n) * q.min(k)];

    let mut js = 0;
    while js < n {
        let min_j = (n - js).min(r);
        let mut ls = 0;
        while ls < k {
            let min_l = half_block(k - ls, q, um);
            // First, pack rows 0..min_i of op(A) ("move data A to L2 cache").
            let mut min_i = m;
            let mut l1stride = 1;
            if min_i >= 2 * p {
                min_i = p;
            } else if min_i > p {
                min_i = (min_i / 2).div_ceil(um) * um;
            } else {
                l1stride = 0;
            }
            icopy(a_trans, min_l, min_i, a, lda, ls, 0, &mut sa);
            let mut jjs = js;
            while jjs < js + min_j {
                let mut min_jj = min_j + js - jjs;
                if min_jj >= 3 * un {
                    min_jj = 3 * un;
                } else if min_jj > un {
                    min_jj = un;
                }
                let sbo = min_l * (jjs - js) * 2 * l1stride;
                ocopy(b_trans, min_l, min_jj, b, ldb, ls, jjs, &mut sb[sbo..]);
                zgemm_kernel(conj, min_i, min_jj, min_l, alpha, &sa, &sb[sbo..], c.off(jjs * ldc), ldc);
                jjs += min_jj;
            }
            let mut is = min_i;
            while is < m {
                let min_i = half_block(m - is, p, um);
                icopy(a_trans, min_l, min_i, a, lda, ls, is, &mut sa);
                zgemm_kernel(conj, min_i, min_j, min_l, alpha, &sa, &sb, c.off(is + js * ldc), ldc);
                is += min_i;
            }
            ls += min_l;
        }
        js += r;
    }
}

/// `ICOPY_OPERATION(min_l, min_i, a, lda, ls, is, sa)`.
#[inline(always)]
unsafe fn icopy(trans: bool, min_l: usize, min_i: usize, a: ZV, lda: usize, ls: usize, is: usize, sa: &mut [f64]) {
    if !trans {
        gemm_itcopy(min_l, min_i, a, is + ls * lda, lda, sa);
    } else {
        gemm_incopy(min_l, min_i, a, ls + is * lda, lda, sa);
    }
}

/// `OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs, buf)`.
#[inline(always)]
unsafe fn ocopy(trans: bool, min_l: usize, min_jj: usize, b: ZV, ldb: usize, ls: usize, jjs: usize, buf: &mut [f64]) {
    if !trans {
        gemm_oncopy(min_l, min_jj, b, ls + jjs * ldb, ldb, buf);
    } else {
        gemm_otcopy(min_l, min_jj, b, jjs + ls * ldb, ldb, buf);
    }
}

// =============================================================================================
// ZHER2K (TRANS = 'N'): interface/syr2k.c + driver/level3/zher2k_k.c + level3_syr2k.c
// =============================================================================================

/// `dscal_k_SKYLAKEX(n, 0, 0, da, x, 1, NULL, 0, NULL, 0)` (flag `dummy2 = 0`): `da == 0`
/// stores `+0.0`, otherwise `x * da`; element-wise.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn dscal_flag0(n: usize, da: f64, x: &mut [f64]) {
    for v in x.iter_mut().take(n) {
        *v = if da == 0.0 { 0.0 } else { *v * da };
    }
}

/// `syrk_beta(0, n, 0, n, beta, c, ldc)` of `zher2k_k.c`: `SCAL_K` over the stored triangle of
/// each column (both parts, as one `2*len` double vector), then `Im(C[j,j]) = ZERO`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn her2k_beta(upper: bool, n: usize, beta: f64, c: ZV, ldc: usize) {
    // Work on the interleaved view the C code sees, one column at a time.
    let mut tmp = Vec::with_capacity(2 * n);
    for i in 0..n {
        let (r0, len) = if upper { (0, (i + 1).min(n)) } else { (i, n - i) };
        let col = c.off(r0 + i * ldc);
        tmp.clear();
        for t in 0..len {
            let z = col.at(t);
            tmp.push(z.re);
            tmp.push(z.im);
        }
        dscal_flag0(2 * len, beta, &mut tmp);
        for t in 0..len {
            col.set(t, Z::new(tmp[2 * t], tmp[2 * t + 1]));
        }
        // upper: `if (i + 1 <= n) c[i*2 + 1] = ZERO`; lower: `c[1] = ZERO` at the diagonal.
        let d = c.off(i + i * ldc);
        d.im.set(0, 0.0);
    }
}

/// `zher2k_kernel_{UN,LN}(m, n, k, alpha_r, alpha_i, a, b, c, ldc, offset, flag)`
/// (`driver/level3/zher2k_kernel.c`, `GEMM_KERNEL = zgemm_kernel_r`, conjugating `b`).
///
/// Splits the `m x n` block at `c` (whose element `(i, j)` is `C[i + offset + j0, j + j0]`,
/// i.e. `offset = row - column` of its corner) into the part strictly inside the stored
/// triangle, handed to the GEMM kernel as is, and the `GEMM_UNROLL_MN` diagonal blocks. Those
/// are computed into a zeroed `nn x nn` subbuffer by the GEMM kernel and folded as
/// `C += (S[i,j] + S[j,i])` (real) / `C += (S[i,j] - S[j,i])` (imag, off-diagonal) with the
/// diagonal imaginary part set to `ZERO` (only when `flag`, i.e. the first of the two passes).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn her2k_kernel(
    upper: bool,
    mut m: isize,
    mut n: isize,
    k: usize,
    alpha: Z,
    a: &[f64],
    b: &[f64],
    c: ZV,
    ldc: usize,
    mut offset: isize,
    flag: bool,
) {
    let conj = Conj { a: false, b: true };
    let (ku, mn) = (k as isize, ZGEMM_UNROLL_MN as isize);
    let gk = |m: isize, n: isize, a: &[f64], b: &[f64], c: ZV| {
        debug_assert!(m >= 0 && n >= 0);
        zgemm_kernel(conj, m as usize, n as usize, k, alpha, a, b, c, ldc)
    };
    let (mut a, mut b, mut c) = (a, b, c);
    if m + offset < 0 {
        if upper {
            gk(m, n, a, b, c);
        }
        return;
    }
    if n < offset {
        if !upper {
            gk(m, n, a, b, c);
        }
        return;
    }
    if offset > 0 {
        if !upper {
            gk(m, offset, a, b, c);
        }
        b = &b[(offset * ku * 2) as usize..];
        c = c.off(offset as usize * ldc);
        n -= offset;
        offset = 0;
        if n <= 0 {
            return;
        }
    }
    if n > m + offset {
        if upper {
            let s = (m + offset) as usize;
            gk(m, n - m - offset, a, &b[s * k * 2..], c.off(s * ldc));
        }
        n = m + offset;
        if n <= 0 {
            return;
        }
    }
    if offset < 0 {
        if upper {
            gk(-offset, n, a, b, c);
        }
        a = &a[(-offset * ku * 2) as usize..];
        c = c.off((-offset) as usize);
        m += offset;
        offset = 0;
        if m <= 0 {
            return;
        }
    }
    if m > n - offset {
        if !upper {
            let s = (n - offset) as usize;
            gk(m - n + offset, n, &a[s * k * 2..], b, c.off(s));
        }
        m = n + offset;
        if m <= 0 {
            return;
        }
    }
    let _ = offset;
    let mut sub_re = vec![0.0f64; ZGEMM_UNROLL_MN * ZGEMM_UNROLL_MN];
    let mut sub_im = vec![0.0f64; ZGEMM_UNROLL_MN * ZGEMM_UNROLL_MN];
    let mut lp = 0isize;
    while lp < n {
        let mm = (lp / mn) * mn;
        let nn = mn.min(n - lp);
        let (lpu, nnu) = (lp as usize, nn as usize);
        if upper {
            gk(mm, nn, a, &b[lpu * k * 2..], c.off(lpu * ldc));
        }
        if flag {
            let sub = ZV::new(crate::ptr::V::new(&mut sub_re), crate::ptr::V::new(&mut sub_im));
            zgemm_beta(nnu, nnu, Z::new(0.0, 0.0), sub, nnu);
            zgemm_kernel(conj, nnu, nnu, k, alpha, &a[lpu * k * 2..], &b[lpu * k * 2..], sub, nnu);
            for j in 0..nnu {
                let (lo, hi) = if upper { (0, j + 1) } else { (j, nnu) };
                for i in lo..hi {
                    let p = (i + lpu) + (j + lpu) * ldc;
                    let (sij, sji) = (sub.at(i + j * nnu), sub.at(j + i * nnu));
                    c.re.set(p, (sij.re + sji.re) + c.re.at(p));
                    if i != j {
                        c.im.set(p, (sij.im - sji.im) + c.im.at(p));
                    } else {
                        c.im.set(p, 0.0);
                    }
                }
            }
        }
        if !upper {
            let s = (mm + nn) as usize;
            gk(m - mm - nn, nn, &a[s * k * 2..], &b[lpu * k * 2..], c.off(s + lpu * ldc));
        }
        lp += mn;
    }
}

/// `ZHER2K` with `TRANS = 'N'` through the Fortran interface (`beta` real).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zher2k_n(
    upper: bool,
    n: usize,
    k: usize,
    alpha: Z,
    a: ZV,
    lda: usize,
    b: ZV,
    ldb: usize,
    beta: f64,
    c: ZV,
    ldc: usize,
) {
    if n == 0 {
        return;
    }
    // level3_syr2k.c (HER2K): `if (beta[0] != ONE) syrk_beta(..)`.
    if beta != 1.0 {
        her2k_beta(upper, n, beta, c, ldc);
    }
    if k == 0 {
        return;
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }
    let (p, q, r, mnu) = (ZGEMM_P, ZGEMM_Q, ZGEMM_R, ZGEMM_UNROLL_MN);
    let alpha_c = Z::new(alpha.re, -alpha.im); // KERNEL_OPERATION_C
    let min_i_rule = |x: usize| -> usize {
        if x >= 2 * p {
            p
        } else if x > p {
            (x / 2).div_ceil(mnu) * mnu
        } else {
            x
        }
    };
    let mut sa = vec![0.0f64; 2 * p.min(n) * q.min(k)];
    let mut sb = vec![0.0f64; 2 * r.min(n) * q.min(k)];
    // ICOPY = GEMM_ITCOPY(min_l, min_i, X + (is + ls*ldx)), OCOPY = GEMM_OTCOPY(min_l, cols,
    // Y + (j0 + ls*ldy)).
    let icopy = |x: ZV, ldx: usize, min_l: usize, min_i: usize, ls: usize, is: usize, sa: &mut [f64]| {
        gemm_itcopy(min_l, min_i, x, is + ls * ldx, ldx, sa)
    };
    let ocopy = |y: ZV, ldy: usize, min_l: usize, cols: usize, ls: usize, j0: usize, buf: &mut [f64]| {
        gemm_otcopy(min_l, cols, y, j0 + ls * ldy, ldy, buf)
    };
    // KERNEL_OPERATION(M, N, K, alpha, sa, sb, c, ldc, X, Y, FLAG).
    let kop = |upper: bool,
               mm: usize,
               nn: usize,
               min_l: usize,
               al: Z,
               sa: &[f64],
               sb: &[f64],
               x: usize,
               y: usize,
               flag: bool| {
        her2k_kernel(
            upper,
            mm as isize,
            nn as isize,
            min_l,
            al,
            sa,
            sb,
            c.off(x + y * ldc),
            ldc,
            x as isize - y as isize,
            flag,
        )
    };

    let mut js = 0;
    while js < n {
        let min_j = (n - js).min(r);
        let (m_start, m_end) = if upper { (0, (js + min_j).min(n)) } else { (js, n) };
        let mut ls = 0;
        while ls < k {
            let mut min_l = k - ls;
            if min_l >= 2 * q {
                min_l = q;
            } else if min_l > q {
                min_l = min_l.div_ceil(2); // `(min_l + 1) / 2`
            }
            // Pass 1: (A, conj B, alpha, flag 1); pass 2: (B, conj A, conj alpha, flag 0).
            for (x, ldx, y, ldy, al, flag) in [(a, lda, b, ldb, alpha, true), (b, ldb, a, lda, alpha_c, false)] {
                let min_i = min_i_rule(m_end - m_start);
                if upper {
                    let mut jjs;
                    if m_start >= js {
                        icopy(x, ldx, min_l, min_i, ls, m_start, &mut sa);
                        let aa = min_l * (m_start - js) * 2;
                        ocopy(y, ldy, min_l, min_i, ls, m_start, &mut sb[aa..]);
                        kop(true, min_i, min_i, min_l, al, &sa, &sb[aa..], m_start, m_start, flag);
                        jjs = m_start + min_i;
                    } else {
                        icopy(x, ldx, min_l, min_i, ls, m_start, &mut sa);
                        jjs = js;
                    }
                    while jjs < js + min_j {
                        let min_jj = (min_j + js - jjs).min(mnu);
                        let o = min_l * (jjs - js) * 2;
                        ocopy(y, ldy, min_l, min_jj, ls, jjs, &mut sb[o..]);
                        kop(true, min_i, min_jj, min_l, al, &sa, &sb[o..], m_start, jjs, flag);
                        jjs += mnu;
                    }
                    let mut is = m_start + min_i;
                    while is < m_end {
                        let min_i = min_i_rule(m_end - is);
                        icopy(x, ldx, min_l, min_i, ls, is, &mut sa);
                        kop(true, min_i, min_j, min_l, al, &sa, &sb, is, js, flag);
                        is += min_i;
                    }
                } else {
                    let aa = min_l * (m_start - js) * 2;
                    icopy(x, ldx, min_l, min_i, ls, m_start, &mut sa);
                    ocopy(y, ldy, min_l, min_i, ls, m_start, &mut sb[aa..]);
                    kop(
                        false,
                        min_i,
                        min_i.min(min_j + js - m_start),
                        min_l,
                        al,
                        &sa,
                        &sb[aa..],
                        m_start,
                        m_start,
                        flag,
                    );
                    let mut jjs = js;
                    while jjs < m_start {
                        let min_jj = (m_start - jjs).min(mnu);
                        let o = min_l * (jjs - js) * 2;
                        ocopy(y, ldy, min_l, min_jj, ls, jjs, &mut sb[o..]);
                        kop(false, min_i, min_jj, min_l, al, &sa, &sb[o..], m_start, jjs, flag);
                        jjs += mnu;
                    }
                    let mut is = m_start + min_i;
                    while is < m_end {
                        let min_i = min_i_rule(m_end - is);
                        let aa = min_l * (is - js) * 2;
                        if is < js + min_j {
                            icopy(x, ldx, min_l, min_i, ls, is, &mut sa);
                            ocopy(y, ldy, min_l, min_i, ls, is, &mut sb[aa..]);
                            kop(false, min_i, min_i.min(min_j - is + js), min_l, al, &sa, &sb[aa..], is, is, flag);
                            kop(false, min_i, is - js, min_l, al, &sa, &sb, is, js, flag);
                        } else {
                            icopy(x, ldx, min_l, min_i, ls, is, &mut sa);
                            kop(false, min_i, min_j, min_l, al, &sa, &sb, is, js, flag);
                        }
                        is += min_i;
                    }
                }
            }
            ls += min_l;
        }
        js += r;
    }
}

// =============================================================================================
// ZTRMM packing: ztrmm_{i,o}{un,ln,ut,lt}{u,n}copy (kernel/generic/ztrmm_*copy_{4,2}.c)
// =============================================================================================

/// Which triangular copy routine: `upper`, `trans` (`t` = row form `A[u, l]`, `n` = column
/// form `A[l, u]`), `unit`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TriCopy {
    pub upper: bool,
    pub trans: bool,
    pub unit: bool,
}

/// `l`-groups the copy routines walk for a unit group of size `g`: for `g = 4` (only in the
/// `_4` copies) groups of 4 and then ONE tail group of `m & 3`; for `g = 2` pairs and a single;
/// for `g = 1` singles.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn l_groups(m: usize, g: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut t = 0;
    match g {
        4 => {
            while t + 4 <= m {
                v.push((t, 4));
                t += 4;
            }
            if t < m {
                v.push((t, m - t));
            }
        }
        2 => {
            while t + 2 <= m {
                v.push((t, 2));
                t += 2;
            }
            if t < m {
                v.push((t, 1));
            }
        }
        _ => v.extend((0..m).map(|t| (t, 1))),
    }
    v
}

/// `ZTRMM_{I,O}{UN,LN,UT,LT}{U,N}COPY(m, n, a, lda, posX, posY, b)`, `width` 4 (inner, `_4.c`)
/// or 2 (outer, `_2.c`).
///
/// Packs `n` "units" `u = posY..` (rows of op(A) for the inner copies, columns for the outer
/// ones) against `m` entries `l = posX..`, in the GEMM layout (unit groups of `width`, then 2,
/// then 1; per `l`, the group's units). Element `(u, l)` is `A[u, l]` (`trans`) or `A[l, u]`.
/// The routines decide per (unit group start `Y`, `l`-group start `X`), not per element:
/// for the `l >= u` triangles (UT, LN) `X < Y` skips the block (slots NOT written), `X > Y`
/// copies it whole, `X == Y` writes the diagonal block with `ZERO` where `t < q`
/// (`t`, `q` = offsets inside the block) and `ONE` on `t == q` if `unit`; the `l <= u` triangles
/// (LT, UN) mirror it. Pure data movement (and constants).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn trmm_copy(
    cp: TriCopy,
    width: usize,
    m: usize,
    n: usize,
    a: ZV,
    lda: usize,
    pos_x: usize,
    pos_y: usize,
    buf: &mut [f64],
) {
    let ge = cp.upper == cp.trans; // UT and LN hold l >= u
    let mut p = 0usize;
    let mut groups = Vec::new();
    let mut u = 0;
    while u + width <= n {
        groups.push((u, width));
        u += width;
    }
    if width == 4 && n - u >= 2 {
        groups.push((u, 2));
        u += 2;
    }
    if n - u == 1 {
        groups.push((u, 1));
    }
    for (u0, g) in groups {
        let y = pos_y + u0;
        for (t0, h) in l_groups(m, g) {
            let x = pos_x + t0;
            if (ge && x < y) || (!ge && x > y) {
                p += 2 * h * g;
                continue;
            }
            for t in 0..h {
                for q in 0..g {
                    let (uu, ll) = (y + q, x + t);
                    let z = if x != y {
                        if width == 4 && g == 1 && !cp.upper && !cp.trans && pos_x <= pos_y {
                            // ztrmm_lncopy_4.c `n & 1` tail: the diagonal step does
                            // `ao1 += lda` (not `+= 2`), so past the diagonal it reads
                            // A[l - 1, u + 1]. The drivers never read these slots (the single
                            // unit is always the last row of its l range).
                            a.at((ll - 1) + (uu + 1) * lda)
                        } else if cp.trans {
                            a.at(uu + ll * lda)
                        } else {
                            a.at(ll + uu * lda)
                        }
                    } else if t == q {
                        if cp.unit {
                            Z::new(1.0, 0.0)
                        } else if cp.trans {
                            a.at(uu + ll * lda)
                        } else {
                            a.at(ll + uu * lda)
                        }
                    } else if (ge && t < q) || (!ge && t > q) {
                        Z::new(0.0, 0.0)
                    } else if cp.trans {
                        a.at(uu + ll * lda)
                    } else {
                        a.at(ll + uu * lda)
                    };
                    buf[p] = z.re;
                    buf[p + 1] = z.im;
                    p += 2;
                }
            }
        }
    }
}

/// Component entry point of the triangular copies, for component tests.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn trmm_copy_component(
    cp: TriCopy,
    width: usize,
    m: usize,
    n: usize,
    a: ZV,
    lda: usize,
    pos_x: usize,
    pos_y: usize,
    buf: &mut [f64],
) {
    trmm_copy(cp, width, m, n, a, lda, pos_x, pos_y, buf)
}

// =============================================================================================
// ztrmm_kernel_{LN,LT,LR,LC,RN,RT,RR,RC}_SKYLAKEX (kernel/x86_64/zgemm_kernel_4x2_haswell.S,
// TRMMKERNEL section)
// =============================================================================================

/// Conjugation define of a TRMM kernel: `NN` (LN, LT, RN, RT), `CN` (LR, LC: conjugate the
/// packed `a`), `NC` (RR, RC: conjugate the packed `b`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrmmConj {
    NN,
    CN,
    NC,
}

/// `ztrmm_kernel_*(m, n, k, alpha_r, alpha_i, a, b, c, ldc, offset)`.
///
/// Same packed layouts as the GEMM kernel, but `C[i,j] = alpha*S` (no `+ C`) and each element
/// sums only part of `l = 0..k`: `KK = offset` (`LEFT`) or `-offset`, advanced by the row
/// group size (`LEFT`, per row group, reset per column group) or by the column group size
/// (`!LEFT`, per column group). `LEFT && !TRANSA` / `!LEFT && TRANSA` sum `l = KK..k`, the other
/// two `l = 0..KK + g` with `g` the row (`LEFT`) or column (`!LEFT`) group size. Groups: column
/// pairs then one single column; rows 4, 2, 1 as in [`row_groups`].
///
/// Per element: two chains `R = sum a*br` and `I = sum a*bi` over both parts of `a` (fused,
/// `vfnmadd` for the conjugated operand's part), then with separate `vmulpd`:
/// `s = (R.re -+ I.im, R.im +- I.re)`, `out = (s.re*ar - s.im*ai, s.im*ar + s.re*ai)`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn trmm_kernel(
    left: bool,
    transa: bool,
    conj: TrmmConj,
    m: usize,
    n: usize,
    k: usize,
    alpha: Z,
    a: &[f64],
    b: &[f64],
    c: ZV,
    ldc: usize,
    offset: isize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let ki = k as isize;
    let mut kk = if left { offset } else { -offset };
    let tail_first = (left && !transa) || (!left && transa); // sums l = KK..k
    let groups: Vec<(usize, usize)> = row_groups(m).collect();
    let mut b_base = 0usize;
    let mut col = 0usize;
    for w in [2usize, 1] {
        let reps = if w == 2 { n / 2 } else { n % 2 };
        for _ in 0..reps {
            // `.L2_00_02b`: the column group's packed b is copied to BUFFER1 first.
            let bb = &b[b_base..b_base + 2 * w * k];
            if left {
                kk = offset;
            }
            let mut ao = 0usize;
            for &(r0, g) in &groups {
                let (start, count) =
                    if tail_first { (kk, ki - kk) } else { (0, kk + if left { g as isize } else { w as isize }) };
                debug_assert!(start >= 0 && count >= 0 && start + count <= ki, "trmm kernel range");
                let (start, count) = (start as usize, count as usize);
                let ab = &a[ao..ao + 2 * g * k];
                for t in 0..w {
                    let cc = c.off(r0 + (col + t) * ldc);
                    for q in 0..g {
                        // R = (re lane, im lane) of a*br; I = of a*bi.
                        let (mut rr, mut ri, mut ir, mut ii) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                        for l in start..start + count {
                            let (ar_, ai_) = (ab[l * 2 * g + 2 * q], ab[l * 2 * g + 2 * q + 1]);
                            let (br, bi) = (bb[l * 2 * w + 2 * t], bb[l * 2 * w + 2 * t + 1]);
                            match conj {
                                TrmmConj::NN => {
                                    rr = ar_.mul_add(br, rr);
                                    ri = ai_.mul_add(br, ri);
                                    ir = ar_.mul_add(bi, ir);
                                    ii = ai_.mul_add(bi, ii);
                                }
                                TrmmConj::CN => {
                                    rr = (-ar_).mul_add(br, rr);
                                    ri = (-ai_).mul_add(br, ri);
                                    ir = ar_.mul_add(bi, ir);
                                    ii = ai_.mul_add(bi, ii);
                                }
                                TrmmConj::NC => {
                                    rr = ar_.mul_add(br, rr);
                                    ri = ai_.mul_add(br, ri);
                                    ir = (-ar_).mul_add(bi, ir);
                                    ii = (-ai_).mul_add(bi, ii);
                                }
                            }
                        }
                        // SAVE: vshufpd swaps I to (ii, ir); vaddsubpd (even: src1 - src2,
                        // odd: src1 + src2).
                        let (s_re, s_im) = match conj {
                            TrmmConj::NN | TrmmConj::NC => (rr - ii, ri + ir),
                            TrmmConj::CN => (ii - rr, ir + ri),
                        };
                        let (x_re, x_im) = (s_re * alpha.re, s_im * alpha.re);
                        let (y_re, y_im) = (s_im * alpha.im, s_re * alpha.im);
                        cc.set(q, Z::new(x_re - y_re, x_im + y_im));
                    }
                }
                ao += 2 * g * k;
                if left {
                    kk += g as isize;
                }
            }
            if !left {
                kk += w as isize;
            }
            b_base += 2 * w * k;
            col += w;
        }
    }
}

// =============================================================================================
// ZTRMM: interface/trsm.c (TRMM) + driver/level3/trmm_L.c, trmm_R.c
// =============================================================================================

/// `min_jj` of the `jjs` loops (driver built without `SKYLAKEX`): `>= 3*UN -> 3*UN`,
/// `> UN -> UN`.
#[inline(always)]
fn chunk_jj(x: usize) -> usize {
    let un = ZGEMM_UNROLL_N;
    if x >= 3 * un {
        3 * un
    } else if x > un {
        un
    } else {
        x
    }
}

/// `min_i` rule of `trmm_L.c`: `min(x, P)`, then floored to a multiple of `UNROLL_M` if larger.
#[inline(always)]
fn floor_m(x: usize) -> usize {
    let v = x.min(ZGEMM_P);
    if v > ZGEMM_UNROLL_M { (v / ZGEMM_UNROLL_M) * ZGEMM_UNROLL_M } else { v }
}

/// `ZTRMM` through the Fortran interface (`OPENBLAS_NUM_THREADS=1`).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn ztrmm(
    left: bool,
    upper: bool,
    trans: Tr,
    nounit: bool,
    m: usize,
    n: usize,
    alpha: Z,
    a: ZV,
    lda: usize,
    b: ZV,
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    // trmm_L.c / trmm_R.c: `beta` is ZTRMM's alpha.
    if alpha.re != 1.0 || alpha.im != 0.0 {
        zgemm_beta(m, n, alpha, b, ldb);
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }
    let transa = trans != Tr::N;
    let conj = trans == Tr::C;
    let one = Z::new(1.0, 0.0);
    if left {
        trmm_l(upper, transa, conj, !nounit, m, n, one, a, lda, b, ldb);
    } else {
        trmm_r(upper, transa, conj, !nounit, m, n, one, a, lda, b, ldb);
    }
}

/// `driver/level3/trmm_L.c` (`B := op(A) B`, `A` is `m x m`).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn trmm_l(
    upper: bool,
    transa: bool,
    conj: bool,
    unit: bool,
    m: usize,
    n: usize,
    one: Z,
    a: ZV,
    lda: usize,
    b: ZV,
    ldb: usize,
) {
    let (q, r) = (ZGEMM_Q, ZGEMM_R);
    let gconj = Conj { a: conj, b: false }; // GEMM_KERNEL_L when CONJ
    let tconj = if conj { TrmmConj::CN } else { TrmmConj::NN };
    let mut sa = vec![0.0f64; 2 * ZGEMM_P.min(m) * q.min(m)];
    let mut sb = vec![0.0f64; 2 * r.min(n) * q.min(m)];
    // GEMM_ITCOPY / GEMM_INCOPY of A: rows `i0..` against columns `l0..` of op(A).
    let gcopy = |min_l: usize, min_i: usize, l0: usize, i0: usize, sa: &mut [f64]| {
        if !transa {
            gemm_itcopy(min_l, min_i, a, i0 + l0 * lda, lda, sa)
        } else {
            gemm_incopy(min_l, min_i, a, l0 + i0 * lda, lda, sa)
        }
    };
    let oncopy = |min_l: usize, min_jj: usize, i0: usize, j0: usize, buf: &mut [f64]| {
        gemm_oncopy(min_l, min_jj, b, i0 + j0 * ldb, ldb, buf)
    };
    let kern = |mi: usize, nj: usize, kl: usize, sa: &[f64], sb: &[f64], i0: usize, j0: usize| {
        zgemm_kernel(gconj, mi, nj, kl, one, sa, sb, b.off(i0 + j0 * ldb), ldb)
    };
    let tkern =
        |tr: bool, mi: usize, nj: usize, kl: usize, sa: &[f64], sb: &[f64], i0: usize, j0: usize, off: usize| {
            trmm_kernel(true, tr, tconj, mi, nj, kl, one, sa, sb, b.off(i0 + j0 * ldb), ldb, off as isize)
        };

    let mut js = 0;
    while js < n {
        let min_j = (n - js).min(r);
        if upper != transa {
            // (UPPER && !TRANSA) || (!UPPER && TRANSA): IUTCOPY / ILNCOPY, TRMM_KERNEL_N (LN/LR).
            let cp = TriCopy { upper, trans: !transa, unit };
            let mut min_l = m.min(q);
            let mut min_i = floor_m(min_l);
            trmm_copy(cp, 4, min_l, min_i, a, lda, 0, 0, &mut sa);
            let mut jjs = js;
            while jjs < js + min_j {
                let min_jj = chunk_jj(min_j + js - jjs);
                let o = min_l * (jjs - js) * 2;
                oncopy(min_l, min_jj, 0, jjs, &mut sb[o..]);
                tkern(false, min_i, min_jj, min_l, &sa, &sb[o..], 0, jjs, 0);
                jjs += min_jj;
            }
            let mut is = min_i;
            while is < min_l {
                min_i = floor_m(min_l - is);
                trmm_copy(cp, 4, min_l, min_i, a, lda, 0, is, &mut sa);
                tkern(false, min_i, min_j, min_l, &sa, &sb, is, js, is);
                is += min_i;
            }
            let mut ls = min_l;
            while ls < m {
                min_l = (m - ls).min(q);
                min_i = floor_m(ls);
                gcopy(min_l, min_i, ls, 0, &mut sa);
                let mut jjs = js;
                while jjs < js + min_j {
                    let min_jj = chunk_jj(min_j + js - jjs);
                    let o = min_l * (jjs - js) * 2;
                    oncopy(min_l, min_jj, ls, jjs, &mut sb[o..]);
                    kern(min_i, min_jj, min_l, &sa, &sb[o..], 0, jjs);
                    jjs += min_jj;
                }
                let mut is = min_i;
                while is < ls {
                    min_i = floor_m(ls - is);
                    gcopy(min_l, min_i, ls, is, &mut sa);
                    kern(min_i, min_j, min_l, &sa, &sb, is, js);
                    is += min_i;
                }
                let mut is = ls;
                while is < ls + min_l {
                    min_i = floor_m(ls + min_l - is);
                    trmm_copy(cp, 4, min_l, min_i, a, lda, ls, is, &mut sa);
                    tkern(false, min_i, min_j, min_l, &sa, &sb, is, js, is - ls);
                    is += min_i;
                }
                ls += q;
            }
        } else {
            // (UPPER && TRANSA) || (!UPPER && !TRANSA): ILTCOPY / IUNCOPY, TRMM_KERNEL_T (LT/LC).
            let cp = TriCopy { upper, trans: !transa, unit };
            let mut min_l = m.min(q);
            let mut min_i = floor_m(min_l);
            trmm_copy(cp, 4, min_l, min_i, a, lda, m - min_l, m - min_l, &mut sa);
            let mut jjs = js;
            while jjs < js + min_j {
                let min_jj = chunk_jj(min_j + js - jjs);
                let o = min_l * (jjs - js) * 2;
                oncopy(min_l, min_jj, m - min_l, jjs, &mut sb[o..]);
                tkern(true, min_i, min_jj, min_l, &sa, &sb[o..], m - min_l, jjs, 0);
                jjs += min_jj;
            }
            let mut is = m - min_l + min_i;
            while is < m {
                min_i = floor_m(m - is);
                trmm_copy(cp, 4, min_l, min_i, a, lda, m - min_l, is, &mut sa);
                tkern(true, min_i, min_j, min_l, &sa, &sb, is, js, is - m + min_l);
                is += min_i;
            }
            let mut ls = m - min_l;
            while ls > 0 {
                min_l = ls.min(q);
                min_i = floor_m(min_l);
                trmm_copy(cp, 4, min_l, min_i, a, lda, ls - min_l, ls - min_l, &mut sa);
                let mut jjs = js;
                while jjs < js + min_j {
                    let min_jj = chunk_jj(min_j + js - jjs);
                    let o = min_l * (jjs - js) * 2;
                    oncopy(min_l, min_jj, ls - min_l, jjs, &mut sb[o..]);
                    tkern(true, min_i, min_jj, min_l, &sa, &sb[o..], ls - min_l, jjs, 0);
                    jjs += min_jj;
                }
                let mut is = ls - min_l + min_i;
                while is < ls {
                    min_i = floor_m(ls - is);
                    trmm_copy(cp, 4, min_l, min_i, a, lda, ls - min_l, is, &mut sa);
                    tkern(true, min_i, min_j, min_l, &sa, &sb, is, js, is - ls + min_l);
                    is += min_i;
                }
                let mut is = ls;
                while is < m {
                    min_i = floor_m(m - is);
                    gcopy(min_l, min_i, ls - min_l, is, &mut sa);
                    kern(min_i, min_j, min_l, &sa, &sb, is, js);
                    is += min_i;
                }
                ls = ls.saturating_sub(q);
            }
        }
        js += r;
    }
}

/// `driver/level3/trmm_R.c` (`B := B op(A)`, `A` is `n x n`).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn trmm_r(
    upper: bool,
    transa: bool,
    conj: bool,
    unit: bool,
    m: usize,
    n: usize,
    one: Z,
    a: ZV,
    lda: usize,
    b: ZV,
    ldb: usize,
) {
    let (p, q, r) = (ZGEMM_P, ZGEMM_Q, ZGEMM_R);
    let gconj = Conj { a: false, b: conj }; // GEMM_KERNEL_R when CONJ
    let tconj = if conj { TrmmConj::NC } else { TrmmConj::NN };
    let mut sa = vec![0.0f64; 2 * p.min(m) * q.min(n)];
    let mut sb = vec![0.0f64; 2 * r.min(n) * q.min(n)];
    // GEMM_ITCOPY of B: rows `i0..` against columns `l0..`.
    let bcopy = |min_l: usize, min_i: usize, i0: usize, l0: usize, sa: &mut [f64]| {
        gemm_itcopy(min_l, min_i, b, i0 + l0 * ldb, ldb, sa)
    };
    // GEMM_ONCOPY(min_l, min_jj, a + (l0 + j0*lda)) / GEMM_OTCOPY(.., a + (j0 + l0*lda)).
    let acopy = |min_l: usize, min_jj: usize, l0: usize, j0: usize, buf: &mut [f64]| {
        if !transa {
            gemm_oncopy(min_l, min_jj, a, l0 + j0 * lda, lda, buf)
        } else {
            gemm_otcopy(min_l, min_jj, a, j0 + l0 * lda, lda, buf)
        }
    };
    let kern = |mi: usize, nj: usize, kl: usize, sa: &[f64], sb: &[f64], i0: usize, j0: usize| {
        zgemm_kernel(gconj, mi, nj, kl, one, sa, sb, b.off(i0 + j0 * ldb), ldb)
    };
    let tkern =
        |tr: bool, mi: usize, nj: usize, kl: usize, sa: &[f64], sb: &[f64], i0: usize, j0: usize, off: isize| {
            trmm_kernel(false, tr, tconj, mi, nj, kl, one, sa, sb, b.off(i0 + j0 * ldb), ldb, off)
        };
    // (!UPPER && !TRANSA) || (UPPER && TRANSA): OLNCOPY / OUTCOPY, TRMM_KERNEL_T (RT/RC).
    // Otherwise OUNCOPY / OLTCOPY, TRMM_KERNEL_N (RN/RR).
    let cp = TriCopy { upper, trans: transa, unit };
    if upper == transa {
        let mut js = 0;
        while js < n {
            let min_j = (n - js).min(r);
            let mut ls = js;
            while ls < js + min_j {
                let min_l = (js + min_j - ls).min(q);
                let mut min_i = m.min(p);
                bcopy(min_l, min_i, 0, ls, &mut sa);
                let mut jjs = 0;
                while jjs < ls - js {
                    let min_jj = chunk_jj(ls - js - jjs);
                    let o = min_l * jjs * 2;
                    acopy(min_l, min_jj, ls, js + jjs, &mut sb[o..]);
                    kern(min_i, min_jj, min_l, &sa, &sb[o..], 0, js + jjs);
                    jjs += min_jj;
                }
                let mut jjs = 0;
                while jjs < min_l {
                    let min_jj = chunk_jj(min_l - jjs);
                    let o = min_l * (ls - js + jjs) * 2;
                    trmm_copy(cp, 2, min_l, min_jj, a, lda, ls, ls + jjs, &mut sb[o..]);
                    tkern(true, min_i, min_jj, min_l, &sa, &sb[o..], 0, ls + jjs, -(jjs as isize));
                    jjs += min_jj;
                }
                let mut is = min_i;
                while is < m {
                    min_i = (m - is).min(p);
                    bcopy(min_l, min_i, is, ls, &mut sa);
                    kern(min_i, ls - js, min_l, &sa, &sb, is, js);
                    tkern(true, min_i, min_l, min_l, &sa, &sb[(ls - js) * min_l * 2..], is, ls, 0);
                    is += p;
                }
                ls += q;
            }
            // Only for n > GEMM_R (not reachable by an n x n A in the tests).
            let mut ls = js + min_j;
            while ls < n {
                let min_l = (n - ls).min(q);
                let mut min_i = m.min(p);
                bcopy(min_l, min_i, 0, ls, &mut sa);
                let mut jjs = js;
                while jjs < js + min_j {
                    let min_jj = chunk_jj(min_j + js - jjs);
                    let o = min_l * (jjs - js) * 2;
                    acopy(min_l, min_jj, ls, jjs, &mut sb[o..]);
                    kern(min_i, min_jj, min_l, &sa, &sb[o..], 0, jjs);
                    jjs += min_jj;
                }
                let mut is = min_i;
                while is < m {
                    min_i = (m - is).min(p);
                    bcopy(min_l, min_i, is, ls, &mut sa);
                    kern(min_i, min_j, min_l, &sa, &sb, is, js);
                    is += p;
                }
                ls += q;
            }
            js += r;
        }
    } else {
        let mut js = n as isize;
        while js > 0 {
            let jsu = js as usize;
            let min_j = jsu.min(r);
            let mut start_ls = jsu - min_j;
            while start_ls + q < jsu {
                start_ls += q;
            }
            let mut ls = start_ls as isize;
            while ls >= (jsu - min_j) as isize {
                let lsu = ls as usize;
                let min_l = (jsu - lsu).min(q);
                let mut min_i = m.min(p);
                bcopy(min_l, min_i, 0, lsu, &mut sa);
                let mut jjs = 0;
                while jjs < min_l {
                    let min_jj = chunk_jj(min_l - jjs);
                    let o = min_l * jjs * 2;
                    trmm_copy(cp, 2, min_l, min_jj, a, lda, lsu, lsu + jjs, &mut sb[o..]);
                    tkern(false, min_i, min_jj, min_l, &sa, &sb[o..], 0, lsu + jjs, -(jjs as isize));
                    jjs += min_jj;
                }
                let mut jjs = 0;
                while jjs < jsu - lsu - min_l {
                    let min_jj = chunk_jj(jsu - lsu - min_l - jjs);
                    let o = min_l * (min_l + jjs) * 2;
                    acopy(min_l, min_jj, lsu, lsu + min_l + jjs, &mut sb[o..]);
                    kern(min_i, min_jj, min_l, &sa, &sb[o..], 0, lsu + min_l + jjs);
                    jjs += min_jj;
                }
                let mut is = min_i;
                while is < m {
                    min_i = (m - is).min(p);
                    bcopy(min_l, min_i, is, lsu, &mut sa);
                    tkern(false, min_i, min_l, min_l, &sa, &sb, is, lsu, 0);
                    if jsu - lsu - min_l > 0 {
                        kern(min_i, jsu - lsu - min_l, min_l, &sa, &sb[min_l * min_l * 2..], is, lsu + min_l);
                    }
                    is += p;
                }
                ls -= q as isize;
            }
            // Only for n > GEMM_R (not reachable by an n x n A in the tests).
            let mut ls = 0;
            while ls < jsu - min_j {
                let min_l = (jsu - min_j - ls).min(q);
                let mut min_i = m.min(p);
                bcopy(min_l, min_i, 0, ls, &mut sa);
                let mut jjs = jsu;
                while jjs < jsu + min_j {
                    let min_jj = chunk_jj(min_j + jsu - jjs);
                    let o = min_l * (jjs - jsu) * 2;
                    acopy(min_l, min_jj, ls, jjs - min_j, &mut sb[o..]);
                    kern(min_i, min_jj, min_l, &sa, &sb[o..], 0, jjs - min_j);
                    jjs += min_jj;
                }
                let mut is = min_i;
                while is < m {
                    min_i = (m - is).min(p);
                    bcopy(min_l, min_i, is, ls, &mut sa);
                    kern(min_i, min_j, min_l, &sa, &sb, is, jsu - min_j);
                    is += p;
                }
                ls += q;
            }
            js -= r as isize;
        }
    }
}
