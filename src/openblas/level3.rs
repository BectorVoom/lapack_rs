//! **OpenBLAS 0.3.30 SkylakeX track** (third oracle): real double level-3 BLAS
//! `DGEMM`, `DSYR2K` (`TRANS = 'N'`) and `DTRMM`, bit-exact to `scipy_dgemm_`,
//! `scipy_dsyr2k_`, `scipy_dtrmm_` of scipy's bundled
//! `libscipy_openblas-6cdc3b4a.so` (`OPENBLAS_CORETYPE=SkylakeX`,
//! `OPENBLAS_NUM_THREADS=1`). Never compared with the Reference-LAPACK or
//! libcint oracles.
//!
//! Transcribed from OpenBLAS 0.3.30 (checked against the disassembly):
//! * `interface/gemm.c` (`scipy_dgemm_`; no GEMV forwarding in this x86_64
//!   build), `interface/syr2k.c` (`scipy_dsyr2k_`), `interface/trsm.c` built
//!   with `TRMM` (`scipy_dtrmm_`).
//! * `driver/level3/level3.c` (`dgemm_nn`, `dgemm_nt`, `dgemm_tn`, `dgemm_tt`),
//!   `driver/level3/syr2k_k.c` + `level3_syr2k.c` (`dsyr2k_UN`, `dsyr2k_LN`),
//!   `driver/level3/syr2k_kernel.c` (`dsyr2k_kernel_U`, `dsyr2k_kernel_L`),
//!   `driver/level3/trmm_L.c`, `trmm_R.c` (`dtrmm_{L,R}{N,T}{U,L}{U,N}`).
//! * `kernel/x86_64/dgemm_small_kernel_permit_skylakex.c`
//!   (`dgemm_small_matrix_permit_SKYLAKEX`),
//!   `dgemm_small_kernel_{nn,nt,tn,tt}_skylakex.c`
//!   (`dgemm_small_kernel_{nn,nt,tn,tt}_SKYLAKEX`, and the `B0` builds
//!   `dgemm_small_kernel_b0_*_SKYLAKEX`), `dgemm_beta_skylakex.c`
//!   (`dgemm_beta_SKYLAKEX`), `dgemm_kernel_16x2_skylakex.c`
//!   (`dgemm_kernel_SKYLAKEX`, and with `TRMMKERNEL`
//!   `dtrmm_kernel_{LN,LT,RN,RT}_SKYLAKEX`), `dscal.c` (`dscal_k_SKYLAKEX`,
//!   as `SCAL_K` in `syrk_beta`).
//! * Packing: `dgemm_tcopy_16_skylakex.c` (`dgemm_itcopy_SKYLAKEX`),
//!   `generic/gemm_ncopy_16.c` (`dgemm_incopy_SKYLAKEX`),
//!   `generic/gemm_ncopy_2.c` (`dgemm_oncopy_SKYLAKEX`),
//!   `generic/gemm_tcopy_2.c` (`dgemm_otcopy_SKYLAKEX`),
//!   `generic/trmm_{u,l}{n,t}copy_16.c` (`dtrmm_i{u,l}{n,t}{u,n}copy_SKYLAKEX`),
//!   `generic/trmm_{u,l}{n,t}copy_2.c` (`dtrmm_o{u,l}{n,t}{u,n}copy_SKYLAKEX`).
//!
//! Arithmetic (from the disassembly): every inner product is a chain of
//! `vfmadd231` (`acc = a*b + acc`, `a.mul_add(b, acc)`), one chain per output
//! element unless a comment says the chain is split across lanes and merged.
//! The data flow is kept explicit: pack (`*copy`) -> micro-kernel (reads only
//! the packed buffers) -> write into C/B.
//!
//! **Port notes (CubeCL).** Inside one micro-kernel call every output element
//! is independent (element-wise, lane-parallel); what fixes its bits is the
//! *tile* it falls in (row tile 16/8/4/2/1, column tile 12/10/8/6/4/2/1), the
//! `k` window, and, for `DGEMM`/`DSYR2K`, the sequence of kernel calls that add
//! into it (one per `GEMM_Q` block of `k`, in order). `DTRMM` is different:
//! the driver overwrites B in place block by block, and later blocks pack
//! (copy) B values produced by earlier ones, so the blocks form an ordered
//! chain; only elements within one kernel call are independent.

use crate::ptr::{D, V};

/// `gotoblas->dgemm_p` of the pinned library (`driver/others/parameter.c`,
/// read live at 0x2e0 of `gotoblas_SKYLAKEX`; asserted by the tests).
pub const GEMM_P: usize = 192;
/// `gotoblas->dgemm_q` (0x2e4).
pub const GEMM_Q: usize = 384;
/// `gotoblas->dgemm_r` (0x2e8). Derived from the buffer size and `GEMM_P*GEMM_Q`
/// at init; never splits the column range for the sizes this crate uses.
pub const GEMM_R: usize = 10704;
/// `gotoblas->dgemm_unroll_m` (0x2ec).
pub const GEMM_UNROLL_M: usize = 16;
/// `gotoblas->dgemm_unroll_n` (0x2f0).
pub const GEMM_UNROLL_N: usize = 2;
/// `gotoblas->dgemm_unroll_mn` (0x2f4).
pub const GEMM_UNROLL_MN: usize = 32;

// ---------------------------------------------------------------------------
// dgemm_beta_SKYLAKEX
// ---------------------------------------------------------------------------

/// `dgemm_beta_skylakex.c`: `C = beta*C` element-wise (`vmulsd`/`vmulpd`), or
/// zeros when `beta == 0` (never reads C then).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn gemm_beta(m: usize, n: usize, beta: f64, c: D, ldc: usize) {
    if m == 0 || n == 0 {
        return;
    }
    for j in 0..n {
        let cj = c.off(j * ldc);
        for i in 0..m {
            if beta == 0.0 {
                cj.set(i, 0.0);
            } else {
                cj.set(i, cj.at(i) * beta);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Small-matrix path: dgemm_small_matrix_permit / dgemm_small_kernel_*_SKYLAKEX
// ---------------------------------------------------------------------------

/// `dgemm_small_kernel_permit_skylakex.c`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn small_matrix_permit(transa: bool, transb: bool, m: usize, n: usize, k: usize) -> bool {
    let mnk = m as f64 * n as f64 * k as f64;
    if mnk > 100.0 * 100.0 * 100.0 {
        return false;
    }
    if transa && !transb && (m * n > 1200 || k < 32) {
        return false;
    }
    true
}

/// Element chain: `result = fmadd(A, B, result)` over `k` ascending.
#[inline(always)]
fn chain(k: usize, ab: impl Fn(usize) -> (f64, f64)) -> f64 {
    let mut acc = 0.0f64;
    for l in 0..k {
        let (a, b) = ab(l);
        acc = a.mul_add(b, acc);
    }
    acc
}

/// "Dot style" accumulation of the small kernels (`LOAD_KA_512`/`LOAD_KB_512`):
/// lane `t` of one 512-bit accumulator takes the terms `l ≡ t (mod 8)`; the
/// last partial group is a masked load, so lanes `>= remains` add `0*0`.
#[inline(always)]
fn dot_lanes(k: usize, ab: impl Fn(usize) -> (f64, f64)) -> [f64; 8] {
    let mut r = [0.0f64; 8];
    let k8 = k & !7;
    let mut l = 0;
    while l < k8 {
        for (t, rt) in r.iter_mut().enumerate() {
            let (a, b) = ab(l + t);
            *rt = a.mul_add(b, *rt);
        }
        l += 8;
    }
    let remains = k - k8;
    if remains != 0 {
        for (t, rt) in r.iter_mut().enumerate() {
            let (a, b) = if t < remains { ab(k8 + t) } else { (0.0, 0.0) };
            *rt = a.mul_add(b, *rt);
        }
    }
    r
}

/// `REDUCE_4` (unpack/permutex2var then three `vaddpd` levels):
/// `((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7))`.
#[inline(always)]
fn reduce_4(r: [f64; 8]) -> f64 {
    ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
}

/// `_mm512_reduce_add_pd` as gcc lowers it (hi4+lo4, hi2+lo2, `vhaddpd`):
/// `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))`.
#[inline(always)]
fn reduce_add_512(r: [f64; 8]) -> f64 {
    ((r[0] + r[4]) + (r[2] + r[6])) + ((r[1] + r[5]) + (r[3] + r[7]))
}

/// How one small-kernel output element is reduced and stored.
#[derive(Clone, Copy)]
enum SmallStore {
    /// `STORE_512` family: `r = acc*alpha` (`vmulpd`), then
    /// `vfmadd231pd (C), beta, r` -> `beta*C + r`.
    Chain,
    /// Dot style, `STORE_REDUCE_M4/N4`: `s = alpha*reduce_4` then `beta*C + s`.
    Reduce4,
    /// Dot style, scalar `STORE_REDUCE`: gcc contracts
    /// `alpha*reduce_add + beta*C` into `vmulsd beta,C; vfmadd132sd alpha`
    /// -> `sum*alpha + (C*beta)`.
    ReduceScalar,
}

/// Small kernels (`DGEMM_SMALL_K_*` and `DGEMM_SMALL_K_B0_*`). `beta == None`
/// is the `B0` build (C is write-only). `a(i,l)` / `b(l,j)` are `op(A)` and
/// `op(B)`; the row/column split into element chains vs dot style and the
/// store variant per element follow each file's loop structure:
/// * `nt`, `tt`: every element is a chain (their 32/16/8-row tiles, masks,
///   gathers and scatters only move whole lanes).
/// * `nn`: rows `< M & ~7` are chains; the remaining `mm` rows are chains when
///   `mm > 4 || K < 16`, otherwise dot style (the `mbuf` copy of A's rows).
/// * `tn`: dot style everywhere.
///
/// Dot style: rows of a 4-row group use `REDUCE_4` for every column; rows of
/// 2-row groups and single rows use `REDUCE_4` for columns `< N & ~3` and the
/// scalar store for the rest.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn small_kernel(
    transa: bool,
    transb: bool,
    m: usize,
    n: usize,
    k: usize,
    a: D,
    lda: usize,
    alpha: f64,
    b: D,
    ldb: usize,
    beta: Option<f64>,
    c: D,
    ldc: usize,
) {
    let opa = |i: usize, l: usize| if transa { a.at(l + i * lda) } else { a.at(i + l * lda) };
    let opb = |l: usize, j: usize| if transb { b.at(j + l * ldb) } else { b.at(l + j * ldb) };
    // First row of the dot-style region (`m` = none).
    let dot_from = match (transa, transb) {
        (false, false) => {
            let m8 = m & !7;
            if m - m8 > 4 || k < 16 { m } else { m8 }
        }
        (true, false) => 0,
        _ => m,
    };
    let m4 = m & !3;
    let n4 = n & !3;
    for j in 0..n {
        for i in 0..m {
            let p = i + j * ldc;
            let ab = |l: usize| (opa(i, l), opb(l, j));
            let (sum, store) = if i < dot_from {
                (chain(k, ab), SmallStore::Chain)
            } else {
                let r = dot_lanes(k, ab);
                // 4-row groups start at `dot_from` and end at `m & ~3`.
                if i < m4 || j < n4 {
                    (reduce_4(r), SmallStore::Reduce4)
                } else {
                    (reduce_add_512(r), SmallStore::ReduceScalar)
                }
            };
            let v = match (store, beta) {
                (SmallStore::Chain, None) => sum * alpha,
                (SmallStore::Chain, Some(be)) => be.mul_add(c.at(p), sum * alpha),
                (SmallStore::Reduce4, None) => alpha * sum,
                (SmallStore::Reduce4, Some(be)) => be.mul_add(c.at(p), alpha * sum),
                (SmallStore::ReduceScalar, None) => alpha * sum,
                (SmallStore::ReduceScalar, Some(be)) => sum.mul_add(alpha, c.at(p) * be),
            };
            c.set(p, v);
        }
    }
}

// ---------------------------------------------------------------------------
// Packing (gemm/trmm copy routines) and packed layouts
// ---------------------------------------------------------------------------

/// Row tiles of `dgemm_kernel_16x2_skylakex.c` (`M`: 16s, then one each of
/// 8, 4, 2, 1). The inner copies (`*_16`) use the same grid.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn m_tiles(m: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut i = 0;
    while m - i >= 16 {
        v.push((i, 16));
        i += 16;
    }
    for d in [8, 4, 2, 1] {
        if m - i >= d {
            v.push((i, d));
            i += d;
        }
    }
    v
}

/// Column tiles of `dgemm_kernel_16x2_skylakex.c` (`CNAME`'s `n_count` loops).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn n_tiles(n: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut j = 0;
    let mut nc = n;
    for (d, gt) in [(12, 11), (10, 9), (8, 7), (6, 5), (4, 3), (2, 1)] {
        while nc > gt {
            v.push((j, d));
            j += d;
            nc -= d;
        }
    }
    if nc > 0 {
        v.push((j, 1));
    }
    v
}

/// Inner packed buffer (`sa`, "A" side of the kernel): for each row tile
/// `(i0, md)`, for each `l`, the `md` values of rows `i0..i0+md` — offset
/// `i0*k + l*md + (i-i0)`. `dgemm_itcopy` (rows of A), `dgemm_incopy`
/// (columns of A, i.e. `op(A) = A'`) and the triangular `dtrmm_i*copy`
/// produce this layout; `val(i, l)` is the value the copy writes.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn pack_inner(buf: &mut Vec<f64>, rows: usize, k: usize, val: impl Fn(usize, usize) -> f64) {
    buf.clear();
    buf.resize(rows * k, 0.0);
    for (i0, md) in m_tiles(rows) {
        for l in 0..k {
            for r in 0..md {
                buf[i0 * k + l * md + r] = val(i0 + r, l);
            }
        }
    }
}

/// Outer packed buffer (`sb`, "B" side): columns in pairs, each pair `k` rows
/// of `[b(l,j), b(l,j+1)]`; an odd last column is `k` single values.
/// `dgemm_oncopy`/`dgemm_otcopy` (`generic/gemm_{n,t}copy_2.c`) and
/// `dtrmm_o*copy` (`generic/trmm_*copy_2.c`) produce this layout; appended at
/// `buf[at..]`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn pack_outer(buf: &mut Vec<f64>, at: usize, k: usize, cols: usize, val: impl Fn(usize, usize) -> f64) {
    if buf.len() < at + k * cols {
        buf.resize(at + k * cols, 0.0);
    }
    for j in 0..cols {
        for l in 0..k {
            buf[at + outer_at(cols, k, l, j)] = val(l, j);
        }
    }
}

#[inline(always)]
fn outer_at(cols: usize, k: usize, l: usize, j: usize) -> usize {
    if cols % 2 == 1 && j == cols - 1 { j * k + l } else { (j & !1) * k + 2 * l + (j & 1) }
}

// ---------------------------------------------------------------------------
// dgemm_kernel_16x2_skylakex.c (GEMM and TRMMKERNEL builds)
// ---------------------------------------------------------------------------

/// Which build of `dgemm_kernel_16x2_skylakex.c`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kern {
    /// `dgemm_kernel_SKYLAKEX`: `C += alpha*acc` (`vfmadd213pd (C), alpha`).
    Gemm,
    /// `dtrmm_kernel_LN_SKYLAKEX` (`LEFT`, backwards).
    TrmmLN,
    /// `dtrmm_kernel_LT_SKYLAKEX` (`LEFT TRANSA`).
    TrmmLT,
    /// `dtrmm_kernel_RN_SKYLAKEX`.
    TrmmRN,
    /// `dtrmm_kernel_RT_SKYLAKEX` (`TRANSA`, backwards).
    TrmmRT,
}

/// Accumulation over a window `l0..l0+cnt` of one element's chain.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Acc {
    /// One lane, `acc = a*b + acc`.
    Plain,
    /// `KERNEL_k1m4n12` with `k_count >= 2`: pairs of `l` go to two lanes of a
    /// zmm (the running chain continues in the low lane, the high lane starts
    /// at +0); merged by `vextractf64x4 + vaddpd` (hi + lo); the odd
    /// leftover `l` continues on the merged value.
    Pair,
    /// `KERNEL_k1m2n12` / `KERNEL_k1m1n12` with `k_count >= 4`: four lanes
    /// per element (running chain in lane 0), merged
    /// `(s3+s2) + (s1+s0)`, leftovers (< 4) continue on the merged value.
    Quad,
}

#[inline(always)]
fn accumulate(mode: Acc, mut acc: f64, l0: usize, cnt: usize, ab: &impl Fn(usize) -> (f64, f64)) -> f64 {
    let mut l = l0;
    let end = l0 + cnt;
    match mode {
        Acc::Pair if cnt >= 2 => {
            let (mut lo, mut hi) = (acc, 0.0f64);
            while end - l >= 2 {
                let (a0, b0) = ab(l);
                let (a1, b1) = ab(l + 1);
                lo = a0.mul_add(b0, lo);
                hi = a1.mul_add(b1, hi);
                l += 2;
            }
            acc = hi + lo;
        }
        Acc::Quad if cnt >= 4 => {
            let mut s = [acc, 0.0, 0.0, 0.0];
            while end - l >= 4 {
                for (t, st) in s.iter_mut().enumerate() {
                    let (a, b) = ab(l + t);
                    *st = a.mul_add(b, *st);
                }
                l += 4;
            }
            acc = (s[3] + s[2]) + (s[1] + s[0]);
        }
        _ => {}
    }
    while l < end {
        let (a, b) = ab(l);
        acc = a.mul_add(b, acc);
        l += 1;
    }
    acc
}

/// `dgemm_kernel_16x2_skylakex.c`: `m x n` block of C (at `c`, stride `ldc`)
/// from the packed buffers `sa` (inner layout, `m` rows) and `sb` (outer
/// layout, `n` columns), `k` terms per element.
///
/// Per element `(i, j)` in row tile `(i0, md)` and column tile `(j0, nd)`,
/// pair `p = (j-j0)/2`, the chain is over these `l` windows (from the asm
/// macros `INITASM_SET_K`, `INIT_set_papb`, `init/save_update_k`,
/// `HEAD/TAIL_SET_OFF`, `kernel_kstart_n*`, `kernel_kend_n*`):
/// * `Gemm`: `0..k`.
/// * `TrmmLN` (`off = offset`): `off+i0 .. k`.
/// * `TrmmLT` (`off = offset`): `0 .. off+i0+md`.
/// * `TrmmRN` (`off = -offset + j0`, `r13 = off + min(nd,2)`): main `0..r13`,
///   then `kend` continues pair `p` over `r13 .. r13+2p` (plain).
/// * `TrmmRT` (`off = -offset + j0`, `s = nd/2-1` steps when `nd >= 4`):
///   `kstart` runs pair `p` plain over `off+2p .. off+2s`, then main
///   `off+2s .. k`.
///
/// The main window is `Acc::Pair` for `(md, nd) = (4, 12)`, `Acc::Quad` for
/// `md <= 2, nd = 12`, else plain. Store: `Gemm` `C = acc*alpha + C`
/// (`vfmadd213`); TRMM builds `C = alpha*acc` (`vmulpd`, overwrite).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dgemm_kernel(
    kern: Kern,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    sa: &[f64],
    sb: &[f64],
    c: D,
    ldc: usize,
    offset: isize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let ki = k as isize;
    let mut off: isize = match kern {
        Kern::Gemm => 0,
        Kern::TrmmLN | Kern::TrmmLT => offset,
        Kern::TrmmRN | Kern::TrmmRT => -offset,
    };
    let mt = m_tiles(m);
    for (j0, nd) in n_tiles(n) {
        if kern == Kern::TrmmRN {
            off += nd.min(2) as isize; // HEAD_SET_OFF
        }
        // INITASM_SET_K
        let mut r13: isize = match kern {
            Kern::Gemm => ki,
            Kern::TrmmLN | Kern::TrmmRT => ki - off,
            Kern::TrmmLT | Kern::TrmmRN => off,
        };
        for &(i0, md) in &mt {
            if kern == Kern::TrmmLT {
                r13 += md as isize; // init_update_k
            }
            let main = match (md, nd) {
                (4, 12) => Acc::Pair,
                (1 | 2, 12) => Acc::Quad,
                _ => Acc::Plain,
            };
            // The main loop's window (`INIT_set_papb` skips `k - r13` on the
            // backwards builds).
            let (l0, cnt) = match kern {
                Kern::Gemm => (0usize, k),
                Kern::TrmmLN | Kern::TrmmRT => ((ki - r13) as usize, r13 as usize),
                Kern::TrmmLT | Kern::TrmmRN => (0usize, r13 as usize),
            };
            let steps = if kern == Kern::TrmmRT && nd >= 4 { nd / 2 - 1 } else { 0 };
            for j in j0..j0 + nd {
                let p = (j - j0) / 2;
                for i in i0..i0 + md {
                    // sa: tile base `i0*k`, `md` rows per `l`; sb: pair base.
                    let ab = |l: usize| (sa[i0 * k + l * md + (i - i0)], sb[outer_at(n, k, l, j)]);
                    let acc = match kern {
                        Kern::TrmmRT => {
                            let a0 = accumulate(Acc::Plain, 0.0, l0 + 2 * p, 2 * steps.saturating_sub(p), &ab);
                            accumulate(main, a0, l0 + 2 * steps, cnt - 2 * steps, &ab)
                        }
                        Kern::TrmmRN => {
                            let a0 = accumulate(main, 0.0, l0, cnt, &ab);
                            accumulate(Acc::Plain, a0, l0 + cnt, 2 * p, &ab)
                        }
                        _ => accumulate(main, 0.0, l0, cnt, &ab),
                    };
                    let q = i + j * ldc;
                    if kern == Kern::Gemm {
                        c.set(q, acc.mul_add(alpha, c.at(q)));
                    } else {
                        c.set(q, alpha * acc);
                    }
                }
            }
            if kern == Kern::TrmmLN {
                r13 -= md as isize; // save_update_k
            }
        }
        match kern {
            Kern::TrmmRN => off += nd.saturating_sub(2) as isize, // TAIL_SET_OFF
            Kern::TrmmRT => off += nd as isize,
            _ => {}
        }
    }
}

/// Column block of the first row block in `level3.c`, `trmm_L.c`, `trmm_R.c`.
/// The drivers are compiled once for DYNAMIC_ARCH, so the
/// `#if defined(SKYLAKEX)` branch (`6*GEMM_UNROLL_N`) is *not* taken; the
/// generic one is (checked in `dgemm_nn`: `lea (%rax,%rax,2)` of
/// `dgemm_unroll_n`): `>= 3*UNROLL_N -> 6`, else `> UNROLL_N -> 2`, else all.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn min_jj_unroll(rem: usize) -> usize {
    if rem >= 3 * GEMM_UNROLL_N {
        3 * GEMM_UNROLL_N
    } else if rem > GEMM_UNROLL_N {
        GEMM_UNROLL_N
    } else {
        rem
    }
}

// ---------------------------------------------------------------------------
// DGEMM: interface/gemm.c + driver/level3/level3.c
// ---------------------------------------------------------------------------

/// `DGEMM` through `scipy_dgemm_` (single thread, SkylakeX).
///
/// `m == 0 || n == 0` returns. Then `GEMM_SMALL_MATRIX_PERMIT` (MNK <= 1e6,
/// and for `TN` also `m*n <= 1200 && k >= 32`) selects the small kernels
/// (`beta == 0` -> `B0` build) — note: no `alpha == 0` or `k == 0` shortcut
/// there. Otherwise the blocked driver `dgemm_{n,t}{n,t}`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dgemm(
    transa: bool,
    transb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: D,
    lda: usize,
    b: D,
    ldb: usize,
    beta: f64,
    c: D,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if small_matrix_permit(transa, transb, m, n, k) {
        let be = if beta == 0.0 { None } else { Some(beta) };
        small_kernel(transa, transb, m, n, k, a, lda, alpha, b, ldb, be, c, ldc);
        return;
    }
    gemm_driver(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/// `driver/level3/level3.c` (`dgemm_nn/nt/tn/tt`), `range_m = range_n = NULL`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn gemm_driver(
    transa: bool,
    transb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: D,
    lda: usize,
    b: D,
    ldb: usize,
    beta: f64,
    c: D,
    ldc: usize,
) {
    if beta != 1.0 {
        gemm_beta(m, n, beta, c, ldc); // BETA_OPERATION
    }
    if k == 0 || alpha == 0.0 {
        return;
    }
    // op(A)(i, l) and op(B)(l, j): ICOPY is `itcopy` (A rows) or `incopy`
    // (A columns); OCOPY `oncopy` (B columns) or `otcopy` (B rows).
    let opa = |i: usize, l: usize| if transa { a.at(l + i * lda) } else { a.at(i + l * lda) };
    let opb = |l: usize, j: usize| if transb { b.at(j + l * ldb) } else { b.at(l + j * ldb) };
    let (mut sa, mut sb) = (Vec::new(), Vec::new());
    let mut js = 0;
    while js < n {
        let min_j = (n - js).min(GEMM_R);
        let mut ls = 0;
        while ls < k {
            let mut min_l = k - ls;
            if min_l >= GEMM_Q * 2 {
                min_l = GEMM_Q;
            } else if min_l > GEMM_Q {
                min_l = (min_l / 2).div_ceil(GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }
            let mut min_i = m;
            let mut l1stride = 1;
            if min_i >= GEMM_P * 2 {
                min_i = GEMM_P;
            } else if min_i > GEMM_P {
                min_i = (min_i / 2).div_ceil(GEMM_UNROLL_M) * GEMM_UNROLL_M;
            } else {
                l1stride = 0;
            }
            pack_inner(&mut sa, min_i, min_l, |i, l| opa(i, ls + l));
            let mut jjs = js;
            while jjs < js + min_j {
                let min_jj = min_jj_unroll(min_j + js - jjs);
                let at = min_l * (jjs - js) * l1stride;
                pack_outer(&mut sb, at, min_l, min_jj, |l, j| opb(ls + l, jjs + j));
                dgemm_kernel(Kern::Gemm, min_i, min_jj, min_l, alpha, &sa, &sb[at..], c.off(jjs * ldc), ldc, 0);
                jjs += min_jj;
            }
            let mut is = min_i;
            while is < m {
                min_i = m - is;
                if min_i >= GEMM_P * 2 {
                    min_i = GEMM_P;
                } else if min_i > GEMM_P {
                    min_i = (min_i / 2).div_ceil(GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }
                pack_inner(&mut sa, min_i, min_l, |i, l| opa(is + i, ls + l));
                dgemm_kernel(Kern::Gemm, min_i, min_j, min_l, alpha, &sa, &sb, c.off(is + js * ldc), ldc, 0);
                is += min_i;
            }
            ls += min_l;
        }
        js += GEMM_R;
    }
}

// ---------------------------------------------------------------------------
// DSYR2K ('N'): interface/syr2k.c + driver/level3/syr2k_k.c, level3_syr2k.c,
// syr2k_kernel.c
// ---------------------------------------------------------------------------

/// `DSYR2K` with `TRANS = 'N'` through `scipy_dsyr2k_` (`dsyr2k_UN` /
/// `dsyr2k_LN`, single thread). `n == 0` returns; no small-matrix path.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dsyr2k_n(
    upper: bool,
    n: usize,
    k: usize,
    alpha: f64,
    a: D,
    lda: usize,
    b: D,
    ldb: usize,
    beta: f64,
    c: D,
    ldc: usize,
) {
    if n == 0 {
        return;
    }
    // syrk_beta (`beta != ONE`): SCAL_K (`dscal_k`, last flag 0) on the
    // triangle's column pieces: `beta == 0` writes zeros, else `x*beta`.
    if beta != 1.0 {
        for j in 0..n {
            let (i0, i1) = if upper { (0, j + 1) } else { (j, n) };
            for i in i0..i1 {
                let p = i + j * ldc;
                c.set(p, if beta == 0.0 { 0.0 } else { c.at(p) * beta });
            }
        }
    }
    if k == 0 || alpha == 0.0 {
        return;
    }
    let (mut sa, mut sb) = (Vec::new(), Vec::new());
    let p_rule = |min_i: usize| {
        if min_i >= GEMM_P * 2 {
            GEMM_P
        } else if min_i > GEMM_P {
            (min_i / 2).div_ceil(GEMM_UNROLL_MN) * GEMM_UNROLL_MN
        } else {
            min_i
        }
    };
    let mut js = 0;
    while js < n {
        let min_j = (n - js).min(GEMM_R);
        let (m_start, m_end) = if upper { (0, (js + min_j).min(n)) } else { (js, n) };
        let mut ls = 0;
        while ls < k {
            let mut min_l = k - ls;
            if min_l >= GEMM_Q * 2 {
                min_l = GEMM_Q;
            } else if min_l > GEMM_Q {
                min_l = min_l.div_ceil(2);
            }
            // Two passes: flag 1 packs rows of A inside / B outside
            // (`alpha*A*B'`), flag 0 swaps them (`KERNEL_OPERATION_C`).
            for flag in [true, false] {
                let (inn, lin, out, lout) = if flag { (a, lda, b, ldb) } else { (b, ldb, a, lda) };
                // ICOPY = dgemm_itcopy (rows of `inn`), OCOPY = dgemm_otcopy
                // (rows of `out`, as columns of the outer buffer).
                let icopy = |sa: &mut Vec<f64>, rows: usize, row0: usize| {
                    pack_inner(sa, rows, min_l, |i, l| inn.at(row0 + i + (ls + l) * lin))
                };
                let ocopy = |sb: &mut Vec<f64>, at: usize, cols: usize, col0: usize| {
                    pack_outer(sb, at, min_l, cols, |l, j| out.at(col0 + j + (ls + l) * lout))
                };
                let mut min_i = p_rule(m_end - m_start);
                if upper {
                    let mut jjs;
                    if m_start >= js {
                        icopy(&mut sa, min_i, m_start);
                        let aa = min_l * (m_start - js);
                        ocopy(&mut sb, aa, min_i, m_start);
                        syr2k_kernel(upper, min_i, min_i, min_l, alpha, &sa, &sb[aa..], c, m_start, m_start, ldc, flag);
                        jjs = m_start + min_i;
                    } else {
                        icopy(&mut sa, min_i, m_start);
                        jjs = js;
                    }
                    while jjs < js + min_j {
                        let min_jj = (min_j + js - jjs).min(GEMM_UNROLL_MN);
                        let at = min_l * (jjs - js);
                        ocopy(&mut sb, at, min_jj, jjs);
                        syr2k_kernel(upper, min_i, min_jj, min_l, alpha, &sa, &sb[at..], c, m_start, jjs, ldc, flag);
                        jjs += GEMM_UNROLL_MN;
                    }
                    let mut is = m_start + min_i;
                    while is < m_end {
                        min_i = p_rule(m_end - is);
                        icopy(&mut sa, min_i, is);
                        syr2k_kernel(upper, min_i, min_j, min_l, alpha, &sa, &sb, c, is, js, ldc, flag);
                        is += min_i;
                    }
                } else {
                    let aa = min_l * (m_start - js);
                    icopy(&mut sa, min_i, m_start);
                    ocopy(&mut sb, aa, min_i, m_start);
                    syr2k_kernel(
                        upper,
                        min_i,
                        min_i.min(min_j + js - m_start),
                        min_l,
                        alpha,
                        &sa,
                        &sb[aa..],
                        c,
                        m_start,
                        m_start,
                        ldc,
                        flag,
                    );
                    let mut jjs = js;
                    while jjs < m_start {
                        let min_jj = (m_start - jjs).min(GEMM_UNROLL_MN);
                        let at = min_l * (jjs - js);
                        ocopy(&mut sb, at, min_jj, jjs);
                        syr2k_kernel(upper, min_i, min_jj, min_l, alpha, &sa, &sb[at..], c, m_start, jjs, ldc, flag);
                        jjs += GEMM_UNROLL_MN;
                    }
                    let mut is = m_start + min_i;
                    while is < m_end {
                        min_i = p_rule(m_end - is);
                        let aa = min_l * (is - js);
                        if is < js + min_j {
                            icopy(&mut sa, min_i, is);
                            ocopy(&mut sb, aa, min_i, is);
                            syr2k_kernel(
                                upper,
                                min_i,
                                min_i.min(min_j - is + js),
                                min_l,
                                alpha,
                                &sa,
                                &sb[aa..],
                                c,
                                is,
                                is,
                                ldc,
                                flag,
                            );
                            syr2k_kernel(upper, min_i, is - js, min_l, alpha, &sa, &sb, c, is, js, ldc, flag);
                        } else {
                            icopy(&mut sa, min_i, is);
                            syr2k_kernel(upper, min_i, min_j, min_l, alpha, &sa, &sb, c, is, js, ldc, flag);
                        }
                        is += min_i;
                    }
                }
            }
            ls += min_l;
        }
        js += GEMM_R;
    }
}

/// `syr2k_kernel.c` (`dsyr2k_kernel_U/L`) for the block of C at `(x, y)`
/// (`offset = x - y`). The packed buffers are advanced by whole rows
/// (`a + r*k`) and columns (`b + c*k`) exactly as the C does; the kernel
/// re-tiles the rows it is given. The diagonal `32 x 32` pieces (`flag`)
/// go through a zeroed `subbuffer` and are symmetrised with
/// `c += sub[i+j*nn] + sub[j+i*nn]` (`addsd` twice, that order).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn syr2k_kernel(
    upper: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    c: D,
    x: usize,
    y: usize,
    ldc: usize,
    flag: bool,
) {
    let gemm = |m: isize, n: isize, a_off: isize, b_off: isize, c_off: isize| {
        if m > 0 && n > 0 {
            let base = (x + y * ldc) as isize + c_off;
            dgemm_kernel(
                Kern::Gemm,
                m as usize,
                n as usize,
                k,
                alpha,
                &a[(a_off * k as isize) as usize..],
                &b[(b_off * k as isize) as usize..],
                c.off(base as usize),
                ldc,
                0,
            );
        }
    };
    let ldci = ldc as isize;
    let (mut m, mut n) = (m as isize, n as isize);
    let mut offset = x as isize - y as isize;
    // Running pointer offsets: rows into `a`, columns into `b`, elements into `c`.
    let (mut ao, mut bo, mut co) = (0isize, 0isize, 0isize);
    if m + offset < 0 {
        if upper {
            gemm(m, n, ao, bo, co);
        }
        return;
    }
    if n < offset {
        if !upper {
            gemm(m, n, ao, bo, co);
        }
        return;
    }
    if offset > 0 {
        if !upper {
            gemm(m, offset, ao, bo, co);
        }
        bo += offset;
        co += offset * ldci;
        n -= offset;
        offset = 0;
        if n <= 0 {
            return;
        }
    }
    if n > m + offset {
        if upper {
            gemm(m, n - m - offset, ao, bo + m + offset, co + (m + offset) * ldci);
        }
        n = m + offset;
        if n <= 0 {
            return;
        }
    }
    if offset < 0 {
        if upper {
            gemm(-offset, n, ao, bo, co);
        }
        ao -= offset;
        co -= offset;
        m += offset;
        offset = 0;
        if m <= 0 {
            return;
        }
    }
    if m > n - offset {
        if !upper {
            gemm(m - n + offset, n, ao + n - offset, bo, co + n - offset);
        }
        m = n + offset;
        if m <= 0 {
            return;
        }
    }
    let mn = GEMM_UNROLL_MN as isize;
    let mut sub = vec![0.0f64; GEMM_UNROLL_MN * GEMM_UNROLL_MN];
    let mut lp = 0isize;
    while lp < n {
        let mm = lp & !(mn - 1);
        let nn = mn.min(n - lp);
        if upper {
            gemm(mm, nn, ao, bo + lp, co + lp * ldci);
        }
        if flag {
            let nnu = nn as usize;
            gemm_beta(nnu, nnu, 0.0, V::new(&mut sub), nnu);
            dgemm_kernel(
                Kern::Gemm,
                nnu,
                nnu,
                k,
                alpha,
                &a[((ao + lp) * k as isize) as usize..],
                &b[((bo + lp) * k as isize) as usize..],
                V::new(&mut sub),
                nnu,
                0,
            );
            let cb = (x + y * ldc) as isize + co;
            for j in 0..nnu {
                let (i0, i1) = if upper { (0, j + 1) } else { (j, nnu) };
                for i in i0..i1 {
                    let p = (cb + lp + i as isize + (j as isize + lp) * ldci) as usize;
                    c.set(p, c.at(p) + (sub[i + j * nnu] + sub[j + i * nnu]));
                }
            }
        }
        if !upper {
            gemm(m - mm - nn, nn, ao + mm + nn, bo + lp, co + mm + nn + lp * ldci);
        }
        lp += mn;
    }
}

// ---------------------------------------------------------------------------
// DTRMM: interface/trsm.c (TRMM) + driver/level3/trmm_L.c, trmm_R.c
// ---------------------------------------------------------------------------

/// Floor to `GEMM_UNROLL_M` when above it (`trmm_L.c` row blocks).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn floor_unroll_m(min_i: usize) -> usize {
    if min_i > GEMM_UNROLL_M { (min_i / GEMM_UNROLL_M) * GEMM_UNROLL_M } else { min_i }
}

/// `DTRMM` through `scipy_dtrmm_` (single thread): `m == 0 || n == 0`
/// returns; the driver scales B by alpha with `dgemm_beta` when
/// `alpha != 1` (zeros when 0, then returns), then runs `trmm_L` / `trmm_R`
/// with the kernels' alpha fixed at 1.
///
/// Packed values: the triangular copies (`dtrmm_i*copy` inner, `dtrmm_o*copy`
/// outer) write `op(A)(r, c)` on the triangle's side, `ONE` (unit) or the
/// diagonal on it, and `ZERO` on the other side inside the diagonal tiles
/// (the tiles they skip are never read by the kernel windows). `r` is the
/// kernel's row/`l` index and `c` its `l`/column index, both absolute.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dtrmm(
    left: bool,
    upper: bool,
    trans: bool,
    nounit: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: D,
    lda: usize,
    b: D,
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha != 1.0 {
        gemm_beta(m, n, alpha, b, ldb);
    }
    if alpha == 0.0 {
        return;
    }
    let opa = |r: usize, c: usize| if trans { a.at(c + r * lda) } else { a.at(r + c * lda) };
    let diag = |r: usize| if nounit { opa(r, r) } else { 1.0 };
    // "Upper in (r, c)": nonzero for c >= r. "Lower": c <= r.
    let tri_up = |r: usize, c: usize| {
        if c > r {
            opa(r, c)
        } else if c == r {
            diag(r)
        } else {
            0.0
        }
    };
    let tri_lo = |r: usize, c: usize| {
        if c < r {
            opa(r, c)
        } else if c == r {
            diag(r)
        } else {
            0.0
        }
    };
    let bget = |r: usize, c: usize| b.at(r + c * ldb);
    let (mut sa, mut sb) = (Vec::new(), Vec::new());
    if left {
        if upper != trans {
            // trmm_L.c, `(UPPER && !TRANSA) || (!UPPER && TRANSA)`:
            // TRMM_KERNEL_LN with TRMM_IUTCOPY / TRMM_ILNCOPY.
            let mut js = 0;
            while js < n {
                let min_j = (n - js).min(GEMM_R);
                let mut min_l = m.min(GEMM_Q);
                let mut min_i = floor_unroll_m(min_l.min(GEMM_P));
                pack_inner(&mut sa, min_i, min_l, tri_up);
                let mut jjs = js;
                while jjs < js + min_j {
                    let min_jj = min_jj_unroll(min_j + js - jjs);
                    let at = min_l * (jjs - js);
                    pack_outer(&mut sb, at, min_l, min_jj, |l, j| bget(l, jjs + j));
                    dgemm_kernel(Kern::TrmmLN, min_i, min_jj, min_l, 1.0, &sa, &sb[at..], b.off(jjs * ldb), ldb, 0);
                    jjs += min_jj;
                }
                let mut is = min_i;
                while is < min_l {
                    min_i = floor_unroll_m((min_l - is).min(GEMM_P));
                    pack_inner(&mut sa, min_i, min_l, |i, l| tri_up(is + i, l));
                    dgemm_kernel(
                        Kern::TrmmLN,
                        min_i,
                        min_j,
                        min_l,
                        1.0,
                        &sa,
                        &sb,
                        b.off(is + js * ldb),
                        ldb,
                        is as isize,
                    );
                    is += min_i;
                }
                let mut ls = min_l;
                while ls < m {
                    min_l = (m - ls).min(GEMM_Q);
                    min_i = floor_unroll_m(ls.min(GEMM_P));
                    // GEMM_ITCOPY / GEMM_INCOPY
                    pack_inner(&mut sa, min_i, min_l, |i, l| opa(i, ls + l));
                    let mut jjs = js;
                    while jjs < js + min_j {
                        let min_jj = min_jj_unroll(min_j + js - jjs);
                        let at = min_l * (jjs - js);
                        pack_outer(&mut sb, at, min_l, min_jj, |l, j| bget(ls + l, jjs + j));
                        dgemm_kernel(Kern::Gemm, min_i, min_jj, min_l, 1.0, &sa, &sb[at..], b.off(jjs * ldb), ldb, 0);
                        jjs += min_jj;
                    }
                    let mut is = min_i;
                    while is < ls {
                        min_i = floor_unroll_m((ls - is).min(GEMM_P));
                        pack_inner(&mut sa, min_i, min_l, |i, l| opa(is + i, ls + l));
                        dgemm_kernel(Kern::Gemm, min_i, min_j, min_l, 1.0, &sa, &sb, b.off(is + js * ldb), ldb, 0);
                        is += min_i;
                    }
                    let mut is = ls;
                    while is < ls + min_l {
                        min_i = floor_unroll_m((ls + min_l - is).min(GEMM_P));
                        pack_inner(&mut sa, min_i, min_l, |i, l| tri_up(is + i, ls + l));
                        dgemm_kernel(
                            Kern::TrmmLN,
                            min_i,
                            min_j,
                            min_l,
                            1.0,
                            &sa,
                            &sb,
                            b.off(is + js * ldb),
                            ldb,
                            (is - ls) as isize,
                        );
                        is += min_i;
                    }
                    ls += GEMM_Q;
                }
                js += GEMM_R;
            }
        } else {
            // trmm_L.c, `(!UPPER && !TRANSA) || (UPPER && TRANSA)`:
            // TRMM_KERNEL_LT with TRMM_ILTCOPY / TRMM_IUNCOPY.
            let mut js = 0;
            while js < n {
                let min_j = (n - js).min(GEMM_R);
                let mut min_l = m.min(GEMM_Q);
                let mut min_i = floor_unroll_m(min_l.min(GEMM_P));
                let base = m - min_l;
                pack_inner(&mut sa, min_i, min_l, |i, l| tri_lo(base + i, base + l));
                let mut jjs = js;
                while jjs < js + min_j {
                    let min_jj = min_jj_unroll(min_j + js - jjs);
                    let at = min_l * (jjs - js);
                    pack_outer(&mut sb, at, min_l, min_jj, |l, j| bget(base + l, jjs + j));
                    dgemm_kernel(
                        Kern::TrmmLT,
                        min_i,
                        min_jj,
                        min_l,
                        1.0,
                        &sa,
                        &sb[at..],
                        b.off(base + jjs * ldb),
                        ldb,
                        0,
                    );
                    jjs += min_jj;
                }
                let mut is = base + min_i;
                while is < m {
                    min_i = floor_unroll_m((m - is).min(GEMM_P));
                    pack_inner(&mut sa, min_i, min_l, |i, l| tri_lo(is + i, base + l));
                    dgemm_kernel(
                        Kern::TrmmLT,
                        min_i,
                        min_j,
                        min_l,
                        1.0,
                        &sa,
                        &sb,
                        b.off(is + js * ldb),
                        ldb,
                        (is - base) as isize,
                    );
                    is += min_i;
                }
                let mut lsi = (m - min_l) as isize;
                while lsi > 0 {
                    let ls = lsi as usize;
                    min_l = ls.min(GEMM_Q);
                    min_i = floor_unroll_m(min_l.min(GEMM_P));
                    let base = ls - min_l;
                    pack_inner(&mut sa, min_i, min_l, |i, l| tri_lo(base + i, base + l));
                    let mut jjs = js;
                    while jjs < js + min_j {
                        let min_jj = min_jj_unroll(min_j + js - jjs);
                        let at = min_l * (jjs - js);
                        pack_outer(&mut sb, at, min_l, min_jj, |l, j| bget(base + l, jjs + j));
                        dgemm_kernel(
                            Kern::TrmmLT,
                            min_i,
                            min_jj,
                            min_l,
                            1.0,
                            &sa,
                            &sb[at..],
                            b.off(base + jjs * ldb),
                            ldb,
                            0,
                        );
                        jjs += min_jj;
                    }
                    let mut is = base + min_i;
                    while is < ls {
                        min_i = floor_unroll_m((ls - is).min(GEMM_P));
                        pack_inner(&mut sa, min_i, min_l, |i, l| tri_lo(is + i, base + l));
                        dgemm_kernel(
                            Kern::TrmmLT,
                            min_i,
                            min_j,
                            min_l,
                            1.0,
                            &sa,
                            &sb,
                            b.off(is + js * ldb),
                            ldb,
                            (is - base) as isize,
                        );
                        is += min_i;
                    }
                    let mut is = ls;
                    while is < m {
                        min_i = floor_unroll_m((m - is).min(GEMM_P));
                        pack_inner(&mut sa, min_i, min_l, |i, l| opa(is + i, base + l));
                        dgemm_kernel(Kern::Gemm, min_i, min_j, min_l, 1.0, &sa, &sb, b.off(is + js * ldb), ldb, 0);
                        is += min_i;
                    }
                    lsi -= GEMM_Q as isize;
                }
                js += GEMM_R;
            }
        }
    } else if upper == trans {
        // trmm_R.c, `(!UPPER && !TRANSA) || (UPPER && TRANSA)`:
        // TRMM_KERNEL_RT with TRMM_OLNCOPY / TRMM_OUTCOPY.
        let mut js = 0;
        while js < n {
            let min_j = (n - js).min(GEMM_R);
            let mut ls = js;
            while ls < js + min_j {
                let min_l = (js + min_j - ls).min(GEMM_Q);
                let mut min_i = m.min(GEMM_P);
                pack_inner(&mut sa, min_i, min_l, |i, l| bget(i, ls + l)); // GEMM_ITCOPY of B
                let mut jjs = 0;
                while jjs < ls - js {
                    let min_jj = min_jj_unroll(ls - js - jjs);
                    let at = min_l * jjs;
                    pack_outer(&mut sb, at, min_l, min_jj, |l, j| opa(ls + l, js + jjs + j));
                    dgemm_kernel(
                        Kern::Gemm,
                        min_i,
                        min_jj,
                        min_l,
                        1.0,
                        &sa,
                        &sb[at..],
                        b.off((js + jjs) * ldb),
                        ldb,
                        0,
                    );
                    jjs += min_jj;
                }
                let mut jjs = 0;
                while jjs < min_l {
                    let min_jj = min_jj_unroll(min_l - jjs);
                    let at = min_l * (ls - js + jjs);
                    pack_outer(&mut sb, at, min_l, min_jj, |l, j| tri_lo(ls + l, ls + jjs + j));
                    dgemm_kernel(
                        Kern::TrmmRT,
                        min_i,
                        min_jj,
                        min_l,
                        1.0,
                        &sa,
                        &sb[at..],
                        b.off((ls + jjs) * ldb),
                        ldb,
                        -(jjs as isize),
                    );
                    jjs += min_jj;
                }
                let mut is = min_i;
                while is < m {
                    min_i = (m - is).min(GEMM_P);
                    pack_inner(&mut sa, min_i, min_l, |i, l| bget(is + i, ls + l));
                    dgemm_kernel(Kern::Gemm, min_i, ls - js, min_l, 1.0, &sa, &sb, b.off(is + js * ldb), ldb, 0);
                    dgemm_kernel(
                        Kern::TrmmRT,
                        min_i,
                        min_l,
                        min_l,
                        1.0,
                        &sa,
                        &sb[(ls - js) * min_l..],
                        b.off(is + ls * ldb),
                        ldb,
                        0,
                    );
                    is += GEMM_P;
                }
                ls += GEMM_Q;
            }
            // `for (ls = js + min_j; ls < n; ...)`: empty while n <= GEMM_R.
            assert!(js + min_j >= n, "dtrmm: n > GEMM_R column blocks are not transcribed");
            js += GEMM_R;
        }
    } else {
        // trmm_R.c, `(UPPER && !TRANSA) || (!UPPER && TRANSA)`:
        // TRMM_KERNEL_RN with TRMM_OUNCOPY / TRMM_OLTCOPY, columns backwards.
        assert!(n <= GEMM_R, "dtrmm: n > GEMM_R column blocks are not transcribed");
        let js = n;
        let min_j = js.min(GEMM_R);
        let mut start_ls = js - min_j;
        while start_ls + GEMM_Q < js {
            start_ls += GEMM_Q;
        }
        let mut ls = start_ls as isize;
        while ls >= (js - min_j) as isize {
            let lsu = ls as usize;
            let min_l = (js - lsu).min(GEMM_Q);
            let mut min_i = m.min(GEMM_P);
            pack_inner(&mut sa, min_i, min_l, |i, l| bget(i, lsu + l)); // GEMM_ITCOPY of B
            let mut jjs = 0;
            while jjs < min_l {
                let min_jj = min_jj_unroll(min_l - jjs);
                let at = min_l * jjs;
                pack_outer(&mut sb, at, min_l, min_jj, |l, j| tri_up(lsu + l, lsu + jjs + j));
                dgemm_kernel(
                    Kern::TrmmRN,
                    min_i,
                    min_jj,
                    min_l,
                    1.0,
                    &sa,
                    &sb[at..],
                    b.off((lsu + jjs) * ldb),
                    ldb,
                    -(jjs as isize),
                );
                jjs += min_jj;
            }
            let mut jjs = 0;
            while jjs < js - lsu - min_l {
                let min_jj = min_jj_unroll(js - lsu - min_l - jjs);
                let at = min_l * (min_l + jjs);
                pack_outer(&mut sb, at, min_l, min_jj, |l, j| opa(lsu + l, lsu + min_l + jjs + j));
                dgemm_kernel(
                    Kern::Gemm,
                    min_i,
                    min_jj,
                    min_l,
                    1.0,
                    &sa,
                    &sb[at..],
                    b.off((lsu + min_l + jjs) * ldb),
                    ldb,
                    0,
                );
                jjs += min_jj;
            }
            let mut is = min_i;
            while is < m {
                min_i = (m - is).min(GEMM_P);
                pack_inner(&mut sa, min_i, min_l, |i, l| bget(is + i, lsu + l));
                dgemm_kernel(Kern::TrmmRN, min_i, min_l, min_l, 1.0, &sa, &sb, b.off(is + lsu * ldb), ldb, 0);
                if js - lsu - min_l > 0 {
                    dgemm_kernel(
                        Kern::Gemm,
                        min_i,
                        js - lsu - min_l,
                        min_l,
                        1.0,
                        &sa,
                        &sb[min_l * min_l..],
                        b.off(is + (lsu + min_l) * ldb),
                        ldb,
                        0,
                    );
                }
                is += GEMM_P;
            }
            ls -= GEMM_Q as isize;
        }
    }
}
