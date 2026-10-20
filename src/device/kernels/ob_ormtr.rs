//! **OpenBLAS track.** `DLARFB('L','N',*,'C')`'s BLAS steps in OpenBLAS
//! 0.3.30 SkylakeX order (host transcription:
//! [`crate::openblas::level3`]), on the row-major `n × kpad` work buffers of
//! the Reference `DORMTR` engine (`W(i, j)` at `i·kpad + j`).
//!
//! * [`ob_trmm_kernel`] — `DTRMM('R', UPLO, TRANS, DIAG, n, k, ONE, A, k,
//!   W, n)`: rows of `W` are independent (the driver packs each row block
//!   before the micro-kernel overwrites it), so `W_out(i, j)` is the TRMM
//!   micro-kernel's chain over the packed row `W_in(i, ·)` and the packed
//!   triangle column `op(A)(·, j)` (explicit zeros inside the diagonal tiles,
//!   the unit diagonal as `1`). Its `l` window and lane mode follow the
//!   element's row tile (16/8/4/2/1 inside a `GEMM_P` row block), its column
//!   tile (`6`/`2` chunks in the first row block, 12/10/.../1 tiles later)
//!   and the `RT`/`RN` window rules of `dgemm_kernel_16x2_skylakex.c`.
//! * [`ob_gemm_tn_kernel`] — `DGEMM('T','N', n, k, T, ONE, C₂, V₂, ONE, W)`.
//! * [`ob_gemm_nt_kernel`] — `DGEMM('N','T', T, n, k, −ONE, V₂, W, ONE, C₂)`,
//!   in place on `C₂`.

use cubecl::prelude::*;

use super::ob_syr2k::acc_mode_pub as acc_mode;

const GEMM_P: u32 = 192;
const GEMM_Q: u32 = 384;
const UNROLL_M: u32 = 16;

/// `level3.c` block rule along `m` (`p = GEMM_P`) or `k` (`p = GEMM_Q`).
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

/// TRMM micro-kernel element: `acc` over `l in lo..hi` with lane `mode`
/// continuing from `acc0`, terms `W_in(i, l)·tri(l, j)` (`tri` the packed
/// triangle, column-major `k × k`: explicit zeros, unit diagonal as `1`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn trmm_acc<F: Float + CubeElement>(
    w_in: &Array<F>,
    tri: &Array<F>,
    kpad: u32,
    k: u32,
    i: u32,
    j: u32,
    lo: u32,
    hi: u32,
    mode: u32,
    acc0: F,
) -> F {
    let wb = i * kpad;
    let tb = j * k;
    let mut acc = acc0;
    let mut l = lo;
    let zero = F::from_int(0);
    if mode == 1u32 && hi >= lo + 2u32 {
        let mut lo2 = acc;
        let mut hi2 = zero;
        while l + 2u32 <= hi {
            lo2 = fma(w_in[(wb + l) as usize], tri[(tb + l) as usize], lo2);
            hi2 = fma(w_in[(wb + l + 1u32) as usize], tri[(tb + l + 1u32) as usize], hi2);
            l += 2u32;
        }
        acc = hi2 + lo2;
    }
    if mode == 2u32 && hi >= lo + 4u32 {
        let mut s0 = acc;
        let mut s1 = zero;
        let mut s2 = zero;
        let mut s3 = zero;
        while l + 4u32 <= hi {
            s0 = fma(w_in[(wb + l) as usize], tri[(tb + l) as usize], s0);
            s1 = fma(w_in[(wb + l + 1u32) as usize], tri[(tb + l + 1u32) as usize], s1);
            s2 = fma(w_in[(wb + l + 2u32) as usize], tri[(tb + l + 2u32) as usize], s2);
            s3 = fma(w_in[(wb + l + 3u32) as usize], tri[(tb + l + 3u32) as usize], s3);
            l += 4u32;
        }
        acc = (s3 + s2) + (s1 + s0);
    }
    while l < hi {
        acc = fma(w_in[(wb + l) as usize], tri[(tb + l) as usize], acc);
        l += 1u32;
    }
    acc
}

/// `DTRMM('R', UPLO, TRANS, DIAG)` on `W` (`n × k`, row-major `kpad`), out of
/// place. `rt`: the `RT` kernel (`UPLO == TRANS` in `trmm_R.c`), else `RN`.
/// The host supplies the packed triangle `tri(l, j)` (column-major `k × k`),
/// each row's tile size and whether it is in the first `GEMM_P` row block
/// (`rows[i] = md | first << 8`), and each column's tile start and size in
/// the first row block (`cols[2j]`, `cols[2j+1]`, the `6`/`2` chunks) and in
/// later ones (`cols[2k+2j]`, `cols[2k+2j+1]`). One unit per element.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_trmm_kernel<F: Float + CubeElement>(
    w_in: &Array<F>,
    w_out: &mut Array<F>,
    tri: &Array<F>,
    rows: &Array<u32>,
    cols: &Array<u32>,
    n: u32,
    k: u32,
    kpad: u32,
    #[comptime] rt: bool,
) {
    if ABSOLUTE_POS < (n * k) as usize {
        let u = ABSOLUTE_POS as u32;
        let i = u / k;
        let j = u % k;
        let ri = rows[i as usize];
        let md = ri & 255u32;
        let later = 1u32 - (ri >> 8u32);
        let ci = (2u32 * k * later + 2u32 * j) as usize;
        let j0 = cols[ci];
        let nd = cols[ci + 1];
        let mut mode = 0u32;
        if nd == 12u32 && md == 4u32 {
            mode = 1u32;
        }
        if nd == 12u32 && md <= 2u32 {
            mode = 2u32;
        }
        let p = (j - j0) / 2u32;
        let zero = F::from_int(0);
        if rt {
            let steps = select(nd >= 4u32, nd / 2u32 - 1u32, 0u32);
            let a0 = trmm_acc::<F>(w_in, tri, kpad, k, i, j, j0 + 2u32 * p, j0 + 2u32 * steps, 0u32, zero);
            w_out[(i * kpad + j) as usize] = trmm_acc::<F>(w_in, tri, kpad, k, i, j, j0 + 2u32 * steps, k, mode, a0);
        } else {
            let r13 = j0 + select(nd < 2u32, nd, 2u32);
            let a0 = trmm_acc::<F>(w_in, tri, kpad, k, i, j, 0u32, r13, mode, zero);
            w_out[(i * kpad + j) as usize] = trmm_acc::<F>(w_in, tri, kpad, k, i, j, r13, r13 + 2u32 * p, 0u32, a0);
        }
    }
}

/// `W_out(i, j) = W_in(i, j) + (C₂ᵀ·V₂)(i, j)`: `DGEMM('T','N', n, k, t, ONE,
/// C, lda, V, ·, ONE, W)` in OpenBLAS order, `C(l, i) = c[i·lda + c_row0 + l]`
/// (`i` indexes the columns of `Z`), `V(l, j) = v_rm[(vrow0 + l)·kpad + j]`.
/// `small`: the small-matrix `tn` path (decided on the host). Units are the
/// elements `(i, j) = (elems[2u], elems[2u+1])`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_gemm_tn_kernel<F: Float + CubeElement>(
    c: &Array<F>,
    v_rm: &Array<F>,
    w_in: &Array<F>,
    w_out: &mut Array<F>,
    elems: &Array<u32>,
    lda: u32,
    kpad: u32,
    c_row0: u32,
    vrow0: u32,
    t: u32,
    n: u32,
    k: u32,
    small: u32,
    count: u32,
) {
    if ABSOLUTE_POS < count as usize {
        let u = ABSOLUTE_POS as u32;
        let i = elems[(2u32 * u) as usize];
        let j = elems[(2u32 * u + 1u32) as usize];
        let zero = F::from_int(0);
        let cb = i * lda + c_row0;
        let vb = vrow0 * kpad + j;
        let mut val = w_in[(i * kpad + j) as usize];
        if small != 0u32 {
            // tn: dot style everywhere; every store is `C + sum` exactly
            // (alpha = beta = 1).
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
                e0 = fma(c[(cb + l) as usize], v_rm[(vb + l * kpad) as usize], e0);
                e1 = fma(c[(cb + l + 1u32) as usize], v_rm[(vb + (l + 1u32) * kpad) as usize], e1);
                e2 = fma(c[(cb + l + 2u32) as usize], v_rm[(vb + (l + 2u32) * kpad) as usize], e2);
                e3 = fma(c[(cb + l + 3u32) as usize], v_rm[(vb + (l + 3u32) * kpad) as usize], e3);
                e4 = fma(c[(cb + l + 4u32) as usize], v_rm[(vb + (l + 4u32) * kpad) as usize], e4);
                e5 = fma(c[(cb + l + 5u32) as usize], v_rm[(vb + (l + 5u32) * kpad) as usize], e5);
                e6 = fma(c[(cb + l + 6u32) as usize], v_rm[(vb + (l + 6u32) * kpad) as usize], e6);
                e7 = fma(c[(cb + l + 7u32) as usize], v_rm[(vb + (l + 7u32) * kpad) as usize], e7);
                l += 8u32;
            }
            let rem = t - t8;
            if rem != 0u32 {
                let mut lane = 0u32;
                while lane < 8u32 {
                    let mut x = zero;
                    let mut y = zero;
                    if lane < rem {
                        x = c[(cb + t8 + lane) as usize];
                        y = v_rm[(vb + (t8 + lane) * kpad) as usize];
                    }
                    if lane == 0u32 {
                        e0 = fma(x, y, e0);
                    } else if lane == 1u32 {
                        e1 = fma(x, y, e1);
                    } else if lane == 2u32 {
                        e2 = fma(x, y, e2);
                    } else if lane == 3u32 {
                        e3 = fma(x, y, e3);
                    } else if lane == 4u32 {
                        e4 = fma(x, y, e4);
                    } else if lane == 5u32 {
                        e5 = fma(x, y, e5);
                    } else if lane == 6u32 {
                        e6 = fma(x, y, e6);
                    } else {
                        e7 = fma(x, y, e7);
                    }
                    lane += 1u32;
                }
            }
            if i < (n & !3u32) || j < (k & !3u32) {
                let sum4 = ((e0 + e1) + (e2 + e3)) + ((e4 + e5) + (e6 + e7));
                val = fma(F::from_int(1), val, sum4);
            } else {
                let sum512 = ((e0 + e4) + (e2 + e6)) + ((e1 + e5) + (e3 + e7));
                val = fma(sum512, F::from_int(1), val);
            }
        } else {
            let first = gemm_block(n, GEMM_P);
            let mut mode = 0u32;
            if i >= first {
                let mut is = first;
                let mut mi = gemm_block(n - is, GEMM_P);
                while i >= is + mi {
                    is += mi;
                    mi = gemm_block(n - is, GEMM_P);
                }
                mode = acc_mode(i - is, mi, j, k);
            }
            let mut ls = 0u32;
            while ls < t {
                let ml = gemm_block(t - ls, GEMM_Q);
                let end = ls + ml;
                let mut acc = zero;
                let mut l = ls;
                if mode == 1u32 && ml >= 2u32 {
                    let mut lo2 = zero;
                    let mut hi2 = zero;
                    while end - l >= 2u32 {
                        lo2 = fma(c[(cb + l) as usize], v_rm[(vb + l * kpad) as usize], lo2);
                        hi2 = fma(c[(cb + l + 1u32) as usize], v_rm[(vb + (l + 1u32) * kpad) as usize], hi2);
                        l += 2u32;
                    }
                    acc = hi2 + lo2;
                } else if mode == 2u32 && ml >= 4u32 {
                    let mut s0 = zero;
                    let mut s1 = zero;
                    let mut s2 = zero;
                    let mut s3 = zero;
                    while end - l >= 4u32 {
                        s0 = fma(c[(cb + l) as usize], v_rm[(vb + l * kpad) as usize], s0);
                        s1 = fma(c[(cb + l + 1u32) as usize], v_rm[(vb + (l + 1u32) * kpad) as usize], s1);
                        s2 = fma(c[(cb + l + 2u32) as usize], v_rm[(vb + (l + 2u32) * kpad) as usize], s2);
                        s3 = fma(c[(cb + l + 3u32) as usize], v_rm[(vb + (l + 3u32) * kpad) as usize], s3);
                        l += 4u32;
                    }
                    acc = (s3 + s2) + (s1 + s0);
                }
                while l < end {
                    acc = fma(c[(cb + l) as usize], v_rm[(vb + l * kpad) as usize], acc);
                    l += 1u32;
                }
                val = fma(acc, F::from_int(1), val);
                ls += ml;
            }
        }
        w_out[(i * kpad + j) as usize] = val;
    }
}

/// `C(r, i) -= (V₂·Wᵀ)(r, i)` in place: `DGEMM('N','T', t, n, k, −ONE, V, ·,
/// W, n, ONE, C)` in OpenBLAS order, `C(r, i) = c[i·lda + c_row0 + r]`,
/// `V(r, l) = v_rm[(vrow0 + r)·kpad + l]`, `W(i, l) = w[i·kpad + l]`. Units
/// are the elements `(r, i) = (elems[2u], elems[2u+1])`; `small`: the
/// small-matrix `nt` path.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_gemm_nt_kernel<F: Float + CubeElement>(
    c: &mut Array<F>,
    v_rm: &Array<F>,
    w: &Array<F>,
    elems: &Array<u32>,
    lda: u32,
    kpad: u32,
    c_row0: u32,
    vrow0: u32,
    t: u32,
    n: u32,
    k: u32,
    small: u32,
    count: u32,
) {
    if ABSOLUTE_POS < count as usize {
        let u = ABSOLUTE_POS as u32;
        let r = elems[(2u32 * u) as usize];
        let i = elems[(2u32 * u + 1u32) as usize];
        let zero = F::from_int(0);
        let vb = (vrow0 + r) * kpad;
        let wb = i * kpad;
        let p = (i * lda + c_row0 + r) as usize;
        let mut mode = 0u32;
        if small == 0u32 {
            let first = gemm_block(t, GEMM_P);
            if r >= first {
                let mut is = first;
                let mut mi = gemm_block(t - is, GEMM_P);
                while r >= is + mi {
                    is += mi;
                    mi = gemm_block(t - is, GEMM_P);
                }
                mode = acc_mode(r - is, mi, i, n);
            }
        }
        let mut acc = zero;
        let mut l = 0u32;
        if mode == 1u32 && k >= 2u32 {
            let mut lo2 = zero;
            let mut hi2 = zero;
            while k - l >= 2u32 {
                lo2 = fma(v_rm[(vb + l) as usize], w[(wb + l) as usize], lo2);
                hi2 = fma(v_rm[(vb + l + 1u32) as usize], w[(wb + l + 1u32) as usize], hi2);
                l += 2u32;
            }
            acc = hi2 + lo2;
        } else if mode == 2u32 && k >= 4u32 {
            let mut s0 = zero;
            let mut s1 = zero;
            let mut s2 = zero;
            let mut s3 = zero;
            while k - l >= 4u32 {
                s0 = fma(v_rm[(vb + l) as usize], w[(wb + l) as usize], s0);
                s1 = fma(v_rm[(vb + l + 1u32) as usize], w[(wb + l + 1u32) as usize], s1);
                s2 = fma(v_rm[(vb + l + 2u32) as usize], w[(wb + l + 2u32) as usize], s2);
                s3 = fma(v_rm[(vb + l + 3u32) as usize], w[(wb + l + 3u32) as usize], s3);
                l += 4u32;
            }
            acc = (s3 + s2) + (s1 + s0);
        }
        while l < k {
            acc = fma(v_rm[(vb + l) as usize], w[(wb + l) as usize], acc);
            l += 1u32;
        }
        let neg = F::from_int(-1);
        if small != 0u32 {
            // nt chain store: beta*C + (sum*alpha), beta = 1.
            c[p] = fma(F::from_int(1), c[p], acc * neg);
        } else {
            // Kernel store: acc*alpha + C.
            c[p] = fma(acc, neg, c[p]);
        }
    }
}

/// `acc` over `l in lo..hi` of lane-vector terms `x(l)·y(l)` with lane `mode`
/// (0 plain chain, 1 two lanes, 2 four lanes), `x` a scalar broadcast read
/// from `c[cb + l]` and `y` the row-major vector `v[(vb + l)·kv + jv]`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn acc_bv<F: Float + CubeElement, N: Size>(
    c: &Array<F>,
    v: &Array<Vector<F, N>>,
    cb: u32,
    vb: u32,
    kv: u32,
    jv: u32,
    lo: u32,
    hi: u32,
    mode: u32,
) -> Vector<F, N> {
    let zero = Vector::<F, N>::new(F::from_int(0));
    let mut acc = zero;
    let mut l = lo;
    if mode == 1u32 && hi >= lo + 2u32 {
        let mut lo2 = zero;
        let mut hi2 = zero;
        while l + 2u32 <= hi {
            lo2 = fma(Vector::<F, N>::new(c[(cb + l) as usize]), v[((vb + l) * kv + jv) as usize], lo2);
            hi2 = fma(Vector::<F, N>::new(c[(cb + l + 1u32) as usize]), v[((vb + l + 1u32) * kv + jv) as usize], hi2);
            l += 2u32;
        }
        acc = hi2 + lo2;
    }
    if mode == 2u32 && hi >= lo + 4u32 {
        let mut s0 = zero;
        let mut s1 = zero;
        let mut s2 = zero;
        let mut s3 = zero;
        while l + 4u32 <= hi {
            s0 = fma(Vector::<F, N>::new(c[(cb + l) as usize]), v[((vb + l) * kv + jv) as usize], s0);
            s1 = fma(Vector::<F, N>::new(c[(cb + l + 1u32) as usize]), v[((vb + l + 1u32) * kv + jv) as usize], s1);
            s2 = fma(Vector::<F, N>::new(c[(cb + l + 2u32) as usize]), v[((vb + l + 2u32) * kv + jv) as usize], s2);
            s3 = fma(Vector::<F, N>::new(c[(cb + l + 3u32) as usize]), v[((vb + l + 3u32) * kv + jv) as usize], s3);
            l += 4u32;
        }
        acc = (s3 + s2) + (s1 + s0);
    }
    while l < hi {
        acc = fma(Vector::<F, N>::new(c[(cb + l) as usize]), v[((vb + l) * kv + jv) as usize], acc);
        l += 1u32;
    }
    acc
}

/// [`ob_gemm_tn_kernel`] (blocked driver only) with lane-vectors over the
/// columns `j` of `W`: units `(i, jv)`. The element mode depends on the row
/// `i` (its tile inside a later `GEMM_P` row block) and on `j < 12·q12(k)`;
/// units whose real lanes (`j < k`) do not share one mode are skipped (the
/// scalar list kernel covers them), padding lanes are don't-care.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_gemm_tn_vec_kernel<F: Float + CubeElement, N: Size>(
    c: &Array<F>,
    v_rm: &Array<Vector<F, N>>,
    w_in: &Array<Vector<F, N>>,
    w_out: &mut Array<Vector<F, N>>,
    lda: u32,
    kpad: u32,
    c_row0: u32,
    vrow0: u32,
    t: u32,
    nrows: u32,
    k: u32,
) {
    let nl = N::value() as u32;
    let kv = kpad / nl;
    if ABSOLUTE_POS < (nrows * kv) as usize {
        let u = ABSOLUTE_POS as u32;
        let i = u / kv;
        let jv = u % kv;
        let jf = jv * nl;
        let jl_end = select(jf + nl < k, jf + nl, k);
        let first = gemm_block(nrows, GEMM_P);
        let mut mode_f = 0u32;
        let mut mode_l = 0u32;
        if i >= first && jf < k {
            let mut is = first;
            let mut mi = gemm_block(nrows - is, GEMM_P);
            while i >= is + mi {
                is += mi;
                mi = gemm_block(nrows - is, GEMM_P);
            }
            mode_f = acc_mode(i - is, mi, jf, k);
            mode_l = acc_mode(i - is, mi, jl_end - 1u32, k);
        }
        if mode_f == mode_l {
            let cb = i * lda + c_row0;
            let mut val = w_in[(i * kv + jv) as usize];
            let one = Vector::<F, N>::new(F::from_int(1));
            let mut ls = 0u32;
            while ls < t {
                let ml = gemm_block(t - ls, GEMM_Q);
                let acc = acc_bv::<F, N>(c, v_rm, cb, vrow0, kv, jv, ls, ls + ml, mode_f);
                val = fma(acc, one, val);
                ls += ml;
            }
            w_out[(i * kv + jv) as usize] = val;
        }
    }
}

/// [`ob_gemm_nt_kernel`] with lane-vectors over the rows of `C₂` (column
/// `i` of `Z`): units `(rv, i)`, `V₂` column-major `v_cm[l·ldv + g − vbase]`
/// for global row `g` (`vbase = c_row0 − pad`, a lane-vector boundary). Every
/// lane is `C − Σ_l V₂(g, l)·W(i, l)` as the plain chain; units with a row in
/// a special interval (`sp`, block-relative pairs: the 4/2/1 row tiles of a
/// later `GEMM_P` block) of a column `i < thr` are skipped, and rows outside
/// `[c_row0, c_row0 + t)` keep their value.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_gemm_nt_vec_kernel<F: Float + CubeElement, N: Size>(
    z: &mut Array<Vector<F, N>>,
    v_cm: &Array<Vector<F, N>>,
    w: &Array<F>,
    sp: &Array<u32>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda_vecs: u32,
    ldv_vecs: u32,
    vbase_vecs: u32,
    rv0: u32,
    units_x: u32,
    units: u32,
    c_row0: u32,
    t: u32,
    kpad: u32,
    k: u32,
    thr: u32,
    nsp: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let rv = rv0 + u % units_x;
        let i = u / units_x;
        let nl = N::value() as u32;
        let first = rv * nl;
        let r0 = select(first > c_row0, first - c_row0, 0u32);
        let lastp = first + nl;
        let hi = select(lastp < c_row0 + t, lastp, c_row0 + t);
        let r1 = select(hi > c_row0, hi - c_row0, 0u32);
        let mut special = false;
        if i < thr {
            for q in 0..nsp {
                if sp[(2u32 * q) as usize] < r1 && r0 < sp[(2u32 * q + 1u32) as usize] {
                    special = true;
                }
            }
        }
        if !special {
            let rows = Vector::<u32, N>::new(first) + lane_ofs[0];
            let keep = rows
                .greater_equal(Vector::<u32, N>::new(c_row0))
                .and(rows.less_than(Vector::<u32, N>::new(c_row0 + t)));
            let idx = (i * lda_vecs + rv) as usize;
            let old = z[idx];
            let mut acc = Vector::<F, N>::new(F::from_int(0));
            let vr = rv - vbase_vecs;
            for l in 0..k {
                acc = fma(v_cm[(l * ldv_vecs + vr) as usize], Vector::<F, N>::new(w[(i * kpad + l) as usize]), acc);
            }
            let val = fma(acc, Vector::<F, N>::new(F::from_int(-1)), old);
            z[idx] = select_many(keep, val, old);
        }
    }
}
