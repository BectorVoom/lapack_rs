//! K1: `DSYMV` (`BETA = 0`) on a fully symmetric device matrix.
//!
//! A unit owns one `Vector<F, N>` of `N` adjacent rows of the trailing block
//! `M[off.., off..]`. Each column step `p` loads one lane-vector from column
//! `p` (stride 1 across lanes) and the scalar `x[p]` (a broadcast). Because
//! both triangles hold identical bits, the row read `M[r, p]` is exactly the
//! reference's `A(p, r)` when `p > r`, so each lane replays reference `dsymv`
//! for its own row:
//!
//! * lower: `y = Σ_{p<r} t1_p·A(r,p)` then `+ t1_r·A(r,r)` (in `p` order),
//!   and `+ alpha·temp2` with `temp2 = Σ_{p>r} A(p,r)·x_p`;
//! * upper: `temp2 = Σ_{p<r} A(p,r)·x_p`, `y = t1_r·A(r,r) + alpha·temp2`,
//!   then `+ t1_p·A(r,p)` for `p > r`.
//!
//! Lane roles are chosen with `select_many` masks, never a masked multiply,
//! so every lane's chain is the reference chain bit for bit. Lanes outside
//! the block are computed and discarded by the launcher.

use cubecl::prelude::*;

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn symv_kernel<F: Float + CubeElement, N: Size>(
    m_buf: &Array<Vector<F, N>>,
    x: &Array<F>,
    lane_ofs: &Array<Vector<u32, N>>,
    y: &mut Array<Vector<F, N>>,
    lda_vecs: u32,
    off: u32,
    m: u32,
    rv0: u32,
    units: u32,
    alpha: F,
    #[comptime] upper: bool,
) {
    if ABSOLUTE_POS < units as usize {
        let rv = rv0 + ABSOLUTE_POS as u32;
        let rows = Vector::<u32, N>::new(rv * N::value() as u32) + lane_ofs[0];
        let mut acc_y = Vector::<F, N>::new(F::from_int(0));
        let mut acc_t = Vector::<F, N>::new(F::from_int(0));
        let va = Vector::<F, N>::new(alpha);
        // Columns p below the unit's first row (every lane has p < r), the N
        // columns straddling its rows (per-lane masks) and columns past its
        // last row (every lane has p > r). A uniform mask selects the same
        // value the unmasked update produces, so the split drops the unused
        // accumulator's arithmetic without changing any lane's chain.
        let first = rv * N::value() as u32;
        let q_lo = select(first > off, first - off, 0u32);
        let q_lo = select(q_lo < m, q_lo, m);
        let q_hi = select(first + N::value() as u32 > off, first + N::value() as u32 - off, 0u32);
        let q_hi = select(q_hi < m, q_hi, m);
        for q in 0..q_lo {
            let p = off + q;
            let col = m_buf[(p * lda_vecs + rv) as usize];
            let xp = x[q as usize];
            if upper {
                acc_t = acc_t + col * Vector::<F, N>::new(xp);
            } else {
                acc_y = acc_y + Vector::<F, N>::new(alpha * xp) * col;
            }
        }
        for q in q_lo..q_hi {
            let p = off + q;
            let col = m_buf[(p * lda_vecs + rv) as usize];
            let xp = x[q as usize];
            let vx = Vector::<F, N>::new(xp);
            let vt1 = Vector::<F, N>::new(alpha * xp);
            let pv = Vector::<u32, N>::new(p);
            if upper {
                let lt = pv.less_than(rows);
                let eq = pv.equal(rows);
                let gt = pv.greater_than(rows);
                acc_t = select_many(lt, acc_t + col * vx, acc_t);
                acc_y = select_many(eq, (acc_y + vt1 * col) + va * acc_t, acc_y);
                acc_y = select_many(gt, acc_y + vt1 * col, acc_y);
            } else {
                let le = pv.less_equal(rows);
                let gt = pv.greater_than(rows);
                acc_y = select_many(le, acc_y + vt1 * col, acc_y);
                acc_t = select_many(gt, acc_t + col * vx, acc_t);
            }
        }
        for q in q_hi..m {
            let p = off + q;
            let col = m_buf[(p * lda_vecs + rv) as usize];
            let xp = x[q as usize];
            if upper {
                acc_y = acc_y + Vector::<F, N>::new(alpha * xp) * col;
            } else {
                acc_t = acc_t + col * Vector::<F, N>::new(xp);
            }
        }
        if upper {
            y[rv as usize] = acc_y;
        } else {
            y[rv as usize] = acc_y + va * acc_t;
        }
    }
}

/// K1col: [`symv_kernel`] with scalar units, one per row `r`, reading row `r`
/// of the trailing block as column `r` (`M[p, r]`, identical bits to
/// `M[r, p]` because both triangles are kept bitwise equal). Consecutive
/// iterations of a unit load adjacent elements (stride 1 down the column)
/// instead of jumping a whole column per step; adjacent units stride `lda`.
/// Each row's chain is exactly K1's (same regions and operand order).
/// `alpha_one`: `ALPHA = 1` (every `DLATRD` call), where `ALPHA·x` is `x`
/// bit for bit, so the multiply is dropped.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn symv_col_kernel<F: Float + CubeElement>(
    m_buf: &Array<F>,
    x: &Array<F>,
    y: &mut Array<F>,
    lda: u32,
    off: u32,
    m: u32,
    alpha: F,
    #[comptime] upper: bool,
    #[comptime] alpha_one: bool,
) {
    if ABSOLUTE_POS < m as usize {
        let d = ABSOLUTE_POS as u32;
        let r = off + d;
        let base = r * lda + off;
        let mut acc_y = F::from_int(0);
        let mut acc_t = F::from_int(0);
        if upper {
            // p < r: temp2; p == r: y = (y + t1·A(r,r)) + alpha·temp2; p > r: y.
            for q in 0..d {
                acc_t = acc_t + m_buf[(base + q) as usize] * x[q as usize];
            }
            let a_rr = m_buf[(base + d) as usize];
            if alpha_one {
                acc_y = (acc_y + x[d as usize] * a_rr) + acc_t;
                for q in d + 1..m {
                    acc_y = acc_y + x[q as usize] * m_buf[(base + q) as usize];
                }
            } else {
                acc_y = (acc_y + (alpha * x[d as usize]) * a_rr) + alpha * acc_t;
                for q in d + 1..m {
                    acc_y = acc_y + (alpha * x[q as usize]) * m_buf[(base + q) as usize];
                }
            }
            y[r as usize] = acc_y;
        } else {
            // p <= r: y; p > r: temp2; then y + alpha·temp2.
            if alpha_one {
                for q in 0..d + 1 {
                    acc_y = acc_y + x[q as usize] * m_buf[(base + q) as usize];
                }
            } else {
                for q in 0..d + 1 {
                    acc_y = acc_y + (alpha * x[q as usize]) * m_buf[(base + q) as usize];
                }
            }
            for q in d + 1..m {
                acc_t = acc_t + m_buf[(base + q) as usize] * x[q as usize];
            }
            if alpha_one {
                y[r as usize] = acc_y + acc_t;
            } else {
                y[r as usize] = acc_y + alpha * acc_t;
            }
        }
    }
}

/// One reference step of K1col's chain: `acc + x·A`, `acc + (α·x)·A` or
/// `acc + A·x`, matching the reference expression of the region it is used
/// for. Written with a mutable local rather than an `if` expression (manual:
/// Conditionals).
#[cube]
fn symv_step<F: Float + CubeElement>(
    acc: F,
    a: F,
    xq: F,
    alpha: F,
    #[comptime] x_lhs: bool,
    #[comptime] scale: bool,
) -> F {
    let mut out = acc + a * xq;
    if x_lhs {
        if scale {
            out = acc + (alpha * xq) * a;
        } else {
            out = acc + xq * a;
        }
    }
    out
}

/// `acc + Σ_{q=lo}^{hi−1} (…)` in ascending `q`, reading `M`'s column through
/// the lane-vector view where the global index allows it.
///
/// K1col issues one 8-byte load per step against a 32-byte sector: `ncu`
/// counts 1,184,301 L1 sector requests for 8.3 MB of useful data (4.6x), and
/// with only `m` units — 32 warps at `m = 1021` — the kernel is latency-bound
/// (`sm__throughput` 24%, `dram__throughput` 28%). A `Vector<F, N>` load
/// covers `N` steps with one request.
///
/// The global index of step `q` is `r·lda + off + q`, and `lda` is a multiple
/// of the lane width, so the vector body needs `(off + q) % N == 0`: a scalar
/// prologue walks to that boundary and a scalar tail finishes. The lanes are
/// consumed in ascending order with the region's own operand order, so the
/// chain is the reference chain bit for bit — only the load width changes.
#[cube]
#[allow(clippy::too_many_arguments)]
fn symv_vec_dot<F: Float + CubeElement, N: Size>(
    acc_in: F,
    m_v: &Array<Vector<F, N>>,
    m_s: &Array<F>,
    x: &Array<F>,
    base: u32,
    off: u32,
    lo: u32,
    hi: u32,
    alpha: F,
    #[comptime] x_lhs: bool,
    #[comptime] scale: bool,
) -> F {
    let nl = N::value() as u32;
    let mut acc = acc_in;
    let mut q = lo;
    while q < hi && (off + q) % nl != 0 {
        acc = symv_step::<F>(acc, m_s[(base + q) as usize], x[q as usize], alpha, x_lhs, scale);
        q += 1u32;
    }
    while q + nl <= hi {
        let v = m_v[((base + q) / nl) as usize];
        #[unroll]
        for l in 0..nl {
            acc = symv_step::<F>(acc, v[l as usize], x[(q + l) as usize], alpha, x_lhs, scale);
        }
        q += nl;
    }
    while q < hi {
        acc = symv_step::<F>(acc, m_s[(base + q) as usize], x[q as usize], alpha, x_lhs, scale);
        q += 1u32;
    }
    acc
}

/// K1colv: [`symv_col_kernel`] reading `M` through the lane-vector view (see
/// [`symv_vec_dot`]). Bit-identical to K1col by construction.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn symv_col_vec_kernel<F: Float + CubeElement, N: Size>(
    m_v: &Array<Vector<F, N>>,
    m_s: &Array<F>,
    x: &Array<F>,
    y: &mut Array<F>,
    lda: u32,
    off: u32,
    m: u32,
    alpha: F,
    #[comptime] upper: bool,
    #[comptime] alpha_one: bool,
) {
    if ABSOLUTE_POS < m as usize {
        let d = ABSOLUTE_POS as u32;
        let r = off + d;
        let base = r * lda + off;
        let mut acc_y = F::from_int(0);
        let mut acc_t = F::from_int(0);
        if upper {
            acc_t = symv_vec_dot::<F, N>(acc_t, m_v, m_s, x, base, off, 0, d, alpha, false, false);
            let a_rr = m_s[(base + d) as usize];
            if alpha_one {
                acc_y = (acc_y + x[d as usize] * a_rr) + acc_t;
                acc_y = symv_vec_dot::<F, N>(acc_y, m_v, m_s, x, base, off, d + 1, m, alpha, true, false);
            } else {
                acc_y = (acc_y + (alpha * x[d as usize]) * a_rr) + alpha * acc_t;
                acc_y = symv_vec_dot::<F, N>(acc_y, m_v, m_s, x, base, off, d + 1, m, alpha, true, true);
            }
            y[r as usize] = acc_y;
        } else {
            if alpha_one {
                acc_y = symv_vec_dot::<F, N>(acc_y, m_v, m_s, x, base, off, 0, d + 1, alpha, true, false);
            } else {
                acc_y = symv_vec_dot::<F, N>(acc_y, m_v, m_s, x, base, off, 0, d + 1, alpha, true, true);
            }
            acc_t = symv_vec_dot::<F, N>(acc_t, m_v, m_s, x, base, off, d + 1, m, alpha, false, false);
            if alpha_one {
                y[r as usize] = acc_y + acc_t;
            } else {
                y[r as usize] = acc_y + alpha * acc_t;
            }
        }
    }
}
