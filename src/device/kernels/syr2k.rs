//! K2: rank-2k update `M -= V·Wᵀ + W·Vᵀ` on **both** triangles of the trailing
//! block, reproducing reference `DSYR2K('L'|'U', 'N', ..., ALPHA, V, W, 1, C)`.
//!
//! A unit is one lane-vector of `N` rows × one column `j` (2-D flattened).
//! Lower elements (`i ≥ j`) evaluate the reference expression
//! `(C + A(i,l)·(α·B(j,l))) + B(i,l)·(α·A(j,l))` with the column-uniform
//! zero-skip on `A(j,l), B(j,l)`; the mirrored upper element `(i, j)`, `i < j`,
//! evaluates the identical expression of `(j, i)`, so both triangles stay
//! bitwise equal. `V(i,l)`/`W(i,l)` are stride-1 across lanes; `V(j,l)`/
//! `W(j,l)` are broadcasts. Writes are element-disjoint and in place; lanes
//! outside the block keep their old value.
//!
//! `upper` only selects which reference triangle is "own" (`DSYR2K` with
//! `UPLO = 'U'` computes elements `i ≤ j`); the mirror rule is the same.

use cubecl::prelude::*;

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn syr2k_kernel<F: Float + CubeElement, N: Size>(
    c: &mut Array<Vector<F, N>>,
    vw: &Array<Vector<F, N>>,
    vw_s: &Array<F>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda: u32,
    off: u32,
    m: u32,
    rv0: u32,
    units_x: u32,
    units: u32,
    alpha: F,
    #[comptime] nb: u32,
    #[comptime] upper: bool,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let rv = rv0 + u % units_x;
        let j = off + u / units_x;
        let nlanes = N::value() as u32;
        let lda_vecs = lda / nlanes;
        let rows = Vector::<u32, N>::new(rv * nlanes) + lane_ofs[0];
        let jv = Vector::<u32, N>::new(j);
        let idx = (j * lda_vecs + rv) as usize;
        let old = c[idx];
        let mut acc = old;
        let zero = Vector::<F, N>::new(F::from_int(0));
        let va = Vector::<F, N>::new(alpha);
        // Own triangle of the reference call: i >= j (lower) or i <= j (upper).
        let mut own = rows.greater_equal(jv);
        if upper {
            own = rows.less_equal(jv);
        }
        // A unit whose rows lie entirely in one triangle evaluates only that
        // triangle's expression (the uniform mask would select it anyway).
        let first = rv * nlanes;
        let last = first + nlanes - 1;
        let mut all_own = first >= j;
        let mut all_mir = last < j;
        if upper {
            all_own = last <= j;
            all_mir = first > j;
        }
        if all_own {
            #[unroll]
            for l in 0..nb {
                let a_j = vw_s[(l * lda + j) as usize];
                let b_j = vw_s[((nb + l) * lda + j) as usize];
                // Column-uniform reference zero-skip on A(j,l), B(j,l).
                if a_j != F::from_int(0) || b_j != F::from_int(0) {
                    let a_i = vw[(l * lda_vecs + rv) as usize];
                    let b_i = vw[((nb + l) * lda_vecs + rv) as usize];
                    acc = (acc + a_i * Vector::<F, N>::new(alpha * b_j)) + b_i * Vector::<F, N>::new(alpha * a_j);
                }
            }
        } else if all_mir {
            #[unroll]
            for l in 0..nb {
                let a_i = vw[(l * lda_vecs + rv) as usize];
                let b_i = vw[((nb + l) * lda_vecs + rv) as usize];
                let a_j = vw_s[(l * lda + j) as usize];
                let b_j = vw_s[((nb + l) * lda + j) as usize];
                // Mirror element (i, j) = reference element (j, i): skip on A(i,l), B(i,l).
                let mir_new = (acc + Vector::<F, N>::new(a_j) * (va * b_i)) + Vector::<F, N>::new(b_j) * (va * a_i);
                let mir_run = a_i.not_equal(zero).or(b_i.not_equal(zero));
                acc = select_many(mir_run, mir_new, acc);
            }
        } else {
            #[unroll]
            for l in 0..nb {
                let a_i = vw[(l * lda_vecs + rv) as usize];
                let b_i = vw[((nb + l) * lda_vecs + rv) as usize];
                let a_j = vw_s[(l * lda + j) as usize];
                let b_j = vw_s[((nb + l) * lda + j) as usize];
                let va_j = Vector::<F, N>::new(a_j);
                let vb_j = Vector::<F, N>::new(b_j);
                let t1 = Vector::<F, N>::new(alpha * b_j);
                let t2 = Vector::<F, N>::new(alpha * a_j);
                let own_new = (acc + a_i * t1) + b_i * t2;
                let own_run = Vector::<bool, N>::new(a_j != F::from_int(0) || b_j != F::from_int(0));
                let m_t1 = va * b_i;
                let m_t2 = va * a_i;
                let mir_new = (acc + va_j * m_t1) + vb_j * m_t2;
                let mir_run = a_i.not_equal(zero).or(b_i.not_equal(zero));
                let own_acc = select_many(own_run, own_new, acc);
                let mir_acc = select_many(mir_run, mir_new, acc);
                acc = select_many(own, own_acc, mir_acc);
            }
        }
        let lo = Vector::<u32, N>::new(off);
        let hi = Vector::<u32, N>::new(off + m);
        let in_block = rows.greater_equal(lo).and(rows.less_than(hi));
        c[idx] = select_many(in_block, acc, old);
    }
}

/// K2r (GPU): [`syr2k_kernel`] with one unit owning `rows` adjacent columns
/// of one row lane-vector. The row loads `V(i,l)`, `W(i,l)` are made once per
/// `l` and reused by every column accumulator (register locals); each
/// accumulator evaluates exactly K2's per-element chain, so the result is
/// bit-identical to K2. Columns past the block end clamp to the first column
/// of the tile and are never stored.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn syr2k_rt_kernel<F: Float + CubeElement, N: Size>(
    c: &mut Array<Vector<F, N>>,
    vw: &Array<Vector<F, N>>,
    vw_s: &Array<F>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda: u32,
    off: u32,
    m: u32,
    rv0: u32,
    units_x: u32,
    units: u32,
    alpha: F,
    #[comptime] nb: u32,
    #[comptime] upper: bool,
    #[comptime] rows: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let rv = rv0 + u % units_x;
        let jb = off + (u / units_x) * rows;
        let nlanes = N::value() as u32;
        let lda_vecs = lda / nlanes;
        let row_ids = Vector::<u32, N>::new(rv * nlanes) + lane_ofs[0];
        let zero = Vector::<F, N>::new(F::from_int(0));
        let va = Vector::<F, N>::new(alpha);
        let mut cols = Array::<u32>::new(rows as usize);
        let mut acc = Array::<Vector<F, N>>::new(rows as usize);
        let mut own = Array::<Vector<bool, N>>::new(rows as usize);
        // Per column: 0 = every lane in the own triangle, 1 = every lane
        // mirrored, 2 = mixed (masks); a uniform class skips the other
        // expression.
        let mut class = Array::<u32>::new(rows as usize);
        let first = rv * nlanes;
        let last = first + nlanes - 1;
        #[unroll]
        for t in 0..rows {
            let j = select(jb + t < off + m, jb + t, jb);
            cols[t as usize] = j;
            acc[t as usize] = c[(j * lda_vecs + rv) as usize];
            let jv = Vector::<u32, N>::new(j);
            let mut o = row_ids.greater_equal(jv);
            let mut all_own = first >= j;
            let mut all_mir = last < j;
            if upper {
                o = row_ids.less_equal(jv);
                all_own = last <= j;
                all_mir = first > j;
            }
            own[t as usize] = o;
            class[t as usize] = select(all_own, 0u32, select(all_mir, 1u32, 2u32));
        }
        #[unroll]
        for l in 0..nb {
            let a_i = vw[(l * lda_vecs + rv) as usize];
            let b_i = vw[((nb + l) * lda_vecs + rv) as usize];
            let m_t1 = va * b_i;
            let m_t2 = va * a_i;
            let mir_run = a_i.not_equal(zero).or(b_i.not_equal(zero));
            #[unroll]
            for t in 0..rows {
                let j = cols[t as usize];
                let a_j = vw_s[(l * lda + j) as usize];
                let b_j = vw_s[((nb + l) * lda + j) as usize];
                let cur = acc[t as usize];
                let cl = class[t as usize];
                if cl == 0u32 {
                    if a_j != F::from_int(0) || b_j != F::from_int(0) {
                        acc[t as usize] =
                            (cur + a_i * Vector::<F, N>::new(alpha * b_j)) + b_i * Vector::<F, N>::new(alpha * a_j);
                    }
                } else if cl == 1u32 {
                    let mir_new = (cur + Vector::<F, N>::new(a_j) * m_t1) + Vector::<F, N>::new(b_j) * m_t2;
                    acc[t as usize] = select_many(mir_run, mir_new, cur);
                } else {
                    let t1 = Vector::<F, N>::new(alpha * b_j);
                    let t2 = Vector::<F, N>::new(alpha * a_j);
                    let own_new = (cur + a_i * t1) + b_i * t2;
                    let own_run = Vector::<bool, N>::new(a_j != F::from_int(0) || b_j != F::from_int(0));
                    let mir_new = (cur + Vector::<F, N>::new(a_j) * m_t1) + Vector::<F, N>::new(b_j) * m_t2;
                    let own_acc = select_many(own_run, own_new, cur);
                    let mir_acc = select_many(mir_run, mir_new, cur);
                    acc[t as usize] = select_many(own[t as usize], own_acc, mir_acc);
                }
            }
        }
        let lo = Vector::<u32, N>::new(off);
        let hi = Vector::<u32, N>::new(off + m);
        let in_block = row_ids.greater_equal(lo).and(row_ids.less_than(hi));
        #[unroll]
        for t in 0..rows {
            if jb + t < off + m {
                let idx = ((jb + t) * lda_vecs + rv) as usize;
                c[idx] = select_many(in_block, acc[t as usize], c[idx]);
            }
        }
    }
}
