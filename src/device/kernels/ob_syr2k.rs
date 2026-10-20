//! **OpenBLAS track.** K2ob: `DSYR2K( UPLO, 'N', m, nb, alpha, V, lda, W,
//! lda, ONE, C, lda )` bit-exact to OpenBLAS 0.3.30 `dsyr2k_{U,L}N` (the host
//! transcription is [`crate::openblas::level3::dsyr2k_n`]), for one `k` block
//! (`nb < 2·GEMM_Q`) and `beta = 1`, on a fully symmetric device matrix.
//!
//! The driver packs row blocks of `C` (`GEMM_P` rule, multiples of
//! `GEMM_UNROLL_MN`) and calls `syr2k_kernel` twice (`V·Wᵀ`, then `W·Vᵀ`).
//! Every triangle element is written by exactly one micro-kernel call per
//! pass, `c = fma(acc, alpha, c)`, except the diagonal `32 × 32` pieces,
//! which the first pass fills as `c + (sub_ij + sub_ji)` from a zeroed
//! scratch (`sub = fma(acc, alpha, +0)`) and the second pass skips. `acc` is
//! `Σ_l X(i,l)·Y(j,l)` in ascending `l`, as one fused chain (`Plain`), two
//! lanes (`Pair`, row tile 4 × column tile 12) or four lanes (`Quad`, row
//! tile ≤ 2 × column tile 12) of `dgemm_kernel_16x2_skylakex.c`. The tile of
//! an element follows from its position inside its kernel call, which this
//! kernel recomputes from the same block rules.
//!
//! One unit per column `j` of the block walks its triangle rows and writes
//! both `C(i, j)` and the mirror `C(j, i)`; a unit only reads its own
//! column's triangle, and mirrors land in the other triangle of later
//! (lower) or earlier (upper) columns, so units never race.

use cubecl::prelude::*;

/// `GEMM_P` of the pinned library.
const GEMM_P: u32 = 192;
/// `GEMM_UNROLL_MN`.
const UNROLL_MN: u32 = 32;

/// Row-block size of `dsyr2k_{U,L}N` for `rem` rows left.
#[cube]
fn p_rule(rem: u32) -> u32 {
    let mut r = rem;
    if rem >= GEMM_P * 2u32 {
        r = GEMM_P;
    } else if rem > GEMM_P {
        r = ((rem / 2u32) + UNROLL_MN - 1u32) / UNROLL_MN * UNROLL_MN;
    }
    r
}

/// Accumulation mode of element `(r, c)` of an `m × n` kernel call:
/// 0 = plain chain, 1 = two lanes, 2 = four lanes.
#[cube]
pub fn acc_mode_pub(r: u32, m: u32, c: u32, n: u32) -> u32 {
    acc_mode(r, m, c, n)
}

#[cube]
fn acc_mode(r: u32, m: u32, c: u32, n: u32) -> u32 {
    let mut mode = 0u32;
    let mut q12 = 0u32;
    if n > 11u32 {
        q12 = (n - 12u32) / 12u32 + 1u32;
    }
    if c < 12u32 * q12 {
        let full16 = (m / 16u32) * 16u32;
        if r >= full16 {
            let rem = m - full16;
            let t = r - full16;
            let mut start = 0u32;
            let mut md = 0u32;
            let mut d = 8u32;
            while d > 0u32 {
                if (rem & d) != 0u32 {
                    if md == 0u32 && t < start + d {
                        md = d;
                    }
                    start += d;
                }
                d = d / 2u32;
            }
            if md == 4u32 {
                mode = 1u32;
            } else if md <= 2u32 {
                mode = 2u32;
            }
        }
    }
    mode
}

/// `acc` of `dgemm_kernel_16x2` over `l in 0..k` for `X(xi, l)·Y(yj, l)`,
/// reading column `l` of `X` at `vw[(xl + l)·lda + g0 + xi]`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn chain_acc<F: Float + CubeElement>(
    vw: &Array<F>,
    lda: u32,
    g0: u32,
    k: u32,
    xl: u32,
    xi: u32,
    yl: u32,
    yj: u32,
    mode: u32,
) -> F {
    let mut acc = F::from_int(0);
    let mut l = 0u32;
    if mode == 1u32 && k >= 2u32 {
        let mut lo = acc;
        let mut hi = F::from_int(0);
        while k - l >= 2u32 {
            lo = fma(vw[((xl + l) * lda + g0 + xi) as usize], vw[((yl + l) * lda + g0 + yj) as usize], lo);
            hi =
                fma(vw[((xl + l + 1u32) * lda + g0 + xi) as usize], vw[((yl + l + 1u32) * lda + g0 + yj) as usize], hi);
            l += 2u32;
        }
        acc = hi + lo;
    } else if mode == 2u32 && k >= 4u32 {
        let mut s0 = acc;
        let mut s1 = F::from_int(0);
        let mut s2 = F::from_int(0);
        let mut s3 = F::from_int(0);
        while k - l >= 4u32 {
            s0 = fma(vw[((xl + l) * lda + g0 + xi) as usize], vw[((yl + l) * lda + g0 + yj) as usize], s0);
            s1 =
                fma(vw[((xl + l + 1u32) * lda + g0 + xi) as usize], vw[((yl + l + 1u32) * lda + g0 + yj) as usize], s1);
            s2 =
                fma(vw[((xl + l + 2u32) * lda + g0 + xi) as usize], vw[((yl + l + 2u32) * lda + g0 + yj) as usize], s2);
            s3 =
                fma(vw[((xl + l + 3u32) * lda + g0 + xi) as usize], vw[((yl + l + 3u32) * lda + g0 + yj) as usize], s3);
            l += 4u32;
        }
        acc = (s3 + s2) + (s1 + s0);
    }
    while l < k {
        acc = fma(vw[((xl + l) * lda + g0 + xi) as usize], vw[((yl + l) * lda + g0 + yj) as usize], acc);
        l += 1u32;
    }
    acc
}

/// New value of the triangle element `(p, q)` (`p ≥ q` lower, `p ≤ q`
/// upper) of the block, from its old value `c0`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn tri_value<F: Float + CubeElement>(
    vw: &Array<F>,
    lda: u32,
    g0: u32,
    m: u32,
    nb: u32,
    alpha: F,
    p: u32,
    q: u32,
    c0: F,
    #[comptime] upper: bool,
) -> F {
    let zero = F::from_int(0);
    let mut s = 0u32;
    let mut e = p_rule(m);
    while e <= p {
        s = e;
        e = s + p_rule(m - s);
    }
    let mb = e - s;
    let i = p;
    let j = q;
    let mut diag = false;
    let mut rr = 0u32;
    let mut cm = 0u32;
    let mut rc = 0u32;
    let mut cn = 0u32;
    let mut dbase = 0u32;
    let mut dnn = 0u32;
    if upper {
        if j >= e {
            if s == 0u32 {
                let jjs = e + ((j - e) / UNROLL_MN) * UNROLL_MN;
                rr = i;
                cm = e;
                rc = j - jjs;
                cn = select(m - jjs < UNROLL_MN, m - jjs, UNROLL_MN);
            } else {
                rr = i - s;
                cm = mb;
                rc = j - e;
                cn = m - e;
            }
        } else {
            let lp = ((j - s) / UNROLL_MN) * UNROLL_MN;
            let nn = select(mb - lp < UNROLL_MN, mb - lp, UNROLL_MN);
            if i - s >= lp {
                diag = true;
                dbase = s + lp;
                dnn = nn;
            } else {
                rr = i - s;
                cm = lp;
                rc = j - (s + lp);
                cn = nn;
            }
        }
    } else if j < s {
        rr = i - s;
        cm = mb;
        rc = j;
        cn = s;
    } else {
        let lp = ((j - s) / UNROLL_MN) * UNROLL_MN;
        let nn = select(mb - lp < UNROLL_MN, mb - lp, UNROLL_MN);
        if i - s < lp + nn {
            diag = true;
            dbase = s + lp;
            dnn = nn;
        } else {
            rr = i - (s + lp + nn);
            cm = mb - lp - nn;
            rc = j - (s + lp);
            cn = nn;
        }
    }
    let mut c = c0;
    if diag {
        let ri = i - dbase;
        let rj = j - dbase;
        let a_ij = chain_acc::<F>(vw, lda, g0, nb, 0u32, i, nb, j, acc_mode(ri, dnn, rj, dnn));
        let a_ji = chain_acc::<F>(vw, lda, g0, nb, 0u32, j, nb, i, acc_mode(rj, dnn, ri, dnn));
        let sub_ij = fma(a_ij, alpha, zero);
        let sub_ji = fma(a_ji, alpha, zero);
        c = c + (sub_ij + sub_ji);
    } else {
        let mode = acc_mode(rr, cm, rc, cn);
        let acc1 = chain_acc::<F>(vw, lda, g0, nb, 0u32, i, nb, j, mode);
        c = fma(acc1, alpha, c);
        let acc2 = chain_acc::<F>(vw, lda, g0, nb, nb, i, 0u32, j, mode);
        c = fma(acc2, alpha, c);
    }
    c
}

/// K2ob, scalar units over a list of block elements `(elems[2u], elems[2u+1])
/// = (i, j)`: element `(i, j)` (either triangle) takes the new value of its
/// triangle element. Used for the elements whose accumulation is not the
/// plain chain; each unit writes only its own element.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_syr2k_list_kernel<F: Float + CubeElement>(
    m_buf: &mut Array<F>,
    vw: &Array<F>,
    elems: &Array<u32>,
    lda: u32,
    g0: u32,
    m: u32,
    nb: u32,
    count: u32,
    alpha: F,
    #[comptime] upper: bool,
) {
    if ABSOLUTE_POS < count as usize {
        let u = ABSOLUTE_POS as u32;
        let i = elems[(2u32 * u) as usize];
        let j = elems[(2u32 * u + 1u32) as usize];
        let mut p = select(i > j, i, j);
        let mut q = select(i > j, j, i);
        if upper {
            p = select(i > j, j, i);
            q = select(i > j, i, j);
        }
        let idx = ((j + g0) * lda + g0 + i) as usize;
        m_buf[idx] = tri_value::<F>(vw, lda, g0, m, nb, alpha, p, q, m_buf[idx], upper);
    }
}

/// K2ob, lane-vector units `(row lane-vector, block column j)`: for units
/// whose rows (inside the block) avoid every special interval of column `j`
/// (`ioff[j]..ioff[j+1]` pairs in `ivals`, block-relative), every lane is the
/// plain chain `c = fma(Σ X·Y, α, c)` twice; own lanes (`i ≥ j` lower,
/// `i ≤ j` upper) read `V(i,·)`/`W(i,·)` from the lanes and `W(j,·)`/`V(j,·)`
/// as broadcasts, mirror lanes the other way round (the triangle element's
/// operands), selected per lane. Special units and rows outside the block
/// keep their value.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_syr2k_vec_kernel<F: Float + CubeElement, N: Size>(
    c: &mut Array<Vector<F, N>>,
    vw: &Array<Vector<F, N>>,
    vw_s: &Array<F>,
    ioff: &Array<u32>,
    ivals: &Array<u32>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda: u32,
    g0: u32,
    m: u32,
    rv0: u32,
    units_x: u32,
    units: u32,
    nb: u32,
    alpha: F,
    #[comptime] upper: bool,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let rv = rv0 + u % units_x;
        let jb = u / units_x;
        let j = g0 + jb;
        let nl = N::value() as u32;
        let lda_vecs = lda / nl;
        // Rows of the unit inside the block, block-relative [r0, r1).
        let first = rv * nl;
        let r0 = select(first > g0, first - g0, 0u32);
        let lastp = first + nl;
        let hi = select(lastp < g0 + m, lastp, g0 + m);
        let r1 = select(hi > g0, hi - g0, 0u32);
        let mut special = false;
        for t in ioff[jb as usize]..ioff[(jb + 1u32) as usize] {
            let lo_t = ivals[(2u32 * t) as usize];
            let hi_t = ivals[(2u32 * t + 1u32) as usize];
            if lo_t < r1 && r0 < hi_t {
                special = true;
            }
        }
        if !special && r0 < r1 {
            let rows = Vector::<u32, N>::new(first) + lane_ofs[0];
            let jv = Vector::<u32, N>::new(j);
            let mut own = rows.greater_equal(jv);
            if upper {
                own = rows.less_equal(jv);
            }
            let idx = (j * lda_vecs + rv) as usize;
            let old = c[idx];
            let mut acc1 = Vector::<F, N>::new(F::from_int(0));
            let mut acc2 = Vector::<F, N>::new(F::from_int(0));
            for l in 0..nb {
                let v_i = vw[(l * lda_vecs + rv) as usize];
                let w_i = vw[((nb + l) * lda_vecs + rv) as usize];
                let v_j = Vector::<F, N>::new(vw_s[(l * lda + j) as usize]);
                let w_j = Vector::<F, N>::new(vw_s[((nb + l) * lda + j) as usize]);
                acc1 = fma(select_many(own, v_i, v_j), select_many(own, w_j, w_i), acc1);
                acc2 = fma(select_many(own, w_i, w_j), select_many(own, v_j, v_i), acc2);
            }
            let va = Vector::<F, N>::new(alpha);
            let mut val = fma(acc1, va, old);
            val = fma(acc2, va, val);
            let in_block =
                rows.greater_equal(Vector::<u32, N>::new(g0)).and(rows.less_than(Vector::<u32, N>::new(g0 + m)));
            c[idx] = select_many(in_block, val, old);
        }
    }
}

/// K2ob. `m_buf` is `lda × ncols` (fully symmetric), `vw` packs `nb` V
/// columns then `nb` W columns (`lda` rows aligned with `M`), block
/// `M[g0..g0+m, g0..g0+m]`. One unit per column `j < m`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_syr2k_col_kernel<F: Float + CubeElement>(
    m_buf: &mut Array<F>,
    vw: &Array<F>,
    lda: u32,
    g0: u32,
    m: u32,
    nb: u32,
    alpha: F,
    #[comptime] upper: bool,
) {
    if ABSOLUTE_POS < m as usize {
        let j = ABSOLUTE_POS as u32;
        let zero = F::from_int(0);
        // Row blocks [s, e) of the driver, walked in step with i.
        let mut s = 0u32;
        let mut e = p_rule(m);
        let mut i = 0u32;
        let mut i_end = j + 1u32;
        if !upper {
            i = j;
            i_end = m;
        }
        while e <= i {
            s = e;
            e = s + p_rule(m - s);
        }
        while i < i_end {
            if i >= e {
                s = e;
                e = s + p_rule(m - s);
            }
            let mb = e - s;
            let p = ((j + g0) * lda + g0 + i) as usize;
            let c0 = m_buf[p];
            // Kernel call of the element: (rel row, call m, rel col, call n),
            // or the diagonal scratch piece.
            let mut diag = false;
            let mut rr = 0u32;
            let mut cm = 0u32;
            let mut rc = 0u32;
            let mut cn = 0u32;
            let mut dbase = 0u32;
            let mut dnn = 0u32;
            if upper {
                if j >= e {
                    if s == 0u32 {
                        let jjs = e + ((j - e) / UNROLL_MN) * UNROLL_MN;
                        rr = i;
                        cm = e;
                        rc = j - jjs;
                        cn = select(m - jjs < UNROLL_MN, m - jjs, UNROLL_MN);
                    } else {
                        rr = i - s;
                        cm = mb;
                        rc = j - e;
                        cn = m - e;
                    }
                } else {
                    let lp = ((j - s) / UNROLL_MN) * UNROLL_MN;
                    let nn = select(mb - lp < UNROLL_MN, mb - lp, UNROLL_MN);
                    if i - s >= lp {
                        diag = true;
                        dbase = s + lp;
                        dnn = nn;
                    } else {
                        rr = i - s;
                        cm = lp;
                        rc = j - (s + lp);
                        cn = nn;
                    }
                }
            } else if j < s {
                rr = i - s;
                cm = mb;
                rc = j;
                cn = s;
            } else {
                let lp = ((j - s) / UNROLL_MN) * UNROLL_MN;
                let nn = select(mb - lp < UNROLL_MN, mb - lp, UNROLL_MN);
                if i - s < lp + nn {
                    diag = true;
                    dbase = s + lp;
                    dnn = nn;
                } else {
                    rr = i - (s + lp + nn);
                    cm = mb - lp - nn;
                    rc = j - (s + lp);
                    cn = nn;
                }
            }
            let mut c = c0;
            if diag {
                let ri = i - dbase;
                let rj = j - dbase;
                let a_ij = chain_acc::<F>(vw, lda, g0, nb, 0u32, i, nb, j, acc_mode(ri, dnn, rj, dnn));
                let a_ji = chain_acc::<F>(vw, lda, g0, nb, 0u32, j, nb, i, acc_mode(rj, dnn, ri, dnn));
                let sub_ij = fma(a_ij, alpha, zero);
                let sub_ji = fma(a_ji, alpha, zero);
                c = c + (sub_ij + sub_ji);
            } else {
                let mode = acc_mode(rr, cm, rc, cn);
                let acc1 = chain_acc::<F>(vw, lda, g0, nb, 0u32, i, nb, j, mode);
                c = fma(acc1, alpha, c);
                let acc2 = chain_acc::<F>(vw, lda, g0, nb, nb, i, 0u32, j, mode);
                c = fma(acc2, alpha, c);
            }
            m_buf[p] = c;
            m_buf[((i + g0) * lda + g0 + j) as usize] = c;
            i += 1u32;
        }
    }
}
