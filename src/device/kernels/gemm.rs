//! K3 (`Exactness::Reference`): the `DLARFB('L','N',*,'C')` sequence on the
//! device, each step element-wise in reference order.
//!
//! `W` is row-major `n × kpad` (row `i` = column `i` of `C`, lanes over the
//! block columns `j`), which makes every hot load stride-1 across lanes:
//!
//! * [`trmm_chain_kernel`] — the three right-side `DTRMM`s of a block. Each
//!   element `W(i,j)` is an out-of-place chain over `kk` (the in-place
//!   reference only ever reads not-yet-modified columns), `start` optionally
//!   scaled by the diagonal, then `+ coef(j,kk)·W(i,kk)` over the variant's
//!   range and direction with the reference zero-skip;
//! * [`gemm_t_kernel`] — `DGEMM('T','N', N, K, M−K, 1, C₂, V₂, 1, W)`:
//!   `W(i,j) = temp + W(i,j)`, `temp = Σ_l C(l,i)·V(l,j)`;
//! * [`gemm_n_kernel`] — `DGEMM('N','T', M−K, N, K, −1, V₂, W, 1, C₂)`:
//!   `C(r,i) = C(r,i) + (−W(i,l))·V(r,l)`, `l` ascending (scalar units: the
//!   row range is not lane-aligned).

use cubecl::prelude::*;

/// `coef` is column-major `kpad × kpad` holding `coef(j, kk)` at
/// `kk·kpad + j`; `diag` holds `coef(j, j)` (used when `scale`).
/// `above`: the range is `kk > j`, else `kk < j`; `asc`: `kk` ascending.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn trmm_chain_kernel<F: Float + CubeElement, N: Size>(
    w_in: &Array<Vector<F, N>>,
    w_in_s: &Array<F>,
    w_out: &mut Array<Vector<F, N>>,
    coef: &Array<Vector<F, N>>,
    diag: &Array<Vector<F, N>>,
    lane_ofs: &Array<Vector<u32, N>>,
    kpad: u32,
    k: u32,
    units: u32,
    #[comptime] above: bool,
    #[comptime] asc: bool,
    #[comptime] scale: bool,
) {
    if ABSOLUTE_POS < units as usize {
        let nl = N::value() as u32;
        let kpad_vecs = kpad / nl;
        let u = ABSOLUTE_POS as u32;
        let jv = u % kpad_vecs;
        let i = u / kpad_vecs;
        let js = Vector::<u32, N>::new(jv * nl) + lane_ofs[0];
        let zero = Vector::<F, N>::new(F::from_int(0));
        let mut acc = w_in[(i * kpad_vecs + jv) as usize];
        if scale {
            acc = diag[jv as usize] * acc;
        }
        for t in 0..k {
            let mut kk = t;
            if !asc {
                kk = k - 1 - t;
            }
            let wk = Vector::<F, N>::new(w_in_s[(i * kpad + kk) as usize]);
            let c = coef[(kk * kpad_vecs + jv) as usize];
            let kv = Vector::<u32, N>::new(kk);
            let mut in_range = kv.greater_than(js);
            if !above {
                in_range = kv.less_than(js);
            }
            let run = in_range.and(c.not_equal(zero));
            acc = select_many(run, acc + c * wk, acc);
        }
        w_out[(i * kpad_vecs + jv) as usize] = acc;
    }
}

/// `W_out(i,j) = (Σ_{l<cnt} C[c_row0 + l, i] · V_rm[vrow0 + l, j]) + W_in(i,j)`.
/// `v_rm` is row-major with `kpad` columns.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn gemm_t_kernel<F: Float + CubeElement, N: Size>(
    c: &Array<F>,
    v_rm: &Array<Vector<F, N>>,
    w_in: &Array<Vector<F, N>>,
    w_out: &mut Array<Vector<F, N>>,
    lda: u32,
    kpad: u32,
    c_row0: u32,
    vrow0: u32,
    cnt: u32,
    units: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let kpad_vecs = kpad / N::value() as u32;
        let u = ABSOLUTE_POS as u32;
        let jv = u % kpad_vecs;
        let i = u / kpad_vecs;
        let mut temp = Vector::<F, N>::new(F::from_int(0));
        for l in 0..cnt {
            let cli = Vector::<F, N>::new(c[(i * lda + c_row0 + l) as usize]);
            temp = temp + cli * v_rm[((vrow0 + l) * kpad_vecs + jv) as usize];
        }
        let idx = (i * kpad_vecs + jv) as usize;
        w_out[idx] = temp + w_in[idx];
    }
}

/// `C[c_row0 − pad + r, i] += Σ_{l<k} (−W(i,l)) · V[r + l·ldv]` for
/// `pad ≤ r < pad + cnt`. `c_row0 − pad` and `ldv` are multiples of the lane
/// width and `V` is uploaded with `pad` leading rows per column, so adjacent
/// units read and write adjacent, sector-aligned addresses. `v` is
/// column-major with leading dimension `ldv`; `w` row-major `kpad` (a
/// broadcast per unit).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn gemm_n_kernel<F: Float + CubeElement>(
    c: &mut Array<F>,
    v: &Array<F>,
    w: &Array<F>,
    lda: u32,
    ldv: u32,
    kpad: u32,
    c_row0: u32,
    pad: u32,
    cnt: u32,
    k: u32,
    units: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let r = u % ldv;
        let i = u / ldv;
        if r >= pad && r < pad + cnt {
            let p = (i * lda + c_row0 - pad + r) as usize;
            let mut acc = c[p];
            for l in 0..k {
                let temp = -w[(i * kpad + l) as usize];
                acc = acc + temp * v[(r + l * ldv) as usize];
            }
            c[p] = acc;
        }
    }
}

/// K3r (GPU): [`gemm_t_kernel`] with one unit owning `rows` adjacent rows
/// `i` of `W` at one lane-vector `jv`. The `V_rm` row load is made once per
/// `l` and reused by every accumulator; each accumulator is K3's chain, so
/// the result is bit-identical to K3. Rows past `nrows` clamp and are not
/// stored.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn gemm_t_rt_kernel<F: Float + CubeElement, N: Size>(
    c: &Array<F>,
    v_rm: &Array<Vector<F, N>>,
    w_in: &Array<Vector<F, N>>,
    w_out: &mut Array<Vector<F, N>>,
    lda: u32,
    kpad: u32,
    c_row0: u32,
    vrow0: u32,
    cnt: u32,
    nrows: u32,
    units: u32,
    #[comptime] rows: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let kpad_vecs = kpad / N::value() as u32;
        let u = ABSOLUTE_POS as u32;
        let jv = u % kpad_vecs;
        let ib = (u / kpad_vecs) * rows;
        let mut ids = Array::<u32>::new(rows as usize);
        let mut acc = Array::<Vector<F, N>>::new(rows as usize);
        #[unroll]
        for t in 0..rows {
            ids[t as usize] = select(ib + t < nrows, ib + t, ib);
            acc[t as usize] = Vector::<F, N>::new(F::from_int(0));
        }
        for l in 0..cnt {
            let vl = v_rm[((vrow0 + l) * kpad_vecs + jv) as usize];
            #[unroll]
            for t in 0..rows {
                let cli = Vector::<F, N>::new(c[(ids[t as usize] * lda + c_row0 + l) as usize]);
                acc[t as usize] = acc[t as usize] + cli * vl;
            }
        }
        #[unroll]
        for t in 0..rows {
            if ib + t < nrows {
                let idx = ((ib + t) * kpad_vecs + jv) as usize;
                w_out[idx] = acc[t as usize] + w_in[idx];
            }
        }
    }
}
