//! K3m: the fused divide-and-conquer merge (staged U), reproducing the
//! `DLAED2` gather plus `DLAED3`'s two `DGEMM`s in reference order.
//!
//! For a merge block of order `n` at global offset `g0`, output column `j`:
//! * `j < k`, top rows `r < n1`: `Σ_{l<n12} S_top(l, j) · Z[r, perm_top[l]]`;
//! * `j < k`, bottom rows: `Σ_{l<n23} S_bot(l, j) · Z[r, perm_bot[l]]`;
//! * `j ≥ k`: `Z[r, perm4[j − k]]` (the deflated columns, a gather).
//!
//! `Z[r, perm[l]]` is exactly `Q2(r, l)` of the reference (the column gather
//! is a copy), the accumulator starts from `0` as `DGEMM` with `BETA = 0`
//! does, and `l` ascends, so the sums are the reference's bit for bit. Units
//! are (row lane-vector, output column); the `j < k` / gather branch depends
//! only on `j`, so every lane of a unit takes the same path. Rows outside
//! `[lo, hi)` keep their previous `dst` value (the top and bottom row ranges
//! are two launches over the same aligned lane grid).
//!
//! `perm_top`, `perm_bot`, `perm4` hold GLOBAL column indices; `s` holds
//! `S(l, j)` at `j·lds + s_row0 + l` (the packed row block, or the whole
//! `K × K` U on the device with the block's first row as `s_row0`). `dst` is a row-chunk
//! scratch of `dst_rvecs` lane-vectors per output column starting at global
//! row lane-vector `dst_rv0` (the in-place merge computes `R` rows at a time
//! and copies them back, so no second `n²` buffer exists).

use cubecl::prelude::*;

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn merge_staged_kernel<F: Float + CubeElement, N: Size>(
    z: &Array<Vector<F, N>>,
    dst: &mut Array<Vector<F, N>>,
    perm: &Array<u32>,
    perm4: &Array<u32>,
    s: &Array<F>,
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
    j0: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let rv = rv0 + u % units_x;
        let j = j0 + u / units_x;
        let rows = Vector::<u32, N>::new(rv * N::value() as u32) + lane_ofs[0];
        let keep = rows.greater_equal(Vector::<u32, N>::new(lo)).and(rows.less_than(Vector::<u32, N>::new(hi)));
        let di = (j * dst_rvecs + rv - dst_rv0) as usize;
        let mut val = Vector::<F, N>::new(F::from_int(0));
        if j < k {
            for l in 0..nterms {
                let q2 = z[(perm[l as usize] * lda_vecs + rv) as usize];
                val = val + Vector::<F, N>::new(s[(j * lds + s_row0 + l) as usize]) * q2;
            }
        } else {
            val = z[(perm4[(j - k) as usize] * lda_vecs + rv) as usize];
        }
        dst[di] = select_many(keep, val, dst[di]);
    }
}

/// `DLASET( 'A', rows, K, 0, 0, Q(r0, 1) )` for the `N12 = 0` / `N23 = 0`
/// cases: output columns `j < k` of rows `[lo, hi)` are zero.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn merge_zero_kernel<F: Float + CubeElement, N: Size>(
    dst: &mut Array<Vector<F, N>>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda_vecs: u32,
    rv0: u32,
    units_x: u32,
    units: u32,
    lo: u32,
    hi: u32,
    dst_rvecs: u32,
    dst_rv0: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let rv = rv0 + u % units_x;
        let j = u / units_x;
        let rows = Vector::<u32, N>::new(rv * N::value() as u32) + lane_ofs[0];
        let keep = rows.greater_equal(Vector::<u32, N>::new(lo)).and(rows.less_than(Vector::<u32, N>::new(hi)));
        let di = (j * dst_rvecs + rv - dst_rv0) as usize;
        let _ = lda_vecs;
        dst[di] = select_many(keep, Vector::<F, N>::new(F::from_int(0)), dst[di]);
    }
}

/// K3m, lean U (`Exactness::Fast`): as [`merge_staged_kernel`] but the
/// secular eigenvector entry is evaluated per flop,
/// `U(l, j) = (ẑ[s] / ((Λ[s] − Λ[o_j]) − τ_j)) / ν_j` with `s = sec[l]`, so
/// no `k × k` matrix exists anywhere. Never bit-exact against the
/// reference (it reconstructs `DELTA`).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn merge_lean_kernel<F: Float + CubeElement, N: Size>(
    z: &Array<Vector<F, N>>,
    dst: &mut Array<Vector<F, N>>,
    perm: &Array<u32>,
    sec: &Array<u32>,
    perm4: &Array<u32>,
    zhat: &Array<F>,
    dlam: &Array<F>,
    orig: &Array<u32>,
    tau: &Array<F>,
    norm: &Array<F>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda_vecs: u32,
    rv0: u32,
    units_x: u32,
    units: u32,
    lo: u32,
    hi: u32,
    k: u32,
    nterms: u32,
    dst_rvecs: u32,
    dst_rv0: u32,
    j0: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let rv = rv0 + u % units_x;
        let j = j0 + u / units_x;
        let rows = Vector::<u32, N>::new(rv * N::value() as u32) + lane_ofs[0];
        let keep = rows.greater_equal(Vector::<u32, N>::new(lo)).and(rows.less_than(Vector::<u32, N>::new(hi)));
        let di = (j * dst_rvecs + rv - dst_rv0) as usize;
        let mut val = Vector::<F, N>::new(F::from_int(0));
        if j < k {
            let lo_j = dlam[orig[j as usize] as usize];
            let tj = tau[j as usize];
            let nj = norm[j as usize];
            for l in 0..nterms {
                let s = sec[l as usize] as usize;
                let ulj = (zhat[s] / ((dlam[s] - lo_j) - tj)) / nj;
                let q2 = z[(perm[l as usize] * lda_vecs + rv) as usize];
                val = val + Vector::<F, N>::new(ulj) * q2;
            }
        } else {
            val = z[(perm4[(j - k) as usize] * lda_vecs + rv) as usize];
        }
        dst[di] = select_many(keep, val, dst[di]);
    }
}
