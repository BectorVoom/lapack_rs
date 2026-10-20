//! K7: copy, scatter, gather, rotation and swap kernels. Every write is an
//! exact copy or a reference-ordered element chain; units write disjoint
//! elements. Offsets are explicit scalars (base-offset indexing, portable to
//! backends that do not honour offset bindings).

use cubecl::prelude::*;

/// `Z[g0.., g0..]` block of order `m` := identity; the rest of the columns
/// `g0..g0+m` (rows `0..nrows`) := 0. Unit = one element (flattened).
#[cube(launch, launch_unchecked)]
pub fn fill_identity_kernel<F: Float + CubeElement>(z: &mut Array<F>, lda: u32, nrows: u32, g0: u32, m: u32) {
    let u = ABSOLUTE_POS as u32;
    if u < nrows * m {
        let r = u % nrows;
        let c = u / nrows;
        let mut v = F::from_int(0);
        if r == g0 + c {
            v = F::from_int(1);
        }
        z[((g0 + c) * lda + r) as usize] = v;
    }
}

/// `Z[g0 + i, g0 + j] := src[i + j*m]` for an `m × m` host block.
#[cube(launch, launch_unchecked)]
pub fn scatter_block_kernel<F: Float + CubeElement>(z: &mut Array<F>, src: &Array<F>, lda: u32, g0: u32, m: u32) {
    let u = ABSOLUTE_POS as u32;
    if u < m * m {
        let i = u % m;
        let j = u / m;
        z[((g0 + j) * lda + g0 + i) as usize] = src[u as usize];
    }
}

/// `out[q] := Z[row, col0 + q]` for `q < len` (a strided row gather).
#[cube(launch, launch_unchecked)]
pub fn gather_row_kernel<F: Float + CubeElement>(
    z: &Array<F>,
    out: &mut Array<F>,
    lda: u32,
    row: u32,
    col0: u32,
    len: u32,
) {
    let q = ABSOLUTE_POS as u32;
    if q < len {
        out[q as usize] = z[((col0 + q) * lda + row) as usize];
    }
}

/// Apply `nrot` Givens rotations in list order to the rows `[lo, hi)` of
/// `Z` (reference `DROT`: `x' = c·x + s·y`, `y' = c·y − s·x`). Unit = one row
/// lane-vector; each lane is its own row, so rows are independent. Lanes
/// outside `[lo, hi)` keep their values (a rotated structural zero could
/// change sign).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn rotations_kernel<F: Float + CubeElement, N: Size>(
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
        for t in 0..nrot {
            let px = (cols[(2 * t) as usize] * lda_vecs + rv) as usize;
            let py = (cols[(2 * t + 1) as usize] * lda_vecs + rv) as usize;
            let c = Vector::<F, N>::new(cs[(2 * t) as usize]);
            let s = Vector::<F, N>::new(cs[(2 * t + 1) as usize]);
            let x = z[px];
            let y = z[py];
            let dtemp = c * x + s * y;
            z[py] = select_many(keep, c * y - s * x, y);
            z[px] = select_many(keep, dtemp, x);
        }
    }
}

/// `Z[r, c0 + j] := src[r, j]` for rows of the lane-vectors `rv0..` and
/// `j < ncols`, keeping `Z` outside rows `lo..hi`. `src` is a row-chunk
/// scratch (`src_rvecs` lane-vectors per column from row lane-vector
/// `src_rv0`).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn copy_cols_kernel<F: Float + CubeElement, N: Size>(
    src: &Array<Vector<F, N>>,
    z: &mut Array<Vector<F, N>>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda_vecs: u32,
    c0: u32,
    rv0: u32,
    units_x: u32,
    units: u32,
    lo: u32,
    hi: u32,
    src_rvecs: u32,
    src_rv0: u32,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let rv = rv0 + u % units_x;
        let j = u / units_x;
        let rows = Vector::<u32, N>::new(rv * N::value() as u32) + lane_ofs[0];
        let keep = rows.greater_equal(Vector::<u32, N>::new(lo)).and(rows.less_than(Vector::<u32, N>::new(hi)));
        let di = ((c0 + j) * lda_vecs + rv) as usize;
        z[di] = select_many(keep, src[(j * src_rvecs + rv - src_rv0) as usize], z[di]);
    }
}

/// Swap columns `ca` and `cb` of `Z` over all `lda` rows. Unit = lane-vector.
#[cube(launch, launch_unchecked)]
pub fn swap_cols_kernel<F: Float + CubeElement, N: Size>(z: &mut Array<Vector<F, N>>, lda_vecs: u32, ca: u32, cb: u32) {
    if ABSOLUTE_POS < lda_vecs as usize {
        let rv = ABSOLUTE_POS as u32;
        let pa = (ca * lda_vecs + rv) as usize;
        let pb = (cb * lda_vecs + rv) as usize;
        let t = z[pa];
        z[pa] = z[pb];
        z[pb] = t;
    }
}

/// `dlarfb` step 1: `W[i, j] := C[c_row0 + j, i]` for `j < k` (W row-major
/// `n × kpad`). Scalar units `(i, j)` flattened as `i*kpad + j`: adjacent
/// units read adjacent rows of column `i`.
#[cube(launch, launch_unchecked)]
pub fn rows_to_w_kernel<F: Float + CubeElement>(
    c: &Array<F>,
    w: &mut Array<F>,
    lda: u32,
    kpad: u32,
    k: u32,
    c_row0: u32,
    n: u32,
) {
    let u = ABSOLUTE_POS as u32;
    if u < n * kpad {
        let j = u % kpad;
        let i = u / kpad;
        let mut v = F::from_int(0);
        if j < k {
            v = c[(i * lda + c_row0 + j) as usize];
        }
        w[u as usize] = v;
    }
}

/// `dlarfb` step 7: `C[c_row0 + j, i] := C[c_row0 + j, i] - W[i, j]`
/// (forward and backward share it; the backward caller passes
/// `c_row0 = M - K`). Scalar units `(i, j)`.
#[cube(launch, launch_unchecked)]
pub fn sub_w_from_rows_kernel<F: Float + CubeElement>(
    c: &mut Array<F>,
    w: &Array<F>,
    lda: u32,
    kpad: u32,
    k: u32,
    c_row0: u32,
    n: u32,
) {
    let u = ABSOLUTE_POS as u32;
    if u < n * kpad {
        let j = u % kpad;
        let i = u / kpad;
        if j < k {
            let p = (i * lda + c_row0 + j) as usize;
            c[p] = c[p] - w[u as usize];
        }
    }
}

/// Stage an `n × n` column-major matrix held on the device (valid in its
/// `upper`/lower triangle) as the `lda × n` image with both triangles:
/// `dst[i + j·lda] = src[own(i, j)]`, padding rows zero. With `herm_im` the
/// source is an imaginary part: the mirrored triangle is negated and the
/// diagonal is `+0.0` (the staging of `zheevd_device`). Scalar units over
/// `dst`.
#[cube(launch, launch_unchecked)]
pub fn stage_sym_kernel<F: Float + CubeElement>(
    src: &Array<F>,
    dst: &mut Array<F>,
    n: u32,
    lda: u32,
    #[comptime] upper: bool,
    #[comptime] herm_im: bool,
) {
    let u = ABSOLUTE_POS as u32;
    if u < lda * n {
        let i = u % lda;
        let j = u / lda;
        let mut v = F::from_int(0);
        if i < n {
            let mut own = i >= j;
            if upper {
                own = i <= j;
            }
            if own {
                v = src[(i + j * n) as usize];
            } else {
                v = src[(j + i * n) as usize];
                if herm_im {
                    v = -v;
                }
            }
            if herm_im && i == j {
                v = F::from_int(0);
            }
        }
        dst[u as usize] = v;
    }
}

/// `dst[i + j·n] = src[i + j·lda]` for `i, j < n` (drop the padding rows).
#[cube(launch, launch_unchecked)]
pub fn compact_kernel<F: Float + CubeElement>(src: &Array<F>, dst: &mut Array<F>, n: u32, lda: u32) {
    let u = ABSOLUTE_POS as u32;
    if u < n * n {
        let i = u % n;
        let j = u / n;
        dst[u as usize] = src[(i + j * lda) as usize];
    }
}
