//! Complex (planar) kernels for `zheevd_device`: K1c `ZHEMV`, K2c `ZHER2K`
//! and the `ZLARFB('L','N',*,'C')` steps.
//!
//! Real and imaginary parts live in two buffers with identical layout, so
//! every load stays stride-1 across lanes in each (100 % coalesced, the same
//! sector score as the real kernels). Every complex product is gfortran's full
//! formula `(ar·br − ai·bi, ar·bi + ai·br)`, including products with `CONE`
//! and with real-valued operands, because those change signed zeros; `conj`
//! is an exact negation of the imaginary part.
//!
//! Hermitian storage: the device matrix holds both triangles with
//! `M[r,c] = conj(M[c,r])` bitwise and an exactly `+0.0` imaginary diagonal,
//! so a row read `M[r,p]` with `p > r` is the reference's
//! `DCONJG(A(p,r))`.

use cubecl::prelude::*;

/// K1c: `y(off..off+m) = alpha · M[off.., off..] · x` (`BETA = 0`) in
/// reference `ZHEMV` order (see the real K1 for the lane roles).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hemv_kernel<F: Float + CubeElement, N: Size>(
    m_re: &Array<Vector<F, N>>,
    m_im: &Array<Vector<F, N>>,
    x_re: &Array<F>,
    x_im: &Array<F>,
    lane_ofs: &Array<Vector<u32, N>>,
    y_re: &mut Array<Vector<F, N>>,
    y_im: &mut Array<Vector<F, N>>,
    lda_vecs: u32,
    off: u32,
    m: u32,
    rv0: u32,
    units: u32,
    al_re: F,
    al_im: F,
    #[comptime] upper: bool,
) {
    if ABSOLUTE_POS < units as usize {
        let rv = rv0 + ABSOLUTE_POS as u32;
        let rows = Vector::<u32, N>::new(rv * N::value() as u32) + lane_ofs[0];
        let zero = Vector::<F, N>::new(F::from_int(0));
        let mut ay_r = zero;
        let mut ay_i = zero;
        let mut at_r = zero;
        let mut at_i = zero;
        let var = Vector::<F, N>::new(al_re);
        let vai = Vector::<F, N>::new(al_im);
        for q in 0..m {
            let p = off + q;
            let cr = m_re[(p * lda_vecs + rv) as usize];
            let ci = m_im[(p * lda_vecs + rv) as usize];
            let xr = x_re[q as usize];
            let xi = x_im[q as usize];
            let vxr = Vector::<F, N>::new(xr);
            let vxi = Vector::<F, N>::new(xi);
            // temp1 = alpha * x(p)
            let t1r = Vector::<F, N>::new(al_re * xr - al_im * xi);
            let t1i = Vector::<F, N>::new(al_re * xi + al_im * xr);
            // y += temp1 * col ; temp2 += conj(A(p,r)) * x(p) = M[r,p] * x(p)
            let yr = ay_r + (t1r * cr - t1i * ci);
            let yi = ay_i + (t1r * ci + t1i * cr);
            let tr = at_r + (cr * vxr - ci * vxi);
            let ti = at_i + (cr * vxi + ci * vxr);
            let pv = Vector::<u32, N>::new(p);
            if upper {
                let lt = pv.less_than(rows);
                let eq = pv.equal(rows);
                let gt = pv.greater_than(rows);
                at_r = select_many(lt, tr, at_r);
                at_i = select_many(lt, ti, at_i);
                // (y + temp1*DBLE(A(r,r))) + alpha*temp2
                let dr = yr + (var * at_r - vai * at_i);
                let di = yi + (var * at_i + vai * at_r);
                ay_r = select_many(eq, dr, select_many(gt, yr, ay_r));
                ay_i = select_many(eq, di, select_many(gt, yi, ay_i));
            } else {
                let le = pv.less_equal(rows);
                let gt = pv.greater_than(rows);
                ay_r = select_many(le, yr, ay_r);
                ay_i = select_many(le, yi, ay_i);
                at_r = select_many(gt, tr, at_r);
                at_i = select_many(gt, ti, at_i);
            }
        }
        if upper {
            y_re[rv as usize] = ay_r;
            y_im[rv as usize] = ay_i;
        } else {
            y_re[rv as usize] = ay_r + (var * at_r - vai * at_i);
            y_im[rv as usize] = ay_i + (var * at_i + vai * at_r);
        }
    }
}

/// K1c-col: [`hemv_kernel`] with scalar units, one per row `r`, reading row
/// `r` as column `r`: `M[r,p] = conj(M[p,r])` bitwise, so the imaginary part
/// is negated exactly, except on the diagonal whose stored `+0.0` imaginary
/// part is used as is (the reference multiplies by `DBLE(A(r,r))`).
/// Consecutive steps load adjacent elements; each row's chain is K1c's.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn hemv_col_kernel<F: Float + CubeElement>(
    m_re: &Array<F>,
    m_im: &Array<F>,
    x_re: &Array<F>,
    x_im: &Array<F>,
    y_re: &mut Array<F>,
    y_im: &mut Array<F>,
    lda: u32,
    off: u32,
    m: u32,
    al_re: F,
    al_im: F,
    #[comptime] upper: bool,
) {
    if ABSOLUTE_POS < m as usize {
        let d = ABSOLUTE_POS as u32;
        let r = off + d;
        let base = r * lda + off;
        let zero = F::from_int(0);
        let mut ay_r = zero;
        let mut ay_i = zero;
        let mut at_r = zero;
        let mut at_i = zero;
        if upper {
            for q in 0..d {
                let cr = m_re[(base + q) as usize];
                let ci = -m_im[(base + q) as usize];
                let xr = x_re[q as usize];
                let xi = x_im[q as usize];
                at_r = at_r + (cr * xr - ci * xi);
                at_i = at_i + (cr * xi + ci * xr);
            }
            let cr = m_re[(base + d) as usize];
            let ci = m_im[(base + d) as usize];
            let xr = x_re[d as usize];
            let xi = x_im[d as usize];
            let t1r = al_re * xr - al_im * xi;
            let t1i = al_re * xi + al_im * xr;
            let yr = ay_r + (t1r * cr - t1i * ci);
            let yi = ay_i + (t1r * ci + t1i * cr);
            ay_r = yr + (al_re * at_r - al_im * at_i);
            ay_i = yi + (al_re * at_i + al_im * at_r);
            for q in d + 1..m {
                let cr = m_re[(base + q) as usize];
                let ci = -m_im[(base + q) as usize];
                let xr = x_re[q as usize];
                let xi = x_im[q as usize];
                let t1r = al_re * xr - al_im * xi;
                let t1i = al_re * xi + al_im * xr;
                ay_r = ay_r + (t1r * cr - t1i * ci);
                ay_i = ay_i + (t1r * ci + t1i * cr);
            }
            y_re[r as usize] = ay_r;
            y_im[r as usize] = ay_i;
        } else {
            for q in 0..d + 1 {
                let cr = m_re[(base + q) as usize];
                let mut ci = -m_im[(base + q) as usize];
                if q == d {
                    ci = m_im[(base + q) as usize];
                }
                let xr = x_re[q as usize];
                let xi = x_im[q as usize];
                let t1r = al_re * xr - al_im * xi;
                let t1i = al_re * xi + al_im * xr;
                ay_r = ay_r + (t1r * cr - t1i * ci);
                ay_i = ay_i + (t1r * ci + t1i * cr);
            }
            for q in d + 1..m {
                let cr = m_re[(base + q) as usize];
                let ci = -m_im[(base + q) as usize];
                let xr = x_re[q as usize];
                let xi = x_im[q as usize];
                at_r = at_r + (cr * xr - ci * xi);
                at_i = at_i + (cr * xi + ci * xr);
            }
            y_re[r as usize] = ay_r + (al_re * at_r - al_im * at_i);
            y_im[r as usize] = ay_i + (al_re * at_i + al_im * at_r);
        }
    }
}

/// K2c: `ZHER2K(UPLO, 'N', M, NB, ALPHA, V, ·, W, ·, ONE, C, ·)` on both
/// triangles. Lanes of row `i` in column `j` play three roles: own
/// (`i > j` lower / `i < j` upper) evaluates the reference element
/// `(C + A(i,l)·T1) + B(i,l)·T2` with the column-uniform skip on
/// `A(j,l), B(j,l)`; diagonal evaluates `DBLE(C) + DBLE(A(j,l)·T1 + B(j,l)·T2)`
/// (imaginary part `+0.0`); mirror stores the conjugate of the reference
/// element `(j, i)`, with that column's temporaries and skip.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn her2k_kernel<F: Float + CubeElement, N: Size>(
    c_re: &mut Array<Vector<F, N>>,
    c_im: &mut Array<Vector<F, N>>,
    vw_re: &Array<Vector<F, N>>,
    vw_im: &Array<Vector<F, N>>,
    vw_s_re: &Array<F>,
    vw_s_im: &Array<F>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda: u32,
    off: u32,
    m: u32,
    rv0: u32,
    units_x: u32,
    units: u32,
    al_re: F,
    al_im: F,
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
        let old_r = c_re[idx];
        let old_i = c_im[idx];
        let zero = Vector::<F, N>::new(F::from_int(0));
        let var = Vector::<F, N>::new(al_re);
        let vai = Vector::<F, N>::new(al_im);
        let diag = rows.equal(jv);
        let mut own = rows.greater_than(jv);
        if upper {
            own = rows.less_than(jv);
        }
        // DBLE( C(J,J) ) for BETA = ONE.
        let mut acc_r = old_r;
        let mut acc_i = select_many(diag, zero, old_i);
        // A unit whose rows lie entirely on one side of the diagonal
        // evaluates only that side's expression (the uniform mask would
        // select it anyway); the mixed units keep the masked form.
        let first = rv * nlanes;
        let last = first + nlanes - 1;
        let mut all_own = first > j;
        let mut all_mir = last < j;
        if upper {
            all_own = last < j;
            all_mir = first > j;
        }
        if all_own {
            #[unroll]
            for l in 0..nb {
                let air = vw_re[(l * lda_vecs + rv) as usize];
                let aii = vw_im[(l * lda_vecs + rv) as usize];
                let bir = vw_re[((nb + l) * lda_vecs + rv) as usize];
                let bii = vw_im[((nb + l) * lda_vecs + rv) as usize];
                let ajr = vw_s_re[(l * lda + j) as usize];
                let aji = vw_s_im[(l * lda + j) as usize];
                let bjr = vw_s_re[((nb + l) * lda + j) as usize];
                let bji = vw_s_im[((nb + l) * lda + j) as usize];
                // Column-uniform reference zero-skip on A(j,l), B(j,l).
                if ajr != F::from_int(0) || aji != F::from_int(0) || bjr != F::from_int(0) || bji != F::from_int(0) {
                    let t1r = al_re * bjr - al_im * (-bji);
                    let t1i = al_re * (-bji) + al_im * bjr;
                    let t2r = al_re * ajr - al_im * aji;
                    let t2i = -(al_re * aji + al_im * ajr);
                    let (vt1r, vt1i, vt2r, vt2i) = (
                        Vector::<F, N>::new(t1r),
                        Vector::<F, N>::new(t1i),
                        Vector::<F, N>::new(t2r),
                        Vector::<F, N>::new(t2i),
                    );
                    let p1r = air * vt1r - aii * vt1i;
                    let p1i = air * vt1i + aii * vt1r;
                    let p2r = bir * vt2r - bii * vt2i;
                    let p2i = bir * vt2i + bii * vt2r;
                    acc_r = (acc_r + p1r) + p2r;
                    acc_i = (acc_i + p1i) + p2i;
                }
            }
        } else if all_mir {
            #[unroll]
            for l in 0..nb {
                let air = vw_re[(l * lda_vecs + rv) as usize];
                let aii = vw_im[(l * lda_vecs + rv) as usize];
                let bir = vw_re[((nb + l) * lda_vecs + rv) as usize];
                let bii = vw_im[((nb + l) * lda_vecs + rv) as usize];
                let ajr = vw_s_re[(l * lda + j) as usize];
                let aji = vw_s_im[(l * lda + j) as usize];
                let bjr = vw_s_re[((nb + l) * lda + j) as usize];
                let bji = vw_s_im[((nb + l) * lda + j) as usize];
                // Mirrored reference element (j, i); negations folded as above.
                let m1r = var * bir + vai * bii;
                let m1i = vai * bir - var * bii;
                let m2r = var * air - vai * aii;
                let m2s = var * aii + vai * air;
                let (vajr, vaji, vbjr, vbji) = (
                    Vector::<F, N>::new(ajr),
                    Vector::<F, N>::new(aji),
                    Vector::<F, N>::new(bjr),
                    Vector::<F, N>::new(bji),
                );
                let q1r = vajr * m1r - vaji * m1i;
                let q1i = vajr * m1i + vaji * m1r;
                let q2r = vbjr * m2r + vbji * m2s;
                let q2i = vbji * m2r - vbjr * m2s;
                let mir_run =
                    air.not_equal(zero).or(aii.not_equal(zero)).or(bir.not_equal(zero)).or(bii.not_equal(zero));
                let mir_r = (acc_r + q1r) + q2r;
                let mir_i = (acc_i - q1i) - q2i;
                acc_r = select_many(mir_run, mir_r, acc_r);
                acc_i = select_many(mir_run, mir_i, acc_i);
            }
        } else {
            #[unroll]
            for l in 0..nb {
                let air = vw_re[(l * lda_vecs + rv) as usize];
                let aii = vw_im[(l * lda_vecs + rv) as usize];
                let bir = vw_re[((nb + l) * lda_vecs + rv) as usize];
                let bii = vw_im[((nb + l) * lda_vecs + rv) as usize];
                let ajr = vw_s_re[(l * lda + j) as usize];
                let aji = vw_s_im[(l * lda + j) as usize];
                let bjr = vw_s_re[((nb + l) * lda + j) as usize];
                let bji = vw_s_im[((nb + l) * lda + j) as usize];
                // Column j temporaries: T1 = alpha*conj(B(j,l)), T2 = conj(alpha*A(j,l)).
                let t1r = al_re * bjr - al_im * (-bji);
                let t1i = al_re * (-bji) + al_im * bjr;
                let t2r = al_re * ajr - al_im * aji;
                let t2i = -(al_re * aji + al_im * ajr);
                let (vt1r, vt1i, vt2r, vt2i) = (
                    Vector::<F, N>::new(t1r),
                    Vector::<F, N>::new(t1i),
                    Vector::<F, N>::new(t2r),
                    Vector::<F, N>::new(t2i),
                );
                let p1r = air * vt1r - aii * vt1i;
                let p1i = air * vt1i + aii * vt1r;
                let p2r = bir * vt2r - bii * vt2i;
                let p2i = bir * vt2i + bii * vt2r;
                let own_run = Vector::<bool, N>::new(
                    ajr != F::from_int(0) || aji != F::from_int(0) || bjr != F::from_int(0) || bji != F::from_int(0),
                );
                let own_r = (acc_r + p1r) + p2r;
                let own_i = (acc_i + p1i) + p2i;
                let dia_r = acc_r + (p1r + p2r);
                // Row i temporaries for the mirrored reference element (j, i):
                // T1' = alpha*conj(B(i,l)), T2' = conj(alpha*A(i,l)). Vector negation
                // does not compile on every backend (HIP's C has no unary minus on
                // vector types), so the negations are folded with the exact IEEE
                // identities a − (−p) = a + p and (−x) + y = y − x.
                let m1r = var * bir + vai * bii; // alr*bir − ali*(−bii)
                let m1i = vai * bir - var * bii; // alr*(−bii) + ali*bir
                let m2r = var * air - vai * aii;
                let m2s = var * aii + vai * air; // T2'.im = −m2s
                let (vajr, vaji, vbjr, vbji) = (
                    Vector::<F, N>::new(ajr),
                    Vector::<F, N>::new(aji),
                    Vector::<F, N>::new(bjr),
                    Vector::<F, N>::new(bji),
                );
                let q1r = vajr * m1r - vaji * m1i;
                let q1i = vajr * m1i + vaji * m1r;
                let q2r = vbjr * m2r + vbji * m2s; // bjr*m2r − bji*(−m2s)
                let q2i = vbji * m2r - vbjr * m2s; // bjr*(−m2s) + bji*m2r
                let mir_run =
                    air.not_equal(zero).or(aii.not_equal(zero)).or(bir.not_equal(zero)).or(bii.not_equal(zero));
                let mir_r = (acc_r + q1r) + q2r;
                // conj of the reference sum: (−acc) + (−q1i) + (−q2i), stored negated.
                let mir_i = (acc_i - q1i) - q2i;
                let new_r = select_many(own, select_many(own_run, own_r, acc_r), select_many(mir_run, mir_r, acc_r));
                let new_i = select_many(own, select_many(own_run, own_i, acc_i), select_many(mir_run, mir_i, acc_i));
                acc_r = select_many(diag, select_many(own_run, dia_r, acc_r), new_r);
                acc_i = select_many(diag, zero, new_i);
            }
        }

        let lo = Vector::<u32, N>::new(off);
        let hi = Vector::<u32, N>::new(off + m);
        let in_block = rows.greater_equal(lo).and(rows.less_than(hi));
        c_re[idx] = select_many(in_block, acc_r, old_r);
        c_im[idx] = select_many(in_block, acc_i, old_i);
    }
}

/// Complex right-side `ZTRMM` chain of a `ZLARFB` block (row-major
/// `n × kpad` `W`): `acc = [flag_j] diag_j·W(i,j)`, then
/// `acc += coef(j,kk)·W(i,kk)` over the variant's range and direction with
/// the reference zero-skip. `coef`/`diag` already include `ALPHA` (and the
/// conjugation for `'C'`), computed on the host with the full formula.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ztrmm_chain_kernel<F: Float + CubeElement, N: Size>(
    w_in_re: &Array<Vector<F, N>>,
    w_in_im: &Array<Vector<F, N>>,
    w_in_s_re: &Array<F>,
    w_in_s_im: &Array<F>,
    w_out_re: &mut Array<Vector<F, N>>,
    w_out_im: &mut Array<Vector<F, N>>,
    coef_re: &Array<Vector<F, N>>,
    coef_im: &Array<Vector<F, N>>,
    diag: &Array<Vector<F, N>>,
    lane_ofs: &Array<Vector<u32, N>>,
    kpad: u32,
    k: u32,
    units: u32,
    #[comptime] above: bool,
    #[comptime] asc: bool,
) {
    if ABSOLUTE_POS < units as usize {
        let nl = N::value() as u32;
        let kpad_vecs = kpad / nl;
        let u = ABSOLUTE_POS as u32;
        let jv = u % kpad_vecs;
        let i = u / kpad_vecs;
        let js = Vector::<u32, N>::new(jv * nl) + lane_ofs[0];
        let zero = Vector::<F, N>::new(F::from_int(0));
        let wr = w_in_re[(i * kpad_vecs + jv) as usize];
        let wi = w_in_im[(i * kpad_vecs + jv) as usize];
        // diag layout: [re | im | flag], kpad each.
        let dr = diag[jv as usize];
        let di = diag[(kpad_vecs + jv) as usize];
        let flag = diag[(2 * kpad_vecs + jv) as usize].not_equal(zero);
        let mut acc_r = select_many(flag, dr * wr - di * wi, wr);
        let mut acc_i = select_many(flag, dr * wi + di * wr, wi);
        for t in 0..k {
            let mut kk = t;
            if !asc {
                kk = k - 1 - t;
            }
            let wkr = Vector::<F, N>::new(w_in_s_re[(i * kpad + kk) as usize]);
            let wki = Vector::<F, N>::new(w_in_s_im[(i * kpad + kk) as usize]);
            let cr = coef_re[(kk * kpad_vecs + jv) as usize];
            let ci = coef_im[(kk * kpad_vecs + jv) as usize];
            let kv = Vector::<u32, N>::new(kk);
            let mut in_range = kv.greater_than(js);
            if !above {
                in_range = kv.less_than(js);
            }
            let run = in_range.and(cr.not_equal(zero).or(ci.not_equal(zero)));
            acc_r = select_many(run, acc_r + (cr * wkr - ci * wki), acc_r);
            acc_i = select_many(run, acc_i + (cr * wki + ci * wkr), acc_i);
        }
        w_out_re[(i * kpad_vecs + jv) as usize] = acc_r;
        w_out_im[(i * kpad_vecs + jv) as usize] = acc_i;
    }
}

/// `ZGEMM('C','N', N, K, M−K, ALPHA, C₂, V₂, BETA, W)`:
/// `W_out(i,j) = ALPHA·temp + BETA·W_in(i,j)`,
/// `temp = Σ_l DCONJG(C[c_row0+l, i])·V_rm[vrow0+l, j]`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn zgemm_c_kernel<F: Float + CubeElement, N: Size>(
    c_re: &Array<F>,
    c_im: &Array<F>,
    v_re: &Array<Vector<F, N>>,
    v_im: &Array<Vector<F, N>>,
    w_in_re: &Array<Vector<F, N>>,
    w_in_im: &Array<Vector<F, N>>,
    w_out_re: &mut Array<Vector<F, N>>,
    w_out_im: &mut Array<Vector<F, N>>,
    lda: u32,
    kpad: u32,
    c_row0: u32,
    vrow0: u32,
    cnt: u32,
    units: u32,
    al_re: F,
    al_im: F,
    be_re: F,
    be_im: F,
) {
    if ABSOLUTE_POS < units as usize {
        let kpad_vecs = kpad / N::value() as u32;
        let u = ABSOLUTE_POS as u32;
        let jv = u % kpad_vecs;
        let i = u / kpad_vecs;
        let zero = Vector::<F, N>::new(F::from_int(0));
        let mut tr = zero;
        let mut ti = zero;
        for l in 0..cnt {
            let pr = Vector::<F, N>::new(c_re[(i * lda + c_row0 + l) as usize]);
            let pi = Vector::<F, N>::new(-c_im[(i * lda + c_row0 + l) as usize]);
            let xr = v_re[((vrow0 + l) * kpad_vecs + jv) as usize];
            let xi = v_im[((vrow0 + l) * kpad_vecs + jv) as usize];
            tr = tr + (pr * xr - pi * xi);
            ti = ti + (pr * xi + pi * xr);
        }
        let idx = (i * kpad_vecs + jv) as usize;
        let (var, vai) = (Vector::<F, N>::new(al_re), Vector::<F, N>::new(al_im));
        let (vbr, vbi) = (Vector::<F, N>::new(be_re), Vector::<F, N>::new(be_im));
        let wr = w_in_re[idx];
        let wi = w_in_im[idx];
        w_out_re[idx] = (var * tr - vai * ti) + (vbr * wr - vbi * wi);
        w_out_im[idx] = (var * ti + vai * tr) + (vbr * wi + vbi * wr);
    }
}

/// `ZGEMM('N','C', M−K, N, K, ALPHA, V₂, W, ONE, C₂)`:
/// `C[c_row0 − pad + r, i] += Σ_l (ALPHA·DCONJG(W(i,l)))·V[r + l·ldv]` for
/// `pad ≤ r < pad + cnt` (sector-aligned scalar units as the real K3).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn zgemm_n_kernel<F: Float + CubeElement>(
    c_re: &mut Array<F>,
    c_im: &mut Array<F>,
    v_re: &Array<F>,
    v_im: &Array<F>,
    w_re: &Array<F>,
    w_im: &Array<F>,
    lda: u32,
    ldv: u32,
    kpad: u32,
    c_row0: u32,
    pad: u32,
    cnt: u32,
    k: u32,
    units: u32,
    al_re: F,
    al_im: F,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let r = u % ldv;
        let i = u / ldv;
        if r >= pad && r < pad + cnt {
            let p = (i * lda + c_row0 - pad + r) as usize;
            let mut ar = c_re[p];
            let mut ai = c_im[p];
            for l in 0..k {
                let br = w_re[(i * kpad + l) as usize];
                let bi = -w_im[(i * kpad + l) as usize];
                let tr = al_re * br - al_im * bi;
                let ti = al_re * bi + al_im * br;
                let vr = v_re[(r + l * ldv) as usize];
                let vi = v_im[(r + l * ldv) as usize];
                ar = ar + (tr * vr - ti * vi);
                ai = ai + (tr * vi + ti * vr);
            }
            c_re[p] = ar;
            c_im[p] = ai;
        }
    }
}

/// `ZLARFB` step 1: `W(i,j) := DCONJG(C[c_row0 + j, i])` for `j < k`, zero
/// padding.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn zrows_to_w_kernel<F: Float + CubeElement>(
    c_re: &Array<F>,
    c_im: &Array<F>,
    w_re: &mut Array<F>,
    w_im: &mut Array<F>,
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
        let mut vr = F::from_int(0);
        let mut vi = F::from_int(0);
        if j < k {
            vr = c_re[(i * lda + c_row0 + j) as usize];
            vi = -c_im[(i * lda + c_row0 + j) as usize];
        }
        w_re[u as usize] = vr;
        w_im[u as usize] = vi;
    }
}

/// `ZLARFB` step 7: `C[c_row0 + j, i] := C[c_row0 + j, i] − DCONJG(W(i,j))`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn zsub_w_from_rows_kernel<F: Float + CubeElement>(
    c_re: &mut Array<F>,
    c_im: &mut Array<F>,
    w_re: &Array<F>,
    w_im: &Array<F>,
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
            c_re[p] = c_re[p] - w_re[u as usize];
            c_im[p] = c_im[p] - (-w_im[u as usize]);
        }
    }
}

/// `z[0..len] := 0`.
#[cube(launch, launch_unchecked)]
pub fn fill_zero_kernel<F: Float + CubeElement>(z: &mut Array<F>, len: u32) {
    if (ABSOLUTE_POS as u32) < len {
        z[ABSOLUTE_POS] = F::from_int(0);
    }
}
