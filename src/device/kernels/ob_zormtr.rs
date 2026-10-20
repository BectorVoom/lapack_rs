//! **OpenBLAS track.** The legacy unblocked `ZLARF('L')` of OpenBLAS 0.3.30's
//! `ZUNM2L`/`ZUNM2R` on the device (host transcription:
//! [`crate::openblas::zlevel2`]): `ZGEMV('C', lastv, lastc, ONE, C, LDC, V,
//! 1, ZERO, WORK, 1)` then `ZGERC(lastv, lastc, −TAU, V, 1, WORK, 1, C,
//! LDC)`, planar storage, in `zgemv_c_SKYLAKEX` / `zgerc_k` + `zaxpy_k`
//! order.

use cubecl::prelude::*;

/// `WORK(j) = ZGEMV('C')` column `j` (one unit per column `j < LASTC`,
/// `LASTC = lastc_buf[0]` from [`ob_zlastc_kernel`]) over
/// global rows `crow..crow+lastv`: `NBMAX = 1024` row blocks of the
/// `m − m%4` kernel rows, each a pair of even/odd lanes of four fused chains
/// (`[ar·xr, ai·xr, ar·xi, ai·xi]` from `+0`) combined conjugated and added
/// into `WORK` (from `+0`, `BETA = 0`), then the fused `m % 4` tail (`ALPHA =
/// ONE`).
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_zgemv_c_kernel<F: Float + CubeElement>(
    zr: &Array<F>,
    zi: &Array<F>,
    vr: &Array<F>,
    vi: &Array<F>,
    wr: &mut Array<F>,
    wi: &mut Array<F>,
    lastc_buf: &Array<u32>,
    lda: u32,
    crow: u32,
    lastv: u32,
    voff: u32,
) {
    if ABSOLUTE_POS < lastc_buf[0] as usize {
        let j = ABSOLUTE_POS as u32;
        let base = j * lda + crow;
        let zero = F::from_int(0);
        let neg = F::from_int(-1);
        let m3 = lastv % 4u32;
        let mk = lastv - m3;
        let mut yr = zero;
        let mut yi = zero;
        let mut s = 0u32;
        while s < mk {
            let nb = select(mk - s < 1024u32, mk - s, 1024u32);
            let mut e0 = zero;
            let mut e1 = zero;
            let mut e2 = zero;
            let mut e3 = zero;
            let mut o0 = zero;
            let mut o1 = zero;
            let mut o2 = zero;
            let mut o3 = zero;
            let mut q = 0u32;
            while q + 2u32 <= nb {
                let r = s + q;
                let ar = zr[(base + r) as usize];
                let ai = zi[(base + r) as usize];
                let xr = vr[(voff + r) as usize];
                let xi = vi[(voff + r) as usize];
                e0 = fma(ar, xr, e0);
                e1 = fma(ai, xr, e1);
                e2 = fma(ar, xi, e2);
                e3 = fma(ai, xi, e3);
                let ar1 = zr[(base + r + 1u32) as usize];
                let ai1 = zi[(base + r + 1u32) as usize];
                let xr1 = vr[(voff + r + 1u32) as usize];
                let xi1 = vi[(voff + r + 1u32) as usize];
                o0 = fma(ar1, xr1, o0);
                o1 = fma(ai1, xr1, o1);
                o2 = fma(ar1, xi1, o2);
                o3 = fma(ai1, xi1, o3);
                q += 2u32;
            }
            let tr = (e3 + e0) + (o3 + o0);
            let ti = (e2 - e1) + (o2 - o1);
            // alpha = (1, 0): or = tr*1 - ti*0, oi = ti*1 + tr*0.
            let one = F::from_int(1);
            let orr = (tr * one) - (ti * zero);
            let oii = (ti * one) + (tr * zero);
            yr = orr + yr;
            yi = oii + yi;
            s += nb;
        }
        if m3 != 0u32 {
            let mut tr = zero;
            let mut ti = zero;
            for r in mk..lastv {
                let ar = zr[(base + r) as usize];
                let ai = zi[(base + r) as usize];
                let xr = vr[(voff + r) as usize];
                let xi = vi[(voff + r) as usize];
                let pr = fma(ar, xr, ai * xi);
                let pi = fma(ar, xi, (ai * xr) * neg);
                if r == mk {
                    tr = pr;
                    ti = pi;
                } else {
                    tr = tr + pr;
                    ti = ti + pi;
                }
            }
            let one = F::from_int(1);
            yr = yr + fma(one, tr, (zero * ti) * neg);
            yi = yi + fma(one, ti, zero * tr);
        }
        wr[j as usize] = yr;
        wi[j as usize] = yi;
    }
}

/// `C(crow + r, j) += conj-weighted zaxpy` of `ZGERC(lastv, lastc, (ar, ai),
/// V, 1, WORK, 1, C)`: per column `da = (fma(yr, ar, yi·ai), fma(−yi, ar,
/// yr·ai))`, then `zaxpy_k` down the column — the AVX-512 kernel form for the
/// first `lastv & !15` rows, the scalar tail form after. Lane-vector units
/// `(rv, j)` over global rows; rows outside `[crow, crow + lastv)` keep their
/// value. The two forms are selected per lane.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_zgerc_vec_kernel<F: Float + CubeElement, N: Size>(
    cr: &mut Array<Vector<F, N>>,
    ci: &mut Array<Vector<F, N>>,
    xr_v: &Array<Vector<F, N>>,
    xi_v: &Array<Vector<F, N>>,
    wr: &Array<F>,
    wi: &Array<F>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda_vecs: u32,
    rv0: u32,
    units_x: u32,
    units: u32,
    crow: u32,
    lastv: u32,
    xbase_vecs: u32,
    alpha_r: F,
    alpha_i: F,
    lastc_buf: &Array<u32>,
) {
    if ABSOLUTE_POS < units as usize && (ABSOLUTE_POS as u32) / units_x < lastc_buf[0] {
        let u = ABSOLUTE_POS as u32;
        let rv = rv0 + u % units_x;
        let j = u / units_x;
        let nl = N::value() as u32;
        let rows = Vector::<u32, N>::new(rv * nl) + lane_ofs[0];
        let keep =
            rows.greater_equal(Vector::<u32, N>::new(crow)).and(rows.less_than(Vector::<u32, N>::new(crow + lastv)));
        let kern = rows.less_than(Vector::<u32, N>::new(crow + (lastv & !15u32)));
        let yr = wr[j as usize];
        let yi = wi[j as usize];
        let neg1 = F::from_int(-1);
        let da_r = fma(yr, alpha_r, yi * alpha_i);
        let da_i = fma(yi * neg1, alpha_r, yr * alpha_i);
        let idx = (j * lda_vecs + rv) as usize;
        let xidx = (rv - xbase_vecs) as usize;
        let c_r = cr[idx];
        let c_i = ci[idx];
        let x_r = xr_v[xidx];
        let x_i = xi_v[xidx];
        let vdr = Vector::<F, N>::new(da_r);
        // Kernel region.
        let mi = Vector::<F, N>::new(da_i * neg1);
        let pi = Vector::<F, N>::new(da_i * F::from_int(1));
        let k_yr = fma(vdr, x_r, c_r);
        let k_yi = fma(vdr, x_i, c_i);
        let k_r = fma(x_i, mi, k_yr);
        let k_i = fma(x_r, pi, k_yi);
        // Scalar tail.
        let vdi = Vector::<F, N>::new(da_i);
        let vneg = Vector::<F, N>::new(neg1);
        let t_r = fma(vdr, x_r, (vdi * x_i) * vneg) + c_r;
        let t_i = fma(vdr, x_i, vdi * x_r) + c_i;
        let n_r = select_many(kern, k_r, t_r);
        let n_i = select_many(kern, k_i, t_i);
        cr[idx] = select_many(keep, n_r, c_r);
        ci[idx] = select_many(keep, n_i, c_i);
    }
}

/// `ILAZLC( LASTV, N, C, LDC )` on the device, the reference loop exactly
/// (corners first, then columns from `N` down, rows ascending, `≠ 0` on either
/// part), into `out[0]`. One unit; the counters ascend (`c = N − 1 − t`)
/// because cubecl-cpu 0.10 cannot lower a decremented `while` counter.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_zlastc_kernel<F: Float + CubeElement>(
    zr: &Array<F>,
    zi: &Array<F>,
    out: &mut Array<u32>,
    lda: u32,
    crow: u32,
    lastv: u32,
    n: u32,
) {
    if ABSOLUTE_POS == 0 {
        let zero = F::from_int(0);
        let top = ((n - 1u32) * lda + crow) as usize;
        let bot = ((n - 1u32) * lda + crow + lastv - 1u32) as usize;
        let mut lastc = 0u32;
        if zr[top] != zero || zi[top] != zero || zr[bot] != zero || zi[bot] != zero {
            lastc = n;
        } else {
            let mut found = false;
            let mut t = 0u32;
            while t < n && !found {
                let c = n - 1u32 - t;
                let base = c * lda + crow;
                let mut r = 0u32;
                while r < lastv && !found {
                    if zr[(base + r) as usize] != zero || zi[(base + r) as usize] != zero {
                        found = true;
                        lastc = c + 1u32;
                    }
                    r += 1u32;
                }
                t += 1u32;
            }
        }
        out[0] = lastc;
    }
}
