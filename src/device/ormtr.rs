//! Hybrid `DORMTR('L', UPLO, 'N')`. Under `Exactness::Reference` `T` is built
//! on the host (`DLARFT`), and `DLARFB`'s own step sequence runs on the
//! device on `Z` in place — copy `C₁ᵀ`, `DTRMM(V₁)`, `DGEMM(C₂ᵀV₂)`,
//! `DTRMM(T)`, `C₂ −= V₂Wᵀ`, `DTRMM(V₁ᵀ)`, `C₁ −= Wᵀ` — each element-wise in
//! reference order (see `kernels::gemm`). Scratch: two row-major `n × kpad`
//! `W` buffers.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ops::{self, Chain, Lanes};
use super::workspace::{Counters, Span, upload_vec};
use crate::ptr::{D, ix};
use crate::sytrd::OrmEngine;

pub(crate) struct DeviceOrm<'a, R: Runtime> {
    pub client: &'a ComputeClient<R>,
    pub lanes: &'a Lanes,
    pub counters: &'a Counters,
    pub z: Handle,
    pub lda: usize,
    pub ncols: usize,
    /// `Exactness::Fast`: the fused `C −= V·((CᵀV)·Tᵀ)ᵀ` form.
    pub fast: bool,
}

impl<'a, R: Runtime> DeviceOrm<'a, R> {
    fn up(&self, v: Vec<f64>) -> Handle {
        self.counters.upload(v.len() * 8);
        upload_vec(self.client, v)
    }

    /// `Exactness::Fast` `DLARFB`: with the dense `V` (unit diagonal and
    /// structural zeros explicit), `W = CᵀV` (K3 with a zero input),
    /// `X = W·Tᵀ` (one triangular chain including the diagonal) and
    /// `C −= V·Xᵀ` — three launches instead of seven, reordered sums, so
    /// tolerance-gated only.
    unsafe fn larfb_fast(
        &mut self,
        forward: bool,
        m: usize,
        n: usize,
        k: usize,
        v: D,
        ldv: usize,
        t: D,
        ldt: usize,
        crow: usize,
    ) {
        let (client, lanes, lda, ncols) = (self.client, self.lanes, self.lda, self.ncols);
        let z = self.z.clone();
        let w = lanes.width;
        let kpad = k.div_ceil(w) * w;
        // Dense V(r, p): forward unit lower (diagonal r = p), backward unit
        // upper with the unit diagonal at r = m − k + p.
        let dense = |r: usize, p: usize| -> f64 {
            let dr = if forward { p } else { m - k + p };
            if r == dr {
                1.0
            } else if (forward && r < dr) || (!forward && r > dr) {
                0.0
            } else {
                v.at(ix(r + 1, p + 1, ldv))
            }
        };
        let pad = crow % w;
        let ldvp = (pad + m).div_ceil(w) * w;
        let mut v_rm = vec![0.0f64; m * kpad];
        let mut v_cm = vec![0.0f64; ldvp * k];
        for r in 0..m {
            for p in 0..k {
                let x = dense(r, p);
                v_rm[r * kpad + p] = x;
                v_cm[pad + r + p * ldvp] = x;
            }
        }
        // X = W·Tᵀ: X(i,p) = Σ_q W(i,q)·T(p,q), coef(p,q) = T(p,q) with the
        // diagonal T(p,p); q > p for the upper (forward) T, q < p for the
        // lower (backward) T.
        let mut coef = vec![0.0f64; kpad * kpad];
        let mut diag = vec![0.0f64; kpad];
        for p in 0..k {
            diag[p] = t.at(ix(p + 1, p + 1, ldt));
            for q in 0..k {
                coef[q * kpad + p] = t.at(ix(p + 1, q + 1, ldt));
            }
        }
        let (h_vrm, h_vcm, h_coef, h_diag) = (self.up(v_rm), self.up(v_cm), self.up(coef), self.up(diag));
        let zero = client.empty(n * kpad * 8);
        ops::fill_zero::<R, f64>(client, &zero, n * kpad);
        let wa = client.empty(n * kpad * 8);
        let wb = client.empty(n * kpad * 8);
        ops::gemm_t::<R, f64>(client, lanes, &z, lda, ncols, &h_vrm, m, &zero, &wa, n, kpad, crow, 0, m);
        ops::trmm_chain::<R, f64>(
            client,
            lanes,
            &wa,
            &wb,
            &h_coef,
            &h_diag,
            n,
            k,
            kpad,
            Chain { above: forward, asc: true, scale: true },
        );
        ops::gemm_n::<R, f64>(client, w, &z, lda, ncols, &h_vcm, ldvp, &wb, n, kpad, k, crow, m);
        self.counters.sample(client);
        for _ in 0..4 {
            self.counters.launch();
        }
    }
}

impl<'a, R: Runtime> OrmEngine for DeviceOrm<'a, R> {
    type B = crate::flavor::RefBlas;
    unsafe fn larfb(
        &mut self,
        forward: bool,
        m: usize,
        n: usize,
        k: usize,
        v: D,
        ldv: usize,
        t: D,
        ldt: usize,
        _c: D,
        crow: usize,
        _ldc: usize,
        _work: D,
        _ldwork: usize,
    ) {
        if m == 0 || n == 0 || k == 0 {
            return;
        }
        let span = self.counters.span_start();
        if self.fast {
            self.larfb_fast(forward, m, n, k, v, ldv, t, ldt, crow);
            self.counters.span_end(Span::OrmBlock, span);
            return;
        }
        let (client, lanes, z, lda, ncols) = (self.client, self.lanes, &self.z, self.lda, self.ncols);
        let w = lanes.width;
        let kpad = k.div_ceil(w) * w;
        let vv = |i: usize, j: usize| v.at(ix(i + 1, j + 1, ldv));
        let tt = |i: usize, j: usize| t.at(ix(i + 1, j + 1, ldt));
        // Square block of V: rows 0..k (forward, unit lower) or m-k..m (backward, unit upper).
        let sq0 = if forward { 0 } else { m - k };
        // Rectangular part: rows k..m (forward) or 0..m-k (backward).
        let rect0 = if forward { k } else { 0 };
        let nrect = m - k;
        let coef_of = |f: &dyn Fn(usize, usize) -> f64| {
            let mut c = vec![0.0f64; kpad * kpad];
            for kk in 0..k {
                for j in 0..k {
                    c[kk * kpad + j] = f(j, kk);
                }
            }
            c
        };
        let mut diag_t = vec![0.0f64; kpad];
        for j in 0..k {
            diag_t[j] = tt(j, j);
        }
        let zero_diag = self.up(vec![0.0f64; kpad]);
        let h_diag_t = self.up(diag_t);
        let (c2, c4, c6, ch2, ch4, ch6) = if forward {
            (
                coef_of(&|j, kk| vv(sq0 + kk, j)),
                coef_of(&|j, kk| tt(j, kk)),
                coef_of(&|j, kk| vv(sq0 + j, kk)),
                Chain { above: true, asc: true, scale: false },
                Chain { above: true, asc: true, scale: true },
                Chain { above: false, asc: false, scale: false },
            )
        } else {
            (
                coef_of(&|j, kk| vv(sq0 + kk, j)),
                coef_of(&|j, kk| tt(j, kk)),
                coef_of(&|j, kk| vv(sq0 + j, kk)),
                Chain { above: false, asc: true, scale: false },
                Chain { above: false, asc: false, scale: true },
                Chain { above: true, asc: true, scale: false },
            )
        };
        let (h2, h4, h6) = (self.up(c2), self.up(c4), self.up(c6));
        let wa = client.empty(n * kpad * 8);
        let wb = client.empty(n * kpad * 8);
        // 1. W = rows of C.
        ops::rows_to_w::<R, f64>(client, z, lda, ncols, &wa, n, kpad, k, crow + sq0);
        // 2. DTRMM with the square V block.
        ops::trmm_chain::<R, f64>(client, lanes, &wa, &wb, &h2, &zero_diag, n, k, kpad, ch2);
        let mut cur = wb;
        let mut spare = wa;
        let mut launches = 2;
        let mut h_vcm = None;
        if nrect > 0 {
            // 3. W = C_rectᵀ · V_rect + W.
            let pad = (crow + rect0) % w;
            let ldv = (pad + nrect).div_ceil(w) * w;
            let mut v_rm = vec![0.0f64; nrect * kpad];
            let mut v_cm = vec![0.0f64; ldv * k];
            for l in 0..nrect {
                for j in 0..k {
                    v_rm[l * kpad + j] = vv(rect0 + l, j);
                    v_cm[pad + l + j * ldv] = vv(rect0 + l, j);
                }
            }
            let h_vrm = self.up(v_rm);
            ops::gemm_t::<R, f64>(
                client,
                lanes,
                z,
                lda,
                ncols,
                &h_vrm,
                nrect,
                &cur,
                &spare,
                n,
                kpad,
                crow + rect0,
                0,
                nrect,
            );
            std::mem::swap(&mut cur, &mut spare);
            launches += 1;
            h_vcm = Some((self.up(v_cm), ldv));
        }
        // 4. DTRMM with T.
        ops::trmm_chain::<R, f64>(client, lanes, &cur, &spare, &h4, &h_diag_t, n, k, kpad, ch4);
        std::mem::swap(&mut cur, &mut spare);
        launches += 1;
        if let Some((h_vcm, ldv)) = &h_vcm {
            // 5. C_rect -= V_rect · Wᵀ.
            ops::gemm_n::<R, f64>(client, w, z, lda, ncols, h_vcm, *ldv, &cur, n, kpad, k, crow + rect0, nrect);
            launches += 1;
        }
        // 6. DTRMM with the square V block, transposed.
        ops::trmm_chain::<R, f64>(client, lanes, &cur, &spare, &h6, &zero_diag, n, k, kpad, ch6);
        std::mem::swap(&mut cur, &mut spare);
        // 7. C_sq -= Wᵀ.
        ops::sub_w_from_rows::<R, f64>(client, z, lda, ncols, &cur, n, kpad, k, crow + sq0);
        launches += 2;
        self.counters.sample(client);
        for _ in 0..launches {
            self.counters.launch();
        }
        self.counters.span_end(Span::OrmBlock, span);
    }
}

/// **OpenBLAS track.** `DLARFB('L','N',*,'C')` on the device with the
/// OpenBLAS-order kernels (`ops::ob_trmm_right`, `ops::ob_gemm_tn`,
/// `ops::ob_gemm_nt`) and the shared copies; `T` from the host `DLARFT`
/// (OpenBLAS flavour). Bit-identical to
/// `crate::sytrd::dlarfb_lnc::<OpenBlas>`.
pub(crate) struct DeviceOrmOb<'a, R: Runtime> {
    pub client: &'a ComputeClient<R>,
    pub lanes: &'a Lanes,
    pub counters: &'a Counters,
    pub z: Handle,
    pub lda: usize,
    pub ncols: usize,
}

impl<'a, R: Runtime> OrmEngine for DeviceOrmOb<'a, R> {
    type B = crate::openblas::OpenBlas;
    unsafe fn larfb(
        &mut self,
        forward: bool,
        m: usize,
        n: usize,
        k: usize,
        v: D,
        ldv: usize,
        t: D,
        ldt: usize,
        _c: D,
        crow: usize,
        _ldc: usize,
        _work: D,
        _ldwork: usize,
    ) {
        if m == 0 || n == 0 || k == 0 {
            return;
        }
        let span = self.counters.span_start();
        let (client, lanes, z, lda, ncols) = (self.client, self.lanes, &self.z, self.lda, self.ncols);
        let width = lanes.width;
        let kpad = k.div_ceil(width) * width;
        let up = |v: Vec<f64>| {
            self.counters.upload(v.len() * 8);
            upload_vec(client, v)
        };
        let sq0 = if forward { 0 } else { m - k };
        let rect0 = if forward { k } else { 0 };
        let nrect = m - k;
        let mut v1 = vec![0.0f64; k * k];
        let mut tk = vec![0.0f64; k * k];
        for c in 0..k {
            for r in 0..k {
                v1[r + c * k] = v.at(ix(sq0 + r + 1, c + 1, ldv));
                tk[r + c * k] = t.at(ix(r + 1, c + 1, ldt));
            }
        }
        let wa = client.empty(n * kpad * 8);
        let wb = client.empty(n * kpad * 8);
        let mut launches = 0;
        // 1. W = rows of C (square block).
        ops::rows_to_w::<R, f64>(client, z, lda, ncols, &wa, n, kpad, k, crow + sq0);
        // 2. DTRMM('R', L/U, 'N', 'U', V1).
        ops::ob_trmm_right::<R, f64>(client, &wa, &wb, &v1, n, k, kpad, !forward, false, false);
        let (mut cur, mut spare) = (wb, wa);
        launches += 2;
        let mut h_vrm = None;
        if nrect > 0 {
            let pad = (crow + rect0) % width;
            let ldvp = (pad + nrect).div_ceil(width) * width;
            let mut v_rm = vec![0.0f64; nrect * kpad];
            let mut v_cm = vec![0.0f64; ldvp * k];
            for l in 0..nrect {
                for j in 0..k {
                    let x = v.at(ix(rect0 + l + 1, j + 1, ldv));
                    v_rm[l * kpad + j] = x;
                    v_cm[pad + l + j * ldvp] = x;
                }
            }
            let h = up(v_rm);
            // 3. W = W + C_rectᵀ·V_rect.
            ops::ob_gemm_tn::<R, f64>(
                client,
                lanes,
                z,
                lda,
                ncols,
                &h,
                nrect,
                &cur,
                &spare,
                n,
                k,
                kpad,
                crow + rect0,
                0,
                nrect,
            );
            std::mem::swap(&mut cur, &mut spare);
            launches += 1;
            h_vrm = Some((h, up(v_cm), ldvp));
        }
        // 4. DTRMM('R', U/L, 'T', 'N', T).
        ops::ob_trmm_right::<R, f64>(client, &cur, &spare, &tk, n, k, kpad, forward, true, true);
        std::mem::swap(&mut cur, &mut spare);
        launches += 1;
        if let Some((h, h_cm, ldvp)) = &h_vrm {
            // 5. C_rect -= V_rect·Wᵀ.
            let fits = crow + rect0 - (crow + rect0) % width + ldvp <= lda;
            let cm = if fits { Some((h_cm, *ldvp)) } else { None };
            ops::ob_gemm_nt::<R, f64>(
                client,
                lanes,
                z,
                lda,
                ncols,
                h,
                nrect,
                cm,
                &cur,
                n,
                k,
                kpad,
                crow + rect0,
                0,
                nrect,
            );
            launches += 1;
        }
        // 6. DTRMM('R', L/U, 'T', 'U', V1).
        ops::ob_trmm_right::<R, f64>(client, &cur, &spare, &v1, n, k, kpad, !forward, true, false);
        std::mem::swap(&mut cur, &mut spare);
        // 7. C_sq -= Wᵀ.
        ops::sub_w_from_rows::<R, f64>(client, z, lda, ncols, &cur, n, kpad, k, crow + sq0);
        launches += 2;
        self.counters.sample(client);
        for _ in 0..launches {
            self.counters.launch();
        }
        self.counters.span_end(Span::OrmBlock, span);
    }
}
