//! `zheevd_device`: the hybrid host/device `ZHEEVD` on planar storage.
//!
//! Same structure as [`super::syevd`]: `M` (real and imaginary parts, both
//! `lda × n`) holds A during `ZHETRD` (K1c/K2c), then Z. `ZSTEDC('I')` runs the
//! real divide and conquer in the real part (the imaginary part is zeroed,
//! `ZLACP2`), and `ZUNMTR` applies `ZLARFB`'s own sequence with the complex
//! K3c kernels. Bit-identical to [`crate::zheevd_alloc`] under
//! `Exactness::Reference` on a non-contracting backend.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ops::{self, Lanes, ZHandle};
use super::syevd::{DeviceWorkspace, read_columns, read_flat_chunked};
use super::workspace::{Counters, Phase, read_range, upload_vec};
use super::{geometry, probe, stedc};
use crate::aux::{PREC, SAFMIN};
use crate::hetrd::{ZOrmEngine, ZTrdEngine};
use crate::ptr::{V, ix};
use crate::stedc::SMLSIZ;
use crate::sytrd::{ORM_NB, SYTRD_NB, TSIZE};
use crate::zblas::{CONE, Z, ZV};
use crate::{Job, LapackError, Uplo};

/// Device state for one complex tridiagonalisation.
struct DeviceZTrd<'a, R: Runtime, B: super::flavor::DeviceBlas> {
    client: &'a ComputeClient<R>,
    lanes: &'a Lanes,
    counters: &'a Counters,
    m: ZHandle,
    lda: usize,
    n: usize,
    y: ZHandle,
    y_len: usize,
    buf_re: Vec<f64>,
    buf_im: Vec<f64>,
    flavor: std::marker::PhantomData<B>,
}

impl<'a, R: Runtime, B: super::flavor::DeviceBlas> DeviceZTrd<'a, R, B> {
    fn read_flat(&mut self, first: usize, len: usize) {
        if self.buf_re.len() < len {
            self.buf_re.resize(len, 0.0);
            self.buf_im.resize(len, 0.0);
        }
        read_range(self.client, &self.m.re, self.lda * self.n, first, len, &mut self.buf_re[..len]);
        read_range(self.client, &self.m.im, self.lda * self.n, first, len, &mut self.buf_im[..len]);
        self.counters.read(len * 8);
        self.counters.read(len * 8);
    }

    unsafe fn copy_tri(
        &self,
        upper: bool,
        c0: usize,
        c1: usize,
        r0: usize,
        r1: usize,
        first: usize,
        a: ZV,
        lda: usize,
    ) {
        for c in c0..=c1 {
            for r in r0..=r1 {
                if (upper && r <= c) || (!upper && r >= c) {
                    let q = (c - 1) * self.lda + (r - 1) - first;
                    a.set(ix(r, c, lda), Z::new(self.buf_re[q], self.buf_im[q]));
                }
            }
        }
    }
}

impl<'a, R: Runtime, B: super::flavor::DeviceBlas> ZTrdEngine for DeviceZTrd<'a, R, B> {
    type B = B;
    unsafe fn panel(&mut self, upper: bool, n: usize, i: usize, nb: usize, a: ZV, lda: usize) {
        let last = (i + nb - 1).min(n);
        let ld = self.lda;
        if upper {
            let first = (i - 1) * ld;
            let len = (last - 1) * ld + last - first;
            self.read_flat(first, len);
            self.copy_tri(true, i, last, 1, last, first, a, lda);
        } else {
            let first = (i - 1) * ld + (i - 1);
            let len = (last - 1) * ld + n - first;
            self.read_flat(first, len);
            self.copy_tri(false, i, last, i, n, first, a, lda);
        }
    }

    unsafe fn hemv(&mut self, upper: bool, m: usize, g0: usize, _blk: ZV, _lda: usize, x: ZV, y: ZV) {
        if m == 0 {
            return;
        }
        let mut xr = vec![0.0f64; self.lda];
        let mut xi = vec![0.0f64; self.lda];
        for q in 0..m {
            xr[q] = x.re.at(q);
            xi[q] = x.im.at(q);
        }
        if B::HEMV_T1 {
            let (tr, ti) = ops::ob_hemv_t1(&xr[..m], &xi[..m], (CONE.re, CONE.im));
            xr[..m].copy_from_slice(&tr);
            xi[..m].copy_from_slice(&ti);
        }
        self.counters.upload(self.lda * 8);
        self.counters.upload(self.lda * 8);
        let hx = ZHandle { re: upload_vec(self.client, xr), im: upload_vec(self.client, xi) };
        B::dev_zhemv::<R>(self.client, self.lanes, &self.m, self.lda, self.n, g0, m, &hx, &self.y, upper);
        self.counters.launch();
        read_range(self.client, &self.y.re, self.y_len, g0, m, y.re.slice_mut(m));
        read_range(self.client, &self.y.im, self.y_len, g0, m, y.im.slice_mut(m));
        self.counters.read(m * 8);
        self.counters.read(m * 8);
    }

    unsafe fn her2k(
        &mut self,
        upper: bool,
        m: usize,
        nb: usize,
        g0: usize,
        v: ZV,
        lda: usize,
        w: ZV,
        ldw: usize,
        _c: ZV,
    ) {
        if m == 0 {
            return;
        }
        let ld = self.lda;
        let mut pr = vec![0.0f64; 2 * nb * ld];
        let mut pi = vec![0.0f64; 2 * nb * ld];
        for l in 0..nb {
            for r in 0..m {
                let vv = v.at(ix(r + 1, l + 1, lda));
                let ww = w.at(ix(r + 1, l + 1, ldw));
                pr[g0 + r + l * ld] = vv.re;
                pi[g0 + r + l * ld] = vv.im;
                pr[g0 + r + (nb + l) * ld] = ww.re;
                pi[g0 + r + (nb + l) * ld] = ww.im;
            }
        }
        self.counters.upload(pr.len() * 8);
        self.counters.upload(pi.len() * 8);
        let hvw = ZHandle { re: upload_vec(self.client, pr), im: upload_vec(self.client, pi) };
        B::dev_zher2k::<R>(self.client, self.lanes, &self.m, ld, self.n, g0, m, &hvw, nb, upper);
        self.counters.launch();
        self.counters.sample(self.client);
    }

    unsafe fn tail(&mut self, upper: bool, _n: usize, g0: usize, mt: usize, a: ZV, lda: usize) {
        if mt == 0 {
            return;
        }
        let ld = self.lda;
        let first = g0 * ld + g0;
        let len = (g0 + mt - 1) * ld + g0 + mt - first;
        self.read_flat(first, len);
        self.copy_tri(upper, g0 + 1, g0 + mt, g0 + 1, g0 + mt, first, a, lda);
    }
}

/// `conj(T(p, q))` (0-based).
unsafe fn tt_conj(t: ZV, ldt: usize, p: usize, q: usize) -> Z {
    t.at(ix(p + 1, q + 1, ldt)).conj()
}

/// Device `ZLARFB('L','N',*,'C')` on the planar Z.
struct DeviceZOrm<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    lanes: &'a Lanes,
    counters: &'a Counters,
    z: ZHandle,
    lda: usize,
    ncols: usize,
    fast: bool,
}

impl<'a, R: Runtime> DeviceZOrm<'a, R> {
    /// `Exactness::Fast` `ZLARFB`: dense `V`, `W = CᴴV`, `X = W·Tᴴ`,
    /// `C −= V·Xᴴ` (see the real `DeviceOrm::larfb_fast`).
    unsafe fn larfb_fast(
        &mut self,
        forward: bool,
        m: usize,
        n: usize,
        k: usize,
        v: ZV,
        ldv: usize,
        t: ZV,
        ldt: usize,
        crow: usize,
    ) {
        let (client, lanes, lda, ncols) = (self.client, self.lanes, self.lda, self.ncols);
        let z = self.z.clone();
        let w = lanes.width;
        let kpad = k.div_ceil(w) * w;
        let dense = |r: usize, p: usize| -> Z {
            let dr = if forward { p } else { m - k + p };
            if r == dr {
                CONE
            } else if (forward && r < dr) || (!forward && r > dr) {
                Z::new(0.0, 0.0)
            } else {
                v.at(ix(r + 1, p + 1, ldv))
            }
        };
        let pad = crow % w;
        let ldvp = (pad + m).div_ceil(w) * w;
        let (mut rm_re, mut rm_im) = (vec![0.0f64; m * kpad], vec![0.0f64; m * kpad]);
        let (mut cm_re, mut cm_im) = (vec![0.0f64; ldvp * k], vec![0.0f64; ldvp * k]);
        for r in 0..m {
            for p in 0..k {
                let x = dense(r, p);
                rm_re[r * kpad + p] = x.re;
                rm_im[r * kpad + p] = x.im;
                cm_re[pad + r + p * ldvp] = x.re;
                cm_im[pad + r + p * ldvp] = x.im;
            }
        }
        // X(i,p) = Σ_q W(i,q)·conj(T(p,q)).
        let (mut c_re, mut c_im) = (vec![0.0f64; kpad * kpad], vec![0.0f64; kpad * kpad]);
        let mut diag = vec![0.0f64; 3 * kpad];
        for p in 0..k {
            let d = tt_conj(t, ldt, p, p);
            diag[p] = d.re;
            diag[kpad + p] = d.im;
            diag[2 * kpad + p] = 1.0;
            for q in 0..k {
                let c = tt_conj(t, ldt, p, q);
                c_re[q * kpad + p] = c.re;
                c_im[q * kpad + p] = c.im;
            }
        }
        let h_vrm = self.up_z(rm_re, rm_im);
        let h_vcm = self.up_z(cm_re, cm_im);
        let h_coef = self.up_z(c_re, c_im);
        let h_diag = self.up(diag);
        let empty = |c: &ComputeClient<R>| ZHandle { re: c.empty(n * kpad * 8), im: c.empty(n * kpad * 8) };
        let (zero, wa, wb) = (empty(client), empty(client), empty(client));
        ops::fill_zero::<R, f64>(client, &zero.re, n * kpad);
        ops::fill_zero::<R, f64>(client, &zero.im, n * kpad);
        ops::zgemm_c::<R, f64>(
            client,
            lanes,
            &z,
            lda,
            ncols,
            &h_vrm,
            m,
            &zero,
            &wa,
            n,
            kpad,
            crow,
            m,
            (1.0, 0.0),
            (1.0, 0.0),
        );
        ops::ztrmm_chain::<R, f64>(client, lanes, &wa, &wb, &h_coef, &h_diag, n, k, kpad, forward, true);
        ops::zgemm_n::<R, f64>(client, w, &z, lda, ncols, &h_vcm, ldvp, &wb, n, kpad, k, crow, m, (-1.0, -0.0));
        self.counters.sample(client);
        for _ in 0..5 {
            self.counters.launch();
        }
    }

    fn up(&self, v: Vec<f64>) -> cubecl::server::Handle {
        self.counters.upload(v.len() * 8);
        upload_vec(self.client, v)
    }
    fn up_z(&self, re: Vec<f64>, im: Vec<f64>) -> ZHandle {
        ZHandle { re: self.up(re), im: self.up(im) }
    }
}

impl<'a, R: Runtime> ZOrmEngine for DeviceZOrm<'a, R> {
    type B = crate::flavor::RefBlas;
    unsafe fn larfb(
        &mut self,
        forward: bool,
        m: usize,
        n: usize,
        k: usize,
        v: ZV,
        ldv: usize,
        t: ZV,
        ldt: usize,
        _c: ZV,
        crow: usize,
        _ldc: usize,
        _work: ZV,
        _ldwork: usize,
    ) {
        if m == 0 || n == 0 || k == 0 {
            return;
        }
        if self.fast {
            return self.larfb_fast(forward, m, n, k, v, ldv, t, ldt, crow);
        }
        let (client, lanes, lda, ncols) = (self.client, self.lanes, self.lda, self.ncols);
        let z = self.z.clone();
        let w = lanes.width;
        let kpad = k.div_ceil(w) * w;
        let vv = |i: usize, j: usize| v.at(ix(i + 1, j + 1, ldv));
        let tt = |i: usize, j: usize| t.at(ix(i + 1, j + 1, ldt));
        let sq0 = if forward { 0 } else { m - k };
        let rect0 = if forward { k } else { 0 };
        let nrect = m - k;
        // coef(j, kk) at kk·kpad + j, ALPHA (= CONE) folded in with the full formula.
        let coef_of = |f: &dyn Fn(usize, usize) -> Z| {
            let mut re = vec![0.0f64; kpad * kpad];
            let mut im = vec![0.0f64; kpad * kpad];
            for kk in 0..k {
                for j in 0..k {
                    let c = CONE * f(j, kk);
                    re[kk * kpad + j] = c.re;
                    im[kk * kpad + j] = c.im;
                }
            }
            (re, im)
        };
        // diag packs [re | im | flag]: 'N' variants always scale by ALPHA
        // (= CONE); 'C' variants scale by ALPHA·conj(T(j,j)) when it is not
        // CONE (unit V: never).
        let diag_n = {
            let mut d = vec![0.0f64; 3 * kpad];
            for j in 0..k {
                d[j] = CONE.re;
                d[kpad + j] = CONE.im;
                d[2 * kpad + j] = 1.0;
            }
            d
        };
        let diag_t = {
            let mut d = vec![0.0f64; 3 * kpad];
            for j in 0..k {
                let s = CONE * tt(j, j).conj();
                d[j] = s.re;
                d[kpad + j] = s.im;
                d[2 * kpad + j] = if s.eq(CONE) { 0.0 } else { 1.0 };
            }
            d
        };
        let diag_none = vec![0.0f64; 3 * kpad];
        // (coef source, above, asc, diag) for steps 2, 4, 6.
        let (c2, c4, c6) = (
            coef_of(&|j, kk| vv(sq0 + kk, j)),
            coef_of(&|j, kk| tt(j, kk).conj()),
            coef_of(&|j, kk| vv(sq0 + j, kk).conj()),
        );
        let (a2, a4, a6) = if forward { (true, true, false) } else { (false, false, true) };
        let (s2, s4, s6) = if forward { (true, true, false) } else { (true, false, true) };
        let h2 = self.up_z(c2.0, c2.1);
        let h4 = self.up_z(c4.0, c4.1);
        let h6 = self.up_z(c6.0, c6.1);
        let hd_n = self.up(diag_n);
        let hd_t = self.up(diag_t);
        let hd_none = self.up(diag_none);
        let empty = |c: &ComputeClient<R>| ZHandle { re: c.empty(n * kpad * 8), im: c.empty(n * kpad * 8) };
        let mut cur = empty(client);
        let mut spare = empty(client);
        let mut launches = 0;
        // 1. W = conj(rows of C).
        ops::zrows_to_w::<R, f64>(client, &z, lda, ncols, &cur, n, kpad, k, crow + sq0);
        // 2. ZTRMM with the square V block ('N', unit).
        ops::ztrmm_chain::<R, f64>(client, lanes, &cur, &spare, &h2, &hd_n, n, k, kpad, a2, s2);
        std::mem::swap(&mut cur, &mut spare);
        launches += 2;
        let mut h_vcm = None;
        if nrect > 0 {
            let pad = (crow + rect0) % w;
            let ldvp = (pad + nrect).div_ceil(w) * w;
            let (mut rm_re, mut rm_im) = (vec![0.0f64; nrect * kpad], vec![0.0f64; nrect * kpad]);
            let (mut cm_re, mut cm_im) = (vec![0.0f64; ldvp * k], vec![0.0f64; ldvp * k]);
            for l in 0..nrect {
                for j in 0..k {
                    let x = vv(rect0 + l, j);
                    rm_re[l * kpad + j] = x.re;
                    rm_im[l * kpad + j] = x.im;
                    cm_re[pad + l + j * ldvp] = x.re;
                    cm_im[pad + l + j * ldvp] = x.im;
                }
            }
            let h_vrm = self.up_z(rm_re, rm_im);
            // 3. W = CONE·(C_rectᴴ V_rect) + CONE·W.
            ops::zgemm_c::<R, f64>(
                client,
                lanes,
                &z,
                lda,
                ncols,
                &h_vrm,
                nrect,
                &cur,
                &spare,
                n,
                kpad,
                crow + rect0,
                nrect,
                (CONE.re, CONE.im),
                (CONE.re, CONE.im),
            );
            std::mem::swap(&mut cur, &mut spare);
            launches += 1;
            h_vcm = Some((self.up_z(cm_re, cm_im), ldvp));
        }
        // 4. ZTRMM with T ('C', non-unit).
        ops::ztrmm_chain::<R, f64>(client, lanes, &cur, &spare, &h4, &hd_t, n, k, kpad, a4, s4);
        std::mem::swap(&mut cur, &mut spare);
        launches += 1;
        if let Some((h_vcm, ldvp)) = &h_vcm {
            // 5. C_rect += V_rect · (−CONE·Wᴴ).
            let mone = -CONE;
            ops::zgemm_n::<R, f64>(
                client,
                w,
                &z,
                lda,
                ncols,
                h_vcm,
                *ldvp,
                &cur,
                n,
                kpad,
                k,
                crow + rect0,
                nrect,
                (mone.re, mone.im),
            );
            launches += 1;
        }
        // 6. ZTRMM with the square V block ('C', unit).
        ops::ztrmm_chain::<R, f64>(client, lanes, &cur, &spare, &h6, &hd_none, n, k, kpad, a6, s6);
        std::mem::swap(&mut cur, &mut spare);
        // 7. C_sq -= conj(W).
        ops::zsub_w_from_rows::<R, f64>(client, &z, lda, ncols, &cur, n, kpad, k, crow + sq0);
        launches += 2;
        self.counters.sample(client);
        for _ in 0..launches {
            self.counters.launch();
        }
    }
}

/// `ZHEEVD` on the device (planar `a_re`/`a_im`, column-major `n × n`, valid
/// in the `uplo` triangle). On return `w` holds ascending eigenvalues and,
/// for [`Job::Vectors`], `a_re`/`a_im` hold the eigenvectors. Bit-identical
/// to [`crate::zheevd_alloc`] under `Exactness::Reference` on a
/// non-contracting backend.
pub fn zheevd_device<R: Runtime>(
    client: &ComputeClient<R>,
    job: Job,
    uplo: Uplo,
    n: usize,
    a_re: &mut [f64],
    a_im: &mut [f64],
    w: &mut [f64],
    ws: &mut DeviceWorkspace<R>,
) -> Result<(), LapackError> {
    zheevd_device_impl::<R, crate::flavor::RefBlas>(client, job, uplo, n, a_re, a_im, w, ws, None, false).map(|_| ())
}

/// A planar complex column-major `lda × n` matrix on the device.
#[derive(Clone, Debug)]
pub struct DeviceZMatrix {
    pub re: Handle,
    pub im: Handle,
    pub lda: usize,
    pub n: usize,
}

impl DeviceZMatrix {
    /// New `n × n` column-major `(re, im)` buffers holding the leading block.
    pub fn compact<R: Runtime>(&self, client: &ComputeClient<R>) -> (Handle, Handle) {
        (
            ops::compact::<R, f64>(client, &self.re, self.n, self.lda),
            ops::compact::<R, f64>(client, &self.im, self.n, self.lda),
        )
    }
}

/// Device-resident [`zheevd_device`]: `a_re`/`a_im` are `n × n` column-major
/// f64 device buffers (valid in the `uplo` triangle, unchanged). Returns the
/// eigenvectors on the device for [`Job::Vectors`]. Bit-identical to
/// [`zheevd_device`].
#[allow(clippy::too_many_arguments)]
pub fn zheevd_device_resident<R: Runtime>(
    client: &ComputeClient<R>,
    job: Job,
    uplo: Uplo,
    n: usize,
    a_re: &Handle,
    a_im: &Handle,
    w: &mut [f64],
    ws: &mut DeviceWorkspace<R>,
) -> Result<Option<DeviceZMatrix>, LapackError> {
    probe::require_f64(client)?;
    let (mut hr, mut hi) = (vec![0.0f64; n * n], vec![0.0f64; n * n]);
    read_flat_chunked(client, &ws.counters, a_re, n * n, &mut hr);
    read_flat_chunked(client, &ws.counters, a_im, n * n, &mut hi);
    zheevd_device_impl::<R, crate::flavor::RefBlas>(
        client,
        job,
        uplo,
        n,
        &mut hr,
        &mut hi,
        w,
        ws,
        Some((a_re, a_im)),
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn zheevd_device_impl<R: Runtime, B: super::flavor::DeviceBlas>(
    client: &ComputeClient<R>,
    job: Job,
    uplo: Uplo,
    n: usize,
    a_re: &mut [f64],
    a_im: &mut [f64],
    w: &mut [f64],
    ws: &mut DeviceWorkspace<R>,
    source: Option<(&Handle, &Handle)>,
    resident: bool,
) -> Result<Option<DeviceZMatrix>, LapackError> {
    probe::require_f64(client)?;
    if a_re.len() < n * n || a_im.len() < n * n {
        return Err(LapackError::InvalidArgument {
            arg: "a_re/a_im",
            reason: format!("need n*n = {} entries each", n * n),
        });
    }
    if w.len() < n {
        return Err(LapackError::InvalidArgument { arg: "w", reason: format!("need n = {n} entries") });
    }
    let wantz = job == Job::Vectors;
    if n == 0 {
        return Ok(None);
    }
    if n == 1 {
        w[0] = a_re[0];
        if !wantz {
            return Ok(None);
        }
        a_re[0] = 1.0;
        a_im[0] = 0.0;
        if !resident {
            return Ok(None);
        }
        let lda = geometry::lda_for(client, 1);
        let mut img = vec![0.0f64; lda];
        img[0] = 1.0;
        ws.stage_complex(img, vec![0.0f64; lda]);
        return Ok(Some(DeviceZMatrix { re: ws.m.clone(), im: ws.m_im.clone(), lda, n }));
    }
    let (lean, chunk_rows) = ws.plan(n, wantz, 2)?;
    let lean = lean && B::FAST_OK;
    ws.reserve(n);
    let upper = uplo == Uplo::Upper;
    let lower = !upper;
    let lda = geometry::lda_for(client, n);
    let mut t = std::time::Instant::now();

    let smlnum = SAFMIN / PREC;
    let bignum = 1.0 / smlnum;
    let (rmin, rmax) = (smlnum.sqrt(), bignum.sqrt());
    // SAFETY: a_re/a_im have n*n entries (checked).
    let anrm = unsafe { crate::heevd::zlanhe_m(upper, n, ZV::new(V::new(a_re), V::new(a_im)), n) };
    let sigma = if anrm > 0.0 && anrm < rmin {
        Some(rmin / anrm)
    } else if anrm > rmax {
        Some(rmax / anrm)
    } else {
        None
    };
    if let Some(s) = sigma {
        // SAFETY: as above.
        unsafe { crate::heevd::zlascl_tri(lower, 1.0, s, n, ZV::new(V::new(a_re), V::new(a_im)), n) };
    }

    let mut e = vec![0.0f64; n - 1];
    let (mut tau_re, mut tau_im) = (vec![0.0f64; n - 1], vec![0.0f64; n - 1]);
    let lwork = (n * SYTRD_NB).max(n * ORM_NB + TSIZE);
    let (mut work_re, mut work_im) = (vec![0.0f64; lwork], vec![0.0f64; lwork]);

    match source {
        Some((sr, si)) if sigma.is_none() => ws.stage_complex_from_device(sr, si, n, lda, upper),
        _ => ws.stage_complex_from_host(a_re, a_im, n, lda, upper),
    }
    ws.counters.mark(client, Phase::Stage, &mut t);
    if !B::ZTRD_ON_DEVICE {
        // ZHETRD on the host in the flavour's BLAS (the staged M is only the
        // Z buffer from here on).
        // SAFETY: as the device engine below, with host views only.
        unsafe {
            crate::hetrd::zhetrd_with(
                &mut crate::hetrd::HostZTrd::<B>::new(),
                upper,
                n,
                ZV::new(V::new(a_re), V::new(a_im)),
                n,
                V::new(w),
                V::new(&mut e),
                ZV::new(V::new(&mut tau_re), V::new(&mut tau_im)),
                ZV::new(V::new(&mut work_re), V::new(&mut work_im)),
                n * SYTRD_NB,
            );
        }
    } else {
        let DeviceWorkspace { lanes, counters, m, m_im, y, y_im, y_reals, .. } = &*ws;
        let mut eng = DeviceZTrd::<'_, R, B> {
            client,
            lanes,
            counters,
            m: ZHandle { re: m.clone(), im: m_im.clone() },
            lda,
            n,
            y: ZHandle { re: y.clone(), im: y_im.clone() },
            y_len: *y_reals,
            buf_re: Vec::new(),
            buf_im: Vec::new(),
            flavor: std::marker::PhantomData,
        };
        // SAFETY: host lengths checked or allocated; the engine touches device
        // buffers sized `lda × n`.
        unsafe {
            crate::hetrd::zhetrd_with(
                &mut eng,
                upper,
                n,
                ZV::new(V::new(a_re), V::new(a_im)),
                n,
                V::new(w),
                V::new(&mut e),
                ZV::new(V::new(&mut tau_re), V::new(&mut tau_im)),
                ZV::new(V::new(&mut work_re), V::new(&mut work_im)),
                n * SYTRD_NB,
            );
        }
    }
    ws.counters.mark(client, Phase::Trd, &mut t);

    let info;
    if !wantz {
        // SAFETY: w has n entries, e has n-1.
        info = unsafe { crate::steqr::dsterf_raw(n, V::new(w), V::new(&mut e)) };
        ws.counters.mark(client, Phase::Stedc, &mut t);
    } else {
        let mut dummy = [0.0f64; 1];
        if n <= SMLSIZ {
            // ZSTEQR on the host (n ≤ 25), then upload Z.
            let (mut zr, mut zi) = (vec![0.0f64; n * n], vec![0.0f64; n * n]);
            let mut rw = vec![0.0f64; 2 * n];
            // SAFETY: w/e/z/rw lengths allocated above.
            info = unsafe {
                crate::heevd::zsteqr_i(
                    n,
                    V::new(w),
                    V::new(&mut e),
                    ZV::new(V::new(&mut zr), V::new(&mut zi)),
                    n,
                    V::new(&mut rw),
                )
            };
            let (mut ir, mut ii) = (vec![0.0f64; lda * n], vec![0.0f64; lda * n]);
            for j in 0..n {
                ir[j * lda..j * lda + n].copy_from_slice(&zr[j * n..(j + 1) * n]);
                ii[j * lda..j * lda + n].copy_from_slice(&zi[j * n..(j + 1) * n]);
            }
            ws.stage_complex(ir, ii);
        } else {
            let (lw, liw) = crate::stedc::stedc_work_len(n);
            let mut dwork = vec![0.0f64; lw.max(1)];
            if ws.iwork.len() < liw.max(1) {
                ws.iwork.resize(liw.max(1), 0);
            }
            let DeviceWorkspace { lanes, counters, m, m_im, iwork, .. } = &mut *ws;
            // ZLACP2: the imaginary part of Z is exactly zero.
            ops::fill_zero::<R, f64>(client, m_im, lda * n);
            counters.launch();
            let mut eng =
                stedc::DeviceDc::<'_, R, B>::new_flavor(client, lanes, counters, m.clone(), lda, n, chunk_rows);
            eng.lean_min_k = if lean { super::syevd::LEAN_MIN_K } else { usize::MAX };
            // SAFETY: w/e/dwork/iwork allocated above; the engine never
            // touches the host Z view.
            info = unsafe {
                crate::stedc::dstedc_i_with(
                    &mut eng,
                    n,
                    V::new(w),
                    V::new(&mut e),
                    V::new(&mut dummy),
                    n,
                    V::new(&mut dwork),
                    V::new(iwork),
                )
            };
        }
        ws.counters.mark(client, Phase::Stedc, &mut t);
        let lwork_orm = B::zorm_lwork(n);
        let DeviceWorkspace { lanes, counters, m, m_im, exactness, .. } = &*ws;
        let a = ZV::new(V::new(a_re), V::new(a_im));
        let tau = ZV::new(V::new(&mut tau_re), V::new(&mut tau_im));
        let work = ZV::new(V::new(&mut work_re), V::new(&mut work_im));
        if B::ZORM_KIND == 2 {
            let mut eng =
                DeviceZOrmOb { client, lanes, counters, z: ZHandle { re: m.clone(), im: m_im.clone() }, lda, ncols: n };
            let (mut dummy_re, mut dummy_im) = ([0.0f64; 1], [0.0f64; 1]);
            // SAFETY: reflectors, tau and work allocated/checked; the device
            // engine applies every reflector to Z on the device and never
            // touches the host C view.
            unsafe {
                crate::hetrd::zunmtr_ln_with(
                    &mut eng,
                    upper,
                    n,
                    n,
                    a,
                    n,
                    tau,
                    ZV::new(V::new(&mut dummy_re), V::new(&mut dummy_im)),
                    n,
                    work,
                    lwork_orm,
                );
            }
            counters.mark(client, Phase::Orm, &mut t);
            if !resident {
                read_columns(client, counters, m, lda, n, a_re);
                read_columns(client, counters, m_im, lda, n, a_im);
            }
            counters.mark(client, Phase::Readback, &mut t);
        } else if crate::sytrd::orm_blocking(n, n - 1, lwork_orm).is_some() {
            let mut eng = DeviceZOrm {
                client,
                lanes,
                counters,
                z: ZHandle { re: m.clone(), im: m_im.clone() },
                lda,
                ncols: n,
                fast: *exactness == super::syevd::Exactness::Fast,
            };
            let mut dummy_im = [0.0f64; 1];
            // SAFETY: reflectors, tau and work allocated/checked; the device
            // engine never touches the host C view on the blocked path.
            unsafe {
                crate::hetrd::zunmtr_ln_with(
                    &mut eng,
                    upper,
                    n,
                    n,
                    a,
                    n,
                    tau,
                    ZV::new(V::new(&mut dummy), V::new(&mut dummy_im)),
                    n,
                    work,
                    lwork_orm,
                );
            }
            counters.mark(client, Phase::Orm, &mut t);
            if !resident {
                read_columns(client, counters, m, lda, n, a_re);
                read_columns(client, counters, m_im, lda, n, a_im);
            }
            counters.mark(client, Phase::Readback, &mut t);
        } else {
            let (mut zr, mut zi) = (vec![0.0f64; n * n], vec![0.0f64; n * n]);
            read_columns(client, counters, m, lda, n, &mut zr);
            read_columns(client, counters, m_im, lda, n, &mut zi);
            // SAFETY: host views of n*n entries.
            unsafe {
                crate::hetrd::zunmtr_ln_with(
                    &mut crate::hetrd::HostZOrm::<B>::new(),
                    upper,
                    n,
                    n,
                    a,
                    n,
                    tau,
                    ZV::new(V::new(&mut zr), V::new(&mut zi)),
                    n,
                    work,
                    lwork_orm,
                );
            }
            a_re[..n * n].copy_from_slice(&zr);
            a_im[..n * n].copy_from_slice(&zi);
            if resident {
                let (mut ir, mut ii) = (vec![0.0f64; lda * n], vec![0.0f64; lda * n]);
                for j in 0..n {
                    ir[j * lda..j * lda + n].copy_from_slice(&zr[j * n..(j + 1) * n]);
                    ii[j * lda..j * lda + n].copy_from_slice(&zi[j * n..(j + 1) * n]);
                }
                ws.stage_complex(ir, ii);
            }
        }
    }
    if let Some(s) = sigma {
        let imax = if info == 0 { n } else { (info - 1).min(n) };
        // SAFETY: w has n entries.
        unsafe { B::dscal(imax, 1.0 / s, V::new(w), 1) };
    }
    if info != 0 {
        return Err(LapackError::NoConvergence { routine: if wantz { "zstedc" } else { "dsterf" }, info });
    }
    Ok(if resident && wantz { Some(DeviceZMatrix { re: ws.m.clone(), im: ws.m_im.clone(), lda, n }) } else { None })
}

/// **OpenBLAS track.** `ZUNMTR` engine for scipy's workspace query, where
/// `ZUNMQL`/`ZUNMQR` always take the unblocked legacy `ZUNM2L`/`ZUNM2R`:
/// every `ZLARF` runs on the device (`ops::ob_zlarf_left`).
pub(crate) struct DeviceZOrmOb<'a, R: Runtime> {
    pub client: &'a ComputeClient<R>,
    pub lanes: &'a Lanes,
    pub counters: &'a Counters,
    pub z: ZHandle,
    pub lda: usize,
    pub ncols: usize,
}

impl<'a, R: Runtime> ZOrmEngine for DeviceZOrmOb<'a, R> {
    type B = crate::openblas::OpenBlas;
    unsafe fn larfb(
        &mut self,
        _forward: bool,
        m: usize,
        n: usize,
        k: usize,
        _v: ZV,
        _ldv: usize,
        _t: ZV,
        _ldt: usize,
        _c: ZV,
        _crow: usize,
        _ldc: usize,
        _work: ZV,
        _ldwork: usize,
    ) {
        unreachable!("OpenBLAS ZUNMTR with the query LWORK is unblocked (m={m}, n={n}, k={k})")
    }
    unsafe fn larf_left(&mut self, m: usize, n: usize, v: ZV, tau: Z, crow: usize) -> bool {
        let (vr, vi) = v.zs(m);
        ops::ob_zlarf_left::<R, f64>(
            self.client,
            self.lanes,
            &self.z,
            self.lda,
            self.ncols,
            crow,
            m,
            n,
            vr,
            vi,
            (tau.re, tau.im),
            (-tau.re, -tau.im),
        );
        self.counters.launch();
        self.counters.launch();
        self.counters.launch();
        true
    }
}

/// **OpenBLAS track.** [`zheevd_device`] bit-identical to
/// [`crate::openblas::zheevd_host`] (scipy's OpenBLAS 0.3.30, SkylakeX core,
/// one thread, its own workspace query): `ZHETRD` on the host, `DSTEDC` and
/// the unblocked `ZUNMTR` with the OpenBLAS-order device kernels.
pub fn zheevd_device_openblas<R: Runtime>(
    client: &ComputeClient<R>,
    job: Job,
    uplo: Uplo,
    n: usize,
    a_re: &mut [f64],
    a_im: &mut [f64],
    w: &mut [f64],
    ws: &mut DeviceWorkspace<R>,
) -> Result<(), LapackError> {
    crate::openblas::require_fma();
    zheevd_device_impl::<R, crate::openblas::OpenBlas>(client, job, uplo, n, a_re, a_im, w, ws, None, false).map(|_| ())
}
