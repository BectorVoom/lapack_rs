//! Hybrid `DSTEDC('I')`: deflation, sorting, the secular equation and the
//! `K × K` eigenvector matrix ("staged U") stay on the host; the eigenvector
//! matrix `Z` stays on the device. The host transcription
//! ([`crate::stedc::dstedc_i_with`]) is shared, so control flow and every
//! scalar are the reference's; the device engine replaces each `Q` access:
//!
//! * leaves: host `DSTEQR` into a scratch block, one upload, one K7 scatter
//!   (`Z` is a true identity first: `DLAED2`'s Givens deflation rotates whole
//!   columns, so structural zeros — including their sign — reach the output);
//! * `DLAED1`'s two row copies: K7 row gathers;
//! * `DLAED2`'s rotations: batched per merge into one K7 launch that masks
//!   rows outside the block;
//! * the `Q2` gather plus `DLAED3`'s two `DGEMM`s: K3m over row chunks of `R`
//!   rows into scratch, copied back in place (no second `n²` buffer);
//! * `DLAED0`'s reorder and `DSTEDC`'s final swaps: K3m gathers / K7 swaps.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::flavor::DeviceBlas;
use super::ops::{self, Chunk, Lanes};
use super::workspace::{Counters, Span, read_range, upload, upload_vec};
use crate::flavor::RefBlas;
use crate::ptr::{D, I, V, ix};
use crate::stedc::{DcEngine, SecularMerge};
use crate::steqr::{Compz, dsteqr_raw};
use std::marker::PhantomData;

/// Smallest merge rank `K` whose `DLAED3` runs on the device (K4); smaller
/// merges keep the host transcription. Bits are identical either way.
pub const SECULAR_MIN_K: usize = 64;

static SECULAR_MIN_K_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Override [`SECULAR_MIN_K`] (tests: 3 sends every `K > 2` merge to the
/// device, `usize::MAX` keeps all on the host; 0 restores the default). K4
/// still only runs where `probe::k4_supported` reports it does (any runtime
/// whose cubecl-opt has the PRE/GVN hoist fix; unsupported runtimes fall
/// back to the host automatically).
#[doc(hidden)]
pub fn set_secular_min_k(k: usize) {
    SECULAR_MIN_K_OVERRIDE.store(k, std::sync::atomic::Ordering::Relaxed);
}

/// `DLAMCH` and `dnrm2.f90` constants for K4, evaluated as the host does.
pub(crate) fn secular_consts() -> ops::SecularConsts<f64> {
    use crate::blas::{SBIG, SSML, TBIG, TSML};
    ops::SecularConsts {
        eps: crate::aux::EPS,
        small1: crate::laed4::dlaed6_small1(),
        tsml: TSML,
        tbig: TBIG,
        ssml: SSML,
        sbig: SBIG,
        maxn: f64::MAX,
    }
}

pub(crate) struct DeviceDc<'a, R: Runtime, B: DeviceBlas = RefBlas> {
    pub client: &'a ComputeClient<R>,
    pub lanes: &'a Lanes,
    pub counters: &'a Counters,
    pub z: Handle,
    pub lda: usize,
    pub ncols: usize,
    /// Merge row chunk (rows).
    pub chunk_rows: usize,
    pending: Vec<(u32, u32, f64, f64)>,
    pending_block: (usize, usize),
    uk: Vec<f64>,
    leaf: Vec<f64>,
    /// Lean merges (no `k × k` U) for `k >= lean_min_k`; `usize::MAX` = staged.
    pub lean_min_k: usize,
    flavor: PhantomData<B>,
}

impl<'a, R: Runtime> DeviceDc<'a, R, RefBlas> {
    pub fn new(
        client: &'a ComputeClient<R>,
        lanes: &'a Lanes,
        counters: &'a Counters,
        z: Handle,
        lda: usize,
        ncols: usize,
        chunk_rows: usize,
    ) -> Self {
        Self::new_flavor(client, lanes, counters, z, lda, ncols, chunk_rows)
    }
}

impl<'a, R: Runtime, B: DeviceBlas> DeviceDc<'a, R, B> {
    pub fn new_flavor(
        client: &'a ComputeClient<R>,
        lanes: &'a Lanes,
        counters: &'a Counters,
        z: Handle,
        lda: usize,
        ncols: usize,
        chunk_rows: usize,
    ) -> Self {
        DeviceDc {
            client,
            lanes,
            counters,
            z,
            lda,
            ncols,
            chunk_rows: chunk_rows.max(1),
            pending: Vec::new(),
            pending_block: (0, 0),
            uk: Vec::new(),
            leaf: Vec::new(),
            lean_min_k: usize::MAX,
            flavor: PhantomData,
        }
    }

    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let span = self.counters.span_start();
        let (g0, n) = self.pending_block;
        let mut cols = Vec::with_capacity(2 * self.pending.len());
        let mut cs = Vec::with_capacity(2 * self.pending.len());
        for &(x, y, c, s) in &self.pending {
            cols.extend_from_slice(&[x, y]);
            cs.extend_from_slice(&[c, s]);
        }
        let nrot = self.pending.len();
        self.counters.upload(cols.len() * 4 + cs.len() * 8);
        let hc = upload_vec(self.client, cols);
        let hs = upload_vec(self.client, cs);
        B::rotations::<R>(self.client, self.lanes, &self.z, self.lda, self.ncols, &hc, &hs, nrot, g0, g0 + n);
        self.counters.launch();
        self.pending.clear();
        self.counters.span_end(Span::StedcGather, span);
    }

    /// Row chunks covering `[g0, g0+n)`, aligned to lane-vectors.
    fn chunks(&self, g0: usize, n: usize) -> Vec<(Chunk, usize, usize)> {
        let w = self.lanes.width;
        let r = self.chunk_rows.div_ceil(w).max(1);
        let (first, last) = (g0 / w, (g0 + n).div_ceil(w));
        let mut out = Vec::new();
        let mut rv = first;
        while rv < last {
            let rvecs = r.min(last - rv);
            let lo = (rv * w).max(g0);
            let hi = ((rv + rvecs) * w).min(g0 + n);
            out.push((Chunk { rv0: rv, rvecs }, lo, hi));
            rv += rvecs;
        }
        out
    }

    /// Gather-only merge: block columns `j` := old column `perm[j]` (global).
    fn gather_block(&mut self, g0: usize, n: usize, perm: Vec<u32>) {
        let span = self.counters.span_start();
        self.counters.upload(perm.len() * 4);
        let hp = upload_vec(self.client, perm);
        let dummy_perm = upload(self.client, &[0u32]);
        let dummy_s = upload(self.client, &[0.0f64]);
        for (chunk, lo, hi) in self.chunks(g0, n) {
            let dst = self.client.empty(chunk.rvecs * self.lanes.width * n * 8);
            ops::merge_rows::<R, f64>(
                self.client,
                self.lanes,
                &self.z,
                self.lda,
                self.ncols,
                &dst,
                chunk,
                n,
                0,
                &dummy_perm,
                0,
                &hp,
                &dummy_s,
                1,
                0,
                lo,
                hi,
            );
            self.counters.launch();
            ops::copy_back::<R, f64>(
                self.client,
                self.lanes,
                &self.z,
                self.lda,
                self.ncols,
                &dst,
                chunk,
                g0,
                n,
                lo,
                hi,
            );
            self.counters.launch();
        }
        self.counters.span_end(Span::StedcGather, span);
    }
}

impl<'a, R: Runtime, B: DeviceBlas> DeviceDc<'a, R, B> {
    /// Label 110 over the row chunks: for the top rows `[g0, g0+n1)` and the
    /// bottom rows `[g0+n1, g0+n)`, `(perm, nterms, s, lds, s_row0)` drive
    /// K3m; every chunk is then copied back into `Z`.
    fn merge_chunks(
        &mut self,
        g0: usize,
        n: usize,
        n1: usize,
        k: usize,
        blocks: [(&Handle, usize, &Handle, usize, usize); 2],
        perm4: &Handle,
    ) {
        let ranges = [(g0, g0 + n1), (g0 + n1, g0 + n)];
        let blks = [(g0, n1), (g0 + n1, n - n1)];
        for (chunk, lo, hi) in self.chunks(g0, n) {
            let dst = self.client.empty(chunk.rvecs * self.lanes.width * n * 8);
            for (((perm, nterms, s, lds, s_row0), (rlo, rhi)), (blk_r0, blk_m)) in blocks.iter().zip(ranges).zip(blks) {
                let (bl, bh) = (lo.max(rlo), hi.min(rhi));
                if bl < bh {
                    B::merge_rows::<R>(
                        self.client,
                        self.lanes,
                        &self.z,
                        self.lda,
                        self.ncols,
                        &dst,
                        chunk,
                        n,
                        k,
                        perm,
                        *nterms,
                        perm4,
                        s,
                        *lds,
                        *s_row0,
                        bl,
                        bh,
                        blk_r0,
                        blk_m,
                    );
                    self.counters.launch();
                }
            }
            self.counters.sample(self.client);
            ops::copy_back::<R, f64>(
                self.client,
                self.lanes,
                &self.z,
                self.lda,
                self.ncols,
                &dst,
                chunk,
                g0,
                n,
                lo,
                hi,
            );
            self.counters.launch();
        }
    }
}

impl<'a, R: Runtime, B: DeviceBlas> DcEngine for DeviceDc<'a, R, B> {
    type B = B;
    fn split(&self) -> Option<(Self, Self)> {
        None
    }

    unsafe fn identity(&mut self, n: usize, _z: D, _ldz: usize) {
        ops::fill_identity::<R, f64>(self.client, &self.z, self.lda, self.ncols, 0, n);
        self.counters.launch();
    }

    unsafe fn steqr(&mut self, g0: usize, m: usize, d: D, e: D, _zblk: D, _ldz: usize, work: D) -> usize {
        let span = self.counters.span_start();
        self.flush();
        if self.leaf.len() < m * m {
            self.leaf.resize(m * m, 0.0);
        }
        let zb = V::new(&mut self.leaf[..m * m]);
        let info = dsteqr_raw(Compz::I, m, d, e, zb, m, work);
        self.counters.upload(m * m * 8);
        let h = upload(self.client, &self.leaf[..m * m]);
        ops::scatter_block::<R, f64>(self.client, &self.z, self.lda, self.ncols, &h, g0, m);
        self.counters.launch();
        self.counters.span_end(Span::StedcLeaf, span);
        info
    }

    unsafe fn read_row(&mut self, g0: usize, row: usize, c0: usize, len: usize, _q: D, _ldq: usize, out: D) {
        if len == 0 {
            return;
        }
        self.flush();
        let h = self.client.empty(len * 8);
        ops::gather_row::<R, f64>(self.client, &self.z, self.lda, self.ncols, &h, g0 + row - 1, g0 + c0 - 1, len);
        self.counters.launch();
        read_range(self.client, &h, len, 0, len, out.slice_mut(len));
        self.counters.read(len * 8);
    }

    unsafe fn drot(&mut self, g0: usize, n: usize, pj: usize, nj: usize, c: f64, s: f64, _q: D, _ldq: usize) {
        self.pending_block = (g0, n);
        self.pending.push(((g0 + pj - 1) as u32, (g0 + nj - 1) as u32, c, s));
    }

    unsafe fn gather_all(&mut self, g0: usize, n: usize, indx: I, _q: D, _ldq: usize, _q2: D) {
        self.flush();
        let perm: Vec<u32> = (0..n).map(|j| (g0 + indx.at(j) - 1) as u32).collect();
        self.gather_block(g0, n, perm);
    }

    unsafe fn build_q2(
        &mut self,
        _g0: usize,
        _n: usize,
        _n1: usize,
        _k: usize,
        _ctot: [usize; 4],
        _indx: I,
        _q: D,
        _ldq: usize,
        _q2: D,
    ) {
        self.flush();
    }

    unsafe fn uk(&mut self, k: usize, _q: D, _ldq: usize) -> (D, usize) {
        if self.uk.len() < k * k {
            self.uk.resize(k * k, 0.0);
        }
        (V::new(&mut self.uk[..k * k]), k)
    }

    unsafe fn back(
        &mut self,
        g0: usize,
        n: usize,
        n1: usize,
        k: usize,
        ctot: [usize; 4],
        indx: I,
        uk: D,
        lduk: usize,
        _q: D,
        _ldq: usize,
        _q2: D,
        _s: D,
    ) {
        self.flush();
        let span = self.counters.span_start();
        let n12 = ctot[0] + ctot[1];
        let n23 = ctot[1] + ctot[2];
        let gcol = |p: usize| (g0 + indx.at(p) - 1) as u32;
        let perm_top: Vec<u32> = (0..n12).map(gcol).collect();
        let perm_bot: Vec<u32> = (ctot[0]..ctot[0] + n23).map(gcol).collect();
        let perm4: Vec<u32> = (k..n).map(gcol).collect();
        let mut s_top = vec![0.0f64; (n12 * k).max(1)];
        let mut s_bot = vec![0.0f64; (n23 * k).max(1)];
        for j in 0..k {
            for l in 0..n12 {
                s_top[l + j * n12] = uk.at(ix(l + 1, j + 1, lduk));
            }
            for l in 0..n23 {
                s_bot[l + j * n23] = uk.at(ix(ctot[0] + l + 1, j + 1, lduk));
            }
        }
        let bytes = 4 * (n12 + n23 + (n - k)) + 8 * (s_top.len() + s_bot.len());
        self.counters.upload(bytes);
        let h_pt = upload_vec(self.client, if perm_top.is_empty() { vec![0] } else { perm_top });
        let h_pb = upload_vec(self.client, if perm_bot.is_empty() { vec![0] } else { perm_bot });
        let h_p4 = upload_vec(self.client, if perm4.is_empty() { vec![0] } else { perm4 });
        let h_st = upload_vec(self.client, s_top);
        let h_sb = upload_vec(self.client, s_bot);
        self.merge_chunks(g0, n, n1, k, [(&h_pt, n12, &h_st, n12, 0), (&h_pb, n23, &h_sb, n23, 0)], &h_p4);
        self.counters.span_end(Span::StedcMerge, span);
    }

    unsafe fn secular(&mut self, g: SecularMerge<'_>) -> Option<usize> {
        let min_k = match SECULAR_MIN_K_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
            0 => SECULAR_MIN_K,
            v => v,
        };
        // K4 (`secular_roots_kernel`, which inlines `dlaed4_mid`) used to be
        // gated on `has_planes` (GPU runtimes only), because of two distinct
        // cubecl-cpu 0.10 / cubecl-opt 0.10 bugs found while bringing it up
        // on the CPU runtime: a `while` loop with a decremented counter
        // failed to lower (fixed by rewriting `dlaed4_mid`'s four such loops
        // as ascending `for` loops over the identical visit order, see
        // `src/device/kernels/secular.rs`), and a cubecl-opt PRE/GVN bug
        // (`cubecl-opt-0.10.0/src/gvn/convert.rs:22:39: "no entry found for
        // key"`, root-caused in `src/gvn/apply.rs::insert_block`: it hoisted
        // an expression into a predecessor block that had no leaders for
        // that expression's own operands). Both are now fixed: the vendored
        // `crates/cubecl-opt` (`[patch.crates-io]` in `Cargo.toml`) carries a
        // one-function patch that skips a hoist unless every predecessor can
        // actually materialise it — PRE/CSE never reassociates, so skipping
        // a hoist only forgoes an optimisation and cannot change a
        // floating-point value (see `crates/cubecl-opt/PATCHED.md`). K4 now
        // runs correctly on the CubeCL CPU runtime
        // (`tests/device_secular_cpu.rs`).
        //
        // `[patch.crates-io]` only rewires builds from THIS workspace: a
        // downstream consumer of a published `lapack_rs` with `--features
        // cpu` (or `rocm`/`cuda`, which share this same cubecl-opt) would
        // get upstream, unpatched cubecl-opt, hit the identical ICE on a
        // cubecl worker thread, and get a launch that silently no-ops and
        // reads back zeros — wrong results with no error, not a build
        // failure. So the gate is not simply gone; it is replaced by a
        // cached, per-runtime-type capability probe
        // (`probe::k4_supported`) that launches the real
        // `secular_roots_kernel` on a tiny `k = 3` fixture and checks its
        // `dlam`/`info` against the host `DLAED4` computed at runtime: an
        // unpatched build's silent zeros fail that comparison and this
        // falls back to the host transcription automatically, exactly as
        // `has_planes` used to for a GPU without planes.
        if !B::SECULAR_ON_DEVICE || g.k < min_k || !super::probe::k4_supported(self.client) {
            return None;
        }
        self.flush();
        let span = self.counters.span_start();
        let (client, k, n) = (self.client, g.k, g.n);
        let neg: Vec<u32> = g.w.iter().map(|x| x.is_sign_negative() as u32).collect();
        // `INDXC` is DLAED2's 1-based row permutation of the K deflated
        // columns; the K4 `U` kernel indexes `delta` with it unchecked.
        assert!(g.indx.iter().all(|&i| (1..=k).contains(&i)), "DLAED3: INDX out of range");
        let indx0: Vec<u32> = g.indx.iter().map(|&i| (i - 1) as u32).collect();
        let h_d = upload(client, g.dlambda);
        let h_z = upload(client, g.w);
        let h_neg = upload_vec(client, neg);
        let h_indx = upload_vec(client, indx0);
        self.counters.upload(k * (8 + 8 + 4 + 4));
        let delta = client.empty(k * k * 8);
        let u = client.empty(k * k * 8);
        let dlam = client.empty(k * 8);
        let info = client.empty(k * 4);
        let wv = client.empty(k * 8);
        let consts = secular_consts();
        ops::secular_roots::<R, f64>(client, &h_d, &h_z, &delta, &dlam, &info, k, g.rho, &consts);
        self.counters.launch();
        let mut infos = vec![0u32; k];
        read_range(client, &info, k, 0, k, &mut infos);
        read_range(client, &dlam, k, 0, k, g.d.slice_mut(k));
        self.counters.read(k * 12);
        if let Some(&bad) = infos.iter().find(|&&v| v != 0) {
            self.counters.span_end(Span::StedcMerge, span);
            return Some(bad as usize);
        }
        ops::secular_w::<R, f64>(client, &delta, &h_d, &h_neg, &wv, k);
        ops::secular_u::<R, f64>(client, &wv, &h_indx, &delta, &u, k, &consts);
        self.counters.launch();
        self.counters.launch();
        drop(delta);
        // Label 110 with U on the device: top rows 0..N12, bottom rows CTOT(1)..
        let ct = g.ctot;
        let (n12, n23) = (ct[0] + ct[1], ct[1] + ct[2]);
        let gcol = |p: usize| (g.g0 + g.types[p] - 1) as u32;
        let nz = |v: Vec<u32>| if v.is_empty() { vec![0] } else { v };
        let h_pt = upload_vec(client, nz((0..n12).map(gcol).collect()));
        let h_pb = upload_vec(client, nz((ct[0]..ct[0] + n23).map(gcol).collect()));
        let h_p4 = upload_vec(client, nz((k..n).map(gcol).collect()));
        self.counters.upload(4 * (n12 + n23 + (n - k)));
        self.merge_chunks(g.g0, n, g.n1, k, [(&h_pt, n12, &u, k, 0), (&h_pb, n23, &u, k, ct[0])], &h_p4);
        self.counters.span_end(Span::StedcMerge, span);
        Some(0)
    }

    fn lean(&self, k: usize) -> bool {
        k >= self.lean_min_k
    }

    unsafe fn back_lean(&mut self, g: crate::stedc::LeanMerge<'_>) {
        self.flush();
        let span = self.counters.span_start();
        let (g0, n, n1, k, ctot) = (g.g0, g.n, g.n1, g.k, g.ctot);
        let n12 = ctot[0] + ctot[1];
        let n23 = ctot[1] + ctot[2];
        let gcol = |p: usize| (g0 + g.types.at(p) - 1) as u32;
        let sec = |p: usize| (g.indx.at(p) - 1) as u32;
        let nz = |v: Vec<u32>| if v.is_empty() { vec![0] } else { v };
        let h_pt = upload_vec(self.client, nz((0..n12).map(gcol).collect()));
        let h_pb = upload_vec(self.client, nz((ctot[0]..ctot[0] + n23).map(gcol).collect()));
        let h_st = upload_vec(self.client, nz((0..n12).map(sec).collect()));
        let h_sb = upload_vec(self.client, nz((ctot[0]..ctot[0] + n23).map(sec).collect()));
        let h_p4 = upload_vec(self.client, nz((k..n).map(gcol).collect()));
        let h_zhat = upload(self.client, g.zhat);
        let h_dlam = upload(self.client, g.dlambda);
        let h_orig = upload(self.client, g.orig);
        let h_tau = upload(self.client, g.tau);
        let h_norm = upload(self.client, g.norm);
        self.counters.upload(4 * (2 * n12 + 2 * n23 + (n - k)) + k * (8 * 4 + 4));
        let top = ops::LeanU { sec: &h_st, zhat: &h_zhat, dlam: &h_dlam, orig: &h_orig, tau: &h_tau, norm: &h_norm };
        let bot = ops::LeanU { sec: &h_sb, ..top };
        let (top_lo, top_hi, bot_lo, bot_hi) = (g0, g0 + n1, g0 + n1, g0 + n);
        for (chunk, lo, hi) in self.chunks(g0, n) {
            let dst = self.client.empty(chunk.rvecs * self.lanes.width * n * 8);
            let (tl, th) = (lo.max(top_lo), hi.min(top_hi));
            if tl < th {
                ops::merge_rows_lean::<R, f64>(
                    self.client,
                    self.lanes,
                    &self.z,
                    self.lda,
                    self.ncols,
                    &dst,
                    chunk,
                    n,
                    k,
                    &h_pt,
                    n12,
                    &h_p4,
                    &top,
                    tl,
                    th,
                );
                self.counters.launch();
            }
            let (bl, bh) = (lo.max(bot_lo), hi.min(bot_hi));
            if bl < bh {
                ops::merge_rows_lean::<R, f64>(
                    self.client,
                    self.lanes,
                    &self.z,
                    self.lda,
                    self.ncols,
                    &dst,
                    chunk,
                    n,
                    k,
                    &h_pb,
                    n23,
                    &h_p4,
                    &bot,
                    bl,
                    bh,
                );
                self.counters.launch();
            }
            self.counters.sample(self.client);
            ops::copy_back::<R, f64>(
                self.client,
                self.lanes,
                &self.z,
                self.lda,
                self.ncols,
                &dst,
                chunk,
                g0,
                n,
                lo,
                hi,
            );
            self.counters.launch();
        }
        self.counters.span_end(Span::StedcMerge, span);
    }

    unsafe fn reorder(&mut self, g0: usize, n: usize, indxq: I, _q: D, _ldq: usize, _work: D) {
        self.flush();
        let perm: Vec<u32> = (0..n).map(|i| (g0 + indxq.at(i) - 1) as u32).collect();
        self.gather_block(g0, n, perm);
    }

    unsafe fn swap(&mut self, _n: usize, i: usize, k: usize, _z: D, _ldz: usize) {
        self.flush();
        ops::swap_cols::<R, f64>(self.client, self.lanes, &self.z, self.lda, self.ncols, i - 1, k - 1);
        self.counters.launch();
    }
}
