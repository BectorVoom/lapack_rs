//! Hybrid `DSYTRD`: the trailing matrix lives on the device; `DLATRD`'s
//! Householder generation and `DGEMV` corrections, and the `DSYTD2` tail,
//! run on the host.
//!
//! Per panel: one sub-range read of the panel columns (their final LAPACK home
//! is the caller's `a`, so reflectors are never uploaded back). Per column:
//! one upload of `v` (always `lda` reals so the pool recycles one page), one
//! K1 launch, one sub-range read of `w`. Per panel update: one upload of the
//! packed `[V|W]` (`2·NB·lda` reals, rows aligned with `M`) and one K2 launch
//! that updates both triangles. The host transcription is shared with the
//! host path through [`crate::sytrd::TrdEngine`], so the operation order is
//! the reference's by construction.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::flavor::DeviceBlas;
use super::ops::Lanes;
use super::workspace::{Counters, Span, read_range, upload_vec};
use crate::flavor::RefBlas;
use crate::ptr::{D, ix};
use crate::sytrd::TrdEngine;
use std::marker::PhantomData;

/// Device state for one tridiagonalisation.
pub(crate) struct DeviceTrd<'a, R: Runtime, B: DeviceBlas = RefBlas> {
    pub client: &'a ComputeClient<R>,
    pub lanes: &'a Lanes,
    pub counters: &'a Counters,
    /// `lda × n` fully symmetric matrix.
    pub m: Handle,
    pub lda: usize,
    pub n: usize,
    /// K1 output (at least `lda` reals).
    pub y: Handle,
    /// Logical length of `y` in reals.
    pub y_len: usize,
    /// Host scratch reused for reads.
    pub buf: Vec<f64>,
    /// The BLAS flavour of the launched kernels and the host-side calls.
    pub flavor: PhantomData<B>,
}

impl<'a, R: Runtime, B: DeviceBlas> DeviceTrd<'a, R, B> {
    /// Read `M[first..first+len]` (column-major flat range) into `self.buf`.
    fn read_flat(&mut self, first: usize, len: usize) {
        if self.buf.len() < len {
            self.buf.resize(len, 0.0);
        }
        read_range(self.client, &self.m, self.lda * self.n, first, len, &mut self.buf[..len]);
        self.counters.read(len * 8);
    }

    /// Copy the `upper`/lower triangle of columns `c0..=c1` (1-based), rows
    /// `r0..=r1`, from `self.buf` (starting at flat index `first`) into `a`.
    unsafe fn copy_tri(&self, upper: bool, c0: usize, c1: usize, r0: usize, r1: usize, first: usize, a: D, lda: usize) {
        for c in c0..=c1 {
            for r in r0..=r1 {
                if (upper && r <= c) || (!upper && r >= c) {
                    a.set(ix(r, c, lda), self.buf[(c - 1) * self.lda + (r - 1) - first]);
                }
            }
        }
    }
}

impl<'a, R: Runtime, B: DeviceBlas> TrdEngine for DeviceTrd<'a, R, B> {
    type B = B;
    unsafe fn panel(&mut self, upper: bool, n: usize, i: usize, nb: usize, a: D, lda: usize) {
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

    unsafe fn symv(&mut self, upper: bool, m: usize, alpha: f64, g0: usize, _blk: D, _lda: usize, x: D, y: D) {
        if m == 0 {
            return;
        }
        let mut xs = vec![0.0f64; self.lda];
        for q in 0..m {
            xs[q] = x.at(q);
        }
        self.counters.upload(self.lda * 8);
        let hx = upload_vec(self.client, xs);
        B::symv::<R>(self.client, self.lanes, &self.m, self.lda, self.n, g0, m, &hx, &self.y, alpha, upper);
        self.counters.launch();
        let ys = y.slice_mut(m);
        read_range(self.client, &self.y, self.y_len, g0, m, ys);
        self.counters.read(m * 8);
    }

    unsafe fn syr2k(&mut self, upper: bool, m: usize, nb: usize, g0: usize, v: D, lda: usize, w: D, ldw: usize, _c: D) {
        if m == 0 {
            return;
        }
        let span = self.counters.span_start();
        let ld = self.lda;
        let mut packed = vec![0.0f64; 2 * nb * ld];
        for l in 0..nb {
            for r in 0..m {
                packed[g0 + r + l * ld] = v.at(ix(r + 1, l + 1, lda));
                packed[g0 + r + (nb + l) * ld] = w.at(ix(r + 1, l + 1, ldw));
            }
        }
        self.counters.upload(packed.len() * 8);
        let hvw = upload_vec(self.client, packed);
        B::syr2k::<R>(self.client, self.lanes, &self.m, ld, self.n, g0, m, &hvw, nb, -1.0, upper);
        self.counters.launch();
        self.counters.sample(self.client);
        self.counters.span_end(Span::TrdUpdate, span);
    }

    unsafe fn tail(&mut self, upper: bool, _n: usize, g0: usize, mt: usize, a: D, lda: usize) {
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

/// Stage a host `n × n` matrix (valid in its `upper`/lower triangle) as a
/// fully symmetric `lda × n` device image in ONE copy loop.
pub(crate) fn stage_symmetric(a: &[f64], n: usize, lda: usize, upper: bool) -> Vec<f64> {
    let mut s = vec![0.0f64; lda * n];
    for j in 0..n {
        for i in 0..n {
            let (r, c) = if (upper && i <= j) || (!upper && i >= j) { (i, j) } else { (j, i) };
            s[i + j * lda] = a[r + c * n];
        }
    }
    s
}
