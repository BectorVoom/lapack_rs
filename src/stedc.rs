//! Divide-and-conquer tridiagonal eigensolver.
//!
//! Transcribed from Reference-LAPACK v3.12.1 `SRC/dstedc.f`, `SRC/dlaed0.f`
//! (`ICOMPQ = 2`), `SRC/dlaed1.f`, `SRC/dlaed2.f` and `SRC/dlaed3.f`.
//!
//! Parallelism never changes a value:
//! * `dlaed0` merges bottom-up level by level; every merge only reads its two
//!   children, so the tree runs as `rayon::join` over disjoint sub-blocks of
//!   `D`, `Q`, `INDXQ` and of the workspace (a node of size `m` owns
//!   `4m + m^2` reals and `4m` integers, which its children split).
//! * In `dlaed3`, each root of the secular equation, each row of the
//!   Gu–Eisenstat `ẑ` products and each eigenvector column is independent.
//!
//! The only observable difference is the scratch content of `WORK`/`IWORK`,
//! which LAPACK does not specify.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::aux::{EPS, SclType, dlacpy_all, dlamrg, dlanst_m, dlapy2, dlascl, dlaset_full, fmax};
use crate::blas::{dcopy, dswap};
use crate::flavor::{Blas, RefBlas};
use crate::laed4::dlaed4;
use std::marker::PhantomData;

/// Host time of `DLAED3`'s three sections (secular roots, `W` products,
/// eigenvector columns), accumulated only while [`DLAED3_TIMING`] is on
/// (the device profiler turns it on with its phase timers).
pub(crate) static DLAED3_NS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
pub(crate) static DLAED3_TIMING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn section_start() -> Option<std::time::Instant> {
    DLAED3_TIMING.load(std::sync::atomic::Ordering::Relaxed).then(std::time::Instant::now)
}

fn section_end(i: usize, t: Option<std::time::Instant>) {
    if let Some(t) = t {
        DLAED3_NS[i].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}
use crate::ptr::{D, I, ix};
use crate::steqr::{Compz, dsteqr_raw};

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;
const TWO: f64 = 2.0;
const EIGHT: f64 = 8.0;

/// `ILAENV( 9, 'DSTEDC', ... )`.
pub const SMLSIZ: usize = 25;

/// Branch counters for deflation fixtures (feature `stats`).
#[cfg(feature = "stats")]
pub mod stats {
    use std::sync::atomic::{AtomicUsize, Ordering};
    pub static MERGES: AtomicUsize = AtomicUsize::new(0);
    pub static FULL_DEFLATION: AtomicUsize = AtomicUsize::new(0);
    pub static SMALL_Z_DEFLATION: AtomicUsize = AtomicUsize::new(0);
    pub static ROTATION_DEFLATION: AtomicUsize = AtomicUsize::new(0);
    pub static K1: AtomicUsize = AtomicUsize::new(0);
    pub static K2: AtomicUsize = AtomicUsize::new(0);
    pub static SECULAR_ROOTS: AtomicUsize = AtomicUsize::new(0);
    pub static SPLITS: AtomicUsize = AtomicUsize::new(0);

    /// Snapshot of all counters.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct DcStats {
        pub merges: usize,
        pub full_deflation: usize,
        pub small_z_deflation: usize,
        pub rotation_deflation: usize,
        pub k1: usize,
        pub k2: usize,
        pub secular_roots: usize,
        pub splits: usize,
    }

    pub fn reset() {
        for c in [&MERGES, &FULL_DEFLATION, &SMALL_Z_DEFLATION, &ROTATION_DEFLATION, &K1, &K2, &SECULAR_ROOTS, &SPLITS]
        {
            c.store(0, Ordering::Relaxed);
        }
    }

    pub fn snapshot() -> DcStats {
        DcStats {
            merges: MERGES.load(Ordering::Relaxed),
            full_deflation: FULL_DEFLATION.load(Ordering::Relaxed),
            small_z_deflation: SMALL_Z_DEFLATION.load(Ordering::Relaxed),
            rotation_deflation: ROTATION_DEFLATION.load(Ordering::Relaxed),
            k1: K1.load(Ordering::Relaxed),
            k2: K2.load(Ordering::Relaxed),
            secular_roots: SECULAR_ROOTS.load(Ordering::Relaxed),
            splits: SPLITS.load(Ordering::Relaxed),
        }
    }

    #[inline]
    pub(crate) fn bump(c: &AtomicUsize, by: usize) {
        c.fetch_add(by, Ordering::Relaxed);
    }
}

macro_rules! stat {
    ($name:ident, $by:expr) => {
        #[cfg(feature = "stats")]
        stats::bump(&stats::$name, $by);
    };
}

/// `DSTEDC` minimum workspace for `COMPZ = 'I'`: `(LWMIN, LIWMIN)`.
pub fn stedc_work_len(n: usize) -> (usize, usize) {
    if n <= 1 {
        (1, 1)
    } else if n <= SMLSIZ {
        (2 * (n - 1), 1)
    } else {
        (1 + 4 * n + n * n, 3 + 5 * n)
    }
}

/// Inputs of an engine-side `DLAED3` (`K > 2`): the merge geometry, the
/// deflated `DLAMBDA` and z-vector `W` (both `K`), `RHO`, the row
/// permutation `INDX` (`K`, 1-based), `CTOT` and the column map `TYPES`
/// (the `INDX` argument of label 110). The roots go to `d` (`K`).

pub(crate) struct SecularMerge<'a> {
    pub g0: usize,
    pub n: usize,
    pub n1: usize,
    pub k: usize,
    pub rho: f64,
    pub dlambda: &'a [f64],
    pub w: &'a [f64],
    pub indx: &'a [usize],
    pub ctot: [usize; 4],
    pub types: &'a [usize],
    pub d: D,
}

/// Where `DSTEDC`'s eigenvector matrix lives. The host transcription below
/// drives every access to `Q`/`Z` through this trait: [`HostDc`] runs the
/// reference BLAS on host memory (and parallelises the tree), the device
/// engine (`device::stedc`) keeps `Z` on the device. Block coordinates are
/// 1-based and relative to the current block; `g0` is the block's 0-based
/// global offset on the diagonal of `Z`.
pub(crate) trait DcEngine: Send + Sized {
    /// The BLAS flavour of the host-side calls (`DNRM2`, `DSCAL`, `IDAMAX`, ...).
    type B: Blas;
    /// Two engines for concurrent subtrees, or `None` to serialise.
    fn split(&self) -> Option<(Self, Self)>;
    /// `DLASET( 'Full', N, N, ZERO, ONE, Z, LDZ )`.
    unsafe fn identity(&mut self, n: usize, z: D, ldz: usize);
    /// `DSTEQR( 'I', M, D, E, Z(G0+1,G0+1), LDZ, WORK )`; returns `INFO`.
    unsafe fn steqr(&mut self, g0: usize, m: usize, d: D, e: D, zblk: D, ldz: usize, work: D) -> usize;
    /// `OUT(Q) = Q(ROW, C0+Q)` for `Q < LEN` (the `DLAED1` row copies).
    unsafe fn read_row(&mut self, g0: usize, row: usize, c0: usize, len: usize, q: D, ldq: usize, out: D);
    /// `DROT( N, Q(1,PJ), 1, Q(1,NJ), 1, C, S )`.
    unsafe fn drot(&mut self, g0: usize, n: usize, pj: usize, nj: usize, c: f64, s: f64, q: D, ldq: usize);
    /// `DLAED2` full deflation: `Q(:,J) = Q(:,INDX(J))` via `Q2`.
    unsafe fn gather_all(&mut self, g0: usize, n: usize, indx: I, q: D, ldq: usize, q2: D);
    /// `DLAED2`'s `Q2` build and the copy of the deflated columns into
    /// `Q(1, K+1)`.
    unsafe fn build_q2(
        &mut self,
        g0: usize,
        n: usize,
        n1: usize,
        k: usize,
        ctot: [usize; 4],
        indx: I,
        q: D,
        ldq: usize,
        q2: D,
    );
    /// The `K × K` matrix `DLAED3` computes in (`Q(1..K,1..K)` on the host).
    unsafe fn uk(&mut self, k: usize, q: D, ldq: usize) -> (D, usize);
    /// `DLAED3` label 110: `Q(1..N, 1..K) = Q2 · S` for both row blocks.
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
        q: D,
        ldq: usize,
        q2: D,
        s: D,
    );
    /// `DLAED0` loop 110: `Q(:,I) = Q(:,INDXQ(I))` via `WORK(N+1:)`.
    unsafe fn reorder(&mut self, g0: usize, n: usize, indxq: I, q: D, ldq: usize, work: D);
    /// `DSWAP( N, Z(1,I), 1, Z(1,K), 1 )`.
    unsafe fn swap(&mut self, n: usize, i: usize, k: usize, z: D, ldz: usize);
    /// `DLAED3` for `K > 2` performed entirely by the engine (roots into
    /// `D(1..K)`, `W`, `U` and the label-110 back-transform), or `None` to
    /// run the host transcription. Returns `INFO`.
    unsafe fn secular(&mut self, _g: SecularMerge<'_>) -> Option<usize> {
        None
    }
    /// Whether `DLAED3` with this `K` uses the lean merge (never for the host
    /// reference engine; tolerance only).
    fn lean(&self, _k: usize) -> bool {
        false
    }
    /// Lean label 110: `Q(1..N, 1..K) = Q2 · U` with `U` never materialised,
    /// `U(l, j) = (ẑ(s) / ((Λ(s) − Λ(o_j)) − τ_j)) / ν_j`, `s = INDX(l)`, where
    /// `Λ` = `DLAMBDA`, `ẑ` the Gu–Eisenstat vector, `(o_j, τ_j)` the origin
    /// and offset of root `j` (0-based origin) and `ν_j` the column norm.
    unsafe fn back_lean(&mut self, _g: LeanMerge<'_>) {
        unreachable!("back_lean on an engine without a lean merge")
    }
}

/// Everything the lean `DLAED3` back-transform needs (all O(n)).

pub(crate) struct LeanMerge<'a> {
    pub g0: usize,
    pub n: usize,
    pub n1: usize,
    pub k: usize,
    pub ctot: [usize; 4],
    /// Type-sorted `INDX` (Q2 column order), 1-based.
    pub types: I,
    /// `DLAED3`'s row permutation `INDX`, 1-based.
    pub indx: I,
    pub dlambda: &'a [f64],
    pub zhat: &'a [f64],
    pub orig: &'a [u32],
    pub tau: &'a [f64],
    pub norm: &'a [f64],
}

/// The reference transcription on host memory.
pub(crate) struct HostDc<B>(pub(crate) PhantomData<B>);

impl<B> Clone for HostDc<B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B> Copy for HostDc<B> {}

impl<B: Blas> HostDc<B> {
    pub(crate) const fn new() -> Self {
        HostDc(PhantomData)
    }
}

impl<B: Blas> DcEngine for HostDc<B> {
    type B = B;
    fn split(&self) -> Option<(Self, Self)> {
        Some((HostDc::new(), HostDc::new()))
    }
    unsafe fn identity(&mut self, n: usize, z: D, ldz: usize) {
        dlaset_full(n, n, ZERO, ONE, z, ldz);
    }
    unsafe fn steqr(&mut self, _g0: usize, m: usize, d: D, e: D, zblk: D, ldz: usize, work: D) -> usize {
        dsteqr_raw(Compz::I, m, d, e, zblk, ldz, work)
    }
    unsafe fn read_row(&mut self, _g0: usize, row: usize, c0: usize, len: usize, q: D, ldq: usize, out: D) {
        dcopy(len, q.off(ix(row, c0, ldq)), ldq, out, 1);
    }
    unsafe fn drot(&mut self, _g0: usize, n: usize, pj: usize, nj: usize, c: f64, s: f64, q: D, ldq: usize) {
        B::drot(n, q.off(ix(1, pj, ldq)), 1, q.off(ix(1, nj, ldq)), 1, c, s);
    }
    unsafe fn gather_all(&mut self, _g0: usize, n: usize, indx: I, q: D, ldq: usize, q2: D) {
        let mut iq2 = 0usize;
        for j in 1..=n {
            let i = indx.at(j - 1);
            dcopy(n, q.off(ix(1, i, ldq)), 1, q2.off(iq2), 1);
            iq2 += n;
        }
        dlacpy_all(n, n, q2, n, q, ldq);
    }
    unsafe fn build_q2(
        &mut self,
        _g0: usize,
        n: usize,
        n1: usize,
        k: usize,
        ctot: [usize; 4],
        indx: I,
        q: D,
        ldq: usize,
        q2: D,
    ) {
        let n2 = n - n1;
        let mut i = 1usize;
        let mut iq1 = 0usize;
        let mut iq2 = (ctot[0] + ctot[1]) * n1;
        for _ in 0..ctot[0] {
            let js = indx.at(i - 1);
            dcopy(n1, q.off(ix(1, js, ldq)), 1, q2.off(iq1), 1);
            i += 1;
            iq1 += n1;
        }
        for _ in 0..ctot[1] {
            let js = indx.at(i - 1);
            dcopy(n1, q.off(ix(1, js, ldq)), 1, q2.off(iq1), 1);
            dcopy(n2, q.off(ix(n1 + 1, js, ldq)), 1, q2.off(iq2), 1);
            i += 1;
            iq1 += n1;
            iq2 += n2;
        }
        for _ in 0..ctot[2] {
            let js = indx.at(i - 1);
            dcopy(n2, q.off(ix(n1 + 1, js, ldq)), 1, q2.off(iq2), 1);
            i += 1;
            iq2 += n2;
        }
        let iq1 = iq2;
        for _ in 0..ctot[3] {
            let js = indx.at(i - 1);
            dcopy(n, q.off(ix(1, js, ldq)), 1, q2.off(iq2), 1);
            iq2 += n;
            i += 1;
        }
        if k < n {
            dlacpy_all(n, ctot[3], q2.off(iq1), n, q.off(ix(1, k + 1, ldq)), ldq);
        }
    }
    unsafe fn uk(&mut self, _k: usize, q: D, ldq: usize) -> (D, usize) {
        (q, ldq)
    }
    unsafe fn back(
        &mut self,
        _g0: usize,
        n: usize,
        n1: usize,
        k: usize,
        ctot: [usize; 4],
        _indx: I,
        _uk: D,
        _lduk: usize,
        q: D,
        ldq: usize,
        q2: D,
        s: D,
    ) {
        let n2 = n - n1;
        let n12 = ctot[0] + ctot[1];
        let n23 = ctot[1] + ctot[2];
        dlacpy_all(n23, k, q.off(ix(ctot[0] + 1, 1, ldq)), ldq, s, n23.max(1));
        let iq2 = n1 * n12;
        if n23 != 0 {
            B::dgemm(
                false,
                false,
                n2,
                k,
                n23,
                ONE,
                q2.off(iq2),
                n2.max(1),
                s,
                n23,
                ZERO,
                q.off(ix(n1 + 1, 1, ldq)),
                ldq,
            );
        } else {
            for j in 1..=k {
                for i in 1..=n2 {
                    q.set(ix(n1 + i, j, ldq), ZERO);
                }
            }
        }
        dlacpy_all(n12, k, q, ldq, s, n12.max(1));
        if n12 != 0 {
            B::dgemm(false, false, n1, k, n12, ONE, q2, n1.max(1), s, n12, ZERO, q, ldq);
        } else {
            for j in 1..=k {
                for i in 1..=n1 {
                    q.set(ix(i, j, ldq), ZERO);
                }
            }
        }
    }
    unsafe fn reorder(&mut self, _g0: usize, n: usize, indxq: I, q: D, ldq: usize, work: D) {
        for i in 1..=n {
            let j = indxq.at(i - 1);
            dcopy(n, q.off(ix(1, j, ldq)), 1, work.off(n * i), 1);
        }
        dlacpy_all(n, n, work.off(n), n, q, ldq);
    }
    unsafe fn swap(&mut self, n: usize, i: usize, k: usize, z: D, ldz: usize) {
        dswap(n, z.off(ix(1, i, ldz)), 1, z.off(ix(1, k, ldz)), 1);
    }
}

/// `DLAED3`. `indxp` is integer scratch of length `k` (the caller's unused
/// `INDXP` block); it replaces the reference's `S` scratch in the column loop.
unsafe fn dlaed3<E: DcEngine>(
    eng: &mut E,
    g0: usize,
    k: usize,
    n: usize,
    n1: usize,
    d: D,
    q: D,
    ldq: usize,
    rho: f64,
    dlambda: D,
    q2: D,
    indx: I,
    ctot: I,
    w: D,
    s: D,
    iscratch: I,
    types: I,
) -> usize {
    if k == 0 {
        return 0;
    }
    if k > 2 && eng.lean(k) {
        return dlaed3_lean(eng, g0, k, n, n1, d, rho, dlambda, indx, ctot, w, types);
    }
    stat!(SECULAR_ROOTS, k);
    if k > 2 {
        let g = SecularMerge {
            g0,
            n,
            n1,
            k,
            rho,
            dlambda: dlambda.slice(k),
            w: w.slice(k),
            indx: indx.slice(k),
            ctot: [ctot.at(0), ctot.at(1), ctot.at(2), ctot.at(3)],
            types: types.slice(n),
            d,
        };
        if let Some(info) = eng.secular(g) {
            return info;
        }
    }
    let (uk, lduk) = eng.uk(k, q, ldq);
    // Roots of the secular equation, one column of the K x K matrix each.
    let fail = std::sync::atomic::AtomicUsize::new(0);
    let t = section_start();
    {
        let (lo, hi) = (0usize, k);

        for j in (lo + 1)..=hi {
            let (dlam, info) = dlaed4(k, j, dlambda, w, uk.off(ix(1, j, lduk)), rho);
            d.set(j - 1, dlam);
            if info != 0 {
                fail.store(info, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    section_end(0, t);
    let info = fail.load(std::sync::atomic::Ordering::Relaxed);
    if info != 0 {
        return info;
    }

    if k == 2 {
        for j in 1..=k {
            w.set(0, uk.at(ix(1, j, lduk)));
            w.set(1, uk.at(ix(2, j, lduk)));
            let ii = indx.at(0);
            uk.set(ix(1, j, lduk), w.at(ii - 1));
            let ii = indx.at(1);
            uk.set(ix(2, j, lduk), w.at(ii - 1));
        }
    } else if k > 2 {
        let t = section_start();
        dcopy(k, w, 1, s, 1);
        dcopy(k, uk, lduk + 1, w, 1);
        // W(I) = W(I)*( Q(I,J) / ( DLAMBDA(I)-DLAMBDA(J) ) ), J ascending, J /= I:
        // an independent chain per row I.
        {
            let (lo, hi) = (0usize, k);

            for i in (lo + 1)..=hi {
                let mut wi = w.at(i - 1);
                let li = dlambda.at(i - 1);
                for j in 1..=k {
                    if j != i {
                        wi *= uk.at(ix(i, j, lduk)) / (li - dlambda.at(j - 1));
                    }
                }
                w.set(i - 1, wi);
            }
        }
        for i in 1..=k {
            w.set(i - 1, (-w.at(i - 1)).sqrt().copysign(s.at(i - 1)));
        }
        section_end(1, t);
        let t = section_start();
        // Cycle leaders of the row permutation INDX (0 unvisited, 1 visited,
        // 2 leader), so each column permutes in place without the S scratch.
        for i in 0..k {
            iscratch.set(i, 0);
        }
        for i in 1..=k {
            if iscratch.at(i - 1) == 0 {
                iscratch.set(i - 1, 2);
                let mut cur = indx.at(i - 1);
                while cur != i {
                    iscratch.set(cur - 1, 1);
                    cur = indx.at(cur - 1);
                }
            }
        }
        // Column J: S(I) = W(I)/Q(I,J); TEMP = DNRM2(K,S); Q(I,J) = S(INDX(I))/TEMP.
        {
            let (lo, hi) = (0usize, k);

            for j in (lo + 1)..=hi {
                let qj = uk.off(ix(1, j, lduk));
                for i in 0..k {
                    qj.set(i, w.at(i) / qj.at(i));
                }
                let temp = E::B::dnrm2(k, qj, 1);
                for i in 0..k {
                    qj.set(i, qj.at(i) / temp);
                }
                for l in 1..=k {
                    if iscratch.at(l - 1) != 2 {
                        continue;
                    }
                    let tmp = qj.at(l - 1);
                    let mut cur = l;
                    loop {
                        let nxt = indx.at(cur - 1);
                        if nxt == l {
                            qj.set(cur - 1, tmp);
                            break;
                        }
                        qj.set(cur - 1, qj.at(nxt - 1));
                        cur = nxt;
                    }
                }
            }
        }
        section_end(2, t);
    }

    // 110: back-transform with the deflated eigenvectors.
    let ct = [ctot.at(0), ctot.at(1), ctot.at(2), ctot.at(3)];
    eng.back(g0, n, n1, k, ct, types, uk, lduk, q, ldq, q2, s);
    0
}

/// Lean `DLAED3` (`K > 2`): the same roots `D` (bit-identical), but the
/// eigenvector matrix is never stored. Each root's origin and offset are
/// recovered exactly from its own `DELTA` (`DELTA(origin) = −TAU` through all
/// of `DLAED4`'s updates), and every later use of `DELTA` is recomputed in
/// the accurate form `(Λ(s) − Λ(o)) − τ`. Host memory O(K) per thread.
#[allow(clippy::too_many_arguments)]
unsafe fn dlaed3_lean<E: DcEngine>(
    eng: &mut E,
    g0: usize,
    k: usize,
    n: usize,
    n1: usize,
    d: D,
    rho: f64,
    dlambda: D,
    indx: I,
    ctot: I,
    w: D,
    types: I,
) -> usize {
    let mut orig = vec![0u32; k];
    let mut tau = vec![0.0f64; k];
    let fail = std::sync::atomic::AtomicUsize::new(0);
    {
        let (ov, tv) = (crate::ptr::V::new(&mut orig[..]), crate::ptr::V::new(&mut tau[..]));
        {
            let (lo, hi) = (0usize, k);

            let mut delta = vec![0.0f64; k];
            let dv = crate::ptr::V::new(&mut delta[..]);
            for j in (lo + 1)..=hi {
                let (dlam, info) = dlaed4(k, j, dlambda, w, dv, rho);
                d.set(j - 1, dlam);
                if info != 0 {
                    fail.store(info, std::sync::atomic::Ordering::Relaxed);
                }
                let o = if j == k || dv.at(j - 1).abs() <= dv.at(j).abs() { j - 1 } else { j };
                ov.set(j - 1, o as u32);
                tv.set(j - 1, -dv.at(o));
            }
        }
    }
    let info = fail.load(std::sync::atomic::Ordering::Relaxed);
    if info != 0 {
        return info;
    }
    stat!(SECULAR_ROOTS, k);
    let lam: Vec<f64> = (0..k).map(|i| dlambda.at(i)).collect();
    let delta = |s: usize, j: usize| (lam[s] - lam[orig[j] as usize]) - tau[j];
    // Gu–Eisenstat: ẑ(i) = sign(z(i)) · sqrt(−Π_j δ_i(j)/(Λ(i)−Λ(j)) · δ_i(i)).
    let mut zhat = vec![0.0f64; k];
    {
        let zv = crate::ptr::V::new(&mut zhat[..]);
        {
            let (lo, hi) = (0usize, k);

            for i in lo..hi {
                let mut wi = delta(i, i);
                for j in 0..k {
                    if j != i {
                        wi *= delta(i, j) / (lam[i] - lam[j]);
                    }
                }
                zv.set(i, (-wi).sqrt().copysign(w.at(i)));
            }
        }
    }
    let mut norm = vec![0.0f64; k];
    {
        let nv = crate::ptr::V::new(&mut norm[..]);
        {
            let (lo, hi) = (0usize, k);

            let mut col = vec![0.0f64; k];
            for j in lo..hi {
                for (i, c) in col.iter_mut().enumerate() {
                    *c = zhat[i] / delta(i, j);
                }
                nv.set(j, E::B::dnrm2(k, crate::ptr::V::new(&mut col[..]), 1));
            }
        }
    }
    let ct = [ctot.at(0), ctot.at(1), ctot.at(2), ctot.at(3)];
    eng.back_lean(LeanMerge {
        g0,
        n,
        n1,
        k,
        ctot: ct,
        types,
        indx,
        dlambda: &lam,
        zhat: &zhat,
        orig: &orig,
        tau: &tau,
        norm: &norm,
    });
    0
}

/// `DLAED2`. Returns `K`.
unsafe fn dlaed2<E: DcEngine>(
    eng: &mut E,
    g0: usize,
    n: usize,
    n1: usize,
    d: D,
    q: D,
    ldq: usize,
    indxq: I,
    rho: &mut f64,
    z: D,
    dlambda: D,
    w: D,
    q2: D,
    indx: I,
    indxc: I,
    indxp: I,
    coltyp: I,
) -> usize {
    if n == 0 {
        return 0;
    }
    let n2 = n - n1;
    let n1p1 = n1 + 1;
    if *rho < ZERO {
        E::B::dscal(n2, -ONE, z.off(n1p1 - 1), 1);
    }
    let t = ONE / TWO.sqrt();
    E::B::dscal(n, t, z, 1);
    *rho = (TWO * *rho).abs();
    for i in n1p1..=n {
        indxq.set(i - 1, indxq.at(i - 1) + n1);
    }
    for i in 1..=n {
        dlambda.set(i - 1, d.at(indxq.at(i - 1) - 1));
    }
    dlamrg(n1, n2, dlambda, 1, 1, indxc);
    for i in 1..=n {
        indx.set(i - 1, indxq.at(indxc.at(i - 1) - 1));
    }
    let imax = E::B::idamax(n, z, 1);
    let jmax = E::B::idamax(n, d, 1);
    let eps = EPS;
    let tol = EIGHT * eps * fmax(d.at(jmax - 1).abs(), z.at(imax - 1).abs());
    let rho_v = *rho;

    if rho_v * z.at(imax - 1).abs() <= tol {
        stat!(FULL_DEFLATION, 1);
        for j in 1..=n {
            let i = indx.at(j - 1);
            dlambda.set(j - 1, d.at(i - 1));
        }
        eng.gather_all(g0, n, indx, q, ldq, q2);
        dcopy(n, dlambda, 1, d, 1);
        return 0;
    }

    for i in 1..=n1 {
        coltyp.set(i - 1, 1);
    }
    for i in n1p1..=n {
        coltyp.set(i - 1, 3);
    }

    let mut k = 0usize;
    let mut k2 = n + 1;
    let mut pj = 0usize;
    let mut j = 1usize;
    // Loop 70: find the first non-deflated entry.
    let mut all_done = false;
    while j <= n {
        let nj = indx.at(j - 1);
        if rho_v * z.at(nj - 1).abs() <= tol {
            stat!(SMALL_Z_DEFLATION, 1);
            k2 -= 1;
            coltyp.set(nj - 1, 4);
            indxp.set(k2 - 1, nj);
            if j == n {
                all_done = true;
                break;
            }
        } else {
            pj = nj;
            break;
        }
        j += 1;
    }
    if !all_done {
        // Loop 80.
        loop {
            j += 1;
            if j > n {
                break;
            }
            let nj = indx.at(j - 1);
            if rho_v * z.at(nj - 1).abs() <= tol {
                stat!(SMALL_Z_DEFLATION, 1);
                k2 -= 1;
                coltyp.set(nj - 1, 4);
                indxp.set(k2 - 1, nj);
            } else {
                let mut s = z.at(pj - 1);
                let mut c = z.at(nj - 1);
                let tau = dlapy2(c, s);
                let t = d.at(nj - 1) - d.at(pj - 1);
                c /= tau;
                s = -s / tau;
                if (t * c * s).abs() <= tol {
                    stat!(ROTATION_DEFLATION, 1);
                    z.set(nj - 1, tau);
                    z.set(pj - 1, ZERO);
                    if coltyp.at(nj - 1) != coltyp.at(pj - 1) {
                        coltyp.set(nj - 1, 2);
                    }
                    coltyp.set(pj - 1, 4);
                    eng.drot(g0, n, pj, nj, c, s, q, ldq);
                    let t = d.at(pj - 1) * (c * c) + d.at(nj - 1) * (s * s);
                    d.set(nj - 1, d.at(pj - 1) * (s * s) + d.at(nj - 1) * (c * c));
                    d.set(pj - 1, t);
                    k2 -= 1;
                    let mut i = 1usize;
                    loop {
                        if k2 + i <= n {
                            if d.at(pj - 1) < d.at(indxp.at(k2 + i - 1) - 1) {
                                indxp.set(k2 + i - 2, indxp.at(k2 + i - 1));
                                indxp.set(k2 + i - 1, pj);
                                i += 1;
                                continue;
                            } else {
                                indxp.set(k2 + i - 2, pj);
                            }
                        } else {
                            indxp.set(k2 + i - 2, pj);
                        }
                        break;
                    }
                    pj = nj;
                } else {
                    k += 1;
                    dlambda.set(k - 1, d.at(pj - 1));
                    w.set(k - 1, z.at(pj - 1));
                    indxp.set(k - 1, pj);
                    pj = nj;
                }
            }
        }
    }
    // 100: record the last eigenvalue.
    k += 1;
    dlambda.set(k - 1, d.at(pj - 1));
    w.set(k - 1, z.at(pj - 1));
    indxp.set(k - 1, pj);

    let mut ctot = [0usize; 4];
    for j in 1..=n {
        let ct = coltyp.at(j - 1);
        ctot[ct - 1] += 1;
    }
    let mut psm = [0usize; 4];
    psm[0] = 1;
    psm[1] = 1 + ctot[0];
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];
    let k = n - ctot[3];

    for j in 1..=n {
        let js = indxp.at(j - 1);
        let ct = coltyp.at(js - 1);
        indx.set(psm[ct - 1] - 1, js);
        indxc.set(psm[ct - 1] - 1, j);
        psm[ct - 1] += 1;
    }

    // Z(I) = D(INDX(I)) in type order (the reference does this inside the
    // Q2 copy loops; it only reads D).
    for i in 1..=n {
        z.set(i - 1, d.at(indx.at(i - 1) - 1));
    }
    eng.build_q2(g0, n, n1, k, ctot, indx, q, ldq, q2);
    if k < n {
        dcopy(n - k, z.off(k), 1, d.off(k), 1);
    }
    for j in 0..4 {
        coltyp.set(j, ctot[j]);
    }
    k
}

/// `DLAED1`. `work` holds `4n + n^2` reals, `iwork` `4n` integers. Returns
/// `INFO`.
unsafe fn dlaed1<E: DcEngine>(
    eng: &mut E,
    g0: usize,
    n: usize,
    d: D,
    q: D,
    ldq: usize,
    indxq: I,
    rho: &mut f64,
    cutpnt: usize,
    work: D,
    iwork: I,
) -> usize {
    if n == 0 {
        return 0;
    }
    stat!(MERGES, 1);
    let iz = 0usize;
    let idlmda = iz + n;
    let iw = idlmda + n;
    let iq2 = iw + n;
    let indx = 0usize;
    let indxc = indx + n;
    let coltyp = indxc + n;
    let indxp = coltyp + n;

    eng.read_row(g0, cutpnt, 1, cutpnt, q, ldq, work.off(iz));
    let zpp1 = cutpnt + 1;
    eng.read_row(g0, zpp1, zpp1, n - cutpnt, q, ldq, work.off(iz + cutpnt));

    let k = dlaed2(
        eng,
        g0,
        n,
        cutpnt,
        d,
        q,
        ldq,
        indxq,
        rho,
        work.off(iz),
        work.off(idlmda),
        work.off(iw),
        work.off(iq2),
        iwork.off(indx),
        iwork.off(indxc),
        iwork.off(indxp),
        iwork.off(coltyp),
    );

    if k != 0 {
        if k == 1 {
            stat!(K1, 1);
        }
        if k == 2 {
            stat!(K2, 1);
        }
        let c = |i: usize| iwork.at(coltyp + i);
        let is = (c(0) + c(1)) * cutpnt + (c(1) + c(2)) * (n - cutpnt) + iq2;
        let info = dlaed3(
            eng,
            g0,
            k,
            n,
            cutpnt,
            d,
            q,
            ldq,
            *rho,
            work.off(idlmda),
            work.off(iq2),
            iwork.off(indxc),
            iwork.off(coltyp),
            work.off(iw),
            work.off(is),
            iwork.off(indxp),
            iwork.off(indx),
        );
        if info != 0 {
            return info;
        }
        dlamrg(k, n - k, d, 1, -1, indxq);
    } else {
        for i in 1..=n {
            indxq.set(i - 1, i);
        }
    }
    0
}

/// Failure location in the merge tree: `(level, submat, matsiz)` where
/// leaves are level 0. The reference processes leaves first, then merges level
/// by level left to right, and reports the first failure in that order.
type Fail = Option<(usize, usize, usize)>;

fn first_fail(a: Fail, b: Fail) -> Fail {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(if (x.0, x.1) <= (y.0, y.1) { x } else { y }),
    }
}

/// Recursive `DLAED0` tree node covering 1-based `submat..submat+m-1` of the
/// block at global offset `gblk`. `depth` is the number of halvings left.
unsafe fn dlaed0_node<E: DcEngine>(
    eng: &mut E,
    gblk: usize,
    submat: usize,
    m: usize,
    depth: usize,
    d: D,
    e: D,
    q: D,
    ldq: usize,
    indxq: I,
    work: D,
    iwork: I,
) -> Fail {
    let g0 = gblk + submat - 1;
    if depth == 0 {
        // Leaf: DSTEQR('I') on the sub-block, then INDXQ = 1..m.
        let info = eng.steqr(g0, m, d.off(submat - 1), e.off(submat - 1), q.off(ix(submat, submat, ldq)), ldq, work);
        for j in 1..=m {
            indxq.set(submat + j - 2, j);
        }
        return if info != 0 { Some((0, submat, m)) } else { None };
    }
    let m1 = m / 2;
    let m2 = m - m1;
    let w1 = 4 * m1 + m1 * m1;
    let w2 = 4 * m2 + m2 * m2;
    let _ = w2;
    let (fa, fb) = if let Some((mut ea, mut eb)) = eng.split() {
        // The two halves are disjoint sub-blocks, so their order is free;
        // they now run one after the other (the rayon `join` is gone).
        let fa: Fail = dlaed0_node(&mut ea, gblk, submat, m1, depth - 1, d, e, q, ldq, indxq, work, iwork);
        let fb: Fail = dlaed0_node(
            &mut eb,
            gblk,
            submat + m1,
            m2,
            depth - 1,
            d,
            e,
            q,
            ldq,
            indxq,
            work.off(w1),
            iwork.off(4 * m1),
        );
        (fa, fb)
    } else {
        let fa = dlaed0_node(eng, gblk, submat, m1, depth - 1, d, e, q, ldq, indxq, work, iwork);
        let fb =
            dlaed0_node(eng, gblk, submat + m1, m2, depth - 1, d, e, q, ldq, indxq, work.off(w1), iwork.off(4 * m1));
        (fa, fb)
    };
    let child = first_fail(fa, fb);
    // A failed subtree leaves garbage; the reference stops before merging any
    // node above the failure, and only the reported INFO matters.
    if child.is_some() {
        return child;
    }
    let mut rho = e.at(submat + m1 - 2);
    let info = dlaed1(
        eng,
        g0,
        m,
        d.off(submat - 1),
        q.off(ix(submat, submat, ldq)),
        ldq,
        indxq.off(submat - 1),
        &mut rho,
        m1,
        work,
        iwork,
    );
    e.set(submat + m1 - 2, rho);
    if info != 0 { Some((depth, submat, m)) } else { None }
}

/// `DLAED0( 2, QSIZ, N, D, E, Q, LDQ, QSTORE, LDQS, WORK, IWORK, INFO )`.
/// `work` holds `n + n^2` (final copy) and at least `4n + n^2` reals;
/// `iwork` at least `5n` integers. Returns `INFO`.
unsafe fn dlaed0<E: DcEngine>(
    eng: &mut E,
    gblk: usize,
    n: usize,
    d: D,
    e: D,
    q: D,
    ldq: usize,
    work: D,
    iwork: I,
) -> usize {
    if n == 0 {
        return 0;
    }
    // Number of halvings: the rightmost (largest) leaf must be <= SMLSIZ.
    let mut depth = 0usize;
    let mut last = n;
    while last > SMLSIZ {
        last = last.div_ceil(2);
        depth += 1;
    }
    // Loop 40: D(SMM1) and D(SUBMAT) lose |E(SMM1)| at every internal split,
    // in the reference's left-to-right order over leaf boundaries.
    split_adjust(1, n, depth, d, e);

    // IWORK layout: 1..4n merge scratch, 4n+1..5n INDXQ.
    let indxq = iwork.off(4 * n);
    if let Some((_level, submat, matsiz)) = dlaed0_node(eng, gblk, 1, n, depth, d, e, q, ldq, indxq, work, iwork) {
        return submat * (n + 1) + submat + matsiz - 1;
    }

    // Loop 110: re-order eigenvalues and vectors by INDXQ.
    for i in 1..=n {
        let j = indxq.at(i - 1);
        work.set(i - 1, d.at(j - 1));
    }
    eng.reorder(gblk, n, indxq, q, ldq, work);
    dcopy(n, work, 1, d, 1);
    0
}

/// Apply the reference's loop 40 (`D(SMM1) -= |E(SMM1)|`, `D(SUBMAT) -= |E(SMM1)|`)
/// for every leaf boundary, left to right.
unsafe fn split_adjust(start: usize, m: usize, depth: usize, d: D, e: D) {
    if depth == 0 {
        return;
    }
    let m1 = m / 2;
    split_adjust(start, m1, depth - 1, d, e);
    // Boundary between the left subtree's last leaf and the right subtree's
    // first leaf: SUBMAT = start + m1, SMM1 = SUBMAT - 1.
    let submat = start + m1;
    let smm1 = submat - 1;
    let ae = e.at(smm1 - 1).abs();
    d.set(smm1 - 1, d.at(smm1 - 1) - ae);
    d.set(submat - 1, d.at(submat - 1) - ae);
    stat!(SPLITS, 1);
    split_adjust(submat, m - m1, depth - 1, d, e);
}

/// `DSTEDC( 'I', ... )` on raw views. `work` has `lwork` reals, `iwork`
/// `liwork` integers (both at least [`stedc_work_len`]). Returns `INFO`.
pub(crate) unsafe fn dstedc_i(n: usize, d: D, e: D, z: D, ldz: usize, work: D, iwork: I) -> usize {
    dstedc_i_with(&mut HostDc::<RefBlas>::new(), n, d, e, z, ldz, work, iwork)
}

/// `DSTEDC( 'I', ... )` with the eigenvector matrix behind `eng`.
pub(crate) unsafe fn dstedc_i_with<E: DcEngine>(
    eng: &mut E,
    n: usize,
    d: D,
    e: D,
    z: D,
    ldz: usize,
    work: D,
    iwork: I,
) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        eng.identity(1, z, ldz);
        return 0;
    }
    if n <= SMLSIZ {
        return eng.steqr(0, n, d, e, z, ldz, work);
    }
    eng.identity(n, z, ldz);
    let orgnrm = dlanst_m(n, d, e);
    if orgnrm == ZERO {
        return 0;
    }
    let eps = EPS;
    let mut start = 1usize;
    while start <= n {
        let mut finish = start;
        while finish < n {
            let tiny = eps * d.at(finish - 1).abs().sqrt() * d.at(finish).abs().sqrt();
            if e.at(finish - 1).abs() > tiny {
                finish += 1;
            } else {
                break;
            }
        }
        let m = finish - start + 1;
        if m == 1 {
            start = finish + 1;
            continue;
        }
        if m > SMLSIZ {
            let orgnrm = dlanst_m(m, d.off(start - 1), e.off(start - 1));
            dlascl(SclType::G, orgnrm, ONE, m, 1, d.off(start - 1), m);
            dlascl(SclType::G, orgnrm, ONE, m - 1, 1, e.off(start - 1), m - 1);
            let info = dlaed0(
                eng,
                start - 1,
                m,
                d.off(start - 1),
                e.off(start - 1),
                z.off(ix(start, start, ldz)),
                ldz,
                work,
                iwork,
            );
            if info != 0 {
                return (info / (m + 1) + start - 1) * (n + 1) + info % (m + 1) + start - 1;
            }
            dlascl(SclType::G, ONE, orgnrm, m, 1, d.off(start - 1), m);
        } else {
            let info =
                eng.steqr(start - 1, m, d.off(start - 1), e.off(start - 1), z.off(ix(start, start, ldz)), ldz, work);
            if info != 0 {
                return start * (n + 1) + finish;
            }
        }
        start = finish + 1;
    }
    for ii in 2..=n {
        let i = ii - 1;
        let mut k = i;
        let mut p = d.at(i - 1);
        for j in ii..=n {
            if d.at(j - 1) < p {
                k = j;
                p = d.at(j - 1);
            }
        }
        if k != i {
            d.set(k - 1, d.at(i - 1));
            d.set(i - 1, p);
            eng.swap(n, i, k, z, ldz);
        }
    }
    0
}
