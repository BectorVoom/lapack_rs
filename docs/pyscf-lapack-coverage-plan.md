# Implementation plan: LAPACK/BLAS coverage for pyscf_rs consumers

## 1. Background

`pyscf_rs` reproduces a slice of LAPACK/BLAS on the host today, mostly through
faer 0.24 (see the inventory in §3). The standard symmetric/Hermitian
eigendecomposition paths are already migrated to this crate:

- `pyscf-algebra/src/lapack_backend.rs` (`sym_eigh` → `dsyevd_alloc`,
  `hermitian_eigh` → `zheevd_alloc`), consumed by
  `host_fallback::eigh`, `df_metric_fit`, and the two inner standard
  decompositions of the Löwdin generalized solvers (`eigh_gen`,
  `zeigh::zeigh_gen_faer`).

Everything in §3 that this crate does not yet implement still runs on faer
(or on hand-written loops). This plan lists that missing functionality as
transcription tasks following this repo's standing rules (AGENTS.md §5):
transcribe from the pinned Reference-LAPACK 3.12.1 sources under
`target/ref-lapack/lapack-3.12.1`, keep every output bit-identical to the
reference built by `cargo xtask build-ref-lapack`, expose LAPACK workspace
semantics (`*_work_len` / `*_opt_work_len` / `*_alloc`, caller-supplied
workspace, drivers allocate nothing), use column-major storage with `LDA = n`
and planar complex storage, and gate each routine with `ref_*_bitexact`
oracle tests plus the established `cargo test` gates.

## 2. Already covered (no work)

| Need | lapack_rs surface |
|---|---|
| Real symmetric standard eig (`dsyev`/`dsyevd` role) | `dsyevd`, `dsyevd_alloc`, `blocks::{dsytrd, dormtr_ln, dstedc, dsterf}` |
| Complex Hermitian standard eig (`zheevd` role) | `zheevd`, `zheevd_alloc` |
| Real `dgemm` (host reference) | `blocks::dgemm` |
| Level-1/2 real+complex BLAS used inside the D&C tree | `blas::*`, `zblas::*` (`pub(crate)`) |

## 3. pyscf_rs needs vs. status

| # | Need (pyscf_rs site) | LAPACK equivalent | Status |
|---|---|---|---|
| 1 | `host_fallback::cholesky`, `zeigh::zcholesky_faer`, `zcholesky_crout`, DF metric Cholesky (`pyscf-df/cholesky_eri.rs`), `partial_cholesky_orth_` | `DPOTRF` / `ZPOTRF` | MISSING — Task A |
| 2 | `solve_linear` (DIIS B-matrix), `zeigh::zsolve_linear_faer`, `cho_solve`, CPHF solves | `DGETRF`+`DGETRS` / `DGESV`, complex `ZGESV` | MISSING — Task B |
| 3 | `host_fallback::qr` | `DGEQRF` + `DORGQR` | MISSING — Task C |
| 4 | `host_fallback::svd` | `DGESVD` (or `DGESDD`) | MISSING — Task D |
| 5 | Löwdin glue in `eigh_gen` / `zeigh_gen_faer` (`X = U·diag(s⁻¹ᐟ²)`, `F' = XᵀFX`, `C = XV`) | `DSYGV` / `ZHEGV` (type 1, all pairs, with LN-range truncation kept in the caller) | MISSING — Task E |
| 6 | `davidson::eig_general` / `zeig_general`, `pbc-adc/roots.rs`, `pbc-symm/group.rs` character table | `DGEEV` / `ZGEEV` (values + right vectors) | MISSING — Task F |
| 7 | `partial_cholesky_orth_`, `remove_linear_dep_` (pivoted path) | `DPSTRF` / `ZPSTRF` | MISSING — Task G |
| 8 | Triangular solves behind `cho_solve`, `X = solve(clᵀ, csᵀ)ᵀ` (sfx2c1e) | `DTRTRS` / `DPOTRS`, `ZTRTRS` | MISSING — Task H (fold into A/B) |
| 9 | Public complex `zgemm` / level-2 BLAS surface | `ZGEMM`, `DGEMV`, `ZGEMV` as `pub` (already `pub(crate)`) | MISSING, small — Task I |

Intentionally out of scope (not LAPACK replacements):

- CubeCL device GEMM/GEMV/DOT/AXPY/SCAL kernels (`pyscf-algebra/src/gemm.rs`,
  `gemv.rs`, `dot.rs`, …, `zgemm.rs`): device-resident compute with no host
  round-trip. This crate's host `dgemm` must not replace them; the `device`
  feature path (`src/device`) is the counterpart for on-device work.
- Analytic 3×3 inverse/determinant (`pyscf-pbc-tools/src/mat3.rs`,
  `pyscf-pbc-gto/src/cell.rs`): closed-form, no LAPACK call upstream either.
- Python `scipy.linalg.*` calls under `pyscf/`: already LAPACK-backed;
  nothing to reproduce or replace.
- `df_metric_fit`'s rank-revealing truncation and the Davidson iteration
  itself: solver-level orchestration, stays in pyscf_rs even after Tasks A–F.

## 4. Tasks

### Task A — `dpotrf` / `zpotrf` (unblocked + blocked)

Transcribe `dpotrf.f` / `zpotrf.f` (recursive panel via `dpotf2`/`zpotf2`,
blocked update via `dsyrk`/`dgemm`, `zherk`/`zgemm`; needs public or
`pub(crate)` `dsyrk`/`zherk` first). Signatures:

```rust
pub fn dpotrf(uplo: Uplo, n: usize, a: &mut [f64]) -> Result<(), LapackError>;
pub fn zpotrf(uplo: Uplo, n: usize, a_re: &mut [f64], a_im: &mut [f64]) -> Result<(), LapackError>;
```

`INFO > 0` (leading minor not positive-definite) maps to
`LapackError::NoConvergence { routine: "dpotrf", info }` so pyscf_rs can keep
its non-PD error contract. Consumers migrated after landing:
`host_fallback::cholesky`, `zcholesky_faer` (+ delete `zcholesky_crout`'s
hand-written Crout or keep it as the cross-check route), DF Cholesky metric.
Oracle: `ref_potrf_bitexact.rs` vs `target/ref-lapack`, property test
`L·Lᵀ = A` + non-PD rejection.

### Task B — `dgetrf` / `dgetrs` / `zgetrf` / `zgetrs`

Transcribe `dgetrf.f` (`dlaswp`, `dtrsm`, `dgemm` panel), `dgetrs.f`
(`dlaswp` + `dtrsm`), and the `z` counterparts (`zlaswp`, `ztrsm`,
`zgemm` are already `pub(crate)` in `zblas.rs`; `dtrsm` needs transcribing —
`dtrmm` exists but `dtrsm` does not). Pivot vector `ipiv` is 1-based
`Vec<usize>` exactly as LAPACK. Consumers: `solve_linear`
(`dgesv`-equivalent: `dgetrf` + `dgetrs`, singularity → non-finite check
stays in the caller), `zsolve_linear_faer`, `cho_solve` (via Task A+H).
Oracle: `ref_getrf_bitexact.rs`, solve-residual property tests.

### Task C — `dgeqrf` / `dorgqr`

Transcribe `dgeqrf.f` (`dlarfb`, `dlarft`, `dlarfg` — `dlarfg` already exists
as `blocks::dlarfg`) and `dorgqr.f`. Consumer: `host_fallback::qr`
(rectangular support can stay square-only in pyscf_rs). Oracle:
`ref_geqrf_bitexact.rs`, `Q·R = A`, `QᵀQ = I` property tests.

### Task D — `dgesvd` (QR-iteration path)

Transcribe `dgesvd.f` + callees (`dgebrd`, `dorgbr`, `dbdsqr`). This is the
largest task; `DGESDD` (divide-and-conquer) would reuse this crate's D&C
machinery but is a bigger transcription — prefer `DGESVD` first. Consumer:
`host_fallback::svd`. Oracle: `ref_gesvd_bitexact.rs`,
`U·diag(s)·Vᵀ = A` property tests.

### Task E — `dsygv` / `zhegv` (type 1)

Transcribe `dsygv.f` / `zhegv.f` (`dsygst`/`zhegst` reduction to standard
form via the Task-A Cholesky factor, then `dsyevd`/`zheevd`, then back-transform
`dsytrs`-style through `dtrsm`/`ztrsm`). pyscf_rs keeps its
linear-dependency truncation (eigenvalues ≤ tol dropped, `+∞` padding in
`eigh_gen`): the driver returns the full spectrum and the caller truncates,
matching today's contract. Consumers: replace the Löwdin glue in `eigh_gen`
and `zeigh_gen_faer`, deleting the faer `Mat` products there. Oracle:
`ref_sygv_bitexact.rs`, `F·C = S·C·diag` property tests.

### Task F — `dgeev` / `zgeev` (values + right vectors)

Transcribe `dgeev.f` / `zgeev.f` (Hessenberg reduction `dgehrd`, QR iteration
`dhseqr`, back-transform `dorghr`/`dtrevc`). Consumers:
`davidson::eig_general` / `zeig_general`, `pbc-adc/roots.rs`,
`pbc-symm/group.rs`. Start with `dgeev`; `zgeev` follows the same structure
on planar storage. Oracle: `ref_geev_bitexact.rs`, `A·V = V·diag(w)`
property tests (complex pairs for real input).

### Task G — `dpstrf` / `zpstrf` (pivoted Cholesky)

Transcribe `dpstrf.f` / `zpstrf.f` (rank-revealing, tolerance-driven stop,
pivot vector). Consumers: `partial_cholesky_orth_` and the pivoted branch of
`remove_linear_dep_`. Oracle: `PᵀAP = L·Lᵀ` truncated property tests.

### Task H — `dtrtrs` / `dpotrs` (+ `z`)

Small: transcribe `dtrtrs.f` (`dtrsm` wrapper with checks) and `dpotrs.f`.
Fold into Task A/B work. Consumers: `cho_solve`, sfx2c1e triangular solves.

### Task I — public BLAS surface

Promote the already-`pub(crate)` `dgemv`/`dger`/`dsymv`/`dsyr2k`/`dtrmm` and
`zgemm`/`zgemv`/`zhemv`/`zher2k`/`ztrmm` to `pub` LAPACK-style wrappers in
`blocks` (column-major, `lda` args, `Tr`/`Uplo` enums), mirroring the existing
`blocks::dgemm`. No new transcription, only API + oracle coverage
(`ref_blas_bitexact.rs` already covers the kernels).

## 5. Suggested order

A (POTRF) → B (GETRF/GETRS) → H (TRTRS/POTRS, folded in) → E (SYGV/HEGV:
removes the last faer use from the SCF hot path) → F (GEEV) → C (GEQRF) →
D (GESVD) → G (PSTRF) → I (public BLAS).

A+B+H unblock the two most-used remaining faer sites (Cholesky factor and LU
solve); E deletes the Löwdin glue's faer dependency so `pyscf-algebra` needs
faer only for QR/SVD/non-symmetric eig; F–G finish the host migration.

## 6. Per-task checklist

1. Transcribe from `target/ref-lapack/lapack-3.12.1/...`, naming netlib files
   in the module header; no FMA / `mul_add` / fast-math; gfortran complex
   lowering (full products, signed zeros).
2. Column-major, `LDA = n`, `Uplo`/`Job` enums, workspace query
   (`*_work_len`), caller-supplied workspace, `*_alloc` convenience wrapper;
   planar complex; `LapackError` mapping (`INFO < 0` → `InvalidArgument`,
   `INFO > 0` → `NoConvergence`).
3. `tests/ref_<routine>_bitexact.rs` against `LAPACK_RS_REF_DIR` (SKIP without
   it, fail with `LAPACK_RS_REQUIRE_REF=1`) + property tests
   (`A·x = b` residual, `L·Lᵀ = A`, `Q·R = A`, …).
4. Device kernels only where the routine belongs on-device per the MAGMA-style
   hybrid rule; host-only routines (GETRF panel, GEEV Hessenberg-QR) stay host.
5. Migrate the pyscf_rs consumer listed in the task to the new surface and
   delete the superseded faer/manual code (keeping tests green:
   `pyscf-algebra`, `pyscf-scf`, `pyscf-diis`, `pyscf-df`, `pyscf-pbc-x2c`).

---

## Addendum (2026-09-17): Task I is delivered

`DGEMV` (and `ZGEMV`) are now a public surface — `lapack_rs::blocks::dgemv` /
`blocks::zgemv`, with the public `lapack_rs::Trans` enum and planar complex
slices — gated bit for bit against the reference BLAS by
`tests/ref_blas_bitexact.rs::blocks_dgemv_and_zgemv` over every `TRANS`, both
increments, `LDA > M` and the `alpha`/`beta` special cases.

This is the same deliverable as Task C of
`docs/cintx-lapack-blas-adoption-plan.md` ("do it once, for both consumers"),
and was done there; nothing further is needed on this plan's side.
