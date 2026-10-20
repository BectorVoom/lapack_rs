# Implementation plan: bit-exact parity with OpenBLAS

> **Status (2026-09-17): done — see `docs/openblas-bitexact-result.md`.**
> O0 was run first (`tests/openblas_o0_differential.rs`) and confirmed §3:
> OpenBLAS vendors Reference-LAPACK 3.12.1 unchanged except `DORM2L/R`
> (`ZUNM2L/R`, legacy `DLARF`), so the whole gap is the BLAS. By the user's
> decision the implementation took **Path O-Reimpl on CubeCL** instead of
> Path O-FFI: every BLAS routine of the `DSYEVD`/`ZHEEVD` call tree was
> transcribed from the pinned binary (`src/openblas/`) and the heavy ones run
> as OpenBLAS-order CubeCL kernels on the CPU runtime (`src/device/kernels/
> ob_*.rs`), so no OpenBLAS is linked at run time (only by the test-only
> oracle `crates/openblas_ref`, O3). `openblas::{dsyevd, zheevd}` are
> `max_ulp = 0` against `scipy_dsyevd_`/`scipy_zheevd_` up to `n = 1024`
> (O4 gates), and the real driver runs at or above the Reference device path's speed
> (0.31 s vs 0.33 s at `n = 1024`, 2.04 s vs 2.64 s at `n = 2048`). The text below is
> the original plan, kept for its reasoning.

## 1. What this plan is answering

pyscf_rs's PBC band-structure oracle (`krhf_bands_oracle.rs`) compares Rust
KRHF/band eigenvalues against upstream PySCF, running under this workspace's
`.venv`, whose `scipy` is linked against **OpenBLAS**, not Reference-LAPACK.
Measured on the cheapest fixture (He, fcc, `sto-3g`, all-electron, 2×2×2):
`0/10` eigenvalues bit-identical, worst `|Δ| = 4.24e-13`. The question this
plan scopes is whether `lapack_rs` — whose stated contract (AGENTS.md §5) is
`max_ulp = 0` against **Reference-LAPACK 3.12.1** — can *also* be made
`max_ulp = 0` against the specific OpenBLAS build `scipy` actually calls, and
if so, how.

**This plan does not assume the answer.** `docs/a0-wheeler-differential.md`
already measured, on this exact codebase, that two independently-"correct"
eigensolvers from the same algorithm family (Reference `DSTEMR` vs libcint's
vendored MRRR) disagree by up to 4.25×10⁹ ULP with only 23% bit agreement —
not a rounding difference, a different answer. OpenBLAS vs Reference-LAPACK
is a less adversarial pairing (same LAPACK *algorithm* in most cases, see
§3), but the lesson stands: this is measured, not assumed, before any
engineering is committed. §5 Task O0 is that measurement, and it runs BEFORE
Task O1.

## 2. The exact target, pinned

`max_ulp = 0` against "OpenBLAS" is not a single, stable target — OpenBLAS is
a `DYNAMIC_ARCH` dispatcher that picks a different hand-written kernel per
CPU, and its LAPACK layer is only bit-stable at a fixed thread count. Probed
against this workspace's own `.venv` on the current dev machine (AMD Ryzen AI
7 350, gfx1151 host):

```
scipy 1.17.1, scipy.libs/libscipy_openblas-6cdc3b4a.so
  scipy_openblas_get_config()   = "OpenBLAS 0.3.30 DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=64"
  scipy_openblas_get_corename() = "SkylakeX"      # AVX512 kernel, picked by CPUID, not by vendor
  scipy_openblas_get_parallel() = 1               # pthreads, not sequential, not OpenMP
  compiled max threads          = 16              # == nproc
```

Three consequences, each a precondition the gate must assert, not hope for:

1. **`OPENBLAS_CORETYPE` must be pinned.** `DYNAMIC_ARCH` means a different
   CPU (or the same CPU with different CPUID-visible features masked) runs a
   different micro-kernel with a different SIMD width and blocking factor —
   a different summation order, hence different bits. Pin with
   `OPENBLAS_CORETYPE=SkylakeX` and assert `scipy_openblas_get_corename()`
   returns that string before trusting a comparison; a gate that does not
   check this is silently CPU-dependent.
2. **`OPENBLAS_NUM_THREADS=1` must be set.** Parallel model `1` (pthreads)
   means GEMM-shaped calls above OpenBLAS's internal size threshold split
   across threads, and a threaded reduction's summation order is not fixed
   by the algorithm — it depends on scheduling. `n` in the PySCF band
   fixtures is tiny (`nao = 8` at 2×2×2 on He/sto-3g) and single-threaded
   already at default settings, but any corpus task that scales `n` up must
   pin this explicitly or its own baseline stops being reproducible run to
   run, independent of lapack_rs entirely.
3. **The comparison is to *this* OpenBLAS/scipy pair, not "OpenBLAS" as a
   concept.** A `scipy`/`numpy` upgrade can rebuild against a newer OpenBLAS
   minor version with different kernels for the same `CORETYPE` string. The
   gate must record and assert the exact `scipy.__version__` and
   `scipy_openblas_get_config()` string it was measured against, the same
   way `krhf_bands_oracle.rs` already asserts PySCF `== "2.12.1"`.

## 3. Why independent re-transcription is very unlikely to reach `max_ulp = 0`

`lapack_rs`'s existing bit-exactness (AGENTS.md §5) is reached by
transcribing Reference-LAPACK's Fortran control flow *and* its reference
BLAS, both built with `-ffp-contract=off -fno-fast-math`, so every
floating-point value has one prescribed operation order. OpenBLAS breaks that
premise at the BLAS layer on purpose:

- **SIMD-blocked reductions.** OpenBLAS's `DGEMM`/`DSYMV`/`DSYR2K` micro-
  kernels accumulate in vector lanes (AVX-512 for the `SkylakeX` kernel: 8
  `f64` lanes) and then horizontally reduce — a different summation tree than
  the reference BLAS's scalar loop. Floating-point addition is not
  associative, so this changes the last few bits even when both sides
  compute mathematically the same sum.
- **FMA contraction.** The reference build explicitly disables it
  (`-ffp-contract=off`); OpenBLAS's hand-written kernels use `vfmadd*` by
  design on AVX2/AVX-512 targets — a fused multiply-add rounds once instead
  of twice, which is a bit-level difference on almost every accumulate.
- **Threading.** Per §2, above the internal size threshold OpenBLAS's thread
  pool changes which partial sums combine in which order depending on
  scheduling, so the *same* OpenBLAS binary is not even bit-stable against
  itself across runs at large `n` without pinning thread count.
- **Version/kernel churn.** The exact instruction sequence a micro-kernel
  emits is undocumented, unversioned API surface — a hand transcription
  target that can silently change on any OpenBLAS point release, and would
  need a separate transcription per `CORETYPE` (SkylakeX, Zen, Haswell, …).

None of this touches the *LAPACK*-level control flow, though. OpenBLAS does
not reimplement `DSYEVD`/`ZHEEVD` — for the divide-and-conquer symmetric/
Hermitian eigensolvers it vendors the same family of netlib LAPACK Fortran
sources (see Task O0) and links them against its own BLAS. So the two
solvers likely agree on *algorithm* (same block-size heuristics, same
tridiagonalization, same D&C merge/deflation), and disagree only in what
each individual `DSYMV`/`DGEMM`/`DLARFB` call inside that algorithm computes.
That structural fact is what makes §4's Path O-FFI plausible where a from-
scratch BLAS transcription is not.

## 4. Two candidate paths

### Path O-FFI (recommended candidate, pending Task O0)

Keep `lapack_rs`'s existing Reference-LAPACK-transcribed `DSYEVD`/`ZHEEVD`
control flow (block sizes, D&C tree, deflation, convergence tests —
unchanged, still transcribed from `target/ref-lapack/lapack-3.12.1`) but give
it a second BLAS backend that calls the *actual* OpenBLAS shared library
(`libscipy_openblas-*.so`) via FFI for the underlying `DGEMM`/`DSYMV`/
`DSYR2K`/`DLARFG`/`DLARFB`-equivalent calls, instead of `lapack_rs`'s own
reference-transcribed BLAS. Because the arithmetic is then literally the same
compiled routine OpenBLAS itself executes, agreement is exact **by
construction** wherever the LAPACK-level control flow already matches — no
kernel reverse-engineering, no per-`CORETYPE` transcription, and it survives
OpenBLAS point releases automatically (it *is* that release).

Cost: this is a real architectural exception, not a free win.
`AGENTS.md` §5 states "Backends: CubeCL only, no rayon... There is no host
parallel path" for this crate. Linking a prebuilt, non-CubeCL, non-Rust
shared library for one path is the same category of exception the crate
already grants `cint_stemr` (§`Two oracles, never mixed`) for a *different*
oracle — it must be built the same way: a separate, clearly-labeled module
(`src/openblas_ref.rs` or a `crates/openblas_ref` FFI shim, mirroring
`crates/cint_ref`), gated by an env var (`LAPACK_RS_OPENBLAS_DIR` or similar,
`OPENBLAS_REQUIRE_REF=1` to make a gate run fail instead of `SKIP`), and
**never** merged into or asserted equal to the Reference-LAPACK path's own
bit-exactness gate. This needs an explicit go-ahead from whoever owns the
"CubeCL only" rule before Task O2 starts, since it is a deliberate, scoped
exception to it, not a bug fix.

### Path O-Reimpl (not recommended)

Independently transcribe OpenBLAS's `SkylakeX` (or whichever `CORETYPE`)
AVX-512 micro-kernels into Rust/CubeCL, byte for byte. Rejected for three
reasons: (a) the exact vectorized summation order is not documented API and
would need to be extracted from disassembly per kernel per `CORETYPE`, a
maintenance liability with no upstream contract behind it; (b) it must be
redone for every `CORETYPE` a user's machine might dispatch to (§2) — the
`DYNAMIC_ARCH` dispatch table has dozens of entries; (c) it duplicates work
Path O-FFI gets for free by construction. Keep this path only as a fallback
if Task O2's licensing/build review rules out linking OpenBLAS at all (§6).

## 5. Tasks

### Task O0 — Measure before building anything (report-only, like A0)

Mirrors `a0_wheeler_differential.rs`'s method exactly: measure first, decide
after.

1. Identify which netlib LAPACK release/patch level OpenBLAS 0.3.30 vendors
   for `dsyevd.f`/`zheevd.f` and its callees (`dsytrd`/`zhetrd`, `dstedc`,
   `dlaed*`), and diff its control flow against
   `target/ref-lapack/lapack-3.12.1`'s. Confirms or refutes §3's assumption
   that the two sides run the *same* algorithm before any FFI work is built
   on top of that assumption.
2. Build a small FFI probe (throwaway, like the deleted `faer_c64_probe.rs`
   pattern) that calls `libscipy_openblas-*.so`'s `dsyevd_`/`zheevd_` symbols
   directly (not through Python), with `OPENBLAS_CORETYPE` and
   `OPENBLAS_NUM_THREADS=1` pinned per §2, asserted via
   `scipy_openblas_get_corename()`/`get_parallel()` equivalents exposed by
   the same library.
3. Differential-test `lapack_rs`'s existing Reference-LAPACK-bit-exact
   `dsyevd_alloc_host`/`zheevd_alloc_host` against that probe over: (a) a
   random symmetric/Hermitian corpus spanning `n = 1..64` (small `n` first —
   §3's SIMD/FMA divergence should still appear even at `n` too small to
   thread), and (b) the real Fock/overlap matrices from the He/Si PBC test
   fixtures already in `pyscf_rs` (`he_all_electron`, `si()`), since those
   are the matrices the actual band-comparison gate cares about.
4. Report the same shape of table `a0-wheeler-differential.md` uses: % bit-
   identical, max ULP by `n`, magnitude-worst first. **This table is the
   go/no-go for Task O1.** If agreement is already high without any FFI
   change (§3's structural argument holds and the D&C tree is short/`n` is
   small enough that only a handful of BLAS calls fire), Path O-FFI may be
   unnecessary for the sizes pyscf_rs actually needs. If it is as adversarial
   as the Wheeler corpus, that is Task O0's answer too, and it is written
   down instead of assumed.

### Task O1 — Decide the path from O0's numbers

Write the verdict (mirrors `a0-wheeler-differential.md` §"What this decides"):
which routines/size ranges need Path O-FFI, which already agree without it,
and whether Path O-Reimpl's fallback is warranted anywhere. No engineering
before this verdict is written down.

### Task O2 — Build/link review for Path O-FFI

Before writing any FFI code: get sign-off that pulling a prebuilt OpenBLAS
`.so` into this crate's build (even behind a feature flag, even test-only) is
an acceptable, explicitly-scoped exception to AGENTS.md §5's "CubeCL only, no
rayon, no host parallel path" rule. Resolve: vendored via a `-sys` crate
pinned to the exact `scipy-openblas32`/`64` wheel version, or a build-time
path env var (`LAPACK_RS_OPENBLAS_DIR`, mirroring `LAPACK_RS_REF_DIR`) that
points at the `.venv`'s own copy so the oracle is provably the same binary
`scipy` uses, not a separately-built OpenBLAS that could pick a different
kernel. The latter is preferred: it removes an entire axis of "which OpenBLAS
build" ambiguity by reusing pyscf_rs's own `.venv`.

### Task O3 — `openblas_ref` FFI shim crate

Mirrors `crates/cint_ref`: a thin FFI crate exposing OpenBLAS's `dsyevd_`/
`zheevd_` (and, if O0 shows BLAS-level substitution is the fix rather than
whole-routine delegation, the individual `dgemm_`/`dsymv_`/`dsyr2k_`/
`zgemm_`/`zhemv_`/`zher2k_` symbols) with the pinning from §2 asserted at
init (fail loud, not `SKIP`, if `OPENBLAS_CORETYPE`/thread count don't match
what was recorded). Gated by `LAPACK_RS_OPENBLAS_DIR`; without it, tests
print `SKIP` exactly like the existing `LAPACK_RS_REF_DIR`/
`CINTX_LIBCINT_DIR` gates.

### Task O4 — Oracle tests

`tests/ref_syevd_openblas_bitexact.rs`, `tests/ref_heevd_openblas_bitexact.rs`
— same corpus and reporting shape as Task O0's probe, now gating CI instead
of just reporting. Per the "two oracles, never mixed" rule (AGENTS.md §5):
this is a **third** oracle track alongside Reference-LAPACK (`ref_*_bitexact`)
and libcint (`cint_stemr_bitexact`) — never asserted equal to either of the
other two, and its own module headers say so in the first line, same as
`cint_stemr.rs` does today.

### Task O5 — Document the verdict

A `docs/openblas-bitexact-result.md` in the shape of
`a0-wheeler-differential.md`: the measured table, which routines/`n` ranges
reach `max_ulp = 0`, which don't and why, and update
`krhf_bands_oracle.rs`'s doc comment (currently: "the most that can be
expected here" is ~1e-9, reasoned from the GDF J/K chain) with the measured
number once Path O is live, rather than the reasoned estimate it uses today.

## 6. Suggested order

O0 (measure) → O1 (verdict) → O2 (build/link sign-off) → O3 (FFI shim) → O4
(gate) → O5 (document). O0 is cheap (a throwaway FFI probe plus a
differential test, no new production code) and is the only step that can
show Path O-FFI is unnecessary or insufficient before any of O2/O3's real
engineering cost is spent — do not skip ahead to O3 on the strength of §3's
structural argument alone; that argument is *why O0 is worth running*, not a
substitute for running it.

## 7. Open risks

- **O0 may find the LAPACK-level control flow itself differs** (different
  netlib patch level, different `ILAENV` block-size table, or an OpenBLAS-
  side patch to `dsyevd`/`zheevd`). If so, Path O-FFI's "same algorithm, only
  the BLAS differs" premise is false and O1 must scope down to whichever
  routines/sizes it still holds for, or fall back to Path O-Reimpl for the
  rest.
- **`DYNAMIC_ARCH` moves the goalposts per machine.** A gate that hard-codes
  `SkylakeX` will `SKIP`/fail-loud (not silently pass) on a different CPU
  that dispatches a different kernel; document this as a known, accepted
  scope limit rather than trying to cover every `CORETYPE`.
- **This is a genuinely different goal from the crate's existing contract.**
  §"Bit-exactness is the contract" (AGENTS.md §5) is scoped to Reference-
  LAPACK 3.12.1 today; nothing in this plan changes that default path or its
  existing gates. Path O is additive and opt-in, exactly like `cint_stemr`'s
  libcint oracle already is.
