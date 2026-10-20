# [SYSTEM RULE] Cubecl Implementation and Error Handling Protocol

## 1. Role and Core Mission
You are an expert Rust development agent tasked with modifying, implementing, or generating code for projects utilizing the `cubecl` crate. You must adhere to the following execution protocols without exception. Any violation of these procedures is considered a critical failure.

---

## 2. Mandatory Code Structure Rule (Separation of Source and Test)
You must strictly separate source code and test code into independent files. **Do not write unit tests inside the production source files (e.g., embedding `mod tests` at the bottom of a source file is strictly prohibited).**

### Required Structure:
*   **Production Source Code:** Keep it clean and dedicated solely to implementation logic.
*   **Test Code:** All tests (both unit tests and integration tests) must be written in separate, dedicated test files. Follow the standard Rust conventions for multi-file test organization (e.g., using `tests/` directory or explicit separate module files like `src/foo.rs` and `src/foo_test.rs` / `tests/foo_tests.rs` as specified by the project structure).

---

## 3. Mandatory Pre-Implementation Check
Please write all computation engines according to the CubeCL manual. Cubecl kernel need generics-float in this project. CubeCL documentation is available at: /home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md
Please **read the manual before writing any code.**

---

## 4. Strict Build Error Resolution Protocol
If a build error occurs in any Rust project using the `cubecl` crate, you are **STRICTLY PROHIBITED** from attempting blind fixes or proposing changes without consulting the dedicated guideline.

### Trigger Conditions
This protocol is activated immediately upon detecting any failure related to `cubecl` concerning:
*   Building, compiling, or linking
*   Dependency resolution or feature flags
*   Toolchain configuration or CI execution

### Step-by-Step Procedure
1.  **Read the Guideline:** Immediately load and read the documents in the directory `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/`.
2.  **Follow the Process:** Execute the exact troubleshooting and resolution process documented in that manual.
3.  **Align Communication:** When reporting the issue or proposing a fix, format your explanation to strictly align with the structure and terminology defined in the manual.
4.  **Document and Prevent:** Once resolved, document the root cause, specific resolution steps, verification results, and prevention measures in strict accordance with the manual's template (Problem Description → Root Cause → Recommended Solutions → Best Practices → Summary). Say explicitly when no document covers the error.

## Cubecl manual
/home/user/Documents/workspace/cubecl_manual/manual/Cubecl

---

## 5. lapack_rs rules

### Backends: CubeCL only, no rayon
There is **no host parallel path**. `rayon` and `src/par.rs` were deleted on
2026-09-17 by the user's decision; `cubecl` is a non-optional dependency and
`default = ["cpu"]` (the CubeCL CPU runtime). `cpu`, `rocm` and `cuda` select
only the *runtime* — one kernel set, identical bits on all three.

* `dsyevd_alloc` / `zheevd_alloc` run the device drivers on
  `runtime::DefaultRuntime` (`cuda` > `rocm` > `cpu`), with a per-thread
  cached client and `DeviceWorkspace`.
* `dsyevd` / `zheevd` (explicit `LWORK`) stay the **sequential** reference
  transcription: the reference's block sizes, and so its bits, depend on the
  workspace lengths the caller passes, which the device drivers do not model.
  `dsyevd_alloc_host` / `zheevd_alloc_host` are that path with the opt
  workspace, and are what the `_alloc` entry points must reproduce bit for
  bit.
* **Known regression, accepted by the user:** on a 16-core host the
  CpuRuntime path is 0.54–0.60x the deleted rayon path (n=256 … 1024); the
  gap is the `DORMTR` phase, which replays `DLARFB` as seven element-wise
  reference-order passes — right for a GPU, worse than blocked BLAS on a CPU.
  See `docs/device-speed-plan.md` §3.
* GPU gates build with `--no-default-features --features cuda|rocm` so the
  heavy `cubecl/cpu` (MLIR/LLVM) stack is not pulled in too.

### Two oracles, never mixed
Almost everything in this crate is bit-exact to **Reference-LAPACK 3.12.1**.
`src/cint_stemr.rs` and `src/device/kernels/cint_stemr.rs` (Path C) are the
one exception: they are bit-exact to **libcint 6.1.3**'s `src/eigh.c`
`#else` branch, which is a *different* MRRR with different constants, and
`ref_*_bitexact` does not and must not cover them. The two cannot be satisfied
by one routine — `docs/a0-wheeler-differential.md` measures the gap at up to
4.25e9 ULP, with only 23.1% of eigenvalues agreeing. Their oracle is
`crates/cint_ref` (build-gated by `CINTX_LIBCINT_DIR`, required by
`CINT_REF_REQUIRE=1`), their gate is `tests/cint_stemr_bitexact.rs`, and their
module headers say so in the first line.

### Third oracle: scipy's OpenBLAS (never mixed either)
`src/openblas/`, `src/device/kernels/ob_*.rs` and the `openblas::*` /
`device::*_openblas` entry points are bit-exact to **scipy's bundled OpenBLAS
0.3.30** (`libscipy_openblas-6cdc3b4a.so`, `SkylakeX` core, one thread, its own
`LWORK` query) — see `docs/openblas-bitexact-result.md`. They share the one
LAPACK transcription through `crate::flavor::Blas` / `device::flavor::DeviceBlas`
(`RefBlas` vs `OpenBlas`); nothing may be asserted equal across the tracks.
* The BLAS is transcribed from the **disassembly** (source for structure):
  `target/ob/all.s`, `target/ob/dis_name.sh`, `target/openblas-src/OpenBLAS-0.3.30`.
  gcc fused `a*b + c` into `vfmadd*`: use `f64::mul_add` / CubeCL `fma` exactly
  where the binary fuses and nowhere else. **This track is the only exception
  to "No FMA".** `dnrm2`/`dznrm2` are x87 80-bit; `DORM2L/R`, `ZUNM2L/R` call
  the legacy `DLARF`/`ZLARF` (`Blas::LEGACY_DLARF`).
* The host modules are compiled with `target_feature(enable = "fma")`; public
  entry points call `openblas::require_fma()` first.
* Oracle: `crates/openblas_ref` (build-gated by `LAPACK_RS_OPENBLAS_DIR` = the
  wheel's `scipy.libs`, required by `OPENBLAS_REQUIRE_REF=1`; runtime pins
  `OPENBLAS_CORETYPE=SkylakeX OPENBLAS_NUM_THREADS=1` are asserted). Every
  device kernel is gated against the host transcription
  (`tests/device_openblas_cpu.rs`, no library needed).
* Device kernels whose element formula varies with tile position run the
  plain-chain elements as lane-vectors and the rest from host-computed special
  intervals as scalar element lists (`ops::ob_*_special_intervals`, each with an
  element-wise cross-check test). A kernel that does not lower on cubecl-cpu:
  move its control flow to host-built tables (`ob_trmm_kernel`).

### Bit-exactness is the contract
Every host floating-point value must be the one Reference-LAPACK 3.12.1 (plus
the reference BLAS of the same tree) computes when built by
`cargo xtask build-ref-lapack` (`-O2 -march=x86-64 -mtune=generic
-ffp-contract=off -fno-fast-math -frecursive`, gfortran).

* Transcribe from the pinned sources under `target/ref-lapack/lapack-3.12.1`,
  never from memory; each module header names its netlib files.
* Speed may only come from reorderings that cannot change a value:
  element-wise kernels split across units/lanes (each element keeps its own
  reference-ordered chain), independent D&C subproblems, secular roots and
  eigenvector columns. Reductions stay sequential chains.
* Zero-skip branches, `LWORK`-dependent block sizes, `MAX`/`MIN`/`SIGN`
  lowering and complex arithmetic follow gfortran exactly. Complex products
  are always the full `(ar*br - ai*bi, ar*bi + ai*br)` (GCC honours signed
  zeros, verified on the oracle's `zlascl.f.o`).
* No FMA, no `mul_add`, no fast-math anywhere (the OpenBLAS track above is the
  single, scoped exception).
* A mismatch is bisected to one routine with `tests/ref_lapack_components.rs`
  and `tests/ref_blas_bitexact.rs` before anything is changed.

### Gates
```
export LAPACK_RS_REF_DIR=$PWD/target/ref-lapack/install LAPACK_RS_REQUIRE_REF=1
export CINTX_LIBCINT_DIR=/home/user/Documents/workspace/cintx/libcint-master CINT_REF_REQUIRE=1
cargo test --release --test ref_oracle_canary --test ref_blas_bitexact --test ref_lapack_components \
  --test ref_syevd_bitexact --test ref_heevd_bitexact \
  --test ref_mrrr_components --test ref_stemr_bitexact --test cint_stemr_bitexact
cargo test --release --test memory --test threads --test steqr_oracle --test syevd_oracle
cargo test --release --features stats --test stedc_deflation
cargo test --release --test bench -- --ignored --nocapture --test-threads=1
# OpenBLAS track (third oracle)
export LAPACK_RS_OPENBLAS_DIR=/home/user/Documents/workspace/pyscf_rs/.venv/lib/python3.13/site-packages/scipy.libs
export OPENBLAS_CORETYPE=SkylakeX OPENBLAS_NUM_THREADS=1 OPENBLAS_REQUIRE_REF=1
cargo test --release --test openblas_level1_bitexact --test openblas_level2_bitexact --test openblas_level3_bitexact \
  --test openblas_zlevel1_bitexact --test openblas_zlevel2_bitexact --test openblas_zlevel3_bitexact \
  --test openblas_zlevel3_components --test openblas_lapack_components --test openblas_syevd_bitexact \
  --test openblas_heevd_bitexact --test device_openblas_cpu --test openblas_device_syevd --test openblas_device_heevd
```
Without `LAPACK_RS_REF_DIR` the oracle tests print `SKIP`; with
`LAPACK_RS_REQUIRE_REF=1` they fail instead, so a gate run is never vacuous.

### Device rules
* Launch geometry: `geometry::launch_1d` sizes the GPU cube so the cube
  *count* reaches `num_streaming_multiprocessors · CUBES_PER_SM`; cubecl's
  `CubeDim::new` is the ceiling only. Do not re-tune `CUBES_PER_SM` (measured
  flat over 1–16; only cubecl's own default is slow).
* Kernels: `#[cube(launch, launch_unchecked)]`, generic `F: Float + CubeElement`
  (and `N: Size` for lane vectors), launched `::<F, R>` only from
  `src/device/ops.rs` through `kl!` (checked/unchecked switch). Every
  `ArrayArg::from_raw_parts` sits in an `unsafe` block with `// SAFETY:`
  (`tests/unsafe_launchers.rs`).
* Under `Exactness::Reference` a kernel element reproduces the host chain
  bit for bit: same operand order, full complex products (also with `CONE`
  and real operands), selects instead of masked multiplies, reference
  zero-skips. Bits must not depend on vector width, planes, cube dimension,
  kernel variant (K2/K2r, K3/K3r) or checked launches
  (`tests/device_launch_parity.rs`, `tests/device_rocm_vs_cpu.rs`).
* No vector unary minus (HIP's C rejects it): fold negations with exact IEEE
  identities (`a − (−p) = a + p`), never `0 − x`.
* **`F::new` takes an `f32`.** `F::new(0.333)` is the `f32` nearest 0.333
  widened back to f64 — about 1e-8 out, enough to change a shift. Only dyadic
  values survive it; build those from `F::from_int` and pass every other
  constant in as an argument (`ops::SecularConsts`, `ops::StemrConsts`,
  `ops::cint_stemr_consts`). `F::is_nan` returns `WithScalar<bool>`, not
  `bool`; write `DISNAN( X )` as `x != x`, which is what `dlaisnan.f` computes.
  The natural log is `F::ln`, not `F::log`.
* **Per-unit scratch needs a per-unit base.** A batched kernel that forgets to
  offset one workspace races silently: the eigenvalues can come out identical
  and only `INFO` differs. `tests/device_rocm.rs::cint_stemr_launch_shapes_
  agree_f32` caught exactly that in K9.
* **cubecl-cpu 0.10 cannot lower every kernel.** Several K9 routines fail with
  `operation with block successors must terminate its parent block` on a
  worker thread and the launch **silently no-ops**, so the caller reads zeros.
  It is not size alone, not a construct, and the vendored `cubecl-opt` does
  not fix it; hipRTC and NVRTC lower them. Full write-up and the checklist for
  the next occurrence: `docs/cubecl-cpu-lowering-limit.md`; regression test
  `tests/cubecl_cpu_limits.rs::cint_dlasq2_exceeds_cpu_runtime_lowering_limit`
  (`#[ignore]`d). Treat a `CpuRuntime` launch that returns zeros as a compile
  failure until proven otherwise.
* `read_range` takes the buffer's logical length: `Handle::size()` is padded
  on HIP.
* Hot index expressions stay stride-1 across lanes
  (`tests/device_sector_model.rs`); new accesses are added there. The
  exception is a kernel whose inner loop is O(m) long, where locality across
  iterations wins on a GPU — see the CUDA notes.

### Device gates
```
export LAPACK_RS_REF_DIR=$PWD/target/ref-lapack/install LAPACK_RS_REQUIRE_REF=1
export CINTX_LIBCINT_DIR=/home/user/Documents/workspace/cintx/libcint-master CINT_REF_REQUIRE=1
cargo test --release
export ROCM_PATH=$HOME/rocm/opt/rocm HIP_PATH=$HOME/rocm/opt/rocm LD_LIBRARY_PATH=$HOME/rocm/opt/rocm/lib
HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off cargo test --release --features rocm --test device_rocm --test device_rocm_vs_cpu
cargo run --release --example device_memory -- 1024
# CUDA (f64 NVIDIA GPU, e.g. a Colab T4; `crates/cubecl-cuda` is a patched 0.10.0)
NVRTC_COMPILE_OPTIONS_APPEND=--fmad=false CUDA_PATH=/usr/local/cuda \
  cargo test --release --no-default-features --features cuda --test device_cuda
```

### CUDA notes
* NVRTC contracts into FMA by default and upstream cubecl-cuda 0.10 exposes no
  switch, and it leaves f64 out of the supported types (a matmul issue in
  cubecl-cpp). Both are fixed in the vendored `crates/cubecl-cuda`
  (`PATCHED.md`); the contraction probe must report no contraction before any
  bit-exactness claim.
* `DLAED3` on the device (K4, `src/device/kernels/secular.rs`) now runs on
  the CubeCL CPU runtime too, not GPU runtimes only. Two distinct cubecl-cpu
  0.10 / cubecl-opt 0.10 bugs blocked this and BOTH are now fixed:
  cubecl-cpu 0.10 cannot lower a `while` loop whose counter is DECREMENTED
  (`dlaed4_mid`'s four `jr1`..`jr4` phi/dphi/erretm loops used to be exactly
  that; they are now ascending `for t in bound..n + 1 { let jrN = n + bound
  - t; .. }` over the identical visit order — fixed by avoiding the
  pattern; the underlying backend limitation itself is permanent and stays
  documented, regression test `tests/cubecl_cpu_limits.rs::descending_
  while_counter_fails_on_cpu_runtime`, `#[ignore]`d); and a cubecl-opt
  PRE/GVN bug (`cubecl-opt-0.10.0/src/gvn/convert.rs:22:39: "no entry found
  for key"`, root-caused in `gvn/apply.rs::insert_block`: it hoisted an
  expression into a predecessor that had no leaders for that expression's
  own operands) is fixed by vendoring `crates/cubecl-opt`
  (`[patch.crates-io]` in `Cargo.toml`; see `crates/cubecl-opt/PATCHED.md`
  for the exact change, why it cannot alter a floating-point value, and how
  to reproduce the original ICE — regression test now asserting SUCCESS,
  `tests/cubecl_cpu_limits.rs::dlaed4_mid_gvn_convert_compiles_and_runs_on_
  cpu_runtime`). `secular_roots_kernel` now compiles and matches the host
  bit for bit on the CPU runtime
  (`tests/device_secular_cpu.rs::secular_roots_kernel_cpu_runtime_matches_
  host`, plus a full `dstedc_device` driver test).

  `[patch.crates-io]` only rewires builds from this workspace, so a
  downstream consumer of a published `lapack_rs` with the `cpu`, `rocm` or
  `cuda` feature (all three share this one vendored `cubecl-opt`) would get
  the unpatched crate, hit the identical ICE on a cubecl worker thread, and
  get a launch that silently no-ops and reads back zeros — wrong results,
  not a build error. `src/device/stedc.rs::DeviceDc::secular` therefore
  gates K4 on a runtime capability probe, `device::probe::k4_supported`
  (modelled on `contraction_probe`), not on `has_planes` or on this crate
  compiling: it launches the real kernel on a tiny `k = 3` fixture and
  compares against the host `DLAED4` computed at runtime, caches the
  verdict per runtime type, and an unpatched build's silent zeros fail the
  comparison and fall back to the host automatically (bit-exact either way,
  just slower). `tests/device_secular_cpu.rs::k4_supported_reports_true_on_
  cpu_runtime` asserts the probe reports `true` here, so a regression to
  the vendored patch fails loudly instead of silently taking the host path.

  `tests/device_cuda.rs::cuda_secular_equation_matches_host` remains K4's
  GPU-runtime gate (an f64 GPU is not available locally; see "Memory and
  machine notes"), but is no longer the only bit-exactness gate for K4 —
  `tests/device_secular_cpu.rs` covers the CPU runtime.
* Kernel speed on a GPU is set by the memory access pattern ACROSS loop
  iterations as much as across lanes: kernels whose inner loop is long (K1,
  K1c) read rows as columns; kernels whose inner loop is short (K2, K3
  `gemm_n`, 32 steps) keep the lane-coalesced column layout. Measure with
  `examples/device_kernel_bench.rs` before changing a default.

### Memory and machine notes
* The session scratchpad is a RAM tmpfs: build dirs and big logs go under
  `target/` on /home.
* Cap heavy runs with `systemd-run --user --scope -p MemoryMax=16G`, never
  `ulimit -v` (it breaks the CubeCL CPU runtime).
* The only GPU here (Radeon 860M, ROCm at `~/rocm/opt/rocm`) has no f64: the
  f64 drivers are validated on the CubeCL CPU runtime; every kernel is
  validated on the GPU in f32 against the CPU runtime (bit for bit with
  contraction off); f64 GPU speed (G1) cannot be measured locally.
