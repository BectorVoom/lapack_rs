# cubecl-cpu 0.10: a per-function MLIR lowering limit

*Reported following AGENTS.md §4. **No document in
`/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/`
covers this error** — the guide has three entries (calling a normal Rust
function from a `#[cube]` body, `ArrayArg::from_raw_parts` outside `unsafe`,
and `mismatched types` from an `if` expression), none of which applies. This
document follows the manual's template (Problem Description → Root Cause →
Recommended Solutions → Best Practices → Summary) so it can be folded into the
guide.*

## Problem description

Launching a `#[cube(launch, launch_unchecked)]` kernel on `CpuRuntime` prints

```
error: operation with block successors must terminate its parent block
thread 'DSD-0-0' panicked at cubecl-cpu-0.10.0/src/compiler/module.rs:94:13:
failed to run pass
```

on a cubecl worker thread. **The launching thread does not see the panic**: the
launch silently no-ops and the output buffers keep whatever they held, so the
first symptom on the host side is a result of all zeros, not a build or launch
error. That is the same silent-failure shape AGENTS.md records for an
unpatched `cubecl-opt` (`src/device/probe.rs::k4_supported` exists because of
it).

Reproduced on:

* kernel: `src/device/kernels/cint_stemr.rs`'s `cint_dlasq2_c`,
  `cint_compute_eigenvalues_c` (which inlines it) and
  `cint_compute_eigenvectors_c`,
* runtime: `cubecl-cpu` 0.10.0 with this repo's vendored `cubecl-opt`,
* element type: `f64`, `CubeCount::Static(1,1,1)`, `CubeDim::new_1d(1)`,
  both `launch` and `launch_unchecked`.

and on `cint_dlasq5_c` and `cint_dlarrb_c`, which are small.

`cint_dlarrk_c`, `cint_dlarrc_c`, `cint_dlaneg_c`, `cint_dlarrf_c`,
`cint_dlar1v_c`, `cint_dlaev2_c` and `cint_dlasq4_c` — the same file, the same
idioms, and in `cint_dlasq4_c`'s case a much larger function — all compile and
run correctly, and are bit-identical to the host
(`tests/device_cint_stemr_cpu.rs`).

## Root cause

**A control-flow lowering bug in cubecl-cpu 0.10, not a language construct and
not (only) a size threshold.** The exact trigger is *not* identified; what is
established is below. Do not spend another bisect on it without new
information — record the affected kernel and move it to a GPU gate.

The evidence:

1. **It is not a pure size threshold.** `cint_dlasq5_c` is ~40 lines with one
   `while` loop and no nesting, and it fails; `cint_dlasq4_c`, which is five
   times larger with six branches and four loops, compiles and is bit-exact.
2. **It is not any single statement.** Stubbing `cint_dlasq5_c`'s only loop
   makes it compile, but removing any individual statement from that loop
   (`if diag < tol`, the division that forms `temp`, the `e1p` store, the
   `fmin_c` call) does not. Likewise, stubbing *any one* of five unrelated
   loops inside `cint_dlasq2_c` makes that compile — including one of the two
   independent "dqd maps Z to ZZ" setup loops, which shares no control flow
   with the rest.
3. **It is not variable shadowing.** Rewriting `cint_dlasq5_c`'s two unrolled
   tail steps to use fresh names instead of shadowing the loop's `j`/`temp`
   changes nothing (the rename is kept anyway, as it reads better).
6. **The idioms are individually fine.** A compound short-circuit loop
   condition (`while i < n && !stop`), a sentinel exit (`counter = bound`),
   nested `while` inside `while`, and a `#[cube]` callee containing its own
   loops all compile and run correctly in isolation (probed directly) and in
   the nine leaf kernels above.
4. **It is not the decremented-counter limitation either.** `cint_dlasq2_c`
   originally decremented three variables (the reversal index `j`, the setup
   index `j`, and `n0` itself, which drives both the outer and the deflation
   loop). All three were rewritten ascending — `j` as `(n-1) - i`, and `n0` as
   `n - ndef` with `ndef` ascending — which is required by AGENTS.md's device
   rules anyway, and the error is unchanged.
5. **Another backend compiles the same kernel.** hipRTC lowers the *whole*
   composite (`cint_stemr_dev`, which inlines all eleven routines) and
   launches it, under `--no-default-features --features rocm` with
   `HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off`. So the kernel is
   well-formed CubeCL; the CPU backend's lowering is what gives out.

The failing message is an MLIR **verifier** error raised from
`PassManager::run` (`cubecl-cpu-0.10.0/src/compiler/module.rs:94`), i.e. the
IR reaching or leaving one of the passes (`control_flow_to_llvm`,
`func_to_llvm`, `inliner`, `reconcile_unrealized_casts`, `sccp`, `mem2reg`,
`control_flow_sink`, `cse`) has a branch-like operation that is not its
block's terminator. The size dependence points at block splitting during
lowering rather than at the CubeCL front end.

This is a **third**, distinct cubecl-cpu 0.10 limitation, alongside the two
AGENTS.md already records (a `while` loop with a **decremented** counter,
worked around by ascending counters; and the `cubecl-opt` PRE/GVN
`no entry found for key` ICE, fixed by vendoring `crates/cubecl-opt`). Unlike
the PRE/GVN one it is **not** fixed by the vendored `cubecl-opt` — it is in
`cubecl-cpu` itself.

## Recommended solutions

In order of preference:

1. **Find out early which of your kernels the CPU runtime accepts.** Launch
   each `#[cube]` leaf once from a probe with stderr captured, and check both
   for `failed to run pass` **and** that the output buffer actually changed;
   `eprintln!` the results, because stdout buffering will otherwise interleave
   them with MLIR's stderr and make the bisect unreadable. Value-preserving
   index hoisting reduces the IR and costs nothing numerically, but it did not
   help here.
2. **Validate that kernel on a GPU backend instead.** hipRTC and NVRTC lower
   it. For this crate the f64 bit-exactness gate for such a kernel is CUDA
   (`tests/device_cuda.rs`), with ROCm f32 for the shape/consistency gate; the
   CPU runtime keeps the *leaf* kernels, which is where every reference
   operand order actually lives.
3. **Never let a silent no-op stand in for a result.** A runtime capability
   probe, like `device::probe::k4_supported`, turns "the launch produced
   zeros" into "fall back to the host, bit-exact either way". Any production
   path through a kernel near this threshold needs one.

Do **not** "fix" it by splitting the `#[cube]` callee into smaller `#[cube]`
functions: they are expanded into the caller, so the lowered function is the
same size. Splitting into separate *launch* kernels does reduce per-function
size, but halving was not enough here (`cint_compute_eigenvalues_c` and
`cint_compute_eigenvectors_c` each still fail on their own).

## Best practices

* Treat a `CpuRuntime` launch that returns zeros as a **compile failure until
  proven otherwise**: the panic is on a worker thread and never reaches the
  caller. Run the launch once with stderr captured and grep for
  `failed to run pass`.
* Gate the leaves on the CPU runtime and the composite on a GPU. The leaves
  are where the reference operand order, the `MAX`/`MIN` lowering and the
  zero-skips live, so leaf-level parity is most of the bit-exactness value.
* Record the limit as a regression test so a cubecl bump that lifts it is
  noticed: `tests/cubecl_cpu_limits.rs::cint_dlasq2_exceeds_cpu_runtime_
  lowering_limit`, `#[ignore]`d like the descending-counter one.

## Summary

| | |
|---|---|
| **Symptom** | `error: operation with block successors must terminate its parent block`, then `failed to run pass` on a cubecl worker thread; the launch silently no-ops and the caller reads zeros. |
| **Trigger** | Not identified. Affects `cint_dlasq5_c`, `cint_dlarrb_c`, `cint_dlasq2_c` and the two composites; `cint_dlarrk_c`, `cint_dlarrc_c`, `cint_dlaneg_c`, `cint_dlarrf_c`, `cint_dlar1v_c`, `cint_dlaev2_c`, `cint_dlasq4_c` all lower and are bit-exact. |
| **Not the cause** | Function size alone, any single statement, variable shadowing, decremented counters, compound loop conditions, sentinel loop exits, nested loops, `#[cube]` callees with loops, or the vendored `cubecl-opt`. |
| **Confirmed elsewhere** | hipRTC compiles and launches the identical kernel. |
| **Workaround taken** | Leaf kernels gated on the CPU runtime (f64, bit-exact vs host); the composite gated on GPU backends. |
| **Regression test** | `tests/cubecl_cpu_limits.rs`, `#[ignore]`d. |
