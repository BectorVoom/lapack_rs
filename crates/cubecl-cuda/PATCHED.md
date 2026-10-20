# Vendored cubecl-cuda 0.10.0 (lapack_rs patch)

Verbatim copy of `cubecl-cuda` 0.10.0 from crates.io with two changes:

1. `src/compute/context.rs`: NVRTC options are extended with the
   whitespace-split contents of `NVRTC_COMPILE_OPTIONS_APPEND`.
2. `src/runtime.rs`: `f64` is registered as a supported type. cubecl-cpp
   0.10.0 comments it out of the shared list because of a matmul error, so
   upstream every CUDA device reports no f64; lapack_rs never uses matmul.

NVRTC defaults to `--fmad=true`, which contracts `a * b + c` into a fused
multiply-add and makes device results differ from the unfused
Reference-LAPACK chain. Upstream 0.10.0 hard-codes its NVRTC options, so there
is no other lever (the ROCm analogue is `HIPRTC_COMPILE_OPTIONS_APPEND`).

Bit-exact runs set `NVRTC_COMPILE_OPTIONS_APPEND=--fmad=false`; the
`lapack_rs::device::probe::contraction_probe` confirms it on the live device.

## Caveat for publishing

`lapack_rs` wires this in with `[patch.crates-io]`, which applies to builds
**from this workspace only**: a consumer who depends on a published
`lapack_rs` and enables the `cuda` feature would get upstream
`cubecl-cuda` 0.10.0, whose CUDA devices report no f64 (so every f64 entry
point returns `DeviceUnsupported`) and whose NVRTC contracts into FMA. Until
the fixes are upstream, the `cuda` feature is for use from this repository,
and the contraction probe plus `tests/device_cuda.rs` are what prove a given
build is exact.
