# lapack_rs

A divide-and-conquer symmetric/Hermitian eigensolver (`dsyevd`, `zheevd` and
their building blocks) transcribed from **Reference-LAPACK 3.12.1** so that
every output is **bit-identical** to the reference library built with pinned
flags. Everything runs on CubeCL: the CPU runtime by default, ROCm or CUDA
behind a feature. There is one kernel set and no host parallel path.

```rust
use lapack_rs::{dsyevd, syevd_work_len, Job, Uplo};

let n = 3;
let mut a = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0]; // column-major
let mut w = vec![0.0; n];
let (lwork, liwork) = syevd_work_len(n, Job::Vectors);
let (mut work, mut iwork) = (vec![0.0; lwork], vec![0usize; liwork]);
dsyevd(Job::Vectors, Uplo::Lower, n, &mut a, &mut w, &mut work, &mut iwork)?;
# Ok::<(), lapack_rs::LapackError>(())
```

* Column-major storage, `LDA = n`, LAPACK workspace semantics: `work.len()` is
  `LWORK`, and the block sizes the reference derives from it are reproduced.
  `*_work_len` returns LAPACK's query value, `*_opt_work_len` the size at which
  every stage runs blocked, `*_alloc` allocates once at that size.
* The drivers allocate nothing.
* Complex matrices are planar (`a_re`, `a_im`); `zheevd`'s complex `WORK` is
  planar too (`work.len() = 2*LWORK`), with `RWORK` and `IWORK` as in LAPACK.

## On-device path

CubeCL kernels generic over `F: Float + CubeElement` and the runtime `R`,
driven by the same host transcription (MAGMA-style hybrid: matrix-sized data
stays on the device, `DLARFG`/`DLAED2`–`DLAED4`/`DLARFT` stay on the host).

```rust,ignore
use lapack_rs::device::{DeviceBudget, DeviceWorkspace, MergeU, dsyevd_device, dsyevd_device_resident};

let mut ws = DeviceWorkspace::new(&client, n, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
dsyevd_device(&client, Job::Vectors, Uplo::Lower, n, &mut a, &mut w, &mut ws)?;       // host slices
let z = dsyevd_device_resident(&client, Job::Vectors, Uplo::Lower, n, &a_handle, &mut w, &mut ws)?; // device buffers
```

* `dsyevd_device`/`zheevd_device` (host slices) and `*_device_resident`
  (input `n × n` device buffer; eigenvectors returned as a `DeviceMatrix` /
  `DeviceZMatrix`, `lda × n`, with `compact()` for `n × n`).
* `Exactness::Reference` (default): staged merge U and `DLARFB`'s own step
  sequence — bit-identical to Reference-LAPACK on a backend whose
  contraction probe reports none (the CubeCL CPU runtime; CUDA with
  `NVRTC_COMPILE_OPTIONS_APPEND=--fmad=false`; ROCm/HIP with
  `HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off`).
  `Exactness::Fast`: fused back-transform and, with `MergeU::Lean` (or `Auto`
  when staged U exceeds the `DeviceBudget`), a merge that never materialises
  U — tolerance-gated.
* f64 is required: a device without it returns `LapackError::DeviceUnsupported`.
* Device memory: `lda·n` resident plus O(NB·lda) scratch, plus ≤ 2k² per merge
  when staged (measured peak 2.03·n² at n = 1024) or O(n) lean.
* Complex buffers are planar (two `lda × n` parts).

On a GPU the drivers pick the layouts that measured fastest there: `DSYMV`
and `ZHEMV` read each row as a column (both triangles hold identical bits,
and consecutive steps then load adjacent elements), `DLAED3` — the secular
roots, the Löwner products and the eigenvector columns — runs on the device
(K4) for merges of rank ≥ 64, and the simple K2/K3 kernels are used rather
than the row-tiled ones. Every choice is bit-identical to the others and is
covered by `tests/device_launch_parity.rs`.

```
cargo test --release                                  # CPU runtime, incl. device_ref_bitexact
# CUDA (CUDA_PATH set or /usr/local/cuda; an f64-capable NVIDIA GPU):
NVRTC_COMPILE_OPTIONS_APPEND=--fmad=false cargo test --release --no-default-features --features cuda --test device_cuda
# ROCm (hipconfig on PATH, ROCm lib on LD_LIBRARY_PATH):
HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off cargo test --release --features rocm \
  --test device_rocm --test device_rocm_vs_cpu
```

The `cuda` feature builds against `crates/cubecl-cuda`, a vendored
cubecl-cuda 0.10.0 with two fixes (an NVRTC options hook so FMA contraction
can be turned off, and f64 registered as a supported type); it is wired in
with `[patch.crates-io]`, so it applies to builds from this workspace — see
`crates/cubecl-cuda/PATCHED.md`.

`cubecl.toml` turns CubeCL's on-disk kernel cache on, and `build.rs` clears it
whenever a `#[cube]` source changes (the cache key does not cover the kernel
body). Without it every process recompiles its kernels: a cold `n = 1024`
solve on a T4 costs 7.4 s of NVRTC time, a warm one 0.58 s in total. Clear
`target/cuda` (or `target/hip`) by hand after changing a device compiler flag.

### Measured on a Tesla T4

f64, CUDA 12.8, 2 vCPU host, warm kernel cache, `Exactness::Reference` —
every device row is bit-identical to Reference-LAPACK 3.12.1:

| n | `dsyevd_device` | Reference-LAPACK 3.12.1 (1 thread) | SciPy `eigh` (OpenBLAS, 2 threads) | faer 0.24 (2 threads) |
|---:|---:|---:|---:|---:|
| 512 | 0.097 s | 0.29 s | — | 0.05 s |
| 1024 | 0.35 s | 2.24 s | 0.26 s | 0.31 s |
| 2048 | 1.60 s | 17.6 s | 1.72 s | 2.05 s |
| 4096 | 8.18 s | 150.4 s | 16.46 s | 15.25 s |

`zheevd_device` (bit-exact too): 0.28 s at n = 512, 1.21 s at n = 1024
(Reference-LAPACK 3.87 s) and 5.87 s at n = 2048 (34.4 s for the host path on
this VM, 4.8 s for faer). The host column is what a 2-vCPU VM can do — on a
many-core host it is far closer — and below n ≈ 1024 the per-column round
trips make the device path the wrong choice for real matrices.

```
cargo run --release --features cuda --example device_profile -- 1024,2048,4096
cargo run --release --features cuda --example device_kernel_bench -- 4096
```

## Bit-exact oracle

```
cargo xtask build-ref-lapack   # needs gfortran and cmake; SHA-256-verified download
export LAPACK_RS_REF_DIR=$PWD/target/ref-lapack/install LAPACK_RS_REQUIRE_REF=1
cargo test --release --test ref_oracle_canary --test ref_blas_bitexact \
  --test ref_lapack_components --test ref_syevd_bitexact --test ref_heevd_bitexact
```

## License

MIT OR Apache-2.0. The algorithms and their exact floating-point operation
order are those of Reference-LAPACK (BSD-3-Clause, © The University of
Tennessee and others), which is the algorithmic reference for this crate.
