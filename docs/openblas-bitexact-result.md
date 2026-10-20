# OpenBLAS bit-exact track: result

Task O5 of `docs/openblas-bitexact-plan.md`. Everything below was measured on
the dev machine (AMD Ryzen AI 7 350, 16 threads, AVX-512) against
`pyscf_rs/.venv/lib/python3.13/site-packages/scipy.libs/libscipy_openblas-6cdc3b4a.so`
(`OpenBLAS 0.3.30 DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=64`, pinned with
`OPENBLAS_CORETYPE=SkylakeX OPENBLAS_NUM_THREADS=1`).

**Verdict: `max_ulp = 0`.** `lapack_rs::openblas::{dsyevd, zheevd}` (CubeCL
device drivers, default runtime) and `lapack_rs::openblas::{dsyevd_host,
zheevd_host}` (sequential host transcription) reproduce `scipy_dsyevd_` /
`scipy_zheevd_` called with their own workspace query **bit for bit**: every
eigenvalue, every eigenvector entry (both parts), `INFO`, both `UPLO`, both
`JOBZ`, over the oracle size grid up to `n = 1024` and the dense fixtures
(identity, diagonal, block diagonal, `1e±160` scaled).

The path taken is **not** the plan's recommended Path O-FFI (link OpenBLAS at
run time). By the user's decision the OpenBLAS arithmetic was transcribed —
Path O-Reimpl — and the heavy kernels run as CubeCL kernels on the CPU
runtime, so the crate stays "CubeCL only" and needs no OpenBLAS at run time.
OpenBLAS is linked only by the test-only oracle crate `crates/openblas_ref`.

## 1. What O0 measured

Before any change, lapack_rs (Reference-LAPACK-exact) vs OpenBLAS,
`tests/openblas_o0_differential.rs` (report only):

| n | `dsyevd` w same | w max ULP | Z same | Z max ULP |
|---|---|---|---|---|
| 1, 2 | 100% | 0 | 100% | 0 |
| 3 | 0–33% | 4 | 22–56% | 16 |
| 16 | 6–19% | 96 | 1–2% | 3 078 |
| 64 | 8–16% | 141 | 1% | 40 336 |
| 129 | 19–23% | 123 | 1% | 4 194 560 |
| 257 | 16–20% | 158 | 0.3% | 25 184 308 |

Component level (same test), OpenBLAS differs from the reference BLAS in
almost every routine and size: `ddot` from `n = 2`, `dsyr2` and `dgemv('T')`
from `n = 1`, `daxpy`, `dnrm2`, `dsymv`, `dgemm`, `dsyr2k`, `dtrmm`. `dscal`
was the only exact one.

The LAPACK layer is **not** where the difference is: OpenBLAS 0.3.30 vendors
Reference-LAPACK 3.12.1 unchanged (a whitespace/continuation-normalised diff
of every routine in the `DSYEVD`/`ZHEEVD` call tree is empty) except
`DORM2L`/`DORM2R`/`ZUNM2L`/`ZUNM2R`, which still call `DLARF`/`ZLARF` (3.12.0)
instead of 3.12.1's `DLARF1L`/`DLARF1F`. The disassembly shows that Fortran is
compiled with scalar SSE2 and no FMA. So O1's decision was: keep the one
LAPACK transcription, swap the BLAS.

## 2. What was built

### 2.1 One LAPACK transcription, two BLAS flavours

`src/flavor.rs` introduces `trait Blas` (every arithmetic BLAS call of the
`DSYEVD`/`ZHEEVD` call tree) with `RefBlas` (the reference BLAS, the crate's
default contract) and `LEGACY_DLARF`. The host engines (`HostTrd`, `HostDc`,
`HostOrm`, `HostZTrd`, `HostZOrm`) and the device engines are generic over it;
`dsyevd_raw::<B>` / `zheevd_raw::<B>` are the drivers. Pure moves (`DCOPY`,
`DSWAP`, `ZCOPY`, `ZSWAP`) are identical in every flavour and stay direct
calls. The Reference gates were re-run after the refactor: unchanged.

### 2.2 The OpenBLAS BLAS, transcribed from the binary

`src/openblas/` (`OpenBlas: Blas`), one module per family, each bit-exact
against the library itself (`tests/openblas_*_bitexact.rs`, every output bit
including untouched triangles, padding rows and stride gaps):

| module | routines | notable arithmetic |
|---|---|---|
| `level1` | `ddot daxpy dscal dnrm2 dcopy dswap drot idamax` | `dnrm2` is the x87 `nrm2.S`: four 80-bit extended accumulators — unit stride runs the same instruction sequence on the x87 FPU (`asm!`), other strides an exact software emulation; `dscal(0)` multiplies (NaN stays NaN); `idamax` depends on `x`'s address mod 16 |
| `level2` | `dgemv dsymv dsyr2 dger` | 4-column fused kernels, 2048-row blocks, row-parity lanes, region-dependent row formulas in `dsymv` |
| `level3` | `dgemm dsyr2k dtrmm` | small-matrix kernels (`MNK <= 1e6`), `GEMM_P/Q/R = 192/384/10704` read from the live `gotoblas` table, 16×2 micro-kernel with 2/4-lane splits, TRMM windows |
| `zlevel1` | `zdotc zaxpy zscal zdscal dznrm2 zcopy zswap` | `dznrm2` x87 too; `zscal` NaN branches by position |
| `zlevel2` | `zhemv zgemv zher2 zgerc` | `zhemv` is SSE2 assembly (no FMA, column pairs); `zgemv` 1024-row blocks, even/odd lanes; the undocumented `vaddsubpd` opcode |
| `zlevel3` | `zgemm zher2k ztrmm` | 4×2 complex kernel: 4-row groups keep four fused sums, 2/1-row groups fold |

Rules that decided the bits (and are easy to get wrong): gcc contracted
ordinary C `a*b + c` and intrinsic mul/add pairs into `vfmadd*`, so the
transcription uses `f64::mul_add` exactly where the binary has a fused
instruction and nowhere else; lane reductions have fixed trees; interface
shortcuts (`alpha == 0`, `beta != 1` pre-scaling, small paths) are
transcribed; `DYNAMIC_ARCH` drivers were compiled without the `SKYLAKEX`
branch (column chunks 6/2, not 12). **This track is the one place FMA is
allowed** (AGENTS.md §5). The host modules are compiled with
`target_feature(enable = "fma")` so `mul_add` is one instruction; every public
entry point checks the CPU first.

Another thing that decides bits: **the workspace.** scipy passes `DSYEVD`'s
query `LWORK`, which leaves `DORMTR` unblocked for `n < 82` and `ZUNMTR`
always unblocked (`LLWRK2 = N`); `openblas::*` reproduce exactly that
(`syevd_work_len`, `heevd_work_len`), not lapack_rs's blocked
`*_opt_work_len`.

### 2.3 OpenBLAS-order device kernels (the CPU path)

`src/device/flavor.rs` (`trait DeviceBlas`) selects the kernels the hybrid
drivers launch; `dsyevd_device_impl::<R, B>` / `zheevd_device_impl::<R, B>`
are shared with the Reference track. New kernels, all generic
`F: Float + CubeElement`, launched through `kl!`, each gated bit for bit
against the host OpenBLAS transcription on the CubeCL CPU runtime
(`tests/device_openblas_cpu.rs`):

| kernel | replaces | unit / parallelism |
|---|---|---|
| K1ob `ob_symv_row_kernel` | `dsymv` in `DLATRD` | one unit per row: each row is a chain over column groups plus its own column sum, independent of other rows |
| K2ob `ob_syr2k_vec_kernel` + `ob_syr2k_list_kernel` | `dsyr2k` in `DSYTRD` | lane-vectors over rows for plain-chain elements; the diagonal `32×32` scratch pieces and 2/4-lane tiles (closed-form intervals, `ops::ob_syr2k_special_intervals`) as a scalar element list |
| K3m-ob `ob_merge_vec_kernel` + `ob_merge_kernel` | `DLAED2` gather + `DLAED3` `dgemm` | lane-vectors for plain rows, scalar for tail-tile / dot-style rows |
| K7rot-ob `ob_rotations_kernel` | `drot` | lane-vectors, `fma(c,x,s·y)`, `fma(y,c,−s·x)` |
| `ob_trmm_kernel` | `dtrmm('R')` in `DLARFB` | one unit per element; tile tables and packed triangle built on the host |
| `ob_gemm_tn_vec_kernel` / `ob_gemm_nt_vec_kernel` (+ scalar lists) | `dgemm('T','N')`, `dgemm('N','T')` in `DLARFB` | lane-vectors over columns / rows |
| K1c-ob `ob_hemv_row_kernel` | `zhemv` in `ZLATRD` | one unit per row (column-pair chains), `T1 = alpha·x` precomputed |
| K2c-ob `ob_her2k_vec_kernel` + `ob_her2k_list_kernel` | `zher2k` in `ZHETRD` | as K2ob, 4-row groups, `4×4` diagonal pieces |
| `ob_zlastc_kernel`, `ob_zgemv_c_kernel`, `ob_zgerc_vec_kernel` | the legacy `ZLARF` of `ZUNM2L/R` | device `ILAZLC` (no host sync per reflector), per-column `zgemv('C')`, lane-vector `zgerc` |

Kept on the host: `DLARFG`, the `DLATRD` panel `DGEMV`s, `DLAED3`'s secular
roots and eigenvector columns (their normalisation is the x87 `dnrm2`), and
`DLARFT` — all in the OpenBLAS flavour.

## 3. Speed

`examples/openblas_vs_device.rs`, `examples/openblas_zheevd_timing.rs`
(min of 3 warm passes; CubeCL CPU runtime on 16 threads; scipy's OpenBLAS on
one thread, as the oracle is pinned):

| `dsyevd`, n | Reference device | **OpenBLAS device** | OpenBLAS host transcription | scipy OpenBLAS (1 thread) |
|---|---|---|---|---|
| 256 | 0.011 s | 0.019 s | — | 0.004 s |
| 512 | 0.056 s | 0.062 s | — | 0.021 s |
| 1024 | 0.332 s | **0.308 s** | 1.66 s | 0.131 s |
| 2048 | 2.643 s | **2.044 s** | — | 0.968 s |

| `zheevd`, n | Reference device | **OpenBLAS device** | OpenBLAS host transcription | scipy OpenBLAS (1 thread) |
|---|---|---|---|---|
| 256 | 0.017 s | 0.039 s | 0.039 s | 0.009 s |
| 512 | 0.084 s | 0.183 s | 0.264 s | 0.068 s |
| 1024 | 0.579 s | **0.979 s** | 2.08 s | 0.566 s |

How the real `n = 1024` solve got from the first bit-exact version to parity
(each step re-gated bit for bit):

| step | total |
|---|---|
| host transcription (no FMA inlining) | 5.09 s |
| device `DSYTRD` (K1ob, K2ob) + `DSTEDC` merges; host `DORMTR` | 3.02 s |
| device `DLARFB` (scalar `trmm`/`gemm` kernels) | 0.79 s |
| host OpenBLAS modules compiled with FMA | 0.70 s |
| K3m-ob lane-vectors | 0.55 s |
| K2ob lane-vectors | 0.48 s |
| `DLARFB` `gemm` lane-vectors | 0.33 s |
| `dnrm2`/`dznrm2` fast path on the x87 FPU itself (`asm!`, same instructions as `nrm2.S`) | **0.31 s** |

Phase split at `n = 1024` (real): tridiagonalisation 0.16 s, `DSTEDC` 0.07 s
(0.02 s of it host `DLAED3`), back-transformation 0.07 s — the
back-transformation is faster than the Reference track's (0.15 s) because the
OpenBLAS `dgemm` store is a single fused add per element.

## 4. Limits and open items

* **Pinned binary.** The track reproduces *this* library on *this* core. A
  different `CORETYPE`, thread count or scipy/OpenBLAS release is a different
  target; `openblas_ref::require` refuses to compare unless the config
  string, the core name and `OPENBLAS_NUM_THREADS=1` match, and
  `gotoblas_parameters_match_live_table` fails if the cache-derived block
  sizes move.
* **x87 `dnrm2`.** K4 (device `DLAED3`) normalises with the reference
  `dnrm2.f90`, so under the OpenBLAS flavour `DLAED3` runs on the host
  (`DeviceBlas::SECULAR_ON_DEVICE = false`; 0.02 s at `n = 1024` since the
  norms run on the x87 FPU). A device version needs an integer-limb 80-bit
  accumulator. The `asm!` fast path relies on the default x87 control word
  (64-bit precision, round to nearest), which
  `x87_control_word_is_default` asserts.
* **Complex tridiagonalisation.** K1c-ob (≈0.29 s) and K2c-ob (≈0.3 s) at
  `n = 1024` only match the FMA host `ZHETRD`; the next speed step is a
  lane-vector `zhemv` (rows of one parity share a formula).
* **Unblocked `ZUNMTR`.** scipy's query workspace forces the level-2
  back-transformation (0.33 s at `n = 1024` vs the Reference track's blocked
  0.27 s); that is the oracle's algorithm, not a choice.
* **Not covered:** NaN/Inf inputs at the whole-driver level; negative
  increments; `n > GEMM_R` (10 704) column blocking of `dtrmm('R')`.

## 5. Gates

```
export LAPACK_RS_OPENBLAS_DIR=/home/user/Documents/workspace/pyscf_rs/.venv/lib/python3.13/site-packages/scipy.libs
export OPENBLAS_CORETYPE=SkylakeX OPENBLAS_NUM_THREADS=1 OPENBLAS_REQUIRE_REF=1
cargo test --release --test openblas_level1_bitexact --test openblas_level2_bitexact \
  --test openblas_level3_bitexact --test openblas_zlevel1_bitexact --test openblas_zlevel2_bitexact \
  --test openblas_zlevel3_bitexact --test openblas_zlevel3_components --test openblas_lapack_components \
  --test openblas_syevd_bitexact --test openblas_heevd_bitexact
cargo test --release --test device_openblas_cpu --test openblas_device_syevd --test openblas_device_heevd
LAPACK_RS_MAX_N=1024 cargo test --release --test openblas_device_syevd --test openblas_device_heevd
```

The first group needs the library (without `LAPACK_RS_OPENBLAS_DIR` it prints
`SKIP`, with `OPENBLAS_REQUIRE_REF=1` it fails); the second compares the
device kernels and drivers with the host transcription and needs nothing.
`tests/openblas_o0_differential.rs` is the O0 report (it needs both oracles
and never gates).
