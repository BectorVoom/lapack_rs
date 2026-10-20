# Device speed plan (CUDA first, then the CubeCL CPU runtime)

Measurement machine: Google Colab, **Tesla T4** (40 SMs, f64 1/32 rate, CUDA
12.8, driver 580.82.07), 2 vCPU host, `crates/cubecl-cuda` patched
(`--fmad=false`, f64 registered). Every number below is `min` over the stated
passes, warm (the JIT call is excluded), and every device result in every run
was checked bit-for-bit against Reference-LAPACK 3.12.1.

Methodology follows the CubeCL manual: attribution before optimisation
(16 §5.1 fix hierarchy), one variable per measurement, interleaved candidates,
re-profile after every change.

## 1. Status: landed

### 1.1 Launch geometry — cube count sized to the SM count (landed)

`geometry::launch_1d` asked cubecl for `CubeDim::new(client, units)`. That
helper caps at 8 planes and takes the cube *as large as possible*, which is
right for an element-wise kernel with millions of units and wrong for this
crate's `O(n)`-unit kernels. `ncu` on `symv_col_kernel_f_f64` at `m = 1021`:

```
Block (32, 8, 1) = 256 units      Grid (4, 1, 1) = 4 cubes
sm__throughput                    8.21 %
gpu__time_duration                574,816 ns
```

Four cubes on a 40-SM GPU: **36 SMs idle**. Sizing the cube so the *count*
reaches `SMs · CUBES_PER_SM` instead spreads the same units over the whole
device; kernels that already have enough units keep cubecl's 8-plane cube
(the helper's value is used as the ceiling, never exceeded).

```
Block (32, 1, 1) = 32 units       Grid (32, 1, 1) = 32 cubes
sm__throughput                    23.51 %      dram__throughput  28.24 %
gpu__time_duration                200,768 ns   (2.86x)
```

This cannot change a value: every kernel's units are element-disjoint, so the
geometry is free (`tests/device_launch_parity.rs`, and the CUDA gate passes
unchanged). `LAPACK_RS_CUBES_PER_SM` overrides the target (`0` restores
cubecl's default) for benchmarking.

Whole-solve effect, controlled A/B (same binary, `LAPACK_RS_CUBES_PER_SM=0`
vs `4`, alternating, 2 passes of `--passes 2|3`):

| case | cubecl default | SM-sized | speedup |
|---|---|---|---|
| `dsyevd` n=1024 | 0.3525 s | 0.2753 s | **1.28x** |
| `dsyevd` n=2048 | 1.5751 s | 1.2435 s | **1.27x** |
| `dsyevd` n=4096 | 8.3802 s | 7.5808 s | **1.11x** |
| `zheevd` n=2048 | 5.8182 s | 3.9415 s | **1.48x** |

Per-kernel device time (`CUBECL_DEBUG_OPTION=profile`, n=2048, 3 solves):

| kernel | before | after |
|---|---|---|
| `SymvColKernel` (K1) | 1410.5 ms (41.5%) | 712.2 ms (28.4%) |
| `Syr2kKernel` (K2) | 652.9 ms | 660.8 ms |
| `SecularRootsKernel` (K4) | 317.7 ms | 197.0 ms |
| `GemmTKernel` (K3) | 302.7 ms | 264.7 ms |
| `MergeStagedKernel` | 272.6 ms | 267.5 ms |
| `GemmNKernel` | 244.8 ms | 240.7 ms |
| **total GPU** | **3.396 s** | **2.507 s** |

The target value itself is flat: 1, 2, 4, 8 and 16 cubes/SM all land within
run-to-run noise (n=2048: 1.2414–1.3020 s), so `CUBES_PER_SM = 4` is kept.
Only `0` (cubecl's default) is slow. Do not re-tune it.

### 1.2 Measured and rejected: strip-mining K1's chain

K1col is latency-bound, not throughput-bound, so per-unit memory-level
parallelism looked like the next lever: issue `STRIP = 4` loads before the
four accumulations, keeping the additions in the reference's ascending order
(bit-exact by construction — it was, `bits ok` at every size).

It is **slower at every size** (`examples/device_kernel_bench`, interleaved,
min of 3, n=2048):

| m | K1col | K1col + strip |
|---|---|---|
| 512 | 0.108 ms | 0.128 ms |
| 1024 | 0.228 ms | 0.254 ms |
| 2047 | 0.485–0.683 ms | 0.535–0.673 ms |

Reverted. NVRTC already schedules the simple loop at least as well, and the
strip costs registers.

### 1.3 Measured and rejected: vectorised column loads in K1 (K1colv)

The other half of §2.2's idea: read `M`'s column through the `Vector<F, N>`
view (one 16-byte request per two steps instead of two 8-byte ones), with a
scalar prologue to the `(off + q) % N == 0` boundary and a scalar tail, lanes
consumed in ascending order. Bit-identical by construction and in fact — it
passes the CPU-runtime parity gate at lane width 8, which exercises the
prologue, the vector body and the tail.

On the T4 it is **2.2x slower** at every size, despite halving the load
instruction count (`examples/device_kernel_bench`, interleaved, min of 3):

| m | K1col | K1colv |
|---|---|---|
| 512 | 0.092 ms | 0.181 ms |
| 1024 | 0.207 ms | 0.447 ms |
| 2047 | 0.424 ms | 0.902 ms |

Kept as mode 3 (a tuning knob and parity coverage, like K2r/K3r) but **off by
default**. Together with §1.2 this closes the "make K1's loads wider or more
numerous" family: both directions were measured and both lose on a T4. K1's
remaining headroom, if any, is not in its load schedule.

### 1.4 Staging on the device (landed)

`dsyevd`/`zheevd` expanded the caller's triangle into the full symmetric
`lda × n` image with an O(n²) host loop, then uploaded it. The staging kernel
that the device-resident entry points already used does the same mirroring on
the device, so the host loop is gone and the upload is `n²` reals instead of
`lda·n`. A pure copy — the image is bit-identical.

| n | stage before | stage after |
|---|---|---|
| 1024 | 22.8 ms | 15.6 ms |
| 2048 | 103.9 ms | 86.9 ms |
| 4096 | 476.1 ms | 370.0 ms |

Worth 1.065x on the whole n=4096 solve and 1.04x on `zheevd` at n=2048.

## 2. Ranked remaining candidates

Ordered by measured payoff. Re-profile after each — every one of these moves
the bottleneck.

### 2.1 R1 — `DLATRD`'s host residue and the per-column round trip

Largest single item. `trd` at n=4096 is 4147 ms of the 7.64 s solve, and
**3832 ms of that is `read_wait`**: `DeviceTrd::symv` uploads `x`, launches
K1 and then *blocks* reading `y`, once per column, `n` times per solve. At
n=2048 the round trip costs ~246 µs per column against ~117 µs of K1 device
time, i.e. roughly **130 µs per column is pure synchronisation** (~266 ms per
solve).

Fix: move the rest of `DLATRD` (the two `DGEMV` corrections, `DLARFG`, the
scalings) onto the device so the whole panel runs without a host sync. This
removes `n` synchronisation points *and* the ~1.1 s of host `DLATRD` work at
n=4096 measured on the 2-vCPU host. It is the only remaining big `trd` win
and it is a transcription job, not a tuning job: each column's chain is
already specified by the host code the device engine shares.

Risk: `DLARFG` is a sequential reduction over the column; it must stay one
reference-ordered chain, so it becomes a one-unit kernel or a plane reduction
with the reference's operand order.

### 2.2 R2 — K1's sector efficiency (attempted; both directions lost)

After §1.1, K1 is still 28.4% of device time at `sm__throughput` 23.5% /
`dram__throughput` 28.2% — neither pipe is saturated, and L1 sees
1,184,301 sector requests for 8.3 MB of useful data (**4.6x**): a unit issues
one 8-byte scalar load per step against a 32-byte sector.

This was attempted from both ends and **both lost on a T4**: more loads in
flight (§1.2, 1.1x slower) and wider loads (§1.3, 2.2x slower). The
useful-bytes roofline is ~71 ms per solve at n=2048 against ~237 ms today, so
the headroom is real, but it is not reachable by changing the load schedule
of a kernel that only has `m` units. Anything further here needs either more
parallelism than `m` rows — which reference chain order forbids — or a
different algorithm. Treat this entry as closed unless the hardware changes.

### 2.3 R3 — `DORMTR`'s host packing

Downgraded after attribution. The host span `orm blocks` (1177 ms at n=4096)
looked like the phase's cost, but host spans are measured **without a sync**
and overlap the device: at n=2048 the `orm` phase is 189 ms against 185 ms of
`GemmT`+`GemmN`+`TrmmChain` device time, so the phase is GPU-bound and the
host packing is already hidden. Keeping the reflector panel resident would
still remove the **641 MiB** a n=4096 solve uploads (~50–100 ms of PCIe), but
that is the whole prize, not 1.2 s. Do R1 first.

### 2.4 R4 — `Syr2k` (K2): leave alone

`ncu`: `sm__throughput` **82.12%**, `dram__throughput` 0.90% — compute-bound
and close to the T4's f64 ceiling. There is no headroom here; the row-tiled
variant K2r is already known to be 1.1–1.8x slower.

### 2.5 R5 — `stedc`: `MergeStaged` + `SecularRoots`

89 ms + 66 ms per solve at n=2048 after §1.1 (the secular kernel already
gained 1.6x from the geometry fix). Profile before touching.

## 3. The CPU path: rayon deleted, CubeCL CPU runtime by default (landed)

Done 2026-09-17. `rayon` and `src/par.rs` are gone, `cubecl` is a
non-optional dependency and `default = ["cpu"]`. `cpu`, `rocm` and `cuda`
select only the *runtime*; there is one kernel set.

* `dsyevd_alloc` / `zheevd_alloc` run the device drivers on
  `runtime::DefaultRuntime` (`cuda` > `rocm` > `cpu`) with a per-thread cached
  client and `DeviceWorkspace`; `runtime::release_workspace()` gives the
  device buffers back.
* `dsyevd` / `zheevd` (explicit `LWORK`) stay the **sequential** reference
  transcription. Their bits depend on the workspace lengths the caller passes
  — the reference derives its block sizes from `LWORK` — and the device
  drivers do not model that; they reproduce the reference at
  `syevd_opt_work_len`. `dsyevd_alloc_host` / `zheevd_alloc_host` expose that
  path with the opt workspace and are what the `_alloc` entry points are
  gated against, bit for bit.
* The three `par::` call sites became the whole range (`blas.rs`, `zblas.rs`)
  and two sequential recursive calls (`stedc.rs`'s D&C halves). Every host
  oracle gate still passes, so the transcription is unchanged.

### 3.1 The regression this costs

Measured before the deletion, 16-core host, bit-identical either way:

| n | rayon host path | CubeCL CpuRuntime | |
|---|---|---|---|
| 256 | 0.0067 s | 0.0124 s | 0.54x |
| 512 | 0.0492 s | 0.0617 s | 0.80x |
| 1024 | 0.2876 s | 0.3939 s | 0.73x |

The user was shown these numbers and chose to delete rayon anyway. The gap is
**not** launch geometry — sweeping the CPU cube dim over 1/4/16/32/64/128 at
n=1024 moves the total between 0.374 s and 1.13 s with the existing choice
already at the optimum (0.374 s at dim 16 vs 0.387 s automatic). It is the
`orm` phase: 185 ms of that 374 ms, because `DORMTR` replays `DLARFB` as seven
element-wise reference-order passes. That decomposition exists to keep each
element's chain in reference order on a GPU; on a CPU it loses to ordinary
blocked BLAS, which is what the deleted host path used.

Closing it means a second decomposition for a runtime with no planes — which
is the thing the one-kernel-set decision was meant to avoid. It is left open
deliberately, and is the first thing to revisit if the CPU default matters.

## 4. Where this stands against LAPACK

`dsyevd`, Vectors, T4, every value bit-identical to Reference-LAPACK 3.12.1:

End of the 2026-09-16/17 session, `--no-default-features --features cuda`,
min over 2 interleaved passes:

| n | lapack_rs device | at session start | Reference-LAPACK (1 thread) | faer |
|---|---|---|---|---|
| 1024 | **0.275 s** | 0.353 s (1.28x) | 2.261 s (**8.2x**) | 0.434 s |
| 2048 | **1.291 s** | 1.575 s (1.22x) | 18.613 s (**14.4x**) | 2.138 s (1.66x) |
| 4096 | **7.118 s** | 8.380 s (1.18x) | 150.4 s¹ (**21.1x**) | 16.559 s (2.33x) |

`zheevd` n=2048: **3.776 s**, from 5.818 s at session start (**1.54x**), against
faer 4.780 s.

¹ n=4096 Reference-LAPACK is the 2026-09-16 measurement on the same VM; the
runs above cap the single-threaded oracle at n=2048 to keep a pass under an
hour.

The Reference-LAPACK and faer columns were measured in the same process as an
earlier build; on a 2-vCPU VM running the host solvers in-process disturbs the
device timing at small `n`, so the device column above is from runs with
`--no-host --no-faer`. The ratios are therefore indicative, not same-process,
except where §1.1 says otherwise.

## 5. Gates

Every change in this document is gated by, in addition to the standing host
gates in `AGENTS.md` §5:

```
export LAPACK_RS_REF_DIR=$PWD/target/ref-lapack/install LAPACK_RS_REQUIRE_REF=1
cargo test --release                       # default = CubeCL CPU runtime
NVRTC_COMPILE_OPTIONS_APPEND=--fmad=false CUDA_PATH=/usr/local/cuda \
  cargo test --release --no-default-features --features cuda --test device_cuda
```

`tests/threads.rs` is the contract for §3: `dsyevd_alloc`/`zheevd_alloc` on
the default runtime against `dsyevd_alloc_host`/`zheevd_alloc_host`, bit for
bit, at every launch geometry.

`tests/device_cuda.rs` refuses to run without a no-contraction probe, so a
green run is never vacuous. A speed claim is made only from
`examples/device_profile` (warm, `--passes >= 2`, interleaved against the
alternative) and attributed with `CUBECL_DEBUG_OPTION=profile` plus `ncu`
before any kernel is changed.
