# Implementation plan: LAPACK/BLAS coverage for cintx consumers

*Revision 2 (2026-09-17): speed-ordered, device-resident. Revision 1 is in
`git log` / `docs/` history; §0 lists what changed and why.*

> **Status, 2026-09-17 — the plan has been executed. §12 records what landed,
> what it is gated by, and the two things that are not done.**

## 0. What changed in revision 1 → 2

| # | Revision 1 | Revision 2 | Why |
|---|---|---|---|
| 0.1 | Task A (≈5 000 lines of MRRR dependency closure) is on the critical path to the adoption gate A3 | **A0**, a ≈1-day differential harness, decides A3 *before* any transcription | A3 as specified is provably unreachable — §4 names the six constant-level divergences between Reference-LAPACK 3.12.1 and libcint's vendored solver. R1 spent the whole budget before the decidable gate. |
| 0.2 | One deliverable (`DSTEMR`) serving both LAPACK completeness and cintx adoption | Two independent tracks: **Path R** (reference `dstemr`, oracle = Reference-LAPACK) and **Path C** (`cint_stemr`, oracle = libcint 6.1.3) | The two have different, mutually incompatible bit contracts. Conflating them is what made A3 look reachable. |
| 0.3 | Device deliverable = `#[cube(launch, launch_unchecked)]` launched from `ops.rs` through `kl!` | Device deliverable = an **inline `#[cube]` callee** (production) + a **batched kernel** (standalone) + a one-problem launcher (parity only) | For an `n ≤ 13` problem called once per shell quartet, a launch + read-back is ~10⁵× the arithmetic. §6 has the measured basis. The launcher form is the slow shape and cintx already knows it (`cint_diagonalize_dev`). |
| 0.4 | Tasks strictly serial A → A2 → A3 | Dependency graph with three concurrent fronts (§5) | B and C have no dependency on A at all and were needlessly queued behind it. |
| 0.5 | Kernels described in prose | **Full on-device kernel contents** (§7), transcribed line-by-line from the pinned sources, with the reference operand order, the gfortran lowerings and the two backend workarounds this repo has already paid for | The descending-counter rewrite (§7.3) and the ITMAX integer risk (§7.2) are not discoverable from prose; they are the two things that would have cost a debugging cycle each. |

## 1. Background

`cintx` (`/home/user/Documents/workspace/cintx`) reproduces a slice of
LAPACK/BLAS on the host today: one real-BLAS `dgemm_` call site (PySCF
`nr_ecp.c`, vendored under `cintx/vendor/pyscf-nr-ecp`) and one LAPACK
tridiagonal eigensolver role (libcint `eigh.c`, vendored under
`cintx/libcint-master`). The `dgemm` site is already migrated to this crate
(§2). Everything in §3 that this crate does not yet implement still runs on
cintx's own code. This plan lists that missing functionality as transcription
tasks following this repo's standing rules (AGENTS.md §5): transcribe from the
pinned Reference-LAPACK 3.12.1 sources under
`target/ref-lapack/lapack-3.12.1`, keep every output bit-identical to the
reference built by `cargo xtask build-ref-lapack`, expose LAPACK workspace
semantics (`*_work_len` / `*_opt_work_len` / `*_alloc`, caller-supplied
workspace, drivers allocate nothing), use column-major storage with `LDA = n`,
and gate each routine with `ref_*_bitexact` oracle tests plus the established
`cargo test` gates.

**The two oracles are different oracles.** cintx's AGENTS.md makes
bit-exactness non-negotiable *against libcint 6.1.3*, not against
Reference-LAPACK; cintx's oracle builds without `LAPACK_FOUND`, so the
vendored `eigh.c` `#else` branch is cintx's ground truth. This crate's
AGENTS.md makes bit-exactness non-negotiable *against Reference-LAPACK
3.12.1*. §4 shows the two cannot be satisfied by one routine.

## 2. Already covered (migrated)

| Need (cintx site) | lapack_rs surface |
|---|---|
| `type2_facs_rad_host` contraction `facs[m×nc] = buf[m×np]·ci[np×nc]`, replicating the real-BLAS `dgemm_(N,N,m,nc,np)` at PySCF `nr_ecp.c:5184` (`crates/cintx-cubecl/src/math/ecp_k_taylor.rs`) | `blocks::dgemm` (`transa/transb = false`, `alpha = 1`, `beta = 0`). Order proof: the NOTA branch zeroes each column (`+0.0` seed, as the replaced `s = 0.0`) then accumulates l-ascending axpy steps — per element `((0 + A·B) + A·B) + …` in p-ascending order, identical to the replaced loop and to the vendor `dgemm_` call. `np == 0` is guarded on the caller (leading-dimension asserts), `m/nc == 0` early-returns inside. |

## 3. cintx needs vs. status

| # | Need (cintx site) | LAPACK equivalent | Status |
|---|---|---|---|
| 1 | `cint_diagonalize` / `cint_diagonalize_host` / `cint_diagonalize_dev` (`crates/cintx-cubecl/src/math/eigh.rs`, consumed by `rys_wheeler.rs` at four host sites and inline in `ext_eigen_transform_dev`): symmetric-tridiagonal values + vectors, `n <= MXRYSROOTS (13)`, Wheeler convention (`diag` overwritten, `diag_off1[0..n-1]` input with `[n-1]` scratch, ascending `eig`, `vec[i*n+j]`, `i32` info) | `DSTEMR` (`JOBZ='V'`, `RANGE='A'`), i.e. MRRR — the `LAPACK_FOUND` branch of `eigh.c` (`dstemr_` prototype at `eigh.c:15-24`) | MISSING. **Path R** (§4.3) gives the reference routine; **Path C** (§4.4) is what cintx can actually adopt. |
| 2 | 2×2 closed form `_dlaev2` (`eigh.c:1381`, `n == 2` fast path) | `DLAEV2` | MISSING, small — **Task B**, no dependency on anything else, full kernel in §7.1 |
| 3 | `c2s_axis` / `CINTc2s_bra_sph` cart→sph application if ever routed through BLAS (`crates/cintx-cubecl/src/transform/c2s.rs`, `crates/cintx-compat/src/transform.rs`) | `DGEMV` as `pub` (already `pub(crate)` in `blas.rs`; `testing::dgemv` wrapper exists) | MISSING, small — **Task C**, which is the *same* work as `pyscf-lapack-coverage-plan.md` Task I: do it once, for both consumers. Only if a consumer asks; cintx's loops are libcint-table-specific and order-critical today |

Intentionally out of scope (not LAPACK replacements):

- ECP Type-2 angular splice device kernel (`ecp_type2_angular_kernel`,
  `crates/cintx-cubecl/src/kernels/ecp.rs:1273`): fused on-device recompute of
  the two `dgemm_` calls at `nr_ecp.c:5505-5510` with `common_fac` folded
  in-kernel. Device-resident compute under cintx's `#[cube]` integer-flag
  rules; this crate's host `dgemm` must not replace it. The `device` feature
  path (`src/device/kernels/gemm.rs`) is the counterpart for on-device work.
- Type-2 splice host test oracle (`ecp.rs:~2997-3012`): must stay an
  *independent* transcription of `nr_ecp.c:5505-5510` (device-vs-host gate);
  routing it through this crate would dissolve the oracle.
- `transform_angj` / `ECPtype_so_cart` dgemms (`nr_ecp.c:4616,5722-5731`):
  spin-orbit ECP, out of scope in cintx (Phase 19 D-12); no cintx counterpart.
- `_matmul_14_14` (`crates/cintx-cubecl/src/math/stg.rs:305`): fixed 14×14
  DCT from `stg_roots.c`, not BLAS.
- `cintx-oracle` `dgemm_shim.c` / `fblas.c` build: oracle harness glue, not
  library code.
- Wheeler/Q-matrix recurrences (`rys_wheeler.rs`), primitive contractions,
  Rys/Jacobi recurrences: libcint's own recurrence code, no BLAS call
  upstream either.

---

## 4. The adoption gate is decidable *before* the transcription

Revision 1 put the cintx adoption gate (A3: `max_ulp = 0` between a Reference
`DSTEMR` transcription and libcint's vendored solver) at the end of a
≈5 000-line dependency closure. The gate is decidable now, from the sources,
and the answer is **no**.

### 4.1 libcint's `#else` path is a *different* MRRR, not a reference one

`libcint-master/src/eigh.c` implements the same algorithm *family* —
`_dlarrk` (62), `_dlarrc` (118), `_dlasq4` (149), `_dlasq5` (388), `_dlasq2`
(431), `_compute_eigenvalues` (629), `_dlarrf` (853), `_dlaneg` (897),
`_dlarrb` (934), `_dlar1v` (992), `_compute_eigenvectors` (1095), `_dlaev2`
(1381) — with the block splitting, `DLARRA`, `DLARRD`, `DLARRJ`, `DLARRR` and
the `DSTEMR` glue removed and several constants replaced by literals.

### 4.2 Six constant-level divergences, each sufficient on its own

| Quantity | Reference-LAPACK 3.12.1 | libcint 6.1.3 `eigh.c` | Consequence |
|---|---|---|---|
| Initial bisection bracket | `LEFT = GL − FUDGE·TNORM·EPS·N − FUDGE·TWO·PIVMIN` (`dlarrk.f:198`) | `left = gl − tnorm·2·DBL_EPSILON·n` (`eigh.c:77`) — **no `−4·PIVMIN` term** | The very first `MID = HALF·(LEFT+RIGHT)` differs ⇒ every subsequent bit differs |
| Pivot floor in the Sturm count | `IF(ABS(TMP1).LT.PIVMIN) TMP1 = −PIVMIN` (`dlarrk.f:222,229`) | absent (`eigh.c:90-103`) | Different negcount near a zero pivot |
| Iteration cap | `ITMAX = INT((LOG(TNORM+PIVMIN)−LOG(PIVMIN))/LOG(TWO))+2` (`dlarrk.f:193`) | fixed `ITMAX 1000` macro (`eigh.c:55,80`) | Different number of halvings ⇒ different `W` |
| Convergence test | `TMP1 < MAX(ATOLI, PIVMIN, RTOLI·TMP2)` with `ATOLI = FUDGE·TWO·PIVMIN` (`dlarrk.f:191,208`) | `tmp1 <= 0 \|\| tmp1 < reltol·MAX(…)` (`eigh.c:82-86`) | Stops at a different iterate |
| `RTL` fed to the extremal-eigenvalue bisections | `RTL = SQRT(EPS)` = `1.4901161193847656e-08` (`dlarre.f:395`) | hard-coded `rtl = 2.1e-8` (`eigh.c:656`) | Different `ISLEFT`/`ISRGHT` ⇒ different shift |
| Working `EPS` | `EPS = DLAMCH('P')` = `2.220446049250313e-16` (`dlarre.f:392`) | `eps = 2·DBL_EPSILON` (`eigh.c:655`) | Doubles every `EPS`-scaled tolerance |

None of these is a reordering. Each changes a value, so
`max_ulp = 0` between Reference `DSTEMR` and libcint's solver cannot hold in
general, and agreeing on a particular Wheeler corpus would be luck, not a
contract.

### 4.3 Path R — reference `DSTEMR` for this crate's own LAPACK surface

Still worth doing, on its own track, for LAPACK API completeness — it is the
MRRR entry point this crate lacks next to `dsteqr`/`dstedc`/`dsterf`, and no
consumer plan currently asks for it, so it carries no external deadline.
Oracle: `cargo xtask build-ref-lapack`. **Not** a cintx deliverable, and never
on cintx's critical path again.

### 4.4 Path C — `cint_stemr`, bit-exact to libcint 6.1.3

What cintx can actually adopt. Transcribe `libcint-master/src/eigh.c`'s
`#else` branch — the twelve `static` routines listed in §4.1 — as
`src/cint_stemr.rs` + `src/device/kernels/cint_stemr.rs`, under an explicit
module header saying the oracle is libcint 6.1.3 and **not** Reference-LAPACK.
Roughly 1 400 lines rather than 5 000, one oracle, and `max_ulp = 0` is
reachable because it is the same algorithm with the same constants.

This is a new kind of module for this repo, so the header must say so
loudly; AGENTS.md §5's "transcribe from the pinned sources, never from memory"
applies unchanged, with `cintx/libcint-master/src/eigh.c` as the pinned source
and its SHA recorded in the header.

### 4.5 Task A0 — run this decision first (≈1 day, no transcription)

A differential harness with **no** new numerics:

1. Build the Wheeler input corpus: `n = 1..=13` tridiagonals as
   `rys_wheeler.rs` produces them, dumped from cintx at its four
   `cint_diagonalize` call sites plus `ext_eigen_transform_dev`.
2. Run each through (a) `LAPACK_RS_REF_DIR`'s real `dstemr_`
   (`JOBZ='V'`, `RANGE='A'`) and (b) cintx's `cint_diagonalize_host`.
3. Report share of bit-identical elements and max ULP per `n`,
   magnitude-worst first (cintx's ULP-table method).

Acceptance: A0 **reports**; it does not gate. Its output picks Path R only,
Path C only, or both, and it is the artefact the cintx side needs to justify
keeping its vendored solver. Expected result, from §4.2: a non-zero max ULP at
some `n`, so Path C is the cintx deliverable.

---

## 5. Plan critical path (plan-speed)

Revision 1: `A → A2 → A3`, one front, ≈5 000 lines before the first decidable
gate, with the highest-risk item last.

Revision 2 — three concurrent fronts, highest-risk item first:

```
 day 0 ─┬─ A0  differential harness, no new numerics ............ 1 d  ─┐
        ├─ B   DLAEV2 host + kernel (§7.1), no deps ............. 1 d   │
        └─ C   dgemv/zgemv visibility + oracle test, no deps .... 0.5 d │
                                                                        │
 A0 reports ────────────────────────────────────────────────────────────┘
        │
        ├─ Path C (cintx)                     ├─ Path R (LAPACK surface)
        │   C1 cint_stemr host transcription  │   R1 dlarrk/dlarrc/dlaneg/dlar1v
        │      (eigh.c #else, ~1 400 ln)      │   R2 dlasq2-family (dqds)
        │   C2 device kernels §7              │   R3 dlarre / dlarrb / dlarrf / dlarrj
        │   C3 inline-callee surface §6.2     │   R4 dlarrv
        │   C4 max_ulp = 0 gate vs libcint    │   R5 dstemr driver + ref_stemr_bitexact
        │   C5 cintx switches over            │
```

Front-loading facts:

* **B has no dependency on A.** `DLAEV2` is 80 lines with its own oracle test,
  it is cintx's `n == 2` fast path, and its kernel (§7.1) is the smallest
  end-to-end exercise of the whole device surface — the right first commit.
* **C has no dependency on A.** It is a visibility change plus an oracle test.
* **C1..C5 and R1..R5 share `dlarrk`, `dlaneg`, `dlar1v` in *structure* but
  not in *constants*.** Write the kernels parameterised over the constants
  (§7.2 takes `eps`, `pivmin`, `reltol`, `itmax` as arguments, never as
  literals), and both paths launch the same code with different scalars. This
  is the one place where the two oracles can share an implementation, and it
  is why §7 passes every tolerance in rather than reading `crate::aux::EPS`.
* **R2 (the `dlasq2` dqds family, ≈2 100 reference lines) is the long pole of
  Path R and nothing on Path C waits for it.** libcint's `_dlasq2` is a
  reduced variant, so Path C transcribes ≈400 lines there instead.

---

## 6. Why revision 1's device shape was the slow one (run-speed)

### 6.1 The cost model

cintx's consumer calls the eigensolver **once per shell-quartet primitive**,
on an `n ≤ 13` tridiagonal, from *inside* `two_electron_scalar_kernel`.

| Quantity | Value | Source |
|---|---|---|
| Arithmetic in one `n = 13` MRRR solve | ~10⁴ flops (dqds `O(n²)` + `n` twisted factorisations `O(n)` each + bisection ≤ 60 halvings × `O(n)`) | count |
| Launch + blocking read-back on this crate's measured path | ~246 µs per round trip, of which ~130 µs is pure synchronisation | `docs/device-speed-plan.md` §2.1 (n=2048, `DeviceTrd::symv`) |
| Ratio | **~10⁵** | — |

A `#[cube(launch, launch_unchecked)]` kernel launched from `src/device/ops.rs`
through `kl!` — revision 1's Task A2 — pays that round trip once per quartet.
It is the same failure mode `device-speed-plan.md` §2.1 calls "the largest
single item" in `DLATRD`, reintroduced deliberately. cintx already avoids it:
`cint_diagonalize_dev` (`eigh.rs:749`) is an **inline `#[cube]` callee** with
caller-owned comptime-sized scratch, called from `ext_eigen_transform_dev`
(`rys_wheeler.rs:3447`) inside the caller's own launch.

### 6.2 Three surfaces, one body

Exactly the shape cintx proved out, and the same
"one body, several entry points" discipline `symv`/`ops.rs` already use here:

1. **`#[cube] pub fn cint_stemr_dev<F: Float + CubeElement>(…)` — production.**
   No `launch`. `pub` so a foreign crate's `#[cube]` kernel can call it inside
   its own launch. Every buffer caller-owned (`&Array<F>` / `&mut Array<F>`),
   every loop bound comptime or `n`-bounded, no `Vec`, no `usize` indexing
   arithmetic beyond `(expr) as usize`. This is what cintx links against.
2. **`#[cube(launch, launch_unchecked)] pub fn cint_stemr_batch_kernel<…>` —
   standalone.** One *problem* per unit; `ABSOLUTE_POS` selects the problem;
   `d`, `e`, `w`, `z` are concatenated with an `nptr[]` offset table
   (manual 11 §5, "collapse per-item launches into one"). `B` problems cost
   one dispatch and one batched read-back, not `B` of each.
3. **`cint_stemr_kernel` (one problem, `CubeDim::new_1d(1)`) — parity/debug
   only.** Marked `#[doc(hidden)]`, used by `tests/device_*_cpu.rs` to pin
   surface 1 against the host transcription. Never a production path.

Surfaces 2 and 3 are one-line wrappers over 1, so there is one body to keep
bit-exact, not three.

### 6.3 Launch geometry and host-side rules

* Geometry from `geometry::launch_1d(client, n_problems, work_per_problem)`.
  The `CUBES_PER_SM` sizing (AGENTS.md "Device rules", `device-speed-plan.md`
  §1.1) matters *more* here than for K1: a batch of a few hundred problems is
  exactly the `O(n)`-unit regime where `CubeDim::new`'s 8-plane cube leaves
  most SMs idle. Do not re-tune `CUBES_PER_SM`.
* One upload of the whole batch, one `client.read(vec![…])` of every output
  handle (manual 11 §3). No per-problem `read_one`.
* Scratch (`lplus`, `uminus`, `s`, `p`, `z`, `work`) is one device buffer of
  `B · 4n` reals carved by problem index, allocated once in `DeviceWorkspace`
  and reused — never a per-call `client.empty` (manual 13).
* `launch_unchecked` via `kl!`, with the host asserting `nptr` monotone,
  `nptr[B] == total`, `n_b ≤ N_MAX` and every output length, in one
  `// SAFETY:` block in `src/device/ops.rs` (`tests/unsafe_launchers.rs`).
* Hot index expressions stride-1 across lanes. **Exception, deliberately
  taken:** the batched kernel's inner loops are `O(n)` long with `n ≤ 13` and
  each unit owns a whole problem, so problem-major (`base = nptr[b]`,
  consecutive `j`) is the layout that wins — the same trade-off the CUDA notes
  record for K1/K1c. Record it in `tests/device_sector_model.rs` as an
  explicit exception rather than letting the test drift.

### 6.4 Second axis of parallelism, if a batch is ever small

MRRR's eigenvalue loop is embarrassingly parallel: `dlarrk` for index `iw` is
independent of every other `iw`, and `dlar1v` for eigenvector `j` is
independent of every other `j`. When `B` is small (a single Wheeler call from
a host site), launch `B · n` units instead of `B` and let
`ABSOLUTE_POS` decode `(b, iw)`. Both forms are bit-identical — each unit
keeps its own reference-ordered chain — so this is a pure geometry knob, and
belongs behind the same kind of `set_*_mode` override the repo uses for K1/K2
variants. Measure before defaulting it on (`examples/device_kernel_bench.rs`).

---

## 7. Full on-device kernel contents

`src/device/kernels/cint_stemr.rs` (Path C) and
`src/device/kernels/stemr.rs` (Path R) share these bodies; the constants
(`eps`, `pivmin`, `reltol`, `itmax`, `rtl`) are **arguments**, never literals,
which is what lets one body serve two oracles (§5).

Written against the CubeCL 0.10 manual: `if` as a statement and never as an
expression (Conditionals §"Avoid If Expressions"); no `continue`
(Loop Control); `while` with an **ascending** counter only (§7.3);
`#[comptime]` for every array size and unroll bound
(Comptime Specialization); `Array::<F>::new(comptime!(..))` for per-unit
scratch; explicit `as usize` / `as u32` casts on every builtin
(Plane §2.3 type gotcha).

### 7.0 Shared helpers

```rust
//! Device transcriptions of the MRRR kernels. Reference-LAPACK 3.12.1
//! `dlaev2.f`, `dlarrk.f`, `dlaneg.f`, `dlar1v.f` (Path R) and the matching
//! `#else`-branch routines of libcint 6.1.3 `src/eigh.c` (Path C). Same
//! bodies, different constants: every tolerance is an argument.
//!
//! Bit contract: one unit owns one (problem, index) pair, so each unit
//! replays the host chain in the reference's operand order and the result
//! is independent of geometry, vector width, planes and checked launches.

// `let mut x = zero;` before branch assignment is how a #[cube] kernel
// declares a variable; rustc sees the initial value as unused.
#![allow(unused_assignments)]

use cubecl::prelude::*;

/// gfortran's `MAX(a, b)` lowering (AGENTS.md §5).
#[cube]
fn fmax_c<F: Float>(a: F, b: F) -> F {
    select(a > b, a, b)
}

/// gfortran's `MIN(a, b)` lowering.
#[cube]
fn fmin_c<F: Float>(a: F, b: F) -> F {
    select(a < b, a, b)
}
```

`fmax_c`/`fmin_c` are the same helpers `src/device/kernels/secular.rs` already
defines; factor them into a shared `kernels/gf.rs` rather than copying, so
there is one lowering of `MAX` in the crate.

### 7.1 `DLAEV2` — the 2×2 closed form (Task B)

Line-by-line from `SRC/dlaev2.f`. The reference's comment
"*Order of execution important*" on `RT2` is load-bearing: the parenthesisation
`(ACMX/RT1)*ACMN - (B/RT1)*B` is kept exactly. `x**2` is written `x*x`
(gfortran's `powi(2)` lowering). `SGN1`/`SGN2` are `bool` flags, not integers,
because the only use is the final `SGN1.EQ.SGN2` comparison.

```rust
/// `DLAEV2` (`dlaev2.f`): eigenvalues and the first eigenvector of
/// `[[a, b], [b, c]]`. Writes `RT1, RT2, CS1, SN1` to `out[slot..slot+4]`.
///
/// `|RT1| >= |RT2|`; `[CS1, SN1]` is the eigenvector for `RT1`.
#[cube]
pub fn dlaev2_c<F: Float>(a: F, b: F, c: F, out: &mut Array<F>, slot: usize) {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let two = F::from_int(2);
    let half = F::new(0.5);

    let sm = a + c;
    let df = a - c;
    let adf = F::abs(df);
    let tb = b + b;
    let ab = F::abs(tb);

    // IF( ABS(A).GT.ABS(C) ) THEN ACMX = A; ACMN = C ELSE ACMX = C; ACMN = A
    let mut acmx = c;
    let mut acmn = a;
    if F::abs(a) > F::abs(c) {
        acmx = a;
        acmn = c;
    }

    // RT = ADF*SQRT(1+(AB/ADF)**2) / AB*SQRT(1+(ADF/AB)**2) / AB*SQRT(2)
    let mut rt = ab * F::sqrt(two);
    if adf > ab {
        let t = ab / adf;
        rt = adf * F::sqrt(one + t * t);
    } else if adf < ab {
        let t = adf / ab;
        rt = ab * F::sqrt(one + t * t);
    }

    let mut rt1 = half * rt;
    let mut rt2 = -(half * rt);
    let mut sgn1_pos: bool = true;
    if sm < zero {
        rt1 = half * (sm - rt);
        sgn1_pos = false;
        // Order of execution important (dlaev2.f:179).
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if sm > zero {
        rt1 = half * (sm + rt);
        sgn1_pos = true;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    }

    // IF( DF.GE.ZERO ) THEN CS = DF + RT; SGN2 = 1 ELSE CS = DF - RT; SGN2 = -1
    let mut cs = df - rt;
    let mut sgn2_pos: bool = false;
    if df >= zero {
        cs = df + rt;
        sgn2_pos = true;
    }

    let acs = F::abs(cs);
    let mut cs1 = one;
    let mut sn1 = zero;
    if acs > ab {
        let ct = -tb / cs;
        sn1 = one / F::sqrt(one + ct * ct);
        cs1 = ct * sn1;
    } else if ab != zero {
        let tn = -cs / tb;
        cs1 = one / F::sqrt(one + tn * tn);
        sn1 = tn * cs1;
    }

    if sgn1_pos == sgn2_pos {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }

    out[slot] = rt1;
    out[slot + 1] = rt2;
    out[slot + 2] = cs1;
    out[slot + 3] = sn1;
}
```

Notes that are not optional:

* `-(half * rt)`, `-tb / cs`, `-cs / tb`, `-sn1` are **scalar** unary minus,
  which is exact and legal. AGENTS.md's ban is on *vector* unary minus (HIP's
  C rejects it) and on the `0 - x` workaround, which loses the sign of zero.
  If this body is ever vectorised over lanes, fold the negation into the
  division (`tb / (-cs)` is *not* equivalent — use a sign-flipped operand
  built with `F::copysign`-style selects, and re-run `device_rocm_vs_cpu`).
* `ab != zero` reproduces `IF( AB.EQ.ZERO )` with the branches swapped; the
  defaults `cs1 = 1, sn1 = 0` above are that branch's assignment, which is the
  Conditionals manual's recommended "initialise then overwrite" form.
* Ordering `-0.0`: `RT2 = -HALF*RT` when `SM == 0` and `RT == 0` yields `-0.0`.
  Keep it; `dlaev2`'s callers compare `RT2` against zero with `.EQ.`, which is
  sign-blind, but the oracle test compares bits.

### 7.2 `DLARRK` — Sturm bisection, one unit per eigenvalue

From `dlarrk.f`, the whole body. `ITMAX`, `EPS`, `PIVMIN` and `RELTOL` are
arguments (§5). `FUDGE = TWO` and `HALF = 0.5D0` are the reference's own
`PARAMETER`s, so they stay literal.

The parenthesisation of `FUDGE*TNORM*EPS*N` is left-to-right as gfortran
evaluates it, with `N` converted to double last — written explicitly rather
than left to Rust's precedence, because getting it wrong is a silent
one-ULP bracket shift.

```rust
/// `DLARRK` (`dlarrk.f`): bisection for the `iw`-th eigenvalue (1-based) of
/// the tridiagonal `(d[base..base+n], e2[base..base+n-1])` inside
/// `[gl, gu]`. Returns `W`, writes `WERR` to `werr[slot]` and `INFO`
/// (0 converged, 1 = the reference's -1) to `info[slot]`.
///
/// `itmax` is the reference's
/// `INT((LOG(TNORM+PIVMIN)-LOG(PIVMIN))/LOG(TWO))+2` — see
/// [`larrk_itmax`] and the integer gate in §8.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn dlarrk_c<F: Float>(
    d: &Array<F>,
    e2: &Array<F>,
    base: u32,
    n: u32,
    iw: u32,
    gl: F,
    gu: F,
    reltol: F,
    pivmin: F,
    eps: F,
    itmax: u32,
    werr: &mut Array<F>,
    info: &mut Array<u32>,
    slot: usize,
) -> F {
    let zero = F::from_int(0);
    let two = F::from_int(2);
    let half = F::new(0.5);
    let fudge = two;

    // Quick return: N <= 0 -> INFO = 0. The caller never passes n == 0;
    // the guard is the reference's and is kept so the branch counts match.
    let mut w = zero;
    if n > 0 {
        let tnorm = fmax_c::<F>(F::abs(gl), F::abs(gu));
        let rtoli = reltol;
        let atoli = fudge * two * pivmin;
        let mut inf: u32 = 1; // INFO = -1

        // LEFT  = GL - FUDGE*TNORM*EPS*N - FUDGE*TWO*PIVMIN
        // RIGHT = GU + FUDGE*TNORM*EPS*N + FUDGE*TWO*PIVMIN
        let pert = ((fudge * tnorm) * eps) * F::cast_from(n);
        let ftp = fudge * two * pivmin;
        let mut left = gl - pert - ftp;
        let mut right = gu + pert + ftp;

        let mut it: u32 = 0;
        let mut step: u32 = 0;
        let bound = itmax + 2; // the reference exits on IT > ITMAX
        while step < bound {
            let tmp1 = F::abs(right - left);
            let tmp2 = fmax_c::<F>(F::abs(right), F::abs(left));
            // IF( TMP1.LT.MAX( ATOLI, PIVMIN, RTOLI*TMP2 ) ) -> converged
            let tol = fmax_c::<F>(fmax_c::<F>(atoli, pivmin), rtoli * tmp2);
            let mut done: bool = false;
            if tmp1 < tol {
                inf = 0;
                done = true;
            }
            if it > itmax {
                done = true;
            }
            if done {
                step = bound; // `break` out of the bounded loop
            } else {
                it += 1;
                let mid = half * (left + right);
                let mut negcnt: u32 = 0;
                let mut t = d[(base) as usize] - mid;
                if F::abs(t) < pivmin {
                    t = -pivmin;
                }
                if t <= zero {
                    negcnt += 1;
                }
                let mut i: u32 = 1;
                while i < n {
                    t = d[(base + i) as usize] - e2[(base + i - 1) as usize] / t - mid;
                    if F::abs(t) < pivmin {
                        t = -pivmin;
                    }
                    if t <= zero {
                        negcnt += 1;
                    }
                    i += 1;
                }
                if negcnt >= iw {
                    right = mid;
                } else {
                    left = mid;
                }
                step += 1;
            }
        }

        w = half * (left + right);
        werr[slot] = half * F::abs(right - left);
        info[slot] = inf;
    } else {
        info[slot] = 0;
    }
    w
}
```

`step = bound` instead of `break`: `break` *is* supported (Loop Control
manual), and either form generates the same values. Use `break` if the
generated IR is cleaner on all three backends; the sentinel form is written
here because it survives the `#[cube]` lowering unchanged on cubecl-cpu 0.10,
which is the runtime the f64 gates run on. Pick one, measure, and note it —
do not leave both in the tree.

`ITMAX` is the one integer this kernel cannot get from the reference for free
in the fused path, because `gl`/`gu` are device values there:

```rust
/// `dlarrk.f:193`: `ITMAX = INT((LOG(TNORM+PIVMIN)-LOG(PIVMIN))/LOG(TWO))+2`.
/// `log_two` is passed in (gfortran constant-folds `LOG(TWO)`), so only the
/// two data-dependent logs run on the device.
#[cube]
pub fn larrk_itmax<F: Float>(tnorm: F, pivmin: F, log_two: F) -> u32 {
    let q = (F::log(tnorm + pivmin) - F::log(pivmin)) / log_two;
    u32::cast_from(F::floor(q)) + 2u32
}
```

**Named risk, must be gated as an integer.** `F::log` lowers to the backend's
`log`, which is not required to agree with glibc's `log` that gfortran calls.
Any disagreement that crosses an integer boundary changes the halving count
and so changes `W`. `tests/device_stemr_cpu.rs` must compare
`larrk_itmax` against the host's value **as a `u32`, for every problem in the
corpus**, so a libm difference fails loudly instead of showing up as a
mysterious one-ULP eigenvalue. Where the caller is on the host (surfaces 2
and 3 of §6.2), pass the host's `itmax` and skip the device log entirely;
only the fused inline callee needs `larrk_itmax`.

### 7.3 `DLANEG` — negcount, and the descending-counter rewrite

`dlaneg.f` with `BLKLEN = 128` (`dlaneg.f:142`). Part II walks **downward**:

```fortran
DO 230 BJ = N-1, R, -BLKLEN
   DO 23 J = BJ, MAX(BJ-BLKLEN+1, R), -1
```

That is exactly the pattern cubecl-cpu 0.10 cannot lower — a `while` loop
whose counter is decremented (AGENTS.md "CUDA notes"; regression test
`tests/cubecl_cpu_limits.rs::descending_while_counter_fails_on_cpu_runtime`).
`dlaed4_mid` hit it and was fixed by rewriting to an ascending counter over
the identical visit order; do the same here from the start rather than
rediscovering it. The rewrite is
`for t in 0..count { let j = top - t; … }`, which visits `top, top-1, …`
in the same order with the same operands, so it cannot change a value.

```rust
/// `DLANEG` (`dlaneg.f`): negcount of `L D L^T - sigma I` with twist index
/// `r` (1-based, `1 <= r <= n`). `d`, `lld` are this problem's block at
/// `base`. `blklen` is the reference's `BLKLEN = 128`, comptime so the
/// NaN-rerun blocks fold.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn dlaneg_c<F: Float>(
    d: &Array<F>,
    lld: &Array<F>,
    base: u32,
    n: u32,
    sigma: F,
    r: u32,
    #[comptime] blklen: u32,
) -> u32 {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let mut negcnt: u32 = 0;

    // I) upper part: DO 210 BJ = 1, R-1, BLKLEN   (ascending already)
    let mut t = -sigma;
    let mut bj: u32 = 1;
    while bj < r {
        let mut hi = bj + blklen - 1;
        if hi > r - 1 {
            hi = r - 1;
        }
        let mut neg1: u32 = 0;
        let bsav = t;
        let mut j = bj;
        while j <= hi {
            let dplus = d[(base + j - 1) as usize] + t;
            if dplus < zero {
                neg1 += 1;
            }
            let tmp = t / dplus;
            t = tmp * lld[(base + j - 1) as usize] - sigma;
            j += 1;
        }
        // A NaN can only follow a zero pivot after an infinite pivot; the
        // reference reruns the block substituting 1 for Inf/Inf.
        if F::is_nan(t) {
            neg1 = 0;
            t = bsav;
            let mut j2 = bj;
            while j2 <= hi {
                let dplus = d[(base + j2 - 1) as usize] + t;
                if dplus < zero {
                    neg1 += 1;
                }
                let mut tmp = t / dplus;
                if F::is_nan(tmp) {
                    tmp = one;
                }
                t = tmp * lld[(base + j2 - 1) as usize] - sigma;
                j2 += 1;
            }
        }
        negcnt += neg1;
        bj += blklen;
    }

    // II) lower part. The reference counter DECREMENTS; cubecl-cpu 0.10
    // cannot lower that (AGENTS.md, `tests/cubecl_cpu_limits.rs`). Same
    // visit order from an ASCENDING counter: BJ = (N-1) - b*BLKLEN,
    // J = BJ - s.
    let mut p = d[(base + n - 1) as usize] - sigma;
    if n > r {
        let nblk = (n - 1 - r) / blklen + 1;
        let mut b: u32 = 0;
        while b < nblk {
            let top = (n - 1) - b * blklen; // BJ
            let mut lo = r; // MAX(BJ-BLKLEN+1, R)
            if top + 1 > blklen {
                let cand = top + 1 - blklen;
                if cand > r {
                    lo = cand;
                }
            }
            let cnt = top - lo + 1;
            let mut neg2: u32 = 0;
            let bsav = p;
            let mut s: u32 = 0;
            while s < cnt {
                let j = top - s;
                let dminus = lld[(base + j - 1) as usize] + p;
                if dminus < zero {
                    neg2 += 1;
                }
                let tmp = p / dminus;
                p = tmp * d[(base + j - 1) as usize] - sigma;
                s += 1;
            }
            if F::is_nan(p) {
                neg2 = 0;
                p = bsav;
                let mut s2: u32 = 0;
                while s2 < cnt {
                    let j = top - s2;
                    let dminus = lld[(base + j - 1) as usize] + p;
                    if dminus < zero {
                        neg2 += 1;
                    }
                    let mut tmp = p / dminus;
                    if F::is_nan(tmp) {
                        tmp = one;
                    }
                    p = tmp * d[(base + j - 1) as usize] - sigma;
                    s2 += 1;
                }
            }
            negcnt += neg2;
            b += 1;
        }
    }

    // III) twist index. T was shifted by SIGMA initially.
    let gamma = (t + sigma) + p;
    if gamma < zero {
        negcnt += 1;
    }
    negcnt
}
```

`if F::is_nan(t)` replaces `SAWNAN = DISNAN(T)`. `DISNAN` is `x /= x`; cubecl's
`IsNan` lowers to the backend predicate, which agrees on every value including
signalling NaNs. The reference's NaN reruns must be kept — they are not dead
code on the Wheeler corpus, where a zero pivot after an infinite pivot is
exactly what a nearly-degenerate Rys tridiagonal produces.

### 7.4 `DLAR1V` — twisted factorisation and the FP vector

`dlar1v.f`, all of it. The reference's `WORK` is carved into `LPLUS`,
`UMINUS`, `S`, `P` at offsets `0, N, 2N, 3N`; keep that carve so the
transcription reads against the source, but give each a name. Three loops
descend (`DO 80`, `DO 210`, `DO 230`) and get the §7.3 rewrite. The
`GOTO 220/240/260/280` early exits become a `done` flag, per the Loop Control
manual's `continue`-free idiom.

```rust
/// `DLAR1V` (`dlar1v.f`): the FP vector `z` for `lambda`, the twist index
/// `r`, `MINGMA`, `NEGCNT`, `NRMINV`, `RESID`, `RQCORR`.
///
/// `work` is the reference's `WORK` carved as
/// `lplus = work[0..n]`, `uminus = work[n..2n]`,
/// `s = work[2n..3n]`, `p = work[3n..4n]`, all relative to `wbase`.
/// Scalars come back in `out[slot..slot+4]` as
/// `[MINGMA, NRMINV, RESID, RQCORR]`; `isuppz` gets `[B1, BN]`;
/// the chosen twist index and `NEGCNT` go to `iout[slot2..slot2+2]`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn dlar1v_c<F: Float>(
    d: &Array<F>,
    l: &Array<F>,
    ld: &Array<F>,
    lld: &Array<F>,
    base: u32,
    n: u32,
    b1: u32,
    bn: u32,
    lambda: F,
    r_in: u32,
    gaptol: F,
    pivmin: F,
    eps: F,
    wantnc: bool,
    z: &mut Array<F>,
    zbase: u32,
    work: &mut Array<F>,
    wbase: u32,
    isuppz: &mut Array<u32>,
    out: &mut Array<F>,
    iout: &mut Array<u32>,
    slot: usize,
    slot2: usize,
) {
    let zero = F::from_int(0);
    let one = F::from_int(1);

    let indlpl = wbase;
    let indumn = wbase + n;
    let inds = wbase + 2u32 * n;
    let indp = wbase + 3u32 * n;

    let mut r1 = r_in;
    let mut r2 = r_in;
    if r_in == 0 {
        r1 = b1;
        r2 = bn;
    }

    if b1 == 1 {
        work[(inds) as usize] = zero;
    } else {
        work[(inds + b1 - 1) as usize] = lld[(base + b1 - 2) as usize];
    }

    // Stationary transform (differential form) up to R2.
    let mut neg1: u32 = 0;
    let mut s = work[(inds + b1 - 1) as usize] - lambda;
    let mut i = b1;
    while i < r1 {
        let dplus = d[(base + i - 1) as usize] + s;
        work[(indlpl + i - 1) as usize] = ld[(base + i - 1) as usize] / dplus;
        if dplus < zero {
            neg1 += 1;
        }
        work[(inds + i - 1) as usize] =
            s * work[(indlpl + i - 1) as usize] * l[(base + i - 1) as usize];
        s = work[(inds + i - 1) as usize] - lambda;
        i += 1;
    }
    let mut sawnan1: bool = F::is_nan(s);
    if !sawnan1 {
        let mut i2 = r1;
        while i2 < r2 {
            let dplus = d[(base + i2 - 1) as usize] + s;
            work[(indlpl + i2 - 1) as usize] = ld[(base + i2 - 1) as usize] / dplus;
            work[(inds + i2 - 1) as usize] =
                s * work[(indlpl + i2 - 1) as usize] * l[(base + i2 - 1) as usize];
            s = work[(inds + i2 - 1) as usize] - lambda;
            i2 += 1;
        }
        sawnan1 = F::is_nan(s);
    }
    if sawnan1 {
        // Slower version: floor the pivot, substitute LLD for a zero LPLUS.
        neg1 = 0;
        s = work[(inds + b1 - 1) as usize] - lambda;
        let mut i3 = b1;
        while i3 < r1 {
            let mut dplus = d[(base + i3 - 1) as usize] + s;
            if F::abs(dplus) < pivmin {
                dplus = -pivmin;
            }
            work[(indlpl + i3 - 1) as usize] = ld[(base + i3 - 1) as usize] / dplus;
            if dplus < zero {
                neg1 += 1;
            }
            work[(inds + i3 - 1) as usize] =
                s * work[(indlpl + i3 - 1) as usize] * l[(base + i3 - 1) as usize];
            if work[(indlpl + i3 - 1) as usize] == zero {
                work[(inds + i3 - 1) as usize] = lld[(base + i3 - 1) as usize];
            }
            s = work[(inds + i3 - 1) as usize] - lambda;
            i3 += 1;
        }
        let mut i4 = r1;
        while i4 < r2 {
            let mut dplus = d[(base + i4 - 1) as usize] + s;
            if F::abs(dplus) < pivmin {
                dplus = -pivmin;
            }
            work[(indlpl + i4 - 1) as usize] = ld[(base + i4 - 1) as usize] / dplus;
            work[(inds + i4 - 1) as usize] =
                s * work[(indlpl + i4 - 1) as usize] * l[(base + i4 - 1) as usize];
            if work[(indlpl + i4 - 1) as usize] == zero {
                work[(inds + i4 - 1) as usize] = lld[(base + i4 - 1) as usize];
            }
            s = work[(inds + i4 - 1) as usize] - lambda;
            i4 += 1;
        }
    }

    // Progressive transform down to R1. Reference: DO 80 I = BN-1, R1, -1.
    // Ascending counter, identical visit order (§7.3).
    let mut neg2: u32 = 0;
    work[(indp + bn - 1) as usize] = d[(base + bn - 1) as usize] - lambda;
    let mut cnt = 0u32;
    if bn > r1 {
        cnt = bn - r1;
    }
    let mut t: u32 = 0;
    while t < cnt {
        let i5 = (bn - 1) - t;
        let dminus = lld[(base + i5 - 1) as usize] + work[(indp + i5) as usize];
        let tmp = d[(base + i5 - 1) as usize] / dminus;
        if dminus < zero {
            neg2 += 1;
        }
        work[(indumn + i5 - 1) as usize] = l[(base + i5 - 1) as usize] * tmp;
        work[(indp + i5 - 1) as usize] = work[(indp + i5) as usize] * tmp - lambda;
        t += 1;
    }
    let sawnan2: bool = F::is_nan(work[(indp + r1 - 1) as usize]);
    if sawnan2 {
        neg2 = 0;
        let mut t2: u32 = 0;
        while t2 < cnt {
            let i6 = (bn - 1) - t2;
            let mut dminus = lld[(base + i6 - 1) as usize] + work[(indp + i6) as usize];
            if F::abs(dminus) < pivmin {
                dminus = -pivmin;
            }
            let tmp = d[(base + i6 - 1) as usize] / dminus;
            if dminus < zero {
                neg2 += 1;
            }
            work[(indumn + i6 - 1) as usize] = l[(base + i6 - 1) as usize] * tmp;
            work[(indp + i6 - 1) as usize] = work[(indp + i6) as usize] * tmp - lambda;
            if tmp == zero {
                work[(indp + i6 - 1) as usize] = d[(base + i6 - 1) as usize] - lambda;
            }
            t2 += 1;
        }
    }

    // Largest (in magnitude) diagonal element of the inverse, R1..R2.
    let mut mingma = work[(inds + r1 - 1) as usize] + work[(indp + r1 - 1) as usize];
    if mingma < zero {
        neg1 += 1;
    }
    let mut negcnt: u32 = 0;
    if wantnc {
        negcnt = neg1 + neg2;
    }
    if F::abs(mingma) == zero {
        mingma = eps * work[(inds + r1 - 1) as usize];
    }
    let mut r = r1;
    let mut i7 = r1;
    while i7 < r2 {
        let mut tmp = work[(inds + i7 - 1) as usize] + work[(indp + i7 - 1) as usize];
        if tmp == zero {
            tmp = eps * work[(inds + i7 - 1) as usize];
        }
        if F::abs(tmp) <= F::abs(mingma) {
            mingma = tmp;
            r = i7 + 1;
        }
        i7 += 1;
    }

    // Solve N^T v = e_r.
    let mut lo = b1;
    let mut hi = bn;
    z[(zbase + r - 1) as usize] = one;
    let mut ztz = one;

    // Upwards from R. Reference: DO 210/230 I = R-1, B1, -1. Ascending.
    let mut ucnt = 0u32;
    if r > b1 {
        ucnt = r - b1;
    }
    let mut u: u32 = 0;
    let mut done: bool = false;
    while u < ucnt {
        if !done {
            let i8 = (r - 1) - u;
            let mut zi = -(work[(indlpl + i8 - 1) as usize] * z[(zbase + i8) as usize]);
            if sawnan1 || sawnan2 {
                // Slower loop: Z(I+1) == 0 -> Z(I) = -(LD(I+1)/LD(I))*Z(I+2)
                if z[(zbase + i8) as usize] == zero {
                    zi = -(ld[(base + i8) as usize] / ld[(base + i8 - 1) as usize])
                        * z[(zbase + i8 + 1) as usize];
                }
            }
            z[(zbase + i8 - 1) as usize] = zi;
            if (F::abs(zi) + F::abs(z[(zbase + i8) as usize]))
                * F::abs(ld[(base + i8 - 1) as usize])
                < gaptol
            {
                z[(zbase + i8 - 1) as usize] = zero;
                lo = i8 + 1;
                done = true;
            } else {
                ztz = ztz + zi * zi;
            }
        }
        u += 1;
    }

    // Downwards from R. Reference: DO 250/270 I = R, BN-1 (already ascending).
    let mut i9 = r;
    let mut done2: bool = false;
    while i9 < bn {
        if !done2 {
            let mut zn = -(work[(indumn + i9 - 1) as usize] * z[(zbase + i9 - 1) as usize]);
            if sawnan1 || sawnan2 {
                if z[(zbase + i9 - 1) as usize] == zero {
                    zn = -(ld[(base + i9 - 2) as usize] / ld[(base + i9 - 1) as usize])
                        * z[(zbase + i9 - 2) as usize];
                }
            }
            z[(zbase + i9) as usize] = zn;
            if (F::abs(z[(zbase + i9 - 1) as usize]) + F::abs(zn))
                * F::abs(ld[(base + i9 - 1) as usize])
                < gaptol
            {
                z[(zbase + i9) as usize] = zero;
                hi = i9;
                done2 = true;
            } else {
                ztz = ztz + zn * zn;
            }
        }
        i9 += 1;
    }

    let tmp = one / ztz;
    let nrminv = F::sqrt(tmp);
    out[slot] = mingma;
    out[slot + 1] = nrminv;
    out[slot + 2] = F::abs(mingma) * nrminv;
    out[slot + 3] = mingma * tmp;
    isuppz[slot2] = lo;
    isuppz[slot2 + 1] = hi;
    iout[slot2] = r;
    iout[slot2 + 1] = if wantnc { negcnt } else { 0 };
}
```

Two things in that body that a reviewer should check against the source rather
than against intuition:

* `NEGCNT = -1` when `WANTNC` is false. The `u32` return carries `0` plus the
  `wantnc` flag; the host mapping back to `-1` lives in the driver, not here,
  because `#[cube]` has no signed sentinel that survives all three backends
  identically. `iout[slot2 + 1]` is only read when `wantnc`.
* The early exits set `ISUPPZ` and **stop accumulating `ZTZ`** — the reference
  jumps out of the loop, it does not zero the tail. The `done` flag reproduces
  that; an `if` that merely skipped the *store* would keep accumulating and
  change `NRMINV`.

Also note `iout[slot2 + 1] = if wantnc { … }` is a **host-side-style `if`
expression**, which the Conditionals manual warns against inside `#[cube]`.
Write it as the two-statement form before committing; it is spelled as an
expression here only to keep the listing short.

### 7.5 The batched driver and its launcher

```rust
/// One MRRR problem per unit. `nptr[b]..nptr[b+1]` is problem `b`'s block in
/// the concatenated `d`/`e` buffers; `zptr[b]` its block in `z`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn stemr_batch_kernel<F: Float + CubeElement>(
    d: &Array<F>,
    e: &Array<F>,
    nptr: &Array<u32>,
    zptr: &Array<u32>,
    consts: &Array<F>,      // [eps, pivmin, rtl, rtol1, rtol2, log_two, ...]
    w: &mut Array<F>,
    z: &mut Array<F>,
    work: &mut Array<F>,
    iwork: &mut Array<u32>,
    info: &mut Array<u32>,
    nprob: u32,
    #[comptime] n_max: u32,
) {
    if ABSOLUTE_POS < nprob as usize {
        let b = ABSOLUTE_POS as u32;
        let base = nptr[(b) as usize];
        let n = nptr[(b + 1) as usize] - base;
        stemr_dev::<F>(
            d, e, base, n, zptr[(b) as usize], consts, w, z, work, iwork, info,
            b, n_max,
        );
    }
}
```

and, in `src/device/ops.rs`, the only place `ArrayArg::from_raw_parts` may
appear:

```rust
/// K8: batched `DSTEMR` (`JOBZ='V'`, `RANGE='A'`).
pub fn stemr_batch<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    b: &Batch,
) {
    // Host validation discharges every device access (manual 11 §4).
    assert_eq!(b.nptr.len(), b.nprob + 1, "nptr is nprob+1 offsets");
    assert!(b.nptr.windows(2).all(|p| p[0] <= p[1]), "nptr must be monotone");
    assert_eq!(*b.nptr.last().unwrap() as usize, b.dlen, "nptr covers d");
    assert!(b.nmax <= N_MAX, "problem larger than the comptime cap");
    assert_eq!(b.wlen, b.dlen, "one eigenvalue per row");
    assert_eq!(b.zlen, b.zptr_total, "z holds every n_b x n_b block");
    assert_eq!(b.worklen, 4 * b.dlen, "dlar1v carve: lplus|uminus|s|p");

    let (count, dim) = super::geometry::launch_1d(client, b.nprob, work_per(b.nmax));
    // SAFETY: every length below is the allocation's own length, asserted
    // above; `nptr` is monotone and ends at `dlen`, so `base + n <= dlen` for
    // every unit; `zptr[b] + n*n <= zlen` by `zptr_total`; `work`'s carve is
    // `4n` per problem and `n <= N_MAX`. The kernel bounds-checks `b < nprob`.
    unsafe {
        kl!(
            kernels::stemr::stemr_batch_kernel;
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(&b.d, b.dlen, 1),
            ArrayArg::from_raw_parts(&b.e, b.dlen, 1),
            ArrayArg::from_raw_parts(&b.nptr_h, b.nprob + 1, 1),
            ArrayArg::from_raw_parts(&b.zptr_h, b.nprob + 1, 1),
            ArrayArg::from_raw_parts(&b.consts, CONSTS, 1),
            ArrayArg::from_raw_parts(&b.w, b.wlen, 1),
            ArrayArg::from_raw_parts(&b.z, b.zlen, 1),
            ArrayArg::from_raw_parts(&b.work, b.worklen, 1),
            ArrayArg::from_raw_parts(&b.iwork, b.iworklen, 1),
            ArrayArg::from_raw_parts(&b.info, b.nprob, 1),
            b.nprob as u32,
            N_MAX,
        );
    }
}
```

`Batch` lives in `src/device/workspace.rs` next to `DeviceWorkspace`, holds
every handle for the lifetime of the caller's loop (manual 11 §2), and is
released through `runtime::release_workspace()` like the existing buffers.

### 7.6 What §7 does **not** cover, and why

`DLASQ2`/`DLASQ3`/`DLASQ4`/`DLASQ5`/`DLASQ6` (the dqds eigenvalue engine,
≈2 100 reference lines) and `DLARRE`/`DLARRB`/`DLARRF`/`DLARRJ`/`DLARRV` are
transcriptions with no new device-side technique: they are sequential over the
problem, use the same `while`-ascending idiom, the same `fmax_c`/`fmin_c`
lowerings, the same host-supplied constants and the same caller-owned scratch.
Writing them out here would add pages without adding a decision. The three
kernels in §7.2–7.4 are the ones with a *device-specific* hazard
(`ITMAX` and libm, the descending counters, the NaN reruns), and they are the
inner loops that dominate the runtime.

`DLASQ3`'s `DLASQ4` shift selection has a dense `GOTO` ladder; transcribe it
to a `mut` shift variable plus the Conditionals manual's statement form, and
gate it on its own with `tests/ref_lapack_components.rs` **before** it is
wired into anything, exactly as `dlaed4`/`dlaed6` were.

---

## 8. Bit-exactness checklist for every kernel in §7

Per AGENTS.md §5 and "Device rules", each of these is a review item, not a
guideline:

1. **Operand order and parenthesisation** are the reference's, written
   explicitly (`((fudge * tnorm) * eps) * F::cast_from(n)`), never left to
   Rust precedence to "come out the same".
2. **`MAX`/`MIN`** go through `fmax_c`/`fmin_c` (`select`), never `F::max`,
   so `NaN` and `-0.0` follow gfortran. Three-argument `MAX` nests left.
3. **`x**2`** is `x * x` (gfortran `powi(2)`); `SIGN` comes from a host sign
   flag; `ABS` is `F::abs`.
4. **No FMA anywhere.** The CUDA gate runs with
   `NVRTC_COMPILE_OPTIONS_APPEND=--fmad=false` and refuses to run without a
   green `contraction_probe`; ROCm with
   `HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off`.
5. **Selects, not masked multiplies**; reference zero-skips are kept as
   branches (`if dplus < zero`, `if tmp == zero`), because removing them
   changes which values are computed.
6. **No vector unary minus** (HIP). The scalar `-x` in §7.1/§7.4 is fine;
   anything vectorised folds the sign with an IEEE identity, never `0 - x`.
7. **Ascending `while` counters only.** Every descending reference loop is
   rewritten as `top - t` over the identical visit order, with a comment
   naming the reference loop label.
8. **`read_range` takes the logical length**, not `Handle::size()` (padded on
   HIP).
9. **Bits must not depend on** vector width, plane count, cube dimension,
   kernel variant or checked vs unchecked launch
   (`tests/device_launch_parity.rs`, `tests/device_rocm_vs_cpu.rs`).
10. **`ITMAX` is compared as an integer**, per §7.2, on every corpus problem.

---

## 9. Gates

Standing host and device gates from AGENTS.md §5 apply unchanged. Added:

```
# Path R
export LAPACK_RS_REF_DIR=$PWD/target/ref-lapack/install LAPACK_RS_REQUIRE_REF=1
cargo test --release --test ref_lapack_components -- laev2 larrk laneg lar1v
cargo test --release --test ref_stemr_bitexact          # values, vectors, ISUPPZ, TRYRAC, n = 0/1/2

# Path C (oracle = libcint 6.1.3, NOT Reference-LAPACK)
cargo test --release --test cint_stemr_bitexact         # vs cintx's cint_diagonalize_host

# device, all paths
cargo test --release --test device_stemr_cpu            # CPU runtime f64 vs host, + the ITMAX integer gate
HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off \
  cargo test --release --features rocm --test device_rocm_vs_cpu
NVRTC_COMPILE_OPTIONS_APPEND=--fmad=false CUDA_PATH=/usr/local/cuda \
  cargo test --release --no-default-features --features cuda --test device_cuda

# speed, never a claim without it
cargo run --release --example device_kernel_bench -- stemr   # interleaved, min of 3
```

A speed claim follows `device-speed-plan.md`'s methodology: attribution before
optimisation, one variable per measurement, interleaved candidates, re-profile
after every change, `CUBECL_DEBUG_OPTION=profile` plus `ncu` before any kernel
is changed.

`tests/unsafe_launchers.rs` must see the new `from_raw_parts` block;
`tests/device_sector_model.rs` must record §6.3's stride exception.

---

## 10. Task list

| Task | Deliverable | Depends on | Size | Track |
|---|---|---|---|---|
| **A0** | Differential harness: Reference `dstemr_` vs cintx `cint_diagonalize_host` over the Wheeler corpus; ULP table, magnitude-worst first | — | ~1 d | decision |
| **B** | `dlaev2` host (`aux.rs`) + `dlaev2_c` kernel (§7.1) + `ref_lapack_components` gate | — | ~1 d | shared |
| **C** | `dgemv`/`zgemv` promoted to `pub` blocks surface + oracle test (**on request only**; same deliverable as `pyscf-lapack-coverage-plan.md` Task I) | — | ~0.5 d | shared |
| **R1** | `dlarrk`, `dlarrc`, `dlaneg`, `dlar1v` host + kernels (§7.2–7.4) | B | ~4 d | Path R |
| **R2** | `dlasq2`/`3`/`4`/`5`/`6` (dqds) | — | ~6 d | Path R |
| **R3** | `dlarre`, `dlarrb`, `dlarrf`, `dlarrj`, `dlarrr`, `dlarra` | R1, R2 | ~6 d | Path R |
| **R4** | `dlarrv` | R3 | ~4 d | Path R |
| **R5** | `dstemr` driver, `dstemr_min_work_len`/`dstemr_work_len`, `ref_stemr_bitexact` | R4 | ~2 d | Path R |
| **C1** | `cint_stemr` host: the twelve `#else` routines of `eigh.c`, pinned by SHA in the module header | A0, B | ~5 d | Path C |
| **C2** | Device kernels (§7 bodies with libcint's constants) + `cint_stemr_dev` inline callee + batch kernel + launcher | C1 | ~3 d | Path C |
| **C3** | `max_ulp = 0` gate vs `cint_diagonalize_host` over the Wheeler corpus | C2 | ~1 d | Path C |
| **C4** | cintx switches `eigh.rs` to `lapack_rs::cint_stemr_dev`; cintx's own gates stay green | C3 | cintx-side | Path C |

Public surface, once R5 lands:

```rust
pub fn dstemr(job: Job, n: usize, d: &mut [f64], e: &mut [f64],
              w: &mut [f64], z: &mut [f64], isuppz: &mut [i32],
              tryrac: &mut bool, work: &mut [f64], iwork: &mut [i32])
    -> Result<usize, LapackError>;          // Ok(m)
pub fn dstemr_min_work_len(n: usize, job: Job) -> (usize, usize);
pub fn dstemr_work_len(n: usize, job: Job) -> (usize, usize);
```

column-major `Z` with `LDZ = n`, caller-supplied workspace, driver allocates
nothing — the `dsteqr`/`dstedc` conventions unchanged.

## 11. Standing rules

AGENTS.md §5 applies unchanged: transcribe from the pinned sources, never from
memory; speed only from value-preserving reorderings; no FMA / `mul_add` /
fast-math; zero-skip, `LWORK`-dependent blocking, `MAX`/`MIN`/`SIGN` lowering
and complex arithmetic follow gfortran exactly; bisect every mismatch to one
routine with the oracle tests before changing anything. Device work follows
the Device rules (`kl!` launchers, `// SAFETY:` on `from_raw_parts`, no vector
unary minus for HIP, stride-1 hot indices except where §6.3 records the
exception) and the Device gates (CPU-runtime f64 parity, ROCm f32 parity,
CUDA `--fmad=false`).

**One addition, for Path C only:** `src/cint_stemr.rs` and its kernels are
bit-exact to **libcint 6.1.3**, not to Reference-LAPACK. The module header
must say so in its first line, name `cintx/libcint-master/src/eigh.c` with its
SHA as the pinned source, and state that `ref_*_bitexact` does not and must
not cover it. Mixing the two contracts in one routine is what §4 rules out.


---

## 12. Execution record (2026-09-17)

Every task in §10 is delivered except where noted. Each line names the gate
that holds it.

| Task | State | Deliverable | Gate |
|---|---|---|---|
| **A0** | **done** | `tests/a0_wheeler_differential.rs` + `crates/cint_ref` (libcint's own `eigh.c`, with a recording interposer on `_CINTdiagonalize` so the corpus is the vendor's, not a reconstruction) | reports; `docs/a0-wheeler-differential.md` |
| **B** | **done** | `blocks::dlaev2` / `blocks::dlae2` (the host transcription already existed in `aux.rs`); `dlaev2_c` kernel (§7.1) | `ref_mrrr_components::laev2_and_lae2`, `device_stemr_cpu::laev2_kernel_matches_host` |
| **C** | **done** | `blocks::dgemv` / `blocks::zgemv` + the public `Trans` enum | `ref_blas_bitexact`, `ref_stemr_bitexact::stemr_work_len_and_errors` |
| **R1** | **done** | `src/larr.rs`: `dlarrc`, `dlarrk`, `larrk_itmax`, `dlaneg`, `dlar1v` + the §7.2–7.4 kernels | `ref_mrrr_components`, `device_stemr_cpu` (incl. the ITMAX **integer** gate) |
| **R2** | **done** | `src/lasq.rs`: `dlasq2`/`3`/`4`/`5`/`6` | `ref_mrrr_components::lasq2`, `::lasq6_lasq5_lasq4` |
| **R3** | **done** | `src/larr2.rs` (`dlarra`, `dlarrr`, `dlarrb`, `dlarrj`, `dlarrf`, `dlarnv`/`dlaruv`) and `src/larre.rs` (`dlarre`) | `ref_mrrr_components::larra_and_larrr`, `::larrb_and_larrj`, `::larrf`, `::larnv_and_laruv`, `::larre` |
| **R4** | **done** | `src/larrv.rs`: `dlarrv` | via `ref_stemr_bitexact` |
| **R5** | **done** | `src/stemr.rs`: `dstemr`, `dstemr_min_work_len`, `dstemr_work_len` | `ref_stemr_bitexact` — 18 sizes × 13 fixtures × 2 `JOBZ` × 2 `TRYRAC`, `W`, `Z`, `ISUPPZ`, `M`, `TRYRAC` and `INFO` all bit-identical |
| **C1** | **done** | `src/cint_stemr.rs`: the twelve `#else` routines of `eigh.c`, SHA-pinned in the header | `cint_stemr_bitexact` |
| **C2** | **done, with one backend caveat** | `src/device/kernels/cint_stemr.rs`: all eleven routines, plus the three surfaces of §6.2 and the `ops.rs` launchers | `device_cint_stemr_cpu` (leaves, f64), `device_rocm::cint_stemr_launch_shapes_agree_f32`, `device_cuda::cint_stemr_device_matches_host` (f64, bit-exact) |
| **C3** | **done** | `max_ulp = 0` vs libcint on the Wheeler corpus | `cint_stemr_bitexact` — **4 610 unique Wheeler tridiagonals, zero ULP, no NaN**, plus 1 560 synthetic fixtures |
| **C4** | cintx-side | — | — |

### What A0 decided

`docs/a0-wheeler-differential.md`. Reference-LAPACK `DSTEMR` and libcint's
solver agree on **23.1%** of the Wheeler corpus's eigenvalues, and disagree by
up to **4.25 × 10⁹ ULP**. §4.2's prediction is confirmed by measurement:
**Path C is the cintx deliverable; Path R is this crate's own LAPACK surface
and is never on cintx's critical path.** A second finding, outside A0's remit:
cintx's `cint_diagonalize_host` is a QL solver with Rayleigh/bisection
refinement, not a transcription of `eigh.c`'s `#else` branch, so A0 compares
against the **C**, which is cintx's stated ground truth.

### Corrections to §7 found during implementation

* **§7.4's `WORK` carve is off by one for `WORK( INDS+I )`.** The listing uses
  `work[inds + i - 1]` for both `WORK( INDS+B1-1 )` and `WORK( INDS+I )`;
  the two forms differ by one, and mixing them aliases `S(1)` onto
  `WORK( INDS )`. The kernel keeps the reference's 1-based carve
  (`INDS = 2*N+1`) so both sides index by the same expression.
* **`F::is_nan` is not a `bool`** in cubecl 0.10; it returns
  `WithScalar<bool>`. `DISNAN( X )` is `X /= X` anyway, so the kernels write
  `x != x`, which is what `dlaisnan.f` computes.
* **`F::new` takes an `f32`.** `F::new(0.333)` is the `f32` nearest `0.333`
  widened back — ~1e-8 from libcint's `double`, which shows up immediately in
  `_dlasq4`'s shift. Every non-dyadic constant is now a launch argument
  (`ops::cint_stemr_consts`), which is what §5 asked for in the first place;
  dyadic ones are built from `F::from_int`.
* **`larrk_itmax` needs `F::ln`, not `F::log`**, and `HALF` must be
  `F::from_int(1) / two` rather than `F::new(0.5)`.

### The two things that are not done

1. **The composite Path C kernels do not lower on cubecl-cpu 0.10.**
   `cint_dlarrb_c`, `cint_dlasq5_c`, `cint_dlasq2_c` and the two composite
   routines hit a control-flow lowering bug: the launch prints
   `operation with block successors must terminate its parent block` on a
   worker thread and **silently no-ops**. It is not size alone (the much
   larger `cint_dlasq4_c` lowers), not any single statement, not shadowing and
   not the decremented-counter limitation, and the vendored `cubecl-opt` does
   not fix it; hipRTC lowers all of them. Full write-up:
   `docs/cubecl-cpu-lowering-limit.md`; regression test (`#[ignore]`d):
   `tests/cubecl_cpu_limits.rs::cint_dlasq2_exceeds_cpu_runtime_lowering_limit`.
   Seven of the eleven K9 leaves are still gated on the CPU runtime in f64,
   bit-exact; the composite's f64 gate is CUDA.
   **A production caller must not assume the composite runs on `CpuRuntime`.**
   `device::probe::k9_supported` is that guard, modelled on
   `k4_supported`: one `n = 5` launch compared bit for bit against
   `lapack_rs::cint_stemr`, cached per runtime type. It reports `false` on the
   CubeCL CPU runtime and `true` on hipRTC, and both directions are asserted
   (`device_cint_stemr_cpu::k9_supported_reports_false_on_cpu_runtime`;
   `device_cuda::cint_stemr_device_matches_host` asserts `true`. On the local
   ROCm GPU the probe is `false` because that GPU has no f64 at all, which
   `device_rocm::cint_stemr_launch_shapes_agree_f32` asserts explicitly rather
   than letting a vacuous pass hide the distinction — the f32 run there is
   what shows hipRTC lowers the kernel.) A `false` verdict means
   "run the host transcription" — bit-exact to libcint either way, just
   slower.
2. **The CUDA gate has not been run.** No f64 NVIDIA GPU was reachable during
   this session — `colab new --gpu T4` returned `Service Unavailable` on 90+
   consecutive attempts over ~3 hours (a CPU runtime was granted immediately
   on the same account, so it is T4 capacity, not authentication).
   `tests/device_cuda.rs::cint_stemr_device_matches_host` is written and
   compiles under `--no-default-features --features cuda`; it is the last
   outstanding gate. Run it with:

   ```
   NVRTC_COMPILE_OPTIONS_APPEND=--fmad=false CUDA_PATH=/usr/local/cuda \
     cargo test --release --no-default-features --features cuda --test device_cuda
   ```

### One real bug the GPU gate caught

`cint_stemr_dev` originally ignored its `iwbase` argument, so every unit of the
batched kernel wrote the same `iwork` block: the twist indices and the cluster
range tables raced. It was invisible on a single problem and invisible in the
eigenvalues (they came out identical); only `INFO` differed between the
one-problem and batched launch shapes.
`tests/device_rocm.rs::cint_stemr_launch_shapes_agree_f32` is what found it,
and is what keeps it fixed — which is the point of AGENTS.md's rule that bits
must not depend on the launch shape.

### Surface 1 is proven from outside the crate

`tests/device_rocm.rs::cint_stemr_inline_callee_from_a_foreign_kernel_f32`
defines a `#[cube(launch)]` kernel **in the test crate** — a crate that merely
depends on `lapack_rs`, exactly cintx's position — and calls
`cint_stemr_dev` inside its own launch, with no `launch` attribute on the
callee and no round trip. It compiles, runs on hipRTC and produces bits
identical to `lapack_rs`'s own one-problem launcher. That the signature
compiles from outside is itself the deliverable: every type in it and every
constant slot (`SC_LEN`, `ICNT_LEN`, `CC_*`, `cint_stemr_consts`) has to be
reachable from a foreign crate, and nothing about the callee may require a
launch of its own. §6.2's production surface is real, not just written down.

### Also delivered, and one thing deliberately deferred

* `examples/device_kernel_bench.rs` gained a `stemr <B>` mode: `B` Wheeler-sized
  problems in one dispatch against the same problems one dispatch each, which
  is §6.1's cost model measured rather than argued. It refuses to report
  numbers when the launch wrote nothing, so it cannot quietly time a failed
  compile on the CPU runtime. **No speed claim is made here** — per
  `docs/device-speed-plan.md`'s methodology a claim needs a measurement, and
  the only backend that can run the composite locally is ROCm in f32.
* `tests/device_launch_parity.rs::mrrr_leaves_checked_unchecked_parity` closes
  §8 item 9 for K8 on the CPU runtime; `tests/device_rocm.rs` closes the
  launch-shape half for K9.
* **§6.4's second axis of parallelism (`B · n` units instead of `B`) is not
  implemented.** §6.4 itself says "measure before defaulting it on", and the
  measurement needs an f64 GPU, which was not reachable. The batched kernel is
  one problem per unit today; adding the `(b, iw)` decode is a geometry knob
  behind a `set_*_mode` override, exactly as K1/K2 have, and should be done
  *with* the `device_kernel_bench -- stemr` numbers in hand, not before.

### New gates

```
# Path R (Reference-LAPACK oracle)
export LAPACK_RS_REF_DIR=$PWD/target/ref-lapack/install LAPACK_RS_REQUIRE_REF=1
cargo test --release --test ref_mrrr_components --test ref_stemr_bitexact

# Path C (libcint 6.1.3 oracle — NOT Reference-LAPACK)
export CINTX_LIBCINT_DIR=/home/user/Documents/workspace/cintx/libcint-master CINT_REF_REQUIRE=1
cargo test --release --test cint_stemr_bitexact --test a0_wheeler_differential -- --nocapture

# device
cargo test --release --test device_stemr_cpu --test device_cint_stemr_cpu
HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off \
  cargo test --release --no-default-features --features rocm --test device_rocm
NVRTC_COMPILE_OPTIONS_APPEND=--fmad=false CUDA_PATH=/usr/local/cuda \
  cargo test --release --no-default-features --features cuda --test device_cuda
```
