# A0 — Reference-LAPACK `DSTEMR` vs libcint 6.1.3's vendored solver

*Task A0 of `docs/cintx-lapack-blas-adoption-plan.md` §4.5. A0 **reports**; it
does not gate. Its output picks Path R only, Path C only, or both.*

## Verdict

**`max_ulp = 0` between Reference-LAPACK 3.12.1 `DSTEMR` and libcint 6.1.3's
`_CINTdiagonalize` is not reachable on the Wheeler corpus. Path C
(`cint_stemr`, oracle = libcint) is the cintx deliverable; Path R (reference
`DSTEMR`) stays on its own track for this crate's LAPACK surface.**

This confirms the prediction of plan §4.2 from measurement rather than from
reading the sources: 23.1% of eigenvalues agree bit for bit, and the worst
disagreement is ~4.25 × 10⁹ ULP — nine orders of magnitude past a rounding
difference, i.e. a different answer, not a differently-rounded one.

## How it is measured

`tests/a0_wheeler_differential.rs`, no new numerics:

1. **The corpus is the vendor's own.** `crates/cint_ref` compiles libcint
   6.1.3's `src/eigh.c` with `-D_CINTdiagonalize=cintref_diagonalize_impl` and
   `src/rys_wheeler.c` + `src/fmt.c` without it, so a shim
   (`crates/cint_ref/src/shim.c`) supplies the `_CINTdiagonalize` symbol
   `rys_wheeler.c` calls, records `(n, diag, diag_off1)` and forwards to the
   real body. Driving `CINTrys_jacobi`, `CINTrys_laguerre` and their
   long-double variants over `n = 1..=13`, 34 `x` values spanning the
   small-x / mid / large-x Rys regimes and `lower ∈ {0, 0.3, 0.8}` therefore
   captures the exact tridiagonals libcint hands its eigensolver — including
   the ones where the Wheeler recurrence goes singular and `n` is truncated
   mid-problem, which is why the recorded `n` spread is wider than the
   requested one.
2. Each unique tridiagonal goes through (a) Reference-LAPACK's real
   `dstemr_` (`JOBZ = 'V'`, `RANGE = 'A'`, via `crates/lapack_ref`) and
   (b) libcint's `_CINTdiagonalize`.
3. The report is the share of bit-identical elements and the max ULP per `n`,
   magnitude-worst first (cintx's ULP-table method). Eigenvector components
   are compared as `|Z(1, i)|`, because the sign is a free choice of the
   solver and `weights[i] = c0[i*n]² · mu0` is what Wheeler consumes.

Both oracles are build-time gated: `LAPACK_RS_REF_DIR` for Reference-LAPACK
and `CINTX_LIBCINT_DIR` for libcint. Without either the test prints `SKIP`.

```
export LAPACK_RS_REF_DIR=$PWD/target/ref-lapack/install
export CINTX_LIBCINT_DIR=/home/user/Documents/workspace/cintx/libcint-master
cargo test --release --test a0_wheeler_differential -- --nocapture
```

## Result (2026-09-17)

Corpus: 5 304 recorded Wheeler problems, **4 610 unique** tridiagonals.

| n | problems | W bit-identical | share | **W max ULP** | \|w\| at worst | \|Z(1,i)\| max ULP |
|---|---|---|---|---|---|---|
| 6 | 416 | 828 | 33.17% | **4 253 539 634** | 4.633e-1 | 2 355 |
| 3 | 409 | 471 | 38.39% | **2 039 868 242** | 4.633e-1 | 37 |
| 5 | 405 | 415 | 20.49% | 107 693 | 5.684e-2 | 568 154 |
| 9 | 363 | 617 | 18.89% | 13 312 | 7.376e-4 | 473 |
| 13 | 195 | 407 | 16.06% | 5 632 | 1.208e-3 | 612 |
| 12 | 227 | 489 | 17.95% | 4 096 | 4.634e-4 | 396 |
| 8 | 445 | 743 | 20.87% | 3 584 | 3.159e-3 | 580 |
| 11 | 270 | 540 | 18.18% | 2 048 | 1.464e-3 | 933 |
| 7 | 430 | 561 | 18.64% | 1 920 | 5.350e-3 | 1 097 |
| 10 | 307 | 503 | 16.38% | 1 536 | 9.887e-4 | 662 |
| 4 | 401 | 364 | 22.69% | 256 | 3.012e-2 | 40 |
| 2 | 398 | 566 | 71.11% | 4 | 2.742e-2 | 2 |
| 1 | 344 | 344 | 100.00% | 0 | 0 | 0 |

Overall: **6 848 / 29 628 eigenvalues bit-identical (23.11%)**.

## Reading the table

* **`n = 1` is exact and `n = 2` is nearly so.** Both sides short-circuit:
  `n = 1` is `W(1) = D(1)`, and `n = 2` is `DLAEV2` on one side and libcint's
  `_dlaev2` (`eigh.c:1381`) on the other — the same closed form. The residual
  4-ULP spread at `n = 2` comes from `DSTEMR`'s scaling
  (`DLANST`/`RMIN`/`RMAX`, `dstemr.f:598-615`), which libcint does not do.
* **`n >= 3` is where the six constant-level divergences of plan §4.2 bite**,
  and they bite hardest where the spectrum is clustered: the two
  multi-billion-ULP rows are eigenvalues near `4.6e-1` whose neighbours are
  within a few ULP of them, so a different initial bisection bracket puts the
  two solvers on opposite sides of a cluster boundary and they converge to
  *different members of the cluster*. That is not a tolerance question.
* **Eigenvectors are worse than eigenvalues**, as expected: `n = 5` has a
  568 154-ULP worst case on `|Z(1, i)|` against 107 693 on `W`, because an
  eigenvector inherits the eigenvalue error through the twisted factorisation
  and then amplifies it by `1/gap`.
* The Wheeler corpus is unusually adversarial by construction: `b[i] < 0`
  truncations (libcint prints `libcint rys_wheeler singular value …` to stderr
  during the sweep) leave nearly-degenerate tridiagonals, which is exactly the
  regime where two MRRR variants with different brackets diverge.

## What this decides

| Track | Oracle | Status after A0 |
|---|---|---|
| **Path R** — `dstemr` in this crate | Reference-LAPACK 3.12.1 | Worth doing for LAPACK API completeness; **never on cintx's critical path**. (Delivered: `src/stemr.rs`, gated by `tests/ref_stemr_bitexact.rs`.) |
| **Path C** — `cint_stemr` | libcint 6.1.3 `eigh.c` `#else` | **The cintx deliverable.** `max_ulp = 0` is reachable there because it is the same algorithm with the same constants. |

The artefact the cintx side needs to justify keeping its vendored solver is
this table: adopting Reference-LAPACK's `DSTEMR` would change Rys roots and
weights on 77% of the Wheeler corpus, by up to 4 × 10⁹ ULP.

### A second finding, outside A0's remit

cintx's `cint_diagonalize_host` (`crates/cintx-cubecl/src/math/eigh.rs:381`) is
**not** a transcription of `eigh.c`'s `#else` branch: it is a QL solver
(`tqli_impl`, Numerical-Recipes `tqli` with a Wilkinson shift) followed by
Rayleigh and bisection refinement, where libcint's `#else` branch is the
MRRR/dqds family (`_compute_eigenvalues` + `_compute_eigenvectors`). A0 there-
fore compares against the **C**, which is cintx's stated ground truth
(`LAPACK_FOUND` is undefined in cintx's oracle build), rather than against the
Rust port. Whether the Rust port itself is bit-exact to the C is a cintx-side
question this plan does not own, but it is worth raising there: Path C's
`max_ulp = 0` gate (C3) is against libcint, so if the port and the C disagree,
C4 changes cintx's bits.
