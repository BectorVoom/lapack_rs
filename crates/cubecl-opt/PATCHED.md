# Vendored cubecl-opt 0.10.0 (lapack_rs patch)

Verbatim copy of `cubecl-opt` 0.10.0 from crates.io with one change, in
`src/gvn/apply.rs::GvnState::insert_block`.

## The bug

`insert_block` implements the GVN/PRE ("partial redundancy elimination")
pass's hoisting step: for a join block with more than one predecessor, it
finds expressions that are available on *some* predecessors but not others
(`partially_avail`), and materialises the missing ones so the expression
becomes fully redundant everywhere.

Upstream, the "is this expression a hoist candidate" filter (`partially_avail`,
the first `filter` in the method) only checked that the CURRENT predecessor
being scanned has a leader for the expression's value. It did not check
whether every OTHER predecessor — the ones the expression will actually be
materialised into — has leaders for that expression's own operands. The
materialisation loop right after calls
`Expression::to_operation(leaders)`, which indexes `leaders[val]` for each
operand; when a predecessor lacks one, that index panics:

```
cubecl-opt-0.10.0/src/gvn/convert.rs:22:39: no entry found for key
```

This is a compiler ICE (internal compiler error) in cubecl-opt's optimizer,
not a bug in the kernel source it is compiling. It surfaces on a cubecl
worker thread, so the panic does not propagate to the caller: a launch that
hits it silently no-ops, and every buffer it was supposed to write reads
back as zero.

### Reproducing the original ICE

`tests/cubecl_cpu_limits.rs::dlaed4_mid_gvn_convert_compiles_and_runs_on_cpu_runtime`
calls `dlaed4_mid` (`src/device/kernels/secular.rs`) — the specific real
kernel that triggered this during K4 CPU-runtime bring-up — through a
minimal launcher. With the patch below reverted (or against a plain
`cubecl-opt = "=0.10.0"` dependency instead of the `[patch.crates-io]`
below), that same test panics with the `convert.rs:22` message above instead
of returning the correct roots. An extensive bisection (see the test's doc
comment) found no smaller single DSL construct that reproduces it — only
`dlaed4_mid`'s own size/shape triggers the panic, because a larger function
has more partially-available, operand-starved hoist candidates than a small
one.

## The fix

`src/gvn/apply.rs::GvnState::insert_block`, in the `partially_avail`
computation: after the existing filter, a second `.filter(...)` keeps only
the expressions where **every** predecessor can actually materialise the
translated expression — either it already has a leader for the value, or it
has leaders for all of the translated expression's own operands
(`expr.depends_on()`):

```rust
let partially_avail = partially_avail
    .into_iter()
    .filter(|k| {
        translated.iter().all(|(pred, exprs)| {
            let leaders = &self.block_sets[pred].leaders;
            match exprs.iter().nth(*k) {
                Some((val, expr)) => {
                    leaders.contains_key(val)
                        || expr.depends_on().iter().all(|d| leaders.contains_key(d))
                }
                None => false,
            }
        })
    })
    .collect::<HashSet<_>>();
```

## Why this cannot change a floating-point value

`insert`/`insert_block` is a hoisting (code-motion) step of PRE: its entire
job is to relocate an *already-computed* expression's evaluation earlier
(into a dominating predecessor) so a later, fully-redundant recomputation can
be replaced by a load of that hoisted value instead. It never reorders
operands, never re-associates, and never changes which operations execute
against which operand values — it only changes *where* (in which basic
block) a value that will be computed exactly once gets computed.

The patch narrows `partially_avail` to a subset of what upstream would have
attempted to hoist: it never makes the pass hoist anything upstream would
not have (a fully-available superset is not created), it only *skips* a
hoist when a predecessor cannot yet materialise the expression's operands.
Skipping a hoist purely forgoes an optimisation opportunity for one
iteration — the outer `while changed` loop in `insert` (`apply.rs::insert`)
keeps looping over the whole dominator tree until nothing changes, so once
the skipped expression's operands *do* become available (via a hoist in a
different block on a later iteration), it becomes a hoist candidate again
and gets picked up then. No candidate is dropped permanently, no value is
computed by a different formula or in a different order — only later, or via
its original (unhoisted) recomputation. Either way the floating-point value
that ends up in every use site is the one the un-optimized program computes.

## Reproducing the LIMITATION 1 bug (unrelated, not patched here)

This vendored crate does not touch the separate, permanent cubecl-cpu 0.10
backend limitation where a `while` loop with a decremented counter fails to
lower (`cubecl-cpu-0.10.0/src/compiler/module.rs:94`). That bug lives in
`cubecl-cpu`, not `cubecl-opt`, and this crate's kernels avoid the pattern
entirely (ascending `for` loops with the identical visit order) rather than
requiring a fix; see
`tests/cubecl_cpu_limits.rs::descending_while_counter_fails_on_cpu_runtime`
(kept `#[ignore]`d as the exact reproduction) and
`src/device/kernels/secular.rs`.

## Caveat for publishing

`lapack_rs` wires this in with `[patch.crates-io]` in the top-level
`Cargo.toml`, which applies to builds **from this workspace only** (same
caveat as `crates/cubecl-cuda/PATCHED.md`). A consumer who depends on a
published `lapack_rs` and enables the `cpu`, `rocm` or `cuda` feature — all
three share this one `cubecl-opt` dependency — would get upstream, unpatched
`cubecl-opt` 0.10.0. Compiling a K4 kernel (`secular_roots_kernel`, which
inlines `dlaed4_mid`) would then hit the identical ICE on a cubecl worker
thread and, because the panic does not propagate to the calling thread,
silently produce a launch that no-ops and reads back **zeros** — a silently
wrong answer, not a build failure or a runtime error.

Until the fix lands upstream, this is mitigated (not merely documented) by a
runtime capability probe rather than a compile-time or `has_planes` gate:
`lapack_rs::device::probe::k4_supported` launches the real
`secular_roots_kernel` on a tiny fixed `k = 3` fixture and compares its
`dlam`/`info` against the host `DLAED4`, computed at runtime, bit for bit.
An unpatched build's silent zeros fail that comparison, `k4_supported`
caches and returns `false`, and `src/device/stedc.rs::DeviceDc::secular`
falls back to the host transcription automatically — bit-exact either way,
just slower without the patch. `tests/device_secular_cpu.rs::
k4_supported_reports_true_on_cpu_runtime` asserts the probe reports `true`
in this workspace, so a regression to the vendored patch (or an upstream
revert) fails a test here instead of silently taking the slower host path
unnoticed.
