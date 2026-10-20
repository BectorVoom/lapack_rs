//! THREE DISTINCT cubecl-cpu 0.10 / cubecl-opt 0.10 compiler limitations. The
//! first two were found while bringing K4 (`DLAED3` on the device,
//! `src/device/kernels/secular.rs`) up on the CubeCL CPU runtime; the third
//! while bringing K9 (Path C, `src/device/kernels/cint_stemr.rs`) up:
//!
//! 1. `descending_while_counter_fails_on_cpu_runtime`, `#[ignore]`d: a
//!    genuine, permanent cubecl-cpu 0.10 backend limitation (a `while` loop
//!    with a decremented counter does not lower) that this crate avoids by
//!    never writing that pattern — it does not need fixing upstream because
//!    the ascending-`for` rewrite in `secular.rs` is itself bit-exact.
//!    Kept as the exact reproduction; re-run manually after any
//!    cubecl/cubecl-opt version bump to check whether it is fixed upstream:
//!    `cargo test --release --features cpu --test cubecl_cpu_limits -- --ignored --nocapture --test-threads=1`
//! 2. `dlaed4_mid_gvn_convert_compiles_and_runs_on_cpu_runtime`, NOT
//!    `#[ignore]`d: was a real cubecl-opt 0.10 GVN/PRE bug
//!    (`gvn/convert.rs:22` "no entry found for key"); now fixed in this
//!    workspace by the vendored `crates/cubecl-opt` patch (see
//!    `crates/cubecl-opt/PATCHED.md`) and kept here as a permanent
//!    regression test against that patch, now asserting success and a
//!    bit-exact match against the host `DLAED4` rather than merely
//!    "non-zero". K4 is fully supported on the CPU runtime today
//!    (`tests/device_secular_cpu.rs`); nothing in this file claims
//!    otherwise.
//! 3. `cint_dlasq2_exceeds_cpu_runtime_lowering_limit`, `#[ignore]`d: a
//!    control-flow lowering bug in cubecl-cpu 0.10 whose exact trigger is
//!    unidentified. Past it the launch prints `error: operation with block
//!    successors must terminate its parent block` / `failed to run pass` on a
//!    worker thread and **silently no-ops**, so the caller reads zeros.
//!    `docs/cubecl-cpu-lowering-limit.md` has the full write-up: it is not a
//!    language construct (stubbing any one of five unrelated loops fixes it),
//!    the vendored `cubecl-opt` does not fix it, and hipRTC lowers the very
//!    same kernel. Re-run after any cubecl bump to see whether it is fixed
//!    upstream.
#![cfg(feature = "cpu")]

use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::kernels::secular::dlaed4_mid;
use lapack_rs::device::workspace::{read_range, upload};

fn client() -> ComputeClient<CpuRuntime> {
    CpuRuntime::client(&CpuDevice)
}

/// LIMITATION 1 (fixed in this crate by avoiding the pattern; not fixed in
/// cubecl-cpu itself): a `while` loop whose counter is DECREMENTED fails to
/// lower, with
/// `cubecl-cpu-0.10.0/src/compiler/module.rs:94:13: failed to run pass` /
/// `operation with block successors must terminate its parent block`,
/// reproduced here in isolation. Ascending `while` loops, flag-terminated
/// `while` loops, nested `while` loops and `#[cube]` helper calls are all
/// fine — only the decremented counter breaks. `src/device/kernels/
/// secular.rs`'s `dlaed4_mid` used to have four such loops (`jr1`..`jr4`,
/// the reference `DLAED4`'s `DO 60`-style phi sums); they are now written as
/// ascending `for t in bound..n + 1 { let j = n + bound - t; .. }`, which
/// walks the identical `n, n-1, .., bound` visit order (proven bit-exact
/// against the original descending form for many `(n, bound)` pairs,
/// including the empty-range and full-range edges, in a standalone
/// host-only check outside this crate) and compiles fine (probe D of the
/// original investigation, and every `secular.rs` site today).
#[test]
#[ignore = "cubecl-cpu 0.10 backend limitation: a `while` loop with a decremented counter does not lower. Re-check on a cubecl/cubecl-opt version bump."]
fn descending_while_counter_fails_on_cpu_runtime() {
    let c = client();
    let n = 4usize;
    let xs = vec![7.0f64; n];
    let x = upload(&c, &xs);
    let out = c.empty(n * 8);

    #[cube(launch, launch_unchecked)]
    fn descending_while<F: Float + CubeElement>(x: &Array<F>, out: &mut Array<F>, n: u32) {
        let u = ABSOLUTE_POS as u32;
        if u < n {
            let mut acc = F::from_int(0);
            let mut j = n;
            while j >= 2u32 {
                acc = acc + x[(j - 1) as usize];
                j -= 1u32;
            }
            out[u as usize] = acc;
        }
    }

    let count = CubeCount::Static(1, 1, 1);
    let dim = CubeDim::new_1d(n as u32);
    unsafe {
        let xa: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(x.clone(), n);
        let oa: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(out.clone(), n);
        descending_while::launch_unchecked::<f64, CpuRuntime>(&c, count, dim, xa, oa, n as u32);
    }
    let mut got = vec![0.0f64; n];
    read_range(&c, &out, n, 0, n, &mut got);
    assert_eq!(got[0], 21.0, "descending-while counter kernel should sum x[0..3] = 7*3 = 21");
}

/// LIMITATION 2, NOW FIXED (was NOT the same bug as limitation 1, and was
/// NOT reducible to a single minimal DSL construct): with every descending
/// `while` gone, `dlaed4_mid` alone (imported straight from `secular.rs`,
/// `pub`) used to still panic on the CPU runtime with a DIFFERENT internal
/// compiler error: `cubecl-opt-0.10.0/src/gvn/convert.rs:22:39: no entry
/// found for key`.
///
/// A second, independent bisection (distinct from the descending-while one)
/// ruled out every single-construct hypothesis tried:
/// * `dlaed6_c` alone: compiled and ran correctly (it has its own `while`
///   loop and is called from a plain `if` guard with no other loop nearby).
/// * `dlaed4_last` alone: compiled and ran correctly (one `while` loop,
///   plus calls to the two-argument `fmax_c`/`fmin_c` helpers, i.e. the
///   SAME "one while loop plus helper calls inside `if !done { .. }`" shape
///   `dlaed4_mid` has — and yet `dlaed4_mid` failed).
/// * Two independent, non-nested, ascending `while` loops (flag-terminated
///   or plain counters) back to back in one otherwise-trivial kernel:
///   compiled and ran correctly — "more than one `while` loop in a
///   function" was NOT the trigger by itself.
/// * A `while` loop that conditionally calls a helper `#[cube]` fn which
///   itself contains its own `while` loop: compiled and ran correctly — a
///   call INSIDE a loop to a loop-bearing callee was NOT the trigger by
///   itself either.
/// * Reproducing `dlaed4_mid`'s exact shape (its own refinement `while`
///   loop, PLUS a conditional call at the `swtch3` dispatch point) with the
///   call replaced by a scalar-only, loop-free, single-line helper: STILL
///   failed identically. So it was not about `dlaed6_c`'s size or its own
///   `while` loop either.
/// * Replacing that same call with a hand-inlined (no cross-function call
///   at all) miniature `while` block at the identical position: STILL
///   failed identically. So it was not about a *function-call boundary*
///   either.
/// * Removing `dlaed4_mid`'s own refinement `while` loop while keeping a
///   real call to `dlaed6_c`: compiled and ran correctly.
/// * Removing every call at the `swtch3` dispatch point (both sites) while
///   keeping `dlaed4_mid`'s own refinement `while` loop: compiled and ran
///   correctly.
///
/// Net at the time: only reducing `dlaed4_mid`'s OWN complexity (either its
/// call sites or its own loop) avoided the panic; no minimal analog of
/// comparable individual constructs reproduced it — a compiler bug whose
/// trigger was IR size/shape-sensitive in cubecl-opt 0.10's GVN pass, not a
/// single addressable DSL pattern. No document under
/// `cubecl_error_solution_guide/` covered this error (checked: no match for
/// "gvn", "convert.rs", "value numbering" or "optimizer" anywhere under
/// `cubecl_manual/`).
///
/// ROOT CAUSE (found after the bisection above): the bug is in
/// `cubecl-opt-0.10.0/src/gvn/apply.rs::insert_block`. `partially_avail`
/// marks an expression for hoisting into a predecessor block whenever ANY
/// OTHER predecessor already has it available, then materialises it in
/// every predecessor that lacks it via `Expression::to_operation(leaders)`
/// — but the predecessor being hoisted into may have no leaders for that
/// expression's own operands (translated through `phi_translate`), and
/// `leaders[val]` panics with exactly this "no entry found for key". A
/// larger, more deeply nested function like `dlaed4_mid` has more of these
/// partially-available, operand-starved candidates than `dlaed4_last` or
/// `dlaed6_c` alone, which is why only `dlaed4_mid`'s own complexity
/// mattered in the bisection above — there was never a smaller single DSL
/// construct to find.
///
/// FIX: vendored at `crates/cubecl-opt` (`[patch.crates-io]` in
/// `Cargo.toml`, see `crates/cubecl-opt/PATCHED.md`). The patch filters
/// `partially_avail` down to expressions every predecessor can actually
/// materialise (has leaders for the value itself, or for all of the
/// translated expression's operands). Skipping a hoist only forgoes an
/// optimisation — PRE/CSE never reassociates, so no floating-point value
/// can change; the outer `while changed` loop in `insert` retries once the
/// operands do become available.
///
/// `secular_roots_kernel` (which inlines `dlaed4_mid`) now runs correctly
/// on the CPU runtime (`tests/device_secular_cpu.rs`), and
/// `DeviceDc::secular` in `src/device/stedc.rs` gates K4 on the runtime
/// `device::probe::k4_supported` capability probe rather than on
/// `has_planes`, because `[patch.crates-io]` only rewires builds from this
/// workspace (see that probe's doc comment and
/// `crates/cubecl-opt/PATCHED.md`'s "Caveat for publishing").
#[test]
fn dlaed4_mid_gvn_convert_compiles_and_runs_on_cpu_runtime() {
    let c = client();
    let n = 3usize;
    let (eps, small1) = (lapack_rs::testing::secular_eps(), lapack_rs::testing::secular_small1());
    let (d0, z0, rho) = ([0.1f64, 1.4, 2.7], [0.37f64, 0.5, 0.6], 1.0f64);
    let d = upload(&c, &d0);
    let z = upload(&c, &z0);
    let delta = c.empty(n * n * 8);
    let dlam = c.empty((n - 1) * 8);
    let info = c.empty((n - 1) * 4);

    #[cube(launch, launch_unchecked)]
    #[allow(clippy::too_many_arguments)]
    fn call_dlaed4_mid<F: Float + CubeElement>(
        d: &Array<F>,
        z: &Array<F>,
        delta: &mut Array<F>,
        dlam: &mut Array<F>,
        info: &mut Array<u32>,
        n: u32,
        rho: F,
        eps: F,
        small1: F,
    ) {
        if ABSOLUTE_POS < (n - 1) as usize {
            let slot = ABSOLUTE_POS;
            let i = ABSOLUTE_POS as u32 + 1;
            info[slot] = 0u32;
            dlam[slot] = dlaed4_mid::<F>(d, z, delta, info, slot, n, i, (i - 1) * n, rho, eps, small1);
        }
    }

    let count = CubeCount::Static(1, 1, 1);
    let dim = CubeDim::new_1d((n - 1) as u32);
    unsafe {
        let da: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(d.clone(), n);
        let za: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(z.clone(), n);
        let delta_a: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(delta.clone(), n * n);
        let dlam_a: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(dlam.clone(), n - 1);
        let info_a: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(info.clone(), n - 1);
        call_dlaed4_mid::launch_unchecked::<f64, CpuRuntime>(
            &c, count, dim, da, za, delta_a, dlam_a, info_a, n as u32, rho, eps, small1,
        );
    }
    let mut got = vec![0.0f64; n - 1];
    read_range(&c, &dlam, n - 1, 0, n - 1, &mut got);

    // `dlaed4_mid` covers every root but the last (`i < n`); compare both
    // `i = 1` and `i = 2` against the host `DLAED4`, computed at runtime.
    let mut want = vec![0.0f64; n - 1];
    for i in 1..n {
        let mut dd = d0.to_vec();
        let mut zz = z0.to_vec();
        let mut col = vec![0.0f64; n];
        let (lam, info) = unsafe { lapack_rs::testing::dlaed4(n, i, &mut dd, &mut zz, &mut col, rho) };
        assert_eq!(info, 0, "host DLAED4 non-convergence at i={i}");
        want[i - 1] = lam;
    }
    assert_eq!(got, want, "dlaed4_mid on the CPU runtime must match host DLAED4 bit for bit");
}

/// cubecl-cpu 0.10 cannot lower `cint_dlasq2_c` (Path C's dqds engine), and
/// fails by **silently producing nothing**: the MLIR verifier error and the
/// `failed to run pass` panic are on a cubecl worker thread, the launching
/// thread sees success, and the output buffer keeps its previous contents.
///
/// The assertion is therefore "the output was not written", which is what the
/// silent failure looks like from the caller. If a future cubecl fixes the
/// limit this test starts failing — that is the point.
///
/// `docs/cubecl-cpu-lowering-limit.md` records the bisect, the probes showing
/// the individual idioms are fine, and the hipRTC cross-check. The same bug
/// also blocks `cint_dlasq5_c` (40 lines) and `cint_dlarrb_c`, while the much
/// larger `cint_dlasq4_c` lowers fine — so it is not size alone.
///
/// `#[ignore]`d because it prints an MLIR error and a worker-thread panic to
/// stderr on every run.
#[test]
#[ignore = "documents a cubecl-cpu 0.10 lowering limit; prints an MLIR error and a worker-thread panic"]
fn cint_dlasq2_exceeds_cpu_runtime_lowering_limit() {
    #[cube(launch, launch_unchecked)]
    fn call_dlasq2<F: Float + CubeElement>(
        work: &mut Array<F>,
        diag: &mut Array<F>,
        offd: &mut Array<F>,
        cc: &Array<F>,
        sc: &mut Array<F>,
        info: &mut Array<u32>,
        eps: F,
        n: u32,
    ) {
        if ABSOLUTE_POS < 1usize {
            info[0] = lapack_rs::device::kernels::cint_stemr::cint_dlasq2_c::<F>(
                n, work, 0u32, diag, 0u32, offd, 0u32, eps, cc, sc, 0u32,
            ) + 1u32;
        }
    }

    let c = CpuRuntime::client(&CpuDevice);
    let n = 5usize;
    let up = |v: Vec<f64>| c.create(cubecl::bytes::Bytes::from_elems(v));
    let upi = |v: Vec<u32>| c.create(cubecl::bytes::Bytes::from_elems(v));
    let work = up(vec![1.0f64; 8 * n]);
    let diag = up(vec![1.0f64; n]);
    let offd = up(vec![0.5f64; n]);
    let cc = up(lapack_rs::device::ops::cint_stemr_consts().to_vec());
    let sc = up(vec![0.0f64; 16]);
    // A sentinel the kernel would overwrite with `INFO + 1 >= 1` if it ran.
    let info = upi(vec![7u32; 4]);
    unsafe {
        let wa: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(work.clone(), 8 * n);
        let da: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(diag.clone(), n);
        let oa: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(offd.clone(), n);
        let ca: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(cc.clone(), 12);
        let sa: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(sc.clone(), 16);
        let ia: ArrayArg<CpuRuntime> = ArrayArg::from_raw_parts(info.clone(), 4);
        call_dlasq2::launch_unchecked::<f64, CpuRuntime>(
            &c,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            wa,
            da,
            oa,
            ca,
            sa,
            ia,
            f64::EPSILON,
            n as u32,
        );
    }
    let mut got = vec![0u32; 4];
    read_range(&c, &info, 4, 0, 4, &mut got);
    assert_eq!(
        got[0], 7,
        "cubecl-cpu 0.10 now lowers cint_dlasq2_c: drop the GPU-only gating in \
         tests/device_cint_stemr_cpu.rs and update docs/cubecl-cpu-lowering-limit.md"
    );
}
