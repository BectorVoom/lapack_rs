//! K4 (`DLAED3` secular roots + `W` + `U`) launched directly on the CubeCL
//! CPU runtime (f64), against the host transcription (`src/laed4.rs`,
//! `src/stedc.rs::dlaed3`'s `K > 2` branch), bit for bit — including the
//! full `secular_roots_kernel` root solve (previously blocked, now fixed;
//! see the doc comment on `secular_roots_kernel_cpu_runtime_matches_host`),
//! the full `DeviceDc::secular` driver through `dstedc_device`, and the
//! `device::probe::k4_supported` capability probe that gates it.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::ops::{self, SecularConsts};
use lapack_rs::device::workspace::{read_range, upload};
use lapack_rs::testing as t;

fn client() -> ComputeClient<CpuRuntime> {
    CpuRuntime::client(&CpuDevice)
}

/// The exact constants `device::stedc::secular_consts()` (private) builds,
/// via the `testing` accessors — never reformulated.
fn consts() -> SecularConsts<f64> {
    SecularConsts {
        eps: t::secular_eps(),
        small1: t::secular_small1(),
        tsml: t::secular_tsml(),
        tbig: t::secular_tbig(),
        ssml: t::secular_ssml(),
        sbig: t::secular_sbig(),
        maxn: f64::MAX,
    }
}

/// Distinct ascending `DLAMBDA`, nonzero `Z`, `rho > 0` — a well-posed `K`
/// root secular-equation fixture, retried across seeds until every root
/// converges on the host (some `edge()` draws leave a root pathologically
/// close to its neighbour, which is a fixture-quality issue, not something
/// K4 or the host solver should be asked to paper over).
fn fixture(k: usize, seed: u64) -> (Vec<f64>, Vec<f64>, f64) {
    for attempt in 0..64u64 {
        let s = seed.wrapping_add(attempt.wrapping_mul(104_729));
        let mut d: Vec<f64> = (0..k).map(|i| i as f64 * 1.3 + 0.1).collect();
        // A plain uniform `Z` (not `edge()`'s wild-magnitude spread): DLAED2
        // hands DLAED4/DLAED3 an actual deflated update vector, never
        // something spanning 1e-7..1e5, so `edge()`'s stress magnitudes are
        // the wrong fixture here and starve Newton's method of convergence
        // within its reference 30-iteration cap — a fixture-quality issue,
        // not a K4 correctness question.
        let mut rng = Rng::new(s);
        let z: Vec<f64> = (0..k).map(|_| rng.uniform()).collect();
        // Guarantee no exact zeros (the kernel/host both special-case those
        // upstream in DLAED2, never inside DLAED4/DLAED3).
        let z: Vec<f64> = z.iter().map(|&v| if v == 0.0 { 0.37 } else { v }).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rho = 1.0 + (s % 7) as f64 * 0.25;
        let (_, _, infos) = host_roots(k, &d, &z, rho);
        if infos.iter().all(|&i| i == 0) {
            return (d, z, rho);
        }
    }
    panic!("no converging fixture found for k={k} seed={seed}");
}

/// Host roots via the exposed `t::dlaed4`, one call per root — the same
/// per-root loop `dlaed3`'s `K > 2` branch runs.
fn host_roots(k: usize, d: &[f64], z: &[f64], rho: f64) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut delta = vec![0.0f64; k * k];
    let mut dlam = vec![0.0f64; k];
    let mut infos = vec![0usize; k];
    for j in 1..=k {
        let mut dd = d.to_vec();
        let mut zz = z.to_vec();
        let mut col = vec![0.0f64; k];
        let (lam, info) = unsafe { t::dlaed4(k, j, &mut dd, &mut zz, &mut col, rho) };
        dlam[j - 1] = lam;
        infos[j - 1] = info;
        delta[(j - 1) * k..j * k].copy_from_slice(&col);
    }
    (delta, dlam, infos)
}

/// The host `W` step (`dlaed3.f` loop 60 / `DLAED3` `K > 2` branch,
/// transcribed in `src/stedc.rs::dlaed3`): `W(I) = DELTA(I,I) *
/// prod_{J != I} DELTA(J,I) / (DLAMBDA(I) - DLAMBDA(J))`, signed by the
/// original `W(I)`.
fn host_w(k: usize, delta: &[f64], d: &[f64], z: &[f64]) -> Vec<f64> {
    let mut w = vec![0.0f64; k];
    for i in 0..k {
        let mut wi = delta[i * k + i];
        let li = d[i];
        for j in 0..k {
            if j != i {
                wi *= delta[j * k + i] / (li - d[j]);
            }
        }
        w[i] = (-wi).sqrt().copysign(z[i]);
    }
    w
}

/// The host `U` step (`dlaed3.f` loop 110): `S(I) = W(I)/DELTA(I,J)`,
/// `TEMP = DNRM2(K, S)`, `U(I,J) = S(INDX(I))/TEMP`.
fn host_u(k: usize, w: &[f64], delta: &[f64], indx: &[u32]) -> Vec<f64> {
    let mut u = vec![0.0f64; k * k];
    for j in 0..k {
        let mut s = vec![0.0f64; k];
        for i in 0..k {
            s[i] = w[i] / delta[j * k + i];
        }
        let temp = unsafe {
            let mut ss = s.clone();
            t::dnrm2(k, &mut ss, 1)
        };
        for i in 0..k {
            u[j * k + i] = s[indx[i] as usize] / temp;
        }
    }
    u
}

fn run_roots(
    c: &ComputeClient<CpuRuntime>,
    k: usize,
    d: &[f64],
    z: &[f64],
    rho: f64,
) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    let cst = consts();
    let h_d = upload(c, d);
    let h_z = upload(c, z);
    let delta = c.empty(k * k * 8);
    let dlam = c.empty(k * 8);
    let info = c.empty(k * 4);
    ops::secular_roots::<CpuRuntime, f64>(c, &h_d, &h_z, &delta, &dlam, &info, k, rho, &cst);
    let mut got_delta = vec![0.0f64; k * k];
    let mut got_dlam = vec![0.0f64; k];
    let mut got_info = vec![0u32; k];
    read_range(c, &delta, k * k, 0, k * k, &mut got_delta);
    read_range(c, &dlam, k, 0, k, &mut got_dlam);
    read_range(c, &info, k, 0, k, &mut got_info);
    (got_delta, got_dlam, got_info)
}

/// `secular_roots_kernel` (inlines `dlaed4_mid`, `dlaed4_last`, `dlaed6_c`)
/// now compiles AND runs correctly on the CubeCL CPU runtime. Two distinct
/// upstream bugs blocked this and both are now fixed:
///
/// 1. `dlaed4_mid` had four descending `while jrN >= bound { ..; jrN -= 1 }`
///    loops (the reference `DLAED4`'s phi/dphi/erretm sums,
///    `src/device/kernels/secular.rs` around `jr1`..`jr4`). cubecl-cpu 0.10
///    cannot lower ANY `while` loop whose counter is decremented — ascending
///    `while`, flag-terminated `while`, nested `while` and `#[cube]` helper
///    calls are all fine, only the decrement breaks, with `cubecl-cpu-
///    0.10.0/src/compiler/module.rs:94:13: failed to run pass` / "operation
///    with block successors must terminate its parent block" (regression
///    test: `tests/cubecl_cpu_limits.rs::descending_while_counter_fails_on_
///    cpu_runtime`, `#[ignore]`d — a genuine, permanent backend limitation).
///    The four loops are now ascending `for t in bound..n + 1 { let jrN = n
///    + bound - t; .. }`, which walks the IDENTICAL `n, n-1, .., bound`
///    visit order (bit-exact against the original form, checked standalone
///    in plain Rust for many `(n, bound)` pairs including the empty- and
///    full-range edges) and compiles.
///
/// 2. With every descending `while` gone, `dlaed4_mid` (and hence
///    `secular_roots_kernel`, which inlines it) still panicked, with
///    `cubecl-opt-0.10.0/src/gvn/convert.rs:22:39: no entry found for key`
///    — a cubecl-opt PRE/GVN bug, root-caused in
///    `cubecl-opt-0.10.0/src/gvn/apply.rs::insert_block`: `partially_avail`
///    marked an expression for hoisting whenever ANY predecessor had it
///    available, then materialised it in every predecessor lacking it via
///    `Expression::to_operation(leaders)` — but a predecessor could lack
///    leaders for that expression's own operands, and `leaders[val]`
///    panicked. Fixed by vendoring the crate at `crates/cubecl-opt`
///    (`[patch.crates-io]` in `Cargo.toml`): the patch filters
///    `partially_avail` down to expressions every predecessor can actually
///    materialise. Skipping a hoist only forgoes an optimisation — PRE/CSE
///    never reassociates, so no floating-point value can change (see
///    `crates/cubecl-opt/PATCHED.md`).
///
/// `[patch.crates-io]` only applies to builds from this workspace, so
/// `src/device/stedc.rs::DeviceDc::secular` gates K4 on
/// `device::probe::k4_supported` (a runtime capability probe), not on this
/// crate compiling — a downstream consumer without the patch gets `false`
/// and falls back to the host instead of a wrong answer.
#[test]
fn secular_roots_kernel_cpu_runtime_matches_host() {
    let c = client();
    for &k in &[3usize, 5, 17, 64] {
        let (d, z, rho) = fixture(k, 1000 + k as u64);
        let (got_delta, got_dlam, got_info) = run_roots(&c, k, &d, &z, rho);
        let (want_delta, want_dlam, want_info) = host_roots(k, &d, &z, rho);
        println!(
            "k={k}: got_dlam[0..3.min(k)]={:?} want_dlam[0..3.min(k)]={:?} got_info={:?} want_info={:?}",
            &got_dlam[..3.min(k)],
            &want_dlam[..3.min(k)],
            &got_info[..3.min(k)],
            &want_info[..3.min(k)]
        );
        assert_bits(&format!("dlam k={k}"), &got_dlam, &want_dlam);
        assert_bits(&format!("delta k={k}"), &got_delta, &want_delta);
        assert_eq!(got_info, want_info.iter().map(|&v| v as u32).collect::<Vec<_>>(), "info k={k}");
    }
}

/// `secular_w_kernel`/`secular_u_kernel` are small, loop-only (no `while`,
/// no early `break`) kernels that always compiled and ran correctly on
/// cubecl-cpu, fed with host-computed roots. Kept as a standalone check of
/// the `W`/`U` steps in isolation now that `secular_roots_kernel` (above)
/// is also validated.
#[test]
fn secular_w_and_u_kernels_cpu_runtime_match_host() {
    let c = client();
    let cst = consts();
    for &k in &[3usize, 5, 17, 64] {
        let (d, z, rho) = fixture(k, 2000 + k as u64);
        let (delta, _dlam, infos) = host_roots(k, &d, &z, rho);
        assert!(infos.iter().all(|&i| i == 0), "host DLAED4 non-convergence in fixture k={k}");
        let want_w = host_w(k, &delta, &d, &z);
        let neg: Vec<u32> = want_w.iter().map(|x| x.is_sign_negative() as u32).collect();
        let indx: Vec<u32> = (0..k as u32).collect();
        let want_u = host_u(k, &want_w, &delta, &indx);

        let h_delta = upload(&c, &delta);
        let h_d = upload(&c, &d);
        let h_neg = lapack_rs::device::workspace::upload_vec(&c, neg);
        let h_indx = lapack_rs::device::workspace::upload_vec(&c, indx);
        let w_handle = c.empty(k * 8);
        ops::secular_w::<CpuRuntime, f64>(&c, &h_delta, &h_d, &h_neg, &w_handle, k);
        let mut got_w = vec![0.0f64; k];
        read_range(&c, &w_handle, k, 0, k, &mut got_w);
        assert_bits(&format!("W k={k}"), &got_w, &want_w);

        // `secular_u` overwrites `delta` in place with `S`, exactly as the
        // host loop 110 does; re-upload a fresh copy.
        let h_delta2 = upload(&c, &delta);
        let u_handle = c.empty(k * k * 8);
        ops::secular_u::<CpuRuntime, f64>(&c, &w_handle, &h_indx, &h_delta2, &u_handle, k, &cst);
        let mut got_u = vec![0.0f64; k * k];
        read_range(&c, &u_handle, k * k, 0, k * k, &mut got_u);
        assert_bits(&format!("U k={k}"), &got_u, &want_u);
    }
}

/// Restores [`lapack_rs::device::stedc::set_secular_min_k`]'s process-global
/// override even if an assertion panics.
struct RestoreMinK;

impl Drop for RestoreMinK {
    fn drop(&mut self) {
        lapack_rs::device::stedc::set_secular_min_k(0);
    }
}

/// Fixture names (from `tridiag_fixtures`) that never reach a `K > 2`
/// merge on these sizes: `tight_cluster` and `repeated` deflate every
/// merge down to `K <= 2` (dense near-duplicate/repeated eigenvalues), and
/// `diagonal` (every off-diagonal exactly zero) never merges at all. Their
/// `dstedc_device` run takes the SAME code path — the host fallback —
/// whether `set_secular_min_k` is 3 or `usize::MAX`, so the `launches`
/// transfer counter is equal, not merely `<=`, on these names. Verified
/// empirically (`launches` printed for both settings across every fixture
/// at n = 27, 45, 64, 100, 60, 151): every other name here strictly
/// increases `launches` under `min_k = 3`.
const NEVER_REACHES_K4: &[&str] = &["tight_cluster", "repeated", "diagonal"];

/// Full-driver check, modelled on `tests/device_cuda.rs`'s
/// `cuda_secular_equation_matches_host`: `dstedc_device` is compared
/// against the host `lapack_rs::blocks::dstedc` bit for bit under BOTH
/// `set_secular_min_k(3)` (every `K > 2` merge routed through K4 on the CPU
/// runtime) and `set_secular_min_k(usize::MAX)` (host `DLAED3` only, the
/// same fallback `k4_supported() == false` would take).
///
/// Bit-exactness alone does not prove K4 ran: the host fallback produces
/// the identical bits by construction, so a regression that silently
/// disabled K4 (e.g. `probe::k4_supported` wrongly returning `false`, or
/// the `g.k < min_k` gate mis-firing) would still pass a bits-only check.
/// `dstedc_device`'s returned `TransferCounters::launches` is what
/// distinguishes the two: K4 adds one `secular_roots`, one `secular_w` and
/// one `secular_u` kernel launch (plus a `dlam`/`info` readback) per
/// `K > 2` merge that the host path never issues, so on every fixture that
/// genuinely reaches such a merge, `min_k = 3` must launch strictly more
/// than `min_k = MAX` — not merely `>=`. [`NEVER_REACHES_K4`] fixtures
/// (which never reach `K > 2` regardless of the threshold) assert equality
/// instead, so the check stays honest rather than being weakened globally.
#[test]
fn secular_roots_kernel_cpu_runtime_drives_dstedc_device() {
    let c = client();
    let _restore = RestoreMinK;
    let mut cases = Vec::new();
    for &n in &[27usize, 45, 64, 100] {
        cases.extend(tridiag_fixtures(n).into_iter().map(|(name, d, e)| (name.to_string(), n, d, e)));
    }
    for &n in &[60usize, 151] {
        let (d, e) = random_tridiag(n, 9200 + n as u64);
        cases.push(("random".to_string(), n, d, e));
    }
    for (name, n, d0, e0) in cases {
        let (lw, liw) = lapack_rs::stedc_work_len(n);
        let (mut dh, mut eh) = (d0.clone(), e0.clone());
        eh.push(0.0);
        let mut zh = vec![0.0; n * n];
        let (mut w, mut iw) = (vec![0.0; lw.max(1)], vec![0usize; liw.max(1)]);
        lapack_rs::blocks::dstedc(n, &mut dh, &mut eh, &mut zh, &mut w, &mut iw).expect("host");

        lapack_rs::device::stedc::set_secular_min_k(3);
        let (mut d_k4, mut e_k4) = (d0.clone(), e0.clone());
        e_k4.push(0.0);
        let mut z_k4 = vec![0.0; n * n];
        let ctr_k4 = lapack_rs::device::dstedc_device(&c, n, &mut d_k4, &mut e_k4, &mut z_k4, 0).expect("device k4");

        lapack_rs::device::stedc::set_secular_min_k(usize::MAX);
        let (mut d_host, mut e_host) = (d0.clone(), e0.clone());
        e_host.push(0.0);
        let mut z_host = vec![0.0; n * n];
        let ctr_host =
            lapack_rs::device::dstedc_device(&c, n, &mut d_host, &mut e_host, &mut z_host, 0).expect("device host");

        let tag = format!("K4 cpu {name} n={n}");
        assert_bits(&format!("d k4 {tag}"), &d_k4, &dh);
        assert_bits(&format!("Z k4 {tag}"), &z_k4, &zh);
        assert_bits(&format!("d host {tag}"), &d_host, &dh);
        assert_bits(&format!("Z host {tag}"), &z_host, &zh);

        println!("{tag}: k4 launches={} host launches={}", ctr_k4.launches, ctr_host.launches);
        if NEVER_REACHES_K4.contains(&name.as_str()) {
            assert_eq!(
                ctr_k4.launches, ctr_host.launches,
                "{tag}: expected no K>2 merge (launches should match the host path exactly)"
            );
        } else {
            assert!(
                ctr_k4.launches > ctr_host.launches,
                "{tag}: min_k=3 launches ({}) must exceed min_k=MAX launches ({}) — K4 was not demonstrably taken",
                ctr_k4.launches,
                ctr_host.launches
            );
        }
    }
}

/// `device::probe::k4_supported` must report `true` on the CPU runtime in
/// this workspace: the vendored `crates/cubecl-opt` patch
/// (`crates/cubecl-opt/PATCHED.md`) is what makes that so. If a future
/// cubecl/cubecl-opt bump or an accidental edit to the vendored crate
/// regresses the underlying GVN fix, this test fails loudly here instead of
/// K4 silently falling back to the host path (which would still be
/// bit-exact, just slower, and would hide the regression).
#[test]
fn k4_supported_reports_true_on_cpu_runtime() {
    let c = client();
    assert!(
        lapack_rs::device::probe::k4_supported(&c),
        "K4 capability probe reports unsupported on the CPU runtime: either the vendored \
         crates/cubecl-opt patch regressed, or cubecl-opt reverted upstream (see \
         crates/cubecl-opt/PATCHED.md)"
    );
}
