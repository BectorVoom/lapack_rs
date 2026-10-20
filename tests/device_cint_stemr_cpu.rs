//! **K9, Path C.** The libcint 6.1.3 leaf kernels launched on the CubeCL CPU
//! runtime (f64) against the host transcription (`src/cint_stemr.rs`), bit for
//! bit — the part of task **C2** the CPU runtime can carry.
//!
//! The oracle chain is `crates/cint_ref` (the C) == `lapack_rs::cint_stemr`
//! (host, gated by `tests/cint_stemr_bitexact.rs`) == the device kernels
//! (gated here).
//!
//! **What is *not* here, and why.** `cint_dlarrb_c`, `cint_dlasq5_c`, the dqds
//! engine `cint_dlasq2_c` and the two composite routines
//! `cint_compute_eigenvalues_c` / `cint_compute_eigenvectors_c` hit a
//! cubecl-cpu 0.10 control-flow lowering bug: the launch prints `operation with block successors must terminate
//! its parent block` on a worker thread and **silently no-ops**. That is a
//! backend limitation, not a kernel one — hipRTC lowers and launches the whole
//! `cint_stemr_dev`. `docs/cubecl-cpu-lowering-limit.md` has the bisect, the
//! probes showing each idiom is individually fine, and the cross-check;
//! `tests/cubecl_cpu_limits.rs` keeps the reproduction. The composite's
//! bit-exactness gate is therefore `tests/device_cuda.rs` (f64) with
//! `tests/device_rocm.rs` for the f32 shape gate.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;
use lapack_rs::device::kernels::cint_stemr as k;
use lapack_rs::device::workspace::{read_range, upload};
use lapack_rs::testing as t;

fn client() -> ComputeClient<CpuRuntime> {
    CpuRuntime::client(&CpuDevice)
}

fn up(c: &ComputeClient<CpuRuntime>, v: &[f64]) -> Handle {
    upload(c, v)
}
fn upi(c: &ComputeClient<CpuRuntime>, v: &[u32]) -> Handle {
    upload(c, v)
}
fn down(c: &ComputeClient<CpuRuntime>, h: &Handle, n: usize) -> Vec<f64> {
    let mut o = vec![0.0f64; n];
    read_range(c, h, n, 0, n, &mut o);
    o
}
fn downi(c: &ComputeClient<CpuRuntime>, h: &Handle, n: usize) -> Vec<u32> {
    let mut o = vec![0u32; n];
    read_range(c, h, n, 0, n, &mut o);
    o
}

/// A tridiagonal and its `L D L^T` factors, as `_compute_eigenvectors` builds
/// them (`buf_ld[j] = d[j]*e[j]`, `buf_lld[j] = buf_ld[j]*e[j]`).
fn fixture(n: usize, kind: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut r = Rng::new(50_000 + seed * 7919 + n as u64 * 31 + kind as u64);
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    for i in 0..n {
        d[i] = match kind {
            0 => r.uniform(),
            1 => i as f64 + 1.0,
            2 => 1.0,
            3 => 1.0 + 1e-13 * (i / 2) as f64,
            4 => 10f64.powi(-(i as i32 % 8)),
            _ => r.uniform().abs() + 0.25,
        };
        if i + 1 < n {
            e[i] = match kind {
                0 => r.uniform(),
                1 => 1.0,
                2 => 1e-16,
                3 => 1e-14,
                4 => 10f64.powi(-(i as i32 % 8) - 1),
                _ => r.uniform(),
            };
        }
    }
    let e2: Vec<f64> = e.iter().map(|v| v * v).collect();
    let ld: Vec<f64> = (0..n).map(|i| d[i] * e[i]).collect();
    let lld: Vec<f64> = (0..n).map(|i| ld[i] * e[i]).collect();
    (d, e, e2, ld, lld)
}

// --- one-problem launch wrappers over each `#[cube]` leaf ------------------

#[cube(launch, launch_unchecked)]
fn p_larrk<F: Float + CubeElement>(
    diag: &Array<F>,
    work: &Array<F>,
    sc: &mut Array<F>,
    info: &mut Array<u32>,
    n: u32,
    iw: u32,
    gl: F,
    gu: F,
    reltol: F,
    eps: F,
) {
    if ABSOLUTE_POS < 1usize {
        info[0] = k::cint_dlarrk_c::<F>(diag, 0u32, work, 0u32, n, iw, gl, gu, reltol, eps, sc, 0u32);
    }
}

#[cube(launch, launch_unchecked)]
fn p_larrc<F: Float + CubeElement>(diag: &Array<F>, work: &Array<F>, icnt: &mut Array<u32>, n: u32, vl: F, vu: F) {
    if ABSOLUTE_POS < 1usize {
        k::cint_dlarrc_c::<F>(n, vl, vu, diag, 0u32, work, 0u32, icnt, 0u32);
    }
}

#[cube(launch, launch_unchecked)]
fn p_laneg<F: Float + CubeElement>(
    diag: &Array<F>,
    work: &Array<F>,
    out: &mut Array<u32>,
    n: u32,
    sigma: F,
    twist: u32,
) {
    if ABSOLUTE_POS < 1usize {
        out[0] = k::cint_dlaneg_c::<F>(diag, 0u32, work, 0u32, n, sigma, twist);
    }
}

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn p_larrf<F: Float + CubeElement>(
    diag: &Array<F>,
    offd: &Array<F>,
    work: &Array<F>,
    cc: &Array<F>,
    vec: &mut Array<F>,
    sc: &mut Array<F>,
    info: &mut Array<u32>,
    n: u32,
    clstrt: u32,
    clgapl: F,
    eps: F,
) {
    if ABSOLUTE_POS < 1usize {
        // work: 0..n ld | n..2n w | 2n..3n+1 wgap(-1 based) | 3n+1..4n+1 werr
        // vec:  0..n dplus | n..2n lplus
        info[0] = k::cint_dlarrf_c::<F>(
            diag,
            0u32,
            offd,
            0u32,
            work,
            0u32,
            n,
            n * 2u32,
            n * 3u32 + 1u32,
            n,
            clstrt,
            clgapl,
            eps,
            vec,
            0u32,
            n,
            cc,
            sc,
            0u32,
        );
    }
}

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn p_lar1v<F: Float + CubeElement>(
    diag: &Array<F>,
    offd: &Array<F>,
    work: &mut Array<F>,
    vec: &mut Array<F>,
    iwork: &mut Array<u32>,
    sc: &mut Array<F>,
    out: &mut Array<u32>,
    n: u32,
    lambda: F,
) {
    if ABSOLUTE_POS < 1usize {
        // work: 0..n ld | n..2n lld | 2n..5n scratch
        out[0] = k::cint_dlar1v_c::<F>(
            n,
            lambda,
            diag,
            0u32,
            offd,
            0u32,
            work,
            0u32,
            n,
            n * 2u32,
            vec,
            0u32,
            iwork,
            0u32,
            sc,
            0u32,
        );
    }
}

#[cube(launch, launch_unchecked)]
fn p_laev2<F: Float + CubeElement>(diag: &Array<F>, offd: &Array<F>, eig: &mut Array<F>, vec: &mut Array<F>) {
    if ABSOLUTE_POS < 1usize {
        k::cint_dlaev2_c::<F>(diag, 0u32, offd, 0u32, eig, 0u32, vec, 0u32);
    }
}

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn p_lasq4<F: Float + CubeElement>(
    work: &Array<F>,
    cc: &Array<F>,
    sc: &mut Array<F>,
    n: u32,
    i0: u32,
    n0: u32,
    n0init: u32,
) {
    if ABSOLUTE_POS < 1usize {
        k::cint_dlasq4_c::<F>(work, 0u32, n, n * 2u32, n * 3u32, i0, n0, n0init, cc, sc, 0u32);
    }
}

const CC: CubeCount = CubeCount::Static(1, 1, 1);

#[test]
fn cint_leaf_kernels_match_host() {
    let c = client();
    let dim = CubeDim::new_1d(1);
    let eps = f64::EPSILON;
    // libcint's non-dyadic constants: they cannot be literals in a `#[cube]`
    // body because `F::new` takes an `f32` (see `ops::cint_stemr_consts`).
    let consts = lapack_rs::device::ops::cint_stemr_consts();
    let hcc = up(&c, &consts);
    let mut checked = 0usize;

    for n in 3..=lapack_rs::MXRYSROOTS {
        for kind in 0..6usize {
            for seed in 0..3u64 {
                let (d0, e0, e2, ld, lld) = fixture(n, kind, seed);
                let tag = format!("n={n} kind={kind} seed={seed}");

                // Gerschgorin bracket, as `_compute_eigenvalues` builds it.
                let (mut gl, mut gu) = (d0[0], d0[0]);
                let mut eold = 0.0;
                for i in 0..n {
                    let t1 = e0[i].abs() + eold;
                    gl = gl.min(d0[i] - t1);
                    gu = gu.max(d0[i] + t1);
                    eold = e0[i].abs();
                }

                // --- _dlarrk -------------------------------------------------
                let hdiag = up(&c, &d0);
                let he2 = up(&c, &e2);
                for reltol in [2.1e-8f64, 1e-14, 1e-3] {
                    for iw in [1usize, (n + 1) / 2, n] {
                        let hsc = up(&c, &vec![0.0f64; 16]);
                        let hinf = upi(&c, &[0u32; 4]);
                        // SAFETY: lengths match the buffers.
                        unsafe {
                            p_larrk::launch_unchecked::<f64, CpuRuntime>(
                                &c,
                                CC,
                                dim,
                                ArrayArg::from_raw_parts(hdiag.clone(), n),
                                ArrayArg::from_raw_parts(he2.clone(), n),
                                ArrayArg::from_raw_parts(hsc.clone(), 16),
                                ArrayArg::from_raw_parts(hinf.clone(), 4),
                                n as u32,
                                iw as u32,
                                gl,
                                gu,
                                reltol,
                                eps,
                            );
                        }
                        let got = down(&c, &hsc, 16);
                        let gi = downi(&c, &hinf, 4);
                        let (w, werr, info) =
                            unsafe { t::cint_dlarrk(n, iw as i32, gl, gu, &mut d0.clone(), &mut e2.clone(), reltol) };
                        assert_bits(&format!("cint_dlarrk_c {tag} iw={iw} rt={reltol:e}"), &got[7..9], &[w, werr]);
                        assert_eq!(gi[0] as i32, if info == 0 { 0 } else { 1 }, "cint_dlarrk_c INFO {tag}");
                    }
                }

                // --- _dlarrc -------------------------------------------------
                for &(vl, vu) in &[(gl, gu), (-1.0, 1.0), (0.0, 0.5)] {
                    let hic = upi(&c, &[0u32; 4]);
                    // SAFETY: lengths match the buffers.
                    unsafe {
                        p_larrc::launch_unchecked::<f64, CpuRuntime>(
                            &c,
                            CC,
                            dim,
                            ArrayArg::from_raw_parts(hdiag.clone(), n),
                            ArrayArg::from_raw_parts(he2.clone(), n),
                            ArrayArg::from_raw_parts(hic.clone(), 4),
                            n as u32,
                            vl,
                            vu,
                        );
                    }
                    let gi = downi(&c, &hic, 4);
                    let want = unsafe { t::cint_dlarrc(n, vl, vu, &mut d0.clone(), &mut e2.clone()) };
                    assert_eq!((gi[0] as i32, gi[1] as i32), want, "cint_dlarrc_c {tag} vl={vl:e} vu={vu:e}");
                }

                // --- _dlaneg -------------------------------------------------
                let hlld = up(&c, &lld);
                for sigma in [0.0f64, 0.5, -0.5, 1e-14] {
                    for twist in [1usize, (n + 1) / 2, n] {
                        let hout = upi(&c, &[0u32; 4]);
                        // SAFETY: lengths match the buffers.
                        unsafe {
                            p_laneg::launch_unchecked::<f64, CpuRuntime>(
                                &c,
                                CC,
                                dim,
                                ArrayArg::from_raw_parts(hdiag.clone(), n),
                                ArrayArg::from_raw_parts(hlld.clone(), n),
                                ArrayArg::from_raw_parts(hout.clone(), 4),
                                n as u32,
                                sigma,
                                twist as u32,
                            );
                        }
                        let gi = downi(&c, &hout, 4);
                        let want = unsafe { t::cint_dlaneg(n, &mut d0.clone(), &mut lld.clone(), sigma, twist as i32) };
                        assert_eq!(gi[0] as i32, want, "cint_dlaneg_c {tag} sigma={sigma:e} r={twist}");
                    }
                }

                // --- _dlarrb, _dlasq5: CPU-runtime-blocked ------------------
                // `cint_dlarrb_c` and `cint_dlasq5_c` hit the cubecl-cpu 0.10
                // control-flow lowering bug: the launch prints `operation with
                // block successors must terminate its parent block` and
                // silently no-ops (`docs/cubecl-cpu-lowering-limit.md`). Their
                // gate is `tests/device_cuda.rs` / `tests/device_rocm.rs`, and
                // the host side is covered by `tests/cint_stemr_bitexact.rs`,
                // which drives them through the full solver on 4 610 Wheeler
                // tridiagonals at `max_ulp = 0`.
                let w0: Vec<f64> = (0..n).map(|i| d0[i]).collect();
                let wgap0: Vec<f64> = (0..=n).map(|_| 1e-3).collect();
                let werr0: Vec<f64> = (0..n).map(|_| 1e-2).collect();
                // --- _dlarrf -------------------------------------------------
                let mut fbuf = Vec::new();
                fbuf.extend_from_slice(&ld);
                fbuf.extend_from_slice(&w0);
                fbuf.extend_from_slice(&wgap0);
                fbuf.extend_from_slice(&werr0);
                let hoffd = up(&c, &e0);
                for clgapl in [1e-3f64, 1e-12, 1.0] {
                    for clstrt in [0usize, n / 2] {
                        let hf = up(&c, &fbuf);
                        let hv = up(&c, &vec![0.0f64; 2 * n]);
                        let hsc = up(&c, &vec![0.0f64; 16]);
                        let hinf = upi(&c, &[0u32; 4]);
                        // SAFETY: lengths match the buffers.
                        unsafe {
                            p_larrf::launch_unchecked::<f64, CpuRuntime>(
                                &c,
                                CC,
                                dim,
                                ArrayArg::from_raw_parts(hdiag.clone(), n),
                                ArrayArg::from_raw_parts(hoffd.clone(), n),
                                ArrayArg::from_raw_parts(hf.clone(), fbuf.len()),
                                ArrayArg::from_raw_parts(hcc.clone(), consts.len()),
                                ArrayArg::from_raw_parts(hv.clone(), 2 * n),
                                ArrayArg::from_raw_parts(hsc.clone(), 16),
                                ArrayArg::from_raw_parts(hinf.clone(), 4),
                                n as u32,
                                clstrt as u32,
                                clgapl,
                                eps,
                            );
                        }
                        let gv = down(&c, &hv, 2 * n);
                        let gsc = down(&c, &hsc, 16);
                        let gi = downi(&c, &hinf, 4);
                        let (mut hdp, mut hlp) = (vec![0.0f64; n], vec![0.0f64; n]);
                        let want = unsafe {
                            t::cint_dlarrf(
                                n,
                                &mut d0.clone(),
                                &mut e0.clone(),
                                &mut ld.clone(),
                                clstrt,
                                &mut w0.clone(),
                                &mut wgap0.clone(),
                                &mut werr0.clone(),
                                clgapl,
                                &mut hdp,
                                &mut hlp,
                            )
                        };
                        let tg = format!("cint_dlarrf_c {tag} clgapl={clgapl:e} clstrt={clstrt}");
                        assert_eq!(gi[0] as i32, want.1, "{tg} INFO");
                        assert_bits(&format!("{tg} SIGMA"), &gsc[9..10], &[want.0]);
                        assert_bits(&format!("{tg} DPLUS"), &gv[..n], &hdp);
                        assert_bits(&format!("{tg} LPLUS"), &gv[n..2 * n - 1], &hlp[..n - 1]);
                    }
                }

                // --- _dlar1v -------------------------------------------------
                let mut rbuf = Vec::new();
                rbuf.extend_from_slice(&ld);
                rbuf.extend_from_slice(&lld);
                rbuf.extend_from_slice(&vec![0.0f64; 3 * n]);
                for lambda in [0.0f64, 0.5, -0.25] {
                    for twist_in in [-1i32, 0, (n as i32) / 2] {
                        let hr = up(&c, &rbuf);
                        let hv = up(&c, &vec![0.0f64; n]);
                        let hiw = upi(&c, &[if twist_in < 0 { u32::MAX } else { twist_in as u32 }; 4]);
                        let hsc = up(&c, &vec![0.0f64; 16]);
                        let hout = upi(&c, &[0u32; 4]);
                        // SAFETY: lengths match the buffers.
                        unsafe {
                            p_lar1v::launch_unchecked::<f64, CpuRuntime>(
                                &c,
                                CC,
                                dim,
                                ArrayArg::from_raw_parts(hdiag.clone(), n),
                                ArrayArg::from_raw_parts(hoffd.clone(), n),
                                ArrayArg::from_raw_parts(hr.clone(), rbuf.len()),
                                ArrayArg::from_raw_parts(hv.clone(), n),
                                ArrayArg::from_raw_parts(hiw.clone(), 4),
                                ArrayArg::from_raw_parts(hsc.clone(), 16),
                                ArrayArg::from_raw_parts(hout.clone(), 4),
                                n as u32,
                                lambda,
                            );
                        }
                        let gr = down(&c, &hr, rbuf.len());
                        let gv = down(&c, &hv, n);
                        let gsc = down(&c, &hsc, 16);
                        let gi = downi(&c, &hout, 4);
                        let gtw = downi(&c, &hiw, 4);
                        let mut hv_ = vec![0.0f64; n];
                        let mut hwk = vec![0.0f64; 3 * n];
                        let want = unsafe {
                            t::cint_dlar1v(
                                n,
                                lambda,
                                &mut d0.clone(),
                                &mut e0.clone(),
                                &mut ld.clone(),
                                &mut lld.clone(),
                                &mut hv_,
                                twist_in,
                                &mut hwk,
                            )
                        };
                        let tg = format!("cint_dlar1v_c {tag} lam={lambda:e} tw={twist_in}");
                        assert_eq!(gi[0] as i32, want.0, "{tg} NEGCNT");
                        assert_eq!(gtw[0] as i32, want.1, "{tg} TWIST");
                        assert_bits(&format!("{tg} VEC"), &gv, &hv_);
                        assert_bits(&format!("{tg} WORK"), &gr[2 * n..5 * n], &hwk);
                        assert_bits(&format!("{tg} RESID/RQCORR"), &gsc[10..12], &[want.2, want.3]);
                    }
                }

                // --- _dlasq4 / _dlasq5 ---------------------------------------
                // work: qvec | qvec1 | evec | evec1, positive as dqds needs.
                let mut qbuf = vec![0.0f64; 4 * n];
                let mut rr = Rng::new(900 + n as u64 + kind as u64);
                for i in 0..n {
                    qbuf[i] = rr.uniform().abs() + 0.5;
                    qbuf[n + i] = rr.uniform().abs() + 0.5;
                    qbuf[2 * n + i] = if i + 1 < n { rr.uniform().abs() } else { 0.0 };
                    qbuf[3 * n + i] = if i + 1 < n { rr.uniform().abs() } else { 0.0 };
                }
                let n0 = n;
                for n0init in [n0, n0 + 1, n0 + 2, n0 + 3] {
                    for case in 0..4usize {
                        let dn = [rr.uniform(), rr.uniform(), rr.uniform()];
                        let dmin = match case {
                            0 => [dn[0], dn[1], dn[2]],
                            1 => [dn[1], dn[1], dn[2]],
                            2 => [dn[2], rr.uniform(), dn[2]],
                            _ => [rr.uniform(), rr.uniform(), rr.uniform()],
                        };
                        let hq = up(&c, &qbuf);
                        let mut sc0 = vec![0.0f64; 16];
                        sc0[0..3].copy_from_slice(&dmin);
                        sc0[3..6].copy_from_slice(&dn);
                        sc0[6] = 0.125;
                        let hsc = up(&c, &sc0);
                        // SAFETY: lengths match the buffers.
                        unsafe {
                            p_lasq4::launch_unchecked::<f64, CpuRuntime>(
                                &c,
                                CC,
                                dim,
                                ArrayArg::from_raw_parts(hq.clone(), 4 * n),
                                ArrayArg::from_raw_parts(hcc.clone(), consts.len()),
                                ArrayArg::from_raw_parts(hsc.clone(), 16),
                                n as u32,
                                0,
                                n0 as u32,
                                n0init as u32,
                            );
                        }
                        let gsc = down(&c, &hsc, 16);
                        let want = unsafe {
                            t::cint_dlasq4(&mut qbuf.clone(), 0, n, 2 * n, 3 * n, 0, n0, n0init, dmin, dn, 0.125)
                        };
                        assert_bits(&format!("cint_dlasq4_c {tag} n0init={n0init} case={case}"), &gsc[6..7], &[want]);
                    }
                }
                checked += 1;
            }
        }
    }

    // --- _dlaev2, the n = 2 closed form ------------------------------------
    let mut r = Rng::new(60_001);
    for _ in 0..200 {
        let s = |r: &mut Rng| (2.0 * r.uniform() - 1.0) * 10f64.powi((6.0 * r.uniform() - 3.0) as i32);
        for (a, b, cc) in [(s(&mut r), s(&mut r), s(&mut r)), (1.0, 0.0, 1.0), (1.0, 0.0, -1.0), (0.0, 1.0, 0.0)] {
            let hd = up(&c, &[a, cc]);
            let ho = up(&c, &[b, 0.0]);
            let he = up(&c, &[0.0f64; 2]);
            let hv = up(&c, &[0.0f64; 4]);
            // SAFETY: lengths match the buffers.
            unsafe {
                p_laev2::launch_unchecked::<f64, CpuRuntime>(
                    &c,
                    CC,
                    dim,
                    ArrayArg::from_raw_parts(hd.clone(), 2),
                    ArrayArg::from_raw_parts(ho.clone(), 2),
                    ArrayArg::from_raw_parts(he.clone(), 2),
                    ArrayArg::from_raw_parts(hv.clone(), 4),
                );
            }
            let ge = down(&c, &he, 2);
            let gv = down(&c, &hv, 4);
            let (mut heig, mut hvec) = (vec![0.0f64; 2], vec![0.0f64; 4]);
            unsafe { t::cint_dlaev2(&mut heig, &mut hvec, &mut [a, cc], &mut [b, 0.0]) };
            let tg = format!("cint_dlaev2_c a={a:e} b={b:e} c={cc:e}");
            assert_bits(&format!("{tg} EIG"), &ge, &heig);
            assert_bits(&format!("{tg} VEC"), &gv, &hvec);
        }
    }

    assert!(checked > 100, "only {checked} fixtures checked");
    eprintln!("K9 leaves on the CPU runtime: {checked} fixtures, bit-identical to the host");
}

/// The K9 capability probe must report **`false`** on the CubeCL CPU runtime,
/// because cubecl-cpu 0.10 cannot lower the composite kernel and the launch
/// silently no-ops (`docs/cubecl-cpu-lowering-limit.md`).
///
/// This is the assertion that keeps a production caller safe: without the
/// probe it would read zeros and call them eigenvalues. If a future cubecl
/// fixes the lowering, this test starts failing — drop the GPU-only gating
/// here and in the plan's §12 when it does.
#[test]
fn k9_supported_reports_false_on_cpu_runtime() {
    let client = client();
    assert!(
        !lapack_rs::device::probe::k9_supported(&client),
        "cubecl-cpu 0.10 now lowers the K9 composite: re-enable the CPU-runtime gates for \
         cint_stemr_dev and update docs/cubecl-cpu-lowering-limit.md"
    );
    // The verdict is cached per runtime type, so a second call must agree.
    assert!(!lapack_rs::device::probe::k9_supported(&client));
}
