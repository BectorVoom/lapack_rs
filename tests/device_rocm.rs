//! ROCm/HIP gates (feature `rocm`, a real AMD GPU):
//!
//! * f64 support and the contraction probe are reported (manual 15 §3.3);
//!   without f64 every f64 entry point returns `DeviceUnsupported`;
//! * K1/K2 in f32 against a host f32 transcription of the reference chains
//!   (bit for bit when the probe reports no contraction, else 1e-5);
//! * `dsyevd_device`/`zheevd_device` in f64 against the host path: bit for
//!   bit when HIP does not contract (`HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off`),
//!   otherwise eigenvalues ≤ 1e-10 relative and residuals ≤ 1e-9 (1e-10
//!   Hermitian); run-to-run bit identity always.
//!
//! Environment: `hipconfig` on `PATH` at build time and the ROCm `lib` on
//! `LD_LIBRARY_PATH` at run time (manual 15 §3).

#![cfg(feature = "rocm")]

mod common;
use common::*;
use cubecl::prelude::*;
use cubecl_hip::{AmdDevice, HipRuntime};
use lapack_rs::device::geometry::vector_width;
use lapack_rs::device::ops::{self, Lanes};
use lapack_rs::device::probe::{contraction_probe, supports_f64};
use lapack_rs::device::workspace::{read_range, upload};
use lapack_rs::device::{DeviceBudget, DeviceWorkspace, MergeU, dsyevd_device, zheevd_device};
use lapack_rs::{Job, LapackError, Uplo};

fn client() -> ComputeClient<HipRuntime> {
    HipRuntime::client(&AmdDevice::default())
}

fn exact_f64(c: &ComputeClient<HipRuntime>) -> bool {
    supports_f64(c) && contraction_probe::<HipRuntime, f64>(c, f64::EPSILON, 1.0, 2.0).no_contraction()
}

#[test]
fn rocm_reports_capabilities() {
    let c = client();
    let hw = &c.properties().hardware;
    println!(
        "hip: plane_size_max={} max_cube_count={:?} f64={} f64 widths={:?} f32 widths={:?}",
        hw.plane_size_max,
        hw.max_cube_count,
        supports_f64(&c),
        c.io_optimized_vector_sizes(8).collect::<Vec<_>>(),
        c.io_optimized_vector_sizes(4).collect::<Vec<_>>()
    );
    let p32 = contraction_probe::<HipRuntime, f32>(&c, f32::EPSILON, 1.0, 2.0);
    println!("contraction f32: {p32:?}");
    if supports_f64(&c) {
        let p64 = contraction_probe::<HipRuntime, f64>(&c, f64::EPSILON, 1.0, 2.0);
        println!("contraction f64: {p64:?}");
    } else {
        let n = 8;
        let (mut a, mut w) = (random_sym(n, 1), vec![0.0; n]);
        let mut ws = DeviceWorkspace::new(&c, n, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
        let r = dsyevd_device(&c, Job::Vectors, Uplo::Lower, n, &mut a, &mut w, &mut ws);
        assert!(matches!(r, Err(LapackError::DeviceUnsupported { .. })), "{r:?}");
    }
}

/// Reference `DSYMV` (`BETA = 0`, unit increments) in f32, the chain K1
/// reproduces.
fn ssymv(upper: bool, n: usize, alpha: f32, a: &[f32], x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    if upper {
        for j in 0..n {
            let t1 = alpha * x[j];
            let mut t2 = 0.0f32;
            for i in 0..j {
                y[i] += t1 * a[i + j * n];
                t2 += a[i + j * n] * x[i];
            }
            y[j] = y[j] + t1 * a[j + j * n] + alpha * t2;
        }
    } else {
        for j in 0..n {
            let t1 = alpha * x[j];
            let mut t2 = 0.0f32;
            y[j] += t1 * a[j + j * n];
            for i in j + 1..n {
                y[i] += t1 * a[i + j * n];
                t2 += a[i + j * n] * x[i];
            }
            y[j] += alpha * t2;
        }
    }
    y
}

#[test]
fn rocm_k1_symv_f32() {
    let c = client();
    let exact = contraction_probe::<HipRuntime, f32>(&c, f32::EPSILON, 1.0, 2.0).no_contraction();
    let width = vector_width(&c, 4);
    let lanes = Lanes::new(&c, width);
    for &(n, off) in &[(37usize, 0usize), (200, 13), (333, 100)] {
        let lda = n.div_ceil(width) * width;
        let a64 = random_sym(n, 40 + n as u64);
        let a: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
        let mut img = vec![0.0f32; lda * n];
        for j in 0..n {
            for i in 0..n {
                img[i + j * lda] = a[i + j * n];
            }
        }
        let m = n - off;
        let x: Vec<f32> = edge(m, 3).iter().map(|&v| v as f32).collect();
        let (hm, hx) = (upload(&c, &img), upload(&c, &x));
        for upper in [false, true] {
            let hy = c.empty(lda * 4);
            ops::symv::<HipRuntime, f32>(&c, &lanes, &hm, lda, n, off, m, &hx, &hy, -0.75, upper);
            let mut got = vec![0.0f32; m];
            read_range(&c, &hy, lda, off, m, &mut got);
            let mut blk = vec![0.0f32; m * m];
            for j in 0..m {
                for i in 0..m {
                    blk[i + j * m] = a[(off + i) + (off + j) * n];
                }
            }
            let want = ssymv(upper, m, -0.75, &blk, &x);
            for i in 0..m {
                if exact {
                    assert_eq!(got[i].to_bits(), want[i].to_bits(), "K1 f32 n={n} off={off} upper={upper} i={i}");
                } else {
                    // Contracted chains of m f32 terms: each fused step moves the
                    // result by at most ~eps·|term|, so bound by m·eps·Σ|terms|.
                    let scale: f32 = (0..m).map(|p| blk[i + p * m].abs() * x[p].abs()).sum::<f32>() * 0.75;
                    let tol = 4.0 * m as f32 * f32::EPSILON * scale.max(1.0);
                    assert!(
                        (got[i] - want[i]).abs() <= tol,
                        "K1 f32 n={n} i={i}: {} vs {} (tol {tol})",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }
    println!("K1 f32 on HIP: {}", if exact { "bit-exact" } else { "tolerance (contracting)" });
}

#[test]
fn rocm_dsyevd_and_zheevd_device() {
    let c = client();
    if !supports_f64(&c) {
        println!("SKIP: no f64 on this device");
        return;
    }
    let exact = exact_f64(&c);
    println!("f64 exactness on HIP: {}", if exact { "bit-exact gate" } else { "tolerance gate (contracting)" });
    let mut ws = DeviceWorkspace::new(&c, 64, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
    for &n in &[34usize, 100, 257] {
        let a0 = random_sym(n, 600 + n as u64);
        for uplo in [Uplo::Lower, Uplo::Upper] {
            let (mut ah, mut wh) = (a0.clone(), vec![0.0; n]);
            lapack_rs::dsyevd_alloc(Job::Vectors, uplo, n, &mut ah, &mut wh).expect("host");
            let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
            dsyevd_device(&c, Job::Vectors, uplo, n, &mut a, &mut w, &mut ws).expect("device");
            let (mut a2, mut w2) = (a0.clone(), vec![0.0; n]);
            dsyevd_device(&c, Job::Vectors, uplo, n, &mut a2, &mut w2, &mut ws).expect("device 2");
            let tag = format!("dsyevd n={n} {uplo:?}");
            assert_bits(&format!("run-to-run w {tag}"), &w2, &w);
            assert_bits(&format!("run-to-run z {tag}"), &a2, &a);
            if exact {
                assert_bits(&format!("w {tag}"), &w, &wh);
                assert_bits(&format!("z {tag}"), &a, &ah);
            } else {
                for i in 0..n {
                    assert!((w[i] - wh[i]).abs() <= 1e-10 * wh[i].abs().max(1.0), "{tag} w[{i}]");
                }
                let (res, orth) = sym_residuals(n, &a0, &a, &w);
                assert!(res <= 1e-9 && orth <= 1e-9, "{tag}: {res:e} {orth:e}");
            }
        }
        let (re0, im0) = random_herm(n, 700 + n as u64);
        let (mut hr, mut hi, mut wh) = (re0.clone(), im0.clone(), vec![0.0; n]);
        lapack_rs::zheevd_alloc(Job::Vectors, Uplo::Lower, n, &mut hr, &mut hi, &mut wh).expect("host");
        let (mut ar, mut ai, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
        zheevd_device(&c, Job::Vectors, Uplo::Lower, n, &mut ar, &mut ai, &mut w, &mut ws).expect("device");
        let tag = format!("zheevd n={n}");
        if exact {
            assert_bits(&format!("w {tag}"), &w, &wh);
            assert_bits(&format!("z re {tag}"), &ar, &hr);
            assert_bits(&format!("z im {tag}"), &ai, &hi);
        } else {
            for i in 0..n {
                assert!((w[i] - wh[i]).abs() <= 1e-10 * wh[i].abs().max(1.0), "{tag} w[{i}]");
            }
            let (res, orth) = herm_residuals(n, &re0, &im0, &ar, &ai, &w);
            assert!(res <= 1e-10 && orth <= 1e-10, "{tag}: {res:e} {orth:e}");
        }
    }
}

/// **Surface 1, the production shape.** A kernel defined *in this test crate*
/// — i.e. in a crate that merely depends on `lapack_rs`, exactly as cintx does
/// — calls `cint_stemr_dev` inside its own launch. No `launch` attribute on
/// the callee, no round trip: that is the whole point of
/// `docs/cintx-lapack-blas-adoption-plan.md` §6.2, where a launch plus a
/// blocking read-back is ~10^5 times the arithmetic of an `n <= 13` solve.
///
/// That this compiles at all is the deliverable: every type in the signature
/// and every constant slot has to be reachable from outside the crate.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn foreign_caller_kernel<F: Float + CubeElement>(
    diag: &mut Array<F>,
    offd: &mut Array<F>,
    eig: &mut Array<F>,
    vec: &mut Array<F>,
    work: &mut Array<F>,
    iwork: &mut Array<u32>,
    cc: &Array<F>,
    sc: &mut Array<F>,
    icnt: &mut Array<u32>,
    info: &mut Array<u32>,
    eps: F,
    big: F,
    n: u32,
) {
    if ABSOLUTE_POS < 1usize {
        // Some other per-quartet work would go here; the solve is inline.
        lapack_rs::device::kernels::cint_stemr::cint_stemr_dev::<F>(
            diag, 0u32, offd, 0u32, eig, 0u32, vec, 0u32, work, 0u32, iwork, 0u32, cc, sc, 0u32, icnt, 0u32, eps, big,
            info, 0u32, n,
        );
    }
}

/// The inline callee, launched from a foreign crate's own kernel, must give
/// the same bits as `lapack_rs`'s own one-problem launcher.
#[test]
fn cint_stemr_inline_callee_from_a_foreign_kernel_f32() {
    let c = client();
    use lapack_rs::device::kernels::cint_stemr as ck;
    let consts64 = ops::cint_stemr_consts();
    let mut consts: Vec<f32> = consts64.iter().map(|&v| v as f32).collect();
    consts[ck::CC_EPS as usize] = f32::EPSILON;
    consts[ck::CC_RTL as usize] = f32::EPSILON.sqrt();
    consts[ck::CC_BIG as usize] = f32::MAX;
    consts[ck::CC_FAIL as usize] = 1e16;
    let (eps, big) = (consts[ck::CC_EPS as usize], consts[ck::CC_BIG as usize]);

    let n = 7usize;
    let mut r = Rng::new(91_001);
    let d0: Vec<f32> = (0..n).map(|_| (r.uniform() as f32).abs() + 0.5).collect();
    let e0: Vec<f32> = (0..n).map(|i| if i + 1 < n { r.uniform() as f32 } else { 0.0 }).collect();
    let (lw, liw) = lapack_rs::cint_stemr_work_len(n);

    let run = |inline: bool| -> (Vec<f32>, Vec<f32>, u32) {
        let hd = upload(&c, &d0);
        let ho = upload(&c, &e0);
        let he = upload(&c, &vec![0.0f32; n]);
        let hv = upload(&c, &vec![0.0f32; n * n]);
        let hw = upload(&c, &vec![0.0f32; lw]);
        let hi = upload(&c, &vec![0u32; liw.max(1)]);
        let hc = upload(&c, &consts);
        let hs = upload(&c, &vec![0.0f32; ck::SC_LEN as usize]);
        let hic = upload(&c, &vec![0u32; ck::ICNT_LEN as usize]);
        let hin = upload(&c, &vec![0u32; 1]);
        if inline {
            // SAFETY: every length below is the allocation's own length and
            // matches `cint_stemr_work_len(n)`; the single unit indexes
            // `diag`/`offd`/`eig` by `< n` and `vec` by `< n*n`.
            unsafe {
                foreign_caller_kernel::launch_unchecked::<f32, HipRuntime>(
                    &c,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts(hd.clone(), n),
                    ArrayArg::from_raw_parts(ho.clone(), n),
                    ArrayArg::from_raw_parts(he.clone(), n),
                    ArrayArg::from_raw_parts(hv.clone(), n * n),
                    ArrayArg::from_raw_parts(hw.clone(), lw),
                    ArrayArg::from_raw_parts(hi.clone(), liw.max(1)),
                    ArrayArg::from_raw_parts(hc.clone(), consts.len()),
                    ArrayArg::from_raw_parts(hs.clone(), ck::SC_LEN as usize),
                    ArrayArg::from_raw_parts(hic.clone(), ck::ICNT_LEN as usize),
                    ArrayArg::from_raw_parts(hin.clone(), 1),
                    eps,
                    big,
                    n as u32,
                );
            }
        } else {
            ops::cint_stemr_one::<HipRuntime, f32>(&c, &hd, &ho, &he, &hv, &hw, &hi, &hc, &hs, &hic, &hin, n, eps, big);
        }
        let mut eig = vec![0.0f32; n];
        let mut vecv = vec![0.0f32; n * n];
        let mut info = vec![0u32; 1];
        read_range(&c, &he, n, 0, n, &mut eig);
        read_range(&c, &hv, n * n, 0, n * n, &mut vecv);
        read_range(&c, &hin, 1, 0, 1, &mut info);
        (eig, vecv, info[0])
    };

    let inline = run(true);
    let launcher = run(false);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(inline.2, launcher.2, "inline callee INFO differs from the launcher");
    assert_eq!(bits(&inline.0), bits(&launcher.0), "inline callee EIG differs from the launcher");
    assert_eq!(bits(&inline.1), bits(&launcher.1), "inline callee VEC differs from the launcher");
    assert!(inline.0.iter().any(|&v| v != 0.0), "the inline callee wrote nothing");
    eprintln!("K9 surface 1: a foreign crate's kernel calls cint_stemr_dev inline, bits identical");
}

/// **K9 (Path C, libcint 6.1.3) on ROCm.** This GPU has no f64, so the gate
/// here is not bit-exactness against the host (which is f64-only) but the two
/// things f32 can still prove:
///
/// 1. **hipRTC lowers the whole solver.** That is the cross-check behind
///    `docs/cubecl-cpu-lowering-limit.md`: several K9 routines do not lower on
///    cubecl-cpu 0.10 and silently no-op there, so a backend that does lower
///    them is what shows the kernel itself is well-formed.
/// 2. **Bits do not depend on the launch shape.** The one-problem parity
///    launcher (surface 3) and the batched kernel (surface 2) must produce
///    identical bits for the same problem, as AGENTS.md's device rules
///    require. The f64 bit-exactness gate is `tests/device_cuda.rs`.
#[test]
fn cint_stemr_launch_shapes_agree_f32() {
    let c = client();
    println!("contraction f32: {:?}", contraction_probe::<HipRuntime, f32>(&c, f32::EPSILON, 1.0, 2.0));
    // libcint's table, retuned for f32 where the constant is a *tolerance*
    // rather than an algorithmic coefficient. `rtl = 2.1e-8` and
    // `DBL_EPSILON = 2.2e-16` are both far below `f32::EPSILON = 1.2e-7`, so
    // with the f64 values `_dlarrk` can never satisfy its convergence test in
    // f32 and the solver returns `INFO = -1` having written nothing. Every
    // coefficient that is part of the *algorithm* (`CNST1`, `CNST2`, `CNST3`,
    // the thirds and hundredths) keeps libcint's value. This is exactly why
    // they are arguments: one kernel body, two precisions.
    use lapack_rs::device::kernels::cint_stemr as ck;
    let consts64 = ops::cint_stemr_consts();
    let mut consts: Vec<f32> = consts64.iter().map(|&v| v as f32).collect();
    consts[ck::CC_EPS as usize] = f32::EPSILON;
    consts[ck::CC_RTL as usize] = f32::EPSILON.sqrt();
    consts[ck::CC_BIG as usize] = f32::MAX;
    consts[ck::CC_FAIL as usize] = 1e16;
    let big = consts[ck::CC_BIG as usize];

    let ns: Vec<usize> = vec![3, 5, 7, 13, 4, 9];
    let n_max = *ns.iter().max().expect("ns") as u32;
    let mut nptr = vec![0u32];
    let mut vptr = vec![0u32];
    let (mut diag, mut offd) = (Vec::new(), Vec::new());
    for (b, &n) in ns.iter().enumerate() {
        let mut r = Rng::new(70_000 + b as u64);
        for i in 0..n {
            diag.push((r.uniform() as f32).abs() + 0.5);
            offd.push(if i + 1 < n { r.uniform() as f32 } else { 0.0 });
        }
        nptr.push(diag.len() as u32);
        vptr.push(vptr[b] + (n * n) as u32);
    }
    let nprob = ns.len();
    let dlen = diag.len();
    let vlen = *vptr.last().expect("vptr") as usize;
    let wper = 9 * n_max as usize + 1;
    let iwper = 5 * n_max as usize;

    // --- batched (surface 2) ----------------------------------------------
    let gdiag = upload(&c, &diag);
    let goffd = upload(&c, &offd);
    let gnptr = upload(&c, &nptr);
    let gvptr = upload(&c, &vptr);
    let geig = upload(&c, &vec![0.0f32; dlen]);
    let gvec = upload(&c, &vec![0.0f32; vlen]);
    let gwork = upload(&c, &vec![0.0f32; nprob * wper]);
    let giwork = upload(&c, &vec![0u32; nprob * iwper]);
    let gcc = upload(&c, &consts);
    let gsc = upload(&c, &vec![0.0f32; nprob * 16]);
    let gicnt = upload(&c, &vec![0u32; nprob * 4]);
    let ginfo = upload(&c, &vec![0u32; nprob]);
    ops::cint_stemr_batch::<HipRuntime, f32>(
        &c,
        &ops::CintStemrBatch {
            diag: &gdiag,
            offd: &goffd,
            nptr: &gnptr,
            vptr: &gvptr,
            eig: &geig,
            vec: &gvec,
            work: &gwork,
            iwork: &giwork,
            cc: &gcc,
            sc: &gsc,
            icnt: &gicnt,
            info: &ginfo,
        },
        nprob,
        dlen,
        vlen,
        n_max,
        consts[ck::CC_EPS as usize],
        big,
    );
    let mut beig = vec![0.0f32; dlen];
    let mut bvec = vec![0.0f32; vlen];
    let mut binfo = vec![0u32; nprob];
    read_range(&c, &geig, dlen, 0, dlen, &mut beig);
    read_range(&c, &gvec, vlen, 0, vlen, &mut bvec);
    read_range(&c, &ginfo, nprob, 0, nprob, &mut binfo);

    // --- one problem per launch (surface 3) -------------------------------
    for (b, &n) in ns.iter().enumerate() {
        let lo = nptr[b] as usize;
        let vlo = vptr[b] as usize;
        let (lw, liw) = lapack_rs::cint_stemr_work_len(n);
        let sd = upload(&c, &diag[lo..lo + n]);
        let so = upload(&c, &offd[lo..lo + n]);
        let se = upload(&c, &vec![0.0f32; n]);
        let sv = upload(&c, &vec![0.0f32; n * n]);
        let sw = upload(&c, &vec![0.0f32; lw]);
        let si = upload(&c, &vec![0u32; liw.max(1)]);
        let scc = upload(&c, &consts);
        let ssc = upload(&c, &vec![0.0f32; 16]);
        let sic = upload(&c, &vec![0u32; 4]);
        let sinf = upload(&c, &vec![0u32; 1]);
        ops::cint_stemr_one::<HipRuntime, f32>(
            &c,
            &sd,
            &so,
            &se,
            &sv,
            &sw,
            &si,
            &scc,
            &ssc,
            &sic,
            &sinf,
            n,
            consts[ck::CC_EPS as usize],
            big,
        );
        let mut oeig = vec![0.0f32; n];
        let mut ovec = vec![0.0f32; n * n];
        let mut oinfo = vec![0u32; 1];
        read_range(&c, &se, n, 0, n, &mut oeig);
        read_range(&c, &sv, n * n, 0, n * n, &mut ovec);
        read_range(&c, &sinf, 1, 0, 1, &mut oinfo);

        let tag = format!("b={b} n={n}");
        assert_eq!(oinfo[0], binfo[b], "K9 INFO differs by launch shape {tag}");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&oeig), bits(&beig[lo..lo + n]), "K9 EIG differs by launch shape {tag}");
        assert_eq!(bits(&ovec), bits(&bvec[vlo..vlo + n * n]), "K9 VEC differs by launch shape {tag}");
        assert_eq!(oinfo[0], 0, "K9 INFO {tag}: {:?}", oeig);
        assert!(oeig.iter().all(|v| v.is_finite()), "K9 produced a non-finite eigenvalue {tag}");
        // The kernel actually ran: a silent no-op would leave the uploaded
        // zeros, which is the cubecl-cpu failure mode this gate exists to
        // rule out for hipRTC.
        assert!(oeig.iter().any(|&v| v != 0.0), "K9 wrote nothing (silent no-op?) {tag}");
        // Ascending eigenvalues and normalised eigenvectors: the algorithm ran
        // to completion, not just to a `return`.
        for i in 1..n {
            assert!(oeig[i] >= oeig[i - 1], "K9 eigenvalues not ascending {tag}: {:?}", oeig);
        }
        for i in 0..n {
            let nrm: f32 = (0..n).map(|j| ovec[i * n + j] * ovec[i * n + j]).sum();
            assert!((nrm - 1.0).abs() < 1e-3, "K9 eigenvector {i} not normalised {tag}: {nrm}");
        }
    }
    // `k9_supported` is an **f64** probe, and this GPU has none, so on this
    // machine it must report `false` *for that reason* — not because hipRTC
    // failed to lower anything, which the f32 run above just disproved. On an
    // f64 AMD GPU it must report `true`; that branch is asserted, and is the
    // one `tests/device_cuda.rs` exercises for real.
    let f64_ok = lapack_rs::device::probe::supports_f64(&c);
    let probe = lapack_rs::device::probe::k9_supported(&c);
    println!("k9_supported={probe} (f64 available: {f64_ok})");
    if f64_ok {
        assert!(probe, "hipRTC lowered K9 in f32 but the f64 probe disagrees");
    } else {
        assert!(!probe, "k9_supported must be false without f64");
    }
    eprintln!("K9 on ROCm: {nprob} problems, identical bits from both launch shapes");
}
