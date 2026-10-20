//! CUDA gates (feature `cuda`, an NVIDIA GPU with f64):
//!
//! * capabilities and the contraction probe (f32 and f64) are reported; the
//!   gate refuses to run without `NVRTC_COMPILE_OPTIONS_APPEND=--fmad=false`
//!   (probe: no contraction), because NVRTC fuses `a*b + c` by default;
//! * `dsyevd_device`/`zheevd_device` (`Exactness::Reference`) against
//!   Reference-LAPACK 3.12.1 bit for bit, both `UPLO`, both `JOBZ`, across
//!   the blocking and D&C thresholds and at sizes where the GPU kernels run
//!   their row-tiled variants;
//! * the same results under every kernel variant, under bounds-checked
//!   launches and through the device-resident entry points;
//! * K4 (`DLAED3` on the device: secular roots, `W`, `U`) for every merge
//!   with `K > 2` against the host `DSTEDC` on the deflation fixtures;
//! * **K9 (Path C, libcint 6.1.3): the whole `cint_stemr_dev` solver against
//!   `lapack_rs::cint_stemr` on the host, bit for bit, through both the
//!   one-problem parity launcher and the batched kernel.** This is the gate
//!   the CubeCL CPU runtime cannot carry — several K9 routines hit a
//!   cubecl-cpu 0.10 control-flow lowering bug and silently no-op
//!   (`docs/cubecl-cpu-lowering-limit.md`).
//!
//! Environment: `CUDA_PATH` (or `/usr/local/cuda`) for NVRTC headers,
//! `LAPACK_RS_REF_DIR` at build time and `LAPACK_RS_REQUIRE_REF=1`.

#![cfg(feature = "cuda")]

mod common;
use common::*;
use cubecl::prelude::*;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use lapack_rs::device::geometry;
use lapack_rs::device::probe::{contraction_probe, supports_f64};
use lapack_rs::device::workspace::{read_range, upload};
use lapack_rs::device::{
    DeviceBudget, DeviceWorkspace, MergeU, dsyevd_device, dsyevd_device_resident, ops, zheevd_device,
};
use lapack_rs::{Job, Uplo, heevd_opt_work_len, syevd_opt_work_len};

fn client() -> ComputeClient<CudaRuntime> {
    CudaRuntime::client(&CudaDevice::default())
}

/// Restores the process-global test switches even if an assertion panics.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        ops::set_variant(ops::Variant::Auto);
        ops::set_checked_launch(false);
        geometry::set_cubes_per_sm(u32::MAX);
        lapack_rs::device::stedc::set_secular_min_k(0);
    }
}

/// f64 and no contraction, or panic: a CUDA gate run is never vacuous.
fn require_exact(c: &ComputeClient<CudaRuntime>) {
    assert!(supports_f64(c), "this CUDA device reports no f64");
    let p = contraction_probe::<CudaRuntime, f64>(c, f64::EPSILON, 1.0, 2.0);
    assert!(
        p.no_contraction(),
        "NVRTC contracts ({p:?}): set NVRTC_COMPILE_OPTIONS_APPEND=--fmad=false (and clear any PTX cache)"
    );
}

fn ref_dsyevd(job: Job, uplo: Uplo, n: usize, a0: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (lw, liw) = syevd_opt_work_len(n, job);
    let (mut a, mut w) = (a0.to_vec(), vec![0.0; n]);
    let (mut work, mut iwork) = (vec![0.0; lw], vec![0i32; liw]);
    let jz = if job == Job::Vectors { 'V' } else { 'N' };
    let ul = if uplo == Uplo::Upper { 'U' } else { 'L' };
    assert_eq!(lapack_ref::dsyevd(jz, ul, n, &mut a, &mut w, &mut work, &mut iwork), 0);
    (a, w)
}

fn check_dsyevd(
    c: &ComputeClient<CudaRuntime>,
    ws: &mut DeviceWorkspace<CudaRuntime>,
    tag: &str,
    n: usize,
    a0: &[f64],
) {
    for uplo in [Uplo::Lower, Uplo::Upper] {
        for job in [Job::ValuesOnly, Job::Vectors] {
            let (mut a, mut w) = (a0.to_vec(), vec![0.0; n]);
            dsyevd_device(c, job, uplo, n, &mut a, &mut w, ws).expect("dsyevd_device");
            let (ar, wr) = ref_dsyevd(job, uplo, n, a0);
            let t = format!("{tag} n={n} {uplo:?} {job:?}");
            assert_bits(&format!("w {t}"), &w, &wr);
            if job == Job::Vectors {
                assert_bits(&format!("z {t}"), &a, &ar);
            }
        }
    }
}

#[test]
fn cuda_reports_capabilities() {
    let c = client();
    let hw = &c.properties().hardware;
    println!(
        "cuda: plane_size_max={} max_units_per_cube={} max_cube_count={:?} f64={} f64 widths={:?} f32 widths={:?}",
        hw.plane_size_max,
        hw.max_units_per_cube,
        hw.max_cube_count,
        supports_f64(&c),
        c.io_optimized_vector_sizes(8).collect::<Vec<_>>(),
        c.io_optimized_vector_sizes(4).collect::<Vec<_>>()
    );
    println!("contraction f32: {:?}", contraction_probe::<CudaRuntime, f32>(&c, f32::EPSILON, 1.0, 2.0));
    println!("contraction f64: {:?}", contraction_probe::<CudaRuntime, f64>(&c, f64::EPSILON, 1.0, 2.0));
    require_exact(&c);
}

#[test]
fn dsyevd_cuda_matches_reference_lapack() {
    if !lapack_ref::require("dsyevd_cuda_matches_reference_lapack") {
        return;
    }
    let c = client();
    require_exact(&c);
    let mut ws = DeviceWorkspace::new(&c, 64, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
    for &n in &[2usize, 17, 33, 34, 64, 65, 100, 257, 513, 1024] {
        check_dsyevd(&c, &mut ws, "random", n, &random_sym(n, 7000 + n as u64));
    }
    for (name, a) in dense_fixtures(90) {
        check_dsyevd(&c, &mut ws, name, 90, &a);
    }
    let mut tiny = random_sym(120, 77);
    tiny.iter_mut().for_each(|v| *v *= 1e-160);
    check_dsyevd(&c, &mut ws, "tiny (scaled)", 120, &tiny);
}

#[test]
fn zheevd_cuda_matches_reference_lapack() {
    if !lapack_ref::require("zheevd_cuda_matches_reference_lapack") {
        return;
    }
    let c = client();
    require_exact(&c);
    let mut ws = DeviceWorkspace::new(&c, 64, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto);
    for &n in &[2usize, 20, 33, 34, 64, 131, 300, 600] {
        let (re0, mut im0) = random_herm(n, 8100 + n as u64);
        if n % 2 == 0 {
            for i in 0..n {
                im0[i + i * n] = -1.5 + i as f64; // ignored by ZHEEVD
            }
        }
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for job in [Job::ValuesOnly, Job::Vectors] {
                let (mut ar, mut ai, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
                zheevd_device(&c, job, uplo, n, &mut ar, &mut ai, &mut w, &mut ws).expect("zheevd_device");
                let (lw, lrw, liw) = heevd_opt_work_len(n, job);
                let mut a = to_c64(&re0, &im0);
                let mut wr = vec![0.0; n];
                let mut work = vec![lapack_ref::C64 { re: 0.0, im: 0.0 }; lw];
                let (mut rwork, mut iwork) = (vec![0.0; lrw], vec![0i32; liw]);
                let jz = if job == Job::Vectors { 'V' } else { 'N' };
                let ul = if uplo == Uplo::Upper { 'U' } else { 'L' };
                let info = lapack_ref::zheevd(jz, ul, n, &mut a, &mut wr, &mut work, &mut rwork, &mut iwork);
                assert_eq!(info, 0);
                let t = format!("n={n} {uplo:?} {job:?}");
                assert_bits(&format!("w {t}"), &w, &wr);
                if job == Job::Vectors {
                    let (rr, ri) = from_c64(&a);
                    assert_bits(&format!("z re {t}"), &ar, &rr);
                    assert_bits(&format!("z im {t}"), &ai, &ri);
                }
            }
        }
    }
}

/// One test: the variant and checked-launch switches are process-global.
#[test]
fn cuda_variants_checked_launch_and_resident() {
    if !lapack_ref::require("cuda_variants_checked_launch_and_resident") {
        return;
    }
    let c = client();
    require_exact(&c);
    let _restore = Restore;
    let mut ws = DeviceWorkspace::new(&c, 64, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Staged);
    for &(n, uplo) in &[(131usize, Uplo::Upper), (700, Uplo::Lower)] {
        let a0 = random_sym(n, 4242 + n as u64);
        let (ar, wr) = ref_dsyevd(Job::Vectors, uplo, n, &a0);
        let run = |tag: &str, ws: &mut DeviceWorkspace<CudaRuntime>| {
            let (mut a, mut w) = (a0.clone(), vec![0.0; n]);
            dsyevd_device(&c, Job::Vectors, uplo, n, &mut a, &mut w, ws).expect("device");
            assert_bits(&format!("{tag} w n={n}"), &w, &wr);
            assert_bits(&format!("{tag} z n={n}"), &a, &ar);
        };
        for (tag, v) in [("simple", ops::Variant::Simple), ("row-tiled", ops::Variant::RowTiled)] {
            ops::set_variant(v);
            run(tag, &mut ws);
        }
        ops::set_variant(ops::Variant::Auto);
        ops::set_checked_launch(true);
        run("checked", &mut ws);
        ops::set_checked_launch(false);
        // Launch geometry: every kernel's units are element-disjoint, so the
        // cubes-per-SM target is free. `0` is cubecl's element-wise default
        // cube, the rest span the measured sweep.
        for per_sm in [0u32, 1, 4, 16] {
            geometry::set_cubes_per_sm(per_sm);
            run(&format!("cubes_per_sm={per_sm}"), &mut ws);
        }
        geometry::set_cubes_per_sm(u32::MAX);

        let h = upload(&c, &a0);
        let mut w = vec![0.0; n];
        let z = dsyevd_device_resident(&c, Job::Vectors, uplo, n, &h, &mut w, &mut ws).expect("resident");
        let z = z.expect("vectors");
        let zc = z.compact(&c);
        let mut zr = vec![0.0; n * n];
        read_range(&c, &zc, n * n, 0, n * n, &mut zr);
        assert_bits(&format!("resident w n={n}"), &w, &wr);
        assert_bits(&format!("resident z n={n}"), &zr, &ar);
    }
}

/// K4 for every `K > 2` merge (the default threshold only covers large
/// merges) against the host transcription, bit for bit.
#[test]
fn cuda_secular_equation_matches_host() {
    let c = client();
    require_exact(&c);
    let _restore = Restore;
    lapack_rs::device::stedc::set_secular_min_k(3);
    let mut cases = Vec::new();
    for &n in &[27usize, 45, 64, 100, 200] {
        cases.extend(tridiag_fixtures(n).into_iter().map(|(name, d, e)| (name.to_string(), n, d, e)));
    }
    for &n in &[60usize, 151, 300, 1000] {
        let (d, e) = random_tridiag(n, 9100 + n as u64);
        cases.push(("random".to_string(), n, d, e));
    }
    for (name, n, d0, e0) in cases {
        let (lw, liw) = lapack_rs::stedc_work_len(n);
        let (mut dh, mut eh) = (d0.clone(), e0.clone());
        eh.push(0.0);
        let mut zh = vec![0.0; n * n];
        let (mut w, mut iw) = (vec![0.0; lw.max(1)], vec![0usize; liw.max(1)]);
        lapack_rs::blocks::dstedc(n, &mut dh, &mut eh, &mut zh, &mut w, &mut iw).expect("host");
        let (mut d, mut e) = (d0.clone(), e0.clone());
        e.push(0.0);
        let mut z = vec![0.0; n * n];
        lapack_rs::device::dstedc_device(&c, n, &mut d, &mut e, &mut z, 0).expect("device");
        let tag = format!("K4 {name} n={n}");
        assert_bits(&format!("d {tag}"), &d, &dh);
        assert_bits(&format!("Z {tag}"), &z, &zh);
    }
    lapack_rs::device::stedc::set_secular_min_k(0);
}

// ---------------------------------------------------------------------------
// K8 (Reference-LAPACK MRRR leaves) and K9 (Path C, libcint 6.1.3).
//
// These are the gates cubecl-cpu 0.10 cannot carry: `cint_dlarrb_c`,
// `cint_dlasq5_c`, `cint_dlasq2_c` and the two composite routines hit a
// control-flow lowering bug on the CPU runtime and silently no-op
// (`docs/cubecl-cpu-lowering-limit.md`). NVRTC lowers all of them, so this is
// where the whole Path C solver is pinned against the host, in f64, bit for
// bit — task C2 of `docs/cintx-lapack-blas-adoption-plan.md`.
// ---------------------------------------------------------------------------

/// The synthetic Path C sweep of `tests/cint_stemr_bitexact.rs`, so the device
/// is pinned on exactly the fixtures the host is pinned on.
fn cint_fixture(n: usize, kind: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut r = Rng::new(40_000 + seed * 7919 + n as u64 * 31 + kind as u64);
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    for i in 0..n {
        d[i] = match kind {
            0 => r.uniform(),
            1 => i as f64 + 1.0,
            2 => 1.0,
            3 => 1.0 + 1e-13 * (i / 2) as f64,
            4 => 10f64.powi(-(i as i32)),
            5 => r.uniform() * 1e6,
            6 => r.uniform() * 1e-6,
            _ => r.uniform().abs(),
        };
        if i + 1 < n {
            e[i] = match kind {
                0 => r.uniform(),
                1 => 1.0,
                2 => 1e-16,
                3 => 1e-14,
                4 => 10f64.powi(-(i as i32) - 1),
                5 => r.uniform() * 1e-3,
                6 => r.uniform() * 1e-9,
                _ => r.uniform(),
            };
        }
    }
    (d, e)
}

/// **K9, Path C.** `cint_stemr_dev` on the GPU against `lapack_rs::cint_stemr`
/// on the host, bit for bit, through both the one-problem parity launcher
/// (surface 3) and the batched kernel (surface 2). The host side is itself
/// `max_ulp = 0` against libcint 6.1.3 (`tests/cint_stemr_bitexact.rs`), so
/// this closes the chain C == host == device.
#[test]
fn cint_stemr_device_matches_host() {
    let client = client();
    require_exact(&client);
    // The capability probe a production caller must consult: NVRTC lowers the
    // composite, so it has to report `true` here. Its mirror,
    // `device_cint_stemr_cpu::k9_supported_reports_false_on_cpu_runtime`,
    // asserts `false` on the CubeCL CPU runtime.
    assert!(lapack_rs::device::probe::k9_supported(&client), "NVRTC did not lower K9 (or it disagrees with the host)");
    let consts = ops::cint_stemr_consts();
    let big = consts[lapack_rs::device::kernels::cint_stemr::CC_BIG as usize];

    // --- surface 3: one problem per launch -------------------------------
    let mut checked = 0usize;
    for n in 1..=lapack_rs::MXRYSROOTS {
        for kind in 0..8usize {
            for seed in 0..3u64 {
                let (d0, e0) = cint_fixture(n, kind, seed);
                let (lw, liw) = lapack_rs::cint_stemr_work_len(n);
                let (mut hd, mut he) = (d0.clone(), e0.clone());
                let mut heig = vec![0.0f64; n];
                let mut hvec = vec![0.0f64; n * n];
                let (mut hw, mut hiw) = (vec![0.0f64; lw], vec![0i32; liw.max(1)]);
                let hinfo = lapack_rs::cint_stemr(n, &mut hd, &mut he, &mut heig, &mut hvec, &mut hw, &mut hiw);
                let tag = format!("n={n} kind={kind} seed={seed}");

                let gdiag = upload(&client, &d0);
                let goffd = upload(&client, &e0);
                let geig = upload(&client, &vec![0.0f64; n]);
                let gvec = upload(&client, &vec![0.0f64; n * n]);
                let gwork = upload(&client, &vec![0.0f64; lw]);
                let giwork = upload(&client, &vec![0u32; liw.max(1)]);
                let gcc = upload(&client, &consts);
                let gsc = upload(&client, &vec![0.0f64; 16]);
                let gicnt = upload(&client, &vec![0u32; 4]);
                let ginfo = upload(&client, &vec![0u32; 1]);
                ops::cint_stemr_one::<CudaRuntime, f64>(
                    &client,
                    &gdiag,
                    &goffd,
                    &geig,
                    &gvec,
                    &gwork,
                    &giwork,
                    &gcc,
                    &gsc,
                    &gicnt,
                    &ginfo,
                    n,
                    f64::EPSILON,
                    big,
                );
                let mut geigv = vec![0.0f64; n];
                let mut gvecv = vec![0.0f64; n * n];
                let mut gdv = vec![0.0f64; n];
                let mut gev = vec![0.0f64; n];
                let mut gi = vec![0u32; 1];
                read_range(&client, &geig, n, 0, n, &mut geigv);
                read_range(&client, &gvec, n * n, 0, n * n, &mut gvecv);
                read_range(&client, &gdiag, n, 0, n, &mut gdv);
                read_range(&client, &goffd, n, 0, n, &mut gev);
                read_range(&client, &ginfo, 1, 0, 1, &mut gi);

                assert_eq!(gi[0] as i32, hinfo.abs(), "cint_stemr_dev INFO {tag}");
                if hinfo != 0 {
                    continue;
                }
                assert_bits(&format!("cint_stemr_dev EIG {tag}"), &geigv, &heig);
                assert_bits(&format!("cint_stemr_dev VEC {tag}"), &gvecv, &hvec);
                assert_bits(&format!("cint_stemr_dev DIAG {tag}"), &gdv, &hd);
                assert_bits(&format!("cint_stemr_dev OFFD {tag}"), &gev, &he);
                checked += 1;
            }
        }
    }
    assert!(checked > 100, "only {checked} problems checked");

    // --- surface 2: one batched launch over mixed sizes -------------------
    let ns: Vec<usize> = vec![3, 5, 13, 4, 7, 9, 13, 6, 11, 8, 3, 12];
    let n_max = *ns.iter().max().expect("ns") as u32;
    let mut nptr = vec![0u32];
    let mut vptr = vec![0u32];
    let (mut diag, mut offd) = (Vec::new(), Vec::new());
    let mut hosts = Vec::new();
    for (b, &n) in ns.iter().enumerate() {
        let (d0, e0) = cint_fixture(n, b % 8, b as u64);
        let (lw, liw) = lapack_rs::cint_stemr_work_len(n);
        let (mut hd, mut he) = (d0.clone(), e0.clone());
        let mut heig = vec![0.0f64; n];
        let mut hvec = vec![0.0f64; n * n];
        let (mut hw, mut hiw) = (vec![0.0f64; lw], vec![0i32; liw.max(1)]);
        let info = lapack_rs::cint_stemr(n, &mut hd, &mut he, &mut heig, &mut hvec, &mut hw, &mut hiw);
        hosts.push((info, heig, hvec, hd, he));
        diag.extend_from_slice(&d0);
        offd.extend_from_slice(&e0);
        nptr.push(diag.len() as u32);
        vptr.push(vptr[b] + (n * n) as u32);
    }
    let nprob = ns.len();
    let dlen = diag.len();
    let vlen = *vptr.last().expect("vptr") as usize;
    let wper = 9 * n_max as usize + 1;
    let iwper = 5 * n_max as usize;

    let gdiag = upload(&client, &diag);
    let goffd = upload(&client, &offd);
    let gnptr = upload(&client, &nptr);
    let gvptr = upload(&client, &vptr);
    let geig = upload(&client, &vec![0.0f64; dlen]);
    let gvec = upload(&client, &vec![0.0f64; vlen]);
    let gwork = upload(&client, &vec![0.0f64; nprob * wper]);
    let giwork = upload(&client, &vec![0u32; nprob * iwper]);
    let gcc = upload(&client, &consts);
    let gsc = upload(&client, &vec![0.0f64; nprob * 16]);
    let gicnt = upload(&client, &vec![0u32; nprob * 4]);
    let ginfo = upload(&client, &vec![0u32; nprob]);
    ops::cint_stemr_batch::<CudaRuntime, f64>(
        &client,
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
        f64::EPSILON,
        big,
    );
    let mut eig = vec![0.0f64; dlen];
    let mut v = vec![0.0f64; vlen];
    let mut info = vec![0u32; nprob];
    read_range(&client, &geig, dlen, 0, dlen, &mut eig);
    read_range(&client, &gvec, vlen, 0, vlen, &mut v);
    read_range(&client, &ginfo, nprob, 0, nprob, &mut info);
    for (b, &n) in ns.iter().enumerate() {
        let lo = nptr[b] as usize;
        let vlo = vptr[b] as usize;
        let (hinfo, heig, hvec, _, _) = &hosts[b];
        let tag = format!("batch b={b} n={n}");
        assert_eq!(info[b] as i32, hinfo.abs(), "cint_stemr_batch INFO {tag}");
        if *hinfo != 0 {
            continue;
        }
        assert_bits(&format!("cint_stemr_batch EIG {tag}"), &eig[lo..lo + n], heig);
        assert_bits(&format!("cint_stemr_batch VEC {tag}"), &v[vlo..vlo + n * n], hvec);
    }
    eprintln!("K9 on the GPU: {checked} single + {nprob} batched problems, bit-identical to the host");
}
