//! Speed profile of the device drivers against the host path, Reference-LAPACK
//! and faer, with a bit-for-bit check of every device result.
//!
//! ```text
//! cargo run --release --features cuda --example device_profile -- 1024,2048,4096
//! cargo run --release --features device,cpu --example device_profile -- --complex 512
//! ```
//!
//! Options: `--complex` (`zheevd`), `--values` (no eigenvectors),
//! `--passes P` (min over P timed passes, default 3), `--ref-max N` (run the
//! single-threaded Reference-LAPACK oracle up to N, default 2048; needs
//! `LAPACK_RS_REF_DIR` at build time), `--no-host`, `--no-faer`.
//!
//! Per size it prints the device time (warm: the first call at each size pays
//! JIT compilation and is not timed), one phase-profiled pass (every phase
//! closed by a device sync; `read_wait` is the time spent blocked in reads),
//! the transfer counters, and the other solvers' times. For a per-kernel
//! device-time table run with CubeCL's profiler, e.g.
//! `CUBECL_DEBUG_LOG=target/prof/kernels.log CUBECL_DEBUG_OPTION=profile`.
//!
//! Backend: `cuda` when enabled, otherwise the CubeCL CPU runtime.

use std::time::Instant;

use cubecl::prelude::*;
use lapack_rs::device::workspace::set_phase_profiling;
use lapack_rs::device::{DeviceBudget, DeviceWorkspace, MergeU, dsyevd_device, zheevd_device};
use lapack_rs::{Job, Uplo};

#[cfg(not(any(feature = "cuda", feature = "cpu")))]
compile_error!("device_profile needs feature `cuda` or `cpu`");

#[cfg(feature = "cuda")]
type Rt = cubecl_cuda::CudaRuntime;
#[cfg(all(not(feature = "cuda"), feature = "cpu"))]
type Rt = cubecl::cpu::CpuRuntime;

#[cfg(feature = "cuda")]
fn client() -> ComputeClient<Rt> {
    Rt::client(&cubecl_cuda::CudaDevice::default())
}
#[cfg(all(not(feature = "cuda"), feature = "cpu"))]
fn client() -> ComputeClient<Rt> {
    Rt::client(&cubecl::cpu::CpuDevice)
}

struct Opts {
    ns: Vec<usize>,
    complex: bool,
    job: Job,
    passes: usize,
    ref_max: usize,
    host: bool,
    faer: bool,
}

fn opts() -> Opts {
    let mut o = Opts {
        ns: vec![256, 512, 1024],
        complex: false,
        job: Job::Vectors,
        passes: 3,
        ref_max: 2048,
        host: true,
        faer: true,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--complex" => o.complex = true,
            "--values" => o.job = Job::ValuesOnly,
            "--passes" => o.passes = args.next().and_then(|s| s.parse().ok()).expect("--passes P"),
            "--ref-max" => o.ref_max = args.next().and_then(|s| s.parse().ok()).expect("--ref-max N"),
            "--no-host" => o.host = false,
            "--no-faer" => o.faer = false,
            list => o.ns = list.split(',').map(|t| t.parse().expect("n list")).collect(),
        }
    }
    o
}

/// SplitMix64 in [-1, 1): the same generator as `tests/common`.
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 * (1.0 / (1u64 << 52) as f64) - 1.0
    }
}

fn random_sym(n: usize, seed: u64) -> Vec<f64> {
    let mut r = Rng(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            let v = r.uniform();
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
    }
    a
}

fn random_herm(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let re = random_sym(n, seed);
    let mut r = Rng(seed.wrapping_mul(31) ^ 0x5555);
    let mut im = vec![0.0; n * n];
    for j in 0..n {
        for i in j + 1..n {
            let v = r.uniform();
            im[i + j * n] = v;
            im[j + i * n] = -v;
        }
    }
    (re, im)
}

fn secs(f: impl FnOnce()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64()
}

fn bits_equal(a: &[f64], b: &[f64]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn main() {
    let o = opts();
    let client = client();
    let props = client.properties();
    let probe = lapack_rs::device::probe::contraction_probe::<Rt, f64>(&client, f64::EPSILON, 1.0, 2.0);
    println!(
        "backend {} | plane {} | f64 width {} | contraction probe {:?} | rayon threads {} | ref-lapack {}",
        std::any::type_name::<Rt>(),
        props.hardware.plane_size_max,
        lapack_rs::device::geometry::vector_width(&client, 8),
        probe,
        rayon::current_num_threads(),
        if lapack_ref::AVAILABLE { "linked" } else { "not linked" },
    );
    let job = o.job;
    let wantz = job == Job::Vectors;
    for &n in &o.ns {
        let mut ws = DeviceWorkspace::new(&client, n, job, DeviceBudget::UNLIMITED, MergeU::Staged);
        let kind = if o.complex { "zheevd" } else { "dsyevd" };
        let (re0, im0) = if o.complex { random_herm(n, 43 + n as u64) } else { (random_sym(n, 42 + n as u64), vec![]) };

        // One device solve on fresh copies; returns (w, z_re, z_im).
        let device = |ws: &mut DeviceWorkspace<Rt>| {
            let (mut re, mut im, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
            if o.complex {
                zheevd_device(&client, job, Uplo::Lower, n, &mut re, &mut im, &mut w, ws).expect("zheevd_device");
            } else {
                dsyevd_device(&client, job, Uplo::Lower, n, &mut re, &mut w, ws).expect("dsyevd_device");
            }
            (w, re, im)
        };

        let t_jit = secs(|| {
            device(&mut ws);
        });
        let mut t_dev = f64::MAX;
        let mut out = None;
        for _ in 0..o.passes {
            let t = Instant::now();
            let r = device(&mut ws);
            t_dev = t_dev.min(t.elapsed().as_secs_f64());
            out = Some(r);
        }
        let (w_dev, zr_dev, zi_dev) = out.expect("passes >= 1");

        set_phase_profiling(true);
        ws.reset_counters();
        let _ = lapack_rs::device::workspace::take_dlaed3_times();
        let t_prof = secs(|| {
            device(&mut ws);
        });
        set_phase_profiling(false);
        let ph = ws.phase_times();
        let c = ws.counters();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;

        println!("\n{kind} n={n} {job:?}");
        println!("  device        {t_dev:>9.4} s   (first call incl. JIT {t_jit:.3} s)");
        println!(
            "  phases (profiled pass {:.4} s): stage {:.1} ms | trd {:.1} ms | stedc {:.1} ms | orm {:.1} ms | readback {:.1} ms | read_wait {:.1} ms",
            t_prof,
            ms(ph.stage),
            ms(ph.trd),
            ms(ph.stedc),
            ms(ph.orm),
            ms(ph.readback),
            ms(ph.read_wait)
        );
        println!(
            "  host spans (no sync): trd update {:.1} ms | stedc leaves {:.1} ms | stedc merges {:.1} ms | stedc gathers {:.1} ms | orm blocks {:.1} ms",
            ms(ph.spans[0]),
            ms(ph.spans[1]),
            ms(ph.spans[2]),
            ms(ph.spans[3]),
            ms(ph.spans[4])
        );
        println!(
            "  read wait per phase: trd {:.1} ms | stedc {:.1} ms | orm {:.1} ms | readback {:.1} ms",
            ms(ph.phase_wait[1]),
            ms(ph.phase_wait[2]),
            ms(ph.phase_wait[3]),
            ms(ph.phase_wait[4])
        );
        let d3 = lapack_rs::device::workspace::take_dlaed3_times();
        println!(
            "  host DLAED3: secular roots {:.1} ms | W products {:.1} ms | U columns {:.1} ms",
            ms(d3[0]),
            ms(d3[1]),
            ms(d3[2])
        );
        println!(
            "  traffic: {} launches, {} uploads ({:.1} MiB), {} reads ({:.1} MiB)",
            c.launches,
            c.uploads,
            c.upload_bytes as f64 / 1048576.0,
            c.reads,
            c.read_bytes as f64 / 1048576.0
        );

        if o.host {
            let (mut re, mut im, mut w) = (re0.clone(), im0.clone(), vec![0.0; n]);
            let mut t_host = f64::MAX;
            for _ in 0..o.passes.max(1) {
                re.copy_from_slice(&re0);
                im.copy_from_slice(&im0);
                t_host = t_host.min(secs(|| {
                    if o.complex {
                        lapack_rs::zheevd_alloc(job, Uplo::Lower, n, &mut re, &mut im, &mut w).expect("zheevd");
                    } else {
                        lapack_rs::dsyevd_alloc(job, Uplo::Lower, n, &mut re, &mut w).expect("dsyevd");
                    }
                }));
            }
            let mut diff = bits_equal(&w_dev, &w);
            if wantz {
                diff += bits_equal(&zr_dev, &re) + if o.complex { bits_equal(&zi_dev, &im) } else { 0 };
            }
            println!(
                "  host lapack_rs {t_host:>9.4} s   device speedup {:.2}x | device vs host: {}",
                t_host / t_dev,
                if diff == 0 { "bit-identical".to_string() } else { format!("{diff} values DIFFER") }
            );
        }

        if lapack_ref::AVAILABLE && n <= o.ref_max {
            let mut t_ref = f64::MAX;
            let mut diff = 0;
            for _ in 0..o.passes.clamp(1, if n >= 2048 { 1 } else { 3 }) {
                let mut w = vec![0.0; n];
                let jz = if wantz { 'V' } else { 'N' };
                if o.complex {
                    let (lw, lrw, liw) = lapack_rs::heevd_opt_work_len(n, job);
                    let mut a: Vec<lapack_ref::C64> =
                        re0.iter().zip(&im0).map(|(&re, &im)| lapack_ref::C64 { re, im }).collect();
                    let mut work = vec![lapack_ref::C64::default(); lw];
                    let (mut rwork, mut iwork) = (vec![0.0; lrw], vec![0i32; liw]);
                    t_ref = t_ref.min(secs(|| {
                        let info = lapack_ref::zheevd(jz, 'L', n, &mut a, &mut w, &mut work, &mut rwork, &mut iwork);
                        assert_eq!(info, 0);
                    }));
                    diff = bits_equal(&w_dev, &w);
                    if wantz {
                        let zr: Vec<f64> = a.iter().map(|c| c.re).collect();
                        let zi: Vec<f64> = a.iter().map(|c| c.im).collect();
                        diff += bits_equal(&zr_dev, &zr) + bits_equal(&zi_dev, &zi);
                    }
                } else {
                    let (lw, liw) = lapack_rs::syevd_opt_work_len(n, job);
                    let mut a = re0.clone();
                    let (mut work, mut iwork) = (vec![0.0; lw], vec![0i32; liw]);
                    t_ref = t_ref.min(secs(|| {
                        let info = lapack_ref::dsyevd(jz, 'L', n, &mut a, &mut w, &mut work, &mut iwork);
                        assert_eq!(info, 0);
                    }));
                    diff = bits_equal(&w_dev, &w) + if wantz { bits_equal(&zr_dev, &a) } else { 0 };
                }
            }
            println!(
                "  ref-lapack(1t) {t_ref:>9.4} s   device speedup {:.2}x | device vs Reference-LAPACK: {}",
                t_ref / t_dev,
                if diff == 0 { "BIT-EXACT".to_string() } else { format!("{diff} values DIFFER") }
            );
        }

        if o.faer {
            use faer::{Mat, Side};
            let t_faer = if o.complex {
                let fa = Mat::<faer::c64>::from_fn(n, n, |i, j| faer::c64::new(re0[i + j * n], im0[i + j * n]));
                let _ = fa.self_adjoint_eigen(Side::Lower);
                (0..o.passes.max(1))
                    .map(|_| {
                        secs(|| {
                            if wantz {
                                let _ = fa.self_adjoint_eigen(Side::Lower).expect("faer");
                            } else {
                                let _ = fa.self_adjoint_eigenvalues(Side::Lower).expect("faer");
                            }
                        })
                    })
                    .fold(f64::MAX, f64::min)
            } else {
                let fa = Mat::<f64>::from_fn(n, n, |i, j| re0[i + j * n]);
                let _ = fa.self_adjoint_eigen(Side::Lower);
                (0..o.passes.max(1))
                    .map(|_| {
                        secs(|| {
                            if wantz {
                                let _ = fa.self_adjoint_eigen(Side::Lower).expect("faer");
                            } else {
                                let _ = fa.self_adjoint_eigenvalues(Side::Lower).expect("faer");
                            }
                        })
                    })
                    .fold(f64::MAX, f64::min)
            };
            println!("  faer          {t_faer:>9.4} s   device speedup {:.2}x", t_faer / t_dev);
        }
    }
}
