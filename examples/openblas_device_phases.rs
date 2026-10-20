//! **OpenBLAS track.** Phase profile of `dsyevd_device_openblas` on the CubeCL
//! CPU runtime (diagnostic): `cargo run --release --example
//! openblas_device_phases -- 512 1024`.
use cubecl::prelude::*;
use lapack_rs::device::workspace::{set_phase_profiling, take_dlaed3_times};
use lapack_rs::device::{
    DeviceBudget, DeviceWorkspace, MergeU, dsyevd_device, dsyevd_device_openblas, zheevd_device, zheevd_device_openblas,
};
use lapack_rs::{Job, Uplo};

fn sym(n: usize) -> Vec<f64> {
    let mut s = 12345u64;
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
    }
    a
}

fn main() {
    let client = cubecl::cpu::CpuRuntime::client(&cubecl::cpu::CpuDevice);
    let ns: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    for n in if ns.is_empty() { vec![1024] } else { ns } {
        let a = sym(n);
        for ob in [false, true] {
            let mut ws = DeviceWorkspace::new(&client, n, Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Staged);
            let complex = std::env::var("COMPLEX").is_ok();
            let run = |ws: &mut DeviceWorkspace<_>| {
                let (mut x, mut w) = (a.clone(), vec![0.0; n]);
                if complex {
                    let mut xi: Vec<f64> = (0..n * n)
                        .map(|p| {
                            if p % (n + 1) == 0 {
                                0.0
                            } else if (p / n) > (p % n) {
                                0.25 * a[p]
                            } else {
                                -0.25 * a[(p % n) * n + p / n]
                            }
                        })
                        .collect();
                    if ob {
                        zheevd_device_openblas(&client, Job::Vectors, Uplo::Lower, n, &mut x, &mut xi, &mut w, ws)
                            .expect("ob");
                    } else {
                        zheevd_device(&client, Job::Vectors, Uplo::Lower, n, &mut x, &mut xi, &mut w, ws).expect("ref");
                    }
                    return;
                }
                if ob {
                    dsyevd_device_openblas(&client, Job::Vectors, Uplo::Lower, n, &mut x, &mut w, ws).expect("ob");
                } else {
                    dsyevd_device(&client, Job::Vectors, Uplo::Lower, n, &mut x, &mut w, ws).expect("ref");
                }
            };
            run(&mut ws);
            set_phase_profiling(true);
            ws.reset_counters();
            let _ = take_dlaed3_times();
            let t = std::time::Instant::now();
            run(&mut ws);
            let total = t.elapsed();
            set_phase_profiling(false);
            let p = ws.phase_times();
            let d3 = take_dlaed3_times();
            println!(
                "n={n} {}: total {:.3}s | stage {:.3} trd {:.3} stedc {:.3} (dlaed3 host: roots {:.3} w {:.3} cols {:.3}) orm {:.3} read {:.3}",
                if ob { "openblas " } else { "reference" },
                total.as_secs_f64(),
                p.stage.as_secs_f64(),
                p.trd.as_secs_f64(),
                p.stedc.as_secs_f64(),
                d3[0].as_secs_f64(),
                d3[1].as_secs_f64(),
                d3[2].as_secs_f64(),
                p.orm.as_secs_f64(),
                p.readback.as_secs_f64()
            );
            println!(
                "        spans: trd_update {:.3} stedc_leaf {:.3} stedc_merge {:.3} stedc_gather {:.3} orm_block {:.3} | read_wait {:.3}",
                p.spans[0].as_secs_f64(),
                p.spans[1].as_secs_f64(),
                p.spans[2].as_secs_f64(),
                p.spans[3].as_secs_f64(),
                p.spans[4].as_secs_f64(),
                p.read_wait.as_secs_f64()
            );
        }
    }
}
