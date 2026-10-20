//! Launch geometry for the device kernels.
//!
//! Modelled on `pyscf-algebra/src/launch.rs` (manual: Hardware-Adaptive Launch
//! Geometry). A "unit" is one lane-vector of `N` adjacent rows (or one 2-D
//! (lane, column) pair flattened into `ABSOLUTE_POS`); the cube dimension comes
//! from the device (whole planes on a GPU, a work-proportional thread count on
//! the CPU runtime) and the cube count from `calculate_cube_count_elemwise`,
//! which respects `max_cube_count`. Kernels decode the flattened
//! `ABSOLUTE_POS` and bounds-check; no aggregate `CUBE_COUNT`/`CUBE_DIM`
//! builtins (they panic on the CPU runtime).

use std::sync::atomic::{AtomicU32, Ordering};

use cubecl::prelude::*;

/// Test override of the CPU-runtime cube dimension (0 = automatic).
static CPU_DIM_OVERRIDE: AtomicU32 = AtomicU32::new(0);

/// Force every CPU-runtime launch to `dim` units per cube (0 restores the
/// work-proportional choice). Results must not depend on it.
#[doc(hidden)]
pub fn set_cpu_dim_override(dim: u32) {
    CPU_DIM_OVERRIDE.store(dim, Ordering::Relaxed);
}

/// [`set_cpu_dim_override`], or `LAPACK_RS_CPU_DIM` when it has not been set
/// in-process (0 = the work-proportional choice). A tuning knob only.
fn cpu_dim_override() -> u32 {
    let v = CPU_DIM_OVERRIDE.load(Ordering::Relaxed);
    if v != 0 {
        return v;
    }
    static ENV: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("LAPACK_RS_CPU_DIM").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// Element operations one CPU-runtime unit (an OS thread) should be worth.
pub const WORK_PER_CPU_UNIT: usize = 32 * 1024;
/// Ceiling on CPU-runtime units per cube.
pub const CPU_CUBE_DIM_MAX: u32 = 64;
/// Ceiling on grid-stride reduction lanes on a GPU.
pub const GPU_LANES_MAX: usize = 8192;

/// Whether the device has hardware planes (a GPU-like runtime).
pub fn has_planes<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().hardware.plane_size_max > 1
}

/// The widest I/O-optimised vector width for an element of `elem_bytes`.
pub fn vector_width<R: Runtime>(client: &ComputeClient<R>, elem_bytes: usize) -> usize {
    client.io_optimized_vector_sizes(elem_bytes).next().unwrap_or(1).max(1)
}

/// Leading dimension of device matrices: `n` rounded up to a multiple of the
/// widest f64 vector width and of the plane size, so every column is a whole
/// number of lane-vectors (at most 63 padding rows).
pub fn lda_for<R: Runtime>(client: &ComputeClient<R>, n: usize) -> usize {
    let w = vector_width(client, 8);
    let plane = client.properties().hardware.plane_size_max.max(1) as usize;
    let align = lcm(w, plane).max(1);
    n.div_ceil(align).max(1) * align
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 { a.max(b) } else { a / gcd(a, b) * b }
}

/// Cubes per streaming multiprocessor the GPU cube dimension aims for.
///
/// `CubeDim::new` grows the cube to 8 planes (256 units) as soon as there are
/// 256 units of work, which starves a GPU on the kernels whose unit count is
/// only `O(n)`: K1's `m` rows at `m = 1021` became 4 cubes of 256, i.e. 4 of
/// the T4's 40 SMs (measured 8.2% `sm__throughput`, `ncu`). Sizing the cube
/// so the count reaches `SMS · CUBES_PER_SM` instead spreads the same units
/// over every SM. Kernels with many units keep the 8-plane cube.
pub const CUBES_PER_SM: u32 = 4;

/// Override of [`CUBES_PER_SM`] (`u32::MAX` = take it from the environment
/// once).
static CUBES_OVERRIDE: AtomicU32 = AtomicU32::new(u32::MAX);

/// Force the cubes-per-SM target (benchmarks and the geometry parity gate);
/// `0` restores cubecl's element-wise default cube, `u32::MAX` re-reads
/// `LAPACK_RS_CUBES_PER_SM`. A tuning knob only: the units of every kernel
/// are element-disjoint, so no geometry changes a bit
/// (`tests/device_launch_parity.rs`,
/// `tests/device_cuda.rs::cuda_variants_checked_launch_and_resident`).
#[doc(hidden)]
pub fn set_cubes_per_sm(v: u32) {
    CUBES_OVERRIDE.store(v, Ordering::Relaxed);
}

/// [`CUBES_PER_SM`], or `LAPACK_RS_CUBES_PER_SM` / [`set_cubes_per_sm`] when
/// set (0 restores cubecl's element-wise default cube).
fn cubes_per_sm() -> u32 {
    let v = CUBES_OVERRIDE.load(Ordering::Relaxed);
    if v != u32::MAX {
        return v;
    }
    let env = std::env::var("LAPACK_RS_CUBES_PER_SM").ok().and_then(|v| v.parse().ok()).unwrap_or(CUBES_PER_SM);
    CUBES_OVERRIDE.store(env, Ordering::Relaxed);
    env
}

/// `(CubeCount, CubeDim)` for `units` units of `work_per_unit` flops each.
pub fn launch_1d<R: Runtime>(client: &ComputeClient<R>, units: usize, work_per_unit: usize) -> (CubeCount, CubeDim) {
    let hardware = &client.properties().hardware;
    let dim = if hardware.plane_size_max > 1 {
        let plane = hardware.plane_size_max.max(1);
        // cubecl's element-wise default is the ceiling: never a larger cube,
        // so a kernel that already fills the device is left untouched.
        let max_units = CubeDim::new(client, units).num_elems().max(plane);
        match (hardware.num_streaming_multiprocessors, cubes_per_sm()) {
            (Some(sms), per_sm) if sms > 0 && per_sm > 0 => {
                let target_cubes = sms.saturating_mul(per_sm).max(1);
                let want = (units as u32).div_ceil(target_cubes).max(1);
                let planes = want.div_ceil(plane).clamp(1, max_units / plane);
                CubeDim::new_2d(plane, planes)
            }
            _ => CubeDim::new(client, units),
        }
    } else {
        let cores = hardware.num_cpu_cores.unwrap_or(1).max(1) as usize;
        let total = units.saturating_mul(work_per_unit.max(1));
        let n = (total / WORK_PER_CPU_UNIT).clamp(1, cores.min(units.max(1)));
        match cpu_dim_override() {
            0 => CubeDim::new_1d((n as u32).min(CPU_CUBE_DIM_MAX)),
            o => CubeDim::new_1d(o.min(units.max(1) as u32)),
        }
    };
    (cubecl::calculate_cube_count_elemwise(client, units, dim), dim)
}

/// A 2-D launch (`lanes_x` lane-vectors × `cols` columns) flattened into
/// `ABSOLUTE_POS = col * lanes_x + lane`: adjacent units are adjacent rows of
/// one column, which is the coalesced axis. Returns the geometry and the unit
/// count the kernel bounds-checks against.
pub fn launch_2d<R: Runtime>(
    client: &ComputeClient<R>,
    lanes_x: usize,
    cols: usize,
    work_per_unit: usize,
) -> (CubeCount, CubeDim, usize) {
    let units = lanes_x.saturating_mul(cols);
    let (count, dim) = launch_1d(client, units, work_per_unit);
    (count, dim, units)
}
