//! Device capability probes: f64 support, floating-point contraction and K4
//! (`DLAED3` secular roots on the device).

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Mutex;

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;

use crate::LapackError;

/// Whether the client can store and compute `f64`.
pub fn supports_f64<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().supports_type(ElemType::Float(FloatKind::F64))
}

/// Refuse an f64 launch on a device without f64 (never fall back silently).
pub fn require_f64<R: Runtime>(client: &ComputeClient<R>) -> Result<(), LapackError> {
    if supports_f64(client) {
        Ok(())
    } else {
        Err(LapackError::DeviceUnsupported { reason: format!("{} reports no f64 support", std::any::type_name::<R>()) })
    }
}

/// Result of the manual 15 §3.3 contraction probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContractionProbe {
    /// `a*b + c == 0` with `a = 1+eps`, `b = 1-eps/2`, `c = -1`: the multiply
    /// was rounded on its own (no contraction), scalar form.
    pub separate_mul_add_scalar: bool,
    /// The same property through `Vector<F, N>` lanes (the kernels' form).
    pub separate_mul_add_vector: bool,
}

impl ContractionProbe {
    /// Bit-exactness against the unfused reference is possible only when no
    /// form contracts.
    pub fn no_contraction(&self) -> bool {
        self.separate_mul_add_scalar && self.separate_mul_add_vector
    }
}

#[cube(launch)]
fn contraction_probe_kernel<F: Float + CubeElement, N: Size>(
    inp: &Array<F>,
    vin: &Array<Vector<F, N>>,
    out: &mut Array<F>,
    vout: &mut Array<Vector<F, N>>,
) {
    if ABSOLUTE_POS < 1usize {
        let a = inp[0];
        let b = inp[1];
        let c = inp[2];
        out[0] = a * b + c;
        vout[0] = vin[0] * vin[1] + vin[2];
    }
}

/// Run the contraction probe for `F` on `client` (vector width = the widest
/// the device reports for `F`).
pub fn contraction_probe<R: Runtime, F: Float + CubeElement + bytemuck::Pod>(
    client: &ComputeClient<R>,
    eps: F,
    one: F,
    two: F,
) -> ContractionProbe {
    let width = client.io_optimized_vector_sizes(core::mem::size_of::<F>()).next().unwrap_or(1);
    let a = one + eps;
    let b = one - eps / two;
    let c = F::new(-1.0);
    let inp = [a, b, c];
    let mut vin = vec![a; 3 * width];
    vin[width..2 * width].fill(b);
    vin[2 * width..].fill(c);
    let h_in = super::workspace::upload(client, &inp);
    let h_vin = super::workspace::upload(client, &vin);
    let h_out = client.empty(core::mem::size_of::<F>());
    let h_vout = client.empty(3 * width * core::mem::size_of::<F>());
    // SAFETY: lengths match the allocations above (3, 3*width, 1, 3*width).
    unsafe {
        contraction_probe_kernel::launch::<F, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            width,
            ArrayArg::from_raw_parts(h_in, 3),
            ArrayArg::from_raw_parts(h_vin, 3 * width),
            ArrayArg::from_raw_parts(h_out.clone(), 1),
            ArrayArg::from_raw_parts(h_vout.clone(), 3 * width),
        );
    }
    let bytes = client.read(vec![h_out, h_vout]);
    let out: &[F] = bytemuck::cast_slice(&bytes[0]);
    let vout: &[F] = bytemuck::cast_slice(&bytes[1]);
    let zero = F::new(0.0);
    ContractionProbe {
        separate_mul_add_scalar: out[0] == zero,
        separate_mul_add_vector: vout[..width].iter().all(|&v| v == zero),
    }
}

/// Per-runtime-type cache for [`k4_supported`]. The failure mode the probe
/// detects (`crates/cubecl-opt/PATCHED.md`) is a codegen bug in the
/// runtime's backend compiler, a property of `R`/`R::Compiler`, not of any
/// one physical device or `ComputeClient` instance (which exposes no public
/// identity to key a cache on anyway) — so caching by `TypeId::of::<R>()` is
/// both the right granularity and cheap: every call site for a given
/// runtime shares one verdict after the first `k = 3` probe.
static K4_SUPPORTED_CACHE: Mutex<Option<HashMap<TypeId, bool>>> = Mutex::new(None);

/// Fixed `k = 3` secular-equation fixture for [`k4_supported`]: distinct
/// ascending `DLAMBDA`, nonzero `Z`, `rho > 0` (the same shape
/// `tests/device_secular_cpu.rs`'s `fixture` builds), inlined so the probe
/// has no test-only dependency.
fn k4_probe_fixture() -> ([f64; 3], [f64; 3], f64) {
    ([0.1, 1.4, 2.7], [0.37, 0.5, 0.6], 1.0)
}

/// Runs the real `secular_roots_kernel` (`src/device/kernels/secular.rs`,
/// via [`super::ops::secular_roots`]) once on [`k4_probe_fixture`] and
/// compares `dlam`/`info` bit for bit against the host `DLAED4`
/// ([`crate::testing::dlaed4`]) on the identical fixture, computed at
/// runtime every call — never a hardcoded literal, so this stays correct
/// even if the fixture or either implementation ever changes.
fn k4_probe_matches_host<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let (d, z, rho) = k4_probe_fixture();
    let k = d.len();
    let consts = super::stedc::secular_consts();
    let h_d = super::workspace::upload(client, &d);
    let h_z = super::workspace::upload(client, &z);
    let delta = client.empty(k * k * 8);
    let dlam = client.empty(k * 8);
    let info = client.empty(k * 4);
    super::ops::secular_roots::<R, f64>(client, &h_d, &h_z, &delta, &dlam, &info, k, rho, &consts);
    let mut got_dlam = vec![0.0f64; k];
    let mut got_info = vec![0u32; k];
    super::workspace::read_range(client, &dlam, k, 0, k, &mut got_dlam);
    super::workspace::read_range(client, &info, k, 0, k, &mut got_info);

    let mut want_dlam = vec![0.0f64; k];
    let mut want_info = vec![0u32; k];
    for j in 1..=k {
        let mut dd = d.to_vec();
        let mut zz = z.to_vec();
        let mut col = vec![0.0f64; k];
        // SAFETY: `dd`, `zz`, `col` all have length `k` and `1 <= j <= k`,
        // exactly `DLAED4`'s Fortran shape contract.
        let (lam, info) = unsafe { crate::testing::dlaed4(k, j, &mut dd, &mut zz, &mut col, rho) };
        want_dlam[j - 1] = lam;
        want_info[j - 1] = info as u32;
    }
    got_dlam == want_dlam && got_info == want_info
}

/// Whether `client` can run K4 (`DLAED3`'s secular-root solve,
/// `src/device/kernels/secular.rs`) correctly. Two upstream cubecl 0.10 bugs
/// (documented in `crates/cubecl-opt/PATCHED.md` and
/// `src/device/stedc.rs::DeviceDc::secular`) can make the compiled kernel
/// panic on a cubecl worker thread; the panic does not propagate to the
/// calling thread, so the launch silently no-ops and every output reads
/// back as zero. This probe turns that silent-zero hazard into an explicit
/// `false` verdict — a build without the vendored `crates/cubecl-opt` patch
/// (any consumer outside this workspace; see the "Caveat for publishing"
/// notes) reports `k4_supported() == false` and callers fall back to the
/// host transcription, instead of returning a wrong answer. The verdict is
/// cached per runtime type; see [`K4_SUPPORTED_CACHE`].
pub fn k4_supported<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let ty = TypeId::of::<R>();
    if let Some(&v) = K4_SUPPORTED_CACHE.lock().unwrap().as_ref().and_then(|m| m.get(&ty)) {
        return v;
    }
    let v = k4_probe_matches_host(client);
    K4_SUPPORTED_CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(ty, v);
    v
}

/// Same caching rationale as [`K4_SUPPORTED_CACHE`], for K9.
static K9_SUPPORTED_CACHE: Mutex<Option<HashMap<TypeId, bool>>> = Mutex::new(None);

/// The K9 probe fixture: one `n = 5` tridiagonal with a well-separated
/// spectrum, so the answer does not depend on a tolerance. Deliberately tiny —
/// the probe runs once per runtime type.
fn k9_probe_fixture() -> (Vec<f64>, Vec<f64>) {
    (vec![4.0, 3.0, 2.0, 1.0, 0.5], vec![0.25, 0.125, 0.0625, 0.03125, 0.0])
}

/// Launches the one-problem K9 parity kernel on [`k9_probe_fixture`] and
/// compares every output bit for bit against [`crate::cint_stemr`] on the
/// identical fixture, computed at runtime on every first call — never a
/// hardcoded literal, so this stays correct if either side changes.
fn k9_probe_matches_host<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let (d0, e0) = k9_probe_fixture();
    let n = d0.len();
    let (lw, liw) = crate::cint_stemr::cint_stemr_work_len(n);
    let consts = super::ops::cint_stemr_consts();
    let big = consts[super::kernels::cint_stemr::CC_BIG as usize];

    let hd = super::workspace::upload(client, &d0);
    let he = super::workspace::upload(client, &e0);
    let heig = super::workspace::upload(client, &vec![0.0f64; n]);
    let hvec = super::workspace::upload(client, &vec![0.0f64; n * n]);
    let hwork = super::workspace::upload(client, &vec![0.0f64; lw]);
    let hiwork = super::workspace::upload(client, &vec![0u32; liw.max(1)]);
    let hcc = super::workspace::upload(client, &consts);
    let hsc = super::workspace::upload(client, &vec![0.0f64; super::kernels::cint_stemr::SC_LEN as usize]);
    let hicnt = super::workspace::upload(client, &vec![0u32; super::kernels::cint_stemr::ICNT_LEN as usize]);
    let hinfo = super::workspace::upload(client, &vec![0u32; 1]);
    super::ops::cint_stemr_one::<R, f64>(
        client,
        &hd,
        &he,
        &heig,
        &hvec,
        &hwork,
        &hiwork,
        &hcc,
        &hsc,
        &hicnt,
        &hinfo,
        n,
        f64::EPSILON,
        big,
    );
    let mut got_eig = vec![0.0f64; n];
    let mut got_vec = vec![0.0f64; n * n];
    let mut got_info = vec![0u32; 1];
    super::workspace::read_range(client, &heig, n, 0, n, &mut got_eig);
    super::workspace::read_range(client, &hvec, n * n, 0, n * n, &mut got_vec);
    super::workspace::read_range(client, &hinfo, 1, 0, 1, &mut got_info);

    let (mut d, mut e) = (d0.clone(), e0.clone());
    let mut want_eig = vec![0.0f64; n];
    let mut want_vec = vec![0.0f64; n * n];
    let (mut work, mut iwork) = (vec![0.0f64; lw], vec![0i32; liw.max(1)]);
    let want_info =
        crate::cint_stemr::cint_stemr(n, &mut d, &mut e, &mut want_eig, &mut want_vec, &mut work, &mut iwork);

    got_info[0] as i32 == want_info.abs() && got_eig == want_eig && got_vec == want_vec
}

/// Whether `client` can run **K9** (Path C, libcint 6.1.3's tridiagonal
/// eigensolver, `src/device/kernels/cint_stemr.rs`) correctly.
///
/// **This is not optional for a production caller.** cubecl-cpu 0.10 fails to
/// lower several of the K9 routines: the MLIR verifier error and the panic
/// land on a cubecl worker thread, the launching thread sees success, and the
/// output buffers keep their previous contents — so a caller that does not
/// probe reads **zeros**, not an error
/// (`docs/cubecl-cpu-lowering-limit.md`). hipRTC and NVRTC lower the whole
/// solver, so the verdict really is per backend.
///
/// Modelled on [`k4_supported`] and cached per runtime type: one `n = 5`
/// launch, compared bit for bit against [`crate::cint_stemr`]. A `false`
/// verdict means "run the host transcription instead" — which is bit-exact to
/// libcint either way, just slower.
pub fn k9_supported<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let ty = TypeId::of::<R>();
    if let Some(&v) = K9_SUPPORTED_CACHE.lock().unwrap().as_ref().and_then(|m| m.get(&ty)) {
        return v;
    }
    let v = supports_f64(client) && k9_probe_matches_host(client);
    K9_SUPPORTED_CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(ty, v);
    v
}
