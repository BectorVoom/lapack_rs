//! The default CubeCL backend.
//!
//! There is no host parallel path: the entry points that own their workspace
//! ([`crate::dsyevd_alloc`], [`crate::zheevd_alloc`]) run the device drivers
//! on the runtime selected at compile time. `cuda` wins over `rocm`, which
//! wins over the CPU runtime, and the bits are the same on all three.
//!
//! The explicit-`LWORK` entry points ([`crate::dsyevd`], [`crate::zheevd`])
//! stay the sequential reference transcription: their results depend on the
//! workspace lengths the caller passes (the reference derives its block sizes
//! from `LWORK`), which the device drivers do not model — they reproduce the
//! reference at [`crate::syevd_opt_work_len`].

use std::cell::RefCell;

use cubecl::prelude::*;

use crate::device::{DeviceBudget, DeviceWorkspace, MergeU};

#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "cpu")))]
compile_error!("lapack_rs needs a CubeCL runtime: enable `cpu` (the default), `rocm` or `cuda`");

/// The runtime the no-workspace entry points use.
#[cfg(feature = "cuda")]
pub type DefaultRuntime = cubecl_cuda::CudaRuntime;
/// The runtime the no-workspace entry points use.
#[cfg(all(not(feature = "cuda"), feature = "rocm"))]
pub type DefaultRuntime = cubecl_hip::HipRuntime;
/// The runtime the no-workspace entry points use.
#[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "cpu"))]
pub type DefaultRuntime = cubecl::cpu::CpuRuntime;

/// A client on the default device. cubecl caches clients per device, so this
/// is cheap to call.
pub fn default_client() -> ComputeClient<DefaultRuntime> {
    #[cfg(feature = "cuda")]
    return DefaultRuntime::client(&cubecl_cuda::CudaDevice::default());
    #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
    return DefaultRuntime::client(&cubecl_hip::AmdDevice::default());
    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "cpu"))]
    return DefaultRuntime::client(&cubecl::cpu::CpuDevice);
}

thread_local! {
    /// One workspace per thread, grown on demand: `DeviceWorkspace::reserve`
    /// keeps the device buffers across calls, which is what makes repeated
    /// solves (an SCF loop) pay the allocation once.
    static WORKSPACE: RefCell<Option<DeviceWorkspace<DefaultRuntime>>> = const { RefCell::new(None) };
}

/// Run `f` with the thread's cached client and workspace, sized for `n`.
pub(crate) fn with_workspace<T>(
    n: usize,
    f: impl FnOnce(&ComputeClient<DefaultRuntime>, &mut DeviceWorkspace<DefaultRuntime>) -> T,
) -> T {
    let client = default_client();
    WORKSPACE.with(|cell| {
        let mut slot = cell.borrow_mut();
        let ws = slot.get_or_insert_with(|| {
            DeviceWorkspace::new(&client, n, crate::Job::Vectors, DeviceBudget::UNLIMITED, MergeU::Auto)
        });
        f(&client, ws)
    })
}

/// Drop the calling thread's cached workspace and its device buffers.
pub fn release_workspace() {
    WORKSPACE.with(|cell| *cell.borrow_mut() = None);
}
