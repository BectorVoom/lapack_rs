#![allow(dead_code)]
//! Component-level access to the linked OpenBLAS: the live `gotoblas`
//! parameter table and the internal (non-exported) SkylakeX kernels, called
//! at their `nm -n` addresses relative to `scipy_zgemm_`. Used to bisect a
//! mismatch to one kernel/copy routine; the end-to-end oracle stays the
//! exported `scipy_*` entry points.
use std::ffi::{c_char, c_void};

unsafe extern "C" {
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}
const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

/// `nm -n libscipy_openblas-6cdc3b4a.so` addresses (see `target/ob/syms_sorted.txt`).
pub const SYM_SCIPY_ZGEMM: usize = 0x113ab0;
pub const SYM_ZGEMM_BETA: usize = 0xb7bc00;
pub const SYM_ZGEMM_KERNEL_N: usize = 0xb6ab70;
pub const SYM_ZGEMM_KERNEL_R: usize = 0xb6ea90;
pub const SYM_ZGEMM_KERNEL_L: usize = 0xb729b0;
pub const SYM_ZGEMM_KERNEL_B: usize = 0xb768d0;
pub const SYM_ZGEMM_INCOPY: usize = 0xb7a7f0;
pub const SYM_ZGEMM_ITCOPY: usize = 0xb7aeb0;
pub const SYM_ZGEMM_ONCOPY: usize = 0xb7b5e0;
pub const SYM_ZGEMM_OTCOPY: usize = 0xb7b880;
pub const SYM_ZTRMM_KERNEL_LN: usize = 0xb7be00;
pub const SYM_ZTRMM_KERNEL_LT: usize = 0xb7e200;
pub const SYM_ZTRMM_KERNEL_LR: usize = 0xb80800;
pub const SYM_ZTRMM_KERNEL_LC: usize = 0xb82e00;
pub const SYM_ZTRMM_KERNEL_RN: usize = 0xb85400;
pub const SYM_ZTRMM_KERNEL_RT: usize = 0xb87a00;
pub const SYM_ZTRMM_KERNEL_RR: usize = 0xb89e00;
pub const SYM_ZTRMM_KERNEL_RC: usize = 0xb8c400;
pub const SYM_ZTRMM_IUNUCOPY: usize = 0xb94a90;
pub const SYM_ZTRMM_IUNNCOPY: usize = 0xb95520;
pub const SYM_ZTRMM_ILNUCOPY: usize = 0xb95f50;
pub const SYM_ZTRMM_ILNNCOPY: usize = 0xb96a20;
pub const SYM_ZTRMM_IUTUCOPY: usize = 0xb97460;
pub const SYM_ZTRMM_IUTNCOPY: usize = 0xb97f70;
pub const SYM_ZTRMM_ILTUCOPY: usize = 0xb98940;
pub const SYM_ZTRMM_ILTNCOPY: usize = 0xb99370;
pub const SYM_ZTRMM_OUNUCOPY: usize = 0xb99d30;
pub const SYM_ZTRMM_OUNNCOPY: usize = 0xb9a090;
pub const SYM_ZTRMM_OLNUCOPY: usize = 0xb9a3e0;
pub const SYM_ZTRMM_OLNNCOPY: usize = 0xb9a720;
pub const SYM_ZTRMM_OUTUCOPY: usize = 0xb9aa40;
pub const SYM_ZTRMM_OUTNCOPY: usize = 0xb9adb0;
pub const SYM_ZTRMM_OLTUCOPY: usize = 0xb9b0b0;
pub const SYM_ZTRMM_OLTNCOPY: usize = 0xb9b3f0;
pub const SYM_ZHER2K_KERNEL_UN: usize = 0x3119e0;
pub const SYM_ZHER2K_KERNEL_LN: usize = 0x312220;

/// Load address of the linked `.so`, checked against the first bytes of
/// `zgemm_beta_SKYLAKEX` (`pxor %xmm15,%xmm15`).
pub fn base() -> usize {
    // SAFETY: dlsym with a NUL-terminated name.
    let p = unsafe { dlsym(RTLD_DEFAULT, c"scipy_zgemm_".as_ptr()) } as usize;
    assert!(p != 0, "scipy_zgemm_ not found");
    let base = p - SYM_SCIPY_ZGEMM;
    // SAFETY: reading code bytes of the mapped library.
    let head = unsafe { std::slice::from_raw_parts((base + SYM_ZGEMM_BETA) as *const u8, 5) };
    assert_eq!(head, &[0x66, 0x45, 0x0f, 0xef, 0xff], "library layout differs from the pinned .so");
    base
}

/// `int` field of the live `gotoblas_t` at byte offset `off`.
pub fn gotoblas_int(off: usize) -> i32 {
    // SAFETY: `gotoblas` is a data symbol holding a pointer to the table.
    unsafe {
        let pp = dlsym(RTLD_DEFAULT, c"gotoblas".as_ptr()) as *const *const u8;
        assert!(!pp.is_null());
        let t = *pp;
        assert!(!t.is_null(), "gotoblas table not initialised");
        std::ptr::read_unaligned(t.add(off) as *const i32)
    }
}

pub type GemmKernel = unsafe extern "C" fn(i64, i64, i64, f64, f64, *const f64, *const f64, *mut f64, i64) -> i32;
pub type TrmmKernel = unsafe extern "C" fn(i64, i64, i64, f64, f64, *const f64, *const f64, *mut f64, i64, i64) -> i32;
pub type GemmCopy = unsafe extern "C" fn(i64, i64, *const f64, i64, *mut f64) -> i32;
pub type TrmmCopy = unsafe extern "C" fn(i64, i64, *const f64, i64, i64, i64, *mut f64) -> i32;
pub type Beta = unsafe extern "C" fn(i64, i64, i64, f64, f64, *const f64, i64, *const f64, i64, *mut f64, i64) -> i32;
pub type Her2kKernel =
    unsafe extern "C" fn(i64, i64, i64, f64, f64, *const f64, *const f64, *mut f64, i64, i64, i32) -> i32;

/// # Safety
/// `sym` must be the address of a function with signature `T`.
pub unsafe fn func<T: Copy>(sym: usize) -> T {
    let addr = base() + sym;
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<usize>());
    // SAFETY: caller guarantees the signature.
    unsafe { std::mem::transmute_copy(&addr) }
}
