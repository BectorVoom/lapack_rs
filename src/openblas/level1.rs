//! OpenBLAS 0.3.30 SkylakeX track: BLAS level 1, real double precision.
//!
//! Bit-exact to `scipy_{ddot,daxpy,dscal,dnrm2,dcopy,dswap,drot,idamax}_` of
//! `libscipy_openblas-6cdc3b4a.so` (`OPENBLAS_CORETYPE=SkylakeX`,
//! `OPENBLAS_NUM_THREADS=1`), positive increments only. Transcribed from the
//! disassembly, with the source for structure:
//!
//! | routine | interface          | kernel source (`kernel/x86_64/`)                 | symbol               |
//! |---------|--------------------|--------------------------------------------------|----------------------|
//! | ddot    | `interface/dot.c`  | `ddot.c` + `ddot_microk_skylakex-2.c`            | `ddot_k_SKYLAKEX` (`dot_compute`) |
//! | daxpy   | `interface/axpy.c` | `daxpy.c` + `daxpy_microk_skylakex-2.c`          | `daxpy_k_SKYLAKEX`   |
//! | dscal   | `interface/scal.c` | `dscal.c` + `dscal_microk_skylakex-2.c`          | `dscal_k_SKYLAKEX`   |
//! | dnrm2   | `interface/nrm2.c` | `nrm2.S` (x87, `KERNEL` default)                 | `dnrm2_k_SKYLAKEX`   |
//! | dcopy   | `interface/copy.c` | `copy_sse2.S`                                    | `dcopy_k_SKYLAKEX`   |
//! | dswap   | `interface/swap.c` | `swap_sse2.S`                                    | `dswap_k_SKYLAKEX`   |
//! | drot    | `interface/rot.c`  | `drot.c` + `drot_microk_skylakex-2.c`            | `drot_k_SKYLAKEX` (`rot_compute`) |
//! | idamax  | `interface/imax.c` | `iamax_sse2.S`                                   | `idamax_k_SKYLAKEX`  |
//!
//! (`KERNEL.SKYLAKEX` includes `KERNEL.HASWELL`, which sets only the ddot,
//! daxpy, dscal and drot kernels; the rest fall back to `kernel/x86_64/KERNEL`.)
//!
//! Arithmetic: gcc fused `acc += a*b` into `vfmadd*`, written `a.mul_add(b,
//! acc)`; separate `vmul*` stay separate. `DNRM2` runs on the x87 stack in
//! 64-bit-mantissa extended precision (the process default control word), so
//! it is emulated exactly by [`x87`]. `IDAMAX` is an SSE2 `maxpd` lane scan
//! followed by a search whose block filter and NaN behaviour depend on the
//! address of `X` modulo 16; both are emulated at register level.
use crate::ptr::D;

// ---------------------------------------------------------------- DDOT ----

/// `ddot_kernel_8` (AVX-512 path): `n` is a multiple of 16.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn ddot_kernel_8(n: usize, x: D, y: D) -> f64 {
    let mut acc5 = [[0.0f64; 8]; 4];
    let n32 = n & !31;
    let mut i = 0;
    while i < n32 {
        for (b, a) in acc5.iter_mut().enumerate() {
            for l in 0..8 {
                let k = i + 8 * b + l;
                a[l] = x.at(k).mul_add(y.at(k), a[l]);
            }
        }
        i += 32;
    }
    // Fold the 512-bit accumulators to 256 bits: lane l + lane l+4.
    let mut acc = [[0.0f64; 4]; 4];
    for b in 0..4 {
        for l in 0..4 {
            acc[b][l] = acc5[b][l] + acc5[b][l + 4];
        }
    }
    while i < n {
        for (b, a) in acc.iter_mut().enumerate() {
            for l in 0..4 {
                let k = i + 4 * b + l;
                a[l] = x.at(k).mul_add(y.at(k), a[l]);
            }
        }
        i += 16;
    }
    // accum_0 + accum_1 + accum_2 + accum_3, then (l0 + l2) + (l1 + l3).
    let mut s = [0.0f64; 4];
    for l in 0..4 {
        s[l] = ((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l];
    }
    (s[0] + s[2]) + (s[1] + s[3])
}

/// `DDOT`. Single-thread `dot_compute`: `n & -16` through the AVX-512 kernel,
/// fused scalar tail; `inc != 1` is a 4-unrolled pair of chains.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn ddot(n: usize, x: D, incx: usize, y: D, incy: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if incx == 1 && incy == 1 {
        let n1 = n & !15;
        let mut dot = 0.0f64;
        if n1 != 0 {
            dot = ddot_kernel_8(n1, x, y);
        }
        for i in n1..n {
            dot = y.at(i).mul_add(x.at(i), dot);
        }
        return dot;
    }
    let (mut t1, mut t2) = (0.0f64, 0.0f64);
    let n1 = n & !3;
    let (mut ix, mut iy, mut i) = (0usize, 0usize, 0usize);
    while i < n1 {
        let m3 = y.at(iy + 2 * incy) * x.at(ix + 2 * incx);
        let m4 = y.at(iy + 3 * incy) * x.at(ix + 3 * incx);
        t1 += y.at(iy).mul_add(x.at(ix), m3);
        t2 += y.at(iy + incy).mul_add(x.at(ix + incx), m4);
        ix += 4 * incx;
        iy += 4 * incy;
        i += 4;
    }
    while i < n {
        t1 = y.at(iy).mul_add(x.at(ix), t1);
        ix += incx;
        iy += incy;
        i += 1;
    }
    t2 + t1
}

// --------------------------------------------------------------- DAXPY ----

/// `DAXPY`. Interface: `n <= 0` or `alpha == 0` (ordered compare, so a NaN
/// alpha proceeds) returns; `n > 10000` asks for threads but
/// `blas_cpu_number == 1` keeps the single call. Every path of
/// `daxpy_k_SKYLAKEX` (zmm 32-blocks, ymm 16-blocks, scalar tail; the
/// 4-unrolled `inc != 1` loop and its tail) computes
/// `vfmadd213(alpha, x, y) = alpha*x + y` in one rounding per element.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn daxpy(n: usize, da: f64, x: D, incx: usize, y: D, incy: usize) {
    if n == 0 || da == 0.0 {
        return;
    }
    let (mut ix, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        y.set(iy, da.mul_add(x.at(ix), y.at(iy)));
        ix += incx;
        iy += incy;
    }
}

// --------------------------------------------------------------- DSCAL ----

/// `DSCAL`. Interface: `incx <= 0`, `n <= 0` or `alpha == 1` (ordered)
/// returns; `n <= 1048576` single-threaded; the kernel is called with
/// `dummy2 = 1`, which selects the issue-4728 branch of `dscal.c`: even
/// `alpha == 0` multiplies (`x *= 0`, so NaN stays NaN, ±Inf becomes the
/// default NaN, negatives become -0) instead of zero-filling. Every path
/// (`vmulpd` 8-blocks, `mulpd` in `dscal_kernel_inc_8`, scalar tails) is a
/// plain `alpha * x`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dscal(n: usize, da: f64, x: D, incx: usize) {
    if n == 0 || incx == 0 || da == 1.0 {
        return;
    }
    let mut i = 0usize;
    for _ in 0..n {
        x.set(i, da * x.at(i));
        i += incx;
    }
}

// --------------------------------------------------------------- DNRM2 ----

/// Exact emulation of the x87 operations `nrm2.S` uses, in the default
/// precision control (64-bit mantissa, round to nearest even). Only
/// non-negative finite values, +Inf and NaNs occur (the inputs are squares).
pub(crate) mod x87 {
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Ext {
        /// `m * 2^e`; `m == 0` is +0, otherwise bit 63 of `m` is set.
        Fin {
            m: u64,
            e: i32,
        },
        Inf,
        /// Extended NaN: 64-bit significand (bit 63 integer bit, bit 62 quiet).
        Nan {
            sig: u64,
            neg: bool,
        },
    }

    pub const ZERO: Ext = Ext::Fin { m: 0, e: 0 };
    const FRAC: u64 = (1 << 52) - 1;

    /// Round `p * 2^e` (`p != 0`) to a 64-bit mantissa.
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    fn round128(p: u128, e: i32) -> Ext {
        let lz = p.leading_zeros();
        let p = p << lz;
        let mut e = e - lz as i32 + 64;
        let mut hi = (p >> 64) as u64;
        let lo = p as u64;
        if lo > 1 << 63 || (lo == 1 << 63 && hi & 1 == 1) {
            hi = hi.wrapping_add(1);
            if hi == 0 {
                hi = 1 << 63;
                e += 1;
            }
        }
        Ext::Fin { m: hi, e }
    }

    /// `fldl (x); fmul %st(0), %st`.
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    pub fn load_square(d: f64) -> Ext {
        let bits = d.to_bits();
        let exp = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & FRAC;
        if exp == 0x7ff {
            if frac == 0 {
                return Ext::Inf;
            }
            // fld quiets a signalling NaN; squaring a NaN returns it.
            return Ext::Nan { sig: (1 << 63) | (1 << 62) | (frac << 11), neg: bits >> 63 == 1 };
        }
        if exp == 0 && frac == 0 {
            return ZERO;
        }
        let (m, e) = if exp == 0 { (frac, -1074) } else { (frac | (1 << 52), exp - 1075) };
        let lz = m.leading_zeros();
        let (m, e) = (m << lz, e - lz as i32);
        round128((m as u128) * (m as u128), 2 * e)
    }

    /// `faddp %st, %st(i)`: `dst = dst + src`.
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    pub fn add(dst: Ext, src: Ext) -> Ext {
        match (dst, src) {
            (Ext::Nan { sig: a, neg: na }, Ext::Nan { sig: b, .. }) => {
                // Two QNaNs: the one with the larger significand; on a tie the
                // positive one, whichever operand it is (measured against the
                // oracle with both orders).
                if b > a || (b == a && na) { src } else { dst }
            }
            (Ext::Nan { .. }, _) => dst,
            (_, Ext::Nan { .. }) => src,
            (Ext::Inf, _) | (_, Ext::Inf) => Ext::Inf,
            (Ext::Fin { m: 0, .. }, _) => src,
            (_, Ext::Fin { m: 0, .. }) => dst,
            (Ext::Fin { m: ma, e: ea }, Ext::Fin { m: mb, e: eb }) => {
                let ((ma, ea), (mb, eb)) = if ea >= eb { ((ma, ea), (mb, eb)) } else { ((mb, eb), (ma, ea)) };
                let d = (ea - eb) as u32;
                let a = (ma as u128) << 62;
                let b = (mb as u128) << 62;
                let bs = if d >= 127 {
                    1
                } else {
                    let s = b >> d;
                    if s << d != b { s | 1 } else { s }
                };
                round128(a + bs, ea - 62)
            }
        }
    }

    /// `fsqrt`.
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    pub fn sqrt(a: Ext) -> Ext {
        match a {
            Ext::Fin { m: 0, .. } | Ext::Inf | Ext::Nan { .. } => a,
            Ext::Fin { m, e } => {
                let (big, ee) =
                    if e.rem_euclid(2) == 0 { ((m as u128) << 64, e - 64) } else { ((m as u128) << 63, e - 63) };
                let q = big.isqrt();
                let rem = big - q * q;
                // sqrt(big) is never q + 1/2 exactly; round up iff rem > q.
                let (mut q, mut ee2) = (q as u64, ee / 2);
                if rem > q as u128 {
                    q = q.wrapping_add(1);
                    if q == 0 {
                        q = 1 << 63;
                        ee2 += 1;
                    }
                }
                Ext::Fin { m: q, e: ee2 }
            }
        }
    }

    /// `fstpl`: round to double (nearest even; overflow to +Inf, gradual
    /// underflow).
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    pub fn to_f64(a: Ext) -> f64 {
        match a {
            Ext::Nan { sig, neg } => f64::from_bits(((neg as u64) << 63) | (0x7ff << 52) | ((sig >> 11) & FRAC)),
            Ext::Inf => f64::INFINITY,
            Ext::Fin { m: 0, .. } => 0.0,
            Ext::Fin { m, e } => {
                let mut ex = e + 63;
                if ex >= -1022 {
                    let mut mant = m >> 11;
                    let rem = m & 0x7ff;
                    if rem > 0x400 || (rem == 0x400 && mant & 1 == 1) {
                        mant += 1;
                        if mant == 1 << 53 {
                            mant >>= 1;
                            ex += 1;
                        }
                    }
                    if ex > 1023 {
                        return f64::INFINITY;
                    }
                    f64::from_bits(((ex + 1023) as u64) << 52 | (mant & FRAC))
                } else {
                    let s = (-(e + 1074)) as u32;
                    if s >= 127 {
                        return 0.0;
                    }
                    let t = m as u128;
                    let mut mant = t >> s;
                    let rem = t & ((1u128 << s) - 1);
                    let half = 1u128 << (s - 1);
                    if rem > half || (rem == half && mant & 1 == 1) {
                        mant += 1;
                    }
                    f64::from_bits(mant as u64)
                }
            }
        }
    }
}

/// `DNRM2`. Interface (`interface/nrm2.c`): `n <= 0` gives 0, `n == 1` gives
/// `|x(1)|` (`andpd`) without the kernel. `nrm2.S` pushes four x87
/// accumulators `A, B, C, R` (all +0) and, per block of 8, adds the squares
/// of elements `4k+3, 4k+2, 4k+1, 4k` into `R, C, B, A` (`dst = dst + x²`,
/// each square rounded once to 64 bits); the `n % 8` tail goes into `A`;
/// the result is `sqrt(R + ((C + A) + B))` rounded to double. No `scale`/
/// `ssq`: extended range cannot overflow or underflow on double squares.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dnrm2(n: usize, x: D, incx: usize) -> f64 {
    use x87::{add, load_square, sqrt, to_f64};
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return x.at(0).abs();
    }
    #[cfg(target_arch = "x86_64")]
    if incx == 1 {
        return nrm2_x87_unit(n, x.slice(n).as_ptr());
    }
    let (mut a, mut b, mut c, mut r) = (x87::ZERO, x87::ZERO, x87::ZERO, x87::ZERO);
    let mut p = 0usize;
    for _ in 0..n >> 3 {
        for _ in 0..2 {
            let s0 = load_square(x.at(p));
            let s1 = load_square(x.at(p + incx));
            let s2 = load_square(x.at(p + 2 * incx));
            let s3 = load_square(x.at(p + 3 * incx));
            r = add(r, s3);
            c = add(c, s2);
            b = add(b, s1);
            a = add(a, s0);
            p += 4 * incx;
        }
    }
    for _ in 0..n & 7 {
        a = add(a, load_square(x.at(p)));
        p += incx;
    }
    c = add(c, a);
    c = add(c, b);
    r = add(r, c);
    to_f64(sqrt(r))
}

/// `dnrm2_k_SKYLAKEX`'s unit-stride path (`nrm2.S`) executed on the x87 FPU
/// itself, instruction for instruction: bit-identical to the emulation in
/// [`x87`] (both are the hardware's 64-bit-mantissa arithmetic in the default
/// control word), and far faster. The x87 stack is empty on entry and exit.
#[cfg(target_arch = "x86_64")]
unsafe fn nrm2_x87_unit(n: usize, x: *const f64) -> f64 {
    let mut out = 0.0f64;
    // SAFETY: reads n doubles from x; pushes four x87 slots and pops them all.
    unsafe {
        core::arch::asm!(
            "fldz",
            "fldz",
            "fldz",
            "fldz",
            "mov rax, rdi",
            "sar rax, 3",
            "jle 3f",
            "2:",
            "fld qword ptr [rsi]",
            "fmul st(0), st(0)",
            "fld qword ptr [rsi + 8]",
            "fmul st(0), st(0)",
            "fld qword ptr [rsi + 16]",
            "fmul st(0), st(0)",
            "fld qword ptr [rsi + 24]",
            "fmul st(0), st(0)",
            "faddp st(7), st(0)",
            "faddp st(5), st(0)",
            "faddp st(3), st(0)",
            "faddp st(1), st(0)",
            "fld qword ptr [rsi + 32]",
            "fmul st(0), st(0)",
            "fld qword ptr [rsi + 40]",
            "fmul st(0), st(0)",
            "fld qword ptr [rsi + 48]",
            "fmul st(0), st(0)",
            "fld qword ptr [rsi + 56]",
            "fmul st(0), st(0)",
            "add rsi, 64",
            "faddp st(7), st(0)",
            "faddp st(5), st(0)",
            "faddp st(3), st(0)",
            "faddp st(1), st(0)",
            "dec rax",
            "jg 2b",
            "3:",
            "and rdi, 7",
            "jle 5f",
            "4:",
            "fld qword ptr [rsi]",
            "fmul st(0), st(0)",
            "faddp st(1), st(0)",
            "add rsi, 8",
            "dec rdi",
            "jg 4b",
            "5:",
            "faddp st(2), st(0)",
            "faddp st(1), st(0)",
            "faddp st(1), st(0)",
            "fsqrt",
            "fstp qword ptr [rcx]",
            inout("rdi") n => _,
            inout("rsi") x => _,
            in("rcx") &mut out as *mut f64,
            out("rax") _,
            out("st(0)") _,
            out("st(1)") _,
            out("st(2)") _,
            out("st(3)") _,
            out("st(4)") _,
            out("st(5)") _,
            out("st(6)") _,
            out("st(7)") _,
        );
    }
    out
}

// --------------------------------------------------------- DCOPY/DSWAP ----

/// `DCOPY`. `copy_sse2.S` only moves bits (`movsd`/`movaps`/`movups`), for
/// any stride.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dcopy(n: usize, x: D, incx: usize, y: D, incy: usize) {
    let (mut ix, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        y.set(iy, x.at(ix));
        ix += incx;
        iy += incy;
    }
}

/// `DSWAP`. `swap_sse2.S` only moves bits.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dswap(n: usize, x: D, incx: usize, y: D, incy: usize) {
    let (mut ix, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        let t = x.at(ix);
        x.set(ix, y.at(iy));
        y.set(iy, t);
        ix += incx;
        iy += incy;
    }
}

// ---------------------------------------------------------------- DROT ----

/// `DROT`. Interface passes straight to `drot_k`; `n <= 100000` (or one
/// CPU) runs `rot_compute`. Its AVX-512 kernel (32-blocks, 8-blocks, masked
/// tail) and the `inc != 1` scalar loop compute the same per element:
/// `x' = vfmadd231(c, x, s*y) = c*x + (s*y)` and
/// `y' = vfmsub132(y, c, s*x) = y*c - (s*x)`, each with one rounding and the
/// `s*` products rounded separately.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn drot(n: usize, x: D, incx: usize, y: D, incy: usize, c: f64, s: f64) {
    let (mut ix, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        let xv = x.at(ix);
        let yv = y.at(iy);
        let sy = s * yv;
        let sx = s * xv;
        x.set(ix, c.mul_add(xv, sy));
        y.set(iy, yv.mul_add(c, -sx));
        ix += incx;
        iy += incy;
    }
}

// -------------------------------------------------------------- IDAMAX ----

type P = [f64; 2];

/// `maxsd d, s` lane: `d > s ? d : s` (the source on NaN or equality).
#[inline(always)]
fn mx(d: f64, s: f64) -> f64 {
    if d > s { d } else { s }
}

/// `maxpd s, d`.
#[inline(always)]
fn maxpd(d: P, s: P) -> P {
    [mx(d[0], s[0]), mx(d[1], s[1])]
}

/// `comisd m, v; je`: ZF is set on equality or unordered.
#[inline(always)]
fn comi_eq(v: f64, m: f64) -> bool {
    !(v < m) && !(v > m)
}

/// The register fold `maxpd %xmm1,%xmm0; maxpd %xmm3,%xmm2; maxpd %xmm2,%xmm0;
/// movapd %xmm0,%xmm1; unpckhpd %xmm0,%xmm0; maxsd %xmm1,%xmm0` (without the
/// trailing `unpcklpd`, which only makes the search value `r[0][0]`).
#[inline(always)]
fn fold(r: &mut [P; 4]) {
    r[0] = maxpd(r[0], r[1]);
    r[2] = maxpd(r[2], r[3]);
    r[0] = maxpd(r[0], r[2]);
    r[1] = r[0];
    r[0] = [r[0][1], r[0][1]];
    r[0][0] = mx(r[0][0], r[1][0]);
}

/// `IDAMAX` (1-based result, 0 when `n == 0`), `iamax_sse2.S`.
///
/// Scan: `|x(1)|` seeds all four 2-lane registers; the remaining elements go
/// in pairs to registers 0,1,2,3,0,1,2,3 per 16, then the `& 8`, `& 4`, `& 2`
/// tails to registers 0..3, 0..1, 2 and a final single element to both lanes
/// of register 3. With `incx == 1`, more than 8 elements and `X+8` 8-aligned
/// (the "aligned" path), an element that makes `X` 16-aligned goes first to
/// register 3. The `incx != 1` path folds twice when the single tail element
/// exists (a duplicated fold in the assembly source).
///
/// Search from `x(1)` over all `n`: on the aligned path a 16-misaligned `X`
/// compares `x(1)` alone first; then blocks of 8 are skipped unless one
/// element is ordered-equal to the max (`cmpeqpd`), in which case the first
/// seven are tested with `comisd` (equal *or unordered*) and the eighth is
/// taken; then 4 and 2 element tails with `comisd`; otherwise the next index
/// is returned. The interface clamps the result to `n`. Without NaNs this is
/// the first index of the largest `|x|`; with NaNs it depends on the lane
/// layout and on `X mod 16`, which is read from the view's address.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn idamax(n: usize, x: D, incx: usize) -> usize {
    if n == 0 || incx == 0 {
        return 0;
    }
    let addr = x.slice(0).as_ptr() as usize;
    let at = |k: usize| x.at(k * incx).abs();
    let pair = |k: usize| [at(k), at(k + 1)];

    let v0 = at(0);
    let mut r = [[v0, v0]; 4];
    let mut m = n - 1;
    let mut k = 1usize;
    let aligned = incx == 1 && m > 7 && (addr + 8) & 7 == 0;
    if aligned && (addr + 8) & 15 != 0 {
        let v = at(k);
        r[3] = maxpd(r[3], [v, v]);
        m -= 1;
        k += 1;
    }
    for _ in 0..m >> 4 {
        for _ in 0..2 {
            for reg in r.iter_mut() {
                *reg = maxpd(*reg, pair(k));
                k += 2;
            }
        }
    }
    m &= 15;
    if m != 0 {
        if m & 8 != 0 {
            for reg in r.iter_mut() {
                *reg = maxpd(*reg, pair(k));
                k += 2;
            }
        }
        if m & 4 != 0 {
            for reg in r[..2].iter_mut() {
                *reg = maxpd(*reg, pair(k));
                k += 2;
            }
        }
        if m & 2 != 0 {
            r[2] = maxpd(r[2], pair(k));
            k += 2;
        }
        if m & 1 != 0 {
            let v = at(k);
            r[3] = maxpd(r[3], [v, v]);
            if incx != 1 {
                fold(&mut r);
            }
        }
    }
    fold(&mut r);
    let max = r[0][0];

    let mut ret = 0usize;
    let mut k = 0usize;
    let mut m = n;
    let found = 'search: {
        if aligned && addr & 15 != 0 {
            ret += 1;
            if comi_eq(at(0), max) {
                break 'search ret;
            }
            k += 1;
            m -= 1;
        }
        for _ in 0..m >> 3 {
            if !(0..8).any(|j| at(k + j) == max) {
                ret += 8;
                k += 8;
                continue;
            }
            for j in 0..7 {
                ret += 1;
                if comi_eq(at(k + j), max) {
                    break 'search ret;
                }
            }
            break 'search ret + 1;
        }
        if m & 4 != 0 {
            for j in 0..4 {
                ret += 1;
                if comi_eq(at(k + j), max) {
                    break 'search ret;
                }
            }
            k += 4;
        }
        if m & 2 != 0 {
            for j in 0..2 {
                ret += 1;
                if comi_eq(at(k + j), max) {
                    break 'search ret;
                }
            }
        }
        ret + 1
    };
    found.min(n)
}
