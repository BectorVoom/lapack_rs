//! OpenBLAS 0.3.30 SkylakeX track: complex double level-1 BLAS as
//! `libscipy_openblas-6cdc3b4a.so` computes it (`OPENBLAS_CORETYPE=SkylakeX`,
//! `OPENBLAS_NUM_THREADS=1`), bit for bit.
//!
//! Transcribed (interface -> kernel symbol -> source):
//! * `ZDOTC`:  `interface/zdot.c` (`scipy_zdotc_`) -> `zdotc_k_SKYLAKEX`
//!   (`kernel/x86_64/zdot.c`, `CONJ`) -> `zdot_compute` + `zdot_kernel_8`
//!   (`zdot_microk_haswell-2.c`, AVX2 asm).
//! * `ZAXPY`:  `interface/zaxpy.c` -> `zaxpy_k_SKYLAKEX`
//!   (`kernel/x86_64/zaxpy.c`) -> `zaxpy_kernel_4`
//!   (`zaxpy_microk_haswell-2.c`, AVX2 asm).
//! * `ZSCAL`, `ZDSCAL`: `interface/zscal.c` (plain and `SSCAL`) ->
//!   `zscal_k_SKYLAKEX` (`kernel/x86_64/zscal.c`, `fp-contract=off`, called
//!   with `dummy2 = 1`) -> `zscal_kernel_8` (`zscal_microk_skylakex-2.c`,
//!   AVX-512) and `zscal_kernel_inc_8`.
//! * `DZNRM2`: `interface/nrm2.c` (`scipy_dznrm2_`) -> `znrm2_k_SKYLAKEX`
//!   (`kernel/x86_64/znrm2.S`, x87 80-bit extended precision).
//! * `ZCOPY`:  `interface/zcopy.c` -> `zcopy_k_SKYLAKEX` (`zcopy_sse2.S`).
//! * `ZSWAP`:  `interface/zswap.c` -> `zswap_k_SKYLAKEX` (`zswap_sse2.S`).
//!
//! FMA: `zdot_kernel_8`, `zdot_compute` and `zaxpy_kernel_4` / the `zaxpy_k`
//! tails are fused (`vfmadd231`, `vfmsub231`); `zscal_k` is built with
//! `fp-contract=off` and has no fused operation anywhere. `znrm2.S` is x87
//! and is emulated here with an exact 64-bit-mantissa soft float
//! ([`Ext`]); the process x87 control word must be the Linux default
//! (extended precision, round to nearest), which the oracle test checks.
//! `zcopy`/`zswap` only move bits (`movaps`/`movsd`, no arithmetic).
use crate::zblas::{Z, ZV};

/// The C `NAN` the kernels store (`.rodata` `0x7ff8000000000000`).
const C_NAN: f64 = f64::from_bits(0x7ff8_0000_0000_0000);

// NaN propagation. Only NaN payloads/signs depend on it, but they are output
// bits too, and LLVM may commute the operands of a plain `a * b`. x86 SSE/AVX
// binary ops return src1's NaN, else src2's; FMA (`vfmadd*`, `vfmsub*`)
// returns the first NaN among (vvvv operand, other multiplicand, addend), the
// probed order on this machine, and never negates a NaN addend of `vfmsub`.
// Without NaN inputs every result, including a generated default NaN
// (`0xfff8000000000000`), is the plain IEEE operation.

/// `vmulsd src2, src1`: `src1 * src2`.
#[inline(always)]
fn mul(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else {
        a * b
    }
}
/// `vaddsd src2, src1`: `src1 + src2`.
#[inline(always)]
fn add(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else {
        a + b
    }
}
/// `vsubsd src2, src1`: `src1 - src2`.
#[inline(always)]
fn sub(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else {
        a - b
    }
}
/// `p1*p2 + c` rounded once (`p1` the vvvv operand of the FMA instruction).
#[inline(always)]
fn fmadd(p1: f64, p2: f64, c: f64) -> f64 {
    if p1.is_nan() {
        p1
    } else if p2.is_nan() {
        p2
    } else if c.is_nan() {
        c
    } else {
        p1.mul_add(p2, c)
    }
}
/// `p1*p2 - c` rounded once (`vfmsub`).
#[inline(always)]
fn fmsub(p1: f64, p2: f64, c: f64) -> f64 {
    if p1.is_nan() {
        p1
    } else if p2.is_nan() {
        p2
    } else if c.is_nan() {
        c
    } else {
        p1.mul_add(p2, -c)
    }
}
/// `vmulpd` of a `±1.0` mask (src1) by `v`: `-v`, but a NaN `v` keeps its sign.
#[inline(always)]
fn neg_mask(v: f64) -> f64 {
    if v.is_nan() { v } else { -v }
}

// ---------------------------------------------------------------- ZDOTC --

/// Double `p` of interleaved COMPLEX*16 storage (`re, im` per element).
#[inline(always)]
unsafe fn il(v: ZV, p: usize) -> f64 {
    if p & 1 == 0 { v.re.at(p >> 1) } else { v.im.at(p >> 1) }
}

/// `zdot_kernel_8` (`zdot_microk_haswell-2.c`, `n` a positive multiple of
/// 8): four ymm accumulators per product class, 8 complex elements per step.
/// Returns `dot[0..4]` = (Σ xr·yr, Σ xi·yi, Σ xr·yi, Σ xi·yr).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn zdot_kernel_8(n: usize, x: ZV, y: ZV) -> [f64; 4] {
    // a[k][l]: ymm0..3 += x[j+4k+l]·y[j+4k+l]; b[k][l]: ymm4..7 with y
    // permuted by vpermilpd $5 (lanes 1,0,3,2).
    let mut a = [[0.0f64; 4]; 4];
    let mut b = [[0.0f64; 4]; 4];
    const PERM: [usize; 4] = [1, 0, 3, 2];
    let mut j = 0;
    while j < 2 * n {
        for k in 0..4 {
            for l in 0..4 {
                let p = j + 4 * k + l;
                let xp = il(x, p);
                a[k][l] = fmadd(il(y, p), xp, a[k][l]);
                b[k][l] = fmadd(il(y, j + 4 * k + PERM[l]), xp, b[k][l]);
            }
        }
        j += 16;
    }
    // (A3 + A2) + (A1 + A0), then lanes (2 + 0), (3 + 1) (src1 first).
    let fold = |v: &[[f64; 4]; 4]| {
        let mut s = [0.0f64; 4];
        for l in 0..4 {
            s[l] = add(add(v[3][l], v[2][l]), add(v[1][l], v[0][l]));
        }
        [add(s[2], s[0]), add(s[3], s[1])]
    };
    let da = fold(&a);
    let db = fold(&b);
    [da[0], da[1], db[0], db[1]]
}

/// `ZDOTC` through `scipy_zdotc_` (positive increments).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zdotc(n: usize, x: ZV, incx: usize, y: ZV, incy: usize) -> Z {
    if n == 0 {
        return Z::new(0.0, 0.0);
    }
    // `dot[k]` in the order (xr·yr, xi·yi, xr·yi, xi·yr).
    let mut dot = [0.0f64; 4];
    if incx == 1 && incy == 1 {
        let n1 = n & !7;
        if n1 != 0 {
            dot = zdot_kernel_8(n1, x, y);
        }
        for i in n1..n {
            let (xr, xi, yr, yi) = (x.re.at(i), x.im.at(i), y.re.at(i), y.im.at(i));
            dot[0] = fmadd(xr, yr, dot[0]);
            dot[1] = fmadd(xi, yi, dot[1]);
            dot[2] = fmadd(xr, yi, dot[2]);
            dot[3] = fmadd(yr, xi, dot[3]);
        }
    } else {
        let (mut ix, mut iy) = (0usize, 0usize);
        for _ in 0..n {
            let (xr, xi, yr, yi) = (x.re.at(ix), x.im.at(ix), y.re.at(iy), y.im.at(iy));
            dot[0] = fmadd(xr, yr, dot[0]);
            dot[1] = fmadd(xi, yi, dot[1]);
            dot[2] = fmadd(xr, yi, dot[2]);
            dot[3] = fmadd(yr, xi, dot[3]);
            ix += incx;
            iy += incy;
        }
    }
    // `OPENBLAS_MAKE_COMPLEX_FLOAT(dot[0]+dot[1], dot[2]-dot[3])` is
    // `re + im*I`; gcc lowers the real part to `im*0.0 + re`, fused.
    let im = sub(dot[2], dot[3]);
    let re = fmadd(im, 0.0, add(dot[0], dot[1]));
    Z::new(re, im)
}

// ---------------------------------------------------------------- ZAXPY --

/// `ZAXPY` through `scipy_zaxpy_` (positive increments).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zaxpy(n: usize, za: Z, x: ZV, incx: usize, y: ZV, incy: usize) {
    if n == 0 {
        return;
    }
    if za.re == 0.0 && za.im == 0.0 {
        return;
    }
    let (da_r, da_i) = (za.re, za.im);
    if incx == 1 && incy == 1 {
        let n1 = n & !15;
        // zaxpy_kernel_4: mvec = {-1, 1, -1, 1} scales da_i;
        // y_re = x_im·(-da_i) + (x_re·da_r + y_re)   (vfmadd213 then 231)
        // y_im = x_re·( da_i) + (x_im·da_r + y_im)
        let ndi = neg_mask(da_i);
        for i in 0..n1 {
            let (xr, xi) = (x.re.at(i), x.im.at(i));
            y.re.set(i, fmadd(xi, ndi, fmadd(da_r, xr, y.re.at(i))));
            y.im.set(i, fmadd(xr, da_i, fmadd(da_r, xi, y.im.at(i))));
        }
        for i in n1..n {
            let (xr, xi) = (x.re.at(i), x.im.at(i));
            // vmulsd; vfmsub231sd (da_r·x_re - t); vaddsd y.
            y.re.set(i, add(fmsub(da_r, xr, mul(da_i, xi)), y.re.at(i)));
            y.im.set(i, add(fmadd(da_r, xi, mul(da_i, xr)), y.im.at(i)));
        }
        return;
    }
    let (mut ix, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        let (xr, xi) = (x.re.at(ix), x.im.at(ix));
        y.re.set(iy, add(fmsub(da_r, xr, mul(da_i, xi)), y.re.at(iy)));
        y.im.set(iy, add(fmadd(da_r, xi, mul(da_i, xr)), y.im.at(iy)));
        ix += incx;
        iy += incy;
    }
}

// ---------------------------------------------------------------- ZSCAL --

/// `zscal_k_SKYLAKEX(n, 0, 0, da_r, da_i, x, inc_x, NULL, 0, NULL, 1)`:
/// `kernel/x86_64/zscal.c` with `dummy2 = 1` (both interfaces pass 1).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn zscal_k(n: usize, da_r: f64, da_i: f64, x: ZV, inc_x: usize) {
    if inc_x != 1 {
        if da_r == 0.0 {
            if da_i == 0.0 {
                // `temp = NAN` if either part is NaN, else 0.0.
                let mut i = 0;
                for _ in 0..n {
                    let t = if x.re.at(i).is_nan() || x.im.at(i).is_nan() { C_NAN } else { 0.0 };
                    x.re.set(i, t);
                    x.im.set(i, t);
                    i += inc_x;
                }
            } else {
                let ndi = -da_i;
                let mut i = 0;
                for _ in 0..n {
                    let (xr, xi) = (x.re.at(i), x.im.at(i));
                    let t0 = if xr.is_nan() || xr.is_infinite() { C_NAN } else { mul(xi, ndi) };
                    let t1 = if xi.is_infinite() { C_NAN } else { mul(xr, da_i) };
                    x.im.set(i, t1);
                    x.re.set(i, t0);
                    i += inc_x;
                }
            }
        } else if da_i == 0.0 {
            let mut i = 0;
            for _ in 0..n {
                let (xr, xi) = (x.re.at(i), x.im.at(i));
                x.re.set(i, mul(da_r, xr));
                x.im.set(i, mul(da_r, xi));
                i += inc_x;
            }
        } else {
            let n1 = n & !7;
            let mut i = 0;
            // zscal_kernel_inc_8 (no FMA): x_im = xr·da_i + xi·da_r. gcc
            // emitted element 0 of each block of 4 with the x operand as
            // src1 of the products and elements 1..3 with alpha as src1.
            for k in 0..n1 {
                let (xr, xi) = (x.re.at(i), x.im.at(i));
                let (t0, t1) = if k % 4 == 0 {
                    (sub(mul(xr, da_r), mul(xi, da_i)), add(mul(xr, da_i), mul(xi, da_r)))
                } else {
                    (sub(mul(da_r, xr), mul(da_i, xi)), add(mul(da_i, xr), mul(da_r, xi)))
                };
                x.im.set(i, t1);
                x.re.set(i, t0);
                i += inc_x;
            }
            // Tail: x_im = xi·da_r + xr·da_i.
            for _ in n1..n {
                let (xr, xi) = (x.re.at(i), x.im.at(i));
                let t0 = sub(mul(xr, da_r), mul(xi, da_i));
                x.im.set(i, add(mul(xi, da_r), mul(xr, da_i)));
                x.re.set(i, t0);
                i += inc_x;
            }
        }
        return;
    }

    let n1 = n & !7;
    // With dummy2 = 1 every alpha takes zscal_kernel_8 (AVX-512):
    // lanes (x_re, x_im) = ((-1·da_i)·xi + da_r·xr, (1·da_i)·xr + da_r·xi).
    let mdi = neg_mask(da_i);
    for i in 0..n1 {
        let (xr, xi) = (x.re.at(i), x.im.at(i));
        x.re.set(i, add(mul(mdi, xi), mul(da_r, xr)));
        x.im.set(i, add(mul(da_i, xr), mul(da_r, xi)));
    }
    if n1 == n {
        return;
    }
    if da_r == 0.0 || da_r.is_nan() {
        if da_i == 0.0 {
            // `res` is decided once, from the first tail element.
            let mut res = if da_r.is_nan() { da_r } else { 0.0 };
            if x.re.at(n1).is_nan() || x.im.at(n1).is_nan() {
                res = C_NAN;
            }
            for i in n1..n {
                x.re.set(i, res);
                x.im.set(i, res);
            }
        } else if da_r.is_infinite() {
            // Unreachable (da_r is 0 or NaN here); kept as the source has it.
            for i in n1..n {
                x.re.set(i, C_NAN);
                x.im.set(i, da_r);
            }
        } else {
            let ndi = -da_i;
            for i in n1..n {
                let (xr, xi) = (x.re.at(i), x.im.at(i));
                let t0 = if xr.is_infinite() { C_NAN } else { mul(xi, ndi) };
                x.im.set(i, if xi.is_infinite() { C_NAN } else { mul(xr, da_i) });
                if !xr.is_nan() {
                    x.re.set(i, t0);
                }
            }
        }
    } else if da_i == 0.0 {
        for i in n1..n {
            let (xr, xi) = (x.re.at(i), x.im.at(i));
            let t0 = mul(xr, da_r);
            x.im.set(i, if xr.is_nan() { C_NAN } else { mul(da_r, xi) });
            x.re.set(i, t0);
        }
    } else {
        for i in n1..n {
            let (xr, xi) = (x.re.at(i), x.im.at(i));
            x.im.set(i, add(mul(xi, da_r), mul(xr, da_i)));
            if !xr.is_nan() {
                x.re.set(i, sub(mul(xr, da_r), mul(xi, da_i)));
            }
        }
    }
}

/// `ZSCAL` through `scipy_zscal_` (positive increments).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zscal(n: usize, za: Z, x: ZV, incx: usize) {
    if incx == 0 || n == 0 {
        return;
    }
    if za.re == 1.0 && za.im == 0.0 {
        return;
    }
    zscal_k(n, za.re, za.im, x, incx);
}

/// `ZDSCAL` through `scipy_zdscal_` (`zscal.c` with `SSCAL`: alpha = (da, 0)).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zdscal(n: usize, da: f64, x: ZV, incx: usize) {
    if incx == 0 || n == 0 || da == 1.0 {
        return;
    }
    zscal_k(n, da, 0.0, x, incx);
}

// --------------------------------------------------------------- DZNRM2 --

/// Non-negative x87 extended-precision value: finite `m · 2^e` with `m`
/// normalised (bit 63 set) or zero, `+Inf`, or a NaN (the double's bits,
/// which the x87 round trip `fldl`/`fstpl` preserves for quiet NaNs).
///
/// Sums of squares of doubles stay in `[2^-2148, 2^2048·n]`, far inside the
/// extended exponent range, so the exponent is unbounded here and only the
/// 64-bit significand rounding (to nearest, ties to even) is emulated.
#[derive(Clone, Copy, Debug)]
enum Ext {
    Zero,
    Fin { m: u64, e: i32 },
    Inf,
    Nan(u64),
}

/// Round `m · 2^e` (`m != 0`, plus a sticky bit below it) to 64 bits.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
fn round64(m: u128, e: i32, sticky: bool) -> Ext {
    let lz = m.leading_zeros() as i32;
    let m = m << lz;
    let e = e - lz;
    let mut hi = (m >> 64) as u64;
    let lo = m as u64;
    let round = lo >> 63 != 0;
    let st = (lo << 1) != 0 || sticky;
    let mut e = e + 64;
    if round && (st || hi & 1 != 0) {
        let (h, c) = hi.overflowing_add(1);
        hi = h;
        if c {
            hi = 1 << 63;
            e += 1;
        }
    }
    Ext::Fin { m: hi, e }
}

impl Ext {
    /// `fldl x; fmul %st(0),%st`.
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    fn square(x: f64) -> Ext {
        if x.is_nan() {
            return Ext::Nan(x.to_bits());
        }
        if x.is_infinite() {
            return Ext::Inf;
        }
        if x == 0.0 {
            return Ext::Zero;
        }
        let bits = x.to_bits();
        let be = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & ((1u64 << 52) - 1);
        let (mant, ex) = if be == 0 { (frac, -1074) } else { (frac | (1 << 52), be - 1075) };
        round64(mant as u128 * mant as u128, 2 * ex, false)
    }

    /// `faddp`: both operands non-negative.
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    fn add(self, other: Ext) -> Ext {
        match (self, other) {
            (Ext::Nan(a), Ext::Nan(b)) => {
                // Two QNaNs: the larger significand; on a tie the positive
                // one (probed with `faddp` on this machine).
                let (sa, sb) = (a & ((1 << 52) - 1), b & ((1 << 52) - 1));
                Ext::Nan(if sb > sa || (sb == sa && b >> 63 == 0) { b } else { a })
            }
            (Ext::Nan(a), _) | (_, Ext::Nan(a)) => Ext::Nan(a),
            (Ext::Inf, _) | (_, Ext::Inf) => Ext::Inf,
            (Ext::Zero, v) | (v, Ext::Zero) => v,
            (Ext::Fin { m: ma, e: ea }, Ext::Fin { m: mb, e: eb }) => {
                let ((ma, ea), (mb, eb)) = if ea >= eb { ((ma, ea), (mb, eb)) } else { ((mb, eb), (ma, ea)) };
                let d = (ea - eb) as u32;
                let a = (ma as u128) << 63;
                let b0 = (mb as u128) << 63;
                let (b, sticky) = if d >= 127 { (0, true) } else { (b0 >> d, (b0 & ((1u128 << d) - 1)) != 0) };
                round64(a + b, ea - 63, sticky)
            }
        }
    }

    /// `fsqrt` (correctly rounded to 64 bits).
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    fn sqrt(self) -> Ext {
        match self {
            Ext::Fin { m, e } => {
                // N = m · 2^s with e - s even; floor(sqrt(N)) has >= 66 bits.
                let s: i32 = if (e - 68) % 2 == 0 { 68 } else { 67 };
                let half = (e - s) / 2;
                let bit = |p: i32| -> u128 { if p >= s && p - s < 64 { ((m >> (p - s)) & 1) as u128 } else { 0 } };
                let (mut r, mut rem) = (0u128, 0u128);
                for i in (0..66).rev() {
                    rem = (rem << 2) | (bit(2 * i + 1) << 1) | bit(2 * i);
                    let trial = (r << 2) | 1;
                    if rem >= trial {
                        rem -= trial;
                        r = (r << 1) | 1;
                    } else {
                        r <<= 1;
                    }
                }
                round64(r, half, rem != 0)
            }
            v => v,
        }
    }

    /// `fstpl` (round to nearest double).
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
    fn to_f64(self) -> f64 {
        match self {
            Ext::Zero => 0.0,
            Ext::Inf => f64::INFINITY,
            Ext::Nan(b) => f64::from_bits(b),
            Ext::Fin { m, e } => {
                // Value in [2^(e+63), 2^(e+64)).
                let top = e + 63;
                let lsb = (top - 52).max(-1074);
                let drop = lsb - e; // >= 11
                if drop >= 66 {
                    return 0.0;
                }
                let m128 = m as u128;
                let mut q = m128 >> drop;
                let round = (m128 >> (drop - 1)) & 1 != 0;
                let sticky = m128 & ((1u128 << (drop - 1)) - 1) != 0;
                if round && (sticky || q & 1 != 0) {
                    q += 1;
                }
                let mut q = q as u64;
                let mut lsb = lsb;
                if q >= 1 << 53 {
                    q >>= 1;
                    lsb += 1;
                }
                if q >= 1 << 52 {
                    let be = lsb + 52 + 1023;
                    if be >= 2047 {
                        return f64::INFINITY;
                    }
                    f64::from_bits(((be as u64) << 52) | (q & ((1 << 52) - 1)))
                } else {
                    f64::from_bits(q)
                }
            }
        }
    }
}

/// `DZNRM2` through `scipy_dznrm2_` -> `znrm2_k_SKYLAKEX` (x87, positive
/// increments): four extended accumulators a, b (real/imaginary parts of
/// elements 0, 2 mod 4 in the 4-element blocks), c, d (elements 1, 3 mod 4),
/// the `n & 3` tail into a, b, then `sqrt(d + ((c + a) + b))`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dznrm2(n: usize, x: ZV, incx: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    if incx == 1 {
        return znrm2_x87_unit(n, x.re.slice(n).as_ptr(), x.im.slice(n).as_ptr());
    }
    let (mut a, mut b, mut c, mut d) = (Ext::Zero, Ext::Zero, Ext::Zero, Ext::Zero);
    let mut i = 0usize;
    for _ in 0..n >> 2 {
        for _ in 0..2 {
            let s0 = Ext::square(x.re.at(i));
            let s1 = Ext::square(x.im.at(i));
            let s2 = Ext::square(x.re.at(i + incx));
            let s3 = Ext::square(x.im.at(i + incx));
            d = d.add(s3);
            c = c.add(s2);
            b = b.add(s1);
            a = a.add(s0);
            i += 2 * incx;
        }
    }
    for _ in 0..n & 3 {
        let s0 = Ext::square(x.re.at(i));
        let s1 = Ext::square(x.im.at(i));
        b = b.add(s1);
        a = a.add(s0);
        i += incx;
    }
    let c = c.add(a);
    let c = c.add(b);
    let d = d.add(c);
    d.sqrt().to_f64()
}

/// `znrm2_k_SKYLAKEX`'s unit-stride path (`znrm2.S`) on the x87 FPU,
/// instruction for instruction, reading the planar parts where the kernel
/// reads the interleaved pairs: bit-identical to the [`Ext`] emulation (both
/// are the hardware's extended arithmetic), and far faster.
#[cfg(target_arch = "x86_64")]
unsafe fn znrm2_x87_unit(n: usize, re: *const f64, im: *const f64) -> f64 {
    let mut out = 0.0f64;
    // SAFETY: reads n doubles from each of re, im; pushes four x87 slots and
    // pops them all.
    unsafe {
        core::arch::asm!(
            "fldz",
            "fldz",
            "fldz",
            "fldz",
            "mov rax, rdi",
            "sar rax, 2",
            "jle 3f",
            "2:",
            "fld qword ptr [rsi]",
            "fmul st(0), st(0)",
            "fld qword ptr [rdx]",
            "fmul st(0), st(0)",
            "fld qword ptr [rsi + 8]",
            "fmul st(0), st(0)",
            "fld qword ptr [rdx + 8]",
            "fmul st(0), st(0)",
            "faddp st(7), st(0)",
            "faddp st(5), st(0)",
            "faddp st(3), st(0)",
            "faddp st(1), st(0)",
            "fld qword ptr [rsi + 16]",
            "fmul st(0), st(0)",
            "fld qword ptr [rdx + 16]",
            "fmul st(0), st(0)",
            "fld qword ptr [rsi + 24]",
            "fmul st(0), st(0)",
            "fld qword ptr [rdx + 24]",
            "fmul st(0), st(0)",
            "add rsi, 32",
            "add rdx, 32",
            "faddp st(7), st(0)",
            "faddp st(5), st(0)",
            "faddp st(3), st(0)",
            "faddp st(1), st(0)",
            "dec rax",
            "jg 2b",
            "3:",
            "and rdi, 3",
            "jle 5f",
            "4:",
            "fld qword ptr [rsi]",
            "fmul st(0), st(0)",
            "fld qword ptr [rdx]",
            "fmul st(0), st(0)",
            "faddp st(3), st(0)",
            "faddp st(1), st(0)",
            "add rsi, 8",
            "add rdx, 8",
            "dec rdi",
            "jg 4b",
            "5:",
            "faddp st(2), st(0)",
            "faddp st(1), st(0)",
            "faddp st(1), st(0)",
            "fsqrt",
            "fstp qword ptr [rcx]",
            inout("rdi") n => _,
            inout("rsi") re => _,
            inout("rdx") im => _,
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

// ---------------------------------------------------------- ZCOPY/ZSWAP --

/// `ZCOPY` through `scipy_zcopy_`: bit moves only.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zcopy(n: usize, x: ZV, incx: usize, y: ZV, incy: usize) {
    let (mut ix, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        y.re.set(iy, x.re.at(ix));
        y.im.set(iy, x.im.at(ix));
        ix += incx;
        iy += incy;
    }
}

/// `ZSWAP` through `scipy_zswap_`: bit moves only.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zswap(n: usize, x: ZV, incx: usize, y: ZV, incy: usize) {
    let (mut ix, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        let (tr, ti) = (x.re.at(ix), x.im.at(ix));
        x.re.set(ix, y.re.at(iy));
        x.im.set(ix, y.im.at(iy));
        y.re.set(iy, tr);
        y.im.set(iy, ti);
        ix += incx;
        iy += incy;
    }
}
