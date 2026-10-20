//! OpenBLAS 0.3.30 SkylakeX track (scipy's `libscipy_openblas-6cdc3b4a.so`,
//! `OPENBLAS_CORETYPE=SkylakeX OPENBLAS_NUM_THREADS=1`): complex double level 2.
//!
//! Transcribed, bit for bit, from the binary and these sources:
//! * `interface/zhemv.c`, `interface/zgemv.c`, `interface/zher2.c`,
//!   `interface/zger.c` (Fortran entry points `scipy_z{hemv,gemv,her2,gerc}_`).
//! * `kernel/x86_64/zsymv_U_sse2.S`, `zsymv_L_sse2.S` built with `-DHEMV`
//!   (`zhemv_U_SKYLAKEX`, `zhemv_L_SKYLAKEX`): legacy SSE2 assembly, no FMA.
//! * `kernel/x86_64/zgemv_n_4.c` + `zgemv_n_microk_haswell-4.c`
//!   (`zgemv_n_SKYLAKEX`, `zgemv_kernel_4x4/4x2/4x1`, `add_y`).
//! * `kernel/x86_64/zgemv_t_4.c` + `zgemv_t_microk_haswell-4.c`
//!   (`zgemv_t_SKYLAKEX`, `zgemv_c_SKYLAKEX` = `-DTRANS -DCONJ`).
//! * `driver/level2/zher2_k.c` (`zher2_U`, `zher2_L`: core-independent,
//!   plain SSE2, no FMA) calling `zaxpy_k_SKYLAKEX`.
//! * `kernel/generic/zger.c` with `-DCONJ` (`zgerc_k_SKYLAKEX`) calling
//!   `zaxpy_k_SKYLAKEX`.
//! * `kernel/x86_64/zaxpy.c` + `zaxpy_microk_haswell-2.c` (`zaxpy_k_SKYLAKEX`,
//!   `zaxpy_kernel_4`).
//! * `kernel/x86_64/zscal.c` + `zscal_microk_skylakex-2.c` (`zscal_k_SKYLAKEX`,
//!   `zscal_kernel_8` AVX-512 branch): the interfaces' `beta` pre-scaling.
//!
//! Arithmetic notation: `vfmadd` is `a.mul_add(b, c)`, `vfmsub` is
//! `a.mul_add(b, -c)`, `vfnmadd` is `(-a).mul_add(b, c)`. Every other `a*b + c`
//! below is a separate `vmul`/`mul` followed by `vadd`/`add` in the binary.
//!
//! `vaddsubpd` (opcode `66 0F D0`, used by the haswell microkernels) computes
//! `src1 - src2` on even lanes and `src1 + src2` on odd lanes.
//!
//! `zgemv` and `zgerc` accept any positive increments (non-unit `incx`
//! changes the leftover-column grouping of `zgemv_n`; non-unit `incy` selects
//! the scalar `add_y`, the `ybuffer` path of `zgemv_t` and the `inc_x != 1`
//! branch of `zscal.c`); `zhemv`/`zher2` are unit-increment as in `zblas.rs`.
// `x * -1.0` is kept where the binary multiplies (`mulsd`), not negates.
#![allow(clippy::neg_multiply)]

use crate::ptr::D;
use crate::zblas::{Tr, Z, ZV};

/// `NBMAX` of `zgemv_n_4.c` / `zgemv_t_4.c`: rows per kernel block.
const NBMAX: usize = 1024;

// ---------------------------------------------------------------------------
// zscal_k_SKYLAKEX (unit increment, dummy2 = 0 as the level-2 interfaces pass).
// ---------------------------------------------------------------------------

/// `ZSCAL_K(n, 0, 0, da_r, da_i, x, 1, NULL, 0, NULL, 0)`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn zscal_k(n: usize, da_r: f64, da_i: f64, x: ZV) {
    let n1 = n & !7;
    if n1 > 0 {
        if da_r == 0.0 && da_i == 0.0 {
            // zscal_kernel_8_zero
            for k in 0..n1 {
                x.re.set(k, 0.0);
                x.im.set(k, 0.0);
            }
        } else {
            // zscal_kernel_8, AVX-512: da_i * (-1, 1, ...) times the swapped
            // pair, plus da_r * x; vmulpd and vaddpd, no FMA.
            let mi = da_i * -1.0;
            let pi = da_i * 1.0;
            for k in 0..n1 {
                let xr = x.re.at(k);
                let xi = x.im.at(k);
                x.re.set(k, (mi * xi) + (da_r * xr));
                x.im.set(k, (pi * xr) + (da_r * xi));
            }
        }
    }
    if da_r == 0.0 || da_r.is_nan() {
        if da_i == 0.0 {
            let res = if da_r.is_nan() { da_r } else { 0.0 };
            for k in n1..n {
                x.re.set(k, res);
                x.im.set(k, res);
            }
        } else if da_r.is_infinite() {
            for k in n1..n {
                x.re.set(k, f64::NAN);
                x.im.set(k, da_r);
            }
        } else {
            for k in n1..n {
                let xr = x.re.at(k);
                let xi = x.im.at(k);
                let mut t0 = (-da_i) * xi;
                if xr.is_infinite() {
                    t0 = f64::NAN;
                }
                if !xi.is_infinite() {
                    x.im.set(k, da_i * xr);
                } else {
                    x.im.set(k, f64::NAN);
                }
                if !xr.is_nan() {
                    x.re.set(k, t0);
                }
            }
        }
    } else if da_i == 0.0 {
        for k in n1..n {
            let xr = x.re.at(k);
            let t0 = da_r * xr;
            if xr.is_nan() {
                x.im.set(k, f64::NAN);
            } else {
                x.im.set(k, da_r * x.im.at(k));
            }
            x.re.set(k, t0);
        }
    } else {
        for k in n1..n {
            let xr = x.re.at(k);
            let xi = x.im.at(k);
            x.im.set(k, (da_r * xi) + (da_i * xr));
            if !xr.is_nan() {
                x.re.set(k, (da_r * xr) - (da_i * xi));
            }
        }
    }
}

/// `ZSCAL_K(n, 0, 0, da_r, da_i, x, incx, NULL, 0, NULL, 0)` with `incx > 1`
/// (the `inc_x != 1` branch of `zscal.c`; `zscal_kernel_inc_8` and the scalar
/// tail are both plain `vmulsd`/`vsubsd`/`vaddsd`, no FMA, no NaN guard).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn zscal_k_inc(n: usize, da_r: f64, da_i: f64, x: ZV, incx: usize) {
    if da_r == 0.0 {
        if da_i == 0.0 {
            for k in 0..n {
                x.re.set(k * incx, 0.0);
                x.im.set(k * incx, 0.0);
            }
        } else {
            for k in 0..n {
                let p = k * incx;
                let (xr, xi) = (x.re.at(p), x.im.at(p));
                let t0 = if xr.is_nan() || xr.is_infinite() { f64::NAN } else { (-da_i) * xi };
                x.im.set(p, if !xi.is_infinite() { da_i * xr } else { f64::NAN });
                x.re.set(p, t0);
            }
        }
        return;
    }
    // da_i == 0 with dummy2 == 0 takes this general path too.
    for k in 0..n {
        let p = k * incx;
        let (xr, xi) = (x.re.at(p), x.im.at(p));
        x.re.set(p, (da_r * xr) - (da_i * xi));
        x.im.set(p, (da_i * xr) + (da_r * xi));
    }
}

// ---------------------------------------------------------------------------
// zaxpy_k_SKYLAKEX (unit increments).
// ---------------------------------------------------------------------------

/// `ZAXPYU_K(n, 0, 0, da_r, da_i, x, 1, y, 1, NULL, 0)`: `y += da * x`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn zaxpy_k(n: usize, da_r: f64, da_i: f64, x: ZV, y: ZV) {
    let n1 = n & !15;
    // zaxpy_kernel_4: vfmadd213pd (da_r*x + y), then vfmadd231pd of the
    // swapped pair times (-da_i, da_i).
    let mi = da_i * -1.0;
    let pi = da_i * 1.0;
    for k in 0..n1 {
        let xr = x.re.at(k);
        let xi = x.im.at(k);
        let yr = da_r.mul_add(xr, y.re.at(k));
        let yi = da_r.mul_add(xi, y.im.at(k));
        y.re.set(k, xi.mul_add(mi, yr));
        y.im.set(k, xr.mul_add(pi, yi));
    }
    // Scalar tail: vfmsub231sd / vfmadd231sd, then vaddsd.
    for k in n1..n {
        let xr = x.re.at(k);
        let xi = x.im.at(k);
        y.re.set(k, da_r.mul_add(xr, -(da_i * xi)) + y.re.at(k));
        y.im.set(k, da_r.mul_add(xi, da_i * xr) + y.im.at(k));
    }
}

// ---------------------------------------------------------------------------
// ZHEMV
// ---------------------------------------------------------------------------

#[inline(always)]
fn add2(p: [f64; 2], q: [f64; 2]) -> [f64; 2] {
    [p[0] + q[0], p[1] + q[1]]
}

#[inline(always)]
fn sub2(p: [f64; 2], q: [f64; 2]) -> [f64; 2] {
    [p[0] - q[0], p[1] - q[1]]
}

/// `mulpd` of a `movddup`ed scalar with a pair.
#[inline(always)]
fn mul2(s: f64, p: [f64; 2]) -> [f64; 2] {
    [s * p[0], s * p[1]]
}

#[inline(always)]
unsafe fn zget(v: ZV, i: usize) -> [f64; 2] {
    [v.re.at(i), v.im.at(i)]
}

#[inline(always)]
unsafe fn zput(v: ZV, i: usize, p: [f64; 2]) {
    v.re.set(i, p[0]);
    v.im.set(i, p[1]);
}

/// The `NEW_X` buffer of `zsymv_[UL]_sse2.S`: `T1 = alpha*x` (as
/// `x_r*(ar, ai) + x_i*(-ai, ar)`) and `T2 = (-T1_i, T1_r)` (`i*T1`).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn hemv_buffer(n: usize, alpha: Z, x: ZV) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let mut t1 = Vec::with_capacity(n);
    let mut t2 = Vec::with_capacity(n);
    let ai_neg = -alpha.im;
    for k in 0..n {
        let xr = x.re.at(k);
        let xi = x.im.at(k);
        let t = [(xr * alpha.re) + (xi * ai_neg), (xr * alpha.im) + (xi * alpha.re)];
        t1.push(t);
        t2.push([-t[1], t[0]]);
    }
    (t1, t2)
}

/// One pair of rows `(b, b + 1)` against the column pair `(j, j + 1)`: the
/// body shared by `.L12`, `.L15` of both `zsymv_U_sse2.S` and
/// `zsymv_L_sse2.S` (`ADD` = `subpd` under `HEMV`).
#[inline(always)]
unsafe fn hemv_pair_rows(
    b: usize,
    c1: ZV,
    c2: ZV,
    at: [[f64; 2]; 4],
    t1: &[[f64; 2]],
    t2: &[[f64; 2]],
    xs1: &mut [f64; 2],
    xs2: &mut [f64; 2],
    y: ZV,
) {
    let (r0, r1) = (b, b + 1);
    let (a1r0, a1i0, a1r1, a1i1) = (c1.re.at(r0), c1.im.at(r0), c1.re.at(r1), c1.im.at(r1));
    let (a2r0, a2i0, a2r1, a2i1) = (c2.re.at(r0), c2.im.at(r0), c2.re.at(r1), c2.im.at(r1));
    // Column j: rows r0, r1 in order, real term then imaginary term.
    *xs1 = add2(*xs1, mul2(a1r0, t1[r0]));
    *xs1 = sub2(*xs1, mul2(a1i0, t2[r0]));
    *xs1 = add2(*xs1, mul2(a1r1, t1[r1]));
    *xs1 = sub2(*xs1, mul2(a1i1, t2[r1]));
    // Column j + 1: row r1 first, then r0.
    *xs2 = add2(*xs2, mul2(a2r1, t1[r1]));
    *xs2 = sub2(*xs2, mul2(a2i1, t2[r1]));
    *xs2 = add2(*xs2, mul2(a2r0, t1[r0]));
    *xs2 = sub2(*xs2, mul2(a2i0, t2[r0]));
    // y(r0): column j, then column j + 1.
    let mut yy = zget(y, r0);
    yy = add2(yy, mul2(a1r0, at[0]));
    yy = add2(yy, mul2(a1i0, at[1]));
    yy = add2(yy, mul2(a2r0, at[2]));
    yy = add2(yy, mul2(a2i0, at[3]));
    zput(y, r0, yy);
    // y(r1): column j + 1, then column j.
    let mut yy = zget(y, r1);
    yy = add2(yy, mul2(a2r1, at[2]));
    yy = add2(yy, mul2(a2i1, at[3]));
    yy = add2(yy, mul2(a1r1, at[0]));
    yy = add2(yy, mul2(a1i1, at[1]));
    zput(y, r1, yy);
}

/// `zhemv_U_SKYLAKEX` (`zsymv_U_sse2.S`, `HEMV`), `m = offset = n`, unit
/// increments.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn zhemv_u_k(n: usize, alpha: Z, a: ZV, lda: usize, x: ZV, y: ZV) {
    let (t1, t2) = hemv_buffer(n, alpha, x);
    let mut is = 0usize;
    while is + 2 <= n {
        let (j, j1) = (is, is + 1);
        let c1 = a.off(j * lda);
        let c2 = a.off(j1 * lda);
        let at = [t1[j], t2[j], t1[j1], t2[j1]];
        let mut xs1 = [0.0f64; 2];
        let mut xs2 = [0.0f64; 2];
        let mut b = 0;
        while b + 2 <= is {
            hemv_pair_rows(b, c1, c2, at, &t1, &t2, &mut xs1, &mut xs2, y);
            b += 2;
        }
        // .L18: the 2x2 diagonal block.
        let (ajj, a2r, a2i, aj1j1) = (c1.re.at(j), c2.re.at(j), c2.im.at(j), c2.re.at(j1));
        xs1 = add2(xs1, mul2(ajj, at[0]));
        xs2 = add2(xs2, mul2(a2r, at[0]));
        xs2 = sub2(xs2, mul2(a2i, at[1]));
        xs1 = add2(xs1, mul2(a2r, at[2]));
        xs2 = add2(xs2, mul2(aj1j1, at[2]));
        xs1 = add2(xs1, mul2(a2i, at[3]));
        zput(y, j, add2(zget(y, j), xs1));
        zput(y, j1, add2(zget(y, j1), xs2));
        is += 2;
    }
    if n & 1 == 1 {
        // .L20 / .L22 / .L28: the last column alone; real and imaginary
        // parts of the column accumulate separately.
        let j = is;
        let c1 = a.off(j * lda);
        let (at1, at2) = (t1[j], t2[j]);
        let mut xs1 = [0.0f64; 2];
        let mut xs2 = [0.0f64; 2];
        for r in 0..j {
            let (ar, ai) = (c1.re.at(r), c1.im.at(r));
            xs1 = add2(xs1, mul2(ar, t1[r]));
            xs2 = sub2(xs2, mul2(ai, t2[r]));
            let mut yy = zget(y, r);
            yy = add2(yy, mul2(ar, at1));
            yy = add2(yy, mul2(ai, at2));
            zput(y, r, yy);
        }
        xs1 = add2(xs1, mul2(c1.re.at(j), at1));
        xs1 = add2(xs1, xs2);
        zput(y, j, add2(zget(y, j), xs1));
    }
}

/// `zhemv_L_SKYLAKEX` (`zsymv_L_sse2.S`, `HEMV`), `m = offset = n`, unit
/// increments.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn zhemv_l_k(n: usize, alpha: Z, a: ZV, lda: usize, x: ZV, y: ZV) {
    let (t1, t2) = hemv_buffer(n, alpha, x);
    let mut is = 0usize;
    while is + 2 <= n {
        let (j, j1) = (is, is + 1);
        let c1 = a.off(j * lda);
        let c2 = a.off(j1 * lda);
        let at = [t1[j], t2[j], t1[j1], t2[j1]];
        // .L11: the 2x2 diagonal block seeds the sums (no zero start).
        let (ajj, a1r, a1i, aj1j1) = (c1.re.at(j), c1.re.at(j1), c1.im.at(j1), c2.re.at(j1));
        let mut xs1 = mul2(ajj, at[0]);
        let mut xs2 = mul2(a1r, at[0]);
        xs2 = add2(xs2, mul2(a1i, at[1]));
        xs1 = add2(xs1, mul2(a1r, at[2]));
        xs2 = add2(xs2, mul2(aj1j1, at[2]));
        xs1 = sub2(xs1, mul2(a1i, at[3]));
        let mut b = j + 2;
        while b + 2 <= n {
            hemv_pair_rows(b, c1, c2, at, &t1, &t2, &mut xs1, &mut xs2, y);
            b += 2;
        }
        if b < n {
            // .L16: one row left.
            let (a1r, a1i, a2r, a2i) = (c1.re.at(b), c1.im.at(b), c2.re.at(b), c2.im.at(b));
            let mut yy = zget(y, b);
            xs1 = add2(xs1, mul2(a1r, t1[b]));
            yy = add2(yy, mul2(a1r, at[0]));
            xs1 = sub2(xs1, mul2(a1i, t2[b]));
            yy = add2(yy, mul2(a1i, at[1]));
            xs2 = add2(xs2, mul2(a2r, t1[b]));
            yy = add2(yy, mul2(a2r, at[2]));
            xs2 = sub2(xs2, mul2(a2i, t2[b]));
            yy = add2(yy, mul2(a2i, at[3]));
            zput(y, b, yy);
        }
        zput(y, j, add2(zget(y, j), xs1));
        zput(y, j1, add2(zget(y, j1), xs2));
        is += 2;
    }
    if n & 1 == 1 {
        // .L20
        let j = is;
        let ajj = a.re.at(j + j * lda);
        zput(y, j, add2(zget(y, j), mul2(ajj, t1[j])));
    }
}

/// `ZHEMV` through the Fortran interface (`interface/zhemv.c`), unit
/// increments. `n < 362` or one thread: the single-threaded kernel.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zhemv(upper: bool, n: usize, alpha: Z, a: ZV, lda: usize, x: ZV, beta: Z, y: ZV) {
    if n == 0 {
        return;
    }
    if beta.re != 1.0 || beta.im != 0.0 {
        zscal_k(n, beta.re, beta.im, y);
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }
    if upper {
        zhemv_u_k(n, alpha, a, lda, x, y);
    } else {
        zhemv_l_k(n, alpha, a, lda, x, y);
    }
}

// ---------------------------------------------------------------------------
// ZGEMV
// ---------------------------------------------------------------------------

/// `zgemv_n_SKYLAKEX` (`zgemv_n_4.c`, no CONJ/XCONJ). With `incx == 1` the
/// `n % 4` leftover columns go through `zgemv_kernel_4x2` then `4x1`; with
/// `incx > 1` (x copied into `xbuffer`) each leftover column is its own
/// `zgemv_kernel_4x1` call. `incy > 1` takes the scalar, FMA-contracted
/// branch of `add_y`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn zgemv_n_k(m: usize, n: usize, alr: f64, ali: f64, a: ZV, lda: usize, x: ZV, incx: usize, y: ZV, incy: usize) {
    let m3 = m & 3;
    let mk = m - m3;
    let n4 = n & !3;
    // Rows below `mk` go through zgemv_kernel_4x4/4x2/4x1 into a zeroed
    // ybuffer and then add_y; every row is an independent chain, so the
    // NBMAX blocking does not change a value.
    let group = |r: usize, j0: usize, len: usize, br: &mut f64, bi: &mut f64| {
        let p = r + j0 * lda;
        let (ar, ai, xr, xi) = (a.re.at(p), a.im.at(p), x.re.at(j0 * incx), x.im.at(j0 * incx));
        let mut srr = ar * xr;
        let mut sir = ai * xr;
        let mut sri = ar * xi;
        let mut sii = ai * xi;
        for c in 1..len {
            let jj = j0 + c;
            let p = r + jj * lda;
            let (ar, ai, xr, xi) = (a.re.at(p), a.im.at(p), x.re.at(jj * incx), x.im.at(jj * incx));
            srr = ar.mul_add(xr, srr);
            sir = ai.mul_add(xr, sir);
            sri = ar.mul_add(xi, sri);
            sii = ai.mul_add(xi, sii);
        }
        // vaddsubpd: even lane subtracts, odd lane adds; then vaddpd into y.
        *br += srr - sii;
        *bi += sir + sri;
    };
    for r in 0..mk {
        let mut br = 0.0f64;
        let mut bi = 0.0f64;
        let mut j = 0;
        while j < n4 {
            group(r, j, 4, &mut br, &mut bi);
            j += 4;
        }
        if incx == 1 {
            if n & 2 != 0 {
                group(r, j, 2, &mut br, &mut bi);
                j += 2;
            }
            if n & 1 != 0 {
                group(r, j, 1, &mut br, &mut bi);
            }
        } else {
            while j < n {
                group(r, j, 1, &mut br, &mut bi);
                j += 1;
            }
        }
        let q = r * incy;
        if incy == 1 {
            // add_y, inc_dest == 2: vmulpd, vaddsubpd, vaddpd.
            y.re.set(q, y.re.at(q) + ((br * alr) - (bi * ali)));
            y.im.set(q, y.im.at(q) + ((bi * alr) + (br * ali)));
        } else {
            // add_y, inc_dest != 2: vfmsub231sd / vfmadd132sd, then vaddsd.
            y.re.set(q, alr.mul_add(br, -(ali * bi)) + y.re.at(q));
            y.im.set(q, alr.mul_add(bi, ali * br) + y.im.at(q));
        }
    }
    // m % 4 tail rows: scalar C, FMA-contracted.
    for r in mk..m {
        let mut tr = 0.0f64;
        let mut ti = 0.0f64;
        for j in 0..n {
            let p = r + j * lda;
            let (ar, ai, xr, xi) = (a.re.at(p), a.im.at(p), x.re.at(j * incx), x.im.at(j * incx));
            tr += ar.mul_add(xr, -(ai * xi));
            ti += ar.mul_add(xi, ai * xr);
        }
        let q = r * incy;
        y.re.set(q, y.re.at(q) + alr.mul_add(tr, -(ali * ti)));
        y.im.set(q, y.im.at(q) + alr.mul_add(ti, ali * tr));
    }
}

/// `zgemv_t_SKYLAKEX` (`conj == false`) and `zgemv_c_SKYLAKEX`
/// (`conj == true`, `-DCONJ`). `incx > 1` only copies x (`copy_x`, exact).
/// `incy > 1` makes the kernels write into a zeroed `ybuffer`, which is then
/// added to y: `y + (out + 0)`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn zgemv_t_k(
    m: usize,
    n: usize,
    alr: f64,
    ali: f64,
    a: ZV,
    lda: usize,
    x: ZV,
    incx: usize,
    y: ZV,
    incy: usize,
    conj: bool,
) {
    let m3 = m & 3;
    let mk = m - m3;
    // Blocks of NBMAX rows (then the m % NBMAX - m % 4 remainder), each a
    // separate dot product added into y.
    let mut s = 0usize;
    while s < mk {
        let nb = NBMAX.min(mk - s);
        for j in 0..n {
            let col = j * lda;
            // zgemv_kernel_4x{4,2,1}: identical per column. Lanes 0/1 take
            // the even rows of the block, lanes 2/3 the odd rows; each is a
            // vfmadd231pd chain from +0: [a_r*x_r, a_i*x_r, a_r*x_i, a_i*x_i].
            let mut acc = [[0.0f64; 4]; 2];
            for k in 0..nb {
                let r = s + k;
                let (ar, ai, xr, xi) = (a.re.at(col + r), a.im.at(col + r), x.re.at(r * incx), x.im.at(r * incx));
                let l = &mut acc[k & 1];
                l[0] = ar.mul_add(xr, l[0]);
                l[1] = ai.mul_add(xr, l[1]);
                l[2] = ar.mul_add(xi, l[2]);
                l[3] = ai.mul_add(xi, l[3]);
            }
            let (e, o) = (acc[0], acc[1]);
            let (tr, ti) = if !conj {
                ((e[0] - e[3]) + (o[0] - o[3]), (e[1] + e[2]) + (o[1] + o[2]))
            } else {
                ((e[3] + e[0]) + (o[3] + o[0]), (e[2] - e[1]) + (o[2] - o[1]))
            };
            let (or, oi) = ((tr * alr) - (ti * ali), (ti * alr) + (tr * ali));
            let q = j * incy;
            if incy == 1 {
                y.re.set(q, or + y.re.at(q));
                y.im.set(q, oi + y.im.at(q));
            } else {
                y.re.set(q, y.re.at(q) + (or + 0.0));
                y.im.set(q, y.im.at(q) + (oi + 0.0));
            }
        }
        s += nb;
    }
    if m3 == 0 {
        return;
    }
    // m % 4 tail rows: scalar C, FMA-contracted; first term assigned.
    for j in 0..n {
        let col = j * lda;
        let mut tr = 0.0f64;
        let mut ti = 0.0f64;
        for r in mk..m {
            let (ar, ai, xr, xi) = (a.re.at(col + r), a.im.at(col + r), x.re.at(r * incx), x.im.at(r * incx));
            let (pr, pi) = if !conj {
                (ar.mul_add(xr, -(ai * xi)), ar.mul_add(xi, ai * xr))
            } else {
                (ar.mul_add(xr, ai * xi), ar.mul_add(xi, -(ai * xr)))
            };
            if r == mk {
                tr = pr;
                ti = pi;
            } else {
                tr += pr;
                ti += pi;
            }
        }
        let q = j * incy;
        y.re.set(q, y.re.at(q) + alr.mul_add(tr, -(ali * ti)));
        y.im.set(q, y.im.at(q) + alr.mul_add(ti, ali * tr));
    }
}

/// `ZGEMV` through the Fortran interface (`interface/zgemv.c`), any positive
/// increments. One thread (`m*n < 1024 *
/// GEMM_MULTITHREAD_THRESHOLD` or `OPENBLAS_NUM_THREADS=1`).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zgemv(
    trans: Tr,
    m: usize,
    n: usize,
    alpha: Z,
    a: ZV,
    lda: usize,
    x: ZV,
    incx: usize,
    beta: Z,
    y: ZV,
    incy: usize,
) {
    assert!(incx >= 1 && incy >= 1, "zgemv: positive increments only");
    if m == 0 || n == 0 {
        return;
    }
    let leny = if trans == Tr::N { m } else { n };
    if beta.re != 1.0 || beta.im != 0.0 {
        if incy == 1 {
            zscal_k(leny, beta.re, beta.im, y);
        } else {
            zscal_k_inc(leny, beta.re, beta.im, y, incy);
        }
    }
    if alpha.re == 0.0 && alpha.im == 0.0 {
        return;
    }
    match trans {
        Tr::N => zgemv_n_k(m, n, alpha.re, alpha.im, a, lda, x, incx, y, incy),
        Tr::T => zgemv_t_k(m, n, alpha.re, alpha.im, a, lda, x, incx, y, incy, false),
        Tr::C => zgemv_t_k(m, n, alpha.re, alpha.im, a, lda, x, incx, y, incy, true),
    }
}

// ---------------------------------------------------------------------------
// ZHER2
// ---------------------------------------------------------------------------

/// `ZHER2` through the Fortran interface (`interface/zher2.c` ->
/// `driver/level2/zher2_k.c`), unit increments. Per column `i`:
/// `a(:, i) += conj(alpha*x_i) * y`, then `+= alpha*conj(y_i) * x` (two
/// `zaxpy_k` calls over rows `0..=i` or `i..n`), then `Im a(i,i) = 0`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zher2(upper: bool, n: usize, alpha: Z, x: ZV, y: ZV, a: ZV, lda: usize) {
    if n == 0 || (alpha.re == 0.0 && alpha.im == 0.0) {
        return;
    }
    let (ar, ai) = (alpha.re, alpha.im);
    let mai = -ai;
    for i in 0..n {
        let (xr, xi) = (x.re.at(i), x.im.at(i));
        let d1r = (xr * ar) - (xi * ai);
        let d1i = (mai * xr) - (ar * xi);
        let (yr, yi) = (y.re.at(i), y.im.at(i));
        let d2r = (yr * ar) + (yi * ai);
        let d2i = (ai * yr) - (ar * yi);
        let col = a.off(i * lda);
        if upper {
            zaxpy_k(i + 1, d1r, d1i, y, col);
            zaxpy_k(i + 1, d2r, d2i, x, col);
        } else {
            zaxpy_k(n - i, d1r, d1i, y.off(i), col.off(i));
            zaxpy_k(n - i, d2r, d2i, x.off(i), col.off(i));
        }
        col.im.set(i, 0.0);
    }
}

// ---------------------------------------------------------------------------
// ZGERC
// ---------------------------------------------------------------------------

/// `ZGERC` through the Fortran interface (`interface/zger.c` ->
/// `zgerc_k_SKYLAKEX` = `kernel/generic/zger.c -DCONJ`): per column `j`,
/// `a(:, j) += (alpha*conj(y_j)) * x` by `zaxpy_k`. Positive increments
/// (`incx != 1` copies `x` first, which is exact).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn zgerc(m: usize, n: usize, alpha: Z, x: ZV, incx: usize, y: ZV, incy: usize, a: ZV, lda: usize) {
    if m == 0 || n == 0 || (alpha.re == 0.0 && alpha.im == 0.0) {
        return;
    }
    let (mut xr_buf, mut xi_buf);
    let xv = if incx == 1 {
        x
    } else {
        xr_buf = (0..m).map(|i| x.re.at(i * incx)).collect::<Vec<f64>>();
        xi_buf = (0..m).map(|i| x.im.at(i * incx)).collect::<Vec<f64>>();
        ZV::new(D::new(&mut xr_buf), D::new(&mut xi_buf))
    };
    let (ar, ai) = (alpha.re, alpha.im);
    for j in 0..n {
        let (yr, yi) = (y.re.at(j * incy), y.im.at(j * incy));
        // vfmadd132sd (y_r*ar + y_i*ai) and vfnmadd231sd (-(y_i*ar) + y_r*ai).
        let da_r = yr.mul_add(ar, yi * ai);
        let da_i = (-yi).mul_add(ar, yr * ai);
        zaxpy_k(m, da_r, da_i, xv, a.off(j * lda));
    }
}
