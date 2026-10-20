//! OpenBLAS 0.3.30 SkylakeX track (scipy's `libscipy_openblas`, `OPENBLAS_CORETYPE=SkylakeX`,
//! `OPENBLAS_NUM_THREADS=1`): real double level-2 BLAS, bit for bit.
//!
//! Transcribed from the source *and* the disassembly of `libscipy_openblas-6cdc3b4a.so`
//! (the binary decides every fused multiply-add and every reduction tree):
//!
//! * `DGEMV`: `interface/gemv.c` (`scipy_dgemv_`) -> `SCAL_K` = `dscal_k_SKYLAKEX`
//!   (`kernel/x86_64/dscal.c`, flag 0) -> `dgemv_n_SKYLAKEX` (`kernel/x86_64/dgemv_n_4.c` with
//!   `dgemv_n_microk_skylakex-4.c`: `dgemv_kernel_4x4.isra.0`, inlined `4x2`, `dgemv_kernel_4x1`,
//!   `add_y`) / `dgemv_t_SKYLAKEX` (`kernel/x86_64/dgemv_t_4.c` with
//!   `dgemv_t_microk_haswell-4.c`: `dgemv_kernel_4x4`, `dgemv_kernel_4x2`, `dgemv_kernel_4x1`,
//!   `add_y`).
//! * `DSYMV`: `interface/symv.c` (`scipy_dsymv_`) -> `dscal_k_SKYLAKEX` ->
//!   `dsymv_U_SKYLAKEX` (`kernel/x86_64/dsymv_U.c` with `dsymv_U_microk_haswell-2.c`:
//!   `dsymv_kernel_4x4.constprop.0`, inlined `1x4`/`8x1`) / `dsymv_L_SKYLAKEX`
//!   (`kernel/x86_64/dsymv_L.c` with `dsymv_L_microk_skylakex-2.c`, all inlined).
//! * `DSYR2`: `interface/syr2.c` (`scipy_dsyr2_`, `n < 100` inline loop) and
//!   `driver/level2/syr2_k.c` (`dsyr2_U`/`dsyr2_L`) -> `daxpy_k_SKYLAKEX`
//!   (`kernel/x86_64/daxpy.c` with `daxpy_microk_skylakex-2.c`).
//! * `DGER`: `interface/ger.c` (`scipy_dger_`) -> `dger_k_SKYLAKEX` (`kernel/generic/ger.c`,
//!   `DGERKERNEL` default from `kernel/Makefile.L2`) -> `daxpy_k_SKYLAKEX`.
//!
//! `x.mul_add(y, acc)` is one `vfmadd*` (one rounding); a separate `*` then `+` is an unfused
//! `vmulsd`/`mulpd` + `vaddsd`/`addpd`.
use crate::ptr::D;

/// `NBMAX` of `dgemv_n_4.c` / `dgemv_t_4.c`.
const NBMAX: usize = 2048;

// ---------------------------------------------------------------------------------------------
// Shared level-1 pieces.
// ---------------------------------------------------------------------------------------------

/// `dscal_k_SKYLAKEX(n, 0, 0, da, x, inc, NULL, 0, NULL, 0)` as the level-2 interfaces call it
/// (flag `dummy2 = 0`): `da == 0` writes `+0.0`, otherwise each element is `x * da` (one
/// rounding, order-free).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn scal_flag0(n: usize, da: f64, x: D, inc: usize) {
    let mut ix = 0;
    for _ in 0..n {
        if da == 0.0 {
            x.set(ix, 0.0);
        } else {
            x.set(ix, x.at(ix) * da);
        }
        ix += inc;
    }
}

/// `daxpy_k_SKYLAKEX` with unit increments: every path (AVX-512 32-blocks, AVX2 16-blocks,
/// scalar tail) is `y = da*x + y` fused, so the result is element-wise.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn axpy_unit(n: usize, da: f64, x: D, y: D) {
    for i in 0..n {
        y.set(i, x.at(i).mul_add(da, y.at(i)));
    }
}

// ---------------------------------------------------------------------------------------------
// DGEMV 'N': dgemv_n_4.c
// ---------------------------------------------------------------------------------------------

/// `dgemv_kernel_4x4.isra.0` (n): four columns `ap[c] = a + (col+c)*lda + row`; per row
/// `s = a1*x1; s = fma(a0,x0,s); s = fma(a2,x2,s); s = fma(a3,x3,s); y = fma(s,alpha,y)`.
/// The 512- and 256-bit loops compute the identical per-element formula.
#[inline(always)]
unsafe fn gemv_n_k4x4(nb: usize, a: D, base: [usize; 4], xv: [f64; 4], y: D, alpha: f64) {
    for i in 0..nb {
        let mut s = a.at(base[1] + i) * xv[1];
        s = a.at(base[0] + i).mul_add(xv[0], s);
        s = a.at(base[2] + i).mul_add(xv[2], s);
        s = a.at(base[3] + i).mul_add(xv[3], s);
        y.set(i, s.mul_add(alpha, y.at(i)));
    }
}

/// Inlined `dgemv_kernel_4x2` (microk skylakex): `s = a1*x1; s = fma(a0,x0,s); y = fma(s,alpha,y)`.
#[inline(always)]
unsafe fn gemv_n_k4x2(nb: usize, a: D, base: [usize; 2], xv: [f64; 2], y: D, alpha: f64) {
    for i in 0..nb {
        let mut s = a.at(base[1] + i) * xv[1];
        s = a.at(base[0] + i).mul_add(xv[0], s);
        y.set(i, s.mul_add(alpha, y.at(i)));
    }
}

/// `dgemv_kernel_4x1` (SSE2 asm): `t = x0*alpha; y = y + a*t`, unfused.
#[inline(always)]
unsafe fn gemv_n_k4x1(nb: usize, a: D, base: usize, x0: f64, y: D, alpha: f64) {
    let t = x0 * alpha;
    for i in 0..nb {
        y.set(i, y.at(i) + a.at(base + i) * t);
    }
}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn gemv_n(m: usize, n: usize, alpha: f64, a: D, lda: usize, x: D, incx: usize, y: D, incy: usize) {
    if m < 1 || n < 1 {
        return;
    }
    let n1 = n >> 2;
    let n2 = n & 3;
    let m3 = m & 3;
    let mut m1 = (m & !3) as isize;
    let m2 = (m & (NBMAX - 1)) - m3;

    let mut ybuf = if incy != 1 { vec![0.0f64; NBMAX.min(m)] } else { Vec::new() };
    let mut a_row = 0usize; // `a += NB`: rows already consumed
    let mut y_off = 0usize;
    let mut nb = NBMAX;
    while nb == NBMAX {
        m1 -= NBMAX as isize;
        if m1 < 0 {
            if m2 == 0 {
                break;
            }
            nb = m2;
        }
        let yb = if incy != 1 {
            ybuf[..nb].fill(0.0);
            D::new(&mut ybuf)
        } else {
            y.off(y_off)
        };
        let col = |c: usize| a_row + c * lda;
        let mut c = 0usize;
        if incx == 1 {
            for _ in 0..n1 {
                gemv_n_k4x4(
                    nb,
                    a,
                    [col(c), col(c + 1), col(c + 2), col(c + 3)],
                    [x.at(c), x.at(c + 1), x.at(c + 2), x.at(c + 3)],
                    yb,
                    alpha,
                );
                c += 4;
            }
            if n2 & 2 != 0 {
                gemv_n_k4x2(nb, a, [col(c), col(c + 1)], [x.at(c), x.at(c + 1)], yb, alpha);
                c += 2;
            }
            if n2 & 1 != 0 {
                gemv_n_k4x1(nb, a, col(c), x.at(c), yb, alpha);
            }
        } else {
            let mut ix = 0usize;
            for _ in 0..n1 {
                let xv = [x.at(ix), x.at(ix + incx), x.at(ix + 2 * incx), x.at(ix + 3 * incx)];
                ix += 4 * incx;
                gemv_n_k4x4(nb, a, [col(c), col(c + 1), col(c + 2), col(c + 3)], xv, yb, alpha);
                c += 4;
            }
            for _ in 0..n2 {
                let x0 = x.at(ix);
                ix += incx;
                gemv_n_k4x1(nb, a, col(c), x0, yb, alpha);
                c += 1;
            }
        }
        a_row += nb;
        if incy != 1 {
            // add_y: dest = dest + src.
            for i in 0..nb {
                let k = y_off + i * incy;
                y.set(k, y.at(k) + ybuf[i]);
            }
            y_off += nb * incy;
        } else {
            y_off += nb;
        }
    }

    if m3 == 0 {
        return;
    }
    let ar = |i: usize, j: usize| a.at(a_row + i + j * lda);
    let xa = |j: usize| x.at(j * incx);
    if m3 == 3 {
        let mut t = [0.0f64; 3];
        if lda == 3 && incx == 1 {
            let n4 = n & !3;
            let mut j = 0;
            while j < n4 {
                for (k, tk) in t.iter_mut().enumerate() {
                    *tk += ar(k, j).mul_add(xa(j), ar(k, j + 1) * xa(j + 1));
                }
                for (k, tk) in t.iter_mut().enumerate() {
                    *tk += ar(k, j + 2).mul_add(xa(j + 2), ar(k, j + 3) * xa(j + 3));
                }
                j += 4;
            }
            for j in n4..n {
                for (k, tk) in t.iter_mut().enumerate() {
                    *tk = ar(k, j).mul_add(xa(j), *tk);
                }
            }
        } else {
            for j in 0..n {
                for (k, tk) in t.iter_mut().enumerate() {
                    *tk = ar(k, j).mul_add(xa(j), *tk);
                }
            }
        }
        for (k, tk) in t.iter().enumerate() {
            let iy = y_off + k * incy;
            y.set(iy, tk.mul_add(alpha, y.at(iy)));
        }
        return;
    }
    if m3 == 2 {
        let mut t = [0.0f64; 2];
        if lda == 2 && incx == 1 {
            let n4 = n & !3;
            let mut j = 0;
            while j < n4 {
                for (k, tk) in t.iter_mut().enumerate() {
                    *tk += ar(k, j).mul_add(xa(j), ar(k, j + 1) * xa(j + 1));
                }
                for (k, tk) in t.iter_mut().enumerate() {
                    *tk += ar(k, j + 2).mul_add(xa(j + 2), ar(k, j + 3) * xa(j + 3));
                }
                j += 4;
            }
            for j in n4..n {
                for (k, tk) in t.iter_mut().enumerate() {
                    *tk = ar(k, j).mul_add(xa(j), *tk);
                }
            }
        } else {
            for j in 0..n {
                for (k, tk) in t.iter_mut().enumerate() {
                    *tk = ar(k, j).mul_add(xa(j), *tk);
                }
            }
        }
        for (k, tk) in t.iter().enumerate() {
            let iy = y_off + k * incy;
            y.set(iy, tk.mul_add(alpha, y.at(iy)));
        }
        return;
    }
    // m3 == 1
    let mut t = 0.0f64;
    if lda == 1 && incx == 1 {
        let n4 = n & !3;
        let mut j = 0;
        while j < n4 {
            let mut s = ar(0, j + 1) * xa(j + 1);
            s = ar(0, j).mul_add(xa(j), s);
            s = ar(0, j + 2).mul_add(xa(j + 2), s);
            s = ar(0, j + 3).mul_add(xa(j + 3), s);
            t += s;
            j += 4;
        }
        for j in n4..n {
            t = ar(0, j).mul_add(xa(j), t);
        }
    } else {
        for j in 0..n {
            t = ar(0, j).mul_add(xa(j), t);
        }
    }
    y.set(y_off, t.mul_add(alpha, y.at(y_off)));
}

// ---------------------------------------------------------------------------------------------
// DGEMV 'T': dgemv_t_4.c
// ---------------------------------------------------------------------------------------------

/// `dgemv_kernel_4x4` (haswell asm): per column four lanes by row index mod 4, each
/// `acc = fma(x, a, acc)` in row order; result `(acc0 + acc2) + (acc1 + acc3)`.
#[inline(always)]
unsafe fn gemv_t_k4x4(nb: usize, a: D, base: usize, xb: D) -> f64 {
    let mut acc = [0.0f64; 4];
    for i in 0..nb {
        acc[i & 3] = xb.at(i).mul_add(a.at(base + i), acc[i & 3]);
    }
    (acc[0] + acc[2]) + (acc[1] + acc[3])
}

/// `dgemv_kernel_4x2` (SSE2 asm): lanes by row mod 2, unfused `acc + a*x`; `acc0 + acc1`.
#[inline(always)]
unsafe fn gemv_t_k4x2(nb: usize, a: D, base: usize, xb: D) -> f64 {
    let mut acc = [0.0f64; 2];
    for i in 0..nb {
        acc[i & 1] += a.at(base + i) * xb.at(i);
    }
    acc[0] + acc[1]
}

/// `dgemv_kernel_4x1` (SSE2 asm): two 2-lane accumulators (rows mod 4 in {0,1} and {2,3}),
/// unfused; `(acc0 + acc2) + (acc1 + acc3)`.
#[inline(always)]
unsafe fn gemv_t_k4x1(nb: usize, a: D, base: usize, xb: D) -> f64 {
    let mut acc = [0.0f64; 4];
    for i in 0..nb {
        acc[i & 3] += a.at(base + i) * xb.at(i);
    }
    (acc[0] + acc[2]) + (acc[1] + acc[3])
}

/// `add_y(n, da, src, dest, inc)`: `inc != 1` fused `src*da + dest`; `inc == 1` SSE2 unfused
/// `dest + src*da`.
#[inline(always)]
unsafe fn gemv_t_add_y(n: usize, da: f64, src: &[f64], y: D, y_off: usize, inc: usize) {
    if inc != 1 {
        for (i, s) in src[..n].iter().enumerate() {
            let k = y_off + i * inc;
            y.set(k, s.mul_add(da, y.at(k)));
        }
    } else {
        for (i, s) in src[..n].iter().enumerate() {
            y.set(y_off + i, y.at(y_off + i) + s * da);
        }
    }
}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn gemv_t(m: usize, n: usize, alpha: f64, a: D, lda: usize, x: D, incx: usize, y: D, incy: usize) {
    if m < 1 || n < 1 {
        return;
    }
    let n0 = n / NBMAX;
    let n1 = (n % NBMAX) >> 2;
    let n2 = n & 3;
    let m3 = m & 3;
    let mut m1 = (m & !3) as isize;
    let m2 = (m & (NBMAX - 1)) - m3;

    let mut xbuf = if incx != 1 { vec![0.0f64; NBMAX.min(m)] } else { Vec::new() };
    let mut ytemp = vec![0.0f64; NBMAX];
    let mut a_row = 0usize;
    let mut x_off = 0usize;
    let mut nb = NBMAX;
    while nb == NBMAX {
        m1 -= NBMAX as isize;
        if m1 < 0 {
            if m2 == 0 {
                break;
            }
            nb = m2;
        }
        let mut y_off = 0usize;
        let xb = if incx == 1 {
            x.off(x_off)
        } else {
            for i in 0..nb {
                xbuf[i] = x.at(x_off + i * incx);
            }
            D::new(&mut xbuf)
        };
        let mut c = 0usize;
        for _ in 0..n0 {
            for i in 0..NBMAX / 4 {
                for q in 0..4 {
                    ytemp[4 * i + q] = gemv_t_k4x4(nb, a, a_row + (c + q) * lda, xb);
                }
                c += 4;
            }
            gemv_t_add_y(NBMAX, alpha, &ytemp, y, y_off, incy);
            y_off += NBMAX * incy;
        }
        for i in 0..n1 {
            for q in 0..4 {
                ytemp[4 * i + q] = gemv_t_k4x4(nb, a, a_row + (c + q) * lda, xb);
            }
            c += 4;
        }
        if n1 > 0 {
            gemv_t_add_y(n1 * 4, alpha, &ytemp, y, y_off, incy);
            y_off += n1 * 4 * incy;
        }
        if n2 & 2 != 0 {
            let yb0 = gemv_t_k4x2(nb, a, a_row + c * lda, xb);
            let yb1 = gemv_t_k4x2(nb, a, a_row + (c + 1) * lda, xb);
            c += 2;
            y.set(y_off, yb0.mul_add(alpha, y.at(y_off)));
            y_off += incy;
            y.set(y_off, yb1.mul_add(alpha, y.at(y_off)));
            y_off += incy;
        }
        if n2 & 1 != 0 {
            let yb = gemv_t_k4x1(nb, a, a_row + c * lda, xb);
            y.set(y_off, yb.mul_add(alpha, y.at(y_off)));
        }
        a_row += nb;
        x_off += nb * incx;
    }

    if m3 == 0 {
        return;
    }
    let aj = |i: usize, j: usize| a.at(a_row + i + j * lda);
    if m3 == 3 {
        let xt0 = x.at(x_off) * alpha;
        let xt1 = x.at(x_off + incx) * alpha;
        let xt2 = x.at(x_off + 2 * incx) * alpha;
        for j in 0..n {
            let mut s = aj(1, j) * xt1;
            s = aj(0, j).mul_add(xt0, s);
            s = aj(2, j).mul_add(xt2, s);
            let k = j * incy;
            y.set(k, s + y.at(k));
        }
        return;
    }
    if m3 == 2 {
        let xt0 = x.at(x_off) * alpha;
        let xt1 = x.at(x_off + incx) * alpha;
        for j in 0..n {
            let mut s = aj(1, j) * xt1;
            s = aj(0, j).mul_add(xt0, s);
            let k = j * incy;
            y.set(k, s + y.at(k));
        }
        return;
    }
    let xt = x.at(x_off) * alpha;
    for j in 0..n {
        let k = j * incy;
        y.set(k, aj(0, j).mul_add(xt, y.at(k)));
    }
}

/// `DGEMV` through `scipy_dgemv_` (`trans = false` is `'N'`, `true` is `'T'`), positive
/// increments. The interface's stack-or-heap work buffer (`(m+n+19)&~3 <= 256` doubles on the
/// stack) only changes where `ybuffer`/`xbuffer`/`ytemp` live, not the kernels or the bits; with
/// one thread `gemv_thread_*` is never taken.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dgemv(
    trans: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: D,
    lda: usize,
    x: D,
    incx: usize,
    beta: f64,
    y: D,
    incy: usize,
) {
    assert!(incx >= 1 && incy >= 1, "positive increments only");
    if m == 0 || n == 0 {
        return;
    }
    let leny = if trans { n } else { m };
    if beta != 1.0 {
        scal_flag0(leny, beta, y, incy);
    }
    if alpha == 0.0 {
        return;
    }
    if trans {
        gemv_t(m, n, alpha, a, lda, x, incx, y, incy);
    } else {
        gemv_n(m, n, alpha, a, lda, x, incx, y, incy);
    }
}

// ---------------------------------------------------------------------------------------------
// DSYMV: dsymv_U.c / dsymv_L.c
// ---------------------------------------------------------------------------------------------

/// `dsymv_U_SKYLAKEX(m = n, offset = n)`, unit increments.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn symv_u(n: usize, alpha: f64, a: D, lda: usize, x: D, y: D) {
    let at = |i: usize, j: usize| a.at(i + j * lda);
    if n < 16 {
        // `mrange < 16`: plain column loop.
        for j in 0..n {
            let t1 = alpha * x.at(j);
            let mut t2 = 0.0f64;
            for i in 0..j {
                y.set(i, t1.mul_add(at(i, j), y.at(i)));
                t2 = at(i, j).mul_add(x.at(i), t2);
            }
            y.set(j, t1.mul_add(at(j, j), alpha * t2) + y.at(j));
        }
        return;
    }
    let m2 = n - (n % 4);
    let mut j = 0usize;
    while j < m2 {
        let t = [alpha * x.at(j), alpha * x.at(j + 1), alpha * x.at(j + 2), alpha * x.at(j + 3)];
        let mut tmp2 = [0.0f64; 4];
        let j1 = (j / 8) * 8;
        if j1 != 0 {
            // dsymv_kernel_4x4 (haswell asm), rows 0..j1: y chain over the four columns in
            // order; column accumulators in four lanes by row mod 4.
            let mut acc = [[0.0f64; 4]; 4];
            for i in 0..j1 {
                let xi = x.at(i);
                let mut yi = y.at(i);
                for (k, acck) in acc.iter_mut().enumerate() {
                    let aki = at(i, j + k);
                    yi = aki.mul_add(t[k], yi);
                    acck[i & 3] = aki.mul_add(xi, acck[i & 3]);
                }
                y.set(i, yi);
            }
            for k in 0..4 {
                tmp2[k] = ((acc[k][0] + acc[k][2]) + (acc[k][1] + acc[k][3])) + tmp2[k];
            }
        }
        if j1 < j {
            // Inlined dsymv_kernel_1x4, rows j1..j.
            let mut tacc = [0.0f64; 4];
            for i in j1..j {
                let xi = x.at(i);
                let mut s = t[1] * at(i, j + 1);
                s = at(i, j).mul_add(t[0], s);
                s = at(i, j + 2).mul_add(t[2], s);
                s = at(i, j + 3).mul_add(t[3], s);
                for (k, tk) in tacc.iter_mut().enumerate() {
                    *tk = at(i, j + k).mul_add(xi, *tk);
                }
                y.set(i, s + y.at(i));
            }
            for k in 0..4 {
                tmp2[k] += tacc[k];
            }
        }
        for k in 0..4 {
            let jj = j + k;
            let t1 = t[k];
            let mut t2 = tmp2[k];
            for i in j..jj {
                y.set(i, t1.mul_add(at(i, jj), y.at(i)));
                t2 = at(i, jj).mul_add(x.at(i), t2);
            }
            y.set(jj, t1.mul_add(at(jj, jj), alpha * t2) + y.at(jj));
        }
        j += 4;
    }
    for j in m2..n {
        let t1 = alpha * x.at(j);
        let mut t2 = 0.0f64;
        // Inlined dsymv_kernel_8x1 (rows 0..(j/8)*8, scalar despite its 4-step loop) and the
        // scalar rows up to j: one ordered chain for `t2`, element-wise for `y`.
        for i in 0..j {
            y.set(i, at(i, j).mul_add(t1, y.at(i)));
            t2 = at(i, j).mul_add(x.at(i), t2);
        }
        y.set(j, t1.mul_add(at(j, j), alpha * t2) + y.at(j));
    }
}

/// Row `i` of the four-column chain in `dsymv_L.c`'s scalar loops: `y = fma(t_k, a_k, y)` for
/// k = 0..4 in order, `tmp2_k = fma(a_k, x, tmp2_k)`.
#[inline(always)]
unsafe fn symv_l_row4(i: usize, a: D, cols: [usize; 4], x: D, y: D, t: &[f64; 4], tmp2: &mut [f64; 4]) {
    let xi = x.at(i);
    let mut yi = y.at(i);
    for k in 0..4 {
        let aki = a.at(cols[k] + i);
        yi = t[k].mul_add(aki, yi);
        tmp2[k] = aki.mul_add(xi, tmp2[k]);
    }
    y.set(i, yi);
}

/// `dsymv_kernel_4x4` of `dsymv_L_microk_skylakex-2.c`, rows `from..to` (`to - from` a
/// multiple of 4). Per row `s = a1*t1; s = fma(a0,t0,s); s = fma(a2,t2,s); s = fma(a3,t3,s);
/// y = s + y`. Accumulators: 8 lanes (`(i-from) mod 8`) over the AVX-512 part, folded
/// `lane l + lane l+4`, then 4 lanes over the AVX2 part; `(l0 + l2) + (l1 + l3)` added to `tmp2`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn symv_l_kernel(from: usize, to: usize, a: D, cols: [usize; 4], x: D, y: D, t: &[f64; 4], tmp2: &mut [f64; 4]) {
    let delta = (to - from) & !7;
    let to2 = from + delta;
    let mut acc5 = [[0.0f64; 8]; 4];
    let row = |i: usize, acc: &mut [[f64; 8]; 4], lane: usize| {
        let xi = x.at(i);
        let a0 = a.at(cols[0] + i);
        let a1 = a.at(cols[1] + i);
        let a2 = a.at(cols[2] + i);
        let a3 = a.at(cols[3] + i);
        let mut s = t[1] * a1;
        s = a0.mul_add(t[0], s);
        s = a2.mul_add(t[2], s);
        s = a3.mul_add(t[3], s);
        y.set(i, s + y.at(i));
        acc[0][lane] = a0.mul_add(xi, acc[0][lane]);
        acc[1][lane] = a1.mul_add(xi, acc[1][lane]);
        acc[2][lane] = a2.mul_add(xi, acc[2][lane]);
        acc[3][lane] = a3.mul_add(xi, acc[3][lane]);
    };
    for i in from..to2 {
        row(i, &mut acc5, (i - from) & 7);
    }
    let mut acc = [[0.0f64; 8]; 4];
    for k in 0..4 {
        for l in 0..4 {
            acc[k][l] = acc5[k][l + 4] + acc5[k][l];
        }
    }
    for i in to2..to {
        row(i, &mut acc, (i - to2) & 3);
    }
    for k in 0..4 {
        tmp2[k] += (acc[k][0] + acc[k][2]) + (acc[k][1] + acc[k][3]);
    }
}

/// `dsymv_L_SKYLAKEX(m = n, offset = n)`, unit increments.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
unsafe fn symv_l(n: usize, alpha: f64, a: D, lda: usize, x: D, y: D) {
    let offset1 = (n / 4) * 4;
    let m2 = (n / 4) * 4;
    let mut j = 0usize;
    while j < offset1 {
        let t = [alpha * x.at(j), alpha * x.at(j + 1), alpha * x.at(j + 2), alpha * x.at(j + 3)];
        let mut tmp2 = [0.0f64; 4];
        let cols = [j * lda, (j + 1) * lda, (j + 2) * lda, (j + 3) * lda];
        for k in 0..4 {
            let d = j + k;
            y.set(d, t[k].mul_add(a.at(cols[k] + d), y.at(d)));
        }
        for k in 0..3 {
            for i in j + 1 + k..j + 4 {
                let aki = a.at(cols[k] + i);
                y.set(i, t[k].mul_add(aki, y.at(i)));
                tmp2[k] = aki.mul_add(x.at(i), tmp2[k]);
            }
        }
        let from = j + 1;
        if n - from >= 12 {
            if m2 > j + 4 {
                symv_l_kernel(j + 4, m2, a, cols, x, y, &t, &mut tmp2);
            }
            for i in m2..n {
                symv_l_row4(i, a, cols, x, y, &t, &mut tmp2);
            }
        } else {
            for i in j + 4..n {
                symv_l_row4(i, a, cols, x, y, &t, &mut tmp2);
            }
        }
        for k in 0..4 {
            y.set(j + k, tmp2[k].mul_add(alpha, y.at(j + k)));
        }
        j += 4;
    }
    for j in offset1..n {
        let col = j * lda;
        let t1 = alpha * x.at(j);
        let mut t2 = 0.0f64;
        y.set(j, t1.mul_add(a.at(col + j), y.at(j)));
        // The `m - from >= 8` split into three ranges is one ordered chain.
        for i in j + 1..n {
            let aij = a.at(col + i);
            y.set(i, t1.mul_add(aij, y.at(i)));
            t2 = aij.mul_add(x.at(i), t2);
        }
        y.set(j, t2.mul_add(alpha, y.at(j)));
    }
}

/// `DSYMV` through `scipy_dsymv_`, unit increments (the `n >= 200` threaded branch is not taken
/// with one thread).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dsymv(upper: bool, n: usize, alpha: f64, a: D, lda: usize, x: D, beta: f64, y: D) {
    if n == 0 {
        return;
    }
    if beta != 1.0 {
        scal_flag0(n, beta, y, 1);
    }
    if alpha == 0.0 {
        return;
    }
    if upper {
        symv_u(n, alpha, a, lda, x, y);
    } else {
        symv_l(n, alpha, a, lda, x, y);
    }
}

// ---------------------------------------------------------------------------------------------
// DSYR2 / DGER
// ---------------------------------------------------------------------------------------------

/// `DSYR2` through `scipy_dsyr2_`, unit increments. The `n < 100` inline loop and
/// `dsyr2_U`/`dsyr2_L` are the same two `AXPYU_K` calls per column; no zero-skip.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dsyr2(upper: bool, n: usize, alpha: f64, x: D, y: D, a: D, lda: usize) {
    if n == 0 || alpha == 0.0 {
        return;
    }
    for i in 0..n {
        if upper {
            let col = a.off(i * lda);
            axpy_unit(i + 1, alpha * x.at(i), y, col);
            axpy_unit(i + 1, alpha * y.at(i), x, col);
        } else {
            let col = a.off(i * lda + i);
            axpy_unit(n - i, alpha * x.at(i), y.off(i), col);
            axpy_unit(n - i, alpha * y.at(i), x.off(i), col);
        }
    }
}

/// `DGER` through `scipy_dger_`, positive increments. The `m*n <= 8192` no-buffer call and the
/// `STACK_ALLOC(m)` call reach the same `dger_k_SKYLAKEX`; `incx != 1` copies `x` first (exact).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma"))]
pub unsafe fn dger(m: usize, n: usize, alpha: f64, x: D, incx: usize, y: D, incy: usize, a: D, lda: usize) {
    assert!(incx >= 1 && incy >= 1, "positive increments only");
    if m == 0 || n == 0 || alpha == 0.0 {
        return;
    }
    let mut xbuf;
    let xv = if incx != 1 {
        xbuf = (0..m).map(|i| x.at(i * incx)).collect::<Vec<f64>>();
        D::new(&mut xbuf)
    } else {
        x
    };
    for j in 0..n {
        axpy_unit(m, alpha * y.at(j * incy), xv, a.off(j * lda));
    }
}
