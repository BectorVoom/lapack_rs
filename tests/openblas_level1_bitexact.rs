//! Oracle gates for `lapack_rs::testing::openblas::level1` against scipy's OpenBLAS 0.3.30
//! (SkylakeX, one thread). Every output bit is compared, including the
//! elements a stride skips.
mod common;
use common::ob::*;
use lapack_rs::testing::V;
use lapack_rs::testing::openblas::level1 as l1;
use openblas_ref as ob;

/// `common::sizes(1000)` plus the `n > 10000` threading-threshold sizes.
fn all_sizes() -> Vec<usize> {
    let mut v = sizes(1000);
    v.extend([10000, 10001, 20000]);
    v
}

fn i32s(n: usize) -> i32 {
    i32::try_from(n).expect("fits i32")
}

const NAN_P: u64 = 0x7ff8_0000_0000_0000;
const NAN_N: u64 = 0xfff8_0000_0000_0000;
const NAN_PAYLOAD: u64 = 0x7ff8_0000_0000_1234;
const SNAN: u64 = 0x7ff0_0000_0000_0001;

/// Values that break a naive sum of squares, plus IEEE specials.
fn special(r: &mut Rng) -> f64 {
    match r.next_u64() % 14 {
        0 => 1e-200 * (1.0 + r.uniform()),
        1 => 1e200 * r.uniform(),
        2 => 0.0,
        3 => -0.0,
        4 => f64::from_bits(1 + (r.next_u64() % 1000)), // subnormal
        5 => f64::MAX * r.uniform(),
        6 => 1e154 * r.uniform(),
        7 => f64::MIN_POSITIVE * r.uniform(),
        _ => r.uniform(),
    }
}

#[test]
fn ddot_bitexact() {
    if !ob::require("ddot_bitexact") {
        return;
    }
    let mut r = Rng::new(7);
    for n in all_sizes() {
        for (incx, incy) in [(1usize, 1usize), (2, 3), (1, 2)] {
            for trial in 0..4 {
                let mut x = if trial % 2 == 0 { r.vec(n * incx + 1) } else { r.vec_edge(n * incx + 1) };
                let mut y = r.vec(n * incy + 1);
                let (n_, ix, iy) = (i32s(n), incx as i32, incy as i32);
                // SAFETY: n*inc entries each.
                let want = unsafe { ob::ddot(&n_, x.as_ptr(), &ix, y.as_ptr(), &iy) };
                // SAFETY: as above.
                let got = unsafe { l1::ddot(n, V::new(&mut x), incx, V::new(&mut y), incy) };
                assert_bits(&format!("ddot n={n} inc=({incx},{incy}) trial={trial}"), &[got], &[want]);
            }
        }
    }
}

#[test]
fn daxpy_bitexact() {
    if !ob::require("daxpy_bitexact") {
        return;
    }
    let mut r = Rng::new(11);
    for n in all_sizes() {
        for (incx, incy) in [(1usize, 1usize), (2, 3)] {
            for alpha in [0.0, 1.0, -1.0, r.uniform(), 1e7 * r.uniform()] {
                for trial in 0..2 {
                    let mut x = if trial == 0 { r.vec(n * incx + 1) } else { r.vec_edge(n * incx + 1) };
                    let y0 = if trial == 0 { r.vec(n * incy + 1) } else { r.vec_edge(n * incy + 1) };
                    let (mut yw, mut yg) = (y0.clone(), y0);
                    let (n_, ix, iy) = (i32s(n), incx as i32, incy as i32);
                    // SAFETY: n*inc entries each.
                    unsafe { ob::daxpy(&n_, &alpha, x.as_ptr(), &ix, yw.as_mut_ptr(), &iy) };
                    // SAFETY: as above.
                    unsafe { l1::daxpy(n, alpha, V::new(&mut x), incx, V::new(&mut yg), incy) };
                    assert_bits(&format!("daxpy n={n} inc=({incx},{incy}) alpha={alpha:e} trial={trial}"), &yg, &yw);
                }
            }
        }
    }
}

#[test]
fn dscal_bitexact() {
    if !ob::require("dscal_bitexact") {
        return;
    }
    let mut r = Rng::new(13);
    for n in all_sizes() {
        for incx in [1usize, 2] {
            for alpha in [0.0, -0.0, 1.0, -1.0, r.uniform(), 1e-300, 1e300] {
                for trial in 0..3 {
                    let x0: Vec<f64> = match trial {
                        0 => r.vec(n * incx + 1),
                        1 => r.vec_edge(n * incx + 1),
                        _ => (0..n * incx + 1)
                            .map(|i| match i % 7 {
                                0 => f64::from_bits(NAN_P),
                                1 => f64::INFINITY,
                                2 => f64::NEG_INFINITY,
                                3 => -0.0,
                                4 => f64::from_bits(NAN_PAYLOAD),
                                5 => f64::from_bits(NAN_N),
                                _ => special(&mut r),
                            })
                            .collect(),
                    };
                    let (mut xw, mut xg) = (x0.clone(), x0);
                    let (n_, ix) = (i32s(n), incx as i32);
                    // SAFETY: n*incx entries.
                    unsafe { ob::dscal(&n_, &alpha, xw.as_mut_ptr(), &ix) };
                    // SAFETY: as above.
                    unsafe { l1::dscal(n, alpha, V::new(&mut xg), incx) };
                    assert_bits(&format!("dscal n={n} incx={incx} alpha={alpha:e} trial={trial}"), &xg, &xw);
                }
            }
        }
    }
}

fn nrm2_case(what: &str, x: &mut [f64], n: usize, incx: usize) {
    let (n_, ix) = (i32s(n), incx as i32);
    // SAFETY: n*incx entries.
    let want = unsafe { ob::dnrm2(&n_, x.as_ptr(), &ix) };
    // SAFETY: as above.
    let got = unsafe { l1::dnrm2(n, V::new(x), incx) };
    assert_bits(what, &[got], &[want]);
}

#[test]
fn dnrm2_bitexact() {
    if !ob::require("dnrm2_bitexact") {
        return;
    }
    let mut r = Rng::new(17);
    for n in all_sizes() {
        for incx in [1usize, 2] {
            let len = n * incx + 1;
            for trial in 0..10 {
                let mut x: Vec<f64> = match trial {
                    0 => r.vec(len),
                    1 => r.vec_edge(len),
                    2 => (0..len).map(|_| 1e-200 * r.uniform()).collect(),
                    3 => (0..len).map(|_| 1e200 * r.uniform()).collect(),
                    4 => vec![0.0; len],
                    5 => {
                        let mut v = vec![0.0; len];
                        if n > 0 {
                            v[incx * (r.next_u64() as usize % n)] = 3.0 * r.uniform();
                        }
                        v
                    }
                    6 => (0..len).map(|_| special(&mut r)).collect(),
                    7 => (0..len).map(|_| f64::MAX * (1.0 - 1e-3 * (1.0 + r.uniform()))).collect(),
                    8 => (0..len).map(|_| f64::from_bits(1 + r.next_u64() % 4)).collect(),
                    _ => (0..len).map(|_| 1.0 + 1e-15 * r.uniform()).collect(),
                };
                nrm2_case(&format!("dnrm2 n={n} incx={incx} trial={trial}"), &mut x, n, incx);
            }
            // Inf / NaN placed in every lane and in the tail.
            if n <= 70 || n % 100 == 0 {
                let positions: Vec<usize> = if n <= 70 { (0..n).collect() } else { vec![0, 1, 2, 3, n / 2, n - 1] };
                for &p in &positions {
                    for (si, s) in [
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::from_bits(NAN_P),
                        f64::from_bits(NAN_N),
                        f64::from_bits(NAN_PAYLOAD),
                        f64::from_bits(SNAN),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let mut x = r.vec(n * incx + 1);
                        x[p * incx] = s;
                        nrm2_case(&format!("dnrm2 special={si} at {p} n={n} incx={incx}"), &mut x, n, incx);
                    }
                }
            }
        }
    }
    // Pairs of NaNs with different signs/payloads, and NaN with Inf, in all
    // lane combinations (the x87 NaN selection rule).
    let specials = [NAN_P, NAN_N, NAN_PAYLOAD, 0xfff8_0000_0000_1234, SNAN, 0x7ff0_0000_0000_0000];
    for n in [2usize, 3, 5, 8, 9, 13, 16, 17, 24] {
        for p in 0..n {
            for q in 0..n {
                if p == q {
                    continue;
                }
                for &a in &specials {
                    for &b in &specials {
                        let mut x = r.vec(n);
                        x[p] = f64::from_bits(a);
                        x[q] = f64::from_bits(b);
                        nrm2_case(&format!("dnrm2 pair n={n} {p}={a:x} {q}={b:x}"), &mut x, n, 1);
                    }
                }
            }
        }
    }
}

#[test]
fn dcopy_bitexact() {
    if !ob::require("dcopy_bitexact") {
        return;
    }
    let mut r = Rng::new(19);
    for n in all_sizes() {
        for (incx, incy) in [(1usize, 1usize), (3, 1), (7, 1), (64, 1), (129, 1)] {
            if n * incx > 400_000 {
                continue;
            }
            let mut x = r.vec_edge(n * incx + 1);
            if n > 0 {
                x[0] = f64::from_bits(SNAN);
            }
            let y0 = r.vec(n * incy + 1);
            let (mut yw, mut yg) = (y0.clone(), y0);
            let (n_, ix, iy) = (i32s(n), incx as i32, incy as i32);
            // SAFETY: n*inc entries each.
            unsafe { ob::dcopy(&n_, x.as_ptr(), &ix, yw.as_mut_ptr(), &iy) };
            // SAFETY: as above.
            unsafe { l1::dcopy(n, V::new(&mut x), incx, V::new(&mut yg), incy) };
            assert_bits(&format!("dcopy n={n} inc=({incx},{incy})"), &yg, &yw);
        }
    }
}

#[test]
fn dswap_bitexact() {
    if !ob::require("dswap_bitexact") {
        return;
    }
    let mut r = Rng::new(23);
    for n in all_sizes() {
        let x0 = r.vec_edge(n + 1);
        let y0 = r.vec_edge(n + 1);
        let (mut xw, mut yw, mut xg, mut yg) = (x0.clone(), y0.clone(), x0, y0);
        let n_ = i32s(n);
        // SAFETY: n entries each.
        unsafe { ob::dswap(&n_, xw.as_mut_ptr(), &1, yw.as_mut_ptr(), &1) };
        // SAFETY: as above.
        unsafe { l1::dswap(n, V::new(&mut xg), 1, V::new(&mut yg), 1) };
        assert_bits(&format!("dswap x n={n}"), &xg, &xw);
        assert_bits(&format!("dswap y n={n}"), &yg, &yw);
    }
}

#[test]
fn drot_bitexact() {
    if !ob::require("drot_bitexact") {
        return;
    }
    let mut r = Rng::new(29);
    for n in all_sizes() {
        for (incx, incy) in [(1usize, 1usize), (2, 3)] {
            let t = std::f64::consts::PI * r.uniform();
            for (c, s) in [(t.cos(), t.sin()), (1.0, 0.0), (r.uniform(), r.uniform()), (0.0, 1.0), (-1.0, -0.0)] {
                for trial in 0..2 {
                    let x0 = if trial == 0 { r.vec(n * incx + 1) } else { r.vec_edge(n * incx + 1) };
                    let y0 = if trial == 0 { r.vec(n * incy + 1) } else { r.vec_edge(n * incy + 1) };
                    let (mut xw, mut yw, mut xg, mut yg) = (x0.clone(), y0.clone(), x0, y0);
                    let (n_, ix, iy) = (i32s(n), incx as i32, incy as i32);
                    // SAFETY: n*inc entries each.
                    unsafe { ob::drot(&n_, xw.as_mut_ptr(), &ix, yw.as_mut_ptr(), &iy, &c, &s) };
                    // SAFETY: as above.
                    unsafe { l1::drot(n, V::new(&mut xg), incx, V::new(&mut yg), incy, c, s) };
                    let what = format!("drot n={n} inc=({incx},{incy}) c={c:e} s={s:e} trial={trial}");
                    assert_bits(&format!("{what} x"), &xg, &xw);
                    assert_bits(&format!("{what} y"), &yg, &yw);
                }
            }
        }
    }
}

/// Compare on `buf[off..]`, so the kernel sees `X mod 16` in {0, 8}.
fn iamax_case(what: &str, buf: &mut [f64], off: usize, n: usize, incx: usize) {
    let (n_, ix) = (i32s(n), incx as i32);
    let x = &mut buf[off..];
    // SAFETY: n*incx entries.
    let want = unsafe { ob::idamax(&n_, x.as_ptr(), &ix) };
    // SAFETY: as above.
    let got = unsafe { l1::idamax(n, V::new(x), incx) };
    assert_eq!(got as i64, want as i64, "{what} off={off} addr%16={}", (x.as_ptr() as usize) % 16);
}

#[test]
fn idamax_bitexact() {
    if !ob::require("idamax_bitexact") {
        return;
    }
    let mut r = Rng::new(31);
    let nan = f64::from_bits(NAN_P);
    for n in all_sizes() {
        for incx in [1usize, 2] {
            let len = n * incx + 2;
            for off in [0usize, 1] {
                for trial in 0..6 {
                    let mut x: Vec<f64> = match trial {
                        0 => r.vec(len),
                        1 => r.vec_edge(len),
                        2 => vec![0.0; len],
                        3 => (0..len).map(|i| if i % 2 == 0 { -0.0 } else { 0.0 }).collect(),
                        // Ties: few distinct magnitudes with both signs.
                        4 => (0..len)
                            .map(|_| ((r.next_u64() % 3) as f64) * if r.next_u64() % 2 == 0 { 1.0 } else { -1.0 })
                            .collect(),
                        _ => (0..len).map(|_| special(&mut r)).collect(),
                    };
                    iamax_case(&format!("idamax n={n} incx={incx} trial={trial}"), &mut x, off, n, incx);
                }
                // One or two NaNs (and Inf) at every position for small n, a
                // spread of positions for large n.
                let positions: Vec<usize> = if n <= 70 {
                    (0..n).collect()
                } else {
                    let mut p = vec![0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, n / 2, n - 3, n - 2, n - 1];
                    p.retain(|&q| q < n);
                    p
                };
                for &p in &positions {
                    for trial in 0..5 {
                        let mut x = r.vec(len);
                        match trial {
                            0 => x[off + p * incx] = nan,
                            1 => x[off + p * incx] = -nan,
                            2 => x[off + p * incx] = f64::INFINITY,
                            3 => {
                                x[off + p * incx] = nan;
                                let q = r.next_u64() as usize % n;
                                x[off + q * incx] = nan;
                            }
                            _ => {
                                // NaN plus a tie with the maximum elsewhere.
                                x[off + p * incx] = nan;
                                let q = r.next_u64() as usize % n;
                                x[off + q * incx] = -2.0;
                                let q2 = r.next_u64() as usize % n;
                                if x[off + q2 * incx].is_finite() {
                                    x[off + q2 * incx] = 2.0;
                                }
                            }
                        }
                        iamax_case(&format!("idamax nan-trial={trial} at {p} n={n} incx={incx}"), &mut x, off, n, incx);
                    }
                }
                // All NaN.
                let mut x = vec![nan; len];
                iamax_case(&format!("idamax all-nan n={n} incx={incx}"), &mut x, off, n, incx);
            }
        }
    }
    // Every NaN mask for short vectors (all lane/tail layouts).
    for n in 1usize..=18 {
        for mask in 0u32..(1 << n) {
            for off in [0usize, 1] {
                for incx in [1usize, 2] {
                    let mut x = r.vec(n * incx + 2);
                    for i in 0..n {
                        if mask >> i & 1 == 1 {
                            x[off + i * incx] = nan;
                        }
                    }
                    iamax_case(&format!("idamax mask={mask:b} n={n} incx={incx}"), &mut x, off, n, incx);
                }
            }
        }
    }
}

/// The x87 lane layout and final fold order only show in the double result
/// when a 64-bit rounding difference lands on a 53-bit rounding boundary, so
/// this sweeps many short vectors with spread magnitudes (it kills the
/// "wrong lane" and "wrong fold order" mutations that the grid above misses).
#[test]
fn dnrm2_extended_rounding_stress() {
    if !ob::require("dnrm2_extended_rounding_stress") {
        return;
    }
    let mut r = Rng::new(37);
    for case in 0..400_000usize {
        let n = 2 + (r.next_u64() % 40) as usize;
        let incx = 1 + (case % 2);
        let spread = (r.next_u64() % 64) as i32;
        let mut x: Vec<f64> = (0..n * incx + 1)
            .map(|_| {
                let e = (r.next_u64() % (2 * spread as u64 + 1)) as i32 - spread;
                r.uniform() * 2f64.powi(e)
            })
            .collect();
        nrm2_case(&format!("dnrm2 stress case={case} n={n} incx={incx} spread={spread}"), &mut x, n, incx);
    }
}
