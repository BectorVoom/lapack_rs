//! Oracle tests: `obk_zlevel1` against scipy's OpenBLAS 0.3.30 (SkylakeX,
//! one thread), every output bit including untouched stride gaps.
mod common;
use common::ob::*;
use lapack_rs::testing::V;
use lapack_rs::testing::openblas::zlevel1 as t;
use lapack_rs::testing::{Z, ZV};
use openblas_ref::{self as ob, C64};

/// `common::assert_bits` with the differing bit patterns in the message
/// (NaN payloads and signs print as plain `NaN` otherwise).
#[track_caller]
fn assert_bits_hex(what: &str, got: &[f64], want: &[f64]) {
    if let Some(i) = got.iter().zip(want).position(|(g, w)| g.to_bits() != w.to_bits()) {
        eprintln!("{what}: first differing index {i}: got {:#018x} want {:#018x}", got[i].to_bits(), want[i].to_bits());
    }
    assert_bits(what, got, want);
}

fn all_sizes() -> Vec<usize> {
    let mut v = sizes(1000);
    v.extend([10000, 10001]);
    v
}

/// Planar complex vector with its interleaved oracle copy.
struct Cv {
    re: Vec<f64>,
    im: Vec<f64>,
}

impl Cv {
    fn new(re: Vec<f64>, im: Vec<f64>) -> Self {
        assert_eq!(re.len(), im.len());
        Cv { re, im }
    }
    fn il(&self) -> Vec<C64> {
        self.re.iter().zip(&self.im).map(|(&re, &im)| C64 { re, im }).collect()
    }
    fn view(&mut self) -> ZV {
        ZV::new(V::new(&mut self.re), V::new(&mut self.im))
    }
    #[track_caller]
    fn check(&self, what: &str, want: &[C64]) {
        let wr: Vec<f64> = want.iter().map(|c| c.re).collect();
        let wi: Vec<f64> = want.iter().map(|c| c.im).collect();
        assert_bits_hex(&format!("{what} re"), &self.re, &wr);
        assert_bits_hex(&format!("{what} im"), &self.im, &wi);
    }
}

/// Infinities, NaNs (both signs), signed zeros, huge and subnormal values
/// sprinkled into mostly ordinary data.
fn special(r: &mut Rng) -> f64 {
    match r.next_u64() % 24 {
        0 => f64::INFINITY,
        1 => f64::NEG_INFINITY,
        2 => f64::NAN,
        3 => -f64::NAN,
        4 => 0.0,
        5 => -0.0,
        6 => 1e300 * r.uniform(),
        7 => 4.9e-324 * (r.next_u64() % 1000) as f64 * r.uniform().signum(),
        8 => f64::MAX * r.uniform(),
        _ => r.uniform(),
    }
}

/// Input vector of `len` elements for trial kind `k`: 0 uniform, 1 edge,
/// 2 specials (Inf/NaN/huge/tiny).
fn make(r: &mut Rng, len: usize, k: usize) -> Cv {
    match k {
        0 => Cv::new(r.vec(len), r.vec(len)),
        1 => Cv::new(r.vec_edge(len), r.vec_edge(len)),
        _ => Cv::new((0..len).map(|_| special(r)).collect(), (0..len).map(|_| special(r)).collect()),
    }
}

#[test]
fn x87_control_word_is_default() {
    let mut cw: u16 = 0;
    // SAFETY: stores the x87 control word into a local u16.
    unsafe { std::arch::asm!("fnstcw word ptr [{}]", in(reg) &mut cw) };
    assert_eq!(cw & 0x0f00, 0x0300, "x87 must be extended precision, round to nearest (cw={cw:#06x})");
}

#[test]
fn zdotc_bitexact() {
    if !ob::require("zdotc_bitexact") {
        return;
    }
    let mut r = Rng::new(11);
    for n in all_sizes() {
        for (incx, incy) in [(1usize, 1usize), (2, 3)] {
            for trial in 0..3 {
                let mut x = make(&mut r, n * incx + 1, trial);
                let mut y = make(&mut r, n * incy + 1, trial % 2);
                let (xi, yi) = (x.il(), y.il());
                let (n_, ix, iy) = (n as i32, incx as i32, incy as i32);
                // SAFETY: n*inc elements each.
                let want = unsafe { ob::zdotc(&n_, xi.as_ptr(), &ix, yi.as_ptr(), &iy) };
                // SAFETY: as above.
                let got = unsafe { t::zdotc(n, x.view(), incx, y.view(), incy) };
                assert_bits_hex(
                    &format!("zdotc n={n} inc=({incx},{incy}) trial={trial}"),
                    &[got.re, got.im],
                    &[want.re, want.im],
                );
            }
        }
    }
}

#[test]
fn zaxpy_bitexact() {
    if !ob::require("zaxpy_bitexact") {
        return;
    }
    let mut r = Rng::new(12);
    for n in all_sizes() {
        let alphas =
            [Z::new(0.0, 0.0), Z::new(1.0, 0.0), Z::new(-1.0, 0.0), Z::new(0.0, 1.0), Z::new(r.uniform(), r.uniform())];
        for za in alphas {
            for (incx, incy) in [(1usize, 1usize), (2, 3)] {
                for trial in 0..3 {
                    let mut x = make(&mut r, n * incx + 1, trial);
                    let mut y = make(&mut r, n * incy + 2, trial % 2);
                    let xi = x.il();
                    let mut yi = y.il();
                    let (n_, ix, iy) = (n as i32, incx as i32, incy as i32);
                    let a = C64 { re: za.re, im: za.im };
                    // SAFETY: n*inc elements each.
                    unsafe { ob::zaxpy(&n_, &a, xi.as_ptr(), &ix, yi.as_mut_ptr(), &iy) };
                    // SAFETY: as above.
                    unsafe { t::zaxpy(n, za, x.view(), incx, y.view(), incy) };
                    y.check(&format!("zaxpy n={n} alpha={za:?} inc=({incx},{incy}) trial={trial}"), &yi);
                }
            }
        }
    }
}

#[test]
fn zscal_bitexact() {
    if !ob::require("zscal_bitexact") {
        return;
    }
    let mut r = Rng::new(13);
    for n in all_sizes() {
        let alphas = [
            Z::new(0.0, 0.0),
            Z::new(-0.0, 0.0),
            Z::new(1.0, 0.0),
            Z::new(r.uniform(), r.uniform()),
            Z::new(r.uniform(), 0.0),
            Z::new(0.0, r.uniform()),
        ];
        for za in alphas {
            for incx in [1usize, 2] {
                for trial in 0..3 {
                    let mut x = make(&mut r, n * incx + 1, trial);
                    let mut xi = x.il();
                    let (n_, ix) = (n as i32, incx as i32);
                    let a = C64 { re: za.re, im: za.im };
                    // SAFETY: n*incx elements.
                    unsafe { ob::zscal(&n_, &a, xi.as_mut_ptr(), &ix) };
                    // SAFETY: as above.
                    unsafe { t::zscal(n, za, x.view(), incx) };
                    x.check(&format!("zscal n={n} alpha={za:?} inc={incx} trial={trial}"), &xi);
                }
            }
        }
    }
}

#[test]
fn zdscal_bitexact() {
    if !ob::require("zdscal_bitexact") {
        return;
    }
    let mut r = Rng::new(14);
    for n in all_sizes() {
        for da in [0.0, -0.0, 1.0, r.uniform(), 1e300 * r.uniform()] {
            for incx in [1usize, 2] {
                for trial in 0..3 {
                    let mut x = make(&mut r, n * incx + 1, trial);
                    let mut xi = x.il();
                    let (n_, ix) = (n as i32, incx as i32);
                    // SAFETY: n*incx elements.
                    unsafe { ob::zdscal(&n_, &da, xi.as_mut_ptr(), &ix) };
                    // SAFETY: as above.
                    unsafe { t::zdscal(n, da, x.view(), incx) };
                    x.check(&format!("zdscal n={n} da={da:e} inc={incx} trial={trial}"), &xi);
                }
            }
        }
    }
}

#[test]
fn dznrm2_bitexact() {
    if !ob::require("dznrm2_bitexact") {
        return;
    }
    let mut r = Rng::new(15);
    let scaled = |r: &mut Rng, len: usize, s: f64| {
        Cv::new((0..len).map(|_| s * r.uniform()).collect(), (0..len).map(|_| s * r.uniform()).collect())
    };
    for n in all_sizes() {
        for incx in [1usize, 2] {
            let len = n * incx + 1;
            let mut cases: Vec<(String, Cv)> = Vec::new();
            for trial in 0..3 {
                cases.push((format!("trial={trial}"), make(&mut r, len, trial)));
            }
            for s in [1e300, f64::MAX, 1e154, 1e-154, 1e-300, 1e-320, 4.9e-324] {
                cases.push((format!("scale={s:e}"), scaled(&mut r, len, s)));
            }
            cases.push(("zeros".into(), Cv::new(vec![0.0; len], vec![-0.0; len])));
            // A single Inf / NaN in an otherwise ordinary vector.
            if n > 0 {
                for (k, v) in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -f64::NAN].into_iter().enumerate() {
                    let mut c = make(&mut r, len, 0);
                    let p = (r.next_u64() as usize % n) * incx;
                    if k % 2 == 0 {
                        c.re[p] = v
                    } else {
                        c.im[p] = v
                    }
                    cases.push((format!("special={v}"), c));
                }
            }
            for (name, mut x) in cases {
                let xi = x.il();
                let (n_, ix) = (n as i32, incx as i32);
                // SAFETY: n*incx elements.
                let want = unsafe { ob::dznrm2(&n_, xi.as_ptr(), &ix) };
                // SAFETY: as above.
                let got = unsafe { t::dznrm2(n, x.view(), incx) };
                assert_bits_hex(&format!("dznrm2 n={n} inc={incx} {name}"), &[got], &[want]);
            }
        }
    }
}

#[test]
fn zcopy_bitexact() {
    if !ob::require("zcopy_bitexact") {
        return;
    }
    let mut r = Rng::new(16);
    for n in all_sizes() {
        // (1,1) and ZLARFB's `ZCOPY( N, C( J, 1 ), LDC, WORK( 1, J ), 1 )`.
        for (incx, incy) in [(1usize, 1usize), (7, 1), (n.max(1) + 3, 1)] {
            for trial in 0..3 {
                let mut x = make(&mut r, n * incx + 1, trial);
                let mut y = make(&mut r, n * incy + 2, 1);
                let xi = x.il();
                let mut yi = y.il();
                let (n_, ix, iy) = (n as i32, incx as i32, incy as i32);
                // SAFETY: n*inc elements each.
                unsafe { ob::zcopy(&n_, xi.as_ptr(), &ix, yi.as_mut_ptr(), &iy) };
                // SAFETY: as above.
                unsafe { t::zcopy(n, x.view(), incx, y.view(), incy) };
                y.check(&format!("zcopy n={n} inc=({incx},{incy}) trial={trial}"), &yi);
            }
        }
    }
}

#[test]
fn zswap_bitexact() {
    if !ob::require("zswap_bitexact") {
        return;
    }
    let mut r = Rng::new(17);
    for n in all_sizes() {
        for trial in 0..3 {
            let mut x = make(&mut r, n + 1, trial);
            let mut y = make(&mut r, n + 2, 2 - trial);
            let mut xi = x.il();
            let mut yi = y.il();
            let (n_, one) = (n as i32, 1i32);
            // SAFETY: n elements each, disjoint.
            unsafe { ob::zswap(&n_, xi.as_mut_ptr(), &one, yi.as_mut_ptr(), &one) };
            // SAFETY: as above.
            unsafe { t::zswap(n, x.view(), 1, y.view(), 1) };
            x.check(&format!("zswap x n={n} trial={trial}"), &xi);
            y.check(&format!("zswap y n={n} trial={trial}"), &yi);
        }
    }
}
