mod common;
use common::ob::*;
use lapack_rs::testing::Tr;
use lapack_rs::testing::V;
use lapack_rs::testing::openblas::zlevel2::{zgemv, zgerc, zhemv, zher2};
use lapack_rs::testing::{Z, ZV};
use openblas_ref as ob;
use openblas_ref::C64;

/// Interleaved complex vector of length `n`; `edge` mixes in zeros, signed
/// zeros and spread magnitudes.
fn cvec(r: &mut Rng, n: usize, edge: bool) -> Vec<C64> {
    let re = if edge { r.vec_edge(n) } else { r.vec(n) };
    let im = if edge { r.vec_edge(n) } else { r.vec(n) };
    re.into_iter().zip(im).map(|(re, im)| C64 { re, im }).collect()
}

fn planar(v: &[C64]) -> (Vec<f64>, Vec<f64>) {
    (v.iter().map(|c| c.re).collect(), v.iter().map(|c| c.im).collect())
}

fn flat_c(v: &[C64]) -> Vec<f64> {
    v.iter().flat_map(|c| [c.re, c.im]).collect()
}

fn flat_p(re: &[f64], im: &[f64]) -> Vec<f64> {
    re.iter().zip(im).flat_map(|(r, i)| [*r, *i]).collect()
}

fn zv(re: &mut [f64], im: &mut [f64]) -> ZV {
    ZV::new(V::new(re), V::new(im))
}

fn alphas(r: &mut Rng) -> Vec<C64> {
    vec![
        C64 { re: 0.0, im: 0.0 },
        C64 { re: 1.0, im: 0.0 },
        C64 { re: -1.0, im: 0.0 },
        C64 { re: r.uniform(), im: r.uniform() },
        C64 { re: 0.0, im: r.uniform() },
    ]
}

fn betas(r: &mut Rng) -> Vec<C64> {
    vec![
        C64 { re: 0.0, im: 0.0 },
        C64 { re: 1.0, im: 0.0 },
        C64 { re: r.uniform(), im: r.uniform() },
        C64 { re: r.uniform(), im: 0.0 },
        C64 { re: 0.0, im: r.uniform() },
    ]
}

fn z(c: C64) -> Z {
    Z::new(c.re, c.im)
}

#[test]
fn zhemv_bitexact() {
    if !ob::require("zhemv_bitexact") {
        return;
    }
    let mut r = Rng::new(11);
    let mut ns = sizes(700);
    ns.extend([1023, 1024, 1025, 1100]);
    let mut cases = 0usize;
    for &n in &ns {
        let big = n > 400;
        for upper in [true, false] {
            for (ia, alpha) in alphas(&mut r).into_iter().enumerate() {
                for (ib, beta) in betas(&mut r).into_iter().enumerate() {
                    if big && !(ia == 3 || (ia == 1 && ib == 0)) {
                        continue;
                    }
                    for extra in [0usize, 3] {
                        if big && extra == 3 {
                            continue;
                        }
                        let lda = n.max(1) + extra;
                        let edge = (ia + ib + extra) % 2 == 1;
                        let a = cvec(&mut r, lda * n, edge);
                        let x = cvec(&mut r, n, edge);
                        let y0 = cvec(&mut r, n, !edge);
                        let mut yw = y0.clone();
                        let uplo = ob::ch(if upper { 'U' } else { 'L' });
                        let (n_, lda_, one) = (n as i32, lda as i32, 1i32);
                        // SAFETY: lda*n, n, n entries.
                        unsafe {
                            ob::zhemv(
                                &uplo,
                                &n_,
                                &alpha,
                                a.as_ptr(),
                                &lda_,
                                x.as_ptr(),
                                &one,
                                &beta,
                                yw.as_mut_ptr(),
                                &one,
                                1,
                            )
                        };
                        let (mut ar, mut ai) = planar(&a);
                        let (mut xr, mut xi) = planar(&x);
                        let (mut yr, mut yi) = planar(&y0);
                        // SAFETY: as above.
                        unsafe {
                            zhemv(
                                upper,
                                n,
                                z(alpha),
                                zv(&mut ar, &mut ai),
                                lda,
                                zv(&mut xr, &mut xi),
                                z(beta),
                                zv(&mut yr, &mut yi),
                            )
                        };
                        assert_bits(
                            &format!("zhemv upper={upper} n={n} lda={lda} alpha={alpha:?} beta={beta:?}"),
                            &flat_p(&yr, &yi),
                            &flat_c(&yw),
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("zhemv: {cases} cases bit-exact");
}

/// Dimension grid crossing the 4-lane, 8-lane, 16-lane and 1024-row blocks.
const DIMS: [usize; 24] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 257];

fn gemv_shapes() -> Vec<(usize, usize, bool)> {
    let mut v = Vec::new();
    for &m in &DIMS {
        for &n in &DIMS {
            v.push((m, n, false));
        }
    }
    // Row blocking (NBMAX = 1024) of zgemv_t/zgemv_c and big square cases.
    for m in [1020, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1031, 1100, 2047, 2048, 2051, 2100] {
        for n in [1, 2, 3, 4, 5, 7, 8, 17, 64] {
            v.push((m, n, true));
        }
    }
    for (m, n) in [(300, 520), (520, 300), (1100, 1100), (1100, 37), (37, 1100), (700, 700)] {
        v.push((m, n, true));
    }
    v
}

#[test]
fn zgemv_bitexact() {
    if !ob::require("zgemv_bitexact") {
        return;
    }
    let mut r = Rng::new(12);
    let mut cases = 0usize;
    for (m, n, big) in gemv_shapes() {
        for trans in [Tr::N, Tr::T, Tr::C] {
            for (ia, alpha) in alphas(&mut r).into_iter().enumerate() {
                for (ib, beta) in betas(&mut r).into_iter().enumerate() {
                    if big && !(ia == 3 || (ia == 2 && ib == 1) || (ia == 1 && ib == 0)) {
                        continue;
                    }
                    for extra in [0usize, 2] {
                        if big && extra == 0 && ia != 3 {
                            continue;
                        }
                        let lda = m.max(1) + extra;
                        let (lenx, leny) = if trans == Tr::N { (n, m) } else { (m, n) };
                        let edge = (ia + ib + extra / 2) % 2 == 1;
                        let a = cvec(&mut r, lda * n, edge);
                        let x = cvec(&mut r, lenx, edge);
                        let y0 = cvec(&mut r, leny, !edge);
                        let mut yw = y0.clone();
                        let t = ob::ch(match trans {
                            Tr::N => 'N',
                            Tr::T => 'T',
                            Tr::C => 'C',
                        });
                        let (m_, n_, lda_, one) = (m as i32, n as i32, lda as i32, 1i32);
                        // SAFETY: lda*n, lenx, leny entries.
                        unsafe {
                            ob::zgemv(
                                &t,
                                &m_,
                                &n_,
                                &alpha,
                                a.as_ptr(),
                                &lda_,
                                x.as_ptr(),
                                &one,
                                &beta,
                                yw.as_mut_ptr(),
                                &one,
                                1,
                            )
                        };
                        let (mut ar, mut ai) = planar(&a);
                        let (mut xr, mut xi) = planar(&x);
                        let (mut yr, mut yi) = planar(&y0);
                        // SAFETY: as above.
                        unsafe {
                            zgemv(
                                trans,
                                m,
                                n,
                                z(alpha),
                                zv(&mut ar, &mut ai),
                                lda,
                                zv(&mut xr, &mut xi),
                                1,
                                z(beta),
                                zv(&mut yr, &mut yi),
                                1,
                            )
                        };
                        assert_bits(
                            &format!("zgemv trans={trans:?} m={m} n={n} lda={lda} alpha={alpha:?} beta={beta:?}"),
                            &flat_p(&yr, &yi),
                            &flat_c(&yw),
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("zgemv: {cases} cases bit-exact");
}

#[test]
fn zher2_bitexact() {
    if !ob::require("zher2_bitexact") {
        return;
    }
    let mut r = Rng::new(13);
    let mut ns = sizes(700);
    ns.extend([1025, 1100]);
    let mut cases = 0usize;
    for &n in &ns {
        let big = n > 400;
        for upper in [true, false] {
            let al = vec![
                C64 { re: -1.0, im: 0.0 },
                C64 { re: 1.0, im: 0.0 },
                C64 { re: r.uniform(), im: r.uniform() },
                C64 { re: 0.0, im: r.uniform() },
                C64 { re: 0.0, im: 0.0 },
            ];
            for (ia, alpha) in al.into_iter().enumerate() {
                if big && ia != 2 && ia != 0 {
                    continue;
                }
                for extra in [0usize, 2] {
                    let lda = n.max(1) + extra;
                    let edge = (ia + extra / 2) % 2 == 1;
                    let a0 = cvec(&mut r, lda * n, edge);
                    let x = cvec(&mut r, n, edge);
                    let y = cvec(&mut r, n, !edge);
                    let mut aw = a0.clone();
                    let uplo = ob::ch(if upper { 'U' } else { 'L' });
                    let (n_, lda_, one) = (n as i32, lda as i32, 1i32);
                    // SAFETY: n, n, lda*n entries.
                    unsafe {
                        ob::zher2(&uplo, &n_, &alpha, x.as_ptr(), &one, y.as_ptr(), &one, aw.as_mut_ptr(), &lda_, 1)
                    };
                    let (mut ar, mut ai) = planar(&a0);
                    let (mut xr, mut xi) = planar(&x);
                    let (mut yr, mut yi) = planar(&y);
                    // SAFETY: as above.
                    unsafe {
                        zher2(upper, n, z(alpha), zv(&mut xr, &mut xi), zv(&mut yr, &mut yi), zv(&mut ar, &mut ai), lda)
                    };
                    assert_bits(
                        &format!("zher2 upper={upper} n={n} lda={lda} alpha={alpha:?}"),
                        &flat_p(&ar, &ai),
                        &flat_c(&aw),
                    );
                    cases += 1;
                }
            }
        }
    }
    eprintln!("zher2: {cases} cases bit-exact");
}

#[test]
fn zgerc_bitexact() {
    if !ob::require("zgerc_bitexact") {
        return;
    }
    let mut r = Rng::new(14);
    let mut cases = 0usize;
    for (m, n, big) in gemv_shapes() {
        let al = vec![
            C64 { re: -1.0, im: 0.0 },
            C64 { re: -r.uniform(), im: -r.uniform() },
            C64 { re: 1.0, im: 0.0 },
            C64 { re: 0.0, im: 0.0 },
        ];
        for (ia, alpha) in al.into_iter().enumerate() {
            if big && ia != 1 {
                continue;
            }
            for (incx, incy) in [(1usize, 1usize), (2, 3)] {
                for extra in [0usize, 2] {
                    if big && (incx != 1 || extra == 0) {
                        continue;
                    }
                    let lda = m.max(1) + extra;
                    let edge = (ia + extra / 2 + incx) % 2 == 1;
                    let a0 = cvec(&mut r, lda * n, edge);
                    let x = cvec(&mut r, m * incx, edge);
                    let y = cvec(&mut r, n * incy, !edge);
                    let mut aw = a0.clone();
                    let (m_, n_, lda_, ix, iy) = (m as i32, n as i32, lda as i32, incx as i32, incy as i32);
                    // SAFETY: m*incx, n*incy, lda*n entries.
                    unsafe { ob::zgerc(&m_, &n_, &alpha, x.as_ptr(), &ix, y.as_ptr(), &iy, aw.as_mut_ptr(), &lda_) };
                    let (mut ar, mut ai) = planar(&a0);
                    let (mut xr, mut xi) = planar(&x);
                    let (mut yr, mut yi) = planar(&y);
                    // SAFETY: as above.
                    unsafe {
                        zgerc(
                            m,
                            n,
                            z(alpha),
                            zv(&mut xr, &mut xi),
                            incx,
                            zv(&mut yr, &mut yi),
                            incy,
                            zv(&mut ar, &mut ai),
                            lda,
                        )
                    };
                    assert_bits(
                        &format!("zgerc m={m} n={n} lda={lda} inc=({incx},{incy}) alpha={alpha:?}"),
                        &flat_p(&ar, &ai),
                        &flat_c(&aw),
                    );
                    cases += 1;
                }
            }
        }
    }
    eprintln!("zgerc: {cases} cases bit-exact");
}

/// Non-unit increments: ZLATRD's `ZGEMV('N', .., W(I,IW+1), LDW, ONE, A(1,I), 1)`
/// (incx = LDW/LDA, incy = 1), plus `'T'`/`'C'` with incy > 1 and the
/// remaining combinations.
#[test]
fn zgemv_inc_bitexact() {
    if !ob::require("zgemv_inc_bitexact") {
        return;
    }
    let mut r = Rng::new(15);
    let mut cases = 0usize;
    for (m, n, big) in gemv_shapes() {
        for trans in [Tr::N, Tr::T, Tr::C] {
            let incs: Vec<(usize, usize)> = match trans {
                Tr::N => vec![(2, 1), (3, 1), (7, 1), (m + 5, 1), (2, 3)],
                _ => vec![(1, 2), (1, 3), (1, 7), (3, 2)],
            };
            let al = [C64 { re: -1.0, im: 0.0 }, C64 { re: r.uniform(), im: r.uniform() }];
            let be = [
                C64 { re: 1.0, im: 0.0 },
                C64 { re: r.uniform(), im: r.uniform() },
                C64 { re: 0.0, im: 0.0 },
                C64 { re: r.uniform(), im: 0.0 },
                C64 { re: 0.0, im: r.uniform() },
            ];
            for (ii, &(incx, incy)) in incs.iter().enumerate() {
                for (ia, alpha) in al.iter().enumerate() {
                    for (ib, beta) in be.iter().enumerate() {
                        if big && !((ia + ib + ii) % 3 == 0) {
                            continue;
                        }
                        if ib >= 2 && ia == 0 && ii > 1 {
                            continue;
                        }
                        let lda = m.max(1) + (ii % 2) * 3;
                        let (lenx, leny) = if trans == Tr::N { (n, m) } else { (m, n) };
                        let edge = (ia + ib + ii) % 2 == 1;
                        let a = cvec(&mut r, lda * n, edge);
                        let x = cvec(&mut r, lenx * incx, edge);
                        let y0 = cvec(&mut r, leny * incy, !edge);
                        let mut yw = y0.clone();
                        let t = ob::ch(match trans {
                            Tr::N => 'N',
                            Tr::T => 'T',
                            Tr::C => 'C',
                        });
                        let (m_, n_, lda_, ix, iy) = (m as i32, n as i32, lda as i32, incx as i32, incy as i32);
                        // SAFETY: lda*n, lenx*incx, leny*incy entries.
                        unsafe {
                            ob::zgemv(
                                &t,
                                &m_,
                                &n_,
                                alpha,
                                a.as_ptr(),
                                &lda_,
                                x.as_ptr(),
                                &ix,
                                beta,
                                yw.as_mut_ptr(),
                                &iy,
                                1,
                            )
                        };
                        let (mut ar, mut ai) = planar(&a);
                        let (mut xr, mut xi) = planar(&x);
                        let (mut yr, mut yi) = planar(&y0);
                        // SAFETY: as above.
                        unsafe {
                            zgemv(
                                trans,
                                m,
                                n,
                                z(*alpha),
                                zv(&mut ar, &mut ai),
                                lda,
                                zv(&mut xr, &mut xi),
                                incx,
                                z(*beta),
                                zv(&mut yr, &mut yi),
                                incy,
                            )
                        };
                        assert_bits(
                            &format!(
                                "zgemv trans={trans:?} m={m} n={n} lda={lda} inc=({incx},{incy}) alpha={alpha:?} beta={beta:?}"
                            ),
                            &flat_p(&yr, &yi),
                            &flat_c(&yw),
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("zgemv (increments): {cases} cases bit-exact");
}

/// Signed zeros that only the `ybuffer` path of `zgemv_t` (incy > 1) and the
/// `+0` starts of the lane accumulators can distinguish: zero `a`/`x`,
/// `y = -0`, `beta = 1`, `alpha = (-1, 1)` makes the kernel output `-0`.
#[test]
fn zgemv_signed_zero_bitexact() {
    if !ob::require("zgemv_signed_zero_bitexact") {
        return;
    }
    let nz = C64 { re: -0.0, im: -0.0 };
    for trans in [Tr::N, Tr::T, Tr::C] {
        for (m, n) in [(1usize, 1usize), (4, 4), (5, 3), (8, 9), (1027, 2)] {
            for (incx, incy) in [(1usize, 1usize), (1, 2), (2, 1), (3, 2)] {
                for alpha in [C64 { re: -1.0, im: 1.0 }, C64 { re: 1.0, im: -1.0 }, C64 { re: -1.0, im: 0.0 }] {
                    for zero in [C64 { re: 0.0, im: 0.0 }, nz] {
                        let lda = m;
                        let (lenx, leny) = if trans == Tr::N { (n, m) } else { (m, n) };
                        let a = vec![zero; lda * n];
                        let x = vec![zero; lenx * incx];
                        let y0 = vec![nz; leny * incy];
                        let beta = C64 { re: 1.0, im: 0.0 };
                        let mut yw = y0.clone();
                        let t = ob::ch(match trans {
                            Tr::N => 'N',
                            Tr::T => 'T',
                            Tr::C => 'C',
                        });
                        let (m_, n_, lda_, ix, iy) = (m as i32, n as i32, lda as i32, incx as i32, incy as i32);
                        // SAFETY: lda*n, lenx*incx, leny*incy entries.
                        unsafe {
                            ob::zgemv(
                                &t,
                                &m_,
                                &n_,
                                &alpha,
                                a.as_ptr(),
                                &lda_,
                                x.as_ptr(),
                                &ix,
                                &beta,
                                yw.as_mut_ptr(),
                                &iy,
                                1,
                            )
                        };
                        let (mut ar, mut ai) = planar(&a);
                        let (mut xr, mut xi) = planar(&x);
                        let (mut yr, mut yi) = planar(&y0);
                        // SAFETY: as above.
                        unsafe {
                            zgemv(
                                trans,
                                m,
                                n,
                                z(alpha),
                                zv(&mut ar, &mut ai),
                                lda,
                                zv(&mut xr, &mut xi),
                                incx,
                                z(beta),
                                zv(&mut yr, &mut yi),
                                incy,
                            )
                        };
                        assert_bits(
                            &format!(
                                "zgemv signed zero trans={trans:?} m={m} n={n} inc=({incx},{incy}) alpha={alpha:?} zero={zero:?}"
                            ),
                            &flat_p(&yr, &yi),
                            &flat_c(&yw),
                        );
                    }
                }
            }
        }
    }
}
