//! Every transcribed BLAS/auxiliary routine against the reference on edge
//! inputs, bit for bit, plus the gfortran lowering probes (complex division
//! and multiplication, `x**n`, `ABS(COMPLEX)`, `MAX`/`MIN`/`SIGN`).

mod common;
use common::*;
use lapack_rs::testing as t;

/// Edge-heavy value generator: random, signed zeros, tiny, huge, subnormal.
fn edge_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut r = Rng::new(seed);
    (0..n)
        .map(|_| match r.next_u64() % 16 {
            0 => 0.0,
            1 => -0.0,
            2 => 1e-300 * r.uniform(),
            3 => 1e300 * r.uniform(),
            4 => 5e-320 * r.uniform(),
            5 => r.uniform() * 2f64.powi(-511),
            6 => r.uniform() * 2f64.powi(486),
            _ => r.uniform(),
        })
        .collect()
}

fn gate(name: &str) -> bool {
    lapack_ref::require(name)
}

#[test]
fn ddot_dnrm2() {
    if !gate("ddot_dnrm2") {
        return;
    }
    for n in [0usize, 1, 2, 4, 5, 6, 7, 11, 64, 257] {
        for seed in 0..8u64 {
            let mut x = edge_vec(n * 2, seed * 31 + n as u64);
            let mut y = edge_vec(n * 3, seed * 17 + 5);
            for (incx, incy) in [(1, 1), (2, 3)] {
                let want = lapack_ref::ddot(n, &x, incx, &y, incy);
                let got = unsafe { t::ddot(n, &mut x, incx, &mut y, incy) };
                assert_eq!(got.to_bits(), want.to_bits(), "ddot n={n} inc=({incx},{incy})");
            }
            let want = lapack_ref::dnrm2(&x[..n]);
            let got = unsafe { t::dnrm2(n, &mut x, 1) };
            assert_eq!(got.to_bits(), want.to_bits(), "dnrm2 n={n} seed={seed}");
        }
    }
}

#[test]
fn dgemv_dsymv_dsyr2() {
    if !gate("dgemv_dsymv_dsyr2") {
        return;
    }
    for (m, n) in [(1usize, 1usize), (3, 5), (17, 9), (40, 33), (200, 180)] {
        let a = edge_vec(m * n, 7 + m as u64);
        for trans in [false, true] {
            let (lx, ly) = if trans { (m, n) } else { (n, m) };
            for (alpha, beta) in [(1.0, 0.0), (-0.75, 1.0), (2.5, -0.5), (0.0, 0.5)] {
                for incx in [1usize, 2] {
                    let x = edge_vec(lx * incx, 11 + n as u64);
                    let y0 = edge_vec(ly, 13);
                    let mut yr = y0.clone();
                    lapack_ref::dgemv(if trans { 'T' } else { 'N' }, m, n, alpha, &a, m, &x, incx, beta, &mut yr, 1);
                    let (mut a2, mut x2, mut y) = (a.clone(), x.clone(), y0.clone());
                    unsafe { t::dgemv(trans, m, n, alpha, &mut a2, m, &mut x2, incx, beta, &mut y) };
                    assert_bits(&format!("dgemv {m}x{n} trans={trans} a={alpha} b={beta} incx={incx}"), &y, &yr);
                }
            }
        }
    }
    for n in [1usize, 2, 7, 33, 150] {
        let a = edge_vec(n * n, 21 + n as u64);
        let x = edge_vec(n, 23);
        let y0 = edge_vec(n, 29);
        for upper in [true, false] {
            let u = if upper { 'U' } else { 'L' };
            for (alpha, beta) in [(1.0, 0.0), (-1.5, 1.0), (0.3, 2.0)] {
                let mut yr = y0.clone();
                lapack_ref::dsymv(u, n, alpha, &a, n, &x, beta, &mut yr);
                let (mut a2, mut x2, mut y) = (a.clone(), x.clone(), y0.clone());
                unsafe { t::dsymv(upper, n, alpha, &mut a2, n, &mut x2, beta, &mut y) };
                assert_bits(&format!("dsymv n={n} {u} a={alpha} b={beta}"), &y, &yr);
            }
            let mut ar = a.clone();
            lapack_ref::dsyr2(u, n, -1.0, &x, &y0, &mut ar, n);
            let (mut a2, mut x2, mut y2) = (a.clone(), x.clone(), y0.clone());
            unsafe { t::dsyr2(upper, n, -1.0, &mut x2, &mut y2, &mut a2, n) };
            assert_bits(&format!("dsyr2 n={n} {u}"), &a2, &ar);
        }
    }
}

#[test]
fn dsyr2k_dgemm_dtrmm() {
    if !gate("dsyr2k_dgemm_dtrmm") {
        return;
    }
    for (n, k) in [(1usize, 1usize), (5, 3), (40, 32), (97, 17)] {
        let a = edge_vec(n * k, 41 + n as u64);
        let b = edge_vec(n * k, 43 + k as u64);
        let c0 = edge_vec(n * n, 47);
        for upper in [true, false] {
            for (alpha, beta) in [(-1.0, 1.0), (0.5, 0.0), (1.25, -2.0)] {
                let mut cr = c0.clone();
                lapack_ref::dsyr2k(if upper { 'U' } else { 'L' }, 'N', n, k, alpha, &a, n, &b, n, beta, &mut cr, n);
                let (mut a2, mut b2, mut c) = (a.clone(), b.clone(), c0.clone());
                unsafe { t::dsyr2k_n(upper, n, k, alpha, &mut a2, n, &mut b2, n, beta, &mut c, n) };
                assert_bits(&format!("dsyr2k n={n} k={k} upper={upper}"), &c, &cr);
            }
        }
    }
    for (m, n, k) in [(1usize, 1usize, 1usize), (3, 4, 5), (45, 37, 29), (130, 70, 64)] {
        for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
            let (ra, ca) = if ta { (k, m) } else { (m, k) };
            let (rb, cb) = if tb { (n, k) } else { (k, n) };
            let a = edge_vec(ra * ca, 51 + m as u64);
            let b = edge_vec(rb * cb, 53 + n as u64);
            let c0 = edge_vec(m * n, 59);
            for (alpha, beta) in [(1.0, 0.0), (-1.0, 1.0), (0.7, -0.3)] {
                let mut cr = c0.clone();
                lapack_ref::dgemm(
                    if ta { 'T' } else { 'N' },
                    if tb { 'T' } else { 'N' },
                    m,
                    n,
                    k,
                    alpha,
                    &a,
                    ra,
                    &b,
                    rb,
                    beta,
                    &mut cr,
                    m,
                );
                let (mut a2, mut b2, mut c) = (a.clone(), b.clone(), c0.clone());
                unsafe { t::dgemm(ta, tb, m, n, k, alpha, &mut a2, ra, &mut b2, rb, beta, &mut c, m) };
                assert_bits(&format!("dgemm {m}x{n}x{k} ta={ta} tb={tb}"), &c, &cr);
            }
        }
    }
    for (m, n) in [(1usize, 1usize), (4, 6), (33, 20), (70, 90)] {
        for left in [true, false] {
            let na = if left { m } else { n };
            let a = edge_vec(na * na, 61 + na as u64);
            let b0 = edge_vec(m * n, 67);
            for upper in [true, false] {
                for trans in [false, true] {
                    for nounit in [true, false] {
                        for alpha in [1.0, -1.0, 0.5] {
                            let mut br = b0.clone();
                            lapack_ref::dtrmm(
                                if left { 'L' } else { 'R' },
                                if upper { 'U' } else { 'L' },
                                if trans { 'T' } else { 'N' },
                                if nounit { 'N' } else { 'U' },
                                m,
                                n,
                                alpha,
                                &a,
                                na,
                                &mut br,
                                m,
                            );
                            let (mut a2, mut b) = (a.clone(), b0.clone());
                            unsafe { t::dtrmm(left, upper, trans, nounit, m, n, alpha, &mut a2, na, &mut b, m) };
                            assert_bits(
                                &format!("dtrmm {m}x{n} L={left} U={upper} T={trans} NU={nounit} a={alpha}"),
                                &b,
                                &br,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn dlartg_dlapy2_dlarfg_dlascl() {
    if !gate("dlartg_dlapy2_dlarfg_dlascl") {
        return;
    }
    let vals = edge_vec(64, 71);
    for &f in &vals {
        for &g in vals.iter().step_by(3) {
            let (c, s, r) = t::dlartg(f, g);
            let (cr, sr, rr) = lapack_ref::dlartg(f, g);
            assert_eq!(
                (c.to_bits(), s.to_bits(), r.to_bits()),
                (cr.to_bits(), sr.to_bits(), rr.to_bits()),
                "dlartg({f:e},{g:e})"
            );
            assert_eq!(t::dlapy2(f, g).to_bits(), lapack_ref::dlapy2(f, g).to_bits(), "dlapy2({f:e},{g:e})");
        }
    }
    for n in [0usize, 1, 2, 3, 10, 100] {
        for seed in 0..6u64 {
            let mut x = edge_vec(n.saturating_sub(1), 73 + seed);
            if seed == 4 {
                // Force the KNT rescaling loop: tiny alpha and x.
                for v in &mut x {
                    *v *= 1e-300;
                }
            }
            let alpha = if seed == 4 { 1e-305 } else { vals[seed as usize] };
            let mut xr = x.clone();
            let (ar, taur) = lapack_ref::dlarfg(n, alpha, &mut xr);
            let (a, tau) = unsafe { t::dlarfg(n, alpha, &mut x) };
            assert_eq!((a.to_bits(), tau.to_bits()), (ar.to_bits(), taur.to_bits()), "dlarfg n={n} seed={seed}");
            assert_bits(&format!("dlarfg x n={n} seed={seed}"), &x, &xr);
        }
    }
    let n = 13;
    let a0 = edge_vec(n * n, 79);
    for ty in ['G', 'L', 'U'] {
        for (cfrom, cto) in [(1.0, 3.0), (1e-300, 1e300), (1e300, 1e-300), (2.0, 2.0), (7.0, -0.0)] {
            let mut ar = a0.clone();
            lapack_ref::dlascl(ty, cfrom, cto, n, n, &mut ar, n);
            let mut a = a0.clone();
            unsafe { t::dlascl(ty, cfrom, cto, n, n, &mut a, n) };
            assert_bits(&format!("dlascl {ty} {cfrom:e}->{cto:e}"), &a, &ar);
        }
    }
}

#[test]
fn complex_helpers_and_lowering_probes() {
    if !gate("complex_helpers_and_lowering_probes") {
        return;
    }
    use lapack_ref::{C64, probes};
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(probes::AVAILABLE, "liblapackrs_extras.a missing; rerun `cargo xtask build-ref-lapack`");
    }
    let v = edge_vec(40, 83);
    let pairs: Vec<(f64, f64)> = v.chunks(2).map(|c| (c[0], c[1])).collect();
    let c = |p: (f64, f64)| C64 { re: p.0, im: p.1 };
    let bits = |p: (f64, f64)| (p.0.to_bits(), p.1.to_bits());
    for &x in &pairs {
        assert_eq!(t::zabs(x).to_bits(), probes::cabs(c(x)).expect("probe").to_bits(), "ABS({x:?})");
        for &y in &pairs {
            if y.0 != 0.0 || y.1 != 0.0 {
                let r = lapack_ref::zladiv(c(x), c(y));
                assert_eq!(bits(t::zladiv(x, y)), (r.re.to_bits(), r.im.to_bits()), "zladiv {x:?}/{y:?}");
                let r = probes::zdiv(c(x), c(y)).expect("probe");
                assert_eq!(
                    bits(t::zdiv_gfortran(x, y)),
                    (r.re.to_bits(), r.im.to_bits()),
                    "inline division {x:?}/{y:?}"
                );
            }
            let r = probes::zmul(c(x), c(y)).expect("probe");
            assert_eq!(bits(t::zmul(x, y)), (r.re.to_bits(), r.im.to_bits()), "inline multiplication {x:?}*{y:?}");
            let r = probes::rzmul(x.0, c(y)).expect("probe");
            assert_eq!(bits(t::rzmul(x.0, y)), (r.re.to_bits(), r.im.to_bits()), "REAL*COMPLEX {x:?}*{y:?}");
        }
    }
    for &x in &[2.0, -2.0, 0.5, 1.0 + f64::EPSILON, 3.7, -1e-10, 1e10] {
        for n in -12..=12 {
            assert_eq!(t::powi(x, n).to_bits(), probes::powi(x, n).expect("probe").to_bits(), "{x}**{n}");
        }
    }
    let specials = [0.0, -0.0, 1.0, -1.0, 1e-310, -2.5];
    for &a in &specials {
        for &b in &specials {
            assert_eq!(t::fmax(a, b).to_bits(), probes::max(a, b).expect("probe").to_bits(), "MAX({a:e},{b:e})");
            assert_eq!(t::fmin(a, b).to_bits(), probes::min(a, b).expect("probe").to_bits(), "MIN({a:e},{b:e})");
            assert_eq!(t::sign(a, b).to_bits(), probes::sign(a, b).expect("probe").to_bits(), "SIGN({a:e},{b:e})");
        }
    }
    // ZLARFG on edge inputs, including a pure-real alpha and the KNT path.
    for n in [1usize, 2, 5, 40] {
        for seed in 0..5u64 {
            let re = edge_vec(n - 1, 89 + seed);
            let im = edge_vec(n - 1, 97 + seed);
            let scale = if seed == 3 { 1e-300 } else { 1.0 };
            let (mut xr, mut xi): (Vec<f64>, Vec<f64>) =
                (re.iter().map(|v| v * scale).collect(), im.iter().map(|v| v * scale).collect());
            let alpha = if seed == 2 {
                (0.75, 0.0)
            } else {
                (pairs[seed as usize].0 * scale, pairs[seed as usize + 1].1 * scale)
            };
            let mut xc = to_c64(&xr, &xi);
            let (ar, taur) = lapack_ref::zlarfg(n, c(alpha), &mut xc);
            let (a, tau) = unsafe { t::zlarfg(n, alpha, &mut xr, &mut xi) };
            assert_eq!(bits(a), (ar.re.to_bits(), ar.im.to_bits()), "zlarfg alpha n={n} seed={seed}");
            assert_eq!(bits(tau), (taur.re.to_bits(), taur.im.to_bits()), "zlarfg tau n={n} seed={seed}");
            let (cr, ci) = from_c64(&xc);
            assert_bits("zlarfg x re", &xr, &cr);
            assert_bits("zlarfg x im", &xi, &ci);
        }
    }
}

/// **Task C** of `docs/cintx-lapack-blas-adoption-plan.md` (and Task I of
/// `docs/pyscf-lapack-coverage-plan.md`): the *public* `blocks::dgemv` /
/// `blocks::zgemv` surface against the reference BLAS, bit for bit, over
/// every `TRANS`, both increments and the shapes a cart→sph application
/// produces. The internal transcriptions are already covered by
/// `dgemv_dsymv_dsyr2`; this pins the wrapper's argument mapping, including
/// the planar complex convention.
#[test]
fn blocks_dgemv_and_zgemv() {
    if !gate("blocks_dgemv_and_zgemv") {
        return;
    }
    use lapack_rs::{Trans, blocks};
    let mut r = Rng::new(77_001);
    for &(m, n) in &[(1usize, 1usize), (1, 7), (7, 1), (3, 5), (16, 16), (31, 17), (64, 5)] {
        for lda in [m, m + 3] {
            for trans in [Trans::NoTrans, Trans::Trans, Trans::ConjTrans] {
                let tc = match trans {
                    Trans::NoTrans => 'N',
                    Trans::Trans => 'T',
                    Trans::ConjTrans => 'C',
                };
                let (lenx, leny) = if matches!(trans, Trans::NoTrans) { (n, m) } else { (m, n) };
                for &(incx, incy) in &[(1usize, 1usize), (2, 1), (1, 3)] {
                    for &(alpha, beta) in &[(1.0f64, 0.0f64), (0.0, 1.0), (-2.5, 0.75), (1.0, 1.0)] {
                        let tag = format!("m={m} n={n} lda={lda} {tc} inc=({incx},{incy}) a={alpha} b={beta}");
                        let a: Vec<f64> = (0..lda * n).map(|_| r.uniform()).collect();
                        let x: Vec<f64> = (0..(lenx - 1) * incx + 1).map(|_| r.uniform()).collect();
                        let y0: Vec<f64> = (0..(leny - 1) * incy + 1).map(|_| r.uniform()).collect();

                        let mut y = y0.clone();
                        blocks::dgemv(
                            trans,
                            m,
                            n,
                            alpha,
                            &mut a.clone(),
                            lda,
                            &mut x.clone(),
                            incx,
                            beta,
                            &mut y,
                            incy,
                        )
                        .expect("dgemv");
                        let mut yr = y0.clone();
                        lapack_ref::dgemv(tc, m, n, alpha, &a, lda, &x, incx, beta, &mut yr, incy);
                        assert_bits(&format!("blocks::dgemv {tag}"), &y, &yr);

                        // Complex: planar here, interleaved for the oracle.
                        let ai: Vec<f64> = (0..lda * n).map(|_| r.uniform()).collect();
                        let xi: Vec<f64> = (0..x.len()).map(|_| r.uniform()).collect();
                        let yi0: Vec<f64> = (0..y0.len()).map(|_| r.uniform()).collect();
                        let (are, aim) = (a.clone(), ai.clone());
                        let (xre, xim) = (x.clone(), xi.clone());
                        let (mut yre, mut yim) = (y0.clone(), yi0.clone());
                        let (al, be) = ((alpha, -0.5), (beta, 0.25));
                        blocks::zgemv(
                            trans,
                            m,
                            n,
                            al,
                            &mut are.clone(),
                            &mut aim.clone(),
                            lda,
                            &mut xre.clone(),
                            &mut xim.clone(),
                            incx,
                            be,
                            &mut yre,
                            &mut yim,
                            incy,
                        )
                        .expect("zgemv");
                        let ac = to_c64(&are, &aim);
                        let xc = to_c64(&xre, &xim);
                        let mut yc = to_c64(&y0, &yi0);
                        lapack_ref::zgemv(
                            tc,
                            m,
                            n,
                            lapack_ref::C64 { re: al.0, im: al.1 },
                            &ac,
                            lda,
                            &xc,
                            incx,
                            lapack_ref::C64 { re: be.0, im: be.1 },
                            &mut yc,
                            incy,
                        );
                        let (rr, ri) = from_c64(&yc);
                        assert_bits(&format!("blocks::zgemv re {tag}"), &yre, &rr);
                        assert_bits(&format!("blocks::zgemv im {tag}"), &yim, &ri);
                    }
                }
            }
        }
    }
}
