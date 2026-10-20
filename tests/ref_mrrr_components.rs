//! MRRR component routines against Reference-LAPACK 3.12.1, bit for bit, so a
//! `dstemr` mismatch localises to one routine: `dlaev2`/`dlae2`, `dlarrc`,
//! `dlarrk`, `dlaneg`, `dlar1v` (R1), the `dlasq` dqds family (R2),
//! `dlarra`/`dlarrr`/`dlarrb`/`dlarrf`/`dlarrj`/`dlarre` (R3) and `dlarrv`
//! (R4).

mod common;
use common::*;
use lapack_rs::testing as t;

/// A tridiagonal `(d, e)` and the derived `e2 = e**2` the MRRR leaves take.
fn tri(n: usize, seed: u64, kind: usize) -> (Vec<f64>, Vec<f64>) {
    let mut r = Rng::new(seed);
    let (mut d, mut e) = (vec![0.0; n], vec![0.0; n.max(2) - 1]);
    for i in 0..n {
        d[i] = match kind {
            0 => 2.0 * r.uniform() - 1.0,
            1 => i as f64 + 1.0,
            2 => 1.0,
            3 => (i as f64 - n as f64 / 2.0) * 1e-8,
            _ => (2.0 * r.uniform() - 1.0) * 1e6,
        };
    }
    for i in 0..n.max(2) - 1 {
        e[i] = match kind {
            0 => 2.0 * r.uniform() - 1.0,
            1 => 1.0,
            2 => 1e-300,
            3 => 1e-9,
            _ => (2.0 * r.uniform() - 1.0) * 1e-3,
        };
    }
    if n <= 1 {
        e.iter_mut().for_each(|v| *v = 0.0);
    }
    (d, e)
}

#[test]
fn laev2_and_lae2() {
    if !lapack_ref::require("laev2_and_lae2") {
        return;
    }
    let mut r = Rng::new(9_001);
    let mut cases: Vec<(f64, f64, f64)> = Vec::new();
    // The reference's branches: SM < 0 / = 0 / > 0, ADF > / = / < AB,
    // DF >= 0 / < 0, ACS > AB, AB = 0, and the sign-agreement swap.
    for &(a, b, c) in &[
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (-1.0, 0.0, -1.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.5, -1.0),
        (-3.0, 2.0, 5.0),
        (1e300, 1e-300, -1e300),
        (1e-300, 1e300, 1e-300),
        (2.0, 1.0, 2.0),
        (0.0, 1.0, 0.0),
        (-0.0, 0.0, 0.0),
        (1.0, 2.0, 1.0),
        (5.0, 0.0, 5.0),
    ] {
        cases.push((a, b, c));
    }
    for _ in 0..4000 {
        let s = |r: &mut Rng| (2.0 * r.uniform() - 1.0) * 10f64.powi((6.0 * r.uniform() - 3.0) as i32);
        let (a, b, c) = (s(&mut r), s(&mut r), s(&mut r));
        cases.push((a, b, c));
        cases.push((a, 0.0, c));
        cases.push((a, b, a));
    }
    for (a, b, c) in cases {
        let tag = format!("a={a:e} b={b:e} c={c:e}");
        let (rt1, rt2, cs1, sn1) = t::dlaev2(a, b, c);
        let (r1, r2, c1, s1) = lapack_ref::dlaev2(a, b, c);
        assert_bits(&format!("dlaev2 {tag}"), &[rt1, rt2, cs1, sn1], &[r1, r2, c1, s1]);
        let (rt1, rt2) = t::dlae2(a, b, c);
        let (r1, r2) = lapack_ref::dlae2(a, b, c);
        assert_bits(&format!("dlae2 {tag}"), &[rt1, rt2], &[r1, r2]);
    }
}

#[test]
fn larrc() {
    if !lapack_ref::require("larrc") {
        return;
    }
    for &n in &[1usize, 2, 3, 7, 13, 31, 64] {
        for kind in 0..5 {
            let (d0, e0) = tri(n, 1_000 + n as u64 + 17 * kind as u64, kind);
            for pivmin in [f64::MIN_POSITIVE, 1e-30, 1e-8] {
                for &(vl, vu) in &[(-1.0, 1.0), (-1e8, 1e8), (0.0, 0.0), (-0.5, 0.25), (1e-9, 2.0)] {
                    for matt in [true, false] {
                        let tag = format!("n={n} kind={kind} matt={matt} vl={vl:e} vu={vu:e} pm={pivmin:e}");
                        let (mut d, mut e) = (d0.clone(), e0.clone());
                        let got = unsafe { t::dlarrc(matt, n, vl, vu, &mut d, &mut e) };
                        let want = lapack_ref::dlarrc(if matt { 'T' } else { 'L' }, n, vl, vu, &d0, &e0, pivmin);
                        assert_eq!(got, want, "dlarrc {tag}");
                    }
                }
            }
        }
    }
}

#[test]
fn larrk() {
    if !lapack_ref::require("larrk") {
        return;
    }
    for &n in &[1usize, 2, 3, 7, 13, 31, 64] {
        for kind in 0..5 {
            let (d0, e0) = tri(n, 2_000 + n as u64 + 17 * kind as u64, kind);
            let e2: Vec<f64> = e0.iter().map(|v| v * v).collect();
            // Gerschgorin-style bracket, as DLARRE builds it.
            let mut gl = f64::INFINITY;
            let mut gu = f64::NEG_INFINITY;
            for i in 0..n {
                let em = if i == 0 { 0.0 } else { e0[i - 1].abs() };
                let ep = if i + 1 < n { e0[i].abs() } else { 0.0 };
                gl = gl.min(d0[i] - em - ep);
                gu = gu.max(d0[i] + em + ep);
            }
            for pivmin in [f64::MIN_POSITIVE, 1e-30, 1e-13] {
                for reltol in [1e-16, 1e-8, 1e-3] {
                    for iw in 1..=n as i32 {
                        let tag = format!("n={n} kind={kind} iw={iw} pm={pivmin:e} rt={reltol:e}");
                        let (mut d, mut e2m) = (d0.clone(), e2.clone());
                        let got = unsafe { t::dlarrk(n, iw, gl, gu, &mut d, &mut e2m, pivmin, reltol) };
                        let want = lapack_ref::dlarrk(n, iw as usize, gl, gu, &d0, &e2, pivmin, reltol);
                        assert_bits(&format!("dlarrk w/werr {tag}"), &[got.0, got.1], &[want.0, want.1]);
                        assert_eq!(got.2, want.2, "dlarrk INFO {tag}");
                        // The device kernel takes ITMAX as an integer argument
                        // (plan §7.2): a libm difference must fail here, not as
                        // a one-ULP eigenvalue.
                        let tnorm = gl.abs().max(gu.abs());
                        assert!(t::larrk_itmax(tnorm, pivmin) >= 2, "itmax {tag}");
                    }
                }
            }
        }
    }
}

#[test]
fn laneg() {
    if !lapack_ref::require("laneg") {
        return;
    }
    // Long enough to cross DLANEG's BLKLEN = 128 blocking in both parts.
    for &n in &[1usize, 2, 5, 13, 127, 128, 129, 200, 257, 400] {
        for kind in 0..5 {
            let (d0, e0) = tri(n, 3_000 + n as u64 + 17 * kind as u64, kind);
            // L D L^T factors: LLD(i) = L(i)**2 * D(i), as DLARRB passes them.
            let l: Vec<f64> = (0..n.max(2) - 1).map(|i| e0[i] / d0[i].max(1e-3)).collect();
            let lld: Vec<f64> = (0..n.max(2) - 1).map(|i| l[i] * l[i] * d0[i]).collect();
            for pivmin in [f64::MIN_POSITIVE, 1e-30] {
                for sigma in [0.0, 0.5, -0.5, 1e-14, 1e6] {
                    for r in [1usize, (n + 1) / 2, n] {
                        let tag = format!("n={n} kind={kind} sigma={sigma:e} r={r}");
                        let (mut d, mut lldm) = (d0.clone(), lld.clone());
                        let got = unsafe { t::dlaneg(n, &mut d, &mut lldm, sigma, pivmin, r) };
                        let want = lapack_ref::dlaneg(n, &d0, &lld, sigma, pivmin, r);
                        assert_eq!(got, want, "dlaneg {tag}");
                    }
                }
            }
        }
    }
}

#[test]
fn lar1v() {
    if !lapack_ref::require("lar1v") {
        return;
    }
    for &n in &[1usize, 2, 3, 7, 13, 31, 64] {
        for kind in 0..5 {
            let (d0, e0) = tri(n, 4_000 + n as u64 + 17 * kind as u64, kind);
            let m1 = n.max(2) - 1;
            let l: Vec<f64> = (0..m1).map(|i| e0[i] / if d0[i].abs() < 1e-3 { 1e-3 } else { d0[i] }).collect();
            let ld: Vec<f64> = (0..m1).map(|i| l[i] * d0[i]).collect();
            let lld: Vec<f64> = (0..m1).map(|i| ld[i] * l[i]).collect();
            let pivmin = f64::MIN_POSITIVE;
            for lambda in [0.0, 0.5, -0.25, 1e-13, 3.0] {
                for gaptol in [0.0, 1e-8, 1e300] {
                    for wantnc in [true, false] {
                        for r_in in [0usize, 1, (n + 1) / 2, n] {
                            let tag = format!("n={n} kind={kind} lam={lambda:e} gt={gaptol:e} nc={wantnc} r={r_in}");
                            let (mut dm, mut lm) = (d0.clone(), pad(&l, n));
                            let (mut ldm, mut lldm) = (pad(&ld, n), pad(&lld, n));
                            let mut z = vec![0.0; n];
                            let mut work = vec![0.0; 4 * n];
                            let got = unsafe {
                                t::dlar1v(
                                    n, 1, n, lambda, &mut dm, &mut lm, &mut ldm, &mut lldm, pivmin, gaptol, &mut z,
                                    wantnc, r_in, &mut work,
                                )
                            };
                            let mut zr = vec![0.0; n];
                            let mut workr = vec![0.0; 4 * n];
                            let want = lapack_ref::dlar1v(
                                n,
                                1,
                                n,
                                lambda,
                                &d0,
                                &pad(&l, n),
                                &pad(&ld, n),
                                &pad(&lld, n),
                                pivmin,
                                gaptol,
                                &mut zr,
                                wantnc,
                                r_in,
                                &mut workr,
                            );
                            assert_bits(&format!("dlar1v Z {tag}"), &z, &zr);
                            assert_bits(&format!("dlar1v WORK {tag}"), &work, &workr);
                            assert_eq!(got.0, want.0, "dlar1v NEGCNT {tag}");
                            assert_bits(
                                &format!("dlar1v scalars {tag}"),
                                &[got.1, got.2, got.6, got.7, got.8],
                                &[want.1, want.2, want.5, want.6, want.7],
                            );
                            assert_eq!(got.3 as i32, want.3, "dlar1v R {tag}");
                            assert_eq!([got.4 as i32, got.5 as i32], want.4, "dlar1v ISUPPZ {tag}");
                        }
                    }
                }
            }
        }
    }
}

/// `n`-long copy of an `n-1`-long factor array (the reference reads `L(N)` as
/// scratch in some branches, so both sides must see the same trailing value).
fn pad(v: &[f64], n: usize) -> Vec<f64> {
    let mut o = vec![0.0; n];
    o[..v.len().min(n)].copy_from_slice(&v[..v.len().min(n)]);
    o
}

/// A qd array for `DLASQ2`: `Z( 2*i-1 ) = q_i > 0`, `Z( 2*i ) = e_i >= 0`,
/// length `4*n` (the reference uses the whole buffer as scratch).
fn qd(n: usize, seed: u64, kind: usize) -> Vec<f64> {
    let mut r = Rng::new(seed);
    let mut z = vec![0.0; 4 * n + 8];
    for i in 1..=n {
        // `DLASQ2` rejects a negative q or e with `XERBLA`, which `STOP`s the
        // process, so every entry here is non-negative by construction.
        z[2 * i - 2] = match kind {
            0 => r.uniform().abs() + 0.5,
            1 => i as f64,
            2 => 1.0,
            3 => 10f64.powi((8.0 * r.uniform() - 4.0) as i32),
            4 => r.uniform().abs() * 1e-6 + 1e-9,
            // Graded, clustered and denormal-adjacent qd arrays: these are
            // what drive `Inf`/`NaN`/`-0.0` through the `dlasq5` `MIN`s.
            5 => 10f64.powi(-(i as i32)),
            6 => {
                if i % 3 == 0 {
                    1e-290
                } else {
                    1.0
                }
            }
            7 => (n + 1 - i) as f64,
            _ => 1.0 + 1e-14 * i as f64,
        };
        if i < n {
            z[2 * i - 1] = match kind {
                0 => r.uniform().abs(),
                1 => 1.0,
                2 => 1e-16,
                3 => 10f64.powi((8.0 * r.uniform() - 4.0) as i32),
                4 => 0.0,
                5 => 10f64.powi(-(i as i32)),
                6 => 1e290,
                7 => (n + 1 - i) as f64,
                _ => 1e-14,
            };
        }
    }
    z
}

#[test]
fn lasq6_lasq5_lasq4() {
    if !lapack_ref::require("lasq6_lasq5_lasq4") {
        return;
    }
    let ieee = lapack_ref::ilaenv(10, "DLASQ2", "N", 1, 2, 3, 4) == 1;
    assert!(ieee, "the oracle reports a non-IEEE machine");
    let eps = t::DLAMCH.iter().find(|(c, _)| *c == 'P').expect("P").1;
    for &n0 in &[4usize, 5, 6, 7, 13, 30] {
        for kind in 0..9 {
            // A ping/pong qd layout: Z = (q1, qq1, e1, ee1, q2, ...).
            let base = qd(n0, 7_000 + n0 as u64 + 31 * kind as u64, kind);
            let mut z0 = vec![0.0; 4 * n0 + 16];
            for i in 1..=n0 {
                z0[4 * i - 4] = base[2 * i - 2];
                z0[4 * i - 3] = base[2 * i - 2];
                z0[4 * i - 2] = base[2 * i - 1];
                z0[4 * i - 1] = base[2 * i - 1];
            }
            for pp in [0usize, 1] {
                let tag0 = format!("n0={n0} kind={kind} pp={pp}");
                // DLASQ6.
                let (mut z, mut zr) = (z0.clone(), z0.clone());
                let got = unsafe { t::dlasq6(1, n0, &mut z, pp) };
                let want = lapack_ref::dlasq6(1, n0, &mut zr, pp);
                assert_bits(&format!("dlasq6 Z {tag0}"), &z, &zr);
                assert_bits(
                    &format!("dlasq6 out {tag0}"),
                    &[got.0, got.1, got.2, got.3, got.4, got.5],
                    &[want.0, want.1, want.2, want.3, want.4, want.5],
                );
                // DLASQ5.
                for tau in [0.0, 1e-30, 0.25, 1.0, 1e3] {
                    for sigma in [0.0, 1.0, 1e-8] {
                        let tag = format!("{tag0} tau={tau:e} sigma={sigma:e}");
                        let (mut z, mut zr) = (z0.clone(), z0.clone());
                        let got = unsafe { t::dlasq5(1, n0, &mut z, pp, tau, sigma, ieee, eps) };
                        let want = lapack_ref::dlasq5(1, n0, &mut zr, pp, tau, sigma, ieee, eps);
                        assert_bits(&format!("dlasq5 Z {tag}"), &z, &zr);
                        assert_bits(
                            &format!("dlasq5 out {tag}"),
                            &[got.0, got.1, got.2, got.3, got.4, got.5, got.6],
                            &[want.0, want.1, want.2, want.3, want.4, want.5, want.6],
                        );
                    }
                }
                // DLASQ4: every N0IN branch and every TTYPE seed.
                let mut r = Rng::new(8_000 + n0 as u64);
                for n0in in [n0, n0 + 1, n0 + 2, n0 + 3] {
                    for ttype in [0i32, -2, -6, -18, -7, -10] {
                        for case in 0..6 {
                            // DMIN chosen to hit the `DMIN == DN`/`DN1`/`DN2`
                            // equality branches as well as the generic one.
                            let (dn, dn1, dn2) = (r.uniform(), r.uniform(), r.uniform());
                            let (dmin1, dmin2) = match case {
                                0 => (dn1, dn2),
                                1 => (dn1, r.uniform()),
                                _ => (r.uniform(), r.uniform()),
                            };
                            let dmin = match case {
                                0 | 1 => dn,
                                2 => dn1,
                                3 => dn2,
                                4 => r.uniform(),
                                _ => -r.uniform(),
                            };
                            let st = (dmin, dmin1, dmin2, dn, dn1, dn2, 0.125, 0.25, ttype);
                            let tag = format!("{tag0} n0in={n0in} ttype={ttype} case={case}");
                            let z = z0.clone();
                            let got = unsafe { t::dlasq4(1, n0, &mut z.clone(), pp, n0in, st) };
                            let want = lapack_ref::dlasq4(1, n0, &z, pp, n0in, st);
                            assert_bits(&format!("dlasq4 tau/g {tag}"), &[got.0, got.1], &[want.0, want.1]);
                            assert_eq!(got.2, want.2, "dlasq4 TTYPE {tag}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn lasq2() {
    if !lapack_ref::require("lasq2") {
        return;
    }
    let ieee = lapack_ref::ilaenv(10, "DLASQ2", "N", 1, 2, 3, 4) == 1;
    for &n in &[1usize, 2, 3, 4, 5, 7, 13, 31, 64, 100, 257] {
        for kind in 0..9 {
            let z0 = qd(n, 9_000 + n as u64 + 31 * kind as u64, kind);
            let tag = format!("n={n} kind={kind}");
            let (mut z, mut zr) = (z0.clone(), z0.clone());
            let info = unsafe { t::dlasq2(n, &mut z, ieee) };
            let infor = lapack_ref::dlasq2(n, &mut zr);
            assert_eq!(info, infor, "dlasq2 INFO {tag}");
            assert_bits(&format!("dlasq2 Z {tag}"), &z, &zr);
        }
    }
}

/// `L D L^T` factors of a tridiagonal, as `DLARRE` hands them to `DLARRB`
/// and `DLARRF`: `d`, `l(i) = e(i)/d(i)`, `ld(i) = l(i)*d(i)`,
/// `lld(i) = ld(i)*l(i)`.
fn ldl(n: usize, d0: &[f64], e0: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let m1 = n.max(2) - 1;
    let mut l = vec![0.0; n];
    let mut ld = vec![0.0; n];
    let mut lld = vec![0.0; n];
    for i in 0..m1 {
        let den = if d0[i].abs() < 1e-3 { 1e-3 } else { d0[i] };
        l[i] = e0[i] / den;
        ld[i] = l[i] * d0[i];
        lld[i] = ld[i] * l[i];
    }
    (d0.to_vec(), l, ld, lld)
}

#[test]
fn larnv_and_laruv() {
    if !lapack_ref::require("larnv_and_laruv") {
        return;
    }
    // Every seed DLARRE uses, plus the wrap-around of the 128-entry table.
    for seed in [[0i32, 0, 0, 1], [1, 1, 1, 1], [4095, 4095, 4095, 4095], [2, 3, 5, 7], [0, 0, 0, 4095]] {
        for &n in &[1usize, 2, 63, 64, 65, 127, 128, 129, 200, 300] {
            // `IDIST = 3` (normal) is the one distribution whose bits are not
            // gated: it goes through `COS`, and Rust's `f64::cos` and the
            // glibc `cos` gfortran calls differ in the last place. Nothing in
            // the MRRR path reaches it — `DLARRE` calls `DLARNV( 2, .. )`.
            for idist in [1i32, 2] {
                let tag = format!("seed={seed:?} n={n} idist={idist}");
                let (mut s1, mut s2) = (seed, seed);
                let (mut x, mut xr) = (vec![0.0; n], vec![0.0; n]);
                unsafe { t::dlarnv(idist, &mut s1, n, &mut x) };
                lapack_ref::dlarnv(idist, &mut s2, n, &mut xr);
                assert_bits(&format!("dlarnv X {tag}"), &x, &xr);
                assert_eq!(s1, s2, "dlarnv ISEED {tag}");
            }
        }
    }
}

#[test]
fn larra_and_larrr() {
    if !lapack_ref::require("larra_and_larrr") {
        return;
    }
    for &n in &[1usize, 2, 3, 7, 13, 31, 64] {
        for kind in 0..5 {
            let (d0, e0) = tri(n, 11_000 + n as u64 + 17 * kind as u64, kind);
            let tnrm = d0.iter().chain(e0.iter()).fold(0.0f64, |a, v| a.max(v.abs()));
            for spltol in [-1e-8, -1e-300, 1e-8, 1e-3, 0.0] {
                let tag = format!("n={n} kind={kind} spltol={spltol:e}");
                let m1 = n.max(2) - 1;
                let e2: Vec<f64> = (0..m1).map(|i| e0[i] * e0[i]).collect();
                let (mut e, mut e2m) = (e0.clone(), e2.clone());
                let mut isplit = vec![0i32; n];
                let got = unsafe { t::dlarra(n, &mut d0.clone(), &mut e, &mut e2m, spltol, tnrm, &mut isplit) };
                let (mut er, mut e2r) = (e0.clone(), e2.clone());
                let mut isplitr = vec![0i32; n];
                let want = lapack_ref::dlarra(n, &d0, &mut er, &mut e2r, spltol, tnrm, &mut isplitr);
                assert_bits(&format!("dlarra E {tag}"), &e, &er);
                assert_bits(&format!("dlarra E2 {tag}"), &e2m, &e2r);
                assert_eq!(isplit, isplitr, "dlarra ISPLIT {tag}");
                assert_eq!(got, want, "dlarra NSPLIT/INFO {tag}");
            }
            // DLARRR only reads D and E.
            let tag = format!("n={n} kind={kind}");
            let mut e = e0.clone();
            let got = unsafe { t::dlarrr(n, &mut d0.clone(), &mut e) };
            let mut er = e0.clone();
            let want = lapack_ref::dlarrr(n, &d0, &mut er);
            assert_eq!(got, want, "dlarrr INFO {tag}");
            assert_bits(&format!("dlarrr E {tag}"), &e, &er);
        }
    }
}

#[test]
fn larrb_and_larrj() {
    if !lapack_ref::require("larrb_and_larrj") {
        return;
    }
    for &n in &[1usize, 2, 3, 7, 13, 31, 64] {
        for kind in 0..5 {
            let (d0, e0) = tri(n, 12_000 + n as u64 + 17 * kind as u64, kind);
            let (dd, _l, _ld, lld) = ldl(n, &d0, &e0);
            let m1 = n.max(2) - 1;
            let e2: Vec<f64> = (0..m1).map(|i| e0[i] * e0[i]).collect();
            let spdiam = d0.iter().fold(0.0f64, |a, v| a.max(v.abs())) * 4.0 + 1.0;
            let pivmin = f64::MIN_POSITIVE;
            // Eigenvalue approximations: crude, but they only have to bracket.
            let w0: Vec<f64> = (0..n).map(|i| d0[i]).collect();
            let werr0: Vec<f64> = (0..n).map(|_| 1e-2).collect();
            let wgap0: Vec<f64> = (0..n).map(|_| 1e-3).collect();
            for &(rtol1, rtol2) in &[(1e-2, 1e-2), (1e-8, 1e-8), (1e-16, 1e-16)] {
                for twist in [-1i32, 1, n as i32, (n as i32 + 1) / 2] {
                    let tag = format!("n={n} kind={kind} rtol={rtol1:e} twist={twist}");
                    let (mut w, mut wg, mut we) = (w0.clone(), wgap0.clone(), werr0.clone());
                    let (mut work, mut iwork) = (vec![0.0; 2 * n], vec![0i32; 2 * n]);
                    let info = unsafe {
                        t::dlarrb(
                            n,
                            &mut dd.clone(),
                            &mut lld.clone(),
                            1,
                            n,
                            rtol1,
                            rtol2,
                            0,
                            &mut w,
                            &mut wg,
                            &mut we,
                            &mut work,
                            &mut iwork,
                            pivmin,
                            spdiam,
                            twist,
                        )
                    };
                    let (mut wr, mut wgr, mut wer) = (w0.clone(), wgap0.clone(), werr0.clone());
                    let (mut workr, mut iworkr) = (vec![0.0; 2 * n], vec![0i32; 2 * n]);
                    let infor = lapack_ref::dlarrb(
                        n,
                        &dd,
                        &lld,
                        1,
                        n,
                        rtol1,
                        rtol2,
                        0,
                        &mut wr,
                        &mut wgr,
                        &mut wer,
                        &mut workr,
                        &mut iworkr,
                        pivmin,
                        spdiam,
                        twist,
                    );
                    assert_eq!(info, infor, "dlarrb INFO {tag}");
                    assert_bits(&format!("dlarrb W {tag}"), &w, &wr);
                    assert_bits(&format!("dlarrb WGAP {tag}"), &wg, &wgr);
                    assert_bits(&format!("dlarrb WERR {tag}"), &we, &wer);
                    assert_bits(&format!("dlarrb WORK {tag}"), &work, &workr);
                    assert_eq!(iwork, iworkr, "dlarrb IWORK {tag}");
                }
                // DLARRJ on the same data.
                let tag = format!("n={n} kind={kind} rtol={rtol1:e}");
                let (mut w, mut we) = (w0.clone(), werr0.clone());
                let (mut work, mut iwork) = (vec![0.0; 2 * n], vec![0i32; 2 * n]);
                let info = unsafe {
                    t::dlarrj(
                        n,
                        &mut d0.clone(),
                        &mut pad(&e2, n),
                        1,
                        n,
                        rtol1,
                        0,
                        &mut w,
                        &mut we,
                        &mut work,
                        &mut iwork,
                        pivmin,
                        spdiam,
                    )
                };
                let (mut wr, mut wer) = (w0.clone(), werr0.clone());
                let (mut workr, mut iworkr) = (vec![0.0; 2 * n], vec![0i32; 2 * n]);
                let infor = lapack_ref::dlarrj(
                    n,
                    &d0,
                    &pad(&e2, n),
                    1,
                    n,
                    rtol1,
                    0,
                    &mut wr,
                    &mut wer,
                    &mut workr,
                    &mut iworkr,
                    pivmin,
                    spdiam,
                );
                assert_eq!(info, infor, "dlarrj INFO {tag}");
                assert_bits(&format!("dlarrj W {tag}"), &w, &wr);
                assert_bits(&format!("dlarrj WERR {tag}"), &we, &wer);
                assert_bits(&format!("dlarrj WORK {tag}"), &work, &workr);
                assert_eq!(iwork, iworkr, "dlarrj IWORK {tag}");
            }
        }
    }
}

#[test]
fn larrf() {
    if !lapack_ref::require("larrf") {
        return;
    }
    for &n in &[2usize, 3, 7, 13, 31] {
        for kind in 0..5 {
            let (d0, e0) = tri(n, 13_000 + n as u64 + 17 * kind as u64, kind);
            let (dd, l, ld, _lld) = ldl(n, &d0, &e0);
            let spdiam = d0.iter().fold(0.0f64, |a, v| a.max(v.abs())) * 4.0 + 1.0;
            let pivmin = f64::MIN_POSITIVE;
            let w0: Vec<f64> = (0..n).map(|i| d0[i]).collect();
            for werr_scale in [1e-2, 1e-8, 1e-14] {
                let werr: Vec<f64> = (0..n).map(|_| werr_scale).collect();
                for wgap_scale in [1e-3, 1e-10, 0.0] {
                    let wgap0: Vec<f64> = (0..n).map(|_| wgap_scale).collect();
                    for &(clgapl, clgapr) in &[(1e-3, 1e-3), (1e-12, 1.0), (1e300, 1e300)] {
                        for &(cs, ce) in &[(1usize, n), (1, 2.min(n)), (n.max(2) - 1, n)] {
                            if ce <= cs {
                                continue;
                            }
                            let tag = format!(
                                "n={n} kind={kind} we={werr_scale:e} wg={wgap_scale:e} cl=({clgapl:e},{clgapr:e}) cluster=({cs},{ce})"
                            );
                            let mut wg = wgap0.clone();
                            let (mut dp, mut lp, mut work) = (vec![0.0; n], vec![0.0; n], vec![0.0; 2 * n]);
                            let got = unsafe {
                                t::dlarrf(
                                    n,
                                    &mut dd.clone(),
                                    &mut l.clone(),
                                    &mut ld.clone(),
                                    cs,
                                    ce,
                                    &mut w0.clone(),
                                    &mut wg,
                                    &mut werr.clone(),
                                    spdiam,
                                    clgapl,
                                    clgapr,
                                    pivmin,
                                    &mut dp,
                                    &mut lp,
                                    &mut work,
                                )
                            };
                            let mut wgr = wgap0.clone();
                            let (mut dpr, mut lpr, mut workr) = (vec![0.0; n], vec![0.0; n], vec![0.0; 2 * n]);
                            let want = lapack_ref::dlarrf(
                                n, &dd, &l, &ld, cs, ce, &w0, &mut wgr, &werr, spdiam, clgapl, clgapr, pivmin,
                                &mut dpr, &mut lpr, &mut workr,
                            );
                            assert_eq!(got.1, want.1, "dlarrf INFO {tag}");
                            assert_bits(&format!("dlarrf SIGMA {tag}"), &[got.0], &[want.0]);
                            if got.1 == 0 {
                                assert_bits(&format!("dlarrf DPLUS {tag}"), &dp, &dpr);
                                assert_bits(&format!("dlarrf LPLUS {tag}"), &lp[..n - 1], &lpr[..n - 1]);
                            }
                            assert_bits(&format!("dlarrf WGAP {tag}"), &wg, &wgr);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn larre() {
    if !lapack_ref::require("larre") {
        return;
    }
    for &n in &[1usize, 2, 3, 5, 7, 13, 31, 64, 129] {
        for kind in 0..5 {
            let (d0, e0) = tri(n, 14_000 + n as u64 + 17 * kind as u64, kind);
            // DSTEMR's own call shape: E has N entries (DLARRE writes E(N)).
            let mut e_full = vec![0.0; n];
            e_full[..n.max(2) - 1].copy_from_slice(&e0[..n.max(2) - 1]);
            if n == 1 {
                e_full[0] = 0.0;
            }
            let e2_full: Vec<f64> = e_full.iter().map(|v| v * v).collect();
            for &(rtol1, rtol2) in &[(1e-3, 1e-3), (1e-8, 1e-8)] {
                for spltol in [-1e-8, 1e-8, 0.0] {
                    let tag = format!("n={n} kind={kind} rtol={rtol1:e} spltol={spltol:e}");
                    let mk = || {
                        (
                            d0.clone(),
                            e_full.clone(),
                            e2_full.clone(),
                            vec![0i32; n],
                            vec![0.0; n],
                            vec![0.0; n],
                            vec![0.0; n],
                            vec![0i32; n],
                            vec![0i32; n],
                            vec![0.0; 2 * n],
                            vec![0.0; 6 * n],
                            vec![0i32; 5 * n],
                        )
                    };
                    let (mut d, mut e, mut e2, mut isp, mut w, mut we, mut wg, mut ib, mut iw, mut ge, mut wk, mut jw) =
                        mk();
                    let got = unsafe {
                        t::dlarre_all(
                            n, 0.0, 0.0, &mut d, &mut e, &mut e2, rtol1, rtol2, spltol, &mut isp, &mut w, &mut we,
                            &mut wg, &mut ib, &mut iw, &mut ge, &mut wk, &mut jw,
                        )
                    };
                    let (
                        mut dr,
                        mut er,
                        mut e2r,
                        mut ispr,
                        mut wr,
                        mut wer,
                        mut wgr,
                        mut ibr,
                        mut iwr,
                        mut ger,
                        mut wkr,
                        mut jwr,
                    ) = mk();
                    let want = lapack_ref::dlarre(
                        'A', n, 0.0, 0.0, 1, n, &mut dr, &mut er, &mut e2r, rtol1, rtol2, spltol, &mut ispr, &mut wr,
                        &mut wer, &mut wgr, &mut ibr, &mut iwr, &mut ger, &mut wkr, &mut jwr,
                    );
                    assert_eq!(got.5, want.5, "dlarre INFO {tag}");
                    assert_eq!(got.2, want.2, "dlarre NSPLIT {tag}");
                    assert_eq!(got.3, want.3, "dlarre M {tag}");
                    assert_bits(
                        &format!("dlarre VL/VU/PIVMIN {tag}"),
                        &[got.0, got.1, got.4],
                        &[want.0, want.1, want.4],
                    );
                    assert_eq!(isp, ispr, "dlarre ISPLIT {tag}");
                    assert_eq!(ib, ibr, "dlarre IBLOCK {tag}");
                    assert_eq!(iw, iwr, "dlarre INDEXW {tag}");
                    assert_bits(&format!("dlarre D {tag}"), &d, &dr);
                    assert_bits(&format!("dlarre E {tag}"), &e, &er);
                    assert_bits(&format!("dlarre E2 {tag}"), &e2, &e2r);
                    assert_bits(&format!("dlarre W {tag}"), &w, &wr);
                    assert_bits(&format!("dlarre WERR {tag}"), &we, &wer);
                    assert_bits(&format!("dlarre WGAP {tag}"), &wg, &wgr);
                    assert_bits(&format!("dlarre GERS {tag}"), &ge, &ger);
                }
            }
        }
    }
}
