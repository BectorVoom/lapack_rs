//! Component routines against Reference-LAPACK 3.12.1, bit for bit, so a
//! driver mismatch localises to one routine: `dsytrd`/`zhetrd` (d, e, tau,
//! reflectors) at LWORK values driving the blocked, reduced-NB and unblocked
//! paths; `dormtr`/`zunmtr`; `dsteqr`, `dsterf`, `dstedc` on tridiagonal
//! fixtures; `dlaed4` per root.

mod common;
use common::*;
use lapack_rs::testing as t;

fn lwork_variants(n: usize) -> Vec<usize> {
    let mut v = vec![(n * 32).max(1), (n * 8).max(1), (n + n / 2).max(1), 1];
    v.dedup();
    v
}

#[test]
fn sytrd_and_ormtr() {
    if !lapack_ref::require("sytrd_and_ormtr") {
        return;
    }
    for &n in &[1usize, 2, 16, 31, 32, 33, 65, 100, 129] {
        let a0 = random_sym(n, 3000 + n as u64);
        for upper in [true, false] {
            let u = if upper { 'U' } else { 'L' };
            for lw in lwork_variants(n) {
                let tag = format!("n={n} {u} lwork={lw}");
                let (mut a, mut ar) = (a0.clone(), a0.clone());
                let (mut d, mut dr) = (vec![0.0; n], vec![0.0; n]);
                let (mut e, mut er) = (vec![0.0; n.max(2) - 1], vec![0.0; n.max(2) - 1]);
                let (mut tau, mut taur) = (vec![0.0; n.max(2) - 1], vec![0.0; n.max(2) - 1]);
                let (mut w, mut wr) = (vec![0.0; lw], vec![0.0; lw]);
                unsafe { t::dsytrd(upper, n, &mut a, &mut d, &mut e, &mut tau, &mut w) };
                lapack_ref::dsytrd(u, n, &mut ar, &mut dr, &mut er, &mut taur, &mut wr);
                assert_bits(&format!("dsytrd A {tag}"), &a, &ar);
                assert_bits(&format!("dsytrd d {tag}"), &d, &dr);
                assert_bits(&format!("dsytrd e {tag}"), &e, &er);
                assert_bits(&format!("dsytrd tau {tag}"), &tau, &taur);
                // DORMTR on a dense C with several LWORK values (NB reduction).
                for lw2 in [(n * 32 + 4160).max(1), (n * 3 + 4160).max(1), n.max(1)] {
                    let c0 = random_sym(n, 4000 + n as u64);
                    let (mut c, mut cr) = (c0.clone(), c0.clone());
                    let (mut ww, mut wwr) = (vec![0.0; lw2], vec![0.0; lw2]);
                    let (mut a2, mut tau2) = (a.clone(), tau.clone());
                    unsafe { t::dormtr_ln(upper, n, &mut a2, &mut tau2, &mut c, &mut ww) };
                    lapack_ref::dormtr_ln(u, n, &mut ar.clone(), &mut taur.clone(), &mut cr, &mut wwr);
                    assert_bits(&format!("dormtr C {tag} lwork2={lw2}"), &c, &cr);
                }
            }
        }
    }
}

#[test]
fn hetrd_and_unmtr() {
    if !lapack_ref::require("hetrd_and_unmtr") {
        return;
    }
    for &n in &[1usize, 2, 16, 31, 33, 65, 100] {
        let (re0, im0) = random_herm(n, 5000 + n as u64);
        for upper in [true, false] {
            let u = if upper { 'U' } else { 'L' };
            for lw in lwork_variants(n) {
                let tag = format!("n={n} {u} lwork={lw}");
                let (mut re, mut im) = (re0.clone(), im0.clone());
                let mut ac = to_c64(&re0, &im0);
                let m1 = n.max(2) - 1;
                let (mut d, mut dr) = (vec![0.0; n], vec![0.0; n]);
                let (mut e, mut er) = (vec![0.0; m1], vec![0.0; m1]);
                let (mut tre, mut tim) = (vec![0.0; m1], vec![0.0; m1]);
                let mut tauc = vec![lapack_ref::C64::default(); m1];
                let mut w = vec![0.0; 2 * lw];
                let mut wc = vec![lapack_ref::C64::default(); lw];
                unsafe { t::zhetrd(upper, n, &mut re, &mut im, &mut d, &mut e, &mut tre, &mut tim, &mut w) };
                lapack_ref::zhetrd(u, n, &mut ac, &mut dr, &mut er, &mut tauc, &mut wc);
                let (rr, ri) = from_c64(&ac);
                assert_bits(&format!("zhetrd A.re {tag}"), &re, &rr);
                assert_bits(&format!("zhetrd A.im {tag}"), &im, &ri);
                assert_bits(&format!("zhetrd d {tag}"), &d, &dr);
                assert_bits(&format!("zhetrd e {tag}"), &e, &er);
                let (trr, tri) = from_c64(&tauc);
                assert_bits(&format!("zhetrd tau.re {tag}"), &tre, &trr);
                assert_bits(&format!("zhetrd tau.im {tag}"), &tim, &tri);
                for lw2 in [(n * 32 + 4160).max(1), n.max(1)] {
                    let (cre0, cim0) = random_herm(n, 6000 + n as u64);
                    let (mut cre, mut cim) = (cre0.clone(), cim0.clone());
                    let mut cc = to_c64(&cre0, &cim0);
                    let mut ww = vec![0.0; 2 * lw2];
                    let mut wwc = vec![lapack_ref::C64::default(); lw2];
                    let (mut a2r, mut a2i, mut t2r, mut t2i) = (re.clone(), im.clone(), tre.clone(), tim.clone());
                    unsafe {
                        t::zunmtr_ln(upper, n, &mut a2r, &mut a2i, &mut t2r, &mut t2i, &mut cre, &mut cim, &mut ww)
                    };
                    lapack_ref::zunmtr_ln(u, n, &mut ac.clone(), &mut tauc.clone(), &mut cc, &mut wwc);
                    let (ccr, cci) = from_c64(&cc);
                    assert_bits(&format!("zunmtr C.re {tag} lwork2={lw2}"), &cre, &ccr);
                    assert_bits(&format!("zunmtr C.im {tag} lwork2={lw2}"), &cim, &cci);
                }
            }
        }
    }
}

#[test]
fn steqr_sterf_stedc_fixtures() {
    if !lapack_ref::require("steqr_sterf_stedc_fixtures") {
        return;
    }
    for &n in &[1usize, 2, 3, 15, 25, 26, 27, 50, 64, 100, 257, 513] {
        for (name, d0, e0) in tridiag_fixtures(n) {
            let tag = format!("{name} n={n}");
            // dsteqr('I')
            if n <= 257 {
                let (mut d, mut dr) = (d0.clone(), d0.clone());
                let (mut e, mut er) = (e0.clone(), e0.clone());
                e.push(0.0);
                er.push(0.0);
                let (mut z, mut zr) = (vec![0.0; n * n], vec![0.0; n * n]);
                let (mut w, mut wr) = (vec![0.0; (2 * n).max(2)], vec![0.0; (2 * n).max(2)]);
                let info = unsafe { t::dsteqr_i(n, &mut d, &mut e, &mut z, &mut w) };
                let infor = lapack_ref::dsteqr('I', n, &mut dr, &mut er, &mut zr, &mut wr);
                assert_eq!(info as i32, infor, "dsteqr INFO {tag}");
                assert_bits(&format!("dsteqr d {tag}"), &d, &dr);
                assert_bits(&format!("dsteqr Z {tag}"), &z, &zr);
            }
            // dsterf
            {
                let (mut d, mut dr) = (d0.clone(), d0.clone());
                let (mut e, mut er) = (e0.clone(), e0.clone());
                e.push(0.0);
                er.push(0.0);
                let info = unsafe { t::dsterf(n, &mut d, &mut e) };
                let infor = lapack_ref::dsterf(n, &mut dr, &mut er);
                assert_eq!(info as i32, infor, "dsterf INFO {tag}");
                assert_bits(&format!("dsterf d {tag}"), &d, &dr);
            }
            // dstedc('I')
            {
                let (lw, liw) = lapack_rs::stedc_work_len(n);
                let (mut d, mut dr) = (d0.clone(), d0.clone());
                let (mut e, mut er) = (e0.clone(), e0.clone());
                e.push(0.0);
                er.push(0.0);
                let (mut z, mut zr) = (vec![0.0; n * n], vec![0.0; n * n]);
                let (mut w, mut wr) = (vec![0.0; lw], vec![0.0; lw]);
                let (mut iw, mut iwr) = (vec![0usize; liw], vec![0i32; liw]);
                let info = unsafe { t::dstedc_i(n, &mut d, &mut e, &mut z, &mut w, &mut iw) };
                let infor = lapack_ref::dstedc('I', n, &mut dr, &mut er, &mut zr, &mut wr, &mut iwr);
                assert_eq!(info as i32, infor, "dstedc INFO {tag}");
                assert_bits(&format!("dstedc d {tag}"), &d, &dr);
                assert_bits(&format!("dstedc Z {tag}"), &z, &zr);
            }
        }
    }
}

#[test]
fn laed4_per_root() {
    if !lapack_ref::require("laed4_per_root") {
        return;
    }
    let mut r = Rng::new(424242);
    for &k in &[1usize, 2, 3, 4, 10, 57, 200] {
        for case in 0..4 {
            // Strictly increasing D, normalised Z, positive RHO.
            let mut d: Vec<f64> = (0..k).map(|i| i as f64 + 0.5 * r.uniform()).collect();
            if case == 1 {
                // Near-coincident poles.
                for i in (1..k).step_by(2) {
                    d[i] = d[i - 1] + 1e-12;
                }
            }
            d.sort_by(f64::total_cmp);
            let mut z: Vec<f64> = (0..k).map(|_| r.uniform()).collect();
            if case == 2 {
                z[k / 2] *= 1e-9;
            }
            let nrm = z.iter().map(|v| v * v).sum::<f64>().sqrt();
            z.iter_mut().for_each(|v| *v /= nrm);
            let rho = [1.0, 1e-8, 3.5, 100.0][case];
            for i in 1..=k {
                let (mut dm, mut zm) = (d.clone(), z.clone());
                let (mut del, mut delr) = (vec![0.0; k], vec![0.0; k]);
                let (dlam, info) = unsafe { t::dlaed4(k, i, &mut dm, &mut zm, &mut del, rho) };
                let (dlamr, infor) = lapack_ref::dlaed4(k, i, &d, &z, &mut delr, rho);
                let tag = format!("k={k} case={case} i={i}");
                assert_eq!(info as i32, infor, "dlaed4 INFO {tag}");
                assert_eq!(dlam.to_bits(), dlamr.to_bits(), "dlaed4 DLAM {tag}");
                assert_bits(&format!("dlaed4 DELTA {tag}"), &del, &delr);
            }
        }
    }
}
