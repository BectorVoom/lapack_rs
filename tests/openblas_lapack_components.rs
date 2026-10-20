//! **OpenBLAS track.** `DSYEVD`'s LAPACK components in the OpenBLAS flavour
//! against the same routines of scipy's OpenBLAS 0.3.30, bit for bit: the
//! bisection gate for `tests/openblas_syevd_bitexact.rs`.

mod common;
use common::*;
use lapack_rs::testing::openblas as tob;
use openblas_ref as ob;

const NS: &[usize] = &[2, 3, 16, 17, 25, 26, 27, 32, 33, 34, 40, 63, 64, 65, 100, 129, 200];

#[test]
fn openblas_dsytrd() {
    if !ob::require("openblas_dsytrd") {
        return;
    }
    for &n in NS {
        for upper in [false, true] {
            let a0 = random_sym(n, 70 + n as u64);
            let lwork = n * 32;
            let (mut a, mut d, mut e, mut tau, mut work) =
                (a0.clone(), vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; lwork]);
            // SAFETY: shapes as allocated.
            unsafe { tob::dsytrd(upper, n, &mut a, &mut d, &mut e, &mut tau, &mut work) };
            let (mut ar, mut dr, mut er, mut taur, mut workr) =
                (a0.clone(), vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; lwork]);
            let (ul, n_, lw, mut info) = (ob::ch(if upper { 'U' } else { 'L' }), n as i32, lwork as i32, 0);
            // SAFETY: as above.
            unsafe {
                ob::dsytrd(
                    &ul,
                    &n_,
                    ar.as_mut_ptr(),
                    &n_,
                    dr.as_mut_ptr(),
                    er.as_mut_ptr(),
                    taur.as_mut_ptr(),
                    workr.as_mut_ptr(),
                    &lw,
                    &mut info,
                    1,
                )
            };
            let tag = format!("dsytrd n={n} upper={upper}");
            assert_bits(&format!("{tag} a"), &a, &ar);
            assert_bits(&format!("{tag} d"), &d, &dr);
            assert_bits(&format!("{tag} e"), &e[..n - 1], &er[..n - 1]);
            assert_bits(&format!("{tag} tau"), &tau[..n - 1], &taur[..n - 1]);
        }
    }
}

#[test]
fn openblas_dstedc() {
    if !ob::require("openblas_dstedc") {
        return;
    }
    for &n in NS {
        for (name, d0, e0) in tridiag_fixtures(n) {
            let (lw, liw) = lapack_rs::stedc_work_len(n);
            let (mut d, mut e, mut z) = (d0.clone(), e0.clone(), vec![0.0; n * n]);
            let (mut work, mut iwork) = (vec![0.0; lw], vec![0usize; liw]);
            // SAFETY: shapes as allocated.
            let info = unsafe { tob::dstedc_i(n, &mut d, &mut e, &mut z, &mut work, &mut iwork) };
            let (mut dr, mut er, mut zr) = (d0.clone(), e0.clone(), vec![0.0; n * n]);
            let (mut workr, mut iworkr) = (vec![0.0; lw], vec![0i32; liw]);
            let (cz, n_, lw_, liw_, mut infor) = (ob::ch('I'), n as i32, lw as i32, liw as i32, 0);
            er.push(0.0);
            // SAFETY: as above.
            unsafe {
                ob::dstedc(
                    &cz,
                    &n_,
                    dr.as_mut_ptr(),
                    er.as_mut_ptr(),
                    zr.as_mut_ptr(),
                    &n_,
                    workr.as_mut_ptr(),
                    &lw_,
                    iworkr.as_mut_ptr(),
                    &liw_,
                    &mut infor,
                    1,
                )
            };
            let tag = format!("dstedc {name} n={n}");
            assert_eq!(info as i32, infor, "{tag} info");
            assert_bits(&format!("{tag} d"), &d, &dr);
            assert_bits(&format!("{tag} z"), &z, &zr);
        }
    }
}

#[test]
fn openblas_dormtr() {
    if !ob::require("openblas_dormtr") {
        return;
    }
    for &n in NS {
        for upper in [false, true] {
            // Reflectors from our own (gated) dsytrd, applied to a random C.
            let mut a = random_sym(n, 90 + n as u64);
            let (mut d, mut e, mut tau, mut work) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n * 32]);
            // SAFETY: shapes as allocated.
            unsafe { tob::dsytrd(upper, n, &mut a, &mut d, &mut e, &mut tau, &mut work) };
            let c0: Vec<f64> = {
                let mut r = Rng::new(5 + n as u64);
                (0..n * n).map(|_| r.uniform()).collect()
            };
            for lwork in [n.max(1), n * 32 + 4160] {
                let (mut aa, mut c, mut w1) = (a.clone(), c0.clone(), vec![0.0; lwork]);
                // SAFETY: as above.
                unsafe { tob::dormtr_ln(upper, n, &mut aa, &mut tau.clone(), &mut c, &mut w1) };
                let (mut ar, mut cr, mut wr, mut taur) = (a.clone(), c0.clone(), vec![0.0; lwork], tau.clone());
                let (s, ul, t, n_, lw, mut info) =
                    (ob::ch('L'), ob::ch(if upper { 'U' } else { 'L' }), ob::ch('N'), n as i32, lwork as i32, 0);
                // SAFETY: as above.
                unsafe {
                    ob::dormtr(
                        &s,
                        &ul,
                        &t,
                        &n_,
                        &n_,
                        ar.as_mut_ptr(),
                        &n_,
                        taur.as_mut_ptr(),
                        cr.as_mut_ptr(),
                        &n_,
                        wr.as_mut_ptr(),
                        &lw,
                        &mut info,
                        1,
                        1,
                        1,
                    )
                };
                assert_bits(&format!("dormtr n={n} upper={upper} lwork={lwork}"), &c, &cr);
            }
        }
    }
}
