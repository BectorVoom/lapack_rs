//! Oracle gate: `ZGEMM`, `ZHER2K` and `ZTRMM` against scipy's OpenBLAS 0.3.30 (SkylakeX, one
//! thread), every output bit including untouched entries.
#![allow(clippy::too_many_arguments)]
mod common;
#[path = "common/ob_support.rs"]
mod support;
use common::ob::*;
use lapack_rs::testing::Tr;
use lapack_rs::testing::V;
use lapack_rs::testing::openblas::zlevel3 as l3;
use lapack_rs::testing::{Z, ZV};
use openblas_ref::{self as ob, C64};

fn to_c64(re: &[f64], im: &[f64]) -> Vec<C64> {
    re.iter().zip(im).map(|(&re, &im)| C64 { re, im }).collect()
}
fn flat(x: &[C64]) -> Vec<f64> {
    x.iter().flat_map(|z| [z.re, z.im]).collect()
}
fn flat2(re: &[f64], im: &[f64]) -> Vec<f64> {
    re.iter().zip(im).flat_map(|(a, b)| [*a, *b]).collect()
}
fn tr_ch(t: Tr) -> u8 {
    match t {
        Tr::N => b'N',
        Tr::T => b'T',
        Tr::C => b'C',
    }
}

/// Random complex matrix storage (planar), edge values on odd trials.
fn cmat(r: &mut Rng, len: usize, edge: bool) -> (Vec<f64>, Vec<f64>) {
    if edge { (r.vec_edge(len), r.vec_edge(len)) } else { (r.vec(len), r.vec(len)) }
}

#[test]
fn gotoblas_params_are_the_hardcoded_ones() {
    if !ob::require("gotoblas_params_are_the_hardcoded_ones") {
        return;
    }
    let _ = support::base();
    assert_eq!(support::gotoblas_int(0xd90) as usize, l3::ZGEMM_P);
    assert_eq!(support::gotoblas_int(0xd94) as usize, l3::ZGEMM_Q);
    assert_eq!(support::gotoblas_int(0xd98) as usize, l3::ZGEMM_R);
    assert_eq!(support::gotoblas_int(0xd9c) as usize, l3::ZGEMM_UNROLL_M);
    assert_eq!(support::gotoblas_int(0xda0) as usize, l3::ZGEMM_UNROLL_N);
    assert_eq!(support::gotoblas_int(0xda4) as usize, l3::ZGEMM_UNROLL_MN);
}

fn scalars(r: &mut Rng) -> (Vec<Z>, Vec<Z>) {
    let alphas = vec![Z::new(1.0, 0.0), Z::new(-1.0, 0.0), Z::new(r.uniform(), r.uniform()), Z::new(0.0, 0.0)];
    let betas = vec![Z::new(0.0, 0.0), Z::new(1.0, 0.0), Z::new(r.uniform(), r.uniform())];
    (alphas, betas)
}

/// One ZGEMM comparison; `extra` is added to every leading dimension.
fn zgemm_case(r: &mut Rng, ta: Tr, tb: Tr, m: usize, n: usize, k: usize, alpha: Z, beta: Z, extra: usize, edge: bool) {
    let (ra, ca) = if ta == Tr::N { (m, k) } else { (k, m) };
    let (rb, cb) = if tb == Tr::N { (k, n) } else { (n, k) };
    let lda = ra.max(1) + extra;
    let ldb = rb.max(1) + extra;
    let ldc = m.max(1) + extra;
    let (mut ar, mut ai) = cmat(r, lda * ca.max(1), edge);
    let (mut br, mut bi) = cmat(r, ldb * cb.max(1), edge);
    let (mut cr, mut ci) = cmat(r, ldc * n.max(1), edge);
    let a64 = to_c64(&ar, &ai);
    let b64 = to_c64(&br, &bi);
    let mut want = to_c64(&cr, &ci);
    let (m_, n_, k_, lda_, ldb_, ldc_) = (m as i32, n as i32, k as i32, lda as i32, ldb as i32, ldc as i32);
    let (al, be) = (C64 { re: alpha.re, im: alpha.im }, C64 { re: beta.re, im: beta.im });
    // SAFETY: shapes per the ZGEMM contract.
    unsafe {
        ob::zgemm(
            &tr_ch(ta),
            &tr_ch(tb),
            &m_,
            &n_,
            &k_,
            &al,
            a64.as_ptr(),
            &lda_,
            b64.as_ptr(),
            &ldb_,
            &be,
            want.as_mut_ptr(),
            &ldc_,
            1,
            1,
        )
    };
    let a = ZV::new(V::new(&mut ar), V::new(&mut ai));
    let b = ZV::new(V::new(&mut br), V::new(&mut bi));
    let c = ZV::new(V::new(&mut cr), V::new(&mut ci));
    // SAFETY: as above.
    unsafe { l3::zgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) };
    assert_bits(
        &format!("zgemm {ta:?}{tb:?} m={m} n={n} k={k} alpha={alpha:?} beta={beta:?} extra={extra} edge={edge}"),
        &flat2(&cr, &ci),
        &flat(&want),
    );
}

const TRANS: [Tr; 3] = [Tr::N, Tr::T, Tr::C];
const GRID3: [usize; 23] = [0, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 257];

#[test]
fn zgemm_bitexact_small_grid() {
    if !ob::require("zgemm_bitexact_small_grid") {
        return;
    }
    let mut r = Rng::new(101);
    let (alphas, betas) = scalars(&mut r);
    let small = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13];
    for ta in TRANS {
        for tb in TRANS {
            for &m in &small {
                for &n in &small {
                    for &k in &small {
                        let alpha = alphas[(m + 3 * n + 7 * k) % alphas.len()];
                        let beta = betas[(m + n + k) % betas.len()];
                        let extra = (m + n) % 2 * 3;
                        zgemm_case(&mut r, ta, tb, m, n, k, alpha, beta, extra, (m * n + k) % 2 == 1);
                    }
                }
            }
        }
    }
}

#[test]
fn zgemm_bitexact_scalars() {
    if !ob::require("zgemm_bitexact_scalars") {
        return;
    }
    let mut r = Rng::new(102);
    let (alphas, betas) = scalars(&mut r);
    let mut betas = betas;
    betas.push(Z::new(-0.0, 0.0));
    betas.push(Z::new(0.0, 1.0));
    for ta in TRANS {
        for tb in TRANS {
            for &alpha in &alphas {
                for &beta in &betas {
                    for (m, n, k) in [(5, 7, 3), (17, 9, 33), (4, 13, 1)] {
                        for extra in [0, 2] {
                            zgemm_case(&mut r, ta, tb, m, n, k, alpha, beta, extra, extra == 2);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn zgemm_bitexact_block_grid() {
    if !ob::require("zgemm_bitexact_block_grid") {
        return;
    }
    let mut r = Rng::new(103);
    let (alphas, betas) = scalars(&mut r);
    let mut count = 0usize;
    for (ti, ta) in TRANS.into_iter().enumerate() {
        for (tj, tb) in TRANS.into_iter().enumerate() {
            // Sparse crossing of the grid: rotate the offsets per trans pair.
            for (i, &m) in GRID3.iter().enumerate() {
                let n = GRID3[(i * 7 + ti * 5 + tj * 3 + 4) % GRID3.len()];
                let k = GRID3[(i * 11 + ti * 3 + tj * 7 + 9) % GRID3.len()];
                let alpha = alphas[(i + ti) % alphas.len()];
                let beta = betas[(i + tj) % betas.len()];
                zgemm_case(&mut r, ta, tb, m, n, k, alpha, beta, i % 2, i % 3 == 0);
                count += 1;
            }
        }
    }
    eprintln!("zgemm block grid: {count} cases");
}

#[test]
fn zgemm_bitexact_large() {
    if !ob::require("zgemm_bitexact_large") {
        return;
    }
    let mut r = Rng::new(104);
    let (alphas, _) = scalars(&mut r);
    let beta = Z::new(r.uniform(), r.uniform());
    // m crosses P=256 (min_i halving) and 2P=512 (min_i = P); k crosses Q=128 and 2Q=256.
    let cases = [
        (257usize, 31usize, 129usize, Tr::N, Tr::N),
        (300, 33, 255, Tr::N, Tr::C),
        (511, 17, 64, Tr::C, Tr::N),
        (513, 9, 300, Tr::C, Tr::C),
        (520, 40, 257, Tr::T, Tr::N),
        (64, 300, 520, Tr::N, Tr::T),
        (129, 128, 127, Tr::T, Tr::C),
        (400, 7, 400, Tr::C, Tr::T),
        (500, 500, 32, Tr::N, Tr::N),
        (100, 490, 130, Tr::C, Tr::C),
    ];
    for (i, (m, n, k, ta, tb)) in cases.into_iter().enumerate() {
        zgemm_case(&mut r, ta, tb, m, n, k, alphas[i % 3], beta, i % 2, i % 2 == 1);
    }
}

/// `n > GEMM_R = 16112` crosses the `js` blocking of `level3.c`.
#[test]
fn zgemm_bitexact_gemm_r_blocking() {
    if !ob::require("zgemm_bitexact_gemm_r_blocking") {
        return;
    }
    let mut r = Rng::new(105);
    let (alphas, betas) = scalars(&mut r);
    for (i, (ta, tb)) in [(Tr::N, Tr::N), (Tr::C, Tr::T), (Tr::T, Tr::C)].into_iter().enumerate() {
        for (m, k) in [(2usize, 3usize), (5, 130)] {
            zgemm_case(&mut r, ta, tb, m, l3::ZGEMM_R + 33, k, alphas[(i + 2) % 3], betas[2], i % 2, m == 5);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ZHER2K
// ---------------------------------------------------------------------------------------------

fn zher2k_case(r: &mut Rng, upper: bool, n: usize, k: usize, alpha: Z, beta: f64, extra: usize, edge: bool) {
    let lda = n.max(1) + extra;
    let ldc = n.max(1) + extra;
    let (mut ar, mut ai) = cmat(r, lda * k.max(1), edge);
    let (mut br, mut bi) = cmat(r, lda * k.max(1), edge);
    let (mut cr, mut ci) = cmat(r, ldc * n.max(1), edge);
    let a64 = to_c64(&ar, &ai);
    let b64 = to_c64(&br, &bi);
    let mut want = to_c64(&cr, &ci);
    let (n_, k_, lda_, ldc_) = (n as i32, k as i32, lda as i32, ldc as i32);
    let al = C64 { re: alpha.re, im: alpha.im };
    let uplo = if upper { b'U' } else { b'L' };
    // SAFETY: shapes per the ZHER2K contract (TRANS='N', ldb = lda).
    unsafe {
        ob::zher2k(
            &uplo,
            &b'N',
            &n_,
            &k_,
            &al,
            a64.as_ptr(),
            &lda_,
            b64.as_ptr(),
            &lda_,
            &beta,
            want.as_mut_ptr(),
            &ldc_,
            1,
            1,
        )
    };
    let a = ZV::new(V::new(&mut ar), V::new(&mut ai));
    let b = ZV::new(V::new(&mut br), V::new(&mut bi));
    let c = ZV::new(V::new(&mut cr), V::new(&mut ci));
    // SAFETY: as above.
    unsafe { l3::zher2k_n(upper, n, k, alpha, a, lda, b, lda, beta, c, ldc) };
    assert_bits(
        &format!("zher2k upper={upper} n={n} k={k} alpha={alpha:?} beta={beta} extra={extra} edge={edge}"),
        &flat2(&cr, &ci),
        &flat(&want),
    );
}

#[test]
fn zher2k_bitexact_zhetrd_shape() {
    if !ob::require("zher2k_bitexact_zhetrd_shape") {
        return;
    }
    let mut r = Rng::new(201);
    // ZHETRD: ZHER2K(UPLO, 'No transpose', N-I+1.., NB, -CONE, A, LDA, WORK, LDWORK, ONE, A, LDA).
    for upper in [true, false] {
        for n in [
            0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 257, 258, 300,
            383, 511, 512, 513,
        ] {
            for k in [32usize, 1, 7] {
                zher2k_case(&mut r, upper, n, k, Z::new(-1.0, 0.0), 1.0, n % 3, n % 2 == 1);
            }
        }
    }
}

#[test]
fn zher2k_bitexact_grid() {
    if !ob::require("zher2k_bitexact_grid") {
        return;
    }
    let mut r = Rng::new(202);
    let alphas = [Z::new(-1.0, 0.0), Z::new(1.0, 0.0), Z::new(r.uniform(), r.uniform()), Z::new(0.0, 0.0)];
    let betas = [1.0, 0.0, r.uniform(), -0.0];
    let mut count = 0;
    for upper in [true, false] {
        for (i, &n) in GRID3.iter().enumerate() {
            for (j, &k) in GRID3.iter().enumerate() {
                if (i * 5 + j * 3) % 4 != 0 && n > 9 && k > 9 {
                    continue; // sparse crossing above the small sizes
                }
                let alpha = alphas[(i + j) % alphas.len()];
                let beta = betas[(i * 3 + j) % betas.len()];
                zher2k_case(&mut r, upper, n, k, alpha, beta, (i + j) % 2 * 2, (i + j) % 3 == 0);
                count += 1;
            }
        }
    }
    for upper in [true, false] {
        for (n, k) in [(300usize, 260usize), (520, 40), (129, 300), (400, 129)] {
            zher2k_case(&mut r, upper, n, k, alphas[2], betas[2], 1, false);
            count += 1;
        }
    }
    eprintln!("zher2k grid: {count} cases");
}

// ---------------------------------------------------------------------------------------------
// ZTRMM
// ---------------------------------------------------------------------------------------------

fn ztrmm_case(
    r: &mut Rng,
    left: bool,
    upper: bool,
    trans: Tr,
    nounit: bool,
    m: usize,
    n: usize,
    alpha: Z,
    extra: usize,
    edge: bool,
) {
    let na = if left { m } else { n };
    let lda = na.max(1) + extra;
    let ldb = m.max(1) + extra;
    // Full random A: the routine must ignore the other triangle (and the diagonal if unit).
    let (mut ar, mut ai) = cmat(r, lda * na.max(1), edge);
    let (mut br, mut bi) = cmat(r, ldb * n.max(1), edge);
    let a64 = to_c64(&ar, &ai);
    let mut want = to_c64(&br, &bi);
    let (m_, n_, lda_, ldb_) = (m as i32, n as i32, lda as i32, ldb as i32);
    let al = C64 { re: alpha.re, im: alpha.im };
    let (side, uplo, diag) =
        (if left { b'L' } else { b'R' }, if upper { b'U' } else { b'L' }, if nounit { b'N' } else { b'U' });
    // SAFETY: shapes per the ZTRMM contract.
    unsafe {
        ob::ztrmm(
            &side,
            &uplo,
            &tr_ch(trans),
            &diag,
            &m_,
            &n_,
            &al,
            a64.as_ptr(),
            &lda_,
            want.as_mut_ptr(),
            &ldb_,
            1,
            1,
            1,
            1,
        )
    };
    let a = ZV::new(V::new(&mut ar), V::new(&mut ai));
    let b = ZV::new(V::new(&mut br), V::new(&mut bi));
    // SAFETY: as above.
    unsafe { l3::ztrmm(left, upper, trans, nounit, m, n, alpha, a, lda, b, ldb) };
    assert_bits(
        &format!(
            "ztrmm side={} uplo={} trans={trans:?} diag={} m={m} n={n} alpha={alpha:?} extra={extra} edge={edge}",
            side as char, uplo as char, diag as char
        ),
        &flat2(&br, &bi),
        &flat(&want),
    );
}

fn trmm_flags() -> Vec<(bool, bool, Tr, bool)> {
    let mut v = Vec::new();
    for left in [true, false] {
        for upper in [true, false] {
            for trans in TRANS {
                for nounit in [false, true] {
                    v.push((left, upper, trans, nounit));
                }
            }
        }
    }
    v
}

#[test]
fn ztrmm_bitexact_small_grid() {
    if !ob::require("ztrmm_bitexact_small_grid") {
        return;
    }
    let mut r = Rng::new(301);
    let one = Z::new(1.0, 0.0);
    for (left, upper, trans, nounit) in trmm_flags() {
        for m in 0..14usize {
            for n in 0..14usize {
                ztrmm_case(&mut r, left, upper, trans, nounit, m, n, one, (m + n) % 2 * 2, (m * 3 + n) % 2 == 1);
            }
        }
    }
}

#[test]
fn ztrmm_bitexact_alphas() {
    if !ob::require("ztrmm_bitexact_alphas") {
        return;
    }
    let mut r = Rng::new(302);
    let alphas = [Z::new(-1.0, 0.0), Z::new(0.0, 0.0), Z::new(r.uniform(), r.uniform()), Z::new(1.0, 0.0)];
    for (left, upper, trans, nounit) in trmm_flags() {
        for alpha in alphas {
            for (m, n) in [(5usize, 7usize), (17, 9), (33, 40)] {
                ztrmm_case(&mut r, left, upper, trans, nounit, m, n, alpha, 1, true);
            }
        }
    }
}

#[test]
fn ztrmm_bitexact_block_grid() {
    if !ob::require("ztrmm_bitexact_block_grid") {
        return;
    }
    let mut r = Rng::new(303);
    let one = Z::new(1.0, 0.0);
    let mut count = 0;
    for (f, (left, upper, trans, nounit)) in trmm_flags().into_iter().enumerate() {
        for (i, &m) in GRID3.iter().enumerate() {
            let n = GRID3[(i * 7 + f * 5 + 3) % GRID3.len()];
            ztrmm_case(&mut r, left, upper, trans, nounit, m, n, one, i % 2, i % 3 == 0);
            count += 1;
        }
    }
    eprintln!("ztrmm block grid: {count} cases");
}

#[test]
fn ztrmm_bitexact_large() {
    if !ob::require("ztrmm_bitexact_large") {
        return;
    }
    let mut r = Rng::new(304);
    let one = Z::new(1.0, 0.0);
    // m or n cross Q=128, 2Q, P=256 and ~500 on the triangular side.
    let sizes = [
        (129usize, 31usize),
        (31, 129),
        (257, 40),
        (40, 257),
        (300, 3),
        (3, 300),
        (500, 20),
        (20, 500),
        (385, 65),
        (65, 385),
    ];
    for (f, (left, upper, trans, nounit)) in trmm_flags().into_iter().enumerate() {
        for (i, &(m, n)) in sizes.iter().enumerate() {
            ztrmm_case(&mut r, left, upper, trans, nounit, m, n, one, (f + i) % 2, (f + i) % 3 == 0);
        }
    }
}

/// `n > GEMM_R = 16112` crosses the `js` blocking of `trmm_L.c` (cheap: `A` is `m x m`).
#[test]
fn ztrmm_bitexact_gemm_r_blocking() {
    if !ob::require("ztrmm_bitexact_gemm_r_blocking") {
        return;
    }
    let mut r = Rng::new(305);
    let one = Z::new(1.0, 0.0);
    for (left, upper, trans, nounit) in trmm_flags().into_iter().filter(|f| f.0) {
        for m in [3usize, 130] {
            ztrmm_case(&mut r, left, upper, trans, nounit, m, l3::ZGEMM_R + 97, one, 1, m == 3);
        }
    }
}
