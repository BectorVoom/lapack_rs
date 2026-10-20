//! Component gates: each transcribed kernel / copy routine against the SAME internal routine of
//! the pinned `.so`, called at its symbol address (see `support`). Used to bisect; the
//! end-to-end gate is `tests/zlevel3.rs`.
mod common;
#[path = "common/ob_support.rs"]
mod support;
use common::ob::*;
use lapack_rs::testing::V;
use lapack_rs::testing::openblas::zlevel3 as l3;
use lapack_rs::testing::{Z, ZV};
use openblas_ref as ob;
use support::*;

pub fn join(re: &[f64], im: &[f64]) -> Vec<f64> {
    re.iter().zip(im).flat_map(|(a, b)| [*a, *b]).collect()
}

const SENT: f64 = f64::from_bits(0x7ff8_dead_beef_0001);

#[test]
fn gemm_copies_match_binary() {
    if !ob::require("gemm_copies_match_binary") {
        return;
    }
    let mut r = Rng::new(11);
    // SAFETY: signatures from kernel/generic/zgemm_{n,t}copy_{2,4}.c.
    let (itc, inc, onc, otc): (GemmCopy, GemmCopy, GemmCopy, GemmCopy) =
        unsafe { (func(SYM_ZGEMM_ITCOPY), func(SYM_ZGEMM_INCOPY), func(SYM_ZGEMM_ONCOPY), func(SYM_ZGEMM_OTCOPY)) };
    for m in 0..13usize {
        for n in 0..13usize {
            let lda = m.max(n).max(1) + 2;
            let mut ar = r.vec(lda * lda.max(n).max(m));
            let mut ai = r.vec(ar.len());
            let ail = join(&ar, &ai);
            let len = 2 * m * n;
            for (name, bin, mine) in [
                ("itcopy", itc, l3::pack::itcopy as unsafe fn(usize, usize, ZV, usize, &mut [f64])),
                ("incopy", inc, l3::pack::incopy),
                ("oncopy", onc, l3::pack::oncopy),
                ("otcopy", otc, l3::pack::otcopy),
            ] {
                let mut want = vec![SENT; len + 8];
                // SAFETY: a has lda*max(m,n) columns, buffer m*n complex.
                unsafe { bin(m as i64, n as i64, ail.as_ptr(), lda as i64, want.as_mut_ptr()) };
                let mut got = vec![SENT; len + 8];
                let zv = ZV::new(V::new(&mut ar), V::new(&mut ai));
                // SAFETY: as above.
                unsafe { mine(m, n, zv, lda, &mut got) };
                assert_bits(&format!("{name} m={m} n={n}"), &got, &want);
            }
        }
    }
}

#[test]
fn gemm_beta_matches_binary() {
    if !ob::require("gemm_beta_matches_binary") {
        return;
    }
    let mut r = Rng::new(12);
    // SAFETY: zgemm_beta.S signature.
    let bin: Beta = unsafe { func(SYM_ZGEMM_BETA) };
    let betas = [
        Z::new(0.0, 0.0),
        Z::new(-0.0, 0.0),
        Z::new(0.0, -0.0),
        Z::new(1.0, 0.0),
        Z::new(-1.0, 0.0),
        Z::new(0.0, 1.0),
        Z::new(r.uniform(), r.uniform()),
        Z::new(r.uniform(), 0.0),
    ];
    for m in [0usize, 1, 2, 3, 5, 8] {
        for n in [0usize, 1, 2, 7] {
            for beta in betas {
                let ldc = m + 1;
                let mut re = r.vec_edge(ldc * n.max(1));
                let mut im = r.vec_edge(ldc * n.max(1));
                let mut want = join(&re, &im);
                // SAFETY: C has ldc*n complex.
                unsafe {
                    bin(
                        m as i64,
                        n as i64,
                        0,
                        beta.re,
                        beta.im,
                        std::ptr::null(),
                        0,
                        std::ptr::null(),
                        0,
                        want.as_mut_ptr(),
                        ldc as i64,
                    )
                };
                let c = ZV::new(V::new(&mut re), V::new(&mut im));
                // SAFETY: as above.
                unsafe { l3::zgemm_beta(m, n, beta, c, ldc) };
                assert_bits(&format!("beta m={m} n={n} {beta:?}"), &join(&re, &im), &want);
            }
        }
    }
}

#[test]
fn gemm_kernels_match_binary() {
    if !ob::require("gemm_kernels_match_binary") {
        return;
    }
    let mut r = Rng::new(13);
    for (sym, conj) in [
        (SYM_ZGEMM_KERNEL_N, l3::Conj { a: false, b: false }),
        (SYM_ZGEMM_KERNEL_L, l3::Conj { a: true, b: false }),
        (SYM_ZGEMM_KERNEL_R, l3::Conj { a: false, b: true }),
        (SYM_ZGEMM_KERNEL_B, l3::Conj { a: true, b: true }),
    ] {
        // SAFETY: zgemm_kernel_4x2_skylakex.c signature.
        let bin: GemmKernel = unsafe { func(sym) };
        for m in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 13] {
            for n in [1usize, 2, 3, 5, 6, 7, 11, 12, 13, 23, 25] {
                for k in [1usize, 2, 5, 19, 20, 21, 26, 40] {
                    for trial in 0..2 {
                        let alpha = match trial {
                            0 => Z::new(r.uniform(), r.uniform()),
                            _ => Z::new(1.0, 0.0),
                        };
                        let a = r.vec_edge(2 * m * k);
                        let b = r.vec_edge(2 * n * k);
                        let ldc = m + 1;
                        let mut re = r.vec_edge(ldc * n);
                        let mut im = r.vec_edge(ldc * n);
                        let mut want = join(&re, &im);
                        // SAFETY: packed blocks of m*k and k*n complex, C ldc*n.
                        unsafe {
                            bin(
                                m as i64,
                                n as i64,
                                k as i64,
                                alpha.re,
                                alpha.im,
                                a.as_ptr(),
                                b.as_ptr(),
                                want.as_mut_ptr(),
                                ldc as i64,
                            )
                        };
                        let c = ZV::new(V::new(&mut re), V::new(&mut im));
                        // SAFETY: as above.
                        unsafe { l3::zgemm_kernel(conj, m, n, k, alpha, &a, &b, c, ldc) };
                        assert_bits(&format!("kernel {conj:?} m={m} n={n} k={k}"), &join(&re, &im), &want);
                    }
                }
            }
        }
    }
}

#[test]
fn trmm_copies_match_binary() {
    if !ob::require("trmm_copies_match_binary") {
        return;
    }
    let mut r = Rng::new(21);
    let table = [
        (SYM_ZTRMM_IUNUCOPY, 4, true, false, true),
        (SYM_ZTRMM_IUNNCOPY, 4, true, false, false),
        (SYM_ZTRMM_ILNUCOPY, 4, false, false, true),
        (SYM_ZTRMM_ILNNCOPY, 4, false, false, false),
        (SYM_ZTRMM_IUTUCOPY, 4, true, true, true),
        (SYM_ZTRMM_IUTNCOPY, 4, true, true, false),
        (SYM_ZTRMM_ILTUCOPY, 4, false, true, true),
        (SYM_ZTRMM_ILTNCOPY, 4, false, true, false),
        (SYM_ZTRMM_OUNUCOPY, 2, true, false, true),
        (SYM_ZTRMM_OUNNCOPY, 2, true, false, false),
        (SYM_ZTRMM_OLNUCOPY, 2, false, false, true),
        (SYM_ZTRMM_OLNNCOPY, 2, false, false, false),
        (SYM_ZTRMM_OUTUCOPY, 2, true, true, true),
        (SYM_ZTRMM_OUTNCOPY, 2, true, true, false),
        (SYM_ZTRMM_OLTUCOPY, 2, false, true, true),
        (SYM_ZTRMM_OLTNCOPY, 2, false, true, false),
    ];
    let lda = 40usize;
    let mut ar = r.vec(lda * lda);
    let mut ai = r.vec(lda * lda);
    let ail = join(&ar, &ai);
    let mut count = 0;
    for (sym, width, upper, trans, unit) in table {
        // SAFETY: ztrmm_*copy signature.
        let bin: TrmmCopy = unsafe { func(sym) };
        let cp = l3::TriCopy { upper, trans, unit };
        // The drivers only produce n <= m with posY - posX a multiple of the copy width
        // (`is`/`jjs` advance in whole groups); misaligned calls walk the pointers differently
        // and are not transcribed.
        for m in 0..14usize {
            for n in 0..=m {
                for px in [0usize, 1, 4, 8] {
                    for dy in (0..=(m.saturating_sub(n))).step_by(width) {
                        let py = px + dy;
                        let len = 2 * m * n;
                        let mut want = vec![SENT; len + 8];
                        // SAFETY: indices stay below 2*14+8 < lda.
                        unsafe {
                            bin(m as i64, n as i64, ail.as_ptr(), lda as i64, px as i64, py as i64, want.as_mut_ptr())
                        };
                        let mut got = vec![SENT; len + 8];
                        let zv = ZV::new(V::new(&mut ar), V::new(&mut ai));
                        // SAFETY: as above.
                        unsafe { l3::trmm_copy_component(cp, width, m, n, zv, lda, px, py, &mut got) };
                        assert_bits(&format!("trmm copy {cp:?} w={width} m={m} n={n} px={px} py={py}"), &got, &want);
                        count += 1;
                    }
                }
            }
        }
    }
    eprintln!("trmm copies: {count} cases");
}

#[test]
fn trmm_kernels_match_binary() {
    if !ob::require("trmm_kernels_match_binary") {
        return;
    }
    let mut r = Rng::new(22);
    use l3::TrmmConj::*;
    let table = [
        (SYM_ZTRMM_KERNEL_LN, true, false, NN),
        (SYM_ZTRMM_KERNEL_LT, true, true, NN),
        (SYM_ZTRMM_KERNEL_LR, true, false, CN),
        (SYM_ZTRMM_KERNEL_LC, true, true, CN),
        (SYM_ZTRMM_KERNEL_RN, false, false, NN),
        (SYM_ZTRMM_KERNEL_RT, false, true, NN),
        (SYM_ZTRMM_KERNEL_RR, false, false, NC),
        (SYM_ZTRMM_KERNEL_RC, false, true, NC),
    ];
    let mut count = 0;
    for (sym, left, transa, conj) in table {
        // SAFETY: TRMM kernel signature (offset on the stack after ldc).
        let bin: TrmmKernel = unsafe { func(sym) };
        for m in 1..11usize {
            for n in 1..9usize {
                for k in 1..14usize {
                    // Offsets the drivers use: LEFT 0..=k-m, RIGHT -(0..=k-n).
                    let span = if left { k as isize - m as isize } else { k as isize - n as isize };
                    if span < 0 {
                        continue;
                    }
                    for o in 0..=span {
                        let offset = if left { o } else { -o };
                        for alpha in [Z::new(1.0, 0.0), Z::new(r.uniform(), r.uniform())] {
                            let a = r.vec_edge(2 * m * k);
                            let b = r.vec_edge(2 * n * k);
                            let ldc = m + 2;
                            let mut re = r.vec(ldc * n);
                            let mut im = r.vec(ldc * n);
                            let mut want = join(&re, &im);
                            // SAFETY: packed m*k and k*n complex, C ldc*n.
                            unsafe {
                                bin(
                                    m as i64,
                                    n as i64,
                                    k as i64,
                                    alpha.re,
                                    alpha.im,
                                    a.as_ptr(),
                                    b.as_ptr(),
                                    want.as_mut_ptr(),
                                    ldc as i64,
                                    offset as i64,
                                )
                            };
                            let c = ZV::new(V::new(&mut re), V::new(&mut im));
                            // SAFETY: as above.
                            unsafe { l3::trmm_kernel(left, transa, conj, m, n, k, alpha, &a, &b, c, ldc, offset) };
                            assert_bits(
                                &format!(
                                    "trmm kernel left={left} transa={transa} {conj:?} m={m} n={n} k={k} off={offset}"
                                ),
                                &join(&re, &im),
                                &want,
                            );
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("trmm kernels: {count} cases");
}
