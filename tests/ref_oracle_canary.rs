//! Oracle canary: the linked oracle is the pinned Reference-LAPACK build, and
//! a comparison against it is sensitive to term order (so an accidentally
//! linked optimised BLAS fails loudly instead of passing loosely).

#[test]
fn manifest_matches_pinned_build() {
    if !lapack_ref::require("manifest_matches_pinned_build") {
        return;
    }
    let path = lapack_ref::manifest_path().expect("manifest path recorded at build time");
    let text = std::fs::read_to_string(path).expect("read ref-lapack.toml");
    for needle in [
        "version = \"3.12.1\"",
        "sha256 = \"2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422\"",
        "fortran_flags = \"-O2 -march=x86-64 -mtune=generic -ffp-contract=off -fno-fast-math -frecursive\"",
        "cmake_build_type = \"None\"",
        "nm_openblas_symbols = 0",
        "nm_has_dsyevd = true",
    ] {
        assert!(text.contains(needle), "ref-lapack.toml lacks `{needle}`:\n{text}");
    }
}

#[test]
fn dgemm_reference_term_order() {
    if !lapack_ref::require("dgemm_reference_term_order") {
        return;
    }
    // C(1,1) = ((0 + t1*A(1,1)) + t2*A(1,2)) + t3*A(1,3) with t = B(:,1) = 1:
    // 1 + 2^-53 rounds back to 1 twice in reference order, but a reordered sum
    // (2^-53 + 2^-53) + 1 is 1 + 2^-52. An FMA-contracted chain differs too.
    let tiny = 2f64.powi(-53);
    let a = [1.0, tiny, tiny];
    let b = [1.0, 1.0, 1.0];
    let mut c = [0.0];
    lapack_ref::dgemm('N', 'N', 1, 1, 3, 1.0, &a, 1, &b, 3, 0.0, &mut c, 1);
    let reference_order = ((0.0 + a[0]) + a[1]) + a[2];
    let reordered = (a[2] + a[1]) + a[0];
    assert_ne!(reference_order.to_bits(), reordered.to_bits(), "canary input is not order-sensitive");
    assert_eq!(
        c[0].to_bits(),
        reference_order.to_bits(),
        "oracle dgemm is not reference-ordered (optimised BLAS linked?)"
    );

    // FMA sensitivity: 1 + (1+eps)(1-eps) on one term.
    let e = f64::EPSILON;
    let a = [1.0 + e];
    let b = [1.0 - e];
    let mut c = [1.0];
    lapack_ref::dgemm('N', 'N', 1, 1, 1, 1.0, &a, 1, &b, 1, 1.0, &mut c, 1);
    let unfused = 1.0 + (1.0 - e) * (1.0 + e);
    assert_eq!(c[0].to_bits(), unfused.to_bits(), "oracle dgemm contracted a*b + c");

    // And ours agrees on both.
    let mut a = [1.0, tiny, tiny];
    let mut b = [1.0, 1.0, 1.0];
    let mut c = [0.0];
    // SAFETY: shapes as documented for DGEMM.
    unsafe { lapack_rs::testing::dgemm(false, false, 1, 1, 3, 1.0, &mut a, 1, &mut b, 3, 0.0, &mut c, 1) };
    assert_eq!(c[0].to_bits(), reference_order.to_bits());
}

#[test]
fn dlamch_values() {
    if !lapack_ref::require("dlamch_values") {
        return;
    }
    for &(c, v) in lapack_rs::testing::DLAMCH {
        assert_eq!(lapack_ref::dlamch(c).to_bits(), v.to_bits(), "DLAMCH('{c}')");
    }
}
