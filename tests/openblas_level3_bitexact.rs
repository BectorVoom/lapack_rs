//! Oracle tests for the OpenBLAS 0.3.30 SkylakeX level-3 track
//! (`lapack_rs::testing::openblas::level3`) against scipy's bundled OpenBLAS.
#![allow(clippy::too_many_arguments, clippy::manual_is_multiple_of)]
mod common;
use common::ob::*;
use lapack_rs::testing::V;
use lapack_rs::testing::openblas::level3::*;
use openblas_ref as ob;

// Kernel-level symbols of the pinned library (exported by the DYNAMIC_ARCH
// build), resolved at run time with `dlsym` so the test binary still links
// when the oracle is not (the tests then SKIP), used to bisect a mismatch to
// one micro-kernel.
unsafe extern "C" {
    fn dlsym(handle: *mut std::ffi::c_void, name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
}
type GemmK = unsafe extern "C" fn(isize, isize, isize, f64, *const f64, *const f64, *mut f64, isize) -> i32;
type TrmmK = unsafe extern "C" fn(isize, isize, isize, f64, *const f64, *const f64, *mut f64, isize, isize) -> i32;
fn sym(name: &std::ffi::CStr) -> *mut std::ffi::c_void {
    // SAFETY: dlsym with RTLD_DEFAULT and a NUL-terminated name.
    let p = unsafe { dlsym(std::ptr::null_mut(), name.as_ptr()) };
    assert!(!p.is_null(), "{name:?} not found in the linked OpenBLAS");
    p
}
fn gemm_kernel() -> GemmK {
    // SAFETY: the exported symbol has this C signature (kernel/x86_64).
    unsafe { std::mem::transmute::<*mut std::ffi::c_void, GemmK>(sym(c"dgemm_kernel_SKYLAKEX")) }
}
fn trmm_kernel(name: &std::ffi::CStr) -> TrmmK {
    // SAFETY: as above.
    unsafe { std::mem::transmute::<*mut std::ffi::c_void, TrmmK>(sym(name)) }
}

const DIMS: [usize; 23] = [0, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 257];

fn alphas(r: &mut Rng) -> [f64; 4] {
    [0.0, 1.0, -1.0, r.uniform() * 3.0]
}
fn betas(r: &mut Rng) -> [f64; 3] {
    [0.0, 1.0, r.uniform() * 2.0]
}

fn ch(b: bool, t: char, f: char) -> u8 {
    (if b { t } else { f }) as u8
}

/// The block sizes hard-coded in the transcription are the live ones.
#[test]
fn gotoblas_parameters_match_live_table() {
    if !ob::require("gotoblas_parameters_match_live_table") {
        return;
    }
    // SAFETY: `gotoblas` points at the dispatched `gotoblas_t` (int fields).
    let table = unsafe { *(sym(c"gotoblas") as *const *const i32) };
    let g = |off: usize| unsafe { *table.add(off / 4) } as usize;
    assert_eq!(g(0x2e0), GEMM_P, "dgemm_p");
    assert_eq!(g(0x2e4), GEMM_Q, "dgemm_q");
    assert_eq!(g(0x2e8), GEMM_R, "dgemm_r");
    assert_eq!(g(0x2ec), GEMM_UNROLL_M, "dgemm_unroll_m");
    assert_eq!(g(0x2f0), GEMM_UNROLL_N, "dgemm_unroll_n");
    assert_eq!(g(0x2f4), GEMM_UNROLL_MN, "dgemm_unroll_mn");
}

/// `dgemm_kernel_16x2_skylakex.c` (GEMM and the four TRMM builds) on random
/// packed buffers: every position random, so a wrong `l` window reads a value
/// the library does not.
#[test]
fn micro_kernel_16x2_bitexact() {
    if !ob::require("micro_kernel_16x2_bitexact") {
        return;
    }
    let mut r = Rng::new(11);
    let ms = [1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 45, 64];
    let ns = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 25, 36];
    let ks = [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 31, 32, 33, 40, 64, 100];
    let kerns = [Kern::Gemm, Kern::TrmmLN, Kern::TrmmLT, Kern::TrmmRN, Kern::TrmmRT];
    let mut cases = 0;
    for &m in &ms {
        for &n in &ns {
            for &k in &ks {
                for kern in kerns {
                    // Offsets the drivers use keep the window inside 0..k.
                    let dim = match kern {
                        Kern::Gemm => 0,
                        Kern::TrmmLN | Kern::TrmmLT => m,
                        _ => n + 1,
                    };
                    if kern != Kern::Gemm && k < dim {
                        continue;
                    }
                    let offs: Vec<isize> = if kern == Kern::Gemm {
                        vec![0]
                    } else {
                        vec![0, ((k - dim) / 2) as isize, (k - dim) as isize]
                    };
                    for off in offs {
                        let sa = r.vec_edge(m * k + 1);
                        let sb = r.vec_edge(n * k + 1);
                        let ldc = m + 3;
                        let c0 = r.vec(ldc * n);
                        let alpha = if kern == Kern::Gemm { r.uniform() } else { 1.0 };
                        let mut want = c0.clone();
                        let (mi, ni, ki, li) = (m as isize, n as isize, k as isize, ldc as isize);
                        // SAFETY: buffers sized for the packed layouts and ldc*n.
                        unsafe {
                            let (pa, pb, pc) = (sa.as_ptr(), sb.as_ptr(), want.as_mut_ptr());
                            match kern {
                                Kern::Gemm => gemm_kernel()(mi, ni, ki, alpha, pa, pb, pc, li),
                                Kern::TrmmLN => {
                                    trmm_kernel(c"dtrmm_kernel_LN_SKYLAKEX")(mi, ni, ki, alpha, pa, pb, pc, li, off)
                                }
                                Kern::TrmmLT => {
                                    trmm_kernel(c"dtrmm_kernel_LT_SKYLAKEX")(mi, ni, ki, alpha, pa, pb, pc, li, off)
                                }
                                // Right-side drivers pass `offset = -jjs`.
                                Kern::TrmmRN => {
                                    trmm_kernel(c"dtrmm_kernel_RN_SKYLAKEX")(mi, ni, ki, alpha, pa, pb, pc, li, -off)
                                }
                                Kern::TrmmRT => {
                                    trmm_kernel(c"dtrmm_kernel_RT_SKYLAKEX")(mi, ni, ki, alpha, pa, pb, pc, li, -off)
                                }
                            };
                        }
                        let mut got = c0.clone();
                        let o = match kern {
                            Kern::TrmmRN | Kern::TrmmRT => -off,
                            _ => off,
                        };
                        // SAFETY: as above.
                        unsafe { dgemm_kernel(kern, m, n, k, alpha, &sa, &sb, V::new(&mut got), ldc, o) };
                        assert_bits(&format!("{kern:?} m={m} n={n} k={k} off={off}"), &got, &want);
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("micro_kernel_16x2_bitexact: {cases} cases");
}

fn gemm_case(
    r: &mut Rng,
    ta: bool,
    tb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
    pad: usize,
    edge: bool,
) {
    let (ra, ca) = if ta { (k, m) } else { (m, k) };
    let (rb, cb) = if tb { (n, k) } else { (k, n) };
    let lda = ra.max(1) + pad;
    let ldb = rb.max(1) + pad;
    let ldc = m.max(1) + pad;
    let fill = |r: &mut Rng, len: usize| if edge { r.vec_edge(len) } else { r.vec(len) };
    let mut a = fill(r, lda * ca.max(1));
    let mut b = fill(r, ldb * cb.max(1));
    let c0 = fill(r, ldc * n.max(1));
    let mut want = c0.clone();
    let (mi, ni, ki, la, lb, lc) = (m as i32, n as i32, k as i32, lda as i32, ldb as i32, ldc as i32);
    // SAFETY: arrays sized lda*cols etc.
    unsafe {
        ob::dgemm(
            &ch(ta, 'T', 'N'),
            &ch(tb, 'T', 'N'),
            &mi,
            &ni,
            &ki,
            &alpha,
            a.as_ptr(),
            &la,
            b.as_ptr(),
            &lb,
            &beta,
            want.as_mut_ptr(),
            &lc,
            1,
            1,
        )
    };
    let mut got = c0.clone();
    // SAFETY: as above.
    unsafe { dgemm(ta, tb, m, n, k, alpha, V::new(&mut a), lda, V::new(&mut b), ldb, beta, V::new(&mut got), ldc) };
    assert_bits(
        &format!(
            "dgemm {}{} m={m} n={n} k={k} alpha={alpha} beta={beta} pad={pad} edge={edge}",
            ch(ta, 'T', 'N') as char,
            ch(tb, 'T', 'N') as char
        ),
        &got,
        &want,
    );
}

/// Small-matrix kernels: every trans combination around the permit
/// thresholds (MNK <= 1e6; TN also needs m*n <= 1200 and k >= 32) and the
/// 32/16/8/4/2 row and 8/6/4/2 column tails.
#[test]
fn dgemm_small_bitexact() {
    if !ob::require("dgemm_small_bitexact") {
        return;
    }
    let mut r = Rng::new(21);
    let small = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 31, 32, 33, 36, 40];
    let mut cases = 0;
    for (ta, tb) in [(false, false), (false, true), (true, false), (true, true)] {
        for &m in &small {
            for &n in &small {
                for &k in &[0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 63, 64, 65] {
                    let al = alphas(&mut r);
                    let be = betas(&mut r);
                    let t = r.next_u64() as usize;
                    gemm_case(
                        &mut r,
                        ta,
                        tb,
                        m,
                        n,
                        k,
                        al[t % 4],
                        be[(t >> 2) % 3],
                        (t >> 4) % 2 * 3,
                        (t >> 5) % 2 == 1,
                    );
                    cases += 1;
                }
            }
        }
    }
    eprintln!("dgemm_small_bitexact: {cases} cases");
}

/// Every alpha/beta combination on a sparse grid of the brief's sizes (small
/// and blocked paths).
#[test]
fn dgemm_grid_bitexact() {
    if !ob::require("dgemm_grid_bitexact") {
        return;
    }
    let mut r = Rng::new(22);
    let mut cases = 0;
    for (ta, tb) in [(false, false), (false, true), (true, false), (true, true)] {
        for (x, &m) in DIMS.iter().enumerate() {
            for (y, &n) in DIMS.iter().enumerate() {
                // Sparse cross: about a third of the (m, n) pairs.
                if (x * 7 + y * 3 + ta as usize + 2 * tb as usize) % 3 != 0 {
                    continue;
                }
                let k = DIMS[(x * 5 + y * 11 + 3) % DIMS.len()];
                let al = alphas(&mut r);
                let be = betas(&mut r);
                for (ia, &alpha) in al.iter().enumerate() {
                    let beta = be[(ia + x + y) % 3];
                    let pad = (ia + x) % 2 * 2;
                    gemm_case(&mut r, ta, tb, m, n, k, alpha, beta, pad, ia % 2 == 1);
                    cases += 1;
                }
            }
        }
    }
    eprintln!("dgemm_grid_bitexact: {cases} cases");
}

/// Blocked driver: GEMM_P (192/384) splits along m, GEMM_Q (384/768) along k,
/// 12-column tiles along n, both trans flags, betas 0/1/random.
#[test]
fn dgemm_blocked_bitexact() {
    if !ob::require("dgemm_blocked_bitexact") {
        return;
    }
    let mut r = Rng::new(23);
    let shapes = [
        (101, 100, 100),
        (193, 37, 61),
        (200, 129, 130),
        (385, 13, 257),
        (300, 301, 302),
        (257, 385, 200),
        (129, 60, 385),
        (64, 70, 500),
        (520, 30, 90),
        (97, 11, 770),
        (600, 600, 16),
        (33, 600, 600),
        (383, 3, 1000),
        (16, 1000, 70),
        (400, 25, 400),
    ];
    let mut cases = 0;
    for (ta, tb) in [(false, false), (false, true), (true, false), (true, true)] {
        for (s, &(m, n, k)) in shapes.iter().enumerate() {
            let al = alphas(&mut r);
            let be = betas(&mut r);
            let alpha = al[(s + ta as usize) % 4];
            let beta = be[(s + tb as usize) % 3];
            gemm_case(&mut r, ta, tb, m, n, k, alpha, beta, s % 2, s % 3 == 0);
            cases += 1;
        }
    }
    eprintln!("dgemm_blocked_bitexact: {cases} cases");
}

fn syr2k_case(r: &mut Rng, upper: bool, n: usize, k: usize, alpha: f64, beta: f64, pad: usize, edge: bool) {
    let lda = n.max(1) + pad;
    let ldc = n.max(1) + 2 * pad;
    let fill = |r: &mut Rng, len: usize| if edge { r.vec_edge(len) } else { r.vec(len) };
    let mut a = fill(r, lda * k.max(1));
    let mut b = fill(r, lda * k.max(1));
    let c0 = fill(r, ldc * n.max(1));
    let mut want = c0.clone();
    let (ni, ki, la, lc) = (n as i32, k as i32, lda as i32, ldc as i32);
    // SAFETY: arrays sized lda*k and ldc*n.
    unsafe {
        ob::dsyr2k(
            &ch(upper, 'U', 'L'),
            &b'N',
            &ni,
            &ki,
            &alpha,
            a.as_ptr(),
            &la,
            b.as_ptr(),
            &la,
            &beta,
            want.as_mut_ptr(),
            &lc,
            1,
            1,
        )
    };
    let mut got = c0.clone();
    // SAFETY: as above.
    unsafe { dsyr2k_n(upper, n, k, alpha, V::new(&mut a), lda, V::new(&mut b), lda, beta, V::new(&mut got), ldc) };
    assert_bits(
        &format!("dsyr2k {}N n={n} k={k} alpha={alpha} beta={beta} pad={pad} edge={edge}", ch(upper, 'U', 'L') as char),
        &got,
        &want,
    );
}

/// DSYTRD's call (`alpha = -1`, `beta = 1`, `k = NB = 32`) for n across the
/// UNROLL_MN (32), GEMM_P (192/384) boundaries, plus a sparse (n, k) grid
/// with every alpha and beta, and GEMM_Q splits of k.
#[test]
fn dsyr2k_bitexact() {
    if !ob::require("dsyr2k_bitexact") {
        return;
    }
    let mut r = Rng::new(31);
    let mut cases = 0;
    for upper in [true, false] {
        for n in sizes(700).into_iter().chain([193, 250, 320, 383, 384, 385, 400, 450, 520]) {
            syr2k_case(&mut r, upper, n, 32, -1.0, 1.0, n % 3, n % 5 == 0);
            cases += 1;
        }
        for (x, &n) in DIMS.iter().enumerate() {
            for (y, &k) in DIMS.iter().enumerate() {
                if (x * 5 + y * 3 + upper as usize) % 3 != 0 {
                    continue;
                }
                let al = alphas(&mut r);
                let be = betas(&mut r);
                for (ia, &alpha) in al.iter().enumerate() {
                    syr2k_case(&mut r, upper, n, k, alpha, be[(ia + x + y) % 3], (ia + y) % 2 * 2, ia % 2 == 0);
                    cases += 1;
                }
            }
        }
        for &(n, k) in &[(300usize, 385usize), (129, 800), (400, 400), (520, 64), (33, 1000), (385, 7)] {
            let alpha = r.uniform();
            syr2k_case(&mut r, upper, n, k, alpha, 0.0, 1, false);
            syr2k_case(&mut r, upper, n, k, -1.0, 1.0, 0, true);
            cases += 2;
        }
    }
    eprintln!("dsyr2k_bitexact: {cases} cases");
}

fn trmm_case(
    r: &mut Rng,
    left: bool,
    upper: bool,
    trans: bool,
    nounit: bool,
    m: usize,
    n: usize,
    alpha: f64,
    pad: usize,
    edge: bool,
) {
    let na = if left { m } else { n };
    let lda = na.max(1) + pad;
    let ldb = m.max(1) + 2 * pad;
    let fill = |r: &mut Rng, len: usize| if edge { r.vec_edge(len) } else { r.vec(len) };
    let mut a = fill(r, lda * na.max(1));
    let b0 = fill(r, ldb * n.max(1));
    let mut want = b0.clone();
    let (mi, ni, la, lb) = (m as i32, n as i32, lda as i32, ldb as i32);
    let flags = [ch(left, 'L', 'R'), ch(upper, 'U', 'L'), ch(trans, 'T', 'N'), ch(nounit, 'N', 'U')];
    // SAFETY: arrays sized lda*na and ldb*n.
    unsafe {
        ob::dtrmm(
            &flags[0],
            &flags[1],
            &flags[2],
            &flags[3],
            &mi,
            &ni,
            &alpha,
            a.as_ptr(),
            &la,
            want.as_mut_ptr(),
            &lb,
            1,
            1,
            1,
            1,
        )
    };
    let mut got = b0.clone();
    // SAFETY: as above.
    unsafe { dtrmm(left, upper, trans, nounit, m, n, alpha, V::new(&mut a), lda, V::new(&mut got), ldb) };
    let f: String = flags.iter().map(|&c| c as char).collect();
    assert_bits(&format!("dtrmm {f} m={m} n={n} alpha={alpha} pad={pad} edge={edge}"), &got, &want);
}

/// All 16 flag combinations; sizes crossing the 16-row tiles, the 6/2
/// column blocks, GEMM_P (192) and GEMM_Q (384) on both m and n.
#[test]
fn dtrmm_bitexact() {
    if !ob::require("dtrmm_bitexact") {
        return;
    }
    let mut r = Rng::new(41);
    let mut cases = 0;
    for code in 0..16 {
        let (left, upper, trans, nounit) = (code & 8 != 0, code & 4 != 0, code & 2 != 0, code & 1 != 0);
        for (x, &m) in DIMS.iter().enumerate() {
            for (y, &n) in DIMS.iter().enumerate() {
                if (x * 3 + y * 7 + code) % 4 != 0 {
                    continue;
                }
                let al = alphas(&mut r);
                let alpha = if (x + y) % 3 == 0 { al[(x * 2 + y) % 4] } else { 1.0 };
                trmm_case(&mut r, left, upper, trans, nounit, m, n, alpha, (x + y) % 2, (x + code) % 3 == 0);
                cases += 1;
            }
        }
        for &(m, n) in &[
            (300usize, 70usize),
            (70, 300),
            (385, 400),
            (400, 385),
            (520, 193),
            (193, 520),
            (600, 600),
            (777, 20),
            (20, 777),
        ] {
            trmm_case(&mut r, left, upper, trans, nounit, m, n, 1.0, 1, code % 2 == 0);
            cases += 1;
        }
    }
    eprintln!("dtrmm_bitexact: {cases} cases");
}
