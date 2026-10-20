//! **Path C gate.** `lapack_rs::cint_stemr` against **libcint 6.1.3**'s
//! vendored `_CINTdiagonalize` (`crates/cint_ref`), bit for bit —
//! `max_ulp = 0`, which is task **C3** of
//! `docs/cintx-lapack-blas-adoption-plan.md`.
//!
//! The oracle here is **not** Reference-LAPACK: `docs/a0-wheeler-differential.md`
//! measures 4.25e9 ULP between the two solvers, so `ref_*_bitexact` does not
//! and must not cover this module.
//!
//! The corpus is the vendor's own: libcint's `CINTrys_jacobi` /
//! `CINTrys_laguerre` are driven over the Rys regimes and every tridiagonal
//! they hand `_CINTdiagonalize` is recorded, then replayed through both sides.
//! A synthetic sweep (random, clustered, graded, split and degenerate
//! tridiagonals at every `n <= MXRYSROOTS`) covers the branches the Wheeler
//! corpus happens not to reach.

mod common;
use common::*;
use lapack_rs::{MXRYSROOTS, cint_stemr, cint_stemr_work_len};

/// Bit-for-bit, except that **two `NaN`s compare equal whatever their sign
/// bit**. IEEE-754 §6.2 leaves the sign of a `NaN` produced by an operation
/// unspecified, and neither gcc nor rustc promises one: on the degenerate
/// synthetic fixtures (an exactly zero off-diagonal, which drives `0/0`
/// through `_dlar1v`'s chain) the two sides agree on every finite value and
/// differ only in that bit. Returns how many such differences were tolerated,
/// so a real regression cannot hide behind the exemption.
fn assert_bits_nan_sign(what: &str, got: &[f64], want: &[f64]) -> usize {
    assert_eq!(got.len(), want.len(), "{what}: length {} vs {}", got.len(), want.len());
    let mut nan_sign = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() == w.to_bits() {
            continue;
        }
        if g.is_nan() && w.is_nan() {
            nan_sign += 1;
            continue;
        }
        panic!("{what}: entry {i} differs: got {g:e} (0x{:016x}) want {w:e} (0x{:016x})", g.to_bits(), w.to_bits());
    }
    nan_sign
}

/// Both sides on the same input; every output compared bit for bit.
/// Returns the number of `NaN`-sign-only differences (0 on any real corpus).
fn check(tag: &str, n: usize, diag: &[f64], off: &[f64]) -> usize {
    let (lw, liw) = cint_stemr_work_len(n);

    let (mut d, mut e) = (diag[..n].to_vec(), off[..n].to_vec());
    let mut eig = vec![0.0f64; n];
    let mut vec_ = vec![0.0f64; n * n];
    let (mut work, mut iwork) = (vec![0.0f64; lw], vec![0i32; liw]);
    let info = cint_stemr(n, &mut d, &mut e, &mut eig, &mut vec_, &mut work, &mut iwork);

    let (mut dr, mut er) = (diag[..n].to_vec(), off[..n].to_vec());
    let mut eigr = vec![0.0f64; n];
    let mut vecr = vec![0.0f64; n * n];
    let infor = cint_ref::diagonalize(n, &mut dr, &mut er, &mut eigr, &mut vecr);

    assert_eq!(info, infor, "cint_stemr INFO {tag}");
    if info != 0 {
        return 0;
    }
    let mut ns = assert_bits_nan_sign(&format!("cint_stemr EIG {tag}"), &eig, &eigr);
    ns += assert_bits_nan_sign(&format!("cint_stemr VEC {tag}"), &vec_, &vecr);
    // `diag` / `diag_off1` are documented as destroyed; the destruction is
    // part of the contract because `_compute_eigenvectors` reads the RRR back
    // out of them, so compare those too.
    ns += assert_bits_nan_sign(&format!("cint_stemr DIAG {tag}"), &d, &dr);
    ns += assert_bits_nan_sign(&format!("cint_stemr OFF {tag}"), &e, &er);
    ns
}

#[test]
fn cint_stemr_matches_libcint_on_the_wheeler_corpus() {
    if !cint_ref::require("cint_stemr_wheeler") {
        return;
    }
    let mut cases: Vec<(cint_ref::Entry, usize, f64, f64)> = Vec::new();
    let xs: Vec<f64> = {
        let mut v = vec![1e-8, 1e-4, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0, 80.0, 120.0];
        v.extend((1..=20).map(|k| k as f64 * 3.7));
        v
    };
    for &entry in
        &[cint_ref::Entry::Jacobi, cint_ref::Entry::Laguerre, cint_ref::Entry::LJacobi, cint_ref::Entry::LLaguerre]
    {
        for n in 1..=MXRYSROOTS {
            for &x in &xs {
                for &lower in &[0.0f64, 0.3, 0.8] {
                    cases.push((entry, n, x, lower));
                }
            }
        }
    }
    let corpus = cint_ref::wheeler_corpus(&cases, 200_000);
    assert!(!corpus.is_empty(), "libcint's Wheeler path produced no tridiagonals");

    let mut seen: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();
    let mut checked = 0usize;
    for p in &corpus {
        let n = p.n;
        if n == 0 || n > MXRYSROOTS {
            continue;
        }
        let key: Vec<u64> = std::iter::once(n as u64)
            .chain(p.diag[..n].iter().map(|v| v.to_bits()))
            .chain(p.off[..n].iter().map(|v| v.to_bits()))
            .collect();
        if !seen.insert(key) {
            continue;
        }
        let ns = check(&format!("wheeler #{checked} n={n}"), n, &p.diag, &p.off);
        // The real domain produces no NaN at all, so the exemption must never
        // fire here: `max_ulp = 0` on the Wheeler corpus is literal.
        assert_eq!(ns, 0, "wheeler #{checked} n={n}: {ns} NaN-sign differences");
        checked += 1;
    }
    assert!(checked > 1000, "only {checked} unique Wheeler problems checked");
    eprintln!("cint_stemr: {checked} unique Wheeler tridiagonals, max_ulp = 0, no NaN");
}

#[test]
fn cint_stemr_matches_libcint_on_synthetic_fixtures() {
    if !cint_ref::require("cint_stemr_synthetic") {
        return;
    }
    let mut checked = 0usize;
    let mut nan_sign = 0usize;

    // Named pathological tridiagonals, on top of the ten random families:
    // Wilkinson W+ (pathologically close pairs), one long cluster at the
    // bisection floor, element growth, and a spectrum straddling the double
    // range. These are what drive `_compute_eigenvectors`' representation
    // tree deepest and `_dlarrf`'s shift retries.
    for n in 3..=MXRYSROOTS {
        let half = (n / 2) as i64;
        let named: [(&str, Vec<f64>, Vec<f64>); 4] = [
            (
                "wilkinson",
                (0..n).map(|i| (i as i64 - half).abs() as f64).collect(),
                (0..n).map(|i| if i + 1 < n { 1.0 } else { 0.0 }).collect(),
            ),
            (
                "one_cluster",
                (0..n).map(|_| 1.0).collect(),
                (0..n).map(|i| if i + 1 < n { 1e-15 } else { 0.0 }).collect(),
            ),
            (
                "growth",
                (0..n).map(|i| if i % 2 == 0 { 1e-14 } else { 1e14 }).collect(),
                (0..n).map(|i| if i + 1 < n { 1.0 } else { 0.0 }).collect(),
            ),
            (
                "straddle",
                (0..n).map(|i| if i % 2 == 0 { 1e-290 } else { -1e290 }).collect(),
                (0..n).map(|i| if i + 1 < n { 1e150 } else { 0.0 }).collect(),
            ),
        ];
        for (name, d, e) in named {
            nan_sign += check(&format!("named n={n} {name}"), n, &d, &e);
            checked += 1;
        }
    }

    for n in 1..=MXRYSROOTS {
        for kind in 0..10usize {
            for seed in 0..12u64 {
                let mut r = Rng::new(40_000 + seed * 7919 + n as u64 * 31 + kind as u64);
                let mut d = vec![0.0f64; n];
                let mut e = vec![0.0f64; n];
                for i in 0..n {
                    d[i] = match kind {
                        0 => r.uniform(),
                        1 => i as f64 + 1.0,
                        2 => 1.0,
                        3 => 1.0 + 1e-13 * (i / 2) as f64,
                        4 => 10f64.powi(-(i as i32)),
                        5 => r.uniform() * 1e6,
                        6 => r.uniform() * 1e-6,
                        7 => (i as f64 - n as f64 / 2.0) * 1e-8,
                        8 => r.uniform().abs(),
                        _ => -r.uniform().abs(),
                    };
                    if i + 1 < n {
                        e[i] = match kind {
                            0 => r.uniform(),
                            1 => 1.0,
                            2 => 1e-16,
                            3 => 1e-14,
                            4 => 10f64.powi(-(i as i32) - 1),
                            5 => r.uniform() * 1e-3,
                            6 => r.uniform() * 1e-9,
                            7 => 1e-9,
                            // Split blocks: an exact zero off-diagonal.
                            8 => {
                                if i % 3 == 2 {
                                    0.0
                                } else {
                                    r.uniform()
                                }
                            }
                            _ => r.uniform(),
                        };
                    }
                }
                // `diag_off1[n-1]` is scratch on entry; libcint overwrites it,
                // so it must start identical on both sides — it does, both are
                // the same `e` vector.
                nan_sign += check(&format!("synthetic n={n} kind={kind} seed={seed}"), n, &d, &e);
                checked += 1;
            }
        }
    }
    eprintln!("cint_stemr: {checked} synthetic fixtures, max_ulp = 0, {nan_sign} NaN-sign-only differences");
}

#[test]
fn cint_stemr_work_len_and_edges() {
    // `eigh.c:1462`: double work[MXRYSROOTS*9 + 1], int iwork[MXRYSROOTS*5].
    for n in 0..=MXRYSROOTS {
        assert_eq!(cint_stemr_work_len(n), (9 * n + 1, 5 * n));
    }
    // n = 0 returns 0 and touches nothing.
    let mut empty: [f64; 0] = [];
    let mut emptyi: [i32; 0] = [];
    assert_eq!(cint_stemr(0, &mut [], &mut [], &mut [], &mut empty, &mut [], &mut emptyi), 0);
    // n = 1 is the `eig[0] = diag[0]`, `vec[0] = 1` short-circuit.
    let (lw, liw) = cint_stemr_work_len(1);
    let (mut d, mut e) = (vec![7.5], vec![0.0]);
    let (mut eig, mut v) = (vec![0.0], vec![0.0]);
    let (mut w, mut iw) = (vec![0.0; lw], vec![0i32; liw]);
    assert_eq!(cint_stemr(1, &mut d, &mut e, &mut eig, &mut v, &mut w, &mut iw), 0);
    assert_eq!((eig[0].to_bits(), v[0].to_bits()), (7.5f64.to_bits(), 1.0f64.to_bits()));
}
