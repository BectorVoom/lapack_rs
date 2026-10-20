//! **Task A0** of `docs/cintx-lapack-blas-adoption-plan.md` §4.5: the
//! differential harness that decides the cintx adoption gate *before* any
//! transcription.
//!
//! No new numerics. It
//!
//! 1. builds the Wheeler input corpus by driving **libcint 6.1.3's own**
//!    `CINTrys_jacobi` / `CINTrys_laguerre` (and their long-double variants)
//!    and recording every tridiagonal they hand `_CINTdiagonalize`
//!    (`crates/cint_ref`), so the corpus is the vendor's, not a reconstruction;
//! 2. runs each through (a) Reference-LAPACK 3.12.1's real `dstemr_`
//!    (`JOBZ = 'V'`, `RANGE = 'A'`) and (b) libcint's vendored solver;
//! 3. reports the share of bit-identical elements and the max ULP per `n`,
//!    magnitude-worst first.
//!
//! **A0 reports; it does not gate.** The assertion at the end is only that the
//! harness ran over a non-empty corpus — the ULP table is the deliverable, and
//! `docs/a0-wheeler-differential.md` records the run.
//!
//! Run it with
//! `cargo test --release --test a0_wheeler_differential -- --nocapture`.

/// ULP distance between two finite doubles, saturating at `u64::MAX` for a
/// sign flip across zero or a non-finite value. The cintx ULP-table method.
fn ulps(a: f64, b: f64) -> u64 {
    if a.to_bits() == b.to_bits() {
        return 0;
    }
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return u64::MAX;
    }
    let key = |x: f64| -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    (key(a) - key(b)).unsigned_abs()
}

struct Row {
    n: usize,
    problems: usize,
    /// Eigenvalue comparisons.
    w_total: usize,
    w_same: usize,
    w_max_ulp: u64,
    w_worst_mag: f64,
    /// First-component-of-each-eigenvector comparisons (what Wheeler reads).
    v_total: usize,
    v_same: usize,
    v_max_ulp: u64,
}

#[test]
fn a0_reference_dstemr_vs_libcint_solver() {
    if !cint_ref::require("a0_wheeler_differential") {
        return;
    }
    if !lapack_ref::require("a0_wheeler_differential") {
        return;
    }

    // The four `cint_diagonalize` call sites plus `ext_eigen_transform_dev`
    // reach libcint's Wheeler path through these entry points. The `x` sweep
    // spans the small-x, mid and large-x regimes the Rys drivers see.
    let mut cases: Vec<(cint_ref::Entry, usize, f64, f64)> = Vec::new();
    let xs: Vec<f64> = {
        let mut v = vec![1e-8, 1e-4, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0, 80.0, 120.0];
        v.extend((1..=20).map(|k| k as f64 * 3.7));
        v
    };
    for &entry in
        &[cint_ref::Entry::Jacobi, cint_ref::Entry::Laguerre, cint_ref::Entry::LJacobi, cint_ref::Entry::LLaguerre]
    {
        for n in 1..=cint_ref::MXRYSROOTS {
            for &x in &xs {
                for &lower in &[0.0f64, 0.3, 0.8] {
                    cases.push((entry, n, x, lower));
                }
            }
        }
    }
    let corpus = cint_ref::wheeler_corpus(&cases, 200_000);
    assert!(!corpus.is_empty(), "A0: libcint's Wheeler path produced no tridiagonals");

    let mut rows: Vec<Row> = (0..=cint_ref::MXRYSROOTS)
        .map(|n| Row {
            n,
            problems: 0,
            w_total: 0,
            w_same: 0,
            w_max_ulp: 0,
            w_worst_mag: 0.0,
            v_total: 0,
            v_same: 0,
            v_max_ulp: 0,
        })
        .collect();

    // De-duplicate: the sweep revisits the same tridiagonal many times.
    let mut seen: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();
    let mut unique = 0usize;

    for p in &corpus {
        let n = p.n;
        if n == 0 || n > cint_ref::MXRYSROOTS {
            continue;
        }
        let key: Vec<u64> = std::iter::once(n as u64)
            .chain(p.diag[..n].iter().map(|v| v.to_bits()))
            .chain(p.off[..n].iter().map(|v| v.to_bits()))
            .collect();
        if !seen.insert(key) {
            continue;
        }
        unique += 1;

        // (b) libcint's vendored solver: the Wheeler convention.
        let (mut cd, mut ce) = (p.diag[..n].to_vec(), p.off[..n].to_vec());
        let mut ceig = vec![0.0f64; n];
        let mut cvec = vec![0.0f64; n * n];
        let cinfo = cint_ref::diagonalize(n, &mut cd, &mut ce, &mut ceig, &mut cvec);

        // (a) Reference-LAPACK `DSTEMR( 'V', 'A', .. )` on the same input.
        // DSTEMR wants E(1..N-1) with E(N) scratch, which is exactly the
        // `diag_off1` convention; Z is column-major with LDZ = n, so
        // `Z[j + i*n]` is the j-th component of the i-th eigenvector, matching
        // libcint's `vec[i*n + j]`.
        let (mut rd, mut re) = (p.diag[..n].to_vec(), p.off[..n].to_vec());
        let mut rw = vec![0.0f64; n];
        let mut rz = vec![0.0f64; n * n];
        let mut risuppz = vec![0i32; 2 * n];
        let (lw, liw) = (18 * n, 10 * n);
        let (mut rwork, mut riwork) = (vec![0.0f64; lw.max(1)], vec![0i32; liw.max(1)]);
        let (rm, _rac, rinfo) = lapack_ref::dstemr(
            'V',
            'A',
            n,
            &mut rd,
            &mut re,
            0.0,
            0.0,
            1,
            n,
            &mut rw,
            &mut rz,
            n.max(1),
            n as i32,
            &mut risuppz,
            true,
            &mut rwork,
            &mut riwork,
        );
        if cinfo != 0 || rinfo != 0 || rm as usize != n {
            // A problem one side refuses is not an ULP comparison; count it.
            continue;
        }

        let row = &mut rows[n];
        row.problems += 1;
        for i in 0..n {
            let u = ulps(ceig[i], rw[i]);
            row.w_total += 1;
            if u == 0 {
                row.w_same += 1;
            }
            if u > row.w_max_ulp {
                row.w_max_ulp = u;
                row.w_worst_mag = ceig[i].abs().max(rw[i].abs());
            }
            // Wheeler reads `vec[i*n]`, the first component of eigenvector i.
            // Sign is a free choice of the eigensolver, so compare magnitudes:
            // that is what `weights[i] = c0[i*n]^2 * mu0` consumes.
            let u = ulps(cvec[i * n].abs(), rz[i * n].abs());
            row.v_total += 1;
            if u == 0 {
                row.v_same += 1;
            }
            if u > row.v_max_ulp {
                row.v_max_ulp = u;
            }
        }
    }

    // Report, magnitude-worst first.
    let mut ranked: Vec<&Row> = rows.iter().filter(|r| r.problems > 0).collect();
    ranked.sort_by(|a, b| b.w_max_ulp.cmp(&a.w_max_ulp).then(b.w_worst_mag.total_cmp(&a.w_worst_mag)));

    println!("\nA0 — Reference-LAPACK 3.12.1 DSTEMR('V','A') vs libcint 6.1.3 _CINTdiagonalize");
    println!("corpus: {} recorded Wheeler problems, {unique} unique", corpus.len());
    println!(
        "\n {:>3} {:>6} {:>10} {:>12} {:>12} {:>10} {:>12}",
        "n", "probs", "W same", "W same %", "W max ULP", "|w| worst", "Z(1,i) maxULP"
    );
    for r in &ranked {
        let pct = 100.0 * r.w_same as f64 / r.w_total as f64;
        let mu = if r.w_max_ulp == u64::MAX { "inf/nan".to_string() } else { r.w_max_ulp.to_string() };
        let vu = if r.v_max_ulp == u64::MAX { "inf/nan".to_string() } else { r.v_max_ulp.to_string() };
        println!(
            " {:>3} {:>6} {:>10} {:>11.3}% {:>12} {:>10.3e} {:>12}",
            r.n, r.problems, r.w_same, pct, mu, r.w_worst_mag, vu
        );
    }
    let any_nonzero = ranked.iter().any(|r| r.w_max_ulp > 0);
    let total_w: usize = ranked.iter().map(|r| r.w_total).sum();
    let same_w: usize = ranked.iter().map(|r| r.w_same).sum();
    println!(
        "\noverall: {same_w}/{total_w} eigenvalues bit-identical ({:.3}%)",
        100.0 * same_w as f64 / total_w as f64
    );
    println!(
        "verdict: max_ulp = 0 between Reference-LAPACK DSTEMR and libcint's solver is {}.",
        if any_nonzero { "NOT reachable — Path C is the cintx deliverable" } else { "reachable on this corpus" }
    );

    // A0 reports; it does not gate. The only assertion is that it ran.
    assert!(!ranked.is_empty(), "A0: no comparable problem in the corpus");
}
