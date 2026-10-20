//! Shared fixtures for the oracle tests: seeded inputs written as identical
//! bits to both sides, and bitwise comparison helpers.
#![allow(dead_code)]

pub mod ob;

/// SplitMix64: seeded, portable, no dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 52) as f64) - 1.0
    }
}

/// The oracle's n grid (crosses NX, NB, SMLSIZ and the D&C split points).
pub const N_GRID: &[usize] =
    &[0, 1, 2, 3, 15, 16, 17, 25, 26, 27, 31, 32, 33, 63, 64, 65, 127, 128, 129, 200, 257, 513, 1024];

/// Random dense symmetric column-major matrix (both triangles filled).
pub fn random_sym(n: usize, seed: u64) -> Vec<f64> {
    let mut r = Rng::new(seed);
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            let v = r.uniform();
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
    }
    a
}

/// Fill only one triangle; the other gets garbage that must be ignored.
pub fn poison_other_triangle(a: &mut [f64], n: usize, upper_used: bool) {
    for j in 0..n {
        for i in 0..n {
            let in_upper = i <= j;
            let used = if upper_used { in_upper } else { i >= j };
            if !used {
                a[i + j * n] = 1.0e300 * if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
    }
}

/// Symmetric tridiagonal (d, e) seeded.
pub fn random_tridiag(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut r = Rng::new(seed);
    let d = (0..n).map(|_| r.uniform()).collect();
    let e = (0..n.saturating_sub(1)).map(|_| r.uniform()).collect();
    (d, e)
}

/// Assert two float slices are bitwise identical; reports the first mismatch.
#[track_caller]
pub fn assert_bits(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut first = None;
    let mut count = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            count += 1;
            if first.is_none() {
                first = Some((i, *g, *w));
            }
        }
    }
    if let Some((i, g, w)) = first {
        panic!(
            "{what}: {count} of {} entries differ; first at {i}: got {g:e} ({:#018x}) want {w:e} ({:#018x})",
            got.len(),
            g.to_bits(),
            w.to_bits()
        );
    }
}

/// Random Hermitian planar (re, im), both triangles consistent.
pub fn random_herm(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut r = Rng::new(seed);
    let mut re = vec![0.0; n * n];
    let mut im = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            let a = r.uniform();
            let b = if i == j { 0.0 } else { r.uniform() };
            re[i + j * n] = a;
            im[i + j * n] = b;
            re[j + i * n] = a;
            im[j + i * n] = -b;
        }
    }
    (re, im)
}

/// Interleave planar parts into the oracle's `COMPLEX*16` layout.
pub fn to_c64(re: &[f64], im: &[f64]) -> Vec<lapack_ref::C64> {
    re.iter().zip(im).map(|(&re, &im)| lapack_ref::C64 { re, im }).collect()
}

pub fn from_c64(z: &[lapack_ref::C64]) -> (Vec<f64>, Vec<f64>) {
    (z.iter().map(|c| c.re).collect(), z.iter().map(|c| c.im).collect())
}

/// Tridiagonal fixtures that exercise deflation and splitting.
pub fn tridiag_fixtures(n: usize) -> Vec<(&'static str, Vec<f64>, Vec<f64>)> {
    let mut out = Vec::new();
    let (d, e) = random_tridiag(n, 77 + n as u64);
    out.push(("random", d, e));
    // Glued Wilkinson: W21+ blocks glued by small couplings.
    let m = 21usize;
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n.saturating_sub(1)];
    for i in 0..n {
        let k = i % m;
        d[i] = (k as f64 - (m as f64 - 1.0) / 2.0).abs();
        if i + 1 < n {
            e[i] = if (i + 1) % m == 0 { 1e-10 } else { 1.0 };
        }
    }
    out.push(("glued_wilkinson", d, e));
    // Tight cluster: eigenvalues within ~1e-14 of each other.
    let mut r = Rng::new(3 + n as u64);
    let d = (0..n).map(|i| 1.0 + 1e-14 * (i as f64) + 1e-15 * r.uniform()).collect();
    let e = (0..n.saturating_sub(1)).map(|_| 1e-15 * r.uniform()).collect();
    out.push(("tight_cluster", d, e));
    // Repeated eigenvalues: constant diagonal with tiny off-diagonals in blocks.
    let d = (0..n).map(|i| [2.0, 2.0, -1.0, 2.0][i % 4]).collect();
    let e = (0..n.saturating_sub(1)).map(|i| if i % 5 == 4 { 0.0 } else { 1e-3 }).collect();
    out.push(("repeated", d, e));
    // Near-pole: pairs of nearly equal diagonal values with moderate coupling.
    let d = (0..n).map(|i| (i / 2) as f64 + if i % 2 == 0 { 0.0 } else { 1e-9 }).collect();
    let e = (0..n.saturating_sub(1)).map(|i| if i % 2 == 0 { 1e-7 } else { 0.5 }).collect();
    out.push(("near_pole", d, e));
    // Zero subdiagonal splits (forces dstedc block splitting).
    let (d, mut e) = random_tridiag(n, 5 + n as u64);
    for (i, v) in e.iter_mut().enumerate() {
        if i % 37 == 36 {
            *v = 0.0;
        }
    }
    out.push(("zero_splits", d, e));
    // Diagonal (every off-diagonal zero).
    let (d, _) = random_tridiag(n, 9 + n as u64);
    out.push(("diagonal", d, vec![0.0; n.saturating_sub(1)]));
    out
}

/// Dense matrix fixtures (column-major, symmetric).
pub fn dense_fixtures(n: usize) -> Vec<(&'static str, Vec<f64>)> {
    let mut out = vec![("random", random_sym(n, 500 + n as u64))];
    let mut id = vec![0.0; n * n];
    for i in 0..n {
        id[i + i * n] = 1.0;
    }
    out.push(("identity", id));
    let mut dg = vec![0.0; n * n];
    let (d, _) = random_tridiag(n, 600 + n as u64);
    for i in 0..n {
        dg[i + i * n] = d[i];
    }
    out.push(("diagonal", dg));
    let tiny: Vec<f64> = random_sym(n, 700 + n as u64).iter().map(|v| v * 1e-160).collect();
    out.push(("scaled_tiny", tiny));
    let huge: Vec<f64> = random_sym(n, 800 + n as u64).iter().map(|v| v * 1e160).collect();
    out.push(("scaled_huge", huge));
    // Block-diagonal (zero subdiagonal after reduction on one side).
    let mut bd = random_sym(n, 900 + n as u64);
    let h = n / 2;
    for j in 0..n {
        for i in 0..n {
            if (i < h) != (j < h) {
                bd[i + j * n] = 0.0;
            }
        }
    }
    out.push(("block_diagonal", bd));
    out
}

/// Run `f` inside a dedicated rayon pool with `threads` workers.
pub fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("rayon pool").install(f)
}

/// max |A - Z diag(w) Z^T| and max |Z^T Z - I| for a real symmetric problem
/// (`a` full symmetric column-major).
pub fn sym_residuals(n: usize, a: &[f64], z: &[f64], w: &[f64]) -> (f64, f64) {
    let (mut rec, mut orth) = (0.0f64, 0.0f64);
    for j in 0..n {
        for i in 0..n {
            let mut s = 0.0;
            let mut o = 0.0;
            for k in 0..n {
                s += z[i + k * n] * w[k] * z[j + k * n];
                o += z[k + i * n] * z[k + j * n];
            }
            rec = rec.max((a[i + j * n] - s).abs());
            orth = orth.max((o - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    (rec, orth)
}

/// Hermitian analogue of [`sym_residuals`].
pub fn herm_residuals(n: usize, ar: &[f64], ai: &[f64], zr: &[f64], zi: &[f64], w: &[f64]) -> (f64, f64) {
    let (mut rec, mut orth) = (0.0f64, 0.0f64);
    for j in 0..n {
        for i in 0..n {
            let (mut sr, mut si, mut or, mut oi) = (0.0, 0.0, 0.0, 0.0);
            for k in 0..n {
                // Z(i,k) w_k conj(Z(j,k))
                let (a, b) = (zr[i + k * n], zi[i + k * n]);
                let (c, d) = (zr[j + k * n], -zi[j + k * n]);
                sr += w[k] * (a * c - b * d);
                si += w[k] * (a * d + b * c);
                // conj(Z(k,i)) Z(k,j)
                let (a, b) = (zr[k + i * n], -zi[k + i * n]);
                let (c, d) = (zr[k + j * n], zi[k + j * n]);
                or += a * c - b * d;
                oi += a * d + b * c;
            }
            rec = rec.max((ar[i + j * n] - sr).abs()).max((ai[i + j * n] - si).abs());
            orth = orth.max((or - if i == j { 1.0 } else { 0.0 }).abs()).max(oi.abs());
        }
    }
    (rec, orth)
}

/// Random values with exact zeros, signed zeros and a spread of magnitudes.
pub fn edge(n: usize, seed: u64) -> Vec<f64> {
    let mut r = Rng::new(seed);
    (0..n)
        .map(|_| match r.next_u64() % 12 {
            0 => 0.0,
            1 => -0.0,
            2 => 1e-7 * r.uniform(),
            3 => 1e5 * r.uniform(),
            _ => r.uniform(),
        })
        .collect()
}
