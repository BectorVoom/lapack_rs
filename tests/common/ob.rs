#![allow(dead_code)]
//! Seeded inputs and bitwise comparison against the OpenBLAS oracle.

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
    /// Mostly uniform, with exact zeros, signed zeros and spread magnitudes.
    pub fn edge(&mut self) -> f64 {
        match self.next_u64() % 16 {
            0 => 0.0,
            1 => -0.0,
            2 => 1e-9 * self.uniform(),
            3 => 1e7 * self.uniform(),
            _ => self.uniform(),
        }
    }
    pub fn vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform()).collect()
    }
    pub fn vec_edge(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.edge()).collect()
    }
}

/// Panics with the first differing index when `got` and `want` differ in any bit.
#[track_caller]
pub fn assert_bits(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut count = 0usize;
    let mut first = None;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            count += 1;
            first.get_or_insert((i, *g, *w));
        }
    }
    if let Some((i, g, w)) = first {
        panic!("{what}: {count}/{} differ; first at {i}: got {g:e} want {w:e}", got.len());
    }
}

/// Size grid crossing 4/8/16/32/64 lane and block boundaries.
pub fn sizes(max: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..=70).collect();
    v.extend([95, 96, 97, 127, 128, 129, 191, 192, 193, 255, 256, 257, 300, 383, 384, 385, 511, 512, 513, 700]);
    v.retain(|&n| n <= max);
    v
}
