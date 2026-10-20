//! Every device kernel, in f32, on ROCm/HIP (32-lane planes, its own vector
//! width) and on the CPU runtime (whose f64 kernels are bit-exact against the
//! host reference), on identical inputs. With
//! `HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off` (probe: no contraction)
//! the outputs must agree bit for bit — any difference is an indexing or
//! geometry bug; with contraction they must agree to a roundoff bound.
//! Both GPU kernel variants (simple and row-tiled K2r/K3r) are checked.
//! Features `rocm,cpu`.

#![cfg(all(feature = "rocm", feature = "cpu"))]

mod common;
use common::Rng;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl_hip::{AmdDevice, HipRuntime};
use lapack_rs::device::geometry::vector_width;
use lapack_rs::device::ops::{self, Chain, Chunk, Lanes, ZHandle};
use lapack_rs::device::probe::contraction_probe;
use lapack_rs::device::workspace::{read_range, upload};

const N: usize = 75;
const LDA: usize = 96; // a multiple of every width and of the 32-lane plane
const KPAD: usize = 32;

fn rnd(len: usize, seed: u64, zeros: bool) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..len).map(|i| if zeros && i % 7 == 3 { 0.0 } else { (r.uniform() * 2.0 - 1.0) as f32 }).collect()
}

fn sym_image(seed: u64) -> Vec<f32> {
    let a = rnd(N * N, seed, false);
    let mut s = vec![0.0f32; LDA * N];
    for j in 0..N {
        for i in 0..N {
            s[i + j * LDA] = if i >= j { a[i + j * N] } else { a[j + i * N] };
        }
    }
    s
}

fn herm_image(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let re = sym_image(seed);
    let a = rnd(N * N, seed + 1, false);
    let mut im = vec![0.0f32; LDA * N];
    for j in 0..N {
        for i in 0..N {
            im[i + j * LDA] = if i > j {
                a[i + j * N]
            } else if i < j {
                -a[j + i * N]
            } else {
                0.0
            };
        }
    }
    (re, im)
}

fn read<R: Runtime>(c: &ComputeClient<R>, h: &Handle, len: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; len];
    read_range(c, h, len, 0, len, &mut v);
    v
}

fn zup<R: Runtime>(c: &ComputeClient<R>, re: &[f32], im: &[f32]) -> ZHandle {
    ZHandle { re: upload(c, re), im: upload(c, im) }
}

fn zread<R: Runtime>(c: &ComputeClient<R>, h: &ZHandle, len: usize) -> Vec<f32> {
    let mut v = read(c, &h.re, len);
    v.extend(read(c, &h.im, len));
    v
}

/// Run every kernel once on fixed inputs; return named outputs.
fn exercise<R: Runtime>(c: &ComputeClient<R>) -> Vec<(String, Vec<f32>)> {
    let w = vector_width(c, 4);
    assert!(LDA.is_multiple_of(w) && KPAD.is_multiple_of(w));
    let lanes = Lanes::new(c, w);
    let mut out = Vec::new();

    // K1 / K2.
    for upper in [false, true] {
        let hm = upload(c, &sym_image(1));
        let (off, m) = (5usize, N - 5);
        let hx = upload(c, &rnd(m, 2, true));
        let hy = c.empty(LDA * 4);
        ops::symv::<R, f32>(c, &lanes, &hm, LDA, N, off, m, &hx, &hy, -0.75, upper);
        out.push((format!("K1 upper={upper}"), read(c, &hy, LDA)[off..off + m].to_vec()));
        let nb = 4;
        let mut vw = vec![0.0f32; 2 * nb * LDA];
        let src = rnd(2 * nb * LDA, 3, true);
        for l in 0..2 * nb {
            for r in 7..N {
                vw[r + l * LDA] = src[r + l * LDA];
            }
        }
        let hvw = upload(c, &vw);
        ops::syr2k::<R, f32>(c, &lanes, &hm, LDA, N, 7, N - 7, &hvw, nb, -1.0, upper);
        out.push((format!("K2 upper={upper}"), read(c, &hm, LDA * N)));
    }

    // K7 rotations, K3m merge + copy-back, swap.
    let hz = upload(c, &rnd(LDA * N, 4, false));
    let cols: Vec<u32> = vec![3, 9, 9, 40, 12, 13, 70, 2];
    let cs = rnd(8, 5, false);
    ops::rotations::<R, f32>(c, &lanes, &hz, LDA, N, &upload(c, &cols), &upload(c, &cs), 4, 6, 71);
    ops::swap_cols::<R, f32>(c, &lanes, &hz, LDA, N, 1, 50);
    let (g0, n, k) = (10usize, 60usize, 41usize);
    let perm: Vec<u32> = (0..33u32).map(|i| g0 as u32 + (i * 7) % n as u32).collect();
    let perm4: Vec<u32> = (0..(n - k) as u32).map(|i| g0 as u32 + (i * 11 + 3) % n as u32).collect();
    let s = rnd(33 * k, 6, true);
    let (hp, hp4, hs) = (upload(c, &perm), upload(c, &perm4), upload(c, &s));
    for (lo, hi) in [(g0, g0 + 23), (g0 + 23, g0 + n)] {
        let chunk = Chunk { rv0: lo / w, rvecs: hi.div_ceil(w) - lo / w };
        let dst = c.empty(chunk.rvecs * w * n * 4);
        ops::merge_rows::<R, f32>(c, &lanes, &hz, LDA, N, &dst, chunk, n, k, &hp, 33, &hp4, &hs, 33, 0, lo, hi);
        ops::copy_back::<R, f32>(c, &lanes, &hz, LDA, N, &dst, chunk, g0, n, lo, hi);
    }
    out.push(("K7/K3m rotations+swap+merge".into(), read(c, &hz, LDA * N)));

    // K3 dlarfb steps.
    let (nn, kk) = (N, 29usize);
    let hw_in = upload(c, &rnd(nn * KPAD, 7, true));
    let hcoef = upload(c, &rnd(KPAD * KPAD, 8, true));
    let hdiag = upload(c, &rnd(KPAD, 9, false));
    for (above, asc, scale) in
        [(true, true, false), (true, true, true), (false, false, false), (false, true, false), (false, false, true)]
    {
        let hw_out = c.empty(nn * KPAD * 4);
        ops::trmm_chain::<R, f32>(
            c,
            &lanes,
            &hw_in,
            &hw_out,
            &hcoef,
            &hdiag,
            nn,
            kk,
            KPAD,
            Chain { above, asc, scale },
        );
        out.push((format!("K3 trmm {above} {asc} {scale}"), read(c, &hw_out, nn * KPAD)));
    }
    let hc = upload(c, &rnd(LDA * N, 10, false));
    let nrect = 40usize;
    let hv_rm = upload(c, &rnd(nrect * KPAD, 11, true));
    let hw_out = c.empty(nn * KPAD * 4);
    ops::gemm_t::<R, f32>(c, &lanes, &hc, LDA, N, &hv_rm, nrect, &hw_in, &hw_out, nn, KPAD, 33, 0, nrect);
    out.push(("K3 gemm_t".into(), read(c, &hw_out, nn * KPAD)));
    let crow = 33usize;
    let pad = crow % w;
    let ldv = (pad + nrect).div_ceil(w) * w;
    let vsrc = rnd(nrect * kk, 12, true);
    let mut v_cm = vec![0.0f32; ldv * kk];
    for j in 0..kk {
        for l in 0..nrect {
            v_cm[pad + l + j * ldv] = vsrc[l + j * nrect];
        }
    }
    ops::gemm_n::<R, f32>(c, w, &hc, LDA, N, &upload(c, &v_cm), ldv, &hw_out, nn, KPAD, kk, crow, nrect);
    let hw2 = c.empty(nn * KPAD * 4);
    ops::rows_to_w::<R, f32>(c, &hc, LDA, N, &hw2, nn, KPAD, kk, 3);
    ops::sub_w_from_rows::<R, f32>(c, &hc, LDA, N, &hw_in, nn, KPAD, kk, 50);
    out.push(("K3 gemm_n + K7 sub_w".into(), read(c, &hc, LDA * N)));
    out.push(("K7 rows_to_w".into(), read(c, &hw2, nn * KPAD)));

    // Complex K1c / K2c / K3c.
    let alpha = (-0.5f32, 0.25f32);
    for upper in [false, true] {
        let (re, im) = herm_image(20);
        let hm = zup(c, &re, &im);
        let (off, m) = (3usize, N - 3);
        let hx = zup(c, &rnd(m, 21, true), &rnd(m, 22, true));
        let hy = ZHandle { re: c.empty(LDA * 4), im: c.empty(LDA * 4) };
        ops::hemv::<R, f32>(c, &lanes, &hm, LDA, N, off, m, &hx, &hy, alpha, upper);
        let y = zread(c, &hy, LDA);
        out.push((format!("K1c upper={upper}"), [&y[off..off + m], &y[LDA + off..LDA + off + m]].concat()));
        let nb = 3;
        let (mut vr, mut vi) = (vec![0.0f32; 2 * nb * LDA], vec![0.0f32; 2 * nb * LDA]);
        let (sr, si) = (rnd(2 * nb * LDA, 23, true), rnd(2 * nb * LDA, 24, true));
        for l in 0..2 * nb {
            for r in 9..N {
                vr[r + l * LDA] = sr[r + l * LDA];
                vi[r + l * LDA] = si[r + l * LDA];
            }
        }
        ops::her2k::<R, f32>(c, &lanes, &hm, LDA, N, 9, N - 9, &zup(c, &vr, &vi), nb, alpha, upper);
        out.push((format!("K2c upper={upper}"), zread(c, &hm, LDA * N)));
    }
    let hzw = zup(c, &rnd(nn * KPAD, 30, true), &rnd(nn * KPAD, 31, true));
    let hzc = zup(c, &rnd(KPAD * KPAD, 32, true), &rnd(KPAD * KPAD, 33, true));
    let mut diag = rnd(3 * KPAD, 34, false);
    for j in 0..KPAD {
        diag[2 * KPAD + j] = if j % 3 == 0 { 0.0 } else { 1.0 };
    }
    let hzd = upload(c, &diag);
    for (above, asc) in [(true, true), (false, false), (false, true)] {
        let o = ZHandle { re: c.empty(nn * KPAD * 4), im: c.empty(nn * KPAD * 4) };
        ops::ztrmm_chain::<R, f32>(c, &lanes, &hzw, &o, &hzc, &hzd, nn, kk, KPAD, above, asc);
        out.push((format!("K3c ztrmm {above} {asc}"), zread(c, &o, nn * KPAD)));
    }
    let hzz = zup(c, &rnd(LDA * N, 40, false), &rnd(LDA * N, 41, false));
    let hzv = zup(c, &rnd(nrect * KPAD, 42, true), &rnd(nrect * KPAD, 43, true));
    let o = ZHandle { re: c.empty(nn * KPAD * 4), im: c.empty(nn * KPAD * 4) };
    ops::zgemm_c::<R, f32>(c, &lanes, &hzz, LDA, N, &hzv, nrect, &hzw, &o, nn, KPAD, 33, nrect, alpha, (1.0, 0.0));
    out.push(("K3c zgemm_c".into(), zread(c, &o, nn * KPAD)));
    let (mut cr, mut ci) = (vec![0.0f32; ldv * kk], vec![0.0f32; ldv * kk]);
    let (s1, s2) = (rnd(nrect * kk, 44, true), rnd(nrect * kk, 45, true));
    for j in 0..kk {
        for l in 0..nrect {
            cr[pad + l + j * ldv] = s1[l + j * nrect];
            ci[pad + l + j * ldv] = s2[l + j * nrect];
        }
    }
    ops::zgemm_n::<R, f32>(c, w, &hzz, LDA, N, &zup(c, &cr, &ci), ldv, &o, nn, KPAD, kk, crow, nrect, (-1.0, -0.0));
    let o2 = ZHandle { re: c.empty(nn * KPAD * 4), im: c.empty(nn * KPAD * 4) };
    ops::zrows_to_w::<R, f32>(c, &hzz, LDA, N, &o2, nn, KPAD, kk, 5);
    ops::zsub_w_from_rows::<R, f32>(c, &hzz, LDA, N, &hzw, nn, KPAD, kk, 44);
    out.push(("K3c zgemm_n + zsub_w".into(), zread(c, &hzz, LDA * N)));
    out.push(("K7c zrows_to_w".into(), zread(c, &o2, nn * KPAD)));
    let hf = upload(c, &rnd(100, 50, false));
    ops::fill_zero::<R, f32>(c, &hf, 77);
    out.push(("K7 fill_zero".into(), read(c, &hf, 100)));
    out
}

#[test]
fn kernels_agree_between_rocm_and_cpu_runtime() {
    let gpu = HipRuntime::client(&AmdDevice::default());
    let cpu = CpuRuntime::client(&CpuDevice);
    let exact = contraction_probe::<HipRuntime, f32>(&gpu, f32::EPSILON, 1.0, 2.0).no_contraction();
    assert!(contraction_probe::<CpuRuntime, f32>(&cpu, f32::EPSILON, 1.0, 2.0).no_contraction());
    println!(
        "HIP f32 width {} ({}), CPU f32 width {}",
        vector_width(&gpu, 4),
        if exact { "bit-exact gate" } else { "tolerance gate" },
        vector_width(&cpu, 4)
    );
    let h = exercise(&cpu);
    for variant in [ops::Variant::Simple, ops::Variant::RowTiled] {
        ops::set_variant(variant);
        let g = exercise(&gpu);
        println!("-- HIP variant {variant:?}");
        compare(&g, &h, exact);
    }
    ops::set_variant(ops::Variant::Auto);
}

fn compare(g: &[(String, Vec<f32>)], h: &[(String, Vec<f32>)], exact: bool) {
    assert_eq!(g.len(), h.len());
    let mut bad = Vec::new();
    for ((name, a), (_, b)) in g.iter().zip(h) {
        assert_eq!(a.len(), b.len(), "{name}");
        let diff = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1.0)).fold(0.0f32, f32::max);
        println!("{name:<32} {} values, {diff} differ, worst rel {worst:e}", a.len());
        if (exact && diff > 0) || worst > 1e-4 {
            bad.push(name.clone());
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}
