//! Device time per launch of the hot kernels under each variant, at
//! realistic trailing-block sizes (diagnostic; picks the GPU defaults).
//!
//! ```text
//! cargo run --release --features cuda --example device_kernel_bench -- 2048
//! # K8/K9 (the MRRR batches): `stemr <batch-size>` instead of a matrix size
//! cargo run --release --features cuda --example device_kernel_bench -- stemr 4096
//! ```
//!
//! Each measurement: one warm-up launch (JIT), then `reps` launches each
//! closed by a device sync, minimum reported. Variants of one kernel compute
//! identical bits; the bench also checks that.

use std::time::Instant;

use cubecl::prelude::*;
use cubecl::server::Handle;
use lapack_rs::device::geometry::{lda_for, vector_width};
use lapack_rs::device::ops::{self, Lanes, Variant};
use lapack_rs::device::workspace::{read_range, upload};

#[cfg(not(any(feature = "cuda", feature = "cpu")))]
compile_error!("device_kernel_bench needs feature `cuda` or `cpu`");

#[cfg(feature = "cuda")]
type Rt = cubecl_cuda::CudaRuntime;
#[cfg(all(not(feature = "cuda"), feature = "cpu"))]
type Rt = cubecl::cpu::CpuRuntime;

#[cfg(feature = "cuda")]
fn client() -> ComputeClient<Rt> {
    Rt::client(&cubecl_cuda::CudaDevice::default())
}
#[cfg(all(not(feature = "cuda"), feature = "cpu"))]
fn client() -> ComputeClient<Rt> {
    Rt::client(&cubecl::cpu::CpuDevice)
}

fn rnd(len: usize, mut s: u64) -> Vec<f64> {
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

fn timed(c: &ComputeClient<Rt>, reps: usize, f: &dyn Fn()) -> f64 {
    f();
    let _ = cubecl::future::block_on(c.sync());
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            let _ = cubecl::future::block_on(c.sync());
            t.elapsed().as_secs_f64()
        })
        .fold(f64::MAX, f64::min)
}

fn read(c: &ComputeClient<Rt>, h: &Handle, len: usize) -> Vec<f64> {
    let mut v = vec![0.0; len];
    read_range(c, h, len, 0, len, &mut v);
    v
}

fn main() {
    // `stemr <B>`: the K8/K9 batch bench (`docs/cintx-lapack-blas-adoption-plan.md`
    // §9). A separate mode because these are `n <= 13` problems measured per
    // *batch*, not per trailing block — the whole point of §6.1 is that one
    // dispatch has to amortise across hundreds of them.
    if std::env::args().nth(1).as_deref() == Some("stemr") {
        let b: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
        let reps: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(5);
        bench_stemr(b, reps);
        return;
    }
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let c = client();
    let w = vector_width(&c, 8);
    let lda = lda_for(&c, n);
    let lanes = Lanes::new(&c, w);
    println!("backend {} n={n} lda={lda} width={w}", std::any::type_name::<Rt>());

    // Symmetric image with both triangles.
    let a = rnd(n * n, 7);
    let mut img = vec![0.0; lda * n];
    for j in 0..n {
        for i in 0..n {
            img[i + j * lda] = if i >= j { a[i + j * n] } else { a[j + i * n] };
        }
    }
    let hm = upload(&c, &img);

    // K1: device time per launch across trailing sizes, and the driver's full
    // per-column round trip (upload `lda` reals, launch, read `m`), which
    // separates the fixed per-launch cost from the m² compute.
    let mut ms: Vec<usize> = vec![32, 128, 512, n / 4, n / 2, n - 1];
    ms.retain(|&m| m < n);
    ms.sort();
    ms.dedup();
    let hy = c.empty(lda * 8);
    for &m in &ms {
        let off = n - m;
        let x = rnd(m, 11);
        let hx = upload(&c, &x);
        let mut bits: Option<Vec<f64>> = None;
        for (tag, mode) in [("K1    ", 1u8), ("K1col ", 2), ("K1colv", 3)].repeat(3) {
            ops::set_symv_mode(mode);
            let t_launch =
                timed(&c, reps, &|| ops::symv::<Rt, f64>(&c, &lanes, &hm, lda, n, off, m, &hx, &hy, 1.0, false));
            let out = read(&c, &hy, lda)[off..off + m].to_vec();
            let same =
                bits.get_or_insert_with(|| out.clone()).iter().zip(&out).all(|(a, b)| a.to_bits() == b.to_bits());
            println!(
                "{tag:<12} m={m:<5} launch+sync {:>8.3} ms | {:>6.2} GB/s load traffic | {}",
                t_launch * 1e3,
                (m * m * 8) as f64 / t_launch / 1e9,
                if same { "bits ok" } else { "BITS DIFFER" }
            );
        }
        ops::set_symv_mode(0);
    }

    // K3 gemm_t (DLARFB C2ᵀV2) at the first block.
    let kpad = 32usize.div_ceil(w) * w;
    let cnt = n - 32;
    let hv = upload(&c, &rnd(cnt * kpad, 17));
    let hw_in = upload(&c, &rnd(n * kpad, 19));
    let hw_out = c.empty(n * kpad * 8);
    let mut reference = None;
    for (tag, variant) in [("K3 gemm_t", Variant::Simple), ("K3r gemm_t", Variant::RowTiled)].repeat(3) {
        ops::set_variant(variant);
        let t = timed(&c, reps, &|| {
            ops::gemm_t::<Rt, f64>(&c, &lanes, &hm, lda, n, &hv, cnt, &hw_in, &hw_out, n, kpad, 32, 0, cnt)
        });
        let out = read(&c, &hw_out, n * kpad);
        let same =
            reference.get_or_insert_with(|| out.clone()).iter().zip(&out).all(|(a, b)| a.to_bits() == b.to_bits());
        println!("{tag:<14} cnt={cnt:<5} {:>9.3} ms  {}", t * 1e3, if same { "bits ok" } else { "BITS DIFFER" });
    }
    ops::set_variant(Variant::Auto);
}

/// K9 (Path C): `B` Wheeler-sized problems in one dispatch, against the same
/// `B` problems one dispatch each — the §6.1 cost model, measured.
///
/// The batched form is the one cintx would launch standalone; the inline
/// callee (surface 1) pays neither, which is why it is the production shape.
fn bench_stemr(b: usize, reps: usize) {
    let c = client();
    println!("backend {} K9 batch B={b} reps={reps}", std::any::type_name::<Rt>());
    let consts = ops::cint_stemr_consts();
    let big = consts[lapack_rs::device::kernels::cint_stemr::CC_BIG as usize];

    // The Wheeler size distribution: n = 3..=13, cycling.
    let ns: Vec<usize> = (0..b).map(|i| 3 + i % 11).collect();
    let n_max = *ns.iter().max().expect("ns") as u32;
    let mut nptr = vec![0u32];
    let mut vptr = vec![0u32];
    let (mut diag, mut offd) = (Vec::new(), Vec::new());
    for (k, &n) in ns.iter().enumerate() {
        let d = rnd(n, 31 + k as u64);
        let e = rnd(n, 97 + k as u64);
        for i in 0..n {
            diag.push(d[i].abs() + 0.5);
            offd.push(if i + 1 < n { e[i] } else { 0.0 });
        }
        nptr.push(diag.len() as u32);
        vptr.push(vptr[k] + (n * n) as u32);
    }
    let dlen = diag.len();
    let vlen = *vptr.last().expect("vptr") as usize;
    let wper = 9 * n_max as usize + 1;
    let iwper = 5 * n_max as usize;

    let hdiag = upload(&c, &diag);
    let hoffd = upload(&c, &offd);
    let hnptr = upload(&c, &nptr);
    let hvptr = upload(&c, &vptr);
    let heig = upload(&c, &vec![0.0f64; dlen]);
    let hvec = upload(&c, &vec![0.0f64; vlen]);
    let hwork = upload(&c, &vec![0.0f64; b * wper]);
    let hiwork = upload(&c, &vec![0u32; b * iwper]);
    let hcc = upload(&c, &consts);
    let hsc = upload(&c, &vec![0.0f64; b * 16]);
    let hicnt = upload(&c, &vec![0u32; b * 4]);
    let hinfo = upload(&c, &vec![0u32; b]);
    let batch = ops::CintStemrBatch {
        diag: &hdiag,
        offd: &hoffd,
        nptr: &hnptr,
        vptr: &hvptr,
        eig: &heig,
        vec: &hvec,
        work: &hwork,
        iwork: &hiwork,
        cc: &hcc,
        sc: &hsc,
        icnt: &hicnt,
        info: &hinfo,
    };

    // One launch first, to check the kernel lowered at all: on the CubeCL CPU
    // runtime the K9 composite fails to lower, the launch silently no-ops and
    // the "timings" below would be the cost of a failed compile
    // (`docs/cubecl-cpu-lowering-limit.md`). This mode is for a GPU backend.
    ops::cint_stemr_batch::<Rt, f64>(&c, &batch, b, dlen, vlen, n_max, f64::EPSILON, big);
    let probe = read(&c, &heig, dlen.min(64));
    if probe.iter().all(|&v| v == 0.0) {
        println!(
            "K9 did not run on this backend (the launch wrote nothing). cubecl-cpu 0.10 cannot\n\
             lower the composite kernel; see docs/cubecl-cpu-lowering-limit.md. Re-run with\n\
             --features cuda on an f64 GPU."
        );
        return;
    }

    let t_batch =
        timed(&c, reps, &|| ops::cint_stemr_batch::<Rt, f64>(&c, &batch, b, dlen, vlen, n_max, f64::EPSILON, big));
    println!(
        "K9 batch      B={b:<6} {:>9.3} ms/dispatch | {:>9.3} us/problem",
        t_batch * 1e3,
        t_batch * 1e6 / b as f64
    );

    // The same work as `B` one-problem dispatches: §6.1's "launch + read-back
    // is ~10^5 the arithmetic" claim, on this machine.
    let one = 64.min(b);
    let t_one = timed(&c, reps, &|| {
        for k in 0..one {
            let n = ns[k];
            ops::cint_stemr_one::<Rt, f64>(
                &c,
                &hdiag,
                &hoffd,
                &heig,
                &hvec,
                &hwork,
                &hiwork,
                &hcc,
                &hsc,
                &hicnt,
                &hinfo,
                n,
                f64::EPSILON,
                big,
            );
        }
    });
    println!(
        "K9 one-at-a-time  x{one:<4} {:>9.3} ms total     | {:>9.3} us/problem  ({:.1}x the batch)",
        t_one * 1e3,
        t_one * 1e6 / one as f64,
        (t_one / one as f64) / (t_batch / b as f64)
    );
}
