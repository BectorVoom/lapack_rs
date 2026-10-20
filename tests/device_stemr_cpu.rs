//! K8 (the MRRR leaf kernels `DLAEV2`, `DLARRK`, `DLANEG`, `DLAR1V`) launched
//! on the CubeCL CPU runtime (f64) against the host transcriptions
//! (`src/larr.rs`, `src/aux.rs::dlaev2`), bit for bit.
//!
//! Includes the `ITMAX` **integer** gate of
//! `docs/cintx-lapack-blas-adoption-plan.md` §7.2: `larrk_itmax` lowers two
//! `log` calls to the backend's libm, and any disagreement with the glibc
//! `log` gfortran calls that crosses an integer boundary changes `DLARRK`'s
//! halving count and so changes `W`. Comparing the integer here turns that
//! into a loud failure instead of a mysterious one-ULP eigenvalue.

#![cfg(feature = "cpu")]

mod common;
use common::*;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;
use lapack_rs::device::ops::{self, Lar1vBatch};
use lapack_rs::device::workspace::{read_range, upload};
use lapack_rs::testing as t;

fn client() -> ComputeClient<CpuRuntime> {
    CpuRuntime::client(&CpuDevice)
}

/// A batch of `nprob` tridiagonals of the sizes in `ns`, concatenated with the
/// `nptr` offset table the batched kernels take.
struct Batch {
    nptr: Vec<u32>,
    d: Vec<f64>,
    e: Vec<f64>,
}

fn batch(ns: &[usize], seed: u64) -> Batch {
    let mut r = Rng::new(seed);
    let mut nptr = vec![0u32];
    let (mut d, mut e) = (Vec::new(), Vec::new());
    for (b, &n) in ns.iter().enumerate() {
        for i in 0..n {
            d.push(match b % 4 {
                0 => r.uniform(),
                1 => i as f64 + 1.0,
                2 => 1.0,
                _ => r.uniform() * 1e6,
            });
            // The last entry of each block is scratch for `e`.
            e.push(if i + 1 == n { 0.0 } else { r.uniform() });
        }
        nptr.push(d.len() as u32);
    }
    Batch { nptr, d, e }
}

#[test]
fn laev2_kernel_matches_host() {
    let client = client();
    let mut r = Rng::new(31_001);
    let mut abc: Vec<f64> = Vec::new();
    // Every branch of `dlaev2.f`: SM </=/> 0, ADF </=/> AB, DF >= 0, AB = 0,
    // and the SGN1 == SGN2 swap.
    for &(a, b, c) in &[
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (-1.0, 0.0, -1.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.5, -1.0),
        (-3.0, 2.0, 5.0),
        (2.0, 1.0, 2.0),
        (0.0, 1.0, 0.0),
        (1.0, 2.0, 1.0),
    ] {
        abc.extend_from_slice(&[a, b, c]);
    }
    for _ in 0..500 {
        let s = |r: &mut Rng| (2.0 * r.uniform() - 1.0) * 10f64.powi((6.0 * r.uniform() - 3.0) as i32);
        let (a, b, c) = (s(&mut r), s(&mut r), s(&mut r));
        abc.extend_from_slice(&[a, b, c]);
        abc.extend_from_slice(&[a, 0.0, c]);
    }
    let nprob = abc.len() / 3;

    let habc = upload(&client, &abc);
    let hout = client.empty(4 * nprob * size_of::<f64>());
    ops::laev2_batch::<CpuRuntime, f64>(&client, &habc, &hout, nprob);
    let mut got = vec![0.0f64; 4 * nprob];
    read_range(&client, &hout, 4 * nprob, 0, 4 * nprob, &mut got);

    for b in 0..nprob {
        let (a, bb, c) = (abc[3 * b], abc[3 * b + 1], abc[3 * b + 2]);
        let (rt1, rt2, cs1, sn1) = t::dlaev2(a, bb, c);
        assert_bits(&format!("dlaev2_c b={b} a={a:e} b={bb:e} c={c:e}"), &got[4 * b..4 * b + 4], &[rt1, rt2, cs1, sn1]);
    }
}

#[test]
fn larrk_kernel_and_itmax_match_host() {
    let client = client();
    let ns: Vec<usize> = vec![1, 2, 3, 5, 7, 13, 13, 8, 4, 11];
    let bt = batch(&ns, 31_002);
    let nprob = ns.len();
    let nw = *ns.iter().max().expect("ns");
    let dlen = bt.d.len();
    let e2: Vec<f64> = bt.e.iter().map(|v| v * v).collect();

    // Gerschgorin brackets, as DLARRE builds them per block.
    let (mut gl, mut gu) = (vec![0.0f64; nprob], vec![0.0f64; nprob]);
    for b in 0..nprob {
        let (lo, hi) = (bt.nptr[b] as usize, bt.nptr[b + 1] as usize);
        let (mut l, mut u) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in lo..hi {
            let em = if i == lo { 0.0 } else { bt.e[i - 1].abs() };
            let ep = if i + 1 < hi { bt.e[i].abs() } else { 0.0 };
            l = l.min(bt.d[i] - em - ep);
            u = u.max(bt.d[i] + em + ep);
        }
        gl[b] = l;
        gu[b] = u;
    }

    let pivmin = f64::MIN_POSITIVE;
    let reltol = f64::EPSILON.sqrt();
    let eps = t::DLAMCH.iter().find(|(c, _)| *c == 'P').expect("P").1;

    // The ITMAX integer gate: the host value, and the device's own `log`.
    let tnorm: Vec<f64> = (0..nprob).map(|b| gl[b].abs().max(gu[b].abs())).collect();
    let pivv = vec![pivmin; nprob];
    let htn = upload(&client, &tnorm);
    let hpv = upload(&client, &pivv);
    let hit = client.empty(nprob * size_of::<u32>());
    ops::larrk_itmax_batch::<CpuRuntime, f64>(&client, &htn, &hpv, 2.0f64.ln(), &hit, nprob);
    let mut dev_itmax = vec![0u32; nprob];
    read_range(&client, &hit, nprob, 0, nprob, &mut dev_itmax);
    let host_itmax: Vec<u32> = (0..nprob).map(|b| t::larrk_itmax(tnorm[b], pivmin) as u32).collect();
    assert_eq!(dev_itmax, host_itmax, "larrk_itmax is an INTEGER gate (plan §7.2)");

    let hd = upload(&client, &bt.d);
    let he2 = upload(&client, &e2);
    let hnptr = upload(&client, &bt.nptr);
    let hgl = upload(&client, &gl);
    let hgu = upload(&client, &gu);
    let hc = upload(&client, &[reltol, pivmin, eps]);
    let hw = client.empty(nprob * nw * size_of::<f64>());
    let hwe = client.empty(nprob * nw * size_of::<f64>());
    let hi = client.empty(nprob * nw * size_of::<u32>());
    ops::larrk_batch::<CpuRuntime, f64>(
        &client, &hd, &he2, &hnptr, &hgl, &hgu, &hc, &hit, &hw, &hwe, &hi, nprob, nw, dlen,
    );
    let mut w = vec![0.0f64; nprob * nw];
    let mut werr = vec![0.0f64; nprob * nw];
    let mut info = vec![0u32; nprob * nw];
    read_range(&client, &hw, nprob * nw, 0, nprob * nw, &mut w);
    read_range(&client, &hwe, nprob * nw, 0, nprob * nw, &mut werr);
    read_range(&client, &hi, nprob * nw, 0, nprob * nw, &mut info);

    for b in 0..nprob {
        let (lo, hi_) = (bt.nptr[b] as usize, bt.nptr[b + 1] as usize);
        let n = hi_ - lo;
        let mut dblk = bt.d[lo..hi_].to_vec();
        let mut e2blk = e2[lo..hi_].to_vec();
        for k in 0..n {
            let (hw_, hwe_, hinfo) =
                unsafe { t::dlarrk(n, k as i32 + 1, gl[b], gu[b], &mut dblk, &mut e2blk, pivmin, reltol) };
            let tag = format!("dlarrk_c b={b} n={n} iw={}", k + 1);
            assert_bits(&format!("{tag} W/WERR"), &[w[b * nw + k], werr[b * nw + k]], &[hw_, hwe_]);
            // The kernel reports the reference's -1 as 1.
            assert_eq!(info[b * nw + k], if hinfo == 0 { 0 } else { 1 }, "{tag} INFO");
        }
    }
}

#[test]
fn laneg_kernel_matches_host() {
    let client = client();
    // Long enough to cross DLANEG's BLKLEN = 128 blocking in both parts.
    let ns: Vec<usize> = vec![1, 2, 5, 13, 127, 128, 129, 200, 257];
    let bt = batch(&ns, 31_003);
    let nprob = ns.len();
    let dlen = bt.d.len();
    // L D L^T factors, as DLARRB passes them.
    let lld: Vec<f64> = (0..dlen)
        .map(|i| {
            let den = if bt.d[i].abs() < 1e-3 { 1e-3 } else { bt.d[i] };
            let l = bt.e[i] / den;
            l * l * bt.d[i]
        })
        .collect();
    let pivmin = f64::MIN_POSITIVE;

    let hd = upload(&client, &bt.d);
    let hlld = upload(&client, &lld);
    let hnptr = upload(&client, &bt.nptr);
    for &sig in &[0.0f64, 0.5, -0.5, 1e-14, 1e6] {
        for rsel in 0..3usize {
            let sigma = vec![sig; nprob];
            let rs: Vec<u32> = ns
                .iter()
                .map(|&n| match rsel {
                    0 => 1u32,
                    1 => ((n + 1) / 2) as u32,
                    _ => n as u32,
                })
                .collect();
            let hs = upload(&client, &sigma);
            let hr = upload(&client, &rs);
            let hn = client.empty(nprob * size_of::<u32>());
            ops::laneg_batch::<CpuRuntime, f64>(&client, &hd, &hlld, &hnptr, &hs, &hr, &hn, nprob, dlen);
            let mut got = vec![0u32; nprob];
            read_range(&client, &hn, nprob, 0, nprob, &mut got);
            for b in 0..nprob {
                let (lo, hi) = (bt.nptr[b] as usize, bt.nptr[b + 1] as usize);
                let n = hi - lo;
                let mut dblk = bt.d[lo..hi].to_vec();
                let mut lblk = lld[lo..hi].to_vec();
                let want = unsafe { t::dlaneg(n, &mut dblk, &mut lblk, sig, pivmin, rs[b] as usize) };
                assert_eq!(got[b] as i32, want, "dlaneg_c b={b} n={n} sigma={sig:e} r={}", rs[b]);
            }
        }
    }
}

#[test]
fn lar1v_kernel_matches_host() {
    let client = client();
    let ns: Vec<usize> = vec![1, 2, 3, 5, 7, 13, 31, 64];
    let bt = batch(&ns, 31_004);
    let nprob = ns.len();
    let dlen = bt.d.len();
    let (mut l, mut ld, mut lld) = (vec![0.0; dlen], vec![0.0; dlen], vec![0.0; dlen]);
    for i in 0..dlen {
        let den = if bt.d[i].abs() < 1e-3 { 1e-3 } else { bt.d[i] };
        l[i] = bt.e[i] / den;
        ld[i] = l[i] * bt.d[i];
        lld[i] = ld[i] * l[i];
    }
    let pivmin = f64::MIN_POSITIVE;
    let eps = t::DLAMCH.iter().find(|(c, _)| *c == 'P').expect("P").1;

    let hd = upload(&client, &bt.d);
    let hl = upload(&client, &l);
    let hld = upload(&client, &ld);
    let hlld = upload(&client, &lld);
    let hnptr = upload(&client, &bt.nptr);

    for &lam in &[0.0f64, 0.5, -0.25, 1e-13, 3.0] {
        for &gaptol in &[0.0f64, 1e-8, 1e300] {
            for wantnc in [true, false] {
                for rsel in 0..4usize {
                    let lambda = vec![lam; nprob];
                    let rin: Vec<u32> = ns
                        .iter()
                        .map(|&n| match rsel {
                            0 => 0u32,
                            1 => 1,
                            2 => ((n + 1) / 2) as u32,
                            _ => n as u32,
                        })
                        .collect();
                    let wnc = vec![u32::from(wantnc); nprob];
                    let hlam = upload(&client, &lambda);
                    let hr = upload(&client, &rin);
                    let hc = upload(&client, &[gaptol, pivmin, eps]);
                    let hwnc = upload(&client, &wnc);
                    // `client.empty` is uninitialised; `DLAR1V` leaves the
                    // tail of `Z` and any unvisited `LPLUS`/`UMINUS` slot
                    // untouched (the reference's callers pre-zero them), so
                    // both sides must start from the same zeros.
                    let hz = upload(&client, &vec![0.0f64; dlen]);
                    let hwork = upload(&client, &vec![0.0f64; 4 * dlen]);
                    let hisz = client.empty(2 * nprob * size_of::<u32>());
                    let hout = client.empty(4 * nprob * size_of::<f64>());
                    let hiout = client.empty(2 * nprob * size_of::<u32>());
                    ops::lar1v_batch::<CpuRuntime, f64>(
                        &client,
                        &Lar1vBatch {
                            d: &hd,
                            l: &hl,
                            ld: &hld,
                            lld: &hlld,
                            nptr: &hnptr,
                            lambda: &hlam,
                            r_in: &hr,
                            consts: &hc,
                            wantnc: &hwnc,
                            z: &hz,
                            work: &hwork,
                            isuppz: &hisz,
                            out: &hout,
                            iout: &hiout,
                        },
                        nprob,
                        dlen,
                    );
                    let mut z = vec![0.0f64; dlen];
                    let mut work = vec![0.0f64; 4 * dlen];
                    let mut isz = vec![0u32; 2 * nprob];
                    let mut out = vec![0.0f64; 4 * nprob];
                    let mut iout = vec![0u32; 2 * nprob];
                    read_range(&client, &hz, dlen, 0, dlen, &mut z);
                    read_range(&client, &hwork, 4 * dlen, 0, 4 * dlen, &mut work);
                    read_range(&client, &hisz, 2 * nprob, 0, 2 * nprob, &mut isz);
                    read_range(&client, &hout, 4 * nprob, 0, 4 * nprob, &mut out);
                    read_range(&client, &hiout, 2 * nprob, 0, 2 * nprob, &mut iout);

                    for b in 0..nprob {
                        let (lo, hi) = (bt.nptr[b] as usize, bt.nptr[b + 1] as usize);
                        let n = hi - lo;
                        let tag = format!("dlar1v_c b={b} n={n} lam={lam:e} gt={gaptol:e} nc={wantnc} r={}", rin[b]);
                        let (mut dblk, mut lblk) = (bt.d[lo..hi].to_vec(), l[lo..hi].to_vec());
                        let (mut ldblk, mut lldblk) = (ld[lo..hi].to_vec(), lld[lo..hi].to_vec());
                        let mut hz_ = vec![0.0f64; n];
                        let mut hw_ = vec![0.0f64; 4 * n];
                        let want = unsafe {
                            t::dlar1v(
                                n,
                                1,
                                n,
                                lam,
                                &mut dblk,
                                &mut lblk,
                                &mut ldblk,
                                &mut lldblk,
                                pivmin,
                                gaptol,
                                &mut hz_,
                                wantnc,
                                rin[b] as usize,
                                &mut hw_,
                            )
                        };
                        assert_bits(&format!("{tag} Z"), &z[lo..hi], &hz_);
                        assert_bits(&format!("{tag} WORK"), &work[4 * lo..4 * lo + 4 * n], &hw_);
                        assert_bits(
                            &format!("{tag} scalars"),
                            &out[4 * b..4 * b + 4],
                            &[want.2, want.6, want.7, want.8],
                        );
                        assert_eq!([isz[2 * b] as usize, isz[2 * b + 1] as usize], [want.4, want.5], "{tag} ISUPPZ");
                        assert_eq!(iout[2 * b] as usize, want.3, "{tag} R");
                        // NEGCNT: the kernel carries 0 plus the WANTNC flag;
                        // the reference's -1 sentinel is the driver's job.
                        let want_nc = if wantnc { want.0 } else { 0 };
                        assert_eq!(iout[2 * b + 1] as i32, want_nc, "{tag} NEGCNT");
                    }
                }
            }
        }
    }
}
