//! Coalescing gate (manual 16 §2.5, no profiler): every hot load and store
//! index expression of the device kernels, expanded over one 32-lane plane of
//! f64, touches 32.0 useful bytes per 32-byte sector. The closures transcribe
//! the kernels' index expressions (`src/device/kernels/*.rs`) at
//! representative parameters; "lane" is one row of a lane-vector (vector
//! kernels) or one unit (scalar kernels). Broadcast loads (one element per
//! unit: `x[p]`, `vw_s`, `perm`, `s`, `c` in `gemm_t`, `w_in_s` in the trmm
//! chain) are listed separately and not gated; two small offset copies may
//! lose one sector per plane (documented below).

type Access = (&'static str, Box<dyn Fn(u32) -> u64>);

const SECTOR_BYTES: u64 = 32;
const PLANE: u32 = 32;
const ELEM: u64 = 8;

fn bytes_per_sector(addr_of_lane: impl Fn(u32) -> u64) -> f64 {
    let mut sectors = std::collections::BTreeSet::new();
    for lane in 0..PLANE {
        let base = addr_of_lane(lane) * ELEM;
        for b in 0..ELEM {
            sectors.insert((base + b) / SECTOR_BYTES);
        }
    }
    (PLANE as u64 * ELEM) as f64 / sectors.len() as f64
}

#[test]
fn hot_accesses_are_fully_coalesced() {
    // Representative geometry: n = 1000, lda = 1024, a lane-vector of 8 rows.
    let (lda, n, kpad, nb) = (1024u64, 1000u64, 32u64, 32u64);
    let row0 = 96u64; // first row of the plane (a multiple of the width)
    let (p, j, l, i, jj, r_i) = (517u64, 400u64, 5u64, 123u64, 17u64, 9u64);
    let rvecs = 64u64; // merge row chunk (rows)
    let accesses: Vec<Access> = vec![
        ("K1 symv load M[row, p]", Box::new(move |x| p * lda + row0 + x as u64)),
        ("K1 symv store y[row]", Box::new(move |x| row0 + x as u64)),
        ("K2 syr2k load/store C[row, j]", Box::new(move |x| j * lda + row0 + x as u64)),
        ("K2 syr2k load V[row, l]", Box::new(move |x| l * lda + row0 + x as u64)),
        ("K2 syr2k load W[row, l]", Box::new(move |x| (nb + l) * lda + row0 + x as u64)),
        ("K3m merge load Z[row, perm[l]]", Box::new(move |x| p * lda + row0 + x as u64)),
        ("K3m merge store S[row - chunk0, j]", Box::new(move |x| j * rvecs + x as u64)),
        ("K7 copy_cols store Z[row, c0 + j]", Box::new(move |x| (j + 3) * lda + row0 + x as u64)),
        ("K7 copy_cols load S[row, j]", Box::new(move |x| j * rvecs + x as u64)),
        ("K7 rotations load/store Z[row, col]", Box::new(move |x| p * lda + row0 + x as u64)),
        ("K7 swap_cols Z[row, ca]", Box::new(move |x| p * lda + row0 + x as u64)),
        // Row-tiled K2r: row loads once per l; the t-th column accumulator
        // stores C[row, jb + t] across lanes.
        ("K2r syr2k_rt store C[row, jb + t]", Box::new(move |x| (j + 5) * lda + row0 + x as u64)),
        ("K2r syr2k_rt load V[row, l]", Box::new(move |x| l * lda + row0 + x as u64)),
        // Row-tiled K3r: V_rm row loads once per l, W(ib + t, j) stores.
        ("K3r gemm_t_rt load V_rm(l, j)", Box::new(move |x| (l + 40) * kpad + x as u64)),
        ("K3r gemm_t_rt store W(ib + t, j)", Box::new(move |x| (i + 3) * kpad + x as u64)),
        ("K3 trmm load/store W(i, j)", Box::new(move |x| i * kpad + x as u64)),
        ("K3 trmm load coef(j, kk)", Box::new(move |x| jj * kpad + x as u64)),
        ("K3 trmm load diag(j)", Box::new(move |x| x as u64)),
        ("K3 gemm_t load V_rm(l, j)", Box::new(move |x| (l + 40) * kpad + x as u64)),
        ("K3 gemm_t load/store W(i, j)", Box::new(move |x| i * kpad + x as u64)),
        // gemm_n is scalar with sector-aligned units: row c_row0 − pad + r.
        ("K3 gemm_n load/store C[c_row0 − pad + r, i]", Box::new(move |x| i * lda + 32 + x as u64)),
        ("K3 gemm_n load V[r + l·ldv] (ldv = 64)", Box::new(move |x| x as u64 + l * 64)),
        ("K7 rows_to_w / sub_w store W[i·kpad + j]", Box::new(move |x| i * kpad + x as u64)),
        ("K7 fill_identity store Z[r, c]", Box::new(move |x| p * lda + x as u64)),
    ];
    let _ = (n, r_i);
    let mut failures = Vec::new();
    for (name, f) in &accesses {
        let bps = bytes_per_sector(f);
        println!("{name:<45} {bps:>6.2} bytes/sector");
        if bps != 32.0 {
            failures.push(format!("{name}: {bps}"));
        }
    }
    // Documented partial exceptions (at most one extra sector per plane, the
    // reference row offset is not a multiple of the width): the O(n·NB)
    // `DLARFB` copies rows_to_w/sub_w (C side) and the ≤ 25² leaf scatter.
    for (name, f) in [
        (
            "K7 rows_to_w / sub_w load C[c_row0 + j, i]",
            Box::new(move |x: u32| i * lda + 33 + x as u64) as Box<dyn Fn(u32) -> u64>,
        ),
        ("K7 scatter_block Z[g0 + i, g0 + j]", Box::new(move |x: u32| (p + 1) * lda + p + x as u64)),
    ] {
        let worst = (0..8u64).map(|o| bytes_per_sector(|x| f(x) + o)).fold(f64::MAX, f64::min);
        println!("{name:<45} {worst:>6.2} bytes/sector worst offset (documented)");
        assert!(worst >= 32.0 * 8.0 / 9.0 - 1e-9, "{name}: {worst}");
    }
    // Not gated, and a DELIBERATE exception: K8 (the batched MRRR leaves,
    // `src/device/kernels/stemr.rs`) is problem-major — unit `b` owns a whole
    // problem at `nptr[b]` and walks it with consecutive `j`, so lane `x`
    // starts a whole problem away from lane `x+1`. That is the opposite of
    // stride-1 across lanes and is the right layout here for the same reason
    // the CUDA notes record for K1/K1c: the inner loops are `O(n)` long
    // (`n <= 13` for the cintx corpus, a few hundred for a LAPACK block),
    // each unit reuses its own block across every iteration, and a
    // lane-interleaved layout would cost a gather per iteration instead of a
    // sequential walk. Recorded here so the model does not silently drift.
    let k8 = bytes_per_sector(move |x| (x as u64) * 13 + 3);
    println!("{:<45} {k8:>6.2} bytes/sector (problem-major, deliberate)", "K8 stemr load d[nptr[b] + j]");
    // Not gated: the only other strided expression, the O(n) host-bound row
    // gather (`gather_row`, one launch per merge read of `Q(n1, ·)`/`Q(n1+1, ·)`).
    let strided = bytes_per_sector(move |x| (x as u64) * lda + row0);
    println!("{:<45} {strided:>6.2} bytes/sector (O(n), documented)", "K7 gather_row load Z[row, col0 + q]");
    assert!(failures.is_empty(), "strided hot accesses: {failures:?}");
}
