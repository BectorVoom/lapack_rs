//! **OpenBLAS track.** `ZHETRD`'s device kernels in OpenBLAS 0.3.30 SkylakeX
//! order (host transcriptions: [`crate::openblas::zlevel2::zhemv`],
//! [`crate::openblas::zlevel3::zher2k_n`]), planar storage, on a fully
//! Hermitian device matrix (the mirrored triangle holds exact conjugates).
//!
//! K1c-ob (`zhemv_{L,U}_SKYLAKEX` = `zsymv_{L,U}_sse2.S` built with `HEMV`):
//! no fused operations. The assembly walks column pairs `(j, j+1)`; for each
//! pair it accumulates the pair's two column sums `xs1`, `xs2` over the rows
//! beyond the pair (in row pairs, column `j` rows ascending, column `j+1`
//! rows swapped) and adds four products into each of those rows (column `j`
//! first on even rows, column `j+1` first on odd rows). Every output row is
//! therefore a chain over the pairs in ascending order plus its own pair's
//! sum, independent of every other row: one unit per row `r`, reading column
//! `r` (for `A(r, q) = conj(A(q, r))`) and its own pair's two columns.

use cubecl::prelude::*;

/// `T1 = alpha·x(k)` (the assembly's `NEW_X` buffer, computed on the host with
/// the same unfused operations): real part.
#[cube]
fn t1r<F: Float + CubeElement>(xr: &Array<F>, xi: &Array<F>, k: u32, alr: F, ali: F) -> F {
    let _ = (xi, alr, ali);
    xr[k as usize]
}

/// Imaginary part of `T1`.
#[cube]
fn t1i<F: Float + CubeElement>(xr: &Array<F>, xi: &Array<F>, k: u32, alr: F, ali: F) -> F {
    let _ = (xr, alr, ali);
    xi[k as usize]
}

/// K1c-ob: `y(off + r) = row r of ZHEMV(UPLO, m, alpha, M[off.., off..], x, 0)`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_hemv_row_kernel<F: Float + CubeElement>(
    mr: &Array<F>,
    mi: &Array<F>,
    xr: &Array<F>,
    xi: &Array<F>,
    yr: &mut Array<F>,
    yi: &mut Array<F>,
    lda: u32,
    off: u32,
    m: u32,
    alr: F,
    ali: F,
    #[comptime] upper: bool,
) {
    if ABSOLUTE_POS < m as usize {
        let r = ABSOLUTE_POS as u32;
        let n = m;
        let zero = F::from_int(0);
        let neg = F::from_int(-1);
        let base_r = (off + r) * lda + off;
        let pairs_end = n - n % 2u32;
        let jown = r - r % 2u32;
        let mut y_re = zero;
        let mut y_im = zero;
        if !upper {
            // Earlier pairs: every pair j with j + 2 <= r.
            let np = select(r < pairs_end, jown / 2u32, pairs_end / 2u32);
            for p in 0..np {
                let j = 2u32 * p;
                let j1 = j + 1u32;
                let a1r = mr[(base_r + j) as usize];
                let a1i = mi[(base_r + j) as usize] * neg;
                let a2r = mr[(base_r + j1) as usize];
                let a2i = mi[(base_r + j1) as usize] * neg;
                let at0r = t1r::<F>(xr, xi, j, alr, ali);
                let at0i = t1i::<F>(xr, xi, j, alr, ali);
                let at1r = at0i * neg;
                let at1i = at0r;
                let at2r = t1r::<F>(xr, xi, j1, alr, ali);
                let at2i = t1i::<F>(xr, xi, j1, alr, ali);
                let at3r = at2i * neg;
                let at3i = at2r;
                let leftover = (n - (j + 2u32)) % 2u32 == 1u32 && r == n - 1u32;
                let b1role = !leftover && (r - (j + 2u32)) % 2u32 == 1u32;
                if b1role {
                    y_re = y_re + a2r * at2r;
                    y_im = y_im + a2r * at2i;
                    y_re = y_re + a2i * at3r;
                    y_im = y_im + a2i * at3i;
                    y_re = y_re + a1r * at0r;
                    y_im = y_im + a1r * at0i;
                    y_re = y_re + a1i * at1r;
                    y_im = y_im + a1i * at1i;
                } else {
                    y_re = y_re + a1r * at0r;
                    y_im = y_im + a1r * at0i;
                    y_re = y_re + a1i * at1r;
                    y_im = y_im + a1i * at1i;
                    y_re = y_re + a2r * at2r;
                    y_im = y_im + a2r * at2i;
                    y_re = y_re + a2i * at3r;
                    y_im = y_im + a2i * at3i;
                }
            }
            if r < pairs_end {
                let j = jown;
                let j1 = j + 1u32;
                let bj = (off + j) * lda + off;
                let bj1 = (off + j1) * lda + off;
                let at0r = t1r::<F>(xr, xi, j, alr, ali);
                let at0i = t1i::<F>(xr, xi, j, alr, ali);
                let at1r = at0i * neg;
                let at1i = at0r;
                let at2r = t1r::<F>(xr, xi, j1, alr, ali);
                let at2i = t1i::<F>(xr, xi, j1, alr, ali);
                let at3r = at2i * neg;
                let at3i = at2r;
                let ajj = mr[(bj + j) as usize];
                let a1r = mr[(bj + j1) as usize];
                let a1i = mi[(bj + j1) as usize];
                let aj1j1 = mr[(bj1 + j1) as usize];
                // .L11: the 2x2 block seeds the sums.
                let mut s1r = ajj * at0r;
                let mut s1i = ajj * at0i;
                let mut s2r = a1r * at0r;
                let mut s2i = a1r * at0i;
                s2r = s2r + a1i * at1r;
                s2i = s2i + a1i * at1i;
                s1r = s1r + a1r * at2r;
                s1i = s1i + a1r * at2i;
                s2r = s2r + aj1j1 * at2r;
                s2i = s2i + aj1j1 * at2i;
                s1r = s1r - a1i * at3r;
                s1i = s1i - a1i * at3i;
                let mut b = j + 2u32;
                while b + 2u32 <= n {
                    let b1 = b + 1u32;
                    let p1r0 = mr[(bj + b) as usize];
                    let p1i0 = mi[(bj + b) as usize];
                    let p1r1 = mr[(bj + b1) as usize];
                    let p1i1 = mi[(bj + b1) as usize];
                    let p2r0 = mr[(bj1 + b) as usize];
                    let p2i0 = mi[(bj1 + b) as usize];
                    let p2r1 = mr[(bj1 + b1) as usize];
                    let p2i1 = mi[(bj1 + b1) as usize];
                    let u0r = t1r::<F>(xr, xi, b, alr, ali);
                    let u0i = t1i::<F>(xr, xi, b, alr, ali);
                    let u1r = t1r::<F>(xr, xi, b1, alr, ali);
                    let u1i = t1i::<F>(xr, xi, b1, alr, ali);
                    // T2 = (-T1.im, T1.re).
                    s1r = s1r + p1r0 * u0r;
                    s1i = s1i + p1r0 * u0i;
                    s1r = s1r - p1i0 * (u0i * neg);
                    s1i = s1i - p1i0 * u0r;
                    s1r = s1r + p1r1 * u1r;
                    s1i = s1i + p1r1 * u1i;
                    s1r = s1r - p1i1 * (u1i * neg);
                    s1i = s1i - p1i1 * u1r;
                    s2r = s2r + p2r1 * u1r;
                    s2i = s2i + p2r1 * u1i;
                    s2r = s2r - p2i1 * (u1i * neg);
                    s2i = s2i - p2i1 * u1r;
                    s2r = s2r + p2r0 * u0r;
                    s2i = s2i + p2r0 * u0i;
                    s2r = s2r - p2i0 * (u0i * neg);
                    s2i = s2i - p2i0 * u0r;
                    b += 2u32;
                }
                if b < n {
                    let p1r = mr[(bj + b) as usize];
                    let p1i = mi[(bj + b) as usize];
                    let p2r = mr[(bj1 + b) as usize];
                    let p2i = mi[(bj1 + b) as usize];
                    let ur = t1r::<F>(xr, xi, b, alr, ali);
                    let ui = t1i::<F>(xr, xi, b, alr, ali);
                    s1r = s1r + p1r * ur;
                    s1i = s1i + p1r * ui;
                    s1r = s1r - p1i * (ui * neg);
                    s1i = s1i - p1i * ur;
                    s2r = s2r + p2r * ur;
                    s2i = s2i + p2r * ui;
                    s2r = s2r - p2i * (ui * neg);
                    s2i = s2i - p2i * ur;
                }
                if r == j {
                    y_re = y_re + s1r;
                    y_im = y_im + s1i;
                } else {
                    y_re = y_re + s2r;
                    y_im = y_im + s2i;
                }
            } else {
                // .L20: odd n, the last diagonal.
                let ajj = mr[(base_r + r) as usize];
                y_re = y_re + ajj * t1r::<F>(xr, xi, r, alr, ali);
                y_im = y_im + ajj * t1i::<F>(xr, xi, r, alr, ali);
            }
        } else {
            if r < pairs_end {
                let j = jown;
                let j1 = j + 1u32;
                let bj = (off + j) * lda + off;
                let bj1 = (off + j1) * lda + off;
                let at0r = t1r::<F>(xr, xi, j, alr, ali);
                let at0i = t1i::<F>(xr, xi, j, alr, ali);
                let at1r = at0i * neg;
                let at1i = at0r;
                let at2r = t1r::<F>(xr, xi, j1, alr, ali);
                let at2i = t1i::<F>(xr, xi, j1, alr, ali);
                let at3r = at2i * neg;
                let at3i = at2r;
                let mut s1r = zero;
                let mut s1i = zero;
                let mut s2r = zero;
                let mut s2i = zero;
                let mut b = 0u32;
                while b + 2u32 <= j {
                    let b1 = b + 1u32;
                    let p1r0 = mr[(bj + b) as usize];
                    let p1i0 = mi[(bj + b) as usize];
                    let p1r1 = mr[(bj + b1) as usize];
                    let p1i1 = mi[(bj + b1) as usize];
                    let p2r0 = mr[(bj1 + b) as usize];
                    let p2i0 = mi[(bj1 + b) as usize];
                    let p2r1 = mr[(bj1 + b1) as usize];
                    let p2i1 = mi[(bj1 + b1) as usize];
                    let u0r = t1r::<F>(xr, xi, b, alr, ali);
                    let u0i = t1i::<F>(xr, xi, b, alr, ali);
                    let u1r = t1r::<F>(xr, xi, b1, alr, ali);
                    let u1i = t1i::<F>(xr, xi, b1, alr, ali);
                    s1r = s1r + p1r0 * u0r;
                    s1i = s1i + p1r0 * u0i;
                    s1r = s1r - p1i0 * (u0i * neg);
                    s1i = s1i - p1i0 * u0r;
                    s1r = s1r + p1r1 * u1r;
                    s1i = s1i + p1r1 * u1i;
                    s1r = s1r - p1i1 * (u1i * neg);
                    s1i = s1i - p1i1 * u1r;
                    s2r = s2r + p2r1 * u1r;
                    s2i = s2i + p2r1 * u1i;
                    s2r = s2r - p2i1 * (u1i * neg);
                    s2i = s2i - p2i1 * u1r;
                    s2r = s2r + p2r0 * u0r;
                    s2i = s2i + p2r0 * u0i;
                    s2r = s2r - p2i0 * (u0i * neg);
                    s2i = s2i - p2i0 * u0r;
                    b += 2u32;
                }
                // .L18: the 2x2 diagonal block.
                let ajj = mr[(bj + j) as usize];
                let a2r = mr[(bj1 + j) as usize];
                let a2i = mi[(bj1 + j) as usize];
                let aj1j1 = mr[(bj1 + j1) as usize];
                s1r = s1r + ajj * at0r;
                s1i = s1i + ajj * at0i;
                s2r = s2r + a2r * at0r;
                s2i = s2i + a2r * at0i;
                s2r = s2r - a2i * at1r;
                s2i = s2i - a2i * at1i;
                s1r = s1r + a2r * at2r;
                s1i = s1i + a2r * at2i;
                s2r = s2r + aj1j1 * at2r;
                s2i = s2i + aj1j1 * at2i;
                s1r = s1r + a2i * at3r;
                s1i = s1i + a2i * at3i;
                if r == j {
                    y_re = y_re + s1r;
                    y_im = y_im + s1i;
                } else {
                    y_re = y_re + s2r;
                    y_im = y_im + s2i;
                }
                // Later pairs: rows above them.
                let p0 = jown / 2u32 + 1u32;
                for p in p0..pairs_end / 2u32 {
                    let jj = 2u32 * p;
                    let jj1 = jj + 1u32;
                    let a1r = mr[(base_r + jj) as usize];
                    let a1i = mi[(base_r + jj) as usize] * neg;
                    let b2r = mr[(base_r + jj1) as usize];
                    let b2i = mi[(base_r + jj1) as usize] * neg;
                    let c0r = t1r::<F>(xr, xi, jj, alr, ali);
                    let c0i = t1i::<F>(xr, xi, jj, alr, ali);
                    let c1r = c0i * neg;
                    let c1i = c0r;
                    let c2r = t1r::<F>(xr, xi, jj1, alr, ali);
                    let c2i = t1i::<F>(xr, xi, jj1, alr, ali);
                    let c3r = c2i * neg;
                    let c3i = c2r;
                    if r % 2u32 == 1u32 {
                        y_re = y_re + b2r * c2r;
                        y_im = y_im + b2r * c2i;
                        y_re = y_re + b2i * c3r;
                        y_im = y_im + b2i * c3i;
                        y_re = y_re + a1r * c0r;
                        y_im = y_im + a1r * c0i;
                        y_re = y_re + a1i * c1r;
                        y_im = y_im + a1i * c1i;
                    } else {
                        y_re = y_re + a1r * c0r;
                        y_im = y_im + a1r * c0i;
                        y_re = y_re + a1i * c1r;
                        y_im = y_im + a1i * c1i;
                        y_re = y_re + b2r * c2r;
                        y_im = y_im + b2r * c2i;
                        y_re = y_re + b2i * c3r;
                        y_im = y_im + b2i * c3i;
                    }
                }
            }
            if n % 2u32 == 1u32 {
                // .L20/.L22/.L28: the last column alone.
                let jn = n - 1u32;
                let tnr = t1r::<F>(xr, xi, jn, alr, ali);
                let tni = t1i::<F>(xr, xi, jn, alr, ali);
                if r < jn {
                    let ar = mr[(base_r + jn) as usize];
                    let ai = mi[(base_r + jn) as usize] * neg;
                    // at1 = T1(jn), at2 = T2(jn).
                    y_re = y_re + ar * tnr;
                    y_im = y_im + ar * tni;
                    y_re = y_re + ai * (tni * neg);
                    y_im = y_im + ai * tnr;
                } else {
                    let bn = (off + jn) * lda + off;
                    let mut s1r = zero;
                    let mut s1i = zero;
                    let mut s2r = zero;
                    let mut s2i = zero;
                    for q in 0..jn {
                        let ar = mr[(bn + q) as usize];
                        let ai = mi[(bn + q) as usize];
                        let ur = t1r::<F>(xr, xi, q, alr, ali);
                        let ui = t1i::<F>(xr, xi, q, alr, ali);
                        s1r = s1r + ar * ur;
                        s1i = s1i + ar * ui;
                        s2r = s2r - ai * (ui * neg);
                        s2i = s2i - ai * ur;
                    }
                    let ajj = mr[(bn + jn) as usize];
                    s1r = s1r + ajj * tnr;
                    s1i = s1i + ajj * tni;
                    s1r = s1r + s2r;
                    s1i = s1i + s2i;
                    y_re = y_re + s1r;
                    y_im = y_im + s1i;
                }
            }
        }
        yr[(off + r) as usize] = y_re;
        yi[(off + r) as usize] = y_im;
    }
}

/// `ZGEMM_P`, `ZGEMM_UNROLL_MN` of the pinned library.
const ZGEMM_P: u32 = 256;
const ZUNROLL_MN: u32 = 4;

/// Row-block size of `zher2k_{U,L}N` for `rem` rows left.
#[cube]
fn zp_rule(rem: u32) -> u32 {
    let mut r = rem;
    if rem >= ZGEMM_P * 2u32 {
        r = ZGEMM_P;
    } else if rem > ZGEMM_P {
        r = ((rem / 2u32) + ZUNROLL_MN - 1u32) / ZUNROLL_MN * ZUNROLL_MN;
    }
    r
}

/// A complex value held by a `#[cube]` function (real and imaginary parts).
#[derive(CubeType, Clone, Copy)]
pub struct ZPair<F: Float> {
    pub re: F,
    pub im: F,
}

/// [`zker_part`] computing both parts in one pass over `l`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn zker<F: Float + CubeElement>(
    vr: &Array<F>,
    vi: &Array<F>,
    lda: u32,
    g0: u32,
    k: u32,
    xl: u32,
    p: u32,
    yl: u32,
    q: u32,
    m4: bool,
    ar: F,
    ai: F,
    cr: F,
    ci: F,
) -> ZPair<F> {
    let zero = F::from_int(0);
    let one = F::from_int(1);
    let neg = F::from_int(-1);
    let mut out = ZPair::<F> { re: zero, im: zero };
    if m4 {
        let mut le0 = zero;
        let mut le1 = zero;
        let mut ri0 = zero;
        let mut ri1 = zero;
        for l in 0..k {
            let xp = ((xl + l) * lda + g0 + p) as usize;
            let yp = ((yl + l) * lda + g0 + q) as usize;
            let r = vr[xp];
            let i = vi[xp];
            let br = vr[yp];
            let bi = vi[yp];
            le0 = fma(br, r, le0);
            le1 = fma(bi, r, le1);
            ri0 = fma(br, i, ri0);
            ri1 = fma(bi, i, ri1);
        }
        let s_re = fma(ri1 * neg, neg, le0);
        let s_im = fma(ri0 * neg, one, le1);
        let x_re = fma(ar, s_re, cr);
        let x_im = fma(ar, s_im, ci * neg);
        out.re = fma(ai, s_im, x_re);
        out.im = fma(ai, s_re, x_im * neg);
    } else {
        let mut s0 = zero;
        let mut s1 = zero;
        for l in 0..k {
            let xp = ((xl + l) * lda + g0 + p) as usize;
            let yp = ((yl + l) * lda + g0 + q) as usize;
            let r = vr[xp];
            let i = vi[xp];
            let br = vr[yp];
            let bi = vi[yp];
            let t_re = fma(bi, i, s0);
            let t_im = fma(bi, r, s1 * neg);
            s0 = fma(br, r, t_re);
            s1 = fma(br, i, t_im * neg);
        }
        let y_re = fma(ai, s1, cr * neg);
        let y_im = fma(ai, s0, ci);
        out.re = fma(ar, s0, y_re * neg);
        out.im = fma(ar, s1, y_im);
    }
    out
}

/// K2c-ob, scalar units over a list of block elements `(i, j) = (elems[2u],
/// elems[2u+1])`: element `(i, j)` takes the new value of its triangle element
/// `(p, q)` (conjugated in the mirrored triangle; the diagonal's imaginary
/// part is zero). Used for the special elements (diagonal `4 × 4` pieces and
/// the `m % 4` tail rows of a kernel call); each unit writes only its element.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_her2k_list_kernel<F: Float + CubeElement>(
    mr: &mut Array<F>,
    mi: &mut Array<F>,
    vr: &Array<F>,
    vi: &Array<F>,
    elems: &Array<u32>,
    lda: u32,
    g0: u32,
    m: u32,
    nb: u32,
    count: u32,
    alr: F,
    ali: F,
    #[comptime] upper: bool,
) {
    if ABSOLUTE_POS < count as usize {
        let u = ABSOLUTE_POS as u32;
        let ei = elems[(2u32 * u) as usize];
        let ej = elems[(2u32 * u + 1u32) as usize];
        let own_lower = ei >= ej;
        let own_upper = ei <= ej;
        let mut own = own_lower;
        if upper {
            own = own_upper;
        }
        let i = select(own, ei, ej);
        let j = select(own, ej, ei);
        let neg = F::from_int(-1);
        let mut s = 0u32;
        let mut e = zp_rule(m);
        while e <= i {
            s = e;
            e = s + zp_rule(m - s);
        }
        let mb = e - s;
        let pidx = ((ej + g0) * lda + g0 + ei) as usize;
        let c0r = mr[pidx];
        let c0i = select(own, mi[pidx], mi[pidx] * neg);
        let mut diag = false;
        let mut rr = 0u32;
        let mut cm = 0u32;
        let mut dbase = 0u32;
        let mut dnn = 0u32;
        if upper {
            if j >= e {
                if s == 0u32 {
                    rr = i;
                    cm = e;
                } else {
                    rr = i - s;
                    cm = mb;
                }
            } else {
                let lp = ((j - s) / ZUNROLL_MN) * ZUNROLL_MN;
                let nn = select(mb - lp < ZUNROLL_MN, mb - lp, ZUNROLL_MN);
                if i - s >= lp {
                    diag = true;
                    dbase = s + lp;
                    dnn = nn;
                } else {
                    rr = i - s;
                    cm = lp;
                }
            }
        } else if j < s {
            rr = i - s;
            cm = mb;
        } else {
            let lp = ((j - s) / ZUNROLL_MN) * ZUNROLL_MN;
            let nn = select(mb - lp < ZUNROLL_MN, mb - lp, ZUNROLL_MN);
            if i - s < lp + nn {
                diag = true;
                dbase = s + lp;
                dnn = nn;
            } else {
                rr = i - (s + lp + nn);
                cm = mb - lp - nn;
            }
        }
        let zero = F::from_int(0);
        if diag {
            let ri = i - dbase;
            let rj = j - dbase;
            let m4i = ri < dnn - dnn % 4u32;
            let m4j = rj < dnn - dnn % 4u32;
            let sij = zker::<F>(vr, vi, lda, g0, nb, 0u32, i, nb, j, m4i, alr, ali, zero, zero);
            let sji = zker::<F>(vr, vi, lda, g0, nb, 0u32, j, nb, i, m4j, alr, ali, zero, zero);
            mr[pidx] = (sij.re + sji.re) + c0r;
            if i == j {
                mi[pidx] = zero;
            } else {
                let di = (sij.im - sji.im) + c0i;
                mi[pidx] = select(own, di, di * neg);
            }
        } else {
            let m4 = rr < cm - cm % 4u32;
            let p1 = zker::<F>(vr, vi, lda, g0, nb, 0u32, i, nb, j, m4, alr, ali, c0r, c0i);
            let p2 = zker::<F>(vr, vi, lda, g0, nb, nb, i, 0u32, j, m4, alr, ali * neg, p1.re, p1.im);
            mr[pidx] = p2.re;
            mi[pidx] = select(own, p2.im, p2.im * neg);
        }
    }
}

/// K2c-ob, lane-vector units `(row lane-vector, block column j)` for the plain
/// elements (4-row-group rows of a kernel call, outside the diagonal pieces):
/// both passes (`V`, conj `W`, `alpha`) then (`W`, conj `V`, conj `alpha`) with
/// the four fused chains per lane and `SAVE`. Own lanes (`i ≥ j` lower,
/// `i ≤ j` upper) read row `i` from the lanes and row `j` as broadcasts, mirror
/// lanes the other way round, and store the conjugate. Units touching a
/// special interval of column `j` (`ioff`/`ivals`, block-relative) and rows
/// outside the block keep their value.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ob_her2k_vec_kernel<F: Float + CubeElement, N: Size>(
    mr: &mut Array<Vector<F, N>>,
    mi: &mut Array<Vector<F, N>>,
    vr: &Array<Vector<F, N>>,
    vi: &Array<Vector<F, N>>,
    vr_s: &Array<F>,
    vi_s: &Array<F>,
    ioff: &Array<u32>,
    ivals: &Array<u32>,
    lane_ofs: &Array<Vector<u32, N>>,
    lda: u32,
    g0: u32,
    m: u32,
    rv0: u32,
    units_x: u32,
    units: u32,
    nb: u32,
    alr: F,
    ali: F,
    #[comptime] upper: bool,
) {
    if ABSOLUTE_POS < units as usize {
        let u = ABSOLUTE_POS as u32;
        let rv = rv0 + u % units_x;
        let jb = u / units_x;
        let j = g0 + jb;
        let nl = N::value() as u32;
        let lda_vecs = lda / nl;
        let first = rv * nl;
        let r0 = select(first > g0, first - g0, 0u32);
        let lastp = first + nl;
        let hi = select(lastp < g0 + m, lastp, g0 + m);
        let r1 = select(hi > g0, hi - g0, 0u32);
        let mut special = false;
        for t in ioff[jb as usize]..ioff[(jb + 1u32) as usize] {
            if ivals[(2u32 * t) as usize] < r1 && r0 < ivals[(2u32 * t + 1u32) as usize] {
                special = true;
            }
        }
        if !special && r0 < r1 {
            let rows = Vector::<u32, N>::new(first) + lane_ofs[0];
            let jv = Vector::<u32, N>::new(j);
            let mut own = rows.greater_equal(jv);
            if upper {
                own = rows.less_equal(jv);
            }
            let idx = (j * lda_vecs + rv) as usize;
            let old_r = mr[idx];
            let old_i = mi[idx];
            let zero = Vector::<F, N>::new(F::from_int(0));
            let negv = Vector::<F, N>::new(F::from_int(-1));
            let onev = Vector::<F, N>::new(F::from_int(1));
            let var = Vector::<F, N>::new(alr);
            let vai = Vector::<F, N>::new(ali);
            let vai_c = Vector::<F, N>::new(ali * F::from_int(-1));
            let c_r = old_r;
            let c_i = select_many(own, old_i, old_i * negv);
            // Pass 1: X = V, Y = W.
            let mut le0 = zero;
            let mut le1 = zero;
            let mut ri0 = zero;
            let mut ri1 = zero;
            for l in 0..nb {
                let xr_i = vr[(l * lda_vecs + rv) as usize];
                let xi_i = vi[(l * lda_vecs + rv) as usize];
                let yr_i = vr[((nb + l) * lda_vecs + rv) as usize];
                let yi_i = vi[((nb + l) * lda_vecs + rv) as usize];
                let xr_j = Vector::<F, N>::new(vr_s[(l * lda + j) as usize]);
                let xi_j = Vector::<F, N>::new(vi_s[(l * lda + j) as usize]);
                let yr_j = Vector::<F, N>::new(vr_s[((nb + l) * lda + j) as usize]);
                let yi_j = Vector::<F, N>::new(vi_s[((nb + l) * lda + j) as usize]);
                let r = select_many(own, xr_i, xr_j);
                let i = select_many(own, xi_i, xi_j);
                let br = select_many(own, yr_j, yr_i);
                let bi = select_many(own, yi_j, yi_i);
                le0 = fma(br, r, le0);
                le1 = fma(bi, r, le1);
                ri0 = fma(br, i, ri0);
                ri1 = fma(bi, i, ri1);
            }
            let s_re = fma(ri1 * negv, negv, le0);
            let s_im = fma(ri0 * negv, onev, le1);
            let x_re = fma(var, s_re, c_r);
            let x_im = fma(var, s_im, c_i * negv);
            let p_r = fma(vai, s_im, x_re);
            let p_i = fma(vai, s_re, x_im * negv);
            // Pass 2: X = W, Y = V, conj(alpha).
            let mut me0 = zero;
            let mut me1 = zero;
            let mut mi0 = zero;
            let mut mi1 = zero;
            for l in 0..nb {
                let xr_i = vr[((nb + l) * lda_vecs + rv) as usize];
                let xi_i = vi[((nb + l) * lda_vecs + rv) as usize];
                let yr_i = vr[(l * lda_vecs + rv) as usize];
                let yi_i = vi[(l * lda_vecs + rv) as usize];
                let xr_j = Vector::<F, N>::new(vr_s[((nb + l) * lda + j) as usize]);
                let xi_j = Vector::<F, N>::new(vi_s[((nb + l) * lda + j) as usize]);
                let yr_j = Vector::<F, N>::new(vr_s[(l * lda + j) as usize]);
                let yi_j = Vector::<F, N>::new(vi_s[(l * lda + j) as usize]);
                let r = select_many(own, xr_i, xr_j);
                let i = select_many(own, xi_i, xi_j);
                let br = select_many(own, yr_j, yr_i);
                let bi = select_many(own, yi_j, yi_i);
                me0 = fma(br, r, me0);
                me1 = fma(bi, r, me1);
                mi0 = fma(br, i, mi0);
                mi1 = fma(bi, i, mi1);
            }
            let t_re = fma(mi1 * negv, negv, me0);
            let t_im = fma(mi0 * negv, onev, me1);
            let y_re = fma(var, t_re, p_r);
            let y_im = fma(var, t_im, p_i * negv);
            let n_r = fma(vai_c, t_im, y_re);
            let n_i = fma(vai_c, t_re, y_im * negv);
            let in_block =
                rows.greater_equal(Vector::<u32, N>::new(g0)).and(rows.less_than(Vector::<u32, N>::new(g0 + m)));
            mr[idx] = select_many(in_block, n_r, old_r);
            mi[idx] = select_many(in_block, select_many(own, n_i, n_i * negv), old_i);
        }
    }
}
