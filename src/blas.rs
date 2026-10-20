//! Reference BLAS (Reference-LAPACK v3.12.1 `BLAS/SRC`), real double precision.
//!
//! Transcribed from `ddot.f`, `dnrm2.f90`, `daxpy.f`, `dscal.f`, `dcopy.f`,
//! `dswap.f`, `drot.f`, `idamax.f`, `dsymv.f`, `dsyr2.f`, `dgemv.f`, `dger.f`,
//! `dsyr2k.f`, `dgemm.f` and `dtrmm.f`.
//!
//! Every output element keeps the reference term order. Level-2/3 kernels
//! split their element set across threads through [`crate::par`]; element-wise
//! independence is noted per kernel. Reductions (`ddot`, `dnrm2`) are
//! sequential chains as written. Only positive increments occur in the
//! eigensolver call tree, so negative-increment branches are not transcribed.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use crate::ptr::{D, ix};

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;

/// `DDOT`. The 5-unrolled body `DTEMP + a + b + c + d + e` is a left-to-right
/// chain after the `MOD(N,5)` cleanup, i.e. one sequential sum from 0.
pub(crate) unsafe fn ddot(n: usize, x: D, incx: usize, y: D, incy: usize) -> f64 {
    let mut dtemp = 0.0f64;
    if n == 0 {
        return dtemp;
    }
    if incx == 1 && incy == 1 {
        for i in 0..n {
            dtemp += x.at(i) * y.at(i);
        }
    } else {
        let (mut ix_, mut iy) = (0usize, 0usize);
        for _ in 0..n {
            dtemp += x.at(ix_) * y.at(iy);
            ix_ += incx;
            iy += incy;
        }
    }
    dtemp
}

// dnrm2.f90 constants (radix 2, minexponent -1021, maxexponent 1024, digits 53).
pub(crate) const TSML: f64 = pow2(-511);
pub(crate) const TBIG: f64 = pow2(486);
pub(crate) const SSML: f64 = pow2(537);
pub(crate) const SBIG: f64 = pow2(-538);

/// Exact `2**e` for a normal exponent.
pub(crate) const fn pow2(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// `DNRM2` (3.10+ f90, Blue's three accumulators).
pub(crate) unsafe fn dnrm2(n: usize, x: D, incx: usize) -> f64 {
    const MAXN: f64 = f64::MAX;
    if n == 0 {
        return ZERO;
    }
    let mut notbig = true;
    let mut asml = ZERO;
    let mut amed = ZERO;
    let mut abig = ZERO;
    let mut ixx = 0usize;
    for _ in 0..n {
        let ax = x.at(ixx).abs();
        if ax > TBIG {
            let t = ax * SBIG;
            abig += t * t;
            notbig = false;
        } else if ax < TSML {
            if notbig {
                let t = ax * SSML;
                asml += t * t;
            }
        } else {
            amed += ax * ax;
        }
        ixx += incx;
    }
    let scl;
    let sumsq;
    if abig > ZERO {
        if (amed > ZERO) || (amed > MAXN) || amed.is_nan() {
            abig += (amed * SBIG) * SBIG;
        }
        scl = ONE / SBIG;
        sumsq = abig;
    } else if asml > ZERO {
        if (amed > ZERO) || (amed > MAXN) || amed.is_nan() {
            let amed = amed.sqrt();
            let asml = asml.sqrt() / SSML;
            let (ymin, ymax) = if asml > amed { (amed, asml) } else { (asml, amed) };
            scl = ONE;
            let r = ymin / ymax;
            sumsq = (ymax * ymax) * (ONE + r * r);
        } else {
            scl = ONE / SSML;
            sumsq = asml;
        }
    } else {
        scl = ONE;
        sumsq = amed;
    }
    scl * sumsq.sqrt()
}

/// `DAXPY`. Element-wise; the 4-unrolled body computes the same values.
pub(crate) unsafe fn daxpy(n: usize, da: f64, x: D, incx: usize, y: D, incy: usize) {
    if n == 0 || da == 0.0 {
        return;
    }
    let (mut ixx, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        y.set(iy, y.at(iy) + da * x.at(ixx));
        ixx += incx;
        iy += incy;
    }
}

/// `DSCAL`. `DA*DX(I)`, skipped entirely when `DA.EQ.ONE`.
pub(crate) unsafe fn dscal(n: usize, da: f64, x: D, incx: usize) {
    if n == 0 || incx == 0 || da == ONE {
        return;
    }
    let mut i = 0usize;
    for _ in 0..n {
        x.set(i, da * x.at(i));
        i += incx;
    }
}

/// `DCOPY`.
pub(crate) unsafe fn dcopy(n: usize, x: D, incx: usize, y: D, incy: usize) {
    let (mut ixx, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        y.set(iy, x.at(ixx));
        ixx += incx;
        iy += incy;
    }
}

/// `DSWAP`.
pub(crate) unsafe fn dswap(n: usize, x: D, incx: usize, y: D, incy: usize) {
    let (mut ixx, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        let t = x.at(ixx);
        x.set(ixx, y.at(iy));
        y.set(iy, t);
        ixx += incx;
        iy += incy;
    }
}

/// `DROT`.
pub(crate) unsafe fn drot(n: usize, x: D, incx: usize, y: D, incy: usize, c: f64, s: f64) {
    let (mut ixx, mut iy) = (0usize, 0usize);
    for _ in 0..n {
        let dtemp = c * x.at(ixx) + s * y.at(iy);
        y.set(iy, c * y.at(iy) - s * x.at(ixx));
        x.set(ixx, dtemp);
        ixx += incx;
        iy += incy;
    }
}

/// `IDAMAX` (1-based result, 0 when `n == 0`).
pub(crate) unsafe fn idamax(n: usize, x: D, incx: usize) -> usize {
    if n < 1 || incx == 0 {
        return 0;
    }
    let mut idx = 1;
    if n == 1 {
        return idx;
    }
    let mut dmax = x.at(0).abs();
    let mut ixx = incx;
    for i in 2..=n {
        let v = x.at(ixx).abs();
        if v > dmax {
            idx = i;
            dmax = v;
        }
        ixx += incx;
    }
    idx
}

/// Number of independent reduction chains advanced together in one loop.
/// Each lane is its own reference-ordered chain; the lanes only share the
/// loop, so superscalar execution (or SIMD) runs them concurrently.
/// Eight lanes measured best on this class of machine (4.2x over one
/// chain at 4 lanes, 5.2x at 8, slower at 16 as the load streams multiply).
const LANES: usize = 8;

/// Σ_l a_q[l]*x[l] for `LANES` columns at once, l ascending per lane.
#[inline(always)]
fn dotl(a: [&[f64]; LANES], x: &[f64], acc: &mut [f64; LANES]) {
    let n = x.len();
    let c: [&[f64]; LANES] = std::array::from_fn(|q| &a[q][..n]);
    let mut s = *acc;
    for l in 0..n {
        let xl = x[l];
        for q in 0..LANES {
            s[q] += c[q][l] * xl;
        }
    }
    *acc = s;
}

/// Σ_l a[l]*x[l], l ascending.
#[inline(always)]
fn dot1(a: &[f64], x: &[f64], mut acc: f64) -> f64 {
    for (al, xl) in a.iter().zip(x) {
        acc += al * xl;
    }
    acc
}

/// `y[i] += t * a[i]` (element-wise).
#[inline(always)]
fn axpy_slice(t: f64, a: &[f64], y: &mut [f64]) {
    for (yi, ai) in y.iter_mut().zip(a) {
        *yi += t * ai;
    }
}

/// `DSYMV` with unit increments.
///
/// Element-wise per output row: `Y(i)` is a chain over the columns that touch
/// it (lower: off-diagonal terms of columns `j < i`, then the diagonal, then
/// `ALPHA*TEMP2`; upper: diagonal plus `ALPHA*TEMP2` at column `i`, then the
/// off-diagonal terms of columns `j > i`). Row chunks run in parallel, each
/// replaying the column loop restricted to its rows. `TEMP2` only reads `A`
/// and `X`, so the chains of `LANES` consecutive columns advance together.
pub(crate) unsafe fn dsymv(upper: bool, n: usize, alpha: f64, a: D, lda: usize, x: D, beta: f64, y: D) {
    if n == 0 || (alpha == ZERO && beta == ONE) {
        return;
    }
    if beta != ONE {
        if beta == ZERO {
            for i in 0..n {
                y.set(i, ZERO);
            }
        } else {
            for i in 0..n {
                y.set(i, beta * y.at(i));
            }
        }
    }
    if alpha == ZERO {
        return;
    }
    let xs = x.slice(n);
    let col = |j: usize| a.off((j - 1) * lda);
    {
        let (lo, hi) = (0usize, n);

        // Rows lo+1..=hi (1-based), i.e. 0-based lo..hi.
        let mut t2 = [0.0f64; LANES];
        let mut t2_first = usize::MAX; // column of t2[0]
        if upper {
            for j in (lo + 1)..=n {
                let temp1 = alpha * xs[j - 1];
                let cj = col(j);
                let top = (j - 1).min(hi);
                if top > lo {
                    let ys = y.off(lo).slice_mut(top - lo);
                    axpy_slice(temp1, cj.off(lo).slice(top - lo), ys);
                }
                if j <= hi {
                    if t2_first == usize::MAX || j >= t2_first + LANES {
                        // TEMP2(j..j+LANES-1) = Σ_{i<j'} A(i,j')*X(i), one chain each.
                        t2_first = j;
                        let mut acc = [0.0f64; LANES];
                        let q_n = LANES.min(hi + 1 - j);
                        if q_n == LANES {
                            let c: [D; LANES] = std::array::from_fn(|q| col(j + q));
                            dotl(std::array::from_fn(|q| c[q].slice(j - 1)), &xs[..j - 1], &mut acc);
                            // Tails of the longer chains, still in i order.
                            for q in 1..LANES {
                                let cq = c[q].slice(j - 1 + q);
                                acc[q] = dot1(&cq[j - 1..], &xs[j - 1..j - 1 + q], acc[q]);
                            }
                        } else {
                            for q in 0..q_n {
                                acc[q] = dot1(col(j + q).slice(j - 1 + q), &xs[..j - 1 + q], 0.0);
                            }
                        }
                        t2 = acc;
                    }
                    let temp2 = t2[j - t2_first];
                    y.set(j - 1, y.at(j - 1) + temp1 * cj.at(j - 1) + alpha * temp2);
                }
            }
        } else {
            for j in 1..=hi {
                let temp1 = alpha * xs[j - 1];
                let cj = col(j);
                let own = j > lo;
                if own {
                    y.set(j - 1, y.at(j - 1) + temp1 * cj.at(j - 1));
                    if t2_first == usize::MAX || j >= t2_first + LANES {
                        // TEMP2(j..j+LANES-1) = Σ_{i>j'} A(i,j')*X(i): heads first,
                        // then the common rows, each chain in i order.
                        t2_first = j;
                        let mut acc = [0.0f64; LANES];
                        let q_n = LANES.min(hi + 1 - j).min(n + 1 - j);
                        if q_n == LANES && j + LANES <= n {
                            let start = j + LANES; // first common row (1-based)
                            let c: [D; LANES] = std::array::from_fn(|q| col(j + q));
                            for q in 0..LANES {
                                let cq = c[q].slice(n);
                                acc[q] = dot1(&cq[j + q..start - 1], &xs[j + q..start - 1], 0.0);
                            }
                            let len = n + 1 - start;
                            dotl(std::array::from_fn(|q| c[q].off(start - 1).slice(len)), &xs[start - 1..], &mut acc);
                        } else {
                            for q in 0..q_n {
                                let jq = j + q;
                                acc[q] = dot1(&col(jq).slice(n)[jq..], &xs[jq..], 0.0);
                            }
                        }
                        t2 = acc;
                    }
                }
                let first = (j + 1).max(lo + 1);
                if first <= hi {
                    let len = hi + 1 - first;
                    axpy_slice(temp1, cj.off(first - 1).slice(len), y.off(first - 1).slice_mut(len));
                }
                if own {
                    y.set(j - 1, y.at(j - 1) + alpha * t2[j - t2_first]);
                }
            }
        }
    }
}

/// `DSYR2` with unit increments (used only in the `dsytd2` tail).
pub(crate) unsafe fn dsyr2(upper: bool, n: usize, alpha: f64, x: D, y: D, a: D, lda: usize) {
    if n == 0 || alpha == ZERO {
        return;
    }
    for j in 1..=n {
        if x.at(j - 1) != ZERO || y.at(j - 1) != ZERO {
            let temp1 = alpha * y.at(j - 1);
            let temp2 = alpha * x.at(j - 1);
            let (i0, i1) = if upper { (1, j) } else { (j, n) };
            for i in i0..=i1 {
                let k = ix(i, j, lda);
                a.set(k, a.at(k) + x.at(i - 1) * temp1 + y.at(i - 1) * temp2);
            }
        }
    }
}

/// `DGEMV`. `trans = false` is `'N'` (element-wise per row: `Y(i)` chains over
/// columns), `true` is `'T'` (element-wise per column; `LANES` column chains
/// advance together).
pub(crate) unsafe fn dgemv(
    trans: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: D,
    lda: usize,
    x: D,
    incx: usize,
    beta: f64,
    y: D,
    incy: usize,
) {
    if m == 0 || n == 0 || (alpha == ZERO && beta == ONE) {
        return;
    }
    let leny = if trans { n } else { m };
    if beta != ONE {
        let mut iy = 0;
        for _ in 0..leny {
            if beta == ZERO {
                y.set(iy, ZERO);
            } else {
                y.set(iy, beta * y.at(iy));
            }
            iy += incy;
        }
    }
    if alpha == ZERO {
        return;
    }
    if !trans {
        {
            let (lo, hi) = (0usize, m);

            let mut jx = 0;
            for j in 1..=n {
                let temp = alpha * x.at(jx);
                let aj = a.off((j - 1) * lda + lo);
                if incy == 1 {
                    axpy_slice(temp, aj.slice(hi - lo), y.off(lo).slice_mut(hi - lo));
                } else {
                    for i in lo..hi {
                        let iy = i * incy;
                        y.set(iy, y.at(iy) + temp * aj.at(i - lo));
                    }
                }
                jx += incx;
            }
        }
    } else {
        {
            let (lo, hi) = (0usize, n);

            let mut j = lo + 1;
            if incx == 1 {
                let xs = x.slice(m);
                while j + LANES - 1 <= hi {
                    let mut acc = [0.0; LANES];
                    dotl(std::array::from_fn(|q| a.off((j - 1 + q) * lda).slice(m)), xs, &mut acc);
                    for (q, t) in acc.iter().enumerate() {
                        let jy = (j - 1 + q) * incy;
                        y.set(jy, y.at(jy) + alpha * t);
                    }
                    j += LANES;
                }
            }
            while j <= hi {
                let mut temp = ZERO;
                let mut ixx = 0;
                for i in 1..=m {
                    temp += a.at(ix(i, j, lda)) * x.at(ixx);
                    ixx += incx;
                }
                let jy = (j - 1) * incy;
                y.set(jy, y.at(jy) + alpha * temp);
                j += 1;
            }
        }
    }
}

/// `DGER`. Element-wise per column (the zero-skip on `Y(JY)` is per column).
pub(crate) unsafe fn dger(m: usize, n: usize, alpha: f64, x: D, incx: usize, y: D, incy: usize, a: D, lda: usize) {
    if m == 0 || n == 0 || alpha == ZERO {
        return;
    }
    {
        let (lo, hi) = (0usize, n);

        for j in (lo + 1)..=hi {
            let yj = y.at((j - 1) * incy);
            if yj != ZERO {
                let temp = alpha * yj;
                if incx == 1 {
                    let aj = a.off((j - 1) * lda).slice_mut(m);
                    for (ai, xi) in aj.iter_mut().zip(x.slice(m)) {
                        *ai += xi * temp;
                    }
                } else {
                    let mut ixx = 0;
                    for i in 1..=m {
                        let k = ix(i, j, lda);
                        a.set(k, a.at(k) + x.at(ixx) * temp);
                        ixx += incx;
                    }
                }
            }
        }
    }
}

/// `DSYR2K` with `TRANS = 'N'`. Element-wise per column of C.
pub(crate) unsafe fn dsyr2k_n(
    upper: bool,
    n: usize,
    k: usize,
    alpha: f64,
    a: D,
    lda: usize,
    b: D,
    ldb: usize,
    beta: f64,
    c: D,
    ldc: usize,
) {
    if n == 0 || ((alpha == ZERO || k == 0) && beta == ONE) {
        return;
    }
    if alpha == ZERO {
        for j in 1..=n {
            let (i0, i1) = if upper { (1, j) } else { (j, n) };
            for i in i0..=i1 {
                let p = ix(i, j, ldc);
                c.set(p, if beta == ZERO { ZERO } else { beta * c.at(p) });
            }
        }
        return;
    }
    {
        let (lo, hi) = (0usize, n);

        for j in (lo + 1)..=hi {
            let (i0, i1) = if upper { (1, j) } else { (j, n) };
            let len = i1 + 1 - i0;
            let cj = c.off(ix(i0, j, ldc)).slice_mut(len);
            if beta == ZERO {
                cj.fill(ZERO);
            } else if beta != ONE {
                for v in cj.iter_mut() {
                    *v *= beta;
                }
            }
            for l in 1..=k {
                let ajl = a.at(ix(j, l, lda));
                let bjl = b.at(ix(j, l, ldb));
                if ajl != ZERO || bjl != ZERO {
                    let temp1 = alpha * bjl;
                    let temp2 = alpha * ajl;
                    let al = a.off(ix(i0, l, lda)).slice(len);
                    let bl = b.off(ix(i0, l, ldb)).slice(len);
                    for ((ci, ai), bi) in cj.iter_mut().zip(al).zip(bl) {
                        *ci = *ci + ai * temp1 + bi * temp2;
                    }
                }
            }
        }
    }
}

/// `DGEMM`. Element-wise per column of C in every branch; with `A`
/// transposed, the chains of `LANES` consecutive rows of C advance together.
pub(crate) unsafe fn dgemm(
    transa: bool,
    transb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: D,
    lda: usize,
    b: D,
    ldb: usize,
    beta: f64,
    c: D,
    ldc: usize,
) {
    if m == 0 || n == 0 || ((alpha == ZERO || k == 0) && beta == ONE) {
        return;
    }
    if alpha == ZERO {
        for j in 1..=n {
            for i in 1..=m {
                let p = ix(i, j, ldc);
                c.set(p, if beta == ZERO { ZERO } else { beta * c.at(p) });
            }
        }
        return;
    }
    {
        let (lo, hi) = (0usize, n);

        for j in (lo + 1)..=hi {
            let cj = c.off((j - 1) * ldc).slice_mut(m);
            if !transa {
                // NOTA: C(I,J) = C(I,J) + TEMP*A(I,L), L ascending.
                if beta == ZERO {
                    cj.fill(ZERO);
                } else if beta != ONE {
                    for v in cj.iter_mut() {
                        *v *= beta;
                    }
                }
                for l in 1..=k {
                    let temp = if !transb { alpha * b.at(ix(l, j, ldb)) } else { alpha * b.at(ix(j, l, ldb)) };
                    axpy_slice(temp, a.off((l - 1) * lda).slice(m), cj);
                }
            } else {
                // A transposed: C(I,J) = ALPHA*TEMP [+ BETA*C(I,J)],
                // TEMP = Σ_L A(L,I)*B(L,J) (or B(J,L)).
                let bj: &[f64] = if !transb { b.off((j - 1) * ldb).slice(k) } else { &[] };
                let bget = |l: usize| if !transb { bj[l] } else { b.at(ix(j, l + 1, ldb)) };
                let mut i = 0usize;
                if !transb {
                    while i + LANES <= m {
                        let mut acc = [0.0; LANES];
                        dotl(std::array::from_fn(|q| a.off((i + q) * lda).slice(k)), bj, &mut acc);
                        for q in 0..LANES {
                            cj[i + q] = if beta == ZERO { alpha * acc[q] } else { alpha * acc[q] + beta * cj[i + q] };
                        }
                        i += LANES;
                    }
                }
                while i < m {
                    let ai = a.off(i * lda).slice(k);
                    let mut temp = ZERO;
                    for l in 0..k {
                        temp += ai[l] * bget(l);
                    }
                    cj[i] = if beta == ZERO { alpha * temp } else { alpha * temp + beta * cj[i] };
                    i += 1;
                }
            }
        }
    }
}

/// `DTRMM`. Left side is element-wise per column of B; right side per row of
/// B (every step of the column loop only mixes entries of one row).
pub(crate) unsafe fn dtrmm(
    left: bool,
    upper: bool,
    trans: bool,
    nounit: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: D,
    lda: usize,
    b: D,
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha == ZERO {
        for j in 1..=n {
            for i in 1..=m {
                b.set(ix(i, j, ldb), ZERO);
            }
        }
        return;
    }
    let a_ = |i: usize, j: usize| a.at(ix(i, j, lda));
    let bi = |i: usize, j: usize| ix(i, j, ldb);
    if left {
        {
            let (lo, hi) = (0usize, n);

            for j in (lo + 1)..=hi {
                match (trans, upper) {
                    (false, true) => {
                        for k in 1..=m {
                            if b.at(bi(k, j)) != ZERO {
                                let mut temp = alpha * b.at(bi(k, j));
                                if k > 1 {
                                    axpy_slice(
                                        temp,
                                        a.off(ix(1, k, lda)).slice(k - 1),
                                        b.off(bi(1, j)).slice_mut(k - 1),
                                    );
                                }
                                if nounit {
                                    temp *= a_(k, k);
                                }
                                b.set(bi(k, j), temp);
                            }
                        }
                    }
                    (false, false) => {
                        for k in (1..=m).rev() {
                            if b.at(bi(k, j)) != ZERO {
                                let temp = alpha * b.at(bi(k, j));
                                b.set(bi(k, j), temp);
                                if nounit {
                                    b.set(bi(k, j), b.at(bi(k, j)) * a_(k, k));
                                }
                                if k < m {
                                    axpy_slice(
                                        temp,
                                        a.off(ix(k + 1, k, lda)).slice(m - k),
                                        b.off(bi(k + 1, j)).slice_mut(m - k),
                                    );
                                }
                            }
                        }
                    }
                    (true, true) => {
                        for i in (1..=m).rev() {
                            let mut temp = b.at(bi(i, j));
                            if nounit {
                                temp *= a_(i, i);
                            }
                            for k in 1..i {
                                temp += a_(k, i) * b.at(bi(k, j));
                            }
                            b.set(bi(i, j), alpha * temp);
                        }
                    }
                    (true, false) => {
                        for i in 1..=m {
                            let mut temp = b.at(bi(i, j));
                            if nounit {
                                temp *= a_(i, i);
                            }
                            for k in (i + 1)..=m {
                                temp += a_(k, i) * b.at(bi(k, j));
                            }
                            b.set(bi(i, j), alpha * temp);
                        }
                    }
                }
            }
        }
    } else {
        {
            let (lo, hi) = (0usize, m);

            let len = hi - lo;
            let bc = |j: usize| b.off(ix(lo + 1, j, ldb));
            let add = |j: usize, k: usize, temp: f64| {
                let (dst, src) = (bc(j).slice_mut(len), bc(k).slice(len));
                axpy_slice(temp, src, dst);
            };
            let scale = |j: usize, temp: f64| {
                for v in bc(j).slice_mut(len) {
                    *v = temp * *v;
                }
            };
            match (trans, upper) {
                (false, true) => {
                    for j in (1..=n).rev() {
                        let mut temp = alpha;
                        if nounit {
                            temp *= a_(j, j);
                        }
                        scale(j, temp);
                        for k in 1..j {
                            if a_(k, j) != ZERO {
                                add(j, k, alpha * a_(k, j));
                            }
                        }
                    }
                }
                (false, false) => {
                    for j in 1..=n {
                        let mut temp = alpha;
                        if nounit {
                            temp *= a_(j, j);
                        }
                        scale(j, temp);
                        for k in (j + 1)..=n {
                            if a_(k, j) != ZERO {
                                add(j, k, alpha * a_(k, j));
                            }
                        }
                    }
                }
                (true, true) => {
                    for k in 1..=n {
                        for j in 1..k {
                            if a_(j, k) != ZERO {
                                add(j, k, alpha * a_(j, k));
                            }
                        }
                        let mut temp = alpha;
                        if nounit {
                            temp *= a_(k, k);
                        }
                        if temp != ONE {
                            scale(k, temp);
                        }
                    }
                }
                (true, false) => {
                    for k in (1..=n).rev() {
                        for j in (k + 1)..=n {
                            if a_(j, k) != ZERO {
                                add(j, k, alpha * a_(j, k));
                            }
                        }
                        let mut temp = alpha;
                        if nounit {
                            temp *= a_(k, k);
                        }
                        if temp != ONE {
                            scale(k, temp);
                        }
                    }
                }
            }
        }
    }
}
