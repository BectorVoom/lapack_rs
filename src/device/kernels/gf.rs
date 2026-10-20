//! gfortran's scalar lowerings, shared by every kernel.
//!
//! `MAX(A, B)` lowers to `MAX_EXPR`, which gcc emits as x86 `maxsd`: the
//! second operand wins on equality, on a signed-zero tie and on `NaN`. `MIN`
//! is the mirror image. Both are written as `select`, never as `F::max` /
//! `F::min`, because the backends' `max`/`min` intrinsics do not agree with
//! that on `NaN` or `-0.0`.
//!
//! One caveat the host side pays for too: `MAX_EXPR` is commutative in GIMPLE
//! and gcc canonicalises its operand order, so a Fortran `MIN( A, B )` is not
//! always `minsd src=B, dst=A` — `dlasq5.f`'s `EMIN` is emitted the other way
//! round (see [`crate::lasq`]). Transcribe the emitted order, not the source
//! order, wherever `NaN` or a signed zero can reach the comparison.

use cubecl::prelude::*;

/// gfortran's `MAX(a, b)`.
#[cube]
pub fn fmax_c<F: Float>(a: F, b: F) -> F {
    select(a > b, a, b)
}

/// gfortran's `MIN(a, b)`.
#[cube]
pub fn fmin_c<F: Float>(a: F, b: F) -> F {
    select(a < b, a, b)
}
