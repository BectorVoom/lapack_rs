//! `#[cube]` kernels (generic `F: Float + CubeElement`, launched `::<F, R>`).

pub mod cint_stemr;
pub mod copy;
pub mod gemm;
pub mod gf;
pub mod merge;
pub mod ob_ormtr;
pub mod ob_stedc;
pub mod ob_symv;
pub mod ob_syr2k;
pub mod ob_zhetrd;
pub mod ob_zormtr;
pub mod secular;
pub mod stemr;
pub mod symv;
pub mod syr2k;
pub mod zkernels;
