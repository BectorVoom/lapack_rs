//! Links Reference-LAPACK 3.12.1 + reference BLAS from `LAPACK_RS_REF_DIR`
//! (the `install` prefix written by `cargo xtask build-ref-lapack`).
//!
//! Without the variable the crate builds with `cfg(lapack_ref)` unset: every
//! wrapper reports "unavailable" and the oracle tests SKIP, or FAIL when
//! `LAPACK_RS_REQUIRE_REF=1`.

use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(lapack_ref)");
    println!("cargo:rustc-check-cfg=cfg(lapack_ref_extras)");
    println!("cargo:rerun-if-env-changed=LAPACK_RS_REF_DIR");
    let Some(dir) = std::env::var_os("LAPACK_RS_REF_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let lib = dir.join("lib");
    for a in ["liblapack.a", "librefblas.a"] {
        let p = lib.join(a);
        if !p.exists() {
            panic!("LAPACK_RS_REF_DIR={} has no lib/{a}; run `cargo xtask build-ref-lapack`", dir.display());
        }
        println!("cargo:rerun-if-changed={}", p.display());
    }
    let manifest = dir.join("ref-lapack.toml");
    println!("cargo:rerun-if-changed={}", manifest.display());
    println!("cargo:rustc-env=LAPACK_RS_REF_MANIFEST={}", manifest.display());
    println!("cargo:rustc-link-search=native={}", lib.display());
    // Static archives, LAPACK before the BLAS it calls.
    if lib.join("liblapackrs_extras.a").exists() {
        println!("cargo:rustc-link-lib=static=lapackrs_extras");
        println!("cargo:rustc-cfg=lapack_ref_extras");
    }
    println!("cargo:rustc-link-lib=static=lapack");
    println!("cargo:rustc-link-lib=static=refblas");
    println!("cargo:rustc-link-lib=dylib=gfortran");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-cfg=lapack_ref");
}
