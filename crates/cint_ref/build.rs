//! Builds libcint 6.1.3's `src/eigh.c` (the `#else`, no-`LAPACK_FOUND` branch)
//! and `src/rys_wheeler.c` as the test-only oracle for Path C of
//! `docs/cintx-lapack-blas-adoption-plan.md`.
//!
//! The pinned tree is `CINTX_LIBCINT_DIR` (cintx's `libcint-master/`). Without
//! it nothing is compiled and every wrapper reports "unavailable", exactly as
//! `lapack_ref` does for Reference-LAPACK.
//!
//! Flags follow cintx's own oracle build (`cintx/crates/cintx-oracle/build.rs`)
//! so the bits are the ones cintx's ground truth produces: `-std=gnu89`, cc's
//! default optimisation, `HAVE_QUADMATH_H`/`HAVE_EXPL`/`HAVE_SQRTL`/
//! `HAVE_FABSL`/`WITH_RANGE_COULOMB` all off, `FINT = int`.
//!
//! `_CINTdiagonalize` is renamed to `cintref_diagonalize_impl` on the command
//! line so `shim.c` can interpose a recording wrapper: driving
//! `CINTrys_jacobi`/`CINTrys_laguerre` then yields the *actual* Wheeler
//! tridiagonals libcint feeds its eigensolver, which is the A0 corpus.

use std::path::PathBuf;
use std::{env, fs};

fn main() {
    println!("cargo::rerun-if-env-changed=CINTX_LIBCINT_DIR");
    println!("cargo::rerun-if-changed=src/shim.c");
    println!("cargo::rustc-check-cfg=cfg(cint_ref)");

    let Some(root) = env::var_os("CINTX_LIBCINT_DIR").map(PathBuf::from) else {
        return;
    };
    let src = root.join("src");
    if !src.join("eigh.c").exists() {
        println!("cargo::warning=CINTX_LIBCINT_DIR has no src/eigh.c; cint_ref stays unavailable");
        return;
    }
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    // cint.h.in / cint_config.h.in are cmake templates; resolve them exactly
    // as cintx's oracle does.
    let cint_h = fs::read_to_string(root.join("include/cint.h.in"))
        .or_else(|_| fs::read_to_string(src.join("cint.h.in")))
        .expect("cint.h.in")
        .replace("@cint_VERSION@", "6.1.3")
        .replace("@cint_SOVERSION@", "6")
        .replace("#cmakedefine I8", "/* I8 disabled: FINT = int */")
        .replace("#cmakedefine CACHE_SIZE_I8", "/* CACHE_SIZE_I8 disabled */");
    let cint_h_path = out.join("cint.h");
    fs::write(&cint_h_path, &cint_h).expect("write cint.h");

    let cfg = fs::read_to_string(src.join("cint_config.h.in"))
        .expect("cint_config.h.in")
        .replace("#include \"cint.h\"", &format!("#include \"{}\"", cint_h_path.display()))
        .replace("#cmakedefine HAVE_EXPL", "/* HAVE_EXPL */")
        .replace("#cmakedefine HAVE_SQRTL", "/* HAVE_SQRTL */")
        .replace("#cmakedefine HAVE_FABSL", "/* HAVE_FABSL */")
        .replace("#cmakedefine HAVE_QUADMATH_H", "/* HAVE_QUADMATH_H */")
        .replace("#cmakedefine WITH_RANGE_COULOMB", "/* WITH_RANGE_COULOMB */");
    fs::write(out.join("cint_config.h"), &cfg).expect("write cint_config.h");

    // `eigh.c` alone is compiled with the rename, so its `_CINTdiagonalize`
    // body becomes `cintref_diagonalize_impl`.
    cc::Build::new()
        .warnings(false)
        .flag("-std=gnu89")
        .flag("-Wno-implicit-function-declaration")
        .define("_CINTdiagonalize", "cintref_diagonalize_impl")
        .include(&out)
        .include(root.join("include"))
        .include(&src)
        .file(src.join("eigh.c"))
        .compile("cint_ref_eigh");

    // `rys_wheeler.c` (and the Boys-function moments it needs) is compiled
    // WITHOUT the rename, so its `_CINTdiagonalize` call binds to the shim and
    // every Wheeler tridiagonal is recorded on the way through.
    cc::Build::new()
        .warnings(false)
        .flag("-std=gnu89")
        .flag("-Wno-implicit-function-declaration")
        .include(&out)
        .include(root.join("include"))
        .include(&src)
        .file(src.join("rys_wheeler.c"))
        .file(src.join("fmt.c"))
        .file("src/shim.c")
        .compile("cint_ref_wheeler");

    println!("cargo::rustc-cfg=cint_ref");
    println!("cargo::rustc-env=CINT_REF_DIR={}", root.display());
}
