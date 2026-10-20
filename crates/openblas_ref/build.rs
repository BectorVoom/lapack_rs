//! Links scipy's bundled OpenBLAS (`libscipy_openblas-*.so`) from
//! `LAPACK_RS_OPENBLAS_DIR` (the wheel's `scipy.libs` directory), so the
//! oracle is provably the binary `scipy` itself calls.
//!
//! Without the variable the crate builds with `cfg(openblas_ref)` unset: every
//! wrapper reports "unavailable" and the oracle tests SKIP, or FAIL when
//! `OPENBLAS_REQUIRE_REF=1`.
//!
//! The library and its bundled `libgfortran` are symlinked into `OUT_DIR`,
//! which cargo puts on the dynamic loader path of `cargo test`/`cargo run`.

use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(openblas_ref)");
    println!("cargo:rerun-if-env-changed=LAPACK_RS_OPENBLAS_DIR");
    let Some(dir) = std::env::var_os("LAPACK_RS_OPENBLAS_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    let mut lib = None;
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("LAPACK_RS_OPENBLAS_DIR={}: {e}", dir.display())) {
        let entry = entry.expect("dir entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.contains(".so") {
            continue;
        }
        let link = out.join(&name);
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(entry.path(), &link).expect("symlink");
        println!("cargo:rerun-if-changed={}", entry.path().display());
        // The LP64 scipy build (`libscipy_openblas-<hash>.so`), not numpy's
        // ILP64 `libscipy_openblas64_`.
        if name.starts_with("libscipy_openblas-") && name.ends_with(".so") {
            lib = Some(name);
        }
    }
    let lib = lib.unwrap_or_else(|| panic!("LAPACK_RS_OPENBLAS_DIR={} has no libscipy_openblas-*.so", dir.display()));
    println!("cargo:rustc-env=LAPACK_RS_OPENBLAS_LIB={}", dir.join(&lib).display());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib:+verbatim={lib}");
    println!("cargo:rustc-cfg=openblas_ref");
}
