//! `cargo xtask build-ref-lapack`: builds the bit-exact oracle.
//!
//! Reference-LAPACK v3.12.1 plus the reference BLAS from the same tree, from a
//! SHA-256-verified tarball, with pinned flags:
//!
//! * `-march=x86-64` has no FMA instructions;
//! * `-ffp-contract=off` stops contraction even if the arch is raised later;
//! * GCC never reassociates FP without `-fassociative-math`, so `-O2`
//!   vectorisation keeps the loop order.
//!
//! Output under `target/ref-lapack/install/`: `lib/librefblas.a`,
//! `lib/liblapack.a` and `ref-lapack.toml` (version, SHA-256, flags,
//! `gfortran --version`, and `nm` proof that no `openblas_*` symbol exists).
//!
//! The oracle is rebuilt only deliberately: an existing `ref-lapack.toml`
//! whose flags or compiler differ aborts the build unless `--force` is given.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

const VERSION: &str = "3.12.1";
const URL: &str = "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz";
const SHA256: &str = "2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422";
pub const FORTRAN_FLAGS: &str = "-O2 -march=x86-64 -mtune=generic -ffp-contract=off -fno-fast-math -frecursive";

const EXTRAS_F: &str = r#"
      SUBROUTINE XZDIV( A, B, C )
      COMPLEX*16 A, B, C
      C = A / B
      END
      SUBROUTINE XZMUL( A, B, C )
      COMPLEX*16 A, B, C
      C = A * B
      END
      SUBROUTINE XRZMUL( R, A, C )
      DOUBLE PRECISION R
      COMPLEX*16 A, C
      C = R * A
      END
      SUBROUTINE XPOWI( X, N, Y )
      DOUBLE PRECISION X, Y
      INTEGER N
      Y = X**N
      END
      SUBROUTINE XCABS( A, R )
      COMPLEX*16 A
      DOUBLE PRECISION R
      R = ABS( A )
      END
      SUBROUTINE XMAX( A, B, R )
      DOUBLE PRECISION A, B, R
      R = MAX( A, B )
      END
      SUBROUTINE XMIN( A, B, R )
      DOUBLE PRECISION A, B, R
      R = MIN( A, B )
      END
      SUBROUTINE XSIGN( A, B, R )
      DOUBLE PRECISION A, B, R
      R = SIGN( A, B )
      END
"#;

fn run(cmd: &mut Command) -> String {
    eprintln!("+ {cmd:?}");
    let out = cmd.output().unwrap_or_else(|e| {
        eprintln!("failed to spawn {cmd:?}: {e}");
        exit(1)
    });
    if !out.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&out.stdout));
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        eprintln!("command failed: {cmd:?}");
        exit(1);
    }
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().expect("xtask sits in the workspace").to_path_buf()
}

fn sha256(path: &Path) -> String {
    let out = run(Command::new("sha256sum").arg(path));
    out.split_whitespace().next().unwrap_or_default().to_string()
}

fn main() {
    let force = std::env::args().any(|a| a == "--force");
    let root = workspace_root().join("target/ref-lapack");
    fs::create_dir_all(&root).expect("create target/ref-lapack");
    let tarball = root.join(format!("lapack-{VERSION}.tar.gz"));
    if !tarball.exists() {
        run(Command::new("curl").args(["-sSL", "-o"]).arg(&tarball).arg(URL));
    }
    let got = sha256(&tarball);
    if got != SHA256 {
        eprintln!("SHA-256 mismatch for {}: got {got}, want {SHA256}", tarball.display());
        exit(1);
    }
    let src = root.join(format!("lapack-{VERSION}"));
    if !src.join("CMakeLists.txt").exists() {
        run(Command::new("tar").arg("xzf").arg(&tarball).arg("-C").arg(&root));
    }

    let gfortran = run(Command::new("gfortran").arg("--version"));
    let gfortran = gfortran.lines().next().unwrap_or_default().to_string();
    let install = root.join("install");
    let manifest = install.join("ref-lapack.toml");
    if manifest.exists() && !force {
        let old = fs::read_to_string(&manifest).unwrap_or_default();
        let same = old.contains(&format!("fortran_flags = \"{FORTRAN_FLAGS}\""))
            && old.contains(&format!("gfortran = \"{gfortran}\""));
        if !same {
            eprintln!(
                "existing oracle at {} was built with different flags or compiler;\n\
                 rebuilding changes the oracle. Re-run with --force to do it deliberately.\n\
                 old manifest:\n{old}",
                manifest.display()
            );
            exit(1);
        }
    }

    let build = root.join("build");
    fs::create_dir_all(&build).expect("create build dir");
    run(Command::new("cmake")
        .arg("-S")
        .arg(&src)
        .arg("-B")
        .arg(&build)
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DCBLAS=OFF")
        .arg("-DLAPACKE=OFF")
        .arg("-DBUILD_TESTING=OFF")
        .arg("-DUSE_OPTIMIZED_BLAS=OFF")
        .arg("-DBUILD_SINGLE=ON")
        .arg("-DBUILD_DOUBLE=ON")
        .arg("-DBUILD_COMPLEX=OFF")
        .arg("-DBUILD_COMPLEX16=ON")
        .arg("-DCMAKE_BUILD_TYPE=None")
        .arg(format!("-DCMAKE_Fortran_FLAGS={FORTRAN_FLAGS}"))
        .arg("-DCMAKE_Fortran_COMPILER=gfortran")
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", install.display()))
        .arg("-DCMAKE_INSTALL_LIBDIR=lib"));
    let jobs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    run(Command::new("cmake").arg("--build").arg(&build).arg("--parallel").arg(jobs.to_string()));
    run(Command::new("cmake").arg("--install").arg(&build));

    let lib = install.join("lib");
    // 3.12.1 installs the reference BLAS as `libblas.a`; the oracle names it
    // `librefblas.a` so a linker search path can never pick a system BLAS.
    if lib.join("libblas.a").exists() {
        fs::rename(lib.join("libblas.a"), lib.join("librefblas.a")).expect("rename libblas.a");
    }
    // gfortran-lowering probes compiled with the oracle's exact flags: inline
    // complex division and multiplication, `x**n`, `ABS(COMPLEX)`,
    // `MAX`/`MIN`/`SIGN` on signed zeros.
    let extras_src = root.join("lapackrs_extras.f");
    fs::write(&extras_src, EXTRAS_F).expect("write extras source");
    let extras_obj = root.join("lapackrs_extras.o");
    let mut gf = Command::new("gfortran");
    gf.args(FORTRAN_FLAGS.split_whitespace()).arg("-c").arg(&extras_src).arg("-o").arg(&extras_obj);
    run(&mut gf);
    let extras_lib = lib.join("liblapackrs_extras.a");
    let _ = fs::remove_file(&extras_lib);
    run(Command::new("ar").arg("rcs").arg(&extras_lib).arg(&extras_obj));

    let mut nm_openblas = 0usize;
    for a in ["librefblas.a", "liblapack.a", "liblapackrs_extras.a"] {
        let path = lib.join(a);
        if !path.exists() {
            eprintln!("missing {}", path.display());
            exit(1);
        }
        let syms = run(Command::new("nm").arg(&path));
        nm_openblas += syms.lines().filter(|l| l.contains("openblas")).count();
    }
    if nm_openblas != 0 {
        eprintln!("oracle archives contain {nm_openblas} openblas symbols");
        exit(1);
    }
    let dsyevd = run(Command::new("nm").arg(lib.join("liblapack.a")));
    let has_dsyevd = dsyevd.lines().any(|l| l.ends_with(" T dsyevd_"));
    let text = format!(
        "# Written by `cargo xtask build-ref-lapack`. Checked by tests/ref_oracle_canary.rs.\n\
         version = \"{VERSION}\"\n\
         url = \"{URL}\"\n\
         sha256 = \"{SHA256}\"\n\
         fortran_flags = \"{FORTRAN_FLAGS}\"\n\
         cmake_build_type = \"None\"\n\
         blas = \"reference (same tree, USE_OPTIMIZED_BLAS=OFF)\"\n\
         extras = \"liblapackrs_extras.a (gfortran lowering probes, same flags)\"\n\
         gfortran = \"{gfortran}\"\n\
         nm_openblas_symbols = {nm_openblas}\n\
         nm_has_dsyevd = {has_dsyevd}\n"
    );
    fs::write(&manifest, text).expect("write ref-lapack.toml");
    println!("oracle ready: LAPACK_RS_REF_DIR={}", install.display());
}
