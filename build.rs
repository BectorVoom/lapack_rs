//! Clears CubeCL's on-disk kernel cache whenever anything that changes the
//! generated device code changes.
//!
//! `cubecl.toml` turns the compilation cache on (`[compilation] cache =
//! "target"`), which is worth seconds of NVRTC/hipRTC time per process. The
//! cache key is `KernelId::stable_hash` — type name, address type, cube dim,
//! execution mode and `#[comptime]` arguments — and covers neither the kernel
//! body nor the device compiler flags (CubeCL manual 15 §2.5 and §4), so a
//! warm cache would serve binaries compiled from other source, or with FMA
//! contraction left on. Both would silently break the bit-exactness contract.
//!
//! The stamp therefore digests everything that can change the device binary:
//!
//! * every `.rs` under `src/device` (the `#[cube]` kernels and their
//!   launchers),
//! * every `.rs` under `crates/cubecl-cuda/src` (the vendored, patched
//!   backend),
//! * `NVRTC_COMPILE_OPTIONS_APPEND` and `HIPRTC_COMPILE_OPTIONS_APPEND`.
//!
//! and it is written **next to the cache it guards** (`target/<backend>/`),
//! not into `OUT_DIR`: `OUT_DIR` is per profile and per feature set, while the
//! cache directory is shared by all of them, so an `OUT_DIR` stamp can say
//! "unchanged" about a cache another build filled from different source (the
//! edit/revert loop of a bisect is enough to hit it).
//!
//! A CubeCL version bump is already handled upstream: the cache lives in a
//! versioned subdirectory.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::{env, fs, io};

/// Backends with an on-disk cache under `target/` (the CPU runtime has none).
const BACKENDS: [&str; 3] = ["cuda", "hip", "vulkan"];

fn digest_dir(dir: &Path, hasher: &mut DefaultHasher) -> io::Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            digest_dir(&path, hasher)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            path.file_name().hash(hasher);
            fs::read(&path)?.hash(hasher);
        }
    }
    Ok(())
}

fn main() {
    for var in ["NVRTC_COMPILE_OPTIONS_APPEND", "HIPRTC_COMPILE_OPTIONS_APPEND"] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    for path in ["src/device", "crates/cubecl-cuda/src", "build.rs", "cubecl.toml"] {
        println!("cargo::rerun-if-changed={path}");
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()));
    let mut hasher = DefaultHasher::new();
    for dir in ["src/device", "crates/cubecl-cuda/src"] {
        let dir = manifest.join(dir);
        if dir.exists() && digest_dir(&dir, &mut hasher).is_err() {
            return; // Unreadable sources: leave the cache alone rather than guess.
        }
    }
    for var in ["NVRTC_COMPILE_OPTIONS_APPEND", "HIPRTC_COMPILE_OPTIONS_APPEND"] {
        env::var(var).unwrap_or_default().hash(&mut hasher);
    }
    let digest = format!("{:016x}", hasher.finish());

    // `cache = "target"` resolves to the first `Cargo.toml` above the running
    // process's working directory (`CacheConfig::Target`), which for this
    // crate's `cargo test`/`cargo run` is this package. Only directories under
    // it are ever removed.
    for backend in BACKENDS {
        let dir = manifest.join("target").join(backend);
        let stamp = dir.join(".cube-sources.digest");
        if fs::read_to_string(&stamp).is_ok_and(|prev| prev.trim() == digest) {
            continue; // This cache was filled from exactly these sources and flags.
        }
        let _ = fs::remove_dir_all(&dir);
        if fs::create_dir_all(&dir).is_ok() {
            let _ = fs::write(&stamp, &digest);
        }
    }
}
