//! Tolerance report against `scipy.linalg.lapack.dsyevd`/`zheevd` (OpenBLAS
//! LAPACK). Report only: OpenBLAS kernels are not a bit-exact target. Runs
//! when `LAPACK_RS_ORACLE_PYTHON` names a Python with numpy and scipy.

mod common;
use common::*;
use lapack_rs::{Job, Uplo, dsyevd_alloc, zheevd_alloc};
use std::io::Write;
use std::process::{Command, Stdio};

const SCRIPT: &str = r#"
import sys, struct, numpy as np
from scipy.linalg import lapack
data = sys.stdin.buffer.read()
kind, n = struct.unpack('<ii', data[:8])
buf = np.frombuffer(data[8:], dtype='<f8')
if kind == 0:
    a = buf[:n*n].reshape((n, n), order='F').copy()
    w, v, info = lapack.dsyevd(a, compute_v=1, lower=1)
else:
    a = (buf[:n*n] + 1j*buf[n*n:2*n*n]).reshape((n, n), order='F').copy()
    w, v, info = lapack.zheevd(a, compute_v=1, lower=1)
sys.stdout.buffer.write(struct.pack('<i', info) + np.asarray(w, dtype='<f8').tobytes())
"#;

fn scipy_eigenvalues(py: &str, kind: i32, n: usize, parts: &[&[f64]]) -> Vec<f64> {
    let mut child = Command::new(py)
        .arg("-c")
        .arg(SCRIPT)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn python");
    let mut input = Vec::new();
    input.extend_from_slice(&kind.to_le_bytes());
    input.extend_from_slice(&(n as i32).to_le_bytes());
    for p in parts {
        for v in *p {
            input.extend_from_slice(&v.to_le_bytes());
        }
    }
    child.stdin.take().expect("stdin").write_all(&input).expect("write");
    let out = child.wait_with_output().expect("python");
    assert!(out.status.success(), "python failed");
    let info = i32::from_le_bytes(out.stdout[..4].try_into().expect("info"));
    assert_eq!(info, 0, "scipy INFO");
    out.stdout[4..].chunks(8).map(|c| f64::from_le_bytes(c.try_into().expect("f64"))).collect()
}

#[test]
fn scipy_tolerance_report() {
    let Ok(py) = std::env::var("LAPACK_RS_ORACLE_PYTHON") else {
        eprintln!("SKIP scipy_tolerance_report: LAPACK_RS_ORACLE_PYTHON not set");
        return;
    };
    println!("{:>6} {:>8} {:>14} {:>14}", "n", "kind", "max|dw|", "bit-equal w");
    for &n in &[16usize, 64, 200, 513] {
        let a = random_sym(n, 900 + n as u64);
        let (mut z, mut w) = (a.clone(), vec![0.0; n]);
        dsyevd_alloc(Job::Vectors, Uplo::Lower, n, &mut z, &mut w).expect("dsyevd");
        let ws = scipy_eigenvalues(&py, 0, n, &[&a]);
        let dmax = w.iter().zip(&ws).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        let same = w.iter().zip(&ws).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        println!("{n:>6} {:>8} {dmax:>14.3e} {same:>10}/{n}", "real");
        assert!(dmax <= 1e-10 * n as f64, "real n={n}: {dmax:e}");

        let (re, im) = random_herm(n, 950 + n as u64);
        let (mut zr, mut zi, mut w) = (re.clone(), im.clone(), vec![0.0; n]);
        zheevd_alloc(Job::Vectors, Uplo::Lower, n, &mut zr, &mut zi, &mut w).expect("zheevd");
        let ws = scipy_eigenvalues(&py, 1, n, &[&re, &im]);
        let dmax = w.iter().zip(&ws).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        let same = w.iter().zip(&ws).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        println!("{n:>6} {:>8} {dmax:>14.3e} {same:>10}/{n}", "complex");
        assert!(dmax <= 1e-10 * n as f64, "complex n={n}: {dmax:e}");
    }
}
