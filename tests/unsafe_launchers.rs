//! Error-guide prevention step: every `ArrayArg::from_raw_parts` sits inside
//! an `unsafe` block with a `// SAFETY:` note, confined to the launcher
//! modules (`device/ops.rs`, `device/probe.rs`).

use std::path::Path;

fn rs_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).expect("read_dir") {
        let p = e.expect("entry").path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn from_raw_parts_is_confined_and_annotated() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rs_files(&root.join("src"), &mut files);
    let allowed = ["src/device/ops.rs", "src/device/probe.rs"];
    let mut bad = Vec::new();
    let mut seen = 0;
    for f in &files {
        let rel = f.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        let text = std::fs::read_to_string(f).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        for (ln, line) in lines.iter().enumerate() {
            if !line.contains("ArrayArg::from_raw_parts") || line.trim_start().starts_with("//") {
                continue;
            }
            seen += 1;
            if !allowed.contains(&rel.as_str()) {
                bad.push(format!("{rel}:{}: outside launcher modules", ln + 1));
                continue;
            }
            // Walk back to the enclosing block opener; it must be `unsafe {`
            // preceded (directly) by a `// SAFETY:` comment block.
            let mut depth = 0i32;
            let mut ok = false;
            for back in (0..ln).rev() {
                let t = lines[back];
                depth += t.matches('}').count() as i32 - t.matches('{').count() as i32;
                if depth < 0 {
                    if t.trim_start().starts_with("unsafe {") {
                        let mut c = back;
                        while c > 0 && lines[c - 1].trim_start().starts_with("//") {
                            c -= 1;
                            if lines[c].contains("SAFETY:") {
                                ok = true;
                                break;
                            }
                        }
                    }
                    // An intermediate `kl!(` / call opener is a paren, not a brace.
                    break;
                }
            }
            if !ok {
                bad.push(format!("{rel}:{}: not inside an annotated `unsafe {{` block", ln + 1));
            }
        }
    }
    assert!(seen > 0);
    assert!(bad.is_empty(), "{bad:#?}");
}
