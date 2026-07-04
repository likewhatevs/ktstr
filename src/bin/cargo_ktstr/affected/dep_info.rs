//! Parse a cargo/rustc per-artifact `.d` dep-info file into the set of
//! input source files, canonicalized to repo-root-relative paths.
//!
//! Cargo writes a `<artifact>.d` next to every built binary: a single make
//! rule `<abs-target-path>: <space-separated abs prereq paths>`. The prereq
//! list is every source file that fed the artifact -- for an scx scheduler
//! that includes the Rust sources, the generated skeleton bindings, and the
//! BPF `.bpf.c` / `.h` inputs (the BPF header search path is a symlink,
//! `rust/scx_cargo/bpf_h -> scheds/include`, so shared headers appear under
//! `bpf_h/...`; [`normalize_to_repo_relative`] resolves that symlink so a
//! change to the real `scheds/include/...` source is matched).
//!
//! Two-stage: [`parse_dep_info`] is a pure string->paths parser (no I/O,
//! unit-testable); [`normalize_to_repo_relative`] canonicalizes each path
//! (resolving symlinks + `..`) and returns it repo-root-relative, dropping
//! toolchain/registry headers that canonicalize OUTSIDE the repo (they can
//! never appear in a repo diff). A prereq that fails to canonicalize (deleted
//! since the build) falls back to a lexical normalization and is KEPT if it is
//! lexically inside the repo -- a deleted in-repo source is exactly a change
//! the affected computation must see, so it is never silently dropped.

use std::path::{Component, Path, PathBuf};

/// Parse a `.d` dep-info file's contents into its prerequisite input paths.
///
/// Handles the general make-rule shape defensively: backslash-newline line
/// continuations are joined; `#`-comment and empty lines are skipped; each
/// rule is split on its first colon (the target -- an absolute path with no
/// colon on unix -- precedes it); the prereq list is split on unescaped
/// whitespace with make-style `\<c>` unescaping (`\ ` is a literal space in a
/// path, not a separator). A phony `<prereq>:` rule (empty right-hand side)
/// contributes nothing.
pub(crate) fn parse_dep_info(contents: &str) -> Vec<PathBuf> {
    // Join `\`-at-end-of-line continuations into one logical line each.
    let joined = contents.replace("\\\n", " ");
    let mut out = Vec::new();
    for line in joined.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // The target precedes the first colon; everything after is prereqs.
        let Some((_, rhs)) = line.split_once(':') else {
            continue;
        };
        for tok in split_prereqs(rhs) {
            out.push(PathBuf::from(tok));
        }
    }
    out
}

/// Split a make-rule prerequisite list on unescaped whitespace, unescaping
/// `\<c>` (make escapes a literal space/hash/colon/backslash in a path).
fn split_prereqs(rhs: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    let mut chars = rhs.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                // The escaped next char is literal (make: `\ `, `\#`, `\\`,
                // `\:`). A trailing backslash is kept verbatim.
                match chars.next() {
                    Some(nc) => cur.push(nc),
                    None => cur.push('\\'),
                }
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    toks
}

/// Canonicalize a prereq path and return it repo-root-relative, or `None` if
/// it resolves OUTSIDE the repo (toolchain/registry headers).
///
/// `repo_root_canonical` MUST already be `fs::canonicalize`d by the caller
/// (once): `fs::canonicalize` on a prereq resolves symlinks, but
/// `gix::Repository::workdir()` is NOT symlink-resolved, so a symlinked repo
/// root would make every `strip_prefix` fail -- silently emptying the affected
/// set. Canonicalizing both sides keeps the prefix comparable.
///
/// A prereq that cannot be canonicalized (missing/deleted) falls back to a
/// lexical normalization; if that is still inside the repo it is KEPT (a
/// deleted in-repo source is a real change), otherwise dropped.
pub(crate) fn normalize_to_repo_relative(
    prereq: &Path,
    repo_root_canonical: &Path,
) -> Option<String> {
    let resolved =
        std::fs::canonicalize(prereq).unwrap_or_else(|_| lexically_normalize(prereq));
    let rel = resolved.strip_prefix(repo_root_canonical).ok()?;
    // Lossy is safe for scx (ASCII paths); the diff side uses the same
    // conversion so the two path spaces intersect consistently.
    Some(rel.to_string_lossy().into_owned())
}

/// Resolve `.`/`..` components without touching the filesystem, for a prereq
/// that no longer exists (so `canonicalize` would fail). Does NOT resolve
/// symlinks -- there is nothing on disk to resolve.
fn lexically_normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_rule_space_separated() {
        let d = "/repo/target/release/sched: /repo/src/main.rs /repo/src/bpf/main.bpf.c /usr/include/stdio.h";
        let got = parse_dep_info(d);
        assert_eq!(
            got,
            vec![
                PathBuf::from("/repo/src/main.rs"),
                PathBuf::from("/repo/src/bpf/main.bpf.c"),
                PathBuf::from("/usr/include/stdio.h"),
            ],
        );
    }

    #[test]
    fn parse_unescapes_spaces_in_paths() {
        // A path with a literal space is escaped `\ `; it must NOT split.
        let d = r"/repo/t: /repo/a\ b.rs /repo/c.rs";
        let got = parse_dep_info(d);
        assert_eq!(
            got,
            vec![PathBuf::from("/repo/a b.rs"), PathBuf::from("/repo/c.rs")],
        );
    }

    #[test]
    fn parse_joins_line_continuations() {
        let d = "/repo/t: /repo/a.rs \\\n    /repo/b.rs";
        let got = parse_dep_info(d);
        assert_eq!(
            got,
            vec![PathBuf::from("/repo/a.rs"), PathBuf::from("/repo/b.rs")],
        );
    }

    #[test]
    fn parse_skips_comments_and_phony_and_empty() {
        // `#`-comments, blank lines, and phony `<prereq>:` rules (empty RHS)
        // contribute nothing; only the real rule's prereqs survive.
        let d = "# a dep-info comment\n\n/repo/t: /repo/a.rs\n/repo/a.rs:\n";
        let got = parse_dep_info(d);
        assert_eq!(got, vec![PathBuf::from("/repo/a.rs")]);
    }

    #[test]
    fn normalize_resolves_symlinked_include_dir() {
        // Mirror the scx layout: bpf_h -> scheds/include, so a .d prereq under
        // bpf_h/ must resolve to the real repo source under scheds/include/.
        let tmp = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        std::fs::create_dir_all(root.join("scheds/include/scx")).unwrap();
        std::fs::write(root.join("scheds/include/scx/common.bpf.h"), "x").unwrap();
        std::os::unix::fs::symlink(root.join("scheds/include"), root.join("bpf_h")).unwrap();

        let via_symlink = root.join("bpf_h/scx/common.bpf.h");
        let rel = normalize_to_repo_relative(&via_symlink, &root)
            .expect("symlinked in-repo header resolves inside the repo");
        assert_eq!(rel, "scheds/include/scx/common.bpf.h");
    }

    #[test]
    fn normalize_drops_paths_outside_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        // A toolchain header outside the repo canonicalizes elsewhere -> None.
        assert!(normalize_to_repo_relative(Path::new("/usr/include/stdio.h"), &root).is_none());
    }

    #[test]
    fn normalize_keeps_deleted_in_repo_prereq_via_lexical_fallback() {
        // A prereq that no longer exists (canonicalize fails) but is lexically
        // inside the repo is KEPT -- a deleted in-repo source is a real change,
        // never silently dropped.
        let tmp = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        let deleted = root.join("src/gone.rs"); // never created
        let rel = normalize_to_repo_relative(&deleted, &root)
            .expect("deleted in-repo prereq is kept via the lexical fallback");
        assert_eq!(rel, "src/gone.rs");
    }

    #[test]
    fn lexically_normalize_collapses_dot_and_dotdot() {
        assert_eq!(
            lexically_normalize(Path::new("/repo/a/../b/./c.rs")),
            PathBuf::from("/repo/b/c.rs"),
        );
    }
}
