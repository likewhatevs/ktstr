//! Regression pin: forbid hardcoded `<fn_name>` string-literal args to
//! sidecar-path helpers in any tests/*.rs file, AND forbid raw
//! `sidecar_dir().join("<literal>.wprof.pb")`-style path-assembly
//! that bypasses the [`VmResult`] method form.
//!
//! ## The drift class
//!
//! The legacy pattern
//!
//! ```ignore
//! fn assert_wprof_pb_landed(result: &VmResult) -> Result<()> {
//!     let pb = wprof_pb_path("my_test_fn_name");
//!     assert_wprof_pb_shape(&pb)
//! }
//! ```
//!
//! silently drifts when `my_test_fn_name` is renamed: the macro
//! stamps the new name into the sidecar path, but the callback
//! reads the OLD literal → ENOENT → spurious "wprof chain broken"
//! diagnostic that's actually a name-drift bug. The structural fix
//! moved path derivation onto
//! [`VmResult::wprof_pb_path`](ktstr::vmm::VmResult::wprof_pb_path),
//! which reads the macro-stamped
//! [`VmResult::entry_name`](ktstr::vmm::VmResult::entry_name) at
//! call time — no literal to drift.
//!
//! ## Two vectors to defend against
//!
//! 1. **Function-arg literal**: `wprof_pb_path("foo")` /
//!    `failure_dump_path("foo")` — direct literal as the path
//!    helper's argument.
//! 2. **Path-assembly literal**: `sidecar_dir().join("foo.wprof.pb")`
//!    — bypasses the helper entirely; the fn name lives in the
//!    suffix-bearing string.
//!
//! Vector 2 is the auto-repro fault-injection pattern
//! (`wprof_auto_repro_e2e.rs` body fn that hardcodes
//! `"neg_fixture_name.repro.wprof.pb"` for the operator diagnostic
//! string). Body fns get `&Ctx` not `&VmResult`, so the method-form
//! migration that fixes vector 1 doesn't apply directly to vector 2;
//! the regression pin catches both vectors so a future migration
//! covers the body-callsite class via the same enforcement.
//!
//! ## EXEMPT_FILES
//!
//! Files KNOWN to violate the patterns today and tracked under the
//! follow-up migration are listed in [`EXEMPT_FILES`] below. The
//! exempt list shrinks as the body-callsite migration lands; each
//! removal from the list is the structural pin that the migration
//! actually fixed the site rather than just suppressing the
//! regression test.

#![cfg(unix)]

use std::path::Path;

/// File the regression pin lives in — exempted via top-level path
/// match (not bare filename) to prevent a future
/// `tests/<subdir>/no_hardcoded_dump_path_literals.rs` from also
/// being exempted by accident.
const SELF_FILE: &str = "no_hardcoded_dump_path_literals.rs";

/// Files KNOWN to contain forbidden literal patterns today, tracked
/// under follow-up migrations. Each entry is the path relative to
/// `tests/` (e.g. `wprof_auto_repro_e2e.rs`). The exempt list
/// shrinks as migrations land; an entry that no longer contains a
/// violation can be safely removed — at which point this test
/// guards that the site stays clean.
///
/// Current entries: none.
///
/// All test-body-scope callsites have migrated to the
/// `ctx.failure_dump_path()` / `ctx.wprof_pb_path()` /
/// `ctx.repro_wprof_pb_path()` method form. The duplicate
/// `fn failure_dump_path` definitions in `cast_analysis_e2e.rs` and
/// `vm_integration.rs` and the sibling `fn watch_snapshot_dump_path`
/// in `snapshot_e2e.rs` are deleted; the path-assembly bypass in
/// `wprof_auto_repro_e2e.rs` is gone; the `common/dump_paths.rs`
/// helper module is gone too (zero remaining consumers after the
/// callsite migration, and the format-string single-source-of-truth
/// now lives in the body of `Ctx::failure_dump_path` and
/// `VmResult::failure_dump_path`).
///
/// The empty exempt list makes the FORBIDDEN_PATTERNS sweep
/// load-bearing for every test file in the tree — a future attempt
/// to reintroduce a `failure_dump_path("literal")` callsite or
/// a `sidecar_dir().join("<fn>.failure-dump.json")` path-assembly
/// trips the meta-grep test rather than silently shipping a
/// drift-fragile literal.
const EXEMPT_FILES: &[&str] = &[];

/// Forbidden literal-arg + path-assembly patterns.
const FORBIDDEN_PATTERNS: &[&str] = &[
    // Vector 1: helper-fn literal arg.
    "wprof_pb_path(\"",
    "failure_dump_path(\"",
    // Vector 2: path-assembly with the suffix bearing the literal.
    // `sidecar_dir().join("<...>.wprof.pb")` — catches the
    // wprof_auto_repro_e2e.rs body-callsite pattern. Excludes the
    // intentional patterns above (which contain `.wprof.pb"` as a
    // substring of the helper name, but always preceded by an open
    // paren — the literal-suffix check below requires the `.pb"`
    // pattern at the very end of a string literal, which the
    // helper-name patterns above never produce).
    ".wprof.pb\"",
    ".repro.wprof.pb\"",
    ".failure-dump.json\"",
    ".repro.failure-dump.json\"",
];

#[test]
fn no_hardcoded_dump_path_literals_in_tests() {
    let tests_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut violations: Vec<String> = Vec::new();
    for entry in walkdir::WalkDir::new(&tests_dir)
        .into_iter()
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        // Top-level path match (not bare filename) so a future
        // `tests/<subdir>/no_hardcoded_dump_path_literals.rs` does
        // not auto-exempt.
        let rel = match path.strip_prefix(&tests_dir) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let rel_str = match rel.to_str() {
            Some(s) => s,
            None => continue,
        };
        if rel_str == SELF_FILE {
            continue;
        }
        if EXEMPT_FILES.contains(&rel_str) {
            continue;
        }
        let content = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for (i, line) in content.lines().enumerate() {
            for pat in FORBIDDEN_PATTERNS {
                if line.contains(pat) {
                    violations.push(format!("{}:{}  {}", rel_str, i + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "hardcoded fn-name literal arg or path-assembly suffix detected in {} \
         site(s) — use the drift-safe method form `result.wprof_pb_path()` / \
         `result.repro_wprof_pb_path()` / `result.failure_dump_path()` on \
         `&VmResult` (or `result.assert_wprof_pb_landed()` for the \
         assertion+shape-check wrapper). The path derives from the \
         macro-stamped entry name, so a future test fn rename surfaces as \
         a compile-time `None` bail rather than a runtime ENOENT against a \
         stale literal. If the violation site is a body fn that has no \
         `&VmResult` in scope (test-body-callsite class), add it to \
         EXEMPT_FILES at the head of this file as a follow-up migration \
         site.\n\nViolations:\n{}",
        violations.len(),
        violations.join("\n"),
    );
}

/// Pins [`EXEMPT_FILES`] empty post-migration. A future entry
/// added without an explicit follow-up migration plan trips this
/// guard — the empty-list shape is what makes
/// [`FORBIDDEN_PATTERNS`] load-bearing for every test file in
/// the tree.
///
/// A regression that reintroduces an exempt-listed test file
/// (e.g. someone adds a literal-arg callsite back, then exempts
/// the file to silence the regression pin) flips this assertion
/// from PASS to FAIL with an actionable message naming the new
/// exemption — the only legitimate way past this gate is to
/// migrate the underlying callsite to the drift-safe form OR
/// extend the list with a focused doc-comment explaining why the
/// exemption is acceptable, both of which a reviewer would
/// see in the pull request diff for this file.
#[test]
fn exempt_files_is_empty_post_migration() {
    assert!(
        EXEMPT_FILES.is_empty(),
        "EXEMPT_FILES must stay empty post-migration; an entry \
         added without a follow-up migration plan re-opens the \
         literal-drift class for that file. Current entries: {:?}",
        EXEMPT_FILES,
    );
}
