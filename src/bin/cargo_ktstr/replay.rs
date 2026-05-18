//! `cargo ktstr replay` — re-run the failing subset of a prior
//! sidecar pool.
//!
//! Reads a sidecar root (defaults to `target/ktstr/`), selects every
//! sidecar whose run failed and was not a skip, dedupes the
//! resulting test names, and emits a `cargo nextest run`-compatible
//! filter expression that targets exactly that subset. With
//! `--exec`, also invokes nextest and waits for it; default is
//! dry-run so an operator can paste the filter expression into CI
//! or refine it by hand before committing to a re-run.
//!
//! ## Why this lives outside the existing in-VM auto-repro
//!
//! ktstr's in-VM auto-repro (`attempt_auto_repro` at
//! `src/test_support/probe.rs`) handles within-process retry:
//! when a primary VM run fails AND `auto_repro = true` on the
//! `KtstrTestEntry`, the harness boots a second VM with probe BPF
//! programs attached and stitches the probe events into the
//! failure narrative. That covers the "literal scenario replay on
//! the same test process" use case.
//!
//! `cargo ktstr replay` covers a DIFFERENT use case: post-hoc
//! subset re-run after the test process has exited. A test author
//! runs the suite, gets failures, fixes code, and wants to re-run
//! only the failed tests against the new code on a different host
//! or commit. The sidecar pool already carries the per-test
//! outcome; this command turns that pool into a nextest filter
//! expression and (optionally) invokes nextest with it.
//!
//! ## Sidecar selection
//!
//! The pool is loaded via
//! [`ktstr::test_support::collect_pool`], which walks the
//! sidecar root + one level of per-job subdirectories. Sidecars
//! are filtered by `!passed && !skipped` — a skipped run is not
//! a failure (the test refused to execute), and a passed run is
//! not a candidate for re-running. The `!skipped` clause is
//! defensive: per the SidecarResult contract at
//! `src/test_support/sidecar/mod.rs:179-185`, a skipped sidecar
//! always carries `passed = true`, so `!passed` alone already
//! excludes skips; keeping `!skipped` guards against a future
//! contract change that decouples those two fields.
//!
//! ## Filter expression shape
//!
//! Multiple sidecars per test_name (one per topology variant, one
//! per scheduler) fold into a single nextest filter entry via
//! [`std::collections::BTreeSet`]. The output filter uses
//! nextest's `test(/regex/)` form anchored at end-of-identifier:
//! `test(/^(.*::)?NAME$/) | test(/^(.*::)?NAME2$/)`.
//!
//! Why the regex form rather than the simpler `test(=NAME)` or
//! `test(NAME)`:
//! - `test(=NAME)` matches the FULL nextest identifier
//!   (`<binary_id>::<path>::<test_name>`). SidecarResult.test_name
//!   stores only the bare function name (per sidecar/mod.rs:107),
//!   so the equality match never fires against production tests.
//! - `test(NAME)` is a substring match — would shadow if one
//!   test name is a substring of another (e.g.
//!   `phase_pipeline_two_step_e2e` vs
//!   `phase_pipeline_no_periodic_samples_yields_empty_phases`
//!   share the `phase_pipeline_` prefix).
//! - `test(/^(.*::)?NAME$/)` matches the bare name as the
//!   terminal component of any nextest path, with the `$` anchor
//!   preventing substring shadowing. The optional `(.*::)?`
//!   prefix tolerates both `binary::name` and
//!   `binary::module::name` shapes nextest emits.
//!
//! Empty selection (no failures in the pool) prints a
//! pipeline-safe no-op expression (`test(/^__ktstr_no_failures_to_replay__$/)`)
//! to stdout so a CI pipeline of
//! `cargo ktstr replay | cargo nextest run -E -` parses cleanly
//! and runs zero tests instead of erroring on empty stdin. A
//! stderr line explains the no-op.

use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Sentinel filter expression emitted when the pool has no
/// failures to replay. nextest parses it cleanly and matches
/// zero tests; the leading double underscores make collision
/// with any real test name astronomically unlikely.
const EMPTY_POOL_FILTER: &str = "test(/^__ktstr_no_failures_to_replay__$/)";

/// Entry point for the `cargo ktstr replay` subcommand.
///
/// `dir` overrides the sidecar root (default:
/// `ktstr::test_support::runs_root()`). `filter` narrows the
/// failed-sidecar selection to test_names matching the given
/// substring (case-sensitive). `exec` invokes nextest with the
/// computed filter; otherwise the filter is printed and the
/// caller can pipe it into nextest themselves.
///
/// Returns `Ok(0)` on a clean dry-run or successful exec.
/// Returns `Ok(N)` with nextest's exit code when `exec` is set
/// and nextest exits non-zero. Returns `Err` only for genuine
/// errors (unreadable sidecar root, nextest spawn failure).
pub(crate) fn run_replay(
    dir: Option<&Path>,
    filter: Option<&str>,
    exec: bool,
) -> Result<i32> {
    let root: PathBuf = dir
        .map(Path::to_path_buf)
        .unwrap_or_else(ktstr::test_support::runs_root);

    let pool = ktstr::test_support::collect_pool(&root);
    if pool.is_empty() {
        anyhow::bail!(
            "ktstr replay: no sidecars found under {} — \
             run the suite first (cargo ktstr test) or pass \
             --dir <DIR> to point at an archived pool",
            root.display(),
        );
    }

    let failed_names = select_failed_names(&pool, filter);

    if failed_names.is_empty() {
        eprintln!(
            "ktstr replay: no failed sidecars in pool at {} \
             (filter: {:?}) — nothing to re-run",
            root.display(),
            filter,
        );
        // Emit a pipeline-safe no-op filter so a CI script
        // doing `cargo ktstr replay | cargo nextest run -E -`
        // doesn't fail on empty stdin or on a malformed expr.
        // The sentinel matches zero tests by construction —
        // double-underscore prefix collides with no real test
        // name in the registry.
        println!("{EMPTY_POOL_FILTER}");
        return Ok(0);
    }

    let filter_expr = build_nextest_filter(&failed_names);

    if !exec {
        // Dry-run: print the computed filter expression so the
        // operator can paste it into their own nextest invocation
        // (or CI pipeline). The dry-run default matches the unix-
        // tool tradition of "show me what you'd do before doing
        // it" (cf. `rm -i`, `git push --dry-run`).
        println!("{filter_expr}");
        eprintln!(
            "ktstr replay: {} failed test name(s) selected. \
             Pipe the printed filter into `cargo nextest run -E` \
             or re-run with --exec to invoke nextest directly.",
            failed_names.len(),
        );
        return Ok(0);
    }

    invoke_nextest(&filter_expr).with_context(|| {
        format!("ktstr replay: cargo nextest run -E {filter_expr:?} failed to spawn")
    })
}

/// Select the set of test_names from `pool` whose sidecars
/// represent real failures (`!passed && !skipped`), optionally
/// narrowed by a substring filter on test_name. Returns a
/// BTreeSet for deterministic ascending-order iteration when
/// the renderer builds the nextest filter expression.
///
/// Extracted as a pub(crate) free function so the scan-path
/// logic can be unit-tested against synthetic SidecarResult
/// fixtures without instantiating a real sidecar pool on
/// disk. Mirrors the closure-extraction pattern used elsewhere
/// for `matches_phase` / `passes_delta_threshold`.
pub(crate) fn select_failed_names<'a>(
    pool: &'a [ktstr::test_support::SidecarResult],
    filter: Option<&str>,
) -> BTreeSet<&'a str> {
    pool.iter()
        .filter(|s| !s.passed && !s.skipped)
        .map(|s| s.test_name.as_str())
        .filter(|n| match filter {
            Some(f) => n.contains(f),
            None => true,
        })
        .collect()
}

/// Format a `BTreeSet<&str>` of test names as a nextest filter
/// expression using the regex `test(/^(.*::)?NAME$/)` form.
/// See the module-level "Filter expression shape" section for
/// the rationale behind the regex form (over `test(=NAME)` or
/// bare `test(NAME)`). Empty set is rejected by the caller
/// before reaching this fn — callers emit
/// `EMPTY_POOL_FILTER` instead so the downstream nextest
/// invocation has a parseable input.
fn build_nextest_filter(names: &BTreeSet<&str>) -> String {
    let parts: Vec<String> = names
        .iter()
        .map(|n| format!("test(/^(.*::)?{}$/)", regex_escape(n)))
        .collect();
    parts.join(" | ")
}

/// Escape regex metacharacters in a test name. Conservative
/// escape set covers every metacharacter the `regex` crate
/// honors that could appear in a Rust identifier-shaped test
/// name (mostly defensive — Rust identifiers can't contain
/// `.`, `*`, `+`, `?`, `(`, `)`, `[`, `]`, `{`, `}`, `^`, `$`,
/// `|`, `\\` — but a future `#[ktstr_test(name = "literal")]`
/// attribute could permit any string).
fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|'
            | '\\' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

/// Invoke `cargo nextest run -E '<filter>'` and forward its exit
/// code. Inherits stdout/stderr so the operator sees nextest's
/// live progress. Returns the nextest exit code; an `Err` here
/// is only for spawn failure (nextest binary missing,
/// `Command::status()` failed at the syscall level).
fn invoke_nextest(filter_expr: &str) -> Result<i32> {
    use std::process::Command;
    let status = Command::new("cargo")
        .args(["nextest", "run", "-E", filter_expr])
        .status()
        .context("spawn `cargo nextest run`")?;
    Ok(status.code().unwrap_or(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktstr::test_support::SidecarResult;

    /// Build a minimal SidecarResult fixture for the scan-path
    /// tests. Only fields the selector consults (test_name,
    /// passed, skipped) are meaningful; the rest are
    /// placeholders that satisfy the struct.
    fn synth_sidecar(test_name: &str, passed: bool, skipped: bool) -> SidecarResult {
        SidecarResult {
            test_name: test_name.to_string(),
            topology: "synth".into(),
            scheduler: "synth".into(),
            scheduler_commit: None,
            project_commit: None,
            payload: None,
            metrics: Vec::new(),
            passed,
            skipped,
            stats: ktstr::assert::ScenarioStats::default(),
            monitor: None,
            stimulus_events: Vec::new(),
            work_type: "synth".into(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            sysctls: Vec::new(),
            kargs: Vec::new(),
            kernel_version: None,
            kernel_commit: None,
            timestamp: "synth".into(),
            run_id: "synth".into(),
            host: None,
            cleanup_duration_ms: None,
            run_source: None,
        }
    }

    fn synth_pool(rows: &[(&str, bool, bool)]) -> Vec<SidecarResult> {
        rows.iter()
            .map(|(n, p, s)| synth_sidecar(n, *p, *s))
            .collect()
    }

    // -- build_nextest_filter (formatting layer) --

    #[test]
    fn build_nextest_filter_single_name_emits_regex_anchored_form() {
        let mut names = BTreeSet::new();
        names.insert("scheduler_smoke_test");
        let expr = build_nextest_filter(&names);
        assert_eq!(
            expr,
            "test(/^(.*::)?scheduler_smoke_test$/)",
            "single-name filter wraps in regex with optional path prefix + end anchor"
        );
    }

    /// Multiple names produce a `|`-joined expression in
    /// BTreeSet ascending order — pins deterministic ordering
    /// so the same input always produces the same filter
    /// string (CI scripts can diff the output across runs).
    #[test]
    fn build_nextest_filter_multiple_names_sorted_and_joined() {
        let mut names = BTreeSet::new();
        names.insert("z_test");
        names.insert("a_test");
        names.insert("m_test");
        let expr = build_nextest_filter(&names);
        assert_eq!(
            expr,
            "test(/^(.*::)?a_test$/) | test(/^(.*::)?m_test$/) | test(/^(.*::)?z_test$/)"
        );
    }

    /// Names that are substrings of each other both appear in
    /// the filter — the `$` end-anchor guarantees they don't
    /// shadow each other. Pins the substring-safety contract
    /// that motivates the regex form over plain `test(name)`
    /// (which matches by substring).
    #[test]
    fn build_nextest_filter_substring_names_both_present() {
        let mut names = BTreeSet::new();
        names.insert("phase_pipeline_two_step_e2e");
        names.insert("phase_pipeline_no_periodic_samples_yields_empty_phases");
        let expr = build_nextest_filter(&names);
        assert!(
            expr.contains("phase_pipeline_two_step_e2e$"),
            "two_step_e2e present with end anchor"
        );
        assert!(
            expr.contains("phase_pipeline_no_periodic_samples_yields_empty_phases$"),
            "no_periodic_samples present with end anchor"
        );
        assert_eq!(
            expr.matches(" | ").count(),
            1,
            "BTreeSet dedups; exactly one `|` between the two entries"
        );
    }

    // -- regex_escape (defensive against future hand-named tests) --

    #[test]
    fn regex_escape_passes_through_identifier_chars() {
        assert_eq!(regex_escape("phase_pipeline_e2e"), "phase_pipeline_e2e");
        assert_eq!(regex_escape("test123"), "test123");
    }

    #[test]
    fn regex_escape_escapes_metacharacters() {
        // Defensive — Rust idents can't contain these, but
        // `#[ktstr_test(name = "...")]` literal names could.
        assert_eq!(regex_escape("a.b"), "a\\.b");
        assert_eq!(regex_escape("(group)"), "\\(group\\)");
        assert_eq!(regex_escape("a|b"), "a\\|b");
        assert_eq!(regex_escape("end$"), "end\\$");
    }

    // -- select_failed_names (scan-path selector) --

    /// Pool with mixed states — only `!passed && !skipped`
    /// rows are selected. Pins the load-bearing failed-sidecar
    /// selector logic so a skipped row never collapses into the
    /// failed set.
    #[test]
    fn select_failed_skips_passed_and_skipped_keeps_only_real_failures() {
        let pool = synth_pool(&[
            ("test_pass", true, false),     // not selected (passed)
            ("test_skip", true, true),      // not selected (passed+skipped)
            ("test_fail1", false, false),   // SELECTED
            ("test_fail2", false, false),   // SELECTED
            ("test_corner", false, true),   // not selected — !passed but skipped
        ]);
        let result = select_failed_names(&pool, None);
        let expected: BTreeSet<&str> =
            ["test_fail1", "test_fail2"].iter().copied().collect();
        assert_eq!(result, expected);
    }

    /// Filter narrows by substring match on test_name. Pins
    /// the substring-match behavior of `--filter / -E` against
    /// a regression to exact-match (which would silently drop
    /// all filter-using invocations).
    #[test]
    fn select_failed_with_filter_substring_match_keeps_matching_failures() {
        let pool = synth_pool(&[
            ("scheduler_smoke_a", false, false),
            ("scheduler_smoke_b", false, false),
            ("workload_perf", false, false),
        ]);
        let result = select_failed_names(&pool, Some("scheduler_"));
        let expected: BTreeSet<&str> = ["scheduler_smoke_a", "scheduler_smoke_b"]
            .iter()
            .copied()
            .collect();
        assert_eq!(result, expected);
    }

    /// Filter that matches zero test_names yields an empty set
    /// — caller decides the empty-pool semantic, the selector
    /// doesn't synthesize.
    #[test]
    fn select_failed_with_filter_no_match_returns_empty_set() {
        let pool = synth_pool(&[
            ("test_pass", true, false),
            ("test_fail", false, false),
        ]);
        let result = select_failed_names(&pool, Some("nonexistent"));
        assert!(result.is_empty());
    }

    /// The `!passed && !skipped` condition (vs `!passed` alone)
    /// — failed+skipped is the awkward intermediate state.
    /// Pins that this state is EXCLUDED per the explicit
    /// `&& !skipped` clause; if a future refactor changes it
    /// to `!passed` alone, this test surfaces the change.
    #[test]
    fn select_failed_corner_case_failed_and_skipped_excluded() {
        let pool = synth_pool(&[("test_fail_skip", false, true)]);
        let result = select_failed_names(&pool, None);
        assert!(
            result.is_empty(),
            "failed+skipped must be excluded; the && !skipped guard is load-bearing"
        );
    }
}
