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
use std::collections::{BTreeMap, BTreeSet};
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

    // Snapshot the failed names BEFORE invoking nextest so we
    // can re-look them up in the post-exec pool. `failed_names`
    // is a `BTreeSet<&str>` borrowing from the pre-exec pool;
    // own the strings so the references survive the
    // post-exec re-scan that builds a fresh pool Vec.
    let queued: BTreeSet<String> = failed_names.iter().map(|s| s.to_string()).collect();

    let exit = invoke_nextest(&filter_expr).with_context(|| {
        format!("ktstr replay: cargo nextest run -E {filter_expr:?} failed to spawn")
    })?;

    // Post-exec outcome diff. Re-scan the sidecar pool so the
    // newly-written sidecars from the replay run reach the
    // classification. nextest's sidecar writes hit deterministic
    // paths per (test_name, topology, scheduler), so the new
    // contents overwrite the old contents in-place; the pool
    // re-collection reads the post-replay state.
    let post_pool = ktstr::test_support::collect_pool(&root);
    let queued_refs: BTreeSet<&str> = queued.iter().map(String::as_str).collect();
    let outcomes = classify_replay(&queued_refs, &post_pool);
    render_outcome_diff(&outcomes);

    Ok(exit)
}

/// Per-test outcome classification after a replay invocation.
/// Each test_name in the pre-exec failed set lands in exactly
/// one variant based on the post-exec sidecar pool — with one
/// crucial wrinkle: a test_name with multiple sidecars (one
/// per topology × scheduler variant) may have variants that
/// disagree. The classifier surfaces that disagreement as
/// [`Self::Mixed`] rather than silently collapsing to "any
/// variant passed = fixed" — silent collapse would let a
/// half-broken parameterized test report green, which is
/// the silent-drop failure mode this surface exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReplayOutcome {
    /// EVERY post-replay sidecar for the test_name reports
    /// `passed=true && !skipped`. Either the fix landed
    /// between the original run and this replay, or the test
    /// was passing intermittently (flake) and now passes.
    /// Operator sees a green signal here.
    Fixed,
    /// EVERY post-replay sidecar for the test_name reports
    /// `!passed || skipped`. The load-bearing case the replay
    /// command exists to surface: a regression that survived
    /// the operator's change.
    Persistent,
    /// Test_name has no post-replay sidecar. Three plausible
    /// causes the operator should triage in order:
    /// - The test was removed from the suite between runs.
    /// - The `--filter` narrowed past it.
    /// - nextest crashed before reaching the test.
    /// The classifier itself doesn't distinguish — the inline
    /// triage hint in [`render_outcome_diff`] surfaces all
    /// three causes for the operator.
    Dropped,
    /// Test_name has multiple post-replay sidecars and they
    /// DISAGREE — at least one passed and at least one failed.
    /// Common when a parameterized test runs across topology
    /// variants and only some variants reproduce the
    /// regression. The operator MUST drill in to see which
    /// variant is still red; surfacing this as `Fixed` would
    /// silently hide the failing variant.
    ///
    /// Never collapse variant disagreement — that would silently
    /// hide a failing variant behind a passing one.
    Mixed {
        /// Count of post-replay sidecars for this test_name
        /// that passed.
        fixed_count: usize,
        /// Count of post-replay sidecars for this test_name
        /// that failed or skipped.
        persistent_count: usize,
    },
}


/// Classify the replay outcome for each test_name in `queued`
/// against the post-replay sidecar pool. Returns a BTreeMap so
/// the renderer iterates in deterministic ascending order.
///
/// Lookup strategy: group `post_pool` by test_name into a
/// `BTreeMap<&str, Vec<&SidecarResult>>` so EVERY topology ×
/// scheduler variant for a name is visible. Then for each
/// queued name:
/// - Empty group → [`ReplayOutcome::Dropped`]
/// - All variants passed && !skipped → [`ReplayOutcome::Fixed`]
/// - All variants failed/skipped → [`ReplayOutcome::Persistent`]
/// - Variants disagree → [`ReplayOutcome::Mixed`] carrying
///   per-side counts so the operator sees the disagreement
///   instead of an erroneously-green verdict.
///
/// The Mixed case is the load-bearing addition over a naive
/// "any-variant-fixed" semantic: collapsing variant
/// disagreement to Fixed is a silent drop — a parameterized
/// test where variant A is fixed and variant B is still red
/// would silently report green.
pub(crate) fn classify_replay<'a>(
    queued: &'a BTreeSet<&'a str>,
    post_pool: &'a [ktstr::test_support::SidecarResult],
) -> BTreeMap<&'a str, ReplayOutcome> {
    let mut by_name: BTreeMap<&str, Vec<&ktstr::test_support::SidecarResult>> = BTreeMap::new();
    for sc in post_pool {
        by_name.entry(sc.test_name.as_str()).or_default().push(sc);
    }
    queued
        .iter()
        .map(|name| {
            let outcome = match by_name.get(name) {
                None => ReplayOutcome::Dropped,
                Some(variants) => {
                    let fixed_count = variants
                        .iter()
                        .filter(|sc| sc.passed && !sc.skipped)
                        .count();
                    let persistent_count = variants.len() - fixed_count;
                    match (fixed_count, persistent_count) {
                        (n, 0) if n > 0 => ReplayOutcome::Fixed,
                        (0, n) if n > 0 => ReplayOutcome::Persistent,
                        _ => ReplayOutcome::Mixed {
                            fixed_count,
                            persistent_count,
                        },
                    }
                }
            };
            (*name, outcome)
        })
        .collect()
}

/// Render the outcome-diff summary to stderr (the narrative
/// stream — stdout stays clean for the dry-run filter path,
/// which is the primary pipeable surface). Header line carries
/// the counts; per-test lines name each PERSISTENT/DROPPED
/// entry so the operator can drill in without parsing nextest
/// output. FIXED entries are aggregated to a count only to
/// keep the diff short on healthy days; the operator who wants
/// per-test FIXED detail can grep the live nextest output above.
fn render_outcome_diff(outcomes: &BTreeMap<&str, ReplayOutcome>) {
    let (mut fixed, mut persistent, mut dropped, mut mixed) = (0usize, 0usize, 0usize, 0usize);
    for o in outcomes.values() {
        match o {
            ReplayOutcome::Fixed => fixed += 1,
            ReplayOutcome::Persistent => persistent += 1,
            ReplayOutcome::Dropped => dropped += 1,
            ReplayOutcome::Mixed { .. } => mixed += 1,
        }
    }
    eprintln!();
    eprintln!(
        "ktstr replay: {fixed} FIXED, {persistent} PERSISTENT, {mixed} MIXED, {dropped} DROPPED",
    );
    if persistent > 0 || dropped > 0 || mixed > 0 {
        for (name, outcome) in outcomes {
            match outcome {
                ReplayOutcome::Persistent => {
                    eprintln!("  PERSISTENT {name}");
                }
                ReplayOutcome::Dropped => {
                    // Triage hint: the operator sees DROPPED
                    // with zero context without this. Listing
                    // the 3 plausible causes inline turns an
                    // opaque verdict into actionable diagnosis
                    // (test removed is the most common cause,
                    // --filter narrowed second-most-common,
                    // nextest crash a distant third).
                    eprintln!(
                        "  DROPPED {name} \
                         (not run — test removed, --filter narrowed past, \
                         or nextest skipped/crashed before reaching it)",
                    );
                }
                ReplayOutcome::Mixed {
                    fixed_count,
                    persistent_count,
                } => {
                    // Surface variant disagreement explicitly:
                    // a parameterized test with some variants
                    // fixed and some still red must NOT collapse
                    // to a green Fixed verdict.
                    eprintln!(
                        "  MIXED {name} \
                         ({fixed_count} variant(s) fixed, \
                         {persistent_count} variant(s) still failing — \
                         drill into the per-variant sidecars to triage)",
                    );
                }
                ReplayOutcome::Fixed => {}
            }
        }
    }
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

    // -- classify_replay (phase 3 outcome diff) --

    /// Test was failing in pre-replay; post-replay sidecar
    /// reports passed=true → FIXED. Pins the primary "happy
    /// path" outcome — the operator fixed the bug between runs.
    #[test]
    fn classify_replay_failing_then_passing_classifies_as_fixed() {
        let post_pool = synth_pool(&[("test_fix_me", true, false)]);
        let queued: BTreeSet<&str> = ["test_fix_me"].iter().copied().collect();
        let outcomes = classify_replay(&queued, &post_pool);
        assert_eq!(outcomes.get("test_fix_me"), Some(&ReplayOutcome::Fixed));
    }

    /// Test was failing in pre-replay; post-replay sidecar
    /// still reports passed=false → PERSISTENT. The
    /// load-bearing case the replay command exists to surface:
    /// the operator's fix didn't take.
    #[test]
    fn classify_replay_still_failing_classifies_as_persistent() {
        let post_pool = synth_pool(&[("test_still_broken", false, false)]);
        let queued: BTreeSet<&str> = ["test_still_broken"].iter().copied().collect();
        let outcomes = classify_replay(&queued, &post_pool);
        assert_eq!(
            outcomes.get("test_still_broken"),
            Some(&ReplayOutcome::Persistent)
        );
    }

    /// Test was failing in pre-replay but missing from
    /// post-pool → DROPPED. Triage candidates: test removed,
    /// --filter narrowed past, nextest crashed before
    /// reaching it. The classifier doesn't distinguish; the
    /// operator triages from context.
    #[test]
    fn classify_replay_missing_from_post_pool_classifies_as_dropped() {
        let post_pool = synth_pool(&[("unrelated_test", true, false)]);
        let queued: BTreeSet<&str> = ["test_was_removed"].iter().copied().collect();
        let outcomes = classify_replay(&queued, &post_pool);
        assert_eq!(
            outcomes.get("test_was_removed"),
            Some(&ReplayOutcome::Dropped)
        );
    }

    /// Mixed pool exercises all three branches in one call:
    /// FIXED + PERSISTENT + DROPPED in deterministic
    /// ascending order via the BTreeMap iteration. Pins both
    /// the per-test classification AND the multi-test
    /// orchestration through `classify_replay`.
    #[test]
    fn classify_replay_mixed_outcomes_classifies_each_correctly() {
        let post_pool = synth_pool(&[
            ("test_a_fixed", true, false),
            ("test_b_persistent", false, false),
            // test_c_dropped absent from post_pool by design
            ("unrelated_pass", true, false),
        ]);
        let queued: BTreeSet<&str> = ["test_a_fixed", "test_b_persistent", "test_c_dropped"]
            .iter()
            .copied()
            .collect();
        let outcomes = classify_replay(&queued, &post_pool);
        assert_eq!(outcomes.len(), 3, "every queued name gets exactly one outcome");
        assert_eq!(outcomes.get("test_a_fixed"), Some(&ReplayOutcome::Fixed));
        assert_eq!(
            outcomes.get("test_b_persistent"),
            Some(&ReplayOutcome::Persistent)
        );
        assert_eq!(
            outcomes.get("test_c_dropped"),
            Some(&ReplayOutcome::Dropped)
        );
    }

    /// Post-replay sidecar that reports `passed=true skipped=true`
    /// (the SidecarResult "skipped" convention) classifies as
    /// PERSISTENT, NOT Fixed. A skipped re-run is not a pass —
    /// the test didn't actually run, so the original failure
    /// isn't validated as fixed. Pin the `&& !sc.skipped` guard
    /// in classify_replay against a future regression that
    /// treats skipped as passed.
    #[test]
    fn classify_replay_post_skipped_is_persistent_not_fixed() {
        let post_pool = synth_pool(&[("test_skipped", true, true)]);
        let queued: BTreeSet<&str> = ["test_skipped"].iter().copied().collect();
        let outcomes = classify_replay(&queued, &post_pool);
        assert_eq!(
            outcomes.get("test_skipped"),
            Some(&ReplayOutcome::Persistent),
            "post-replay skipped means the original failure is unvalidated; \
             classifier must NOT treat skip as Fixed"
        );
    }

    /// Test_name with 2 post-replay sidecars (variants) where
    /// one passed and one failed → Mixed. The classifier must
    /// NOT silently collapse variant disagreement to Fixed.
    /// Surfacing as Mixed lets the operator drill in instead
    /// of mistakenly closing the bug.
    #[test]
    fn classify_replay_mixed_variants_classifies_as_mixed() {
        let post_pool = synth_pool(&[
            ("test_param", true, false),  // variant A: fixed
            ("test_param", false, false), // variant B: still failing
        ]);
        let queued: BTreeSet<&str> = ["test_param"].iter().copied().collect();
        let outcomes = classify_replay(&queued, &post_pool);
        assert_eq!(
            outcomes.get("test_param"),
            Some(&ReplayOutcome::Mixed {
                fixed_count: 1,
                persistent_count: 1,
            }),
            "variant disagreement must surface as Mixed; \
             silent collapse to Fixed would hide the failing variant"
        );
    }

    /// All variants pass → Fixed (not Mixed). Pins the
    /// "every sidecar passed" branch of the new variant-aware
    /// classifier against the all-pass case.
    #[test]
    fn classify_replay_all_variants_pass_classifies_as_fixed() {
        let post_pool = synth_pool(&[
            ("test_consistent", true, false),
            ("test_consistent", true, false),
            ("test_consistent", true, false),
        ]);
        let queued: BTreeSet<&str> = ["test_consistent"].iter().copied().collect();
        let outcomes = classify_replay(&queued, &post_pool);
        assert_eq!(
            outcomes.get("test_consistent"),
            Some(&ReplayOutcome::Fixed),
            "3-of-3 variants passed → Fixed, not Mixed"
        );
    }

    /// All variants fail → Persistent. Pins the "every sidecar
    /// failed" branch — a parameterized test that's broken on
    /// every variant should not silently become Mixed.
    #[test]
    fn classify_replay_all_variants_fail_classifies_as_persistent() {
        let post_pool = synth_pool(&[
            ("test_broken", false, false),
            ("test_broken", false, false),
        ]);
        let queued: BTreeSet<&str> = ["test_broken"].iter().copied().collect();
        let outcomes = classify_replay(&queued, &post_pool);
        assert_eq!(
            outcomes.get("test_broken"),
            Some(&ReplayOutcome::Persistent),
            "2-of-2 variants failed → Persistent, not Mixed"
        );
    }
}
