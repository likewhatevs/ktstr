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
//! sidecar root + one level of per-job subdirectories.
//! [`select_failed_names`] filters on
//! [`ktstr::test_support::SidecarResult::is_fail`], which
//! evaluates `!passed && !skipped && !inconclusive` — Pass,
//! Skip, and Inconclusive are all excluded from the replay
//! queue. Inconclusive runs are excluded because a
//! zero-denominator gate produced no signal, so re-running the
//! same scenario would just reproduce the non-measurement (see
//! the function-level doc on `select_failed_names` for the full
//! rationale). The four mutually-exclusive states `(passed,
//! skipped, inconclusive, fail)` — three stored bits plus the
//! derived all-false Fail state — are pinned by the SidecarResult
//! contract; `write_skip_sidecar` emits a Skip row as
//! `passed=false, skipped=true, inconclusive=false`, so a Skip
//! is correctly excluded by both the `!skipped` and `!passed`
//! conjuncts of `is_fail`.
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
//!
//! ## Host-context attachment
//!
//! Every render path also surfaces a host-context section so the
//! operator knows what host the failures were captured on and
//! how that compares to the host running the replay. The section
//! has three shapes:
//!
//! - `(no host context captured)` — the pool predates host
//!   capture, or every collection failed. Operator gets the hint
//!   that the comparison is not available.
//! - `(host context unchanged since capture)` — captured host
//!   matches the current host field-for-field. Reassures the
//!   operator that the replay environment matches the capture
//!   environment; reproduction failure cannot be blamed on host
//!   drift.
//! - `host context drift since capture:` followed by the per-field
//!   diff body (`key: before → after` lines). The load-bearing case
//!   — environment changes between capture and replay are exactly
//!   what would silently cause a "persistent" failure to become
//!   "fixed" or vice versa.
//!
//! The host extraction policy is "first sidecar with `host =
//! Some(_)`". Pool iteration order is deterministic (collect_pool
//! walks dirs in `read_dir` order, which the kernel returns in
//! arbitrary-but-fixed order per inode layout). When sidecars
//! disagree on captured host (mixed-host pool from archived
//! cross-machine runs), the diff is against the first encountered
//! capture — sufficient for the single-host common case and
//! deterministic for the archived case.

use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use ktstr::host_context::{HostContext, collect_host_context};

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
/// `profile` (`--profile <NAME>`, the scheduler BUILD profile) and
/// `nextest_profile` (`--nextest-profile <NAME>`, the nextest test
/// profile) shape the `--exec` re-run only — the dry-run path runs no
/// tests. `profile` omitted leaves the scheduler at its release default.
///
/// Returns `Ok(0)` on a clean dry-run or successful exec.
/// Returns `Ok(N)` with nextest's exit code when `exec` is set
/// and nextest exits non-zero. Returns `Err` only for genuine
/// errors (unreadable sidecar root, nextest spawn failure).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_replay(
    dir: Option<&Path>,
    filter: Option<&str>,
    exec: bool,
    profile: Option<&str>,
    nextest_profile: Option<&str>,
    args: &[String],
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

    // Host-context comparison is identical between the dry-run and
    // --exec paths (same `pool` source for the captured host, same
    // `collect_host_context()` for current). Hoisting once avoids a
    // duplicate `/proc` + `/sys` probe (`collect_host_context`
    // touches ~10 pseudo-files) and a duplicate
    // `extract_captured_host` walk. Render still fires at the END
    // of each path so the section order matches the rest of the
    // path's narrative.
    let host_section = compute_host_diff(extract_captured_host(&pool), &collect_host_context());

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
        render_host_diff_section(&host_section);
        return Ok(0);
    }

    // Snapshot the failed names BEFORE invoking nextest so we
    // can re-look them up in the post-exec pool. `failed_names`
    // is a `BTreeSet<&str>` borrowing from the pre-exec pool;
    // own the strings so the references survive the
    // post-exec re-scan that builds a fresh pool Vec.
    let queued: BTreeSet<String> = failed_names.iter().map(|s| s.to_string()).collect();

    let exit = invoke_nextest(&filter_expr, profile, nextest_profile, args).with_context(|| {
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

    // Host-context diff was computed ABOVE (hoisted across both
    // paths) against the PRE-exec `pool` — that snapshot is the
    // source of truth for "what host did the original failures
    // happen on", whereas `post_pool` reflects post-replay
    // sidecars which (if successful) also carry a host capture
    // but from the replay host, not the original failure host.
    // Rendering here keeps the host section in the operator's
    // attention path right after the outcome diff.
    render_host_diff_section(&host_section);

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
    /// EVERY post-replay sidecar for the test_name reports a real
    /// pass (`passed=true && !skipped && !inconclusive`). Either
    /// the fix landed between the original run and this replay, or
    /// the test was passing intermittently (flake) and now passes.
    /// Operator sees a green signal here.
    Fixed,
    /// EVERY post-replay sidecar for the test_name reports a
    /// non-pass that is NOT inconclusive (`!passed || skipped`,
    /// and not `inconclusive`). The load-bearing case the replay
    /// command exists to surface: a regression that survived
    /// the operator's change.
    Persistent,
    /// EVERY post-replay sidecar for the test_name reports
    /// `inconclusive = true`. The replay produced no signal the
    /// gate could evaluate — e.g. every variant hit a
    /// zero-denominator ratio. Distinct from [`Self::Persistent`]
    /// (which means the gate ran and failed) and from
    /// [`Self::Fixed`] (which means the gate ran and passed):
    /// operator cannot conclude the fix worked, only that the
    /// replay could not measure. The triage signal is "re-run
    /// against a workload that produces the missing signal."
    Inconclusive,
    /// Test_name has no post-replay sidecar. Three plausible
    /// causes the operator should triage in order:
    /// - The test was removed from the suite between runs.
    /// - The `--filter` narrowed past it.
    /// - nextest crashed before reaching the test.
    ///
    /// The classifier itself doesn't distinguish — the inline
    /// triage hint in [`render_outcome_diff`] surfaces all three
    /// causes for the operator.
    Dropped,
    /// Test_name has multiple post-replay sidecars and they
    /// DISAGREE — at least two of [`fixed_count`/`persistent_count`/
    /// `inconclusive_count`] are non-zero. Common when a
    /// parameterized test runs across topology variants and only
    /// some variants reproduce the regression or produce the
    /// signal. The operator MUST drill in to see which variant
    /// landed in which bucket; surfacing this as `Fixed` would
    /// silently hide failing or inconclusive variants.
    ///
    /// Never collapse variant disagreement — that would silently
    /// hide a failing variant behind a passing one.
    ///
    /// [`fixed_count`]: ReplayOutcome::Mixed::fixed_count
    /// [`persistent_count`]: ReplayOutcome::Mixed::persistent_count
    /// [`inconclusive_count`]: ReplayOutcome::Mixed::inconclusive_count
    Mixed {
        /// Count of post-replay sidecars for this test_name
        /// that passed (real `passed=true && !skipped &&
        /// !inconclusive`).
        fixed_count: usize,
        /// Count of post-replay sidecars for this test_name
        /// that failed or were skipped (non-pass, non-inconclusive).
        persistent_count: usize,
        /// Count of post-replay sidecars for this test_name
        /// that were inconclusive (zero-denominator gate; the
        /// signal needed to evaluate was absent).
        inconclusive_count: usize,
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
/// - All variants real pass → [`ReplayOutcome::Fixed`]
/// - All variants real fail/skip (not inconclusive) →
///   [`ReplayOutcome::Persistent`]
/// - All variants inconclusive → [`ReplayOutcome::Inconclusive`]
/// - Variants disagree across the three buckets →
///   [`ReplayOutcome::Mixed`] carrying per-bucket counts so the
///   operator sees the disagreement instead of an erroneously-
///   green or erroneously-red verdict.
///
/// The Mixed case is the load-bearing addition over a naive
/// "any-variant-fixed" semantic: collapsing variant
/// disagreement to Fixed is a silent drop — a parameterized
/// test where variant A is fixed and variant B is still red
/// would silently report green. The Inconclusive variant
/// addition keeps zero-denominator runs from masquerading as
/// Persistent regressions — a replay that landed in a workload
/// with no measurable signal is not a fix-failure, just a
/// non-measurement, and the operator needs to know the
/// difference to triage.
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
                    let inconclusive_count =
                        variants.iter().filter(|sc| sc.is_inconclusive()).count();
                    let fixed_count = variants.iter().filter(|sc| sc.is_pass()).count();
                    let persistent_count = variants.len() - fixed_count - inconclusive_count;
                    match (fixed_count, persistent_count, inconclusive_count) {
                        (n, 0, 0) if n > 0 => ReplayOutcome::Fixed,
                        (0, n, 0) if n > 0 => ReplayOutcome::Persistent,
                        (0, 0, n) if n > 0 => ReplayOutcome::Inconclusive,
                        _ => ReplayOutcome::Mixed {
                            fixed_count,
                            persistent_count,
                            inconclusive_count,
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
/// the counts; per-test lines name each
/// PERSISTENT/INCONCLUSIVE/DROPPED/MIXED entry so the operator
/// can drill in without parsing nextest
/// output. FIXED entries are aggregated to a count only to
/// keep the diff short on healthy days; the operator who wants
/// per-test FIXED detail can grep the live nextest output above.
fn render_outcome_diff(outcomes: &BTreeMap<&str, ReplayOutcome>) {
    let (mut fixed, mut persistent, mut inconclusive, mut dropped, mut mixed) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for o in outcomes.values() {
        match o {
            ReplayOutcome::Fixed => fixed += 1,
            ReplayOutcome::Persistent => persistent += 1,
            ReplayOutcome::Inconclusive => inconclusive += 1,
            ReplayOutcome::Dropped => dropped += 1,
            ReplayOutcome::Mixed { .. } => mixed += 1,
        }
    }
    eprintln!();
    eprintln!(
        "ktstr replay: {fixed} FIXED, {persistent} PERSISTENT, \
         {inconclusive} INCONCLUSIVE, {mixed} MIXED, {dropped} DROPPED",
    );
    if persistent > 0 || dropped > 0 || mixed > 0 || inconclusive > 0 {
        for (name, outcome) in outcomes {
            match outcome {
                ReplayOutcome::Persistent => {
                    eprintln!("  PERSISTENT {name}");
                }
                ReplayOutcome::Inconclusive => {
                    // Triage hint: an Inconclusive replay means
                    // the gate could not measure (zero-denominator
                    // ratio across every variant). The operator
                    // sees INCONCLUSIVE without context otherwise;
                    // surfacing the remediation inline ("re-run
                    // against a workload that produces the missing
                    // signal") turns the opaque verdict into
                    // actionable diagnosis without forcing them to
                    // open a sidecar.
                    eprintln!(
                        "  INCONCLUSIVE {name} \
                         (zero-denominator gate across every variant — \
                         re-run against a workload that produces the \
                         missing signal, or drill into the per-variant \
                         sidecars to see which gates went unevaluated)",
                    );
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
                    inconclusive_count,
                } => {
                    // Surface variant disagreement explicitly:
                    // a parameterized test with some variants
                    // fixed and some still red (or inconclusive)
                    // must NOT collapse to a green Fixed verdict.
                    eprintln!(
                        "  MIXED {name} \
                         ({fixed_count} variant(s) fixed, \
                         {persistent_count} variant(s) still failing, \
                         {inconclusive_count} variant(s) inconclusive — \
                         drill into the per-variant sidecars to triage)",
                    );
                }
                ReplayOutcome::Fixed => {}
            }
        }
    }
}

/// Host-context section attached to every replay render path.
///
/// `Clone` deliberately omitted: HostDiffSection is constructed
/// once per `run_replay` invocation and consumed once by the
/// renderer; no production or test site needs to duplicate the
/// value, so the derive stays minimal.
/// Three shapes for the three possible relationships between the
/// captured-failure host and the current replay host:
///
/// - `NoCapture` — no sidecar in the input pool carried a
///   populated `host` field (older sidecar, host-context
///   collection failure, or test that ran without
///   `collect_host_context` in its path). The operator sees a
///   "(no host context captured)" hint so they know the
///   comparison is unavailable rather than silently absent.
///
/// - `Unchanged` — captured host and current host compare equal
///   field-for-field via `HostContext` `PartialEq`. The operator
///   sees a "(host context unchanged)" reassurance: replay env
///   matches capture env, so reproduction failures cannot be
///   blamed on host drift.
///
/// - `Changed(diff)` — the load-bearing case. `diff` carries the
///   `HostContext::diff` output (`key: before → after` lines, one
///   per differing field). The operator sees the per-field drift
///   inline and can correlate it with persistent/mixed outcomes
///   above.
///
/// Exhaustive matching is intentional inside the renderer —
/// every variant has a distinct user-visible string and a future
/// fourth variant would need explicit handling.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum HostDiffSection {
    /// No sidecar in the input pool carried `host = Some(_)`.
    NoCapture,
    /// Captured host equals current host field-for-field.
    Unchanged,
    /// Captured host differs from current host. Body is the
    /// already-rendered `HostContext::diff` output — one
    /// `  key: before → after` line per differing field. Empty
    /// body is impossible by construction (the only way to land
    /// here is `captured != current`, which by `HostContext`
    /// `Eq` semantics means at least one field differs and
    /// `diff` will emit at least one line). The renderer trusts
    /// that invariant — it does not re-check the body length.
    Changed(String),
}

/// Pick the captured host from the first sidecar in `pool` that
/// carries `host = Some(_)`. Returns `None` when no sidecar has
/// a populated host capture — older sidecars from before
/// `host_context` was wired, or runs where every
/// `collect_host_context` invocation failed.
///
/// Borrows from the pool rather than cloning: a `HostContext`
/// carries non-trivial heap data (sched_tunables `BTreeMap`,
/// cpufreq_governor `BTreeMap`, several `String` fields) — the
/// downstream consumer `compute_host_diff` only needs read access,
/// so a borrowed return avoids one full deep-clone per
/// `run_replay` invocation.
///
/// First-match-wins is acceptable because (a) the single-host
/// common case has every sidecar reporting the same host, and
/// (b) for the archived cross-machine case the pool order is
/// deterministic (collect_pool walks dirs in the order
/// `std::fs::read_dir` returns them). A future variant that wants
/// per-test host capture would attach the host to each `Mixed` /
/// `Persistent` entry directly rather than reaching for the
/// pool-level capture.
pub(crate) fn extract_captured_host(
    pool: &[ktstr::test_support::SidecarResult],
) -> Option<&HostContext> {
    pool.iter().find_map(|sc| sc.host.as_ref())
}

/// Classify the relationship between a captured host (from a
/// pre-replay sidecar) and the current host (collected at replay
/// time). Returns a [`HostDiffSection`] for the renderer to
/// translate into operator-facing output.
///
/// Pure function — takes `current` as a parameter rather than
/// calling [`collect_host_context`] internally so unit tests can
/// drive the classification with synthetic [`HostContext`]
/// fixtures (host_context.rs:[`HostContext::test_fixture`])
/// instead of standing up a real `/proc` + `/sys` probe in the
/// test process.
///
/// Implementation: defers all field-level comparison to
/// [`HostContext::diff`] at host_context.rs:642. That function
/// already handles every dimension of the struct (CPU identity,
/// memory, NUMA, kernel uname triple, cmdline, sched tunables,
/// THP, cpufreq governors) plus the `Option` / `BTreeMap` edge
/// cases (`None → Some`, `(absent)` map entries, per-CPU governor
/// drift). Re-implementing the comparison here would duplicate
/// that logic without adding anything new.
///
/// `heap_state` is stripped from BOTH sides before comparison —
/// it captures the *running process's* jemalloc footprint, not a
/// host attribute. The captured snapshot comes from the test
/// runner process; the current snapshot comes from the `cargo
/// ktstr replay` binary process. These are always different
/// processes, so `heap_state` will essentially always differ even
/// when the host is unchanged. Leaving it in would make
/// [`HostDiffSection::Unchanged`] unreachable and pollute every
/// [`HostDiffSection::Changed`] body with a noise line that has
/// nothing to do with host environment drift.
pub(crate) fn compute_host_diff(
    captured: Option<&HostContext>,
    current: &HostContext,
) -> HostDiffSection {
    let Some(captured) = captured else {
        return HostDiffSection::NoCapture;
    };
    let captured = without_heap_state(captured);
    let current = without_heap_state(current);
    if captured == current {
        HostDiffSection::Unchanged
    } else {
        HostDiffSection::Changed(captured.diff(&current))
    }
}

/// Return a clone of `host` with `heap_state` cleared to `None`.
/// Companion to [`compute_host_diff`] — see that function's docs
/// for the rationale (process-local jemalloc state is not a host
/// attribute, so the host comparison must skip it).
///
/// Owned-clone is acceptable here: a replay invocation runs at
/// most one host diff, so the two clones (captured + current) are
/// O(1) per process lifetime and the alternative (a "view" type
/// that excludes heap_state) would be over-engineered for a
/// once-per-CLI cost.
fn without_heap_state(host: &HostContext) -> HostContext {
    let mut stripped = host.clone();
    stripped.heap_state = None;
    stripped
}

/// Render the host-context section to stderr (the narrative
/// stream — stdout stays clean for the dry-run filter expression
/// pipe target).
///
/// Output shape per variant:
/// - `NoCapture` → one line: `ktstr replay: (no host context captured)`
/// - `Unchanged` → one line: `ktstr replay: (host context unchanged since capture)`
/// - `Changed(body)` → header line + indented diff body:
///   `ktstr replay: host context drift since capture:` followed
///   by the body verbatim (already two-space-indented per the
///   `HostContext::diff` line format).
///
/// The header line carries the `ktstr replay:` prefix on every
/// variant so an operator scanning stderr can grep for `ktstr
/// replay:` and pick up every replay-emitted line, matching the
/// prefix convention `render_outcome_diff` uses.
fn render_host_diff_section(section: &HostDiffSection) {
    match section {
        HostDiffSection::NoCapture => {
            eprintln!("ktstr replay: (no host context captured)");
        }
        HostDiffSection::Unchanged => {
            eprintln!("ktstr replay: (host context unchanged since capture)");
        }
        HostDiffSection::Changed(body) => {
            eprintln!("ktstr replay: host context drift since capture:");
            // `HostContext::diff` already terminates each line with
            // a newline AND uses two-space indentation, so a single
            // `eprint!` (NOT `eprintln!`) emits the body verbatim
            // without a trailing blank line.
            eprint!("{body}");
        }
    }
}

/// Select the set of test_names from `pool` whose sidecars
/// represent real failures (`s.is_fail()`), optionally narrowed
/// by a substring filter on test_name. Returns a BTreeSet for
/// deterministic ascending-order iteration when the renderer
/// builds the nextest filter expression.
///
/// Inconclusive runs are deliberately EXCLUDED from the replay
/// selection: `is_fail()` returns false for Inconclusive, so a
/// zero-denominator gate that emitted Inconclusive is not
/// auto-replayed. Inconclusive surfaces in the per-name
/// outcome diff for operator triage (see `render_outcome_diff`)
/// — replaying it would re-run the same zero-signal scenario.
///
/// Extracted as a pub(crate) free function so the scan-path
/// logic can be unit-tested against synthetic SidecarResult
/// fixtures without instantiating a real sidecar pool on
/// disk. Mirrors the closure-extraction pattern used elsewhere
/// for `matches_phase`.
pub(crate) fn select_failed_names<'a>(
    pool: &'a [ktstr::test_support::SidecarResult],
    filter: Option<&str>,
) -> BTreeSet<&'a str> {
    pool.iter()
        .filter(|s| s.is_fail())
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
pub(crate) fn build_nextest_filter(names: &BTreeSet<&str>) -> String {
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
            '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' | '\\' => {
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
///
/// `nextest_profile` becomes nextest's own `--profile <NAME>` (the test
/// profile), placed before the user's trailing `args` so a passthrough
/// token can't shadow it. `profile` sets `KTSTR_SCHEDULER_PROFILE` for
/// the scheduler-under-test BUILD; absent, `build_and_find_binary`
/// defaults the scheduler to `release`.
fn invoke_nextest(
    filter_expr: &str,
    profile: Option<&str>,
    nextest_profile: Option<&str>,
    args: &[String],
) -> Result<i32> {
    use std::process::Command;
    let mut cmd = Command::new("cargo");
    cmd.args(["nextest", "run", "-E", filter_expr]);
    // `--nextest-profile <NAME>` selects the NEXTEST test profile;
    // nextest's own flag is `--profile`.
    if let Some(np) = nextest_profile {
        cmd.args(["--profile", np]);
    }
    // Forward the operator's cargo/nextest flags (features,
    // `--cargo-profile`, …) verbatim so the replay re-run builds
    // identically to the original suite. No `--` separator is required.
    cmd.args(args);
    // `--profile <NAME>` sets the scheduler-under-test's cargo BUILD
    // profile via `KTSTR_SCHEDULER_PROFILE`.
    if let Some(p) = profile {
        cmd.env(ktstr::KTSTR_SCHEDULER_PROFILE_ENV, p);
    }
    let status = cmd.status().context("spawn `cargo nextest run`")?;
    Ok(status.code().unwrap_or(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktstr::test_support::SidecarResult;

    /// Build a minimal SidecarResult fixture for the scan-path
    /// tests. The selector consults `passed`/`skipped`/`inconclusive`
    /// via [`SidecarResult::is_fail`] (strict 4-state mutex); this
    /// helper hardpins `inconclusive: false` for the legacy 3-state
    /// test cases. Tests that exercise the Inconclusive arm of the
    /// selector must use [`synth_sidecar_with_inconclusive`] instead.
    /// The rest of the SidecarResult fields are placeholders that
    /// satisfy the struct.
    fn synth_sidecar(test_name: &str, passed: bool, skipped: bool) -> SidecarResult {
        SidecarResult {
            test_name: test_name.to_string(),
            perf_delta_assertions: Vec::new(),
            topology: "synth".into(),
            scheduler: "synth".into(),
            vcpus: 1,
            cpu_budget: 1,
            host_dilation: None,
            throughput_denomination: ktstr::test_support::ThroughputDenomination::CpuSec,
            scheduler_commit: None,
            resolve_source: None,
            project_commit: None,
            payload: None,
            metrics: Vec::new(),
            passed,
            skipped,
            inconclusive: false,
            expected_failure: false,
            stats: ktstr::assert::ScenarioStats::default(),
            monitor: None,
            periodic_fired: 0,
            periodic_target: 0,
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

    /// Inconclusive-bit-aware synth helper. Mirrors [`synth_sidecar`]
    /// but lets the test pin the `inconclusive` field too — used by
    /// the replay classifier tests that exercise the new
    /// [`ReplayOutcome::Inconclusive`] / mixed-with-inconclusive
    /// branches. A sidecar where `inconclusive = true` always has
    /// `passed = false` (the 4-state encoding is mutually
    /// exclusive); pass `passed = false` at every call site so the
    /// synth row encodes a legal sidecar shape.
    fn synth_sidecar_with_inconclusive(
        test_name: &str,
        passed: bool,
        skipped: bool,
        inconclusive: bool,
    ) -> SidecarResult {
        let set_count = u8::from(passed) + u8::from(skipped) + u8::from(inconclusive);
        debug_assert!(
            set_count <= 1,
            "SidecarResult strict 4-state mutex: at most one of \
             passed/skipped/inconclusive may be true; got \
             passed={passed}, skipped={skipped}, inconclusive={inconclusive}",
        );
        let mut sc = synth_sidecar(test_name, passed, skipped);
        sc.inconclusive = inconclusive;
        sc
    }

    // -- build_nextest_filter (formatting layer) --

    #[test]
    fn build_nextest_filter_single_name_emits_regex_anchored_form() {
        let mut names = BTreeSet::new();
        names.insert("scheduler_smoke_test");
        let expr = build_nextest_filter(&names);
        assert_eq!(
            expr, "test(/^(.*::)?scheduler_smoke_test$/)",
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

    /// Pool with mixed states — only `is_fail()` rows are
    /// selected. Pins the load-bearing failed-sidecar selector
    /// logic so a skipped row never collapses into the failed
    /// set. All fixture rows have `inconclusive=false` (see
    /// `synth_sidecar`), so the selector reduces to
    /// `!passed && !skipped` for these inputs; the
    /// `!inconclusive` conjunct of `is_fail` is exercised by
    /// [`select_failed_corner_case_inconclusive_excluded`].
    #[test]
    fn select_failed_skips_passed_and_skipped_keeps_only_real_failures() {
        let pool = synth_pool(&[
            ("test_pass", true, false),   // not selected (passed)
            ("test_skip", true, true),    // not selected (legacy passed+skipped fixture shape)
            ("test_fail1", false, false), // SELECTED
            ("test_fail2", false, false), // SELECTED
            ("test_corner", false, true), // not selected — !passed but skipped
        ]);
        let result = select_failed_names(&pool, None);
        let expected: BTreeSet<&str> = ["test_fail1", "test_fail2"].iter().copied().collect();
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
        let pool = synth_pool(&[("test_pass", true, false), ("test_fail", false, false)]);
        let result = select_failed_names(&pool, Some("nonexistent"));
        assert!(result.is_empty());
    }

    /// The `!skipped` conjunct of [`SidecarResult::is_fail`]
    /// (alongside `!passed && !inconclusive`) is what excludes
    /// the failed+skipped intermediate state from the replay
    /// queue. If a future refactor drops the `!skipped` clause
    /// from `is_fail`, this test surfaces the change.
    #[test]
    fn select_failed_corner_case_failed_and_skipped_excluded() {
        let pool = synth_pool(&[("test_fail_skip", false, true)]);
        let result = select_failed_names(&pool, None);
        assert!(
            result.is_empty(),
            "failed+skipped must be excluded; the && !skipped guard is load-bearing"
        );
    }

    /// The `!inconclusive` conjunct of `SidecarResult::is_fail`
    /// must exclude inconclusive sidecars from the replay queue —
    /// a zero-denominator gate that couldn't evaluate is not a
    /// regression to re-run, and queuing it as a "failure to
    /// reproduce" would mis-spend replay budget on runs that
    /// produced no signal in the first place. Pins the
    /// `inconclusive = true` exclusion at the selector layer
    /// (the classifier layer is exercised separately by the
    /// `classify_replay_*_inconclusive*` tests).
    #[test]
    fn select_failed_corner_case_inconclusive_excluded() {
        let pool = vec![
            synth_sidecar_with_inconclusive("test_inconclusive_only", false, false, true),
            synth_sidecar_with_inconclusive("test_real_fail", false, false, false),
        ];
        let result = select_failed_names(&pool, None);
        let expected: BTreeSet<&str> = ["test_real_fail"].iter().copied().collect();
        assert_eq!(
            result, expected,
            "inconclusive sidecars must NOT enter the replay queue — \
             SidecarResult::is_fail requires !inconclusive; a replay of a \
             zero-denominator run would re-spend budget on a non-measurement"
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
        assert_eq!(
            outcomes.len(),
            3,
            "every queued name gets exactly one outcome"
        );
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

    /// Post-replay sidecar that is skipped classifies as
    /// PERSISTENT, NOT Fixed. A skipped re-run is not a pass —
    /// the test didn't actually run, so the original failure
    /// isn't validated as fixed. Pins that `is_pass()`'s
    /// `!skipped` conjunct keeps a Skip row out of the
    /// `fixed_count` bucket. The fixture uses `(true, true)`
    /// (the legacy `synth_sidecar` shape); production
    /// `write_skip_sidecar` emits `passed=false, skipped=true,
    /// inconclusive=false` instead, but both shapes yield
    /// `is_pass() = false` because the `!skipped` conjunct
    /// dominates either way.
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
                inconclusive_count: 0,
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
        let post_pool = synth_pool(&[("test_broken", false, false), ("test_broken", false, false)]);
        let queued: BTreeSet<&str> = ["test_broken"].iter().copied().collect();
        let outcomes = classify_replay(&queued, &post_pool);
        assert_eq!(
            outcomes.get("test_broken"),
            Some(&ReplayOutcome::Persistent),
            "2-of-2 variants failed → Persistent, not Mixed"
        );
    }

    /// All variants inconclusive → Inconclusive. Pins the "every
    /// variant zero-denominator" branch — an inconclusive replay
    /// must not masquerade as Persistent (which would mislead the
    /// operator into thinking the fix failed) nor as Fixed (which
    /// would silently green-light an unmeasured replay).
    #[test]
    fn classify_replay_all_variants_inconclusive_classifies_as_inconclusive() {
        let post_pool = vec![
            synth_sidecar_with_inconclusive("test_zero_denom", false, false, true),
            synth_sidecar_with_inconclusive("test_zero_denom", false, false, true),
        ];
        let queued: BTreeSet<&str> = ["test_zero_denom"].iter().copied().collect();
        let outcomes = classify_replay(&queued, &post_pool);
        assert_eq!(
            outcomes.get("test_zero_denom"),
            Some(&ReplayOutcome::Inconclusive),
            "2-of-2 variants inconclusive → Inconclusive, not Persistent or Fixed"
        );
    }

    /// Mixed variants where one passed, one failed, and one was
    /// inconclusive → Mixed carrying all three counts. Pins the
    /// three-way disagreement branch — silently dropping the
    /// inconclusive count would hide a variant the operator needs
    /// to triage.
    #[test]
    fn classify_replay_three_way_mix_classifies_as_mixed_with_inconclusive() {
        let post_pool = vec![
            synth_sidecar_with_inconclusive("test_param", true, false, false),
            synth_sidecar_with_inconclusive("test_param", false, false, false),
            synth_sidecar_with_inconclusive("test_param", false, false, true),
        ];
        let queued: BTreeSet<&str> = ["test_param"].iter().copied().collect();
        let outcomes = classify_replay(&queued, &post_pool);
        assert_eq!(
            outcomes.get("test_param"),
            Some(&ReplayOutcome::Mixed {
                fixed_count: 1,
                persistent_count: 1,
                inconclusive_count: 1,
            }),
            "three-way variant disagreement surfaces every bucket"
        );
    }

    // -- phase 4: host-context diff (extract / compute / render) --

    /// Override one sidecar in a synth pool to carry a populated
    /// host capture so the extract+compute path can be exercised
    /// against fixture data without standing up a real `/proc` +
    /// `/sys` probe.
    fn synth_sidecar_with_host(
        test_name: &str,
        passed: bool,
        skipped: bool,
        host: HostContext,
    ) -> SidecarResult {
        let mut sc = synth_sidecar(test_name, passed, skipped);
        sc.host = Some(host);
        sc
    }

    /// G-P3.4 vacuity: empty pool → NoCapture. Pins that the
    /// extract path returns `None` for an empty pool and the
    /// compute path translates that to the NoCapture section
    /// — the operator gets the hint that host comparison is
    /// unavailable rather than silently dropping the section.
    #[test]
    fn compute_host_diff_empty_pool_yields_no_capture() {
        let pool: Vec<SidecarResult> = Vec::new();
        let captured = extract_captured_host(&pool);
        assert!(
            captured.is_none(),
            "empty pool must yield None — no sidecar means no captured host"
        );
        let current = HostContext::test_fixture();
        let section = compute_host_diff(captured, &current);
        assert_eq!(section, HostDiffSection::NoCapture);
    }

    /// G-P3.3 partial: pool with only host=None sidecars → also
    /// NoCapture. The extractor short-circuits as soon as it sees
    /// the first `Some(host)`; a pool where every sidecar has
    /// `host = None` (older sidecars from before host-context was
    /// wired, or runs where every collect failed) lands here.
    #[test]
    fn compute_host_diff_pool_without_host_yields_no_capture() {
        let pool = synth_pool(&[("test_a", false, false), ("test_b", false, false)]);
        // synth_pool sets host=None on every sidecar; pin the
        // assumption here so a future refactor that adds default
        // host to synth_pool surfaces this test.
        assert!(pool.iter().all(|sc| sc.host.is_none()));
        let captured = extract_captured_host(&pool);
        assert!(captured.is_none());
        let section = compute_host_diff(captured, &HostContext::test_fixture());
        assert_eq!(section, HostDiffSection::NoCapture);
    }

    /// G-P3.1 matched: captured host equals current host →
    /// Unchanged. Pins the "replay env matches capture env"
    /// reassurance signal — the operator gets confirmation that
    /// host drift cannot explain reproduction outcomes.
    #[test]
    fn compute_host_diff_captured_equals_current_yields_unchanged() {
        let host = HostContext::test_fixture();
        let pool = vec![synth_sidecar_with_host(
            "test_a",
            false,
            false,
            host.clone(),
        )];
        let captured = extract_captured_host(&pool);
        assert_eq!(
            captured,
            Some(&host),
            "extracted host must equal the one we attached to the sidecar"
        );
        let section = compute_host_diff(captured, &host);
        assert_eq!(section, HostDiffSection::Unchanged);
    }

    /// Captured host differs from current host in a load-bearing
    /// field (kernel_release) → Changed with a non-empty diff
    /// body that names the differing field. The operator MUST
    /// see the drift — if the host changed underneath, the
    /// renderer must surface that.
    #[test]
    fn compute_host_diff_differing_kernel_release_yields_changed_with_diff_body() {
        let mut captured = HostContext::test_fixture();
        captured.kernel_release = Some("6.16.0-test".to_string());
        let mut current = HostContext::test_fixture();
        current.kernel_release = Some("6.17.0-test".to_string());
        let pool = vec![synth_sidecar_with_host(
            "test_a",
            false,
            false,
            captured.clone(),
        )];
        let extracted = extract_captured_host(&pool);
        let section = compute_host_diff(extracted, &current);
        let HostDiffSection::Changed(body) = section else {
            panic!("kernel_release mismatch must yield Changed, got non-Changed variant");
        };
        assert!(
            body.contains("kernel_release"),
            "diff body must name the differing field; got: {body:?}"
        );
        assert!(
            body.contains("6.16.0-test") && body.contains("6.17.0-test"),
            "diff body must carry both before and after values; got: {body:?}"
        );
    }

    /// First-match-wins on the extractor: pool with multiple
    /// sidecars carrying different host captures returns the
    /// FIRST `Some(host)` encountered. Pins the deterministic
    /// pick policy documented at `extract_captured_host` so a
    /// future change to "last wins" or "most-common wins" must
    /// either preserve the semantic in tests or surface here.
    #[test]
    fn extract_captured_host_first_some_wins_over_later_sidecars() {
        let mut host_first = HostContext::test_fixture();
        host_first.kernel_release = Some("FIRST-release".to_string());
        let mut host_second = HostContext::test_fixture();
        host_second.kernel_release = Some("SECOND-release".to_string());
        let pool = vec![
            synth_sidecar_with_host("test_a", false, false, host_first.clone()),
            synth_sidecar_with_host("test_b", false, false, host_second.clone()),
        ];
        let extracted = extract_captured_host(&pool);
        assert_eq!(
            extracted.and_then(|h| h.kernel_release.as_deref()),
            Some("FIRST-release"),
            "first-match-wins: earlier sidecar's host capture must win"
        );
    }

    /// Table-driven coverage of the [`HostDiffSection::Changed`]
    /// diff body across every shape class HostContext exposes.
    /// kernel_release-only (covered by the focused test above) was
    /// insufficient — different field shapes flow through different
    /// branches of `HostContext::diff` (scalar `Option<T>`,
    /// `BTreeMap` field-with-default-empty, `Option<BTreeMap>`),
    /// and a regression in (say) the per-CPU `cpufreq_governor`
    /// diff branch would silently produce malformed diff lines.
    ///
    /// Each entry is a short label paired with a mutator that
    /// drifts one dimension and the substring that must appear in
    /// the diff body. Catches a regression where a particular
    /// shape class stops emitting rows OR emits rows under a wrong
    /// field-name key.
    #[test]
    fn compute_host_diff_changed_body_covers_each_dimension_class() {
        fn drift_body(mutate: impl FnOnce(&mut HostContext, &mut HostContext)) -> String {
            let mut captured = HostContext::test_fixture();
            let mut current = HostContext::test_fixture();
            mutate(&mut captured, &mut current);
            let pool = vec![synth_sidecar_with_host("test", false, false, captured)];
            let extracted = extract_captured_host(&pool);
            match compute_host_diff(extracted, &current) {
                HostDiffSection::Changed(body) => body,
                other => panic!("expected Changed, got {other:?}"),
            }
        }

        type Mutator = fn(&mut HostContext, &mut HostContext);

        // (label, mutator, substring-that-must-appear-in-body)
        let cases: &[(&str, Mutator, &str)] = &[
            (
                "scalar Option<String> drift (kernel_release)",
                |c, n| {
                    c.kernel_release = Some("6.16.0-a".to_string());
                    n.kernel_release = Some("6.16.0-b".to_string());
                },
                "kernel_release",
            ),
            (
                "scalar Option<u64> drift (total_memory_kib)",
                |c, n| {
                    c.total_memory_kib = Some(32 * 1024 * 1024);
                    n.total_memory_kib = Some(64 * 1024 * 1024);
                },
                "total_memory_kib",
            ),
            (
                "scalar Option<usize> drift (online_cpus)",
                |c, n| {
                    c.online_cpus = Some(8);
                    n.online_cpus = Some(16);
                },
                "online_cpus",
            ),
            (
                "scalar Option<usize> drift (numa_nodes)",
                |c, n| {
                    c.numa_nodes = Some(1);
                    n.numa_nodes = Some(2);
                },
                "numa_nodes",
            ),
            (
                "scalar Option<String> drift (arch)",
                |c, n| {
                    c.arch = Some("x86_64".to_string());
                    n.arch = Some("aarch64".to_string());
                },
                "arch",
            ),
            (
                "scalar Option<String> drift (kernel_cmdline)",
                |c, n| {
                    c.kernel_cmdline = Some("root=/dev/sda1".to_string());
                    n.kernel_cmdline = Some("root=/dev/sda2".to_string());
                },
                "kernel_cmdline",
            ),
            (
                "BTreeMap<usize,String> per-CPU drift (cpufreq_governor)",
                |c, n| {
                    c.cpufreq_governor.insert(0, "performance".to_string());
                    n.cpufreq_governor.insert(0, "powersave".to_string());
                },
                "cpufreq_governor.cpu0",
            ),
            (
                "Option<BTreeMap<String,String>> entry drift (sched_tunables)",
                |_c, n| {
                    // test_fixture pre-populates sched_tunables; mutate
                    // a known key on the current side only.
                    if let Some(m) = n.sched_tunables.as_mut() {
                        m.insert("sched_migration_cost_ns".to_string(), "999999".to_string());
                    }
                },
                "sched_tunables.sched_migration_cost_ns",
            ),
            (
                "Option None vs Some transition (cpu_vendor)",
                |c, n| {
                    c.cpu_vendor = None;
                    n.cpu_vendor = Some("GenuineIntel".to_string());
                },
                "cpu_vendor",
            ),
        ];

        for (label, mutate, expected_substr) in cases {
            let body = drift_body(*mutate);
            assert!(
                body.contains(expected_substr),
                "{label}: diff body missing expected substring \
                 {expected_substr:?}; got body: {body:?}"
            );
        }
    }

    /// `heap_state` drift between captured (different sidecar
    /// writer process) and current (replay binary process)
    /// MUST NOT cause a [`HostDiffSection::Changed`] verdict —
    /// jemalloc allocations are process-local, not host attributes.
    /// Pins the [`compute_host_diff`] heap_state strip + the
    /// matching documentation against a regression that drops
    /// the strip and re-introduces noise-line emission for
    /// every replay invocation.
    ///
    /// Without the strip, Unchanged is unreachable:
    /// captured.heap_state is the original-process jemalloc
    /// snapshot, current.heap_state is the replay-process
    /// snapshot — always different.
    #[test]
    fn compute_host_diff_strips_heap_state_so_process_local_drift_is_unchanged() {
        use ktstr::host_heap::HostHeapState;
        let mut captured = HostContext::test_fixture();
        let mut current = HostContext::test_fixture();
        // Two clearly-distinct heap states (different bytes-allocated
        // values) so a regression that DOESN'T strip would emit a
        // Changed body.
        let mut h1 = HostHeapState::test_fixture();
        h1.allocated_bytes = Some(1_000_000);
        captured.heap_state = Some(h1);
        let mut h2 = HostHeapState::test_fixture();
        h2.allocated_bytes = Some(2_000_000);
        current.heap_state = Some(h2);
        // Every other field matches (both came from test_fixture).
        let pool = vec![synth_sidecar_with_host("test", false, false, captured)];
        let extracted = extract_captured_host(&pool);
        let section = compute_host_diff(extracted, &current);
        assert_eq!(
            section,
            HostDiffSection::Unchanged,
            "heap_state drift must NOT yield Changed; if it does, \
             the strip in compute_host_diff regressed and every \
             replay will spam process-local jemalloc deltas as \
             host drift"
        );
    }

    /// Pool where the first sidecar has host=None and a LATER
    /// sidecar has host=Some(_) → extractor returns the LATER
    /// host. Pins the `find_map`-based first-Some-wins semantic
    /// against a regression to "first sidecar wins regardless of
    /// host=None": a None capture must not shadow a real
    /// downstream capture.
    #[test]
    fn extract_captured_host_skips_none_to_first_some() {
        let mut host = HostContext::test_fixture();
        host.kernel_release = Some("LATE-CAPTURE".to_string());
        let pool = vec![
            synth_sidecar("test_a_no_host", false, false), // host=None
            synth_sidecar_with_host("test_b_with_host", false, false, host.clone()),
        ];
        let extracted = extract_captured_host(&pool);
        assert_eq!(
            extracted.and_then(|h| h.kernel_release.as_deref()),
            Some("LATE-CAPTURE"),
            "extractor must skip host=None and land on first host=Some entry"
        );
    }
}
