//! `cargo ktstr perf-delta` — resolve the baseline commit a perf run is
//! compared against (the branch merge-base, a PR target, or an explicit
//! override) and surface the A/B commit pair the existing
//! `stats compare` engine pairs on.
//!
//! v0 (this increment): baseline resolution + reporting the resolved
//! `(baseline, head)` commit pair and the perf-test selector. The
//! dual-run (run perf-mode tests at HEAD and at the baseline in a gix
//! worktree, then invoke `compare_partitions`) is the follow-up
//! increment — see task: mergebase perf-delta.

/// How the baseline commit (the "compare-current-to" point) is resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BaseSelection {
    /// Compare HEAD directly against this commit-ish (`--base`). The
    /// testing / override knob — no merge-base is computed, so a run can
    /// be diffed against any chosen commit (or a downloaded cached
    /// baseline) without a real branch divergence.
    ExplicitCommit(String),
    /// Compare HEAD against `merge_base(HEAD, <ref>)` — the point where
    /// the branch diverged from `<ref>`.
    MergeBaseWith(String),
}

/// Resolve which baseline to use, by precedence:
/// 1. `--base <commit>` → [`BaseSelection::ExplicitCommit`] (skips
///    merge-base entirely; the testability override).
/// 2. `--base-ref <ref>` → merge-base against that ref.
/// 3. `$GITHUB_BASE_REF` (a PR's target branch, set only on
///    `pull_request` events) → merge-base against `origin/<ref>`, the
///    fetched remote-tracking ref.
/// 4. otherwise → merge-base against `default_branch`.
///
/// Empty env values are treated as unset (a `GITHUB_BASE_REF=`
/// pass-through on a non-PR run must not select an empty ref).
pub(crate) fn select_base(
    explicit_base: Option<&str>,
    base_ref_flag: Option<&str>,
    env_github_base_ref: Option<&str>,
    default_branch: &str,
) -> BaseSelection {
    if let Some(commit) = explicit_base {
        return BaseSelection::ExplicitCommit(commit.to_string());
    }
    if let Some(r) = base_ref_flag {
        return BaseSelection::MergeBaseWith(r.to_string());
    }
    if let Some(r) = env_github_base_ref.filter(|s| !s.is_empty()) {
        return BaseSelection::MergeBaseWith(format!("origin/{r}"));
    }
    BaseSelection::MergeBaseWith(default_branch.to_string())
}

use anyhow::{Context, Result};

/// 7-hex short hash, `-dirty` suffixed only when `id` is current HEAD
/// AND the worktree is dirty — matching the sidecar `project_commit`
/// format so the resolved commit lines up with pool entries that the
/// `stats compare` engine pairs on.
fn short_hash(repo: &gix::Repository, id: gix::ObjectId) -> String {
    let head_oid = repo.head_id().ok().map(|i| i.detach());
    let dirty = head_oid
        .as_ref()
        .and_then(|_| ktstr::test_support::repo_is_dirty(repo))
        .unwrap_or(false);
    let short = id.to_hex_with_len(7).to_string();
    if dirty && head_oid == Some(id) {
        format!("{short}-dirty")
    } else {
        short
    }
}

/// Resolve a revspec to a single commit ObjectId (mirrors
/// `stats::resolve_commit_specs`'s Include/ExcludeParents handling).
fn rev_parse_commit(repo: &gix::Repository, spec: &str) -> Result<gix::ObjectId> {
    let parsed = repo
        .rev_parse(spec)
        .with_context(|| format!("resolve revision '{spec}'"))?;
    match parsed.detach() {
        gix::revision::plumbing::Spec::Include(id)
        | gix::revision::plumbing::Spec::ExcludeParents(id) => Ok(id),
        other => anyhow::bail!("'{spec}' did not resolve to a single commit ({other:?})"),
    }
}

/// Resolve the baseline commit per the [`BaseSelection`]: an explicit
/// `--base` commit verbatim, or `merge_base(HEAD, <ref>)`.
pub(crate) fn resolve_baseline(repo: &gix::Repository, sel: &BaseSelection) -> Result<gix::ObjectId> {
    match sel {
        BaseSelection::ExplicitCommit(c) => rev_parse_commit(repo, c),
        BaseSelection::MergeBaseWith(r) => {
            let head = repo.head_id().context("resolve HEAD")?.detach();
            let base = rev_parse_commit(repo, r)?;
            let mb = repo
                .merge_base(head, base)
                .with_context(|| format!("compute merge-base(HEAD, {r})"))?;
            Ok(mb.detach())
        }
    }
}

// ---------------------------------------------------------------------------
// Dual-run (increment 3b): produce the baseline run in a git worktree
// ---------------------------------------------------------------------------

use std::path::{Path, PathBuf};
use std::process::Command;

/// Run-pool leaf the baseline run's sidecars are written to — a
/// DISTINCT subdir from HEAD's `{kernel}-{HEAD}` default leaf so both
/// commits' sidecars coexist under the shared [`runs_root`] without
/// collision. The compare partitions by the sidecar `project_commit`
/// FIELD (set from the worktree's git checkout), not by this dir name,
/// so the name only needs to be unique; `baseline_short` (7-hex) makes
/// it so. Absolute (rooted at the MAIN tree's runs-root) so the
/// worktree-cwd child writes into the main pool, not the worktree's own
/// `target/ktstr`.
///
/// [`runs_root`]: ktstr::test_support::runs_root
fn baseline_sidecar_leaf(runs_root_abs: &Path, baseline_short: &str) -> PathBuf {
    runs_root_abs.join(format!("perf-delta-baseline-{baseline_short}"))
}

/// Checkout path for the baseline worktree, under `temp_root` (the
/// system temp dir in production) so the full source checkout never
/// lands inside the runs-root run-dir walk and never pollutes the
/// source tree. `baseline_short` keeps concurrent baselines distinct.
fn worktree_checkout_dir(temp_root: &Path, baseline_short: &str) -> PathBuf {
    temp_root.join(format!("ktstr-perf-delta-wt-{baseline_short}"))
}

/// Env pairs for the baseline child `cargo ktstr test`: restrict to
/// `performance_mode` tests ([`KTSTR_PERF_ONLY_ENV`]) and redirect its
/// sidecar output into the main pool leaf
/// ([`KTSTR_SIDECAR_DIR_ENV`], absolute) so it pools with HEAD's run.
///
/// [`KTSTR_PERF_ONLY_ENV`]: ktstr::KTSTR_PERF_ONLY_ENV
/// [`KTSTR_SIDECAR_DIR_ENV`]: ktstr::KTSTR_SIDECAR_DIR_ENV
fn baseline_child_env(sidecar_leaf_abs: &Path) -> Vec<(&'static str, String)> {
    vec![
        (ktstr::KTSTR_PERF_ONLY_ENV, "1".to_string()),
        (
            ktstr::KTSTR_SIDECAR_DIR_ENV,
            sidecar_leaf_abs.to_string_lossy().into_owned(),
        ),
    ]
}

/// `git worktree add --detach <dir> <commit>` argv (after the `git`
/// program name). `--detach` checks out `commit` without creating a
/// branch — the worktree is build-and-discard, never committed to.
/// `commit` is the FULL hex oid (not the 7-hex short) so the checkout
/// is unambiguous.
fn worktree_add_argv(wt_dir: &Path, commit_full_hex: &str) -> Vec<String> {
    vec![
        "worktree".to_string(),
        "add".to_string(),
        "--detach".to_string(),
        wt_dir.to_string_lossy().into_owned(),
        commit_full_hex.to_string(),
    ]
}

/// `git worktree remove --force <dir>` argv. `--force` removes despite
/// the build artifacts the checkout accumulates (an unforced remove
/// refuses a dirty worktree).
fn worktree_remove_argv(wt_dir: &Path) -> Vec<String> {
    vec![
        "worktree".to_string(),
        "remove".to_string(),
        "--force".to_string(),
        wt_dir.to_string_lossy().into_owned(),
    ]
}

/// `cargo ktstr test --kernel <k> [-E <filter>]` argv (after the
/// `cargo` program name). The baseline child runs the SAME selection
/// as HEAD; `KTSTR_PERF_ONLY` in the env (see [`baseline_child_env`])
/// restricts both ends to `performance_mode` tests, and `-E` narrows
/// within that set.
fn perf_test_argv(kernel: &str, filter: Option<&str>) -> Vec<String> {
    let mut v = vec![
        "ktstr".to_string(),
        "test".to_string(),
        "--kernel".to_string(),
        kernel.to_string(),
    ];
    if let Some(f) = filter {
        v.push("-E".to_string());
        v.push(f.to_string());
    }
    v
}

/// Count `*.ktstr.json` sidecars directly in `dir` (non-recursive). A
/// missing or unreadable dir counts as `0`. The dual-run path uses this
/// on the baseline leaf to detect "no `performance_mode` tests produced
/// any sidecar" and exit cleanly, instead of running the compare into a
/// confusing empty-pool bail.
fn count_sidecars(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.ends_with(".ktstr.json"))
        })
        .count()
}

/// RAII cleanup for the baseline worktree: `git worktree remove
/// --force` on drop so an early return, `?`, or panic in [`dual_run`]
/// never leaks a checkout. Removal runs from the MAIN tree
/// (`repo_root`), the only place `git worktree remove` resolves the
/// linked worktree.
struct WorktreeGuard {
    repo_root: PathBuf,
    wt_dir: PathBuf,
}

impl Drop for WorktreeGuard {
    fn drop(&mut self) {
        // Best-effort: this is the cleanup path, so a failure cannot
        // propagate. `git worktree remove` prints its own diagnostic to
        // stderr on failure, and a leftover dir surfaces as an explicit
        // "already exists" error on the next run's `worktree add` rather
        // than a silent stale-data reuse — so a swallowed Err here is
        // visible downstream, not hidden.
        let _ = Command::new("git")
            .current_dir(&self.repo_root)
            .args(worktree_remove_argv(&self.wt_dir))
            .status();
    }
}

/// Produce BOTH commits' `performance_mode` sidecars so the compare has
/// fresh data: check the baseline commit out in a detached git worktree
/// and run its perf tests there (sidecars redirected into the main
/// pool), then run HEAD's perf tests in the main tree. The worktree is
/// removed on return via [`WorktreeGuard`].
///
/// A non-zero exit from a child test run is logged but does NOT abort —
/// some `performance_mode` failures (e.g. LLC-contention) are expected,
/// and the sidecars that DID get written are still comparable. A
/// spawn failure (git/cargo not runnable) IS a hard error.
///
/// `gix` 0.83 has no worktree-creation API (only list/inspect), so the
/// worktree is managed by shelling `git worktree add/remove`.
fn dual_run(
    repo_root: &Path,
    baseline_full_hex: &str,
    baseline_short: &str,
    kernel: &str,
    filter: Option<&str>,
) -> Result<()> {
    let runs_root_abs = repo_root.join(ktstr::test_support::runs_root());
    let leaf = baseline_sidecar_leaf(&runs_root_abs, baseline_short);
    let wt_dir = worktree_checkout_dir(&std::env::temp_dir(), baseline_short);

    // Create the detached baseline worktree.
    let add = Command::new("git")
        .current_dir(repo_root)
        .args(worktree_add_argv(&wt_dir, baseline_full_hex))
        .status()
        .with_context(|| format!("spawn `git worktree add` for baseline {baseline_short}"))?;
    if !add.success() {
        anyhow::bail!(
            "`git worktree add {} {baseline_full_hex}` failed ({add}) — \
             a leftover worktree from a prior run may need `git worktree prune`",
            wt_dir.display(),
        );
    }
    let _guard = WorktreeGuard {
        repo_root: repo_root.to_path_buf(),
        wt_dir: wt_dir.clone(),
    };

    // Baseline run: in the worktree, perf-only, sidecars into the main pool leaf.
    println!("perf-delta: running baseline {baseline_short} perf tests in worktree");
    let baseline_status = Command::new("cargo")
        .current_dir(&wt_dir)
        .args(perf_test_argv(kernel, filter))
        .envs(baseline_child_env(&leaf))
        .status()
        .with_context(|| format!("spawn baseline `cargo ktstr test` in {}", wt_dir.display()))?;
    if !baseline_status.success() {
        eprintln!(
            "perf-delta: warning: baseline perf run exited {baseline_status} — \
             comparing the sidecars that were written"
        );
    }

    // HEAD run: in the main tree, perf-only, default {kernel}-{HEAD} leaf.
    println!("perf-delta: running HEAD perf tests in the working tree");
    let head_status = Command::new("cargo")
        .current_dir(repo_root)
        .args(perf_test_argv(kernel, filter))
        .env(ktstr::KTSTR_PERF_ONLY_ENV, "1")
        .status()
        .context("spawn HEAD `cargo ktstr test`")?;
    if !head_status.success() {
        eprintln!(
            "perf-delta: warning: HEAD perf run exited {head_status} — \
             comparing the sidecars that were written"
        );
    }
    Ok(())
}

/// CLI args for `cargo ktstr perf-delta`.
pub(crate) struct PerfDeltaArgs<'a> {
    /// `--base <commit>` override (skips merge-base).
    pub base: Option<&'a str>,
    /// `--base-ref <ref>` override (merge-base against this ref).
    pub base_ref: Option<&'a str>,
    /// Nextest `-E` filter narrowing within the perf-mode set.
    pub filter: Option<&'a str>,
    /// Branch to merge-base against when no override / env is present.
    pub default_branch: &'a str,
    /// `--kernel <SPEC>` — required with `--dual-run` (the kernel the
    /// baseline and HEAD perf tests boot). `None` is valid only for the
    /// cached-baseline path (no `--dual-run`), which runs no tests.
    pub kernel: Option<&'a str>,
    /// `--dual-run` — produce both commits' runs via a baseline git
    /// worktree before comparing, instead of comparing sidecars already
    /// pooled from prior / downloaded runs.
    pub dual_run: bool,
    /// `--threshold <PCT>` — uniform relative regression threshold
    /// (percent). Mutually exclusive with `policy`.
    pub threshold: Option<f64>,
    /// `--policy <PATH>` — per-metric threshold JSON. Mutually
    /// exclusive with `threshold`.
    pub policy: Option<&'a std::path::Path>,
    /// Per-phase render projection (`--no-phases` / `--phases-only` /
    /// `--steps-only` / `--phase` / `--phase-threshold`), threaded to
    /// the compare's `PhaseDisplayOptions`. Render-only — does not
    /// change the regression verdict / exit code.
    pub phase_display: cli::PhaseDisplayOptions,
}

/// Resolve the `(baseline, HEAD)` commit pair and A/B-compare their
/// sidecars via the existing `compare_partitions` engine, returning its
/// exit code (non-zero on a regression). The compare pairs per-scenario
/// and applies each metric's polarity + abs/rel thresholds.
///
/// Two source models for the baseline run's sidecars:
/// - default (cached-baseline): compares sidecars ALREADY pooled under
///   the runs-root from a prior run or a downloaded CI artifact.
/// - `--dual-run`: [`dual_run`] PRODUCES both commits' runs first —
///   the baseline in a detached git worktree, HEAD in the working tree,
///   both `performance_mode`-only (`KTSTR_PERF_ONLY`) and narrowed by
///   the `-E` filter — then this compares the freshly pooled sidecars.
pub(crate) fn run(args: &PerfDeltaArgs<'_>) -> Result<i32> {
    let cwd = std::env::current_dir().context("get cwd")?;
    let repo = gix::discover(&cwd).context("discover git repository")?;
    let env_base = std::env::var("GITHUB_BASE_REF").ok();
    let sel = select_base(args.base, args.base_ref, env_base.as_deref(), args.default_branch);
    let baseline_oid = resolve_baseline(&repo, &sel)?;
    let baseline = short_hash(&repo, baseline_oid);
    let head = short_hash(&repo, repo.head_id().context("resolve HEAD")?.detach());
    if baseline == head {
        anyhow::bail!(
            "baseline ({baseline}) resolves to HEAD — nothing to compare; \
             choose a different --base / --base-ref"
        );
    }
    println!("perf-delta: candidate HEAD {head} vs baseline {baseline}");
    match &sel {
        BaseSelection::ExplicitCommit(c) => println!("  baseline: explicit --base {c}"),
        BaseSelection::MergeBaseWith(r) => println!("  baseline: merge-base(HEAD, {r})"),
    }
    println!(
        "  perf tests: {}",
        args.filter.unwrap_or("all performance_mode tests")
    );

    if args.dual_run {
        let kernel = args
            .kernel
            .context("--dual-run requires --kernel (the kernel both perf runs boot)")?;
        dual_run(
            &cwd,
            &baseline_oid.to_hex().to_string(),
            &baseline,
            kernel,
            args.filter,
        )?;
        // If the baseline run produced no sidecars, no `performance_mode`
        // tests were selected (none are defined yet, or the `-E` filter
        // matched none). There is nothing to compare — exit cleanly
        // rather than letting the compare bail on an empty pool. HEAD
        // runs the same selection, so a zero-baseline implies no delta
        // is computable regardless of HEAD's output.
        let leaf = baseline_sidecar_leaf(&cwd.join(ktstr::test_support::runs_root()), &baseline);
        if count_sidecars(&leaf) == 0 {
            println!(
                "perf-delta: no performance_mode sidecars produced at baseline {baseline} \
                 — nothing to compare (define #[ktstr_test(performance_mode)] tests, or widen \
                 the -E filter, to enable the delta)"
            );
            return Ok(0);
        }
    }

    // Reuse the stats-compare engine: partition the pooled sidecars by
    // project_commit (baseline = A, HEAD = B) and emit the polarity-aware
    // regression verdict. The resolved short hashes match the sidecar
    // project_commit format, so they line up with pool entries directly.
    let build = crate::stats::BuildCompareFilters {
        a_project_commit: vec![baseline],
        b_project_commit: vec![head],
        ..Default::default()
    };
    let (filter_a, filter_b) = build.build();
    // Resolve the regression sensitivity from --threshold / --policy
    // via the shared resolver (same as `stats compare`); neither flag
    // falls through to the registry per-metric defaults.
    let policy = cli::ComparisonPolicy::from_cli_flags(args.threshold, args.policy)
        .context("resolve --threshold / --policy")?;
    cli::compare_partitions(
        &filter_a,
        &filter_b,
        None,
        &policy,
        None,
        false,
        &args.phase_display,
    )
}

use ktstr::cli;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_base_wins_and_skips_merge_base() {
        assert_eq!(
            select_base(Some("abc123"), Some("main"), Some("release"), "main"),
            BaseSelection::ExplicitCommit("abc123".to_string()),
            "--base must take precedence over every ref source and skip merge-base",
        );
    }

    #[test]
    fn base_ref_flag_beats_env_and_default() {
        assert_eq!(
            select_base(None, Some("topic"), Some("release"), "main"),
            BaseSelection::MergeBaseWith("topic".to_string()),
        );
    }

    #[test]
    fn github_base_ref_resolves_to_origin_tracking_ref() {
        assert_eq!(
            select_base(None, None, Some("release-1.2"), "main"),
            BaseSelection::MergeBaseWith("origin/release-1.2".to_string()),
            "a PR target branch is the fetched remote-tracking ref",
        );
    }

    #[test]
    fn empty_github_base_ref_is_unset_falls_back_to_default() {
        assert_eq!(
            select_base(None, None, Some(""), "main"),
            BaseSelection::MergeBaseWith("main".to_string()),
            "empty GITHUB_BASE_REF (non-PR run) must not select an empty ref",
        );
    }

    #[test]
    fn no_inputs_falls_back_to_default_branch() {
        assert_eq!(
            select_base(None, None, None, "main"),
            BaseSelection::MergeBaseWith("main".to_string()),
        );
    }

    // ---- dual-run pure helpers (increment 3b) ----

    #[test]
    fn baseline_sidecar_leaf_is_distinct_subdir_of_runs_root() {
        let root = Path::new("/work/target/ktstr");
        let leaf = baseline_sidecar_leaf(root, "abc1234");
        assert_eq!(leaf, Path::new("/work/target/ktstr/perf-delta-baseline-abc1234"));
        // A distinct baseline yields a distinct leaf (no collision); and
        // neither collides with HEAD's `{kernel}-{HEAD}` default leaf.
        assert_ne!(leaf, baseline_sidecar_leaf(root, "def5678"));
    }

    #[test]
    fn worktree_checkout_dir_under_temp_root() {
        assert_eq!(
            worktree_checkout_dir(Path::new("/tmp"), "abc1234"),
            Path::new("/tmp/ktstr-perf-delta-wt-abc1234"),
        );
    }

    #[test]
    fn baseline_child_env_sets_perf_only_and_absolute_sidecar_dir() {
        let leaf = Path::new("/work/target/ktstr/perf-delta-baseline-abc1234");
        let env = baseline_child_env(leaf);
        assert_eq!(
            env,
            vec![
                ("KTSTR_PERF_ONLY", "1".to_string()),
                (
                    "KTSTR_SIDECAR_DIR",
                    "/work/target/ktstr/perf-delta-baseline-abc1234".to_string(),
                ),
            ],
        );
        // Pin the env-var names against the library constants so a rename
        // of either keeps the child pointed at the right vars.
        assert_eq!(env[0].0, ktstr::KTSTR_PERF_ONLY_ENV);
        assert_eq!(env[1].0, ktstr::KTSTR_SIDECAR_DIR_ENV);
    }

    #[test]
    fn worktree_add_argv_is_detached_checkout_of_full_oid() {
        assert_eq!(
            worktree_add_argv(Path::new("/tmp/wt"), "0123456789abcdef"),
            vec![
                "worktree".to_string(),
                "add".to_string(),
                "--detach".to_string(),
                "/tmp/wt".to_string(),
                "0123456789abcdef".to_string(),
            ],
        );
    }

    #[test]
    fn worktree_remove_argv_forces_removal() {
        assert_eq!(
            worktree_remove_argv(Path::new("/tmp/wt")),
            vec![
                "worktree".to_string(),
                "remove".to_string(),
                "--force".to_string(),
                "/tmp/wt".to_string(),
            ],
        );
    }

    #[test]
    fn count_sidecars_counts_ktstr_json_and_zero_for_missing() {
        let base = std::env::temp_dir().join(format!("ktstr-pd-count-{}", std::process::id()));
        // A missing directory counts as zero (the no-tests-ran case).
        assert_eq!(count_sidecars(&base.join("absent")), 0);
        std::fs::create_dir_all(&base).expect("mk tempdir");
        std::fs::write(base.join("a.ktstr.json"), "{}").expect("write a");
        std::fs::write(base.join("b.ktstr.json"), "{}").expect("write b");
        std::fs::write(base.join("notes.txt"), "x").expect("write txt");
        assert_eq!(
            count_sidecars(&base),
            2,
            "only *.ktstr.json sidecars count, not sibling files",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn perf_test_argv_appends_filter_only_when_present() {
        assert_eq!(
            perf_test_argv("6.14", None),
            vec![
                "ktstr".to_string(),
                "test".to_string(),
                "--kernel".to_string(),
                "6.14".to_string(),
            ],
        );
        assert_eq!(
            perf_test_argv("6.14", Some("test(perf_smoke)")),
            vec![
                "ktstr".to_string(),
                "test".to_string(),
                "--kernel".to_string(),
                "6.14".to_string(),
                "-E".to_string(),
                "test(perf_smoke)".to_string(),
            ],
        );
    }
}
