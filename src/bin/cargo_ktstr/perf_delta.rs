//! `cargo ktstr perf-delta` — resolve the baseline commit a perf run is
//! compared against (the branch merge-base, a PR target, or an explicit
//! override) and surface the A/B commit pair the shared compare engine
//! (`compare_partitions` / `compare_partitions_noise`) pairs on.
//!
//! Baseline resolution (`select_base` / `resolve_baseline`) feeds the COMMIT
//! axis in `run`: HEAD vs a baseline commit, partitioned by `project_commit`
//! (the sole perf-delta axis — a cross-config question like scheduler A vs B is
//! answered in-test via the Verdict DSL's `better_across_phases`, not here). The
//! `--dual-run` and `--noise-adjust` paths PRODUCE both commits'
//! `performance_mode` sidecars — checking the baseline out in a detached
//! `git worktree` (shelled, because `gix` 0.83 has no worktree-creation
//! API) before invoking `compare_partitions` / `compare_partitions_noise`.

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
/// shared compare engine (`compare_partitions`) pairs on.
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

/// Resolve a revspec to a single commit ObjectId, accepting both the
/// `Include` and `ExcludeParents` (`^rev` / single-ended range) spec forms.
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
pub(crate) fn resolve_baseline(
    repo: &gix::Repository,
    sel: &BaseSelection,
) -> Result<gix::ObjectId> {
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

use ktstr::flock::{FlockMode, try_flock};
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

/// Advisory-lock path for a baseline worktree: `<wt_dir>.lock`, a SIBLING of
/// `wt_dir` (not inside it, which cleanup removes) so the flock outlives the
/// worktree it guards. The path is shared per baseline commit, matching
/// `worktree_checkout_dir`, so two runs of the same baseline contend on it.
fn worktree_lock_path(wt_dir: &Path) -> PathBuf {
    let mut s = wt_dir.as_os_str().to_owned();
    s.push(".lock");
    PathBuf::from(s)
}

/// Take the exclusive per-baseline worktree lock, or bail if another perf-delta
/// run holds it. Returns the lock fd to hold for the worktree's whole lifetime
/// (drop releases; the OS also releases on process death). See the call sites
/// for why the shared per-baseline path makes this serialization load-bearing.
fn acquire_baseline_worktree_lock(
    wt_dir: &Path,
    baseline_short: &str,
) -> Result<std::os::fd::OwnedFd> {
    let lock_path = worktree_lock_path(wt_dir);
    try_flock(&lock_path, FlockMode::Exclusive)
        .with_context(|| format!("lock baseline worktree {}", lock_path.display()))?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "another `cargo ktstr perf-delta` run is using the baseline worktree \
                 for {baseline_short}; retry once it finishes"
            )
        })
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

/// `cargo ktstr test --kernel <k> [-E <filter>] [<passthrough>...]` argv
/// (after the `cargo` program name). The baseline child runs the SAME
/// selection as HEAD; `KTSTR_PERF_ONLY` in the env (see
/// [`baseline_child_env`]) restricts both ends to `performance_mode`
/// tests, and `-E` narrows within that set.
///
/// `filter` (`-E`) and `passthrough` (the user's cargo/nextest flags,
/// e.g. `--features integration,wprof`) become the inner `test`'s
/// passthrough args, which `run_cargo_sub` forwards verbatim to
/// `cargo nextest run`. Forwarding to BOTH sides is load-bearing: a
/// delta computed across mismatched feature selections is not a
/// like-for-like comparison, and — before this — the baseline/HEAD
/// `cargo ktstr test` built with default features, so feature-gated
/// `performance_mode` tests produced no sidecars and the compare found
/// nothing.
///
/// `profile` (`--profile <NAME>`, the scheduler BUILD profile) and
/// `nextest_profile` (`--nextest-profile <NAME>`, the nextest test
/// profile) are emitted as NATIVE `cargo ktstr test` flags, right after
/// `--kernel <k>` and before the `-E` / passthrough args. The child
/// re-splits its argv by flag name (the bin's `argsplit` module), so this
/// ordering is no longer load-bearing — it is kept only for a stable,
/// readable child command line. Omitting `profile`
/// leaves the scheduler at its release default (see
/// [`ktstr::build_and_find_binary`]); both are forwarded to BOTH sides so
/// baseline and HEAD build identically.
fn perf_test_argv(
    kernel: &str,
    filter: Option<&str>,
    profile: Option<&str>,
    nextest_profile: Option<&str>,
    passthrough: &[String],
) -> Vec<String> {
    let mut v = vec![
        "ktstr".to_string(),
        "test".to_string(),
        "--kernel".to_string(),
        kernel.to_string(),
    ];
    // Emit native `cargo ktstr test` flags before the `-E` / passthrough
    // args. The child re-splits its argv by flag name (the bin's argsplit
    // module), so a native flag placed after `-E` would still parse
    // correctly — this ordering is kept only for a stable, readable child
    // command line.
    if let Some(p) = profile {
        v.push("--profile".to_string());
        v.push(p.to_string());
    }
    if let Some(np) = nextest_profile {
        v.push("--nextest-profile".to_string());
        v.push(np.to_string());
    }
    if let Some(f) = filter {
        v.push("-E".to_string());
        v.push(f.to_string());
    }
    v.extend(passthrough.iter().cloned());
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
        // stderr on failure. A swallowed Err (or a skipped drop on Ctrl-C /
        // SIGKILL / crash) leaves a leftover, but the next run's
        // `create_baseline_worktree` calls `prune_leftover_worktree` before
        // `git worktree add`, which recovers it — so a failure here degrades
        // to a re-created worktree next run, not a stuck "already exists".
        let _ = Command::new("git")
            .current_dir(&self.repo_root)
            .args(worktree_remove_argv(&self.wt_dir))
            .status();
    }
}

/// Best-effort self-heal of a leftover baseline worktree before a fresh
/// `git worktree add`. Ctrl-C (SIGINT), SIGKILL, or a crash kills the process
/// before [`WorktreeGuard`]`::drop` runs, orphaning the temp checkout AND its
/// registration — so the next run's `git worktree add` hard-fails
/// `'<dir>' already exists`. Idempotent recovery here covers EVERY interruption
/// mode (a SIGINT handler could not cover SIGKILL / crash). Three steps, each
/// swallowing failure because a clean tree makes them no-ops: remove a
/// still-registered worktree (clears registration + dir), delete a bare leftover
/// dir that carried no / a broken registration, then prune a dangling
/// registration whose dir is now gone.
fn prune_leftover_worktree(repo_root: &Path, wt_dir: &Path) {
    let _ = Command::new("git")
        .current_dir(repo_root)
        .args(worktree_remove_argv(wt_dir))
        .status();
    if wt_dir.exists() {
        let _ = std::fs::remove_dir_all(wt_dir);
    }
    let _ = Command::new("git")
        .current_dir(repo_root)
        .args(["worktree", "prune"])
        .status();
}

/// Create the detached baseline worktree at `wt_dir`, first self-healing any
/// leftover from an interrupted prior run via [`prune_leftover_worktree`] so a
/// re-run always recovers rather than dying on "already exists".
fn create_baseline_worktree(
    repo_root: &Path,
    wt_dir: &Path,
    baseline_full_hex: &str,
    baseline_short: &str,
) -> Result<()> {
    prune_leftover_worktree(repo_root, wt_dir);
    let add = Command::new("git")
        .current_dir(repo_root)
        .args(worktree_add_argv(wt_dir, baseline_full_hex))
        .status()
        .with_context(|| format!("spawn `git worktree add` for baseline {baseline_short}"))?;
    if !add.success() {
        anyhow::bail!(
            "`git worktree add {} {baseline_full_hex}` failed ({add})",
            wt_dir.display(),
        );
    }
    Ok(())
}

/// Absolutize a relative kernel-source path against the MAIN tree so the
/// baseline child resolves the SAME kernel HEAD does. The baseline child
/// runs with its cwd in the worktree (a temp dir), where a relative
/// `--kernel ../linux` would resolve against the worktree instead of the
/// project root — pointing at the wrong (or a nonexistent) tree. A
/// kernel VERSION (e.g. `6.14`), an already-absolute path, and a
/// relative string that is not an existing path (a version, or a typo
/// ktstr surfaces identically from either cwd) all pass through
/// unchanged; only a relative path that EXISTS from the main tree is
/// rewritten to its absolute form.
fn resolve_kernel_for_children(repo_root: &Path, kernel: &str) -> String {
    let p = Path::new(kernel);
    if p.is_relative() && repo_root.join(p).exists() {
        repo_root.join(p).to_string_lossy().into_owned()
    } else {
        kernel.to_string()
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
#[allow(clippy::too_many_arguments)]
fn dual_run(
    repo_root: &Path,
    baseline_full_hex: &str,
    baseline_short: &str,
    kernel: &str,
    filter: Option<&str>,
    profile: Option<&str>,
    nextest_profile: Option<&str>,
    passthrough: &[String],
) -> Result<()> {
    // runs_root() is absolute under the cargo-ktstr orchestrator
    // (install_runs_root_env stamps KTSTR_RUNS_ROOT), so this join
    // yields that shared dir — Path::join discards `repo_root` against
    // an absolute rhs, and both the baseline worktree child and HEAD
    // inherit KTSTR_RUNS_ROOT and write there. join() also anchors the
    // relative raw-nextest fallback to repo_root; correct for both.
    let runs_root_abs = repo_root.join(ktstr::test_support::runs_root());
    let leaf = baseline_sidecar_leaf(&runs_root_abs, baseline_short);
    let wt_dir = worktree_checkout_dir(&std::env::temp_dir(), baseline_short);
    // Absolutize a relative kernel path so the baseline child (cwd =
    // worktree) resolves the same tree HEAD does (cwd = main).
    let kernel_arg = resolve_kernel_for_children(repo_root, kernel);

    // Serialize concurrent perf-delta runs against the SAME baseline: they share
    // one per-baseline worktree path, and create_baseline_worktree force-removes
    // any leftover — which would destroy a live concurrent run's checkout. The
    // exclusive advisory flock (OS-released on process death) makes a live
    // concurrent run bail cleanly here, while an interrupted run's lock frees for
    // the next run (which then self-heals the leftover). Held for the whole
    // baseline lifetime; declared before `_guard` so it drops AFTER the worktree
    // is removed.
    let _wt_lock = acquire_baseline_worktree_lock(&wt_dir, baseline_short)?;
    // Create the detached baseline worktree, self-healing any leftover from an
    // interrupted prior run (Ctrl-C / SIGKILL / crash skip WorktreeGuard::drop).
    create_baseline_worktree(repo_root, &wt_dir, baseline_full_hex, baseline_short)?;
    let _guard = WorktreeGuard {
        repo_root: repo_root.to_path_buf(),
        wt_dir: wt_dir.clone(),
    };

    // Baseline run: in the worktree, perf-only, sidecars into the main pool leaf.
    println!("perf-delta: running baseline {baseline_short} perf tests in worktree");
    let baseline_status = Command::new("cargo")
        .current_dir(&wt_dir)
        .args(perf_test_argv(
            &kernel_arg,
            filter,
            profile,
            nextest_profile,
            passthrough,
        ))
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
        .args(perf_test_argv(
            &kernel_arg,
            filter,
            profile,
            nextest_profile,
            passthrough,
        ))
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

/// `--noise-adjust N` run-production: like [`dual_run`] but loops, writing each
/// of the N baseline + N HEAD runs into its OWN sidecar leaf so the N runs per
/// side accumulate. Two runs of the same test+config write the SAME sidecar
/// filename — the variant hash covers run-INDEPENDENT identity only (topology /
/// scheduler / payload / work_type / sysctls / kargs) — so a single shared leaf would
/// last-writer-wins OVERWRITE all but the final run, collapsing N samples to 1.
/// (The `KTSTR_SIDECAR_DIR` override these children set ALSO disables the run-epoch
/// pre_clear, so that is not the mechanism.) Distinct per-run leaves give distinct
/// dirs -> distinct files -> all N survive. All leaves live under the shared
/// runs-root, so the noise
/// compare's pool scan finds them and partitions by `project_commit`
/// (baseline=A, HEAD=B), not by leaf. The detached baseline worktree is created
/// once and removed on return via [`WorktreeGuard`]. A non-zero child exit is
/// logged, not fatal (some `performance_mode` failures are expected; the
/// sidecars that landed are still comparable).
#[allow(clippy::too_many_arguments)]
fn noise_dual_run(
    repo_root: &Path,
    baseline_full_hex: &str,
    baseline_short: &str,
    head_short: &str,
    kernel: &str,
    filter: Option<&str>,
    profile: Option<&str>,
    nextest_profile: Option<&str>,
    passthrough: &[String],
    runs: usize,
) -> Result<()> {
    let runs_root_abs = repo_root.join(ktstr::test_support::runs_root());
    let wt_dir = worktree_checkout_dir(&std::env::temp_dir(), baseline_short);
    let kernel_arg = resolve_kernel_for_children(repo_root, kernel);

    // Serialize same-baseline runs before the self-heal force-removes the shared
    // worktree (see dual_run for the full rationale). Held for the whole run;
    // declared before `_guard` so it drops after the worktree is removed.
    let _wt_lock = acquire_baseline_worktree_lock(&wt_dir, baseline_short)?;
    // Self-heal a leftover worktree, then create the detached baseline (see
    // create_baseline_worktree — Ctrl-C / SIGKILL skip WorktreeGuard::drop).
    create_baseline_worktree(repo_root, &wt_dir, baseline_full_hex, baseline_short)?;
    let _guard = WorktreeGuard {
        repo_root: repo_root.to_path_buf(),
        wt_dir: wt_dir.clone(),
    };

    for i in 0..runs {
        // Per-run leaves: two runs of the same test+config share a variant-hashed
        // filename, so a single leaf would overwrite all but the last; distinct
        // dirs keep all N. The compare still partitions by project_commit
        // (baseline vs HEAD), not by leaf name.
        let base_leaf = runs_root_abs.join(format!("perf-delta-noise-base-{baseline_short}-{i}"));
        let head_leaf = runs_root_abs.join(format!("perf-delta-noise-head-{head_short}-{i}"));
        println!("perf-delta --noise-adjust: run {}/{runs}", i + 1);

        let bs = Command::new("cargo")
            .current_dir(&wt_dir)
            .args(perf_test_argv(
                &kernel_arg,
                filter,
                profile,
                nextest_profile,
                passthrough,
            ))
            .envs(baseline_child_env(&base_leaf))
            .status()
            .with_context(|| format!("spawn baseline `cargo ktstr test` (noise run {i})"))?;
        if !bs.success() {
            eprintln!("perf-delta --noise-adjust: warning: baseline run {i} exited {bs}");
        }

        let hs = Command::new("cargo")
            .current_dir(repo_root)
            .args(perf_test_argv(
                &kernel_arg,
                filter,
                profile,
                nextest_profile,
                passthrough,
            ))
            .env(ktstr::KTSTR_PERF_ONLY_ENV, "1")
            .env(ktstr::KTSTR_SIDECAR_DIR_ENV, &head_leaf)
            .status()
            .with_context(|| format!("spawn HEAD `cargo ktstr test` (noise run {i})"))?;
        if !hs.success() {
            eprintln!("perf-delta --noise-adjust: warning: HEAD run {i} exited {hs}");
        }
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
    /// `--noise-adjust <N>` — run each side N times and decide significance from
    /// whether the two sides are SEPARATED (a Welch two-sample t-test, or fully
    /// disjoint `[min, max]` bands) AND the delta is MATERIAL (the registry
    /// dual-gate), instead of a fixed single-run threshold. Commit axis only — it
    /// implies the dual-run production, looped N times. `None` = the default
    /// single-run threshold compare.
    pub noise_adjust: Option<usize>,
    /// `--noise-spread-threshold <PCT>` — the per-side relative-spread limit (in
    /// percent) above which `--noise-adjust` ANNOTATES a metric with an advisory
    /// "noisy spread" flag. Advisory only: it never suppresses a separated +
    /// material regression. Defaults to `5.0` when `--noise-adjust` is set.
    pub noise_spread_threshold: Option<f64>,
    /// `--profile <NAME>` — the scheduler-under-test's cargo BUILD profile,
    /// forwarded to BOTH sides' `cargo ktstr test` so baseline and HEAD
    /// build the scheduler identically. `None` leaves it at the release
    /// default. Only meaningful on the run-producing paths (`--dual-run` /
    /// `--noise-adjust`); [`run`] rejects it on the paths that run no tests.
    pub profile: Option<&'a str>,
    /// `--nextest-profile <NAME>` — the nextest test profile, forwarded to
    /// BOTH sides' `cargo ktstr test`. Only meaningful on the run-producing
    /// paths; [`run`] rejects it on the paths that run no tests.
    pub nextest_profile: Option<&'a str>,
    /// cargo/nextest flags forwarded verbatim to BOTH sides' `cargo ktstr
    /// test` → `cargo nextest run` (e.g. `--features integration,wprof`);
    /// no `--` separator required (but native flags must precede it — see
    /// the CLI `args` doc). Only meaningful on the run-producing paths
    /// (`--dual-run` / `--noise-adjust`); [`run`] rejects a non-empty
    /// passthrough on the cached-baseline path (which runs no tests) rather
    /// than silently dropping it.
    pub passthrough: &'a [String],
}

/// A/B-compare two perf runs via the existing `compare_partitions`
/// engine, returning its exit code (non-zero on a regression). The
/// compare pairs per-scenario and applies each metric's polarity +
/// abs/rel thresholds on the COMMIT axis: HEAD vs a baseline commit,
/// partitioned by `project_commit`. Sidecars come either from the pool
/// already (cached-baseline) or from `--dual-run`, which [`dual_run`]
/// PRODUCES in a detached git worktree (baseline) + the working tree
/// (HEAD), both `performance_mode`-only and `-E`-narrowed. (Cross-config
/// questions — e.g. scheduler A vs B — are answered in-test via the
/// Verdict DSL's `better_across_phases`, not by this command.)
pub(crate) fn run(args: &PerfDeltaArgs<'_>) -> Result<i32> {
    // Regression sensitivity from --threshold / --policy via the shared
    // ComparisonPolicy resolver.
    let policy = cli::ComparisonPolicy::from_cli_flags(args.threshold, args.policy)
        .context("resolve --threshold / --policy")?;

    // Fail loud rather than SILENTLY DROP the build-shaping flags: the
    // cargo/nextest passthrough AND `--profile` / `--nextest-profile` only
    // reach a `cargo ktstr test` run on the run-PRODUCING paths
    // (--dual-run / --noise-adjust N). The cached-baseline compare runs no
    // tests, so any of these there would vanish with no effect — surface the
    // misuse instead.
    let runs_tests = args.dual_run || args.noise_adjust.is_some();
    if !runs_tests
        && (!args.passthrough.is_empty()
            || args.profile.is_some()
            || args.nextest_profile.is_some())
    {
        anyhow::bail!(
            "build-shaping flags (cargo/nextest passthrough {:?}, --profile {:?}, \
             --nextest-profile {:?}) are only applied by the run-producing paths \
             (--dual-run / --noise-adjust N); this invocation compares already-pooled \
             sidecars and runs no tests, so they would be ignored. Add --dual-run \
             (or --noise-adjust N), or drop the flags.",
            args.passthrough,
            args.profile,
            args.nextest_profile,
        );
    }

    let cwd = std::env::current_dir().context("get cwd")?;
    let repo = gix::discover(&cwd).context("discover git repository")?;
    let env_base = std::env::var("GITHUB_BASE_REF").ok();
    let sel = select_base(
        args.base,
        args.base_ref,
        env_base.as_deref(),
        args.default_branch,
    );
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

    // Noise-adjusted axis: produce N runs per side (looped dual-run into
    // per-run leaves), then compare from the observed spread instead of a fixed
    // threshold. Commit axis only.
    if let Some(n) = args.noise_adjust {
        let kernel = args
            .kernel
            .context("--noise-adjust requires --kernel (the kernel both perf runs boot)")?;
        noise_dual_run(
            &cwd,
            &baseline_oid.to_hex().to_string(),
            &baseline,
            &head,
            kernel,
            args.filter,
            args.profile,
            args.nextest_profile,
            args.passthrough,
            n,
        )?;
        let threshold = args.noise_spread_threshold.unwrap_or(5.0);
        let build = crate::stats::BuildCompareFilters {
            a_project_commit: vec![baseline.clone()],
            b_project_commit: vec![head.clone()],
            ..Default::default()
        };
        let (filter_a, filter_b) = build.build();
        return cli::compare_partitions_noise(
            &filter_a,
            &filter_b,
            None,
            threshold,
            &args.phase_display,
        );
    }

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
            args.profile,
            args.nextest_profile,
            args.passthrough,
        )?;
        // If the baseline run produced no sidecars, no `performance_mode`
        // tests were selected (none are defined yet, or the `-E` filter
        // matched none). There is nothing to compare — exit cleanly
        // rather than letting the compare bail on an empty pool. HEAD
        // runs the same selection, so a zero-baseline implies no delta
        // is computable regardless of HEAD's output.
        // Absolute runs_root() (orchestrator-stamped) makes this join
        // return the shared dir; relative falls back anchored to cwd.
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
    cli::compare_partitions(&filter_a, &filter_b, None, &policy, None)
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
        assert_eq!(
            leaf,
            Path::new("/work/target/ktstr/perf-delta-baseline-abc1234")
        );
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
    fn prune_leftover_worktree_removes_a_bare_leftover_dir() {
        // Simulate an interrupted prior run's orphaned worktree: a populated dir
        // the next `git worktree add` would trip over. With a non-repo repo_root
        // the `git worktree remove/prune` calls fail harmlessly (not a repo), so
        // this pins the fs fallback that clears the bare leftover — the exact
        // "'<dir>' already exists" recovery a Ctrl-C'd run needs.
        let base = std::env::temp_dir().join(format!("ktstr-pd-selfheal-{}", std::process::id()));
        let repo_root = base.join("not-a-repo");
        let wt = base.join("wt-leftover");
        std::fs::create_dir_all(&repo_root).expect("mk repo_root");
        std::fs::create_dir_all(wt.join("scx-ktstr")).expect("mk leftover subtree");
        std::fs::write(wt.join("scx-ktstr/build.rs"), "x").expect("write leftover file");
        assert!(wt.exists(), "leftover present before self-heal");
        prune_leftover_worktree(&repo_root, &wt);
        assert!(
            !wt.exists(),
            "self-heal must remove a bare leftover worktree dir so `git worktree add` can proceed",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn baseline_worktree_lock_blocks_a_concurrent_same_baseline_run() {
        // Two perf-delta runs against the same baseline share worktree_lock_path;
        // the exclusive flock must let only ONE proceed so the self-heal never
        // force-removes a live concurrent run's checkout (the medium-severity
        // regression this lock closes). Also pins the sibling `.lock` path shape
        // and that the lock frees for the next run once released (the
        // interrupted-run recovery path — the OS releases it on process death).
        let wt = std::env::temp_dir().join(format!("ktstr-pd-locktest-wt-{}", std::process::id()));
        assert_eq!(
            worktree_lock_path(&wt),
            PathBuf::from(format!("{}.lock", wt.display())),
            "lock is a sibling <wt_dir>.lock, outside the worktree cleanup removes",
        );
        let lock_path = worktree_lock_path(&wt);
        let _ = std::fs::remove_file(&lock_path);

        let held =
            acquire_baseline_worktree_lock(&wt, "abc1234").expect("first run acquires the lock");
        assert!(
            acquire_baseline_worktree_lock(&wt, "abc1234").is_err(),
            "a concurrent same-baseline run must fail to acquire, not clobber the live worktree",
        );
        drop(held);
        let after = acquire_baseline_worktree_lock(&wt, "abc1234")
            .expect("lock frees for the next run once released");
        drop(after);
        let _ = std::fs::remove_file(&lock_path);
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
    fn perf_test_argv_appends_filter_and_passthrough() {
        let expect = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        let feat = ["--features".to_string(), "integration,wprof".to_string()];

        // No filter, no passthrough.
        assert_eq!(
            perf_test_argv("6.14", None, None, None, &[]),
            expect(&["ktstr", "test", "--kernel", "6.14"]),
        );
        // Filter only, appended as `test`'s trailing args.
        assert_eq!(
            perf_test_argv("6.14", Some("test(perf_smoke)"), None, None, &[]),
            expect(&[
                "ktstr",
                "test",
                "--kernel",
                "6.14",
                "-E",
                "test(perf_smoke)"
            ]),
        );
        // Passthrough forwarded verbatim AFTER the filter, so feature
        // selection reaches BOTH baseline and HEAD `cargo ktstr test`.
        assert_eq!(
            perf_test_argv("6.14", Some("test(perf_smoke)"), None, None, &feat),
            expect(&[
                "ktstr",
                "test",
                "--kernel",
                "6.14",
                "-E",
                "test(perf_smoke)",
                "--features",
                "integration,wprof",
            ]),
        );
        // Passthrough with no filter.
        assert_eq!(
            perf_test_argv("6.14", None, None, None, &feat),
            expect(&[
                "ktstr",
                "test",
                "--kernel",
                "6.14",
                "--features",
                "integration,wprof",
            ]),
        );
    }

    /// `--profile` / `--nextest-profile` are emitted as NATIVE `cargo
    /// ktstr test` flags after `--kernel <k>` and before the `-E` /
    /// passthrough args. The child re-splits its argv by flag name (the
    /// bin's `argsplit` module), so the ordering is no longer required for
    /// correctness; this pins the stable, readable child command line
    /// `perf_test_argv` emits.
    #[test]
    fn perf_test_argv_emits_profiles_before_passthrough() {
        let expect = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        let feat = ["--features".to_string(), "integration".to_string()];

        // Both profiles, with filter + passthrough: profiles precede `-E`.
        assert_eq!(
            perf_test_argv("6.14", Some("test(perf)"), Some("dev"), Some("ci"), &feat),
            expect(&[
                "ktstr",
                "test",
                "--kernel",
                "6.14",
                "--profile",
                "dev",
                "--nextest-profile",
                "ci",
                "-E",
                "test(perf)",
                "--features",
                "integration",
            ]),
        );
        // Scheduler profile only, no filter/passthrough.
        assert_eq!(
            perf_test_argv("6.14", None, Some("dev"), None, &[]),
            expect(&["ktstr", "test", "--kernel", "6.14", "--profile", "dev"]),
        );
        // Nextest profile only.
        assert_eq!(
            perf_test_argv("6.14", None, None, Some("ci"), &[]),
            expect(&[
                "ktstr",
                "test",
                "--kernel",
                "6.14",
                "--nextest-profile",
                "ci",
            ]),
        );
    }

    #[test]
    fn resolve_kernel_absolutizes_relative_existing_path() {
        // A relative kernel path that exists from the main tree must be
        // absolutized so the worktree-cwd baseline child resolves the
        // same tree (the bug: `../linux` resolved against the worktree).
        let base = std::env::temp_dir().join(format!("ktstr-pd-kresolve-{}", std::process::id()));
        std::fs::create_dir_all(base.join("linux")).expect("mk linux dir");
        assert_eq!(
            resolve_kernel_for_children(&base, "linux"),
            base.join("linux").to_string_lossy(),
        );
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn resolve_kernel_passes_version_and_absolute_and_missing_through() {
        let repo = Path::new("/no/such/repo-xyz");
        // A version (not an existing path) passes through verbatim.
        assert_eq!(resolve_kernel_for_children(repo, "6.14"), "6.14");
        // An already-absolute path passes through unchanged.
        assert_eq!(
            resolve_kernel_for_children(repo, "/abs/linux"),
            "/abs/linux",
        );
        // A relative non-existent path passes through (version or typo —
        // ktstr surfaces the same error from either cwd).
        assert_eq!(
            resolve_kernel_for_children(repo, "../nonexistent-xyz"),
            "../nonexistent-xyz",
        );
    }

    /// Exercise the gix-resolution path against a real throwaway repo:
    /// `rev_parse_commit` for revspecs, `resolve_baseline` for BOTH the
    /// explicit-commit and merge-base arms, and `short_hash`'s 7-hex
    /// format. These are the load-bearing baseline-resolution functions
    /// that the pure-helper unit tests can't reach (they need commits).
    /// Shells `git` to build the fixture — the same `git` dependency
    /// perf-delta's worktree path already requires.
    #[test]
    fn gix_resolution_against_a_temp_repo() {
        use std::process::Command;
        let dir = std::env::temp_dir().join(format!("ktstr-pd-gitfix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mk tempdir");
        let run = |args: &[&str]| {
            let ok = Command::new("git")
                .current_dir(&dir)
                .args([
                    "-c",
                    "user.email=t@example.invalid",
                    "-c",
                    "user.name=t",
                    "-c",
                    "commit.gpgsign=false",
                ])
                .args(args)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            assert!(ok, "git {args:?} failed");
        };
        run(&["init", "-q"]);
        std::fs::write(dir.join("f"), "1").expect("write f");
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "first"]);
        std::fs::write(dir.join("f"), "2").expect("write f");
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "second"]);

        let repo = gix::discover(&dir).expect("discover temp repo");
        let head = rev_parse_commit(&repo, "HEAD").expect("rev-parse HEAD");
        let first = rev_parse_commit(&repo, "HEAD~1").expect("rev-parse HEAD~1");
        assert_ne!(head, first, "HEAD and HEAD~1 are distinct commits");

        // ExplicitCommit resolves the revspec verbatim (no merge-base).
        assert_eq!(
            resolve_baseline(&repo, &BaseSelection::ExplicitCommit("HEAD~1".to_string())).unwrap(),
            first,
        );
        // MergeBaseWith(HEAD~1): HEAD~1 is HEAD's ancestor, so the
        // merge-base of (HEAD, HEAD~1) is HEAD~1 itself.
        assert_eq!(
            resolve_baseline(&repo, &BaseSelection::MergeBaseWith("HEAD~1".to_string())).unwrap(),
            first,
            "merge-base(HEAD, HEAD~1) is HEAD~1",
        );
        // short_hash: 7 lowercase hex, no -dirty for a clean non-HEAD commit.
        let sh = short_hash(&repo, first);
        assert_eq!(sh.len(), 7, "short hash is 7 hex chars: {sh}");
        assert!(sh.bytes().all(|b| b.is_ascii_hexdigit()), "hex only: {sh}");

        std::fs::remove_dir_all(&dir).ok();
    }

}
