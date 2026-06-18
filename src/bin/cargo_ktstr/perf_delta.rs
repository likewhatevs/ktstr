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
}

/// v0: resolve and report the `(baseline, HEAD)` commit pair the run
/// would A/B-compare, plus the perf-test selector. Always returns 0.
/// The dual-run (perf-mode tests at HEAD and at the baseline in a gix
/// worktree) + the `compare_partitions` regression verdict are the
/// follow-up increment — see task "mergebase perf-delta".
pub(crate) fn run(args: &PerfDeltaArgs<'_>) -> Result<i32> {
    let repo = gix::discover(std::env::current_dir().context("get cwd")?)
        .context("discover git repository")?;
    let env_base = std::env::var("GITHUB_BASE_REF").ok();
    let sel = select_base(args.base, args.base_ref, env_base.as_deref(), args.default_branch);
    let baseline = short_hash(&repo, resolve_baseline(&repo, &sel)?);
    let head = short_hash(&repo, repo.head_id().context("resolve HEAD")?.detach());
    println!("perf-delta: candidate HEAD {head} vs baseline {baseline}");
    match &sel {
        BaseSelection::ExplicitCommit(c) => println!("  baseline: explicit --base {c}"),
        BaseSelection::MergeBaseWith(r) => println!("  baseline: merge-base(HEAD, {r})"),
    }
    println!(
        "  perf tests: {}",
        args.filter.unwrap_or("all performance_mode tests")
    );
    eprintln!(
        "perf-delta v0: resolved the A/B commit pair only; the dual-run \
         (perf-mode tests at HEAD and at the baseline) + regression compare \
         is the next increment"
    );
    Ok(0)
}

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
}
