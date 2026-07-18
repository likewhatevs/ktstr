//! `cargo ktstr perf-delta` — resolve the baseline commit a perf run is
//! compared against (the branch merge-base, a PR target, or an explicit
//! override) and surface the A/B commit pair the shared compare engine
//! (`compare_partitions` / `compare_partitions_noise`) pairs on.
//!
//! Baseline resolution (`select_base` / `resolve_baseline`) feeds the COMMIT
//! axis in `run`: HEAD vs a baseline commit, partitioned by `project_commit`
//! (the sole perf-delta axis — a cross-config question like scheduler A vs B is
//! answered in-test via the Verdict DSL's `better_across_phases`, not here). The
//! `--noise-adjust` path PRODUCES both commits'
//! `performance_mode` sidecars — checking the baseline out into a plain temp
//! dir via `gix` (`index_from_tree` + `gix::worktree::state::checkout`, no
//! linked worktree and no `.git`) before invoking `compare_partitions_noise`;
//! without it, `run` compares sidecars already pooled from a prior run or a
//! downloaded artifact via `compare_partitions`.

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
// Run production (--noise-adjust): check both commits out into plain gix
// checkouts and run their performance_mode tests
// ---------------------------------------------------------------------------

use ktstr::flock::{FlockMode, try_flock};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Checkout path for the baseline, under `temp_root` (the system temp dir in
/// production) so the full source checkout never lands inside the runs-root
/// run-dir walk and never pollutes the source tree. `baseline_short` keeps
/// concurrent baselines distinct.
fn checkout_dir(temp_root: &Path, baseline_short: &str) -> PathBuf {
    temp_root.join(format!("ktstr-perf-delta-co-{baseline_short}"))
}

/// Advisory-lock path for a baseline checkout: `<wt_dir>.lock`, a SIBLING of
/// `wt_dir` (not inside it, which cleanup removes) so the flock outlives the
/// checkout it guards. The path is shared per baseline commit, matching
/// `checkout_dir`, so two runs of the same baseline contend on it.
fn checkout_lock_path(wt_dir: &Path) -> PathBuf {
    let mut s = wt_dir.as_os_str().to_owned();
    s.push(".lock");
    PathBuf::from(s)
}

/// Take the exclusive per-baseline checkout lock, or bail if another perf-delta
/// run holds it. Returns the lock fd to hold for the checkout's whole lifetime
/// (drop releases; the OS also releases on process death). See the call sites
/// for why the shared per-baseline path makes this serialization load-bearing.
fn acquire_baseline_checkout_lock(
    wt_dir: &Path,
    baseline_short: &str,
) -> Result<std::os::fd::OwnedFd> {
    let lock_path = checkout_lock_path(wt_dir);
    try_flock(&lock_path, FlockMode::Exclusive)
        .with_context(|| format!("lock baseline checkout {}", lock_path.display()))?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "another `cargo ktstr perf-delta` run is using the baseline checkout \
                 for {baseline_short}; retry once it finishes"
            )
        })
}

/// Env pairs for the baseline child `cargo ktstr test`: restrict to
/// `performance_mode` tests ([`KTSTR_PERF_ONLY_ENV`]), redirect its sidecar
/// output into the main pool leaf ([`KTSTR_SIDECAR_DIR_ENV`], absolute) so
/// it pools with HEAD's run, and stamp the recorded `project_commit`
/// ([`KTSTR_PROJECT_COMMIT_ENV`] = `project_commit`, the parent's resolved
/// baseline label). The child's checkout is a plain gix checkout with no
/// `.git`, so it CANNOT derive `project_commit` from a `gix::discover`; the
/// explicit label also guarantees the recorded field equals what the
/// compare filters the pool on.
///
/// [`KTSTR_PERF_ONLY_ENV`]: ktstr::KTSTR_PERF_ONLY_ENV
/// [`KTSTR_SIDECAR_DIR_ENV`]: ktstr::KTSTR_SIDECAR_DIR_ENV
/// [`KTSTR_PROJECT_COMMIT_ENV`]: ktstr::KTSTR_PROJECT_COMMIT_ENV
fn baseline_child_env(
    sidecar_leaf_abs: &Path,
    project_commit: &str,
) -> Vec<(&'static str, String)> {
    vec![
        (ktstr::KTSTR_PERF_ONLY_ENV, "1".to_string()),
        (
            ktstr::KTSTR_SIDECAR_DIR_ENV,
            sidecar_leaf_abs.to_string_lossy().into_owned(),
        ),
        (ktstr::KTSTR_PROJECT_COMMIT_ENV, project_commit.to_string()),
    ]
}

/// Reclaim baseline checkouts orphaned by prior interrupted runs.
///
/// [`create_checkout_from_tree`] only self-heals the CURRENT run's
/// checkout dir; a checkout left by an interrupted run against a
/// DIFFERENT baseline (Ctrl-C / SIGKILL / crash skip
/// [`CheckoutGuard`]`::drop`) accumulates in `temp_root` forever. This
/// runs at perf-delta startup on the run-producing paths and removes
/// every `ktstr-perf-delta-co-*` directory whose sibling `.lock` is NOT
/// held by a live run — mirroring `cleanup_stale_shm`'s lock-gated
/// reclamation, so it never removes a concurrent run's checkout.
/// Best-effort: each step swallows failure (a busy or racing entry is
/// skipped and reclaimed by a later sweep). The checkout is a plain dir
/// (no linked-worktree registration), so `remove_dir_all` fully reclaims
/// it — no `git worktree prune` is needed.
fn sweep_stale_checkouts(temp_root: &Path) {
    let Ok(entries) = std::fs::read_dir(temp_root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        // The checkout DIRS only — not the sibling `<...>.lock` files.
        if !name.starts_with("ktstr-perf-delta-co-") || name.ends_with(".lock") {
            continue;
        }
        let wt_dir = entry.path();
        if !wt_dir.is_dir() {
            continue;
        }
        // A live run holds `<wt_dir>.lock` exclusively (see
        // acquire_baseline_checkout_lock); only an orphan's lock is free.
        // try_flock -> Some(fd): WE took it (no live owner) -> reclaim.
        // None: held by a live run -> skip. Err: skip.
        let held = match try_flock(checkout_lock_path(&wt_dir), FlockMode::Exclusive) {
            Ok(Some(fd)) => fd,
            _ => continue,
        };
        let _ = std::fs::remove_dir_all(&wt_dir);
        // Unlink the sibling `.lock` WHILE still holding it, then release —
        // mirroring cleanup_shm (run_cargo.rs). try_flock O_CREATs the lock
        // file, so without this every swept orphan would leave 0-byte lock
        // debris. Holding across the unlink preserves serialization: a
        // concurrent same-baseline acquire either sees our held lock (and
        // bails) or, after the unlink, O_CREATs a fresh one — never both. (A
        // release-BEFORE-unlink would let a racer acquire the stale inode
        // while a third run O_CREATs a new one, yielding two "owners".)
        let _ = std::fs::remove_file(checkout_lock_path(&wt_dir));
        drop(held); // release the (now-unlinked) orphan lock
    }
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

/// RAII cleanup for the baseline checkout: drop removes the temp dir so an
/// early return, `?`, or panic in [`noise_dual_run`] — or a record-and-survive
/// Ctrl-C — never leaks it. The checkout is a plain dir (no `.git`), so a
/// `remove_dir_all` fully reclaims it.
struct CheckoutGuard {
    wt_dir: PathBuf,
}

impl Drop for CheckoutGuard {
    fn drop(&mut self) {
        // Best-effort teardown: a failure here can't propagate and nothing
        // runs downstream. `remove_dir_all` is recursive and unconditional, so
        // it clears the checkout regardless of what the interrupted build left
        // behind — no "Directory not empty" the old `git worktree remove` hit.
        // A skipped drop on SIGKILL / crash is reclaimed by the next run's
        // `create_checkout_from_tree` pre-checkout self-heal (and the
        // cross-baseline `sweep_stale_checkouts`).
        let _ = std::fs::remove_dir_all(&self.wt_dir);
    }
}

/// Materialize a git `tree_oid` into a PLAIN dir at `wt_dir` (no linked
/// worktree, no `.git`) via `gix`, first self-healing any leftover checkout
/// from an interrupted prior run so the checkout writes into an empty
/// destination. `label` names the checkout in diagnostics (a short commit
/// hash). Both the baseline (its commit's tree) and the HEAD snapshot (HEAD's
/// tree, or a worktree-snapshot tree for a dirty HEAD) feed this one path.
///
/// READ-ONLY on the repo: `gix::worktree::state::checkout` reads objects from
/// the ODB and writes ONLY into `wt_dir` — no worktree registration, and the
/// index is built in memory from `tree_oid` (`index_from_tree`), never
/// persisted. (For a dirty HEAD the ephemeral loose blob/tree objects are
/// written earlier, in [`build_head_snapshot_tree`], not here.)
/// `destination_is_initially_empty` is set because the self-heal above
/// guarantees an empty target, enabling exclusive-create writes.
fn create_checkout_from_tree(
    repo: &gix::Repository,
    wt_dir: &Path,
    tree_oid: gix::ObjectId,
    label: &str,
) -> Result<()> {
    // Clear any leftover checkout from an interrupted prior run (Ctrl-C /
    // SIGKILL / crash skip CheckoutGuard::drop) so the checkout below writes
    // into an empty dir.
    let _ = std::fs::remove_dir_all(wt_dir);
    // gix's checkout writes entries UNDER `wt_dir` but does not create the
    // root itself, so materialize a fresh empty dir for it.
    std::fs::create_dir_all(wt_dir)
        .with_context(|| format!("create {label} checkout dir {}", wt_dir.display()))?;
    let mut index = repo
        .index_from_tree(&tree_oid)
        .with_context(|| format!("build index from {label} tree"))?;
    let mut opts = repo
        .checkout_options(gix::worktree::stack::state::attributes::Source::IdMapping)
        .context("build checkout options")?;
    opts.destination_is_initially_empty = true;
    // Fail loud on any per-entry checkout error rather than silently
    // collecting it — a broken checkout must not be compared.
    opts.keep_going = false;
    // `checkout` requires a `Send` object handle (it may use worker threads);
    // the repo's default handle is `Rc`-backed, so convert to the `Arc`-backed
    // one — the same `into_arc` gix's own clone-checkout uses.
    let objects = repo
        .objects
        .clone()
        .into_arc()
        .context("build thread-safe object handle for checkout")?;
    // Poll perf-delta's interrupt flag so a Ctrl-C aborts the checkout
    // promptly; the caller re-checks `interrupt::caught()` right after and
    // bails before spawning the build, and a partial checkout dir is reclaimed
    // by CheckoutGuard::drop. checkout returns Ok even on abort, so the
    // post-return caught() check is what actually stops the run.
    gix::worktree::state::checkout(
        &mut index,
        wt_dir,
        objects,
        &gix::progress::Discard,
        &gix::progress::Discard,
        &crate::interrupt::INTERRUPTED,
        opts,
    )
    .with_context(|| format!("check out {label} into {}", wt_dir.display()))?;
    Ok(())
}

/// Build the tree to check the HEAD/B side out from, hermetically snapshotting
/// the working tree at command-start.
///
/// Clean HEAD (`dirty == false`): returns `head_tree_oid` unchanged — zero ODB
/// writes; [`create_checkout_from_tree`] then materializes HEAD exactly (the
/// read-only baseline path).
///
/// Dirty HEAD: snapshots the WORKING TREE (what a fresh checkout of the current
/// state would build), so edits during the run don't perturb the B build and
/// the sidecar's `-dirty` label matches the built content. Bases a tree editor
/// on HEAD's tree and, for every path the status walk surfaces — staged AND
/// unstaged AND untracked-non-ignored (both status legs) — RE-READS the
/// worktree file as the single source of truth: present -> write its blob and
/// upsert (`BlobExecutable` / `Link` / `Blob` per the worktree mode); absent
/// -> remove. Rename detection is disabled, so a rename is a plain
/// remove(old) + upsert(new). Ignored files (`target/`, editor swap files) are
/// excluded, so a WIP file must be under `.gitignore` (or stashed) to be kept
/// out of the comparison.
///
/// WRITES ephemeral loose blob+tree objects to the repo's ODB (unlike the
/// read-only baseline path) — they are unreferenced and reclaimed by
/// `git gc`. Only the dirty path writes; a clean HEAD touches nothing.
fn build_head_snapshot_tree(
    repo: &gix::Repository,
    head_tree_oid: gix::ObjectId,
    dirty: bool,
) -> Result<gix::ObjectId> {
    if !dirty {
        return Ok(head_tree_oid);
    }
    use gix::bstr::{BString, ByteSlice};
    use gix::objs::tree::EntryKind;
    use std::os::unix::fs::PermissionsExt;

    let workdir = repo
        .workdir()
        .context("cannot snapshot a dirty HEAD without a working tree")?
        .to_path_buf();
    let mut editor = repo
        .edit_tree(head_tree_oid)
        .context("open a tree editor on HEAD for the dirty snapshot")?;

    // Both status legs (HEAD-vs-index staged + index-vs-worktree unstaged +
    // untracked), renames off so each change is a plain path add/modify/delete.
    let status = repo
        .status(gix::progress::Discard)
        .context("configure working-tree status for the snapshot")?
        .untracked_files(gix::status::UntrackedFiles::Files)
        .index_worktree_rewrites(None)
        .tree_index_track_renames(gix::status::tree_index::TrackRenames::Disabled);

    let mut changed = 0usize;
    for item in status
        .into_iter(Vec::new())
        .context("start the working-tree status walk for the snapshot")?
    {
        let item = item.context("read a working-tree status entry")?;
        // worktree-file-is-truth: the status leg only tells us WHICH path
        // changed; re-derive content/mode/existence from the filesystem so a
        // staged-then-reverted file snapshots as its (reverted) worktree state.
        let rela: BString = item.location().to_owned();
        let abs = workdir.join(gix::path::from_bstr(rela.as_bstr()).as_ref());
        match std::fs::symlink_metadata(&abs) {
            Ok(md) if md.file_type().is_symlink() => {
                let target = std::fs::read_link(&abs)
                    .with_context(|| format!("read symlink {abs:?} for snapshot"))?;
                let blob = repo
                    .write_blob(gix::path::into_bstr(target).as_ref())
                    .with_context(|| format!("write symlink blob for {rela}"))?
                    .detach();
                editor
                    .upsert(rela.as_bstr(), EntryKind::Link, blob)
                    .with_context(|| format!("snapshot symlink {rela}"))?;
                changed += 1;
            }
            Ok(md) if md.is_file() => {
                let bytes =
                    std::fs::read(&abs).with_context(|| format!("read {abs:?} for snapshot"))?;
                let blob = repo
                    .write_blob(&bytes)
                    .with_context(|| format!("write blob for {rela}"))?
                    .detach();
                let kind = if md.permissions().mode() & 0o111 != 0 {
                    EntryKind::BlobExecutable
                } else {
                    EntryKind::Blob
                };
                editor
                    .upsert(rela.as_bstr(), kind, blob)
                    .with_context(|| format!("snapshot {rela}"))?;
                changed += 1;
            }
            // An (empty) untracked directory the walk may emit, or a
            // HEAD-tracked file now REPLACED by a directory: skip the dir-path
            // item. Git tracks no empty dirs; for a file->dir replacement the
            // dir's own files (upsert-ing `foo/bar`) auto-convert `foo` to a
            // tree, and a remove("foo") here could delete that fresh tree.
            Ok(md) if md.is_dir() => continue,
            // Absent (deleted), or a non-file at that path: drop it from the
            // snapshot tree.
            _ => {
                editor
                    .remove(rela.as_bstr())
                    .with_context(|| format!("drop {rela} from snapshot"))?;
                changed += 1;
            }
        }
    }
    let tree = editor
        .write()
        .context("write the dirty-HEAD snapshot tree")?
        .detach();
    eprintln!(
        "perf-delta: HEAD is dirty; snapshotting {changed} uncommitted change(s) \
         (incl. untracked non-ignored files) so the comparison is hermetic"
    );
    Ok(tree)
}

/// Resolve the two trees the run-producing paths check out: the baseline
/// commit's tree, and the HEAD snapshot tree (HEAD's tree for a clean HEAD, or
/// a working-tree snapshot for a dirty HEAD — see [`build_head_snapshot_tree`]).
/// Computed ONCE at command start so the HEAD/B side is a point-in-time capture
/// that edits during the (long) run cannot perturb.
fn resolve_run_trees(
    repo: &gix::Repository,
    baseline_oid: gix::ObjectId,
) -> Result<(gix::ObjectId, gix::ObjectId)> {
    let baseline_tree = repo
        .find_commit(baseline_oid)
        .context("find baseline commit")?
        .tree_id()
        .context("resolve baseline tree")?
        .detach();
    let head_oid = repo.head_id().context("resolve HEAD")?.detach();
    let head_tree = repo
        .find_commit(head_oid)
        .context("find HEAD commit")?
        .tree_id()
        .context("resolve HEAD tree")?
        .detach();
    let dirty = ktstr::test_support::repo_is_dirty(repo) == Some(true);
    let head_snapshot_tree = build_head_snapshot_tree(repo, head_tree, dirty)?;
    Ok((baseline_tree, head_snapshot_tree))
}

/// Absolutize a relative kernel-source path against the MAIN tree so the
/// baseline child resolves the SAME kernel HEAD does. The baseline child
/// runs with its cwd in the checkout (a temp dir), where a relative
/// `--kernel ../linux` would resolve against the checkout instead of the
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

/// `--noise-adjust N` run-production: loops the two-checkout production, writing each
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
/// (baseline=A, HEAD=B), not by leaf. Both checkouts (baseline + HEAD
/// snapshot) are created once and removed on return via [`CheckoutGuard`],
/// so the HEAD side is hermetic like the baseline. A non-zero child exit is
/// logged, not fatal (some `performance_mode` failures are expected; the
/// sidecars that landed are still comparable).
#[allow(clippy::too_many_arguments)]
fn noise_dual_run(
    repo: &gix::Repository,
    repo_root: &Path,
    baseline_tree: gix::ObjectId,
    head_tree: gix::ObjectId,
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
    let wt_dir = checkout_dir(&std::env::temp_dir(), baseline_short);
    let head_wt_dir = checkout_dir(&std::env::temp_dir(), head_short);
    let kernel_arg = resolve_kernel_for_children(repo_root, kernel);

    // Serialize concurrent runs that share a checkout path: create_checkout_from_tree
    // remove_dir_all's any leftover, so an exclusive advisory flock makes a live
    // concurrent run bail cleanly. Both locks declared before both guards so each
    // lock drops after its checkout is removed (LIFO: guards drop first).
    let _baseline_lock = acquire_baseline_checkout_lock(&wt_dir, baseline_short)?;
    let _head_lock = acquire_baseline_checkout_lock(&head_wt_dir, head_short)?;
    // Install each guard before its checkout starts so an error or interrupt
    // during materialization cannot leak a partial directory.
    let _baseline_guard = CheckoutGuard {
        wt_dir: wt_dir.clone(),
    };
    create_checkout_from_tree(repo, &wt_dir, baseline_tree, baseline_short)?;
    if crate::interrupt::caught().is_some() {
        return Ok(());
    }
    let _head_guard = CheckoutGuard {
        wt_dir: head_wt_dir.clone(),
    };
    create_checkout_from_tree(repo, &head_wt_dir, head_tree, head_short)?;
    // Ctrl-C during either checkout: bail before the run loop. The guards remove
    // the (possibly partial) checkouts.
    if crate::interrupt::caught().is_some() {
        return Ok(());
    }

    for i in 0..runs {
        // Per-run leaves: two runs of the same test+config share a variant-hashed
        // filename, so a single leaf would overwrite all but the last; distinct
        // dirs keep all N. The compare still partitions by project_commit
        // (baseline vs HEAD), not by leaf name.
        let base_leaf = noise_run_leaf(&runs_root_abs, "base", baseline_short, i);
        let head_leaf = noise_run_leaf(&runs_root_abs, "head", head_short, i);
        println!("perf-delta --noise-adjust: run {}/{runs}", i + 1);

        let mut baseline_command = Command::new("cargo");
        baseline_command
            .current_dir(&wt_dir)
            .args(perf_test_argv(
                &kernel_arg,
                filter,
                profile,
                nextest_profile,
                passthrough,
            ))
            .envs(baseline_child_env(&base_leaf, baseline_short));
        let bs = crate::interrupt::run_status(baseline_command)
            .with_context(|| format!("spawn baseline `cargo ktstr test` (noise run {i})"))?;
        if !bs.success() {
            eprintln!("perf-delta --noise-adjust: warning: baseline run {i} exited {bs}");
        }
        if crate::interrupt::caught().is_some() {
            break;
        }

        let mut head_command = Command::new("cargo");
        head_command
            .current_dir(&head_wt_dir)
            .args(perf_test_argv(
                &kernel_arg,
                filter,
                profile,
                nextest_profile,
                passthrough,
            ))
            .env(ktstr::KTSTR_PERF_ONLY_ENV, "1")
            .env(ktstr::KTSTR_SIDECAR_DIR_ENV, &head_leaf)
            .env(ktstr::KTSTR_PROJECT_COMMIT_ENV, head_short);
        let hs = crate::interrupt::run_status(head_command)
            .with_context(|| format!("spawn HEAD `cargo ktstr test` (noise run {i})"))?;
        if !hs.success() {
            eprintln!("perf-delta --noise-adjust: warning: HEAD run {i} exited {hs}");
        }
        if crate::interrupt::caught().is_some() {
            break;
        }
    }
    Ok(())
}

/// Per-run sidecar leaf for the `--noise-adjust` production: `side` is `"base"`
/// or `"head"`, `short` the commit label, `i` the run index. [`noise_dual_run`]
/// writes into these and the empty-perf-set guard in [`run`] counts them, so both
/// must agree on the naming — hence this single source.
fn noise_run_leaf(runs_root_abs: &Path, side: &str, short: &str, i: usize) -> PathBuf {
    runs_root_abs.join(format!("perf-delta-noise-{side}-{short}-{i}"))
}

/// Count `*.ktstr.json` sidecars directly in `dir` (non-recursive). A missing or
/// unreadable dir counts as `0`. The `--noise-adjust` empty-perf-set guard sums
/// this over the baseline run leaves to detect "no `performance_mode` test
/// produced any sidecar" and exit cleanly, instead of running the compare into a
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

/// CLI args for `cargo ktstr perf-delta`.
pub(crate) struct PerfDeltaArgs<'a> {
    /// `--base <commit>` override (skips merge-base).
    pub base: Option<&'a str>,
    /// `--base-ref <ref>` override (merge-base against this ref).
    pub base_ref: Option<&'a str>,
    /// Nextest `-E` filter narrowing within the perf-mode set.
    pub filter: Option<&'a str>,
    /// `--relevant` — additionally narrow the compared perf set to the tests
    /// the `base..HEAD` diff (UNIONed with the working tree) touches, computed
    /// from the SAME `base` / `base_ref` / `default_branch` as the baseline.
    /// Intersected with `filter`. A broad / unattributable change does not
    /// narrow (compares everything); a docs-only change narrows to nothing.
    pub relevant: bool,
    /// Branch to merge-base against when no override / env is present.
    pub default_branch: &'a str,
    /// `--kernel <SPEC>` — required with `--noise-adjust` (the kernel the
    /// baseline and HEAD perf tests boot). `None` is valid only for the
    /// cached-baseline path (no `--noise-adjust`), which runs no tests.
    pub kernel: Option<&'a str>,
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
    /// `--all-metrics` — show every compared metric row (stable + noisy
    /// included). Default (false) shows only meaningful rows. Display-only:
    /// never changes the failure gate.
    pub all_metrics: bool,
    /// `--fail-threshold <N>` — fail only when >= N metrics regress
    /// (`None` = 5, tolerating a handful of noisy regressions; `1` restores
    /// fail-on-any; `0` = never fail on the count, leaving `--must-fail` as
    /// the only failing condition).
    pub fail_threshold: Option<usize>,
    /// `--must-fail <M1,M2,...>` — metric registry names that fail the run
    /// if ANY regresses, regardless of the count gate (ORed on top). The
    /// raw comma-separated string; validated + split in [`run`].
    pub must_fail: Option<&'a str>,
    /// `--noise-adjust <N>` — run each side N times and decide significance from
    /// whether the two sides are SEPARATED (a Welch two-sample t-test, or fully
    /// disjoint `[min, max]` bands) AND the delta is MATERIAL (the registry
    /// dual-gate). Commit axis only — it PRODUCES N runs per side (each checked
    /// out into a plain gix checkout). `None` runs no tests and compares sidecars
    /// already pooled from a prior run or a downloaded artifact.
    pub noise_adjust: Option<usize>,
    /// `--noise-spread-threshold <PCT>` — the per-side relative-spread limit (in
    /// percent) above which `--noise-adjust` ANNOTATES a metric with an advisory
    /// "noisy spread" flag. Advisory only: it never suppresses a separated +
    /// material regression. Defaults to `5.0` when `--noise-adjust` is set.
    pub noise_spread_threshold: Option<f64>,
    /// `--profile <NAME>` — the scheduler-under-test's cargo BUILD profile,
    /// forwarded to BOTH sides' `cargo ktstr test` so baseline and HEAD
    /// build the scheduler identically. `None` leaves it at the release
    /// default. Only meaningful on the run-producing path (`--noise-adjust`);
    /// [`run`] rejects it on the cached-baseline path that runs no tests.
    pub profile: Option<&'a str>,
    /// `--nextest-profile <NAME>` — the nextest test profile, forwarded to
    /// BOTH sides' `cargo ktstr test`. Only meaningful on the run-producing
    /// paths; [`run`] rejects it on the paths that run no tests.
    pub nextest_profile: Option<&'a str>,
    /// cargo/nextest flags forwarded verbatim to BOTH sides' `cargo ktstr
    /// test` → `cargo nextest run` (e.g. `--features integration,wprof`);
    /// no `--` separator required (but native flags must precede it — see
    /// the CLI `args` doc). Only meaningful on the run-producing path
    /// (`--noise-adjust`); [`run`] rejects a non-empty passthrough on the
    /// cached-baseline path (which runs no tests) rather than silently
    /// dropping it.
    pub passthrough: &'a [String],
}

/// A/B-compare two perf runs via the existing `compare_partitions`
/// engine, returning its exit code (non-zero on a regression). The
/// compare pairs per-scenario and applies each metric's polarity +
/// abs/rel thresholds on the COMMIT axis: HEAD vs a baseline commit,
/// partitioned by `project_commit`. Sidecars come either from the pool
/// already (cached-baseline) or from `--noise-adjust`, which [`noise_dual_run`]
/// PRODUCES by checking BOTH commits out into plain gix checkouts (no `.git`)
/// and running their `performance_mode`-only, `-E`-narrowed tests. (Cross-config
/// questions — e.g. scheduler A vs B — are answered in-test via the
/// Verdict DSL's `better_across_phases`, not by this command.)
pub(crate) fn run(args: &PerfDeltaArgs<'_>) -> Result<i32> {
    // Regression sensitivity from --threshold / --policy via the shared
    // ComparisonPolicy resolver.
    let policy = cli::ComparisonPolicy::from_cli_flags(args.threshold, args.policy)
        .context("resolve --threshold / --policy")?;

    // Operator failure-gate + render options. --must-fail is validated UP
    // FRONT (before running any tests) — see parse_must_fail.
    let gate = cli::GateOptions {
        fail_threshold: args.fail_threshold,
        must_fail: parse_must_fail(args.must_fail, args.noise_adjust.is_some())?,
        show_all: args.all_metrics,
    };

    // Fail loud rather than SILENTLY DROP the build-shaping flags: the
    // cargo/nextest passthrough AND `--profile` / `--nextest-profile` only
    // reach a `cargo ktstr test` run on the run-PRODUCING path
    // (--noise-adjust N). The cached-baseline compare runs no tests, so any of
    // these there would vanish with no effect — surface the misuse instead.
    let runs_tests = args.noise_adjust.is_some();
    if !runs_tests
        && (!args.passthrough.is_empty()
            || args.profile.is_some()
            || args.nextest_profile.is_some())
    {
        anyhow::bail!(
            "build-shaping flags (cargo/nextest passthrough {:?}, --profile {:?}, \
             --nextest-profile {:?}) are only applied by the run-producing path \
             (--noise-adjust N); this invocation compares already-pooled sidecars \
             and runs no tests, so they would be ignored. Add --noise-adjust N, \
             or drop the flags.",
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

    // `--relevant`: fold the change-scoped filterset into the effective `-E`
    // used for BOTH sides. `relevant_test_filter` returns `None` for a broad /
    // unattributable change (do not narrow -> fall back to the user's `--filter`)
    // and `Some("none()")` for a docs-only change (narrow to nothing -> the run
    // produces no sidecars and the empty-perf-set guard below exits cleanly (0)). The
    // composed expression intersects (`&`) so it narrows, never widens.
    let relevant_expr = if args.relevant {
        crate::affected::relevant_test_filter(args.base, args.base_ref, args.default_branch)
            .context("compute --relevant perf-delta test set")?
    } else {
        None
    };
    let effective_filter: Option<String> = match (relevant_expr.as_deref(), args.filter) {
        (Some(rel), Some(user)) => Some(format!("({rel}) & ({user})")),
        (Some(rel), None) => Some(rel.to_string()),
        (None, user) => user.map(str::to_string),
    };
    let effective_filter = effective_filter.as_deref();
    println!(
        "  perf tests: {}",
        effective_filter.unwrap_or("all performance_mode tests")
    );

    // Noise-adjusted axis: produce N runs per side (looped run production into
    // per-run leaves), then compare from the observed spread instead of a fixed
    // threshold. Commit axis only.
    if let Some(n) = args.noise_adjust {
        let kernel = args
            .kernel
            .context("--noise-adjust requires --kernel (the kernel both perf runs boot)")?;
        // Snapshot both trees ONCE at command start (hermetic HEAD/B side).
        let (baseline_tree, head_snapshot_tree) = resolve_run_trees(&repo, baseline_oid)?;
        // Reclaim checkouts orphaned by prior interrupted runs against
        // OTHER baselines (the per-baseline self-heal misses them).
        sweep_stale_checkouts(&std::env::temp_dir());
        // Everything through stale-checkout reclamation is preflight. Enter
        // cleanup ownership immediately before taking the live checkout locks;
        // the top-level guard remains installed until both CheckoutGuards have
        // dropped and then re-raises any caught signal.
        crate::interrupt::enter_cleanup_phase().context("enter cleanup phase for perf-delta")?;
        noise_dual_run(
            &repo,
            &cwd,
            baseline_tree,
            head_snapshot_tree,
            &baseline,
            &head,
            kernel,
            effective_filter,
            args.profile,
            args.nextest_profile,
            args.passthrough,
            n,
        )?;
        if crate::interrupt::caught().is_some() {
            return Ok(0);
        }
        // If the perf run produced no sidecars at all — no
        // `#[ktstr_test(performance_mode)]` tests are defined, or `-E` / `--relevant`
        // narrowed the selection to none — there is nothing to compare. Exit cleanly
        // (Ok(0)) rather than letting compare_partitions_noise bail on an empty pool.
        // All N runs share the baseline selection, so the total across the baseline
        // leaves is zero iff no perf test ran. (The deleted `--dual-run` path carried
        // the same guard on its single leaf.)
        let runs_root_abs = cwd.join(ktstr::test_support::runs_root());
        let baseline_sidecars: usize = (0..n)
            .map(|i| count_sidecars(&noise_run_leaf(&runs_root_abs, "base", &baseline, i)))
            .sum();
        if baseline_sidecars == 0 {
            println!(
                "perf-delta: no performance_mode sidecars produced at baseline {baseline} \
                 — nothing to compare (define #[ktstr_test(performance_mode)] tests, or widen \
                 the -E filter, to enable the delta)"
            );
            return Ok(0);
        }
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
            &gate,
        );
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
    cli::compare_partitions(&filter_a, &filter_b, None, &policy, None, &gate)
}

/// Parse + validate the `--must-fail` CSV: split on commas, trim, skip
/// empties, and reject every name that cannot fire the gate under the ACTIVE
/// comparison mode, so a name that would silently never fail errors before any
/// tests run. Rejected:
/// - unknown metric names;
/// - render-suppressed rate COMPONENTS (never surface as any finding);
/// - names whose value never reaches the aggregate compare in this mode
///   (`MetricDef::gates_aggregate` is false) — per-phase-only metrics (rejected
///   in BOTH modes) and whole-run distribution metrics (rejected on the scalar
///   path only; they ARE read per-run under `--noise-adjust`);
/// - on the default (scalar) comparison only, informational metrics (registry
///   polarity has no regression direction, so `compare_partitions` classifies
///   them `Informational`, never `Regression`) — ACCEPTED under `--noise-adjust`
///   (`noise_adjust == true`) because a per-test direction override can then
///   classify one as a regression (see the noise classifier
///   `compare_partitions_noise`).
///
/// `metric_def` is `Some` past the unknown-name gate, and a
/// `classify_direction()` of `None` is exactly `Polarity::Informational`.
fn parse_must_fail(csv: Option<&str>, noise_adjust: bool) -> Result<Vec<String>> {
    let Some(csv) = csv else {
        return Ok(Vec::new());
    };
    let mut names = Vec::new();
    for raw in csv.split(',') {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        let Some(def) = cli::metric_def(name) else {
            anyhow::bail!(
                "--must-fail: unknown metric {name:?} \
                 (see `cargo ktstr stats list-metrics`)"
            );
        };
        if cli::is_render_suppressed_component(name) {
            anyhow::bail!(
                "--must-fail: {name:?} is an internal rate COMPONENT that never \
                 surfaces as a regression — gate on its user-facing rate metric instead"
            );
        }
        if !def.gates_aggregate(noise_adjust) {
            // The value never lands on a compared row in this mode, so the gate
            // could never fire. Tailor the hint: PerRunDistribution IS gateable
            // under --noise-adjust; PerPhase never is (per-phase-only).
            let hint = if def.gates_aggregate(true) {
                " — it only reports per-run; add --noise-adjust to gate on it"
            } else {
                " — it is a per-phase-only metric that never produces an aggregate \
                 regression; assert it per-phase in the test's Verdict DSL instead"
            };
            anyhow::bail!("--must-fail: {name:?} never reaches the aggregate comparison{hint}");
        }
        if !noise_adjust && def.classify_direction().is_none() {
            anyhow::bail!(
                "--must-fail: {name:?} is an informational metric with no regression \
                 direction — it can never fail the default comparison. Add \
                 --noise-adjust (a per-test direction override can then make it gate), \
                 or gate on a directional metric"
            );
        }
        names.push(name.to_string());
    }
    Ok(names)
}

use ktstr::cli;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_must_fail_trims_skips_empties_and_rejects_bad_names() {
        // Split + trim + skip empties (leading / trailing / double comma).
        // worst_spread + avg_dsq_depth are both directional (LowerBetter), so
        // they parse under either mode; use the scalar mode here.
        let ok = parse_must_fail(Some(" worst_spread , ,avg_dsq_depth,"), false)
            .expect("known directional metrics parse");
        assert_eq!(
            ok,
            vec!["worst_spread".to_string(), "avg_dsq_depth".to_string()]
        );
        // None -> empty set (both modes).
        assert!(parse_must_fail(None, false).unwrap().is_empty());
        assert!(parse_must_fail(None, true).unwrap().is_empty());
        // Unknown name -> Err.
        assert!(parse_must_fail(Some("definitely_not_a_metric"), false).is_err());
        // A render-suppressed rate COMPONENT -> Err in BOTH modes (it can never
        // surface as any finding, so gating on it is always a silent no-op).
        assert!(
            parse_must_fail(Some("total_taobench_ops"), false).is_err(),
            "a render-suppressed rate component must be rejected on the scalar path"
        );
        assert!(
            parse_must_fail(Some("total_taobench_ops"), true).is_err(),
            "a render-suppressed rate component must be rejected under --noise-adjust too"
        );
        // An informational metric (no regression direction) can never gate the
        // default comparison -> Err on the scalar path, so a typo-safe operator
        // learns up front instead of shipping a silently-inert gate.
        assert!(
            parse_must_fail(Some("total_ttwu_count"), false).is_err(),
            "an informational metric must be rejected on the scalar path"
        );
        // ...but under --noise-adjust a per-test direction override can classify
        // it as a regression, so it is accepted then.
        assert_eq!(
            parse_must_fail(Some("total_ttwu_count"), true)
                .expect("informational metric accepted under --noise-adjust"),
            vec!["total_ttwu_count".to_string()],
        );
        // A PerPhase metric (per-phase-only, accessor None, no run-level
        // producer) never lands on an aggregate row, so it can never fire the
        // gate in EITHER mode -> Err both. (Directional (LowerBetter), so this
        // is the gates_aggregate rejection, not the informational one.)
        assert!(
            parse_must_fail(Some("wakeup_p99_latency_us"), false).is_err(),
            "a per-phase-only metric must be rejected on the scalar path"
        );
        assert!(
            parse_must_fail(Some("wakeup_p99_latency_us"), true).is_err(),
            "a per-phase-only metric must be rejected under --noise-adjust too"
        );
        // A PerRunDistribution metric is gated out of the cross-run fold, so it
        // is inert on the scalar compare (Err) but IS read per-run by the noise
        // compare (Ok).
        assert!(
            parse_must_fail(Some("wakeup_p99_latency_us_whole"), false).is_err(),
            "a whole-run distribution metric cannot gate the scalar compare"
        );
        assert_eq!(
            parse_must_fail(Some("wakeup_p99_latency_us_whole"), true)
                .expect("whole-run distribution metric gates under --noise-adjust"),
            vec!["wakeup_p99_latency_us_whole".to_string()],
        );
    }

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

    // ---- checkout / child-run pure helpers ----

    #[test]
    fn checkout_dir_under_temp_root() {
        assert_eq!(
            checkout_dir(Path::new("/tmp"), "abc1234"),
            Path::new("/tmp/ktstr-perf-delta-co-abc1234"),
        );
    }

    #[test]
    fn noise_run_leaf_names_per_side_per_run() {
        let root = Path::new("/work/target/ktstr");
        assert_eq!(
            noise_run_leaf(root, "base", "abc1234", 0),
            Path::new("/work/target/ktstr/perf-delta-noise-base-abc1234-0"),
        );
        assert_eq!(
            noise_run_leaf(root, "head", "def5678", 2),
            Path::new("/work/target/ktstr/perf-delta-noise-head-def5678-2"),
        );
    }

    #[test]
    fn count_sidecars_counts_ktstr_json_and_zero_for_missing() {
        let base = std::env::temp_dir().join(format!("ktstr-pd-count-{}", std::process::id()));
        // A missing directory counts as zero (the no-perf-tests-ran case the
        // empty-perf-set guard exits cleanly on).
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
    fn baseline_child_env_sets_perf_only_sidecar_dir_and_project_commit() {
        let leaf = Path::new("/work/target/ktstr/perf-delta-baseline-abc1234");
        let env = baseline_child_env(leaf, "abc1234");
        assert_eq!(
            env,
            vec![
                ("KTSTR_PERF_ONLY", "1".to_string()),
                (
                    "KTSTR_SIDECAR_DIR",
                    "/work/target/ktstr/perf-delta-baseline-abc1234".to_string(),
                ),
                ("KTSTR_PROJECT_COMMIT", "abc1234".to_string()),
            ],
        );
        // Pin the env-var names against the library constants so a rename
        // of any keeps the child pointed at the right vars. The explicit
        // project_commit is what makes the sidecar record the SAME label the
        // compare filters the pool on (the -dirty-mismatch fix).
        assert_eq!(env[0].0, ktstr::KTSTR_PERF_ONLY_ENV);
        assert_eq!(env[1].0, ktstr::KTSTR_SIDECAR_DIR_ENV);
        assert_eq!(env[2].0, ktstr::KTSTR_PROJECT_COMMIT_ENV);
    }

    #[test]
    fn sweep_reclaims_unlocked_but_keeps_locked_checkouts() {
        // Two orphan-shaped baseline checkout dirs under a throwaway temp
        // root: one whose sibling `.lock` is held (a live run), one free.
        // The cross-baseline sweep must reclaim the free one (pure
        // `remove_dir_all`, no git) and NEVER touch the locked one — pins
        // the lock-gated fs reclamation.
        let base = std::env::temp_dir().join(format!("ktstr-pd-sweep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("mk temp root");
        let unlocked = base.join("ktstr-perf-delta-co-aaaaaaa");
        let locked = base.join("ktstr-perf-delta-co-bbbbbbb");
        std::fs::create_dir(&unlocked).expect("mk unlocked checkout");
        std::fs::create_dir(&locked).expect("mk locked checkout");
        // Hold the locked checkout's advisory lock for the sweep's duration.
        let held = try_flock(checkout_lock_path(&locked), FlockMode::Exclusive)
            .expect("flock call")
            .expect("lock free initially");

        sweep_stale_checkouts(&base);

        assert!(
            !unlocked.exists(),
            "an unlocked orphan checkout is reclaimed"
        );
        assert!(
            !checkout_lock_path(&unlocked).exists(),
            "the reclaimed orphan's sibling .lock is removed too (no lock debris)"
        );
        assert!(locked.exists(), "a locked (live) checkout is never removed");
        assert!(
            checkout_lock_path(&locked).exists(),
            "a live checkout's held .lock is left intact"
        );

        drop(held);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn checkout_guard_drop_removes_the_checkout_dir() {
        // RAII cleanup: dropping the guard removes the checkout dir
        // recursively (incl. any build artifacts an interrupted run left) —
        // the plain-fs replacement for the old `git worktree remove`, and the
        // reason a Ctrl-C never leaks a /tmp checkout.
        let base = std::env::temp_dir().join(format!("ktstr-pd-guard-{}", std::process::id()));
        let wt = base.join("wt");
        std::fs::create_dir_all(wt.join("scx-ktstr")).expect("mk checkout subtree");
        std::fs::write(wt.join("scx-ktstr/build.rs"), "x").expect("write checkout file");
        assert!(wt.exists(), "checkout present before drop");
        {
            let _guard = CheckoutGuard { wt_dir: wt.clone() };
        }
        assert!(!wt.exists(), "CheckoutGuard::drop removes the checkout dir");
        // Drop on an already-absent dir is a harmless no-op (the SIGKILL /
        // already-swept case).
        {
            let _guard = CheckoutGuard { wt_dir: wt.clone() };
        }
        assert!(!wt.exists(), "drop on an absent checkout stays a no-op");
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn create_checkout_from_tree_materializes_the_tree() {
        let _serial = crate::interrupt::test_serial_guard();
        // Build a throwaway repo with one blob+tree, then check that
        // create_checkout_from_tree materializes it into a PLAIN dir with no
        // `.git`. Pins our wiring (index_from_tree -> checkout_options ->
        // gix::worktree::state::checkout), not gix itself. A committed HEAD is
        // NOT needed — the fn checks out a tree oid directly.
        let base = std::env::temp_dir().join(format!("ktstr-pd-gixco-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let repo_dir = base.join("repo");
        std::fs::create_dir_all(&repo_dir).expect("mk repo dir");

        let repo = gix::init(&repo_dir).expect("gix::init");
        let blob = repo.write_blob(b"baseline\n").expect("write blob").detach();
        let tree = gix::objs::Tree {
            entries: vec![gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Blob.into(),
                filename: "file.txt".into(),
                oid: blob,
            }],
        };
        let tree_id: gix::ObjectId = repo.write_object(&tree).expect("write tree").detach();

        let wt = base.join("checkout");
        create_checkout_from_tree(&repo, &wt, tree_id, "baseline").expect("gix checkout");
        assert_eq!(
            std::fs::read_to_string(wt.join("file.txt")).expect("checked-out file present"),
            "baseline\n",
            "the tree's file must be materialized into the plain checkout",
        );
        assert!(
            !wt.join(".git").exists(),
            "the checkout is a PLAIN dir — no .git linkage, so cleanup is a plain remove_dir_all",
        );

        // Self-heal: a leftover populated checkout is cleared before re-checkout.
        std::fs::write(wt.join("file.txt"), b"stale\n").expect("dirty the leftover");
        std::fs::write(wt.join("untracked.tmp"), b"junk\n").expect("leftover untracked");
        create_checkout_from_tree(&repo, &wt, tree_id, "baseline").expect("re-checkout self-heals");
        assert_eq!(
            std::fs::read_to_string(wt.join("file.txt")).expect("file present after re-checkout"),
            "baseline\n",
            "re-checkout must overwrite a stale leftover with the tree content",
        );
        assert!(
            !wt.join("untracked.tmp").exists(),
            "pre-checkout self-heal (remove_dir_all) clears a leftover's untracked debris",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Throwaway git repo: commit `files` (HEAD + a persisted index + matching
    /// worktree files, so `status` starts clean and any later worktree edit is
    /// the only change). Returns the repo and HEAD's tree oid. Mirrors the
    /// sidecar commit fixtures.
    fn init_committed_repo(
        dir: &std::path::Path,
        files: &[(&str, &[u8])],
    ) -> (gix::Repository, gix::ObjectId) {
        std::fs::create_dir_all(dir).expect("mk repo dir");
        let mut repo = gix::init(dir).expect("gix::init");
        let _ = repo
            .committer_or_set_generic_fallback()
            .expect("committer fallback");
        {
            use gix::config::tree::gitoxide;
            let mut cfg = gix::config::File::new(gix::config::file::Metadata::api());
            cfg.set_raw_value(gitoxide::Author::NAME_FALLBACK, "ktstr-test")
                .expect("author name fallback");
            cfg.set_raw_value(
                gitoxide::Author::EMAIL_FALLBACK,
                "ktstr-test@example.invalid",
            )
            .expect("author email fallback");
            repo.config_snapshot_mut().append(cfg);
        }
        let mut entries: Vec<gix::objs::tree::Entry> = files
            .iter()
            .map(|(name, content)| {
                let oid = repo.write_blob(content).expect("write blob").detach();
                gix::objs::tree::Entry {
                    mode: gix::objs::tree::EntryKind::Blob.into(),
                    filename: (*name).into(),
                    oid,
                }
            })
            .collect();
        // git tree entries must be sorted by filename.
        entries.sort_by(|a, b| a.filename.cmp(&b.filename));
        let tree = gix::objs::Tree { entries };
        let tree_id: gix::ObjectId = repo.write_object(&tree).expect("write tree").detach();
        repo.commit("HEAD", "init", tree_id, std::iter::empty::<gix::ObjectId>())
            .expect("commit");
        // Persist an index from the tree + write the worktree files so status
        // sees a clean baseline until the test edits the worktree.
        let mut idx = repo.index_from_tree(&tree_id).expect("index_from_tree");
        idx.write(gix::index::write::Options::default())
            .expect("write index");
        for (name, content) in files {
            std::fs::write(dir.join(name), content).expect("write worktree file");
        }
        (repo, tree_id)
    }

    /// The load-bearing dirty-HEAD property: a dirty HEAD snapshots the WORKTREE
    /// bytes, not HEAD's — so perf-delta builds the WIP, and the recorded
    /// `-dirty` label matches the built content.
    #[test]
    fn dirty_head_snapshot_captures_worktree_bytes_not_head() {
        let _serial = crate::interrupt::test_serial_guard();
        let base = std::env::temp_dir().join(format!("ktstr-pd-snap-dirty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let repo_dir = base.join("repo");
        let (repo, head_tree) = init_committed_repo(&repo_dir, &[("file.txt", b"v1\n")]);
        // Modify the worktree (unstaged) so HEAD/index=v1 but worktree=v2.
        std::fs::write(repo_dir.join("file.txt"), b"v2\n").expect("dirty the worktree");

        let snap = build_head_snapshot_tree(&repo, head_tree, true).expect("build snapshot");
        assert_ne!(
            snap, head_tree,
            "a dirty snapshot must differ from HEAD's tree"
        );
        let wt = base.join("checkout");
        create_checkout_from_tree(&repo, &wt, snap, "head").expect("checkout snapshot");
        assert_eq!(
            std::fs::read_to_string(wt.join("file.txt")).expect("snapshot file"),
            "v2\n",
            "the snapshot holds the WORKTREE bytes (v2), not HEAD's (v1)",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Worktree-wins over the index: a file STAGED as v2 but then reverted in
    /// the worktree to v1 (== HEAD) snapshots as v1 — the staged v2 is ignored
    /// (perf-delta builds the WORKING TREE, not the index). This is the
    /// documented "snapshot != git stash" corner. Exercises the TreeIndex
    /// (staged) status leg specifically.
    #[test]
    fn dirty_head_snapshot_worktree_wins_over_staged() {
        let base = std::env::temp_dir().join(format!("ktstr-pd-snap-stg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let repo_dir = base.join("repo");
        let (repo, head_tree) = init_committed_repo(&repo_dir, &[("file.txt", b"v1\n")]);
        // Stage v2: persist an index built from a v2 tree (index = v2, HEAD = v1).
        let blob_v2 = repo.write_blob(b"v2\n").expect("write v2 blob").detach();
        let tree_v2 = gix::objs::Tree {
            entries: vec![gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Blob.into(),
                filename: "file.txt".into(),
                oid: blob_v2,
            }],
        };
        let tree_v2_id: gix::ObjectId =
            repo.write_object(&tree_v2).expect("write v2 tree").detach();
        let mut idx = repo
            .index_from_tree(&tree_v2_id)
            .expect("index_from_tree v2");
        idx.write(gix::index::write::Options::default())
            .expect("persist v2 index");
        // Revert the worktree to v1 (== HEAD): now index=v2 but worktree=v1.
        std::fs::write(repo_dir.join("file.txt"), b"v1\n").expect("revert worktree to v1");

        let snap = build_head_snapshot_tree(&repo, head_tree, true).expect("build snapshot");
        assert_eq!(
            snap, head_tree,
            "worktree wins: a staged-then-reverted file snapshots as its worktree \
             content (v1 == HEAD), ignoring the staged v2",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// A clean HEAD returns HEAD's tree verbatim (zero ODB writes) — the
    /// read-only clean-HEAD path, no editor.
    #[test]
    fn clean_head_snapshot_is_head_tree_unchanged() {
        let base = std::env::temp_dir().join(format!("ktstr-pd-snap-clean-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let repo_dir = base.join("repo");
        let (repo, head_tree) = init_committed_repo(&repo_dir, &[("file.txt", b"v1\n")]);
        let snap = build_head_snapshot_tree(&repo, head_tree, false).expect("clean snapshot");
        assert_eq!(
            snap, head_tree,
            "a clean HEAD snapshot is HEAD's tree unchanged (no editor, no writes)",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// An untracked, non-ignored file (uncommitted WIP not yet `git add`ed) IS
    /// in the snapshot; an ignored file is NOT.
    #[test]
    fn dirty_head_snapshot_includes_untracked_but_excludes_ignored() {
        let _serial = crate::interrupt::test_serial_guard();
        let base = std::env::temp_dir().join(format!("ktstr-pd-snap-untr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let repo_dir = base.join("repo");
        let (repo, head_tree) = init_committed_repo(
            &repo_dir,
            &[("file.txt", b"v1\n"), (".gitignore", b"ignore.me\n")],
        );
        std::fs::write(repo_dir.join("new.rs"), b"fn wip() {}\n").expect("untracked WIP");
        std::fs::write(repo_dir.join("ignore.me"), b"scratch\n").expect("ignored scratch");

        let snap = build_head_snapshot_tree(&repo, head_tree, true).expect("build snapshot");
        let wt = base.join("checkout");
        create_checkout_from_tree(&repo, &wt, snap, "head").expect("checkout snapshot");
        assert!(
            wt.join("new.rs").exists(),
            "an untracked non-ignored WIP file is included in the snapshot",
        );
        assert!(
            !wt.join("ignore.me").exists(),
            "an ignored file is excluded from the snapshot",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// A file deleted from the worktree is removed from the snapshot tree.
    #[test]
    fn dirty_head_snapshot_drops_a_deleted_file() {
        let _serial = crate::interrupt::test_serial_guard();
        let base = std::env::temp_dir().join(format!("ktstr-pd-snap-del-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let repo_dir = base.join("repo");
        let (repo, head_tree) = init_committed_repo(
            &repo_dir,
            &[("keep.txt", b"keep\n"), ("gone.txt", b"gone\n")],
        );
        std::fs::remove_file(repo_dir.join("gone.txt")).expect("delete gone.txt");

        let snap = build_head_snapshot_tree(&repo, head_tree, true).expect("build snapshot");
        let wt = base.join("checkout");
        create_checkout_from_tree(&repo, &wt, snap, "head").expect("checkout snapshot");
        assert!(wt.join("keep.txt").exists(), "the kept file survives");
        assert!(
            !wt.join("gone.txt").exists(),
            "a worktree-deleted file is dropped from the snapshot",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// The snapshot preserves the worktree's executable bit (a dropped +x on a
    /// perf script would silently change behavior).
    #[test]
    fn dirty_head_snapshot_preserves_the_worktree_exec_bit() {
        let _serial = crate::interrupt::test_serial_guard();
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("ktstr-pd-snap-exec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let repo_dir = base.join("repo");
        // Committed as a plain (non-exec) blob.
        let (repo, head_tree) = init_committed_repo(&repo_dir, &[("run.sh", b"#!/bin/sh\n")]);
        // Worktree copy: modified AND made executable.
        std::fs::write(repo_dir.join("run.sh"), b"#!/bin/sh\necho hi\n").expect("edit");
        std::fs::set_permissions(
            repo_dir.join("run.sh"),
            std::fs::Permissions::from_mode(0o755),
        )
        .expect("chmod +x");

        let snap = build_head_snapshot_tree(&repo, head_tree, true).expect("build snapshot");
        let wt = base.join("checkout");
        create_checkout_from_tree(&repo, &wt, snap, "head").expect("checkout snapshot");
        assert!(
            std::fs::metadata(wt.join("run.sh"))
                .expect("run.sh present")
                .permissions()
                .mode()
                & 0o111
                != 0,
            "the snapshot preserves the worktree's exec bit (BlobExecutable)",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn baseline_checkout_lock_blocks_a_concurrent_same_baseline_run() {
        // Two perf-delta runs against the same baseline share checkout_lock_path;
        // the exclusive flock must let only ONE proceed so the self-heal never
        // remove_dir_all's a live concurrent run's checkout (the medium-severity
        // regression this lock closes). Also pins the sibling `.lock` path shape
        // and that the lock frees for the next run once released (the
        // interrupted-run recovery path — the OS releases it on process death).
        let wt = std::env::temp_dir().join(format!("ktstr-pd-locktest-wt-{}", std::process::id()));
        assert_eq!(
            checkout_lock_path(&wt),
            PathBuf::from(format!("{}.lock", wt.display())),
            "lock is a sibling <wt_dir>.lock, outside the checkout cleanup removes",
        );
        let lock_path = checkout_lock_path(&wt);
        let _ = std::fs::remove_file(&lock_path);

        let held =
            acquire_baseline_checkout_lock(&wt, "abc1234").expect("first run acquires the lock");
        assert!(
            acquire_baseline_checkout_lock(&wt, "abc1234").is_err(),
            "a concurrent same-baseline run must fail to acquire, not clobber the live checkout",
        );
        drop(held);
        let after = acquire_baseline_checkout_lock(&wt, "abc1234")
            .expect("lock frees for the next run once released");
        drop(after);
        let _ = std::fs::remove_file(&lock_path);
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
        // absolutized so the checkout-cwd baseline child resolves the
        // same tree (the bug: `../linux` resolved against the checkout).
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
    /// perf-delta's checkout path already requires.
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
