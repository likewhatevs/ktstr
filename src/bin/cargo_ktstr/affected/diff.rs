//! gix change detection -> sets of changed repo-root-relative paths.
//!
//! [`changed_paths_committed`] diffs the `base..HEAD` trees; [`changed_paths_worktree`]
//! diffs `HEAD` against the working tree (uncommitted + untracked). A `--relevant`
//! run unions both to get everything changed relative to `base`.
//!
//! Wraps [`gix::Repository::diff_tree_to_tree`] (verified against gix 0.83 /
//! gix-diff 0.63: `diff_tree_to_tree(old, new, None)` returns
//! `Vec<ChangeDetached>` and fills options from the repo config with full path
//! tracking on, so `location`/`source_location` are populated).
//!
//! Both the destination (`location`) and source (`source_location`) of every
//! change are collected. With rename tracking ON a rename is a single
//! `Rewrite` whose source path is reachable ONLY via `source_location`; with
//! it OFF a rename is a Deletion+Addition and both paths appear via
//! `location`. The union is correct either way -- and the source MUST be
//! included, or a scheduler whose `.d` referenced the pre-rename path would be
//! wrongly judged unaffected (a false-negative, the worst outcome).

use std::collections::BTreeSet;

use anyhow::{Context, Result};
use gix::bstr::{BString, ByteSlice};

/// Repo-root-relative paths changed between the `base` and `head` commit trees
/// (the changes that turn `base` into `head`).
pub(crate) fn changed_paths_committed(
    repo: &gix::Repository,
    base: gix::ObjectId,
    head: gix::ObjectId,
) -> Result<BTreeSet<String>> {
    let base_tree = repo
        .find_commit(base)
        .context("find base commit")?
        .tree()
        .context("read base commit tree")?;
    let head_tree = repo
        .find_commit(head)
        .context("find head commit")?
        .tree()
        .context("read head commit tree")?;
    let changes = repo
        .diff_tree_to_tree(Some(&base_tree), Some(&head_tree), None)
        .context("diff base..HEAD trees")?;

    let mut paths = BTreeSet::new();
    for change in &changes {
        paths.insert(
            change
                .location()
                .to_str()
                .context("changed path is not valid UTF-8")?
                .to_owned(),
        );
        paths.insert(
            change
                .source_location()
                .to_str()
                .context("changed source path is not valid UTF-8")?
                .to_owned(),
        );
    }
    Ok(paths)
}

/// Repo-root-relative paths that differ between `HEAD` and the current working
/// tree -- every staged and unstaged change plus untracked files. Unioned with
/// [`changed_paths_committed`] over `base..HEAD`, this yields the full set of
/// paths a local `--relevant` run has touched relative to `base`. A clean tree
/// yields an empty set, so on a committed-only checkout the union degrades to
/// exactly the `base..HEAD` result.
///
/// Rename tracking is forced off on the tree->index leg
/// (`TrackRenames::Disabled`) -- the `status()` default is `AsConfigured`,
/// which tracks staged renames as a single rewrite exposing only the
/// destination via `Item::location`. Disabling it makes a rename surface as a
/// deletion (old path) plus an addition (new path), so both reach the set,
/// matching the source+dest union [`changed_paths_committed`] performs. The
/// index->worktree leg already defaults to no rewrite tracking. Untracked files
/// are emitted individually (`UntrackedFiles::Files`) so a new native source or
/// scheduler input is not collapsed into its parent directory and missed (a
/// false-negative).
pub(crate) fn changed_paths_worktree(repo: &gix::Repository) -> Result<BTreeSet<String>> {
    let iter = repo
        .status(gix::progress::Discard)
        .context("open worktree status")?
        .untracked_files(gix::status::UntrackedFiles::Files)
        .tree_index_track_renames(gix::status::tree_index::TrackRenames::Disabled)
        .index_worktree_options_mut(ktstr::git_status::configure_index_worktree_parallelism)
        .into_iter(Vec::<BString>::new())
        .context("create worktree status iterator")?;

    let mut paths = BTreeSet::new();
    for item in iter {
        let item = item.context("read worktree status item")?;
        paths.insert(
            item.location()
                .to_str()
                .context("worktree-changed path is not valid UTF-8")?
                .to_owned(),
        );
    }
    Ok(paths)
}
