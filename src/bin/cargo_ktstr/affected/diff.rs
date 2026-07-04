//! gix `base..HEAD` tree diff -> the set of changed repo-root-relative paths.
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
use gix::bstr::ByteSlice;

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
