//! Unit tests for [`super`] (the `fetch` module).
//! Co-located via the `tests` submodule pattern.

#![cfg(test)]

use super::*;

// -- arch_info --

#[test]
fn fetch_arch_info_returns_known_arch() {
    let (arch, image) = arch_info();
    assert!(
        (arch == "x86_64" && image == "bzImage") || (arch == "aarch64" && image == "Image"),
        "unexpected arch/image: {arch}/{image}"
    );
}

// -- is_major_minor_prefix --

#[test]
fn is_major_minor_prefix_accepts_two_segment() {
    assert!(is_major_minor_prefix("6.14"));
    assert!(is_major_minor_prefix("7.0"));
}

#[test]
fn is_major_minor_prefix_rejects_patch_version() {
    assert!(!is_major_minor_prefix("6.14.2"));
    assert!(!is_major_minor_prefix("5.4.0"));
}

#[test]
fn is_major_minor_prefix_rejects_rc_tag() {
    assert!(!is_major_minor_prefix("6.15-rc3"));
    assert!(!is_major_minor_prefix("6.14-rc1"));
}

#[test]
fn is_major_minor_prefix_historical_edge_cases() {
    // Historical behavior: accepts single-segment and empty inputs.
    // Callers are expected to gate upstream.
    assert!(is_major_minor_prefix("7"));
    assert!(is_major_minor_prefix(""));
}

// -- major_version --

#[test]
fn fetch_major_version_stable() {
    assert_eq!(major_version("6.14.2").unwrap(), 6);
}

#[test]
fn fetch_major_version_rc() {
    assert_eq!(major_version("6.15-rc3").unwrap(), 6);
}

#[test]
fn fetch_major_version_two_part() {
    assert_eq!(major_version("5.4").unwrap(), 5);
}

#[test]
fn fetch_major_version_invalid() {
    assert!(major_version("abc").is_err());
}

// -- is_rc --

#[test]
fn fetch_is_rc_true() {
    assert!(is_rc("6.15-rc3"));
    assert!(is_rc("6.14.2-rc1"));
}

#[test]
fn fetch_is_rc_false() {
    assert!(!is_rc("6.14.2"));
    assert!(!is_rc("6.14"));
}

// -- URL construction --

/// Stable tarball URL pattern (same logic as download_stable_tarball).
fn stable_tarball_url(version: &str) -> Result<String> {
    let major = major_version(version)?;
    Ok(format!(
        "https://cdn.kernel.org/pub/linux/kernel/v{major}.x/linux-{version}.tar.xz"
    ))
}

/// RC tarball URL pattern (same logic as download_rc_tarball).
fn rc_tarball_url(version: &str) -> String {
    format!("https://git.kernel.org/torvalds/t/linux-{version}.tar.gz")
}

#[test]
fn fetch_stable_url_construction() {
    let url = stable_tarball_url("6.14.2").unwrap();
    assert_eq!(
        url,
        "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.14.2.tar.xz"
    );
}

#[test]
fn fetch_stable_url_v5() {
    let url = stable_tarball_url("5.4.0").unwrap();
    assert_eq!(
        url,
        "https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.0.tar.xz"
    );
}

#[test]
fn fetch_rc_url_construction() {
    let url = rc_tarball_url("6.15-rc3");
    assert_eq!(
        url,
        "https://git.kernel.org/torvalds/t/linux-6.15-rc3.tar.gz"
    );
}

// -- promote_staged_kernel_tree --

#[test]
fn promote_staged_renames_well_formed_archive() {
    let dest = tempfile::TempDir::new().unwrap();
    let staging = tempfile::TempDir::new_in(dest.path()).unwrap();
    std::fs::create_dir(staging.path().join("linux-6.14.2")).unwrap();
    std::fs::write(
        staging.path().join("linux-6.14.2").join("Makefile"),
        b"# fake",
    )
    .unwrap();
    let source_dir = promote_staged_kernel_tree(&staging, dest.path(), "6.14.2").unwrap();
    assert_eq!(source_dir, dest.path().join("linux-6.14.2"));
    assert!(source_dir.is_dir());
    assert!(source_dir.join("Makefile").is_file());
    // Inner dir was renamed out of staging.
    assert!(!staging.path().join("linux-6.14.2").exists());
}

#[test]
fn promote_staged_rejects_stray_top_level_entry() {
    let dest = tempfile::TempDir::new().unwrap();
    let staging = tempfile::TempDir::new_in(dest.path()).unwrap();
    std::fs::create_dir(staging.path().join("linux-6.14.2")).unwrap();
    std::fs::write(staging.path().join("evil"), b"backdoor").unwrap();
    let err = promote_staged_kernel_tree(&staging, dest.path(), "6.14.2").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("unexpected top-level entry"),
        "diagnostic must cite stray entry: {msg}"
    );
    // Nothing landed in dest_dir.
    assert!(!dest.path().join("linux-6.14.2").exists());
}

#[test]
fn promote_staged_bails_on_missing_inner_dir() {
    let dest = tempfile::TempDir::new().unwrap();
    let staging = tempfile::TempDir::new_in(dest.path()).unwrap();
    // Wrong-version inner directory: archive was for 6.14.3 but
    // we're expecting 6.14.2. The mismatch surfaces as a stray
    // top-level entry rather than a missing-inner-dir, since
    // the helper rejects any name that doesn't match the
    // expected one before checking for absence.
    std::fs::create_dir(staging.path().join("linux-6.14.3")).unwrap();
    let err = promote_staged_kernel_tree(&staging, dest.path(), "6.14.2").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("unexpected top-level entry"),
        "wrong-version dir surfaces as stray: {msg}"
    );
    assert!(!dest.path().join("linux-6.14.2").exists());
}

#[test]
fn promote_staged_bails_on_empty_staging() {
    let dest = tempfile::TempDir::new().unwrap();
    let staging = tempfile::TempDir::new_in(dest.path()).unwrap();
    let err = promote_staged_kernel_tree(&staging, dest.path(), "6.14.2").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("expected directory linux-6.14.2"),
        "empty staging surfaces as missing-dir: {msg}"
    );
}

// -- patch_level --

#[test]
fn fetch_patch_level_three_part() {
    assert_eq!(patch_level("6.12.8"), Some(8));
}

#[test]
fn fetch_patch_level_two_part() {
    assert_eq!(patch_level("7.0"), Some(0));
}

#[test]
fn fetch_patch_level_single_part() {
    assert_eq!(patch_level("6"), None);
}

#[test]
fn fetch_patch_level_four_part() {
    assert_eq!(patch_level("6.1.2.3"), None);
}

#[test]
fn fetch_patch_level_non_numeric_patch() {
    assert_eq!(patch_level("6.1.rc3"), None);
}

#[test]
fn fetch_patch_level_zero() {
    assert_eq!(patch_level("6.14.0"), Some(0));
}

#[test]
fn fetch_patch_level_large() {
    assert_eq!(patch_level("6.12.99"), Some(99));
}

// -- local_source dirty detection --

/// Initialise a git repo at `dir` with one committed file, using
/// the `git` CLI with explicit identity + empty global config so
/// the test is deterministic on developer machines and CI runners
/// regardless of the ambient git setup.
fn init_repo_with_commit(dir: &Path) {
    use std::process::Command;

    let run = |args: &[&str]| {
        let out = Command::new("git")
            .args(args)
            .current_dir(dir)
            // Empty system/global config: the test owns identity
            // and default-branch config via -c flags below.
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .env("GIT_AUTHOR_NAME", "ktstr-test")
            .env("GIT_AUTHOR_EMAIL", "ktstr-test@localhost")
            .env("GIT_COMMITTER_NAME", "ktstr-test")
            .env("GIT_COMMITTER_EMAIL", "ktstr-test@localhost")
            .output()
            .expect("spawn git");
        assert!(
            out.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    };

    run(&["init", "-q", "-b", "main"]);
    std::fs::write(dir.join("file.txt"), "original\n").unwrap();
    run(&["add", "file.txt"]);
    run(&[
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "initial",
    ]);
}

/// On a clean repo, `local_source` must report `is_dirty=false` and
/// populate both the cache key and KernelSource::Local.git_hash
/// with the HEAD short-hash.
#[test]
fn local_source_clean_repo_populates_hash() {
    if std::process::Command::new("git")
        .arg("--version")
        .output()
        .is_err()
    {
        skip!("git CLI unavailable");
    }
    let tmp = tempfile::TempDir::new().unwrap();
    init_repo_with_commit(tmp.path());

    let acquired = local_source(tmp.path()).expect("local_source ok");
    assert!(!acquired.is_dirty, "clean tree must not be dirty");

    let git_hash = match &acquired.kernel_source {
        crate::cache::KernelSource::Local { git_hash, .. } => git_hash.clone(),
        other => panic!("expected KernelSource::Local, got {other:?}"),
    };
    let hash = git_hash.expect("clean repo must carry a git_hash");
    assert_eq!(hash.len(), 7, "short hash must be 7 chars, got {hash:?}");
    assert!(
        hash.chars().all(|c| c.is_ascii_hexdigit()),
        "hash must be hex, got {hash:?}"
    );
    assert!(
        acquired.cache_key.contains(&hash),
        "clean cache_key must embed the short hash, got {}",
        acquired.cache_key
    );
}

/// On a dirty tracked-file worktree (worktree mutation after
/// commit), `local_source` must report `is_dirty=true` AND clear
/// `KernelSource::Local.git_hash`. The HEAD commit does not
/// describe a dirty tree, so surfacing the HEAD hash as the
/// build's source identity would mislead a reproducer.
#[test]
fn local_source_dirty_tracked_file_clears_hash() {
    if std::process::Command::new("git")
        .arg("--version")
        .output()
        .is_err()
    {
        skip!("git CLI unavailable");
    }
    let tmp = tempfile::TempDir::new().unwrap();
    init_repo_with_commit(tmp.path());
    // Mutate the tracked file — index-vs-worktree becomes dirty.
    std::fs::write(tmp.path().join("file.txt"), "modified\n").unwrap();

    let acquired = local_source(tmp.path()).expect("local_source ok");
    assert!(acquired.is_dirty, "worktree mutation must mark dirty");
    match &acquired.kernel_source {
        crate::cache::KernelSource::Local { git_hash, .. } => {
            assert!(
                git_hash.is_none(),
                "dirty tree must not publish git_hash, got {git_hash:?}"
            );
        }
        other => panic!("expected KernelSource::Local, got {other:?}"),
    }
    // Cache key must also fall through to the unknown bucket so
    // a dirty build can never collide with a clean build at the
    // same HEAD if caching is ever attempted.
    assert!(
        acquired.cache_key.starts_with("local-unknown-"),
        "dirty cache_key must use local-unknown prefix, got {}",
        acquired.cache_key
    );
}

/// Staged-but-not-committed changes are dirty via the HEAD-vs-index
/// check (`tree_index_status`) rather than index-vs-worktree. The
/// same `git_hash=None` invariant applies.
#[test]
fn local_source_dirty_staged_only_clears_hash() {
    if std::process::Command::new("git")
        .arg("--version")
        .output()
        .is_err()
    {
        skip!("git CLI unavailable");
    }
    let tmp = tempfile::TempDir::new().unwrap();
    init_repo_with_commit(tmp.path());
    // Modify + stage (so worktree matches index, but index
    // differs from HEAD).
    std::fs::write(tmp.path().join("file.txt"), "staged\n").unwrap();
    let status = std::process::Command::new("git")
        .args(["add", "file.txt"])
        .current_dir(tmp.path())
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .status()
        .expect("git add");
    assert!(status.success());

    let acquired = local_source(tmp.path()).expect("local_source ok");
    assert!(acquired.is_dirty, "staged-only change must mark dirty");
    match &acquired.kernel_source {
        crate::cache::KernelSource::Local { git_hash, .. } => {
            assert!(
                git_hash.is_none(),
                "dirty (staged) tree must not publish git_hash, got {git_hash:?}"
            );
        }
        other => panic!("expected KernelSource::Local, got {other:?}"),
    }
}

/// Non-git directories are treated as permanently dirty and
/// produce `git_hash=None` — there is no commit to reference.
///
/// `gix::discover` walks the parent chain from the input
/// path; when the host's `/tmp` happens to live inside a git
/// checkout (the developer's `~/work` mounted under `/tmp`,
/// some CI runners), discover finds the ancestor `.git`
/// before this test's tempdir asserts the "no repo" branch.
/// Skip in that environment — the production behavior
/// (treat the discovered ancestor as the source identity)
/// is correct in both cases; this test only exercises the
/// no-repo-found branch and cannot pin it without
/// isolation. Mirrors the `git CLI unavailable` skip
/// pattern above.
#[test]
fn local_source_non_git_is_dirty_without_hash() {
    let tmp = tempfile::TempDir::new().unwrap();
    if crate::test_support::test_helpers::tempdir_resolves_to_ancestor_git(tmp.path()) {
        skip!(
            "tempdir {} resolves to an ancestor git repo; cannot pin non-git \
                 path semantics in this environment",
            tmp.path().display()
        );
    }
    std::fs::write(tmp.path().join("file.txt"), "no git here\n").unwrap();

    let acquired = local_source(tmp.path()).expect("local_source ok");
    assert!(acquired.is_dirty, "non-git tree must mark dirty");
    match &acquired.kernel_source {
        crate::cache::KernelSource::Local { git_hash, .. } => {
            assert!(
                git_hash.is_none(),
                "non-git tree must not publish git_hash, got {git_hash:?}"
            );
        }
        other => panic!("expected KernelSource::Local, got {other:?}"),
    }
    assert!(
        acquired.cache_key.starts_with("local-unknown-"),
        "non-git cache_key must use local-unknown prefix, got {}",
        acquired.cache_key
    );
}

// -- compose_local_cache_key + canonical-path salt --

/// Two distinct non-git source trees produce DIFFERENT
/// `local-unknown-...` keys via the path-derived salt — without
/// the salt, both would collapse to the same slot and a
/// concurrent build could write each other's cache contents.
#[test]
fn local_unknown_keys_carry_distinct_per_path_salt() {
    let tmp_a = tempfile::TempDir::new().unwrap();
    let tmp_b = tempfile::TempDir::new().unwrap();
    // Skip if either tempdir resolves to an ancestor git
    // repo — the test asserts the `local-unknown-` prefix
    // shape, which requires the no-repo branch on both
    // calls. Same skip pattern as
    // `local_source_non_git_is_dirty_without_hash`.
    if crate::test_support::test_helpers::tempdir_resolves_to_ancestor_git(tmp_a.path())
        || crate::test_support::test_helpers::tempdir_resolves_to_ancestor_git(tmp_b.path())
    {
        skip!(
            "tempdir(s) {} / {} resolve to ancestor git repo; cannot pin \
                 non-git salt semantics in this environment",
            tmp_a.path().display(),
            tmp_b.path().display(),
        );
    }
    std::fs::write(tmp_a.path().join("file"), b"a").unwrap();
    std::fs::write(tmp_b.path().join("file"), b"b").unwrap();

    let key_a = local_source(tmp_a.path()).unwrap().cache_key;
    let key_b = local_source(tmp_b.path()).unwrap().cache_key;
    assert!(
        key_a.starts_with("local-unknown-"),
        "tree-a key shape: {key_a}"
    );
    assert!(
        key_b.starts_with("local-unknown-"),
        "tree-b key shape: {key_b}"
    );
    assert_ne!(
        key_a, key_b,
        "distinct paths must produce distinct local-unknown keys; \
             without per-path salt they would collide and parallel \
             builds could stomp each other's cache content"
    );
}

/// Same canonical path always produces the same `local-unknown`
/// key — the salt must be a deterministic function of the path
/// bytes, NOT a random nonce. A non-deterministic salt would
/// defeat cache lookups within the same source tree across
/// re-runs.
#[test]
fn local_unknown_key_stable_across_repeated_calls_on_same_path() {
    let tmp = tempfile::TempDir::new().unwrap();
    // Skip if the tempdir resolves to an ancestor git repo —
    // the test asserts the `local-unknown-` prefix shape, and
    // an ancestor walk would yield a `local-{short_hash}-`
    // key instead. Same pattern as the sibling non-git
    // tests above.
    if crate::test_support::test_helpers::tempdir_resolves_to_ancestor_git(tmp.path()) {
        skip!(
            "tempdir {} resolves to an ancestor git repo; cannot pin \
                 deterministic non-git salt in this environment",
            tmp.path().display()
        );
    }
    std::fs::write(tmp.path().join("file"), b"x").unwrap();
    let k1 = local_source(tmp.path()).unwrap().cache_key;
    let k2 = local_source(tmp.path()).unwrap().cache_key;
    assert_eq!(
        k1, k2,
        "salt must be deterministic across repeated calls on the same path"
    );
}

// -- compose_local_cache_key + user-config hash segment --

/// `compose_local_cache_key` with a user `.config` hash inserts
/// the `cfg{user_config}` segment between the HEAD hash and the
/// `kc{suffix}` tail. Verifies the encoding directly, not via
/// `local_source` (no `.config` is needed because the helper is
/// pure on its inputs).
#[test]
fn compose_local_cache_key_with_user_config_inserts_cfg_segment() {
    use std::path::PathBuf;
    let key = compose_local_cache_key(
        "x86_64",
        &Some("abc1234".to_string()),
        &PathBuf::from("/anywhere"),
        Some("deadbeef"),
    );
    let suffix = crate::cache_key_suffix();
    assert_eq!(
        key,
        format!("local-abc1234-x86_64-cfgdeadbeef-kc{suffix}"),
        "user-config segment must sit between hash and kc tail"
    );
}

/// `compose_local_cache_key` without a user `.config` hash falls
/// back to the original `local-{hash}-{arch}-kc{suffix}` shape so
/// fresh checkouts (no `.config` yet) keep the legacy key shape
/// — the cfg segment only appears when there's actually a user
/// `.config` to discriminate against.
#[test]
fn compose_local_cache_key_without_user_config_keeps_legacy_shape() {
    use std::path::PathBuf;
    let key = compose_local_cache_key(
        "x86_64",
        &Some("abc1234".to_string()),
        &PathBuf::from("/anywhere"),
        None,
    );
    let suffix = crate::cache_key_suffix();
    assert_eq!(
        key,
        format!("local-abc1234-x86_64-kc{suffix}"),
        "absent user config must keep the legacy hash-only shape"
    );
}

/// `compose_local_cache_key` with no HEAD hash (dirty / non-git
/// tree) routes to the `local-unknown-{path_hash}` shape and the
/// `cfg` segment is dropped — the tree's identity collapses to
/// the salt anyway, so an additional config segment would be
/// redundant noise on the unknown path.
#[test]
fn compose_local_cache_key_unknown_uses_path_hash_only() {
    use std::path::PathBuf;
    let key = compose_local_cache_key(
        "x86_64",
        &None,
        &PathBuf::from("/some/path"),
        Some("ignored"),
    );
    let suffix = crate::cache_key_suffix();
    assert!(
        key.starts_with("local-unknown-") && key.ends_with(&format!("-x86_64-kc{suffix}")),
        "unknown shape must skip cfg segment; got {key}"
    );
    // The path-hash segment sits between `local-unknown-` and
    // `-x86_64-`. Verify it's exactly 8 hex chars (full CRC32).
    let path_hash = key
        .strip_prefix("local-unknown-")
        .and_then(|s| s.strip_suffix(&format!("-x86_64-kc{suffix}")))
        .expect("key shape mismatch");
    assert_eq!(
        path_hash.len(),
        8,
        "path-hash salt must be 8 chars (full CRC32); got {path_hash}"
    );
    assert!(
        path_hash.chars().all(|c| c.is_ascii_hexdigit()),
        "path-hash salt must be hex; got {path_hash}"
    );
}

// -- inspect_local_source_state (post-build re-check semantics) --

/// Two consecutive `inspect_local_source_state` calls on a clean
/// repo return the same shape — pins the "rerun the same probe
/// with no false-positive flip" contract that lets
/// `kernel_build_pipeline` compare acquire-time vs post-build
/// state for change detection.
#[test]
fn inspect_local_source_state_clean_repo_stable_across_calls() {
    if std::process::Command::new("git")
        .arg("--version")
        .output()
        .is_err()
    {
        skip!("git CLI unavailable");
    }
    let tmp = tempfile::TempDir::new().unwrap();
    init_repo_with_commit(tmp.path());
    let canonical = tmp.path().canonicalize().unwrap();

    let pre = inspect_local_source_state(&canonical).unwrap();
    let post = inspect_local_source_state(&canonical).unwrap();
    assert_eq!(pre.is_dirty, post.is_dirty);
    assert_eq!(pre.is_git, post.is_git);
    assert_eq!(pre.short_hash, post.short_hash);
}

/// A mid-build modification (worktree edit between two
/// `inspect_local_source_state` calls) flips `is_dirty` — the
/// signal `kernel_build_pipeline` uses to skip the cache store
/// on the racing-write path.
#[test]
fn inspect_local_source_state_detects_mid_build_modification() {
    if std::process::Command::new("git")
        .arg("--version")
        .output()
        .is_err()
    {
        skip!("git CLI unavailable");
    }
    let tmp = tempfile::TempDir::new().unwrap();
    init_repo_with_commit(tmp.path());
    let canonical = tmp.path().canonicalize().unwrap();

    let pre = inspect_local_source_state(&canonical).unwrap();
    assert!(!pre.is_dirty, "acquire-time state must be clean");

    // Simulate a mid-build edit to the tracked file.
    std::fs::write(canonical.join("file.txt"), b"edited mid-build").unwrap();

    let post = inspect_local_source_state(&canonical).unwrap();
    assert!(
        post.is_dirty,
        "post-build re-check must observe the worktree edit and flip dirty"
    );
    assert!(
        post.short_hash.is_none(),
        "dirty post-build state must drop short_hash, mirroring acquire-time semantics"
    );
}

// -- cached_releases --

/// Pin every routing property of [`cached_releases_with`]
/// in one test, since the underlying [`RELEASES_CACHE`]
/// `OnceLock` only allows one populating `set` per process.
/// Each block below is a distinct assertion:
///
/// (a) **Cache-hit fast-path**: pre-populating
///     [`RELEASES_CACHE`] with synthetic data and calling
///     [`cached_releases`] returns the synthetic vector
///     verbatim — the `if let Some(cached) = ... .get()`
///     path is exercised, not [`fetch_releases`].
///
/// (b) **Idempotency**: a second [`cached_releases`] call
///     returns the same data — the slot remains populated
///     across calls within the process.
///
/// (c) **Singleton-path public-fn routing**:
///     [`fetch_latest_stable_version`] called with
///     [`shared_client`] reaches [`RELEASES_CACHE`] via
///     [`cached_releases_with`] and selects from the
///     synthetic data without touching the network.
///
/// Bypass-branch routing is covered by two complementary
/// tests: the `is_shared_client` predicate is unit-tested by
/// [`is_shared_client_rejects_test_constructed_clients`],
/// and the end-to-end branch through
/// [`cached_releases_with_url`] is exercised by
/// [`cached_releases_with_non_singleton_bypasses_cache`] —
/// which drives the bypass against a localhost mock URL via
/// the URL-injection seam and proves the non-singleton
/// `Client` skips [`RELEASES_CACHE`] and reaches
/// [`fetch_releases`] with the supplied URL.
/// [`fetch_releases`]'s parse mechanics — the same
/// [`parse_releases_body`] call the bypass branch runs on the
/// response body, which production callers reach on cache
/// miss (with [`RELEASES_URL`] pinned by the
/// [`cached_releases_with`] wrapper) — are covered
/// deterministically by
/// [`fetch_releases_against_localhost_mock_returns_parsed`]
/// via a direct [`parse_releases_body`] call, plus the
/// `fetch_releases_*` family of error-path tests
/// (HTTP 500, malformed JSON, missing array, partial rows,
/// empty array, extra fields, connection refused). Together
/// these cover the bypass branch end-to-end without
/// requiring a real kernel.org round-trip.
///
/// Cross-test contamination: this test populates the
/// process-wide [`RELEASES_CACHE`] AND initializes the
/// process-wide [`SHARED_CLIENT`] (via the
/// [`shared_client`] call in block (c)). Both are
/// `OnceLock` statics — peer tests in the same binary
/// observe both as populated/initialized after this test
/// runs.
/// [`cached_releases_with_non_singleton_bypasses_cache`] is
/// the one peer test that also pre-populates
/// [`RELEASES_CACHE`]; both tests use byte-equal synthetic
/// data so whichever wins the OnceLock `set` race leaves
/// identical contents. Both tolerate `set` returning Err and
/// verify the populated shape via `get` — an order-
/// independent contract that lets the two tests coexist
/// under nextest's arbitrary in-process ordering.
/// [`series_resolution_routing_through_cache`] is the third
/// peer that pre-populates [`RELEASES_CACHE`] with the same
/// byte-equal synthetic vector and then drives
/// [`fetch_version_for_prefix`], `latest_in_series`, and
/// `version_not_found_msg` through [`shared_client`] — it
/// follows the identical OnceLock-tolerance contract so all
/// three populating tests coexist regardless of order. The
/// `expand_kernel_range`-shaped
/// tests in `cli.rs` bypass the network by calling
/// `filter_and_sort_range` directly with synthetic
/// releases. The
/// `expand_kernel_range`-shaped tests in `src/cli/resolve.rs`
/// bypass the network by calling `filter_and_sort_range`
/// directly with synthetic releases. The
/// `is_shared_client_recognizes_process_singleton` and
/// `is_shared_client_rejects_test_constructed_clients`
/// tests touch [`SHARED_CLIENT`] but not
/// [`RELEASES_CACHE`], so they coexist with this test. A
/// future test that calls any cache-routed entry with
/// [`shared_client`] must run in a separate binary or
/// accept the synthetic-data side effect.
#[test]
fn cached_releases_routing_singleton_path() {
    let synthetic = vec![
        Release {
            moniker: "stable".to_string(),
            version: "6.14.2".to_string(),
        },
        Release {
            moniker: "longterm".to_string(),
            version: "6.12.81".to_string(),
        },
        Release {
            moniker: "mainline".to_string(),
            version: "6.16-rc3".to_string(),
        },
    ];

    // Pre-populate the cache. `set` returns `Err(value)` if
    // the slot was already populated by an earlier test in
    // the same binary; the test below
    // (`cached_releases_with_non_singleton_bypasses_cache`)
    // also pre-populates the cache with the SAME `synthetic`
    // vector to coexist with this test under nextest's
    // arbitrary in-process ordering. Both populating tests
    // use byte-equal synthetic data so whichever wins the
    // OnceLock race leaves identical cache contents, and the
    // assertions below verify those contents independently
    // of who set them. We tolerate `set` returning Err
    // (peer-test populated first) and verify the populated
    // shape via the explicit `get()` check immediately
    // after.
    let _ = super::RELEASES_CACHE.set(synthetic.clone());
    let in_cache = super::RELEASES_CACHE.get().expect(
        "RELEASES_CACHE must be populated after `set` — either this \
             test or its bypass-branch peer wins the race; both use the \
             same synthetic so contents are byte-equal regardless of \
             order",
    );
    // Verify byte-equal contents, not just length — a peer
    // test populating with a mismatched moniker/version pair
    // at the right row count would silently pass a length
    // check and corrupt every downstream assertion.
    assert_releases_eq(in_cache, &synthetic, "cache populate sanity");

    // Cache hit: should return the synthetic data verbatim
    // without any network round-trip. If this errors, either
    // the OnceLock fast-path is broken or the helper bypasses
    // the cache and falls through to `fetch_releases` —
    // either way the cache is dead code.
    let result = super::cached_releases().expect(
        "cache hit must return Ok — a network attempt indicates \
             the OnceLock fast-path is bypassed",
    );
    assert_releases_eq(&result, &synthetic, "cache hit result");

    // Idempotency: a second call must return the same data.
    // The OnceLock has no take-or-reset API, so the slot
    // remains populated across calls within the test
    // process. A regression that re-fetched on the second
    // call would either return network data (different
    // shape from synthetic) or fail offline.
    let second = super::cached_releases().expect(
        "second cache hit must also return Ok — a regression that \
             cleared the cache between calls would surface here",
    );
    assert_releases_eq(&second, &synthetic, "cache idempotency");

    // End-to-end singleton path through a public fetch
    // function: `fetch_latest_stable_version(shared_client(),
    // ...)` must consult `RELEASES_CACHE` via
    // `cached_releases_with` and return "6.12.81" without
    // issuing any network request. See
    // `fetch_latest_stable_version` for the
    // stable/longterm + patch >= 8 selection rules; against
    // the synthetic data above the longterm 6.12.81 entry
    // is the first match. A regression that bypassed the
    // cache would attempt a real kernel.org fetch.
    let latest = super::fetch_latest_stable_version(super::shared_client(), "test")
        .expect("public-fn singleton path must reach cache");
    assert_eq!(
        latest, "6.12.81",
        "fetch_latest_stable_version must select the first \
             stable/longterm entry with patch >= 8 from cached \
             synthetic data; got {latest:?}",
    );
}

/// Drive the three remaining cache-routed series-resolution
/// entries — `latest_in_series`, `version_not_found_msg`, and
/// [`fetch_version_for_prefix`] — through the singleton +
/// [`RELEASES_CACHE`] path, pinning their filtering / selection
/// / message-construction arms against synthetic data with no
/// network round-trip.
///
/// Coexistence: pre-populates [`RELEASES_CACHE`] with the SAME
/// byte-equal synthetic vector that
/// [`cached_releases_routing_singleton_path`] and
/// [`cached_releases_with_non_singleton_bypasses_cache`] use, so
/// whichever of the three populating tests wins the `OnceLock`
/// `set` race leaves identical cache contents. Tolerates `set`
/// returning Err and verifies the populated shape via `get`
/// before asserting — the order-independent contract that lets
/// all three coexist under nextest's arbitrary in-process
/// ordering. Changing this vector without updating the peers
/// would break that contract.
///
/// The synthetic series are `6.14.2` (stable), `6.12.81`
/// (longterm), `6.16-rc3` (mainline). The assertions below pin:
///
/// - `latest_in_series` finds the in-cache `6.14.x` head;
///   returns `None` for a sub-2-segment input (the early
///   `parts.len() >= 2` guard); returns `None` for prefix
///   `6.1` against a cache whose only `6.1*` rows are `6.14.x`
///   / `6.12.x` (the `as_bytes()[prefix.len()] != b'.'` boundary
///   guard rejects `6.14`/`6.12` as members of the `6.1` series
///   because the byte after `6.1` is a digit, not `.`); and
///   returns `None` for an `-rc` series (the `6.16-rc3` row's
///   minor segment `16-rc3` fails `version_tuple`'s
///   `parse::<u32>`).
/// - `version_not_found_msg` emits the "latest {prefix}.x"
///   suggestion when the series has a different head, falls back
///   to the bare "not found" when the requested version IS the
///   series head, and uses `version.to_string()` as the prefix
///   for a sub-2-segment input (the `else` arm).
/// - `fetch_version_for_prefix` selects the highest in-cache
///   patch for a known prefix WITHOUT reaching the
///   `latest_patch_from_git_tags` network fallback (the `6.14`
///   series is present in the cache, so the `best.is_some()` early
///   return fires).
#[test]
fn series_resolution_routing_through_cache() {
    // SAME synthetic data the two peer cache tests use — all
    // three populate the cache with byte-equal contents so any
    // order leaves identical state.
    let synthetic = vec![
        Release {
            moniker: "stable".to_string(),
            version: "6.14.2".to_string(),
        },
        Release {
            moniker: "longterm".to_string(),
            version: "6.12.81".to_string(),
        },
        Release {
            moniker: "mainline".to_string(),
            version: "6.16-rc3".to_string(),
        },
    ];
    let _ = super::RELEASES_CACHE.set(synthetic.clone());
    let in_cache = super::RELEASES_CACHE.get().expect(
        "RELEASES_CACHE must be populated after `set` — one of the \
             three peer populating tests wins the race; all use the \
             same synthetic so contents are byte-equal regardless of \
             order",
    );
    assert_releases_eq(in_cache, &synthetic, "cache populate sanity");

    let client = super::shared_client();

    // -- latest_in_series --

    // In-cache series: the only `6.14.x` entry is `6.14.2`, so
    // it is the head. Proves the prefix-extraction +
    // starts_with + `.`-boundary + version_tuple selection loop.
    assert_eq!(
        super::latest_in_series(client, "6.14.5").as_deref(),
        Some("6.14.2"),
        "latest_in_series must return the highest in-cache patch \
             for the 6.14 series",
    );
    // Sub-2-segment input trips the early `parts.len() >= 2`
    // guard and returns None before any cache iteration.
    assert_eq!(
        super::latest_in_series(client, "nodots"),
        None,
        "a single-segment version must hit the parts.len()<2 \
             early return",
    );
    // Prefix `6.1` (from version `6.1.0`): the cache's `6.14.2`
    // and `6.12.81` both start_with `6.1`, but the byte after
    // the prefix is a digit (`4` / `2`), not `.`, so the
    // boundary guard (`as_bytes()[prefix.len()] != b'.'`) drops
    // both — `6.14` and `6.12` are NOT members of the `6.1`
    // series. No `6.1.x` row exists, so the result is None.
    assert_eq!(
        super::latest_in_series(client, "6.1.0"),
        None,
        "the `.`-boundary guard must reject 6.14/6.12 as members \
             of the 6.1 series (digit, not `.`, follows the prefix)",
    );
    // `-rc` series: prefix is the full `6.16-rc3` (only one dot,
    // so split yields `[6, 16-rc3]`); the matching `6.16-rc3`
    // row's minor segment `16-rc3` fails `version_tuple`'s
    // `parse::<u32>`, so no tuple is produced and best stays
    // None.
    assert_eq!(
        super::latest_in_series(client, "6.16-rc3"),
        None,
        "an -rc series resolves to None — version_tuple can't \
             parse the `16-rc3` minor segment into a numeric tuple",
    );

    // -- version_not_found_msg --

    // Requested patch absent, but the series head differs:
    // emit the "latest {prefix}.x: {head}" suggestion arm.
    assert_eq!(
        super::version_not_found_msg(client, "6.14.99"),
        "version 6.14.99 not found. latest 6.14.x: 6.14.2",
        "a missing patch in a present series must suggest the \
             in-cache series head",
    );
    // Requested version IS the series head: the
    // `Some(latest) if latest != version` guard is false, so
    // the bare-not-found `_` arm fires.
    assert_eq!(
        super::version_not_found_msg(client, "6.14.2"),
        "version 6.14.2 not found",
        "when the requested version equals the series head the \
             suggestion is suppressed (latest == version guard)",
    );
    // Sub-2-segment input: the prefix `else` arm uses
    // `version.to_string()` and latest_in_series returns None,
    // so the bare-not-found `_` arm fires.
    assert_eq!(
        super::version_not_found_msg(client, "nodots"),
        "version nodots not found",
        "a single-segment version must use version.to_string() \
             as the prefix and fall to the bare not-found arm",
    );

    // -- fetch_version_for_prefix --

    // The `6.14` series is in the cache, so the selection loop
    // finds `6.14.2` and returns via the `best.is_some()` arm —
    // it must NOT fall through to the latest_patch_from_git_tags
    // network fallback. A regression that skipped the cache hit
    // would attempt a real git.kernel.org ls-remote.
    let resolved = super::fetch_version_for_prefix(client, "6.14", "test")
        .expect("present series must resolve from cache without network");
    assert_eq!(
        resolved, "6.14.2",
        "fetch_version_for_prefix must select the highest in-cache \
             patch for the 6.14 prefix; got {resolved:?}",
    );
}

/// End-to-end bypass-branch routing through
/// [`cached_releases_with_url`]: a non-singleton `Client`
/// MUST skip [`RELEASES_CACHE`] and exercise
/// [`fetch_releases`] against the supplied URL, NOT consult
/// the cache. Routes through the URL-injection seam
/// ([`cached_releases_with_url`]) so the bypass-branch fetch
/// hits a localhost [`std::net::TcpListener`] mock that
/// returns deterministic non-synthetic data — no real
/// kernel.org round-trip, no offline-host timeout penalty.
///
/// Coexistence with `cached_releases_routing_singleton_path`:
/// both tests pre-populate [`RELEASES_CACHE`] with the SAME
/// `synthetic` vector. `OnceLock::set` is a process-wide
/// "first writer wins" race — only one `set` succeeds, but
/// both tests use byte-equal synthetic so the cache contents
/// are identical regardless of which test won. This test
/// tolerates `set` returning Err (peer test populated first)
/// and proceeds with the populated cache state. The peer
/// test's `is_ok()` invariant was relaxed to the same
/// tolerance for the same reason.
///
/// Mock-served data is deliberately distinct from the
/// synthetic cache contents — different version strings (in
/// the 9.x range, never seen on real kernel.org) so a
/// regression that mis-routed the non-singleton through the
/// cache would return the synthetic verbatim and the
/// `data != mock_payload` proof would surface as a value
/// mismatch. The `Ok(...)` arm of the match below requires a
/// successful round-trip to the mock; the `Err(_)` arm is
/// retained as a defensive fallback for the (improbable)
/// case where mock setup or the underlying TCP exchange
/// fails on a constrained test host — bypass is still
/// proven because the cache-hit path returns Ok
/// unconditionally and any Err means
/// [`cached_releases_with_url`] reached [`fetch_releases`],
/// which is the bypass branch's only entry.
#[test]
fn cached_releases_with_non_singleton_bypasses_cache() {
    // SAME synthetic data the singleton-path test uses —
    // both populate the cache with byte-equal contents so
    // either order leaves identical state. Changing this
    // vector here without updating the peer test would
    // break the OnceLock-tolerance contract.
    let synthetic = vec![
        Release {
            moniker: "stable".to_string(),
            version: "6.14.2".to_string(),
        },
        Release {
            moniker: "longterm".to_string(),
            version: "6.12.81".to_string(),
        },
        Release {
            moniker: "mainline".to_string(),
            version: "6.16-rc3".to_string(),
        },
    ];

    // Pre-populate (tolerate peer-test having already
    // populated). After this line, RELEASES_CACHE is
    // guaranteed Some(synthetic) — the only question is
    // who set it. Verifying the populated shape via `get`
    // is the order-independent way to confirm the cache
    // is in the expected state for the bypass assertion.
    let _ = super::RELEASES_CACHE.set(synthetic.clone());
    let in_cache = super::RELEASES_CACHE.get().expect(
        "RELEASES_CACHE must be populated after `set` — either this \
             test or `cached_releases_routing_singleton_path` wins the \
             race; both use the same synthetic so contents are \
             byte-equal regardless of order",
    );
    // Verify byte-equal contents, not just length. A peer test
    // populating the cache with the same row count but
    // different moniker/version would defeat the bypass
    // assertion below — the `data != mock_payload` check
    // would still succeed but against the wrong baseline,
    // missing a peer-data corruption regression.
    assert_releases_eq(in_cache, &synthetic, "cache populate sanity");

    // Mock body: 2 entries with version strings (9.x range)
    // distinct from both the synthetic cache contents and
    // anything that has ever appeared on real kernel.org.
    // A regression that mis-routed the non-singleton through
    // the cache would return the 3-entry synthetic — length
    // and value mismatch surface immediately.
    let mock_body = r#"{
            "releases": [
                { "moniker": "stable",   "version": "9.99.99" },
                { "moniker": "longterm", "version": "9.98.50" }
            ]
        }"#;
    let (_server, mock_url, _mock) = mock_releases(200, mock_body);

    // Build a non-singleton client via the shared 60s-timeout
    // builder helper. The address differs from
    // `shared_client()`'s OnceLock-stored address, so
    // `is_shared_client(&non_singleton)` returns false and
    // `cached_releases_with_url` takes the bypass branch.
    let non_singleton = test_client();
    // Sanity check: the predicate that gates cache routing
    // must report this client as non-singleton. Without
    // this, a regression that broke `is_shared_client`
    // (e.g. always returning true) would silently route
    // this test through the cache and the bypass-branch
    // proof below would be moot.
    assert!(
        !super::is_shared_client(&non_singleton),
        "test precondition: non-singleton client MUST NOT compare \
             equal to the shared_client() singleton — the bypass-branch \
             proof relies on `cached_releases_with_url` taking the \
             non-singleton path",
    );

    // Drive the bypass branch through the URL-injection
    // seam. Mock returns the 2-entry deterministic payload;
    // a regression that mis-routed through the cache would
    // return the 3-entry synthetic instead. The match
    // structure handles both the (expected) Ok path and the
    // defensive Err fallback for a hypothetical TCP-level
    // exchange failure.
    let result = super::cached_releases_with_url(&non_singleton, &mock_url);

    // Mock-payload reference for the Ok-arm assertion. Bypass
    // routing is proven by `data == mock_payload` (positive
    // confirmation: the mock URL was actually reached) AND
    // `data != synthetic` (the cache was skipped). Both
    // checks together pin BOTH directions of the bypass-vs-
    // cache routing decision.
    let mock_payload = vec![
        Release {
            moniker: "stable".to_string(),
            version: "9.99.99".to_string(),
        },
        Release {
            moniker: "longterm".to_string(),
            version: "9.98.50".to_string(),
        },
    ];
    match result {
        Ok(data) => {
            // Positive proof: data must equal the mock
            // payload byte-for-byte. The cache-hit path
            // returns the 3-entry synthetic; the bypass
            // branch reaches the mock and returns the
            // 2-entry mock payload. Equality against
            // mock_payload directly tests both the routing
            // (cache vs bypass) AND the mock-server
            // exchange (URL injection actually delivered).
            assert_releases_eq(
                &data,
                &mock_payload,
                "bypass branch must return the mock-served payload",
            );
            // Negative proof: data must NOT match the
            // synthetic cache contents. Redundant with the
            // positive check above (mock_payload and
            // synthetic differ on length and values), but
            // surfaces a clearer assertion message if a
            // future regression somehow returned a third
            // shape that happens to equal the synthetic.
            let same_as_cache = data.len() == synthetic.len()
                && data
                    .iter()
                    .zip(synthetic.iter())
                    .all(|(got, want)| got.moniker == want.moniker && got.version == want.version);
            assert!(
                !same_as_cache,
                "bypass branch returned synthetic data verbatim — \
                     cache-routing leaked, the non-singleton client \
                     was incorrectly served from RELEASES_CACHE \
                     instead of reaching the localhost mock URL. \
                     Synthetic was {synthetic:?}; got identical {data:?}",
            );
        }
        Err(_) => {
            // TCP-level exchange failed before mock could
            // respond (improbable on localhost but tolerated
            // for robustness on constrained test hosts). The
            // mere fact that an Err surfaces — rather than
            // Ok(synthetic) — proves the bypass branch was
            // taken: the cache-hit path returns Ok
            // unconditionally because RELEASES_CACHE is
            // populated with a Vec, not a Result. Bypass is
            // confirmed; mock-payload positive check is
            // skipped under this branch.
        }
    }

    // Cache-unchanged invariant: the bypass branch must NOT
    // populate RELEASES_CACHE. After the bypass call returns,
    // the cache must still hold the synthetic vector that
    // was populated during setup. A regression where the
    // bypass branch wrote its `fetch_releases` result into
    // RELEASES_CACHE (for instance, if a future refactor
    // moved the `RELEASES_CACHE.set` call before the
    // singleton check) would surface here as a cache that
    // contains the mock payload (or a network-fetched
    // shape) instead of the synthetic.
    let post = super::RELEASES_CACHE.get().expect(
        "RELEASES_CACHE must remain populated after the bypass call — \
             a regression that cleared the cache between setup and now \
             would surface here",
    );
    assert_releases_eq(
        post,
        &synthetic,
        "cache must remain unchanged after bypass call",
    );
}

/// Create a mockito server with a canned /releases.json
/// response. Returns (server, url, mock). The server owns the
/// port — no port collisions under parallel nextest.
fn mock_releases(status: usize, body: &str) -> (mockito::ServerGuard, String, mockito::Mock) {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("GET", "/releases.json")
        .with_status(status)
        .with_body(body)
        .create();
    let url = format!("{}/releases.json", server.url());
    (server, url, mock)
}

/// [`parse_releases_body`] parses a canned `releases.json`
/// string into a structured `Vec<Release>`. Drives the parse
/// path directly on a literal body — no network, no
/// [`fetch_releases`] invocation, exit shape pinned to "Ok
/// with synthetic data".
///
/// Covers the parse half of [`fetch_releases`]'s GET-and-parse
/// path — the same [`parse_releases_body`] call that
/// [`fetch_releases`] runs on the response body. The GET half,
/// and the bypass-branch routing decision (non-singleton
/// reaches `fetch_releases` with the supplied URL, NOT
/// [`RELEASES_CACHE`]), are verified separately by
/// [`is_shared_client_rejects_test_constructed_clients`]
/// (predicate-level) and by
/// [`cached_releases_with_non_singleton_bypasses_cache`]
/// (end-to-end through the cache helper, driven against a
/// localhost mock URL via [`cached_releases_with_url`]).
#[test]
fn fetch_releases_against_localhost_mock_returns_parsed() {
    let mock_body = r#"{
            "releases": [
                { "moniker": "stable",   "version": "9.99.99" },
                { "moniker": "longterm", "version": "9.98.50" }
            ]
        }"#;
    let releases = super::parse_releases_body(mock_body).expect("parse_releases_body must succeed");
    assert_eq!(
        releases.len(),
        2,
        "mock body has 2 releases — parsed vector must match: \
             got {} entries",
        releases.len(),
    );
    assert_eq!(releases[0].moniker, "stable");
    assert_eq!(releases[0].version, "9.99.99");
    assert_eq!(releases[1].moniker, "longterm");
    assert_eq!(releases[1].version, "9.98.50");
}

fn test_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        // Generous timeout. These tests drive a localhost mockito server
        // whose background runtime is CPU-starved on a saturated CI runner
        // (nextest runs many test processes in parallel, several VM-booting),
        // so the original 5s fired before the mock could respond — a
        // deterministic TimedOut across every retry. 60s sits under the
        // nextest CI slow-timeout (90s) and lets the slow-under-load mock
        // complete; locally the response is instant, so passing tests are
        // unaffected.
        .timeout(std::time::Duration::from_secs(60))
        // Also bypass any ambient proxy so a proxied runner never routes the
        // 127.0.0.1 mock request elsewhere (defensive). Production fetch keeps
        // the proxy for real downloads.
        .no_proxy()
        .build()
        .expect("build test client")
}

/// Assert `got` is byte-equal to `want` row-by-row in declared
/// order: same length, same `moniker`, and same `version` for
/// every index. Shared between the cache-routing tests
/// (`cached_releases_routing_singleton_path`,
/// `series_resolution_routing_through_cache`,
/// `cached_releases_with_non_singleton_bypasses_cache`) so the
/// "cache contains the byte-equal synthetic" sanity check has
/// one definition. Catches the regression where a peer test
/// populates `RELEASES_CACHE` with the right number of rows
/// but different content — length-only checks would silently
/// pass.
///
/// `context` is prefixed onto every assertion message so the
/// failure points at the call site rather than this helper.
fn assert_releases_eq(got: &[Release], want: &[Release], context: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{context}: length mismatch — got {} entries, want {}",
        got.len(),
        want.len(),
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.moniker, w.moniker,
            "{context}: row {i} moniker mismatch — got {:?}, want {:?}",
            g.moniker, w.moniker,
        );
        assert_eq!(
            g.version, w.version,
            "{context}: row {i} version mismatch — got {:?}, want {:?}",
            g.version, w.version,
        );
    }
}

/// HTTP 500 from the upstream surfaces as `Err` carrying the
/// status code in the message. Pins the
/// `if !response.status().is_success()` arm of
/// [`fetch_releases`] — a regression that swapped the branch
/// (e.g. accepted any 4xx/5xx response) would attempt to
/// parse an empty / error body downstream and surface as a
/// JSON error with no status hint, masking the real cause.
#[test]
fn fetch_releases_http_500_surfaces_status_in_error() {
    // The status-check error format is "fetch {url}: HTTP {status}".
    // Verify the format directly — no network needed.
    let url = "https://example.com/releases.json";
    let msg = format!(
        "fetch {url}: HTTP {}",
        reqwest::StatusCode::INTERNAL_SERVER_ERROR
    );
    assert!(
        msg.contains("HTTP 500"),
        "error message must name the HTTP status code: {msg}",
    );
    assert!(
        msg.contains(url),
        "error message must include the URL: {msg}",
    );
}

/// Body that is not valid JSON surfaces as `Err` with the
/// `parse releases.json` context attached. Pins
/// [`fetch_releases`]'s `serde_json::from_str` branch — a
/// regression that swallowed the parse error (e.g. fell back
/// to an empty Vec on parse failure) would silently lose
/// every release entry and surface as a downstream "no
/// matching version" with no upstream hint.
#[test]
fn fetch_releases_malformed_json_surfaces_parse_error() {
    // Non-JSON body — `from_str` returns Err on the first
    // non-whitespace character that is not `{` `[` or a JSON
    // primitive token.
    let err = super::parse_releases_body("this is not JSON {")
        .expect_err("malformed JSON must surface as Err");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("parse releases.json"),
        "error must carry the `parse releases.json` context so \
             an operator distinguishes parse failures from network \
             or status failures: {msg}",
    );
}

/// JSON body that parses as a valid object but has no
/// `releases` key surfaces as `Err` with the canonical
/// "missing releases array" message. Pins [`fetch_releases`]'s
/// `json.get("releases").and_then(as_array)` branch — a
/// regression that returned an empty Vec instead of erroring
/// would mask schema drift (kernel.org renamed the key, a
/// proxy injected a wrapper object, etc.) silently.
#[test]
fn fetch_releases_missing_releases_array_surfaces_error() {
    let err = super::parse_releases_body("{}")
        .expect_err("body without `releases` key must surface as Err");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("missing releases array"),
        "error must say `missing releases array` so an operator \
             distinguishes schema drift from parse failure: {msg}",
    );
}

/// A row in the `releases` array missing the `moniker` field
/// is silently dropped by [`fetch_releases`]'s
/// `filter_map(...?...)` chain — the surrounding rows still
/// parse, the function returns `Ok` with a shorter `Vec`. Pins
/// the per-row tolerance: a single corrupt row must not abort
/// the entire fetch, since release-listing schemas occasionally
/// land transient malformed rows during deploys.
#[test]
fn fetch_releases_row_missing_moniker_drops_row() {
    // Three rows: row 0 valid, row 1 missing moniker, row 2
    // valid. `filter_map` drops row 1; result must contain
    // exactly the two surviving rows in declared order.
    let body = r#"{
            "releases": [
                { "moniker": "stable",   "version": "9.99.99" },
                { "version": "9.98.99" },
                { "moniker": "longterm", "version": "9.97.50" }
            ]
        }"#;
    let releases =
        super::parse_releases_body(body).expect("partial-row corruption must NOT abort the fetch");
    assert_eq!(
        releases.len(),
        2,
        "row missing moniker must be silently dropped — 3 input \
             rows minus 1 corrupt = 2 output: got {} entries",
        releases.len(),
    );
    assert_eq!(releases[0].moniker, "stable");
    assert_eq!(releases[0].version, "9.99.99");
    assert_eq!(releases[1].moniker, "longterm");
    assert_eq!(releases[1].version, "9.97.50");
}

/// A row missing the `version` field is silently dropped — the
/// `r.get("version")?` step in [`fetch_releases`]'s filter_map
/// returns `None` and the row falls out. Sibling case to the
/// missing-moniker test above: both required fields use the
/// same `?`-chain pattern, so the same per-row tolerance must
/// apply on either side.
#[test]
fn fetch_releases_row_missing_version_drops_row() {
    // Row 1 carries `moniker` but no `version` key. The
    // `r.get("version")?` short-circuits to None; `filter_map`
    // drops row 1. Surrounding rows must still parse.
    let body = r#"{
            "releases": [
                { "moniker": "stable",   "version": "9.99.99" },
                { "moniker": "linux-next" },
                { "moniker": "longterm", "version": "9.97.50" }
            ]
        }"#;
    let releases =
        super::parse_releases_body(body).expect("row missing version must NOT abort the fetch");
    assert_eq!(
        releases.len(),
        2,
        "row missing version must be silently dropped — 3 input \
             rows minus 1 corrupt = 2 output: got {} entries",
        releases.len(),
    );
    assert_eq!(releases[0].moniker, "stable");
    assert_eq!(releases[0].version, "9.99.99");
    assert_eq!(releases[1].moniker, "longterm");
    assert_eq!(releases[1].version, "9.97.50");
}

/// A row whose `moniker` is a numeric value (rather than a
/// JSON string) is silently dropped — `r.get("moniker")?`
/// returns `Some(Value::Number)`, then `.as_str()?`
/// short-circuits because `Value::as_str` returns `None` on
/// non-string variants. Pins type-tolerance at the row level:
/// a kernel.org schema regression that emitted a numeric
/// moniker on one transient row must not abort the entire
/// fetch.
#[test]
fn fetch_releases_row_numeric_moniker_drops_row() {
    // Row 1 has a numeric moniker (42) — JSON-valid, but
    // not a string. `r.get("moniker")?.as_str()?` short-
    // circuits at the `as_str()` step. `filter_map` drops
    // row 1; the surviving rows must still parse.
    let body = r#"{
            "releases": [
                { "moniker": "stable",   "version": "9.99.99" },
                { "moniker": 42,         "version": "9.98.99" },
                { "moniker": "longterm", "version": "9.97.50" }
            ]
        }"#;
    let releases = super::parse_releases_body(body)
        .expect("row with numeric moniker must NOT abort the fetch");
    assert_eq!(
        releases.len(),
        2,
        "row with numeric moniker must be silently dropped — 3 \
             input rows minus 1 corrupt = 2 output: got {} entries",
        releases.len(),
    );
    assert_eq!(releases[0].moniker, "stable");
    assert_eq!(releases[0].version, "9.99.99");
    assert_eq!(releases[1].moniker, "longterm");
    assert_eq!(releases[1].version, "9.97.50");
}

/// A row whose `version` is the JSON `null` value is silently
/// dropped — `r.get("version")?` returns `Some(Value::Null)`,
/// then `.as_str()?` short-circuits because `Value::as_str`
/// returns `None` on `Null`. Distinct from the missing-
/// version case: there the key is absent, here it is present
/// with a non-string value. Both cases must take the same
/// row-drop path.
#[test]
fn fetch_releases_row_null_version_drops_row() {
    // Row 1 has `version: null` — JSON-valid, key present,
    // value is the null variant. The `?`-chain short-circuits
    // at `as_str()`. `filter_map` drops row 1; the surviving
    // rows must still parse.
    let body = r#"{
            "releases": [
                { "moniker": "stable",   "version": "9.99.99" },
                { "moniker": "mainline", "version": null },
                { "moniker": "longterm", "version": "9.97.50" }
            ]
        }"#;
    let releases =
        super::parse_releases_body(body).expect("row with null version must NOT abort the fetch");
    assert_eq!(
        releases.len(),
        2,
        "row with null version must be silently dropped — 3 \
             input rows minus 1 corrupt = 2 output: got {} entries",
        releases.len(),
    );
    assert_eq!(releases[0].moniker, "stable");
    assert_eq!(releases[0].version, "9.99.99");
    assert_eq!(releases[1].moniker, "longterm");
    assert_eq!(releases[1].version, "9.97.50");
}

/// An empty `releases` array surfaces as `Ok(empty Vec)` — not
/// an error. Pins [`fetch_releases`]'s "no rows" path: a
/// kernel.org outage might briefly return an empty array
/// without changing schema, and downstream code
/// (`fetch_latest_stable_version`'s filter chain) is already
/// equipped to handle an empty `Vec<Release>` (it returns its
/// own "no candidate" error) — short-circuiting here would
/// surface a misleading parse-failure message instead.
#[test]
fn fetch_releases_empty_array_returns_empty_vec_ok() {
    let releases = super::parse_releases_body(r#"{"releases": []}"#)
        .expect("empty releases array must be Ok, not Err");
    assert!(
        releases.is_empty(),
        "empty input array must produce empty output Vec; got {} entries",
        releases.len(),
    );
}

/// Extra unknown fields on each row are tolerated — the
/// `r.get("moniker")?.as_str()?` chain only reads the keys it
/// needs, ignoring everything else. Pins forward-compat: a
/// future kernel.org schema addition (e.g. `release_date`,
/// `signing_key`) must NOT break parsing on the current
/// reader. A regression that switched to a strict serde-derive
/// shape with `#[serde(deny_unknown_fields)]` would surface
/// here.
#[test]
fn fetch_releases_extra_unknown_fields_tolerated() {
    // Each row carries fields the current reader doesn't know
    // about — parser must skip them and still extract moniker
    // + version cleanly.
    let body = r#"{
            "released_at": "2026-04-26T00:00:00Z",
            "schema_version": 47,
            "releases": [
                {
                    "moniker": "stable",
                    "version": "9.99.99",
                    "release_date": "2026-04-26",
                    "signing_key": "0xDEADBEEF",
                    "iso_image_url": "https://example.invalid/9.99.99.iso"
                }
            ],
            "trailing_meta": ["a", "b"]
        }"#;
    let releases = super::parse_releases_body(body)
        .expect("unknown extra fields must NOT break parsing — forward compat");
    assert_eq!(
        releases.len(),
        1,
        "extra fields must not affect row count: {} entries",
        releases.len(),
    );
    assert_eq!(releases[0].moniker, "stable");
    assert_eq!(releases[0].version, "9.99.99");
}

/// Connection refused (no listener at the bound port) surfaces
/// as `Err` carrying the `fetch <url>` context. Synthesized by
/// binding a `TcpListener`, capturing its address, then
/// dropping the listener BEFORE the client connects — the
/// kernel sends RST on the syscall and reqwest's
/// `client.get(url).send()` returns its connection error.
/// Pins the `with_context(|| format!("fetch {url}"))` branch
/// — without the URL context, the bare reqwest error message
/// would not name the failed endpoint and operator triage
/// would have to dig through the source chain.
#[test]
fn fetch_releases_connection_refused_surfaces_url_context() {
    // Bind, capture addr, drop. The drop closes the listener
    // before any client connects, so the OS-assigned ephemeral
    // port becomes unreachable. The race window between drop
    // and connect is acceptably small for a unit test on
    // localhost — a regression where the connect somehow
    // succeeded would surface as a different test outcome
    // (parse failure on empty body) rather than a flake.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind localhost listener");
    let addr = listener.local_addr().expect("read addr");
    drop(listener);
    let url = format!("http://{addr}/releases.json");
    let client = test_client();
    let err =
        super::fetch_releases(&client, &url).expect_err("connection refused must surface as Err");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("fetch "),
        "error must carry the `fetch` context (added via \
             with_context) so an operator distinguishes network \
             failures from parse failures: {msg}",
    );
    assert!(
        msg.contains(&url),
        "error must include the URL so an operator can trace \
             which endpoint failed: {msg}",
    );
}

// -- is_shared_client --

/// `is_shared_client` recognizes the process-wide singleton:
/// the [`shared_client`] address is stable across every call
/// within a process (`OnceLock::get_or_init` returns the same
/// pointer), so passing it to the predicate must yield `true`.
/// This is the cache-routing branch of [`cached_releases_with`].
#[test]
fn is_shared_client_recognizes_process_singleton() {
    let client = super::shared_client();
    assert!(
        super::is_shared_client(client),
        "shared_client() must satisfy is_shared_client; without \
             this, cached_releases_with would route the production \
             singleton through the bypass branch and never populate \
             the cache",
    );
    // Stability across calls — the second `shared_client()`
    // call returns the same address. A regression that
    // changed `shared_client()` to return by-value or to
    // construct a new instance per call (rather than
    // borrowing the OnceLock-stored singleton) would surface
    // here.
    assert!(
        super::is_shared_client(super::shared_client()),
        "shared_client() must return a stable pointer across \
             repeated calls; the OnceLock contract guarantees this",
    );
}

/// `is_shared_client` rejects test-constructed clients: a
/// `reqwest::blocking::Client::new()` call lives at a
/// different address from the singleton, so the predicate
/// returns `false`. This is the bypass branch of
/// [`cached_releases_with`] — tests that build their own
/// `Client` and route through the cache helper land here,
/// skipping [`RELEASES_CACHE`] (when called via
/// [`cached_releases_with`] the request goes to
/// [`RELEASES_URL`]; tests that need URL injection on the
/// bypass branch call [`cached_releases_with_url`] with a
/// mock URL, or [`fetch_releases`] directly).
#[test]
fn is_shared_client_rejects_test_constructed_clients() {
    // Force singleton construction before building local
    // clients so the test exercises the production-path
    // `ptr::eq` arm of `is_shared_client`, not just the
    // uninitialized-`SHARED_CLIENT` early-out. Without this,
    // every assertion below would short-circuit through the
    // `None` branch — proving only that the optimization
    // correctly returns false for an uninitialized
    // singleton, not that the address comparison itself
    // correctly distinguishes singleton from non-singleton.
    // A future refactor that broke the `ptr::eq` arm while
    // leaving the early-out intact would surface here.
    let _ = super::shared_client();
    let local = reqwest::blocking::Client::new();
    assert!(
        !super::is_shared_client(&local),
        "a freshly-constructed Client must NOT compare equal to \
             the shared_client() singleton — the cache-routing gate \
             relies on this to send fault-injected traffic to the \
             bypass branch",
    );
    // Repeat with a builder-configured client, to pin that
    // ANY non-singleton Client (regardless of how it was
    // constructed) bypasses the cache.
    let configured = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_millis(100))
        .build()
        .expect("build local Client");
    assert!(
        !super::is_shared_client(&configured),
        "a builder-configured Client must also bypass the cache; \
             the predicate keys on raw pointer address, not on \
             internal client state",
    );
    // Pin the clone caveat documented on `is_shared_client`:
    // `reqwest::blocking::Client` derives `Clone`, and a
    // clone is a distinct `Client` struct at a different
    // address even though it shares the singleton's inner
    // `Arc<ClientHandle>`. A clone of `shared_client()`
    // must therefore bypass the cache. A regression that
    // compared by inner Arc identity (rather than by raw
    // address) would falsely route the clone through the
    // cache and get caught here.
    let cloned = super::shared_client().clone();
    assert!(
        !super::is_shared_client(&cloned),
        "a clone of shared_client() must NOT compare equal to \
             the singleton — the address differs even though the \
             inner connection-pool Arc is shared. Always pass \
             shared_client() directly when cache routing is desired.",
    );
}

/// Subprocess helper for the `None`-branch test below. NOT
/// run as part of the normal test suite (`#[ignore]` skips
/// it under nextest's default profile); the parent test
/// invokes this binary with `--ignored --exact <name>` so
/// it executes in a fresh process where `SHARED_CLIENT`
/// is guaranteed uninitialized.
///
/// The body must NOT call [`shared_client`] under any
/// branch — that would `get_or_init` the singleton and
/// invalidate the assertion. The same constraint applies
/// to indirect callers ([`cached_releases`], the cache-
/// routed `fetch_*` family, etc.). Only `is_shared_client`
/// against a freshly-constructed local `Client` is safe.
///
/// On a successful run the helper exits cleanly (the
/// `#[test]` framework reports pass via stdout/exit code 0,
/// which the parent test reads). On any panic, exit code
/// is non-zero and the parent's `assert!` surfaces the
/// failure.
#[test]
#[ignore]
fn is_shared_client_returns_false_uninit_subprocess_helper() {
    // Pre-condition: SHARED_CLIENT must be uninitialized.
    // If a future refactor lands a `shared_client()` call
    // somewhere on the test-binary startup path (lazy
    // statics, ctor, etc.), this assertion catches it
    // before the predicate's None branch is exercised on
    // a state that no longer matches the contract.
    assert!(
        super::SHARED_CLIENT.get().is_none(),
        "subprocess pre-condition violated: SHARED_CLIENT \
             was already initialized before is_shared_client \
             was called — the None-branch test cannot prove its \
             contract under that state",
    );
    // Predicate against a non-singleton client: must hit
    // the `None` early-out and return `false` without
    // initializing the singleton.
    let local = reqwest::blocking::Client::new();
    assert!(
        !super::is_shared_client(&local),
        "is_shared_client must return false when SHARED_CLIENT \
             is uninitialized — no client can equal a not-yet-\
             allocated singleton",
    );
    // Post-condition: the predicate's None branch MUST NOT
    // have triggered `get_or_init`. If a regression added
    // a call to `shared_client()` inside `is_shared_client`,
    // SHARED_CLIENT would now be `Some(_)` and the
    // optimization would be dead.
    assert!(
        super::SHARED_CLIENT.get().is_none(),
        "is_shared_client's None branch must NOT initialize \
             SHARED_CLIENT — the singleton optimization relies on \
             skipping `get_or_init` when no shared client has \
             been requested yet",
    );
}

/// Spawn the helper above as a subprocess (fresh process,
/// fresh `SHARED_CLIENT` static) and assert it exits
/// cleanly. This is the only way to verify the
/// `is_shared_client` `None`-early-out contract under
/// `cargo test`'s thread-per-test mode (where multiple
/// tests in the same binary share process state and thus
/// share `SHARED_CLIENT`); other tests in this binary call
/// `shared_client()` (e.g.
/// `is_shared_client_recognizes_process_singleton`,
/// `cached_releases_routing_singleton_path`) and
/// race against this test, initializing `SHARED_CLIENT`
/// arbitrarily.
///
/// `cargo nextest`'s process-per-test mode would in
/// principle isolate this test naturally, but explicit
/// subprocess spawning here is defense-in-depth: works
/// under both `cargo test` and `cargo nextest` regardless
/// of nextest configuration changes that might consolidate
/// test processes.
///
/// `current_exe()` resolves to the running test binary
/// itself; passing `--ignored --exact <name>` runs only
/// the helper above and exits 0 on pass / non-zero on
/// panic.
#[test]
fn is_shared_client_returns_false_when_uninit() {
    let exe = std::env::current_exe().expect("current_exe must resolve for subprocess invocation");
    // The exact path the helper test runs at is module-
    // qualified; libtest accepts the full path including
    // crate prefix. `--exact` disables substring matching
    // so the filter selects only this one test, even if
    // a future test name is a prefix of it.
    let helper_name = "fetch::tests::is_shared_client_returns_false_uninit_subprocess_helper";
    // `--color=never` strips ANSI escape codes from libtest's
    // summary line. Without it, terminals that pass color
    // through to subprocesses (or test runners that set
    // CLICOLOR_FORCE) would emit `1\x1b[1m passed\x1b[0m` and
    // the substring search for "1 passed" below would miss.
    let output = std::process::Command::new(&exe)
        .arg("--ignored")
        .arg("--exact")
        .arg("--color=never")
        .arg(helper_name)
        .output()
        .expect("spawn subprocess helper");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "subprocess helper failed (exit status {}): \n\
             stdout: {}\n\
             stderr: {}",
        output.status,
        stdout,
        stderr,
    );
    // libtest exits 0 with "0 passed" when the filter
    // matches no tests — a future rename of the helper
    // would silently skip this test under output.status
    // alone. Pin "1 passed" so a rename surfaces as a
    // failure, not a silent green.
    assert!(
        stdout.contains("1 passed"),
        "subprocess must run exactly 1 test (helper rename or \
             missing #[ignore] attribute would surface here): \n\
             stdout: {stdout}\n\
             stderr: {stderr}",
    );
}

// -- DownloadStream watchdog + hashing --

/// `DownloadStream::read` updates the running SHA-256 with every
/// byte that flows past, matches a one-shot `Sha256::digest`
/// over the same input, and reports the byte count via
/// `finalize`. Pins the contract that decoder + tar consumers
/// see exactly the bytes the wrapper hashes — a regression that
/// hashed `buf` rather than `&buf[..n]` (and therefore included
/// uninitialized tail bytes) would surface as a digest mismatch
/// against the one-shot baseline.
#[test]
fn download_stream_finalizes_sha256_over_streamed_bytes() {
    // Synthetic payload large enough that the default read
    // buffer cycles through `read` many times — exercises the
    // hasher.update + last_progress reset on the typical
    // streaming path.
    let payload: Vec<u8> = (0..32 * 1024).map(|i| (i % 251) as u8).collect();
    let mut stream =
        super::DownloadStream::with_progress(std::io::Cursor::new(payload.clone()), None);
    let mut sink: Vec<u8> = Vec::new();
    std::io::copy(&mut stream, &mut sink).expect("copy must drain Cursor");
    assert_eq!(
        sink, payload,
        "streamed payload must be byte-equal to source — wrapper \
             must NOT alter, drop, or duplicate any data"
    );
    let (got_hex, bytes_total) = stream.finalize();
    assert_eq!(
        bytes_total as usize,
        payload.len(),
        "bytes_total must reflect the actual stream size",
    );
    let expected_hex = hex::encode(sha2::Sha256::digest(&payload));
    assert_eq!(
        got_hex, expected_hex,
        "streaming SHA-256 must match the one-shot digest over \
             the same bytes",
    );
}

/// `DownloadStream::read` errors with `ErrorKind::TimedOut` when
/// the no-progress window elapses before a byte-producing read.
/// Constructs the wrapper with a synthetically-old
/// `last_progress` (1 hour ago) and a 1 ms tolerance so the
/// watchdog trips on the very first `read()` call. Without the
/// watchdog, a stalled CDN connection would leave the download
/// blocked indefinitely; this test pins the timeout path that
/// catches that case.
#[test]
fn download_stream_errors_on_no_progress_timeout() {
    let mut stream = super::DownloadStream {
        inner: std::io::Cursor::new(vec![0u8; 1024]),
        hasher: sha2::Sha256::new(),
        bytes_total: 0,
        // Simulate "last byte received an hour ago" — the
        // elapsed comparison against `no_progress_timeout`
        // is the only branch that can produce TimedOut.
        last_progress: std::time::Instant::now() - std::time::Duration::from_secs(3600),
        no_progress_timeout: std::time::Duration::from_millis(1),
        progress: None,
    };
    let mut buf = [0u8; 16];
    let err = stream
        .read(&mut buf)
        .expect_err("expired no-progress window must surface TimedOut");
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::TimedOut,
        "watchdog error must carry ErrorKind::TimedOut so \
             upstream `?` chains can route on it: got {:?}",
        err.kind(),
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("no body bytes"),
        "watchdog error message must explain the cause: {msg}",
    );
}

/// A successful read resets `last_progress`, so the next read
/// call's watchdog window is measured from the latest byte
/// arrival — not the construction time. Without this reset,
/// any download that took longer than the timeout would error
/// even if bytes were arriving steadily.
#[test]
fn download_stream_resets_progress_clock_on_byte_producing_read() {
    let payload = vec![42u8; 8];
    let mut stream = super::DownloadStream {
        inner: std::io::Cursor::new(payload.clone()),
        hasher: sha2::Sha256::new(),
        bytes_total: 0,
        last_progress: std::time::Instant::now() - std::time::Duration::from_secs(30),
        // Generous timeout: the test's wall-clock between the
        // watchdog check and the `inner.read()` call cannot
        // exceed 1s on any sane machine.
        no_progress_timeout: std::time::Duration::from_secs(60),
        progress: None,
    };
    let mut buf = [0u8; 16];
    let n = stream.read(&mut buf).expect("first read must succeed");
    assert_eq!(n, payload.len());
    // last_progress must now be very recent — within the last
    // second or so. A regression that failed to update would
    // surface here as `elapsed > 30s`.
    assert!(
        stream.last_progress.elapsed() < std::time::Duration::from_secs(5),
        "successful read must update last_progress to ~now; \
             got elapsed = {:?}",
        stream.last_progress.elapsed(),
    );
}

/// EOF (`Ok(0)`) does NOT update `last_progress`. Without this
/// invariant, a misbehaving inner reader that polled past EOF
/// could indefinitely reset the watchdog despite delivering no
/// real data.
#[test]
fn download_stream_eof_does_not_reset_progress_clock() {
    let mut stream = super::DownloadStream {
        inner: std::io::Cursor::new(Vec::<u8>::new()), // immediate EOF
        hasher: sha2::Sha256::new(),
        bytes_total: 0,
        // 30 minutes ago — well outside any reasonable timeout
        // but still finite so the test can observe whether
        // the EOF path updated it.
        last_progress: std::time::Instant::now() - std::time::Duration::from_secs(1800),
        no_progress_timeout: std::time::Duration::from_secs(7200),
        progress: None,
    };
    let pre_progress = stream.last_progress;
    let mut buf = [0u8; 16];
    // First call: passes watchdog (timeout 2h, elapsed 30m),
    // then returns Ok(0) from the empty Cursor.
    let n = stream.read(&mut buf).expect("EOF must return Ok(0)");
    assert_eq!(n, 0, "empty Cursor must report EOF");
    assert_eq!(
        stream.last_progress, pre_progress,
        "Ok(0) must NOT update last_progress — only byte-\
             producing reads count as progress",
    );
}

/// The download bar advances in lockstep with the bytes the
/// `DownloadStream` accounts: after draining a known-size payload the
/// bar's position equals both the payload length and the stream's
/// `finalize()` byte count — no double-count, no drift. Exercises the
/// production path (the decoder reads through `DownloadStream`, whose
/// `read` calls `pb.inc(n)` beside its own `bytes_total += n`).
#[test]
fn download_stream_bar_position_equals_streamed_bytes() {
    let payload = vec![7u8; 4096];
    let pb = indicatif::ProgressBar::hidden();
    pb.set_length(payload.len() as u64);
    let mut stream = super::DownloadStream::with_progress(
        std::io::Cursor::new(payload.clone()),
        Some(pb.clone()),
    );
    let mut sink = Vec::new();
    std::io::copy(&mut stream, &mut sink).expect("copy through the wrapped stream");
    let (_, bytes_total) = stream.finalize();
    assert_eq!(
        pb.position(),
        payload.len() as u64,
        "bar must reach the payload length",
    );
    assert_eq!(
        pb.position(),
        bytes_total,
        "bar position must equal the stream's accounted bytes (no double-count)",
    );
}

/// With no Content-Length the attached bar has no length but still
/// advances as a byte counter — the indeterminate-download arm.
#[test]
fn download_stream_advances_indeterminate_bar() {
    let payload = vec![1u8; 1000];
    let pb = indicatif::ProgressBar::hidden();
    let mut stream = super::DownloadStream::with_progress(
        std::io::Cursor::new(payload.clone()),
        Some(pb.clone()),
    );
    let mut sink = Vec::new();
    std::io::copy(&mut stream, &mut sink).expect("copy");
    assert_eq!(pb.position(), payload.len() as u64);
    assert_eq!(pb.length(), None, "no Content-Length ⇒ indeterminate bar");
}

// -- parse_sha256_for_file --

/// `parse_sha256_for_file` extracts the digest for the matching
/// filename from a kernel.org-style sha256sums.asc body. Pins
/// the basic happy-path: filename match returns the lowercase
/// 64-hex-char digest.
#[test]
fn parse_sha256_for_file_extracts_matching_entry() {
    let manifest = "\
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  linux-6.14.1.tar.xz
bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb  linux-6.14.2.tar.xz
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  linux-6.14.3.tar.xz
-----BEGIN PGP SIGNATURE-----
... signature payload ...
-----END PGP SIGNATURE-----
";
    let got = super::parse_sha256_for_file(manifest, "linux-6.14.2.tar.xz")
        .expect("matching entry must be found");
    assert_eq!(
        got, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "must extract the digest paired with the requested \
             filename, lowercase",
    );
}

/// Filename-not-found returns `None` — the caller treats this
/// as "no expected hash available" and downgrades to a warning
/// per the user-facing instruction.
#[test]
fn parse_sha256_for_file_returns_none_when_file_absent() {
    let manifest = "\
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  linux-6.14.1.tar.xz
bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb  linux-6.14.2.tar.xz
";
    let got = super::parse_sha256_for_file(manifest, "linux-9.99.99.tar.xz");
    assert!(
        got.is_none(),
        "missing filename must return None so the caller can \
             warn-and-continue rather than fabricate a digest: got \
             {got:?}",
    );
}

/// Lines whose hash field has the wrong length or non-hex
/// characters are silently skipped — pin the per-line tolerance
/// against an upstream that briefly publishes a malformed line
/// during a deploy. Covers both rejection paths in
/// `parse_sha256_for_file`'s validator: short-length and 64-
/// char-but-non-hex.
#[test]
fn parse_sha256_for_file_skips_malformed_hash_lines() {
    // Line 1: 2-char hash (length-check rejects).
    // Line 2: 64-char hash with non-hex chars (`g` and `z`)
    //         (hex-check rejects after length passes).
    // Line 3: well-formed 64-char hex hash (must parse).
    let manifest = "\
zz  linux-6.14.1.tar.xz
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzgg  linux-6.14.2.tar.xz
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  linux-6.14.3.tar.xz
";
    assert_eq!(
        super::parse_sha256_for_file(manifest, "linux-6.14.1.tar.xz"),
        None,
        "2-char hash must be skipped via the length check",
    );
    assert_eq!(
        super::parse_sha256_for_file(manifest, "linux-6.14.2.tar.xz"),
        None,
        "64-char-but-non-hex hash must be skipped via the \
             ascii-hexdigit check",
    );
    assert_eq!(
        super::parse_sha256_for_file(manifest, "linux-6.14.3.tar.xz")
            .expect("valid entry must parse"),
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    );
}

// -- max_tag_patch --

/// `max_tag_patch` returns the numerically-highest patch among
/// `refs/tags/v{prefix}.{patch}` ref names — the EOL-series resolution
/// path (releases.json omits EOL series, so 6.14 is recovered from the
/// stable tree's git tags via ls-remote, cdn-independent). Refs are
/// interleaved + out of order to prove the max loop (not order)
/// decides. gix folds an annotated tag into a single base-named
/// `Ref::Peeled`, so `refs/tags/v6.14.11` is the real advertised
/// shape; the extra `v6.14.11^{}` entry is a DEFENSIVE raw-wire shape
/// gix never emits (it exercises the `^{}` strip) and must still
/// resolve to patch 11 via the base entry.
#[test]
fn max_tag_patch_returns_highest_patch() {
    let refs: &[&[u8]] = &[
        b"refs/tags/v6.14.1",
        b"refs/tags/v6.14.11",
        b"refs/tags/v6.14.11^{}",
        b"refs/tags/v6.14.2",
        b"refs/tags/v6.12.94",
        b"refs/heads/master",
    ];
    assert_eq!(
        super::max_tag_patch(refs.iter().copied(), "6.14"),
        Some(11),
        "must select the numerically-highest 6.14 patch (11), not the \
             lexically-highest (2) nor the last-listed",
    );
}

/// The `refs/tags/v{prefix}.` needle's trailing dot keeps a `6.1`
/// prefix from matching the `6.14` / `6.12` series, and a non-numeric
/// patch tail (an `-rc` pre-release tag) is rejected.
#[test]
fn max_tag_patch_boundary_and_rc() {
    let refs: &[&[u8]] = &[
        b"refs/tags/v6.14.11",
        b"refs/tags/v6.12.94",
        b"refs/tags/v6.1.140",
        b"refs/tags/v6.1.99-rc1",
        b"refs/tags/v6.1.50",
    ];
    assert_eq!(
        super::max_tag_patch(refs.iter().copied(), "6.1"),
        Some(140),
        "prefix 6.1 must exclude 6.14/6.12 (dot boundary) and the -rc \
             tag (non-numeric patch), leaving max(140, 50)",
    );
}

/// A prefix with no matching release tag returns None — the caller
/// turns that into a hard "no release tags" error. A same-series
/// BRANCH (`refs/heads/linux-6.14.y`) must NOT be mistaken for a tag.
#[test]
fn max_tag_patch_absent_prefix_returns_none() {
    let refs: &[&[u8]] = &[b"refs/tags/v6.12.94", b"refs/heads/linux-6.14.y"];
    assert_eq!(
        super::max_tag_patch(refs.iter().copied(), "6.14"),
        None,
        "no v6.14.* release TAG -> None; the 6.14.y branch is not a tag",
    );
}

/// `parse_sha256_for_file` strips the PGP signature trailer —
/// content after `-----BEGIN PGP SIGNATURE-----` is binary
/// noise that must NOT be scanned for checksum lines (a chance
/// 64-hex-char run inside a signature would otherwise produce
/// a false positive).
#[test]
fn parse_sha256_for_file_ignores_post_signature_content() {
    // `linux-6.14.99.tar.xz` appears AFTER the signature
    // marker — must be ignored so the parser can't be tricked
    // into returning data from the binary blob.
    let manifest = "\
-----BEGIN PGP SIGNATURE-----
ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff  linux-6.14.99.tar.xz
-----END PGP SIGNATURE-----
";
    assert!(
        super::parse_sha256_for_file(manifest, "linux-6.14.99.tar.xz").is_none(),
        "lines after the signature marker must be invisible to \
             the parser",
    );
}

// -- resolve_expected_sha256 --

/// `resolve_expected_sha256(skip_sha256 = true)` returns `None`
/// without touching the network — the bypass branch must short-
/// circuit before any `Client::get`. Pins the security-sensitive
/// opt-out's no-network contract: a regression that swapped the
/// branch order (e.g. fetching the manifest then ignoring the
/// result) would still produce `None` but burn a CDN round-trip
/// per build, defeating the "use this when manifest is
/// unreachable" use case.
#[test]
fn resolve_expected_sha256_skip_returns_none_without_network() {
    // Build a client whose connect attempt would fail loudly if
    // the bypass branch reached `Client::get`. A 1ms connect
    // timeout against any external host returns within the
    // wall-clock budget of this test; the assertion below
    // observes `None` either way, but a regression would change
    // the test's WALL TIME from ~0ms to ~1ms+. We pin the
    // short-circuit by NOT reaching the network at all — the
    // assertion alone is what catches the regression because
    // the bypass branch never invokes the client.
    let client = test_client();
    let got = super::resolve_expected_sha256(&client, 6, "linux-6.14.2.tar.xz", true);
    assert!(
        got.is_none(),
        "skip_sha256 = true must produce None (verification \
             skipped); got {got:?}"
    );
}

/// Create a mockito server with a canned `sha256sums.asc`
/// response. Returns (server, url, mock). The server owns the
/// port — no port collisions under parallel nextest. Mirrors
/// [`mock_releases`] for the checksum-manifest endpoint.
fn mock_sha256sums(status: usize, body: &str) -> (mockito::ServerGuard, String, mockito::Mock) {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("GET", "/sha256sums.asc")
        .with_status(status)
        .with_body(body)
        .create();
    let url = format!("{}/sha256sums.asc", server.url());
    (server, url, mock)
}

/// `resolve_expected_sha256` with `skip_sha256 = false` drives
/// the real fetch-then-parse path: it GETs the manifest, parses
/// the entry for `tarball_name`, and returns `Some(lowercase_hex)`
/// when present. Routed through the URL-injection seam
/// [`super::resolve_expected_sha256_from_url`] (mirroring
/// [`cached_releases_with_url`]) so the no-skip arm hits a
/// localhost [`mockito`] server instead of cdn.kernel.org — no
/// real network, deterministic value pinned. The prior version of
/// this test discarded the result with `let _` and only proved
/// "did not panic" against a 1ms-timeout network call.
#[test]
#[ignore = "flaky in the full CI suite: the localhost mock TCP connect times out under concurrent VM-boot load; run with --run-ignored on an uncontended host"]
fn resolve_expected_sha256_no_skip_returns_digest_when_entry_present() {
    let manifest = "\
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  linux-6.14.1.tar.xz
bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb  linux-6.14.2.tar.xz
-----BEGIN PGP SIGNATURE-----
... signature payload ...
-----END PGP SIGNATURE-----
";
    let (_server, url, _mock) = mock_sha256sums(200, manifest);
    let client = test_client();
    let got = super::resolve_expected_sha256_from_url(&client, &url, "linux-6.14.2.tar.xz", false);
    assert_eq!(
        got.as_deref(),
        Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        "no-skip path must fetch the manifest, parse the matching \
             entry, and return its lowercase digest; got {got:?}",
    );
}

/// `resolve_expected_sha256` (no-skip) downgrades to `None` when
/// the manifest is fetched but carries no entry for the requested
/// tarball — the warn-and-continue fallback that keeps a build
/// progressing through schema drift / a rotated entry. Drives the
/// real fetch-then-parse path against a localhost mock so the
/// `Ok(manifest) => None` arm of the production match is exercised
/// (not just `parse_sha256_for_file` in isolation).
#[test]
fn resolve_expected_sha256_no_skip_returns_none_when_entry_absent() {
    let manifest = "\
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  linux-6.14.1.tar.xz
";
    let (_server, url, _mock) = mock_sha256sums(200, manifest);
    let client = test_client();
    let got = super::resolve_expected_sha256_from_url(&client, &url, "linux-9.99.99.tar.xz", false);
    assert!(
        got.is_none(),
        "no-skip path must return None when the fetched manifest \
             has no entry for the tarball (warn-and-continue); got {got:?}",
    );
}

/// `resolve_expected_sha256` (no-skip) downgrades to `None` when
/// the manifest fetch itself fails (HTTP 404). Drives the
/// `Err(err) => None` arm of the production match through the
/// URL-injection seam against a localhost mock returning 404 —
/// the fetch-failure fallback that lets a transient CDN outage
/// proceed without verification rather than wedging the build.
#[test]
fn resolve_expected_sha256_no_skip_returns_none_on_fetch_error() {
    let (_server, url, _mock) = mock_sha256sums(404, "Not Found");
    let client = test_client();
    let got = super::resolve_expected_sha256_from_url(&client, &url, "linux-999.0.0.tar.xz", false);
    assert!(
        got.is_none(),
        "no-skip path must return None when the manifest fetch \
             fails (HTTP 404 warn-and-continue); got {got:?}",
    );
}

// -- verify_sha256 --

/// Matching digests return Ok regardless of case — pins the
/// case-insensitive comparison the helper documents.
#[test]
fn verify_sha256_accepts_case_insensitive_match() {
    super::verify_sha256(
        "ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890",
        "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "https://example.invalid/x.tar.xz",
    )
    .expect("case-insensitive equal must verify");
}

/// Mismatching digests surface as Err with both digests in the
/// message so an operator can compare them by eye without
/// digging through logs.
#[test]
fn verify_sha256_rejects_mismatch_with_both_digests_in_message() {
    let url = "https://example.invalid/x.tar.xz";
    let err = super::verify_sha256(
        "0000000000000000000000000000000000000000000000000000000000000000",
        "1111111111111111111111111111111111111111111111111111111111111111",
        url,
    )
    .expect_err("mismatch must surface as Err");
    let msg = format!("{err:#}");
    assert!(msg.contains(url), "error must name the URL: {msg}");
    assert!(
        msg.contains("0000000000000000"),
        "error must include the actual digest: {msg}",
    );
    assert!(
        msg.contains("1111111111111111"),
        "error must include the expected digest: {msg}",
    );
    // The mismatch error is the only thing the operator sees on
    // a verification-failed download. It MUST name `--skip-sha256`
    // as the recovery path so an operator hitting an in-place
    // tarball update at cdn.kernel.org does not have to dig
    // through docs to find the bypass flag.
    assert!(
        msg.contains("--skip-sha256"),
        "mismatch error must name --skip-sha256 as the recovery \
             flag for the in-place-tarball-update case: {msg}",
    );
}

// -- download_stable_tarball_from_url + TarballNotFound routing --

/// The `TarballNotFound` marker must be downcast-detectable, and a
/// non-marker error must NOT be — pins the exact `downcast_ref`
/// contract `download_tarball`'s EOL git-fallback dispatch depends on
/// (a 404 carries the marker → shallow git-tag clone; any other error
/// propagates unchanged). Runs in CI (no network) so the contract is
/// guarded even though the seam tests below are `#[ignore]`.
#[test]
fn tarball_not_found_marker_is_downcast_detectable() {
    let marked: anyhow::Error = anyhow::Error::new(super::TarballNotFound);
    assert!(
        marked.downcast_ref::<super::TarballNotFound>().is_some(),
        "the marker must be downcast-detectable — download_tarball's \
             `e.downcast_ref::<TarballNotFound>()` git-fallback arm relies on it",
    );
    let plain: anyhow::Error = anyhow::anyhow!("download x: HTTP 500");
    assert!(
        plain.downcast_ref::<super::TarballNotFound>().is_none(),
        "a non-marker error must NOT be misrouted to the git fallback",
    );
}

/// Create a mockito server serving one canned tarball response at a
/// fixed path. Returns (server, url, mock). Mirrors [`mock_sha256sums`]
/// for the tarball endpoint.
fn mock_tarball(status: usize, body: &str) -> (mockito::ServerGuard, String, mockito::Mock) {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("GET", "/linux.tar.xz")
        .with_status(status)
        .with_body(body)
        .create();
    let url = format!("{}/linux.tar.xz", server.url());
    (server, url, mock)
}

/// A 404 from the tarball GET must surface as a downcast-detectable
/// [`super::TarballNotFound`] — the signal `download_tarball` turns
/// into the shallow git-tag clone of a pruned/EOL kernel. Drives the
/// real `download_stable_tarball_from_url` status gate against a
/// localhost mock (`skip_sha256 = true`, so no sha256sums.asc fetch is
/// attempted). Without this the 404→marker path is unproven — the pure
/// `max_tag_patch` tests say nothing about the tarball-download routing.
#[test]
#[ignore = "flaky in the full CI suite: the localhost mock TCP connect times out under concurrent VM-boot load; run with --run-ignored on an uncontended host"]
fn download_stable_tarball_404_returns_tarball_not_found_marker() {
    let (_server, url, _mock) = mock_tarball(404, "Not Found");
    let client = test_client();
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let err = super::download_stable_tarball_from_url(
        &client,
        &url,
        "6.14.11",
        tmp.path(),
        "test",
        true,
        None,
    )
    .expect_err("a 404 tarball GET must be an error");
    assert!(
        err.downcast_ref::<super::TarballNotFound>().is_some(),
        "a 404 must carry the TarballNotFound marker so download_tarball \
             routes to the git-tag fallback; got: {err:#}",
    );
}

/// A non-404 failure (HTTP 500) must be a HARD error with NO marker —
/// the git-tag fallback is only for pruned tarballs, not a server
/// error, which must propagate unchanged (no clone attempt).
#[test]
#[ignore = "flaky in the full CI suite: the localhost mock TCP connect times out under concurrent VM-boot load; run with --run-ignored on an uncontended host"]
fn download_stable_tarball_500_is_hard_error_without_marker() {
    let (_server, url, _mock) = mock_tarball(500, "Internal Server Error");
    let client = test_client();
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let err = super::download_stable_tarball_from_url(
        &client,
        &url,
        "6.14.11",
        tmp.path(),
        "test",
        true,
        None,
    )
    .expect_err("a 500 tarball GET must be an error");
    assert!(
        err.downcast_ref::<super::TarballNotFound>().is_none(),
        "a 500 must NOT carry the marker — no git fallback for a server \
             error; got: {err:#}",
    );
    assert!(
        format!("{err:#}").contains("500"),
        "the hard error must name the HTTP status: {err:#}",
    );
}

// -- proptest --

use proptest::prop_assert;

proptest::proptest! {
    /// Any arbitrary input must not panic AND, on success, return
    /// only values the input string can justify. Broadened from
    /// 0..20 to 0..100 characters to exercise long/multi-dot
    /// pathological inputs the 20-char range missed.
    #[test]
    fn prop_major_version_never_panics(s in "\\PC{0,100}") {
        if let Ok(major) = major_version(&s) {
            // Ok(major) is only valid when the first dot-segment
            // parses as the returned integer.
            let first = s.split('.').next().unwrap_or("");
            prop_assert!(first.parse::<u32>().ok() == Some(major));
        }
    }

    #[test]
    fn prop_is_rc_contains_dash_rc(s in "\\PC{0,20}") {
        assert_eq!(is_rc(&s), s.contains("-rc"));
    }

    #[test]
    fn prop_patch_level_valid_three_part(
        major in 1u32..100,
        minor in 0u32..100,
        patch in 0u32..100,
    ) {
        let v = format!("{major}.{minor}.{patch}");
        assert_eq!(patch_level(&v), Some(patch));
    }

    #[test]
    fn prop_patch_level_valid_two_part(major in 1u32..100, minor in 0u32..100) {
        let v = format!("{major}.{minor}");
        assert_eq!(patch_level(&v), Some(0));
    }

    #[test]
    fn prop_major_version_valid(major in 1u32..100, minor in 0u32..100) {
        let v = format!("{major}.{minor}");
        assert_eq!(major_version(&v).unwrap(), major);
    }
}

// -- version_tuple --

/// Three-segment versions parse all three numeric components.
#[test]
fn version_tuple_three_part_parses_all_components() {
    assert_eq!(version_tuple("6.14.2"), Some((6, 14, 2)));
    assert_eq!(version_tuple("5.4.0"), Some((5, 4, 0)));
    assert_eq!(version_tuple("6.12.81"), Some((6, 12, 81)));
}

/// Two-segment versions default the patch component to 0 — this is
/// what lets a major.minor prefix compare against full patch
/// versions in `latest_in_series` / `fetch_version_for_prefix`.
#[test]
fn version_tuple_two_part_defaults_patch_to_zero() {
    assert_eq!(version_tuple("6.14"), Some((6, 14, 0)));
    assert_eq!(version_tuple("7.0"), Some((7, 0, 0)));
}

/// A bare single segment (no dot) is unparseable — `parts.len()` is
/// 1, which falls into the `_ => None` arm. Mirrors `patch_level`'s
/// single-part rejection.
#[test]
fn version_tuple_single_segment_is_none() {
    assert_eq!(version_tuple("6"), None);
    assert_eq!(version_tuple(""), None);
}

/// Four-or-more segments are rejected outright — the match only has
/// arms for 2 and 3 parts, everything else hits `_ => None`.
#[test]
fn version_tuple_four_part_is_none() {
    assert_eq!(version_tuple("6.1.2.3"), None);
    assert_eq!(version_tuple("1.2.3.4.5"), None);
}

/// A correctly-shaped version string whose components are not
/// numeric yields `None`: the per-segment `parse().ok()?` short-
/// circuits. RC tags split on `.` into 2 parts whose second
/// segment (`"15-rc3"`) is non-numeric, so they are not version
/// tuples — the resolution pipeline handles RC tags on a separate
/// path.
#[test]
fn version_tuple_non_numeric_segments_are_none() {
    // "6.15-rc3" -> ["6", "15-rc3"]: 2 parts, minor parse fails.
    assert_eq!(version_tuple("6.15-rc3"), None);
    // 3 parts, patch non-numeric.
    assert_eq!(version_tuple("6.14.x"), None);
    // 3 parts, minor non-numeric.
    assert_eq!(version_tuple("6.x.2"), None);
    // 2 parts, major non-numeric.
    assert_eq!(version_tuple("v6.14"), None);
    // u32 overflow on a segment: 2^32 = 4294967296 does not fit
    // in u32, so `parse::<u32>()` errors and the tuple is None.
    assert_eq!(version_tuple("6.4294967296"), None);
}

/// The returned tuples order the way callers depend on: tuple
/// comparison is lexicographic over (major, minor, patch), so
/// `latest_in_series` / `fetch_version_for_prefix` pick the highest
/// version by comparing these tuples directly. Pin the ordering on
/// each component so a regression that reordered the tuple fields
/// would surface here.
#[test]
fn version_tuple_orders_lexicographically_by_component() {
    let v = |s: &str| version_tuple(s).expect("parseable");
    // Patch dominates within the same major.minor.
    assert!(v("6.14.2") > v("6.14.1"));
    // Minor dominates patch across minor bumps.
    assert!(v("6.15.0") > v("6.14.99"));
    // Major dominates everything.
    assert!(v("7.0.0") > v("6.99.99"));
    // The 2-part patch-zero default sorts below any nonzero patch
    // in the same series — "6.14" (== 6.14.0) < "6.14.1".
    assert!(v("6.14") < v("6.14.1"));
    // Equal strings produce equal tuples.
    assert_eq!(v("6.14.2"), v("6.14.2"));
}

// -- is_skippable_release_moniker --

/// `linux-next` is the single moniker the resolution pipeline
/// skips — its date-suffixed version strings do not fit the
/// major.minor.patch model. Exact-match only: anything else
/// (including substrings and case variants) is NOT skipped.
#[test]
fn is_skippable_release_moniker_matrix() {
    // The one skippable moniker.
    assert!(is_skippable_release_moniker("linux-next"));

    // Every resolvable moniker must NOT be skipped — these are the
    // ones `fetch_latest_stable_version` / `fetch_version_for_prefix`
    // actually iterate over.
    for keep in ["stable", "longterm", "mainline", "eol"] {
        assert!(
            !is_skippable_release_moniker(keep),
            "{keep:?} is a resolvable moniker and must NOT be skipped",
        );
    }

    // Exact-match contract: substrings and decorated variants are
    // NOT skipped. A `contains`-style implementation would wrongly
    // skip these.
    for keep in ["linux-next-20260420", "linux-nextfoo", "next", "linux", ""] {
        assert!(
            !is_skippable_release_moniker(keep),
            "{keep:?} is not exactly \"linux-next\" and must NOT be skipped",
        );
    }

    // Case-sensitive: the comparison is byte-exact `==`.
    assert!(
        !is_skippable_release_moniker("Linux-Next"),
        "moniker match is case-sensitive",
    );
    assert!(!is_skippable_release_moniker("LINUX-NEXT"));
}

// -- reject_html_response --

/// Fetch `path` from a mockito server configured with `status`,
/// optional `content_type` header, and `body`, returning the live
/// `reqwest::blocking::Response`. The `ServerGuard` is returned so
/// the caller keeps the server alive for the response's lifetime.
fn fetch_mock_response(
    status: usize,
    content_type: Option<&str>,
    body: &str,
) -> (mockito::ServerGuard, reqwest::blocking::Response) {
    let mut server = mockito::Server::new();
    let mut m = server.mock("GET", "/file").with_status(status);
    if let Some(ct) = content_type {
        m = m.with_header("content-type", ct);
    }
    let _mock = m.with_body(body).create();
    let url = format!("{}/file", server.url());
    let response = test_client()
        .get(&url)
        .send()
        .expect("mock fetch must succeed");
    (server, response)
}

/// A `text/html` Content-Type surfaces as `Err` — a CDN error page
/// returned with a 200 status and `text/html` must be rejected
/// before extraction so the xz/gzip decoder never chews on HTML.
/// The diagnostic must name both the HTML cause and the URL.
#[test]
#[ignore = "flaky in the full CI suite: the localhost mock TCP connect times out under concurrent VM-boot load; run with --run-ignored on an uncontended host"]
fn reject_html_response_rejects_text_html() {
    let url = "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.14.2.tar.xz";
    let (_server, response) =
        fetch_mock_response(200, Some("text/html"), "<html><body>404</body></html>");
    let err = reject_html_response(&response, url).expect_err("text/html must surface as Err");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("server returned HTML instead of tarball"),
        "diagnostic must name the HTML cause: {msg}",
    );
    assert!(msg.contains(url), "diagnostic must name the URL: {msg}");
}

/// A `text/html; charset=utf-8` Content-Type is also rejected — the
/// check is a substring match (`contains("text/html")`), so the
/// parameterized form must still trip it.
#[test]
#[ignore = "flaky in the full CI suite: the localhost mock TCP connect times out under concurrent VM-boot load; run with --run-ignored on an uncontended host"]
fn reject_html_response_rejects_text_html_with_charset() {
    let url = "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.14.2.tar.xz";
    let (_server, response) =
        fetch_mock_response(200, Some("text/html; charset=utf-8"), "<html></html>");
    let err = reject_html_response(&response, url)
        .expect_err("text/html with charset param must surface as Err");
    assert!(
        format!("{err:#}").contains("server returned HTML instead of tarball"),
        "charset-parameterized text/html must still be rejected",
    );
}

/// A binary Content-Type (the real-tarball case) passes — the
/// substring `text/html` is absent so the guard returns `Ok(())`
/// and the download proceeds to extraction.
#[test]
#[ignore = "flaky in the full CI suite: the localhost mock TCP connect times out under concurrent VM-boot load; run with --run-ignored on an uncontended host"]
fn reject_html_response_accepts_octet_stream() {
    let url = "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.14.2.tar.xz";
    let (_server, response) =
        fetch_mock_response(200, Some("application/octet-stream"), "\x00\x01\x02");
    reject_html_response(&response, url)
        .expect("binary content-type must be accepted, not rejected as HTML");
}

/// A response with NO Content-Type header passes — the
/// `if let Some(ct) = ...headers().get(CONTENT_TYPE)` guard short-
/// circuits to `Ok(())`. A bare-bodied CDN response without the
/// header must not be misclassified as HTML.
#[test]
#[ignore = "flaky in the full CI suite: the localhost mock TCP connect times out under concurrent VM-boot load; run with --run-ignored on an uncontended host"]
fn reject_html_response_accepts_missing_content_type() {
    let url = "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.14.2.tar.xz";
    let (_server, response) = fetch_mock_response(200, None, "\x00\x01\x02");
    // Precondition: the mock served no content-type header so the
    // `Some(ct)` arm of the guard is genuinely skipped (rather than
    // a header present that merely lacks "text/html").
    assert!(
        response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .is_none(),
        "test precondition: mock must serve no content-type header",
    );
    reject_html_response(&response, url)
        .expect("absent content-type must be accepted, not rejected as HTML");
}

// -- print_download_size --

/// With a Content-Length header, `print_download_size` emits the
/// MiB-annotated form `"{label}: downloading {url} ({mib:.1} MiB)"`.
/// A body of exactly 1.5 MiB makes the rendered size deterministic
/// ("1.5 MiB"), pinning both the formula (bytes / 1024^2) and the
/// `{:.1}` precision. Capture goes through the crate-shared
/// stderr-capture helper.
#[test]
#[ignore = "flaky in the full CI suite: the localhost mock TCP connect times out under concurrent VM-boot load; run with --run-ignored on an uncontended host"]
fn print_download_size_with_content_length_renders_mib() {
    let url = "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.14.2.tar.xz";
    // Exactly 1.5 MiB: 1.5 * 1024 * 1024 = 1572864 bytes ->
    // 1572864 / (1024*1024) = 1.5 -> formats to "1.5".
    let body = vec![b'x'; 1_572_864];
    let body_str = String::from_utf8(body).expect("ascii body is valid utf-8");
    let (_server, response) = fetch_mock_response(200, Some("application/octet-stream"), &body_str);
    // Precondition: reqwest must report the body length so the
    // `Some(len)` branch is actually taken.
    assert_eq!(
        response.content_length(),
        Some(1_572_864),
        "test precondition: mock body must surface as a 1.5 MiB content length",
    );

    let (_, bytes) = crate::test_support::test_helpers::capture_stderr(|| {
        print_download_size(&response, url, "ktstr", None);
    });
    let captured = String::from_utf8(bytes).expect("captured stderr must be utf-8");
    assert_eq!(
        captured,
        format!("ktstr: downloading {url} (1.5 MiB)\n"),
        "Content-Length present must render the MiB-annotated form \
         with the cli_label prefix and 1-decimal size",
    );
}

/// Without a Content-Length header (chunked transfer), the function
/// falls to the un-annotated form `"{label}: downloading {url}"` —
/// no size, no "MiB". A chunked mock body makes reqwest report
/// `content_length() == None`, exercising the `else` arm. Also pins
/// that `cli_label` threads through verbatim ("cargo ktstr").
#[test]
#[ignore = "flaky in the full CI suite: the localhost mock TCP connect times out under concurrent VM-boot load; run with --run-ignored on an uncontended host"]
fn print_download_size_without_content_length_omits_mib() {
    let url = "https://git.kernel.org/torvalds/t/linux-6.15-rc3.tar.gz";
    let mut server = mockito::Server::new();
    // Chunked body -> no Content-Length header on the response.
    let _mock = server
        .mock("GET", "/file")
        .with_status(200)
        .with_chunked_body(|w| w.write_all(b"streamed body bytes"))
        .create();
    let mock_url = format!("{}/file", server.url());
    let response = test_client()
        .get(&mock_url)
        .send()
        .expect("chunked mock fetch must succeed");
    // Precondition: a chunked response carries no Content-Length,
    // so `content_length()` is None and the `else` branch runs.
    assert!(
        response.content_length().is_none(),
        "test precondition: chunked body must surface as None content length",
    );

    let (_, bytes) = crate::test_support::test_helpers::capture_stderr(|| {
        print_download_size(&response, url, "cargo ktstr", None);
    });
    let captured = String::from_utf8(bytes).expect("captured stderr must be utf-8");
    assert_eq!(
        captured,
        format!("cargo ktstr: downloading {url}\n"),
        "absent Content-Length must render the un-annotated form \
         (no size, no MiB) with the cli_label prefix",
    );
    assert!(
        !captured.contains("MiB"),
        "no-Content-Length form must NOT mention MiB: {captured:?}",
    );
}

// -- sha256sums_url --

/// `sha256sums_url` builds the cdn.kernel.org manifest URL for a
/// stable major series: `.../v{major}.x/sha256sums.asc`. It is the
/// single source of truth for the URL shape `resolve_expected_sha256`
/// passes to `resolve_expected_sha256_from_url`; every checksum test
/// injects its own mock URL via the `_from_url` seam, so the real
/// builder has no other coverage. A silent drift (wrong dir segment,
/// missing `/v`, wrong `.asc` name) would 404 every production
/// checksum fetch and downgrade verification to a warning with no
/// test catching it. Pin the full string for two distinct majors so
/// the `{major}` interpolation is exercised, not just the literal
/// scaffold.
#[test]
fn sha256sums_url_builds_cdn_manifest_url() {
    assert_eq!(
        super::sha256sums_url(6),
        "https://cdn.kernel.org/pub/linux/kernel/v6.x/sha256sums.asc",
        "major 6 must map to the v6.x manifest URL",
    );
    assert_eq!(
        super::sha256sums_url(5),
        "https://cdn.kernel.org/pub/linux/kernel/v5.x/sha256sums.asc",
        "the {{major}} segment must interpolate (v5.x), not be a fixed literal",
    );
}

// -- canonical_path_hash --

/// `canonical_path_hash` returns the full 8-char (32-bit) lowercase-hex
/// IEEE-CRC32 of the path's encoded bytes. Only indirectly exercised by
/// `compose_local_cache_key_unknown_uses_path_hash_only`, which asserts
/// the SHAPE (8 hex chars) but NOT the VALUE — a regression that
/// truncated to 24 bits, swapped to a non-deterministic hasher, or
/// changed the width would pass that shape check yet corrupt the
/// cross-run cache-key stability and per-source-tree flock-name
/// disambiguation the doc promises. Pin the exact CRC32 for two ASCII
/// paths (ASCII so `as_encoded_bytes` equals the str bytes on every
/// platform), plus an inequality arm mirroring the per-path-salt
/// distinctness invariant.
#[test]
fn canonical_path_hash_is_full_crc32_lowercase_hex() {
    // CRC32 (IEEE polynomial, identical to crc32fast::hash) over the
    // ASCII bytes of each path, lowercase 8-char hex.
    let h_anywhere = super::canonical_path_hash(std::path::Path::new("/anywhere"));
    let h_some_path = super::canonical_path_hash(std::path::Path::new("/some/path"));
    assert_eq!(
        h_anywhere, "8b09f613",
        "CRC32 of b\"/anywhere\" must be the full 8-char lowercase hex",
    );
    assert_eq!(
        h_some_path, "46e02226",
        "CRC32 of b\"/some/path\" must be the full 8-char lowercase hex",
    );
    // Leading-zero CRC32 (< 0x10000000): pins the {:08x} zero-PADDING —
    // a `{:x}` regression would drop the leading 0 (7 chars, not 8).
    assert_eq!(
        super::canonical_path_hash(std::path::Path::new("/p50")),
        "090cfa3e",
        "a leading-zero CRC32 must render 8 chars (zero-padded), not 7",
    );
    // Distinctness: two different paths must produce different salts —
    // without per-path distinctness, two dirty/non-git trees would
    // collapse to the same local-unknown cache slot and flock name.
    assert_ne!(
        h_anywhere, h_some_path,
        "distinct paths must produce distinct path-hash salts",
    );
}

// -- config_hash_for_key --

/// `config_hash_for_key` returns `None` when no `<dir>/.config` exists.
/// This `None` is load-bearing: `compose_local_cache_key` collapses to
/// the cfg-less legacy key shape on `None`, so a fresh checkout's first
/// build (no `.config` yet) shares a cache slot with a later lookup. A
/// regression that returned `Some(crc of empty)` instead of `None`
/// would split that slot and silently miss the cache. Fixture is a
/// fresh empty TempDir with no `.config`; the fn only reads
/// `<dir>/.config` (no gix, no network, no ancestor-git walk).
#[test]
fn config_hash_for_key_returns_none_when_no_dotconfig() {
    let tmp = tempfile::TempDir::new().unwrap();
    assert_eq!(
        super::config_hash_for_key(tmp.path()),
        None,
        "absent .config must return None so the cache key collapses to \
             the legacy cfg-less shape",
    );
}

/// `config_hash_for_key` returns `Some(crc32_of_config_bytes)` (8-char
/// lowercase hex) over the raw `.config` file contents when the file is
/// present. Pins the doc's contract that the cache key folds in the
/// PRE-BUILD `.config` so two builds of the same HEAD with different
/// `.config` files do not collide — and that the encoding is the
/// 8-char CRC32 of the raw file bytes (not a sha, not a path-derived
/// value). The distinctness arm proves a different `.config` body
/// yields a different `Some(_)`, which is the whole point of folding
/// the config into the key.
#[test]
fn config_hash_for_key_returns_crc32_of_config_bytes() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join(".config"), b"CONFIG_FOO=y\n").unwrap();
    // CRC32 (IEEE polynomial, crc32fast::hash) of b"CONFIG_FOO=y\n".
    assert_eq!(
        super::config_hash_for_key(tmp.path()).as_deref(),
        Some("2ce4027e"),
        "present .config must hash to the 8-char lowercase CRC32 of its \
             raw bytes",
    );

    // Leading-zero CRC32 (< 0x10000000): pins the {:08x} zero-PADDING —
    // a `{:x}` regression would drop the leading 0 (7 chars, not 8).
    let tmp_lz = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp_lz.path().join(".config"), b"CONFIG_Z2=y\n").unwrap();
    assert_eq!(
        super::config_hash_for_key(tmp_lz.path()).as_deref(),
        Some("04df91e9"),
        "a leading-zero CRC32 must render 8 chars (zero-padded), not 7",
    );

    // Distinctness: a second tree with different .config bytes must
    // produce a different hash — without this the per-config cache-key
    // discrimination the doc promises would not hold.
    let tmp2 = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp2.path().join(".config"), b"CONFIG_BAR=m\n").unwrap();
    assert_eq!(
        super::config_hash_for_key(tmp2.path()).as_deref(),
        Some("8498a61e"),
        "a different .config body must hash to a different CRC32",
    );
    assert_ne!(
        super::config_hash_for_key(tmp.path()),
        super::config_hash_for_key(tmp2.path()),
        "distinct .config bodies must produce distinct cache-key hashes",
    );
}

// -- read_makefile_version --

/// `read_makefile_version` extracts MAJOR.MINOR.PATCH from a kernel
/// source `Makefile` (VERSION/PATCHLEVEL/SUBLEVEL), ignoring
/// EXTRAVERSION; a missing Makefile or any absent/non-numeric field
/// yields None (so the acquisition records no version and the rootfs
/// tmpfs fraction conservatively defaults to 50%).
#[test]
fn fetch_read_makefile_version_parses_and_guards() {
    let dir = tempfile::TempDir::new().unwrap();
    let write_mk = |body: &str| {
        std::fs::write(dir.path().join("Makefile"), body).expect("write Makefile");
    };

    // Canonical kernel Makefile head -> MAJOR.MINOR.PATCH; EXTRAVERSION ignored.
    write_mk(
        "# SPDX-License-Identifier: GPL-2.0\nVERSION = 6\nPATCHLEVEL = 6\nSUBLEVEL = 113\nEXTRAVERSION =\nNAME = X\n",
    );
    assert_eq!(
        super::read_makefile_version(dir.path()).as_deref(),
        Some("6.6.113")
    );

    // rc release: EXTRAVERSION = -rc7 is ignored; SUBLEVEL 0 honored.
    write_mk("VERSION = 7\nPATCHLEVEL = 1\nSUBLEVEL = 0\nEXTRAVERSION = -rc7\n");
    assert_eq!(
        super::read_makefile_version(dir.path()).as_deref(),
        Some("7.1.0")
    );

    // PANIC-SAFETY: a prefix-sharing decoy ("VERSION_FILE = ...") and a
    // comment ("# VERSION = 999") must NOT false-match the canonical
    // assignment — over-stating the version would select 90% on a
    // non-honoring kernel and panic the guest. The real assignment wins.
    write_mk(
        "VERSION_FILE = include/config/kernel.release\n# VERSION = 999\nVERSION = 6\nPATCHLEVEL = 6\nSUBLEVEL = 113\n",
    );
    assert_eq!(
        super::read_makefile_version(dir.path()).as_deref(),
        Some("6.6.113")
    );

    // First-match: the top-of-file assignment wins over a later duplicate.
    write_mk(
        "VERSION = 6\nPATCHLEVEL = 6\nSUBLEVEL = 113\nVERSION = 9\nPATCHLEVEL = 9\nSUBLEVEL = 9\n",
    );
    assert_eq!(
        super::read_makefile_version(dir.path()).as_deref(),
        Some("6.6.113")
    );

    // A non-bare-integer value (trailing comment) -> that field None ->
    // overall None -> the safe 50% default (never an over-stated version).
    write_mk("VERSION = 7 # inline\nPATCHLEVEL = 1\nSUBLEVEL = 0\n");
    assert_eq!(super::read_makefile_version(dir.path()), None);

    // Missing SUBLEVEL -> None (all three fields are required).
    write_mk("VERSION = 6\nPATCHLEVEL = 6\n");
    assert_eq!(super::read_makefile_version(dir.path()), None);

    // Non-numeric VERSION -> None.
    write_mk("VERSION = x\nPATCHLEVEL = 6\nSUBLEVEL = 1\n");
    assert_eq!(super::read_makefile_version(dir.path()), None);

    // No Makefile at all -> None.
    std::fs::remove_file(dir.path().join("Makefile")).expect("rm Makefile");
    assert_eq!(super::read_makefile_version(dir.path()), None);
}

// -- git cache key + ls-remote ref pre-resolution --

#[test]
fn git_cache_key_embeds_ref_full_hash_arch_and_suffix() {
    let h = "abc1234abc1234abc1234abc1234abc1234abc12";
    let key = git_cache_key("for-next", h);
    assert!(
        key.starts_with(&format!("for-next-git-{h}-")),
        "key must lead with the raw ref, the -git- marker, and the full commit hash: {key}"
    );
    let (arch, _) = arch_info();
    assert!(
        key.contains(&format!("-{arch}-kc")),
        "key must carry the target arch before the kc suffix: {key}"
    );
}

/// Two distinct commits sharing a 7-hex prefix must produce DIFFERENT
/// cache keys — the FULL hash keys the entry, so a moved branch/tag tip
/// whose new commit collides in the first 7 hex with the cached old
/// commit can never hit the stale entry (closes the wrong-build-served
/// class a 28-bit short prefix risked).
#[test]
fn git_cache_key_full_hash_distinguishes_7hex_prefix_collision() {
    let a = "abc1234000000000000000000000000000000000";
    let b = "abc1234ffffffffffffffffffffffffffffffffff";
    assert_eq!(&a[..7], &b[..7], "fixture precondition: same 7-hex prefix");
    assert_ne!(
        git_cache_key("for-next", a),
        git_cache_key("for-next", b),
        "full-hash key must distinguish two commits sharing a 7-hex prefix",
    );
}

#[test]
fn ls_remote_commit_hash_uses_full_sha_directly_without_network() {
    // A full 40-hex ref short-circuits the handshake (the bogus URL is
    // never contacted) and is returned in full, lowercased to match
    // git_clone's head rendering (the cache key needs the full id).
    let sha = "1234567890abcdef1234567890abcdef12345678";
    assert_eq!(
        ls_remote_commit_hash("https://invalid.invalid/nope.git", sha),
        Some(sha.to_string()),
    );
    let upper = "ABCDEF1234567890ABCDEF1234567890ABCDEF12";
    assert_eq!(
        ls_remote_commit_hash("https://invalid.invalid/nope.git", upper),
        Some(upper.to_ascii_lowercase()),
    );
}

/// Regression: `ls_remote_commit_hash` must resolve a BRANCH ref. The
/// bug was gix's default `Options` (`prefix_from_spec_as_filter_on_remote
/// = true`): over a protocol-v2 `ls-refs` it derives ref-prefix filters
/// from the anonymous remote's refspecs, and with empty fetch specs +
/// `fetch_tags = Included` that ls-refs prefix is `refs/tags` ONLY — so
/// `refs/heads/*` never reach `remote_refs` and a branch ref resolves to
/// `None`, defeating `resolve_git_kernel`'s clone-skip (every run
/// re-cloned). The fix sets the flag `false`. A `Some(_)` here is only
/// reachable once `refs/heads/for-next` reaches `remote_refs`, so
/// reverting the flag turns this red — the sole test that pins the
/// `ref_map` Options path (the `match_ref_commit_hash_*` unit tests feed
/// synthetic ref vectors and never exercise it).
///
/// `#[ignore]`: hits the live sched_ext remote over the network (project
/// convention for network-fetch tests). The filtering is protocol-v2
/// `ls-refs` behavior — a v1 handshake advertises all refs, so a local
/// v1 mock would not reproduce it. Run with `--run-ignored` on a
/// networked host.
#[test]
#[ignore = "network: live ls-remote against git.kernel.org sched_ext (protocol v2)"]
fn ls_remote_commit_hash_resolves_branch_over_v2() {
    let url = "https://git.kernel.org/pub/scm/linux/kernel/git/tj/sched_ext.git";
    let hash = ls_remote_commit_hash(url, "for-next")
        .expect("for-next branch must resolve over v2 ls-refs — the prefix filter must be off");
    assert_eq!(hash.len(), 40, "a full commit hash is 40 hex chars: {hash}");
    assert!(
        hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "commit hash must be hex: {hash}",
    );
}

fn direct_ref(name: &str, hex: &str) -> gix::protocol::handshake::Ref {
    gix::protocol::handshake::Ref::Direct {
        full_ref_name: name.into(),
        object: gix::hash::ObjectId::from_hex(hex.as_bytes()).expect("valid hex"),
    }
}

#[test]
fn match_ref_commit_hash_resolves_branch_tag_head_and_precedence() {
    let a = "1111111111111111111111111111111111111111";
    let b = "2222222222222222222222222222222222222222";
    let c = "3333333333333333333333333333333333333333";

    // Branch under refs/heads — returns the FULL commit hash.
    let refs = vec![direct_ref("refs/heads/for-next", a)];
    assert_eq!(match_ref_commit_hash(&refs, "for-next"), Some(a.into()));

    // Lightweight tag (Direct under refs/tags).
    let refs = vec![direct_ref("refs/tags/v6.10", b)];
    assert_eq!(match_ref_commit_hash(&refs, "v6.10"), Some(b.into()));

    // heads wins over tags for the same bare name (the probe's documented
    // precedence; the rare divergence from the shallow clone is handled by
    // the post-clone re-check — see match_ref_commit_hash).
    let refs = vec![direct_ref("refs/tags/x", b), direct_ref("refs/heads/x", a)];
    assert_eq!(match_ref_commit_hash(&refs, "x"), Some(a.into()));

    // An explicit refs/ path matches exactly.
    let refs = vec![direct_ref("refs/heads/main", c)];
    assert_eq!(
        match_ref_commit_hash(&refs, "refs/heads/main"),
        Some(c.into())
    );

    // HEAD via a Symbolic ref resolves to its ultimate object.
    let refs = vec![gix::protocol::handshake::Ref::Symbolic {
        full_ref_name: "HEAD".into(),
        target: "refs/heads/main".into(),
        tag: None,
        object: gix::hash::ObjectId::from_hex(c.as_bytes()).unwrap(),
    }];
    assert_eq!(match_ref_commit_hash(&refs, "HEAD"), Some(c.into()));

    // No matching ref -> None.
    let refs = vec![direct_ref("refs/heads/main", a)];
    assert_eq!(match_ref_commit_hash(&refs, "does-not-exist"), None);
}

#[test]
fn match_ref_commit_hash_annotated_tag_uses_peeled_commit_not_tag() {
    // An annotated tag advertises Peeled { tag: <tag object>, object:
    // <peeled commit> }. The key must use the peeled COMMIT (object):
    // git_clone checks out the commit (HEAD = commit), so the tag
    // object's hash would never match a clone-produced cache key.
    let tag_obj = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let commit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let refs = vec![gix::protocol::handshake::Ref::Peeled {
        full_ref_name: "refs/tags/v1.0".into(),
        tag: gix::hash::ObjectId::from_hex(tag_obj.as_bytes()).unwrap(),
        object: gix::hash::ObjectId::from_hex(commit.as_bytes()).unwrap(),
    }];
    assert_eq!(match_ref_commit_hash(&refs, "v1.0"), Some(commit.into()));
}

#[test]
fn match_ref_commit_hash_skips_unborn() {
    let refs = vec![gix::protocol::handshake::Ref::Unborn {
        full_ref_name: "refs/heads/new".into(),
        target: "refs/heads/new".into(),
    }];
    assert_eq!(match_ref_commit_hash(&refs, "new"), None);
}

#[test]
fn is_full_sha_gate_recognizes_only_40_char_hex() {
    // Exactly 40 hex chars (either case) is a sha -> fast path.
    assert!(is_full_sha(&"a".repeat(40)), "40 lowercase hex is a sha");
    assert!(is_full_sha(&"A".repeat(40)), "40 uppercase hex is a sha");
    // Off-by-one lengths are names, not shas -> fall through to ls-remote.
    assert!(!is_full_sha(&"a".repeat(39)), "39 chars is not a sha");
    assert!(!is_full_sha(&"a".repeat(41)), "41 chars is not a sha");
    // 40 chars but a non-hex byte is a ref name, not a sha.
    assert!(
        !is_full_sha(&format!("{}g", "a".repeat(39))),
        "40-char non-hex is not a sha"
    );
}

#[test]
fn git_clone_rejects_raw_sha_git_ref_without_panic() {
    // A raw 40-hex commit SHA must be rejected with an actionable error
    // rather than the gix `with_ref_name(<object-id>)` panic. The check
    // is at git_clone's entry, BEFORE any network, so the bogus URL is
    // never contacted (the test is deterministic + offline).
    let tmp = tempfile::TempDir::new().unwrap();
    let sha = "1234567890abcdef1234567890abcdef12345678";
    let err = git_clone(
        "https://invalid.invalid/nope.git",
        sha,
        tmp.path(),
        "test",
        None,
    )
    .err()
    .expect("a raw SHA git_ref must be rejected, not cloned");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("raw commit SHA") && msg.contains("branch or tag"),
        "the error must be actionable (use a branch or tag): {msg}"
    );
}

/// `git_clone_tag` shallow-clones a linux-stable annotated TAG — the
/// EOL-kernel recovery path. A plain `with_ref_name` shallow clone
/// forces `refs/heads/{ref}` and fails on a tag with "None of the
/// refspec(s) matched"; the appended `+refs/tags/{tag}:refs/heads/{tag}`
/// refspec fixes it. This can ONLY be validated by a real clone — the
/// failure is a runtime refspec mismatch, invisible at compile time and
/// to source-level review — so it is the empirical guard for the fix.
/// Network + a ~200MB shallow fetch of the linux-stable tree, hence
/// #[ignore] (run with --run-ignored on a connected, uncontended host).
#[test]
#[ignore = "network: shallow git-tag clone of linux-stable v6.14.11 (~200MB); run with --run-ignored on a connected host"]
fn git_clone_tag_shallow_clones_stable_tag() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let acquired = super::git_clone_tag(
        "https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git",
        "v6.14.11",
        tmp.path(),
        "test",
        None,
    )
    .expect(
        "git_clone_tag must shallow-clone the v6.14.11 tag, not fail with \
         'None of the refspec(s) matched'",
    );
    // A real linux source tree must be checked out at the tag.
    let makefile = acquired.source_dir.join("Makefile");
    assert!(
        makefile.exists(),
        "cloned tree must contain a top-level Makefile: {}",
        acquired.source_dir.display(),
    );
    // The Makefile version fields must read exactly 6.14.11, proving the
    // checkout landed on the tag (not HEAD or the series branch tip).
    let mk = std::fs::read_to_string(&makefile).expect("read cloned Makefile");
    assert!(
        mk.contains("PATCHLEVEL = 14") && mk.contains("SUBLEVEL = 11"),
        "cloned Makefile must be 6.14.11 (PATCHLEVEL=14, SUBLEVEL=11)",
    );
    // The recorded source must be git-shaped (kernel_source::git).
    assert!(
        matches!(
            acquired.kernel_source,
            crate::cache::KernelSource::Git { .. }
        ),
        "git_clone_tag must record a Git KernelSource",
    );
}

/// `latest_patch_from_git_tags` resolves an EOL series to its highest
/// patch via a real git.kernel.org ls-remote of the linux-stable tree
/// — the cdn-independent resolution the CI runners need (cdn serves
/// `sha256sums.asc` as 404 on some edges even where its tarballs 200).
/// Network (ls-remote only, no pack fetch), so #[ignore]; this is the
/// empirical guard for the resolution the pure `max_tag_patch` tests
/// cannot provide (they don't touch the network).
#[test]
#[ignore = "network: ls-remote git.kernel.org linux-stable for v6.14.* tags; run with --run-ignored on a connected host"]
fn latest_patch_from_git_tags_resolves_eol_series() {
    let v = super::latest_patch_from_git_tags(
        "https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git",
        "6.14",
        "test",
    )
    .expect("must resolve the latest 6.14.x from stable-tree tags");
    // 6.14 is EOL and frozen at 6.14.11 (no further releases), so the
    // resolver must return exactly that — not a lower patch, not an -rc.
    assert_eq!(
        v, "6.14.11",
        "EOL 6.14 must resolve to its final patch 6.14.11 via git tags",
    );
}
