//! Disk-template cache cleanup: orphaned per-test temp-dir sweeping and
//! whole-cache teardown. Split from `mod.rs` for module locality; reaches
//! the cache-path and store helpers (`cache_root`, `store_atomic`,
//! `ensure_template`, `clone_to_per_test`, `lookup`, …) via `use super::*`.
use super::*;

/// Sweep stale staging debris out of the disk-template cache root.
///
/// Three debris shapes accumulate when a template-build peer or a
/// per-test consumer dies before its cleanup arm completes:
///
/// 1. **`template.img.in-flight.<cache_key>.<pid>`** — sparse
///    staging images created by [`create_and_size_staging_image`]
///    when [`build_template_via_vm`] runs. Normally unlinked at
///    the failure-cleanup arms inside that function, AND moved
///    into a `.tmp.<pid>/` directory by `store_atomic` on success.
///    A SIGKILL between size-up and store_atomic leaks the file.
/// 2. **`<cache_key>.tmp.<pid>/`** — staging directories created
///    by [`store_atomic`] for the rename-into-place dance.
///    Normally renamed onto the final `<cache_key>/` directory at
///    the end of `store_atomic`. A SIGKILL during the
///    src→staging_image rename or the staging→final_dir rename
///    leaves the populated tmpdir on disk.
/// 3. **`.per-test-<pid>-<ns>-<rnd>.img`** — per-test FICLONE
///    backing files created by `KtstrVm::init_virtio_blk`
///    for the `Filesystem::Btrfs` branch. The setter unlinks the
///    path immediately after FICLONE (the open `File` keeps the
///    inode alive for the device's lifetime), but a SIGKILL
///    between FICLONE and unlink — or an unlink failure surfaced
///    only as a `tracing::warn!` — leaves the dest path on disk.
///    Without sweeping, every crashed test accumulates one such
///    file in the cache root forever.
///
/// All three shapes embed the originating peer's pid in the filename.
/// The sweep parses that pid and probes liveness via
/// `kill(pid, None)` (rust-side: [`nix::sys::signal::kill`] passing
/// `Option::None` for the signal — no signal, probe only). The
/// kernel returns:
/// - `Ok(())` — pid is live AND in-policy for our uid (the signal
///   COULD have been delivered). Debris is owned by a peer that
///   may still publish; leave alone.
/// - `Err(ESRCH)` — pid does not exist. Debris is safe to remove.
/// - `Err(EPERM)` — pid is live but owned by a different uid.
///   Not ours to clean up; leave alone.
/// - any other errno — treat as live and skip; false negatives
///   (debris left on disk) are recoverable, false positives
///   (deleting live state) are not.
///
/// Mirrors `crate::cache::housekeeping::clean_orphaned_tmp_dirs` in
/// `src/cache/housekeeping.rs` — the disk-template cache and the kernel-image
/// cache use the same pid-in-suffix + ESRCH-probe contract for
/// cross-process cleanup. The two are independent because their
/// debris namespaces don't overlap (kernel cache uses `.tmp-`
/// prefix, disk-template cache uses `.tmp.` infix on the
/// directories, a `template.img.in-flight.` prefix on the
/// staging images, and a `.per-test-` prefix on per-test
/// backing files).
///
/// Returns the count of debris entries removed. Errors during
/// individual `remove_dir_all` / `remove_file` calls are logged
/// at `warn` and the sweep continues — operator visibility into
/// "this entry could not be cleaned" beats abandoning the rest of
/// the sweep on the first failure.
///
/// Refuses to descend into the `.locks/` subdirectory (the only
/// non-debris namespace inside the cache root); the prefix filter
/// excludes it via the `template.img.in-flight.`, `*.tmp.*`, and
/// `.per-test-` pattern match. Published cache entries
/// (`<cache_key>/`) are left untouched — they have no pid suffix
/// and don't match any debris shape.
///
/// # When to call this
///
/// **Library code (the steady state):** [`clean_all`] invokes this
/// before walking published entries. Library callers do NOT need to invoke this
/// directly to make a workload run — `ensure_template` does not
/// trip on stale debris because each new build picks a unique
/// `(cache_key, pid)` filename via [`staging_image_path`].
///
/// **Operator-driven (the rare case):** call this from a host
/// admin tool or a CI cleanup hook when:
/// - The host has hosted long-running ktstr peers that crashed
///   without graceful shutdown (SIGKILL, kernel oops, OOM kill,
///   panic) and the cache root is accumulating
///   `template.img.in-flight.*` / `*.tmp.*` / `.per-test-*`
///   entries.
/// - Disk pressure is rising and an inventory of the cache root
///   shows debris files significantly outweigh published entries.
/// - You're scripting a "clean cache" subcommand that does NOT
///   want to remove published entries (use [`clean_all`] for that).
///
/// **What this does NOT do:**
/// - Does not remove published cache entries — those have no pid
///   suffix and are filtered out by the prefix patterns. Use
///   [`clean_all`] when you want a full cache wipe.
/// - Does not remove the `.locks/` subdirectory — lockfile inodes
///   may be held by live peers and dropping them would orphan
///   their fds.
/// - Does not coordinate with live peers via flock — the pid-
///   liveness probe (`kill(pid, None)` returning `ESRCH`) is the
///   only synchronization. A peer in the brief window between
///   pid allocation and store_atomic completion may have its
///   debris removed mid-transaction; the pid-liveness probe
///   protects against this by reporting "live" until the peer
///   actually exits.
///
/// Returns the count of removed debris entries (info-level
/// tracing also logs each removal).
///
/// Called once per process from `ensure_template` (via
/// `sweep_cache_debris_once`) to GC dead-pid debris on the production
/// path, plus the operator-facing `clean_all`.
pub fn clean_orphaned_tmp_dirs(cache_root: &Path) -> Result<usize> {
    if !cache_root.is_dir() {
        // Cache root not yet materialised — nothing to sweep.
        // Mirrors the early-return at the head of
        // [`crate::cache::housekeeping::clean_orphaned_tmp_dirs`].
        return Ok(0);
    }
    let read_dir = match std::fs::read_dir(cache_root) {
        Ok(rd) => rd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(e) => {
            return Err(anyhow!("read cache root {cache_root:?}: {e}"));
        }
    };
    let mut removed: usize = 0;
    for dir_entry in read_dir {
        let dir_entry = match dir_entry {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    "skip unreadable disk-template cache root entry",
                );
                continue;
            }
        };
        let name = match dir_entry.file_name().into_string() {
            Ok(n) => n,
            // Non-UTF-8 filename — neither of our patterns can
            // match (both contain only ASCII), so it's foreign
            // debris (not ours to touch).
            Err(_) => continue,
        };
        // Identify which debris shape this entry is, and extract
        // the pid suffix. Patterns:
        //
        //   - `template.img.in-flight.<cache_key>.<pid>` — staging
        //     image (see [`staging_image_path`]).
        //   - `<cache_key>.tmp.<pid>` — staging directory (see
        //     [`store_atomic`]).
        //   - `.per-test-<pid>-<ns>-<rnd>.img` — per-test FICLONE
        //     backing file (see
        //     `KtstrVm::init_virtio_blk`'s `Btrfs`
        //     branch). Pid is the FIRST `-`-separated token after
        //     the `.per-test-` prefix; subsequent tokens encode
        //     timestamp + randomness for collision-freedom across
        //     concurrent tests in the same process.
        //
        // Anything else (notably the `.locks/` subdirectory and
        // the published `<cache_key>/` entries) is skipped.
        let pid_str = if let Some(rest) = name.strip_prefix("template.img.in-flight.") {
            // The trailing `.<pid>` is what we need; key may
            // itself contain `-` / `.` so we split at the LAST
            // `.` token.
            match rest.rsplit_once('.') {
                Some((_, suffix)) if !suffix.is_empty() => suffix,
                _ => continue,
            }
        } else if let Some(rest) = name.strip_prefix(".per-test-") {
            // `.per-test-<pid>-<ns>-<rnd>.img` — pid is the FIRST
            // `-`-separated token after the prefix. `split_once`
            // (not `rsplit_once`) because the random/timestamp
            // tokens follow the pid, not precede it.
            match rest.split_once('-') {
                Some((pid_token, _)) if !pid_token.is_empty() => pid_token,
                _ => continue,
            }
        } else if name.contains(".tmp.") {
            // `<cache_key>.tmp.<pid>` — the pid is everything
            // after the LAST `.tmp.`.
            match name.rsplit_once(".tmp.") {
                Some((_, suffix)) if !suffix.is_empty() => suffix,
                _ => continue,
            }
        } else {
            continue;
        };
        let pid: i32 = match pid_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        // Reject non-positive pids defensively — `kill(0, ...)`
        // probes the caller's own process group, `kill(-N, ...)`
        // probes process group N. Same hardening as
        // [`crate::cache::housekeeping::clean_orphaned_tmp_dirs`].
        if pid <= 0 {
            continue;
        }
        let dead = matches!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
            Err(nix::errno::Errno::ESRCH),
        );
        if !dead {
            continue;
        }
        let path = dir_entry.path();
        // The debris shapes need two different removers (file vs
        // directory). Probe via `file_type()` rather than
        // re-checking the prefix — the prefix already classified
        // the path; `file_type()` picks the right `remove_*` arm.
        let result = match dir_entry.file_type() {
            Ok(ft) if ft.is_dir() => std::fs::remove_dir_all(&path),
            Ok(_) => std::fs::remove_file(&path),
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    path = %path.display(),
                    "skip disk-template cache entry; \
                     file_type() failed",
                );
                continue;
            }
        };
        match result {
            Ok(()) => {
                tracing::info!(
                    path = %path.display(),
                    orphan_pid = pid,
                    "cleaned orphaned disk-template debris from \
                     prior crashed process",
                );
                removed += 1;
            }
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    path = %path.display(),
                    "failed to remove orphaned disk-template debris; \
                     leaving in place",
                );
            }
        }
    }
    Ok(removed)
}

/// Remove every published disk-template cache entry, returning the
/// count of entries actually removed.
///
/// Mirrors [`crate::cache::CacheDir::clean_all`] in `src/cache.rs`.
///
/// # When to call this
///
/// **Operator-driven only.** No production code path calls
/// `clean_all` automatically — the framework's runtime path is
/// `ensure_template` → cache hit / build, never a full sweep.
/// Operators reach for `clean_all` in three scenarios:
///
/// 1. **Disk-pressure escape hatch.** A long-running host has
///    accumulated dozens of cache entries across distinct
///    `(fs, capacity, mkfs version)` triples (each capacity-mib
///    setting and each mkfs upgrade rotates the key). When disk
///    pressure rises, `clean_all` is the nuclear option — wipe
///    every published template and let the next test run rebuild
///    only what it needs.
///
/// 2. **Defense against a fingerprint-blind upgrade.** The cache
///    key includes a fingerprint derived from
///    [`mkfs_version_fingerprint`] (the SHA-256 prefix of
///    `mkfs.<fstype> --version` output), so an mkfs upgrade that
///    changes the version banner rotates the key automatically
///    and the cache self-invalidates. `clean_all` remains the
///    fallback when the version banner does NOT change across an
///    upgrade (a downstream patch that bumps the on-disk format
///    without bumping `--version`) — a rare distro-specific case
///    that operators discover via "the new kernel rejects the
///    cached template" failures.
///
/// 3. **Cleanup before benchmarking.** Empty cache state lets a
///    benchmark measure the full `(template build + clone)` cost
///    deterministically. `clean_all` followed by `ensure_template`
///    is the canonical "cold cache" sequence.
///
/// **What this does NOT do:**
/// - Does not remove the `.locks/` subdirectory — lockfile inodes
///   may be held by live peers and dropping them would orphan
///   their fds (see "What gets skipped" below).
/// - Does not block on live peers — entries whose flock is held
///   by a live peer are skipped (logged at `info`); only quiescent
///   entries are removed.
/// - Does not fall back to a per-key wipe loop on a busy cache —
///   if every entry is locked the function returns 0, not an
///   error. Operators who need to force-remove a locked entry
///   should kill the holder and re-run.
///
/// # Companion: stale-debris sweep
///
/// `clean_all` calls [`clean_orphaned_tmp_dirs`] up front so a
/// rebuilding peer that hits the freshly-empty cache doesn't trip
/// on stale staging debris from a crashed predecessor during its
/// first `store_atomic`. Operators who want ONLY the debris sweep
/// (without removing published entries) should call
/// [`clean_orphaned_tmp_dirs`] directly.
///
/// # Concurrency
///
/// Each entry's per-key lockfile is acquired non-blocking in
/// `LOCK_EX` mode via [`crate::flock::try_flock`]. An entry whose
/// lock is held by a live peer (a concurrent template build that
/// finished its rename but is still inside the lock holder's
/// critical section) is skipped rather than removed — the skip
/// prevents racing a builder's publish. Per-test clones
/// ([`clone_to_per_test`]) take no lock, so a held lock never
/// indicates a live clone; those clones are protected instead by
/// Unix open-fd inode semantics (an already-open fd survives the
/// remove-tree).
///
/// The flock is held across the `remove_dir_all` so a peer that
/// blocks on the lock while we're removing observes a clean
/// "entry gone, rebuild from scratch" sequence: their post-lock
/// `lookup()` returns `None` and `ensure_template` proceeds to
/// rebuild. Without holding the lock during removal, a peer that
/// raced through `acquire_template_lock` → `lookup` between our
/// lock-release and our `remove_dir_all` would see the template
/// path, `clone_to_per_test` would race against the rmtree, and
/// either side could win unpredictably.
///
/// The lockfile inode itself is NOT removed — other peers may
/// have it open, and dropping the file while peers wait on it
/// would orphan their fds. Lockfile inodes are sized at a few
/// bytes each and accumulate at the rate of distinct
/// `(fs, capacity, mkfs version)` keys; leaving them is bounded
/// growth, not a leak.
///
/// # Sweeps debris first
///
/// Calls [`clean_orphaned_tmp_dirs`] before walking published
/// entries so a rebuilding peer that hits the freshly-empty cache
/// doesn't trip on stale staging debris from a crashed predecessor
/// during its first `store_atomic`.
///
/// # What gets skipped
///
/// - The `.locks/` subdirectory (lockfile namespace).
/// - Any cache entry whose lockfile is currently held by a live
///   peer (logged at `info` so the operator sees what was kept).
/// - Non-UTF-8 entry names (foreign — not produced by ktstr).
/// - Files at the cache root (only directories are cache entries;
///   `clean_orphaned_tmp_dirs` already swept the staging-image
///   files before we got here).
///
/// `dead_code` allow: kept as the operator-facing entry point
/// for a future `cargo ktstr clean --all` subcommand; not yet
/// wired into any command surface.
#[allow(dead_code)]
pub fn clean_all() -> Result<usize> {
    let root = cache_root()?;
    if !root.is_dir() {
        return Ok(0);
    }
    // Sweep staging debris first so a peer that re-acquires a
    // freshly-emptied cache doesn't trip on leftover .tmp.<pid>
    // / template.img.in-flight.* files from a crashed predecessor.
    // The result is logged but not fed into the return count —
    // `clean_all` reports published-entry removals only, matching
    // the [`crate::cache::CacheDir::clean_all`] contract.
    let _debris = clean_orphaned_tmp_dirs(&root)?;
    // Ensure the lockfile parent directory exists. `try_flock` opens
    // the lockfile with `O_CREAT`, but the open fails with `ENOENT`
    // when the parent `.locks/` subdirectory is absent — which is
    // the steady state on a freshly-published cache that was never
    // touched by `acquire_template_lock` (e.g. a cache populated by
    // an earlier ktstr run that crashed between `store_atomic` and
    // first lock acquire). Without this, `clean_all` would silently
    // skip every published entry on such a cache: the `try_flock`
    // call returns `Err(ENOENT)`, the loop's tracing-warn branch
    // logs and `continue`s, and the operator-driven `clean_all`
    // becomes a no-op. `create_dir_all` is idempotent — it's a
    // no-op when `.locks/` already exists, so this also covers the
    // mixed-cache case (some keys lock-touched, others not).
    let lock_dir = root.join(LOCK_DIR_NAME);
    std::fs::create_dir_all(&lock_dir).with_context(|| {
        format!(
            "create disk-template lock subdirectory {} for clean_all",
            lock_dir.display(),
        )
    })?;
    let read_dir = match std::fs::read_dir(&root) {
        Ok(rd) => rd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(e) => {
            return Err(anyhow!("read cache root {root:?}: {e}"));
        }
    };
    let mut removed: usize = 0;
    for dir_entry in read_dir {
        let dir_entry = match dir_entry {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    "skip unreadable disk-template cache root entry \
                     during clean_all",
                );
                continue;
            }
        };
        let file_type = match dir_entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    path = %dir_entry.path().display(),
                    "skip disk-template entry; file_type() failed",
                );
                continue;
            }
        };
        // Only published cache entries are directories at the
        // cache root. Files are staging images already swept by
        // clean_orphaned_tmp_dirs above; skip them here so we
        // don't double-account.
        if !file_type.is_dir() {
            continue;
        }
        let name = match dir_entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        // Skip the lockfile subdirectory — it's not a cache
        // entry. Its pathname is fixed by [`LOCK_DIR_NAME`].
        if name == LOCK_DIR_NAME {
            continue;
        }
        // Skip staging directories left by store_atomic (handled
        // by clean_orphaned_tmp_dirs above; defense-in-depth in
        // case the sweep returned early on a syscall error).
        if name.contains(".tmp.") {
            continue;
        }
        let entry_path = dir_entry.path();
        // Probe via try_flock that no live peer is currently
        // using this entry. The lockfile is acquired non-blocking
        // in LOCK_EX mode: success means there are zero readers
        // AND zero writers on this key. Failure (Ok(None)) means
        // a peer holds the lock — skip this entry.
        let lock_path = match lock_path_for_key(&name) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    cache_key = %name,
                    "skip disk-template entry; lock_path resolution \
                     failed",
                );
                continue;
            }
        };
        let lock_fd = match try_flock(&lock_path, FlockMode::Exclusive) {
            Ok(Some(fd)) => fd,
            Ok(None) => {
                // A live peer holds the lock. Skip — its work
                // would race a `remove_dir_all` against a
                // concurrent builder's publish.
                tracing::info!(
                    cache_key = %name,
                    lockfile = %lock_path.display(),
                    "skip disk-template entry during clean_all — \
                     locked by live peer",
                );
                continue;
            }
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    cache_key = %name,
                    "skip disk-template entry; try_flock failed",
                );
                continue;
            }
        };
        // Lock acquired. Perform the removal while holding the
        // lock so any peer that subsequently blocks on this
        // lockfile observes "no entry, rebuild from scratch" via
        // their re-check after acquire (see [`ensure_template`]).
        match std::fs::remove_dir_all(&entry_path) {
            Ok(()) => {
                tracing::info!(
                    cache_key = %name,
                    path = %entry_path.display(),
                    "removed disk-template cache entry during clean_all",
                );
                removed += 1;
            }
            Err(e) => {
                tracing::warn!(
                    err = %format!("{e:#}"),
                    cache_key = %name,
                    path = %entry_path.display(),
                    "failed to remove disk-template cache entry \
                     during clean_all; leaving in place",
                );
            }
        }
        // OwnedFd `lock_fd` drops here, releasing the per-key
        // flock. The lockfile inode at `lock_path` stays — see
        // the doc comment "the lockfile inode itself is NOT
        // removed".
        drop(lock_fd);
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_orphaned_tmp_dirs_removes_dead_pid_debris_keeps_live_and_published() {
        // pid 2147483647 (i32::MAX) is above pid_max, so kill() -> ESRCH
        // (dead); our own pid is live. Verify all three debris shapes are
        // reclaimed only for the dead owner, and published entries / .locks
        // / live-pid in-flight files are never touched.
        let base = std::env::temp_dir().join(format!("ktstr-dtsweep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("mk cache root");
        let live = std::process::id();

        let dead_staging = base.join("template.img.in-flight.somekey.2147483647");
        let dead_pertest = base.join(".per-test-2147483647-aa-bb.img");
        let dead_stagedir = base.join("somekey.tmp.2147483647");
        std::fs::write(&dead_staging, b"x").unwrap();
        std::fs::write(&dead_pertest, b"x").unwrap();
        std::fs::create_dir(&dead_stagedir).unwrap();

        let live_staging = base.join(format!("template.img.in-flight.somekey.{live}"));
        let live_pertest = base.join(format!(".per-test-{live}-cc-dd.img"));
        std::fs::write(&live_staging, b"x").unwrap();
        std::fs::write(&live_pertest, b"x").unwrap();

        let published = base.join("somekey");
        let locks = base.join(".locks");
        std::fs::create_dir(&published).unwrap();
        std::fs::create_dir(&locks).unwrap();

        let removed = clean_orphaned_tmp_dirs(&base).expect("sweep");

        assert!(!dead_staging.exists(), "dead-pid staging image reclaimed");
        assert!(
            !dead_pertest.exists(),
            "dead-pid per-test backing reclaimed"
        );
        assert!(!dead_stagedir.exists(), "dead-pid staging dir reclaimed");
        assert!(live_staging.exists(), "live-pid staging image kept");
        assert!(live_pertest.exists(), "live-pid per-test backing kept");
        assert!(published.exists(), "published cache entry untouched");
        assert!(locks.exists(), ".locks dir untouched");
        assert_eq!(removed, 3, "exactly the three dead-pid entries removed");

        let _ = std::fs::remove_dir_all(&base);
    }
}
