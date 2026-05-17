//! Locate the `vmlinux` ELF that pairs with a guest kernel image.
//!
//! Used by the host monitor and BPF reader to resolve symbols and
//! BTF offsets against the running guest kernel.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

/// Process-global cache of vmlinux ELF bytes keyed by canonical path.
///
/// `collect_verifier_stats` is called once per failure-dump cycle; in a
/// nextest test process running many `#[ktstr_test]` cases that boot
/// fresh VMs against the same kernel, repeating the file read for every
/// VM costs 50-340 MB of disk I/O on the freeze-coord cleanup critical
/// path. Caching the bytes once per canonical path collapses every
/// subsequent VM's read to a hash lookup + `Arc::clone`.
///
/// The cached entry pairs the bytes with the file's mtime at read
/// time. On every lookup we re-stat the file: if the stat'd mtime
/// matches the cached mtime, the cached bytes are reused; otherwise
/// the entry is replaced with a fresh read. This catches the case
/// where a developer rebuilds vmlinux mid-process — the user gets
/// the new bytes instead of stale cached bytes that would mismatch
/// against the running guest kernel. Stat-cost is microseconds vs
/// ~100ms for the cached read it gates, so the invalidation check
/// is effectively free on the hot path.
///
/// The cache key is the canonicalized path so symlinks across cache
/// / source-tree layouts collapse to one entry. A `canonicalize` or
/// `metadata` failure (EACCES, missing target) skips the cache and
/// falls through to the direct read. The error case is not cached:
/// a transient EACCES (e.g. a half-written cache entry whose
/// permissions arrive on the next ms) should not poison the cache
/// for the rest of the process.
static VMLINUX_BYTES_CACHE: OnceLock<RwLock<std::collections::HashMap<PathBuf, CachedEntry>>> =
    OnceLock::new();

/// One slot in [`VMLINUX_BYTES_CACHE`]. The mtime gates the bytes —
/// a mismatch on lookup invalidates and triggers a re-read.
struct CachedEntry {
    mtime: std::time::SystemTime,
    bytes: Arc<Vec<u8>>,
}

/// Return the cached vmlinux ELF bytes for `path`, populating the cache
/// on first read and invalidating on file modification.
///
/// Returns `None` when `path` is unreadable (stat or read failure).
/// The error case is not cached: a transient EACCES (e.g. a
/// half-written cache entry whose permissions arrive on the next
/// ms) should not poison the cache for the rest of the process.
pub(crate) fn cached_vmlinux_bytes(path: &Path) -> Option<Arc<Vec<u8>>> {
    let canon = std::fs::canonicalize(path)
        .ok()
        .unwrap_or_else(|| path.to_path_buf());
    // mtime captured before the read so a concurrent write that
    // finishes mid-read produces a mtime-bumped entry on the NEXT
    // lookup (this insert may carry the M1 mtime with mid-stream
    // bytes; the next lookup sees M2 ≠ M1 and re-reads cleanly).
    let mtime = std::fs::metadata(&canon).and_then(|m| m.modified()).ok()?;
    let slot = VMLINUX_BYTES_CACHE.get_or_init(|| RwLock::new(std::collections::HashMap::new()));
    {
        let read = slot.read().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = read.get(&canon)
            && entry.mtime == mtime
        {
            return Some(Arc::clone(&entry.bytes));
        }
    }
    // Read outside the write lock so a slow read doesn't block other
    // canonical paths' lookups. A racing second reader will pay the
    // same read once each — acceptable: if mtime matches both reads
    // produce the same bytes.
    let bytes = std::fs::read(&canon).ok()?;
    let arc = Arc::new(bytes);
    let mut write = slot.write().unwrap_or_else(|e| e.into_inner());
    // Always overwrite: if no entry, insert; if entry exists with
    // matching mtime (racing reader won the insert race), our overwrite
    // is identical; if entry exists with stale mtime (file rewrote
    // between our read-lock release and our write-lock acquire), the
    // stale entry is replaced.
    write.insert(
        canon,
        CachedEntry {
            mtime,
            bytes: Arc::clone(&arc),
        },
    );
    Some(arc)
}

/// Clear every cached entry. Used by `#[cfg(test)]` tests that need
/// to assert against a clean cache state without inheriting entries
/// from prior tests in the same process — a regular use case for
/// invalidation-coverage tests where we want to compare cache-miss
/// vs cache-hit behaviour deterministically.
#[cfg(test)]
pub(crate) fn clear_vmlinux_cache_for_tests() {
    if let Some(slot) = VMLINUX_BYTES_CACHE.get() {
        slot.write().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

/// Find the vmlinux ELF next to a kernel image path.
///
/// Shared across x86_64 and aarch64. Both architectures follow the
/// kernel build's `<root>/arch/<arch>/boot/<image>` layout, so
/// stepping 3 directories up from `kernel_path` lands on `<root>`
/// where `vmlinux` sits. Distro paths diverge: x86_64 ships debug
/// vmlinux at `/usr/lib/debug/boot/vmlinux-<version>`, aarch64 splits
/// between `/boot/vmlinux-<version>` and
/// `/lib/modules/<version>/build/vmlinux`. Both distro layouts are
/// probed regardless of arch — the arch-specific filename prefix
/// (`bzImage` vs `Image`) only tells us where to look, not which
/// layout owns the match.
pub(crate) fn find_vmlinux(kernel_path: &Path) -> Option<PathBuf> {
    let dir = kernel_path.parent()?;
    let candidate = dir.join("vmlinux");
    if candidate.exists() {
        return Some(candidate);
    }
    // Kernel build tree: <root>/arch/<arch>/boot/<image> -> <root>/vmlinux.
    if let Ok(root) = dir.join("../../..").canonicalize() {
        let candidate = root.join("vmlinux");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    // Distro layouts keyed by the image's version suffix
    // (`vmlinuz-<version>`).
    if let Some(name) = kernel_path.file_name().and_then(|n| n.to_str()) {
        let version = name.strip_prefix("vmlinuz-").unwrap_or(name);
        for candidate in [
            PathBuf::from(format!("/usr/lib/debug/boot/vmlinux-{version}")),
            PathBuf::from(format!("/boot/vmlinux-{version}")),
            PathBuf::from(format!("/lib/modules/{version}/build/vmlinux")),
        ] {
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    // `/lib/modules/<version>/vmlinuz` layout: version is the parent
    // directory name, and the sibling `build/vmlinux` is the target.
    if let Some(parent_name) = dir.file_name().and_then(|n| n.to_str()) {
        for candidate in [
            dir.join("build/vmlinux"),
            PathBuf::from(format!("/boot/vmlinux-{parent_name}")),
        ] {
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn find_vmlinux_from_bzimage_path() {
        // Create a temp dir simulating <root>/arch/x86/boot/bzImage with vmlinux at <root>.
        let tmp = tempfile::TempDir::new().unwrap();
        let boot_dir = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot_dir).unwrap();
        let vmlinux = tmp.path().join("vmlinux");
        std::fs::write(&vmlinux, b"ELF").unwrap();
        let bzimage = boot_dir.join("bzImage");
        std::fs::write(&bzimage, b"kernel").unwrap();

        let found = find_vmlinux(&bzimage);
        assert_eq!(found, Some(vmlinux));
    }

    #[test]
    fn find_vmlinux_sibling() {
        // vmlinux in the same directory as the kernel image.
        let tmp = tempfile::TempDir::new().unwrap();
        let vmlinux = tmp.path().join("vmlinux");
        std::fs::write(&vmlinux, b"ELF").unwrap();
        let kernel = tmp.path().join("bzImage");
        std::fs::write(&kernel, b"kernel").unwrap();

        let found = find_vmlinux(&kernel);
        assert_eq!(found, Some(vmlinux));
    }

    #[test]
    fn find_vmlinux_bare_filename() {
        // A bare filename — parent is "" so no vmlinux sibling found.
        assert_eq!(find_vmlinux(Path::new("vmlinuz")), None);
    }

    #[test]
    fn find_vmlinux_root_parent() {
        // /vmlinuz has parent "/" — no vmlinux there (or if there is, fine).
        // The function should not panic.
        let result = find_vmlinux(Path::new("/vmlinuz"));
        // /vmlinux almost certainly doesn't exist; if it does, that's still valid.
        if !Path::new("/vmlinux").exists() {
            assert_eq!(result, None);
        }
    }

    #[test]
    fn find_vmlinux_missing_returns_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let kernel = tmp.path().join("bzImage");
        std::fs::write(&kernel, b"kernel").unwrap();

        assert_eq!(find_vmlinux(&kernel), None);
    }

    /// First call reads from disk; second call returns a clone of the
    /// cached `Arc<Vec<u8>>`, proving the cache hit path does not re-
    /// read. `Arc::ptr_eq` is the load-bearing assertion: the bytes
    /// would compare equal even from a re-read, but only the cache
    /// hit returns the same allocation.
    #[test]
    fn cached_vmlinux_bytes_hits_on_second_call() {
        let tmp = tempfile::TempDir::new().unwrap();
        let vmlinux = tmp.path().join("vmlinux-test-cache");
        std::fs::write(&vmlinux, b"FAKE_VMLINUX_BYTES").unwrap();

        let first = cached_vmlinux_bytes(&vmlinux).expect("first read populates cache");
        let second = cached_vmlinux_bytes(&vmlinux).expect("second read hits cache");
        assert_eq!(first.as_slice(), b"FAKE_VMLINUX_BYTES");
        assert!(
            Arc::ptr_eq(&first, &second),
            "cache hit must return the same Arc; got fresh allocations on each call"
        );
    }

    /// Unreadable path returns `None` without populating the cache;
    /// a subsequent successful path is unaffected.
    #[test]
    fn cached_vmlinux_bytes_missing_returns_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let nonexistent = tmp.path().join("missing-xyzzy");
        assert!(cached_vmlinux_bytes(&nonexistent).is_none());
    }

    /// Two distinct symlink paths pointing at the SAME real file must
    /// dedup to one cache entry — canonicalize collapses both keys to
    /// the same canonical PathBuf, so the second lookup hits the cache
    /// populated by the first and returns a clone of the same `Arc`.
    /// Verified via `Arc::ptr_eq` rather than byte equality: the bytes
    /// would compare equal even from a re-read; only a true cache hit
    /// returns the same allocation.
    #[test]
    #[cfg(unix)]
    fn cached_vmlinux_bytes_dedups_symlinks_to_same_target() {
        let tmp = tempfile::TempDir::new().unwrap();
        let real = tmp.path().join("vmlinux-real");
        std::fs::write(&real, b"SYMLINK_DEDUP_BYTES").unwrap();
        let link_a = tmp.path().join("vmlinux-link-a");
        let link_b = tmp.path().join("vmlinux-link-b");
        std::os::unix::fs::symlink(&real, &link_a).unwrap();
        std::os::unix::fs::symlink(&real, &link_b).unwrap();

        let via_a = cached_vmlinux_bytes(&link_a).expect("read via symlink A");
        let via_b = cached_vmlinux_bytes(&link_b).expect("read via symlink B");
        assert!(
            Arc::ptr_eq(&via_a, &via_b),
            "two symlinks to the same target must canonicalize to the \
             same cache key and return the same Arc; got fresh \
             allocations, suggesting the canonicalize-then-key path \
             regressed to keying on the raw symlink path."
        );
    }

    /// A dangling symlink (target deleted before any read) makes
    /// `canonicalize` fail. The function falls back to using the
    /// symlink's own path as the cache key, then `fs::read` fails to
    /// open the dangling target and returns `None`. The cache is not
    /// populated for the dangling path.
    #[test]
    #[cfg(unix)]
    fn cached_vmlinux_bytes_dangling_symlink_returns_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("vmlinux-gone");
        let link = tmp.path().join("vmlinux-dangling");
        std::fs::write(&target, b"ELF").unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        std::fs::remove_file(&target).unwrap();

        assert!(cached_vmlinux_bytes(&link).is_none());
    }

    /// Rewriting the file with new bytes between two lookups must
    /// invalidate the cache and surface the new bytes on the
    /// second lookup. Catches the "stale cached bytes after a
    /// rebuild" regression that the pre-mtime version had: a
    /// developer who rebuilds vmlinux while a long-lived test
    /// process is running would get the stale bytes forever
    /// without this invalidation. Verifies via NON-`Arc::ptr_eq`
    /// (the new bytes must be in a fresh allocation) plus a byte-
    /// content comparison (the new content actually reached the
    /// reader). Bumps mtime explicitly via `libc::utimes` (rather
    /// than sleeping for FS-granularity) so the test runs in
    /// microseconds and survives FS variants with 1-second mtime
    /// resolution.
    #[test]
    #[cfg(unix)]
    fn cached_vmlinux_bytes_invalidates_on_mtime_change() {
        let tmp = tempfile::TempDir::new().unwrap();
        let vmlinux = tmp.path().join("vmlinux-mtime-test");
        std::fs::write(&vmlinux, b"FIRST_BYTES").unwrap();
        clear_vmlinux_cache_for_tests();

        let first = cached_vmlinux_bytes(&vmlinux).expect("first read");
        assert_eq!(first.as_slice(), b"FIRST_BYTES");

        // Rewrite with new content, then bump mtime to a sentinel
        // value far in the past via libc::utimes so the captured
        // mtime is guaranteed != the cached one regardless of FS
        // mtime resolution. Setting both atime and mtime to
        // 1970-01-02T00:00:00 (86400 sec since epoch) makes the
        // pre-write mtime (now-ish) vs post-utimes mtime
        // (1970-01-02) trivially distinct.
        std::fs::write(&vmlinux, b"SECOND_BYTES_DIFFERENT").unwrap();
        let path_c = std::ffi::CString::new(vmlinux.as_os_str().as_encoded_bytes()).unwrap();
        let sentinel = libc::timeval {
            tv_sec: 86_400,
            tv_usec: 0,
        };
        let times = [sentinel, sentinel];
        // SAFETY: path_c is a valid NUL-terminated path; times is a
        // 2-element timeval array (atime, mtime) as utimes(2) requires.
        let rc = unsafe { libc::utimes(path_c.as_ptr(), times.as_ptr()) };
        assert_eq!(rc, 0, "libc::utimes must succeed on the temp file");

        let second = cached_vmlinux_bytes(&vmlinux).expect("second read");
        assert_eq!(
            second.as_slice(),
            b"SECOND_BYTES_DIFFERENT",
            "mtime change must invalidate cache and surface the rewritten bytes"
        );
        assert!(
            !Arc::ptr_eq(&first, &second),
            "post-rewrite second lookup must return a fresh Arc, \
             not the stale cached one — Arc::ptr_eq returning true \
             means the invalidation path didn't fire."
        );
    }
}
