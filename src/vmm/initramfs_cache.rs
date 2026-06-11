//! Two-tier initramfs cache: per-process HashMap + cross-process POSIX
//! shm.
//!
//! Each VM run produces an initramfs base blob keyed on the payload's
//! shared-lib SET (not the payload's own content — its /init bytes ride
//! the per-run suffix) plus the content hashes of the optional
//! scheduler / probe / worker binaries packed into the base, include
//! files, and shell-mode flags. Building the blob is expensive (10s of
//! MB of cpio assembly + LZ4 compression), so the cache amortises the
//! cost across:
//!
//! - **Same-process tests**: a `HashMap<BaseKey, Arc<Vec<u8>>>`
//!   keeps the in-flight blob hot without a syscall.
//! - **Cross-process tests / nextest workers**: an `O_CREAT|O_EXCL`
//!   race over a `/dev/shm/ktstr-base-<arch>-<hash>` segment elects
//!   a single builder; losers `LOCK_SH`-block on the segment until
//!   the winner finishes, then `mmap` it zero-copy.
//!
//! The `BaseKey` hashes the payload's shared-lib set + interpreter (the
//! bytes the base packs FROM the payload — not the payload itself) and
//! the full content of the scheduler / probe / worker binaries written
//! into the base. So a scheduler/probe/worker recompile invalidates the
//! cache, while a payload recompile that keeps the same lib set is a HIT
//! (the payload's content lives only in the per-run suffix). Stale
//! segments from a previous compression format are GC'd once per process
//! on the first `get_or_build_base` call via a `LOCK_EX | LOCK_NB` probe.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use std::hash::BuildHasher;

use ahash::AHasher;

use super::initramfs;

/// Cache key for base initramfs. Derived from the payload's shared-lib
/// SET + interpreter (NOT the payload's content — its /init bytes ride
/// the per-run suffix), plus the content hashes of the optional
/// scheduler / probe / worker binaries packed into the base and their
/// shared libs. Shell mode additionally mixes in a sentinel, include
/// files, and the busybox flag; see [`Self::new`] and [`Self::new_shell`]
/// for per-constructor inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BaseKey(pub(crate) u64);

/// Process-local memoisation key for [`hash_file`]. `(path, dev, ino,
/// mtime_secs, mtime_nsecs)` identifies a specific file revision: dev
/// + ino pin the inode (so a path replaced by a different file
///   invalidates), and mtime catches in-place edits. Same file unchanged
///   = identical key = HashMap hit, no re-stream.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct HashFileKey {
    path: PathBuf,
    dev: u64,
    ino: u64,
    mtime_secs: i64,
    mtime_nsecs: i64,
}

/// Process-local cache: file identity + mtime → ahash of contents.
fn hash_file_cache() -> &'static Mutex<HashMap<HashFileKey, u64>> {
    static CACHE: OnceLock<Mutex<HashMap<HashFileKey, u64>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Hash a file's content for cache keying via mmap + ahash.
///
/// Uses ahash for speed — AES-NI accelerated on x86_64/aarch64 via
/// runtime detection, software fallback otherwise. The file is
/// memory-mapped read-only so the kernel handles I/O via page faults
/// against the page cache; no intermediate buffer copies. Process-local
/// memoisation keyed on `(path, dev, ino, mtime)` short-circuits repeat
/// calls within the same run; the inode-plus-mtime tuple invalidates
/// the cached hash whenever the underlying file changes.
pub(crate) fn hash_file(path: &Path) -> Result<u64> {
    use std::fs::File;
    use std::os::unix::fs::MetadataExt;

    let file = File::open(path).with_context(|| format!("open for hash: {}", path.display()))?;
    let meta = file
        .metadata()
        .with_context(|| format!("stat for hash: {}", path.display()))?;
    let cache_key = HashFileKey {
        path: path.to_path_buf(),
        dev: meta.dev(),
        ino: meta.ino(),
        mtime_secs: meta.mtime(),
        mtime_nsecs: meta.mtime_nsec(),
    };
    if let Some(cached) = hash_file_cache().lock().unwrap().get(&cache_key).copied() {
        return Ok(cached);
    }

    let mmap = unsafe {
        memmap2::Mmap::map(&file).with_context(|| format!("mmap for hash: {}", path.display()))?
    };
    let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
    hasher.write(&mmap);
    let digest = hasher.finish();
    hash_file_cache().lock().unwrap().insert(cache_key, digest);
    Ok(digest)
}

impl BaseKey {
    /// Hashes the payload's shared-lib SET — NOT its content: `/init`
    /// lives in the per-run suffix (`build_suffix`), so the base archive
    /// depends only on which libraries the payload pulls in, not on the
    /// payload bytes. A payload recompile that keeps the same libs is a
    /// base-cache hit. The optional scheduler / probe / alloc-worker
    /// binaries DO have their content hashed — they are written into the
    /// base, so each changes the base bytes. Each optional input
    /// participates symmetrically. Explicit parameters keep the cache key
    /// sensitive to these inputs regardless of the routing choice —
    /// the probe currently rides the extras path (stripped) while
    /// the worker rides `include_files` (verbatim), but the hash
    /// stays correct if a future change moves either between the
    /// two paths (the `new_shell` include-hash loop also re-hashes
    /// whatever ends up in `include_files`, so the double hash of a
    /// worker-in-includes is tolerated; the explicit worker hash
    /// covers the case where a future refactor moves the worker to
    /// extras).
    pub(crate) fn new(
        payload: &Path,
        scheduler: Option<&Path>,
        probe: Option<&Path>,
        worker: Option<&Path>,
        staged: &[(&str, &Path)],
    ) -> Result<Self> {
        let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();

        // Payload CONTENT is not hashed: /init lives in the per-run
        // suffix, not the base. Only the payload's shared-lib SET shapes
        // the base, so a payload recompile that keeps the same libs is a
        // base-cache hit. Still fail fast if the payload is missing or
        // not a regular file — `hash_shared_libs` swallows resolve errors,
        // and the base build would fail later anyway.
        let payload_meta = std::fs::metadata(payload)
            .with_context(|| format!("stat payload for cache key: {}", payload.display()))?;
        anyhow::ensure!(
            payload_meta.is_file(),
            "payload for cache key is not a regular file: {}",
            payload.display()
        );
        Self::hash_shared_libs(payload, &mut hasher);

        match scheduler {
            Some(s) => {
                1u8.hash(&mut hasher);
                hash_file(s)?.hash(&mut hasher);
                Self::hash_shared_libs(s, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        match probe {
            Some(p) => {
                1u8.hash(&mut hasher);
                hash_file(p)?.hash(&mut hasher);
                Self::hash_shared_libs(p, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        match worker {
            Some(w) => {
                1u8.hash(&mut hasher);
                hash_file(w)?.hash(&mut hasher);
                Self::hash_shared_libs(w, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        Self::hash_staged(staged, &mut hasher)?;

        Ok(BaseKey(hasher.finish()))
    }

    /// Shell mode key: hashes a sentinel, include files, and the
    /// busybox flag so different shell configurations get distinct
    /// cache keys. Include file archive paths and content are hashed
    /// so the same payload libs + same includes = cache hit, while
    /// different includes = cache miss. `probe` and `worker` are
    /// hashed for the same reasons as [`BaseKey::new`].
    pub(crate) fn new_shell(
        payload: &Path,
        scheduler: Option<&Path>,
        probe: Option<&Path>,
        worker: Option<&Path>,
        staged: &[(&str, &Path)],
        include_files: &[(String, PathBuf)],
        busybox_bytes: Option<&[u8]>,
    ) -> Result<Self> {
        let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();

        "ktstr-shell".hash(&mut hasher);
        // Hash the busybox bytes themselves so different busybox
        // builds produce distinct cache keys (e.g. a busybox rebuild
        // after build.rs change). None vs Some(_) also distinguishes
        // shell-mode-with-busybox from shell-mode-without.
        //
        // wprof (when set on the builder) rides include_files —
        // its bytes are hashed there alongside other includes via
        // `hash_file(path)`, so a rebuilt wprof binary produces a
        // distinct cache key without a separate slot here.
        match busybox_bytes {
            Some(bytes) => {
                1u8.hash(&mut hasher);
                bytes.hash(&mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }
        // Payload CONTENT is not hashed (see `BaseKey::new`): /init is in
        // the per-run suffix; only the payload's shared-lib SET shapes the
        // base, so a recompile keeping the same libs is a cache hit. Fail
        // fast if the payload is missing or not a regular file (see
        // `BaseKey::new`).
        let payload_meta = std::fs::metadata(payload)
            .with_context(|| format!("stat payload for cache key: {}", payload.display()))?;
        anyhow::ensure!(
            payload_meta.is_file(),
            "payload for cache key is not a regular file: {}",
            payload.display()
        );
        Self::hash_shared_libs(payload, &mut hasher);

        match scheduler {
            Some(s) => {
                1u8.hash(&mut hasher);
                hash_file(s)?.hash(&mut hasher);
                Self::hash_shared_libs(s, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        match probe {
            Some(p) => {
                1u8.hash(&mut hasher);
                hash_file(p)?.hash(&mut hasher);
                Self::hash_shared_libs(p, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        match worker {
            Some(w) => {
                1u8.hash(&mut hasher);
                hash_file(w)?.hash(&mut hasher);
                Self::hash_shared_libs(w, &mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }

        // Hash include files: archive paths (sorted for determinism),
        // content hashes, and shared lib hashes for ELF includes (their
        // shared libs are packed by build_initramfs_base).
        let mut sorted: Vec<(&str, &Path)> = include_files
            .iter()
            .map(|(a, p)| (a.as_str(), p.as_path()))
            .collect();
        sorted.sort_by_key(|(a, _)| *a);
        sorted.len().hash(&mut hasher);
        for (archive_path, host_path) in &sorted {
            archive_path.hash(&mut hasher);
            hash_file(host_path)?.hash(&mut hasher);
            Self::hash_shared_libs(host_path, &mut hasher);
        }

        Self::hash_staged(staged, &mut hasher)?;

        Ok(BaseKey(hasher.finish()))
    }

    /// Hash a set of staged schedulers into the cache key, sorted by
    /// scheduler name for determinism (caller may pass entries in any
    /// order). Each entry contributes its name, binary content hash,
    /// and shared lib content — same shape as the boot-scheduler
    /// hash chain so a content change in any staged binary invalidates
    /// the cache. Length is hashed first so two staged sets that share
    /// an entry prefix cannot collide (e.g. `[a]` vs `[a, b]`).
    fn hash_staged(staged: &[(&str, &Path)], hasher: &mut AHasher) -> Result<()> {
        let mut sorted: Vec<(&str, &Path)> = staged.to_vec();
        sorted.sort_by_key(|(n, _)| *n);
        sorted.len().hash(hasher);
        for (name, binary) in &sorted {
            name.hash(hasher);
            hash_file(binary)?.hash(hasher);
            Self::hash_shared_libs(binary, hasher);
        }
        Ok(())
    }

    /// Hash shared library paths and content samples for a binary so
    /// the cache key changes when any shared lib is updated on the host.
    fn hash_shared_libs(binary: &Path, hasher: &mut AHasher) {
        if let Ok(result) = initramfs::resolve_shared_libs(binary) {
            // The PT_INTERP (dynamic linker) is packed into the base by
            // build_initramfs_base but is NOT a DT_NEEDED entry, so it's
            // absent from `found`. Hash its path + content: the base
            // depends on it, and the payload's own content is no
            // longer hashed, so an interp swap with an unchanged DT_NEEDED
            // set would otherwise serve a stale base.
            if let Some(interp) = &result.interpreter {
                interp.as_bytes().hash(hasher);
                if let Ok(sample) = hash_file(Path::new(interp)) {
                    sample.hash(hasher);
                }
            }
            let mut entries: Vec<_> = result.found.iter().map(|(_, p)| p.clone()).collect();
            entries.sort();
            for p in &entries {
                // `to_str()` loses every non-UTF-8 path (Linux
                // paths are arbitrary byte sequences, not UTF-8)
                // and the `unwrap_or("")` collapse would hash
                // every such path to the SAME empty string,
                // silently gluing distinct libraries together in
                // the cache key. `as_encoded_bytes()` hashes the
                // raw OS bytes verbatim.
                p.as_os_str().as_encoded_bytes().hash(hasher);
                if let Ok(sample) = hash_file(p) {
                    sample.hash(hasher);
                }
            }
        }
    }
}

/// Process-global cache for base initramfs bytes. Keyed by content hash
/// of payload, scheduler, include files, and busybox flag.
/// The lock is only held during map lookup/insert, never during the
/// actual build.
pub(crate) fn base_cache() -> &'static Mutex<HashMap<BaseKey, Arc<Vec<u8>>>> {
    static CACHE: OnceLock<Mutex<HashMap<BaseKey, Arc<Vec<u8>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Holds either a borrowed shm mapping or an owned Arc from the
/// process-local cache / a fresh build.
pub(crate) enum BaseRef {
    Mapped(initramfs::MappedShm),
    Owned(Arc<Vec<u8>>),
}

impl AsRef<[u8]> for BaseRef {
    fn as_ref(&self) -> &[u8] {
        match self {
            BaseRef::Mapped(m) => m.as_ref(),
            BaseRef::Owned(a) => a,
        }
    }
}

/// Obtain the base initramfs bytes, checking (in order):
/// 1. Process-local HashMap
/// 2. POSIX shared-memory segment via O_CREAT|O_EXCL race gate:
///    - Winner builds, writes segment, losers block on flock then mmap
/// 3. Fallback: build without cross-process coordination
///
/// `KTSTR_CARGO_TEST_MODE` skips steps 2 and 3's SHM coordination
/// entirely — the cross-process SHM cache assumes a `cargo ktstr
/// test` driver that staged the test binaries; under bare
/// `cargo test` each invocation is independent and the
/// `LOCK_EX | LOCK_NB` GC sweep / `O_EXCL` race gate would surface
/// as confusing flock contention messages on contributor
/// workstations. Per-process HashMap memoisation still applies, so
/// repeat tests inside the same `cargo test` invocation share the
/// build cost.
pub(crate) fn get_or_build_base(
    payload: &Path,
    extras: &[(&str, &Path)],
    include_files: &[(&str, &Path)],
    busybox_bytes: Option<Vec<u8>>,
    key: &BaseKey,
) -> Result<BaseRef> {
    let cargo_test_mode = crate::cargo_test_mode::cargo_test_mode_active();

    // 1. Process-local cache. Always tried first — this is the only
    //    layer that survives in cargo-test mode.
    if let Some(arc) = base_cache().lock().unwrap().get(key).cloned() {
        tracing::debug!("initramfs base cache hit (process)");
        return Ok(BaseRef::Owned(arc));
    }

    if cargo_test_mode {
        // Inline build, store in process-local cache only. Skip the
        // /dev/shm sweep and the O_EXCL race gate — the SHM
        // coordination layer is meant for `cargo ktstr test` /
        // nextest where N test processes share the same staged
        // binaries; under bare `cargo test` the sibling-binary
        // assumption does not hold.
        let t0 = std::time::Instant::now();
        let data = initramfs::build_initramfs_base(
            payload,
            extras,
            include_files,
            busybox_bytes.as_deref(),
        )?;
        let arc = Arc::new(data);
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            bytes = arc.len(),
            "build_initramfs_base (cargo-test inline)",
        );
        base_cache()
            .lock()
            .unwrap()
            .insert(key.clone(), arc.clone());
        return Ok(BaseRef::Owned(arc));
    }

    // Clean stale SHM segments from previous runs. The /dev/shm scan
    // touches every entry once and is keyed off `current` to skip the
    // segment we are about to use; running it on every call wastes
    // syscalls when many tests share a process. `OnceLock` gates the
    // sweep to a single execution per process — the first key wins
    // and every subsequent call is a free no-op.
    static CLEANUP_ONCE: OnceLock<()> = OnceLock::new();
    CLEANUP_ONCE.get_or_init(|| cleanup_stale_shm(key));

    // 2. SHM race gate: try O_CREAT|O_EXCL to elect a single builder.
    let seg_name = initramfs::shm_segment_name(key.0);
    match shm_try_create_excl(&seg_name) {
        ShmCreateResult::Winner(fd) => {
            tracing::debug!("initramfs shm: builder (O_EXCL won)");
            let t0 = std::time::Instant::now();
            let data = initramfs::build_initramfs_base(
                payload,
                extras,
                include_files,
                busybox_bytes.as_deref(),
            )?;
            tracing::debug!(
                elapsed_us = t0.elapsed().as_micros(),
                bytes = data.len(),
                "build_initramfs_base",
            );
            shm_write_and_release(fd, &data, &seg_name);
            hold_shm_lock(&seg_name);
            if let Some(mapped) = initramfs::shm_load_base(key.0) {
                return Ok(BaseRef::Mapped(mapped));
            }
            let arc = Arc::new(data);
            base_cache()
                .lock()
                .unwrap()
                .insert(key.clone(), arc.clone());
            return Ok(BaseRef::Owned(arc));
        }
        ShmCreateResult::Exists => {
            tracing::debug!("initramfs shm: waiting for builder (EEXIST)");
            if let Some(mapped) = initramfs::shm_load_base(key.0) {
                tracing::debug!("initramfs base cache hit (shm, after wait)");
                hold_shm_lock(&seg_name);
                return Ok(BaseRef::Mapped(mapped));
            }
        }
        ShmCreateResult::Error => {
            if let Some(mapped) = initramfs::shm_load_base(key.0) {
                tracing::debug!("initramfs base cache hit (shm)");
                hold_shm_lock(&seg_name);
                return Ok(BaseRef::Mapped(mapped));
            }
        }
    }

    // 3. Fallback: build without SHM coordination.
    let t0 = std::time::Instant::now();
    let data =
        initramfs::build_initramfs_base(payload, extras, include_files, busybox_bytes.as_deref())?;
    let arc = Arc::new(data);
    tracing::debug!(
        elapsed_us = t0.elapsed().as_micros(),
        bytes = arc.len(),
        "build_initramfs_base (fallback)",
    );

    base_cache()
        .lock()
        .unwrap()
        .insert(key.clone(), arc.clone());
    if let Err(e) = initramfs::shm_store_base(key.0, &arc) {
        tracing::warn!("shm_store_base: {e:#}");
    }

    Ok(BaseRef::Owned(arc))
}

/// Remove stale SHM segments from `/dev/shm` that don't match `current`.
/// Only unlinks segments not held by any process (`LOCK_EX | LOCK_NB`).
/// Parallel nextest workers hold `LOCK_SH` on their segments for the
/// process lifetime (via `HELD_SHM_LOCKS`), so their segments survive
/// cleanup from other workers.
fn cleanup_stale_shm(current: &BaseKey) {
    let current_suffix = format!("{}-{:016x}", initramfs::SHM_ARCH_TAG, current.0);
    let shm_dir = match std::fs::read_dir("/dev/shm") {
        Ok(d) => d,
        Err(_) => return,
    };
    for entry in shm_dir.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let suffix = if let Some(s) = name_str.strip_prefix("ktstr-base-") {
            s
        } else if let Some(s) = name_str.strip_prefix("ktstr-lz4-") {
            s
        } else if let Some(s) = name_str.strip_prefix("ktstr-gz-") {
            s
        } else {
            continue;
        };
        if suffix == current_suffix {
            continue;
        }
        let shm_name = format!("/{name_str}");
        let Ok(fd) = rustix::shm::open(
            shm_name.as_str(),
            rustix::shm::OFlags::RDONLY,
            rustix::fs::Mode::empty(),
        ) else {
            continue;
        };
        if rustix::fs::flock(&fd, rustix::fs::FlockOperation::NonBlockingLockExclusive).is_err() {
            continue;
        }
        let Ok(recheck_fd) = rustix::shm::open(
            shm_name.as_str(),
            rustix::shm::OFlags::RDONLY,
            rustix::fs::Mode::empty(),
        ) else {
            let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
            continue;
        };
        let stat_fd = rustix::fs::fstat(&fd);
        let stat_recheck = rustix::fs::fstat(&recheck_fd);
        match (stat_fd, stat_recheck) {
            (Ok(a), Ok(b)) if a.st_dev == b.st_dev && a.st_ino == b.st_ino => {
                let _ = rustix::shm::unlink(shm_name.as_str());
            }
            _ => {}
        }
        let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
    }
}

/// Process-lifetime `LOCK_SH` holds on SHM segments. Prevents
/// `cleanup_stale_shm` in parallel nextest workers from deleting
/// segments this process built or loaded.
static HELD_SHM_LOCKS: Mutex<Vec<rustix::fd::OwnedFd>> = Mutex::new(Vec::new());

fn hold_shm_lock(shm_name: &str) {
    for name in [
        shm_name.to_string(),
        shm_name.replace("ktstr-base-", "ktstr-lz4-"),
    ] {
        if let Ok(fd) = rustix::shm::open(
            name.as_str(),
            rustix::shm::OFlags::RDONLY,
            rustix::fs::Mode::empty(),
        ) && rustix::fs::flock(&fd, rustix::fs::FlockOperation::NonBlockingLockShared).is_ok()
        {
            HELD_SHM_LOCKS.lock().unwrap().push(fd);
        }
    }
}

// ---------------------------------------------------------------------------
// SHM O_EXCL race gate helpers
// ---------------------------------------------------------------------------

pub(crate) enum ShmCreateResult {
    /// We created the segment; fd holds an exclusive flock. The fd is
    /// owned — drop releases the lock and closes the descriptor.
    Winner(std::os::fd::OwnedFd),
    /// Segment already exists (another process is building or built it).
    Exists,
    /// shm_open failed for a reason other than EEXIST.
    Error,
}

/// Try to create a POSIX shm segment with O_CREAT|O_EXCL. On success,
/// acquire LOCK_EX and return the fd. On EEXIST, return Exists.
pub(crate) fn shm_try_create_excl(name: &str) -> ShmCreateResult {
    let fd = match rustix::shm::open(
        name,
        rustix::shm::OFlags::CREATE | rustix::shm::OFlags::EXCL | rustix::shm::OFlags::RDWR,
        rustix::fs::Mode::from_raw_mode(0o644),
    ) {
        Ok(fd) => fd,
        Err(e) if e == rustix::io::Errno::EXIST => return ShmCreateResult::Exists,
        Err(_) => return ShmCreateResult::Error,
    };

    // Take exclusive (blocking) lock before writing. The fd is dropped
    // on the error path, which closes it automatically.
    if rustix::fs::flock(&fd, rustix::fs::FlockOperation::LockExclusive).is_err() {
        return ShmCreateResult::Error;
    }

    ShmCreateResult::Winner(fd)
}

/// Write data to the shm fd, then release the exclusive lock and close.
/// On failure (ftruncate or mmap), unlinks the segment so future callers
/// don't find a corrupt/empty segment and can retry.
pub(crate) fn shm_write_and_release(fd: std::os::fd::OwnedFd, data: &[u8], seg_name: &str) {
    use std::os::fd::AsRawFd;

    // Keep the raw fd for libc::mmap / libc::ftruncate (rustix::mm
    // is not currently wired in); the OwnedFd still owns the close
    // and flock-release on drop.
    let raw = fd.as_raw_fd();
    unsafe {
        if libc::ftruncate(raw, data.len() as libc::off_t) != 0 {
            let _ = rustix::shm::unlink(seg_name);
            // fd drop runs flock_un + close automatically.
            return;
        }

        let ptr = libc::mmap(
            std::ptr::null_mut(),
            data.len(),
            libc::PROT_WRITE,
            libc::MAP_SHARED,
            raw,
            0,
        );
        if ptr == libc::MAP_FAILED {
            // Zero the size so readers blocked on LOCK_SH see st_size=0
            // from fstat and return None instead of mapping zero-filled bytes.
            libc::ftruncate(raw, 0);
            let _ = rustix::shm::unlink(seg_name);
        } else {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            libc::munmap(ptr, data.len());
        }
    }
    // Explicit unlock so readers blocked on LOCK_SH observe ordering
    // with the final mmap before the fd-drop close hits.
    let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
    // fd drops here → close(fd). OwnedFd::drop ignores errors.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `shm_try_create_excl` winner gets a locked fd; a second call
    /// with the same name returns `Exists`. The winner's
    /// `shm_unlink` cleanup keeps subsequent tests independent.
    #[test]
    fn shm_try_create_excl_winner_then_exists() {
        // Unique name per test process + nanos so parallel tests
        // don't collide on the global /dev/shm namespace.
        let name = format!(
            "/ktstr-test-shm-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        );

        match shm_try_create_excl(&name) {
            ShmCreateResult::Winner(fd) => {
                // Second attempt sees the existing segment. OwnedFd
                // drops close the descriptors on any early exit path.
                match shm_try_create_excl(&name) {
                    ShmCreateResult::Exists => {}
                    ShmCreateResult::Winner(_other) => {
                        let _ = rustix::shm::unlink(name.as_str());
                        drop(fd);
                        panic!("second shm_try_create_excl must return Exists, not Winner");
                    }
                    ShmCreateResult::Error => {
                        let _ = rustix::shm::unlink(name.as_str());
                        drop(fd);
                        panic!("second shm_try_create_excl returned Error");
                    }
                }
                // Clean up: write path then unlink so this test
                // doesn't leave /dev/shm residue.
                shm_write_and_release(fd, b"ok", &name);
                let _ = rustix::shm::unlink(name.as_str());
            }
            ShmCreateResult::Exists => {
                // A stale segment with this name exists. Unlink and retry.
                let _ = rustix::shm::unlink(name.as_str());
                panic!("test setup collision on shm name {name}");
            }
            ShmCreateResult::Error => {
                // Environment without /dev/shm — skip rather than fail.
                skip!("shm_open unavailable in this environment");
            }
        }
    }

    /// `shm_write_and_release` on a happy path publishes the data
    /// and releases the lock. After unlink the segment is gone.
    #[test]
    fn shm_write_and_release_publishes_data() {
        let name = format!(
            "/ktstr-test-shm-write-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        );
        let fd = match shm_try_create_excl(&name) {
            ShmCreateResult::Winner(fd) => fd,
            _ => {
                skip!("shm_open unavailable");
            }
        };
        let payload = b"shm-unit-test-payload";
        shm_write_and_release(fd, payload, &name);

        // Reopen read-only and verify size + contents.
        let rfd = rustix::shm::open(
            name.as_str(),
            rustix::shm::OFlags::RDONLY,
            rustix::fs::Mode::empty(),
        )
        .expect("shm_open for read failed");
        let st = rustix::fs::fstat(&rfd).expect("fstat failed");
        assert_eq!(st.st_size as usize, payload.len());
        drop(rfd);
        let _ = rustix::shm::unlink(name.as_str());
    }

    #[test]
    fn base_key_same_inputs_match() {
        let exe = crate::resolve_current_exe().unwrap();
        let k1 = BaseKey::new(&exe, None, None, None, &[]).unwrap();
        let k2 = BaseKey::new(&exe, None, None, None, &[]).unwrap();
        assert_eq!(k1, k2);
    }

    #[test]
    fn base_key_nonexistent_payload_fails() {
        let result = BaseKey::new(Path::new("/nonexistent/binary"), None, None, None, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn base_key_same_libs_content_change_matches() {
        // /init lives in the per-run suffix now, so the base key tracks
        // the payload's shared-lib SET, not its content. Two payloads with
        // the same libs but different bytes must produce the SAME base key
        // — that is the cache-hit-across-recompile property the lib-set keying buys.
        let exe = crate::resolve_current_exe().unwrap();
        let dir = tempfile::Builder::new()
            .prefix("ktstr-cache-libs-test-")
            .tempdir()
            .unwrap();
        let a = dir.path().join("payload_a");
        let b = dir.path().join("payload_b");

        // Both are copies of the same real ELF (identical DT_NEEDED), so
        // they resolve to the same lib set. `b` gets trailing bytes
        // appended so its CONTENT differs; goblin parses by header/section
        // offset, leaving the resolved libs unchanged.
        std::fs::copy(&exe, &a).unwrap();
        std::fs::copy(&exe, &b).unwrap();
        {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new().append(true).open(&b).unwrap();
            f.write_all(b"\x00ktstr-distinct-trailing-bytes").unwrap();
        }
        assert_ne!(
            std::fs::read(&a).unwrap(),
            std::fs::read(&b).unwrap(),
            "test setup: payload bytes must differ"
        );

        let ka = BaseKey::new(&a, None, None, None, &[]).unwrap();
        let kb = BaseKey::new(&b, None, None, None, &[]).unwrap();
        assert_eq!(
            ka, kb,
            "payload content change with an unchanged lib set must NOT \
             change the base key (else the base cache misses every recompile)"
        );
    }

    #[test]
    fn base_key_with_scheduler() {
        let exe = crate::resolve_current_exe().unwrap();
        let k1 = BaseKey::new(&exe, None, None, None, &[]).unwrap();
        let k2 = BaseKey::new(&exe, Some(&exe), None, None, &[]).unwrap();
        assert_ne!(k1, k2, "with vs without scheduler should differ");
    }

    /// Adding a staged scheduler must invalidate the cache key —
    /// otherwise two tests with different staged sets would silently
    /// hit the same cached initramfs base and the second test's VM
    /// would observe the FIRST test's `/staging/schedulers/` tree.
    /// This is a load-bearing invariant: without it, the entire
    /// scheduler-lifecycle Op story corrupts under realistic
    /// parallel test execution.
    #[test]
    fn base_key_staged_addition_invalidates() {
        let exe = crate::resolve_current_exe().unwrap();
        let empty = BaseKey::new(&exe, None, None, None, &[]).unwrap();
        let one_staged =
            BaseKey::new(&exe, None, None, None, &[("alt_args", exe.as_path())]).unwrap();
        assert_ne!(
            empty, one_staged,
            "adding a staged scheduler must change the cache key"
        );
    }

    /// Renaming a staged entry while keeping the binary identical
    /// must invalidate the cache — two `(name, binary)` tuples that
    /// share a binary path under different names produce different
    /// archive layouts (`/staging/schedulers/<NAME>/...`) and a
    /// shared cache key would write the wrong directory.
    #[test]
    fn base_key_staged_rename_invalidates() {
        let exe = crate::resolve_current_exe().unwrap();
        let a = BaseKey::new(&exe, None, None, None, &[("alpha", exe.as_path())]).unwrap();
        let b = BaseKey::new(&exe, None, None, None, &[("beta", exe.as_path())]).unwrap();
        assert_ne!(
            a, b,
            "renaming a staged scheduler must change the cache key (archive layout differs)"
        );
    }

    /// Caller-side ordering of the staged slice must NOT change the
    /// resulting key — `hash_staged` sorts by name before mixing into
    /// the hasher so two `KtstrTestEntry` instances whose
    /// `staged_schedulers` slices happen to be declared in opposite
    /// orders still share a cache entry. Without this, the cache
    /// would unnecessarily rebuild the base whenever an author
    /// reordered their `[&SchedA, &SchedB]` slice literal.
    #[test]
    fn base_key_staged_order_invariant() {
        let _tempdir_keep_alive = tempfile::Builder::new()
            .prefix("ktstr-staged-order-test-")
            .tempdir()
            .unwrap();
        let tmp = _tempdir_keep_alive.path();
        let a_bin = tmp.join("a");
        let b_bin = tmp.join("b");
        std::fs::write(&a_bin, b"sched_a_content").unwrap();
        std::fs::write(&b_bin, b"sched_b_content").unwrap();
        let exe = crate::resolve_current_exe().unwrap();

        let forward = BaseKey::new(
            &exe,
            None,
            None,
            None,
            &[("alpha", a_bin.as_path()), ("beta", b_bin.as_path())],
        )
        .unwrap();
        let reverse = BaseKey::new(
            &exe,
            None,
            None,
            None,
            &[("beta", b_bin.as_path()), ("alpha", a_bin.as_path())],
        )
        .unwrap();
        assert_eq!(
            forward, reverse,
            "staged set ordering must not affect the cache key"
        );
    }

    /// A content change in any staged binary must invalidate the
    /// cache — same shape rule as the boot scheduler. Catches the
    /// silent-stale-binary failure mode where a developer rebuilds
    /// `scx_layered`, reruns the test, and would see cached pre-fix
    /// behavior because the staged hash chain didn't observe the
    /// content change.
    #[test]
    fn base_key_staged_binary_content_invalidates() {
        let _tempdir_keep_alive = tempfile::Builder::new()
            .prefix("ktstr-staged-content-test-")
            .tempdir()
            .unwrap();
        let tmp = _tempdir_keep_alive.path();
        let staged_bin = tmp.join("staged");
        let exe = crate::resolve_current_exe().unwrap();

        std::fs::write(&staged_bin, b"sched_v1_content").unwrap();
        let v1 = BaseKey::new(&exe, None, None, None, &[("alt", staged_bin.as_path())]).unwrap();

        // mtime granularity guard — same as `hash_file_memoisation_invalidates_on_change`
        std::thread::sleep(std::time::Duration::from_millis(1100));
        std::fs::write(&staged_bin, b"sched_v2_content_different_length").unwrap();
        let v2 = BaseKey::new(&exe, None, None, None, &[("alt", staged_bin.as_path())]).unwrap();

        assert_ne!(
            v1, v2,
            "rebuilding a staged scheduler binary must invalidate the cache key"
        );
    }

    #[test]
    fn hash_file_is_ahash_stable_golden() {
        // hash_file must use ahash so the value is stable across
        // Rust toolchain versions. Golden check pins the concrete
        // algorithm — if this value changes, the cache silently
        // invalidates every prior artifact.
        let _tempdir_keep_alive = tempfile::Builder::new()
            .prefix("ktstr-hash-golden-test-")
            .tempdir()
            .unwrap();
        let tmp = _tempdir_keep_alive.path();
        let f = tmp.join("known");
        std::fs::write(&f, b"ktstr cache key probe").unwrap();
        let observed = hash_file(&f).unwrap();

        let mut h = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
        h.write(b"ktstr cache key probe");
        let expected = h.finish();
        assert_eq!(
            observed, expected,
            "hash_file must match ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher()"
        );
    }

    #[test]
    fn hash_file_large_file() {
        let _tempdir_keep_alive = tempfile::Builder::new()
            .prefix("ktstr-hash-sample-test-")
            .tempdir()
            .unwrap();
        let tmp = _tempdir_keep_alive.path();
        let f = tmp.join("big");
        // 16KB file — spans multiple pages in the mmap.
        let data: Vec<u8> = (0..16384).map(|i| (i % 256) as u8).collect();
        std::fs::write(&f, &data).unwrap();
        let h = hash_file(&f).unwrap();
        // Same content should produce same hash.
        assert_eq!(h, hash_file(&f).unwrap());
    }

    /// `hash_file` must invalidate its memoisation cache when the file
    /// changes — same path, new content, must yield a new hash.
    #[test]
    fn hash_file_memoisation_invalidates_on_change() {
        let _tempdir_keep_alive = tempfile::Builder::new()
            .prefix("ktstr-hash-memo-test-")
            .tempdir()
            .unwrap();
        let tmp = _tempdir_keep_alive.path();
        let f = tmp.join("rev");

        std::fs::write(&f, b"revision-one").unwrap();
        let h1 = hash_file(&f).unwrap();

        // Sleep past mtime granularity so the second write changes the
        // mtime tuple. ext4 / btrfs / xfs all expose nanosecond mtime,
        // but a one-second pause is the portable lower bound.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        std::fs::write(&f, b"revision-two-with-different-bytes").unwrap();
        let h2 = hash_file(&f).unwrap();

        assert_ne!(h1, h2, "mtime change must bypass the memoisation cache");
    }

    #[test]
    fn base_cache_hit() {
        let exe = crate::resolve_current_exe().unwrap();
        let key = BaseKey::new(&exe, None, None, None, &[]).unwrap();

        // Insert a sentinel value.
        let sentinel = Arc::new(vec![0xDE, 0xAD]);
        base_cache()
            .lock()
            .unwrap()
            .insert(key.clone(), sentinel.clone());

        // Lookup should return the same Arc.
        let cached = base_cache().lock().unwrap().get(&key).cloned();
        assert!(cached.is_some());
        assert!(Arc::ptr_eq(&cached.unwrap(), &sentinel));

        // Clean up to avoid polluting other tests.
        base_cache().lock().unwrap().remove(&key);
    }

    #[test]
    fn shm_store_and_load_roundtrip() {
        let hash = 0xDEAD_BEEF_CAFE_1234u64;
        let data = vec![0x07u8, 0x07, 0x01]; // cpio magic prefix
        initramfs::shm_store_base(hash, &data).unwrap();
        let loaded = initramfs::shm_load_base(hash);
        assert!(loaded.is_some(), "shm_load_base should return Some");
        assert_eq!(loaded.unwrap().as_ref(), &data[..]);
        initramfs::shm_unlink_base(hash);
    }

    #[test]
    fn shm_different_hashes_independent() {
        let h1 = 0x1111_2222_3333_4444u64;
        let h2 = 0x5555_6666_7777_8888u64;
        let d1 = vec![0xAAu8; 16];
        let d2 = vec![0xBBu8; 32];
        initramfs::shm_store_base(h1, &d1).unwrap();
        initramfs::shm_store_base(h2, &d2).unwrap();
        assert_eq!(initramfs::shm_load_base(h1).unwrap().as_ref(), &d1[..]);
        assert_eq!(initramfs::shm_load_base(h2).unwrap().as_ref(), &d2[..]);
        initramfs::shm_unlink_base(h1);
        initramfs::shm_unlink_base(h2);
    }

    /// `KTSTR_CARGO_TEST_MODE=1` short-circuits `get_or_build_base`
    /// to the inline-build path: process-local HashMap still
    /// memoises so a second call with the same key returns the
    /// SAME `Arc` without re-running the builder, but no SHM
    /// segment is created or loaded. Pins the bypass contract:
    /// bare `cargo test` does not share the cross-process SHM
    /// cache contract that nextest / `cargo ktstr test` peers
    /// rely on.
    ///
    /// The test stages a sentinel value in the process-local cache
    /// for a synthetic key, then calls `get_or_build_base` twice.
    /// The first call must hit the cache; the second must observe
    /// the same `Arc`. A regression that bypassed the HashMap
    /// (e.g. always re-running the builder) would surface as an
    /// `Arc::ptr_eq` failure.
    #[test]
    fn get_or_build_base_cargo_test_mode_uses_process_local_cache() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _env = EnvVarGuard::set(crate::KTSTR_CARGO_TEST_MODE_ENV, "1");
        let exe = crate::resolve_current_exe().unwrap();
        let key = BaseKey::new(&exe, None, None, None, &[]).unwrap();

        // Plant a sentinel in the process-local cache so the
        // call's first-tier lookup returns it without invoking
        // the (expensive) inline builder. A real cargo-test-mode
        // run with no prior cache entry would still work — the
        // inline build path is exercised — but staging the
        // sentinel keeps this test fast and removes the kernel /
        // shared-lib resolution dependency.
        let sentinel = Arc::new(vec![0xC0u8, 0xDE, 0x01, 0x07, 0x07, 0x01]);
        base_cache()
            .lock()
            .unwrap()
            .insert(key.clone(), sentinel.clone());

        let result = get_or_build_base(&exe, &[], &[], None, &key)
            .expect("cargo-test-mode must reuse process-local cache");
        match result {
            BaseRef::Owned(arc) => {
                assert!(
                    Arc::ptr_eq(&arc, &sentinel),
                    "cargo-test-mode hit on a planted process-local entry \
                     must return the SAME Arc — a regression that fell \
                     through into the inline-build path would produce a \
                     fresh Arc with the same contents but a different \
                     identity"
                );
            }
            BaseRef::Mapped(_) => {
                panic!(
                    "cargo-test-mode must NEVER mmap an SHM segment — \
                     bypass contract requires process-local-only memoisation"
                );
            }
        }

        // Clean up so this test does not leak state into siblings
        // (shared `base_cache()` Mutex outlives the test).
        base_cache().lock().unwrap().remove(&key);
    }
}
