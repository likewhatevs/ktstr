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
#[cfg(test)]
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
#[cfg(test)]
use std::hash::Hash;
use std::hash::Hasher;
use std::io::{Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::fs::{FileExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

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
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct HashFileKey {
    dev: u64,
    ino: u64,
    size: u64,
    mtime_secs: i64,
    mtime_nsecs: i64,
    ctime_secs: i64,
    ctime_nsecs: i64,
}

/// Process-local cache: file identity + mtime → ahash of contents.
#[cfg(test)]
fn hash_file_cache() -> &'static Mutex<HashMap<HashFileKey, u64>> {
    static CACHE: OnceLock<Mutex<HashMap<HashFileKey, u64>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Hash a file's content for cache keying via mmap + fixed-seed ahash.
///
/// The stable open-file identity first checks the process-local map, then a
/// machine-wide digest memo under the initramfs CAS. Exactly one process
/// streams a given inode revision; storm peers read the 80-byte memo. The
/// final digest (not the identity) remains the semantic cache-key input, so
/// byte-identical files at different paths/inodes still converge.
#[cfg(test)]
pub(crate) fn hash_file(path: &Path) -> Result<u64> {
    let file = File::open(path).with_context(|| format!("open for hash: {}", path.display()))?;
    let identity = StableFileIdentity::from_file(&file)
        .with_context(|| format!("stat for hash: {}", path.display()))?;
    let cache_key = HashFileKey {
        dev: identity.dev,
        ino: identity.ino,
        size: identity.size,
        mtime_secs: identity.mtime_secs,
        mtime_nsecs: identity.mtime_nsecs,
        ctime_secs: identity.ctime_secs,
        ctime_nsecs: identity.ctime_nsecs,
    };
    if let Some(cached) = hash_file_cache().lock().unwrap().get(&cache_key).copied() {
        return Ok(cached);
    }

    let root = prepared_cache_root()?;
    ensure_prepared_cache_dirs(&root)?;
    let digest = cached_file_digest(&root, &file, identity)
        .with_context(|| format!("machine-wide hash memo: {}", path.display()))?;
    hash_file_cache().lock().unwrap().insert(cache_key, digest);
    Ok(digest)
}

#[cfg(test)]
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
            // The payload's DT_NEEDED closure; the interp's own deps (below)
            // are folded in before the single sorted hash pass.
            let mut entries: Vec<_> = result.found.iter().map(|(_, p)| p.clone()).collect();
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
                // A non-standard interp (custom toolchain linker) can itself
                // be dynamically linked; build_initramfs_base packs its own
                // dep chain, so fold those host paths into the hashed set too
                // — else an interp-dep change (interp binary unchanged) would
                // serve a stale base. resolve_interpreter_deps is the same
                // resolution the base packer uses (empty for a standard ld.so).
                if let Ok(interp_deps) = initramfs::resolve_interpreter_deps(interp) {
                    entries.extend(interp_deps.found.into_iter().map(|(_, p)| p));
                }
            }
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

/// Process-global cache for base initramfs bytes. Keyed by [`BaseKey`]
/// (a `u64`): the payload's shared-lib set + interpreter (NOT the
/// payload's content), plus the content hashes of the scheduler /
/// probe / worker and staged binaries. Shell keys additionally mix in
/// a sentinel, include files, and the busybox bytes.
/// The lock is only held during map lookup/insert, never during the
/// actual build.
#[cfg(test)]
pub(crate) fn base_cache() -> &'static Mutex<HashMap<BaseKey, Arc<Vec<u8>>>> {
    static CACHE: OnceLock<Mutex<HashMap<BaseKey, Arc<Vec<u8>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Holds either a borrowed shm mapping or an owned Arc from the
/// process-local cache / a fresh build.
#[cfg(test)]
pub(crate) enum BaseRef {
    Mapped(initramfs::MappedShm),
    Owned(Arc<Vec<u8>>),
}

#[cfg(test)]
impl AsRef<[u8]> for BaseRef {
    fn as_ref(&self) -> &[u8] {
        match self {
            BaseRef::Mapped(m) => m.as_ref(),
            BaseRef::Owned(a) => a,
        }
    }
}

// Prepared initrd CAS and direct-COW mapping live here alongside the
// pre-existing uncompressed-base cache. The two layers deliberately have
// different lifetimes: base SHM is a build accelerator, while the regular
// file CAS is the immutable backing mapped into every guest.

const PREPARED_CAS_SCHEMA: u32 = 3;
const PREPARED_CAS_MAGIC: &[u8; 8] = b"KTSTRIR\0";
const PREPARED_CAS_HEADER_LEN: usize = 128;
const PREPARED_CAS_MAX_BYTES: u64 = 8 << 30;
const PREPARED_CAS_MAX_AGE_SECS: u64 = 30 * 24 * 60 * 60;
const PREPARED_CAS_GC_INTERVAL_SECS: u64 = 60 * 60;
const FILE_DIGEST_MAGIC: &[u8; 8] = b"KTSTRDG\0";
const FILE_DIGEST_RECORD_LEN: usize = 88;
const COVERAGE_PROBE_MAGIC: &[u8; 8] = b"KTSTRCV\0";
const COVERAGE_PROBE_RECORD_LEN: usize = 48;
const CLOSURE_RECORD_MAGIC: &[u8; 8] = b"KTSTRCL\0";
const CLOSURE_RECORD_HEADER_LEN: usize = 40;
const PREPARED_OBJECTS_DIR: &str = "prepared-initrd-objects";
const PREPARED_LOCKS_DIR: &str = ".prepared-initrd-locks";
const PREPARED_DIGESTS_DIR: &str = "prepared-initrd-digests";
const PREPARED_PROBES_DIR: &str = "prepared-initrd-probes";
const PREPARED_CLOSURES_DIR: &str = "prepared-initrd-closures";
const PREPARED_GC_STAMP: &str = ".prepared-initrd-gc-stamp";
/// Uniform direct-map granule. It is host-page aligned on every supported
/// host and also satisfies Linux hugetlb's MAP_FIXED unmap boundary rule for
/// the VM's optional MAP_HUGETLB|MAP_HUGE_2MB guest backing.
pub(crate) const PREPARED_MAPPING_GRANULE: usize = 2 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum PreparedObjectKind {
    Base = 1,
    Payload = 2,
    Modules = 3,
    Tail = 4,
    PayloadView = 5,
    ModulesView = 6,
    Stitch = 7,
}

impl PreparedObjectKind {
    fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::Base),
            2 => Ok(Self::Payload),
            3 => Ok(Self::Modules),
            4 => Ok(Self::Tail),
            5 => Ok(Self::PayloadView),
            6 => Ok(Self::ModulesView),
            7 => Ok(Self::Stitch),
            _ => anyhow::bail!("unknown prepared initrd object kind {tag}"),
        }
    }

    fn stem(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Payload => "payload",
            Self::Modules => "modules",
            Self::Tail => "tail",
            Self::PayloadView => "payload-view",
            Self::ModulesView => "modules-view",
            Self::Stitch => "stitch",
        }
    }
}

fn compression_tag(compression: initramfs::InitrdCompression) -> u8 {
    match compression {
        initramfs::InitrdCompression::Lz4 => 1,
        initramfs::InitrdCompression::Zstd => 2,
        initramfs::InitrdCompression::Gzip => 3,
        initramfs::InitrdCompression::Uncompressed => 4,
    }
}

fn compression_from_tag(tag: u8) -> Result<initramfs::InitrdCompression> {
    match tag {
        1 => Ok(initramfs::InitrdCompression::Lz4),
        2 => Ok(initramfs::InitrdCompression::Zstd),
        3 => Ok(initramfs::InitrdCompression::Gzip),
        4 => Ok(initramfs::InitrdCompression::Uncompressed),
        _ => anyhow::bail!("unknown prepared initrd compression tag {tag}"),
    }
}

fn fixed_hasher() -> AHasher {
    ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher()
}

fn ahash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = fixed_hasher();
    hasher.write(bytes);
    hasher.finish()
}

fn hash_u8(hasher: &mut AHasher, value: u8) {
    hasher.write(&[value]);
}

fn hash_u32(hasher: &mut AHasher, value: u32) {
    hasher.write(&value.to_le_bytes());
}

fn hash_u64(hasher: &mut AHasher, value: u64) {
    hasher.write(&value.to_le_bytes());
}

fn hash_i64(hasher: &mut AHasher, value: i64) {
    hasher.write(&value.to_le_bytes());
}

fn hash_len_prefixed(hasher: &mut AHasher, bytes: &[u8]) {
    hash_u64(hasher, bytes.len() as u64);
    hasher.write(bytes);
}

fn hash_string_slice(hasher: &mut AHasher, values: &[String]) {
    hash_u64(hasher, values.len() as u64);
    for value in values {
        hash_len_prefixed(hasher, value.as_bytes());
    }
}

fn hash_optional_str(hasher: &mut AHasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hash_u8(hasher, 1);
            hash_len_prefixed(hasher, value.as_bytes());
        }
        None => hash_u8(hasher, 0),
    }
}

fn round_up(value: usize, align: usize) -> Result<usize> {
    anyhow::ensure!(align.is_power_of_two(), "alignment is not a power of two");
    value
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .context("aligned length overflow")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct StableFileIdentity {
    dev: u64,
    ino: u64,
    size: u64,
    mtime_secs: i64,
    mtime_nsecs: i64,
    ctime_secs: i64,
    ctime_nsecs: i64,
}

impl StableFileIdentity {
    fn from_metadata(meta: &std::fs::Metadata) -> Self {
        Self {
            dev: meta.dev(),
            ino: meta.ino(),
            size: meta.size(),
            mtime_secs: meta.mtime(),
            mtime_nsecs: meta.mtime_nsec(),
            ctime_secs: meta.ctime(),
            ctime_nsecs: meta.ctime_nsec(),
        }
    }

    fn from_file(file: &File) -> Result<Self> {
        let meta = file.metadata().context("stat pinned input")?;
        anyhow::ensure!(
            meta.is_file(),
            "prepared initrd input is not a regular file"
        );
        Ok(Self::from_metadata(&meta))
    }

    fn from_resolver(identity: initramfs::ResolverFileIdentity) -> Self {
        Self {
            dev: identity.dev,
            ino: identity.ino,
            size: identity.size,
            mtime_secs: identity.mtime_secs,
            mtime_nsecs: identity.mtime_nsecs,
            ctime_secs: identity.ctime_secs,
            ctime_nsecs: identity.ctime_nsecs,
        }
    }

    fn hash_into(self, hasher: &mut AHasher) {
        hash_u64(hasher, self.dev);
        hash_u64(hasher, self.ino);
        hash_u64(hasher, self.size);
        hash_i64(hasher, self.mtime_secs);
        hash_i64(hasher, self.mtime_nsecs);
        hash_i64(hasher, self.ctime_secs);
        hash_i64(hasher, self.ctime_nsecs);
    }

    fn encode(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.dev.to_le_bytes());
        out.extend_from_slice(&self.ino.to_le_bytes());
        out.extend_from_slice(&self.size.to_le_bytes());
        out.extend_from_slice(&self.mtime_secs.to_le_bytes());
        out.extend_from_slice(&self.mtime_nsecs.to_le_bytes());
        out.extend_from_slice(&self.ctime_secs.to_le_bytes());
        out.extend_from_slice(&self.ctime_nsecs.to_le_bytes());
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(bytes.len() == 56, "invalid file identity record length");
        Ok(Self {
            dev: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            ino: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            size: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            mtime_secs: i64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            mtime_nsecs: i64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            ctime_secs: i64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            ctime_nsecs: i64::from_le_bytes(bytes[48..56].try_into().unwrap()),
        })
    }
}

#[derive(Debug)]
struct PinnedInput {
    file: File,
    identity: StableFileIdentity,
    content_hash: u64,
    display_path: PathBuf,
}

impl PinnedInput {
    fn proc_path(&self) -> PathBuf {
        PathBuf::from(format!(
            "/proc/{}/fd/{}",
            std::process::id(),
            self.file.as_raw_fd()
        ))
    }

    fn verify_unchanged(&self) -> Result<()> {
        let after = StableFileIdentity::from_file(&self.file)?;
        anyhow::ensure!(
            after == self.identity,
            "prepared initrd input changed while in use: {}",
            self.display_path.display()
        );
        Ok(())
    }

    fn is_elf(&self) -> Result<bool> {
        let mut magic = [0u8; 4];
        let read = self
            .file
            .read_at(&mut magic, 0)
            .with_context(|| format!("probe ELF input {}", self.display_path.display()))?;
        Ok(read == magic.len() && magic == *b"\x7fELF")
    }
}

fn prepared_cache_root() -> Result<PathBuf> {
    crate::cache::resolve_cache_root_with_suffix("initramfs")
}

fn ensure_prepared_cache_dirs(root: &Path) -> Result<()> {
    std::fs::create_dir_all(root.join(PREPARED_OBJECTS_DIR))
        .with_context(|| format!("create prepared initrd CAS {}", root.display()))?;
    std::fs::create_dir_all(root.join(PREPARED_LOCKS_DIR))
        .with_context(|| format!("create prepared initrd lock dir {}", root.display()))?;
    std::fs::create_dir_all(root.join(PREPARED_DIGESTS_DIR))
        .with_context(|| format!("create prepared initrd digest dir {}", root.display()))?;
    std::fs::create_dir_all(root.join(PREPARED_PROBES_DIR))
        .with_context(|| format!("create prepared initrd probe dir {}", root.display()))?;
    std::fs::create_dir_all(root.join(PREPARED_CLOSURES_DIR))
        .with_context(|| format!("create prepared initrd closure dir {}", root.display()))?;
    Ok(())
}

fn open_coord_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open prepared initrd lock {}", path.display()))
}

fn hash_pinned_file(file: &File, identity: StableFileIdentity) -> Result<u64> {
    let before = StableFileIdentity::from_file(file)?;
    anyhow::ensure!(before == identity, "pinned input changed before hashing");
    // Never mmap a mutable input here: a concurrent in-place truncate can
    // SIGBUS an mmap reader before the post-stat check gets a chance to turn
    // the mutation into a normal error. Exactly one elected process performs
    // this streaming read, so a fixed-size pread buffer is cheap and safe.
    let mut hasher = fixed_hasher();
    let mut offset = 0u64;
    let mut buffer = vec![0u8; 1 << 20];
    while offset < identity.size {
        let remaining = usize::try_from((identity.size - offset).min(buffer.len() as u64))?;
        let read = file
            .read_at(&mut buffer[..remaining], offset)
            .context("pread pinned input for digest")?;
        anyhow::ensure!(
            read > 0,
            "pinned input was truncated while hashing its content"
        );
        hasher.write(&buffer[..read]);
        offset = offset
            .checked_add(read as u64)
            .context("pinned input digest offset overflow")?;
    }
    let digest = hasher.finish();
    let after = StableFileIdentity::from_file(file)?;
    anyhow::ensure!(
        after == identity,
        "pinned input changed while hashing its content"
    );
    Ok(digest)
}

fn read_file_digest_record(
    record_path: &Path,
    identity: StableFileIdentity,
) -> Result<Option<u64>> {
    let bytes = match std::fs::read(record_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read file digest memo {}", record_path.display()));
        }
    };
    anyhow::ensure!(
        bytes.len() == FILE_DIGEST_RECORD_LEN,
        "file digest memo is truncated: {}",
        record_path.display()
    );
    anyhow::ensure!(
        &bytes[..8] == FILE_DIGEST_MAGIC,
        "file digest memo magic mismatch"
    );
    anyhow::ensure!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) == PREPARED_CAS_SCHEMA,
        "file digest memo schema mismatch"
    );
    anyhow::ensure!(
        bytes[12..16] == [0; 4],
        "file digest memo reserved bytes changed"
    );
    let recorded_identity = StableFileIdentity::decode(&bytes[16..72])?;
    anyhow::ensure!(
        recorded_identity == identity,
        "file digest memo identity collision"
    );
    let recorded_checksum = u64::from_le_bytes(bytes[80..88].try_into().unwrap());
    let mut canonical = bytes;
    canonical[80..88].fill(0);
    anyhow::ensure!(
        ahash_bytes(&canonical) == recorded_checksum,
        "file digest memo checksum mismatch"
    );
    Ok(Some(u64::from_le_bytes(
        canonical[72..80].try_into().unwrap(),
    )))
}

fn cached_file_digest(root: &Path, file: &File, identity: StableFileIdentity) -> Result<u64> {
    let mut identity_hasher = fixed_hasher();
    hash_len_prefixed(&mut identity_hasher, b"ktstr-file-digest-identity");
    identity.hash_into(&mut identity_hasher);
    let identity_key = identity_hasher.finish();
    let record_path = root
        .join(PREPARED_DIGESTS_DIR)
        .join(format!("{identity_key:016x}.digest"));
    if let Some(digest) = read_file_digest_record(&record_path, identity)? {
        return Ok(digest);
    }

    let lock_path = root
        .join(PREPARED_LOCKS_DIR)
        .join(format!("digest-{identity_key:016x}.lock"));
    loop {
        let coordination = open_coord_file(&lock_path)?;
        match rustix::fs::flock(
            &coordination,
            rustix::fs::FlockOperation::NonBlockingLockExclusive,
        ) {
            Ok(()) => {
                if let Some(digest) = read_file_digest_record(&record_path, identity)? {
                    return Ok(digest);
                }
                let digest = hash_pinned_file(file, identity)?;
                let mut bytes = Vec::with_capacity(FILE_DIGEST_RECORD_LEN);
                bytes.extend_from_slice(FILE_DIGEST_MAGIC);
                bytes.extend_from_slice(&PREPARED_CAS_SCHEMA.to_le_bytes());
                bytes.extend_from_slice(&[0; 4]);
                identity.encode(&mut bytes);
                bytes.extend_from_slice(&digest.to_le_bytes());
                bytes.extend_from_slice(&0u64.to_le_bytes());
                let checksum = ahash_bytes(&bytes);
                bytes[80..88].copy_from_slice(&checksum.to_le_bytes());
                debug_assert_eq!(bytes.len(), FILE_DIGEST_RECORD_LEN);

                let mut temp = tempfile::Builder::new()
                    .prefix(&format!(".tmp-digest-{identity_key:016x}-"))
                    .tempfile_in(root.join(PREPARED_DIGESTS_DIR))
                    .context("create file digest memo temp")?;
                temp.write_all(&bytes).context("write file digest memo")?;
                temp.as_file().sync_all().context("sync file digest memo")?;
                temp.persist(&record_path)
                    .map_err(|error| error.error)
                    .with_context(|| {
                        format!("publish file digest memo {}", record_path.display())
                    })?;
                return Ok(digest);
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                // One builder streams this inode. Peers wait as shared
                // readers, wake together when its EX lock drops, and read
                // the tiny record concurrently. If the builder died before
                // publication, loop and elect a successor.
                rustix::fs::flock(&coordination, rustix::fs::FlockOperation::LockShared)
                    .with_context(|| format!("wait for file digest {}", record_path.display()))?;
                if let Some(digest) = read_file_digest_record(&record_path, identity)? {
                    return Ok(digest);
                }
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("elect file digest builder {}", record_path.display())
                });
            }
        }
    }
}

fn pin_input(root: &Path, path: &Path) -> Result<PinnedInput> {
    let file =
        File::open(path).with_context(|| format!("open prepared input {}", path.display()))?;
    let identity = StableFileIdentity::from_file(&file)
        .with_context(|| format!("stat prepared input {}", path.display()))?;
    let content_hash = cached_file_digest(root, &file, identity)
        .with_context(|| format!("digest prepared input {}", path.display()))?;
    Ok(PinnedInput {
        file,
        identity,
        content_hash,
        display_path: path.to_path_buf(),
    })
}

fn read_coverage_probe_record(path: &Path, key: u64) -> Result<Option<(bool, u64)>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read coverage probe memo {}", path.display()));
        }
    };
    anyhow::ensure!(
        bytes.len() == COVERAGE_PROBE_RECORD_LEN,
        "coverage probe memo is truncated: {}",
        path.display()
    );
    anyhow::ensure!(
        &bytes[..8] == COVERAGE_PROBE_MAGIC,
        "coverage probe memo magic mismatch"
    );
    anyhow::ensure!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) == PREPARED_CAS_SCHEMA,
        "coverage probe memo schema mismatch"
    );
    anyhow::ensure!(
        bytes[12..16] == [0; 4]
            && u64::from_le_bytes(bytes[16..24].try_into().unwrap()) == key
            && bytes[25..32] == [0; 7],
        "coverage probe memo recipe mismatch"
    );
    let recorded_checksum = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
    let mut canonical = bytes;
    canonical[40..48].fill(0);
    anyhow::ensure!(
        ahash_bytes(&canonical) == recorded_checksum,
        "coverage probe memo checksum mismatch"
    );
    let instrumented = match canonical[24] {
        0 => false,
        1 => true,
        value => anyhow::bail!("coverage probe memo has invalid boolean tag {value}"),
    };
    let reserve = u64::from_le_bytes(canonical[32..40].try_into().unwrap());
    Ok(Some((instrumented, reserve)))
}

fn inspect_coverage_from_pinned(payload: &PinnedInput) -> Result<(bool, u64)> {
    payload.verify_unchanged()?;
    if payload.identity.size == 0 {
        return Ok((false, 0));
    }
    let mmap = unsafe { memmap2::Mmap::map(&payload.file) }
        .with_context(|| format!("mmap coverage payload {}", payload.display_path.display()))?;
    let Ok(elf) = goblin::elf::Elf::parse(&mmap) else {
        payload.verify_unchanged()?;
        return Ok((false, 0));
    };
    let vaddrs = crate::test_support::find_symbol_vaddrs(
        &elf,
        &[
            "__llvm_profile_write_buffer",
            "__llvm_profile_get_size_for_buffer",
        ],
    );
    let instrumented = vaddrs
        .iter()
        .any(|value| value.is_some_and(|address| address != 0));
    let reserve = if instrumented {
        elf.section_headers
            .iter()
            .filter_map(|section| {
                elf.shdr_strtab
                    .get_at(section.sh_name)
                    .filter(|name| *name == "__llvm_prf_cnts" || *name == "__llvm_prf_data")
                    .map(|_| section.sh_size)
            })
            .fold(0u64, u64::saturating_add)
    } else {
        0
    };
    drop(elf);
    drop(mmap);
    payload.verify_unchanged()?;
    Ok((instrumented, reserve))
}

fn cached_coverage_probe(root: &Path, payload: &PinnedInput) -> Result<(bool, u64)> {
    let mut key_hasher = fixed_hasher();
    hash_len_prefixed(&mut key_hasher, b"ktstr-coverage-probe");
    hash_u32(&mut key_hasher, PREPARED_CAS_SCHEMA);
    hash_u64(&mut key_hasher, payload.identity.size);
    hash_u64(&mut key_hasher, payload.content_hash);
    let key = key_hasher.finish();
    let record_path = root
        .join(PREPARED_PROBES_DIR)
        .join(format!("{key:016x}.coverage"));
    if let Some(result) = read_coverage_probe_record(&record_path, key)? {
        return Ok(result);
    }
    let lock_path = root
        .join(PREPARED_LOCKS_DIR)
        .join(format!("coverage-{key:016x}.lock"));
    loop {
        let coordination = open_coord_file(&lock_path)?;
        match rustix::fs::flock(
            &coordination,
            rustix::fs::FlockOperation::NonBlockingLockExclusive,
        ) {
            Ok(()) => {
                if let Some(result) = read_coverage_probe_record(&record_path, key)? {
                    return Ok(result);
                }
                let result = inspect_coverage_from_pinned(payload)?;
                let mut bytes = Vec::with_capacity(COVERAGE_PROBE_RECORD_LEN);
                bytes.extend_from_slice(COVERAGE_PROBE_MAGIC);
                bytes.extend_from_slice(&PREPARED_CAS_SCHEMA.to_le_bytes());
                bytes.extend_from_slice(&[0; 4]);
                bytes.extend_from_slice(&key.to_le_bytes());
                bytes.push(u8::from(result.0));
                bytes.extend_from_slice(&[0; 7]);
                bytes.extend_from_slice(&result.1.to_le_bytes());
                bytes.extend_from_slice(&0u64.to_le_bytes());
                let checksum = ahash_bytes(&bytes);
                bytes[40..48].copy_from_slice(&checksum.to_le_bytes());
                let mut temp = tempfile::Builder::new()
                    .prefix(&format!(".tmp-coverage-{key:016x}-"))
                    .tempfile_in(root.join(PREPARED_PROBES_DIR))
                    .context("create coverage probe memo temp")?;
                temp.write_all(&bytes)
                    .context("write coverage probe memo")?;
                temp.as_file()
                    .sync_all()
                    .context("sync coverage probe memo")?;
                temp.persist(&record_path)
                    .map_err(|error| error.error)
                    .with_context(|| {
                        format!("publish coverage probe memo {}", record_path.display())
                    })?;
                return Ok(result);
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                rustix::fs::flock(&coordination, rustix::fs::FlockOperation::LockShared)
                    .with_context(|| {
                        format!("wait for coverage probe {}", record_path.display())
                    })?;
                if let Some(result) = read_coverage_probe_record(&record_path, key)? {
                    return Ok(result);
                }
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("elect coverage probe builder {}", record_path.display())
                });
            }
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClosureEntryRecord {
    guest_path: String,
    host_path: PathBuf,
    identity: StableFileIdentity,
    content_hash: u64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClosureSearchRecord {
    path: PathBuf,
    identity: Option<StableFileIdentity>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClosureRecord {
    entries: Vec<ClosureEntryRecord>,
    search_paths: Vec<ClosureSearchRecord>,
}

#[derive(Debug)]
struct PinnedClosureEntry {
    guest_path: String,
    input: PinnedInput,
}

#[derive(Debug)]
struct PinnedClosure {
    entries: Vec<PinnedClosureEntry>,
}

#[derive(Debug)]
struct PinnedLoaderInputs {
    cwd: PathBuf,
    ld_library_path_present: bool,
    ld_library_path_raw: Vec<u8>,
    ld_library_path_dirs: Vec<PathBuf>,
    ld_so_cache: Option<PinnedInput>,
}

impl PinnedLoaderInputs {
    fn pin(root: &Path) -> Result<Self> {
        let cwd = std::env::current_dir().context("read loader current directory")?;
        let cwd = std::fs::canonicalize(&cwd)
            .with_context(|| format!("canonicalize loader current directory {}", cwd.display()))?;
        let ld_library_path = std::env::var_os("LD_LIBRARY_PATH");
        let ld_library_path_raw = ld_library_path
            .as_deref()
            .map(|value| value.as_encoded_bytes().to_vec())
            .unwrap_or_default();
        let ld_library_path_dirs = ld_library_path
            .as_deref()
            .map(|value| std::env::split_paths(value).collect())
            .unwrap_or_default();
        let ld_so_cache = match pin_input(root, Path::new("/etc/ld.so.cache")) {
            Ok(input) => Some(input),
            Err(error)
                if error
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound) =>
            {
                None
            }
            Err(error) => return Err(error).context("pin /etc/ld.so.cache"),
        };
        Ok(Self {
            cwd,
            ld_library_path_present: ld_library_path.is_some(),
            ld_library_path_raw,
            ld_library_path_dirs,
            ld_so_cache,
        })
    }

    fn cache_proc_path(&self) -> Option<PathBuf> {
        self.ld_so_cache.as_ref().map(PinnedInput::proc_path)
    }

    fn verify_unchanged(&self) -> Result<()> {
        if let Some(cache) = &self.ld_so_cache {
            cache.verify_unchanged()?;
        }
        Ok(())
    }
}

fn closure_recipe_key(
    binary: &PinnedInput,
    original_path: &Path,
    loader: &PinnedLoaderInputs,
) -> u64 {
    let mut hasher = fixed_hasher();
    hash_len_prefixed(&mut hasher, b"ktstr-loader-closure");
    hash_u32(&mut hasher, PREPARED_CAS_SCHEMA);
    hash_len_prefixed(&mut hasher, std::env::consts::ARCH.as_bytes());
    hash_u64(&mut hasher, binary.identity.size);
    hash_u64(&mut hasher, binary.content_hash);
    hash_len_prefixed(&mut hasher, original_path.as_os_str().as_encoded_bytes());
    hash_len_prefixed(&mut hasher, loader.cwd.as_os_str().as_encoded_bytes());
    hash_u8(&mut hasher, u8::from(loader.ld_library_path_present));
    hash_len_prefixed(&mut hasher, &loader.ld_library_path_raw);
    match &loader.ld_so_cache {
        Some(cache) => {
            hash_u8(&mut hasher, 1);
            hash_u64(&mut hasher, cache.identity.size);
            hash_u64(&mut hasher, cache.content_hash);
        }
        None => hash_u8(&mut hasher, 0),
    }
    hasher.finish()
}

fn path_identity(path: &Path) -> Result<Option<StableFileIdentity>> {
    match std::fs::metadata(path) {
        Ok(metadata) => Ok(Some(StableFileIdentity::from_metadata(&metadata))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => {
            Err(error).with_context(|| format!("stat loader search path {}", path.display()))
        }
    }
}

fn open_recorded_closure_entry(record: &ClosureEntryRecord) -> Result<Option<PinnedClosureEntry>> {
    let file = match File::open(&record.host_path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "open recorded loader dependency {}",
                    record.host_path.display()
                )
            });
        }
    };
    let identity = StableFileIdentity::from_file(&file).with_context(|| {
        format!(
            "stat recorded loader dependency {}",
            record.host_path.display()
        )
    })?;
    if identity != record.identity {
        return Ok(None);
    }
    Ok(Some(PinnedClosureEntry {
        guest_path: record.guest_path.clone(),
        input: PinnedInput {
            file,
            identity,
            content_hash: record.content_hash,
            display_path: record.host_path.clone(),
        },
    }))
}

fn encode_closure_record(record: &ClosureRecord, key: u64) -> Result<Vec<u8>> {
    let payload = postcard::to_stdvec(record).context("encode loader closure payload")?;
    let mut bytes = Vec::with_capacity(CLOSURE_RECORD_HEADER_LEN + payload.len());
    bytes.extend_from_slice(CLOSURE_RECORD_MAGIC);
    bytes.extend_from_slice(&PREPARED_CAS_SCHEMA.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&key.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&payload);
    let checksum = ahash_bytes(&bytes);
    bytes[32..40].copy_from_slice(&checksum.to_le_bytes());
    Ok(bytes)
}

fn read_pinned_closure_record(
    record_path: &Path,
    expected_key: u64,
) -> Result<Option<PinnedClosure>> {
    let mut bytes = match std::fs::read(record_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read loader closure {}", record_path.display()));
        }
    };
    anyhow::ensure!(
        bytes.len() >= CLOSURE_RECORD_HEADER_LEN,
        "loader closure envelope is truncated: {}",
        record_path.display()
    );
    anyhow::ensure!(
        &bytes[..8] == CLOSURE_RECORD_MAGIC
            && u32::from_le_bytes(bytes[8..12].try_into().unwrap()) == PREPARED_CAS_SCHEMA
            && bytes[12..16] == [0; 4]
            && u64::from_le_bytes(bytes[16..24].try_into().unwrap()) == expected_key,
        "loader closure envelope recipe mismatch: {}",
        record_path.display()
    );
    let payload_len = usize::try_from(u64::from_le_bytes(bytes[24..32].try_into().unwrap()))
        .context("loader closure payload length exceeds usize")?;
    anyhow::ensure!(
        bytes.len() == CLOSURE_RECORD_HEADER_LEN + payload_len,
        "loader closure envelope length mismatch: {}",
        record_path.display()
    );
    let recorded_checksum = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
    bytes[32..40].fill(0);
    anyhow::ensure!(
        ahash_bytes(&bytes) == recorded_checksum,
        "loader closure checksum mismatch: {}",
        record_path.display()
    );
    let record: ClosureRecord = postcard::from_bytes(&bytes[CLOSURE_RECORD_HEADER_LEN..])
        .with_context(|| format!("decode loader closure {}", record_path.display()))?;
    for search in &record.search_paths {
        if path_identity(&search.path)? != search.identity {
            return Ok(None);
        }
    }
    let mut entries = Vec::with_capacity(record.entries.len());
    for entry in &record.entries {
        let Some(entry) = open_recorded_closure_entry(entry)? else {
            return Ok(None);
        };
        entries.push(entry);
    }
    Ok(Some(PinnedClosure { entries }))
}

fn merge_resolver_observations(
    destination: &mut Vec<(PathBuf, StableFileIdentity)>,
    source: Vec<(PathBuf, initramfs::ResolverFileIdentity)>,
) -> Result<()> {
    for (path, identity) in source {
        let identity = StableFileIdentity::from_resolver(identity);
        if let Some((_, previous)) = destination.iter().find(|(seen, _)| *seen == path) {
            anyhow::ensure!(
                *previous == identity,
                "ELF source changed between dependency walks: {}",
                path.display()
            );
        } else {
            destination.push((path, identity));
        }
    }
    Ok(())
}

fn merge_search_observations(
    destination: &mut Vec<ClosureSearchRecord>,
    source: Vec<initramfs::ResolverPathObservation>,
) -> Result<()> {
    for observation in source {
        let identity = observation.identity.map(StableFileIdentity::from_resolver);
        if let Some(previous) = destination
            .iter()
            .find(|seen| seen.path == observation.path)
        {
            anyhow::ensure!(
                previous.identity == identity,
                "loader search path changed between dependency walks: {}",
                observation.path.display()
            );
        } else {
            destination.push(ClosureSearchRecord {
                path: observation.path,
                identity,
            });
        }
    }
    Ok(())
}

fn build_pinned_closure(
    root: &Path,
    binary: &PinnedInput,
    original_path: &Path,
    loader: &PinnedLoaderInputs,
) -> Result<(ClosureRecord, PinnedClosure)> {
    binary.verify_unchanged()?;
    loader.verify_unchanged()?;
    let cache_proc_path = loader.cache_proc_path();
    let mut result = initramfs::resolve_shared_libs_from_pinned(
        &binary.proc_path(),
        original_path,
        &loader.cwd,
        &loader.ld_library_path_dirs,
        cache_proc_path.as_deref(),
    )
    .with_context(|| {
        format!(
            "resolve pinned shared-library closure for {}",
            original_path.display()
        )
    })?;
    let mut observed_files = Vec::new();
    merge_resolver_observations(
        &mut observed_files,
        std::mem::take(&mut result.observed_files),
    )?;
    let mut search_paths = Vec::new();
    merge_search_observations(&mut search_paths, std::mem::take(&mut result.search_paths))?;

    let root_observation = observed_files
        .iter()
        .find(|(path, _)| path == original_path)
        .with_context(|| {
            format!(
                "resolver omitted root ELF observation for {}",
                original_path.display()
            )
        })?;
    anyhow::ensure!(
        root_observation.1 == binary.identity,
        "pinned ELF changed while resolving dependencies: {}",
        original_path.display()
    );
    anyhow::ensure!(
        result.missing.is_empty(),
        "{}: missing shared libraries: {}",
        original_path.display(),
        result
            .missing
            .iter()
            .map(|missing| missing.soname.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut resolved = result.found;
    if let Some(interpreter) = result.interpreter {
        let interpreter_path = PathBuf::from(&interpreter);
        let interpreter_input = pin_input(root, &interpreter_path).with_context(|| {
            format!(
                "pin PT_INTERP {} for {}",
                interpreter_path.display(),
                original_path.display()
            )
        })?;
        let canonical =
            std::fs::canonicalize(&interpreter_path).unwrap_or_else(|_| interpreter_path.clone());
        let canonical_string = canonical.to_string_lossy();
        let canonical_guest = canonical_string
            .strip_prefix('/')
            .unwrap_or(&canonical_string)
            .to_owned();
        resolved.push((canonical_guest.clone(), canonical.clone()));
        let original_guest = interpreter
            .strip_prefix('/')
            .unwrap_or(&interpreter)
            .to_owned();
        if original_guest != canonical_guest {
            resolved.push((original_guest, canonical.clone()));
        }

        let mut interpreter_result = initramfs::resolve_interpreter_deps_from_pinned(
            &interpreter_input.proc_path(),
            &interpreter_path,
            &loader.cwd,
            &loader.ld_library_path_dirs,
            cache_proc_path.as_deref(),
        )
        .with_context(|| {
            format!(
                "resolve pinned interpreter dependencies for {}",
                interpreter_path.display()
            )
        })?;
        anyhow::ensure!(
            interpreter_result.missing.is_empty(),
            "{}: missing interpreter shared libraries: {}",
            interpreter_path.display(),
            interpreter_result
                .missing
                .iter()
                .map(|missing| missing.soname.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        merge_resolver_observations(
            &mut observed_files,
            std::mem::take(&mut interpreter_result.observed_files),
        )?;
        merge_search_observations(
            &mut search_paths,
            std::mem::take(&mut interpreter_result.search_paths),
        )?;
        resolved.extend(interpreter_result.found);
        drop(interpreter_input);
    }

    resolved.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.as_os_str().cmp(right.1.as_os_str()))
    });
    resolved.dedup();
    let mut entries = Vec::with_capacity(resolved.len());
    let mut records = Vec::with_capacity(resolved.len());
    for (guest_path, host_path) in resolved {
        let input = pin_input(root, &host_path)
            .with_context(|| format!("pin resolved dependency {}", host_path.display()))?;
        if let Some((_, observed)) = observed_files.iter().find(|(path, _)| *path == host_path) {
            anyhow::ensure!(
                *observed == input.identity,
                "resolved dependency changed before pin: {}",
                host_path.display()
            );
        }
        records.push(ClosureEntryRecord {
            guest_path: guest_path.clone(),
            host_path,
            identity: input.identity,
            content_hash: input.content_hash,
        });
        entries.push(PinnedClosureEntry { guest_path, input });
    }

    for (path, observed) in &observed_files {
        if path == original_path {
            continue;
        }
        let current = path_identity(path)?;
        anyhow::ensure!(
            current == Some(*observed),
            "ELF source changed before closure publication: {}",
            path.display()
        );
    }
    for search in &search_paths {
        anyhow::ensure!(
            path_identity(&search.path)? == search.identity,
            "loader search path changed before closure publication: {}",
            search.path.display()
        );
    }
    binary.verify_unchanged()?;
    loader.verify_unchanged()?;

    let record = ClosureRecord {
        entries: records,
        search_paths,
    };
    Ok((record, PinnedClosure { entries }))
}

fn get_or_resolve_pinned_closure(
    root: &Path,
    binary: &PinnedInput,
    original_path: &Path,
    loader: &PinnedLoaderInputs,
) -> Result<PinnedClosure> {
    let key = closure_recipe_key(binary, original_path, loader);
    let record_path = root
        .join(PREPARED_CLOSURES_DIR)
        .join(format!("{key:016x}.closure"));
    if let Some(closure) = read_pinned_closure_record(&record_path, key)? {
        return Ok(closure);
    }
    let lock_path = root
        .join(PREPARED_LOCKS_DIR)
        .join(format!("closure-{key:016x}.lock"));
    loop {
        let coordination = open_coord_file(&lock_path)?;
        match rustix::fs::flock(
            &coordination,
            rustix::fs::FlockOperation::NonBlockingLockExclusive,
        ) {
            Ok(()) => {
                if let Some(closure) = read_pinned_closure_record(&record_path, key)? {
                    return Ok(closure);
                }
                let (record, closure) = build_pinned_closure(root, binary, original_path, loader)?;
                let bytes = encode_closure_record(&record, key)?;
                let mut temp = tempfile::Builder::new()
                    .prefix(&format!(".tmp-closure-{key:016x}-"))
                    .tempfile_in(root.join(PREPARED_CLOSURES_DIR))
                    .context("create loader closure temp")?;
                temp.write_all(&bytes).context("write loader closure")?;
                temp.as_file().sync_all().context("sync loader closure")?;
                temp.persist(&record_path)
                    .map_err(|error| error.error)
                    .with_context(|| format!("publish loader closure {}", record_path.display()))?;
                return Ok(closure);
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                rustix::fs::flock(&coordination, rustix::fs::FlockOperation::LockShared)
                    .with_context(|| {
                        format!("wait for loader closure {}", record_path.display())
                    })?;
                if let Some(closure) = read_pinned_closure_record(&record_path, key)? {
                    return Ok(closure);
                }
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("elect loader closure builder {}", record_path.display())
                });
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreparedObjectHeader {
    kind: PreparedObjectKind,
    compression: initramfs::InitrdCompression,
    key: u64,
    page_size: u64,
    payload_len: u64,
    payload_hash: u64,
    part_uncompressed_len: u64,
    part_compressed_len: u64,
    leading_pad: u64,
    stream_offset_mod: u64,
    reserved_shape: u64,
    parent_key: u64,
    reserved_len: u64,
}

impl PreparedObjectHeader {
    fn encode(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(PREPARED_CAS_HEADER_LEN);
        out.extend_from_slice(PREPARED_CAS_MAGIC);
        out.extend_from_slice(&PREPARED_CAS_SCHEMA.to_le_bytes());
        out.push(self.kind as u8);
        out.push(compression_tag(self.compression));
        out.extend_from_slice(&[0; 2]);
        out.extend_from_slice(&self.key.to_le_bytes());
        out.extend_from_slice(&self.page_size.to_le_bytes());
        out.extend_from_slice(&self.payload_len.to_le_bytes());
        out.extend_from_slice(&self.payload_hash.to_le_bytes());
        out.extend_from_slice(&self.part_uncompressed_len.to_le_bytes());
        out.extend_from_slice(&self.part_compressed_len.to_le_bytes());
        out.extend_from_slice(&self.leading_pad.to_le_bytes());
        out.extend_from_slice(&self.stream_offset_mod.to_le_bytes());
        out.extend_from_slice(&self.reserved_shape.to_le_bytes());
        out.extend_from_slice(&self.parent_key.to_le_bytes());
        out.extend_from_slice(&self.reserved_len.to_le_bytes());
        out.resize(PREPARED_CAS_HEADER_LEN, 0);
        let header_hash = ahash_bytes(&out);
        out[104..112].copy_from_slice(&header_hash.to_le_bytes());
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            bytes.len() == PREPARED_CAS_HEADER_LEN,
            "prepared object header is truncated"
        );
        anyhow::ensure!(
            &bytes[..8] == PREPARED_CAS_MAGIC,
            "prepared object magic mismatch"
        );
        let schema = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        anyhow::ensure!(
            schema == PREPARED_CAS_SCHEMA,
            "prepared object schema {schema} != {PREPARED_CAS_SCHEMA}"
        );
        anyhow::ensure!(
            bytes[14..16] == [0; 2],
            "prepared object reserved tag bytes changed"
        );
        anyhow::ensure!(
            bytes[112..].iter().all(|byte| *byte == 0),
            "prepared object reserved header bytes changed"
        );
        let recorded_header_hash = u64::from_le_bytes(bytes[104..112].try_into().unwrap());
        let mut canonical_header = bytes.to_vec();
        canonical_header[104..112].fill(0);
        anyhow::ensure!(
            ahash_bytes(&canonical_header) == recorded_header_hash,
            "prepared object header checksum mismatch"
        );
        Ok(Self {
            kind: PreparedObjectKind::from_tag(bytes[12])?,
            compression: compression_from_tag(bytes[13])?,
            key: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            page_size: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            payload_len: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            payload_hash: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            part_uncompressed_len: u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            part_compressed_len: u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
            leading_pad: u64::from_le_bytes(bytes[64..72].try_into().unwrap()),
            stream_offset_mod: u64::from_le_bytes(bytes[72..80].try_into().unwrap()),
            reserved_shape: u64::from_le_bytes(bytes[80..88].try_into().unwrap()),
            parent_key: u64::from_le_bytes(bytes[88..96].try_into().unwrap()),
            reserved_len: u64::from_le_bytes(bytes[96..104].try_into().unwrap()),
        })
    }

    fn validate_shape(self) -> Result<()> {
        let page = usize::try_from(self.page_size).context("page size exceeds usize")?;
        anyhow::ensure!(page.is_power_of_two(), "cached page size is invalid");
        anyhow::ensure!(
            page >= PREPARED_CAS_HEADER_LEN,
            "cached mapping granule is smaller than the header"
        );
        let leading =
            usize::try_from(self.leading_pad).context("part leading pad exceeds usize")?;
        let compressed =
            usize::try_from(self.part_compressed_len).context("part length exceeds usize")?;
        anyhow::ensure!(
            leading < page && self.stream_offset_mod == self.leading_pad,
            "cached part leading-pad geometry is inconsistent"
        );
        anyhow::ensure!(
            self.reserved_shape == 0 && self.reserved_len == 0,
            "cached prepared object reserved shape fields changed"
        );

        match self.kind {
            PreparedObjectKind::Base => {
                anyhow::ensure!(
                    leading == 0 && self.payload_len == self.part_compressed_len,
                    "base object layout is not an unshifted compressed part"
                );
                anyhow::ensure!(
                    self.parent_key == self.key,
                    "base object key linkage mismatch"
                );
                anyhow::ensure!(
                    self.part_uncompressed_len > 0 || self.payload_len > 0,
                    "base object has no source or compressed bytes"
                );
            }
            PreparedObjectKind::Payload
            | PreparedObjectKind::Modules
            | PreparedObjectKind::Tail
            | PreparedObjectKind::PayloadView
            | PreparedObjectKind::ModulesView => {
                let expected = leading
                    .checked_add(compressed)
                    .context("prepared part layout length overflow")?;
                anyhow::ensure!(
                    self.payload_len == expected as u64,
                    "prepared part layout length mismatch"
                );
            }
            PreparedObjectKind::Stitch => {
                anyhow::ensure!(
                    leading == 0
                        && self.payload_len == self.page_size
                        && self.part_compressed_len == self.page_size
                        && self.part_uncompressed_len == 0,
                    "stitch object must contain exactly one page"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct PreparedObjectExpectation {
    kind: PreparedObjectKind,
    compression: initramfs::InitrdCompression,
    key: u64,
    page_size: u64,
    leading_pad: Option<u64>,
    parent_key: Option<u64>,
}

impl PreparedObjectExpectation {
    fn validate(self, header: PreparedObjectHeader) -> Result<()> {
        anyhow::ensure!(header.kind == self.kind, "prepared object kind mismatch");
        anyhow::ensure!(
            header.compression == self.compression,
            "prepared object compression mismatch"
        );
        anyhow::ensure!(header.key == self.key, "prepared object key mismatch");
        anyhow::ensure!(
            header.page_size == self.page_size,
            "prepared object page-size mismatch"
        );
        if let Some(expected) = self.leading_pad {
            anyhow::ensure!(
                header.leading_pad == expected,
                "prepared object leading pad mismatch"
            );
        }
        if let Some(expected) = self.parent_key {
            anyhow::ensure!(
                header.parent_key == expected,
                "prepared object parent-key linkage mismatch"
            );
        }
        header.validate_shape()
    }
}

#[derive(Debug)]
struct PreparedObject {
    fd: Option<OwnedFd>,
    header: PreparedObjectHeader,
    data_offset: usize,
    cache_hit: bool,
}

impl PreparedObject {
    fn read_exact_at(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        let end = offset
            .checked_add(len)
            .context("prepared object read overflow")?;
        let payload_len =
            usize::try_from(self.header.payload_len).context("payload length exceeds usize")?;
        anyhow::ensure!(end <= payload_len, "prepared object read exceeds payload");
        let file_offset = self
            .data_offset
            .checked_add(offset)
            .context("prepared object file offset overflow")?;
        let mut out = vec![0u8; len];
        let mut done = 0usize;
        while done < len {
            let absolute = file_offset
                .checked_add(done)
                .context("prepared object pread offset overflow")?;
            let absolute = libc::off_t::try_from(absolute).context("pread offset exceeds off_t")?;
            let read = unsafe {
                libc::pread(
                    self.fd
                        .as_ref()
                        .context("prepared object fd already consumed")?
                        .as_raw_fd(),
                    out[done..].as_mut_ptr().cast(),
                    len - done,
                    absolute,
                )
            };
            if read < 0 {
                return Err(std::io::Error::last_os_error()).context("pread prepared object");
            }
            anyhow::ensure!(read != 0, "prepared object truncated during pread");
            done += read as usize;
        }
        Ok(out)
    }
}

fn prepared_object_path(root: &Path, kind: PreparedObjectKind, key: u64) -> PathBuf {
    root.join(PREPARED_OBJECTS_DIR)
        .join(format!("{}-{key:016x}.bin", kind.stem()))
}

fn prepared_object_lock_path(root: &Path, kind: PreparedObjectKind, key: u64) -> PathBuf {
    root.join(PREPARED_LOCKS_DIR)
        .join(format!("{}-{key:016x}.lock", kind.stem()))
}

fn read_header_at(file: &File) -> Result<PreparedObjectHeader> {
    let mut bytes = [0u8; PREPARED_CAS_HEADER_LEN];
    let mut done = 0usize;
    while done < bytes.len() {
        let read = unsafe {
            libc::pread(
                file.as_raw_fd(),
                bytes[done..].as_mut_ptr().cast(),
                bytes.len() - done,
                done as libc::off_t,
            )
        };
        if read < 0 {
            return Err(std::io::Error::last_os_error()).context("pread prepared object header");
        }
        anyhow::ensure!(read != 0, "prepared object header is truncated");
        done += read as usize;
    }
    PreparedObjectHeader::decode(&bytes)
}

fn validate_open_prepared_object(
    path: &Path,
    file: File,
    expected: PreparedObjectExpectation,
) -> Result<PreparedObject> {
    rustix::fs::flock(&file, rustix::fs::FlockOperation::LockShared)
        .with_context(|| format!("lock prepared initrd object {}", path.display()))?;
    let header = read_header_at(&file)?;
    expected.validate(header)?;
    let data_offset =
        usize::try_from(header.page_size).context("prepared object data offset exceeds usize")?;
    anyhow::ensure!(
        data_offset >= PREPARED_CAS_HEADER_LEN,
        "host page is too small for prepared object header"
    );
    let payload_len =
        usize::try_from(header.payload_len).context("prepared payload length exceeds usize")?;
    let padded_payload_len = round_up(payload_len, data_offset)?;
    let expected_file_len = data_offset
        .checked_add(padded_payload_len)
        .context("prepared object file length overflow")?;
    let actual_file_len =
        usize::try_from(file.metadata()?.len()).context("prepared object file too large")?;
    anyhow::ensure!(
        actual_file_len == expected_file_len,
        "prepared object file length {actual_file_len} != {expected_file_len}"
    );
    Ok(PreparedObject {
        fd: Some(file.into()),
        header,
        data_offset,
        cache_hit: true,
    })
}

fn try_open_prepared_object(
    path: &Path,
    expected: PreparedObjectExpectation,
) -> Result<Option<PreparedObject>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("open prepared initrd object {}", path.display()));
        }
    };
    validate_open_prepared_object(path, file, expected).map(Some)
}

#[derive(Debug)]
struct BuiltPreparedObject {
    header: PreparedObjectHeader,
    payload: Vec<u8>,
}

fn publish_prepared_object(
    root: &Path,
    final_path: &Path,
    built: &BuiltPreparedObject,
) -> Result<()> {
    built.header.validate_shape()?;
    anyhow::ensure!(
        built.payload.len() == usize::try_from(built.header.payload_len)?,
        "prepared object builder payload length mismatch"
    );
    anyhow::ensure!(
        ahash_bytes(&built.payload) == built.header.payload_hash,
        "prepared object builder payload digest mismatch"
    );
    let page = usize::try_from(built.header.page_size).context("page size exceeds usize")?;
    anyhow::ensure!(
        page >= PREPARED_CAS_HEADER_LEN && page.is_power_of_two(),
        "invalid prepared object data alignment"
    );
    let padded_payload_len = round_up(built.payload.len(), page)?;
    let file_len = page
        .checked_add(padded_payload_len)
        .context("prepared object file length overflow")?;

    let objects_dir = root.join(PREPARED_OBJECTS_DIR);
    let mut temp = tempfile::Builder::new()
        .prefix(&format!(
            ".tmp-object-{}-{:016x}-",
            built.header.kind.stem(),
            built.header.key
        ))
        .tempfile_in(&objects_dir)
        .with_context(|| format!("create prepared object temp in {}", objects_dir.display()))?;
    temp.as_file_mut()
        .set_len(u64::try_from(file_len)?)
        .context("size prepared object temp")?;
    temp.as_file_mut()
        .seek(SeekFrom::Start(0))
        .context("seek prepared object header")?;
    temp.write_all(&built.header.encode())
        .context("write prepared object header")?;
    temp.as_file_mut()
        .seek(SeekFrom::Start(built.header.page_size))
        .context("seek prepared object payload")?;
    temp.write_all(&built.payload)
        .context("write prepared object payload")?;
    temp.as_file()
        .sync_all()
        .context("sync prepared object temp")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        temp.as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o444))
            .context("mark prepared object read-only")?;
    }
    temp.persist(final_path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish prepared object {}", final_path.display()))?;
    if let Err(error) = crate::cache::fsync_parent(final_path) {
        tracing::warn!(
            path = %final_path.display(),
            %error,
            "prepared initrd CAS parent fsync failed; validation remains fail-closed"
        );
    }
    Ok(())
}

fn get_or_build_prepared_object<F>(
    root: &Path,
    expected: PreparedObjectExpectation,
    build: F,
) -> Result<PreparedObject>
where
    F: FnOnce() -> Result<BuiltPreparedObject>,
{
    let final_path = prepared_object_path(root, expected.kind, expected.key);
    if let Some(opened) = try_open_prepared_object(&final_path, expected).with_context(|| {
        format!(
            "prepared initrd CAS object is corrupt (refusing copy fallback): {}",
            final_path.display()
        )
    })? {
        return Ok(opened);
    }

    let lock_path = prepared_object_lock_path(root, expected.kind, expected.key);
    let mut build = Some(build);
    loop {
        let coordination = open_coord_file(&lock_path)?;
        match rustix::fs::flock(
            &coordination,
            rustix::fs::FlockOperation::NonBlockingLockExclusive,
        ) {
            Ok(()) => {
                if let Some(opened) =
                    try_open_prepared_object(&final_path, expected).with_context(|| {
                        format!(
                            "prepared initrd CAS object is corrupt (refusing copy fallback): {}",
                            final_path.display()
                        )
                    })?
                {
                    return Ok(opened);
                }

                let built = build
                    .take()
                    .context("prepared object builder closure was already consumed")?(
                )?;
                anyhow::ensure!(
                    built.header.kind == expected.kind
                        && built.header.compression == expected.compression
                        && built.header.key == expected.key
                        && built.header.page_size == expected.page_size,
                    "prepared object builder returned a different recipe"
                );
                expected.validate(built.header)?;
                publish_prepared_object(root, &final_path, &built)?;
                let mut opened = try_open_prepared_object(&final_path, expected)?
                    .context("published prepared object disappeared before open")?;
                opened.cache_hit = false;
                // `opened` takes LOCK_SH on the immutable object before the
                // coordination EX lock drops. GC takes the same per-key lock
                // and then object LOCK_EX, closing both publish/open and
                // live-mapping unlink races.
                return Ok(opened);
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                rustix::fs::flock(&coordination, rustix::fs::FlockOperation::LockShared)
                    .with_context(|| {
                        format!("wait for prepared initrd object {}", final_path.display())
                    })?;
                if let Some(opened) =
                    try_open_prepared_object(&final_path, expected).with_context(|| {
                        format!(
                            "prepared initrd CAS object is corrupt (refusing copy fallback): {}",
                            final_path.display()
                        )
                    })?
                {
                    return Ok(opened);
                }
                // A killed/failed winner dropped EX without publishing.
                // Release our shared lock by dropping `coordination`, then
                // race to elect exactly one successor.
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "elect prepared initrd object builder {}",
                        final_path.display()
                    )
                });
            }
        }
    }
}

fn read_gc_stamp(path: &Path) -> Option<u64> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != 8 {
        return None;
    }
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[derive(Debug)]
struct GcCandidate {
    path: PathBuf,
    kind: PreparedObjectKind,
    key: u64,
    modified_secs: u64,
    len: u64,
}

fn parse_object_filename(name: &str) -> Option<(PreparedObjectKind, u64)> {
    let stem = name.strip_suffix(".bin")?;
    for kind in [
        PreparedObjectKind::Base,
        PreparedObjectKind::Payload,
        PreparedObjectKind::Modules,
        PreparedObjectKind::Tail,
        PreparedObjectKind::PayloadView,
        PreparedObjectKind::ModulesView,
        PreparedObjectKind::Stitch,
    ] {
        if let Some(key) = stem.strip_prefix(&format!("{}-", kind.stem()))
            && key.len() == 16
        {
            return u64::from_str_radix(key, 16).ok().map(|key| (kind, key));
        }
    }
    None
}

fn parse_temp_object_filename(name: &str) -> Option<(PreparedObjectKind, u64)> {
    let stem = name.strip_prefix(".tmp-object-")?;
    for kind in [
        PreparedObjectKind::Base,
        PreparedObjectKind::Payload,
        PreparedObjectKind::Modules,
        PreparedObjectKind::Tail,
        PreparedObjectKind::PayloadView,
        PreparedObjectKind::ModulesView,
        PreparedObjectKind::Stitch,
    ] {
        let Some(key_and_random) = stem.strip_prefix(&format!("{}-", kind.stem())) else {
            continue;
        };
        let (key, random) = key_and_random.split_once('-')?;
        if key.len() == 16 && !random.is_empty() {
            return u64::from_str_radix(key, 16).ok().map(|key| (kind, key));
        }
    }
    None
}

fn parse_temp_digest_filename(name: &str) -> Option<u64> {
    let key_and_random = name.strip_prefix(".tmp-digest-")?;
    let (key, random) = key_and_random.split_once('-')?;
    if key.len() != 16 || random.is_empty() {
        return None;
    }
    u64::from_str_radix(key, 16).ok()
}

fn try_collect_temp(path: &Path, lock_path: &Path) -> Result<bool> {
    let coordination = open_coord_file(lock_path)?;
    if rustix::fs::flock(
        &coordination,
        rustix::fs::FlockOperation::NonBlockingLockExclusive,
    )
    .is_err()
    {
        return Ok(false);
    }
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(error) => {
            Err(error).with_context(|| format!("remove stale prepared temp {}", path.display()))
        }
    }
}

fn try_collect_object(root: &Path, candidate: &GcCandidate) -> Result<bool> {
    let lock_path = prepared_object_lock_path(root, candidate.kind, candidate.key);
    let coord = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)?;
    if rustix::fs::flock(&coord, rustix::fs::FlockOperation::NonBlockingLockExclusive).is_err() {
        return Ok(false);
    }
    let object = match File::open(&candidate.path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(error) => return Err(error.into()),
    };
    if rustix::fs::flock(
        &object,
        rustix::fs::FlockOperation::NonBlockingLockExclusive,
    )
    .is_err()
    {
        return Ok(false);
    }
    std::fs::remove_file(&candidate.path)
        .with_context(|| format!("remove stale prepared object {}", candidate.path.display()))?;
    Ok(true)
}

fn run_prepared_cache_gc(root: &Path, now: u64) -> Result<()> {
    let mut candidates = Vec::new();
    let mut total = 0u64;
    for entry in std::fs::read_dir(root.join(PREPARED_OBJECTS_DIR))? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let metadata = entry.metadata()?;
        if let Some((kind, key)) = parse_temp_object_filename(&name) {
            let modified_secs = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs())
                .unwrap_or(0);
            if now.saturating_sub(modified_secs) > PREPARED_CAS_GC_INTERVAL_SECS {
                let lock_path = prepared_object_lock_path(root, kind, key);
                let _ = try_collect_temp(&entry.path(), &lock_path);
            } else {
                total = total.saturating_add(metadata.len());
            }
            continue;
        }
        let Some((kind, key)) = parse_object_filename(&name) else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let modified_secs = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        total = total.saturating_add(metadata.len());
        candidates.push(GcCandidate {
            path: entry.path(),
            kind,
            key,
            modified_secs,
            len: metadata.len(),
        });
    }
    candidates.sort_by_key(|candidate| candidate.modified_secs);
    for candidate in &candidates {
        let too_old = now.saturating_sub(candidate.modified_secs) > PREPARED_CAS_MAX_AGE_SECS;
        let over_size = total > PREPARED_CAS_MAX_BYTES;
        if !too_old && !over_size {
            continue;
        }
        if try_collect_object(root, candidate)? {
            total = total.saturating_sub(candidate.len);
        }
    }

    for entry in std::fs::read_dir(root.join(PREPARED_DIGESTS_DIR))? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let metadata = entry.metadata()?;
        let modified_secs = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        if let Some(key) = parse_temp_digest_filename(&name) {
            if now.saturating_sub(modified_secs) > PREPARED_CAS_GC_INTERVAL_SECS {
                let lock_path = root
                    .join(PREPARED_LOCKS_DIR)
                    .join(format!("digest-{key:016x}.lock"));
                let _ = try_collect_temp(&entry.path(), &lock_path);
            }
            continue;
        }
        let Some(key) = name
            .strip_suffix(".digest")
            .filter(|key| key.len() == 16)
            .and_then(|key| u64::from_str_radix(key, 16).ok())
        else {
            continue;
        };
        if now.saturating_sub(modified_secs) <= PREPARED_CAS_MAX_AGE_SECS {
            continue;
        }
        let lock_path = root
            .join(PREPARED_LOCKS_DIR)
            .join(format!("digest-{key:016x}.lock"));
        let Ok(lock) = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)
        else {
            continue;
        };
        if rustix::fs::flock(&lock, rustix::fs::FlockOperation::NonBlockingLockExclusive).is_ok() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    Ok(())
}

fn maybe_gc_prepared_cache(root: &Path) {
    let now = unix_now_secs();
    let gc_lock_path = root.join(PREPARED_LOCKS_DIR).join("gc.lock");
    let Ok(gc_lock) = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&gc_lock_path)
    else {
        return;
    };
    if rustix::fs::flock(
        &gc_lock,
        rustix::fs::FlockOperation::NonBlockingLockExclusive,
    )
    .is_err()
    {
        return;
    }
    let stamp_path = root.join(PREPARED_GC_STAMP);
    if read_gc_stamp(&stamp_path)
        .is_some_and(|last| now.saturating_sub(last) < PREPARED_CAS_GC_INTERVAL_SECS)
    {
        return;
    }
    if let Err(error) = run_prepared_cache_gc(root, now) {
        tracing::warn!(%error, "prepared initrd CAS GC failed");
    }
    if let Err(error) = std::fs::write(&stamp_path, now.to_le_bytes()) {
        tracing::warn!(%error, "prepared initrd CAS GC stamp write failed");
    }
}

#[derive(Debug)]
struct PinnedModule {
    archive_name: String,
    input: PinnedInput,
}

#[derive(Debug)]
struct PinnedSuffixInputs {
    payload: Option<PinnedInput>,
    modules: Vec<PinnedModule>,
}

impl PinnedSuffixInputs {
    fn pin(root: &Path, params: &initramfs::SuffixParams<'_>) -> Result<Self> {
        let payload = params
            .payload
            .map(|path| pin_input(root, path))
            .transpose()?;
        let mut modules = Vec::with_capacity(params.kernel_modules.len());
        for path in params.kernel_modules {
            let archive_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .with_context(|| {
                    format!(
                        "kernel module path has no valid filename: {}",
                        path.display()
                    )
                })?
                .to_owned();
            modules.push(PinnedModule {
                archive_name,
                input: pin_input(root, path)?,
            });
        }
        Ok(Self { payload, modules })
    }

    fn verify_unchanged(&self) -> Result<()> {
        if let Some(payload) = &self.payload {
            payload.verify_unchanged()?;
        }
        for module in &self.modules {
            module.input.verify_unchanged()?;
        }
        Ok(())
    }

    fn module_sources(&self) -> Vec<(String, PathBuf)> {
        self.modules
            .iter()
            .map(|module| (module.archive_name.clone(), module.input.proc_path()))
            .collect()
    }
}

fn recipe_prefix(
    domain: &[u8],
    page_size: usize,
    compression: initramfs::InitrdCompression,
) -> AHasher {
    let mut hasher = fixed_hasher();
    hash_len_prefixed(&mut hasher, domain);
    hash_u32(&mut hasher, PREPARED_CAS_SCHEMA);
    hash_len_prefixed(&mut hasher, std::env::consts::ARCH.as_bytes());
    hash_u64(&mut hasher, page_size as u64);
    hash_u8(&mut hasher, compression_tag(compression));
    hasher
}

fn base_recipe_key(
    base_key: &BaseKey,
    page_size: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(b"ktstr-prepared-initrd-base", page_size, compression);
    hash_u64(&mut hasher, base_key.0);
    hasher.finish()
}

fn payload_recipe_key(
    payload: &PinnedInput,
    coverage: (bool, u64),
    page_size: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(b"ktstr-prepared-initrd-payload", page_size, compression);
    hash_u64(&mut hasher, payload.identity.size);
    hash_u64(&mut hasher, payload.content_hash);
    hash_u8(&mut hasher, u8::from(coverage.0));
    hash_u64(&mut hasher, coverage.1);
    hasher.finish()
}

fn modules_recipe_key(
    modules: &[PinnedModule],
    page_size: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(b"ktstr-prepared-initrd-modules", page_size, compression);
    hash_u64(&mut hasher, modules.len() as u64);
    for module in modules {
        hash_len_prefixed(&mut hasher, module.archive_name.as_bytes());
        hash_u64(&mut hasher, module.input.identity.size);
        hash_u64(&mut hasher, module.input.content_hash);
    }
    hasher.finish()
}

fn part_view_recipe_key(
    canonical_key: u64,
    kind: PreparedObjectKind,
    leading_pad: usize,
    page_size: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(b"ktstr-prepared-initrd-part-view", page_size, compression);
    hash_u8(&mut hasher, kind as u8);
    hash_u64(&mut hasher, canonical_key);
    hash_u64(&mut hasher, leading_pad as u64);
    hasher.finish()
}

fn tail_recipe_key(
    prefix_uncompressed_len: usize,
    leading_pad: usize,
    page_size: usize,
    compression: initramfs::InitrdCompression,
    params: &initramfs::SuffixParams<'_>,
) -> u64 {
    let mut hasher = recipe_prefix(b"ktstr-prepared-initrd-tail", page_size, compression);
    hash_u64(&mut hasher, prefix_uncompressed_len as u64);
    hash_u64(&mut hasher, leading_pad as u64);
    hash_string_slice(&mut hasher, params.args);
    hash_string_slice(&mut hasher, params.sched_args);
    hash_string_slice(&mut hasher, params.sched_enable);
    hash_string_slice(&mut hasher, params.sched_disable);
    hash_optional_str(&mut hasher, params.exec_cmd);
    hash_u64(&mut hasher, params.staged_sched_args.len() as u64);
    for (name, args) in params.staged_sched_args {
        hash_len_prefixed(&mut hasher, name.as_bytes());
        hash_string_slice(&mut hasher, args);
    }
    hash_optional_str(&mut hasher, params.workload_root_cgroup);
    hash_optional_str(&mut hasher, params.scheduler_cgroup_parent);
    hasher.finish()
}

#[derive(Clone, Copy, Debug)]
struct PageSegment {
    part_index: usize,
    part_offset: usize,
    page_offset: usize,
    len: usize,
}

fn stitch_recipe_key(
    segments: &[PageSegment],
    parts: &[PreparedPart],
    page_size: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(b"ktstr-prepared-initrd-stitch", page_size, compression);
    hash_u64(&mut hasher, segments.len() as u64);
    for segment in segments {
        hash_u64(&mut hasher, parts[segment.part_index].object.header.key);
        hash_u64(&mut hasher, segment.part_offset as u64);
        hash_u64(&mut hasher, segment.page_offset as u64);
        hash_u64(&mut hasher, segment.len as u64);
    }
    hasher.finish()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PreparedRangePlan {
    pub(crate) part_count: usize,
    pub(crate) direct_ranges: usize,
    pub(crate) stitch_pages: usize,
    pub(crate) total_compressed_len: usize,
}

#[derive(Debug)]
pub(crate) struct PreparedMapping {
    pub(crate) fd: OwnedFd,
    pub(crate) file_offset: u64,
    pub(crate) guest_offset: u64,
    pub(crate) map_len: usize,
}

#[derive(Debug)]
pub(crate) struct PreparedInitrd {
    uncompressed_len: usize,
    compressed_len: usize,
    ranges: Vec<PreparedMapping>,
    cache_hits: usize,
    plan: PreparedRangePlan,
    mapping_granule: usize,
    coverage_instrumented: bool,
    coverage_reserve_bytes: u64,
}

impl PreparedInitrd {
    pub(crate) fn uncompressed_len(&self) -> usize {
        self.uncompressed_len
    }

    pub(crate) fn compressed_len(&self) -> usize {
        self.compressed_len
    }

    pub(crate) fn cache_hits(&self) -> usize {
        self.cache_hits
    }

    pub(crate) fn plan(&self) -> PreparedRangePlan {
        self.plan
    }

    pub(crate) fn mapping_granule(&self) -> usize {
        self.mapping_granule
    }

    pub(crate) fn coverage(&self) -> (bool, u64) {
        (self.coverage_instrumented, self.coverage_reserve_bytes)
    }

    pub(crate) fn into_ranges(self) -> Vec<PreparedMapping> {
        self.ranges
    }
}

#[derive(Debug)]
pub(crate) struct PreparedBase {
    root: PathBuf,
    object: PreparedObject,
    page_size: usize,
    compression: initramfs::InitrdCompression,
}

#[derive(Debug)]
struct PinnedArchiveInput {
    archive_name: String,
    input: PinnedInput,
    mode: u32,
}

/// Exact, open-file description backed inputs for the stable base archive.
///
/// Preparation pins every explicit source and every resolved loader
/// dependency before deriving [`BaseKey`]. The elected CAS builder reads only
/// `/proc/<this-pid>/fd/N` paths owned by this value, so pathname replacement
/// cannot make keying and archive construction observe different revisions.
#[derive(Debug)]
pub(crate) struct PreparedBaseInputs {
    key: BaseKey,
    payload: PinnedInput,
    extras: Vec<PinnedArchiveInput>,
    includes: Vec<PinnedArchiveInput>,
    shared_libs: Vec<PinnedClosureEntry>,
    busybox_bytes: Option<Vec<u8>>,
}

impl PreparedBaseInputs {
    pub(crate) fn key(&self) -> &BaseKey {
        &self.key
    }

    fn verify_unchanged(&self) -> Result<()> {
        self.payload.verify_unchanged()?;
        for entry in self.extras.iter().chain(&self.includes) {
            entry.input.verify_unchanged()?;
        }
        for dependency in &self.shared_libs {
            dependency.input.verify_unchanged()?;
        }
        Ok(())
    }

    pub(crate) fn build(&self) -> Result<Vec<u8>> {
        self.verify_unchanged()?;
        let extra_paths: Vec<PathBuf> = self
            .extras
            .iter()
            .map(|entry| entry.input.proc_path())
            .collect();
        let extras: Vec<(&str, &Path)> = self
            .extras
            .iter()
            .zip(&extra_paths)
            .map(|(entry, path)| (entry.archive_name.as_str(), path.as_path()))
            .collect();
        let include_paths: Vec<PathBuf> = self
            .includes
            .iter()
            .map(|entry| entry.input.proc_path())
            .collect();
        let includes: Vec<(&str, &Path)> = self
            .includes
            .iter()
            .zip(&include_paths)
            .map(|(entry, path)| (entry.archive_name.as_str(), path.as_path()))
            .collect();
        let shared_paths: Vec<PathBuf> = self
            .shared_libs
            .iter()
            .map(|entry| entry.input.proc_path())
            .collect();
        let shared_libs: Vec<(String, PathBuf, u64, u64)> = self
            .shared_libs
            .iter()
            .zip(shared_paths)
            .map(|(entry, path)| {
                (
                    entry.guest_path.clone(),
                    path,
                    entry.input.identity.size,
                    entry.input.content_hash,
                )
            })
            .collect();
        let archive = initramfs::build_initramfs_base_from_resolved(
            &extras,
            &includes,
            self.busybox_bytes.as_deref(),
            &shared_libs,
        )?;
        self.verify_unchanged()?;
        Ok(archive)
    }
}

fn merge_pinned_closure_entries(
    destination: &mut Vec<PinnedClosureEntry>,
    closure: PinnedClosure,
) -> Result<()> {
    for entry in closure.entries {
        if let Some(previous) = destination
            .iter()
            .find(|previous| previous.guest_path == entry.guest_path)
        {
            anyhow::ensure!(
                previous.input.content_hash == entry.input.content_hash
                    && previous.input.identity.size == entry.input.identity.size,
                "loader closures resolve guest path '{}' to different contents: {} vs {}",
                entry.guest_path,
                previous.input.display_path.display(),
                entry.input.display_path.display()
            );
            continue;
        }
        destination.push(entry);
    }
    Ok(())
}

fn prepared_base_semantic_key(
    extras: &[PinnedArchiveInput],
    includes: &[PinnedArchiveInput],
    shared_libs: &[PinnedClosureEntry],
    busybox_bytes: Option<&[u8]>,
) -> BaseKey {
    let mut hasher = fixed_hasher();
    hash_len_prefixed(&mut hasher, b"ktstr-prepared-base");
    hash_u32(&mut hasher, PREPARED_CAS_SCHEMA);
    hash_len_prefixed(&mut hasher, std::env::consts::ARCH.as_bytes());
    match busybox_bytes {
        Some(bytes) => {
            hash_u8(&mut hasher, 1);
            hash_u64(&mut hasher, ahash_bytes(bytes));
            hash_u64(&mut hasher, bytes.len() as u64);
        }
        None => hash_u8(&mut hasher, 0),
    }

    let mut extra_order: Vec<_> = extras.iter().collect();
    extra_order.sort_by(|left, right| left.archive_name.cmp(&right.archive_name));
    hash_u64(&mut hasher, extra_order.len() as u64);
    for entry in extra_order {
        hash_len_prefixed(&mut hasher, entry.archive_name.as_bytes());
        hash_u64(&mut hasher, entry.input.identity.size);
        hash_u64(&mut hasher, entry.input.content_hash);
    }

    let mut include_order: Vec<_> = includes.iter().collect();
    include_order.sort_by(|left, right| left.archive_name.cmp(&right.archive_name));
    hash_u64(&mut hasher, include_order.len() as u64);
    for entry in include_order {
        hash_len_prefixed(&mut hasher, entry.archive_name.as_bytes());
        hash_u32(&mut hasher, entry.mode);
        hash_u64(&mut hasher, entry.input.identity.size);
        hash_u64(&mut hasher, entry.input.content_hash);
    }

    let mut dependency_order: Vec<_> = shared_libs.iter().collect();
    dependency_order.sort_by(|left, right| left.guest_path.cmp(&right.guest_path));
    hash_u64(&mut hasher, dependency_order.len() as u64);
    for dependency in dependency_order {
        hash_len_prefixed(&mut hasher, dependency.guest_path.as_bytes());
        hash_u64(&mut hasher, dependency.input.identity.size);
        hash_u64(&mut hasher, dependency.input.content_hash);
    }
    BaseKey(hasher.finish())
}

pub(crate) fn prepare_base_inputs(
    payload_path: &Path,
    extra_binaries: &[(&str, &Path)],
    include_files: &[(String, PathBuf)],
    busybox_bytes: Option<&[u8]>,
) -> Result<PreparedBaseInputs> {
    let root = prepared_cache_root()?;
    ensure_prepared_cache_dirs(&root)?;
    maybe_gc_prepared_cache(&root);
    let loader = PinnedLoaderInputs::pin(&root)?;
    let payload = pin_input(&root, payload_path)?;
    let mut extras = Vec::with_capacity(extra_binaries.len());
    for (archive_name, path) in extra_binaries {
        extras.push(PinnedArchiveInput {
            archive_name: (*archive_name).to_owned(),
            input: pin_input(&root, path)?,
            mode: 0o100755,
        });
    }
    extras.sort_by(|left, right| left.archive_name.cmp(&right.archive_name));
    anyhow::ensure!(
        extras
            .windows(2)
            .all(|pair| pair[0].archive_name != pair[1].archive_name),
        "duplicate extra-binary archive path"
    );
    let mut includes = Vec::with_capacity(include_files.len());
    for (archive_name, path) in include_files {
        let input = pin_input(&root, path)?;
        let mode = input
            .file
            .metadata()
            .with_context(|| format!("stat include file {}", path.display()))?
            .permissions()
            .mode();
        includes.push(PinnedArchiveInput {
            archive_name: archive_name.clone(),
            input,
            mode,
        });
    }
    includes.sort_by(|left, right| left.archive_name.cmp(&right.archive_name));
    anyhow::ensure!(
        includes
            .windows(2)
            .all(|pair| pair[0].archive_name != pair[1].archive_name),
        "duplicate include-file archive path"
    );
    for extra in &extras {
        anyhow::ensure!(
            !includes
                .iter()
                .any(|include| include.archive_name == extra.archive_name),
            "archive path '{}' is used by both an extra binary and include file",
            extra.archive_name
        );
    }

    let mut shared_libs = Vec::new();
    let payload_closure = get_or_resolve_pinned_closure(&root, &payload, payload_path, &loader)?;
    merge_pinned_closure_entries(&mut shared_libs, payload_closure)?;
    for entry in extras.iter().chain(&includes) {
        if !entry.input.is_elf()? {
            continue;
        }
        let closure =
            get_or_resolve_pinned_closure(&root, &entry.input, &entry.input.display_path, &loader)?;
        merge_pinned_closure_entries(&mut shared_libs, closure)?;
    }
    shared_libs.sort_by(|left, right| left.guest_path.cmp(&right.guest_path));

    loader.verify_unchanged()?;
    let key = prepared_base_semantic_key(&extras, &includes, &shared_libs, busybox_bytes);
    let prepared = PreparedBaseInputs {
        key,
        payload,
        extras,
        includes,
        shared_libs,
        busybox_bytes: busybox_bytes.map(ToOwned::to_owned),
    };
    prepared.verify_unchanged()?;
    Ok(prepared)
}

#[derive(Debug)]
struct PreparedPart {
    object: PreparedObject,
    stream_offset: usize,
    uncompressed_len: usize,
    compressed_len: usize,
}

fn get_or_build_part<F>(
    root: &Path,
    kind: PreparedObjectKind,
    key: u64,
    leading_pad: usize,
    page_size: usize,
    compression: initramfs::InitrdCompression,
    build: F,
) -> Result<PreparedObject>
where
    F: FnOnce() -> Result<Vec<u8>>,
{
    anyhow::ensure!(
        kind != PreparedObjectKind::Base && kind != PreparedObjectKind::Stitch,
        "generic archive part has an invalid kind"
    );
    let expectation = PreparedObjectExpectation {
        kind,
        compression,
        key,
        page_size: page_size as u64,
        leading_pad: Some(leading_pad as u64),
        parent_key: Some(key),
    };
    get_or_build_prepared_object(root, expectation, || {
        let uncompressed = build()?;
        let compressed = initramfs::compress_initrd_part(compression, &uncompressed)?;
        let mut layout = Vec::with_capacity(leading_pad + compressed.len());
        layout.resize(leading_pad, 0);
        layout.extend_from_slice(&compressed);
        let header = PreparedObjectHeader {
            kind,
            compression,
            key,
            page_size: page_size as u64,
            payload_len: layout.len() as u64,
            payload_hash: ahash_bytes(&layout),
            part_uncompressed_len: uncompressed.len() as u64,
            part_compressed_len: compressed.len() as u64,
            leading_pad: leading_pad as u64,
            stream_offset_mod: leading_pad as u64,
            reserved_shape: 0,
            parent_key: key,
            reserved_len: 0,
        };
        Ok(BuiltPreparedObject {
            header,
            payload: layout,
        })
    })
}

fn get_or_build_part_view(
    root: &Path,
    canonical: &PreparedObject,
    view_kind: PreparedObjectKind,
    leading_pad: usize,
    page_size: usize,
    compression: initramfs::InitrdCompression,
) -> Result<PreparedObject> {
    anyhow::ensure!(
        matches!(
            view_kind,
            PreparedObjectKind::PayloadView | PreparedObjectKind::ModulesView
        ) && leading_pad > 0
            && leading_pad < page_size,
        "invalid shifted prepared-part view geometry"
    );
    anyhow::ensure!(
        canonical.header.leading_pad == 0,
        "shifted prepared-part view source is not canonical"
    );
    let key = part_view_recipe_key(
        canonical.header.key,
        view_kind,
        leading_pad,
        page_size,
        compression,
    );
    let expectation = PreparedObjectExpectation {
        kind: view_kind,
        compression,
        key,
        page_size: page_size as u64,
        leading_pad: Some(leading_pad as u64),
        parent_key: Some(canonical.header.key),
    };
    get_or_build_prepared_object(root, expectation, || {
        let compressed_len = usize::try_from(canonical.header.part_compressed_len)?;
        let compressed = canonical.read_exact_at(0, compressed_len)?;
        let mut layout = Vec::with_capacity(leading_pad + compressed.len());
        layout.resize(leading_pad, 0);
        layout.extend_from_slice(&compressed);
        let header = PreparedObjectHeader {
            kind: view_kind,
            compression,
            key,
            page_size: page_size as u64,
            payload_len: layout.len() as u64,
            payload_hash: ahash_bytes(&layout),
            part_uncompressed_len: canonical.header.part_uncompressed_len,
            part_compressed_len: canonical.header.part_compressed_len,
            leading_pad: leading_pad as u64,
            stream_offset_mod: leading_pad as u64,
            reserved_shape: 0,
            parent_key: canonical.header.key,
            reserved_len: 0,
        };
        Ok(BuiltPreparedObject {
            header,
            payload: layout,
        })
    })
}

/// Open or build the immutable compressed-base object.
///
/// The semantic [`BaseKey`] is available before any cpio bytes exist, so it
/// elects one cross-process builder. Cache-hit cells never materialize the
/// uncompressed base at all; the closure runs only in the elected winner.
pub(crate) fn get_or_prepare_base<F>(
    semantic_base_key: &BaseKey,
    compression: initramfs::InitrdCompression,
    build: F,
) -> Result<PreparedBase>
where
    F: FnOnce() -> Result<Vec<u8>>,
{
    let root = prepared_cache_root()?;
    ensure_prepared_cache_dirs(&root)?;
    maybe_gc_prepared_cache(&root);
    let page_size = PREPARED_MAPPING_GRANULE;
    let base_key = base_recipe_key(semantic_base_key, page_size, compression);
    let expectation = PreparedObjectExpectation {
        kind: PreparedObjectKind::Base,
        compression,
        key: base_key,
        page_size: page_size as u64,
        leading_pad: Some(0),
        parent_key: Some(base_key),
    };
    let object = get_or_build_prepared_object(&root, expectation, || {
        let uncompressed = build()?;
        let compressed = initramfs::compress_initrd_part(compression, &uncompressed)?;
        let header = PreparedObjectHeader {
            kind: PreparedObjectKind::Base,
            compression,
            key: base_key,
            page_size: page_size as u64,
            payload_len: compressed.len() as u64,
            payload_hash: ahash_bytes(&compressed),
            part_uncompressed_len: uncompressed.len() as u64,
            part_compressed_len: compressed.len() as u64,
            leading_pad: 0,
            stream_offset_mod: 0,
            reserved_shape: 0,
            parent_key: base_key,
            reserved_len: 0,
        };
        Ok(BuiltPreparedObject {
            header,
            payload: compressed,
        })
    })?;
    Ok(PreparedBase {
        root,
        object,
        page_size,
        compression,
    })
}

enum PlannedSource {
    Part(usize),
    Stitch(PreparedObject),
}

struct PlannedMapping {
    source: PlannedSource,
    file_offset: usize,
    guest_offset: usize,
    map_len: usize,
}

fn page_segments(
    parts: &[PreparedPart],
    page_start: usize,
    page_size: usize,
    total_compressed_len: usize,
) -> Result<Vec<PageSegment>> {
    let page_end = page_start
        .checked_add(page_size)
        .context("prepared page end overflow")?
        .min(total_compressed_len);
    let mut segments = Vec::new();
    for (part_index, part) in parts.iter().enumerate() {
        let part_end = part
            .stream_offset
            .checked_add(part.compressed_len)
            .context("prepared part end overflow")?;
        let start = page_start.max(part.stream_offset);
        let end = page_end.min(part_end);
        if start >= end {
            continue;
        }
        segments.push(PageSegment {
            part_index,
            part_offset: start - part.stream_offset,
            page_offset: start - page_start,
            len: end - start,
        });
    }
    let covered: usize = segments.iter().map(|segment| segment.len).sum();
    anyhow::ensure!(
        covered == page_end - page_start,
        "prepared parts contain a logical stream gap or overlap"
    );
    Ok(segments)
}

fn plan_prepared_mappings(
    root: &Path,
    parts: &mut [PreparedPart],
    page_size: usize,
    compression: initramfs::InitrdCompression,
) -> Result<(Vec<PreparedMapping>, usize, usize, usize)> {
    let total_compressed_len = parts.iter().try_fold(0usize, |total, part| {
        anyhow::ensure!(
            part.stream_offset == total,
            "prepared part stream offsets are not contiguous"
        );
        total
            .checked_add(part.compressed_len)
            .context("compressed initrd length overflow")
    })?;
    anyhow::ensure!(total_compressed_len > 0, "prepared initrd is empty");
    let mapped_len = round_up(total_compressed_len, page_size)?;
    let mut planned: Vec<PlannedMapping> = Vec::new();
    let mut stitch_pages = 0usize;
    let mut stitch_cache_hits = 0usize;

    for page_start in (0..mapped_len).step_by(page_size) {
        let segments = page_segments(parts, page_start, page_size, total_compressed_len)?;
        let direct = if let [segment] = segments.as_slice() {
            let part = &parts[segment.part_index];
            let layout_offset = usize::try_from(part.object.header.leading_pad)?
                .checked_add(segment.part_offset)
                .context("prepared part layout offset overflow")?;
            (segment.page_offset == 0 && layout_offset % page_size == 0)
                .then_some((segment.part_index, layout_offset))
        } else {
            None
        };

        if let Some((part_index, layout_offset)) = direct {
            let file_offset = parts[part_index]
                .object
                .data_offset
                .checked_add(layout_offset)
                .context("prepared direct-map file offset overflow")?;
            if let Some(previous) = planned.last_mut()
                && matches!(previous.source, PlannedSource::Part(index) if index == part_index)
                && previous.guest_offset + previous.map_len == page_start
                && previous.file_offset + previous.map_len == file_offset
            {
                previous.map_len = previous
                    .map_len
                    .checked_add(page_size)
                    .context("prepared direct range length overflow")?;
            } else {
                planned.push(PlannedMapping {
                    source: PlannedSource::Part(part_index),
                    file_offset,
                    guest_offset: page_start,
                    map_len: page_size,
                });
            }
            continue;
        }

        let stitch_key = stitch_recipe_key(&segments, parts, page_size, compression);
        let expectation = PreparedObjectExpectation {
            kind: PreparedObjectKind::Stitch,
            compression,
            key: stitch_key,
            page_size: page_size as u64,
            leading_pad: Some(0),
            parent_key: Some(stitch_key),
        };
        let stitch = get_or_build_prepared_object(root, expectation, || {
            let mut page = vec![0u8; page_size];
            for segment in &segments {
                let part = &parts[segment.part_index];
                let layout_offset = usize::try_from(part.object.header.leading_pad)?
                    .checked_add(segment.part_offset)
                    .context("stitch source offset overflow")?;
                let bytes = part.object.read_exact_at(layout_offset, segment.len)?;
                page[segment.page_offset..segment.page_offset + segment.len]
                    .copy_from_slice(&bytes);
            }
            let header = PreparedObjectHeader {
                kind: PreparedObjectKind::Stitch,
                compression,
                key: stitch_key,
                page_size: page_size as u64,
                payload_len: page_size as u64,
                payload_hash: ahash_bytes(&page),
                part_uncompressed_len: 0,
                part_compressed_len: page_size as u64,
                leading_pad: 0,
                stream_offset_mod: 0,
                reserved_shape: 0,
                parent_key: stitch_key,
                reserved_len: 0,
            };
            Ok(BuiltPreparedObject {
                header,
                payload: page,
            })
        })?;
        stitch_cache_hits += usize::from(stitch.cache_hit);
        stitch_pages += 1;
        planned.push(PlannedMapping {
            file_offset: stitch.data_offset,
            source: PlannedSource::Stitch(stitch),
            guest_offset: page_start,
            map_len: page_size,
        });
    }

    let direct_ranges = planned
        .iter()
        .filter(|range| matches!(range.source, PlannedSource::Part(_)))
        .count();
    let mut mappings = Vec::with_capacity(planned.len());
    for planned in planned {
        let fd = match planned.source {
            PlannedSource::Part(index) => parts[index]
                .object
                .fd
                .take()
                .context("prepared part fd was needed by multiple disjoint ranges")?,
            PlannedSource::Stitch(mut stitch) => stitch
                .fd
                .take()
                .context("prepared stitch fd already consumed")?,
        };
        mappings.push(PreparedMapping {
            fd,
            file_offset: planned.file_offset as u64,
            guest_offset: planned.guest_offset as u64,
            map_len: planned.map_len,
        });
    }
    Ok((mappings, direct_ranges, stitch_pages, stitch_cache_hits))
}

/// Complete a prepared base with independently cached immutable fragments.
///
/// The loader is uniform for verifier and ordinary VMs, every compression
/// format, and both supported architectures. `/init` stripping/compression is
/// keyed only by the pinned payload recipe, modules form another stable part,
/// and only the tiny control tail varies per cell. Logical pages wholly owned
/// by one part map its inode directly; pages spanning any number of adjacent
/// parts use a single immutable content-addressed stitch.
pub(crate) fn complete_prepared_initrd(
    prepared_base: PreparedBase,
    params: &initramfs::SuffixParams<'_>,
) -> Result<PreparedInitrd> {
    let PreparedBase {
        root,
        object: base,
        page_size,
        compression,
    } = prepared_base;
    let pinned = PinnedSuffixInputs::pin(&root, params)?;
    let coverage = pinned
        .payload
        .as_ref()
        .map(|payload| cached_coverage_probe(&root, payload))
        .transpose()?
        .unwrap_or((false, 0));
    let mut parts = Vec::with_capacity(4);
    let mut stream_offset = 0usize;
    let mut uncompressed_len = 0usize;
    let mut cache_hits = 0usize;

    let base_uncompressed_len = usize::try_from(base.header.part_uncompressed_len)?;
    let base_compressed_len = usize::try_from(base.header.part_compressed_len)?;
    stream_offset = stream_offset
        .checked_add(base_compressed_len)
        .context("compressed base length overflow")?;
    uncompressed_len = uncompressed_len
        .checked_add(base_uncompressed_len)
        .context("uncompressed base length overflow")?;
    cache_hits += usize::from(base.cache_hit);
    parts.push(PreparedPart {
        object: base,
        stream_offset: 0,
        uncompressed_len: base_uncompressed_len,
        compressed_len: base_compressed_len,
    });

    if let Some(payload) = &pinned.payload {
        let leading_pad = stream_offset % page_size;
        let key = payload_recipe_key(payload, coverage, page_size, compression);
        let payload_path = payload.proc_path();
        let canonical = get_or_build_part(
            &root,
            PreparedObjectKind::Payload,
            key,
            0,
            page_size,
            compression,
            || {
                let part = initramfs::build_payload_part_from_pinned(&payload_path)?;
                payload.verify_unchanged()?;
                Ok(part)
            },
        )?;
        payload.verify_unchanged()?;
        cache_hits += usize::from(canonical.cache_hit);
        let object = if leading_pad == 0 {
            canonical
        } else {
            let view = get_or_build_part_view(
                &root,
                &canonical,
                PreparedObjectKind::PayloadView,
                leading_pad,
                page_size,
                compression,
            )?;
            cache_hits += usize::from(view.cache_hit);
            view
        };
        let part_uncompressed_len = usize::try_from(object.header.part_uncompressed_len)?;
        let part_compressed_len = usize::try_from(object.header.part_compressed_len)?;
        parts.push(PreparedPart {
            object,
            stream_offset,
            uncompressed_len: part_uncompressed_len,
            compressed_len: part_compressed_len,
        });
        stream_offset = stream_offset
            .checked_add(part_compressed_len)
            .context("compressed payload length overflow")?;
        uncompressed_len = uncompressed_len
            .checked_add(part_uncompressed_len)
            .context("uncompressed payload length overflow")?;
    }

    if !pinned.modules.is_empty() {
        let leading_pad = stream_offset % page_size;
        let key = modules_recipe_key(&pinned.modules, page_size, compression);
        let module_sources = pinned.module_sources();
        let canonical = get_or_build_part(
            &root,
            PreparedObjectKind::Modules,
            key,
            0,
            page_size,
            compression,
            || {
                let part = initramfs::build_modules_part_from_pinned(&module_sources)?;
                pinned.verify_unchanged()?;
                Ok(part)
            },
        )?;
        pinned.verify_unchanged()?;
        cache_hits += usize::from(canonical.cache_hit);
        let object = if leading_pad == 0 {
            canonical
        } else {
            let view = get_or_build_part_view(
                &root,
                &canonical,
                PreparedObjectKind::ModulesView,
                leading_pad,
                page_size,
                compression,
            )?;
            cache_hits += usize::from(view.cache_hit);
            view
        };
        let part_uncompressed_len = usize::try_from(object.header.part_uncompressed_len)?;
        let part_compressed_len = usize::try_from(object.header.part_compressed_len)?;
        parts.push(PreparedPart {
            object,
            stream_offset,
            uncompressed_len: part_uncompressed_len,
            compressed_len: part_compressed_len,
        });
        stream_offset = stream_offset
            .checked_add(part_compressed_len)
            .context("compressed module length overflow")?;
        uncompressed_len = uncompressed_len
            .checked_add(part_uncompressed_len)
            .context("uncompressed module length overflow")?;
    }

    let leading_pad = stream_offset % page_size;
    let tail_key = tail_recipe_key(
        uncompressed_len,
        leading_pad,
        page_size,
        compression,
        params,
    );
    let tail = get_or_build_part(
        &root,
        PreparedObjectKind::Tail,
        tail_key,
        leading_pad,
        page_size,
        compression,
        || initramfs::build_dynamic_tail(uncompressed_len, params),
    )?;
    let tail_uncompressed_len = usize::try_from(tail.header.part_uncompressed_len)?;
    let tail_compressed_len = usize::try_from(tail.header.part_compressed_len)?;
    cache_hits += usize::from(tail.cache_hit);
    parts.push(PreparedPart {
        object: tail,
        stream_offset,
        uncompressed_len: tail_uncompressed_len,
        compressed_len: tail_compressed_len,
    });
    stream_offset = stream_offset
        .checked_add(tail_compressed_len)
        .context("compressed tail length overflow")?;
    uncompressed_len = uncompressed_len
        .checked_add(tail_uncompressed_len)
        .context("uncompressed tail length overflow")?;
    pinned.verify_unchanged()?;

    debug_assert_eq!(
        parts
            .iter()
            .map(|part| part.uncompressed_len)
            .sum::<usize>(),
        uncompressed_len
    );
    let part_count = parts.len();
    let (ranges, direct_ranges, stitch_pages, stitch_cache_hits) =
        plan_prepared_mappings(&root, &mut parts, page_size, compression)?;
    cache_hits += stitch_cache_hits;
    let plan = PreparedRangePlan {
        part_count,
        direct_ranges,
        stitch_pages,
        total_compressed_len: stream_offset,
    };
    Ok(PreparedInitrd {
        uncompressed_len,
        compressed_len: stream_offset,
        ranges,
        cache_hits,
        plan,
        mapping_granule: page_size,
        coverage_instrumented: coverage.0,
        coverage_reserve_bytes: coverage.1,
    })
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
#[cfg(test)]
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

/// Get or build the LZ4-compressed base, electing a single compressor
/// via the same `O_EXCL` race gate as [`get_or_build_base`] -- rather
/// than letting every cold-cache worker recompress and race to write.
///
/// Fast path: load the cached compressed segment. On a miss, `O_EXCL`
/// elects one Winner to compress + write; losers block on the Winner's
/// `LOCK_EX` (via `shm_open_lz4`'s blocking `LOCK_SH`) and then load the
/// result. Every failure path falls back to local compression and
/// always yields bytes -- there is no fatal error.
///
/// `KTSTR_CARGO_TEST_MODE` skips the SHM coordination entirely, like
/// `get_or_build_base`: under bare `cargo test` each invocation is
/// independent and the `O_EXCL` gate would only surface as confusing
/// flock contention.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn get_or_compress_base_shm(content_hash: u64, base_bytes: &[u8]) -> Vec<u8> {
    // Bare `cargo test`: skip the SHM layer entirely, compress locally.
    if crate::cargo_test_mode::cargo_test_mode_active() {
        return initramfs::lz4_legacy_compress(base_bytes);
    }

    let seg_name = initramfs::shm_lz4_segment_name(content_hash);

    // Fast path: an already-written compressed segment. Pin it so a
    // peer's cleanup_stale_shm cannot unlink it before try_cow_overlay.
    if let Some(lz4) = initramfs::shm_load_lz4(content_hash) {
        hold_shm_lock(&seg_name);
        return lz4;
    }

    // SHM race gate: O_EXCL elects a single compressor.
    match shm_try_create_excl(&seg_name) {
        ShmCreateResult::Winner(fd) => {
            tracing::debug!("lz4_base: compressor (O_EXCL won)");
            let lz4 = initramfs::lz4_legacy_compress(base_bytes);
            shm_write_and_release(fd, &lz4, &seg_name);
            // Pin the segment for the process lifetime so a peer's
            // cleanup_stale_shm cannot unlink it before this VM's
            // try_cow_overlay re-opens it by hash for the COW mapping.
            hold_shm_lock(&seg_name);
            return lz4;
        }
        ShmCreateResult::Exists => {
            tracing::debug!("lz4_base: waiting for compressor (EEXIST)");
            // Blocks on the Winner's LOCK_EX inside shm_open_lz4, so by
            // the time the read is granted the Winner has written; on
            // Winner failure the segment is unlinked/zeroed and the
            // load misses, falling through to the fallback.
            if let Some(lz4) = initramfs::shm_load_lz4(content_hash) {
                hold_shm_lock(&seg_name);
                return lz4;
            }
        }
        ShmCreateResult::Error => {
            if let Some(lz4) = initramfs::shm_load_lz4(content_hash) {
                hold_shm_lock(&seg_name);
                return lz4;
            }
        }
    }

    // Fallback: the Winner failed (write error -> segment unlinked) or
    // the post-wait load missed. Compress locally and best-effort store
    // so a later reader still finds it. Mirrors get_or_build_base's
    // fallback (the store is the non-blocking-skip writer).
    let lz4 = initramfs::lz4_legacy_compress(base_bytes);
    match initramfs::shm_store_lz4(content_hash, &lz4) {
        // Stored (or skipped because a peer holds it) -- pin so it
        // survives cleanup until try_cow_overlay.
        Ok(()) => hold_shm_lock(&seg_name),
        Err(e) => tracing::warn!("shm_store_lz4: {e:#}"),
    }
    lz4
}

/// Remove stale SHM segments from `/dev/shm` that don't match `current`.
/// Only unlinks segments not held by any process (`LOCK_EX | LOCK_NB`).
/// Parallel nextest workers hold `LOCK_SH` on their segments for the
/// process lifetime (via `HELD_SHM_LOCKS`), so their segments survive
/// cleanup from other workers.
#[cfg(test)]
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
#[cfg(test)]
static HELD_SHM_LOCKS: Mutex<Vec<rustix::fd::OwnedFd>> = Mutex::new(Vec::new());

/// Pin one SHM segment with a process-lifetime `LOCK_SH` so a peer's
/// `cleanup_stale_shm` cannot unlink it. Best-effort: a miss (segment
/// absent, or `LOCK_SH` contended by a writer's `LOCK_EX`) is a no-op.
/// The base and lz4 segments are pinned separately, each after its own
/// segment exists -- `get_or_build_base` pins the base, and
/// `get_or_compress_base_shm` pins the lz4 once it has been written.
#[cfg(test)]
fn hold_shm_lock(shm_name: &str) {
    if let Ok(fd) = rustix::shm::open(
        shm_name,
        rustix::shm::OFlags::RDONLY,
        rustix::fs::Mode::empty(),
    ) && rustix::fs::flock(&fd, rustix::fs::FlockOperation::NonBlockingLockShared).is_ok()
    {
        HELD_SHM_LOCKS.lock().unwrap().push(fd);
    }
}

// ---------------------------------------------------------------------------
// SHM O_EXCL race gate helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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
        let hash = initramfs::unique_test_shm_hash(7);
        let data = vec![0x07u8, 0x07, 0x01]; // cpio magic prefix
        initramfs::shm_store_base(hash, &data).unwrap();
        let loaded = initramfs::shm_load_base(hash);
        assert!(loaded.is_some(), "shm_load_base should return Some");
        assert_eq!(loaded.unwrap().as_ref(), &data[..]);
        initramfs::shm_unlink_base(hash);
    }

    #[test]
    fn shm_different_hashes_independent() {
        let h1 = initramfs::unique_test_shm_hash(8);
        let h2 = initramfs::unique_test_shm_hash(9);
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

    fn test_base_object(key: u64, page: usize, payload: Vec<u8>) -> BuiltPreparedObject {
        BuiltPreparedObject {
            header: PreparedObjectHeader {
                kind: PreparedObjectKind::Base,
                compression: initramfs::InitrdCompression::Lz4,
                key,
                page_size: page as u64,
                payload_len: payload.len() as u64,
                payload_hash: ahash_bytes(&payload),
                part_uncompressed_len: payload.len() as u64,
                part_compressed_len: payload.len() as u64,
                leading_pad: 0,
                stream_offset_mod: 0,
                reserved_shape: 0,
                parent_key: key,
                reserved_len: 0,
            },
            payload,
        }
    }

    fn test_base_expectation(key: u64, page: usize) -> PreparedObjectExpectation {
        PreparedObjectExpectation {
            kind: PreparedObjectKind::Base,
            compression: initramfs::InitrdCompression::Lz4,
            key,
            page_size: page as u64,
            leading_pad: Some(0),
            parent_key: Some(key),
        }
    }

    #[test]
    fn prepared_object_rejects_torn_and_header_corruption() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let key = 0x4142;
        let path = prepared_object_path(temp.path(), PreparedObjectKind::Base, key);
        std::fs::write(&path, b"torn").unwrap();
        assert!(try_open_prepared_object(&path, test_base_expectation(key, 4096)).is_err());

        let built = test_base_object(key, 4096, vec![7; 5000]);
        std::fs::remove_file(&path).unwrap();
        publish_prepared_object(temp.path(), &path, &built).unwrap();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(24)).unwrap();
        file.write_all(&8192u64.to_le_bytes()).unwrap();
        drop(file);
        assert!(
            try_open_prepared_object(&path, test_base_expectation(key, 4096)).is_err(),
            "header checksum must reject metadata corruption in O(1)"
        );
    }

    #[test]
    fn prepared_object_single_writer_and_lock_free_fast_hits() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        const CELLS: usize = 12;
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let key = 0x7777;
        let builds = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(std::sync::Barrier::new(CELLS));
        let mut workers = Vec::new();
        for _ in 0..CELLS {
            let root = temp.path().to_path_buf();
            let builds = builds.clone();
            let barrier = barrier.clone();
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                get_or_build_prepared_object(&root, test_base_expectation(key, 4096), || {
                    builds.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    Ok(test_base_object(key, 4096, vec![0x2a; 8193]))
                })
                .unwrap()
            }));
        }
        let objects: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(objects.iter().filter(|object| !object.cache_hit).count(), 1);
        assert!(
            objects
                .iter()
                .all(|object| object.header.payload_len == 8193)
        );
    }

    /// Regression: get_or_compress_base_shm pins the lz4 segment via
    /// hold_shm_lock so a peer's cleanup_stale_shm cannot unlink it.
    /// cleanup only unlinks a segment it can grab LOCK_EX on (its
    /// NonBlockingLockExclusive unlink-gate); a held LOCK_SH makes that
    /// probe return EWOULDBLOCK, so cleanup skips the segment. This pins
    /// that gate deterministically and single-process (same two-fd flock
    /// independence as shm_load_base_holds_lock_until_drop).
    #[test]
    fn held_lz4_segment_survives_cleanup_lock_probe() {
        let hash = initramfs::unique_test_shm_hash(10);
        let name = initramfs::shm_lz4_segment_name(hash);
        let _ = rustix::shm::unlink(name.as_str()); // clean any stale segment

        // Create the lz4 segment, then pin it (process-lifetime LOCK_SH).
        let mut data = initramfs::LZ4_LEGACY_MAGIC.to_vec();
        data.extend_from_slice(&[0x33u8; 64]);
        initramfs::shm_store_lz4(hash, &data).unwrap();
        hold_shm_lock(&name);

        // cleanup_stale_shm's unlink gate is a NonBlockingLockExclusive
        // probe; from a fresh fd it must fail while our LOCK_SH is held,
        // so cleanup would `continue` past this segment instead of
        // unlinking it.
        let fd = rustix::shm::open(
            name.as_str(),
            rustix::shm::OFlags::RDONLY,
            rustix::fs::Mode::empty(),
        )
        .expect("open the pinned lz4 segment");
        let probe = rustix::fs::flock(&fd, rustix::fs::FlockOperation::NonBlockingLockExclusive);
        assert!(
            matches!(probe, Err(e) if e == rustix::io::Errno::WOULDBLOCK),
            "cleanup's LOCK_EX probe must be blocked by the held LOCK_SH (got {probe:?})",
        );
        drop(fd);

        let _ = rustix::shm::unlink(name.as_str());
    }
}
