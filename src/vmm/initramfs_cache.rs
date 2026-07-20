//! Persistent, content-addressed initramfs preparation for VM storms.
//!
//! Inputs are pinned by open file description, content-digested once across
//! processes, and normalized into an exact shared-library/archive recipe.
//! Independently reusable base, payload, and module parts are compressed in
//! every supported initrd format and published as immutable regular files.
//! Only the small control tail varies per test cell.
//!
//! A 2 MiB planner maps pages owned by one part directly. Ordinary mixed-part
//! pages use content-addressed stitches; the final per-cell tail instead
//! composes one reusable full-page underlay with the smallest host-page-aligned
//! tail overlay. The VMM installs every layer over anonymous guest RAM with
//! `MAP_PRIVATE | MAP_FIXED`, so parallel VMs share clean page-cache pages and
//! pay for private memory only when the guest writes. Per-key `flock` election
//! ensures one transform per recipe across nextest workers; a versioned
//! namespace gate makes lock-file GC race-safe.
//!
//! The older process-local/POSIX-SHM base cache below is retained only for
//! compatibility unit tests. Production verifier and ordinary VM paths use
//! the regular-file CAS uniformly.

use anyhow::{Context, Result};
#[cfg(test)]
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
#[cfg(test)]
use std::hash::Hash;
use std::hash::Hasher;
use std::io::{Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;
use std::sync::OnceLock;

use std::hash::BuildHasher;

use ahash::AHasher;

use super::initramfs;
use crate::cache::content::{
    CoordinationFile, StableFileIdentity, cached_file_digest as shared_cached_file_digest,
    open_cache_record, read_fixed_cache_record,
};

/// Semantic cache key for a prepared base initramfs. It covers the exact
/// normalized archive recipe: packed binary and include contents, modes,
/// busybox bytes, and the content-addressed shared-library closure. `/init`
/// itself lives in an independently cached payload part.
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

    let digest = shared_cached_file_digest(&file, identity)
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

// The production prepared-initrd CAS lives alongside a cfg(test)-only
// compatibility cache for older base/suffix fixtures.

const PREPARED_CAS_SCHEMA: u32 = 7;
const PREPARED_CAS_MAGIC: &[u8; 8] = b"KTSTRIR\0";
const PREPARED_CAS_HEADER_LEN: usize = 128;
const PREPARED_CAS_MAX_BYTES: u64 = 8 << 30;
const PREPARED_CAS_MAX_AGE_SECS: u64 = 30 * 24 * 60 * 60;
const PREPARED_CAS_GC_INTERVAL_SECS: u64 = 60 * 60;
const PREPARED_VALIDATION_MAGIC: &[u8; 8] = b"KTSTRPV\0";
const PREPARED_VALIDATION_RECORD_LEN: usize = 128;
const COVERAGE_PROBE_MAGIC: &[u8; 8] = b"KTSTRCV\0";
const COVERAGE_PROBE_RECORD_LEN: usize = 48;
const CLOSURE_RECORD_MAGIC: &[u8; 8] = b"KTSTRCL\0";
const CLOSURE_RECORD_HEADER_LEN: usize = 40;
const CLOSURE_RECORD_MAX_PAYLOAD_LEN: usize = 16 << 20;
const CLOSURE_RECORD_MAX_LEN: usize = CLOSURE_RECORD_HEADER_LEN + CLOSURE_RECORD_MAX_PAYLOAD_LEN;
const CLOSURE_RECORD_MAX_ENTRY_COUNT: usize = 16 << 10;
const CLOSURE_RECORD_MAX_SEARCH_PATH_COUNT: usize = 128 << 10;
const CLOSURE_RECORD_MAX_ENTRY_PATH_LEN: usize = 4096;
const CLOSURE_RECORD_MAX_SEARCH_PATH_LEN: usize = 1 << 20;
const CLOSURE_RECORD_MAX_INITIAL_SEQUENCE_ALLOCATION: usize = 1 << 20;
// Version the complete namespace, not only the object recipes. Older ktstr
// binaries do not participate in the namespace-gate protocol below; keeping
// their objects, memos, and locks in separate directories prevents an old GC
// from unlinking a live object coordinated by this version's lock inode.
const PREPARED_OBJECTS_DIR: &str = "prepared-initrd-v7-objects";
const PREPARED_LOCKS_DIR: &str = ".prepared-initrd-v7-locks";
const PREPARED_DIGESTS_DIR: &str = "prepared-initrd-v7-digests";
const PREPARED_PROBES_DIR: &str = "prepared-initrd-v7-probes";
const PREPARED_CLOSURES_DIR: &str = "prepared-initrd-v7-closures";
const PREPARED_GC_STAMP: &str = ".prepared-initrd-v7-gc-stamp";
const CONTENT_HASH_CHUNK_LEN: usize = 1 << 20;
const PREPARED_NAMESPACE_GATE: &str = "namespace-v1.lock";
const PREPARED_GC_LOCK: &str = "gc.lock";
/// Uniform direct-map granule. It is host-page aligned on every supported
/// host and also satisfies Linux hugetlb's MAP_FIXED unmap boundary rule for
/// the VM's optional MAP_HUGETLB|MAP_HUGE_2MB guest backing.
pub(crate) const PREPARED_MAPPING_GRANULE: usize = 2 << 20;

fn prepared_file_alignment() -> Result<usize> {
    static HOST_PAGE: OnceLock<usize> = OnceLock::new();
    let host_page = *HOST_PAGE.get_or_init(|| {
        // SAFETY: sysconf is thread-safe and _SC_PAGESIZE has no caller-side
        // invariants. Linux always supplies a positive power-of-two value;
        // retain the 4 KiB fallback so an unexpected libc failure cannot
        // introduce a zero divisor into cache geometry.
        let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if value > 0 { value as usize } else { 0x1000 }
    });
    anyhow::ensure!(
        host_page.is_power_of_two() && PREPARED_MAPPING_GRANULE.is_multiple_of(host_page),
        "host page size {host_page} is incompatible with the \
         {PREPARED_MAPPING_GRANULE}-byte prepared mapping granule"
    );
    Ok(host_page)
}

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
    Boundary = 8,
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
            8 => Ok(Self::Boundary),
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
            Self::Boundary => "boundary",
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

/// Hash a prepared payload with fixed call boundaries.
///
/// `Hasher::write(a); write(b)` is not required to equal `write(a || b)`.
/// Builders hold the complete payload while cache-hit validation streams it,
/// so both sides must use the same content-derived chunking.
fn content_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = fixed_hasher();
    for chunk in bytes.chunks(CONTENT_HASH_CHUNK_LEN) {
        hasher.write(chunk);
    }
    hasher.finish()
}

fn pread_exact(file: &File, mut offset: u64, mut buffer: &mut [u8], subject: &str) -> Result<()> {
    while !buffer.is_empty() {
        let read = loop {
            match file.read_at(buffer, offset) {
                Ok(read) => break read,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error).with_context(|| format!("pread {subject}")),
            }
        };
        anyhow::ensure!(read > 0, "{subject} was truncated while reading");
        offset = offset
            .checked_add(u64::try_from(read)?)
            .with_context(|| format!("{subject} offset overflow"))?;
        buffer = &mut buffer[read..];
    }
    Ok(())
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

fn stable_identity_from_resolver(identity: initramfs::ResolverFileIdentity) -> StableFileIdentity {
    StableFileIdentity {
        dev: identity.dev,
        ino: identity.ino,
        size: identity.size,
        mtime_secs: identity.mtime_secs,
        mtime_nsecs: identity.mtime_nsecs,
        ctime_secs: identity.ctime_secs,
        ctime_nsecs: identity.ctime_nsecs,
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
            after.same_open_content_version(self.identity),
            "prepared initrd input changed while in use: {}",
            self.display_path.display()
        );
        Ok(())
    }

    fn is_elf(&self) -> Result<bool> {
        self.verify_unchanged()?;
        if self.identity.size < 4 {
            self.verify_unchanged()?;
            return Ok(false);
        }
        let mut magic = [0u8; 4];
        pread_exact(&self.file, 0, &mut magic, "ELF input")
            .with_context(|| format!("probe ELF input {}", self.display_path.display()))?;
        self.verify_unchanged()?;
        Ok(magic == *b"\x7fELF")
    }

    fn read_owned_stable(&self) -> Result<Vec<u8>> {
        self.read_owned_stable_with_hook(|_| {})
    }

    fn read_owned_stable_with_hook(&self, mut after_chunk: impl FnMut(usize)) -> Result<Vec<u8>> {
        self.verify_unchanged()?;
        let len = usize::try_from(self.identity.size)
            .with_context(|| format!("input is too large: {}", self.display_path.display()))?;
        let mut bytes = vec![0u8; len];
        let mut offset = 0usize;
        while offset < bytes.len() {
            let read = loop {
                match self.file.read_at(&mut bytes[offset..], offset as u64) {
                    Ok(read) => break read,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("pread input {}", self.display_path.display())
                        });
                    }
                }
            };
            anyhow::ensure!(
                read > 0,
                "input was truncated while reading: {}",
                self.display_path.display()
            );
            offset = offset
                .checked_add(read)
                .context("pinned input read offset overflow")?;
            after_chunk(offset);
        }
        self.verify_unchanged()?;
        Ok(bytes)
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

fn open_lock_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open prepared initrd lock {}", path.display()))
}

fn open_namespace_gate(root: &Path) -> Result<File> {
    open_lock_file(&root.join(PREPARED_LOCKS_DIR).join(PREPARED_NAMESPACE_GATE))
}

pub(super) use crate::cache::content::flock_retry;

fn open_coord_file(root: &Path, path: &Path) -> Result<CoordinationFile> {
    crate::cache::content::open_coord_file(
        &root.join(PREPARED_LOCKS_DIR).join(PREPARED_NAMESPACE_GATE),
        path,
    )
}

/// Open an explicit or recorded host input without blocking on a substituted
/// FIFO. Deliberately follow symlinks: package-manager and linker paths
/// commonly use them, and the open file description is pinned immediately
/// afterward by its regular-file identity.
fn open_pinned_input_path(path: &Path) -> std::io::Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
}

fn pin_input(_root: &Path, path: &Path) -> Result<PinnedInput> {
    let file = open_pinned_input_path(path)
        .with_context(|| format!("open prepared input {}", path.display()))?;
    let identity = StableFileIdentity::from_file(&file)
        .with_context(|| format!("stat prepared input {}", path.display()))?;
    let content_hash = shared_cached_file_digest(&file, identity)
        .with_context(|| format!("digest prepared input {}", path.display()))?;
    Ok(PinnedInput {
        file,
        identity,
        content_hash,
        display_path: path.to_path_buf(),
    })
}

fn read_coverage_probe_record(path: &Path, key: u64) -> Result<Option<(bool, u64)>> {
    let Some(bytes) =
        read_fixed_cache_record::<COVERAGE_PROBE_RECORD_LEN>(path, "coverage probe memo")?
    else {
        return Ok(None);
    };
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
    if payload.identity.size == 0 {
        payload.verify_unchanged()?;
        return Ok((false, 0));
    }
    // Parse owned bytes. An mmap of the mutable original fd can SIGBUS if a
    // concurrent writer truncates the inode before goblin faults a page;
    // pread turns that race into a normal short-read/identity error.
    let bytes = payload.read_owned_stable()?;
    let Ok(elf) = goblin::elf::Elf::parse(&bytes) else {
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
    let mut wait_for_successor = false;
    loop {
        let mut coordination = open_coord_file(root, &lock_path)?;
        let election = if wait_for_successor {
            coordination.lock_exclusive()
        } else {
            coordination.try_lock_exclusive()
        };
        match election {
            Ok(()) => {
                coordination.release_namespace_gate();
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
                temp.persist(&record_path)
                    .map_err(|error| error.error)
                    .with_context(|| {
                        format!("publish coverage probe memo {}", record_path.display())
                    })?;
                return Ok(result);
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                coordination.lock_shared().with_context(|| {
                    format!("wait for coverage probe {}", record_path.display())
                })?;
                coordination.release_namespace_gate();
                if let Some(result) = read_coverage_probe_record(&record_path, key)? {
                    return Ok(result);
                }
                wait_for_successor = true;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("elect coverage probe builder {}", record_path.display())
                });
            }
        }
    }
}

#[derive(Debug, serde::Serialize)]
struct ClosureEntryRecord {
    guest_path: String,
    host_path: PathBuf,
    identity: StableFileIdentity,
    content_hash: u64,
}

#[derive(Debug, serde::Serialize)]
struct ClosureSearchRecord {
    path: PathBuf,
    identity: Option<StableFileIdentity>,
}

#[derive(Debug, serde::Serialize)]
struct ClosureRecord {
    entries: Vec<ClosureEntryRecord>,
    search_paths: Vec<ClosureSearchRecord>,
}

#[derive(Debug)]
struct BoundedStr<'a, const MAX_LEN: usize>(&'a str);

impl<'de: 'a, 'a, const MAX_LEN: usize> serde::Deserialize<'de> for BoundedStr<'a, MAX_LEN> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct BoundedStrVisitor<'a, const MAX_LEN: usize>(std::marker::PhantomData<&'a str>);

        impl<'de: 'a, 'a, const MAX_LEN: usize> serde::de::Visitor<'de> for BoundedStrVisitor<'a, MAX_LEN> {
            type Value = BoundedStr<'a, MAX_LEN>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "a UTF-8 path no longer than {MAX_LEN} bytes")
            }

            fn visit_borrowed_str<E>(self, value: &'de str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value.len() > MAX_LEN {
                    return Err(E::invalid_length(value.len(), &self));
                }
                Ok(BoundedStr(value))
            }
        }

        deserializer.deserialize_str(BoundedStrVisitor::<MAX_LEN>(std::marker::PhantomData))
    }
}

#[derive(Debug)]
struct BoundedVec<T, const MAX_LEN: usize>(Vec<T>);

impl<'de, T, const MAX_LEN: usize> serde::Deserialize<'de> for BoundedVec<T, MAX_LEN>
where
    T: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct BoundedVecVisitor<T, const MAX_LEN: usize>(std::marker::PhantomData<T>);

        struct RejectExcessElement;

        impl<'de> serde::de::DeserializeSeed<'de> for RejectExcessElement {
            type Value = ();

            fn deserialize<D>(self, _deserializer: D) -> std::result::Result<Self::Value, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                Err(serde::de::Error::custom(
                    "loader closure sequence exceeds its element limit",
                ))
            }
        }

        impl<'de, T, const MAX_LEN: usize> serde::de::Visitor<'de> for BoundedVecVisitor<T, MAX_LEN>
        where
            T: serde::Deserialize<'de>,
        {
            type Value = BoundedVec<T, MAX_LEN>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "a sequence with at most {MAX_LEN} elements")
            }

            fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let hint = sequence.size_hint();
                if hint.is_some_and(|length| length > MAX_LEN) {
                    return Err(serde::de::Error::invalid_length(
                        hint.unwrap_or(usize::MAX),
                        &self,
                    ));
                }
                let allocation_element_size = std::mem::size_of::<T>().max(1);
                let initial_capacity = hint
                    .unwrap_or(0)
                    .min(MAX_LEN)
                    .min(CLOSURE_RECORD_MAX_INITIAL_SEQUENCE_ALLOCATION / allocation_element_size);
                let mut values = Vec::with_capacity(initial_capacity);
                loop {
                    if values.len() == MAX_LEN {
                        if sequence.next_element_seed(RejectExcessElement)?.is_none() {
                            return Ok(BoundedVec(values));
                        }
                        unreachable!("reject seed never returns a decoded element");
                    }
                    match sequence.next_element()? {
                        Some(value) => values.push(value),
                        None => return Ok(BoundedVec(values)),
                    }
                }
            }
        }

        deserializer.deserialize_seq(BoundedVecVisitor::<T, MAX_LEN>(std::marker::PhantomData))
    }
}

#[derive(Debug, serde::Deserialize)]
struct ClosureEntryRecordWire<'a> {
    #[serde(borrow)]
    guest_path: BoundedStr<'a, CLOSURE_RECORD_MAX_ENTRY_PATH_LEN>,
    #[serde(borrow)]
    host_path: BoundedStr<'a, CLOSURE_RECORD_MAX_ENTRY_PATH_LEN>,
    identity: StableFileIdentity,
    content_hash: u64,
}

#[derive(Debug, serde::Deserialize)]
struct ClosureSearchRecordWire<'a> {
    #[serde(borrow)]
    path: BoundedStr<'a, CLOSURE_RECORD_MAX_SEARCH_PATH_LEN>,
    identity: Option<StableFileIdentity>,
}

#[derive(Debug, serde::Deserialize)]
struct ClosureRecordWire<'a> {
    #[serde(borrow)]
    entries: BoundedVec<ClosureEntryRecordWire<'a>, CLOSURE_RECORD_MAX_ENTRY_COUNT>,
    #[serde(borrow)]
    search_paths: BoundedVec<ClosureSearchRecordWire<'a>, CLOSURE_RECORD_MAX_SEARCH_PATH_COUNT>,
}

impl From<ClosureRecordWire<'_>> for ClosureRecord {
    fn from(record: ClosureRecordWire<'_>) -> Self {
        Self {
            entries: record
                .entries
                .0
                .into_iter()
                .map(|entry| ClosureEntryRecord {
                    guest_path: entry.guest_path.0.to_owned(),
                    host_path: PathBuf::from(entry.host_path.0),
                    identity: entry.identity,
                    content_hash: entry.content_hash,
                })
                .collect(),
            search_paths: record
                .search_paths
                .0
                .into_iter()
                .map(|search| ClosureSearchRecord {
                    path: PathBuf::from(search.path.0),
                    identity: search.identity,
                })
                .collect(),
        }
    }
}

fn validate_closure_record_limits(record: &ClosureRecord) -> Result<()> {
    anyhow::ensure!(
        record.entries.len() <= CLOSURE_RECORD_MAX_ENTRY_COUNT,
        "loader closure has more than {} entries",
        CLOSURE_RECORD_MAX_ENTRY_COUNT
    );
    anyhow::ensure!(
        record.search_paths.len() <= CLOSURE_RECORD_MAX_SEARCH_PATH_COUNT,
        "loader closure has more than {} search paths",
        CLOSURE_RECORD_MAX_SEARCH_PATH_COUNT
    );
    for entry in &record.entries {
        anyhow::ensure!(
            entry.guest_path.len() <= CLOSURE_RECORD_MAX_ENTRY_PATH_LEN,
            "loader closure guest path exceeds {} bytes",
            CLOSURE_RECORD_MAX_ENTRY_PATH_LEN
        );
        let host_path = entry.host_path.to_str().with_context(|| {
            format!(
                "loader closure host path is not UTF-8: {}",
                entry.host_path.display()
            )
        })?;
        anyhow::ensure!(
            host_path.len() <= CLOSURE_RECORD_MAX_ENTRY_PATH_LEN,
            "loader closure host path exceeds {} bytes",
            CLOSURE_RECORD_MAX_ENTRY_PATH_LEN
        );
    }
    for search in &record.search_paths {
        let path = search.path.to_str().with_context(|| {
            format!(
                "loader closure search path is not UTF-8: {}",
                search.path.display()
            )
        })?;
        anyhow::ensure!(
            path.len() <= CLOSURE_RECORD_MAX_SEARCH_PATH_LEN,
            "loader closure search path exceeds {} bytes",
            CLOSURE_RECORD_MAX_SEARCH_PATH_LEN
        );
    }
    Ok(())
}

fn decode_closure_record(payload: &[u8]) -> Result<ClosureRecord> {
    let (record, trailing) = postcard::take_from_bytes::<ClosureRecordWire<'_>>(payload)
        .context("decode bounded loader closure payload")?;
    anyhow::ensure!(
        trailing.is_empty(),
        "loader closure payload has {} trailing bytes",
        trailing.len()
    );
    Ok(record.into())
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
    let file = match open_pinned_input_path(&record.host_path) {
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
    validate_closure_record_limits(record)?;
    let payload = postcard::to_stdvec(record).context("encode loader closure payload")?;
    anyhow::ensure!(
        payload.len() <= CLOSURE_RECORD_MAX_PAYLOAD_LEN,
        "loader closure payload exceeds {} bytes",
        CLOSURE_RECORD_MAX_PAYLOAD_LEN
    );
    let record_len = CLOSURE_RECORD_HEADER_LEN
        .checked_add(payload.len())
        .context("loader closure envelope length overflow")?;
    let mut bytes = Vec::with_capacity(record_len);
    bytes.extend_from_slice(CLOSURE_RECORD_MAGIC);
    bytes.extend_from_slice(&PREPARED_CAS_SCHEMA.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&key.to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(payload.len())?.to_le_bytes());
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
    let subject = "loader closure";
    let Some((file, identity)) = open_cache_record(record_path, subject)? else {
        return Ok(None);
    };
    anyhow::ensure!(
        identity.size >= CLOSURE_RECORD_HEADER_LEN as u64,
        "loader closure envelope is truncated: {}",
        record_path.display()
    );
    anyhow::ensure!(
        identity.size <= CLOSURE_RECORD_MAX_LEN as u64,
        "loader closure envelope exceeds {} bytes: {}",
        CLOSURE_RECORD_MAX_LEN,
        record_path.display()
    );
    let mut header = [0u8; CLOSURE_RECORD_HEADER_LEN];
    pread_exact(&file, 0, &mut header, subject)?;
    anyhow::ensure!(
        &header[..8] == CLOSURE_RECORD_MAGIC
            && u32::from_le_bytes(header[8..12].try_into().unwrap()) == PREPARED_CAS_SCHEMA
            && header[12..16] == [0; 4]
            && u64::from_le_bytes(header[16..24].try_into().unwrap()) == expected_key,
        "loader closure envelope recipe mismatch: {}",
        record_path.display()
    );
    let payload_len = usize::try_from(u64::from_le_bytes(header[24..32].try_into().unwrap()))
        .context("loader closure payload length exceeds usize")?;
    let record_len = CLOSURE_RECORD_HEADER_LEN
        .checked_add(payload_len)
        .context("loader closure envelope length overflow")?;
    anyhow::ensure!(
        record_len == usize::try_from(identity.size)?,
        "loader closure envelope length mismatch: {}",
        record_path.display()
    );
    let mut bytes = vec![0u8; record_len];
    bytes[..CLOSURE_RECORD_HEADER_LEN].copy_from_slice(&header);
    pread_exact(
        &file,
        CLOSURE_RECORD_HEADER_LEN as u64,
        &mut bytes[CLOSURE_RECORD_HEADER_LEN..],
        subject,
    )?;
    anyhow::ensure!(
        StableFileIdentity::from_file(&file)? == identity,
        "loader closure changed while reading: {}",
        record_path.display()
    );
    let recorded_checksum = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
    bytes[32..40].fill(0);
    anyhow::ensure!(
        ahash_bytes(&bytes) == recorded_checksum,
        "loader closure checksum mismatch: {}",
        record_path.display()
    );
    let record = decode_closure_record(&bytes[CLOSURE_RECORD_HEADER_LEN..])
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
        let identity = stable_identity_from_resolver(identity);
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
        let identity = observation.identity.map(stable_identity_from_resolver);
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
    let mut wait_for_successor = false;
    loop {
        let mut coordination = open_coord_file(root, &lock_path)?;
        let election = if wait_for_successor {
            coordination.lock_exclusive()
        } else {
            coordination.try_lock_exclusive()
        };
        match election {
            Ok(()) => {
                coordination.release_namespace_gate();
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
                temp.persist(&record_path)
                    .map_err(|error| error.error)
                    .with_context(|| format!("publish loader closure {}", record_path.display()))?;
                return Ok(closure);
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                coordination.lock_shared().with_context(|| {
                    format!("wait for loader closure {}", record_path.display())
                })?;
                coordination.release_namespace_gate();
                if let Some(closure) = read_pinned_closure_record(&record_path, key)? {
                    return Ok(closure);
                }
                wait_for_successor = true;
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
    mapping_granule: u64,
    payload_len: u64,
    payload_hash: u64,
    part_uncompressed_len: u64,
    part_compressed_len: u64,
    leading_pad: u64,
    stream_offset_mod: u64,
    file_alignment: u64,
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
        out.extend_from_slice(&self.mapping_granule.to_le_bytes());
        out.extend_from_slice(&self.payload_len.to_le_bytes());
        out.extend_from_slice(&self.payload_hash.to_le_bytes());
        out.extend_from_slice(&self.part_uncompressed_len.to_le_bytes());
        out.extend_from_slice(&self.part_compressed_len.to_le_bytes());
        out.extend_from_slice(&self.leading_pad.to_le_bytes());
        out.extend_from_slice(&self.stream_offset_mod.to_le_bytes());
        out.extend_from_slice(&self.file_alignment.to_le_bytes());
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
            mapping_granule: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            payload_len: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            payload_hash: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            part_uncompressed_len: u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            part_compressed_len: u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
            leading_pad: u64::from_le_bytes(bytes[64..72].try_into().unwrap()),
            stream_offset_mod: u64::from_le_bytes(bytes[72..80].try_into().unwrap()),
            file_alignment: u64::from_le_bytes(bytes[80..88].try_into().unwrap()),
            parent_key: u64::from_le_bytes(bytes[88..96].try_into().unwrap()),
            reserved_len: u64::from_le_bytes(bytes[96..104].try_into().unwrap()),
        })
    }

    fn validate_shape(self) -> Result<()> {
        let granule =
            usize::try_from(self.mapping_granule).context("mapping granule exceeds usize")?;
        anyhow::ensure!(
            granule.is_power_of_two(),
            "cached mapping granule is invalid"
        );
        let file_alignment =
            usize::try_from(self.file_alignment).context("file alignment exceeds usize")?;
        anyhow::ensure!(
            file_alignment.is_power_of_two()
                && file_alignment >= PREPARED_CAS_HEADER_LEN
                && granule.is_multiple_of(file_alignment),
            "cached prepared-object file alignment is incompatible with its mapping granule"
        );
        let leading =
            usize::try_from(self.leading_pad).context("part leading pad exceeds usize")?;
        let compressed =
            usize::try_from(self.part_compressed_len).context("part length exceeds usize")?;
        anyhow::ensure!(
            leading < file_alignment && self.stream_offset_mod == self.leading_pad,
            "cached part leading-pad geometry is inconsistent"
        );
        anyhow::ensure!(
            self.reserved_len == 0,
            "cached prepared object reserved length changed"
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
                        && self.payload_len == self.mapping_granule
                        && self.part_compressed_len == self.mapping_granule
                        && self.part_uncompressed_len == 0,
                    "stitch object must contain exactly one page"
                );
            }
            PreparedObjectKind::Boundary => {
                anyhow::ensure!(
                    leading == 0
                        && self.payload_len > 0
                        && self.payload_len <= self.mapping_granule
                        && self.payload_len.is_multiple_of(self.file_alignment)
                        && self.part_compressed_len == self.payload_len
                        && self.part_uncompressed_len == 0,
                    "boundary object must contain a non-empty host-page-aligned overlay"
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
    mapping_granule: u64,
    file_alignment: u64,
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
            header.mapping_granule == self.mapping_granule,
            "prepared object mapping-granule mismatch"
        );
        anyhow::ensure!(
            header.file_alignment == self.file_alignment,
            "prepared object file-alignment mismatch"
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
    fn read_exact_into(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(out.len())
            .context("prepared object read overflow")?;
        let payload_len =
            usize::try_from(self.header.payload_len).context("payload length exceeds usize")?;
        anyhow::ensure!(end <= payload_len, "prepared object read exceeds payload");
        let file_offset = self
            .data_offset
            .checked_add(offset)
            .context("prepared object file offset overflow")?;
        let mut done = 0usize;
        while done < out.len() {
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
                    out.len() - done,
                    absolute,
                )
            };
            if read < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error).context("pread prepared object");
            }
            anyhow::ensure!(read != 0, "prepared object truncated during pread");
            done += read as usize;
        }
        Ok(())
    }

    fn read_exact_at(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; len];
        self.read_exact_into(offset, &mut out)?;
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

fn prepared_validation_key(kind: PreparedObjectKind, object_key: u64) -> u64 {
    let mut hasher = fixed_hasher();
    hash_len_prefixed(&mut hasher, b"ktstr-prepared-payload-validation");
    hash_u32(&mut hasher, PREPARED_CAS_SCHEMA);
    hash_u8(&mut hasher, kind as u8);
    hash_u64(&mut hasher, object_key);
    hasher.finish()
}

fn prepared_validation_record_path(
    root: &Path,
    kind: PreparedObjectKind,
    object_key: u64,
) -> PathBuf {
    let key = prepared_validation_key(kind, object_key);
    root.join(PREPARED_DIGESTS_DIR)
        .join(format!("{key:016x}.validation"))
}

fn prepared_validation_lock_path(
    root: &Path,
    kind: PreparedObjectKind,
    object_key: u64,
) -> PathBuf {
    let key = prepared_validation_key(kind, object_key);
    root.join(PREPARED_LOCKS_DIR)
        .join(format!("validation-{key:016x}.lock"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreparedPayloadValidation {
    payload_hash: u64,
    padding_is_zero: bool,
}

fn read_prepared_validation_record(
    record_path: &Path,
    header: PreparedObjectHeader,
    identity: StableFileIdentity,
    data_offset: usize,
) -> Result<Option<PreparedPayloadValidation>> {
    let Some(bytes) = read_fixed_cache_record::<PREPARED_VALIDATION_RECORD_LEN>(
        record_path,
        "prepared-object validation memo",
    )?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        &bytes[..8] == PREPARED_VALIDATION_MAGIC,
        "prepared-object validation memo magic mismatch"
    );
    anyhow::ensure!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) == PREPARED_CAS_SCHEMA,
        "prepared-object validation memo schema mismatch"
    );
    anyhow::ensure!(
        bytes[13] & !1 == 0
            && bytes[14..16] == [0; 2]
            && bytes[112..].iter().all(|byte| *byte == 0),
        "prepared-object validation memo reserved bytes changed"
    );
    let recorded_checksum = u64::from_le_bytes(bytes[104..112].try_into().unwrap());
    let mut canonical = bytes;
    canonical[104..112].fill(0);
    anyhow::ensure!(
        ahash_bytes(&canonical) == recorded_checksum,
        "prepared-object validation memo checksum mismatch"
    );
    anyhow::ensure!(
        PreparedObjectKind::from_tag(canonical[12])? == header.kind
            && u64::from_le_bytes(canonical[16..24].try_into().unwrap()) == header.key,
        "prepared-object validation memo recipe mismatch"
    );

    let recorded_identity = StableFileIdentity::decode(&canonical[24..80])?;
    if recorded_identity != identity {
        // The same semantic recipe may be rebuilt after GC. Its fixed memo
        // path then names the previous immutable inode revision, which is a
        // normal cache miss rather than a record collision.
        return Ok(None);
    }
    anyhow::ensure!(
        u64::from_le_bytes(canonical[80..88].try_into().unwrap()) == u64::try_from(data_offset)?
            && u64::from_le_bytes(canonical[88..96].try_into().unwrap()) == header.payload_len,
        "prepared-object validation memo payload geometry mismatch"
    );
    Ok(Some(PreparedPayloadValidation {
        payload_hash: u64::from_le_bytes(canonical[96..104].try_into().unwrap()),
        padding_is_zero: canonical[13] & 1 != 0,
    }))
}

fn publish_prepared_validation_record(
    root: &Path,
    header: PreparedObjectHeader,
    identity: StableFileIdentity,
    data_offset: usize,
    validation: PreparedPayloadValidation,
) -> Result<()> {
    let validation_key = prepared_validation_key(header.kind, header.key);
    let record_path = prepared_validation_record_path(root, header.kind, header.key);
    let mut bytes = Vec::with_capacity(PREPARED_VALIDATION_RECORD_LEN);
    bytes.extend_from_slice(PREPARED_VALIDATION_MAGIC);
    bytes.extend_from_slice(&PREPARED_CAS_SCHEMA.to_le_bytes());
    bytes.push(header.kind as u8);
    bytes.push(u8::from(validation.padding_is_zero));
    bytes.extend_from_slice(&[0; 2]);
    bytes.extend_from_slice(&header.key.to_le_bytes());
    identity.encode(&mut bytes);
    bytes.extend_from_slice(&u64::try_from(data_offset)?.to_le_bytes());
    bytes.extend_from_slice(&header.payload_len.to_le_bytes());
    bytes.extend_from_slice(&validation.payload_hash.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&[0; 16]);
    debug_assert_eq!(bytes.len(), PREPARED_VALIDATION_RECORD_LEN);
    let checksum = ahash_bytes(&bytes);
    bytes[104..112].copy_from_slice(&checksum.to_le_bytes());

    let mut temp = tempfile::Builder::new()
        .prefix(&format!(".tmp-validation-{validation_key:016x}-"))
        .tempfile_in(root.join(PREPARED_DIGESTS_DIR))
        .context("create prepared-object validation memo temp")?;
    temp.write_all(&bytes)
        .context("write prepared-object validation memo")?;
    temp.persist(&record_path)
        .map_err(|error| error.error)
        .with_context(|| {
            format!(
                "publish prepared-object validation memo {}",
                record_path.display()
            )
        })?;
    Ok(())
}

#[cfg(test)]
fn note_prepared_payload_hash_for_test() -> Result<()> {
    use std::io::Write as _;

    let Some(counter) = std::env::var_os("KTSTR_PREPARED_VALIDATION_COUNTER").map(PathBuf::from)
    else {
        return Ok(());
    };
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&counter)
        .with_context(|| format!("open prepared validation counter {}", counter.display()))?;
    writeln!(file, "{}", std::process::id()).context("write prepared validation counter")?;
    file.sync_all()
        .context("sync prepared validation counter")?;
    Ok(())
}

fn hash_prepared_payload(
    file: &File,
    identity: StableFileIdentity,
    data_offset: usize,
    payload_len: usize,
    padded_payload_len: usize,
) -> Result<PreparedPayloadValidation> {
    let before = StableFileIdentity::from_file(file)?;
    anyhow::ensure!(
        before == identity,
        "prepared object changed before payload validation"
    );
    #[cfg(test)]
    note_prepared_payload_hash_for_test()?;

    let mut hasher = fixed_hasher();
    let mut buffer = vec![0u8; CONTENT_HASH_CHUNK_LEN];
    let mut payload_done = 0usize;
    while payload_done < payload_len {
        let chunk_len = (payload_len - payload_done).min(buffer.len());
        let file_offset = data_offset
            .checked_add(payload_done)
            .context("prepared payload validation offset overflow")?;
        pread_exact(
            file,
            u64::try_from(file_offset)?,
            &mut buffer[..chunk_len],
            "prepared object payload for validation",
        )?;
        hasher.write(&buffer[..chunk_len]);
        payload_done = payload_done
            .checked_add(chunk_len)
            .context("prepared payload validation length overflow")?;
    }

    let padding_len = padded_payload_len
        .checked_sub(payload_len)
        .context("prepared padded payload is shorter than its content")?;
    let mut padding_done = 0usize;
    let mut padding_is_zero = true;
    while padding_done < padding_len {
        let chunk_len = (padding_len - padding_done).min(buffer.len());
        let file_offset = data_offset
            .checked_add(payload_len)
            .and_then(|offset| offset.checked_add(padding_done))
            .context("prepared padding validation offset overflow")?;
        pread_exact(
            file,
            u64::try_from(file_offset)?,
            &mut buffer[..chunk_len],
            "prepared object padding for validation",
        )?;
        padding_is_zero &= buffer[..chunk_len].iter().all(|byte| *byte == 0);
        padding_done = padding_done
            .checked_add(chunk_len)
            .context("prepared padding validation length overflow")?;
    }
    let digest = hasher.finish();
    let after = StableFileIdentity::from_file(file)?;
    anyhow::ensure!(
        after == identity,
        "prepared object changed while validating its payload"
    );
    Ok(PreparedPayloadValidation {
        payload_hash: digest,
        padding_is_zero,
    })
}

fn cached_prepared_payload_validation(
    root: &Path,
    file: &File,
    header: PreparedObjectHeader,
    identity: StableFileIdentity,
    data_offset: usize,
) -> Result<PreparedPayloadValidation> {
    // Callers already hold the immutable object's LOCK_SH. The global order is
    // namespace gate -> object-build coordination (when present) -> immutable
    // object -> validation coordination. Publication takes build coordination
    // -> validation coordination but releases validation before opening the
    // immutable object. GC takes the namespace exclusively and only probes all
    // subordinate locks nonblocking, so no path can invert this edge.
    let record_path = prepared_validation_record_path(root, header.kind, header.key);
    if let Some(digest) =
        read_prepared_validation_record(&record_path, header, identity, data_offset)?
    {
        return Ok(digest);
    }

    let lock_path = prepared_validation_lock_path(root, header.kind, header.key);
    let payload_len =
        usize::try_from(header.payload_len).context("prepared payload length exceeds usize")?;
    let padded_payload_len = round_up(payload_len, data_offset)?;
    let mut wait_for_successor = false;
    loop {
        let mut coordination = open_coord_file(root, &lock_path)?;
        let election = if wait_for_successor {
            coordination.lock_exclusive()
        } else {
            coordination.try_lock_exclusive()
        };
        match election {
            Ok(()) => {
                coordination.release_namespace_gate();
                if let Some(digest) =
                    read_prepared_validation_record(&record_path, header, identity, data_offset)?
                {
                    return Ok(digest);
                }
                let digest = hash_prepared_payload(
                    file,
                    identity,
                    data_offset,
                    payload_len,
                    padded_payload_len,
                )?;
                // Publish the observed digest even when it does not match the
                // header. Every peer then rejects the same corrupt immutable
                // inode in O(1) rather than serially streaming it again.
                publish_prepared_validation_record(root, header, identity, data_offset, digest)?;
                return Ok(digest);
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                coordination.lock_shared().with_context(|| {
                    format!(
                        "wait for prepared-object validation {}",
                        record_path.display()
                    )
                })?;
                coordination.release_namespace_gate();
                if let Some(digest) =
                    read_prepared_validation_record(&record_path, header, identity, data_offset)?
                {
                    return Ok(digest);
                }
                wait_for_successor = true;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "elect prepared-object payload verifier {}",
                        record_path.display()
                    )
                });
            }
        }
    }
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
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error).context("pread prepared object header");
        }
        anyhow::ensure!(read != 0, "prepared object header is truncated");
        done += read as usize;
    }
    PreparedObjectHeader::decode(&bytes)
}

fn validate_open_prepared_object(
    root: &Path,
    path: &Path,
    file: File,
    expected: PreparedObjectExpectation,
) -> Result<PreparedObject> {
    let metadata = file
        .metadata()
        .with_context(|| format!("stat prepared initrd object {}", path.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "prepared initrd object is not a regular file: {}",
        path.display()
    );
    anyhow::ensure!(
        metadata.permissions().mode() & 0o222 == 0,
        "prepared initrd object is writable rather than immutable: {}",
        path.display()
    );
    flock_retry(&file, rustix::fs::FlockOperation::LockShared)
        .with_context(|| format!("lock prepared initrd object {}", path.display()))?;
    let header = read_header_at(&file)?;
    expected.validate(header)?;
    let data_offset = usize::try_from(header.file_alignment)
        .context("prepared object data offset exceeds usize")?;
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
        usize::try_from(metadata.len()).context("prepared object file too large")?;
    anyhow::ensure!(
        actual_file_len == expected_file_len,
        "prepared object file length {actual_file_len} != {expected_file_len}"
    );
    let identity = StableFileIdentity::from_metadata(&metadata);
    let validation =
        cached_prepared_payload_validation(root, &file, header, identity, data_offset)?;
    anyhow::ensure!(
        validation.payload_hash == header.payload_hash,
        "prepared object payload checksum mismatch: {}",
        path.display()
    );
    anyhow::ensure!(
        validation.padding_is_zero,
        "prepared object mapped padding is not zero: {}",
        path.display()
    );
    Ok(PreparedObject {
        fd: Some(file.into()),
        header,
        data_offset,
        cache_hit: true,
    })
}

fn try_open_prepared_object(
    root: &Path,
    path: &Path,
    expected: PreparedObjectExpectation,
) -> Result<Option<PreparedObject>> {
    try_open_prepared_object_with_open_hook(root, path, expected, || {})
}

fn try_open_prepared_object_with_open_hook(
    root: &Path,
    path: &Path,
    expected: PreparedObjectExpectation,
    after_open: impl FnOnce(),
) -> Result<Option<PreparedObject>> {
    // Close the object open→LOCK_SH race with GC. GC holds this gate
    // exclusively while it takes the per-key/object locks and unlinks. Once
    // validation below acquires object LOCK_SH, that object lock itself
    // protects the inode and the namespace gate can be released.
    let namespace_gate = open_namespace_gate(root)?;
    flock_retry(&namespace_gate, rustix::fs::FlockOperation::LockShared)
        .context("lock prepared initrd object namespace")?;
    let result = try_open_prepared_object_under_namespace(root, path, expected, after_open);
    drop(namespace_gate);
    result
}

/// Open and lock an object while the caller already holds the namespace gate
/// shared. Coordination paths use this form to preserve the global lock order:
/// namespace gate -> per-key build coordination -> immutable object ->
/// validation coordination.
fn try_open_prepared_object_under_namespace(
    root: &Path,
    path: &Path,
    expected: PreparedObjectExpectation,
    after_open: impl FnOnce(),
) -> Result<Option<PreparedObject>> {
    let file = match OpenOptions::new()
        .read(true)
        // O_NONBLOCK is ignored for regular files and prevents a malformed
        // cache entry from hanging the opener on a FIFO before the fstat
        // regular-file gate can reject it.
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("open prepared initrd object {}", path.display()));
        }
    };
    after_open();
    validate_open_prepared_object(root, path, file, expected).map(Some)
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
    let file_alignment =
        usize::try_from(built.header.file_alignment).context("file alignment exceeds usize")?;
    anyhow::ensure!(
        file_alignment >= PREPARED_CAS_HEADER_LEN && file_alignment.is_power_of_two(),
        "invalid prepared object data alignment"
    );
    let padded_payload_len = round_up(built.payload.len(), file_alignment)?;
    let file_len = file_alignment
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
        .seek(SeekFrom::Start(built.header.file_alignment))
        .context("seek prepared object payload")?;
    // Preserve complete all-zero host pages as holes. Boundary stitches are
    // one guest granule wide but commonly contain only a few KiB of a tiny
    // per-cell tail; eagerly writing the zero suffix made every cell consume
    // another physical 2 MiB despite those bytes never carrying content.
    for chunk in built.payload.chunks(file_alignment) {
        if chunk.iter().all(|byte| *byte == 0) {
            temp.as_file_mut()
                .seek(SeekFrom::Current(i64::try_from(chunk.len())?))
                .context("seek across sparse prepared-object payload page")?;
        } else {
            temp.write_all(chunk)
                .context("write prepared object payload page")?;
        }
    }
    #[cfg(unix)]
    {
        temp.as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o444))
            .context("mark prepared object read-only")?;
    }

    // Readers use this separate lock only when the tiny validation memo is
    // missing or names an older inode revision. Taking it before rename closes
    // the publish→memo window: a reader that sees the final pathname waits for
    // this lock and then consumes the publisher's already-known digest rather
    // than streaming a hundreds-of-MiB object itself.
    let validation_lock = prepared_validation_lock_path(root, built.header.kind, built.header.key);
    let mut validation = open_coord_file(root, &validation_lock)?;
    validation
        .lock_exclusive()
        .context("lock prepared-object validation publication")?;

    let published = temp
        .persist(final_path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish prepared object {}", final_path.display()))?;
    let identity = StableFileIdentity::from_file(&published)
        .context("stat published prepared object for validation memo")?;
    publish_prepared_validation_record(
        root,
        built.header,
        identity,
        file_alignment,
        PreparedPayloadValidation {
            payload_hash: built.header.payload_hash,
            padding_is_zero: true,
        },
    )?;
    Ok(())
}

#[cfg(test)]
fn note_prepared_object_waiter_for_test() -> Result<()> {
    if std::env::var("KTSTR_PREPARED_CHILD_MODE").as_deref() != Ok("wait-killed-builder") {
        return Ok(());
    }
    let ready = std::env::var_os("KTSTR_PREPARED_CHILD_READY")
        .map(PathBuf::from)
        .context("waiter child has no ready directory")?;
    let index = std::env::var("KTSTR_PREPARED_CHILD_INDEX").context("waiter child has no index")?;
    std::fs::write(ready.join(index), b"waiting")
        .context("publish prepared-object waiter state")?;
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
    if let Some(opened) =
        try_open_prepared_object(root, &final_path, expected).with_context(|| {
            format!(
                "prepared initrd CAS object is corrupt (refusing copy fallback): {}",
                final_path.display()
            )
        })?
    {
        return Ok(opened);
    }

    let lock_path = prepared_object_lock_path(root, expected.kind, expected.key);
    let mut build = Some(build);
    let mut wait_for_successor = false;
    loop {
        let mut coordination = open_coord_file(root, &lock_path)?;
        let election = if wait_for_successor {
            coordination.lock_exclusive()
        } else {
            coordination.try_lock_exclusive()
        };
        match election {
            Ok(()) => {
                if let Some(opened) = try_open_prepared_object_under_namespace(
                    root,
                    &final_path,
                    expected,
                    || {},
                )
                .with_context(|| {
                    format!(
                        "prepared initrd CAS object is corrupt (refusing copy fallback): {}",
                        final_path.display()
                    )
                })? {
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
                        && built.header.mapping_granule == expected.mapping_granule
                        && built.header.file_alignment == expected.file_alignment,
                    "prepared object builder returned a different recipe"
                );
                expected.validate(built.header)?;
                publish_prepared_object(root, &final_path, &built)?;
                let mut opened =
                    try_open_prepared_object_under_namespace(root, &final_path, expected, || {})?
                        .context("published prepared object disappeared before open")?;
                opened.cache_hit = false;
                // `opened` takes LOCK_SH on the immutable object before the
                // coordination EX lock drops. GC takes the same per-key lock
                // and then object LOCK_EX, closing both publish/open and
                // live-mapping unlink races.
                return Ok(opened);
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                #[cfg(test)]
                note_prepared_object_waiter_for_test()?;
                coordination.lock_shared().with_context(|| {
                    format!("wait for prepared initrd object {}", final_path.display())
                })?;
                if let Some(opened) = try_open_prepared_object_under_namespace(
                    root,
                    &final_path,
                    expected,
                    || {},
                )
                .with_context(|| {
                    format!(
                        "prepared initrd CAS object is corrupt (refusing copy fallback): {}",
                        final_path.display()
                    )
                })? {
                    return Ok(opened);
                }
                // A killed/failed winner dropped EX without publishing.
                // Release our shared lock by dropping `coordination`, then
                // enter the kernel's blocking writer queue. This guarantees
                // progress even when a large reader herd survived the dead
                // winner.
                wait_for_successor = true;
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
    let bytes = read_fixed_cache_record::<8>(path, "prepared initrd GC stamp")
        .ok()
        .flatten()?;
    Some(u64::from_le_bytes(bytes))
}

fn publish_gc_stamp(root: &Path, path: &Path, now: u64) -> Result<()> {
    let mut temp = tempfile::Builder::new()
        .prefix(".tmp-prepared-gc-stamp-")
        .tempfile_in(root)
        .context("create prepared initrd GC stamp temp")?;
    temp.write_all(&now.to_le_bytes())
        .context("write prepared initrd GC stamp")?;
    temp.persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish prepared initrd GC stamp {}", path.display()))?;
    Ok(())
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

#[derive(Clone, Copy, Debug)]
enum GcMemoKind {
    Digest,
    Validation,
    Coverage,
    Closure,
}

impl GcMemoKind {
    fn directory(self) -> &'static str {
        match self {
            Self::Digest => PREPARED_DIGESTS_DIR,
            Self::Validation => PREPARED_DIGESTS_DIR,
            Self::Coverage => PREPARED_PROBES_DIR,
            Self::Closure => PREPARED_CLOSURES_DIR,
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Self::Digest => ".digest",
            Self::Validation => ".validation",
            Self::Coverage => ".coverage",
            Self::Closure => ".closure",
        }
    }

    fn temp_prefix(self) -> &'static str {
        match self {
            Self::Digest => ".tmp-digest-",
            Self::Validation => ".tmp-validation-",
            Self::Coverage => ".tmp-coverage-",
            Self::Closure => ".tmp-closure-",
        }
    }

    fn lock_name(self, key: u64) -> String {
        let prefix = match self {
            Self::Digest => "digest",
            Self::Validation => "validation",
            Self::Coverage => "coverage",
            Self::Closure => "closure",
        };
        format!("{prefix}-{key:016x}.lock")
    }
}

#[derive(Debug)]
struct GcMemoCandidate {
    path: PathBuf,
    kind: GcMemoKind,
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
        PreparedObjectKind::Boundary,
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
        PreparedObjectKind::Boundary,
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

fn parse_keyed_temp_filename(name: &str, prefix: &str) -> Option<u64> {
    let key_and_random = name.strip_prefix(prefix)?;
    let (key, random) = key_and_random.split_once('-')?;
    if key.len() != 16 || random.is_empty() {
        return None;
    }
    u64::from_str_radix(key, 16).ok()
}

struct GcNamespaceGuard {
    _file: File,
}

fn try_lock_gc_namespace(root: &Path) -> Result<Option<GcNamespaceGuard>> {
    let file = open_namespace_gate(root)?;
    match flock_retry(&file, rustix::fs::FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => Ok(Some(GcNamespaceGuard { _file: file })),
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => Ok(None),
        Err(error) => Err(error).context("lock prepared initrd coordination namespace for GC"),
    }
}

fn try_collect_temp(path: &Path, lock_path: &Path, _namespace: &GcNamespaceGuard) -> Result<bool> {
    let coordination = open_lock_file(lock_path)?;
    if flock_retry(
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

fn try_collect_object(
    root: &Path,
    candidate: &GcCandidate,
    _namespace: &GcNamespaceGuard,
) -> Result<bool> {
    let lock_path = prepared_object_lock_path(root, candidate.kind, candidate.key);
    let coord = open_lock_file(&lock_path)?;
    if flock_retry(&coord, rustix::fs::FlockOperation::NonBlockingLockExclusive).is_err() {
        return Ok(false);
    }
    let object = match File::open(&candidate.path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(error) => return Err(error.into()),
    };
    if flock_retry(
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

fn try_collect_memo(
    root: &Path,
    candidate: &GcMemoCandidate,
    namespace: &GcNamespaceGuard,
) -> Result<bool> {
    let lock_path = root
        .join(PREPARED_LOCKS_DIR)
        .join(candidate.kind.lock_name(candidate.key));
    try_collect_temp(&candidate.path, &lock_path, namespace)
}

fn metadata_modified_secs(metadata: &std::fs::Metadata) -> u64 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn metadata_allocated_bytes(metadata: &std::fs::Metadata) -> u64 {
    metadata.blocks().saturating_mul(512)
}

fn collect_memo_gc_candidates(
    root: &Path,
    kind: GcMemoKind,
    now: u64,
    total: &mut u64,
    candidates: &mut Vec<GcMemoCandidate>,
    namespace: &GcNamespaceGuard,
) -> Result<()> {
    for entry in std::fs::read_dir(root.join(kind.directory()))? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let metadata = entry.metadata()?;
        if !metadata.is_file() {
            continue;
        }
        let modified_secs = metadata_modified_secs(&metadata);
        let allocated_bytes = metadata_allocated_bytes(&metadata);
        if let Some(key) = parse_keyed_temp_filename(&name, kind.temp_prefix()) {
            let candidate = GcMemoCandidate {
                path: entry.path(),
                kind,
                key,
                modified_secs,
                len: allocated_bytes,
            };
            if now.saturating_sub(modified_secs) > PREPARED_CAS_GC_INTERVAL_SECS
                && try_collect_memo(root, &candidate, namespace)?
            {
                continue;
            }
            *total = total.saturating_add(allocated_bytes);
            continue;
        }
        let Some(key) = name
            .strip_suffix(kind.suffix())
            .filter(|key| key.len() == 16)
            .and_then(|key| u64::from_str_radix(key, 16).ok())
        else {
            continue;
        };
        *total = total.saturating_add(allocated_bytes);
        candidates.push(GcMemoCandidate {
            path: entry.path(),
            kind,
            key,
            modified_secs,
            len: allocated_bytes,
        });
    }
    Ok(())
}

fn sweep_idle_coordination_locks(root: &Path, _namespace: &GcNamespaceGuard) -> Result<usize> {
    let lock_dir = root.join(PREPARED_LOCKS_DIR);
    let mut removed = 0usize;
    for entry in std::fs::read_dir(&lock_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name == PREPARED_NAMESPACE_GATE
            || name == PREPARED_GC_LOCK
            || !name.ends_with(".lock")
            || !entry.file_type()?.is_file()
        {
            continue;
        }
        let path = entry.path();
        let file = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("open idle coordination lock {}", path.display()));
            }
        };
        match flock_retry(&file, rustix::fs::FlockOperation::NonBlockingLockExclusive) {
            Ok(()) => {}
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("probe coordination lock {}", path.display()));
            }
        }
        match std::fs::remove_file(&path) {
            Ok(()) => removed += 1,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("remove idle coordination lock {}", path.display()));
            }
        }
    }
    Ok(removed)
}

fn run_prepared_cache_gc_with_limit(
    root: &Path,
    now: u64,
    namespace: &GcNamespaceGuard,
    max_bytes: u64,
) -> Result<()> {
    let mut candidates = Vec::new();
    let mut total = 0u64;
    for entry in std::fs::read_dir(root.join(PREPARED_OBJECTS_DIR))? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let metadata = entry.metadata()?;
        let allocated_bytes = metadata_allocated_bytes(&metadata);
        if let Some((kind, key)) = parse_temp_object_filename(&name) {
            let modified_secs = metadata_modified_secs(&metadata);
            if now.saturating_sub(modified_secs) > PREPARED_CAS_GC_INTERVAL_SECS {
                let lock_path = prepared_object_lock_path(root, kind, key);
                if try_collect_temp(&entry.path(), &lock_path, namespace)? {
                    continue;
                }
            }
            total = total.saturating_add(allocated_bytes);
            continue;
        }
        let Some((kind, key)) = parse_object_filename(&name) else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let modified_secs = metadata_modified_secs(&metadata);
        total = total.saturating_add(allocated_bytes);
        candidates.push(GcCandidate {
            path: entry.path(),
            kind,
            key,
            modified_secs,
            len: allocated_bytes,
        });
    }
    let mut memo_candidates = Vec::new();
    for kind in [
        GcMemoKind::Digest,
        GcMemoKind::Validation,
        GcMemoKind::Coverage,
        GcMemoKind::Closure,
    ] {
        collect_memo_gc_candidates(root, kind, now, &mut total, &mut memo_candidates, namespace)?;
    }

    #[derive(Clone, Copy)]
    enum CandidateIndex {
        Object(usize),
        Memo(usize),
    }
    let mut order = Vec::with_capacity(candidates.len() + memo_candidates.len());
    order.extend((0..candidates.len()).map(CandidateIndex::Object));
    order.extend((0..memo_candidates.len()).map(CandidateIndex::Memo));
    order.sort_by_key(|candidate| match candidate {
        CandidateIndex::Object(index) => candidates[*index].modified_secs,
        CandidateIndex::Memo(index) => memo_candidates[*index].modified_secs,
    });
    for candidate in order {
        let (modified_secs, len) = match candidate {
            CandidateIndex::Object(index) => {
                (candidates[index].modified_secs, candidates[index].len)
            }
            CandidateIndex::Memo(index) => (
                memo_candidates[index].modified_secs,
                memo_candidates[index].len,
            ),
        };
        let too_old = now.saturating_sub(modified_secs) > PREPARED_CAS_MAX_AGE_SECS;
        let over_size = total > max_bytes;
        if !too_old && (!over_size || len == 0) {
            continue;
        }
        let removed = match candidate {
            CandidateIndex::Object(index) => {
                try_collect_object(root, &candidates[index], namespace)?
            }
            CandidateIndex::Memo(index) => {
                try_collect_memo(root, &memo_candidates[index], namespace)?
            }
        };
        if removed {
            total = total.saturating_sub(len);
        }
    }
    sweep_idle_coordination_locks(root, namespace)?;
    Ok(())
}

fn run_prepared_cache_gc(root: &Path, now: u64, namespace: &GcNamespaceGuard) -> Result<()> {
    run_prepared_cache_gc_with_limit(root, now, namespace, PREPARED_CAS_MAX_BYTES)
}

fn maybe_gc_prepared_cache(root: &Path) {
    let now = unix_now_secs();
    let gc_lock_path = root.join(PREPARED_LOCKS_DIR).join(PREPARED_GC_LOCK);
    let Ok(gc_lock) = open_lock_file(&gc_lock_path) else {
        return;
    };
    if flock_retry(
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
    let Ok(Some(namespace)) = try_lock_gc_namespace(root) else {
        return;
    };
    if let Err(error) = run_prepared_cache_gc(root, now, &namespace) {
        tracing::warn!(%error, "prepared initrd CAS GC failed");
    }
    if let Err(error) = publish_gc_stamp(root, &stamp_path, now) {
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
    fn pin(
        root: &Path,
        params: &initramfs::SuffixParams<'_>,
        payload: Option<PinnedInput>,
    ) -> Result<Self> {
        match (&payload, params.payload) {
            (Some(pinned), Some(path)) => {
                anyhow::ensure!(
                    pinned.display_path == path,
                    "prepared base payload path changed before initrd completion: {} != {}",
                    pinned.display_path.display(),
                    path.display()
                );
            }
            (None, None) => {}
            _ => anyhow::bail!(
                "prepared base and suffix disagree about whether an init payload is present"
            ),
        }
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
    mapping_granule: usize,
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
) -> AHasher {
    let mut hasher = fixed_hasher();
    hash_len_prefixed(&mut hasher, domain);
    hash_u32(&mut hasher, PREPARED_CAS_SCHEMA);
    hash_len_prefixed(&mut hasher, std::env::consts::ARCH.as_bytes());
    hash_u64(&mut hasher, mapping_granule as u64);
    hash_u64(&mut hasher, file_alignment as u64);
    hash_u8(&mut hasher, compression_tag(compression));
    hasher
}

fn base_recipe_key(
    base_key: &BaseKey,
    mapping_granule: usize,
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(
        b"ktstr-prepared-initrd-base",
        mapping_granule,
        file_alignment,
        compression,
    );
    hash_u64(&mut hasher, base_key.0);
    hasher.finish()
}

fn payload_recipe_key(
    payload: &PinnedInput,
    coverage: (bool, u64),
    mapping_granule: usize,
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(
        b"ktstr-prepared-initrd-payload",
        mapping_granule,
        file_alignment,
        compression,
    );
    hash_u64(&mut hasher, payload.identity.size);
    hash_u64(&mut hasher, payload.content_hash);
    hash_u8(&mut hasher, u8::from(coverage.0));
    hash_u64(&mut hasher, coverage.1);
    hasher.finish()
}

fn modules_recipe_key(
    modules: &[PinnedModule],
    mapping_granule: usize,
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(
        b"ktstr-prepared-initrd-modules",
        mapping_granule,
        file_alignment,
        compression,
    );
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
    mapping_granule: usize,
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(
        b"ktstr-prepared-initrd-part-view",
        mapping_granule,
        file_alignment,
        compression,
    );
    hash_u8(&mut hasher, kind as u8);
    hash_u64(&mut hasher, canonical_key);
    hash_u64(&mut hasher, leading_pad as u64);
    hasher.finish()
}

fn tail_recipe_key(
    prefix_uncompressed_len: usize,
    leading_pad: usize,
    mapping_granule: usize,
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
    params: &initramfs::SuffixParams<'_>,
) -> u64 {
    let mut hasher = recipe_prefix(
        b"ktstr-prepared-initrd-tail",
        mapping_granule,
        file_alignment,
        compression,
    );
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
    mapping_granule: usize,
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(
        b"ktstr-prepared-initrd-stitch",
        mapping_granule,
        file_alignment,
        compression,
    );
    hash_u64(&mut hasher, segments.len() as u64);
    for segment in segments {
        hash_u64(&mut hasher, parts[segment.part_index].object.header.key);
        hash_u64(&mut hasher, segment.part_offset as u64);
        hash_u64(&mut hasher, segment.page_offset as u64);
        hash_u64(&mut hasher, segment.len as u64);
    }
    hasher.finish()
}

fn boundary_underlay_recipe_key(
    stable_segments: &[PageSegment],
    parts: &[PreparedPart],
    mapping_granule: usize,
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(
        b"ktstr-prepared-initrd-boundary-underlay",
        mapping_granule,
        file_alignment,
        compression,
    );
    hash_u64(&mut hasher, stable_segments.len() as u64);
    for segment in stable_segments {
        hash_u64(&mut hasher, parts[segment.part_index].object.header.key);
        hash_u64(&mut hasher, segment.part_offset as u64);
        hash_u64(&mut hasher, segment.page_offset as u64);
        hash_u64(&mut hasher, segment.len as u64);
    }
    hasher.finish()
}

fn boundary_overlay_recipe_key(
    segments: &[PageSegment],
    parts: &[PreparedPart],
    overlay_start: usize,
    overlay_len: usize,
    mapping_granule: usize,
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
) -> u64 {
    let mut hasher = recipe_prefix(
        b"ktstr-prepared-initrd-boundary-overlay",
        mapping_granule,
        file_alignment,
        compression,
    );
    hash_u64(&mut hasher, overlay_start as u64);
    hash_u64(&mut hasher, overlay_len as u64);
    hash_u64(&mut hasher, segments.len() as u64);
    for segment in segments {
        hash_u64(&mut hasher, parts[segment.part_index].object.header.key);
        hash_u64(&mut hasher, segment.part_offset as u64);
        hash_u64(&mut hasher, segment.page_offset as u64);
        hash_u64(&mut hasher, segment.len as u64);
    }
    hasher.finish()
}

fn materialize_page_slice(
    parts: &[PreparedPart],
    segments: &[PageSegment],
    slice_start: usize,
    slice_len: usize,
) -> Result<Vec<u8>> {
    let slice_end = slice_start
        .checked_add(slice_len)
        .context("prepared page slice end overflow")?;
    let mut bytes = vec![0u8; slice_len];
    for segment in segments {
        let segment_end = segment
            .page_offset
            .checked_add(segment.len)
            .context("prepared page segment end overflow")?;
        let copy_start = slice_start.max(segment.page_offset);
        let copy_end = slice_end.min(segment_end);
        if copy_start >= copy_end {
            continue;
        }
        let source_part_offset = segment
            .part_offset
            .checked_add(copy_start - segment.page_offset)
            .context("prepared page-slice source offset overflow")?;
        let layout_offset = usize::try_from(parts[segment.part_index].object.header.leading_pad)?
            .checked_add(source_part_offset)
            .context("prepared page-slice layout offset overflow")?;
        let len = copy_end - copy_start;
        let source = parts[segment.part_index]
            .object
            .read_exact_at(layout_offset, len)?;
        let destination_start = copy_start - slice_start;
        bytes[destination_start..destination_start + len].copy_from_slice(&source);
    }
    Ok(bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PreparedRangePlan {
    pub(crate) part_count: usize,
    pub(crate) direct_ranges: usize,
    pub(crate) stitch_pages: usize,
    pub(crate) total_compressed_len: usize,
}

#[derive(Debug)]
pub(crate) struct PreparedOverlay {
    pub(crate) fd: OwnedFd,
    pub(crate) file_offset: u64,
    pub(crate) guest_offset: u64,
    pub(crate) map_len: usize,
}

#[derive(Debug)]
pub(crate) struct PreparedMapping {
    pub(crate) fd: OwnedFd,
    pub(crate) file_offset: u64,
    pub(crate) guest_offset: u64,
    pub(crate) map_len: usize,
    /// Host-page overlays installed after this mapping.
    ///
    /// A hugetlb-backed guest first needs one complete 2 MiB replacement to
    /// dissolve the original hugetlb VMA. A small boundary object can then
    /// replace only the host pages which contain the per-cell tail. Nesting
    /// those overlays under their complete primary range keeps the logical
    /// stream non-overlapping while preserving the exact mmap order.
    pub(crate) overlays: Vec<PreparedOverlay>,
}

#[derive(Debug)]
pub(crate) struct PreparedInitrd {
    uncompressed_len: usize,
    compressed_len: usize,
    ranges: Vec<PreparedMapping>,
    cache_hits: usize,
    plan: PreparedRangePlan,
    mapping_granule: usize,
    compression: initramfs::InitrdCompression,
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

    pub(crate) fn compression(&self) -> initramfs::InitrdCompression {
        self.compression
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
    mapping_granule: usize,
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
    // Keep the exact open file description used to derive the base's loader
    // closure alive through /init stripping. Reopening the pathname after KVM
    // setup could otherwise combine revision A's libraries with revision B's
    // executable.
    payload: Option<PinnedInput>,
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
        let includes: Vec<(&str, &Path, u32)> = self
            .includes
            .iter()
            .zip(&include_paths)
            .map(|(entry, path)| (entry.archive_name.as_str(), path.as_path(), entry.mode))
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

fn remove_explicit_dependency_collisions(
    extras: &[PinnedArchiveInput],
    includes: &[PinnedArchiveInput],
    shared_libs: &mut Vec<PinnedClosureEntry>,
) -> Result<()> {
    let mut normalized = Vec::with_capacity(shared_libs.len());
    for dependency in shared_libs.drain(..) {
        if let Some(extra) = extras
            .iter()
            .find(|entry| entry.archive_name == dependency.guest_path)
        {
            anyhow::bail!(
                "extra binary archive path '{}' collides with a resolved shared library: {} vs {}",
                dependency.guest_path,
                extra.input.display_path.display(),
                dependency.input.display_path.display()
            );
        }
        let Some(include) = includes
            .iter()
            .find(|entry| entry.archive_name == dependency.guest_path)
        else {
            normalized.push(dependency);
            continue;
        };
        anyhow::ensure!(
            include.input.identity.size == dependency.input.identity.size
                && include.input.content_hash == dependency.input.content_hash
                && include.mode == 0o100755,
            "explicit archive path '{}' conflicts with a resolved shared library: {} vs {}",
            dependency.guest_path,
            include.input.display_path.display(),
            dependency.input.display_path.display()
        );
        // A verbatim include with identical bytes and identical archive mode
        // is semantically the same entry. Extras are never collapsed here:
        // their debug-stripping transform can change bytes even when the
        // source digest matches the dependency.
    }
    *shared_libs = normalized;
    Ok(())
}

fn ensure_explicit_paths_disjoint(
    extras: &[PinnedArchiveInput],
    includes: &[PinnedArchiveInput],
) -> Result<()> {
    for extra in extras {
        anyhow::ensure!(
            !includes
                .iter()
                .any(|include| include.archive_name == extra.archive_name),
            "archive path '{}' is used by both an extra binary and include file",
            extra.archive_name
        );
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
    ensure_explicit_paths_disjoint(&extras, &includes)?;

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
    remove_explicit_dependency_collisions(&extras, &includes, &mut shared_libs)?;

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

#[derive(Clone, Copy)]
struct PreparedGeometry {
    mapping_granule: usize,
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
}

fn layout_compressed_part(compressed: Vec<u8>, leading_pad: usize) -> Result<Vec<u8>> {
    if leading_pad == 0 {
        return Ok(compressed);
    }
    let layout_len = leading_pad
        .checked_add(compressed.len())
        .context("prepared part layout length overflow")?;
    let mut layout = Vec::with_capacity(layout_len);
    layout.resize(leading_pad, 0);
    layout.extend_from_slice(&compressed);
    Ok(layout)
}

fn get_or_build_part<F>(
    root: &Path,
    kind: PreparedObjectKind,
    key: u64,
    leading_pad: usize,
    geometry: PreparedGeometry,
    build: F,
) -> Result<PreparedObject>
where
    F: FnOnce() -> Result<Vec<u8>>,
{
    let PreparedGeometry {
        mapping_granule,
        file_alignment,
        compression,
    } = geometry;
    anyhow::ensure!(
        kind != PreparedObjectKind::Base
            && kind != PreparedObjectKind::Stitch
            && kind != PreparedObjectKind::Boundary,
        "generic archive part has an invalid kind"
    );
    let expectation = PreparedObjectExpectation {
        kind,
        compression,
        key,
        mapping_granule: mapping_granule as u64,
        file_alignment: file_alignment as u64,
        leading_pad: Some(leading_pad as u64),
        parent_key: Some(key),
    };
    get_or_build_prepared_object(root, expectation, || {
        let uncompressed = build()?;
        let uncompressed_len = uncompressed.len();
        let compressed = initramfs::compress_initrd_part(compression, &uncompressed)?;
        drop(uncompressed);
        let compressed_len = compressed.len();
        let layout = layout_compressed_part(compressed, leading_pad)?;
        let header = PreparedObjectHeader {
            kind,
            compression,
            key,
            mapping_granule: mapping_granule as u64,
            payload_len: layout.len() as u64,
            payload_hash: content_hash_bytes(&layout),
            part_uncompressed_len: uncompressed_len as u64,
            part_compressed_len: compressed_len as u64,
            leading_pad: leading_pad as u64,
            stream_offset_mod: leading_pad as u64,
            file_alignment: file_alignment as u64,
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
    geometry: PreparedGeometry,
) -> Result<PreparedObject> {
    let PreparedGeometry {
        mapping_granule,
        file_alignment,
        compression,
    } = geometry;
    anyhow::ensure!(
        matches!(
            view_kind,
            PreparedObjectKind::PayloadView | PreparedObjectKind::ModulesView
        ) && leading_pad > 0
            && leading_pad < file_alignment,
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
        mapping_granule,
        file_alignment,
        compression,
    );
    let expectation = PreparedObjectExpectation {
        kind: view_kind,
        compression,
        key,
        mapping_granule: mapping_granule as u64,
        file_alignment: file_alignment as u64,
        leading_pad: Some(leading_pad as u64),
        parent_key: Some(canonical.header.key),
    };
    get_or_build_prepared_object(root, expectation, || {
        let compressed_len = usize::try_from(canonical.header.part_compressed_len)?;
        let layout_len = leading_pad
            .checked_add(compressed_len)
            .context("shifted prepared-part view length overflow")?;
        let mut layout = vec![0u8; layout_len];
        canonical.read_exact_into(0, &mut layout[leading_pad..])?;
        let header = PreparedObjectHeader {
            kind: view_kind,
            compression,
            key,
            mapping_granule: mapping_granule as u64,
            payload_len: layout.len() as u64,
            payload_hash: content_hash_bytes(&layout),
            part_uncompressed_len: canonical.header.part_uncompressed_len,
            part_compressed_len: canonical.header.part_compressed_len,
            leading_pad: leading_pad as u64,
            stream_offset_mod: leading_pad as u64,
            file_alignment: file_alignment as u64,
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
pub(crate) fn get_or_prepare_base(
    inputs: PreparedBaseInputs,
    compression: initramfs::InitrdCompression,
) -> Result<PreparedBase> {
    let root = prepared_cache_root()?;
    ensure_prepared_cache_dirs(&root)?;
    maybe_gc_prepared_cache(&root);
    let mapping_granule = PREPARED_MAPPING_GRANULE;
    let file_alignment = prepared_file_alignment()?;
    let base_key = base_recipe_key(inputs.key(), mapping_granule, file_alignment, compression);
    let expectation = PreparedObjectExpectation {
        kind: PreparedObjectKind::Base,
        compression,
        key: base_key,
        mapping_granule: mapping_granule as u64,
        file_alignment: file_alignment as u64,
        leading_pad: Some(0),
        parent_key: Some(base_key),
    };
    let object = get_or_build_prepared_object(&root, expectation, || {
        let uncompressed = inputs.build()?;
        let uncompressed_len = uncompressed.len();
        let compressed = initramfs::compress_initrd_part(compression, &uncompressed)?;
        drop(uncompressed);
        let header = PreparedObjectHeader {
            kind: PreparedObjectKind::Base,
            compression,
            key: base_key,
            mapping_granule: mapping_granule as u64,
            payload_len: compressed.len() as u64,
            payload_hash: content_hash_bytes(&compressed),
            part_uncompressed_len: uncompressed_len as u64,
            part_compressed_len: compressed.len() as u64,
            leading_pad: 0,
            stream_offset_mod: 0,
            file_alignment: file_alignment as u64,
            parent_key: base_key,
            reserved_len: 0,
        };
        Ok(BuiltPreparedObject {
            header,
            payload: compressed,
        })
    })?;
    let payload = inputs.payload;
    Ok(PreparedBase {
        root,
        object,
        mapping_granule,
        file_alignment,
        compression,
        payload: Some(payload),
    })
}

enum PlannedSource {
    Part(usize),
    Stitch(PreparedObject),
    Boundary {
        underlay: PreparedObject,
        overlay: PreparedObject,
        overlay_guest_offset: usize,
    },
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
    file_alignment: usize,
    compression: initramfs::InitrdCompression,
) -> Result<(Vec<PreparedMapping>, usize, usize, usize)> {
    anyhow::ensure!(
        file_alignment.is_power_of_two() && page_size.is_multiple_of(file_alignment),
        "prepared file alignment {file_alignment} does not divide mapping granule {page_size}"
    );
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
            let payload_len = usize::try_from(part.object.header.payload_len)
                .context("prepared part payload length exceeds usize")?;
            let padded_payload_len = round_up(payload_len, file_alignment)?;
            let mapping_end = layout_offset
                .checked_add(page_size)
                .context("prepared direct mapping end overflow")?;
            (segment.page_offset == 0
                && layout_offset % file_alignment == 0
                && mapping_end <= padded_payload_len)
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

        // The final page normally combines a large immutable part with the
        // tiny per-cell tail. Materializing that whole 2 MiB page under the
        // tail-derived key made every cell copy and publish the same large
        // prefix again. Keep the hugetlb-safe complete replacement, but key
        // it only from the stable prefix. After that primary mmap dissolves
        // any hugetlb VMA, install the smallest host-page-aligned boundary
        // object over the bytes which can actually differ.
        let page_content_len = total_compressed_len
            .saturating_sub(page_start)
            .min(page_size);
        let tail_position = segments.iter().position(|segment| {
            parts[segment.part_index].object.header.kind == PreparedObjectKind::Tail
        });
        if page_start
            .checked_add(page_size)
            .is_some_and(|page_end| page_end >= total_compressed_len)
            && let Some(tail_position) = tail_position
            && tail_position + 1 == segments.len()
        {
            let tail_segment = segments[tail_position];
            let overlay_start = tail_segment.page_offset / file_alignment * file_alignment;
            let overlay_end = round_up(page_content_len, file_alignment)?;
            let overlay_len = overlay_end
                .checked_sub(overlay_start)
                .context("prepared boundary overlay underflow")?;
            if overlay_len > 0 && overlay_len < page_size {
                let stable_segments = &segments[..tail_position];
                let underlay_key = boundary_underlay_recipe_key(
                    stable_segments,
                    parts,
                    page_size,
                    file_alignment,
                    compression,
                );
                let underlay_expectation = PreparedObjectExpectation {
                    kind: PreparedObjectKind::Stitch,
                    compression,
                    key: underlay_key,
                    mapping_granule: page_size as u64,
                    file_alignment: file_alignment as u64,
                    leading_pad: Some(0),
                    parent_key: Some(underlay_key),
                };
                let underlay = get_or_build_prepared_object(root, underlay_expectation, || {
                    let payload = materialize_page_slice(parts, stable_segments, 0, page_size)?;
                    Ok(BuiltPreparedObject {
                        header: PreparedObjectHeader {
                            kind: PreparedObjectKind::Stitch,
                            compression,
                            key: underlay_key,
                            mapping_granule: page_size as u64,
                            payload_len: page_size as u64,
                            payload_hash: content_hash_bytes(&payload),
                            part_uncompressed_len: 0,
                            part_compressed_len: page_size as u64,
                            leading_pad: 0,
                            stream_offset_mod: 0,
                            file_alignment: file_alignment as u64,
                            parent_key: underlay_key,
                            reserved_len: 0,
                        },
                        payload,
                    })
                })?;

                let overlay_key = boundary_overlay_recipe_key(
                    &segments,
                    parts,
                    overlay_start,
                    overlay_len,
                    page_size,
                    file_alignment,
                    compression,
                );
                let overlay_expectation = PreparedObjectExpectation {
                    kind: PreparedObjectKind::Boundary,
                    compression,
                    key: overlay_key,
                    mapping_granule: page_size as u64,
                    file_alignment: file_alignment as u64,
                    leading_pad: Some(0),
                    parent_key: Some(overlay_key),
                };
                let overlay = get_or_build_prepared_object(root, overlay_expectation, || {
                    let payload =
                        materialize_page_slice(parts, &segments, overlay_start, overlay_len)?;
                    Ok(BuiltPreparedObject {
                        header: PreparedObjectHeader {
                            kind: PreparedObjectKind::Boundary,
                            compression,
                            key: overlay_key,
                            mapping_granule: page_size as u64,
                            payload_len: overlay_len as u64,
                            payload_hash: content_hash_bytes(&payload),
                            part_uncompressed_len: 0,
                            part_compressed_len: overlay_len as u64,
                            leading_pad: 0,
                            stream_offset_mod: 0,
                            file_alignment: file_alignment as u64,
                            parent_key: overlay_key,
                            reserved_len: 0,
                        },
                        payload,
                    })
                })?;
                stitch_cache_hits += usize::from(underlay.cache_hit);
                stitch_cache_hits += usize::from(overlay.cache_hit);
                stitch_pages += 1;
                planned.push(PlannedMapping {
                    source: PlannedSource::Boundary {
                        underlay,
                        overlay,
                        overlay_guest_offset: page_start + overlay_start,
                    },
                    file_offset: 0,
                    guest_offset: page_start,
                    map_len: page_size,
                });
                continue;
            }
        }

        let stitch_key =
            stitch_recipe_key(&segments, parts, page_size, file_alignment, compression);
        let expectation = PreparedObjectExpectation {
            kind: PreparedObjectKind::Stitch,
            compression,
            key: stitch_key,
            mapping_granule: page_size as u64,
            file_alignment: file_alignment as u64,
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
                mapping_granule: page_size as u64,
                payload_len: page_size as u64,
                payload_hash: content_hash_bytes(&page),
                part_uncompressed_len: 0,
                part_compressed_len: page_size as u64,
                leading_pad: 0,
                stream_offset_mod: 0,
                file_alignment: file_alignment as u64,
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
        let (fd, file_offset, overlays) = match planned.source {
            PlannedSource::Part(index) => (
                parts[index]
                    .object
                    .fd
                    .take()
                    .context("prepared part fd was needed by multiple disjoint ranges")?,
                planned.file_offset,
                Vec::new(),
            ),
            PlannedSource::Stitch(mut stitch) => (
                stitch
                    .fd
                    .take()
                    .context("prepared stitch fd already consumed")?,
                planned.file_offset,
                Vec::new(),
            ),
            PlannedSource::Boundary {
                mut underlay,
                mut overlay,
                overlay_guest_offset,
            } => {
                let overlay_fd = overlay
                    .fd
                    .take()
                    .context("prepared boundary overlay fd already consumed")?;
                let overlay_len = usize::try_from(overlay.header.payload_len)?;
                (
                    underlay
                        .fd
                        .take()
                        .context("prepared boundary underlay fd already consumed")?,
                    underlay.data_offset,
                    vec![PreparedOverlay {
                        fd: overlay_fd,
                        file_offset: overlay.data_offset as u64,
                        guest_offset: overlay_guest_offset as u64,
                        map_len: overlay_len,
                    }],
                )
            }
        };
        mappings.push(PreparedMapping {
            fd,
            file_offset: file_offset as u64,
            guest_offset: planned.guest_offset as u64,
            map_len: planned.map_len,
            overlays,
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
/// by one part map its inode directly. Ordinary mixed-part pages use a single
/// immutable stitch; a final page containing the tail uses a shared stable
/// underlay plus a minimal tail-sensitive overlay.
pub(crate) fn complete_prepared_initrd(
    prepared_base: PreparedBase,
    params: &initramfs::SuffixParams<'_>,
) -> Result<PreparedInitrd> {
    let PreparedBase {
        root,
        object: base,
        mapping_granule,
        file_alignment,
        compression,
        payload,
    } = prepared_base;
    let geometry = PreparedGeometry {
        mapping_granule,
        file_alignment,
        compression,
    };
    let pinned = PinnedSuffixInputs::pin(&root, params, payload)?;
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
        let leading_pad = stream_offset % file_alignment;
        let key = payload_recipe_key(
            payload,
            coverage,
            mapping_granule,
            file_alignment,
            compression,
        );
        let payload_path = payload.proc_path();
        let canonical =
            get_or_build_part(&root, PreparedObjectKind::Payload, key, 0, geometry, || {
                let part = initramfs::build_payload_part_from_pinned(&payload_path)?;
                payload.verify_unchanged()?;
                Ok(part)
            })?;
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
                geometry,
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
        let leading_pad = stream_offset % file_alignment;
        let key = modules_recipe_key(
            &pinned.modules,
            mapping_granule,
            file_alignment,
            compression,
        );
        let module_sources = pinned.module_sources();
        let canonical =
            get_or_build_part(&root, PreparedObjectKind::Modules, key, 0, geometry, || {
                let part = initramfs::build_modules_part_from_pinned(&module_sources)?;
                pinned.verify_unchanged()?;
                Ok(part)
            })?;
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
                geometry,
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

    let leading_pad = stream_offset % file_alignment;
    let tail_key = tail_recipe_key(
        uncompressed_len,
        leading_pad,
        mapping_granule,
        file_alignment,
        compression,
        params,
    );
    let tail = get_or_build_part(
        &root,
        PreparedObjectKind::Tail,
        tail_key,
        leading_pad,
        geometry,
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
    let (ranges, direct_ranges, stitch_pages, stitch_cache_hits) = plan_prepared_mappings(
        &root,
        &mut parts,
        mapping_granule,
        file_alignment,
        compression,
    )?;
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
        mapping_granule,
        compression,
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
        // Cross two complete logical digest chunks and end on a short chunk.
        // The file reader must make the same Hasher::write calls as the
        // in-memory content path even if pread itself returns short.
        let data: Vec<u8> = (0..CONTENT_HASH_CHUNK_LEN * 2 + 173)
            .map(|i| (i % 251) as u8)
            .collect();
        std::fs::write(&f, &data).unwrap();
        let h = hash_file(&f).unwrap();
        assert_eq!(
            h,
            content_hash_bytes(&data),
            "file and in-memory content hashing must use identical fixed boundaries"
        );
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

    fn test_base_object(
        key: u64,
        mapping_granule: usize,
        file_alignment: usize,
        payload: Vec<u8>,
    ) -> BuiltPreparedObject {
        BuiltPreparedObject {
            header: PreparedObjectHeader {
                kind: PreparedObjectKind::Base,
                compression: initramfs::InitrdCompression::Lz4,
                key,
                mapping_granule: mapping_granule as u64,
                payload_len: payload.len() as u64,
                payload_hash: content_hash_bytes(&payload),
                part_uncompressed_len: payload.len() as u64,
                part_compressed_len: payload.len() as u64,
                leading_pad: 0,
                stream_offset_mod: 0,
                file_alignment: file_alignment as u64,
                parent_key: key,
                reserved_len: 0,
            },
            payload,
        }
    }

    fn test_base_expectation(
        key: u64,
        mapping_granule: usize,
        file_alignment: usize,
    ) -> PreparedObjectExpectation {
        PreparedObjectExpectation {
            kind: PreparedObjectKind::Base,
            compression: initramfs::InitrdCompression::Lz4,
            key,
            mapping_granule: mapping_granule as u64,
            file_alignment: file_alignment as u64,
            leading_pad: Some(0),
            parent_key: Some(key),
        }
    }

    fn test_prepared_part(
        root: &Path,
        kind: PreparedObjectKind,
        key: u64,
        mapping_granule: usize,
        file_alignment: usize,
        leading_pad: usize,
        compressed_len: usize,
    ) -> PreparedObject {
        let expectation = PreparedObjectExpectation {
            kind,
            compression: initramfs::InitrdCompression::Lz4,
            key,
            mapping_granule: mapping_granule as u64,
            file_alignment: file_alignment as u64,
            leading_pad: Some(leading_pad as u64),
            parent_key: Some(key),
        };
        get_or_build_prepared_object(root, expectation, || {
            let mut payload = vec![0; leading_pad];
            payload.resize(leading_pad + compressed_len, key as u8);
            Ok(BuiltPreparedObject {
                header: PreparedObjectHeader {
                    kind,
                    compression: initramfs::InitrdCompression::Lz4,
                    key,
                    mapping_granule: mapping_granule as u64,
                    payload_len: payload.len() as u64,
                    payload_hash: content_hash_bytes(&payload),
                    part_uncompressed_len: compressed_len as u64,
                    part_compressed_len: compressed_len as u64,
                    leading_pad: leading_pad as u64,
                    stream_offset_mod: leading_pad as u64,
                    file_alignment: file_alignment as u64,
                    parent_key: key,
                    reserved_len: 0,
                },
                payload,
            })
        })
        .unwrap()
    }

    #[test]
    fn zero_pad_compressed_layout_reuses_the_original_allocation() {
        let compressed: Vec<u8> = (0..8192).map(|index| (index % 251) as u8).collect();
        let allocation = compressed.as_ptr();
        let expected = compressed.clone();

        let layout = layout_compressed_part(compressed, 0).unwrap();

        assert_eq!(layout, expected);
        assert_eq!(
            layout.as_ptr(),
            allocation,
            "the canonical zero-pad path must move the compressed Vec instead of copying it"
        );
        assert_eq!(
            layout_compressed_part(vec![1, 2, 3], 4).unwrap(),
            vec![0, 0, 0, 0, 1, 2, 3],
            "the padded path must retain its exact wire layout"
        );
    }

    #[test]
    fn shifted_part_view_preserves_padding_body_and_cache_identity() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let mapping_granule = PREPARED_MAPPING_GRANULE;
        let file_alignment = 4096;
        let leading_pad = 3072;
        let compressed_len = 8193;
        let canonical_key = 0xa5;
        let canonical = test_prepared_part(
            temp.path(),
            PreparedObjectKind::Payload,
            canonical_key,
            mapping_granule,
            file_alignment,
            0,
            compressed_len,
        );
        let geometry = PreparedGeometry {
            mapping_granule,
            file_alignment,
            compression: initramfs::InitrdCompression::Lz4,
        };

        let first = get_or_build_part_view(
            temp.path(),
            &canonical,
            PreparedObjectKind::PayloadView,
            leading_pad,
            geometry,
        )
        .unwrap();
        assert!(!first.cache_hit);
        let layout = first
            .read_exact_at(0, leading_pad + compressed_len)
            .unwrap();
        assert!(layout[..leading_pad].iter().all(|byte| *byte == 0));
        assert!(
            layout[leading_pad..]
                .iter()
                .all(|byte| *byte == canonical_key as u8)
        );
        assert_eq!(first.header.payload_hash, content_hash_bytes(&layout));
        drop(first);

        let second = get_or_build_part_view(
            temp.path(),
            &canonical,
            PreparedObjectKind::PayloadView,
            leading_pad,
            geometry,
        )
        .unwrap();
        assert!(
            second.cache_hit,
            "the direct-read view must retain the existing recipe and cache identity"
        );
        assert_eq!(
            second
                .read_exact_at(0, leading_pad + compressed_len)
                .unwrap(),
            layout
        );
    }

    fn test_prepared_base(
        root: &Path,
        key: u64,
        compression: initramfs::InitrdCompression,
        uncompressed: &[u8],
    ) -> PreparedBase {
        let mapping_granule = PREPARED_MAPPING_GRANULE;
        let file_alignment = prepared_file_alignment().unwrap();
        let compressed = initramfs::compress_initrd_part(compression, uncompressed).unwrap();
        let expectation = PreparedObjectExpectation {
            kind: PreparedObjectKind::Base,
            compression,
            key,
            mapping_granule: mapping_granule as u64,
            file_alignment: file_alignment as u64,
            leading_pad: Some(0),
            parent_key: Some(key),
        };
        let object = get_or_build_prepared_object(root, expectation, || {
            Ok(BuiltPreparedObject {
                header: PreparedObjectHeader {
                    kind: PreparedObjectKind::Base,
                    compression,
                    key,
                    mapping_granule: mapping_granule as u64,
                    payload_len: compressed.len() as u64,
                    payload_hash: content_hash_bytes(&compressed),
                    part_uncompressed_len: uncompressed.len() as u64,
                    part_compressed_len: compressed.len() as u64,
                    leading_pad: 0,
                    stream_offset_mod: 0,
                    file_alignment: file_alignment as u64,
                    parent_key: key,
                    reserved_len: 0,
                },
                payload: compressed,
            })
        })
        .unwrap();
        PreparedBase {
            root: root.to_path_buf(),
            object,
            mapping_granule,
            file_alignment,
            compression,
            payload: None,
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TestBoundaryIdentity {
        underlay_device: u64,
        underlay_inode: u64,
        underlay_key: u64,
        boundary_device: u64,
        boundary_inode: u64,
        boundary_key: u64,
        boundary_len: usize,
    }

    fn inspect_boundary_mapping(
        prepared: &PreparedInitrd,
        file_alignment: usize,
    ) -> TestBoundaryIdentity {
        let boundary_ranges: Vec<_> = prepared
            .ranges
            .iter()
            .filter(|range| !range.overlays.is_empty())
            .collect();
        assert_eq!(
            boundary_ranges.len(),
            1,
            "a tiny final tail must require exactly one boundary range"
        );
        let range = boundary_ranges[0];
        assert_eq!(
            range.overlays.len(),
            1,
            "a boundary range must carry exactly one minimal overlay"
        );
        assert_eq!(range.map_len, PREPARED_MAPPING_GRANULE);

        let underlay_file = File::from(range.fd.try_clone().unwrap());
        let underlay_header = read_header_at(&underlay_file).unwrap();
        let underlay_metadata = underlay_file.metadata().unwrap();
        assert_eq!(underlay_header.kind, PreparedObjectKind::Stitch);
        assert_eq!(
            underlay_header.payload_len, PREPARED_MAPPING_GRANULE as u64,
            "the stable underlay must still dissolve a complete hugetlb mapping"
        );

        let overlay = &range.overlays[0];
        let overlay_file = File::from(overlay.fd.try_clone().unwrap());
        let overlay_header = read_header_at(&overlay_file).unwrap();
        let overlay_metadata = overlay_file.metadata().unwrap();
        assert_eq!(overlay_header.kind, PreparedObjectKind::Boundary);
        assert_eq!(overlay_header.payload_len, overlay.map_len as u64);
        assert!(
            overlay.map_len > 0,
            "a boundary overlay must never be empty"
        );
        assert!(
            overlay.map_len < PREPARED_MAPPING_GRANULE,
            "a boundary overlay must remain smaller than its 2 MiB underlay"
        );
        assert_eq!(
            overlay.map_len % file_alignment,
            0,
            "boundary length must be host-page aligned"
        );
        assert_eq!(
            overlay.guest_offset % file_alignment as u64,
            0,
            "boundary guest offset must be host-page aligned"
        );
        assert_eq!(
            overlay.file_offset % file_alignment as u64,
            0,
            "boundary file offset must be host-page aligned"
        );
        assert!(overlay.guest_offset >= range.guest_offset);
        assert!(
            overlay.guest_offset + overlay.map_len as u64
                <= range.guest_offset + range.map_len as u64,
            "the minimal overlay must be contained by its complete underlay"
        );

        TestBoundaryIdentity {
            underlay_device: underlay_metadata.dev(),
            underlay_inode: underlay_metadata.ino(),
            underlay_key: underlay_header.key,
            boundary_device: overlay_metadata.dev(),
            boundary_inode: overlay_metadata.ino(),
            boundary_key: overlay_header.key,
            boundary_len: overlay.map_len,
        }
    }

    fn read_prepared_stream(prepared: PreparedInitrd) -> Vec<u8> {
        let compressed_len = prepared.compressed_len();
        let mut stream = Vec::new();
        for range in prepared.into_ranges() {
            let PreparedMapping {
                fd,
                file_offset,
                guest_offset,
                map_len,
                overlays,
            } = range;
            let file = File::from(fd);
            let mut bytes = vec![0u8; map_len];
            let mut offset = 0usize;
            while offset < bytes.len() {
                let read = file
                    .read_at(&mut bytes[offset..], file_offset + offset as u64)
                    .unwrap();
                assert!(read > 0);
                offset += read;
            }
            for overlay in overlays {
                let overlay_offset = usize::try_from(overlay.guest_offset - guest_offset).unwrap();
                let overlay_file = File::from(overlay.fd);
                let mut done = 0usize;
                while done < overlay.map_len {
                    let read = overlay_file
                        .read_at(
                            &mut bytes[overlay_offset + done..overlay_offset + overlay.map_len],
                            overlay.file_offset + done as u64,
                        )
                        .unwrap();
                    assert!(read > 0);
                    done += read;
                }
            }
            stream.extend_from_slice(&bytes);
        }
        stream.truncate(compressed_len);
        stream
    }

    fn decompress_prepared_stream(
        compression: initramfs::InitrdCompression,
        stream: &[u8],
    ) -> Vec<u8> {
        use std::io::Read as _;

        match compression {
            initramfs::InitrdCompression::Lz4 => {
                let mut output = Vec::new();
                let mut offset = 0usize;
                while offset < stream.len() {
                    assert_eq!(stream[offset..offset + 4], initramfs::LZ4_LEGACY_MAGIC);
                    offset += 4;
                    while offset < stream.len()
                        && stream[offset..offset + 4] != initramfs::LZ4_LEGACY_MAGIC
                    {
                        let chunk_len =
                            u32::from_le_bytes(stream[offset..offset + 4].try_into().unwrap())
                                as usize;
                        offset += 4;
                        let end = offset + chunk_len;
                        output.extend(
                            lz4_flex::block::decompress(&stream[offset..end], 8 << 20).unwrap(),
                        );
                        offset = end;
                    }
                }
                output
            }
            initramfs::InitrdCompression::Zstd => {
                zstd::stream::decode_all(std::io::Cursor::new(stream)).unwrap()
            }
            initramfs::InitrdCompression::Gzip => {
                let mut output = Vec::new();
                flate2::read::MultiGzDecoder::new(std::io::Cursor::new(stream))
                    .read_to_end(&mut output)
                    .unwrap();
                output
            }
            initramfs::InitrdCompression::Uncompressed => stream.to_vec(),
        }
    }

    #[test]
    fn prepared_stream_reconstructs_every_compression_and_empty_base() {
        let compressions = [
            initramfs::InitrdCompression::Lz4,
            initramfs::InitrdCompression::Zstd,
            initramfs::InitrdCompression::Gzip,
            initramfs::InitrdCompression::Uncompressed,
        ];
        let mut state = 0x1234_5678_9abc_def0u64;
        let incompressible: Vec<u8> = (0..(3 << 20))
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();

        for (compression_index, compression) in compressions.into_iter().enumerate() {
            for (case_index, base_bytes) in [incompressible.as_slice(), &[] as &[u8]]
                .into_iter()
                .enumerate()
            {
                let temp = tempfile::tempdir().unwrap();
                ensure_prepared_cache_dirs(temp.path()).unwrap();
                let key = 0x9000 + (compression_index * 2 + case_index) as u64;
                let prepared_base = test_prepared_base(temp.path(), key, compression, base_bytes);
                let params = initramfs::SuffixParams::default();
                let mut expected = base_bytes.to_vec();
                expected.extend(initramfs::build_dynamic_tail(base_bytes.len(), &params).unwrap());
                let prepared = complete_prepared_initrd(prepared_base, &params).unwrap();
                assert_eq!(prepared.uncompressed_len(), expected.len());
                let stream = read_prepared_stream(prepared);
                assert_eq!(
                    decompress_prepared_stream(compression, &stream),
                    expected,
                    "{compression:?} prepared stream did not reconstruct"
                );
            }
        }
    }

    #[test]
    fn distinct_tails_share_boundary_underlay_but_not_minimal_overlay() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let compression = initramfs::InitrdCompression::Uncompressed;
        let file_alignment = prepared_file_alignment().unwrap();
        let mut identities = Vec::new();

        for value in ["short-tail", "a-distinct-and-slightly-longer-tail"] {
            let args = vec![value.to_owned()];
            let params = initramfs::SuffixParams {
                args: &args,
                ..Default::default()
            };
            let expected = initramfs::build_dynamic_tail(0, &params).unwrap();
            let prepared_base = test_prepared_base(temp.path(), 0xb001_1000, compression, &[]);
            let prepared = complete_prepared_initrd(prepared_base, &params).unwrap();
            let identity = inspect_boundary_mapping(&prepared, file_alignment);
            assert_eq!(
                identity.boundary_len,
                round_up(expected.len(), file_alignment).unwrap(),
                "the tail-sensitive overlay must contain only the touched host pages"
            );
            assert_eq!(
                read_prepared_stream(prepared),
                expected,
                "underlay plus minimal overlay must reconstruct the exact tail"
            );
            identities.push(identity);
        }

        assert_eq!(
            (
                identities[0].underlay_device,
                identities[0].underlay_inode,
                identities[0].underlay_key,
            ),
            (
                identities[1].underlay_device,
                identities[1].underlay_inode,
                identities[1].underlay_key,
            ),
            "different tails must reuse the identical stable underlay object"
        );
        assert_ne!(
            identities[0].boundary_key, identities[1].boundary_key,
            "boundary identity must include the tail content"
        );
        assert_ne!(
            (identities[0].boundary_device, identities[0].boundary_inode,),
            (identities[1].boundary_device, identities[1].boundary_inode,),
            "different boundary keys must publish distinct immutable inodes"
        );
    }

    #[test]
    fn prepared_completion_keeps_the_base_payload_open_description() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let payload_path = temp.path().join("payload");
        std::fs::copy("/bin/true", &payload_path).unwrap();

        let pinned = pin_input(temp.path(), &payload_path).unwrap();
        let old_identity = pinned.identity;
        let expected_payload_part =
            initramfs::build_payload_part_from_pinned(&pinned.proc_path()).unwrap();

        let replacement = temp.path().join("replacement");
        std::fs::write(&replacement, b"revision B must never become /init").unwrap();
        std::fs::rename(&replacement, &payload_path).unwrap();
        assert_ne!(
            StableFileIdentity::from_metadata(&std::fs::metadata(&payload_path).unwrap()),
            old_identity,
            "the pathname replacement must exercise a genuinely different inode"
        );

        let mut prepared_base = test_prepared_base(
            temp.path(),
            0xb001,
            initramfs::InitrdCompression::Uncompressed,
            &[],
        );
        prepared_base.payload = Some(pinned);
        let params = initramfs::SuffixParams {
            payload: Some(&payload_path),
            ..Default::default()
        };
        let prepared = complete_prepared_initrd(prepared_base, &params).unwrap();
        let archive = read_prepared_stream(prepared);
        assert!(
            archive.starts_with(&expected_payload_part),
            "completion must strip and archive the exact payload fd pinned during base preparation"
        );
        assert!(
            !archive
                .windows(b"revision B must never become /init".len())
                .any(|window| window == b"revision B must never become /init"),
            "the replacement pathname revision leaked into the prepared image"
        );
    }

    #[test]
    fn many_distinct_tails_are_host_page_aligned_and_sparse() {
        const CELLS: usize = 52;
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let compression = initramfs::InitrdCompression::Uncompressed;
        let file_alignment = prepared_file_alignment().unwrap();
        let mut underlay_identities = std::collections::HashSet::new();
        let mut boundary_identities = std::collections::HashSet::new();
        let mut total_boundary_bytes = 0u64;

        for index in 0..CELLS {
            let args = vec![format!("cell-{index:02}-has-distinct-tail-content")];
            let params = initramfs::SuffixParams {
                args: &args,
                ..Default::default()
            };
            let prepared_base = test_prepared_base(temp.path(), 0xb002, compression, &[]);
            let expected = (index == 0).then(|| initramfs::build_dynamic_tail(0, &params).unwrap());
            let prepared = complete_prepared_initrd(prepared_base, &params).unwrap();
            let identity = inspect_boundary_mapping(&prepared, file_alignment);
            underlay_identities.insert((
                identity.underlay_device,
                identity.underlay_inode,
                identity.underlay_key,
            ));
            boundary_identities.insert((
                identity.boundary_device,
                identity.boundary_inode,
                identity.boundary_key,
            ));
            total_boundary_bytes += identity.boundary_len as u64;
            if let Some(expected) = expected {
                assert_eq!(
                    read_prepared_stream(prepared),
                    expected,
                    "sparse storage must preserve the exact directly mapped stream"
                );
            }
        }
        assert_eq!(
            underlay_identities.len(),
            1,
            "all {CELLS} tails must reuse one stable 2 MiB underlay inode and key"
        );
        assert_eq!(
            boundary_identities.len(),
            CELLS,
            "every distinct tail must retain its own tail-sensitive boundary identity"
        );
        assert!(
            total_boundary_bytes <= CELLS as u64 * 2 * file_alignment as u64,
            "{CELLS} tiny tails consumed {total_boundary_bytes} bytes of boundary overlays"
        );

        let mut logical_bytes = 0u64;
        let mut allocated_bytes = 0u64;
        let mut stitch_objects = 0usize;
        let mut boundary_objects = 0usize;
        for entry in std::fs::read_dir(temp.path().join(PREPARED_OBJECTS_DIR)).unwrap() {
            let entry = entry.unwrap();
            let metadata = entry.metadata().unwrap();
            if metadata.is_file() {
                logical_bytes += metadata.len();
                allocated_bytes += metadata.blocks() * 512;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            match parse_object_filename(&name).map(|(kind, _)| kind) {
                Some(PreparedObjectKind::Stitch) => stitch_objects += 1,
                Some(PreparedObjectKind::Boundary) => boundary_objects += 1,
                _ => {}
            }
        }
        assert_eq!(
            stitch_objects, 1,
            "stable final-page bytes must occupy one shared underlay object"
        );
        assert_eq!(boundary_objects, CELLS);

        let compact_upper_bound = PREPARED_MAPPING_GRANULE as u64
            + (CELLS as u64) * 4 * file_alignment as u64
            + 8 * file_alignment as u64;
        assert!(
            logical_bytes <= compact_upper_bound,
            "{CELLS} tiny tails consumed {logical_bytes} logical bytes, above the \
             host-page-aligned bound {compact_upper_bound}"
        );
        assert!(
            allocated_bytes <= compact_upper_bound,
            "{CELLS} tiny tails physically consumed {allocated_bytes} bytes, above the \
             reduced unique-storage bound {compact_upper_bound}"
        );
    }

    #[test]
    fn prepared_planner_uses_host_page_file_alignment_not_mapping_granule() {
        let page = PREPARED_MAPPING_GRANULE;
        let first_len = 0x3_2345;

        for (index, file_alignment) in [0x1000, 0x4000, 0x1_0000, page].into_iter().enumerate() {
            let temp = tempfile::tempdir().unwrap();
            ensure_prepared_cache_dirs(temp.path()).unwrap();
            let leading_pad = first_len % file_alignment;
            let second_len = page * 2 - first_len;
            let mut parts = vec![
                PreparedPart {
                    object: test_prepared_part(
                        temp.path(),
                        PreparedObjectKind::Payload,
                        0x100 + index as u64,
                        page,
                        file_alignment,
                        0,
                        first_len,
                    ),
                    stream_offset: 0,
                    uncompressed_len: first_len,
                    compressed_len: first_len,
                },
                PreparedPart {
                    object: test_prepared_part(
                        temp.path(),
                        PreparedObjectKind::PayloadView,
                        0x200 + index as u64,
                        page,
                        file_alignment,
                        leading_pad,
                        second_len,
                    ),
                    stream_offset: first_len,
                    uncompressed_len: second_len,
                    compressed_len: second_len,
                },
            ];

            let (ranges, direct_ranges, stitch_pages, _) = plan_prepared_mappings(
                temp.path(),
                &mut parts,
                page,
                file_alignment,
                initramfs::InitrdCompression::Lz4,
            )
            .unwrap();
            assert_eq!(stitch_pages, 1);
            assert_eq!(direct_ranges, 1);
            assert_eq!(ranges.len(), 2);
            assert_eq!(ranges[1].guest_offset, page as u64);
            assert_eq!(ranges[1].file_offset % file_alignment as u64, 0);
            assert!(
                leading_pad < file_alignment,
                "shifted views must be bounded by the host page, not 2 MiB"
            );
            if file_alignment < page {
                assert_ne!(
                    ranges[1].file_offset % page as u64,
                    0,
                    "ordinary-file direct ranges should not require 2 MiB file offsets"
                );
            }
            for range in &ranges {
                let file_len = rustix::fs::fstat(&range.fd).unwrap().st_size as u64;
                assert!(
                    range.file_offset + range.map_len as u64 <= file_len,
                    "every planned mapping must be wholly inside its backing file"
                );
            }
        }
    }

    #[test]
    fn prepared_planner_stitches_shifted_final_page_before_eof() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let page = PREPARED_MAPPING_GRANULE;
        let file_alignment = 4096;
        let first_len = page - 1024;
        let leading_pad = first_len % file_alignment;
        assert_eq!(leading_pad, 3072);
        let mut parts = vec![
            PreparedPart {
                object: test_prepared_part(
                    temp.path(),
                    PreparedObjectKind::Payload,
                    0xc001,
                    page,
                    file_alignment,
                    0,
                    first_len,
                ),
                stream_offset: 0,
                uncompressed_len: first_len,
                compressed_len: first_len,
            },
            PreparedPart {
                object: test_prepared_part(
                    temp.path(),
                    PreparedObjectKind::PayloadView,
                    0xc002,
                    page,
                    file_alignment,
                    leading_pad,
                    10 << 10,
                ),
                stream_offset: first_len,
                uncompressed_len: 10 << 10,
                compressed_len: 10 << 10,
            },
        ];
        let (ranges, direct_ranges, stitch_pages, _) = plan_prepared_mappings(
            temp.path(),
            &mut parts,
            page,
            file_alignment,
            initramfs::InitrdCompression::Lz4,
        )
        .unwrap();
        assert_eq!(direct_ranges, 0);
        assert_eq!(stitch_pages, 2);
        for range in ranges {
            let file_len = rustix::fs::fstat(&range.fd).unwrap().st_size as u64;
            assert!(range.file_offset + range.map_len as u64 <= file_len);
        }
    }

    #[test]
    fn prepared_planner_stitches_three_and_four_parts_inside_one_page_exactly() {
        let page = PREPARED_MAPPING_GRANULE;
        let file_alignment = 4096;
        let lengths = [3001usize, 5003, 7007, 9001];
        for part_count in [3usize, 4] {
            let temp = tempfile::tempdir().unwrap();
            ensure_prepared_cache_dirs(temp.path()).unwrap();
            let mut stream_offset = 0usize;
            let mut expected = Vec::new();
            let mut parts = Vec::new();
            for (index, compressed_len) in lengths[..part_count].iter().copied().enumerate() {
                let key = 0xd151 + index as u64;
                let leading_pad = stream_offset % file_alignment;
                parts.push(PreparedPart {
                    object: test_prepared_part(
                        temp.path(),
                        PreparedObjectKind::Payload,
                        key,
                        page,
                        file_alignment,
                        leading_pad,
                        compressed_len,
                    ),
                    stream_offset,
                    uncompressed_len: compressed_len,
                    compressed_len,
                });
                expected.resize(expected.len() + compressed_len, key as u8);
                stream_offset += compressed_len;
            }
            assert!(stream_offset < page);

            let (ranges, direct_ranges, stitch_pages, _) = plan_prepared_mappings(
                temp.path(),
                &mut parts,
                page,
                file_alignment,
                initramfs::InitrdCompression::Lz4,
            )
            .unwrap();
            assert_eq!(direct_ranges, 0);
            assert_eq!(stitch_pages, 1);
            assert_eq!(ranges.len(), 1);
            let range = ranges.into_iter().next().unwrap();
            let file = File::from(range.fd);
            let mut actual = vec![0u8; range.map_len];
            let mut done = 0usize;
            while done < actual.len() {
                let read = file
                    .read_at(&mut actual[done..], range.file_offset + done as u64)
                    .unwrap();
                assert!(read > 0);
                done += read;
            }
            assert_eq!(&actual[..expected.len()], expected);
            assert!(
                actual[expected.len()..].iter().all(|byte| *byte == 0),
                "the remainder of the {part_count}-part stitched final page \
                 must be zero padding"
            );
        }
    }

    #[test]
    fn prepared_planner_rejects_gap_overlap_and_reordered_parts() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let page = PREPARED_MAPPING_GRANULE;
        let file_alignment = 4096;
        let part_len = 4096usize;

        for (case_index, (name, offsets)) in [
            ("gap", [0, part_len + 1]),
            ("overlap", [0, part_len - 1]),
            ("reordering", [part_len, 0]),
        ]
        .into_iter()
        .enumerate()
        {
            let mut parts = offsets
                .into_iter()
                .enumerate()
                .map(|(index, stream_offset)| {
                    let key = 0xd200 + (case_index * 16 + index) as u64;
                    PreparedPart {
                        object: test_prepared_part(
                            temp.path(),
                            PreparedObjectKind::Payload,
                            key,
                            page,
                            file_alignment,
                            stream_offset % file_alignment,
                            part_len,
                        ),
                        stream_offset,
                        uncompressed_len: part_len,
                        compressed_len: part_len,
                    }
                })
                .collect::<Vec<_>>();
            let error = plan_prepared_mappings(
                temp.path(),
                &mut parts,
                page,
                file_alignment,
                initramfs::InitrdCompression::Lz4,
            )
            .unwrap_err();
            assert!(
                format!("{error:#}").contains("stream offsets are not contiguous"),
                "{name} was not rejected by the planner's contiguous-stream gate: {error:#}"
            );
        }
    }

    #[test]
    fn prepared_object_rejects_torn_and_header_corruption() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let key = 0x4142;
        let path = prepared_object_path(temp.path(), PreparedObjectKind::Base, key);
        std::fs::write(&path, b"torn").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
        assert!(
            try_open_prepared_object(temp.path(), &path, test_base_expectation(key, 4096, 4096),)
                .is_err()
        );

        let built = test_base_object(key, 4096, 4096, vec![7; 5000]);
        std::fs::remove_file(&path).unwrap();
        publish_prepared_object(temp.path(), &path, &built).unwrap();
        let mut corrupted = std::fs::read(&path).unwrap();
        corrupted[24..32].copy_from_slice(&8192u64.to_le_bytes());
        // Published CAS objects are intentionally 0444. Replace this private
        // test fixture through the directory rather than weakening the
        // object's permissions or expecting a writable open to succeed.
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, corrupted).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
        assert!(
            try_open_prepared_object(temp.path(), &path, test_base_expectation(key, 4096, 4096),)
                .is_err(),
            "header checksum must reject metadata corruption in O(1)"
        );
    }

    #[test]
    fn prepared_object_requires_regular_read_only_nofollow_backing() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let key = 0x4143;
        let expected = test_base_expectation(key, 4096, 4096);
        let path = prepared_object_path(temp.path(), PreparedObjectKind::Base, key);
        publish_prepared_object(
            temp.path(),
            &path,
            &test_base_object(key, 4096, 4096, vec![0x31; 5000]),
        )
        .unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o444,
            "publication must expose an immutable backing inode"
        );

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let error = try_open_prepared_object(temp.path(), &path, expected).unwrap_err();
        assert!(
            format!("{error:#}").contains("writable rather than immutable"),
            "owner-writable prepared objects must fail closed: {error:#}"
        );
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();

        let symlink = temp.path().join("prepared-symlink");
        std::os::unix::fs::symlink(&path, &symlink).unwrap();
        assert!(
            try_open_prepared_object(temp.path(), &symlink, expected).is_err(),
            "O_NOFOLLOW must reject a substituted prepared-object symlink"
        );

        let directory = temp.path().join("prepared-directory");
        std::fs::create_dir(&directory).unwrap();
        let error = try_open_prepared_object(temp.path(), &directory, expected).unwrap_err();
        assert!(
            format!("{error:#}").contains("not a regular file"),
            "non-regular prepared objects must fail closed: {error:#}"
        );

        let fifo = temp.path().join("prepared-fifo");
        let fifo_c = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
        assert_eq!(
            unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o444) },
            0,
            "create prepared-object FIFO fixture: {}",
            std::io::Error::last_os_error()
        );
        let error = try_open_prepared_object(temp.path(), &fifo, expected).unwrap_err();
        assert!(
            format!("{error:#}").contains("not a regular file"),
            "a FIFO substitution must be rejected without blocking: {error:#}"
        );
    }

    #[test]
    fn prepared_object_same_length_payload_corruption_is_memoized_and_rejected() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let key = 0x4144;
        let expected = test_base_expectation(key, 4096, 4096);
        let path = prepared_object_path(temp.path(), PreparedObjectKind::Base, key);
        publish_prepared_object(
            temp.path(),
            &path,
            &test_base_object(key, 4096, 4096, vec![0x52; 8193]),
        )
        .unwrap();

        // Keep the published inode open so its replacement cannot reuse the
        // same inode number even on a filesystem with aggressive recycling.
        let old_inode = File::open(&path).unwrap();
        let mut corrupted = std::fs::read(&path).unwrap();
        corrupted[4096 + 137] ^= 0x80;
        let original_len = corrupted.len();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, corrupted).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len() as usize,
            original_len
        );

        let first_error = try_open_prepared_object(temp.path(), &path, expected).unwrap_err();
        assert!(
            format!("{first_error:#}").contains("payload checksum mismatch"),
            "same-length payload corruption must be detected by streaming validation: \
             {first_error:#}"
        );
        let validation_path =
            prepared_validation_record_path(temp.path(), PreparedObjectKind::Base, key);
        let memo_after_first_rejection = std::fs::read(&validation_path).unwrap();
        let memo_inode = File::open(&validation_path).unwrap();
        let memo_identity = StableFileIdentity::from_file(&memo_inode).unwrap();

        let second_error = try_open_prepared_object(temp.path(), &path, expected).unwrap_err();
        assert!(
            format!("{second_error:#}").contains("payload checksum mismatch"),
            "the memoized corrupt inode must remain fail-closed: {second_error:#}"
        );
        assert_eq!(
            std::fs::read(&validation_path).unwrap(),
            memo_after_first_rejection,
            "a stable corrupt inode must be rejected from its O(1) memo without rehashing"
        );
        assert_eq!(
            StableFileIdentity::from_metadata(&std::fs::metadata(&validation_path).unwrap()),
            memo_identity,
            "the second rejection must not atomically republish an identical memo"
        );
        drop(old_inode);
    }

    #[test]
    fn prepared_object_rejects_nonzero_mapped_padding() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let key = 0x4146;
        let expected = test_base_expectation(key, 4096, 4096);
        let path = prepared_object_path(temp.path(), PreparedObjectKind::Base, key);
        let payload_len = 5000usize;
        publish_prepared_object(
            temp.path(),
            &path,
            &test_base_object(key, 4096, 4096, vec![0x72; payload_len]),
        )
        .unwrap();

        let old_inode = File::open(&path).unwrap();
        let mut corrupted = std::fs::read(&path).unwrap();
        corrupted[4096 + payload_len + 37] = 0xa5;
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, corrupted).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();

        let error = try_open_prepared_object(temp.path(), &path, expected).unwrap_err();
        assert!(
            format!("{error:#}").contains("mapped padding is not zero"),
            "all bytes exposed by a prepared mapping must be validated: {error:#}"
        );
        drop(old_inode);
    }

    #[test]
    fn prepared_validation_record_corruption_fails_closed() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let key = 0x4145;
        let expected = test_base_expectation(key, 4096, 4096);
        let path = prepared_object_path(temp.path(), PreparedObjectKind::Base, key);
        publish_prepared_object(
            temp.path(),
            &path,
            &test_base_object(key, 4096, 4096, vec![0x62; 8193]),
        )
        .unwrap();
        let validation_path =
            prepared_validation_record_path(temp.path(), PreparedObjectKind::Base, key);
        let mut record = std::fs::read(&validation_path).unwrap();
        record[96] ^= 1;
        std::fs::write(&validation_path, record).unwrap();

        let error = try_open_prepared_object(temp.path(), &path, expected).unwrap_err();
        assert!(
            format!("{error:#}").contains("validation memo checksum mismatch"),
            "a corrupt integrity memo must never authorize a cache hit: {error:#}"
        );
    }

    #[test]
    fn fixed_cache_record_reader_rejects_unsafe_or_wrong_sized_paths_promptly() {
        const RECORD_LEN: usize = 8;

        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("missing");
        assert!(
            read_fixed_cache_record::<RECORD_LEN>(&missing, "test cache record")
                .unwrap()
                .is_none(),
            "a missing record is a cache miss"
        );

        let valid = temp.path().join("valid");
        let expected = *b"12345678";
        std::fs::write(&valid, expected).unwrap();
        assert_eq!(
            read_fixed_cache_record::<RECORD_LEN>(&valid, "test cache record")
                .unwrap()
                .unwrap(),
            expected
        );

        let symlink = temp.path().join("symlink");
        std::os::unix::fs::symlink(&valid, &symlink).unwrap();
        assert!(
            read_fixed_cache_record::<RECORD_LEN>(&symlink, "test cache record").is_err(),
            "O_NOFOLLOW must reject a record symlink even when its target is valid"
        );

        let directory = temp.path().join("directory");
        std::fs::create_dir(&directory).unwrap();
        let error =
            read_fixed_cache_record::<RECORD_LEN>(&directory, "test cache record").unwrap_err();
        assert!(
            format!("{error:#}").contains("not a regular file"),
            "a record directory must fail closed: {error:#}"
        );

        let short = temp.path().join("short");
        std::fs::write(&short, vec![0u8; RECORD_LEN - 1]).unwrap();
        let error = read_fixed_cache_record::<RECORD_LEN>(&short, "test cache record").unwrap_err();
        assert!(
            format!("{error:#}").contains("invalid length"),
            "an N-1 record must be rejected before reading: {error:#}"
        );

        let long = temp.path().join("long");
        std::fs::write(&long, vec![0u8; RECORD_LEN + 1]).unwrap();
        let error = read_fixed_cache_record::<RECORD_LEN>(&long, "test cache record").unwrap_err();
        assert!(
            format!("{error:#}").contains("invalid length"),
            "an N+1 record must be rejected before reading: {error:#}"
        );

        let fifo = temp.path().join("fifo");
        let fifo_c = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
        assert_eq!(
            unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) },
            0,
            "create cache-record FIFO fixture: {}",
            std::io::Error::last_os_error()
        );
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let fifo_for_reader = fifo.clone();
        let reader = std::thread::spawn(move || {
            sender
                .send(read_fixed_cache_record::<RECORD_LEN>(
                    &fifo_for_reader,
                    "test cache record",
                ))
                .unwrap();
        });
        let result = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("O_NONBLOCK cache-record FIFO rejection must return promptly");
        reader.join().unwrap();
        let error = result.unwrap_err();
        assert!(
            format!("{error:#}").contains("not a regular file"),
            "a record FIFO must fail closed: {error:#}"
        );
    }

    #[test]
    fn oversized_sparse_closure_record_is_rejected_before_payload_read() {
        let root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(root.path()).unwrap();
        let key = 0xabc0_5678;
        let path = root
            .path()
            .join(PREPARED_CLOSURES_DIR)
            .join(format!("{key:016x}.closure"));
        let file = File::create(&path).unwrap();
        file.set_len((CLOSURE_RECORD_MAX_LEN + 1) as u64).unwrap();
        drop(file);

        let error = read_pinned_closure_record(&path, key).unwrap_err();
        assert!(
            format!("{error:#}").contains("envelope exceeds"),
            "the fstat length gate must reject an oversized sparse record before allocating or \
             parsing its payload: {error:#}"
        );
    }

    #[test]
    fn gc_stamp_reader_rejects_invalid_paths_promptly_and_publish_replaces_them() {
        let root = tempfile::tempdir().unwrap();
        let stamp = root.path().join(PREPARED_GC_STAMP);

        publish_gc_stamp(root.path(), &stamp, 11).unwrap();
        assert_eq!(read_gc_stamp(&stamp), Some(11));

        let target = root.path().join("symlink-target");
        let target_bytes = 0xfeed_face_cafe_beefu64.to_le_bytes();
        std::fs::write(&target, target_bytes).unwrap();
        std::fs::remove_file(&stamp).unwrap();
        std::os::unix::fs::symlink(&target, &stamp).unwrap();
        assert_eq!(
            read_gc_stamp(&stamp),
            None,
            "O_NOFOLLOW must reject a substituted GC stamp symlink"
        );
        publish_gc_stamp(root.path(), &stamp, 22).unwrap();
        assert!(
            std::fs::symlink_metadata(&stamp)
                .unwrap()
                .file_type()
                .is_file(),
            "atomic publication must replace the symlink with a regular inode"
        );
        assert_eq!(read_gc_stamp(&stamp), Some(22));
        assert_eq!(
            std::fs::read(&target).unwrap(),
            target_bytes,
            "GC stamp publication must not write through the substituted symlink"
        );

        std::fs::remove_file(&stamp).unwrap();
        let fifo_c = std::ffi::CString::new(stamp.to_str().unwrap()).unwrap();
        assert_eq!(
            unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) },
            0,
            "create GC-stamp FIFO fixture: {}",
            std::io::Error::last_os_error()
        );
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let fifo_for_reader = stamp.clone();
        let reader = std::thread::spawn(move || {
            sender.send(read_gc_stamp(&fifo_for_reader)).unwrap();
        });
        assert_eq!(
            receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("O_NONBLOCK GC-stamp FIFO rejection must return promptly"),
            None
        );
        reader.join().unwrap();
        publish_gc_stamp(root.path(), &stamp, 33).unwrap();
        assert_eq!(
            read_gc_stamp(&stamp),
            Some(33),
            "atomic publication must replace a FIFO without opening it"
        );

        std::fs::remove_file(&stamp).unwrap();
        std::fs::create_dir(&stamp).unwrap();
        assert_eq!(
            read_gc_stamp(&stamp),
            None,
            "a GC stamp directory is stale rather than readable metadata"
        );
    }

    #[test]
    fn prepared_validation_record_rejects_nonregular_or_wrong_sized_entries() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let key = 0x4147;
        let expected = test_base_expectation(key, 4096, 4096);
        let path = prepared_object_path(temp.path(), PreparedObjectKind::Base, key);
        publish_prepared_object(
            temp.path(),
            &path,
            &test_base_object(key, 4096, 4096, vec![0x73; 8193]),
        )
        .unwrap();
        let validation_path =
            prepared_validation_record_path(temp.path(), PreparedObjectKind::Base, key);

        std::fs::remove_file(&validation_path).unwrap();
        let fifo_c = std::ffi::CString::new(validation_path.to_str().unwrap()).unwrap();
        assert_eq!(
            unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) },
            0,
            "create validation-memo FIFO fixture: {}",
            std::io::Error::last_os_error()
        );
        let error = try_open_prepared_object(temp.path(), &path, expected).unwrap_err();
        assert!(
            format!("{error:#}").contains("validation memo is not a regular file"),
            "a memo FIFO must fail closed without blocking: {error:#}"
        );

        std::fs::remove_file(&validation_path).unwrap();
        std::fs::write(
            &validation_path,
            vec![0u8; PREPARED_VALIDATION_RECORD_LEN + 1],
        )
        .unwrap();
        let error = try_open_prepared_object(temp.path(), &path, expected).unwrap_err();
        assert!(
            format!("{error:#}").contains("validation memo has invalid length"),
            "a wrong-sized memo must be rejected before it is read: {error:#}"
        );
    }

    #[test]
    fn closure_key_reuses_identical_root_and_loader_cache_replacements() {
        let root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(root.path()).unwrap();
        let cwd = std::fs::canonicalize(root.path()).unwrap();
        let binary_path = root.path().join("payload");
        let cache_path = root.path().join("ld.so.cache");
        std::fs::write(&binary_path, b"identical payload bytes").unwrap();
        std::fs::write(&cache_path, b"identical loader cache bytes").unwrap();
        let old_binary = pin_input(root.path(), &binary_path).unwrap();
        let old_cache = pin_input(root.path(), &cache_path).unwrap();
        let old_loader = PinnedLoaderInputs {
            cwd: cwd.clone(),
            ld_library_path_present: false,
            ld_library_path_raw: Vec::new(),
            ld_library_path_dirs: Vec::new(),
            ld_so_cache: Some(old_cache),
        };
        let old_key = closure_recipe_key(&old_binary, &binary_path, &old_loader);

        let replacement_binary = root.path().join("payload.new");
        let replacement_cache = root.path().join("ld.so.cache.new");
        std::fs::write(&replacement_binary, b"identical payload bytes").unwrap();
        std::fs::write(&replacement_cache, b"identical loader cache bytes").unwrap();
        std::fs::rename(&replacement_binary, &binary_path).unwrap();
        std::fs::rename(&replacement_cache, &cache_path).unwrap();
        let new_binary = pin_input(root.path(), &binary_path).unwrap();
        let new_cache = pin_input(root.path(), &cache_path).unwrap();
        assert_ne!(
            (old_binary.identity.dev, old_binary.identity.ino),
            (new_binary.identity.dev, new_binary.identity.ino),
            "fixture must replace the root inode"
        );
        assert_ne!(
            (
                old_loader.ld_so_cache.as_ref().unwrap().identity.dev,
                old_loader.ld_so_cache.as_ref().unwrap().identity.ino,
            ),
            (new_cache.identity.dev, new_cache.identity.ino),
            "fixture must replace the loader-cache inode"
        );
        let new_loader = PinnedLoaderInputs {
            cwd,
            ld_library_path_present: false,
            ld_library_path_raw: Vec::new(),
            ld_library_path_dirs: Vec::new(),
            ld_so_cache: Some(new_cache),
        };
        assert_eq!(
            old_key,
            closure_recipe_key(&new_binary, &binary_path, &new_loader),
            "content-identical inode replacement must retain the closure key"
        );
    }

    #[test]
    fn closure_record_checksum_rejects_one_byte_corruption() {
        let root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(root.path()).unwrap();
        let key = 0xabc0_1234;
        let path = root
            .path()
            .join(PREPARED_CLOSURES_DIR)
            .join(format!("{key:016x}.closure"));
        let record = ClosureRecord {
            entries: Vec::new(),
            search_paths: vec![ClosureSearchRecord {
                path: root.path().join("missing"),
                identity: None,
            }],
        };
        let mut bytes = encode_closure_record(&record, key).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        assert!(read_pinned_closure_record(&path, key).unwrap().is_some());
        let last = bytes.len() - 1;
        bytes[last] ^= 0x40;
        std::fs::write(&path, bytes).unwrap();
        assert!(
            read_pinned_closure_record(&path, key).is_err(),
            "a decodable postcard payload must still fail its envelope checksum"
        );
    }

    #[test]
    fn closure_record_decode_is_exact_and_count_bounded_before_allocation() {
        let record = ClosureRecord {
            entries: Vec::new(),
            search_paths: Vec::new(),
        };
        let mut payload = postcard::to_stdvec(&record).unwrap();
        assert!(
            decode_closure_record(&payload).is_ok(),
            "the bounded wire representation must remain compatible with records we publish"
        );
        payload.push(0);
        let error = decode_closure_record(&payload).unwrap_err();
        assert!(
            format!("{error:#}").contains("trailing bytes"),
            "valid postcard followed by unconsumed bytes must fail closed: {error:#}"
        );

        let excessive_count = CLOSURE_RECORD_MAX_ENTRY_COUNT + 1;
        let mut count_payload = postcard::to_stdvec(&excessive_count).unwrap();
        // Postcard reports a sequence size hint only when at least one input
        // byte remains per claimed element. Supply that cheap hostile shape:
        // the bounded visitor must reject the count before decoding or
        // reserving storage for any entry.
        count_payload.resize(count_payload.len() + excessive_count, 0);
        assert!(
            decode_closure_record(&count_payload).is_err(),
            "an excessive declared entry count must fail before element allocation"
        );
    }

    #[test]
    fn closure_record_encode_enforces_the_decode_limits() {
        let oversized_entry_path = "x".repeat(CLOSURE_RECORD_MAX_ENTRY_PATH_LEN + 1);
        let entry_error = encode_closure_record(
            &ClosureRecord {
                entries: vec![ClosureEntryRecord {
                    guest_path: oversized_entry_path,
                    host_path: PathBuf::from("/valid/host/path"),
                    identity: StableFileIdentity {
                        dev: 0,
                        ino: 0,
                        size: 0,
                        mtime_secs: 0,
                        mtime_nsecs: 0,
                        ctime_secs: 0,
                        ctime_nsecs: 0,
                    },
                    content_hash: 0,
                }],
                search_paths: Vec::new(),
            },
            1,
        )
        .unwrap_err();
        assert!(
            format!("{entry_error:#}").contains("guest path exceeds"),
            "the writer must not publish a record its bounded decoder rejects: {entry_error:#}"
        );

        let oversized_search_path = "x".repeat(CLOSURE_RECORD_MAX_SEARCH_PATH_LEN + 1);
        let search_error = encode_closure_record(
            &ClosureRecord {
                entries: Vec::new(),
                search_paths: vec![ClosureSearchRecord {
                    path: PathBuf::from(oversized_search_path),
                    identity: None,
                }],
            },
            2,
        )
        .unwrap_err();
        assert!(
            format!("{search_error:#}").contains("search path exceeds"),
            "search-path limits must be identical on encode and decode: {search_error:#}"
        );
    }

    #[test]
    fn shared_digest_record_checksum_fails_closed() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("input");
        std::fs::write(&path, b"digest checksum fixture").unwrap();
        let file = File::open(&path).unwrap();
        let identity = StableFileIdentity::from_file(&file).unwrap();
        crate::cache::content::cached_file_digest_at_root(root.path(), &file, identity).unwrap();
        let current_key = crate::cache::content::file_digest_identity_key(identity);
        let record_path = root
            .path()
            .join("digests-v1")
            .join(format!("{current_key:016x}.digest"));
        let mut bytes = std::fs::read(&record_path).unwrap();
        bytes[72] ^= 1;
        std::fs::write(&record_path, bytes).unwrap();
        assert!(
            crate::cache::content::cached_file_digest_at_root(root.path(), &file, identity)
                .is_err(),
            "digest corruption must not silently key downstream objects"
        );
    }

    #[test]
    fn stable_owned_read_reports_midstream_truncate_without_sigbus() {
        let root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(root.path()).unwrap();
        let path = root.path().join("large-payload");
        std::fs::write(&path, vec![0x5a; 3 << 20]).unwrap();
        let input = pin_input(root.path(), &path).unwrap();
        let mut truncated = false;
        let result = input.read_owned_stable_with_hook(|offset| {
            if !truncated && offset >= 1 << 20 {
                OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_len(0)
                    .unwrap();
                truncated = true;
            }
        });
        assert!(result.is_err(), "mid-read truncate must be a normal error");
    }

    #[test]
    fn pinned_input_elf_probe_handles_short_files_symlinks_and_fifos() {
        let root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(root.path()).unwrap();

        let short = root.path().join("short");
        std::fs::write(&short, b"\x7fEL").unwrap();
        let short_input = pin_input(root.path(), &short).unwrap();
        assert!(
            !short_input.is_elf().unwrap(),
            "a short ELF prefix is an ordinary non-ELF input"
        );

        let elf = root.path().join("elf");
        let elf_symlink = root.path().join("elf-link");
        std::fs::write(&elf, b"\x7fELFfixture").unwrap();
        std::os::unix::fs::symlink(&elf, &elf_symlink).unwrap();
        let symlink_input = pin_input(root.path(), &elf_symlink).unwrap();
        assert!(
            symlink_input.is_elf().unwrap(),
            "the nonblocking opener must preserve intentional symlink following"
        );

        let fifo = root.path().join("fifo");
        let fifo_c = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
        assert_eq!(
            unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) },
            0,
            "create pinned-input FIFO fixture: {}",
            std::io::Error::last_os_error()
        );
        let cache_root = root.path().to_path_buf();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let result = pin_input(&cache_root, &fifo).map_err(|error| format!("{error:#}"));
            sender.send(result).unwrap();
        });
        let error = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("O_NONBLOCK pinned-input FIFO rejection must return promptly")
            .unwrap_err();
        worker.join().unwrap();
        assert!(
            error.contains("not a regular file"),
            "a FIFO input must fail the regular-file identity gate: {error}"
        );

        let record = ClosureEntryRecord {
            guest_path: "lib64/libfifo.so".to_owned(),
            host_path: root.path().join("fifo"),
            identity: short_input.identity,
            content_hash: short_input.content_hash,
        };
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let result = open_recorded_closure_entry(&record).map_err(|error| format!("{error:#}"));
            sender.send(result).unwrap();
        });
        let error = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("O_NONBLOCK recorded-dependency FIFO rejection must return promptly")
            .unwrap_err();
        worker.join().unwrap();
        assert!(
            error.contains("not a regular file"),
            "a recorded FIFO dependency must fail the regular-file identity gate: {error}"
        );
    }

    fn pinned_closure_entry(root: &Path, guest_path: &str, host_path: &Path) -> PinnedClosureEntry {
        PinnedClosureEntry {
            guest_path: guest_path.to_owned(),
            input: pin_input(root, host_path).unwrap(),
        }
    }

    fn pinned_archive_entry(
        root: &Path,
        archive_name: &str,
        host_path: &Path,
        mode: u32,
    ) -> PinnedArchiveInput {
        PinnedArchiveInput {
            archive_name: archive_name.to_owned(),
            input: pin_input(root, host_path).unwrap(),
            mode,
        }
    }

    fn newc_entry_mode(archive: &[u8], target: &str) -> u32 {
        let mut offset = 0usize;
        while offset
            .checked_add(110)
            .is_some_and(|header_end| header_end <= archive.len())
        {
            let header = &archive[offset..offset + 110];
            assert_eq!(&header[..6], b"070701", "invalid newc header");
            let field = |range: std::ops::Range<usize>| {
                u32::from_str_radix(std::str::from_utf8(&header[range]).unwrap(), 16).unwrap()
            };
            let mode = field(14..22);
            let file_size = field(54..62) as usize;
            let name_size = field(94..102) as usize;
            let name_start = offset + 110;
            let name_end = name_start + name_size;
            assert!(name_size > 0 && name_end <= archive.len());
            let name = std::str::from_utf8(&archive[name_start..name_end - 1]).unwrap();
            if name == target {
                return mode;
            }
            let data_start = (name_end + 3) & !3;
            offset = (data_start + file_size + 3) & !3;
        }
        panic!("newc archive has no entry named {target}");
    }

    #[test]
    fn prepared_base_build_uses_the_include_mode_captured_by_its_key() {
        let root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(root.path()).unwrap();
        let payload_path = root.path().join("payload");
        let include_path = root.path().join("include");
        std::fs::write(&payload_path, b"not an ELF payload").unwrap();
        std::fs::write(&include_path, b"verbatim include bytes").unwrap();
        let mut permissions = std::fs::metadata(&include_path).unwrap().permissions();
        permissions.set_mode(0o644);
        std::fs::set_permissions(&include_path, permissions).unwrap();
        let captured_mode = std::fs::metadata(&include_path)
            .unwrap()
            .permissions()
            .mode();

        let payload = pin_input(root.path(), &payload_path).unwrap();
        let include = pinned_archive_entry(
            root.path(),
            "include-files/mode-fixture",
            &include_path,
            captured_mode,
        );
        let key = prepared_base_semantic_key(&[], std::slice::from_ref(&include), &[], None);
        let changed_mode_include = pinned_archive_entry(
            root.path(),
            "include-files/mode-fixture",
            &include_path,
            captured_mode | 0o111,
        );
        let changed_mode_key =
            prepared_base_semantic_key(&[], std::slice::from_ref(&changed_mode_include), &[], None);
        assert_ne!(
            key, changed_mode_key,
            "the captured include mode must participate in the semantic key"
        );
        let inputs = PreparedBaseInputs {
            key,
            payload,
            extras: Vec::new(),
            includes: vec![include],
            shared_libs: Vec::new(),
            busybox_bytes: None,
        };

        let mut permissions = std::fs::metadata(&include_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&include_path, permissions).unwrap();
        let archive = inputs.build().unwrap();
        assert_eq!(
            newc_entry_mode(&archive, "include-files/mode-fixture"),
            captured_mode,
            "archive mode must remain the mode represented by the semantic key"
        );
    }

    #[test]
    fn explicit_archive_collisions_collapse_only_exact_verbatim_dependencies() {
        let root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(root.path()).unwrap();
        let same = root.path().join("same");
        let different = root.path().join("different");
        std::fs::write(&same, b"dependency bytes").unwrap();
        std::fs::write(&different, b"different bytes").unwrap();
        let guest_path = "lib64/libcollision.so";

        let executable_include = pinned_archive_entry(root.path(), guest_path, &same, 0o100755);
        let mut dependencies = vec![pinned_closure_entry(root.path(), guest_path, &same)];
        remove_explicit_dependency_collisions(&[], &[executable_include], &mut dependencies)
            .unwrap();
        assert!(
            dependencies.is_empty(),
            "an identical verbatim executable include is the same archive entry"
        );

        let non_executable_include = pinned_archive_entry(root.path(), guest_path, &same, 0o100644);
        let mut dependencies = vec![pinned_closure_entry(root.path(), guest_path, &same)];
        assert!(
            remove_explicit_dependency_collisions(
                &[],
                &[non_executable_include],
                &mut dependencies
            )
            .is_err(),
            "identical bytes with a different archive mode must not collapse"
        );

        let different_include = pinned_archive_entry(root.path(), guest_path, &different, 0o100755);
        let mut dependencies = vec![pinned_closure_entry(root.path(), guest_path, &same)];
        assert!(
            remove_explicit_dependency_collisions(&[], &[different_include], &mut dependencies)
                .is_err(),
            "the same guest path with different bytes must fail"
        );

        let stripped_extra = pinned_archive_entry(root.path(), guest_path, &same, 0o100755);
        let mut dependencies = vec![pinned_closure_entry(root.path(), guest_path, &same)];
        assert!(
            remove_explicit_dependency_collisions(&[stripped_extra], &[], &mut dependencies)
                .is_err(),
            "extra binaries undergo stripping and therefore cannot be \
             collapsed from their source digest"
        );
    }

    #[test]
    fn extra_and_include_archive_paths_must_be_disjoint() {
        let root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(root.path()).unwrap();
        let file = root.path().join("file");
        std::fs::write(&file, b"bytes").unwrap();
        let extra = pinned_archive_entry(root.path(), "usr/bin/tool", &file, 0o100755);
        let include = pinned_archive_entry(root.path(), "usr/bin/tool", &file, 0o100755);
        assert!(ensure_explicit_paths_disjoint(&[extra], &[include]).is_err());
    }

    #[test]
    fn closure_guest_collisions_collapse_by_content_and_reject_conflicts() {
        let root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(root.path()).unwrap();
        let first = root.path().join("first");
        let identical_inode = root.path().join("identical-copy");
        let different = root.path().join("different");
        std::fs::write(&first, b"same bytes").unwrap();
        std::fs::write(&identical_inode, b"same bytes").unwrap();
        std::fs::write(&different, b"different bytes").unwrap();

        let mut merged = Vec::new();
        merge_pinned_closure_entries(
            &mut merged,
            PinnedClosure {
                entries: vec![pinned_closure_entry(
                    root.path(),
                    "lib64/libdemo.so",
                    &first,
                )],
            },
        )
        .unwrap();
        merge_pinned_closure_entries(
            &mut merged,
            PinnedClosure {
                entries: vec![pinned_closure_entry(
                    root.path(),
                    "lib64/libdemo.so",
                    &identical_inode,
                )],
            },
        )
        .unwrap();
        assert_eq!(merged.len(), 1);
        assert!(
            merge_pinned_closure_entries(
                &mut merged,
                PinnedClosure {
                    entries: vec![pinned_closure_entry(
                        root.path(),
                        "lib64/libdemo.so",
                        &different,
                    )],
                },
            )
            .is_err(),
            "same guest path with different content must fail deterministically"
        );
    }

    #[test]
    fn prepared_archive_symlink_shape_is_content_not_inode_based() {
        let root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(root.path()).unwrap();
        let first = root.path().join("first");
        let copy = root.path().join("copy");
        std::fs::write(&first, b"library bytes").unwrap();
        std::fs::write(&copy, b"library bytes").unwrap();
        let first_input = pin_input(root.path(), &first).unwrap();
        let copy_input = pin_input(root.path(), &copy).unwrap();
        assert_ne!(first_input.identity.ino, copy_input.identity.ino);
        let first_proc = first_input.proc_path();
        let copy_proc = copy_input.proc_path();
        let separate_inodes = vec![
            (
                "lib64/libsame.so".to_owned(),
                first_proc.clone(),
                first_input.identity.size,
                first_input.content_hash,
            ),
            (
                "usr/lib64/libsame.so".to_owned(),
                copy_proc,
                copy_input.identity.size,
                copy_input.content_hash,
            ),
        ];
        let one_inode = vec![
            (
                "lib64/libsame.so".to_owned(),
                first_proc.clone(),
                first_input.identity.size,
                first_input.content_hash,
            ),
            (
                "usr/lib64/libsame.so".to_owned(),
                first_proc,
                first_input.identity.size,
                first_input.content_hash,
            ),
        ];
        assert_eq!(
            initramfs::build_initramfs_base_from_resolved(&[], &[], None, &separate_inodes,)
                .unwrap(),
            initramfs::build_initramfs_base_from_resolved(&[], &[], None, &one_inode).unwrap(),
            "byte-identical inode copies must produce the same archive graph"
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
                get_or_build_prepared_object(&root, test_base_expectation(key, 4096, 4096), || {
                    builds.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    Ok(test_base_object(key, 4096, 4096, vec![0x2a; 8193]))
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

    const PREPARED_CHILD_TEST: &str =
        "vmm::initramfs_cache::tests::prepared_object_cross_process_child";
    const PREPARED_CHILD_ROOT: &str = "KTSTR_PREPARED_CHILD_ROOT";
    const PREPARED_CHILD_MODE: &str = "KTSTR_PREPARED_CHILD_MODE";
    const PREPARED_CHILD_INDEX: &str = "KTSTR_PREPARED_CHILD_INDEX";
    const PREPARED_CHILD_KEY: &str = "KTSTR_PREPARED_CHILD_KEY";
    const PREPARED_CHILD_COUNTER: &str = "KTSTR_PREPARED_CHILD_COUNTER";
    const PREPARED_CHILD_VALIDATION_COUNTER: &str = "KTSTR_PREPARED_VALIDATION_COUNTER";
    const PREPARED_CHILD_START: &str = "KTSTR_PREPARED_CHILD_START";
    const PREPARED_CHILD_READY: &str = "KTSTR_PREPARED_CHILD_READY";
    const PREPARED_CHILD_RESULTS: &str = "KTSTR_PREPARED_CHILD_RESULTS";
    const PREPARED_CHILD_WINNER: &str = "KTSTR_PREPARED_CHILD_WINNER";

    struct ChildProcesses(Vec<Option<std::process::Child>>);

    impl ChildProcesses {
        fn push(&mut self, child: std::process::Child) {
            self.0.push(Some(child));
        }

        fn wait_all_success(&mut self, timeout: std::time::Duration) {
            let deadline = std::time::Instant::now() + timeout;
            while self.0.iter().any(Option::is_some) {
                for child in &mut self.0 {
                    let Some(process) = child.as_mut() else {
                        continue;
                    };
                    if let Some(status) = process.try_wait().unwrap() {
                        assert!(status.success(), "prepared child exited with {status}");
                        *child = None;
                    }
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "prepared child processes did not finish before timeout"
                );
                std::thread::yield_now();
            }
        }
    }

    impl Drop for ChildProcesses {
        fn drop(&mut self) {
            for process in self.0.iter_mut().flatten() {
                let _ = process.kill();
                let _ = process.wait();
            }
        }
    }

    struct PreparedChildPaths<'a> {
        root: &'a Path,
        counter: &'a Path,
        validation_counter: &'a Path,
        start: &'a Path,
        ready: &'a Path,
        results: &'a Path,
        winner: &'a Path,
    }

    fn spawn_prepared_child(
        paths: &PreparedChildPaths<'_>,
        mode: &str,
        index: usize,
        key: u64,
    ) -> std::process::Child {
        std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg(PREPARED_CHILD_TEST)
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(PREPARED_CHILD_ROOT, paths.root)
            .env(PREPARED_CHILD_MODE, mode)
            .env(PREPARED_CHILD_INDEX, index.to_string())
            .env(PREPARED_CHILD_KEY, format!("{key:016x}"))
            .env(PREPARED_CHILD_COUNTER, paths.counter)
            .env(PREPARED_CHILD_VALIDATION_COUNTER, paths.validation_counter)
            .env(PREPARED_CHILD_START, paths.start)
            .env(PREPARED_CHILD_READY, paths.ready)
            .env(PREPARED_CHILD_RESULTS, paths.results)
            .env(PREPARED_CHILD_WINNER, paths.winner)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .unwrap()
    }

    fn wait_for_child_files(
        directory: &Path,
        expected: usize,
        children: &mut ChildProcesses,
        timeout: std::time::Duration,
    ) {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let count = std::fs::read_dir(directory).unwrap().count();
            if count == expected {
                return;
            }
            for child in &mut children.0 {
                if let Some(process) = child
                    && let Some(status) = process.try_wait().unwrap()
                {
                    panic!("prepared child exited before barrier with {status}");
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "only {count}/{expected} prepared children reached the barrier"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn prepared_object_cross_process_child() {
        use std::io::Write as _;

        let Some(root) = std::env::var_os(PREPARED_CHILD_ROOT).map(PathBuf::from) else {
            return;
        };
        let mode = std::env::var(PREPARED_CHILD_MODE).unwrap();
        let index: usize = std::env::var(PREPARED_CHILD_INDEX)
            .unwrap()
            .parse()
            .unwrap();
        let key = u64::from_str_radix(&std::env::var(PREPARED_CHILD_KEY).unwrap(), 16).unwrap();
        let counter = PathBuf::from(std::env::var_os(PREPARED_CHILD_COUNTER).unwrap());
        let start = PathBuf::from(std::env::var_os(PREPARED_CHILD_START).unwrap());
        let ready = PathBuf::from(std::env::var_os(PREPARED_CHILD_READY).unwrap());
        let results = PathBuf::from(std::env::var_os(PREPARED_CHILD_RESULTS).unwrap());
        let winner = PathBuf::from(std::env::var_os(PREPARED_CHILD_WINNER).unwrap());
        ensure_prepared_cache_dirs(&root).unwrap();

        if matches!(mode.as_str(), "storm" | "validation-storm") {
            let start = File::open(start).unwrap();
            std::fs::write(ready.join(index.to_string()), b"ready").unwrap();
            rustix::fs::flock(&start, rustix::fs::FlockOperation::LockShared).unwrap();
        }

        let page = PREPARED_MAPPING_GRANULE;
        let file_alignment = prepared_file_alignment().unwrap();
        let object = get_or_build_prepared_object(
            &root,
            test_base_expectation(key, page, file_alignment),
            || {
                let mut attempts = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&counter)
                    .unwrap();
                writeln!(attempts, "{mode}:{index}").unwrap();
                attempts.sync_all().unwrap();
                if mode == "kill-builder" {
                    std::fs::write(&winner, b"exclusive-lock-held").unwrap();
                    loop {
                        std::thread::park();
                    }
                }
                // Large enough to make duplicate transformations/publications
                // visible without adding an artificial sleep.
                Ok(test_base_object(
                    key,
                    page,
                    file_alignment,
                    vec![0x6d; 8 << 20],
                ))
            },
        )
        .unwrap();

        let fd = object.fd.as_ref().unwrap();
        let stat = rustix::fs::fstat(fd).unwrap();
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                page,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(address, libc::MAP_FAILED);
        let address = address.cast::<u8>();
        unsafe {
            initramfs::cow_overlay_file_borrowed(address, page, fd, object.data_offset as u64)
                .unwrap();
        }
        assert_eq!(unsafe { address.read_volatile() }, 0x6d);
        let private_value = (index as u8).wrapping_add(1);
        unsafe {
            address.write_volatile(private_value);
        }
        assert_eq!(unsafe { address.read_volatile() }, private_value);
        let mut persisted = [0u8; 1];
        let read = unsafe {
            libc::pread(
                fd.as_raw_fd(),
                persisted.as_mut_ptr().cast(),
                1,
                object.data_offset as libc::off_t,
            )
        };
        assert_eq!(read, 1);
        assert_eq!(persisted[0], 0x6d);
        unsafe {
            libc::munmap(address.cast(), page);
        }
        let result = if mode == "storm" {
            let args = vec![format!("cross-process-boundary-{index:02}")];
            let params = initramfs::SuffixParams {
                args: &args,
                ..Default::default()
            };
            let expected = initramfs::build_dynamic_tail(0, &params).unwrap();
            let prepared_base = test_prepared_base(
                &root,
                key.wrapping_add(0x1000_0000),
                initramfs::InitrdCompression::Uncompressed,
                &[],
            );
            let prepared = complete_prepared_initrd(prepared_base, &params).unwrap();
            let boundary_cache_hits = prepared.cache_hits();
            let identity = inspect_boundary_mapping(&prepared, file_alignment);
            assert_eq!(read_prepared_stream(prepared), expected);
            format!(
                "{} {} {} {} {} {} {} {} {} {} {} {} {}",
                stat.st_dev,
                stat.st_ino,
                persisted[0],
                private_value,
                u8::from(object.cache_hit),
                identity.underlay_device,
                identity.underlay_inode,
                identity.underlay_key,
                identity.boundary_device,
                identity.boundary_inode,
                identity.boundary_key,
                identity.boundary_len,
                boundary_cache_hits,
            )
        } else {
            format!(
                "{} {} {} {} {}",
                stat.st_dev,
                stat.st_ino,
                persisted[0],
                private_value,
                u8::from(object.cache_hit)
            )
        };
        std::fs::write(results.join(index.to_string()), result).unwrap();
    }

    #[test]
    fn prepared_object_52_process_storm_builds_once_and_cow_isolates() {
        const CELLS: usize = 52;
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let counter = temp.path().join("attempts");
        let validation_counter = temp.path().join("validation-attempts");
        let start_path = temp.path().join("start");
        let ready = temp.path().join("ready");
        let results = temp.path().join("results");
        let winner = temp.path().join("winner");
        std::fs::create_dir(&ready).unwrap();
        std::fs::create_dir(&results).unwrap();
        let start = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&start_path)
            .unwrap();
        rustix::fs::flock(&start, rustix::fs::FlockOperation::LockExclusive).unwrap();
        let paths = PreparedChildPaths {
            root: temp.path(),
            counter: &counter,
            validation_counter: &validation_counter,
            start: &start_path,
            ready: &ready,
            results: &results,
            winner: &winner,
        };
        let key = 0xd001;
        let mut children = ChildProcesses(Vec::with_capacity(CELLS));
        for index in 0..CELLS {
            children.push(spawn_prepared_child(&paths, "storm", index, key));
        }
        wait_for_child_files(
            &ready,
            CELLS,
            &mut children,
            std::time::Duration::from_secs(30),
        );
        rustix::fs::flock(&start, rustix::fs::FlockOperation::Unlock).unwrap();
        children.wait_all_success(std::time::Duration::from_secs(60));

        let attempts = std::fs::read_to_string(&counter).unwrap();
        assert_eq!(
            attempts.lines().count(),
            1,
            "the barriered process storm must run exactly one transform"
        );
        let mut inodes = std::collections::HashSet::new();
        let mut boundary_underlays = std::collections::HashSet::new();
        let mut boundaries = std::collections::HashSet::new();
        let mut cache_hits = 0usize;
        let mut boundary_cache_hits = 0usize;
        for index in 0..CELLS {
            let result = std::fs::read_to_string(results.join(index.to_string())).unwrap();
            let fields: Vec<u64> = result
                .split_whitespace()
                .map(|field| field.parse().unwrap())
                .collect();
            assert_eq!(fields.len(), 13);
            inodes.insert((fields[0], fields[1]));
            assert_eq!(fields[2], 0x6d);
            assert_eq!(fields[3] as u8, (index as u8).wrapping_add(1));
            cache_hits += fields[4] as usize;
            boundary_underlays.insert((fields[5], fields[6], fields[7]));
            boundaries.insert((fields[8], fields[9], fields[10]));
            assert!(fields[11] > 0);
            assert!(fields[11] < PREPARED_MAPPING_GRANULE as u64);
            assert_eq!(fields[11] % prepared_file_alignment().unwrap() as u64, 0);
            boundary_cache_hits += fields[12] as usize;
        }
        assert_eq!(inodes.len(), 1, "every process must map the same CAS inode");
        assert_eq!(cache_hits, CELLS - 1);
        assert_eq!(
            boundary_underlays.len(),
            1,
            "all processes must map the same elected stable underlay inode and key"
        );
        assert_eq!(
            boundaries.len(),
            CELLS,
            "distinct process-local tails must publish distinct minimal boundaries"
        );
        assert_eq!(
            boundary_cache_hits,
            2 * (CELLS - 1),
            "one process must build each stable base/underlay while every peer reuses both"
        );
    }

    #[test]
    fn prepared_object_cross_process_cache_hit_hashes_new_inode_once() {
        const CELLS: usize = 52;
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let counter = temp.path().join("build-attempts");
        let validation_counter = temp.path().join("validation-attempts");
        let start_path = temp.path().join("start");
        let ready = temp.path().join("ready");
        let results = temp.path().join("results");
        let winner = temp.path().join("winner");
        std::fs::create_dir(&ready).unwrap();
        std::fs::create_dir(&results).unwrap();
        let key = 0xd00b;
        let object_path = prepared_object_path(temp.path(), PreparedObjectKind::Base, key);
        publish_prepared_object(
            temp.path(),
            &object_path,
            &test_base_object(
                key,
                PREPARED_MAPPING_GRANULE,
                prepared_file_alignment().unwrap(),
                vec![0x6d; 8 << 20],
            ),
        )
        .unwrap();
        std::fs::remove_file(prepared_validation_record_path(
            temp.path(),
            PreparedObjectKind::Base,
            key,
        ))
        .unwrap();

        let start = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&start_path)
            .unwrap();
        rustix::fs::flock(&start, rustix::fs::FlockOperation::LockExclusive).unwrap();
        let paths = PreparedChildPaths {
            root: temp.path(),
            counter: &counter,
            validation_counter: &validation_counter,
            start: &start_path,
            ready: &ready,
            results: &results,
            winner: &winner,
        };
        let mut children = ChildProcesses(Vec::with_capacity(CELLS));
        for index in 0..CELLS {
            children.push(spawn_prepared_child(&paths, "validation-storm", index, key));
        }
        wait_for_child_files(
            &ready,
            CELLS,
            &mut children,
            std::time::Duration::from_secs(30),
        );
        rustix::fs::flock(&start, rustix::fs::FlockOperation::Unlock).unwrap();
        children.wait_all_success(std::time::Duration::from_secs(60));

        assert!(
            !counter.exists(),
            "an already-published object must never rerun its transform"
        );
        assert_eq!(
            std::fs::read_to_string(&validation_counter)
                .unwrap()
                .lines()
                .count(),
            1,
            "one process must stream-validate a new inode and every peer must use its memo"
        );
        let mut inodes = std::collections::HashSet::new();
        for index in 0..CELLS {
            let result = std::fs::read_to_string(results.join(index.to_string())).unwrap();
            let fields: Vec<u64> = result
                .split_whitespace()
                .map(|field| field.parse().unwrap())
                .collect();
            assert_eq!(fields.len(), 5);
            inodes.insert((fields[0], fields[1]));
            assert_eq!(
                fields[4], 1,
                "every prepublished-object open is a cache hit"
            );
        }
        assert_eq!(inodes.len(), 1);

        let mut warm = ChildProcesses(Vec::with_capacity(1));
        warm.push(spawn_prepared_child(&paths, "validation-hit", CELLS, key));
        warm.wait_all_success(std::time::Duration::from_secs(30));
        assert_eq!(
            std::fs::read_to_string(&validation_counter)
                .unwrap()
                .lines()
                .count(),
            1,
            "a stable-inode cache hit must validate in O(1) without streaming again"
        );
    }

    #[test]
    fn prepared_object_killed_builder_elects_successor() {
        const WAITERS: usize = 52;
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let counter = temp.path().join("attempts");
        let validation_counter = temp.path().join("validation-attempts");
        let start = temp.path().join("unused-start");
        let ready = temp.path().join("ready");
        let results = temp.path().join("results");
        let winner = temp.path().join("winner");
        std::fs::create_dir(&ready).unwrap();
        std::fs::create_dir(&results).unwrap();
        let paths = PreparedChildPaths {
            root: temp.path(),
            counter: &counter,
            validation_counter: &validation_counter,
            start: &start,
            ready: &ready,
            results: &results,
            winner: &winner,
        };
        let key = 0xd002;
        let mut children = ChildProcesses(Vec::with_capacity(WAITERS + 1));
        children.push(spawn_prepared_child(&paths, "kill-builder", WAITERS, key));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !winner.exists() {
            if let Some(status) = children.0[0].as_mut().unwrap().try_wait().unwrap() {
                panic!("elected builder exited before kill point: {status}");
            }
            assert!(
                std::time::Instant::now() < deadline,
                "elected builder did not reach its publication hold point"
            );
            std::thread::yield_now();
        }
        for index in 0..WAITERS {
            children.push(spawn_prepared_child(
                &paths,
                "wait-killed-builder",
                index,
                key,
            ));
        }
        wait_for_child_files(
            &ready,
            WAITERS,
            &mut children,
            std::time::Duration::from_secs(30),
        );

        let mut killed = children.0[0].take().unwrap();
        killed.kill().unwrap();
        assert!(!killed.wait().unwrap().success());

        children.wait_all_success(std::time::Duration::from_secs(60));
        assert_eq!(
            std::fs::read_to_string(&counter).unwrap().lines().count(),
            2,
            "one killed attempt and one elected successor must run"
        );
        let mut inodes = std::collections::HashSet::new();
        let mut cache_hits = 0usize;
        for index in 0..WAITERS {
            let result = std::fs::read_to_string(results.join(index.to_string())).unwrap();
            let fields: Vec<u64> = result
                .split_whitespace()
                .map(|field| field.parse().unwrap())
                .collect();
            assert_eq!(fields.len(), 5);
            inodes.insert((fields[0], fields[1]));
            cache_hits += fields[4] as usize;
        }
        assert_eq!(
            inodes.len(),
            1,
            "every surviving waiter must reuse the successor's inode"
        );
        assert_eq!(
            cache_hits,
            WAITERS - 1,
            "exactly one waiter must become the successor builder"
        );
        let object_path = prepared_object_path(temp.path(), PreparedObjectKind::Base, key);
        assert!(object_path.exists());
        assert!(
            try_open_prepared_object(
                temp.path(),
                &object_path,
                test_base_expectation(
                    key,
                    PREPARED_MAPPING_GRANULE,
                    prepared_file_alignment().unwrap(),
                ),
            )
            .unwrap()
            .is_some()
        );
    }

    #[test]
    fn prepared_object_coordination_cannot_invert_with_gc_namespace_lock() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let key = 0xd00a;
        let expected = test_base_expectation(key, 4096, 4096);
        drop(
            get_or_build_prepared_object(temp.path(), expected, || {
                Ok(test_base_object(key, 4096, 4096, vec![0x41; 8193]))
            })
            .unwrap(),
        );

        let object_path = prepared_object_path(temp.path(), PreparedObjectKind::Base, key);
        let lock_path = prepared_object_lock_path(temp.path(), PreparedObjectKind::Base, key);
        let mut coordination = open_coord_file(temp.path(), &lock_path).unwrap();
        coordination.try_lock_exclusive().unwrap();

        assert!(
            try_lock_gc_namespace(temp.path()).unwrap().is_none(),
            "a per-key holder must retain namespace LOCK_SH until object LOCK_SH \
             closes the gate -> per-key -> object lock chain"
        );
        let opened =
            try_open_prepared_object_under_namespace(temp.path(), &object_path, expected, || {})
                .unwrap()
                .expect("published object must open under the retained namespace gate");

        coordination.release_namespace_gate();
        let namespace = try_lock_gc_namespace(temp.path())
            .unwrap()
            .expect("object LOCK_SH makes the inode self-protecting after namespace release");
        drop(coordination);
        assert!(
            !try_collect_object(
                temp.path(),
                &GcCandidate {
                    path: object_path.clone(),
                    kind: PreparedObjectKind::Base,
                    key,
                    modified_secs: 0,
                    len: std::fs::metadata(&object_path).unwrap().len(),
                },
                &namespace,
            )
            .unwrap(),
            "the opened object's LOCK_SH must prevent GC after the gate is released"
        );
        drop(opened);
        assert!(
            try_collect_object(
                temp.path(),
                &GcCandidate {
                    path: object_path,
                    kind: PreparedObjectKind::Base,
                    key,
                    modified_secs: 0,
                    len: 0,
                },
                &namespace,
            )
            .unwrap(),
            "the object becomes collectible only after its shared lock drops"
        );
        drop(namespace);

        let validation_lock =
            prepared_validation_lock_path(temp.path(), PreparedObjectKind::Base, key);
        let mut validation = open_coord_file(temp.path(), &validation_lock).unwrap();
        validation.try_lock_exclusive().unwrap();
        assert!(
            try_lock_gc_namespace(temp.path()).unwrap().is_none(),
            "validation election must retain namespace LOCK_SH while acquiring \
             the per-recipe validation lock"
        );
        validation.release_namespace_gate();
        let namespace = try_lock_gc_namespace(temp.path())
            .unwrap()
            .expect("GC must acquire the namespace after the validation holder releases it");
        let validation_key = prepared_validation_key(PreparedObjectKind::Base, key);
        assert!(
            !try_collect_memo(
                temp.path(),
                &GcMemoCandidate {
                    path: prepared_validation_record_path(
                        temp.path(),
                        PreparedObjectKind::Base,
                        key,
                    ),
                    kind: GcMemoKind::Validation,
                    key: validation_key,
                    modified_secs: 0,
                    len: 0,
                },
                &namespace,
            )
            .unwrap(),
            "GC must probe the held validation lock nonblocking and leave its \
             memo in place rather than invert the lock order"
        );
    }

    #[test]
    fn gc_object_open_is_namespace_safe_and_reuses_only_the_published_inode() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let key = 0xd003;
        let expected = test_base_expectation(key, 4096, 4096);
        let object = get_or_build_prepared_object(temp.path(), expected, || {
            Ok(test_base_object(key, 4096, 4096, vec![0x31; 8193]))
        })
        .unwrap();
        let object_path = prepared_object_path(temp.path(), PreparedObjectKind::Base, key);
        let old_stat = rustix::fs::fstat(object.fd.as_ref().unwrap()).unwrap();
        drop(object);

        let namespace = try_lock_gc_namespace(temp.path())
            .unwrap()
            .expect("no other operation should hold the private test namespace");
        let coordination = open_lock_file(&prepared_object_lock_path(
            temp.path(),
            PreparedObjectKind::Base,
            key,
        ))
        .unwrap();
        rustix::fs::flock(&coordination, rustix::fs::FlockOperation::LockExclusive).unwrap();
        let old_inode = File::open(&object_path).unwrap();
        rustix::fs::flock(&old_inode, rustix::fs::FlockOperation::LockExclusive).unwrap();

        let (attempted_tx, attempted_rx) = std::sync::mpsc::channel();
        let (opened_tx, opened_rx) = std::sync::mpsc::channel();
        let (continue_tx, continue_rx) = std::sync::mpsc::channel();
        let root = temp.path().to_path_buf();
        let reader_path = object_path.clone();
        let reader = std::thread::spawn(move || {
            attempted_tx.send(()).unwrap();
            try_open_prepared_object_with_open_hook(&root, &reader_path, expected, || {
                opened_tx.send(()).unwrap();
                continue_rx.recv().unwrap();
            })
            .unwrap()
        });
        attempted_rx.recv().unwrap();
        let opened_before_gc = opened_rx
            .recv_timeout(std::time::Duration::from_millis(500))
            .is_ok();

        std::fs::remove_file(&object_path).unwrap();
        drop(coordination);
        drop(namespace);

        if opened_before_gc {
            // Let a buggy open-before-namespace reader continue and unblock
            // it from the unlinked old inode so the regression fails cleanly
            // instead of leaving a stuck test thread behind.
            continue_tx.send(()).unwrap();
            drop(old_inode);
            let observed = reader.join().unwrap();
            assert!(
                observed.is_none(),
                "a reader opened the object before taking the namespace gate"
            );
            return;
        }

        assert!(
            reader.join().unwrap().is_none(),
            "the path was absent when the namespace gate admitted the reader"
        );
        let replacement = get_or_build_prepared_object(temp.path(), expected, || {
            Ok(test_base_object(key, 4096, 4096, vec![0x32; 8193]))
        })
        .unwrap();
        let replacement_stat = rustix::fs::fstat(replacement.fd.as_ref().unwrap()).unwrap();
        assert_ne!(
            (replacement_stat.st_dev, replacement_stat.st_ino),
            (old_stat.st_dev, old_stat.st_ino),
            "the still-open unlinked inode cannot be the replacement namespace object"
        );
        drop(old_inode);

        for _ in 0..8 {
            let reopened = try_open_prepared_object(temp.path(), &object_path, expected)
                .unwrap()
                .expect("replacement object must remain published");
            let stat = rustix::fs::fstat(reopened.fd.as_ref().unwrap()).unwrap();
            assert_eq!(
                (stat.st_dev, stat.st_ino),
                (replacement_stat.st_dev, replacement_stat.st_ino),
                "all cache hits must reuse the currently published inode"
            );
        }
    }

    #[test]
    fn gc_namespace_gate_closes_open_flock_unlink_race_and_sweeps_idle_locks() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let lock_path = temp
            .path()
            .join(PREPARED_LOCKS_DIR)
            .join("payload-000000000000abcd.lock");

        let mut holder = open_coord_file(temp.path(), &lock_path).unwrap();
        holder.try_lock_exclusive().unwrap();

        let (opened_tx, opened_rx) = std::sync::mpsc::channel();
        let (acquired_tx, acquired_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let root = temp.path().to_path_buf();
        let waiter_path = lock_path.clone();
        let waiter = std::thread::spawn(move || {
            let mut coordination = open_coord_file(&root, &waiter_path).unwrap();
            assert_eq!(
                coordination.try_lock_exclusive().unwrap_err(),
                rustix::io::Errno::WOULDBLOCK
            );
            // The waiter still holds namespace LOCK_SH at this point and
            // retains it while blocking for the per-key shared lock.
            opened_tx.send(()).unwrap();
            coordination.lock_shared().unwrap();
            coordination.release_namespace_gate();
            acquired_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });

        opened_rx.recv().unwrap();
        assert!(
            try_lock_gc_namespace(temp.path()).unwrap().is_none(),
            "GC must not enter the unlink namespace while an opener is \
             between open and successful per-key flock"
        );

        drop(holder);
        acquired_rx.recv().unwrap();
        let namespace = try_lock_gc_namespace(temp.path())
            .unwrap()
            .expect("waiter releases namespace gate after taking per-key LOCK_SH");
        assert_eq!(
            sweep_idle_coordination_locks(temp.path(), &namespace).unwrap(),
            0,
            "a live per-key shared holder must prevent lock inode collection"
        );
        assert!(lock_path.exists());

        release_tx.send(()).unwrap();
        waiter.join().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while lock_path.exists() {
            sweep_idle_coordination_locks(temp.path(), &namespace).unwrap();
            assert!(
                std::time::Instant::now() < deadline,
                "idle coordination lock remained live after every owner dropped"
            );
            // A parallel self-spawn test may have forked while the fd was
            // open. O_CLOEXEC closes that inherited reference at exec, but
            // GC must correctly observe it as live during the brief fork→exec
            // window and retry rather than unlinking under it.
            std::thread::yield_now();
        }
        assert!(!lock_path.exists());
        drop(namespace);

        let mut replacement = open_coord_file(temp.path(), &lock_path).unwrap();
        replacement.try_lock_exclusive().unwrap();
        assert!(
            lock_path.exists(),
            "the first post-GC operation must safely recreate the coordination inode"
        );
    }

    #[test]
    fn gc_preserves_live_object_and_temp_locks_then_collects_them() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let object_key = 0xa001;
        let object = test_prepared_part(
            temp.path(),
            PreparedObjectKind::Payload,
            object_key,
            4096,
            4096,
            0,
            1024,
        );
        let object_path =
            prepared_object_path(temp.path(), PreparedObjectKind::Payload, object_key);

        let temp_key = 0xa002;
        let temp_path = temp
            .path()
            .join(PREPARED_OBJECTS_DIR)
            .join(format!(".tmp-object-payload-{temp_key:016x}-interrupted"));
        std::fs::write(&temp_path, b"partial").unwrap();
        let temp_lock_path =
            prepared_object_lock_path(temp.path(), PreparedObjectKind::Payload, temp_key);
        let mut temp_holder = open_coord_file(temp.path(), &temp_lock_path).unwrap();
        temp_holder.try_lock_exclusive().unwrap();
        temp_holder.release_namespace_gate();

        let namespace = try_lock_gc_namespace(temp.path())
            .unwrap()
            .expect("coordination holders release the namespace after flock");
        let far_future =
            unix_now_secs() + PREPARED_CAS_MAX_AGE_SECS + PREPARED_CAS_GC_INTERVAL_SECS + 1;
        run_prepared_cache_gc(temp.path(), far_future, &namespace).unwrap();
        assert!(
            object_path.exists(),
            "the object's LOCK_SH must prevent final-object collection"
        );
        assert!(
            temp_path.exists(),
            "the interrupted winner's per-key EX lock must protect its temp"
        );

        drop(object);
        drop(temp_holder);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while object_path.exists() || temp_path.exists() {
            run_prepared_cache_gc(temp.path(), far_future, &namespace).unwrap();
            assert!(
                std::time::Instant::now() < deadline,
                "GC did not collect objects after every live owner dropped"
            );
            std::thread::yield_now();
        }
        assert!(!object_path.exists());
        assert!(!temp_path.exists());
    }

    #[test]
    fn gc_accounts_for_memo_age_and_size() {
        let temp = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(temp.path()).unwrap();
        let coverage = temp
            .path()
            .join(PREPARED_PROBES_DIR)
            .join("000000000000b001.coverage");
        let closure = temp
            .path()
            .join(PREPARED_CLOSURES_DIR)
            .join("000000000000b002.closure");
        std::fs::write(&coverage, b"old coverage").unwrap();
        std::fs::write(&closure, b"old closure").unwrap();
        let namespace = try_lock_gc_namespace(temp.path()).unwrap().unwrap();
        run_prepared_cache_gc(
            temp.path(),
            unix_now_secs() + PREPARED_CAS_MAX_AGE_SECS + 1,
            &namespace,
        )
        .unwrap();
        assert!(!coverage.exists());
        assert!(!closure.exists());

        let sparse = temp
            .path()
            .join(PREPARED_DIGESTS_DIR)
            .join("000000000000b003.digest");
        File::create(&sparse)
            .unwrap()
            .set_len(PREPARED_CAS_MAX_BYTES + 1)
            .unwrap();
        run_prepared_cache_gc(temp.path(), unix_now_secs(), &namespace).unwrap();
        assert!(
            sparse.exists(),
            "a sparse memo must be charged by allocated storage, not logical length"
        );

        let allocated = temp
            .path()
            .join(PREPARED_DIGESTS_DIR)
            .join("000000000000b004.digest");
        std::fs::write(&allocated, vec![0xa5; 16 << 10]).unwrap();
        run_prepared_cache_gc_with_limit(temp.path(), unix_now_secs(), &namespace, 4 << 10)
            .unwrap();
        assert!(
            !allocated.exists(),
            "allocated memo blocks must count toward the cap"
        );
        assert!(
            sparse.exists(),
            "size GC must not evict a zero-block sparse memo"
        );
    }

    #[test]
    fn gc_accounts_for_sparse_and_allocated_prepared_objects_and_temps() {
        let finals = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(finals.path()).unwrap();
        let sparse_final = prepared_object_path(finals.path(), PreparedObjectKind::Stitch, 0xb101);
        File::create(&sparse_final)
            .unwrap()
            .set_len(PREPARED_CAS_MAX_BYTES + 1)
            .unwrap();
        let allocated_final =
            prepared_object_path(finals.path(), PreparedObjectKind::Boundary, 0xb102);
        std::fs::write(&allocated_final, vec![0xa5; 16 << 10]).unwrap();
        assert_eq!(
            metadata_allocated_bytes(&sparse_final.metadata().unwrap()),
            0,
            "the sparse-final fixture must have no allocated data blocks"
        );
        assert!(
            metadata_allocated_bytes(&allocated_final.metadata().unwrap()) > 0,
            "the allocated-final fixture must consume data blocks"
        );
        let namespace = try_lock_gc_namespace(finals.path()).unwrap().unwrap();
        run_prepared_cache_gc_with_limit(finals.path(), unix_now_secs(), &namespace, 0).unwrap();
        assert!(
            sparse_final.exists(),
            "size GC must not evict a zero-block sparse final object"
        );
        assert!(
            !allocated_final.exists(),
            "allocated final-object blocks must count toward the cap"
        );

        let sparse_temp_root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(sparse_temp_root.path()).unwrap();
        let sparse_temp = sparse_temp_root
            .path()
            .join(PREPARED_OBJECTS_DIR)
            .join(".tmp-object-boundary-000000000000b103-sparse");
        File::create(&sparse_temp)
            .unwrap()
            .set_len(PREPARED_CAS_MAX_BYTES + 1)
            .unwrap();
        let sparse_sentinel = prepared_object_path(
            sparse_temp_root.path(),
            PreparedObjectKind::Boundary,
            0xb104,
        );
        std::fs::write(&sparse_sentinel, vec![0x5a; 16 << 10]).unwrap();
        let sparse_sentinel_bytes = metadata_allocated_bytes(&sparse_sentinel.metadata().unwrap());
        assert_eq!(
            metadata_allocated_bytes(&sparse_temp.metadata().unwrap()),
            0,
            "the sparse-temp fixture must have no allocated data blocks"
        );
        let namespace = try_lock_gc_namespace(sparse_temp_root.path())
            .unwrap()
            .unwrap();
        run_prepared_cache_gc_with_limit(
            sparse_temp_root.path(),
            unix_now_secs(),
            &namespace,
            sparse_sentinel_bytes,
        )
        .unwrap();
        assert!(sparse_temp.exists());
        assert!(
            sparse_sentinel.exists(),
            "a huge but zero-block fresh temp must not force final-object eviction"
        );

        let allocated_temp_root = tempfile::tempdir().unwrap();
        ensure_prepared_cache_dirs(allocated_temp_root.path()).unwrap();
        let allocated_temp = allocated_temp_root
            .path()
            .join(PREPARED_OBJECTS_DIR)
            .join(".tmp-object-boundary-000000000000b105-allocated");
        std::fs::write(&allocated_temp, vec![0xc3; 16 << 10]).unwrap();
        let allocated_sentinel = prepared_object_path(
            allocated_temp_root.path(),
            PreparedObjectKind::Boundary,
            0xb106,
        );
        std::fs::write(&allocated_sentinel, vec![0x3c; 16 << 10]).unwrap();
        let allocated_sentinel_bytes =
            metadata_allocated_bytes(&allocated_sentinel.metadata().unwrap());
        assert!(
            metadata_allocated_bytes(&allocated_temp.metadata().unwrap()) > 0,
            "the allocated-temp fixture must consume data blocks"
        );
        let namespace = try_lock_gc_namespace(allocated_temp_root.path())
            .unwrap()
            .unwrap();
        run_prepared_cache_gc_with_limit(
            allocated_temp_root.path(),
            unix_now_secs(),
            &namespace,
            allocated_sentinel_bytes,
        )
        .unwrap();
        assert!(
            allocated_temp.exists(),
            "fresh temp objects remain protected from direct size eviction"
        );
        assert!(
            !allocated_sentinel.exists(),
            "allocated temp blocks must still contribute to total cache pressure"
        );
    }

    fn smaps_value_kib(address: *mut u8, field: &str) -> Option<u64> {
        let start = format!("{:x}-", address as usize);
        let field = format!("{field}:");
        let smaps = std::fs::read_to_string("/proc/self/smaps").ok()?;
        let mut in_mapping = false;
        for line in smaps.lines() {
            if !in_mapping {
                in_mapping = line.starts_with(&start);
                continue;
            }
            if line.as_bytes().first().is_some_and(u8::is_ascii_hexdigit) && line.contains('-') {
                return None;
            }
            if let Some(value) = line.strip_prefix(&field) {
                return value.split_whitespace().next()?.parse().ok();
            }
        }
        None
    }

    #[test]
    fn prepared_cow_overlay_after_anon_prefault_stays_clean_and_isolated() {
        let host_page = crate::vmm::setup::host_page_size() as usize;
        let len = host_page * 4;
        let backing_bytes: Vec<u8> = (0..len).map(|index| (index % 251) as u8).collect();
        let temp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temp.path(), &backing_bytes).unwrap();
        temp.as_file().sync_all().unwrap();
        let file: OwnedFd = OpenOptions::new()
            .read(true)
            .open(temp.path())
            .unwrap()
            .into();
        rustix::fs::flock(&file, rustix::fs::FlockOperation::LockShared).unwrap();

        let reserve = || unsafe {
            let address = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            assert_ne!(address, libc::MAP_FAILED);
            address.cast::<u8>()
        };
        let first = reserve();
        let second = reserve();
        // Model performance-mode's anonymous prefault before the prepared
        // overlay. Touch every page even on kernels lacking
        // MADV_POPULATE_WRITE.
        for address in [first, second] {
            for offset in (0..len).step_by(host_page) {
                unsafe {
                    address.add(offset).write_volatile(0xa5);
                }
            }
            unsafe {
                initramfs::cow_overlay_file_borrowed(address, len, &file, 0).unwrap();
            }
        }

        assert_eq!(
            unsafe { std::slice::from_raw_parts(first, len) },
            backing_bytes,
            "MAP_FIXED must replace the prefaulted anonymous pages"
        );
        assert_eq!(
            smaps_value_kib(first, "Anonymous"),
            Some(0),
            "installing the prepared mapping after prefault must not leave \
             anonymous COW pages behind"
        );

        unsafe {
            first.add(host_page).write_volatile(0xee);
        }
        assert_eq!(
            unsafe { second.add(host_page).read_volatile() },
            backing_bytes[host_page]
        );
        let mut persisted = [0u8; 1];
        File::open(temp.path())
            .unwrap()
            .read_at(&mut persisted, host_page as u64)
            .unwrap();
        assert_eq!(persisted[0], backing_bytes[host_page]);

        unsafe {
            libc::munmap(first.cast(), len);
            libc::munmap(second.cast(), len);
        }
    }

    #[test]
    fn prepared_cow_overlay_replaces_real_hugetlb_vma_when_available() {
        let len = PREPARED_MAPPING_GRANULE;
        let host_page = crate::vmm::setup::host_page_size() as usize;
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | libc::MAP_HUGE_2MB,
                -1,
                0,
            )
        };
        if address == libc::MAP_FAILED {
            let error = std::io::Error::last_os_error();
            skip!(
                "2 MiB MAP_HUGETLB reservation unavailable; \
                 regular-file MAP_FIXED regression not applicable: {error}"
            );
        }
        let address = address.cast::<u8>();
        assert_eq!(
            (address as usize) % PREPARED_MAPPING_GRANULE,
            0,
            "the kernel must return a hugepage-aligned VMA"
        );

        // Deliberately use a host-page-aligned but ordinarily non-2-MiB file
        // offset. The destination and length retain hugetlb's 2 MiB geometry;
        // the replacement regular-file mmap only requires host-page file
        // alignment.
        let mut backing_bytes = vec![0x5a; host_page];
        backing_bytes.extend((0..len).map(|index| (index % 251) as u8));
        let temp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temp.path(), &backing_bytes).unwrap();
        temp.as_file().sync_all().unwrap();
        let file: OwnedFd = OpenOptions::new()
            .read(true)
            .open(temp.path())
            .unwrap()
            .into();
        rustix::fs::flock(&file, rustix::fs::FlockOperation::LockShared).unwrap();

        unsafe {
            initramfs::cow_overlay_file_borrowed(address, len, &file, host_page as u64)
                .expect("replace 2 MiB hugetlb VMA with host-page-aligned regular file");
        }
        assert_eq!(unsafe { address.read_volatile() }, backing_bytes[host_page]);
        assert_eq!(
            unsafe { address.add(len - 1).read_volatile() },
            backing_bytes[host_page + len - 1]
        );

        unsafe {
            address.add(host_page).write_volatile(0xee);
        }
        let mut persisted = [0u8; 1];
        File::open(temp.path())
            .unwrap()
            .read_at(&mut persisted, (host_page * 2) as u64)
            .unwrap();
        assert_eq!(
            persisted[0],
            backing_bytes[host_page * 2],
            "MAP_PRIVATE write must not mutate the prepared CAS file"
        );

        unsafe {
            libc::munmap(address.cast(), len);
        }
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
