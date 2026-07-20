//! Shared content-addressed file coordination.
//!
//! Large immutable inputs are opened and pinned by file descriptor, their
//! stable inode revision is digested once across processes, and per-content
//! builders use one common nonblocking-winner/shared-reader/successor election
//! protocol. Callers own their record wire format; this module owns the
//! expensive-input and cross-process coordination invariants.

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::hash::{BuildHasher, Hasher};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use ahash::AHasher;

const FILE_DIGEST_MAGIC: &[u8; 8] = b"KTSTRDG\0";
const FILE_DIGEST_RECORD_LEN: usize = 88;
const FILE_DIGEST_SCHEMA: u32 = 1;
const CONTENT_HASH_CHUNK_LEN: usize = 1 << 20;
const FILE_DIGEST_MEMO_DIR: &str = "digests-v1";
const FILE_DIGEST_LOCK_DIR: &str = ".locks-v1";
const FILE_DIGEST_NAMESPACE_GATE: &str = "namespace.lock";
const CONTENT_OBJECT_DIR: &str = "objects-v2";
const CONTENT_GC_STAMP: &str = ".gc-v2";
const CONTENT_GC_INTERVAL: Duration = Duration::from_secs(60 * 60);
const CONTENT_MAX_AGE: Duration = Duration::from_secs(30 * 24 * 60 * 60);
const CONTENT_MAX_OBJECT_BYTES: u64 = 8 << 30;

fn fixed_hasher() -> AHasher {
    ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher()
}

fn ahash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = fixed_hasher();
    hasher.write(bytes);
    hasher.finish()
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

fn pread_exact(file: &File, mut offset: u64, mut bytes: &mut [u8], subject: &str) -> Result<()> {
    while !bytes.is_empty() {
        let read = loop {
            match file.read_at(bytes, offset) {
                Ok(read) => break read,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error).with_context(|| format!("pread {subject}")),
            }
        };
        anyhow::ensure!(read > 0, "{subject} was truncated while reading");
        offset = offset
            .checked_add(u64::try_from(read)?)
            .with_context(|| format!("{subject} offset overflow"))?;
        bytes = &mut bytes[read..];
    }
    Ok(())
}

/// Stable identity for one opened regular-file revision.
///
/// `ctime` participates in the machine-wide memo key so an in-place rewrite
/// that restores size and mtime cannot reuse the old digest. Comparisons while
/// an fd is actively pinned intentionally omit ctime: unlinking or renaming the
/// pathname changes ctime without changing the bytes behind the open file
/// description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct StableFileIdentity {
    pub(crate) dev: u64,
    pub(crate) ino: u64,
    pub(crate) size: u64,
    pub(crate) mtime_secs: i64,
    pub(crate) mtime_nsecs: i64,
    pub(crate) ctime_secs: i64,
    pub(crate) ctime_nsecs: i64,
}

impl StableFileIdentity {
    pub(crate) fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            dev: metadata.dev(),
            ino: metadata.ino(),
            size: metadata.size(),
            mtime_secs: metadata.mtime(),
            mtime_nsecs: metadata.mtime_nsec(),
            ctime_secs: metadata.ctime(),
            ctime_nsecs: metadata.ctime_nsec(),
        }
    }

    pub(crate) fn from_file(file: &File) -> Result<Self> {
        let metadata = file.metadata().context("stat pinned content input")?;
        anyhow::ensure!(
            metadata.is_file(),
            "content-addressed input is not a regular file"
        );
        Ok(Self::from_metadata(&metadata))
    }

    pub(crate) fn same_open_content_version(self, other: Self) -> bool {
        self.dev == other.dev
            && self.ino == other.ino
            && self.size == other.size
            && self.mtime_secs == other.mtime_secs
            && self.mtime_nsecs == other.mtime_nsecs
    }

    pub(crate) fn hash_into(self, hasher: &mut AHasher) {
        hash_u64(hasher, self.dev);
        hash_u64(hasher, self.ino);
        hash_u64(hasher, self.size);
        hash_i64(hasher, self.mtime_secs);
        hash_i64(hasher, self.mtime_nsecs);
        hash_i64(hasher, self.ctime_secs);
        hash_i64(hasher, self.ctime_nsecs);
    }

    pub(crate) fn encode(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.dev.to_le_bytes());
        out.extend_from_slice(&self.ino.to_le_bytes());
        out.extend_from_slice(&self.size.to_le_bytes());
        out.extend_from_slice(&self.mtime_secs.to_le_bytes());
        out.extend_from_slice(&self.mtime_nsecs.to_le_bytes());
        out.extend_from_slice(&self.ctime_secs.to_le_bytes());
        out.extend_from_slice(&self.ctime_nsecs.to_le_bytes());
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
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

/// Retry an advisory-flock transition after signals.
pub(crate) fn flock_retry(
    file: impl std::os::fd::AsFd,
    operation: rustix::fs::FlockOperation,
) -> rustix::io::Result<()> {
    loop {
        match rustix::fs::flock(&file, operation) {
            Err(error) if error == rustix::io::Errno::INTR => continue,
            result => return result,
        }
    }
}

fn open_lock_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open content coordination lock {}", path.display()))
}

fn ensure_content_dirs(root: &Path) -> Result<()> {
    for path in [
        root.join(FILE_DIGEST_MEMO_DIR),
        root.join(FILE_DIGEST_LOCK_DIR),
        root.join(CONTENT_OBJECT_DIR),
    ] {
        std::fs::create_dir_all(&path)
            .with_context(|| format!("create shared content cache dir {}", path.display()))?;
    }
    Ok(())
}

fn cache_entry_is_old(metadata: &std::fs::Metadata, now: SystemTime, max_age: Duration) -> bool {
    metadata
        .modified()
        .ok()
        .and_then(|modified| now.duration_since(modified).ok())
        .is_some_and(|age| age >= max_age)
}

fn remove_if_present(path: &Path) -> Result<bool> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => {
            Err(error).with_context(|| format!("remove content cache {}", path.display()))
        }
    }
}

fn try_remove_coordinated_pair(
    namespace_locked: &File,
    lock_path: &Path,
    data_path: &Path,
) -> Result<bool> {
    let lock = open_lock_file(lock_path)?;
    match flock_retry(&lock, rustix::fs::FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => {
            let removed = remove_if_present(data_path)?;
            // The namespace gate is exclusively held, so no process can open
            // this pathname between unlinking the data and lock. A process
            // which already opened the lock either made the try-lock fail or
            // no longer needs the pathname.
            let _ = namespace_locked;
            remove_if_present(lock_path)?;
            Ok(removed)
        }
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => Ok(false),
        Err(error) => Err(error).with_context(|| {
            format!(
                "try-lock content cache entry for cleanup {}",
                lock_path.display()
            )
        }),
    }
}

fn parse_cache_key(name: &str, prefix: &str, suffix: &str) -> Option<String> {
    let key = name.strip_prefix(prefix)?.strip_suffix(suffix)?;
    (key.len() == 16 && key.bytes().all(|byte| byte.is_ascii_hexdigit())).then(|| key.to_string())
}

fn gc_content_cache_at(
    root: &Path,
    now: SystemTime,
    max_age: Duration,
    max_object_bytes: u64,
) -> Result<()> {
    ensure_content_dirs(root)?;
    let lock_dir = root.join(FILE_DIGEST_LOCK_DIR);
    let namespace_gate_path = lock_dir.join(FILE_DIGEST_NAMESPACE_GATE);
    let namespace_gate = open_lock_file(&namespace_gate_path)?;
    match flock_retry(
        &namespace_gate,
        rustix::fs::FlockOperation::NonBlockingLockExclusive,
    ) {
        Ok(()) => {}
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => return Ok(()),
        Err(error) => return Err(error).context("lock shared content namespace for cleanup"),
    }

    let digest_dir = root.join(FILE_DIGEST_MEMO_DIR);
    for entry in std::fs::read_dir(&digest_dir)
        .with_context(|| format!("scan content digest cache {}", digest_dir.display()))?
    {
        let entry = entry.context("read content digest cache entry")?;
        let name = entry.file_name();
        let Some(key) = name
            .to_str()
            .and_then(|name| parse_cache_key(name, "", ".digest"))
        else {
            continue;
        };
        let metadata = entry
            .metadata()
            .with_context(|| format!("stat content digest memo {}", entry.path().display()))?;
        if metadata.is_file() && cache_entry_is_old(&metadata, now, max_age) {
            try_remove_coordinated_pair(
                &namespace_gate,
                &lock_dir.join(format!("digest-{key}.lock")),
                &entry.path(),
            )?;
        }
    }

    struct ObjectCandidate {
        path: PathBuf,
        lock: PathBuf,
        modified: SystemTime,
        len: u64,
        expired: bool,
    }
    let object_dir = root.join(CONTENT_OBJECT_DIR);
    let mut objects = Vec::new();
    let mut total_bytes = 0u64;
    for entry in std::fs::read_dir(&object_dir)
        .with_context(|| format!("scan content object cache {}", object_dir.display()))?
    {
        let entry = entry.context("read content object cache entry")?;
        let name = entry.file_name();
        let Some(key) = name
            .to_str()
            .and_then(|name| parse_cache_key(name, "", ".object"))
        else {
            continue;
        };
        let metadata = entry
            .metadata()
            .with_context(|| format!("stat content object {}", entry.path().display()))?;
        if !metadata.is_file() {
            continue;
        }
        total_bytes = total_bytes.saturating_add(metadata.len());
        objects.push(ObjectCandidate {
            path: entry.path(),
            lock: lock_dir.join(format!("object-{key}.lock")),
            modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            len: metadata.len(),
            expired: cache_entry_is_old(&metadata, now, max_age),
        });
    }
    objects.sort_by_key(|candidate| candidate.modified);
    for object in objects {
        if !object.expired && total_bytes <= max_object_bytes {
            continue;
        }
        if try_remove_coordinated_pair(&namespace_gate, &object.lock, &object.path)? {
            total_bytes = total_bytes.saturating_sub(object.len);
        }
    }

    // Reclaim old orphan per-key locks left by a killed process after its data
    // entry was already removed. Only the fixed current namespace is parsed.
    for entry in std::fs::read_dir(&lock_dir)
        .with_context(|| format!("scan content lock cache {}", lock_dir.display()))?
    {
        let entry = entry.context("read content lock cache entry")?;
        let name = entry.file_name();
        let Some((kind, key)) = name.to_str().and_then(|name| {
            parse_cache_key(name, "digest-", ".lock")
                .map(|key| ("digest", key))
                .or_else(|| parse_cache_key(name, "object-", ".lock").map(|key| ("object", key)))
        }) else {
            continue;
        };
        let data_path = match kind {
            "digest" => digest_dir.join(format!("{key}.digest")),
            "object" => object_dir.join(format!("{key}.object")),
            _ => unreachable!(),
        };
        let metadata = entry
            .metadata()
            .with_context(|| format!("stat content lock {}", entry.path().display()))?;
        if !data_path.exists() && cache_entry_is_old(&metadata, now, max_age) {
            try_remove_coordinated_pair(&namespace_gate, &entry.path(), &data_path)?;
        }
    }

    let stamp = root.join(CONTENT_GC_STAMP);
    let stamp_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&stamp)
        .with_context(|| format!("update content GC stamp {}", stamp.display()))?;
    drop(stamp_file);
    Ok(())
}

fn maybe_gc_content_cache(root: &Path) -> Result<()> {
    let stamp = root.join(CONTENT_GC_STAMP);
    if stamp.metadata().ok().is_some_and(|metadata| {
        !cache_entry_is_old(&metadata, SystemTime::now(), CONTENT_GC_INTERVAL)
    }) {
        return Ok(());
    }
    gc_content_cache_at(
        root,
        SystemTime::now(),
        CONTENT_MAX_AGE,
        CONTENT_MAX_OBJECT_BYTES,
    )
}

/// A per-key coordination inode opened while holding its namespace gate.
///
/// Keeping the gate until the per-key flock is established closes the
/// open-before-flock unlink race for cache namespaces with garbage collection.
pub(crate) struct CoordinationFile {
    file: File,
    namespace_gate: Option<File>,
}

impl CoordinationFile {
    pub(crate) fn try_lock_exclusive(&mut self) -> rustix::io::Result<()> {
        flock_retry(
            &self.file,
            rustix::fs::FlockOperation::NonBlockingLockExclusive,
        )
    }

    pub(crate) fn lock_shared(&mut self) -> rustix::io::Result<()> {
        flock_retry(&self.file, rustix::fs::FlockOperation::LockShared)
    }

    pub(crate) fn try_lock_shared(&mut self) -> rustix::io::Result<()> {
        rustix::fs::flock(
            &self.file,
            rustix::fs::FlockOperation::NonBlockingLockShared,
        )
    }

    /// Enter the kernel writer queue after a winner vanished without
    /// publication. A blocking writer cannot be starved by a reader herd.
    pub(crate) fn lock_exclusive(&mut self) -> rustix::io::Result<()> {
        flock_retry(&self.file, rustix::fs::FlockOperation::LockExclusive)
    }

    pub(crate) fn release_namespace_gate(&mut self) {
        self.namespace_gate.take();
    }
}

pub(crate) fn open_coord_file(
    namespace_gate_path: &Path,
    lock_path: &Path,
) -> Result<CoordinationFile> {
    let namespace_gate = open_lock_file(namespace_gate_path)?;
    flock_retry(&namespace_gate, rustix::fs::FlockOperation::LockShared).with_context(|| {
        format!(
            "lock content coordination namespace {}",
            namespace_gate_path.display()
        )
    })?;
    let file = open_lock_file(lock_path)?;
    Ok(CoordinationFile {
        file,
        namespace_gate: Some(namespace_gate),
    })
}

/// Load one published entry or elect exactly one builder across processes.
///
/// The first contender takes nonblocking `LOCK_EX`. Losers wait under
/// `LOCK_SH`, then wake and read the publication concurrently. If the winner
/// exits without publishing, one waiter joins the blocking writer queue as the
/// successor instead of repeatedly racing a reader herd.
pub(crate) fn load_or_build<T, L, B>(
    namespace_gate_path: &Path,
    lock_path: &Path,
    subject: &str,
    load: L,
    build: B,
) -> Result<T>
where
    L: FnMut() -> Result<Option<T>>,
    B: FnOnce() -> Result<T>,
{
    load_or_build_with_wait(
        namespace_gate_path,
        lock_path,
        subject,
        load,
        build,
        |coordination| coordination.lock_shared().map_err(anyhow::Error::from),
        |coordination| coordination.lock_exclusive().map_err(anyhow::Error::from),
    )
}

/// [`load_or_build`] with caller-owned behavior around contended waits.
///
/// Cache layers whose producer can legitimately run for minutes use this to
/// surface a heartbeat while retaining this module's exact election,
/// successor, and namespace-gate semantics. Each hook must ultimately acquire
/// the requested lock on the supplied coordination file.
pub(crate) fn load_or_build_with_wait<T, L, B, WS, WX>(
    namespace_gate_path: &Path,
    lock_path: &Path,
    subject: &str,
    mut load: L,
    build: B,
    mut wait_shared: WS,
    mut wait_exclusive: WX,
) -> Result<T>
where
    L: FnMut() -> Result<Option<T>>,
    B: FnOnce() -> Result<T>,
    WS: FnMut(&mut CoordinationFile) -> Result<()>,
    WX: FnMut(&mut CoordinationFile) -> Result<()>,
{
    if let Some(value) = load()? {
        return Ok(value);
    }

    let mut build = Some(build);
    let mut wait_for_successor = false;
    loop {
        let mut coordination = open_coord_file(namespace_gate_path, lock_path)?;
        if wait_for_successor {
            wait_exclusive(&mut coordination)
                .with_context(|| format!("elect successor {subject} builder"))?;
            coordination.release_namespace_gate();
            if let Some(value) = load()? {
                return Ok(value);
            }
            return build
                .take()
                .context("content builder closure was already consumed")?();
        }
        match coordination.try_lock_exclusive() {
            Ok(()) => {
                coordination.release_namespace_gate();
                if let Some(value) = load()? {
                    return Ok(value);
                }
                return build
                    .take()
                    .context("content builder closure was already consumed")?(
                );
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                wait_shared(&mut coordination).with_context(|| format!("wait for {subject}"))?;
                coordination.release_namespace_gate();
                if let Some(value) = load()? {
                    return Ok(value);
                }
                wait_for_successor = true;
            }
            Err(error) => {
                return Err(error).with_context(|| format!("elect {subject} builder"));
            }
        }
    }
}

pub(crate) fn open_cache_record(
    path: &Path,
    subject: &str,
) -> Result<Option<(File, StableFileIdentity)>> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("open {subject} {}", path.display()));
        }
    };
    let metadata = file
        .metadata()
        .with_context(|| format!("stat {subject} {}", path.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "{subject} is not a regular file: {}",
        path.display()
    );
    Ok(Some((file, StableFileIdentity::from_metadata(&metadata))))
}

pub(crate) fn read_fixed_cache_record<const N: usize>(
    path: &Path,
    subject: &str,
) -> Result<Option<[u8; N]>> {
    let Some((file, identity)) = open_cache_record(path, subject)? else {
        return Ok(None);
    };
    anyhow::ensure!(
        identity.size == N as u64,
        "{subject} has invalid length {}: {}",
        identity.size,
        path.display()
    );
    let mut bytes = [0u8; N];
    pread_exact(&file, 0, &mut bytes, subject)?;
    anyhow::ensure!(
        StableFileIdentity::from_file(&file)? == identity,
        "{subject} changed while reading: {}",
        path.display()
    );
    Ok(Some(bytes))
}

pub(crate) fn file_digest_identity_key(identity: StableFileIdentity) -> u64 {
    let mut hasher = fixed_hasher();
    hash_len_prefixed(&mut hasher, b"ktstr-file-digest-identity");
    hash_u32(&mut hasher, FILE_DIGEST_SCHEMA);
    identity.hash_into(&mut hasher);
    hasher.finish()
}

fn read_file_digest_record(
    record_path: &Path,
    identity: StableFileIdentity,
) -> Result<Option<u64>> {
    let Some(bytes) =
        read_fixed_cache_record::<FILE_DIGEST_RECORD_LEN>(record_path, "file digest memo")?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        &bytes[..8] == FILE_DIGEST_MAGIC,
        "file digest memo magic mismatch"
    );
    anyhow::ensure!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) == FILE_DIGEST_SCHEMA,
        "file digest memo schema mismatch"
    );
    anyhow::ensure!(
        bytes[12..16] == [0; 4],
        "file digest memo reserved bytes changed"
    );
    anyhow::ensure!(
        StableFileIdentity::decode(&bytes[16..72])? == identity,
        "file digest memo identity collision"
    );
    let checksum = u64::from_le_bytes(bytes[80..88].try_into().unwrap());
    let mut canonical = bytes;
    canonical[80..88].fill(0);
    anyhow::ensure!(
        ahash_bytes(&canonical) == checksum,
        "file digest memo checksum mismatch"
    );
    Ok(Some(u64::from_le_bytes(
        canonical[72..80].try_into().unwrap(),
    )))
}

fn hash_pinned_file(file: &File, identity: StableFileIdentity) -> Result<u64> {
    anyhow::ensure!(
        StableFileIdentity::from_file(file)?.same_open_content_version(identity),
        "pinned input changed before hashing"
    );
    let mut hasher = fixed_hasher();
    let mut offset = 0u64;
    let buffer_len = usize::try_from(identity.size.min(CONTENT_HASH_CHUNK_LEN as u64))?;
    let mut buffer = vec![0u8; buffer_len];
    while offset < identity.size {
        let remaining = usize::try_from((identity.size - offset).min(buffer.len() as u64))?;
        pread_exact(
            file,
            offset,
            &mut buffer[..remaining],
            "pinned input for digest",
        )?;
        hasher.write(&buffer[..remaining]);
        offset = offset
            .checked_add(u64::try_from(remaining)?)
            .context("pinned input digest offset overflow")?;
    }
    let digest = hasher.finish();
    anyhow::ensure!(
        StableFileIdentity::from_file(file)?.same_open_content_version(identity),
        "pinned input changed while hashing its content"
    );
    Ok(digest)
}

/// Capture the pinned source revision immediately before a cache lookup.
///
/// The caller-provided identity was captured when the fd was opened. A size or
/// mtime change means the content revision changed and is always fatal. A
/// ctime-only change can be an unlink/rename of the still-pinned inode, so it
/// is retained for exact hash validation after a cache hit.
fn source_identity_before_cache_lookup(
    file: &File,
    opened_identity: StableFileIdentity,
    subject: &str,
) -> Result<StableFileIdentity> {
    let current = StableFileIdentity::from_file(file)?;
    anyhow::ensure!(
        current.same_open_content_version(opened_identity),
        "{subject} source changed before cache lookup"
    );
    Ok(current)
}

/// Validate a cache lookup/build result against the pinned source revision.
///
/// The common case is metadata-only: identical full identities on both sides
/// of the lookup prove that no revision transition occurred. If ctime alone
/// changed, rehash the fd against the cache key while requiring the complete
/// post-lookup identity to remain stable around that read. This accepts a
/// harmless unlink/rename but rejects an in-place rewrite whose size and mtime
/// were restored.
fn validate_source_after_cache_operation(
    file: &File,
    opened_identity: StableFileIdentity,
    before_lookup: StableFileIdentity,
    expected_hash: u64,
    subject: &str,
) -> Result<()> {
    let after_lookup = StableFileIdentity::from_file(file)?;
    anyhow::ensure!(
        after_lookup.same_open_content_version(opened_identity),
        "{subject} source changed during cache lookup"
    );
    if before_lookup == opened_identity && after_lookup == opened_identity {
        return Ok(());
    }

    anyhow::ensure!(
        before_lookup.same_open_content_version(opened_identity),
        "{subject} source changed before cache lookup"
    );
    let actual_hash = hash_pinned_file_exact_identity(file, after_lookup)?;
    anyhow::ensure!(
        actual_hash == expected_hash,
        "{subject} cache result does not match the pinned source after a ctime change"
    );
    Ok(())
}

fn hash_pinned_file_exact_identity(file: &File, identity: StableFileIdentity) -> Result<u64> {
    anyhow::ensure!(
        StableFileIdentity::from_file(file)? == identity,
        "pinned input changed before exact content validation"
    );
    let mut hasher = fixed_hasher();
    let mut offset = 0u64;
    let buffer_len = usize::try_from(identity.size.min(CONTENT_HASH_CHUNK_LEN as u64))?;
    let mut buffer = vec![0u8; buffer_len];
    while offset < identity.size {
        let remaining = usize::try_from((identity.size - offset).min(buffer.len() as u64))?;
        pread_exact(
            file,
            offset,
            &mut buffer[..remaining],
            "pinned input for exact content validation",
        )?;
        hasher.write(&buffer[..remaining]);
        offset = offset
            .checked_add(u64::try_from(remaining)?)
            .context("pinned input exact validation offset overflow")?;
    }
    let digest = hasher.finish();
    anyhow::ensure!(
        StableFileIdentity::from_file(file)? == identity,
        "pinned input changed during exact content validation"
    );
    Ok(digest)
}

/// Digest one pinned file revision exactly once across processes.
pub(crate) fn cached_file_digest(file: &File, identity: StableFileIdentity) -> Result<u64> {
    let root = super::resolve_cache_root_with_suffix("content")
        .context("resolve machine-wide content digest cache")?;
    cached_file_digest_at_root(&root, file, identity)
}

/// Implementation with an explicit root for isolated cache tests.
pub(crate) fn cached_file_digest_at_root(
    root: &Path,
    file: &File,
    identity: StableFileIdentity,
) -> Result<u64> {
    ensure_content_dirs(root)?;
    maybe_gc_content_cache(root)?;
    let memo_dir = root.join(FILE_DIGEST_MEMO_DIR);
    let lock_dir = root.join(FILE_DIGEST_LOCK_DIR);
    let namespace_gate = lock_dir.join(FILE_DIGEST_NAMESPACE_GATE);

    let identity_key = file_digest_identity_key(identity);
    let record_path = memo_dir.join(format!("{identity_key:016x}.digest"));
    let lock_path = lock_dir.join(format!("digest-{identity_key:016x}.lock"));
    let before_lookup = source_identity_before_cache_lookup(file, identity, "file digest")?;
    let digest = load_or_build(
        &namespace_gate,
        &lock_path,
        &format!("file digest {}", record_path.display()),
        || read_file_digest_record(&record_path, identity),
        || {
            let digest = hash_pinned_file(file, identity)?;
            let mut bytes = Vec::with_capacity(FILE_DIGEST_RECORD_LEN);
            bytes.extend_from_slice(FILE_DIGEST_MAGIC);
            bytes.extend_from_slice(&FILE_DIGEST_SCHEMA.to_le_bytes());
            bytes.extend_from_slice(&[0; 4]);
            identity.encode(&mut bytes);
            bytes.extend_from_slice(&digest.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
            let checksum = ahash_bytes(&bytes);
            bytes[80..88].copy_from_slice(&checksum.to_le_bytes());
            debug_assert_eq!(bytes.len(), FILE_DIGEST_RECORD_LEN);

            let mut temporary = tempfile::Builder::new()
                .prefix(&format!(".tmp-digest-{identity_key:016x}-"))
                .tempfile_in(&memo_dir)
                .context("create file digest memo temp")?;
            temporary
                .write_all(&bytes)
                .context("write file digest memo")?;
            // This is reconstructible cache state. The checksum and fixed
            // envelope make a torn post-crash record fail closed, while
            // persist keeps live readers from observing a partial write.
            temporary
                .persist(&record_path)
                .map_err(|error| error.error)
                .with_context(|| format!("publish file digest memo {}", record_path.display()))?;
            Ok(digest)
        },
    )?;
    validate_source_after_cache_operation(file, identity, before_lookup, digest, "file digest")?;
    Ok(digest)
}

/// Open a regular file without blocking on a substituted FIFO and pin its
/// revision for later digest/read verification.
pub(crate) fn open_pinned_file(path: &Path) -> Result<(File, StableFileIdentity)> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
        .with_context(|| format!("open content-addressed input {}", path.display()))?;
    let identity = StableFileIdentity::from_file(&file)
        .with_context(|| format!("stat content-addressed input {}", path.display()))?;
    Ok((file, identity))
}

fn content_cache_root() -> Result<PathBuf> {
    super::resolve_cache_root_with_suffix("content").context("resolve machine-wide content cache")
}

fn content_object_path(root: &Path, content_hash: u64) -> PathBuf {
    root.join(CONTENT_OBJECT_DIR)
        .join(format!("{content_hash:016x}.object"))
}

fn content_object_lock_path(root: &Path, content_hash: u64) -> PathBuf {
    root.join(FILE_DIGEST_LOCK_DIR)
        .join(format!("object-{content_hash:016x}.lock"))
}

fn open_content_object_at(path: &Path, expected_len: u64) -> Result<Option<File>> {
    use std::os::unix::fs::PermissionsExt as _;

    let Some((file, identity)) = open_cache_record(path, "content object")? else {
        return Ok(None);
    };
    anyhow::ensure!(
        identity.size == expected_len,
        "content object {} has length {}, expected {expected_len}",
        path.display(),
        identity.size
    );
    let mode = file
        .metadata()
        .with_context(|| format!("stat content object mode {}", path.display()))?
        .permissions()
        .mode();
    anyhow::ensure!(
        mode & 0o222 == 0 && mode & 0o111 != 0,
        "content object must be read-only executable: {}",
        path.display()
    );
    Ok(Some(file))
}

/// Open one immutable machine-wide content object and validate its byte length.
#[cfg(test)]
pub(crate) fn open_content_object(content_hash: u64, expected_len: u64) -> Result<Option<File>> {
    let root = content_cache_root()?;
    ensure_content_dirs(&root)?;
    maybe_gc_content_cache(&root)?;
    open_content_object_at(&content_object_path(&root, content_hash), expected_len)
}

/// Publish a pinned input as one immutable content object.
///
/// FICLONE shares source extents when the filesystem supports it. Other local
/// filesystems stream the pinned descriptor once into the same immutable CAS
/// inode; all consumers still mmap that single inode with private/COW
/// semantics. Publication is atomic and source mutation is checked on both
/// sides of the copy.
fn publish_content_object(
    expected_hash: u64,
    source: &File,
    source_identity: StableFileIdentity,
    final_path: &Path,
) -> Result<File> {
    let parent = final_path
        .parent()
        .context("content object path has no parent")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create content object dir {}", parent.display()))?;
    let before_copy = StableFileIdentity::from_file(source)?;
    anyhow::ensure!(
        before_copy.same_open_content_version(source_identity),
        "content source changed before CAS publication"
    );

    let mut temporary = tempfile::Builder::new()
        .prefix(".content-object-")
        .tempfile_in(parent)
        .context("create content object temp")?;
    match crate::reflink::ficlone(temporary.as_file(), source) {
        Ok(()) => {}
        Err(error)
            if error.raw_os_error().is_some_and(|code| {
                matches!(code, libc::EXDEV | libc::EOPNOTSUPP | libc::ENOTTY)
            }) =>
        {
            let mut offset = 0u64;
            let mut buffer =
                vec![
                    0u8;
                    usize::try_from(source_identity.size.min(CONTENT_HASH_CHUNK_LEN as u64))?
                ];
            while offset < source_identity.size {
                let chunk_len =
                    usize::try_from((source_identity.size - offset).min(buffer.len() as u64))?;
                pread_exact(
                    source,
                    offset,
                    &mut buffer[..chunk_len],
                    "pinned content object source",
                )?;
                temporary
                    .write_all(&buffer[..chunk_len])
                    .context("write content object temp")?;
                offset = offset
                    .checked_add(u64::try_from(chunk_len)?)
                    .context("content object copy offset overflow")?;
            }
        }
        Err(error) => return Err(error).context("FICLONE pinned content object"),
    }
    let after_copy = StableFileIdentity::from_file(source)?;
    anyhow::ensure!(
        after_copy.same_open_content_version(source_identity),
        "content source changed during CAS publication"
    );
    if before_copy != source_identity || after_copy != source_identity || before_copy != after_copy
    {
        let temporary_identity = StableFileIdentity::from_file(temporary.as_file())
            .context("stat content object temp for source-race validation")?;
        let actual_hash = hash_pinned_file_exact_identity(temporary.as_file(), temporary_identity)
            .context("hash content object temp after source revision transition")?;
        anyhow::ensure!(
            actual_hash == expected_hash,
            "content object captured bytes from a transient source revision"
        );
    }
    temporary
        .as_file()
        .set_permissions(std::fs::Permissions::from_mode(0o555))
        .context("make content object read-only executable")?;
    // The object is reconstructible from the still-pinned source. Atomic
    // publication and chmod-before-rename provide the live-process
    // invariants without putting every cold builder behind storage barriers.
    let published = temporary
        .persist(final_path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish content object {}", final_path.display()))?;
    Ok(published)
}

/// Open or publish one immutable object under the machine-wide content key.
///
/// The content namespace owns this election rather than any caller-specific
/// analysis/preparation cache, so identical bytes converge on one inode even
/// when different components and checkouts request them concurrently.
fn open_or_publish_content_object_at_root(
    root: &Path,
    content_hash: u64,
    source: &File,
    source_identity: StableFileIdentity,
) -> Result<File> {
    ensure_content_dirs(root)?;
    maybe_gc_content_cache(root)?;
    let object_path = content_object_path(root, content_hash);
    let lock_path = content_object_lock_path(root, content_hash);
    let namespace_gate = root
        .join(FILE_DIGEST_LOCK_DIR)
        .join(FILE_DIGEST_NAMESPACE_GATE);
    let before_lookup =
        source_identity_before_cache_lookup(source, source_identity, "content object")?;
    let object = load_or_build(
        &namespace_gate,
        &lock_path,
        &format!("content object {content_hash:016x}"),
        || open_content_object_at(&object_path, source_identity.size),
        || publish_content_object(content_hash, source, source_identity, &object_path),
    )?;
    validate_source_after_cache_operation(
        source,
        source_identity,
        before_lookup,
        content_hash,
        "content object",
    )?;
    Ok(object)
}

pub(crate) fn open_or_publish_content_object(
    content_hash: u64,
    source: &File,
    source_identity: StableFileIdentity,
) -> Result<File> {
    let root = content_cache_root()?;
    open_or_publish_content_object_at_root(&root, content_hash, source, source_identity)
}

/// Shared-lock lease over one immutable content object.
///
/// The path is the canonical machine-CAS pathname used by every process. The
/// held per-key shared flock prevents GC from unlinking that pathname while a
/// child manifest refers to it. Kernel flock ownership makes the lease
/// crash-recoverable: process exit, including SIGKILL, releases it without a
/// stale lease record.
pub(crate) struct ContentObjectLease {
    _file: File,
    _coordination: CoordinationFile,
    path: PathBuf,
}

impl ContentObjectLease {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

fn lease_content_object_at_root(
    root: &Path,
    content_hash: u64,
    expected_len: u64,
) -> Result<Option<ContentObjectLease>> {
    let object_path = content_object_path(root, content_hash);
    let lock_path = content_object_lock_path(root, content_hash);
    let namespace_gate = root
        .join(FILE_DIGEST_LOCK_DIR)
        .join(FILE_DIGEST_NAMESPACE_GATE);
    let mut coordination = open_coord_file(&namespace_gate, &lock_path)?;
    coordination
        .lock_shared()
        .with_context(|| format!("lease content object {content_hash:016x}"))?;
    coordination.release_namespace_gate();
    let Some(file) = open_content_object_at(&object_path, expected_len)? else {
        return Ok(None);
    };
    let path = std::fs::canonicalize(&object_path)
        .with_context(|| format!("canonicalize content object {}", object_path.display()))?;
    Ok(Some(ContentObjectLease {
        _file: file,
        _coordination: coordination,
        path,
    }))
}

/// Publish and lease one immutable object under the machine-wide content key.
///
/// Publication may briefly race an unrelated GC pass before the shared lease
/// is established. If GC wins that gap, this retries publication; once the
/// shared flock is held, the canonical pathname is stable for the lease
/// lifetime.
pub(crate) fn open_or_publish_content_object_lease(
    content_hash: u64,
    source: &File,
    source_identity: StableFileIdentity,
) -> Result<ContentObjectLease> {
    let root = content_cache_root()?;
    loop {
        drop(open_or_publish_content_object_at_root(
            &root,
            content_hash,
            source,
            source_identity,
        )?);
        if let Some(lease) =
            lease_content_object_at_root(&root, content_hash, source_identity.size)?
        {
            return Ok(lease);
        }
    }
}

/// Read-only private mapping of one immutable content object.
///
/// `map_copy_read_only` requests MAP_PRIVATE: every process maps the same
/// page-cache pages and receives COW pages only if a future parser explicitly
/// asks for a writable view. Empty files carry no mapping but retain the same
/// slice API.
pub(crate) struct CowMappedFile {
    _file: File,
    mapping: Option<memmap2::Mmap>,
}

impl CowMappedFile {
    pub(crate) fn map(file: File) -> Result<Self> {
        let len = file
            .metadata()
            .context("stat content object for mmap")?
            .len();
        let mapping = if len == 0 {
            None
        } else {
            // SAFETY: the fd names a private, read-only CAS object whose inode
            // remains open for the mapping lifetime. Callers publish it
            // atomically and never mutate it in place.
            Some(
                unsafe { memmap2::MmapOptions::new().map_copy_read_only(&file) }
                    .context("mmap content object MAP_PRIVATE read-only")?,
            )
        };
        Ok(Self {
            _file: file,
            mapping,
        })
    }
}

impl AsRef<[u8]> for CowMappedFile {
    fn as_ref(&self) -> &[u8] {
        self.mapping
            .as_ref()
            .map_or(&[], |mapping| mapping.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn restore_original_mtime(file: &File, identity: StableFileIdentity) {
        use std::os::fd::AsRawFd as _;

        let times = [
            libc::timespec {
                tv_sec: 0,
                tv_nsec: libc::UTIME_OMIT,
            },
            libc::timespec {
                tv_sec: identity.mtime_secs,
                tv_nsec: identity.mtime_nsecs,
            },
        ];
        // SAFETY: `file` is live for the call and `times` points to two valid
        // timespec values. UTIME_OMIT leaves atime unchanged.
        let result = unsafe { libc::futimens(file.as_raw_fd(), times.as_ptr()) };
        assert_eq!(
            result,
            0,
            "restore source mtime: {}",
            std::io::Error::last_os_error()
        );
    }

    fn rewrite_same_size_and_restore_mtime(
        path: &Path,
        replacement: &[u8],
        identity: StableFileIdentity,
    ) {
        use std::io::Write as _;

        assert_eq!(replacement.len() as u64, identity.size);
        let mut writer = OpenOptions::new().write(true).open(path).unwrap();
        writer.write_all(replacement).unwrap();
        writer.sync_all().unwrap();
        restore_original_mtime(&writer, identity);
    }

    #[test]
    fn digest_memo_accepts_harmless_ctime_only_unlink_after_exact_rehash() {
        let root = tempfile::TempDir::new().unwrap();
        let sources = tempfile::TempDir::new().unwrap();
        let source_path = sources.path().join("source");
        let alias_path = sources.path().join("alias");
        std::fs::write(&source_path, b"same immutable bytes").unwrap();
        std::fs::hard_link(&source_path, &alias_path).unwrap();
        let (source, identity) = open_pinned_file(&source_path).unwrap();
        let expected = cached_file_digest_at_root(root.path(), &source, identity).unwrap();

        std::fs::remove_file(alias_path).unwrap();
        let changed = StableFileIdentity::from_file(&source).unwrap();
        assert!(changed.same_open_content_version(identity));
        assert_ne!(changed, identity, "unlink must advance the inode ctime");
        assert_eq!(
            cached_file_digest_at_root(root.path(), &source, identity).unwrap(),
            expected,
        );
    }

    #[test]
    fn digest_memo_rejects_same_size_rewrite_with_restored_mtime() {
        let root = tempfile::TempDir::new().unwrap();
        let sources = tempfile::TempDir::new().unwrap();
        let source_path = sources.path().join("source");
        std::fs::write(&source_path, b"old scheduler bytes").unwrap();
        let (source, identity) = open_pinned_file(&source_path).unwrap();
        cached_file_digest_at_root(root.path(), &source, identity).unwrap();

        rewrite_same_size_and_restore_mtime(&source_path, b"new scheduler bytes", identity);
        let changed = StableFileIdentity::from_file(&source).unwrap();
        assert!(changed.same_open_content_version(identity));
        assert_ne!(changed, identity, "rewrite must advance the inode ctime");
        let error = cached_file_digest_at_root(root.path(), &source, identity).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not match the pinned source"),
            "unexpected stale digest rejection: {error:#}",
        );
    }

    #[test]
    fn content_object_hit_rejects_same_size_rewrite_with_restored_mtime() {
        let root = tempfile::TempDir::new().unwrap();
        let sources = tempfile::TempDir::new().unwrap();
        let source_path = sources.path().join("source");
        std::fs::write(&source_path, b"old scheduler bytes").unwrap();
        let (source, identity) = open_pinned_file(&source_path).unwrap();
        let content_hash = hash_pinned_file(&source, identity).unwrap();
        drop(
            open_or_publish_content_object_at_root(root.path(), content_hash, &source, identity)
                .unwrap(),
        );

        rewrite_same_size_and_restore_mtime(&source_path, b"new scheduler bytes", identity);
        let error =
            open_or_publish_content_object_at_root(root.path(), content_hash, &source, identity)
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not match the pinned source"),
            "unexpected stale object rejection: {error:#}",
        );
    }

    #[test]
    fn content_object_miss_never_publishes_a_restored_mtime_revision_under_old_hash() {
        let root = tempfile::TempDir::new().unwrap();
        let sources = tempfile::TempDir::new().unwrap();
        let source_path = sources.path().join("source");
        std::fs::write(&source_path, b"old scheduler bytes").unwrap();
        let (source, identity) = open_pinned_file(&source_path).unwrap();
        let content_hash = hash_pinned_file(&source, identity).unwrap();

        rewrite_same_size_and_restore_mtime(&source_path, b"new scheduler bytes", identity);
        let error =
            open_or_publish_content_object_at_root(root.path(), content_hash, &source, identity)
                .unwrap_err();
        assert!(
            format!("{error:#}").contains("transient source revision"),
            "unexpected stale publication rejection: {error:#}",
        );
        assert!(
            !content_object_path(root.path(), content_hash).exists(),
            "a mismatched revision must never be published under the old key",
        );
    }

    #[test]
    fn gc_bounds_objects_digests_and_locks_but_skips_live_builder() {
        let root = tempfile::TempDir::new().unwrap();
        ensure_content_dirs(root.path()).unwrap();
        let digest_dir = root.path().join(FILE_DIGEST_MEMO_DIR);
        let object_dir = root.path().join(CONTENT_OBJECT_DIR);
        let lock_dir = root.path().join(FILE_DIGEST_LOCK_DIR);

        let digest_key = "1111111111111111";
        let object_key = "2222222222222222";
        let live_key = "3333333333333333";
        std::fs::write(digest_dir.join(format!("{digest_key}.digest")), b"memo").unwrap();
        std::fs::write(lock_dir.join(format!("digest-{digest_key}.lock")), b"").unwrap();
        std::fs::write(
            object_dir.join(format!("{object_key}.object")),
            vec![0u8; 4096],
        )
        .unwrap();
        std::fs::write(lock_dir.join(format!("object-{object_key}.lock")), b"").unwrap();
        std::fs::write(
            object_dir.join(format!("{live_key}.object")),
            vec![0u8; 4096],
        )
        .unwrap();
        let live_lock_path = lock_dir.join(format!("object-{live_key}.lock"));
        let live_lock = open_lock_file(&live_lock_path).unwrap();
        flock_retry(&live_lock, rustix::fs::FlockOperation::LockExclusive).unwrap();

        let future = SystemTime::now() + CONTENT_MAX_AGE + Duration::from_secs(1);
        gc_content_cache_at(root.path(), future, CONTENT_MAX_AGE, 0).unwrap();
        assert!(!digest_dir.join(format!("{digest_key}.digest")).exists());
        assert!(!lock_dir.join(format!("digest-{digest_key}.lock")).exists());
        assert!(!object_dir.join(format!("{object_key}.object")).exists());
        assert!(!lock_dir.join(format!("object-{object_key}.lock")).exists());
        assert!(
            object_dir.join(format!("{live_key}.object")).exists(),
            "GC must not unlink an object whose per-key builder lock is live"
        );
        assert!(live_lock_path.exists());

        drop(live_lock);
        gc_content_cache_at(root.path(), future, CONTENT_MAX_AGE, 0).unwrap();
        assert!(!object_dir.join(format!("{live_key}.object")).exists());
        assert!(!live_lock_path.exists());
        assert!(
            lock_dir.join(FILE_DIGEST_NAMESPACE_GATE).exists(),
            "the namespace gate itself is never a GC candidate"
        );
    }

    #[test]
    fn cow_mapping_is_private_and_uses_the_content_inode() {
        let root = tempfile::TempDir::new().unwrap();
        let path = root.path().join("object");
        std::fs::write(&path, vec![0x5a; 8192]).unwrap();
        let file = File::open(&path).unwrap();
        let inode = file.metadata().unwrap().ino();
        let mapped = CowMappedFile::map(file).unwrap();
        assert_eq!(mapped.as_ref()[0], 0x5a);

        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        let mapping = maps
            .lines()
            .find(|line| {
                line.split_whitespace()
                    .nth(4)
                    .and_then(|value| value.parse::<u64>().ok())
                    == Some(inode)
            })
            .expect("content inode must appear in /proc/self/maps");
        let permissions = mapping.split_whitespace().nth(1).unwrap();
        assert_eq!(
            permissions, "r--p",
            "content mapping must be read-only MAP_PRIVATE"
        );
    }

    #[test]
    fn executable_leases_converge_and_block_gc_until_drop() {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

        let root = tempfile::TempDir::new().unwrap();
        let sources = tempfile::TempDir::new().unwrap();
        let bytes = b"identical scheduler executable";
        let hash = ahash_bytes(bytes);
        let mut leases = Vec::new();
        for name in ["scheduler-a", "scheduler-b"] {
            let path = sources.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            let (file, identity) = open_pinned_file(&path).unwrap();
            drop(
                open_or_publish_content_object_at_root(root.path(), hash, &file, identity).unwrap(),
            );
            leases.push(
                lease_content_object_at_root(root.path(), hash, identity.size)
                    .unwrap()
                    .expect("published object lease"),
            );
        }

        assert_eq!(leases[0].path(), leases[1].path());
        let metadata = std::fs::metadata(leases[0].path()).unwrap();
        assert_eq!(metadata.permissions().mode() & 0o777, 0o555);
        assert_eq!(
            metadata.ino(),
            std::fs::metadata(leases[1].path()).unwrap().ino(),
            "identical scheduler bytes must share one machine-CAS inode"
        );

        let future = SystemTime::now() + CONTENT_MAX_AGE + Duration::from_secs(1);
        gc_content_cache_at(root.path(), future, CONTENT_MAX_AGE, 0).unwrap();
        assert!(
            leases[0].path().exists(),
            "shared lease must prevent GC from unlinking a live manifest path"
        );

        let object_path = leases[0].path().to_path_buf();
        drop(leases);
        gc_content_cache_at(root.path(), future, CONTENT_MAX_AGE, 0).unwrap();
        assert!(
            !object_path.exists(),
            "dropping every flock lease must make the object reclaimable"
        );
    }
}
