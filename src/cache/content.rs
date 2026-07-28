//! Shared content-addressed file coordination.
//!
//! Large immutable inputs are opened and pinned by file descriptor, their
//! stable inode revision is digested once across processes, and per-content
//! builders use one common nonblocking-winner/shared-reader/successor election
//! protocol. Callers own their record wire format; this module owns the
//! expensive-input and cross-process coordination invariants.

use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::fs::{File, OpenOptions};
use std::hash::{BuildHasher, Hasher};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{Duration, SystemTime};

use ahash::AHasher;

const FILE_DIGEST_MAGIC: &[u8; 8] = b"KTSTRDG\0";
const FILE_DIGEST_RECORD_LEN: usize = 88;
const FILE_DIGEST_SCHEMA: u32 = 1;
const CONTENT_HASH_CHUNK_LEN: usize = 1 << 20;
const FILE_DIGEST_MEMO_DIR: &str = "digests-v1";
const CONTENT_NAMESPACE_GATE: &str = ".namespace-gate-v3.lock";
const CONTENT_DIGEST_LOCK_PREFIX: &str = ".digest-lock-v3-";
const CONTENT_OBJECT_LOCK_PREFIX: &str = ".object-lock-v3-";
const CONTENT_OBJECT_DIR: &str = "objects-v3";
const CONTENT_STAGING_DIR: &str = ".staging-v3";
const CONTENT_RETIRED_V2_DIR: &str = ".retired-content-v2";
const ARTIFACT_REFERENCE_DIR: &str = "artifact-references-v1";
const ARTIFACT_REFERENCE_SCHEMA: u32 = 1;
const ARTIFACT_REFERENCE_MAX_BYTES: u64 = 64 << 20;
const CONTENT_GC_STAMP: &str = ".gc-v3";
const CONTENT_GC_INTERVAL: Duration = Duration::from_secs(60 * 60);
const CONTENT_MAX_AGE: Duration = Duration::from_secs(30 * 24 * 60 * 60);
const CONTENT_ORPHAN_TEMP_GRACE: Duration = Duration::from_secs(60);
const LEGACY_CONTENT_V2_ENTRIES: &[&str] = &["objects-v2", ".locks-v1", ".gc-v2"];

static CONTENT_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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

struct ContentCacheDirs {
    display_root: PathBuf,
    root: OwnedFd,
    digests: OwnedFd,
    objects: OwnedFd,
    staging: OwnedFd,
    retired_v2: OwnedFd,
    references: OwnedFd,
}

fn create_and_open_content_dir(root: &OwnedFd, display_root: &Path, name: &str) -> Result<OwnedFd> {
    match rustix::fs::mkdirat(root, name, rustix::fs::Mode::from_raw_mode(0o777)) {
        Ok(()) => {}
        Err(error) if error == rustix::io::Errno::EXIST => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "create shared content cache namespace {}",
                    display_root.join(name).display()
                )
            });
        }
    }
    rustix::fs::openat(
        root,
        name,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| {
        format!(
            "open shared content cache namespace without following links {}",
            display_root.join(name).display()
        )
    })
}

fn open_content_cache_dirs(root: &Path) -> Result<ContentCacheDirs> {
    let requested_root = if root.is_absolute() {
        root.to_path_buf()
    } else {
        std::env::current_dir()
            .context("resolve current directory for shared content cache")?
            .join(root)
    };
    std::fs::create_dir_all(&requested_root).with_context(|| {
        format!(
            "create shared content cache root {}",
            requested_root.display()
        )
    })?;
    let root_fd = rustix::fs::open(
        &requested_root,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| {
        format!(
            "open shared content cache root without following links {}",
            requested_root.display()
        )
    })?;
    let display_root = std::fs::read_link(format!("/proc/self/fd/{}", root_fd.as_raw_fd()))
        .with_context(|| {
            format!(
                "resolve canonical path of pinned shared content cache root {}",
                requested_root.display()
            )
        })?;
    anyhow::ensure!(
        display_root.is_absolute(),
        "pinned shared content cache root did not resolve to an absolute path: {}",
        display_root.display(),
    );
    let digests = create_and_open_content_dir(&root_fd, &display_root, FILE_DIGEST_MEMO_DIR)?;
    let objects = create_and_open_content_dir(&root_fd, &display_root, CONTENT_OBJECT_DIR)?;
    let staging = create_and_open_content_dir(&root_fd, &display_root, CONTENT_STAGING_DIR)?;
    let retired_v2 = create_and_open_content_dir(&root_fd, &display_root, CONTENT_RETIRED_V2_DIR)?;
    let references = create_and_open_content_dir(&root_fd, &display_root, ARTIFACT_REFERENCE_DIR)?;
    let dirs = ContentCacheDirs {
        display_root,
        root: root_fd,
        digests,
        objects,
        staging,
        retired_v2,
        references,
    };
    validate_content_cache_dirs_reachable(&dirs)?;
    #[cfg(test)]
    record_test_content_cache_dir_open(&requested_root);
    Ok(dirs)
}

#[cfg(test)]
fn test_content_cache_dir_opens()
-> &'static std::sync::Mutex<std::collections::BTreeMap<PathBuf, usize>> {
    static OPENS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::BTreeMap<PathBuf, usize>>,
    > = std::sync::OnceLock::new();
    OPENS.get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()))
}

#[cfg(test)]
fn record_test_content_cache_dir_open(root: &Path) {
    *test_content_cache_dir_opens()
        .lock()
        .expect("content cache directory open counter poisoned")
        .entry(root.to_path_buf())
        .or_default() += 1;
}

#[cfg(test)]
fn reset_test_content_cache_dir_open_count(root: &Path) {
    test_content_cache_dir_opens()
        .lock()
        .expect("content cache directory open counter poisoned")
        .remove(root);
}

#[cfg(test)]
fn test_content_cache_dir_open_count(root: &Path) -> usize {
    test_content_cache_dir_opens()
        .lock()
        .expect("content cache directory open counter poisoned")
        .get(root)
        .copied()
        .unwrap_or_default()
}

fn same_content_cache_inode(left: &rustix::fs::Stat, right: &rustix::fs::Stat) -> bool {
    left.st_dev == right.st_dev && left.st_ino == right.st_ino
}

fn validate_content_cache_dirs_reachable(dirs: &ContentCacheDirs) -> Result<()> {
    let reopened_root = rustix::fs::open(
        &dirs.display_root,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .map_err(|error| {
        anyhow::anyhow!(
            "shared content cache root was detached or replaced: {}: {error}",
            dirs.display_root.display()
        )
    })?;
    let pinned_root = rustix::fs::fstat(&dirs.root).context("stat pinned content cache root")?;
    let current_root =
        rustix::fs::fstat(&reopened_root).context("stat reachable content cache root")?;
    anyhow::ensure!(
        same_content_cache_inode(&pinned_root, &current_root),
        "shared content cache root was detached or replaced: {}",
        dirs.display_root.display(),
    );

    for (name, pinned) in [
        (FILE_DIGEST_MEMO_DIR, &dirs.digests),
        (CONTENT_OBJECT_DIR, &dirs.objects),
        (CONTENT_STAGING_DIR, &dirs.staging),
        (CONTENT_RETIRED_V2_DIR, &dirs.retired_v2),
        (ARTIFACT_REFERENCE_DIR, &dirs.references),
    ] {
        let reachable = rustix::fs::openat(
            &dirs.root,
            name,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )
        .map_err(|error| {
            anyhow::anyhow!(
                "shared content cache namespace was detached or replaced: {}: {error}",
                dirs.display_root.join(name).display()
            )
        })?;
        let pinned = rustix::fs::fstat(pinned)
            .with_context(|| format!("stat pinned content cache namespace {name}"))?;
        let reachable = rustix::fs::fstat(&reachable)
            .with_context(|| format!("stat reachable content cache namespace {name}"))?;
        anyhow::ensure!(
            same_content_cache_inode(&pinned, &reachable),
            "shared content cache namespace was detached or replaced: {}",
            dirs.display_root.join(name).display(),
        );
    }
    Ok(())
}

#[cfg(test)]
fn ensure_content_dirs(root: &Path) -> Result<()> {
    drop(open_content_cache_dirs(root)?);
    Ok(())
}

fn pinned_directory_entry_names(directory: &OwnedFd, description: &str) -> Result<Vec<OsString>> {
    let proc_path = PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd()));
    std::fs::read_dir(&proc_path)
        .with_context(|| format!("scan pinned {description}"))?
        .map(|entry| {
            entry
                .with_context(|| format!("read pinned {description} entry"))
                .map(|entry| entry.file_name())
        })
        .collect()
}

fn stat_entry_at(
    directory: &OwnedFd,
    name: &OsStr,
    subject: &str,
) -> Result<Option<rustix::fs::Stat>> {
    match rustix::fs::statat(directory, name, rustix::fs::AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => Ok(Some(stat)),
        Err(error) if error == rustix::io::Errno::NOENT => Ok(None),
        Err(error) => Err(error).with_context(|| format!("stat pinned {subject}")),
    }
}

fn remove_retired_content_entry_at(
    parent: &OwnedFd,
    name: &OsStr,
    display_path: &Path,
) -> Result<()> {
    let Some(before) = stat_entry_at(parent, name, "retired content generation")? else {
        return Ok(());
    };
    if !rustix::fs::FileType::from_raw_mode(before.st_mode).is_dir() {
        let Some(current) = stat_entry_at(parent, name, "retired content generation")? else {
            return Ok(());
        };
        anyhow::ensure!(
            same_content_cache_inode(&before, &current),
            "retired content entry changed before removal: {}",
            display_path.display(),
        );
        return match rustix::fs::unlinkat(parent, name, rustix::fs::AtFlags::empty()) {
            Ok(()) => Ok(()),
            Err(error) if error == rustix::io::Errno::NOENT => Ok(()),
            Err(error) => Err(error).with_context(|| {
                format!("remove retired content entry {}", display_path.display())
            }),
        };
    }

    let directory = match rustix::fs::openat(
        parent,
        name,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    ) {
        Ok(directory) => directory,
        Err(error)
            if error == rustix::io::Errno::NOENT
                || error == rustix::io::Errno::NOTDIR
                || error == rustix::io::Errno::LOOP =>
        {
            return Ok(());
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "open retired content directory without following links {}",
                    display_path.display()
                )
            });
        }
    };
    let opened = rustix::fs::fstat(&directory)
        .with_context(|| format!("stat retired content directory {}", display_path.display()))?;
    anyhow::ensure!(
        same_content_cache_inode(&before, &opened),
        "retired content directory changed while opening: {}",
        display_path.display(),
    );

    loop {
        let proc_path = PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd()));
        let mut found = false;
        for entry in std::fs::read_dir(&proc_path)
            .with_context(|| format!("scan retired content directory {}", display_path.display()))?
        {
            let entry = entry.with_context(|| {
                format!(
                    "read retired content entry beneath {}",
                    display_path.display()
                )
            })?;
            found = true;
            let child_name = entry.file_name();
            remove_retired_content_entry_at(
                &directory,
                &child_name,
                &display_path.join(&child_name),
            )?;
        }
        if !found {
            break;
        }
    }

    let Some(current) = stat_entry_at(parent, name, "emptied retired content directory")? else {
        return Ok(());
    };
    anyhow::ensure!(
        same_content_cache_inode(&opened, &current),
        "retired content directory changed before removal: {}",
        display_path.display(),
    );
    match rustix::fs::unlinkat(parent, name, rustix::fs::AtFlags::REMOVEDIR) {
        Ok(()) => Ok(()),
        Err(error) if error == rustix::io::Errno::NOENT => Ok(()),
        Err(error) => Err(error).with_context(|| {
            format!(
                "remove retired content directory {}",
                display_path.display()
            )
        }),
    }
}

fn retire_obsolete_content_v2(dirs: &ContentCacheDirs) -> Result<Vec<OsString>> {
    let mut retired = BTreeSet::new();
    for &legacy_name in LEGACY_CONTENT_V2_ENTRIES {
        if stat_entry_at(
            &dirs.root,
            OsStr::new(legacy_name),
            "obsolete content-v2 namespace",
        )?
        .is_none()
        {
            continue;
        }
        let sequence = CONTENT_TEMP_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
        let retired_name = OsString::from(format!(
            "{legacy_name}-{}-{sequence:016x}",
            std::process::id(),
        ));
        match rustix::fs::renameat(
            &dirs.root,
            OsStr::new(legacy_name),
            &dirs.retired_v2,
            &retired_name,
        ) {
            Ok(()) => {
                retired.insert(retired_name);
            }
            Err(error) if error == rustix::io::Errno::NOENT => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("retire obsolete content-v2 namespace {legacy_name}")
                });
            }
        }
    }
    for name in pinned_directory_entry_names(&dirs.retired_v2, "retired content generations")? {
        retired.insert(name);
    }
    Ok(retired.into_iter().collect())
}

fn cache_stat_is_old(stat: &rustix::fs::Stat, now: SystemTime, max_age: Duration) -> bool {
    if stat.st_mtime < 0 || !(0..1_000_000_000).contains(&stat.st_mtime_nsec) {
        return true;
    }
    SystemTime::UNIX_EPOCH
        .checked_add(Duration::new(
            stat.st_mtime as u64,
            stat.st_mtime_nsec as u32,
        ))
        .and_then(|modified| now.duration_since(modified).ok())
        .is_some_and(|age| age >= max_age)
}

fn remove_if_present_at(directory: &OwnedFd, name: &OsStr, subject: &str) -> Result<bool> {
    match rustix::fs::unlinkat(directory, name, rustix::fs::AtFlags::empty()) {
        Ok(()) => Ok(true),
        Err(error) if error == rustix::io::Errno::NOENT => Ok(false),
        Err(error) => Err(error).with_context(|| format!("remove pinned {subject}")),
    }
}

fn open_lock_file_at(directory: &OwnedFd, name: &OsStr, subject: &str) -> Result<File> {
    let lock = rustix::fs::openat(
        directory,
        name,
        rustix::fs::OFlags::RDWR
            | rustix::fs::OFlags::CREATE
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::from_raw_mode(0o666),
    )
    .with_context(|| format!("open pinned {subject}"))?;
    Ok(File::from(lock))
}

fn unlink_opened_file_at(
    directory: &OwnedFd,
    name: &OsStr,
    opened: &File,
    subject: &str,
) -> Result<bool> {
    let opened = rustix::fs::fstat(opened).with_context(|| format!("stat opened {subject}"))?;
    let Some(current) = stat_entry_at(directory, name, subject)? else {
        return Ok(false);
    };
    if opened.st_dev != current.st_dev || opened.st_ino != current.st_ino {
        return Ok(false);
    }
    remove_if_present_at(directory, name, subject)
}

fn create_content_temporary_at(
    directory: &OwnedFd,
    prefix: &str,
    subject: &str,
) -> Result<(File, OsString)> {
    loop {
        let sequence = CONTENT_TEMP_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
        let name = OsString::from(format!("{prefix}{}-{sequence:016x}", std::process::id()));
        match rustix::fs::openat(
            directory,
            &name,
            rustix::fs::OFlags::RDWR
                | rustix::fs::OFlags::CREATE
                | rustix::fs::OFlags::EXCL
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::from_raw_mode(0o600),
        ) {
            Ok(file) => return Ok((File::from(file), name)),
            Err(error) if error == rustix::io::Errno::EXIST => continue,
            Err(error) => return Err(error).with_context(|| format!("create pinned {subject}")),
        }
    }
}

struct BytesPublication<'a> {
    temporary_directory: &'a OwnedFd,
    final_directory: &'a OwnedFd,
    temporary_prefix: &'a str,
    final_name: &'a OsStr,
    bytes: &'a [u8],
    mode: u32,
    subject: &'a str,
}

fn publish_bytes_at<F>(publication: BytesPublication<'_>, post_rename: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    let BytesPublication {
        temporary_directory,
        final_directory,
        temporary_prefix,
        final_name,
        bytes,
        mode,
        subject,
    } = publication;
    let (mut temporary, temporary_name) =
        create_content_temporary_at(temporary_directory, temporary_prefix, subject)?;
    let mut published = false;
    let result: Result<()> = (|| {
        temporary
            .write_all(bytes)
            .with_context(|| format!("write {subject}"))?;
        temporary
            .flush()
            .with_context(|| format!("flush {subject}"))?;
        rustix::fs::fchmod(&temporary, rustix::fs::Mode::from_raw_mode(mode))
            .with_context(|| format!("seal {subject}"))?;
        rustix::fs::renameat(
            temporary_directory,
            &temporary_name,
            final_directory,
            final_name,
        )
        .with_context(|| format!("publish {subject}"))?;
        published = true;
        post_rename()?;
        Ok(())
    })();
    if result.is_err() {
        if published {
            let _ = unlink_opened_file_at(
                final_directory,
                final_name,
                &temporary,
                &format!("failed published {subject}"),
            );
        } else {
            let _ = unlink_opened_file_at(
                temporary_directory,
                &temporary_name,
                &temporary,
                &format!("failed temporary {subject}"),
            );
        }
    }
    result
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactContentReference {
    version: u32,
    owner: u64,
    identity: u64,
    record_path: Vec<u8>,
    objects: Vec<(u64, u64)>,
    integrity_ahash: u64,
}

fn artifact_reference_owner(record_path: &Path, identity: u64) -> u64 {
    let mut hasher = fixed_hasher();
    hash_len_prefixed(&mut hasher, b"ktstr-artifact-content-reference-owner");
    hash_len_prefixed(&mut hasher, record_path.as_os_str().as_bytes());
    hash_u64(&mut hasher, identity);
    hasher.finish()
}

fn artifact_reference_integrity(reference: &ArtifactContentReference) -> u64 {
    let mut hasher = fixed_hasher();
    hash_len_prefixed(&mut hasher, b"ktstr-artifact-content-reference");
    hash_u32(&mut hasher, reference.version);
    hash_u64(&mut hasher, reference.owner);
    hash_u64(&mut hasher, reference.identity);
    hash_len_prefixed(&mut hasher, &reference.record_path);
    hash_u64(&mut hasher, reference.objects.len() as u64);
    for (content_hash, len) in &reference.objects {
        hash_u64(&mut hasher, *content_hash);
        hash_u64(&mut hasher, *len);
    }
    hasher.finish()
}

fn validate_artifact_reference(reference: &ArtifactContentReference) -> Result<()> {
    anyhow::ensure!(
        reference.version == ARTIFACT_REFERENCE_SCHEMA,
        "unsupported artifact content-reference version {}",
        reference.version
    );
    let record_path = PathBuf::from(std::ffi::OsString::from_vec(reference.record_path.clone()));
    anyhow::ensure!(
        record_path.is_absolute(),
        "artifact content-reference record path is not absolute"
    );
    anyhow::ensure!(
        artifact_reference_owner(&record_path, reference.identity) == reference.owner,
        "artifact content-reference owner mismatch"
    );
    anyhow::ensure!(
        reference.integrity_ahash == artifact_reference_integrity(reference),
        "artifact content-reference integrity mismatch"
    );
    anyhow::ensure!(
        reference.objects.windows(2).all(|pair| pair[0] < pair[1]),
        "artifact content-reference objects are not sorted and unique"
    );
    Ok(())
}

fn absolute_cache_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(std::env::current_dir()
        .context("resolve current directory for artifact reference")?
        .join(path))
}

fn try_remove_coordinated_pair_at(
    namespace_locked: &File,
    lock_dir: &OwnedFd,
    lock_name: &OsStr,
    data_dir: &OwnedFd,
    data_name: &OsStr,
    subject: &str,
) -> Result<bool> {
    let lock = open_lock_file_at(lock_dir, lock_name, subject)?;
    match flock_retry(&lock, rustix::fs::FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => {
            let removed = remove_if_present_at(data_dir, data_name, subject)?;
            // The namespace gate is exclusively held, so no process can open
            // this pathname between unlinking the data and lock. A process
            // which already opened the lock either made the try-lock fail or
            // no longer needs the pathname.
            let _ = namespace_locked;
            unlink_opened_file_at(lock_dir, lock_name, &lock, subject)?;
            Ok(removed)
        }
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => Ok(false),
        Err(error) => Err(error).with_context(|| format!("try-lock {subject} for cleanup")),
    }
}

struct TemporaryEntryCleanup<'a> {
    namespace_locked: &'a File,
    lock_dir: &'a OwnedFd,
    lock_name: &'a OsStr,
    temporary_dir: &'a OwnedFd,
    temporary_name: &'a OsStr,
    data_dir: &'a OwnedFd,
    data_name: &'a OsStr,
    subject: &'a str,
}

fn try_remove_temporary_entry_at(cleanup: TemporaryEntryCleanup<'_>) -> Result<bool> {
    let TemporaryEntryCleanup {
        namespace_locked,
        lock_dir,
        lock_name,
        temporary_dir,
        temporary_name,
        data_dir,
        data_name,
        subject,
    } = cleanup;
    let lock = open_lock_file_at(lock_dir, lock_name, subject)?;
    match flock_retry(&lock, rustix::fs::FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => {
            let _ = namespace_locked;
            let removed = remove_if_present_at(temporary_dir, temporary_name, subject)?;
            if removed && stat_entry_at(data_dir, data_name, subject)?.is_none() {
                unlink_opened_file_at(lock_dir, lock_name, &lock, subject)?;
            }
            Ok(removed)
        }
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => Ok(false),
        Err(error) => Err(error).with_context(|| format!("try-lock {subject} for cleanup")),
    }
}

fn parse_cache_key(name: &str, prefix: &str, suffix: &str) -> Option<String> {
    let key = name.strip_prefix(prefix)?.strip_suffix(suffix)?;
    (key.len() == 16 && key.bytes().all(|byte| byte.is_ascii_hexdigit())).then(|| key.to_string())
}

fn parse_temporary_cache_key<'a>(name: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = name.strip_prefix(prefix)?;
    rest.get(..16)
        .filter(|key| key.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .filter(|_| rest.as_bytes().get(16) == Some(&b'-'))
}

fn digest_lock_name(key: &str) -> OsString {
    OsString::from(format!("{CONTENT_DIGEST_LOCK_PREFIX}{key}.lock"))
}

fn object_lock_name(key: &str) -> OsString {
    OsString::from(format!("{CONTENT_OBJECT_LOCK_PREFIX}{key}.lock"))
}

#[cfg(test)]
fn gc_content_cache_at(root: &Path, now: SystemTime, max_age: Duration) -> Result<()> {
    let dirs = open_content_cache_dirs(root)?;
    gc_content_cache_in(&dirs, now, max_age)
}

fn gc_content_cache_in(dirs: &ContentCacheDirs, now: SystemTime, max_age: Duration) -> Result<()> {
    validate_content_cache_dirs_reachable(dirs)?;
    let namespace_gate = open_lock_file_at(
        &dirs.root,
        OsStr::new(CONTENT_NAMESPACE_GATE),
        "content namespace gate",
    )?;
    match flock_retry(
        &namespace_gate,
        rustix::fs::FlockOperation::NonBlockingLockExclusive,
    ) {
        Ok(()) => {}
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => return Ok(()),
        Err(error) => return Err(error).context("lock shared content namespace for cleanup"),
    }
    validate_content_cache_dirs_reachable(dirs)?;
    let retired_content_v2 = retire_obsolete_content_v2(dirs)?;

    for name in pinned_directory_entry_names(&dirs.digests, "content digest cache")? {
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if let Some(key) = parse_temporary_cache_key(name_str, ".tmp-digest-") {
            let Some(stat) = stat_entry_at(&dirs.digests, &name, "temporary digest memo")? else {
                continue;
            };
            if rustix::fs::FileType::from_raw_mode(stat.st_mode).is_file()
                && cache_stat_is_old(&stat, now, CONTENT_ORPHAN_TEMP_GRACE)
            {
                let lock_name = digest_lock_name(key);
                try_remove_temporary_entry_at(TemporaryEntryCleanup {
                    namespace_locked: &namespace_gate,
                    lock_dir: &dirs.root,
                    lock_name: &lock_name,
                    temporary_dir: &dirs.digests,
                    temporary_name: &name,
                    data_dir: &dirs.digests,
                    data_name: &OsString::from(format!("{key}.digest")),
                    subject: "temporary digest memo",
                })?;
            }
            continue;
        }
        let Some(key) = name
            .to_str()
            .and_then(|name| parse_cache_key(name, "", ".digest"))
        else {
            continue;
        };
        let Some(stat) = stat_entry_at(&dirs.digests, &name, "content digest memo")? else {
            continue;
        };
        if rustix::fs::FileType::from_raw_mode(stat.st_mode).is_file()
            && cache_stat_is_old(&stat, now, max_age)
        {
            let lock_name = digest_lock_name(&key);
            try_remove_coordinated_pair_at(
                &namespace_gate,
                &dirs.root,
                &lock_name,
                &dirs.digests,
                &name,
                "content digest memo",
            )?;
        }
    }

    // Transient files live in their own same-filesystem namespace. Periodic
    // cleanup is therefore proportional to active/crashed publishers rather
    // than to the cardinality of the immutable multi-terabyte object CAS.
    for name in pinned_directory_entry_names(&dirs.staging, "content staging cache")? {
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let Some(stat) = stat_entry_at(&dirs.staging, &name, "content staging file")? else {
            continue;
        };
        if !rustix::fs::FileType::from_raw_mode(stat.st_mode).is_file()
            || !cache_stat_is_old(&stat, now, CONTENT_ORPHAN_TEMP_GRACE)
        {
            continue;
        }
        if name_str.starts_with(".generated-artifact-") {
            remove_if_present_at(
                &dirs.staging,
                &name,
                "orphan generated artifact staging file",
            )?;
            continue;
        }
        if name_str.starts_with(".content-gc-stamp-") {
            remove_if_present_at(&dirs.staging, &name, "temporary content GC stamp")?;
            continue;
        }
        if let Some(key) = parse_temporary_cache_key(name_str, ".content-object-") {
            let lock_name = object_lock_name(key);
            try_remove_temporary_entry_at(TemporaryEntryCleanup {
                namespace_locked: &namespace_gate,
                lock_dir: &dirs.root,
                lock_name: &lock_name,
                temporary_dir: &dirs.staging,
                temporary_name: &name,
                data_dir: &dirs.objects,
                data_name: &OsString::from(format!("{key}.object")),
                subject: "temporary content object",
            })?;
        }
    }

    // Immutable objects and their durable per-key locks are deliberately not
    // scanned here. Closure-aware artifact GC owns object reclamation; keyed
    // crash temporaries above remove their own orphan lock when no canonical
    // data exists. This keeps periodic metadata GC independent of CAS size.

    validate_content_cache_dirs_reachable(dirs)?;
    publish_bytes_at(
        BytesPublication {
            temporary_directory: &dirs.staging,
            final_directory: &dirs.root,
            temporary_prefix: ".content-gc-stamp-",
            final_name: OsStr::new(CONTENT_GC_STAMP),
            bytes: &[],
            mode: 0o644,
            subject: "content GC stamp",
        },
        || validate_content_cache_dirs_reachable(dirs),
    )?;
    validate_content_cache_dirs_reachable(dirs)?;
    // The v3 namespace is live before any retired tree is removed. Release its
    // global gate first so a one-time multi-terabyte v2 unlink walk cannot
    // serialize current publishers and consumers behind obsolete storage.
    drop(namespace_gate);
    for retired_name in retired_content_v2 {
        let display_path = dirs
            .display_root
            .join(CONTENT_RETIRED_V2_DIR)
            .join(&retired_name);
        if let Err(error) =
            remove_retired_content_entry_at(&dirs.retired_v2, &retired_name, &display_path)
        {
            tracing::warn!(
                path = %display_path.display(),
                error = %error,
                "deferred incomplete retired content-v2 cleanup",
            );
        }
    }
    Ok(())
}

/// Collect immutable content objects which are outside every retained
/// artifact-tree closure.
///
/// The artifact layer computes whole-closure reachability while holding its
/// global lifecycle gate. This layer then takes the content namespace gate in
/// EX mode and each object's own EX lock before unlinking the canonical object
/// and lock pathname. An in-flight publisher or non-artifact object lease
/// therefore makes the object a clean skip instead of a partial collection.
pub(super) fn collect_unreachable_content_objects(
    retained: &BTreeSet<(u64, u64)>,
    now: SystemTime,
    blocking: bool,
) -> Result<bool> {
    let root = content_cache_root()?;
    collect_unreachable_content_objects_at_root(&root, retained, now, blocking)
}

fn collect_unreachable_content_objects_at_root(
    root: &Path,
    retained: &BTreeSet<(u64, u64)>,
    now: SystemTime,
    blocking: bool,
) -> Result<bool> {
    let dirs = open_content_cache_dirs(root)?;
    collect_unreachable_content_objects_in(&dirs, retained, now, blocking)
}

fn collect_unreachable_content_objects_in(
    dirs: &ContentCacheDirs,
    retained: &BTreeSet<(u64, u64)>,
    now: SystemTime,
    blocking: bool,
) -> Result<bool> {
    validate_content_cache_dirs_reachable(dirs)?;
    let namespace_gate = open_lock_file_at(
        &dirs.root,
        OsStr::new(CONTENT_NAMESPACE_GATE),
        "content namespace gate",
    )?;
    let operation = if blocking {
        rustix::fs::FlockOperation::LockExclusive
    } else {
        rustix::fs::FlockOperation::NonBlockingLockExclusive
    };
    match flock_retry(&namespace_gate, operation) {
        Ok(()) => {}
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => return Ok(false),
        Err(error) => return Err(error).context("lock content objects for closure collection"),
    }
    validate_content_cache_dirs_reachable(dirs)?;

    let mut globally_retained = retained.clone();
    for name in pinned_directory_entry_names(&dirs.references, "artifact content references")? {
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with(".artifact-reference-") {
            let Some(stat) = stat_entry_at(
                &dirs.references,
                OsStr::new(name),
                "temporary artifact content reference",
            )?
            else {
                continue;
            };
            if rustix::fs::FileType::from_raw_mode(stat.st_mode).is_file()
                && cache_stat_is_old(&stat, now, CONTENT_ORPHAN_TEMP_GRACE)
            {
                remove_if_present_at(
                    &dirs.references,
                    OsStr::new(name),
                    "temporary artifact content reference",
                )?;
            }
            continue;
        }
        let Some(owner) = parse_cache_key(name, "", ".json") else {
            continue;
        };
        let display_path = dirs.display_root.join(ARTIFACT_REFERENCE_DIR).join(name);
        let reference = match read_artifact_reference_at(&dirs.references, OsStr::new(name)) {
            Ok(reference) => reference,
            Err(error) => {
                // The edge itself is reconstructible cache state. Remove it
                // under namespace EX, but abort this sweep so objects are not
                // collected in the same pass. A later record hit republishes
                // the exact edge; otherwise a later pass reclaims its orphaned
                // objects and that record becomes a clean rebuild miss.
                tracing::warn!(
                    path = %display_path.display(),
                    error = %error,
                    "removing invalid artifact reference and deferring content sweep",
                );
                remove_if_present_at(
                    &dirs.references,
                    OsStr::new(name),
                    "invalid artifact content reference",
                )?;
                return Ok(false);
            }
        };
        if format!("{:016x}", reference.owner) != owner {
            tracing::warn!(
                path = %display_path.display(),
                "removing mismatched artifact reference and deferring content sweep",
            );
            remove_if_present_at(
                &dirs.references,
                OsStr::new(name),
                "mismatched artifact content reference",
            )?;
            return Ok(false);
        }
        let record_path =
            PathBuf::from(std::ffi::OsString::from_vec(reference.record_path.clone()));
        match open_cache_record(
            &record_path,
            "artifact record referenced by global CAS edge",
        ) {
            Ok(Some((_record, _))) => {
                globally_retained.extend(reference.objects);
            }
            Ok(None) => {
                // Publication orders ref before record while namespace SH is
                // held; under this EX gate, a missing record is therefore a
                // crash leftover or an eviction whose record was hidden first.
                remove_if_present_at(
                    &dirs.references,
                    OsStr::new(name),
                    "orphan artifact content reference",
                )?;
            }
            Err(error) => {
                tracing::warn!(
                    path = %record_path.display(),
                    error = %error,
                    "skipping content sweep because a referenced artifact record cannot be checked",
                );
                return Ok(false);
            }
        }
    }

    let retained_hashes = globally_retained
        .iter()
        .map(|(content_hash, _)| *content_hash)
        .collect::<BTreeSet<_>>();
    for name in pinned_directory_entry_names(&dirs.objects, "content objects")? {
        let Some(name) = name.to_str() else {
            continue;
        };
        if let Some(key) = parse_cache_key(name, "", ".object") {
            let content_hash = u64::from_str_radix(&key, 16)
                .with_context(|| format!("parse content object key {key}"))?;
            if retained_hashes.contains(&content_hash) {
                continue;
            }
            let lock_name = object_lock_name(&key);
            try_remove_coordinated_pair_at(
                &namespace_gate,
                &dirs.root,
                &lock_name,
                &dirs.objects,
                OsStr::new(name),
                "unreachable content object",
            )?;
        }
    }
    validate_content_cache_dirs_reachable(dirs)?;
    Ok(true)
}

fn read_artifact_reference_at(
    reference_dir: &OwnedFd,
    name: &OsStr,
) -> Result<ArtifactContentReference> {
    let opened = rustix::fs::openat(
        reference_dir,
        name,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::NONBLOCK
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .context("open pinned artifact content reference")?;
    let stat = rustix::fs::fstat(&opened).context("stat pinned artifact content reference")?;
    anyhow::ensure!(
        rustix::fs::FileType::from_raw_mode(stat.st_mode).is_file()
            && stat.st_size >= 0
            && stat.st_size as u64 <= ARTIFACT_REFERENCE_MAX_BYTES,
        "artifact content reference has invalid size/type",
    );
    let opened = File::from(opened);
    let mut bytes = Vec::with_capacity(stat.st_size as usize);
    opened
        .take(ARTIFACT_REFERENCE_MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .context("read pinned artifact content reference")?;
    anyhow::ensure!(
        bytes.len() as u64 <= ARTIFACT_REFERENCE_MAX_BYTES,
        "artifact content reference grew past the maximum size while reading",
    );
    let reference: ArtifactContentReference =
        serde_json::from_slice(&bytes).context("parse pinned artifact content reference")?;
    validate_artifact_reference(&reference)?;
    Ok(reference)
}

fn maybe_gc_content_cache_in(dirs: &ContentCacheDirs) -> Result<()> {
    let now = SystemTime::now();
    if stat_entry_at(&dirs.root, OsStr::new(CONTENT_GC_STAMP), "content GC stamp")?.is_some_and(
        |stat| {
            rustix::fs::FileType::from_raw_mode(stat.st_mode).is_file()
                && !cache_stat_is_old(&stat, now, CONTENT_GC_INTERVAL)
        },
    ) {
        return Ok(());
    }
    gc_content_cache_in(dirs, now, CONTENT_MAX_AGE)
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

    /// Retain a lock acquired on a replacement open file description.
    ///
    /// The namespace gate remains held while the caller enters the blocking
    /// kernel queue, so adopting the granted descriptor preserves the same
    /// open-before-unlink invariant as locking `self.file` directly.
    pub(crate) fn adopt_locked_file(&mut self, file: std::os::fd::OwnedFd) {
        self.file = File::from(file);
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

fn open_content_coord_file(dirs: &ContentCacheDirs, lock_name: &OsStr) -> Result<CoordinationFile> {
    validate_content_cache_dirs_reachable(dirs)?;
    let namespace_gate = open_lock_file_at(
        &dirs.root,
        OsStr::new(CONTENT_NAMESPACE_GATE),
        "content namespace gate",
    )?;
    flock_retry(&namespace_gate, rustix::fs::FlockOperation::LockShared)
        .context("lock root-pinned content coordination namespace")?;
    validate_content_cache_dirs_reachable(dirs)?;
    let file = open_lock_file_at(&dirs.root, lock_name, "content coordination lock")?;
    validate_content_cache_dirs_reachable(dirs)?;
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
    load: L,
    build: B,
    wait_shared: WS,
    wait_exclusive: WX,
) -> Result<T>
where
    L: FnMut() -> Result<Option<T>>,
    B: FnOnce() -> Result<T>,
    WS: FnMut(&mut CoordinationFile) -> Result<()>,
    WX: FnMut(&mut CoordinationFile) -> Result<()>,
{
    load_or_build_with_wait_open(subject, load, build, wait_shared, wait_exclusive, || {
        open_coord_file(namespace_gate_path, lock_path)
    })
}

fn load_or_build_with_wait_open<T, L, B, WS, WX, O>(
    subject: &str,
    mut load: L,
    build: B,
    mut wait_shared: WS,
    mut wait_exclusive: WX,
    mut open_coordination: O,
) -> Result<T>
where
    L: FnMut() -> Result<Option<T>>,
    B: FnOnce() -> Result<T>,
    WS: FnMut(&mut CoordinationFile) -> Result<()>,
    WX: FnMut(&mut CoordinationFile) -> Result<()>,
    O: FnMut() -> Result<CoordinationFile>,
{
    if let Some(value) = load()? {
        return Ok(value);
    }

    let mut build = Some(build);
    let mut wait_for_successor = false;
    loop {
        let mut coordination = open_coordination()?;
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

fn load_or_build_content<T, L, B>(
    dirs: &ContentCacheDirs,
    lock_name: &OsStr,
    subject: &str,
    mut load: L,
    build: B,
) -> Result<T>
where
    L: FnMut() -> Result<Option<T>>,
    B: FnOnce() -> Result<T>,
{
    load_or_build_with_wait_open(
        subject,
        || {
            validate_content_cache_dirs_reachable(dirs)?;
            let value = load()?;
            validate_content_cache_dirs_reachable(dirs)?;
            Ok(value)
        },
        || {
            validate_content_cache_dirs_reachable(dirs)?;
            let value = build()?;
            validate_content_cache_dirs_reachable(dirs)?;
            Ok(value)
        },
        |coordination| coordination.lock_shared().map_err(anyhow::Error::from),
        |coordination| coordination.lock_exclusive().map_err(anyhow::Error::from),
        || open_content_coord_file(dirs, lock_name),
    )
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

fn read_fixed_cache_record_at<const N: usize>(
    directory: &OwnedFd,
    name: &OsStr,
    display_path: &Path,
    subject: &str,
) -> Result<Option<[u8; N]>> {
    let Some((file, identity)) = open_cache_record_at(directory, name, display_path, subject)?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        identity.size == N as u64,
        "{subject} has invalid length {}: {}",
        identity.size,
        display_path.display(),
    );
    let mut bytes = [0u8; N];
    pread_exact(&file, 0, &mut bytes, subject)?;
    anyhow::ensure!(
        StableFileIdentity::from_file(&file)? == identity,
        "{subject} changed while reading: {}",
        display_path.display(),
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

fn read_file_digest_record_at(
    dirs: &ContentCacheDirs,
    name: &OsStr,
    identity: StableFileIdentity,
) -> Result<Option<u64>> {
    let display_path = dirs.display_root.join(FILE_DIGEST_MEMO_DIR).join(name);
    let Some(bytes) = read_fixed_cache_record_at::<FILE_DIGEST_RECORD_LEN>(
        &dirs.digests,
        name,
        &display_path,
        "file digest memo",
    )?
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
    #[cfg(test)]
    record_test_content_hash_read(identity);
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
    #[cfg(test)]
    record_test_content_hash_read(identity);
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

#[cfg(test)]
fn test_content_hash_reads()
-> &'static std::sync::Mutex<std::collections::BTreeMap<(u64, u64), usize>> {
    static READS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::BTreeMap<(u64, u64), usize>>,
    > = std::sync::OnceLock::new();
    READS.get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()))
}

#[cfg(test)]
fn record_test_content_hash_read(identity: StableFileIdentity) {
    *test_content_hash_reads()
        .lock()
        .expect("content hash read counter poisoned")
        .entry((identity.dev, identity.ino))
        .or_default() += 1;
}

#[cfg(test)]
pub(crate) fn reset_test_content_hash_read_count(identity: StableFileIdentity) {
    test_content_hash_reads()
        .lock()
        .expect("content hash read counter poisoned")
        .remove(&(identity.dev, identity.ino));
}

#[cfg(test)]
pub(crate) fn test_content_hash_read_count(identity: StableFileIdentity) -> usize {
    test_content_hash_reads()
        .lock()
        .expect("content hash read counter poisoned")
        .get(&(identity.dev, identity.ino))
        .copied()
        .unwrap_or_default()
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
    let dirs = open_content_cache_dirs(root)?;
    maybe_gc_content_cache_in(&dirs)?;

    cached_file_digest_in(&dirs, file, identity)
}

/// Digest one pinned revision while reusing an already-opened content-cache
/// namespace. Large source snapshots contain thousands of files; reopening
/// the root and every descriptor-relative subdirectory for each file turns a
/// metadata-only cache hit into thousands of avoidable namespace walks.
fn cached_file_digest_in(
    dirs: &ContentCacheDirs,
    file: &File,
    identity: StableFileIdentity,
) -> Result<u64> {
    let identity_key = file_digest_identity_key(identity);
    let record_name = OsString::from(format!("{identity_key:016x}.digest"));
    let record_path = dirs
        .display_root
        .join(FILE_DIGEST_MEMO_DIR)
        .join(&record_name);
    let lock_name = digest_lock_name(&format!("{identity_key:016x}"));
    let before_lookup = source_identity_before_cache_lookup(file, identity, "file digest")?;
    let digest = load_or_build_content(
        dirs,
        &lock_name,
        &format!("file digest {}", record_path.display()),
        || read_file_digest_record_at(dirs, &record_name, identity),
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

            // This is reconstructible cache state. The checksum and fixed
            // envelope make a torn post-crash record fail closed, while
            // descriptor-relative rename keeps live readers from observing a
            // partial write or a substituted namespace.
            publish_bytes_at(
                BytesPublication {
                    temporary_directory: &dirs.digests,
                    final_directory: &dirs.digests,
                    temporary_prefix: &format!(".tmp-digest-{identity_key:016x}-"),
                    final_name: &record_name,
                    bytes: &bytes,
                    mode: 0o600,
                    subject: "file digest memo",
                },
                || validate_content_cache_dirs_reachable(dirs),
            )?;
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

/// Create and pin generated artifact bytes on the content-CAS filesystem.
///
/// Artifact-tree publication requires FICLONE without a byte-copy fallback.
/// Staging generated metadata in the process-global temporary directory would
/// therefore fail with EXDEV whenever `/tmp` and `KTSTR_CACHE_DIR` are on
/// different mounts. The bytes originate in memory, so write them once into an
/// unlinked staging inode beside the CAS objects; publication can then clone
/// that inode through the same strict path as every other artifact input.
pub(crate) fn open_pinned_generated_artifact(
    bytes: &[u8],
    mode: u32,
) -> Result<(File, StableFileIdentity)> {
    let root = content_cache_root()?;
    let dirs = open_content_cache_dirs(&root)?;
    validate_content_cache_dirs_reachable(&dirs)?;
    let namespace_gate = open_lock_file_at(
        &dirs.root,
        OsStr::new(CONTENT_NAMESPACE_GATE),
        "content namespace gate for generated staging",
    )?;
    flock_retry(&namespace_gate, rustix::fs::FlockOperation::LockShared)
        .context("lease content namespace for generated artifact staging")?;
    validate_content_cache_dirs_reachable(&dirs)?;
    let (mut temporary, temporary_name) = create_content_temporary_at(
        &dirs.staging,
        ".generated-artifact-",
        "generated artifact staging file",
    )?;
    let result: Result<StableFileIdentity> = (|| {
        temporary
            .write_all(bytes)
            .context("write generated artifact staging file")?;
        temporary
            .flush()
            .context("flush generated artifact staging file")?;
        rustix::fs::fchmod(&temporary, rustix::fs::Mode::from_raw_mode(mode))
            .context("set generated artifact staging mode")?;
        validate_content_cache_dirs_reachable(&dirs)?;
        anyhow::ensure!(
            unlink_opened_file_at(
                &dirs.staging,
                &temporary_name,
                &temporary,
                "generated artifact staging file",
            )?,
            "generated artifact staging pathname changed before unlink",
        );
        validate_content_cache_dirs_reachable(&dirs)?;
        StableFileIdentity::from_file(&temporary).context("stat generated artifact staging file")
    })();
    if result.is_err() {
        let _ = unlink_opened_file_at(
            &dirs.staging,
            &temporary_name,
            &temporary,
            "generated artifact staging file",
        );
    }
    drop(namespace_gate);
    Ok((temporary, result?))
}

fn content_cache_root() -> Result<PathBuf> {
    super::resolve_cache_root_with_suffix("content").context("resolve machine-wide content cache")
}

#[cfg(test)]
fn content_object_path(root: &Path, content_hash: u64) -> PathBuf {
    root.join(CONTENT_OBJECT_DIR)
        .join(format!("{content_hash:016x}.object"))
}

/// One shared lease over the complete content-object namespace.
///
/// Artifact trees can reference thousands of immutable objects. Holding a
/// per-object flock and object descriptor for the complete tree scales file
/// descriptor use with the tree size. The collector already takes the
/// namespace gate exclusively before unlinking any object or coordination
/// pathname, so one shared gate is sufficient to keep every pathname alive
/// while a producer publishes its record or a consumer opens bounded batches
/// for FICLONE materialization.
pub(super) struct ContentNamespaceLease {
    _gate: File,
    dirs: ContentCacheDirs,
}

impl ContentNamespaceLease {
    /// Digest one exact source inode through this already-pinned namespace.
    /// The caller still receives the same per-inode cross-process memo and
    /// before/after revision validation as [`cached_file_digest`].
    pub(super) fn digest_file(&self, source: &File, identity: StableFileIdentity) -> Result<u64> {
        cached_file_digest_in(&self.dirs, source, identity)
    }

    /// Publish one exact source inode into the content CAS while retaining
    /// this tree-wide namespace lease. A separate per-object lease would be
    /// redundant: the namespace gate already prevents object and election
    /// pathnames from being reclaimed until the complete tree record is
    /// published.
    pub(super) fn digest_and_publish_file(
        &self,
        source: &File,
        identity: StableFileIdentity,
    ) -> Result<(u64, u64)> {
        let content_hash = self.digest_file(source, identity)?;
        drop(open_or_publish_content_object_in(
            &self.dirs,
            content_hash,
            source,
            identity,
        )?);
        Ok((content_hash, identity.size))
    }

    /// Open and validate one immutable object while the namespace remains
    /// protected from garbage collection. The returned descriptor pins the
    /// exact inode even if another valid publisher replaces its pathname.
    pub(super) fn open_object(
        &self,
        content_hash: u64,
        expected_len: u64,
    ) -> Result<Option<(File, PathBuf)>> {
        validate_content_cache_dirs_reachable(&self.dirs)?;
        let name = OsString::from(format!("{content_hash:016x}.object"));
        let path = self.dirs.display_root.join(CONTENT_OBJECT_DIR).join(&name);
        let object = open_content_object_in(&self.dirs.objects, &name, &path, expected_len)?;
        validate_content_cache_dirs_reachable(&self.dirs)?;
        Ok(object.map(|file| (file, path)))
    }

    /// Publish the global reachability edge for one artifact-tree record.
    ///
    /// Callers hold this namespace SH lease from object publication through
    /// this reference and then local-record publication. A content-EX sweep
    /// can therefore observe either no objects, or the complete global edge,
    /// but never sweep the pre-record publication window.
    pub(super) fn publish_artifact_reference(
        &self,
        record_path: &Path,
        identity: u64,
        objects: &BTreeSet<(u64, u64)>,
    ) -> Result<()> {
        validate_content_cache_dirs_reachable(&self.dirs)?;
        let record_path = absolute_cache_path(record_path)?;
        let owner = artifact_reference_owner(&record_path, identity);
        let mut reference = ArtifactContentReference {
            version: ARTIFACT_REFERENCE_SCHEMA,
            owner,
            identity,
            record_path: record_path.as_os_str().as_bytes().to_vec(),
            objects: objects.iter().copied().collect(),
            integrity_ahash: 0,
        };
        reference.integrity_ahash = artifact_reference_integrity(&reference);
        validate_artifact_reference(&reference)?;

        let bytes =
            serde_json::to_vec(&reference).context("serialize artifact content reference")?;
        let final_name = OsString::from(format!("{owner:016x}.json"));
        publish_bytes_at(
            BytesPublication {
                temporary_directory: &self.dirs.references,
                final_directory: &self.dirs.references,
                temporary_prefix: &format!(".artifact-reference-{owner:016x}-"),
                final_name: &final_name,
                bytes: &bytes,
                mode: 0o444,
                subject: "artifact content reference",
            },
            || validate_content_cache_dirs_reachable(&self.dirs),
        )?;
        validate_content_cache_dirs_reachable(&self.dirs)
    }
}

/// Prevent content GC from unlinking any object pathname until this owner is
/// dropped. Unlike per-object leases, this consumes one descriptor regardless
/// of artifact-tree cardinality.
pub(super) fn lease_content_namespace() -> Result<ContentNamespaceLease> {
    let root = content_cache_root()?;
    lease_content_namespace_at_root(&root)
}

fn lease_content_namespace_at_root(root: &Path) -> Result<ContentNamespaceLease> {
    let dirs = open_content_cache_dirs(root)?;
    maybe_gc_content_cache_in(&dirs)?;
    let gate = open_lock_file_at(
        &dirs.root,
        OsStr::new(CONTENT_NAMESPACE_GATE),
        "content namespace gate",
    )?;
    flock_retry(&gate, rustix::fs::FlockOperation::LockShared)
        .with_context(|| format!("lease content namespace {}", root.display()))?;
    validate_content_cache_dirs_reachable(&dirs)?;
    Ok(ContentNamespaceLease { _gate: gate, dirs })
}

fn open_cache_record_at(
    directory: &OwnedFd,
    name: &OsStr,
    display_path: &Path,
    subject: &str,
) -> Result<Option<(File, StableFileIdentity)>> {
    let opened = match rustix::fs::openat(
        directory,
        name,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::NONBLOCK
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    ) {
        Ok(opened) => opened,
        Err(error) if error == rustix::io::Errno::NOENT => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("open {subject} {}", display_path.display()));
        }
    };
    let file = File::from(opened);
    let identity = StableFileIdentity::from_file(&file)
        .with_context(|| format!("stat {subject} {}", display_path.display()))?;
    Ok(Some((file, identity)))
}

fn open_content_object_in(
    directory: &OwnedFd,
    name: &OsStr,
    display_path: &Path,
    expected_len: u64,
) -> Result<Option<File>> {
    let Some((file, identity)) =
        open_cache_record_at(directory, name, display_path, "content object")?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        identity.size == expected_len,
        "content object {} has length {}, expected {expected_len}",
        display_path.display(),
        identity.size
    );
    let mode = file
        .metadata()
        .with_context(|| format!("stat content object mode {}", display_path.display()))?
        .permissions()
        .mode();
    anyhow::ensure!(
        mode & 0o222 == 0 && mode & 0o111 != 0,
        "content object must be read-only executable: {}",
        display_path.display()
    );
    Ok(Some(file))
}

/// Open one immutable machine-wide content object and validate its byte length.
#[cfg(test)]
pub(crate) fn open_content_object(content_hash: u64, expected_len: u64) -> Result<Option<File>> {
    let root = content_cache_root()?;
    let dirs = open_content_cache_dirs(&root)?;
    maybe_gc_content_cache_in(&dirs)?;
    validate_content_cache_dirs_reachable(&dirs)?;
    let name = OsString::from(format!("{content_hash:016x}.object"));
    let path = dirs.display_root.join(CONTENT_OBJECT_DIR).join(&name);
    let object = open_content_object_in(&dirs.objects, &name, &path, expected_len)?;
    validate_content_cache_dirs_reachable(&dirs)?;
    Ok(object)
}

/// Publish a pinned input as one immutable content object.
///
/// Every publication uses FICLONE so all consumers share one backing extent
/// and retain private/COW semantics across processes. There is deliberately no
/// byte-copy fallback. Publication is descriptor-relative and atomic, and the
/// pinned source revision is checked on both sides of the clone.
fn publish_content_object(
    dirs: &ContentCacheDirs,
    expected_hash: u64,
    source: &File,
    source_identity: StableFileIdentity,
) -> Result<File> {
    publish_content_object_with_post_rename(dirs, expected_hash, source, source_identity, || Ok(()))
}

fn publish_content_object_with_post_rename<F>(
    dirs: &ContentCacheDirs,
    expected_hash: u64,
    source: &File,
    source_identity: StableFileIdentity,
    post_rename: F,
) -> Result<File>
where
    F: FnOnce() -> Result<()>,
{
    validate_content_cache_dirs_reachable(dirs)?;
    let before_copy = StableFileIdentity::from_file(source)?;
    anyhow::ensure!(
        before_copy.same_open_content_version(source_identity),
        "content source changed before CAS publication"
    );

    let (temporary, temporary_name) = create_content_temporary_at(
        &dirs.staging,
        &format!(".content-object-{expected_hash:016x}-"),
        "content object temporary",
    )?;
    let final_name = OsString::from(format!("{expected_hash:016x}.object"));
    let result: Result<File> = (|| {
        crate::reflink::ficlone(&temporary, source).context("FICLONE pinned content object")?;
        let after_copy = StableFileIdentity::from_file(source)?;
        anyhow::ensure!(
            after_copy.same_open_content_version(source_identity),
            "content source changed during CAS publication"
        );
        if before_copy != source_identity
            || after_copy != source_identity
            || before_copy != after_copy
        {
            let temporary_identity = StableFileIdentity::from_file(&temporary)
                .context("stat content object temp for source-race validation")?;
            let actual_hash = hash_pinned_file_exact_identity(&temporary, temporary_identity)
                .context("hash content object temp after source revision transition")?;
            anyhow::ensure!(
                actual_hash == expected_hash,
                "content object captured bytes from a transient source revision"
            );
        }
        rustix::fs::fchmod(&temporary, rustix::fs::Mode::from_raw_mode(0o555))
            .context("make content object read-only executable")?;
        validate_content_cache_dirs_reachable(dirs)?;
        rustix::fs::renameat(&dirs.staging, &temporary_name, &dirs.objects, &final_name)
            .context("publish content object beneath pinned namespace")?;
        post_rename()?;
        let published = open_content_object_in(
            &dirs.objects,
            &final_name,
            &dirs.display_root.join(CONTENT_OBJECT_DIR).join(&final_name),
            source_identity.size,
        )?
        .context("reopen published content object read-only")?;
        let temporary_stat = rustix::fs::fstat(&temporary)
            .context("stat writable content object construction descriptor")?;
        let published_stat =
            rustix::fs::fstat(&published).context("stat reopened content object descriptor")?;
        anyhow::ensure!(
            same_content_cache_inode(&temporary_stat, &published_stat)
                && temporary_stat.st_size == published_stat.st_size
                && published_stat.st_size >= 0
                && published_stat.st_size as u64 == source_identity.size
                && temporary_stat.st_mode == published_stat.st_mode
                && published_stat.st_mode & 0o7777 == 0o555,
            "reopened content object does not exactly match the published inode",
        );
        validate_content_cache_dirs_reachable(dirs)?;
        Ok(published)
    })();
    match result {
        Ok(published) => {
            drop(temporary);
            Ok(published)
        }
        Err(error) => {
            let _ = unlink_opened_file_at(
                &dirs.staging,
                &temporary_name,
                &temporary,
                "content object temporary",
            );
            let _ = unlink_opened_file_at(
                &dirs.objects,
                &final_name,
                &temporary,
                "failed published content object",
            );
            drop(temporary);
            Err(error)
        }
    }
}

/// Open or publish one immutable object under the machine-wide content key.
///
/// The content namespace owns this election rather than any caller-specific
/// analysis/preparation cache, so identical bytes converge on one inode even
/// when different components and checkouts request them concurrently.
fn open_or_publish_content_object_in(
    dirs: &ContentCacheDirs,
    content_hash: u64,
    source: &File,
    source_identity: StableFileIdentity,
) -> Result<File> {
    let object_name = OsString::from(format!("{content_hash:016x}.object"));
    let object_path = dirs
        .display_root
        .join(CONTENT_OBJECT_DIR)
        .join(&object_name);
    let lock_name = object_lock_name(&format!("{content_hash:016x}"));
    let before_lookup =
        source_identity_before_cache_lookup(source, source_identity, "content object")?;
    let object = load_or_build_content(
        dirs,
        &lock_name,
        &format!("content object {content_hash:016x}"),
        || {
            open_content_object_in(
                &dirs.objects,
                &object_name,
                &object_path,
                source_identity.size,
            )
        },
        || publish_content_object(dirs, content_hash, source, source_identity),
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

fn open_or_publish_content_object_at_root(
    root: &Path,
    content_hash: u64,
    source: &File,
    source_identity: StableFileIdentity,
) -> Result<File> {
    let dirs = open_content_cache_dirs(root)?;
    maybe_gc_content_cache_in(&dirs)?;
    open_or_publish_content_object_in(&dirs, content_hash, source, source_identity)
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
    _dirs: ContentCacheDirs,
    path: PathBuf,
}

impl ContentObjectLease {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

fn lease_content_object_in(
    dirs: &ContentCacheDirs,
    content_hash: u64,
    expected_len: u64,
) -> Result<Option<(File, CoordinationFile, PathBuf)>> {
    validate_content_cache_dirs_reachable(dirs)?;
    let key = format!("{content_hash:016x}");
    let object_name = OsString::from(format!("{key}.object"));
    let object_path = dirs
        .display_root
        .join(CONTENT_OBJECT_DIR)
        .join(&object_name);
    let lock_name = object_lock_name(&key);
    let mut coordination = open_content_coord_file(dirs, &lock_name)?;
    coordination
        .lock_shared()
        .with_context(|| format!("lease content object {content_hash:016x}"))?;
    coordination.release_namespace_gate();
    let Some(file) =
        open_content_object_in(&dirs.objects, &object_name, &object_path, expected_len)?
    else {
        return Ok(None);
    };
    validate_content_cache_dirs_reachable(dirs)?;
    Ok(Some((file, coordination, object_path)))
}

#[cfg(test)]
fn lease_content_object_at_root(
    root: &Path,
    content_hash: u64,
    expected_len: u64,
) -> Result<Option<ContentObjectLease>> {
    let dirs = open_content_cache_dirs(root)?;
    maybe_gc_content_cache_in(&dirs)?;
    let Some((file, coordination, path)) =
        lease_content_object_in(&dirs, content_hash, expected_len)?
    else {
        return Ok(None);
    };
    Ok(Some(ContentObjectLease {
        _file: file,
        _coordination: coordination,
        _dirs: dirs,
        path,
    }))
}

pub(crate) fn open_or_publish_content_object_lease(
    content_hash: u64,
    source: &File,
    source_identity: StableFileIdentity,
) -> Result<ContentObjectLease> {
    let root = content_cache_root()?;
    let dirs = open_content_cache_dirs(&root)?;
    maybe_gc_content_cache_in(&dirs)?;
    loop {
        drop(open_or_publish_content_object_in(
            &dirs,
            content_hash,
            source,
            source_identity,
        )?);
        if let Some((file, coordination, path)) =
            lease_content_object_in(&dirs, content_hash, source_identity.size)?
        {
            return Ok(ContentObjectLease {
                _file: file,
                _coordination: coordination,
                _dirs: dirs,
                path,
            });
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
    fn namespace_session_batches_digest_and_object_publication_without_reopening_dirs() {
        let root = tempfile::TempDir::new().unwrap();
        let sources = tempfile::TempDir::new().unwrap();
        let first_path = sources.path().join("first");
        let second_path = sources.path().join("second");
        std::fs::write(&first_path, b"first immutable source revision").unwrap();
        std::fs::write(&second_path, b"second immutable source revision").unwrap();
        let (first, first_identity) = open_pinned_file(&first_path).unwrap();
        let (second, second_identity) = open_pinned_file(&second_path).unwrap();
        reset_test_content_hash_read_count(first_identity);
        reset_test_content_hash_read_count(second_identity);
        reset_test_content_cache_dir_open_count(root.path());

        let lease = lease_content_namespace_at_root(root.path()).unwrap();
        let (first_hash, first_len) = lease
            .digest_and_publish_file(&first, first_identity)
            .unwrap();
        assert_eq!(
            lease.digest_file(&first, first_identity).unwrap(),
            first_hash
        );
        let (second_hash, second_len) = lease
            .digest_and_publish_file(&second, second_identity)
            .unwrap();

        assert_eq!(first_len, first_identity.size);
        assert_eq!(second_len, second_identity.size);
        assert_ne!(first_hash, second_hash);
        assert_eq!(test_content_hash_read_count(first_identity), 1);
        assert_eq!(test_content_hash_read_count(second_identity), 1);
        assert_eq!(
            test_content_cache_dir_open_count(root.path()),
            1,
            "the namespace lease must pin one directory set for the complete batch",
        );

        let (_, first_object) = lease
            .open_object(first_hash, first_len)
            .unwrap()
            .expect("published first object");
        let (_, second_object) = lease
            .open_object(second_hash, second_len)
            .unwrap()
            .expect("published second object");
        assert_eq!(
            std::fs::read(first_object).unwrap(),
            b"first immutable source revision"
        );
        assert_eq!(
            std::fs::read(second_object).unwrap(),
            b"second immutable source revision"
        );
        assert_eq!(test_content_cache_dir_open_count(root.path()), 1);
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
    fn published_content_object_returns_read_only_descriptor() {
        use std::io::Write as _;

        let root = tempfile::TempDir::new().unwrap();
        let source_dir = tempfile::TempDir::new().unwrap();
        let source_path = source_dir.path().join("source");
        std::fs::write(&source_path, b"strict reflink content").unwrap();
        let (source, identity) = open_pinned_file(&source_path).unwrap();
        let content_hash = hash_pinned_file(&source, identity).unwrap();

        let published =
            open_or_publish_content_object_at_root(root.path(), content_hash, &source, identity)
                .unwrap();
        let access = rustix::fs::fcntl_getfl(&published).unwrap() & rustix::fs::OFlags::ACCMODE;
        assert_eq!(access, rustix::fs::OFlags::RDONLY);
        let mut attempted_writer = published.try_clone().unwrap();
        let error = attempted_writer.write_all(b"x").unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::EBADF));
    }

    #[test]
    fn strict_content_namespace_retires_and_never_reuses_legacy_v2_object() {
        let root = tempfile::TempDir::new().unwrap();
        let legacy_dir = root.path().join("objects-v2");
        std::fs::create_dir(&legacy_dir).unwrap();
        let source_dir = tempfile::TempDir::new().unwrap();
        let source_path = source_dir.path().join("source");
        std::fs::write(&source_path, b"strict-object").unwrap();
        let (source, identity) = open_pinned_file(&source_path).unwrap();
        let content_hash = hash_pinned_file(&source, identity).unwrap();
        let legacy_path = legacy_dir.join(format!("{content_hash:016x}.object"));
        std::fs::write(&legacy_path, b"legacy-object").unwrap();
        std::fs::set_permissions(&legacy_path, std::fs::Permissions::from_mode(0o555)).unwrap();
        let legacy_locks = root.path().join(".locks-v1");
        std::fs::create_dir(&legacy_locks).unwrap();
        std::fs::write(legacy_locks.join("obsolete.lock"), b"").unwrap();
        std::fs::write(root.path().join(".gc-v2"), b"").unwrap();

        let published =
            open_or_publish_content_object_at_root(root.path(), content_hash, &source, identity)
                .unwrap();
        let current_path = content_object_path(root.path(), content_hash);
        assert_eq!(std::fs::read(&current_path).unwrap(), b"strict-object");
        assert!(!legacy_path.exists());
        assert!(!legacy_dir.exists());
        assert!(!legacy_locks.exists());
        assert!(!root.path().join(".gc-v2").exists());
        assert_eq!(
            std::fs::read_dir(root.path().join(CONTENT_RETIRED_V2_DIR))
                .unwrap()
                .count(),
            0,
        );
        assert_eq!(
            published.metadata().unwrap().ino(),
            std::fs::metadata(&current_path).unwrap().ino(),
        );
        assert!(current_path.starts_with(root.path().join(CONTENT_OBJECT_DIR)));
    }

    #[test]
    fn leased_content_path_survives_ancestor_symlink_retarget() {
        use std::os::unix::fs::symlink;

        let base = tempfile::TempDir::new().unwrap();
        let first = base.path().join("first");
        let second = base.path().join("second");
        std::fs::create_dir(&first).unwrap();
        std::fs::create_dir(&second).unwrap();
        let alias = base.path().join("alias");
        symlink(&first, &alias).unwrap();
        let requested_root = alias.join("content");
        let source_path = first.join("source");
        std::fs::write(&source_path, b"canonical pinned path").unwrap();
        let (source, identity) = open_pinned_file(&source_path).unwrap();
        let content_hash = hash_pinned_file(&source, identity).unwrap();
        drop(
            open_or_publish_content_object_at_root(
                &requested_root,
                content_hash,
                &source,
                identity,
            )
            .unwrap(),
        );
        let lease = lease_content_object_at_root(&requested_root, content_hash, identity.size)
            .unwrap()
            .unwrap();
        let expected_path = first
            .join("content")
            .join(CONTENT_OBJECT_DIR)
            .join(format!("{content_hash:016x}.object"));
        assert_eq!(lease.path(), expected_path);

        std::fs::remove_file(&alias).unwrap();
        symlink(&second, &alias).unwrap();

        assert_eq!(lease.path(), expected_path);
        assert_eq!(
            std::fs::read(lease.path()).unwrap(),
            b"canonical pinned path"
        );
    }

    #[test]
    fn failed_post_rename_validation_removes_exact_published_inode() {
        let root = tempfile::TempDir::new().unwrap();
        let source_dir = tempfile::TempDir::new().unwrap();
        let source_path = source_dir.path().join("source");
        std::fs::write(&source_path, b"post rename cleanup").unwrap();
        let (source, identity) = open_pinned_file(&source_path).unwrap();
        let content_hash = hash_pinned_file(&source, identity).unwrap();
        let dirs = open_content_cache_dirs(root.path()).unwrap();
        let object_namespace = root.path().join(CONTENT_OBJECT_DIR);
        let detached_namespace = root.path().join("detached-objects");

        let error =
            publish_content_object_with_post_rename(&dirs, content_hash, &source, identity, || {
                std::fs::rename(&object_namespace, &detached_namespace)?;
                std::fs::create_dir(&object_namespace)?;
                Ok(())
            })
            .unwrap_err();

        assert!(format!("{error:#}").contains("detached or replaced"));
        assert!(
            !detached_namespace
                .join(format!("{content_hash:016x}.object"))
                .exists()
        );
        assert_eq!(std::fs::read_dir(&object_namespace).unwrap().count(), 0);
    }

    #[test]
    fn failed_small_record_post_rename_validation_removes_exact_published_inode() {
        let root = tempfile::TempDir::new().unwrap();
        let dirs = open_content_cache_dirs(root.path()).unwrap();
        let reference_namespace = root.path().join(ARTIFACT_REFERENCE_DIR);
        let detached_namespace = root.path().join("detached-references");
        let final_name = OsStr::new("1111222233334444.json");

        let error = publish_bytes_at(
            BytesPublication {
                temporary_directory: &dirs.references,
                final_directory: &dirs.references,
                temporary_prefix: ".artifact-reference-test-",
                final_name,
                bytes: b"published bytes",
                mode: 0o444,
                subject: "test artifact reference",
            },
            || {
                std::fs::rename(&reference_namespace, &detached_namespace)?;
                std::fs::create_dir(&reference_namespace)?;
                validate_content_cache_dirs_reachable(&dirs)
            },
        )
        .unwrap_err();

        assert!(format!("{error:#}").contains("detached or replaced"));
        assert!(!detached_namespace.join(final_name).exists());
        assert_eq!(std::fs::read_dir(&reference_namespace).unwrap().count(), 0);
    }

    #[test]
    fn gc_collects_crash_orphaned_digest_object_and_generated_temporaries_safely() {
        let root = tempfile::TempDir::new().unwrap();
        let dirs = open_content_cache_dirs(root.path()).unwrap();
        let key = "1111222233334444";
        let digest_temporary = root
            .path()
            .join(FILE_DIGEST_MEMO_DIR)
            .join(format!(".tmp-digest-{key}-crash"));
        let generated_temporary = root
            .path()
            .join(CONTENT_STAGING_DIR)
            .join(".generated-artifact-crash");
        let immutable_namespace_decoy = root
            .path()
            .join(CONTENT_OBJECT_DIR)
            .join(".generated-artifact-not-staging");
        let object_key = "5555666677778888";
        let object_temporary = root
            .path()
            .join(CONTENT_STAGING_DIR)
            .join(format!(".content-object-{object_key}-crash"));
        std::fs::write(&digest_temporary, b"partial digest").unwrap();
        std::fs::write(&generated_temporary, b"partial generated artifact").unwrap();
        std::fs::write(&immutable_namespace_decoy, b"immutable namespace decoy").unwrap();
        std::fs::write(&object_temporary, b"partial object").unwrap();
        let digest_lock = open_lock_file_at(
            &dirs.root,
            &digest_lock_name(key),
            "test live digest publisher",
        )
        .unwrap();
        flock_retry(&digest_lock, rustix::fs::FlockOperation::LockExclusive).unwrap();
        let object_lock = open_lock_file_at(
            &dirs.root,
            &object_lock_name(object_key),
            "test live object publisher",
        )
        .unwrap();
        flock_retry(&object_lock, rustix::fs::FlockOperation::LockExclusive).unwrap();
        let future = SystemTime::now() + CONTENT_ORPHAN_TEMP_GRACE + Duration::from_secs(1);

        gc_content_cache_at(root.path(), future, CONTENT_MAX_AGE).unwrap();
        assert!(digest_temporary.exists());
        assert!(root.path().join(digest_lock_name(key)).exists());
        assert!(!generated_temporary.exists());
        assert!(object_temporary.exists());
        assert!(root.path().join(object_lock_name(object_key)).exists());
        assert!(
            immutable_namespace_decoy.exists(),
            "periodic temp GC must not scan the immutable object namespace",
        );

        drop(digest_lock);
        drop(object_lock);
        gc_content_cache_at(root.path(), future, CONTENT_MAX_AGE).unwrap();
        assert!(!digest_temporary.exists());
        assert!(!root.path().join(digest_lock_name(key)).exists());
        assert!(!object_temporary.exists());
        assert!(!root.path().join(object_lock_name(object_key)).exists());

        std::fs::write(&generated_temporary, b"live generated artifact").unwrap();
        let generated_future =
            SystemTime::now() + CONTENT_ORPHAN_TEMP_GRACE + Duration::from_secs(1);
        let namespace_gate = open_lock_file_at(
            &dirs.root,
            OsStr::new(CONTENT_NAMESPACE_GATE),
            "test live generated publisher",
        )
        .unwrap();
        flock_retry(&namespace_gate, rustix::fs::FlockOperation::LockShared).unwrap();
        gc_content_cache_at(root.path(), generated_future, CONTENT_MAX_AGE).unwrap();
        assert!(generated_temporary.exists());
        drop(namespace_gate);
        gc_content_cache_at(root.path(), generated_future, CONTENT_MAX_AGE).unwrap();
        assert!(!generated_temporary.exists());
    }

    #[test]
    fn gc_reclaims_metadata_but_never_partially_collects_content_closures() {
        let root = tempfile::TempDir::new().unwrap();
        ensure_content_dirs(root.path()).unwrap();
        let digest_dir = root.path().join(FILE_DIGEST_MEMO_DIR);
        let object_dir = root.path().join(CONTENT_OBJECT_DIR);
        let lock_dir = root.path();

        let digest_key = "1111111111111111";
        let object_key = "2222222222222222";
        let live_key = "3333333333333333";
        let orphan_key = "4444444444444444";
        std::fs::write(digest_dir.join(format!("{digest_key}.digest")), b"memo").unwrap();
        std::fs::write(lock_dir.join(digest_lock_name(digest_key)), b"").unwrap();
        std::fs::write(
            object_dir.join(format!("{object_key}.object")),
            vec![0u8; 4096],
        )
        .unwrap();
        std::fs::write(lock_dir.join(object_lock_name(object_key)), b"").unwrap();
        std::fs::write(
            object_dir.join(format!("{live_key}.object")),
            vec![0u8; 4096],
        )
        .unwrap();
        let live_lock_path = lock_dir.join(object_lock_name(live_key));
        let live_lock = open_lock_file(&live_lock_path).unwrap();
        flock_retry(&live_lock, rustix::fs::FlockOperation::LockExclusive).unwrap();
        let orphan_lock_path = lock_dir.join(object_lock_name(orphan_key));
        std::fs::write(&orphan_lock_path, b"").unwrap();
        let orphan_temporary = root
            .path()
            .join(CONTENT_STAGING_DIR)
            .join(format!(".content-object-{orphan_key}-crash"));
        std::fs::write(&orphan_temporary, b"partial object").unwrap();

        let future = SystemTime::now() + CONTENT_MAX_AGE + Duration::from_secs(1);
        gc_content_cache_at(root.path(), future, CONTENT_MAX_AGE).unwrap();
        assert!(!digest_dir.join(format!("{digest_key}.digest")).exists());
        assert!(!lock_dir.join(digest_lock_name(digest_key)).exists());
        assert!(
            object_dir.join(format!("{object_key}.object")).exists(),
            "GC must retain an unleased object until closure-level collection exists"
        );
        assert!(lock_dir.join(object_lock_name(object_key)).exists());
        assert!(
            object_dir.join(format!("{live_key}.object")).exists(),
            "GC must retain every member of a possible artifact closure"
        );
        assert!(live_lock_path.exists());
        assert!(
            !orphan_lock_path.exists(),
            "a crashed keyed object temporary and its orphan lock are reconstructible"
        );
        assert!(!orphan_temporary.exists());

        drop(live_lock);
        gc_content_cache_at(root.path(), future, CONTENT_MAX_AGE).unwrap();
        assert!(object_dir.join(format!("{live_key}.object")).exists());
        assert!(live_lock_path.exists());
        assert!(
            lock_dir.join(CONTENT_NAMESPACE_GATE).exists(),
            "the namespace gate itself is never a GC candidate"
        );
    }

    #[test]
    fn closure_gc_marks_reachable_objects_and_skips_live_object_leases() {
        let root = tempfile::TempDir::new().unwrap();
        ensure_content_dirs(root.path()).unwrap();
        let retained_hash = 0x1111_2222_3333_4444;
        let live_hash = 0x5555_6666_7777_8888;
        let retained_path = content_object_path(root.path(), retained_hash);
        let live_path = content_object_path(root.path(), live_hash);
        std::fs::write(&retained_path, b"retained").unwrap();
        std::fs::write(&live_path, b"live").unwrap();
        for path in [&retained_path, &live_path] {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o555)).unwrap();
        }
        let live = lease_content_object_at_root(root.path(), live_hash, 4)
            .unwrap()
            .expect("lease live content object");
        let retained = BTreeSet::from([(retained_hash, 8)]);

        collect_unreachable_content_objects_at_root(
            root.path(),
            &retained,
            SystemTime::now(),
            false,
        )
        .unwrap();
        assert!(
            retained_path.exists(),
            "marked closure object was collected"
        );
        assert!(
            live_path.exists(),
            "an active non-artifact object lease must make collection skip"
        );

        drop(live);
        collect_unreachable_content_objects_at_root(
            root.path(),
            &retained,
            SystemTime::now(),
            false,
        )
        .unwrap();
        assert!(retained_path.exists());
        assert!(
            !live_path.exists(),
            "unmarked inactive object was not collected"
        );
    }

    #[test]
    fn closure_gc_ignores_legacy_unkeyed_content_temporaries() {
        let root = tempfile::TempDir::new().unwrap();
        ensure_content_dirs(root.path()).unwrap();
        let temporary = root
            .path()
            .join(CONTENT_OBJECT_DIR)
            .join(".content-object-legacy-publisher");
        std::fs::write(&temporary, b"possibly-live legacy publisher").unwrap();

        collect_unreachable_content_objects_at_root(
            root.path(),
            &BTreeSet::new(),
            SystemTime::now() + CONTENT_ORPHAN_TEMP_GRACE + Duration::from_secs(1),
            false,
        )
        .unwrap();

        assert!(
            temporary.exists(),
            "uncoordinated legacy temporaries cannot be proven dead by the current protocol",
        );
    }

    #[test]
    fn content_cache_rejects_root_and_fixed_namespace_symlinks_without_touching_sentinel() {
        use std::os::unix::fs::symlink;

        for namespace in [
            None,
            Some(FILE_DIGEST_MEMO_DIR),
            Some(CONTENT_OBJECT_DIR),
            Some(CONTENT_STAGING_DIR),
            Some(CONTENT_RETIRED_V2_DIR),
            Some(ARTIFACT_REFERENCE_DIR),
        ] {
            let base = tempfile::TempDir::new().unwrap();
            let external = tempfile::TempDir::new().unwrap();
            let sentinel = external.path().join("sentinel");
            std::fs::write(&sentinel, b"external sentinel").unwrap();
            let root = base.path().join("content");

            match namespace {
                None => symlink(external.path(), &root).unwrap(),
                Some(namespace) => {
                    ensure_content_dirs(&root).unwrap();
                    let path = root.join(namespace);
                    std::fs::remove_dir(&path).unwrap();
                    symlink(external.path(), path).unwrap();
                }
            }

            let error = collect_unreachable_content_objects_at_root(
                &root,
                &BTreeSet::new(),
                SystemTime::now(),
                false,
            )
            .unwrap_err();
            assert!(
                format!("{error:#}").contains("without following links"),
                "unexpected fixed-namespace rejection: {error:#}",
            );
            assert_eq!(std::fs::read(&sentinel).unwrap(), b"external sentinel");
            assert_eq!(
                std::fs::read_dir(external.path()).unwrap().count(),
                1,
                "cache initialization escaped through {namespace:?}",
            );
        }
    }

    #[test]
    fn closure_gc_unlinks_reference_symlink_without_touching_external_sentinel() {
        use std::os::unix::fs::symlink;

        let root = tempfile::TempDir::new().unwrap();
        ensure_content_dirs(root.path()).unwrap();
        let external = tempfile::TempDir::new().unwrap();
        let sentinel = external.path().join("sentinel.json");
        std::fs::write(&sentinel, b"external sentinel").unwrap();
        let reference = root
            .path()
            .join(ARTIFACT_REFERENCE_DIR)
            .join("1111222233334444.json");
        symlink(&sentinel, &reference).unwrap();

        assert!(
            !collect_unreachable_content_objects_at_root(
                root.path(),
                &BTreeSet::new(),
                SystemTime::now(),
                false,
            )
            .unwrap(),
            "an invalid reference must defer its content sweep",
        );
        assert!(!reference.exists());
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"external sentinel");
    }

    #[test]
    fn content_gc_stamp_replaces_symlink_without_touching_external_sentinel() {
        use std::os::unix::fs::symlink;

        let root = tempfile::TempDir::new().unwrap();
        ensure_content_dirs(root.path()).unwrap();
        let external = tempfile::TempDir::new().unwrap();
        let sentinel = external.path().join("sentinel");
        std::fs::write(&sentinel, b"external sentinel").unwrap();
        let stamp = root.path().join(CONTENT_GC_STAMP);
        symlink(&sentinel, &stamp).unwrap();

        gc_content_cache_at(root.path(), SystemTime::now(), CONTENT_MAX_AGE).unwrap();

        assert!(std::fs::symlink_metadata(&stamp).unwrap().is_file());
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"external sentinel");
    }

    #[test]
    fn content_gc_fails_closed_after_pinned_root_path_swap() {
        use std::os::unix::fs::symlink;

        let base = tempfile::TempDir::new().unwrap();
        let root = base.path().join("content");
        let dirs = open_content_cache_dirs(&root).unwrap();
        let pinned_root = base.path().join("content-pinned");
        std::fs::rename(&root, &pinned_root).unwrap();
        let external = tempfile::TempDir::new().unwrap();
        let sentinel = external.path().join("sentinel");
        std::fs::write(&sentinel, b"external sentinel").unwrap();
        symlink(external.path(), &root).unwrap();

        let error = gc_content_cache_in(&dirs, SystemTime::now(), CONTENT_MAX_AGE).unwrap_err();

        assert!(format!("{error:#}").contains("detached or replaced"));
        assert!(!pinned_root.join(CONTENT_GC_STAMP).exists());
        assert!(!external.path().join(CONTENT_GC_STAMP).exists());
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"external sentinel");
    }

    #[test]
    fn reference_publication_fails_closed_after_namespace_path_swap() {
        use std::os::unix::fs::symlink;

        let root = tempfile::TempDir::new().unwrap();
        let lease = lease_content_namespace_at_root(root.path()).unwrap();
        let reference_path = root.path().join(ARTIFACT_REFERENCE_DIR);
        let pinned_reference_path = root.path().join("references-pinned");
        std::fs::rename(&reference_path, &pinned_reference_path).unwrap();
        let external = tempfile::TempDir::new().unwrap();
        let sentinel = external.path().join("sentinel");
        std::fs::write(&sentinel, b"external sentinel").unwrap();
        symlink(external.path(), &reference_path).unwrap();

        let record_path = root.path().join("record.json");
        let identity = 0x1111_2222_3333_4444;
        let objects = BTreeSet::from([(0x5555_6666_7777_8888, 4096)]);
        let error = lease
            .publish_artifact_reference(&record_path, identity, &objects)
            .unwrap_err();

        let owner = artifact_reference_owner(&record_path, identity);
        assert!(format!("{error:#}").contains("detached or replaced"));
        assert!(
            !pinned_reference_path
                .join(format!("{owner:016x}.json"))
                .is_file(),
            "reference was published into a detached directory",
        );
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"external sentinel");
        assert_eq!(std::fs::read_dir(external.path()).unwrap().count(), 1);
    }

    #[test]
    fn closure_gc_fails_closed_after_object_namespace_path_swap() {
        use std::os::unix::fs::symlink;

        let root = tempfile::TempDir::new().unwrap();
        ensure_content_dirs(root.path()).unwrap();
        let content_hash = 0x1111_2222_3333_4444;
        let object_path = content_object_path(root.path(), content_hash);
        std::fs::write(&object_path, b"unreachable").unwrap();
        let dirs = open_content_cache_dirs(root.path()).unwrap();
        let object_namespace = root.path().join(CONTENT_OBJECT_DIR);
        let pinned_object_namespace = root.path().join("objects-pinned");
        std::fs::rename(&object_namespace, &pinned_object_namespace).unwrap();
        let external = tempfile::TempDir::new().unwrap();
        let sentinel = external.path().join("sentinel");
        std::fs::write(&sentinel, b"external sentinel").unwrap();
        symlink(external.path(), &object_namespace).unwrap();

        let error = collect_unreachable_content_objects_in(
            &dirs,
            &BTreeSet::new(),
            SystemTime::now(),
            false,
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("detached or replaced"));
        assert!(
            pinned_object_namespace
                .join(format!("{content_hash:016x}.object"))
                .exists()
        );
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"external sentinel");
        assert_eq!(std::fs::read_dir(external.path()).unwrap().count(), 1);
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
    fn executable_leases_converge_and_content_survives_lease_drop() {
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
        gc_content_cache_at(root.path(), future, CONTENT_MAX_AGE).unwrap();
        assert!(
            leases[0].path().exists(),
            "shared lease must prevent GC from unlinking a live manifest path"
        );

        let object_path = leases[0].path().to_path_buf();
        drop(leases);
        gc_content_cache_at(root.path(), future, CONTENT_MAX_AGE).unwrap();
        assert!(
            object_path.exists(),
            "per-object GC must not tear down a multi-object closure after leases drop"
        );
    }
}
