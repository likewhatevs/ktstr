//! Reusable immutable artifact trees backed by the shared content CAS.
//!
//! A producer builds in its own private directory, pins the exact regular
//! files it wants to publish, and records a relocatable tree made of content
//! objects, directories, and relative symlinks. Consumers materialize private
//! trees with reflink/COW clones. Cargo's mutable target state is never shared.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::hash::{BuildHasher as _, Hasher as _};
use std::io::{Read as _, Write as _};
use std::mem::MaybeUninit;
use std::os::unix::ffi::{OsStrExt as _, OsStringExt as _};
use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{
    AtomicBool as ProcessAtomicBool, AtomicU64 as ProcessAtomicU64, Ordering as AtomicOrdering,
};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context as _, Result};

const RECORD_SCHEMA: u32 = 1;
const RECORD_DIR: &str = "records-v1";
// Artifact elections must never alias the content CAS namespace gate. In
// particular, KTSTR_CACHE_DIR intentionally places every cache kind at one
// common root; sharing `.locks-v1/namespace.lock` made a producer's forced
// content sweep wait for EX on the gate held SH by its own election waiters.
// This protocol intentionally has no old-generation compatibility rail.
const LOCK_DIR: &str = ".artifact-tree-locks-v2";
const NAMESPACE_GATE: &str = "namespace.lock";
const RECORD_MAX_BYTES: u64 = 64 << 20;
const MATERIALIZATION_PREFIX: &str = "ktstr-artifacts-";
const MATERIALIZATION_LIVE_LOCK: &str = ".ktstr-live.lock";
const MATERIALIZATION_GC_GRACE: Duration = Duration::from_secs(60);
const MATERIALIZATION_GC_INTERVAL: Duration = Duration::from_secs(30);
const MATERIALIZATION_GC_SCAN_LIMIT: usize = 256;
const MATERIALIZATION_GC_LOCK: &str = ".ktstr-materialization-gc.lock";
const MATERIALIZATION_GC_STAMP: &str = ".ktstr-materialization-gc.stamp";
const MATERIALIZATION_GC_CURSOR: &str = ".ktstr-materialization-gc.cursor";
const ARTIFACT_IO_WORKERS_MAX: usize = 16;
const STABLE_TREE_MARKER: &str = ".ktstr-artifact-tree-v2";
const STABLE_BUILD_MARKER: &str = ".ktstr-stable-build-v2";
const STABLE_VALIDATION_LOCK_DIR: &str = ".stable-root-validation-v1";
const STABLE_ROOT_MARKER_MAGIC: &[u8; 8] = b"KTSTRSR2";
const STABLE_ROOT_MARKER_SCHEMA: u32 = 2;
const STABLE_ROOT_MARKER_LEN: usize = 56;
const STABLE_ROOT_LISTING_BUFFER_LEN: usize = 64 << 10;
const LIFECYCLE_DIR: &str = ".artifact-tree-lifecycle-v1";
const LIFECYCLE_GATE: &str = "namespace.lock";
const LIFECYCLE_CLOSURE_LOCK_DIR: &str = "closures";
const LIFECYCLE_ACCESS_DIR: &str = "access";
const LIFECYCLE_RESERVATION_DIR: &str = "reservations";
const LIFECYCLE_GC_STAMP: &str = "gc.stamp";
const BUILD_SPACE_HIGH_WATER: &str = ".build-space-high-water-v1";
const BUILD_SPACE_HIGH_WATER_LOCK: &str = ".build-space-high-water-v1.lock";
const BUILD_SPACE_HIGH_WATER_MAGIC: u64 = 0x4b54_5354_5242_5731;
const LIFECYCLE_ACCESS_INTERVAL: Duration = Duration::from_secs(60);
const LIFECYCLE_ORPHAN_GRACE: Duration = Duration::from_secs(60);
const LIFECYCLE_MIN_FREE_RESERVE: u64 = 32 << 30;
const LIFECYCLE_CACHE_CAPACITY_DIVISOR: u64 = 3;
const LIFECYCLE_FREE_RESERVE_DIVISOR: u64 = 5;
const LIFECYCLE_BUILD_RESERVATION_HEADROOM_DIVISOR: u64 = 4;
const LIFECYCLE_BUILD_RESERVATION_MAX: u64 = 64 << 30;
const LIFECYCLE_BUILD_RESERVATION_MIN: u64 = 8 << 30;
static OPENAT2_UNAVAILABLE: ProcessAtomicBool = ProcessAtomicBool::new(false);
static STABLE_REBASE_SEQUENCE: ProcessAtomicU64 = ProcessAtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum StableRootDomain {
    Source = 1,
    CargoBuild = 2,
}

impl StableRootDomain {
    fn marker_name(self) -> &'static str {
        match self {
            Self::Source => STABLE_TREE_MARKER,
            Self::CargoBuild => STABLE_BUILD_MARKER,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Source),
            2 => Some(Self::CargoBuild),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StableRootSeal {
    domain: StableRootDomain,
    identity: u64,
    root_dev: u64,
    root_ino: u64,
    entry_count: u64,
    listing_hash: u64,
}

#[derive(Clone, Debug)]
struct StableRootListingEntry {
    path: Vec<u8>,
    kind: u8,
    ino: u64,
    symlink_target: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct StableRootListing {
    entry_count: u64,
    hash: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct StableRootTestCounters {
    record_reads: usize,
    cas_object_opens: usize,
    listing_directories: usize,
    listing_entries: usize,
    listing_fallback_stats: usize,
    lifecycle_collections: usize,
}

#[cfg(test)]
thread_local! {
    static STABLE_ROOT_TEST_COUNTERS: std::cell::Cell<StableRootTestCounters> =
        const { std::cell::Cell::new(StableRootTestCounters {
            record_reads: 0,
            cas_object_opens: 0,
            listing_directories: 0,
            listing_entries: 0,
            listing_fallback_stats: 0,
            lifecycle_collections: 0,
        }) };
    static STABLE_ROOT_PRE_PUBLISH_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        std::cell::RefCell::new(None);
}

#[cfg(test)]
fn reset_stable_root_test_counters() {
    STABLE_ROOT_TEST_COUNTERS.set(StableRootTestCounters::default());
}

#[cfg(test)]
fn stable_root_test_counters() -> StableRootTestCounters {
    STABLE_ROOT_TEST_COUNTERS.get()
}

#[cfg(test)]
fn update_stable_root_test_counters(update: impl FnOnce(&mut StableRootTestCounters)) {
    STABLE_ROOT_TEST_COUNTERS.with(|slot| {
        let mut counters = slot.get();
        update(&mut counters);
        slot.set(counters);
    });
}

#[cfg(test)]
fn set_stable_root_pre_publish_hook(hook: impl FnOnce() + 'static) {
    STABLE_ROOT_PRE_PUBLISH_HOOK.with(|slot| {
        assert!(slot.borrow_mut().replace(Box::new(hook)).is_none());
    });
}

#[cfg(test)]
fn run_stable_root_pre_publish_hook() {
    STABLE_ROOT_PRE_PUBLISH_HOOK.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
}

enum SourceEntry {
    Directory { mode: u32 },
    File { file: SourceFile, mode: u32 },
    Symlink { target: PathBuf },
}

enum SourceFile {
    /// Exact source inode retained until the next bounded publication wave.
    Pinned(super::PinnedContentFile),
    /// Immutable CAS object protected by the tree's one namespace lease.
    Published { content_hash: u64, len: u64 },
    /// Temporary state while a bounded worker owns the pinned descriptor. A
    /// publication error aborts the entire source construction, so this can
    /// never reach a successfully published tree record.
    Publishing,
}

enum PendingTreeEntry {
    Directory { relative: PathBuf, mode: u32 },
    File { relative: PathBuf, source: PathBuf },
    Symlink { relative: PathBuf, target: PathBuf },
}

/// One relocatable artifact tree ready for immutable publication.
///
/// Regular files are pinned as they are inserted, then moved through bounded
/// parallel waves into immutable CAS objects. Replacing a Cargo output
/// pathname after insertion cannot retarget the bytes eventually published.
#[doc(hidden)]
pub struct ArtifactTreeSource {
    entries: BTreeMap<PathBuf, SourceEntry>,
    // At most one bounded worker wave of source descriptors is retained.
    // Published objects are protected collectively by this single shared
    // namespace gate instead of one object lease per file.
    content_namespace: Option<super::content::ContentNamespaceLease>,
    pending_pinned_paths: Vec<PathBuf>,
    // Number of descendants already inserted below each normalized path.
    // This makes rejecting a later file/symlink parent logarithmic instead of
    // rescanning the complete tree after every insertion.
    descendant_counts: BTreeMap<PathBuf, usize>,
    #[cfg(test)]
    insertion_validation_visits: usize,
    #[cfg(test)]
    peak_pinned_files: usize,
}

impl ArtifactTreeSource {
    /// Start an empty tree.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            content_namespace: None,
            pending_pinned_paths: Vec::new(),
            descendant_counts: BTreeMap::new(),
            #[cfg(test)]
            insertion_validation_visits: 0,
            #[cfg(test)]
            peak_pinned_files: 0,
        }
    }

    /// Number of explicit entries in this tree.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether this tree contains no explicit entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add one directory with its final permission bits.
    pub fn insert_directory(&mut self, relative: impl AsRef<Path>, mode: u32) -> Result<()> {
        let relative = checked_relative_path(relative.as_ref())?;
        checked_mode(mode)?;
        self.insert_entry(relative, SourceEntry::Directory { mode })
    }

    /// Open, pin, and add one regular file.
    pub fn insert_file(
        &mut self,
        relative: impl AsRef<Path>,
        source: impl AsRef<Path>,
    ) -> Result<()> {
        let pinned = super::pin_content_file(source.as_ref())?;
        self.insert_pinned_file(relative, pinned)
    }

    /// Add an already-opened exact regular-file revision.
    pub fn insert_pinned_file(
        &mut self,
        relative: impl AsRef<Path>,
        pinned: super::PinnedContentFile,
    ) -> Result<()> {
        self.insert_pinned_file_with_policy(relative, pinned, false)
    }

    /// Add an already-opened regular-file revision and strip every write bit
    /// from its materialized mode.
    #[doc(hidden)]
    pub fn insert_immutable_pinned_file(
        &mut self,
        relative: impl AsRef<Path>,
        pinned: super::PinnedContentFile,
    ) -> Result<()> {
        self.insert_pinned_file_with_policy(relative, pinned, true)
    }

    fn insert_pinned_file_with_policy(
        &mut self,
        relative: impl AsRef<Path>,
        pinned: super::PinnedContentFile,
        immutable: bool,
    ) -> Result<()> {
        let relative = checked_relative_path(relative.as_ref())?;
        let metadata = pinned
            .source()
            .metadata()
            .with_context(|| format!("stat pinned artifact {}", pinned.source_path().display()))?;
        anyhow::ensure!(
            metadata.is_file(),
            "artifact tree input is not a regular file: {}",
            pinned.source_path().display()
        );
        let mut mode = metadata.permissions().mode() & 0o7777;
        if immutable {
            mode &= !0o222;
        }
        self.insert_entry(
            relative.clone(),
            SourceEntry::File {
                file: SourceFile::Pinned(pinned),
                mode,
            },
        )?;
        self.pending_pinned_paths.push(relative);
        #[cfg(test)]
        {
            self.peak_pinned_files = self.peak_pinned_files.max(self.pending_pinned_paths.len());
        }
        if self.pending_pinned_paths.len() >= ARTIFACT_IO_WORKERS_MAX {
            self.publish_pinned_window()?;
        }
        Ok(())
    }

    /// Add small generated metadata bytes without exposing a mutable staging
    /// pathname to the eventual publisher.
    pub fn insert_bytes(
        &mut self,
        relative: impl AsRef<Path>,
        bytes: &[u8],
        mode: u32,
    ) -> Result<()> {
        checked_mode(mode)?;
        let pinned = super::pin_generated_artifact_bytes(bytes, mode)?;
        self.insert_pinned_file(relative, pinned)
    }

    /// Add a relative symlink whose lexical target remains inside the tree.
    pub fn insert_symlink(
        &mut self,
        relative: impl AsRef<Path>,
        target: impl AsRef<Path>,
    ) -> Result<()> {
        let relative = checked_relative_path(relative.as_ref())?;
        let target = target.as_ref().to_path_buf();
        validate_symlink_target(&relative, &target)?;
        self.insert_entry(relative, SourceEntry::Symlink { target })
    }

    /// Capture one filesystem node without following symlinks.
    pub fn insert_path(
        &mut self,
        relative: impl AsRef<Path>,
        source: impl AsRef<Path>,
    ) -> Result<()> {
        self.insert_path_with_policy(relative, source, false)
    }

    /// Capture one filesystem node and strip write bits from regular files
    /// and directories in the resulting tree.
    #[doc(hidden)]
    pub fn insert_immutable_path(
        &mut self,
        relative: impl AsRef<Path>,
        source: impl AsRef<Path>,
    ) -> Result<()> {
        self.insert_path_with_policy(relative, source, true)
    }

    fn insert_path_with_policy(
        &mut self,
        relative: impl AsRef<Path>,
        source: impl AsRef<Path>,
        immutable: bool,
    ) -> Result<()> {
        let relative = checked_relative_path(relative.as_ref())?;
        let source = source.as_ref();
        let metadata = std::fs::symlink_metadata(source)
            .with_context(|| format!("inspect artifact tree input {}", source.display()))?;
        if metadata.file_type().is_symlink() {
            let target = std::fs::read_link(source)
                .with_context(|| format!("read artifact symlink {}", source.display()))?;
            self.insert_symlink(relative, target)
        } else if metadata.is_dir() {
            let mut mode = metadata.permissions().mode() & 0o7777;
            if immutable {
                mode &= !0o222;
            }
            self.insert_directory(relative, mode)
        } else if metadata.is_file() {
            let pinned = super::pin_content_file(source)?;
            self.insert_pinned_file_with_policy(relative, pinned, immutable)
        } else {
            anyhow::bail!(
                "artifact tree input has unsupported file type: {}",
                source.display()
            );
        }
    }

    /// Recursively capture a directory, regular file, or relative symlink.
    ///
    /// The walk never follows symlinks. Callers which need nextest's bounded
    /// OUT_DIR/include traversal can select those nodes themselves and use
    /// [`Self::insert_path`].
    pub fn insert_tree(
        &mut self,
        relative: impl AsRef<Path>,
        source: impl AsRef<Path>,
    ) -> Result<()> {
        self.insert_tree_with_policy(relative.as_ref(), source.as_ref(), false)
    }

    /// Recursively capture an immutable source tree.
    #[doc(hidden)]
    pub fn insert_immutable_tree(
        &mut self,
        relative: impl AsRef<Path>,
        source: impl AsRef<Path>,
    ) -> Result<()> {
        self.insert_tree_with_policy(relative.as_ref(), source.as_ref(), true)
    }

    fn insert_tree_with_policy(
        &mut self,
        relative: &Path,
        source: &Path,
        immutable: bool,
    ) -> Result<()> {
        let relative = checked_relative_path(relative)?;
        let source = source.to_path_buf();
        let mut pending = vec![(relative, source)];
        let mut captured = Vec::new();
        while let Some((relative, source)) = pending.pop() {
            let metadata = std::fs::symlink_metadata(&source)
                .with_context(|| format!("inspect artifact tree input {}", source.display()))?;
            if metadata.file_type().is_symlink() {
                let target = std::fs::read_link(&source)
                    .with_context(|| format!("read artifact symlink {}", source.display()))?;
                validate_symlink_target(&relative, &target)?;
                captured.push(PendingTreeEntry::Symlink { relative, target });
            } else if metadata.is_dir() {
                let mut mode = metadata.permissions().mode() & 0o7777;
                if immutable {
                    mode &= !0o222;
                }
                captured.push(PendingTreeEntry::Directory {
                    relative: relative.clone(),
                    mode,
                });
                let mut children = std::fs::read_dir(&source)
                    .with_context(|| format!("read artifact directory {}", source.display()))?
                    .map(|entry| {
                        let entry = entry.with_context(|| {
                            format!("read artifact directory entry in {}", source.display())
                        })?;
                        Ok((relative.join(entry.file_name()), entry.path()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                children.sort_by(|left, right| left.0.cmp(&right.0));
                pending.extend(children.into_iter().rev());
            } else if metadata.is_file() {
                captured.push(PendingTreeEntry::File { relative, source });
            } else {
                anyhow::bail!(
                    "artifact tree input has unsupported file type: {}",
                    source.display()
                );
            }
        }

        for entry in captured {
            match entry {
                PendingTreeEntry::Directory { relative, mode } => {
                    self.insert_directory(relative, mode)?;
                }
                PendingTreeEntry::File { relative, source } => {
                    let pinned = super::pin_content_file(&source)
                        .with_context(|| format!("pin artifact tree input {}", source.display()))?;
                    self.insert_pinned_file_with_policy(relative, pinned, immutable)?;
                }
                PendingTreeEntry::Symlink { relative, target } => {
                    self.insert_symlink(relative, target)?;
                }
            }
        }
        Ok(())
    }

    /// Publish the currently pinned descriptor window in parallel, then retain
    /// only compact CAS identities. One namespace gate protects every object
    /// from GC until the record has been materialized, so descriptor use is
    /// bounded by the worker count rather than the source-tree cardinality.
    fn publish_pinned_window(&mut self) -> Result<()> {
        if self.pending_pinned_paths.is_empty() {
            return Ok(());
        }
        if self.content_namespace.is_none() {
            self.content_namespace = Some(super::content::lease_content_namespace()?);
        }

        let pending_paths = std::mem::take(&mut self.pending_pinned_paths);
        let mut pending = Vec::with_capacity(pending_paths.len());
        for path in pending_paths {
            let SourceEntry::File { file, .. } = self
                .entries
                .get_mut(&path)
                .with_context(|| format!("pinned artifact path disappeared: {}", path.display()))?
            else {
                anyhow::bail!("pinned artifact changed type: {}", path.display());
            };
            let SourceFile::Pinned(pinned) = std::mem::replace(file, SourceFile::Publishing) else {
                anyhow::bail!(
                    "pinned artifact file changed state before publication: {}",
                    path.display()
                );
            };
            pending.push((path, pinned));
        }

        let namespace = self
            .content_namespace
            .as_ref()
            .expect("artifact publication initialized its content namespace");
        let published = parallel_indexed(pending, |(path, pinned)| {
            let (source, identity, _) = pinned.into_parts();
            let (content_hash, len) = namespace
                .digest_and_publish_file(&source, identity)
                .with_context(|| format!("publish artifact-tree file {}", path.display()))?;
            Ok((path, SourceFile::Published { content_hash, len }))
        })?;
        for (path, published) in published {
            let SourceEntry::File { file, .. } =
                self.entries.get_mut(&path).with_context(|| {
                    format!("published artifact path disappeared: {}", path.display())
                })?
            else {
                anyhow::bail!("published artifact changed type: {}", path.display());
            };
            anyhow::ensure!(
                matches!(file, SourceFile::Publishing),
                "published artifact file was not awaiting publication: {}",
                path.display()
            );
            *file = published;
        }
        Ok(())
    }

    fn insert_entry(&mut self, path: PathBuf, entry: SourceEntry) -> Result<()> {
        #[cfg(test)]
        {
            self.insertion_validation_visits += 1;
        }
        anyhow::ensure!(
            !self.entries.contains_key(&path),
            "duplicate artifact tree path: {}",
            path.display()
        );

        let mut parent = path.parent();
        while let Some(ancestor) = parent.filter(|ancestor| !ancestor.as_os_str().is_empty()) {
            #[cfg(test)]
            {
                self.insertion_validation_visits += 1;
            }
            anyhow::ensure!(
                self.entries
                    .get(ancestor)
                    .is_none_or(|entry| matches!(entry, SourceEntry::Directory { .. })),
                "artifact tree non-directory {} is an ancestor of {}",
                ancestor.display(),
                path.display(),
            );
            parent = ancestor.parent();
        }

        if !matches!(&entry, SourceEntry::Directory { .. }) {
            #[cfg(test)]
            {
                self.insertion_validation_visits += 1;
            }
            anyhow::ensure!(
                !self.descendant_counts.contains_key(&path),
                "artifact tree non-directory {} is an ancestor of an existing entry",
                path.display(),
            );
        }

        self.entries.insert(path.clone(), entry);
        let mut parent = path.parent();
        while let Some(ancestor) = parent.filter(|ancestor| !ancestor.as_os_str().is_empty()) {
            *self
                .descendant_counts
                .entry(ancestor.to_path_buf())
                .or_default() += 1;
            parent = ancestor.parent();
        }
        Ok(())
    }

    #[cfg(test)]
    fn insertion_validation_visits(&self) -> usize {
        self.insertion_validation_visits
    }
}

impl Default for ArtifactTreeSource {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum RecordEntry {
    Directory {
        path: Vec<u8>,
        mode: u32,
    },
    File {
        path: Vec<u8>,
        mode: u32,
        content_hash: u64,
        len: u64,
    },
    Symlink {
        path: Vec<u8>,
        target: Vec<u8>,
    },
}

impl RecordEntry {
    fn path_bytes(&self) -> &[u8] {
        match self {
            Self::Directory { path, .. } | Self::File { path, .. } | Self::Symlink { path, .. } => {
                path
            }
        }
    }

    fn path(&self) -> Result<PathBuf> {
        checked_relative_bytes(self.path_bytes())
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactTreeRecord {
    version: u32,
    identity: u64,
    entries: Vec<RecordEntry>,
    integrity_ahash: u64,
}

struct LeasedRecord {
    record: ArtifactTreeRecord,
    // One shared namespace gate prevents GC from unlinking every referenced
    // CAS pathname. Individual object descriptors are opened only inside the
    // bounded materialization worker window.
    content_namespace: super::content::ContentNamespaceLease,
}

struct SelectedRecord {
    leased: LeasedRecord,
    cache_hit: bool,
}

/// Crash-recoverable ownership of one complete artifact closure.
///
/// Acquisition is serialized against the global lifecycle collector, then
/// this per-identity shared flock remains live through runtime execution. A
/// collector can therefore reclaim unrelated inactive closures without ever
/// unlinking an embedded stable source/OUT_DIR pathname still used by a test.
struct ArtifactClosureLease {
    lock: Option<std::os::fd::OwnedFd>,
    lifecycle_root: PathBuf,
    identity: u64,
}

impl ArtifactClosureLease {
    fn release(&mut self) {
        self.lock.take();
    }

    fn acquire_shared(&mut self) -> Result<()> {
        if self.lock.is_some() {
            return Ok(());
        }
        let global = crate::flock::block_flock(
            lifecycle_gate_path(&self.lifecycle_root),
            crate::flock::FlockMode::Shared,
        )?;
        let closure = crate::flock::block_flock(
            lifecycle_closure_lock_path(&self.lifecycle_root, self.identity),
            crate::flock::FlockMode::Shared,
        )?;
        drop(global);
        self.lock = Some(closure);
        Ok(())
    }

    fn with_exclusive_rebuild<T>(
        &mut self,
        operation: impl FnOnce(&ArtifactClosureLease) -> Result<T>,
    ) -> Result<T> {
        self.release();
        let closure_path = lifecycle_closure_lock_path(&self.lifecycle_root, self.identity);
        loop {
            // Never sleep on a closure while holding global EX. Runtime owners
            // may acquire a second identity (stable source -> build closure);
            // blocking here would form global-EX/closure-SH ABBA.
            let global = crate::flock::block_flock(
                lifecycle_gate_path(&self.lifecycle_root),
                crate::flock::FlockMode::Exclusive,
            )?;
            if let Some(closure) =
                crate::flock::try_flock(&closure_path, crate::flock::FlockMode::Exclusive)?
            {
                self.lock = Some(closure);
                drop(global);
                break;
            }
            drop(global);

            // Join the kernel's writer queue without the namespace gate, then
            // release and retry the atomic global->try-closure transition.
            let observed =
                crate::flock::block_flock(&closure_path, crate::flock::FlockMode::Exclusive)?;
            drop(observed);
        }
        let result = operation(self);
        if let Some(lock) = &self.lock {
            super::content::flock_retry(lock, rustix::fs::FlockOperation::LockShared)
                .with_context(|| {
                    format!(
                        "downgrade rebuilt artifact closure {:016x} to shared ownership",
                        self.identity
                    )
                })?;
        }
        result
    }
}

/// One cold producer's conservative disk-working-set reservation.
///
/// The exclusive flock is released by the kernel on every exit path,
/// including SIGKILL. The collector ignores unlocked crash leftovers and
/// removes them while holding its global namespace gate.
struct BuildSpaceReservation {
    // NamedTempFile retries randomized names, so a crash leftover can never
    // collide with a later process which happens to reuse the same PID. Its
    // open file owns the exclusive flock and normal Drop atomically removes
    // the pathname; SIGKILL leaves an unlocked file for lifecycle GC.
    _temporary: tempfile::NamedTempFile,
}

/// One live validation epoch for a stable source root.
///
/// The first owner after the previous epoch drains recursively validates the
/// root. Concurrent owners join that epoch under a shared kernel flock and
/// need only recheck the marker/root inode. This keeps one complete source
/// walk as the deletion-recovery boundary without making every Cargo lane
/// repeat it while the same immutable root is already in use.
struct StableRootValidationLease {
    _epoch: std::os::fd::OwnedFd,
}

/// One private materialization of a reusable artifact tree.
///
/// Every regular file is an independent FICLONE inode by construction. CAS
/// descriptors and the namespace lease are therefore released as soon as the
/// bounded materialization pass completes.
#[doc(hidden)]
pub struct MaterializedArtifactTree {
    directory: tempfile::TempDir,
    // Declared after `directory` deliberately: Rust drops fields in
    // declaration order, so TempDir removes the namespace while this lock is
    // still held. A collector can never acquire the liveness lock in the
    // middle of normal teardown.
    _live: std::fs::File,
    // This must outlive every execution-facing pathname in `directory`.
    _closure: ArtifactClosureLease,
    cache_root: PathBuf,
    identity: u64,
    cache_hit: bool,
    waited: bool,
    elapsed: Duration,
}

/// Deterministic persistent materialization used when built artifacts embed
/// their source pathname (notably `env!("CARGO_MANIFEST_DIR")`).
///
/// Unlike [`MaterializedArtifactTree`], this owner does not remove its root on
/// drop. The root is an immutable cache object and can therefore remain the
/// pathname embedded in binaries reused by later processes/checkouts.
#[doc(hidden)]
pub struct StableArtifactTree {
    root: PathBuf,
    identity: u64,
    cache_hit: bool,
    // Stable trees embed this absolute pathname in later Cargo output. Keep
    // the closure lease independent of record validity for the full owner
    // lifetime.
    _closure: ArtifactClosureLease,
    // Keep the validated source-root epoch live for every consumer of the
    // embedded pathname. Once the last owner drops, the next invocation must
    // perform a fresh recursive validation before starting a new epoch.
    _validation: StableRootValidationLease,
}

impl StableArtifactTree {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }

    pub fn cache_hit(&self) -> bool {
        self.cache_hit
    }
}

impl MaterializedArtifactTree {
    /// Root of the private relocatable tree.
    pub fn root(&self) -> &Path {
        self.directory.path()
    }

    /// Move this private COW materialization into a caller-owned persistent
    /// directory.
    ///
    /// The rename occurs while the materialization liveness lock and artifact
    /// closure lease are still held. This closes the otherwise-racy gap
    /// between disabling [`tempfile::TempDir`] cleanup and installing the tree
    /// in a lifecycle-managed namespace. The destination must be on the same
    /// filesystem and must not already exist.
    #[doc(hidden)]
    pub fn persist_at(self, destination: &Path) -> Result<PathBuf> {
        anyhow::ensure!(
            !destination.exists(),
            "persistent artifact-tree destination already exists: {}",
            destination.display(),
        );
        let parent = destination.parent().with_context(|| {
            format!(
                "persistent artifact-tree destination has no parent: {}",
                destination.display(),
            )
        })?;
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create persistent artifact-tree parent {}",
                parent.display(),
            )
        })?;

        let MaterializedArtifactTree {
            directory,
            _live,
            _closure,
            ..
        } = self;
        let source = directory.path().to_path_buf();
        std::fs::rename(&source, destination).with_context(|| {
            format!(
                "persist artifact-tree materialization {} -> {}",
                source.display(),
                destination.display(),
            )
        })?;
        // The directory has moved while its liveness lock was held. Disarm
        // TempDir cleanup before releasing that lock; the caller's lifecycle
        // manager owns the destination from this point forward.
        let _detached_source = directory.keep();
        drop(_live);
        drop(_closure);
        Ok(destination.to_path_buf())
    }

    /// Input-addressed identity of this tree.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Whether an existing published record supplied this tree.
    pub fn cache_hit(&self) -> bool {
        self.cache_hit
    }

    /// Whether this process waited behind another same-identity producer.
    pub fn waited(&self) -> bool {
        self.waited
    }

    /// Total lookup, election/build, publication, and materialization time.
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Persist one compact cache decision beside captured build diagnostics.
    ///
    /// Diagnostics are best-effort and deliberately separate from the cache
    /// protocol: a missing or unwritable diagnostics directory can never turn
    /// a valid cache hit/build into a test failure.
    pub fn persist_decision_diagnostic(
        &self,
        kind: &str,
        semantic_components: serde_json::Value,
    ) -> Result<Option<PathBuf>> {
        let Some(root) = std::env::var_os("KTSTR_BUILD_DIAGNOSTICS_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
        else {
            return Ok(None);
        };
        std::fs::create_dir_all(&root)
            .with_context(|| format!("create artifact-cache diagnostics dir {}", root.display()))?;
        let component = diagnostic_component(kind);
        let path = root.join(format!(
            "artifact-cache-{component}-{}-{:016x}.json",
            std::process::id(),
            self.identity,
        ));
        let role = if !self.cache_hit {
            if self.waited {
                "successor_producer"
            } else {
                "producer"
            }
        } else if self.waited {
            "waiter"
        } else {
            "direct_hit"
        };
        let record = serde_json::json!({
            "version": 1,
            "kind": kind,
            "identity": format!("{:016x}", self.identity),
            "semantic_components": semantic_components,
            "outcome": if self.cache_hit { "hit" } else { "miss" },
            "role": role,
            "cache_root": self.cache_root,
            "materialization_root": self.root(),
            "elapsed_ms": self.elapsed.as_millis(),
        });
        let mut temporary = tempfile::Builder::new()
            .prefix(&format!(".artifact-cache-{component}-"))
            .tempfile_in(&root)
            .with_context(|| format!("create artifact-cache diagnostic in {}", root.display()))?;
        serde_json::to_writer_pretty(temporary.as_file_mut(), &record)
            .context("serialize artifact-cache decision")?;
        temporary
            .as_file_mut()
            .flush()
            .context("flush artifact-cache decision")?;
        temporary
            .persist(&path)
            .map_err(|error| error.error)
            .with_context(|| format!("publish artifact-cache diagnostic {}", path.display()))?;
        Ok(Some(path))
    }
}

fn diagnostic_component(value: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() {
            out.push(char::from(byte.to_ascii_lowercase()));
            dash = false;
        } else if !out.is_empty() && !dash {
            out.push('-');
            dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        out.push_str("tree");
    }
    out
}

/// Best-effort disk-pressure reclaimer invoked at the cold-build reservation
/// boundary when the filesystem is below its build-space reserve even after
/// lifecycle collection ran.
///
/// Given the shortfall in bytes, it frees pure-cache scratch space owned by the
/// caller (worst case one cold rebuild) and returns the bytes it reclaimed.
/// Installed via [`ArtifactTreeCache::with_build_space_reclaimer`]; the library
/// stays agnostic to the caller's scratch layout and locking scheme. Must not
/// touch the output the pending build is about to use — the caller exempts it.
pub type BuildSpaceReclaimer = Box<dyn Fn(u64) -> u64 + Send + Sync>;

/// Cross-process cache for immutable, relocatable artifact trees.
#[doc(hidden)]
pub struct ArtifactTreeCache {
    root: PathBuf,
    /// Optional caller hook, run only when a cold-build reservation still finds
    /// the filesystem below its reserve after collection. See
    /// [`BuildSpaceReclaimer`].
    build_space_reclaimer: Option<BuildSpaceReclaimer>,
}

/// Deterministic persistent Cargo output pathname for one artifact identity.
///
/// Producers build directly here so absolute compile-time `OUT_DIR` values
/// remain valid. The generic cache seals the complete root before publishing
/// its relocatable execution closure.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct StableCargoBuild {
    pub root: PathBuf,
    pub target_directory: PathBuf,
}

impl ArtifactTreeCache {
    /// Use an explicit record/election root. Content bytes still converge in
    /// ktstr's ordinary machine-wide content CAS.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            build_space_reclaimer: None,
        }
    }

    /// Install a [`BuildSpaceReclaimer`] hook run at the cold-build reservation
    /// boundary when the filesystem remains below its build-space reserve after
    /// lifecycle collection. Additive builder: cold builds through a cache
    /// without one keep the historical warn-only behavior.
    pub fn with_build_space_reclaimer(mut self, reclaimer: BuildSpaceReclaimer) -> Self {
        self.build_space_reclaimer = Some(reclaimer);
        self
    }

    fn lifecycle_root(&self) -> PathBuf {
        self.root
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(&self.root)
            .to_path_buf()
    }

    fn acquire_closure(&self, identity: u64) -> Result<ArtifactClosureLease> {
        let lifecycle_root = self.lifecycle_root();
        ensure_lifecycle_dirs(&lifecycle_root)?;

        // Lock order is global SH -> closure SH. The global gate is released
        // immediately after the closure lock is established, allowing GC of
        // unrelated identities during a long build or test run.
        let global = crate::flock::block_flock(
            lifecycle_gate_path(&lifecycle_root),
            crate::flock::FlockMode::Shared,
        )?;
        let closure = crate::flock::block_flock(
            lifecycle_closure_lock_path(&lifecycle_root, identity),
            crate::flock::FlockMode::Shared,
        )?;
        drop(global);
        Ok(ArtifactClosureLease {
            lock: Some(closure),
            lifecycle_root,
            identity,
        })
    }

    fn acquire_stable_tree_validation(
        &self,
        root: &Path,
        identity: u64,
        freshly_published: bool,
    ) -> Result<Option<StableRootValidationLease>> {
        let validation_root = self.root.join(STABLE_VALIDATION_LOCK_DIR);
        std::fs::create_dir_all(&validation_root).with_context(|| {
            format!(
                "create stable-root validation lock directory {}",
                validation_root.display(),
            )
        })?;
        let coordination_path = validation_root.join(format!("{identity:016x}.coord"));
        let epoch_path = validation_root.join(format!("{identity:016x}.epoch"));

        // Serialize the EX-probe -> SH-join transition. Without this mutex a
        // follower could observe contention, the last epoch owner could drop,
        // and a new leader could begin rebuilding before the follower joined.
        let coordination =
            crate::flock::block_flock(&coordination_path, crate::flock::FlockMode::Exclusive)?;
        let (epoch, leader) = if let Some(epoch) =
            crate::flock::try_flock(&epoch_path, crate::flock::FlockMode::Exclusive)?
        {
            (epoch, true)
        } else {
            (
                crate::flock::block_flock(&epoch_path, crate::flock::FlockMode::Shared)?,
                false,
            )
        };

        let complete = if leader && !freshly_published {
            // No validated owner remains, so a new invocation must prove the
            // complete recursive seal. Partial manual cache deletion therefore
            // becomes a normal reconstructible miss on the very next epoch.
            stable_tree_is_complete(root, identity)?
        } else {
            // A live epoch leader already proved the listing, or this caller
            // just published it with the same double-scan protocol. Bind this
            // owner to the exact marker and root inode without walking files.
            stable_root_marker_matches(root, StableRootDomain::Source, identity, false)?
        };
        if !complete {
            drop(epoch);
            drop(coordination);
            return Ok(None);
        }
        if leader {
            super::content::flock_retry(&epoch, rustix::fs::FlockOperation::LockShared)
                .with_context(|| {
                    format!(
                        "downgrade stable-root validation epoch {identity:016x} to shared ownership"
                    )
                })?;
        }
        drop(coordination);
        Ok(Some(StableRootValidationLease { _epoch: epoch }))
    }

    fn reserve_cold_build_space(
        &self,
        closure: &ArtifactClosureLease,
    ) -> Result<BuildSpaceReservation> {
        anyhow::ensure!(
            closure.lock.is_none(),
            "artifact build-space reservation attempted while holding closure {:016x}; lifecycle lock order is global then closure",
            closure.identity,
        );
        let bytes = historical_build_space_reservation(&self.root)?;
        let directory = closure
            .lifecycle_root
            .join(LIFECYCLE_DIR)
            .join(LIFECYCLE_RESERVATION_DIR);
        let global = crate::flock::block_flock(
            lifecycle_gate_path(&closure.lifecycle_root),
            crate::flock::FlockMode::Shared,
        )?;
        let mut temporary = tempfile::Builder::new()
            .prefix(&format!(
                "{:016x}-{}-",
                closure.identity,
                std::process::id(),
            ))
            .suffix(".reserve")
            .tempfile_in(&directory)
            .with_context(|| {
                format!(
                    "create artifact build reservation in {}",
                    directory.display()
                )
            })?;
        super::content::flock_retry(
            temporary.as_file(),
            rustix::fs::FlockOperation::LockExclusive,
        )
        .with_context(|| {
            format!(
                "lock artifact build reservation {}",
                temporary.path().display()
            )
        })?;
        writeln!(temporary.as_file_mut(), "{bytes}").with_context(|| {
            format!(
                "write artifact build reservation {}",
                temporary.path().display()
            )
        })?;
        drop(global);
        let reservation = BuildSpaceReservation {
            _temporary: temporary,
        };
        collect_artifact_cache(
            &closure.lifecycle_root,
            Some(closure.identity),
            Some(&self.root),
            true,
        )?;
        let global = crate::flock::block_flock(
            lifecycle_gate_path(&closure.lifecycle_root),
            crate::flock::FlockMode::Shared,
        )?;
        let (active_reservations, _) = active_build_reservations(&closure.lifecycle_root)?;
        let mut after = lifecycle_filesystem_space(&closure.lifecycle_root)?;
        drop(global);
        let required = LIFECYCLE_MIN_FREE_RESERVE
            .max(after.capacity / LIFECYCLE_FREE_RESERVE_DIVISOR)
            .saturating_add(active_reservations);
        // Lifecycle collection reclaims only the managed closure/content cache.
        // A caller (e.g. the Cargo producer) may own a separate pure-cache
        // scratch space on the same filesystem whose orphans lifecycle GC never
        // sees. When the reserve is still short after collection, invoke the
        // caller's reclaimer so this disk-pressure event drives that cleanup at
        // the same reservation boundary the lifecycle GC uses — rather than
        // waiting for the scratch's own slow idle timer.
        if after.available < required
            && let Some(reclaimer) = &self.build_space_reclaimer
        {
            let shortfall = required - after.available;
            let reclaimed = reclaimer(shortfall);
            if reclaimed > 0 {
                after = lifecycle_filesystem_space(&closure.lifecycle_root)?;
                tracing::info!(
                    shortfall_bytes = shortfall,
                    reclaimed_bytes = reclaimed,
                    available_bytes = after.available,
                    required_bytes = required,
                    "reclaimed caller scratch space under cold-build disk pressure",
                );
            }
        }
        if after.available < required {
            tracing::warn!(
                available_bytes = after.available,
                required_bytes = required,
                active_reservations_bytes = active_reservations,
                "artifact cache reclaimed every inactive closure but the filesystem remains below its build-space reserve",
            );
        }
        Ok(reservation)
    }

    /// Reuse or build and publish one exact artifact-tree identity.
    ///
    /// The producer closure must build in caller-owned private state and
    /// return pinned outputs. `validate_identity` runs immediately before a
    /// hit is accepted and immediately before a cold record is published.
    pub fn load_or_build<F, V, C>(
        &self,
        identity: u64,
        materialize_parent: &Path,
        progress_label: &str,
        validate_identity: V,
        cancelled: C,
        build: F,
    ) -> Result<MaterializedArtifactTree>
    where
        F: FnOnce() -> Result<ArtifactTreeSource>,
        V: Fn() -> Result<bool>,
        C: Fn() -> bool,
    {
        self.load_or_build_with_validators(
            identity,
            materialize_parent,
            progress_label,
            &validate_identity,
            &validate_identity,
            cancelled,
            build,
        )
    }

    /// Variant of [`Self::load_or_build`] with distinct validation for a
    /// previously published hit and a newly produced closure.
    ///
    /// Callers which have just computed an input identity may accept an
    /// immutable matching hit without repeating that expensive scan, while
    /// still revalidating mutable producer inputs after a long cold build.
    #[allow(clippy::too_many_arguments)] // Keep cache lifecycle hooks explicit at the API boundary.
    pub fn load_or_build_with_validators<F, VH, VP, C>(
        &self,
        identity: u64,
        materialize_parent: &Path,
        progress_label: &str,
        validate_cached_identity: VH,
        validate_published_identity: VP,
        cancelled: C,
        build: F,
    ) -> Result<MaterializedArtifactTree>
    where
        F: FnOnce() -> Result<ArtifactTreeSource>,
        VH: Fn() -> Result<bool>,
        VP: Fn() -> Result<bool>,
        C: Fn() -> bool,
    {
        self.load_or_build_with_validators_and_post_publish(
            identity,
            materialize_parent,
            progress_label,
            |_| Ok(true),
            validate_cached_identity,
            validate_published_identity,
            cancelled,
            build,
            |_| Ok(()),
        )
    }

    /// Internal publication form with a producer-only step after every pinned
    /// source inode has reached the content CAS but before the record becomes
    /// visible. Stable Cargo outputs use this boundary to seal their producer
    /// tree without invalidating the identities used for content hashing.
    #[allow(clippy::too_many_arguments)] // Keep cache lifecycle hooks explicit at the API boundary.
    fn load_or_build_with_validators_and_post_publish<F, U, VH, VP, C, P>(
        &self,
        identity: u64,
        materialize_parent: &Path,
        progress_label: &str,
        cache_usable: U,
        validate_cached_identity: VH,
        validate_published_identity: VP,
        cancelled: C,
        build: F,
        post_publish: P,
    ) -> Result<MaterializedArtifactTree>
    where
        F: FnOnce() -> Result<ArtifactTreeSource>,
        U: Fn(&ArtifactTreeRecord) -> Result<bool>,
        VH: Fn() -> Result<bool>,
        VP: Fn() -> Result<bool>,
        C: Fn() -> bool,
        P: FnOnce(&LeasedRecord) -> Result<()>,
    {
        anyhow::ensure!(!progress_label.is_empty(), "artifact tree label is empty");
        if cancelled() {
            anyhow::bail!("artifact tree cache operation interrupted");
        }
        // The closure lease precedes the very first record/stable-root check.
        // Record corruption is reconstructible, but its paired stable path can
        // still be used by a live binary through an embedded OUT_DIR.
        let closure = std::cell::RefCell::new(self.acquire_closure(identity)?);
        ensure_cache_dirs(&self.root)?;
        std::fs::create_dir_all(materialize_parent).with_context(|| {
            format!(
                "create artifact materialization parent {}",
                materialize_parent.display()
            )
        })?;
        gc_stale_materializations(materialize_parent, SystemTime::now())?;
        let (record_path, lock_path, namespace_gate) = cache_paths(&self.root, identity);
        let started = Instant::now();
        let waited = std::cell::Cell::new(false);
        let selected = super::content::load_or_build_with_wait(
            &namespace_gate,
            &lock_path,
            &format!("artifact tree {identity:016x}"),
            || {
                if cancelled() {
                    anyhow::bail!("artifact tree cache operation interrupted");
                }
                closure.borrow_mut().acquire_shared()?;
                let leased = match read_and_lease_record(&record_path, identity) {
                    Ok(leased) => leased,
                    Err(error) => {
                        tracing::debug!(
                            path = %record_path.display(),
                            error = %error,
                            "rebuilding invalid reconstructible artifact tree record",
                        );
                        None
                    }
                };
                let Some(leased) = leased else {
                    closure.borrow_mut().release();
                    return Ok(None);
                };
                if !cache_usable(&leased.record)? {
                    tracing::debug!(
                        path = %record_path.display(),
                        "rebuilding artifact tree whose coupled stable output is incomplete",
                    );
                    closure.borrow_mut().release();
                    return Ok(None);
                }
                anyhow::ensure!(
                    validate_cached_identity()?,
                    "artifact tree inputs changed before accepting cached build {identity:016x}"
                );
                if cancelled() {
                    anyhow::bail!("artifact tree cache operation interrupted");
                }
                Ok(Some(SelectedRecord {
                    leased,
                    cache_hit: true,
                }))
            },
            || {
                if cancelled() {
                    anyhow::bail!("artifact tree cache operation interrupted");
                }
                // Capacity reclamation must complete before closure EX. The
                // global lifecycle gate is always acquired before a closure
                // lock; reversing that order would deadlock a new waiter
                // holding global SH while draining this identity.
                closure.borrow_mut().release();
                let _space = {
                    let closure = closure.borrow();
                    self.reserve_cold_build_space(&closure)?
                };
                closure.borrow_mut().with_exclusive_rebuild(|_closure| {
                    let source = build()?;
                    anyhow::ensure!(
                        !source.is_empty(),
                        "artifact tree producer emitted no entries"
                    );
                    let leased = publish_source(identity, source)?;
                    post_publish(&leased)?;
                    anyhow::ensure!(
                        validate_published_identity()?,
                        "artifact tree inputs changed before publishing build {identity:016x}"
                    );
                    if cancelled() {
                        anyhow::bail!("artifact tree cache operation interrupted");
                    }
                    leased.content_namespace.publish_artifact_reference(
                        &record_path,
                        identity,
                        &record_content_objects(&leased.record),
                    )?;
                    write_record(&record_path, &leased.record)?;
                    if let Err(error) = update_build_space_high_water(&self.root, &leased.record) {
                        tracing::debug!(
                            namespace = %self.root.display(),
                            error = %error,
                            "could not update reconstructible artifact build-space history",
                        );
                    }
                    Ok(SelectedRecord {
                        leased,
                        cache_hit: false,
                    })
                })
            },
            |coordination| {
                waited.set(true);
                wait_for_lock(
                    coordination,
                    &lock_path,
                    progress_label,
                    identity,
                    false,
                    &cancelled,
                )
            },
            |coordination| {
                waited.set(true);
                wait_for_lock(
                    coordination,
                    &lock_path,
                    progress_label,
                    identity,
                    true,
                    &cancelled,
                )
            },
        )?;
        let closure = closure.into_inner();
        // `load_or_build_with_wait` owns the election file. It has returned,
        // so its LOCK_EX/LOCK_SH is gone before any potentially large clone
        // fanout starts. Same-key consumers may materialize concurrently.
        let mut result = materialize(
            selected.leased,
            materialize_parent,
            &self.root,
            identity,
            selected.cache_hit,
            waited.get(),
            closure,
        )?;
        touch_closure_access(&result._closure);
        result.elapsed = started.elapsed();
        eprintln!(
            "{progress_label}: artifact-tree {} {identity:016x}; role={}; elapsed={}",
            if result.cache_hit { "hit" } else { "miss" },
            if !result.cache_hit {
                if result.waited {
                    "successor-producer"
                } else {
                    "producer"
                }
            } else if result.waited {
                "waiter"
            } else {
                "direct-hit"
            },
            humantime::format_duration(result.elapsed),
        );
        Ok(result)
    }

    /// Reuse or build an artifact closure whose Cargo producer writes at a
    /// deterministic persistent pathname.
    ///
    /// This is the uniform producer shape for harness, coverage, and
    /// scheduler builds. The stable output is lifecycle-coupled to the record
    /// election while each execution still receives a private reflink tree.
    #[allow(clippy::too_many_arguments)] // Keep cache lifecycle hooks explicit at the API boundary.
    pub fn load_or_build_with_stable_cargo_output<F, V, C>(
        &self,
        identity: u64,
        stable_parent: &Path,
        materialize_parent: &Path,
        progress_label: &str,
        validate_identity: V,
        cancelled: C,
        build: F,
    ) -> Result<MaterializedArtifactTree>
    where
        F: FnOnce(&StableCargoBuild) -> Result<ArtifactTreeSource>,
        V: Fn() -> Result<bool>,
        C: Fn() -> bool,
    {
        self.load_or_build_with_stable_cargo_output_validators(
            identity,
            stable_parent,
            materialize_parent,
            progress_label,
            &validate_identity,
            &validate_identity,
            cancelled,
            build,
        )
    }

    /// Stable-output variant with a cheap cached-hit validator and a distinct
    /// post-build validator for mutable producer inputs.
    #[allow(clippy::too_many_arguments)] // Keep cache lifecycle hooks explicit at the API boundary.
    pub fn load_or_build_with_stable_cargo_output_validators<F, VH, VP, C>(
        &self,
        identity: u64,
        stable_parent: &Path,
        materialize_parent: &Path,
        progress_label: &str,
        validate_cached_identity: VH,
        validate_published_identity: VP,
        cancelled: C,
        build: F,
    ) -> Result<MaterializedArtifactTree>
    where
        F: FnOnce(&StableCargoBuild) -> Result<ArtifactTreeSource>,
        VH: Fn() -> Result<bool>,
        VP: Fn() -> Result<bool>,
        C: Fn() -> bool,
    {
        std::fs::create_dir_all(stable_parent).with_context(|| {
            format!(
                "create stable Cargo output parent {}",
                stable_parent.display()
            )
        })?;
        let stable_root = stable_parent.join(format!("{identity:016x}"));
        let stable_to_seal = StableCargoBuild {
            root: stable_root.clone(),
            target_directory: stable_root.join("target"),
        };
        let tree = self.load_or_build_with_validators_and_post_publish(
            identity,
            materialize_parent,
            progress_label,
            |record| stable_cargo_build_is_complete(&stable_root, identity, Some(record)),
            validate_cached_identity,
            || {
                Ok(
                    stable_cargo_build_is_complete(&stable_root, identity, None)?
                        && validate_published_identity()?,
                )
            },
            cancelled,
            || {
                let stable = prepare_stable_cargo_build(&stable_root)?;
                let result = build(&stable);
                if result.is_err()
                    && let Err(error) = remove_stable_tree(&stable_root)
                {
                    tracing::warn!(
                        error = %error,
                        path = %stable_root.display(),
                        "could not remove failed stable Cargo output",
                    );
                }
                result
            },
            |leased| {
                let result = distill_stable_cargo_build(
                    &stable_to_seal,
                    &leased.record,
                    &leased.content_namespace,
                )
                .and_then(|()| seal_stable_cargo_build(&stable_to_seal, identity));
                if result.is_err()
                    && let Err(error) = remove_stable_tree(&stable_to_seal.root)
                {
                    tracing::warn!(
                        error = %error,
                        path = %stable_to_seal.root.display(),
                        "could not remove failed stable Cargo output",
                    );
                }
                result
            },
        )?;
        anyhow::ensure!(
            stable_cargo_build_is_complete(&stable_root, identity, None)?,
            "artifact {identity:016x} has no complete stable Cargo output at {}",
            stable_root.display()
        );
        Ok(tree)
    }

    /// Reuse or atomically install one deterministic immutable tree.
    ///
    /// This is intentionally a separate shape from private run trees: only
    /// source snapshots whose pathname is itself part of compiled output may
    /// use it. A per-identity kernel flock serializes the one-time rename;
    /// ordinary output trees retain concurrent private COW materialization.
    pub fn load_or_build_stable<F, V, C>(
        &self,
        identity: u64,
        stable_parent: &Path,
        progress_label: &str,
        validate_identity: V,
        cancelled: C,
        build: F,
    ) -> Result<StableArtifactTree>
    where
        F: FnOnce() -> Result<ArtifactTreeSource>,
        V: Fn() -> Result<bool>,
        C: Fn() -> bool,
    {
        self.load_or_build_stable_with_validators(
            identity,
            stable_parent,
            progress_label,
            &validate_identity,
            &validate_identity,
            cancelled,
            build,
        )
    }

    /// Stable-tree variant with distinct validators for an immutable cache hit
    /// and a freshly captured source closure.
    #[allow(clippy::too_many_arguments)] // Keep cache lifecycle hooks explicit at the API boundary.
    pub fn load_or_build_stable_with_validators<F, VH, VP, C>(
        &self,
        identity: u64,
        stable_parent: &Path,
        progress_label: &str,
        validate_cached_identity: VH,
        validate_published_identity: VP,
        cancelled: C,
        build: F,
    ) -> Result<StableArtifactTree>
    where
        F: FnOnce() -> Result<ArtifactTreeSource>,
        VH: Fn() -> Result<bool>,
        VP: Fn() -> Result<bool>,
        C: Fn() -> bool,
    {
        // Unlike the generic path this function has a direct-hit fast path,
        // so it must establish liveness before inspecting `final_root`.
        let mut closure = self.acquire_closure(identity)?;
        std::fs::create_dir_all(stable_parent).with_context(|| {
            format!(
                "create stable artifact-tree parent {}",
                stable_parent.display()
            )
        })?;
        let final_root = stable_parent.join(format!("{identity:016x}"));
        if let Some(validation) =
            self.acquire_stable_tree_validation(&final_root, identity, false)?
        {
            anyhow::ensure!(
                validate_cached_identity()?,
                "artifact tree inputs changed before accepting stable build {identity:016x}"
            );
            touch_closure_access(&closure);
            return Ok(StableArtifactTree {
                root: final_root,
                identity,
                cache_hit: true,
                _closure: closure,
                _validation: validation,
            });
        }

        // A contender waiting on the stable installer lock must not retain
        // closure SH: the installer may need closure EX to replace an
        // incomplete root. Reacquire SH only after this process owns the
        // installer turnstile, then repeat the direct-hit check.
        closure.release();
        let stable_locks = self.root.join(".stable-materialization-locks-v1");
        std::fs::create_dir_all(&stable_locks).with_context(|| {
            format!(
                "create stable artifact-tree lock dir {}",
                stable_locks.display()
            )
        })?;
        let _lock = crate::flock::block_flock(
            stable_locks.join(format!("{identity:016x}.lock")),
            crate::flock::FlockMode::Exclusive,
        )?;
        closure.acquire_shared()?;
        if let Some(validation) =
            self.acquire_stable_tree_validation(&final_root, identity, false)?
        {
            anyhow::ensure!(
                validate_cached_identity()?,
                "artifact tree inputs changed before accepting stable build {identity:016x}"
            );
            touch_closure_access(&closure);
            return Ok(StableArtifactTree {
                root: final_root,
                identity,
                cache_hit: true,
                _closure: closure,
                _validation: validation,
            });
        }
        // This must inspect the directory entry itself rather than `exists()`:
        // a dangling attacker-controlled symlink is still stale cache state
        // which must be unlinked before the atomic installation below.
        closure.release();
        closure.with_exclusive_rebuild(|_| {
            remove_stable_tree(&final_root).with_context(|| {
                format!(
                    "remove incomplete stable artifact tree {}",
                    final_root.display()
                )
            })
        })?;
        // The generic cache operation establishes its own lease before
        // checking the record. Releasing this direct-hit lease prevents a
        // same-process SH owner from blocking its cold-builder EX transition.
        closure.release();

        let tree = self.load_or_build_with_validators(
            identity,
            stable_parent,
            progress_label,
            validate_cached_identity,
            validate_published_identity,
            cancelled,
            build,
        )?;
        let cache_hit = tree.cache_hit();

        // Drop the liveness lock only after detaching TempDir cleanup. The
        // installed FICLONEs are independent inodes and retain no CAS lease.
        let MaterializedArtifactTree {
            directory,
            _live,
            _closure: materialized_closure,
            ..
        } = tree;
        let staged = directory.keep();
        drop(_live);
        std::fs::remove_file(staged.join(MATERIALIZATION_LIVE_LOCK)).with_context(|| {
            format!(
                "remove private liveness marker from stable artifact tree {}",
                staged.display()
            )
        })?;
        // Stable files remain read-only, while directories remain owner
        // writable/searchable. Directory write protection is not an
        // immutability boundary for the owning uid (which can chmod it back),
        // and 0555 cache trees poison ordinary checkout/rm cleanup after a
        // cancelled process. Marker/record validation is the cache protocol.
        for entry in walkdir::WalkDir::new(&staged).follow_links(false) {
            let entry = entry.with_context(|| {
                format!("walk stable artifact-tree directories {}", staged.display())
            })?;
            if !entry.file_type().is_dir() {
                continue;
            }
            let metadata = entry.metadata().with_context(|| {
                format!(
                    "stat stable artifact-tree directory {}",
                    entry.path().display()
                )
            })?;
            let mode = metadata.permissions().mode() & 0o7777;
            std::fs::set_permissions(entry.path(), std::fs::Permissions::from_mode(mode | 0o700))
                .with_context(|| {
                format!(
                    "make stable artifact-tree directory owner-removable {}",
                    entry.path().display()
                )
            })?;
        }
        publish_stable_root_marker(&staged, StableRootDomain::Source, identity)?;
        std::fs::rename(&staged, &final_root).with_context(|| {
            format!(
                "atomically install stable artifact tree {} -> {}",
                staged.display(),
                final_root.display(),
            )
        })?;
        let validation = self
            .acquire_stable_tree_validation(&final_root, identity, true)?
            .with_context(|| {
                format!(
                    "freshly published stable artifact tree is incomplete: {}",
                    final_root.display(),
                )
            })?;
        touch_closure_access(&materialized_closure);
        Ok(StableArtifactTree {
            root: final_root,
            identity,
            cache_hit,
            _closure: materialized_closure,
            _validation: validation,
        })
    }
}

#[derive(Clone, Copy)]
struct LifecycleFilesystemSpace {
    capacity: u64,
    available: u64,
}

struct LifecycleRecordCandidate {
    namespace: PathBuf,
    path: PathBuf,
    identity: u64,
    record: Option<ArtifactTreeRecord>,
    last_used: SystemTime,
}

fn lifecycle_directory(root: &Path) -> PathBuf {
    root.join(LIFECYCLE_DIR)
}

fn lifecycle_gate_path(root: &Path) -> PathBuf {
    lifecycle_directory(root).join(LIFECYCLE_GATE)
}

fn lifecycle_closure_lock_path(root: &Path, identity: u64) -> PathBuf {
    lifecycle_directory(root)
        .join(LIFECYCLE_CLOSURE_LOCK_DIR)
        .join(format!("{identity:016x}.lock"))
}

fn lifecycle_access_path(root: &Path, identity: u64) -> PathBuf {
    lifecycle_directory(root)
        .join(LIFECYCLE_ACCESS_DIR)
        .join(format!("{identity:016x}.stamp"))
}

fn ensure_lifecycle_dirs(root: &Path) -> Result<()> {
    for directory in [
        root.to_path_buf(),
        lifecycle_directory(root),
        lifecycle_directory(root).join(LIFECYCLE_CLOSURE_LOCK_DIR),
        lifecycle_directory(root).join(LIFECYCLE_ACCESS_DIR),
        lifecycle_directory(root).join(LIFECYCLE_RESERVATION_DIR),
    ] {
        std::fs::create_dir_all(&directory).with_context(|| {
            format!(
                "create artifact lifecycle directory {}",
                directory.display()
            )
        })?;
    }
    Ok(())
}

fn lifecycle_filesystem_space(root: &Path) -> Result<LifecycleFilesystemSpace> {
    let stat = rustix::fs::statvfs(root)
        .with_context(|| format!("stat artifact-cache filesystem {}", root.display()))?;
    let block_size = stat.f_frsize.max(1);
    Ok(LifecycleFilesystemSpace {
        capacity: stat.f_blocks.saturating_mul(block_size),
        available: stat.f_bavail.saturating_mul(block_size),
    })
}

fn cache_entry_is_older_than(metadata: &std::fs::Metadata, now: SystemTime, age: Duration) -> bool {
    metadata
        .modified()
        .ok()
        .and_then(|modified| now.duration_since(modified).ok())
        .is_some_and(|elapsed| elapsed >= age)
}

fn touch_closure_access(lease: &ArtifactClosureLease) {
    let path = lifecycle_access_path(&lease.lifecycle_root, lease.identity);
    if path.metadata().ok().is_some_and(|metadata| {
        !cache_entry_is_older_than(&metadata, SystemTime::now(), LIFECYCLE_ACCESS_INTERVAL)
    }) {
        return;
    }
    if let Err(error) = std::fs::write(&path, format!("{:016x}\n", lease.identity)) {
        tracing::debug!(
            path = %path.display(),
            error = %error,
            "could not update reconstructible artifact closure access stamp",
        );
    }
}

fn parse_identity_filename(name: &std::ffi::OsStr, suffix: &str) -> Option<u64> {
    let name = name.to_str()?;
    let identity = name.strip_suffix(suffix)?;
    (identity.len() == 16 && identity.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| u64::from_str_radix(identity, 16).ok())
        .flatten()
}

fn read_record_for_lifecycle(path: &Path, identity: u64) -> Result<ArtifactTreeRecord> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("stat lifecycle artifact record {}", path.display()))?;
    anyhow::ensure!(
        metadata.is_file() && metadata.len() <= RECORD_MAX_BYTES,
        "invalid lifecycle artifact record size/type: {}",
        path.display(),
    );
    let bytes = std::fs::read(path)
        .with_context(|| format!("read lifecycle artifact record {}", path.display()))?;
    let record: ArtifactTreeRecord = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse lifecycle artifact record {}", path.display()))?;
    validate_record(&record, identity)?;
    Ok(record)
}

fn record_build_space_bytes(record: &ArtifactTreeRecord) -> u64 {
    record.entries.iter().fold(0u64, |sum, entry| {
        sum.saturating_add(match entry {
            RecordEntry::File { len, .. } => *len,
            RecordEntry::Directory { .. } | RecordEntry::Symlink { .. } => 0,
        })
    })
}

fn scan_build_space_high_water(namespace: &Path) -> Result<Option<u64>> {
    let records = namespace.join(RECORD_DIR);
    let entries = match std::fs::read_dir(&records) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("scan artifact build history {}", records.display()));
        }
    };
    let mut high_water = None::<u64>;
    for entry in entries {
        let entry = entry.context("read artifact build-history record entry")?;
        let Some(identity) = parse_identity_filename(&entry.file_name(), ".json") else {
            continue;
        };
        let record = match read_record_for_lifecycle(&entry.path(), identity) {
            Ok(record) => record,
            Err(error) => {
                tracing::debug!(
                    path = %entry.path().display(),
                    error = %error,
                    "ignoring reconstructible invalid artifact build-history record",
                );
                continue;
            }
        };
        let bytes = record_build_space_bytes(&record);
        high_water = Some(high_water.unwrap_or_default().max(bytes));
    }
    Ok(high_water)
}

fn read_build_space_high_water(namespace: &Path) -> Option<u64> {
    let path = namespace.join(BUILD_SPACE_HIGH_WATER);
    match std::fs::read_to_string(&path) {
        Ok(encoded) => {
            let mut fields = encoded.split_whitespace();
            let value = fields
                .next()
                .and_then(|value| u64::from_str_radix(value, 16).ok());
            let check = fields
                .next()
                .and_then(|value| u64::from_str_radix(value, 16).ok());
            if let (Some(value), Some(check), None) = (value, check, fields.next())
                && check == value ^ BUILD_SPACE_HIGH_WATER_MAGIC
            {
                Some(value)
            } else {
                tracing::debug!(
                    path = %path.display(),
                    "ignoring reconstructible invalid artifact build-space history",
                );
                None
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            tracing::debug!(
                path = %path.display(),
                error = %error,
                "ignoring unreadable reconstructible artifact build-space history",
            );
            None
        }
    }
}

fn write_build_space_high_water(namespace: &Path, high_water: u64) -> Result<()> {
    let path = namespace.join(BUILD_SPACE_HIGH_WATER);
    let mut temporary = tempfile::Builder::new()
        .prefix(".build-space-high-water-")
        .tempfile_in(namespace)
        .with_context(|| {
            format!(
                "create artifact build-space history temp in {}",
                namespace.display()
            )
        })?;
    writeln!(
        temporary.as_file_mut(),
        "{high_water:016x} {:016x}",
        high_water ^ BUILD_SPACE_HIGH_WATER_MAGIC,
    )
    .context("write artifact build-space history")?;
    temporary
        .as_file_mut()
        .flush()
        .context("flush artifact build-space history")?;
    temporary
        .as_file()
        .set_permissions(std::fs::Permissions::from_mode(0o444))
        .context("make artifact build-space history immutable")?;
    temporary
        .persist(&path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish artifact build-space history {}", path.display()))?;
    Ok(())
}

fn update_build_space_high_water(namespace: &Path, record: &ArtifactTreeRecord) -> Result<()> {
    let _lock = crate::flock::block_flock(
        namespace.join(BUILD_SPACE_HIGH_WATER_LOCK),
        crate::flock::FlockMode::Exclusive,
    )?;
    let observed = read_build_space_high_water(namespace);
    let prior = match observed {
        Some(high_water) => high_water,
        None => scan_build_space_high_water(namespace)?.unwrap_or_default(),
    };
    let high_water = prior.max(record_build_space_bytes(record));
    if observed != Some(high_water) {
        write_build_space_high_water(namespace, high_water)?;
    }
    Ok(())
}

fn historical_build_space_reservation(namespace: &Path) -> Result<u64> {
    if !namespace.is_dir() {
        return Ok(LIFECYCLE_BUILD_RESERVATION_MAX);
    }
    let _lock = crate::flock::block_flock(
        namespace.join(BUILD_SPACE_HIGH_WATER_LOCK),
        crate::flock::FlockMode::Exclusive,
    )?;
    let Some(high_water) = read_build_space_high_water(namespace) else {
        // Seed upgrades and recover a manually damaged sidecar from every
        // valid retained record. This caller still takes the conservative
        // ceiling: retained records cannot prove that a larger observation
        // was not deleted along with the sidecar. The next caller can use the
        // repaired high-water value in O(1).
        if let Some(high_water) = scan_build_space_high_water(namespace)? {
            write_build_space_high_water(namespace, high_water)?;
        }
        return Ok(LIFECYCLE_BUILD_RESERVATION_MAX);
    };

    // Output history predicts a producer's demand much more closely than the
    // filesystem capacity. In particular, capacity/8 made every producer on
    // a 1 TiB runner advertise 64 GiB, so a small storm could evict otherwise
    // reusable closures before building sub-GiB source or scheduler trees.
    // Keep bounded headroom for transient Cargo outputs.
    let headroom = high_water / LIFECYCLE_BUILD_RESERVATION_HEADROOM_DIVISOR;
    Ok(high_water.saturating_add(headroom).clamp(
        LIFECYCLE_BUILD_RESERVATION_MIN,
        LIFECYCLE_BUILD_RESERVATION_MAX,
    ))
}

fn scan_lifecycle_records(root: &Path) -> Result<Vec<LifecycleRecordCandidate>> {
    let mut candidates = Vec::new();
    for namespace in std::fs::read_dir(root)
        .with_context(|| format!("scan artifact cache namespaces {}", root.display()))?
    {
        let namespace = namespace.context("read artifact cache namespace")?;
        let namespace_path = namespace.path();
        if !namespace
            .file_type()
            .with_context(|| format!("inspect cache namespace {}", namespace_path.display()))?
            .is_dir()
        {
            continue;
        }
        let records = namespace_path.join(RECORD_DIR);
        if !records.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&records)
            .with_context(|| format!("scan artifact records {}", records.display()))?
        {
            let entry = entry.context("read artifact record entry")?;
            let Some(identity) = parse_identity_filename(&entry.file_name(), ".json") else {
                continue;
            };
            let path = entry.path();
            let record = match read_record_for_lifecycle(&path, identity) {
                Ok(record) => Some(record),
                Err(error) => {
                    tracing::debug!(
                        path = %path.display(),
                        error = %error,
                        "artifact lifecycle found a reconstructible invalid record",
                    );
                    None
                }
            };
            let last_used = lifecycle_access_path(root, identity)
                .metadata()
                .or_else(|_| entry.metadata())
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            candidates.push(LifecycleRecordCandidate {
                namespace: namespace_path.clone(),
                path,
                identity,
                record,
                last_used,
            });
        }
    }
    Ok(candidates)
}

fn open_lifecycle_election_lock_dirs(
    root: &Path,
) -> Result<BTreeMap<PathBuf, std::os::fd::OwnedFd>> {
    use std::os::fd::AsRawFd as _;

    let root_fd = rustix::fs::open(
        root,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| format!("open artifact lifecycle root {}", root.display()))?;
    let proc_path = PathBuf::from(format!("/proc/self/fd/{}", root_fd.as_raw_fd()));
    let mut lock_dirs = BTreeMap::new();
    for entry in std::fs::read_dir(&proc_path)
        .with_context(|| format!("scan pinned artifact lifecycle root {}", root.display()))?
    {
        let entry = entry.context("read pinned artifact lifecycle namespace")?;
        let name = entry.file_name();
        let namespace = match rustix::fs::openat(
            &root_fd,
            &name,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        ) {
            Ok(namespace) => namespace,
            Err(error)
                if error == rustix::io::Errno::NOENT
                    || error == rustix::io::Errno::NOTDIR
                    || error == rustix::io::Errno::LOOP =>
            {
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "open artifact cache namespace {}",
                        root.join(&name).display()
                    )
                });
            }
        };
        let lock_dir = match rustix::fs::openat(
            &namespace,
            LOCK_DIR,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        ) {
            Ok(lock_dir) => lock_dir,
            Err(error)
                if error == rustix::io::Errno::NOENT
                    || error == rustix::io::Errno::NOTDIR
                    || error == rustix::io::Errno::LOOP =>
            {
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "open artifact election directory {}",
                        root.join(&name).join(LOCK_DIR).display()
                    )
                });
            }
        };
        lock_dirs.insert(root.join(name), lock_dir);
    }
    Ok(lock_dirs)
}

fn pinned_directory_entry_names(
    directory: &std::os::fd::OwnedFd,
    description: &str,
) -> Result<Vec<OsString>> {
    use std::os::fd::AsRawFd as _;

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

fn open_election_lock_at(
    lock_dir: &std::os::fd::OwnedFd,
    identity: u64,
    create: bool,
) -> Result<Option<std::os::fd::OwnedFd>> {
    let name = format!("{identity:016x}.lock");
    let mut flags =
        rustix::fs::OFlags::RDWR | rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::CLOEXEC;
    if create {
        flags |= rustix::fs::OFlags::CREATE;
    }
    match rustix::fs::openat(
        lock_dir,
        name.as_str(),
        flags,
        rustix::fs::Mode::from_raw_mode(0o666),
    ) {
        Ok(lock) => Ok(Some(lock)),
        Err(error) if error == rustix::io::Errno::NOENT || error == rustix::io::Errno::LOOP => {
            Ok(None)
        }
        Err(error) => Err(error).with_context(|| format!("open artifact election lock {name}")),
    }
}

fn open_namespace_gate_at(lock_dir: &std::os::fd::OwnedFd) -> Result<Option<std::os::fd::OwnedFd>> {
    match rustix::fs::openat(
        lock_dir,
        NAMESPACE_GATE,
        rustix::fs::OFlags::RDWR
            | rustix::fs::OFlags::CREATE
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::from_raw_mode(0o666),
    ) {
        Ok(gate) => Ok(Some(gate)),
        Err(error) if error == rustix::io::Errno::NOENT || error == rustix::io::Errno::LOOP => {
            Ok(None)
        }
        Err(error) => Err(error).context("open artifact election namespace gate"),
    }
}

fn unlink_owned_election_lock(
    lock_dir: &std::os::fd::OwnedFd,
    identity: u64,
    lock: &std::os::fd::OwnedFd,
) -> Result<()> {
    let name = format!("{identity:016x}.lock");
    let opened = rustix::fs::fstat(lock).context("stat owned artifact election lock")?;
    let current = match rustix::fs::statat(
        lock_dir,
        name.as_str(),
        rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
    ) {
        Ok(current) => current,
        Err(error) if error == rustix::io::Errno::NOENT => return Ok(()),
        Err(error) => return Err(error).context("restat owned artifact election lock"),
    };
    if same_stable_tree_inode(&opened, &current) {
        rustix::fs::unlinkat(lock_dir, name.as_str(), rustix::fs::AtFlags::empty())
            .context("unlink owned artifact election lock")?;
    }
    Ok(())
}

fn active_build_reservations(root: &Path) -> Result<(u64, BTreeSet<u64>)> {
    let directory = lifecycle_directory(root).join(LIFECYCLE_RESERVATION_DIR);
    let mut total = 0u64;
    let mut identities = BTreeSet::new();
    for entry in std::fs::read_dir(&directory)
        .with_context(|| format!("scan artifact build reservations {}", directory.display()))?
    {
        let entry = entry.context("read artifact build reservation")?;
        if entry.path().extension().and_then(|value| value.to_str()) != Some("reserve") {
            continue;
        }
        // Open the exact observed inode without creating it. NamedTempFile
        // normally unlinks reservations on Drop, so read_dir -> open races
        // are expected and must be benign. Creating through `try_flock`
        // would resurrect a phantom zero-byte reservation instead.
        let file = match rustix::fs::open(
            entry.path(),
            rustix::fs::OFlags::RDWR | rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        ) {
            Ok(file) => std::fs::File::from(file),
            Err(error) if error == rustix::io::Errno::NOENT || error == rustix::io::Errno::LOOP => {
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("open artifact build reservation {}", entry.path().display())
                });
            }
        };
        match super::content::flock_retry(
            &file,
            rustix::fs::FlockOperation::NonBlockingLockExclusive,
        ) {
            Ok(()) => {
                let opened = file.metadata().with_context(|| {
                    format!("stat opened build reservation {}", entry.path().display())
                })?;
                let current = match std::fs::symlink_metadata(entry.path()) {
                    Ok(metadata) => metadata,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(error) => return Err(error).context("restat build reservation"),
                };
                if !current.file_type().is_symlink()
                    && opened.dev() == current.dev()
                    && opened.ino() == current.ino()
                {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                let mut value = String::new();
                let mut reader = &file;
                reader.read_to_string(&mut value).with_context(|| {
                    format!("read build reservation {}", entry.path().display())
                })?;
                let bytes = value.trim().parse::<u64>().unwrap_or(0);
                total = total.saturating_add(bytes);
                if let Some(identity) = entry
                    .file_name()
                    .to_str()
                    .and_then(|name| name.get(..16))
                    .filter(|identity| identity.bytes().all(|byte| byte.is_ascii_hexdigit()))
                    .and_then(|identity| u64::from_str_radix(identity, 16).ok())
                {
                    identities.insert(identity);
                }
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("try-lock build reservation {}", entry.path().display())
                });
            }
        }
    }
    Ok((total, identities))
}

fn collect_artifact_cache(
    root: &Path,
    protected: Option<u64>,
    protected_namespace: Option<&Path>,
    forced: bool,
) -> Result<()> {
    #[cfg(test)]
    update_stable_root_test_counters(|counters| counters.lifecycle_collections += 1);
    ensure_lifecycle_dirs(root)?;
    let gate_path = lifecycle_gate_path(root);
    let Some(global) = (if forced {
        Some(crate::flock::block_flock(
            &gate_path,
            crate::flock::FlockMode::Exclusive,
        )?)
    } else {
        crate::flock::try_flock(&gate_path, crate::flock::FlockMode::Exclusive)?
    }) else {
        return Ok(());
    };

    let now = SystemTime::now();
    let mut candidates = scan_lifecycle_records(root)?;
    let mut identities = candidates
        .iter()
        .map(|candidate| candidate.identity)
        .collect::<BTreeSet<_>>();
    for stable_parent in [
        root.join("stable-builds-v1"),
        root.join("stable-sources-v2"),
    ] {
        let Ok(entries) = std::fs::read_dir(&stable_parent) else {
            continue;
        };
        for entry in entries.flatten() {
            if let Some(identity) = parse_identity_filename(&entry.file_name(), "") {
                identities.insert(identity);
            }
        }
    }

    // Pin each namespace and election directory with O_NOFOLLOW descriptors.
    // GC must never discover a symlink namespace and later unlink an external
    // artifact-election pathname after a same-uid rename/symlink swap.
    let election_lock_dirs = open_lifecycle_election_lock_dirs(root)?;
    let mut election_keys = BTreeMap::new();
    for candidate in &candidates {
        election_keys.insert((candidate.namespace.clone(), candidate.identity), true);
    }
    for (namespace, lock_dir) in &election_lock_dirs {
        for name in pinned_directory_entry_names(lock_dir, "artifact election directory")? {
            let Some(identity) = parse_identity_filename(&name, ".lock") else {
                continue;
            };
            identities.insert(identity);
            election_keys
                .entry((namespace.clone(), identity))
                .or_insert(false);
        }
    }
    let mut election_locks = BTreeMap::new();
    let mut election_active = BTreeSet::new();
    let mut namespace_gates = BTreeMap::new();
    for (namespace, lock_dir) in &election_lock_dirs {
        if protected_namespace == Some(namespace.as_path()) {
            // The caller may already own an election lock in this namespace.
            // Waiters retain namespace SH until that election is released, so
            // taking namespace EX here would form election-EX -> namespace-EX
            // / namespace-SH -> election-SH ABBA. Keep the whole namespace
            // active for this collection pass and reclaim other namespaces.
            for (candidate_namespace, identity) in election_keys.keys() {
                if candidate_namespace == namespace {
                    election_active.insert(*identity);
                }
            }
            continue;
        }
        let Some(gate) = open_namespace_gate_at(lock_dir)? else {
            for (candidate_namespace, identity) in election_keys.keys() {
                if candidate_namespace == namespace {
                    election_active.insert(*identity);
                }
            }
            continue;
        };
        match super::content::flock_retry(
            &gate,
            rustix::fs::FlockOperation::NonBlockingLockExclusive,
        ) {
            Ok(()) => {
                namespace_gates.insert(namespace.clone(), gate);
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                // A producer may have opened its election inode but not yet
                // flocked it. Without namespace EX, unlinking that pathname
                // could create two live lock generations and two builders.
                for (candidate_namespace, identity) in election_keys.keys() {
                    if candidate_namespace == namespace {
                        election_active.insert(*identity);
                    }
                }
            }
            Err(error) => return Err(error).context("try-lock artifact namespace gate"),
        }
    }
    for ((namespace, identity), create) in election_keys {
        let (Some(lock_dir), Some(_namespace_gate)) = (
            election_lock_dirs.get(&namespace),
            namespace_gates.get(&namespace),
        ) else {
            // A record whose namespace/lock directory was concurrently
            // removed, replaced, or actively publishing cannot be safely
            // elected for deletion this pass.
            election_active.insert(identity);
            continue;
        };
        let Some(lock) = open_election_lock_at(lock_dir, identity, create)? else {
            election_active.insert(identity);
            continue;
        };
        match super::content::flock_retry(
            &lock,
            rustix::fs::FlockOperation::NonBlockingLockExclusive,
        ) {
            Ok(()) => {
                election_locks.insert((namespace, identity), lock);
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {
                election_active.insert(identity);
            }
            Err(error) => return Err(error).context("try-lock artifact producer election"),
        }
    }

    let mut inactive_locks = BTreeMap::new();
    let mut active = election_active;
    for identity in identities {
        if protected == Some(identity) {
            active.insert(identity);
            continue;
        }
        match crate::flock::try_flock(
            lifecycle_closure_lock_path(root, identity),
            crate::flock::FlockMode::Exclusive,
        )? {
            Some(lock) => {
                inactive_locks.insert(identity, lock);
            }
            None => {
                active.insert(identity);
            }
        }
    }

    let filesystem = lifecycle_filesystem_space(root)?;
    let (reservation_bytes, reserved_identities) = active_build_reservations(root)?;
    for identity in reserved_identities {
        active.insert(identity);
        inactive_locks.remove(&identity);
    }
    let free_reserve = LIFECYCLE_MIN_FREE_RESERVE
        .max(filesystem.capacity / LIFECYCLE_FREE_RESERVE_DIVISOR)
        .saturating_add(reservation_bytes);

    let mut all_objects = BTreeSet::new();
    for candidate in &candidates {
        if let Some(record) = &candidate.record {
            for entry in &record.entries {
                if let RecordEntry::File {
                    content_hash, len, ..
                } = entry
                {
                    all_objects.insert((*content_hash, *len));
                }
            }
        }
    }
    let all_bytes = all_objects
        .iter()
        .fold(0u64, |sum, (_, len)| sum.saturating_add(*len));
    let pressure_budget = all_bytes
        .saturating_add(filesystem.available)
        .saturating_sub(free_reserve);
    let cache_budget =
        (filesystem.capacity / LIFECYCLE_CACHE_CAPACITY_DIVISOR).min(pressure_budget);

    candidates.sort_by(|left, right| {
        right
            .last_used
            .cmp(&left.last_used)
            .then_with(|| left.identity.cmp(&right.identity))
            .then_with(|| left.namespace.cmp(&right.namespace))
    });
    let mut retained_objects = BTreeSet::new();
    let mut retained_records = BTreeSet::new();
    let mut retained_identities = active.clone();

    for (index, candidate) in candidates.iter().enumerate() {
        if !active.contains(&candidate.identity) {
            continue;
        }
        retained_records.insert(index);
        if let Some(record) = &candidate.record {
            for entry in &record.entries {
                if let RecordEntry::File {
                    content_hash, len, ..
                } = entry
                {
                    retained_objects.insert((*content_hash, *len));
                }
            }
        }
    }

    let mut retained_bytes = retained_objects
        .iter()
        .fold(0u64, |sum, (_, len)| sum.saturating_add(*len));
    for (index, candidate) in candidates.iter().enumerate() {
        if retained_records.contains(&index) {
            continue;
        }
        let Some(record) = &candidate.record else {
            continue;
        };
        let additions = record
            .entries
            .iter()
            .filter_map(|entry| match entry {
                RecordEntry::File {
                    content_hash, len, ..
                } if !retained_objects.contains(&(*content_hash, *len)) => {
                    Some((*content_hash, *len))
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let added_bytes = additions
            .iter()
            .fold(0u64, |sum, (_, len)| sum.saturating_add(*len));
        if retained_bytes.saturating_add(added_bytes) > cache_budget {
            continue;
        }
        retained_bytes = retained_bytes.saturating_add(added_bytes);
        retained_objects.extend(additions);
        retained_records.insert(index);
        retained_identities.insert(candidate.identity);
    }

    for (index, candidate) in candidates.iter().enumerate() {
        if retained_records.contains(&index) || active.contains(&candidate.identity) {
            continue;
        }
        remove_record_visibility(&candidate.path, candidate.identity)?;
    }

    for stable_parent in [
        root.join("stable-builds-v1"),
        root.join("stable-sources-v2"),
    ] {
        remove_unretained_stable_roots(&stable_parent, &retained_identities, &active, now)?;
    }

    for identity in inactive_locks.keys().copied().collect::<Vec<_>>() {
        if retained_identities.contains(&identity) {
            continue;
        }
        let _ = std::fs::remove_file(lifecycle_access_path(root, identity));
        let _ = std::fs::remove_file(lifecycle_closure_lock_path(root, identity));
    }
    for ((namespace, identity), lock) in &election_locks {
        if retained_identities.contains(identity) || active.contains(identity) {
            continue;
        }
        if let Some(lock_dir) = election_lock_dirs.get(namespace) {
            unlink_owned_election_lock(lock_dir, *identity, lock)?;
        }
    }

    // Content publication may begin before an artifact lookup and hold
    // content SH while it later acquires lifecycle SH. Never wait for content
    // EX while retaining lifecycle/election/closure locks, or unrelated cold
    // producers can form content-SH -> lifecycle-SH / lifecycle-EX ->
    // content-EX ABBA. Global artifact references remain authoritative while
    // content EX excludes concurrent reference publication.
    drop(election_locks);
    drop(namespace_gates);
    drop(election_lock_dirs);
    drop(inactive_locks);
    drop(global);
    // Even a forced pre-build pass must not block here: the opaque producer
    // closure may already own content SH (ArtifactTreeSource publishes in
    // bounded descriptor windows before entering this cache). Waiting for
    // content EX would self-deadlock before the closure can be consumed and
    // release that lease. Record/stable-root reclamation above is still
    // synchronous. Lifecycle collection runs at cold-build reservation
    // boundaries, where new cache growth can actually require reclamation;
    // direct hits never turn the complete namespace scan into a lookup storm.
    let content_sweep_complete =
        super::content::collect_unreachable_content_objects(&retained_objects, now, false)?;
    if !content_sweep_complete {
        tracing::debug!(
            "content CAS collection was contended; deferring it until the next lifecycle interval"
        );
    }
    std::fs::write(
        lifecycle_directory(root).join(LIFECYCLE_GC_STAMP),
        format!(
            "{}\n",
            now.duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ),
    )
    .context("update artifact lifecycle GC stamp")?;
    Ok(())
}

fn remove_record_visibility(path: &Path, identity: u64) -> Result<()> {
    let parent = path.parent().context("artifact record has no parent")?;
    let sequence = STABLE_REBASE_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
    let trash = parent.join(format!(
        ".gc-{identity:016x}-{}-{sequence:016x}",
        std::process::id()
    ));
    match std::fs::rename(path, &trash) {
        Ok(()) => {
            std::fs::remove_file(&trash)
                .with_context(|| format!("remove evicted artifact record {}", trash.display()))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("hide evicted artifact record {}", path.display()));
        }
    }
    Ok(())
}

fn remove_unretained_stable_roots(
    stable_parent: &Path,
    retained: &BTreeSet<u64>,
    active: &BTreeSet<u64>,
    now: SystemTime,
) -> Result<()> {
    let entries = match std::fs::read_dir(stable_parent) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!("scan stable artifact roots {}", stable_parent.display())
            });
        }
    };
    for entry in entries {
        let entry = entry.context("read stable artifact root")?;
        let Some(identity) = parse_identity_filename(&entry.file_name(), "") else {
            continue;
        };
        if retained.contains(&identity) || active.contains(&identity) {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error).context("stat stable artifact root"),
        };
        if !cache_entry_is_older_than(&metadata, now, LIFECYCLE_ORPHAN_GRACE) {
            continue;
        }
        remove_stable_tree(&entry.path())?;
    }
    Ok(())
}

fn prepare_stable_cargo_build(root: &Path) -> Result<StableCargoBuild> {
    // `remove_stable_tree` also handles dangling symlinks, for which
    // `Path::exists()` deliberately returns false.
    remove_stable_tree(root)?;
    let target_directory = root.join("target");
    std::fs::create_dir_all(&target_directory).with_context(|| {
        format!(
            "create persistent stable Cargo target {}",
            target_directory.display()
        )
    })?;
    Ok(StableCargoBuild {
        root: root.to_path_buf(),
        target_directory,
    })
}

/// Whether a recorded path is a Cargo output the sealed stable root physically
/// owns (retains, rebases from the CAS, and validates in place).
///
/// Only `target/` — the final artifacts whose absolute pathnames may be embedded
/// (e.g. `CARGO_BIN_EXE_*`) — is sealed under the per-source stable root. The
/// Cargo build closure (`build/`: dependency intermediates and build-script
/// `OUT_DIR`s) is written to a persistent, non-source-keyed scratch directory
/// shared across digests, never under the stable root, so the distillation walk
/// must not expect a physical `build/` here. Any `build/**` record entries the
/// nextest producer captures are still materialized into the per-source private
/// tree from the CAS by `materialize`, which is what nextest's build-dir-remap
/// resolves at test-execution time — the stable root plays no part in that.
fn stable_cargo_output_path(path: &Path) -> bool {
    path.starts_with("target")
}

fn open_stable_cargo_directory_at(
    root: &std::os::fd::OwnedFd,
    relative: &Path,
) -> Result<std::os::fd::OwnedFd> {
    if relative.as_os_str().is_empty() {
        return rustix::io::fcntl_dupfd_cloexec(root, 0).with_context(|| {
            format!("duplicate stable Cargo root fd for {}", relative.display(),)
        });
    }
    checked_relative_path(relative)?;
    if !OPENAT2_UNAVAILABLE.load(AtomicOrdering::Relaxed) {
        match rustix::fs::openat2(
            root,
            relative,
            rustix::fs::OFlags::PATH
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
            rustix::fs::ResolveFlags::BENEATH
                | rustix::fs::ResolveFlags::NO_SYMLINKS
                | rustix::fs::ResolveFlags::NO_MAGICLINKS,
        ) {
            Ok(directory) => return Ok(directory),
            Err(error) if error == rustix::io::Errno::NOSYS => {
                // Old kernels and seccomp profiles can both report ENOSYS.
                // Remember it process-wide so a large sparse-tree walk does
                // not pay one rejected syscall per directory.
                OPENAT2_UNAVAILABLE.store(true, AtomicOrdering::Relaxed);
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "open stable Cargo directory beneath its root without following links {}",
                        relative.display(),
                    )
                });
            }
        }
    }
    open_stable_cargo_directory_at_components(root, relative)
}

/// `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` expressed with portable
/// `openat` operations for hosts where the syscall is unavailable or filtered.
///
/// Every path component is normalized before this function is entered and is
/// opened relative to the descriptor for its exact parent with O_NOFOLLOW and
/// O_DIRECTORY. Renames cannot redirect an already-opened ancestor, `..` can
/// never escape the root, and symlinks (including procfs magic links) cannot be
/// traversed. This is a security-preserving syscall fallback, not a pathname
/// or permissive fallback.
fn open_stable_cargo_directory_at_components(
    root: &std::os::fd::OwnedFd,
    relative: &Path,
) -> Result<std::os::fd::OwnedFd> {
    if relative.as_os_str().is_empty() {
        return rustix::io::fcntl_dupfd_cloexec(root, 0)
            .context("duplicate stable Cargo root fd for component walk");
    }
    checked_relative_path(relative)?;
    let mut current = rustix::io::fcntl_dupfd_cloexec(root, 0)
        .context("duplicate stable Cargo root fd for component walk")?;
    for component in relative.components() {
        let Component::Normal(component) = component else {
            unreachable!("checked stable Cargo path contained a non-normal component")
        };
        current = rustix::fs::openat(
            &current,
            component,
            rustix::fs::OFlags::PATH
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )
        .with_context(|| {
            format!(
                "open stable Cargo directory component {} without following links while resolving {}",
                component.to_string_lossy(),
                relative.display(),
            )
        })?;
    }
    Ok(current)
}

fn unlink_stable_cargo_path(
    root: &std::os::fd::OwnedFd,
    relative: &Path,
    directory: bool,
) -> Result<()> {
    let name = relative
        .file_name()
        .with_context(|| format!("stable Cargo path has no file name: {}", relative.display()))?;
    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let parent = open_stable_cargo_directory_at(root, parent)?;
    let flags = if directory {
        rustix::fs::AtFlags::REMOVEDIR
    } else {
        rustix::fs::AtFlags::empty()
    };
    rustix::fs::unlinkat(&parent, name, flags)
        .with_context(|| format!("remove unrecorded stable Cargo path {}", relative.display()))
}

fn validate_stable_cargo_output_entry(
    root: &std::os::fd::OwnedFd,
    entry: &RecordEntry,
) -> Result<()> {
    let path = entry.path()?;
    if !stable_cargo_output_path(&path) {
        return Ok(());
    }
    validate_stable_record_entry(root, entry, &path)
}

fn validate_stable_record_entry(
    root: &std::os::fd::OwnedFd,
    entry: &RecordEntry,
    path: &Path,
) -> Result<()> {
    let name = path
        .file_name()
        .with_context(|| format!("stable output has no file name: {}", path.display()))?;
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let parent = open_stable_cargo_directory_at(root, parent)?;
    let stat = rustix::fs::statat(&parent, name, rustix::fs::AtFlags::SYMLINK_NOFOLLOW)
        .with_context(|| format!("stat recorded stable Cargo output {}", path.display()))?;
    let file_type = rustix::fs::FileType::from_raw_mode(stat.st_mode);
    match entry {
        RecordEntry::Directory { .. } => anyhow::ensure!(
            file_type.is_dir(),
            "recorded stable Cargo directory changed type: {}",
            path.display(),
        ),
        RecordEntry::File { len, .. } => {
            anyhow::ensure!(
                file_type.is_file(),
                "recorded stable Cargo file changed type: {}",
                path.display(),
            );
            anyhow::ensure!(
                u64::try_from(stat.st_size).ok() == Some(*len),
                "recorded stable Cargo file changed length: {}",
                path.display(),
            );
        }
        RecordEntry::Symlink { target, .. } => {
            anyhow::ensure!(
                file_type.is_symlink(),
                "recorded stable Cargo symlink changed type: {}",
                path.display(),
            );
            let actual = rustix::fs::readlinkat(&parent, name, Vec::new()).with_context(|| {
                format!("read recorded stable Cargo symlink {}", path.display())
            })?;
            anyhow::ensure!(
                actual.as_bytes() == target,
                "recorded stable Cargo symlink changed target: {}",
                path.display(),
            );
        }
    }
    Ok(())
}

/// Reduce a completed Cargo target to the exact runtime closure before its
/// deterministic pathname becomes a reusable cache anchor.
///
/// The content CAS already owns strict FICLONE snapshots of every recorded
/// regular file when this runs. Keeping the original recorded output paths
/// preserves embedded `OUT_DIR`/`CARGO_BIN_EXE_*` strings, while removing
/// incremental state and unrelated dependency products avoids retaining and
/// recursively sealing the producer's complete target directory. All removal
/// is relative to a no-follow-opened root and no-follow-opened parent fds.
fn distill_stable_cargo_build(
    build: &StableCargoBuild,
    record: &ArtifactTreeRecord,
    content_namespace: &super::content::ContentNamespaceLease,
) -> Result<()> {
    let root = rustix::fs::open(
        &build.root,
        rustix::fs::OFlags::PATH
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| {
        format!(
            "open stable Cargo root without following links {}",
            build.root.display(),
        )
    })?;
    let root_identity = rustix::fs::fstat(&root)
        .with_context(|| format!("stat stable Cargo root fd {}", build.root.display()))?;

    let mut retained = BTreeSet::from([PathBuf::from("target")]);
    let mut required_directories = BTreeSet::from([PathBuf::from("target")]);
    for entry in &record.entries {
        let path = entry.path()?;
        if !stable_cargo_output_path(&path) {
            continue;
        }
        retained.insert(path.clone());
        if matches!(entry, RecordEntry::Directory { .. }) {
            required_directories.insert(path.clone());
        }
        let mut parent = path.parent();
        while let Some(directory) = parent.filter(|path| !path.as_os_str().is_empty()) {
            retained.insert(directory.to_path_buf());
            required_directories.insert(directory.to_path_buf());
            parent = directory.parent();
        }
    }

    for entry in walkdir::WalkDir::new(&build.root)
        .follow_links(false)
        .contents_first(true)
        .sort_by_file_name()
    {
        let entry = entry.with_context(|| {
            format!(
                "walk stable Cargo output for sparse distillation {}",
                build.root.display(),
            )
        })?;
        let relative = entry.path().strip_prefix(&build.root).with_context(|| {
            format!(
                "stable Cargo output escaped its root: {}",
                entry.path().display(),
            )
        })?;
        if relative.as_os_str().is_empty() || retained.contains(relative) {
            continue;
        }
        let relative = checked_relative_path(relative)?;
        unlink_stable_cargo_path(&root, &relative, entry.file_type().is_dir())?;
    }

    for directory in required_directories {
        drop(open_stable_cargo_directory_at(&root, &directory)?);
    }
    for entry in &record.entries {
        rebase_stable_cargo_output_file(&root, entry, content_namespace)?;
    }
    for entry in &record.entries {
        validate_stable_cargo_output_entry(&root, entry)?;
    }

    let recheck = rustix::fs::open(
        &build.root,
        rustix::fs::OFlags::PATH
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| {
        format!(
            "reopen distilled stable Cargo root {}",
            build.root.display()
        )
    })?;
    let recheck_identity = rustix::fs::fstat(&recheck).with_context(|| {
        format!(
            "restat distilled stable Cargo root {}",
            build.root.display()
        )
    })?;
    anyhow::ensure!(
        root_identity.st_dev == recheck_identity.st_dev
            && root_identity.st_ino == recheck_identity.st_ino,
        "stable Cargo root pathname changed during distillation: {}",
        build.root.display(),
    );
    Ok(())
}

fn rebase_stable_cargo_output_file(
    root: &std::os::fd::OwnedFd,
    entry: &RecordEntry,
    content_namespace: &super::content::ContentNamespaceLease,
) -> Result<()> {
    let RecordEntry::File {
        mode,
        content_hash,
        len,
        ..
    } = entry
    else {
        return Ok(());
    };
    let path = entry.path()?;
    if !stable_cargo_output_path(&path) {
        return Ok(());
    }
    validate_stable_cargo_output_entry(root, entry)?;
    let (object, object_path) = content_namespace
        .open_object(*content_hash, *len)?
        .with_context(|| {
            format!("stable Cargo output object missing for {content_hash:016x}/{len}")
        })?;
    let name = path
        .file_name()
        .with_context(|| format!("stable Cargo output has no file name: {}", path.display()))?;
    let parent_path = path.parent().unwrap_or_else(|| Path::new(""));
    let parent = open_stable_cargo_directory_at(root, parent_path)?;

    let sequence = STABLE_REBASE_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
    let temporary_name = OsString::from(format!(
        ".ktstr-cas-rebase-{content_hash:016x}-{}-{sequence:016x}",
        std::process::id(),
    ));
    let temporary = rustix::fs::openat(
        &parent,
        &temporary_name,
        rustix::fs::OFlags::RDWR
            | rustix::fs::OFlags::CREATE
            | rustix::fs::OFlags::EXCL
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::from_raw_mode(0o600),
    )
    .with_context(|| {
        format!(
            "create stable Cargo CAS-rebase temporary beside {}",
            path.display()
        )
    })?;
    let temporary = std::fs::File::from(temporary);
    let result: Result<()> = (|| {
        crate::reflink::ficlone(&temporary, &object).with_context(|| {
            format!(
                "FICLONE stable Cargo output from canonical CAS {} -> {}",
                object_path.display(),
                path.display(),
            )
        })?;
        let actual_len = temporary
            .metadata()
            .with_context(|| format!("stat rebased stable Cargo output {}", path.display()))?
            .len();
        anyhow::ensure!(
            actual_len == *len,
            "rebased stable Cargo output {} has length {actual_len}, expected {len}",
            path.display(),
        );
        temporary
            .set_permissions(std::fs::Permissions::from_mode(*mode))
            .with_context(|| format!("set rebased stable Cargo mode {}", path.display()))?;
        rustix::fs::renameat(&parent, &temporary_name, &parent, name).with_context(|| {
            format!(
                "atomically replace stable Cargo output with CAS reflink {}",
                path.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = rustix::fs::unlinkat(&parent, &temporary_name, rustix::fs::AtFlags::empty());
    }
    result
}

fn seal_stable_cargo_build(build: &StableCargoBuild, identity: u64) -> Result<()> {
    let root = rustix::fs::open(
        &build.root,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| {
        format!(
            "open stable Cargo root for descriptor-relative sealing {}",
            build.root.display()
        )
    })?;
    let root_identity = rustix::fs::fstat(&root).context("stat stable Cargo sealing root")?;
    seal_open_stable_cargo_directory(&root, &build.root)?;

    let recheck = rustix::fs::open(
        &build.root,
        rustix::fs::OFlags::PATH
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| format!("reopen sealed stable Cargo root {}", build.root.display()))?;
    let recheck_identity = rustix::fs::fstat(&recheck).context("restat stable Cargo root")?;
    anyhow::ensure!(
        same_stable_tree_inode(&root_identity, &recheck_identity),
        "stable Cargo root changed during sealing: {}",
        build.root.display()
    );

    // Completeness becomes visible last. A killed or failed sealer therefore
    // leaves an ordinary reconstructible miss, never a marker-backed partial
    // tree. The generic seal is shared with stable source snapshots.
    publish_stable_root_marker(&build.root, StableRootDomain::CargoBuild, identity)
}

fn seal_open_stable_cargo_directory(
    directory: &std::os::fd::OwnedFd,
    display_path: &Path,
) -> Result<()> {
    for name in pinned_directory_entry_names(directory, "stable Cargo sealing directory")? {
        let child_display = display_path.join(&name);
        let before = rustix::fs::statat(directory, &name, rustix::fs::AtFlags::SYMLINK_NOFOLLOW)
            .with_context(|| format!("stat stable Cargo output {}", child_display.display()))?;
        let file_type = rustix::fs::FileType::from_raw_mode(before.st_mode);
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            let child = rustix::fs::openat(
                directory,
                &name,
                rustix::fs::OFlags::RDONLY
                    | rustix::fs::OFlags::DIRECTORY
                    | rustix::fs::OFlags::NOFOLLOW
                    | rustix::fs::OFlags::CLOEXEC,
                rustix::fs::Mode::empty(),
            )
            .with_context(|| format!("open stable Cargo directory {}", child_display.display()))?;
            let opened = rustix::fs::fstat(&child).with_context(|| {
                format!(
                    "stat opened stable Cargo directory {}",
                    child_display.display()
                )
            })?;
            anyhow::ensure!(
                same_stable_tree_inode(&before, &opened),
                "stable Cargo directory changed while sealing: {}",
                child_display.display()
            );
            seal_open_stable_cargo_directory(&child, &child_display)?;
            continue;
        }
        anyhow::ensure!(
            file_type.is_file(),
            "stable Cargo output has unsupported file type: {}",
            child_display.display()
        );
        let child = rustix::fs::openat(
            directory,
            &name,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::NONBLOCK
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )
        .with_context(|| format!("open stable Cargo file {}", child_display.display()))?;
        let opened = rustix::fs::fstat(&child)
            .with_context(|| format!("stat stable Cargo file {}", child_display.display()))?;
        anyhow::ensure!(
            same_stable_tree_inode(&before, &opened)
                && rustix::fs::FileType::from_raw_mode(opened.st_mode).is_file(),
            "stable Cargo file changed while sealing: {}",
            child_display.display()
        );
        let mode = opened.st_mode & 0o7777;
        rustix::fs::fchmod(&child, rustix::fs::Mode::from_raw_mode(mode & !0o222))
            .with_context(|| format!("seal stable Cargo file {}", child_display.display()))?;
    }
    let metadata = rustix::fs::fstat(directory)
        .with_context(|| format!("stat stable Cargo directory {}", display_path.display()))?;
    let mode = metadata.st_mode & 0o7777;
    rustix::fs::fchmod(directory, rustix::fs::Mode::from_raw_mode(mode | 0o700)).with_context(
        || {
            format!(
                "make stable Cargo directory owner-removable {}",
                display_path.display()
            )
        },
    )?;
    Ok(())
}

fn stable_cargo_build_is_complete(
    root: &Path,
    identity: u64,
    record: Option<&ArtifactTreeRecord>,
) -> Result<bool> {
    match record {
        Some(_) => stable_root_marker_matches(root, StableRootDomain::CargoBuild, identity, true),
        None => stable_root_marker_matches(root, StableRootDomain::CargoBuild, identity, false),
    }
}

fn remove_stable_tree(root: &Path) -> Result<()> {
    let name = root
        .file_name()
        .with_context(|| format!("stable-tree root has no file name: {}", root.display()))?;
    let parent_path = root.parent().unwrap_or_else(|| Path::new("."));
    let parent = rustix::fs::open(
        parent_path,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| {
        format!(
            "open stable-tree parent without following links {}",
            parent_path.display()
        )
    })?;

    loop {
        let before = match rustix::fs::statat(&parent, name, rustix::fs::AtFlags::SYMLINK_NOFOLLOW)
        {
            Ok(stat) => stat,
            Err(error) if error == rustix::io::Errno::NOENT => return Ok(()),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("stat stale stable tree {}", root.display()));
            }
        };
        if !rustix::fs::FileType::from_raw_mode(before.st_mode).is_dir() {
            // Recheck the exact entry before unlinking. If it was replaced,
            // retry instead of deleting the replacement. `unlinkat` never
            // follows a symlink even if one is installed after this check.
            let current =
                match rustix::fs::statat(&parent, name, rustix::fs::AtFlags::SYMLINK_NOFOLLOW) {
                    Ok(stat) => stat,
                    Err(error) if error == rustix::io::Errno::NOENT => return Ok(()),
                    Err(error) => return Err(error).context("restat stale stable-tree entry"),
                };
            if !same_stable_tree_inode(&before, &current) {
                continue;
            }
            match rustix::fs::unlinkat(&parent, name, rustix::fs::AtFlags::empty()) {
                Ok(()) => return Ok(()),
                Err(error)
                    if error == rustix::io::Errno::NOENT || error == rustix::io::Errno::ISDIR =>
                {
                    continue;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("remove stale stable-tree entry {}", root.display())
                    });
                }
            }
        }

        let directory = match rustix::fs::openat(
            &parent,
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
                continue;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("open stale stable-tree root {}", root.display()));
            }
        };
        let opened = rustix::fs::fstat(&directory)
            .with_context(|| format!("stat opened stable-tree root {}", root.display()))?;
        if !same_stable_tree_inode(&before, &opened) {
            continue;
        }
        remove_open_stable_tree_directory(&parent, name, &directory, &opened, root)?;
        return Ok(());
    }
}

fn same_stable_tree_inode(left: &rustix::fs::Stat, right: &rustix::fs::Stat) -> bool {
    left.st_dev == right.st_dev && left.st_ino == right.st_ino
}

fn same_stable_tree_revision(left: &rustix::fs::Stat, right: &rustix::fs::Stat) -> bool {
    same_stable_tree_inode(left, right)
        && left.st_mode == right.st_mode
        && left.st_nlink == right.st_nlink
        && left.st_uid == right.st_uid
        && left.st_gid == right.st_gid
        && left.st_size == right.st_size
        && left.st_mtime == right.st_mtime
        && left.st_mtime_nsec == right.st_mtime_nsec
        && left.st_ctime == right.st_ctime
        && left.st_ctime_nsec == right.st_ctime_nsec
}

fn encode_stable_root_seal(seal: StableRootSeal) -> [u8; STABLE_ROOT_MARKER_LEN] {
    let mut bytes = [0_u8; STABLE_ROOT_MARKER_LEN];
    bytes[0..8].copy_from_slice(STABLE_ROOT_MARKER_MAGIC);
    bytes[8..12].copy_from_slice(&STABLE_ROOT_MARKER_SCHEMA.to_le_bytes());
    bytes[12] = seal.domain as u8;
    bytes[16..24].copy_from_slice(&seal.identity.to_le_bytes());
    bytes[24..32].copy_from_slice(&seal.root_dev.to_le_bytes());
    bytes[32..40].copy_from_slice(&seal.root_ino.to_le_bytes());
    bytes[40..48].copy_from_slice(&seal.entry_count.to_le_bytes());
    bytes[48..56].copy_from_slice(&seal.listing_hash.to_le_bytes());
    bytes
}

fn decode_stable_root_seal(bytes: &[u8]) -> Result<StableRootSeal> {
    anyhow::ensure!(
        bytes.len() == STABLE_ROOT_MARKER_LEN,
        "stable-root marker has length {}, expected {STABLE_ROOT_MARKER_LEN}",
        bytes.len(),
    );
    anyhow::ensure!(
        &bytes[0..8] == STABLE_ROOT_MARKER_MAGIC,
        "stable-root marker has unknown magic",
    );
    let schema = u32::from_le_bytes(bytes[8..12].try_into().expect("fixed schema field"));
    anyhow::ensure!(
        schema == STABLE_ROOT_MARKER_SCHEMA,
        "stable-root marker schema {schema} is not {STABLE_ROOT_MARKER_SCHEMA}",
    );
    anyhow::ensure!(
        bytes[13..16] == [0, 0, 0],
        "stable-root marker reserved bytes are nonzero",
    );
    let domain =
        StableRootDomain::from_tag(bytes[12]).context("stable-root marker has unknown domain")?;
    Ok(StableRootSeal {
        domain,
        identity: u64::from_le_bytes(bytes[16..24].try_into().expect("fixed identity field")),
        root_dev: u64::from_le_bytes(bytes[24..32].try_into().expect("fixed device field")),
        root_ino: u64::from_le_bytes(bytes[32..40].try_into().expect("fixed inode field")),
        entry_count: u64::from_le_bytes(bytes[40..48].try_into().expect("fixed entry-count field")),
        listing_hash: u64::from_le_bytes(
            bytes[48..56].try_into().expect("fixed listing-hash field"),
        ),
    })
}

struct OpenedStableRootMarker {
    bytes: [u8; STABLE_ROOT_MARKER_LEN],
    seal: StableRootSeal,
    stat: rustix::fs::Stat,
}

fn read_stable_root_marker_at(
    root: &std::os::fd::OwnedFd,
    domain: StableRootDomain,
) -> Result<OpenedStableRootMarker> {
    let marker_name = domain.marker_name();
    let marker = rustix::fs::openat(
        root,
        marker_name,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::NONBLOCK
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| format!("open stable-root marker {marker_name}"))?;
    let mut marker = std::fs::File::from(marker);
    let before = rustix::fs::fstat(&marker).context("stat opened stable-root marker")?;
    anyhow::ensure!(
        rustix::fs::FileType::from_raw_mode(before.st_mode).is_file(),
        "stable-root marker is not a regular file",
    );
    anyhow::ensure!(
        before.st_mode & 0o222 == 0,
        "stable-root marker is writable",
    );
    anyhow::ensure!(
        u64::try_from(before.st_size).ok() == Some(STABLE_ROOT_MARKER_LEN as u64),
        "stable-root marker has unexpected on-disk length {}",
        before.st_size,
    );
    let mut bytes = Vec::with_capacity(STABLE_ROOT_MARKER_LEN + 1);
    (&mut marker)
        .take((STABLE_ROOT_MARKER_LEN + 1) as u64)
        .read_to_end(&mut bytes)
        .context("read stable-root marker")?;
    let after = rustix::fs::fstat(&marker).context("restat opened stable-root marker")?;
    anyhow::ensure!(
        same_stable_tree_revision(&before, &after),
        "stable-root marker changed while reading",
    );
    let seal = decode_stable_root_seal(&bytes)?;
    let bytes: [u8; STABLE_ROOT_MARKER_LEN] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("stable-root marker length changed while decoding"))?;
    Ok(OpenedStableRootMarker {
        bytes,
        seal,
        stat: after,
    })
}

#[derive(Clone, Debug)]
struct RawStableRootEntry {
    name: OsString,
    file_type: rustix::fs::FileType,
    ino: u64,
}

fn read_raw_stable_root_entries(
    directory: &std::os::fd::OwnedFd,
) -> Result<Vec<RawStableRootEntry>> {
    // `getdents64` advances the open file description's directory offset.
    // Reopen `.` beneath the already-verified descriptor so every validation
    // pass starts at offset zero without sharing seek state with the caller.
    let listing_directory = rustix::fs::openat(
        directory,
        ".",
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .context("reopen stable-root directory for a fresh listing")?;
    let mut buffer = vec![MaybeUninit::uninit(); STABLE_ROOT_LISTING_BUFFER_LEN];
    let mut raw = rustix::fs::RawDir::new(&listing_directory, &mut buffer);
    let mut entries = Vec::new();
    while let Some(entry) = raw.next() {
        let entry = entry.context("read stable-root directory entry")?;
        let name = entry.file_name().to_bytes();
        if name == b"." || name == b".." {
            continue;
        }
        entries.push(RawStableRootEntry {
            name: OsString::from_vec(name.to_vec()),
            file_type: entry.file_type(),
            ino: entry.ino(),
        });
    }
    entries.sort_unstable_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    Ok(entries)
}

fn collect_stable_root_listing(
    directory: &std::os::fd::OwnedFd,
    relative_parent: &Path,
    marker_name: &str,
    entries: &mut Vec<StableRootListingEntry>,
) -> Result<()> {
    #[cfg(test)]
    update_stable_root_test_counters(|counters| counters.listing_directories += 1);

    let before = rustix::fs::fstat(directory).context("stat stable-root listing directory")?;
    anyhow::ensure!(
        rustix::fs::FileType::from_raw_mode(before.st_mode).is_dir(),
        "stable-root listing descriptor is not a directory",
    );
    for raw in read_raw_stable_root_entries(directory)? {
        if relative_parent.as_os_str().is_empty() && raw.name.as_bytes() == marker_name.as_bytes() {
            continue;
        }

        let relative = relative_parent.join(&raw.name);
        let mut file_type = raw.file_type;
        let mut ino = raw.ino;
        if file_type == rustix::fs::FileType::Unknown || ino == 0 {
            #[cfg(test)]
            update_stable_root_test_counters(|counters| {
                counters.listing_fallback_stats += 1;
            });
            let stat =
                rustix::fs::statat(directory, &raw.name, rustix::fs::AtFlags::SYMLINK_NOFOLLOW)
                    .with_context(|| {
                        format!("stat unknown stable-root entry {}", relative.display())
                    })?;
            file_type = rustix::fs::FileType::from_raw_mode(stat.st_mode);
            ino = stat.st_ino;
        }

        let (kind, symlink_target) = if file_type.is_dir() {
            (1, None)
        } else if file_type.is_file() {
            (2, None)
        } else if file_type.is_symlink() {
            let target = rustix::fs::readlinkat(directory, &raw.name, Vec::new())
                .with_context(|| format!("read stable-root symlink {}", relative.display()))?;
            (3, Some(target.as_bytes().to_vec()))
        } else {
            anyhow::bail!(
                "stable-root entry has unsupported file type: {}",
                relative.display(),
            );
        };
        anyhow::ensure!(
            ino != 0,
            "stable-root entry has no inode: {}",
            relative.display()
        );
        entries.push(StableRootListingEntry {
            path: relative.as_os_str().as_bytes().to_vec(),
            kind,
            ino,
            symlink_target,
        });
        #[cfg(test)]
        update_stable_root_test_counters(|counters| counters.listing_entries += 1);

        if file_type.is_dir() {
            let child = rustix::fs::openat(
                directory,
                &raw.name,
                rustix::fs::OFlags::RDONLY
                    | rustix::fs::OFlags::DIRECTORY
                    | rustix::fs::OFlags::NOFOLLOW
                    | rustix::fs::OFlags::CLOEXEC,
                rustix::fs::Mode::empty(),
            )
            .with_context(|| format!("open stable-root directory {}", relative.display()))?;
            let opened = rustix::fs::fstat(&child)
                .with_context(|| format!("stat stable-root directory {}", relative.display()))?;
            anyhow::ensure!(
                rustix::fs::FileType::from_raw_mode(opened.st_mode).is_dir()
                    && opened.st_ino == ino,
                "stable-root directory changed while opening: {}",
                relative.display(),
            );
            collect_stable_root_listing(&child, &relative, marker_name, entries)?;
        }
    }
    let after = rustix::fs::fstat(directory).context("restat stable-root listing directory")?;
    anyhow::ensure!(
        same_stable_tree_revision(&before, &after),
        "stable-root directory changed while enumerating {}",
        relative_parent.display(),
    );
    Ok(())
}

fn compute_stable_root_listing(
    root: &std::os::fd::OwnedFd,
    root_stat: &rustix::fs::Stat,
    domain: StableRootDomain,
    identity: u64,
) -> Result<StableRootListing> {
    let mut entries = Vec::new();
    collect_stable_root_listing(root, Path::new(""), domain.marker_name(), &mut entries)?;
    let after = rustix::fs::fstat(root).context("restat stable-root after listing")?;
    anyhow::ensure!(
        same_stable_tree_revision(root_stat, &after),
        "stable-root changed while computing its listing",
    );
    entries.sort_unstable_by(|left, right| left.path.cmp(&right.path));
    let entry_count = u64::try_from(entries.len()).context("stable-root entry count overflow")?;
    let mut hasher = fixed_hasher();
    hash_bytes(&mut hasher, b"ktstr-stable-root-listing-v2");
    hasher.write_u8(domain as u8);
    hasher.write_u64(identity);
    hasher.write_u64(root_stat.st_dev);
    hasher.write_u64(root_stat.st_ino);
    hasher.write_u64(entry_count);
    for entry in entries {
        hash_bytes(&mut hasher, &entry.path);
        hasher.write_u8(entry.kind);
        hasher.write_u64(entry.ino);
        if let Some(target) = entry.symlink_target {
            hash_bytes(&mut hasher, &target);
        }
    }
    Ok(StableRootListing {
        entry_count,
        hash: hasher.finish(),
    })
}

fn open_stable_root(root: &Path) -> Result<std::os::fd::OwnedFd> {
    rustix::fs::open(
        root,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| {
        format!(
            "open stable-root without following links {}",
            root.display()
        )
    })
}

fn validate_stable_root_stat(stat: &rustix::fs::Stat) -> Result<()> {
    anyhow::ensure!(
        rustix::fs::FileType::from_raw_mode(stat.st_mode).is_dir(),
        "stable-root is not a directory",
    );
    anyhow::ensure!(
        stat.st_mode & 0o300 == 0o300,
        "stable-root is not owner-writable and searchable",
    );
    Ok(())
}

fn recheck_stable_root_path(root: &Path, expected: &rustix::fs::Stat) -> Result<()> {
    let current = rustix::fs::open(
        root,
        rustix::fs::OFlags::PATH
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| format!("reopen stable-root path {}", root.display()))?;
    let current = rustix::fs::fstat(&current).context("restat stable-root path")?;
    anyhow::ensure!(
        same_stable_tree_revision(expected, &current),
        "stable-root path changed while validating {}",
        root.display(),
    );
    Ok(())
}

fn publish_stable_root_marker(
    root_path: &Path,
    domain: StableRootDomain,
    identity: u64,
) -> Result<()> {
    let root = open_stable_root(root_path)?;
    let root_stat = rustix::fs::fstat(&root).context("stat stable-root before publication")?;
    validate_stable_root_stat(&root_stat)?;
    let listing = compute_stable_root_listing(&root, &root_stat, domain, identity)?;
    recheck_stable_root_path(root_path, &root_stat)?;
    let bytes = encode_stable_root_seal(StableRootSeal {
        domain,
        identity,
        root_dev: root_stat.st_dev,
        root_ino: root_stat.st_ino,
        entry_count: listing.entry_count,
        listing_hash: listing.hash,
    });

    #[cfg(test)]
    run_stable_root_pre_publish_hook();

    let sequence = STABLE_REBASE_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
    let temporary_name = format!(
        ".ktstr-stable-root-marker-{}-{sequence:016x}",
        std::process::id(),
    );
    let temporary = rustix::fs::openat(
        &root,
        temporary_name.as_str(),
        rustix::fs::OFlags::RDWR
            | rustix::fs::OFlags::CREATE
            | rustix::fs::OFlags::EXCL
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::from_raw_mode(0o600),
    )
    .context("create stable-root marker beneath pinned root")?;
    let mut temporary = std::fs::File::from(temporary);
    let result: Result<()> = (|| {
        temporary
            .write_all(&bytes)
            .context("write stable-root marker")?;
        temporary.flush().context("flush stable-root marker")?;
        rustix::fs::fchmod(&temporary, rustix::fs::Mode::from_raw_mode(0o444))
            .context("make stable-root marker immutable")?;
        rustix::fs::renameat(&root, temporary_name.as_str(), &root, domain.marker_name())
            .context("publish stable-root marker")?;
        Ok(())
    })();
    if result.is_err() {
        let _ = rustix::fs::unlinkat(&root, temporary_name.as_str(), rustix::fs::AtFlags::empty());
    }
    result?;
    let published = read_stable_root_marker_at(&root, domain)?;
    anyhow::ensure!(
        published.bytes == bytes,
        "published stable-root marker differs from generated seal",
    );
    // The marker rename is itself a root-directory mutation, so establish a
    // fresh revision and repeat the complete descriptor-relative listing.
    // This closes the publication window between the pre-marker scan and the
    // atomic marker rename: concurrent entry changes cannot turn a stale seal
    // into a successful publication.
    let published_root_stat =
        rustix::fs::fstat(&root).context("stat stable-root after marker publication")?;
    anyhow::ensure!(
        same_stable_tree_inode(&root_stat, &published_root_stat),
        "stable-root inode changed during marker publication",
    );
    let published_listing =
        compute_stable_root_listing(&root, &published_root_stat, domain, identity)?;
    anyhow::ensure!(
        published_listing == listing,
        "stable-root listing changed during marker publication: before={listing:?}, \
         after={published_listing:?}",
    );
    recheck_stable_root_path(root_path, &published_root_stat)?;
    let rechecked_marker = read_stable_root_marker_at(&root, domain)?;
    anyhow::ensure!(
        rechecked_marker.bytes == published.bytes
            && same_stable_tree_revision(&rechecked_marker.stat, &published.stat),
        "stable-root marker changed during post-publication validation",
    );
    Ok(())
}

fn stable_root_marker_matches_inner(
    root_path: &Path,
    domain: StableRootDomain,
    identity: u64,
    verify_listing: bool,
) -> Result<()> {
    let root = open_stable_root(root_path)?;
    let root_stat = rustix::fs::fstat(&root).context("stat stable-root before validation")?;
    validate_stable_root_stat(&root_stat)?;
    let marker = read_stable_root_marker_at(&root, domain)?;
    anyhow::ensure!(
        marker.seal.domain == domain,
        "stable-root marker domain does not match its pathname",
    );
    anyhow::ensure!(
        marker.seal.identity == identity,
        "stable-root marker identity does not match {identity:016x}",
    );
    anyhow::ensure!(
        marker.seal.root_dev == root_stat.st_dev && marker.seal.root_ino == root_stat.st_ino,
        "stable-root marker names a different root inode",
    );
    if verify_listing {
        let listing = compute_stable_root_listing(&root, &root_stat, domain, identity)?;
        anyhow::ensure!(
            listing.entry_count == marker.seal.entry_count
                && listing.hash == marker.seal.listing_hash,
            "stable-root listing does not match its published seal",
        );
    }
    recheck_stable_root_path(root_path, &root_stat)?;
    let rechecked_marker = read_stable_root_marker_at(&root, domain)?;
    anyhow::ensure!(
        rechecked_marker.bytes == marker.bytes
            && same_stable_tree_revision(&rechecked_marker.stat, &marker.stat),
        "stable-root marker changed while validating its tree",
    );
    Ok(())
}

fn stable_root_marker_matches(
    root_path: &Path,
    domain: StableRootDomain,
    identity: u64,
    verify_listing: bool,
) -> Result<bool> {
    match stable_root_marker_matches_inner(root_path, domain, identity, verify_listing) {
        Ok(()) => Ok(true),
        Err(error) => {
            tracing::debug!(
                root = %root_path.display(),
                marker = domain.marker_name(),
                error = %error,
                "stable-root seal is incomplete; rebuilding",
            );
            Ok(false)
        }
    }
}

/// Recursively empty one already-opened directory and unlink only the parent
/// entry which still names that exact inode. All chmod/stat/open/unlink
/// operations are descriptor-relative and every open rejects symlinks.
fn remove_open_stable_tree_directory(
    parent: &std::os::fd::OwnedFd,
    name: &std::ffi::OsStr,
    directory: &std::os::fd::OwnedFd,
    identity: &rustix::fs::Stat,
    display_path: &Path,
) -> Result<()> {
    use std::os::fd::AsRawFd as _;

    rustix::fs::fchmod(directory, rustix::fs::Mode::from_raw_mode(0o700)).with_context(|| {
        format!(
            "make stale stable-tree directory removable {}",
            display_path.display()
        )
    })?;

    loop {
        let proc_path = PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd()));
        let entries = std::fs::read_dir(&proc_path).with_context(|| {
            format!(
                "enumerate pinned stale stable-tree directory {}",
                display_path.display()
            )
        })?;
        let mut found = false;
        for entry in entries {
            let entry = entry.with_context(|| {
                format!(
                    "read stale stable-tree entry beneath {}",
                    display_path.display()
                )
            })?;
            found = true;
            let child_name = entry.file_name();
            let child_display = display_path.join(&child_name);
            let before = match rustix::fs::statat(
                directory,
                &child_name,
                rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
            ) {
                Ok(stat) => stat,
                Err(error) if error == rustix::io::Errno::NOENT => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("stat stale stable-tree child {}", child_display.display())
                    });
                }
            };
            if rustix::fs::FileType::from_raw_mode(before.st_mode).is_dir() {
                let child = match rustix::fs::openat(
                    directory,
                    &child_name,
                    rustix::fs::OFlags::RDONLY
                        | rustix::fs::OFlags::DIRECTORY
                        | rustix::fs::OFlags::NOFOLLOW
                        | rustix::fs::OFlags::CLOEXEC,
                    rustix::fs::Mode::empty(),
                ) {
                    Ok(child) => child,
                    Err(error)
                        if error == rustix::io::Errno::NOENT
                            || error == rustix::io::Errno::NOTDIR
                            || error == rustix::io::Errno::LOOP =>
                    {
                        continue;
                    }
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("open stale stable-tree child {}", child_display.display())
                        });
                    }
                };
                let opened = rustix::fs::fstat(&child).with_context(|| {
                    format!("stat opened stable-tree child {}", child_display.display())
                })?;
                if !same_stable_tree_inode(&before, &opened) {
                    continue;
                }
                remove_open_stable_tree_directory(
                    directory,
                    &child_name,
                    &child,
                    &opened,
                    &child_display,
                )?;
            } else {
                match rustix::fs::unlinkat(directory, &child_name, rustix::fs::AtFlags::empty()) {
                    Ok(()) => {}
                    Err(error)
                        if error == rustix::io::Errno::NOENT
                            || error == rustix::io::Errno::ISDIR => {}
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("remove stale stable-tree child {}", child_display.display())
                        });
                    }
                }
            }
        }
        if !found {
            break;
        }
    }

    let current = rustix::fs::statat(parent, name, rustix::fs::AtFlags::SYMLINK_NOFOLLOW)
        .with_context(|| {
            format!(
                "restat emptied stable-tree directory {}",
                display_path.display()
            )
        })?;
    anyhow::ensure!(
        same_stable_tree_inode(identity, &current),
        "stable-tree directory entry changed while removing {}",
        display_path.display()
    );
    rustix::fs::unlinkat(parent, name, rustix::fs::AtFlags::REMOVEDIR).with_context(|| {
        format!(
            "remove stale stable-tree directory {}",
            display_path.display()
        )
    })
}

fn stable_tree_is_complete(root: &Path, identity: u64) -> Result<bool> {
    stable_root_marker_matches(root, StableRootDomain::Source, identity, true)
}

fn checked_mode(mode: u32) -> Result<()> {
    anyhow::ensure!(
        mode & !0o7777 == 0,
        "artifact tree mode contains non-permission bits: {mode:#o}"
    );
    Ok(())
}

fn checked_relative_path(path: &Path) -> Result<PathBuf> {
    checked_relative_bytes(path.as_os_str().as_bytes())
}

fn checked_relative_bytes(bytes: &[u8]) -> Result<PathBuf> {
    anyhow::ensure!(!bytes.is_empty(), "artifact tree path is empty");
    let path = PathBuf::from(OsString::from_vec(bytes.to_vec()));
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(component) => normalized.push(component),
            _ => anyhow::bail!(
                "artifact tree path is not a normalized relative path: {}",
                path.display()
            ),
        }
    }
    anyhow::ensure!(
        normalized.as_os_str().as_bytes() == bytes,
        "artifact tree path is not canonically spelled: {}",
        path.display()
    );
    Ok(normalized)
}

fn validate_symlink_target(path: &Path, target: &Path) -> Result<()> {
    anyhow::ensure!(
        !target.as_os_str().is_empty() && !target.is_absolute(),
        "artifact symlink {} has an empty or absolute target: {}",
        path.display(),
        target.display()
    );
    let mut depth = path
        .parent()
        .map_or(0, |parent| parent.components().count());
    for component in target.components() {
        match component {
            Component::Normal(_) => depth += 1,
            Component::CurDir => {}
            Component::ParentDir if depth > 0 => depth -= 1,
            Component::ParentDir => anyhow::bail!(
                "artifact symlink {} escapes the materialized tree: {}",
                path.display(),
                target.display()
            ),
            Component::RootDir | Component::Prefix(_) => anyhow::bail!(
                "artifact symlink {} has an absolute target: {}",
                path.display(),
                target.display()
            ),
        }
    }
    Ok(())
}

fn validate_source_shape(entries: &BTreeMap<PathBuf, SourceEntry>) -> Result<()> {
    let nondirectories = entries
        .iter()
        .filter_map(|(path, entry)| {
            (!matches!(entry, SourceEntry::Directory { .. })).then_some(path)
        })
        .cloned()
        .collect::<BTreeSet<_>>();
    for path in entries.keys() {
        let mut parent = path.parent();
        while let Some(ancestor) = parent.filter(|ancestor| !ancestor.as_os_str().is_empty()) {
            anyhow::ensure!(
                !nondirectories.contains(ancestor),
                "artifact tree non-directory {} is an ancestor of {}",
                ancestor.display(),
                path.display()
            );
            parent = ancestor.parent();
        }
    }
    Ok(())
}

fn cache_paths(root: &Path, identity: u64) -> (PathBuf, PathBuf, PathBuf) {
    let records = root.join(RECORD_DIR);
    let locks = root.join(LOCK_DIR);
    (
        records.join(format!("{identity:016x}.json")),
        locks.join(format!("{identity:016x}.lock")),
        locks.join(NAMESPACE_GATE),
    )
}

fn ensure_cache_dirs(root: &Path) -> Result<()> {
    for directory in [
        root.to_path_buf(),
        root.join(RECORD_DIR),
        root.join(LOCK_DIR),
    ] {
        std::fs::create_dir_all(&directory)
            .with_context(|| format!("create artifact tree cache dir {}", directory.display()))?;
    }
    Ok(())
}

fn fixed_hasher() -> ahash::AHasher {
    ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher()
}

fn hash_bytes(hasher: &mut ahash::AHasher, bytes: &[u8]) {
    hasher.write_u64(bytes.len() as u64);
    hasher.write(bytes);
}

fn record_integrity(record: &ArtifactTreeRecord) -> u64 {
    let mut hasher = fixed_hasher();
    hash_bytes(&mut hasher, b"ktstr-artifact-tree-record");
    hasher.write_u32(record.version);
    hasher.write_u64(record.identity);
    hasher.write_u64(record.entries.len() as u64);
    for entry in &record.entries {
        match entry {
            RecordEntry::Directory { path, mode } => {
                hasher.write_u8(0);
                hash_bytes(&mut hasher, path);
                hasher.write_u32(*mode);
            }
            RecordEntry::File {
                path,
                mode,
                content_hash,
                len,
            } => {
                hasher.write_u8(1);
                hash_bytes(&mut hasher, path);
                hasher.write_u32(*mode);
                hasher.write_u64(*content_hash);
                hasher.write_u64(*len);
            }
            RecordEntry::Symlink { path, target } => {
                hasher.write_u8(2);
                hash_bytes(&mut hasher, path);
                hash_bytes(&mut hasher, target);
            }
        }
    }
    hasher.finish()
}

fn validate_record(record: &ArtifactTreeRecord, expected_identity: u64) -> Result<()> {
    anyhow::ensure!(
        record.version == RECORD_SCHEMA,
        "unsupported artifact tree record version {}",
        record.version
    );
    anyhow::ensure!(
        record.identity == expected_identity,
        "artifact tree identity collision: expected {expected_identity:016x}, got {:016x}",
        record.identity
    );
    anyhow::ensure!(!record.entries.is_empty(), "artifact tree record is empty");
    anyhow::ensure!(
        record.integrity_ahash == record_integrity(record),
        "artifact tree record integrity mismatch"
    );
    let mut prior: Option<&[u8]> = None;
    let mut source_shape = BTreeMap::new();
    for entry in &record.entries {
        if let Some(prior) = prior {
            anyhow::ensure!(
                prior < entry.path_bytes(),
                "artifact tree record paths are not sorted and unique"
            );
        }
        prior = Some(entry.path_bytes());
        let path = entry.path()?;
        match entry {
            RecordEntry::Directory { mode, .. } => {
                checked_mode(*mode)?;
                source_shape.insert(path, false);
            }
            RecordEntry::File { mode, .. } => {
                checked_mode(*mode)?;
                source_shape.insert(path, true);
            }
            RecordEntry::Symlink { target, .. } => {
                let target = PathBuf::from(OsString::from_vec(target.clone()));
                validate_symlink_target(&path, &target)?;
                source_shape.insert(path, true);
            }
        }
    }
    let nondirectories = source_shape
        .iter()
        .filter(|(_, nondirectory)| **nondirectory)
        .map(|(path, _)| path.clone())
        .collect::<BTreeSet<_>>();
    for path in source_shape.keys() {
        let mut parent = path.parent();
        while let Some(ancestor) = parent.filter(|ancestor| !ancestor.as_os_str().is_empty()) {
            anyhow::ensure!(
                !nondirectories.contains(ancestor),
                "artifact tree non-directory {} is an ancestor of {}",
                ancestor.display(),
                path.display()
            );
            parent = ancestor.parent();
        }
    }
    Ok(())
}

fn read_and_lease_record(path: &Path, identity: u64) -> Result<Option<LeasedRecord>> {
    #[cfg(test)]
    update_stable_root_test_counters(|counters| counters.record_reads += 1);
    // Acquire the one namespace-wide shared gate before observing the record.
    // A collector can finish before this point, in which case a missing object
    // is a clean cache miss; after this point it cannot unlink an object before
    // the bounded materializer opens it.
    let content_namespace = super::content::lease_content_namespace()?;
    let Some((mut file, before)) = super::content::open_cache_record(path, "artifact tree record")?
    else {
        return Ok(None);
    };
    let mode = file
        .metadata()
        .with_context(|| format!("stat artifact tree record mode {}", path.display()))?
        .permissions()
        .mode();
    anyhow::ensure!(
        mode & 0o222 == 0,
        "artifact tree record is mutable: {}",
        path.display()
    );
    anyhow::ensure!(
        before.size <= RECORD_MAX_BYTES,
        "artifact tree record {} is too large: {} bytes",
        path.display(),
        before.size
    );
    let mut bytes = Vec::with_capacity(usize::try_from(before.size)?);
    file.read_to_end(&mut bytes)
        .with_context(|| format!("read artifact tree record {}", path.display()))?;
    anyhow::ensure!(
        super::content::StableFileIdentity::from_file(&file)? == before,
        "artifact tree record changed while reading: {}",
        path.display()
    );
    let record: ArtifactTreeRecord = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse artifact tree record {}", path.display()))?;
    validate_record(&record, identity)?;
    let mut objects = BTreeSet::new();
    for entry in &record.entries {
        let RecordEntry::File {
            content_hash, len, ..
        } = entry
        else {
            continue;
        };
        let key = (*content_hash, *len);
        if !objects.insert(key) {
            continue;
        }
        #[cfg(test)]
        update_stable_root_test_counters(|counters| counters.cas_object_opens += 1);
        let Some((object, _)) = content_namespace.open_object(*content_hash, *len)? else {
            return Ok(None);
        };
        drop(object);
    }
    content_namespace.publish_artifact_reference(path, identity, &objects)?;
    Ok(Some(LeasedRecord {
        record,
        content_namespace,
    }))
}

fn record_content_objects(record: &ArtifactTreeRecord) -> BTreeSet<(u64, u64)> {
    record
        .entries
        .iter()
        .filter_map(|entry| match entry {
            RecordEntry::File {
                content_hash, len, ..
            } => Some((*content_hash, *len)),
            _ => None,
        })
        .collect()
}

fn parallel_indexed<T, R, F>(items: Vec<T>, operation: F) -> Result<Vec<R>>
where
    T: Send,
    R: Send,
    F: Fn(T) -> Result<R> + Sync,
{
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let item_count = items.len();
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(ARTIFACT_IO_WORKERS_MAX)
        .min(item_count);
    if workers == 1 {
        return items.into_iter().map(operation).collect();
    }

    let queue = std::sync::Mutex::new(items.into_iter().enumerate());
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let sender = sender.clone();
            let queue = &queue;
            let operation = &operation;
            scope.spawn(move || {
                loop {
                    let next = queue
                        .lock()
                        .expect("artifact I/O work queue poisoned")
                        .next();
                    let Some((index, item)) = next else {
                        break;
                    };
                    if sender.send((index, operation(item))).is_err() {
                        break;
                    }
                }
            });
        }
    });
    drop(sender);

    let mut ordered = std::iter::repeat_with(|| None)
        .take(item_count)
        .collect::<Vec<Option<Result<R>>>>();
    for (index, result) in receiver {
        ordered[index] = Some(result);
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.with_context(|| format!("artifact I/O worker {index} returned no result"))?
        })
        .collect()
}

fn publish_source(identity: u64, mut source: ArtifactTreeSource) -> Result<LeasedRecord> {
    source.publish_pinned_window()?;
    validate_source_shape(&source.entries)?;
    let content_namespace = match source.content_namespace.take() {
        Some(namespace) => namespace,
        None => super::content::lease_content_namespace()?,
    };
    let mut entries = source
        .entries
        .into_iter()
        .map(|(path, entry)| {
            let path_bytes = path.as_os_str().as_bytes().to_vec();
            match entry {
                SourceEntry::Directory { mode } => Ok(RecordEntry::Directory {
                    path: path_bytes,
                    mode,
                }),
                SourceEntry::File {
                    file: SourceFile::Published { content_hash, len },
                    mode,
                } => Ok(RecordEntry::File {
                    path: path_bytes,
                    mode,
                    content_hash,
                    len,
                }),
                SourceEntry::File {
                    file: SourceFile::Pinned(_) | SourceFile::Publishing,
                    ..
                } => anyhow::bail!(
                    "artifact-tree file escaped bounded CAS publication: {}",
                    path.display()
                ),
                SourceEntry::Symlink { target } => Ok(RecordEntry::Symlink {
                    path: path_bytes,
                    target: target.as_os_str().as_bytes().to_vec(),
                }),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    entries.sort_by(|left, right| left.path_bytes().cmp(right.path_bytes()));
    let mut record = ArtifactTreeRecord {
        version: RECORD_SCHEMA,
        identity,
        entries,
        integrity_ahash: 0,
    };
    record.integrity_ahash = record_integrity(&record);
    validate_record(&record, identity)?;
    Ok(LeasedRecord {
        record,
        content_namespace,
    })
}

fn write_record(path: &Path, record: &ArtifactTreeRecord) -> Result<()> {
    let parent = path
        .parent()
        .context("artifact tree record path has no parent")?;
    let mut temporary = tempfile::Builder::new()
        .prefix(&format!(".tmp-{:016x}-", record.identity))
        .tempfile_in(parent)
        .with_context(|| format!("create artifact tree record temp in {}", parent.display()))?;
    serde_json::to_writer(temporary.as_file_mut(), record)
        .context("serialize artifact tree record")?;
    temporary
        .as_file_mut()
        .flush()
        .context("flush artifact tree record")?;
    temporary
        .as_file()
        .set_permissions(std::fs::Permissions::from_mode(0o444))
        .context("make artifact tree record immutable")?;
    // NamedTempFile::persist uses same-filesystem rename with replacement on
    // Unix. A reconstructible corrupt/stale record is therefore atomically
    // replaced; readers see either the complete old revision or this complete
    // new revision, never the temporary file.
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish artifact tree record {}", path.display()))?;
    Ok(())
}

fn materialize(
    leased: LeasedRecord,
    parent: &Path,
    cache_root: &Path,
    identity: u64,
    cache_hit: bool,
    waited: bool,
    closure: ArtifactClosureLease,
) -> Result<MaterializedArtifactTree> {
    let directory = tempfile::Builder::new()
        .prefix(&format!("{MATERIALIZATION_PREFIX}{identity:016x}-"))
        .tempdir_in(parent)
        .with_context(|| format!("create private artifact tree in {}", parent.display()))?;
    let root = directory.path();
    let live_path = root.join(MATERIALIZATION_LIVE_LOCK);
    let live = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&live_path)
        .with_context(|| format!("create artifact-tree liveness lock {}", live_path.display()))?;
    super::content::flock_retry(&live, rustix::fs::FlockOperation::LockExclusive)
        .with_context(|| format!("lock artifact-tree liveness file {}", live_path.display()))?;

    for entry in &leased.record.entries {
        let path = entry.path()?;
        let destination = root.join(&path);
        if let RecordEntry::Directory { .. } = entry {
            std::fs::create_dir_all(&destination).with_context(|| {
                format!("create materialized directory {}", destination.display())
            })?;
        }
    }
    let materialization_work = leased
        .record
        .entries
        .iter()
        .filter(|entry| !matches!(entry, RecordEntry::Directory { .. }))
        .collect::<Vec<_>>();
    parallel_indexed(materialization_work, |entry| {
        let path = entry.path()?;
        let destination = root.join(&path);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create materialized parent {}", parent.display()))?;
        }
        match entry {
            RecordEntry::Directory { .. } => unreachable!("directories were filtered above"),
            RecordEntry::File {
                mode,
                content_hash,
                len,
                ..
            } => {
                let (object, object_path) = leased
                    .content_namespace
                    .open_object(*content_hash, *len)?
                    .with_context(|| {
                        format!("artifact tree object missing for {content_hash:016x}/{len}")
                    })?;
                let copied = reflink_required(&object, &object_path, &destination)?;
                anyhow::ensure!(
                    copied == *len,
                    "materialized artifact {} has length {copied}, expected {len}",
                    destination.display()
                );
                std::fs::set_permissions(&destination, std::fs::Permissions::from_mode(*mode))
                    .with_context(|| {
                        format!("set materialized mode on {}", destination.display())
                    })?;
            }
            RecordEntry::Symlink { target, .. } => {
                let target = OsString::from_vec(target.clone());
                std::os::unix::fs::symlink(&target, &destination).with_context(|| {
                    format!(
                        "create materialized symlink {} -> {}",
                        destination.display(),
                        target.to_string_lossy()
                    )
                })?;
            }
        }
        Ok(())
    })?;
    let mut directory_modes = leased
        .record
        .entries
        .iter()
        .filter_map(|entry| match entry {
            RecordEntry::Directory { path, mode } => Some((path, mode)),
            _ => None,
        })
        .collect::<Vec<_>>();
    directory_modes.sort_by_key(|(path, _)| std::cmp::Reverse(path.len()));
    for (path, mode) in directory_modes {
        let path = checked_relative_bytes(path)?;
        // Directory write protection is not an immutability boundary for the
        // owning uid, but it can poison a runner workspace if the process is
        // killed before TempDir or stable-tree cleanup gets to chmod it back.
        // Keep every materialized directory owner-removable from the moment
        // it becomes visible; regular-file modes remain exact and read-only.
        std::fs::set_permissions(
            root.join(path),
            std::fs::Permissions::from_mode(*mode | 0o700),
        )?;
    }
    Ok(MaterializedArtifactTree {
        directory,
        _live: live,
        _closure: closure,
        cache_root: cache_root.to_path_buf(),
        identity,
        cache_hit,
        waited,
        elapsed: Duration::ZERO,
    })
}

struct MaterializationGcGate(std::os::fd::OwnedFd);

impl Drop for MaterializationGcGate {
    fn drop(&mut self) {
        // Closing a CLOEXEC descriptor normally releases its flock, but a
        // concurrent fork can inherit the same open-file description until
        // exec closes its copy. Explicit LOCK_UN applies to that shared OFD,
        // so a completed collector cannot leave a transient phantom owner
        // which makes the next opportunistic pass skip.
        let _ = super::content::flock_retry(&self.0, rustix::fs::FlockOperation::Unlock);
    }
}

/// Remove crash-left private trees without touching a live consumer.
///
/// Every materialization owns an exclusive flock inside its directory. One
/// nonblocking cross-process collector runs per interval; callers on the hot
/// path otherwise do no directory scan. Candidates are globally sorted, then
/// selected after a persisted cursor so a bounded pass eventually covers the
/// complete namespace instead of repeatedly examining read_dir's first page.
/// The grace interval closes the tiny create-before-flock window.
fn gc_stale_materializations(parent: &Path, now: SystemTime) -> Result<()> {
    let lock_path = parent.join(MATERIALIZATION_GC_LOCK);
    let Some(collector) =
        crate::flock::try_flock(&lock_path, crate::flock::FlockMode::Exclusive)
            .with_context(|| format!("acquire materialization GC gate {}", lock_path.display()))?
    else {
        return Ok(());
    };
    let _collector = MaterializationGcGate(collector);
    let stamp_path = parent.join(MATERIALIZATION_GC_STAMP);
    let now_nanos = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos();
    if let Ok(stamp) = std::fs::read_to_string(&stamp_path)
        && let Ok(last_nanos) = stamp.trim().parse::<u128>()
        && now_nanos.saturating_sub(last_nanos) < MATERIALIZATION_GC_INTERVAL.as_nanos()
    {
        return Ok(());
    }

    let mut candidates = std::fs::read_dir(parent)
        .with_context(|| format!("scan artifact materializations in {}", parent.display()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .as_bytes()
                .starts_with(MATERIALIZATION_PREFIX.as_bytes())
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.file_name()
            .as_bytes()
            .cmp(right.file_name().as_bytes())
    });

    let cursor_path = parent.join(MATERIALIZATION_GC_CURSOR);
    let cursor = match std::fs::read(&cursor_path) {
        Ok(cursor) => cursor,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => {
            return Err(error).with_context(|| {
                format!("read materialization GC cursor {}", cursor_path.display())
            });
        }
    };
    let start = if cursor.is_empty() {
        0
    } else {
        candidates.partition_point(|entry| entry.file_name().as_bytes() <= cursor.as_slice())
    };
    let selected = (0..candidates.len().min(MATERIALIZATION_GC_SCAN_LIMIT))
        .map(|offset| (start + offset) % candidates.len())
        .collect::<Vec<_>>();

    for &index in &selected {
        let entry = &candidates[index];
        let path = entry.path();
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_dir()
            || now
                .duration_since(metadata.modified().unwrap_or(now))
                .unwrap_or(Duration::ZERO)
                < MATERIALIZATION_GC_GRACE
        {
            continue;
        }
        let live_path = path.join(MATERIALIZATION_LIVE_LOCK);
        let live = match OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&live_path)
        {
            Ok(file) => Some(file),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(_) => continue,
        };
        if let Some(live) = &live {
            match rustix::fs::flock(live, rustix::fs::FlockOperation::NonBlockingLockExclusive) {
                Ok(()) => {}
                Err(error) if error == rustix::io::Errno::WOULDBLOCK => continue,
                Err(_) => continue,
            }
        }
        if let Err(error) = std::fs::remove_dir_all(&path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::debug!(
                path = %path.display(),
                error = %error,
                "could not remove stale artifact materialization",
            );
        }
    }
    if let Some(index) = selected.last() {
        std::fs::write(&cursor_path, candidates[*index].file_name().as_bytes()).with_context(
            || format!("write materialization GC cursor {}", cursor_path.display()),
        )?;
    }
    std::fs::write(&stamp_path, format!("{now_nanos}\n"))
        .with_context(|| format!("write materialization GC stamp {}", stamp_path.display()))?;
    Ok(())
}

fn reflink_required(
    source_file: &std::fs::File,
    source_display: &Path,
    destination: &Path,
) -> Result<u64> {
    let destination_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(destination)
        .with_context(|| format!("create artifact-tree COW clone {}", destination.display()))?;
    if let Err(error) = crate::reflink::ficlone(&destination_file, source_file) {
        drop(destination_file);
        let _ = std::fs::remove_file(destination);
        return Err(error).with_context(|| {
            format!(
                "FICLONE is required for artifact-tree materialization {} -> {}; \
                 place KTSTR_CACHE_DIR on a reflink-capable filesystem",
                source_display.display(),
                destination.display(),
            )
        });
    }
    destination_file
        .metadata()
        .map(|metadata| metadata.len())
        .with_context(|| format!("stat artifact-tree COW clone {}", destination.display()))
}

fn wait_for_lock<C>(
    coordination: &mut super::content::CoordinationFile,
    lock_path: &Path,
    label: &str,
    identity: u64,
    successor: bool,
    cancelled: &C,
) -> Result<()>
where
    C: Fn() -> bool,
{
    const RETRY: Duration = Duration::from_millis(100);
    const HEARTBEAT: Duration = Duration::from_secs(10);
    let started = Instant::now();
    let role = if successor { "successor" } else { "producer" };
    eprintln!("{label}: waiting for artifact-tree {role} {identity:016x}; elapsed=0s");
    let mut next_heartbeat = HEARTBEAT;
    loop {
        if cancelled() {
            anyhow::bail!("artifact tree cache wait interrupted");
        }
        if successor {
            // Join the kernel's writer queue. A bounded signal tick returns
            // control only to observe cancellation/progress; this is not a
            // repeated nonblocking election and therefore cannot be starved
            // by the reader herd which follows a successful publication.
            let deadline = Instant::now() + HEARTBEAT;
            match crate::flock::block_flock_step(
                lock_path,
                crate::flock::FlockMode::Exclusive,
                deadline,
                HEARTBEAT,
            )? {
                crate::flock::FlockWait::Granted(fd) => {
                    coordination.adopt_locked_file(fd);
                    break;
                }
                crate::flock::FlockWait::Tick | crate::flock::FlockWait::DeadlineExpired => {}
            }
        } else {
            match coordination.try_lock_shared() {
                Ok(()) => break,
                Err(error) if error == rustix::io::Errno::WOULDBLOCK => {}
                Err(error) => return Err(error.into()),
            }
        }
        let elapsed = started.elapsed();
        if elapsed >= next_heartbeat {
            eprintln!(
                "{label}: still waiting for artifact-tree {role} {identity:016x}; elapsed={}",
                humantime::format_duration(elapsed)
            );
            next_heartbeat = elapsed + HEARTBEAT;
        }
        std::thread::park_timeout(RETRY.min(next_heartbeat.saturating_sub(elapsed)));
    }
    eprintln!(
        "{label}: acquired artifact-tree {role} {identity:016x}; elapsed={}",
        humantime::format_duration(started.elapsed())
    );
    Ok(())
}

#[cfg(test)]
#[path = "artifact_tree_tests.rs"]
mod tests;
