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
use std::os::unix::ffi::{OsStrExt as _, OsStringExt as _};
use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context as _, Result};

const RECORD_SCHEMA: u32 = 1;
const RECORD_DIR: &str = "records-v1";
const LOCK_DIR: &str = ".locks-v1";
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
const STABLE_TREE_MARKER: &str = ".ktstr-artifact-tree-v1";
const STABLE_BUILD_MARKER: &str = ".ktstr-stable-build-v1";

enum SourceEntry {
    Directory {
        mode: u32,
    },
    File {
        pinned: super::PinnedContentFile,
        mode: u32,
    },
    Symlink {
        target: PathBuf,
    },
}

enum PendingTreeEntry {
    Directory { relative: PathBuf, mode: u32 },
    File { relative: PathBuf, source: PathBuf },
    Symlink { relative: PathBuf, target: PathBuf },
}

/// One relocatable artifact tree ready for immutable publication.
///
/// Regular files are pinned as they are inserted. Replacing a Cargo output
/// pathname after this point cannot retarget the bytes eventually published.
#[doc(hidden)]
pub struct ArtifactTreeSource {
    entries: BTreeMap<PathBuf, SourceEntry>,
    // Number of descendants already inserted below each normalized path.
    // This makes rejecting a later file/symlink parent logarithmic instead of
    // rescanning the complete tree after every insertion.
    descendant_counts: BTreeMap<PathBuf, usize>,
    #[cfg(test)]
    insertion_validation_visits: usize,
}

impl ArtifactTreeSource {
    /// Start an empty tree.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            descendant_counts: BTreeMap::new(),
            #[cfg(test)]
            insertion_validation_visits: 0,
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
        self.insert_entry(relative, SourceEntry::File { pinned, mode })
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

        let file_work = captured
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| match entry {
                PendingTreeEntry::File { source, .. } => Some((index, source.as_path())),
                PendingTreeEntry::Directory { .. } | PendingTreeEntry::Symlink { .. } => None,
            })
            .collect::<Vec<_>>();
        let pinned = parallel_indexed(file_work, |(index, source)| {
            let file = super::pin_content_file(source)
                .with_context(|| format!("pin artifact tree input {}", source.display()))?;
            Ok((index, file))
        })?;
        let mut pinned_by_index = std::iter::repeat_with(|| None)
            .take(captured.len())
            .collect::<Vec<Option<super::PinnedContentFile>>>();
        for (index, pinned) in pinned {
            pinned_by_index[index] = Some(pinned);
        }

        for (index, entry) in captured.into_iter().enumerate() {
            match entry {
                PendingTreeEntry::Directory { relative, mode } => {
                    self.insert_directory(relative, mode)?;
                }
                PendingTreeEntry::File { relative, .. } => {
                    let pinned = pinned_by_index[index]
                        .take()
                        .context("parallel artifact-tree pin returned no file")?;
                    self.insert_pinned_file_with_policy(relative, pinned, immutable)?;
                }
                PendingTreeEntry::Symlink { relative, target } => {
                    self.insert_symlink(relative, target)?;
                }
            }
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

enum ObjectLease {
    Published(super::ContentFileSnapshot),
    Existing(super::content::ContentObjectLease),
}

impl ObjectLease {
    fn path(&self) -> &Path {
        match self {
            Self::Published(snapshot) => snapshot.path(),
            Self::Existing(lease) => lease.path(),
        }
    }

    fn file(&self) -> &std::fs::File {
        match self {
            Self::Published(snapshot) => snapshot.file(),
            Self::Existing(lease) => lease.file(),
        }
    }
}

struct LeasedRecord {
    record: ArtifactTreeRecord,
    objects: BTreeMap<(u64, u64), ObjectLease>,
}

struct SelectedRecord {
    leased: LeasedRecord,
    cache_hit: bool,
}

/// One private materialization of a reusable artifact tree.
///
/// The content leases remain live until this value is dropped, so content GC
/// cannot unlink a source object while nextest executes from the reflinked
/// tree.
#[doc(hidden)]
pub struct MaterializedArtifactTree {
    directory: tempfile::TempDir,
    // Declared after `directory` deliberately: Rust drops fields in
    // declaration order, so TempDir removes the namespace while this lock is
    // still held. A collector can never acquire the liveness lock in the
    // middle of normal teardown.
    _live: std::fs::File,
    _objects: BTreeMap<(u64, u64), ObjectLease>,
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

/// Cross-process cache for immutable, relocatable artifact trees.
#[doc(hidden)]
pub struct ArtifactTreeCache {
    root: PathBuf,
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
        Self { root: root.into() }
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
            || Ok(true),
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
        U: Fn() -> Result<bool>,
        VH: Fn() -> Result<bool>,
        VP: Fn() -> Result<bool>,
        C: Fn() -> bool,
        P: FnOnce(&LeasedRecord) -> Result<()>,
    {
        anyhow::ensure!(!progress_label.is_empty(), "artifact tree label is empty");
        if cancelled() {
            anyhow::bail!("artifact tree cache operation interrupted");
        }
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
                    return Ok(None);
                };
                if !cache_usable()? {
                    tracing::debug!(
                        path = %record_path.display(),
                        "rebuilding artifact tree whose coupled stable output is incomplete",
                    );
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
                write_record(&record_path, &leased.record)?;
                Ok(SelectedRecord {
                    leased,
                    cache_hit: false,
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
        )?;
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
            || stable_cargo_build_is_complete(&stable_root, identity),
            validate_cached_identity,
            || {
                Ok(stable_cargo_build_is_complete(&stable_root, identity)?
                    && validate_published_identity()?)
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
                let result = distill_stable_cargo_build(&stable_to_seal, &leased.record)
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
            stable_cargo_build_is_complete(&stable_root, identity)?,
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
        std::fs::create_dir_all(stable_parent).with_context(|| {
            format!(
                "create stable artifact-tree parent {}",
                stable_parent.display()
            )
        })?;
        let final_root = stable_parent.join(format!("{identity:016x}"));
        if stable_tree_is_complete(&final_root, identity)? {
            anyhow::ensure!(
                validate_cached_identity()?,
                "artifact tree inputs changed before accepting stable build {identity:016x}"
            );
            return Ok(StableArtifactTree {
                root: final_root,
                identity,
                cache_hit: true,
            });
        }

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
        if stable_tree_is_complete(&final_root, identity)? {
            anyhow::ensure!(
                validate_cached_identity()?,
                "artifact tree inputs changed before accepting stable build {identity:016x}"
            );
            return Ok(StableArtifactTree {
                root: final_root,
                identity,
                cache_hit: true,
            });
        }
        if final_root.exists() {
            remove_stable_tree(&final_root).with_context(|| {
                format!(
                    "remove incomplete stable artifact tree {}",
                    final_root.display()
                )
            })?;
        }

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
        let marker = tree.root().join(STABLE_TREE_MARKER);
        std::fs::write(&marker, format!("{identity:016x}\n"))
            .with_context(|| format!("write stable artifact-tree marker {}", marker.display()))?;
        std::fs::set_permissions(&marker, std::fs::Permissions::from_mode(0o444)).with_context(
            || {
                format!(
                    "make stable artifact-tree marker immutable {}",
                    marker.display()
                )
            },
        )?;

        // Drop liveness/CAS leases only after detaching TempDir cleanup. The
        // installed FICLONEs are independent inodes and need no source lease.
        let MaterializedArtifactTree {
            directory,
            _live,
            _objects,
            ..
        } = tree;
        let staged = directory.keep();
        drop(_live);
        drop(_objects);
        std::fs::remove_file(staged.join(MATERIALIZATION_LIVE_LOCK)).with_context(|| {
            format!(
                "remove private liveness marker from stable artifact tree {}",
                staged.display()
            )
        })?;
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o555)).with_context(
            || {
                format!(
                    "make stable artifact-tree root immutable {}",
                    staged.display()
                )
            },
        )?;
        std::fs::rename(&staged, &final_root).with_context(|| {
            format!(
                "atomically install stable artifact tree {} -> {}",
                staged.display(),
                final_root.display(),
            )
        })?;
        Ok(StableArtifactTree {
            root: final_root,
            identity,
            cache_hit,
        })
    }
}

fn prepare_stable_cargo_build(root: &Path) -> Result<StableCargoBuild> {
    if root.exists() {
        remove_stable_tree(root)?;
    }
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

fn stable_cargo_output_path(path: &Path) -> bool {
    path.starts_with("target") || path.starts_with("build")
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
    rustix::fs::openat2(
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
    )
    .with_context(|| {
        format!(
            "open stable Cargo directory beneath its root without following links {}",
            relative.display(),
        )
    })
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
    let name = path
        .file_name()
        .with_context(|| format!("stable Cargo output has no file name: {}", path.display()))?;
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
fn distill_stable_cargo_build(build: &StableCargoBuild, record: &ArtifactTreeRecord) -> Result<()> {
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

fn seal_stable_cargo_build(build: &StableCargoBuild, identity: u64) -> Result<()> {
    let marker = build.root.join(STABLE_BUILD_MARKER);
    std::fs::write(&marker, format!("{identity:016x}\n"))
        .with_context(|| format!("write stable Cargo output marker {}", marker.display()))?;
    std::fs::set_permissions(&marker, std::fs::Permissions::from_mode(0o444))
        .with_context(|| format!("seal stable Cargo output marker {}", marker.display()))?;
    let mut files = Vec::new();
    let mut directories = Vec::new();
    for entry in walkdir::WalkDir::new(&build.root)
        .follow_links(false)
        .sort_by_file_name()
    {
        let entry = entry.with_context(|| {
            format!(
                "walk completed stable Cargo output {}",
                build.root.display()
            )
        })?;
        if entry.file_type().is_symlink() {
            continue;
        }
        if entry.file_type().is_dir() {
            directories.push((entry.depth(), entry.into_path()));
        } else if entry.file_type().is_file() {
            files.push(entry.into_path());
        } else {
            anyhow::bail!(
                "stable Cargo output has unsupported file type: {}",
                entry.path().display()
            );
        }
    }
    files.sort();
    parallel_indexed(files, |path| seal_stable_cargo_node(&path, false))?;
    directories.sort_by(|(left_depth, left_path), (right_depth, right_path)| {
        right_depth
            .cmp(left_depth)
            .then_with(|| left_path.cmp(right_path))
    });
    for (_, directory) in directories {
        seal_stable_cargo_node(&directory, true)?;
    }
    Ok(())
}

fn seal_stable_cargo_node(path: &Path, directory: bool) -> Result<()> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    if directory {
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_DIRECTORY);
    }
    let file = options
        .open(path)
        .with_context(|| format!("open stable Cargo output for sealing {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("stat stable Cargo output for sealing {}", path.display()))?;
    anyhow::ensure!(
        if directory {
            metadata.is_dir()
        } else {
            metadata.is_file()
        },
        "stable Cargo output changed type while sealing: {}",
        path.display(),
    );
    let mode = metadata.permissions().mode() & 0o7777;
    let sealed_mode = mode & !0o222;
    if sealed_mode != mode {
        file.set_permissions(std::fs::Permissions::from_mode(sealed_mode))
            .with_context(|| format!("seal stable Cargo output {}", path.display()))?;
    }
    let path_metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("restat sealed stable Cargo output {}", path.display()))?;
    anyhow::ensure!(
        !path_metadata.file_type().is_symlink()
            && path_metadata.dev() == metadata.dev()
            && path_metadata.ino() == metadata.ino(),
        "stable Cargo output pathname changed while sealing: {}",
        path.display(),
    );
    Ok(())
}

fn stable_cargo_build_is_complete(root: &Path, identity: u64) -> Result<bool> {
    let root_metadata = match std::fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("stat stable Cargo output {}", root.display()));
        }
    };
    if !root_metadata.is_dir()
        || root_metadata.file_type().is_symlink()
        || root_metadata.permissions().mode() & 0o222 != 0
    {
        return Ok(false);
    }
    let marker = root.join(STABLE_BUILD_MARKER);
    let marker_metadata = match std::fs::symlink_metadata(&marker) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("stat stable Cargo output marker {}", marker.display()));
        }
    };
    if !marker_metadata.is_file()
        || marker_metadata.file_type().is_symlink()
        || marker_metadata.permissions().mode() & 0o222 != 0
    {
        return Ok(false);
    }
    let bytes = std::fs::read(&marker)
        .with_context(|| format!("read stable Cargo output marker {}", marker.display()))?;
    Ok(bytes == format!("{identity:016x}\n").as_bytes() && root.join("target").is_dir())
}

fn remove_stable_tree(root: &Path) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in walkdir::WalkDir::new(root)
        .follow_links(false)
        .contents_first(false)
    {
        let entry = entry.with_context(|| format!("walk stale stable tree {}", root.display()))?;
        if entry.file_type().is_dir() {
            std::fs::set_permissions(entry.path(), std::fs::Permissions::from_mode(0o700))
                .with_context(|| {
                    format!(
                        "make stale stable-tree directory removable {}",
                        entry.path().display()
                    )
                })?;
        }
    }
    std::fs::remove_dir_all(root)
        .with_context(|| format!("remove stale stable tree {}", root.display()))
}

fn stable_tree_is_complete(root: &Path, identity: u64) -> Result<bool> {
    let marker = root.join(STABLE_TREE_MARKER);
    let root_metadata = match std::fs::metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("stat stable artifact-tree root {}", root.display()));
        }
    };
    if !root_metadata.is_dir() || root_metadata.permissions().mode() & 0o222 != 0 {
        return Ok(false);
    }
    let marker_metadata = match std::fs::symlink_metadata(&marker) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("stat stable artifact-tree marker {}", marker.display()));
        }
    };
    if !marker_metadata.is_file() || marker_metadata.permissions().mode() & 0o222 != 0 {
        return Ok(false);
    }
    let bytes = match std::fs::read(&marker) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read stable artifact-tree marker {}", marker.display()));
        }
    };
    Ok(bytes == format!("{identity:016x}\n").as_bytes())
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
        .filter_map(|(path, nondirectory)| (*nondirectory).then(|| path.clone()))
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
    let mut objects = BTreeMap::new();
    for entry in &record.entries {
        let RecordEntry::File {
            content_hash, len, ..
        } = entry
        else {
            continue;
        };
        let key = (*content_hash, *len);
        if objects.contains_key(&key) {
            continue;
        }
        let Some(lease) = super::content::lease_content_object(*content_hash, *len)? else {
            return Ok(None);
        };
        objects.insert(key, ObjectLease::Existing(lease));
    }
    Ok(Some(LeasedRecord { record, objects }))
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

fn publish_source(identity: u64, source: ArtifactTreeSource) -> Result<LeasedRecord> {
    validate_source_shape(&source.entries)?;
    let published = parallel_indexed(source.entries.into_iter().collect(), |(path, entry)| {
        let path_bytes = path.as_os_str().as_bytes().to_vec();
        match entry {
            SourceEntry::Directory { mode } => Ok((
                RecordEntry::Directory {
                    path: path_bytes,
                    mode,
                },
                None,
            )),
            SourceEntry::File { pinned, mode } => {
                let snapshot = super::snapshot_pinned_artifact_file(pinned)
                    .with_context(|| format!("publish artifact-tree file {}", path.display()))?;
                let key = (snapshot.content_hash(), snapshot.len());
                Ok((
                    RecordEntry::File {
                        path: path_bytes,
                        mode,
                        content_hash: key.0,
                        len: key.1,
                    },
                    Some((key, snapshot)),
                ))
            }
            SourceEntry::Symlink { target } => Ok((
                RecordEntry::Symlink {
                    path: path_bytes,
                    target: target.as_os_str().as_bytes().to_vec(),
                },
                None,
            )),
        }
    })?;
    let mut entries = Vec::with_capacity(published.len());
    let mut objects = BTreeMap::new();
    for (entry, snapshot) in published {
        entries.push(entry);
        if let Some((key, snapshot)) = snapshot {
            objects
                .entry(key)
                .or_insert(ObjectLease::Published(snapshot));
        }
    }
    entries.sort_by(|left, right| left.path_bytes().cmp(right.path_bytes()));
    let mut record = ArtifactTreeRecord {
        version: RECORD_SCHEMA,
        identity,
        entries,
        integrity_ahash: 0,
    };
    record.integrity_ahash = record_integrity(&record);
    validate_record(&record, identity)?;
    Ok(LeasedRecord { record, objects })
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
                let object = leased
                    .objects
                    .get(&(*content_hash, *len))
                    .with_context(|| {
                        format!("artifact tree object lease missing for {content_hash:016x}/{len}")
                    })?;
                let copied = reflink_required(object.file(), object.path(), &destination)?;
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
        std::fs::set_permissions(root.join(path), std::fs::Permissions::from_mode(*mode))?;
    }
    Ok(MaterializedArtifactTree {
        directory,
        _live: live,
        _objects: leased.objects,
        cache_root: cache_root.to_path_buf(),
        identity,
        cache_hit,
        waited,
        elapsed: Duration::ZERO,
    })
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
    let Some(_collector) = crate::flock::try_flock(&lock_path, crate::flock::FlockMode::Exclusive)
        .with_context(|| format!("acquire materialization GC gate {}", lock_path.display()))?
    else {
        return Ok(());
    };
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
        if let Err(error) = std::fs::remove_dir_all(&path) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::debug!(
                    path = %path.display(),
                    error = %error,
                    "could not remove stale artifact materialization",
                );
            }
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
