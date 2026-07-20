//! Parent-built scheduler artifact handoff for orchestrated test runs.
//!
//! A `cargo ktstr` parent discovers every `Path` and `Discover` scheduler
//! referenced by the warmed test executables before nextest starts. It builds
//! each distinct `Discover` package once, writes one immutable manifest, and
//! exports [`crate::KTSTR_SCHEDULER_MANIFEST_ENV`] to every child. This module
//! is the single reader and wire format shared by ordinary tests, coverage,
//! raw `llvm-cov nextest`, staged schedulers, and verifier cells.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::hash::{BuildHasher as _, Hasher as _};
use std::io::{Read, Write as _};
use std::os::unix::ffi::OsStrExt as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// Current schema version for [`SchedulerArtifactManifest`].
pub const SCHEDULER_ARTIFACT_MANIFEST_VERSION: u32 = 1;

const SCHEDULER_BUILD_CACHE_SCHEMA: u32 = 1;
const SCHEDULER_BUILD_RECORD_DIR: &str = "records-v1";
const SCHEDULER_BUILD_LOCK_DIR: &str = ".locks-v1";
const SCHEDULER_BUILD_NAMESPACE_GATE: &str = "namespace.lock";
const SCHEDULER_BUILD_RECORD_MAX_BYTES: u64 = 1 << 20;

/// Parent-owned scheduler artifacts for one orchestrated test run.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerArtifactManifest {
    /// Wire schema version.
    pub version: u32,
    /// Effective Cargo profile used for every scheduler build.
    pub profile: String,
    /// Exact package-and-declaring-workspace artifact mappings.
    pub entries: Vec<SchedulerArtifactEntry>,
}

/// Exact scheduler executable specification carried in a manifest entry.
///
/// The tagged variant is part of the identity: `Discover("x")` and
/// `Path("x")` can never alias even though their string payloads match.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum SchedulerArtifactSpec {
    /// Cargo package built by the parent.
    Discover(String),
    /// Explicit source path pinned and snapshotted by the parent.
    Path(String),
}

impl SchedulerArtifactSpec {
    fn from_scheduler_spec(spec: &crate::test_support::SchedulerSpec) -> anyhow::Result<Self> {
        match spec {
            crate::test_support::SchedulerSpec::Discover(package) => {
                Ok(Self::Discover((*package).to_string()))
            }
            crate::test_support::SchedulerSpec::Path(path) => Ok(Self::Path((*path).to_string())),
            crate::test_support::SchedulerSpec::Eevdf
            | crate::test_support::SchedulerSpec::KernelBuiltin { .. } => {
                anyhow::bail!("kernel-only scheduler has no artifact manifest identity")
            }
        }
    }
}

/// One exact scheduler artifact and every scheduler name which requires it.
///
/// `(binary, manifest_dir, manifest.profile)` is the executable identity.
/// `schedulers` is validation metadata: ordinary resolution needs only the
/// exact artifact identity, while verifier dispatch additionally proves that
/// the selected declared scheduler was part of the parent's build plan.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerArtifactEntry {
    /// Exact tagged scheduler executable specification.
    pub binary: SchedulerArtifactSpec,
    /// Exact declaring `CARGO_MANIFEST_DIR`.
    pub manifest_dir: String,
    /// Sorted, unique declared scheduler names which use this artifact.
    pub schedulers: Vec<String>,
    /// Canonical absolute executable path emitted or snapshotted by the parent.
    pub path: PathBuf,
}

#[derive(Clone)]
struct CachedSchedulerArtifactManifest {
    identity: crate::cache::content::StableFileIdentity,
    manifest: Arc<SchedulerArtifactManifest>,
}

fn scheduler_manifest_cache() -> &'static Mutex<BTreeMap<PathBuf, CachedSchedulerArtifactManifest>>
{
    static CACHE: OnceLock<Mutex<BTreeMap<PathBuf, CachedSchedulerArtifactManifest>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Read one manifest through its pinned inode and memoize immutable revisions.
///
/// Parent-published manifests are mode-0444 and shared by every nextest child.
/// A process may resolve hundreds of scheduler declarations, so repeatedly
/// reading and parsing the same JSON is avoidable. The cache is keyed by the
/// absolute pathname and exact inode revision; atomic replacement therefore
/// invalidates it naturally. Mutable manifests are never cached, preserving
/// direct test/development rewrites.
fn read_scheduler_artifact_manifest(
    manifest_path: &Path,
) -> anyhow::Result<Arc<SchedulerArtifactManifest>> {
    use std::os::unix::fs::PermissionsExt as _;

    let mut file = std::fs::File::open(manifest_path).map_err(|error| {
        anyhow::anyhow!(
            "read scheduler artifact manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    let metadata = file.metadata().map_err(|error| {
        anyhow::anyhow!(
            "stat scheduler artifact manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    if !metadata.is_file() {
        anyhow::bail!(
            "scheduler artifact manifest is not a regular file: {}",
            manifest_path.display()
        );
    }
    let identity = crate::cache::content::StableFileIdentity::from_metadata(&metadata);
    let immutable = metadata.permissions().mode() & 0o222 == 0;
    if immutable {
        let cached = scheduler_manifest_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(manifest_path)
            .filter(|cached| cached.identity == identity)
            .cloned();
        if let Some(cached) = cached {
            anyhow::ensure!(
                crate::cache::content::StableFileIdentity::from_file(&file)? == identity,
                "scheduler artifact manifest changed during cached lookup: {}",
                manifest_path.display()
            );
            return Ok(cached.manifest);
        }
    }

    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|error| {
        anyhow::anyhow!(
            "read scheduler artifact manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    anyhow::ensure!(
        crate::cache::content::StableFileIdentity::from_file(&file)? == identity,
        "scheduler artifact manifest changed while reading: {}",
        manifest_path.display()
    );
    let manifest: SchedulerArtifactManifest = serde_json::from_slice(&bytes).map_err(|error| {
        anyhow::anyhow!(
            "parse scheduler artifact manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    let manifest = Arc::new(manifest);
    if immutable {
        scheduler_manifest_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                manifest_path.to_path_buf(),
                CachedSchedulerArtifactManifest {
                    identity,
                    manifest: Arc::clone(&manifest),
                },
            );
    }
    Ok(manifest)
}

fn validate_scheduler_artifact(path: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    if !path.is_absolute() {
        anyhow::bail!(
            "scheduler artifact path must be absolute, got {}",
            path.display()
        );
    }
    let canonical = std::fs::canonicalize(path).map_err(|error| {
        anyhow::anyhow!(
            "canonicalize scheduler artifact {}: {error}",
            path.display()
        )
    })?;
    if canonical.as_path() != path {
        anyhow::bail!(
            "scheduler artifact path is not canonical: {} resolves to {}",
            path.display(),
            canonical.display(),
        );
    }
    let metadata = std::fs::metadata(path)
        .map_err(|error| anyhow::anyhow!("stat scheduler artifact {}: {error}", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("scheduler artifact is not a file: {}", path.display());
    }
    if metadata.permissions().mode() & 0o111 == 0 {
        anyhow::bail!("scheduler artifact is not executable: {}", path.display());
    }
    if metadata.permissions().mode() & 0o222 != 0 {
        anyhow::bail!(
            "scheduler artifact is mutable; parent snapshots must be read-only: {}",
            path.display()
        );
    }
    Ok(())
}

fn scheduler_artifact_from_manifest_path(
    manifest_path: Option<&OsStr>,
    expected_scheduler: Option<&str>,
    binary: &SchedulerArtifactSpec,
    manifest_dir: &str,
) -> anyhow::Result<Option<PathBuf>> {
    let Some(manifest_path) = manifest_path else {
        return Ok(None);
    };
    if manifest_path.is_empty() {
        anyhow::bail!("{} is set but empty", crate::KTSTR_SCHEDULER_MANIFEST_ENV);
    }
    let manifest_path = PathBuf::from(manifest_path);
    if !manifest_path.is_absolute() {
        anyhow::bail!(
            "{} must name an absolute path, got {}",
            crate::KTSTR_SCHEDULER_MANIFEST_ENV,
            manifest_path.display(),
        );
    }
    let manifest = read_scheduler_artifact_manifest(&manifest_path)?;
    if manifest.version != SCHEDULER_ARTIFACT_MANIFEST_VERSION {
        anyhow::bail!(
            "unsupported scheduler artifact manifest version {} in {} (expected {})",
            manifest.version,
            manifest_path.display(),
            SCHEDULER_ARTIFACT_MANIFEST_VERSION,
        );
    }
    let expected_profile = crate::scheduler_profile_name();
    if manifest.profile != expected_profile {
        anyhow::bail!(
            "scheduler artifact manifest {} was built with profile {:?}, \
             but this child resolves scheduler profile {:?}",
            manifest_path.display(),
            manifest.profile,
            expected_profile,
        );
    }

    let mut identities = BTreeSet::new();
    let mut matched = None;
    for entry in &manifest.entries {
        let binary_value = match &entry.binary {
            SchedulerArtifactSpec::Discover(value) | SchedulerArtifactSpec::Path(value) => value,
        };
        if binary_value.is_empty() || entry.manifest_dir.is_empty() {
            anyhow::bail!(
                "scheduler artifact manifest {} contains an empty binary value or manifest_dir",
                manifest_path.display(),
            );
        }
        if !identities.insert((&entry.binary, &entry.manifest_dir)) {
            anyhow::bail!(
                "duplicate scheduler artifact manifest entry for binary {:?}, \
                 manifest_dir {:?}",
                entry.binary,
                entry.manifest_dir,
            );
        }
        if entry.schedulers.is_empty() {
            anyhow::bail!(
                "scheduler artifact manifest entry for binary {:?}, manifest_dir {:?} \
                 has no requiring scheduler names",
                entry.binary,
                entry.manifest_dir,
            );
        }
        let mut scheduler_names = BTreeSet::new();
        for scheduler in &entry.schedulers {
            if scheduler.is_empty() || !scheduler_names.insert(scheduler) {
                anyhow::bail!(
                    "scheduler artifact manifest entry for binary {:?}, manifest_dir {:?} \
                     has an empty or duplicate scheduler name",
                    entry.binary,
                    entry.manifest_dir,
                );
            }
        }
        if entry.binary == *binary && entry.manifest_dir == manifest_dir {
            matched = Some(entry);
        }
    }

    let Some(entry) = matched else {
        anyhow::bail!(
            "scheduler artifact manifest {} has no exact entry for binary {binary:?}, \
             manifest_dir {manifest_dir:?}",
            manifest_path.display(),
        );
    };
    validate_scheduler_artifact(&entry.path)?;
    if let Some(scheduler) = expected_scheduler
        && !entry.schedulers.iter().any(|name| name == scheduler)
    {
        anyhow::bail!(
            "scheduler artifact manifest entry for binary {binary:?}, \
             manifest_dir {manifest_dir:?} was not prepared for scheduler {scheduler:?}",
        );
    }
    Ok(Some(entry.path.clone()))
}

/// Resolve one `Discover` scheduler through the orchestrator manifest.
///
/// `Ok(None)` means no orchestrator manifest is present, which preserves the
/// direct/manual test-binary resolution path. Once the environment variable is
/// present, the manifest is authoritative: every parse, version, identity,
/// scheduler-name, profile, and artifact failure is returned and callers must
/// not invoke Cargo as a fallback.
pub(crate) fn scheduler_artifact_from_env(
    expected_scheduler: Option<&str>,
    spec: &crate::test_support::SchedulerSpec,
    manifest_dir: &str,
) -> anyhow::Result<Option<PathBuf>> {
    if std::env::var_os(crate::KTSTR_SCHEDULER_MANIFEST_ENV).is_none() {
        return Ok(None);
    }
    let binary = SchedulerArtifactSpec::from_scheduler_spec(spec)?;
    let manifest_path = std::env::var_os(crate::KTSTR_SCHEDULER_MANIFEST_ENV);
    scheduler_artifact_from_manifest_path(
        manifest_path.as_deref(),
        expected_scheduler,
        &binary,
        manifest_dir,
    )
}

/// A live lease over one immutable scheduler executable in the machine CAS.
///
/// Every process which snapshots identical bytes receives the same canonical
/// pathname and inode. Keeping this value alive prevents cache collection from
/// unlinking that pathname while a child manifest refers to it.
pub struct SchedulerArtifactSnapshot {
    snapshot: crate::cache::ContentFileSnapshot,
}

impl SchedulerArtifactSnapshot {
    /// Canonical shared-CAS pathname for the immutable executable.
    pub fn path(&self) -> &Path {
        self.snapshot.path()
    }
}

/// Pin and snapshot one scheduler executable into the shared content CAS.
///
/// The source is opened before hashing or publication, which pins its inode
/// against Cargo's atomic replacement. Identical bytes across components,
/// workspaces, and processes converge on one immutable mode-0555 inode.
pub fn snapshot_scheduler_artifact(
    source_path: &Path,
) -> Result<SchedulerArtifactSnapshot, String> {
    let pinned = crate::cache::pin_content_file(source_path).map_err(|error| {
        format!(
            "open scheduler artifact {} for immutable snapshot: {error:#}",
            source_path.display()
        )
    })?;
    snapshot_pinned_scheduler_artifact(pinned)
}

/// Snapshot an already-pinned scheduler descriptor.
///
/// This is public for cargo-ktstr's race test; production callers normally use
/// [`snapshot_scheduler_artifact`].
#[doc(hidden)]
pub fn snapshot_pinned_scheduler_artifact(
    pinned: crate::cache::PinnedContentFile,
) -> Result<SchedulerArtifactSnapshot, String> {
    use std::os::unix::fs::PermissionsExt;

    let source_path = pinned.source_path().to_path_buf();
    let source_metadata = pinned.source().metadata().map_err(|error| {
        format!(
            "stat opened scheduler artifact {}: {error}",
            source_path.display()
        )
    })?;
    if !source_metadata.is_file() {
        return Err(format!(
            "scheduler artifact is not a file: {}",
            source_path.display()
        ));
    }
    if source_metadata.permissions().mode() & 0o111 == 0 {
        return Err(format!(
            "scheduler artifact is not executable: {}",
            source_path.display()
        ));
    }

    let snapshot = crate::cache::snapshot_pinned_content_file(pinned).map_err(|error| {
        format!(
            "publish pinned scheduler artifact {} in shared content cache: {error:#}",
            source_path.display()
        )
    })?;
    Ok(SchedulerArtifactSnapshot { snapshot })
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SchedulerBuildCacheRecord {
    version: u32,
    identity: u64,
    packages: Vec<String>,
    artifacts: BTreeMap<String, PathBuf>,
    integrity_ahash: u64,
}

/// One input-addressed scheduler workspace build result.
///
/// Every path names the ordinary immutable machine content CAS. The snapshot
/// leases keep those pathnames alive while the caller's child manifest uses
/// them. A cache hit therefore has exactly the same artifact ownership as a
/// fresh Cargo build, but never enters Cargo or the target-directory lease.
#[doc(hidden)]
pub struct CachedSchedulerWorkspaceArtifacts {
    pub paths: BTreeMap<String, PathBuf>,
    pub snapshots: Vec<SchedulerArtifactSnapshot>,
    pub cache_hit: bool,
}

fn scheduler_build_fixed_hasher() -> ahash::AHasher {
    ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher()
}

fn hash_scheduler_build_bytes(hasher: &mut ahash::AHasher, bytes: &[u8]) {
    hasher.write_u64(bytes.len() as u64);
    hasher.write(bytes);
}

fn scheduler_build_record_integrity(
    version: u32,
    identity: u64,
    packages: &[String],
    artifacts: &BTreeMap<String, PathBuf>,
) -> u64 {
    let mut hasher = scheduler_build_fixed_hasher();
    hash_scheduler_build_bytes(&mut hasher, b"ktstr-scheduler-build-record");
    hasher.write_u32(version);
    hasher.write_u64(identity);
    hasher.write_u64(packages.len() as u64);
    for package in packages {
        hash_scheduler_build_bytes(&mut hasher, package.as_bytes());
    }
    hasher.write_u64(artifacts.len() as u64);
    for (package, path) in artifacts {
        hash_scheduler_build_bytes(&mut hasher, package.as_bytes());
        hash_scheduler_build_bytes(&mut hasher, path.as_os_str().as_bytes());
    }
    hasher.finish()
}

fn scheduler_build_cache_paths(root: &Path, identity: u64) -> (PathBuf, PathBuf, PathBuf) {
    let records = root.join(SCHEDULER_BUILD_RECORD_DIR);
    let locks = root.join(SCHEDULER_BUILD_LOCK_DIR);
    (
        records.join(format!("{identity:016x}.json")),
        locks.join(format!("{identity:016x}.lock")),
        locks.join(SCHEDULER_BUILD_NAMESPACE_GATE),
    )
}

fn ensure_scheduler_build_cache_dirs(root: &Path) -> anyhow::Result<()> {
    let directories = [
        root.to_path_buf(),
        root.join(SCHEDULER_BUILD_RECORD_DIR),
        root.join(SCHEDULER_BUILD_LOCK_DIR),
    ];
    for directory in &directories {
        std::fs::create_dir_all(directory).map_err(|error| {
            anyhow::anyhow!(
                "create scheduler build cache directory {}: {error}",
                directory.display()
            )
        })?;
    }
    Ok(())
}

fn scheduler_build_error_is_not_found(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
    })
}

fn reopen_cached_scheduler_artifact(
    path: &Path,
) -> Result<Option<SchedulerArtifactSnapshot>, String> {
    let pinned = match crate::cache::pin_content_file(path) {
        Ok(pinned) => pinned,
        Err(error) if scheduler_build_error_is_not_found(&error) => return Ok(None),
        Err(error) => {
            return Err(format!(
                "open cached scheduler artifact {}: {error:#}",
                path.display()
            ));
        }
    };
    let snapshot = snapshot_pinned_scheduler_artifact(pinned)?;
    if snapshot.path() != path {
        return Err(format!(
            "cached scheduler artifact {} did not reopen as its canonical content object {}",
            path.display(),
            snapshot.path().display(),
        ));
    }
    Ok(Some(snapshot))
}

fn read_scheduler_build_cache_record(
    path: &Path,
    identity: u64,
    packages: &[String],
) -> Result<Option<CachedSchedulerWorkspaceArtifacts>, String> {
    let Some((mut file, file_identity)) =
        crate::cache::content::open_cache_record(path, "scheduler build cache record")
            .map_err(|error| error.to_string())?
    else {
        return Ok(None);
    };
    if file_identity.size > SCHEDULER_BUILD_RECORD_MAX_BYTES {
        return Err(format!(
            "scheduler build cache record {} is too large: {} bytes",
            path.display(),
            file_identity.size,
        ));
    }
    let mut bytes = Vec::with_capacity(file_identity.size as usize);
    file.read_to_end(&mut bytes).map_err(|error| {
        format!(
            "read scheduler build cache record {}: {error}",
            path.display()
        )
    })?;
    let after = crate::cache::content::StableFileIdentity::from_file(&file)
        .map_err(|error| error.to_string())?;
    if after != file_identity {
        return Err(format!(
            "scheduler build cache record changed while reading: {}",
            path.display()
        ));
    }
    let record: SchedulerBuildCacheRecord = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "parse scheduler build cache record {}: {error}",
            path.display()
        )
    })?;
    if record.version != SCHEDULER_BUILD_CACHE_SCHEMA {
        return Err(format!(
            "unsupported scheduler build cache record version {} in {}",
            record.version,
            path.display(),
        ));
    }
    if record.identity != identity {
        return Err(format!(
            "scheduler build cache identity collision in {}: expected {identity:016x}, got \
             {:016x}",
            path.display(),
            record.identity,
        ));
    }
    if record.packages != packages {
        return Err(format!(
            "scheduler build cache package-set collision in {}: expected {:?}, got {:?}",
            path.display(),
            packages,
            record.packages,
        ));
    }
    if record.artifacts.keys().ne(packages.iter()) {
        return Err(format!(
            "scheduler build cache artifact set in {} does not match packages {:?}",
            path.display(),
            packages,
        ));
    }
    let expected_integrity = scheduler_build_record_integrity(
        record.version,
        record.identity,
        &record.packages,
        &record.artifacts,
    );
    if record.integrity_ahash != expected_integrity {
        return Err(format!(
            "scheduler build cache record integrity mismatch: {}",
            path.display()
        ));
    }

    let mut paths = BTreeMap::new();
    let mut snapshots = Vec::with_capacity(record.artifacts.len());
    for (package, artifact) in record.artifacts {
        let Some(snapshot) = reopen_cached_scheduler_artifact(&artifact)? else {
            // Content GC may legally reclaim an object while no manifest
            // leases it. The immutable build record then becomes a cache miss;
            // the elected successor rebuilds and atomically replaces it.
            return Ok(None);
        };
        paths.insert(package, snapshot.path().to_path_buf());
        snapshots.push(snapshot);
    }
    Ok(Some(CachedSchedulerWorkspaceArtifacts {
        paths,
        snapshots,
        cache_hit: true,
    }))
}

fn publish_scheduler_build_cache_record(
    path: &Path,
    identity: u64,
    packages: &[String],
    artifacts: &BTreeMap<String, PathBuf>,
) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt as _;

    let mut record = SchedulerBuildCacheRecord {
        version: SCHEDULER_BUILD_CACHE_SCHEMA,
        identity,
        packages: packages.to_vec(),
        artifacts: artifacts.clone(),
        integrity_ahash: 0,
    };
    record.integrity_ahash = scheduler_build_record_integrity(
        record.version,
        record.identity,
        &record.packages,
        &record.artifacts,
    );
    let parent = path.parent().ok_or_else(|| {
        format!(
            "scheduler build cache path has no parent: {}",
            path.display()
        )
    })?;
    let mut temporary = tempfile::Builder::new()
        .prefix(&format!(".tmp-{identity:016x}-"))
        .tempfile_in(parent)
        .map_err(|error| {
            format!(
                "create scheduler build cache record temp in {}: {error}",
                parent.display()
            )
        })?;
    serde_json::to_writer(temporary.as_file_mut(), &record)
        .map_err(|error| format!("serialize scheduler build cache record: {error}"))?;
    temporary
        .as_file_mut()
        .flush()
        .map_err(|error| format!("flush scheduler build cache record: {error}"))?;
    temporary
        .as_file()
        .set_permissions(std::fs::Permissions::from_mode(0o444))
        .map_err(|error| format!("make scheduler build cache record immutable: {error}"))?;
    temporary.persist(path).map_err(|error| {
        format!(
            "atomically publish scheduler build cache record {}: {}",
            path.display(),
            error.error,
        )
    })?;
    Ok(())
}

#[derive(Clone, Copy)]
enum SchedulerBuildWaitState {
    Started,
    Heartbeat,
    Acquired,
}

#[derive(Clone, Copy)]
enum SchedulerBuildWaitKind {
    Producer,
    Successor,
}

fn format_scheduler_build_wait(
    label: &str,
    identity: u64,
    elapsed: std::time::Duration,
    kind: SchedulerBuildWaitKind,
    state: SchedulerBuildWaitState,
) -> String {
    let elapsed = humantime::format_duration(elapsed).to_string();
    match (kind, state) {
        (SchedulerBuildWaitKind::Producer, SchedulerBuildWaitState::Started) => format!(
            "{label}: waiting for input-addressed scheduler workspace build \
             {identity:016x}; elapsed={elapsed}"
        ),
        (SchedulerBuildWaitKind::Producer, SchedulerBuildWaitState::Heartbeat) => format!(
            "{label}: still waiting for input-addressed scheduler workspace build \
             {identity:016x}; elapsed={elapsed}"
        ),
        (SchedulerBuildWaitKind::Producer, SchedulerBuildWaitState::Acquired) => format!(
            "{label}: scheduler workspace build producer {identity:016x} finished; \
             validating reusable artifacts after {elapsed}"
        ),
        (SchedulerBuildWaitKind::Successor, SchedulerBuildWaitState::Started) => format!(
            "{label}: waiting to become successor for scheduler workspace build \
             {identity:016x}; elapsed={elapsed}"
        ),
        (SchedulerBuildWaitKind::Successor, SchedulerBuildWaitState::Heartbeat) => format!(
            "{label}: still waiting to become successor for scheduler workspace build \
             {identity:016x}; elapsed={elapsed}"
        ),
        (SchedulerBuildWaitKind::Successor, SchedulerBuildWaitState::Acquired) => format!(
            "{label}: acquired successor election for scheduler workspace build \
             {identity:016x} after {elapsed}"
        ),
    }
}

fn wait_for_scheduler_build_lock<C>(
    coordination: &mut crate::cache::content::CoordinationFile,
    label: &str,
    identity: u64,
    kind: SchedulerBuildWaitKind,
    cancelled: &C,
) -> anyhow::Result<()>
where
    C: Fn() -> bool,
{
    const RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);
    const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);
    let started = std::time::Instant::now();
    eprintln!(
        "{}",
        format_scheduler_build_wait(
            label,
            identity,
            std::time::Duration::ZERO,
            kind,
            SchedulerBuildWaitState::Started,
        )
    );
    let mut next_heartbeat = HEARTBEAT_INTERVAL;
    loop {
        if cancelled() {
            return Err(
                anyhow::Error::new(std::io::Error::from_raw_os_error(libc::EINTR))
                    .context("scheduler workspace build cache wait interrupted"),
            );
        }
        let acquired = match kind {
            SchedulerBuildWaitKind::Producer => coordination.try_lock_shared(),
            SchedulerBuildWaitKind::Successor => coordination.try_lock_exclusive(),
        };
        match acquired {
            Ok(()) => break,
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => {}
            Err(error) => return Err(error.into()),
        }
        let elapsed = started.elapsed();
        if elapsed >= next_heartbeat {
            eprintln!(
                "{}",
                format_scheduler_build_wait(
                    label,
                    identity,
                    elapsed,
                    kind,
                    SchedulerBuildWaitState::Heartbeat,
                )
            );
            next_heartbeat = elapsed + HEARTBEAT_INTERVAL;
        }
        std::thread::park_timeout(RETRY_INTERVAL.min(next_heartbeat.saturating_sub(elapsed)));
    }
    eprintln!(
        "{}",
        format_scheduler_build_wait(
            label,
            identity,
            started.elapsed(),
            kind,
            SchedulerBuildWaitState::Acquired,
        )
    );
    Ok(())
}

fn load_or_build_scheduler_workspace_artifacts_at_root<F, V, C>(
    root: &Path,
    identity: u64,
    packages: &[String],
    progress_label: &str,
    validate_identity: V,
    cancelled: C,
    build: F,
) -> Result<CachedSchedulerWorkspaceArtifacts, String>
where
    F: FnOnce() -> Result<BTreeMap<String, crate::cache::PinnedContentFile>, String>,
    V: Fn() -> Result<bool, String>,
    C: Fn() -> bool,
{
    if packages.is_empty()
        || packages.windows(2).any(|pair| pair[0] >= pair[1])
        || packages.iter().any(String::is_empty)
    {
        return Err(format!(
            "scheduler build cache packages must be non-empty, sorted, unique names: {packages:?}"
        ));
    }
    ensure_scheduler_build_cache_dirs(root).map_err(|error| error.to_string())?;
    let (record_path, lock_path, namespace_gate) = scheduler_build_cache_paths(root, identity);
    crate::cache::content::load_or_build_with_wait(
        &namespace_gate,
        &lock_path,
        &format!("scheduler workspace build {identity:016x}"),
        || match read_scheduler_build_cache_record(&record_path, identity, packages) {
            Ok(Some(record)) => {
                if !validate_identity().map_err(anyhow::Error::msg)? {
                    anyhow::bail!(
                        "scheduler workspace inputs changed before accepting cached build \
                         {identity:016x}"
                    );
                }
                Ok(Some(record))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                tracing::debug!(
                    path = %record_path.display(),
                    error = %error,
                    "rebuilding invalid reconstructible scheduler build cache record",
                );
                Ok(None)
            }
        },
        || {
            let pinned = build().map_err(anyhow::Error::msg)?;
            if pinned.keys().ne(packages.iter()) {
                anyhow::bail!(
                    "scheduler workspace builder emitted packages {:?}, expected {packages:?}",
                    pinned.keys().collect::<Vec<_>>(),
                );
            }
            let mut paths = BTreeMap::new();
            let mut snapshots = Vec::with_capacity(pinned.len());
            for (package, artifact) in pinned {
                let snapshot =
                    snapshot_pinned_scheduler_artifact(artifact).map_err(anyhow::Error::msg)?;
                paths.insert(package, snapshot.path().to_path_buf());
                snapshots.push(snapshot);
            }
            if !validate_identity().map_err(anyhow::Error::msg)? {
                anyhow::bail!(
                    "scheduler workspace inputs changed before publishing build \
                     {identity:016x}"
                );
            }
            publish_scheduler_build_cache_record(&record_path, identity, packages, &paths)
                .map_err(anyhow::Error::msg)?;
            Ok(CachedSchedulerWorkspaceArtifacts {
                paths,
                snapshots,
                cache_hit: false,
            })
        },
        |coordination| {
            wait_for_scheduler_build_lock(
                coordination,
                progress_label,
                identity,
                SchedulerBuildWaitKind::Producer,
                &cancelled,
            )
        },
        |coordination| {
            wait_for_scheduler_build_lock(
                coordination,
                progress_label,
                identity,
                SchedulerBuildWaitKind::Successor,
                &cancelled,
            )
        },
    )
    .map_err(|error| format!("{error:#}"))
}

/// Reuse or build one exact scheduler workspace output identity.
///
/// The key is supplied by cargo-ktstr's source/toolchain/environment
/// fingerprint planner. This function owns cross-process election and atomic
/// publication. Same-key consumers wait for one producer and then reacquire
/// ordinary immutable content-CAS leases; they never invoke the builder.
#[doc(hidden)]
pub fn load_or_build_scheduler_workspace_artifacts<F, V, C>(
    identity: u64,
    packages: &[String],
    progress_label: &str,
    validate_identity: V,
    cancelled: C,
    build: F,
) -> Result<CachedSchedulerWorkspaceArtifacts, String>
where
    F: FnOnce() -> Result<BTreeMap<String, crate::cache::PinnedContentFile>, String>,
    V: Fn() -> Result<bool, String>,
    C: Fn() -> bool,
{
    let root = crate::cache::scheduler_build_cache_root()
        .map_err(|error| format!("resolve scheduler build cache root: {error:#}"))?;
    load_or_build_scheduler_workspace_artifacts_at_root(
        &root,
        identity,
        packages,
        progress_label,
        validate_identity,
        cancelled,
        build,
    )
}

/// Atomically publish one immutable scheduler artifact manifest.
pub fn write_scheduler_artifact_manifest(
    directory: &Path,
    manifest: &SchedulerArtifactManifest,
) -> Result<PathBuf, String> {
    use std::os::unix::fs::PermissionsExt;

    let final_path = directory.join("scheduler-artifacts-v1.json");
    let mut temporary = tempfile::NamedTempFile::new_in(directory).map_err(|error| {
        format!(
            "create temporary scheduler artifact manifest in {}: {error}",
            directory.display()
        )
    })?;
    serde_json::to_writer_pretty(temporary.as_file_mut(), manifest)
        .map_err(|error| format!("serialize scheduler artifact manifest: {error}"))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| format!("sync scheduler artifact manifest: {error}"))?;
    temporary
        .as_file()
        .set_permissions(std::fs::Permissions::from_mode(0o444))
        .map_err(|error| format!("make scheduler artifact manifest read-only: {error}"))?;
    temporary.persist(&final_path).map_err(|error| {
        format!(
            "atomically install scheduler artifact manifest {}: {}",
            final_path.display(),
            error.error,
        )
    })?;
    std::fs::canonicalize(&final_path).map_err(|error| {
        format!(
            "canonicalize scheduler artifact manifest {}: {error}",
            final_path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    fn executable(dir: &Path) -> PathBuf {
        let path = dir.join("scx_sched");
        std::fs::write(&path, b"#!/bin/sh\nexit 0\n").expect("write executable");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o555))
            .expect("chmod executable");
        std::fs::canonicalize(path).expect("canonical executable")
    }

    fn write_manifest(
        dir: &Path,
        executable: &Path,
        profile: String,
    ) -> (PathBuf, SchedulerArtifactManifest) {
        let manifest = SchedulerArtifactManifest {
            version: SCHEDULER_ARTIFACT_MANIFEST_VERSION,
            profile,
            entries: vec![SchedulerArtifactEntry {
                binary: SchedulerArtifactSpec::Discover("scx_sched".into()),
                manifest_dir: "/workspace/member".into(),
                schedulers: vec!["sched".into()],
                path: executable.to_path_buf(),
            }],
        };
        let path = dir.join("manifest.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        (path, manifest)
    }

    #[test]
    fn absent_manifest_preserves_direct_resolution() {
        assert!(
            scheduler_artifact_from_manifest_path(
                None,
                None,
                &SchedulerArtifactSpec::Discover("scx_sched".into()),
                "/workspace/member",
            )
            .expect("absence is not an error")
            .is_none(),
        );
    }

    #[test]
    fn manifest_requires_exact_identity_and_optional_scheduler_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let executable = executable(dir.path());
        let (manifest_path, _) =
            write_manifest(dir.path(), &executable, crate::scheduler_profile_name());

        assert_eq!(
            scheduler_artifact_from_manifest_path(
                Some(manifest_path.as_os_str()),
                Some("sched"),
                &SchedulerArtifactSpec::Discover("scx_sched".into()),
                "/workspace/member",
            )
            .expect("exact manifest lookup"),
            Some(executable),
        );
        let wrong_dir = scheduler_artifact_from_manifest_path(
            Some(manifest_path.as_os_str()),
            None,
            &SchedulerArtifactSpec::Discover("scx_sched".into()),
            "/workspace/other",
        )
        .expect_err("manifest_dir is part of the exact identity");
        assert!(
            wrong_dir.to_string().contains("has no exact entry"),
            "unexpected error: {wrong_dir:#}"
        );
        let wrong_scheduler = scheduler_artifact_from_manifest_path(
            Some(manifest_path.as_os_str()),
            Some("other"),
            &SchedulerArtifactSpec::Discover("scx_sched".into()),
            "/workspace/member",
        )
        .expect_err("verifier scheduler validation is exact");
        assert!(
            wrong_scheduler.to_string().contains("was not prepared"),
            "unexpected error: {wrong_scheduler:#}"
        );
        let wrong_kind = scheduler_artifact_from_manifest_path(
            Some(manifest_path.as_os_str()),
            None,
            &SchedulerArtifactSpec::Path("scx_sched".into()),
            "/workspace/member",
        )
        .expect_err("Path and Discover identities never alias");
        assert!(
            wrong_kind.to_string().contains("has no exact entry"),
            "unexpected error: {wrong_kind:#}"
        );
    }

    #[test]
    fn immutable_manifest_cache_tracks_atomic_path_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = executable(dir.path());
        let second = dir.path().join("scx_sched_replacement");
        std::fs::write(&second, b"#!/bin/sh\nexit 1\n").expect("write replacement executable");
        std::fs::set_permissions(&second, std::fs::Permissions::from_mode(0o555))
            .expect("chmod replacement executable");
        let second = std::fs::canonicalize(second).expect("canonical replacement executable");
        let mut manifest = SchedulerArtifactManifest {
            version: SCHEDULER_ARTIFACT_MANIFEST_VERSION,
            profile: crate::scheduler_profile_name(),
            entries: vec![SchedulerArtifactEntry {
                binary: SchedulerArtifactSpec::Discover("scx_sched".into()),
                manifest_dir: "/workspace/member".into(),
                schedulers: vec!["sched".into()],
                path: first.clone(),
            }],
        };
        let manifest_path =
            write_scheduler_artifact_manifest(dir.path(), &manifest).expect("write first manifest");
        assert_eq!(
            scheduler_artifact_from_manifest_path(
                Some(manifest_path.as_os_str()),
                Some("sched"),
                &SchedulerArtifactSpec::Discover("scx_sched".into()),
                "/workspace/member",
            )
            .expect("resolve first immutable manifest"),
            Some(first),
        );

        manifest.entries[0].path = second.clone();
        assert_eq!(
            write_scheduler_artifact_manifest(dir.path(), &manifest)
                .expect("atomically replace immutable manifest"),
            manifest_path,
        );
        assert_eq!(
            scheduler_artifact_from_manifest_path(
                Some(manifest_path.as_os_str()),
                Some("sched"),
                &SchedulerArtifactSpec::Discover("scx_sched".into()),
                "/workspace/member",
            )
            .expect("resolve replacement immutable manifest"),
            Some(second),
            "the process-local parse memo must key the exact inode revision",
        );
    }

    #[test]
    fn lookup_only_stats_the_exact_artifact_but_still_validates_the_wire() {
        let dir = tempfile::tempdir().expect("tempdir");
        let executable = executable(dir.path());
        let (manifest_path, mut manifest) =
            write_manifest(dir.path(), &executable, crate::scheduler_profile_name());
        manifest.entries.push(SchedulerArtifactEntry {
            binary: SchedulerArtifactSpec::Discover("unrelated".into()),
            manifest_dir: "/workspace/other".into(),
            schedulers: vec!["other".into()],
            path: dir.path().join("missing-unrelated-artifact"),
        });
        std::fs::write(
            &manifest_path,
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        assert_eq!(
            scheduler_artifact_from_manifest_path(
                Some(manifest_path.as_os_str()),
                Some("sched"),
                &SchedulerArtifactSpec::Discover("scx_sched".into()),
                "/workspace/member",
            )
            .expect("unrelated artifact paths are not touched"),
            Some(executable),
        );

        manifest.entries[1].schedulers.clear();
        std::fs::write(
            &manifest_path,
            serde_json::to_vec(&manifest).expect("serialize malformed manifest"),
        )
        .expect("write malformed manifest");
        let error = scheduler_artifact_from_manifest_path(
            Some(manifest_path.as_os_str()),
            Some("sched"),
            &SchedulerArtifactSpec::Discover("scx_sched".into()),
            "/workspace/member",
        )
        .expect_err("structural validation remains manifest-wide");
        assert!(
            error.to_string().contains("no requiring scheduler names"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn present_manifest_never_degrades_to_fallback() {
        let dir = tempfile::tempdir().expect("tempdir");
        let malformed = dir.path().join("malformed.json");
        std::fs::write(&malformed, b"{").expect("write malformed manifest");
        let error = scheduler_artifact_from_manifest_path(
            Some(malformed.as_os_str()),
            None,
            &SchedulerArtifactSpec::Discover("pkg".into()),
            "/workspace",
        )
        .expect_err("malformed present manifest is fatal");
        assert!(
            error
                .to_string()
                .contains("parse scheduler artifact manifest"),
            "unexpected error: {error:#}"
        );

        let executable = executable(dir.path());
        let (missing_path, mut manifest) =
            write_manifest(dir.path(), &executable, crate::scheduler_profile_name());
        manifest.entries[0].path = dir.path().join("does-not-exist");
        std::fs::write(
            &missing_path,
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write missing artifact manifest");
        let error = scheduler_artifact_from_manifest_path(
            Some(missing_path.as_os_str()),
            None,
            &SchedulerArtifactSpec::Discover("scx_sched".into()),
            "/workspace/member",
        )
        .expect_err("missing artifact is fatal");
        assert!(
            error
                .to_string()
                .contains("canonicalize scheduler artifact"),
            "unexpected error: {error:#}"
        );

        manifest.entries[0].path = executable;
        manifest.profile = format!("{}-other", crate::scheduler_profile_name());
        std::fs::write(
            &missing_path,
            serde_json::to_vec(&manifest).expect("serialize mismatched manifest"),
        )
        .expect("write mismatched manifest");
        let error = scheduler_artifact_from_manifest_path(
            Some(missing_path.as_os_str()),
            None,
            &SchedulerArtifactSpec::Discover("scx_sched".into()),
            "/workspace/member",
        )
        .expect_err("profile mismatch is fatal");
        assert!(
            error.to_string().contains("was built with profile"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn snapshot_pins_bytes_and_inode_across_source_replacement() {
        let source_dir = tempfile::tempdir().expect("source tempdir");
        let source = source_dir.path().join("scheduler");
        std::fs::write(&source, b"old scheduler bytes").expect("write source");
        std::fs::set_permissions(&source, std::fs::Permissions::from_mode(0o755))
            .expect("chmod source");

        let snapshot = snapshot_scheduler_artifact(&source).expect("snapshot scheduler");
        let replacement = source_dir.path().join("replacement");
        std::fs::write(&replacement, b"new scheduler bytes").expect("write replacement");
        std::fs::set_permissions(&replacement, std::fs::Permissions::from_mode(0o755))
            .expect("chmod replacement");
        std::fs::rename(&replacement, &source).expect("atomically replace source");

        assert_eq!(
            std::fs::read(snapshot.path()).expect("read snapshot"),
            b"old scheduler bytes",
        );
        assert_eq!(
            std::fs::metadata(snapshot.path())
                .expect("stat snapshot")
                .permissions()
                .mode()
                & 0o777,
            0o555,
        );
        assert_ne!(
            std::fs::metadata(snapshot.path())
                .expect("stat snapshot")
                .ino(),
            std::fs::metadata(&source).expect("stat replacement").ino(),
            "the child artifact must remain a parent-owned immutable inode",
        );
    }

    fn write_scheduler_build_fixture(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).expect("write scheduler build fixture");
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .expect("chmod scheduler build fixture");
    }

    #[test]
    fn scheduler_build_cache_hit_skips_builder_and_reconstructible_damage_rebuilds() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let _environment = lock_env();
        let root = tempfile::tempdir().expect("scheduler build cache root");
        let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, root.path());
        let source = root.path().join("scheduler-source");
        write_scheduler_build_fixture(&source, b"scheduler-cache-fixture-v1");
        let packages = vec!["scx_cache_fixture".to_string()];
        let builds = AtomicUsize::new(0);

        let build = || {
            builds.fetch_add(1, Ordering::SeqCst);
            Ok(BTreeMap::from([(
                packages[0].clone(),
                crate::cache::pin_content_file(&source).expect("pin fixture"),
            )]))
        };
        let first = load_or_build_scheduler_workspace_artifacts_at_root(
            root.path(),
            0x51ceda7a,
            &packages,
            "cargo ktstr test",
            || Ok(true),
            || false,
            build,
        )
        .expect("cold scheduler workspace build");
        assert!(!first.cache_hit);
        let object = first.paths[&packages[0]].clone();
        drop(first);

        let hit = load_or_build_scheduler_workspace_artifacts_at_root(
            root.path(),
            0x51ceda7a,
            &packages,
            "cargo ktstr test",
            || Ok(true),
            || false,
            || -> Result<BTreeMap<String, crate::cache::PinnedContentFile>, String> {
                panic!("a valid cache hit must not invoke the builder")
            },
        )
        .expect("warm scheduler workspace build");
        assert!(hit.cache_hit);
        assert_eq!(hit.paths[&packages[0]], object);
        drop(hit);
        assert_eq!(builds.load(Ordering::SeqCst), 1);

        let (record, _, _) = scheduler_build_cache_paths(root.path(), 0x51ceda7a);
        std::fs::remove_file(&record).expect("remove immutable cache record");
        std::fs::write(&record, b"{").expect("write corrupt cache record");
        let rebuilt = load_or_build_scheduler_workspace_artifacts_at_root(
            root.path(),
            0x51ceda7a,
            &packages,
            "cargo ktstr test",
            || Ok(true),
            || false,
            build,
        )
        .expect("corrupt record rebuild");
        assert!(!rebuilt.cache_hit);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        drop(rebuilt);

        std::fs::remove_file(&object).expect("simulate content-CAS collection");
        let rebuilt = load_or_build_scheduler_workspace_artifacts_at_root(
            root.path(),
            0x51ceda7a,
            &packages,
            "cargo ktstr test",
            || Ok(true),
            || false,
            build,
        )
        .expect("stale record rebuild");
        assert!(!rebuilt.cache_hit);
        assert_eq!(builds.load(Ordering::SeqCst), 3);
        assert_eq!(
            std::fs::read(&rebuilt.paths[&packages[0]]).expect("read rebuilt object"),
            b"scheduler-cache-fixture-v1",
        );
    }

    #[test]
    fn scheduler_build_cache_rejects_hit_and_publication_input_drift() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        use std::cell::Cell;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let _environment = lock_env();
        let root = tempfile::tempdir().expect("scheduler build cache root");
        let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, root.path());
        let source = root.path().join("scheduler-source");
        write_scheduler_build_fixture(&source, b"scheduler-drift-fixture");
        let packages = vec!["scx_drift_fixture".to_string()];
        let builds = AtomicUsize::new(0);
        let drifted = Cell::new(false);
        let identity = 0xd11f7ed;

        let publication_error = match load_or_build_scheduler_workspace_artifacts_at_root(
            root.path(),
            identity,
            &packages,
            "cargo ktstr test",
            || Ok(!drifted.get()),
            || false,
            || {
                builds.fetch_add(1, Ordering::SeqCst);
                let pinned =
                    crate::cache::pin_content_file(&source).map_err(|error| error.to_string())?;
                drifted.set(true);
                Ok(BTreeMap::from([(packages[0].clone(), pinned)]))
            },
        ) {
            Ok(_) => panic!("mutation during a cold build must prevent publication"),
            Err(error) => error,
        };
        assert!(
            publication_error.contains("inputs changed before publishing"),
            "unexpected publication drift error: {publication_error}",
        );
        let (record, _, _) = scheduler_build_cache_paths(root.path(), identity);
        assert!(
            !record.exists(),
            "a drifted cold build must not publish a reusable record",
        );

        drifted.set(false);
        let warmable = load_or_build_scheduler_workspace_artifacts_at_root(
            root.path(),
            identity,
            &packages,
            "cargo ktstr test",
            || Ok(true),
            || false,
            || {
                builds.fetch_add(1, Ordering::SeqCst);
                Ok(BTreeMap::from([(
                    packages[0].clone(),
                    crate::cache::pin_content_file(&source).map_err(|error| error.to_string())?,
                )]))
            },
        )
        .expect("stable cold build");
        assert!(!warmable.cache_hit);
        drop(warmable);

        let hit_error = match load_or_build_scheduler_workspace_artifacts_at_root(
            root.path(),
            identity,
            &packages,
            "cargo ktstr test",
            || Ok(false),
            || false,
            || -> Result<BTreeMap<String, crate::cache::PinnedContentFile>, String> {
                panic!("a drifted cache hit must not invoke the old-key builder")
            },
        ) {
            Ok(_) => panic!("mutation before hit acceptance must reject stale artifacts"),
            Err(error) => error,
        };
        assert!(
            hit_error.contains("inputs changed before accepting cached build"),
            "unexpected hit drift error: {hit_error}",
        );
        assert_eq!(builds.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn scheduler_build_cache_wait_is_promptly_interruptible() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let root = tempfile::tempdir().expect("scheduler build cache root");
        ensure_scheduler_build_cache_dirs(root.path()).expect("create scheduler cache dirs");
        let packages = vec!["scx_interrupt_fixture".to_string()];
        let identity = 0x1a7e22;
        let (_, lock, namespace) = scheduler_build_cache_paths(root.path(), identity);
        let mut producer =
            crate::cache::content::open_coord_file(&namespace, &lock).expect("open producer lock");
        producer
            .try_lock_exclusive()
            .expect("hold scheduler producer election");

        let interrupted = Arc::new(AtomicBool::new(false));
        let wake = Arc::clone(&interrupted);
        let trigger = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(25));
            wake.store(true, Ordering::Release);
        });
        let started = std::time::Instant::now();
        let error = match load_or_build_scheduler_workspace_artifacts_at_root(
            root.path(),
            identity,
            &packages,
            "cargo ktstr test",
            || Ok(true),
            || interrupted.load(Ordering::Acquire),
            || -> Result<BTreeMap<String, crate::cache::PinnedContentFile>, String> {
                panic!("an interrupted waiter must not become the builder")
            },
        ) {
            Ok(_) => panic!("cancelled cache wait must not succeed"),
            Err(error) => error,
        };
        trigger.join().expect("interrupt trigger");
        assert!(
            error.contains("scheduler workspace build cache wait interrupted"),
            "unexpected interrupt error: {error}",
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "100ms parked retries must observe cancellation promptly",
        );
    }

    #[test]
    fn scheduler_build_cache_successor_wait_is_promptly_interruptible() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let root = tempfile::tempdir().expect("scheduler build cache root");
        ensure_scheduler_build_cache_dirs(root.path()).expect("create scheduler cache dirs");
        let identity = 0x5acce5502;
        let (_, lock, namespace) = scheduler_build_cache_paths(root.path(), identity);
        let mut producer =
            crate::cache::content::open_coord_file(&namespace, &lock).expect("open producer lock");
        producer
            .try_lock_exclusive()
            .expect("hold scheduler producer election");
        let mut successor =
            crate::cache::content::open_coord_file(&namespace, &lock).expect("open successor lock");

        let interrupted = Arc::new(AtomicBool::new(false));
        let wake = Arc::clone(&interrupted);
        let trigger = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(25));
            wake.store(true, Ordering::Release);
        });
        let started = std::time::Instant::now();
        let error = wait_for_scheduler_build_lock(
            &mut successor,
            "cargo ktstr test",
            identity,
            SchedulerBuildWaitKind::Successor,
            &|| interrupted.load(Ordering::Acquire),
        )
        .expect_err("cancelled successor election");
        trigger.join().expect("interrupt trigger");
        assert!(
            error
                .to_string()
                .contains("scheduler workspace build cache wait interrupted"),
            "unexpected successor interrupt error: {error:#}",
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "100ms parked retries must observe successor cancellation promptly",
        );
    }

    #[test]
    fn scheduler_build_cache_wait_messages_are_deterministic() {
        assert_eq!(
            format_scheduler_build_wait(
                "cargo ktstr test",
                0x123,
                std::time::Duration::from_secs(20),
                SchedulerBuildWaitKind::Producer,
                SchedulerBuildWaitState::Heartbeat,
            ),
            "cargo ktstr test: still waiting for input-addressed scheduler workspace build \
             0000000000000123; elapsed=20s",
        );
        assert_eq!(
            format_scheduler_build_wait(
                "cargo ktstr test",
                0x123,
                std::time::Duration::from_secs(20),
                SchedulerBuildWaitKind::Successor,
                SchedulerBuildWaitState::Heartbeat,
            ),
            "cargo ktstr test: still waiting to become successor for scheduler workspace build \
             0000000000000123; elapsed=20s",
        );
    }

    const SCHEDULER_BUILD_CHILD_TEST: &str =
        "scheduler_artifact::tests::scheduler_build_cache_cross_process_child";
    const SCHEDULER_BUILD_CHILD_ROOT: &str = "KTSTR_SCHEDULER_BUILD_CHILD_ROOT";
    const SCHEDULER_BUILD_CHILD_INDEX: &str = "KTSTR_SCHEDULER_BUILD_CHILD_INDEX";
    const SCHEDULER_BUILD_CHILD_READY: &str = "KTSTR_SCHEDULER_BUILD_CHILD_READY";
    const SCHEDULER_BUILD_CHILD_START: &str = "KTSTR_SCHEDULER_BUILD_CHILD_START";
    const SCHEDULER_BUILD_CHILD_COUNTER: &str = "KTSTR_SCHEDULER_BUILD_CHILD_COUNTER";
    const SCHEDULER_BUILD_CHILD_RESULTS: &str = "KTSTR_SCHEDULER_BUILD_CHILD_RESULTS";

    #[test]
    fn scheduler_build_cache_cross_process_child() {
        use std::io::Write as _;
        use std::os::unix::ffi::OsStrExt as _;

        let Some(root) = std::env::var_os(SCHEDULER_BUILD_CHILD_ROOT).map(PathBuf::from) else {
            return;
        };
        let index = std::env::var(SCHEDULER_BUILD_CHILD_INDEX)
            .expect("child index")
            .parse::<usize>()
            .expect("parse child index");
        let ready = PathBuf::from(
            std::env::var_os(SCHEDULER_BUILD_CHILD_READY).expect("child ready directory"),
        );
        let start =
            PathBuf::from(std::env::var_os(SCHEDULER_BUILD_CHILD_START).expect("child start lock"));
        let counter = PathBuf::from(
            std::env::var_os(SCHEDULER_BUILD_CHILD_COUNTER).expect("child build counter"),
        );
        let results = PathBuf::from(
            std::env::var_os(SCHEDULER_BUILD_CHILD_RESULTS).expect("child result directory"),
        );
        // SAFETY: this exact subprocess runs one selected test with one libtest
        // thread; no peer thread can concurrently read the process environment.
        unsafe {
            std::env::set_var(crate::KTSTR_CACHE_DIR_ENV, &root);
        }
        let start = std::fs::File::open(start).expect("open child start lock");
        std::fs::write(ready.join(index.to_string()), b"ready").expect("publish child ready");
        rustix::fs::flock(&start, rustix::fs::FlockOperation::LockShared)
            .expect("wait for child start");

        let packages = vec!["scx_cross_process".to_string()];
        let result = load_or_build_scheduler_workspace_artifacts_at_root(
            &root,
            0xc2055cace,
            &packages,
            "cargo ktstr test",
            || Ok(true),
            || false,
            || {
                let mut attempts = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&counter)
                    .map_err(|error| error.to_string())?;
                writeln!(attempts, "builder-{index}").map_err(|error| error.to_string())?;
                attempts.sync_all().map_err(|error| error.to_string())?;
                let source = root.join(format!("scheduler-source-{index}"));
                write_scheduler_build_fixture(&source, b"cross-process-scheduler-bytes");
                Ok(BTreeMap::from([(
                    packages[0].clone(),
                    crate::cache::pin_content_file(source).map_err(|error| error.to_string())?,
                )]))
            },
        )
        .expect("cross-process scheduler build cache");
        std::fs::write(
            results.join(index.to_string()),
            result.paths[&packages[0]].as_os_str().as_bytes(),
        )
        .expect("write child cache result");
    }

    #[test]
    fn scheduler_build_cache_elects_one_cross_process_builder() {
        const CHILDREN: usize = 8;

        let temp = tempfile::tempdir().expect("cross-process scheduler cache tempdir");
        let root = temp.path().join("cache");
        let ready = temp.path().join("ready");
        let results = temp.path().join("results");
        std::fs::create_dir_all(&root).expect("create scheduler cache root");
        std::fs::create_dir(&ready).expect("create ready directory");
        std::fs::create_dir(&results).expect("create result directory");
        let start_path = temp.path().join("start");
        let start = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&start_path)
            .expect("create start lock");
        rustix::fs::flock(&start, rustix::fs::FlockOperation::LockExclusive)
            .expect("lock start barrier");
        let counter = temp.path().join("builds");
        let executable = std::env::current_exe().expect("current test executable");
        let mut children = (0..CHILDREN)
            .map(|index| {
                std::process::Command::new(&executable)
                    .arg("--exact")
                    .arg(SCHEDULER_BUILD_CHILD_TEST)
                    .arg("--nocapture")
                    .arg("--test-threads=1")
                    .env(SCHEDULER_BUILD_CHILD_ROOT, &root)
                    .env(SCHEDULER_BUILD_CHILD_INDEX, index.to_string())
                    .env(SCHEDULER_BUILD_CHILD_READY, &ready)
                    .env(SCHEDULER_BUILD_CHILD_START, &start_path)
                    .env(SCHEDULER_BUILD_CHILD_COUNTER, &counter)
                    .env(SCHEDULER_BUILD_CHILD_RESULTS, &results)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::inherit())
                    .spawn()
                    .expect("spawn scheduler cache child")
            })
            .collect::<Vec<_>>();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while std::fs::read_dir(&ready)
            .expect("read ready directory")
            .count()
            != CHILDREN
        {
            for child in &mut children {
                if let Some(status) = child.try_wait().expect("poll child") {
                    panic!("scheduler cache child exited before barrier: {status}");
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "scheduler cache children did not reach start barrier"
            );
            std::thread::yield_now();
        }
        rustix::fs::flock(&start, rustix::fs::FlockOperation::Unlock)
            .expect("release start barrier");
        for child in &mut children {
            assert!(child.wait().expect("wait for cache child").success());
        }
        let attempts = std::fs::read_to_string(counter).expect("read build counter");
        assert_eq!(
            attempts.lines().count(),
            1,
            "all same-input processes must share one scheduler Cargo producer"
        );
        let mut published = std::fs::read_dir(results)
            .expect("read cache results")
            .map(|entry| {
                std::fs::read(entry.expect("result entry").path()).expect("read cache result")
            })
            .collect::<Vec<_>>();
        published.sort();
        published.dedup();
        assert_eq!(
            published.len(),
            1,
            "every process must lease the same immutable content-CAS object"
        );
    }
}
