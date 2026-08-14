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
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// Current schema version for [`SchedulerArtifactManifest`].
pub const SCHEDULER_ARTIFACT_MANIFEST_VERSION: u32 = 1;

const SCHEDULER_ARTIFACT_TREE_NAMESPACE: &str = "scheduler-workspace-v1";

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

    let snapshot = crate::cache::snapshot_pinned_artifact_file(pinned).map_err(|error| {
        format!(
            "publish pinned scheduler artifact {} in shared content cache: {error:#}",
            source_path.display()
        )
    })?;
    Ok(SchedulerArtifactSnapshot { snapshot })
}

/// One input-addressed scheduler workspace build result.
///
/// Every path names a private COW materialization of the generic Cargo
/// artifact-tree cache. Keeping `tree` alive keeps both that materialization
/// and its immutable content-object leases alive through the child run. A
/// cache hit never invokes Cargo or enters the target-directory/build leases.
#[doc(hidden)]
pub struct CachedSchedulerWorkspaceArtifacts {
    pub paths: BTreeMap<String, PathBuf>,
    pub tree: crate::cache::artifact_tree::MaterializedArtifactTree,
    pub cache_hit: bool,
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
    let cache = crate::cache::artifact_tree::ArtifactTreeCache::new(
        root.join(SCHEDULER_ARTIFACT_TREE_NAMESPACE),
    );
    let materializations = root.join("materialized");
    let tree = cache
        .load_or_build(
            identity,
            &materializations,
            progress_label,
            || validate_identity().map_err(anyhow::Error::msg),
            &cancelled,
            || {
                let pinned = build().map_err(anyhow::Error::msg)?;
                anyhow::ensure!(
                    pinned.keys().eq(packages.iter()),
                    "scheduler workspace builder emitted packages {:?}, expected {packages:?}",
                    pinned.keys().collect::<Vec<_>>(),
                );
                let mut source = crate::cache::artifact_tree::ArtifactTreeSource::new();
                for (package, artifact) in pinned {
                    source.insert_immutable_pinned_file(
                        Path::new("artifacts").join(package),
                        artifact,
                    )?;
                }
                Ok(source)
            },
        )
        .map_err(|error| format!("{error:#}"))?;
    let paths = packages
        .iter()
        .map(|package| (package.clone(), tree.root().join("artifacts").join(package)))
        .collect();
    let cache_hit = tree.cache_hit();
    if let Err(error) = tree.persist_decision_diagnostic(
        "scheduler-workspace",
        serde_json::json!({
            "namespace": SCHEDULER_ARTIFACT_TREE_NAMESPACE,
            "normalized_inputs_digest": format!("{identity:016x}"),
            "packages": packages,
        }),
    ) {
        tracing::warn!(error = %error, "could not persist scheduler artifact-cache decision");
    }
    Ok(CachedSchedulerWorkspaceArtifacts {
        paths,
        tree,
        cache_hit,
    })
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
    let root = crate::cache::cargo_artifact_tree_cache_root()
        .map_err(|error| format!("resolve Cargo artifact tree cache root: {error:#}"))?;
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

/// Build scheduler workspaces through the same deterministic stable-output
/// producer used by ordinary, coverage, and verifier harness closures.
///
/// Scheduler binaries may embed `OUT_DIR`, so the Cargo child must write at
/// the final persistent pathname supplied to `build`; execution still uses a
/// private reflink materialization.
#[doc(hidden)]
pub fn load_or_build_scheduler_workspace_artifacts_stable<F, V, C>(
    identity: u64,
    packages: &[String],
    progress_label: &str,
    validate_identity: V,
    cancelled: C,
    build: F,
) -> Result<CachedSchedulerWorkspaceArtifacts, String>
where
    F: FnOnce(
        &crate::cache::artifact_tree::StableCargoBuild,
    ) -> Result<BTreeMap<String, crate::cache::PinnedContentFile>, String>,
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
    let root = crate::cache::cargo_artifact_tree_cache_root()
        .map_err(|error| format!("resolve Cargo artifact tree cache root: {error:#}"))?;
    let cache = crate::cache::artifact_tree::ArtifactTreeCache::new(
        root.join(SCHEDULER_ARTIFACT_TREE_NAMESPACE),
    );
    let tree = cache
        .load_or_build_with_stable_cargo_output(
            identity,
            &root.join("stable-builds-v1"),
            &root.join("materialized"),
            progress_label,
            || validate_identity().map_err(anyhow::Error::msg),
            cancelled,
            |stable| {
                let pinned = build(stable).map_err(anyhow::Error::msg)?;
                anyhow::ensure!(
                    pinned.keys().eq(packages.iter()),
                    "scheduler workspace builder emitted packages {:?}, expected {packages:?}",
                    pinned.keys().collect::<Vec<_>>(),
                );
                let mut source = crate::cache::artifact_tree::ArtifactTreeSource::new();
                for (package, artifact) in pinned {
                    let stable_path = artifact.source_path().to_path_buf();
                    let stable_relative = stable_path.strip_prefix(&stable.root).map_err(|_| {
                        anyhow::anyhow!(
                            "scheduler artifact {} is outside stable Cargo output {}",
                            stable_path.display(),
                            stable.root.display(),
                        )
                    })?;
                    anyhow::ensure!(
                        stable_relative.starts_with("target"),
                        "scheduler artifact {} is outside stable Cargo target {}",
                        stable_path.display(),
                        stable.target_directory.display(),
                    );
                    source.insert_immutable_pinned_file(
                        Path::new("artifacts").join(package),
                        artifact,
                    )?;
                    // Preserve the exact absolute Cargo output pathname as
                    // part of the generic closure. Scheduler binaries may
                    // embed OUT_DIR/CARGO_BIN_EXE paths even though consumers
                    // execute the convenient `artifacts/<package>` alias.
                    source.insert_immutable_path(stable_relative, &stable_path)?;
                }
                // The build closure (dependency objects and build-script
                // OUT_DIRs) is deliberately not sealed here. Cargo now writes it
                // to a persistent, non-source-keyed directory shared across
                // digests for dependency and OUT_DIR reuse, and no scheduler
                // consumer reads it: production snapshots the `artifacts/<pkg>`
                // alias into the machine content CAS and executes from there.
                // scx-ktstr's only OUT_DIR use is a compile-time `include!`, so
                // nothing resolves the build closure at runtime.
                Ok(source)
            },
        )
        .map_err(|error| format!("{error:#}"))?;
    let paths = packages
        .iter()
        .map(|package| (package.clone(), tree.root().join("artifacts").join(package)))
        .collect();
    let cache_hit = tree.cache_hit();
    if let Err(error) = tree.persist_decision_diagnostic(
        "scheduler-workspace",
        serde_json::json!({
            "namespace": SCHEDULER_ARTIFACT_TREE_NAMESPACE,
            "normalized_inputs_digest": format!("{identity:016x}"),
            "packages": packages,
        }),
    ) {
        tracing::warn!(error = %error, "could not persist scheduler artifact-cache decision");
    }
    Ok(CachedSchedulerWorkspaceArtifacts {
        paths,
        tree,
        cache_hit,
    })
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
    fn snapshot_publishes_across_filesystems_via_byte_copy() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _environment = lock_env();
        // /dev/shm is tmpfs on every supported host: it can never reflink
        // (no remap_file_range) and is a different mount from the cache
        // tempdir below, so publication from here always exercises the
        // cross-filesystem byte-copy fallback — the shape a user hits when
        // their project checkout and KTSTR_CACHE_DIR live on different
        // mounts.
        let source_dir = tempfile::tempdir_in("/dev/shm").expect("tmpfs source tempdir");
        let cache_dir = tempfile::tempdir().expect("cache tempdir");
        let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, cache_dir.path());
        let source = source_dir.path().join("scheduler");
        std::fs::write(&source, b"cross-fs scheduler bytes").expect("write source");
        std::fs::set_permissions(&source, std::fs::Permissions::from_mode(0o755))
            .expect("chmod source");

        let snapshot = snapshot_scheduler_artifact(&source).expect("snapshot across filesystems");
        assert_eq!(
            std::fs::read(snapshot.path()).expect("read snapshot"),
            b"cross-fs scheduler bytes",
        );
        assert_eq!(
            std::fs::metadata(snapshot.path())
                .expect("stat snapshot")
                .permissions()
                .mode()
                & 0o7777,
            0o555,
            "byte-copied publication must land with the same immutable mode \
             as a reflinked one",
        );
    }

    #[test]
    fn snapshot_pins_bytes_and_inode_across_source_replacement() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _environment = lock_env();
        let source_dir = tempfile::tempdir().expect("source tempdir");
        let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, source_dir.path().join("cache"));
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

    #[test]
    fn scheduler_workspace_build_uses_generic_artifact_tree_and_skips_warm_builder() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let root = tempfile::tempdir().expect("scheduler artifact-tree root");
        let source = root.path().join("scheduler-source");
        std::fs::write(&source, b"generic-scheduler-tree").expect("write scheduler fixture");
        std::fs::set_permissions(&source, std::fs::Permissions::from_mode(0o755))
            .expect("chmod scheduler fixture");
        let packages = vec!["scx_generic_cache".to_string()];
        let builds = AtomicUsize::new(0);
        let first = load_or_build_scheduler_workspace_artifacts_at_root(
            root.path(),
            0x5ced_7aee,
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
        .expect("cold scheduler artifact tree");
        assert!(!first.cache_hit);
        assert_eq!(
            std::fs::read(&first.paths[&packages[0]]).expect("read cold scheduler artifact"),
            b"generic-scheduler-tree",
        );
        drop(first);

        let hit = load_or_build_scheduler_workspace_artifacts_at_root(
            root.path(),
            0x5ced_7aee,
            &packages,
            "cargo ktstr test",
            || Ok(true),
            || false,
            || -> Result<BTreeMap<String, crate::cache::PinnedContentFile>, String> {
                panic!("a generic artifact-tree hit must not invoke the Cargo builder")
            },
        )
        .expect("warm scheduler artifact tree");
        assert!(hit.cache_hit);
        assert_eq!(
            std::fs::read(&hit.paths[&packages[0]]).expect("read warm scheduler artifact"),
            b"generic-scheduler-tree",
        );
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    /// The sealed per-source tree preserves the scheduler binary at both the
    /// convenient `artifacts/<pkg>` alias and its exact Cargo output pathname,
    /// but deliberately does **not** preserve the Cargo build closure
    /// (dependency objects, build-script `OUT_DIR`s).
    ///
    /// Production consumes only the `artifacts/<pkg>` alias — it snapshots that
    /// executable into the machine content CAS and executes from there, never
    /// resolving the build closure at runtime (scx-ktstr's sole `OUT_DIR` use is
    /// a compile-time `include!`). The build closure now lives in a persistent,
    /// non-source-keyed directory shared across source digests so Cargo reuses
    /// unchanged dependencies; sealing a per-source copy of it would be dead
    /// weight. Even when the producer writes an `OUT_DIR` payload under the
    /// stable root, distillation retains only `target/`.
    #[test]
    fn stable_scheduler_workspace_seals_target_alias_but_not_build_closure() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _environment = lock_env();
        let temp = tempfile::tempdir().unwrap();
        let cache_root = temp.path().join("cache");
        let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, &cache_root);
        let identity = 0x5ced_57ab;
        let packages = vec!["scx_stable_cache".to_string()];

        let cached = load_or_build_scheduler_workspace_artifacts_stable(
            identity,
            &packages,
            "stable scheduler artifact-tree test",
            || Ok(true),
            || false,
            |stable| {
                let scheduler = stable.target_directory.join("release/scx_stable_cache");
                let out_file = stable.root.join("build/scx-stable/out/generated");
                std::fs::create_dir_all(scheduler.parent().unwrap())
                    .map_err(|error| error.to_string())?;
                std::fs::create_dir_all(out_file.parent().unwrap())
                    .map_err(|error| error.to_string())?;
                std::fs::write(&scheduler, b"stable scheduler")
                    .map_err(|error| error.to_string())?;
                std::fs::set_permissions(&scheduler, std::fs::Permissions::from_mode(0o755))
                    .map_err(|error| error.to_string())?;
                std::fs::write(&out_file, b"embedded OUT_DIR payload")
                    .map_err(|error| error.to_string())?;
                let junk = stable.target_directory.join("release/deps/unrecorded");
                std::fs::create_dir_all(&junk).map_err(|error| error.to_string())?;
                std::fs::write(junk.join("junk"), b"discard").map_err(|error| error.to_string())?;
                Ok(BTreeMap::from([(
                    packages[0].clone(),
                    crate::cache::pin_content_file(&scheduler)
                        .map_err(|error| error.to_string())?,
                )]))
            },
        )
        .unwrap();

        assert_eq!(
            std::fs::read(&cached.paths[&packages[0]]).unwrap(),
            b"stable scheduler",
        );
        assert_eq!(
            std::fs::read(cached.tree.root().join("target/release/scx_stable_cache")).unwrap(),
            b"stable scheduler",
        );
        // The build closure is intentionally not part of the sealed tree.
        assert!(
            !cached.tree.root().join("build").exists(),
            "the build closure must not be preserved in the materialized tree",
        );
        let stable_root = cache_root
            .join("stable-builds-v1")
            .join(format!("{identity:016x}"));
        assert!(
            stable_root
                .join("target/release/scx_stable_cache")
                .is_file()
        );
        // Distillation retains only `target/`; any `build/` the producer wrote
        // under the stable root is discarded, along with unrecorded deps.
        assert!(
            !stable_root.join("build").exists(),
            "distillation must not seal the build closure under the stable root",
        );
        assert!(!stable_root.join("target/release/deps/unrecorded").exists());
    }
}
