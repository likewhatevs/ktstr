//! Relocatable, content-addressed nextest build closures.
//!
//! One cold producer captures nextest's exact binary metadata, the files that
//! metadata can execute/load, Cargo metadata, and ktstr's decoded scheduler /
//! admission stamps. Every later process reconstructs a private FICLONE tree
//! and drives nextest's supported reuse-build interface without entering
//! Cargo or reopening the large test ELFs.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::hash::{BuildHasher as _, Hasher as _};
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const IDENTITY_SCHEMA: u32 = 9;
const SOURCE_IDENTITY_SCHEMA: u32 = 5;
const STAMP_SCHEMA: u32 = 1;
const CARGO_METADATA_PATH: &str = "meta/cargo-metadata.json";
const BINARIES_METADATA_PATH: &str = "meta/binaries-metadata.json";
const KTSTR_STAMPS_PATH: &str = "meta/ktstr-scheduler-admission.json";
const COVERAGE_PRODUCER_PROFDATA_PATH: &str = "meta/coverage-producer.profdata";
const METADATA_MAX_BYTES: usize = 64 << 20;

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct BinaryMetadata {
    rust_build_meta: BuildMetadata,
    rust_binaries: BTreeMap<String, TestBinary>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct BuildMetadata {
    target_directory: PathBuf,
    #[serde(default)]
    build_directory: Option<PathBuf>,
    #[serde(default)]
    base_output_directories: BTreeSet<PathBuf>,
    #[serde(default)]
    non_test_binaries: BTreeMap<String, BTreeSet<NonTestBinary>>,
    #[serde(default)]
    build_script_out_dirs: BTreeMap<String, PathBuf>,
    #[serde(default)]
    linked_paths: BTreeSet<PathBuf>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct TestBinary {
    binary_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct NonTestBinary {
    path: PathBuf,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CachedStampSet {
    version: u32,
    binaries: Vec<CachedBinaryStamp>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct CachedBinaryStamp {
    /// Path relative to the materialized build directory.
    pub relative_binary: PathBuf,
    pub manifest: ktstr::test_support::SchedulerManifestProbe,
}

pub(crate) struct MaterializedNextestArtifacts {
    pub tree: ktstr::cache::artifact_tree::MaterializedArtifactTree,
    /// Immutable, deterministic source root used only while building and for
    /// interpreting build metadata.
    pub stable_workspace_root: PathBuf,
    /// Immutable invocation directory paired with `stable_workspace_root`.
    pub stable_invocation_root: PathBuf,
    /// The caller's writable workspace to which nextest remaps test CWDs.
    pub writable_workspace_root: PathBuf,
    /// The caller's writable invocation directory for the nextest process.
    pub writable_invocation_root: PathBuf,
    pub cargo_metadata: PathBuf,
    pub binaries_metadata: PathBuf,
    pub target_directory: PathBuf,
    pub build_directory: PathBuf,
    /// Producer/build-script coverage compacted before cache publication.
    ///
    /// This is deliberately outside `target`: cargo-llvm-cov must not mistake
    /// it for the final workspace-named profile, and no raw profile is allowed
    /// into the immutable artifact record or content CAS.
    pub producer_profdata: Option<PathBuf>,
    pub test_binaries: Vec<PathBuf>,
    pub loader_paths: Vec<PathBuf>,
    pub scheduler_stamps: Vec<CachedBinaryStamp>,
    // Retain the stable source owner until the nextest child (and a possible
    // llvm-cov report child) has exited.
    _stable_source: StableCargoSource,
}

pub(crate) use ktstr::cache::artifact_tree::StableCargoBuild;

impl MaterializedNextestArtifacts {
    pub fn cache_hit(&self) -> bool {
        self.tree.cache_hit()
    }

    /// Install the complete instrumented closure in a persistent coverage
    /// recovery bundle.
    ///
    /// `--no-report` must retain more than raw counters: a later
    /// `cargo llvm-cov report` also needs the exact instrumented binaries and
    /// build layout which produced them. All regular files remain private
    /// FICLONE inodes; this moves the already-materialized tree without
    /// copying its contents.
    pub(crate) fn persist_for_coverage(
        self,
        bundle_root: &Path,
    ) -> Result<(PathBuf, PathBuf), String> {
        let materialization_root = self.tree.root().to_path_buf();
        let target_relative = self
            .target_directory
            .strip_prefix(&materialization_root)
            .map_err(|_| {
                format!(
                    "cached coverage target {} is outside materialization {}",
                    self.target_directory.display(),
                    materialization_root.display(),
                )
            })?
            .to_path_buf();
        let build_relative = self
            .build_directory
            .strip_prefix(&materialization_root)
            .map_err(|_| {
                format!(
                    "cached coverage build directory {} is outside materialization {}",
                    self.build_directory.display(),
                    materialization_root.display(),
                )
            })?
            .to_path_buf();
        let destination = bundle_root.join("artifacts");
        self.tree.persist_at(&destination).map_err(|error| {
            format!(
                "persist cached coverage artifact closure at {}: {error:#}",
                destination.display(),
            )
        })?;
        Ok((
            destination.join(target_relative),
            destination.join(build_relative),
        ))
    }

    pub(crate) fn remap_cargo_args(&self, arguments: &[String]) -> Vec<String> {
        self._stable_source.remap_cargo_args(arguments)
    }

    /// Apply the runtime half of a reused build to a nextest command.
    ///
    /// Cargo compilation and metadata capture use the immutable stable roots;
    /// tests execute from the caller's checkout. The exported root map lets
    /// runtime helpers repair compile-time source paths for the workspace and
    /// every linked local dependency without making shared snapshots writable.
    pub(crate) fn apply_execution_context(
        &self,
        command: &mut std::process::Command,
    ) -> Result<(), String> {
        apply_execution_context(
            command,
            &self._stable_source.execution_source_root_remaps(),
            &self.writable_invocation_root,
        )
    }

    /// Nextest's supported reuse-build remapping arguments for this private
    /// COW closure. Callers append run/filter/report arguments separately.
    pub fn reuse_build_args(&self) -> Vec<String> {
        let mut args = vec![
            "--cargo-metadata".to_string(),
            self.cargo_metadata.display().to_string(),
            "--binaries-metadata".to_string(),
            self.binaries_metadata.display().to_string(),
            "--workspace-remap".to_string(),
            self.writable_workspace_root.display().to_string(),
            "--target-dir-remap".to_string(),
            self.target_directory.display().to_string(),
        ];
        if self.build_directory != self.target_directory {
            args.extend([
                "--build-dir-remap".to_string(),
                self.build_directory.display().to_string(),
            ]);
        }
        args
    }
}

fn apply_execution_context(
    command: &mut std::process::Command,
    source_root_remaps: &[(PathBuf, PathBuf)],
    writable_invocation_root: &Path,
) -> Result<(), String> {
    let encoded = ktstr::encode_source_root_remaps(source_root_remaps)?;
    command
        .current_dir(writable_invocation_root)
        .env(ktstr::KTSTR_SOURCE_ROOT_REMAPS_ENV, encoded);
    Ok(())
}

fn checked_relative(path: &Path, what: &str) -> Result<PathBuf, String> {
    if path.as_os_str().is_empty()
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "{what} is not a normalized relative path: {}",
            path.display()
        ));
    }
    Ok(path.to_path_buf())
}

fn build_directory(metadata: &BinaryMetadata) -> &Path {
    metadata
        .rust_build_meta
        .build_directory
        .as_deref()
        .unwrap_or(&metadata.rust_build_meta.target_directory)
}

fn build_prefix(metadata: &BinaryMetadata) -> &'static Path {
    if build_directory(metadata) == metadata.rust_build_meta.target_directory {
        Path::new("target")
    } else {
        Path::new("build")
    }
}

fn map_absolute_under(
    path: &Path,
    source_root: &Path,
    destination_root: &Path,
    what: &str,
) -> Result<PathBuf, String> {
    if !path.is_absolute() || !source_root.is_absolute() {
        return Err(format!(
            "{what} and its metadata root must be absolute: {} under {}",
            path.display(),
            source_root.display(),
        ));
    }
    let relative = path.strip_prefix(source_root).map_err(|_| {
        format!(
            "{what} {} is outside metadata root {}",
            path.display(),
            source_root.display(),
        )
    })?;
    Ok(destination_root.join(checked_relative(relative, what)?))
}

fn add_unique_file(
    files: &mut BTreeMap<PathBuf, PathBuf>,
    relative: PathBuf,
    source: PathBuf,
) -> Result<(), String> {
    match files.entry(relative) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(source);
        }
        std::collections::btree_map::Entry::Occupied(entry) if entry.get() == &source => {}
        std::collections::btree_map::Entry::Occupied(entry) => {
            return Err(format!(
                "nextest artifact path {} aliases both {} and {}",
                entry.key().display(),
                entry.get().display(),
                source.display(),
            ));
        }
    }
    Ok(())
}

fn insert_depth_one(
    source: &mut ktstr::cache::artifact_tree::ArtifactTreeSource,
    relative: &Path,
    path: &Path,
) -> Result<(), String> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "inspect nextest loader path {}: {error}",
                path.display()
            ));
        }
    };
    source
        .insert_path(relative, path)
        .map_err(|error| format!("capture nextest loader path {}: {error:#}", path.display()))?;
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        let mut entries = std::fs::read_dir(path)
            .map_err(|error| format!("read nextest loader directory {}: {error}", path.display()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("read nextest loader entry in {}: {error}", path.display()))?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            source
                .insert_path(relative.join(entry.file_name()), entry.path())
                .map_err(|error| {
                    format!(
                        "capture nextest loader entry {}: {error:#}",
                        entry.path().display()
                    )
                })?;
        }
    }
    Ok(())
}

fn insert_structural_directory(
    source: &mut ktstr::cache::artifact_tree::ArtifactTreeSource,
    relative: &Path,
    path: &Path,
    what: &str,
) -> Result<(), String> {
    let directory = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {what} {}: {error}", path.display()))?;
    if !directory.is_dir() || directory.file_type().is_symlink() {
        return Err(format!("{what} is not a directory: {}", path.display()));
    }
    source
        .insert_directory(relative, directory.permissions().mode() & 0o7777)
        .map_err(|error| format!("capture {what} {}: {error:#}", path.display()))
}

fn read_stamps_parallel(binaries: &[(PathBuf, PathBuf)]) -> Result<Vec<CachedBinaryStamp>, String> {
    if binaries.is_empty() {
        return Ok(Vec::new());
    }
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(16)
        .min(binaries.len());
    let next = AtomicUsize::new(0);
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let sender = sender.clone();
            let next = &next;
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some((relative, binary)) = binaries.get(index) else {
                        break;
                    };
                    let result =
                        ktstr::test_support::read_scheduler_manifest_and_validate_admission_stamp(
                            binary,
                        )
                        .map(|manifest| {
                            manifest.map(|manifest| CachedBinaryStamp {
                                relative_binary: relative.clone(),
                                manifest,
                            })
                        });
                    if sender.send((index, result)).is_err() {
                        break;
                    }
                }
            });
        }
    });
    drop(sender);
    let mut results = receiver.into_iter().collect::<Vec<_>>();
    results.sort_by_key(|(index, _)| *index);
    results
        .into_iter()
        .filter_map(|(_, result)| match result {
            Ok(Some(stamp)) => Some(Ok(stamp)),
            Ok(None) => None,
            Err(error) => Some(Err(error)),
        })
        .collect()
}

/// Pin the exact closure described by one successful nextest binary listing.
///
/// Call this while holding exclusive ownership of the producer target/build
/// directories. Every regular file is opened before that ownership is
/// released, so later Cargo atomic replacements cannot retarget publication.
pub(crate) fn capture_source(
    binaries_metadata: &[u8],
    cargo_metadata: &[u8],
) -> Result<ktstr::cache::artifact_tree::ArtifactTreeSource, String> {
    capture_source_with_producer_profdata(binaries_metadata, cargo_metadata, None)
}

/// Capture a coverage producer's already-merged build-script/proc-macro
/// profile beside the reusable target closure.
///
/// Raw producer shards are intentionally excluded. The producer compacts them
/// before calling this function, so neither the stable Cargo tree nor the
/// content CAS can retain one raw file per compiler/build-script process.
pub(crate) fn capture_source_with_producer_profdata(
    binaries_metadata: &[u8],
    cargo_metadata: &[u8],
    producer_profdata: Option<&Path>,
) -> Result<ktstr::cache::artifact_tree::ArtifactTreeSource, String> {
    if binaries_metadata.len() > METADATA_MAX_BYTES || cargo_metadata.len() > METADATA_MAX_BYTES {
        return Err("nextest reuse metadata exceeds the 64 MiB safety limit".to_string());
    }
    let metadata: BinaryMetadata = serde_json::from_slice(binaries_metadata)
        .map_err(|error| format!("parse nextest binaries metadata: {error}"))?;
    let target = &metadata.rust_build_meta.target_directory;
    let build = build_directory(&metadata);
    if !target.is_absolute() || !build.is_absolute() {
        return Err(format!(
            "nextest target/build directories must be absolute: target={}, build={}",
            target.display(),
            build.display(),
        ));
    }
    let build_prefix = build_prefix(&metadata);
    let mut files = BTreeMap::new();
    let mut test_binaries = Vec::new();
    for binary in metadata.rust_binaries.values() {
        let relative = map_absolute_under(&binary.binary_path, build, build_prefix, "test binary")?;
        add_unique_file(&mut files, relative.clone(), binary.binary_path.clone())?;
        test_binaries.push((
            relative
                .strip_prefix(build_prefix)
                .expect("mapped build path has build prefix")
                .to_path_buf(),
            binary.binary_path.clone(),
        ));
    }
    for binary in metadata
        .rust_build_meta
        .non_test_binaries
        .values()
        .flatten()
    {
        let relative = checked_relative(&binary.path, "non-test binary")?;
        add_unique_file(
            &mut files,
            Path::new("target").join(&relative),
            target.join(relative),
        )?;
    }

    let stamps = CachedStampSet {
        version: STAMP_SCHEMA,
        binaries: read_stamps_parallel(&test_binaries)?,
    };
    let stamps = serde_json::to_vec(&stamps)
        .map_err(|error| format!("serialize ktstr scheduler/admission metadata: {error}"))?;
    let mut source = ktstr::cache::artifact_tree::ArtifactTreeSource::new();
    // Nextest canonicalizes both reuse-build remap roots before it reads any
    // binary metadata. Cargo's separate build-dir support deliberately puts
    // every executable below `build`, so an otherwise-complete plain build
    // can leave `target` with no captured descendant at all. Preserve both
    // structural roots explicitly instead of relying on file materialization
    // to create their ancestors.
    insert_structural_directory(
        &mut source,
        Path::new("target"),
        target,
        "nextest target directory",
    )?;
    if build != target {
        insert_structural_directory(&mut source, build_prefix, build, "nextest build directory")?;
    }
    for (relative, path) in files {
        source
            .insert_file(&relative, &path)
            .map_err(|error| format!("pin nextest artifact {}: {error:#}", path.display()))?;
    }
    for relative in metadata.rust_build_meta.build_script_out_dirs.values() {
        let relative = checked_relative(relative, "build-script output directory")?;
        let path = build.join(&relative);
        if path.exists() {
            source
                .insert_immutable_tree(build_prefix.join(&relative), &path)
                .map_err(|error| {
                    format!(
                        "capture complete build-script output directory {}: {error:#}",
                        path.display()
                    )
                })?;
        }
    }
    for relative in &metadata.rust_build_meta.linked_paths {
        let relative = checked_relative(relative, "linked loader directory")?;
        insert_depth_one(
            &mut source,
            &build_prefix.join(&relative),
            &build.join(relative),
        )?;
    }
    if let Some(producer_profdata) = producer_profdata {
        let metadata = std::fs::symlink_metadata(producer_profdata).map_err(|error| {
            format!(
                "inspect merged coverage producer profile {}: {error}",
                producer_profdata.display(),
            )
        })?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(format!(
                "merged coverage producer profile is not a regular file: {}",
                producer_profdata.display(),
            ));
        }
        source
            .insert_file(COVERAGE_PRODUCER_PROFDATA_PATH, producer_profdata)
            .map_err(|error| {
                format!(
                    "capture merged coverage producer profile {}: {error:#}",
                    producer_profdata.display(),
                )
            })?;
    }
    source
        .insert_bytes(CARGO_METADATA_PATH, cargo_metadata, 0o444)
        .map_err(|error| format!("capture Cargo metadata: {error:#}"))?;
    source
        .insert_bytes(BINARIES_METADATA_PATH, binaries_metadata, 0o444)
        .map_err(|error| format!("capture nextest binaries metadata: {error:#}"))?;
    source
        .insert_bytes(KTSTR_STAMPS_PATH, &stamps, 0o444)
        .map_err(|error| format!("capture ktstr scheduler/admission metadata: {error:#}"))?;
    Ok(source)
}

pub(crate) fn finish_materialization(
    tree: ktstr::cache::artifact_tree::MaterializedArtifactTree,
    stable_source: StableCargoSource,
) -> Result<MaterializedNextestArtifacts, String> {
    let stable_workspace_root = stable_source.workspace_root.clone();
    let stable_invocation_root = stable_source.invocation_root.clone();
    let writable_workspace_root = stable_source.original_workspace_root.clone();
    let writable_invocation_root = stable_source.original_invocation_root.clone();
    let binaries_metadata = tree.root().join(BINARIES_METADATA_PATH);
    let bytes = std::fs::read(&binaries_metadata).map_err(|error| {
        format!(
            "read materialized nextest metadata {}: {error}",
            binaries_metadata.display()
        )
    })?;
    let metadata: BinaryMetadata = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse materialized nextest metadata: {error}"))?;
    let build_prefix = build_prefix(&metadata);
    let target_directory = tree.root().join("target");
    let producer_profdata_path = tree.root().join(COVERAGE_PRODUCER_PROFDATA_PATH);
    let producer_profdata = match std::fs::symlink_metadata(&producer_profdata_path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            Some(producer_profdata_path)
        }
        Ok(_) => {
            return Err(format!(
                "cached coverage producer profile is not a regular file: {}",
                producer_profdata_path.display(),
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(format!(
                "inspect cached coverage producer profile {}: {error}",
                producer_profdata_path.display(),
            ));
        }
    };
    let materialized_build_directory = tree.root().join(build_prefix);
    let original_build = build_directory(&metadata);
    let mut test_binaries = metadata
        .rust_binaries
        .values()
        .map(|binary| {
            map_absolute_under(
                &binary.binary_path,
                original_build,
                &materialized_build_directory,
                "materialized test binary",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    test_binaries.sort();
    test_binaries.dedup();
    let mut loader_paths = Vec::new();
    for relative in &metadata.rust_build_meta.linked_paths {
        loader_paths.push(
            materialized_build_directory.join(checked_relative(relative, "linked loader path")?),
        );
    }
    for relative in &metadata.rust_build_meta.base_output_directories {
        let base =
            materialized_build_directory.join(checked_relative(relative, "base output directory")?);
        loader_paths.push(base.join("deps"));
        loader_paths.push(base);
    }
    loader_paths.retain(|path| path.is_dir());
    loader_paths.sort();
    loader_paths.dedup();
    let stamp_path = tree.root().join(KTSTR_STAMPS_PATH);
    let stamps: CachedStampSet =
        serde_json::from_slice(&std::fs::read(&stamp_path).map_err(|error| {
            format!(
                "read cached ktstr metadata {}: {error}",
                stamp_path.display()
            )
        })?)
        .map_err(|error| {
            format!(
                "parse cached ktstr metadata {}: {error}",
                stamp_path.display()
            )
        })?;
    if stamps.version != STAMP_SCHEMA {
        return Err(format!(
            "unsupported cached ktstr metadata version {}",
            stamps.version
        ));
    }
    Ok(MaterializedNextestArtifacts {
        stable_workspace_root,
        stable_invocation_root,
        writable_workspace_root,
        writable_invocation_root,
        cargo_metadata: tree.root().join(CARGO_METADATA_PATH),
        binaries_metadata,
        target_directory,
        build_directory: materialized_build_directory,
        producer_profdata,
        test_binaries,
        loader_paths,
        scheduler_stamps: stamps.binaries,
        tree,
        _stable_source: stable_source,
    })
}

fn fixed_hasher() -> ahash::AHasher {
    ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher()
}

fn hash_bytes(hasher: &mut ahash::AHasher, bytes: &[u8]) {
    hasher.write_u64(bytes.len() as u64);
    hasher.write(bytes);
}

fn normalized_argument(argument: &str, source: &SourceLayout) -> Vec<u8> {
    let mut value = argument.as_bytes().to_vec();
    let mut replacements = source
        .outputs
        .iter()
        .map(|path| (path.clone(), b"$OUTPUT".to_vec()))
        .collect::<Vec<_>>();
    replacements.extend(source.roots.iter().map(|root| {
        (
            root.source.clone(),
            format!("$SOURCE/{}", root.semantic_name).into_bytes(),
        )
    }));
    replacements.extend(source.cargo_configs.iter().map(|config| {
        (
            config.source.clone(),
            format!("$CONFIG/{}", config.semantic_name).into_bytes(),
        )
    }));
    replacements.extend(source.target_specs.iter().map(|target| {
        (
            target.source.clone(),
            format!("$TARGET/{}", target.semantic_name).into_bytes(),
        )
    }));
    replacements.push((source.workspace.clone(), b"$WORKSPACE".to_vec()));
    replacements.sort_by_key(|(path, _)| std::cmp::Reverse(path.as_os_str().as_bytes().len()));
    for (path, replacement) in replacements {
        let needle = path.as_os_str().as_bytes();
        if needle.is_empty() {
            continue;
        }
        while let Some(index) = value
            .windows(needle.len())
            .position(|window| window == needle)
        {
            value.splice(index..index + needle.len(), replacement.iter().copied());
        }
    }
    value
}

fn source_path_is_excluded(root: &Path, path: &Path, outputs: &[PathBuf]) -> bool {
    outputs
        .iter()
        // An output can only exclude source files when the output is inside
        // this source root.  Stable scheduler discovery deliberately plans a
        // second source closure from a workspace inside ktstr's cache.  In
        // that case the cache root is an ancestor of the source root, not an
        // output contained by it, and must not exclude the source itself.
        .any(|output| output.starts_with(root) && (path == output || path.starts_with(output)))
        || path.strip_prefix(root).ok().is_some_and(|relative| {
            relative.components().any(|component| {
                component.as_os_str() == ".git"
                    || component
                        .as_os_str()
                        .to_str()
                        .is_some_and(|name| name.starts_with(".env"))
            })
        })
}

fn discover_git_workdir(root: &Path) -> Option<PathBuf> {
    let repository = gix::discover(root).ok()?;
    let workdir = canonical_or_lexical(repository.workdir()?);
    canonical_or_lexical(root)
        .starts_with(&workdir)
        .then_some(workdir)
}

#[derive(Debug, Clone, Default)]
struct GitSourceInputs {
    paths: BTreeSet<PathBuf>,
    gitlinks: BTreeMap<PathBuf, String>,
    status_semantics: BTreeSet<Vec<u8>>,
    metadata: Option<GitMetadataPlan>,
}

#[derive(Debug, Clone)]
struct GitMetadataPlan {
    /// Cache-relative metadata path -> exact source path. The object database
    /// is included, so the stable checkout never depends on the originating
    /// runner checkout remaining alive.
    files: BTreeMap<PathBuf, PathBuf>,
    config: Vec<u8>,
    semantic_digest: u64,
}

fn git_head(root: &Path) -> Option<String> {
    gix::open(root)
        .ok()?
        .head_id()
        .ok()
        .map(|id| id.detach().to_string())
}

fn git_metadata_path_is_transient(relative: &Path) -> bool {
    let first = relative
        .components()
        .next()
        .and_then(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        });
    if matches!(
        first,
        Some("worktrees" | "modules" | "hooks" | "logs" | "objects")
    ) {
        return true;
    }
    if matches!(
        relative.to_str(),
        Some(
            "config"
                | "config.worktree"
                | "commondir"
                | "gitdir"
                | "FETCH_HEAD"
                | "ORIG_HEAD"
                | "COMMIT_EDITMSG"
        )
    ) {
        return true;
    }
    relative
        .file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| name.ends_with(".lock"))
}

fn collect_git_metadata_tree(
    root: &Path,
    files: &mut BTreeMap<PathBuf, PathBuf>,
) -> Result<(), String> {
    let walker = walkdir::WalkDir::new(root)
        .follow_links(false)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|entry| {
            entry
                .path()
                .strip_prefix(root)
                .ok()
                .is_none_or(|relative| !git_metadata_path_is_transient(relative))
        });
    for entry in walker {
        if crate::interrupt::INTERRUPTED.load(Ordering::Acquire) {
            return Err("Git metadata capture interrupted".to_string());
        }
        let entry =
            entry.map_err(|error| format!("walk Git metadata {}: {error}", root.display()))?;
        let relative = entry
            .path()
            .strip_prefix(root)
            .expect("walked Git metadata is below its root");
        if relative.as_os_str().is_empty()
            || relative == Path::new("objects/info/alternates")
            || entry.file_type().is_dir()
        {
            continue;
        }
        if !entry.file_type().is_file() && !entry.file_type().is_symlink() {
            continue;
        }
        files.insert(relative.to_path_buf(), entry.path().to_path_buf());
    }
    Ok(())
}

fn collect_git_object_store(
    object_dir: &Path,
    files: &mut BTreeMap<PathBuf, PathBuf>,
    visited: &mut BTreeSet<PathBuf>,
) -> Result<(), String> {
    let object_dir = canonical_or_lexical(object_dir);
    if !visited.insert(object_dir.clone()) {
        return Ok(());
    }
    let walker = walkdir::WalkDir::new(&object_dir)
        .follow_links(false)
        .sort_by_file_name();
    for entry in walker {
        if crate::interrupt::INTERRUPTED.load(Ordering::Acquire) {
            return Err("Git object database capture interrupted".to_string());
        }
        let entry = entry.map_err(|error| {
            format!("walk Git object database {}: {error}", object_dir.display())
        })?;
        let relative = entry
            .path()
            .strip_prefix(&object_dir)
            .expect("walked Git object is below its root");
        if relative.as_os_str().is_empty()
            || relative == Path::new("info/alternates")
            || entry.file_type().is_dir()
        {
            continue;
        }
        if !entry.file_type().is_file() && !entry.file_type().is_symlink() {
            continue;
        }
        files
            .entry(Path::new("objects").join(relative))
            .or_insert_with(|| entry.path().to_path_buf());
    }

    let alternates = object_dir.join("info/alternates");
    let bytes = match std::fs::read(&alternates) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "read Git object alternates {}: {error}",
                alternates.display()
            ));
        }
    };
    for line in bytes.split(|byte| *byte == b'\n') {
        if line.is_empty() {
            continue;
        }
        let path = PathBuf::from(OsStr::from_bytes(line));
        let path = if path.is_absolute() {
            path
        } else {
            object_dir.join(path)
        };
        collect_git_object_store(&path, files, visited)?;
    }
    Ok(())
}

fn synthesized_git_config(repository: &gix::Repository) -> Vec<u8> {
    use gix::bstr::ByteSlice as _;

    let snapshot = repository.config_snapshot();
    let repository_format = snapshot
        .integer("core.repositoryFormatVersion")
        .unwrap_or(0);
    let mut bytes = format!(
        "[core]\n\trepositoryformatversion = {repository_format}\n\tbare = false\n\tworktree = ..\n"
    )
    .into_bytes();
    for key in [
        "fileMode",
        "symlinks",
        "ignoreCase",
        "precomposeUnicode",
        "sparseCheckout",
        "sparseCheckoutCone",
    ] {
        let full_key = format!("core.{key}");
        if let Some(value) = snapshot.boolean(&full_key) {
            bytes.extend_from_slice(format!("\t{key} = {value}\n").as_bytes());
        }
    }
    let mut extensions = Vec::new();
    for key in ["objectFormat", "refStorage"] {
        let full_key = format!("extensions.{key}");
        let Some(value) = snapshot.string(&full_key) else {
            continue;
        };
        let Some(value) = value.to_str().ok().filter(|value| {
            value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        }) else {
            continue;
        };
        extensions.extend_from_slice(format!("\t{key} = {value}\n").as_bytes());
    }
    if !extensions.is_empty() {
        bytes.extend_from_slice(b"[extensions]\n");
        bytes.extend_from_slice(&extensions);
    }
    bytes
}

fn git_metadata_semantic_digest(
    files: &BTreeMap<PathBuf, PathBuf>,
    config: &[u8],
    index_digest: u64,
) -> Result<u64, String> {
    let mut hasher = fixed_hasher();
    hash_bytes(&mut hasher, b"ktstr-stable-git-metadata");
    hash_bytes(&mut hasher, config);
    hasher.write_u64(index_digest);
    // Loose objects and packs are named by content-derived IDs, so their
    // deterministic relative names distinguish a shallow object database
    // from a deeper one without reopening every object on every invocation.
    // Git's fixed-name object metadata is not content-addressed; hash those
    // small files normally. Index storage is normalized separately below.
    for (relative, source) in files {
        let is_index_storage = relative == Path::new("index")
            || relative
                .file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.starts_with("sharedindex."));
        if is_index_storage {
            // Raw index storage contains filesystem stat caches and differs
            // across equivalent checkouts. Its normalized entries were
            // hashed above; whichever representation won publication remains
            // a complete cache-owned index view.
            continue;
        }
        hash_bytes(&mut hasher, relative.as_os_str().as_bytes());
        if relative.starts_with("objects")
            && !relative.starts_with("objects/info")
            && relative != Path::new("objects/pack/multi-pack-index")
        {
            continue;
        }
        let metadata = std::fs::symlink_metadata(source)
            .map_err(|error| format!("stat Git metadata {}: {error}", source.display()))?;
        if metadata.file_type().is_symlink() {
            let target = std::fs::read_link(source)
                .map_err(|error| format!("read Git metadata link {}: {error}", source.display()))?;
            hash_bytes(&mut hasher, target.as_os_str().as_bytes());
        } else {
            hasher.write_u64(
                ktstr::cache::content_file_digest(source).map_err(|error| {
                    format!("digest Git metadata {}: {error:#}", source.display())
                })?,
            );
        }
    }
    Ok(hasher.finish())
}

fn git_index_semantic_digest(index: &gix::index::State) -> u64 {
    let mut hasher = fixed_hasher();
    hash_bytes(&mut hasher, b"ktstr-stable-git-index");
    for entry in index.entries() {
        hash_bytes(&mut hasher, entry.path(index));
        hash_bytes(&mut hasher, entry.id.as_bytes());
        hasher.write_u32(entry.mode.bits());
        hasher.write_u32(entry.flags.stage_raw());
        for flag in [
            gix::index::entry::Flags::ASSUME_VALID,
            gix::index::entry::Flags::INTENT_TO_ADD,
            gix::index::entry::Flags::SKIP_WORKTREE,
        ] {
            hasher.write_u8(u8::from(entry.flags.contains(flag)));
        }
    }
    hasher.finish()
}

fn plan_git_metadata(
    repository: &gix::Repository,
    index: &gix::index::State,
) -> Result<GitMetadataPlan, String> {
    let git_dir = repository.git_dir();
    let common_dir = repository.common_dir();
    let mut files = BTreeMap::new();
    collect_git_metadata_tree(common_dir, &mut files)?;
    let mut visited_objects = BTreeSet::new();
    collect_git_object_store(
        &common_dir.join("objects"),
        &mut files,
        &mut visited_objects,
    )?;
    if git_dir != common_dir {
        // Worktree-local HEAD/index/ref state wins over the common repository
        // view, but commondir/gitdir are deliberately not retained.
        collect_git_metadata_tree(git_dir, &mut files)?;
    }
    let config = synthesized_git_config(repository);
    let semantic_digest =
        git_metadata_semantic_digest(&files, &config, git_index_semantic_digest(index))?;
    Ok(GitMetadataPlan {
        files,
        config,
        semantic_digest,
    })
}

fn git_source_input_paths(root: &Path) -> Result<GitSourceInputs, String> {
    use gix::bstr::BString;

    let repository = gix::open(root)
        .map_err(|error| format!("open Cargo source repository {}: {error}", root.display()))?;
    let index = repository.index().map_err(|error| {
        format!(
            "read Cargo source repository index {}: {error}",
            root.display()
        )
    })?;
    let mut inputs = GitSourceInputs {
        metadata: Some(plan_git_metadata(&repository, &index)?),
        ..GitSourceInputs::default()
    };
    for entry in index.entries() {
        let path = PathBuf::from(OsStr::from_bytes(entry.path(&index)));
        if entry.mode.is_submodule() {
            inputs.gitlinks.insert(path.clone(), entry.id.to_string());
        }
        inputs.paths.insert(path);
    }
    let status = repository
        .status(gix::progress::Discard)
        .map_err(|error| format!("open Cargo source status {}: {error}", root.display()))?
        .untracked_files(gix::status::UntrackedFiles::Files)
        .tree_index_track_renames(gix::status::tree_index::TrackRenames::Disabled)
        .index_worktree_submodules(gix::status::Submodule::Given {
            ignore: gix::submodule::config::Ignore::All,
            check_dirty: false,
        })
        .index_worktree_options_mut(ktstr::git_status::configure_index_worktree_parallelism)
        .into_iter(Vec::<BString>::new())
        .map_err(|error| {
            format!(
                "create Cargo source status iterator {}: {error}",
                root.display()
            )
        })?;
    for item in status {
        if crate::interrupt::INTERRUPTED.load(Ordering::Acquire) {
            return Err("nextest artifact identity scan interrupted".to_string());
        }
        let item = item.map_err(|error| {
            format!("read Cargo source status item {}: {error}", root.display())
        })?;
        let semantic = match &item {
            gix::status::Item::IndexWorktree(change) => match change {
                gix::status::index_worktree::Item::Modification {
                    rela_path, status, ..
                } => format!("index-worktree:{rela_path:?}:{status:?}"),
                gix::status::index_worktree::Item::DirectoryContents { entry, .. } => format!(
                    "directory:{:?}:{:?}:{:?}:{:?}",
                    entry.rela_path, entry.status, entry.disk_kind, entry.index_kind
                ),
                gix::status::index_worktree::Item::Rewrite { .. } => {
                    "unexpected-rewrite-with-renames-disabled".to_string()
                }
            },
            gix::status::Item::TreeIndex(change) => format!("tree-index:{change:?}"),
        };
        inputs.status_semantics.insert(semantic.into_bytes());
        inputs
            .paths
            .insert(PathBuf::from(OsStr::from_bytes(item.location())));
    }
    Ok(inputs)
}

fn non_git_source_input_paths(
    root: &Path,
    outputs: &[PathBuf],
) -> Result<BTreeSet<PathBuf>, String> {
    let mut paths = BTreeSet::new();
    let walker = walkdir::WalkDir::new(root)
        .follow_links(false)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|entry| !source_path_is_excluded(root, entry.path(), outputs));
    for entry in walker {
        if crate::interrupt::INTERRUPTED.load(Ordering::Acquire) {
            return Err("nextest artifact identity scan interrupted".to_string());
        }
        let entry = entry.map_err(|error| {
            format!("walk non-git Cargo source root {}: {error}", root.display())
        })?;
        if entry.file_type().is_file() || entry.file_type().is_symlink() {
            paths.insert(
                entry
                    .path()
                    .strip_prefix(root)
                    .expect("walked source path is below its root")
                    .to_path_buf(),
            );
        }
    }
    Ok(paths)
}

fn source_tree_digest(
    root: &Path,
    input_paths: &BTreeSet<PathBuf>,
    gitlinks: &BTreeMap<PathBuf, String>,
) -> Result<u64, String> {
    use std::os::unix::fs::PermissionsExt as _;

    let mut hasher = fixed_hasher();
    hash_bytes(&mut hasher, b"ktstr-nextest-source-tree");
    for relative in input_paths {
        if crate::interrupt::INTERRUPTED.load(Ordering::Acquire) {
            return Err("nextest artifact identity scan interrupted".to_string());
        }
        let path = root.join(relative);
        hash_bytes(&mut hasher, relative.as_os_str().as_bytes());
        if let Some(commit) = gitlinks.get(relative) {
            hasher.write_u8(b'g');
            hash_bytes(&mut hasher, commit.as_bytes());
            continue;
        }
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                hasher.write_u8(b'x');
                continue;
            }
            Err(error) => {
                return Err(format!(
                    "stat Cargo source input {}: {error}",
                    path.display()
                ));
            }
        };
        if metadata.file_type().is_symlink() {
            hasher.write_u8(b'l');
            let target = std::fs::read_link(&path).map_err(|error| {
                format!("read Cargo source symlink {}: {error}", path.display())
            })?;
            hash_bytes(&mut hasher, target.as_os_str().as_bytes());
        } else if metadata.is_file() {
            hasher.write_u8(b'f');
            hasher.write_u8(u8::from(metadata.permissions().mode() & 0o111 != 0));
            hasher.write_u64(metadata.len());
            hasher.write_u64(ktstr::cache::content_file_digest(&path).map_err(|error| {
                format!("digest Cargo source input {}: {error:#}", path.display())
            })?);
        } else if metadata.is_dir() {
            return Err(format!(
                "non-gitlink Cargo source input unexpectedly became a directory: {}",
                path.display()
            ));
        } else {
            return Err(format!(
                "unsupported Cargo source input: {}",
                path.display()
            ));
        }
    }
    Ok(hasher.finish())
}

#[derive(Debug, Clone)]
struct SourceRootPlan {
    source: PathBuf,
    virtual_path: PathBuf,
    semantic_name: String,
    is_git: bool,
    input_paths: BTreeSet<PathBuf>,
    git_head: Option<String>,
    git_status_semantics: BTreeSet<Vec<u8>>,
    git_metadata: Option<GitMetadataPlan>,
    digest: u64,
}

#[derive(Debug, Clone)]
struct CargoConfigPlan {
    source: PathBuf,
    virtual_path: Option<PathBuf>,
    semantic_name: String,
    explicit_argument: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct PendingCargoConfig {
    source: PathBuf,
    semantic_name: String,
    materialize: bool,
    explicit_argument: bool,
}

#[derive(Debug, Clone)]
struct CargoTargetSpecPlan {
    source: PathBuf,
    virtual_path: PathBuf,
    semantic_name: String,
}

#[derive(Debug, Clone)]
struct SourceLayout {
    workspace: PathBuf,
    invocation: PathBuf,
    invocation_relation: (usize, Vec<std::ffi::OsString>),
    workspace_virtual: PathBuf,
    invocation_virtual: PathBuf,
    roots: Vec<SourceRootPlan>,
    cargo_configs: Vec<CargoConfigPlan>,
    target_specs: Vec<CargoTargetSpecPlan>,
    declared_package_inputs: BTreeSet<PathBuf>,
    outputs: Vec<PathBuf>,
    identity: u64,
}

fn normalized_output_bytes(bytes: &[u8], outputs: &[PathBuf]) -> Vec<u8> {
    let mut value = bytes.to_vec();
    let mut replacements = outputs
        .iter()
        .map(|path| (path, b"$OUTPUT".to_vec()))
        .collect::<Vec<_>>();
    replacements.sort_by_key(|(path, _)| std::cmp::Reverse(path.as_os_str().as_bytes().len()));
    for (path, replacement) in replacements {
        let needle = path.as_os_str().as_bytes();
        if needle.is_empty() {
            continue;
        }
        while let Some(index) = value
            .windows(needle.len())
            .position(|window| window == needle)
        {
            value.splice(index..index + needle.len(), replacement.iter().copied());
        }
    }
    value
}

fn read_cargo_config_identity(path: &Path, outputs: &[PathBuf]) -> Result<Vec<u8>, String> {
    let metadata = std::fs::metadata(path)
        .map_err(|error| format!("stat Cargo config {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "Cargo config is not a regular file: {}",
            path.display()
        ));
    }
    if metadata.len() > (16 << 20) {
        return Err(format!("Cargo config exceeds 16 MiB: {}", path.display()));
    }
    let bytes = std::fs::read(path)
        .map_err(|error| format!("read Cargo config {}: {error}", path.display()))?;
    Ok(normalized_output_bytes(&bytes, outputs))
}

fn cargo_surface_argument(argument: &str) -> &str {
    argument
        .strip_prefix("nextest:")
        .or_else(|| argument.strip_prefix("cargo:"))
        .or_else(|| argument.strip_prefix("llvm-cov-env:"))
        .unwrap_or(argument)
}

fn explicit_cargo_config_files(
    build_surface: &[String],
    invocation: &Path,
) -> Vec<(usize, PathBuf)> {
    let arguments = build_surface
        .iter()
        .map(|argument| cargo_surface_argument(argument))
        .collect::<Vec<_>>();
    let mut configs = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        let argument = arguments[index];
        let value = if argument == "--config" {
            index += 1;
            arguments.get(index).copied()
        } else {
            argument.strip_prefix("--config=")
        };
        if let Some(value) = value {
            let path = Path::new(value);
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                invocation.join(path)
            };
            if path.is_file() {
                configs.push((index, canonical_or_lexical(&path)));
            }
        }
        index += 1;
    }
    configs
}

fn cargo_config_includes(path: &Path) -> Result<Vec<(PathBuf, bool)>, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("read Cargo config {}: {error}", path.display()))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("Cargo config {} is not UTF-8: {error}", path.display()))?;
    let document = text
        .parse::<toml::Table>()
        .map_err(|error| format!("parse Cargo config {}: {error}", path.display()))?;
    let Some(include) = document.get("include") else {
        return Ok(Vec::new());
    };
    let entries = include.as_array().ok_or_else(|| {
        format!(
            "Cargo config include must be an array in {}",
            path.display()
        )
    })?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut includes = Vec::with_capacity(entries.len());
    for entry in entries {
        let (value, optional) = if let Some(value) = entry.as_str() {
            (value, false)
        } else if let Some(table) = entry.as_table() {
            let value = table
                .get("path")
                .and_then(toml::Value::as_str)
                .ok_or_else(|| {
                    format!(
                        "Cargo config include table has no string path in {}",
                        path.display()
                    )
                })?;
            let optional = table
                .get("optional")
                .map(|value| {
                    value.as_bool().ok_or_else(|| {
                        format!(
                            "Cargo config include optional flag is not boolean in {}",
                            path.display()
                        )
                    })
                })
                .transpose()?
                .unwrap_or(false);
            (value, optional)
        } else {
            return Err(format!(
                "Cargo config include entry is neither a path nor a table in {}",
                path.display()
            ));
        };
        let include = Path::new(value);
        let include = if include.is_absolute() {
            include.to_path_buf()
        } else {
            parent.join(include)
        };
        includes.push((canonical_or_lexical(&include), optional));
    }
    Ok(includes)
}

fn expand_cargo_config_closure(pending: &mut Vec<PendingCargoConfig>) -> Result<(), String> {
    let mut by_path = pending
        .iter()
        .enumerate()
        .map(|(index, config)| (config.source.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut index = 0;
    while index < pending.len() {
        let parent = pending[index].clone();
        for (include_index, (include, optional)) in cargo_config_includes(&parent.source)?
            .into_iter()
            .enumerate()
        {
            if !include.is_file() {
                if optional {
                    continue;
                }
                return Err(format!(
                    "required Cargo config include {} from {} does not exist",
                    include.display(),
                    parent.source.display()
                ));
            }
            if let Some(existing) = by_path.get(&include).copied() {
                pending[existing].materialize |= parent.materialize;
                continue;
            }
            let next = pending.len();
            by_path.insert(include.clone(), next);
            pending.push(PendingCargoConfig {
                source: include,
                semantic_name: format!("{}/include/{include_index}", parent.semantic_name),
                materialize: parent.materialize,
                explicit_argument: false,
            });
        }
        index += 1;
    }
    Ok(())
}

#[derive(Debug)]
struct PackageIncludePattern {
    include: bool,
    matches: Vec<glob::Pattern>,
}

fn compile_package_include_pattern(
    value: &str,
    manifest: &Path,
) -> Result<PackageIncludePattern, String> {
    let (include, value) = value
        .strip_prefix('!')
        .map_or((true, value), |value| (false, value));
    let rooted = value.starts_with('/');
    let directory_only = value.ends_with('/');
    let value = value.trim_matches('/');
    if value.is_empty() {
        return Err(format!(
            "empty package.include pattern in {}",
            manifest.display()
        ));
    }
    let mut bases = vec![value.to_string()];
    if !rooted && !value.contains('/') {
        bases.push(format!("**/{value}"));
    }
    let mut candidates = Vec::new();
    for base in bases {
        if !directory_only {
            candidates.push(base.clone());
        }
        candidates.push(format!("{base}/**"));
    }
    let matches = candidates
        .into_iter()
        .map(|candidate| {
            glob::Pattern::new(&candidate).map_err(|error| {
                format!(
                    "invalid package.include pattern {value:?} in {}: {error}",
                    manifest.display()
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PackageIncludePattern { include, matches })
}

fn package_include_scan_prefix(value: &str) -> Option<PathBuf> {
    let value = value.strip_prefix('!').unwrap_or(value);
    let rooted = value.starts_with('/');
    let value = value.trim_matches('/');
    if !rooted && !value.contains('/') {
        // A slash-free gitignore pattern matches at every depth, even when it
        // begins with a literal component (`foo` is `**/foo`).
        return None;
    }
    let mut prefix = PathBuf::new();
    for component in value.split('/') {
        if component.is_empty() || component == "." {
            continue;
        }
        if component == ".."
            || component
                .bytes()
                .any(|byte| matches!(byte, b'*' | b'?' | b'[' | b']'))
        {
            break;
        }
        prefix.push(component);
    }
    (!prefix.as_os_str().is_empty()).then_some(prefix)
}

fn package_include_matches(patterns: &[PackageIncludePattern], relative: &Path) -> bool {
    let options = glob::MatchOptions {
        case_sensitive: true,
        require_literal_separator: true,
        require_literal_leading_dot: false,
    };
    patterns.iter().fold(false, |included, pattern| {
        if pattern
            .matches
            .iter()
            .any(|candidate| candidate.matches_path_with(relative, options))
        {
            pattern.include
        } else {
            included
        }
    })
}

fn scan_declared_package_include_paths(
    root: &Path,
    values: &[String],
    outputs: &[PathBuf],
    manifest: &Path,
) -> Result<BTreeSet<PathBuf>, String> {
    let patterns = values
        .iter()
        .map(|value| compile_package_include_pattern(value, manifest))
        .collect::<Result<Vec<_>, _>>()?;

    // Every traversal is authorized by a positive package.include pattern.
    // Use its sound anchored prefix when one exists; unrooted and
    // wildcard-first gitignore patterns intentionally scan the package root
    // because they can match at any depth. The walk still prunes all
    // unconditional exclusions and nested Cargo packages.
    let mut scan_roots = values
        .iter()
        .filter(|value| !value.starts_with('!'))
        .map(|value| {
            package_include_scan_prefix(value)
                .map_or_else(|| root.to_path_buf(), |prefix| root.join(prefix))
        })
        .collect::<Vec<_>>();
    scan_roots.sort_by_key(|path| (path.components().count(), path.clone()));
    let mut bounded_roots = Vec::<PathBuf>::new();
    for path in scan_roots {
        if !bounded_roots.iter().any(|parent| path.starts_with(parent)) {
            bounded_roots.push(path);
        }
    }

    let mut package_outputs = outputs.to_vec();
    package_outputs.push(root.join("target"));
    let entry_allowed = |path: &Path, is_directory: bool| {
        let directory = if is_directory {
            Some(path)
        } else {
            path.parent()
        };
        let inside_nested_package = directory.is_some_and(|directory| {
            directory
                .ancestors()
                .take_while(|ancestor| *ancestor != root)
                .any(|ancestor| ancestor.join("Cargo.toml").is_file())
        });
        !source_path_is_excluded(root, path, &package_outputs) && !inside_nested_package
    };

    let mut paths = BTreeSet::new();
    for scan_root in bounded_roots {
        let metadata = match std::fs::symlink_metadata(&scan_root) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(format!(
                    "inspect package.include input {}: {error}",
                    scan_root.display()
                ));
            }
        };
        if metadata.is_file() || metadata.file_type().is_symlink() {
            if entry_allowed(&scan_root, false)
                && let Ok(relative) = scan_root.strip_prefix(root)
                && package_include_matches(&patterns, relative)
            {
                paths.insert(scan_root);
            }
            continue;
        }
        if !metadata.is_dir() || !entry_allowed(&scan_root, true) {
            continue;
        }
        let walker = walkdir::WalkDir::new(&scan_root)
            .follow_links(false)
            .sort_by_file_name()
            .into_iter()
            .filter_entry(|entry| entry_allowed(entry.path(), entry.file_type().is_dir()));
        for entry in walker {
            if crate::interrupt::INTERRUPTED.load(Ordering::Acquire) {
                return Err("package.include scan interrupted".to_string());
            }
            let entry = entry.map_err(|error| {
                format!(
                    "walk package.include inputs below {}: {error}",
                    scan_root.display()
                )
            })?;
            if !entry.file_type().is_file() && !entry.file_type().is_symlink() {
                continue;
            }
            let relative = entry
                .path()
                .strip_prefix(root)
                .expect("package include walk stays below its package root");
            if package_include_matches(&patterns, relative) {
                paths.insert(entry.path().to_path_buf());
            }
        }
    }
    Ok(paths)
}

fn package_declared_include_paths(
    package: &cargo_metadata::Package,
    outputs: &[PathBuf],
) -> Result<BTreeSet<PathBuf>, String> {
    let manifest = package.manifest_path.as_std_path();
    let bytes = std::fs::read(manifest)
        .map_err(|error| format!("read Cargo manifest {}: {error}", manifest.display()))?;
    let text = std::str::from_utf8(&bytes).map_err(|error| {
        format!(
            "Cargo manifest {} is not UTF-8: {error}",
            manifest.display()
        )
    })?;
    let document = text
        .parse::<toml::Table>()
        .map_err(|error| format!("parse Cargo manifest {}: {error}", manifest.display()))?;
    let Some(patterns) = document
        .get("package")
        .and_then(|value| value.get("include"))
        .and_then(toml::Value::as_array)
    else {
        return Ok(BTreeSet::new());
    };
    let root = canonical_or_lexical(
        manifest
            .parent()
            .ok_or_else(|| format!("Cargo manifest has no parent: {}", manifest.display()))?,
    );
    // Non-Git roots are already captured with one filtered tree walk. This
    // extra pass exists solely for ignored Git worktree inputs named by
    // package.include.
    if discover_git_workdir(&root).is_none() {
        return Ok(BTreeSet::new());
    }
    let values = patterns
        .iter()
        .map(|value| {
            let value = value.as_str().ok_or_else(|| {
                format!(
                    "package.include entry is not a string in {}",
                    manifest.display()
                )
            })?;
            Ok(value.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;
    scan_declared_package_include_paths(&root, &values, outputs, manifest)
}

fn explicit_cargo_target_specs(
    build_surface: &[String],
    invocation: &Path,
) -> Vec<(usize, PathBuf)> {
    let arguments = build_surface
        .iter()
        .map(|argument| cargo_surface_argument(argument))
        .collect::<Vec<_>>();
    let mut targets = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        let argument = arguments[index];
        let value = if argument == "--target" {
            index += 1;
            arguments.get(index).copied()
        } else {
            argument.strip_prefix("--target=")
        };
        if let Some(value) = value {
            let path = Path::new(value);
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                invocation.join(path)
            };
            if path.is_file() {
                targets.push((index, canonical_or_lexical(&path)));
            }
        }
        index += 1;
    }
    targets
}

fn canonical_or_lexical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn include_resolved_workspace_lockfile(workspace: &Path, inputs: &mut BTreeSet<PathBuf>) {
    // Identity planning runs only after Cargo's preparatory metadata query has
    // resolved the workspace. Preserve the exact lockfile that query accepted
    // or created even when this is a library workspace whose Cargo.lock is
    // ignored. Deliberately do not create one here: --locked/--frozen must keep
    // failing in the preparatory Cargo command when no usable lockfile exists.
    let lockfile = workspace.join("Cargo.lock");
    if lockfile.is_file() {
        inputs.insert(canonical_or_lexical(&lockfile));
    }
}

fn path_relation(from: &Path, to: &Path) -> Result<(usize, Vec<std::ffi::OsString>), String> {
    let from_components = from.components().collect::<Vec<_>>();
    let to_components = to.components().collect::<Vec<_>>();
    let shared = from_components
        .iter()
        .zip(&to_components)
        .take_while(|(left, right)| left == right)
        .count();
    if shared == 0 {
        return Err(format!(
            "Cargo source roots do not share an absolute filesystem root: {} and {}",
            from.display(),
            to.display()
        ));
    }
    let upward = from_components[shared..]
        .iter()
        .filter(|component| matches!(component, Component::Normal(_)))
        .count();
    let downward = to_components[shared..]
        .iter()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_os_string()),
            _ => None,
        })
        .collect();
    Ok((upward, downward))
}

fn virtual_path_for_relation(
    workspace_virtual: &Path,
    upward: usize,
    downward: &[std::ffi::OsString],
) -> Result<PathBuf, String> {
    let mut components = workspace_virtual
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_os_string()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if upward > components.len() {
        return Err("stable source layout escaped its reserved anchor".to_string());
    }
    components.truncate(components.len() - upward);
    components.extend_from_slice(downward);
    checked_relative(&PathBuf::from_iter(components), "stable Cargo source root")
}

fn hash_path_relation(hasher: &mut ahash::AHasher, relation: &(usize, Vec<std::ffi::OsString>)) {
    hasher.write_u64(relation.0 as u64);
    for component in &relation.1 {
        hash_bytes(hasher, component.as_bytes());
    }
}

fn plan_source_layout(
    metadata: &cargo_metadata::Metadata,
    output_roots: &[PathBuf],
    build_surface: &[String],
    invocation: &Path,
) -> Result<SourceLayout, String> {
    let workspace = canonical_or_lexical(metadata.workspace_root.as_std_path());
    let invocation = canonical_or_lexical(invocation);
    let mut outputs = output_roots
        .iter()
        .map(|path| canonical_or_lexical(path))
        .collect::<BTreeSet<_>>();
    outputs.insert(canonical_or_lexical(
        metadata.target_directory.as_std_path(),
    ));
    if let Ok(cache_root) = ktstr::cache::cargo_artifact_tree_cache_root() {
        outputs.insert(canonical_or_lexical(&cache_root));
    }
    let outputs = outputs.into_iter().collect::<Vec<_>>();

    // Cargo discovers configuration from the invocation directory through
    // every ancestor, plus CARGO_HOME. Ancestor configs must be reproduced at
    // the equivalent stable-source ancestor; CARGO_HOME remains externally
    // discoverable but still participates in the content identity.
    let mut pending_configs = Vec::<PendingCargoConfig>::new();
    for (depth, ancestor) in invocation.ancestors().enumerate() {
        for name in ["config", "config.toml"] {
            let path = ancestor.join(".cargo").join(name);
            if path.is_file() {
                pending_configs.push(PendingCargoConfig {
                    source: canonical_or_lexical(&path),
                    semantic_name: format!("ancestor/{depth}/{name}"),
                    materialize: true,
                    explicit_argument: false,
                });
            }
        }
    }
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")));
    if let Some(cargo_home) = cargo_home {
        for name in ["config", "config.toml"] {
            let path = cargo_home.join(name);
            if path.is_file() {
                pending_configs.push(PendingCargoConfig {
                    source: canonical_or_lexical(&path),
                    semantic_name: format!("cargo-home/{name}"),
                    materialize: false,
                    explicit_argument: false,
                });
            }
        }
    }
    for (index, path) in explicit_cargo_config_files(build_surface, &invocation) {
        pending_configs.push(PendingCargoConfig {
            source: path,
            semantic_name: format!("explicit/{index}"),
            materialize: true,
            explicit_argument: true,
        });
    }
    expand_cargo_config_closure(&mut pending_configs)?;
    let pending_target_specs = explicit_cargo_target_specs(build_surface, &invocation);

    // Ignored files are not swept blindly: that would pull large build trees
    // and secrets such as direnv's `.envrc` into the shared cache. Cargo's
    // package.include list is the bounded declarative escape hatch for an
    // ignored file that is nevertheless a build input.
    let mut declared_package_inputs = BTreeSet::new();
    for package in metadata
        .packages
        .iter()
        .filter(|package| package.source.is_none())
    {
        declared_package_inputs.extend(package_declared_include_paths(package, &outputs)?);
    }
    include_resolved_workspace_lockfile(&workspace, &mut declared_package_inputs);

    let mut package_roots = BTreeSet::from([workspace.clone()]);
    for package in metadata
        .packages
        .iter()
        .filter(|package| package.source.is_none())
    {
        if let Some(parent) = package.manifest_path.parent() {
            package_roots.insert(canonical_or_lexical(parent.as_std_path()));
        }
    }

    // One gix index/status path set covers every ordinary package in the same
    // repository. Nested repositories/submodules remain separate roots. Only
    // external non-git roots need a walk, and overlapping non-git package
    // roots collapse to their shallowest ancestor.
    let mut root_kinds = BTreeMap::<PathBuf, bool>::new();
    let mut non_git = Vec::new();
    let mut git_backed_package_roots = Vec::new();
    for package_root in &package_roots {
        if let Some(repository) = discover_git_workdir(package_root) {
            root_kinds.insert(repository.clone(), true);
            git_backed_package_roots.push((package_root.clone(), repository));
        } else {
            non_git.push(package_root.clone());
        }
    }
    let mut git_inputs = BTreeMap::<PathBuf, GitSourceInputs>::new();
    let mut pending_git = root_kinds
        .iter()
        .filter_map(|(root, git)| (*git).then_some(root.clone()))
        .collect::<Vec<_>>();
    while let Some(root) = pending_git.pop() {
        if git_inputs.contains_key(&root) {
            continue;
        }
        let inputs = git_source_input_paths(&root)?;
        for relative in inputs.gitlinks.keys() {
            let nested = root.join(relative);
            let workdir = discover_git_workdir(&nested).ok_or_else(|| {
                format!(
                    "Cargo source repository {} has uninitialized or unavailable submodule {} \
                     at commit {}; initialize it before building a stable artifact closure",
                    root.display(),
                    relative.display(),
                    inputs.gitlinks[relative]
                )
            })?;
            if workdir != root && !root_kinds.contains_key(&workdir) {
                root_kinds.insert(workdir.clone(), true);
                pending_git.push(workdir);
            }
        }
        git_inputs.insert(root, inputs);
    }

    // A Cargo workspace can deliberately live below a containing repository's
    // ignored/generated directory (test fixtures and generated SDK workspaces
    // are common examples). `gix status` correctly omits those files, but the
    // local Cargo metadata proves that they are build inputs. Without a local
    // overlay the stable invocation directory is empty; Cargo then walks to an
    // ancestor `Cargo.toml` and applies package-qualified features to an
    // entirely different workspace.
    //
    // Detect that shape by the one file every local Cargo root must own. A
    // shallow non-Git overlay captures the complete generated workspace while
    // preserving the containing repository for tracked path dependencies and
    // ancestor Cargo configuration. Overlapping generated package roots
    // collapse below, so one ignored workspace is scanned only once.
    for (package_root, repository) in git_backed_package_roots {
        let manifest = package_root.join("Cargo.toml");
        let Ok(relative_manifest) = manifest.strip_prefix(&repository) else {
            continue;
        };
        let captured = git_inputs
            .get(&repository)
            .is_some_and(|inputs| inputs.paths.contains(relative_manifest));
        if captured {
            continue;
        }
        if package_root == repository {
            // A generated repository may ignore its own Cargo source. The
            // source root cannot appear twice in `root_kinds`, so merge the
            // bounded non-Git view into the explicit-input set while retaining
            // the Git metadata needed by build scripts. Keeping these paths in
            // `declared_package_inputs` also makes identity revalidation apply
            // the exact same overlay after publication.
            let overlay = non_git_source_input_paths(&package_root, &outputs)?;
            declared_package_inputs.extend(
                overlay
                    .into_iter()
                    .map(|relative| package_root.join(relative)),
            );
        } else {
            non_git.push(package_root);
        }
    }
    non_git.sort_by_key(|root| (root.components().count(), root.clone()));
    for root in non_git {
        if !root_kinds
            .iter()
            .any(|(existing, git)| !*git && root.starts_with(existing))
        {
            root_kinds.insert(root, false);
        }
    }
    let roots = root_kinds.keys().cloned().collect::<Vec<_>>();
    let relations = roots
        .iter()
        .map(|root| path_relation(&workspace, root))
        .collect::<Result<Vec<_>, _>>()?;
    let config_relations = pending_configs
        .iter()
        .map(|config| {
            config
                .materialize
                .then(|| path_relation(&workspace, &config.source))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let invocation_relation = path_relation(&workspace, &invocation)?;
    let max_up = relations
        .iter()
        .map(|(up, _)| *up)
        .chain(
            config_relations
                .iter()
                .filter_map(|relation| relation.as_ref().map(|(up, _)| *up)),
        )
        .chain(std::iter::once(invocation_relation.0))
        .max()
        .unwrap_or(0);
    let primary = roots
        .iter()
        .filter(|root| workspace.starts_with(root))
        .max_by_key(|root| root.components().count())
        .cloned()
        .ok_or_else(|| {
            format!(
                "no Cargo source root contains workspace {}",
                workspace.display()
            )
        })?;
    let workspace_suffix = workspace
        .strip_prefix(&primary)
        .expect("selected primary root contains workspace");
    let suffix_len = workspace_suffix.components().count();
    let padding = max_up.saturating_sub(1 + suffix_len);
    let mut primary_virtual = PathBuf::from("source");
    for index in 0..padding {
        primary_virtual.push(format!("anchor-{index}"));
    }
    primary_virtual.push("primary");
    let workspace_virtual = primary_virtual.join(workspace_suffix);
    let invocation_virtual = virtual_path_for_relation(
        &workspace_virtual,
        invocation_relation.0,
        &invocation_relation.1,
    )?;

    let mut target_specs = pending_target_specs
        .into_iter()
        .map(|(index, source)| CargoTargetSpecPlan {
            source,
            virtual_path: workspace_virtual
                .join(".ktstr-target-specs")
                .join(format!("target-{index}.json")),
            semantic_name: format!("target/{index}"),
        })
        .collect::<Vec<_>>();
    target_specs.sort_by(|left, right| left.semantic_name.cmp(&right.semantic_name));

    let mut cargo_configs = pending_configs
        .into_iter()
        .zip(config_relations)
        .map(|(config, relation)| {
            let virtual_path = relation
                .map(|(upward, downward)| {
                    virtual_path_for_relation(&workspace_virtual, upward, &downward)
                })
                .transpose()?;
            let explicit_argument = config.explicit_argument.then(|| config.source.clone());
            Ok(CargoConfigPlan {
                source: config.source,
                virtual_path,
                semantic_name: config.semantic_name,
                explicit_argument,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    cargo_configs.sort_by(|left, right| left.semantic_name.cmp(&right.semantic_name));

    let mut planned = Vec::with_capacity(roots.len());
    let assignment_roots = roots.clone();
    for (root, (upward, downward)) in roots.into_iter().zip(relations) {
        let virtual_path = virtual_path_for_relation(&workspace_virtual, upward, &downward)?;
        if root == primary && virtual_path != primary_virtual {
            return Err(format!(
                "stable source translation mismatch for primary root {}: {} != {}",
                root.display(),
                virtual_path.display(),
                primary_virtual.display()
            ));
        }
        let is_git = root_kinds[&root];
        let (mut input_paths, gitlinks, git_status_semantics, git_metadata) = if is_git {
            let inputs = git_inputs
                .remove(&root)
                .expect("every planned git root has one computed path set");
            (
                inputs.paths,
                inputs.gitlinks,
                inputs.status_semantics,
                inputs.metadata,
            )
        } else {
            (
                non_git_source_input_paths(&root, &outputs)?,
                BTreeMap::new(),
                BTreeSet::new(),
                None,
            )
        };
        for path in &declared_package_inputs {
            let owner = assignment_roots
                .iter()
                .filter(|candidate| path.starts_with(candidate))
                .max_by_key(|candidate| candidate.components().count());
            if owner == Some(&root)
                && !source_path_is_excluded(&root, path, &outputs)
                && let Ok(relative) = path.strip_prefix(&root)
            {
                input_paths.insert(relative.to_path_buf());
            }
        }
        input_paths
            .retain(|relative| !source_path_is_excluded(&root, &root.join(relative), &outputs));
        for config in &cargo_configs {
            if let Ok(relative) = config.source.strip_prefix(&root) {
                input_paths.remove(relative);
            }
        }
        let digest = source_tree_digest(&root, &input_paths, &gitlinks)?;
        let git_head = is_git.then(|| git_head(&root)).flatten();
        planned.push(SourceRootPlan {
            semantic_name: virtual_path.to_string_lossy().into_owned(),
            source: root,
            virtual_path,
            is_git,
            input_paths,
            git_head,
            git_status_semantics,
            git_metadata,
            digest,
        });
    }
    planned.sort_by(|left, right| left.semantic_name.cmp(&right.semantic_name));
    let mut hasher = fixed_hasher();
    hash_bytes(&mut hasher, b"ktstr-stable-cargo-source");
    hasher.write_u32(SOURCE_IDENTITY_SCHEMA);
    hash_bytes(&mut hasher, b"invocation-relative-to-workspace");
    hash_path_relation(&mut hasher, &invocation_relation);
    for root in &planned {
        hash_bytes(&mut hasher, root.semantic_name.as_bytes());
        hasher.write_u64(root.digest);
        hash_bytes(
            &mut hasher,
            root.git_head
                .as_deref()
                .unwrap_or("<non-git-or-unborn>")
                .as_bytes(),
        );
        for status in &root.git_status_semantics {
            hash_bytes(&mut hasher, status);
        }
        if let Some(metadata) = &root.git_metadata {
            hasher.write_u64(metadata.semantic_digest);
        }
    }
    for config in &cargo_configs {
        hash_bytes(&mut hasher, config.semantic_name.as_bytes());
        hash_bytes(
            &mut hasher,
            &read_cargo_config_identity(&config.source, &outputs)?,
        );
    }
    for target in &target_specs {
        hash_bytes(&mut hasher, target.semantic_name.as_bytes());
        let bytes = std::fs::read(&target.source).map_err(|error| {
            format!(
                "read custom target specification {}: {error}",
                target.source.display()
            )
        })?;
        hash_bytes(&mut hasher, &bytes);
    }
    Ok(SourceLayout {
        workspace,
        invocation,
        invocation_relation,
        workspace_virtual,
        invocation_virtual,
        roots: planned,
        cargo_configs,
        target_specs,
        declared_package_inputs,
        outputs,
        identity: hasher.finish(),
    })
}

fn capture_source_layout(
    layout: &SourceLayout,
) -> Result<ktstr::cache::artifact_tree::ArtifactTreeSource, String> {
    let mut source = ktstr::cache::artifact_tree::ArtifactTreeSource::new();
    let mut implicit_directories = BTreeSet::new();
    implicit_directories.insert(layout.invocation_virtual.clone());
    let mut invocation_parent = layout.invocation_virtual.parent();
    while let Some(path) = invocation_parent.filter(|path| !path.as_os_str().is_empty()) {
        implicit_directories.insert(path.to_path_buf());
        invocation_parent = path.parent();
    }
    let mut files = BTreeMap::<PathBuf, PathBuf>::new();
    let mut generated_git_configs = Vec::<(PathBuf, Vec<u8>)>::new();
    for root in &layout.roots {
        implicit_directories.insert(root.virtual_path.clone());
        let mut parent = root.virtual_path.parent();
        while let Some(path) = parent.filter(|path| !path.as_os_str().is_empty()) {
            implicit_directories.insert(path.to_path_buf());
            parent = path.parent();
        }
        for relative in &root.input_paths {
            if crate::interrupt::INTERRUPTED.load(Ordering::Acquire) {
                return Err("stable Cargo source capture interrupted".to_string());
            }
            let path = root.source.join(relative);
            let metadata = match std::fs::symlink_metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(format!(
                        "stat stable Cargo source {}: {error}",
                        path.display()
                    ));
                }
            };
            if metadata.is_dir() && !metadata.file_type().is_symlink() {
                continue;
            }
            let destination = root.virtual_path.join(relative);
            let mut parent = destination.parent();
            while let Some(directory) = parent.filter(|path| !path.as_os_str().is_empty()) {
                implicit_directories.insert(directory.to_path_buf());
                parent = directory.parent();
            }
            match files.entry(destination) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(path);
                }
                std::collections::btree_map::Entry::Occupied(entry) if entry.get() == &path => {}
                std::collections::btree_map::Entry::Occupied(entry) => {
                    return Err(format!(
                        "stable Cargo source path {} aliases {} and {}",
                        entry.key().display(),
                        entry.get().display(),
                        path.display()
                    ));
                }
            }
        }
        if let Some(metadata) = &root.git_metadata {
            let git_root = root.virtual_path.join(".git");
            implicit_directories.insert(git_root.clone());
            for (relative, path) in &metadata.files {
                let destination = git_root.join(relative);
                let mut parent = destination.parent();
                while let Some(directory) = parent.filter(|path| !path.as_os_str().is_empty()) {
                    implicit_directories.insert(directory.to_path_buf());
                    parent = directory.parent();
                }
                files.insert(destination, path.clone());
            }
            generated_git_configs.push((git_root.join("config"), metadata.config.clone()));
        }
    }
    for directory in implicit_directories {
        source
            .insert_directory(directory, 0o555)
            .map_err(|error| format!("capture stable Cargo source parent: {error:#}"))?;
    }
    for (destination, path) in files {
        source
            .insert_immutable_path(destination, &path)
            .map_err(|error| {
                format!("capture stable Cargo source {}: {error:#}", path.display())
            })?;
    }
    for (destination, bytes) in generated_git_configs {
        source
            .insert_bytes(&destination, &bytes, 0o444)
            .map_err(|error| {
                format!(
                    "capture cache-owned Git config {}: {error:#}",
                    destination.display()
                )
            })?;
    }
    for config in &layout.cargo_configs {
        let Some(destination) = &config.virtual_path else {
            continue;
        };
        let bytes = std::fs::read(&config.source).map_err(|error| {
            format!(
                "read Cargo config for stable source {}: {error}",
                config.source.display()
            )
        })?;
        source
            .insert_bytes(destination, &bytes, 0o444)
            .map_err(|error| {
                format!(
                    "capture Cargo config {} at {}: {error:#}",
                    config.source.display(),
                    destination.display()
                )
            })?;
    }
    for target in &layout.target_specs {
        let bytes = std::fs::read(&target.source).map_err(|error| {
            format!(
                "read custom target specification for stable source {}: {error}",
                target.source.display()
            )
        })?;
        source
            .insert_bytes(&target.virtual_path, &bytes, 0o444)
            .map_err(|error| {
                format!(
                    "capture custom target specification {} at {}: {error:#}",
                    target.source.display(),
                    target.virtual_path.display()
                )
            })?;
    }
    Ok(source)
}

fn recompute_source_identity(layout: &SourceLayout) -> Result<u64, String> {
    let mut hasher = fixed_hasher();
    hash_bytes(&mut hasher, b"ktstr-stable-cargo-source");
    hasher.write_u32(SOURCE_IDENTITY_SCHEMA);
    hash_bytes(&mut hasher, b"invocation-relative-to-workspace");
    hash_path_relation(&mut hasher, &layout.invocation_relation);
    for root in &layout.roots {
        let (mut input_paths, gitlinks, git_status_semantics, git_metadata) = if root.is_git {
            let inputs = git_source_input_paths(&root.source)?;
            (
                inputs.paths,
                inputs.gitlinks,
                inputs.status_semantics,
                inputs.metadata,
            )
        } else {
            (
                non_git_source_input_paths(&root.source, &layout.outputs)?,
                BTreeMap::new(),
                BTreeSet::new(),
                None,
            )
        };
        for path in &layout.declared_package_inputs {
            let owner = layout
                .roots
                .iter()
                .filter(|candidate| path.starts_with(&candidate.source))
                .max_by_key(|candidate| candidate.source.components().count());
            if owner.map(|owner| &owner.source) == Some(&root.source)
                && let Ok(relative) = path.strip_prefix(&root.source)
                && !source_path_is_excluded(&root.source, path, &layout.outputs)
            {
                input_paths.insert(relative.to_path_buf());
            }
        }
        input_paths.retain(|relative| {
            !source_path_is_excluded(&root.source, &root.source.join(relative), &layout.outputs)
        });
        for config in &layout.cargo_configs {
            if let Ok(relative) = config.source.strip_prefix(&root.source) {
                input_paths.remove(relative);
            }
        }
        hash_bytes(&mut hasher, root.semantic_name.as_bytes());
        hasher.write_u64(source_tree_digest(&root.source, &input_paths, &gitlinks)?);
        hash_bytes(
            &mut hasher,
            git_head(&root.source)
                .as_deref()
                .unwrap_or("<non-git-or-unborn>")
                .as_bytes(),
        );
        for status in &git_status_semantics {
            hash_bytes(&mut hasher, status);
        }
        if let Some(metadata) = git_metadata {
            hasher.write_u64(metadata.semantic_digest);
        }
    }
    for config in &layout.cargo_configs {
        hash_bytes(&mut hasher, config.semantic_name.as_bytes());
        hash_bytes(
            &mut hasher,
            &read_cargo_config_identity(&config.source, &layout.outputs)?,
        );
    }
    for target in &layout.target_specs {
        hash_bytes(&mut hasher, target.semantic_name.as_bytes());
        let bytes = std::fs::read(&target.source).map_err(|error| {
            format!(
                "read custom target specification {}: {error}",
                target.source.display()
            )
        })?;
        hash_bytes(&mut hasher, &bytes);
    }
    Ok(hasher.finish())
}

fn command_fingerprint(
    program: &OsStr,
    arguments: &[&str],
    current_dir: &Path,
) -> Result<Vec<u8>, String> {
    let resolved = resolve_program(program, current_dir)?;
    let output = std::process::Command::new(program)
        .args(arguments)
        .current_dir(current_dir)
        .output()
        .map_err(|error| {
            format!(
                "run build-compatibility probe {} {}: {error}",
                program.to_string_lossy(),
                arguments.join(" ")
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "build-compatibility probe {} {} failed with {}: {}",
            program.to_string_lossy(),
            arguments.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let mut fingerprint = Vec::new();
    fingerprint.extend_from_slice(b"ktstr-tool-v1\0");
    fingerprint.extend_from_slice(
        format!(
            "{:016x}\0",
            ktstr::cache::content_file_digest(&resolved).map_err(|error| {
                format!("digest build tool {}: {error:#}", resolved.display())
            })?
        )
        .as_bytes(),
    );
    fingerprint.extend_from_slice(&output.stdout);
    fingerprint.extend_from_slice(&output.stderr);
    Ok(fingerprint)
}

fn resolve_program(program: &OsStr, current_dir: &Path) -> Result<PathBuf, String> {
    let path = Path::new(program);
    if path.components().count() > 1 {
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            current_dir.join(path)
        };
        return std::fs::canonicalize(&path)
            .map_err(|error| format!("resolve build tool {}: {error}", path.display()));
    }
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    for directory in std::env::split_paths(&path_env) {
        let candidate = directory.join(program);
        if candidate.is_file() {
            return std::fs::canonicalize(&candidate)
                .map_err(|error| format!("resolve build tool {}: {error}", candidate.display()));
        }
    }
    Err(format!(
        "build tool {} was not found in PATH",
        program.to_string_lossy()
    ))
}

fn selected_tool(environment: &[&str], fallback: &str) -> std::ffi::OsString {
    selected_optional_tool(environment).unwrap_or_else(|| fallback.into())
}

fn selected_optional_tool(environment: &[&str]) -> Option<std::ffi::OsString> {
    environment
        .iter()
        .find_map(|name| std::env::var_os(name).filter(|value| !value.is_empty()))
}

fn optional_tool_fingerprint(
    label: &str,
    program: &OsStr,
    arguments: &[&str],
    workspace: &Path,
) -> (String, Vec<u8>) {
    let fingerprint = command_fingerprint(program, arguments, workspace).unwrap_or_else(|error| {
        format!("unavailable:{}:{error}", program.to_string_lossy()).into_bytes()
    });
    (label.to_string(), fingerprint)
}

fn tool_fingerprints(mode: &str, workspace: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    // Cargo's dedicated tool variables override their `[build]` config
    // counterparts; the historical RUSTC variables in turn remain the
    // highest-precedence direct overrides. Fingerprint the one executable
    // Cargo will actually invoke, including the CARGO_BUILD_* fallback which
    // the sanitized scheduler environment deliberately does not hash.
    let rustc = selected_tool(&["RUSTC", "CARGO_BUILD_RUSTC"], "rustc");
    let mut tools = vec![
        (
            "cargo".to_string(),
            command_fingerprint(OsStr::new("cargo"), &["-vV"], workspace)?,
        ),
        (
            "rustc".to_string(),
            command_fingerprint(&rustc, &["-vV"], workspace)?,
        ),
    ];
    if mode.contains("nextest") {
        tools.push((
            "nextest".to_string(),
            command_fingerprint(OsStr::new("cargo"), &["nextest", "--version"], workspace)?,
        ));
    }
    if mode.contains("coverage") || mode.contains("llvm-cov") {
        tools.push((
            "llvm-cov".to_string(),
            command_fingerprint(OsStr::new("cargo"), &["llvm-cov", "--version"], workspace)?,
        ));
    }
    let clang = selected_tool(&["BPF_CLANG", "CLANG"], "clang");
    let cc = selected_tool(&["CC"], "cc");
    let ar = selected_tool(&["AR"], "ar");
    let pkg_config = selected_tool(&["PKG_CONFIG"], "pkg-config");
    let ld = selected_tool(&["LD"], "ld");
    let bpftool = selected_tool(&["BPFTOOL"], "bpftool");
    let pahole = selected_tool(&["PAHOLE"], "pahole");
    let strip = selected_tool(&["STRIP"], "strip");
    let objcopy = selected_tool(&["OBJCOPY"], "objcopy");
    tools.extend([
        optional_tool_fingerprint("clang", &clang, &["--version"], workspace),
        optional_tool_fingerprint("cc", &cc, &["--version"], workspace),
        optional_tool_fingerprint("ar", &ar, &["--version"], workspace),
        optional_tool_fingerprint("pkg-config", &pkg_config, &["--version"], workspace),
        optional_tool_fingerprint("ld", &ld, &["--version"], workspace),
        optional_tool_fingerprint("mold", OsStr::new("mold"), &["--version"], workspace),
        optional_tool_fingerprint("bpftool", &bpftool, &["version"], workspace),
        optional_tool_fingerprint("pahole", &pahole, &["--version"], workspace),
        optional_tool_fingerprint("strip", &strip, &["--version"], workspace),
        optional_tool_fingerprint("objcopy", &objcopy, &["--version"], workspace),
        optional_tool_fingerprint(
            "llvm-strip",
            OsStr::new("llvm-strip"),
            &["--version"],
            workspace,
        ),
        optional_tool_fingerprint(
            "llvm-objcopy",
            OsStr::new("llvm-objcopy"),
            &["--version"],
            workspace,
        ),
    ]);
    for (label, variables) in [
        (
            "rustc-wrapper",
            ["RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WRAPPER"],
        ),
        (
            "rustc-workspace-wrapper",
            [
                "RUSTC_WORKSPACE_WRAPPER",
                "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
            ],
        ),
    ] {
        if let Some(program) = selected_optional_tool(&variables) {
            let fingerprint = resolve_program(&program, workspace)
                .and_then(|path| {
                    ktstr::cache::content_file_digest(&path)
                        .map(|digest| format!("{digest:016x}").into_bytes())
                        .map_err(|error| {
                            format!("digest build wrapper {}: {error:#}", path.display())
                        })
                })
                .unwrap_or_else(|error| format!("unavailable:{error}").into_bytes());
            tools.push((label.to_string(), fingerprint));
        }
    }
    Ok(tools)
}

pub(crate) struct IdentityPlan {
    pub identity: u64,
    pub components: serde_json::Value,
    source: SourceLayout,
}

pub(crate) struct StableCargoSource {
    _tree: ktstr::cache::artifact_tree::StableArtifactTree,
    pub workspace_root: PathBuf,
    pub invocation_root: PathBuf,
    pub original_workspace_root: PathBuf,
    pub original_invocation_root: PathBuf,
    cargo_argument_remaps: BTreeMap<PathBuf, PathBuf>,
    cargo_root_remaps: Vec<(PathBuf, PathBuf)>,
}

impl StableCargoSource {
    fn execution_source_root_remaps(&self) -> Vec<(PathBuf, PathBuf)> {
        self.cargo_root_remaps
            .iter()
            .map(|(writable, stable)| (stable.clone(), writable.clone()))
            .collect()
    }

    /// Remap explicit Cargo/nextest files and inline path-valued config onto
    /// immutable files and source roots captured in this stable source. Named
    /// target triples and non-path inline config remain unchanged.
    pub(crate) fn remap_cargo_args(&self, arguments: &[String]) -> Vec<String> {
        let absolute_argument_path = |value: &str| {
            let path = Path::new(value);
            let absolute = if path.is_absolute() {
                path.to_path_buf()
            } else {
                self.original_invocation_root.join(path)
            };
            canonical_or_lexical(&absolute)
        };
        let remap_explicit_file = |value: &str| {
            let absolute = absolute_argument_path(value);
            self.cargo_argument_remaps
                .get(&absolute)
                .cloned()
                .map(|path| path.display().to_string())
        };
        let remap_source_path = |value: &str| {
            let absolute = absolute_argument_path(value);
            self.cargo_argument_remaps
                .get(&absolute)
                .cloned()
                .or_else(|| {
                    self.cargo_root_remaps.iter().find_map(|(source, stable)| {
                        absolute
                            .strip_prefix(source)
                            .ok()
                            .map(|relative| stable.join(relative))
                    })
                })
                .map(|path| path.display().to_string())
        };
        let remap_config_value = |value: &str| {
            if let Some((key, path)) = crate::feature_discovery::cargo_inline_config_path(value) {
                return remap_source_path(&path).map(|path| {
                    format!(
                        "{key}={}",
                        serde_json::to_string(&path)
                            .expect("a Cargo argument path is representable as JSON")
                    )
                });
            }
            remap_explicit_file(value)
        };

        let mut remapped = Vec::with_capacity(arguments.len());
        let mut index = 0;
        while index < arguments.len() {
            let argument = &arguments[index];
            if matches!(argument.as_str(), "--config" | "--config-file" | "--target") {
                remapped.push(argument.clone());
                index += 1;
                if let Some(value) = arguments.get(index) {
                    let value = if argument == "--config" {
                        remap_config_value(value)
                    } else if argument == "--config-file" {
                        remap_source_path(value)
                    } else {
                        remap_explicit_file(value)
                    };
                    remapped.push(value.unwrap_or_else(|| arguments[index].clone()));
                }
                index += 1;
                continue;
            }
            if let Some(value) = argument.strip_prefix("--config=") {
                remapped.push(
                    remap_config_value(value)
                        .map(|value| format!("--config={value}"))
                        .unwrap_or_else(|| argument.clone()),
                );
            } else if let Some(value) = argument.strip_prefix("--config-file=") {
                remapped.push(
                    remap_source_path(value)
                        .map(|value| format!("--config-file={value}"))
                        .unwrap_or_else(|| argument.clone()),
                );
            } else if let Some(value) = argument.strip_prefix("--target=") {
                remapped.push(
                    remap_explicit_file(value)
                        .map(|value| format!("--target={value}"))
                        .unwrap_or_else(|| argument.clone()),
                );
            } else {
                remapped.push(argument.clone());
            }
            index += 1;
        }
        remapped
    }
}

fn materialize_stable_source(
    source: &SourceLayout,
    progress_label: &str,
) -> Result<StableCargoSource, String> {
    let cache_root = ktstr::cache::cargo_artifact_tree_cache_root()
        .map_err(|error| format!("resolve Cargo artifact cache root: {error:#}"))?;
    let cache = ktstr::cache::artifact_tree::ArtifactTreeCache::new(
        cache_root.join("stable-source-records-v2"),
    );
    let tree = cache
        .load_or_build_stable_with_validators(
            source.identity,
            &cache_root.join("stable-sources-v2"),
            progress_label,
            || Ok(true),
            || {
                Ok(
                    recompute_source_identity(source).map_err(anyhow::Error::msg)?
                        == source.identity,
                )
            },
            || crate::interrupt::INTERRUPTED.load(Ordering::Acquire),
            || capture_source_layout(source).map_err(anyhow::Error::msg),
        )
        .map_err(|error| format!("materialize stable Cargo source: {error:#}"))?;
    let workspace_root = tree.root().join(&source.workspace_virtual);
    let invocation_root = tree.root().join(&source.invocation_virtual);
    let mut cargo_argument_remaps = BTreeMap::new();
    for config in &source.cargo_configs {
        if let (Some(argument), Some(virtual_path)) =
            (&config.explicit_argument, &config.virtual_path)
        {
            cargo_argument_remaps.insert(argument.clone(), tree.root().join(virtual_path));
        }
    }
    for target in &source.target_specs {
        cargo_argument_remaps.insert(
            target.source.clone(),
            tree.root().join(&target.virtual_path),
        );
    }
    let mut cargo_root_remaps = source
        .roots
        .iter()
        .map(|root| (root.source.clone(), tree.root().join(&root.virtual_path)))
        .collect::<Vec<_>>();
    cargo_root_remaps
        .sort_by_key(|(source, _)| std::cmp::Reverse(source.as_os_str().as_bytes().len()));
    let stable = StableCargoSource {
        _tree: tree,
        workspace_root,
        invocation_root,
        original_workspace_root: source.workspace.clone(),
        original_invocation_root: source.invocation.clone(),
        cargo_argument_remaps,
        cargo_root_remaps,
    };
    Ok(stable)
}

impl IdentityPlan {
    /// Materialize the immutable pathname used for the cold Cargo build and
    /// retained for every later cross-checkout cache hit.
    pub(crate) fn stable_source(&self, progress_label: &str) -> Result<StableCargoSource, String> {
        materialize_stable_source(&self.source, progress_label)
    }

    /// Reuse or produce the exact nextest execution closure.
    ///
    /// The callback is invoked only on a miss and receives the stable source
    /// pathname and deterministic persistent output pathname which Cargo must
    /// use for that build. It must return an
    /// [`ArtifactTreeSource`](ktstr::cache::artifact_tree::ArtifactTreeSource) while still
    /// holding exclusive ownership of every Cargo output it captured (normally
    /// by calling [`capture_source`] inside the output-lease postprocess
    /// callback).
    pub(crate) fn load_or_build<F>(
        &self,
        progress_label: &str,
        build: F,
    ) -> Result<MaterializedNextestArtifacts, String>
    where
        F: FnOnce(
            &StableCargoSource,
            &StableCargoBuild,
        ) -> Result<ktstr::cache::artifact_tree::ArtifactTreeSource, String>,
    {
        let stable_source = self.stable_source(progress_label)?;
        let cache_root = ktstr::cache::cargo_artifact_tree_cache_root()
            .map_err(|error| format!("resolve Cargo artifact cache root: {error:#}"))?;
        let cache = ktstr::cache::artifact_tree::ArtifactTreeCache::new(
            cache_root.join("nextest-build-records-v1"),
        );
        let tree = cache
            .load_or_build_with_stable_cargo_output_validators(
                self.identity,
                &cache_root.join("stable-builds-v1"),
                &cache_root.join("nextest-materialized-v1"),
                progress_label,
                || Ok(true),
                // `stable_source` was captured from the live checkout and
                // publication-validated before the producer started. Cargo
                // consumes only that immutable tree, so a later edit to the
                // writable checkout cannot change this build's inputs. A
                // second live-tree scan here both wastes the cold-build tail
                // and incorrectly rejects the coherent snapshot Cargo used.
                || Ok(true),
                || crate::interrupt::INTERRUPTED.load(Ordering::Acquire),
                |stable_build| build(&stable_source, stable_build).map_err(anyhow::Error::msg),
            )
            .map_err(|error| format!("reuse or build nextest artifact closure: {error:#}"))?;
        if let Err(error) =
            tree.persist_decision_diagnostic("nextest-build", self.components.clone())
        {
            tracing::warn!(error = %error, "could not persist nextest artifact-cache decision");
        }
        finish_materialization(tree, stable_source)
    }
}

fn artifact_identity(
    mode: &str,
    source_identity: u64,
    normalized_args: &[Vec<u8>],
    external_packages: &BTreeSet<String>,
    environment: &[(std::ffi::OsString, Vec<u8>)],
    tools: &[(String, Vec<u8>)],
) -> u64 {
    let mut hasher = fixed_hasher();
    hash_bytes(&mut hasher, b"ktstr-nextest-artifact-tree");
    hasher.write_u32(IDENTITY_SCHEMA);
    hash_bytes(&mut hasher, mode.as_bytes());
    hasher.write_u64(source_identity);
    for argument in normalized_args {
        hash_bytes(&mut hasher, argument);
    }
    for package in external_packages {
        hash_bytes(&mut hasher, package.as_bytes());
    }
    for (name, value) in environment {
        hash_bytes(&mut hasher, name.as_os_str().as_bytes());
        hash_bytes(&mut hasher, value);
    }
    for (name, fingerprint) in tools {
        hash_bytes(&mut hasher, name.as_bytes());
        hash_bytes(&mut hasher, fingerprint);
    }
    hasher.finish()
}

pub(crate) fn identity_plan(
    metadata: &cargo_metadata::Metadata,
    mode: &str,
    build_surface: &[String],
    output_roots: &[PathBuf],
) -> Result<IdentityPlan, String> {
    let invocation = std::env::current_dir()
        .map_err(|error| format!("read Cargo invocation directory: {error}"))?;
    identity_plan_for_invocation(metadata, mode, build_surface, output_roots, &invocation)
}

/// Plan one Cargo producer from the directory it will actually execute in.
///
/// Harness producers execute from the user's invocation directory, while a
/// declared scheduler batch executes from its workspace root. Keeping that
/// distinction explicit makes Cargo's ancestor-config discovery and relative
/// file arguments part of the same identity used by the cold build.
pub(crate) fn identity_plan_for_invocation(
    metadata: &cargo_metadata::Metadata,
    mode: &str,
    build_surface: &[String],
    output_roots: &[PathBuf],
    invocation: &Path,
) -> Result<IdentityPlan, String> {
    let source = plan_source_layout(metadata, output_roots, build_surface, invocation)?;
    let workspace = &source.workspace;
    let normalized_args = build_surface
        .iter()
        .map(|argument| normalized_argument(argument, &source))
        .collect::<Vec<_>>();
    let environment = crate::verifier::scheduler_build_environment(workspace, &|| {
        crate::interrupt::INTERRUPTED.load(Ordering::Acquire)
    })?;
    let external_packages = metadata
        .packages
        .iter()
        .filter_map(|package| {
            package
                .source
                .as_ref()
                .map(|source| format!("{}@{}:{}", package.name, package.version, source))
        })
        .collect::<BTreeSet<_>>();
    let tools = tool_fingerprints(mode, workspace)?;
    let identity = artifact_identity(
        mode,
        source.identity,
        &normalized_args,
        &external_packages,
        &environment,
        &tools,
    );
    let normalized_build_args = normalized_args
        .iter()
        .map(|value| String::from_utf8_lossy(value))
        .collect::<Vec<_>>();
    let source_digests = source
        .roots
        .iter()
        .map(|root| {
            serde_json::json!({
                "name": &root.semantic_name,
                "digest": format!("{:016x}", root.digest),
            })
        })
        .collect::<Vec<_>>();
    let tool_fingerprint_components = tools
        .iter()
        .map(|(name, value)| {
            let mut tool_hasher = fixed_hasher();
            hash_bytes(&mut tool_hasher, value);
            serde_json::json!({
                "name": name,
                "digest": format!("{:016x}", tool_hasher.finish()),
            })
        })
        .collect::<Vec<_>>();
    let environment_digest = {
        let mut env_hasher = fixed_hasher();
        for (name, value) in &environment {
            hash_bytes(&mut env_hasher, name.as_os_str().as_bytes());
            hash_bytes(&mut env_hasher, value);
        }
        format!("{:016x}", env_hasher.finish())
    };
    // Keep cache-miss diagnostics actionable without disclosing environment
    // values. The aggregate digest proves that an environment differs, but it
    // cannot identify the remaining non-semantic coordinate when otherwise
    // identical producers run under separate service instances. Per-name
    // value digests let operators compare those identities safely; the actual
    // values never enter the diagnostic artifact.
    let environment_components = environment_diagnostic_components(&environment);
    Ok(IdentityPlan {
        identity,
        components: serde_json::json!({
            "schema": IDENTITY_SCHEMA,
            "mode": mode,
            "normalized_build_args": normalized_build_args,
            "stable_source_identity": format!("{:016x}", source.identity),
            "source_digests": source_digests,
            "external_packages": external_packages,
            "tool_fingerprints": tool_fingerprint_components,
            "environment_digest": environment_digest,
            "environment_components": environment_components,
        }),
        source,
    })
}

fn environment_diagnostic_components(
    environment: &[(std::ffi::OsString, Vec<u8>)],
) -> Vec<serde_json::Value> {
    environment
        .iter()
        .map(|(name, value)| {
            let mut value_hasher = fixed_hasher();
            hash_bytes(&mut value_hasher, value);
            serde_json::json!({
                "name": name.to_string_lossy(),
                "value_digest": format!("{:016x}", value_hasher.finish()),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    struct EnvVarGuard {
        key: std::ffi::OsString,
        original: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> Self {
            let key = key.as_ref().to_owned();
            let original = std::env::var_os(&key);
            // SAFETY: every test in this module holds `lock_env()` for the
            // complete save/mutate/restore window.
            unsafe { std::env::set_var(&key, value) };
            Self { key, original }
        }

        fn remove(key: impl AsRef<OsStr>) -> Self {
            let key = key.as_ref().to_owned();
            let original = std::env::var_os(&key);
            // SAFETY: every test in this module holds `lock_env()` for the
            // complete save/mutate/restore window.
            unsafe { std::env::remove_var(&key) };
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                // SAFETY: the owning test still holds `lock_env()`.
                Some(value) => unsafe { std::env::set_var(&self.key, value) },
                None => unsafe { std::env::remove_var(&self.key) },
            }
        }
    }

    fn git_output(root: &Path, args: &[&str]) -> Vec<u8> {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .env("GIT_OPTIONAL_LOCKS", "0")
            .output()
            .expect("run git fixture command");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    }

    #[test]
    fn environment_diagnostics_name_inputs_without_disclosing_values() {
        let environment = vec![
            (
                std::ffi::OsString::from("A_STABLE_INPUT"),
                b"ordinary-value".to_vec(),
            ),
            (
                std::ffi::OsString::from("Z_SECRET_INPUT"),
                b"must-never-appear".to_vec(),
            ),
        ];

        let components = environment_diagnostic_components(&environment);
        let rendered = serde_json::to_string(&components).unwrap();
        assert!(rendered.contains("A_STABLE_INPUT"));
        assert!(rendered.contains("Z_SECRET_INPUT"));
        assert!(!rendered.contains("ordinary-value"));
        assert!(!rendered.contains("must-never-appear"));
        assert_ne!(
            components[0]["value_digest"], components[1]["value_digest"],
            "different values should remain distinguishable by digest",
        );
    }

    #[test]
    fn separate_build_directory_materializes_empty_target_remap_root() {
        let _environment = lock_env();
        let temp = tempfile::tempdir().unwrap();
        let cache_root = temp.path().join("cache");
        let _cache = EnvVarGuard::set(ktstr::KTSTR_CACHE_DIR_ENV, &cache_root);
        let target = temp.path().join("producer-target");
        let build = temp.path().join("producer-build");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::create_dir_all(&build).unwrap();
        let metadata = serde_json::json!({
            "rust-build-meta": {
                "target-directory": target,
                "build-directory": build,
            },
            "rust-binaries": {},
        });
        let source = capture_source(&serde_json::to_vec(&metadata).unwrap(), b"{}").unwrap();
        let materialized =
            ktstr::cache::artifact_tree::ArtifactTreeCache::new(cache_root.join("records"))
                .load_or_build(
                    0x5e9a_7a7e,
                    &cache_root.join("materialized"),
                    "separate build-dir fixture",
                    || Ok(true),
                    || false,
                    || Ok(source),
                )
                .unwrap();

        assert!(
            materialized.root().join("target").is_dir(),
            "--target-dir-remap must remain canonicalizable even when the target tree is empty",
        );
        assert!(
            materialized.root().join("build").is_dir(),
            "--build-dir-remap must retain its structural root independently",
        );
    }

    #[test]
    fn coverage_producer_profiles_are_compacted_outside_the_report_scan_root() {
        let _environment = lock_env();
        let temp = tempfile::tempdir().unwrap();
        let cache_root = temp.path().join("cache");
        let _cache = EnvVarGuard::set(ktstr::KTSTR_CACHE_DIR_ENV, &cache_root);
        let target = temp.path().join("producer-target");
        let build = temp.path().join("producer-build");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::create_dir_all(&build).unwrap();
        std::fs::write(target.join("producer-%p.profraw"), b"raw-profile").unwrap();
        let producer_profdata = target.join("producer.profdata");
        std::fs::write(&producer_profdata, b"merged-profile").unwrap();
        let nested = target.join("nested.profraw");
        std::fs::create_dir(&nested).unwrap();
        std::fs::write(nested.join("hidden.profraw"), b"nested-profile").unwrap();
        let metadata = serde_json::json!({
            "rust-build-meta": {
                "target-directory": target,
                "build-directory": build,
            },
            "rust-binaries": {},
        });
        let source = capture_source_with_producer_profdata(
            &serde_json::to_vec(&metadata).unwrap(),
            b"{}",
            Some(&producer_profdata),
        )
        .unwrap();
        let materialized =
            ktstr::cache::artifact_tree::ArtifactTreeCache::new(cache_root.join("records"))
                .load_or_build(
                    0xc0ff_ee42,
                    &cache_root.join("materialized"),
                    "coverage profile fixture",
                    || Ok(true),
                    || false,
                    || Ok(source),
                )
                .unwrap();
        let materialized_target = materialized.root().join("target");

        assert!(!materialized_target.join("producer-%p.profraw").exists());
        assert!(
            !materialized_target.join("ktstr-profraw").exists(),
            "raw profiles must not be hidden elsewhere in the cached target",
        );
        assert!(!materialized_target.join("producer.profdata").exists());
        assert!(!materialized_target.join("nested.profraw").exists());
        assert_eq!(
            std::fs::read(materialized.root().join(COVERAGE_PRODUCER_PROFDATA_PATH),).unwrap(),
            b"merged-profile",
        );
    }

    #[test]
    fn reused_nextest_execution_context_keeps_stable_sources_build_only() {
        let stable = Path::new("/cache/stable/source/primary");
        let writable = Path::new("/work/checkout");
        let stable_external = Path::new("/cache/stable/source/external-scheduler");
        let writable_external = Path::new("/work/path-deps/scheduler");
        let invocation = writable.join("member");
        let remaps = vec![
            (stable.to_path_buf(), writable.to_path_buf()),
            (
                stable_external.to_path_buf(),
                writable_external.to_path_buf(),
            ),
        ];
        let mut command = std::process::Command::new("cargo");
        command
            .current_dir(stable)
            .env(ktstr::KTSTR_SOURCE_ROOT_REMAPS_ENV, "stale");

        apply_execution_context(&mut command, &remaps, &invocation).unwrap();

        assert_eq!(command.get_current_dir(), Some(invocation.as_path()));
        let environment = command
            .get_envs()
            .map(|(name, value)| (name.to_owned(), value.map(OsStr::to_owned)))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            environment.get(OsStr::new(ktstr::KTSTR_SOURCE_ROOT_REMAPS_ENV)),
            Some(&Some(ktstr::encode_source_root_remaps(&remaps).unwrap())),
            "the complete source-root map must replace stale inherited state",
        );
    }

    #[test]
    fn cargo_build_tool_fallbacks_follow_cargo_precedence() {
        let _environment = lock_env();
        for (primary, cargo_build) in [
            ("RUSTC", "CARGO_BUILD_RUSTC"),
            ("RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WRAPPER"),
            (
                "RUSTC_WORKSPACE_WRAPPER",
                "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
            ),
        ] {
            let _primary_unset = EnvVarGuard::remove(primary);
            let _fallback_unset = EnvVarGuard::remove(cargo_build);
            assert_eq!(selected_optional_tool(&[primary, cargo_build]), None);
            let _fallback = EnvVarGuard::set(cargo_build, "/tool/cargo-build-fallback");
            assert_eq!(
                selected_optional_tool(&[primary, cargo_build]),
                Some(std::ffi::OsString::from("/tool/cargo-build-fallback")),
            );
            let _primary = EnvVarGuard::set(primary, "/tool/direct-override");
            assert_eq!(
                selected_optional_tool(&[primary, cargo_build]),
                Some(std::ffi::OsString::from("/tool/direct-override")),
            );
        }
        assert_eq!(
            selected_tool(&[], "rustc"),
            std::ffi::OsString::from("rustc")
        );
    }

    #[test]
    fn source_digest_is_checkout_root_independent_and_tracks_content() {
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("first");
        let second = temp.path().join("second");
        for root in [&first, &second] {
            std::fs::create_dir_all(root.join("src")).unwrap();
            std::fs::write(
                root.join("Cargo.toml"),
                b"[package]\nname='x'\nversion='0.1.0'\n",
            )
            .unwrap();
            std::fs::write(root.join("src/lib.rs"), b"pub fn x() {}\n").unwrap();
            std::fs::create_dir(root.join("target")).unwrap();
            std::fs::write(root.join("target/noise"), root.as_os_str().as_bytes()).unwrap();
        }
        let first_paths = non_git_source_input_paths(&first, &[first.join("target")]).unwrap();
        let second_paths = non_git_source_input_paths(&second, &[second.join("target")]).unwrap();
        let first_digest = source_tree_digest(&first, &first_paths, &BTreeMap::new()).unwrap();
        assert_eq!(
            first_digest,
            source_tree_digest(&second, &second_paths, &BTreeMap::new()).unwrap()
        );
        std::fs::write(second.join("src/lib.rs"), b"pub fn changed() {}\n").unwrap();
        let changed_paths = non_git_source_input_paths(&second, &[second.join("target")]).unwrap();
        assert_ne!(
            first_digest,
            source_tree_digest(&second, &changed_paths, &BTreeMap::new()).unwrap()
        );
    }

    #[test]
    fn nextest_publish_keeps_verified_stable_snapshot_when_live_source_changes() {
        let _environment = lock_env();
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        let _cache = EnvVarGuard::set(ktstr::KTSTR_CACHE_DIR_ENV, &cache);
        let workspace = temp.path().join("workspace");
        std::fs::create_dir_all(workspace.join("src")).unwrap();
        std::fs::write(
            workspace.join("Cargo.toml"),
            b"[package]\nname='stable-snapshot'\nversion='0.1.0'\n",
        )
        .unwrap();
        let live_source = workspace.join("src/lib.rs");
        std::fs::write(&live_source, b"pub fn value() -> u8 { 1 }\n").unwrap();

        let input_paths =
            BTreeSet::from([PathBuf::from("Cargo.toml"), PathBuf::from("src/lib.rs")]);
        let digest = source_tree_digest(&workspace, &input_paths, &BTreeMap::new()).unwrap();
        let mut source = SourceLayout {
            workspace: workspace.clone(),
            invocation: workspace.clone(),
            invocation_relation: (0, Vec::new()),
            workspace_virtual: PathBuf::from("source/primary"),
            invocation_virtual: PathBuf::from("source/primary"),
            roots: vec![SourceRootPlan {
                source: workspace.clone(),
                virtual_path: PathBuf::from("source/primary"),
                semantic_name: "source/primary".to_string(),
                is_git: false,
                input_paths,
                git_head: None,
                git_status_semantics: BTreeSet::new(),
                git_metadata: None,
                digest,
            }],
            cargo_configs: Vec::new(),
            target_specs: Vec::new(),
            declared_package_inputs: BTreeSet::new(),
            outputs: vec![workspace.join("target"), cache.clone()],
            identity: 0,
        };
        source.identity = recompute_source_identity(&source).unwrap();
        let plan = IdentityPlan {
            identity: 0x51ab_1e5a,
            components: serde_json::json!({"fixture": "stable-snapshot-publication"}),
            source,
        };

        let materialized = plan
            .load_or_build("stable snapshot publication fixture", |stable, build| {
                let captured = stable.workspace_root.join("src/lib.rs");
                assert_eq!(
                    std::fs::read(&captured).unwrap(),
                    b"pub fn value() -> u8 { 1 }\n",
                );
                std::fs::write(&live_source, b"pub fn value() -> u8 { 2 }\n").unwrap();
                std::fs::create_dir_all(&build.target_directory).unwrap();
                let metadata = serde_json::json!({
                    "rust-build-meta": {
                        "target-directory": build.target_directory,
                    },
                    "rust-binaries": {},
                });
                capture_source(&serde_json::to_vec(&metadata).unwrap(), b"{}")
            })
            .expect("a verified immutable snapshot remains publishable");

        assert!(!materialized.cache_hit());
        assert_eq!(
            std::fs::read(materialized.stable_workspace_root.join("src/lib.rs")).unwrap(),
            b"pub fn value() -> u8 { 1 }\n",
        );
        assert_eq!(
            std::fs::read(live_source).unwrap(),
            b"pub fn value() -> u8 { 2 }\n",
        );
    }

    fn identity_normalization_fixture(checkout: &Path) -> (SourceLayout, Vec<String>) {
        let workspace = checkout.join("workspace");
        let config = checkout.join("cargo-config.toml");
        let target = checkout.join("custom-target.json");
        let output = workspace.join("target");
        let layout = SourceLayout {
            workspace: workspace.clone(),
            invocation: workspace.clone(),
            invocation_relation: (0, Vec::new()),
            workspace_virtual: PathBuf::from("source/primary"),
            invocation_virtual: PathBuf::from("source/primary"),
            roots: vec![SourceRootPlan {
                source: workspace.clone(),
                virtual_path: PathBuf::from("source/primary"),
                semantic_name: "source/primary".to_string(),
                is_git: false,
                input_paths: BTreeSet::new(),
                git_head: None,
                git_status_semantics: BTreeSet::new(),
                git_metadata: None,
                digest: 0x51,
            }],
            cargo_configs: vec![CargoConfigPlan {
                source: config.clone(),
                virtual_path: Some(PathBuf::from("config/explicit-0.toml")),
                semantic_name: "explicit/0".to_string(),
                explicit_argument: Some(config.clone()),
            }],
            target_specs: vec![CargoTargetSpecPlan {
                source: target.clone(),
                virtual_path: PathBuf::from("target/target-0.json"),
                semantic_name: "target/0".to_string(),
            }],
            declared_package_inputs: BTreeSet::new(),
            outputs: vec![output.clone()],
            identity: 0x5eed,
        };
        let surface = vec![
            "build".to_string(),
            "--profile=release".to_string(),
            format!("--config={}", config.display()),
            format!("--target={}", target.display()),
            format!("--target-dir={}", output.display()),
            format!(
                "--config=patch.crates-io.fixture.path={}",
                serde_json::to_string(&workspace.join("fixture").display().to_string()).unwrap(),
            ),
        ];
        (layout, surface)
    }

    #[test]
    fn artifact_identity_is_checkout_independent_and_tracks_source_surface_and_tools() {
        let (first_layout, first_surface) =
            identity_normalization_fixture(Path::new("/runner-a/checkout"));
        let (second_layout, second_surface) =
            identity_normalization_fixture(Path::new("/runner-b/checkout"));
        let normalize = |layout: &SourceLayout, surface: &[String]| {
            surface
                .iter()
                .map(|argument| normalized_argument(argument, layout))
                .collect::<Vec<_>>()
        };
        let first_args = normalize(&first_layout, &first_surface);
        let second_args = normalize(&second_layout, &second_surface);
        assert_eq!(
            first_args, second_args,
            "equivalent source/config/target/output paths must have one semantic spelling",
        );

        let packages = BTreeSet::from(["libc@0.2.0:registry+fixture".to_string()]);
        let environment = vec![(std::ffi::OsString::from("RUSTFLAGS"), b"-Cfoo".to_vec())];
        let tools = vec![
            ("cargo".to_string(), b"cargo-a".to_vec()),
            ("rustc".to_string(), b"rustc-a".to_vec()),
        ];
        let identity = artifact_identity(
            "scheduler-workspace",
            first_layout.identity,
            &first_args,
            &packages,
            &environment,
            &tools,
        );
        assert_eq!(
            identity,
            artifact_identity(
                "scheduler-workspace",
                second_layout.identity,
                &second_args,
                &packages,
                &environment,
                &tools,
            ),
        );
        assert_ne!(
            identity,
            artifact_identity(
                "scheduler-workspace",
                first_layout.identity + 1,
                &first_args,
                &packages,
                &environment,
                &tools,
            ),
            "source bytes must invalidate",
        );
        let mut changed_surface = first_args.clone();
        changed_surface.push(b"--features=changed".to_vec());
        assert_ne!(
            identity,
            artifact_identity(
                "scheduler-workspace",
                first_layout.identity,
                &changed_surface,
                &packages,
                &environment,
                &tools,
            ),
            "the exact Cargo build surface must invalidate",
        );
        let mut changed_tools = tools.clone();
        changed_tools[1].1 = b"rustc-b".to_vec();
        assert_ne!(
            identity,
            artifact_identity(
                "scheduler-workspace",
                first_layout.identity,
                &first_args,
                &packages,
                &environment,
                &changed_tools,
            ),
            "tool content/version changes must invalidate",
        );
    }

    #[test]
    fn explicit_config_and_custom_target_are_detected_captured_and_remapped() {
        let temp = tempfile::tempdir().unwrap();
        let invocation = temp.path().join("invocation");
        std::fs::create_dir(&invocation).unwrap();
        let config = temp.path().join("external-config.toml");
        let target = temp.path().join("custom-target.json");
        std::fs::write(&config, b"[build]\nrustflags = ['--cfg=from_config']\n").unwrap();
        std::fs::write(&target, b"{\"llvm-target\":\"x86_64-unknown-linux-gnu\"}\n").unwrap();

        let surface = vec![
            "nextest:--config".to_string(),
            format!("nextest:{}", config.display()),
            "nextest:--config=net.retry=2".to_string(),
            format!("llvm-cov-env:--target={}", target.display()),
            "nextest:--target".to_string(),
            "nextest:x86_64-unknown-linux-gnu".to_string(),
        ];
        assert_eq!(
            explicit_cargo_config_files(&surface, &invocation),
            vec![(1, config.clone())],
        );
        assert_eq!(
            explicit_cargo_target_specs(&surface, &invocation),
            vec![(3, target.clone())],
        );

        let cache = temp.path().join("cache");
        let mut source = ktstr::cache::artifact_tree::ArtifactTreeSource::new();
        source.insert_directory("workspace", 0o555).unwrap();
        source
            .insert_bytes(
                "workspace/config.toml",
                &std::fs::read(&config).unwrap(),
                0o444,
            )
            .unwrap();
        source
            .insert_bytes(
                "workspace/target.json",
                &std::fs::read(&target).unwrap(),
                0o444,
            )
            .unwrap();
        let tree = ktstr::cache::artifact_tree::ArtifactTreeCache::new(cache.join("records"))
            .load_or_build_stable(
                0xcafe,
                &cache.join("stable"),
                "surface fixture",
                || Ok(true),
                || false,
                || Ok(source),
            )
            .unwrap();
        let stable_workspace = tree.root().join("workspace");
        let stable_config = stable_workspace.join("config.toml");
        let stable_target = stable_workspace.join("target.json");
        let stable = StableCargoSource {
            _tree: tree,
            workspace_root: stable_workspace.clone(),
            invocation_root: stable_workspace.clone(),
            original_workspace_root: invocation.clone(),
            original_invocation_root: invocation.clone(),
            cargo_argument_remaps: BTreeMap::from([
                (config.clone(), stable_config.clone()),
                (target.clone(), stable_target.clone()),
            ]),
            cargo_root_remaps: vec![(invocation.clone(), stable_workspace.clone())],
        };
        assert_eq!(
            stable.remap_cargo_args(&[
                "--config".to_string(),
                config.display().to_string(),
                "--config=net.retry=2".to_string(),
                format!("--target={}", target.display()),
                "--target".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                format!(
                    "--config=patch.crates-io.fixture.path={}",
                    serde_json::to_string(&invocation.join("fixture").display().to_string())
                        .unwrap(),
                ),
            ]),
            vec![
                "--config".to_string(),
                stable_config.display().to_string(),
                "--config=net.retry=2".to_string(),
                format!("--target={}", stable_target.display()),
                "--target".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                format!(
                    "--config=patch.crates-io.fixture.path={}",
                    serde_json::to_string(&stable_workspace.join("fixture").display().to_string())
                        .unwrap(),
                ),
            ],
        );

        let before = read_cargo_config_identity(&config, &[]).unwrap();
        std::fs::write(&config, b"[build]\nrustflags = ['--cfg=changed']\n").unwrap();
        assert_ne!(before, read_cargo_config_identity(&config, &[]).unwrap());
    }

    #[test]
    fn explicit_config_include_closure_keeps_relative_topology() {
        let temp = tempfile::tempdir().unwrap();
        let workspace = temp.path().join("checkout/workspace");
        let config_dir = temp.path().join("checkout/configs");
        let included_dir = config_dir.join("parts");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&included_dir).unwrap();
        let root = config_dir.join("root.toml");
        let included = included_dir.join("shared.toml");
        std::fs::write(
            &root,
            b"include = ['parts/shared.toml', { path = 'missing.toml', optional = true }]\n\
[target.x86_64-unknown-linux-gnu]\n\
rustflags = ['-C', 'target-cpu=x86-64-v3']\n",
        )
        .unwrap();
        std::fs::write(&included, b"[build]\ntarget-dir = '../build-output'\n").unwrap();

        let mut configs = vec![PendingCargoConfig {
            source: root.clone(),
            semantic_name: "explicit/0".to_string(),
            materialize: true,
            explicit_argument: true,
        }];
        expand_cargo_config_closure(&mut configs).unwrap();
        assert_eq!(configs.len(), 2);
        assert_eq!(configs[1].source, included);

        let workspace_virtual = PathBuf::from("source/anchor/primary/workspace");
        let root_relation = path_relation(&workspace, &root).unwrap();
        let included_relation = path_relation(&workspace, &configs[1].source).unwrap();
        let root_virtual =
            virtual_path_for_relation(&workspace_virtual, root_relation.0, &root_relation.1)
                .unwrap();
        let included_virtual = virtual_path_for_relation(
            &workspace_virtual,
            included_relation.0,
            &included_relation.1,
        )
        .unwrap();
        assert_eq!(
            root_virtual.parent().unwrap().join("parts/shared.toml"),
            included_virtual,
        );
    }

    #[test]
    fn source_selection_never_captures_envrc_outputs_cache_or_git_admin() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("source");
        let target = root.join("target");
        let cache = root.join("cache");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(&target).unwrap();
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::write(root.join("src/lib.rs"), b"pub fn fixture() {}\n").unwrap();
        std::fs::write(root.join(".envrc"), b"export GH_TOKEN=never-cache-this\n").unwrap();
        std::fs::write(root.join(".env"), b"TOKEN=never-cache-this\n").unwrap();
        std::fs::write(root.join(".env.local"), b"TOKEN=never-cache-this\n").unwrap();
        std::fs::write(root.join(".git/config"), b"[core]\n").unwrap();
        std::fs::write(target.join("noise"), b"target\n").unwrap();
        std::fs::write(cache.join("noise"), b"cache\n").unwrap();

        let paths = non_git_source_input_paths(&root, &[target, cache]).unwrap();
        assert_eq!(paths, BTreeSet::from([PathBuf::from("src/lib.rs")]));
    }

    #[test]
    fn ignored_nested_workspace_keeps_its_package_scoped_features_in_stable_source() {
        let _environment = lock_env();
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        let _cache = EnvVarGuard::set(ktstr::KTSTR_CACHE_DIR_ENV, &cache);
        let repository = temp.path().join("repository");
        std::fs::create_dir_all(repository.join("outer/src")).unwrap();
        git_output(&repository, &["init", "-q"]);
        std::fs::write(
            repository.join("Cargo.toml"),
            b"[workspace]\nmembers = ['outer']\nresolver = '3'\n",
        )
        .unwrap();
        std::fs::write(
            repository.join("outer/Cargo.toml"),
            b"[package]\nname = 'outer'\nversion = '0.1.0'\nedition = '2024'\n",
        )
        .unwrap();
        std::fs::write(repository.join("outer/src/lib.rs"), b"pub fn outer() {}\n").unwrap();
        std::fs::write(repository.join(".gitignore"), b"/generated/\n").unwrap();
        git_output(&repository, &["add", "."]);

        let workspace = repository.join("generated/workspace");
        for (package, feature) in [("alpha", "scheduler-tests"), ("beta", "verifier-fixtures")] {
            let root = workspace.join(package);
            std::fs::create_dir_all(root.join("src")).unwrap();
            std::fs::write(
                root.join("Cargo.toml"),
                format!(
                    "[package]\nname = '{package}'\nversion = '0.1.0'\n\
                     edition = '2024'\n\n[features]\n{feature} = []\n"
                ),
            )
            .unwrap();
            std::fs::write(root.join("src/lib.rs"), b"pub fn fixture() {}\n").unwrap();
        }
        std::fs::write(
            workspace.join("Cargo.toml"),
            b"[workspace]\nmembers = ['alpha', 'beta']\nresolver = '3'\n",
        )
        .unwrap();

        let metadata = cargo_metadata::MetadataCommand::new()
            .manifest_path(workspace.join("Cargo.toml"))
            .exec()
            .unwrap();
        assert!(workspace.join("Cargo.lock").is_file());
        let layout = plan_source_layout(
            &metadata,
            &[metadata.target_directory.as_std_path().to_path_buf()],
            &[
                "--features".to_string(),
                "alpha/scheduler-tests,beta/verifier-fixtures".to_string(),
            ],
            &workspace,
        )
        .unwrap();
        let overlay = layout
            .roots
            .iter()
            .find(|root| root.source == workspace)
            .expect("ignored workspace has a local stable-source overlay");
        assert!(!overlay.is_git);
        for input in [
            "Cargo.toml",
            "Cargo.lock",
            "alpha/Cargo.toml",
            "alpha/src/lib.rs",
            "beta/Cargo.toml",
            "beta/src/lib.rs",
        ] {
            assert!(
                overlay.input_paths.contains(Path::new(input)),
                "stable-source overlay is missing {input}",
            );
        }

        let stable = materialize_stable_source(&layout, "ignored workspace fixture").unwrap();
        let output = std::process::Command::new("cargo")
            .args([
                "metadata",
                "--format-version=1",
                "--no-deps",
                "--locked",
                "--features",
                "alpha/scheduler-tests,beta/verifier-fixtures",
            ])
            .current_dir(&stable.invocation_root)
            .env("CARGO_TARGET_DIR", temp.path().join("stable-target"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "package-scoped features must resolve in the captured nested workspace: {}",
            String::from_utf8_lossy(&output.stderr),
        );
        let stable_metadata: cargo_metadata::Metadata =
            serde_json::from_slice(&output.stdout).unwrap();
        let packages = stable_metadata
            .packages
            .iter()
            .map(|package| package.name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(packages, BTreeSet::from(["alpha", "beta"]));
    }

    #[test]
    fn stable_source_can_resnapshot_nested_scheduler_workspace_inside_cache() {
        let _environment = lock_env();
        let checkout = std::env::current_dir().unwrap();
        let temp = tempfile::Builder::new()
            .prefix(".ktstr-nested-stable-source-")
            .tempdir_in(checkout.parent().unwrap())
            .unwrap();
        let cache = temp.path().join("cache");
        let _cache = EnvVarGuard::set(ktstr::KTSTR_CACHE_DIR_ENV, &cache);
        let repository = temp.path().join("repository");
        std::fs::create_dir_all(repository.join("scheduler/src")).unwrap();
        let git_init = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&repository)
            .env("GIT_DIR", repository.join(".git"))
            .env("GIT_WORK_TREE", &repository)
            .output()
            .unwrap();
        assert!(
            git_init.status.success(),
            "git init failed: {}",
            String::from_utf8_lossy(&git_init.stderr),
        );
        assert!(repository.join(".git").is_dir());
        std::fs::write(
            repository.join("Cargo.toml"),
            b"[workspace]\nmembers = ['scheduler']\nresolver = '3'\n",
        )
        .unwrap();
        std::fs::write(
            repository.join("scheduler/Cargo.toml"),
            b"[package]\nname = 'scheduler'\nversion = '0.1.0'\nedition = '2024'\n",
        )
        .unwrap();
        std::fs::write(
            repository.join("scheduler/src/lib.rs"),
            b"pub fn scheduler() {}\n",
        )
        .unwrap();
        let metadata = cargo_metadata::MetadataCommand::new()
            .manifest_path(repository.join("Cargo.toml"))
            .exec()
            .unwrap();
        assert!(repository.join("Cargo.lock").is_file());
        git_output(&repository, &["add", "."]);
        let first = plan_source_layout(
            &metadata,
            &[metadata.target_directory.as_std_path().to_path_buf()],
            &[],
            &repository,
        )
        .unwrap();
        let first = materialize_stable_source(&first, "outer stable-source fixture").unwrap();
        let stable_scheduler = first.workspace_root.join("scheduler");
        assert!(stable_scheduler.join("Cargo.toml").is_file());
        assert!(first.workspace_root.join("Cargo.lock").is_file());

        let scheduler_metadata = cargo_metadata::MetadataCommand::new()
            .manifest_path(stable_scheduler.join("Cargo.toml"))
            .no_deps()
            .exec()
            .unwrap();
        let second = plan_source_layout(
            &scheduler_metadata,
            &[scheduler_metadata
                .target_directory
                .as_std_path()
                .to_path_buf()],
            &[],
            scheduler_metadata.workspace_root.as_std_path(),
        )
        .unwrap();
        let nested_root = second
            .roots
            .iter()
            .find(|root| root.source.starts_with(&cache))
            .unwrap_or_else(|| {
                panic!(
                    "nested source plan escaped its cache tree: {:?}",
                    second
                        .roots
                        .iter()
                        .map(|root| &root.source)
                        .collect::<Vec<_>>()
                )
            });
        assert!(nested_root.input_paths.contains(Path::new("Cargo.toml")));
        assert!(
            nested_root
                .input_paths
                .contains(Path::new("scheduler/Cargo.toml"))
        );

        let second = materialize_stable_source(&second, "nested stable-source fixture").unwrap();
        assert!(second.workspace_root.join("Cargo.toml").is_file());
        assert!(second.workspace_root.join("scheduler/Cargo.toml").is_file());
        cargo_metadata::MetadataCommand::new()
            .manifest_path(second.workspace_root.join("scheduler/Cargo.toml"))
            .no_deps()
            .exec()
            .unwrap();
    }

    #[test]
    fn resolved_ignored_library_lockfile_is_captured_for_read_only_stable_reuse() {
        let _environment = lock_env();
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        let _cache = EnvVarGuard::set(ktstr::KTSTR_CACHE_DIR_ENV, &cache);
        let workspace = temp.path().join("library-workspace");
        std::fs::create_dir_all(workspace.join("src")).unwrap();
        git_output(&workspace, &["init", "-q"]);
        std::fs::write(
            workspace.join("Cargo.toml"),
            b"[package]\nname = \"lock-fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        std::fs::write(workspace.join("src/lib.rs"), b"pub fn fixture() {}\n").unwrap();
        std::fs::write(workspace.join(".gitignore"), b"/Cargo.lock\n").unwrap();
        git_output(&workspace, &["add", "."]);

        let mut declared_inputs = BTreeSet::new();
        include_resolved_workspace_lockfile(&workspace, &mut declared_inputs);
        assert!(
            declared_inputs.is_empty(),
            "identity planning must not synthesize a missing lockfile"
        );

        let lockfile = workspace.join("Cargo.lock");
        let lockfile_bytes =
            b"# exact lockfile emitted by preparatory Cargo metadata\nversion = 4\n";
        std::fs::write(&lockfile, lockfile_bytes).unwrap();
        let inputs = git_source_input_paths(&workspace).unwrap();
        assert!(
            !inputs.paths.contains(Path::new("Cargo.lock")),
            "the fixture must prove that Git status excludes ignored Cargo.lock"
        );
        include_resolved_workspace_lockfile(&workspace, &mut declared_inputs);
        assert_eq!(
            declared_inputs,
            BTreeSet::from([canonical_or_lexical(&lockfile)])
        );

        let mut input_paths = inputs.paths;
        for path in &declared_inputs {
            input_paths.insert(path.strip_prefix(&workspace).unwrap().to_path_buf());
        }
        let digest = source_tree_digest(&workspace, &input_paths, &inputs.gitlinks).unwrap();
        let layout = SourceLayout {
            workspace: workspace.clone(),
            invocation: workspace.clone(),
            invocation_relation: (0, Vec::new()),
            workspace_virtual: PathBuf::from("workspace"),
            invocation_virtual: PathBuf::from("workspace"),
            roots: vec![SourceRootPlan {
                source: workspace.clone(),
                virtual_path: PathBuf::from("workspace"),
                semantic_name: "workspace".to_string(),
                is_git: true,
                input_paths,
                git_head: git_head(&workspace),
                git_status_semantics: inputs.status_semantics,
                git_metadata: inputs.metadata,
                digest,
            }],
            cargo_configs: Vec::new(),
            target_specs: Vec::new(),
            declared_package_inputs: declared_inputs,
            outputs: vec![cache.clone()],
            identity: 0x10c0,
        };
        let source = capture_source_layout(&layout).unwrap();
        let tree_cache =
            ktstr::cache::artifact_tree::ArtifactTreeCache::new(cache.join("lock-records"));
        let stable_parent = cache.join("stable-sources");
        let stable = tree_cache
            .load_or_build_stable(
                layout.identity,
                &stable_parent,
                "ignored lockfile fixture",
                || Ok(true),
                || false,
                || Ok(source),
            )
            .unwrap();
        assert!(!stable.cache_hit());
        let stable_workspace = stable.root().join("workspace");
        let stable_lockfile = stable_workspace.join("Cargo.lock");
        assert_eq!(std::fs::read(&stable_lockfile).unwrap(), lockfile_bytes);
        assert_eq!(
            std::os::unix::fs::PermissionsExt::mode(
                &std::fs::metadata(&stable_lockfile).unwrap().permissions()
            ) & 0o222,
            0,
        );
        drop(stable);

        let reused = tree_cache
            .load_or_build_stable(
                layout.identity,
                &stable_parent,
                "ignored lockfile fixture",
                || Ok(true),
                || false,
                || panic!("complete stable source must be reused"),
            )
            .unwrap();
        assert!(reused.cache_hit());
        assert_eq!(
            std::fs::read(reused.root().join("workspace/Cargo.lock")).unwrap(),
            lockfile_bytes
        );
    }

    #[test]
    fn package_include_matching_is_ordered_and_bounded_by_static_prefixes() {
        let manifest = Path::new("/fixture/Cargo.toml");
        let patterns = ["src/**", "!src/private/**", "assets/*.bin"]
            .into_iter()
            .map(|value| compile_package_include_pattern(value, manifest).unwrap())
            .collect::<Vec<_>>();
        assert!(package_include_matches(
            &patterns,
            Path::new("src/public.rs")
        ));
        assert!(!package_include_matches(
            &patterns,
            Path::new("src/private/secret.rs")
        ));
        assert!(package_include_matches(
            &patterns,
            Path::new("assets/image.bin")
        ));
        assert_eq!(
            package_include_scan_prefix("src/**/*.rs"),
            Some(PathBuf::from("src"))
        );
        assert_eq!(package_include_scan_prefix("foo"), None);
        assert_eq!(package_include_scan_prefix("**/*.rs"), None);
    }

    #[test]
    fn package_include_finds_ignored_unrooted_matches_without_capturing_outputs() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("package");
        std::fs::create_dir_all(root.join("bar")).unwrap();
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::create_dir_all(root.join("nested-package")).unwrap();
        std::fs::write(root.join(".gitignore"), b"bar/\nnested/\n").unwrap();
        std::fs::write(root.join("bar/foo"), b"ignored literal\n").unwrap();
        std::fs::write(root.join("nested/a.rs"), b"ignored wildcard\n").unwrap();
        std::fs::write(root.join("target/leak.rs"), b"output\n").unwrap();
        std::fs::write(root.join("nested-package/Cargo.toml"), b"[package]\n").unwrap();
        std::fs::write(root.join("nested-package/leak.rs"), b"subpackage\n").unwrap();

        let values = vec!["foo".to_string(), "*.rs".to_string()];
        let paths =
            scan_declared_package_include_paths(&root, &values, &[], &root.join("Cargo.toml"))
                .unwrap();
        assert_eq!(
            paths,
            BTreeSet::from([root.join("bar/foo"), root.join("nested/a.rs")])
        );
    }

    #[test]
    fn git_metadata_identity_distinguishes_shallow_and_deep_object_sets() {
        let shallow = BTreeMap::from([(
            PathBuf::from("objects/aa/11111111111111111111111111111111111111"),
            PathBuf::from("/object-content-is-addressed-by-name"),
        )]);
        let mut deep = shallow.clone();
        deep.insert(
            PathBuf::from("objects/bb/22222222222222222222222222222222222222"),
            PathBuf::from("/object-content-is-addressed-by-name"),
        );
        assert_ne!(
            git_metadata_semantic_digest(&shallow, b"", 0).unwrap(),
            git_metadata_semantic_digest(&deep, b"", 0).unwrap(),
        );
    }

    #[test]
    fn stable_source_owns_git_head_status_and_objects_after_origin_disappears() {
        let _environment = lock_env();
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        let _cache = EnvVarGuard::set(ktstr::KTSTR_CACHE_DIR_ENV, &cache);
        let repository = temp.path().join("repository");
        std::fs::create_dir(&repository).unwrap();
        git_output(&repository, &["init", "-q"]);
        std::fs::write(repository.join("tracked"), b"committed\n").unwrap();
        git_output(&repository, &["add", "tracked"]);
        git_output(
            &repository,
            &[
                "-c",
                "user.name=ktstr",
                "-c",
                "user.email=ktstr@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
        );
        git_output(
            &repository,
            &[
                "config",
                "http.https://github.com/.extraheader",
                "SECRET-AUTH-HEADER",
            ],
        );
        std::fs::write(repository.join("tracked"), b"dirty\n").unwrap();
        std::fs::write(repository.join("untracked"), b"untracked\n").unwrap();

        let inputs = git_source_input_paths(&repository).unwrap();
        let digest = source_tree_digest(&repository, &inputs.paths, &inputs.gitlinks).unwrap();
        let expected_head = git_output(&repository, &["rev-parse", "--short", "HEAD"]);
        let expected_status = git_output(
            &repository,
            &["status", "--porcelain=v1", "--untracked-files=all"],
        );
        let layout = SourceLayout {
            workspace: repository.clone(),
            invocation: repository.clone(),
            invocation_relation: (0, Vec::new()),
            workspace_virtual: PathBuf::from("repository"),
            invocation_virtual: PathBuf::from("repository"),
            roots: vec![SourceRootPlan {
                source: repository.clone(),
                virtual_path: PathBuf::from("repository"),
                semantic_name: "repository".to_string(),
                is_git: true,
                input_paths: inputs.paths,
                git_head: git_head(&repository),
                git_status_semantics: inputs.status_semantics,
                git_metadata: inputs.metadata,
                digest,
            }],
            cargo_configs: Vec::new(),
            target_specs: Vec::new(),
            declared_package_inputs: BTreeSet::new(),
            outputs: vec![cache.clone()],
            identity: 0xfeed,
        };
        let source = capture_source_layout(&layout).unwrap();
        let tree_cache =
            ktstr::cache::artifact_tree::ArtifactTreeCache::new(cache.join("git-view-records"));
        let stable = tree_cache
            .load_or_build_stable(
                0xfeed,
                &cache.join("git-views"),
                "git fixture",
                || Ok(true),
                || false,
                || Ok(source),
            )
            .unwrap();
        let stable_repository = stable.root().join("repository");
        let stable = StableCargoSource {
            _tree: stable,
            workspace_root: stable_repository.clone(),
            invocation_root: stable_repository.clone(),
            original_workspace_root: repository.clone(),
            original_invocation_root: repository.clone(),
            cargo_argument_remaps: BTreeMap::new(),
            cargo_root_remaps: vec![(repository.clone(), stable_repository.clone())],
        };
        std::fs::remove_dir_all(&repository).unwrap();
        assert_eq!(
            expected_head,
            git_output(&stable_repository, &["rev-parse", "--short", "HEAD"]),
        );
        assert_eq!(
            expected_status,
            git_output(
                &stable_repository,
                &["status", "--porcelain=v1", "--untracked-files=all"],
            ),
        );
        assert!(stable_repository.join(".git/objects").is_dir());
        for entry in walkdir::WalkDir::new(stable_repository.join(".git")) {
            let entry = entry.unwrap();
            if entry.file_type().is_file() {
                let bytes = std::fs::read(entry.path()).unwrap();
                assert!(
                    !bytes
                        .windows(b"SECRET-AUTH-HEADER".len())
                        .any(|window| window == b"SECRET-AUTH-HEADER"),
                    "captured Git metadata leaked repository credentials through {}",
                    entry.path().display(),
                );
            }
        }
        drop(stable);
    }
}
