//! Kernel image cache for ktstr.
//!
//! Manages a local cache of built kernel images under an XDG-compliant
//! directory. Each cached kernel is a directory containing the boot
//! image, optionally a stripped vmlinux ELF (symbol table, BTF, and
//! the section headers that monitor/probe code reads), and a
//! `metadata.json` descriptor. `CONFIG_HZ` is recovered from the
//! embedded IKCONFIG blob in the stripped vmlinux (ktstr.kconfig
//! forces `CONFIG_IKCONFIG=y`), so no separate `.config` sidecar is
//! cached.
//!
//! # Cache location
//!
//! Resolved in order:
//! 1. `KTSTR_CACHE_DIR` environment variable
//! 2. `$XDG_CACHE_HOME/ktstr/kernels/`
//! 3. `$HOME/.cache/ktstr/kernels/`
//!
//! # Submodule layout
//!
//! - `metadata` — public types: [`KernelSource`], [`KernelMetadata`],
//!   [`CacheArtifacts`], [`KconfigStatus`], [`CacheEntry`],
//!   [`ListedEntry`], plus the internal `classify_corrupt_reason`
//!   dispatcher.
//! - `cache_dir` — [`CacheDir`] handle, lock guards
//!   ([`SharedLockGuard`], [`ExclusiveLockGuard`]), store/lookup/list/
//!   clean lifecycle, and reader/writer-asymmetric lock policy.
//! - `housekeeping` — atomic-rename install primitives, cache-key
//!   and image-name validators, `read_metadata` decoder, and the
//!   `clean_orphaned_tmp_dirs` cross-PID sweep.
//! - `vmlinux_strip` — ELF strip pipeline (`strip_vmlinux_debug`,
//!   `neutralize_relocs`, `strip_keep_list`, `strip_debug_prefix`)
//!   plus the keep-list / zero-data section-name unions.
//! - `resolve` — env-cascade root resolution
//!   (`resolve_cache_root_with_suffix`, `validate_home_for_cache`,
//!   `path_inside_cache_root`) and source-tree path helpers
//!   ([`prefer_source_tree_for_dwarf`], [`recover_local_source_tree`]).
//!
//! Each submodule owns its tests in a `#[cfg(test)] mod tests`
//! block — inline in the same file except `cache_dir`, whose tests
//! live in `cache_dir_tests.rs` via `#[path]`; shared test fixtures
//! used by
//! more than one submodule's tests live in
//! `shared_test_helpers`.

use crate::flock::LOCK_DIR_NAME;

#[doc(hidden)]
pub mod artifact_tree;
mod cache_dir;
pub(crate) mod content;
mod housekeeping;
mod metadata;
mod resolve;
mod vmlinux_strip;

// Public API re-exports — preserve every `crate::cache::*` path that
// external callers (lib.rs, cli, fetch.rs, monitor/*, probe/btf.rs,
// vmm/disk_template, test_support/*, remote_cache.rs, stats,
// flock) rely on.

pub(crate) use cache_dir::GitBuilderLockGuard;
pub use cache_dir::{CacheDir, ExclusiveLockGuard, SharedLockGuard};
pub(crate) use metadata::kernel_config_include_for_image;
pub use metadata::{
    CacheArtifacts, CacheEntry, KconfigStatus, KernelMetadata, KernelSource, ListedEntry,
    boot_modules_for_image, initrd_compression_for_image, ordered_boot_modules_in,
};
pub use resolve::{prefer_source_tree_for_dwarf, recover_local_source_tree};

// Re-export KernelId from kernel_path (canonical definition, std-only).
pub use crate::kernel_path::KernelId;

// Crate-internal API re-exports for callers in other modules:
// path_inside_cache_root (monitor/btf_offsets),
// resolve_cache_root_with_suffix (vmm/disk_template,
// vmm/cast_analysis_load), resolve_lock_dir
// (vmm/host_topology, cli/locks).
pub(crate) use resolve::{
    path_inside_cache_root, resolve_cache_root_with_suffix, resolve_lock_dir,
};

/// Cache root for the `cargo ktstr affected` per-scheduler input-set cache.
///
/// Exposed as `pub` (unlike the crate-internal
/// `resolve_cache_root_with_suffix`) because the affected engine lives in
/// the cargo-ktstr BIN crate, a separate crate that reaches it as
/// `ktstr::cache::affected_cache_root()`. Runs the same `KTSTR_CACHE_DIR` ->
/// `$XDG_CACHE_HOME` -> `$HOME/.cache` cascade as every other ktstr cache,
/// under the `affected` suffix.
pub fn affected_cache_root() -> anyhow::Result<std::path::PathBuf> {
    resolve_cache_root_with_suffix("affected")
}

/// Host-global coordination namespace for wrapper-owned Cargo output locks.
///
/// Output ownership follows the canonical Cargo target directory, not the
/// caller's artifact-cache selection. Two invocations which deliberately use
/// different `KTSTR_CACHE_DIR`s can still write the same `CARGO_TARGET_DIR`;
/// placing this lock below either cache root would let both writers enter
/// Cargo's artifact lock while holding separate ktstr build reservations.
/// `KTSTR_LOCK_DIR` is the existing machine-wide coordination namespace for
/// ktstr participants, with `/tmp` as its normal fallback.
///
/// The version is intentionally part of the directory name. Every ktstr
/// writer participating in this protocol acquires the same per-target-dir
/// lock before spawning Cargo and retains it until emitted artifacts are
/// pinned. There is no lookup of an older cache-rooted namespace.
pub fn cargo_build_output_lock_root() -> anyhow::Result<std::path::PathBuf> {
    Ok(resolve_lock_dir().join("ktstr-cargo-build-output-v3"))
}

/// Machine-local record namespace for reusable Cargo artifact trees.
///
/// The files referenced by these records live in the ordinary shared content
/// CAS; this root contains only small, reconstructible tree manifests and
/// their cross-process election locks.
#[doc(hidden)]
pub fn cargo_artifact_tree_cache_root() -> anyhow::Result<std::path::PathBuf> {
    resolve_cache_root_with_suffix("cargo-artifact-tree-v1")
}

/// Return ktstr's stable, fixed-seed fast digest for one exact file revision.
///
/// This is the source-input half of the scheduler-build cache identity. It
/// uses the same machine-wide inode-revision memo as content-CAS publication,
/// so repeated source-tree walks do not reread unchanged files in one shared
/// checkout.
#[doc(hidden)]
pub fn content_file_digest(path: impl AsRef<std::path::Path>) -> anyhow::Result<u64> {
    let path = path.as_ref();
    let pinned = pin_content_file(path)?;
    content::cached_file_digest(pinned.source(), pinned.identity)
}

/// One immutable, content-addressed snapshot of an input file.
///
/// The original file descriptor and the shared CAS lease remain live until
/// this owner is dropped. [`Self::path`] is therefore a stable pathname for
/// child processes even if the caller's original pathname is replaced or the
/// cache's background garbage collection runs.
pub struct ContentFileSnapshot {
    _source: std::fs::File,
    lease: content::ContentObjectLease,
    content_hash: u64,
    len: u64,
}

/// One opened regular-file revision retained independently of its pathname.
///
/// Cargo installs final artifacts by atomic replacement. Opening the artifact
/// while the wrapper's output lock is held pins that exact inode; a later
/// pathname replacement cannot retarget this owner.
pub struct PinnedContentFile {
    source: std::fs::File,
    identity: content::StableFileIdentity,
    display_path: std::path::PathBuf,
}

impl PinnedContentFile {
    /// Original pathname used to open this revision, for diagnostics and
    /// provenance. Consumers must use the pinned descriptor for bytes.
    pub fn source_path(&self) -> &std::path::Path {
        &self.display_path
    }

    /// Linux descriptor pathname for an exact-inode child execution probe.
    #[doc(hidden)]
    pub fn proc_fd_path(&self) -> std::path::PathBuf {
        use std::os::fd::AsRawFd as _;

        std::path::PathBuf::from(format!("/proc/self/fd/{}", self.source.as_raw_fd()))
    }

    pub(crate) fn source(&self) -> &std::fs::File {
        &self.source
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        std::fs::File,
        content::StableFileIdentity,
        std::path::PathBuf,
    ) {
        (self.source, self.identity, self.display_path)
    }
}

/// Open and pin one regular-file revision without publishing it yet.
pub fn pin_content_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<PinnedContentFile> {
    let path = path.as_ref();
    let (source, identity) = content::open_pinned_file(path)?;
    Ok(PinnedContentFile {
        source,
        identity,
        display_path: path.to_path_buf(),
    })
}

/// Pin generated artifact metadata in an inode on the content-CAS filesystem.
///
/// The returned descriptor has no durable source pathname. Keeping the open
/// file description alive is sufficient for strict FICLONE publication and
/// avoids making `/tmp` filesystem placement part of artifact-cache behavior.
pub(crate) fn pin_generated_artifact_bytes(
    bytes: &[u8],
    mode: u32,
) -> anyhow::Result<PinnedContentFile> {
    let (source, identity) = content::open_pinned_generated_artifact(bytes, mode)?;
    Ok(PinnedContentFile {
        source,
        identity,
        display_path: std::path::PathBuf::from("<generated artifact metadata>"),
    })
}

/// Publish an already-pinned file revision into the shared content CAS.
///
/// Publication is uniformly FICLONE-only so every snapshot preserves shared
/// backing extents and private/COW behavior across processes.
pub fn snapshot_pinned_content_file(
    pinned: PinnedContentFile,
) -> anyhow::Result<ContentFileSnapshot> {
    let (source, identity, _) = pinned.into_parts();
    let content_hash = content::cached_file_digest(&source, identity)?;
    let lease = content::open_or_publish_content_object_lease(content_hash, &source, identity)?;
    Ok(ContentFileSnapshot {
        _source: source,
        lease,
        content_hash,
        len: identity.size,
    })
}

/// Publish an artifact-tree input with a strict FICLONE requirement.
///
/// Artifact closures deliberately have no byte-copy fallback: both CAS
/// publication and private materialization must preserve cross-process COW.
pub(crate) fn snapshot_pinned_artifact_file(
    pinned: PinnedContentFile,
) -> anyhow::Result<ContentFileSnapshot> {
    snapshot_pinned_content_file(pinned)
}

impl ContentFileSnapshot {
    /// Return the canonical pathname of the leased immutable object.
    pub fn path(&self) -> &std::path::Path {
        self.lease.path()
    }

    /// Stable fast content key used by reconstructible cache manifests.
    #[doc(hidden)]
    pub fn content_hash(&self) -> u64 {
        self.content_hash
    }

    /// Exact byte length validated by the content-object lease.
    #[doc(hidden)]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the immutable object is empty.
    #[doc(hidden)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Pin one regular-file revision and publish it into the shared content CAS.
///
/// Opening uses the same nonblocking regular-file guard as scheduler
/// snapshots, so FIFOs and device nodes are rejected without waiting. The
/// digest, publication, and returned lease all validate the single pinned
/// revision.
pub fn snapshot_content_file(
    path: impl AsRef<std::path::Path>,
) -> anyhow::Result<ContentFileSnapshot> {
    snapshot_pinned_content_file(pin_content_file(path)?)
}
// Durable-publish primitives shared by both cache layers
// (cache_dir::CacheDir::store and vmm/disk_template::store_atomic) so
// the two stay consistent — see fsync_staging_dir's contract.
pub(crate) use housekeeping::{fsync_parent, fsync_staging_dir};
// Re-exported so the `[`crate::cache::strip_vmlinux_debug`]`
// intra-doc links in probe/btf.rs, monitor/mod.rs, and
// monitor/symbols.rs resolve under cargo doc. No
// `crate::cache::strip_vmlinux_debug` code call sites today;
// intra-cache callers (cache_dir.rs, tests) reach the function
// via `super::vmlinux_strip::strip_vmlinux_debug`.
#[allow(unused_imports)]
pub(crate) use vmlinux_strip::strip_vmlinux_debug;

/// Filename prefix that marks an in-progress atomic-store directory
/// under the cache root. Format: `{TMP_DIR_PREFIX}{cache_key}-{pid}`.
/// Centralized here so the three roles that reference it — emitter
/// ([`cache_dir::CacheDir::store`]), scanner
/// ([`housekeeping::clean_orphaned_tmp_dirs`]), and validator
/// ([`housekeeping::validate_cache_key`]) — cannot drift.
/// [`cache_dir::CacheDir::list`] does not reference the const; it
/// skips these directories via its broader leading-`.` filter.
pub(crate) const TMP_DIR_PREFIX: &str = ".tmp-";

#[cfg(test)]
pub(crate) mod shared_test_helpers;

#[cfg(test)]
mod coordination_tests {
    use super::*;
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

    #[test]
    fn cargo_output_lock_root_is_independent_of_artifact_cache_root() {
        let _env = lock_env();
        let lock_dir = tempfile::tempdir().expect("lock directory");
        let first_cache = tempfile::tempdir().expect("first cache directory");
        let second_cache = tempfile::tempdir().expect("second cache directory");
        let _lock_dir = EnvVarGuard::set(crate::KTSTR_LOCK_DIR_ENV, lock_dir.path());

        let first = {
            let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, first_cache.path());
            cargo_build_output_lock_root().expect("first output lock root")
        };
        let second = {
            let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, second_cache.path());
            cargo_build_output_lock_root().expect("second output lock root")
        };

        assert_eq!(first, second);
        assert_eq!(first, lock_dir.path().join("ktstr-cargo-build-output-v3"));
    }
}
