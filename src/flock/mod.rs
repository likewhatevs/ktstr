//! Advisory flock(2) primitives shared across every ktstr lock file.
//!
//! ktstr uses advisory `flock(2)` in four places:
//!
//!  - LLC reservation locks at `{lock_dir}/ktstr-llc-{N}.lock` and
//!    per-CPU locks at `{lock_dir}/ktstr-cpu-{C}.lock` where
//!    `lock_dir` is resolved by `crate::cache::resolve_lock_dir`
//!    (`KTSTR_LOCK_DIR` env var, fallback `/tmp`). See
//!    `crate::vmm::host_topology::acquire_resource_locks` and
//!    friends.
//!  - Per-cache-entry coordination locks at
//!    `{cache_root}/.locks/{cache_key}.lock` (see
//!    `crate::cache::CacheDir::acquire_shared_lock` and friends).
//!  - Per-source-tree build locks at
//!    `{cache_root}/.locks/source-{path_hash}.lock` (see
//!    `crate::cli::kernel_build::build::acquire_source_tree_lock`) — serialize concurrent
//!    `make` invocations against the same kernel source checkout.
//!  - Observational enumeration from `ktstr locks --json` — a
//!    read-only scan that does NOT acquire flocks; reads
//!    /proc/locks through `read_holders` to attribute holders
//!    without contending with active acquirers.
//!
//! All four share:
//!  - Non-blocking `LOCK_NB` attempt (the cache-entry path wraps this
//!    in a poll loop for timed-wait semantics).
//!  - `O_CLOEXEC` on every open so the kernel's "release flock when
//!    the last fd referring to the OFD closes" invariant matches what
//!    `OwnedFd::drop` does — a leaked fd across `exec(2)` would keep
//!    the lock alive in the child and fool the next acquirer's
//!    `/proc/locks` scan into naming the wrong pid.
//!  - /proc/locks parsing keyed on the mount-point-derived
//!    `{major:02x}:{minor:02x}:{inode}` triple, resolved via
//!    `/proc/self/mountinfo` (not `stat().st_dev` — see below).
//!  - [`HolderInfo`] with `pid` + truncated `/proc/{pid}/cmdline` for
//!    actionable error messages.
//!
//! # Module layout
//!
//! Each submodule owns a single, cohesive subsystem:
//!
//!  - `fs_filter` — refuses to operate on filesystems where
//!    `flock(2)` is unreliable (NFS, CIFS/SMB, CephFS, AFS, FUSE).
//!  - `primitives` — the kernel-syscall wrappers
//!    ([`try_flock`] / [`block_flock`] / `materialize`) that open a
//!    lockfile and request a flock operation.
//!  - `mountinfo` — `/proc/self/mountinfo` parser and the
//!    `{major:02x}:{minor:02x}:{inode}` needle derivation that
//!    `proc_locks` keys off.
//!  - `proc_locks` — `/proc/locks` scanner that enumerates the
//!    PIDs holding a given lockfile's flock.
//!  - `holder` — converts a PID into a
//!    [`HolderInfo`] (reads `/proc/{pid}/cmdline`) and renders a
//!    `&[HolderInfo]` into a multi-line operator-facing string.
//!  - `acquire` — high-level poll-with-timeout helper that wraps
//!    `primitives::try_flock` in a deadline loop and decorates
//!    timeout errors with the holder list from `proc_locks` and
//!    `holder`.
//!
//! # Why mountinfo, not `stat().st_dev`
//!
//! `/proc/locks` emits `i_sb->s_dev` for each held flock — the
//! filesystem's superblock device id. For most filesystems that
//! matches `stat().st_dev`, but on btrfs, overlayfs, and bind-mounts
//! the kernel installs a custom `getattr` implementation that returns
//! an anonymous device id (`anon_dev`) distinct from `s_dev`. That
//! divergence means the stat-derived needle would never match the
//! /proc/locks line — a naive `read_holders` would silently return
//! empty on every btrfs-backed `/tmp`, every overlay-rootfs
//! container, and every bind-mounted /tmp, which is a silent
//! correctness failure for `--cpu-cap` contention diagnostics and
//! the `ktstr locks` observational command.
//!
//! Needle production (see `mountinfo::needle_from_path`):
//!
//! `mountinfo::needle_from_path` resolves `path` to the mount-point
//! covering it via `/proc/self/mountinfo` (longest-prefix match on
//! the `mount_point` field), then reads the `{major:minor}` field of
//! that mount entry. Combines with `stat().st_ino` for the full
//! triple. The mountinfo `{major:minor}` is the kernel's
//! `i_sb->s_dev` verbatim, so the resulting needle matches
//! /proc/locks by construction. The needle feeds
//! `proc_locks::read_holders_for_needle`, which scans
//! `/proc/locks` exactly once and byte-compares.
//!
//! # Remote-filesystem rejection
//!
//! [`try_flock`] refuses to operate on NFS / CIFS / SMB2 / CEPH /
//! AFS / FUSE (see `fs_filter::reject_remote_fs`). `flock(2)` on
//! those filesystems is either advisory-only under some server
//! configurations (NFSv3 without NLM coordination) or silently
//! returns success without serializing peers (FUSE when the
//! userspace server doesn't implement the flock op). ktstr's
//! resource-budget contract is not robust to that silent
//! degradation, so the safe call is to reject at lockfile-open
//! time with an actionable message.

use serde::Serialize;

pub(crate) mod acquire;
pub(crate) mod fs_filter;
pub(crate) mod holder;
pub(crate) mod mountinfo;
pub(crate) mod primitives;
pub(crate) mod proc_locks;

pub use holder::format_holder_list;
pub use primitives::{block_flock, try_flock};

pub(crate) use acquire::acquire_flock_with_timeout;
pub(crate) use holder::NO_HOLDERS_RECORDED;
pub(crate) use mountinfo::read_mountinfo;
pub(crate) use primitives::materialize;
pub(crate) use proc_locks::{read_holders, read_holders_with_mountinfo};

/// Subdirectory name (under whatever root each caller picks) that
/// holds advisory `flock(2)` sentinels. Both [`crate::cache`] and
/// the run-dir flock surface in `crate::test_support::sidecar`
/// key off this constant for the `.locks/` convention. Also
/// referenced by run-listing walkers' dotfile filter
/// (`is_run_directory` in the same sidecar module) to keep
/// the lock subdirectory out of "list runs" output. `crate::vmm::disk_template`
/// maintains its own local copy of the same value for the
/// cache-side `.locks/` convention; the two are kept in sync by
/// convention rather than via a shared import.
pub(crate) const LOCK_DIR_NAME: &str = ".locks";

/// Requested sharing mode for [`try_flock`]. Translated to the
/// corresponding non-blocking [`rustix::fs::FlockOperation`]
/// internally; callers never see the libc-specific constants.
///
/// Shared between LLC + per-CPU flocks (`vmm::host_topology`) and
/// cache-entry flocks (`cache`). A single type prevents three-enum
/// drift — earlier revisions had `FlockMode` + `FlockKind` +
/// `LlcLockMode` with identical shape. `LlcLockMode` remains distinct
/// as the scheduler-intent layer (perf-mode vs. no-perf-mode
/// request), not a flock operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlockMode {
    /// Exclusive (`LOCK_EX`) — sole access to the lock file.
    Exclusive,
    /// Shared (`LOCK_SH`) — multiple holders can coexist.
    Shared,
}

/// Identity of a process holding an advisory flock. Used by error
/// messages in both LLC-coordination and cache-entry paths, plus the
/// `ktstr locks` observational subcommand.
///
/// Cmdline is read from `/proc/{pid}/cmdline`, NUL-separated by the
/// kernel, lossy-UTF-8 decoded, `\0 → space`, and truncated to
/// roughly 100 chars (the `holder::CMDLINE_MAX_CHARS` cap) with a
/// `…` marker so a log line remains single-line. A missing / racing
/// / permission-denied `/proc/{pid}/cmdline` produces
/// `"<cmdline unavailable>"` so the pid still surfaces with
/// diagnostic value.
///
/// `#[non_exhaustive]` so future fields (`start_time`, `fd_count`,
/// etc.) don't break external match arms or struct literals. Derives
/// `Serialize` (with `snake_case` field renaming for JSON schema
/// stability) for the `ktstr locks --json` surface; no `Deserialize`
/// because this type is produced-only.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct HolderInfo {
    /// PID of the flock holder as reported by `/proc/locks`.
    pub pid: u32,
    /// Truncated `/proc/{pid}/cmdline` of the holder process.
    pub cmdline: String,
}
