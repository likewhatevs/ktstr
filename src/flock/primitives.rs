//! Kernel-syscall wrappers for `flock(2)` acquire/release.
//!
//! Three entry points, each gated through
//! [`super::fs_filter::reject_remote_fs`] so a misconfigured lockfile
//! path on NFS / CIFS / SMB2 / CephFS / AFS / FUSE surfaces actionably
//! at open time rather than silently returning an unserialized fd:
//!
//!  - [`materialize`] — create the lockfile inode without acquiring
//!    a lock. Used by the DISCOVER phase of
//!    `acquire_llc_plan` so the snapshot pass has a target inode
//!    for the subsequent `/proc/locks` match without contending
//!    with live acquirers.
//!  - [`try_flock`] — non-blocking acquire. Returns `Ok(None)` on
//!    `EWOULDBLOCK` so the caller can decide whether to retry, poll,
//!    or surface contention.
//!  - [`block_flock`] — blocking acquire. Parks the calling thread
//!    in the kernel until the lock is available. Used after
//!    [`try_flock`] returns `None` for callers that want to wait
//!    indefinitely; callers with a deadline use
//!    [`super::acquire::acquire_flock_with_timeout`] instead.
//!
//! All three open with `O_CREAT | O_RDWR | O_CLOEXEC | 0o666` so the
//! resulting fd matches the rest of the crate's lockfile contract:
//!
//!  - `O_CLOEXEC` keeps the lock from leaking across `exec(2)` into
//!    spawned subprocesses (cargo subcommands, build pipeline,
//!    initramfs compressor) where the parent's `OwnedFd::drop`
//!    would not release a child-held flock.
//!  - 0o666 mode matches a peer first-acquire so the file's owner
//!    and permissions don't depend on creation order.

use anyhow::Result;
use std::os::fd::OwnedFd;
use std::path::Path;

use super::FlockMode;
use super::fs_filter::reject_remote_fs;

/// Open a lockfile with the crate-wide flock contract: refuses
/// remote filesystems via [`reject_remote_fs`], then opens with
/// `O_CREAT | O_RDWR | O_CLOEXEC | 0o666`. The three module entry
/// points ([`materialize`], [`try_flock`], [`block_flock`]) share
/// this open shape; centralizing it here means a future flag change
/// (or an addition to the remote-fs deny-list) lands in one place
/// instead of drifting across three call sites.
///
/// `O_CLOEXEC` is mandatory: a leaked fd across `exec(2)` (cargo
/// subcommand, build-pipeline subprocess, initramfs compressor)
/// would keep the lock alive in the child after the parent's
/// `OwnedFd::drop`, producing phantom holders the next acquirer
/// would blame on the wrong pid.
///
/// 0o666 mode matches a peer first-acquire so the file's owner and
/// permissions don't depend on creation order.
fn open_lockfile(path: &Path) -> Result<OwnedFd> {
    use rustix::fs::{Mode, OFlags, open};

    reject_remote_fs(path)?;
    open(
        path,
        OFlags::CREATE | OFlags::RDWR | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o666),
    )
    .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))
}

/// Ensure the lockfile exists on disk without acquiring a lock.
/// Used by the DISCOVER phase of `acquire_llc_plan` (see
/// `discover_llc_snapshots` in `crate::vmm::host_topology`): the
/// snapshot pass needs every per-LLC lockfile's inode to exist so a
/// subsequent `/proc/locks` match has a target, but DISCOVER itself
/// must not contend with peer acquires.
///
/// Opens through [`open_lockfile`] so the resulting inode and fd
/// mode match what a first-time acquirer would create. Immediately
/// closes the fd — `OwnedFd::drop` releases the open-file
/// description and (since no flock was ever taken on this fd)
/// cannot release a lock held by a peer fd.
pub(crate) fn materialize<P: AsRef<Path>>(path: P) -> Result<()> {
    let fd = open_lockfile(path.as_ref())?;
    drop(fd);
    Ok(())
}

/// Open a lock file and attempt `flock` with `LOCK_NB`.
///
/// Creates the file with mode 0o666 if absent. Returns
/// `Ok(Some(fd))` on successful acquire, `Ok(None)` on
/// `EWOULDBLOCK` (peer already holds an incompatible lock), and
/// propagates other errors. The returned fd owns the open-file
/// description; dropping it closes the fd AND releases the kernel
/// flock (the kernel releases `flock(2)` only when the last fd
/// referring to its OFD closes — `OwnedFd::drop` is what makes that
/// work).
///
/// `O_CLOEXEC` is mandatory: a leaked fd across `exec(2)` (cargo
/// subcommand, build-pipeline subprocess, initramfs compressor) would
/// keep the lock alive in the child process after the parent's
/// `OwnedFd::drop` runs, producing phantom holders the next acquirer
/// would blame on the wrong pid.
///
/// Calls `super::fs_filter::reject_remote_fs` before the open to
/// fail-fast on NFS / CIFS / SMB2 / CEPH / AFS / FUSE — see the
/// module-level rationale.
///
/// Accepts any `AsRef<Path>` so `&str`, `&Path`, `&PathBuf`, and
/// `String` callers all work without string-ifying round trips. LLC
/// lockfile paths are built as `String` via `format!` and cache
/// lockfile paths are built as `PathBuf` via `Path::join` — both
/// pass straight through.
pub fn try_flock<P: AsRef<Path>>(path: P, mode: FlockMode) -> Result<Option<OwnedFd>> {
    use rustix::fs::{FlockOperation, flock};

    let path = path.as_ref();
    let fd = open_lockfile(path)?;
    let op = match mode {
        FlockMode::Exclusive => FlockOperation::NonBlockingLockExclusive,
        FlockMode::Shared => FlockOperation::NonBlockingLockShared,
    };
    match flock(&fd, op) {
        Ok(()) => Ok(Some(fd)),
        Err(e) if e == rustix::io::Errno::WOULDBLOCK => Ok(None),
        Err(e) => anyhow::bail!("flock {}: {e}", path.display()),
    }
}

/// Blocking variant of [`try_flock`]. Opens the lockfile (creating
/// it if absent), then issues a blocking `flock(2)` that parks the
/// caller in the kernel until the lock is available. Use after
/// [`try_flock`] returns `None` to wait for a live peer to finish.
pub fn block_flock<P: AsRef<Path>>(path: P, mode: FlockMode) -> Result<OwnedFd> {
    use rustix::fs::{FlockOperation, flock};

    let path = path.as_ref();
    let fd = open_lockfile(path)?;
    let op = match mode {
        FlockMode::Exclusive => FlockOperation::LockExclusive,
        FlockMode::Shared => FlockOperation::LockShared,
    };
    flock(&fd, op).map_err(|e| anyhow::anyhow!("flock (blocking) {}: {e}", path.display()))?;
    Ok(fd)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`try_flock`] sets `O_CLOEXEC` on the returned fd. Earlier
    /// revisions missed this flag, which leaked flock-held fds
    /// through `execve` into child processes — the child inherited
    /// the lock, broke assumptions about RAII scope, and
    /// manifested as phantom holders in `/proc/locks` long after
    /// the parent had dropped its guard.
    ///
    /// Verifies the bit directly via `fcntl(F_GETFD)` rather than
    /// asserting via a side-effect (forking an exec'd child is
    /// noisier and harder to match). Failure mode: if the bit is
    /// cleared by a future refactor that re-opens the fd without
    /// re-applying O_CLOEXEC, this test fails the build.
    #[test]
    fn try_flock_sets_cloexec_on_returned_fd() {
        use std::os::fd::AsRawFd;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("cloexec.lock");
        let fd = try_flock(&path, FlockMode::Exclusive)
            .expect("try_flock must succeed on fresh tempfile")
            .expect("EX must acquire on clean pool");

        // SAFETY: fd is a valid OwnedFd — fcntl F_GETFD is a pure
        // accessor, no concurrent modification, no ownership move.
        let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFD) };
        assert!(
            flags >= 0,
            "fcntl F_GETFD must succeed on our fd; got errno={}",
            std::io::Error::last_os_error(),
        );
        assert_eq!(
            flags & libc::FD_CLOEXEC,
            libc::FD_CLOEXEC,
            "FD_CLOEXEC must be set on try_flock-returned fd; \
             flags=0x{flags:x}. Without it, exec'd children \
             inherit the flock and produce phantom holders.",
        );

        drop(fd);
    }
}
