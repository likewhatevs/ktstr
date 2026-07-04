//! Reflink-with-fallback file copy.
//!
//! [`reflink_or_copy`] tries a copy-on-write clone (the `FICLONE` ioctl,
//! btrfs/xfs) and falls back to a plain byte copy when the filesystem cannot
//! reflink or source and destination live on different mounts. The shared
//! [`ficlone`] ioctl inner is also used by
//! [`crate::vmm::disk_template`]'s per-test disk fan-out, which — unlike this
//! module's fallback — REQUIRES the reflink and hard-bails on failure (its
//! per-test disk backing is built on copy-on-write and a byte copy would
//! defeat it).

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;

use anyhow::{Context, Result};

/// `FICLONE` ioctl command number per Linux uapi
/// `include/uapi/linux/fs.h:312` -- `_IOW(0x94, 9, int)`.
///
/// Generic ioctl encoding (shared by x86_64 and aarch64):
/// `(_IOC_WRITE << 30) | (sizeof(int) << 16) | (0x94 << 8) | 9`
/// = `0x40000000 | 0x40000 | 0x9400 | 9` = `0x40049409`.
///
/// This is the same value `reflink-0.1.3/src/sys/unix.rs:10` hardcodes
/// (with a now-stale "is this equal on all archs?" TODO). Pinned here as a
/// `const` so a future arch port can audit it against the host's
/// `<asm/ioctl.h>` instead of grepping a transitive dep.
const FICLONE_IOCTL: libc::c_ulong = 0x4004_9409;

/// Reflink `src`'s extents into `dest` via the `FICLONE` ioctl (btrfs/xfs
/// copy-on-write, O(metadata) — independent of file size). Returns the raw
/// `io::Error` on failure so callers classify the errno: [`reflink_or_copy`]
/// falls back to a byte copy on the "filesystem can't reflink" errnos, while
/// the disk-template per-test fan-out hard-bails on any error.
///
/// `dest` must be opened read+write; `src` must be readable. The ioctl
/// replaces `dest`'s entire contents with a copy-on-write clone of `src`.
pub(crate) fn ficlone(dest: &File, src: &File) -> io::Result<()> {
    // SAFETY: FICLONE takes (dest_fd, FICLONE, src_fd) per Linux uapi. Both
    // fds are valid for the call's duration; ioctl does not retain them. The
    // kernel returns 0 on success and -1 with errno set on failure.
    let rc = unsafe { libc::ioctl(dest.as_raw_fd(), FICLONE_IOCTL, src.as_raw_fd()) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// The errnos that mean "this filesystem cannot reflink" — so a byte copy is
/// the correct result, not an error — per the kernel FICLONE path
/// (`fs/remap_range.c` `vfs_clone_file_range`):
/// - `EXDEV`: source and destination are on different mounts.
/// - `EOPNOTSUPP`: the filesystem has no `remap_file_range` op (tmpfs, ext4, ...).
/// - `ENOTTY`: the filesystem's ioctl dispatch does not recognize FICLONE.
///
/// Everything else (`EBADF`, `EINVAL`, `EISDIR`, `EPERM`, `ENOSPC`,
/// `EDQUOT`, ...) is a genuine failure a copy would MASK, so it propagates.
/// Whitelist, not blacklist: an unknown errno defaults to propagate.
fn is_fallback_errno(errno: i32) -> bool {
    matches!(errno, libc::EXDEV | libc::EOPNOTSUPP | libc::ENOTTY)
}

/// Copy `src` to `dest`, reflinking (copy-on-write) when the filesystem
/// supports it and falling back to a plain byte copy otherwise. Returns the
/// number of bytes `dest` holds.
///
/// Opportunistic: on btrfs/xfs the copy is O(metadata) and extent-shared; on
/// any other filesystem (or across mounts) it degrades to [`std::fs::copy`].
/// Both paths leave `dest` with `src`'s CONTENT and PERMISSION bits — the
/// reflink branch sets the mode explicitly to match `std::fs::copy`, which
/// copies the source's permissions — so a caller cannot tell which path ran.
/// A genuine failure (dest full, quota, permissions, bad path) propagates
/// rather than silently copying.
///
/// `dest` is created-or-truncated (matching `std::fs::copy` semantics), so a
/// re-run into an already-populated path overwrites cleanly.
pub(crate) fn reflink_or_copy(src: &Path, dest: &Path) -> Result<u64> {
    let src_file = File::open(src).with_context(|| format!("open reflink source {src:?}"))?;
    // create-or-truncate (NOT create_new): match std::fs::copy's dest
    // semantics so a re-run into an already-populated dest is idempotent.
    let dest_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(dest)
        .with_context(|| format!("create reflink dest {dest:?}"))?;
    match ficlone(&dest_file, &src_file) {
        Ok(()) => {
            // Perm parity with the fs::copy fallback (which copies the
            // source's mode): set dest's mode from src so both branches leave
            // identical dest state (content + perms).
            let perms = src_file
                .metadata()
                .with_context(|| format!("stat reflink source {src:?}"))?
                .permissions();
            std::fs::set_permissions(dest, perms)
                .with_context(|| format!("set reflink dest perms {dest:?}"))?;
            let len = dest_file
                .metadata()
                .with_context(|| format!("stat reflink dest {dest:?}"))?
                .len();
            Ok(len)
        }
        Err(e) if e.raw_os_error().is_some_and(is_fallback_errno) => {
            // Filesystem can't reflink -> plain copy. Drop the dest fd first
            // so fs::copy's own create+truncate re-opens cleanly; fs::copy
            // copies content AND permission bits.
            drop(dest_file);
            match std::fs::copy(src, dest) {
                Ok(n) => Ok(n),
                Err(copy_err) => {
                    // Symmetric with the genuine-error arm below: never leave
                    // a (partial / 0-byte) dest behind on a failed copy.
                    let _ = std::fs::remove_file(dest);
                    Err(copy_err).with_context(|| {
                        format!("copy {src:?} -> {dest:?} (filesystem has no reflink)")
                    })
                }
            }
        }
        Err(e) => {
            // Genuine failure -- clean up the half-open dest and propagate.
            drop(dest_file);
            let _ = std::fs::remove_file(dest);
            Err(e).with_context(|| {
                format!(
                    "FICLONE {src:?} -> {dest:?} failed and the errno is not a \
                     reflink-unsupported signal"
                )
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// Content + permission parity holds whether the reflink or the copy
    /// fallback ran (a caller can't tell which) — runs on any filesystem.
    #[test]
    fn reflink_or_copy_matches_content_and_perms_either_path() {
        let base = std::env::temp_dir().join(format!("ktstr-reflink-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("mk base");
        let src = base.join("src.bin");
        let dest = base.join("dest.bin");
        std::fs::write(&src, b"reflink-or-copy payload\n").expect("write src");
        // A non-default mode so the perm-parity assertion is meaningful.
        std::fs::set_permissions(&src, std::fs::Permissions::from_mode(0o640)).expect("chmod src");

        let n = reflink_or_copy(&src, &dest).expect("reflink_or_copy");
        assert_eq!(
            n,
            std::fs::metadata(&src).unwrap().len(),
            "returns the byte count",
        );
        assert_eq!(
            std::fs::read(&dest).unwrap(),
            std::fs::read(&src).unwrap(),
            "dest content matches src (reflink or copy fallback)",
        );
        assert_eq!(
            std::fs::metadata(&dest).unwrap().permissions().mode() & 0o777,
            0o640,
            "dest mode matches src on both the reflink and copy paths",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// A genuine failure (dest under a nonexistent parent) must propagate,
    /// never silently succeed as a copy.
    #[test]
    fn reflink_or_copy_propagates_a_real_error() {
        let base = std::env::temp_dir().join(format!("ktstr-reflink-err-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("mk base");
        let src = base.join("src.bin");
        std::fs::write(&src, b"x").expect("write src");
        let dest = base.join("nonexistent-parent").join("dest.bin");
        assert!(
            reflink_or_copy(&src, &dest).is_err(),
            "a dest under a nonexistent parent must error, not silently copy",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// A genuine error (a directory source) must leave NO dest behind,
    /// whichever internal path handles it: on a reflink-capable fs FICLONE
    /// returns a non-fallback errno (the genuine-error arm removes dest); on a
    /// non-reflink fs it falls back to fs::copy, which also fails on a dir
    /// source (the fallback arm now removes dest too). Pins the
    /// no-dest-on-error invariant regardless of the host filesystem.
    #[test]
    fn reflink_or_copy_leaves_no_dest_on_a_genuine_error() {
        let base = std::env::temp_dir().join(format!("ktstr-reflink-dir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("mk base");
        let src_dir = base.join("src-is-a-directory");
        std::fs::create_dir(&src_dir).expect("mk src dir");
        let dest = base.join("dest.bin");
        assert!(
            reflink_or_copy(&src_dir, &dest).is_err(),
            "a directory source is a genuine error (neither reflink nor copy can clone a dir)",
        );
        assert!(
            !dest.exists(),
            "reflink_or_copy leaves no dest behind on a genuine error (both the \
             non-fallback-errno arm and the fallback-copy-failure path clean up)",
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// The errno whitelist: only the "fs can't reflink" errnos fall back to a
    /// copy; every genuine error propagates.
    #[test]
    fn is_fallback_errno_whitelists_only_unsupported() {
        assert!(is_fallback_errno(libc::EXDEV));
        assert!(is_fallback_errno(libc::EOPNOTSUPP));
        assert!(is_fallback_errno(libc::ENOTTY));
        for genuine in [
            libc::EBADF,
            libc::EINVAL,
            libc::EISDIR,
            libc::EPERM,
            libc::ENOSPC,
            0,
        ] {
            assert!(
                !is_fallback_errno(genuine),
                "errno {genuine} must propagate, not fall back",
            );
        }
    }
}
