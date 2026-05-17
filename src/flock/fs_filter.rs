//! Refuse to operate on filesystems where `flock(2)` is unreliable.
//!
//! `flock(2)` on NFS / CIFS / SMB2 / CephFS / AFS / FUSE is either
//! advisory-only under some server configurations or silently returns
//! success without serializing peers. ktstr's resource-budget contract
//! is not robust to that silent degradation, so this module refuses at
//! lockfile-open time with an actionable operator-facing message.
//!
//! Two seams:
//!
//!  - [`reject_remote_fs`] — production wrapper that statfs's the
//!    lockfile path (or its parent on first-create), reads the
//!    filesystem magic, and calls [`classify_fs_magic`].
//!  - [`classify_fs_magic`] — pure classifier over the [`fs_magic`]
//!    deny-list. Separated so tests can feed synthetic magic values
//!    without a real mount on the host.

use anyhow::Result;
use std::path::Path;

/// Filesystem-magic constants for [`reject_remote_fs`]. Values from
/// `<linux/magic.h>`. Kept as a deny-list (reject known-bad) rather
/// than an allow-list — exotic local filesystems (zfs, erofs, …)
/// are safer to accept with unreliable flock than to reject.
pub(crate) mod fs_magic {
    /// `nfs_super_magic` — NFSv2/3/4 mounts. `flock(2)` on NFS is
    /// advisory-only under some server configurations; rejecting
    /// at lockfile open prevents silent false-success.
    pub(crate) const NFS: i64 = 0x6969;
    /// `cifs_magic_number` — CIFS / SMB1.
    pub(crate) const CIFS: i64 = 0xFF53_4D42;
    /// `smb2_magic_number` — SMB2+. Distinct from the CIFS constant.
    pub(crate) const SMB2: i64 = 0xFE53_4D42;
    /// `ceph_super_magic` — CephFS.
    pub(crate) const CEPH: i64 = 0x00c3_6400;
    /// `AFS_FS_MAGIC` — the in-tree kAFS client's superblock magic
    /// (`fs/afs/super.c:460`, linux/magic.h defines the constant as
    /// `0x6B414653`). Distinct from the legacy `AFS_SUPER_MAGIC =
    /// 0x5346414F` that lingers in `<linux/magic.h>` but is not
    /// emitted by any in-tree AFS driver today — we only reject
    /// what the running kernel actually reports.
    pub(crate) const AFS: i64 = 0x6B41_4653;
    /// `FUSE_SUPER_MAGIC` — any FUSE mount (linux/magic.h line 39:
    /// `#define FUSE_SUPER_MAGIC 0x65735546`). FUSE flock
    /// reliability depends on whether the userspace server
    /// implements the flock op; the safe default is to reject.
    pub(crate) const FUSE: i64 = 0x6573_5546;
}

/// Refuse to operate on filesystems where `flock(2)` is unreliable.
/// Called before every [`super::primitives::try_flock`] open so a
/// misconfigured lockfile path (NFS-mounted `/tmp`, bind-mounted over
/// FUSE) surfaces actionably instead of silently returning an
/// unserialized `OwnedFd`.
///
/// Returns `Ok(())` on accepted filesystems and on statfs failure
/// (non-existent path is an allowed pre-create state — the open
/// below will create it on the parent filesystem's type). Only
/// filesystems whose magic appears in [`fs_magic`]'s deny-list
/// produce an error.
pub(crate) fn reject_remote_fs(path: &Path) -> Result<()> {
    // statfs on the path's PARENT when the path itself does not yet
    // exist — we want to classify the filesystem the lockfile will
    // live on, not error on "path doesn't exist" which is normal for
    // a first-time acquire.
    let target: &Path = if path.exists() {
        path
    } else {
        path.parent().unwrap_or(Path::new("/"))
    };
    let sfs = match rustix::fs::statfs(target) {
        Ok(s) => s,
        // Statfs failure (missing parent, unreadable) is not itself
        // a rejection — defer to the open call to produce a
        // canonical "No such file or directory" error with the right
        // context.
        Err(_) => return Ok(()),
    };
    classify_fs_magic(sfs.f_type as i64).map_err(|rejection| {
        anyhow::anyhow!(
            "{}: filesystem {rejection} Move the lockfile path to a \
             local filesystem (tmpfs, ext4, xfs, btrfs, f2fs, bcachefs).",
            path.display()
        )
    })
}

/// Pure classifier over the [`fs_magic`] deny-list. Returns `Ok(())`
/// when `magic` is an accepted (or unknown-but-not-denied)
/// filesystem, and `Err` with an operator-facing "{name} is not
/// supported for ktstr lockfiles ({reason})." string when the
/// filesystem is on the deny-list.
///
/// Separated from [`reject_remote_fs`] so tests can feed synthetic
/// magic values without a real mount. The caller decorates the
/// error with the lockfile path and the "Move to tmpfs, …" hint;
/// this function produces only the fs-specific middle clause.
pub(crate) fn classify_fs_magic(magic: i64) -> Result<()> {
    let (name, reason) = match magic {
        fs_magic::NFS => (
            "NFS",
            "NFSv3 is advisory-only without an NLM peer; NFSv4 byte-range \
             locking does not cover flock(2)",
        ),
        fs_magic::CIFS | fs_magic::SMB2 => (
            "CIFS/SMB",
            "SMB does not emit /proc/locks entries; ktstr cannot enumerate \
             peer holders",
        ),
        fs_magic::CEPH => (
            "CephFS",
            "Ceph MDS does not participate in flock serialization between \
             ktstr peers on distinct nodes",
        ),
        fs_magic::AFS => ("AFS", "AFS does not support flock(2)"),
        fs_magic::FUSE => (
            "FUSE",
            "flock reliability depends on the userspace server's op \
             implementation",
        ),
        _ => return Ok(()),
    };
    anyhow::bail!("{name} is not supported for ktstr lockfiles ({reason}).")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // fs_magic constant pins — regression guards
    // ---------------------------------------------------------------
    //
    // Both constants were originally typed with one-digit errors
    // that silently missed real FUSE / AFS mounts. These tests pin
    // the corrected values against the kernel's own definitions in
    // `linux/magic.h` so a future "clean-up" that reverts to the
    // legacy/typo variant fails the build.

    /// `FUSE_SUPER_MAGIC` per `linux/magic.h` line 39. A prior
    /// typo used `0x65737546` (wrong digit at position 4) so real
    /// FUSE mounts never matched the deny-list. Regression guard.
    #[test]
    fn fuse_magic_matches_linux_magic_h() {
        assert_eq!(fs_magic::FUSE, 0x65735546);
    }

    /// `AFS_FS_MAGIC` per `linux/magic.h` line 56 — the in-tree
    /// kAFS client's superblock magic (`fs/afs/super.c:460`). A
    /// prior revision used the legacy `AFS_SUPER_MAGIC =
    /// 0x5346414F` which no in-tree driver emits today, so real
    /// AFS mounts never matched. Regression guard.
    #[test]
    fn afs_magic_matches_in_tree_kafs() {
        assert_eq!(fs_magic::AFS, 0x6B414653);
    }

    // ---------------------------------------------------------------
    // classify_fs_magic — deny-list coverage
    // ---------------------------------------------------------------

    /// Every deny-listed magic produces an error naming the
    /// filesystem. The user-facing error string must include the
    /// fs name so operators grepping "NFS is not supported" find
    /// the right diagnostic.
    #[test]
    fn classify_fs_magic_rejects_nfs() {
        let err = classify_fs_magic(fs_magic::NFS).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("NFS"), "err={msg}");
        // Reason substring pins the NFSv3 advisory-only rationale
        // so a future refactor that drops the actionable "why" text
        // regresses through this test, not just the name check.
        assert!(
            msg.contains("NFSv3"),
            "err must name NFSv3 in reason: {msg}"
        );
        assert!(
            msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {msg}",
        );
    }

    #[test]
    fn classify_fs_magic_rejects_cifs_and_smb2() {
        let err_cifs = classify_fs_magic(fs_magic::CIFS).unwrap_err();
        let err_smb2 = classify_fs_magic(fs_magic::SMB2).unwrap_err();
        // Both classify as CIFS/SMB — same arm. Reason pins the
        // "/proc/locks entries" rationale from classify_fs_magic.
        let cifs_msg = format!("{err_cifs:#}");
        let smb2_msg = format!("{err_smb2:#}");
        assert!(cifs_msg.contains("CIFS/SMB"));
        assert!(smb2_msg.contains("CIFS/SMB"));
        assert!(
            cifs_msg.contains("/proc/locks"),
            "err must cite /proc/locks: {cifs_msg}",
        );
        assert!(
            smb2_msg.contains("/proc/locks"),
            "err must cite /proc/locks: {smb2_msg}",
        );
        assert!(
            cifs_msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {cifs_msg}",
        );
        assert!(
            smb2_msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {smb2_msg}",
        );
    }

    #[test]
    fn classify_fs_magic_rejects_ceph() {
        let err = classify_fs_magic(fs_magic::CEPH).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("CephFS"));
        // Reason pins the MDS-doesn't-serialize rationale.
        assert!(msg.contains("MDS"), "err must name Ceph MDS: {msg}");
        assert!(
            msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {msg}",
        );
    }

    #[test]
    fn classify_fs_magic_rejects_afs() {
        let err = classify_fs_magic(fs_magic::AFS).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("AFS"));
        // Reason pins the "AFS does not support flock(2)" substring.
        assert!(msg.contains("flock(2)"), "err must cite flock(2): {msg}");
        assert!(
            msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {msg}",
        );
    }

    #[test]
    fn classify_fs_magic_rejects_fuse() {
        let err = classify_fs_magic(fs_magic::FUSE).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("FUSE"));
        // Reason pins the userspace-server rationale.
        assert!(
            msg.contains("userspace server"),
            "err must name userspace server: {msg}",
        );
        assert!(
            msg.contains("is not supported"),
            "err must contain the canonical rejection phrase: {msg}",
        );
    }

    /// Accepted local filesystems pass through `Ok`. Values from
    /// `linux/magic.h`: TMPFS 0x01021994, EXT4 0xEF53 (also
    /// EXT2/3), XFS 0x58465342, BTRFS 0x9123683E, F2FS 0xF2F52010,
    /// BCACHEFS 0xCA451A4E. Deny-list semantics: unknown local
    /// filesystems (zfs, erofs, …) also pass.
    #[test]
    fn classify_fs_magic_accepts_local_filesystems() {
        classify_fs_magic(0x01021994).expect("tmpfs accepted");
        classify_fs_magic(0xEF53).expect("ext4 accepted");
        classify_fs_magic(0x58465342).expect("xfs accepted");
        classify_fs_magic(0x9123683E).expect("btrfs accepted");
        classify_fs_magic(0xF2F52010).expect("f2fs accepted");
        classify_fs_magic(0xCA451A4E).expect("bcachefs accepted");
        // Unknown magic — deny-list semantics mean "not on the
        // reject list" → Ok.
        classify_fs_magic(0xDEAD_BEEF).expect("unknown magic accepted");
    }
}
