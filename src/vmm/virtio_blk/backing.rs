//! Block-device backing abstraction: the storage seam virtio-blk's
//! request handlers read and write through.
//!
//! Production backing is a host [`File`] (`impl Backing for File`).
//! The seam exists so tests can inject a fault-injecting backing
//! (e.g. a short-writing mock) to exercise the write retry-to-
//! completion path and the genuine-failure IOERR path that a real
//! regular File rarely triggers. This mirrors qemu's BlockDriver
//! vtable + blkdebug fault-injection driver: `file-posix` is the
//! production driver, `blkdebug` the test one.
//!
//! Scope is deliberately a thin RAW-byte seam — no image-format
//! (qcow2/vmdk), async-completion, discard, or write-zeroes layer
//! (cloud-hypervisor and libkrun carry those; v0 advertises none of
//! the corresponding feature bits, so a method for them would be dead
//! surface). qemu splits flush into `flush_to_disk` / `flush_to_os` /
//! `flush` for its format-layer stack; v0 is a single raw-file backing
//! with no format layer or backing chain, so only the fdatasync-
//! equivalent ([`Backing::sync_data`]) applies.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;

/// The storage a virtio-blk device reads and writes through.
///
/// `Send` (not `Sync`): the backing is owned by `BlkWorkerState`,
/// which is moved into the dedicated worker thread and accessed from
/// that one thread only — never shared across threads concurrently.
/// `Box<dyn Backing>` is therefore `Box<dyn Backing + Send>`,
/// keeping `BlkWorkerState` movable into the worker.
pub(crate) trait Backing: Send {
    /// Read into `buf` at byte `offset`. Scalar; the per-segment
    /// `handle_read_impl` path. Returns bytes read (a short read is
    /// EOF, matching `FileExt::read_at`).
    ///
    /// Used only by the `cfg(test)` per-segment handlers
    /// (`handle_read_impl` / `handle_write_impl`); production read/write
    /// goes through the vectored `preadv`/`pwritev`, so this scalar pair
    /// is dead in the non-test lib build.
    #[cfg_attr(not(test), allow(dead_code))]
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;

    /// Write `buf` at byte `offset`. Scalar; the per-segment
    /// `handle_write_impl` path. Returns bytes written (may be short).
    #[cfg_attr(not(test), allow(dead_code))]
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize>;

    /// Flush data + sizing metadata to disk (`fdatasync`). v0's only
    /// flush variant; `sync_data` matches `File::sync_data` and qemu's
    /// `raw_co_flush_to_disk` (fdatasync, not fsync) — the
    /// `VIRTIO_BLK_T_FLUSH` semantic.
    fn sync_data(&self) -> io::Result<()>;

    /// Vectored positional read — `preadv(2)`. Returns bytes read; a
    /// short positive return (and `0`) is EOF for a regular file, NOT
    /// a signal interrupt (a catchable-signal interrupt yields
    /// `Err(Interrupted)` with 0 transferred). The read retry loop in
    /// `handle_read_vectored_impl` therefore re-issues ONLY on
    /// `Err(Interrupted)` and treats a short positive read as EOF
    /// (zero-padding the tail).
    ///
    /// # Safety
    /// Each `iovec.iov_base` must point at memory valid for `iov_len`
    /// bytes for the duration of the call. The production caller holds
    /// the guest-memory `PtrGuard`s (`_guards`) across the call to
    /// uphold this.
    unsafe fn preadv(&self, iovs: &[libc::iovec], offset: u64) -> io::Result<usize>;

    /// Vectored positional write — `pwritev(2)`. Returns bytes
    /// written; may be short (`0 < n < total`) on ENOSPC mid-write or
    /// a signal-interrupted partial. The write retry loop in
    /// `handle_write_vectored_impl` advances the iovecs past `n` and
    /// re-issues until the full request lands, treating only `Ok(0)`
    /// (zero forward progress) or `Err` as a genuine failure.
    ///
    /// # Safety
    /// As [`Backing::preadv`].
    unsafe fn pwritev(&self, iovs: &[libc::iovec], offset: u64) -> io::Result<usize>;
}

impl Backing for File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        // UFCS: call the inherent `FileExt::read_at`, not this trait
        // method (a bare `self.read_at(..)` would recurse).
        FileExt::read_at(self, buf, offset)
    }

    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        FileExt::write_at(self, buf, offset)
    }

    fn sync_data(&self) -> io::Result<()> {
        File::sync_data(self)
    }

    unsafe fn preadv(&self, iovs: &[libc::iovec], offset: u64) -> io::Result<usize> {
        // SAFETY: the caller upholds the iovec-validity precondition
        // (the `_guards` PtrGuards keep each `iov_base` live for the
        // call). `iovs.len()` is structurally <= IOV_MAX (see the
        // caller's SAFETY comment), so no EINVAL from iovcnt.
        let r = unsafe {
            libc::preadv(
                self.as_raw_fd(),
                iovs.as_ptr(),
                iovs.len() as libc::c_int,
                offset as libc::off_t,
            )
        };
        if r < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(r as usize)
        }
    }

    unsafe fn pwritev(&self, iovs: &[libc::iovec], offset: u64) -> io::Result<usize> {
        // SAFETY: identical precondition to `preadv` above.
        let r = unsafe {
            libc::pwritev(
                self.as_raw_fd(),
                iovs.as_ptr(),
                iovs.len() as libc::c_int,
                offset as libc::off_t,
            )
        };
        if r < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(r as usize)
        }
    }
}

/// Advance an iovec slice past `n` bytes already transferred: drop the
/// fully-consumed leading iovecs, then trim the partially-consumed
/// one's `iov_base`/`iov_len`. Returns the remaining (not-yet-fully-
/// transferred) iovecs for the next `pwritev`. Used by the write
/// retry loop.
///
/// `n` is always < the total length of `iovs` here (the caller breaks
/// its loop on full completion BEFORE calling this), so the returned
/// slice is non-empty. An iovec with `iov_len == n` is fully consumed
/// (dropped), not left as a zero-length partial.
pub(crate) fn advance_iovecs(iovs: &mut [libc::iovec], mut n: usize) -> &mut [libc::iovec] {
    let mut i = 0;
    while i < iovs.len() && n >= iovs[i].iov_len {
        n -= iovs[i].iov_len;
        i += 1;
    }
    let rest = &mut iovs[i..];
    if n > 0 {
        // Partially-consumed iovec at `rest[0]`: advance its base by
        // the remaining `n` and shrink its length. `n > 0` after the
        // drop loop means the consumed prefix ended mid-iovec, so
        // `rest` is non-empty. The advanced base stays within the same
        // descriptor's buffer (n < this iovec's original iov_len), so
        // it does not walk past the `PtrGuard`'s region.
        rest[0].iov_base = (rest[0].iov_base as *mut u8).wrapping_add(n) as *mut libc::c_void;
        rest[0].iov_len -= n;
    }
    rest
}
