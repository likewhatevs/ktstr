#![cfg(test)]
#![allow(unused_imports)]

use super::testing::*;
use super::*;
use std::io::{Seek, Write};
use std::num::NonZeroU64;
use std::os::unix::fs::FileExt;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tempfile::tempfile;
use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_WRITE;
use virtio_queue::desc::{RawDescriptor, split::Descriptor as SplitDescriptor};
use virtio_queue::mock::MockSplitQueue;
use vm_memory::Address;

// ----------------------------------------------------------------
// MockSplitQueue chain-level tests against process_requests.
//
// These exercise the descriptor-chain parsing path
// (`process_requests` in virtio_blk.rs) that the handler-level
// tests above skip. The handler tests bypass virtio-queue's
// descriptor walker entirely; these drive the full pipeline:
// MockSplitQueue plants a chain → MMIO QUEUE_NOTIFY fires →
// `process_requests` walks the chain → handler runs → status
// byte + add_used → UsedRing reflects completion.
//
// Coverage: the chain-shape classifier in `process_requests`
// (header detection, status detection, data-segment collection),
// the throttle integration, the pre-throttle classification (RO
// writes / RO flushes / unknown types), and `add_used`'s used-ring
// publish. None of this is reachable from the handler-level
// tests above.
// ----------------------------------------------------------------

/// Drive a full READ chain through `process_requests`.
/// Plants a 1-sector read chain (header + data + status), fires
/// `process_requests` via QUEUE_NOTIFY, then verifies:
/// - the data descriptor receives the backing-file pattern,
/// - the status descriptor holds VIRTIO_BLK_S_OK,
/// - the UsedRing reflects exactly one completion,
/// - reads_completed counter ticks.
#[test]
fn process_requests_full_read_chain() {
    let cap = 4096u64;
    // Backing file pre-filled with 0xAB so we can detect the
    // bytes propagating from file → guest mem via the chain.
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    // Place the queue rings at GPA 0; data buffers go at high
    // offsets so they don't collide with the ring region.
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Plant the request header.
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Build the chain: header (RO) + data (WRITE-only) + status
    // (WRITE-only, 1 byte). build_desc_chain writes the
    // descriptors AND publishes them to the avail ring so
    // process_requests sees them.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0, // device-readable
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            512,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);
    // Fire QUEUE_NOTIFY — this drives process_requests, which
    // pops the chain and runs the read handler.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Verify the data buffer now holds the backing file's
    // pattern (0xAB) and the status byte is OK.
    let mut data_buf = [0u8; 512];
    mem.read_slice(&mut data_buf, data_addr).unwrap();
    assert!(
        data_buf.iter().all(|&b| b == 0xAB),
        "data segment must contain backing file's 0xAB pattern after read",
    );
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_OK as u8,
        "status byte must be S_OK after successful read",
    );

    // Used ring must reflect one completion. UsedRing.idx is at
    // mock.used_addr() + 2 (after the 2-byte flags field).
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "exactly one used-ring entry expected");

    // Counters: read counted, no errors.
    let c = dev.counters();
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    assert_eq!(c.bytes_read.load(Ordering::Relaxed), 512);
}

/// Drive a full WRITE chain through `process_requests`.
/// Plants a 1-sector write chain, primes the data segment with a
/// distinctive pattern, fires QUEUE_NOTIFY, then verifies:
/// - the backing file receives the planted bytes (`pwrite`
///   landed at the right offset),
/// - the status byte is VIRTIO_BLK_S_OK,
/// - writes_completed and bytes_written tick.
#[test]
fn process_requests_full_write_chain() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let f_for_verify = f.try_clone().expect("clone backing for verify");
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Plant the request header for a WRITE at sector 1.
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_OUT, 1);
    // Plant the data the device should write to the backing file.
    let payload = vec![0xCDu8; 512];
    mem.write_slice(&payload, data_addr).expect("plant payload");
    // Build the chain: header (RO) + data (RO — write request,
    // device READS the data segment) + status (WRITE-only).
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            512,
            0, // device-readable for write
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Verify backing file at offset 512 (= sector 1) holds the
    // payload — proves the chain dispatched to handle_write_impl
    // and the pwrite landed.
    let mut readback = [0u8; 512];
    FileExt::read_at(&f_for_verify, &mut readback, 512).expect("read backing");
    assert!(
        readback.iter().all(|&b| b == 0xCD),
        "backing file at sector 1 must hold the 0xCD payload after write",
    );

    // Status byte == OK.
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(status_buf[0], VIRTIO_BLK_S_OK as u8);

    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    let c = dev.counters();
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.bytes_written.load(Ordering::Relaxed), 512);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}

// ----------------------------------------------------------------
// Fault-injecting Backing — covers the vectored-IO retry arms that a
// real tmpfs File almost never triggers. A regular file's pwritev
// returns the full count or a hard error; it does not short-write or
// return EINTR on tmpfs. These mocks force those arms so the retry
// loops in handle_write_vectored_impl / handle_read_vectored_impl
// (the guest-data-integrity short-write fix and the EINTR bound) are
// exercised through the real process_requests chain path.
// ----------------------------------------------------------------

/// A `Backing` wrapping a real `File` that injects vectored-IO faults.
/// `cfg(test)` inline-engine use only (single-threaded), so the
/// interior `Cell` fault counters need no synchronization.
struct FaultBacking {
    inner: File,
    /// Number of `pwritev` calls that write only HALF the first iovec
    /// (a genuine short positive return) before completing normally.
    write_partials: std::cell::Cell<u32>,
    /// `pwritev` always returns `Ok(0)` — zero forward progress.
    write_zero: bool,
    /// Number of `preadv` calls returning `Err(EINTR)` before
    /// completing normally.
    read_eintrs: std::cell::Cell<u32>,
    /// `preadv`/`pwritev` always return `Err(EINTR)` — exercises the
    /// `MAX_EINTR_RETRIES` cap.
    eintr_forever: bool,
}

impl FaultBacking {
    fn write_partial_then_complete(inner: File, partials: u32) -> Self {
        Self {
            inner,
            write_partials: std::cell::Cell::new(partials),
            write_zero: false,
            read_eintrs: std::cell::Cell::new(0),
            eintr_forever: false,
        }
    }
    fn write_zero_progress(inner: File) -> Self {
        Self {
            inner,
            write_partials: std::cell::Cell::new(0),
            write_zero: true,
            read_eintrs: std::cell::Cell::new(0),
            eintr_forever: false,
        }
    }
    fn read_eintr_then_complete(inner: File, eintrs: u32) -> Self {
        Self {
            inner,
            write_partials: std::cell::Cell::new(0),
            write_zero: false,
            read_eintrs: std::cell::Cell::new(eintrs),
            eintr_forever: false,
        }
    }
    fn eintr_forever(inner: File) -> Self {
        Self {
            inner,
            write_partials: std::cell::Cell::new(0),
            write_zero: false,
            read_eintrs: std::cell::Cell::new(0),
            eintr_forever: true,
        }
    }
}

impl Backing for FaultBacking {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
        Backing::read_at(&self.inner, buf, offset)
    }
    fn write_at(&self, buf: &[u8], offset: u64) -> std::io::Result<usize> {
        Backing::write_at(&self.inner, buf, offset)
    }
    fn sync_data(&self) -> std::io::Result<()> {
        Backing::sync_data(&self.inner)
    }
    unsafe fn preadv(&self, iovs: &[libc::iovec], offset: u64) -> std::io::Result<usize> {
        if self.eintr_forever {
            return Err(std::io::Error::from_raw_os_error(libc::EINTR));
        }
        if self.read_eintrs.get() > 0 {
            self.read_eintrs.set(self.read_eintrs.get() - 1);
            return Err(std::io::Error::from_raw_os_error(libc::EINTR));
        }
        // SAFETY: forwards the caller's iovec-validity precondition
        // unchanged to the inner File's preadv.
        unsafe { Backing::preadv(&self.inner, iovs, offset) }
    }
    unsafe fn pwritev(&self, iovs: &[libc::iovec], offset: u64) -> std::io::Result<usize> {
        if self.eintr_forever {
            return Err(std::io::Error::from_raw_os_error(libc::EINTR));
        }
        if self.write_zero {
            return Ok(0);
        }
        if self.write_partials.get() > 0 && !iovs.is_empty() && iovs[0].iov_len > 1 {
            self.write_partials.set(self.write_partials.get() - 1);
            // Write only HALF the first iovec — a genuine short positive
            // return that forces the retry loop to advance iov_base
            // mid-iovec (the bug-prone advance_iovecs partial trim).
            let first = iovs[0];
            let partial = libc::iovec {
                iov_base: first.iov_base,
                iov_len: first.iov_len / 2,
            };
            // SAFETY: `partial` aliases the first caller iovec's base
            // with a SHORTER length (half < iov_len), so it stays within
            // the validated region; forwards the validity precondition.
            return unsafe {
                Backing::pwritev(&self.inner, std::slice::from_ref(&partial), offset)
            };
        }
        // SAFETY: forwards the caller's validity precondition unchanged.
        unsafe { Backing::pwritev(&self.inner, iovs, offset) }
    }
}

/// A backing that short-writes the data segment once (writes half,
/// returns a short positive count) must RETRY the remainder and land
/// the FULL payload — not give up with IOERR. Pins the guest-data-
/// integrity short-write retry-to-completion fix: a short positive
/// `pwritev` is recoverable forward progress (the kernel's
/// `generic_perform_write` copy loop likewise re-issues), so a
/// half-completed write must finish, not corrupt the guest's data by
/// reporting success with only half written.
#[test]
fn process_requests_write_chain_short_write_retries_to_completion() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let f_for_verify = f.try_clone().expect("clone backing for verify");
    let fault_inner = f.try_clone().expect("clone backing for fault mock");
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    // Inject the fault backing: one partial write, then complete.
    dev.worker.state_mut().backing =
        Box::new(FaultBacking::write_partial_then_complete(fault_inner, 1));
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_OUT, 1);
    let payload = vec![0xCDu8; 512];
    mem.write_slice(&payload, data_addr).expect("plant payload");
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(data_addr.0, 512, 0, 0)),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // The FULL 512-byte payload must have landed at sector 1 despite
    // the mid-write short return.
    let mut readback = [0u8; 512];
    FileExt::read_at(&f_for_verify, &mut readback, 512).expect("read backing");
    assert!(
        readback.iter().all(|&b| b == 0xCD),
        "full payload must land after a short-write retry, not a half-write",
    );
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_OK as u8,
        "short-then-complete write is S_OK",
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);
    let c = dev.counters();
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 1);
    assert_eq!(
        c.bytes_written.load(Ordering::Relaxed),
        512,
        "bytes_written counts the full payload, not the partial first write",
    );
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}

/// A backing whose `pwritev` makes ZERO forward progress (`Ok(0)`) is
/// a genuine failure — the write handler must give up with S_IOERR
/// rather than spin, bump `io_errors`, and NOT count a completion or
/// any bytes. The (failed) completion is still published so the guest
/// observes the IOERR status.
#[test]
fn process_requests_write_chain_zero_progress_is_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let fault_inner = f.try_clone().expect("clone backing for fault mock");
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    dev.worker.state_mut().backing = Box::new(FaultBacking::write_zero_progress(fault_inner));
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_OUT, 1);
    let payload = vec![0xCDu8; 512];
    mem.write_slice(&payload, data_addr).expect("plant payload");
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(data_addr.0, 512, 0, 0)),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_IOERR as u8,
        "zero-progress write is S_IOERR",
    );
    // The IOERR status byte wrote successfully, so add_used publishes
    // the completion — the guest must see the failure.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);
    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(
        c.writes_completed.load(Ordering::Relaxed),
        0,
        "a failed write is not a completion",
    );
    assert_eq!(c.bytes_written.load(Ordering::Relaxed), 0);
}

/// A backing whose `preadv` returns EINTR twice before completing must
/// RETRY (a catchable-signal interrupt yields `Err(Interrupted)` with
/// zero transfer) and deliver the data — S_OK, not IOERR.
#[test]
fn process_requests_read_chain_eintr_retries_to_completion() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let fault_inner = f.try_clone().expect("clone backing for fault mock");
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    dev.worker.state_mut().backing =
        Box::new(FaultBacking::read_eintr_then_complete(fault_inner, 2));
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            512,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut data_buf = [0u8; 512];
    mem.read_slice(&mut data_buf, data_addr).unwrap();
    assert!(
        data_buf.iter().all(|&b| b == 0xAB),
        "data delivered after the EINTR retries",
    );
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(status_buf[0], VIRTIO_BLK_S_OK as u8);
    let c = dev.counters();
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.bytes_read.load(Ordering::Relaxed), 512);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}

/// A backing whose `preadv` ALWAYS returns EINTR must give up with
/// S_IOERR after `MAX_EINTR_RETRIES` — the bound prevents an unbounded
/// spin when a pending FATAL signal keeps interrupting (a fatal signal
/// does not clear, so re-issue would never succeed and a spinning IO
/// thread would delay the freeze-rendezvous failure dump).
#[test]
fn process_requests_read_chain_eintr_forever_hits_retry_cap() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let fault_inner = f.try_clone().expect("clone backing for fault mock");
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    dev.worker.state_mut().backing = Box::new(FaultBacking::eintr_forever(fault_inner));
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            512,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_IOERR as u8,
        "unbounded EINTR fails with S_IOERR after the retry cap",
    );
    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
}

/// All-zero-length data segments on a T_OUT chain: the chain HAS
/// data descriptors (so `data_segments` is non-empty), but every one
/// has `len == 0`. A conforming Linux guest never emits this shape
/// (`blk_rq_map_sg` yields only non-zero data segments), so it is a
/// malformed/hostile chain — but the virtio-blk spec does not forbid
/// a 0-byte data area, so the device accepts it as a 0-byte op rather
/// than crashing. The drain's zero-data reject gates on
/// `data_segments.is_empty()` (segment COUNT, drain.rs), so this
/// chain PASSES the gate and reaches the vectored write helper, where
/// the empty iovec list triggers the `record_write(0)` early-return
/// (device.rs). Pins that such a chain is ACCEPTED as a 0-byte write
/// (S_OK), NOT rejected — a refactor that re-gated the reject on
/// `data_len == 0` would silently start IOERR-ing it, and this test
/// would catch the change.
#[test]
fn process_requests_write_chain_all_zero_len_segments_is_zero_byte_ok() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data0_addr = GuestAddress(0x5000);
    let data1_addr = GuestAddress(0x5100);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_OUT, 1);
    // header (RO) + 2 zero-length data descs (RO — T_OUT data is
    // device-readable) + status (WRITE). Two segments make the
    // "non-empty but every len==0" case unambiguous.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(data0_addr.0, 0, 0, 0)),
        RawDescriptor::from(SplitDescriptor::new(data1_addr.0, 0, 0, 0)),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_OK as u8,
        "all-zero-len T_OUT must complete S_OK, not be rejected",
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "chain marked used exactly once");
    let c = dev.counters();
    assert_eq!(
        c.writes_completed.load(Ordering::Relaxed),
        1,
        "a 0-byte write counts as one completed write",
    );
    assert_eq!(
        c.bytes_written.load(Ordering::Relaxed),
        0,
        "zero bytes written",
    );
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        0,
        "all-zero-len chain is NOT an IO error",
    );
}

/// Mirror of `process_requests_write_chain_all_zero_len_segments_is_zero_byte_ok`
/// on the T_IN read path: a malformed/hostile chain whose data
/// segments are all `len == 0` (a conforming guest never emits it)
/// reaches the read vectored helper, where empty iovecs trigger the
/// 0-byte early-return (device.rs) → S_OK, 0 bytes. Pins the read
/// side of the same accepted-as-0-byte branch.
#[test]
fn process_requests_read_chain_all_zero_len_segments_is_zero_byte_ok() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data0_addr = GuestAddress(0x5000);
    let data1_addr = GuestAddress(0x5100);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // header (RO) + 2 zero-length data descs (WRITE-only — T_IN data
    // is device-writable) + status (WRITE).
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data0_addr.0,
            0,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data1_addr.0,
            0,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_OK as u8,
        "all-zero-len T_IN must complete S_OK, not be rejected",
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "chain marked used exactly once");
    let c = dev.counters();
    assert_eq!(
        c.reads_completed.load(Ordering::Relaxed),
        1,
        "a 0-byte read counts as one completed read",
    );
    assert_eq!(c.bytes_read.load(Ordering::Relaxed), 0, "zero bytes read",);
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        0,
        "all-zero-len chain is NOT an IO error",
    );
}

/// Drive a chain with an UNKNOWN request type through
/// `process_requests`. The dispatch table pre-classifies any
/// req_type outside `T_IN`/`T_OUT`/`T_FLUSH` as
/// VIRTIO_BLK_S_UNSUPP. Verifies:
/// - status byte is VIRTIO_BLK_S_UNSUPP (2), NOT IOERR (1),
/// - the chain still completes (used ring updated),
/// - io_errors does NOT tick (UNSUPP is not an IO error — the
///   device gracefully declined a request it didn't recognise),
/// - reads/writes/flushes counters all stay at 0.
#[test]
fn process_requests_unknown_type_returns_unsupp() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    // Type 0xBEEF is outside every known T_* opcode (T_IN=0,
    // T_OUT=1, T_FLUSH=4, T_GET_ID=8). Sector irrelevant for an
    // unknown-type rejection — never reaches the backing path.
    write_blk_header(&mem, header_addr, 0xBEEF, 0);
    // No data segment — UNSUPP rejection happens before any
    // data-segment walk. Header + status only is the minimal
    // legal chain shape.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Status byte must be UNSUPP, not IOERR. A regression that
    // mapped unknown types to IOERR would corrupt the
    // pre-throttle classifier's contract; one that mapped them
    // to OK would silently let bogus requests pass.
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_UNSUPP as u8,
        "unknown req_type must produce S_UNSUPP, not S_IOERR or S_OK",
    );

    // Used ring still reflects completion — the device returned
    // the descriptor chain to the guest with the UNSUPP status
    // rather than leaking it.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "UNSUPP completions still update used.idx");

    // io_errors stays 0 — UNSUPP is not classified as an IO
    // error.
    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 0);
}

/// Drive a FLUSH chain through `process_requests`.
/// FLUSH has no data segment per virtio-v1.2 §5.2.6 — the chain
/// is exactly header + status. Verifies the dispatch reaches
/// `handle_flush_impl` (calls fdatasync), increments
/// flushes_completed, and writes S_OK status.
#[test]
fn process_requests_flush_chain() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_FLUSH, 0);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(status_buf[0], VIRTIO_BLK_S_OK as u8);

    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    let c = dev.counters();
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}

/// Drive a chain whose first descriptor is too short
/// to hold the 16-byte `virtio_blk_outhdr`. The chain parser
/// rejects when `first_len < VIRTIO_BLK_OUTHDR_SIZE`, leaving
/// `header_addr` unset; the dispatcher writes S_IOERR to status,
/// increments io_errors, and add_used's the chain.
#[test]
fn process_requests_short_header_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    // Header descriptor with len=8 (too short — outhdr is 16
    // bytes). The bytes there don't matter because the device
    // never reads them: `first_len < OUTHDR_SIZE` skips the
    // read entirely.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            8, // SHORT — half the required 16 bytes
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], VIRTIO_BLK_S_IOERR as u8,
        "short header must be rejected with S_IOERR",
    );

    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 0);
}

/// Drive a chain whose last descriptor is NOT
/// device-writable. Per virtio-v1.2 §5.2.6 the status descriptor
/// MUST be device-writable. The parser only accepts the last
/// descriptor as status when its `is_write_only` flag is set;
/// otherwise `status_addr` stays None and the dispatcher
/// drops the chain entirely (does NOT call `add_used`,
/// does NOT write a status byte).
///
/// CRITICAL: calling `add_used` here would tell the guest the
/// request SUCCEEDED. The kernel driver's `virtblk_done` reads
/// the status from the request's `vbr->in_hdr`, which is stale
/// from prior blk-mq tag use (initially zero from `__GFP_ZERO`
/// at allocation, stale on reuse); `virtblk_result(0) ==
/// BLK_STS_OK` (drivers/block/virtio_blk.c). With the device having
/// written no status byte to any guest-visible location, a
/// completed-but-unstatused request would surface as a phantom
/// success — silent data corruption for reads, silent dropped
/// writes for writes. The fix: leave the descriptor in the
/// avail ring; the guest hangs on this request until
/// `kernel.hung_task_timeout_secs` (default 120 s) fires or a
/// higher layer retries (virtio_blk has no `mq_ops->timeout`,
/// so blk-mq alone won't surface the stall).
#[test]
fn process_requests_status_not_writable_drops_chain() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    // Plant a sentinel byte at status_addr so we can detect
    // whether the device wrote anything (it should NOT).
    // 0xEE chosen because it's distinct from S_OK (0),
    // S_IOERR (1), S_UNSUPP (2).
    mem.write_slice(&[0xEEu8], status_addr).unwrap();
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Last descriptor lacks WRITE flag — disqualifies it as
    // status. The parser reads it as a (degenerate) data
    // segment and finds no status descriptor.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            0, // device-readable, NOT write-only
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Sentinel must remain — device wrote nothing because it
    // had no status_addr.
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(
        status_buf[0], 0xEE,
        "no status descriptor → device must not write a status byte; \
             sentinel 0xEE survives",
    );

    // The chain MUST stay in the avail ring (used.idx unchanged
    // at 0). Calling add_used here would let the guest's
    // virtblk_done observe an in_hdr.status that's stale from
    // prior blk-mq tag use (initially zero from __GFP_ZERO at
    // allocation, stale on reuse) as BLK_STS_OK — phantom
    // success.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 0,
        "no-status chain must NOT advance used.idx; advancing would \
             let the guest's stale in_hdr.status surface as \
             BLK_STS_OK (silent data corruption)",
    );

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    // No-status chain → publish_completion never runs →
    // signal_needed stays false → interrupt_status bit MUST
    // remain 0 (the bit is set inside `if signal_needed`,
    // drain_bracket_impl). A regression that set the bit
    // unconditionally on every notify would leak phantom
    // interrupts to the guest's polling path.
    assert_eq!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0
    );
}

/// Multi-chain FIFO ordering through
/// `process_requests`. Plant 3 distinct READ chains in the
/// avail ring, single QUEUE_NOTIFY drains them all. Verify
/// (a) all 3 complete in the same `process_requests` call,
/// (b) the used.idx advances by exactly 3, (c) reads_completed
/// counter == 3, (d) each chain's data buffer holds the correct
/// per-chain pattern.
#[test]
fn process_requests_multiple_chains_drained_in_one_notify() {
    use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_NEXT;
    let cap = 4096u64;
    let mut f = tempfile().unwrap();
    f.set_len(cap).unwrap();
    // Stamp three distinct sectors with three distinct
    // patterns so each chain's read result is identifiable.
    f.write_all(&[0x11; 512]).unwrap(); // sector 0
    f.write_all(&[0x22; 512]).unwrap(); // sector 1
    f.write_all(&[0x33; 512]).unwrap(); // sector 2
    f.rewind().unwrap();
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);

    let chains = [
        (
            GuestAddress(0x4000),
            GuestAddress(0x4400),
            GuestAddress(0x4800),
            0u64,
        ),
        (
            GuestAddress(0x5000),
            GuestAddress(0x5400),
            GuestAddress(0x5800),
            1u64,
        ),
        (
            GuestAddress(0x6000),
            GuestAddress(0x6400),
            GuestAddress(0x6800),
            2u64,
        ),
    ];
    for &(hdr, _, _, sector) in &chains {
        write_blk_header(&mem, hdr, VIRTIO_BLK_T_IN, sector);
    }

    let mut descs = Vec::new();
    for (chain_i, &(hdr, data, status, _)) in chains.iter().enumerate() {
        // Each chain occupies 3 sequential descriptor-table
        // slots starting at chain_i * 3. The non-last
        // descriptors must point to their successor via the
        // `next` field so the device's queue iterator walks
        // the whole chain. add_desc_chains writes descriptors
        // verbatim — it does NOT auto-link them (only
        // `build_desc_chain` does, and that path takes a
        // single chain).
        let base = (chain_i as u16) * 3;
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            hdr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            VRING_DESC_F_NEXT as u16,
            base + 1,
        )));
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            data.0,
            512,
            VRING_DESC_F_WRITE as u16 | VRING_DESC_F_NEXT as u16,
            base + 2,
        )));
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            status.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )));
    }
    // add_desc_chains writes the descriptor table sequentially
    // and bumps avail.idx for each chain head.
    mock.add_desc_chains(&descs, 0).expect("add 3 chains");

    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 3, "single notify must drain 3 chains");

    let c = dev.counters();
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 3);
    assert_eq!(c.bytes_read.load(Ordering::Relaxed), 3 * 512);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);

    for (i, &(_, data, _, _)) in chains.iter().enumerate() {
        let mut buf = [0u8; 512];
        mem.read_slice(&mut buf, data).unwrap();
        let expected = (i as u8 + 1) * 0x11;
        assert!(
            buf.iter().all(|&b| b == expected),
            "chain {i}'s data must hold sector {i}'s pattern (0x{expected:02X})",
        );
    }
}

/// A `Backing` that records the largest iovcnt passed to
/// `preadv`/`pwritev`, so a chain test can confirm the vectored
/// handlers build the iovec chain within the `SmallVec` inline capacity
/// (`MAX_BLK_IOVECS`) — i.e. with no heap spill — for a legitimate
/// multi-segment chain. The counters are `Arc<AtomicUsize>` because
/// `Backing: Send` (backing.rs) requires the backing — and thus its
/// fields — to be `Send`; an `Rc<Cell>` would not satisfy that bound.
/// `fetch_max` with `Relaxed` ordering suffices (the `cfg(test)` inline
/// engine is single-threaded, so no cross-thread ordering is needed).
struct RecordingBacking {
    inner: File,
    max_read_iovcnt: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    max_write_iovcnt: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Backing for RecordingBacking {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
        Backing::read_at(&self.inner, buf, offset)
    }
    fn write_at(&self, buf: &[u8], offset: u64) -> std::io::Result<usize> {
        Backing::write_at(&self.inner, buf, offset)
    }
    fn sync_data(&self) -> std::io::Result<()> {
        Backing::sync_data(&self.inner)
    }
    unsafe fn preadv(&self, iovs: &[libc::iovec], offset: u64) -> std::io::Result<usize> {
        self.max_read_iovcnt.fetch_max(iovs.len(), Ordering::Relaxed);
        // SAFETY: forwards the caller's iovec-validity precondition unchanged.
        unsafe { Backing::preadv(&self.inner, iovs, offset) }
    }
    unsafe fn pwritev(&self, iovs: &[libc::iovec], offset: u64) -> std::io::Result<usize> {
        self.max_write_iovcnt.fetch_max(iovs.len(), Ordering::Relaxed);
        // SAFETY: forwards the caller's iovec-validity precondition unchanged.
        unsafe { Backing::pwritev(&self.inner, iovs, offset) }
    }
}

/// A legitimate maximal multi-segment read chain (`VIRTIO_BLK_SEG_MAX`
/// = 128 device-writable data descriptors — the drain's
/// `SCRATCH_CAP = SEG_MAX + 2` acceptance boundary — each in the single
/// test memory region so each yields exactly one iovec) must build its
/// iovec chain INLINE in the handler's `SmallVec<[_; MAX_BLK_IOVECS]>` —
/// no heap spill. We cannot observe `SmallVec::spilled()` from outside
/// the handler, so we assert the externally-observable equivalent: the
/// iovcnt the backing sees (== the SmallVec length) is
/// `<= MAX_BLK_IOVECS`, which guarantees a SmallVec of that capacity
/// never spilled. Pins the zero-per-request-allocation fix (the inline
/// `SmallVec` iovec build).
#[test]
fn process_requests_read_chain_builds_iovecs_inline_no_spill() {
    let cap = 1u64 << 20; // 1 MiB backing
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let recording_inner = f.try_clone().expect("clone backing for recording mock");
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let read_iovcnt = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    dev.worker.state_mut().backing = Box::new(RecordingBacking {
        inner: recording_inner,
        max_read_iovcnt: read_iovcnt.clone(),
        max_write_iovcnt: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    });
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 256);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x20000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Header (RO) + SEG_MAX (128) device-writable 512-byte data segments
    // + status = a 130-descriptor chain == the drain's SCRATCH_CAP
    // (SEG_MAX + 2) acceptance boundary. The queue is 256-deep to hold
    // the chain (wire_device_to_mock negotiates QUEUE_MAX_SIZE = 256).
    // Each segment is placed in the single test region so get_slices
    // yields exactly one iovec per segment (no region straddle); the
    // 128 segments span 0x5000..0x15000, clear of the status byte at
    // 0x20000 and the 256-deep rings below 0x4000.
    const N_SEGS: usize = 128;
    let mut descs = vec![RawDescriptor::from(SplitDescriptor::new(
        header_addr.0,
        VIRTIO_BLK_OUTHDR_SIZE as u32,
        0, // device-readable header
        0,
    ))];
    for i in 0..N_SEGS {
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            0x5000 + (i as u64) * 0x200,
            512,
            VRING_DESC_F_WRITE as u16,
            0,
        )));
    }
    descs.push(RawDescriptor::from(SplitDescriptor::new(
        status_addr.0,
        1,
        VRING_DESC_F_WRITE as u16,
        0,
    )));
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Read completed cleanly through the SmallVec path.
    let mut status_buf = [0u8; 1];
    mem.read_slice(&mut status_buf, status_addr).unwrap();
    assert_eq!(status_buf[0], VIRTIO_BLK_S_OK as u8, "read must be S_OK");
    assert_eq!(dev.counters().reads_completed.load(Ordering::Relaxed), 1);

    let seen = read_iovcnt.load(Ordering::Relaxed);
    assert_eq!(
        seen, N_SEGS,
        "{N_SEGS} single-region segments must yield {N_SEGS} iovecs"
    );
    assert!(
        seen <= MAX_BLK_IOVECS,
        "iovcnt {seen} must fit the SmallVec inline cap {MAX_BLK_IOVECS} (no heap spill)",
    );
}
