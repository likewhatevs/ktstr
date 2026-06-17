#![allow(unused_imports)]

use super::super::testing::*;
use super::super::*;
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
// Validation-gap tests against process_requests.
//
// Every test below pins a code path that production exercises
// in real-world failure modes but no upstream test covered. All
// are driven through the PUBLIC interface (`process_requests`
// via QUEUE_NOTIFY + MockSplitQueue) — internal helpers may
// change shape, but the chain-walking → classify → dispatch →
// publish-completion contract is the invariant.
// ----------------------------------------------------------------

/// SEG_MAX enforcement. The device advertises
/// `VIRTIO_BLK_F_SEG_MAX = 128`, so a chain with more than
/// 128 + 2 (header + status) descriptors is malformed.
/// Without this gate, a hostile guest could submit thousands
/// of descriptors and force the device to allocate matching
/// scratch storage per request (heap blowup).
///
/// Oversized chains drop entirely — no add_used, no IOERR
/// publish, no status write. The descriptor walk is bounded
/// at `VIRTIO_BLK_SEG_MAX + 2` to keep the scratch Vec inside
/// its preallocated capacity, but a capped scratch view loses
/// sight of the chain's true last descriptor (the guest's
/// status). Publishing IOERR with a misidentified status_addr
/// would write a status byte to a non-status descriptor while
/// the guest's real status byte stays uninitialized — the
/// guest's `virtblk_done` then reads `vbr->in_hdr.status` from
/// the stale blk-mq tag bytes, which `virtblk_result(0)` maps
/// to `BLK_STS_OK` (the silent-data-corruption pattern).
/// Dropping the chain forces the request to surface via the
/// hung-task watchdog instead — same blast radius as the
/// "no status descriptor" branch, which is the documented
/// fallback for spec-violating chains.
#[test]
fn seg_max_dropped_no_publish() {
    use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_NEXT;
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    // Need queue size > 130 so the chain fits in the table even
    // though the device's SEG_MAX gate rejects it.
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 256);
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);

    // Build 132 descriptors total: 1 header + 130 data + 1 status
    // = 132 > VIRTIO_BLK_SEG_MAX (128) + 2 = 130.
    let header_addr = GuestAddress(0x10000);
    let status_addr = GuestAddress(0x20000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    let total_descs: u16 = 132;
    let data_descs: u16 = total_descs - 2;
    let mut descs = Vec::new();
    descs.push(RawDescriptor::from(SplitDescriptor::new(
        header_addr.0,
        VIRTIO_BLK_OUTHDR_SIZE as u32,
        VRING_DESC_F_NEXT as u16,
        1,
    )));
    for i in 0..data_descs {
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            0x40000 + i as u64 * 8,
            8,
            VRING_DESC_F_WRITE as u16 | VRING_DESC_F_NEXT as u16,
            i + 2,
        )));
    }
    descs.push(RawDescriptor::from(SplitDescriptor::new(
        status_addr.0,
        1,
        VRING_DESC_F_WRITE as u16,
        0,
    )));
    // Pre-fill status_addr with 0xEE — a value distinct from
    // S_OK (0), S_IOERR (1), S_UNSUPP (2). The post-notify
    // assertion expects the device to leave this byte alone:
    // oversized chains drop without any status-byte write.
    mem.write_slice(&[0xEEu8], status_addr).unwrap();
    mock.add_desc_chains(&descs, 0).expect("add chain");
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Used ring stays empty — oversized chain is dropped before
    // status_addr extraction so add_used is never called.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 0, "SEG_MAX drop must NOT advance used.idx");

    let c = dev.counters();
    assert!(c.io_errors.load(Ordering::Relaxed) >= 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    // Throttle untouched — drop fires before token consumption.
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);

    // Status byte stays at the 0xEE sentinel — a SEG_MAX drop
    // must not touch the status descriptor (the capped scratch
    // view cannot identify the true last descriptor, so any
    // write would target the wrong GPA).
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], 0xEE,
        "SEG_MAX drop must leave status descriptor untouched",
    );
}

/// Header read_obj failure. The header descriptor's
/// `addr` points at unmapped guest memory, so `mem.read_obj`
/// fails. The device writes IOERR to status, increments
/// io_errors, calls add_used.
#[test]
fn header_read_obj_failure_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    // Header at GPA 0x200000 — past the 1 MiB region's end.
    // status_addr stays inside the region so the IOERR write
    // can succeed.
    let header_addr = GuestAddress(0x200000);
    let status_addr = GuestAddress(0x4000);
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], VIRTIO_BLK_S_IOERR as u8,
        "header read failure must surface as S_IOERR",
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    let c = dev.counters();
    assert!(c.io_errors.load(Ordering::Relaxed) >= 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
}

/// SIZE_MAX enforcement. A single data descriptor longer
/// than VIRTIO_BLK_SIZE_MAX (1 MiB) is malformed. Without the
/// gate, a guest can force `vec![0u8; 4 GB]` heap allocations.
#[test]
fn size_max_oversized_data_desc_rejected() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x8000);
    let status_addr = GuestAddress(0x9000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Data desc len = 1 MiB + 1.
    let oversize: u32 = (1u32 << 20) + 1;
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            oversize,
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
}

/// Zero-data T_IN. virtio-v1.2 §5.2.6 defines IN/OUT as
/// carrying a non-empty data payload; cloud-hypervisor
/// explicitly rejects header+status-only chains for these
/// request types.
#[test]
fn zero_data_t_in_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x5000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);
    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
}

/// Sub-sector data_len. virtio-v1.2 §5.2.6 defines
/// T_IN/T_OUT as sector-aligned transfers; firecracker's
/// `Request::parse` rejects sub-sector lengths.
#[test]
fn sub_sector_data_len_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Data len = 513.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_addr.0,
            VIRTIO_BLK_OUTHDR_SIZE as u32,
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            data_addr.0,
            513,
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
}

/// Direction violation. T_IN with a non-write-only data
/// descriptor is a spec violation. Must reject; throttle
/// untouched (gate fires pre-consume).
#[test]
fn direction_violation_t_in_with_ro_data_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
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
            0, // device-readable — wrong for T_IN
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        0,
        "direction violation must NOT touch throttle bucket",
    );
}

/// Direction violation. T_OUT with a device-writable data
/// descriptor is a spec violation (virtio-v1.2 §5.2.6: T_OUT
/// data segments must be device-readable). Must reject;
/// throttle untouched (gate fires pre-consume); writes_completed
/// stays at 0; backing file untouched. Exercises the
/// `VIRTIO_BLK_T_OUT => data_segments.iter().any(|d| d.is_write_only)`
/// match arm in `process_requests`.
#[test]
fn direction_violation_t_out_with_writable_data_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Pre-fill the data segment with a sentinel so we can
    // verify the device did NOT read from the backing file
    // into it (which would indicate the wrong code path ran).
    let sentinel = vec![0xCDu8; 512];
    mem.write_slice(&sentinel, data_addr).unwrap();
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_OUT, 1);
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
            VRING_DESC_F_WRITE as u16, // device-writable — wrong for T_OUT
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(
        c.writes_completed.load(Ordering::Relaxed),
        0,
        "direction-violating T_OUT must NOT count as a completed write",
    );
    assert_eq!(
        c.bytes_written.load(Ordering::Relaxed),
        0,
        "direction-violating T_OUT must NOT touch the backing file",
    );
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        0,
        "direction violation must NOT touch throttle bucket",
    );
    // Sentinel must remain — proves the device did not read
    // backing data into the data segment (T_IN handler path
    // would write into it, which would indicate the gate
    // failed and the wrong arm ran).
    let mut data_check = vec![0u8; 512];
    mem.read_slice(&mut data_check, data_addr).unwrap();
    assert!(
        data_check.iter().all(|&b| b == 0xCDu8),
        "data segment sentinel must be intact — device must not run the read or write path",
    );
}

/// Status write_slice failure. The status descriptor
/// points at unmapped guest memory. When status
/// write fails, the device does NOT call add_used. The
/// descriptor head stays in the avail ring; io_errors bumps.
#[test]
fn status_write_slice_failure_no_add_used() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    // Status_addr beyond the 1 MiB region → write_slice fails.
    let status_addr = GuestAddress(0x300000);
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

    // used.idx must NOT advance — no add_used when
    // status write fails.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 0,
        "status write failure must skip add_used (status-write-success gate); used.idx stays at 0",
    );

    let c = dev.counters();
    assert!(
        c.io_errors.load(Ordering::Relaxed) >= 1,
        "status write failure bumps io_errors",
    );
}

/// add_used Err path baseline. A clean fixture cannot
/// force `add_used` to fail (virtio-queue's add_used returns
/// Ok unless the head is out of range or the used ring address
/// is invalid). Best-effort coverage: pin that successful
/// add_used does NOT bump io_errors. A regression that
/// introduced a new add_used-fail site would surface as
/// elevated io_errors here.
///
/// Production add_used Err arms (lines that bump io_errors
/// when add_used returns Err) are reviewed in code: every
/// arm matches the established `if let Err(e) = q.add_used(...)
/// { ... io_errors.fetch_add(1) }` shape.
#[test]
fn add_used_err_path_baseline_io_errors_zero() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
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

    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1);
    let c = dev.counters();
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        0,
        "successful add_used must NOT bump io_errors",
    );
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 1);
}

/// RO-flush through full chain. RO disks accept T_FLUSH
/// and return S_OK. flushes_completed increments.
#[test]
fn ro_flush_full_chain_returns_ok_increments_counter() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::with_options(f, cap, DiskThrottle::default(), true);
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_OK as u8);

    let c = dev.counters();
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
}

/// Multi-byte status descriptor. Status byte goes at
/// the LAST byte of the descriptor (`addr + len - 1`) so the
/// kernel driver's `virtio_blk_outhdr` lookup lines up
/// regardless of leading padding. The status_addr arithmetic
/// in drain_bracket_impl's chain-shape walk implements this;
/// pin the offset.
#[test]
fn multi_byte_status_writes_to_last_byte() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Plant a 4-byte sentinel at status_addr.
    mem.write_slice(&[0xAA, 0xBB, 0xCC, 0xDD], status_addr)
        .unwrap();
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
            4, // multi-byte status
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock(&mut dev, &mock);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let mut buf = [0u8; 4];
    mem.read_slice(&mut buf, status_addr).unwrap();
    assert_eq!(
        buf[0], 0xAA,
        "first byte of multi-byte status must be untouched"
    );
    assert_eq!(buf[1], 0xBB);
    assert_eq!(buf[2], 0xCC);
    assert_eq!(
        buf[3], VIRTIO_BLK_S_OK as u8,
        "status byte must be at the LAST byte (offset len-1)",
    );
}

/// Zero-capacity device. Any read with non-zero data
/// length must IOERR (`base_offset + total_data > 0`).
#[test]
fn zero_capacity_read_returns_ioerr() {
    let cap = 0u64;
    let f = tempfile().unwrap();
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);

    let c = dev.counters();
    assert!(c.io_errors.load(Ordering::Relaxed) >= 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
}

/// Partial-data + zero-pad read. Backing file 100 bytes,
/// device capacity 8 sectors, read 1 sector starting at sector
/// 0. Bytes 0..100 = file pattern, bytes 100..512 = zero-pad.
#[test]
fn partial_data_read_zero_pads_remainder() {
    let cap = 4096u64;
    let mut f = tempfile().unwrap();
    f.set_len(100).unwrap();
    f.write_all(&[0xA5; 100]).unwrap();
    f.rewind().unwrap();
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Pre-fill data buffer with sentinel — must not survive.
    let pre = vec![0xFFu8; 512];
    mem.write_slice(&pre, data_addr).unwrap();
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_OK as u8);

    let mut buf = [0u8; 512];
    mem.read_slice(&mut buf, data_addr).unwrap();
    assert!(
        buf[..100].iter().all(|&b| b == 0xA5),
        "first 100 bytes must match backing file pattern",
    );
    assert!(
        buf[100..].iter().all(|&b| b == 0),
        "bytes 100..512 must be zero-padded",
    );
}

/// sector=u64::MAX overflow. `checked_mul` catches and
/// rejects with IOERR. Without the check, the wraparound
/// would silently land at a low offset.
#[test]
fn write_sector_overflow_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_OUT, u64::MAX);
    let payload = vec![0xCDu8; 512];
    mem.write_slice(&payload, data_addr).unwrap();
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

    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_IOERR as u8);

    let c = dev.counters();
    assert!(c.io_errors.load(Ordering::Relaxed) >= 1);
    assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
}

/// Flush sync_data baseline. Forcing a real sync_data
/// failure from a passing test fixture requires a closed fd
/// or fault injection (libfiu, fioz). Best-effort coverage:
/// pin the OK path increments flushes_completed and not
/// io_errors. The Err arm is reviewed by code inspection
/// (handle_flush_impl writes VIRTIO_BLK_S_IOERR on Err).
#[test]
fn flush_sync_data_baseline_ok_path() {
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

    let c = dev.counters();
    assert_eq!(c.flushes_completed.load(Ordering::Relaxed), 1);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
}

/// Validation-before-consumption invariant. Drain the
/// throttle to 0, submit a sub-sector chain. Pin: io_errors
/// ticks (gate fires) but throttled_count stays 0 (gate is
/// pre-throttle; tokens NOT consumed).
#[test]
fn validation_gates_do_not_consume_throttle_tokens() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);
    // Drain the bucket and pin its last_refill so refill on
    // the next consume yields 0 tokens.
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());
    assert!(dev.worker.state_mut().ops_bucket.consume(1));
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(std::time::Instant::now());

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
            513, // sub-sector → alignment gate fires
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

    let c = dev.counters();
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        1,
        "sub-sector gate must bump io_errors",
    );
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        0,
        "validation gate must NOT consume throttle tokens; \
             throttled_count must stay at 0 even with bucket drained",
    );
}

/// RO-flush vs normal-flush counter symmetry. Both
/// paths must increment flushes_completed by exactly 1.
#[test]
fn ro_flush_and_normal_flush_both_increment_counter() {
    // Normal flush.
    {
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
        assert_eq!(
            dev.counters().flushes_completed.load(Ordering::Relaxed),
            1,
            "normal flush must increment flushes_completed",
        );
    }
    // RO flush.
    {
        let cap = 4096u64;
        let f = make_backed_file_with_pattern(cap, 0x00);
        let mut dev = VirtioBlk::with_options(f, cap, DiskThrottle::default(), true);
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
        assert_eq!(
            dev.counters().flushes_completed.load(Ordering::Relaxed),
            1,
            "RO flush must increment flushes_completed (counter symmetry)",
        );
    }
}
