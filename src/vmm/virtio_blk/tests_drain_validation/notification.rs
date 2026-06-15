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

/// Legacy-path irqfd delivery through a full chain. Before
/// process_requests, irq_evt is unsignalled (read returns
/// EAGAIN). After QUEUE_NOTIFY drains the chain, the post-drain
/// V8-split logic inlined in `process_requests`
/// (interrupt_status bit + needs_notification-gated eventfd
/// write) MUST leave irq_evt readable on the legacy path
/// because `Queue::needs_notification` returns Ok(true)
/// unconditionally when EVENT_IDX is not negotiated. This
/// pins the KVM irqfd delivery contract.
#[test]
fn process_requests_fires_irqfd_on_legacy_path() {
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
    // Before notify: irq_evt must NOT be readable.
    assert!(
        dev.irq_evt.read().is_err(),
        "before process_requests, irq_evt must not be signalled",
    );
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must be readable after notify");
    // Production fires `irq_evt.write(1)` exactly once per
    // signalling drain (process_requests post-drain branch).
    // A loose `> 0` would let a regression that fired the
    // eventfd twice slip through; pin the exact count.
    assert_eq!(
        val, 1,
        "irq_evt counter must be exactly 1 after a single chain drain"
    );
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0
    );
}

/// EVENT_IDX path: when the guest's `used_event` threshold has
/// not been crossed by `next_used`, the device must NOT write
/// the irqfd, even though it advanced the used ring.
/// `Queue::needs_notification` returns false in that window —
/// its `event_idx_enabled` arm runs the
/// `used_idx - used_event - 1 < used_idx - old`
/// wrapping-arithmetic test, which is false when `used_event`
/// is well above `next_used`.
/// The `interrupt_status` bit must still be set so the guest's
/// MMIO read sees pending work — the V8 split between bit and
/// eventfd lets the guest poll without losing context if it
/// happens to read INTERRUPT_STATUS while suppressed.
#[test]
fn event_idx_suppresses_irqfd_when_threshold_unreached() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // Plant `used_event = u16::MAX` BEFORE wiring the device:
    // the guest writes this before the first QUEUE_NOTIFY in
    // real life, and `Queue::needs_notification` reads it
    // every time it's called.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(u16::MAX), used_event)
        .expect("plant used_event");
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
    // used_override: place the used ring at 0x10000, well above
    // the avail ring's used_event field at avail_addr + 36. The
    // mock's default used_addr collides with used_event; see
    // `wire_device_to_mock_with_event_idx` doc comment.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // The chain landed: status byte and counter ticked.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(s[0], VIRTIO_BLK_S_OK as u8);
    assert_eq!(dev.counters().reads_completed.load(Ordering::Relaxed), 1,);
    // V8: interrupt_status bit IS set even when irqfd is
    // suppressed. The guest reads INTERRUPT_STATUS during its
    // ISR (or polling); seeing the bit lets it know there's
    // work even if no IRQ delivered.
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be set when chain published",
    );
    // Same bit observable through the MMIO surface
    // (`read_reg` → `mmio_read` → `interrupt_status` value at
    // VIRTIO_MMIO_INTERRUPT_STATUS). VIRTIO_MMIO_INT_VRING is
    // bit 1 (vring buffer interrupt). Pins that the guest's
    // ISR/polling path sees the bit through the published
    // MMIO contract, not just via the host-internal field.
    let status = read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS);
    assert_eq!(status & 1, 1);
    // irqfd MUST be unsignalled — read returns EAGAIN
    // (counter is 0, eventfd in counter mode blocks/EAGAINs
    // on read of zero-value).
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must be unsignalled when used_event threshold not crossed",
    );
}

/// EVENT_IDX path: when the guest's `used_event` threshold IS
/// crossed (e.g. used_event = 0 and we publish a chain causing
/// next_used = 1), the device fires the irqfd. This is the
/// common case for the first request after the guest sets up
/// the queue.
#[test]
fn event_idx_fires_irqfd_when_threshold_reached() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // used_event = 0: the guest is asking to be notified as
    // soon as next_used reaches 1. After one chain
    // completion, `needs_notification` returns true.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(0), used_event)
        .expect("plant used_event");
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
    // used_override: place the used ring at 0x10000, well above
    // the avail ring's used_event field at avail_addr + 36. The
    // mock's default used_addr collides with used_event; see
    // `wire_device_to_mock_with_event_idx` doc comment.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // irqfd fired exactly once (counter mode: a single write(1)
    // produces read returning 1).
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must be readable when threshold reached");
    assert_eq!(
        val, 1,
        "irq_evt counter must be exactly 1 after a single chain completion",
    );
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0
    );
}

/// EVENT_IDX path: a multi-chain drain consults
/// `needs_notification` exactly once at the END of the drain
/// (V6: only call on the signal_needed=true path), so the
/// irqfd fires at most ONCE regardless of chain count. This
/// is the IRQ-coalescing benefit of EVENT_IDX — without it
/// the legacy path would fire once per drain anyway, but
/// with EVENT_IDX the fire decision is held until the drain
/// completes so `needs_notification` sees the final
/// `next_used` value (`num_added` reflects all 3 chains).
#[test]
fn event_idx_multi_chain_drain_fires_once() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // used_event = 0: notify when next_used reaches 1.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(0), used_event)
        .expect("plant used_event");
    // Build 3 chains, each its own header/data/status triple.
    for i in 0..3u64 {
        let header_addr = GuestAddress(0x4000 + i * 0x1000);
        let data_addr = GuestAddress(0x8000 + i * 0x1000);
        let status_addr = GuestAddress(0xC000 + i * 0x100);
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
    }
    dev.set_mem(mem.clone());
    // used_override: place the used ring at 0x10000, well above
    // the avail ring's used_event field at avail_addr + 36. The
    // mock's default used_addr collides with used_event; see
    // `wire_device_to_mock_with_event_idx` doc comment.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // 3 chains completed.
    assert_eq!(dev.counters().reads_completed.load(Ordering::Relaxed), 3,);
    // irqfd fired exactly once. EventFd in counter mode: one
    // write(1) → read returns 1; three writes → read returns
    // 3. The post-drain gate produces a single write, so
    // read must return 1.
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must be readable after multi-chain drain");
    assert_eq!(
        val, 1,
        "irq_evt must fire exactly once for a multi-chain drain \
             (V6: needs_notification consulted once at end of drain)",
    );
}

/// EVENT_IDX path, multi-chain drain, threshold above the
/// post-drain `next_used` value: 3 chains complete but
/// `needs_notification` returns false because `used_event = 10`
/// (the guest is asking to be notified only once `next_used`
/// crosses 10). Pins suppression under multi-chain load — a
/// regression that fired the irqfd once per chain (or once per
/// drain regardless of threshold) would surface as a non-zero
/// `irq_evt.read()` here. Companion to
/// `event_idx_multi_chain_drain_fires_once` (used_event=0,
/// expected fire) — together the pair pin both halves of the
/// gate at multi-chain load.
#[test]
fn event_idx_multi_chain_drain_suppresses_below_threshold() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // used_event = 10: the guest is asking for notification only
    // once next_used crosses 10. We're going to drain 3 chains
    // (next_used → 3) so the threshold is unreached and the
    // post-drain `needs_notification` returns false. Plant
    // BEFORE wiring the device per the existing EVENT_IDX
    // pattern (Queue reads used_event lazily on each
    // needs_notification call).
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(10), used_event)
        .expect("plant used_event");
    // Build 3 read chains with disjoint addresses so the
    // descriptor table doesn't alias across iterations.
    for i in 0..3u64 {
        let header_addr = GuestAddress(0x4000 + i * 0x1000);
        let data_addr = GuestAddress(0x8000 + i * 0x1000);
        let status_addr = GuestAddress(0xC000 + i * 0x100);
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
    }
    dev.set_mem(mem.clone());
    // used_override: place the used ring at 0x10000, well above
    // the avail ring's used_event field at avail_addr + 36. The
    // mock's default used_addr collides with used_event; see
    // `wire_device_to_mock_with_event_idx` doc comment.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // All 3 chains landed.
    assert_eq!(
        dev.counters().reads_completed.load(Ordering::Relaxed),
        3,
        "all 3 chains must complete in the single QUEUE_NOTIFY drain",
    );
    // Used ring (at the override addr 0x10000) reflects exactly
    // 3 completions. Reads u16 used.idx at offset 2 of the
    // override addr.
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(
        used_idx, 3,
        "exactly three used-ring entries expected after 3-chain drain",
    );
    // V8: interrupt_status bit IS set even when irqfd is
    // suppressed. The guest's ISR or polling path reads
    // INTERRUPT_STATUS to find work; the bit must be visible
    // independent of the irqfd gate.
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be set after 3 completions \
             even when irqfd suppressed",
    );
    // Same bit observable through the MMIO surface.
    let status = read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS);
    assert_eq!(status & 1, 1);
    // irqfd MUST be unsignalled — `needs_notification` saw
    // next_used=3 < used_event=10 so the gate held.
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must be unsignalled when post-drain next_used \
             stays below used_event threshold",
    );
}

/// Legacy path (EVENT_IDX not negotiated):
/// `Queue::needs_notification` always returns Ok(true) (the
/// trailing `Ok(true)` after the `event_idx_enabled` branch),
/// so every drain that publishes any chain fires the irqfd.
/// This test pins the legacy contract — a regression that
/// gated the irqfd write on the wrong path would silently
/// break the legacy guest's IRQ delivery.
#[test]
fn legacy_path_fires_irqfd_every_drain() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // Plant used_event = u16::MAX. In the EVENT_IDX path this
    // would suppress; in the legacy path it's IGNORED — proves
    // the test exercises the legacy path.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(u16::MAX), used_event)
        .expect("plant used_event");
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
    // Legacy path: VIRTIO_RING_F_EVENT_IDX NOT negotiated.
    wire_device_to_mock(&mut dev, &mock);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // irqfd fired despite used_event=u16::MAX, because the
    // legacy path ignores the threshold.
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must be readable on legacy path");
    assert_eq!(
        val, 1,
        "legacy path must fire irq_evt unconditionally — used_event \
             is irrelevant when EVENT_IDX is not negotiated",
    );
    // Symmetry with EVENT_IDX suppressed-irqfd tests: pin that
    // INTERRUPT_STATUS reflects the bit on the legacy fire path
    // too. Production sets the bit unconditionally on any
    // signalling drain (process_requests post-drain branch),
    // independent of which path drives the irqfd gate.
    let status = read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS);
    assert_eq!(status & 1, 1);
}

/// Outer-loop bracket: when 2 chains are queued before
/// QUEUE_NOTIFY, both complete in a single `process_requests`
/// call. This is a deterministic variant of the re-drain
/// coverage — see the doc note below for why the
/// `enable_notification → Ok(true) → continue 'outer` arm
/// itself can't be tested deterministically from a single
/// thread.
///
/// Race-window note: the production re-drain arm fires when
/// `avail_idx != next_avail` AT the moment `enable_notification`
/// runs. In a real VMM, that gap exists between the inner-loop
/// break (next_avail caught up to the avail_idx the device saw)
/// and the `set_avail_event` call inside `enable_notification`
/// — a vCPU can write a fresh chain head and bump `avail_idx`
/// in that window. In a single-threaded test there is no such
/// vCPU; `MockSplitQueue` is the only writer and we control
/// when it writes. To trigger Ok(true) deterministically would
/// require interposing on `enable_notification` itself
/// (e.g. a test-only `Queue` implementation) — too invasive
/// for the value gained. The deterministic variant here pins
/// the WEAKER property: 2 chains queued before notify both
/// complete in one process_requests call. The actual re-drain
/// arm is exercised by the existing
/// `event_idx_multi_chain_drain_fires_once` test which queues
/// 3 chains; both tests share the same single-process-requests
/// shape.
///
/// What this DOES guarantee: a 2-chain pre-notify queue drains
/// fully in one call. A regression that prematurely broke out
/// of the outer loop after the first chain (e.g. dropping
/// `continue 'outer` in favour of `break 'outer`) would leave
/// the second chain unprocessed — that regression IS caught
/// here even though the path through the Ok(true) arm itself
/// isn't directly observed.
#[test]
fn outer_loop_drains_two_pre_queued_chains_in_one_call() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // used_event = 0: notify on first completion. After both
    // chains are processed the post-drain `needs_notification`
    // observes next_used=2, num_added=2, threshold-crossed →
    // fires. Either Ok(true)→Ok(false) (re-drain path) OR
    // Ok(false) directly leaves both chains processed and
    // signal_needed=true.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(0), used_event)
        .expect("plant used_event");
    let header_addr = GuestAddress(0x4000);
    let status_addr = GuestAddress(0x4100);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_FLUSH, 0);
    // Two FLUSH chains pre-queued. FLUSH carries no data
    // (header + status only — virtio-v1.2 §5.2.6.3). Both
    // chains share the same desc_table slots because
    // `MockSplitQueue::add_desc_chains` writes at offset 0
    // each call; the second build_desc_chain overwrites
    // descriptors 0..1 but the avail_ring grows by one each
    // call — so 2 chain heads point at desc_table[0] and the
    // device walks the same descriptors twice. fdatasync on a
    // tempfile is idempotent.
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
    mock.build_desc_chain(&descs).expect("build chain 1");
    mock.build_desc_chain(&descs).expect("build chain 2");
    dev.set_mem(mem.clone());
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Both chains completed. The bracket pattern guarantees
    // EITHER (a) inner loop drains both immediately and
    // enable_notification returns Ok(false) → break, OR (b)
    // inner drains chain 1, enable_notification returns Ok(true)
    // because chain 2's avail-idx advance was visible after the
    // bracket close → continue 'outer drains chain 2. Both end
    // states are observable as flushes_completed == 2.
    let c = dev.counters();
    assert_eq!(
        c.flushes_completed.load(Ordering::Relaxed),
        2,
        "both pre-queued FLUSH chains must complete in a single \
             process_requests call",
    );
    // Used ring (placed at the override addr 0x10000) reflects
    // exactly two completions. The mock's default used ring is
    // unused; read used.idx from the override location.
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(
        used_idx, 2,
        "exactly two used-ring entries expected after two-chain drain",
    );
    // Single irqfd fire: V6 has needs_notification consulted
    // once at end of drain. Whether the path went through the
    // re-drain arm or broke out directly, the tail signal is
    // ONE eventfd write.
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt readable after two-chain drain");
    assert_eq!(
        val, 1,
        "exactly one irq_evt write expected — needs_notification \
             consulted once after the drain settles",
    );
}

/// Bail-out branch: when `enable_notification` returns Err
/// (the `set_avail_event` write to the used ring's
/// `avail_event` field hits unmapped guest memory), the outer
/// loop must break cleanly without hanging, the chain that
/// was already published before the failure stays published
/// (`add_used` succeeded; the failure is in the post-drain
/// notification arming), and the irqfd fires fail-safe via
/// the `unwrap_or(true)` on the post-drain `needs_notification`
/// call.
///
/// Test setup: a multi-region GuestMemoryMmap with a hole
/// straddling the device's `avail_event` GPA. The used ring is
/// placed via `used_override_addr` so its body
/// (header + ring elements at offsets 0..132) lives in the
/// first region and the trailing `avail_event` u16 at
/// `used_addr + 132` lands at the boundary, in the unmapped
/// gap. add_used (offsets 4..12 for index 0) succeeds;
/// `set_avail_event` writing 2 bytes at `used_addr + 132`
/// fails with InvalidGuestAddress.
///
/// Layout: `Queue::set_avail_event` writes at
/// `used_ring + VIRTQ_USED_RING_HEADER_SIZE +
/// VIRTQ_USED_ELEMENT_SIZE * size = used_ring + 4 + 8 * 16 =
/// used_ring + 132`.
#[test]
fn enable_notification_err_breaks_outer_and_fires_irqfd_fail_safe() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    // Multi-region mem: [0, 0x20000) and [0x30000, 0x40000).
    // The hole is [0x20000, 0x30000). With used_addr=0x1FF7C
    // and size=16: avail_event is at 0x20000 (start of the
    // hole), inaccessible. add_used at next_used=0 writes 8
    // bytes to 0x1FF80..0x1FF88 (in-range) plus the 2-byte
    // next_used u16 to 0x1FF7E (in-range).
    let mem = GuestMemoryMmap::from_ranges(&[
        (GuestAddress(0), 0x20000),
        (GuestAddress(0x30000), 0x10000),
    ])
    .expect("create multi-region guest mem");
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
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
    // used_override = 0x1FF7C: with size=16 the used-ring body
    // (header + 16 * 8-byte elements = 132 bytes) ends exactly
    // at 0x20000 (the boundary), and the trailing avail_event
    // u16 store at 0x20000..0x20002 lies in the unmapped hole.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x1FF7C));

    // Pre-notify: irqfd MUST be unsignalled.
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must not be signalled before notify",
    );

    // Fire QUEUE_NOTIFY. Inner drain processes the chain
    // (add_used succeeds at offsets in the mapped region),
    // enable_notification returns Err on the unmapped
    // avail_event store, the outer loop breaks cleanly. If
    // the bail were missing (infinite outer loop on persistent
    // err), this call would hang and the test would time out.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Chain was published before the bail. flushes_completed
    // ticked, used.idx advanced to 1.
    let c = dev.counters();
    assert_eq!(
        c.flushes_completed.load(Ordering::Relaxed),
        1,
        "FLUSH must complete before the enable_notification bail",
    );
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x1FF7C).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(
        used_idx, 1,
        "add_used must have run before the enable_notification bail",
    );

    // V8 + fail-safe: the irqfd MUST fire. The post-drain
    // `needs_notification` reads `used_event` from the avail
    // ring (in the mapped region — only the USED ring's
    // `avail_event` is in the hole), so the call returns
    // Ok(true|false) cleanly. With used_event=0 (default mock
    // initialisation, mock.rs:151) and next_used=1, the
    // formula returns true → fire. Even if it returned an
    // Err, `unwrap_or(true)` would still fire fail-safe.
    let val = dev
        .irq_evt
        .read()
        .expect("irq_evt must fire fail-safe after enable_notification bail");
    assert_eq!(
        val, 1,
        "irq_evt must fire exactly once after the bail (V8 \
             interrupt_status bit + needs_notification gate)",
    );
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be set when chain published, \
             independent of the enable_notification bail",
    );
}

/// Companion to `enable_notification_err_breaks_outer_and_fires_irqfd_fail_safe`:
/// pins the OTHER `enable_notification` call site, the
/// post-stall arm. When the chain stalls on throttle exhaustion,
/// the inner pop loop breaks WITHOUT publishing
/// (`signal_needed` stays false), the outer-loop stall arm calls
/// `enable_notification` to re-arm guest-side wakeups, and on
/// Err logs a warn and breaks 'outer cleanly. Distinct from the
/// Done-path enable_notification (covered above) because the
/// stall path skips the post-drain `signal_needed` block — no
/// `interrupt_status` bit set, no irqfd write, no `add_used`.
///
/// Setup mirrors the Done-path test for guest-memory layout —
/// multi-region GuestMemoryMmap with a hole at the
/// `avail_event` GPA, used ring placed via `used_override_addr`
/// so its trailing 2-byte `avail_event` write lands in the
/// unmapped hole — but adds a drained 1-iops throttle so the
/// chain stalls instead of completing. The single chain pops,
/// the throttle gate fails, the stall path calls
/// `enable_notification` whose `set_avail_event` write hits
/// the hole and returns InvalidGuestAddress.
///
/// Stall-path invariants:
///
///   - `throttled_count` == 1 — the stall event was recorded.
///   - `currently_throttled_gauge` == 1 — the false→true
///     transition fired (per the gauge transition table).
///   - `state.currently_stalled` == true — the head is pinned
///     in the avail ring awaiting refill.
///   - used.idx == 0 (no add_used).
///   - irq_evt unsignalled — `signal_needed` stayed false, so
///     the post-drain V8 block was not entered.
///   - interrupt_status MMIO bit clear (same reason).
///   - status sentinel survives — no publish_completion ran.
///   - Queue cursor rewound to 0 (set_next_avail rolled the
///     pop back so the chain re-pops on retry).
///
/// A regression that propagated the enable_notification Err
/// instead of swallowing-and-breaking would either re-enter the
/// outer loop (livelock) or fail to record the stall counter —
/// both observable via the assertions below.
#[test]
fn enable_notification_err_on_stall_path_breaks_outer_cleanly() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: std::num::NonZeroU64::new(1),
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);

    // Multi-region mem: [0, 0x20000) and [0x30000, 0x40000).
    // The hole [0x20000, 0x30000) covers the avail_event GPA.
    // Same layout as the Done-path test so the
    // `avail_event = used_addr + 132` (size=16) calculation
    // produces 0x20000 — the boundary, in the hole.
    let mem = GuestMemoryMmap::from_ranges(&[
        (GuestAddress(0), 0x20000),
        (GuestAddress(0x30000), 0x10000),
    ])
    .expect("create multi-region guest mem");
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);

    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    // Plant a sentinel at the status byte — survival of this
    // byte through the stall is the key invariant (no
    // publish_completion ran).
    mem.write_slice(&[0xEEu8], status_addr).unwrap();
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
    // used_override = 0x1FF7C: with size=16, the used-ring body
    // (4-byte header + 16 * 8-byte elements = 132 bytes) ends
    // at exactly 0x20000, and the trailing avail_event u16
    // store at 0x20000..0x20002 lies in the unmapped hole.
    // Same address as the Done-path test by design.
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x1FF7C));

    // Drain the iops bucket so the chain stalls. With iops=1
    // and capacity=1, a single consume(1) takes the only token;
    // pin last_refill so the next can_consume sees an empty
    // bucket (no passive wall-clock refill in microseconds
    // between this setup and the QUEUE_NOTIFY).
    let now = std::time::Instant::now();
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(now);
    assert!(dev.worker.state_mut().ops_bucket.consume(1));
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(now);
    assert!(
        !dev.worker.state_mut().ops_bucket.can_consume(1),
        "precondition: ops bucket must be drained so the chain stalls",
    );

    // Pre-notify: every observable surface is at its baseline.
    let c = dev.counters();
    assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
    assert_eq!(c.currently_throttled_gauge.load(Ordering::Relaxed), 0);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    assert!(!dev.worker.state().currently_stalled);
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must be unsignalled before notify",
    );

    // Fire QUEUE_NOTIFY. Inner pop returns the chain, throttle
    // gate fails, stall_outcome = Some(_), break inner. Outer
    // stall arm calls enable_notification → Err on the unmapped
    // avail_event store → log warn, break 'outer. No
    // publish_completion ran; signal_needed stayed false; the
    // post-drain V8 block did not fire. If the bail were
    // missing (continued outer loop on persistent Err), this
    // call would hang.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Status sentinel survives — no publish_completion ran.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], 0xEE,
        "status byte must remain at sentinel — stall must not write status",
    );

    // used.idx unchanged at 0 — no add_used.
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x1FF7C).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(used_idx, 0, "used.idx must be 0 — stall must skip add_used",);

    // Stall counters: event recorded, gauge incremented on
    // false→true, reads_completed untouched.
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        1,
        "stall event must be recorded once",
    );
    assert_eq!(
        c.currently_throttled_gauge.load(Ordering::Relaxed),
        1,
        "gauge must increment on the false→true transition",
    );
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 0);
    assert!(
        dev.worker.state().currently_stalled,
        "currently_stalled flag must be true post-stall",
    );

    // V8 post-drain block did not run — signal_needed stayed
    // false. interrupt_status bit clear, irqfd unsignalled.
    assert_eq!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be clear — stall does not \
             enter the V8 post-drain block",
    );
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must be unsignalled — stall does not fire irqfd",
    );

    // Queue cursor rewound: stall path runs
    // `set_next_avail(prev.wrapping_sub(1))` so the next pop
    // returns the same head. After one pop+rewind on a queue
    // with one chain, next_avail is back at 0.
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        0,
        "queue cursor must be rewound to 0 — set_next_avail \
             rolled the pop back so the chain re-pops on retry",
    );
}

/// Pin the u16-wrap arithmetic the throttle-stall rollback
/// depends on: `set_next_avail(prev.wrapping_sub(1))` at
/// `prev = 0` MUST land at `u16::MAX`, not panic via signed
/// underflow. The companion proptest
/// `throttle_stall_under_random_chain_shapes_holds_invariants`
/// runs at `next_avail = 0 → 1 → 0` (no wrap exercised
/// because `MockSplitQueue::build_desc_chain` only supports
/// fresh avail rings); this dedicated unit test pins the
/// wrap edge by directly calling `set_next_avail` on the
/// queue and asserting the contract `wrapping_sub` provides.
///
/// A regression that swapped `wrapping_sub` for plain `-` or
/// `checked_sub().unwrap()` would panic on this test
/// instead of silently corrupting the cursor in production.
/// `wrapping_sub` matches the virtio ring's u16 wrap
/// semantics (avail/used cursors are u16 modular per
/// virtio-v1.2 §2.7).
#[test]
fn next_avail_zero_rollback_wraps_to_u16_max() {
    let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
    // Drive next_avail to 0 explicitly so the sub-1 wrap is
    // observable via wrapping arithmetic (not a "happened to
    // be at 0" coincidence).
    dev.worker.queues[REQ_QUEUE].set_next_avail(0);
    assert_eq!(dev.worker.queues[REQ_QUEUE].next_avail(), 0);

    // The exact arithmetic the production stall path uses:
    // `set_next_avail(prev.wrapping_sub(1))`.
    let prev = dev.worker.queues[REQ_QUEUE].next_avail();
    dev.worker.queues[REQ_QUEUE].set_next_avail(prev.wrapping_sub(1));

    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        u16::MAX,
        "next_avail rollback at prev=0 must wrap to u16::MAX, \
             matching the virtio ring's u16 modular semantics",
    );

    // Wrap-back: another wrapping_sub from u16::MAX lands at
    // u16::MAX - 1, no panic. Pins the arithmetic in both
    // directions so a regression that handled the 0→u16::MAX
    // case but broke u16::MAX→u16::MAX-1 surfaces here.
    let prev = dev.worker.queues[REQ_QUEUE].next_avail();
    dev.worker.queues[REQ_QUEUE].set_next_avail(prev.wrapping_sub(1));
    assert_eq!(
        dev.worker.queues[REQ_QUEUE].next_avail(),
        u16::MAX - 1,
        "subsequent rollback at prev=u16::MAX must land at u16::MAX-1",
    );
}

/// Fragmented header. The first descriptor is shorter
/// than VIRTIO_BLK_OUTHDR_SIZE — the device cannot read a
/// full header from desc[0] and must reject. Chain layout:
/// [8-byte-RO, 8-byte-RO, status] — the second descriptor's
/// 8 bytes do NOT count toward the header (per virtio_blk.rs's
/// "first_len < OUTHDR_SIZE" gate).
#[test]
fn fragmented_header_returns_ioerr() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_part1_addr = GuestAddress(0x4000);
    let header_part2_addr = GuestAddress(0x4008);
    let status_addr = GuestAddress(0x5000);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_part1_addr.0,
            8, // SHORT
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(header_part2_addr.0, 8, 0, 0)),
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
        "fragmented header (first desc < OUTHDR_SIZE) must IOERR",
    );

    let c = dev.counters();
    assert_eq!(c.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
}

/// EVENT_IDX path with an error chain: the IOERR completion must
/// route through the SAME post-drain `needs_notification` gate
/// as success completions, so a guest that asks for suppression
/// (`used_event = u16::MAX`) does not get spuriously interrupted
/// by an error chain.
///
/// Setup mirrors `fragmented_header_returns_ioerr` (chain has a
/// short first descriptor of 8 bytes — less than
/// `VIRTIO_BLK_OUTHDR_SIZE` = 16 — so the device cannot read a
/// full header from desc[0] and rejects via
/// `publish_completion(..., VIRTIO_BLK_S_IOERR, ...)` at
/// `process_requests`'s "header missing/short" branch). The
/// publish_completion call returns true (status-byte write
/// succeeded, add_used succeeded), so `signal_needed = true` —
/// the chain reaches the post-drain notification arm.
///
/// With EVENT_IDX negotiated and `used_event = u16::MAX`, the
/// post-drain `needs_notification` returns false (next_used=1
/// nowhere near u16::MAX) so the irqfd MUST stay unsignalled.
/// `interrupt_status` is still set (the guest's ISR/polling
/// path needs to see there's work). Pins the contract that
/// error completions are NOT a special-case bypass of the
/// suppression gate.
#[test]
fn event_idx_error_chain_suppressed_when_threshold_unreached() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0x00);
    let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
    let mem = make_chain_test_mem();
    let qsize = 16u16;
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), qsize);
    // Plant `used_event = u16::MAX` BEFORE wiring the device:
    // the guest is asking to never be notified for any normal
    // post-drain `next_used` value (it polls instead). The
    // post-drain `needs_notification` reads `used_event`
    // lazily, so plant before notify, not before wire.
    let used_event = used_event_addr(mock.avail_addr(), qsize);
    mem.write_obj::<u16>(u16::to_le(u16::MAX), used_event)
        .expect("plant used_event");
    // Fragmented-header layout: desc[0] = 8 bytes (< OUTHDR_SIZE
    // = 16) → header rejected, IOERR published via
    // publish_completion. desc[1] is also 8 bytes RO so the
    // device cannot stitch a full header from desc[0]+desc[1]
    // (per the "first_len < OUTHDR_SIZE" gate). desc[2] is the
    // 1-byte writable status descriptor.
    let header_part1_addr = GuestAddress(0x4000);
    let header_part2_addr = GuestAddress(0x4008);
    let status_addr = GuestAddress(0x5000);
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            header_part1_addr.0,
            8, // SHORT — triggers IOERR via publish_completion
            0,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(header_part2_addr.0, 8, 0, 0)),
        RawDescriptor::from(SplitDescriptor::new(
            status_addr.0,
            1,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_device_to_mock_with_event_idx(&mut dev, &mock, qsize, GuestAddress(0x10000));
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // The error chain landed: status=IOERR, io_errors=1.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], VIRTIO_BLK_S_IOERR as u8,
        "fragmented header must produce IOERR even on EVENT_IDX path",
    );
    let c = dev.counters();
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        1,
        "fragmented-header reject must bump io_errors exactly once",
    );
    // The chain WAS add_used'd: error completions reach the
    // post-drain gate via signal_needed=true. used.idx at the
    // override addr advances to 1.
    let used_idx: u16 = mem
        .read_obj(GuestAddress(0x10000).checked_add(2).unwrap())
        .expect("read device used.idx at override addr");
    assert_eq!(
        used_idx, 1,
        "error chain must still be add_used'd so the guest sees \
             the IOERR status — V8 + the publish_completion contract",
    );
    // V8: interrupt_status bit IS set on the error chain too —
    // the guest's polling path reads INTERRUPT_STATUS to learn
    // there's work, regardless of irqfd suppression.
    assert_ne!(
        dev.interrupt_status.load(Ordering::Acquire) & VIRTIO_MMIO_INT_VRING,
        0,
        "interrupt_status bit must be set after error chain \
             completes, independent of irqfd gate",
    );
    // Same bit observable through the MMIO surface.
    let status = read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS);
    assert_eq!(status & 1, 1);
    // The contract this test pins: irqfd suppressed for the
    // error chain because `used_event=u16::MAX` was unreached.
    // A regression that bypassed `needs_notification` for
    // error completions (e.g. firing the irqfd unconditionally
    // on signal_needed=true) would surface here.
    assert!(
        dev.irq_evt.read().is_err(),
        "irq_evt must be unsignalled — error completions route \
             through the same needs_notification gate as success \
             completions, and used_event=u16::MAX was unreached",
    );
}

/// SIZE_MAX advertised in config space. virtio-v1.2
/// §5.2.4: size_max field at config-space offset 0x08
/// (= MMIO offset 0x108) must hold the per-descriptor max
/// byte length. Without the correct value, the guest
/// driver may submit oversize descriptors.
#[test]
fn size_max_advertised_in_config_space() {
    let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
    let mut buf = [0u8; 4];
    dev.mmio_read(0x100 + 0x08, &mut buf);
    assert_eq!(
        u32::from_le_bytes(buf),
        VIRTIO_BLK_SIZE_MAX,
        "config-space size_max must equal VIRTIO_BLK_SIZE_MAX (1 MiB)",
    );
}

/// Validation gate ordering: a chain that would pass the throttle
/// but fails validation must NOT consume throttle tokens.
///
/// Sets up a finite throttle (`iops=1000`, `bytes_per_sec=1_000_000`)
/// stocked at full capacity, then submits a sub-sector T_IN chain
/// (data_len=513). The sector-alignment validation gate in
/// `drain_bracket_impl` rejects it at the
/// `data_len.is_multiple_of(VIRTIO_BLK_SECTOR_SIZE)` arm — BEFORE the
/// `pre_throttle.is_none()` guarded `ops_bucket.can_consume(1)` /
/// `bytes_bucket.can_consume(data_len)` call site.
///
/// Pre-conditions imply the request would have passed throttle if it
/// had reached the bucket: `1 <= 1000` ops capacity, `513 <=
/// 1_000_000` bytes capacity, and both buckets are seeded at
/// capacity by `TokenBucket::new`.
///
/// Pins the gate-ordering invariant: the bucket's `available` field
/// stays at exactly `capacity` across the rejected chain — no token
/// consumption, no debt. Asserting `available` directly (not just
/// `can_consume(capacity)`) catches a regression that consumed a
/// token AND immediately let `refill()` restore it (the wall-clock
/// gap between the consume site and the assertion is well below 1ms,
/// at which `refill()` adds 0 tokens at rate=1000/sec; but pinning
/// `last_refill` and reading `available` directly removes any wall-
/// clock dependency).
///
/// Companion to `validation_precedes_throttle_on_stall` in
/// tests_drain.rs (drained-bucket case): together they pin the
/// gate-ordering contract from both sides — validation rejects on a
/// stocked bucket without consuming tokens (this test) and on a
/// drained bucket without producing a stall (the sibling).
#[test]
fn validation_gate_does_not_consume_throttle_tokens() {
    let cap = 4096u64;
    let f = make_backed_file_with_pattern(cap, 0xAB);
    let throttle = DiskThrottle {
        iops: NonZeroU64::new(1000),
        bytes_per_sec: NonZeroU64::new(1_000_000),
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    };
    let mut dev = VirtioBlk::new(f, cap, throttle);

    // Pin both buckets' last_refill so the in-place refill inside
    // `consume`/`can_consume` cannot grant or revoke tokens between
    // the pre-notify capture and the post-notify assertion.
    let now = Instant::now();
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(now);
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(now);

    // Capture the seeded balance — `TokenBucket::new` sets
    // `available = i64::try_from(capacity).unwrap_or(i64::MAX)`,
    // so this should equal the rate (capacity == rate when
    // *_burst_capacity is None — see buckets_from_throttle).
    let ops_avail_before = dev.worker.state_mut().ops_bucket.available;
    let bytes_avail_before = dev.worker.state_mut().bytes_bucket.available;
    assert_eq!(
        ops_avail_before, 1000,
        "ops bucket must be seeded at capacity (1000) on construction",
    );
    assert_eq!(
        bytes_avail_before, 1_000_000,
        "bytes bucket must be seeded at capacity (1_000_000) on construction",
    );

    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let header_addr = GuestAddress(0x4000);
    let data_addr = GuestAddress(0x5000);
    let status_addr = GuestAddress(0x6000);
    write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
    // Sub-sector data length: 513 bytes is not a multiple of
    // VIRTIO_BLK_SECTOR_SIZE (512). The sector-alignment gate in
    // drain_bracket_impl rejects this BEFORE pre_throttle.is_none()
    // path reaches `ops_bucket.can_consume(1)`. 513 fits well under
    // both bucket capacities (1 op, 513 bytes vs 1000/1_000_000
    // capacities), so a regression that swapped gate ordering and
    // consumed tokens before validating would succeed at the
    // throttle path — and the post-notify assertions below would
    // catch the consumption.
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

    // Re-pin after the FSM walk so MMIO writes' wall time does not
    // leak refill into the buckets between pre-notify and the
    // post-notify assertion.
    let now2 = Instant::now();
    dev.worker
        .state_mut()
        .ops_bucket
        .set_last_refill_for_test(now2);
    dev.worker
        .state_mut()
        .bytes_bucket
        .set_last_refill_for_test(now2);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

    // Validation rejected the chain — IOERR published, io_errors
    // bumped exactly once.
    let mut s = [0u8; 1];
    mem.read_slice(&mut s, status_addr).unwrap();
    assert_eq!(
        s[0], VIRTIO_BLK_S_IOERR as u8,
        "sub-sector chain must be rejected by the sector-alignment validation gate",
    );
    let c = dev.counters();
    assert_eq!(
        c.io_errors.load(Ordering::Relaxed),
        1,
        "validation gate bumps io_errors exactly once",
    );
    assert_eq!(
        c.reads_completed.load(Ordering::Relaxed),
        0,
        "rejected chain must not count as a completed read",
    );
    assert_eq!(
        c.throttled_count.load(Ordering::Relaxed),
        0,
        "validation rejection is not a throttle stall — \
             throttled_count must stay 0",
    );

    // The load-bearing assertion: throttle bucket balances are
    // unchanged. A regression that consumed tokens before the
    // validation gate would surface as `available < capacity`. With
    // last_refill pinned, no in-place refill can mask a consume
    // round-trip.
    assert_eq!(
        dev.worker.state_mut().ops_bucket.available,
        ops_avail_before,
        "ops bucket `available` must be unchanged across a \
             validation rejection — gate ordering requires the \
             validation check to fire BEFORE token consumption",
    );
    assert_eq!(
        dev.worker.state_mut().bytes_bucket.available,
        bytes_avail_before,
        "bytes bucket `available` must be unchanged across a \
             validation rejection — gate ordering requires the \
             validation check to fire BEFORE token consumption",
    );
}
