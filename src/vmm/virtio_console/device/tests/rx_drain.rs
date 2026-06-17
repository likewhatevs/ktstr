use super::super::super::*;
use super::*;

/// Empty pending-rx → drain is a no-op. No queue access, no
/// add_used, no signal_used. Pins the `if pending.is_empty()`
/// fast-exit at the head of `drain_port1_pending_rx`.
#[test]
fn drain_port1_pending_rx_empty_pending_is_noop() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    dev.set_mem(mem.clone());
    wire_port1_rxq_to_mock(&mut dev, &mock);
    open_port1(&mut dev);

    // Plant a write-only chain that would be popped IF the
    // function walked the queue; the empty-pending fast-exit
    // must skip the queue entirely.
    let data_addr = GuestAddress(0x10000);
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        64,
        VRING_DESC_F_WRITE as u16,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");

    assert!(
        dev.ports[1].pending_rx.is_empty(),
        "precondition: port1_pending_rx must start empty"
    );
    let int_before = dev.interrupt_status;

    dev.drain_pending_rx(1);

    // No add_used → used.idx still 0.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 0,
        "empty-pending fast-exit must not touch the queue"
    );
    assert_eq!(
        dev.interrupt_status, int_before,
        "empty-pending fast-exit must not call signal_used"
    );
}

/// DRIVER_OK gate: bytes pushed before DRIVER_OK stay in
/// `port1_pending_rx`. A regression that walked the queue
/// pre-DRIVER_OK would let the device read descriptor addresses
/// the driver has not yet committed (virtio-v1.2 §3.1.1).
#[test]
fn drain_port1_pending_rx_defers_without_driver_ok() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        64,
        VRING_DESC_F_WRITE as u16,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    // Walk FSM up to S_FEAT only — STOP before S_OK. Configure
    // q4 and mark it ready (allowed in S_FEAT..S_OK per
    // `queue_config_allowed`). The DRIVER_OK gate at the head of
    // `drain_port1_pending_rx` must still defer.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << VIRTIO_CONSOLE_F_MULTIPORT,
    );
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT1_RXQ as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
    let desc = mock.desc_table_addr().0;
    let avail = mock.avail_addr().0;
    let used = mock.used_addr().0;
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (desc >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (avail >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (used >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
    // Cannot call `open_port1` — `handle_control_event(PORT_OPEN)`
    // works regardless of FSM state but unrelated to the gate
    // under test. Set the field directly.
    dev.ports[1].opened = true;

    let payload = b"snapshot reply bytes";
    dev.ports[1].pending_rx.extend(payload.iter().copied());

    dev.drain_pending_rx(1);

    // Bytes preserved verbatim in pending_rx — no consumption.
    assert_eq!(
        dev.ports[1].pending_rx.len(),
        payload.len(),
        "DRIVER_OK gate must hold bytes in pending_rx"
    );
    // No add_used → used.idx still 0.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 0, "DRIVER_OK gate must skip add_used");
}

/// F_MULTIPORT runtime gate: even with DRIVER_OK and the queue
/// ready, if the driver did not negotiate F_MULTIPORT then q4
/// should not be walked. Pins the
/// `if !self.multiport_negotiated()` guard at line ~903.
#[test]
fn drain_port1_pending_rx_defers_without_multiport() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        64,
        VRING_DESC_F_WRITE as u16,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());

    // Walk FSM to DRIVER_OK negotiating ONLY VIRTIO_F_VERSION_1
    // (no F_MULTIPORT). `wire_console_queue_to_mock` always
    // negotiates both, so inline a custom version here.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT1_RXQ as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
    let desc = mock.desc_table_addr().0;
    let avail = mock.avail_addr().0;
    let used = mock.used_addr().0;
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (desc >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (avail >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (used >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    assert_eq!(
        dev.device_status, S_OK,
        "precondition: FSM must reach DRIVER_OK"
    );
    assert!(
        !dev.multiport_negotiated(),
        "precondition: F_MULTIPORT must NOT be negotiated"
    );
    // Set port_opened[1] directly so the multiport gate is the
    // ONLY gate the test exercises (hostile guest pretend-state).
    dev.ports[1].opened = true;

    let payload = b"reply bytes that must not leak";
    dev.ports[1].pending_rx.extend(payload.iter().copied());

    dev.drain_pending_rx(1);

    assert_eq!(
        dev.ports[1].pending_rx.len(),
        payload.len(),
        "F_MULTIPORT gate must hold bytes in pending_rx"
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 0, "F_MULTIPORT gate must skip add_used");
}

/// `port_opened[1]` gate: with DRIVER_OK + F_MULTIPORT but BEFORE
/// the guest has sent `PORT_OPEN(id=1, value=1)` on c_ovq, port 1
/// has no userspace reader. The kernel's port-1 buffer-pool
/// allocation only completes after PORT_OPEN; pushing bytes
/// through descriptors that exist before the open lets the
/// kernel discard them with no userspace consumer (per
/// drivers/char/virtio_console.c `port_fops_open`). Pins the
/// `if !self.port_opened[1]` guard at line ~911.
///
/// After the guest opens the port via PORT_OPEN, the deferred
/// drain runs (the open transition itself triggers it at line
/// ~1268); the bytes must then land in the queue.
#[test]
fn drain_port1_pending_rx_defers_until_port_open() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let payload = b"deferred snapshot reply";
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        64,
        VRING_DESC_F_WRITE as u16,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_port1_rxq_to_mock(&mut dev, &mock);

    // Port 1 not yet opened — the `port_opened[1]` gate must
    // defer.
    assert!(
        !dev.ports[1].opened,
        "precondition: port_opened[1] must be false"
    );

    dev.ports[1].pending_rx.extend(payload.iter().copied());
    dev.drain_pending_rx(1);

    // Bytes still pending; queue untouched.
    assert_eq!(
        dev.ports[1].pending_rx.len(),
        payload.len(),
        "port_opened[1] gate must defer when guest has not opened port 1"
    );
    let used_idx_before: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx before open");
    assert_eq!(used_idx_before, 0, "port_opened[1] gate must skip add_used");

    // Now drive PORT_OPEN(id=1, value=1). The handler at line
    // ~1268 calls drain_port1_pending_rx on the closed→open
    // transition, which now must drain.
    open_port1(&mut dev);

    assert!(
        dev.ports[1].pending_rx.is_empty(),
        "after PORT_OPEN, deferred bytes must drain"
    );
    let used_idx_after: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx after open");
    assert_eq!(
        used_idx_after, 1,
        "after PORT_OPEN, the deferred chain must add_used"
    );
    // Bytes landed in the guest descriptor buffer.
    let mut readback = vec![0u8; payload.len()];
    mem.read_slice(&mut readback, data_addr)
        .expect("read back delivered payload");
    assert_eq!(
        readback, payload,
        "delivered bytes must match the queued payload verbatim"
    );
}

/// No-mem gate: a device whose `mem` field is None (the freeze
/// coordinator may push reply bytes during the brief window
/// before `set_mem` lands) must not crash and must hold the
/// bytes for retry. Pins the `match self.mem.as_ref() { ... None
/// => return }` arm at line ~918.
#[test]
fn drain_port1_pending_rx_defers_without_mem() {
    let mut dev = VirtioConsole::new();
    // Walk FSM to DRIVER_OK with F_MULTIPORT but DO NOT call
    // set_mem. `init_device` reaches S_OK without touching mem.
    init_device(&mut dev);
    assert!(dev.mem.is_none(), "precondition: mem must be None");
    dev.ports[1].opened = true;

    let payload = b"reply bytes pre-set_mem";
    dev.ports[1].pending_rx.extend(payload.iter().copied());

    // No panic, no crash.
    dev.drain_pending_rx(1);

    assert_eq!(
        dev.ports[1].pending_rx.len(),
        payload.len(),
        "no-mem gate must hold bytes in pending_rx"
    );
}

/// Queue-not-ready gate: bytes stay pending if PORT1_RXQ has not
/// been marked ready by the driver. Pins the
/// `!self.queues[PORT1_RXQ].ready()` guard at line ~928. The
/// driver writes QUEUE_READY=1 only after the desc/avail/used
/// addresses are committed; reading the queue before that point
/// would walk uninitialized state.
#[test]
fn drain_port1_pending_rx_defers_when_queue_not_ready() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    dev.set_mem(mem);
    // Walk FSM to DRIVER_OK with F_MULTIPORT, but DO NOT
    // configure or ready q4. `init_device` reaches DRIVER_OK
    // without configuring any queue.
    init_device(&mut dev);
    assert!(
        !dev.queues[PORT1_RXQ].ready(),
        "precondition: q4 must NOT be ready"
    );
    dev.ports[1].opened = true;

    let payload = b"reply bytes before queue ready";
    dev.ports[1].pending_rx.extend(payload.iter().copied());

    dev.drain_pending_rx(1);

    assert_eq!(
        dev.ports[1].pending_rx.len(),
        payload.len(),
        "queue-not-ready gate must hold bytes in pending_rx"
    );
}

/// Single-descriptor write-only chain: happy-path baseline for
/// the torn-write tests below. Pins that a normal drain delivers
/// the payload to the descriptor buffer, drains
/// `port1_pending_rx`, advances `used.idx`, and signals the
/// guest via INT_VRING + irq_evt.
#[test]
fn drain_port1_pending_rx_single_descriptor_happy_path() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let payload = b"snapshot reply payload bytes";
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        payload.len() as u32,
        VRING_DESC_F_WRITE as u16,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_port1_rxq_to_mock(&mut dev, &mock);
    open_port1(&mut dev);

    dev.ports[1].pending_rx.extend(payload.iter().copied());
    dev.drain_pending_rx(1);

    // pending_rx fully drained.
    assert!(
        dev.ports[1].pending_rx.is_empty(),
        "happy-path drain must consume all pending bytes"
    );
    // Bytes landed in the guest descriptor.
    let mut readback = vec![0u8; payload.len()];
    mem.read_slice(&mut readback, data_addr)
        .expect("read back delivered payload");
    assert_eq!(
        readback, payload,
        "delivered bytes must equal the queued payload"
    );
    // Used ring reflects exactly one completion.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "happy-path drain must add_used exactly once");
    // signal_used set INT_VRING.
    assert_ne!(
        dev.interrupt_status & VIRTIO_MMIO_INT_VRING,
        0,
        "INT_VRING must be set after a non-zero drain"
    );
    let irq_count = dev.irq_evt.read().expect("irq_evt was written");
    assert!(
        irq_count > 0,
        "irq_evt counter must be non-zero after signal_used"
    );
}

/// Multi-descriptor torn-write recovery: a chain with two
/// write-only descriptors where the second points at unmapped
/// guest memory (GPA past `make_chain_test_mem`'s 2 MiB cap).
/// `mem.write_slice` fails on the second descriptor, triggering
/// the torn-write branch.
///
/// Pins the four invariants of the torn-write recovery (lines
/// ~983-997):
/// (a) `chain_torn = true` triggers `q.add_used(mem, head, 0)`
///     — used.idx advances to 1, but the published `len` is 0;
/// (b) bytes stay in `port1_pending_rx` (the
///     `drain(..consumed_offset)` is in the non-torn branch, so
///     a torn chain does NOT consume bytes);
/// (c) the drain loop breaks (no further chains processed for
///     this notify even if more are available);
/// (d) `total_written` stays 0 for the torn chain — combined
///     with the absence of any prior successful chain in this
///     test, signal_used is NOT called and INT_VRING stays 0.
///
/// Without `add_used(0)` the chain head would leak from the
/// avail ring until reset (the c_ivq, port-0 RX, and port-1 RX
/// torn paths all share this recovery convention).
#[test]
fn drain_port1_pending_rx_torn_write_publishes_head_with_zero_len() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    // Valid GPA for the first descriptor.
    let valid_addr = GuestAddress(0x10000);
    // GPA past the 2 MiB map → write_slice fails. 4 MiB chosen
    // for clarity; any value >= 2 MiB works.
    let unmapped_addr = GuestAddress(4 << 20);
    // Two descriptors, both write-only, chained via NEXT.
    // First desc at index 0 (chain head); second at index 1.
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            valid_addr.0,
            32,
            (VRING_DESC_F_WRITE | VRING_DESC_F_NEXT) as u16,
            1,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            unmapped_addr.0,
            32,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.build_desc_chain(&descs).expect("build torn chain");
    dev.set_mem(mem.clone());
    wire_port1_rxq_to_mock(&mut dev, &mock);
    open_port1(&mut dev);

    // 64-byte payload: first 32 bytes will write to valid_addr,
    // the next 32 will attempt to write to unmapped_addr and
    // fail.
    let payload: Vec<u8> = (0..64u8).collect();
    dev.ports[1].pending_rx.extend(payload.iter().copied());
    let int_before = dev.interrupt_status;
    // Drain irq_evt so the post-drain assertion can detect a
    // spurious signal_used call.
    let _ = dev.irq_evt.read();

    dev.drain_pending_rx(1);

    // (a) used.idx == 1: torn chain head was add_used'd with
    // len=0.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 1,
        "torn-write recovery must add_used the chain head (with len=0)"
    );
    // The published len for that head must be 0. used-ring
    // entry layout: u16 flags, u16 idx, then array of
    // VRING_USED_ELEM { u32 id; u32 len }. The first entry's
    // `len` field sits at used_addr + 4 (flags+idx) + 4 (id).
    let used_elem_len: u32 = mem
        .read_obj(mock.used_addr().checked_add(8).unwrap())
        .expect("read used elem 0 len");
    assert_eq!(
        used_elem_len, 0,
        "torn-write recovery must publish len=0 for the chain head \
         — a non-zero len would tell the guest the descriptor was \
         fully filled, leading to data corruption"
    );

    // (b) Bytes stay in pending_rx. The first descriptor's
    // partial write does not consume bytes from the deque
    // because the torn branch skips the
    // `drain(..consumed_offset)` call (line ~1009 is in the
    // success branch, after the `if chain_torn { ... break; }`
    // arm at line ~983).
    assert_eq!(
        dev.ports[1].pending_rx.len(),
        payload.len(),
        "torn-write recovery must preserve bytes in pending_rx \
         for retry on the next drain cycle"
    );
    // Verify byte identity, not just length: a regression that
    // partially drained then re-extended could mask itself.
    let preserved: Vec<u8> = dev.ports[1].pending_rx.iter().copied().collect();
    assert_eq!(
        preserved, payload,
        "preserved bytes must be the original payload verbatim"
    );

    // (d) signal_used must NOT have been called: total_written
    // stays 0 for the torn chain, so the
    // `if total_written > 0 { signal_used() }` branch at line
    // ~1012 is skipped.
    assert_eq!(
        dev.interrupt_status, int_before,
        "torn-only chain must not trigger signal_used (total_written=0)"
    );
    match dev.irq_evt.read() {
        Ok(n) => panic!("irq_evt must NOT have been written, got {n}"),
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
        Err(e) => panic!("unexpected irq_evt read error: {e}"),
    }
}

/// Torn-write breaks the drain loop: even when MORE chains are
/// available in the avail ring, a torn first chain stops the
/// drain immediately. Pins the `break` after `add_used(0)` at
/// line ~996. A regression that continued to the next chain
/// would (1) interleave torn-recovery with success traffic,
/// confusing failure-mode analysis, and (2) potentially deliver
/// bytes out of order if the torn chain's bytes were retried
/// after a later chain landed.
///
/// Two chains:
///   - chain 0: torn (second desc unmapped)
///   - chain 1: fully valid single descriptor
///
/// After drain, used.idx must be exactly 1 (only the torn head),
/// chain 1 must remain in the avail ring un-consumed (its data
/// region still holds zeros, never the payload), and a follow-up
/// drain cycle must process chain 1.
#[test]
fn drain_port1_pending_rx_torn_write_breaks_drain_loop() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let valid_addr_a = GuestAddress(0x10000);
    let unmapped_addr = GuestAddress(4 << 20);
    let valid_addr_b = GuestAddress(0x20000);

    // `add_desc_chains` publishes one chain head per descriptor
    // entry in the slice that does NOT carry F_NEXT (entries
    // with F_NEXT are linked tails of the prior head). With
    // descs[0] carrying F_NEXT->1 and descs[1]/descs[2] without
    // F_NEXT, two chain heads land in the avail ring:
    //   chain 0 = descs[0] -> descs[1]   (torn: descs[1] unmapped)
    //   chain 1 = descs[2]               (valid, single desc)
    let descs = [
        // Chain 0 head (idx 0): valid + NEXT to idx 1.
        RawDescriptor::from(SplitDescriptor::new(
            valid_addr_a.0,
            32,
            (VRING_DESC_F_WRITE | VRING_DESC_F_NEXT) as u16,
            1,
        )),
        // Chain 0 tail (idx 1): unmapped.
        RawDescriptor::from(SplitDescriptor::new(
            unmapped_addr.0,
            32,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
        // Chain 1 (idx 2): single valid descriptor.
        RawDescriptor::from(SplitDescriptor::new(
            valid_addr_b.0,
            32,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.add_desc_chains(&descs, 0).expect("publish two chains");
    dev.set_mem(mem.clone());
    wire_port1_rxq_to_mock(&mut dev, &mock);
    open_port1(&mut dev);

    // 64-byte payload — enough that the second descriptor of
    // chain 0 would be filled if the unmapped write succeeded.
    let payload: Vec<u8> = (0..64u8).collect();
    dev.ports[1].pending_rx.extend(payload.iter().copied());

    dev.drain_pending_rx(1);

    // used.idx == 1: ONLY chain 0's head was add_used'd (with
    // len=0); chain 1 was NOT processed because the torn-write
    // recovery broke the drain loop.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 1,
        "torn-write recovery must break the drain loop — chain 1 \
         must remain unconsumed even though its descriptor is valid"
    );

    // Chain 1's data region must be untouched: read back
    // valid_addr_b — it must be all zeros (the test memory's
    // initial state), NOT any byte from `payload`.
    let mut readback_b = vec![0u8; 32];
    mem.read_slice(&mut readback_b, valid_addr_b)
        .expect("read back chain 1 data region");
    assert!(
        readback_b.iter().all(|&b| b == 0),
        "chain 1's data region must be untouched — the drain loop \
         must NOT have reached chain 1 after the torn break"
    );

    // Bytes still in pending_rx for retry.
    assert_eq!(
        dev.ports[1].pending_rx.len(),
        payload.len(),
        "all bytes must remain in pending_rx (torn chain consumed nothing)"
    );

    // A second drain cycle must process chain 1 successfully —
    // chain 0's head was already add_used'd, so chain 1 is the
    // next chain to pop.
    dev.drain_pending_rx(1);

    let used_idx_after: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx after second drain");
    assert_eq!(
        used_idx_after, 2,
        "second drain must process chain 1 (used.idx 1 -> 2)"
    );
    // Chain 1 received the first 32 bytes of the still-pending
    // payload.
    let mut readback_b2 = vec![0u8; 32];
    mem.read_slice(&mut readback_b2, valid_addr_b)
        .expect("read chain 1 data after second drain");
    assert_eq!(
        readback_b2,
        payload[..32],
        "chain 1 must hold the first 32 bytes of the preserved payload"
    );
    // 32 bytes consumed from pending_rx; 32 remain.
    assert_eq!(
        dev.ports[1].pending_rx.len(),
        payload.len() - 32,
        "second drain must consume only chain 1's capacity (32 bytes)"
    );
}

/// Torn-write after a successful prior chain in the SAME drain
/// call: the prior chain's `total_written` accumulates, then the
/// torn chain breaks the loop. signal_used MUST be called
/// because `total_written > 0` from the prior chain. Pins that
/// the torn-write break does not suppress the signal for
/// successfully-delivered chains earlier in the same call.
///
/// Two chains:
///   - chain 0: valid single descriptor → drains 32 bytes
///   - chain 1: torn (second desc unmapped) → publishes head=0,
///     breaks loop
///
/// After drain: used.idx==2, INT_VRING set, irq_evt non-zero,
/// 32 bytes consumed from pending_rx (chain 0 only).
#[test]
fn drain_port1_pending_rx_torn_after_success_still_signals() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let valid_addr_a = GuestAddress(0x10000);
    let valid_addr_b = GuestAddress(0x20000);
    let unmapped_addr = GuestAddress(4 << 20);

    // Layout (no F_NEXT means chain end; F_NEXT means link to
    // the named index):
    //   idx 0 = chain 0 (valid, single, no NEXT)
    //   idx 1 = chain 1 head (valid + NEXT -> idx 2)
    //   idx 2 = chain 1 tail (unmapped)
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(
            valid_addr_a.0,
            32,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            valid_addr_b.0,
            32,
            (VRING_DESC_F_WRITE | VRING_DESC_F_NEXT) as u16,
            2,
        )),
        RawDescriptor::from(SplitDescriptor::new(
            unmapped_addr.0,
            32,
            VRING_DESC_F_WRITE as u16,
            0,
        )),
    ];
    mock.add_desc_chains(&descs, 0)
        .expect("publish success+torn chains");
    dev.set_mem(mem.clone());
    wire_port1_rxq_to_mock(&mut dev, &mock);
    open_port1(&mut dev);

    let payload: Vec<u8> = (0..96u8).collect();
    dev.ports[1].pending_rx.extend(payload.iter().copied());
    // Drain irq_evt so the post-drain check sees only the new
    // signal_used call.
    let _ = dev.irq_evt.read();

    dev.drain_pending_rx(1);

    // used.idx == 2: chain 0 add_used with len=32, chain 1
    // (torn) add_used with len=0.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 2,
        "drain must add_used both chain 0 (len=32) and chain 1 \
         (torn, len=0)"
    );
    // INT_VRING must be set: total_written > 0 from chain 0.
    assert_ne!(
        dev.interrupt_status & VIRTIO_MMIO_INT_VRING,
        0,
        "signal_used must have been called because chain 0 \
         delivered 32 bytes"
    );
    let irq_count = dev.irq_evt.read().expect("irq_evt was written");
    assert!(
        irq_count > 0,
        "irq_evt counter must be non-zero after signal_used"
    );

    // Chain 0 received the first 32 bytes.
    let mut readback_a = vec![0u8; 32];
    mem.read_slice(&mut readback_a, valid_addr_a)
        .expect("read chain 0 data");
    assert_eq!(
        readback_a,
        payload[..32],
        "chain 0 must hold the first 32 bytes of the payload"
    );
    // pending_rx: 32 bytes from chain 0 consumed; bytes 32-95
    // remain (chain 1's torn write consumed nothing).
    assert_eq!(
        dev.ports[1].pending_rx.len(),
        payload.len() - 32,
        "only chain 0's bytes were consumed from pending_rx"
    );
    let preserved: Vec<u8> = dev.ports[1].pending_rx.iter().copied().collect();
    assert_eq!(
        preserved,
        payload[32..],
        "preserved bytes must be exactly the suffix not delivered \
         to chain 0 (chain 1's first descriptor's partial write \
         does NOT consume from the deque)"
    );
}
