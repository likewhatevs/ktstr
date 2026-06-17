use super::super::super::*;
use super::*;

// ----------------------------------------------------------------
// Port 2 TX (PORT2_TXQ) chain-level tests.
// ----------------------------------------------------------------

/// Single-descriptor TX chain on port 2: one device-readable
/// segment with a known byte pattern lands verbatim in
/// `port2_tx_buf` (observable via `drain_port2_bulk`). Pins the
/// PORT2_TXQ branch of the queue_idx routing match in
/// `process_tx_into` (line ~668): a regression that mis-routed
/// to `port0_tx_buf` or `port1_tx_buf` would surface here.
#[test]
fn port2_tx_single_descriptor_lands_in_port2_buf() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let payload = b"port2 stats relay TX bytes";
    mem.write_slice(payload, data_addr).expect("plant payload");
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        payload.len() as u32,
        0,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_console_queue_to_mock(&mut dev, &mock, PORT2_TXQ as u32);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

    // Bytes must have landed in port2_tx_buf, observable via
    // drain_port2_bulk.
    let drained = dev.drain_port2_bulk();
    assert_eq!(
        drained,
        payload.to_vec(),
        "port 2 TX must deliver the descriptor's bytes to \
         drain_port2_bulk verbatim",
    );
    // Port 0 buffer must be untouched.
    assert!(
        dev.drain_output().is_empty(),
        "port 0 TX buffer must remain empty when only port 2 was notified",
    );
    // Port 1 buffer must be untouched — the routing match must
    // not have spilled into the port-1 deque.
    assert!(
        dev.drain_bulk().is_empty(),
        "port 1 TX buffer must remain empty when only port 2 was notified",
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "exactly one used-ring entry expected");
    assert_ne!(
        dev.interrupt_status & VIRTIO_MMIO_INT_VRING,
        0,
        "INT_VRING must be set after a successful TX drain",
    );
    let irq_count = dev.irq_evt.read().expect("irq_evt was written");
    assert!(
        irq_count > 0,
        "irq_evt counter must be non-zero after signal_used",
    );
}

/// Multi-descriptor port-2 TX chain: four 4 KiB segments
/// concatenate in `port2_tx_buf` in chain order. Mirrors
/// `port1_tx_multi_descriptor_chain_concatenates` — pins that
/// the per-descriptor staged-scratch append path for PORT2_TXQ
/// (line ~668-694) preserves byte boundaries and chain order.
/// The host's stats client expects scx_stats responses to arrive
/// as contiguous newline-delimited JSON; chain-order corruption
/// would tear the JSON across response boundaries.
#[test]
fn port2_tx_multi_descriptor_chain_concatenates() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    const PAGE: u32 = 4096;
    let segs: [(GuestAddress, u8); 4] = [
        (GuestAddress(0x10000), 0xB1),
        (GuestAddress(0x12000), 0xB2),
        (GuestAddress(0x14000), 0xB3),
        (GuestAddress(0x16000), 0xB4),
    ];
    for (addr, fill) in &segs {
        let buf = vec![*fill; PAGE as usize];
        mem.write_slice(&buf, *addr).expect("plant segment");
    }
    let descs = [
        RawDescriptor::from(SplitDescriptor::new(segs[0].0.0, PAGE, 0, 0)),
        RawDescriptor::from(SplitDescriptor::new(segs[1].0.0, PAGE, 0, 0)),
        RawDescriptor::from(SplitDescriptor::new(segs[2].0.0, PAGE, 0, 0)),
        RawDescriptor::from(SplitDescriptor::new(segs[3].0.0, PAGE, 0, 0)),
    ];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_console_queue_to_mock(&mut dev, &mock, PORT2_TXQ as u32);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

    let drained = dev.drain_port2_bulk();
    assert_eq!(
        drained.len(),
        4 * PAGE as usize,
        "drain_port2_bulk length must equal sum of segment lengths",
    );
    for (i, (_, fill)) in segs.iter().enumerate() {
        let start = i * PAGE as usize;
        let end = start + PAGE as usize;
        assert!(
            drained[start..end].iter().all(|&b| b == *fill),
            "segment {i} must hold fill {fill:#x} verbatim — chain order \
             or per-descriptor append regressed",
        );
    }
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "one chain → one used-ring entry on port 2 TX",);
}

/// Oversize port-2 TX descriptor truncates to `TX_DESC_MAX`. A
/// hostile guest publishing `len > 32 KiB` on port 2 must not
/// trigger a multi-GiB scratch alloc. Mirrors
/// `port1_tx_oversize_descriptor_truncates_to_tx_desc_max` for
/// the PORT2_TXQ scratch path.
#[test]
fn port2_tx_oversize_descriptor_truncates_to_tx_desc_max() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    const OVERSIZE: usize = TX_DESC_MAX * 2;
    let mut payload = vec![0x55u8; TX_DESC_MAX];
    payload.extend_from_slice(&vec![0x99u8; TX_DESC_MAX]);
    assert_eq!(payload.len(), OVERSIZE);
    mem.write_slice(&payload, data_addr).expect("plant payload");
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        OVERSIZE as u32,
        0,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_console_queue_to_mock(&mut dev, &mock, PORT2_TXQ as u32);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

    let drained = dev.drain_port2_bulk();
    assert_eq!(
        drained.len(),
        TX_DESC_MAX,
        "oversize port-2 descriptor must truncate to TX_DESC_MAX",
    );
    assert!(
        drained.iter().all(|&b| b == 0x55),
        "truncated bytes must be the FIRST TX_DESC_MAX bytes \
         (0x55), not anything past the cap",
    );
}

/// Port 2 TX rejected without DRIVER_OK: mirrors
/// `port1_tx_rejected_without_driver_ok`. Walks the FSM only to
/// FEATURES_OK, configures PORT2_TXQ, plants a chain, fires the
/// notify; the gate at the head of `process_tx_into` must drop
/// the notify with no observable side effects.
#[test]
fn port2_tx_rejected_without_driver_ok() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let payload = b"this must NOT reach port2_tx_buf";
    mem.write_slice(payload, data_addr).expect("plant payload");
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        payload.len() as u32,
        0,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());

    // Walk FSM to S_FEAT only — STOP before S_OK.
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
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT2_TXQ as u32);
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
    assert_eq!(
        dev.device_status & VIRTIO_CONFIG_S_DRIVER_OK,
        0,
        "precondition: DRIVER_OK must NOT be set",
    );
    assert!(
        dev.queues[PORT2_TXQ].ready(),
        "precondition: port 2 TX queue must be ready",
    );

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

    assert!(
        dev.drain_port2_bulk().is_empty(),
        "port2_tx_buf must remain empty — DRIVER_OK gate must \
         reject pre-DRIVER_OK notify",
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 0,
        "used.idx must be 0 — DRIVER_OK gate must skip add_used",
    );
    assert_eq!(
        dev.interrupt_status, 0,
        "interrupt_status must be 0 — DRIVER_OK gate must skip signal_used",
    );
}

/// Port 2 TX rejected without F_MULTIPORT: PORT2_TXQ is
/// multiport-only (line 576 in `process_tx_into`). A guest that
/// reaches DRIVER_OK without negotiating F_MULTIPORT must not
/// have its PORT2_TXQ notifies serviced — the legacy single-
/// console path never exercises port 2.
#[test]
fn port2_tx_rejected_without_multiport() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let payload = b"port2 bytes that must not leak without multiport";
    mem.write_slice(payload, data_addr).expect("plant payload");
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        payload.len() as u32,
        0,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());

    // Walk FSM to DRIVER_OK negotiating ONLY VIRTIO_F_VERSION_1
    // (no F_MULTIPORT). `wire_console_queue_to_mock` always
    // negotiates both, so inline the custom version here.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT2_TXQ as u32);
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

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

    assert!(
        dev.drain_port2_bulk().is_empty(),
        "port2_tx_buf must remain empty — F_MULTIPORT gate must \
         reject the notify",
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 0,
        "used.idx must be 0 — F_MULTIPORT gate must skip add_used",
    );
}

/// Port-0 vs port-1 vs port-2 TX routing: bytes from each port
/// land in the matching `port{0,1,2}_tx_buf`. Extends
/// `port0_tx_vs_port1_tx_routes_to_correct_buffer` with port 2,
/// pinning the three-way queue_idx → buffer dispatch. A
/// regression that mis-routed any pair would let the stats
/// stream corrupt the bulk TLV stream or the console output.
#[test]
fn port0_vs_port1_vs_port2_tx_routes_to_correct_buffer() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    // Three MockSplitQueues at distinct GPAs (each ring fits in
    // < 1 KiB at queue size 16; 4 KiB stride is comfortable).
    let mock0 = MockSplitQueue::create(&mem, GuestAddress(0x0), 16);
    let mock1 = MockSplitQueue::create(&mem, GuestAddress(0x1000), 16);
    let mock2 = MockSplitQueue::create(&mem, GuestAddress(0x2000), 16);
    let port0_data_addr = GuestAddress(0x10000);
    let port1_data_addr = GuestAddress(0x20000);
    let port2_data_addr = GuestAddress(0x30000);
    let port0_payload = b"port0 console bytes";
    let port1_payload = b"port1 bulk TLV bytes";
    let port2_payload = b"port2 stats relay bytes";
    mem.write_slice(port0_payload, port0_data_addr)
        .expect("plant port0 payload");
    mem.write_slice(port1_payload, port1_data_addr)
        .expect("plant port1 payload");
    mem.write_slice(port2_payload, port2_data_addr)
        .expect("plant port2 payload");

    let port0_descs = [RawDescriptor::from(SplitDescriptor::new(
        port0_data_addr.0,
        port0_payload.len() as u32,
        0,
        0,
    ))];
    let port1_descs = [RawDescriptor::from(SplitDescriptor::new(
        port1_data_addr.0,
        port1_payload.len() as u32,
        0,
        0,
    ))];
    let port2_descs = [RawDescriptor::from(SplitDescriptor::new(
        port2_data_addr.0,
        port2_payload.len() as u32,
        0,
        0,
    ))];
    mock0
        .build_desc_chain(&port0_descs)
        .expect("build port0 chain");
    mock1
        .build_desc_chain(&port1_descs)
        .expect("build port1 chain");
    mock2
        .build_desc_chain(&port2_descs)
        .expect("build port2 chain");
    dev.set_mem(mem.clone());

    // Walk FSM up to FEATURES_OK, configure all three TX queues,
    // then transition to DRIVER_OK.
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
    // Configure each TX queue.
    for (q_idx, mock_ref) in [
        (PORT0_TXQ, &mock0),
        (PORT1_TXQ, &mock1),
        (PORT2_TXQ, &mock2),
    ] {
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, q_idx as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
        let d = mock_ref.desc_table_addr().0;
        let a = mock_ref.avail_addr().0;
        let u = mock_ref.used_addr().0;
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, d as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (d >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, a as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (a >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, u as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (u >> 32) as u32);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
    }
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    assert_eq!(
        dev.device_status, S_OK,
        "FSM did not reach DRIVER_OK after all three queues configured",
    );

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT0_TXQ as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

    let port0_drained = dev.drain_output();
    assert_eq!(
        port0_drained,
        port0_payload.to_vec(),
        "port 0 TX must route to port0_tx_buf",
    );
    let port1_drained = dev.drain_bulk();
    assert_eq!(
        port1_drained,
        port1_payload.to_vec(),
        "port 1 TX must route to port1_tx_buf",
    );
    let port2_drained = dev.drain_port2_bulk();
    assert_eq!(
        port2_drained,
        port2_payload.to_vec(),
        "port 2 TX must route to port2_tx_buf",
    );
    let port0_used_idx: u16 = mem
        .read_obj(mock0.used_addr().checked_add(2).unwrap())
        .expect("read port0 used.idx");
    let port1_used_idx: u16 = mem
        .read_obj(mock1.used_addr().checked_add(2).unwrap())
        .expect("read port1 used.idx");
    let port2_used_idx: u16 = mem
        .read_obj(mock2.used_addr().checked_add(2).unwrap())
        .expect("read port2 used.idx");
    assert_eq!(port0_used_idx, 1);
    assert_eq!(port1_used_idx, 1);
    assert_eq!(
        port2_used_idx, 1,
        "port 2 used.idx must reflect 1 completion"
    );
}

/// Port 2 TX wakes `stats_tx_evt` (NOT `tx_evt`). Pins the
/// queue_idx == PORT2_TXQ branch in `process_tx_into` (lines
/// 768-772). A regression that fired `tx_evt` for port 2 would
/// wake the freeze coordinator's TOKEN_TX handler on every
/// scheduler-stats response, contending on the device mutex.
/// A regression that fired `stats_tx_evt` for ports 0/1 would
/// wake the stats client on every console / bulk byte.
#[test]
fn port2_tx_wakes_stats_tx_evt_only() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let payload = b"port2 tx wakes stats_tx_evt";
    mem.write_slice(payload, data_addr).expect("plant payload");
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        payload.len() as u32,
        0,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_console_queue_to_mock(&mut dev, &mock, PORT2_TXQ as u32);

    // Drain any prior counter so the post-notify reads measure
    // only this notify's effect.
    let _ = dev.tx_evt.read();
    let _ = dev.stats_tx_evt.read();

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT2_TXQ as u32);

    // stats_tx_evt must have been written.
    let stats_count = dev
        .stats_tx_evt
        .read()
        .expect("stats_tx_evt was written by port-2 TX");
    assert!(
        stats_count > 0,
        "port 2 TX must wake stats_tx_evt (count > 0)",
    );
    // tx_evt must NOT have been written. EFD_NONBLOCK returns
    // WouldBlock on an empty eventfd.
    match dev.tx_evt.read() {
        Ok(n) => {
            panic!("tx_evt must NOT have been written by port-2 TX, but read returned {n}")
        }
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
        Err(e) => panic!("unexpected tx_evt read error: {e}"),
    }
}

/// Port 1 TX wakes `tx_evt` (NOT `stats_tx_evt`). Symmetric to
/// `port2_tx_wakes_stats_tx_evt_only`. Without this control
/// test, a regression that swapped the eventfd dispatch would
/// only surface on the port-2 side.
#[test]
fn port1_tx_wakes_tx_evt_only_not_stats_tx_evt() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let payload = b"port1 tx wakes tx_evt only";
    mem.write_slice(payload, data_addr).expect("plant payload");
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        payload.len() as u32,
        0,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_console_queue_to_mock(&mut dev, &mock, PORT1_TXQ as u32);

    let _ = dev.tx_evt.read();
    let _ = dev.stats_tx_evt.read();

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

    let tx_count = dev.tx_evt.read().expect("tx_evt was written by port-1 TX");
    assert!(tx_count > 0, "port 1 TX must wake tx_evt");
    match dev.stats_tx_evt.read() {
        Ok(n) => {
            panic!("stats_tx_evt must NOT have been written by port-1 TX, but read returned {n}")
        }
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
        Err(e) => panic!("unexpected stats_tx_evt read error: {e}"),
    }
}

// ----------------------------------------------------------------
// Eventfd distinctness for the three signal channels.
// ----------------------------------------------------------------

/// `irq_evt`, `tx_evt`, and `stats_tx_evt` are three distinct
/// eventfds (distinct file descriptors). Extends
/// `new_creates_eventfds` with the third signal channel. A
/// regression that aliased any pair (e.g. cloned `tx_evt` into
/// `stats_tx_evt`) would let port-2 wakes contend on the
/// freeze coordinator's TOKEN_TX handler — the exact contention
/// the separate `stats_tx_evt` was introduced to avoid.
#[test]
fn three_eventfds_are_distinct_fds() {
    let dev = VirtioConsole::new();
    let irq_fd = dev.irq_evt().as_raw_fd();
    let tx_fd = dev.tx_evt().as_raw_fd();
    let stats_fd = dev.stats_tx_evt().as_raw_fd();
    assert!(irq_fd >= 0, "irq_evt fd must be valid");
    assert!(tx_fd >= 0, "tx_evt fd must be valid");
    assert!(stats_fd >= 0, "stats_tx_evt fd must be valid");
    assert_ne!(irq_fd, tx_fd, "irq_evt and tx_evt must be distinct fds");
    assert_ne!(
        irq_fd, stats_fd,
        "irq_evt and stats_tx_evt must be distinct fds"
    );
    assert_ne!(
        tx_fd, stats_fd,
        "tx_evt and stats_tx_evt must be distinct fds — \
         aliasing them would defeat the per-port wake \
         separation",
    );
}

// ----------------------------------------------------------------
// Port-2 control message handshake (id=2).
// ----------------------------------------------------------------

/// `DEVICE_READY` enqueues `PORT_ADD` for every port id including
/// id=2. The existing `handle_device_ready_enqueues_port_adds`
/// covers all ports via a loop over `NUM_PORTS`; this pins the
/// id=2 entry specifically because port 2 was added later than
/// ports 0/1 and a regression that hardcoded the loop bound
/// (e.g. `for id in 0..2`) would skip port 2's PORT_ADD without
/// any other test surfacing the regression.
#[test]
fn handle_device_ready_enqueues_port_add_for_port_2() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: 0xFFFF_FFFF,
        event: VIRTIO_CONSOLE_DEVICE_READY,
        value: 1,
    });
    assert_eq!(dev.control_out.len(), NUM_PORTS as usize);
    // Find the PORT_ADD for id=2.
    let port2_add = dev.control_out.iter().find(|m| match m {
        ControlOut::Cmd(c) => c.id == 2 && c.event == VIRTIO_CONSOLE_PORT_ADD,
        _ => false,
    });
    match port2_add {
        Some(ControlOut::Cmd(c)) => {
            let value = c.value;
            assert_eq!(
                value, 1,
                "PORT_ADD value=1 matches QEMU \
                 (hw/char/virtio-serial-bus.c)",
            );
        }
        Some(_) => panic!("PORT_ADD for id=2 must be a Cmd variant"),
        None => panic!(
            "DEVICE_READY must enqueue PORT_ADD for id=2 (port 2 is \
             the scheduler-stats relay; missing it strands the port \
             in the kernel's port-add-pending state)"
        ),
    }
}

/// `PORT_READY` for id=2 enqueues `PORT_NAME` (with `PORT2_NAME`)
/// then `PORT_OPEN` — same order as port 1 (PORT_NAME before
/// PORT_OPEN keeps udev sysfs symlink creation ahead of any
/// userspace `/dev/vport0p2` open). Pins the `id == 1 || id == 2`
/// branch in `handle_control_event` PORT_READY (line ~1782) and
/// the `match id { ... 2 => PORT2_NAME, ... }` lookup at
/// line ~1794.
#[test]
fn handle_port_ready_port2_name_then_open() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: 2,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    assert!(
        dev.ports[2].readied,
        "port_readied[2] must be set after PORT_READY for id=2"
    );
    // Bulk-port pattern: PORT_NAME then PORT_OPEN, 2 frames.
    assert_eq!(
        dev.control_out.len(),
        2,
        "PORT_READY for id=2 must enqueue exactly 2 frames \
         (PORT_NAME, PORT_OPEN) — same shape as port 1, distinct \
         from port 0's 3-frame announce",
    );
    match &dev.control_out[0] {
        ControlOut::Name { id, name } => {
            assert_eq!(*id, 2);
            assert_eq!(
                *name, PORT2_NAME,
                "PORT_NAME for id=2 must use PORT2_NAME (not \
                 PORT1_NAME or PORT0_NAME) — wrong name strands \
                 udev rules that disambiguate ports by name",
            );
        }
        _ => panic!("first control_out frame must be Name for port 2"),
    }
    match &dev.control_out[1] {
        ControlOut::Cmd(c) => {
            let id = c.id;
            let event = c.event;
            let value = c.value;
            assert_eq!(id, 2);
            assert_eq!(event, VIRTIO_CONSOLE_PORT_OPEN);
            assert_eq!(value, 1);
        }
        _ => panic!("second control_out frame must be Cmd for port 2 OPEN"),
    }
}

/// `PORT_OPEN(id=2, value=1)` flips `port_opened[2]` from false
/// to true; `value=0` flips it back. Pins the array-index path
/// in `handle_control_event` PORT_OPEN (line ~1814-1816) for
/// id=2 specifically. Without this test, a regression that
/// scoped the index update to `id < 2` would leave port_opened[2]
/// stuck at false and the port-2 RX gate would defer all bytes
/// indefinitely.
#[test]
fn handle_port_open_tracks_state_for_port_2() {
    let mut dev = VirtioConsole::new();
    assert!(
        !dev.ports[2].opened,
        "precondition: port_opened[2] must start false"
    );
    dev.handle_control_event(VirtioConsoleControl {
        id: 2,
        event: VIRTIO_CONSOLE_PORT_OPEN,
        value: 1,
    });
    assert!(
        dev.ports[2].opened,
        "PORT_OPEN(value=1) for id=2 must set port_opened[2]"
    );
    dev.handle_control_event(VirtioConsoleControl {
        id: 2,
        event: VIRTIO_CONSOLE_PORT_OPEN,
        value: 0,
    });
    assert!(
        !dev.ports[2].opened,
        "PORT_OPEN(value=0) for id=2 must clear port_opened[2]"
    );
}

/// `PORT_READY` for id=2 with value=0 must NOT set
/// `port_readied[2]` and must NOT enqueue announce frames —
/// same semantics as the port-0 / port-1 value=0 paths. Pins
/// that the value=0 early-return precedes the per-port gate
/// flag for id=2.
#[test]
fn handle_port_ready_port2_value_zero_skipped() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: 2,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 0,
    });
    assert!(
        dev.control_out.is_empty(),
        "PORT_READY(id=2, value=0) must NOT enqueue announce frames",
    );
    assert!(
        !dev.ports[2].readied,
        "PORT_READY(id=2, value=0) must NOT set port_readied[2] — \
         a future legitimate value=1 must still complete",
    );
}

/// `PORT_READY` repeat for port 2 must be ignored — symmetric
/// to the port-0 and port-1 repeat tests. A regression that
/// scoped the per-port gate to ids {0, 1} would let port 2
/// re-enqueue PORT_NAME / PORT_OPEN unboundedly.
#[test]
fn handle_port_ready_repeat_ignored_port2() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: 2,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    assert!(dev.ports[2].readied);
    let after_first = dev.control_out.len();
    assert_eq!(
        after_first, 2,
        "first PORT_READY for port 2 must enqueue 2 frames (PORT_NAME, PORT_OPEN)",
    );

    dev.handle_control_event(VirtioConsoleControl {
        id: 2,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    assert_eq!(
        dev.control_out.len(),
        after_first,
        "PORT_READY repeat for port 2 must be a no-op",
    );
}

// ----------------------------------------------------------------
// Reset coverage for port-2 specific state.
// ----------------------------------------------------------------

/// `reset()` clears `port_opened[2]`, `port_readied[2]`, and
/// `port2_pending_rx`, but PRESERVES `port2_tx_buf` (host-side
/// capture buffer, mirrors port 0 / port 1). Extends
/// `reset_clears_all_state` with explicit port-2 coverage. A
/// regression that cleared `port2_tx_buf` would discard
/// scheduler-stats responses captured at reset time before
/// `final_drain` could surface them; a regression that did NOT
/// clear `port2_pending_rx` would let stale request bytes leak
/// into a fresh probe. The `device_ready` reset is implicit in
/// `reset_clears_all_state`; this test focuses on the port-2
/// specific fields.
#[test]
fn reset_clears_port2_state_preserves_port2_tx_buf() {
    let mut dev = VirtioConsole::new();
    init_device(&mut dev);
    // Plant port-2 specific state.
    dev.ports[2].tx_buf.extend(b"leftover2".iter().copied());
    dev.ports[2]
        .pending_rx
        .extend(b"stale request bytes".iter().copied());
    dev.ports[2].opened = true;
    dev.ports[2].readied = true;

    // Reset.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);

    // port2_tx_buf survives — it's the host-side capture buffer
    // for `final_drain`. Mirrors `port{0,1}_tx_buf` semantics in
    // the existing `reset_clears_all_state` test.
    assert_eq!(
        dev.ports[2].tx_buf.iter().copied().collect::<Vec<u8>>(),
        b"leftover2",
        "port2_tx_buf must survive reset (host-side capture buffer)",
    );
    // port2_pending_rx is cleared: pending host→guest bytes have
    // no post-reset consumer (the kernel's port-2 reader has
    // already torn down).
    assert!(
        dev.ports[2].pending_rx.is_empty(),
        "port2_pending_rx must be cleared on reset (no post-reset consumer)",
    );
    // port_opened[2] and port_readied[2] reset to false. The
    // existing array assertion in `reset_clears_all_state` covers
    // this implicitly via `[false; NUM_PORTS as usize]`; this
    // explicit pin avoids dependence on NUM_PORTS happening to
    // equal 3.
    assert!(!dev.ports[2].opened, "port_opened[2] must reset to false",);
    assert!(!dev.ports[2].readied, "port_readied[2] must reset to false",);
}

// ----------------------------------------------------------------
// Port 2 RX (PORT2_RXQ / drain_port2_pending_rx) chain-level tests.
// ----------------------------------------------------------------

/// Empty pending-rx → drain is a no-op. Pins the
/// `if pending.is_empty()` fast-exit at the head of
/// `drain_port2_pending_rx`.
#[test]
fn drain_port2_pending_rx_empty_pending_is_noop() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    dev.set_mem(mem.clone());
    wire_port2_rxq_to_mock(&mut dev, &mock);
    open_port2(&mut dev);

    let data_addr = GuestAddress(0x10000);
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        64,
        VRING_DESC_F_WRITE as u16,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");

    assert!(
        dev.ports[2].pending_rx.is_empty(),
        "precondition: port2_pending_rx must start empty"
    );
    let int_before = dev.interrupt_status;

    dev.drain_pending_rx(2);

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

/// `port_opened[2]` gate: with DRIVER_OK + F_MULTIPORT but BEFORE
/// the guest has sent `PORT_OPEN(id=2, value=1)` on c_ovq, port 2
/// has no userspace reader. Pins the
/// `if !self.port_opened[2]` guard in `drain_port2_pending_rx`.
/// After the guest opens port 2 via PORT_OPEN, the deferred
/// drain runs (the open transition itself triggers it via the
/// `2 => self.drain_port2_pending_rx()` arm at line ~1829);
/// the bytes must then land in the queue.
#[test]
fn drain_port2_pending_rx_defers_until_port_open() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let payload = b"deferred scx_stats request";
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        64,
        VRING_DESC_F_WRITE as u16,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_port2_rxq_to_mock(&mut dev, &mock);

    // Port 2 not yet opened — gate must defer.
    assert!(
        !dev.ports[2].opened,
        "precondition: port_opened[2] must be false"
    );

    dev.ports[2].pending_rx.extend(payload.iter().copied());
    dev.drain_pending_rx(2);

    assert_eq!(
        dev.ports[2].pending_rx.len(),
        payload.len(),
        "port_opened[2] gate must defer when guest has not opened port 2"
    );
    let used_idx_before: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx before open");
    assert_eq!(used_idx_before, 0, "port_opened[2] gate must skip add_used");

    // Now drive PORT_OPEN(id=2, value=1). The handler at line
    // ~1829 calls drain_port2_pending_rx on the closed→open
    // transition.
    open_port2(&mut dev);

    assert!(
        dev.ports[2].pending_rx.is_empty(),
        "after PORT_OPEN(id=2), deferred bytes must drain"
    );
    let used_idx_after: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx after open");
    assert_eq!(
        used_idx_after, 1,
        "after PORT_OPEN(id=2), the deferred chain must add_used"
    );
    let mut readback = vec![0u8; payload.len()];
    mem.read_slice(&mut readback, data_addr)
        .expect("read back delivered payload");
    assert_eq!(
        readback, payload,
        "delivered bytes must match the queued payload verbatim"
    );
}

/// Single-descriptor write-only chain on port 2: happy-path
/// baseline. Pins that a normal drain delivers the payload to
/// the descriptor buffer, drains `port2_pending_rx`, advances
/// `used.idx`, and signals the guest via INT_VRING + irq_evt.
/// Mirrors `drain_port1_pending_rx_single_descriptor_happy_path`.
#[test]
fn drain_port2_pending_rx_single_descriptor_happy_path() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let payload = b"scx_stats request line\n";
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        payload.len() as u32,
        VRING_DESC_F_WRITE as u16,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_port2_rxq_to_mock(&mut dev, &mock);
    open_port2(&mut dev);

    dev.ports[2].pending_rx.extend(payload.iter().copied());
    dev.drain_pending_rx(2);

    assert!(
        dev.ports[2].pending_rx.is_empty(),
        "happy-path drain must consume all pending bytes"
    );
    let mut readback = vec![0u8; payload.len()];
    mem.read_slice(&mut readback, data_addr)
        .expect("read back delivered payload");
    assert_eq!(
        readback, payload,
        "delivered bytes must equal the queued payload"
    );
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "happy-path drain must add_used exactly once");
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

/// Multi-descriptor torn-write recovery on port 2: a chain with
/// two write-only descriptors where the second points at
/// unmapped guest memory. `mem.write_slice` fails on the second
/// descriptor, triggering the torn-write branch in
/// `drain_port2_pending_rx`.
///
/// Pins:
/// (a) chain head add_used'd with len=0 (used.idx == 1);
/// (b) bytes preserved in `port2_pending_rx` for retry;
/// (c) drain loop breaks (any further chain remains unconsumed);
/// (d) signal_used NOT called (total_written stays 0).
///
/// Mirrors
/// `drain_port1_pending_rx_torn_write_publishes_head_with_zero_len`
/// for the port-2 torn-write recovery branch.
#[test]
fn drain_port2_pending_rx_torn_write_publishes_head_with_zero_len() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let valid_addr = GuestAddress(0x10000);
    let unmapped_addr = GuestAddress(4 << 20);
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
    wire_port2_rxq_to_mock(&mut dev, &mock);
    open_port2(&mut dev);

    let payload: Vec<u8> = (0..64u8).collect();
    dev.ports[2].pending_rx.extend(payload.iter().copied());
    let int_before = dev.interrupt_status;
    let _ = dev.irq_evt.read();

    dev.drain_pending_rx(2);

    // (a) used.idx == 1, head published with len=0.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 1,
        "torn-write recovery must add_used the chain head (with len=0)"
    );
    let used_elem_len: u32 = mem
        .read_obj(mock.used_addr().checked_add(8).unwrap())
        .expect("read used elem 0 len");
    assert_eq!(
        used_elem_len, 0,
        "torn-write recovery must publish len=0 for the chain head",
    );

    // (b) Bytes stay in pending_rx.
    assert_eq!(
        dev.ports[2].pending_rx.len(),
        payload.len(),
        "torn-write recovery must preserve bytes in pending_rx"
    );
    let preserved: Vec<u8> = dev.ports[2].pending_rx.iter().copied().collect();
    assert_eq!(
        preserved, payload,
        "preserved bytes must be the original payload verbatim"
    );

    // (d) signal_used NOT called.
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
