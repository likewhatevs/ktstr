use super::super::super::*;
use super::*;

// ----------------------------------------------------------------
// Chain-level MockSplitQueue tests for the port 1 TX path.
//
// These exercise `process_tx(port_id)` for the port-1 TX path
// end-to-end via the real virtio-queue descriptor walker —
// MockSplitQueue plants a chain in guest memory, MMIO QUEUE_NOTIFY
// fires, and the device's process_tx walks the chain, copies
// device-readable descriptor data into `ports[1].tx_buf`, and
// add_useds the chain.
//
// The handler-level tests above bypass the queue walker and only
// pin MMIO/FSM/control surface; these tests pin the production
// bulk TX path the host-side `bulk_drain` consumer depends on.
// Chain-level MockSplitQueue coverage is mandatory because the
// chain-parsing logic is the highest-risk code on the bulk path:
// a hostile or malformed chain that handler-level tests can't
// construct (multi-segment, mixed-direction, length-cap edges)
// is exactly what the production `process_tx` walker has
// to reject without panicking. Without these chain tests every
// chain-shape regression has to wait for an end-to-end VM run
// to surface.
// ----------------------------------------------------------------

/// Single-descriptor TX chain on port 1: one device-readable
/// segment with a known byte pattern lands verbatim in
/// `ports[1].tx_buf`. Pins the simplest happy-path: chain pop →
/// non-write_only branch → `mem.read_slice` → append to
/// `ports[1].tx_buf` → add_used → `signal_used`. `drain_bulk()`
/// returns the bytes to confirm the routing accumulator was the
/// port-1 buffer (not port 0).
#[test]
fn port1_tx_single_descriptor_lands_in_port1_buf() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let payload = b"port1 single descriptor TX";
    mem.write_slice(payload, data_addr).expect("plant payload");
    // Single device-readable descriptor (flags=0 → guest wrote
    // the bytes; device reads them).
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        payload.len() as u32,
        0,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_console_queue_to_mock(&mut dev, &mock, PORT1_TXQ as u32);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

    // Bytes must have landed in ports[1].tx_buf, observable via
    // drain_bulk.
    let drained = dev.drain_bulk();
    assert_eq!(
        drained,
        payload.to_vec(),
        "port 1 TX must deliver the descriptor's bytes to drain_bulk verbatim",
    );
    // Port 0 buffer must be untouched — process_tx was called
    // with port_id=1 (resolved from PORT1_TXQ by queue_to_port in
    // mmio_write), steering the bytes to ports[1].tx_buf.
    assert!(
        dev.drain_output().is_empty(),
        "port 0 TX buffer must remain empty when only port 1 was notified",
    );
    // Used ring reflects exactly one completion. UsedRing.idx
    // sits at used_addr+2 after the 2-byte flags field
    // (virtio-v1.2 §2.7.8).
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "exactly one used-ring entry expected");
    // signal_used must have set INT_VRING in interrupt_status
    // and written the irq eventfd.
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

/// c_ovq (control-TX, guest->host) chain: the published used-element
/// `len` must be 0, not the bytes-consumed count. virtio v1.2
/// §2.7.8.2: used `len` counts bytes written to the device-WRITABLE
/// portion; a c_ovq buffer is device-readable, so the spec-pure len
/// is 0. Before the fix `process_control_tx` published `total` (bytes
/// read), so seeding a readable chain with VC_CONTROL_SIZE bytes
/// makes this assertion fail on the old behaviour — pinning the
/// regression. Chain-level: drives the real `process_control_tx` via
/// QUEUE_NOTIFY on C_OVQ (wire_console_queue_to_mock ends at S_OK, so
/// the DRIVER_OK gate is satisfied).
#[test]
fn control_tx_publishes_zero_used_len() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    // A full VirtioConsoleControl (id, event, value = 8 bytes) so the
    // chain carries VC_CONTROL_SIZE device-readable bytes → pre-fix
    // `total` would be 8.
    let ctrl: [u8; VC_CONTROL_SIZE] = [2, 0, 0, 0, 6, 0, 1, 0];
    mem.write_slice(&ctrl, data_addr)
        .expect("plant control msg");
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        VC_CONTROL_SIZE as u32,
        0,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_console_queue_to_mock(&mut dev, &mock, C_OVQ as u32);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, C_OVQ as u32);

    // Exactly one completion.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(used_idx, 1, "c_ovq chain must be add_used'd exactly once");
    // The used-element `len` MUST be 0 (device-readable control-TX
    // buffer). used-ring layout: used_addr + 4 (flags+idx) + 4 (id) =
    // +8 for the first entry's `len`. Pre-fix this published `total`
    // (VC_CONTROL_SIZE = 8).
    let used_elem_len: u32 = mem
        .read_obj(mock.used_addr().checked_add(8).unwrap())
        .expect("read used elem 0 len");
    assert_eq!(
        used_elem_len, 0,
        "c_ovq add_used len must be 0 (device-readable control-TX \
         buffer, virtio v1.2 §2.7.8.2); pre-fix it published the \
         bytes-consumed total",
    );
}

/// Multi-descriptor TX chain: four 4 KiB device-readable
/// segments concatenate in `port1_tx_buf`, in chain order. Pins
/// that the descriptor-walker honours the chain order and that
/// the per-descriptor `dst.resize + read_slice` append logic
/// preserves byte boundaries. The kernel's virtio-console driver
/// sends PAGE_SIZE chunks as multi-segment chains under high
/// volume — losing chain order or dropping the second-N segments
/// would corrupt the host-side TLV stream.
#[test]
fn port1_tx_multi_descriptor_chain_concatenates() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    // PAGE_SIZE matches the kernel's typical TX descriptor size
    // (drivers/char/virtio_console.c uses PAGE_SIZE buffers).
    // Four distinct GPAs spaced 8 KiB apart so each descriptor
    // points at a non-overlapping region.
    const PAGE: u32 = 4096;
    let segs: [(GuestAddress, u8); 4] = [
        (GuestAddress(0x10000), 0xA1),
        (GuestAddress(0x12000), 0xA2),
        (GuestAddress(0x14000), 0xA3),
        (GuestAddress(0x16000), 0xA4),
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
    wire_console_queue_to_mock(&mut dev, &mock, PORT1_TXQ as u32);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

    // Expected: 4 × PAGE concatenation, segment N filled with
    // 0xA(N+1).
    let drained = dev.drain_bulk();
    assert_eq!(
        drained.len(),
        4 * PAGE as usize,
        "drain_bulk length must equal sum of segment lengths",
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
    // Used ring reflects exactly one completion — the chain is
    // a single descriptor chain (head + 3 next links), not 4
    // separate chains.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 1,
        "one chain → one used-ring entry, regardless of segment count",
    );
}

/// Oversize descriptor truncates to `TX_DESC_MAX`. A guest that
/// publishes `len > 32 KiB` (max u32 = 4 GiB worst case) is
/// hostile or buggy; the device caps each descriptor at 32 KiB
/// to bound the per-chain heap allocation. Pins the
/// `(desc.len() as usize).min(TX_DESC_MAX)` clamp in `process_tx`
/// (src/vmm/virtio_console/device/mod.rs:205) and the
/// `tx_scratch.resize(dlen, 0)` staging buffer. Without the cap, a single bogus
/// descriptor could trigger a multi-GiB Vec allocation.
#[test]
fn port1_tx_oversize_descriptor_truncates_to_tx_desc_max() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    // Plant a buffer twice TX_DESC_MAX so the truncation can be
    // distinguished from "no data planted past the cap." First
    // half filled with 0x55, second half with 0x99 — the cap
    // should keep only the 0x55 region.
    const OVERSIZE: usize = TX_DESC_MAX * 2;
    let mut payload = vec![0x55u8; TX_DESC_MAX];
    payload.extend_from_slice(&vec![0x99u8; TX_DESC_MAX]);
    assert_eq!(payload.len(), OVERSIZE);
    mem.write_slice(&payload, data_addr).expect("plant payload");
    // One descriptor with len=OVERSIZE > TX_DESC_MAX. The
    // device must cap reads at TX_DESC_MAX.
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        OVERSIZE as u32,
        0,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());
    wire_console_queue_to_mock(&mut dev, &mock, PORT1_TXQ as u32);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

    let drained = dev.drain_bulk();
    // Length pinned to TX_DESC_MAX, not OVERSIZE.
    assert_eq!(
        drained.len(),
        TX_DESC_MAX,
        "oversize descriptor (len > TX_DESC_MAX) must truncate to TX_DESC_MAX",
    );
    // The bytes that DID land must be the first TX_DESC_MAX of
    // the planted buffer (0x55 fill); a regression that read
    // past the cap would surface 0x99 bytes here.
    assert!(
        drained.iter().all(|&b| b == 0x55),
        "truncated bytes must be the FIRST TX_DESC_MAX bytes \
         of the descriptor (0x55), not anything past the cap",
    );
}

/// Port 1 TX with DRIVER_OK NOT set: the device must drop the
/// notify silently. Pins the spec gate at the head of
/// `process_tx`: virtio-v1.2 §3.1.1 forbids the device
/// from accessing virtqueue memory before DRIVER_OK because
/// queue addresses written during FEATURES_OK are not yet
/// committed by the driver. A regression that lifted the gate
/// would let the device walk a queue the guest hasn't fully
/// validated.
///
/// Setup walks the FSM only to FEATURES_OK + queue ready (NOT
/// DRIVER_OK), plants a chain, fires the notify, and verifies
/// (a) port1_tx_buf untouched (drain_bulk empty), (b) used.idx
/// untouched (no add_used), (c) interrupt_status untouched (no
/// signal_used), (d) irq_evt unwritten.
#[test]
fn port1_tx_rejected_without_driver_ok() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    let data_addr = GuestAddress(0x10000);
    let payload = b"this must NOT reach port1_tx_buf";
    mem.write_slice(payload, data_addr).expect("plant payload");
    let descs = [RawDescriptor::from(SplitDescriptor::new(
        data_addr.0,
        payload.len() as u32,
        0,
        0,
    ))];
    mock.build_desc_chain(&descs).expect("build chain");
    dev.set_mem(mem.clone());

    // Walk FSM only up to S_FEAT — STOP before S_OK. Configure
    // the queue and mark it ready (which is allowed in the
    // S_FEAT..S_OK window per `queue_config_allowed`). The
    // process_tx gate should still drop the notify because
    // VIRTIO_CONFIG_S_DRIVER_OK is not in device_status.
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
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT1_TXQ as u32);
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
    // Precondition: device_status reflects FEATURES_OK but NOT
    // DRIVER_OK; the queue is ready.
    assert_eq!(
        dev.device_status & VIRTIO_CONFIG_S_DRIVER_OK,
        0,
        "precondition: DRIVER_OK must NOT be set",
    );
    assert!(
        dev.queues[PORT1_TXQ].ready(),
        "precondition: port 1 TX queue must be ready (the gate \
         we are testing is the DRIVER_OK gate, not a not-ready \
         gate)",
    );

    // Fire the notify. The DRIVER_OK gate at the top of
    // process_tx must early-return with no observable side
    // effects.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

    // No bytes must have landed in port1_tx_buf.
    assert!(
        dev.drain_bulk().is_empty(),
        "port1_tx_buf must remain empty — DRIVER_OK gate must \
         reject pre-DRIVER_OK notify",
    );
    // used.idx must remain 0 — no add_used.
    let used_idx: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx");
    assert_eq!(
        used_idx, 0,
        "used.idx must be 0 — DRIVER_OK gate must skip add_used",
    );
    // interrupt_status must remain 0 — no signal_used.
    assert_eq!(
        dev.interrupt_status, 0,
        "interrupt_status must be 0 — DRIVER_OK gate must skip signal_used",
    );
    // irq_evt counter must be drained / never written. A
    // non-blocking read on an unwritten EFD_NONBLOCK eventfd
    // returns WouldBlock.
    match dev.irq_evt.read() {
        Ok(n) => panic!("irq_evt must NOT have been written, but read returned {n}"),
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
        Err(e) => panic!("unexpected irq_evt read error: {e}"),
    }
}

/// Port 0 vs port 1 TX routing: bytes from port 0 land in
/// port0_tx_buf and bytes from port 1 land in port1_tx_buf.
/// Pins the queue_idx→port_id map (`queue_to_port`) that routes
/// QUEUE_NOTIFY to `process_tx(port_id)`, which appends to
/// `self.ports[port_id].tx_buf`. A regression
/// that swapped the buffers would corrupt the host-side
/// stdout stream with TLV bytes (or vice versa) — neither
/// surface would parse correctly.
///
/// Two MockSplitQueues are used at distinct GPAs because each
/// queue needs its own desc/avail/used rings. Port 0 mock at
/// GPA 0x0; port 1 mock at GPA 0x1000 (4 KiB above port 0,
/// which keeps a comfortable margin: a queue of size 16 uses
/// roughly 16*16 + 4 + 16*2 + 2 + 16*8 + 6 ≈ 432 bytes, well
/// under one 4 KiB page).
#[test]
fn port0_tx_vs_port1_tx_routes_to_correct_buffer() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    let mock0 = MockSplitQueue::create(&mem, GuestAddress(0x0), 16);
    let mock1 = MockSplitQueue::create(&mem, GuestAddress(0x1000), 16);
    let port0_data_addr = GuestAddress(0x10000);
    let port1_data_addr = GuestAddress(0x20000);
    let port0_payload = b"port0 console bytes";
    let port1_payload = b"port1 bulk TLV bytes";
    mem.write_slice(port0_payload, port0_data_addr)
        .expect("plant port0 payload");
    mem.write_slice(port1_payload, port1_data_addr)
        .expect("plant port1 payload");

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
    mock0
        .build_desc_chain(&port0_descs)
        .expect("build port0 chain");
    mock1
        .build_desc_chain(&port1_descs)
        .expect("build port1 chain");
    dev.set_mem(mem.clone());

    // Walk FSM up to FEATURES_OK, configure BOTH queues, then
    // transition to DRIVER_OK. We can't reuse
    // `wire_console_queue_to_mock` for two queues (it wraps a
    // single FSM walk); inline the multi-queue version here.
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
    // Configure port 0 TX queue (idx 1).
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT0_TXQ as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
    let d0 = mock0.desc_table_addr().0;
    let a0 = mock0.avail_addr().0;
    let u0 = mock0.used_addr().0;
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, d0 as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (d0 >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, a0 as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (a0 >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, u0 as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (u0 >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
    // Configure port 1 TX queue (idx 5).
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, PORT1_TXQ as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 16);
    let d1 = mock1.desc_table_addr().0;
    let a1 = mock1.avail_addr().0;
    let u1 = mock1.used_addr().0;
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, d1 as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (d1 >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, a1 as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (a1 >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_LOW, u1 as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (u1 >> 32) as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    assert_eq!(
        dev.device_status, S_OK,
        "FSM did not reach DRIVER_OK after both queues configured",
    );

    // Notify port 0 then port 1; observe the routing.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT0_TXQ as u32);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

    // Port 0 bytes must land in port0_tx_buf, observable via
    // drain_output (which returns port0_tx_buf).
    let port0_drained = dev.drain_output();
    assert_eq!(
        port0_drained,
        port0_payload.to_vec(),
        "port 0 TX bytes must route to port0_tx_buf — drain_output \
         returns port0 bytes verbatim",
    );
    // Port 1 bytes must land in port1_tx_buf, observable via
    // drain_bulk (which returns port1_tx_buf).
    let port1_drained = dev.drain_bulk();
    assert_eq!(
        port1_drained,
        port1_payload.to_vec(),
        "port 1 TX bytes must route to port1_tx_buf — drain_bulk \
         returns port1 bytes verbatim",
    );
    // Each port's used.idx reflects exactly one completion.
    let port0_used_idx: u16 = mem
        .read_obj(mock0.used_addr().checked_add(2).unwrap())
        .expect("read port0 used.idx");
    let port1_used_idx: u16 = mem
        .read_obj(mock1.used_addr().checked_add(2).unwrap())
        .expect("read port1 used.idx");
    assert_eq!(
        port0_used_idx, 1,
        "port 0 used.idx must reflect 1 completion"
    );
    assert_eq!(
        port1_used_idx, 1,
        "port 1 used.idx must reflect 1 completion"
    );
}

/// Port 1 TX `process_tx` honours the per-call cumulative
/// byte cap (`TX_PER_CALL_MAX`). With 9 chains of `TX_DESC_MAX`
/// bytes each (= 9 × 32 KiB = 288 KiB), the first 8 chains
/// drain to `port1_tx_buf` (cumulative 256 KiB hits the cap)
/// and the 9th chain remains unconsumed in the avail ring for
/// the next notify. Pins the per-call drain cap that bounds
/// the per-vCPU MMIO-handler latency budget against a hostile
/// guest publishing thousands of valid PAGE_SIZE chains.
///
/// Verifies:
/// (a) `port1_tx_buf` length == 8 × TX_DESC_MAX (cap drain only);
/// (b) `used.idx` == 8 (only the 8 drained chains were add_used);
/// (c) the 9th descriptor's bytes are NOT in `port1_tx_buf`
///     (each chain's payload uses a distinguishable fill byte
///     so a regression that drained past the cap surfaces here);
/// (d) a second `QUEUE_NOTIFY` drains the remaining 9th chain
///     (cumulative 32 KiB on the second call, well under cap).
#[test]
fn port1_tx_per_call_cap_partial_drain() {
    let mut dev = VirtioConsole::new();
    let mem = make_chain_test_mem();
    // Queue size 16: enough headroom for 9 standalone chains
    // (the test's published count) without wrap-around.
    let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);
    // Buffer count: 9 chains × TX_DESC_MAX bytes = 288 KiB
    // total. With TX_PER_CALL_MAX = 256 KiB, 8 chains hit the
    // cap (cumulative 256 KiB after the 8th chain triggers the
    // break check). The 9th must stay unconsumed.
    const N_CHAINS: usize = 9;
    // Per-chain GPA spacing: each buffer is TX_DESC_MAX (32
    // KiB) bytes, placed at 0x10000, 0x18000, 0x20000, ...
    // (8 KiB stride between starts is too tight; use
    // TX_DESC_MAX stride so buffers don't overlap). Total
    // memory usage: 0x10000 + 9 × TX_DESC_MAX = 64 KiB + 288
    // KiB = 352 KiB, well under the 2 MiB test memory map.
    let mut descs: Vec<RawDescriptor> = Vec::with_capacity(N_CHAINS);
    for i in 0..N_CHAINS {
        let buf_addr = GuestAddress(0x10000 + (i as u64) * (TX_DESC_MAX as u64));
        // Fill byte i+1 (1..=9) so a regression that drained
        // past the cap surfaces an unexpected fill byte. Using
        // (i+1) instead of i keeps every chain's fill byte
        // non-zero so a buffer of zeros from an unrelated
        // region can't pass for chain bytes.
        let fill = (i + 1) as u8;
        let payload = vec![fill; TX_DESC_MAX];
        mem.write_slice(&payload, buf_addr)
            .expect("plant per-chain payload");
        descs.push(RawDescriptor::from(SplitDescriptor::new(
            buf_addr.0,
            TX_DESC_MAX as u32,
            0, // flags: no NEXT — each desc is its own chain
            0,
        )));
    }
    // `add_desc_chains` writes descs at indices 0..N and
    // increments avail.idx for each chain head (every desc
    // here is a standalone chain because none carry the NEXT
    // flag).
    mock.add_desc_chains(&descs, 0)
        .expect("publish 9 standalone chains");
    dev.set_mem(mem.clone());
    wire_console_queue_to_mock(&mut dev, &mock, PORT1_TXQ as u32);

    // First notify: cap drains 8 chains (256 KiB), 9th left.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

    let drained_first = dev.drain_bulk();
    assert_eq!(
        drained_first.len(),
        TX_PER_CALL_MAX,
        "first notify must drain exactly TX_PER_CALL_MAX bytes \
         (8 × TX_DESC_MAX) — the per-call cap stops popping after \
         the 8th chain"
    );
    // First 8 chains drained: bytes are 32 KiB of fill=1 then
    // 32 KiB of fill=2 ... up to 32 KiB of fill=8. The 9th
    // chain (fill=9) must NOT appear.
    for i in 0..8 {
        let start = i * TX_DESC_MAX;
        let end = start + TX_DESC_MAX;
        let expected_fill = (i + 1) as u8;
        assert!(
            drained_first[start..end]
                .iter()
                .all(|&b| b == expected_fill),
            "chain {i} bytes must be fill={expected_fill}; \
             a regression that drained past the cap (or out of \
             chain order) would surface a different byte here"
        );
    }
    assert!(
        !drained_first.contains(&9u8),
        "9th chain (fill=9) must NOT appear in the first drain \
         — the per-call cap must hold the 9th chain back"
    );
    // used.idx must reflect exactly 8 completions. The 9th
    // chain stayed in the avail ring; the device never
    // add_used'd it.
    let used_idx_first: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx after first notify");
    assert_eq!(
        used_idx_first, 8,
        "used.idx must be 8 after first notify — the cap stopped \
         popping after the 8th chain so only 8 add_used calls \
         happened"
    );

    // Second notify: drains the remaining 9th chain.
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, PORT1_TXQ as u32);

    let drained_second = dev.drain_bulk();
    assert_eq!(
        drained_second.len(),
        TX_DESC_MAX,
        "second notify must drain the remaining 9th chain \
         (TX_DESC_MAX bytes) — the cap is per-call, not per-run"
    );
    assert!(
        drained_second.iter().all(|&b| b == 9u8),
        "second drain must contain the 9th chain's bytes (fill=9)"
    );
    let used_idx_second: u16 = mem
        .read_obj(mock.used_addr().checked_add(2).unwrap())
        .expect("read used.idx after second notify");
    assert_eq!(
        used_idx_second, 9,
        "used.idx must be 9 after second notify — every chain \
         eventually drains, the cap only spreads them across \
         multiple notifies"
    );
}
