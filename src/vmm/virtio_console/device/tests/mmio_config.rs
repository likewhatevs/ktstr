use super::super::super::*;
use super::*;

#[test]
fn magic_version_device_id() {
    let dev = VirtioConsole::new();
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_MAGIC_VALUE), 0x7472_6976);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_VERSION), 2);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_DEVICE_ID), VIRTIO_ID_CONSOLE);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_VENDOR_ID), 0);
}

#[test]
fn device_features_advertises_multiport_and_v1() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0);
    let lo = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
    write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 1);
    let hi = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
    let features = (hi as u64) << 32 | lo as u64;
    assert_ne!(features & (1 << VIRTIO_F_VERSION_1), 0);
    assert_ne!(features & (1u64 << (VIRTIO_CONSOLE_F_MULTIPORT as u64)), 0);
}

#[test]
fn config_space_max_nr_ports_at_offset_4() {
    let dev = VirtioConsole::new();
    // Read max_nr_ports (u32 at offset 4 inside config space, which
    // starts at 0x100).
    let mut buf = [0u8; 4];
    dev.mmio_read(0x100 + 4, &mut buf);
    assert_eq!(u32::from_le_bytes(buf), NUM_PORTS);
}

#[test]
fn config_space_cols_rows_zero_without_f_size() {
    let dev = VirtioConsole::new();
    let mut buf = [0u8; 4];
    // cols (u16) + rows (u16) at offsets 0..4 inside config.
    dev.mmio_read(0x100, &mut buf);
    assert_eq!(buf, [0, 0, 0, 0]);
}

#[test]
fn queue_num_max_for_six_queues() {
    let mut dev = VirtioConsole::new();
    for q in 0..NUM_QUEUES {
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, q as u32);
        assert_eq!(
            read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX),
            QUEUE_MAX_SIZE as u32,
            "queue {q} should be available",
        );
    }
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, NUM_QUEUES as u32);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX), 0);
}

#[test]
fn queue_ready_requires_features_ok() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_READY), 0);

    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_READY), 1);
}

#[test]
fn status_state_machine() {
    let mut dev = VirtioConsole::new();
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);

    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    assert_eq!(dev.device_status, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    assert_eq!(dev.device_status, S_DRV);

    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    assert_eq!(
        dev.device_status, S_DRV,
        "skip FEATURES_OK must be rejected"
    );

    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    assert_eq!(
        dev.device_status, S_DRV,
        "clearing DRIVER bit must be rejected"
    );
}

#[test]
fn status_reset_via_zero() {
    let mut dev = VirtioConsole::new();
    init_device(&mut dev);
    assert_eq!(dev.device_status, S_OK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
    assert_eq!(dev.device_status, 0);
}

#[test]
fn interrupt_status_and_ack() {
    let mut dev = VirtioConsole::new();
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS), 0);
    dev.interrupt_status = VIRTIO_MMIO_INT_VRING;
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS),
        VIRTIO_MMIO_INT_VRING
    );
}

#[test]
fn interrupt_ack_clears_bits() {
    let mut dev = VirtioConsole::new();
    dev.interrupt_status = VIRTIO_MMIO_INT_VRING | VIRTIO_MMIO_INT_CONFIG;
    write_reg(&mut dev, VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INT_VRING);
    assert_eq!(
        read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS),
        VIRTIO_MMIO_INT_CONFIG
    );
}

#[test]
fn non_4byte_read_returns_ff() {
    let dev = VirtioConsole::new();
    let mut buf = [0u8; 2];
    dev.mmio_read(0, &mut buf);
    assert_eq!(buf, [0xff, 0xff]);
}

#[test]
fn non_4byte_write_ignored() {
    let mut dev = VirtioConsole::new();
    dev.mmio_write(VIRTIO_MMIO_STATUS as u64, &[0x01, 0x00]);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
}

#[test]
fn driver_features_gated_by_status() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xDEAD);
    assert_eq!(dev.driver_features, 0);

    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xDEAD_BEEF);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xCAFE_BABE);
    assert_eq!(dev.driver_features, 0xCAFE_BABE_DEAD_BEEF);
}

#[test]
fn features_rejected_after_features_ok() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);

    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xFFFF);
    assert_eq!(dev.driver_features & 0xFFFF_FFFF, 0);
}

#[test]
fn queue_desc_addr_requires_features_ok() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, 0x1000);
    assert_ne!(dev.queues[0].desc_table(), 0x1000);

    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        &mut dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, 0x1000);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, 0);
    assert_eq!(dev.queues[0].desc_table(), 0x1000);
}

#[test]
fn reset_clears_all_state() {
    let mut dev = VirtioConsole::new();
    init_device(&mut dev);
    dev.interrupt_status = 0xFF;
    dev.ports[0].tx_buf.extend(b"leftover0".iter().copied());
    // VecDeque has no `extend_from_slice`; copy the byte iterator.
    dev.ports[1].tx_buf.extend(b"leftover1".iter().copied());
    dev.ports[2].tx_buf.extend(b"leftover2".iter().copied());
    // Pending-RX buffers are queue-side scratch — populate
    // them so the test verifies reset clears them.
    dev.ports[0].pending_rx.extend(b"pending0".iter().copied());
    dev.ports[1].pending_rx.extend(b"pending1".iter().copied());
    dev.ports[2].pending_rx.extend(b"pending2".iter().copied());
    dev.ports[0].opened = true;
    dev.ports[1].opened = true;
    dev.ports[2].opened = true;
    dev.device_ready = true;
    dev.ports[0].readied = true;
    dev.ports[1].readied = true;
    dev.ports[2].readied = true;
    dev.control_out
        .push_back(ControlOut::Cmd(VirtioConsoleControl {
            id: 0,
            event: VIRTIO_CONSOLE_PORT_ADD,
            value: 0,
        }));

    write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);

    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS), 0);
    assert_eq!(dev.queue_select, 0);
    assert_eq!(dev.device_features_sel, 0);
    assert_eq!(dev.driver_features, 0);
    // Per-port `tx_buf` survives reset by design — these are
    // host-side capture buffers `collect_results` (and the
    // stats client's drainer) consume after the guest's reboot
    // path has clobbered device state. The pre-existing
    // leftover bytes must remain observable.
    assert_eq!(
        dev.ports[0].tx_buf.iter().copied().collect::<Vec<u8>>(),
        b"leftover0",
        "ports[0].tx_buf must survive reset (host-side capture buffer)"
    );
    assert_eq!(
        dev.ports[1].tx_buf.iter().copied().collect::<Vec<u8>>(),
        b"leftover1",
        "ports[1].tx_buf must survive reset (host-side capture buffer)"
    );
    assert_eq!(
        dev.ports[2].tx_buf.iter().copied().collect::<Vec<u8>>(),
        b"leftover2",
        "ports[2].tx_buf must survive reset (host-side capture buffer)"
    );
    // Per-port `pending_rx` is cleared at reset because the
    // guest's RX queues are about to be reset and the bytes
    // would otherwise sit in the deque indefinitely.
    assert!(dev.ports[0].pending_rx.is_empty());
    assert!(dev.ports[1].pending_rx.is_empty());
    assert!(dev.ports[2].pending_rx.is_empty());
    for p in &dev.ports {
        assert!(!p.opened);
    }
    assert!(!dev.device_ready);
    for p in &dev.ports {
        assert!(!p.readied);
    }
    assert!(dev.control_out.is_empty());
}

#[test]
fn config_generation_initially_zero() {
    let dev = VirtioConsole::new();
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_CONFIG_GENERATION), 0);
}

#[test]
fn new_creates_eventfds() {
    let dev = VirtioConsole::new();
    assert!(dev.irq_evt().as_raw_fd() >= 0);
    assert!(dev.tx_evt().as_raw_fd() >= 0);
    assert!(dev.stats_tx_evt().as_raw_fd() >= 0);
    // All three eventfds must be distinct so wakes on one fd
    // do not bleed into another. A regression that aliased
    // `tx_evt` and `stats_tx_evt` would let port-1 bulk
    // traffic wake the stats drainer (and vice versa),
    // defeating the orthogonality the dedicated stats path
    // depends on.
    let irq = dev.irq_evt().as_raw_fd();
    let tx = dev.tx_evt().as_raw_fd();
    let stats = dev.stats_tx_evt().as_raw_fd();
    assert_ne!(irq, tx);
    assert_ne!(irq, stats);
    assert_ne!(tx, stats);
}

#[test]
fn output_empty_initially() {
    let dev = VirtioConsole::new();
    assert!(dev.output_for_test().is_empty());
}

#[test]
fn drain_output_empty() {
    let mut dev = VirtioConsole::new();
    assert!(dev.drain_output().is_empty());
}

#[test]
fn drain_bulk_empty() {
    let mut dev = VirtioConsole::new();
    assert!(dev.drain_bulk().is_empty());
}

#[test]
fn set_mem_stores_reference() {
    let mut dev = VirtioConsole::new();
    assert!(dev.mem.is_none());
    let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
    dev.set_mem(mem);
    assert!(dev.mem.is_some());
}

#[test]
fn queue_input_no_mem_no_panic() {
    let mut dev = VirtioConsole::new();
    dev.queue_input(b"hello");
}

#[test]
fn unknown_register_returns_zero() {
    let dev = VirtioConsole::new();
    assert_eq!(read_reg(&dev, 0x300), 0);
}

#[test]
fn unknown_register_write_ignored() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, 0x300, 0xDEAD);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
}

#[test]
fn invalid_queue_select_returns_zero() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 99);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX), 0);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_READY), 0);
}

#[test]
fn signal_used_sets_interrupt_and_writes_eventfd() {
    let mut dev = VirtioConsole::new();
    assert_eq!(dev.interrupt_status, 0);
    dev.signal_used();
    assert_ne!(dev.interrupt_status & VIRTIO_MMIO_INT_VRING, 0);
    let val = dev.irq_evt.read().unwrap();
    assert!(val > 0);
}

#[test]
fn features_page_2_returns_zero() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 2);
    assert_eq!(read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES), 0);
}

#[test]
fn tx_evt_silent_on_empty_process_tx() {
    let mut dev = VirtioConsole::new();
    let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
    dev.set_mem(mem);
    let _ = dev.process_tx(0);
    let _ = dev.process_tx(1);
    assert!(dev.tx_evt.read().is_err());
}

#[test]
fn status_skip_acknowledge_rejected() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, VIRTIO_CONFIG_S_DRIVER);
    assert_eq!(dev.device_status, 0);
}

#[test]
fn queue_config_rejected_after_driver_ok() {
    let mut dev = VirtioConsole::new();
    init_device(&mut dev);
    assert_eq!(dev.device_status, S_OK);

    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
    write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 64);
    assert_eq!(dev.queues[0].size(), QUEUE_MAX_SIZE);
}

#[test]
fn config_space_write_ignored() {
    let mut dev = VirtioConsole::new();
    // Try to write to max_nr_ports (offset 4 in config space).
    let buf = 99u32.to_le_bytes();
    dev.mmio_write(0x104, &buf);
    // Read back — must still be NUM_PORTS.
    let mut out = [0u8; 4];
    dev.mmio_read(0x104, &mut out);
    assert_eq!(u32::from_le_bytes(out), NUM_PORTS);
}

#[test]
fn handle_device_ready_enqueues_port_adds() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: 0xFFFF_FFFF,
        event: VIRTIO_CONSOLE_DEVICE_READY,
        value: 1,
    });
    assert!(dev.device_ready);
    assert_eq!(dev.control_out.len(), NUM_PORTS as usize);
    for (i, msg) in dev.control_out.iter().enumerate() {
        match msg {
            ControlOut::Cmd(c) => {
                let id = c.id;
                let event = c.event;
                let value = c.value;
                assert_eq!(id, i as u32);
                assert_eq!(event, VIRTIO_CONSOLE_PORT_ADD);
                // PORT_ADD value=1 matches QEMU
                // (hw/char/virtio-serial-bus.c
                // `send_control_event(... PORT_ADD, 1)`).
                assert_eq!(value, 1);
            }
            _ => panic!("unexpected msg variant"),
        }
    }
}

#[test]
fn handle_port_ready_port0_console_announce() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: 0,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    // Order: CONSOLE_PORT, PORT_NAME, PORT_OPEN. PORT_NAME between
    // CONSOLE_PORT and PORT_OPEN keeps the sysfs `name` attribute
    // (`/sys/class/virtio-ports/vport0p0/name`) created before any
    // userspace `/dev/hvc0` open races with udev symlink creation.
    assert_eq!(dev.control_out.len(), 3);
    let m0 = &dev.control_out[0];
    let m1 = &dev.control_out[1];
    let m2 = &dev.control_out[2];
    match m0 {
        ControlOut::Cmd(c) => {
            let event = c.event;
            let value = c.value;
            assert_eq!(event, VIRTIO_CONSOLE_CONSOLE_PORT);
            assert_eq!(value, 1);
        }
        _ => panic!("expected Cmd"),
    }
    match m1 {
        ControlOut::Name { id, name } => {
            assert_eq!(*id, 0);
            assert_eq!(*name, PORT0_NAME);
        }
        _ => panic!("expected Name"),
    }
    match m2 {
        ControlOut::Cmd(c) => {
            let event = c.event;
            let value = c.value;
            assert_eq!(event, VIRTIO_CONSOLE_PORT_OPEN);
            assert_eq!(value, 1);
        }
        _ => panic!("expected Cmd"),
    }
}

#[test]
fn handle_port_ready_port1_name_then_open() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: 1,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    assert_eq!(dev.control_out.len(), 2);
    // Order matches QEMU's PORT_READY handler
    // (hw/char/virtio-serial-bus.c): PORT_NAME first, PORT_OPEN
    // second. Sending OPEN before NAME would race udev symlink
    // creation against the userspace open of /dev/vport0p1.
    match &dev.control_out[0] {
        ControlOut::Name { id, name } => {
            assert_eq!(*id, 1);
            assert_eq!(*name, PORT1_NAME);
        }
        _ => panic!("expected Name"),
    }
    match &dev.control_out[1] {
        ControlOut::Cmd(c) => {
            let event = c.event;
            let value = c.value;
            let id = c.id;
            assert_eq!(id, 1);
            assert_eq!(event, VIRTIO_CONSOLE_PORT_OPEN);
            assert_eq!(value, 1);
        }
        _ => panic!("expected Cmd"),
    }
}

#[test]
fn handle_port_open_tracks_state() {
    let mut dev = VirtioConsole::new();
    assert!(!dev.ports[1].opened);
    dev.handle_control_event(VirtioConsoleControl {
        id: 1,
        event: VIRTIO_CONSOLE_PORT_OPEN,
        value: 1,
    });
    assert!(dev.ports[1].opened);
    dev.handle_control_event(VirtioConsoleControl {
        id: 1,
        event: VIRTIO_CONSOLE_PORT_OPEN,
        value: 0,
    });
    assert!(!dev.ports[1].opened);
}

#[test]
fn handle_port_ready_unknown_port_ignored() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: 99,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    assert!(dev.control_out.is_empty());
}

#[test]
fn vc_control_size_is_eight_bytes() {
    assert_eq!(VC_CONTROL_SIZE, 8);
    assert_eq!(std::mem::size_of::<VirtioConsoleControl>(), 8);
}

#[test]
fn vc_control_round_trip_through_bytes() {
    let c = VirtioConsoleControl {
        id: 0xDEAD_BEEF,
        event: VIRTIO_CONSOLE_PORT_OPEN,
        value: 1,
    };
    let bytes = c.as_bytes();
    let back = VirtioConsoleControl::read_from_bytes(bytes).unwrap();
    let id = back.id;
    let event = back.event;
    let value = back.value;
    assert_eq!(id, 0xDEAD_BEEF);
    assert_eq!(event, VIRTIO_CONSOLE_PORT_OPEN);
    assert_eq!(value, 1);
}
