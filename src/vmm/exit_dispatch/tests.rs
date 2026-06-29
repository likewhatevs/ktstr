use super::*;

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_io_out_i8042_reset_is_shutdown_signal() {
    // The BSP relies on I8042 reset (port 0x64, 0xFE) for shutdown
    // detection instead of VcpuExit::Hlt. Verify that dispatch_io_out
    // returns true for the reset command.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    assert!(
        dispatch_io_out(&com1, &com2, None,I8042_CMD_PORT, &[I8042_CMD_RESET_CPU]),
        "I8042 reset (0xFE to port 0x64) must signal shutdown"
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_io_out_i8042_non_reset() {
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    assert!(!dispatch_io_out(&com1, &com2, None,I8042_CMD_PORT, &[0x00]));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_io_out_serial_com1() {
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    // Write 'A' to COM1 THR — should not trigger reset.
    assert!(!dispatch_io_out(&com1, &com2, None,console::COM1_BASE, b"A"));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_io_out_serial_com2() {
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    assert!(!dispatch_io_out(&com1, &com2, None,console::COM2_BASE, b"B"));
    let output = com2.lock().output();
    assert!(output.contains('B'));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_io_out_unknown_port() {
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    assert!(!dispatch_io_out(&com1, &com2, None,0x1234, &[0xFF]));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_io_in_i8042_status() {
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut data = [0xFFu8; 1];
    dispatch_io_in(&com1, &com2, None,I8042_CMD_PORT, &mut data);
    assert_eq!(data[0], 0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_io_in_i8042_data() {
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut data = [0xFFu8; 1];
    dispatch_io_in(&com1, &com2, None,I8042_DATA_PORT, &mut data);
    assert_eq!(data[0], 0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_io_in_unknown_port() {
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    // Seed a NON-0xFF value so the test proves the dispatch actively writes the
    // "no device responded" 0xFF (rather than leaving stale KVM_RUN bytes).
    let mut data = [0x42u8; 2];
    dispatch_io_in(&com1, &com2, None, 0x1234, &mut data);
    assert_eq!(
        data,
        [0xFF, 0xFF],
        "unknown IN port must return all-ones, not stale buffer bytes",
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn classify_exit_io_out_i8042_reset_is_shutdown() {
    // Cross-checks the dispatch_io_out i8042-reset path through
    // the classify_exit dispatch table — verifies the "true"
    // return from dispatch_io_out maps to ExitAction::Shutdown.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let data = [I8042_CMD_RESET_CPU];
    let mut exit = VcpuExit::IoOut(I8042_CMD_PORT, &data);
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Shutdown)),
        "IoOut(0x64, [0xFE]) — i8042 reset — must classify as Shutdown"
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn classify_exit_io_out_serial_is_continue() {
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let data = [b'Z'];
    let mut exit = VcpuExit::IoOut(console::COM1_BASE, &data);
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Continue)),
        "IoOut to COM1 must classify as Continue (no reboot)"
    );
    // Confirm the byte landed in COM1's output buffer — pins that
    // the dispatch wired to the right port mutex, not COM2.
    assert!(com1.lock().output().contains('Z'));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn classify_exit_io_in_serial_is_continue() {
    // IoIn on COM1's data port returns a buffered byte if pending,
    // 0 otherwise. classify_exit must map IoIn → Continue
    // unconditionally — the run loop never terminates on a port
    // read.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut data = [0xFFu8; 1];
    let mut exit = VcpuExit::IoIn(console::COM1_BASE, &mut data);
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Continue)),
        "IoIn to COM1 must classify as Continue"
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn classify_exit_x86_mmio_read_unmapped_returns_0xff() {
    // x86_64 MMIO read fallback: when a guest MMIO read addresses
    // a region that is NOT in any of the virtio MMIO windows, the
    // dispatch fills the data buffer with 0xFF (the canonical
    // "no device responded" pattern — matches PCI/MMIO DECODE
    // ERROR behaviour on real hardware). Pinning here so a future
    // refactor cannot accidentally leave the buffer untouched
    // (which would surface as the previous KVM_RUN's stale data
    // appearing in the guest as a phantom value).
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    // Pick an address well below any virtio_*_MMIO_BASE so no
    // device window matches it. 0x1000 is in the BIOS-EBDA area,
    // which ktstr does not back with any MMIO device.
    let mut buf = [0u8; 4];
    let mut exit = VcpuExit::MmioRead(0x1000, &mut buf);
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Continue)),
        "Unmapped MMIO read must classify as Continue (not Fatal)"
    );
    assert_eq!(
        buf,
        [0xff, 0xff, 0xff, 0xff],
        "Unmapped MMIO read must fill the data buffer with 0xFF — \
         leaving stale bytes would surface as phantom guest reads"
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn classify_exit_x86_mmio_write_unmapped_is_continue() {
    // x86_64 MMIO write fallback: an unmapped write is silently
    // dropped (no device matches → control falls through). Pin
    // that the action is Continue so a future change that
    // accidentally classifies "no virtio match" as Fatal would
    // break this test.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let data = [0xAAu8, 0xBB];
    let mut exit = VcpuExit::MmioWrite(0x1000, &data);
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Continue)),
        "Unmapped MMIO write must classify as Continue"
    );
}

// ---------------------------------------------------------------------------
// PCI type-1 CAM (0xCF8/0xCFC) routing + ACPI PM-block stubs.
//
// The PciBus config decode is unit-tested in pci/mod.rs and the PM stub
// values are asserted here; these tests pin the exit_dispatch port -> bus
// routing + PM-block stub wiring, which the booted-guest e2es exercise but
// SKIP-as-PASS without KVM.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn test_pci_bus() -> PiMutex<pci::PciBus> {
    // ECAM base/size are irrelevant to the type-1 CAM (port 0xCF8/0xCFC) path
    // exercised here; any single-bus window works. The host bridge is
    // installed at 00:00.0 by PciBus::new.
    PiMutex::new(pci::PciBus::new(0xE000_0000, pci::ECAM_BYTES_PER_BUS))
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_cam_latch_then_data_reads_host_bridge_id() {
    let bus = test_pci_bus();
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    // CONFIG_ADDRESS = enable(31) | bus0 | dev0 | fn0 | reg0 = 0x8000_0000.
    assert!(!dispatch_io_out(
        &com1,
        &com2,
        Some(&bus),
        PCI_CONFIG_ADDRESS,
        &0x8000_0000u32.to_le_bytes(),
    ));
    // CONFIG_ADDRESS reads back the latched value (pci_check_type1 confirm).
    let mut addr = [0u8; 4];
    dispatch_io_in(&com1, &com2, Some(&bus), PCI_CONFIG_ADDRESS, &mut addr);
    assert_eq!(u32::from_le_bytes(addr), 0x8000_0000);
    // CONFIG_DATA (reg 0) reads the host-bridge vendor/device id. The wire
    // values 0x8086 / 0x0D57 match what both reference VMMs expose at 00:00.0.
    let mut data = [0u8; 4];
    dispatch_io_in(&com1, &com2, Some(&bus), PCI_CONFIG_DATA, &mut data);
    let id = u32::from_le_bytes(data);
    assert_eq!(id & 0xFFFF, 0x8086, "host-bridge vendor id via CAM");
    assert_eq!(id >> 16, 0x0D57, "host-bridge device id via CAM");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_cam_subdword_config_address_does_not_latch() {
    let bus = test_pci_bus();
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    // Latch a known full-dword address.
    dispatch_io_out(
        &com1,
        &com2,
        Some(&bus),
        PCI_CONFIG_ADDRESS,
        &0x8000_0000u32.to_le_bytes(),
    );
    // A sub-dword (len != 4) write to 0xCF8 must NOT change the latch — the
    // guest's pci_check_type1 outb(0x01, 0xCFB) is a legacy no-op the
    // 4-byte gate drops. A regression dropping that gate would corrupt the
    // latch from a sub-dword poke.
    dispatch_io_out(&com1, &com2, Some(&bus), PCI_CONFIG_ADDRESS, &[0x01]);
    let mut addr = [0u8; 4];
    dispatch_io_in(&com1, &com2, Some(&bus), PCI_CONFIG_ADDRESS, &mut addr);
    assert_eq!(
        u32::from_le_bytes(addr),
        0x8000_0000,
        "sub-dword CONFIG_ADDRESS write must not re-latch",
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_pm1_cnt_reports_sci_en() {
    // PM block stubs are pci-independent (the guard sits outside the
    // pci_bus block), so they answer with pci_bus=None — the universal
    // ACPI-enable path every guest takes.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut data = [0xFFu8; 2];
    dispatch_io_in(&com1, &com2, None, kvm::ACPI_PM1_CNT_PORT, &mut data);
    assert_eq!(
        u16::from_le_bytes(data),
        0x0001,
        "PM1_CNT must report SCI_EN (bit0) set",
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_pm1_evt_reads_zero_and_write_is_noop() {
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut data = [0xFFu8; 4];
    dispatch_io_in(&com1, &com2, None, kvm::ACPI_PM1_EVT_PORT, &mut data);
    assert_eq!(data, [0, 0, 0, 0], "PM1 status/enable read 0 (no event armed)");
    // A write is a no-op and must NOT signal shutdown.
    assert!(!dispatch_io_out(
        &com1,
        &com2,
        None,
        kvm::ACPI_PM1_EVT_PORT,
        &[0xFF; 4],
    ));
    let mut after = [0xFFu8; 4];
    dispatch_io_in(&com1, &com2, None, kvm::ACPI_PM1_EVT_PORT, &mut after);
    assert_eq!(after, [0, 0, 0, 0], "PM1_EVT still 0 after a write");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dispatch_pm_timer_is_24bit_and_advances() {
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut t = [0xFFu8; 4];
    dispatch_io_in(&com1, &com2, None, kvm::ACPI_PM_TMR_PORT, &mut t);
    assert!(
        u32::from_le_bytes(t) <= 0x00FF_FFFF,
        "PM timer is a 24-bit value (TMR_VAL_EXT clear)",
    );
    // acpi_pm_timer_value is a wrapping 24-bit upcounter that ADVANCES (it is
    // NOT globally monotonic — it wraps every ~4.6s, like a real ACPI PM
    // timer). Bounded busy sampling — no sleep; the value changes within
    // microseconds, the cap is a stuck-timer backstop.
    // The liveness property is that the counter MOVES (a stuck PM timer trips
    // the guest clocksource watchdog). Spin until it changes, bounded — no
    // strict per-iteration monotonic assert, since the 24-bit counter wraps
    // every ~4.6s and a loop stalled across a wrap would see a legitimate
    // decrease; clock_gettime itself is monotonic, this only samples it.
    let v0 = acpi_pm_timer_value();
    let mut advanced = false;
    for _ in 0..1_000_000u32 {
        if acpi_pm_timer_value() != v0 {
            advanced = true;
            break;
        }
    }
    assert!(advanced, "PM timer must advance (moving liveness counter)");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn acpi_pm_port_claims_only_advertised_blocks() {
    // EVT 0x600..0x604, CNT 0x604..0x606, TMR 0x608..0x60C are advertised;
    // the 0x606-0x607 gap and 0x60C are NOT (PM1_CNT_LEN=2).
    for p in [
        kvm::ACPI_PM1_EVT_PORT,
        0x603,
        kvm::ACPI_PM1_CNT_PORT,
        0x605,
        kvm::ACPI_PM_TMR_PORT,
        0x60B,
    ] {
        assert!(acpi_pm_port(p), "{p:#x} is an advertised PM register port");
    }
    for p in [0x5FFu16, 0x606, 0x607, 0x60C] {
        assert!(!acpi_pm_port(p), "{p:#x} is NOT an advertised PM register port");
    }
}
