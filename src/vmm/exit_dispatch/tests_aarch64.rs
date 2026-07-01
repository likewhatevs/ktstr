use super::*;

// -- aarch64 dispatch_mmio_read / dispatch_mmio_write coverage --
//
// dispatch_mmio_write is unit-typed (always returns) — the aarch64
// shutdown signal arrives via VcpuExit::SystemEvent (PSCI), not an
// MMIO write. dispatch_mmio_read fills with 0xFF when no device
// window matches, so the guest sees a "no device responded" pattern
// rather than stale bytes.

#[test]
fn dispatch_mmio_read_unmapped_returns_0xff() {
    // Pin the canonical "no device responded" fill at 0xFF for
    // unmapped MMIO reads. A future regression that left the
    // buffer untouched would surface as the previous KVM_RUN's
    // stale data appearing as a phantom guest read.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut buf = [0u8; 4];
    // 0x10_0000 is below the device MMIO window (SERIAL_MMIO_BASE) and
    // inside no virtio device window — picked so no early-return matches.
    dispatch_mmio_read(
        &com1, &com2, None, None, None, /* addr */ 0x10_0000, &mut buf,
    );
    assert_eq!(
        buf,
        [0xff, 0xff, 0xff, 0xff],
        "Unmapped MMIO read must fill the data buffer with 0xFF"
    );
}

#[test]
fn dispatch_mmio_read_serial_does_not_fill_0xff() {
    // Reads inside the SERIAL_MMIO_BASE window go through the COM1
    // mutex — the 0xFF fallback fill must NOT run. Verifies the
    // serial branch is taken (data[0] is whatever the LSR returns;
    // we only assert "not 0xFF" because the precise LSR value is
    // a console-internal detail and not what this test pins).
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut buf = [0xAAu8; 1];
    // SERIAL_MMIO_BASE + 5 = LSR offset on the 8250 register map.
    // The serial branch overwrites buf[0] with the LSR value;
    // that value is non-0xFF for an idle Serial.
    dispatch_mmio_read(
        &com1,
        &com2,
        None,
        None,
        None,
        kvm::SERIAL_MMIO_BASE + 5,
        &mut buf,
    );
    assert_ne!(
        buf[0], 0xFF,
        "Serial MMIO read must invoke the COM1 LSR path, not the \
         unmapped 0xFF fallback"
    );
}

#[test]
fn classify_exit_aarch64_mmio_write_serial_is_continue() {
    // aarch64 MmioWrite to the COM1 serial window must classify as
    // Continue (no MMIO-side shutdown signal — reboot is PSCI via
    // SystemEvent). The byte must reach the COM1 output buffer to
    // pin the dispatch wire-up.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let data = [b'Q'];
    let mut exit = VcpuExit::MmioWrite(kvm::SERIAL_MMIO_BASE, &data);
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Continue)),
        "aarch64 MmioWrite to serial must classify as Continue"
    );
    assert!(
        com1.lock().output().contains('Q'),
        "Serial MMIO write must land the byte in COM1 output"
    );
}

#[test]
fn classify_exit_aarch64_mmio_read_unmapped_returns_0xff() {
    // Cross-check: classify_exit + dispatch_mmio_read 0xFF fallback
    // through the public dispatch path. Unmapped MMIO read must be
    // Continue with the buffer overwritten to 0xFF.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut buf = [0u8; 4];
    let mut exit = VcpuExit::MmioRead(0x10_0000, &mut buf);
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Continue)),
        "Unmapped aarch64 MMIO read must classify as Continue"
    );
    // Re-borrow `buf` to read it post-call — the &mut alias inside
    // `exit` is dropped when `exit` is, but we can re-inspect the
    // backing array here because classify_exit returned.
    assert_eq!(
        buf,
        [0xff, 0xff, 0xff, 0xff],
        "Unmapped aarch64 MMIO read must fill with 0xFF"
    );
}
