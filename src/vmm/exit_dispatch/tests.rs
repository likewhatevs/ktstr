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
    let mut data = [0xFFu8; 1];
    dispatch_io_in(&com1, &com2, None,0x1234, &mut data);
    assert_eq!(data[0], 0xFF, "unknown port should not modify data");
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
