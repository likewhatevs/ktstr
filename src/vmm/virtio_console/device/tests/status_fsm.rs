use super::super::super::*;
use super::*;

// ----------------------------------------------------------------
// set_status monotone defense.
//
// virtio-v1.2 §3.1.1 requires status bits to advance monotonically
// within a driver session — the driver MUST NOT clear bits except
// by writing 0 (which triggers reset). A hostile or buggy guest
// that clears bits mid-session would otherwise let the device
// backslide through the FSM (e.g. drop FEATURES_OK while
// queues are configured) and produce undefined behaviour.
//
// The handler enforces monotonicity via:
//   `if val & self.device_status != self.device_status { reject }`
// i.e. `val` must be a SUPERSET of `device_status`. These tests
// pin every facet of that check.
// ----------------------------------------------------------------

/// Clearing ACKNOWLEDGE alone must be rejected. Driver advances
/// to ACK, then writes 0 of all other bits while clearing ACK —
/// the monotone gate must reject because the new value is not
/// a superset of the current device_status.
#[test]
fn set_status_clear_acknowledge_rejected() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    assert_eq!(dev.device_status, S_ACK);
    // Try to clear ACK by writing DRIVER alone (no ACK bit).
    // val = DRIVER; current = ACK. val & current = 0 != ACK → reject.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, VIRTIO_CONFIG_S_DRIVER);
    assert_eq!(
        dev.device_status, S_ACK,
        "writing a value that clears ACKNOWLEDGE must be rejected — \
         monotone gate (virtio-v1.2 §3.1.1)",
    );
}

/// Clearing DRIVER while keeping ACKNOWLEDGE must be rejected.
/// Driver advances to ACK | DRIVER, then writes ACK alone — the
/// monotone gate must reject because DRIVER would silently drop.
#[test]
fn set_status_clear_driver_keeps_ack_rejected() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    assert_eq!(dev.device_status, S_DRV);
    // val = S_ACK; current = S_DRV (= ACK | DRIVER).
    // val & current = ACK != DRV → reject.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    assert_eq!(
        dev.device_status, S_DRV,
        "writing ACK alone after advancing to DRIVER must be \
         rejected — clears DRIVER bit",
    );
}

/// Clearing FEATURES_OK from a fully-initialised device must be
/// rejected. Driver advances all the way to S_OK, then tries to
/// write S_FEAT (= ACK|DRIVER|FEATURES_OK, no DRIVER_OK) — the
/// monotone gate rejects because DRIVER_OK would be cleared.
#[test]
fn set_status_clear_driver_ok_rejected() {
    let mut dev = VirtioConsole::new();
    init_device(&mut dev);
    assert_eq!(dev.device_status, S_OK);
    // val = S_FEAT; current = S_OK. S_FEAT is missing DRIVER_OK.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
    assert_eq!(
        dev.device_status, S_OK,
        "writing S_FEAT after S_OK must be rejected — \
         clears DRIVER_OK",
    );
}

/// Writing the SAME value (no new bits, no cleared bits) must
/// not panic and must leave device_status unchanged. This is
/// not a hostile case but it exercises the boundary: `val ==
/// device_status` is a superset of itself, so the monotone
/// gate does not reject; `new_bits == 0` then falls through to
/// the `_ => false` arm in the valid-transition match, which
/// rejects the write. Pins that the device is idempotent
/// against duplicate writes — the kernel does not retry, but
/// hostile guests might.
#[test]
fn set_status_idempotent_same_value_no_change() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    assert_eq!(dev.device_status, S_ACK);
    // Re-write S_ACK — superset check passes (val==current),
    // but no NEW bits, so the valid-transition match falls
    // through to the catch-all rejection.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    assert_eq!(
        dev.device_status, S_ACK,
        "re-writing the same status must leave device_status \
         unchanged — no advance, no regression",
    );
}

/// Setting two new bits in one write (e.g. FEATURES_OK + DRIVER_OK
/// jumped together) must be rejected — the FSM advances one bit
/// at a time. Pins the `match new_bits` valid-transition arm:
/// only single-bit advances are accepted.
#[test]
fn set_status_two_bits_at_once_rejected() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    // val = S_OK = ACK|DRIVER|FEATURES_OK|DRIVER_OK; current = S_DRV.
    // new_bits = FEATURES_OK | DRIVER_OK. The valid-transition
    // match has no arm for the union; falls through to false.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
    assert_eq!(
        dev.device_status, S_DRV,
        "advancing two FSM bits at once (FEATURES_OK + DRIVER_OK) \
         must be rejected — the FSM advances one bit at a time",
    );
}

/// Writing a status value with an unrecognised bit set (e.g. an
/// undefined flag at 0x40) must be rejected. virtio-v1.2 §2.1
/// defines bits 0x01 (ACK), 0x02 (DRIVER), 0x04 (DRIVER_OK),
/// 0x08 (FEATURES_OK), 0x40 (NEEDS_RESET, device-set only),
/// 0x80 (FAILED). 0x10 and 0x20 are reserved. NEEDS_RESET (0x40)
/// is device-set only, so a guest that writes it is misbehaving;
/// the value is also a non-canonical FSM advance. Pins that
/// these unknown bits land in the `_ => false` valid arm and
/// device_status remains unchanged.
#[test]
fn set_status_unknown_bit_rejected() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    assert_eq!(dev.device_status, S_ACK);
    // val = S_ACK | 0x10 (reserved/unknown bit). new_bits = 0x10.
    // The valid-transition match has no arm for 0x10 → reject.
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK | 0x10);
    assert_eq!(
        dev.device_status, S_ACK,
        "writing a status with an unrecognised bit (0x10) must \
         be rejected — only the defined ACK/DRIVER/FEATURES_OK/\
         DRIVER_OK/FAILED transitions are accepted",
    );
}

/// FAILED (bit 0x80) must be accepted on top of any FSM state.
/// virtio-v1.2 §2.1.1 — `virtio_add_status(dev,
/// VIRTIO_CONFIG_S_FAILED)` is the kernel's exit path on probe
/// failure (drivers/virtio/virtio.c). Pins the FAILED early-accept
/// branch in `set_status`. Verified at every FSM rung.
#[test]
fn set_status_failed_accepted_at_every_fsm_state() {
    // From device_status = 0.
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, VIRTIO_CONFIG_S_FAILED);
    assert_eq!(
        dev.device_status, VIRTIO_CONFIG_S_FAILED,
        "FAILED from status=0 must be accepted",
    );

    // From device_status = ACK.
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK | VIRTIO_CONFIG_S_FAILED);
    assert_eq!(
        dev.device_status,
        S_ACK | VIRTIO_CONFIG_S_FAILED,
        "FAILED from status=ACK must be accepted (ACK preserved)",
    );

    // From device_status = DRV.
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV | VIRTIO_CONFIG_S_FAILED);
    assert_eq!(
        dev.device_status,
        S_DRV | VIRTIO_CONFIG_S_FAILED,
        "FAILED from status=DRV must be accepted (DRV preserved)",
    );

    // From device_status = S_OK.
    let mut dev = VirtioConsole::new();
    init_device(&mut dev);
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK | VIRTIO_CONFIG_S_FAILED);
    assert_eq!(
        dev.device_status,
        S_OK | VIRTIO_CONFIG_S_FAILED,
        "FAILED from status=S_OK must be accepted (S_OK preserved)",
    );
}

/// FAILED combined with a non-FAILED unrecognised new bit must
/// be rejected — the FAILED early-accept only triggers when
/// `new_bits == VIRTIO_CONFIG_S_FAILED` (FAILED alone, no other
/// new bits). A guest mixing FAILED with garbage extra bits is
/// misbehaving in a way unrelated to the legitimate FAILED signal.
#[test]
fn set_status_failed_plus_unknown_bit_rejected() {
    let mut dev = VirtioConsole::new();
    write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
    // val = ACK | FAILED | 0x10. new_bits = FAILED | 0x10 ≠
    // FAILED alone, so the early-accept branch does NOT trigger;
    // the regular valid-transition match has no arm for the
    // union → reject.
    write_reg(
        &mut dev,
        VIRTIO_MMIO_STATUS,
        S_ACK | VIRTIO_CONFIG_S_FAILED | 0x10,
    );
    assert_eq!(
        dev.device_status, S_ACK,
        "FAILED combined with a non-FAILED unknown bit must be \
         rejected — the early-accept is gated on FAILED alone",
    );
}
