use super::super::super::*;

// ----------------------------------------------------------------
// mmio_read non-4-byte and config_read out-of-range defenses.
//
// virtio-v1.2 §4.2.2 specifies 4-byte register access for the MMIO
// control registers. A misbehaving guest issuing 1/2/8-byte reads
// would otherwise let the device return stale stack contents (the
// `data` buffer is the caller's, not zero-initialised by the
// device path). The defensive 0xff-fill makes the protocol
// violation visible — distinct from a register that legitimately
// reads as zero.
// ----------------------------------------------------------------

/// 8-byte mmio_read (above the 4-byte register width) must fill
/// every byte with 0xff. The existing `non_4byte_read_returns_ff`
/// covers the 2-byte case at offset 0; this extends coverage to
/// the upper boundary so a regression that only checked
/// `data.len() < 4` (instead of `!= 4`) would surface here.
#[test]
fn mmio_read_oversized_fills_0xff() {
    let dev = VirtioConsole::new();
    let mut buf = [0u8; 8];
    dev.mmio_read(VIRTIO_MMIO_VERSION as u64, &mut buf);
    assert_eq!(
        buf, [0xff; 8],
        "8-byte read must fill with 0xff — the device is 4-byte \
         register width per virtio-v1.2 §4.2.2",
    );
}

/// 1-byte mmio_read must fill the byte with 0xff. Pins the
/// boundary at the low end (data.len() == 1 < 4). A regression
/// that special-cased 0-length or skipped the fill on 1-byte
/// reads would surface here.
#[test]
fn mmio_read_1byte_fills_0xff() {
    let dev = VirtioConsole::new();
    let mut buf = [0u8; 1];
    dev.mmio_read(VIRTIO_MMIO_VERSION as u64, &mut buf);
    assert_eq!(buf, [0xff], "1-byte read must fill with 0xff",);
}

/// 3-byte mmio_read (one short of the 4-byte register width)
/// must fill every byte with 0xff. Pins the strict equality
/// against `data.len() != 4` — a regression using `< 4` would
/// reject this but the test still passes; using `> 4` would
/// accept this and copy stale data.
#[test]
fn mmio_read_3byte_fills_0xff() {
    let dev = VirtioConsole::new();
    let mut buf = [0u8; 3];
    dev.mmio_read(VIRTIO_MMIO_VERSION as u64, &mut buf);
    assert_eq!(
        buf,
        [0xff, 0xff, 0xff],
        "3-byte read must fill with 0xff — the device is exactly \
         4-byte register width, not 'at least 4'",
    );
}

/// 4-byte mmio_read at a known register (VERSION) returns the
/// register's actual value — pins that the 0xff-fill defense
/// is gated on len != 4 specifically, NOT applied to legitimate
/// 4-byte reads. Without this control test, a regression that
/// always filled 0xff would still pass the misalignment tests.
#[test]
fn mmio_read_4byte_returns_register_value() {
    let dev = VirtioConsole::new();
    let mut buf = [0u8; 4];
    dev.mmio_read(VIRTIO_MMIO_VERSION as u64, &mut buf);
    assert_eq!(
        u32::from_le_bytes(buf),
        MMIO_VERSION,
        "4-byte read at VIRTIO_MMIO_VERSION must return the \
         actual register value, NOT the 0xff-fill defense",
    );
}

/// `config_read` out-of-range fill: a read that starts inside
/// the 12-byte config struct but extends past byte 11 must fill
/// with 0xff. Pins the `if end > cfg.len()` defense — without
/// it, the `data.copy_from_slice(&cfg[start..end])` would
/// panic with "index out of bounds." A guest that issues a
/// 4-byte read at config offset 10 (i.e. straddling the end of
/// the struct) is misbehaving; the device must absorb without
/// panicking.
#[test]
fn config_read_out_of_range_fills_0xff() {
    let dev = VirtioConsole::new();
    let mut buf = [0u8; 4];
    // Config space starts at 0x100; offset 10 inside config
    // (i.e. mmio offset 0x10A) puts start=10, end=14 → out of
    // range (cfg.len() = 12).
    dev.mmio_read(0x100 + 10, &mut buf);
    assert_eq!(
        buf, [0xff; 4],
        "config_read past byte 11 must fill 0xff — the defense \
         prevents a panic from cfg[start..end] when end > 12",
    );
}

/// `config_read` at the exact end boundary (read straddles
/// struct end by 1 byte) still triggers the 0xff fill. Pins
/// the `>` (strict) comparison — `end == cfg.len()` is in
/// range; `end == cfg.len() + 1` is not.
#[test]
fn config_read_one_byte_past_end_fills_0xff() {
    let dev = VirtioConsole::new();
    let mut buf = [0u8; 4];
    // Offset 9 inside config: start=9, end=13. cfg.len()=12,
    // so end=13 > 12 → 0xff fill.
    dev.mmio_read(0x100 + 9, &mut buf);
    assert_eq!(
        buf, [0xff; 4],
        "config_read with end one byte past struct must fill 0xff",
    );
}

/// `config_read` reading the LAST 4 valid bytes (offset 8..12
/// inside config, the emerg_wr field) must return the actual
/// data (zeros, since we don't advertise F_EMERG_WRITE), NOT the
/// 0xff fill. Pins the strict-greater-than boundary in the
/// reverse direction: end == 12 is allowed.
#[test]
fn config_read_at_exact_end_returns_data() {
    let dev = VirtioConsole::new();
    let mut buf = [0u8; 4];
    // Offset 8 inside config: start=8, end=12. cfg.len()=12,
    // so end == cfg.len() → in range, returns emerg_wr (0).
    dev.mmio_read(0x100 + 8, &mut buf);
    assert_eq!(
        buf, [0; 4],
        "read at the last 4-byte slot of the config struct \
         (emerg_wr, offset 8..12) must return actual data, NOT \
         the 0xff out-of-range fill",
    );
}

/// `config_read` 1-byte read at offset 0 (cols low byte) must
/// return the actual data (0). Pins that the out-of-range
/// defense does not over-trigger on small reads inside the
/// struct. Without F_SIZE we never populate cols, so the byte
/// is 0 by initialisation in `let mut cfg = [0u8; 12];`.
#[test]
fn config_read_1byte_inside_struct_returns_data() {
    let dev = VirtioConsole::new();
    let mut buf = [0u8; 1];
    // Offset 0 inside config (cols low byte). start=0, end=1.
    // 1 <= 12, so in range; returns cfg[0] = 0.
    dev.mmio_read(0x100, &mut buf);
    assert_eq!(
        buf,
        [0],
        "1-byte read inside config struct must return actual \
         data, not 0xff fill",
    );
}
