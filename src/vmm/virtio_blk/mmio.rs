//! virtio-MMIO transport facade for [`VirtioBlk`].
//!
//! Decodes the virtio-v1.2 §4.2.2 MMIO register layout (and the
//! `0x100..` device-config window) onto the transport-neutral core API
//! on [`VirtioBlk`] (the register/FSM ops in the sibling
//! [`super::control`] module, the state in [`super::device`]). This file
//! owns ONLY the MMIO address/width decode; all device behaviour (status
//! FSM, feature negotiation, queue assembly, notify, interrupt
//! bookkeeping) lives in the core and is shared verbatim with any other
//! transport facade. Reached as `VirtioBlk::mmio_read` /
//! `VirtioBlk::mmio_write` from the MMIO dispatch in
//! [`crate::vmm::exit_dispatch`].
//!
//! The queue-config / feature-lock gates live INSIDE the core ops (a
//! write to a gated register is a no-op, exactly as the prior per-arm
//! `if queue_config_allowed()` / `if features_write_allowed()`
//! match-guards made it fall through to the no-op default), so the
//! decode `match` here carries no gates — offset maps straight to op.
//! That is the whole point of the extraction: a second transport (the
//! PCI facade) reuses the same op set without re-implementing the gates,
//! so the two transports cannot drift.

use super::*;

impl VirtioBlk {
    /// Handle MMIO read at `offset` within the device's MMIO region.
    ///
    /// Two address ranges:
    /// - `offset >= 0x100`: device-specific config space, dispatched
    ///   to [`Self::read_blk_config`].
    /// - `offset < 0x100`: virtio-mmio common transport registers
    ///   (magic/version/device-id, status, queue config, interrupt
    ///   status). All transport registers are 4-byte u32; non-4-byte
    ///   reads here are guest bugs.
    ///
    /// Non-4-byte fallback fills `data` with `0xff` rather than 0
    /// because 0xff is far easier to spot in a guest crash dump or
    /// hex view than a successful 0 — it surfaces "the device
    /// declined to answer" instead of disguising it as a valid
    /// zero-valued register read. Config space (`offset >= 0x100`)
    /// uses 0-fill instead because virtio-v1.2 §4.2.2.2 specifies
    /// reads past the populated config layout return zero.
    pub fn mmio_read(&self, offset: u64, data: &mut [u8]) {
        if offset >= 0x100 {
            self.read_blk_config(offset - 0x100, data);
            return;
        }
        if data.len() != 4 {
            data.fill(0xff);
            return;
        }
        let val: u32 = match offset as u32 {
            VIRTIO_MMIO_MAGIC_VALUE => MMIO_MAGIC,
            VIRTIO_MMIO_VERSION => MMIO_VERSION,
            VIRTIO_MMIO_DEVICE_ID => VIRTIO_ID_BLOCK,
            VIRTIO_MMIO_VENDOR_ID => VENDOR_ID,
            VIRTIO_MMIO_DEVICE_FEATURES => self.device_features_window(),
            VIRTIO_MMIO_QUEUE_NUM_MAX => self.queue_max_size(),
            VIRTIO_MMIO_QUEUE_READY => self.queue_ready(),
            VIRTIO_MMIO_INTERRUPT_STATUS => self.interrupt_status(),
            VIRTIO_MMIO_STATUS => self.device_status(),
            VIRTIO_MMIO_CONFIG_GENERATION => self.config_generation(),
            _ => 0,
        };
        data.copy_from_slice(&val.to_le_bytes());
    }

    /// Handle MMIO write at `offset` within the device's MMIO region.
    ///
    /// Same two address ranges as [`Self::mmio_read`]:
    /// - `offset >= 0x100`: device config space. Per virtio-v1.2
    ///   §4.2.2 the device owns this region — it's read-only from
    ///   the driver's perspective, populated by the device when
    ///   the driver reads. Guest writes are silently dropped (no
    ///   tracing::warn either; the kernel's virtio_mmio probe path
    ///   has been seen to issue speculative config-space writes
    ///   during feature negotiation, and warning on every one
    ///   would flood the log without identifying any real bug).
    /// - `offset < 0x100`: transport registers, dispatched per
    ///   `match`. Non-4-byte writes are silently dropped — same
    ///   "the spec mandates 4-byte access" reasoning as the read
    ///   path; the device acts on a partial register write at its
    ///   peril, so dropping is safer than wedging an MMIO FSM
    ///   with half-applied state.
    ///
    /// Each arm maps the offset to a transport-neutral core op; the
    /// queue-config / feature-lock gates live inside those ops (a write
    /// to a gated register is a no-op, exactly as the prior per-arm
    /// `if queue_config_allowed()` match-guards made it fall through to
    /// the no-op default).
    pub fn mmio_write(&mut self, offset: u64, data: &[u8]) {
        if offset >= 0x100 {
            // Config space writes are device-owned; drop silently.
            return;
        }
        if data.len() != 4 {
            return;
        }
        let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        match offset as u32 {
            VIRTIO_MMIO_DEVICE_FEATURES_SEL => self.set_device_features_sel(val),
            VIRTIO_MMIO_DRIVER_FEATURES_SEL => self.set_driver_features_sel(val),
            VIRTIO_MMIO_DRIVER_FEATURES => self.set_driver_features_window(val),
            VIRTIO_MMIO_QUEUE_SEL => self.set_queue_select(val),
            VIRTIO_MMIO_QUEUE_NUM => self.set_queue_size(val as u16),
            VIRTIO_MMIO_QUEUE_READY => self.set_queue_ready(val == 1),
            VIRTIO_MMIO_QUEUE_NOTIFY => self.notify_queue(val),
            VIRTIO_MMIO_INTERRUPT_ACK => self.ack_interrupt(val),
            VIRTIO_MMIO_STATUS => self.write_status(val),
            VIRTIO_MMIO_QUEUE_DESC_LOW => self.set_queue_desc_addr(Some(val), None),
            VIRTIO_MMIO_QUEUE_DESC_HIGH => self.set_queue_desc_addr(None, Some(val)),
            VIRTIO_MMIO_QUEUE_AVAIL_LOW => self.set_queue_avail_addr(Some(val), None),
            VIRTIO_MMIO_QUEUE_AVAIL_HIGH => self.set_queue_avail_addr(None, Some(val)),
            VIRTIO_MMIO_QUEUE_USED_LOW => self.set_queue_used_addr(Some(val), None),
            VIRTIO_MMIO_QUEUE_USED_HIGH => self.set_queue_used_addr(None, Some(val)),
            _ => {}
        }
    }
}
