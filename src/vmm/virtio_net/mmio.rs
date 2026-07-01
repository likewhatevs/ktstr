//! virtio-MMIO transport facade for [`VirtioNet`].
//!
//! Decodes the virtio-v1.2 §4.2.2 MMIO register layout (and the
//! `0x100..` device-config window) onto the transport-neutral core API
//! on [`VirtioNet`] (in the sibling [`super::device`] module). This
//! file owns ONLY the MMIO address/width decode; all device behaviour
//! (status FSM, feature negotiation, queue assembly, notify, interrupt
//! bookkeeping) lives in the core and is shared verbatim with any other
//! transport facade. Reached as `VirtioNet::mmio_read` /
//! `VirtioNet::mmio_write` from the MMIO dispatch in
//! [`crate::vmm::exit_dispatch`].

use virtio_bindings::virtio_ids::VIRTIO_ID_NET;
use virtio_bindings::virtio_mmio::{
    VIRTIO_MMIO_CONFIG_GENERATION, VIRTIO_MMIO_DEVICE_FEATURES, VIRTIO_MMIO_DEVICE_FEATURES_SEL,
    VIRTIO_MMIO_DEVICE_ID, VIRTIO_MMIO_DRIVER_FEATURES, VIRTIO_MMIO_DRIVER_FEATURES_SEL,
    VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INTERRUPT_STATUS, VIRTIO_MMIO_MAGIC_VALUE,
    VIRTIO_MMIO_QUEUE_AVAIL_HIGH, VIRTIO_MMIO_QUEUE_AVAIL_LOW, VIRTIO_MMIO_QUEUE_DESC_HIGH,
    VIRTIO_MMIO_QUEUE_DESC_LOW, VIRTIO_MMIO_QUEUE_NOTIFY, VIRTIO_MMIO_QUEUE_NUM,
    VIRTIO_MMIO_QUEUE_NUM_MAX, VIRTIO_MMIO_QUEUE_READY, VIRTIO_MMIO_QUEUE_SEL,
    VIRTIO_MMIO_QUEUE_USED_HIGH, VIRTIO_MMIO_QUEUE_USED_LOW, VIRTIO_MMIO_STATUS,
    VIRTIO_MMIO_VENDOR_ID, VIRTIO_MMIO_VERSION,
};

use super::device::{MMIO_MAGIC, MMIO_VERSION, VENDOR_ID, VirtioNet};

impl VirtioNet {
    /// Handle an MMIO read at `offset` within the device's MMIO region.
    pub fn mmio_read(&self, offset: u64, data: &mut [u8]) {
        // Config-space reads (offsets 0x100..) may be 1, 2, 4, or 8
        // bytes wide depending on the field's type per virtio-v1.2
        // §4.2.2.2; serve them from the static config struct's bytes
        // first so a 1-byte MAC read or 2-byte STATUS read returns
        // the right value rather than the 0xff "non-4-byte" sentinel.
        if offset >= 0x100 {
            self.config_bytes((offset - 0x100) as usize, data);
            return;
        }

        // Register-space reads are 4 bytes wide. Anything else is a
        // protocol violation — return 0xff bytes (matches virtio-blk
        // and virtio-console).
        if data.len() != 4 {
            for b in data.iter_mut() {
                *b = 0xff;
            }
            return;
        }
        let val: u32 = match offset as u32 {
            VIRTIO_MMIO_MAGIC_VALUE => MMIO_MAGIC,
            VIRTIO_MMIO_VERSION => MMIO_VERSION,
            VIRTIO_MMIO_DEVICE_ID => VIRTIO_ID_NET,
            VIRTIO_MMIO_VENDOR_ID => VENDOR_ID,
            VIRTIO_MMIO_DEVICE_FEATURES => self.device_features_window(),
            VIRTIO_MMIO_QUEUE_NUM_MAX => self.queue_max_size(),
            VIRTIO_MMIO_QUEUE_READY => self.queue_ready(),
            VIRTIO_MMIO_INTERRUPT_STATUS => self.interrupt_status(),
            VIRTIO_MMIO_STATUS => self.device_status(),
            VIRTIO_MMIO_CONFIG_GENERATION => self.config_generation(),
            _ => 0,
        };
        tracing::debug!(offset, val, "virtio-net mmio_read");
        data.copy_from_slice(&val.to_le_bytes());
    }

    /// Handle an MMIO write at `offset` within the device's MMIO region.
    pub fn mmio_write(&mut self, offset: u64, data: &[u8]) {
        // Config-space writes are silently ignored (this device is
        // not driver-configurable; STATUS/MQ/MTU are read-only).
        // Matches virtio-console; virtio-v1.2 §4.2.2.2 ("the device
        // MAY ignore writes to config space").
        if offset >= 0x100 {
            tracing::debug!(
                offset,
                len = data.len(),
                "virtio-net config-space write ignored"
            );
            return;
        }

        if data.len() != 4 {
            return;
        }
        let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        tracing::debug!(offset, val, "virtio-net mmio_write");
        // Each arm maps the MMIO register offset to a transport-neutral
        // core op; the queue-config / feature-lock gates live inside
        // those ops (a write to a gated register is a no-op, exactly as
        // the prior per-arm `if queue_config_allowed()` match-guards
        // made it fall through to the no-op default). The RXQ notify is
        // a no-op (the next TX picks up newly posted RX buffers; a
        // future TAP/AF_PACKET backend would drain host->guest frames
        // on RXQ notify). INTERRUPT_ACK just clears the guest-acked
        // bits — the edge irqfd needs no device-side resample on ACK
        // (the kernel's vm_interrupt handshake clears its own view).
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
