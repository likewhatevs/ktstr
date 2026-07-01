//! Network configuration for the virtio-net device.
//!
//! [`NetConfig`] is the descriptor — passed by value, copious defaults,
//! no fd field (the framework owns the in-VMM loopback backend's
//! lifecycle). Mirrors [`super::disk_config::DiskConfig`] for API
//! consistency: chainable setters return `Self`, `Default::default()`
//! produces a working device, and the type lives in the public prelude
//! so test authors can spell it out.
//!
//! v0 backend is in-VMM loopback: TX descriptor bytes are written
//! directly into RX descriptors and the irqfd fires. There is no host
//! networking, no `/dev/net/tun`, no AF_PACKET fd. Each attached
//! virtio-net interface (one per `networks = [..]` element on x86_64,
//! a single MMIO NIC on aarch64) loops its own TX back to its RX
//! verbatim — no MAC swap, no ARP synthesis, no IP routing. The byte
//! flow lives in [`super::virtio_net`] (see `process_tx_loopback` in
//! the device module). AF_PACKET raw sockets bound to the interface
//! generate real virtio TX kicks and observe real virtio RX
//! interrupts — IP-layer self-traffic is forced onto `lo` by the
//! kernel routing layer (`net/ipv4/route.c::ip_route_output_key_hash_rcu`
//! resolves any local destination as `RTN_LOCAL` →
//! `dev_out = net->loopback_dev`) and never reaches virtio-net.

/// Configuration for the virtio-net device attached to the VM.
///
/// `Default::default()` produces a working device with a deterministic
/// locally-administered MAC. Override the MAC with [`Self::mac`] to
/// pin a value across runs (useful for log correlation against
/// AF_PACKET captures).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct NetConfig {
    /// MAC address advertised to the guest via `VIRTIO_NET_F_MAC`.
    /// The locally-administered bit (0x02) is set in the default to
    /// avoid collisions with real-hardware OUIs; operators that
    /// override the MAC are responsible for the bit themselves.
    pub mac: [u8; 6],
    /// Number of virtio-net queue-pairs (one RX + one TX virtqueue per
    /// pair) the device offers. Default `1` — a single queue-pair,
    /// byte-identical to a device with no multiqueue support (no
    /// `VIRTIO_NET_F_MQ`, no control virtqueue).
    ///
    /// Multiqueue is offered only when `queue_pairs > 1` AND the transport
    /// carries MSI-X (the x86_64 PCI NIC). On a non-MSI-X transport — the
    /// aarch64 MMIO NIC, or PCI without MSI-X — the device stays single-pair
    /// regardless of this value (no `VIRTIO_NET_F_MQ`, no control vq,
    /// `max_virtqueue_pairs = 0`): per-queue IRQ steering, the point of
    /// multiqueue, needs the distinct MSI-X vectors that transport lacks.
    ///
    /// When multiqueue IS offered, the device advertises `VIRTIO_NET_F_MQ` +
    /// the control virtqueue (`VIRTIO_NET_F_CTRL_VQ`) and reports
    /// `max_virtqueue_pairs` in config space; the guest brings up
    /// `min(num_online_cpus, queue_pairs)` pairs and spreads its RX/TX across
    /// them.
    ///
    /// Clamped to `[1, 256]` at construction: `0` becomes `1` (the spec
    /// minimum, `VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MIN`) and values above `256`
    /// (`MAX_QUEUE_PAIRS`) are clamped down.
    pub queue_pairs: u16,
}

impl NetConfig {
    /// Const default — MAC `02:00:00:00:00:01`. The leading `0x02` sets
    /// the locally-administered bit per IEEE 802 (bit 1 of the first
    /// octet), keeping the address out of the IEEE OUI namespace; the
    /// trailing `0x01` is the conventional first-NIC suffix — multi-NIC
    /// tests give each element a distinct MAC via [`Self::mac`]. `const`
    /// so it can seed a `const NetConfig` for the
    /// `#[ktstr_test(networks = [...])]` attribute, matching
    /// [`super::disk_config::DiskConfig`]'s `DEFAULT`.
    pub const DEFAULT: NetConfig = NetConfig {
        mac: [0x02, 0x00, 0x00, 0x00, 0x00, 0x01],
        queue_pairs: 1,
    };

    /// Override the advertised MAC. Returns `self` for chained
    /// configuration. `const fn` so a `const NetConfig` can be built via
    /// `NetConfig::DEFAULT.mac(...)`, matching `DiskConfig`'s const-fn
    /// builder style.
    #[must_use = "builder methods consume self; bind the result"]
    pub const fn mac(mut self, mac: [u8; 6]) -> Self {
        self.mac = mac;
        self
    }

    /// Set the number of queue-pairs the device offers (see
    /// [`Self::queue_pairs`]). Returns `self` for chained configuration.
    /// `const fn` so a `const NetConfig` can be built via
    /// `NetConfig::DEFAULT.queue_pairs(...)`, matching [`Self::mac`].
    #[must_use = "builder methods consume self; bind the result"]
    pub const fn queue_pairs(mut self, pairs: u16) -> Self {
        self.queue_pairs = pairs;
        self
    }
}

impl Default for NetConfig {
    /// Delegates to [`Self::DEFAULT`] (MAC `02:00:00:00:00:01`).
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_locally_administered_bit() {
        let cfg = NetConfig::default();
        assert_eq!(
            cfg.mac[0] & 0x02,
            0x02,
            "default MAC must have locally-administered bit (IEEE 802 first-octet bit 1)",
        );
    }

    #[test]
    fn default_is_unicast() {
        let cfg = NetConfig::default();
        assert_eq!(
            cfg.mac[0] & 0x01,
            0x00,
            "default MAC must not be multicast (IEEE 802 first-octet bit 0)",
        );
    }

    #[test]
    fn mac_setter_overrides_default() {
        let cfg = NetConfig::default().mac([0x52, 0x54, 0x00, 0xab, 0xcd, 0xef]);
        assert_eq!(cfg.mac, [0x52, 0x54, 0x00, 0xab, 0xcd, 0xef]);
    }

    #[test]
    fn serde_roundtrip_pins_field_names() {
        let cfg = NetConfig::default().mac([1, 2, 3, 4, 5, 6]).queue_pairs(4);
        let json = serde_json::to_string(&cfg).expect("serialize");
        // Pin the field names so a future rename surfaces here.
        assert!(json.contains("\"mac\""), "missing key `mac`: {json}");
        assert!(
            json.contains("\"queue_pairs\""),
            "missing key `queue_pairs`: {json}"
        );
        let parsed: NetConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, cfg);
    }
}
