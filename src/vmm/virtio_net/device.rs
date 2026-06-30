//! Device-side virtio-net: MMIO dispatch, FSM, counters, in-VMM
//! loopback. See the parent module `super` for the execution-model and
//! "why" doc — header-size invariant, loopback rationale, no-worker
//! decision.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use virtio_bindings::virtio_config::{
    VIRTIO_CONFIG_S_ACKNOWLEDGE, VIRTIO_CONFIG_S_DRIVER, VIRTIO_CONFIG_S_DRIVER_OK,
    VIRTIO_CONFIG_S_FAILED, VIRTIO_CONFIG_S_FEATURES_OK, VIRTIO_CONFIG_S_NEEDS_RESET,
    VIRTIO_F_VERSION_1,
};
// Only the two interrupt-status bits the core sets (`signal_used` /
// `signal_queue_poisoned`) remain here; the MMIO register-offset
// constants moved to the `mmio` facade with `mmio_read`/`mmio_write`.
use virtio_bindings::virtio_mmio::{VIRTIO_MMIO_INT_CONFIG, VIRTIO_MMIO_INT_VRING};
use virtio_bindings::virtio_net::{
    VIRTIO_NET_CTRL_MQ, VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MIN, VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET,
    VIRTIO_NET_ERR, VIRTIO_NET_F_CTRL_VQ, VIRTIO_NET_F_MAC, VIRTIO_NET_F_MQ, VIRTIO_NET_OK,
};
use virtio_queue::{DescriptorChain, Error as VirtioQueueError, Queue, QueueOwnedT, QueueT};
use vm_memory::{Address, ByteValued, Bytes, GuestAddress, GuestMemoryMmap};
use vmm_sys_util::eventfd::EventFd;

use super::counters::VirtioNetCounters;
use super::interrupt::{IrqSource, MsixState};
use crate::vmm::PiMutex;
use crate::vmm::net_config::NetConfig;

pub(crate) const MMIO_MAGIC: u32 = 0x7472_6976; // "virt" in LE
pub(crate) const MMIO_VERSION: u32 = 2; // virtio 1.x MMIO
pub(crate) const VENDOR_ID: u32 = 0;

/// MMIO region size: 4 KB (one page). Matches virtio-console and
/// virtio-blk so the FDT/cmdline emitter and the MMIO range checks in
/// `exit_dispatch` can use a single constant per device class.
pub const VIRTIO_MMIO_SIZE: u64 = 0x1000;

/// Virtqueue layout: data queues interleave per the kernel `init_vqs`
/// order (`drivers/net/virtio_net.c` rxq2vq/txq2vq) — pair `i` has its RX
/// at even index `2i` and TX at odd index `2i+1`; multiqueue
/// (`NetConfig::queue_pairs > 1`) adds a trailing control vq at index
/// `2 * queue_pairs`. `NUM_QUEUES` is the single-pair baseline (RX 0 +
/// TX 1, no control vq) — the device default; the live per-device queue
/// count is `VirtioNet::queues.len()`, and the MSI-X table is sized from it
/// (`num_queues + 1`: one vector per virtqueue plus config).
pub(crate) const NUM_QUEUES: usize = 2;
pub(crate) const QUEUE_MAX_SIZE: u16 = 256;
/// Maximum queue-pairs a NIC may offer; `NetConfig::queue_pairs` is clamped to
/// `[1, MAX_QUEUE_PAIRS]`. Bounds the virtqueue count so the highest virtqueue
/// INDEX — the control vq at `2 * queue_pairs` (= 512 at the cap) — stays within
/// the PCI notify region (`pci::REGION_SIZE / pci::NOTIFY_OFF_MULTIPLIER` = 1024
/// slots) and keeps the per-device queue allocation sane. 256 pairs is 513
/// virtqueues TOTAL (2*256 data + 1 control; top index 512), far exceeding
/// realistic per-node steering needs while leaving notify-region headroom.
pub(crate) const MAX_QUEUE_PAIRS: u16 = 256;

/// Single-pair (pair 0) RX/TX virtqueue indices — RX at 0, TX at 1. Used by
/// the unit tests, which exercise the single-pair loopback path; production
/// code computes per-pair indices (`2*pair` / `2*pair+1`) in
/// `process_tx_loopback`, so these constants are test-only.
#[cfg(test)]
pub(crate) const RXQ: usize = 0;
#[cfg(test)]
pub(crate) const TXQ: usize = 1;

/// Header length the guest expects on every RX delivery and emits on
/// every TX request. `VIRTIO_F_VERSION_1` negotiation forces
/// `vi->hdr_len = sizeof(virtio_net_hdr_mrg_rxbuf) = 12 bytes` in
/// `drivers/net/virtio_net.c::virtnet_probe`, even when
/// `VIRTIO_NET_F_MRG_RXBUF` is NOT negotiated. The mrg_rxbuf form
/// flattens to `virtio_net_hdr_v1` (10 bytes of GSO/csum fields) plus
/// a 2-byte `num_buffers`. The field is only read on RX (the device
/// emits it); on TX the guest writes a copy that the device strips.
pub const VIRTIO_NET_HDR_LEN: usize = 12;

/// Maximum L2 frame size (bytes, excluding the 12-byte virtio-net
/// header) the device accepts on TX or emits on RX. Sized to the
/// largest Ethernet frame the guest can legitimately emit:
///
/// - v0 advertises no `VIRTIO_NET_F_MTU`, so the guest driver keeps
///   `dev->max_mtu = MAX_MTU = ETH_MAX_MTU = 65535`
///   (`drivers/net/virtio_net.c::virtnet_probe` sets `dev->max_mtu =
///   MAX_MTU` unconditionally, overriding only when `VIRTIO_NET_F_MTU`
///   is negotiated).
/// - That 65535-byte L3 payload can ride two stacked VLAN tags
///   (802.1ad QinQ) without MTU reduction: a VLAN device drops
///   `VLAN_HLEN` from its MTU only when `netif_reduces_vlan_mtu`
///   (= `netif_is_macsec`, `include/linux/netdevice.h`) holds for the
///   lower device (`net/8021q/vlan_dev.c::vlan_dev_change_mtu`), which
///   is false for virtio-net. v0 advertises no HW VLAN offload, so both
///   tags land in the linear skb data the guest hands us.
///
/// Worst case is therefore `ETH_HLEN (14) + 2 * VLAN_HLEN (8) +
/// max_mtu (65535) = 65557` — the 802.1ad double-tagged frame. Deeper,
/// non-standard VLAN nesting exceeds this and is dropped as oversize.
///
/// A TX chain whose post-header data exceeds the cap is DROPPED, not
/// truncated: `pop_and_capture_tx` bumps `tx_oversize_dropped` and
/// returns `frame_len: None` (the chain is still marked used so the
/// guest doesn't hang). Silent truncation would corrupt a real frame
/// the guest emitted — a length the guest believes was transmitted
/// intact. An RX frame longer than this cannot arise in v0's
/// pure-loopback backend: the RX source is always a captured TX frame,
/// itself already bounded by this cap.
pub(crate) const MAX_FRAME_SIZE: usize = 65_557;

/// Status bits required before each phase. Mirrors virtio_console.
pub(crate) const S_ACK: u32 = VIRTIO_CONFIG_S_ACKNOWLEDGE;
pub(crate) const S_DRV: u32 = S_ACK | VIRTIO_CONFIG_S_DRIVER;
pub(crate) const S_FEAT: u32 = S_DRV | VIRTIO_CONFIG_S_FEATURES_OK;
/// Test helper — terminal state bits with DRIVER_OK set.
#[cfg(test)]
pub(crate) const S_OK: u32 = S_FEAT | VIRTIO_CONFIG_S_DRIVER_OK;

// ---------------------------------------------------------------------------
// Config space
// ---------------------------------------------------------------------------

/// Net device config space (virtio-v1.2 §5.1.4). Mirrors the kernel
/// uapi `struct virtio_net_config` field-for-field up through `mtu`
/// (the last field whose feature bit governs reads we serve). Trailing
/// fields (`speed`, `duplex`, RSS) are gated on feature bits we don't
/// advertise, so the guest driver's `virtio_cread_feature` returns
/// `-ENOENT` for those reads and never depends on the device-side
/// bytes — we serve zeros for any read past `size_of::<VirtioNetConfig>()`,
/// matching virtio-v1.2 §4.2.2.2 ("reads past the populated config
/// layout return zero").
///
/// The kernel struct is `__attribute__((packed))` (see
/// `include/uapi/linux/virtio_net.h`), so this redeclaration uses
/// `repr(C, packed)` to match the wire layout byte-for-byte. Without
/// the `packed` attribute the compiler would insert padding after
/// `mac` to align `status` (which contains a `u16`) — that padding
/// would shift `status` from offset 0x06 to 0x08 and serve the guest
/// a wrong link-status value silently.
#[repr(C, packed)]
#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct VirtioNetConfig {
    /// MAC address. Always populated; gated on `VIRTIO_NET_F_MAC` from
    /// the guest's perspective (without the bit it generates a random
    /// MAC and never reads this field). v0 always advertises F_MAC.
    pub(crate) mac: [u8; 6],
    /// Link status. `VIRTIO_NET_S_LINK_UP = 1` means the carrier is up.
    /// Gated on `VIRTIO_NET_F_STATUS`. v0 does NOT advertise STATUS,
    /// so the kernel driver assumes link up unconditionally
    /// (`virtnet_probe`: "Assume link up if device can't report link
    /// status"). The field stays zero in this struct; reads past the
    /// populated layout return zero anyway.
    pub(crate) status: u16,
    /// Multiqueue pair count. Gated on `VIRTIO_NET_F_MQ`. Populated with the
    /// offered `queue_pairs` when multiqueue is advertised (`queue_pairs > 1`
    /// on an MSI-X transport — see `device_features`); the guest reads it at
    /// probe (`virtnet_probe`, gated on F_MQ) to size its queue set. Left zero
    /// for a single-pair device, where F_MQ is not advertised so the field is
    /// unread (don't populate config space for an unadvertised feature).
    pub(crate) max_virtqueue_pairs: u16,
    /// Initial MTU. Gated on `VIRTIO_NET_F_MTU`. v0 does NOT advertise
    /// MTU, so this field is unread.
    pub(crate) mtu: u16,
}

// SAFETY: `VirtioNetConfig` is `repr(C, packed)`. With `packed` the
// alignment is 1 and there is no inter-field padding by definition
// (every field is byte-aligned). All fields are integer / fixed-size
// byte-array types for which every bit pattern is a valid value, so
// reading arbitrary bytes into the struct yields a well-defined
// value. The struct is `Copy`, `Send`, and `Sync` (all primitives),
// satisfying the `ByteValued` supertrait bounds. Total size is
// verified against the kernel uapi layout by the
// `VIRTIO_NET_CONFIG_SIZE` const assertion below.
unsafe impl ByteValued for VirtioNetConfig {}

/// Size of the populated portion of net config space (12 bytes:
/// mac 6 + status 2 + max_virtqueue_pairs 2 + mtu 2). Reads at
/// config-space offsets `>= VIRTIO_NET_CONFIG_SIZE` return zero per
/// virtio-v1.2 §4.2.2.2.
pub(crate) const VIRTIO_NET_CONFIG_SIZE: usize = std::mem::size_of::<VirtioNetConfig>();
// Compile-time field-offset checks against the kernel uapi
// `struct virtio_net_config` layout. A mismatch here means either
// Rust's `repr(C, packed)` introduced a divergence from the kernel's
// `__attribute__((packed))` layout, or a field was added/removed —
// in either case the guest would read garbage from a misaligned
// field. Failing to compile is preferable to silently serving wrong
// bytes. Citations: `include/uapi/linux/virtio_net.h` and the
// `virtio_bindings::virtio_net` mod whose own `_padding` static
// assertions pin the same offsets.
const _: () = assert!(std::mem::offset_of!(VirtioNetConfig, mac) == 0x00);
const _: () = assert!(std::mem::offset_of!(VirtioNetConfig, status) == 0x06);
const _: () = assert!(std::mem::offset_of!(VirtioNetConfig, max_virtqueue_pairs) == 0x08);
const _: () = assert!(std::mem::offset_of!(VirtioNetConfig, mtu) == 0x0A);
const _: () = assert!(VIRTIO_NET_CONFIG_SIZE == 12);

// ---------------------------------------------------------------------------
// Device struct
// ---------------------------------------------------------------------------

/// Virtio-net MMIO device with in-VMM loopback backend.
///
/// All state behind a single struct — no separate transport layer.
/// The caller holds this in a `PiMutex` and dispatches MMIO
/// reads/writes; the loopback work runs inline on the vCPU thread
/// inside `mmio_write(QUEUE_NOTIFY)`. See parent module docs for the
/// no-worker-thread rationale.
pub struct VirtioNet {
    /// Data + control virtqueues. Length is `2 * queue_pairs` (the data
    /// vqs) plus one trailing control vq when multiqueue is offered
    /// (`queue_pairs > 1`). Data vqs are interleaved per the kernel
    /// `rxq2vq`/`txq2vq` ordering (`drivers/net/virtio_net.c`): vq `2i` =
    /// RX of pair `i`, vq `2i+1` = TX of pair `i`; the control vq, when
    /// present, is the last index (`2 * queue_pairs`). A `Vec` rather than
    /// a fixed array because the pair count is per-device
    /// (`NetConfig::queue_pairs`). For the default single pair the layout
    /// is byte-identical to the pre-multiqueue device: `[RX(0), TX(1)]`,
    /// no control vq.
    queues: Vec<Queue>,
    /// Queue-pairs offered to the guest (`NetConfig::queue_pairs`, clamped
    /// to `>= 1`). Reported in config space as `max_virtqueue_pairs` and
    /// gates `VIRTIO_NET_F_MQ` + `VIRTIO_NET_F_CTRL_VQ` (offered only when
    /// `> 1`). The guest reads this at probe and commits to
    /// `min(num_online_cpus, queue_pairs)` pairs.
    queue_pairs: u16,
    /// Active queue-pair count, set by the guest via the control vq
    /// (`VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET`). Starts at `1` (virtio-net's
    /// probe default — the guest sends `VQ_PAIRS_SET` to raise it) and is
    /// bounded to `[1, queue_pairs]` by the cvq consumer. A kick on a TX
    /// queue of an inactive pair (`>= curr_queue_pairs`) is inert.
    curr_queue_pairs: u16,
    queue_select: u32,
    device_features_sel: u32,
    driver_features_sel: u32,
    driver_features: u64,
    /// FSM state bits per virtio-v1.2 §3.1.1 plus the
    /// `VIRTIO_CONFIG_S_NEEDS_RESET` bit set on the queue-poison
    /// path. Plain `u32` (not `AtomicU32`): virtio-net processes
    /// every MMIO write inline on the vCPU thread that took the
    /// kick, and there is no worker thread in v0, so all reads
    /// and writes of `device_status` happen on that single
    /// thread. A future TAP / AF_PACKET / threaded-NAPI backend
    /// that moves the drain off-thread would need to convert this
    /// (along with `interrupt_status` and `queue_poisoned`) to
    /// atomic types as part of that migration. virtio-blk's
    /// equivalent uses `Arc<AtomicU32>` because its worker thread
    /// can race-fire `fetch_or(NEEDS_RESET)` with the vCPU's FSM
    /// walk; that race does not exist here.
    device_status: u32,
    /// MMIO interrupt-status register. Two bits set by this
    /// device:
    ///   - `VIRTIO_MMIO_INT_VRING`: on used-ring publish via
    ///     `signal_used` (every drain that advances either
    ///     queue's used.idx).
    ///   - `VIRTIO_MMIO_INT_CONFIG`: on the queue-poison path via
    ///     `signal_queue_poisoned` (paired with NEEDS_RESET in
    ///     `device_status`). Spec-compliant per virtio-v1.2 and
    ///     matches cloud-hypervisor's hostile-guest shutdown
    ///     signal. The kernel callback
    ///     `virtnet_config_changed_work` bails when
    ///     `VIRTIO_NET_F_STATUS` isn't negotiated, so the
    ///     INT_CONFIG dispatch is effectively a one-time
    ///     workqueue-wake on device death — accepted cost for
    ///     spec-compliance and cross-VMM convergence. Operators
    ///     can also detect poison out-of-band via `mmio_read(STATUS)
    ///     & NEEDS_RESET` plus the host counter.
    ///
    /// Cleared by the guest's `INTERRUPT_ACK` writes. Plain
    /// `u32` for the same single-thread reason as `device_status`
    /// — see that field's doc for the invariant and the
    /// future-migration note.
    interrupt_status: u32,
    /// v0 holds this at zero. The kernel driver's
    /// `virtio_config_changed` callback (`virtnet_config_changed`
    /// in `drivers/net/virtio_net.c`) is the only consumer;
    /// nothing in this device mutates config-space content after
    /// construction (MAC is fixed at `new()`, STATUS/MQ/MTU stay
    /// zero), so the generation field never advances. Plain `u32`
    /// (matches `device_status` and `interrupt_status`) — the
    /// single-thread MMIO path means no atomic is needed for the
    /// always-zero v0 value. Upgrade to `AtomicU32` if a future
    /// runtime config-space mutation (e.g. link-status changes
    /// if `VIRTIO_NET_F_STATUS` is later advertised) requires
    /// generation tracking off the vCPU thread.
    config_generation: u32,
    /// Eventfd for KVM irqfd — signals guest interrupt on the INTx path.
    irq_evt: EventFd,
    /// Shared MSI-X delivery state, `Some` only on the PCI transport once the
    /// facade installs it via [`Self::set_msix_state`]; `None` for virtio-MMIO
    /// and aarch64 (INTx-only). When `Some` AND [`MsixState::enabled`], the
    /// signal paths route the interrupt through the guest-assigned MSI-X vector
    /// instead of `irq_evt`. Shared (`Arc<PiMutex<…>>`) with the PCI facade,
    /// which mutates it from config-space decode and owns the KVM GSI-route side;
    /// the lock is uncontended in practice — every access (facade config writes
    /// and these core signals) is on the vCPU thread under the `PciBus` lock.
    msix: Option<Arc<PiMutex<MsixState>>>,
    /// Guest memory reference. Set once at VM init by `set_mem` before
    /// any vCPU runs (and therefore before any QUEUE_NOTIFY can fire).
    /// Wrapped in `Arc<OnceLock<…>>` to mirror virtio-blk's pattern:
    /// `set_mem` runs once, post-init reads on the TX kick path are
    /// lock-free `OnceLock::get` calls returning `&GuestMemoryMmap`,
    /// and a future TAP / AF_PACKET / threaded-NAPI worker can cheaply
    /// share the same handle by cloning the outer `Arc`. The previous
    /// `Option<GuestMemoryMmap>` shape forced a full
    /// `GuestMemoryMmap::clone` on every `process_tx_loopback` call —
    /// the inner `Arc<RegionMmap>` chain is cheap to clone but it is
    /// still atomic refcount traffic per TX kick, which is pure
    /// overhead for a value the device never mutates after init.
    mem: Arc<OnceLock<GuestMemoryMmap>>,
    /// One-shot guard so the "queue notify before set_mem" warning
    /// fires at most once per device instance. Mirrors the virtio-blk
    /// `mem_unset_warned` field. Latched with `Relaxed` because the
    /// log message ordering is not correctness-critical. Without it, a
    /// buggy caller that issues N notifies before `set_mem` would
    /// flood the log with N copies of the same line.
    mem_unset_warned: Arc<AtomicBool>,
    /// Static config-space content: mac, plus `max_virtqueue_pairs` when
    /// multiqueue is offered (`queue_pairs > 1`); STATUS/MTU stay zero (those
    /// features are not advertised). Built at construction from `NetConfig`;
    /// the bytes are `byte_valued` and copied directly into the MMIO read
    /// response when the guest reads at offsets `0x100..0x100+config_size`.
    config: VirtioNetConfig,
    /// Cumulative event counters. `Arc` so external monitor observers
    /// can read them without holding any device borrow.
    counters: Arc<VirtioNetCounters>,
    /// Per-device reusable scratch buffer for one TX frame. Sized by
    /// `resize` to the actual frame length on each TX iteration.
    /// Allocated once and reused across all TX requests; the
    /// underlying capacity grows monotonically up to `MAX_FRAME_SIZE`,
    /// at which point all subsequent TX is amortized to zero
    /// allocation.
    tx_frame_scratch: Vec<u8>,
    /// Per-queue sticky "this queue's avail-ring iterator is
    /// structurally broken; stop calling `iter()` on it" flags,
    /// indexed by virtqueue index. Set ONLY when the corresponding
    /// queue's avail-ring iterator returns `Err(_)` — most commonly
    /// `Error::InvalidAvailRingIndex` (avail.idx more than
    /// `queue.size` ahead of `next_avail`, virtio-v1.2 §2.7.13.3
    /// violation; check sits at queue.rs:707-709 in
    /// `AvailIter::new`), but any structural queue error is
    /// treated identically (cloud-hypervisor convergence — all
    /// `iter()` Err variants represent driver-side state
    /// corruption that cannot recover without a virtio reset).
    ///
    /// **`add_used` failures do NOT poison.** A failed `add_used`
    /// (TX or RX, success or recycle paths) is a transient
    /// used-ring GPA mapping problem — the next QUEUE_NOTIFY may
    /// find the GPA mapped (e.g. if the guest re-binds the used
    /// ring). Counting via `tx_add_used_failures` /
    /// `rx_add_used_failures` gives operator visibility without
    /// permanently halting the queue. virtio-blk follows the same
    /// rule: add_used failures bump io_errors but do NOT set
    /// NEEDS_RESET. Poison is reserved for structural avail.idx
    /// violations.
    ///
    /// Without these flags, every subsequent `pop_descriptor_chain`
    /// (the default `QueueT` impl that swallows the error and
    /// returns `None` — virtio-queue queue.rs:573-587) would let
    /// the next QUEUE_NOTIFY re-trip the same error. virtio-net
    /// has NO `enable_notification` / `disable_notification`
    /// bracket around the drain (no EVENT_IDX negotiated), so the
    /// re-trip happens once per MMIO exit and the function
    /// returns — NOT a livelock at full vCPU cost. The harm
    /// without the flag is concrete but bounded: (a) the
    /// per-event counter taxonomy is violated (counter reflects
    /// kick rate rather than poison-event rate), (b) operators
    /// have no `mmio_read(STATUS)`-visible signal that the device
    /// is wedged, and (c) every kick floods the host log with the
    /// same error line. The flags fix all three.
    ///
    /// **Per-queue, not per-device.** A hostile guest can poison
    /// RX without poisoning TX (or vice versa). Per-queue flags
    /// let the operator's failure-dump distinguish "RX poisoned,
    /// TX fine" from "TX poisoned" from "both poisoned" — a
    /// device-level flag would conflate the three failure modes
    /// and hide which queue the guest broke. The drain consults
    /// the matching flag at each pop site (TX in
    /// `pop_and_capture_tx`, RX in `try_loopback_to_rx`); it does
    /// NOT short-circuit the whole drain on a one-side poison.
    /// When EITHER flag is set the device reports
    /// `VIRTIO_CONFIG_S_NEEDS_RESET` to the guest (single bit, no
    /// per-queue NEEDS_RESET in the virtio-v1.2 spec) but the
    /// per-queue flags govern internal short-circuit behavior.
    ///
    /// Both flags clear only on `VirtioNet::reset()`, matching the
    /// device's `VIRTIO_CONFIG_S_NEEDS_RESET` (virtio-v1.2 §2.1.1
    /// bit 0x40) behaviour: the only escape is a STATUS=0 MMIO
    /// write.
    ///
    /// Single-thread invariant: virtio-net processes all
    /// MMIO/QUEUE_NOTIFY traffic inline on the vCPU thread that
    /// took the kick (no worker thread in v0). All reads and
    /// writes of these flags happen on that thread, so plain
    /// `bool`s are sufficient — no atomics needed. Same rationale
    /// `device_status` and `interrupt_status` use plain `u32`. A
    /// future TAP / AF_PACKET / threaded-NAPI backend that moves
    /// the drain off-thread would need to convert these flags
    /// (along with `device_status` and `interrupt_status`) to
    /// atomic types as part of that migration.
    queue_poisoned: Vec<bool>,
}

impl VirtioNet {
    /// Create a new virtio-net device with the given configuration.
    pub fn new(config: NetConfig) -> Self {
        let irq_evt =
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-net irq eventfd");
        // Clamp to [spec minimum (VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MIN = 1),
        // MAX_QUEUE_PAIRS] — a 0 becomes a single pair; an over-large request
        // is bounded so the queue allocation and the highest notify offset stay
        // within range.
        let queue_pairs = config.queue_pairs.clamp(1, MAX_QUEUE_PAIRS);
        // 2 data vqs per pair (interleaved RX/TX), plus one control vq when
        // multiqueue is offered (queue_pairs > 1 → F_CTRL_VQ). The single-pair
        // default builds exactly [RX(0), TX(1)] — byte-identical to the
        // pre-multiqueue device, no control vq.
        let num_queues = queue_pairs as usize * 2 + usize::from(queue_pairs > 1);
        let queues: Vec<Queue> = (0..num_queues)
            .map(|_| Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"))
            .collect();
        VirtioNet {
            queues,
            queue_pairs,
            // virtio-net's probe default; the guest raises it via the control
            // vq (VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET).
            curr_queue_pairs: 1,
            queue_select: 0,
            device_features_sel: 0,
            driver_features_sel: 0,
            driver_features: 0,
            device_status: 0,
            interrupt_status: 0,
            config_generation: 0,
            irq_evt,
            msix: None,
            mem: Arc::new(OnceLock::new()),
            mem_unset_warned: Arc::new(AtomicBool::new(false)),
            config: VirtioNetConfig {
                mac: config.mac,
                status: 0,
                // Valid only under VIRTIO_NET_F_MQ (offered iff > 1). For a
                // single pair the bit is not advertised, so the guest never
                // reads this field — zero is correct (don't populate config
                // space for an unadvertised feature).
                max_virtqueue_pairs: if queue_pairs > 1 { queue_pairs } else { 0 },
                mtu: 0,
            },
            counters: Arc::new(VirtioNetCounters::default()),
            tx_frame_scratch: Vec::with_capacity(MAX_FRAME_SIZE),
            queue_poisoned: vec![false; num_queues],
        }
    }

    /// Eventfd for KVM irqfd registration.
    pub fn irq_evt(&self) -> &EventFd {
        &self.irq_evt
    }

    /// Install the shared MSI-X delivery state (PCI transport only). The facade
    /// holds the other `Arc` clone and configures it from config-space writes;
    /// the core reaches it from the signal paths when MSI-X is enabled. INTx
    /// transports (virtio-MMIO, aarch64) never call this and stay `None`.
    pub(crate) fn set_msix_state(&mut self, msix: Arc<PiMutex<MsixState>>) {
        self.msix = Some(msix);
    }

    /// Set guest memory reference. Must be called before starting
    /// vCPUs. `OnceLock::set` returns `Err` if the slot is already
    /// populated; the production wiring (`init_virtio_net`) calls
    /// `set_mem` exactly once per device, so the `Err` branch is
    /// unreachable in normal operation. Log on `Err` rather than panic
    /// so a future re-wire bug surfaces as a warning instead of
    /// aborting (a panic here could land mid-teardown when the caller
    /// is already unwinding). Mirrors virtio-blk's `set_mem`.
    pub fn set_mem(&mut self, mem: GuestMemoryMmap) {
        if self.mem.set(mem).is_err() {
            tracing::warn!(
                "virtio-net: set_mem called on already-initialised \
                 device; guest memory binding unchanged (mem is set \
                 once at boot and preserved across reset())"
            );
        }
    }

    /// Cloneable handle to the host-observability counters. The
    /// monitor thread holds an Arc to read counters without locking
    /// the device.
    pub fn counters(&self) -> Arc<VirtioNetCounters> {
        Arc::clone(&self.counters)
    }

    // ===================== transport-neutral core API =====================
    //
    // Semantic register/queue operations shared by every transport
    // facade (virtio-MMIO in `mmio.rs`, virtio-pci-modern in `pci.rs`).
    // Each method is ONE logical operation with its spec gate
    // (`queue_config_allowed` / `features_write_allowed`) applied
    // INSIDE the method, so a facade can never bypass a gate by poking
    // a field — the fields stay private to this module and the facades
    // (sibling submodules) reach them only through these `pub(crate)`
    // ops. Facades own only the transport-specific decode (which
    // offset / cap field maps to which op, and the access width); ALL
    // device behaviour (the status FSM, feature negotiation, queue ring
    // assembly, notify, interrupt bookkeeping) lives here exactly once,
    // so the two transports cannot drift.

    /// Current device-status FSM bits (virtio-v1.2 §2.1).
    pub(crate) fn device_status(&self) -> u32 {
        self.device_status
    }

    /// Pending interrupt-status bits (cleared by [`Self::ack_interrupt`]).
    pub(crate) fn interrupt_status(&self) -> u32 {
        self.interrupt_status
    }

    /// Config-space generation counter (v0 holds zero — see the field).
    pub(crate) fn config_generation(&self) -> u32 {
        self.config_generation
    }

    /// Number of virtqueues the device exposes: `2 * queue_pairs` data
    /// queues, plus a trailing control vq when multiqueue is offered
    /// (`queue_pairs > 1`). The PCI common-cfg `num_queues` register serves
    /// this (see the facade's `CC_NUM_QUEUES`) so the guest sets up exactly
    /// the virtqueues that exist.
    pub(crate) fn num_queues(&self) -> usize {
        self.queues.len()
    }

    /// Active queue-pair count (`curr_queue_pairs`). Starts at 1 and is only
    /// raised by a well-formed `VQ_PAIRS_SET` on the control vq, bounded to
    /// `[1, queue_pairs]` by [`Self::eval_ctrl_chain`]. Test-only window onto
    /// the invariant that hostile control-vq input can never push it out of
    /// range (exercised by the cvq chain-mutation proptest).
    #[cfg(test)]
    pub(crate) fn curr_queue_pairs(&self) -> u16 {
        self.curr_queue_pairs
    }

    /// Offered device features for the latched select window
    /// (`device_features_sel`): window 0 = bits 0..31, window 1 = bits
    /// 32..63, any other window = 0. The guest writes the select
    /// register then reads this window.
    pub(crate) fn device_features_window(&self) -> u32 {
        match self.device_features_sel {
            0 => self.device_features() as u32,
            1 => (self.device_features() >> 32) as u32,
            _ => 0,
        }
    }

    /// Immutable max ring size of the selected queue (0 if the selector is out
    /// of range). This is the advertised ceiling, NOT the guest-configured size
    /// — the PCI common-cfg `queue_size` register serves [`Self::queue_size`]
    /// (the configured value), not this. Used only to seed the reset value.
    pub(crate) fn queue_max_size(&self) -> u32 {
        self.selected_queue()
            .map(|i| self.queues[i].max_size() as u32)
            .unwrap_or(0)
    }

    /// Actual ring size the guest configured for the selected queue (0 if the
    /// selector is out of range). The virtio-pci common-cfg `queue_size`
    /// register is read-write (virtio-v1.2 §4.1.4.3): `virtio_queue::Queue`
    /// initializes and resets it to `max_size` (queue.rs:288/353) and
    /// `set_size` stores the guest's value, so a read-back returns the
    /// currently-configured size — what the PCI facade must serve, not the
    /// immutable max.
    pub(crate) fn queue_size(&self) -> u32 {
        self.selected_queue()
            .map(|i| self.queues[i].size() as u32)
            .unwrap_or(0)
    }

    /// Ready flag of the selected queue (0 if the selector is out of range).
    pub(crate) fn queue_ready(&self) -> u32 {
        self.selected_queue()
            .map(|i| self.queues[i].ready() as u32)
            .unwrap_or(0)
    }

    /// `queue_notify_off` for the selected queue: notify offsets map in queue
    /// order, so it equals the (in-range) selector index, and 0 for an
    /// out-of-range selector. Clamped — unlike the raw `queue_select`
    /// read-back — so a guest that latched a bogus selector cannot read back a
    /// notify offset outside the advertised range and compute a wild notify
    /// address. The out-of-range clamp value (0) aliases queue 0's (RX) real
    /// offset; this is harmless because safety rests on `notify_queue` driving
    /// the loopback only for a TX-queue kick on an active pair, not on this
    /// read-back value — a guest reading back 0 cannot drive a stray drain.
    /// The returned index pairs with
    /// `pci::NOTIFY_OFF_MULTIPLIER` (wire address = index × multiplier).
    pub(crate) fn queue_notify_off(&self) -> u32 {
        self.selected_queue().map(|i| i as u32).unwrap_or(0)
    }

    /// Selected queue's assembled 64-bit (desc, avail, used) ring addresses,
    /// or `None` if the selector is out of range. The PCI common-cfg programs
    /// each as two 32-bit halves (LO then HI via the kernel's
    /// `vp_iowrite64_twopart`); this reads back the merged values so a test
    /// can pin that the high dword lands in the high 32 bits (not dropped or
    /// swapped with the low half). Test-only: the production transports never
    /// read the ring addresses back (the guest owns them), so this would be
    /// dead code outside `cfg(test)`.
    #[cfg(test)]
    pub(crate) fn selected_queue_ring_addrs(&self) -> Option<(u64, u64, u64)> {
        self.selected_queue().map(|i| {
            (
                self.queues[i].desc_table(),
                self.queues[i].avail_ring(),
                self.queues[i].used_ring(),
            )
        })
    }

    /// Latched device-feature-select window — read back by the PCI
    /// common-cfg `device_feature_select` register (the MMIO transport
    /// never reads it back, but the PCI facade serves a register read).
    pub(crate) fn device_features_sel(&self) -> u32 {
        self.device_features_sel
    }

    /// Latched driver-feature-select window — read back by the PCI
    /// common-cfg `driver_feature_select` register.
    pub(crate) fn driver_features_sel(&self) -> u32 {
        self.driver_features_sel
    }

    /// Latched queue selector. The PCI common-cfg `queue_select`
    /// read-back returns it, and `queue_notify_off` equals it (notify
    /// offsets map in queue order, so the selected queue's notify
    /// offset is its index).
    pub(crate) fn queue_select(&self) -> u32 {
        self.queue_select
    }

    /// Copy `data.len()` bytes of device config space starting at byte
    /// `offset` (offset 0 = `mac[0]`). Reads past the populated layout
    /// return zero (virtio-v1.2 §4.2.2.2).
    pub(crate) fn config_bytes(&self, offset: usize, data: &mut [u8]) {
        // SAFETY: `VirtioNetConfig` is `ByteValued` — every bit pattern
        // of the underlying bytes is a valid value, so viewing it as a
        // byte slice is sound.
        let config_bytes = self.config.as_slice();
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = config_bytes.get(offset + i).copied().unwrap_or(0);
        }
    }

    /// Latch the device-feature-select window index (ungated).
    pub(crate) fn set_device_features_sel(&mut self, sel: u32) {
        self.device_features_sel = sel;
    }

    /// Latch the driver-feature-select window index (ungated).
    pub(crate) fn set_driver_features_sel(&mut self, sel: u32) {
        self.driver_features_sel = sel;
    }

    /// Latch the queue selector (ungated).
    pub(crate) fn set_queue_select(&mut self, sel: u32) {
        self.queue_select = sel;
    }

    /// Merge `val` into the negotiated driver features at the latched
    /// driver-feature-select window. No-op once features are locked
    /// (`features_write_allowed` false), per virtio-v1.2 §3.1.1.
    pub(crate) fn set_driver_features_window(&mut self, val: u32) {
        if !self.features_write_allowed() {
            return;
        }
        match self.driver_features_sel {
            0 => self.driver_features = (self.driver_features & 0xFFFF_FFFF_0000_0000) | val as u64,
            1 => {
                self.driver_features =
                    (self.driver_features & 0x0000_0000_FFFF_FFFF) | ((val as u64) << 32)
            }
            _ => {}
        }
    }

    /// Set the selected queue's ring size. Gated on
    /// `queue_config_allowed` (pre-DRIVER_OK only); ignored otherwise.
    pub(crate) fn set_queue_size(&mut self, size: u16) {
        if !self.queue_config_allowed() {
            return;
        }
        if let Some(i) = self.selected_queue() {
            self.queues[i].set_size(size);
        }
    }

    /// Set the selected queue's ready flag. Gated on
    /// `queue_config_allowed`; ignored otherwise.
    pub(crate) fn set_queue_ready(&mut self, ready: bool) {
        if !self.queue_config_allowed() {
            return;
        }
        if let Some(i) = self.selected_queue() {
            self.queues[i].set_ready(ready);
        }
    }

    /// Set the selected queue's descriptor-table address (low and/or
    /// high 32 bits; `None` leaves that half unchanged). Gated on
    /// `queue_config_allowed`; ignored otherwise.
    pub(crate) fn set_queue_desc_addr(&mut self, lo: Option<u32>, hi: Option<u32>) {
        if !self.queue_config_allowed() {
            return;
        }
        if let Some(i) = self.selected_queue() {
            self.queues[i].set_desc_table_address(lo, hi);
        }
    }

    /// Set the selected queue's avail-ring (driver-area) address.
    /// Gated on `queue_config_allowed`; ignored otherwise.
    pub(crate) fn set_queue_avail_addr(&mut self, lo: Option<u32>, hi: Option<u32>) {
        if !self.queue_config_allowed() {
            return;
        }
        if let Some(i) = self.selected_queue() {
            self.queues[i].set_avail_ring_address(lo, hi);
        }
    }

    /// Set the selected queue's used-ring (device-area) address. Gated
    /// on `queue_config_allowed`; ignored otherwise.
    pub(crate) fn set_queue_used_addr(&mut self, lo: Option<u32>, hi: Option<u32>) {
        if !self.queue_config_allowed() {
            return;
        }
        if let Some(i) = self.selected_queue() {
            self.queues[i].set_used_ring_address(lo, hi);
        }
    }

    /// Notify the device that the guest kicked queue `queue_idx`.
    ///
    /// Data virtqueues interleave per the kernel rxq2vq/txq2vq ordering: RX
    /// of pair `i` at even index `2i`, TX at odd `2i+1`. Only a TX kick
    /// drives the loopback (TX→RX of the same pair); an RX kick is a no-op
    /// (the next TX picks up newly posted RX buffers). A TX kick on an
    /// inactive pair (`pair >= curr_queue_pairs`) is inert: the guest must
    /// not transmit on a pair it has not activated via the control vq
    /// (`VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET`), and the legacy-INTx
    /// single-asserter invariant (see the PCI facade module doc) requires no
    /// second independent TX asserter. The control vq — present only when
    /// multiqueue is offered (`queue_pairs > 1`) — is the trailing index
    /// (`2 * queue_pairs`) and routes to [`Self::process_ctrl_queue`].
    pub(crate) fn notify_queue(&mut self, queue_idx: u32) {
        let idx = queue_idx as usize;
        // Control vq (offered iff multiqueue): the trailing index.
        if self.queue_pairs > 1 && idx == self.queue_pairs as usize * 2 {
            self.process_ctrl_queue();
            return;
        }
        // Data vq: a TX kick (odd index) on an active pair drives the
        // loopback. RX kicks (even) and inactive / out-of-range pairs are
        // no-ops (the `pair < curr_queue_pairs` gate bounds the index).
        if idx % 2 == 1 {
            let pair = idx / 2;
            if pair < self.curr_queue_pairs as usize {
                self.process_tx_loopback(pair);
            }
        }
    }

    /// Process the control virtqueue (index `2 * queue_pairs`, present only
    /// when multiqueue is offered). The sole command class v1 handles is
    /// `VIRTIO_NET_CTRL_MQ` / `VQ_PAIRS_SET`, which sets the active
    /// queue-pair count (`curr_queue_pairs`).
    ///
    /// Chain layout (`struct virtio_net_ctrl_hdr` uapi doc + virtio-v1.2
    /// §5.1.6.5.1): the header is the first sg entry and the ack/status the
    /// LAST, command data in between. `virtnet_send_command_reply`
    /// (drivers/net/virtio_net.c) builds three descriptors — readable hdr
    /// (`class u8`, `cmd u8`), readable payload (`virtio_net_ctrl_mq`:
    /// `virtqueue_pairs __virtio16`, LE under VERSION_1), writable status
    /// (`u8`). The guest confirms by polling the cvq used ring
    /// (`virtnet_send_command_reply`, :3579) then reading
    /// `status == VIRTIO_NET_OK`, so the device MUST write the status byte and
    /// `add_used` is gated on that write succeeding (the F15 rule — never
    /// publish a chain whose status the guest cannot observe). The
    /// `curr_queue_pairs` state change is likewise applied ONLY after a
    /// successful `add_used`, so device and guest never disagree on the active
    /// count. Processing is gated on the guest having negotiated
    /// `VIRTIO_NET_F_CTRL_VQ` (the feature the cvq embodies) in addition to
    /// DRIVER_OK.
    ///
    /// Hostile-input defense, three classes:
    ///   - bad COMMAND (too few readable bytes, unknown `(class, cmd)`, or
    ///     `virtqueue_pairs` outside `[1, queue_pairs]`) → `VIRTIO_NET_ERR` via
    ///     the status descriptor, `curr_queue_pairs` unchanged. Recoverable; no
    ///     NEEDS_RESET (qemu NAKs these too).
    ///   - no device-writable status descriptor → dropped WITHOUT `add_used`
    ///     (F15); the cvq stays usable for the next command. Recoverable; no
    ///     NEEDS_RESET (a documented divergence from qemu, which `virtio_error`s
    ///     this — see `eval_ctrl_chain`).
    ///   - dead RING (`iter()` structural error) → poison + NEEDS_RESET (the
    ///     device-death class, matching the data path and qemu's structural
    ///     `virtio_error`).
    ///
    /// Never an out-of-bounds read or write. Matches cloud-hypervisor's
    /// validate-then-ACK control-queue model.
    fn process_ctrl_queue(&mut self) {
        // DRIVER_OK gate (as in process_tx_loopback): ignore a kick before
        // the driver finished initialisation.
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            return;
        }
        // F_CTRL_VQ gate: the control vq IS that feature, so process it only if
        // the guest negotiated the bit. A guest that accepted F_MQ but declined
        // F_CTRL_VQ (non-conformant — virtnet BUG_ONs without it) has no cvq, so
        // a kick at this index is spurious; ignore it rather than act on a
        // feature the guest did not negotiate. (Harmless even unguarded —
        // validate+ack only, and the curr_queue_pairs mutation is gated on a
        // successful publish below — but the device must honor the negotiated
        // feature set.)
        if self.driver_features & (1u64 << VIRTIO_NET_F_CTRL_VQ) == 0 {
            return;
        }
        let cvq_idx = self.queue_pairs as usize * 2;
        // Clone the `Arc<OnceLock>` once (a cheap atomic bump) so the borrow
        // of guest memory does not freeze the rest of `self` — same pattern
        // as `process_tx_loopback`.
        let mem_arc = Arc::clone(&self.mem);
        let Some(mem) = mem_arc.get() else {
            return;
        };
        // The control vq is poisoned like any data queue: a structural
        // iter() error is non-recoverable until a guest virtio reset.
        if self.queue_poisoned[cvq_idx] {
            return;
        }
        let mut had_used_ring_publish = false;
        // The false->true cvq-poison transition observed during THIS drain (a
        // structural iter() error). Mirrors the data path's tx/rx_just_poisoned:
        // the NEEDS_RESET signal fires once per transition, post-loop, AFTER any
        // pending used-ring publishes. The entry gate above (`queue_poisoned[cvq_idx]`
        // early-return) already makes a re-kick on an already-poisoned cvq a no-op,
        // so reaching the poison arm IS the transition.
        let mut cvq_just_poisoned = false;
        loop {
            // Two-step iter()-then-drop pop, mirroring `pop_and_capture_tx`
            // so the queue borrow is scoped to one statement.
            enum IterStep<C> {
                Chain(C),
                Empty,
                Poisoned(VirtioQueueError),
            }
            let step: IterStep<_> = {
                let q = &mut self.queues[cvq_idx];
                match q.iter(mem) {
                    Ok(mut it) => match it.next() {
                        Some(c) => IterStep::Chain(c),
                        None => IterStep::Empty,
                    },
                    Err(e) => IterStep::Poisoned(e),
                }
            };
            let chain = match step {
                IterStep::Empty => break,
                IterStep::Poisoned(err) => {
                    // A structural iter() error (corrupt avail/desc ring)
                    // permanently disables the control vq — every later iter()
                    // would re-error. Poison it (same rule as the data queues;
                    // cloud-hypervisor convergence) AND set NEEDS_RESET post-loop
                    // (status bit only — `mark_needs_reset`; the data path fires
                    // the full `signal_queue_poisoned`, but the cvq is polled so
                    // the inert config interrupt is skipped). This is the SAME
                    // device-death class qemu signals: its virtqueue_pop routes
                    // every structural ring fault through virtio_error →
                    // VIRTIO_CONFIG_S_NEEDS_RESET (hw/virtio/virtio.c). A STATUS
                    // read then sees the device needs reset. (The guest's cvq
                    // poller spins on virtqueue_is_broken, NOT device_status, so
                    // this does not unblock a self-corrupted-ring guest — it is
                    // the honest host-visible signal + data-path parity, not a
                    // hang fix.) This differs from a malformed *command*
                    // (eval_ctrl_chain → VIRTIO_NET_ERR) and a no-writable-status
                    // *chain* (dropped, ring still usable): those are recoverable
                    // and do NOT set NEEDS_RESET — only a dead ring does.
                    self.queue_poisoned[cvq_idx] = true;
                    self.counters.record_invalid_avail_idx();
                    cvq_just_poisoned = true;
                    tracing::warn!(
                        err = %err,
                        "virtio-net control-vq iter() failed; poisoning + \
                         NEEDS_RESET until guest reset (any structural queue \
                         error is non-recoverable)"
                    );
                    break;
                }
                IterStep::Chain(c) => c,
            };
            let head = chain.head_index();
            // Parse + validate the chain off the queue borrow.
            let Some((status_addr, ack, new_pairs)) = self.eval_ctrl_chain(chain, mem) else {
                // No device-writable status descriptor: the guest posted no
                // buffer to observe a reply, so the F15 rule forbids add_used.
                // Drop the chain (it stays owned by the device until reset).
                self.counters.record_ctrl_chain_invalid();
                continue;
            };
            // Write the 1-byte ack BEFORE add_used — the guest reads its
            // status from this descriptor.
            if let Err(e) = mem.write_slice(&[ack], status_addr) {
                // Status GPA unmapped: do NOT add_used (the guest can't see
                // the reply). Queue-state breakage, not a malformed request.
                self.counters.record_ctrl_add_used_failure();
                tracing::warn!(
                    %e,
                    "virtio-net control-vq status write failed (unmapped GPA); \
                     dropping chain"
                );
                continue;
            }
            // A rejected command (NAK) is the event regardless of whether the
            // status publish lands — distinct from a publish failure, and both
            // can co-occur on a malformed-AND-unmapped chain (mirrors the data
            // path's rx_chain_invalid + rx_add_used_failures). Count it now.
            if new_pairs.is_none() {
                self.counters.record_ctrl_chain_invalid();
            }
            // Publish: the device wrote 1 byte (the status) to a writable
            // descriptor, so used.len = 1 (virtio-v1.2 §2.7.8.2). The guest
            // confirms a control command by polling the cvq USED RING
            // (virtnet_send_command_reply, drivers/net/virtio_net.c:3579 —
            // `while (!virtqueue_get_buf(cvq) ...)`) and only then reads the
            // status byte. So the active-pair-count state change is applied
            // ONLY after a successful add_used: a failed publish means the guest
            // never observes completion, and committing curr_queue_pairs there
            // would leave device and guest disagreeing on the active count.
            // Mirrors tx_add_used, which bumps tx_packets only on add_used Ok.
            match self.queues[cvq_idx].add_used(mem, head, 1) {
                Ok(()) => {
                    had_used_ring_publish = true;
                    if let Some(pairs) = new_pairs {
                        self.curr_queue_pairs = pairs;
                        self.counters.record_ctrl_mq_set();
                    }
                }
                Err(e) => {
                    self.counters.record_ctrl_add_used_failure();
                    tracing::warn!(
                        head,
                        %e,
                        "virtio-net control-vq add_used failed (used-ring \
                         address unmapped)"
                    );
                }
            }
        }
        // `virtnet_send_command` polls the control-vq used ring (it does not
        // block on an interrupt), so a signal is not required for
        // correctness; fire one when the used ring advanced for parity with
        // the data path (and so a guest that armed the cvq callback wakes).
        // Ordered BEFORE the poison signal so the guest observes any prior
        // completions before the device-death notice.
        if had_used_ring_publish {
            self.signal_used(cvq_idx);
        }
        // Structural-poison device-death signal: set NEEDS_RESET, STATUS-BIT
        // ONLY, once per false->true transition. Status-only (not the data
        // path's full signal_queue_poisoned) because the cvq is POLLED — the
        // guest spins on the cvq used ring (virtnet_send_command_reply) and arms
        // no cvq callback, so the INT_CONFIG/irqfd kick would wake nothing and
        // only fire the shared vring vector spuriously. A STATUS read still sees
        // the device-death state (data-path + qemu parity). Ordered after
        // signal_used so any prior completions are visible first. Only the
        // dead-ring case reaches here; a malformed command (NAK) or a
        // no-writable-status drop leaves the cvq usable and does not signal.
        if cvq_just_poisoned {
            self.mark_needs_reset();
        }
    }

    /// Walk a control-vq `chain`: collect the leading readable command bytes
    /// (header + payload) and locate the first device-writable status
    /// descriptor. Returns `Some((status_addr, ack, new_pairs))` — the GPA to
    /// write the 1-byte status to, the status byte itself, and the validated
    /// new active-pair count (`Some` only for a well-formed `VQ_PAIRS_SET`
    /// within `[1, queue_pairs]`). Returns `None` only when there is no
    /// device-writable status descriptor (the guest posted no buffer to
    /// observe a reply) — the caller then drops the chain WITHOUT `add_used`
    /// (the F15 rule) and leaves the cvq usable for the next command. A read
    /// error or short/unknown command yields `Some((.., VIRTIO_NET_ERR, None))`
    /// so the device still NAKs through the status descriptor when one exists.
    ///
    /// Descriptor ORDER is not enforced: readable command bytes are collected
    /// wherever they appear (even after the writable status desc), which is safe
    /// because the 4-byte `CTRL_CMD_BYTES` cap bounds the read and every
    /// assembled command is validated below. A conformant guest emits
    /// readable-then-writable (`virtnet_send_command_reply`); a reordered
    /// hostile chain cannot escape the cap or the validation.
    ///
    /// Divergence from qemu: on a malformed chain SHAPE — no writable status
    /// (qemu's "missing headers", hw/net/virtio-net.c:1563) — qemu calls
    /// `virtio_error` (device-kill → NEEDS_RESET); we instead drop the single
    /// malformed chain and keep the cvq alive for subsequent commands (the ring
    /// is still walkable, so this is recoverable, unlike a dead ring). A bad
    /// COMMAND (unknown class/cmd, out-of-range pairs) is NOT a divergence —
    /// qemu NAKs via the status byte too. Only a dead RING (iter() error in
    /// `process_ctrl_queue`) signals NEEDS_RESET, matching qemu's
    /// structural-fault `virtio_error`.
    fn eval_ctrl_chain(
        &self,
        chain: DescriptorChain<&GuestMemoryMmap>,
        mem: &GuestMemoryMmap,
    ) -> Option<(GuestAddress, u8, Option<u16>)> {
        // class(1) + cmd(1) + virtqueue_pairs(2) = 4 bytes is all
        // VQ_PAIRS_SET needs; cap the readable collection at that so a
        // hostile chain cannot force an unbounded read.
        const CTRL_CMD_BYTES: usize = 4;
        let mut cmd = [0u8; CTRL_CMD_BYTES];
        let mut cmd_len = 0usize;
        let mut status: Option<GuestAddress> = None;
        for desc in chain {
            if desc.is_write_only() {
                // The status descriptor. Record the FIRST writable one with
                // room for the 1-byte status; ignore any extras.
                if status.is_none() && desc.len() >= 1 {
                    status = Some(desc.addr());
                }
                continue;
            }
            // Readable command bytes — take only what still fits in `cmd`.
            if cmd_len < CTRL_CMD_BYTES {
                let want = (CTRL_CMD_BYTES - cmd_len).min(desc.len() as usize);
                if want > 0
                    && mem
                        .read_slice(&mut cmd[cmd_len..cmd_len + want], desc.addr())
                        .is_ok()
                {
                    cmd_len += want;
                }
                // A read failure leaves `cmd_len` short → the validation
                // below fails → VIRTIO_NET_ERR (still written to the status
                // descriptor when one exists).
            }
        }
        // No writable status descriptor → the guest cannot observe a reply →
        // the chain is unusable; the caller drops it without add_used.
        let status_addr = status?;
        // Validate: full 4 command bytes, class == CTRL_MQ, cmd ==
        // VQ_PAIRS_SET.
        if cmd_len < CTRL_CMD_BYTES
            || u32::from(cmd[0]) != VIRTIO_NET_CTRL_MQ
            || u32::from(cmd[1]) != VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET
        {
            return Some((status_addr, VIRTIO_NET_ERR as u8, None));
        }
        // virtqueue_pairs is `__virtio16` — little-endian under VERSION_1
        // (which our device requires). The offered `queue_pairs` is the
        // binding upper limit (always <= VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MAX).
        let pairs = u16::from_le_bytes([cmd[2], cmd[3]]);
        if u32::from(pairs) < VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MIN || pairs > self.queue_pairs {
            return Some((status_addr, VIRTIO_NET_ERR as u8, None));
        }
        Some((status_addr, VIRTIO_NET_OK as u8, Some(pairs)))
    }

    /// Clear the interrupt-status bits the guest acknowledged.
    pub(crate) fn ack_interrupt(&mut self, val: u32) {
        self.interrupt_status &= !val;
    }

    /// Apply a device-status write: `0` resets the device, any other
    /// value drives the status FSM via [`Self::set_status`].
    pub(crate) fn write_status(&mut self, val: u32) {
        if val == 0 {
            self.reset();
        } else {
            self.set_status(val);
        }
    }

    // =================== end transport-neutral core API ===================

    /// Feature bits advertised to the guest.
    ///
    /// - `VIRTIO_F_VERSION_1`: modern virtio. Mandatory for the
    ///   12-byte mrg_rxbuf header semantics described at module level.
    /// - `VIRTIO_NET_F_MAC`: device provides the MAC. Without this
    ///   bit the kernel generates a random MAC and the
    ///   `eth_hw_addr_random` path runs; the deterministic MAC from
    ///   `NetConfig` is one of the few values an operator wants to
    ///   pin across runs (for AF_PACKET capture correlation).
    fn device_features(&self) -> u64 {
        let mut features = (1u64 << VIRTIO_F_VERSION_1) | (1u64 << VIRTIO_NET_F_MAC);
        // Multiqueue is offered only when (a) the device was built with >1
        // pair AND (b) the transport carries MSI-X (the PCI facade wired
        // `msix`; virtio-MMIO leaves it `None`). Two reasons, neither a
        // strand-safety claim:
        //   1. Per-queue IRQ steering is the POINT of multiqueue here and needs
        //      a distinct interrupt vector per queue — only MSI-X supplies
        //      that. On a shared INTx/MMIO level line every queue lands on one
        //      IRQ, so multiqueue buys nothing for steering. With MSI-X each data
        //      vq gets its OWN vector + GSI (the guest's EACH policy); SHARED is
        //      the guest's fallback when its queue count exceeds the per-NIC
        //      vector budget, served transparently by the per-queue signal path.
        //   2. The future async-RX backend (TAP/AF_PACKET/threaded-NAPI) will
        //      move RX completion off the vCPU thread and break the legacy-INTx
        //      single-asserter invariant, needing the active resample handler
        //      (a follow-up; see the PCI facade module doc). Gating MQ on MSI-X
        //      keeps that work out of the multiqueue path.
        // NOTE: the gate is "MSI-X capability offered", NOT "guest enabled
        // MSI-X". A guest can negotiate F_MQ and still fall back to a shared
        // INTx line (vp_find_vqs tries three MSI-X policies, then
        // vp_find_vqs_intx, drivers/virtio/virtio_pci_common.c:542). That is
        // sound in v0: every used-ring assert originates on the vCPU thread in
        // process_tx_loopback (single asserter), and the guest's shared-IRQ
        // handler vp_vring_interrupt iterates EVERY vq, scheduling NAPI per vq
        // (virtio_pci_common.c:83-98, "Notify all virtqueues on an interrupt"),
        // so one coalesced level assert drains all queues — nothing strands.
        // F_MQ also requires the control vq: the guest activates pairs via
        // VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET on the cvq, and without F_CTRL_VQ
        // `virtnet_probe` (drivers/net/virtio_net.c) clamps max_queue_pairs to
        // 1. At a single pair, or on a non-MSI-X transport, neither bit is
        // offered — keeping the device byte-identical to the pre-multiqueue
        // facade.
        if self.queue_pairs > 1 && self.msix.is_some() {
            features |= (1u64 << VIRTIO_NET_F_MQ) | (1u64 << VIRTIO_NET_F_CTRL_VQ);
        }
        features
    }

    fn selected_queue(&self) -> Option<usize> {
        let idx = self.queue_select as usize;
        if idx < self.queues.len() { Some(idx) } else { None }
    }

    // Net does not negotiate VIRTIO_RING_F_EVENT_IDX so the combined
    // bit+eventfd pattern is correct here. virtio_blk splits the two
    // because it negotiates EVENT_IDX. Without EVENT_IDX there is no
    // guest-published suppression threshold to consult, so the kick
    // is at the device's discretion. We coalesce to one kick per
    // drain (kick-per-drain, not kick-per-chain): the caller's
    // `had_used_ring_publish` flag accumulates across the whole
    // drain loop and `signal_used` runs once at the end. NAPI on the
    // guest side polls the used ring until empty, so coalescing
    // multiple chain advances under one IRQ is correct and reduces
    // vCPU exits proportional to the burst size.
    //
    // `signal_used` only sets `VIRTIO_MMIO_INT_VRING`. The
    // INT_CONFIG bit is set by the orthogonal `signal_queue_poisoned`
    // path (paired with NEEDS_RESET in device_status) — see that
    // function's doc. Per-event taxonomy: VRING reflects regular
    // used-ring publishes (TX completion, RX delivery, malformed-
    // chain recycle), CONFIG reflects the one-shot device-death
    // signal.
    //
    // The eventfd write below has two possible errno paths,
    // both recoverable:
    //
    //   - `EAGAIN` is impossible at runtime. The eventfd is created
    //     in counter mode (no `EFD_SEMAPHORE`) with `EFD_NONBLOCK`,
    //     so EAGAIN only fires when the internal u64 is at
    //     `u64::MAX - 1` and adding 1 would overflow. That requires
    //     ~2^64 unread kicks in a row — implausible under any
    //     workload because the guest's NAPI consumes (read()s) the
    //     eventfd before the next batch.
    //
    //   - `EBADF` means the device is being torn down: the irqfd
    //     was unregistered or the EventFd dropped. There is no
    //     useful recovery — the VM is shutting down.
    //
    // Either way, the bit-set on `interrupt_status` is the
    // IRQ-handler handshake target — `vm_interrupt`
    // (drivers/virtio/virtio_mmio.c) reads and acks it on each IRQ
    // delivery. The guest does NOT poll this register. We log any
    // errno so a failed write surfaces in tracing rather than
    // silently disappearing.
    fn signal_used(&mut self, queue: usize) {
        // MSI-X path (PCI, once the guest enabled it): deliver the VRING source
        // to virtqueue `queue`'s guest-assigned vector (per-queue IRQ steering).
        // MSI-X does not use the INTx ISR (`interrupt_status`) — the kernel's
        // per-vector handlers never read it (drivers/virtio/virtio_pci_common.c:
        // the ISR read is INTx-only) — so the bit-set + line write below are
        // skipped. On INTx every queue collapses to the one shared line
        // (`queue` ignored); the guest's shared handler vp_vring_interrupt
        // drains all virtqueues per kick.
        if let Some(msix) = &self.msix {
            let mut m = msix.lock();
            if m.enabled() {
                m.signal(IrqSource::Vring { queue });
                return;
            }
        }
        self.interrupt_status |= VIRTIO_MMIO_INT_VRING;
        if let Err(e) = self.irq_evt.write(1) {
            tracing::warn!(%e, "virtio-net irq_evt.write failed");
        }
    }

    /// Surface the queue-poison state to the guest:
    ///   1. Set `VIRTIO_CONFIG_S_NEEDS_RESET` in `device_status`
    ///      (virtio-v1.2 §2.1.1 bit 0x40) so the guest's STATUS
    ///      read sees "device needs reset before it can service
    ///      IO." Cloud-hypervisor uses the same bit for its
    ///      hostile-guest shutdown path. Distinct from the orthogonal
    ///      `VIRTIO_CONFIG_S_FAILED` (bit 0x80, set by
    ///      `set_status` on driver-side feature-negotiation
    ///      violations) — NEEDS_RESET is the device asking for
    ///      help, FAILED is the driver giving up. There is no
    ///      per-queue NEEDS_RESET bit in the spec — the
    ///      device-level bit is the only signal even when only
    ///      one of the two queues is poisoned (per-queue state
    ///      lives in `queue_poisoned[]` and
    ///      `invalid_avail_idx_count`).
    ///   2. Set `VIRTIO_MMIO_INT_CONFIG` in `interrupt_status` so
    ///      the guest's `vm_interrupt` handler dispatches the
    ///      config-change callback alongside the NEEDS_RESET bit.
    ///      Spec-compliant per virtio-v1.2 (config interrupt
    ///      paired with NEEDS_RESET) and matches cloud-hypervisor's
    ///      hostile-guest shutdown signal. virtio-net's kernel
    ///      callback (`virtnet_config_changed_work` in
    ///      `drivers/net/virtio_net.c`) bails when
    ///      `VIRTIO_NET_F_STATUS` isn't negotiated — making the
    ///      callback a no-op rather than productive — so the
    ///      INT_CONFIG dispatch costs one harmless guest-side
    ///      vCPU exit + workqueue wake on device death. That's
    ///      acceptable: the device is already wedged, the cost
    ///      runs once, and the spec-compliance / cross-VMM
    ///      convergence wins outweigh the single wasted exit.
    ///   3. Write the irqfd so KVM delivers the GSI. The
    ///      `vm_interrupt` handler reads INTERRUPT_STATUS and
    ///      dispatches via the set bits.
    ///
    /// **Caller responsibility**: gate this on the false→true
    /// poison transition for a queue. The function itself does
    /// NOT check the flag — callers (TX-side and RX-side poison
    /// arms) must only invoke it when they just transitioned a
    /// queue from clean to poisoned. Re-poisoning an
    /// already-poisoned queue MUST NOT call this — re-firing the
    /// irqfd would generate spurious wakes (counter already
    /// drained by the guest's prior IRQ handler). The counter and
    /// signal must be event-once per false→true transition.
    fn signal_queue_poisoned(&mut self) {
        // NEEDS_RESET is guest-visible device state, set regardless of the
        // interrupt transport (the operator's `mmio_read(STATUS)` surfaces it
        // even if no interrupt is delivered).
        self.mark_needs_reset();
        // MSI-X path (PCI, once enabled): deliver the CONFIG source to its
        // guest-assigned vector. As in `signal_used`, MSI-X skips the INTx ISR
        // (`interrupt_status`) — the kernel's config-change handler is its own
        // MSI-X vector and never reads the ISR byte.
        if let Some(msix) = &self.msix {
            let mut m = msix.lock();
            if m.enabled() {
                m.signal(IrqSource::Config);
                return;
            }
        }
        self.interrupt_status |= VIRTIO_MMIO_INT_CONFIG;
        // Recoverability: EAGAIN requires counter saturation at u64::MAX-1
        // (~1.8e19 unobserved kicks) — implausible. EBADF means
        // the fd closed during shutdown. The NEEDS_RESET +
        // INT_CONFIG bits above are the enduring guest-visible
        // signals: even if this write fails, the operator's
        // `mmio_read(STATUS)` still surfaces NEEDS_RESET. We log
        // any errno so a failed write surfaces in tracing rather
        // than silently disappearing. (Not a `SAFETY:` note — the write
        // is a safe call; SAFETY is reserved for unsafe blocks/impls.)
        if let Err(e) = self.irq_evt.write(1) {
            tracing::warn!(%e, "virtio-net irq_evt.write failed (poison signal)");
        }
    }

    /// Set the guest-visible `VIRTIO_CONFIG_S_NEEDS_RESET` device-status bit
    /// (virtio-v1.2 §2.1.1 bit 0x40) WITHOUT delivering an interrupt.
    /// Idempotent (bitwise OR); the caller gates on the false→true poison
    /// transition. Used for the control-vq structural poison: an
    /// INT_CONFIG/irqfd kick is inert there because the guest POLLS the cvq
    /// (no cvq callback; `virtnet_config_changed_work` only toggles link
    /// carrier, and `vq->broken` is never set by a config-change) — a kick
    /// would wake nothing and only fire the shared vring vector spuriously. The
    /// operator's `mmio_read(STATUS)` still surfaces the device-death state, for
    /// data-path + qemu parity. The data path uses [`Self::signal_queue_poisoned`]
    /// instead (its consumer, NAPI, is IRQ-gated and needs the interrupt).
    fn mark_needs_reset(&mut self) {
        self.device_status |= VIRTIO_CONFIG_S_NEEDS_RESET;
    }

    /// True when device_status has progressed past FEATURES_OK but
    /// not yet reached DRIVER_OK — the window where queue config is
    /// valid.
    fn queue_config_allowed(&self) -> bool {
        self.device_status & S_FEAT == S_FEAT && self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0
    }

    /// True when driver features may be written: DRIVER set,
    /// FEATURES_OK not yet set.
    fn features_write_allowed(&self) -> bool {
        self.device_status & S_DRV == S_DRV && self.device_status & VIRTIO_CONFIG_S_FEATURES_OK == 0
    }

    // ------------------------------------------------------------------
    // Loopback: TX → RX byte echo
    // ------------------------------------------------------------------

    /// Drive the TX queue. For each TX chain, captures the L2 frame
    /// (after stripping the 12-byte virtio header), marks the chain
    /// used, then synthesizes an RX delivery for the same frame.
    ///
    /// vCPU-thread bounded work: the inner loop executes guest-memory
    /// reads + writes (no syscalls, no blocking) plus one irqfd write
    /// per delivered RX. Each TX chain processed contributes
    /// O(`frame_bytes`) memory copy. The MMIO QUEUE_NOTIFY handler
    /// invokes this function and returns; the freeze-rendezvous
    /// timeout is never at risk because there is no syscall to block
    /// SIGRTMIN delivery on.
    fn process_tx_loopback(&mut self, pair: usize) {
        // Pair `pair`'s data virtqueues interleave per the kernel
        // rxq2vq/txq2vq ordering: RX at even index `2*pair`, TX at odd index
        // `2*pair+1`. The caller (`notify_queue`) invokes this only for an
        // active pair's TX kick.
        let rx_idx = pair * 2;
        let tx_idx = pair * 2 + 1;
        // DRIVER_OK gate per virtio-v1.2 §2.1.2: the device MUST NOT
        // process virtqueue requests until the driver has finished
        // initialisation by writing DRIVER_OK. A guest writing
        // QUEUE_NOTIFY while still in the FEATURES_OK..DRIVER_OK
        // window is either buggy or hostile; either way, ignore the
        // kick. virtio_blk and virtio_console both honor this gate
        // in practice via the queue-ready check (Queue::ready
        // returns false until the address registers have been
        // written, which happens between FEATURES_OK and DRIVER_OK)
        // — but our pop_descriptor_chain path would happily drain
        // a queue whose addresses had been written but DRIVER_OK
        // not yet set, so we add the explicit status check here
        // rather than rely on queue-ready as a proxy.
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            return;
        }
        // Clone the `Arc<OnceLock>` once per kick (cheap atomic
        // refcount bump) so the subsequent `OnceLock::get` borrows
        // from this local rather than from `self.mem` — which would
        // freeze every other field for the lifetime of `mem`. The
        // helpers below need `&mut self.queues[...]` and
        // `&mut self.tx_frame_scratch`, so the disjoint-field reborrow
        // through `mem_arc` is what lets the borrow checker see
        // `self.mem` is not aliased while we work the queues. Replaces
        // the prior `self.mem.clone()` (a full
        // `GuestMemoryMmap::clone` traversing every region's inner
        // `Arc<RegionMmap>`); only one atomic bump now per kick.
        let mem_arc = Arc::clone(&self.mem);
        let Some(mem) = mem_arc.get() else {
            if !self.mem_unset_warned.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "virtio-net: queue notify before set_mem; \
                     dropping TX kick until guest memory is wired"
                );
            }
            return;
        };
        // Per-queue poison gating: NO entry-level short-circuit on
        // `queue_poisoned`. The helpers (`pop_and_capture_tx`,
        // `try_loopback_to_rx`) consult their own queue's flag at
        // their pop sites. Per-queue independence: a poisoned RX
        // must not stop the TX path from continuing to drain (the
        // guest can still get TX completions even when its RX side
        // is broken), and a poisoned TX returns Empty so the loop
        // just breaks naturally — no need for a special outer
        // gate.
        //
        // Per-queue used-ring advance tracking for the per-queue MSI-X
        // signal: `tx_advanced` ⇔ this pair's TX queue (tx_idx) advanced its
        // used ring (a TX completion — including a malformed-chain recycle whose
        // `add_used(head, 0)` succeeded, which the guest's NAPI must still see);
        // `rx_advanced` ⇔ the RX queue (rx_idx) advanced (a delivery or an empty
        // completion). Each fires its OWN queue's vector post-loop so a per-queue
        // IRQ lands on that queue's affined CPU. On INTx both collapse to the one
        // shared line (signal_used ignores the index there).
        let mut tx_advanced = false;
        let mut rx_advanced = false;
        // `tx_just_poisoned` / `rx_just_poisoned`: the false→true
        // transition observed during THIS drain. The signal +
        // counter bump are gated on the transition, not on the
        // current state of the flag — re-kicks against an already-
        // poisoned queue must NOT re-fire the signal or re-bump
        // the counter. Each helper sets its corresponding flag if
        // it just transitioned; the flags are inspected post-loop
        // to fire signal_queue_poisoned exactly once per transition,
        // AFTER any pending used-ring publishes have been kicked
        // (signal poison only after the guest can observe the
        // prior completions, so a missed signal_used would not
        // strand actionable TX completions behind the device-death
        // signal).
        let mut tx_just_poisoned = false;
        let mut rx_just_poisoned = false;

        // Borrow-split: the TX queue iterator and the RX queue side
        // both need `&mut self.queues[...]` at non-overlapping times.
        // We iterate TX chains, capture frame bytes into the per-device
        // scratch (releasing the TX borrow), walk RX queue inside
        // `try_loopback_to_rx` (taking the RX borrow), then close
        // the loop iteration with a TX `add_used`.
        loop {
            let pop_outcome = self.pop_and_capture_tx(mem, tx_idx);
            let chain_outcome = match pop_outcome {
                TxPopOutcome::Empty => break,
                TxPopOutcome::JustPoisoned => {
                    // Hostile-guest TX-side iter() error —
                    // `pop_and_capture_tx` performed the false→true
                    // transition, bumped the counter, and set
                    // `queue_poisoned[tx_idx] = true`. No chain was
                    // popped. Break the drain (TX cannot make
                    // forward progress until reset). Signal handled
                    // post-loop alongside any RX poison transition,
                    // ordered after the used-ring kick so the guest
                    // observes prior completions.
                    tx_just_poisoned = true;
                    break;
                }
                TxPopOutcome::Chain(c) => c,
            };
            let TxChainOutcome { head, frame_len } = chain_outcome;

            if let Some(len) = frame_len {
                // Frame captured into self.tx_frame_scratch[..len].
                // Run the RX half before recording any TX-completion
                // counter — the RX outcome determines what byte
                // count we use for rx_bytes (truncation vs full),
                // and the TX add_used at the end of this iteration
                // determines whether tx_packets bumps at all.
                let outcome = self.try_loopback_to_rx(mem, len, rx_idx);
                self.handle_rx_loopback_outcome(
                    outcome,
                    &mut rx_advanced,
                    &mut rx_just_poisoned,
                );
            }
            // else: chain was malformed and tx_chain_invalid was
            // already bumped inside `pop_and_capture_tx`. Neither
            // `tx_packets` nor `rx_packets` advances on this path.
            // Still mark used so the guest doesn't hang.

            self.tx_add_used(mem, head, frame_len, &mut tx_advanced, tx_idx);

            // Partial-RX-poison handling: if the RX-side `iter()`
            // just transitioned false→true this iteration (set by
            // the JustRxPoisoned arm above), break the drain.
            // The in-flight TX chain has been honestly completed
            // via add_used above (steps 1-2 of the partial-poison
            // flow); the per-queue flag was set inside
            // `try_loopback_to_rx` and the post-loop signal will
            // fire NEEDS_RESET + irqfd (steps 3-5). Subsequent
            // kicks against a still-poisoned RX take the entry
            // gate inside `try_loopback_to_rx`
            // (`RxAlreadyPoisoned`), so TX continues servicing
            // kicks across drains — per-queue independence is
            // preserved at the kick boundary, while within this
            // drain we stop after honestly completing the
            // in-flight TX chain. No need to also check
            // `tx_just_poisoned` — the TX-side `JustPoisoned`
            // outcome breaks earlier (no chain was popped).
            if rx_just_poisoned {
                break;
            }
        }

        // Post-loop ordered signal sequence:
        //   1. Per-queue used-ring signal — fire EACH advanced queue's OWN
        //      vector (rx_idx, tx_idx) so the guest's per-queue NAPI wakes on
        //      that queue's affined CPU (the point of per-queue MSI-X). Must
        //      come BEFORE the poison signal — a missed used signal would strand
        //      completions the guest can still consume (TX completions are
        //      actionable even if RX is poisoned). On INTx both collapse to the
        //      one shared line; the guest's vp_vring_interrupt drains all vqs, so
        //      the (at most two) kicks are harmless.
        //   2. signal_queue_poisoned() exactly once if either side transitioned
        //      false→true during this drain. Sets NEEDS_RESET in device_status +
        //      INT_CONFIG in interrupt_status (idempotent under bitwise-OR — one
        //      call is correct whether one or both queues just poisoned) + writes
        //      the irqfd. Spec-compliant per virtio-v1.2 (config interrupt paired
        //      with NEEDS_RESET) and matches cloud-hypervisor. Counter-mode irqfd
        //      coalesces the used + poison signals into one guest-visible IRQ
        //      when they both fire on the same vector.
        // MSI-X: fire EACH advanced queue's OWN vector (per-node steering) so a
        // per-queue IRQ lands on that queue's affined CPU. INTx: every queue
        // shares one line, so a single assert per drain is correct and
        // sufficient — the guest's vp_vring_interrupt drains ALL vqs per IRQ —
        // and coalescing avoids a redundant second eventfd write, preserving the
        // one-kick-per-drain invariant the legacy level path relies on (multiqueue
        // is gated on MSI-X in device_features, so the INTx branch only ever sees
        // the single pair; the queue index is ignored there).
        if self.msix.as_ref().is_some_and(|m| m.lock().enabled()) {
            if rx_advanced {
                self.signal_used(rx_idx);
            }
            if tx_advanced {
                self.signal_used(tx_idx);
            }
        } else if rx_advanced || tx_advanced {
            self.signal_used(rx_idx);
        }
        if tx_just_poisoned || rx_just_poisoned {
            self.signal_queue_poisoned();
        }
    }

    /// Apply the per-chain effects of one `try_loopback_to_rx`
    /// outcome: bumps the matching RX counter and, when the RX
    /// used-ring advanced, sets `had_used_ring_publish`. When the
    /// RX queue's `iter()` just transitioned false→true this drain,
    /// sets `rx_just_poisoned` so the driver loop breaks and the
    /// post-loop signal fires. Performs no queue I/O of its own —
    /// `try_loopback_to_rx` already did the descriptor walk and
    /// add_used; this is the counter/flag bookkeeping for its
    /// result. No break/continue/return: every arm only records a
    /// counter and/or sets a flag, so the loop control flow stays
    /// in `process_tx_loopback`.
    fn handle_rx_loopback_outcome(
        &mut self,
        outcome: LoopbackOutcome,
        had_used_ring_publish: &mut bool,
        rx_just_poisoned: &mut bool,
    ) {
        match outcome {
            LoopbackOutcome::Delivered { l2_bytes_written } => {
                // RX add_used Ok, used-ring advanced.
                // `l2_bytes_written` reflects actual bytes
                // the guest can read past the virtio
                // header — on a too-small RX buffer this
                // is < the source `len`, so rx_bytes never
                // overstates delivery.
                self.counters.record_rx_delivered(l2_bytes_written);
                *had_used_ring_publish = true;
            }
            LoopbackOutcome::DeliveredButAddUsedFailed => {
                // Header + frame DID land in the descriptor
                // but the trailing `add_used` failed.
                // `rx_add_used_failures` was bumped inside
                // `try_loopback_to_rx`. Do NOT bump
                // rx_packets (guest never observes the
                // publish) and do NOT mark the used-ring as
                // advanced (it didn't). Do NOT poison the
                // queue — add_used failure is a transient
                // used-ring GPA mapping issue, not a
                // structural avail.idx violation. Continue
                // the drain; TX `add_used` below still
                // completes for this chain.
            }
            LoopbackOutcome::RxAlreadyPoisoned => {
                // Already-poisoned RX queue (re-kick after
                // a prior poison, OR a prior iteration of
                // this drain already triggered the
                // false→true transition for RX). Drop the
                // captured frame and record the drop. Do NOT
                // re-bump `invalid_avail_idx_count` (the
                // poison event was already counted on its
                // false→true transition) and do NOT re-fire
                // the signal. TX add_used below still runs,
                // so `tx_packets` still bumps for this chain.
                self.counters.record_tx_dropped_rx_poisoned();
            }
            LoopbackOutcome::JustRxPoisoned => {
                // RX-side `iter()` first-time error.
                // `try_loopback_to_rx` performed the
                // false→true RX poison transition, bumped
                // `invalid_avail_idx_count`, and set
                // `queue_poisoned[rx_idx] = true`. The
                // TX-captured frame is dropped (nothing to
                // deliver into) — record the drop. TX
                // add_used below still runs so the in-flight
                // TX request doesn't hang (and `tx_packets`
                // still bumps). RX poison signal is fired
                // post-loop after the used-ring kick.
                self.counters.record_tx_dropped_rx_poisoned();
                *rx_just_poisoned = true;
            }
            LoopbackOutcome::NoRxBuffer => {
                // No chain popped — the RX queue was empty
                // or not ready. The TX-captured frame is
                // dropped on the floor.
                self.counters.record_tx_dropped_no_rx_buffer();
            }
            LoopbackOutcome::RxChainInvalid { add_used_ok } => {
                // Chain rejected during the descriptor walk.
                // Exactly one of `rx_chain_invalid`
                // (chain-shape: read-only descriptor or
                // address overflow on the descriptor's GPA)
                // or `rx_write_failed` (chain shape OK but
                // a guest-memory `write_slice` hit an
                // unmapped GPA mid-walk) was bumped inside
                // `try_loopback_to_rx`; the two are
                // mutually exclusive per chain. Whether the
                // used-ring advanced depends on whether the
                // recycle-add_used succeeded; if it did,
                // the guest's NAPI must wake to see the
                // empty completion (otherwise the buffer
                // sits unrecycled until a virtio reset).
                // Recycle-add_used failure is NOT a poison
                // event — that's a transient used-ring GPA
                // issue, not a structural avail.idx
                // violation. `rx_add_used_failures` was
                // bumped inside the helper for visibility.
                if add_used_ok {
                    *had_used_ring_publish = true;
                }
            }
        }
    }

    /// Mark one drained TX chain used and, on success, set
    /// `had_used_ring_publish` plus record the completion. Closes
    /// the per-iteration TX half after the RX delivery: TX
    /// descriptors are device-readable so used_len is 0, and
    /// tx_packets bumps only when the publish succeeds.
    fn tx_add_used(
        &mut self,
        mem: &GuestMemoryMmap,
        head: u16,
        frame_len: Option<usize>,
        had_used_ring_publish: &mut bool,
        tx_idx: usize,
    ) {
        // Mark the TX chain used. TX descriptors are
        // device-readable, so used_len is 0 — the device wrote
        // nothing back to guest memory on the TX side, and
        // virtio-v1.2 §2.7.8.2 counts only device-WRITABLE bytes in
        // used.len. (Reference divergence: cloud-hypervisor passes
        // the bytes written to its tap as the TX used.len — a
        // bytes-read value the spec does not sanction; the guest's
        // virtnet driver ignores TX used.len so both work in
        // practice, but 0 is the spec-correct value for a wholly
        // device-readable chain.) tx_packets is bumped ONLY on TX
        // add_used success — calling `record_tx_completed` before
        // this point would let the counter lie if the publish fails
        // (the guest never sees the completion). Failed TX add_used
        // bumps `tx_add_used_failures` instead, keeping the
        // per-event counter taxonomy 1:1 with observable events.
        let q = &mut self.queues[tx_idx];
        match q.add_used(mem, head, 0) {
            Ok(()) => {
                if let Some(len) = frame_len {
                    self.counters.record_tx_completed(len as u64);
                }
                *had_used_ring_publish = true;
            }
            Err(e) => {
                // Bump tx_add_used_failures for operator
                // visibility. Do NOT poison the queue: this is
                // a transient used-ring GPA mapping problem,
                // not a structural avail.idx violation. The
                // next QUEUE_NOTIFY may succeed if the guest
                // re-binds. Same rationale as the RX-side
                // add_used handling in `try_loopback_to_rx` —
                // poison is reserved for `iter()` errors
                // (cloud-hypervisor convergence). virtio-blk
                // follows the same rule: add_used failures
                // bump io_errors but never set NEEDS_RESET.
                self.counters.record_tx_add_used_failure();
                tracing::warn!(
                    head,
                    %e,
                    "virtio-net TX add_used failed (used-ring address \
                     likely unmapped); bumped tx_add_used_failures, \
                     will NOT bump tx_packets"
                );
            }
        }
    }

    /// Pop one TX chain, capture the L2 frame bytes (after the
    /// 12-byte virtio header) into `self.tx_frame_scratch`, and
    /// return the chain head index plus the captured frame length.
    ///
    /// Returns `Empty` when the TX queue is empty OR when the
    /// per-queue `queue_poisoned[tx_idx]` flag is already set (the
    /// entry gate short-circuits with `Empty` rather than a
    /// dedicated "AlreadyPoisoned" variant — the drain loop's
    /// only legal action is to break, and `Empty` already conveys
    /// that). Returns `JustPoisoned` when the TX `iter()`
    /// observed any structural error for the FIRST time —
    /// `invalid_avail_idx_count` is bumped and
    /// `queue_poisoned[tx_idx]` is set; the caller breaks the drain
    /// and the post-loop signal handler fires.
    /// Returns `Chain(TxChainOutcome { frame_len: None })` when the
    /// chain is malformed — the caller must still `add_used` the
    /// head so the guest doesn't hang. Returns
    /// `Chain(TxChainOutcome { frame_len: Some(n) })` on success;
    /// `self.tx_frame_scratch[..n]` holds the captured bytes.
    ///
    /// Uses `iter()`/`.next()` directly so we OBSERVE
    /// `Error::InvalidAvailRingIndex` instead of swallowing it —
    /// the default `pop_descriptor_chain` impl in
    /// `virtio_queue::QueueT` (queue.rs:573-587) logs the error
    /// and returns `None`, which masks the structural violation as
    /// "no chain available" and lets every subsequent kick re-trip
    /// the same error. Mirror of the virtio-blk drain pattern.
    fn pop_and_capture_tx(&mut self, mem: &GuestMemoryMmap, tx_idx: usize) -> TxPopOutcome {
        // Two phases: pop one TX chain (entry poison-gate +
        // iter()-pull), then walk its descriptors skipping the
        // 12-byte header and capturing the frame bytes into
        // `tx_frame_scratch`. Split out so each phase stays under the
        // function-size guard; the per-phase bodies are unchanged.
        let (chain, head) = match self.pop_tx_chain(mem, tx_idx) {
            Ok(v) => v,
            Err(outcome) => return outcome,
        };
        self.capture_tx_frame(chain, head, mem)
    }

    /// Pop one chain off the TX queue.
    ///
    /// Applies the per-queue poison entry gate (TX side) and pulls a
    /// single chain via the two-step `iter()`-then-drop pattern. On
    /// success returns the chain and its head index; every early exit
    /// becomes `Err(TxPopOutcome::X)` for the caller to return verbatim.
    fn pop_tx_chain<'a>(
        &mut self,
        mem: &'a GuestMemoryMmap,
        tx_idx: usize,
    ) -> Result<(DescriptorChain<&'a GuestMemoryMmap>, u16), TxPopOutcome> {
        // Per-queue poison gate. If the TX queue's flag is already
        // set, return Empty so the drain loop breaks naturally —
        // no iter() call (avoids re-tripping the same error and
        // re-bumping the per-event counter), no signal (the
        // false→true transition fired on the original poison and
        // the bits/counter remain set), no add_used. The transition
        // gate ensures counter and signal happen only on the
        // false→true crossing, not on every kick. Re-kicks are
        // benign no-ops.
        if self.queue_poisoned[tx_idx] {
            return Err(TxPopOutcome::Empty);
        }
        // Step 1: pull one chain out of the queue. The chain holds
        // its own `mem.clone()` (queue.rs:761-766) so it does NOT
        // borrow from the iter or the queue — we collect it into a
        // tight scope, drop the queue borrow before touching any
        // other field of `self`, then walk the chain afterwards
        // (which needs `&mut self.tx_frame_scratch` and
        // `&self.counters`).
        //
        // Two-step extraction so the queue borrow is strictly
        // scoped to one statement: (a) call `iter().next()` and
        // collect either the chain, an empty marker, or any
        // iter()-error marker; (b) drop the queue borrow; (c)
        // re-borrow `self` to bump counters / set the poison flag.
        //
        // Any iter() error → poison. cloud-hypervisor's pattern
        // for hostile-guest defense: `InvalidAvailRingIndex` is
        // the most specific structural violation the
        // virtio-queue crate currently reports, but the broader
        // `QueueT::iter` contract returns `Err` only for queue
        // state the driver has corrupted (avail-ring read
        // overflow, etc.). All such errors are non-recoverable
        // without a virtio reset; treating them uniformly as
        // poison keeps the failure-classification taxonomy
        // simple and converges with the upstream pattern. Future
        // virtio-queue versions may add new Error variants — they
        // funnel through this arm without code change.
        enum IterStep<C> {
            Chain(C),
            Empty,
            Poisoned(VirtioQueueError),
        }
        let step: IterStep<_> = {
            let q = &mut self.queues[tx_idx];
            match q.iter(mem) {
                Ok(mut iter) => match iter.next() {
                    Some(c) => IterStep::Chain(c),
                    None => IterStep::Empty,
                },
                Err(e) => IterStep::Poisoned(e),
            }
        };
        match step {
            IterStep::Empty => Err(TxPopOutcome::Empty),
            IterStep::Chain(c) => {
                let h = c.head_index();
                Ok((c, h))
            }
            IterStep::Poisoned(err) => {
                // Hostile- or buggy-guest poison — first time. The
                // avail-ring iterator failed with a structural
                // error — most commonly `InvalidAvailRingIndex`
                // (virtio-v1.2 §2.7.13.3 violation: avail.idx more
                // than `queue.size` ahead of next_avail; check
                // sits at queue.rs:707-709 in `AvailIter::new`),
                // but any `iter()` Err is treated identically.
                // Perform the false→true transition: mark the
                // queue dead so future drains short-circuit at
                // the entry gate above, bump the per-event
                // counter, log the error. Return JustPoisoned so
                // the caller breaks the drain and the post-loop
                // signal handler fires `signal_queue_poisoned`
                // exactly once.
                self.queue_poisoned[tx_idx] = true;
                self.counters.record_invalid_avail_idx();
                tracing::warn!(
                    err = %err,
                    "virtio-net TX iter() failed; poisoning TX queue until \
                     guest reset (any structural queue error is \
                     non-recoverable; cloud-hypervisor convergence)"
                );
                Err(TxPopOutcome::JustPoisoned)
            }
        }
    }

    /// Walk the popped TX `chain`, skipping the 12-byte virtio header
    /// and capturing the post-header frame bytes into
    /// `self.tx_frame_scratch`, then classify the outcome.
    ///
    /// Returns `Chain(TxChainOutcome { frame_len: None })` when the
    /// chain is malformed or over-size — the caller must still
    /// `add_used` the head so the guest doesn't hang. Returns
    /// `Chain(TxChainOutcome { frame_len: Some(n) })` on success;
    /// `self.tx_frame_scratch[..n]` holds the captured bytes.
    fn capture_tx_frame(
        &mut self,
        chain: DescriptorChain<&GuestMemoryMmap>,
        head: u16,
        mem: &GuestMemoryMmap,
    ) -> TxPopOutcome {
        // Reset scratch; capacity stays. `clear` is O(1) — it just
        // zeroes the len.
        self.tx_frame_scratch.clear();

        // Track how many of the 12 virtio-net header bytes we've
        // already absorbed across the chain's leading descriptors.
        // The kernel TX path may emit the header in its own
        // descriptor (any_header_sg = true on VERSION_1, but the
        // pushed-into-skb-data path also uses a single combined
        // descriptor when headroom is sufficient). Either layout is
        // legal per virtio-v1.2 §5.1.6.5; the device must skip the
        // first 12 bytes of the chain regardless of how they're
        // distributed.
        let mut hdr_remaining: usize = VIRTIO_NET_HDR_LEN;
        let mut total_data_bytes: usize = 0;
        let mut chain_invalid = false;
        let mut chain_oversize = false;

        for desc in chain {
            if desc.is_write_only() {
                // TX descriptors must be device-readable. A
                // write-only descriptor in a TX chain is a guest
                // protocol violation. Stop reading; the chain is
                // dropped.
                chain_invalid = true;
                break;
            }
            // The TRUE descriptor length — NOT pre-capped. The
            // over-cap check below uses this exact value to detect a
            // frame exceeding MAX_FRAME_SIZE and DROP it (vs silently
            // capping it). A `u32` always widens into `usize` without
            // loss; the size cap is enforced before any allocation.
            let mut desc_len = desc.len() as usize;
            let mut desc_addr = desc.addr();

            // Skip / consume any remaining header bytes from this
            // descriptor first. `checked_add` here is defense in depth
            // against an attacker-controlled `desc.addr() = u64::MAX`:
            // an in-bounds descriptor read would have already failed
            // at `read_slice` below, but a hostile guest could place
            // the header AT a sub-page address near `u64::MAX` whose
            // `+skip` arithmetic wraps. Drop the chain on overflow
            // instead of panicking the vCPU thread (a panic on the
            // vCPU would propagate via `vcpu_panic::install_once` and
            // tear down the VM mid-test).
            if hdr_remaining > 0 {
                let skip = hdr_remaining.min(desc_len);
                let Some(new_addr) = desc_addr.checked_add(skip as u64) else {
                    chain_invalid = true;
                    break;
                };
                hdr_remaining -= skip;
                desc_len -= skip;
                desc_addr = new_addr;
            }

            if desc_len == 0 {
                continue;
            }

            // Enforce MAX_FRAME_SIZE on the post-header data BEFORE any
            // allocation or guest-memory read. A frame whose data
            // exceeds the cap is DROPPED (not truncated) — silently
            // capping would corrupt a real frame the guest emitted.
            // `desc_len` is the TRUE descriptor length, so this single
            // check catches BOTH a single over-size descriptor and a
            // multi-descriptor chain whose lengths sum past the cap.
            // No overflow: `total_data_bytes <= MAX_FRAME_SIZE` from
            // prior iterations and `desc_len <= u32::MAX as usize`, so
            // the sum stays far below `usize::MAX`. Breaking here
            // (before the resize) is the hostile-guest defense: a
            // multi-GiB descriptor never forces a scratch allocation.
            if total_data_bytes + desc_len > MAX_FRAME_SIZE {
                chain_oversize = true;
                break;
            }

            let start = self.tx_frame_scratch.len();
            self.tx_frame_scratch.resize(start + desc_len, 0);
            if mem
                .read_slice(
                    &mut self.tx_frame_scratch[start..start + desc_len],
                    desc_addr,
                )
                .is_err()
            {
                // Guest-memory read failed (unmapped GPA). Drop the
                // chain; the rest of the descriptors are likely also
                // unmapped.
                self.tx_frame_scratch.truncate(start);
                chain_invalid = true;
                break;
            }
            total_data_bytes += desc_len;
        }

        if chain_oversize {
            // The post-header data exceeds MAX_FRAME_SIZE — larger than
            // any standard-conformant frame the guest can emit (max_mtu
            // plus up to 802.1ad QinQ double-tagging; see
            // MAX_FRAME_SIZE). Drop it rather than truncate; the caller
            // still marks the chain used so the guest doesn't hang on
            // the slot. Distinct from `tx_chain_invalid` (malformed
            // shape) so an operator can tell "guest sent an over-size
            // frame" from "guest sent a malformed chain".
            // `chain_oversize` is only ever set after the header was
            // fully consumed, so it is mutually exclusive with the
            // `hdr_remaining != 0` (short-header) path below.
            //
            // Reference-VMM divergence (over-size TX handling): qemu
            // drops SILENTLY with no stat (rejects frames over its
            // max-buffer bound, then pushes the chain used with len 0);
            // firecracker drops, bumps a malformed-frames stat, and
            // marks used (its bound measures the whole chain including
            // the 12-byte header, vs our post-header L2 cap, so the two
            // numbers are intentionally different); libkrun TRUNCATES
            // the frame and marks success
            // (a silent-corruption bug — the prior behavior here);
            // cloud-hypervisor never copies into a bounded buffer
            // (zero-copy writev to its backend) so it has no cap. We
            // drop, bump the dedicated `tx_oversize_dropped` (distinct
            // from the malformed-chain counter), and mark used —
            // firecracker's shape with finer counter taxonomy, and
            // deliberately NOT libkrun's truncate.
            self.counters.record_tx_oversize_dropped();
            return TxPopOutcome::Chain(TxChainOutcome {
                head,
                frame_len: None,
            });
        }

        if chain_invalid || hdr_remaining != 0 {
            // hdr_remaining > 0 means the chain was shorter than 12
            // bytes total — the guest didn't even include the full
            // virtio header. That's a protocol violation per
            // virtio-v1.2 §5.1.6.5 ("A driver MUST set num_buffers
            // to 0" — implies the header is present in full).
            self.counters.record_tx_chain_invalid();
            return TxPopOutcome::Chain(TxChainOutcome {
                head,
                frame_len: None,
            });
        }

        TxPopOutcome::Chain(TxChainOutcome {
            head,
            frame_len: Some(total_data_bytes),
        })
    }

    /// Deliver `self.tx_frame_scratch[..frame_len]` into one RX chain
    /// with a 12-byte virtio header (num_buffers=1, all other fields
    /// zero) prepended.
    ///
    /// Uses `iter()`/`.next()` directly on the RX queue so we OBSERVE
    /// `Error::InvalidAvailRingIndex` instead of swallowing it (the
    /// default `pop_descriptor_chain` impl in
    /// `virtio_queue::QueueT` queue.rs:573-587 logs and returns
    /// `None`). Mirror of the TX-side `pop_and_capture_tx` and
    /// virtio-blk drain pattern.
    ///
    /// Returns one of [`LoopbackOutcome`]'s variants — see the
    /// enum doc for the per-variant routing rules.
    fn try_loopback_to_rx(
        &mut self,
        mem: &GuestMemoryMmap,
        frame_len: usize,
        rx_idx: usize,
    ) -> LoopbackOutcome {
        // Three phases: pop one RX chain (entry poison-gate +
        // iter()-pull), walk its descriptors writing the header +
        // frame bytes, then finalize (counter routing, header
        // rollback on failure, add_used). Split out so each phase
        // stays under the function-size guard; the per-phase bodies
        // are unchanged.
        let (chain, head) = match self.pop_rx_chain(mem, rx_idx) {
            Ok(v) => v,
            Err(outcome) => return outcome,
        };
        let walk = self.write_rx_chain(chain, mem, frame_len);
        self.finalize_rx(walk, head, mem, frame_len, rx_idx)
    }

    /// Pop one chain off the RX queue.
    ///
    /// Applies the per-queue poison entry gate (RX side) and pulls a
    /// single chain via the same two-step `iter()`-then-drop pattern
    /// as `pop_and_capture_tx`. On success returns the chain and its
    /// head index; every early exit becomes `Err(LoopbackOutcome::X)`
    /// for the caller to return verbatim.
    fn pop_rx_chain<'a>(
        &mut self,
        mem: &'a GuestMemoryMmap,
        rx_idx: usize,
    ) -> Result<(DescriptorChain<&'a GuestMemoryMmap>, u16), LoopbackOutcome> {
        // Per-queue poison gate (RX side). If the RX queue's flag
        // is already set, return `RxAlreadyPoisoned` without
        // touching the queue — no iter(), no add_used, no counter
        // bump, no signal. Mirror of `pop_and_capture_tx`'s entry
        // gate. RX poison must not stop TX from continuing to
        // drain — the caller still does TX add_used in this
        // iteration even when RX is poisoned.
        if self.queue_poisoned[rx_idx] {
            return Err(LoopbackOutcome::RxAlreadyPoisoned);
        }
        // Pull one chain out of the RX queue. Same two-step
        // iter()-then-drop pattern as `pop_and_capture_tx`. Any
        // iter() error → poison (cloud-hypervisor convergence;
        // see the rationale on the TX-side variant).
        enum IterStep<C> {
            Chain(C),
            NoBuffer,
            Poisoned(VirtioQueueError),
        }
        let step: IterStep<_> = {
            let q = &mut self.queues[rx_idx];
            if !q.ready() {
                // Driver hasn't published RX buffers yet (init not
                // complete). Drop the frame; future TX after RX is
                // set up will succeed.
                return Err(LoopbackOutcome::NoRxBuffer);
            }
            match q.iter(mem) {
                Ok(mut iter) => match iter.next() {
                    Some(c) => IterStep::Chain(c),
                    None => IterStep::NoBuffer,
                },
                Err(e) => IterStep::Poisoned(e),
            }
        };
        match step {
            IterStep::NoBuffer => Err(LoopbackOutcome::NoRxBuffer),
            IterStep::Chain(c) => {
                let h = c.head_index();
                Ok((c, h))
            }
            IterStep::Poisoned(err) => {
                // Hostile- or buggy-guest poison on the RX queue —
                // first time. Mirror the TX-side handling: perform
                // the false→true transition (set
                // `queue_poisoned[rx_idx]`, bump the per-event counter,
                // log), return `JustRxPoisoned`. Re-kicks
                // against the now-poisoned queue take the entry
                // gate above (returns `RxAlreadyPoisoned`) so the
                // counter and signal are event-once.
                self.queue_poisoned[rx_idx] = true;
                self.counters.record_invalid_avail_idx();
                tracing::warn!(
                    err = %err,
                    "virtio-net RX iter() failed; poisoning RX queue until \
                     guest reset (any structural queue error is \
                     non-recoverable; cloud-hypervisor convergence)"
                );
                Err(LoopbackOutcome::JustRxPoisoned)
            }
        }
    }

    /// Walk the popped RX `chain`, writing the 12-byte virtio header
    /// followed by `self.tx_frame_scratch[..frame_len]` into the
    /// device-writable descriptors. Returns an [`RxWriteResult`]
    /// carrying the bytes written, the unwritten header remainder,
    /// the recorded header-write slots (for rollback), and the
    /// failure classification (`None` on success). Does not touch
    /// counters or the used ring — that is [`Self::finalize_rx`]'s job.
    fn write_rx_chain(
        &self,
        chain: DescriptorChain<&GuestMemoryMmap>,
        mem: &GuestMemoryMmap,
        frame_len: usize,
    ) -> RxWriteResult {
        // Walk RX descriptors. Must be device-writable. Place the
        // 12-byte zero header first, then the captured frame bytes.
        // We do not split the header across descriptors — every
        // reference VMM (libkrun, firecracker, cloud-hypervisor,
        // qemu) and the kernel driver assume the header lives in a
        // single descriptor large enough to hold it. The guest
        // posts RX buffers each at least PAGE_SIZE in practice so
        // the assumption holds; on the rare case of an under-12
        // first descriptor we still try to write whatever fits and
        // walk forward — the resulting chain advertises `used_len =
        // hdr+frame` whether the bytes were split or contiguous.
        let mut bytes_written: u32 = 0;
        let mut hdr_remaining: usize = VIRTIO_NET_HDR_LEN;
        let mut frame_pos: usize = 0;
        // Track every (GPA, len) the header bytes landed at while
        // walking descriptors. On `WriteFailed` (a frame-bytes
        // `write_slice` returned Err after the header had already
        // been placed) we zero these bytes before `add_used(head, 0)`
        // so the guest cannot observe a stale `num_buffers=1` header
        // claiming a frame is present when in fact the recycle path
        // recorded zero used bytes. The cap is `VIRTIO_NET_HDR_LEN`
        // because the worst-case split is one header byte per
        // descriptor (12 entries). `count` is the number of valid
        // entries in `slots`. cloud-hypervisor avoids this entirely
        // by deferring `num_buffers` to a single post-readv write
        // (`net_util/src/queue_pair.rs::process_desc_chain`); we
        // copy bytes inline so we must roll back instead.
        let mut hdr_write_slots: [(GuestAddress, usize); VIRTIO_NET_HDR_LEN] =
            [(GuestAddress(0), 0); VIRTIO_NET_HDR_LEN];
        let mut hdr_write_count: usize = 0;
        let mut chain_invalid: Option<InvalidReason> = None;

        for desc in chain {
            if !desc.is_write_only() {
                // RX descriptors must be device-writable. A
                // read-only descriptor in an RX chain is a guest
                // protocol violation.
                chain_invalid = Some(InvalidReason::Shape);
                break;
            }
            let mut desc_addr = desc.addr();
            let mut desc_len = desc.len() as usize;

            // First, drain any remaining header bytes into this
            // descriptor. The `mrg_rxbuf` header layout (12 bytes
            // matching `struct virtio_net_hdr_v1`): bytes 0..10 are
            // GSO/csum fields the device leaves at zero (no
            // negotiated offload features → `flags=0`,
            // `gso_type=GSO_NONE=0`, csum/hdr_len fields irrelevant);
            // bytes 10..12 are `num_buffers` LE u16 = 1, signalling
            // the kernel's `virtnet_receive_mergeable` /
            // `virtnet_receive_done` "single-buffer frame" path. A
            // zero `num_buffers` would make
            // `drivers/net/virtio_net.c::receive_mergeable` treat the
            // frame as the head of a multi-buffer chain and either
            // wait forever for the next buffer or panic on the
            // shouldn't-happen branch. Pinned at 1 because we never
            // negotiate `VIRTIO_NET_F_MRG_RXBUF`.
            //
            // `checked_add` is defense in depth against an attacker-
            // controlled `desc.addr()` near `u64::MAX`. Drop the
            // chain on overflow instead of panicking the vCPU
            // (a panic propagates via `vcpu_panic::install_once`).
            if hdr_remaining > 0 {
                let take = hdr_remaining.min(desc_len);
                const RX_HDR: [u8; VIRTIO_NET_HDR_LEN] = {
                    let mut h = [0u8; VIRTIO_NET_HDR_LEN];
                    // num_buffers = 1 (LE u16 at offset 10)
                    h[10] = 1;
                    h[11] = 0;
                    h
                };
                let hdr_start = VIRTIO_NET_HDR_LEN - hdr_remaining;
                let hdr_slice = &RX_HDR[hdr_start..hdr_start + take];
                if mem.write_slice(hdr_slice, desc_addr).is_err() {
                    // GPA write failure — chain shape was
                    // acceptable, the descriptor's address just
                    // points at unmapped memory.
                    chain_invalid = Some(InvalidReason::WriteFailed);
                    break;
                }
                // Record the (GPA, len) where the header just
                // landed. The post-walk WriteFailed branch zeros
                // these bytes before `add_used(head, 0)` so the
                // guest never observes a stale `num_buffers=1`
                // header for a chain we're recycling with len=0.
                // `take <= hdr_remaining <= VIRTIO_NET_HDR_LEN` and
                // each iteration consumes >= 1 byte of header, so
                // `hdr_write_count` never exceeds the slot array.
                hdr_write_slots[hdr_write_count] = (desc_addr, take);
                hdr_write_count += 1;
                let Some(new_addr) = desc_addr.checked_add(take as u64) else {
                    // Descriptor's `addr + take` overflows u64 —
                    // an attacker-controlled malformed address.
                    // Routed to chain-shape rejection: the
                    // descriptor itself is malformed, distinct from
                    // a write to an unmapped (but well-formed) GPA.
                    chain_invalid = Some(InvalidReason::Shape);
                    break;
                };
                bytes_written = bytes_written
                    .checked_add(take as u32)
                    .expect("bytes_written cannot overflow u32 — capped by MAX_FRAME_SIZE+12");
                hdr_remaining -= take;
                desc_len -= take;
                desc_addr = new_addr;
            }

            if desc_len == 0 || frame_pos == frame_len {
                continue;
            }

            // Then frame bytes.
            let take = desc_len.min(frame_len - frame_pos);
            if mem
                .write_slice(
                    &self.tx_frame_scratch[frame_pos..frame_pos + take],
                    desc_addr,
                )
                .is_err()
            {
                // GPA write failure on the frame-data path. Same
                // classification as the header `write_slice`
                // failure above — chain shape was fine, the
                // descriptor's GPA is unmapped.
                chain_invalid = Some(InvalidReason::WriteFailed);
                break;
            }
            bytes_written = bytes_written
                .checked_add(take as u32)
                .expect("bytes_written cannot overflow u32 — capped by MAX_FRAME_SIZE+12");
            frame_pos += take;

            if frame_pos == frame_len && hdr_remaining == 0 {
                break;
            }
        }

        RxWriteResult {
            bytes_written,
            hdr_remaining,
            frame_pos,
            chain_invalid,
            hdr_write_slots,
            hdr_write_count,
        }
    }

    /// Finalize the RX delivery for chain `head` given the
    /// descriptor-`walk` output: route the failure counters and roll
    /// back the placed header bytes on a malformed/unmapped chain,
    /// emit the too-small-buffer truncation warning, and call
    /// `add_used`. Returns the [`LoopbackOutcome`] for the caller.
    fn finalize_rx(
        &mut self,
        walk: RxWriteResult,
        head: u16,
        mem: &GuestMemoryMmap,
        frame_len: usize,
        rx_idx: usize,
    ) -> LoopbackOutcome {
        if let Some(reason) = walk.chain_invalid {
            // Malformed RX chain: the frame is dropped, the chain
            // is marked used with `len=0` so the guest can recycle
            // its descriptor (without `add_used` the kernel's
            // virtio core would never recover the buffer until a
            // virtio reset). The counter routing distinguishes
            // shape rejection (`rx_chain_invalid`) from GPA
            // write-failure (`rx_write_failed`); both still
            // signal the caller NOT to also bump
            // `tx_dropped_no_rx_buffer` — those events are
            // mutually exclusive (chain present but malformed
            // vs queue empty), and the failure-classification
            // taxonomy MUST stay 1:1 with chains. Per chain, at
            // most one of `rx_chain_invalid` / `rx_write_failed`
            // is bumped — never both — because we set
            // `chain_invalid` exactly once and break out of the
            // descriptor walk on the first failure observed.
            match reason {
                InvalidReason::Shape => self.counters.record_rx_chain_invalid(),
                InvalidReason::WriteFailed => self.counters.record_rx_write_failed(),
            }
            // Roll back the header bytes we already placed in guest
            // memory. The pre-1.2-baked header carries
            // `num_buffers=1` (LE u16 at offset 10-11); leaving
            // those bytes intact while we hand the chain back with
            // `add_used(head, 0)` would let the guest observe a
            // header that claims a frame is present in a chain
            // we're recycling as empty. The non-mergeable RX path
            // (`drivers/net/virtio_net.c::receive_small`) ignores
            // `num_buffers` for `len=0` short-packet drops, but
            // the kernel's page pool can re-arm the same backing
            // page for a future receive without zeroing it; in
            // mergeable-rxbuf builds (which we don't currently
            // negotiate) the same stale byte would steer
            // `receive_mergeable`'s `--num_buf` loop. Zero
            // unconditionally — a write_slice that fails here
            // means we just leave whatever bytes were already in
            // place; we have no better recovery and the counter
            // (`rx_write_failed` / `rx_chain_invalid`) already
            // covered the original failure. Ignoring the rollback
            // result mirrors `let _` over the already-counted
            // failure path. Both `Shape` (addr-overflow can fire
            // after a successful header write) and `WriteFailed`
            // need this rollback; only the read-only-descriptor
            // form of `Shape` enters with `hdr_write_count == 0`,
            // in which case the loop is a no-op.
            const ZEROS: [u8; VIRTIO_NET_HDR_LEN] = [0u8; VIRTIO_NET_HDR_LEN];
            for &(addr, len) in &walk.hdr_write_slots[..walk.hdr_write_count] {
                let _ = mem.write_slice(&ZEROS[..len], addr);
            }
            // If `add_used` itself fails after a chain-direction
            // violation, the guest's used-ring is broken at the
            // same address the malformed chain came from. Record
            // the queue-state failure separately from
            // `rx_chain_invalid` so operators can distinguish "RX
            // chain shape was bad" (which we already counted) from
            // "RX queue is structurally broken" (this site). Both
            // counters can fire on the same chain because the
            // failure modes describe different problems.
            //
            // `add_used_ok` is propagated to the caller so it can
            // decide whether to kick: if `add_used` succeeded the
            // used-ring advanced and the guest's NAPI must wake to
            // observe the empty completion and recycle the buffer.
            //
            // Do NOT poison on add_used failure — that's a
            // transient used-ring GPA issue, not a structural
            // avail.idx violation. Same rule as the success branch
            // (post-walk add_used path below) and virtio-blk:
            // poison is reserved for `iter()` errors only. See the
            // doc on the success-branch add_used match for the
            // full rationale.
            let add_used_ok = match self.queues[rx_idx].add_used(mem, head, 0) {
                Ok(()) => true,
                Err(e) => {
                    self.counters.record_rx_add_used_failure();
                    tracing::warn!(
                        head,
                        %e,
                        "virtio-net RX add_used failed after malformed-chain \
                         reject (used-ring address likely unmapped); bumped \
                         rx_add_used_failures"
                    );
                    false
                }
            };
            return LoopbackOutcome::RxChainInvalid { add_used_ok };
        }

        if walk.frame_pos < frame_len || walk.hdr_remaining != 0 {
            // RX descriptor chain was too small to hold the full
            // header + frame. virtio-v1.2 §5.1.6.4: the driver
            // SHOULD always provide an RX buffer of at least
            // `vi->hdr_len + 1500` (default MTU) bytes; a chain
            // smaller than that is the guest's fault. Drop the
            // remainder of the frame; the `bytes_written` we
            // already issued is what `add_used` records.
            //
            // Without VIRTIO_NET_F_MRG_RXBUF, frame fragmentation
            // across multiple posted buffers is NOT permitted —
            // each frame must fit in one popped chain. We intentionally
            // do not pop a second RX chain for the spillover.
            tracing::debug!(
                frame_len,
                bytes_written = walk.bytes_written,
                hdr_remaining = walk.hdr_remaining,
                "virtio-net RX buffer too small for full frame; truncating"
            );
        }

        // Compute actual L2 bytes delivered (i.e. the bytes the
        // guest can actually read past the virtio header). On a
        // too-small RX buffer this is `bytes_written - hdr_taken`
        // where `hdr_taken = VIRTIO_NET_HDR_LEN - hdr_remaining`;
        // when the buffer truncated mid-header even the header is
        // partial, in which case the L2 byte count is zero.
        // `saturating_sub` covers both cases without an explicit
        // branch.
        let hdr_taken = (VIRTIO_NET_HDR_LEN - walk.hdr_remaining) as u32;
        let l2_bytes = walk.bytes_written.saturating_sub(hdr_taken) as u64;

        // The guest cannot recover from an `add_used` failure
        // without a virtio reset. Bump `rx_add_used_failures`
        // (queue-state breakage) and route to a distinct outcome
        // so the caller does NOT bump `rx_packets` — the guest
        // never observes the publish. A counter that lies during
        // queue-state breakage would mislead operators into
        // thinking delivery worked.
        //
        // Do NOT poison the queue on `add_used` failure. Unlike
        // an avail-ring iterator error (which means the guest's
        // avail.idx is structurally inconsistent — a virtio-spec
        // violation that cannot be recovered without reset), an
        // add_used failure is a transient used-ring GPA mapping
        // problem. The next QUEUE_NOTIFY may find the GPA mapped
        // (e.g. if the guest re-binds the used ring). Counting
        // the failure via `rx_add_used_failures` gives the
        // operator visibility without permanently halting the
        // RX side. virtio-blk follows the same convention:
        // add_used failures bump io_errors but do NOT set
        // NEEDS_RESET. Poison is reserved for `iter()` errors
        // (cloud-hypervisor convergence: structural avail.idx
        // violations only).
        match self.queues[rx_idx].add_used(mem, head, walk.bytes_written) {
            Ok(()) => LoopbackOutcome::Delivered {
                l2_bytes_written: l2_bytes,
            },
            Err(e) => {
                self.counters.record_rx_add_used_failure();
                tracing::warn!(
                    head,
                    %e,
                    "virtio-net RX add_used failed after successful frame \
                     write (used-ring address likely unmapped); bumped \
                     rx_add_used_failures, will NOT bump rx_packets"
                );
                LoopbackOutcome::DeliveredButAddUsedFailed
            }
        }
    }
}

/// Classification of why an RX descriptor walk failed, recorded by
/// `write_rx_chain` and consumed by `finalize_rx`.
///
/// Distinguishes chain-shape rejection (read-only descriptor,
/// address overflow on the descriptor's GPA) from guest-memory
/// `write_slice` failure (chain shape was fine but a descriptor's
/// GPA is unmapped). The two failure modes route to distinct
/// counters (`rx_chain_invalid` vs `rx_write_failed`) so operators
/// reading the failure dump can separate "guest violated the RX
/// descriptor-direction rule" from "guest posted a buffer at an
/// unmapped GPA". `None` in [`RxWriteResult::chain_invalid`] means
/// the walk succeeded; `finalize_rx` consults this and bumps exactly
/// one counter (or none, on success).
enum InvalidReason {
    Shape,
    WriteFailed,
}

/// Output of the RX descriptor walk (`write_rx_chain`), consumed by
/// `finalize_rx`.
///
/// Carries exactly the state the finalize phase reads:
///   - `bytes_written`: header + frame bytes successfully placed,
///     the value `add_used` records on the success path.
///   - `hdr_remaining`: unwritten header bytes (0 once the full
///     12-byte header landed); a non-zero value means the chain
///     truncated mid-header.
///   - `frame_pos`: frame bytes consumed from `tx_frame_scratch`;
///     `frame_pos < frame_len` flags a too-small RX buffer.
///   - `chain_invalid`: `Some(reason)` when the walk hit a
///     malformed/unmapped descriptor, `None` on success.
///   - `hdr_write_slots` / `hdr_write_count`: the `(GPA, len)`
///     locations the header bytes landed at, rolled back to zero on
///     failure before `add_used(head, 0)`.
struct RxWriteResult {
    bytes_written: u32,
    hdr_remaining: usize,
    frame_pos: usize,
    chain_invalid: Option<InvalidReason>,
    hdr_write_slots: [(GuestAddress, usize); VIRTIO_NET_HDR_LEN],
    hdr_write_count: usize,
}

/// Outcome classification for `try_loopback_to_rx`. Each variant
/// describes both the data-side outcome and whether the RX
/// used-ring advanced — the latter governs whether the irqfd
/// kick is needed.
///
/// Variants:
///   - `Delivered { l2_bytes_written }`: header + frame written,
///     `add_used` returned Ok, used-ring advanced. Caller bumps
///     `rx_packets` / `rx_bytes` and kicks the guest.
///   - `DeliveredButAddUsedFailed`: header + frame landed in
///     the descriptor but the trailing `add_used` failed —
///     queue-state breakage. `rx_add_used_failures` was bumped.
///     The queue is NOT poisoned (add_used failure is a
///     transient used-ring GPA problem, not a structural
///     avail.idx violation; the next kick may find the GPA
///     mapped). Caller does NOT bump `rx_packets` (guest never
///     observes the publish) and does NOT mark the used-ring
///     advanced. TX add_used for this chain still runs.
///   - `NoRxBuffer`: RX queue not ready or empty, no chain
///     popped. Caller bumps `tx_dropped_no_rx_buffer`.
///   - `RxChainInvalid { add_used_ok }`: chain popped but
///     could not be filled. Exactly ONE of two failure-mode
///     counters was bumped (mutually exclusive per chain):
///     - `rx_chain_invalid` for chain-shape rejection (read-only
///       descriptor in an RX chain, or address-overflow on the
///       descriptor's GPA).
///     - `rx_write_failed` for guest-memory `write_slice`
///       failure (chain shape was fine but the descriptor's GPA
///       is unmapped — header or frame `write_slice` returned
///       Err).
///       The recycle `add_used(head, 0)` was attempted:
///     - If `add_used_ok = true`, the used-ring advanced —
///       caller must kick.
///     - If `add_used_ok = false`, the recycle add_used itself
///       failed, `rx_add_used_failures` was bumped. As with
///       `DeliveredButAddUsedFailed`, the queue is NOT poisoned
///       (transient GPA issue).
///   - `JustRxPoisoned`: RX `iter()` returned any `Err`
///     (most commonly `InvalidAvailRingIndex`; cloud-hypervisor
///     pattern treats every structural queue error uniformly).
///     `invalid_avail_idx_count` was bumped and
///     `queue_poisoned[rx_idx]` JUST transitioned false→true.
///     Caller records the transition; post-loop signal fires.
///     This is the ONLY path that poisons the RX queue.
///   - `RxAlreadyPoisoned`: RX queue's poison flag was already
///     true on entry to `try_loopback_to_rx`. NO counter bump,
///     NO transition. The TX-captured frame is silently dropped
///     and the caller's TX add_used still runs. Returned
///     instead of `NoRxBuffer` so an operator reading the trace
///     log can distinguish "RX queue empty" from "RX queue
///     poisoned, gated short-circuit".
enum LoopbackOutcome {
    Delivered { l2_bytes_written: u64 },
    DeliveredButAddUsedFailed,
    NoRxBuffer,
    RxChainInvalid { add_used_ok: bool },
    JustRxPoisoned,
    RxAlreadyPoisoned,
}

/// Outcome of `pop_and_capture_tx`.
///   - `Empty`: TX queue empty (no chain available, or TX queue
///     is already poisoned and the gate short-circuited at
///     entry). Drain loop should break.
///   - `JustPoisoned`: TX `iter()` returned any `Err` (most
///     commonly `InvalidAvailRingIndex`; cloud-hypervisor pattern
///     treats every structural queue error uniformly).
///     `invalid_avail_idx_count` was bumped and
///     `queue_poisoned[tx_idx]` JUST transitioned false→true. The
///     caller breaks the drain loop and the post-loop signal
///     handler fires `signal_queue_poisoned`. Re-kicks against
///     an already-poisoned TX queue return `Empty` (not
///     `JustPoisoned`) so the counter and signal stay event-once.
///   - `Chain(TxChainOutcome)`: a chain was popped (whether
///     well-formed or not) — the caller proceeds with the
///     per-chain processing and `add_used`.
enum TxPopOutcome {
    Empty,
    JustPoisoned,
    Chain(TxChainOutcome),
}

/// Per-chain inner outcome of `pop_and_capture_tx` (carried inside
/// [`TxPopOutcome::Chain`]).
struct TxChainOutcome {
    head: u16,
    /// `Some(n)` when the chain was valid and `n` L2 bytes (excluding
    /// the 12-byte virtio header) were captured into
    /// `self.tx_frame_scratch[..n]`. `None` when the chain was
    /// malformed — the caller still `add_used`s the head so the guest
    /// can't hang on a malformed request.
    frame_len: Option<usize>,
}

// ---------------------------------------------------------------------------
// Device-status FSM + reset
// ---------------------------------------------------------------------------
//
// The MMIO transport facade (`mmio_read` / `mmio_write`) lives in the
// sibling `mmio.rs` submodule; it decodes the MMIO register layout onto
// the transport-neutral core API above. `set_status` / `reset` are the
// shared FSM core it drives via `write_status`.

impl VirtioNet {
    /// Validate and apply a status transition per virtio-v1.2 §3.1.1.
    /// The driver must not clear bits. Each phase requires the
    /// previous phase's bits to be set. Invalid transitions are
    /// ignored.
    ///
    /// **Feature gates on FEATURES_OK**: per virtio-v1.2 §3.1.1
    /// step 6 + §2.2.1, when the driver writes FEATURES_OK the
    /// device MUST verify that:
    ///   1. All features the device requires were negotiated. This
    ///      device requires `VIRTIO_F_VERSION_1` because it emits a
    ///      12-byte `mrg_rxbuf` header on every RX delivery —
    ///      pre-1.0 transitional drivers expect the 10-byte
    ///      `virtio_net_hdr` (no `num_buffers`) and would treat the
    ///      last 2 bytes of our header as the first 2 bytes of L2
    ///      frame data, silently corrupting every received packet.
    ///   2. The negotiated set is a subset of the offered set —
    ///      i.e. `driver_features & !device_features() == 0`.
    ///      virtio-v1.2 §2.2.1: "the driver MUST NOT accept a
    ///      feature which was not offered by the device". A guest
    ///      that accepts an un-offered bit might enable code paths
    ///      we never tested (e.g. setting the F_MQ bit even though
    ///      we didn't advertise multiqueue would have the kernel
    ///      driver read `max_virtqueue_pairs` from config space,
    ///      which we leave at zero — the kernel's `if
    ///      (max_queue_pairs < MIN || max_queue_pairs > MAX)` branch
    ///      then resets it to 1, but the principle stands).
    ///
    /// On either violation the device sets `VIRTIO_CONFIG_S_FAILED`
    /// and refuses to advance to FEATURES_OK. The kernel driver's
    /// `virtio_features_ok` path (drivers/virtio/virtio.c:204-235)
    /// observes that FEATURES_OK didn't stick on the post-write
    /// STATUS read-back and aborts probe with `-ENODEV`. The FAILED
    /// bit we set is informational; the kernel's check is
    /// `!(status & FEATURES_OK)`, not `status & FAILED`.
    ///
    /// **Divergence from QEMU**: QEMU's `virtio-net` accepts a
    /// FEATURES_OK write that the driver-features check would
    /// otherwise reject by silently masking the unoffered bits in
    /// the negotiated set instead of refusing the transition. This
    /// implementation rejects the transition outright and sets
    /// FAILED. Intentional hardening: a hostile or buggy driver
    /// that asked for an unadvertised feature has lost track of
    /// its own state, and silently downgrading produces a
    /// driver-vs-device feature divergence that's invisible to the
    /// operator. Surfacing the rejection via FAILED + warn is
    /// preferable to silent acceptance — matches firecracker and
    /// cloud-hypervisor.
    fn set_status(&mut self, val: u32) {
        let old = self.device_status;
        // Driver must not clear bits (except via reset, which writes 0).
        if val & self.device_status != self.device_status {
            tracing::debug!(old, val, "virtio-net set_status: rejected (clears bits)");
            return;
        }
        let new_bits = val & !self.device_status;
        let valid = match new_bits {
            VIRTIO_CONFIG_S_ACKNOWLEDGE => self.device_status == 0,
            VIRTIO_CONFIG_S_DRIVER => self.device_status == S_ACK,
            VIRTIO_CONFIG_S_FEATURES_OK => self.device_status == S_DRV,
            VIRTIO_CONFIG_S_DRIVER_OK => self.device_status == S_FEAT,
            _ => false,
        };
        if !valid {
            tracing::debug!(
                old,
                val,
                "virtio-net set_status: rejected (invalid transition)"
            );
            return;
        }
        // Feature gates on the FEATURES_OK transition.
        if new_bits == VIRTIO_CONFIG_S_FEATURES_OK {
            let device_features = self.device_features();
            // Subset rule (virtio-v1.2 §2.2.1): driver must not
            // accept any bit the device did not offer. The bitwise
            // AND-NOT extracts driver-only bits; non-zero means
            // the guest violated the protocol.
            let unoffered = self.driver_features & !device_features;
            if unoffered != 0 {
                self.device_status |= VIRTIO_CONFIG_S_FAILED;
                tracing::warn!(
                    old,
                    attempted = val,
                    driver_features = self.driver_features,
                    device_features,
                    unoffered,
                    "virtio-net set_status: driver accepted features not \
                     offered by device; rejecting FEATURES_OK and setting \
                     FAILED bit"
                );
                return;
            }
            // VERSION_1 requirement: the kernel driver MUST
            // negotiate VERSION_1 — without it our 12-byte header
            // would be interpreted as 10 bytes by the guest.
            if (self.driver_features & (1u64 << VIRTIO_F_VERSION_1)) == 0 {
                self.device_status |= VIRTIO_CONFIG_S_FAILED;
                tracing::warn!(
                    old,
                    attempted = val,
                    "virtio-net set_status: VIRTIO_F_VERSION_1 not \
                     negotiated; rejecting FEATURES_OK and setting FAILED bit"
                );
                return;
            }
        }
        self.device_status = val;
        tracing::debug!(old, new = val, "virtio-net set_status: accepted");
    }

    /// Reset the device to the post-construction state. Clears all
    /// MMIO-side state (status, features, queue config, interrupt
    /// status) and rebuilds the queues. Counters are NOT zeroed —
    /// they persist across re-binds for monotonic operator
    /// observability, matching the virtio-blk pattern.
    ///
    /// Clears `queue_poisoned[..]` for all queues: the guest
    /// issued a virtio reset, which is the only documented escape
    /// from a poisoned-queue state (per the field's invariant —
    /// see [`Self::queue_poisoned`]). The
    /// `invalid_avail_idx_count` counter is intentionally NOT
    /// cleared — operators need cumulative-event visibility
    /// across resets to detect repeated hostile-guest behavior.
    /// Same rationale virtio-blk's `reset_engine_inline` uses.
    fn reset(&mut self) {
        self.device_status = 0;
        self.interrupt_status = 0;
        self.queue_select = 0;
        self.device_features_sel = 0;
        self.driver_features_sel = 0;
        self.driver_features = 0;
        self.tx_frame_scratch.clear();
        // Clear the per-queue poison flags in place (the offered queue count
        // is fixed at construction, so the Vec length is preserved).
        self.queue_poisoned.fill(false);
        // The active pair count returns to the probe default (1); the guest
        // re-sends VQ_PAIRS_SET on re-probe to raise it.
        self.curr_queue_pairs = 1;
        // Virtio reset returns the MSI-X vector assignments to NO_VECTOR
        // (virtio-v1.2 §4.1.4.3); the PCI-level MSI-X cap (enable, table, PBA,
        // registered eventfds) persists — only PCI/FLR reset clears it. INTx
        // transports have no MSI-X state (None) and skip this.
        if let Some(msix) = &self.msix {
            msix.lock().reset_virtio_assignments();
        }
        for q in &mut self.queues {
            q.reset();
        }
    }
}
