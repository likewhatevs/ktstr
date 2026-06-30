//! virtio-pci-modern transport facade for [`VirtioNet`].
//!
//! Presents the virtio-net device as a PCI function (the sibling of the
//! virtio-MMIO facade in [`super::mmio`]): a PCI configuration space with the
//! virtio-modern vendor-capability chain, and a single memory BAR (BAR0)
//! whose four 4 KiB sub-regions (common / ISR / device / notify, virtio-v1.2
//! §4.1.4) the guest's `virtio_pci_modern` driver drives. This file owns ONLY
//! the PCI/BAR decode; all device behaviour (status FSM, feature negotiation,
//! queue ring assembly, notify, interrupt bookkeeping, the loopback drain)
//! lives in the transport-neutral core on [`VirtioNet`] and is shared verbatim
//! with the MMIO facade.
//!
//! Interrupts: MSI-X when offered (the cap is advertised whenever a KVM route
//! owner is wired — on BOTH irqchip paths for an x86 PCI guest: the full-irqchip
//! `FullIrqchipRouteOwner` or the split-irqchip `IoapicHandle`, both
//! `MsixRouteSink`s — the per-vector delivery state + the fire-or-pend gate live
//! in the `interrupt` submodule's `MsixState`, and the cap / table /
//! route-install decode is in this file), and INTx as the fallback (no route
//! owner — non-PCI; or a guest that declines MSI-X), delivered via the device's
//! existing `irq_evt`. The INTx KVM wiring (in `setup::init_virtio_net_pci`) branches on the
//! host-side `vm.split_irqchip` bool; the kernel's `kvm_arch_irqfd_allowed` is
//! the rejector that would -EINVAL a resample irqfd on the split path, which is
//! why we pre-select the edge path there rather than reading that predicate.
//! The branch is on max APIC ID (`split_irqchip = max_apic_id > MAX_XAPIC_ID`,
//! kvm.rs), NOT vCPU count — APIC IDs are sparse, so a wide-core topology with
//! FEWER than 254 vCPUs (e.g. 14 LLCs × 18 cores × 1 thread = 252 vCPUs) can
//! exceed 254 max APIC ID and take the split path.
//!  - Full in-kernel irqchip (≤254 max APIC ID): `register_irqfd_with_resample`
//!    on a level GSI. The KVM level line is asserted and deasserted ENTIRELY by
//!    the resampler source — `signal_used`'s `irq_evt.write(1)` asserts it
//!    (`irqfd_inject`, virt/kvm/eventfd.c:42), and KVM's in-kernel IOAPIC
//!    deasserts the same source on guest EOI (`irqfd_resampler_ack`, :73). The
//!    guest's read-to-clear of the ISR region (below) is INDEPENDENT of that
//!    line: it only clears the guest-visible virtio `interrupt_status` bits, not
//!    the KVM hardware line. (This diverges from qemu, whose
//!    `virtio_pci_isr_read` calls `pci_irq_deassert()` on every ISR read —
//!    including the modern ISR-cap path, hw/virtio/virtio-pci.c:1822 — so on
//!    qemu the ISR read lowers INTx; here KVM, not the device, owns the line.)
//!  - Split irqchip (>254 max APIC ID): a plain edge `register_irqfd` (a
//!    resample irqfd is rejected with -EINVAL there) — the userspace IOAPIC
//!    translates the guest's level/active-low RTE to one-shot edge MSIs, one per
//!    assert.
//!
//! v0 needs no active resample handler (and the edge path no level
//! re-injection), but that rests on a specific invariant: the device has ONE TX
//! queue, and the guest serializes all TX submission on that queue's lock, so
//! the core's two `irq_evt` assert sites — `signal_used` (VRING) and
//! `signal_queue_poisoned` (CONFIG/NEEDS_RESET) — are both reached only
//! synchronously from `process_tx_loopback` on the vCPU thread and are thereby
//! serialized — a kick's interrupt is consumed before the next
//! `irq_evt.write(1)`, so no assert is lost to the level being already-high (an
//! idempotent `kvm_set_irq(...,1)`). This is NOT
//! a property of the transport: multiqueue (per-queue kicks from different
//! vCPUs) or an async RX backend (the device asserting independently of a vCPU
//! kick) would let a second assert coalesce into a still-high level and strand
//! a completion until the next unrelated kick. Both need the active resample
//! handler (drain the resample eventfd, re-assert `irq_evt` if
//! `interrupt_status` is still nonzero after EOI) before they land — see the
//! `NUM_QUEUES == 2` static guard below.

use std::sync::Arc;

use virtio_bindings::virtio_mmio::{VIRTIO_MMIO_INT_CONFIG, VIRTIO_MMIO_INT_VRING};

use super::device::{NUM_QUEUES, VirtioNet};
use super::interrupt::{MSIX_VECTORS, MsixRouteSink, MsixState, NO_VECTOR};
use crate::vmm::PiMutex;
use crate::vmm::pci::{
    ConfigSpace, PCI_COMMAND_MEMORY, PCI_COMMAND_WMASK, PCI_STATUS_CAP_LIST, PciFunction,
    REG_BAR0, REG_CAP_PTR, REG_CLASS, REG_COMMAND, REG_DEVICE_ID, REG_INTERRUPT_LINE,
    REG_INTERRUPT_PIN, REG_REVISION_ID, REG_STATUS, REG_SUBCLASS, REG_VENDOR_ID,
};

// ---- PCI identity (virtio-pci modern, virtio-v1.2 §4.1.2) ----
/// Red Hat / virtio vendor id.
const VENDOR_ID: u16 = 0x1AF4;
/// `0x1040 + VIRTIO_ID_NET (1)` — the modern (non-transitional) virtio-net id.
const DEVICE_ID: u16 = 0x1041;
/// Revision >= 1 marks a non-transitional (modern-only) device.
const REVISION: u8 = 0x01;
/// PCI base class 0x02 (network controller), subclass 0x00 (ethernet).
const CLASS_NETWORK: u8 = 0x02;
const SUBCLASS_ETHERNET: u8 = 0x00;
/// Interrupt pin INTA (1) — the device asserts INTA; the DSDT `_PRT` routes it.
const INTERRUPT_PIN_INTA: u8 = 0x01;

// ---- BAR0 layout: one 4 KiB region per modern-transport structure ----
const REGION_SIZE: u64 = 0x1000;
const COMMON_OFFSET: u64 = 0x0000;
const ISR_OFFSET: u64 = 0x1000;
const DEVICE_OFFSET: u64 = 0x2000;
const NOTIFY_OFFSET: u64 = 0x3000;
/// MSI-X table (one 4 KiB page, page-isolated from the other structures so the
/// guest's `ioremap` of the table page — `pci_alloc_irq_vectors` → `msix_setup`,
/// kernel `drivers/pci/msi/msi.c` — maps only the table). Holds
/// `MSIX_TABLE_SIZE` 16-byte entries.
const MSIX_TABLE_OFFSET: u64 = 0x4000;
/// MSI-X Pending Bit Array (one 4 KiB page; one bit per vector, so a single byte
/// covers `MSIX_TABLE_SIZE`≤8 vectors — the whole page is reserved for it per the
/// spec's page-isolation recommendation).
const MSIX_PBA_OFFSET: u64 = 0x5000;
/// Total BAR0 size: rounded UP to a power of two (PCI BAR decode + the
/// size-probe require it — `BAR0_LOW_WMASK = !(BAR0_SIZE-1)` only recovers a
/// power-of-two). Six 4 KiB structures (common / ISR / device / notify /
/// MSI-X-table / MSI-X-PBA) span 0x6000; the next power of two is 0x8000, so
/// `[0x6000, 0x8000)` is unused padding the guest never accesses.
const BAR0_SIZE: u64 = 0x8000;
/// MSI-X table entries as a `u16` for the Message Control Table-Size field
/// (encoded N-1) and the vector-range clamp. Derived from the delivery module's
/// [`MSIX_VECTORS`] (the single source of truth — its array sizes + SHARED-mode
/// fire logic key off it). Two entries: one shared virtqueue vector + one config
/// vector (virtio-v1.2 §4.1.5.1.2); per-vq MSI-X raises `MSIX_VECTORS`.
const MSIX_TABLE_SIZE: u16 = MSIX_VECTORS as u16;
/// Per-entry size in the MSI-X table: msg addr lo/hi, msg data, vector control
/// (4 × u32 = 16 bytes, PCI spec).
const MSIX_ENTRY_SIZE: u64 = 16;
/// `queue_notify_off(i) == i`, so the notify address for queue `i` is
/// `NOTIFY_OFFSET + i * NOTIFY_OFF_MULTIPLIER`. The returned index lives in
/// `device::queue_notify_off`; these two MUST stay in lockstep (decode is
/// `off / NOTIFY_OFF_MULTIPLIER`) — the multiqueue increment edits both.
const NOTIFY_OFF_MULTIPLIER: u32 = 4;

// The v0 INTx model (no active resample drain on the full-irqchip path; no
// userspace-IOAPIC level re-injection on the split-irqchip edge path) is
// correct ONLY because every `irq_evt` assert is reached SYNCHRONOUSLY from
// `process_tx_loopback` on the vCPU thread. There are two assert sites in the
// core — `signal_used` (VRING) and `signal_queue_poisoned` (CONFIG/NEEDS_RESET)
// — and BOTH are invoked only from within process_tx_loopback, which runs on
// the vCPU thread under the single TX queue's serialization; so each interrupt
// is consumed before the next assert (see the module doc). Guest-side kick
// serialization alone is not enough — the device-side asserts must also be
// serialized, which holds precisely because the loopback is the only path that
// reaches either assert site.
//
// This guard catches ONLY the multiqueue half of the hazard: raising
// NUM_QUEUES turns into a compile error pointing at the active-resample-handler
// + level-re-injection requirement. It does NOT (and cannot here) catch the
// other half — adding an async / off-thread RX backend that asserts `irq_evt`
// independently of a vCPU kick — because v0 has no off-thread assert site to
// gate on. That half is a PROCESS gate: must land the active resample
// drain BEFORE any async RX backend (the module doc spells this out).
const _: () = assert!(
    NUM_QUEUES == 2,
    "virtio-net-PCI INTx assumes all irq_evt asserts originate on the vCPU \
     thread in process_tx_loopback (1 RX + 1 TX queue, no off-thread source); \
     raising NUM_QUEUES requires the active resample handler + split-irqchip \
     level re-injection first (see module doc)"
);

/// BAR0 low-dword type bits: bit0=0 (memory), bits\[2:1\]=00 (32-bit), bit3=0
/// (non-prefetchable) → low nibble `0x0` (`include/uapi/linux/pci_regs.h`).
/// 32-bit non-prefetch matches the host bridge's 32-bit non-prefetch DWord
/// `_CRS` window so the guest assigns BAR0 inside that window (a 64-bit
/// prefetchable BAR is placed above 4 GiB, outside our `_CRS` grant; a
/// future >32-bit BAR would need a QWordMemory `_CRS` window — a later
/// increment).
const BAR0_TYPE_BITS: u32 = 0x00;
/// Writable base bits of BAR0 low: the BAR is `BAR0_SIZE`-aligned, so bits
/// below `log2(size)` are read-only and the guest's size-probe (write
/// all-ones, read back) recovers `BAR0_SIZE`.
const BAR0_LOW_WMASK: u32 = !((BAR0_SIZE as u32) - 1);

// ---- vendor capability chain (struct virtio_pci_cap, virtio-v1.2 §4.1.4) ----
const CAP_VNDR: u8 = 0x09; // PCI_CAP_ID_VNDR
const CFG_TYPE_COMMON: u8 = 1;
const CFG_TYPE_NOTIFY: u8 = 2;
const CFG_TYPE_ISR: u8 = 3;
const CFG_TYPE_DEVICE: u8 = 4;
const CAP_LEN_STD: u8 = 16; // struct virtio_pci_cap
const CAP_LEN_NOTIFY: u8 = 20; // + notify_off_multiplier u32
/// virtio_pci_cap field offsets within a capability.
const CAP_OFF_VNDR: u16 = 0;
const CAP_OFF_NEXT: u16 = 1;
const CAP_OFF_LEN: u16 = 2;
const CAP_OFF_CFG_TYPE: u16 = 3;
const CAP_OFF_BAR: u16 = 4;
const CAP_OFF_OFFSET: u16 = 8;
const CAP_OFF_LENGTH: u16 = 12;
const CAP_OFF_NOTIFY_MULT: u16 = 16;
/// Cap-chain config-space offsets (after the 64-byte standard header).
const CAP_COMMON: u16 = 0x40;
const CAP_ISR: u16 = 0x50;
const CAP_DEVICE: u16 = 0x60;
const CAP_NOTIFY: u16 = 0x70;

// ---- MSI-X capability (PCI_CAP_ID_MSIX, PCI spec §6.8.2) ----
const PCI_CAP_ID_MSIX: u8 = 0x11;
/// MSI-X cap config-space offset: immediately after NOTIFY. NOTIFY is at 0x70
/// with `CAP_LEN_NOTIFY`=20 (0x14), occupying 0x70..0x84, so 0x84 is the next
/// free DWORD-aligned slot. The 12-byte MSI-X cap spans 0x84..0x90, inside the
/// 256-byte config space.
const CAP_MSIX: u16 = 0x84;
/// MSI-X cap field offsets within the capability (PCI spec): Message Control
/// (u16) @+2, Table offset|BIR (u32) @+4, PBA offset|BIR (u32) @+8.
const MSIX_OFF_MSG_CTRL: u16 = 2;
const MSIX_OFF_TABLE: u16 = 4;
const MSIX_OFF_PBA: u16 = 8;
/// Message Control writable bits: bit15 = MSI-X Enable, bit14 = Function Mask
/// (both guest-set/RW); the Table Size field `[10:0]` is read-only.
const MSIX_MSG_CTRL_WMASK: u16 = 0xC000;
/// BIR 0: every MSI-X structure lives in BAR0. The Table/PBA offsets are 8-byte
/// aligned, so the low 3 bits of the offset|BIR dword carry the BIR (here 0).
const MSIX_BIR0: u32 = 0;

// ---- common-cfg register offsets (struct virtio_pci_common_cfg) ----
const CC_DEVICE_FEATURE_SELECT: u64 = 0x00;
const CC_DEVICE_FEATURE: u64 = 0x04;
const CC_DRIVER_FEATURE_SELECT: u64 = 0x08;
const CC_DRIVER_FEATURE: u64 = 0x0C;
const CC_MSIX_CONFIG: u64 = 0x10;
const CC_NUM_QUEUES: u64 = 0x12;
const CC_DEVICE_STATUS: u64 = 0x14;
const CC_CONFIG_GENERATION: u64 = 0x15;
const CC_QUEUE_SELECT: u64 = 0x16;
const CC_QUEUE_SIZE: u64 = 0x18;
const CC_QUEUE_MSIX_VECTOR: u64 = 0x1A;
const CC_QUEUE_ENABLE: u64 = 0x1C;
const CC_QUEUE_NOTIFY_OFF: u64 = 0x1E;
const CC_QUEUE_DESC_LO: u64 = 0x20;
const CC_QUEUE_DESC_HI: u64 = 0x24;
const CC_QUEUE_AVAIL_LO: u64 = 0x28;
const CC_QUEUE_AVAIL_HI: u64 = 0x2C;
const CC_QUEUE_USED_LO: u64 = 0x30;
const CC_QUEUE_USED_HI: u64 = 0x34;

/// The virtio-net device exposed as a PCI function. Owns the device core
/// directly; the [`super::super::pci::PciBus`] lock serializes access (the
/// vCPU thread drives it through BAR MMIO exits). The host-side `irq_evt`
/// (registered with KVM by the setup wiring) and counters Arc are taken from
/// the core before it is moved in here.
pub(crate) struct VirtioNetPci {
    cfg: ConfigSpace,
    net: VirtioNet,
    /// The host-bridge `_CRS` MMIO grant `[start, end)` this function's BAR0
    /// must lie within. `bar_window` rejects any guest-programmed base whose
    /// `[base, base + BAR0_SIZE)` falls outside it, so a non-conformant guest
    /// cannot place BAR0 over the ECAM window (which the MMIO dispatch checks
    /// AFTER the BAR) and shadow config space. Arch-neutral data: the x86
    /// caller passes the host-bridge grant; the facade enforces it like the
    /// reference VMMs (cloud-hypervisor/firecracker) bound their VMM-assigned
    /// BARs.
    bar_aperture: (u64, u64),
    /// Shared MSI-X delivery state: the guest-visible table dwords, vector
    /// assignments (`CC_MSIX_CONFIG` / `CC_QUEUE_MSIX_VECTOR`, echoed verbatim —
    /// the kernel's `vp_modern` aborts `-EBUSY` on a read-back mismatch), the
    /// Pending Bit Array, the device-level enable / function-mask, and the
    /// per-vector eventfds + fire-or-pend gate. Single source of truth: every
    /// guest-visible MSI-X register read/write delegates here, and the device
    /// core holds the other `Arc` clone and signals through it.
    msix: Arc<PiMutex<MsixState>>,
    /// Host KVM route programmer — the active route owner as a `MsixRouteSink`
    /// (`FullIrqchipRouteOwner` on full-irqchip, `IoapicHandle` on split-irqchip);
    /// `None` only without an owner (non-PCI), in which case the facade never
    /// installs a route and the guest stays on INTx. Called on a vector
    /// mask/unmask edge to (re)install or remove the GSI's MSI route.
    route_sink: Option<Arc<dyn MsixRouteSink>>,
    /// The KVM GSI assigned to each MSI-X table vector (`virtio_net_msix_gsi`).
    /// The host registered one irqfd per (vector → eventfd) at this GSI at VM
    /// bring-up; the MSI route is installed here on the vector's unmask edge.
    gsis: [u32; MSIX_VECTORS],
    /// Cached BAR0 MMIO window `[base, base + BAR0_SIZE)` — the value
    /// [`Self::bar_window`] returns; `None` when BAR0 is unprogrammed, memory
    /// decode is disabled, or the base falls outside the `_CRS` grant. Recomputed
    /// ([`Self::recompute_bar_window`]) ONLY on a `REG_COMMAND` / `REG_BAR0` write
    /// — the sole inputs to the window — instead of on every MMIO exit: the
    /// run-loop BAR dispatch calls `bar_window` twice per matching exit (the
    /// `bar_mmio_contains` guard + the `bar_owner` dispatch), each iterating every
    /// populated slot, so a per-exit recompute (two config-space reads × slots ×
    /// 2) scales with NIC count. Coherent under the single `PciBus` mutex: a guest
    /// BAR reprogram (an ECAM `config_write`) and a BAR MMIO access are serialized
    /// on the vCPU thread, so the refresh always precedes the next access (no
    /// atomics needed). Mirrors qemu's `pci_update_mappings` (recompute on config
    /// write, read the cached address per access).
    bar_window_cache: Option<(u64, u64)>,
}

impl VirtioNetPci {
    /// Build the PCI function around an already-constructed [`VirtioNet`]
    /// (the caller has already taken its `irq_evt`/`counters` and called
    /// `set_mem`). Lays out the standard header, the four virtio vendor
    /// capabilities, and BAR0. `bar_aperture` is the host-bridge `_CRS` MMIO
    /// grant `[start, end)` BAR0 must stay within (enforced by `bar_window`).
    pub(crate) fn new(
        net: VirtioNet,
        bar_aperture: (u64, u64),
        msix: Arc<PiMutex<MsixState>>,
        route_sink: Option<Arc<dyn MsixRouteSink>>,
        gsis: [u32; MSIX_VECTORS],
    ) -> Self {
        // Hand the device core its clone of the shared MSI-X state ONLY when
        // MSI-X is actually offered (a route sink is present — on either irqchip
        // path). With no sink (non-PCI) the cap is omitted below and the guest
        // stays on INTx, so the core keeps `msix = None`: its signal paths take
        // the INTx branch with no per-signal lock, keeping INTx byte-identical on
        // the no-MSI-X path (MMIO/aarch64).
        let mut net = net;
        if route_sink.is_some() {
            net.set_msix_state(Arc::clone(&msix));
        }
        let mut cfg = ConfigSpace::new();

        // Standard header.
        cfg.set_u16(REG_VENDOR_ID, VENDOR_ID);
        cfg.set_u16(REG_DEVICE_ID, DEVICE_ID);
        cfg.set_u8(REG_REVISION_ID, REVISION);
        cfg.set_u8(REG_SUBCLASS, SUBCLASS_ETHERNET);
        cfg.set_u8(REG_CLASS, CLASS_NETWORK);
        // Subsystem vendor/device IDs (0x2C/0x2E) are intentionally left zero.
        // This matches qemu's pure virtio-1.0 (modern) path, which sets only
        // VENDOR/DEVICE/revision and leaves PCI_SUBSYSTEM_ID unset
        // (hw/virtio/virtio-pci.c:2091-2098; only the legacy/transitional path
        // sets it at :2087). Zeroing is sound because the modern virtio bus
        // binds on id.device alone — vp_modern_probe derives id.device from
        // DEVICE_ID-0x1040 (=VIRTIO_ID_NET) and virtio_net's id_table matches
        // VIRTIO_DEV_ANY_ID for the vendor, so the subsystem fields are never
        // consulted for binding.
        cfg.set_u8(REG_INTERRUPT_PIN, INTERRUPT_PIN_INTA);
        // INTERRUPT_LINE is R/W (PCI spec) and resets to 0. With no firmware
        // to route legacy INTx, the guest's ACPI `_PRT` is the authoritative
        // source for `pci_dev->irq` (acpi_pci_irq_enable overrides this
        // register from `_PRT`). Kept writable — PIN at 0x3D stays read-only
        // under the 0x00FF mask — so a guest that stores its resolved line
        // reads the value back.
        cfg.set_wmask_u16(REG_INTERRUPT_LINE, 0x00FF);
        // COMMAND: writable so the guest can enable memory-space decode +
        // bus-master; STATUS advertises the capability list. PCI_COMMAND_WMASK
        // also makes INTX_DISABLE (bit 10) writable for read-back, but the device
        // does NOT consult it. INTx is suppressed under MSI-X not via this bit but
        // by the device's signal gate: signal_used / signal_queue_poisoned take
        // the MSI-X path (never asserting irq_evt) once `MsixState::enabled`. The
        // guest's virtio_pci_modern driver sets INTX_DISABLE when it enables MSI-X
        // (pci_intx_for_msi), but the MSI-X-enable gate already achieves the same
        // suppression, so consulting the bit would be redundant. This diverges
        // from qemu (pci_irq_disabled keys off the bit itself) and is inert: in
        // INTx mode the guest leaves INTX_DISABLE clear, and in MSI-X mode the
        // signal gate suppresses INTx regardless of the bit.
        cfg.set_wmask_u16(REG_COMMAND, PCI_COMMAND_WMASK);
        cfg.set_u16(REG_STATUS, PCI_STATUS_CAP_LIST);
        cfg.set_u8(REG_CAP_PTR, CAP_COMMON as u8);

        // BAR0: 32-bit non-prefetchable memory BAR. The low nibble carries
        // the read-only type bits (0x0 = memory / 32-bit / non-prefetch); the
        // base bits are writable (size-aligned). The guest assigns the base in
        // the host bridge's 32-bit non-prefetch _CRS window. A 32-bit BAR0
        // occupies only register 0x10; register 0x14 is BAR1 (unimplemented,
        // reads 0).
        cfg.set_u32(REG_BAR0, BAR0_TYPE_BITS);
        cfg.set_wmask_u32(REG_BAR0, BAR0_LOW_WMASK);

        // MSI-X is advertised iff a route sink is wired (either irqchip path's
        // route owner). With no owner (non-PCI) the cap is omitted so the guest
        // stays on INTx rather than enabling an undeliverable MSI-X.
        Self::write_caps(&mut cfg, route_sink.is_some());
        let mut this = Self {
            cfg,
            net,
            bar_aperture,
            msix,
            route_sink,
            gsis,
            bar_window_cache: None,
        };
        // Seed the cache from the freshly built config space (BAR0 unprogrammed +
        // memory decode disabled at reset → None). Subsequent REG_COMMAND /
        // REG_BAR0 writes refresh it in `config_write`.
        this.bar_window_cache = this.recompute_bar_window();
        this
    }

    /// Lay out the four virtio vendor capabilities (COMMON/ISR/DEVICE/NOTIFY),
    /// chained via `cap_next`. The PCI_CFG capability (cfg_type 5) is
    /// intentionally omitted: `vp_modern_probe` never looks it up (it reads
    /// config only through ECAM/BAR), so a guest binds without it.
    fn write_caps(cfg: &mut ConfigSpace, msix: bool) {
        Self::write_cap(cfg, CAP_COMMON, CAP_ISR, CFG_TYPE_COMMON, COMMON_OFFSET);
        Self::write_cap(cfg, CAP_ISR, CAP_DEVICE, CFG_TYPE_ISR, ISR_OFFSET);
        Self::write_cap(cfg, CAP_DEVICE, CAP_NOTIFY, CFG_TYPE_DEVICE, DEVICE_OFFSET);
        // The uniform REGION_SIZE (4 KiB) notify cap MATCHES the references:
        // qemu sizes its notify region to notify_off_multiplier *
        // VIRTIO_QUEUE_MAX = 4 * 1024 = 0x1000 (hw/virtio/virtio-pci.c), and
        // cloud-hypervisor likewise grants a page-sized notify region — both
        // identical to REGION_SIZE here, not a divergence. It covers queue i's
        // notify address (i * NOTIFY_OFF_MULTIPLIER) for NUM_QUEUES up to
        // 0x1000 / NOTIFY_OFF_MULTIPLIER = 1024 slots — far above the current
        // NUM_QUEUES=2 (offsets 0 and 4). The multiqueue increment must revisit
        // this bound if it ever raises NUM_QUEUES past 1024.
        //
        // NOTIFY chains to the MSI-X cap ONLY when MSI-X is offered (a route sink
        // is wired, on either irqchip path); otherwise it terminates the chain
        // (cap_next = 0). MSI-X is advertised iff the host can install its KVM
        // routes — a guest with no route owner (non-PCI) never sees the cap and
        // stays on INTx, so the device never advertises a delivery path it cannot
        // honor.
        let notify_next: u16 = if msix { CAP_MSIX } else { 0 };
        Self::write_cap(cfg, CAP_NOTIFY, notify_next, CFG_TYPE_NOTIFY, NOTIFY_OFFSET);
        cfg.set_u32(CAP_NOTIFY + CAP_OFF_NOTIFY_MULT, NOTIFY_OFF_MULTIPLIER);
        if msix {
            // MSI-X is the last cap (cap_next = 0), chained after NOTIFY.
            Self::write_msix_cap(cfg);
        }
    }

    /// Write the standard MSI-X capability at `CAP_MSIX`: cap id, next = 0 (last
    /// cap), Message Control (Table Size encoded as N-1 per the PCI spec, with
    /// Enable/Function-Mask left guest-writable via `MSIX_MSG_CTRL_WMASK`), and
    /// the Table / PBA `offset|BIR` dwords pointing into BAR0. The
    /// guest's `pci_alloc_irq_vectors` reads Table Size from here and maps the
    /// Table/PBA pages from the advertised BAR0 offsets.
    fn write_msix_cap(cfg: &mut ConfigSpace) {
        cfg.set_u8(CAP_MSIX, PCI_CAP_ID_MSIX);
        cfg.set_u8(CAP_MSIX + 1, 0);
        cfg.set_u16(CAP_MSIX + MSIX_OFF_MSG_CTRL, MSIX_TABLE_SIZE - 1);
        cfg.set_wmask_u16(CAP_MSIX + MSIX_OFF_MSG_CTRL, MSIX_MSG_CTRL_WMASK);
        cfg.set_u32(CAP_MSIX + MSIX_OFF_TABLE, MSIX_TABLE_OFFSET as u32 | MSIX_BIR0);
        cfg.set_u32(CAP_MSIX + MSIX_OFF_PBA, MSIX_PBA_OFFSET as u32 | MSIX_BIR0);
    }

    /// Write one `struct virtio_pci_cap` at config offset `at`, pointing to
    /// the BAR0 sub-region `[region_off, region_off + REGION_SIZE)`. The cap
    /// LENGTH field is the full 4 KiB sub-region (REGION_SIZE), NOT the exact
    /// struct size the references advertise (cloud-hypervisor uses
    /// COMMON_CONFIG_SIZE=56 / ISR=1). Benign + intentional: Linux's vp_modern
    /// addresses each register at its fixed offset and never iterates the region
    /// by LENGTH, and a uniform 4 KiB keeps the sub-regions page-disjoint.
    fn write_cap(cfg: &mut ConfigSpace, at: u16, next: u16, cfg_type: u8, region_off: u64) {
        cfg.set_u8(at + CAP_OFF_VNDR, CAP_VNDR);
        cfg.set_u8(at + CAP_OFF_NEXT, next as u8);
        let cap_len = if cfg_type == CFG_TYPE_NOTIFY {
            CAP_LEN_NOTIFY
        } else {
            CAP_LEN_STD
        };
        cfg.set_u8(at + CAP_OFF_LEN, cap_len);
        cfg.set_u8(at + CAP_OFF_CFG_TYPE, cfg_type);
        cfg.set_u8(at + CAP_OFF_BAR, 0); // all regions live in BAR0
        cfg.set_u32(at + CAP_OFF_OFFSET, region_off as u32);
        cfg.set_u32(at + CAP_OFF_LENGTH, REGION_SIZE as u32);
    }

    /// Pull `data.len()` (≤ 8) bytes of a little-endian register `val` into
    /// `data`; the guest reads each register at its natural width.
    fn put_le(val: u64, data: &mut [u8]) {
        let bytes = val.to_le_bytes();
        data.fill(0);
        let n = data.len().min(8);
        data[..n].copy_from_slice(&bytes[..n]);
    }

    /// Read up to 4 little-endian bytes of `data` as a `u32` (a common-cfg
    /// write is 1/2/4 bytes wide).
    fn get_u32(data: &[u8]) -> u32 {
        let mut buf = [0u8; 4];
        let n = data.len().min(4);
        buf[..n].copy_from_slice(&data[..n]);
        u32::from_le_bytes(buf)
    }

    /// Clamp a guest-written MSI-X vector number: a valid index
    /// `[0, MSIX_TABLE_SIZE)` or `NO_VECTOR` (disable) is kept verbatim; any
    /// other value is out of range and stored as `NO_VECTOR`, so the read-back
    /// signals rejection rather than letting a hostile vector index the table
    /// out of bounds.
    fn clamp_vector(v: u16) -> u16 {
        if v == NO_VECTOR || v < MSIX_TABLE_SIZE {
            v
        } else {
            NO_VECTOR
        }
    }

    /// MSI-X vector assigned to the currently-selected virtqueue, read from the
    /// shared state; `NO_VECTOR` for an out-of-range selector (mirrors the
    /// queue_size / queue_ready clamp on a hostile `queue_select`).
    fn selected_queue_msix_vector(&self) -> u16 {
        let sel = self.net.queue_select() as usize;
        self.msix.lock().queue_vector(sel)
    }

    /// Serve a common-cfg register read at BAR-region offset `off`. The select
    /// registers read back their latched value; the feature/queue data
    /// registers serve the core's current view; device-read registers
    /// (queue desc/avail/used) read 0.
    ///
    /// Dispatch is offset-exact and natural-width: the guest's
    /// `virtio_pci_modern` driver reads each register at its own offset and
    /// width (per-field `ioread8`/`ioread16`/`ioread32`), so the `match off`
    /// keys on the register's base offset and `put_le` serves it at the
    /// access width. A sub-dword read of a packed pair (e.g. a 2-byte read
    /// straddling `device_status`@0x14 + `config_generation`@0x15) matches
    /// only the base register and zero-fills the overhang — the driver never
    /// issues such a straddling access, and pinning the per-field 1-byte reads
    /// is left to the unit tests.
    fn common_read(&self, off: u64, data: &mut [u8]) {
        let val: u64 = match off {
            CC_DEVICE_FEATURE_SELECT => self.net.device_features_sel() as u64,
            CC_DEVICE_FEATURE => self.net.device_features_window() as u64,
            CC_DRIVER_FEATURE_SELECT => self.net.driver_features_sel() as u64,
            CC_MSIX_CONFIG => self.msix.lock().config_vector() as u64,
            CC_NUM_QUEUES => NUM_QUEUES as u64,
            CC_DEVICE_STATUS => self.net.device_status() as u64,
            CC_CONFIG_GENERATION => self.net.config_generation() as u64,
            // queue_select reads back the RAW latched selector per virtio-v1.2
            // §4.1.4.3 (the register is a plain latch). The queue-INDEXED
            // read-backs below are all clamped for an out-of-range selector so
            // a hostile latch reads defined values and drives no op:
            // queue_size/queue_ready/queue_notify_off return 0, and the core's
            // selected_queue() bounds-checks before any actual queue op.
            CC_QUEUE_SELECT => self.net.queue_select() as u64,
            // queue_size is R/W (virtio-v1.2 §4.1.4.3): read back the
            // guest-configured size (resets to max_size), not the immutable max.
            CC_QUEUE_SIZE => self.net.queue_size() as u64,
            CC_QUEUE_MSIX_VECTOR => self.selected_queue_msix_vector() as u64,
            CC_QUEUE_ENABLE => self.net.queue_ready() as u64,
            // notify offsets map in queue order (queue i's offset is its
            // index); clamped to 0 for an out-of-range selector.
            CC_QUEUE_NOTIFY_OFF => self.net.queue_notify_off() as u64,
            // driver_feature (write-only in practice) + the queue ring
            // addresses are not read back by the driver.
            _ => 0,
        };
        Self::put_le(val, data);
    }

    /// Apply a common-cfg register write at BAR-region offset `off`. Each
    /// register maps to a transport-neutral core op (the gates live inside
    /// the op); read-only registers (device_feature/num_queues/
    /// config_generation/queue_notify_off) and the INTx-inert MSI-X
    /// registers are dropped.
    ///
    /// Offset-exact, natural-width (the write mirror of `common_read`): the
    /// guest writes each register at its own offset/width, and the 64-bit
    /// queue ring addresses arrive as two 32-bit halves (the kernel's
    /// `vp_iowrite64_twopart` writes DESC_LO then DESC_HI separately), so
    /// `get_u32` taking the low 4 bytes is correct. A single 8-byte write to a
    /// ring-address register is non-spec and unreachable (Linux always issues
    /// the two halves); were one to occur, only the low dword would land and the
    /// HI half would stay stale — pointing the ring at a wrong GPA. The kernel
    /// never issues it, so the overhang is intentionally ignored rather than
    /// handled.
    fn common_write(&mut self, off: u64, data: &[u8]) {
        let val = Self::get_u32(data);
        match off {
            CC_DEVICE_FEATURE_SELECT => self.net.set_device_features_sel(val),
            CC_DRIVER_FEATURE_SELECT => self.net.set_driver_features_sel(val),
            CC_DRIVER_FEATURE => self.net.set_driver_features_window(val),
            CC_DEVICE_STATUS => self.net.write_status(val & 0xFF),
            CC_QUEUE_SELECT => self.net.set_queue_select(val),
            CC_QUEUE_SIZE => self.net.set_queue_size(val as u16),
            CC_QUEUE_ENABLE => self.net.set_queue_ready(val != 0),
            CC_QUEUE_DESC_LO => self.net.set_queue_desc_addr(Some(val), None),
            CC_QUEUE_DESC_HI => self.net.set_queue_desc_addr(None, Some(val)),
            CC_QUEUE_AVAIL_LO => self.net.set_queue_avail_addr(Some(val), None),
            CC_QUEUE_AVAIL_HI => self.net.set_queue_avail_addr(None, Some(val)),
            CC_QUEUE_USED_LO => self.net.set_queue_used_addr(Some(val), None),
            CC_QUEUE_USED_HI => self.net.set_queue_used_addr(None, Some(val)),
            // MSI-X vector latches: store the guest-assigned vector (clamped to a
            // valid index or NO_VECTOR) in the shared state so the readback
            // echoes it — the kernel's vp_modern aborts -EBUSY on a read-back
            // mismatch. An out-of-range selector drops the queue write
            // (set_queue_vector bounds-checks the queue index).
            CC_MSIX_CONFIG => self.msix.lock().set_config_vector(Self::clamp_vector(val as u16)),
            CC_QUEUE_MSIX_VECTOR => {
                let sel = self.net.queue_select() as usize;
                self.msix
                    .lock()
                    .set_queue_vector(sel, Self::clamp_vector(val as u16));
            }
            _ => {}
        }
    }

    /// Serve the 1-byte ISR register: bit0 = queue interrupt, bit1 =
    /// config-change, mapped from the core's interrupt-status bits
    /// (`INT_VRING`/`INT_CONFIG`, numerically the same low two bits). The read
    /// is read-to-clear — there is no separate ACK register on the modern PCI
    /// ISR cap — so it unconditionally clears BOTH bits (`INT_VRING` and
    /// `INT_CONFIG`) on every served read, regardless of which were set,
    /// matching qemu's `virtio_pci_isr_read` xchg that zeroes the whole register.
    /// Reached only at `ISR_OFFSET` (the caller gates on the exact offset), and
    /// a zero-length read is a no-op: the read-to-clear ack is gated on a byte
    /// actually being served, so a malformed empty access cannot silently
    /// consume a pending interrupt.
    fn isr_read(&mut self, data: &mut [u8]) {
        if data.is_empty() {
            return;
        }
        let isr = self.net.interrupt_status();
        let mut byte = 0u8;
        if isr & VIRTIO_MMIO_INT_VRING != 0 {
            byte |= 0x1;
        }
        if isr & VIRTIO_MMIO_INT_CONFIG != 0 {
            byte |= 0x2;
        }
        self.net.ack_interrupt(VIRTIO_MMIO_INT_VRING | VIRTIO_MMIO_INT_CONFIG);
        data.fill(0);
        if let Some(b) = data.first_mut() {
            *b = byte;
        }
    }

    /// Decode a notify-region write to a queue index and kick it. The index is
    /// `off / NOTIFY_OFF_MULTIPLIER` (floor division — a conformant guest writes
    /// `queue_index * NOTIFY_OFF_MULTIPLIER`, but a misaligned `off` is decoded
    /// by the same floor). `notify_queue` kicks ONLY when the quotient is the TX
    /// index; every other quotient — RX, or an out-of-range value from a hostile
    /// write — is a no-op. So any `off` in `[TXQ*mult, (TXQ+1)*mult)` kicks TX
    /// and every other range is inert (it does not validate the raw offset).
    fn notify_write(&mut self, off: u64) {
        let idx = off / NOTIFY_OFF_MULTIPLIER as u64;
        self.net.notify_queue(idx as u32);
    }

    /// Serve an MSI-X table read at table-relative offset `rel`: decode
    /// `(entry, dword)` and return the stored register; zero for an
    /// out-of-range entry (a conformant guest never reads past the advertised
    /// table size; the bound keeps a hostile read in-bounds).
    fn msix_table_read(&self, rel: u64, data: &mut [u8]) {
        let entry = (rel / MSIX_ENTRY_SIZE) as usize;
        let dword = ((rel % MSIX_ENTRY_SIZE) / 4) as usize;
        let val = self.msix.lock().table_dword(entry, dword);
        Self::put_le(val as u64, data);
    }

    /// Apply an MSI-X table write at table-relative offset `rel`: decode
    /// `(entry, dword)` and store the register in the shared state; out-of-range
    /// entries are dropped (the bound keeps a hostile write in-bounds). Dword 3
    /// is the Vector Control register (bit0 = the per-vector mask). After the
    /// write, reconcile the entry's GSI route to its current deliverability:
    /// INSTALL on the mask→unmask edge (the kernel's `irq_startup` clears bit0
    /// after writing addr/data, drivers/pci/msi/) and on a non-spec addr/data
    /// rewrite while unmasked; REMOVE on the unmask→mask edge so a masked vector
    /// has no live KVM route (masking enforced at the KVM layer, not only the
    /// device-side PBA gate). On install, `install_route` replays any pending bit.
    fn msix_table_write(&mut self, rel: u64, data: &[u8]) {
        let entry = (rel / MSIX_ENTRY_SIZE) as usize;
        let dword = ((rel % MSIX_ENTRY_SIZE) / 4) as usize;
        let val = Self::get_u32(data);
        // The unmask-edge return is informational (unit-tested in `interrupt`);
        // the facade reconciles the entry's route to its current deliverability,
        // which subsumes the unmask edge (install), the mask edge (remove), and a
        // non-spec addr/data rewrite while unmasked (re-install).
        let _ = self.msix.lock().write_table_dword(entry, dword, val);
        self.reconcile_route(entry);
    }

    /// Reconcile vector `idx`'s KVM MSI route to its current deliverability
    /// (enabled && per-vector-unmasked && !function_mask): install it when
    /// deliverable, remove it otherwise. Used on every table-write and Message
    /// Control edge so masking/disable is enforced at the KVM route layer, not
    /// only device-side.
    fn reconcile_route(&mut self, idx: usize) {
        let deliverable = {
            let m = self.msix.lock();
            m.enabled() && m.vector_unmasked(idx)
        };
        if deliverable {
            self.install_route(idx);
        } else {
            self.remove_route(idx);
        }
    }

    /// Reconcile every vector (device-level Message Control edge: enable /
    /// disable / function-mask toggle).
    fn reconcile_routes(&mut self) {
        for idx in 0..MSIX_VECTORS {
            self.reconcile_route(idx);
        }
    }

    /// (Re)install the KVM MSI route for table vector `idx` from its current
    /// table dwords, then replay any pending bit. No-op when no route sink is
    /// wired (non-PCI) — the guest's MSI-X then never delivers and it falls back
    /// to INTx. The msix lock is released before the (SRCU-grace) route ioctl so
    /// the slow ioctl never holds it.
    fn install_route(&mut self, idx: usize) {
        let Some(sink) = self.route_sink.as_ref() else {
            return;
        };
        let Some(&gsi) = self.gsis.get(idx) else {
            return;
        };
        let msg = self.msix.lock().msi_message(idx);
        if let Some(msg) = msg {
            sink.set_route(gsi, Some(msg));
            // Now that the route is live, replay a pending bit recorded while the
            // vector was masked (the eventfd write delivers through the route).
            self.msix.lock().replay_pending(idx);
        }
    }

    /// Remove the KVM MSI route for vector `idx` on a mask / disable /
    /// function-mask edge. The device-side `fire()` gate already suppresses a
    /// masked vector (records a PBA bit instead of writing the eventfd), so this
    /// is belt-and-suspenders that ALSO enforces masking at the KVM layer —
    /// matching firecracker (which drops a masked vector's route from the table)
    /// and removing the latent risk that an off-thread eventfd writer (a
    /// multiqueue / async-RX follow-up) could deliver through a still-live route.
    /// No-op without a route sink. Deduped by the owner's `last_installed`, so
    /// removing an already-absent route costs no ioctl.
    fn remove_route(&mut self, idx: usize) {
        let Some(sink) = self.route_sink.as_ref() else {
            return;
        };
        let Some(&gsi) = self.gsis.get(idx) else {
            return;
        };
        sink.set_route(gsi, None);
    }

    /// Whether a config-space write at `[reg, reg+len)` overlaps the register at
    /// `[target, target+target_len)`.
    fn write_touches(reg: u16, len: usize, target: u16, target_len: u16) -> bool {
        let (r, r_end) = (reg as u32, reg as u32 + len as u32);
        let (t, t_end) = (target as u32, target as u32 + target_len as u32);
        r < t_end && t < r_end
    }

    /// Serve an MSI-X PBA read: the pending-bits byte at `rel` 0, zero
    /// elsewhere (the PBA occupies a page but only `MSIX_VECTORS` ≤ 8 bits
    /// are defined). Read-only — `bar_write` drops PBA writes.
    fn msix_pba_read(&self, rel: u64, data: &mut [u8]) {
        let val = if rel == 0 {
            self.msix.lock().pba_byte() as u64
        } else {
            0
        };
        Self::put_le(val, data);
    }

    /// Recompute the BAR0 MMIO window from config space — the value cached in
    /// `bar_window_cache` and returned by [`Self::bar_window`]. Called at
    /// construction and on each `REG_COMMAND` / `REG_BAR0` write (`config_write`),
    /// NOT per MMIO exit.
    ///
    /// The BAR is live only once the guest enables memory-space decode. Only
    /// `PCI_COMMAND_MEMORY` is consulted, NOT bus-master (bit 2): on real hardware
    /// the device cannot DMA until BM is set, but the in-VMM loopback reads/writes
    /// guest memory unconditionally. Benign — Linux's `virtio_pci_modern_probe`
    /// sets bus-master before DRIVER_OK, and the loopback only touches memory on a
    /// post-DRIVER_OK notify — so BM is always set before first use; honoring it
    /// would land with MSI-X (like INTX_DISABLE above).
    fn recompute_bar_window(&self) -> Option<(u64, u64)> {
        let mut cmd = [0u8; 2];
        self.cfg.read(REG_COMMAND, &mut cmd);
        if u16::from_le_bytes(cmd) & PCI_COMMAND_MEMORY == 0 {
            return None;
        }
        // The 32-bit base from BAR0, masking the read-only type bits.
        let mut lo = [0u8; 4];
        self.cfg.read(REG_BAR0, &mut lo);
        let base = (u32::from_le_bytes(lo) & !(BAR0_SIZE as u32 - 1)) as u64;
        if base == 0 {
            // Fast-path for the common reset value (BAR unprogrammed). Redundant
            // with the grant check below — base 0 is far below grant_start
            // (PCI_MMIO_BAR_BASE) so that check also returns None — but the
            // explicit branch documents the reset case; benign (a guest can only
            // break its own NIC, never host memory).
            return None;
        }
        // Enforce the host-bridge _CRS MMIO grant: the window is only honored when
        // [base, base + BAR0_SIZE) lies fully within the aperture the guest's PCI0
        // _CRS advertises. The MMIO dispatch checks bar_mmio_contains BEFORE
        // ecam_contains, so without this bound a guest that (non-conformantly)
        // programmed BAR0 over the ECAM window would shadow config space; rejecting
        // an out-of-grant base (return None) makes that guest-only breakage
        // impossible. Linux always assigns BARs inside the granted window, so a
        // conformant guest is unaffected. Matches the reference VMMs, which
        // assign/bound BARs from a VMM-owned allocator rather than trusting the
        // guest base.
        let (grant_start, grant_end) = self.bar_aperture;
        if base < grant_start || base.saturating_add(BAR0_SIZE) > grant_end {
            return None;
        }
        Some((base, BAR0_SIZE))
    }
}

impl PciFunction for VirtioNetPci {
    fn config_read(&self, reg: u16, data: &mut [u8]) {
        self.cfg.read(reg, data);
    }

    fn config_write(&mut self, reg: u16, data: &[u8]) {
        self.cfg.write(reg, data);
        // Refresh the cached BAR0 window when the write touched its only inputs —
        // REG_COMMAND (memory-decode enable bit) or REG_BAR0 (base). Uses
        // write_touches (not `reg ==`) so a wide/odd access spanning either
        // register still refreshes (e.g. a 4-byte write at REG_COMMAND spans
        // COMMAND+STATUS; Linux writes BAR0 as the full dword). Recompute AFTER
        // cfg.write so it reads the just-written bytes. See `bar_window_cache`.
        if Self::write_touches(reg, data.len(), REG_COMMAND, 2)
            || Self::write_touches(reg, data.len(), REG_BAR0, 4)
        {
            self.bar_window_cache = self.recompute_bar_window();
        }
        // If the write touched the MSI-X cap Message Control (Enable bit15 /
        // Function Mask bit14), sync the parsed state into the shared delivery
        // state, then reconcile every vector's KVM route. The kernel's setup
        // sequence (Enable+MASKALL, program vectors, clear MASKALL) and teardown
        // (clear Enable) all land here; reconcile_routes installs each now-
        // deliverable vector and removes each now-masked/disabled one in one
        // pass, so the route table tracks the enable/function-mask state without
        // per-edge bookkeeping. (Per-vector unmask edges land in
        // msix_table_write, which reconciles that one entry.)
        let mc_reg = CAP_MSIX + MSIX_OFF_MSG_CTRL;
        if Self::write_touches(reg, data.len(), mc_reg, 2) {
            let mut mc = [0u8; 2];
            self.cfg.read(mc_reg, &mut mc);
            let msg_ctrl = u16::from_le_bytes(mc);
            self.msix.lock().set_message_control(msg_ctrl);
            // Reconcile every vector's KVM route to the new enable / function-mask
            // state: enable or function-unmask installs the unmasked vectors (and
            // install_route replays their pending bits); disable or function-mask
            // removes them so masking is enforced at the KVM layer too. Deduped by
            // the owner, so unchanged vectors cost no ioctl.
            self.reconcile_routes();
        }
    }

    fn bar_window(&self) -> Option<(u64, u64)> {
        // The cached window (`bar_window_cache`), recomputed only on REG_COMMAND /
        // REG_BAR0 writes (see `recompute_bar_window` + `config_write`). This MUST
        // be a field read, not a per-exit recompute: the run-loop BAR dispatch
        // calls it twice per matching MMIO exit (the `bar_mmio_contains` guard +
        // the `bar_owner` dispatch), each over every populated slot.
        self.bar_window_cache
    }

    fn bar_read(&mut self, offset: u64, data: &mut [u8]) {
        // Region dispatch keys on the access START offset only. A spec-compliant
        // guest never issues an access that straddles a REGION_SIZE (4 KiB)
        // modern-transport boundary, so a non-spec cross-region access is served
        // entirely by the start region and any overhang is zero-padded — never
        // OOB (common_read/config_bytes/put_le all length-bound). Same in
        // bar_write.
        match offset {
            o if (COMMON_OFFSET..COMMON_OFFSET + REGION_SIZE).contains(&o) => {
                self.common_read(o - COMMON_OFFSET, data);
            }
            // Only the ISR register itself (region base) read-clears; any other
            // offset in the ISR region reads 0 (falls through). The kernel's
            // vp_interrupt reads the 1-byte ISR at the cap base only, so gating
            // on the exact offset keeps a stray in-region access from acking a
            // pending interrupt. This is STRICTER than qemu, whose
            // virtio_pci_isr_read xchg-clears on any read within the (1-byte)
            // ISR region; the narrowing is safe precisely because the kernel
            // never reads anywhere but the cap base.
            ISR_OFFSET => {
                self.isr_read(data);
            }
            o if (DEVICE_OFFSET..DEVICE_OFFSET + REGION_SIZE).contains(&o) => {
                self.net.config_bytes((o - DEVICE_OFFSET) as usize, data);
            }
            o if (MSIX_TABLE_OFFSET..MSIX_TABLE_OFFSET + REGION_SIZE).contains(&o) => {
                self.msix_table_read(o - MSIX_TABLE_OFFSET, data);
            }
            o if (MSIX_PBA_OFFSET..MSIX_PBA_OFFSET + REGION_SIZE).contains(&o) => {
                self.msix_pba_read(o - MSIX_PBA_OFFSET, data);
            }
            // notify region is write-only; reads return 0.
            _ => data.fill(0),
        }
    }

    fn bar_write(&mut self, offset: u64, data: &[u8]) {
        match offset {
            o if (COMMON_OFFSET..COMMON_OFFSET + REGION_SIZE).contains(&o) => {
                self.common_write(o - COMMON_OFFSET, data);
            }
            o if (NOTIFY_OFFSET..NOTIFY_OFFSET + REGION_SIZE).contains(&o) => {
                self.notify_write(o - NOTIFY_OFFSET);
            }
            o if (MSIX_TABLE_OFFSET..MSIX_TABLE_OFFSET + REGION_SIZE).contains(&o) => {
                self.msix_table_write(o - MSIX_TABLE_OFFSET, data);
            }
            // MSI-X PBA + ISR are read-only; device-config writes are ignored
            // (virtio-v1.2 §4.2.2.2 — the device may ignore config-space writes).
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    //! Host-runnable tests for the virtio-net PCI transport facade — the
    //! decode glue (config space, vendor-cap chain, BAR-region dispatch,
    //! common-cfg register mapping, ISR read-to-clear, notify decode,
    //! bar_window gating) that the booted-guest NIC e2es exercise but cannot
    //! guard in CI (they SKIP-as-PASS without KVM). The transport-neutral
    //! loopback core is covered by `super::super::tests`; here a minimal
    //! loopback is driven ENTIRELY through the PCI facade to prove the decode
    //! path reaches the core and the ISR reflects/clears the interrupt.
    use super::*;
    use crate::vmm::net_config::NetConfig;
    use crate::vmm::virtio_net::device::{
        QUEUE_MAX_SIZE, RXQ, S_ACK, S_DRV, S_FEAT, S_OK, TXQ, VIRTIO_NET_HDR_LEN,
    };
    use proptest::prelude::*;
    use std::sync::atomic::Ordering;
    use virtio_bindings::virtio_config::{VIRTIO_CONFIG_S_NEEDS_RESET, VIRTIO_F_VERSION_1};
    use virtio_bindings::virtio_net::VIRTIO_NET_F_MAC;
    use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};
    use vmm_sys_util::eventfd::EventFd;

    // Compact guest-memory ring layout (the full-fidelity helpers live in
    // super::super::tests; this isolated harness only needs one TX + one RX
    // chain to set the ISR through the facade).
    const GUEST_MEM_SIZE: usize = 0x10_0000;
    const TX_DESC: u64 = 0x1000;
    const TX_AVAIL: u64 = 0x2000;
    const TX_USED: u64 = 0x3000;
    const TX_BUF: u64 = 0x5000;
    const RX_DESC: u64 = 0x6000;
    const RX_AVAIL: u64 = 0x7000;
    const RX_USED: u64 = 0x8000;
    const RX_BUF: u64 = 0x9000;

    // The host-bridge _CRS MMIO grant the production x86 caller passes (literal
    // here to keep this all-arch test module off the x86-only kvm constants):
    // [PCI_MMIO_BAR_BASE, IOAPIC_MMIO_BASE) = [0xE010_0000, 0xFEC0_0000). The
    // ECAM window sits just below it at 0xE000_0000 (out of grant).
    const TEST_BAR_APERTURE: (u64, u64) = (0xE010_0000, 0xFEC0_0000);
    const TEST_BAR_BASE: u32 = 0xE010_0000; // in-grant base used by BAR tests
    const TEST_ECAM_BASE: u32 = 0xE000_0000; // out-of-grant (would shadow ECAM)

    fn test_mem() -> GuestMemoryMmap {
        GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), GUEST_MEM_SIZE)]).unwrap()
    }

    /// A counting MSI-X route sink for the facade tests: records every
    /// `set_route(gsi, msg)` so a test can assert which routes the unmask /
    /// enable edges installed, without a live KVM fd. `Send + Sync` (the trait
    /// requires it) via the inner mutex.
    #[derive(Default)]
    #[allow(clippy::type_complexity)] // (gsi, Option<msi-message>) log, test-only
    struct MockRouteSink {
        installs: std::sync::Mutex<Vec<(u32, Option<(u32, u32, u32)>)>>,
    }
    impl MsixRouteSink for MockRouteSink {
        fn set_route(&self, gsi: u32, msg: Option<(u32, u32, u32)>) {
            self.installs.lock().unwrap().push((gsi, msg));
        }
    }

    /// Default MSI-X GSIs for the tests (`virtio_net_msix_gsi(0, v)` = 24 + v,
    /// kept literal here to keep this all-arch test module off the x86-only kvm
    /// constants).
    fn test_gsis() -> [u32; MSIX_VECTORS] {
        std::array::from_fn(|v| 24 + v as u32)
    }

    /// Wrap `net` in a PCI function with a fresh (disabled) MSI-X state and a
    /// no-asserted mock route sink — so the MSI-X cap IS advertised (the
    /// decode-path tests assert the cap chain) but, since they don't enable
    /// MSI-X, no route is actually installed. The MSI-X delivery tests build
    /// their own state + eventfds + a retained mock sink.
    fn new_pci(net: VirtioNet) -> VirtioNetPci {
        let msix = Arc::new(PiMutex::new(MsixState::new()));
        let sink: Arc<dyn MsixRouteSink> = Arc::new(MockRouteSink::default());
        VirtioNetPci::new(net, TEST_BAR_APERTURE, msix, Some(sink), test_gsis())
    }

    fn build(mem: &GuestMemoryMmap) -> (VirtioNetPci, std::sync::Arc<super::super::VirtioNetCounters>) {
        let mut net = VirtioNet::new(NetConfig::default());
        net.set_mem(mem.clone());
        let counters = net.counters();
        (new_pci(net), counters)
    }

    fn cfg8(pci: &VirtioNetPci, reg: u16) -> u8 {
        let mut b = [0u8; 1];
        pci.config_read(reg, &mut b);
        b[0]
    }
    fn cfg16(pci: &VirtioNetPci, reg: u16) -> u16 {
        let mut b = [0u8; 2];
        pci.config_read(reg, &mut b);
        u16::from_le_bytes(b)
    }
    fn cfg32(pci: &VirtioNetPci, reg: u16) -> u32 {
        let mut b = [0u8; 4];
        pci.config_read(reg, &mut b);
        u32::from_le_bytes(b)
    }

    /// Write a common-cfg register (32-bit) via the BAR0 COMMON region.
    fn cc_w(pci: &mut VirtioNetPci, cc: u64, val: u32) {
        pci.bar_write(COMMON_OFFSET + cc, &val.to_le_bytes());
    }
    /// Read a common-cfg register (32-bit) via the BAR0 COMMON region.
    fn cc_r(pci: &mut VirtioNetPci, cc: u64) -> u32 {
        let mut b = [0u8; 4];
        pci.bar_read(COMMON_OFFSET + cc, &mut b);
        u32::from_le_bytes(b)
    }

    /// Drive ACK -> DRIVER -> negotiate VERSION_1 + MAC -> FEATURES_OK via the
    /// common-cfg device_status + driver_feature registers (the PCI analog of
    /// the MMIO init_until_features_ok). Stops short of DRIVER_OK so queue
    /// config is still allowed.
    fn drive_to_features_ok(pci: &mut VirtioNetPci) {
        cc_w(pci, CC_DEVICE_STATUS, S_ACK);
        cc_w(pci, CC_DEVICE_STATUS, S_DRV);
        cc_w(pci, CC_DRIVER_FEATURE_SELECT, 0);
        cc_w(pci, CC_DRIVER_FEATURE, 1u32 << VIRTIO_NET_F_MAC);
        cc_w(pci, CC_DRIVER_FEATURE_SELECT, 1);
        cc_w(pci, CC_DRIVER_FEATURE, 1u32 << (VIRTIO_F_VERSION_1 - 32));
        cc_w(pci, CC_DEVICE_STATUS, S_FEAT);
    }

    fn program_queue(pci: &mut VirtioNetPci, q: u32, desc: u64, avail: u64, used: u64) {
        cc_w(pci, CC_QUEUE_SELECT, q);
        cc_w(pci, CC_QUEUE_SIZE, 4);
        cc_w(pci, CC_QUEUE_DESC_LO, desc as u32);
        cc_w(pci, CC_QUEUE_AVAIL_LO, avail as u32);
        cc_w(pci, CC_QUEUE_USED_LO, used as u32);
        cc_w(pci, CC_QUEUE_ENABLE, 1);
    }

    fn write_desc(
        mem: &GuestMemoryMmap,
        table: u64,
        idx: u16,
        addr: u64,
        len: u32,
        flags: u16,
        next: u16,
    ) {
        let off = table + (idx as u64) * 16;
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&addr.to_le_bytes());
        buf[8..12].copy_from_slice(&len.to_le_bytes());
        buf[12..14].copy_from_slice(&flags.to_le_bytes());
        buf[14..16].copy_from_slice(&next.to_le_bytes());
        mem.write_slice(&buf, GuestAddress(off)).unwrap();
    }
    fn publish_avail(mem: &GuestMemoryMmap, avail: u64, head: u16) {
        mem.write_slice(&head.to_le_bytes(), GuestAddress(avail + 4))
            .unwrap();
        mem.write_slice(&1u16.to_le_bytes(), GuestAddress(avail + 2))
            .unwrap();
    }

    #[test]
    fn config_identity_and_capability_chain() {
        let mem = test_mem();
        let (pci, _c) = build(&mem);
        // Standard header identity.
        assert_eq!(cfg16(&pci, REG_VENDOR_ID), VENDOR_ID);
        assert_eq!(cfg16(&pci, REG_DEVICE_ID), DEVICE_ID);
        assert_eq!(cfg8(&pci, REG_REVISION_ID), REVISION);
        assert_eq!(cfg8(&pci, REG_CLASS), CLASS_NETWORK);
        assert_eq!(cfg8(&pci, REG_SUBCLASS), SUBCLASS_ETHERNET);
        assert_eq!(cfg8(&pci, REG_INTERRUPT_PIN), INTERRUPT_PIN_INTA);
        assert_ne!(cfg16(&pci, REG_STATUS) & PCI_STATUS_CAP_LIST, 0);
        assert_eq!(cfg8(&pci, REG_CAP_PTR), CAP_COMMON as u8);

        // The four virtio vendor caps chained COMMON->ISR->DEVICE->NOTIFY, then
        // NOTIFY->MSIX->0 (MSI-X is the new last cap; its layout is checked below
        // — it is not a virtio vendor cap so it is not in this loop).
        let chain = [
            (CAP_COMMON, CAP_ISR as u8, CFG_TYPE_COMMON, CAP_LEN_STD, COMMON_OFFSET),
            (CAP_ISR, CAP_DEVICE as u8, CFG_TYPE_ISR, CAP_LEN_STD, ISR_OFFSET),
            (CAP_DEVICE, CAP_NOTIFY as u8, CFG_TYPE_DEVICE, CAP_LEN_STD, DEVICE_OFFSET),
            (CAP_NOTIFY, CAP_MSIX as u8, CFG_TYPE_NOTIFY, CAP_LEN_NOTIFY, NOTIFY_OFFSET),
        ];
        for (at, next, cfg_type, len, region_off) in chain {
            assert_eq!(cfg8(&pci, at + CAP_OFF_VNDR), CAP_VNDR, "cap@{at:#x} vndr");
            assert_eq!(cfg8(&pci, at + CAP_OFF_NEXT), next, "cap@{at:#x} next");
            assert_eq!(cfg8(&pci, at + CAP_OFF_LEN), len, "cap@{at:#x} len");
            assert_eq!(cfg8(&pci, at + CAP_OFF_CFG_TYPE), cfg_type, "cap@{at:#x} cfg_type");
            assert_eq!(cfg8(&pci, at + CAP_OFF_BAR), 0, "cap@{at:#x} bar");
            assert_eq!(
                cfg32(&pci, at + CAP_OFF_OFFSET),
                region_off as u32,
                "cap@{at:#x} offset"
            );
            assert_eq!(
                cfg32(&pci, at + CAP_OFF_LENGTH),
                REGION_SIZE as u32,
                "cap@{at:#x} length"
            );
        }
        // NOTIFY carries the multiplier; queue_notify_off(i)==i so the wire
        // address is i*4.
        assert_eq!(
            cfg32(&pci, CAP_NOTIFY + CAP_OFF_NOTIFY_MULT),
            NOTIFY_OFF_MULTIPLIER
        );
        // MSI-X cap (last in the chain): id, cap_next=0, Table Size (N-1
        // encoding in bits[10:0]), and Table/PBA offset|BIR pointing into BAR0.
        assert_eq!(cfg8(&pci, CAP_MSIX), PCI_CAP_ID_MSIX, "msix cap id");
        assert_eq!(cfg8(&pci, CAP_MSIX + 1), 0, "msix cap_next (last cap)");
        assert_eq!(
            cfg16(&pci, CAP_MSIX + MSIX_OFF_MSG_CTRL) & 0x07FF,
            MSIX_TABLE_SIZE - 1,
            "msix table size (N-1)"
        );
        assert_eq!(
            cfg32(&pci, CAP_MSIX + MSIX_OFF_TABLE),
            MSIX_TABLE_OFFSET as u32 | MSIX_BIR0,
            "msix table offset|BIR"
        );
        assert_eq!(
            cfg32(&pci, CAP_MSIX + MSIX_OFF_PBA),
            MSIX_PBA_OFFSET as u32 | MSIX_BIR0,
            "msix PBA offset|BIR"
        );
    }

    /// MSI-X vector latches echo the guest-written vector verbatim (the -EBUSY
    /// gate: the kernel's vp_modern writes a vector, reads it back, and aborts
    /// if they differ). Config + per-queue vectors are independent latches.
    #[test]
    fn msix_vector_echo_readback() {
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        assert_eq!(cc_r(&mut pci, CC_MSIX_CONFIG) as u16, NO_VECTOR);
        cc_w(&mut pci, CC_MSIX_CONFIG, 0);
        assert_eq!(cc_r(&mut pci, CC_MSIX_CONFIG) as u16, 0);
        // Per-queue vector latches against the selected queue.
        cc_w(&mut pci, CC_QUEUE_SELECT, TXQ as u32);
        cc_w(&mut pci, CC_QUEUE_MSIX_VECTOR, 1);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_MSIX_VECTOR) as u16, 1);
        cc_w(&mut pci, CC_QUEUE_SELECT, RXQ as u32);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_MSIX_VECTOR) as u16, NO_VECTOR);
        cc_w(&mut pci, CC_QUEUE_MSIX_VECTOR, 0);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_MSIX_VECTOR) as u16, 0);
        // The TX queue's vector is unaffected by the RX write.
        cc_w(&mut pci, CC_QUEUE_SELECT, TXQ as u32);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_MSIX_VECTOR) as u16, 1);
    }

    /// An out-of-range vector (>= table size, not NO_VECTOR) is rejected — stored
    /// as NO_VECTOR so the read-back signals rejection rather than letting a
    /// hostile vector index the table out of bounds.
    #[test]
    fn msix_vector_out_of_range_rejected() {
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        cc_w(&mut pci, CC_MSIX_CONFIG, MSIX_TABLE_SIZE as u32);
        assert_eq!(cc_r(&mut pci, CC_MSIX_CONFIG) as u16, NO_VECTOR);
        cc_w(&mut pci, CC_QUEUE_SELECT, TXQ as u32);
        cc_w(&mut pci, CC_QUEUE_MSIX_VECTOR, 0xABCD);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_MSIX_VECTOR) as u16, NO_VECTOR);
    }

    /// MSI-X table reads/writes round-trip through the BAR0 table region: each
    /// 16-byte entry's four dwords (addr lo/hi, msg data, vector control) echo;
    /// an out-of-range entry reads 0 and drops writes (no OOB).
    #[test]
    fn msix_table_roundtrip_and_bounds() {
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        let entry1 = MSIX_TABLE_OFFSET + MSIX_ENTRY_SIZE; // table entry index 1
        let words = [0x1111_0000u32, 0x0000_2222, 0x3333_3333, 0x0000_0001];
        for (i, w) in words.iter().enumerate() {
            pci.bar_write(entry1 + (i as u64) * 4, &w.to_le_bytes());
        }
        for (i, w) in words.iter().enumerate() {
            let mut b = [0u8; 4];
            pci.bar_read(entry1 + (i as u64) * 4, &mut b);
            assert_eq!(u32::from_le_bytes(b), *w, "table entry1 dword{i}");
        }
        let oob = MSIX_TABLE_OFFSET + MSIX_TABLE_SIZE as u64 * MSIX_ENTRY_SIZE;
        pci.bar_write(oob, &0xDEAD_BEEFu32.to_le_bytes());
        let mut b = [0u8; 4];
        pci.bar_read(oob, &mut b);
        assert_eq!(u32::from_le_bytes(b), 0, "out-of-range table entry reads 0");
    }

    /// The MSI-X PBA reads 0 (no pending vectors) and drops writes (read-only).
    #[test]
    fn msix_pba_reads_zero_and_read_only() {
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        pci.bar_write(MSIX_PBA_OFFSET, &0xFFu32.to_le_bytes());
        let mut b = [0u8; 4];
        pci.bar_read(MSIX_PBA_OFFSET, &mut b);
        assert_eq!(u32::from_le_bytes(b), 0, "PBA reads 0 (no pending); write ignored");
    }

    // ---- MSI-X delivery (route install + fire/pend/replay through the facade) ----

    /// Build a PCI function with MSI-X wired: retained per-vector eventfds (so a
    /// test can drain them) + a retained mock route sink. MSI-X is NOT yet
    /// enabled. Returns (pci, counters, shared msix, [eventfds], mock sink).
    #[allow(clippy::type_complexity)]
    fn build_msix(
        mem: &GuestMemoryMmap,
    ) -> (
        VirtioNetPci,
        Arc<super::super::VirtioNetCounters>,
        Arc<PiMutex<MsixState>>,
        [EventFd; MSIX_VECTORS],
        Arc<MockRouteSink>,
    ) {
        let mut net = VirtioNet::new(NetConfig::default());
        net.set_mem(mem.clone());
        let counters = net.counters();
        let msix = Arc::new(PiMutex::new(MsixState::new()));
        // Hand the state a clone of each eventfd; keep the originals to drain.
        let evts: [EventFd; MSIX_VECTORS] =
            std::array::from_fn(|_| EventFd::new(libc::EFD_NONBLOCK).unwrap());
        for (v, e) in evts.iter().enumerate() {
            msix.lock().set_eventfd(v, e.try_clone().unwrap());
        }
        let sink = Arc::new(MockRouteSink::default());
        let pci = VirtioNetPci::new(
            net,
            TEST_BAR_APERTURE,
            Arc::clone(&msix),
            Some(Arc::clone(&sink) as Arc<dyn MsixRouteSink>),
            test_gsis(),
        );
        (pci, counters, msix, evts, sink)
    }

    /// Message Control values: Enable (bit15), Function Mask / MASKALL (bit14).
    const MC_ENABLE: u16 = 0x8000;
    const MC_ENABLE_MASKALL: u16 = 0x8000 | 0x4000;

    /// Enable MSI-X in SHARED mode the way Linux's `msix_capability_init` does:
    /// assign config→vector 0 and all queues→vector 1, write Message Control
    /// Enable+MASKALL FIRST, program each entry's addr/data and clear its
    /// per-vector mask bit (all while function-masked, so no route installs yet),
    /// then clear MASKALL — the function-unmask edge that installs every unmasked
    /// vector. Exercising the real MASKALL sequence means the facade's
    /// function-unmask reconcile edge (not just the cold-enable edge) is covered.
    fn enable_msix_shared(pci: &mut VirtioNetPci) {
        cc_w(pci, CC_MSIX_CONFIG, 0);
        for q in [RXQ, TXQ] {
            cc_w(pci, CC_QUEUE_SELECT, q as u32);
            cc_w(pci, CC_QUEUE_MSIX_VECTOR, 1);
        }
        pci.config_write(CAP_MSIX + MSIX_OFF_MSG_CTRL, &MC_ENABLE_MASKALL.to_le_bytes());
        for v in 0..MSIX_VECTORS {
            let base = MSIX_TABLE_OFFSET + (v as u64) * MSIX_ENTRY_SIZE;
            pci.bar_write(base, &0xFEE0_0000u32.to_le_bytes()); // addr_lo
            pci.bar_write(base + 8, &(0x4000u32 + v as u32).to_le_bytes()); // msg data
            pci.bar_write(base + 12, &0u32.to_le_bytes()); // per-vector unmask (still fn-masked)
        }
        // Clear MASKALL → function-unmask edge installs every unmasked vector.
        pci.config_write(CAP_MSIX + MSIX_OFF_MSG_CTRL, &MC_ENABLE.to_le_bytes());
    }

    /// Lay out one TX chain (12-byte zero header + `payload`) at ring slot 0 plus
    /// an RX buffer, then kick TX through the notify region.
    fn kick_tx_loopback(pci: &mut VirtioNetPci, mem: &GuestMemoryMmap, payload: &[u8]) {
        let zero_hdr = [0u8; VIRTIO_NET_HDR_LEN];
        mem.write_slice(&zero_hdr, GuestAddress(TX_BUF)).unwrap();
        mem.write_slice(payload, GuestAddress(TX_BUF + VIRTIO_NET_HDR_LEN as u64))
            .unwrap();
        write_desc(
            mem,
            TX_DESC,
            0,
            TX_BUF,
            (VIRTIO_NET_HDR_LEN + payload.len()) as u32,
            0,
            0,
        );
        publish_avail(mem, TX_AVAIL, 0);
        write_desc(mem, RX_DESC, 0, RX_BUF, 256, 2, 0); // F_WRITE
        publish_avail(mem, RX_AVAIL, 0);
        pci.bar_write(
            NOTIFY_OFFSET + (TXQ as u64) * NOTIFY_OFF_MULTIPLIER as u64,
            &[0u8; 2],
        );
    }

    /// With MSI-X enabled + both vectors unmasked, a TX loopback delivers the
    /// VRING interrupt to the shared queue vector (1) via its eventfd — NOT the
    /// config vector (0), NOT the INTx ISR — and the routes were installed.
    #[test]
    fn msix_enabled_delivers_vring_to_queue_vector() {
        let mem = test_mem();
        let (mut pci, counters, msix, evts, sink) = build_msix(&mem);
        drive_to_features_ok(&mut pci);
        program_queue(&mut pci, RXQ as u32, RX_DESC, RX_AVAIL, RX_USED);
        program_queue(&mut pci, TXQ as u32, TX_DESC, TX_AVAIL, TX_USED);
        enable_msix_shared(&mut pci);
        cc_w(&mut pci, CC_DEVICE_STATUS, S_OK);

        kick_tx_loopback(&mut pci, &mem, &(0..42u8).collect::<Vec<_>>());

        assert_eq!(counters.tx_packets(), 1, "TX drained");
        assert_eq!(counters.rx_packets(), 1, "RX loopback delivered");
        // VRING delivered to the shared queue vector (1), config vector (0) quiet.
        assert_eq!(evts[1].read().unwrap(), 1, "queue vector 1 fired once");
        assert!(evts[0].read().is_err(), "config vector 0 untouched (EAGAIN)");
        // MSI-X mode does NOT touch the INTx ISR (the kernel's per-vector handlers
        // never read it).
        let mut isr = [0xFFu8; 1];
        pci.bar_read(ISR_OFFSET, &mut isr);
        assert_eq!(isr[0], 0, "MSI-X mode leaves the INTx ISR clear");
        // Both unmasked vectors had their routes installed at their GSIs (24/25).
        let installs = sink.installs.lock().unwrap();
        assert!(
            installs.iter().any(|(g, m)| *g == 24 && m.is_some()),
            "vector 0 route installed at GSI 24"
        );
        assert!(
            installs.iter().any(|(g, m)| *g == 25 && m.is_some()),
            "vector 1 route installed at GSI 25"
        );
        assert_eq!(msix.lock().pba_byte(), 0, "delivered live, nothing pending");
    }

    /// A VRING interrupt to a MASKED vector records a PBA bit (no eventfd write);
    /// unmasking the vector replays the pending interrupt exactly once.
    #[test]
    fn msix_masked_vector_pends_then_replays_on_unmask() {
        let mem = test_mem();
        let (mut pci, _counters, msix, evts, _sink) = build_msix(&mem);
        drive_to_features_ok(&mut pci);
        program_queue(&mut pci, RXQ as u32, RX_DESC, RX_AVAIL, RX_USED);
        program_queue(&mut pci, TXQ as u32, TX_DESC, TX_AVAIL, TX_USED);
        enable_msix_shared(&mut pci);
        cc_w(&mut pci, CC_DEVICE_STATUS, S_OK);
        // Re-mask the queue vector (entry 1, vector control bit0 = 1).
        pci.bar_write(MSIX_TABLE_OFFSET + MSIX_ENTRY_SIZE + 12, &1u32.to_le_bytes());

        kick_tx_loopback(&mut pci, &mem, &[0x5Au8; 20]);

        assert!(evts[1].read().is_err(), "masked vector did not fire (EAGAIN)");
        assert_eq!(
            msix.lock().pba_byte() & (1 << 1),
            1 << 1,
            "pending bit set for the masked queue vector"
        );
        // Unmask → the facade replays the pending interrupt once.
        pci.bar_write(MSIX_TABLE_OFFSET + MSIX_ENTRY_SIZE + 12, &0u32.to_le_bytes());
        assert_eq!(evts[1].read().unwrap(), 1, "unmask replays the pending interrupt");
        assert_eq!(
            msix.lock().pba_byte() & (1 << 1),
            0,
            "pending cleared after replay"
        );
    }

    /// Without a route sink (no route owner — e.g. non-PCI) the MSI-X cap is NOT
    /// advertised — the guest never sees it and stays on INTx, so the device
    /// never offers a delivery path it cannot honor.
    #[test]
    fn msix_cap_absent_without_route_sink() {
        let mem = test_mem();
        let mut net = VirtioNet::new(NetConfig::default());
        net.set_mem(mem.clone());
        let msix = Arc::new(PiMutex::new(MsixState::new()));
        let pci = VirtioNetPci::new(net, TEST_BAR_APERTURE, msix, None, [0; MSIX_VECTORS]);
        // NOTIFY is the last cap (cap_next = 0) — the MSI-X cap is not chained.
        assert_eq!(
            cfg8(&pci, CAP_NOTIFY + CAP_OFF_NEXT),
            0,
            "NOTIFY terminates the chain when MSI-X is not offered"
        );
        // The MSI-X cap slot is unpopulated (id byte stays 0, not PCI_CAP_ID_MSIX).
        assert_eq!(
            cfg8(&pci, CAP_MSIX),
            0,
            "no MSI-X cap written without a route sink"
        );
        // STATUS still advertises a capability list (the virtio vendor caps).
        assert_ne!(cfg16(&pci, REG_STATUS) & PCI_STATUS_CAP_LIST, 0);
    }

    /// A queue poison (hostile avail.idx) under MSI-X delivers to the CONFIG
    /// vector (0), not the queue vector (1), and still sets NEEDS_RESET — the
    /// MSI-X twin of the INTx tests_poison path (signal_queue_poisoned →
    /// IrqSource::Config). Pins the config-vector branch of the MSI-X signal path
    /// (only the VRING branch is covered by msix_enabled_delivers_…).
    #[test]
    fn msix_config_vector_delivers_on_queue_poison() {
        let mem = test_mem();
        let (mut pci, _counters, _msix, evts, _sink) = build_msix(&mem);
        drive_to_features_ok(&mut pci);
        program_queue(&mut pci, RXQ as u32, RX_DESC, RX_AVAIL, RX_USED);
        program_queue(&mut pci, TXQ as u32, TX_DESC, TX_AVAIL, TX_USED);
        enable_msix_shared(&mut pci);
        cc_w(&mut pci, CC_DEVICE_STATUS, S_OK);

        // Plant a bogus TX avail.idx far beyond the (size-4) ring → the loopback
        // poisons the queue → signal_queue_poisoned → MSI-X config vector (0).
        mem.write_slice(&1000u16.to_le_bytes(), GuestAddress(TX_AVAIL + 2))
            .unwrap();
        pci.bar_write(
            NOTIFY_OFFSET + (TXQ as u64) * NOTIFY_OFF_MULTIPLIER as u64,
            &[0u8; 2],
        );

        assert_eq!(evts[0].read().unwrap(), 1, "poison delivered to config vector 0");
        assert!(
            evts[1].read().is_err(),
            "queue vector 1 not fired by a config-change/poison"
        );
        // NEEDS_RESET is set (guest-visible), and MSI-X mode leaves the INTx ISR
        // clear (the bit-set path is skipped on the MSI-X branch).
        let mut sb = [0u8; 1];
        pci.bar_read(COMMON_OFFSET + CC_DEVICE_STATUS, &mut sb);
        assert_ne!(
            sb[0] as u32 & VIRTIO_CONFIG_S_NEEDS_RESET,
            0,
            "poison sets NEEDS_RESET"
        );
        let mut isr = [0xFFu8; 1];
        pci.bar_read(ISR_OFFSET, &mut isr);
        assert_eq!(isr[0], 0, "MSI-X mode leaves the INTx ISR clear on poison");
    }

    /// A virtio reset (device_status = 0) restores the MSI-X vector assignments
    /// to NO_VECTOR (virtio-v1.2 §4.1.4.3) while the PCI-level cap (enable/table)
    /// persists — `reset_virtio_assignments` via the shared state.
    #[test]
    fn reset_clears_msix_vector_assignments() {
        let mem = test_mem();
        let (mut pci, _counters, msix, _evts, _sink) = build_msix(&mem);
        drive_to_features_ok(&mut pci);
        enable_msix_shared(&mut pci);
        assert_eq!(msix.lock().config_vector(), 0, "config vector assigned");
        assert_eq!(msix.lock().queue_vector(TXQ), 1, "queue vector assigned");

        cc_w(&mut pci, CC_DEVICE_STATUS, 0); // virtio reset

        assert_eq!(
            msix.lock().config_vector(),
            NO_VECTOR,
            "reset restores config vector to NO_VECTOR"
        );
        assert_eq!(msix.lock().queue_vector(RXQ), NO_VECTOR, "reset restores RX vector");
        assert_eq!(msix.lock().queue_vector(TXQ), NO_VECTOR, "reset restores TX vector");
    }

    /// Routes install on the FUNCTION-unmask (MASKALL-clear) edge: enable with
    /// MASKALL + program + per-vector-unmask installs NOTHING (function-masked),
    /// and clearing MASKALL then installs every unmasked vector. Pins the
    /// function-unmask half of the facade reconcile (the per-vector-unmask edge
    /// is covered elsewhere; this is the edge the real kernel MASKALL sequence
    /// hits).
    #[test]
    fn msix_routes_install_on_function_unmask_edge() {
        let mem = test_mem();
        let (mut pci, _counters, _msix, _evts, sink) = build_msix(&mem);
        drive_to_features_ok(&mut pci);
        program_queue(&mut pci, RXQ as u32, RX_DESC, RX_AVAIL, RX_USED);
        program_queue(&mut pci, TXQ as u32, TX_DESC, TX_AVAIL, TX_USED);
        cc_w(&mut pci, CC_MSIX_CONFIG, 0);
        for q in [RXQ, TXQ] {
            cc_w(&mut pci, CC_QUEUE_SELECT, q as u32);
            cc_w(&mut pci, CC_QUEUE_MSIX_VECTOR, 1);
        }
        // Enable + MASKALL, then program + per-vector unmask — all function-masked.
        pci.config_write(CAP_MSIX + MSIX_OFF_MSG_CTRL, &MC_ENABLE_MASKALL.to_le_bytes());
        for v in 0..MSIX_VECTORS {
            let base = MSIX_TABLE_OFFSET + (v as u64) * MSIX_ENTRY_SIZE;
            pci.bar_write(base, &0xFEE0_0000u32.to_le_bytes());
            pci.bar_write(base + 8, &(0x4000u32 + v as u32).to_le_bytes());
            pci.bar_write(base + 12, &0u32.to_le_bytes());
        }
        assert!(
            !sink.installs.lock().unwrap().iter().any(|(_, m)| m.is_some()),
            "no route installed while function-masked (MASKALL set)"
        );
        // Clear MASKALL → function-unmask edge installs both unmasked vectors.
        pci.config_write(CAP_MSIX + MSIX_OFF_MSG_CTRL, &MC_ENABLE.to_le_bytes());
        let installs = sink.installs.lock().unwrap();
        assert!(
            installs.iter().any(|(g, m)| *g == 24 && m.is_some()),
            "vector 0 route installed on MASKALL-clear"
        );
        assert!(
            installs.iter().any(|(g, m)| *g == 25 && m.is_some()),
            "vector 1 route installed on MASKALL-clear"
        );
    }

    /// The route REMOVAL contract: after a vector's route is installed,
    /// re-masking it (Vector Control bit0 = 1) must emit a `set_route(gsi, None)`
    /// removal so the masked vector has no live KVM route — masking enforced at
    /// the KVM layer (matching firecracker), not only the device-side PBA gate.
    /// Direct regression pin for the remove branch of `reconcile_route`.
    #[test]
    fn msix_mask_edge_removes_route() {
        let mem = test_mem();
        let (mut pci, _counters, _msix, _evts, sink) = build_msix(&mem);
        drive_to_features_ok(&mut pci);
        program_queue(&mut pci, RXQ as u32, RX_DESC, RX_AVAIL, RX_USED);
        program_queue(&mut pci, TXQ as u32, TX_DESC, TX_AVAIL, TX_USED);
        enable_msix_shared(&mut pci);
        // The shared queue vector (1) is installed at GSI 25 after enable.
        assert!(
            sink.installs.lock().unwrap().iter().any(|(g, m)| *g == 25 && m.is_some()),
            "queue vector route installed after enable"
        );
        // Re-mask the queue vector (entry 1, Vector Control bit0 = 1) → its KVM
        // route must be removed.
        pci.bar_write(MSIX_TABLE_OFFSET + MSIX_ENTRY_SIZE + 12, &1u32.to_le_bytes());
        assert!(
            sink.installs.lock().unwrap().iter().any(|(g, m)| *g == 25 && m.is_none()),
            "mask edge removes the queue vector's route via set_route(gsi, None)"
        );
    }

    #[test]
    fn bar_window_gated_on_memory_enable_and_nonzero_base() {
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        // Unprogrammed: no window.
        assert_eq!(pci.bar_window(), None, "no window before BAR program");
        // BAR base set but COMMAND.MEMORY clear: still no window.
        pci.config_write(REG_BAR0, &0xe010_0000u32.to_le_bytes());
        assert_eq!(pci.bar_window(), None, "no window until memory-space enabled");
        // COMMAND.MEMORY set but base zero: no window.
        let mut pci0 = build(&mem).0;
        pci0.config_write(REG_COMMAND, &PCI_COMMAND_MEMORY.to_le_bytes());
        assert_eq!(pci0.bar_window(), None, "no window with zero base");
        // Both, in-grant base: window published, type bits masked off.
        pci.config_write(REG_COMMAND, &PCI_COMMAND_MEMORY.to_le_bytes());
        assert_eq!(pci.bar_window(), Some((TEST_BAR_BASE as u64, BAR0_SIZE)));
        // The low type nibble is read-only (size-probe recovers BAR0_SIZE).
        pci.config_write(REG_BAR0, &0xffff_ffffu32.to_le_bytes());
        let probed = cfg32(&pci, REG_BAR0);
        assert_eq!(probed & !(BAR0_SIZE as u32 - 1), !(BAR0_SIZE as u32 - 1));
        assert_eq!(probed & 0xF, BAR0_TYPE_BITS, "type bits read-only");
    }

    #[test]
    fn bar_window_rejects_base_outside_crs_grant() {
        // A non-conformant guest programs BAR0 over the ECAM window (just below
        // the _CRS MMIO grant) with COMMAND.MEMORY set. bar_window must reject
        // the out-of-grant base (return None) so the BAR cannot shadow ECAM —
        // the MMIO dispatch checks bar_mmio_contains BEFORE ecam_contains, so an
        // honored window here would steal config accesses.
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        pci.config_write(REG_COMMAND, &PCI_COMMAND_MEMORY.to_le_bytes());
        pci.config_write(REG_BAR0, &TEST_ECAM_BASE.to_le_bytes());
        assert_eq!(
            pci.bar_window(),
            None,
            "BAR base below the _CRS grant (over ECAM) must claim no window"
        );
        // A base whose tail spills past the grant end is also rejected.
        let past_end = (TEST_BAR_APERTURE.1 as u32) & !(BAR0_SIZE as u32 - 1);
        pci.config_write(REG_BAR0, &past_end.to_le_bytes());
        assert_eq!(
            pci.bar_window(),
            None,
            "BAR window extending past the grant end must claim no window"
        );
        // The in-grant base is still honored (regression guard for the bound).
        pci.config_write(REG_BAR0, &TEST_BAR_BASE.to_le_bytes());
        assert_eq!(pci.bar_window(), Some((TEST_BAR_BASE as u64, BAR0_SIZE)));
    }

    /// The cached BAR window refreshes on every REG_COMMAND / REG_BAR0
    /// write — a guest memory-enable toggle or BAR reprogram is reflected
    /// immediately, never stale. The load-bearing assertion is the reprogram:
    /// after a window is established, a BAR0 rewrite must refresh the cache (a
    /// missed refresh would return the prior window).
    #[test]
    fn bar_window_cache_refreshes_on_command_and_bar0_writes() {
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        assert_eq!(pci.bar_window(), None, "reset: no cached window");
        // BAR0 base alone (memory decode still off) → cache refreshed, still None.
        pci.config_write(REG_BAR0, &TEST_BAR_BASE.to_le_bytes());
        assert_eq!(
            pci.bar_window(),
            None,
            "BAR0 write refreshes the cache; memory decode off → still None"
        );
        // Memory-enable write refreshes the cache → window appears.
        pci.config_write(REG_COMMAND, &PCI_COMMAND_MEMORY.to_le_bytes());
        assert_eq!(
            pci.bar_window(),
            Some((TEST_BAR_BASE as u64, BAR0_SIZE)),
            "COMMAND.MEMORY write refreshes the cache → window appears"
        );
        // Reprogram BAR0 out-of-grant → cache refreshes to None, NOT the stale
        // in-grant window. This is the regression a missed refresh would fail
        // (it would return the prior Some(TEST_BAR_BASE)).
        pci.config_write(REG_BAR0, &TEST_ECAM_BASE.to_le_bytes());
        assert_eq!(
            pci.bar_window(),
            None,
            "BAR0 reprogram (out-of-grant) refreshes the cache → None, not stale"
        );
        // Reprogram back in-grant → cache refreshes to the restored window.
        pci.config_write(REG_BAR0, &TEST_BAR_BASE.to_le_bytes());
        assert_eq!(
            pci.bar_window(),
            Some((TEST_BAR_BASE as u64, BAR0_SIZE)),
            "BAR0 reprogram (in-grant) refreshes the cache → window restored"
        );
        // Clear COMMAND.MEMORY → cache refreshes to None.
        pci.config_write(REG_COMMAND, &0u16.to_le_bytes());
        assert_eq!(
            pci.bar_window(),
            None,
            "clearing COMMAND.MEMORY refreshes the cache → no window"
        );
    }

    #[test]
    fn common_cfg_register_reads() {
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        assert_eq!(cc_r(&mut pci, CC_NUM_QUEUES), NUM_QUEUES as u32);
        // MSI-X vectors are unassigned at reset → read NO_VECTOR (the echo /
        // latch behaviour is covered by msix_vector_echo_readback).
        assert_eq!(cc_r(&mut pci, CC_MSIX_CONFIG) as u16, NO_VECTOR);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_MSIX_VECTOR) as u16, NO_VECTOR);
        // device_feature select+window: VERSION_1 in the high word, MAC in low.
        cc_w(&mut pci, CC_DEVICE_FEATURE_SELECT, 0);
        assert_eq!(cc_r(&mut pci, CC_DEVICE_FEATURE_SELECT), 0);
        assert_ne!(cc_r(&mut pci, CC_DEVICE_FEATURE) & (1 << VIRTIO_NET_F_MAC), 0);
        cc_w(&mut pci, CC_DEVICE_FEATURE_SELECT, 1);
        assert_ne!(
            cc_r(&mut pci, CC_DEVICE_FEATURE) & (1 << (VIRTIO_F_VERSION_1 - 32)),
            0
        );
        // Out-of-range device_feature_select reads back 0 (only windows 0/1
        // exist): a hostile guest latching an arbitrary u32 must read a defined
        // 0, never index OOB. Mirrors the driver_feature_select gating.
        cc_w(&mut pci, CC_DEVICE_FEATURE_SELECT, 2);
        assert_eq!(cc_r(&mut pci, CC_DEVICE_FEATURE), 0, "sel=2 window reads 0");
        cc_w(&mut pci, CC_DEVICE_FEATURE_SELECT, 0xFFFF_FFFF);
        assert_eq!(cc_r(&mut pci, CC_DEVICE_FEATURE), 0, "sel=u32::MAX reads 0");
        // device_feature @0x04 is read-only (offered features): a guest write
        // must NOT alter the offered-feature window read-back. Locks the
        // read-only contract against a future match-arm edit that mistook
        // CC_DEVICE_FEATURE for writable.
        cc_w(&mut pci, CC_DEVICE_FEATURE_SELECT, 0);
        let offered = cc_r(&mut pci, CC_DEVICE_FEATURE);
        cc_w(&mut pci, CC_DEVICE_FEATURE, 0xDEAD_BEEF);
        assert_eq!(
            cc_r(&mut pci, CC_DEVICE_FEATURE),
            offered,
            "device_feature is read-only; the write must be dropped"
        );
        // Write-only / device-read registers serve 0 on read-back (the catch-all
        // arm): driver_feature is write-only, and the guest owns the ring
        // addresses so the device never reflects them. Locks the read-0 contract
        // against a future match-arm edit that accidentally serves a stale value.
        assert_eq!(cc_r(&mut pci, CC_DRIVER_FEATURE), 0, "driver_feature reads 0");
        assert_eq!(cc_r(&mut pci, CC_QUEUE_DESC_LO), 0, "queue_desc_lo reads 0");
        assert_eq!(cc_r(&mut pci, CC_QUEUE_DESC_HI), 0, "queue_desc_hi reads 0");
        // queue_select read-back, and queue_notify_off == queue_select.
        cc_w(&mut pci, CC_QUEUE_SELECT, TXQ as u32);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_SELECT), TXQ as u32);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_NOTIFY_OFF), TXQ as u32);
        // Out-of-range queue selector: queue_size / queue_enable / notify_off
        // all read the clamped 0 the common_read doc promises (selected_queue()
        // bounds-checks before any indexed op), so a hostile latch reads defined
        // values and drives no op.
        cc_w(&mut pci, CC_QUEUE_SELECT, NUM_QUEUES as u32);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_SIZE), 0, "OOR selector: queue_size 0");
        assert_eq!(cc_r(&mut pci, CC_QUEUE_ENABLE), 0, "OOR selector: queue_enable 0");
        assert_eq!(cc_r(&mut pci, CC_QUEUE_NOTIFY_OFF), 0, "OOR selector: notify_off 0");
    }

    #[test]
    fn queue_size_reads_back_programmed_not_max() {
        // virtio-v1.2 §4.1.4.3: queue_size is R/W; it resets to max_size and
        // reads back the guest-written value. Pins the fix that serves the
        // configured size, not the immutable max.
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        drive_to_features_ok(&mut pci);
        cc_w(&mut pci, CC_QUEUE_SELECT, TXQ as u32);
        assert_eq!(
            cc_r(&mut pci, CC_QUEUE_SIZE),
            QUEUE_MAX_SIZE as u32,
            "resets to max before the guest writes",
        );
        cc_w(&mut pci, CC_QUEUE_SIZE, 8);
        assert_eq!(
            cc_r(&mut pci, CC_QUEUE_SIZE),
            8,
            "read-back returns the programmed size, not the max",
        );
    }

    #[test]
    fn common_cfg_ring_addresses_merge_low_and_high_dwords() {
        // virtio_pci_modern writes 64-bit ring addresses as two 32-bit halves
        // (the kernel's vp_iowrite64_twopart: LO then HI). Pin that the
        // facade's CC_QUEUE_{DESC,AVAIL,USED}_{LO,HI} arms merge them into the
        // correct dwords — a LO/HI swap would point the ring at the wrong GPA
        // (silent corruption), which get_u32 + the per-arm None/Some split must
        // not do. (`pci.net` is reachable: this test module is a descendant of
        // the pci module.)
        let mem = test_mem();
        let mut net = VirtioNet::new(NetConfig::default());
        net.set_mem(mem.clone());
        let mut pci = new_pci(net);
        drive_to_features_ok(&mut pci); // FEATURES_OK window: queue config allowed
        cc_w(&mut pci, CC_QUEUE_SELECT, TXQ as u32);
        cc_w(&mut pci, CC_QUEUE_DESC_LO, 0x0000_1000);
        cc_w(&mut pci, CC_QUEUE_DESC_HI, 0x0000_ABCD);
        cc_w(&mut pci, CC_QUEUE_AVAIL_LO, 0x0000_2000);
        cc_w(&mut pci, CC_QUEUE_AVAIL_HI, 0x0000_0001);
        cc_w(&mut pci, CC_QUEUE_USED_LO, 0x0000_3000);
        cc_w(&mut pci, CC_QUEUE_USED_HI, 0x0000_0002);
        let (desc, avail, used) = pci.net.selected_queue_ring_addrs().unwrap();
        assert_eq!(desc, 0x0000_ABCD_0000_1000, "DESC high dword in high 32 bits");
        assert_eq!(avail, 0x0000_0001_0000_2000, "AVAIL high dword in high 32 bits");
        assert_eq!(used, 0x0000_0002_0000_3000, "USED high dword in high 32 bits");
    }

    #[test]
    fn status_fsm_and_natural_width_reads() {
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        drive_to_features_ok(&mut pci);
        // 1-byte read of device_status @0x14 returns the status low byte; the
        // packed config_generation @0x15 is a distinct 1-byte register (the
        // driver reads each at its natural width).
        let mut sb = [0u8; 1];
        pci.bar_read(COMMON_OFFSET + CC_DEVICE_STATUS, &mut sb);
        assert_eq!(sb[0] as u32, S_FEAT & 0xFF);
        let mut generation = [0u8; 1];
        pci.bar_read(COMMON_OFFSET + CC_CONFIG_GENERATION, &mut generation);
        assert_eq!(generation[0], 0, "config_generation reads 0 (no config change)");
    }

    #[test]
    fn common_cfg_straddling_read_serves_base_register_zero_filling_overhang() {
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        // After init, device_status @0x14 is nonzero (S_FEAT).
        drive_to_features_ok(&mut pci);
        // A non-spec 4-byte read at num_queues @0x12 spans num_queues (0x12,
        // u16=2), device_status (0x14), and config_generation (0x15). The
        // offset-exact dispatch serves ONLY the base register (num_queues) and
        // zero-fills the overhang — so the nonzero device_status @0x14 must NOT
        // bleed into byte 2. Pins the straddle behavior the common_read doc
        // claims (the driver never issues such an access, but a future match
        // refactor that served adjacent bytes would silently break here).
        let mut wide = [0xAAu8; 4];
        pci.bar_read(COMMON_OFFSET + CC_NUM_QUEUES, &mut wide);
        assert_eq!(
            u32::from_le_bytes(wide),
            NUM_QUEUES as u32,
            "straddling read must serve only num_queues and zero-fill the \
             overhang; device_status @0x14 must not be served into byte 2"
        );
    }

    #[test]
    fn device_config_region_serves_mac_and_drops_writes() {
        let mac = [0x52, 0x54, 0x00, 0xAB, 0xCD, 0xEF];
        let mem = test_mem();
        let mut net = VirtioNet::new(NetConfig::default().mac(mac));
        net.set_mem(mem.clone());
        let mut pci = new_pci(net);
        let mut got = [0u8; 6];
        pci.bar_read(DEVICE_OFFSET, &mut got);
        assert_eq!(got, mac, "device-config region serves the MAC at offset 0");
        // Device-config writes are ignored.
        pci.bar_write(DEVICE_OFFSET, &[0xFF; 6]);
        let mut after = [0u8; 6];
        pci.bar_read(DEVICE_OFFSET, &mut after);
        assert_eq!(after, mac, "device-config writes must be dropped");
        // The notify region is write-only; reads return 0.
        let mut nb = [0xAAu8; 4];
        pci.bar_read(NOTIFY_OFFSET, &mut nb);
        assert_eq!(nb, [0u8; 4], "notify region reads return 0");
    }

    #[test]
    fn loopback_through_facade_sets_isr_and_read_clears() {
        let mem = test_mem();
        let (mut pci, counters) = build(&mem);
        drive_to_features_ok(&mut pci);
        program_queue(&mut pci, RXQ as u32, RX_DESC, RX_AVAIL, RX_USED);
        program_queue(&mut pci, TXQ as u32, TX_DESC, TX_AVAIL, TX_USED);
        cc_w(&mut pci, CC_DEVICE_STATUS, S_OK);

        // One TX chain (12-byte zero header + 42-byte payload) and one RX
        // buffer, laid out in guest memory directly.
        let payload: Vec<u8> = (0..42u8).collect();
        let zero_hdr = [0u8; VIRTIO_NET_HDR_LEN];
        mem.write_slice(&zero_hdr, GuestAddress(TX_BUF)).unwrap();
        mem.write_slice(&payload, GuestAddress(TX_BUF + VIRTIO_NET_HDR_LEN as u64))
            .unwrap();
        write_desc(
            &mem,
            TX_DESC,
            0,
            TX_BUF,
            (VIRTIO_NET_HDR_LEN + payload.len()) as u32,
            0,
            0,
        );
        publish_avail(&mem, TX_AVAIL, 0);
        write_desc(&mem, RX_DESC, 0, RX_BUF, 256, 2, 0); // F_WRITE
        publish_avail(&mem, RX_AVAIL, 0);

        // Kick TX through the NOTIFY region (off = TXQ * multiplier).
        pci.bar_write(
            NOTIFY_OFFSET + (TXQ as u64) * NOTIFY_OFF_MULTIPLIER as u64,
            &[0u8; 2],
        );

        assert_eq!(counters.tx_packets(), 1, "TX drained through the facade");
        assert_eq!(counters.rx_packets(), 1, "RX loopback delivered");
        // RX buffer holds the 12-byte header + echoed payload.
        let mut delivered = vec![0u8; VIRTIO_NET_HDR_LEN + payload.len()];
        mem.read_slice(&mut delivered, GuestAddress(RX_BUF)).unwrap();
        assert_eq!(&delivered[VIRTIO_NET_HDR_LEN..], payload.as_slice());

        // ISR reflects the queue interrupt (bit0) and is read-to-clear.
        let mut isr = [0u8; 1];
        pci.bar_read(ISR_OFFSET, &mut isr);
        assert_eq!(isr[0] & 0x1, 0x1, "ISR queue-interrupt bit set after drain");
        let mut isr2 = [0xFFu8; 1];
        pci.bar_read(ISR_OFFSET, &mut isr2);
        assert_eq!(isr2[0], 0, "ISR read-to-clear: second read is 0");
    }

    #[test]
    fn isr_non_base_in_region_read_does_not_clear() {
        // Pins the STRICTER-than-qemu exact-offset ISR gating: only a read at
        // ISR_OFFSET itself read-clears; a stray in-region read (ISR_OFFSET+4)
        // returns 0 via the catch-all and must NOT ack the pending interrupt.
        // Locks the narrowing against a future refactor that mirrors COMMON/
        // DEVICE's range arm (which would re-introduce qemu's any-offset clear).
        let mem = test_mem();
        let (mut pci, counters) = build(&mem);
        drive_to_features_ok(&mut pci);
        program_queue(&mut pci, RXQ as u32, RX_DESC, RX_AVAIL, RX_USED);
        program_queue(&mut pci, TXQ as u32, TX_DESC, TX_AVAIL, TX_USED);
        cc_w(&mut pci, CC_DEVICE_STATUS, S_OK);
        let zero_hdr = [0u8; VIRTIO_NET_HDR_LEN];
        mem.write_slice(&zero_hdr, GuestAddress(TX_BUF)).unwrap();
        write_desc(&mem, TX_DESC, 0, TX_BUF, VIRTIO_NET_HDR_LEN as u32, 0, 0);
        publish_avail(&mem, TX_AVAIL, 0);
        write_desc(&mem, RX_DESC, 0, RX_BUF, 256, 2, 0); // F_WRITE
        publish_avail(&mem, RX_AVAIL, 0);
        pci.bar_write(
            NOTIFY_OFFSET + (TXQ as u64) * NOTIFY_OFF_MULTIPLIER as u64,
            &[0u8; 2],
        );
        assert!(counters.tx_packets() >= 1, "TX drained, ISR should be pending");
        // A non-base in-region read returns 0 and does NOT clear.
        let mut stray = [0xFFu8; 1];
        pci.bar_read(ISR_OFFSET + 4, &mut stray);
        assert_eq!(stray[0], 0, "non-base ISR-region read returns 0");
        // The pending bit is intact: the base read still surfaces it.
        let mut isr = [0u8; 1];
        pci.bar_read(ISR_OFFSET, &mut isr);
        assert_eq!(
            isr[0] & 0x1,
            0x1,
            "stray ISR-region read must NOT have cleared the pending bit"
        );
    }

    #[test]
    fn two_consecutive_kicks_each_set_and_clear_isr() {
        // Pins the INTx serialization invariant the full-irqchip resample path
        // relies on (module doc): each kick's interrupt is consumed (the guest
        // read-to-clears the ISR) before the next assert. Two back-to-back TX
        // loopbacks must EACH raise the VRING ISR bit and clear on read — so no
        // assert is lost/coalesced in the single-TX-queue v0 model. (The
        // async/multiqueue hazard that would break this is the active resample
        // handler follow-up.)
        let mem = test_mem();
        let (mut pci, counters) = build(&mem);
        drive_to_features_ok(&mut pci);
        program_queue(&mut pci, RXQ as u32, RX_DESC, RX_AVAIL, RX_USED);
        program_queue(&mut pci, TXQ as u32, TX_DESC, TX_AVAIL, TX_USED);
        cc_w(&mut pci, CC_DEVICE_STATUS, S_OK);

        let hdr = [0u8; VIRTIO_NET_HDR_LEN];
        let payload = [0x5Au8; 20];
        let total = (VIRTIO_NET_HDR_LEN + payload.len()) as u32;

        for round in 0u16..2 {
            // Distinct TX/RX buffers per round; descriptor + avail entry at ring
            // slot `round`, avail.idx bumped to round+1.
            let tx_buf = TX_BUF + (round as u64) * 0x200;
            let rx_buf = RX_BUF + (round as u64) * 0x400;
            mem.write_slice(&hdr, GuestAddress(tx_buf)).unwrap();
            mem.write_slice(&payload, GuestAddress(tx_buf + VIRTIO_NET_HDR_LEN as u64))
                .unwrap();
            write_desc(&mem, TX_DESC, round, tx_buf, total, 0, 0);
            write_desc(&mem, RX_DESC, round, rx_buf, 256, 2, 0);
            for avail in [TX_AVAIL, RX_AVAIL] {
                mem.write_slice(&round.to_le_bytes(), GuestAddress(avail + 4 + (round as u64) * 2))
                    .unwrap();
                mem.write_slice(&(round + 1).to_le_bytes(), GuestAddress(avail + 2))
                    .unwrap();
            }

            pci.bar_write(
                NOTIFY_OFFSET + (TXQ as u64) * NOTIFY_OFF_MULTIPLIER as u64,
                &[0u8; 2],
            );

            let mut isr = [0u8; 1];
            pci.bar_read(ISR_OFFSET, &mut isr);
            assert_eq!(isr[0] & 0x1, 0x1, "round {round}: VRING ISR bit set after kick");
            let mut isr_again = [0xFFu8; 1];
            pci.bar_read(ISR_OFFSET, &mut isr_again);
            assert_eq!(isr_again[0], 0, "round {round}: ISR read-to-clear");
        }
        assert_eq!(counters.tx_packets(), 2, "both kicks each drained a TX packet");
        assert_eq!(counters.rx_packets(), 2, "both loopbacks each delivered to RX");
    }

    #[test]
    fn notify_out_of_range_index_is_dropped() {
        let mem = test_mem();
        let (mut pci, counters) = build(&mem);
        drive_to_features_ok(&mut pci);
        program_queue(&mut pci, RXQ as u32, RX_DESC, RX_AVAIL, RX_USED);
        program_queue(&mut pci, TXQ as u32, TX_DESC, TX_AVAIL, TX_USED);
        cc_w(&mut pci, CC_DEVICE_STATUS, S_OK);
        // A notify to a wildly out-of-range queue index (top of the 4 KiB
        // notify region) must be a no-op, not a panic and not a TX drain.
        pci.bar_write(NOTIFY_OFFSET + REGION_SIZE - 4, &[0u8; 2]);
        assert_eq!(counters.tx_packets(), 0, "out-of-range notify drains nothing");
    }

    fn read_used_idx(mem: &GuestMemoryMmap, used_base: u64) -> u16 {
        // Used-ring layout: flags u16 | idx u16 | ring[...]; +2 skips flags.
        let mut b = [0u8; 2];
        mem.read_slice(&mut b, GuestAddress(used_base + 2)).unwrap();
        u16::from_le_bytes(b)
    }

    /// Sum the forward-progress event counters (byte aggregates excluded to
    /// avoid double-counting with the packet counters) — the same taxonomy the
    /// MMIO proptest snapshots.
    fn progress_sum(c: &super::super::VirtioNetCounters) -> u64 {
        c.tx_packets.load(Ordering::Relaxed)
            + c.rx_packets.load(Ordering::Relaxed)
            + c.tx_chain_invalid.load(Ordering::Relaxed)
            + c.tx_oversize_dropped.load(Ordering::Relaxed)
            + c.rx_chain_invalid.load(Ordering::Relaxed)
            + c.rx_write_failed.load(Ordering::Relaxed)
            + c.tx_dropped_no_rx_buffer.load(Ordering::Relaxed)
            + c.tx_dropped_rx_poisoned.load(Ordering::Relaxed)
            + c.tx_add_used_failures.load(Ordering::Relaxed)
            + c.rx_add_used_failures.load(Ordering::Relaxed)
            + c.invalid_avail_idx_count.load(Ordering::Relaxed)
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 256, max_shrink_iters: 1024, ..ProptestConfig::default() })]

        /// PCI-transport twin of the MMIO `tx_chain_progress_under_random_descriptors`
        /// proptest. The mandated chain-mutation fuzz is driven ENTIRELY through
        /// the facade's common-cfg/notify BAR decode (status FSM + queue
        /// programming via `bar_write(COMMON_OFFSET+cc)`, kick via
        /// `bar_write(NOTIFY_OFFSET + i*mult)`), feeding random TX descriptor
        /// chains (random addr/len/flags/next, ~1:15 mapped:unmapped) against a
        /// well-formed RX buffer. Asserts the hostile-input contract: no panic
        /// (proptest catches it), counter monotonicity, and forward progress
        /// (TX/RX used.idx advance OR an event counter bump) — proving the PCI
        /// decode reaches the shared transport-neutral core identically to the
        /// MMIO path, which is the only path x86 NICs actually take.
        #[test]
        fn facade_tx_chain_progress_under_random_descriptors(
            descs in prop::collection::vec(
                (0u64..(1u64 << 24), 0u32..(8 * 1024 * 1024), 0u16..8, any::<u16>()),
                1..=8usize,
            ),
        ) {
            let mem = test_mem();
            let mut net = VirtioNet::new(NetConfig::default());
            net.set_mem(mem.clone());
            let counters = net.counters();
            let mut pci = new_pci(net);
            drive_to_features_ok(&mut pci);
            program_queue(&mut pci, RXQ as u32, RX_DESC, RX_AVAIL, RX_USED);
            program_queue(&mut pci, TXQ as u32, TX_DESC, TX_AVAIL, TX_USED);
            cc_w(&mut pci, CC_DEVICE_STATUS, S_OK);

            for (i, (addr, len, flags, next)) in descs.iter().enumerate() {
                write_desc(&mem, TX_DESC, i as u16, *addr, *len, *flags, *next);
            }
            publish_avail(&mem, TX_AVAIL, 0);
            write_desc(&mem, RX_DESC, 0, RX_BUF, 2048, 2, 0); // F_WRITE, well-formed
            publish_avail(&mem, RX_AVAIL, 0);

            let before_tx = read_used_idx(&mem, TX_USED);
            let before_rx = read_used_idx(&mem, RX_USED);
            let before = progress_sum(&counters);

            // The PCI decode path under test: a notify-region write decodes to
            // the TX queue index and drives process_tx_loopback in the core.
            pci.bar_write(
                NOTIFY_OFFSET + (TXQ as u64) * NOTIFY_OFF_MULTIPLIER as u64,
                &[0u8; 2],
            );

            let after_tx = read_used_idx(&mem, TX_USED);
            let after_rx = read_used_idx(&mem, RX_USED);
            let after = progress_sum(&counters);

            prop_assert!(after >= before, "event counters must be monotonic");
            let progress = (after - before)
                + after_tx.wrapping_sub(before_tx) as u64
                + after_rx.wrapping_sub(before_rx) as u64;
            prop_assert!(
                progress >= 1,
                "no visible progress through the PCI decode: tx_used_delta={} \
                 rx_used_delta={} counter_delta={} (chain len={})",
                after_tx.wrapping_sub(before_tx),
                after_rx.wrapping_sub(before_rx),
                after - before,
                descs.len(),
            );
        }

        /// MSI-X variant of the chain-mutation fuzz: identical random TX chains,
        /// but MSI-X enabled so each used-ring publish drives `MsixState::fire`
        /// (vector resolve + mask/PBA gate + eventfd write) instead of the INTx
        /// line. Proves the MSI-X signal path survives hostile chains (no panic /
        /// OOB) and the same forward-progress invariant holds — chain parsing is
        /// transport-neutral, so progress must match the INTx arm regardless of
        /// the interrupt transport. Each case is a fresh `build_msix` (fresh
        /// eventfds), so vector firings never overflow across cases.
        #[test]
        fn facade_msix_tx_chain_progress_under_random_descriptors(
            descs in prop::collection::vec(
                (0u64..(1u64 << 24), 0u32..(8 * 1024 * 1024), 0u16..8, any::<u16>()),
                1..=8usize,
            ),
        ) {
            let mem = test_mem();
            let (mut pci, counters, _msix, _evts, _sink) = build_msix(&mem);
            drive_to_features_ok(&mut pci);
            program_queue(&mut pci, RXQ as u32, RX_DESC, RX_AVAIL, RX_USED);
            program_queue(&mut pci, TXQ as u32, TX_DESC, TX_AVAIL, TX_USED);
            enable_msix_shared(&mut pci);
            cc_w(&mut pci, CC_DEVICE_STATUS, S_OK);

            for (i, (addr, len, flags, next)) in descs.iter().enumerate() {
                write_desc(&mem, TX_DESC, i as u16, *addr, *len, *flags, *next);
            }
            publish_avail(&mem, TX_AVAIL, 0);
            write_desc(&mem, RX_DESC, 0, RX_BUF, 2048, 2, 0); // F_WRITE, well-formed
            publish_avail(&mem, RX_AVAIL, 0);

            let before_tx = read_used_idx(&mem, TX_USED);
            let before_rx = read_used_idx(&mem, RX_USED);
            let before = progress_sum(&counters);

            pci.bar_write(
                NOTIFY_OFFSET + (TXQ as u64) * NOTIFY_OFF_MULTIPLIER as u64,
                &[0u8; 2],
            );

            let after_tx = read_used_idx(&mem, TX_USED);
            let after_rx = read_used_idx(&mem, RX_USED);
            let after = progress_sum(&counters);

            prop_assert!(after >= before, "event counters must be monotonic");
            let progress = (after - before)
                + after_tx.wrapping_sub(before_tx) as u64
                + after_rx.wrapping_sub(before_rx) as u64;
            prop_assert!(
                progress >= 1,
                "no visible progress through the MSI-X PCI decode (chain len={})",
                descs.len(),
            );
        }
    }
}
