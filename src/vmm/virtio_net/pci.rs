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
//! Interrupts: INTx (no MSI-X in v0), delivered via the device's existing
//! `irq_evt`. The KVM wiring (in `setup::init_virtio_net_pci`) branches on the
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

use virtio_bindings::virtio_mmio::{VIRTIO_MMIO_INT_CONFIG, VIRTIO_MMIO_INT_VRING};

use super::device::{NUM_QUEUES, VirtioNet};
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
/// Total BAR0 size: 4 regions × 4 KiB.
const BAR0_SIZE: u64 = 0x4000;
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

/// `VIRTIO_MSI_NO_VECTOR` — INTx-only, so the MSI-X vector registers read this.
const NO_VECTOR: u16 = 0xFFFF;

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
}

impl VirtioNetPci {
    /// Build the PCI function around an already-constructed [`VirtioNet`]
    /// (the caller has already taken its `irq_evt`/`counters` and called
    /// `set_mem`). Lays out the standard header, the four virtio vendor
    /// capabilities, and BAR0. `bar_aperture` is the host-bridge `_CRS` MMIO
    /// grant `[start, end)` BAR0 must stay within (enforced by `bar_window`).
    pub(crate) fn new(net: VirtioNet, bar_aperture: (u64, u64)) -> Self {
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
        // also makes INTX_DISABLE (bit 10) writable for read-back, but the
        // device does NOT consult it — signal_used asserts INTx regardless.
        // This diverges from qemu (pci_irq_disabled suppresses INTx while
        // INTX_DISABLE is set) and is inert in practice: the Linux
        // virtio_pci_modern driver only sets INTX_DISABLE when switching to
        // MSI-X, which this INTx-only device does not offer, so the guest never
        // exercises the suppressed path. Honoring it lands with MSI-X support.
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

        Self::write_caps(&mut cfg);
        Self {
            cfg,
            net,
            bar_aperture,
        }
    }

    /// Lay out the four virtio vendor capabilities (COMMON/ISR/DEVICE/NOTIFY),
    /// chained via `cap_next`. The PCI_CFG capability (cfg_type 5) is
    /// intentionally omitted: `vp_modern_probe` never looks it up (it reads
    /// config only through ECAM/BAR), so a guest binds without it.
    fn write_caps(cfg: &mut ConfigSpace) {
        Self::write_cap(cfg, CAP_COMMON, CAP_ISR, CFG_TYPE_COMMON, COMMON_OFFSET);
        Self::write_cap(cfg, CAP_ISR, CAP_DEVICE, CFG_TYPE_ISR, ISR_OFFSET);
        Self::write_cap(cfg, CAP_DEVICE, CAP_NOTIFY, CFG_TYPE_DEVICE, DEVICE_OFFSET);
        // NOTIFY is the last cap (cap_next = 0) and carries the multiplier.
        // The uniform REGION_SIZE (4 KiB) notify cap MATCHES the references:
        // qemu sizes its notify region to notify_off_multiplier *
        // VIRTIO_QUEUE_MAX = 4 * 1024 = 0x1000 (hw/virtio/virtio-pci.c), and
        // cloud-hypervisor likewise grants a page-sized notify region — both
        // identical to REGION_SIZE here, not a divergence. It covers queue i's
        // notify address (i * NOTIFY_OFF_MULTIPLIER) for NUM_QUEUES up to
        // 0x1000 / NOTIFY_OFF_MULTIPLIER = 1024 slots — far above the current
        // NUM_QUEUES=2 (offsets 0 and 4). The multiqueue increment must revisit
        // this bound if it ever raises NUM_QUEUES past 1024.
        Self::write_cap(cfg, CAP_NOTIFY, 0, CFG_TYPE_NOTIFY, NOTIFY_OFFSET);
        cfg.set_u32(CAP_NOTIFY + CAP_OFF_NOTIFY_MULT, NOTIFY_OFF_MULTIPLIER);
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
            CC_MSIX_CONFIG => NO_VECTOR as u64,
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
            CC_QUEUE_MSIX_VECTOR => NO_VECTOR as u64,
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
            // msix_config / queue_msix_vector: INTx, no vectors to latch.
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
}

impl PciFunction for VirtioNetPci {
    fn config_read(&self, reg: u16, data: &mut [u8]) {
        self.cfg.read(reg, data);
    }

    fn config_write(&mut self, reg: u16, data: &[u8]) {
        self.cfg.write(reg, data);
    }

    fn bar_window(&self) -> Option<(u64, u64)> {
        // Recomputed from config space on every call (the MMIO dispatch calls
        // this per BAR exit via PciBus::bar_owner). The window is immutable
        // between COMMAND/BAR0 writes, so a per-function cache invalidated on
        // those writes is the obvious optimization — deliberately deferred to
        // the multi-NIC increment, where the 32-slot bar_owner scan it sits
        // behind actually scales; for a single NIC the two reads here are
        // negligible.
        // The BAR is live only once the guest enables memory-space decode.
        // Only PCI_COMMAND_MEMORY is consulted, NOT bus-master (bit 2): on real
        // hardware the device cannot DMA until BM is set, but the in-VMM loopback
        // reads/writes guest memory unconditionally. Benign — Linux's
        // virtio_pci_modern_probe sets bus-master before DRIVER_OK, and the
        // loopback only touches memory on a post-DRIVER_OK notify — so BM is
        // always set before first use; honoring it would land with MSI-X (like
        // INTX_DISABLE above).
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
            // Fast-path for the common reset value (BAR unprogrammed). This is
            // redundant with the grant check below — base 0 is far below
            // grant_start (PCI_MMIO_BAR_BASE) so that check also returns None —
            // but the explicit branch documents the reset case, which yields no
            // window; benign (a guest can only break its own NIC, never host
            // memory).
            return None;
        }
        // Enforce the host-bridge _CRS MMIO grant: the published window is only
        // honored when [base, base + BAR0_SIZE) lies fully within the aperture
        // the guest's PCI0 _CRS advertises. The MMIO dispatch checks
        // bar_mmio_contains BEFORE ecam_contains, so without this bound a guest
        // that (non-conformantly) programmed BAR0 over the ECAM window would
        // shadow config space; rejecting an out-of-grant base (return None,
        // claiming no MMIO) makes that guest-only breakage impossible. Linux
        // always assigns BARs inside the granted window, so a conformant guest
        // is unaffected. This matches the reference VMMs, which assign/bound
        // BARs from a VMM-owned allocator rather than trusting the guest base.
        let (grant_start, grant_end) = self.bar_aperture;
        if base < grant_start || base.saturating_add(BAR0_SIZE) > grant_end {
            return None;
        }
        Some((base, BAR0_SIZE))
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
            // ISR is read-only; device-config writes are ignored (virtio-v1.2
            // §4.2.2.2 — the device may ignore config-space writes).
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
    use virtio_bindings::virtio_config::VIRTIO_F_VERSION_1;
    use virtio_bindings::virtio_net::VIRTIO_NET_F_MAC;
    use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

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

    /// Build a PCI function over `mem`; returns the counters handle (cloned
    /// before the core moves into the facade) so a loopback's effect is
    /// observable.
    fn build(mem: &GuestMemoryMmap) -> (VirtioNetPci, std::sync::Arc<super::super::VirtioNetCounters>) {
        let mut net = VirtioNet::new(NetConfig::default());
        net.set_mem(mem.clone());
        let counters = net.counters();
        (VirtioNetPci::new(net, TEST_BAR_APERTURE), counters)
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

        // The four virtio vendor caps, chained COMMON->ISR->DEVICE->NOTIFY->0.
        let chain = [
            (CAP_COMMON, CAP_ISR as u8, CFG_TYPE_COMMON, CAP_LEN_STD, COMMON_OFFSET),
            (CAP_ISR, CAP_DEVICE as u8, CFG_TYPE_ISR, CAP_LEN_STD, ISR_OFFSET),
            (CAP_DEVICE, CAP_NOTIFY as u8, CFG_TYPE_DEVICE, CAP_LEN_STD, DEVICE_OFFSET),
            (CAP_NOTIFY, 0u8, CFG_TYPE_NOTIFY, CAP_LEN_NOTIFY, NOTIFY_OFFSET),
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

    #[test]
    fn common_cfg_register_reads() {
        let mem = test_mem();
        let (mut pci, _c) = build(&mem);
        assert_eq!(cc_r(&mut pci, CC_NUM_QUEUES), NUM_QUEUES as u32);
        // MSI-X is not implemented: config + per-queue vector read NO_VECTOR.
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
        let mut pci = VirtioNetPci::new(net, TEST_BAR_APERTURE);
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
        let mut pci = VirtioNetPci::new(net, TEST_BAR_APERTURE);
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
            let mut pci = VirtioNetPci::new(net, TEST_BAR_APERTURE);
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
    }
}
