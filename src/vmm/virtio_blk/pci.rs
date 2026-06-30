//! virtio-pci-modern transport facade for [`VirtioBlk`].
//!
//! Presents the virtio-block device as a PCI function (the sibling of the
//! virtio-MMIO facade in [`super::mmio`]): a PCI configuration space with the
//! virtio-modern vendor-capability chain, and a single memory BAR (BAR0) whose
//! four 4 KiB sub-regions (common / ISR / device / notify, virtio-v1.2 §4.1.4)
//! the guest's `virtio_pci_modern` driver drives. This file owns ONLY the
//! PCI/BAR decode; all device behaviour (status FSM, feature negotiation, queue
//! ring assembly, notify, interrupt bookkeeping, the request worker) lives in
//! the transport-neutral core on [`VirtioBlk`] (the register/FSM ops in
//! [`super::control`], the state in [`super::device`]) and is shared verbatim
//! with the MMIO facade.
//!
//! virtio-block is single-queue: one request virtqueue (`REQ_QUEUE` = 0), no
//! control vq and no RX/TX split. So `num_queues` = 1, the MSI-X table is sized
//! `num_queues + 1` (one request-queue vector + one config vector), and a
//! used-ring publish emits `IrqSource::Vring { queue: 0 }` while a
//! NEEDS_RESET/config-change emits `IrqSource::Config` (see the worker's
//! `signal_or_intx` in [`super::drain`]).
//!
//! Interrupts: MSI-X when offered (the cap is advertised whenever a KVM route
//! owner is wired — on BOTH irqchip paths for an x86 PCI guest: the
//! full-irqchip `FullIrqchipRouteOwner` or the split-irqchip `IoapicHandle`,
//! both [`MsixRouteSink`]s — the per-vector delivery state + the fire-or-pend
//! gate live in the shared [`crate::vmm::virtio_msix`] module's [`MsixState`],
//! and the cap / table / route-install decode is in this file), and INTx as the
//! fallback (no route owner — non-PCI; or a guest that declines MSI-X),
//! delivered via the device's existing `irq_evt`. The INTx KVM wiring (in
//! `setup::init_virtio_blk_pci`) branches on the host-side `vm.split_irqchip`
//! bool exactly as the virtio-net facade does.
//!
//! The legacy INTx path needs no active resample handler (and the split-irqchip
//! edge path no level re-injection): every `irq_evt` assert is reached from the
//! dedicated worker thread's `drain_bracket_impl`, which processes one drain at
//! a time off its epoll loop — a kick's interrupt is consumed by the guest's
//! `virtblk_done` before the worker fires the next, so no assert is lost to an
//! already-high level. (virtio-blk has no async-RX backend that would assert
//! independently of a guest kick, and it is single-queue, so the
//! multi-asserter hazard the virtio-net facade documents does not arise here.)

use std::sync::Arc;

use virtio_bindings::virtio_mmio::{VIRTIO_MMIO_INT_CONFIG, VIRTIO_MMIO_INT_VRING};

use super::device::VirtioBlk;
use crate::vmm::PiMutex;
use crate::vmm::pci::{
    ConfigSpace, PCI_COMMAND_MEMORY, PCI_COMMAND_WMASK, PCI_STATUS_CAP_LIST, PciFunction, REG_BAR0,
    REG_CAP_PTR, REG_CLASS, REG_COMMAND, REG_DEVICE_ID, REG_INTERRUPT_LINE, REG_INTERRUPT_PIN,
    REG_REVISION_ID, REG_STATUS, REG_SUBCLASS, REG_VENDOR_ID,
};
use crate::vmm::virtio_msix::{MSIX_TABLE_MAX, MsixRouteSink, MsixState, NO_VECTOR};

// ---- PCI identity (virtio-pci modern, virtio-v1.2 §4.1.2) ----
/// Red Hat / virtio vendor id.
const VENDOR_ID: u16 = 0x1AF4;
/// `0x1040 + VIRTIO_ID_BLOCK (2)` — the modern (non-transitional) virtio-block
/// id (vs virtio-net's 0x1041).
const DEVICE_ID: u16 = 0x1042;
/// Revision >= 1 marks a non-transitional (modern-only) device.
const REVISION: u8 = 0x01;
/// PCI base class 0x01 (mass storage), subclass 0x00 (SCSI). We follow qemu,
/// which presents modern virtio-blk-pci as `PCI_CLASS_STORAGE_SCSI` (0x0100);
/// cloud-hypervisor instead uses 0x01/0x80 (mass-storage-other). The choice is
/// purely cosmetic (lspci text): the guest's `virtio_pci` driver binds on
/// `PCI_DEVICE(PCI_VENDOR_ID_REDHAT_QUMRANET, PCI_ANY_ID)` — vendor id alone,
/// no class predicate (kernel drivers/virtio/virtio_pci_common.c) — so any
/// class/subclass binds. Diverges from the virtio-net facade's 0x02/0x00
/// (network controller / ethernet).
const CLASS_STORAGE: u8 = 0x01;
const SUBCLASS_SCSI: u8 = 0x00;
/// Interrupt pin INTA (1) — the device asserts INTA; the DSDT `_PRT` routes it.
const INTERRUPT_PIN_INTA: u8 = 0x01;

// ---- BAR0 layout: one 4 KiB region per modern-transport structure ----
const REGION_SIZE: u64 = 0x1000;
const COMMON_OFFSET: u64 = 0x0000;
const ISR_OFFSET: u64 = 0x1000;
const DEVICE_OFFSET: u64 = 0x2000;
const NOTIFY_OFFSET: u64 = 0x3000;
/// MSI-X table (one 4 KiB page, page-isolated). Holds up to [`MSIX_TABLE_MAX`]
/// (`0x1000 / 16 = 256`) 16-byte entries; the device advertises `num_vectors`
/// (`num_queues + 1`, capped at `MSIX_TABLE_MAX`) of them.
const MSIX_TABLE_OFFSET: u64 = 0x4000;
/// MSI-X Pending Bit Array (one 4 KiB page; one bit per vector).
const MSIX_PBA_OFFSET: u64 = 0x5000;
/// Total BAR0 size, rounded UP to a power of two (PCI BAR size-probe requires
/// it). Six 4 KiB structures span 0x6000; the next power of two is 0x8000.
const BAR0_SIZE: u64 = 0x8000;
/// Per-entry size in the MSI-X table: msg addr lo/hi, msg data, vector control
/// (4 × u32 = 16 bytes, PCI spec).
const MSIX_ENTRY_SIZE: u64 = 16;
/// The MSI-X table occupies exactly one 4 KiB page, so it holds
/// `0x1000 / 16 = 256` entries — the cap the delivery module enforces on the
/// advertised vector count ([`MSIX_TABLE_MAX`]).
const _: () = assert!(
    (MSIX_PBA_OFFSET - MSIX_TABLE_OFFSET) / MSIX_ENTRY_SIZE == MSIX_TABLE_MAX as u64,
    "MSI-X table page must hold exactly MSIX_TABLE_MAX 16-byte entries"
);
/// `queue_notify_off(i) == i`, so the notify address for queue `i` is
/// `NOTIFY_OFFSET + i * NOTIFY_OFF_MULTIPLIER`. Decode is
/// `off / NOTIFY_OFF_MULTIPLIER`; the core's `queue_notify_off` returns the
/// matching index. virtio-blk's single request queue lives at index
/// `REQ_QUEUE` = 0, so its notify address is `NOTIFY_OFFSET`.
const NOTIFY_OFF_MULTIPLIER: u32 = 4;

/// BAR0 low-dword type bits: bit0=0 (memory), bits\[2:1\]=00 (32-bit), bit3=0
/// (non-prefetchable) → low nibble `0x0`. 32-bit non-prefetch matches the host
/// bridge's 32-bit non-prefetch DWord `_CRS` window (the same deliberate choice
/// the virtio-net facade makes, and the one cloud-hypervisor makes specifically
/// for its block device). Diverges from qemu/firecracker's 64-bit prefetchable
/// virtio-blk BAR — intentional, to fit the x86 32-bit `_CRS` grant.
const BAR0_TYPE_BITS: u32 = 0x00;
/// Writable base bits of BAR0 low: the BAR is `BAR0_SIZE`-aligned, so bits
/// below `log2(size)` are read-only and the guest's size-probe recovers
/// `BAR0_SIZE`.
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
/// MSI-X cap config-space offset: immediately after NOTIFY (0x70 + 0x14 = 0x84).
const CAP_MSIX: u16 = 0x84;
const MSIX_OFF_MSG_CTRL: u16 = 2;
const MSIX_OFF_TABLE: u16 = 4;
const MSIX_OFF_PBA: u16 = 8;
/// Message Control writable bits: bit15 = MSI-X Enable, bit14 = Function Mask
/// (both guest-RW); the Table Size field `[10:0]` is read-only.
const MSIX_MSG_CTRL_WMASK: u16 = 0xC000;
/// BIR 0: every MSI-X structure lives in BAR0.
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

/// The virtio-block device exposed as a PCI function. Holds the device core as
/// a SHARED `Arc<PiMutex<VirtioBlk>>` (the run loop keeps a clone) — a
/// deliberate divergence from the virtio-net facade, which moves its device in
/// by value. virtio-blk runs a dedicated request worker thread that the freeze
/// coordinator must `pause()`/`resume()` across the rendezvous (so the worker
/// does not mutate guest memory while the failure dump reads it) and whose
/// `counters`/`paused_handle`/`parked_evt` the run loop reaches — all through
/// the same Arc the facade decodes BAR accesses against; virtio-net has no
/// worker, so its run loop needs only the moved-out counters. Lock order is
/// PciBus → VirtioBlk on the vCPU BAR path; the freeze coordinator locks
/// VirtioBlk alone (never PciBus), so there is no inversion. The worker thread
/// holds its own Arc-clones of the shared state (queues/backing/counters/paused
/// flag), never the device `PiMutex`, so it does not contend this lock.
pub(crate) struct VirtioBlkPci {
    cfg: ConfigSpace,
    blk: Arc<PiMutex<VirtioBlk>>,
    /// The host-bridge `_CRS` MMIO grant `[start, end)` this function's BAR0
    /// must lie within (enforced by `bar_window`); rejects an out-of-grant base
    /// so a non-conformant guest cannot place BAR0 over the ECAM window.
    bar_aperture: (u64, u64),
    /// Shared MSI-X delivery state — the single source of truth the facade
    /// mutates from config-space writes and the device core (worker thread)
    /// signals through. See [`crate::vmm::virtio_msix`].
    msix: Arc<PiMutex<MsixState>>,
    /// Host KVM route programmer (`FullIrqchipRouteOwner` on full-irqchip,
    /// `IoapicHandle` on split); `None` only without an owner (non-PCI), in
    /// which case the facade never installs a route and the guest stays on INTx.
    route_sink: Option<Arc<dyn MsixRouteSink>>,
    /// The KVM GSI assigned to each MSI-X table vector. Length = the device's
    /// vector count (`num_vectors` = num_queues + 1), so `gsis.len()` is the
    /// MSI-X table size the facade advertises and bounds against.
    gsis: Vec<u32>,
    /// Cached BAR0 MMIO window `[base, base + BAR0_SIZE)` — recomputed only on a
    /// `REG_COMMAND` / `REG_BAR0` write (its sole inputs), not per MMIO exit.
    /// Coherent under the single `PciBus` mutex. Mirrors qemu's
    /// `pci_update_mappings`.
    bar_window_cache: Option<(u64, u64)>,
}

impl VirtioBlkPci {
    /// Build the PCI function around a SHARED [`VirtioBlk`] (the run loop holds
    /// the other Arc clone; the caller has already called `set_mem` /
    /// `set_worker_placement` / `set_parked_evt`). Lays out the standard header,
    /// the four virtio vendor capabilities, BAR0, and the MSI-X cap (when a
    /// route sink is wired).
    pub(crate) fn new(
        blk: Arc<PiMutex<VirtioBlk>>,
        bar_aperture: (u64, u64),
        msix: Arc<PiMutex<MsixState>>,
        route_sink: Option<Arc<dyn MsixRouteSink>>,
        gsis: Vec<u32>,
    ) -> Self {
        // The advertised MSI-X table size + the clamp_vector bound come from
        // `gsis.len()`, while the device core's fire()/table_dword bounds come
        // from `msix.num_vectors()`; the host wiring sizes both to the same
        // per-device vector count, so they MUST agree.
        debug_assert_eq!(
            gsis.len(),
            msix.lock().num_vectors(),
            "MSI-X GSI count must equal the advertised vector count (num_vectors)"
        );
        // Hand the device core its clone of the shared MSI-X state ONLY when
        // MSI-X is actually offered (a route sink is present). With NO sink
        // (non-PCI / aarch64 MMIO) the cap is omitted below and the core keeps
        // `msix = None`: its signal sites take the INTx branch with no
        // per-signal lock — byte-identical to a device built without MSI-X
        // support. When a sink IS wired (x86 PCI), the core holds `msix = Some`
        // from here on, even before the guest enables MSI-X — so a guest that
        // declines MSI-X (older driver, `pci=nomsi`) still delivers over INTx,
        // but `signal_or_intx` then takes the (uncontended) MsixState lock per
        // signal to test `enabled()` before falling through to `irq_evt`. That
        // path is correct, just not lock-free; the lock-free INTx claim holds
        // only for the no-sink case above.
        if route_sink.is_some() {
            blk.lock().set_msix_state(Arc::clone(&msix));
        }
        let mut cfg = ConfigSpace::new();

        // Standard header.
        cfg.set_u16(REG_VENDOR_ID, VENDOR_ID);
        cfg.set_u16(REG_DEVICE_ID, DEVICE_ID);
        cfg.set_u8(REG_REVISION_ID, REVISION);
        cfg.set_u8(REG_SUBCLASS, SUBCLASS_SCSI);
        cfg.set_u8(REG_CLASS, CLASS_STORAGE);
        // Subsystem vendor/device IDs (0x2C/0x2E) left zero — matches qemu's
        // pure virtio-1.0 (modern) path; the modern virtio bus binds on
        // id.device (DEVICE_ID - 0x1040 = VIRTIO_ID_BLOCK) alone.
        cfg.set_u8(REG_INTERRUPT_PIN, INTERRUPT_PIN_INTA);
        // INTERRUPT_LINE is R/W (PCI spec), resets to 0; the guest's ACPI `_PRT`
        // is the authoritative source for `pci_dev->irq`. Kept writable so a
        // guest that stores its resolved line reads it back; PIN stays read-only.
        cfg.set_wmask_u16(REG_INTERRUPT_LINE, 0x00FF);
        // COMMAND writable (memory-space decode + bus-master); STATUS advertises
        // the cap list. INTx suppression under MSI-X is by the device's signal
        // gate (MsixState::enabled), not the INTX_DISABLE bit.
        cfg.set_wmask_u16(REG_COMMAND, PCI_COMMAND_WMASK);
        cfg.set_u16(REG_STATUS, PCI_STATUS_CAP_LIST);
        cfg.set_u8(REG_CAP_PTR, CAP_COMMON as u8);

        // BAR0: 32-bit non-prefetchable memory BAR. The low nibble carries the
        // read-only type bits; the base bits are writable (size-aligned). A
        // 32-bit BAR0 occupies only register 0x10.
        cfg.set_u32(REG_BAR0, BAR0_TYPE_BITS);
        cfg.set_wmask_u32(REG_BAR0, BAR0_LOW_WMASK);

        // MSI-X advertised iff a route sink is wired (either irqchip path).
        Self::write_caps(&mut cfg, route_sink.is_some(), gsis.len() as u16);
        let mut this = Self {
            cfg,
            blk,
            bar_aperture,
            msix,
            route_sink,
            gsis,
            bar_window_cache: None,
        };
        // Seed the cache from the freshly built config space (BAR0 unprogrammed +
        // memory decode disabled at reset → None). Refreshed on REG_COMMAND /
        // REG_BAR0 writes in `config_write`.
        this.bar_window_cache = this.recompute_bar_window();
        this
    }

    /// Lay out the four virtio vendor capabilities (COMMON/ISR/DEVICE/NOTIFY),
    /// chained via `cap_next`, plus the MSI-X cap when offered.
    fn write_caps(cfg: &mut ConfigSpace, msix: bool, table_size: u16) {
        Self::write_cap(cfg, CAP_COMMON, CAP_ISR, CFG_TYPE_COMMON, COMMON_OFFSET);
        Self::write_cap(cfg, CAP_ISR, CAP_DEVICE, CFG_TYPE_ISR, ISR_OFFSET);
        Self::write_cap(cfg, CAP_DEVICE, CAP_NOTIFY, CFG_TYPE_DEVICE, DEVICE_OFFSET);
        // NOTIFY chains to the MSI-X cap only when MSI-X is offered; else
        // terminates the chain (cap_next = 0).
        let notify_next: u16 = if msix { CAP_MSIX } else { 0 };
        Self::write_cap(cfg, CAP_NOTIFY, notify_next, CFG_TYPE_NOTIFY, NOTIFY_OFFSET);
        cfg.set_u32(CAP_NOTIFY + CAP_OFF_NOTIFY_MULT, NOTIFY_OFF_MULTIPLIER);
        if msix {
            Self::write_msix_cap(cfg, table_size);
        }
    }

    /// Write the standard MSI-X capability at `CAP_MSIX`: cap id, next = 0,
    /// Message Control (Table Size encoded as N-1; Enable/Function-Mask
    /// guest-writable), and the Table / PBA `offset|BIR` dwords into BAR0.
    fn write_msix_cap(cfg: &mut ConfigSpace, table_size: u16) {
        cfg.set_u8(CAP_MSIX, PCI_CAP_ID_MSIX);
        cfg.set_u8(CAP_MSIX + 1, 0);
        cfg.set_u16(CAP_MSIX + MSIX_OFF_MSG_CTRL, table_size - 1);
        cfg.set_wmask_u16(CAP_MSIX + MSIX_OFF_MSG_CTRL, MSIX_MSG_CTRL_WMASK);
        cfg.set_u32(
            CAP_MSIX + MSIX_OFF_TABLE,
            MSIX_TABLE_OFFSET as u32 | MSIX_BIR0,
        );
        cfg.set_u32(CAP_MSIX + MSIX_OFF_PBA, MSIX_PBA_OFFSET as u32 | MSIX_BIR0);
    }

    /// Write one `struct virtio_pci_cap` at config offset `at`, pointing to the
    /// BAR0 sub-region `[region_off, region_off + REGION_SIZE)`. The cap LENGTH
    /// is the full 4 KiB sub-region (Linux's vp_modern addresses each register
    /// at its fixed offset and never iterates by LENGTH). 4 KiB clears every
    /// region's kernel floor: `vp_modern_probe` maps each cap with
    /// `vp_modern_map_capability(mdev, cap, minlen, align, start=0, size, ...)`,
    /// which rejects a cap whose `length - start < minlen`. The per-cap `minlen`
    /// is `sizeof(virtio_pci_common_cfg)` (56) for common, 1 for ISR, 2 for
    /// notify, and 0 for the device cap (drivers/virtio/virtio_pci_modern_dev.c).
    /// So the device cap's LENGTH is NOT floored by the map (minlen 0); the map
    /// instead CLAMPS the mapped length to its `size` arg (`PAGE_SIZE` for the
    /// device cap) and records `device_len = min(LENGTH, PAGE_SIZE) = 4096`. The
    /// real lower bound on the device cap is the RUNTIME read: `vp_get` does
    /// `BUG_ON(offset + len > device_len)`, and the guest reads through offset 24
    /// (the `blk_size` u32 sits at field offset 0x14 and ends at 24), so
    /// `device_len` (hence LENGTH) must cover the 24-byte populated
    /// `VirtioBlkConfig`. 4 KiB satisfies that for every region and page-isolates
    /// each one.
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

    /// Clamp a guest-written MSI-X vector number to a valid index
    /// `[0, table_size)` or `NO_VECTOR`; any other value is out of range and
    /// stored as `NO_VECTOR` so a hostile vector cannot index the table OOB.
    fn clamp_vector(&self, v: u16) -> u16 {
        if v == NO_VECTOR || v < self.gsis.len() as u16 {
            v
        } else {
            NO_VECTOR
        }
    }

    /// MSI-X vector assigned to the currently-selected virtqueue, from the
    /// shared state; `NO_VECTOR` for an out-of-range selector.
    fn selected_queue_msix_vector(&self) -> u16 {
        let sel = self.blk.lock().queue_select() as usize;
        self.msix.lock().queue_vector(sel)
    }

    /// Serve a common-cfg register read at BAR-region offset `off`, offset-exact
    /// and natural-width (the guest's `virtio_pci_modern` reads each register at
    /// its own offset/width). Device-read registers (queue desc/avail/used)
    /// read 0.
    fn common_read(&self, off: u64, data: &mut [u8]) {
        // CC_QUEUE_MSIX_VECTOR is served via `selected_queue_msix_vector`, which
        // takes the blk lock itself; resolve it before locking here so the guard
        // is not held re-entrantly.
        if off == CC_QUEUE_MSIX_VECTOR {
            Self::put_le(self.selected_queue_msix_vector() as u64, data);
            return;
        }
        let blk = self.blk.lock();
        let val: u64 = match off {
            CC_DEVICE_FEATURE_SELECT => blk.device_features_sel() as u64,
            CC_DEVICE_FEATURE => blk.device_features_window() as u64,
            CC_DRIVER_FEATURE_SELECT => blk.driver_features_sel() as u64,
            CC_MSIX_CONFIG => self.msix.lock().config_vector() as u64,
            CC_NUM_QUEUES => blk.num_queues() as u64,
            CC_DEVICE_STATUS => blk.device_status() as u64,
            CC_CONFIG_GENERATION => blk.config_generation() as u64,
            // queue_select reads back the raw latched selector (virtio-v1.2
            // §4.1.4.3); the queue-INDEXED read-backs below clamp for an
            // out-of-range selector.
            CC_QUEUE_SELECT => blk.queue_select() as u64,
            // queue_size is R/W: read back the guest-configured size (resets to
            // max_size), not the immutable max.
            CC_QUEUE_SIZE => blk.queue_size() as u64,
            CC_QUEUE_ENABLE => blk.queue_ready() as u64,
            CC_QUEUE_NOTIFY_OFF => blk.queue_notify_off() as u64,
            // driver_feature (write-only in practice) + the queue ring addresses
            // are not read back by the driver.
            _ => 0,
        };
        Self::put_le(val, data);
    }

    /// Apply a common-cfg register write at BAR-region offset `off`. Each
    /// register maps to a transport-neutral core op (the queue-config /
    /// feature-lock gates live inside the op); read-only registers and the
    /// INTx-inert MSI-X registers are dropped.
    ///
    /// Width: the register is decoded from `off` alone and `get_u32` takes the
    /// low <=4 bytes. This relies on the Linux-width access model — `vp_modern`
    /// writes every common-cfg register at its natural width (<=4 bytes), and
    /// the 64-bit ring-address registers (desc/avail/used) as two SEPARATE
    /// 32-bit lo/hi writes (`vp_iowrite64_twopart`), never one 8-byte access —
    /// so a conformant guest never straddles the lo/hi pair. A non-conformant
    /// wide (e.g. 8-byte) write lands only its low dword; the spill is dropped,
    /// which is harmless because the ring stays unprogrammed until both halves
    /// land via the per-half ops. (firecracker instead width-dispatches and
    /// drops mis-width writes; ours is width-agnostic under this assumption.)
    fn common_write(&mut self, off: u64, data: &[u8]) {
        let val = Self::get_u32(data);
        let mut blk = self.blk.lock();
        match off {
            CC_DEVICE_FEATURE_SELECT => blk.set_device_features_sel(val),
            CC_DRIVER_FEATURE_SELECT => blk.set_driver_features_sel(val),
            CC_DRIVER_FEATURE => blk.set_driver_features_window(val),
            CC_DEVICE_STATUS => blk.write_status(val & 0xFF),
            CC_QUEUE_SELECT => blk.set_queue_select(val),
            CC_QUEUE_SIZE => blk.set_queue_size(val as u16),
            CC_QUEUE_ENABLE => blk.set_queue_ready(val != 0),
            CC_QUEUE_DESC_LO => blk.set_queue_desc_addr(Some(val), None),
            CC_QUEUE_DESC_HI => blk.set_queue_desc_addr(None, Some(val)),
            CC_QUEUE_AVAIL_LO => blk.set_queue_avail_addr(Some(val), None),
            CC_QUEUE_AVAIL_HI => blk.set_queue_avail_addr(None, Some(val)),
            CC_QUEUE_USED_LO => blk.set_queue_used_addr(Some(val), None),
            CC_QUEUE_USED_HI => blk.set_queue_used_addr(None, Some(val)),
            // MSI-X vector latches: store the guest-assigned vector (clamped)
            // so the readback echoes it (the kernel's vp_modern aborts -EBUSY on
            // a read-back mismatch). `clamp_vector` reads only `self.gsis` and
            // `self.msix` is a distinct lock, so resolving them under the held
            // `blk` guard does not re-enter it.
            CC_MSIX_CONFIG => {
                let v = self.clamp_vector(val as u16);
                self.msix.lock().set_config_vector(v);
            }
            CC_QUEUE_MSIX_VECTOR => {
                let sel = blk.queue_select() as usize;
                let v = self.clamp_vector(val as u16);
                self.msix.lock().set_queue_vector(sel, v);
            }
            _ => {}
        }
    }

    /// Serve the 1-byte ISR register: bit0 = queue interrupt, bit1 =
    /// config-change, mapped from the core's `interrupt_status` bits
    /// (`INT_VRING`/`INT_CONFIG`). Read-to-clear (no separate ACK register on
    /// the modern PCI ISR cap), so it clears BOTH bits on every served read,
    /// matching the value-clear half of qemu's `virtio_pci_isr_read` xchg. A
    /// zero-length read is a no-op so a malformed empty access cannot consume a
    /// pending interrupt.
    ///
    /// Divergence from qemu (same as the virtio-net facade): qemu's
    /// `virtio_pci_isr_read` ALSO calls `pci_irq_deassert()` to lower the
    /// device-side INTx line on each ISR read. ktstr clears only the
    /// guest-visible virtio bits here; it models no device-side INTx line to
    /// lower because KVM owns INTx delivery — on the full-irqchip path KVM
    /// de-asserts the level GSI on guest EOI via the resample irqfd, and on the
    /// split-irqchip path each `irq_evt` assert is a one-shot edge MSI. So the
    /// device has no level to drop on an ISR read.
    fn isr_read(&mut self, data: &mut [u8]) {
        if data.is_empty() {
            return;
        }
        let mut blk = self.blk.lock();
        let isr = blk.interrupt_status();
        let mut byte = 0u8;
        if isr & VIRTIO_MMIO_INT_VRING != 0 {
            byte |= 0x1;
        }
        if isr & VIRTIO_MMIO_INT_CONFIG != 0 {
            byte |= 0x2;
        }
        blk.ack_interrupt(VIRTIO_MMIO_INT_VRING | VIRTIO_MMIO_INT_CONFIG);
        data.fill(0);
        if let Some(b) = data.first_mut() {
            *b = byte;
        }
    }

    /// Decode a notify-region write to a queue index and kick it. The index is
    /// `off / NOTIFY_OFF_MULTIPLIER` (floor); `notify_queue` dispatches it —
    /// only the request queue (`REQ_QUEUE`) drives a drain, any other index is
    /// inert.
    fn notify_write(&mut self, off: u64) {
        let idx = off / NOTIFY_OFF_MULTIPLIER as u64;
        self.blk.lock().notify_queue(idx as u32);
    }

    /// Serve an MSI-X table read at table-relative offset `rel`: decode
    /// `(entry, dword)` and return the stored register; 0 for an out-of-range
    /// entry.
    fn msix_table_read(&self, rel: u64, data: &mut [u8]) {
        let entry = (rel / MSIX_ENTRY_SIZE) as usize;
        let dword = ((rel % MSIX_ENTRY_SIZE) / 4) as usize;
        let val = self.msix.lock().table_dword(entry, dword);
        Self::put_le(val as u64, data);
    }

    /// Apply an MSI-X table write at table-relative offset `rel`: store the
    /// register, then reconcile the entry's GSI route to its current
    /// deliverability (install on the mask→unmask edge, remove on unmask→mask).
    fn msix_table_write(&mut self, rel: u64, data: &[u8]) {
        let entry = (rel / MSIX_ENTRY_SIZE) as usize;
        let dword = ((rel % MSIX_ENTRY_SIZE) / 4) as usize;
        let val = Self::get_u32(data);
        let _ = self.msix.lock().write_table_dword(entry, dword, val);
        self.reconcile_route(entry);
    }

    /// Reconcile vector `idx`'s KVM MSI route to its current deliverability
    /// (enabled && per-vector-unmasked && !function_mask): install when
    /// deliverable, remove otherwise — so masking/disable is enforced at the KVM
    /// route layer, not only device-side.
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

    /// Reconcile every vector (device-level Message Control edge: MSI-X
    /// Enable or Function Mask / MASKALL toggled). This is the function-mask-CLEAR
    /// PBA-replay mechanism: on a MASKALL 1->0 edge each now-deliverable vector
    /// is re-installed via `install_route`, which calls `replay_pending` — so a
    /// vector that recorded a PBA bit while function-masked fires exactly once on
    /// unmask. Mirrors firecracker's `set_msg_ctl` whole-PBA walk on the
    /// function-mask-clear edge.
    fn reconcile_routes(&mut self) {
        for idx in 0..self.gsis.len() {
            self.reconcile_route(idx);
        }
    }

    /// (Re)install the KVM MSI route for vector `idx` from its table dwords, then
    /// replay any pending bit. No-op without a route sink. The msix lock is
    /// released before the (SRCU-grace) route ioctl.
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
            self.msix.lock().replay_pending(idx);
        }
    }

    /// Remove the KVM MSI route for vector `idx` on a mask / disable edge. The
    /// device-side `fire()` gate already suppresses a masked vector; this also
    /// enforces masking at the KVM layer. No-op without a route sink.
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

    /// Serve an MSI-X PBA read (vector-indexed bitmap, LSB-first). Read-only —
    /// `bar_write` drops PBA writes.
    fn msix_pba_read(&self, rel: u64, data: &mut [u8]) {
        let base = rel as usize;
        let m = self.msix.lock();
        for (i, b) in data.iter_mut().enumerate() {
            *b = m.pba_byte(base + i);
        }
    }

    /// Recompute the BAR0 MMIO window from config space (the cached value).
    /// Called at construction and on each `REG_COMMAND` / `REG_BAR0` write. The
    /// BAR is live only once the guest enables memory-space decode
    /// (`PCI_COMMAND_MEMORY`); bus-master (bit 2) is not consulted (the in-VMM
    /// worker reads/writes guest memory unconditionally; Linux sets BM before
    /// DRIVER_OK regardless).
    fn recompute_bar_window(&self) -> Option<(u64, u64)> {
        let mut cmd = [0u8; 2];
        self.cfg.read(REG_COMMAND, &mut cmd);
        if u16::from_le_bytes(cmd) & PCI_COMMAND_MEMORY == 0 {
            return None;
        }
        let mut lo = [0u8; 4];
        self.cfg.read(REG_BAR0, &mut lo);
        let base = (u32::from_le_bytes(lo) & !(BAR0_SIZE as u32 - 1)) as u64;
        if base == 0 {
            return None;
        }
        // Enforce the host-bridge _CRS MMIO grant: reject a base whose
        // `[base, base + BAR0_SIZE)` is outside the aperture so a non-conformant
        // guest cannot place BAR0 over the ECAM window (the MMIO dispatch checks
        // bar_mmio_contains before ecam_contains). Linux always assigns BARs
        // inside the granted window.
        let (grant_start, grant_end) = self.bar_aperture;
        if base < grant_start || base.saturating_add(BAR0_SIZE) > grant_end {
            return None;
        }
        Some((base, BAR0_SIZE))
    }
}

impl PciFunction for VirtioBlkPci {
    fn config_read(&self, reg: u16, data: &mut [u8]) {
        self.cfg.read(reg, data);
    }

    fn config_write(&mut self, reg: u16, data: &[u8]) {
        self.cfg.write(reg, data);
        // Refresh the cached BAR0 window when the write touched its inputs —
        // REG_COMMAND (memory-decode enable) or REG_BAR0 (base). Uses
        // write_touches so a wide access spanning either register still
        // refreshes. Recompute AFTER cfg.write so it reads the just-written bytes.
        if Self::write_touches(reg, data.len(), REG_COMMAND, 2)
            || Self::write_touches(reg, data.len(), REG_BAR0, 4)
        {
            self.bar_window_cache = self.recompute_bar_window();
        }
        // If the write touched the MSI-X cap Message Control (Enable bit15 /
        // Function Mask bit14), sync the parsed state into the shared delivery
        // state, then reconcile every vector's KVM route.
        let mc_reg = CAP_MSIX + MSIX_OFF_MSG_CTRL;
        if Self::write_touches(reg, data.len(), mc_reg, 2) {
            let mut mc = [0u8; 2];
            self.cfg.read(mc_reg, &mut mc);
            let msg_ctrl = u16::from_le_bytes(mc);
            self.msix.lock().set_message_control(msg_ctrl);
            self.reconcile_routes();
        }
    }

    fn bar_window(&self) -> Option<(u64, u64)> {
        self.bar_window_cache
    }

    fn bar_read(&mut self, offset: u64, data: &mut [u8]) {
        // Region dispatch keys on the access START offset. A spec-compliant
        // guest never straddles a REGION_SIZE boundary; a non-spec cross-region
        // access is served by the start region and any overhang zero-padded
        // (every handler length-bounds). Same in bar_write.
        match offset {
            o if (COMMON_OFFSET..COMMON_OFFSET + REGION_SIZE).contains(&o) => {
                self.common_read(o - COMMON_OFFSET, data);
            }
            // Only the ISR register itself (region base) read-clears; any other
            // offset in the ISR region reads 0. The kernel's vp_interrupt reads
            // the 1-byte ISR at the cap base only.
            ISR_OFFSET => {
                self.isr_read(data);
            }
            o if (DEVICE_OFFSET..DEVICE_OFFSET + REGION_SIZE).contains(&o) => {
                self.blk.lock().read_blk_config(o - DEVICE_OFFSET, data);
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
            // (virtio-v1.2 §4.2.2.2 — the device owns the device-config region).
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    //! Host-runnable tests for the virtio-blk PCI transport facade — the decode
    //! glue (config space, vendor-cap chain, BAR-region dispatch, common-cfg
    //! register mapping onto the shared core ops, ISR read-to-clear, MSI-X cap /
    //! route install, bar_window grant gating) the booted-guest e2e exercises but
    //! cannot guard in CI (it SKIPs-as-PASS without KVM). The transport-neutral
    //! core (FSM, drain, request handling) is covered by the sibling
    //! `tests_fsm` / `tests_drain_*` suites through the MMIO facade; here the
    //! decode path is driven through the PCI facade.
    use super::super::testing::{make_device, make_guest_mem};
    use super::super::{DiskThrottle, S_ACK, S_DRV, S_FEAT, VIRTIO_BLK_SECTOR_SIZE};
    use super::*;
    use crate::vmm::virtio_msix::IrqSource;
    use virtio_bindings::virtio_config::VIRTIO_F_VERSION_1;
    use vmm_sys_util::eventfd::EventFd;

    const GUEST_MEM: usize = 0x10_0000;
    const CAP_BYTES: u64 = 0x10_0000; // 1 MiB → 2048 sectors
    // Host-bridge _CRS MMIO grant the production x86 caller passes (literal here
    // to keep this all-arch test module off the x86-only kvm constants).
    const TEST_BAR_APERTURE: (u64, u64) = (0xE010_0000, 0xFEC0_0000);
    const TEST_BAR_BASE: u32 = 0xE010_0000; // in-grant
    const TEST_ECAM_BASE: u32 = 0xE000_0000; // out-of-grant (would shadow ECAM)
    // num_vectors = num_queues(1) + config = 2.
    const N_VECTORS: usize = 2;

    /// Counting MSI-X route sink: records every `set_route` so a test can assert
    /// which routes the unmask / enable edges installed, without a live KVM fd.
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

    fn test_gsis() -> Vec<u32> {
        (0..N_VECTORS).map(|v| 40 + v as u32).collect()
    }

    fn build_blk() -> Arc<PiMutex<VirtioBlk>> {
        let mut blk = make_device(CAP_BYTES, DiskThrottle::default());
        blk.set_mem(make_guest_mem(GUEST_MEM));
        Arc::new(PiMutex::new(blk))
    }

    fn new_pci() -> VirtioBlkPci {
        let msix = Arc::new(PiMutex::new(MsixState::new(1, MSIX_TABLE_MAX)));
        let sink: Arc<dyn MsixRouteSink> = Arc::new(MockRouteSink::default());
        VirtioBlkPci::new(
            build_blk(),
            TEST_BAR_APERTURE,
            msix,
            Some(sink),
            test_gsis(),
        )
    }

    fn cfg8(pci: &VirtioBlkPci, reg: u16) -> u8 {
        let mut b = [0u8; 1];
        pci.config_read(reg, &mut b);
        b[0]
    }
    fn cfg16(pci: &VirtioBlkPci, reg: u16) -> u16 {
        let mut b = [0u8; 2];
        pci.config_read(reg, &mut b);
        u16::from_le_bytes(b)
    }
    fn cfg8u(pci: &VirtioBlkPci, reg: u16) -> u8 {
        cfg8(pci, reg)
    }
    fn cc_w(pci: &mut VirtioBlkPci, cc: u64, val: u32) {
        pci.bar_write(COMMON_OFFSET + cc, &val.to_le_bytes());
    }
    fn cc_r(pci: &mut VirtioBlkPci, cc: u64) -> u32 {
        let mut b = [0u8; 4];
        pci.bar_read(COMMON_OFFSET + cc, &mut b);
        u32::from_le_bytes(b)
    }

    #[test]
    fn pci_identity() {
        let pci = new_pci();
        assert_eq!(cfg16(&pci, REG_VENDOR_ID), 0x1AF4);
        assert_eq!(cfg16(&pci, REG_DEVICE_ID), 0x1042); // 0x1040 + VIRTIO_ID_BLOCK(2)
        assert_eq!(cfg8(&pci, REG_REVISION_ID), 0x01);
        assert_eq!(cfg8(&pci, REG_CLASS), 0x01); // mass storage
        assert_eq!(cfg8(&pci, REG_SUBCLASS), 0x00); // SCSI
        assert_eq!(cfg8(&pci, REG_INTERRUPT_PIN), 0x01); // INTA
    }

    #[test]
    fn cap_chain_advertises_msix() {
        let pci = new_pci();
        assert_ne!(cfg16(&pci, REG_STATUS) & PCI_STATUS_CAP_LIST, 0);
        assert_eq!(cfg8(&pci, REG_CAP_PTR), CAP_COMMON as u8);
        // NOTIFY chains to the MSI-X cap (a route sink is wired).
        assert_eq!(cfg8u(&pci, CAP_NOTIFY + CAP_OFF_NEXT), CAP_MSIX as u8);
        assert_eq!(cfg8(&pci, CAP_MSIX), PCI_CAP_ID_MSIX);
        // Table Size is encoded N-1.
        assert_eq!(
            cfg16(&pci, CAP_MSIX + MSIX_OFF_MSG_CTRL) & 0x7FF,
            (N_VECTORS as u16) - 1
        );
    }

    #[test]
    fn device_config_capacity_via_bar_device_region() {
        let mut pci = new_pci();
        // BAR0 DEVICE region offset 0 = VirtioBlkConfig.capacity (u64 LE, sectors).
        let mut b = [0u8; 8];
        pci.bar_read(DEVICE_OFFSET, &mut b);
        assert_eq!(
            u64::from_le_bytes(b),
            CAP_BYTES / VIRTIO_BLK_SECTOR_SIZE as u64
        );
    }

    #[test]
    fn common_cfg_status_fsm_and_queue_config() {
        let mut pci = new_pci();
        cc_w(&mut pci, CC_DEVICE_STATUS, S_ACK);
        cc_w(&mut pci, CC_DEVICE_STATUS, S_DRV);
        cc_w(&mut pci, CC_DRIVER_FEATURE_SELECT, 1);
        cc_w(
            &mut pci,
            CC_DRIVER_FEATURE,
            1u32 << (VIRTIO_F_VERSION_1 - 32),
        );
        cc_w(&mut pci, CC_DEVICE_STATUS, S_FEAT);
        assert_eq!(cc_r(&mut pci, CC_DEVICE_STATUS) & 0xFF, S_FEAT);
        assert_eq!(cc_r(&mut pci, CC_NUM_QUEUES) & 0xFFFF, 1); // single request queue
        // Configure the request queue (selector 0) via common-cfg.
        cc_w(&mut pci, CC_QUEUE_SELECT, 0);
        cc_w(&mut pci, CC_QUEUE_SIZE, 8);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_SIZE) & 0xFFFF, 8);
        cc_w(&mut pci, CC_QUEUE_ENABLE, 1);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_ENABLE) & 0xFFFF, 1);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_NOTIFY_OFF) & 0xFFFF, 0); // queue 0 → off 0
    }

    #[test]
    fn isr_reads_zero_with_no_pending_interrupt() {
        let mut pci = new_pci();
        let mut b = [0u8; 1];
        pci.bar_read(ISR_OFFSET, &mut b);
        assert_eq!(b[0], 0);
    }

    #[test]
    fn msix_enable_and_unmask_installs_route() {
        let msix = Arc::new(PiMutex::new(MsixState::new(1, MSIX_TABLE_MAX)));
        let sink = Arc::new(MockRouteSink::default());
        let sink_dyn: Arc<dyn MsixRouteSink> = sink.clone();
        let mut pci = VirtioBlkPci::new(
            build_blk(),
            TEST_BAR_APERTURE,
            msix,
            Some(sink_dyn),
            test_gsis(),
        );
        // Program vector 0's table entry (addr lo / data) then clear its mask bit.
        pci.bar_write(MSIX_TABLE_OFFSET, &0xFEE0_0000u32.to_le_bytes()); // addr lo
        pci.bar_write(MSIX_TABLE_OFFSET + 8, &0x4000u32.to_le_bytes()); // data
        pci.bar_write(MSIX_TABLE_OFFSET + 12, &0u32.to_le_bytes()); // vector ctrl: unmask
        // Enable MSI-X via the cap Message Control (bit15).
        pci.config_write(CAP_MSIX + MSIX_OFF_MSG_CTRL, &0x8000u16.to_le_bytes());
        let installs = sink.installs.lock().unwrap();
        assert!(
            installs
                .iter()
                .any(|(gsi, msg)| *gsi == 40 && msg.is_some()),
            "enable+unmask installs vector 0's route at GSI 40: {installs:?}"
        );
    }

    #[test]
    fn bar_window_honors_crs_grant() {
        let mut pci = new_pci();
        pci.config_write(REG_BAR0, &TEST_BAR_BASE.to_le_bytes());
        pci.config_write(REG_COMMAND, &PCI_COMMAND_MEMORY.to_le_bytes());
        assert_eq!(pci.bar_window(), Some((TEST_BAR_BASE as u64, BAR0_SIZE)));
        // An out-of-grant base (would shadow the ECAM window) is rejected.
        pci.config_write(REG_BAR0, &TEST_ECAM_BASE.to_le_bytes());
        assert_eq!(pci.bar_window(), None);
    }

    // ---- MSI-X decode + hostile input (ports of the virtio-net facade suite,
    // adapted to virtio-blk's single request queue: vector 0 = config,
    // vector 1 = the request queue) ----

    /// MSI-X vector latches echo the guest-written vector verbatim (the kernel's
    /// `vp_modern` writes a vector, reads it back, and aborts with -EBUSY if they
    /// differ). Config + the request-queue vector are independent latches.
    #[test]
    fn msix_vector_echo_readback() {
        let mut pci = new_pci();
        assert_eq!(cc_r(&mut pci, CC_MSIX_CONFIG) as u16, NO_VECTOR);
        cc_w(&mut pci, CC_MSIX_CONFIG, 0);
        assert_eq!(cc_r(&mut pci, CC_MSIX_CONFIG) as u16, 0);
        // The request queue (selector 0) latches its own vector.
        cc_w(&mut pci, CC_QUEUE_SELECT, 0);
        cc_w(&mut pci, CC_QUEUE_MSIX_VECTOR, 1);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_MSIX_VECTOR) as u16, 1);
        // The config vector is unaffected by the queue write.
        assert_eq!(cc_r(&mut pci, CC_MSIX_CONFIG) as u16, 0);
    }

    /// An out-of-range vector (>= table size, not `NO_VECTOR`) is rejected —
    /// stored as `NO_VECTOR` so the read-back signals rejection rather than
    /// letting a hostile vector index the table out of bounds.
    #[test]
    fn msix_vector_out_of_range_rejected() {
        let mut pci = new_pci();
        // N_VECTORS (2) is one past the last valid index (0, 1) → rejected.
        cc_w(&mut pci, CC_MSIX_CONFIG, N_VECTORS as u32);
        assert_eq!(cc_r(&mut pci, CC_MSIX_CONFIG) as u16, NO_VECTOR);
        cc_w(&mut pci, CC_QUEUE_SELECT, 0);
        cc_w(&mut pci, CC_QUEUE_MSIX_VECTOR, 0xABCD);
        assert_eq!(cc_r(&mut pci, CC_QUEUE_MSIX_VECTOR) as u16, NO_VECTOR);
    }

    /// MSI-X table reads/writes round-trip through the BAR0 table region: each
    /// 16-byte entry's four dwords (addr lo/hi, msg data, vector control) echo;
    /// an out-of-range entry reads 0 and drops writes (no OOB).
    #[test]
    fn msix_table_roundtrip_and_bounds() {
        let mut pci = new_pci();
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
        let oob = MSIX_TABLE_OFFSET + N_VECTORS as u64 * MSIX_ENTRY_SIZE;
        pci.bar_write(oob, &0xDEAD_BEEFu32.to_le_bytes());
        let mut b = [0u8; 4];
        pci.bar_read(oob, &mut b);
        assert_eq!(u32::from_le_bytes(b), 0, "out-of-range table entry reads 0");
    }

    /// The MSI-X PBA reads 0 (no pending vectors) and drops writes (read-only).
    #[test]
    fn msix_pba_reads_zero_and_read_only() {
        let mut pci = new_pci();
        pci.bar_write(MSIX_PBA_OFFSET, &0xFFu32.to_le_bytes());
        let mut b = [0u8; 4];
        pci.bar_read(MSIX_PBA_OFFSET, &mut b);
        assert_eq!(
            u32::from_le_bytes(b),
            0,
            "PBA reads 0 (no pending); write ignored"
        );
    }

    // ---- MSI-X delivery (enable → route install → fire/pend/replay) ----

    /// Message Control values: Enable (bit15), Function Mask / MASKALL (bit14).
    const MC_ENABLE: u16 = 0x8000;
    const MC_ENABLE_MASKALL: u16 = 0x8000 | 0x4000;

    /// Build a facade with MSI-X wired: a shared `MsixState` with RETAINED
    /// per-vector eventfds (so the test can drain them) + a retained mock route
    /// sink. MSI-X is NOT yet enabled. Returns (pci, shared msix, [eventfds],
    /// sink).
    #[allow(clippy::type_complexity)]
    fn build_msix() -> (
        VirtioBlkPci,
        Arc<PiMutex<MsixState>>,
        [EventFd; N_VECTORS],
        Arc<MockRouteSink>,
    ) {
        let msix = Arc::new(PiMutex::new(MsixState::new(1, MSIX_TABLE_MAX)));
        // Hand the state a clone of each eventfd; keep the originals to drain.
        let evts: [EventFd; N_VECTORS] =
            std::array::from_fn(|_| EventFd::new(libc::EFD_NONBLOCK).unwrap());
        for (v, e) in evts.iter().enumerate() {
            msix.lock().set_eventfd(v, e.try_clone().unwrap());
        }
        let sink = Arc::new(MockRouteSink::default());
        let pci = VirtioBlkPci::new(
            build_blk(),
            TEST_BAR_APERTURE,
            Arc::clone(&msix),
            Some(Arc::clone(&sink) as Arc<dyn MsixRouteSink>),
            test_gsis(),
        );
        (pci, msix, evts, sink)
    }

    /// Enable MSI-X via the real kernel MASKALL sequence (the way Linux's
    /// `msix_capability_init` does): assign vector 0 = config, vector 1 = the
    /// request queue; set Enable+MASKALL FIRST; program + per-vector-unmask each
    /// table entry while function-masked (so no route installs yet); then clear
    /// MASKALL — the function-unmask edge that installs every unmasked vector.
    /// Exercises the facade's function-mask reconcile edge, not just the
    /// cold-enable edge.
    fn enable_msix(pci: &mut VirtioBlkPci) {
        cc_w(pci, CC_MSIX_CONFIG, 0);
        cc_w(pci, CC_QUEUE_SELECT, 0);
        cc_w(pci, CC_QUEUE_MSIX_VECTOR, 1);
        pci.config_write(
            CAP_MSIX + MSIX_OFF_MSG_CTRL,
            &MC_ENABLE_MASKALL.to_le_bytes(),
        );
        for v in 0..N_VECTORS {
            let base = MSIX_TABLE_OFFSET + (v as u64) * MSIX_ENTRY_SIZE;
            pci.bar_write(base, &0xFEE0_0000u32.to_le_bytes()); // addr lo
            pci.bar_write(base + 8, &(0x4000u32 + v as u32).to_le_bytes()); // msg data
            pci.bar_write(base + 12, &0u32.to_le_bytes()); // per-vector unmask
        }
        // Clear MASKALL → function-unmask edge installs every unmasked vector.
        pci.config_write(CAP_MSIX + MSIX_OFF_MSG_CTRL, &MC_ENABLE.to_le_bytes());
    }

    /// The MASKALL-clear (function-unmask) edge installs every unmasked
    /// vector's route at its OWN GSI (40/41). Pins the Message-Control
    /// function-mask reconcile path that `msix_enable_and_unmask_installs_route`
    /// (cold per-vector unmask) does not cover.
    #[test]
    fn msix_function_mask_clear_installs_routes() {
        let (mut pci, _msix, _evts, sink) = build_msix();
        enable_msix(&mut pci);
        let installs = sink.installs.lock().unwrap();
        for gsi in [40u32, 41] {
            assert!(
                installs.iter().any(|(g, m)| *g == gsi && m.is_some()),
                "MASKALL-clear installs vector route at GSI {gsi}: {installs:?}"
            );
        }
    }

    /// A VRING interrupt to the request queue resolves to its vector (1) and
    /// fires that vector's eventfd; a masked vector records a PBA bit (no fire)
    /// and unmasking replays the pending interrupt exactly once. Drives
    /// `MsixState::signal` directly — the path the worker thread reaches through
    /// `signal_or_intx` (drain.rs) — the blk analog of the net facade's delivery
    /// suite (blk's signal originates on the worker thread, not a synchronous
    /// vCPU loopback, so the signal is driven directly here).
    #[test]
    fn msix_vring_signal_fires_pends_and_replays() {
        let (mut pci, msix, evts, _sink) = build_msix();
        enable_msix(&mut pci);
        // Live delivery: the request-queue vector (1) fires; config (0) quiet.
        msix.lock().signal(IrqSource::Vring { queue: 0 });
        assert_eq!(evts[1].read().unwrap(), 1, "req-queue vector 1 fired");
        assert!(
            evts[0].read().is_err(),
            "config vector 0 untouched (EAGAIN)"
        );
        assert_eq!(
            msix.lock().pba_byte(0),
            0,
            "delivered live, nothing pending"
        );
        // Mask vector 1 (table entry 1, vector-control bit0 = 1) → next signal pends.
        pci.bar_write(
            MSIX_TABLE_OFFSET + MSIX_ENTRY_SIZE + 12,
            &1u32.to_le_bytes(),
        );
        msix.lock().signal(IrqSource::Vring { queue: 0 });
        assert!(
            evts[1].read().is_err(),
            "masked vector did not fire (EAGAIN)"
        );
        assert_eq!(
            msix.lock().pba_byte(0) & (1 << 1),
            1 << 1,
            "pending bit set for the masked request-queue vector"
        );
        // Unmask → the facade replays the pending interrupt once.
        pci.bar_write(
            MSIX_TABLE_OFFSET + MSIX_ENTRY_SIZE + 12,
            &0u32.to_le_bytes(),
        );
        assert_eq!(
            evts[1].read().unwrap(),
            1,
            "unmask replays the pending interrupt"
        );
        assert_eq!(
            msix.lock().pba_byte(0) & (1 << 1),
            0,
            "pending cleared after replay"
        );
    }

    /// A config-change interrupt resolves to the config vector (0), leaving
    /// the request-queue vector quiet.
    #[test]
    fn msix_config_signal_fires_config_vector() {
        let (mut pci, msix, evts, _sink) = build_msix();
        enable_msix(&mut pci);
        msix.lock().signal(IrqSource::Config);
        assert_eq!(evts[0].read().unwrap(), 1, "config vector 0 fired");
        assert!(
            evts[1].read().is_err(),
            "req-queue vector 1 untouched (EAGAIN)"
        );
    }
}
