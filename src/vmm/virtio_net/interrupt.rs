//! MSI-X interrupt delivery for the virtio-net device.
//!
//! The transport-neutral core ([`super::device::VirtioNet`]) signals interrupts
//! by emitting an abstract [`IrqSource`] (a virtqueue used-ring publish or a
//! config-change). On the INTx path the core writes its single `irq_evt` line
//! directly (unchanged for virtio-MMIO and aarch64). When the guest enables
//! MSI-X on the PCI transport, the core instead routes the signal through this
//! module's [`MsixState`], which resolves the source to a guest-assigned vector,
//! gates on the per-vector + function masks, and either fires that vector's
//! irqfd-registered eventfd or records a Pending-Bit-Array bit for replay on
//! unmask.
//!
//! Ownership mirrors cloud-hypervisor's device→`Arc<dyn VirtioInterrupt>`
//! boundary (the device core never sees a vector number, eventfd, or GSI) but
//! adapts it to ktstr's structure: ktstr's PCI facade OWNS the device core (the
//! core is a field of the facade), so an interrupt object injected into the
//! device at activate-time does not fit. Instead the delivery state lives in an
//! `Arc<PiMutex<MsixState>>` SHARED between the core (which holds a clone and
//! calls [`MsixState::signal`] at the trigger site) and the PCI facade (which
//! holds a clone, mutates it from config-space decode, and owns the KVM
//! GSI-route side: the per-vector GSIs and the route owner —
//! `FullIrqchipRouteOwner` on full-irqchip, `IoapicHandle` on split). The
//! gate stays at the trigger site (not a post-hoc drain), matching both
//! cloud-hypervisor (`VirtioInterruptMsix::trigger`) and qemu (`msix_notify`).
//! The mutex is uncontended in practice: every access — facade config writes and
//! core signals alike — is on the vCPU thread under the `PciBus` lock.
//!
//! # Single-vector SHARED mode (v0)
//!
//! v0 advertises [`MSIX_VECTORS`] = 2 table entries. Linux's `virtio_pci_modern`
//! driver's `vp_find_vqs` tries three vector policies in order
//! (drivers/virtio/virtio_pci_common.c): `VP_VQ_VECTOR_POLICY_EACH` (one vector
//! per virtqueue + one for config = 3 for net's config + RX + TX) and
//! `VP_VQ_VECTOR_POLICY_SHARED_SLOW` (per-vq for the data vqs, shared only for a
//! slow-path/admin vq — net has none, so this degenerates to per-vq and also
//! needs 3) BOTH request 3 vectors and so fail against a 2-entry table; it then
//! falls through to `VP_VQ_VECTOR_POLICY_SHARED` (`nvectors = 2`): config →
//! `VP_MSIX_CONFIG_VECTOR` (0), ALL virtqueues → `VP_MSIX_VQ_VECTOR` (1)
//! (virtio_pci_common.h). So both queues map to one shared queue vector, and a
//! coalesced VRING signal resolves to that single vector — there is no need to
//! distinguish RX from TX. Per-virtqueue vectors (Table Size ≥ 3, faithful
//! per-node IRQ steering) are a follow-up that also raises `NUM_QUEUES`; the
//! `NUM_QUEUES == 2` static guard in the PCI facade gates that work.

use vmm_sys_util::eventfd::EventFd;

use super::device::NUM_QUEUES;

/// MSI-X table entries the device advertises (Table Size = `MSIX_VECTORS`,
/// encoded as `MSIX_VECTORS - 1` in Message Control). Two: one for the shared
/// virtqueue interrupt + one for the config/NEEDS_RESET interrupt. Single-queue
/// v0; per-virtqueue MSI-X raises this (and `NUM_QUEUES`) together.
pub(crate) const MSIX_VECTORS: usize = 2;

/// Dwords per MSI-X table entry (PCI spec §6.8.2): message address lo, message
/// address hi, message data, vector control.
pub(crate) const MSIX_ENTRY_DWORDS: usize = 4;

/// Dword index of the Vector Control register within a table entry; bit 0 is the
/// per-vector mask.
const VECTOR_CTRL_DWORD: usize = 3;
/// Vector Control bit 0 — the per-vector mask (PCI spec §6.8.2.9).
const VECTOR_CTRL_MASK_BIT: u32 = 0x1;

/// `VIRTIO_MSI_NO_VECTOR` (virtio-v1.2 §4.1.5.1.2): a source whose MSI-X vector
/// register holds this value is not connected to MSI-X. The guest writes it to
/// disable a source's MSI-X delivery, and the device reset value is this.
pub(crate) const NO_VECTOR: u16 = 0xFFFF;

/// An abstract interrupt source the device core emits. The transport resolves it
/// to an INTx line (the core's `irq_evt`) or an MSI-X vector. Mirrors
/// cloud-hypervisor's `VirtioInterruptType::{Queue, Config}`; v0's coalesced
/// VRING signal carries no queue index (SHARED mode collapses all queues to one
/// vector — see the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IrqSource {
    /// A virtqueue used-ring publish (`VIRTIO_MMIO_INT_VRING`).
    Vring,
    /// A device config-change / NEEDS_RESET (`VIRTIO_MMIO_INT_CONFIG`).
    Config,
}

/// Abstraction over the host's KVM GSI-route programming. Implemented x86-only
/// by `FullIrqchipRouteOwner` (full-irqchip) and `IoapicHandle` (split-irqchip);
/// kept a trait so the (arch-neutral) PCI facade holds
/// `Option<Arc<dyn MsixRouteSink>>` without referencing the x86-only KVM route
/// types. The facade calls this on a vector mask/unmask edge to (re)install or
/// remove the MSI route for a GSI.
pub(crate) trait MsixRouteSink: Send + Sync {
    /// Install (`Some((address_lo, address_hi, data))`) or remove (`None`) the
    /// MSI route for `gsi`, rebuilding the full GSI routing table. Errors are
    /// recorded by the implementation (a failed-install counter surfaced at
    /// teardown via the concrete owner) rather than returned: the caller is a
    /// config-space MMIO write with no error channel back to the guest.
    fn set_route(&self, gsi: u32, msg: Option<(u32, u32, u32)>);
}

/// MSI-X delivery state for one virtio-net device: the guest-visible table
/// entries + vector assignments + Pending Bit Array, the device-level enable /
/// function-mask, and the per-vector irqfd-registered eventfds. Shared via
/// `Arc<PiMutex<…>>` between the device core (signals) and the PCI facade
/// (configures); see the module doc. The KVM GSI routes and the per-vector GSIs
/// live in the facade, not here — this struct owns only what the *device* needs
/// to fire (eventfds) plus the guest-visible register state.
pub(crate) struct MsixState {
    /// Per-vector irqfd eventfds. Index = MSI-X table entry. `None` until the
    /// host wiring registers the eventfd with KVM and hands it here; the device
    /// writes it to fire the vector. Not part of the guest-visible register
    /// state — the host's delivery handle.
    eventfds: [Option<EventFd>; MSIX_VECTORS],
    /// Guest-visible MSI-X table: `[addr_lo, addr_hi, data, vector_ctrl]` per
    /// entry. Reset value per the PCI spec is masked (vector_ctrl bit 0 = 1).
    /// The address/data dwords are the verbatim MSI message the guest composed
    /// (x86 `__irq_msi_compose_msg`), which the KVM `KVM_IRQ_ROUTING_MSI` route
    /// carries unchanged — so the facade reads them here to build the route.
    table: [[u32; MSIX_ENTRY_DWORDS]; MSIX_VECTORS],
    /// Pending Bit Array: one bit per vector, set when a masked vector would
    /// have fired, replayed + cleared on unmask. `MSIX_VECTORS` ≤ 8 fits one
    /// byte (the guest-visible PBA register).
    pba: u8,
    /// Device-level MSI-X Enable (Message Control bit 15). `false` ⇒ the device
    /// delivers via INTx; the core consults [`Self::enabled`] to choose.
    enabled: bool,
    /// Device-level Function Mask (Message Control bit 14): masks ALL vectors
    /// regardless of per-vector mask (PCI spec §6.8.2.9).
    function_mask: bool,
    /// Guest-assigned config-change vector (`CC_MSIX_CONFIG`), `NO_VECTOR` until
    /// the guest programs it.
    config_vector: u16,
    /// Guest-assigned per-queue vectors (`CC_QUEUE_MSIX_VECTOR` per queue),
    /// `NO_VECTOR` until programmed. In SHARED mode every entry is the same
    /// vector (see the module doc).
    queue_vectors: [u16; NUM_QUEUES],
}

impl MsixState {
    /// New state with every vector unprogrammed, every table entry masked (the
    /// PCI reset value), MSI-X disabled.
    pub(crate) fn new() -> Self {
        let mut table = [[0u32; MSIX_ENTRY_DWORDS]; MSIX_VECTORS];
        for entry in &mut table {
            entry[VECTOR_CTRL_DWORD] = VECTOR_CTRL_MASK_BIT;
        }
        MsixState {
            eventfds: std::array::from_fn(|_| None),
            table,
            pba: 0,
            enabled: false,
            function_mask: false,
            config_vector: NO_VECTOR,
            queue_vectors: [NO_VECTOR; NUM_QUEUES],
        }
    }

    /// True once the guest has set MSI-X Enable. The device core routes through
    /// MSI-X only then; otherwise it uses INTx.
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Deliver `source`: resolve to the guest-assigned vector, then fire its
    /// eventfd if deliverable, else record a pending bit. Called at the trigger
    /// site from the device core (cloud-hypervisor `VirtioInterruptMsix::trigger`
    /// / qemu `msix_notify`).
    pub(crate) fn signal(&mut self, source: IrqSource) {
        let vector = match source {
            IrqSource::Config => self.config_vector,
            IrqSource::Vring => self.vring_vector(),
        };
        self.fire(vector);
    }

    /// The shared virtqueue vector: in SHARED mode every queue maps to the same
    /// vector, so a coalesced VRING signal resolves to the first connected queue
    /// vector. `NO_VECTOR` if no queue has been assigned a vector (delivery is
    /// then a no-op, matching the guest's choice not to connect the source).
    fn vring_vector(&self) -> u16 {
        self.queue_vectors
            .iter()
            .copied()
            .find(|&v| v != NO_VECTOR)
            .unwrap_or(NO_VECTOR)
    }

    /// Fire `vector`'s eventfd if it is connected, enabled, and unmasked; else
    /// record a pending bit for replay on unmask. A `NO_VECTOR` or out-of-range
    /// vector is a no-op (the source is not connected to MSI-X).
    fn fire(&mut self, vector: u16) {
        if vector == NO_VECTOR {
            return;
        }
        let idx = vector as usize;
        if idx >= MSIX_VECTORS {
            // Clamped on the write path; defensive here so a stale/hostile
            // vector cannot index out of bounds.
            return;
        }
        if !self.enabled {
            // MSI-X disabled: the core would not have called us; defensive.
            return;
        }
        if self.function_mask || self.vector_masked(idx) {
            // Masked: record pending; replayed by `replay_pending` on unmask.
            self.pba |= 1 << idx;
            return;
        }
        if let Some(evt) = &self.eventfds[idx]
            && let Err(e) = evt.write(1)
        {
            tracing::warn!(%e, vector = idx, "virtio-net MSI-X eventfd write failed");
        }
    }

    /// Replay a vector that became deliverable on an unmask edge: if its pending
    /// bit is set and it is now deliverable, fire it once and clear the bit.
    /// Returns whether a delivery occurred. The facade calls this AFTER
    /// installing the GSI route for the unmasked vector, so the replayed write
    /// has a route to deliver through.
    pub(crate) fn replay_pending(&mut self, idx: usize) -> bool {
        if idx >= MSIX_VECTORS || self.pba & (1 << idx) == 0 {
            return false;
        }
        if self.function_mask || self.vector_masked(idx) || !self.enabled {
            return false;
        }
        self.pba &= !(1 << idx);
        if let Some(evt) = &self.eventfds[idx] {
            if let Err(e) = evt.write(1) {
                tracing::warn!(%e, vector = idx, "virtio-net MSI-X pending replay write failed");
                return false;
            }
            return true;
        }
        false
    }

    /// Per-vector mask bit (Vector Control bit 0) for table entry `idx`.
    fn vector_masked(&self, idx: usize) -> bool {
        self.table
            .get(idx)
            .is_none_or(|e| e[VECTOR_CTRL_DWORD] & VECTOR_CTRL_MASK_BIT != 0)
    }

    // ----- facade-driven configuration (config-space decode) -----

    /// Apply a Message Control write (cap register): Enable (bit 15) and
    /// Function Mask (bit 14). The facade reconciles every vector's KVM route
    /// after this call (install the deliverable, remove the rest), so no
    /// prior-state edge detection is needed here.
    pub(crate) fn set_message_control(&mut self, msg_ctrl: u16) {
        self.enabled = msg_ctrl & 0x8000 != 0;
        self.function_mask = msg_ctrl & 0x4000 != 0;
    }

    /// Set the config-change vector (`CC_MSIX_CONFIG`); `vector` is pre-clamped
    /// by the facade to a valid index or `NO_VECTOR`.
    pub(crate) fn set_config_vector(&mut self, vector: u16) {
        self.config_vector = vector;
    }

    /// Read back the config-change vector (the kernel's `vp_modern` aborts if the
    /// read-back differs from the written vector).
    pub(crate) fn config_vector(&self) -> u16 {
        self.config_vector
    }

    /// Set queue `queue`'s vector (`CC_QUEUE_MSIX_VECTOR`); pre-clamped by the
    /// facade. Out-of-range `queue` is dropped (hostile `queue_select`).
    pub(crate) fn set_queue_vector(&mut self, queue: usize, vector: u16) {
        if let Some(slot) = self.queue_vectors.get_mut(queue) {
            *slot = vector;
        }
    }

    /// Read back queue `queue`'s vector, `NO_VECTOR` for an out-of-range queue.
    pub(crate) fn queue_vector(&self, queue: usize) -> u16 {
        self.queue_vectors.get(queue).copied().unwrap_or(NO_VECTOR)
    }

    /// Virtio reset: restore the per-queue + config vector assignments to their
    /// reset value (`NO_VECTOR`, virtio-v1.2 §4.1.4.3). Does NOT touch the
    /// PCI-level MSI-X cap (enable, function mask, table, PBA, registered
    /// eventfds) — those reset only on PCI/FLR reset, not a virtio status reset.
    pub(crate) fn reset_virtio_assignments(&mut self) {
        self.config_vector = NO_VECTOR;
        self.queue_vectors = [NO_VECTOR; NUM_QUEUES];
    }

    /// Read a table dword `(entry, dword)`; 0 for out-of-range (a conformant
    /// guest never reads past the advertised table; the bound keeps a hostile
    /// read in-bounds).
    pub(crate) fn table_dword(&self, entry: usize, dword: usize) -> u32 {
        self.table
            .get(entry)
            .and_then(|e| e.get(dword))
            .copied()
            .unwrap_or(0)
    }

    /// Write a table dword `(entry, dword)`. Returns `true` iff this write
    /// cleared the Vector Control mask bit (a mask→unmask edge on `entry`), so
    /// the facade installs the route + replays any pending bit. A write that
    /// SETs the mask bit, or any other dword, returns `false`. Out-of-range is
    /// dropped (the bound keeps a hostile write in-bounds).
    pub(crate) fn write_table_dword(&mut self, entry: usize, dword: usize, val: u32) -> bool {
        let Some(slot) = self.table.get_mut(entry).and_then(|e| e.get_mut(dword)) else {
            return false;
        };
        // Only the Vector Control dword carries the mask bit, so only it can
        // produce a mask→unmask edge; addr/data dwords never change mask state.
        if dword == VECTOR_CTRL_DWORD {
            let was_masked = *slot & VECTOR_CTRL_MASK_BIT != 0;
            *slot = val;
            let now_masked = val & VECTOR_CTRL_MASK_BIT != 0;
            was_masked && !now_masked
        } else {
            *slot = val;
            false
        }
    }

    /// True iff table entry `idx` is currently unmasked (per-vector AND function
    /// mask both clear). The facade uses this to decide which vectors to route
    /// on a function-mask / enable edge.
    pub(crate) fn vector_unmasked(&self, idx: usize) -> bool {
        !self.function_mask && !self.vector_masked(idx)
    }

    /// The MSI message `(address_lo, address_hi, data)` programmed into table
    /// entry `idx` — the verbatim route the facade installs via
    /// `KVM_IRQ_ROUTING_MSI`. `None` for an out-of-range entry.
    pub(crate) fn msi_message(&self, idx: usize) -> Option<(u32, u32, u32)> {
        self.table.get(idx).map(|e| (e[0], e[1], e[2]))
    }

    /// Install the irqfd eventfd for vector `idx` (the host wiring registered it
    /// with KVM). Returns the prior eventfd, if any.
    pub(crate) fn set_eventfd(&mut self, idx: usize, evt: EventFd) -> Option<EventFd> {
        let slot = self.eventfds.get_mut(idx)?;
        slot.replace(evt)
    }

    /// The Pending Bit Array byte (one bit per vector) for the guest-visible PBA
    /// register read.
    pub(crate) fn pba_byte(&self) -> u8 {
        self.pba
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    /// Drain `evt`'s counter, returning how many writes it had accumulated. The
    /// eventfd is counter-mode (no EFD_SEMAPHORE), so one read returns the sum.
    fn drain(evt: &EventFd) -> u64 {
        evt.read().unwrap_or(0)
    }

    /// A state with two registered eventfds, MSI-X enabled, config→0, queues→1
    /// (the SHARED-mode assignment), both vectors unmasked.
    fn enabled_state() -> (MsixState, EventFd, EventFd) {
        let mut s = MsixState::new();
        let v0 = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        let v1 = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        // Hand the state a dup of each fd so the test keeps its own readable end.
        s.set_eventfd(0, v0.try_clone().unwrap());
        s.set_eventfd(1, v1.try_clone().unwrap());
        s.set_config_vector(0);
        for q in 0..NUM_QUEUES {
            s.set_queue_vector(q, 1);
        }
        // Program + unmask both table entries (addr/data then clear mask bit).
        for e in 0..MSIX_VECTORS {
            s.write_table_dword(e, 0, 0xFEE0_0000);
            s.write_table_dword(e, 2, 0x4000 + e as u32);
            assert!(s.write_table_dword(e, VECTOR_CTRL_DWORD, 0)); // unmask edge
        }
        s.set_message_control(0x8000); // Enable, function-mask clear
        (s, v0, v1)
    }

    #[test]
    fn reset_state_is_masked_and_disabled() {
        let s = MsixState::new();
        assert!(!s.enabled());
        assert_eq!(s.config_vector(), NO_VECTOR);
        assert_eq!(s.queue_vector(0), NO_VECTOR);
        // Every table entry resets masked (vector control bit0 = 1).
        for e in 0..MSIX_VECTORS {
            assert_eq!(s.table_dword(e, VECTOR_CTRL_DWORD), VECTOR_CTRL_MASK_BIT);
            assert!(!s.vector_unmasked(e));
        }
        assert_eq!(s.pba_byte(), 0);
    }

    #[test]
    fn signal_fires_assigned_vector() {
        let (mut s, v0, v1) = enabled_state();
        // VRING → shared queue vector 1.
        s.signal(IrqSource::Vring);
        assert_eq!(drain(&v1), 1, "VRING fires queue vector 1");
        assert_eq!(drain(&v0), 0, "config vector untouched");
        // Config → vector 0.
        s.signal(IrqSource::Config);
        assert_eq!(drain(&v0), 1, "Config fires config vector 0");
        assert_eq!(s.pba_byte(), 0, "no pending while unmasked");
    }

    #[test]
    fn masked_vector_records_pending_then_replays() {
        let (mut s, _v0, v1) = enabled_state();
        // Re-mask the queue vector (entry 1, vector control bit0 = 1).
        assert!(!s.write_table_dword(1, VECTOR_CTRL_DWORD, VECTOR_CTRL_MASK_BIT));
        s.signal(IrqSource::Vring);
        assert_eq!(drain(&v1), 0, "masked vector does not fire");
        assert_eq!(s.pba_byte() & (1 << 1), 1 << 1, "pending bit set for vector 1");
        // Unmask → replay fires once, pending clears.
        let unmasked = s.write_table_dword(1, VECTOR_CTRL_DWORD, 0);
        assert!(unmasked, "clearing mask bit is an unmask edge");
        assert!(s.replay_pending(1), "pending replays on unmask");
        assert_eq!(drain(&v1), 1, "replay delivered exactly once");
        assert_eq!(s.pba_byte() & (1 << 1), 0, "pending cleared after replay");
        assert!(!s.replay_pending(1), "no double replay");
    }

    #[test]
    fn function_mask_suppresses_all_vectors() {
        let (mut s, v0, v1) = enabled_state();
        s.set_message_control(0x8000 | 0x4000); // Enable + Function Mask
        s.signal(IrqSource::Vring);
        s.signal(IrqSource::Config);
        assert_eq!(drain(&v0), 0);
        assert_eq!(drain(&v1), 0);
        assert_eq!(s.pba_byte(), (1 << 0) | (1 << 1), "both pending under function mask");
    }

    #[test]
    fn no_vector_assignment_is_silent() {
        let (mut s, _v0, v1) = enabled_state();
        // Disconnect every queue vector.
        for q in 0..NUM_QUEUES {
            s.set_queue_vector(q, NO_VECTOR);
        }
        s.signal(IrqSource::Vring);
        assert_eq!(drain(&v1), 0, "unassigned source delivers nothing");
        assert_eq!(s.pba_byte(), 0, "and records no pending");
    }

    #[test]
    fn disabled_state_does_not_fire() {
        let (mut s, _v0, v1) = enabled_state();
        s.set_message_control(0); // clear Enable
        assert!(!s.enabled());
        s.signal(IrqSource::Vring);
        assert_eq!(drain(&v1), 0);
        assert_eq!(s.pba_byte(), 0);
    }

    #[test]
    fn msi_message_reads_programmed_dwords() {
        let (s, _v0, _v1) = enabled_state();
        assert_eq!(s.msi_message(0), Some((0xFEE0_0000, 0, 0x4000)));
        assert_eq!(s.msi_message(1), Some((0xFEE0_0000, 0, 0x4001)));
        assert_eq!(s.msi_message(MSIX_VECTORS), None, "out of range");
    }

    #[test]
    fn out_of_range_vector_is_inert() {
        let mut s = MsixState::new();
        s.set_message_control(0x8000);
        s.set_config_vector(99); // would be clamped by the facade; defensive here
        s.signal(IrqSource::Config);
        assert_eq!(s.pba_byte(), 0, "out-of-range vector neither fires nor pends");
    }

    #[test]
    fn eventfd_replace_returns_prior() {
        let mut s = MsixState::new();
        let a = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        let b = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        let a_raw = a.as_raw_fd();
        assert!(s.set_eventfd(0, a).is_none(), "first install has no prior");
        let prior = s.set_eventfd(0, b).expect("second install returns prior");
        assert_eq!(prior.as_raw_fd(), a_raw, "prior eventfd returned");
        assert!(s.set_eventfd(MSIX_VECTORS, EventFd::new(0).unwrap()).is_none(), "OOB index");
    }
}
