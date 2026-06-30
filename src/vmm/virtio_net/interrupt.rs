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
//! # Per-virtqueue vectors (EACH policy) with SHARED fallback
//!
//! The device advertises one MSI-X table entry per virtqueue plus one for
//! config (`num_queues + 1`, sized per device — see [`MsixState::new`] — and
//! capped at the host's per-NIC GSI budget). Linux's `virtio_pci_modern` driver
//! tries three vector policies in order (drivers/virtio/virtio_pci_common.c
//! `vp_find_vqs`): `VP_VQ_VECTOR_POLICY_EACH`, then
//! `VP_VQ_VECTOR_POLICY_SHARED_SLOW`, then `VP_VQ_VECTOR_POLICY_SHARED`.
//!
//! EACH requests one vector for the config-change interrupt plus one for every
//! virtqueue that has a callback (`vp_find_vqs_msix`: `nvectors = 1 +
//! count(vqs with name && callback)`). virtio-net's data virtqueues (RX/TX)
//! each have a callback, so for `N` queue-pairs EACH needs `2N + 1` vectors; the
//! control vq has NO callback, so `vp_find_one_vq_msix` assigns it
//! `VIRTIO_MSI_NO_VECTOR` (it is polled, not interrupt-driven). The device
//! advertises one entry per virtqueue plus config — `num_queues + 1`, which is
//! `2N + 2` under multiqueue (the control vq is present: `num_queues = 2N + 1`)
//! or `2N + 1` for the single-pair default (no control vq: `N = 1`,
//! `num_queues = 2` → 3 vectors). Either way EACH's `2N + 1` request fits:
//! config → vector 0, data vq `i` → vector `i + 1` (`allocated_vectors++` per
//! vq); under multiqueue the control vq's spare table entry is left unassigned,
//! and single-pair is an exact fit. Each data vq thus fires its OWN vector +
//! GSI — the per-node IRQ steering this device exists to exercise.
//!
//! [`MsixState::signal`] resolves an [`IrqSource::Vring`] to that vq's
//! `queue_vectors[queue]` entry, so RX, TX, and distinct queue-pairs deliver
//! independently. When a queue count exceeds the per-NIC vector budget, EACH
//! (and SHARED_SLOW) request more vectors than the table holds and fail; the
//! driver falls through to SHARED (`nvectors = 2`: config → vector 0, ALL
//! virtqueues → vector 1, virtio_pci_common.h). The per-queue signal path serves
//! SHARED transparently — every `queue_vectors[]` entry the guest programs
//! resolves to the same vector, so a coalesced VRING signal fires it and the
//! guest's shared-vector handler polls every vq mapped to it. Multiqueue is
//! offered only on this MSI-X transport (`VirtioNet::device_features` gates
//! `VIRTIO_NET_F_MQ` on `msix`), so every queue signal rides the one-shot
//! edge-MSI path.

use vmm_sys_util::eventfd::EventFd;

// `NUM_QUEUES` (the single-pair baseline) is referenced only by the unit tests
// below — the production state sizes `queue_vectors` from the per-device queue
// count passed to `new`.
#[cfg(test)]
use super::device::NUM_QUEUES;

/// MSI-X table-page vector capacity: the BAR0 MSI-X-table page (one 4 KiB page
/// in the PCI facade) holds `4096 / 16 = 256` 16-byte entries, so the device
/// can advertise at most this many vectors before the table would overflow its
/// page into the PBA page. The PCI facade static-asserts its table-page byte
/// span matches this. This is the page-capacity ceiling; the BINDING runtime
/// cap on a device's advertised vectors is `max_vectors` passed to
/// [`MsixState::new`] (the host's per-NIC MSI-X GSI budget,
/// `kvm::MSIX_VECTORS_PER_NIC`), which the kvm module const-asserts is `<=` this
/// ceiling. A queue count beyond the cap makes the guest fall back from per-vq
/// (EACH) to SHARED.
pub(crate) const MSIX_TABLE_MAX: usize = 256;

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
/// cloud-hypervisor's `VirtioInterruptType::{Queue, Config}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IrqSource {
    /// A used-ring publish on virtqueue `queue` (`VIRTIO_MMIO_INT_VRING`). The
    /// queue index selects that vq's guest-assigned MSI-X vector
    /// (`queue_vectors[queue]`) for per-queue IRQ steering; on the INTx path the
    /// device collapses every source onto its single shared line regardless.
    Vring { queue: usize },
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
    /// state — the host's delivery handle. Length = `num_vectors` (config +
    /// one per virtqueue), sized at construction.
    eventfds: Vec<Option<EventFd>>,
    /// Guest-visible MSI-X table: `[addr_lo, addr_hi, data, vector_ctrl]` per
    /// entry. Reset value per the PCI spec is masked (vector_ctrl bit 0 = 1).
    /// The address/data dwords are the verbatim MSI message the guest composed
    /// (x86 `__irq_msi_compose_msg`), which the KVM `KVM_IRQ_ROUTING_MSI` route
    /// carries unchanged — so the facade reads them here to build the route.
    /// Length = `num_vectors`, sized at construction.
    table: Vec<[u32; MSIX_ENTRY_DWORDS]>,
    /// Pending Bit Array: one bit per vector, set when a masked vector would
    /// have fired, replayed + cleared on unmask. Byte-addressed to match the
    /// guest-visible PBA register read; `ceil(num_vectors / 8)` bytes, sized at
    /// construction (per-vq MSI-X exceeds the 8 vectors one byte holds).
    pba: Vec<u8>,
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
    /// `NO_VECTOR` until programmed. Length is the device's virtqueue count
    /// (`2 * queue_pairs`, plus the control vq when multiqueue is offered),
    /// fixed at construction. Under EACH each entry holds a distinct vector
    /// (config + one per vq); under SHARED every entry resolves to the same
    /// vector (see the module doc). Either way the vector VALUES stay within
    /// `num_vectors` (the table / PBA / eventfd array length), so only this
    /// per-queue map scales with the queue count, not those arrays.
    queue_vectors: Vec<u16>,
}

impl MsixState {
    /// New state with every vector unprogrammed, every table entry masked (the
    /// PCI reset value), MSI-X disabled. `num_queues` is the device's virtqueue
    /// count — the length of the per-queue vector map ([`Self::queue_vectors`]).
    /// `max_vectors` caps the advertised vector count (the table / PBA / eventfd
    /// array length): the host's per-NIC MSI-X GSI budget
    /// (`kvm::MSIX_VECTORS_PER_NIC`), itself bounded by the table-page capacity
    /// [`MSIX_TABLE_MAX`].
    pub(crate) fn new(num_queues: usize, max_vectors: usize) -> Self {
        // One MSI-X vector per virtqueue plus one for config/NEEDS_RESET — the
        // table size the guest's EACH policy (vp_find_one_vq_msix) requests for
        // per-queue IRQ steering: config → vector 0, each vq → 1..=num_queues.
        // Capped at `max_vectors` (the host's per-NIC MSI-X GSI budget, ≤ the
        // BAR0 MSI-X-table page capacity): for a queue count within budget the
        // full per-vq table fits and the guest assigns each vq its own vector;
        // for a larger count the table is too small for EACH, so the guest falls
        // back to SHARED (all vqs → one vector) — the per-queue signal path
        // handles both transparently (in SHARED mode every queue_vectors[] entry
        // resolves to the same vector). This avoids a multi-page table without
        // capping queue_pairs.
        let num_vectors = (num_queues + 1).min(max_vectors);
        let mut table = vec![[0u32; MSIX_ENTRY_DWORDS]; num_vectors];
        for entry in &mut table {
            entry[VECTOR_CTRL_DWORD] = VECTOR_CTRL_MASK_BIT;
        }
        MsixState {
            eventfds: (0..num_vectors).map(|_| None).collect(),
            table,
            pba: vec![0u8; num_vectors.div_ceil(8)],
            enabled: false,
            function_mask: false,
            config_vector: NO_VECTOR,
            queue_vectors: vec![NO_VECTOR; num_queues],
        }
    }

    /// Number of MSI-X vectors the device advertises (`num_queues + 1`, capped at
    /// the `max_vectors` passed to [`Self::new`]): the table size driving the
    /// Message Control Table Size field and the per-vector GSI/eventfd count the
    /// host wiring allocates.
    pub(crate) fn num_vectors(&self) -> usize {
        self.table.len()
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
            // Per-queue routing: each virtqueue fires its OWN assigned vector
            // (the guest's EACH policy gives every vq a distinct vector). An
            // out-of-range queue index or an unassigned (NO_VECTOR) entry makes
            // fire() a no-op.
            IrqSource::Vring { queue } => {
                self.queue_vectors.get(queue).copied().unwrap_or(NO_VECTOR)
            }
        };
        self.fire(vector);
    }

    /// Fire `vector`'s eventfd if it is connected, enabled, and unmasked; else
    /// record a pending bit for replay on unmask. A `NO_VECTOR` or out-of-range
    /// vector is a no-op (the source is not connected to MSI-X).
    fn fire(&mut self, vector: u16) {
        if vector == NO_VECTOR {
            return;
        }
        let idx = vector as usize;
        if idx >= self.table.len() {
            // Clamped on the write path; defensive here so a stale/hostile
            // vector cannot index out of bounds.
            return;
        }
        if !self.enabled {
            // MSI-X disabled: the core would not have called us; defensive.
            return;
        }
        if self.function_mask || self.vector_masked(idx) {
            // Masked: record pending (PBA bit `idx`); replayed on unmask.
            self.pba[idx / 8] |= 1 << (idx % 8);
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
        if idx >= self.table.len() || self.pba[idx / 8] & (1 << (idx % 8)) == 0 {
            return false;
        }
        if self.function_mask || self.vector_masked(idx) || !self.enabled {
            return false;
        }
        // Clear the pending bit only AFTER a successful eventfd write: a failed
        // write (EBADF on teardown, or the implausible EAGAIN at u64 counter
        // saturation) then leaves the bit set so the interrupt is retried on the
        // next unmask edge — never dropping a pending the guest has not yet
        // observed. (Symmetric with fire(), which records the bit before any
        // write can fail.)
        if let Some(evt) = &self.eventfds[idx] {
            if let Err(e) = evt.write(1) {
                tracing::warn!(%e, vector = idx, "virtio-net MSI-X pending replay write failed");
                return false;
            }
            self.pba[idx / 8] &= !(1 << (idx % 8));
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
        // Keep the per-queue map's length (the device's queue count is fixed
        // at construction); reset every entry to unassigned.
        self.queue_vectors.fill(NO_VECTOR);
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

    /// The Pending Bit Array byte at `byte_off` (8 vectors per byte) for the
    /// guest-visible PBA register read. 0 for an out-of-range offset (a
    /// conformant guest reads only within the advertised PBA).
    pub(crate) fn pba_byte(&self, byte_off: usize) -> u8 {
        self.pba.get(byte_off).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    /// Single-pair MSI-X vector count under the EACH policy: config + one per
    /// virtqueue (RX vq0 + TX vq1), no control vq. `MsixState::new(NUM_QUEUES,
    /// _)` sizes the table to exactly this.
    const SINGLE_PAIR_VECTORS: usize = NUM_QUEUES + 1;

    /// Drain `evt`'s counter, returning how many writes it had accumulated. The
    /// eventfd is counter-mode (no EFD_SEMAPHORE), so one read returns the sum.
    fn drain(evt: &EventFd) -> u64 {
        evt.read().unwrap_or(0)
    }

    /// A single-pair state with one registered eventfd per vector, MSI-X
    /// enabled, the EACH assignment (config → 0, RX vq0 → 1, TX vq1 → 2 —
    /// DISTINCT per vq), every vector unmasked. `evts[v]` is the readable end of
    /// vector `v`'s eventfd.
    fn enabled_state() -> (MsixState, [EventFd; SINGLE_PAIR_VECTORS]) {
        let mut s = MsixState::new(NUM_QUEUES, MSIX_TABLE_MAX);
        let evts: [EventFd; SINGLE_PAIR_VECTORS] =
            std::array::from_fn(|_| EventFd::new(libc::EFD_NONBLOCK).unwrap());
        // Hand the state a dup of each fd so the test keeps its own readable end.
        for (v, evt) in evts.iter().enumerate() {
            s.set_eventfd(v, evt.try_clone().unwrap());
        }
        // EACH assignment: config → vector 0, data vq i → its OWN vector i + 1.
        s.set_config_vector(0);
        for q in 0..NUM_QUEUES {
            s.set_queue_vector(q, (q + 1) as u16);
        }
        // Program + unmask every table entry (addr/data then clear mask bit).
        for e in 0..SINGLE_PAIR_VECTORS {
            s.write_table_dword(e, 0, 0xFEE0_0000);
            s.write_table_dword(e, 2, 0x4000 + e as u32);
            assert!(s.write_table_dword(e, VECTOR_CTRL_DWORD, 0)); // unmask edge
        }
        s.set_message_control(0x8000); // Enable, function-mask clear
        (s, evts)
    }

    #[test]
    fn reset_state_is_masked_and_disabled() {
        let s = MsixState::new(NUM_QUEUES, MSIX_TABLE_MAX);
        assert_eq!(
            s.num_vectors(),
            SINGLE_PAIR_VECTORS,
            "config + one vector per virtqueue"
        );
        assert!(!s.enabled());
        assert_eq!(s.config_vector(), NO_VECTOR);
        for q in 0..NUM_QUEUES {
            assert_eq!(s.queue_vector(q), NO_VECTOR);
        }
        // Every table entry resets masked (vector control bit0 = 1).
        for e in 0..SINGLE_PAIR_VECTORS {
            assert_eq!(s.table_dword(e, VECTOR_CTRL_DWORD), VECTOR_CTRL_MASK_BIT);
            assert!(!s.vector_unmasked(e));
        }
        assert_eq!(s.pba_byte(0), 0);
    }

    #[test]
    fn num_vectors_is_num_queues_plus_one() {
        // Eight virtqueues (e.g. four queue-pairs, no control vq) → nine vectors.
        let s = MsixState::new(8, MSIX_TABLE_MAX);
        assert_eq!(s.num_vectors(), 9);
    }

    #[test]
    fn vector_count_caps_at_max_vectors() {
        // A queue count whose num_queues + 1 exceeds the cap: the table is sized
        // to the cap (the guest then falls back to SHARED across the queues),
        // while the per-queue map still tracks every queue.
        let s = MsixState::new(1000, 8);
        assert_eq!(s.num_vectors(), 8, "advertised vectors capped at max_vectors");
        assert_eq!(s.queue_vector(999), NO_VECTOR, "every queue still tracked");
    }

    #[test]
    fn per_queue_vectors_deliver_independently() {
        let (mut s, evts) = enabled_state();
        // Queue 0 (RX) fires its OWN vector 1; config + queue 1 untouched.
        s.signal(IrqSource::Vring { queue: 0 });
        assert_eq!(drain(&evts[1]), 1, "queue 0 fires its own vector 1");
        assert_eq!(drain(&evts[0]), 0, "config vector untouched");
        assert_eq!(drain(&evts[2]), 0, "queue 1's vector untouched");
        // Queue 1 (TX) fires its OWN DISTINCT vector 2.
        s.signal(IrqSource::Vring { queue: 1 });
        assert_eq!(drain(&evts[2]), 1, "queue 1 fires its own vector 2");
        assert_eq!(drain(&evts[1]), 0, "queue 0's vector untouched");
        // Config → vector 0.
        s.signal(IrqSource::Config);
        assert_eq!(drain(&evts[0]), 1, "config fires vector 0");
        assert_eq!(s.pba_byte(0), 0, "no pending while unmasked");
    }

    #[test]
    fn multi_pair_queues_deliver_to_distinct_vectors() {
        // Two queue-pairs + control vq: num_queues = 2*2 + 1 = 5, num_vectors = 6.
        // EACH assigns config→0 and each data vq its OWN vector (RX0→1, TX0→2,
        // RX1→3, TX1→4); the control vq (queue 4) has no callback → NO_VECTOR.
        // Pins the pair>0 queue-index→vector mapping (queue_vectors[2],[3] →
        // vectors 3,4) that the single-pair tests never exercise — the path an
        // off-by-one in per-pair indexing would slip through.
        let mut s = MsixState::new(5, MSIX_TABLE_MAX);
        assert_eq!(s.num_vectors(), 6, "config + one vector per vq (cvq slot spare)");
        // One eventfd per data vector (0..=4); vector 5 (the cvq spare) is unused.
        let evts: [EventFd; 5] =
            std::array::from_fn(|_| EventFd::new(libc::EFD_NONBLOCK).unwrap());
        for (v, evt) in evts.iter().enumerate() {
            s.set_eventfd(v, evt.try_clone().unwrap());
        }
        s.set_config_vector(0);
        // EACH: data vq i → vector i + 1; the control vq (queue 4) stays NO_VECTOR.
        for q in 0..4 {
            s.set_queue_vector(q, (q + 1) as u16);
        }
        for e in 0..5 {
            s.write_table_dword(e, 0, 0xFEE0_0000);
            s.write_table_dword(e, 2, 0x4000 + e as u32);
            assert!(s.write_table_dword(e, VECTOR_CTRL_DWORD, 0));
        }
        s.set_message_control(0x8000); // Enable

        // Pair 1's RX (queue 2) fires ONLY vector 3 — distinct from pair 0.
        s.signal(IrqSource::Vring { queue: 2 });
        assert_eq!(drain(&evts[3]), 1, "queue 2 (RX pair 1) fires vector 3");
        for (v, evt) in evts.iter().enumerate() {
            if v != 3 {
                assert_eq!(drain(evt), 0, "queue 2 fired only vector 3, not {v}");
            }
        }
        // Pair 1's TX (queue 3) fires ONLY vector 4.
        s.signal(IrqSource::Vring { queue: 3 });
        assert_eq!(drain(&evts[4]), 1, "queue 3 (TX pair 1) fires vector 4");
        for (v, evt) in evts.iter().enumerate() {
            if v != 4 {
                assert_eq!(drain(evt), 0, "queue 3 fired only vector 4, not {v}");
            }
        }
        // Pair 0 still maps to its OWN vectors 1,2 (not pair 1's).
        s.signal(IrqSource::Vring { queue: 0 });
        assert_eq!(drain(&evts[1]), 1, "queue 0 (RX pair 0) fires vector 1");
        s.signal(IrqSource::Vring { queue: 1 });
        assert_eq!(drain(&evts[2]), 1, "queue 1 (TX pair 0) fires vector 2");
        // The control vq (queue 4) has NO_VECTOR → no-op (polled, not IRQ-driven).
        s.signal(IrqSource::Vring { queue: 4 });
        for (v, evt) in evts.iter().enumerate() {
            assert_eq!(drain(evt), 0, "control vq delivers no MSI-X (vector {v})");
        }
        assert_eq!(s.pba_byte(0), 0, "all delivered live, nothing pending");
    }

    #[test]
    fn masked_vector_records_pending_then_replays() {
        let (mut s, evts) = enabled_state();
        // Re-mask queue 0's vector (table entry 1, vector control bit0 = 1).
        assert!(!s.write_table_dword(1, VECTOR_CTRL_DWORD, VECTOR_CTRL_MASK_BIT));
        s.signal(IrqSource::Vring { queue: 0 });
        assert_eq!(drain(&evts[1]), 0, "masked vector does not fire");
        assert_eq!(s.pba_byte(0) & (1 << 1), 1 << 1, "pending bit set for vector 1");
        // Unmask → replay fires once, pending clears.
        let unmasked = s.write_table_dword(1, VECTOR_CTRL_DWORD, 0);
        assert!(unmasked, "clearing mask bit is an unmask edge");
        assert!(s.replay_pending(1), "pending replays on unmask");
        assert_eq!(drain(&evts[1]), 1, "replay delivered exactly once");
        assert_eq!(s.pba_byte(0) & (1 << 1), 0, "pending cleared after replay");
        assert!(!s.replay_pending(1), "no double replay");
    }

    #[test]
    fn function_mask_suppresses_all_vectors() {
        let (mut s, evts) = enabled_state();
        s.set_message_control(0x8000 | 0x4000); // Enable + Function Mask
        s.signal(IrqSource::Vring { queue: 0 });
        s.signal(IrqSource::Vring { queue: 1 });
        s.signal(IrqSource::Config);
        for (v, evt) in evts.iter().enumerate() {
            assert_eq!(drain(evt), 0, "vector {v} suppressed under function mask");
        }
        assert_eq!(
            s.pba_byte(0),
            (1 << 0) | (1 << 1) | (1 << 2),
            "config + both queue vectors pending under function mask"
        );
    }

    #[test]
    fn no_vector_assignment_is_silent() {
        let (mut s, evts) = enabled_state();
        // Disconnect every queue vector.
        for q in 0..NUM_QUEUES {
            s.set_queue_vector(q, NO_VECTOR);
        }
        s.signal(IrqSource::Vring { queue: 0 });
        s.signal(IrqSource::Vring { queue: 1 });
        assert_eq!(drain(&evts[1]), 0, "unassigned queue 0 delivers nothing");
        assert_eq!(drain(&evts[2]), 0, "unassigned queue 1 delivers nothing");
        assert_eq!(s.pba_byte(0), 0, "and records no pending");
    }

    #[test]
    fn disabled_state_does_not_fire() {
        let (mut s, evts) = enabled_state();
        s.set_message_control(0); // clear Enable
        assert!(!s.enabled());
        s.signal(IrqSource::Vring { queue: 0 });
        assert_eq!(drain(&evts[1]), 0);
        assert_eq!(s.pba_byte(0), 0);
    }

    #[test]
    fn msi_message_reads_programmed_dwords() {
        let (s, _evts) = enabled_state();
        assert_eq!(s.msi_message(0), Some((0xFEE0_0000, 0, 0x4000)));
        assert_eq!(s.msi_message(1), Some((0xFEE0_0000, 0, 0x4001)));
        assert_eq!(s.msi_message(2), Some((0xFEE0_0000, 0, 0x4002)));
        assert_eq!(s.msi_message(SINGLE_PAIR_VECTORS), None, "out of range");
    }

    #[test]
    fn out_of_range_vector_is_inert() {
        let mut s = MsixState::new(NUM_QUEUES, MSIX_TABLE_MAX);
        s.set_message_control(0x8000);
        s.set_config_vector(99); // would be clamped by the facade; defensive here
        s.signal(IrqSource::Config);
        assert_eq!(s.pba_byte(0), 0, "out-of-range vector neither fires nor pends");
    }

    #[test]
    fn eventfd_replace_returns_prior() {
        let mut s = MsixState::new(NUM_QUEUES, MSIX_TABLE_MAX);
        let a = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        let b = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        let a_raw = a.as_raw_fd();
        assert!(s.set_eventfd(0, a).is_none(), "first install has no prior");
        let prior = s.set_eventfd(0, b).expect("second install returns prior");
        assert_eq!(prior.as_raw_fd(), a_raw, "prior eventfd returned");
        assert!(
            s.set_eventfd(SINGLE_PAIR_VECTORS, EventFd::new(0).unwrap())
                .is_none(),
            "OOB index"
        );
    }
}
