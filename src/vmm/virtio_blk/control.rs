//! Virtio-blk transport-neutral core: the register reader / FSM-writer
//! ops the transport facades (the virtio-MMIO `mmio.rs`, and any future
//! PCI facade) decode onto, the device-status FSM, the device-config
//! read (`read_blk_config`), and the reset / worker-respawn / pause
//! lifecycle. Split from device.rs; the `impl VirtioBlk` block reaches
//! the device struct, its pub(crate) fields, constants, and crate
//! imports via `use super::*`, and the worker stop/join helpers via the
//! `super::*` re-glob. The transport-specific register decode lives in
//! the facade modules, not here, so the two transports cannot drift.
use super::*;
// MMIO/FSM dispatch compiles in all builds. The spawned-worker reset/respawn
// methods are `#[cfg(not(test))]`; the helpers they alone need — the worker
// entry point and std::thread — are gated to match (worker_thread_main isn't
// reached through the re-glob, and std imports aren't re-exported). The
// lifecycle join helpers arrive via the `super::*` re-glob (mod.rs
// `pub(crate) use lifecycle::*`), so no explicit lifecycle import here.
#[cfg(not(test))]
use super::worker::worker_thread_main;
#[cfg(not(test))]
use std::thread;

// `DrainOutcome` and `drain_bracket_impl` live in `drain.rs`; reach them
// via the `super::*;` glob (sourced from `mod.rs`'s
// `pub(crate) use drain::*;`). Pulled out for module locality so the
// chain-validation/throttle/handler-dispatch/completion-publish pipeline
// sits in one file beside its tests.

impl VirtioBlk {
    // The four `handle_*_impl` per-request-type handlers (T_IN /
    // T_OUT / T_FLUSH / T_GET_ID) and their `cfg(test)` `&self`
    // wrappers live in `handlers.rs` as a separate `impl VirtioBlk`
    // block. Pulled out for module locality so the per-request
    // logic sits beside its tests; this impl block continues with
    // the transport-neutral core register/FSM ops (decoded onto by
    // the facade modules) + the reset/respawn/pause lifecycle.

    /// Read from block config space. virtio-v1.2 §5.2.4 layout, mirrored
    /// in [`VirtioBlkConfig`]:
    ///   - 0x00..0x08: capacity (u64 LE, sectors) — always
    ///   - 0x08..0x0C: size_max (u32 LE) — VIRTIO_BLK_F_SIZE_MAX
    ///   - 0x0C..0x10: seg_max (u32 LE) — VIRTIO_BLK_F_SEG_MAX
    ///   - 0x10..0x14: geometry (4 bytes) — VIRTIO_BLK_F_GEOMETRY (zero;
    ///     feature bit not advertised)
    ///   - 0x14..0x18: blk_size (u32 LE) — VIRTIO_BLK_F_BLK_SIZE
    ///
    /// Reads at offsets `>= VIRTIO_BLK_CONFIG_SIZE` return zero per
    /// virtio-v1.2 §4.2.2.2 ("reads past the populated config layout
    /// return zero") — guarded fields like topology / MQ / discard
    /// have feature bits we don't advertise, so the kernel driver's
    /// `virtio_cread_feature` skips them and never observes the
    /// zero-bytes we serve.
    pub(crate) fn read_blk_config(&self, offset: u64, data: &mut [u8]) {
        let cfg = VirtioBlkConfig {
            capacity: self.capacity_sectors,
            size_max: VIRTIO_BLK_SIZE_MAX,
            seg_max: VIRTIO_BLK_SEG_MAX,
            geometry: VirtioBlkGeometry::default(),
            blk_size: VIRTIO_BLK_SECTOR_SIZE,
        };
        // `as_slice()` returns the struct's wire-format byte
        // representation directly — `repr(C, packed)` guarantees no
        // padding and host-LE u32/u64 stores match the virtio LE wire
        // format on the supported (x86_64, aarch64) hosts. See
        // ByteValued impl SAFETY note above.
        let cfg_bytes = cfg.as_slice();
        let len = data.len();
        let start = offset as usize;
        if start >= cfg_bytes.len() {
            data.fill(0);
            return;
        }
        let end = (start + len).min(cfg_bytes.len());
        let n = end - start;
        data[..n].copy_from_slice(&cfg_bytes[start..end]);
        data[n..].fill(0);
    }

    // =================== transport-neutral core API ===================
    //
    // The register reader / FSM-writer ops the transport facades decode
    // onto. Each facade (the virtio-MMIO `mmio.rs`, and any future PCI
    // facade) owns ONLY the transport-specific decode — which offset /
    // cap field maps to which op, and the access width; ALL device
    // behaviour (the status FSM, feature negotiation, queue ring
    // assembly, notify, interrupt bookkeeping) lives here exactly once,
    // so the transports cannot drift. The queue-config / feature-lock
    // gates live INSIDE the gated ops (a write to a gated register is a
    // no-op), so a facade's decode `match` carries no gates.

    /// Offered device features for the latched select window
    /// (`device_features_sel`): window 0 = bits 0..31, window 1 = bits
    /// 32..63, any other window = 0. The guest writes the select
    /// register (`set_device_features_sel`) then reads this window.
    pub(crate) fn device_features_window(&self) -> u32 {
        match self.device_features_sel {
            0 => self.device_features() as u32,
            1 => (self.device_features() >> 32) as u32,
            _ => 0,
        }
    }

    /// Immutable max ring size of the selected queue (0 if the selector
    /// is out of range).
    pub(crate) fn queue_max_size(&self) -> u32 {
        self.selected_queue()
            .map(|i| self.worker.queues[i].max_size() as u32)
            .unwrap_or(0)
    }

    /// Ready flag of the selected queue (0 if the selector is out of range).
    pub(crate) fn queue_ready(&self) -> u32 {
        self.selected_queue()
            .map(|i| self.worker.queues[i].ready() as u32)
            .unwrap_or(0)
    }

    /// Pending interrupt-status bits (cleared by [`Self::ack_interrupt`]).
    /// `load(Acquire)` pairs with the worker thread's
    /// `fetch_or(INT_VRING/INT_CONFIG, Release)` so a vCPU reading a bit
    /// set is guaranteed to also observe the freshly-published used.idx.
    pub(crate) fn interrupt_status(&self) -> u32 {
        self.interrupt_status.load(Ordering::Acquire)
    }

    /// Current device-status FSM bits (virtio-v1.2 §2.1). `load(Acquire)`
    /// pairs with the worker thread's `fetch_or(NEEDS_RESET)` on the
    /// queue-poison path.
    pub(crate) fn device_status(&self) -> u32 {
        self.device_status.load(Ordering::Acquire)
    }

    /// Config-space generation counter (bumped only by `reset` on the
    /// vCPU thread).
    pub(crate) fn config_generation(&self) -> u32 {
        self.config_generation.load(Ordering::Acquire)
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
    /// `queue_config_allowed` (FEATURES_OK set, DRIVER_OK clear);
    /// ignored otherwise.
    pub(crate) fn set_queue_size(&mut self, size: u16) {
        if !self.queue_config_allowed() {
            return;
        }
        if let Some(i) = self.selected_queue() {
            self.worker.queues[i].set_size(size);
        }
    }

    /// Set the selected queue's ready flag. Gated on
    /// `queue_config_allowed`; ignored otherwise.
    pub(crate) fn set_queue_ready(&mut self, ready: bool) {
        if !self.queue_config_allowed() {
            return;
        }
        if let Some(i) = self.selected_queue() {
            self.worker.queues[i].set_ready(ready);
        }
    }

    /// Notify the device that the guest kicked queue `queue_idx`. Only a
    /// kick of the single request queue (`REQ_QUEUE`) drives a drain via
    /// `process_requests` (which, per the cfg split, kicks the worker
    /// thread in production or runs `drain_inline` under `cfg(test)`);
    /// any other index is inert.
    pub(crate) fn notify_queue(&mut self, queue_idx: u32) {
        let idx = queue_idx as usize;
        if idx == REQ_QUEUE {
            self.process_requests();
        }
    }

    /// Clear the interrupt-status bits the guest acknowledged. AcqRel:
    /// the Acquire half pairs with the worker's Release fetch_or so we
    /// don't lose a bit racing with worker bit-set; the Release half
    /// publishes the cleared state.
    pub(crate) fn ack_interrupt(&mut self, val: u32) {
        self.interrupt_status.fetch_and(!val, Ordering::AcqRel);
    }

    /// Apply a device-status write: `0` resets the device, any other
    /// value drives the status FSM via [`Self::set_status`]. The reset /
    /// FSM side effects (worker respawn on DRIVER_OK, worker stop +
    /// state reclaim on reset) live in `reset` / `set_status`.
    pub(crate) fn write_status(&mut self, val: u32) {
        if val == 0 {
            self.reset();
        } else {
            self.set_status(val);
        }
    }

    /// Set the selected queue's descriptor-table address (low and/or
    /// high 32 bits; `None` leaves that half unchanged). Gated on
    /// `queue_config_allowed`; ignored otherwise.
    ///
    /// QUEUE_{DESC,AVAIL,USED}_{LOW,HIGH} write a 64-bit guest physical
    /// address as two 32-bit halves. Per virtio-v1.2 §4.2.2 the writes
    /// are only valid while FEATURES_OK is set and DRIVER_OK is NOT —
    /// the window between feature negotiation and the driver signalling
    /// "I'm done configuring"; outside it the write is dropped (the
    /// `queue_config_allowed` guard returns false). The virtio-queue
    /// crate accumulates the two halves internally; the guest typically
    /// writes LOW then HIGH but the order is not load-bearing.
    pub(crate) fn set_queue_desc_addr(&mut self, lo: Option<u32>, hi: Option<u32>) {
        if !self.queue_config_allowed() {
            return;
        }
        if let Some(i) = self.selected_queue() {
            self.worker.queues[i].set_desc_table_address(lo, hi);
        }
    }

    /// Set the selected queue's avail-ring (driver-area) address. Gated
    /// on `queue_config_allowed`; ignored otherwise. See
    /// [`Self::set_queue_desc_addr`] for the two-half write semantics.
    pub(crate) fn set_queue_avail_addr(&mut self, lo: Option<u32>, hi: Option<u32>) {
        if !self.queue_config_allowed() {
            return;
        }
        if let Some(i) = self.selected_queue() {
            self.worker.queues[i].set_avail_ring_address(lo, hi);
        }
    }

    /// Set the selected queue's used-ring (device-area) address. Gated
    /// on `queue_config_allowed`; ignored otherwise. See
    /// [`Self::set_queue_desc_addr`] for the two-half write semantics.
    pub(crate) fn set_queue_used_addr(&mut self, lo: Option<u32>, hi: Option<u32>) {
        if !self.queue_config_allowed() {
            return;
        }
        if let Some(i) = self.selected_queue() {
            self.worker.queues[i].set_used_ring_address(lo, hi);
        }
    }

    // ================= end transport-neutral core API =================

    /// Validate and apply a status transition per virtio-v1.2 §3.1.1.
    ///
    /// FEATURES_OK additionally enforces two constraints:
    ///
    /// 1. VIRTIO_F_VERSION_1 must be in `driver_features`
    ///    (virtio-v1.2 §6.1: "A driver MUST accept VIRTIO_F_VERSION_1").
    ///    Modern devices require this bit; a driver that fails to ack
    ///    it (legacy/transitional driver against this modern-only
    ///    device) cannot operate.
    /// 2. `driver_features` must be a SUBSET of `device_features()`
    ///    (virtio-v1.2 §3.1.1 step 5: "the driver MUST NOT set any
    ///    feature bit that the device did not offer"). A driver that
    ///    acks an unadvertised bit has either misread the device
    ///    feature page or is buggy/hostile; either way the device
    ///    cannot honor the implied contract because none of the
    ///    backend code paths for the unadvertised feature exist.
    ///
    /// The kernel's `virtio_features_ok` (drivers/virtio/virtio.c)
    /// writes FEATURES_OK then re-reads STATUS to confirm the bit
    /// stuck — rejecting here clears the path: the FSM leaves
    /// FEATURES_OK unset, the kernel's read-back fails, and the
    /// driver bind surfaces -ENODEV without descending into queue
    /// config.
    ///
    /// Every rejection path emits a `tracing::warn!` with the
    /// `device_status` / requested `val` / `new_bits` payload so an
    /// operator debugging a failed-bind can see which step the FSM
    /// rejected — clearing-bit attempts, ordering violations, multi-
    /// bit transitions, and unknown bits all surface explicitly
    /// rather than as a silent return.
    ///
    /// Idempotent re-writes (the requested `val` equals the
    /// current `device_status`) are a NO-OP, not a rejection: the
    /// monotone-bit gate accepts them (no bits cleared) and the
    /// new_bits-zero short-circuit returns without logging.
    /// Standard drivers go through `virtio_add_status`
    /// (drivers/virtio/virtio.c:196-200), which writes
    /// `STATUS = old | NEW_BIT`; `virtio_features_ok`
    /// (drivers/virtio/virtio.c:230) re-reads via `get_status`
    /// to confirm the bit stuck. Warning on idempotent re-writes
    /// would pollute operator logs without surfacing real bugs.
    pub(crate) fn set_status(&mut self, val: u32) {
        // Snapshot the current FSM state. `set_status` runs on the
        // vCPU thread that received the MMIO write; the FSM walk
        // through ACK → DRIVER → FEATURES_OK → DRIVER_OK happens
        // sequentially within and across calls on that thread. The
        // production worker thread's only write site to
        // device_status is the `fetch_or(NEEDS_RESET, SeqCst)` on
        // the queue-poison path. Whether that write can race the
        // vCPU's FSM-advance store depends on the worker's
        // lifecycle:
        //
        // - **Pre-DRIVER_OK** (initial spawn deferred to the first
        //   `STATUS = DRIVER_OK` per `consume_pending_respawn`):
        //   no worker thread is alive yet, so no concurrent
        //   `fetch_or` can land. Single-writer device_status.
        // - **Between DRIVER_OK and reset**: the worker is alive
        //   and may queue-poison at any point; a vCPU-side
        //   set_status arriving in this window can race its
        //   `fetch_or(NEEDS_RESET)`.
        // - **Between reset and the next DRIVER_OK**: the worker
        //   has been joined (`reset_engine_spawned` →
        //   `stop_worker_and_reclaim_state`); single-writer.
        //
        // The middle bucket is the race that motivates the CAS
        // below. A naive `store(val, Release)` after the snapshot
        // would clobber a NEEDS_RESET bit the worker had just
        // fetch_or'd in — silently lying to the guest by reporting
        // a healthy FSM after the device had already declared
        // itself broken. The CAS below is **load-bearing for race
        // safety**, not defense-in-depth: the worker's
        // `fetch_or(NEEDS_RESET, SeqCst)` can set bits between this
        // load and the CAS attempt, and the CAS is the mechanism
        // that detects the contention. Replacing the store with a
        // compare_exchange against the snapshot detects the race:
        // if the worker advanced device_status concurrently, the
        // CAS fails and we re-snapshot + re-validate. Either the
        // re-validated transition still passes (worker added bits
        // we are about to set anyway — proceed) or it fails
        // (worker added NEEDS_RESET, which is not a legal
        // FSM-advance bit; the new snapshot rejects with the
        // monotone-bit gate or the `valid` match). The Acquire
        // load and the CAS's failure-side Acquire ordering
        // synchronise-with the worker's SeqCst fetch_or at
        // `drain_bracket_impl`'s queue-poison arm — Acquire
        // observation pairs with the SeqCst write side because
        // SeqCst is at least Release on the writer.
        //
        // Snapshot loaded outside the loop; on a CAS failure the
        // `Err(observed)` branch updates `current_status` directly
        // without re-issuing a `load` — saving one redundant
        // atomic read per retry while preserving the same
        // happens-before chain.
        let mut current_status = self.device_status.load(Ordering::Acquire);
        // CAS retry loop. Each iteration re-validates the proposed
        // transition against the freshly-snapshotted `current_status`
        // and attempts a `compare_exchange` to commit. On contention
        // (the worker fetch_or'd NEEDS_RESET between snapshot and
        // commit), the CAS returns `Err(observed)` and we restart
        // the loop with the observed value as the new snapshot.
        // Termination is bounded at AT MOST ONE worker-induced
        // retry: by the worker invariant (see the worker's
        // queue-poison fetch_or site), the worker may only
        // fetch_or `VIRTIO_CONFIG_S_NEEDS_RESET` and the operation
        // is idempotent after the first call. So the worker can
        // transition `device_status` from one observable state
        // (`current_status`) to one other state
        // (`current_status | NEEDS_RESET`) and never to a third
        // value while this set_status is running. After that
        // single retry the snapshot is stable: either the second
        // CAS succeeds, or the monotone-bit gate fires because
        // the new snapshot has NEEDS_RESET and `val` does not
        // include it.
        //
        // Defense-in-depth bounded-retry budget: the proof above
        // says termination is bounded at one worker-induced retry,
        // so any execution exceeding `MAX_CAS_RETRIES` (4) is
        // either an invariant violation (worker fetch_or'ing
        // something other than NEEDS_RESET, multi-writer
        // device_status) or a hardware live-lock. Cap the loop
        // and bail rather than spin the vCPU thread indefinitely
        // — bailing is safe because the guest will simply retry
        // the STATUS write and observe the worker-set NEEDS_RESET
        // on the next attempt. The cap is large enough (4) that
        // proof-respecting execution never reaches it.
        const MAX_CAS_RETRIES: u32 = 4;
        let mut cas_retries: u32 = 0;
        loop {
            if val & current_status != current_status {
                // CORRECT behavior — do NOT "fix" this gate to admit
                // the advance. After the worker's queue-poison path
                // fetch_or'd `VIRTIO_CONFIG_S_NEEDS_RESET` into
                // `current_status`, every subsequent guest STATUS
                // write whose `val` does NOT include the NEEDS_RESET
                // bit (drivers never set it — it is device-emitted
                // per virtio-v1.2 §2.1.1 bit 0x40) trips this check
                // and is rejected. That is the spec-mandated
                // behaviour: the device is dead until a STATUS=0
                // reset, and the kernel's `virtio_features_ok`-style
                // post-write `get_status` re-read sees the FSM bit
                // never stuck (because we rejected here) and
                // surfaces -ENODEV to the bind path. A future
                // refactor that loosens this gate to "allow the
                // advance and clear NEEDS_RESET silently" would
                // restore the silent-corruption hazard the CAS
                // exists to prevent.
                //
                // Distinguish the two failure modes that both surface
                // here as `val & current_status != current_status`:
                //
                // 1. NEEDS_RESET bit (0x40) is set in `current_status`
                //    but not in `val`. This happens when the worker's
                //    queue-poison path fetch_or'd NEEDS_RESET — either
                //    before this set_status call or during a CAS
                //    retry. The driver did NOT try to regress; the
                //    device set NEEDS_RESET on its own. Cite the
                //    queue-poison cause and the STATUS=0 recovery
                //    path so an operator reading the log knows the
                //    fix is a full reset, not a driver bug.
                //
                // 2. Otherwise: the driver attempted to clear a
                //    previously-set bit (per virtio-v1.2 §3.1.1
                //    status bits are monotone within a driver
                //    session) — a regress that surfaces a buggy
                //    driver clearing FEATURES_OK while keeping
                //    ACKNOWLEDGE.
                if current_status & VIRTIO_CONFIG_S_NEEDS_RESET != 0 {
                    tracing::warn!(
                        device_status = current_status,
                        requested = val,
                        "virtio-blk set_status rejected — device in \
                         NEEDS_RESET state from prior queue poison; \
                         guest must write STATUS=0 to reset before any \
                         further FSM advance can succeed"
                    );
                } else {
                    tracing::warn!(
                        device_status = current_status,
                        requested = val,
                        "virtio-blk set_status rejected — attempted to clear \
                         a previously-set status bit without a full reset \
                         (virtio-v1.2 §3.1.1: status bits are monotone within \
                         a driver session)"
                    );
                }
                return;
            }
            let new_bits = val & !current_status;
            // Idempotent re-write of the current device_status: the
            // monotone-bit gate above passed (val is a superset) AND
            // the requested value adds no new bits. This is a
            // legitimate driver pattern — the kernel's
            // `virtio_add_status` (drivers/virtio/virtio.c:196-200)
            // writes `STATUS = old | NEW_BIT` and a subsequent
            // `virtio_features_ok` (drivers/virtio/virtio.c:230)
            // `get_status` read may race a duplicate set, plus an
            // MMIO probe path may issue a duplicate STATUS write.
            // Treat as a no-op rather than a rejection so the
            // rejection-warn path stays a true signal.
            if new_bits == 0 {
                return;
            }
            // FAILED (virtio-v1.2 §2.1.1 bit 0x80) is the driver's
            // "I give up" signal. The kernel's
            // `virtio_add_status(dev, VIRTIO_CONFIG_S_FAILED)` is the
            // exit path on probe failure
            // (drivers/virtio/virtio.c:363, 570, 606, 643): it reads
            // `get_status`, ORs in FAILED, and writes the result. So
            // `val == current_status | FAILED` and `new_bits ==
            // FAILED` regardless of which FSM rung the driver had
            // reached. Accept and store without consulting the
            // FSM-ladder match — FAILED can land at any state, and
            // routing it through the ACK/DRIVER/FEATURES_OK/DRIVER_OK
            // arms would reject the legitimate signal as an "illegal
            // FSM transition" and silently drop the FAILED bit from
            // device_status, leaving operators reading the failure
            // dump unable to see the guest gave up. Reject only when
            // FAILED appears alongside other unrecognised new bits —
            // those are protocol violations unrelated to the
            // legitimate FAILED signal and fall through to the
            // FSM-ladder match below. Mirrors virtio_console.rs's
            // FAILED early-accept pattern at the same location in its
            // set_status.
            if new_bits == VIRTIO_CONFIG_S_FAILED {
                // CAS against the snapshot for the same race-safety
                // reason as the valid-FSM-transition store below: the
                // worker thread can fetch_or NEEDS_RESET between
                // snapshot and store, and a naive `store(val,
                // Release)` would clobber that bit. Acquire on
                // failure synchronizes-with the worker's SeqCst
                // fetch_or so the next iteration's monotone-bit gate
                // (top of the loop) sees the worker's NEEDS_RESET. On
                // CAS-failure retry the new snapshot has NEEDS_RESET
                // but `val` does not (the kernel's `val` was computed
                // from the pre-fetch_or get_status), so the
                // monotone-bit gate fires and rejects — the device is
                // already declaring itself broken via NEEDS_RESET, so
                // dropping the FAILED bit on this path is acceptable;
                // the guest must reset before any further FSM advance
                // can succeed.
                match self.device_status.compare_exchange(
                    current_status,
                    val,
                    Ordering::Release,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        tracing::warn!(
                            old = current_status,
                            new = val,
                            "virtio-blk set_status: guest set FAILED status \
                             (virtio-v1.2 §2.1.1 bit 0x80 — driver gave up on \
                             device probe). Stored without further FSM advance.",
                        );
                        return;
                    }
                    Err(observed) => {
                        debug_assert_eq!(
                            observed & !current_status & !VIRTIO_CONFIG_S_NEEDS_RESET,
                            0,
                            "device_status race: observed bits beyond NEEDS_RESET — \
                             worker invariant violated (snapshot={current_status:#x}, \
                             observed={observed:#x})",
                        );
                        cas_retries += 1;
                        if cas_retries >= MAX_CAS_RETRIES {
                            tracing::error!(
                                device_status = observed,
                                requested = val,
                                retries = cas_retries,
                                "virtio-blk set_status abandoned — \
                                 CAS retry budget exhausted on FAILED \
                                 store; either the worker invariant is \
                                 violated or a hardware live-lock is \
                                 starving the vCPU thread; bailing \
                                 without advancing the FSM",
                            );
                            return;
                        }
                        current_status = observed;
                        continue;
                    }
                }
            }
            let valid = match new_bits {
                VIRTIO_CONFIG_S_ACKNOWLEDGE => current_status == 0,
                VIRTIO_CONFIG_S_DRIVER => current_status == S_ACK,
                VIRTIO_CONFIG_S_FEATURES_OK => {
                    current_status == S_DRV
                        && self.driver_features & (1u64 << VIRTIO_F_VERSION_1) != 0
                        && self.driver_features & !self.device_features() == 0
                }
                VIRTIO_CONFIG_S_DRIVER_OK => current_status == S_FEAT,
                _ => false,
            };
            if valid {
                // compare_exchange against the snapshot. On success
                // the store lands with Release ordering (mirroring
                // the pre-CAS `store(val, Release)` semantics for
                // any vCPU reader doing `load(Acquire)`). On failure
                // the worker raced an additional bit (NEEDS_RESET on
                // queue poison) and we restart the outer loop with
                // the observed value. Acquire on the failure side
                // synchronizes-with the worker's SeqCst fetch_or
                // (which is at least Release on the writer side) so
                // the next iteration's re-validation sees the
                // worker's NEEDS_RESET bit.
                match self.device_status.compare_exchange(
                    current_status,
                    val,
                    Ordering::Release,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {}
                    Err(observed) => {
                        // Verify the worker invariant: the only bits
                        // that can appear in `observed` beyond the
                        // pre-CAS snapshot are NEEDS_RESET. Any other
                        // newly-set bit means a writer beyond the
                        // documented queue-poison fetch_or site
                        // exists — a regression that must surface
                        // loudly in debug builds before the CAS retry
                        // proof's bounded-retry assumption is
                        // silently violated.
                        debug_assert_eq!(
                            observed & !current_status & !VIRTIO_CONFIG_S_NEEDS_RESET,
                            0,
                            "device_status race: observed bits beyond NEEDS_RESET — \
                             worker invariant violated (snapshot={current_status:#x}, \
                             observed={observed:#x})",
                        );
                        cas_retries += 1;
                        if cas_retries >= MAX_CAS_RETRIES {
                            tracing::error!(
                                device_status = observed,
                                requested = val,
                                retries = cas_retries,
                                "virtio-blk set_status abandoned — \
                                 CAS retry budget exhausted; either the \
                                 worker invariant is violated or a \
                                 hardware live-lock is starving the \
                                 vCPU thread; bailing without \
                                 advancing the FSM",
                            );
                            return;
                        }
                        current_status = observed;
                        continue;
                    }
                }
                // Once FEATURES_OK is committed, feature negotiation
                // is closed (virtio-v1.2 §3.1.1) — the negotiated set
                // lives in `driver_features` and the device may rely
                // on it. If VIRTIO_RING_F_EVENT_IDX was negotiated,
                // enable event-idx tracking on the request queue so
                // `Queue::needs_notification` consults the guest's
                // `used_event` threshold instead of always returning
                // true. `QueueT::event_idx_enabled` is documented to
                // return the correct value only after FEATURES_OK,
                // so this is the earliest legal moment to flip it
                // on.
                if new_bits == VIRTIO_CONFIG_S_FEATURES_OK
                    && self.driver_features & (1u64 << VIRTIO_RING_F_EVENT_IDX) != 0
                {
                    self.worker.queues[REQ_QUEUE].set_event_idx(true);
                }
                // DRIVER_OK transition: consume any deferred respawn
                // state stashed by `reset_engine_spawned`. By the
                // time the guest reaches DRIVER_OK it has walked ACK
                // → DRIVER → FEATURES_OK, and the
                // queue_config_allowed gate (S_FEAT && !DRIVER_OK)
                // admitted any DESC/AVAIL/USED address writes plus
                // QUEUE_NUM / QUEUE_READY between FEATURES_OK and
                // now. The kernel virtio-mmio driver's `vm_setup_vq`
                // (drivers/virtio/virtio_mmio.c:346-444) publishes
                // the queue addresses and writes `QUEUE_READY=1` in
                // that window before the DRIVER_OK MMIO write, so
                // the worker spawned here will find a
                // fully-configured queue on its first drain attempt.
                // Production cfg only — the inline-engine test build
                // has no respawn machinery. See the
                // `SpawnedEngine::respawn_pending` doc for the full
                // rationale and race-free invariant.
                #[cfg(not(test))]
                if new_bits == VIRTIO_CONFIG_S_DRIVER_OK {
                    self.consume_pending_respawn();
                }
                return;
            }
            // Rejection paths. The FEATURES_OK case has the richest
            // diagnostic because it's the only transition with
            // sub-conditions beyond simple ordering (subset rule +
            // VERSION_1 mandate); other rejections cite the FSM
            // ordering violation directly.
            if new_bits == VIRTIO_CONFIG_S_FEATURES_OK && current_status == S_DRV {
                // FEATURES_OK with the right ordering but the driver
                // failed the feature-set rules. Report VERSION_1
                // missing first (most common failure mode for a
                // legacy/transitional driver); fall through to the
                // unadvertised-bit case if VERSION_1 is fine.
                if self.driver_features & (1u64 << VIRTIO_F_VERSION_1) == 0 {
                    tracing::warn!(
                        driver_features = ?self.driver_features,
                        "FEATURES_OK rejected — VIRTIO_F_VERSION_1 not negotiated; \
                         legacy/transitional driver against modern-only device",
                    );
                } else {
                    let unadvertised = self.driver_features & !self.device_features();
                    if unadvertised != 0 {
                        tracing::warn!(
                            driver_features = ?self.driver_features,
                            device_features = ?self.device_features(),
                            unadvertised = ?unadvertised,
                            "FEATURES_OK rejected — driver acked unadvertised \
                             feature bits; subset rule (virtio-v1.2 §3.1.1) \
                             violated",
                        );
                    }
                }
            } else if current_status & VIRTIO_CONFIG_S_NEEDS_RESET != 0 {
                // NEEDS_RESET-specific diagnostic — defense in depth
                // alongside the same gate at the monotone-bit branch
                // above. The monotone-bit branch fires for the
                // typical race (val omits NEEDS_RESET, current_status
                // has it), but a future caller that constructed
                // `val` to include NEEDS_RESET (e.g. an internal
                // helper that shouldn't exist but might be added)
                // would slip past the monotone-bit gate and reach
                // this rejection arm. Cite the queue-poison cause
                // here too so the diagnostic taxonomy stays
                // consistent.
                tracing::warn!(
                    device_status = current_status,
                    requested = val,
                    new_bits = new_bits,
                    "virtio-blk set_status rejected — device in \
                     NEEDS_RESET state from prior queue poison; \
                     guest must write STATUS=0 to reset before any \
                     further FSM advance can succeed",
                );
            } else {
                // Generic ordering or unknown-bit rejection: ACK
                // without device_status==0, DRIVER without ACK,
                // FEATURES_OK from the wrong predecessor, DRIVER_OK
                // without FEATURES_OK, or any new_bits that aren't a
                // single virtio-v1.2 status bit (multi-bit
                // transitions, reserved bits set). Citing
                // device_status + new_bits lets an operator identify
                // the ordering violation without rederiving the FSM.
                tracing::warn!(
                    device_status = current_status,
                    requested = val,
                    new_bits = new_bits,
                    "virtio-blk set_status rejected — illegal FSM transition \
                     (virtio-v1.2 §3.1.1 ordering: ACK → DRIVER → FEATURES_OK \
                     → DRIVER_OK, one bit at a time)",
                );
            }
            return;
        }
    }

    /// Reset the device to its initial state per virtio-v1.2 §2.1.
    ///
    /// Two race-free paths, gated by `cfg`:
    ///
    /// - **Production (`cfg(not(test))`):** the worker thread owns
    ///   the `BlkWorkerState` and may be mid-drain when the vCPU
    ///   MMIO write of `STATUS = 0` lands here. Issuing
    ///   `q.reset()` while the worker holds the QueueSync mutex
    ///   (during `pop_descriptor_chain` / `add_used`) would race —
    ///   even worse, the worker may be in `pread`/`pwrite` against
    ///   a soon-to-be-stale guest memory mapping or compute an
    ///   `add_used` against the post-reset queue with cleared
    ///   `next_avail`. We close that window by stopping the worker
    ///   first, joining it (so no concurrent reader exists), then
    ///   running `q.reset()` and re-spawning a fresh worker
    ///   against the post-reset queue.
    ///
    ///   We converge with cloud-hypervisor's pattern of stopping
    ///   the worker on reset and deferring the respawn to the
    ///   guest's next `DRIVER_OK` transition. We still diverge
    ///   from firecracker (whose virtio-block device does not
    ///   implement reset at all — `Reset` returns `None` from the
    ///   device shim and the transport marks the device FAILED).
    ///   The reclaimed `BlkWorkerState` is parked in
    ///   `SpawnedEngine::respawn_pending` until `set_status`
    ///   observes the `STATUS = DRIVER_OK` MMIO write and calls
    ///   `consume_pending_respawn`, which builds fresh kick/stop
    ///   eventfds and a fresh worker thread against the
    ///   re-bound queue. Between reset and DRIVER_OK no worker
    ///   thread is alive, so kicks landing on the stale
    ///   (now-detached) `kick_fd` accumulate harmlessly until the
    ///   re-bind completes — the fresh worker will iter() over
    ///   chains the guest enqueued, since chain state lives in
    ///   guest memory, not the eventfd counter. Deferring saves
    ///   a thread sitting in `epoll_wait` for the duration of the
    ///   guest's rebind sequence (queue addresses zeroed,
    ///   `QUEUE_READY` false) — a window driver implementations
    ///   can stretch into milliseconds.
    ///
    /// - **Tests (`cfg(test)`):** Inline mode runs `drain_inline`
    ///   synchronously on the caller thread, so by the time
    ///   `reset()` is invoked there is no concurrent reader on
    ///   `worker.queues[…]`. The test-mode reset
    ///   (`reset_engine_inline`) resets the queue in place,
    ///   rebuilds the throttle buckets from the captured
    ///   `self.throttle` (so an adversarial test cannot drain the
    ///   bucket and reset to bypass), and clears the scratch Vecs
    ///   (capacity retained).
    ///
    /// # Counter persistence
    ///
    /// `VirtioBlkCounters` (`reads_completed`, `bytes_read`,
    /// `throttled_count`, `io_errors`, etc.) persist across reset.
    /// They are cumulative for the device's lifetime — a guest
    /// re-bind preserves the counter Arc so an operator monitoring
    /// failure-dump counters observes a monotonically
    /// non-decreasing series spanning the device's full IO
    /// history.
    ///
    /// # vCPU thread blocking
    ///
    /// The production path's `handle.join()` runs on the vCPU
    /// thread that received the MMIO write. If the worker is
    /// mid-`pread`/`pwrite` when STOP_TOKEN is signaled, the
    /// syscall completes before the worker reaches the next
    /// `epoll_wait` and observes the stop signal. The vCPU thread
    /// blocks for the duration. This is bounded by the same
    /// backing-speed assumption documented at the module level
    /// (tmpfs / warm page cache). A `reset()` issued during a slow
    /// IO can stretch beyond the freeze coordinator's rendezvous
    /// timeout, so `reset()` caps the worker join at
    /// [`RESET_JOIN_TIMEOUT`] (1 s) via [`join_worker_with_timeout`]
    /// (see [`Self::stop_worker_and_reclaim_state`]); on timeout
    /// the worker is leaked into the permanent-workerless state
    /// rather than hanging the rendezvous indefinitely.
    pub(crate) fn reset(&mut self) {
        // Phase 1 — clear MMIO-side scalar device state. These
        // fields live on `VirtioBlk` only (not shared with the
        // worker thread), so they're safe to mutate before the
        // queue stop+respawn. `interrupt_status` is intentionally
        // NOT cleared here because the worker thread (production)
        // may still race-fire `irq_evt.write(1)` and bit-set
        // INT_VRING; we clear it only after the worker is joined.
        // `device_status` is also deferred to Phase 3 for the same
        // reason: the worker's queue-poison path can fetch_or
        // NEEDS_RESET concurrently with this reset(), and clearing
        // it before the worker is joined would let a phantom
        // NEEDS_RESET bit re-set itself between Phase 1 and Phase 2.
        // `mem_unset_warned` is deferred to Phase 3 for the same
        // reason: the worker thread does
        // `mem_unset_warned.swap(true, Relaxed)` (worker.rs:788)
        // when it observes a missing GuestMemory, and clearing the
        // latch in Phase 1 would let a worker swap-true between
        // Phase 1 and Phase 2 — leaving the latch stuck `true` for
        // the post-reset driver session and silencing the
        // wiring-bug warning we explicitly want for the next
        // bind.
        self.queue_select = 0;
        self.device_features_sel = 0;
        self.driver_features_sel = 0;
        self.driver_features = 0;
        // Bump config_generation on every reset so a re-binding
        // driver observes a different value and re-reads config
        // space (per virtio-v1.2 §4.2.2.1: drivers MUST re-read
        // on changed generation). For v0 the capacity is fixed
        // for the device's lifetime — set once in `new()` and
        // never mutated — so the bump is purely defense-in-depth:
        // a future patch that resizes the disk between resets is
        // the case it guards. wrapping_add is implicit in
        // fetch_add's modular arithmetic.
        //
        // Release ordering: today the only writer is this
        // (vCPU-thread `reset()`), and the only reader is the
        // vCPU-thread `mmio_read(CONFIG_GENERATION)`, so
        // single-threaded access makes Release semantically
        // unnecessary. Release is defense-in-depth against future
        // cross-thread config writers (e.g. a follow-up that
        // resizes the disk from a worker thread or a host
        // monitor); pairs with the Acquire load in `mmio_read`.
        self.config_generation.fetch_add(1, Ordering::Release);

        // Phase 2 — engine-specific quiesce and queue reset
        // (production); respawn deferred to DRIVER_OK via
        // `consume_pending_respawn`. The `cfg(test)` Inline path
        // performs an in-place state reset on the caller thread.
        // Both paths leave the engine in a state where no worker
        // is currently mutating `interrupt_status` / `irq_evt`.
        #[cfg(test)]
        self.reset_engine_inline();
        #[cfg(not(test))]
        self.reset_engine_spawned();

        // Phase 3 — quiesce the IRQ path. With the worker stopped
        // (production) or never-active (test), no new
        // `irq_evt.write(1)`, `interrupt_status` bit-set, or
        // `device_status` fetch_or(NEEDS_RESET) can race us. Drain
        // the eventfd's pending counter so a stale worker write
        // (delivered between the last add_used and the stop signal)
        // doesn't fire a phantom IRQ at the post-reset guest; zero
        // `interrupt_status` so the guest's MMIO read of
        // INTERRUPT_STATUS observes a clean slate; zero
        // `device_status` so the guest re-reads STATUS=0 and walks
        // the FSM from scratch (per virtio-v1.2 §3.1.1: a reset
        // returns the device to its initial state including all FSM
        // bits — the NEEDS_RESET bit set by the worker's
        // queue-poison path is part of that state and clears here).
        // Both stores are Release-ordered to pair with their
        // respective `mmio_read` Acquire loads.
        //
        // Race window: a worker that completed `add_used` +
        // `irq_evt.write(1)` after the vCPU latched STATUS=0 but
        // before the stop signal landed would otherwise leave a
        // pending eventfd counter; KVM's irqfd would deliver the
        // GSI to the guest after reset, with the used ring now
        // empty (post-`q.reset()`), causing the guest's
        // `virtblk_done` to spin chasing a non-existent
        // completion. Draining here closes that window. The
        // device_status store deferral closes the parallel window
        // for the queue-poison path: a worker that ran
        // `fetch_or(NEEDS_RESET)` after Phase 1 but before being
        // joined would otherwise leave the bit set after reset,
        // and the guest's FSM walk from STATUS=0 → ACK → DRIVER →
        // FEATURES_OK → DRIVER_OK would silently transition
        // through a "device still says NEEDS_RESET" state visible
        // through `mmio_read(STATUS)`.
        let _ = self.irq_evt.read();
        // Drain the pause eventfd counter so any `pause()` writes
        // that landed during this reset cycle (e.g. a freeze
        // coordinator that fired between `reset_engine_spawned`'s
        // join and this Phase 3) do not carry a stale tick across
        // the rebind. Without this drain, the next
        // `worker_thread_main` (spawned at the next DRIVER_OK)
        // would observe PAUSE_TOKEN on its first `epoll_wait`,
        // park immediately, and starve the guest's first kicks
        // until the coordinator's eventual `resume()`. The read
        // is best-effort — a `WouldBlock` (counter already 0)
        // is normal, any other error means the eventfd is
        // already torn down which the next worker spawn will
        // re-create.
        let _ = self.pause_evt.read();
        self.interrupt_status.store(0, Ordering::Release);
        self.device_status.store(0, Ordering::Release);
        // Re-arm the "queue notify before set_mem" warning so a
        // post-reset wiring bug surfaces (virtio-v1.2 §3.1.1: a
        // reset puts the device in a state where the driver must
        // rebind and re-publish queue addresses; if a kick reaches
        // us before the rebind completes, that's worth a fresh
        // log line, not a quiet drop based on a latch from a
        // previous lifetime). Deferred to Phase 3 so the worker
        // (which is the only thread that swaps the latch to
        // `true` at worker.rs:788) is joined first — clearing in
        // Phase 1 would race a live worker swap-true and leave
        // the latch stuck `true` for the next driver session,
        // silencing the wiring-bug warning we explicitly want.
        self.mem_unset_warned.store(false, Ordering::Relaxed);
    }

    /// Test-mode engine reset: queue mutation and bucket rebuild
    /// happen on the caller thread (no worker exists). Scratches
    /// keep their capacity.
    #[cfg(test)]
    pub(crate) fn reset_engine_inline(&mut self) {
        for q in &mut self.worker.queues {
            q.reset();
        }
        let WorkerEngine::Inline(engine) = &mut self.worker.engine;
        let (ops_bucket, bytes_bucket) = buckets_from_throttle(self.throttle);
        engine.state.ops_bucket = ops_bucket;
        engine.state.bytes_bucket = bytes_bucket;
        engine.state.all_descs_scratch.clear();
        // Reset throttle-stall gauge state. q.reset() above
        // cleared the queue cursor, so any chain that was
        // rolled-back-pending is now lost from the device's
        // perspective — the guest's re-bind will re-issue
        // chains from a fresh avail.idx=0. The currently_stalled
        // flag must clear and the gauge must decrement to match;
        // otherwise the gauge leaks one increment per reset that
        // happens during a stall window. The gauge is "currently
        // pending throttle-stalled requests"; post-reset there
        // are none until the guest re-issues IO.
        if engine.state.currently_stalled {
            engine.state.currently_stalled = false;
            engine.state.counters.record_throttle_pending_dec();
        }
        // Clear hostile-guest poison: the guest issued a virtio
        // reset, which is the only documented escape from the
        // queue-poisoned state. The `invalid_avail_idx_count`
        // counter is intentionally NOT cleared here — operators
        // need cumulative-event visibility across resets to detect
        // repeated hostile-guest behavior.
        engine.state.queue_poisoned = false;
    }

    /// Production engine reset: stop the worker, join, q.reset(),
    /// stash the reclaimed state in `respawn_pending` for
    /// `set_status` to consume on the next DRIVER_OK transition.
    /// The reclaimed state contributes its long-lived resources
    /// (backing File, scratch capacities, capacity_bytes,
    /// read_only, counters Arc) — only the throttle buckets are
    /// rebuilt by `respawn_worker` once DRIVER_OK fires.
    ///
    /// Why defer the respawn: between `reset()` and DRIVER_OK
    /// the guest is rebinding (queue addresses zeroed,
    /// QUEUE_READY false). A worker spawned eagerly here would
    /// sit in `epoll_wait` doing nothing for the duration of the
    /// rebind. See the `SpawnedEngine::respawn_pending` doc for
    /// the full rationale and race-free invariant.
    #[cfg(not(test))]
    pub(crate) fn reset_engine_spawned(&mut self) {
        // Detect a back-to-back reset (the guest issued STATUS=0
        // twice without an intervening DRIVER_OK). The first
        // reset stashed state in respawn_pending and joined the
        // worker; the second reset has no live worker to stop
        // and must NOT overwrite the pending state (the second
        // `stop_worker_and_reclaim_state` would return None and
        // clobber the first reset's reclaimed state — the
        // backing File and counter Arc would be lost). Skip the
        // worker-quiesce step in that case; the queue reset
        // below still runs because the guest expects a fresh
        // queue cursor.
        let already_pending = {
            let WorkerEngine::Spawned(eng) = &self.worker.engine;
            eng.respawn_pending.is_some()
        };
        if !already_pending {
            // If a freeze coordinator paused the worker via
            // `pause()` and a STATUS=0 reset arrives before
            // `resume()`, the worker is parked in its
            // `park_timeout(10ms)` Acquire-load loop and does NOT
            // observe `stop_fd` — `epoll_wait` is unreachable from
            // the park. Clear `paused` (Release) and unpark BEFORE
            // writing `stop_fd` so the worker wakes within 10 ms
            // (or immediately on the unpark hint), exits the park
            // loop, returns to `epoll_wait`, and observes
            // STOP_TOKEN. Without this, the
            // `join_worker_with_timeout(RESET_JOIN_TIMEOUT, 1s)`
            // would always fire the TimedOut diagnostic when reset
            // races a paused worker. Cloud-hypervisor's epoll-helper
            // teardown follows the same unpause-before-stop ordering
            // (clear the paused flag and wake before signalling the
            // kill eventfd) so a parked worker observes the kill on
            // its first epoll-wake rather than after a 10 ms
            // park-timeout tick.
            self.resume();
            let reclaimed = self.stop_worker_and_reclaim_state();
            // Re-arm the construction-time "paused" sentinel so a
            // freeze that fires between this stop and the next
            // DRIVER_OK respawn passes the rendezvous vacuously
            // (mirrors the `with_options` initialisation). Without
            // this, the prior `resume()` left `paused=false`, and
            // the rendezvous would block until the 30 s timeout
            // waiting for a worker that does not yet exist — the
            // freeze coordinator's failure-dump path would lose
            // the dump for any STALL_DETECTED that lands in the
            // rebind window.
            self.paused.store(true, Ordering::Release);
            // Stash the reclaimed state for the deferred respawn.
            // `set_status` consumes it on the next valid DRIVER_OK
            // transition. `None` (worker had panicked / timed out /
            // helper failed) means no state to respawn from — the
            // device is permanently workerless from this point. The
            // diagnostic was already logged by
            // `stop_worker_and_reclaim_state`; the WorkerEngine
            // remains in `Spawned` form with `handle: None` and
            // `respawn_pending: None`, so future kicks land on the
            // stale `kick_fd` and accumulate harmlessly until the
            // device is destroyed. Only constructing a fresh
            // `VirtioBlk` recovers IO service.
            let WorkerEngine::Spawned(eng) = &mut self.worker.engine;
            eng.respawn_pending = reclaimed;
        }
        // q.reset() runs uncontested: the worker thread is joined
        // (or was never alive in the back-to-back-reset case) and
        // no new one has been spawned yet, so the QueueSync mutex
        // has no other holder.
        for q in &mut self.worker.queues {
            q.reset();
        }
    }

    /// Production: send STOP_TOKEN to the worker, join the
    /// thread with a [`RESET_JOIN_TIMEOUT`] budget, return the
    /// worker state. Returns `None` if the worker had already been
    /// joined (Option already taken — a second `reset()` after a
    /// torn-down engine, or a concurrent Drop racing the MMIO
    /// writer; both are operator bugs but must not panic the vCPU
    /// thread), if the worker panicked, OR if the join timed out
    /// or the helper machinery itself failed.
    ///
    /// # vCPU thread protection
    ///
    /// The unbounded `handle.join()` this function previously used
    /// would block the vCPU thread that received the `STATUS = 0`
    /// MMIO write through any wedged backing-IO path the worker
    /// hit (NFS stall, slow page cache, hung block device). The
    /// freeze coordinator's SIGRTMIN-based rendezvous (30 s wall
    /// budget at the coordinator level) targets that same vCPU
    /// thread; an unbounded reset block would either time out the
    /// rendezvous empty or arrive minutes late. Routing through
    /// [`join_worker_with_timeout`] caps the vCPU's pre-rendezvous
    /// overhead at [`RESET_JOIN_TIMEOUT`] (1 s) — the same
    /// invariant `Drop` enforces via [`DROP_JOIN_TIMEOUT`].
    ///
    /// # Outcomes
    ///
    /// - [`JoinWithTimeoutOutcome::Joined`] → return `Some(state)`;
    ///   reset proceeds to `q.reset()` + respawn.
    /// - [`JoinWithTimeoutOutcome::Panicked`] → log structured
    ///   error (matching Drop's diagnostic), return `None`. Device
    ///   enters permanent-workerless state.
    /// - [`JoinWithTimeoutOutcome::TimedOut`] → log structured
    ///   warn (worker is wedged in a blocking syscall that does
    ///   not check stop_fd), return `None`. Helper retains the
    ///   `JoinHandle` and the underlying `BlkWorkerState`; the
    ///   wedged worker keeps running until its blocking syscall
    ///   returns. Device enters permanent-workerless state — the
    ///   resource-retention trade documented at
    ///   [`join_worker_with_timeout`] applies here too.
    /// - [`JoinWithTimeoutOutcome::HelperSpawnFailed`] /
    ///   [`JoinWithTimeoutOutcome::HelperDisconnected`] → log
    ///   structured error, return `None`. Outer worker is
    ///   detached.
    ///
    /// All four non-Joined outcomes funnel through the
    /// "permanent device death" path documented at
    /// [`VirtioBlk::reset_engine_spawned`] — `reclaimed = None`
    /// skips the respawn and the device serves no further IO
    /// until reconstruction.
    #[cfg(not(test))]
    pub(crate) fn stop_worker_and_reclaim_state(&mut self) -> Option<BlkWorkerState> {
        let WorkerEngine::Spawned(eng) = &mut self.worker.engine;
        // Capture device-identifier fields before the
        // `eng.handle.take()` consumes the Option, so the
        // diagnostic warns can name the wedged device without
        // re-borrowing `self`.
        let stop_fd = eng.stop_fd.as_raw_fd();
        let capacity_sectors = self.capacity_sectors;
        let instance_id = self.instance_id;
        // Signal the worker to exit via the stop_fd helper, which
        // retries on EAGAIN (eventfd counter saturation) up to
        // STOP_FD_WRITE_MAX_RETRIES times before giving up. On
        // exhaustion the worker may not observe the stop signal;
        // the subsequent join's RESET_JOIN_TIMEOUT budget bounds
        // the wait to 1 s and surfaces the stall through the
        // TimedOut diagnostic below.
        signal_worker_stop(&eng.stop_fd, stop_fd, instance_id, capacity_sectors);
        // Re-borrow eng after the immutable reads above — needed
        // because `take()` mutates the Option.
        let WorkerEngine::Spawned(eng) = &mut self.worker.engine;
        let handle = eng.handle.take()?;
        match join_worker_with_timeout(handle, RESET_JOIN_TIMEOUT) {
            JoinWithTimeoutOutcome::Joined(state) => Some(state),
            JoinWithTimeoutOutcome::Panicked(payload) => {
                tracing::error!(
                    panic = panic_payload_str(&*payload),
                    stop_fd,
                    capacity_sectors,
                    instance_id,
                    "virtio-blk worker thread panicked during reset; \
                     no state to reclaim — device will not service IO \
                     until a fresh VirtioBlk is constructed"
                );
                None
            }
            JoinWithTimeoutOutcome::TimedOut => {
                tracing::warn!(
                    timeout_s = RESET_JOIN_TIMEOUT.as_secs_f32(),
                    stop_fd,
                    capacity_sectors,
                    instance_id,
                    "virtio-blk worker did not exit within \
                     RESET_JOIN_TIMEOUT of stop_fd during reset; \
                     leaking the worker thread to avoid blocking the \
                     vCPU thread (which the freeze coordinator may \
                     target with SIGRTMIN). Device enters the \
                     permanent-workerless state — guests will hang \
                     on every request until \
                     kernel.hung_task_timeout_secs (default 120 s) \
                     fires, and only constructing a fresh VirtioBlk \
                     recovers IO service. \
                     hint: identify the wedged device by stop_fd / \
                     instance_id / capacity_sectors above. \
                     hint: check `dmesg` for the backing fd's \
                     storage path stalling on I/O, or kill -USR1 \
                     the host process to dump worker thread \
                     backtraces."
                );
                None
            }
            JoinWithTimeoutOutcome::HelperSpawnFailed => {
                tracing::error!(
                    stop_fd,
                    capacity_sectors,
                    instance_id,
                    "virtio-blk reset helper thread spawn failed; \
                     detaching worker without join — device enters \
                     the permanent-workerless state"
                );
                None
            }
            JoinWithTimeoutOutcome::HelperDisconnected => {
                tracing::error!(
                    stop_fd,
                    capacity_sectors,
                    instance_id,
                    "virtio-blk reset helper thread terminated \
                     without forwarding the worker join result; \
                     device enters the permanent-workerless state"
                );
                None
            }
        }
    }

    /// Drain any state stashed in `SpawnedEngine::respawn_pending`
    /// by a prior `reset_engine_spawned` call and pass it to
    /// `respawn_worker`. Called by `set_status` on the DRIVER_OK
    /// transition — the only legal point at which the guest has
    /// finished publishing fresh queue addresses and the worker
    /// has real work to service.
    ///
    /// `respawn_pending` is `take()`-ed unconditionally even when
    /// `respawn_worker` itself fails to construct fresh fds or
    /// spawn the thread. This avoids leaving stale state holding
    /// scratch buffers and the backing-file `File` handle alive
    /// past the device's effective lifetime — the failure
    /// diagnostics from `respawn_worker` already document the
    /// permanent-workerless outcome. A second DRIVER_OK with no
    /// pending state (e.g. the guest re-binds without an
    /// intervening reset) is a no-op.
    #[cfg(not(test))]
    pub(crate) fn consume_pending_respawn(&mut self) {
        let pending = {
            let WorkerEngine::Spawned(eng) = &mut self.worker.engine;
            eng.respawn_pending.take()
        };
        if let Some(state) = pending {
            self.respawn_worker(state);
        }
    }

    /// Production: build a fresh `SpawnedEngine` (new kick_fd,
    /// stop_fd, worker thread) seeded with the reclaimed
    /// `BlkWorkerState`, and replace `self.worker.engine`. The
    /// throttle buckets in `state` are reconstructed from the
    /// captured `self.throttle` so an adversarial guest cannot
    /// drain the bucket and issue a reset to bypass the rate
    /// limit (spec-compliant: virtio-v1.2 §2.1 requires reset to
    /// return the device to its initial state, and bucket fill is
    /// part of that state).
    ///
    /// The `all_descs_scratch` buffer is `clear()`-ed (length
    /// zeroed, capacity retained) so the next worker iteration
    /// starts with no stale entries but without paying
    /// re-allocation cost on the first request.
    ///
    /// # Failure consequences
    ///
    /// On any resource-creation failure inside this function
    /// (`EventFd::new`, `try_clone`, `thread::Builder::spawn`),
    /// the engine is left holding the *old* `SpawnedEngine` whose
    /// `handle` field is `None` (taken by
    /// `stop_worker_and_reclaim_state` before this respawn).
    /// Future kicks via `process_requests` write to the stale
    /// `kick_fd` that no live worker is reading; the eventfd's
    /// counter increments harmlessly, but no IO completes — the
    /// guest will hang on every request until
    /// `kernel.hung_task_timeout_secs` (default 120 s) fires or
    /// the host destroys the device. The error is logged but not
    /// propagated to the caller (`reset()` returns `()` and the
    /// vCPU thread continues). This is permanent device death;
    /// only constructing a fresh `VirtioBlk` recovers the disk.
    #[cfg(not(test))]
    pub(crate) fn respawn_worker(&mut self, mut state: BlkWorkerState) {
        let (ops_bucket, bytes_bucket) = buckets_from_throttle(self.throttle);
        state.ops_bucket = ops_bucket;
        state.bytes_bucket = bytes_bucket;
        state.all_descs_scratch.clear();
        // Reset throttle-stall gauge state. q.reset() (run by
        // the caller before this) cleared the queue cursor, so
        // any chain that was rolled-back-pending is now lost
        // from the device's perspective — the guest's re-bind
        // will re-issue chains from a fresh avail.idx=0. The
        // currently_stalled flag must clear and the gauge must
        // decrement to match; otherwise the gauge leaks one
        // increment per reset-while-stalled scenario across the
        // device's lifetime.
        if state.currently_stalled {
            state.currently_stalled = false;
            state.counters.record_throttle_pending_dec();
        }
        // Clear hostile-guest poison: the guest issued a virtio
        // reset, which is the only documented escape from the
        // queue-poisoned state. `invalid_avail_idx_count` stays
        // because it tracks cumulative events across the device's
        // lifetime, not per-rebind state.
        state.queue_poisoned = false;

        // Build fresh kick/stop fds — the previous worker's
        // counter values are stale (a kick that arrived during
        // the old worker's drain may have been read but never
        // serviced before the stop, and the stop counter is
        // already incremented), and a hung vCPU mid-write to the
        // old kick_fd has nothing to coalesce against. Fresh fds
        // give a clean slate.
        //
        // The OLD worker's timerfd is owned by `worker_thread_main`'s
        // stack frame and dropped on STOP_TOKEN exit; we do NOT
        // need to migrate it. By the time this respawn runs:
        //   * `q.reset()` (called by the parent `reset_engine_spawned`
        //     just above this respawn) cleared the queue cursor —
        //     any chain that was rolled back via `set_next_avail`
        //     is gone from the device's perspective.
        //   * `state.ops_bucket` and `state.bytes_bucket` are
        //     rebuilt from `self.throttle` to full capacity, so
        //     the new worker's first drain attempt will not stall
        //     on a refill deficit (no timerfd needs to be armed
        //     for a chain that never re-stalls).
        //   * The guest must rebind (publish fresh queue addresses
        //     and set `QUEUE_READY = 1`) before any kick can fire.
        //     Until then `drain_bracket_impl` short-circuits on
        //     the `queues[REQ_QUEUE].ready()` gate — no drain, no
        //     stall, no need for a pending timerfd.
        // The clean-state contract above means a new timerfd
        // arms naturally on the first post-rebind stall, exactly
        // when one is needed.
        let kick_fd = match EventFd::new(libc::EFD_NONBLOCK) {
            Ok(fd) => fd,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: kick eventfd creation failed; \
                     leaving device without a worker — IO will not \
                     be serviced until reconstruction"
                );
                return;
            }
        };
        let stop_fd = match EventFd::new(libc::EFD_NONBLOCK) {
            Ok(fd) => fd,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: stop eventfd creation failed; \
                     leaving device without a worker — IO will not \
                     be serviced until reconstruction"
                );
                return;
            }
        };
        let worker_kick = match kick_fd.try_clone() {
            Ok(fd) => fd,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: kick eventfd clone failed; \
                     leaving device without a worker"
                );
                return;
            }
        };
        let worker_stop = match stop_fd.try_clone() {
            Ok(fd) => fd,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: stop eventfd clone failed; \
                     leaving device without a worker"
                );
                return;
            }
        };
        // Worker-side read clone of the host-owned `pause_evt`.
        // `try_clone` is `dup(2)`: it produces a new file descriptor
        // that points at the SAME underlying eventfd kernel object,
        // so the counter and any pending POLLIN readiness are shared
        // with `self.pause_evt`. The clone exists not to give the
        // worker a private counter (it can't — the kernel object is
        // shared) but because each fd can be registered in only one
        // epoll set: the worker's epoll holds this fd, while the
        // host side keeps `self.pause_evt` for `pause()` /
        // `is_paused()`. Counter cleanliness across respawns is
        // handled separately by `reset_engine_spawned`'s Phase 3
        // `pause_evt.read()` drain (V3) — a stale `1` from a
        // pre-stop write would otherwise carry across to the new
        // worker and trigger an immediate spurious park.
        let pause_fd = match self.pause_evt.try_clone() {
            Ok(fd) => fd,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: pause eventfd clone failed; \
                     leaving device without a worker"
                );
                return;
            }
        };
        // Clone the queue handles and Arcs the worker needs.
        // QueueSync is internally an `Arc<Mutex<Queue>>` so the
        // clone is cheap (refcount bump).
        let worker_queues = [self.worker.queues[REQ_QUEUE].clone()];
        let worker_mem = Arc::clone(&self.mem);
        let worker_irq = Arc::clone(&self.irq_evt);
        let worker_status = Arc::clone(&self.interrupt_status);
        let worker_device_status = Arc::clone(&self.device_status);
        let worker_warned = Arc::clone(&self.mem_unset_warned);
        let worker_paused = Arc::clone(&self.paused);
        let worker_parked_evt_slot = Arc::clone(&self.parked_evt);
        // Snapshot the placement at spawn time. A subsequent
        // `set_worker_placement` call only takes effect on the
        // NEXT respawn; the running worker observes the placement
        // captured here. This matches cloud-hypervisor's "topology
        // applied at activate()" pattern.
        let worker_placement = self.worker_placement.clone();

        let handle = match thread::Builder::new()
            .name("ktstr-vblk".to_string())
            .spawn(move || {
                worker_thread_main(
                    state,
                    worker_queues,
                    worker_mem,
                    worker_irq,
                    worker_status,
                    worker_device_status,
                    worker_warned,
                    worker_paused,
                    worker_placement,
                    worker_kick,
                    worker_stop,
                    pause_fd,
                    worker_parked_evt_slot,
                )
            }) {
            Ok(h) => h,
            Err(e) => {
                tracing::error!(
                    %e,
                    "virtio-blk reset: worker thread spawn failed; \
                     leaving device without a worker"
                );
                return;
            }
        };
        let WorkerEngine::Spawned(eng) = &mut self.worker.engine;
        *eng = SpawnedEngine {
            kick_fd,
            stop_fd,
            handle: Some(handle),
            respawn_pending: None,
        };
    }

    /// Signal the worker thread to park for a failure-dump
    /// rendezvous. Writes 1 to `pause_evt`; the worker's
    /// `epoll_wait` resumes on PAUSE_TOKEN, drains the eventfd
    /// counter, stores `paused=true` (Release), and parks in a
    /// 10 ms `park_timeout` loop until [`Self::resume`] clears
    /// the flag.
    ///
    /// The freeze coordinator polls `paused.load(Acquire)` after
    /// calling this to confirm the worker has reached the parked
    /// state before reading guest memory. The Release/Acquire
    /// pair provides the happens-before edge that makes the
    /// host-side post-rendezvous reads observe every queue
    /// mutation the worker performed pre-pause.
    ///
    /// Cfg-independent: `cfg(test)` builds use the inline engine,
    /// so `pause()` writes to the host eventfd but no worker is
    /// blocked on it; the test harness can inspect
    /// `self.paused.load()` directly to verify the host-side
    /// rendezvous machinery without a worker thread.
    ///
    /// On EAGAIN (counter saturation at u64::MAX-1) or EBADF
    /// (closed fd during shutdown), we log via `tracing::warn!`
    /// and return — the caller's downstream `paused.load(Acquire)`
    /// poll either succeeds (a prior pause ack is still latched) or
    /// times out at the 30s rendezvous deadline. Saturation is
    /// implausible in practice (every `pause()` is paired with a
    /// `resume()` that does NOT increment the counter; the worker's
    /// drain reads it back to 0 each cycle).
    pub fn pause(&self) {
        // No-live-worker fast path. With the deferred-spawn lifecycle
        // (initial worker created on the first DRIVER_OK), there is a
        // window between `with_options` and the guest's bind where no
        // thread is reading `pause_fd`. Writing the eventfd is still
        // safe — counter just accumulates harmlessly, and `reset`'s
        // Phase 3 drain (V3) clears it before the next worker spawns —
        // but the counter would otherwise carry a stale tick across
        // a respawn, and the rendezvous already passes vacuously
        // because `paused` was initialised to `true` and is never
        // cleared until the worker actually starts. Skip the write
        // and log at `debug` level so a misuse (pause without a
        // worker) is observable but not noisy.
        #[cfg(not(test))]
        {
            let WorkerEngine::Spawned(eng) = &self.worker.engine;
            if eng.handle.is_none() {
                tracing::debug!(
                    "virtio-blk pause() with no live worker; \
                     `paused` is already `true` from construction \
                     (or post-stop), rendezvous will pass vacuously"
                );
                return;
            }
        }
        if let Err(e) = self.pause_evt.write(1) {
            tracing::warn!(%e, "virtio-blk pause_evt.write failed");
        }
    }

    /// Clear the worker's parked state. Stores `paused=false`
    /// (Release); the worker's 10 ms `park_timeout` Acquire-load
    /// observes the clear within 10 ms and resumes its
    /// `epoll_wait` loop. The `unpark` call is a hint — the
    /// `park_timeout` already wakes periodically so a missed
    /// unpark is bounded at 10 ms latency, not unbounded.
    ///
    /// Cfg-independent for the same reason as [`Self::pause`].
    /// Returns `true` if a worker thread is alive and was
    /// unparked; `false` if the engine has no live worker (test
    /// mode, post-stop, post-failed-respawn). Callers use the
    /// return value to skip a `resume()` that has nothing to
    /// resume.
    pub fn resume(&self) -> bool {
        // No-live-worker fast path. Mirrors `pause()`'s early-return:
        // when the engine has no live thread (pre-DRIVER_OK, post-stop,
        // post-failed-respawn), preserve the V1 sentinel by RE-ARMING
        // `paused = true` instead of clearing it. Without this, a
        // dual-snapshot freeze (early + late) that calls
        // pause()/resume() across the rebind window would clear the
        // sentinel on the first resume(), and the second freeze's
        // is_paused() poll would observe `false` and time out at
        // FREEZE_RENDEZVOUS_TIMEOUT waiting for a worker that does
        // not exist. Re-arming preserves the vacuous-pass invariant
        // across consecutive freezes.
        #[cfg(not(test))]
        {
            let WorkerEngine::Spawned(eng) = &self.worker.engine;
            if let Some(ref handle) = eng.handle {
                self.paused.store(false, Ordering::Release);
                handle.thread().unpark();
                return true;
            }
            // No live worker — re-arm the sentinel.
            self.paused.store(true, Ordering::Release);
            false
        }
        #[cfg(test)]
        {
            // Inline engine: no worker thread to unpark; the
            // store(Release) above is the entire resume side. A
            // test harness driving pause/resume observes the
            // updated `paused` flag directly.
            self.paused.store(false, Ordering::Release);
            false
        }
    }

    /// Return `true` when the worker has acknowledged a prior
    /// [`Self::pause`] call by parking. The freeze coordinator's
    /// rendezvous loop uses this to wait for the worker's parked
    /// state before reading guest memory. Acquire ordering pairs
    /// with the worker's `paused.store(true, Release)` so the
    /// host-side reads happen-after every queue mutation the
    /// worker performed pre-pause.
    ///
    /// Cfg-independent for the same reason as [`Self::pause`].
    // Production callers retired with the freeze-coordinator queue
    // pause path; preserved for `tests_atomics` Acquire/Release pin.
    #[allow(dead_code)]
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }
}
