//! The `VirtioConsole` device implementation — the full method surface:
//! construction, eventfd accessors, feature/status gates, the per-port TX
//! drain, the guest-input (RX) queue path, the multiport control-message
//! protocol, MMIO register dispatch, config-space reads, and the
//! device-status FSM. Kept as a single `impl` so its private methods stay
//! mutually visible; split into its own submodule only to keep each file
//! under the size ceiling. The shared consts, `Port`/`ControlOut` types,
//! and the `VirtioConsole` struct live in the parent module (`super`),
//! reached via the glob below (a child module sees its ancestor's private
//! items). The test suite is a descendant module (`tests`) so it retains
//! access to the private methods it exercises.
use super::*;

impl VirtioConsole {
    /// Create a new virtio-console device.
    pub fn new() -> Self {
        let irq_evt =
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-console irq eventfd");
        let tx_evt =
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-console tx eventfd");
        let stats_tx_evt = EventFd::new(libc::EFD_NONBLOCK)
            .expect("failed to create virtio-console stats_tx eventfd");
        VirtioConsole {
            queues: [
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
            ],
            queue_select: 0,
            device_features_sel: 0,
            driver_features_sel: 0,
            driver_features: 0,
            device_status: 0,
            interrupt_status: 0,
            config_generation: 0,
            irq_evt,
            tx_evt,
            stats_tx_evt,
            mem: None,
            ports: [
                Port::new(PORT0_NAME),
                Port::new(PORT1_NAME),
                Port::new(PORT2_NAME),
            ],
            tx_scratch: Vec::new(),
            rx_scratch: Vec::new(),
            control_out: VecDeque::new(),
            device_ready: false,
        }
    }

    /// Eventfd for KVM irqfd registration.
    pub fn irq_evt(&self) -> &EventFd {
        &self.irq_evt
    }

    /// Eventfd signaled when new TX data arrives on port 0 or port 1.
    /// Use in the host-side stdout / bulk drain thread's poll set for
    /// zero-latency wakeup. Port 2 TX wakes are delivered separately
    /// via [`Self::stats_tx_evt`] so the stats client does not
    /// contend on this fd.
    pub fn tx_evt(&self) -> &EventFd {
        &self.tx_evt
    }

    /// Eventfd signaled when new TX data arrives on port 2 (scheduler
    /// stats relay). Used by
    /// [`crate::vmm::sched_stats::SchedStatsClient`] to wake on a
    /// stats-response edge without seeing port-0 console / port-1
    /// bulk wakes.
    pub fn stats_tx_evt(&self) -> &EventFd {
        &self.stats_tx_evt
    }

    /// Set guest memory reference. Must be called before starting vCPUs.
    pub fn set_mem(&mut self, mem: GuestMemoryMmap) {
        self.mem = Some(mem);
    }

    fn device_features(&self) -> u64 {
        (1u64 << VIRTIO_F_VERSION_1) | (1u64 << (VIRTIO_CONSOLE_F_MULTIPORT as u64))
    }

    fn selected_queue(&self) -> Option<usize> {
        let idx = self.queue_select as usize;
        if idx < NUM_QUEUES { Some(idx) } else { None }
    }

    // Console does not negotiate VIRTIO_RING_F_EVENT_IDX so the
    // combined bit+eventfd pattern is correct here. virtio_blk
    // splits the two because it negotiates EVENT_IDX.
    fn signal_used(&mut self) {
        self.interrupt_status |= VIRTIO_MMIO_INT_VRING;
        // Success path is silent (high-volume hot path: signal_used
        // fires per drained chain, tens to hundreds of times per
        // console burst); failure logs once per occurrence so a
        // genuine eventfd-write breakage surfaces in tracing rather
        // than silently disappearing.
        if let Err(e) = self.irq_evt.write(1) {
            tracing::warn!(%e, "virtio-console irq_evt.write failed");
        }
    }

    /// True iff the driver negotiated `VIRTIO_CONSOLE_F_MULTIPORT`.
    /// The multiport-only queues (c_ivq, c_ovq, port-1 RX, port-1
    /// TX) are valid only after this feature is acked. Without
    /// F_MULTIPORT the kernel's `init_vqs` allocates only the first
    /// two queues (drivers/char/virtio_console.c — the legacy
    /// single-console path), so any QUEUE_NOTIFY for queues 2-5 in
    /// that case is a guest protocol violation.
    fn multiport_negotiated(&self) -> bool {
        self.driver_features & (1u64 << (VIRTIO_CONSOLE_F_MULTIPORT as u64)) != 0
    }

    /// True when device_status has progressed past FEATURES_OK but not
    /// yet reached DRIVER_OK — the window where queue config is valid.
    fn queue_config_allowed(&self) -> bool {
        self.device_status & S_FEAT == S_FEAT && self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0
    }

    /// True when driver features may be written: DRIVER set, FEATURES_OK
    /// not yet set.
    fn features_write_allowed(&self) -> bool {
        self.device_status & S_DRV == S_DRV && self.device_status & VIRTIO_CONFIG_S_FEATURES_OK == 0
    }

    // ------------------------------------------------------------------
    // Port 0 console: guest → host (TX) and host → guest (RX)
    // ------------------------------------------------------------------

    /// Process a TX queue: drain descriptor data into the matching
    /// port's `tx_buf`. TX descriptors are device-readable (guest
    /// wrote them); the device writes nothing back, so add_used len
    /// is 0.
    ///
    /// Returns true when at least one byte was successfully copied —
    /// the caller uses that to gate `signal_used` + `tx_evt.write`.
    fn process_tx(&mut self, port_id: usize) -> bool {
        let port_label = port_label(port_id);
        let (_, queue_idx) = port_queues(port_id);
        // Spec gate: virtio-v1.2 §3.1.1 forbids the device from
        // accessing virtqueue memory before DRIVER_OK. A QUEUE_NOTIFY
        // arriving in a transient state would otherwise let the
        // device read addresses the guest hasn't validated.
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            tracing::debug!(
                queue = queue_idx,
                status = self.device_status,
                "virtio-console process_tx: DRIVER_OK not set; ignoring notify"
            );
            return false;
        }
        // F_MULTIPORT runtime gate: ports 1 and 2 are multiport-only.
        // Port 0 TX is valid in both single-console and multiport
        // configurations.
        if port_id != 0 && !self.multiport_negotiated() {
            tracing::warn!(
                port = port_label,
                "virtio-console process_tx: F_MULTIPORT not \
                 negotiated; ignoring notify on multiport-only TX queue"
            );
            return false;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => return false,
        };
        let mut had_data = false;
        // Cumulative bytes copied into the per-port accumulator
        // during this call. Compared against `TX_PER_CALL_MAX` after
        // every chain so a hostile guest publishing many small
        // chains cannot grow the host buffer without bound on a
        // single notify.
        let mut cumulative_bytes: usize = 0;
        // Disjoint-field borrows: the queue (`self.queues[queue_idx]`)
        // and the per-port accumulator (`self.ports[port_id].tx_buf`,
        // `self.tx_scratch`) are reborrowed independently inside the
        // loop. `tx_buf` is `VecDeque<u8>` and `read_slice` needs a
        // contiguous `&mut [u8]`, so we stage descriptor bytes in
        // `tx_scratch` then `extend` the deque. The deque lets port
        // 1's [`Self::push_back_bulk`] prepend the freeze
        // coordinator's residual in O(bytes) without an
        // O(buf_len + bytes) `Vec::splice(0..0, _)`.
        let q = &mut self.queues[queue_idx];
        while let Some(chain) = q.pop_descriptor_chain(mem) {
            let head = chain.head_index();
            for desc in chain {
                if !desc.is_write_only() {
                    let guest_addr = desc.addr();
                    let dlen = (desc.len() as usize).min(TX_DESC_MAX);
                    self.tx_scratch.clear();
                    self.tx_scratch.resize(dlen, 0);
                    let read_ok = match mem.read_slice(&mut self.tx_scratch, guest_addr) {
                        Ok(()) => {
                            self.ports[port_id]
                                .tx_buf
                                .extend(self.tx_scratch.iter().copied());
                            true
                        }
                        Err(e) => {
                            tracing::warn!(
                                port = port_label,
                                head,
                                dlen,
                                %e,
                                "virtio-console process_tx: read_slice failed \
                                 (descriptor addr likely unmapped); dropping \
                                 segment from this chain"
                            );
                            false
                        }
                    };
                    // Gate had_data and cumulative_bytes on dlen > 0.
                    // A zero-length descriptor's read_slice trivially
                    // succeeds (read_ok = true) but contributes no
                    // bytes — neither cumulative_bytes (which feeds
                    // the per-call cap at TX_PER_CALL_MAX) nor the
                    // accumulator grows. Setting had_data=true on a
                    // zero-byte chain would still trigger
                    // signal_used (IRQ to guest) and the tx_evt
                    // wake — a hostile guest publishing N
                    // zero-length chains gets N IRQs without any
                    // host-observable byte progress, an
                    // amplification primitive that bypasses the
                    // TX_PER_CALL_MAX bound (cumulative_bytes stays
                    // 0 forever). The kernel's virtio_console TX
                    // path always emits non-empty data segments
                    // (drivers/char/virtio_console.c
                    // __send_to_port via sg_set_buf with non-zero
                    // length), so legitimate guests are unaffected.
                    if read_ok && dlen > 0 {
                        had_data = true;
                        cumulative_bytes = cumulative_bytes.saturating_add(dlen);
                    }
                }
            }
            // TX add_used: on failure the descriptor leaks (guest's
            // TX queue eventually starves and the port stops). Log so
            // the silent-stop is observable rather than disappearing
            // into a swallowed Result. Mirrors virtio-blk's
            // `publish_completion` and virtio-net's TX add_used
            // logging pattern.
            if let Err(e) = q.add_used(mem, head, 0) {
                tracing::warn!(
                    port = port_label,
                    head,
                    %e,
                    "virtio-console TX add_used failed (used-ring address \
                     likely unmapped); guest TX queue will eventually \
                     starve and this port will stop"
                );
            }
            // Per-call drain cap: stop popping chains once the
            // cumulative byte total crosses the budget. Remaining
            // chains stay in the avail ring; the next QUEUE_NOTIFY
            // (or the next host-side drain) picks them up. A hostile
            // guest publishing thousands of valid PAGE_SIZE chains
            // would otherwise let one notify allocate hundreds of
            // megabytes into the host accumulator before returning
            // to the vCPU loop.
            if cumulative_bytes >= TX_PER_CALL_MAX {
                tracing::debug!(
                    port = port_label,
                    cumulative_bytes,
                    cap = TX_PER_CALL_MAX,
                    "virtio-console process_tx: per-call byte cap reached; \
                     remaining chains deferred to next notify"
                );
                break;
            }
        }
        if had_data {
            self.signal_used();
            // Wake the matching host poll thread. Ports 0 and 1 share
            // `tx_evt` (the freeze coordinator's TOKEN_TX handler
            // drains both); port 2 fires its own `stats_tx_evt` so
            // [`crate::vmm::sched_stats::SchedStatsClient`] wakes only on
            // a stats edge. A missed write means the host poll
            // absorbs the latency next cycle — not a correctness
            // failure. Silent swallow is intentional (in contrast to
            // signal_used's irq_evt write, which logs because a
            // missed IRQ stalls the GUEST, not just a host poll
            // cadence).
            if port_id == 2 {
                let _ = self.stats_tx_evt.write(1);
            } else {
                let _ = self.tx_evt.write(1);
            }
        }
        had_data
    }

    /// Drain-only TX walk for the device-reset path. Pops pending
    /// avail-ring chains for `queue_idx`, copies their bytes into the
    /// matching `port{0,1}_tx_buf`, and `add_used`s each chain — but
    /// emits NO `signal_used` (no IRQ to a guest that is rebooting)
    /// and NO `tx_evt.write` (no host wake; `collect_results`'s
    /// `final_drain` walks the buffer synchronously after the
    /// coordinator has already exited).
    ///
    /// Reset is called from `mmio_write` while the vCPU thread holds
    /// the device's outer mutex. Calling `process_tx` here would
    /// fire `tx_evt`, waking the freeze coordinator's TOKEN_TX handler
    /// which races to acquire the same mutex — a lock-contention
    /// feedback loop under burst workloads where 16 workers / 8 vCPUs
    /// flood the bulk port. The drain-only variant breaks the loop:
    /// the coord is never woken about a drain it doesn't need to do
    /// (the bytes are already captured for `final_drain` to surface).
    ///
    /// The DRIVER_OK and F_MULTIPORT gates from `process_tx` are
    /// preserved verbatim — `reset()` invokes this BEFORE clearing
    /// `device_status` and `driver_features`, so both gates pass on
    /// the legitimate reboot path. Calling this method post-reset
    /// (when both gates would fail) is a no-op, mirroring
    /// `process_tx`'s behaviour.
    ///
    /// Per-call byte cap (`TX_PER_CALL_MAX`) is preserved as a
    /// defensive bound against a hostile guest that publishes a
    /// massive avail-ring backlog right before the reboot — the
    /// remainder is left in the avail ring and is unreachable post-
    /// reset (queue GPAs get reset to sentinels), exactly the same
    /// data-loss surface as before this change. Test runs with the
    /// kernel's virtio_console driver never exercise this bound on
    /// the reset path; the cap is hostile-guest defence-in-depth.
    fn drain_tx_into_capture_buf(&mut self, port_id: usize) {
        let port_label = port_label(port_id);
        let (_, queue_idx) = port_queues(port_id);
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            return;
        }
        if port_id != 0 && !self.multiport_negotiated() {
            return;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => return,
        };
        let mut cumulative_bytes: usize = 0;
        let q = &mut self.queues[queue_idx];
        while let Some(chain) = q.pop_descriptor_chain(mem) {
            let head = chain.head_index();
            for desc in chain {
                if !desc.is_write_only() {
                    let guest_addr = desc.addr();
                    let dlen = (desc.len() as usize).min(TX_DESC_MAX);
                    self.tx_scratch.clear();
                    self.tx_scratch.resize(dlen, 0);
                    let read_ok = match mem.read_slice(&mut self.tx_scratch, guest_addr) {
                        Ok(()) => {
                            self.ports[port_id]
                                .tx_buf
                                .extend(self.tx_scratch.iter().copied());
                            true
                        }
                        Err(e) => {
                            tracing::warn!(
                                port = port_label,
                                head,
                                dlen,
                                %e,
                                "virtio-console reset-drain: read_slice failed \
                                 (descriptor addr likely unmapped); dropping \
                                 segment from this chain"
                            );
                            false
                        }
                    };
                    if read_ok && dlen > 0 {
                        cumulative_bytes = cumulative_bytes.saturating_add(dlen);
                    }
                }
            }
            if let Err(e) = q.add_used(mem, head, 0) {
                tracing::warn!(
                    port = port_label,
                    head,
                    %e,
                    "virtio-console reset-drain: add_used failed (used-ring \
                     address likely unmapped); descriptor leaks but the guest \
                     is rebooting so the leak has no observer"
                );
            }
            if cumulative_bytes >= TX_PER_CALL_MAX {
                tracing::debug!(
                    port = port_label,
                    cumulative_bytes,
                    cap = TX_PER_CALL_MAX,
                    "virtio-console reset-drain: per-call byte cap reached; \
                     remaining chains lost to queue reset"
                );
                break;
            }
        }
    }

    /// Drain a port's `tx_buf` and return its bytes as a contiguous
    /// `Vec<u8>`. Capacity-preserving swap: the replacement deque
    /// keeps the drained deque's capacity (capped at 256 KiB so a
    /// single hostile-guest burst that grew the accumulator beyond
    /// the per-call cap does not retain that capacity for the
    /// lifetime of the device). `Vec::from(VecDeque)` reuses the
    /// deque's allocation when the ring is contiguous (O(1)) and
    /// rotates in place when it isn't (O(N) but no realloc).
    fn drain_port_tx(&mut self, port_id: usize) -> Vec<u8> {
        let buf = &mut self.ports[port_id].tx_buf;
        let cap = buf.capacity().min(256 * 1024);
        let old = std::mem::replace(buf, VecDeque::with_capacity(cap));
        Vec::from(old)
    }

    /// Return and clear accumulated port-0 TX output (guest console →
    /// host stdout).
    pub fn drain_output(&mut self) -> Vec<u8> {
        self.drain_port_tx(0)
    }

    /// Return and clear accumulated port-1 TX output (guest bulk TLV
    /// stream). Host-side TLV parsing is in
    /// [`crate::vmm::host_comms::parse_tlv_stream`].
    pub fn drain_bulk(&mut self) -> Vec<u8> {
        self.drain_port_tx(1)
    }

    /// Final post-VM-exit port-1 TX drain. Walks the avail ring once
    /// to convert any descriptor chains the guest published without
    /// a corresponding QUEUE_NOTIFY MMIO arrival into bytes in port
    /// 1's `tx_buf`, then returns the accumulated buffer.
    ///
    /// `pub fn drain_bulk` alone catches only what `process_tx`
    /// has already deposited; on the eevdf-style failure path the
    /// guest writes a `MSG_TYPE_LIFECYCLE` and a `MSG_TYPE_EXIT`
    /// frame to `/dev/vport0p1` and immediately calls `force_reboot`,
    /// and the userspace write's `virtqueue_kick` MMIO can lag
    /// behind so the chains land in the avail ring without a
    /// matching host-side notify. A single explicit `process_tx`
    /// call from the host side picks them up.
    ///
    /// Synchronous call from `collect_results` (scheduler-test path)
    /// and `run_interactive` (shell `--exec` exit-code recovery); the
    /// underlying `process_tx` is the same code path MMIO QUEUE_NOTIFY
    /// uses, including the per-call `TX_PER_CALL_MAX` byte cap and
    /// the `DRIVER_OK` / `F_MULTIPORT` gates. If the guest already
    /// wrote 0 to VIRTIO_MMIO_STATUS, [`Self::reset`] has run; that
    /// path drains every port's TX queue via
    /// [`Self::drain_tx_into_capture_buf`] at its top (before the
    /// `device_status = 0` clobber and the `Queue::reset` GPA
    /// clobber) so any chains pending at reset time are already in
    /// `ports[1].tx_buf`. The `process_tx(1)` call below short-
    /// circuits on `DRIVER_OK == 0` and `drain_bulk` returns the
    /// captured bytes.
    pub(crate) fn final_drain(&mut self) -> Vec<u8> {
        let _ = self.process_tx(1);
        self.drain_bulk()
    }

    /// Push raw bytes back onto the head of the port-1 TX buffer.
    ///
    /// The freeze coordinator's mid-run `bulk_assembler` (see
    /// `crate::vmm::bulk::HostAssembler`) drains port 1's `tx_buf`
    /// via `drain_bulk` and assembles complete TLV frames. Trailing
    /// bytes of a partial frame stay buffered inside the assembler.
    /// When the coordinator thread exits (kill or BSP-done), those
    /// residual bytes would be dropped on the floor — `collect_results`
    /// then calls `drain_bulk` and parses with `parse_tlv_stream`,
    /// but the assembler-buffered tail never reaches that path. This
    /// method lets the coordinator push the residual back so
    /// `collect_results`'s `drain_bulk` returns it and `parse_tlv_stream`
    /// completes the frame. Bytes prepend to preserve the on-wire
    /// chronological order: the assembler's residual was at the head
    /// of the bulk stream relative to any bytes the device accumulated
    /// after the last coordinator drain.
    pub fn push_back_bulk(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        // The port-1 `tx_buf` is a VecDeque — push_front is O(1) per
        // byte (amortised); iterating in reverse and pushing each
        // lands `bytes[0]` at the very front of the deque, preserving
        // the chronological order described by the doc comment above.
        // This replaces a `Vec::splice(0..0, _)` whose cost was
        // O(buf_len + bytes) — under bursty TLV traffic that already
        // grew the buffer to hundreds of KiB the splice would
        // memmove every byte in the buffer to make room, which a
        // residual push at the very end of the run was not worth.
        // `reserve` lifts the capacity once so the push_front loop
        // does not trigger N small grows.
        let buf = &mut self.ports[1].tx_buf;
        buf.reserve(bytes.len());
        for &b in bytes.iter().rev() {
            buf.push_front(b);
        }
    }

    /// Return and clear accumulated port-2 TX output (guest stats
    /// relay → host stats client). Raw byte passthrough — no TLV
    /// parsing is applied. Same capacity-preserving swap as
    /// [`Self::drain_bulk`].
    pub fn drain_port2_bulk(&mut self) -> Vec<u8> {
        self.drain_port_tx(2)
    }

    /// Test helper — return all accumulated port-0 TX output as a string.
    #[cfg(test)]
    pub(crate) fn output_for_test(&self) -> String {
        let bytes: Vec<u8> = self.ports[0].tx_buf.iter().copied().collect();
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Test helper — inject bytes directly into port-2 TX buffer.
    /// Used by the `sched_stats` stale-bytes race-close regression
    /// test to simulate prior-request residue without a real guest
    /// relay. Production code never injects directly; bytes arrive
    /// via the guest's virtio TX queue processed by
    /// `process_tx(2)`.
    #[cfg(test)]
    pub(crate) fn inject_port2_tx_for_test(&mut self, bytes: &[u8]) {
        self.ports[2].tx_buf.extend(bytes);
    }

    /// Test helper — return a copy of the pending port-0 RX bytes
    /// (host → guest direction) that have not yet been delivered to
    /// the guest. Tests that exercise the host-side wake-byte
    /// pushers without a fully-wired guest queue use this to inspect
    /// what would have been delivered.
    #[cfg(test)]
    pub(crate) fn pending_rx_bytes_for_test(&self) -> Vec<u8> {
        self.ports[0].pending_rx.iter().copied().collect()
    }

    // ------------------------------------------------------------------
    // Port 0 RX: host → guest console
    // ------------------------------------------------------------------

    /// Push host data into the guest's port-0 RX buffers. Same
    /// semantics as the original single-port `queue_input` —
    /// undelivered bytes accumulate in port 0's `pending_rx` and
    /// drain on the next QUEUE_NOTIFY for q0.
    pub fn queue_input(&mut self, data: &[u8]) {
        tracing::debug!(bytes = data.len(), "virtio-console queue_input");
        self.ports[0].pending_rx.extend(data);
        self.drain_pending_rx(0);
    }

    // ------------------------------------------------------------------
    // Port 1 RX: host → guest bulk channel (TLV reply frames)
    // ------------------------------------------------------------------

    /// Push host data into the guest's port-1 RX buffers. Used by the
    /// freeze coordinator's snapshot-request handler to deliver a
    /// [`crate::vmm::wire::SnapshotReplyPayload`] back to the in-guest
    /// `request_snapshot` blocking reader. Bytes that cannot be
    /// delivered immediately (no chain available, port not opened
    /// yet, DRIVER_OK not set) accumulate in port 1's `pending_rx`
    /// and drain on the next q4 (`PORT1_RXQ`) notify.
    pub(crate) fn queue_input_port1(&mut self, data: &[u8]) {
        tracing::debug!(bytes = data.len(), "virtio-console queue_input_port1");
        self.ports[1].pending_rx.extend(data);
        self.drain_pending_rx(1);
    }

    // ------------------------------------------------------------------
    // Port 2 RX: host → guest scheduler-stats relay
    // ------------------------------------------------------------------

    /// Push host data into the guest's port-2 RX buffers. Used by the
    /// host's [`crate::vmm::sched_stats::SchedStatsClient`] to deliver
    /// scx_stats request bytes to the in-guest relay thread that
    /// forwards them to `/var/run/scx/root/stats`. Bytes that cannot
    /// be delivered immediately accumulate in port 2's `pending_rx`
    /// and drain on the next q6 (`PORT2_RXQ`) notify. Mirrors
    /// [`Self::queue_input_port1`].
    pub(crate) fn queue_input_port2(&mut self, data: &[u8]) {
        tracing::debug!(bytes = data.len(), "virtio-console queue_input_port2");
        self.ports[2].pending_rx.extend(data);
        self.drain_pending_rx(2);
    }

    /// Drop any host→guest port-2 request bytes that have not yet
    /// been consumed by the guest. Called by
    /// [`crate::vmm::sched_stats::SchedStatsClient::request_raw`] at the
    /// start of every fresh request: a freeze rendezvous that
    /// landed mid-request can leave half a JSON request line in port
    /// 2's `pending_rx`. If the next request just pushed onto the
    /// deque, the guest relay would read the previous-request tail
    /// concatenated with the new request, producing torn JSON the
    /// scheduler can't parse. Returns the number of bytes
    /// discarded so the caller can log/account for them.
    pub(crate) fn clear_port2_pending_rx(&mut self) -> usize {
        let pending = &mut self.ports[2].pending_rx;
        let n = pending.len();
        pending.clear();
        n
    }

    /// Drain a port's pending RX bytes into guest write-only
    /// descriptors. Replaces the prior per-port
    /// `drain_port{0,1,2}_pending_rx` triplet — port 1 and port 2
    /// add F_MULTIPORT and `opened` gates atop the port-0 baseline.
    /// Only publish a chain when ALL writes for that chain
    /// succeeded; otherwise keep bytes in `pending_rx` for retry.
    fn drain_pending_rx(&mut self, port_id: usize) {
        let port_label = port_label(port_id);
        let (queue_idx, _) = port_queues(port_id);
        if self.ports[port_id].pending_rx.is_empty() {
            return;
        }
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            tracing::debug!(
                port = port_label,
                pending = self.ports[port_id].pending_rx.len(),
                status = self.device_status,
                "virtio-console drain_pending_rx: DRIVER_OK not set; deferring"
            );
            return;
        }
        // F_MULTIPORT runtime gate: ports 1 and 2 are multiport-only.
        // The legacy single-console path never sees their traffic, so
        // any QUEUE_NOTIFY for a multiport-only RX queue without
        // F_MULTIPORT is a guest protocol violation. Bytes stay in
        // `pending_rx` for a future probe.
        if port_id != 0 && !self.multiport_negotiated() {
            tracing::warn!(
                port = port_label,
                pending = self.ports[port_id].pending_rx.len(),
                "virtio-console drain_pending_rx: F_MULTIPORT \
                 not negotiated; deferring host→guest bytes"
            );
            return;
        }
        // Per-port `opened` gate: ports 1 and 2 are multiport
        // channels that the kernel only opens after the host's
        // `PORT_OPEN` control message completes the handshake.
        // Pushing bytes before the open landed would let the kernel
        // discard them with no userspace reader. Port 0 (console)
        // starts implicitly open in the hvc-console path.
        if port_id != 0 && !self.ports[port_id].opened {
            tracing::debug!(
                port = port_label,
                pending = self.ports[port_id].pending_rx.len(),
                "virtio-console drain_pending_rx: port not yet opened by guest; deferring"
            );
            return;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => {
                tracing::debug!(
                    port = port_label,
                    pending = self.ports[port_id].pending_rx.len(),
                    "virtio-console drain_pending_rx: no mem"
                );
                return;
            }
        };
        if !self.queues[queue_idx].ready() {
            tracing::debug!(
                port = port_label,
                pending = self.ports[port_id].pending_rx.len(),
                "virtio-console drain_pending_rx: RX queue not ready"
            );
            return;
        }
        let q = &mut self.queues[queue_idx];
        let mut total_written = 0u32;
        let mut chains_drained = 0usize;
        while !self.ports[port_id].pending_rx.is_empty() {
            let Some(chain) = q.pop_descriptor_chain(mem) else {
                break;
            };
            let head = chain.head_index();
            let mut consumed_offset = 0usize;
            let mut written = 0u32;
            let mut chain_torn = false;
            for desc in chain {
                if desc.is_write_only() && consumed_offset < self.ports[port_id].pending_rx.len() {
                    let guest_addr = desc.addr();
                    let avail = desc.len() as usize;
                    let remaining = self.ports[port_id].pending_rx.len() - consumed_offset;
                    let chunk = remaining.min(avail);
                    self.rx_scratch.clear();
                    let (head_slice, tail_slice) = self.ports[port_id].pending_rx.as_slices();
                    let head_skip = consumed_offset.min(head_slice.len());
                    let tail_skip = consumed_offset - head_skip;
                    let head_avail = &head_slice[head_skip..];
                    let tail_avail = if tail_skip < tail_slice.len() {
                        &tail_slice[tail_skip..]
                    } else {
                        &[][..]
                    };
                    let h = head_avail.len().min(chunk);
                    self.rx_scratch.extend_from_slice(&head_avail[..h]);
                    if h < chunk {
                        let t = (chunk - h).min(tail_avail.len());
                        self.rx_scratch.extend_from_slice(&tail_avail[..t]);
                    }
                    if mem.write_slice(&self.rx_scratch, guest_addr).is_ok() {
                        let n = self.rx_scratch.len();
                        consumed_offset += n;
                        written += n as u32;
                    } else {
                        tracing::warn!(
                            port = port_label,
                            head,
                            written,
                            "virtio-console drain_pending_rx: write_slice failed \
                             mid-chain; breaking out to avoid partial-fill corruption"
                        );
                        chain_torn = true;
                        break;
                    }
                }
            }
            if chain_torn {
                // Publish the head with len=0 so the guest reclaims
                // the descriptor instead of leaking it (mirrors the
                // c_ivq torn-write path). Bytes stay in `pending_rx`
                // for retry on the next chain. Without this add_used
                // the chain head is consumed from avail but never
                // returned to the used ring, and the descriptor index
                // leaks until reset.
                if let Err(e) = q.add_used(mem, head, 0) {
                    tracing::warn!(
                        port = port_label,
                        head,
                        %e,
                        "virtio-console drain_pending_rx: add_used(0) \
                         after torn write failed; chain head leaked"
                    );
                }
                break;
            }
            if let Err(e) = q.add_used(mem, head, written) {
                tracing::warn!(
                    port = port_label,
                    head,
                    written,
                    %e,
                    "virtio-console RX add_used failed (used-ring address \
                     likely unmapped); bytes preserved in pending_rx for \
                     retry on the next drain cycle"
                );
                break;
            }
            self.ports[port_id].pending_rx.drain(..consumed_offset);
            total_written += written;
            chains_drained += 1;
            // Per-call chain count cap: bound the per-vCPU
            // MMIO-handler latency under a hostile guest that
            // publishes many zero-progress chains (zero-length or
            // missing-write-only descriptors). Remaining chains
            // stay in the avail ring; the next QUEUE_NOTIFY (or the
            // next host-side push) picks them up. Mirrors the
            // `CONTROL_CHAINS_PER_CALL_MAX` pattern in
            // `process_control_tx`.
            if chains_drained >= RX_CHAINS_PER_CALL_MAX {
                tracing::debug!(
                    port = port_label,
                    chains_drained,
                    cap = RX_CHAINS_PER_CALL_MAX,
                    pending = self.ports[port_id].pending_rx.len(),
                    "virtio-console drain_pending_rx: per-call chain \
                     cap reached; remaining chains deferred to next notify"
                );
                break;
            }
        }
        if total_written > 0 {
            tracing::debug!(
                port = port_label,
                delivered = total_written,
                pending = self.ports[port_id].pending_rx.len(),
                "virtio-console drain_pending_rx: delivered to guest",
            );
            self.signal_used();
        }
    }

    // ------------------------------------------------------------------
    // Control protocol (multiport)
    // ------------------------------------------------------------------

    /// Process the c_ovq (queue 3): guest-originated control messages.
    /// Reads each chain's head descriptor as a `VirtioConsoleControl`
    /// frame, dispatches by `event`, and add_useds the chain.
    fn process_control_tx(&mut self) {
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            tracing::debug!("virtio-console c_ovq: DRIVER_OK not set; ignoring notify");
            return;
        }
        // F_MULTIPORT runtime gate: c_ovq exists only when multiport
        // is negotiated. A guest that didn't ack F_MULTIPORT but
        // still fires QUEUE_NOTIFY for queue 3 is misbehaving;
        // refuse to walk the queue.
        if !self.multiport_negotiated() {
            tracing::warn!(
                "virtio-console c_ovq: F_MULTIPORT not negotiated; \
                 ignoring notify on multiport-only queue"
            );
            return;
        }
        // Move work into local Vec so we can release the queue borrow
        // before calling back into self for control_out enqueue.
        let mut events: Vec<VirtioConsoleControl> = Vec::new();
        // Track whether any add_used call succeeded so the
        // unconditional `signal_used` below becomes conditional —
        // mirrors `drain_control_in`'s `any_published` discipline.
        // A QUEUE_NOTIFY that finds the avail ring empty pops zero
        // chains, publishes zero used entries, and must NOT raise
        // an interrupt: an unprovoked irq turns every spurious
        // notify into work for the guest's vring_interrupt path
        // (which then re-checks an empty used ring and returns).
        let mut any_published = false;
        {
            let mem = match self.mem.as_ref() {
                Some(m) => m,
                None => return,
            };
            let q = &mut self.queues[C_OVQ];
            let mut chains_drained = 0usize;
            while let Some(chain) = q.pop_descriptor_chain(mem) {
                let head = chain.head_index();
                let mut total = 0u32;
                let mut buf = [0u8; VC_CONTROL_SIZE];
                let mut need = VC_CONTROL_SIZE;
                let mut filled = 0usize;
                for desc in chain {
                    if desc.is_write_only() || need == 0 {
                        continue;
                    }
                    let take = desc.len().min(need as u32) as usize;
                    if take == 0 {
                        continue;
                    }
                    if let Err(e) = mem.read_slice(&mut buf[filled..filled + take], desc.addr()) {
                        tracing::warn!(%e, head, "c_ovq read_slice failed");
                        break;
                    }
                    filled += take;
                    need -= take;
                    total += take as u32;
                }
                if filled == VC_CONTROL_SIZE
                    && let Ok(c) = VirtioConsoleControl::read_from_bytes(&buf)
                {
                    events.push(c);
                }
                // c_ovq buffers are device-readable (guest->host control
                // messages); per virtio v1.2 §2.7.8.2 the used-element
                // `len` counts bytes written to the device-WRITABLE
                // portion, which is 0 for a readable-only chain. The
                // guest's __send_control_msg discards this len, so 0 is
                // both spec-correct and guest-harmless; it also makes
                // every readable-consume add_used in this file uniformly
                // 0 (data TX, reset-drain, c_ivq). `total` (bytes
                // consumed) is retained for the diagnostic trace below.
                match q.add_used(mem, head, 0) {
                    Ok(()) => any_published = true,
                    Err(e) => {
                        tracing::warn!(
                            head,
                            total,
                            %e,
                            "virtio-console c_ovq add_used failed"
                        );
                    }
                }
                chains_drained += 1;
                // Per-call chain count cap: bound the per-vCPU
                // MMIO-handler latency under a hostile guest that
                // publishes thousands of control chains. Remaining
                // chains stay in the avail ring; the next
                // QUEUE_NOTIFY picks them up. Mirrors the
                // `TX_PER_CALL_MAX` pattern in `process_tx`.
                if chains_drained >= CONTROL_CHAINS_PER_CALL_MAX {
                    tracing::debug!(
                        chains_drained,
                        cap = CONTROL_CHAINS_PER_CALL_MAX,
                        "virtio-console process_control_tx: per-call chain \
                         cap reached; remaining chains deferred to next notify"
                    );
                    break;
                }
            }
        }
        for c in events {
            self.handle_control_event(c);
        }
        // Only kick the guest when the used ring actually advanced —
        // if no chain was popped (empty avail ring) or every add_used
        // failed, there is nothing for the guest to consume. Mirrors
        // the `if any_published { self.signal_used(); }` gate at the
        // tail of `drain_control_in`. Then attempt to push pending
        // outbound control messages onto c_ivq (which may have been
        // refilled by the guest after this notify) — `drain_control_in`
        // owns its own conditional `signal_used` so it remains correct
        // even when this branch skipped the kick.
        if any_published {
            self.signal_used();
        }
        self.drain_control_in();
    }

    fn handle_control_event(&mut self, c: VirtioConsoleControl) {
        let id = c.id;
        let event = c.event;
        let value = c.value;
        match event {
            VIRTIO_CONSOLE_DEVICE_READY => {
                if value != 1 {
                    tracing::warn!(value, "virtio-console DEVICE_READY value != 1");
                    return;
                }
                // Reject repeats: a hostile or buggy guest sending
                // DEVICE_READY a second time would re-enqueue PORT_ADD
                // for every port and grow `control_out` without
                // bound. The kernel sends DEVICE_READY exactly once
                // per probe (drivers/char/virtio_console.c
                // `virtcons_probe`), so any subsequent message is a
                // guest protocol violation.
                if self.device_ready {
                    tracing::warn!("virtio-console DEVICE_READY repeat ignored");
                    return;
                }
                self.device_ready = true;
                // Send PORT_ADD for every port we expose. value=1
                // matches QEMU (hw/char/virtio-serial-bus.c
                // `send_control_event(... PORT_ADD, 1)` at the
                // probe-time fanout); the kernel parser
                // (`handle_control_message`) ignores `value` for
                // PORT_ADD but the wire convention pins value=1.
                for port_id in 0..NUM_PORTS {
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id: port_id,
                            event: VIRTIO_CONSOLE_PORT_ADD,
                            value: 1,
                        }));
                }
            }
            VIRTIO_CONSOLE_PORT_READY => {
                if value != 1 {
                    // value=0 is the kernel's add_port-failed signal
                    // (drivers/char/virtio_console.c `add_port` fail
                    // path: `__send_control_msg(portdev, id,
                    // VIRTIO_CONSOLE_PORT_READY, 0)`). Surface as
                    // tracing::error so a guest probe failure is
                    // visible at the host log level rather than
                    // wedging silently behind a debug log. Other
                    // values are protocol violations from a
                    // misbehaving guest; warn covers them.
                    if value == 0 {
                        tracing::error!(
                            id,
                            "virtio-console PORT_READY value=0: guest \
                             reports add_port failure for this port \
                             (kernel virtio_console.c add_port error \
                             path). The port will not function and the \
                             control handshake for this port will not \
                             complete."
                        );
                    } else {
                        tracing::warn!(id, value, "virtio-console PORT_READY != 1");
                    }
                    return;
                }
                if id >= NUM_PORTS {
                    tracing::warn!(id, "virtio-console PORT_READY for unknown port");
                    return;
                }
                // Reject repeats: a hostile or buggy guest sending
                // PORT_READY a second time for the same port would
                // re-enqueue CONSOLE_PORT / PORT_OPEN / PORT_NAME
                // and grow `control_out` without bound. The kernel
                // sends PORT_READY exactly once per port
                // (drivers/char/virtio_console.c `add_port`), so any
                // subsequent message for the same port is a guest
                // protocol violation.
                if self.ports[id as usize].readied {
                    tracing::warn!(id, "virtio-console PORT_READY repeat ignored");
                    return;
                }
                self.ports[id as usize].readied = true;
                let name = self.ports[id as usize].name;
                if id == 0 {
                    // Console port: announce, name, then open. The
                    // CONSOLE_PORT marker tells the guest this port
                    // is the system console (drivers/char/virtio_console.c
                    // `handle_control_message` CONSOLE_PORT case sets
                    // `port->cons.hvc` and calls `init_port_console`).
                    // PORT_NAME creates the sysfs `name` attribute
                    // (`/sys/class/virtio-ports/vport0p0/name`) so
                    // tooling that scans port names can find port 0;
                    // without it the attribute does not exist. PORT_OPEN
                    // matches QEMU's emission order
                    // (hw/char/virtio-serial-bus.c — PORT_NAME before
                    // PORT_OPEN), keeping udev symlink creation ahead
                    // of any userspace open of `/dev/hvc0`.
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id,
                            event: VIRTIO_CONSOLE_CONSOLE_PORT,
                            value: 1,
                        }));
                    self.control_out.push_back(ControlOut::Name { id, name });
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id,
                            event: VIRTIO_CONSOLE_PORT_OPEN,
                            value: 1,
                        }));
                } else {
                    // Bulk data port: name then open. Order matches
                    // QEMU's PORT_READY handler in
                    // hw/char/virtio-serial-bus.c (PORT_NAME goes out
                    // before PORT_OPEN at lines 425 / 430). The
                    // kernel's `handle_control_message` PORT_NAME
                    // case creates the sysfs `name` attribute, which
                    // udev rules consume to symlink the port; sending
                    // PORT_OPEN first races udev's symlink creation
                    // against userspace opens of /dev/vport0p{1,2}.
                    self.control_out.push_back(ControlOut::Name { id, name });
                    self.control_out
                        .push_back(ControlOut::Cmd(VirtioConsoleControl {
                            id,
                            event: VIRTIO_CONSOLE_PORT_OPEN,
                            value: 1,
                        }));
                }
            }
            VIRTIO_CONSOLE_PORT_OPEN => {
                if id >= NUM_PORTS {
                    tracing::warn!(id, "virtio-console PORT_OPEN for unknown port");
                    return;
                }
                let now_open = value == 1;
                let was_open = self.ports[id as usize].opened;
                self.ports[id as usize].opened = now_open;
                // When ports 1 or 2 transition closed→open, kick the
                // matching pending-RX drain. The host may have queued
                // snapshot replies (port 1) or stats requests (port 2)
                // before the guest finished its PORT_OPEN handshake
                // (the bulk port appears asynchronously after
                // multiport completes); without a drain trigger here
                // those bytes would sit in the port's `pending_rx`
                // until the next q4 / q6 notify, which a guest still
                // in `read()` may not generate promptly. Port 0
                // (console) does not require this kick because port-0
                // RX has no `opened` gate.
                if now_open && !was_open && id != 0 {
                    self.drain_pending_rx(id as usize);
                }
            }
            other => {
                tracing::debug!(
                    id,
                    event = other,
                    value,
                    "virtio-console: unhandled c_ovq event"
                );
            }
        }
    }

    /// Push pending control messages onto c_ivq (host→guest control).
    /// One message per descriptor chain — we publish only when the
    /// chain has enough write-only space to hold the whole message.
    fn drain_control_in(&mut self) {
        if self.control_out.is_empty() {
            return;
        }
        if self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            return;
        }
        // F_MULTIPORT runtime gate: c_ivq is multiport-only. Without
        // negotiation the kernel never allocates this queue, so any
        // descriptor chain we'd find here is a guest protocol
        // violation. Defer the messages — they'll wait in
        // `control_out` until either F_MULTIPORT is negotiated on a
        // future probe (after reset) or the device is reset.
        if !self.multiport_negotiated() {
            tracing::warn!(
                pending = self.control_out.len(),
                "virtio-console c_ivq: F_MULTIPORT not negotiated; \
                 deferring control messages"
            );
            return;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => return,
        };
        if !self.queues[C_IVQ].ready() {
            return;
        }
        let q = &mut self.queues[C_IVQ];
        // Tracks whether any used-ring entry was published this
        // call (including add_used(0) for too-small / torn chains)
        // so signal_used fires. Without this, a sequence of
        // too-small chains would publish add_used(0) entries that
        // the guest never sees an interrupt for.
        let mut any_published = false;
        let mut scratch: Vec<u8> = Vec::with_capacity(64);
        while let Some(msg) = self.control_out.front() {
            let need = msg.len();
            let Some(chain) = q.pop_descriptor_chain(mem) else {
                break;
            };
            let head = chain.head_index();
            // Tally chain capacity (write-only bytes available).
            let segs: Vec<(u64, usize)> = chain
                .filter(|d| d.is_write_only())
                .map(|d| (d.addr().0, d.len() as usize))
                .collect();
            let avail: usize = segs.iter().map(|(_, l)| *l).sum();
            if avail < need {
                // Chain cannot hold this message — push it back via
                // add_used(0) so the guest reclaims it; the message
                // stays in `control_out` for the next chain. Try
                // the next chain instead of breaking out — the guest
                // may have published a mix of small (single-byte
                // probe) and properly-sized (PAGE_SIZE) chains, and
                // a too-small chain at the front of avail must not
                // strand a large message that a later chain in the
                // ring could hold.
                tracing::warn!(
                    head,
                    avail,
                    need,
                    "virtio-console c_ivq: chain too small for control \
                     message; trying next chain"
                );
                if let Err(e) = q.add_used(mem, head, 0) {
                    tracing::warn!(head, %e, "virtio-console c_ivq add_used(0) failed");
                } else {
                    any_published = true;
                }
                continue;
            }
            scratch.clear();
            msg.write_into(&mut scratch);
            let mut written = 0u32;
            let mut idx = 0usize;
            let mut torn = false;
            for (gpa, seg_len) in &segs {
                if idx >= scratch.len() {
                    break;
                }
                let chunk = (*seg_len).min(scratch.len() - idx);
                if let Err(e) =
                    mem.write_slice(&scratch[idx..idx + chunk], vm_memory::GuestAddress(*gpa))
                {
                    tracing::warn!(
                        head,
                        %e,
                        "virtio-console c_ivq write_slice failed mid-chain"
                    );
                    torn = true;
                    break;
                }
                idx += chunk;
                written += chunk as u32;
            }
            if torn {
                // Torn write: publish the head with len=0 so the
                // guest reclaims the descriptor without leaking it.
                // The kernel does NOT gate cpkt parsing on buf->len
                // — drivers/char/virtio_console.c
                // `handle_control_message` reads `cpkt = (... *)(buf
                // ->buf + buf->offset)` unconditionally, and
                // `control_work_handler` only uses `len` to clamp
                // `buf->len` (used later for PORT_NAME's name_size
                // computation). The actual protection against
                // dispatching a truncated frame is `find_port_by_id`
                // rejecting unknown port ids — for non-PORT_ADD
                // events with a stale/garbage id the kernel drops
                // the message before the switch. For PORT_NAME a
                // buf->len=0 produces an absurd name_size that
                // typically fails the kmalloc and skips the strscpy.
                // None of this is bulletproof; the safe path is to
                // never produce a torn write in the first place.
                // The control message stays at the front of
                // `control_out` for retry on the next chain.
                if let Err(e) = q.add_used(mem, head, 0) {
                    tracing::warn!(head, %e, "virtio-console c_ivq add_used after torn failed");
                } else {
                    any_published = true;
                }
                break;
            }
            if let Err(e) = q.add_used(mem, head, written) {
                tracing::warn!(
                    head,
                    written,
                    %e,
                    "virtio-console c_ivq add_used failed; control message lost"
                );
                break;
            }
            any_published = true;
            // Now safe to consume from the front.
            self.control_out.pop_front();
        }
        // Fire signal_used whenever any used-ring entry was
        // published this call. `any_published` covers the data
        // path AND the add_used(0) paths (too-small / torn) so a
        // sequence of failures still kicks an irq for the guest.
        if any_published {
            self.signal_used();
        }
    }

    // ------------------------------------------------------------------
    // MMIO register dispatch
    // ------------------------------------------------------------------

    /// Handle MMIO read at `offset` within the device's MMIO region.
    pub fn mmio_read(&self, offset: u64, data: &mut [u8]) {
        // Config space lives at 0x100..0x110 in the MMIO layout per
        // virtio-v1.2 §4.2.2; `struct virtio_console_config` is 12
        // bytes (cols u16, rows u16, max_nr_ports u32, emerg_wr u32).
        const CFG_BASE: u64 = 0x100;
        const CFG_END: u64 = CFG_BASE + 12;
        if (CFG_BASE..CFG_END).contains(&offset) {
            self.config_read(offset - CFG_BASE, data);
            return;
        }
        if data.len() != 4 {
            for b in data.iter_mut() {
                *b = 0xff;
            }
            return;
        }
        let val: u32 = match offset as u32 {
            VIRTIO_MMIO_MAGIC_VALUE => MMIO_MAGIC,
            VIRTIO_MMIO_VERSION => MMIO_VERSION,
            VIRTIO_MMIO_DEVICE_ID => VIRTIO_ID_CONSOLE,
            VIRTIO_MMIO_VENDOR_ID => VENDOR_ID,
            VIRTIO_MMIO_DEVICE_FEATURES => {
                let page = self.device_features_sel;
                if page == 0 {
                    self.device_features() as u32
                } else if page == 1 {
                    (self.device_features() >> 32) as u32
                } else {
                    0
                }
            }
            VIRTIO_MMIO_QUEUE_NUM_MAX => self
                .selected_queue()
                .map(|i| self.queues[i].max_size() as u32)
                .unwrap_or(0),
            VIRTIO_MMIO_QUEUE_READY => self
                .selected_queue()
                .map(|i| self.queues[i].ready() as u32)
                .unwrap_or(0),
            VIRTIO_MMIO_INTERRUPT_STATUS => self.interrupt_status,
            VIRTIO_MMIO_STATUS => self.device_status,
            VIRTIO_MMIO_CONFIG_GENERATION => self.config_generation,
            _ => 0,
        };
        tracing::debug!(offset, val, "virtio-console mmio_read");
        data.copy_from_slice(&val.to_le_bytes());
    }

    /// Read from device config space. Layout matches
    /// `struct virtio_console_config` byte-for-byte:
    ///   off 0  u16 cols       — only valid with F_SIZE
    ///   off 2  u16 rows       — only valid with F_SIZE
    ///   off 4  u32 max_nr_ports — only valid with F_MULTIPORT
    ///   off 8  u32 emerg_wr   — only valid with F_EMERG_WRITE
    /// We advertise F_MULTIPORT only, so cols/rows/emerg_wr return 0
    /// (the kernel reads them via `virtio_cread_feature` which is a
    /// no-op when the feature bit is not set).
    fn config_read(&self, offset: u64, data: &mut [u8]) {
        let mut cfg = [0u8; 12];
        // max_nr_ports at offset 4, LE u32.
        cfg[4..8].copy_from_slice(&NUM_PORTS.to_le_bytes());
        let start = offset as usize;
        let end = start.saturating_add(data.len());
        if end > cfg.len() {
            // Out-of-range read: fill with 0xff so a misbehaving guest
            // gets explicit garbage rather than stale stack values.
            for b in data.iter_mut() {
                *b = 0xff;
            }
            return;
        }
        data.copy_from_slice(&cfg[start..end]);
    }

    /// Handle MMIO write at `offset` within the device's MMIO region.
    pub fn mmio_write(&mut self, offset: u64, data: &[u8]) {
        // Config space at 0x100..0x10c is read-only for this device
        // (we do not advertise F_EMERG_WRITE). Drop writes silently
        // and log so a guest stack writing into config surfaces.
        if (0x100..0x10c).contains(&offset) {
            tracing::warn!(
                offset,
                len = data.len(),
                "virtio-console: guest write to read-only config space ignored"
            );
            return;
        }
        if data.len() != 4 {
            return;
        }
        let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        tracing::debug!(offset, val, "virtio-console mmio_write");
        match offset as u32 {
            VIRTIO_MMIO_DEVICE_FEATURES_SEL => self.device_features_sel = val,
            VIRTIO_MMIO_DRIVER_FEATURES_SEL => self.driver_features_sel = val,
            VIRTIO_MMIO_DRIVER_FEATURES => {
                if !self.features_write_allowed() {
                    return;
                }
                let page = self.driver_features_sel;
                if page == 0 {
                    self.driver_features =
                        (self.driver_features & 0xFFFF_FFFF_0000_0000) | val as u64;
                } else if page == 1 {
                    self.driver_features =
                        (self.driver_features & 0x0000_0000_FFFF_FFFF) | ((val as u64) << 32);
                }
            }
            VIRTIO_MMIO_QUEUE_SEL => self.queue_select = val,
            VIRTIO_MMIO_QUEUE_NUM if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_size(val as u16);
                }
            }
            VIRTIO_MMIO_QUEUE_READY if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_ready(val == 1);
                }
            }
            VIRTIO_MMIO_QUEUE_NOTIFY => {
                let idx = val as usize;
                match idx {
                    C_IVQ => self.drain_control_in(),
                    C_OVQ => self.process_control_tx(),
                    _ => match queue_to_port(idx) {
                        // Guest published RX buffers; drain any
                        // pending host→guest bytes (port-0 console
                        // wakes, port-1 TLV replies, port-2 stats
                        // requests) into the freshly available
                        // descriptors. When no bytes are pending the
                        // drain is a quick no-op — the guest
                        // publishes empty buffers as flow control
                        // even when the host has nothing to send.
                        Some((port_id, false)) => self.drain_pending_rx(port_id),
                        Some((port_id, true)) => {
                            let _ = self.process_tx(port_id);
                        }
                        None => {
                            tracing::debug!(idx, "virtio-console: notify on unused queue");
                        }
                    },
                }
            }
            VIRTIO_MMIO_INTERRUPT_ACK => {
                self.interrupt_status &= !val;
            }
            VIRTIO_MMIO_STATUS => {
                if val == 0 {
                    self.reset();
                } else {
                    self.set_status(val);
                }
            }
            VIRTIO_MMIO_QUEUE_DESC_LOW if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_desc_table_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_DESC_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_desc_table_address(None, Some(val));
                }
            }
            VIRTIO_MMIO_QUEUE_AVAIL_LOW if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_avail_ring_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_AVAIL_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_avail_ring_address(None, Some(val));
                }
            }
            VIRTIO_MMIO_QUEUE_USED_LOW if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_used_ring_address(Some(val), None);
                }
            }
            VIRTIO_MMIO_QUEUE_USED_HIGH if self.queue_config_allowed() => {
                if let Some(i) = self.selected_queue() {
                    self.queues[i].set_used_ring_address(None, Some(val));
                }
            }
            _ => {}
        }
    }

    /// Validate and apply a status transition per virtio-v1.2 §3.1.1.
    /// The driver must not clear bits. Each phase requires the previous
    /// phase's bits to be set. Invalid transitions are ignored.
    fn set_status(&mut self, val: u32) {
        let old = self.device_status;
        // Driver must not clear bits (except via reset, which writes 0).
        if val & self.device_status != self.device_status {
            tracing::warn!(
                old,
                val,
                "virtio-console set_status: rejected (clears bits) — \
                 virtio-v1.2 §3.1.1 status bits are monotone within a \
                 driver session. Hostile-guest FSM violations surface at \
                 this log level (matches virtio-blk's set_status \
                 rejection-warn pattern)."
            );
            return;
        }
        let new_bits = val & !self.device_status;
        // FAILED (virtio-v1.2 §2.1.1 bit 0x80) is the driver's
        // "I give up" signal — `virtio_add_status(dev,
        // VIRTIO_CONFIG_S_FAILED)` is the kernel's exit path on probe
        // failure (drivers/virtio/virtio.c:363, 570, 606, 643). It
        // can land at any FSM state, alone or alongside other bits
        // already set (the call is `dev->config->set_status(dev,
        // get_status() | FAILED)` so `val` is `current_status |
        // 0x80`). Accept the bit and store; do not gate on the FSM
        // ladder. Reject only if FAILED appears together with
        // unrecognised new bits — those are protocol violations
        // unrelated to the legitimate FAILED signal. Log warn so a
        // guest probe failure surfaces visibly rather than silently
        // wedging the device behind a rejection.
        if new_bits == VIRTIO_CONFIG_S_FAILED {
            self.device_status = val;
            tracing::warn!(
                old,
                new = val,
                "virtio-console set_status: guest set FAILED status \
                 (virtio-v1.2 §2.1.1 bit 0x80 — driver gave up on \
                 device probe). Stored without further FSM advance."
            );
            return;
        }
        let valid = match new_bits {
            VIRTIO_CONFIG_S_ACKNOWLEDGE => self.device_status == 0,
            VIRTIO_CONFIG_S_DRIVER => self.device_status == S_ACK,
            VIRTIO_CONFIG_S_FEATURES_OK => self.device_status == S_DRV,
            VIRTIO_CONFIG_S_DRIVER_OK => self.device_status == S_FEAT,
            _ => false,
        };
        if valid {
            self.device_status = val;
            tracing::debug!(old, new = val, "virtio-console set_status: accepted");
            // Once DRIVER_OK lands, drain any host bytes that arrived
            // during initialization.
            if new_bits == VIRTIO_CONFIG_S_DRIVER_OK {
                // F_MULTIPORT must be negotiated for the multiport
                // control protocol to function. Without it the
                // kernel's virtcons_probe (drivers/char/virtio_console.c)
                // takes the legacy single-console path
                // (`add_port(portdev, 0)` then no DEVICE_READY) and
                // never sends a control-queue handshake. Port 1
                // bytes would then sit in `ports[1].tx_buf` /
                // `ports[1].pending_rx` unconsumed, and the c_ivq /
                // c_ovq queues would never receive driver buffers.
                // Surface this loudly so the failure is visible
                // rather than wedging behind a silent fallback.
                if self.driver_features & (1u64 << (VIRTIO_CONSOLE_F_MULTIPORT as u64)) == 0 {
                    tracing::warn!(
                        driver_features = self.driver_features,
                        "virtio-console set_status DRIVER_OK: \
                         F_MULTIPORT (bit 1) not negotiated by \
                         driver. Multiport control protocol will \
                         not run; port 1 bulk channel will not \
                         function. Verify the guest kernel has \
                         CONFIG_VIRTIO_CONSOLE enabled and that \
                         feature negotiation completed before \
                         DRIVER_OK."
                    );
                }
                self.drain_pending_rx(0);
                self.drain_control_in();
            }
        } else {
            tracing::warn!(
                old,
                val,
                "virtio-console set_status: rejected (invalid transition) — \
                 virtio-v1.2 §3.1.1 ordering: ACK → DRIVER → FEATURES_OK \
                 → DRIVER_OK, one bit at a time. Hostile-guest FSM \
                 violations surface at this log level."
            );
        }
    }

    fn reset(&mut self) {
        // Drain pending guest TX chains BEFORE clearing device state.
        // The kernel's `kernel_restart` → `device_shutdown` path on
        // `force_reboot()` writes 0 to `VIRTIO_MMIO_STATUS` (the
        // caller of this reset) after the userspace `send_exit` /
        // `send_lifecycle` writes have published descriptor chains
        // into the avail ring but before the host has observed the
        // matching QUEUE_NOTIFY MMIO. Once we run the field-clear
        // sequence below, the queue's `desc_table` / `avail_ring`
        // GuestAddresses are reset to virtio-queue 0.17.0's
        // `DEFAULT_*_ADDR` sentinels (see `Queue::reset` in the
        // virtio-queue crate), making the original avail-ring GPA
        // unreachable. Walk every port's TX queue here while their
        // GPAs and the DRIVER_OK / F_MULTIPORT gates are still
        // valid; the bytes land in each port's `tx_buf`, which we
        // deliberately do NOT clear below so `collect_results`'s
        // `final_drain` can still surface them.
        //
        // `drain_tx_into_capture_buf` is the side-effect-free variant
        // of `process_tx` — no `signal_used` (the rebooting guest
        // will not consume the IRQ), no `tx_evt.write` (the freeze
        // coordinator must NOT be woken to drain the bulk port here:
        // this reset call site already holds the device's outer
        // mutex, and the coord's TOKEN_TX handler races to acquire
        // the same mutex when `tx_evt` fires — under burst workloads
        // where 16 workers / 8 vCPUs flood the port, that wake-and-
        // block pattern stalls coord teardown for the duration of
        // every reset-time drain). The captured bytes are surfaced
        // post-reset via `final_drain` instead.
        for port_id in 0..NUM_PORTS as usize {
            self.drain_tx_into_capture_buf(port_id);
        }
        self.device_status = 0;
        self.interrupt_status = 0;
        self.queue_select = 0;
        self.device_features_sel = 0;
        self.driver_features_sel = 0;
        self.driver_features = 0;
        // Per-port `tx_buf` is a host-side capture buffer; clearing
        // it here would discard bytes the guest already published
        // but the host hasn't drained yet. `pending_rx` is queue-side
        // (host-prepared bytes waiting to travel back into the
        // guest's RX ring) and has no post-reset consumer, so it is
        // still cleared. `opened` and `readied` reset to false so
        // a fresh probe re-runs the per-port handshake.
        for port in &mut self.ports {
            port.pending_rx.clear();
            port.opened = false;
            port.readied = false;
        }
        self.control_out.clear();
        self.device_ready = false;
        for q in &mut self.queues {
            q.reset();
        }
    }
}

#[cfg(test)]
mod tests;
