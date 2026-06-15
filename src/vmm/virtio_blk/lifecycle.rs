//! Virtio-blk worker-thread teardown: best-effort stop-fd signalling,
//! bounded join-with-timeout, panic-payload rendering, and the VirtioBlk
//! Drop impl. Split from device.rs; reaches the device struct + imports
//! via `use super::*`.
use super::*;
use std::sync::mpsc;
use std::thread;


/// Maximum number of retries [`signal_worker_stop`] performs when
/// `EventFd::write` returns `WouldBlock` (EAGAIN). The eventfd
/// counter saturates at `u64::MAX - 1`; reaching that value
/// requires `~2^64` unbalanced writes, which the device never
/// emits — each `reset()`/`Drop` writes the stop_fd exactly once
/// per fresh fd allocation. The retry loop exists strictly as
/// defense-in-depth against a future regression that re-uses a
/// long-lived stop_fd (or any other path that could let the
/// counter accumulate). 4 retries with `thread::yield_now`
/// between each gives the worker thread (running on the same
/// CPU under contention) a chance to drain the counter via its
/// `epoll_wait → read` cycle.
#[cfg(not(test))]
const STOP_FD_WRITE_MAX_RETRIES: u32 = 4;

/// Best-effort signal to the worker thread to exit by writing 1
/// to its `stop_fd`. Retries up to [`STOP_FD_WRITE_MAX_RETRIES`]
/// times on `WouldBlock` (EAGAIN — counter saturation),
/// yielding the scheduler between attempts so a co-located
/// worker can drain the eventfd counter. Logs the per-attempt
/// failure so the operator can see the rare path even when the
/// retry succeeds.
///
/// On exhaustion: log a structured warn and return — the caller
/// (`Drop` / `stop_worker_and_reclaim_state`) proceeds to the
/// join-with-timeout path. If the stop signal never reaches the
/// worker the join will time out and the existing
/// permanent-workerless diagnostic surfaces. The retry exists to
/// surface the failure-path itself; it does NOT promise the
/// worker will exit (only the join timeout does).
///
/// `device_id` is the per-device tracing tuple (stop_fd raw fd,
/// instance_id, capacity_sectors) so a warn can correlate to
/// the wedged device without the caller plumbing the same
/// fields through. Free function (not method) so the borrow is
/// limited to the EventFd reference; the caller still owns
/// `&mut self.worker.engine`.
#[cfg(not(test))]
pub(crate) fn signal_worker_stop(
    stop_fd: &EventFd,
    raw_fd: std::os::unix::io::RawFd,
    instance_id: u64,
    capacity_sectors: u64,
) {
    for attempt in 0..STOP_FD_WRITE_MAX_RETRIES {
        match stop_fd.write(1) {
            Ok(()) => return,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                tracing::warn!(
                    attempt,
                    stop_fd = raw_fd,
                    instance_id,
                    capacity_sectors,
                    "virtio-blk stop_fd write returned WouldBlock; \
                     eventfd counter likely saturated. Yielding and retrying"
                );
                std::thread::yield_now();
            }
            Err(e) => {
                tracing::error!(
                    attempt,
                    stop_fd = raw_fd,
                    instance_id,
                    capacity_sectors,
                    %e,
                    "virtio-blk stop_fd write failed with non-EAGAIN error; \
                     worker may not observe the stop signal — \
                     downstream join will surface the timeout"
                );
                return;
            }
        }
    }
    tracing::error!(
        max_retries = STOP_FD_WRITE_MAX_RETRIES,
        stop_fd = raw_fd,
        instance_id,
        capacity_sectors,
        "virtio-blk stop_fd write exhausted retries on WouldBlock; \
         worker did not consume the eventfd counter in time — \
         downstream join will surface the timeout and the device \
         enters the permanent-workerless state"
    );
}

/// Upper bound on how long [`VirtioBlk::drop`] will block while
/// joining the worker thread.
///
/// 1 s is a deliberate trade between two failure modes. Below 1 s,
/// the timeout would fire on healthy shutdowns under load — the
/// worker may be mid-`pread`/`pwrite` when `stop_fd` is signalled,
/// and a fast-but-not-instant drain (cold page cache, contended
/// disk) can take tens to hundreds of milliseconds before the
/// worker reaches the next `epoll_wait` and observes the stop. A
/// budget shorter than typical drain latency would log false
/// "wedged worker" warnings and detach threads that were about to
/// exit. Above 1 s, the budget would risk vCPU thread starvation
/// during freeze rendezvous: the freeze coordinator's SIGRTMIN
/// rendezvous timeout is 30 s and the vCPU thread can be mid-`drop`
/// at that moment, so any `Drop` blocking budget compounds with
/// other pre-rendezvous overhead.
///
/// The 1 s value is large enough to absorb realistic drain
/// latency on warm caches and small enough to keep the `Drop`
/// completion well below the rendezvous threshold.
pub(crate) const DROP_JOIN_TIMEOUT: Duration = Duration::from_secs(1);

/// Upper bound on how long [`VirtioBlk::reset`] (production
/// `WorkerEngine::Spawned` path) will block while joining the
/// outgoing worker thread before declaring it wedged and entering
/// the permanent-device-death state documented at
/// [`VirtioBlk::reset_engine_spawned`].
///
/// The same budget as [`DROP_JOIN_TIMEOUT`] (1 s) and for the same
/// reasons: a `reset()` runs on the vCPU thread that received the
/// `STATUS = 0` MMIO write, and that vCPU thread can be the next
/// SIGRTMIN target the freeze coordinator picks for a
/// failure-dump rendezvous (30 s wall budget at the coordinator
/// level — see `FREEZE_RENDEZVOUS_TIMEOUT` in
/// `src/vmm/freeze_coord.rs`). An unbounded `handle.join()` here would
/// block the vCPU through the worker's wedged `pread`/`pwrite`
/// (NFS stall, slow page cache, hung block device) and the freeze
/// would either time out empty or arrive minutes late. Capping at
/// the same 1 s the Drop path uses keeps the "reset takes ≤ 1 s
/// of vCPU time" invariant uniform — a guest issuing a re-bind
/// burst (multiple resets in flight from a confused driver) does
/// not compound the per-reset cap into a multi-second freeze
/// blocker.
///
/// Below 1 s would fire false-positive timeouts on healthy resets
/// where the worker is mid-sync on a contended disk; above 1 s
/// would let a single hung worker pin the vCPU past the freeze
/// coordinator's rendezvous tolerance.
///
/// On timeout the device enters the same permanent-workerless
/// state described in [`VirtioBlk::respawn_worker`]'s "Failure
/// consequences" section: future kicks land on a stale `kick_fd`
/// and the guest hangs on every request until
/// `kernel.hung_task_timeout_secs` (default 120 s) fires. Only
/// constructing a fresh `VirtioBlk` recovers IO service. This is
/// the explicit trade chosen over blocking a vCPU thread
/// indefinitely — the same trade [`DROP_JOIN_TIMEOUT`] makes for
/// the destructor path.
///
/// Visible to `cfg(test)` builds so the unit-test module can pin
/// the constant's value via `reset_join_timeout_matches_drop_budget`
/// without duplicating the literal. The production callsite in
/// [`VirtioBlk::stop_worker_and_reclaim_state`] is itself
/// `cfg(not(test))`, so the const stays unread in test builds —
/// the test module references it explicitly.
pub(crate) const RESET_JOIN_TIMEOUT: Duration = Duration::from_secs(1);

/// Outcome of a bounded join attempt by [`join_worker_with_timeout`].
///
/// The variants distinguish observable shutdown states so callers
/// can log appropriately and unit tests can assert which path the
/// worker took. `Joined` carries the recovered `BlkWorkerState`;
/// the other variants are valueless because the state is either
/// lost (panic) or still owned by a detached helper / worker
/// thread (timeout, helper failure).
pub(crate) enum JoinWithTimeoutOutcome {
    /// Worker exited normally and yielded its `BlkWorkerState`.
    /// `dead_code` allow: the carried state is consumed only by
    /// `stop_worker_and_reclaim_state` (cfg(not(test))). Under
    /// `cargo check --tests` no reader exists, but
    /// `join_worker_with_timeout` still constructs the variant
    /// and the value matters for production reset.
    #[allow(dead_code)]
    Joined(BlkWorkerState),
    /// Worker panicked. The variant carries the panic payload
    /// returned by `JoinHandle::join` so the caller can render it
    /// (commonly a `&'static str` or `String` from `panic!(…)`)
    /// into a log message via `Debug` or by downcasting.
    Panicked(Box<dyn std::any::Any + Send>),
    /// Worker did not exit within `timeout`. The original
    /// `JoinHandle` is held by the helper thread, which continues
    /// running until the worker finally exits.
    TimedOut,
    /// `thread::Builder::spawn` for the helper thread failed
    /// (typically `EAGAIN` from `RLIMIT_NPROC` or thread-count
    /// exhaustion). The original handle was dropped — the worker
    /// is detached.
    HelperSpawnFailed,
    /// Helper thread itself panicked before forwarding the join
    /// result. Worker's outcome is unknown.
    HelperDisconnected,
}

/// Best-effort conversion of a `JoinHandle::join` panic payload to
/// a borrowed `&str`. Matches the two variants `panic!(…)` emits
/// in safe code: `&'static str` for `panic!("literal")` and
/// `String` for `panic!("{}", x)` / `panic!(format!(…))`. Other
/// payload types fall through to the placeholder `<non-string panic>`.
pub(crate) fn panic_payload_str(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else {
        "<non-string panic>"
    }
}

/// Join `handle` with an upper bound on the calling thread's wait
/// time.
///
/// Spawns a short-lived `ktstr-vblk-drop` helper thread that
/// performs the blocking `JoinHandle::join` and forwards the
/// result on an `mpsc::channel`. The calling thread waits via
/// `recv_timeout`; on timeout the helper is left running with the
/// handle and the calling thread returns. This bounds the
/// worst-case duration even when the worker is wedged in a
/// blocking syscall that does not check `stop_fd`
/// (`pread`/`pwrite` on slow backing, hung NFS, etc.). The vCPU
/// thread — which calls `VirtioBlk::drop` post-reset — therefore
/// cannot miss a SIGRTMIN delivery during freeze rendezvous
/// because the worker is hung.
///
/// # Outcomes
///
/// - [`JoinWithTimeoutOutcome::Joined`] — worker exited within
///   `timeout`; state recovered.
/// - [`JoinWithTimeoutOutcome::Panicked`] — worker exited within
///   `timeout`, but with a panic; state lost. The `Box<dyn Any +
///   Send>` payload returned by `JoinHandle::join` is propagated
///   so the caller can render it via [`panic_payload_str`] or by
///   downcasting to a concrete type.
/// - [`JoinWithTimeoutOutcome::TimedOut`] — worker did not exit
///   within `timeout`. Helper retains the `JoinHandle` and (through
///   it) the worker's `BlkWorkerState` until the worker finally
///   exits; if the worker never exits (perpetually-stuck IO), the
///   state outlives the device.
/// - [`JoinWithTimeoutOutcome::HelperSpawnFailed`] — the helper
///   thread itself could not be created (`RLIMIT_NPROC`,
///   thread-count exhaustion). Falling back to a direct
///   `handle.join()` would re-introduce the unbounded block this
///   function exists to prevent, so the handle is dropped and the
///   worker is detached.
/// - [`JoinWithTimeoutOutcome::HelperDisconnected`] — the helper
///   thread panicked before forwarding the join result. Worker's
///   outcome is unknown; the helper's `JoinHandle<()>` is dropped
///   when this function returns, detaching it.
///
/// # Resource retention on timeout
///
/// `BlkWorkerState` owns a `File`, an `Arc<VirtioBlkCounters>`,
/// two scratch `Vec`s, and two `TokenBucket`s. On timeout these
/// are reclaimed only when the worker thread finally exits; if it
/// does not, they outlive the device. This is the explicit trade
/// chosen over blocking a vCPU thread indefinitely. (The worker
/// also retains an `Arc<GuestMemoryMmap>` and the queue Arc clones
/// it was spawned with; those are part of the worker thread's
/// stack frame, not `BlkWorkerState`, but the same retention
/// applies — they live until the worker exits.)
pub(crate) fn join_worker_with_timeout(
    handle: thread::JoinHandle<BlkWorkerState>,
    timeout: Duration,
) -> JoinWithTimeoutOutcome {
    let (tx, rx) = mpsc::channel();
    let spawn_result = thread::Builder::new()
        .name("ktstr-vblk-drop".to_string())
        .spawn(move || {
            // Forward the join result. `send` failure means the
            // calling thread already gave up on `recv_timeout`
            // and dropped `rx`; the helper still owns the joined
            // state until this closure returns.
            let _ = tx.send(handle.join());
        });
    let _helper = match spawn_result {
        Ok(h) => h,
        Err(_) => return JoinWithTimeoutOutcome::HelperSpawnFailed,
    };
    match rx.recv_timeout(timeout) {
        Ok(Ok(state)) => JoinWithTimeoutOutcome::Joined(state),
        Ok(Err(payload)) => JoinWithTimeoutOutcome::Panicked(payload),
        Err(mpsc::RecvTimeoutError::Timeout) => JoinWithTimeoutOutcome::TimedOut,
        Err(mpsc::RecvTimeoutError::Disconnected) => JoinWithTimeoutOutcome::HelperDisconnected,
    }
}

/// `Drop` matches on `WorkerEngine` rather than gating the entire
/// impl on `cfg(not(test))`: the Inline branch is a no-op (the
/// default Drop drops `BlkWorkerState` cleanly when the engine
/// goes out of scope), the Spawned branch signals via `stop_fd`
/// and joins the worker thread so its resources (state, queues,
/// Arcs, eventfd clones) are reclaimed before `VirtioBlk` is
/// fully torn down.
///
/// The unconditional impl removes a fragility: a cfg-gated Drop
/// silently disappears in `cfg(test)`, so any pre-Drop side effect
/// added later (e.g. `tracing::debug!` on shutdown) would be
/// missing in tests. Pattern-matching the engine variant inside a
/// single impl keeps the dispatch obvious and makes adding such
/// side effects symmetric across cfgs. A regression that detached
/// the worker thread without stopping it would leave a daemon
/// thread holding the queue Arcs and the backing file open after
/// the device is dropped — visible as "test process leaks fds and
/// threads under stress."
///
/// # Bounded join
///
/// The Spawned arm quiesces the worker thread (production
/// `WorkerEngine::Spawned` path) by writing the `stop_fd` and
/// joining the thread with [`DROP_JOIN_TIMEOUT`] via
/// [`join_worker_with_timeout`]. On timeout the helper thread
/// retains the `JoinHandle` and the calling thread returns
/// without blocking further. The match arms log per-outcome
/// diagnostics — every error arm emits a structured `tracing`
/// event so the operator can correlate a missing-VM teardown
/// against the originating device. `JoinWithTimeoutOutcome::Joined`
/// is silent (clean shutdown is not logged). See
/// [`join_worker_with_timeout`] for full outcome semantics and
/// resource-retention notes, and [`DROP_JOIN_TIMEOUT`] for why
/// the budget is set where it is.
///
/// # Resource retention on `TimedOut`
///
/// When the worker join exceeds [`DROP_JOIN_TIMEOUT`] (the
/// `JoinWithTimeoutOutcome::TimedOut` arm), the [`Drop`] returns
/// without calling [`std::thread::JoinHandle::join`] — the
/// helper thread is detached and the worker keeps running. Every
/// `Arc` the worker holds remains live until the worker thread
/// exits naturally (typically when its blocking syscall
/// returns) and its captured state finally drops.
///
/// The retained Arcs are:
/// - `Arc<OnceLock<GuestMemoryMmap>>` (the `mem` field;
///   cloned into the worker thread frame). The guest memory
///   mapping stays mapped on the host until the worker exits —
///   the parent VM's teardown does NOT free guest memory at the
///   `VirtioBlk::drop` site.
/// - `Arc<EventFd>` (the IRQ eventfd, `irq_evt`). The eventfd's
///   kernel object stays alive; the kvmfd irqfd binding the
///   parent VM held does not unwind synchronously.
/// - `Arc<AtomicU32>` (the `interrupt_status` register, used
///   for the worker's release-store of `VIRTIO_MMIO_INT_VRING`).
/// - `Arc<AtomicBool>` (the `mem_unset_warned` one-shot latch).
/// - `Arc<VirtioBlkCounters>` (the per-device counter Arc the
///   worker increments on each request).
///
/// Operationally: a wedged worker means the VM teardown returns
/// to the caller (the calling thread is freed promptly, which is
/// the [`DROP_JOIN_TIMEOUT`] mechanism's whole point — usually a
/// vCPU thread that the freeze coordinator must not pin) but
/// the per-device shared state stays mapped until the kernel
/// eventually unblocks the worker. For long-lived host
/// processes that build many VMs, this can accumulate retained
/// memory; restart the host process to flush all leaked
/// per-device state. Bug reports mentioning "host RSS keeps
/// climbing across many ktstr test runs even though no VM is
/// active" should investigate `tracing::warn!` lines from this
/// arm to identify the wedged device(s).
impl Drop for VirtioBlk {
    fn drop(&mut self) {
        // Snapshot the device-identifier fields BEFORE the
        // match so the per-arm logs can correlate the device
        // across multiple concurrent VirtioBlk drops without
        // borrowing `self` after the `&mut self.worker.engine`
        // mutable borrow lands. None of the three are stable
        // across host restarts (`stop_fd` recycles, `instance_id`
        // resets at process start) but together they uniquely
        // identify the device within this process run.
        // `instance_id` replaces an earlier `self as *const _`
        // pointer field — the pointer leaked the host's ASLR
        // layout into log output (environment leakage); the
        // process-local counter has the same uniqueness shape
        // without the leak.
        //
        // The cfg(test) Inline arm doesn't consume these
        // snapshots; the `let _ = (capacity_sectors, instance_id);`
        // reference inside that arm satisfies the
        // `unused_variables` lint under cfg(test) where the
        // Spawned arm is excluded. (`stop_fd` is read inside the
        // cfg(not(test)) Spawned arm directly, so it doesn't
        // need the same dead-code dance.)
        let capacity_sectors = self.capacity_sectors;
        let instance_id = self.instance_id;
        match &mut self.worker.engine {
            #[cfg(test)]
            WorkerEngine::Inline(engine) => {
                // Default-drop the inline state when this fn returns.
                // Reference the snapshot vars to avoid `unused`
                // lints in cfg(test).
                let _ = (capacity_sectors, instance_id);
                // Decrement the live "currently waiting for tokens"
                // gauge if the device is being dropped while a
                // chain is rollback-stalled. Symmetric with
                // `reset_engine_inline`'s mid-stall path: the
                // chain is gone from the device's perspective, so
                // the gauge must match. Without this, an external
                // observer that cloned the counters Arc before
                // drop sees one stranded increment per
                // drop-while-stalled. The shared gauge is
                // saturating (see `record_throttle_pending_dec`),
                // so this dec is safe even if a racing path
                // already decremented.
                if engine.state.currently_stalled {
                    engine.state.currently_stalled = false;
                    engine.state.counters.record_throttle_pending_dec();
                }
            }
            #[cfg(not(test))]
            WorkerEngine::Spawned(eng) => {
                // The third device-identifier field (`stop_fd`
                // raw fd) is only meaningful in the Spawned
                // arm — Inline mode has no eventfd to name.
                let stop_fd = eng.stop_fd.as_raw_fd();
                // Unpause first so a parked worker observes the
                // upcoming stop signal. Same rationale as
                // `reset_engine_spawned`: a worker stuck in its
                // `park_timeout(10ms)` Acquire-load loop is
                // unreachable from `epoll_wait`, so STOP_TOKEN
                // would block until the 10 ms tick + Acquire-load
                // sees the cleared flag. Clearing here makes the
                // worker exit the park within 10 ms (faster on
                // the unpark hint) so the join timeout window
                // (DROP_JOIN_TIMEOUT, 1 s) is not consumed by
                // park latency alone.
                self.paused.store(false, Ordering::Release);
                if let Some(ref handle) = eng.handle {
                    handle.thread().unpark();
                }
                // Signal the worker to exit via the stop_fd
                // helper, which retries on EAGAIN (eventfd
                // counter saturation) up to STOP_FD_WRITE_MAX_RETRIES
                // times before giving up. On exhaustion the join
                // below absorbs the failure via DROP_JOIN_TIMEOUT.
                signal_worker_stop(&eng.stop_fd, stop_fd, instance_id, capacity_sectors);
                if let Some(handle) = eng.handle.take() {
                    match join_worker_with_timeout(handle, DROP_JOIN_TIMEOUT) {
                        JoinWithTimeoutOutcome::Joined(state) => {
                            // Clean shutdown. If the worker exited
                            // while a chain was rollback-stalled
                            // (worker observed STOP_TOKEN before
                            // any post-stall successful drain
                            // could clear the per-worker flag),
                            // decrement the live "currently
                            // waiting for tokens" gauge to match —
                            // the chain is gone from the device's
                            // perspective. Without this, every
                            // drop-while-stalled pins one
                            // increment on the shared counters
                            // Arc for any external observer
                            // (failure-dump renderer, host
                            // monitor) that cloned the Arc
                            // before drop. Symmetric with
                            // `reset_engine_spawned`'s mid-stall
                            // path. Saturating dec (see
                            // `record_throttle_pending_dec`)
                            // makes a redundant bump safe.
                            if state.currently_stalled {
                                state.counters.record_throttle_pending_dec();
                            }
                            // State drops at scope end.
                        }
                        JoinWithTimeoutOutcome::Panicked(payload) => {
                            // Worker panicked — its `BlkWorkerState`
                            // is lost (panic propagation drops
                            // owned values without giving us
                            // access). If a stall was in flight,
                            // the gauge increment leaks for the
                            // device's lifetime. The doc on
                            // `VirtioBlkCounters::currently_throttled_gauge`
                            // documents this acceptable leak —
                            // operators must not depend on a
                            // strictly zero-on-shutdown gauge.
                            tracing::error!(
                                panic = panic_payload_str(&*payload),
                                stop_fd,
                                capacity_sectors,
                                instance_id,
                                "virtio-blk worker thread panicked"
                            );
                        }
                        JoinWithTimeoutOutcome::TimedOut => {
                            tracing::warn!(
                                timeout_s = DROP_JOIN_TIMEOUT.as_secs_f32(),
                                stop_fd,
                                capacity_sectors,
                                instance_id,
                                "virtio-blk worker did not exit within \
                                 DROP_JOIN_TIMEOUT of stop_fd; leaking \
                                 the worker thread to avoid blocking the \
                                 calling thread (likely a vCPU). Worker \
                                 is wedged in a blocking syscall that \
                                 does not check stop_fd. \
                                 hint: identify the wedged device by \
                                 stop_fd / instance_id / capacity_sectors \
                                 above; per-device GuestMemoryMmap and \
                                 EventFd Arcs stay live until the worker \
                                 unblocks (see Drop's resource-retention \
                                 doc). hint: kill -USR1 the host process \
                                 to dump worker thread backtraces, OR \
                                 check `dmesg` for the backing fd's \
                                 storage path stalling on I/O."
                            );
                        }
                        JoinWithTimeoutOutcome::HelperSpawnFailed => {
                            tracing::error!(
                                stop_fd,
                                capacity_sectors,
                                instance_id,
                                "virtio-blk drop helper thread spawn \
                                 failed; detaching worker without join"
                            );
                        }
                        JoinWithTimeoutOutcome::HelperDisconnected => {
                            tracing::error!(
                                stop_fd,
                                capacity_sectors,
                                instance_id,
                                "virtio-blk drop helper thread \
                                 terminated without forwarding the \
                                 worker join result"
                            );
                        }
                    }
                }
            }
        }
    }
}
