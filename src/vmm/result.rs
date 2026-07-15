//! Public [`VmResult`] returned from [`super::KtstrVm::run`], plus
//! the internal [`VmRunState`] passed from `run_vm` to
//! `collect_results` and the [`KvmStatsTotals`] aggregate of per-vCPU
//! KVM counters.
//!
//! The split keeps the result-shaping types independent of the
//! orchestration code (which still lives in [`super::KtstrVm`]). Test
//! code outside `vmm/` constructs `VmResult` literals and reads
//! `KvmStatsTotals` fields, so both types stay public; `VmRunState`
//! is `pub(crate)`-only because it's an implementation detail of the
//! run-then-collect handoff.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::console;
use super::host_comms::BulkDrainResult;
use super::kvm;
use super::pi_mutex::PiMutex;
use super::vcpu::{VcpuThread, WatchpointArm};
use super::virtio_blk::{VirtioBlkCounters, VirtioBlkCountersSnapshot};
use super::virtio_net::{VirtioNetCounters, VirtioNetCountersSnapshot};
use super::wire;
use crate::monitor;

/// Which watchdog rule (if any) killed the VM — the public mirror of
/// the watchdog-internal `freeze_coord::KillReasonTag`, rendered as
/// `cause=` in the deadline-expired stderr dump. `None` on
/// [`VmResult::watchdog_kill_reason`] means no watchdog kill was
/// recorded (clean exit, crash, or a timeout path that never stamped a
/// reason).
///
/// The variant IS the mechanism assertion e2e fixtures need: a
/// wedge-injection fixture proves its tier by matching the reason,
/// which stays correct at ANY host dilation — unlike a wall-time bound,
/// which conflates detection latency with host CPU share (an observed
/// arm64 cell proved Tier-1 firing exactly on its 12 s CPU budget while
/// taking 83 s of wall at ~14% share).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchdogKillReason {
    /// Tier-1: max per-vCPU CPU burned past the phase budget with no
    /// milestone — a spinning wedge.
    Tier1CpuBudget,
    /// Tier-2: wall past the phase backstop with live evidence channels
    /// and no runnable demand — a silent idle wedge.
    Tier2IdleWedge,
    /// Tier-3: the hard deadline expired and the deadman was not
    /// deferred (monitor dead, or cell inert past the grace).
    Tier3Deadman,
    /// An AP set the kill flag (panic-driven), not a watchdog expiry.
    ApKill,
}

/// Final guest lifecycle stage as tracked by the progress ledger —
/// the public mirror of `monitor::LifecycleStage` (same forward-only
/// ordinal order, so `Ord` compares boot progress). Snapshotted at
/// run teardown onto [`VmResult::final_guest_phase`].
///
/// Fixtures use this to distinguish "the guest reached the phase under
/// test and the watchdog misfired" (a real detection regression) from
/// "the guest never got there" (an environmental non-verdict: an
/// observed arm64 cell at 0.3% host CPU share sat in `Boot` for 172 s —
/// nothing phase-dependent can be asserted about a guest that could
/// not boot).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GuestLifecyclePhase {
    /// Booting: init up to SYS_RDY.
    Boot,
    /// SYS_RDY seen; scheduler attaching.
    Attach,
    /// Guest dispatch entering the workload preamble.
    Dispatch,
    /// The test body / workload proper.
    Body,
    /// Winding down (`ScenarioEnd` / `Exit` / `TestResult` seen).
    Teardown,
}

/// Result of a VM execution.
///
/// `Clone` is supported, but two field categories have different
/// Clone semantics that callers must understand:
///
/// 1. **Pure-data fields** (the bulk of the struct): primitives,
///    `String`, `Vec`, `Option<_>`, plus `MonitorReport` /
///    `BulkDrainResult` / `ProgVerifierStats` / `StimulusEvent` /
///    `KvmStatsTotals` / `VirtioBlkCountersSnapshot` /
///    `VirtioNetCountersSnapshot`. Every clone produces an
///    independent value — mutations to one do not affect the
///    other. The `virtio_blk_counters` / `virtio_net_counters`
///    fields are materialized `*CountersSnapshot` types (atomic
///    loads done at construction time inside
///    `super::KtstrVm::collect_results`), so clones cannot alias
///    live device state.
///
/// 2. **Arc-shared handles** (`snapshot_bridge`, `stats_client`):
///    these wrap `Arc<Mutex<…>>` / `Arc<AtomicUsize>` and clone via
///    shallow refcount bump. Two `VmResult` clones SHARE the
///    underlying store — calling `snapshot_bridge.drain()` on one
///    clone empties the data visible to the other. See each
///    field's own doc for the precise drain / iteration contract.
///    If you need an independent snapshot view, drain into a local
///    `Vec` before cloning the `VmResult`.
///
/// 3. **The capture-series cache** (`periodic_series_cache`): a
///    `OnceLock<SampleSeries>` that memoizes the one destructive drain
///    of the category-2 `snapshot_bridge` (see
///    [`Self::captures_series`]). Its clone behavior depends on whether
///    it is populated at clone time:
///    - **Populated** (any of [`Self::captures_series`] /
///      [`Self::periodic_series`] / [`Self::phase_buckets`] was already
///      called): the clone carries an INDEPENDENT copy of the cached
///      series — category-1 semantics. Both clones return the same
///      captures without touching the (now-drained) shared bridge.
///    - **Empty**: the clone shares the category-2 bridge, so the
///      FIRST `captures_series()` call on EITHER clone performs the
///      single drain and the other clone — if it later drains the same
///      shared bridge — sees nothing. To give each clone its own
///      buckets, call [`Self::captures_series`] (or any accessor that
///      routes through it) on the original BEFORE cloning.
#[derive(Debug, Clone)]
pub struct VmResult {
    /// Overall success flag: `true` when the test reported a pass AND
    /// the VM exited cleanly without crash, timeout, or watchdog.
    pub success: bool,
    /// Guest vCPU count for this run (topology llcs*cores*threads),
    /// carried from `KtstrVm` to the sidecar `(vcpus, cpu_budget)` stamp.
    pub vcpus: u32,
    /// Effective host-CPU budget the vCPU threads ran on; stamped to the
    /// sidecar. Normally `KtstrVm::effective_cpu_budget` (the build-time
    /// plan's CPU count), but the default-overcommit path (host too small
    /// for a 1:1 pin) overrides it in `run()` with the actual masked
    /// host-CPU count, since the build-time value assumes 1:1 pinning.
    /// Below `vcpus` means host overcommit, which confounds the
    /// guest-scheduler timing metrics.
    pub cpu_budget: u32,
    /// How the userspace scheduler binary was resolved for this run —
    /// the snake_case `crate::test_support::ResolveSource::as_str` tag
    /// (`"auto_built"`, `"target_debug"`, `"path"`, ...). Stamped by the
    /// host eval layer (`run_ktstr_test_inner_impl`) AFTER the run,
    /// alongside `entry_name` / `variant_hash`, then carried to the
    /// sidecar `resolve_source` stamp the same way `vcpus` / `cpu_budget`
    /// are. `None` for VmResults built outside the host eval path (the
    /// freeze coordinator, test fixtures) — those resolve no scheduler
    /// binary.
    pub resolve_source: Option<String>,
    /// True when the `#[ktstr_test(expect_auto_repro)]` attribute set
    /// `expect_auto_repro = true` on the entry AND the auto-repro
    /// path fired with a valid repro artifact during the run — the
    /// signal that the verdict-flip from fail-with-artifact → PASS
    /// is satisfied.
    ///
    /// The eval-layer derives this field AFTER `evaluate_vm_result`
    /// returns (preserving the original `success` + error chain for
    /// diagnostic visibility); the eval layer then wraps any
    /// failure `Err` with the
    /// `crate::test_support::eval::ExpectAutoReproSatisfied`
    /// marker, and the dispatch arm
    /// (`crate::test_support::dispatch::result_to_exit_code`)
    /// downcasts the marker and routes the verdict to `EXIT_PASS`
    /// without mutating the original `success` or stripping the
    /// error chain. Pattern mirrors the `expect_err` matcher
    /// inversion.
    ///
    /// Default `false`. When `expect_auto_repro = false` (the
    /// macro-attribute default) the eval layer skips the artifact
    /// probe entirely and leaves the field at `false`, so the
    /// dispatch arm is never matched and the original verdict
    /// stands.
    pub expect_auto_repro_satisfied: bool,
    /// Guest exit code as surfaced through the SHM ring
    /// (`MSG_TYPE_EXIT`) or COM2 sentinel.
    pub exit_code: i32,
    /// Wall-clock duration of the VM run.
    pub duration: Duration,
    /// True when the host hit its watchdog before the guest exited.
    pub timed_out: bool,
    /// Which watchdog rule killed the VM (see [`WatchdogKillReason`]).
    /// `None` when no watchdog kill was recorded — clean exits, but
    /// also crash paths where the kill flag was never watchdog-set.
    /// Loaded from the watchdog thread's kill-reason latch after its
    /// join, so the value is final.
    pub watchdog_kill_reason: Option<WatchdogKillReason>,
    /// Final guest lifecycle stage per the progress ledger (see
    /// [`GuestLifecyclePhase`]). Snapshotted at run teardown; stays
    /// `Boot` when the guest never advanced (or no dispatch thread
    /// consumed lifecycle frames — e.g. host-only paths).
    pub final_guest_phase: GuestLifecyclePhase,
    /// Final milestone count from the progress ledger (the dump's
    /// `progress_epoch=` value): lifecycle-stage advances plus other
    /// dispatch-recorded milestones. `0` means the guest never
    /// published a milestone the host consumed.
    pub final_progress_epoch: u64,
    /// Whether the entry's host-triggered BPF-map-write injections
    /// (`KtstrTestEntry::bpf_map_write` — e.g. the neg_* fixtures'
    /// crash flag) were delivered: `None` when no writes were
    /// configured, `Some(true)` when every queued write landed and the
    /// guest was signalled, `Some(false)` when the injection thread
    /// never completed (killed mid-retry — e.g. the run ended while a
    /// starved guest was still booting/attaching). Fixtures that
    /// expect an injected bug consult this to distinguish "the bug
    /// never fired because the injection never landed" (environmental
    /// under witnessed contention) from a real detection failure.
    pub bpf_map_writes_delivered: Option<bool>,
    /// Run-relative instant the periodic-capture prereqs (KASLR publish
    /// and the map/prog accessors) became ready; `None` when they never
    /// did. With [`Self::periodic_window_end`] this is the READINESS-VS-WINDOW
    /// evidence: readiness at/after the window end (or never) makes zero
    /// captures structurally inevitable at ANY host dilation — the
    /// capture-starvation gates skip on it instead of a dilation
    /// threshold (a ~7 s cold accessor build outruns a 4-5 s window at
    /// D ≈ 1.2, below any honest contention bar).
    pub periodic_prereqs_ready: Option<Duration>,
    /// Run-relative instant the periodic capture window ends (the
    /// workload-end clamp); `None` when the window never resolved —
    /// which itself means no boundary could ever fire.
    pub periodic_window_end: Option<Duration>,
    /// Captured guest stdout (and any non-dmesg serial console content).
    pub output: String,
    /// Captured guest stderr (separated from `output` when the guest
    /// reported them distinctly).
    pub stderr: String,
    /// Host-side monitor report: sampled per-CPU state, stuck
    /// verdicts, and SCX event deltas. `None` when the monitor did
    /// not run (host-only tests, early VM failure).
    pub monitor: Option<monitor::MonitorReport>,
    /// TLV messages drained from the guest after VM exit. Merges
    /// mid-flight bytes the freeze coordinator pulled off
    /// virtio-console port 1 during the run with the final port-1
    /// `port1_tx_buf` flush.
    pub guest_messages: Option<BulkDrainResult>,
    /// BPF verifier stats collected from host-side memory reads.
    pub verifier_stats: Vec<monitor::bpf_prog::ProgVerifierStats>,
    /// KVM per-vCPU cumulative stats (requires Linux >= 5.14).
    pub kvm_stats: Option<KvmStatsTotals>,
    /// Crash message extracted from COM2 output via
    /// `crate::test_support::extract_panic_message`. The guest
    /// panic hook in `rust_init/init.rs` writes `PANIC: <info>\n<bt>\n`
    /// to `/dev/ttyS1` synchronously inside `KVM_RUN`, so the host
    /// captures the full backtrace in `output` even when the guest
    /// is wedged. `None` when no `PANIC:`-prefixed line was seen.
    pub crash_message: Option<String>,
    /// Wall-clock time from BSP exit to the moment
    /// `super::KtstrVm::collect_results` finishes assembling
    /// [`VmResult`].
    /// Records the host-side cost of every teardown step that runs
    /// after the guest has stopped advancing: watchdog join, AP joins,
    /// monitor join, BPF-writer join, SHM drain, exit/crash-message
    /// extraction, and BPF verifier-stat read. Always `Some(_)` for
    /// VMs whose `super::KtstrVm::run_vm` returns normally —
    /// including the host-watchdog timeout path, because
    /// `run_bsp_loop` exits cleanly with `timed_out = true` and
    /// `collect_results` still executes, populating the field.
    /// `None` only when `run_vm` does not complete (a BSP panic
    /// propagated through `?`, or any pre-BSP setup error that
    /// returns an `Err` before `VmRunState` is constructed) and on
    /// the `test_fixture` / skip-sidecar paths that never boot a VM.
    /// Persisted via
    /// [`SidecarResult`](crate::test_support::SidecarResult) so stats
    /// tooling can flag cleanup regressions across runs.
    pub cleanup_duration: Option<Duration>,
    /// The cleanup window's OWN dilation evidence: the join/drain
    /// performer thread's schedstat delta (on-CPU + runnable-wait)
    /// across exactly the [`Self::cleanup_duration`] window. The
    /// cleanup-budget gate judges overruns against THIS — the
    /// per-phase/whole-run witnesses cannot attest the window (the
    /// monitor that feeds them is itself joined inside it, and join
    /// wake-latency is a tail phenomenon a whole-run average
    /// under-attests). `None` when schedstat was unreadable at either
    /// edge; a `0` on-CPU delta (CONFIG_SCHEDSTATS off) is treated as
    /// unattested by consumers.
    pub cleanup_sched_delta: Option<HostVcpuSchedstat>,
    /// Host-side virtio-blk device counters, snapshotted after the
    /// guest has exited. `Some(_)` when the builder attached a disk
    /// via `super::KtstrVmBuilder::disk`; `None` when no disk was
    /// configured and `super::KtstrVm::init_virtio_blk` returned
    /// `None`. The device increments its internal `AtomicU64`
    /// counters from `drain_bracket_impl` (production cfg: on the
    /// dedicated `ktstr-vblk` worker thread; cfg(test): inline on
    /// the test thread); by the time `collect_results` constructs
    /// the [`VmResult`] every vCPU and the worker have joined and
    /// no further mutation can occur. The snapshot is taken at that
    /// point — readers see plain `u64` fields holding the final
    /// cumulative totals; no atomic load is needed on the consumer
    /// side.
    ///
    /// The counter struct exposes nine `AtomicU64` fields, each
    /// bumped from `drain_bracket_impl` (in `src/vmm/virtio_blk/drain.rs`)
    /// via the `VirtioBlkCounters::record_*` helpers (defined in
    /// `src/vmm/virtio_blk/counters.rs`). Per-request
    /// cumulative counters, per-event cumulative counters, and
    /// per-request live gauges are kept distinct per the
    /// counter-taxonomy doc on `VirtioBlkCounters`:
    ///
    ///   - `reads_completed` — count of `VIRTIO_BLK_T_IN` requests
    ///     that returned `S_OK` to the guest. Bumped together with
    ///     `bytes_read` per `VirtioBlkCounters::record_read`.
    ///   - `writes_completed` — count of `VIRTIO_BLK_T_OUT` requests
    ///     that returned `S_OK`. Bumped together with `bytes_written`.
    ///   - `flushes_completed` — count of `VIRTIO_BLK_T_FLUSH`
    ///     requests that returned `S_OK` (real `fdatasync` for
    ///     read-write disks, no-op for `read_only`).
    ///   - `bytes_read` — total bytes returned to the guest for
    ///     completed reads.
    ///   - `bytes_written` — total bytes accepted from the guest for
    ///     completed writes.
    ///   - `throttled_count` — cumulative token-bucket **stall events**
    ///     for the device's lifetime. The chain is rolled back and
    ///     the worker arms a retry timerfd; the guest does not see
    ///     `S_IOERR` for a stall (the request is deferred until the
    ///     bucket refills). This counter is separate from `io_errors`
    ///     so operators can distinguish "throttle bucket drained,
    ///     request deferred" from "real IO problem". Per-event (NOT
    ///     per-request): a single chain that stalls twice produces
    ///     two bumps.
    ///   - `io_errors` — every path that reports `S_IOERR`:
    ///     spec violations, backend `pread`/`pwrite` errors,
    ///     malformed chains, `add_used` failures.
    ///     Stalls do not report `S_IOERR`; see `throttled_count`.
    ///   - `currently_throttled_gauge` — **live gauge**: how many
    ///     requests are RIGHT NOW waiting for throttle tokens.
    ///     Increments when a chain transitions into stalled,
    ///     decrements on retry success or reset. Bounded at 0 or 1
    ///     on this single-queue device. NOT cumulative — answers
    ///     "what's stuck now," distinct from `throttled_count`
    ///     which answers "how many stall events happened over
    ///     time."
    ///   - `invalid_avail_idx_count` — cumulative count of
    ///     `Error::InvalidAvailRingIndex` events observed by
    ///     `drain_bracket_impl` (avail.idx more than `queue.size`
    ///     ahead of `next_avail` — a virtio-v1.2 §2.7.13.3
    ///     avail.idx-distance violation by the guest). Per-event
    ///     counter; the `queue_poisoned` flag short-circuits
    ///     subsequent kicks so one guest fault produces exactly
    ///     one bump regardless of how many notifications follow
    ///     before reset.
    ///
    /// Counters are cumulative for the device's lifetime. A guest
    /// driver re-bind (writing `STATUS=0` to `VIRTIO_MMIO_STATUS`
    /// triggers `VirtioBlk::reset`) does NOT zero them — the
    /// device's internal `AtomicU64` storage persists across reset
    /// cycles, and the post-exit snapshot captures the final
    /// cumulative totals spanning the entire device lifetime, not
    /// just a post-reset fragment.
    ///
    /// Reading example:
    ///
    /// ```ignore
    /// let r: VmResult = builder.run()?;
    /// let c = r.virtio_blk_counters.expect("disk attached");
    /// assert!(c.reads_completed > 0);
    /// ```
    ///
    /// `#[allow(dead_code)]`: the field is part of the public API
    /// surface and read by user test code outside `lib.rs`, but the
    /// lib build doesn't see any in-tree readers because no lib code
    /// path calls `.virtio_blk_counters` on a `VmResult`. The in-tree
    /// readers live in unit tests.
    #[allow(dead_code)]
    pub virtio_blk_counters: Option<VirtioBlkCountersSnapshot>,
    /// Host-side virtio-net device counters, snapshotted after the
    /// guest has exited — the cross-NIC AGGREGATE (field-wise
    /// saturating sum via `VirtioNetCountersSnapshot::aggregate`) over
    /// every attached NIC. `Some(_)` when the builder attached one or
    /// more networks via `super::KtstrVmBuilder::network`; `None` when
    /// no network was configured (the aggregate over an empty NIC set).
    /// Each NIC's device increments its own `AtomicU64` counters on the
    /// vCPU thread inside `process_tx_loopback`; by the time
    /// `collect_results` constructs the [`VmResult`] every vCPU has
    /// joined and no further mutation can occur. The per-NIC snapshots
    /// are summed at that point — readers see plain `u64` fields holding
    /// the final cumulative totals across all NICs; no atomic load is
    /// needed on the consumer side. Per-NIC IRQ-delivery observability
    /// comes from the per-CPU / per-IRQ metrics axis, not these
    /// device-internal loopback counters.
    ///
    /// The counter struct exposes sixteen `AtomicU64` fields, each
    /// bumped across the TX-drain path rooted at `process_tx_loopback`
    /// (several are bumped inside the `pop_and_capture_tx` /
    /// `try_loopback_to_rx` helpers it calls; the three `ctrl_*`
    /// counters are bumped on the control-vq path, not the TX-drain
    /// path):
    ///
    ///   - `tx_packets` — count of TX chains whose L2 frame was
    ///     captured (`frame_len = Some`) AND whose TX `add_used`
    ///     succeeded. Over-size-dropped and malformed chains are still
    ///     marked used (so the guest doesn't hang) but do NOT advance
    ///     `tx_packets`; a chain whose `add_used` fails advances
    ///     `tx_add_used_failures` instead. So `tx_packets` advances per
    ///     successfully-captured-and-published chain, not per parsed
    ///     chain.
    ///   - `tx_bytes` — bytes of L2 frame data captured from
    ///     successfully parsed TX chains (excludes the 12-byte
    ///     virtio header).
    ///   - `rx_packets` / `rx_bytes` — count + bytes of RX chains
    ///     successfully written and marked used. `rx_packets` and
    ///     `tx_packets` gate INDEPENDENTLY per chain: `rx_packets`
    ///     bumps when the loopback delivers — recorded BEFORE the TX
    ///     `add_used` — while `tx_packets` bumps only when that later
    ///     TX `add_used` succeeds. So the identity
    ///     `rx_packets == tx_packets - tx_dropped_no_rx_buffer
    ///     - tx_dropped_rx_poisoned` holds ONLY when both the RX-side
    ///     failure counters AND `tx_add_used_failures` are zero:
    ///     RX-side failures (`rx_add_used_failures`, `rx_chain_invalid`,
    ///     `rx_write_failed`) make `rx_packets` fall SHORT of
    ///     `tx_packets - drops`, while a `tx_add_used_failures` on a
    ///     chain whose RX already delivered makes `rx_packets` EXCEED
    ///       it. Asymmetric counts surface queue-state breakage on
    ///       either side.
    ///   - `tx_dropped_no_rx_buffer` — successfully-captured TX
    ///     frames the device could not deliver because the RX queue
    ///     was empty (transient back-pressure event).
    ///   - `tx_dropped_rx_poisoned` — successfully-captured TX frames
    ///     dropped because the RX queue was poisoned by a prior guest
    ///     avail-ring violation (wedged until a virtio reset), as
    ///     opposed to the transient empty-queue back-pressure counted
    ///     by `tx_dropped_no_rx_buffer`.
    ///   - `tx_chain_invalid` / `rx_chain_invalid` — chains rejected
    ///     for malformed shape (short header, wrong direction,
    ///     attacker-controlled descriptor address overflow).
    ///   - `tx_oversize_dropped` — TX chains dropped (not truncated)
    ///     because the captured post-header frame data exceeded the
    ///     maximum L2 frame size the guest's `max_mtu` permits.
    ///   - `rx_write_failed` — RX chain whose shape was valid but
    ///     whose guest-memory `write_slice` (header or frame) hit
    ///     an unmapped GPA. Distinct from `rx_chain_invalid` so an
    ///     operator can tell "guest violated the RX descriptor-
    ///     direction rule" from "guest posted a buffer at an
    ///     unmapped GPA"; the two are mutually exclusive per chain.
    ///   - `tx_add_used_failures` / `rx_add_used_failures` —
    ///     `add_used` failures, indicating the queue's used-ring
    ///     address itself is unmapped or otherwise inaccessible.
    ///     Distinct from the `*_chain_invalid` / `rx_write_failed`
    ///     counters so an operator can tell "guest sent malformed
    ///     frame" / "guest's posted buffer GPA was unmapped" from
    ///     "queue itself is broken".
    ///   - `invalid_avail_idx_count` — cumulative count of
    ///     `Error::InvalidAvailRingIndex` events observed by
    ///     `process_tx_loopback` (avail.idx more than `queue.size`
    ///     ahead of `next_avail` — virtio-v1.2 §2.7.13.3 violation
    ///     by the guest). Per-event counter; the per-queue
    ///     `queue_poisoned` flag short-circuits subsequent kicks
    ///     so one guest fault produces exactly one bump regardless
    ///     of how many notifications follow before reset.
    ///   - `ctrl_mq_set` — cumulative count of successful control-vq
    ///     `VIRTIO_NET_CTRL_MQ` / `VQ_PAIRS_SET` commands (the device
    ///     updated `curr_queue_pairs` and wrote `VIRTIO_NET_OK`).
    ///     Per-event counter.
    ///   - `ctrl_chain_invalid` — cumulative count of control-vq
    ///     chains the device could not satisfy: malformed shape (no
    ///     device-writable status descriptor, too few readable command
    ///     bytes), an unknown `(class, cmd)`, or a `virtqueue_pairs`
    ///     outside `[1, queue_pairs]`. Per-event hostile/buggy-guest
    ///     counter; the control-vq analog of `tx_chain_invalid`.
    ///   - `ctrl_add_used_failures` — cumulative count of control-vq
    ///     status-write or `add_used` failures (the status byte's GPA
    ///     or the used-ring address is unmapped). Queue-state breakage
    ///     distinct from `ctrl_chain_invalid`; the control-vq analog
    ///     of `tx_add_used_failures`.
    ///
    /// Counters are cumulative for the device's lifetime — a guest
    /// driver re-bind (writing `STATUS=0`) does NOT zero them.
    #[allow(dead_code)]
    pub virtio_net_counters: Option<VirtioNetCountersSnapshot>,
    /// Snapshot bridge populated by the freeze coordinator over the
    /// run's lifetime. Every `Op::CaptureSnapshot` and `Op::WatchSnapshot`
    /// fire stores a `FailureDumpReport` keyed by its tag.
    ///
    /// `#[ktstr_test]` test bodies whose scenario fires snapshot
    /// ops in the guest assert on the captured reports through a
    /// `post_vm = NAME` attribute. The named callback runs on the
    /// HOST after `vm.run()` returns (see
    /// [`crate::test_support::KtstrTestEntry::post_vm`]) and
    /// receives `&VmResult`; it calls
    /// [`crate::scenario::snapshot::SnapshotBridge::drain`] on
    /// this field to take ownership of the stored reports and
    /// walks them — typically through
    /// [`crate::scenario::snapshot::Snapshot::new`] for typed
    /// access to map values, per-CPU entries, and scalar
    /// variables. Out-of-tree consumers can drain the bridge the
    /// same way: `VmResult` is in `ktstr::prelude`.
    ///
    /// Always present after a successful `run_vm`; `None`-equivalent
    /// (empty) when the VM crashed before any snapshot fired.
    ///
    /// **Drained exactly once, via [`Self::captures_series`]**: the
    /// bridge yields each capture once, so the host-side consumers
    /// share a single drain. The first call to
    /// [`Self::captures_series`] — whether from a `post_vm` callback
    /// (which runs FIRST, via [`Self::periodic_series`] /
    /// [`Self::phase_buckets`]) or from the framework's
    /// `evaluate_vm_result` (which runs AFTER `post_vm` to build
    /// [`crate::assert::ScenarioStats::phases`]) — drains this bridge
    /// and memoizes the resulting
    /// [`crate::scenario::sample::SampleSeries`] on the
    /// `periodic_series_cache` field; every later call reads the
    /// cache. That is why a per-phase `post_vm` and the framework's
    /// `result.stats.phases` build no longer starve each other (a
    /// `post_vm` that drained the bridge first used to leave
    /// `stats.phases` empty). Integration tests under `tests/` that
    /// bypass the series accessors and call
    /// `result.snapshot_bridge.drain*()` directly (e.g.
    /// `tests/stats_bridge_e2e.rs`, `tests/temporal_assertions_e2e.rs`)
    /// are unaffected: the cache is only populated by
    /// [`Self::captures_series`], which those tests never call, so the
    /// raw destructive drain still returns the full capture set.
    pub snapshot_bridge: crate::scenario::snapshot::SnapshotBridge,
    /// Live scheduler-stats client. `Some(_)` when the run wired the
    /// virtio-console port-2 stats bridge (the in-tree path always
    /// does so, but tests that construct a [`VmResult`] manually via
    /// `Self::test_fixture` leave this `None`). Test code that
    /// asserts on scheduler-reported metrics calls
    /// `super::SchedStatsClient::stats` /
    /// `super::SchedStatsClient::stats_meta` on this handle WHILE
    /// the guest is alive — calling after VM exit will time out
    /// because the relay thread has already exited. Cloneable;
    /// multiple test threads may share the same client.
    #[allow(dead_code)]
    pub stats_client: Option<super::SchedStatsClient>,
    /// Number of periodic snapshot boundaries the freeze
    /// coordinator actually fired during this run. Includes both
    /// successful captures and rendezvous-timeout placeholders.
    /// Tests can assert `result.periodic_fired >= some_lower_bound`
    /// to guard periodic-capture coverage; mismatches against
    /// [`Self::periodic_target`] flag missing samples (early VM
    /// exit, kill-flag stop, abandoned-after-timeouts).
    pub periodic_fired: u32,
    /// Periodic captures that landed REAL BPF state — the
    /// placeholder-excluded subset of [`Self::periodic_fired`]
    /// (`periodic_fired` counts rendezvous-timeout placeholders as
    /// fired). Snapshotted from
    /// [`crate::scenario::snapshot::SnapshotBridge::periodic_real_count`]
    /// at result-collection time so it is stable regardless of any
    /// later test-side drain of the bridge. `periodic_real <
    /// periodic_fired` means the gap is placeholder-only fills (the
    /// boundary fired but the dump was degraded); the failure-output
    /// periodic-samples section surfaces this so a "100% fired" run
    /// whose captures were all placeholders does not read as full
    /// coverage.
    pub periodic_real: u32,
    /// Configured `num_snapshots` count for the entry that drove
    /// this run (mirrors the `KtstrTestEntry::num_snapshots` field
    /// the entry was registered with). `0` when periodic capture
    /// was disabled. Pairs with [`Self::periodic_fired`] so a
    /// test can compute coverage without re-reading the entry
    /// table.
    pub periodic_target: u32,
    /// Runtime virt-KASLR offset (kernel-image slide). Captured
    /// from the freeze coordinator's `kern_virt_kaslr` Arc snapshot
    /// at run-end via `load(Acquire).saturating_sub(1)`. `0` means
    /// either (a) KASLR was off — test ran with
    /// `#[ktstr_test(kaslr = false)]` or
    /// `Scheduler::kargs(&["nokaslr"])`, OR (b) the derivation
    /// chain (MSR_LSTAR readback in `vmm::x86_64::msr_kaslr` +
    /// KERN_ADDRS `_text` path in `crate::vmm::freeze_coord::dispatch`) never
    /// published a non-zero value (early-boot crash, kallsyms masked
    /// by kptr_restrict, FRED-enabled kernel). E2E test consumers
    /// distinguish (a) from (b) by reading the test entry's `kaslr`
    /// attribute alongside this field — see
    /// [`Self::kaslr_enabled`] for the binary-question companion.
    pub kern_kaslr_offset: u64,
    /// Name of the `#[ktstr_test]` fn whose execution produced this
    /// result. Stamped from
    /// `crate::test_support::entry::KtstrTestEntry::name` (a
    /// `&'static str` the macro emits at compile time) in
    /// `test_support::eval::run_ktstr_test_inner_impl` immediately
    /// after `super::KtstrVm::run` returns and BEFORE the
    /// `post_vm` callback dispatch runs.
    ///
    /// `Some(_)` for every result that flowed through the real
    /// `run_ktstr_test_inner_impl` path. `None` for the
    /// `freeze_coord::collect_results` direct-synthesis path
    /// (entry-agnostic boundary; entry is not in scope there) and
    /// for `#[cfg(test)]`-only `Self::test_fixture` callers. The
    /// path-derivation methods `wprof_pb_path` and
    /// `repro_wprof_pb_path` (require the `wprof` feature) bail with a loud diagnostic
    /// on `None` so any `VmResult` reaching the derivation path
    /// without going through the eval-layer stamping site
    /// surfaces the misuse rather than producing a garbage-named
    /// path.
    ///
    /// Test authors writing `post_vm` callbacks should derive
    /// per-test sidecar paths via the helper methods rather than
    /// hardcoding a `wprof_pb_path("<literal>")` string against
    /// the fn name — a future rename of the test fn drifts the
    /// hardcoded literal silently, where the method-form derives
    /// from this field automatically.
    pub entry_name: Option<&'static str>,
    /// The run's variant hash (see `variant_hash_from_parts`),
    /// stamped alongside [`Self::entry_name`] after `vm.run()` returns.
    /// The post-VM `failure_dump_path` / `wprof_pb_path` derivations
    /// embed it as the `-{16-hex}` filename suffix so a gauntlet test's
    /// per-preset dumps don't clobber and each matches its sidecar's
    /// variant hash. `0` on a synthesized/fixture result (which has
    /// `entry_name = None` and thus bails before reading this).
    pub variant_hash: u64,
    /// Host-side vCPU scheduling dilation for this run — RAW schedstat
    /// totals summed over the vCPU host threads (see
    /// `HostVcpuSchedstat`). Populated by `run_vm` teardown from each
    /// vCPU thread's `/proc/self/task/<tid>/schedstat` while the threads
    /// are still alive; `None` on hosts without `CONFIG_SCHEDSTATS`, on
    /// synthesized/fixture results, and whenever no vCPU thread was
    /// sampled. Consumers call `HostVcpuSchedstat::dilation` for the
    /// derived `D` ratio. Purely observational — never affects the
    /// verdict or exit code.
    pub host_vcpu_schedstat: Option<HostVcpuSchedstat>,
    /// Per-phase host-contention witness for the "weather witness" latency
    /// model (see [`ContentionWitness`]): the Body-phase dilation `D` plus
    /// the Body-phase peak-window CPU-pressure series that bounds `W(L)`.
    /// Lifecycle events anchor the series and the monitor adds constant-cost
    /// cgroup-PSI samples on its existing wakes. `None` when neither PSI nor
    /// vCPU schedstat supplied evidence. Consumed by a LATER seam pass —
    /// `latency_verdict` — to make latency threshold verdicts tri-state under
    /// contention; purely observational until then.
    pub contention_witness: Option<ContentionWitness>,
    /// Memoized single drain of [`Self::snapshot_bridge`].
    ///
    /// The snapshot bridge yields each capture exactly once, but two
    /// host-side consumers need the captures: a `post_vm` callback
    /// (which runs first) and the framework's `evaluate_vm_result`
    /// (which builds [`crate::assert::ScenarioStats::phases`]). Before
    /// this cache existed, whichever drained first starved the other —
    /// a per-phase `post_vm` calling [`Self::periodic_series`] left
    /// `evaluate_vm_result` with an empty bridge, silently emptying
    /// `result.stats.phases` and the failure-message timeline.
    ///
    /// [`Self::captures_series`] performs the one destructive bridge
    /// drain on first call and stores the resulting full
    /// [`crate::scenario::sample::SampleSeries`] here; every later call
    /// — and [`Self::periodic_series`] / [`Self::phase_buckets`] /
    /// `evaluate_vm_result` — reads the cached series instead of
    /// re-draining. Lazily populated so a consumer that only touches
    /// the raw bridge via `snapshot_bridge.drain*()` (e.g. integration
    /// tests under `tests/`) is unaffected: the cache is never
    /// initialised on that path.
    ///
    /// `pub(crate)` (not `pub`): in-crate constructors
    /// (`freeze_coord::collect_results`, test fixtures) set it to an
    /// empty `OnceLock`, but out-of-tree code cannot struct-literal a
    /// `VmResult` — it flows from `run_vm` — so the cache stays an
    /// implementation detail behind [`Self::captures_series`].
    pub(crate) periodic_series_cache: std::sync::OnceLock<crate::scenario::sample::SampleSeries>,
}

impl VmResult {
    /// Whether the guest kernel booted with KASLR enabled (= a
    /// non-zero virt-KASLR offset published into the freeze
    /// coordinator's `kern_virt_kaslr` Arc). Returns `true` when
    /// [`Self::kern_kaslr_offset`] is non-zero. The inverse case
    /// (returns `false`) covers two scenarios: (a) the test
    /// explicitly opted out via `#[ktstr_test(kaslr = false)]` or
    /// `Scheduler::kargs(&["nokaslr"])`, OR (b) the derivation
    /// chain failed to publish a non-zero value (early-boot crash,
    /// kallsyms masked, kernel built without `CONFIG_RANDOMIZE_BASE`).
    /// E2E test consumers distinguish (a) from (b) by reading the
    /// test entry's `kaslr` attribute alongside this method.
    ///
    /// Companion to [`Self::kern_kaslr_offset`] — use this when the
    /// caller cares about the binary "did KASLR happen?" question
    /// and use the raw field for exact-offset assertions
    /// (alignment, entropy-range, etc.).
    pub fn kaslr_enabled(&self) -> bool {
        self.kern_kaslr_offset != 0
    }

    /// The full capture series for this run — every snapshot the
    /// freeze coordinator stored on [`Self::snapshot_bridge`]
    /// (periodic boundaries AND on-demand `Op::CaptureSnapshot` /
    /// watchpoint-fire captures), in the order the bridge surfaced.
    ///
    /// Performs the bridge's single destructive drain on the first
    /// call and memoizes the resulting
    /// [`crate::scenario::sample::SampleSeries`] on the
    /// `periodic_series_cache` field; every later call — and
    /// [`Self::periodic_series`] / [`Self::phase_buckets`] and the
    /// framework's `evaluate_vm_result` — returns the cached series
    /// without re-draining. This is what lets a `post_vm` callback and
    /// the framework's [`crate::assert::ScenarioStats::phases`] build
    /// share one drain instead of starving each other (the bridge
    /// yields each capture exactly once).
    ///
    /// Takes `&self`: the cache uses interior mutability
    /// ([`std::sync::OnceLock`]) so this composes with the
    /// `#[ktstr_test(post_vm = ...)]` callback signature
    /// (`fn(&VmResult) -> Result<()>`).
    ///
    /// A consumer that calls `snapshot_bridge.drain*()` directly
    /// (e.g. integration tests under `tests/`) bypasses this cache. If
    /// such a raw drain runs BEFORE the first `captures_series()` call
    /// the cache memoizes an empty series, so prefer this accessor over
    /// a raw drain on any path that also reaches `evaluate_vm_result`.
    pub fn captures_series(&self) -> &crate::scenario::sample::SampleSeries {
        self.periodic_series_cache.get_or_init(|| {
            crate::scenario::sample::SampleSeries::from_drained_typed(
                self.snapshot_bridge.drain_ordered_with_stats(),
                self.monitor.clone(),
            )
        })
    }

    /// The periodic-capture-only view of this run's series: the
    /// `"periodic_"`-tagged subset of [`Self::captures_series`] — the
    /// projection the temporal-assertion / per-phase patterns expect
    /// (on-demand `Op::CaptureSnapshot` and watchpoint-fire captures
    /// are filtered out as off-cadence outliers, see
    /// [`crate::scenario::sample::SampleSeries::periodic_only`]).
    ///
    /// Reads the shared [`Self::captures_series`] cache (the single
    /// bridge drain) and returns an owned, periodic-only clone.
    /// Idempotent: calling it twice — or alongside
    /// [`Self::phase_buckets`] / `evaluate_vm_result` — no longer
    /// empties the bridge for the other consumers (the pre-cache
    /// behavior, which silently starved whichever drained second).
    ///
    /// Takes `&self` so it composes with the
    /// `#[ktstr_test(post_vm = ...)]` callback signature.
    pub fn periodic_series(&self) -> crate::scenario::sample::SampleSeries {
        self.captures_series().clone().periodic_only()
    }

    /// The complete per-phase stimulus timeline for `post_vm`
    /// callbacks doing per-phase metric assertions: one
    /// [`crate::timeline::StimulusEvent`] per guest `Stimulus` frame
    /// (the step-start boundaries, via
    /// [`crate::timeline::StimulusEvent::from_wire`]) PLUS the
    /// synthesized terminal scenario-end boundary (from the
    /// `ScenarioEnd` frame's final cumulative count, via
    /// [`crate::timeline::StimulusEvent::terminal`]).
    ///
    /// Fold THIS through
    /// [`crate::assert::build_phase_buckets_with_stimulus`] — it is the
    /// SAME timeline the framework's own `evaluate_vm_result` builds,
    /// so the LAST step gets an `iteration_rate` (the terminal supplies
    /// its right boundary). A hand-rolled map over only the guest
    /// `Stimulus` frames would omit the terminal and silently drop the
    /// final step's rate.
    ///
    /// Non-destructive: reads the already-drained `guest_messages` TLV
    /// log (unlike the bridge-cache accessors [`Self::captures_series`]
    /// / [`Self::periodic_series`] / [`Self::phase_buckets`], which
    /// perform the single destructive snapshot-bridge drain), so it may
    /// be called alongside the bridge drain. CRC-bad / malformed frames
    /// are skipped.
    pub fn stimulus_timeline(&self) -> Vec<crate::timeline::StimulusEvent> {
        let mut out = Vec::new();
        let Some(bulk) = &self.guest_messages else {
            return out;
        };
        for entry in &bulk.entries {
            if !entry.crc_ok {
                continue;
            }
            match wire::MsgType::from_wire(entry.msg_type) {
                Some(wire::MsgType::Stimulus) => {
                    if let Some(ev) = wire::StimulusEvent::from_payload(&entry.payload) {
                        out.push(crate::timeline::StimulusEvent::from_wire(&ev));
                    }
                }
                Some(wire::MsgType::StepEnd) => {
                    // Per-step end-of-hold frame (reuses the StimulusPayload
                    // body). Paired with its StepStart for step-local
                    // iteration_rate in build_phase_buckets_with_stimulus.
                    if let Some(ev) = wire::StimulusEvent::from_payload(&entry.payload) {
                        out.push(crate::timeline::StimulusEvent::from_step_end(&ev));
                    }
                }
                Some(wire::MsgType::ScenarioEnd) => {
                    if let Some((elapsed_ms, total_iterations)) =
                        wire::parse_scenario_end(&entry.payload)
                    {
                        out.push(crate::timeline::StimulusEvent::terminal(
                            elapsed_ms,
                            total_iterations,
                        ));
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// Worker-iteration throughput (iterations/sec) for one scenario
    /// [`Phase`](crate::assert::Phase), from the stimulus timeline's
    /// `StepStart[k]` -> `StepEnd[k]` step-local window (the per-event rate
    /// via [`crate::timeline::StimulusEvent::rate_to`]).
    ///
    /// `None` when the phase has no `StepStart` ([`crate::assert::Phase::BASELINE`], or a
    /// step the run never reached), no right boundary (no `StepEnd`, no
    /// later step, and no scenario-end terminal), or the rate is
    /// unmeasurable (zero-length window / counter went backward). A step
    /// whose workers made zero forward progress over a positive hold
    /// returns `Some(0.0)` (measured zero), not `None`.
    ///
    /// Collapse-immune: the stimulus timeline carries per-step boundaries
    /// independent of the periodic-capture pipeline, so this works even for
    /// `--cell-parent-cgroup` schedulers where the capture-derived
    /// [`PhaseBucket`](crate::assert::PhaseBucket) path can collapse.
    pub fn step_throughput(&self, phase: crate::assert::Phase) -> Option<f64> {
        Self::step_throughput_in(&self.stimulus_timeline(), phase)
    }

    /// Ratio `step_throughput(a) / step_throughput(b)` — e.g.
    /// scheduler-vs-EEVDF throughput when phase `b` runs on the detached
    /// kernel default ([`crate::scenario::ops::Op::detach_scheduler`]).
    /// Walks the stimulus timeline once. `None` when either phase has no
    /// measurable throughput; a `Some(0.0)` denominator yields `inf` so a
    /// collapsed/stalled phase `b` surfaces rather than vanishing to `None`.
    pub fn throughput_ratio(
        &self,
        a: crate::assert::Phase,
        b: crate::assert::Phase,
    ) -> Option<f64> {
        let timeline = self.stimulus_timeline();
        let ta = Self::step_throughput_in(&timeline, a)?;
        let tb = Self::step_throughput_in(&timeline, b)?;
        Some(ta / tb)
    }

    /// Shared core for [`Self::step_throughput`] / [`Self::throughput_ratio`]
    /// over an already-built timeline: pair the phase's `StepStart` with its
    /// own `StepEnd` (step-local), falling back to the next step's
    /// `StepStart` then the scenario-end terminal for the last step on
    /// legacy/sched-died data that lacks a `StepEnd`. Mirrors the boundary
    /// selection in [`crate::timeline::Timeline::build`].
    fn step_throughput_in(
        timeline: &[crate::timeline::StimulusEvent],
        phase: crate::assert::Phase,
    ) -> Option<f64> {
        let start = timeline
            .iter()
            .find(|e| !e.is_terminal && !e.is_step_end && e.phase() == Some(phase))?;
        let end = timeline
            .iter()
            .find(|e| e.is_step_end && e.phase() == Some(phase))
            .or_else(|| {
                timeline
                    .iter()
                    .filter(|e| {
                        !e.is_terminal && !e.is_step_end && e.phase().is_some_and(|p| p > phase)
                    })
                    .min_by_key(|e| e.elapsed_ms)
            })
            .or_else(|| timeline.iter().find(|e| e.is_terminal))?;
        start.rate_to(end)
    }

    /// The guest-side [`crate::assert::AssertResult`] decoded from this
    /// run's `MSG_TYPE_TEST_RESULT` frame — the verdict the in-VM scenario
    /// body produced, carrying the per-phase per-cgroup raw telemetry
    /// carriers in `stats.phases[].per_cgroup`
    /// ([`crate::assert::PhaseCgroupStats`]: `total_migrations`,
    /// `run_delays_ns`, `cpus_used`, …) and the per-cgroup reductions in
    /// `stats.cgroups`.
    ///
    /// Non-destructive: reads the already-drained `guest_messages` TLV log
    /// (the last crc-ok `MSG_TYPE_TEST_RESULT` entry), so it composes with
    /// the snapshot-bridge accessors and may be called repeatedly. Shares
    /// the exact decode the eval layer uses
    /// (`crate::test_support::parse_assert_result_from_drain`).
    ///
    /// `Err` when there is no guest verdict to decode — a host-only run, a
    /// crash before the body emitted its result, or a drain with no
    /// `MSG_TYPE_TEST_RESULT` frame.
    ///
    /// This is the GUEST view: the run-level distribution / `ext_metrics`
    /// re-pools (`worst_*` wake-latency / run-delay aggregates, pooled
    /// `iterations_per_cpu_sec`) are applied HOST-side in `evaluate_vm_result`
    /// AFTER the body returns and are NOT on this value — only
    /// `stats.phases[].per_cgroup` and the per-cgroup `stats.cgroups`
    /// reductions are guest-authoritative. For the per-phase per-cgroup view
    /// aligned to the host capture windows use [`Self::phase_buckets`] (which
    /// folds these carriers in); for one cgroup in one phase use
    /// [`Self::phase_cgroup`].
    pub fn guest_assert_result(&self) -> anyhow::Result<crate::assert::AssertResult> {
        crate::test_support::parse_assert_result_from_drain(self.guest_messages.as_ref())
    }

    /// The framework-computed per-phase metric buckets for this run — the
    /// SAME [`crate::assert::PhaseBucket`] vec the framework folds onto
    /// [`crate::assert::ScenarioStats::phases`] in `evaluate_vm_result`,
    /// INCLUDING the per-phase per-cgroup carriers in
    /// [`crate::assert::PhaseBucket::per_cgroup`].
    ///
    /// This is the answer to "my `post_vm` callback wants the per-phase
    /// metrics the framework already built." It folds the same two sources
    /// `evaluate_vm_result` does:
    /// 1. the host-rebuilt buckets from [`Self::periodic_series`] (the
    ///    periodic-only projection of the shared single drain — on-demand /
    ///    watchpoint captures are off-cadence outliers excluded from
    ///    per-phase folds) through
    ///    [`crate::assert::build_phase_buckets_with_stimulus`] using
    ///    [`Self::stimulus_timeline`] for the step windows (window + metric
    ///    folds; `per_cgroup` empty by construction); and
    /// 2. the guest per-cgroup carriers from [`Self::guest_assert_result`]
    ///    (`stats.phases[].per_cgroup`), folded in by
    ///    `crate::assert::fold_guest_per_cgroup_into_host_buckets` keyed by
    ///    `step_index` (the host window + metrics win; each carrier
    ///    contributes only its `per_cgroup`, an unmatched carrier surfacing
    ///    as a `(0,0)`-window orphan bucket).
    ///
    /// Production builds `stats.phases` from these same two sources (same
    /// periodic-only series, same stimulus timeline, same guest carriers) and
    /// the eval layer's later `populate_run_*` passes write only
    /// `stats.ext_metrics` / `stats.cgroups`, never `phases[].per_cgroup`, so
    /// this returns content IDENTICAL to `result.stats.phases` — the
    /// no-carrier case pinned by
    /// `phase_buckets_equals_stats_phases_and_post_vm_read_does_not_starve`
    /// and the with-per-cgroup-carrier case by
    /// `phase_buckets_equals_stats_phases_with_guest_per_cgroup_carriers`.
    ///
    /// Both sources are non-destructive on the snapshot bridge:
    /// [`Self::periodic_series`] reads the memoized single drain
    /// ([`Self::captures_series`]) and [`Self::guest_assert_result`] reads the
    /// already-drained `guest_messages`. A run with no guest verdict
    /// ([`Self::guest_assert_result`] `Err` — host-only / early crash) folds
    /// no carriers and returns the host-rebuilt buckets alone (the prior
    /// behavior).
    pub fn phase_buckets(&self) -> Vec<crate::assert::PhaseBucket> {
        let mut buckets = self.phase_buckets_pre_derive();
        // Derive the per-phase scalars into the now-final (post-fold) buckets so
        // a per-phase A/B claim reads them via phase_metric / phase_cgroup_metric:
        // the non-schbench carrier scalars (every cgroup) into each pc.metrics,
        // and the schbench scalars into pc.metrics + the pooled bucket.metrics.
        // A no-op only when no phase carries a per-cgroup carrier; must run
        // post-fold (the merge skips is_derived keys, so an earlier derive would
        // be dropped).
        crate::assert::derive_phase_metrics(&mut buckets);
        buckets
    }

    /// The pre-`derive_phase_metrics` phase fold: host buckets from
    /// [`Self::periodic_series`] + [`Self::stimulus_timeline`] with the guest
    /// per-cgroup carriers folded in, BEFORE the per-phase scalar derivation
    /// [`Self::phase_buckets`] applies. This is the exact phase state the eval
    /// layer feeds to the run-level ext-metrics population
    /// (`populate_run_ext_metrics_from_phases` runs on the pre-derive phases;
    /// `derive_phase_metrics` runs AFTER, inside `evaluate_vm_result`), so
    /// [`Self::run_metric`] reuses it to reproduce that sequence by construction
    /// (eval-faithful): post-derive phases yield the same run-level map today (the
    /// run-level phase fold skips `is_derived` keys, and the pooled scalars
    /// `derive_phase_metrics` adds are all `PerPhase`), but the pre-derive fold
    /// avoids depending on that skip, so a pooled key ever registered as
    /// non-derived cannot diverge `run_metric` from the eval map. Non-destructive
    /// on the snapshot bridge, like [`Self::phase_buckets`].
    fn phase_buckets_pre_derive(&self) -> Vec<crate::assert::PhaseBucket> {
        let host = crate::assert::build_phase_buckets_with_stimulus(
            &self.periodic_series(),
            &self.stimulus_timeline(),
        );
        match self.guest_assert_result() {
            Ok(guest) => {
                crate::assert::fold_guest_per_cgroup_into_host_buckets(host, guest.stats.phases)
            }
            Err(_) => host,
        }
    }

    /// One phase's per-cgroup telemetry for `cgroup` — the per-phase analog
    /// of reading `result.stats.cgroups` for a single phase. Reads
    /// [`Self::phase_buckets`] (which folds the guest per-cgroup carriers
    /// against the host capture windows) and returns the
    /// [`crate::assert::PhaseCgroupStats`] keyed by `cgroup` in `phase`.
    ///
    /// `None` when `phase` has no bucket (no capture landed in it AND no
    /// stimulus `StepStart` synthesized one) or the phase carries no carrier
    /// for `cgroup` (the cgroup ran no workers in that phase, or the run had
    /// no step-local cgroups). Owned (cloned from the folded bucket) so it
    /// composes with the `&VmResult` `post_vm` callback signature.
    pub fn phase_cgroup(
        &self,
        phase: crate::assert::Phase,
        cgroup: &str,
    ) -> Option<crate::assert::PhaseCgroupStats> {
        self.phase_buckets()
            .into_iter()
            .find(|b| b.step_index == phase.as_u16())
            .and_then(|b| b.per_cgroup.get(cgroup).cloned())
    }

    /// One framework-computed per-phase metric for `phase` — the
    /// metric-name analog of [`Self::step_throughput`] /
    /// [`Self::throughput_ratio`]. Resolves `metric` (any `impl Into<MetricId>` —
    /// a typed `BuiltinMetric`, typo-proof, or a dynamic scheduler-runtime
    /// string) from the folded
    /// [`Self::phase_buckets`] bucket for `phase`, checking two stores:
    /// 1. [`crate::assert::PhaseBucket::metrics`] (via
    ///    [`crate::assert::PhaseBucket::get`]) — the host-folded
    ///    per-sample / monitor / stimulus metrics: per-phase CPU time
    ///    (`system_time_ns`, `user_time_ns`), scheduling quality
    ///    (`avg_imbalance_ratio`, `avg_dsq_depth`, ...), and
    ///    `iteration_rate`.
    /// 2. failing that, the cross-cgroup phase sum of a per-cgroup Counter
    ///    ([`crate::assert::PhaseBucket::cgroup_counter_total`]) for the
    ///    keys whose value lives ONLY in the per-cgroup carriers —
    ///    `"total_migrations"`, `"total_iterations"`, and
    ///    `"total_cpu_time_ns"`. These are
    ///    registered `Counter`s with no per-sample source
    ///    (`crate::stats::MetricDef::read_sample` returns `None`), so
    ///    they never reach `metrics`; without this fallback
    ///    `phase_metric(phase, "total_migrations")` returned a silent
    ///    `None` even though the value was present per-cgroup.
    ///
    /// The wake-latency / run-delay distributions (`MetricKind::Distribution`)
    /// have no per-phase `metrics` value and no single per-cgroup Counter
    /// (they re-pool run-level from the per-cgroup raw sample vectors), so
    /// they are not readable here — read them per-cgroup via
    /// [`Self::phase_cgroup`].
    ///
    /// Reads the SAME buckets [`Self::phase_buckets`] folds onto
    /// `result.stats.phases`, so a `post_vm` callback can compare any
    /// metric across two phases (e.g. scheduler-vs-detached-EEVDF
    /// `total_migrations` or `system_time_ns`) without re-deriving the
    /// buckets — the general form of the scheduler-vs-EEVDF throughput
    /// compare.
    ///
    /// `None` when `phase` has no bucket — no capture landed in it AND no
    /// stimulus `StepStart` synthesized one (e.g. `Phase::BASELINE` when
    /// the settle window fired no captures, or a `phase` past the last
    /// step) — or the bucket carries no reading for `metric` in either
    /// store. Sentinel-free, distinct from a real `Some(0.0)`: a
    /// per-cgroup counter returns `Some(0.0)` when carriers exist but
    /// counted zero, `None` only when the phase has no carrier at all. A
    /// started-but-uncaptured step (a `StepStart` with zero captures) DOES
    /// produce a synthesized bucket, so `phase_metric` returns its
    /// stimulus-derived `iteration_rate` rather than `None`.
    ///
    /// ```ignore
    /// // Typed (typo-proof) is the primary form; a dynamic scheduler-runtime
    /// // string is the escape hatch through the SAME call.
    /// let p99 = result.phase_metric(Phase::step(0), BuiltinMetric::WakeupP99LatencyUs);
    /// let custom = result.phase_metric(Phase::step(0), "scx_layered_layer0_util");
    /// ```
    pub fn phase_metric(
        &self,
        phase: crate::assert::Phase,
        metric: impl Into<crate::stats::MetricId>,
    ) -> Option<f64> {
        let metric = metric.into();
        self.phase_buckets()
            .into_iter()
            .find(|b| b.step_index == phase.as_u16())
            .and_then(|b| {
                b.get(metric.as_str())
                    .or_else(|| b.cgroup_counter_total(metric.as_str()))
            })
    }

    /// One run-level extensible ("ext") metric by name — the whole-run analog of
    /// [`Self::phase_metric`], for a `post_vm` callback asserting a run-level
    /// aggregate (e.g. `avg_irq_util`, `max_cpu_hardirqs`,
    /// `worst_p99_wake_latency_us`, `iterations_per_cpu_sec`). SELF-COMPUTES the
    /// run-level `ext_metrics` map exactly as the framework's `evaluate_vm_result`
    /// does — a [`VmResult`](Self) carries no stored run-level stats (`post_vm`
    /// runs BEFORE the host populates them) — by replaying the shared
    /// [`crate::assert::populate_run_ext_all`] sequence over
    /// [`Self::periodic_series`], the pre-derive phase fold
    /// (`phase_buckets_pre_derive`), and the guest per-cgroup `stats.cgroups`.
    /// The result is byte-identical to the run-level `ext_metrics` the sidecar
    /// records for this run.
    ///
    /// Resolves the ext-sourced family that
    /// [`crate::assert::ScenarioStats::run_metric`] (the post-merge host
    /// accessor) resolves — the `read_sample`-wired registry metrics, the
    /// phase-only ext metrics (`avg_imbalance_ratio`, `iteration_rate`,
    /// `system_time_ns`, `user_time_ns`, the IRQ counters/rates, the per-CPU
    /// spatial maxes `max_cpu_hardirqs` / `max_cpu_softirq_net_rx` and their
    /// concentrations), the pooled `iterations_per_cpu_sec`, and the run-level
    /// `Distribution` / `WorstLowest` / `WakeLatencyTailRatio` / `WorstCrossNodeRatio`
    /// re-pools — and for
    /// those keys the two accessors return identical values (this one
    /// self-computes pre-merge, the other reads the stored post-merge map).
    ///
    /// ADDITIONALLY resolves the 5 ext-only run-level MONITOR metrics from
    /// [`Self::monitor`]'s summary: `avg_nr_running`, `avg_irq_util`,
    /// `max_avg_irq_util`, `psi_irq_full_avg10`, `total_irq_pressure_us` (folded
    /// via `MonitorSummary::fold_run_level_ext`, shared with the sidecar row).
    /// These DIVERGE from [`crate::assert::ScenarioStats::run_metric`], which has
    /// no `MonitorReport` to fold and returns `None` for them — the one place the
    /// two accessors differ.
    ///
    /// RESOLVED here None-aware (via the delegated `ScenarioStats::run_metric`
    /// typed dispatch): the cross-cgroup metrics `worst_spread`,
    /// `worst_migration_ratio`, `worst_gap_ms`, `total_migrations`,
    /// `total_iterations`, `worst_page_locality`,
    /// `worst_cross_node_migration_ratio`. The dispatch re-derives each from the
    /// carriers (the per-cgroup `stats.cgroups` + the per_cgroup-folded
    /// `stats.phases` this method builds) — `None` when no carrier measured it,
    /// `Some(0.0)` for a measured zero. The 5 non-NUMA metrics are 0.0-sentinel
    /// typed struct fields (re-derived because the field cannot carry the
    /// measured-vs-unmeasured distinction); the two NUMA roll-ups
    /// (`worst_page_locality`, `worst_cross_node_migration_ratio`) have no struct
    /// field and re-pool purely from the per-phase NUMA carriers.
    ///
    /// NOT resolved here:
    /// - the typed-backed monitor run-level metrics (`max_imbalance_ratio`,
    ///   `max_dsq_depth`, `stuck_count`, `total_fallback`, `total_keep_last`) —
    ///   these have typed `GauntletRow` fields (not ext-only); read them
    ///   per-phase via [`Self::phase_metric`].
    ///
    /// Sentinel-free, matching [`Self::phase_metric`]: `None` means the metric is
    /// absent from this run (no populator produced it, or a name not in the map);
    /// `Some(0.0)` is a real measured zero. Check [`crate::stats::MetricId::def`]
    /// on a dynamic key to distinguish an unregistered key from genuinely-absent
    /// data (built-in ids always resolve).
    ///
    /// A host-only run (no guest verdict — [`Self::guest_assert_result`] `Err`)
    /// resolves the SampleSeries + phase families but no per-cgroup pooled /
    /// distribution keys (no per-cgroup data exists), the same absence
    /// [`crate::assert::ScenarioStats::run_metric`] documents.
    ///
    /// Non-destructive: reads the memoized snapshot-bridge drain
    /// ([`Self::periodic_series`] / [`Self::phase_buckets`]) and the
    /// already-drained `guest_messages`, so it composes with [`Self::phase_metric`]
    /// in one `post_vm` callback.
    ///
    /// ```ignore
    /// let irq = result.run_metric(BuiltinMetric::AvgIrqUtil);
    /// let custom = result.run_metric("scx_layered_layer0_util");
    /// ```
    pub fn run_metric(&self, metric: impl Into<crate::stats::MetricId>) -> Option<f64> {
        let metric = metric.into();
        let mut stats = crate::assert::ScenarioStats {
            phases: self.phase_buckets_pre_derive(),
            cgroups: self
                .guest_assert_result()
                .map(|g| g.stats.cgroups)
                .unwrap_or_default(),
            ..Default::default()
        };
        crate::assert::populate_run_ext_all(&mut stats, &self.periodic_series());
        // Fold the run-level ext-only monitor metrics (avg_nr_running + the PELT
        // IRQ load pair + the PSI-irq pair) from the stored MonitorReport
        // summary. populate_run_ext_all can't produce them — they're
        // MonitorSummary-sourced, not phase/series-folded — but VmResult holds
        // self.monitor, so (unlike ScenarioStats::run_metric, which has no
        // monitor) this accessor CAN resolve them. Shared with
        // group::sidecar_to_row via fold_run_level_ext.
        if let Some(report) = self.monitor.as_ref() {
            report.summary.fold_run_level_ext(&mut stats.ext_metrics);
        }
        // Delegate to ScenarioStats::run_metric: it resolves the typed
        // 0.0-sentinel cross-cgroup fields None-aware from the carriers
        // (stats.cgroups + the per_cgroup-folded stats.phases, both populated
        // above) ahead of the ext lookup — so the typed fields resolve here too,
        // matching ScenarioStats::run_metric. The monitor metrics folded into
        // stats.ext_metrics above resolve via that method's ext fallback.
        stats.run_metric(metric)
    }

    /// Polarity-aware "is the `candidate` phase better than the `baseline` phase
    /// on `metric`?" comparator — the per-phase A/B primitive for "assert
    /// scheduler X beats EEVDF across phases" (e.g. an scx Step vs the EEVDF
    /// Step after `Op::DetachScheduler`). Resolves both per-phase values via
    /// [`Self::phase_metric`] and the metric's polarity via
    /// `crate::stats::metric_def`, then returns a
    /// [`crate::assert::temporal::BetterThanPhase`] builder whose terminal
    /// (`better_than` / `by_at_least`) records the outcome into `verdict`.
    /// "Better" is oriented from the registry polarity, so the SAME call works
    /// for a LowerBetter latency (`BuiltinMetric::WakeupP99LatencyUs`) and a
    /// HigherBetter throughput (`BuiltinMetric::SchbenchLoopCount`) with no
    /// caller-specified direction.
    ///
    /// A post_vm callback collapses the verdict to its `anyhow::Result` via
    /// [`crate::assert::Verdict::into_anyhow_or_log`], which bails on a Fail OR
    /// an Inconclusive — so a missing metric (a phase with no schbench carrier),
    /// an undirected metric, or a zero-baseline fractional-margin comparison
    /// does NOT silently pass.
    pub fn better_across_phases<'v>(
        &self,
        verdict: &'v mut crate::assert::Verdict,
        baseline: crate::assert::Phase,
        candidate: crate::assert::Phase,
        metric: impl Into<crate::stats::MetricId>,
    ) -> crate::assert::temporal::BetterThanPhase<'v> {
        let metric = metric.into();
        crate::assert::temporal::BetterThanPhase::new(
            metric.as_str().to_string(),
            verdict,
            baseline,
            candidate,
            self.phase_metric(baseline, metric.clone()),
            self.phase_metric(candidate, metric.clone()),
            metric.def().map(|m| m.polarity),
            None, // pooled producer — no per-cgroup scope label
        )
    }

    /// One per-phase, PER-CGROUP derived metric — the per-cgroup analog of
    /// [`Self::phase_metric`], answering "metric M of cgroup C in phase P" as
    /// readily as the phase aggregate (the N-cgroups-to-N-queryable-sets goal).
    /// Resolves `metric` (any `impl Into<MetricId>` — a typed `BuiltinMetric` or
    /// a dynamic scheduler-runtime string) from this
    /// cgroup's per-phase carrier via [`crate::assert::PhaseCgroupStats::get`] (its
    /// derived `metrics` map), falling back to
    /// [`crate::assert::PhaseCgroupStats::cgroup_counter`] for the per-cgroup
    /// Counters `total_migrations`/`total_iterations`/`total_cpu_time_ns` (carrier
    /// fields, not derived) — symmetric with [`Self::phase_metric`]'s
    /// `cgroup_counter_total` fallback.
    /// `None` when `phase` has no bucket, the bucket has no carrier for `cgroup`, or
    /// the carrier carried no finite value for the metric — sentinel-free, distinct
    /// from a real `Some(0.0)`.
    pub fn phase_cgroup_metric(
        &self,
        phase: crate::assert::Phase,
        cgroup: &str,
        metric: impl Into<crate::stats::MetricId>,
    ) -> Option<f64> {
        let metric = metric.into();
        self.phase_cgroup(phase, cgroup).and_then(|pc| {
            pc.get(metric.as_str())
                .or_else(|| pc.cgroup_counter(metric.as_str()))
        })
    }

    /// Per-cgroup analog of [`Self::better_across_phases`]: "is `metric` of
    /// `cgroup` better in the `candidate` phase than the `baseline` phase?",
    /// oriented from the registry polarity. Reuses the SAME
    /// [`crate::assert::temporal::BetterThanPhase`] comparator/verdict machinery;
    /// only value resolution is re-scoped from the pooled aggregate to the named
    /// cgroup, so a missing carrier / undirected metric / zero-baseline margin
    /// collapses to Inconclusive (not a silent pass), exactly as the pooled form.
    pub fn better_across_phases_cgroup<'v>(
        &self,
        verdict: &'v mut crate::assert::Verdict,
        baseline: crate::assert::Phase,
        candidate: crate::assert::Phase,
        cgroup: &str,
        metric: impl Into<crate::stats::MetricId>,
    ) -> crate::assert::temporal::BetterThanPhase<'v> {
        let metric = metric.into();
        crate::assert::temporal::BetterThanPhase::new(
            metric.as_str().to_string(),
            verdict,
            baseline,
            candidate,
            self.phase_cgroup_metric(baseline, cgroup, metric.clone()),
            self.phase_cgroup_metric(candidate, cgroup, metric.clone()),
            metric.def().map(|m| m.polarity),
            Some(cgroup.to_string()), // per-cgroup scope label for the diagnostics
        )
    }

    /// Minimal "nothing happened" fixture for tests that exercise
    /// code consuming a [`VmResult`] without actually booting a VM
    /// (the sidecar-write tests in `src/test_support/sidecar.rs`
    /// are the primary users). Every field carries the empty /
    /// default / `None` value that `run_vm` would produce for a
    /// VM that launched, exited cleanly with exit code 0, and
    /// produced no telemetry. Tests that need a specific field
    /// override it with a struct-update expression:
    ///
    /// ```ignore
    /// let result = VmResult { success: false, ..VmResult::test_fixture() };
    /// ```
    ///
    /// Gated on `#[cfg(test)]` so the symbol does not appear in
    /// release builds — production `VmResult` values flow from
    /// `run_vm` and never from this fixture. See
    /// `sidecar_vm_result_is_test_fixture_boilerplate` in
    /// `test_support/sidecar.rs` for the motivating deduplication
    /// (7 identical literal constructions collapsed to a single
    /// call).
    #[cfg(test)]
    pub fn test_fixture() -> Self {
        Self {
            success: true,
            vcpus: 1,
            cpu_budget: 1,
            resolve_source: None,
            expect_auto_repro_satisfied: false,
            exit_code: 0,
            duration: Duration::from_secs(1),
            timed_out: false,
            watchdog_kill_reason: None,
            final_guest_phase: GuestLifecyclePhase::Boot,
            final_progress_epoch: 0,
            bpf_map_writes_delivered: None,
            periodic_prereqs_ready: None,
            periodic_window_end: None,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            guest_messages: None,
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
            cleanup_sched_delta: None,
            virtio_blk_counters: None,
            virtio_net_counters: None,
            snapshot_bridge: empty_snapshot_bridge_for_tests(),
            stats_client: None,
            periodic_fired: 0,
            periodic_real: 0,
            periodic_target: 0,
            kern_kaslr_offset: 0,
            entry_name: None,
            variant_hash: 0,
            host_vcpu_schedstat: None,
            contention_witness: None,
            periodic_series_cache: std::sync::OnceLock::new(),
        }
    }

    /// Per-test sidecar path for the `.wprof.pb` artifact:
    /// `{sidecar_dir()}/{entry_name}-{variant_hash:016x}.wprof.pb`.
    #[cfg(feature = "wprof")]
    pub fn wprof_pb_path(&self) -> anyhow::Result<std::path::PathBuf> {
        let name = self.entry_name.ok_or_else(|| {
            anyhow::anyhow!(
                "VmResult.entry_name is None — wprof_pb_path() requires the \
                 macro-stamped entry name set by run_ktstr_test_inner_impl \
                 after vm.run() returns. A `None` here means the VmResult \
                 was constructed via the freeze_coord::collect_results \
                 direct-synthesis path and the eval-layer stamping was \
                 bypassed; route the result through run_ktstr_test_inner_impl \
                 OR assign entry_name = Some(\"<test-fn-name>\") manually \
                 before calling .wprof_pb_path()."
            )
        })?;
        Ok(crate::test_support::sidecar_dir()
            .join(format!("{name}-{:016x}.wprof.pb", self.variant_hash)))
    }

    /// Per-test sidecar path for the `.repro.wprof.pb` artifact.
    #[cfg(feature = "wprof")]
    pub fn repro_wprof_pb_path(&self) -> anyhow::Result<std::path::PathBuf> {
        let name = self.entry_name.ok_or_else(|| {
            anyhow::anyhow!(
                "VmResult.entry_name is None — repro_wprof_pb_path() \
                 requires the macro-stamped entry name set by \
                 run_ktstr_test_inner_impl after vm.run() returns. A `None` \
                 here means the VmResult was constructed via the \
                 freeze_coord::collect_results direct-synthesis path and \
                 the eval-layer stamping was bypassed; route the result \
                 through run_ktstr_test_inner_impl OR assign entry_name \
                 manually before calling."
            )
        })?;
        Ok(crate::test_support::sidecar_dir()
            .join(format!("{name}-{:016x}.repro.wprof.pb", self.variant_hash)))
    }

    /// Per-test failure-dump sidecar path. Derives
    /// `{sidecar_dir()}/{entry_name}-{variant_hash:016x}.failure-dump.json`
    /// from the macro-stamped [`Self::entry_name`].
    ///
    /// # Sibling to
    /// [`crate::scenario::Ctx::failure_dump_path`]
    ///
    /// The pre-VM body context carries its own copy of the
    /// macro-stamped entry name (stamped at Ctx construction by
    /// the dispatch path) and computes the same path string. A
    /// test body invocation `ctx.failure_dump_path()` and a
    /// post-VM `result.failure_dump_path()` resolve to identical
    /// paths because both stamp from the same
    /// `entry.name: &'static str` source — proc-macro emission
    /// at the `#[ktstr_test]` site. This pair gives post_vm
    /// callbacks a symmetric path-derivation surface to the
    /// pre-VM body, so a future post_vm hook that wants to
    /// inspect or clean up the failure dump uses the same method
    /// shape the body uses to look at it.
    ///
    /// # Errors
    ///
    /// Returns `Err` when [`Self::entry_name`] is `None`.
    pub fn failure_dump_path(&self) -> anyhow::Result<std::path::PathBuf> {
        let name = self.entry_name.ok_or_else(|| {
            anyhow::anyhow!(
                "VmResult.entry_name is None — failure_dump_path() \
                 requires the macro-stamped entry name set by \
                 run_ktstr_test_inner_impl after vm.run() returns. \
                 A `None` here means the VmResult was constructed via \
                 the freeze_coord::collect_results direct-synthesis \
                 path and the eval-layer stamping was bypassed; route \
                 the result through run_ktstr_test_inner_impl OR \
                 assign entry_name manually before calling."
            )
        })?;
        Ok(crate::test_support::sidecar_dir().join(format!(
            "{name}-{:016x}.failure-dump.json",
            self.variant_hash
        )))
    }

    /// Concatenated guest `/dev/kmsg` content forwarded via
    /// `crate::send_kmsg`, or empty when no frames arrived (the scenario
    /// did not forward, or the VM exited before the forward completed).
    /// Lets a post_vm callback read the guest kernel log even at the
    /// default `loglevel=0`, where kernel printks never reach the COM1
    /// console (and thus not `stderr`). Encapsulates the bulk-port
    /// `ShmEntry` + `MsgType::Dmesg` filter inside the crate.
    pub fn guest_kmsg(&self) -> String {
        let Some(drain) = self.guest_messages.as_ref() else {
            return String::new();
        };
        drain
            .entries
            .iter()
            .filter(|e| {
                e.crc_ok
                    && !e.payload.is_empty()
                    && matches!(
                        crate::vmm::wire::MsgType::from_wire(e.msg_type),
                        Some(crate::vmm::wire::MsgType::Dmesg)
                    )
            })
            .map(|e| String::from_utf8_lossy(&e.payload).into_owned())
            .collect::<Vec<String>>()
            .join("\n")
    }

    /// The scheduler's captured log — its stderr, including any libbpf /
    /// BPF-verifier output — extracted from the bulk-port
    /// `MSG_TYPE_SCHED_LOG` frames in [`Self::guest_messages`]. Empty when
    /// no scheduler log was captured (no scheduler was spawned, or the
    /// framed markers never arrived).
    ///
    /// Mirrors the extraction [`crate::verifier::collect_verifier_output`]
    /// performs: concatenate the `SchedLog` chunks, slice the span from the
    /// first `SCHED_OUTPUT_START` to the last `SCHED_OUTPUT_END` (spanning
    /// any intervening markers when the guest staged more than one log),
    /// and fall back to `output` when the bulk-port drain carried no
    /// `SchedLog` frames (a kernel without the bulk port). Lets a post_vm
    /// callback assert on what the scheduler logged — e.g. a libbpf
    /// verifier-reject — without reaching into the crate-internal wire
    /// types.
    pub fn scheduler_log(&self) -> String {
        let merged = crate::verifier::concat_sched_log_chunks(self.guest_messages.as_ref());
        let source = if merged.is_empty() {
            &self.output
        } else {
            &merged
        };
        crate::verifier::parse_sched_output(source)
            .unwrap_or("")
            .to_string()
    }

    /// The host watchdog-override readback (`expected_jiffies` =
    /// host-written, `observed_jiffies` = read back from guest memory).
    /// `None` when the scheduler never attached (no readback recorded).
    /// A post_vm callback asserts the two are equal to prove the override
    /// landed; the readback is taken eagerly by the monitor (in-DRAM,
    /// microseconds), so it is immune to the watchdog-kworker starvation
    /// that inflates the kernel-measured stall duration. Encapsulates the
    /// `MonitorReport` access (`WatchdogObservation` is re-exported at the
    /// crate root for the return type).
    pub fn watchdog_observation(&self) -> Option<crate::monitor::WatchdogObservation> {
        self.monitor.as_ref().and_then(|m| m.watchdog_observation)
    }

    /// Assert the primary-VM `.wprof.pb` landed and is shape-valid.
    /// Returns `Ok(())` immediately when `self.success` is false.
    #[cfg(feature = "wprof")]
    pub fn assert_wprof_pb_landed(&self) -> anyhow::Result<()> {
        if !self.success {
            return Ok(());
        }
        // Pre-check `entry_name` with a callable-specific diagnostic
        // BEFORE delegating to `self.wprof_pb_path()` (which would
        // bail with the wprof_pb_path-perspective message). A
        // fixture-constructed VmResult hitting this method should
        // see a diagnostic naming `assert_wprof_pb_landed` so the
        // caller's mental model lines up with the error text.
        anyhow::ensure!(
            self.entry_name.is_some(),
            "VmResult::assert_wprof_pb_landed requires entry_name set by \
             run_ktstr_test_inner_impl after vm.run() returns. This \
             VmResult was constructed manually (freeze_coord direct \
             synthesis path or a test fixture); either route the result \
             through run_ktstr_test_inner_impl OR call \
             crate::test_support::wprof::assert_wprof_pb_shape with a \
             manually-computed path.",
        );
        let path = self.wprof_pb_path()?;
        crate::test_support::wprof::assert_wprof_pb_shape(&path)
    }
}

/// Build an empty `SnapshotBridge` whose capture callback always
/// returns `None`. Used by `VmResult::test_fixture` and the legacy
/// `VmResult` literal constructions in unit tests so they still
/// compile after the snapshot_bridge field landed. Production
/// `run_vm` constructs its own bridge whose callback is
/// intentionally unused — the freeze coordinator stores reports
/// directly via `bridge.store(name, report)`.
#[cfg(test)]
pub(crate) fn empty_snapshot_bridge_for_tests() -> crate::scenario::snapshot::SnapshotBridge {
    let cb: crate::scenario::snapshot::CaptureCallback = std::sync::Arc::new(|_| None);
    crate::scenario::snapshot::SnapshotBridge::new(cb)
}

/// Host-side vCPU scheduling-dilation sample, summed over the run's
/// vCPU host threads.
///
/// Dilation `D = 1 + Σrun_delay / Σon_cpu`, where the two sums are
/// taken over each vCPU host thread's `/proc/self/task/<tid>/schedstat`
/// (field 1 = time on-CPU ns, field 2 = time runnable-but-not-running
/// ns). `D == 1.0` means the vCPU threads always got the CPU the instant
/// they were runnable; `D > 1.0` quantifies host-side scheduling delay
/// (e.g. `1.03x` = threads waited 3% on top of their run time).
///
/// Idle-immune: a halted vCPU is blocked (not runnable), so it accrues
/// no `run_delay` — halt time never inflates `D`. Measured HOST-side
/// deliberately: the guest run-queue's own `run_delay` would measure the
/// scheduler under test, which is the thing being evaluated; the host
/// thread's schedstat measures the CPU the vCPU thread was starved of by
/// the HOST, orthogonal to the guest scheduler.
///
/// RAW totals are stored (not the ratio) so future consumers can
/// recompute or re-aggregate; [`Self::dilation`] derives the ratio.
///
/// `total_on_cpu_ns == 0` yields `dilation() == None` — this happens
/// BOTH when no vCPU thread ran at all AND on a host built without
/// `CONFIG_SCHEDSTATS` (every schedstat line reads `"0 0 0"`). `None` is
/// thus graceful and distinguishable from a genuine `D == 1.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HostVcpuSchedstat {
    /// Σ schedstat field 1 (time the vCPU threads spent ON-CPU), ns.
    pub total_on_cpu_ns: u64,
    /// Σ schedstat field 2 (time the vCPU threads spent runnable but
    /// not running — host-side scheduling delay), ns.
    pub total_run_delay_ns: u64,
    /// Number of vCPU host threads that contributed to the sums (a TID
    /// still 0 — an AP that never stamped its TID — and an unreadable
    /// schedstat are both skipped).
    pub sampled_vcpus: u32,
}

impl HostVcpuSchedstat {
    /// Host dilation `D = 1 + Σrun_delay / Σon_cpu`, or `None` when no
    /// on-CPU time was sampled (no vCPU ran, or the host lacks
    /// `CONFIG_SCHEDSTATS` and every line read `"0 0 0"`). `None` is
    /// deliberately distinguishable from a true `D == 1.0` (which needs
    /// nonzero on-CPU time with zero run-delay).
    pub fn dilation(&self) -> Option<f64> {
        (self.total_on_cpu_ns > 0)
            .then(|| 1.0 + self.total_run_delay_ns as f64 / self.total_on_cpu_ns as f64)
    }

    /// Field-wise saturating delta `self - anchor` over the two schedstat
    /// sums — the on-CPU / run-delay accrued BETWEEN two cumulative
    /// snapshots of the same vCPU threads. `sampled_vcpus` carries `self`'s
    /// count (the reading the delta closes at). Saturating so a schedstat
    /// that momentarily regressed (a TID that vanished then reappeared, a
    /// torn read) contributes 0 rather than wrapping — conservative: a
    /// per-phase `D` can only be UNDER-stated by such a clamp, never
    /// fabricated. The result's [`Self::dilation`] is the phase-local `D`.
    pub fn delta_from(&self, anchor: &HostVcpuSchedstat) -> HostVcpuSchedstat {
        HostVcpuSchedstat {
            total_on_cpu_ns: self.total_on_cpu_ns.saturating_sub(anchor.total_on_cpu_ns),
            total_run_delay_ns: self
                .total_run_delay_ns
                .saturating_sub(anchor.total_run_delay_ns),
            sampled_vcpus: self.sampled_vcpus,
        }
    }
}

/// Number of `LifecycleStage` variants (Boot, Attach,
/// Dispatch, Body, Teardown) — the length of the per-phase witness array.
/// Kept in lock-step with that enum; [`PerPhaseSchedstat::for_stage`] indexes
/// by `stage as usize`, so a new stage variant needs this bumped.
pub const NUM_LIFECYCLE_STAGES: usize = 5;

/// Array index of the BODY stage (the measurement phase) — pinned to the
/// enum discriminant so it can never drift from
/// `LifecycleStage::Body`.
pub const BODY_STAGE_INDEX: usize = crate::monitor::LifecycleStage::Body as usize;

/// Per-lifecycle-phase host-dilation witness: one [`HostVcpuSchedstat`] DELTA
/// per stage, each covering exactly its own stage's span (the schedstat
/// accrued between lifecycle-event boundary snapshots). Indexed by
/// `LifecycleStage` `as usize`.
///
/// Distinct from the whole-run [`VmResult::host_vcpu_schedstat`] (which is
/// the verdict-line `D` over the entire run): here each phase gets its OWN
/// `D` over its OWN span, so the Body phase's dilation — the only phase whose
/// contention actually contaminates the workload measurement — is isolated
/// from the Boot / Attach / Teardown INFRA phases. `Boot`/`Attach` `D` is
/// diagnostic; `Body` is the one the latency verdict consumes.
///
/// The dispatch path takes each boundary snapshot immediately after accepting
/// the CRC-valid lifecycle frame. A wide-VM snapshot is a sequential sweep of
/// the vCPU TIDs rather than an atomic kernel operation, so its uncertainty is
/// the duration of that rare sweep, not the monitor's sampling period.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PerPhaseSchedstat {
    /// Per-stage schedstat delta, indexed by `LifecycleStage as usize`
    /// (Boot=0 .. Teardown=4). A stage never entered stays at the
    /// `Default` all-zero value (`dilation() == None`).
    pub phases: [HostVcpuSchedstat; NUM_LIFECYCLE_STAGES],
}

impl PerPhaseSchedstat {
    /// The Body (measurement) phase's schedstat delta — the phase whose
    /// dilation the latency verdict consumes.
    pub fn body(&self) -> &HostVcpuSchedstat {
        &self.phases[BODY_STAGE_INDEX]
    }

    /// The Body phase's host dilation `D`, or `None` when no on-CPU time was
    /// sampled in the Body span (schedstats off, or the phase never ran).
    pub fn body_dilation(&self) -> Option<f64> {
        self.body().dilation()
    }

    /// The schedstat delta for one lifecycle stage (diagnostic access to the
    /// Boot/Attach/Dispatch/Teardown phases). `pub(crate)`: takes the
    /// crate-internal `LifecycleStage`. `allow(dead_code)`:
    /// the diagnostic Boot/Attach `D` readers land with the LATER seam pass;
    /// `body()` is the wired accessor.
    #[allow(dead_code)]
    pub(crate) fn for_stage(&self, stage: crate::monitor::LifecycleStage) -> &HostVcpuSchedstat {
        &self.phases[stage as usize]
    }
}

/// Body-phase peak-window contamination series. New witnesses store deltas of
/// the ktstr runner cgroup's CPU PSI `some total` clock: time during which at
/// least one runnable task in that cgroup was stalled for CPU. This includes a
/// delayed vCPU even when its competitor is outside the cgroup, without
/// charging pressure in unrelated host cgroups. Each entry has its real
/// observation width in [`Self::tick_widths_ns`], allowing the window
/// calculation to remain sound when the monitor is starved and wakes much
/// later than its nominal cadence.
///
/// Legacy/deserialized witnesses may have no width vector; those retain the
/// fixed-grid `tick_ns` interpretation. Bounded storage: observations are
/// nominally 100 ms apart and Body phases run minutes, so a few thousand
/// entries is typical; the series is capped at 36,000 intervals and
/// `saturated` is set if the cap is hit. When
/// `saturated`, the series is a PREFIX of the phase, so `W` computed from it
/// is a LOWER bound — a later verdict pass should treat that as reducing
/// confidence in a `FailConfirmed` rather than trusting a possibly-short `W`.
///
/// COVERAGE SOUNDNESS: new witnesses sample the cumulative PSI counter at
/// accepted lifecycle boundaries as well as monitor wakes, so the first and
/// last samples explicitly close the entire Body span. A starved monitor may
/// produce one coarse interval instead of many fine ones, but charging that
/// interval's entire delta remains an upper bound for every window it can
/// touch. If either boundary sample fails or storage saturates, `complete` is
/// false and the seam cannot use the series to confirm a failure.
///
/// Legacy witnesses have no explicit completeness bit. For them the seam
/// retains the old conservative check that the fixed-grid samples span enough
/// of Body; an empty or clustered series can only demote a result, never
/// produce a false confirmation.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BodyContentionWindow {
    /// Per-observation CPU-contention delta (ns) over the Body phase, in time
    /// order. New witnesses use runner-cgroup CPU PSI `some total`; legacy
    /// witnesses contain summed vCPU schedstat run-delay.
    pub tick_deltas: Vec<u64>,
    /// Real wall width (ns) of each corresponding delta interval. Empty on
    /// legacy witnesses. When present, length matches `tick_deltas` and the
    /// peak-window calculation uses these widths instead of a nominal grid.
    #[serde(default)]
    pub tick_widths_ns: Vec<u64>,
    /// The nominal monitor tick period (ns) the deltas were sampled at
    /// (100 ms in production). Used only by legacy witnesses without real
    /// interval widths.
    pub tick_ns: u64,
    /// True when [`Self::tick_deltas`] hit its cap and later Body ticks were
    /// dropped — the series is then a prefix and `W` from it is a lower bound.
    pub saturated: bool,
    /// True when lifecycle-event samples anchored BOTH ends of Body (the
    /// start is conservatively pulled back to the preceding lifecycle
    /// boundary to cover virtio dispatch lag), making the cumulative interval
    /// series complete even if it contains only one coarse interval. False on
    /// legacy sampled witnesses, whose coverage is inferred from their
    /// first/last monitor ticks.
    #[serde(default)]
    pub complete: bool,
    /// Complete summed vCPU schedstat run-delay (ns) over the SAME
    /// conservative wall span as this series. When
    /// [`Self::schedstat_cap_complete`] is true, `W(L)` is capped by this
    /// value: a request cannot absorb more task-specific scheduling delay in
    /// one window than all vCPUs accumulated over the enclosing span. This
    /// removes unrelated runner-cgroup PSI without under-bounding the target
    /// VM's delay.
    #[serde(default)]
    pub schedstat_cap_ns: u64,
    /// True only when both cap snapshots sampled the exact same complete set
    /// of vCPU TIDs. False for legacy witnesses and partial/read-raced
    /// lifecycle snapshots; in that case the cap is ignored.
    #[serde(default)]
    pub schedstat_cap_complete: bool,
    /// The host-side wall span (ns) covered by the Body witness. The start is
    /// conservatively pinned to the accepted lifecycle boundary preceding
    /// Body, covering any time the guest's Body frame waited in virtio
    /// dispatch; the end is the first accepted transition after Body. It may
    /// therefore include adjacent-stage time, which makes `W` larger and can
    /// only bias the verdict toward indeterminate. `0` when Body was never
    /// observed.
    pub body_wall_ns: u64,
    /// The wall span (ns) the series actually covered. For event-anchored
    /// complete witnesses this equals `body_wall_ns`; for legacy witnesses it
    /// is the elapsed between the first and last Body monitor tick.
    pub body_covered_ns: u64,
}

/// The event-anchored host-contention witness: per-phase vCPU schedstat
/// dilation plus the Body-phase CPU-pressure interval series. The monitor
/// contributes constant-cost samples on its existing timer wakes; lifecycle
/// dispatch anchors the stage boundaries. Carried on
/// [`VmResult::contention_witness`] so a later seam pass can call
/// `latency_verdict` to turn a latency threshold into a tri-state,
/// contention-aware verdict.
///
/// `None` on [`VmResult`] when neither vCPU schedstat nor CPU PSI supplied any
/// evidence. A present but incomplete Body series is retained for diagnostics
/// but cannot confirm a latency failure.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ContentionWitness {
    /// Per-lifecycle-phase host-dilation deltas (Boot..Teardown).
    pub per_phase: PerPhaseSchedstat,
    /// Body-phase contention intervals for the `W(L)` bound.
    pub body_window: BodyContentionWindow,
}

impl ContentionWitness {
    /// The Body-phase host dilation `D` (annotation for the latency verdict);
    /// `None` when no Body on-CPU time was sampled.
    pub fn body_dilation(&self) -> Option<f64> {
        self.per_phase.body_dilation()
    }
}

/// Per-vCPU KVM stats read after VM exit. Each map holds cumulative
/// counter values from the VM's lifetime.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KvmStatsTotals {
    /// Per-vCPU stat maps. Index is vCPU id.
    pub per_vcpu: Vec<HashMap<String, u64>>,
}

/// KVM stat names surfaced in sidecar output for scheduler testing.
///
/// Covers VM exit rate, halt-polling behavior, preemption notifications,
/// signal-driven exits, and hypercall counts; all fields scheduler
/// authors typically correlate with scx decisions.
///
/// Per-arch availability: `halt_exits`, `preemption_reported`, and
/// `hypercalls` are published by KVM only on x86. On aarch64 the
/// kernel does not expose these stats via `KVM_GET_STATS_FD`; they
/// are absent from the per-vCPU map and read as `0` from
/// [`KvmStatsTotals::sum`] / [`KvmStatsTotals::avg`]. The remaining
/// names (`exits`, `halt_successful_poll`, `halt_attempted_poll`,
/// `halt_wait_ns`, `signal_exits`) are published on both arches.
#[allow(dead_code)]
pub const KVM_INTERESTING_STATS: &[&str] = &[
    "exits",
    "halt_exits",
    "halt_successful_poll",
    "halt_attempted_poll",
    "halt_wait_ns",
    "preemption_reported",
    "signal_exits",
    "hypercalls",
];

impl KvmStatsTotals {
    /// Sum a stat across all vCPUs. Returns 0 BOTH when no vCPU published
    /// the stat and when every vCPU measured zero — use [`Self::try_sum`]
    /// to distinguish "unpublished" from "measured zero".
    pub fn sum(&self, name: &str) -> u64 {
        self.try_sum(name).unwrap_or(0)
    }

    /// Average a stat across all vCPUs (returns 0 if no vCPUs). Same
    /// absent-vs-zero ambiguity as [`Self::sum`]; see [`Self::try_avg`].
    pub fn avg(&self, name: &str) -> u64 {
        self.try_avg(name).unwrap_or(0)
    }

    /// Sum a stat across all vCPUs, or `None` when NO per-vCPU map
    /// published it. Distinguishes an unpublished stat (`None`) from a
    /// genuinely-measured zero (`Some(0)`) — the plain [`Self::sum`]
    /// collapses both to `0`, so a test reading a counter that the kernel
    /// never emitted on this arch cannot tell it apart from a real zero.
    pub fn try_sum(&self, name: &str) -> Option<u64> {
        let mut acc: Option<u64> = None;
        for m in &self.per_vcpu {
            if let Some(&v) = m.get(name) {
                acc = Some(acc.unwrap_or(0) + v);
            }
        }
        acc
    }

    /// Average a stat across all vCPUs, or `None` when there are no vCPUs
    /// or NO per-vCPU map published the stat. The absent-aware counterpart
    /// of [`Self::avg`].
    pub fn try_avg(&self, name: &str) -> Option<u64> {
        let n = self.per_vcpu.len() as u64;
        if n == 0 {
            return None;
        }
        self.try_sum(name).map(|s| s / n)
    }
}

/// State returned by [`super::KtstrVm::run_vm`] after the BSP exits.
/// Passed to [`super::KtstrVm::collect_results`] to produce
/// [`VmResult`].
pub(crate) struct VmRunState {
    pub(crate) exit_code: i32,
    pub(crate) timed_out: bool,
    /// Raw kill-reason byte loaded from the watchdog thread's latch
    /// after its join (`freeze_coord::KillReasonTag` layout; 0 =
    /// unset). Decoded into [`VmResult::watchdog_kill_reason`] by
    /// `collect_results`, which owns the tag type.
    pub(crate) watchdog_kill_reason_raw: u8,
    /// Raw final lifecycle-stage byte from the progress ledger
    /// (`monitor::LifecycleStage` layout). Decoded into
    /// [`VmResult::final_guest_phase`] by `collect_results`.
    pub(crate) final_guest_phase_raw: u8,
    /// Final ledger milestone count → [`VmResult::final_progress_epoch`].
    pub(crate) final_progress_epoch: u64,
    /// Raw BPF-map-write injection delivery state (0 = none
    /// configured, 1 = pending/never delivered, 2 = delivered) —
    /// decoded into [`VmResult::bpf_map_writes_delivered`].
    pub(crate) bpf_map_write_delivery_raw: u8,
    /// Cleanup-window-open schedstat snapshot of the run_vm caller
    /// thread (the join/drain performer) — closed by `collect_results`
    /// into [`VmResult::cleanup_sched_delta`].
    pub(crate) cleanup_sched_t0: Option<HostVcpuSchedstat>,
    /// Run-relative ns when the periodic prereqs (kaslr + accessors)
    /// became ready (0 = never) → [`VmResult::periodic_prereqs_ready`].
    pub(crate) periodic_prereqs_ready_ns_raw: u64,
    /// Run-relative ns of the periodic capture-window end (0 = the
    /// window never resolved) → [`VmResult::periodic_window_end`].
    pub(crate) periodic_window_end_ns_raw: u64,
    /// Event-anchored per-phase dilation + Body contention intervals,
    /// finalized while vCPU `/proc` entries are still alive.
    pub(crate) contention_witness: Option<ContentionWitness>,
    pub(crate) ap_threads: Vec<VcpuThread>,
    pub(crate) monitor_handle: Option<JoinHandle<monitor::reader::MonitorLoopResult>>,
    pub(crate) bpf_write_handle: Option<JoinHandle<()>>,
    /// Freeze coordinator handle, always `None` in the
    /// production path: [`super::KtstrVm::run_vm`] joins the
    /// coordinator BEFORE the BSP `VcpuFd` falls out of scope so the
    /// coordinator's captured BSP `ImmediateExitHandle` cannot
    /// outlive the kvm_run mmap (UAF prevention). The optional shape
    /// is preserved so the field stays trivially constructible in
    /// any future test-only or alternative-orchestration path that
    /// might not perform the early join.
    pub(crate) freeze_coordinator: Option<JoinHandle<()>>,
    pub(crate) com1: Arc<PiMutex<console::Serial>>,
    pub(crate) com2: Arc<PiMutex<console::Serial>>,
    pub(crate) kill: Arc<AtomicBool>,
    /// Wake fd paired with `kill`. Setters that flip `kill`
    /// (`collect_results`, vCPU shutdown classifier, panic hook)
    /// also write to this EventFd so any consumer blocked in
    /// `epoll_wait` (notably the freeze coordinator and the
    /// monitor sampler) wakes within microseconds of the flip
    /// rather than waiting up to one full poll interval. The
    /// AtomicBool above remains the source of truth — the EventFd
    /// is purely a wake signal. EFD_NONBLOCK so a saturated
    /// counter never stalls the writer.
    pub(crate) kill_evt: Arc<vmm_sys_util::eventfd::EventFd>,
    /// Broadcast freeze flag for the failure-dump coordinator. When the
    /// coordinator receives a guest-side error-exit signal it sets this
    /// to true, kicks every vCPU, waits for all `parked` flags to flip
    /// true, and then reads guest BPF map state. Released to false to
    /// resume normal execution. Lives alongside `kill` so the same Arc
    /// pattern (broadcast + per-vCPU ACK) covers both shutdown and
    /// freeze rendezvous.
    pub(crate) freeze: Arc<AtomicBool>,
    /// Hardware-watchpoint arming state Arc, forwarded so
    /// [`super::KtstrVm::collect_results`] can invalidate the
    /// `kind_host_ptr` and `request_kva` slots after every vCPU
    /// thread joins but BEFORE `vm` drops.
    ///
    /// Without the invalidation, the slots' published values
    /// continue to address (a) a host pointer into `vm.guest_mem`'s
    /// mapping that becomes unmapped when `vm` drops and (b) a
    /// guest KVA whose translation goes through the same mapping.
    /// The freeze coordinator joins before `vm` drops in
    /// `run_vm`, and AP threads join inside `collect_results` —
    /// but defense-in-depth says we zero the slots once every
    /// reader is gone so any future restructuring (a stray Arc
    /// clone surviving past teardown, a follow-up that adds a
    /// new reader path) cannot trip a use-after-free.
    ///
    /// Declared before `vm` so the implicit drop order on
    /// `VmRunState` teardown drops `watchpoint` first: any Arc
    /// clone outliving the struct can no longer dereference its
    /// `kind_host_ptr` after `vm.guest_mem` has unmapped, even if
    /// a future caller forgets the explicit pre-drop
    /// invalidation in `collect_results`.
    pub(crate) watchpoint: Arc<WatchpointArm>,
    pub(crate) vm: kvm::KtstrKvm,
    /// Captured immediately after the BSP exits its run loop. Subtracted
    /// from `Instant::now()` in [`super::KtstrVm::collect_results`]
    /// right before the [`VmResult`] is returned to populate
    /// [`VmResult::cleanup_duration`]. Records the wall-clock cost of
    /// every host-side teardown step that runs after the guest has
    /// stopped advancing, in execution order: the watchdog-thread join
    /// in [`super::KtstrVm::run_vm`], then the AP-thread joins, the
    /// monitor-thread join, the BPF-map-writer join, the SHM-ring
    /// drain, the post-exit exit-code/crash-message extraction, and
    /// finally the BPF verifier-stat read inside
    /// [`super::KtstrVm::collect_results`].
    pub(crate) cleanup_start: Instant,
    /// Cloned counter handle from `KtstrVm::init_virtio_blk`
    /// when a disk was attached, captured before the device-arc is
    /// dropped so [`super::KtstrVm::collect_results`] can snapshot
    /// it into [`VmResult::virtio_blk_counters`]. The device worker
    /// bumps these atomics from `drain_bracket_impl` (production cfg:
    /// dedicated `ktstr-vblk` thread; cfg(test): inline on the test
    /// thread); by the time `collect_results` reads this field every
    /// vCPU thread has joined upstream, the worker can receive no
    /// further kicks, and the conversion site
    /// (`run.virtio_blk_counters.as_deref().map(|c| c.snapshot())`)
    /// loads the final cumulative state into a plain-u64 snapshot
    /// before storing on the public `VmResult`.
    pub(crate) virtio_blk_counters: Option<Arc<VirtioBlkCounters>>,
    /// Cloned per-NIC counter handles from the net device init
    /// (`init_virtio_net` on aarch64 / `init_virtio_net_pci` on x86_64, both
    /// arch-gated), one per attached NIC, captured before the device arcs are
    /// dropped so [`super::KtstrVm::collect_results`] can snapshot and aggregate
    /// them into [`VmResult::virtio_net_counters`] via
    /// [`VirtioNetCountersSnapshot::aggregate`]. Empty when no NIC is attached.
    pub(crate) virtio_net_counters: Vec<Arc<VirtioNetCounters>>,
    /// Snapshot bridge owning every report captured during the run.
    /// The freeze coordinator clones this bridge into its closure
    /// state; on every guest-side
    /// [`crate::vmm::wire::MSG_TYPE_SNAPSHOT_REQUEST`] frame the
    /// coordinator's TOKEN_TX handler decoded with kind
    /// [`crate::vmm::wire::SNAPSHOT_KIND_CAPTURE`], the dispatch runs
    /// `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })` and stores the resulting
    /// `FailureDumpReport` here keyed by the snapshot name. After
    /// VM exit, [`super::KtstrVm::collect_results`] forwards the
    /// bridge onto [`VmResult::snapshot_bridge`] so the test code
    /// can drain captured snapshots and walk them via the
    /// [`crate::scenario::snapshot::Snapshot`] accessor surface.
    pub(crate) snapshot_bridge: crate::scenario::snapshot::SnapshotBridge,
    /// Cached aarch64 TCR_EL1 register, populated lazily by the BSP
    /// once the guest kernel programs the MMU. Always `None` on
    /// x86_64 (the register does not exist). Threads that construct
    /// a `GuestKernel` for page-table walks (monitor, BPF map writer,
    /// freeze coordinator, post-exit verifier-stats collector) read
    /// this atomic to feed the granule-agnostic walker (4 KB / 16 KB
    /// / 64 KB). A 0 reading on aarch64 means "kernel hasn't reached
    /// MMU bring-up yet"; the walker's T1SZ=0 gate rejects walks in
    /// that state and the affected lookup returns `None` cleanly.
    pub(crate) tcr_el1: Option<Arc<std::sync::atomic::AtomicU64>>,
    /// Cached BSP CR3 (x86_64) / TTBR1_EL1 (aarch64), populated lazily
    /// by the BSP loop after initial page-table setup. Used by
    /// post-exit `GuestKernel` constructions to walk the live page
    /// tables for `phys_base` resolution. `0` means the cache wasn't
    /// populated (early boot crash); the walk fails and `phys_base`
    /// falls back to `0`, which produces correct translations on
    /// non-KASLR boots.
    pub(crate) cr3: Arc<std::sync::atomic::AtomicU64>,
    /// Cached vmlinux bytes for collect_verifier_stats. Avoids
    /// re-reading from disk (14-28s on cold cache).
    pub(crate) vmlinux_data: Option<Arc<Vec<u8>>>,
    /// Pre-built prog accessor from the accessor-init worker.
    /// When present, `collect_verifier_stats` skips the ~4s
    /// ELF/BTF parse and uses this directly.
    pub(crate) prog_accessor: Option<crate::monitor::bpf_prog::GuestMemProgAccessorOwned>,
    /// Guest-reported phys_base (biased +1). Used by
    /// `collect_verifier_stats` fallback when the pre-built prog
    /// accessor is unavailable.
    pub(crate) kern_phys_base: u64,
    /// Runtime virt-KASLR offset (kernel-image slide), captured from
    /// the freeze coordinator's `kern_virt_kaslr` Arc snapshot at run
    /// end via `load(Acquire).saturating_sub(1)`. `0` means either
    /// (a) KASLR was off (test ran with `#[ktstr_test(kaslr = false)]`
    /// or `Scheduler::kargs(&["nokaslr"])`), or (b) the derivation
    /// chain (MSR_LSTAR readback at `vmm::x86_64::msr_kaslr` +
    /// KERN_ADDRS `_text` path at `crate::vmm::freeze_coord::dispatch`) never
    /// published a non-zero value (early-boot crash, kallsyms masked
    /// by kptr_restrict, FRED-enabled kernel). E2E test consumers
    /// distinguish (a) from (b) by asserting against the test entry's
    /// `kaslr` attribute. The companion [`Self::kern_phys_base`]
    /// carries the kernel-image physical-randomization slide; together
    /// they identify the KASLR-randomized kernel layout.
    pub kern_kaslr_offset: u64,
    /// Virtio-console device shared with vCPU threads. Carries the
    /// port-1 (`/dev/vport0p1`) bulk TLV stream from guest to host;
    /// `collect_results` calls `drain_bulk()` after the run to feed
    /// `parse_tlv_stream` and produce the `BulkDrainResult` that
    /// `VmResult.guest_messages` exposes to test verdicts.
    pub(crate) virtio_con: Arc<crate::vmm::PiMutex<crate::vmm::virtio_console::VirtioConsole>>,
    /// Bulk TLV entries the freeze coordinator parsed from
    /// `port1_tx_buf` mid-run. The coord's TOKEN_TX handler reads
    /// the device's accumulated bulk bytes, feeds them through
    /// [`crate::vmm::bulk::HostAssembler`], and stashes every parsed
    /// frame here so [`super::KtstrVm::collect_results`] can merge
    /// them into `VmResult::guest_messages` alongside the post-exit
    /// `drain_bulk` and the post-mortem SHM CRASH-ring drain.
    /// Without this stash every EXIT / TEST / PAYLOAD_METRICS /
    /// PROFRAW frame consumed by the coord
    /// would vanish — only the leftover bytes that arrived on
    /// `port1_tx_buf` after the coord exited would reach the
    /// verdict, and a typical run would surface no metrics.
    pub(crate) bulk_messages: Arc<std::sync::Mutex<Vec<crate::vmm::wire::ShmEntry>>>,
    /// Scheduler-stats client constructed at the top of `run_vm`,
    /// or `None` when the run has no scheduler attached
    /// (`scheduler_binary` is `None` on the builder). Forwarded
    /// to [`VmResult::stats_client`] so test code can issue
    /// `request_raw` / typed `stats` / `stats_meta` calls through
    /// the run's lifetime. The drainer thread tears down when the
    /// last clone of the client drops; `None` here means no
    /// drainer was spawned at all, so the run pays no
    /// stats-bridge cost.
    pub(crate) stats_client: Option<super::SchedStatsClient>,
    /// Periodic captures actually fired by the freeze coordinator
    /// during the run (success + timeout-placeholder count).
    /// Forwarded to [`VmResult::periodic_fired`] from the run-loop's
    /// `next_periodic_idx` final value.
    pub(crate) periodic_fired: u32,
    /// Configured periodic-snapshot target (mirrors
    /// `KtstrVm::num_snapshots`). Forwarded to
    /// [`VmResult::periodic_target`] so test code can compute
    /// coverage as `fired / target`.
    pub(crate) periodic_target: u32,
    /// Host-side vCPU scheduling-dilation sample, read at `run_vm`
    /// teardown from each vCPU thread's `/proc/self/task/<tid>/schedstat`
    /// (see [`HostVcpuSchedstat`]). Forwarded verbatim to
    /// [`VmResult::host_vcpu_schedstat`]. `None` on hosts without
    /// `CONFIG_SCHEDSTATS` or when no vCPU thread was sampled.
    pub(crate) host_vcpu_schedstat: Option<HostVcpuSchedstat>,
}
#[cfg(test)]
mod tests {
    use super::*;

    /// scheduler_log() concatenates the bulk-port SchedLog frames and
    /// slices the `SCHED_OUTPUT_START`/`END`-bracketed content (mirroring
    /// collect_verifier_output). A CRC-bad frame is dropped; with no valid
    /// SchedLog frames and an empty `output`, the log is empty. CI-runnable
    /// (no VM) — pins the accessor the demo_verifier post_vm assertions
    /// depend on, so a boot-path capture regression is caught even when the
    /// host-gated e2e cells skip under resource contention.
    #[test]
    fn scheduler_log_extracts_bracketed_content_and_drops_crc_bad() {
        use crate::vmm::host_comms::BulkDrainResult;
        use crate::vmm::wire::{MSG_TYPE_SCHED_LOG, ShmEntry};

        let framed = "===SCHED_OUTPUT_START===\n\
            libbpf: prog 'ktstr_dispatch': BPF program load failed: -EACCES\n\
            -- BEGIN PROG LOAD LOG --\n0: (b7) r0 = 0\n-- END PROG LOAD LOG --\n\
            ===SCHED_OUTPUT_END===\n";
        let mut result = VmResult::test_fixture();
        result.guest_messages = Some(BulkDrainResult {
            entries: vec![ShmEntry {
                msg_type: MSG_TYPE_SCHED_LOG,
                payload: framed.as_bytes().to_vec(),
                crc_ok: true,
            }],
        });
        let log = result.scheduler_log();
        assert!(
            log.contains("-- BEGIN PROG LOAD LOG --"),
            "libbpf verifier-reject marker survives extraction: {log}"
        );
        assert!(
            log.contains("BPF program load failed"),
            "scheduler stderr content present: {log}"
        );
        assert!(
            !log.contains("SCHED_OUTPUT_START"),
            "wire brackets stripped by parse_sched_output: {log}"
        );

        // A CRC-bad SchedLog frame is dropped; with no valid frames the
        // merged stream is empty and `output` (empty here) is the fallback,
        // so scheduler_log() is empty.
        let mut bad = VmResult::test_fixture();
        bad.guest_messages = Some(BulkDrainResult {
            entries: vec![ShmEntry {
                msg_type: MSG_TYPE_SCHED_LOG,
                payload: framed.as_bytes().to_vec(),
                crc_ok: false,
            }],
        });
        assert_eq!(bad.scheduler_log(), "", "crc-bad frame dropped -> empty");
    }

    /// `HostVcpuSchedstat::dilation` = `1 + run_delay/on_cpu`, `None`
    /// when no on-CPU time was sampled (idle/no-vCPU or a
    /// CONFIG_SCHEDSTATS-off host reading `0 0 0`). Directional endpoints:
    /// zero run-delay -> exactly 1.0; run-delay == on-cpu -> 2.0; zero
    /// on-cpu -> None (distinguishable from a genuine 1.0).
    #[test]
    fn host_vcpu_schedstat_dilation_endpoints() {
        // R == 0 -> D == 1.0 (always got the CPU when runnable).
        let d = HostVcpuSchedstat {
            total_on_cpu_ns: 1_000,
            total_run_delay_ns: 0,
            sampled_vcpus: 1,
        }
        .dilation();
        assert_eq!(d, Some(1.0), "no run delay -> exactly 1.0");
        // R == C -> D == 2.0 (waited as long as it ran).
        let d = HostVcpuSchedstat {
            total_on_cpu_ns: 1_000,
            total_run_delay_ns: 1_000,
            sampled_vcpus: 2,
        }
        .dilation();
        assert_eq!(d, Some(2.0), "run delay == on-cpu -> 2.0");
        // C == 0 -> None (no on-cpu time: idle, no vCPU, or schedstats off).
        let d = HostVcpuSchedstat {
            total_on_cpu_ns: 0,
            total_run_delay_ns: 500,
            sampled_vcpus: 0,
        }
        .dilation();
        assert_eq!(d, None, "zero on-cpu -> None, not a synthetic 1.0");
    }

    /// A StepStart/StepEnd/terminal `StimulusEvent` for the
    /// `step_throughput_in` pairing tests.
    fn ev(
        elapsed_ms: u64,
        step_index: Option<u16>,
        iters: Option<u64>,
        is_step_end: bool,
        is_terminal: bool,
    ) -> crate::timeline::StimulusEvent {
        crate::timeline::StimulusEvent {
            elapsed_ms,
            label: String::new(),
            op_kind: None,
            detail: None,
            total_iterations: iters,
            step_index,
            is_terminal,
            is_step_end,
        }
    }

    /// `step_throughput_in` pairs `StepStart[k]` -> `StepEnd[k]` of the SAME
    /// `Phase` for the step-local rate; falls back to the next step then the
    /// scenario-end terminal when a StepEnd is absent; a flat counter over a
    /// positive window is measured-zero `Some(0.0)` (not `None`); BASELINE /
    /// an absent step is `None`.
    #[test]
    fn step_throughput_in_pairs_step_local_and_handles_edges() {
        use crate::assert::Phase;
        let tl = vec![
            ev(0, Some(1), Some(0), false, false),       // StepStart[0]
            ev(1000, Some(1), Some(5000), true, false),  // StepEnd[0] -> 5000/s
            ev(1100, Some(2), Some(5000), false, false), // StepStart[1]
            ev(2100, Some(2), Some(5000), true, false),  // StepEnd[1] -> 0/s (flat)
            ev(2200, Some(3), Some(5000), false, false), // StepStart[2] (no StepEnd)
            crate::timeline::StimulusEvent::terminal(3200, 11000), // right boundary for step 2
        ];
        // Step 0: (5000-0)/1s = 5000/s.
        assert_eq!(
            VmResult::step_throughput_in(&tl, Phase::step(0)),
            Some(5000.0)
        );
        // Step 1: counter flat over a positive window -> measured zero
        // Some(0.0), NOT None.
        assert_eq!(VmResult::step_throughput_in(&tl, Phase::step(1)), Some(0.0));
        // Step 2: no StepEnd -> falls back to the terminal:
        // (11000-5000)/1s = 6000/s.
        assert_eq!(
            VmResult::step_throughput_in(&tl, Phase::step(2)),
            Some(6000.0)
        );
        // BASELINE and an absent step have no StepStart -> None.
        assert_eq!(VmResult::step_throughput_in(&tl, Phase::BASELINE), None);
        assert_eq!(VmResult::step_throughput_in(&tl, Phase::step(9)), None);
    }

    #[test]
    fn kvm_stats_try_sum_distinguishes_absent_from_zero() {
        let totals = KvmStatsTotals {
            per_vcpu: vec![
                [("exits".to_string(), 0u64)].into_iter().collect(),
                [("exits".to_string(), 0u64)].into_iter().collect(),
            ],
        };
        // "exits" is published as 0 on every vCPU -> a measured zero.
        assert_eq!(totals.try_sum("exits"), Some(0));
        assert_eq!(totals.try_avg("exits"), Some(0));
        // "halt_exits" was never published -> absent, not zero.
        assert_eq!(totals.try_sum("halt_exits"), None);
        assert_eq!(totals.try_avg("halt_exits"), None);
        // sum/avg keep the 0-coercing behavior for both cases.
        assert_eq!(totals.sum("exits"), 0);
        assert_eq!(totals.sum("halt_exits"), 0);
        // No vCPUs at all -> try_avg None (no div-by-zero, no false 0).
        let empty = KvmStatsTotals { per_vcpu: vec![] };
        assert_eq!(empty.try_sum("exits"), None);
        assert_eq!(empty.try_avg("exits"), None);
    }

    #[test]
    fn vm_result_fields_carry_values() {
        let r = VmResult {
            duration: Duration::from_secs(5),
            output: "hello world".into(),
            stderr: "boot log".into(),
            cleanup_duration: Some(Duration::from_millis(50)),
            ..VmResult::test_fixture()
        };
        assert!(r.success);
        assert_eq!(r.exit_code, 0);
        assert!(!r.timed_out);
        assert_eq!(r.duration, Duration::from_secs(5));
        assert_eq!(r.output, "hello world");
        assert_eq!(r.stderr, "boot log");
        assert!(r.monitor.is_none());
        assert!(r.guest_messages.is_none());
        assert!(r.stimulus_timeline().is_empty());
        assert_eq!(r.cleanup_duration, Some(Duration::from_millis(50)));
        assert!(r.virtio_blk_counters.is_none());
        // Second construction covers the opposite polarity of
        // every boolean/numeric field so no field is silently
        // dropped by a future refactor that only exercises the
        // success path.
        let r2 = VmResult {
            success: false,
            exit_code: 1,
            duration: Duration::from_millis(500),
            timed_out: true,
            virtio_blk_counters: Some(VirtioBlkCountersSnapshot::default()),
            periodic_fired: 3,
            periodic_real: 2,
            periodic_target: 7,
            ..VmResult::test_fixture()
        };
        assert!(!r2.success);
        assert_eq!(r2.exit_code, 1);
        assert!(r2.timed_out);
        assert_eq!(r2.duration, Duration::from_millis(500));
        assert!(r2.cleanup_duration.is_none());
        assert_eq!(r2.periodic_fired, 3);
        assert_eq!(r2.periodic_target, 7);
        // Opposite polarity: counters present. Reads must observe
        // the default-zero values for every field — a future field
        // added to VirtioBlkCountersSnapshot that doesn't initialise
        // to 0 would break the "fresh device reports zero activity"
        // contract that VmResult readers rely on. The snapshot was
        // taken from the device's atomic counters at collect_results
        // time, after every vCPU and worker thread joined; readers
        // see plain `u64` field reads with no atomic ordering needed.
        let counters = r2.virtio_blk_counters.as_ref().unwrap();
        assert_eq!(counters.reads_completed, 0);
        assert_eq!(counters.writes_completed, 0);
        assert_eq!(counters.flushes_completed, 0);
        assert_eq!(counters.bytes_read, 0);
        assert_eq!(counters.bytes_written, 0);
        assert_eq!(counters.throttled_count, 0);
        assert_eq!(counters.io_errors, 0);
        assert_eq!(counters.currently_throttled_gauge, 0);
        assert_eq!(counters.invalid_avail_idx_count, 0);
    }

    #[test]
    fn vm_result_without_monitor_has_no_samples() {
        let r = VmResult {
            output: "test output".into(),
            ..VmResult::test_fixture()
        };
        assert!(r.monitor.is_none());
        // Output and exit_code must still be accessible.
        assert_eq!(r.output, "test output");
        assert_eq!(r.exit_code, 0);
    }

    #[test]
    fn vm_result_with_monitor_carries_summary() {
        let summary = monitor::MonitorSummary {
            prog_stats_deltas: None,
            total_samples: 5,
            max_imbalance_ratio: 3.5,
            max_local_dsq_depth: 10,
            stuck_count: 1,
            event_deltas: None,
            schedstat_deltas: None,
            ..Default::default()
        };
        let report = monitor::MonitorReport {
            samples: vec![],
            summary: summary.clone(),
            ..Default::default()
        };
        let r = VmResult {
            success: false,
            exit_code: 1,
            duration: Duration::from_millis(500),
            timed_out: true,
            stderr: "kernel panic".into(),
            monitor: Some(report),
            ..VmResult::test_fixture()
        };
        let mon = r.monitor.as_ref().unwrap();
        assert_eq!(mon.summary.total_samples, 5);
        assert!((mon.summary.max_imbalance_ratio - 3.5).abs() < f64::EPSILON);
        assert_eq!(mon.summary.max_local_dsq_depth, 10);
        assert!(mon.summary.stuck_count > 0);
        assert!(r.timed_out);
        assert_eq!(r.exit_code, 1);
        assert_eq!(r.stderr, "kernel panic");
    }

    /// Compile-time pin that `VmResult: Clone`. A future field
    /// added with a non-Clone type would break the derive at compile
    /// time and break this test's `let _: Self = self_clone(r)` call.
    /// Cheap insurance that nobody silently strips the Clone derive
    /// or adds a non-Clone field.
    #[test]
    fn vm_result_is_clone() {
        fn self_clone<T: Clone>(t: &T) -> T {
            t.clone()
        }
        let r = VmResult::test_fixture();
        let _: VmResult = self_clone(&r);
    }

    /// Pin the documented aliasing semantic on the Arc-shared
    /// `snapshot_bridge` field: clones of `VmResult` share the
    /// underlying snapshot store. A future refactor that turned
    /// `SnapshotBridge` into a deep-copy struct would break this
    /// test — at which point the doc paragraph at the head of
    /// `VmResult` must be updated to drop the Arc-shared-handle
    /// category. Loud failure on contract drift, not a silent
    /// behavior change.
    #[test]
    fn vm_result_clone_snapshot_bridge_aliases_via_arc() {
        let r = VmResult::test_fixture();
        let c = r.clone();
        // Pre-condition: both bridges start empty.
        assert_eq!(r.snapshot_bridge.len(), 0);
        assert_eq!(c.snapshot_bridge.len(), 0);
        // Store a synthetic report through ONE clone's bridge.
        r.snapshot_bridge.store(
            "regression_pin",
            crate::monitor::dump::FailureDumpReport::default(),
        );
        // The OTHER clone observes the store — proves the Arc<Mutex<…>>
        // is shared, not deep-copied. If this assertion ever fires,
        // SnapshotBridge's Clone has changed shape and VmResult's
        // doc paragraph must be revisited.
        assert_eq!(
            r.snapshot_bridge.len(),
            c.snapshot_bridge.len(),
            "snapshot_bridge clones must observe the same store \
             per the VmResult Clone contract (Arc-shared handle)"
        );
        assert_eq!(c.snapshot_bridge.len(), 1);
    }

    /// Build a `VmResult` whose snapshot bridge holds `n` periodic
    /// captures stamped into Step[0] (`step_index = 1`). No stimulus
    /// frames are attached, so `stimulus_timeline()` is empty and the
    /// bucketer falls back to each capture's stamped `step_index`.
    fn vm_result_with_periodic_captures(n: usize) -> VmResult {
        let r = VmResult {
            periodic_fired: n as u32,
            periodic_target: n as u32,
            ..VmResult::test_fixture()
        };
        for i in 0..n {
            r.snapshot_bridge.store_with_stats_and_step(
                &format!("periodic_{i}"),
                crate::monitor::dump::FailureDumpReport::default(),
                None,
                Some(i as u64 * 100),
                None,
                1,
            );
        }
        r
    }

    /// Regression for the drain-once starvation bug: the snapshot
    /// bridge is drained EXACTLY once and the resulting series is
    /// shared, so a `post_vm`-style consumer reading the series does
    /// not starve a later framework-style consumer. Before
    /// [`VmResult::captures_series`], `periodic_series()` drained the
    /// bridge directly and a second reader saw an empty bridge — the
    /// silent-data-drop this task de-conflicts.
    #[test]
    fn captures_series_shared_across_consumers() {
        let r = vm_result_with_periodic_captures(3);
        // First consumer = post_vm style (drains the bridge into the cache).
        let post_vm_series = r.periodic_series();
        assert_eq!(
            post_vm_series.len(),
            3,
            "post_vm consumer must see all captures"
        );
        // Second consumer = framework style (reads the cache, no re-drain).
        let framework_series = r.captures_series();
        assert_eq!(
            framework_series.len(),
            3,
            "framework consumer must NOT see an empty bridge — the single \
             cached drain is shared, not re-drained (pre-cache this was 0)"
        );
        // The raw bridge was consumed exactly once: a direct drain now
        // yields nothing because captures_series() took ownership of the
        // captures into the cache on first read.
        assert_eq!(
            r.snapshot_bridge.drain_ordered_with_stats().len(),
            0,
            "captures_series() performs the single destructive drain"
        );
    }

    /// `phase_buckets()` folds the cached captures into per-phase
    /// buckets without the caller draining the bridge — the
    /// phase-buckets accessor. Idempotent: a second call returns the same vec from
    /// the shared cache.
    #[test]
    fn phase_buckets_from_cached_captures() {
        let r = vm_result_with_periodic_captures(2);
        let buckets = r.phase_buckets();
        assert!(
            !buckets.is_empty(),
            "phase_buckets must yield buckets from the cached captures"
        );
        assert!(
            buckets.iter().any(|b| b.step_index >= 1),
            "captures stamped step_index=1 must produce a Step bucket, got {:?}",
            buckets.iter().map(|b| b.step_index).collect::<Vec<_>>(),
        );
        assert_eq!(
            r.phase_buckets(),
            buckets,
            "phase_buckets() must be idempotent (shared cache)"
        );
    }

    /// Clone semantics, category 3 (the `periodic_series_cache` field):
    /// a clone taken AFTER the cache is populated carries an
    /// INDEPENDENT copy, so `phase_buckets()` on both the original and
    /// the clone returns identical non-empty buckets without
    /// re-touching the (already-drained) shared bridge. Pins the
    /// documented safe path.
    #[test]
    fn vm_result_clone_after_cache_populated_carries_buckets() {
        let r = vm_result_with_periodic_captures(2);
        // Populate the cache BEFORE cloning (the documented safe path).
        let original = r.phase_buckets();
        assert!(!original.is_empty());
        let c = r.clone();
        let cloned = c.phase_buckets();
        assert!(!cloned.is_empty());
        assert_eq!(
            cloned, original,
            "a clone taken after cache population must carry the same \
             buckets (category-3 independent-once-populated semantics)"
        );
    }

    /// `phase_metric` keys by `Phase` (1-indexed wire `step_index`) and
    /// delegates to the matching bucket's `get()`: it returns each
    /// present metric's value, `None` for an unknown metric on an
    /// existing phase, and `None` for a phase with no bucket. (The
    /// Some-value read itself is `PhaseBucket::get`, pinned by its own
    /// tests; this pins the phase-keying and the None contract.)
    #[test]
    fn phase_metric_keys_by_phase_and_delegates_to_bucket_get() {
        let r = vm_result_with_periodic_captures(2);
        // The captures are stamped step_index=1 -> Phase::step(0).
        let p0 = crate::assert::Phase::step(0);
        let buckets = r.phase_buckets();
        let step0 = buckets
            .iter()
            .find(|b| b.step_index == p0.as_u16())
            .expect("a Step[0] bucket from the stamped captures (keys by step_index)");
        // Some-path delegation: for every metric the matching bucket
        // carries, phase_metric returns that exact value. The
        // default-report fixture may fold no metrics, so this loop can be
        // empty — the keying + None assertions below pin the rest.
        for (name, val) in &step0.metrics {
            assert_eq!(
                r.phase_metric(p0, name),
                Some(*val),
                "phase_metric must return the matching bucket's value for present metric '{name}'",
            );
        }
        // A name absent from the bucket -> None (the bucket is still found).
        assert_eq!(
            r.phase_metric(p0, "definitely_not_a_registry_metric"),
            None,
            "an unknown metric name yields None even when the phase has a bucket",
        );
        // A phase with no bucket -> None (not a panic, not a wrong bucket).
        assert_eq!(
            r.phase_metric(crate::assert::Phase::step(99), "iteration_rate"),
            None,
            "a phase with no bucket yields None",
        );
    }

    /// Build a guest `AssertResult` carrying ONE per-phase per-cgroup
    /// carrier at `step_index` (the merge-neutral `(u64::MAX, 0)` window
    /// `fold_guest_per_cgroup_into_host_buckets` requires) and wrap it in
    /// the `MSG_TYPE_TEST_RESULT` TLV a real run leaves on
    /// `guest_messages`, so `VmResult::guest_assert_result` decodes it.
    fn guest_drain_with_per_cgroup(
        step_index: u16,
        carriers: &[(&str, u64, u64)],
    ) -> crate::vmm::host_comms::BulkDrainResult {
        let mut per_cgroup = std::collections::BTreeMap::new();
        for &(name, migrations, iters) in carriers {
            per_cgroup.insert(
                name.to_string(),
                crate::assert::PhaseCgroupStats {
                    total_migrations: migrations,
                    total_iterations: iters,
                    ..Default::default()
                },
            );
        }
        let mut guest = crate::test_support::test_helpers::build_assert_result(true, vec![]);
        guest.stats.phases = vec![crate::assert::PhaseBucket {
            step_index,
            label: crate::assert::Phase::from(step_index).to_string(),
            // Merge-neutral window: fold_guest_per_cgroup_into_host_buckets
            // debug_asserts guest carriers carry exactly (u64::MAX, 0).
            start_ms: u64::MAX,
            end_ms: 0,
            sample_count: 0,
            metrics: std::collections::BTreeMap::new(),
            per_cgroup,
        }];
        crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &guest,
            )],
        }
    }

    /// `phase_buckets()` folds the guest per-cgroup carriers (parsed from
    /// `guest_messages`) into the host buckets keyed by `step_index`, so the
    /// returned buckets carry `per_cgroup`; `phase_cgroup` reads one cgroup
    /// out of the folded bucket. Pins the guest-carrier fold: the host
    /// captures stamp step_index=1 == `Phase::step(0)`, the carrier matches
    /// it, and the matched arm unions the carrier's `per_cgroup` into that
    /// bucket.
    #[test]
    fn phase_buckets_folds_guest_per_cgroup_carriers() {
        let r = VmResult {
            guest_messages: Some(guest_drain_with_per_cgroup(1, &[("cellA", 7, 11)])),
            ..vm_result_with_periodic_captures(2)
        };
        let buckets = r.phase_buckets();
        let step0 = buckets
            .iter()
            .find(|b| b.step_index == crate::assert::Phase::step(0).as_u16())
            .expect("host bucket at step_index=1 from the stamped captures");
        let cg = step0
            .per_cgroup
            .get("cellA")
            .expect("matched-arm fold must carry the guest carrier's per_cgroup");
        assert_eq!(cg.total_migrations, 7);
        assert_eq!(cg.total_iterations, 11);
        // phase_cgroup is the per-(phase, cgroup) accessor over the same fold.
        let via_accessor = r
            .phase_cgroup(crate::assert::Phase::step(0), "cellA")
            .expect("phase_cgroup must reach the folded carrier");
        assert_eq!(via_accessor.total_migrations, 7);
        // A cgroup the phase never had -> None (not measured).
        assert!(
            r.phase_cgroup(crate::assert::Phase::step(0), "nope")
                .is_none()
        );
    }

    /// `phase_metric("total_migrations")` resolves the per-phase
    /// cross-cgroup SUM via the `cgroup_counter_total` fallback — the key
    /// lives ONLY in `per_cgroup` (its `read_sample` is `None`, so
    /// `build_phase_buckets` never folds it into `bucket.metrics`, so
    /// `get()` misses). Two cgroups in the phase sum: 7 + 3 = 10. Without
    /// the `cgroup_counter_total` fallback, `phase_metric` returns a silent
    /// `None` for these keys.
    #[test]
    fn phase_metric_resolves_total_migrations_from_per_cgroup() {
        let r = VmResult {
            guest_messages: Some(guest_drain_with_per_cgroup(
                1,
                &[("cellA", 7, 5), ("cellB", 3, 6)],
            )),
            ..vm_result_with_periodic_captures(2)
        };
        assert_eq!(
            r.phase_metric(crate::assert::Phase::step(0), "total_migrations"),
            Some(10.0),
            "cross-cgroup sum across per_cgroup carriers (7 + 3)",
        );
        // The total_iterations arm of cgroup_counter_total resolves too (5 + 6).
        assert_eq!(
            r.phase_metric(crate::assert::Phase::step(0), "total_iterations"),
            Some(11.0),
            "cross-cgroup sum of the total_iterations per_cgroup counter (5 + 6)",
        );
        // A carrier-bearing phase that counted zero is a measured Some(0.0),
        // distinct from a phase with no carrier (None).
        let r0 = VmResult {
            guest_messages: Some(guest_drain_with_per_cgroup(1, &[("cellA", 0, 0)])),
            ..vm_result_with_periodic_captures(2)
        };
        assert_eq!(
            r0.phase_metric(crate::assert::Phase::step(0), "total_migrations"),
            Some(0.0),
            "carriers present but counted zero is a measured Some(0.0)",
        );
        // No carriers at all (guest_messages with no per_cgroup) -> None.
        let r_none = vm_result_with_periodic_captures(2);
        assert_eq!(
            r_none.phase_metric(crate::assert::Phase::step(0), "total_migrations"),
            None,
            "no per_cgroup carrier in the phase -> None (not measured)",
        );
    }

    /// A guest carrier whose `step_index` has NO matching host bucket takes
    /// the fold's ORPHAN arm: it is appended with its merge-neutral
    /// `(u64::MAX, 0)` window normalized to `(0, 0)`, and stays reachable
    /// through the public accessors. The host captures stamp step_index=1
    /// only, so a step_index=2 carrier is an orphan. Pins that
    /// `phase_cgroup` reaches the orphan and the window does not underflow
    /// duration consumers (`end_ms - start_ms == 0`, not `0 - u64::MAX`).
    #[test]
    fn phase_buckets_orphan_carrier_reachable_via_phase_cgroup() {
        let r = VmResult {
            // step_index 2 == Phase::step(1); no host capture stamps step 2.
            guest_messages: Some(guest_drain_with_per_cgroup(2, &[("orphanCell", 4, 0)])),
            ..vm_result_with_periodic_captures(2)
        };
        let buckets = r.phase_buckets();
        let orphan = buckets
            .iter()
            .find(|b| b.step_index == crate::assert::Phase::step(1).as_u16())
            .expect("orphan carrier must be appended as its own bucket");
        assert_eq!(
            (orphan.start_ms, orphan.end_ms),
            (0, 0),
            "orphan arm normalizes the (u64::MAX, 0) sentinel window to (0, 0)",
        );
        assert_eq!(
            r.phase_cgroup(crate::assert::Phase::step(1), "orphanCell")
                .map(|c| c.total_migrations),
            Some(4),
            "phase_cgroup must reach an orphan carrier",
        );
    }

    /// With no guest verdict (guest_messages None — host-only run / early
    /// crash), `guest_assert_result()` is Err and `phase_buckets()` returns
    /// the host-rebuilt buckets ALONE: non-empty (captures landed) with every
    /// `per_cgroup` empty. Pins the Err arm (the prior behavior) carries the
    /// host buckets, no panic.
    #[test]
    fn phase_buckets_without_guest_verdict_is_host_only() {
        let r = vm_result_with_periodic_captures(2);
        assert!(
            r.guest_assert_result().is_err(),
            "test_fixture has no guest verdict"
        );
        let buckets = r.phase_buckets();
        assert!(
            !buckets.is_empty(),
            "host captures must still yield buckets"
        );
        assert!(
            buckets.iter().all(|b| b.per_cgroup.is_empty()),
            "no guest carriers -> every per_cgroup stays empty",
        );
    }

    /// A guest per-cgroup carrier carrying NON-EMPTY sample vectors
    /// (run_delays_ns / off_cpu_pcts / wake_latencies_ns) survives the
    /// guest->host postcard round-trip (assert_result_tlv_entry encode ->
    /// guest_assert_result decode) and the fold verbatim, and the
    /// PhaseCgroupStats summary methods reduce them post-decode. The
    /// guest_drain_with_per_cgroup helper carries only counters (empty
    /// vecs), so this pins the sample-vec wire fidelity that helper does not.
    #[test]
    fn guest_carrier_sample_vecs_survive_postcard_and_fold() {
        let mut pc = std::collections::BTreeMap::new();
        pc.insert(
            "cellA".to_string(),
            crate::assert::PhaseCgroupStats {
                total_migrations: 2,
                run_delays_ns: vec![10, 20, 30],
                off_cpu_pcts: vec![1.5, 2.5],
                wake_latencies_ns: vec![100, 200],
                wake_sample_total: 2,
                ..Default::default()
            },
        );
        let mut guest = crate::test_support::test_helpers::build_assert_result(true, vec![]);
        guest.stats.phases = vec![crate::assert::PhaseBucket {
            step_index: 1,
            label: "Step[0]".to_string(),
            start_ms: u64::MAX,
            end_ms: 0,
            sample_count: 0,
            metrics: std::collections::BTreeMap::new(),
            per_cgroup: pc,
        }];
        let r = VmResult {
            guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
                entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                    &guest,
                )],
            }),
            ..vm_result_with_periodic_captures(2)
        };
        let cg = r
            .phase_cgroup(crate::assert::Phase::step(0), "cellA")
            .expect("carrier survives postcard + fold");
        // Sample vectors round-trip byte-for-byte through the wire.
        assert_eq!(cg.run_delays_ns, vec![10, 20, 30]);
        assert_eq!(cg.off_cpu_pcts, vec![1.5, 2.5]);
        assert_eq!(cg.wake_latencies_ns, vec![100, 200]);
        // And the summary methods reduce the decoded samples (worst = max(ns)/1000).
        let (_, worst_us) = cg
            .run_delay_summary()
            .expect("non-empty run_delays -> Some");
        assert_eq!(worst_us, 30.0 / 1000.0);
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn vm_result_wprof_pb_path_bails_when_entry_name_none() {
        let r = VmResult::test_fixture();
        assert!(r.entry_name.is_none());
        let err = r.wprof_pb_path().expect_err("None entry_name must Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("entry_name"),
            "diagnostic must name the missing field: {msg}",
        );
        assert!(
            msg.contains("run_ktstr_test_inner_impl"),
            "diagnostic must name the stamping site so the operator \
             can trace the missing-stamp path: {msg}",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn vm_result_wprof_pb_path_returns_writer_mirror_path() {
        let r = VmResult {
            entry_name: Some("vm_result_wprof_pb_path_returns_writer_mirror_path_fixture"),
            variant_hash: 0xab,
            ..VmResult::test_fixture()
        };
        let path = r.wprof_pb_path().expect("Some entry_name must Ok");
        // The path's file_name must exactly match
        // `<entry_name>-<variant_hash:016x>.wprof.pb` — the writer uses
        // the same variant-keyed pattern (the wprof writer reads this
        // very method). A divergence would mean the method derives a
        // different path than the writer wrote to, surfacing as ENOENT in
        // the post_vm callback.
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap();
        assert_eq!(
            file_name,
            "vm_result_wprof_pb_path_returns_writer_mirror_path_fixture-00000000000000ab.wprof.pb",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn vm_result_repro_wprof_pb_path_bails_when_entry_name_none() {
        let r = VmResult::test_fixture();
        let err = r
            .repro_wprof_pb_path()
            .expect_err("None entry_name must Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("entry_name"));
        assert!(msg.contains("run_ktstr_test_inner_impl"));
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn vm_result_repro_wprof_pb_path_returns_writer_mirror_path() {
        let r = VmResult {
            entry_name: Some("vm_result_repro_wprof_pb_path_fixture"),
            variant_hash: 0xab,
            ..VmResult::test_fixture()
        };
        let path = r.repro_wprof_pb_path().expect("Some entry_name must Ok");
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap();
        assert_eq!(
            file_name,
            "vm_result_repro_wprof_pb_path_fixture-00000000000000ab.repro.wprof.pb"
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn vm_result_assert_wprof_pb_landed_skips_when_success_false() {
        let r = VmResult {
            success: false,
            ..VmResult::test_fixture()
        };
        assert!(r.entry_name.is_none());
        let result = r.assert_wprof_pb_landed();
        assert!(
            result.is_ok(),
            "assert_wprof_pb_landed must Ok-skip on !success EVEN when \
             entry_name is None — the entry_name pre-check is downstream of \
             the success short-circuit. Got: {result:?}",
        );
    }
}
