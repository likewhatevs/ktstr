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
    /// Captured guest stdout (and any non-dmesg serial console content).
    pub output: String,
    /// Captured guest stderr (separated from `output` when the guest
    /// reported them distinctly).
    pub stderr: String,
    /// Host-side monitor report: sampled per-CPU state, stall
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
    /// bumped from `drain_bracket_impl` (in `src/vmm/virtio_blk/device.rs`)
    /// via the `VirtioBlkCounters::record_*` helpers. Per-request
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
    /// guest has exited. `Some(_)` when the builder attached a
    /// network via `super::KtstrVmBuilder::network`; `None` when
    /// no network was configured and
    /// `super::KtstrVm::init_virtio_net` returned `None`. The
    /// device increments its internal `AtomicU64` counters on the
    /// vCPU thread inside `process_tx_loopback`; by the time
    /// `collect_results` constructs the [`VmResult`] every vCPU has
    /// joined and no further mutation can occur. The snapshot is
    /// taken at that point — readers see plain `u64` fields holding
    /// the final cumulative totals; no atomic load is needed on the
    /// consumer side.
    ///
    /// The counter struct exposes thirteen `AtomicU64` fields, each
    /// bumped across the TX-drain path rooted at `process_tx_loopback`
    /// (several are bumped inside the `pop_and_capture_tx` /
    /// `try_loopback_to_rx` helpers it calls):
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

    /// The framework-computed per-phase metric buckets for this run —
    /// the SAME [`crate::assert::PhaseBucket`] vec the framework folds
    /// onto [`crate::assert::ScenarioStats::phases`] in
    /// `evaluate_vm_result`.
    ///
    /// This is the answer to "my `post_vm` callback wants the per-phase
    /// metrics the framework already built." Before this accessor, a
    /// `post_vm` had to re-derive them by hand —
    /// `build_phase_buckets_with_stimulus(&result.periodic_series(),
    /// &result.stimulus_timeline())` — which both duplicated the
    /// framework's logic AND destructively drained the bridge, leaving
    /// the framework's own `result.stats.phases` empty (the drain-once
    /// starvation [`Self::captures_series`] now prevents).
    ///
    /// Folds [`Self::periodic_series`] (the periodic-only projection of
    /// the shared single drain — on-demand / watchpoint captures are
    /// off-cadence outliers excluded from per-phase folds) through
    /// [`crate::assert::build_phase_buckets_with_stimulus`] using
    /// [`Self::stimulus_timeline`] for the step windows. In production
    /// the framework builds `stats.phases` from the same periodic-only
    /// series and the same stimulus timeline, so this returns content
    /// identical to `result.stats.phases` — but ONLY when there are no
    /// step-local-cgroup per_cgroup carriers (pinned by
    /// `phase_buckets_equals_stats_phases_and_post_vm_read_does_not_starve`,
    /// whose fixture has none). With step-local cgroups, `evaluate_vm_result`
    /// folds the guest per_cgroup carriers into `stats.phases` (and may append
    /// orphan buckets) via `fold_guest_per_cgroup_into_host_buckets`, so
    /// `stats.phases` is then a SUPERSET — this accessor still returns the
    /// host-rebuilt buckets with EMPTY per_cgroup and no orphans, i.e. it does
    /// NOT expose per_cgroup.
    pub fn phase_buckets(&self) -> Vec<crate::assert::PhaseBucket> {
        crate::assert::build_phase_buckets_with_stimulus(
            &self.periodic_series(),
            &self.stimulus_timeline(),
        )
    }

    /// One framework-computed per-phase metric for `phase` — the
    /// metric-name analog of [`Self::step_throughput`] /
    /// [`Self::throughput_ratio`], covering the per-phase metrics that
    /// are NOT iteration throughput: per-phase CPU time
    /// (`system_time_ns`, `user_time_ns`) and scheduling quality
    /// (`avg_imbalance_ratio`, `avg_dsq_depth`, ...). `metric` is a
    /// `crate::stats::METRICS` registry name. (The wake-latency / run-delay
    /// distributions are `MetricKind::Distribution`: they have no per-phase
    /// `PhaseBucket::metrics` value — they re-pool run-level from the
    /// per-cgroup carriers — so they are not readable here.)
    ///
    /// Reads the SAME buckets [`Self::phase_buckets`] folds onto
    /// `result.stats.phases`, so a `post_vm` callback can compare any
    /// metric across two phases (e.g. scheduler-vs-detached-EEVDF
    /// `system_time_ns`) without re-deriving the buckets — the
    /// general form of the scheduler-vs-EEVDF throughput compare.
    /// `None` when `phase` has no bucket — no capture landed in it AND no
    /// stimulus `StepStart` synthesized one (e.g. `Phase::BASELINE` when
    /// the settle window fired no captures, or a `phase` past the last
    /// step) — or the bucket carries no reading for `metric` (every
    /// sample's per-phase reading was absent — the sentinel-free "no data"
    /// contract, distinct from a real `Some(0.0)`). A started-but-
    /// uncaptured step (a `StepStart` with zero captures) DOES produce a
    /// synthesized bucket, so `phase_metric` returns its stimulus-derived
    /// `iteration_rate` rather than `None`.
    pub fn phase_metric(&self, phase: crate::assert::Phase, metric: &str) -> Option<f64> {
        self.phase_buckets()
            .into_iter()
            .find(|b| b.step_index == phase.as_u16())
            .and_then(|b| b.get(metric))
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
            expect_auto_repro_satisfied: false,
            exit_code: 0,
            duration: Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            guest_messages: None,
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
            virtio_blk_counters: None,
            virtio_net_counters: None,
            snapshot_bridge: empty_snapshot_bridge_for_tests(),
            stats_client: None,
            periodic_fired: 0,
            periodic_real: 0,
            periodic_target: 0,
            kern_kaslr_offset: 0,
            entry_name: None,
            periodic_series_cache: std::sync::OnceLock::new(),
        }
    }

    /// Per-test sidecar path for the `.wprof.pb` artifact:
    /// `{sidecar_dir()}/{entry_name}.wprof.pb`.
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
        Ok(crate::test_support::sidecar_dir().join(format!("{name}.wprof.pb")))
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
        Ok(crate::test_support::sidecar_dir().join(format!("{name}.repro.wprof.pb")))
    }

    /// Per-test failure-dump sidecar path. Derives
    /// `{sidecar_dir()}/{entry_name}.failure-dump.json` from
    /// the macro-stamped [`Self::entry_name`].
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
        Ok(crate::test_support::sidecar_dir().join(format!("{name}.failure-dump.json")))
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
    /// Cloned counter handle from [`super::KtstrVm::init_virtio_blk`]
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
    /// Cloned counter handle from [`super::KtstrVm::init_virtio_net`]
    /// when a network was attached, captured before the device-arc
    /// is dropped so [`super::KtstrVm::collect_results`] can
    /// snapshot it into [`VmResult::virtio_net_counters`]. Same
    /// Arc-handoff + snapshot-at-assignment pattern as
    /// `virtio_blk_counters` above.
    pub(crate) virtio_net_counters: Option<Arc<VirtioNetCounters>>,
    /// Snapshot bridge owning every report captured during the run.
    /// The freeze coordinator clones this bridge into its closure
    /// state; on every guest-side
    /// [`crate::vmm::wire::MSG_TYPE_SNAPSHOT_REQUEST`] frame the
    /// coordinator's TOKEN_TX handler decoded with kind
    /// [`crate::vmm::wire::SNAPSHOT_KIND_CAPTURE`], the dispatch runs
    /// `freeze_and_capture(false)` and stores the resulting
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
    /// RAW_PAYLOAD_OUTPUT / PROFRAW frame consumed by the coord
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
}
#[cfg(test)]
mod tests {
    use super::*;

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
            ..VmResult::test_fixture()
        };
        let path = r.wprof_pb_path().expect("Some entry_name must Ok");
        // The path's file_name must exactly match `<entry_name>.wprof.pb`
        // — the writer in `run_ktstr_test_inner_impl` uses the same `format!("{}.wprof.pb",
        // entry.name)` pattern. A divergence here would mean the
        // method derives a different path than the writer wrote to,
        // surfacing as ENOENT in the post_vm callback.
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap();
        assert_eq!(
            file_name,
            "vm_result_wprof_pb_path_returns_writer_mirror_path_fixture.wprof.pb",
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
            ..VmResult::test_fixture()
        };
        let path = r.repro_wprof_pb_path().expect("Some entry_name must Ok");
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap();
        assert_eq!(
            file_name,
            "vm_result_repro_wprof_pb_path_fixture.repro.wprof.pb"
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
