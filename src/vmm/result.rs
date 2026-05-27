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
#[derive(Debug, Clone)]
pub struct VmResult {
    /// Overall success flag: `true` when the test reported a pass AND
    /// the VM exited cleanly without crash, timeout, or watchdog.
    pub success: bool,
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
    /// Stimulus events extracted from guest TLV entries.
    #[allow(dead_code)]
    pub stimulus_events: Vec<wire::StimulusEvent>,
    /// BPF verifier stats collected from host-side memory reads.
    pub verifier_stats: Vec<monitor::bpf_prog::ProgVerifierStats>,
    /// KVM per-vCPU cumulative stats (requires Linux >= 5.14).
    pub kvm_stats: Option<KvmStatsTotals>,
    /// Crash message extracted from COM2 output via
    /// `crate::test_support::extract_panic_message`. The guest
    /// panic hook in `rust_init.rs` writes `PANIC: <info>\n<bt>\n`
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
    /// `#[allow(dead_code)]` mirrors `stimulus_events` above: the
    /// field is part of the public API surface and read by user
    /// test code outside `lib.rs`, but the lib build doesn't see
    /// any in-tree readers because no lib code path calls
    /// `.virtio_blk_counters` on a `VmResult`. The in-tree readers
    /// live in unit tests.
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
    /// The counter struct exposes eleven `AtomicU64` fields, each
    /// bumped from `process_tx_loopback`:
    ///
    ///   - `tx_packets` — count of TX chains the device accepted
    ///     and marked used; advances per parsed chain regardless of
    ///     downstream RX outcome.
    ///   - `tx_bytes` — bytes of L2 frame data captured from
    ///     successfully parsed TX chains (excludes the 12-byte
    ///     virtio header).
    ///   - `rx_packets` / `rx_bytes` — count + bytes of RX chains
    ///     successfully written and marked used. In v0's pure-
    ///     loopback mode the steady-state expectation is
    ///     `rx_packets == tx_packets - tx_dropped_no_rx_buffer`;
    ///     asymmetric counts surface RX-side breakage.
    ///   - `tx_dropped_no_rx_buffer` — successfully-captured TX
    ///     frames the device could not deliver because the RX queue
    ///     was empty (back-pressure event).
    ///   - `tx_chain_invalid` / `rx_chain_invalid` — chains rejected
    ///     for malformed shape (short header, wrong direction,
    ///     attacker-controlled descriptor address overflow).
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
    /// **Drained by `evaluate_vm_result`**: the framework's
    /// `crate::test_support::eval` path drains this bridge to
    /// auto-populate [`crate::assert::ScenarioStats::phases`]
    /// before returning the AssertResult. A `post_vm` callback or
    /// any code path that runs THROUGH `evaluate_vm_result`
    /// observes an empty bridge here — the periodic captures the
    /// drain consumed are recovered as the per-phase
    /// [`crate::assert::PhaseBucket`] entries on
    /// `result.stats.phases`, which is the framework-curated
    /// equivalent surface. Integration tests under `tests/` that
    /// bypass `evaluate_vm_result` (e.g. `tests/stats_bridge_e2e.rs`,
    /// `tests/temporal_assertions_e2e.rs`) see the bridge intact
    /// because their entry path never reaches the auto-populate
    /// site; those consumers continue to call
    /// `result.snapshot_bridge.drain*()` directly without
    /// observable contract change.
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
    /// KERN_ADDRS `_text` path in `freeze_coord::dispatch.rs`) never
    /// published a non-zero value (early-boot crash, kallsyms masked
    /// by kptr_restrict, FRED-enabled kernel). E2E test consumers
    /// distinguish (a) from (b) by reading the test entry's `kaslr`
    /// attribute alongside this field — see
    /// [`Self::kaslr_enabled`] for the binary-question companion.
    pub kern_kaslr_offset: u64,
    /// Name of the `#[ktstr_test]` fn whose execution produced this
    /// result. Stamped from
    /// [`crate::test_support::entry::KtstrTestEntry::name`] (a
    /// `&'static str` the macro emits at compile time) in
    /// `test_support::eval::run_ktstr_test_inner_impl` immediately
    /// after [`super::KtstrVm::run`] returns and BEFORE the
    /// `post_vm` callback dispatch runs.
    ///
    /// `Some(_)` for every result that flowed through the real
    /// `run_ktstr_test_inner_impl` path. `None` for the
    /// `freeze_coord::collect_results` direct-synthesis path
    /// (entry-agnostic boundary; entry is not in scope there) and
    /// for `#[cfg(test)]`-only `Self::test_fixture` callers. The
    /// path-derivation methods [`Self::wprof_pb_path`] and
    /// [`Self::repro_wprof_pb_path`] bail with a loud diagnostic
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

    /// One-line sugar for the recurring `post_vm`-callback boilerplate
    /// `SampleSeries::from_drained_typed(self.snapshot_bridge.drain_ordered_with_stats(), self.monitor.clone()).periodic_only()`.
    /// Equivalent in every observable way: same drain, same monitor
    /// clone, same `periodic_only()` filter — exposed as a single
    /// method so every benchmarking / per-phase / cross-phase test
    /// expresses the projection in one statement instead of three.
    ///
    /// The bridge drain is destructive (the snapshot bridge yields
    /// each capture exactly once); calling this method twice on the
    /// same [`VmResult`] leaves the second call with an empty series.
    /// If a post_vm callback needs both the raw drain and a series
    /// view, drain the bridge into a local Vec first and construct
    /// the series via [`crate::scenario::sample::SampleSeries::from_drained_typed`].
    ///
    /// Takes `&self` rather than `&mut self` so it composes with the
    /// `#[ktstr_test(post_vm = ...)]` callback signature
    /// (`fn(&VmResult) -> Result<()>`). The underlying bridge uses
    /// interior mutability for its drain queue, so the destructive
    /// semantics ride on the bridge's lock rather than Rust's
    /// borrow-check exclusivity.
    pub fn periodic_series(&self) -> crate::scenario::sample::SampleSeries {
        crate::scenario::sample::SampleSeries::from_drained_typed(
            self.snapshot_bridge.drain_ordered_with_stats(),
            self.monitor.clone(),
        )
        .periodic_only()
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
            expect_auto_repro_satisfied: false,
            exit_code: 0,
            duration: Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            guest_messages: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
            virtio_blk_counters: None,
            virtio_net_counters: None,
            snapshot_bridge: empty_snapshot_bridge_for_tests(),
            stats_client: None,
            periodic_fired: 0,
            periodic_target: 0,
            kern_kaslr_offset: 0,
            entry_name: None,
        }
    }

    /// Per-test sidecar path where the `.wprof.pb` artifact lands
    /// after `vm.run()` returns. Derives the path as
    /// `{sidecar_dir()}/{entry_name}.wprof.pb` — mirrors the writer
    /// site in `test_support::eval::run_ktstr_test_inner`'s
    /// `MsgType::WprofTrace` handler.
    ///
    /// Returns `Err` with an actionable diagnostic when
    /// [`Self::entry_name`] is `None` (the
    /// `freeze_coord::collect_results` direct-synthesis path leaves
    /// it `None` and the eval-layer's `run_ktstr_test_inner_impl`
    /// stamps `Some(entry.name)` after `vm.run()` returns; a `None`
    /// reaching this method means the stamping path was bypassed).
    /// The loud bail is the structural pin for drift-safe test
    /// naming: a `post_vm` callback that derives the path through
    /// this method cannot hardcode a stale fn-name literal.
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

    /// Per-test sidecar path where the `.repro.wprof.pb` artifact
    /// lands when `entry.auto_repro = true` AND the primary VM
    /// reported failure AND auto-repro fired. Derives the path as
    /// `{sidecar_dir()}/{entry_name}.repro.wprof.pb` — mirrors the
    /// auto-repro writer in
    /// `test_support::eval::run_ktstr_test_inner`. Distinct from
    /// [`Self::wprof_pb_path`] because the primary-VM and auto-repro
    /// VM artifacts have different lifecycles: primary always lands
    /// when wprof attached, auto-repro lands only on the conditional
    /// re-run.
    ///
    /// Returns `Err` symmetric to [`Self::wprof_pb_path`] when
    /// [`Self::entry_name`] is `None`.
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
    /// Returns `Err` symmetric to [`Self::wprof_pb_path`] when
    /// [`Self::entry_name`] is `None`.
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

    /// Skip-on-fail wrapper that asserts the primary-VM `.wprof.pb`
    /// landed at the expected sidecar path and satisfies the wprof
    /// shape contract — file exists, size >=
    /// [`crate::test_support::wprof::WPROF_PB_MIN_BYTES`], leads
    /// with [`crate::test_support::wprof::PERFETTO_TRACE_PACKETS_TAG`].
    ///
    /// Returns `Ok(())` immediately when `self.success` is `false`
    /// (the framework's failure path emits its own diagnostic before
    /// the `post_vm` callback runs; suppressing the assertion here
    /// avoids drowning the real fail message in a secondary
    /// missing-artifact failure). Otherwise delegates to
    /// [`crate::test_support::wprof::assert_wprof_pb_shape`] on
    /// [`Self::wprof_pb_path`]'s result.
    ///
    /// Replaces the inline 7-line `if !result.success { return Ok(());
    /// } let pb = wprof_pb_path("<literal>"); assert_wprof_pb_shape(&pb)`
    /// boilerplate previously duplicated across wprof-artifact tests.
    /// The literal-fn-name footgun is gone — the path derives from
    /// the macro-stamped [`Self::entry_name`].
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
    /// Sum a stat across all vCPUs.
    pub fn sum(&self, name: &str) -> u64 {
        self.per_vcpu.iter().filter_map(|m| m.get(name)).sum()
    }

    /// Average a stat across all vCPUs (returns 0 if no vCPUs).
    pub fn avg(&self, name: &str) -> u64 {
        if self.per_vcpu.is_empty() {
            return 0;
        }
        self.sum(name) / self.per_vcpu.len() as u64
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
    /// KERN_ADDRS `_text` path at `freeze_coord::dispatch.rs`) never
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
        assert!(r.stimulus_events.is_empty());
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
            stuck_detected: true,
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
        assert!(mon.summary.stuck_detected);
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

    /// `wprof_pb_path()` bails on a manually-constructed VmResult
    /// whose `entry_name` is `None`. Pins the structural drift-safe
    /// contract: post_vm callbacks that reach this method without the
    /// eval-layer stamping cannot silently produce a garbage path.
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

    /// Happy-path: `wprof_pb_path()` returns the writer-mirror path
    /// `{sidecar_dir()}/{entry_name}.wprof.pb` when entry_name is
    /// populated. Pins the format string against drift between the
    /// host-side writer (eval.rs MsgType::WprofTrace arm) and this
    /// derivation method.
    #[test]
    fn vm_result_wprof_pb_path_returns_writer_mirror_path() {
        let r = VmResult {
            entry_name: Some("vm_result_wprof_pb_path_returns_writer_mirror_path_fixture"),
            ..VmResult::test_fixture()
        };
        let path = r.wprof_pb_path().expect("Some entry_name must Ok");
        // The path's file_name must exactly match `<entry_name>.wprof.pb`
        // — the writer at eval.rs uses the same `format!("{}.wprof.pb",
        // entry.name)` pattern. A divergence here would mean the
        // method derives a different path than the writer wrote to,
        // surfacing as ENOENT in the post_vm callback.
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap();
        assert_eq!(
            file_name,
            "vm_result_wprof_pb_path_returns_writer_mirror_path_fixture.wprof.pb",
        );
    }

    /// Symmetric bail-on-None for the auto-repro variant. Same
    /// rationale as the primary `.wprof.pb` path: silent garbage-path
    /// construction is the failure mode the loud bail prevents.
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

    /// Happy-path for the auto-repro variant. Format string
    /// `{entry_name}.repro.wprof.pb` mirrors the host-side
    /// auto-repro writer in eval.rs.
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

    /// `assert_wprof_pb_landed()` returns Ok(()) immediately when
    /// `success` is false WITHOUT touching `entry_name` or reading
    /// the sidecar file. Pins the framework's "let the runner render
    /// the real failure" suppression contract: a callback that
    /// asserts wprof shape on a crashed-VM result would otherwise
    /// drown the original failure in a missing-artifact secondary.
    ///
    /// Uses a `test_fixture` with `success: false` AND `entry_name:
    /// None`. If the skip-on-fail short-circuit regressed, the
    /// downstream `entry_name` bail would fire and the test would
    /// see an `Err`. The `Ok(())` result confirms the short-circuit
    /// fired BEFORE the entry_name pre-check.
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
