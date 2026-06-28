//! Worker-side execution: `worker_main`, the per-WorkType loops, and
//! the supporting helpers (futex syscalls, IO backings, schedstat
//! readers, NUMA / clock helpers, scheduling-policy applier). Split
//! out of `workload/mod.rs` to keep the production code path under
//! 3500 lines per file. Tests are co-located in `worker/tests.rs`
//! and re-export production items through this module's `use
//! io::*;` / `use sched::*;` glob imports.

use std::collections::BTreeSet;
use std::io::{Seek, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::affinity::{sched_getcpu, set_thread_affinity};
use super::config::{
    AluWidth, FutexLockMode, MemPolicy, MpolFlags, ReapMode, SchedPolicy, WakeMechanism,
};
use super::spawn::{
    FAN_OUT_POST_WAKE_SPIN_ITERS, FUTEX_WAIT_TIMEOUT, Migration, PhaseSlice, WorkerReport,
    apply_mempolicy_with_flags, apply_nice, build_nodemask, stop_requested,
};
use super::types::*;

/// Per-worker reservoir-sample cap for the wake-latency
/// ([`WorkerReport::wake_latencies_ns`]) and per-iteration
/// cost ([`WorkerReport::iteration_costs_ns`]) buffers.
///
/// The cap bounds memory per worker on long-running scenarios:
/// once filled the reservoir uses Vitter's Algorithm R for uniform
/// sampling, so a 10-hour run with millions of wake events still
/// caps at 100_000 entries (≈ 800 KiB per worker per buffer).
///
/// Promoted to module scope so `tests::max_wake_samples_pins_doc_value`
/// can pin the value against the documentation cite in
/// `doc/guide/src/architecture/workers.md`. Both pieces of text
/// reference 100_000 — the pin trips when one drifts without the
/// other.
///
/// `pub(crate)` so the per-phase per-cgroup carrier builder
/// ([`crate::assert::phase_cgroup_stats`]) can re-cap the POOLED
/// wake-latency samples at the same bound: the carrier concatenates
/// every worker's already-capped vec, so the pool would otherwise grow
/// to `workers × MAX_WAKE_SAMPLES` and overrun the bulk-port frame on a
/// many-core host.
pub(crate) const MAX_WAKE_SAMPLES: usize = 100_000;

/// Per-worker cap for the PER-PHASE wake-latency reservoir carried in
/// [`super::spawn::PhaseSlice::wake_latencies_ns`]. Bounds the WIRE size
/// of each phase slice (a backdrop worker that spans many phases ships
/// one slice per phase, so the per-phase buffer is capped well below the
/// whole-run [`MAX_WAKE_SAMPLES`]); the host re-caps the merged
/// per-cgroup pool at [`MAX_WAKE_SAMPLES`] when pooling slices in
/// [`crate::assert::pool_phase_slice_stats`] (which folds via
/// `PhaseCgroupStats::merge`).
pub(crate) const MAX_PHASE_WAKE_SAMPLES: usize = 8192;

/// Twin wake-latency reservoir: a whole-run buffer (capped at
/// [`MAX_WAKE_SAMPLES`]) plus a per-phase buffer (capped at
/// [`MAX_PHASE_WAKE_SAMPLES`]) drained and reset at each phase boundary.
/// Every wake event records into both via [`Self::push`];
/// [`Self::drain_phase`] takes the per-phase samples for a
/// [`super::spawn::PhaseSlice`] while leaving the whole-run buffer
/// intact. Both reservoirs sample independently (Vitter Algorithm R) at
/// their own caps.
pub(super) struct WakeRec {
    run: Vec<u64>,
    run_total: u64,
    phase: Vec<u64>,
    phase_total: u64,
}

impl WakeRec {
    fn new() -> Self {
        WakeRec {
            run: Vec::with_capacity(MAX_WAKE_SAMPLES),
            run_total: 0,
            phase: Vec::with_capacity(MAX_PHASE_WAKE_SAMPLES),
            phase_total: 0,
        }
    }

    /// Record one wake-latency sample into both the whole-run and
    /// per-phase reservoirs.
    fn push(&mut self, sample: u64) {
        reservoir_push(&mut self.run, &mut self.run_total, sample, MAX_WAKE_SAMPLES);
        reservoir_push(
            &mut self.phase,
            &mut self.phase_total,
            sample,
            MAX_PHASE_WAKE_SAMPLES,
        );
    }

    /// Take the per-phase reservoir (samples + observed total) for a
    /// completed phase and reset it for the next phase; the whole-run
    /// reservoir is untouched.
    fn drain_phase(&mut self) -> (Vec<u64>, u64) {
        (
            std::mem::take(&mut self.phase),
            std::mem::replace(&mut self.phase_total, 0),
        )
    }
}

/// Build a [`PhaseSlice`] from a completed phase's start/end snapshots and
/// per-phase gauges. Pure — no clock or `/proc` reads — so the per-phase
/// delta math is unit-testable in isolation. Every counter field is a
/// saturating `end - start` delta; `off_cpu_ns` is `wall_ns - cpu_delta`
/// (the thread-CPU delta, clamped at 0), independent of the schedstat
/// `sum_exec_runtime` delta carried in `schedstat_cpu_time_ns`; `max_gap_ms`
/// is the ns peak floored to ms; the gauges (`cpus_used`, `numa_pages`, the
/// `(max_gap_ms, max_gap_cpu)` pair, the wake reservoir) pass through. Both
/// the per-transition drain and the final end-of-run drain call this, so the
/// two finalize paths cannot diverge.
#[allow(clippy::too_many_arguments)]
fn build_phase_slice(
    phase_epoch: u32,
    wall_ns: u64,
    cpu_start_ns: u64,
    cpu_end_ns: u64,
    schedstat_start: Option<(u64, u64, u64)>,
    schedstat_end: Option<(u64, u64, u64)>,
    vmstat_migrated_start: u64,
    vmstat_migrated_end: u64,
    iterations_delta: u64,
    migration_count: u64,
    max_gap_ns: u64,
    max_gap_cpu: usize,
    cpus_used: std::collections::BTreeSet<usize>,
    numa_pages: std::collections::BTreeMap<usize, u64>,
    wake: (Vec<u64>, u64),
    timer: (Vec<u64>, u64),
) -> PhaseSlice {
    let cpu_delta = cpu_end_ns.saturating_sub(cpu_start_ns);
    let (run_delay_ns, schedstat_cpu_time_ns) = match (schedstat_start, schedstat_end) {
        (Some((cpu_s, delay_s, _)), Some((cpu_e, delay_e, _))) => {
            (delay_e.saturating_sub(delay_s), cpu_e.saturating_sub(cpu_s))
        }
        _ => (0, 0),
    };
    PhaseSlice {
        phase_epoch,
        cpus_used,
        wake_latencies_ns: wake.0,
        wake_sample_total: wake.1,
        timer_latencies_ns: timer.0,
        timer_sample_total: timer.1,
        run_delay_ns,
        off_cpu_ns: wall_ns.saturating_sub(cpu_delta),
        wall_ns,
        migration_count,
        iterations: iterations_delta,
        schedstat_cpu_time_ns,
        numa_pages,
        vmstat_numa_pages_migrated: vmstat_migrated_end.saturating_sub(vmstat_migrated_start),
        max_gap_ms: max_gap_ns / 1_000_000,
        max_gap_cpu,
        // Generic (non-schbench) drain path: the schbench engine emits its own
        // PhaseSlices directly in the WorkType::Schbench arm.
        schbench: None,
        taobench: None,
    }
}

/// Wrap `FUTEX_WAKE` on `futex_ptr`, waking up to `n_waiters` tasks.
/// Thin wrapper around `libc::syscall(SYS_futex, ...)` — callers of the
/// wake path duplicate the 7-arg layout in every spot otherwise.
///
/// # Safety
/// Clamp a `usize` wake-count to the positive `i32` range before
/// passing to `futex_wake`.
///
/// `FUTEX_WAKE`'s `val` argument is `i32`. A naked `usize → i32`
/// cast wraps to a negative value when the input exceeds `i32::MAX`
/// (~2.1B), and some kernels interpret a negative `val` as "wake
/// every waiter on this futex" — a silent scope explosion from a
/// numeric-overflow bug. The clamp pins the syscall to wake at most
/// `i32::MAX` waiters, which exceeds any realistic topology by
/// orders of magnitude.
///
/// `#[inline]` because the call site is a single cast + `min` and
/// inlining lets the compiler fold the clamp into the surrounding
/// futex_wake syscall setup.
#[inline]
fn clamp_futex_wake_n(n: usize) -> i32 {
    n.min(i32::MAX as usize) as i32
}

/// `futex_ptr` must point to a live `u32` reachable by every thread
/// that might block on this futex word.
unsafe fn futex_wake(futex_ptr: *mut u32, n_waiters: i32) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex_ptr,
            libc::FUTEX_WAKE,
            n_waiters,
            std::ptr::null::<libc::timespec>(),
            std::ptr::null::<u32>(),
            0u32,
        );
    }
}

/// Wrap `FUTEX_WAIT` on `futex_ptr` with expected value `expected` and
/// the given timespec. Returns once the wait returns (wake, timeout, or
/// value mismatch) without inspecting the outcome — callers typically
/// re-check the state via `read_volatile`.
///
/// # Safety
/// `futex_ptr` must point to a live `u32` reachable by every thread
/// that might wake this futex word.
unsafe fn futex_wait(futex_ptr: *mut u32, expected: u32, ts: &libc::timespec) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex_ptr,
            libc::FUTEX_WAIT,
            expected,
            ts as *const libc::timespec,
            std::ptr::null::<u32>(),
            0u32,
        );
    }
}

// IO helpers for IoSyncWrite/IoRandRead/IoConvoy live in io.rs to
// keep this file under the per-file line budget. Re-imported via
// `use io::*;` so the dispatch arms below reference the items
// without qualification, matching the pre-split call sites.
mod io;
use io::*;

/// Derive a worker's off-CPU time from its wall-clock and CPU-time
/// totals: `off_cpu_ns = wall_time_ns - cpu_time_ns`, saturating at
/// 0. Saturating (not wrapping) because `cpu_time_ns` and
/// `wall_time_ns` come from two independent clocks
/// (`CLOCK_THREAD_CPUTIME_ID` vs the monotonic `start.elapsed()`),
/// so a momentary skew where the charged CPU time slightly exceeds
/// the measured wall time must clamp to 0 rather than wrap to a
/// near-`u64::MAX` phantom off-CPU figure. Single source of truth
/// for the `WorkerReport::off_cpu_ns` field so the report builder
/// and its tests agree on the derivation.
pub(super) fn derive_off_cpu_ns(wall_time_ns: u64, cpu_time_ns: u64) -> u64 {
    wall_time_ns.saturating_sub(cpu_time_ns)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn worker_main(
    affinity: Option<BTreeSet<usize>>,
    work_type: WorkType,
    sched_policy: SchedPolicy,
    mem_policy: MemPolicy,
    mpol_flags: MpolFlags,
    nice: Option<i32>,
    comm: Option<&str>,
    uid: Option<u32>,
    gid: Option<u32>,
    numa_node: Option<u32>,
    pipe_fds: Option<(i32, i32)>,
    futex: Option<(*mut u32, usize)>,
    iter_slot: *mut AtomicU64,
    // Single shared phase-epoch word — the SAME pointer for every worker in
    // the group. Allocated non-null for every handle; the scenario engine
    // bumps it only for BACKDROP handles (at each StepStart/StepEnd), so a
    // step-local worker observes no change and emits no PhaseSlices. A null
    // pointer is only the unspawned/default-handle fallback.
    phase_epoch: *mut std::sync::atomic::AtomicU32,
    stop: &AtomicBool,
    group_idx: usize,
) -> WorkerReport {
    // The kernel's per-task identifier is gettid(), not getpid():
    // - For fork-based workers, getpid() == gettid() because the
    //   forked child becomes a thread-group leader (tgid == pid == tid).
    // - For thread-based workers (CloneMode::Thread), every thread shares
    //   getpid() (== parent's tgid) and gettid() is what discriminates
    //   the per-task identity. Reporting gettid() in WorkerReport.tid
    //   keeps the field name accurate across both dispatch paths and
    //   matches what cgroup.threads / sched_setaffinity(tid, ...)
    //   accept.
    let tid: libc::pid_t = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };

    // Soft-fail on EPERM (no CAP_SYS_NICE) — the worker continues
    // with the inherited affinity/class so the test reports a
    // visible failure mode in WorkerReport rather than crashing
    // before any work is done. apply_nice and the per-pos
    // set_sched_policy sites later in this function follow the
    // same policy.
    let affinity_error: Option<String> = if let Some(ref cpus) = affinity {
        set_thread_affinity(tid, cpus)
            .err()
            .map(|e| format!("{e:#}"))
    } else {
        None
    };
    let _ = set_sched_policy(tid, sched_policy);
    apply_mempolicy_with_flags(&mem_policy, mpol_flags);
    if let Some(n) = nice {
        apply_nice(n);
    }
    if let Some(name) = comm {
        let c_name = std::ffi::CString::new(name).unwrap_or_default();
        unsafe { libc::prctl(libc::PR_SET_NAME, c_name.as_ptr()) };
    }
    if let Some(g) = gid {
        unsafe { libc::setresgid(g, g, g) };
    }
    if let Some(u) = uid {
        unsafe { libc::setresuid(u, u, u) };
    }
    if let Some(node) = numa_node {
        let path = format!("/sys/devices/system/node/node{node}/cpulist");
        if let Ok(cpulist) = std::fs::read_to_string(&path) {
            let mut cpus = BTreeSet::new();
            for part in cpulist.trim().split(',') {
                if let Some((lo, hi)) = part.split_once('-') {
                    if let (Ok(a), Ok(b)) = (lo.parse::<usize>(), hi.parse::<usize>()) {
                        cpus.extend(a..=b);
                    }
                } else if let Ok(c) = part.parse::<usize>() {
                    cpus.insert(c);
                }
            }
            if !cpus.is_empty() {
                let _ = set_thread_affinity(tid, &cpus);
            }
        }
    }

    let start = Instant::now();
    let mut work_units: u64 = 0;
    let mut migration_count: u64 = 0;
    let mut cpus_used = BTreeSet::new();
    let mut migrations = Vec::new();
    let mut last_cpu = sched_getcpu();
    cpus_used.insert(last_cpu);
    let mut last_iter_time = start;
    // Wall-clock gate for the per-iteration `iter_slot` publish.
    // Storing the AtomicU64 every iteration churned the cache line
    // (one cross-CPU coherence round-trip per iter) and dwarfed
    // worker work for tight variants like SpinWait. Iteration-count
    // throttling was rejected because a worker with iteration cost
    // > IT_SLOT_PUBLISH_INTERVAL / batch_size would publish less
    // often than the host expects, breaking delta-based snapshot
    // tests that assume forward progress within ~150 ms windows
    // (see tests_integration.rs's PageFaultChurn delta assertion).
    // Wall-clock gating decouples the publish cadence from
    // iteration cost. The interval is a balance: too long produces
    // stale `snapshot_iterations()` reads, too short defeats the
    // batching point. 1 ms matches the snapshot test's resolution
    // ceiling — the 100 ms / 150 ms gaps observe ~150 publishes
    // worst-case, more than enough for a non-zero delta.
    let mut last_iter_slot_publish = start;
    const IT_SLOT_PUBLISH_INTERVAL: Duration = Duration::from_millis(1);
    let mut max_gap_ns: u64 = 0;
    let mut max_gap_cpu: usize = last_cpu;
    let mut max_gap_at_ns: u64 = 0;
    // Lazily allocated per-worker cache buffer (CachePressure, CacheYield, CachePipe, FanOutCompute).
    let mut cache_pressure_buf: Option<Vec<u8>> = None;
    // Separate Vec<u64> for the matrix_multiply helper: the matrix
    // workload interprets its storage as a sequence of u64 operands,
    // and a `Vec<u8>` has only 1-byte alignment. Reinterpreting a
    // u8-backed buffer as `*mut u64` is UB regardless of buffer
    // contents. Vec<u64> gives natural 8-byte alignment from the
    // allocator.
    let mut matrix_buf: Option<Vec<u64>> = None;
    // Persistent /dev/vda fd (or tempfile fallback) for IoSyncWrite /
    // IoRandRead / IoConvoy. Opened on first iteration via
    // [`ensure_io_disk`]; the [`IoBacking`] Drop closes the file
    // and unlinks the host-side tempfile when the worker returns
    // (whether by clean exit, panic, or any other unwinding path).
    let mut io_disk: Option<IoBacking> = None;
    // Logical-block-aligned 4 KiB scratch buffer for O_DIRECT
    // pread/pwrite (IoRandRead, IoConvoy). Allocated lazily on
    // first IO iteration via [`DirectIoBuf::alloc`]; freed by
    // Drop when the worker returns. Reused across iterations so
    // the hot-path issues no per-iteration allocator calls.
    let mut io_buf: Option<DirectIoBuf> = None;
    // Per-worker xorshift PRNG state for IoRandRead / IoConvoy.
    // Seeded from `tid * GOLDEN_RATIO_64` (the same Weyl-sequence
    // golden-ratio increment glibc's `nrand48` family uses) so
    // a tid of 0 still produces a non-zero seed. Kept on the
    // stack (not heap) — pure scalar state.
    let mut io_rng: u64 = (tid as u64).wrapping_mul(GOLDEN_RATIO_64);
    if io_rng == 0 {
        io_rng = GOLDEN_RATIO_64;
    }
    // Sequential write cursor for IoConvoy. Starts at the worker's
    // stripe base (computed lazily once /dev/vda capacity is known)
    // and advances by 4 KiB per pwrite, wrapping at the stripe end.
    let mut io_seq_cursor: u64 = 0;
    let mut io_iter: u64 = 0;
    // WorkPhase::Io still uses the legacy tempfile-on-tmpfs
    // implementation (separate from IoSyncWrite / IoRandRead /
    // IoConvoy). Keep its own slot so worker cleanup is independent.
    // The [`PhaseIoTempfile`] RAII Drop unlinks the tempfile when
    // the worker returns, including on panic / unwind paths the
    // earlier manual `remove_file` could miss.
    let mut io_seq_file: Option<PhaseIoTempfile> = None;
    // PageFaultChurn: persistent anonymous mmap region and PRNG
    // state, allocated on first outer iteration and reused across
    // every subsequent iteration (`madvise(MADV_DONTNEED)` re-faults
    // pages without re-mapping). Keeping the region outside the
    // match arm lets PageFaultChurn return to the outer work loop
    // after each touches_per_cycle + spin_burst cycle. This gives
    // two distinct cadences:
    //   - The iter_slot publish in the outer `worker_main` loop
    //     fires on EVERY outer iteration (unconditional in the
    //     outer-loop tail), so host-side `snapshot_iterations`
    //     sees progress in real time.
    //   - The migration check in the outer `worker_main` loop
    //     fires every outer iteration but only triggers its body
    //     when `work_units.is_multiple_of(1024)`. With 320 units per
    //     PageFaultChurn outer iter and gcd(320, 1024) = 64, that
    //     lands every 1024/64 = 16 outer iterations (see
    //     doc/guide/src/architecture/workers.md).
    let mut page_fault_region: Option<(*mut libc::c_void, usize)> = None;
    let mut page_fault_rng_state: u64 = 0;
    // One-shot guard for per-position policy overrides (AsymmetricWaker
    // applies waker_class to pos == 0 / wakee_class to pos == 1;
    // PriorityInversion, RtStarvation, and PreemptStorm use the same
    // flag for their per-pos FIFO promotions). The override must run
    // AFTER the WorkloadConfig-supplied `set_sched_policy` above so
    // it's the last word on the worker's class, and ONCE so we don't
    // hammer sched_setattr/sched_setscheduler every outer iteration.
    // Stack-local because Thread mode would race a process-wide static
    // across multiple per-pos workers in the same group cohort.
    let mut per_pos_policy_applied = false;
    // One-shot guard for IdleChurn's `precise_timing` opt-in.
    // When set, the IdleChurn dispatch arm calls
    // `prctl(PR_SET_TIMERSLACK, 1)` once per worker — see the
    // dispatch arm's inline comment for the kernel-source
    // citation explaining why `1` (not `0`) is the value that
    // shrinks slack.
    let mut idle_churn_slack_applied = false;
    // One-shot guard for SignalStorm's `tkill` failure log. Stack-
    // local because Thread mode would race a process-wide static
    // across multiple SignalStorm worker pairs in the same process:
    // the first worker to fail would latch the static and silence
    // every sibling's distinct failure. Per-worker latching keeps
    // the log spam bounded (one warn per worker per session) while
    // preserving visibility into per-worker failure modes.
    let mut tkill_warned = false;
    // AluHot: the configured `AluWidth` is resolved to a concrete
    // arch-specific variant once at worker entry rather than on
    // every iteration. `Widest` resolves to the widest variant
    // the host supports; an explicit width that the host cannot
    // run downgrades to the next-widest available with a one-
    // shot warn. The resolved width persists for the worker's
    // lifetime.
    let mut alu_hot_resolved_width: Option<AluWidth> = None;
    // WorkPhase::AluHot: one-shot downgrade-warn dedup. Fires the
    // first time any WorkPhase::AluHot visit resolves to a width
    // different from what was requested (excluding `Widest`,
    // which is documented as "host's widest"). A single flag
    // suffices because the warn is about "any AluHot phase
    // downgraded in this worker" — the per-width identity is
    // already in the log fields. Per-visit resolution prevents
    // sharing alu_hot_resolved_width with the standalone
    // WorkType::AluHot arm, which would let a WorkPhase::AluHot
    // with width X clobber a downstream WorkPhase::AluHot with
    // width Y.
    let mut phase_alu_hot_warn_emitted: bool = false;
    // IpcVariance: persistent 512KB working-set buffer reused
    // across cold phases. Allocated lazily on the first cold
    // phase and reused thereafter. Aligned u64 storage so the
    // u64 multiplies in the hot phase share the same allocation
    // type as the cold-phase reads (avoids the alignment hazard
    // CachePressure documented for u64 reinterpretation of
    // Vec<u8>).
    let mut ipc_variance_buf: Option<Vec<u64>> = None;
    let mut ipc_variance_rng: u64 = (tid as u64).wrapping_mul(GOLDEN_RATIO_64);
    if ipc_variance_rng == 0 {
        ipc_variance_rng = GOLDEN_RATIO_64;
    }
    // Benchmarking: per-wakeup latency samples (reservoir-sampled) and iteration counter.
    let mut wake = WakeRec::new();
    // TimerLatency (cyclictest) reservoir — a distinct WakeRec instance feeding
    // the distinct timer_latencies_ns carrier, parallel to `wake`. Only the
    // WorkType::TimerLatency arm pushes into it; every other variant leaves it
    // empty (like `wake` for non-blocking variants).
    let mut timer = WakeRec::new();
    // Absolute monotonic deadline (ns) for the TimerLatency cyclictest loop:
    // 0 = unseeded (seeded one interval out on the first TimerLatency cycle),
    // then accumulated `+= interval_ns` so a late wake surfaces as latency
    // (CO-free). CLOCK_MONOTONIC ns-since-boot is never 0, so 0 is a safe
    // unseeded sentinel.
    let mut timer_next_deadline_ns: u64 = 0;
    // Per-iteration wall-clock compute duration samples
    // (reservoir-sampled at the same cap as wake_latencies_ns).
    // Populated by AluHot, SmtSiblingSpin, IpcVariance; all other
    // variants leave it empty.
    let mut iteration_costs_ns: Vec<u64> = Vec::with_capacity(MAX_WAKE_SAMPLES);
    let mut iteration_cost_sample_count: u64 = 0;
    let mut iterations: u64 = 0;
    // AffinityChurn / CrossAffinityChurn: read the effective cpuset
    // once at start via sched_getaffinity (read_effective_cpus).
    // Custom: hand the user function a WorkerCtx. Affinity and
    // sched_policy are already applied above.
    if let WorkType::Custom { name, run, cfg, .. } = &work_type {
        // The ctx exposes the same effective cpuset and cgroup-sibling
        // pids the built-in affinity-churn variants compute, so a
        // custom probe need not re-roll sched_getaffinity or
        // cgroup.procs parsing. Read both once here, before handing
        // control to the user function.
        let cpus = read_effective_cpus();
        let sibling_pids = read_sibling_pids();
        let cgroup_dir = read_self_cgroup_dir();
        let ctx = WorkerCtx::new(stop, &cpus, &sibling_pids, cgroup_dir.as_deref(), *cfg);
        let report = run.call(&ctx);
        // Telltale for the work_units-vs-iterations footgun: a closure that
        // counted work but left the outer-loop counter at zero reads as zero
        // throughput (total_iterations / the per-worker rates read
        // `iterations`). Warn once; do not mutate the verbatim report.
        if custom_iterations_telltale(&report) {
            warn_custom_iterations_zero_once(name);
        }
        return report;
    }

    let affinity_churn_cpus: Vec<usize> = if matches!(
        work_type,
        WorkType::AffinityChurn { .. } | WorkType::CrossAffinityChurn { .. }
    ) {
        read_effective_cpus()
    } else {
        Vec::new()
    };
    // CrossAffinityChurn: discover the sibling worker pids once at
    // entry — a flipper sees only the cgroup members present when it
    // starts (targets must be declared before the flipper WorkSpec;
    // apply_setup starts WorkSpecs serially in declaration order — see
    // the WorkType::CrossAffinityChurn doc). Precompute two cpuset
    // sub-masks one CPU apart, so each sched_setaffinity is a genuine
    // mask change.
    let cross_affinity_targets: Vec<libc::pid_t> =
        if matches!(work_type, WorkType::CrossAffinityChurn { .. }) {
            read_sibling_pids()
        } else {
            Vec::new()
        };
    let cross_affinity_masks: Option<(libc::cpu_set_t, libc::cpu_set_t)> =
        if matches!(work_type, WorkType::CrossAffinityChurn { .. }) {
            build_cross_affinity_masks(&affinity_churn_cpus)
        } else {
            None
        };
    let mut cross_affinity_toggle = false;
    // PolicyChurn: build list of (policy, priority) pairs to cycle through.
    // Non-RT policies always available; RT (FIFO/RR) only with CAP_SYS_NICE.
    let policy_churn_policies: Vec<(i32, i32)> =
        if matches!(work_type, WorkType::PolicyChurn { .. }) {
            let mut policies = vec![
                (libc::SCHED_OTHER, 0),
                (libc::SCHED_BATCH, 0),
                (libc::SCHED_IDLE, 0),
            ];
            let param = libc::sched_param { sched_priority: 1 };
            let ret = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
            if ret == 0 {
                // Restore to SCHED_OTHER before entering work loop.
                let normal = libc::sched_param { sched_priority: 0 };
                unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &normal) };
                policies.push((libc::SCHED_FIFO, 1));
                policies.push((libc::SCHED_RR, 1));
            }
            policies
        } else {
            Vec::new()
        };
    // FanOutCompute: pre-compute matrix dimension from cache_footprint_kib.
    let matrix_size: usize = if let WorkType::FanOutCompute {
        cache_footprint_kib,
        operations,
        ..
    } = &work_type
    {
        if *operations > 0 && *cache_footprint_kib > 0 {
            ((cache_footprint_kib * 1024 / 3 / std::mem::size_of::<u64>()) as f64).sqrt() as usize
        } else {
            0
        }
    } else {
        0
    };

    // CgroupChurn: pre-format the per-target `cgroup.procs` paths and
    // the `tid\n` write payload once at worker entry so the dispatch
    // arm avoids the per-iteration `format!()` heap allocation. The
    // file descriptors themselves are cached lazily inside the
    // dispatch arm — opening at entry and never re-opening would be
    // unsafe across an `Op::RemoveCgroup` followed by a recreate at
    // the same path: a write to the cached fd against the rmdir'd
    // kernfs node returns -ENODEV (cgroup_kn_lock_live in
    // kernel/cgroup/cgroup.c rejects a dead cgroup), and the new
    // cgroup gets a fresh inode the cached fd never observes. The
    // dispatch arm therefore opens on demand and invalidates the
    // cached fd on any write/open error, so a recreate is picked up
    // on the next iteration.
    let cgroup_churn_paths: Vec<String> = if let WorkType::CgroupChurn { groups, .. } = &work_type {
        let groups_count = (*groups).max(1);
        // Resolve the workload cgroup root the framework places test cgroups
        // under (default /sys/fs/cgroup/ktstr; per-test override via
        // `workload_root_cgroup` stamped into /workload_root_cgroup) — the same
        // resolver the host-side CgroupManager uses, so the auto-created paths
        // below and the dispatch paths agree by construction. (Previously a
        // hardcoded top-level /sys/fs/cgroup/wt-cgroup-churn-<i> sat outside
        // any managed root and was never created by the framework.)
        let root = crate::test_support::resolve_cgroup_root(&[]);
        // Auto-provision the rotation cgroups as LEAF cgroups under the root.
        // `create_cgroup` on a single-component name validates + mkdirs and
        // leaves the leaf's `cgroup.subtree_control` EMPTY (enable_subtree_cpuset
        // no-ops for <2 components), which the kernel no-internal-process
        // constraint requires: `cgroup_migrate_vet_dst` returns -EBUSY for a
        // `cgroup.procs` migration into a dst whose subtree_control is set
        // (kernel/cgroup/cgroup.c). Best-effort — a create failure (e.g. a
        // non-root host-unit context) falls through to the dispatch arm's
        // lazy-open warn-and-continue, so the variant stays observable.
        let mgr = crate::cgroup::CgroupManager::new(&root);
        for i in 0..groups_count {
            let name = churn_cgroup_name(i);
            if let Err(e) = mgr.create_cgroup(&name) {
                tracing::warn!(?e, %name, %root, "CgroupChurn: auto-create failed");
            }
        }
        churn_cgroup_procs_paths(&root, groups_count)
    } else {
        Vec::new()
    };
    let cgroup_churn_tid_bytes: Vec<u8> = if matches!(work_type, WorkType::CgroupChurn { .. }) {
        format!("{tid}\n").into_bytes()
    } else {
        Vec::new()
    };
    // One slot per target group; each slot is filled on first use
    // and cleared on write/open failure so the next iteration retries
    // the open. Sized to match `cgroup_churn_paths.len()` so the
    // dispatch arm's index-by-`target_idx` is always in-bounds.
    let mut cgroup_churn_files: Vec<Option<std::fs::File>> = if cgroup_churn_paths.is_empty() {
        Vec::new()
    } else {
        (0..cgroup_churn_paths.len()).map(|_| None).collect()
    };

    // CgroupAttachStorm: resolve the sibling `<dest>/cgroup.procs`
    // migration target once — a SIBLING of the worker's own cgroup,
    // resolved with the same helper `WorkerCtx::open_sibling_cgroup_procs`
    // uses (the shared `read_self_cgroup_dir` resolver + name validation +
    // `<parent>/<dest>/cgroup.procs`), so the built-in arm and the
    // Custom-worker affordance address the same target by construction.
    // Pre-flight its writability and, under `ReapMode::SigIgn`, install
    // `SIGCHLD = SIG_IGN` once so each forked child auto-reaps concurrent
    // with the parent's write. The path is resolved here; the dispatch arm
    // caches the fd lazily and invalidates it on error (CgroupChurn
    // posture).
    let (cgroup_attach_storm_path, cgroup_attach_storm_writable): (
        Option<std::path::PathBuf>,
        bool,
    ) = if let WorkType::CgroupAttachStorm { dest, reap } = &work_type {
        match resolve_sibling_cgroup_procs(read_self_cgroup_dir().as_deref(), dest) {
            Ok(path) => {
                // Pre-flight writability (fail-loud per no-silent-drops): an
                // unwritable dest migrates nothing and the scheduler
                // "survives" vacuously. One warn + a no-op gate keeps the
                // worker observable without panicking.
                let writable = match std::fs::OpenOptions::new().write(true).open(&path) {
                    Ok(_) => true,
                    Err(e) => {
                        tracing::warn!(
                            ?e,
                            path = %path.display(),
                            "CgroupAttachStorm: dest cgroup.procs not writable — storm will no-op"
                        );
                        false
                    }
                };
                if writable && matches!(reap, ReapMode::SigIgn) {
                    // SAFETY: signal disposition is process-local; this worker
                    // is its own process under CloneMode::Fork (Thread mode is
                    // rejected at spawn). SIG_IGN here auto-reaps only the
                    // worker's own forked children; the spawn side waitpid's
                    // the WORKER pid, which this disposition does not affect.
                    unsafe {
                        libc::signal(libc::SIGCHLD, libc::SIG_IGN);
                    }
                }
                (Some(path), writable)
            }
            Err(e) => {
                tracing::warn!(
                    ?e,
                    %dest,
                    "CgroupAttachStorm: could not resolve sibling dest cgroup — storm will no-op"
                );
                (None, false)
            }
        }
    } else {
        (None, false)
    };
    // Lazy single-dest fd cache + a reusable child-pid scratch buffer so
    // the dispatch arm allocates nothing per iteration after warm-up.
    let mut cgroup_attach_storm_file: Option<std::fs::File> = None;
    let mut cgroup_attach_storm_pid_buf = String::new();

    // NumaWorkingSetSweep: pre-compute one (nodemask, maxnode) pair
    // per entry in `target_nodes`. The dispatch arm rotates through
    // the slice with `(iterations + tid_phase) % len`; without this
    // hoist, every outer iteration re-allocates the BTreeSet and
    // the nodemask Vec inside `build_nodemask`. The masks are
    // invariant across iterations because `target_nodes` is
    // captured by reference at worker entry.
    let numa_sweep_masks: Vec<(Vec<libc::c_ulong>, libc::c_ulong)> =
        if let WorkType::NumaWorkingSetSweep {
            ref target_nodes, ..
        } = work_type
        {
            target_nodes
                .iter()
                .map(|&node| build_nodemask(&[node].into_iter().collect::<BTreeSet<usize>>()))
                .collect()
        } else {
            Vec::new()
        };

    // Guest-side /proc/vmstat: system-wide in the guest, but the VM is
    // a controlled environment with no other significant processes, so
    // the delta is attributable to this workload. Same rationale as
    // /proc/self/schedstat below. Host-side reading would require
    // accessing the guest kernel's vmstat via GuestMem or BPF.
    let vmstat_migrated_start = read_vmstat_numa_pages_migrated();

    // schedstat snapshot at work-loop start. `None` means schedstats
    // is unavailable on this kernel (CONFIG_SCHEDSTATS off / procfs
    // error); propagate that through as `None` at the end snapshot
    // and we will emit zero deltas with a one-shot stderr warning —
    // previously we could not distinguish "unavailable" from "worker
    // has run for zero ns".
    //
    // Pass `Some(tid)` so the read targets
    // `/proc/self/task/<tid>/schedstat` rather than
    // `/proc/self/schedstat`. For fork-mode workers `tid == tgid` so
    // the two paths return the same data; for thread-mode workers
    // every sibling shares `/proc/self/schedstat` (the test
    // runner's leader stats), and the per-task path is the only
    // way to read a specific thread's `task->sched_info`.
    let schedstat_start = read_schedstat(Some(tid));

    // Per-phase (backdrop) accumulator state. A backdrop worker polls the
    // shared `phase_epoch` word every outer iteration and, when it changes,
    // finalizes a `PhaseSlice` for the phase just ended and re-baselines for
    // the next (drain-on-change). Every handle shares a non-null epoch word;
    // the scenario engine bumps it only for backdrop handles, so a step-local
    // worker observes no change and emits no slices (`observed_change` stays
    // false). The null branch below is the unspawned/default-handle fallback.
    let mut cur_epoch: u32 = if phase_epoch.is_null() {
        0
    } else {
        unsafe { &*phase_epoch }.load(Ordering::Relaxed)
    };
    let mut observed_change = false;
    let mut phase_slices: Vec<PhaseSlice> = Vec::new();
    let mut phase_start = start;
    let mut phase_cpu_start = thread_cpu_time_ns();
    let mut phase_schedstat_start = schedstat_start;
    let mut phase_vmstat_migrated_start = vmstat_migrated_start;
    let mut phase_cpus_used: BTreeSet<usize> = BTreeSet::new();
    phase_cpus_used.insert(last_cpu);
    let mut phase_migration_count: u64 = 0;
    let mut phase_iterations_start: u64 = 0;
    let mut phase_max_gap_ns: u64 = 0;
    let mut phase_max_gap_cpu: usize = last_cpu;

    while !stop_requested(stop) {
        match work_type {
            WorkType::SpinWait => {
                spin_burst(&mut work_units, 1024);
                iterations += 1;
            }
            WorkType::YieldHeavy => {
                work_units = std::hint::black_box(work_units.wrapping_add(1));
                std::thread::yield_now();
                iterations += 1;
            }
            WorkType::Mixed => {
                spin_burst(&mut work_units, 1024);
                std::thread::yield_now();
                iterations += 1;
            }
            WorkType::IoSyncWrite => {
                use std::os::unix::io::AsRawFd;
                if !ensure_io_disk(&mut io_disk, libc::O_SYNC, tid) {
                    std::thread::yield_now();
                    iterations += 1;
                    continue;
                }
                let backing = io_disk.as_ref().unwrap();
                let buf = [0u8; IO_BLOCK_SIZE];
                let base = stripe_base(tid, backing.capacity_bytes);
                let stripe_size = (backing.capacity_bytes / IO_NUM_STRIPES) & !(IO_SECTOR_SIZE - 1);
                // 16 × 4 KiB = 64 KiB per iteration. Walk
                // sequentially within the stripe; wrap at stripe end
                // so a long-running worker re-writes the same
                // 64 KiB → 256 KiB region (depending on stripe size)
                // forever rather than running off the end.
                let stripe_extent = stripe_size.max(IO_BLOCK_SIZE as u64 * 16);
                let iter_off = (io_iter * IO_BLOCK_SIZE as u64 * 16) % stripe_extent;
                let fd = backing.file.as_raw_fd();
                for i in 0..16u64 {
                    let off = base + iter_off + i * IO_BLOCK_SIZE as u64;
                    // SAFETY: `buf` is a valid &[u8] of length
                    // IO_BLOCK_SIZE; `fd` is owned by `backing.file`
                    // which lives for the duration of this match arm.
                    let n = unsafe {
                        libc::pwrite(
                            fd,
                            buf.as_ptr() as *const libc::c_void,
                            IO_BLOCK_SIZE,
                            off as libc::off_t,
                        )
                    };
                    // Surface short writes / errors. A short pwrite
                    // means the device returned fewer bytes than
                    // requested (sparse-file extent boundary, throttle
                    // saturation, S_IOERR after a malformed request);
                    // a -1 return is a kernel-reported failure (EIO,
                    // ENOSPC, ...). Either condition silently drops
                    // observability about disk-IO health if not
                    // logged — the workload keeps "succeeding" while
                    // the backing path is broken.
                    if n < IO_BLOCK_SIZE as isize {
                        tracing::warn!(n, off, "IoSyncWrite short pwrite");
                    }
                    work_units = std::hint::black_box(work_units.wrapping_add(1));
                }
                let before_fsync = Instant::now();
                // SAFETY: `fd` is a valid file descriptor owned by
                // `backing.file`. fdatasync blocks until kernel-
                // level dirty-data flush completes.
                let _ = unsafe { libc::fdatasync(fd) };
                wake.push(before_fsync.elapsed().as_nanos() as u64);
                io_iter = io_iter.wrapping_add(1);
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::IoRandRead => {
                use std::os::unix::io::AsRawFd;
                if !ensure_io_disk(&mut io_disk, libc::O_DIRECT, tid) {
                    std::thread::yield_now();
                    iterations += 1;
                    continue;
                }
                if !ensure_io_buf(&mut io_buf) {
                    // OOM at allocator. Skip the IO this iteration.
                    std::thread::yield_now();
                    iterations += 1;
                    continue;
                }
                let backing = io_disk.as_ref().unwrap();
                let buf = io_buf.as_ref().unwrap();
                let off = rand_io_offset(&mut io_rng, backing.capacity_bytes);
                let fd = backing.file.as_raw_fd();
                let before_pread = Instant::now();
                // SAFETY: `buf.as_ptr()` is logical-block-
                // aligned (4 KiB allocation from the system
                // allocator with a 4 KiB align request, ≥ the
                // 512-byte virtio-blk logical block size required
                // by O_DIRECT) and large enough for IO_BLOCK_SIZE.
                // `fd` is owned and valid for the life of
                // `backing`.
                let _ = unsafe {
                    libc::pread(
                        fd,
                        buf.as_ptr() as *mut libc::c_void,
                        IO_BLOCK_SIZE,
                        off as libc::off_t,
                    )
                };
                wake.push(before_pread.elapsed().as_nanos() as u64);
                work_units = std::hint::black_box(work_units.wrapping_add(1));
                io_iter = io_iter.wrapping_add(1);
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::IoConvoy => {
                use std::os::unix::io::AsRawFd;
                let was_open = io_disk.is_some();
                if !ensure_io_disk(&mut io_disk, libc::O_DIRECT, tid) {
                    std::thread::yield_now();
                    iterations += 1;
                    continue;
                }
                if !was_open {
                    // First-open hook: initialise the per-worker
                    // sequential write cursor at the stripe base.
                    // `ensure_io_disk` doesn't surface this because
                    // only IoConvoy needs the cursor; treating it
                    // as a per-arm post-open step keeps the helper
                    // single-purpose.
                    let cap = io_disk.as_ref().unwrap().capacity_bytes;
                    io_seq_cursor = stripe_base(tid, cap);
                }
                if !ensure_io_buf(&mut io_buf) {
                    std::thread::yield_now();
                    iterations += 1;
                    continue;
                }
                let backing = io_disk.as_ref().unwrap();
                let buf = io_buf.as_ref().unwrap();
                let fd = backing.file.as_raw_fd();
                let stripe_size = (backing.capacity_bytes / IO_NUM_STRIPES) & !(IO_SECTOR_SIZE - 1);
                let stripe_extent = stripe_size.max(IO_BLOCK_SIZE as u64 * 16);
                let base = stripe_base(tid, backing.capacity_bytes);
                // Sequential pwrite at the per-worker cursor. Wrap
                // back to the stripe base when the cursor walks
                // past the stripe end so a long worker re-writes
                // its stripe forever.
                if io_seq_cursor >= base + stripe_extent {
                    io_seq_cursor = base;
                }
                let before_io = Instant::now();
                // SAFETY: `buf.as_ptr()` is the logical-block-
                // aligned 4 KiB allocation (≥ the 512-byte virtio-
                // blk logical block size required by O_DIRECT);
                // treating it as a const slice of IO_BLOCK_SIZE
                // bytes is in-bounds.
                let n = unsafe {
                    libc::pwrite(
                        fd,
                        buf.as_ptr() as *const libc::c_void,
                        IO_BLOCK_SIZE,
                        io_seq_cursor as libc::off_t,
                    )
                };
                // Surface short writes / errors. See the IoSyncWrite
                // arm for the rationale — same observability defense
                // applies to IoConvoy's pwrite half. The pread half
                // (below) does NOT get this check because short reads
                // are a normal sparse-file outcome (a hole reads zero
                // bytes EOF-style), not a defect.
                if n < IO_BLOCK_SIZE as isize {
                    tracing::warn!(n, off = io_seq_cursor, "IoConvoy short pwrite");
                }
                io_seq_cursor = io_seq_cursor.wrapping_add(IO_BLOCK_SIZE as u64);
                // Random pread.
                let r_off = rand_io_offset(&mut io_rng, backing.capacity_bytes);
                // SAFETY: same buffer, same fd; mutating the
                // 4 KiB region in-place.
                let _ = unsafe {
                    libc::pread(
                        fd,
                        buf.as_ptr() as *mut libc::c_void,
                        IO_BLOCK_SIZE,
                        r_off as libc::off_t,
                    )
                };
                // fdatasync every 16 iterations — the
                // convoy/coalescing-failure pathology cadence.
                if io_iter.is_multiple_of(16) {
                    // SAFETY: `fd` is owned and valid.
                    let _ = unsafe { libc::fdatasync(fd) };
                }
                wake.push(before_io.elapsed().as_nanos() as u64);
                work_units = std::hint::black_box(work_units.wrapping_add(2));
                io_iter = io_iter.wrapping_add(1);
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::Bursty {
                burst_duration,
                sleep_duration,
            } => {
                let burst_end = Instant::now() + burst_duration;
                while Instant::now() < burst_end && !stop_requested(stop) {
                    spin_burst(&mut work_units, 1024);
                }
                if !stop_requested(stop) {
                    let before_sleep = Instant::now();
                    std::thread::sleep(sleep_duration);
                    wake.push(before_sleep.elapsed().as_nanos() as u64);
                }
                iterations += 1;
            }
            WorkType::PipeIo { burst_iters } => {
                let (read_fd, write_fd) = pipe_fds.unwrap_or((-1, -1));
                if read_fd < 0 || write_fd < 0 {
                    break;
                }
                spin_burst(&mut work_units, burst_iters);
                pipe_exchange(read_fd, write_fd, &mut wake, stop);
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::FutexPingPong { spin_iters } => {
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let is_first = pos == 0;
                spin_burst(&mut work_units, spin_iters);
                // Worker A waits for 0, wakes partner with 1.
                // Worker B waits for 1, wakes partner with 0.
                let my_val: u32 = if is_first { 0 } else { 1 };
                let partner_val: u32 = if is_first { 1 } else { 0 };
                // Wake partner. The signal value is the token itself;
                // Relaxed matches the FanOutCompute / MutexContention
                // idiom — the futex syscall provides the kernel-side
                // cross-thread ordering, no extra user-space barrier
                // is needed for this single-word handshake.
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                atom.store(partner_val, Ordering::Relaxed);
                unsafe { futex_wake(futex_ptr, 1) };
                // Wait for partner to set our expected value, with timeout
                // to avoid blocking forever if partner has stopped.
                let before_block = Instant::now();
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                loop {
                    if stop_requested(stop) {
                        break;
                    }
                    let cur = atom.load(Ordering::Relaxed);
                    if cur == my_val {
                        wake.push(before_block.elapsed().as_nanos() as u64);
                        break;
                    }
                    unsafe { futex_wait(futex_ptr, partner_val, &FUTEX_WAIT_TIMEOUT) };
                }
                // Reset last_iter_time after blocking step
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::CachePressure { size_kib, stride } => {
                let buf = cache_pressure_buf.get_or_insert_with(|| vec![0u8; size_kib * 1024]);
                if buf.is_empty() || stride == 0 {
                    break;
                }
                cache_rmw_loop(buf, stride, 1024, &mut work_units);
                iterations += 1;
            }
            WorkType::CacheYield { size_kib, stride } => {
                let buf = cache_pressure_buf.get_or_insert_with(|| vec![0u8; size_kib * 1024]);
                if buf.is_empty() || stride == 0 {
                    break;
                }
                cache_rmw_loop(buf, stride, 1024, &mut work_units);
                let before_yield = Instant::now();
                std::thread::yield_now();
                wake.push(before_yield.elapsed().as_nanos() as u64);
                iterations += 1;
            }
            WorkType::CachePipe {
                size_kib,
                burst_iters,
            } => {
                let (read_fd, write_fd) = pipe_fds.unwrap_or((-1, -1));
                if read_fd < 0 || write_fd < 0 {
                    break;
                }
                let buf = cache_pressure_buf.get_or_insert_with(|| vec![0u8; size_kib * 1024]);
                if !buf.is_empty() {
                    cache_rmw_loop(buf, 64, burst_iters, &mut work_units);
                }
                pipe_exchange(read_fd, write_fd, &mut wake, stop);
                // Reset last_iter_time after blocking step
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::FutexFanOut {
                fan_out,
                spin_iters,
            } => {
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let is_messenger = pos == 0;
                spin_burst(&mut work_units, spin_iters);
                // Atomic-Relaxed idiom matches FanOutCompute /
                // MutexContention; futex syscalls supply the kernel-
                // side ordering for this generation-counter advance.
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                if is_messenger {
                    // Increment generation counter and wake all receivers.
                    let next = atom.load(Ordering::Relaxed).wrapping_add(1);
                    let wake_n = clamp_futex_wake_n(fan_out);
                    atom.store(next, Ordering::Relaxed);
                    unsafe { futex_wake(futex_ptr, wake_n) };
                    // Short post-wake spin to let receivers run
                    // before the next wake cycle. Routes through
                    // `spin_burst` for consistency with
                    // `WorkType::FanOutCompute`'s messenger (both
                    // use `FAN_OUT_POST_WAKE_SPIN_ITERS`) so the
                    // messenger also advances `work_units`.
                    spin_burst(&mut work_units, FAN_OUT_POST_WAKE_SPIN_ITERS);
                } else {
                    // Receiver: wait for the generation counter to advance.
                    let expected = atom.load(Ordering::Relaxed);
                    let before_block = Instant::now();
                    loop {
                        if stop_requested(stop) {
                            break;
                        }
                        let cur = atom.load(Ordering::Relaxed);
                        if cur != expected {
                            wake.push(before_block.elapsed().as_nanos() as u64);
                            break;
                        }
                        unsafe { futex_wait(futex_ptr, expected, &FUTEX_WAIT_TIMEOUT) };
                    }
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::Sequence {
                ref first,
                ref rest,
            } => {
                for phase in std::iter::once(first).chain(rest.iter()) {
                    if stop_requested(stop) {
                        break;
                    }
                    match phase {
                        WorkPhase::Spin(dur) => {
                            let end = Instant::now() + *dur;
                            while Instant::now() < end && !stop_requested(stop) {
                                spin_burst(&mut work_units, 1024);
                            }
                        }
                        WorkPhase::Sleep(dur) => {
                            let before_sleep = Instant::now();
                            std::thread::sleep(*dur);
                            wake.push(before_sleep.elapsed().as_nanos() as u64);
                            last_iter_time = Instant::now();
                        }
                        WorkPhase::Yield(dur) => {
                            let end = Instant::now() + *dur;
                            // Batch the deadline + stop check every 64
                            // yields. `yield_now()` is sub-microsecond
                            // on a healthy scheduler; 63 yields of
                            // overshoot is far below the millisecond-
                            // scale `dur` WorkPhase::Yield typically
                            // carries. The two `Instant::now()` calls
                            // around `yield_now` (`before_yield` plus
                            // the `elapsed()` in `reservoir_push`) are
                            // load-bearing for the wake-latency
                            // sample — keep them per-yield. Only the
                            // top-of-loop deadline poll batches.
                            let mut yield_counter: u64 = 0;
                            loop {
                                if yield_counter & 63 == 0
                                    && (Instant::now() >= end || stop_requested(stop))
                                {
                                    break;
                                }
                                yield_counter = yield_counter.wrapping_add(1);
                                work_units = std::hint::black_box(work_units.wrapping_add(1));
                                let before_yield = Instant::now();
                                std::thread::yield_now();
                                wake.push(before_yield.elapsed().as_nanos() as u64);
                            }
                            last_iter_time = Instant::now();
                        }
                        WorkPhase::Io(dur) => {
                            let end = Instant::now() + *dur;
                            let tf = io_seq_file.get_or_insert_with(|| {
                                let path = std::env::temp_dir()
                                    .join(format!("ktstr_seq_{tid}"))
                                    .to_string_lossy()
                                    .to_string();
                                let file = std::fs::OpenOptions::new()
                                    .write(true)
                                    .create(true)
                                    .truncate(true)
                                    .open(&path)
                                    .expect("failed to create WorkPhase::Io temp file");
                                PhaseIoTempfile { file, path }
                            });
                            let f = &mut tf.file;
                            // WorkPhase::Io drives the legacy tempfile-on-tmpfs
                            // IO scenario (separate from IoSyncWrite /
                            // IoRandRead / IoConvoy on /dev/vda). All four
                            // operations below are best-effort: the loop
                            // is bounded by `end` and `stop_requested`,
                            // and a transient ENOSPC / EIO surfaces as
                            // reduced work_units (the visible failure
                            // mode for this test) rather than aborting
                            // the worker. The set_len/seek pair resets
                            // the file pointer for the next 16-write
                            // burst — neither failure prevents the
                            // bursts from making forward progress.
                            while Instant::now() < end && !stop_requested(stop) {
                                let _ = f.set_len(0);
                                let _ = f.seek(std::io::SeekFrom::Start(0));
                                let buf = [0u8; 4096];
                                for _ in 0..16 {
                                    let _ = f.write_all(&buf);
                                    work_units = std::hint::black_box(work_units.wrapping_add(1));
                                }
                                let before_sleep = Instant::now();
                                std::thread::sleep(Duration::from_micros(100));
                                wake.push(before_sleep.elapsed().as_nanos() as u64);
                            }
                            last_iter_time = Instant::now();
                        }
                        WorkPhase::AluHot { width, duration } => {
                            // Resolve per visit. resolve_alu_width's
                            // is_x86_feature_detected! probes are cached
                            // by std after first call, so per-visit
                            // resolution is O(1). Avoids the cache
                            // collision risk that a single shared
                            // Option<AluWidth> across multiple
                            // WorkPhase::AluHot phases with different
                            // widths would create.
                            let resolved = resolve_alu_width(*width);
                            // One-shot downgrade warn — fires at most
                            // once per worker across all WorkPhase::AluHot
                            // visits, matching the standalone
                            // WorkType::AluHot arm's "warn once" shape
                            // without sharing the resolved-width
                            // cache. `Widest` is documented as
                            // "host's widest" so a downgrade from it
                            // is the contracted behaviour, not a
                            // surprise.
                            if !phase_alu_hot_warn_emitted
                                && resolved != *width
                                && !matches!(*width, AluWidth::Widest)
                            {
                                tracing::warn!(
                                    requested = ?width,
                                    resolved = ?resolved,
                                    tid,
                                    "WorkPhase::AluHot width unavailable on this host; \
                                     downgraded — see [`AluWidth`] doc for \
                                     resolution order"
                                );
                                phase_alu_hot_warn_emitted = true;
                            }
                            // Per-chain iteration-cost sampling so
                            // WorkPhase::AluHot exposes the same
                            // scheduler-preemption signal as the
                            // standalone WorkType::AluHot arm:
                            // AluHot never blocks, so the
                            // wake_latencies_ns reservoir would
                            // not capture preemption inflation
                            // here. Variance across iteration_costs_ns
                            // samples encodes the scheduler signal
                            // the "AluHot at 90% with modulation"
                            // use case wants to observe.
                            let end = Instant::now() + *duration;
                            while Instant::now() < end && !stop_requested(stop) {
                                let iter_start = Instant::now();
                                alu_hot_chain(resolved, ALU_HOT_CHAIN_STEPS, &mut work_units);
                                reservoir_push(
                                    &mut iteration_costs_ns,
                                    &mut iteration_cost_sample_count,
                                    iter_start.elapsed().as_nanos() as u64,
                                    MAX_WAKE_SAMPLES,
                                );
                            }
                            // Do NOT reset last_iter_time: AluHot is
                            // CPU-bound (same event class as WorkPhase::Spin),
                            // so a reset here would artificially hide
                            // preemption stretches crossing the phase
                            // boundary from max_gap_ms — silently
                            // diverging the metric vs. an equivalent
                            // WorkPhase::Spin Sequence. Symmetric counter
                            // bookkeeping with WorkPhase::Spin (which also
                            // never resets) keeps max_gap_ms comparable
                            // when a test swaps Spin↔AluHot.
                        }
                    }
                }
                iterations += 1;
            }
            WorkType::ForkExit => {
                let pid = unsafe { libc::fork() };
                match pid {
                    -1 => {
                        work_units = std::hint::black_box(work_units.wrapping_add(1));
                        iterations += 1;
                    }
                    0 => {
                        unsafe { libc::_exit(0) };
                    }
                    child => {
                        let mut status = 0i32;
                        // `waitpid` is a blocking primitive: the
                        // parent sleeps until the child's exit is
                        // reaped. Measuring the interval is the same
                        // "wake latency" signal the other blocking
                        // work types record (pipe read, futex wait,
                        // yield_now, nanosleep), so feed it into the
                        // reservoir on the same contract.
                        let before_wait = Instant::now();
                        unsafe { libc::waitpid(child, &mut status, 0) };
                        wake.push(before_wait.elapsed().as_nanos() as u64);
                        work_units = std::hint::black_box(work_units.wrapping_add(1));
                        iterations += 1;
                    }
                }
            }
            WorkType::NiceSweep => {
                // Determine allowed nice range. Negative nice requires
                // CAP_SYS_NICE; probe once and clamp min_nice on EPERM.
                let effective_min: i32 = {
                    static PROBED_MIN: std::sync::atomic::AtomicI32 =
                        std::sync::atomic::AtomicI32::new(i32::MIN);
                    let cached = PROBED_MIN.load(Ordering::Relaxed);
                    if cached != i32::MIN {
                        cached
                    } else {
                        let ret = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, -20) };
                        let min = if ret == -1 {
                            // EPERM — unprivileged, sweep only non-negative
                            0i32
                        } else {
                            // Succeeded — restore nice 0 and sweep full range
                            unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 0) };
                            -20i32
                        };
                        PROBED_MIN.store(min, Ordering::Relaxed);
                        min
                    }
                };
                let range = (19 - effective_min + 1) as u64;
                let nice_val = effective_min + (iterations % range) as i32;
                spin_burst(&mut work_units, 512);
                unsafe {
                    libc::setpriority(libc::PRIO_PROCESS, 0, nice_val);
                }
                let before_yield = Instant::now();
                std::thread::yield_now();
                wake.push(before_yield.elapsed().as_nanos() as u64);
                iterations += 1;
            }
            WorkType::AffinityChurn { spin_iters } => {
                spin_burst(&mut work_units, spin_iters);
                if !affinity_churn_cpus.is_empty() {
                    use rand::RngExt;
                    let idx = rand::rng().random_range(0..affinity_churn_cpus.len());
                    let target = affinity_churn_cpus[idx];
                    let mut cpu_set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
                    unsafe {
                        libc::CPU_ZERO(&mut cpu_set);
                        libc::CPU_SET(target, &mut cpu_set);
                        libc::sched_setaffinity(
                            0,
                            std::mem::size_of::<libc::cpu_set_t>(),
                            &cpu_set,
                        );
                    }
                }
                let before_yield = Instant::now();
                std::thread::yield_now();
                wake.push(before_yield.elapsed().as_nanos() as u64);
                iterations += 1;
            }
            WorkType::CrossAffinityChurn { spin_iters } => {
                spin_burst(&mut work_units, spin_iters);
                // Toggle every sibling between the two sub-masks. Each
                // call is a genuine mask change (the masks differ by
                // one CPU), so the kernel runs the full
                // set_cpus_allowed path every iteration. No-op when
                // alone (no siblings) or with a <2-CPU cpuset (no
                // distinct masks).
                if let Some((mask_a, mask_b)) = &cross_affinity_masks
                    && !cross_affinity_targets.is_empty()
                {
                    let mask: *const libc::cpu_set_t = if cross_affinity_toggle {
                        mask_a
                    } else {
                        mask_b
                    };
                    for &pid in &cross_affinity_targets {
                        unsafe {
                            libc::sched_setaffinity(
                                pid,
                                std::mem::size_of::<libc::cpu_set_t>(),
                                mask,
                            );
                        }
                    }
                    cross_affinity_toggle = !cross_affinity_toggle;
                }
                let before_yield = Instant::now();
                std::thread::yield_now();
                wake.push(before_yield.elapsed().as_nanos() as u64);
                iterations += 1;
            }
            WorkType::PolicyChurn { spin_iters } => {
                spin_burst(&mut work_units, spin_iters);
                let idx = (iterations as usize) % policy_churn_policies.len().max(1);
                let (pol, prio) = policy_churn_policies[idx];
                let param = libc::sched_param {
                    sched_priority: prio,
                };
                unsafe {
                    libc::sched_setscheduler(0, pol, &param);
                }
                let before_yield = Instant::now();
                std::thread::yield_now();
                wake.push(before_yield.elapsed().as_nanos() as u64);
                iterations += 1;
            }
            WorkType::FanOutCompute {
                fan_out,
                operations,
                sleep_usec,
                ..
            } => {
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let is_messenger = pos == 0;
                // Shared memory layout: [u64 generation @ offset 0]
                // [u64 wake_ns @ offset 8]. The mmap base is
                // page-aligned (see the futex-region MAP_ANONYMOUS
                // allocation in `WorkloadHandle::spawn`), so both
                // offsets are 8-byte aligned.
                //
                // The generation counter is u64 (not u32) to prevent
                // a wraparound-ABA bug in USER-SPACE: with a u32
                // counter the worker's snapshot `expected` could
                // match `cur` again after exactly 2^32 messenger
                // advances, causing the worker's user-space
                // `cur != expected` compare to miss the wake. u64
                // comparisons push that user-space wraparound out
                // to ~585 years at one advance per nanosecond —
                // effectively unreachable.
                //
                // The KERNEL-SIDE futex_wait still compares the low
                // 32 bits at `futex_ptr` to the `expected` u32
                // argument passed into the syscall, so a full
                // 2^32-advance race inside a single futex_wait's
                // microsecond syscall window would still cause a
                // kernel-side EAGAIN miss. That is empirically
                // unreachable (2^32 atomic RMWs in microseconds
                // requires >10^15 advances/sec — orders of
                // magnitude above any realistic sequencer rate),
                // and the 100 ms futex_wait timeout self-heals any
                // hypothetical occurrence: on timeout the outer
                // loop re-reads `cur` as u64 and the mismatch is
                // visible in user space even if the kernel missed
                // the advance. Little-endian x86_64 / aarch64
                // targets guarantee the low 4 bytes of the u64
                // live at offset 0 (enforced by a compile_error!
                // elsewhere in this file); big-endian would flip
                // the layout and is rejected at build time.
                //
                // Use Release/Acquire ordering so that when workers
                // observe the generation advance, the matching
                // wake_ns store is already visible to them.
                // `read_volatile`/`write_volatile` only defeat
                // compiler reordering; on aarch64's weak memory
                // model two independent hazards remain:
                //   (a) the messenger's two stores (wake_ns, then
                //       generation) can be reordered by the CPU so
                //       the generation advance becomes globally
                //       visible before the new wake_ns; and/or
                //   (b) the worker's wake_ns load can be
                //       speculatively issued before its generation
                //       load and satisfied from a stale cache line.
                // Either path yields a fresh generation paired with
                // a stale wake_ns and contaminates the wake-latency
                // histogram.
                let wake_ts_ptr = unsafe { (futex_ptr as *mut u8).add(8) as *mut u64 };
                let gen_atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU64) };
                let wake_atom = unsafe { &*(wake_ts_ptr as *const std::sync::atomic::AtomicU64) };
                if is_messenger {
                    // Messenger: stamp wake time, advance generation, wake workers.
                    // Advance the generation and wake the workers
                    // ONLY after a successful `wake_ns` write. An
                    // earlier draft advanced the generation
                    // unconditionally, which meant a `clock_gettime`
                    // failure would wake workers against the *prior*
                    // round's `wake_ns` — producing an inflated
                    // `now_ns - wake_ns` latency that would
                    // contaminate the p99 tail of the reservoir.
                    // Skipping the whole round (including the wake)
                    // keeps the latency histogram honest; workers
                    // stay parked on `futex_wait` with its 100 ms
                    // timeout and observe the next successful round
                    // normally. `spin_burst` still runs on this
                    // thread so the messenger keeps producing
                    // work_units.
                    if let Some(wake_ns) = clock_gettime_ns(libc::CLOCK_MONOTONIC) {
                        // Relaxed wake_ns store is fine; the subsequent
                        // Release RMW on the generation synchronises
                        // it with the worker's Acquire load.
                        wake_atom.store(wake_ns, Ordering::Relaxed);
                        // fetch_add on u64 wraps at 2^64 and is
                        // sole-writer here, so one Release RMW beats
                        // load-Relaxed + store-Release. On aarch64,
                        // AtomicU64 Release ordering is guaranteed
                        // by LLVM to lower to a release-ordered
                        // instruction — LDADDL on LSE-capable cores
                        // (Armv8.1+), or an LDXR/STLXR retry loop
                        // on pre-LSE cores where STLXR supplies the
                        // release barrier. Either way the store-
                        // release half pairs with the worker's
                        // Acquire load below.
                        gen_atom.fetch_add(1, Ordering::Release);
                        unsafe { futex_wake(futex_ptr, clamp_futex_wake_n(fan_out)) };
                    }
                    spin_burst(&mut work_units, FAN_OUT_POST_WAKE_SPIN_ITERS);
                } else {
                    // Worker: wait for generation advance, then do work.
                    // Initial snapshot can be Relaxed — it only feeds
                    // `futex_wait`'s expected-value check; the real
                    // happens-before edge is established by the
                    // Acquire load below once the generation differs.
                    // u64 snapshot compared against u64 cur so
                    // wraparound cannot create a false-negative
                    // (see region-layout comment above). futex_wait
                    // takes a u32 expected, so the low 32 bits of
                    // the u64 snapshot get truncated for the syscall
                    // only — the messenger's fetch_add changes those
                    // low bits on every increment, so futex_wait's
                    // kernel-side expected-check still fires
                    // correctly on every advance.
                    let expected = gen_atom.load(Ordering::Relaxed);
                    let expected_low = expected as u32;
                    loop {
                        if stop_requested(stop) {
                            break;
                        }
                        let cur = gen_atom.load(Ordering::Acquire);
                        if cur != expected {
                            // Skip the reservoir push entirely on
                            // `clock_gettime` failure — previously
                            // the rc was discarded and a
                            // zeroed/garbage `now_ts` was fed into
                            // `saturating_sub`, silently contaminating
                            // the wake-latency histogram with values
                            // dominated by wake_ns itself.
                            if let Some(now_ns) = clock_gettime_ns(libc::CLOCK_MONOTONIC) {
                                // Acquire load above synchronises-with
                                // the messenger's Release store, so
                                // this wake_ns load sees the value
                                // paired with `cur`.
                                let wake_ns = wake_atom.load(Ordering::Relaxed);
                                let latency = now_ns.saturating_sub(wake_ns);
                                wake.push(latency);
                            }
                            break;
                        }
                        unsafe { futex_wait(futex_ptr, expected_low, &FUTEX_WAIT_TIMEOUT) };
                    }
                    if sleep_usec > 0 && !stop_requested(stop) {
                        std::thread::sleep(Duration::from_micros(sleep_usec));
                    }
                    if matrix_size > 0 && !stop_requested(stop) {
                        let buf = matrix_buf
                            .get_or_insert_with(|| vec![0u64; 3 * matrix_size * matrix_size]);
                        for _ in 0..operations {
                            // matrix_multiply itself folds a black_box-wrapped
                            // C-region read into `work_units` as the post-loop
                            // sink (see matrix_multiply doc), so the per-call
                            // accumulator increment lives inside the helper.
                            matrix_multiply(buf, matrix_size, &mut work_units);
                            work_units = std::hint::black_box(work_units.wrapping_add(1));
                        }
                    }
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::Schbench { ref config } => {
                // Run schbench's full message/worker thread topology to
                // completion -- run() blocks until `stop` -- driven by this one
                // ktstr worker process. The engine is phase-aware: it watches
                // the shared `phase_epoch` and splits its histograms at each
                // step boundary (e.g. scx -> detached-EEVDF), returning one
                // SchbenchPhaseStats per phase. Each becomes a PhaseSlice here.
                //
                // The generic phase machinery never fires for Schbench: run()
                // blocks until stop, so this outer loop runs exactly ONCE,
                // `observed_change` stays false, and the generic drain-on-change
                // / final drain are skipped. These hand-pushed slices are the
                // only ones, so there is no double-count.
                let progress = AtomicU64::new(0);
                let pe = if phase_epoch.is_null() {
                    None
                } else {
                    // SAFETY: `phase_epoch` is the shared per-phase word the
                    // parent set up for this (backdrop) handle; valid for the
                    // worker's lifetime and only read here (shared access).
                    Some(unsafe { &*phase_epoch })
                };
                let outcome = crate::workload::schbench::run::run(config, stop, &progress, pe);
                work_units = work_units.saturating_add(outcome.whole_run.loop_count);
                for (epoch, stats) in outcome.phases {
                    phase_slices.push(PhaseSlice {
                        phase_epoch: epoch,
                        schbench: Some(stats),
                        ..Default::default()
                    });
                }
                iterations += 1;
                // run() owns the whole worker lifetime and only returns once
                // `stop` is set, so this outer loop runs exactly once. break out
                // BEFORE the generic drain block below: on fall-through it would
                // fire spuriously — `cur_epoch` is the spawn-time epoch but the
                // scenario bumped `phase_epoch` across steps during run(), so
                // `epoch_changed` would push a generic PhaseSlice AND set
                // `observed_change`, causing the post-loop final drain to push a
                // second — both polluting the per-epoch buckets alongside the
                // schbench slices above. The post-loop report build still runs.
                break;
            }
            WorkType::Taobench { ref config } => {
                // Mirror the Schbench arm: run() blocks until `stop`, is
                // phase-aware (watches `phase_epoch`, splitting its per-phase op
                // counters at each step boundary), and returns one
                // TaobenchPhaseStats per phase. The generic phase machinery never
                // fires (this loop runs exactly once); break before the generic
                // drain so it cannot push a duplicate slice.
                let progress = AtomicU64::new(0);
                let pe = if phase_epoch.is_null() {
                    None
                } else {
                    // SAFETY: as in the Schbench arm — `phase_epoch` is the
                    // shared per-phase word the parent set up for this backdrop
                    // handle; valid for the worker's lifetime, read-only here.
                    Some(unsafe { &*phase_epoch })
                };
                let outcome = crate::workload::taobench::run::run(config, stop, &progress, pe);
                work_units = work_units.saturating_add(outcome.whole_run.total_ops());
                for (epoch, stats) in outcome.phases {
                    phase_slices.push(PhaseSlice {
                        phase_epoch: epoch,
                        taobench: Some(stats),
                        ..Default::default()
                    });
                }
                iterations += 1;
                break;
            }
            WorkType::PageFaultChurn {
                region_kib,
                touches_per_cycle,
                spin_iters,
            } => {
                let (ptr, region_size) = match page_fault_region {
                    Some(p) => p,
                    None => {
                        // `region_kib * 1024` overflows usize on 32-bit
                        // targets for region_kib >= 4 MiB-equivalent;
                        // `checked_mul` returns None there and the
                        // workload exits this iteration rather than
                        // wrapping to a tiny region. Previously
                        // silent — a test author who typo'd a huge
                        // `region_kib` would see a zero-iteration
                        // worker report with no diagnostic. Surface
                        // the overflow via `tracing::warn!` with the
                        // offending `region_kib` so the configuration
                        // bug is visible in the test log; the early
                        // `break` still keeps the process honest.
                        let region_size = match region_kib.checked_mul(1024) {
                            Some(v) => v,
                            None => {
                                tracing::warn!(
                                    tid,
                                    region_kib,
                                    "PageFaultChurn region_kib * 1024 overflowed usize — worker exiting outer loop without doing page-fault work"
                                );
                                break;
                            }
                        };
                        let ptr = unsafe {
                            libc::mmap(
                                std::ptr::null_mut(),
                                region_size,
                                libc::PROT_READ | libc::PROT_WRITE,
                                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                                -1,
                                0,
                            )
                        };
                        if ptr == libc::MAP_FAILED {
                            break;
                        }
                        unsafe {
                            libc::madvise(ptr, region_size, libc::MADV_NOHUGEPAGE);
                        }
                        // xorshift64 requires a non-zero seed; OR-ing
                        // tid with 1 forces the low bit on.
                        page_fault_rng_state = (tid as u64) | 1;
                        page_fault_region = Some((ptr, region_size));
                        (ptr, region_size)
                    }
                };
                // region_kib < 4 produces region_size < 4096, so
                // `region_size / 4096` truncates to zero and the
                // `% page_count` below would panic (or UB in release
                // with panic=abort). mmap rounds up to a whole page
                // internally regardless of the requested length, so
                // the kernel actually handed us at least one page
                // of mapped memory even for a sub-page `region_kib`.
                // Clamping `page_count` to at least 1 matches that
                // physical reality: the single page gets touched
                // every iteration, preserving the churn intent
                // without introducing a panic edge.
                let page_count = (region_size / 4096).max(1);
                let xorshift64 = |state: &mut u64| -> u64 {
                    let mut x = *state;
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    *state = x;
                    x
                };
                for _ in 0..touches_per_cycle {
                    let page_idx = (xorshift64(&mut page_fault_rng_state) as usize) % page_count;
                    let page_ptr = unsafe { (ptr as *mut u8).add(page_idx * 4096) };
                    unsafe { std::ptr::write_volatile(page_ptr, 1u8) };
                    work_units = std::hint::black_box(work_units.wrapping_add(1));
                }
                unsafe {
                    libc::madvise(ptr, region_size, libc::MADV_DONTNEED);
                }
                spin_burst(&mut work_units, spin_iters);
                iterations += 1;
            }
            WorkType::MutexContention {
                hold_iters,
                work_iters,
                ..
            } => {
                // pos discarded: every contender competes equally on
                // the same futex word — no per-position differentiation.
                let (futex_ptr, _pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                spin_burst(&mut work_units, work_iters);
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                // CAS acquire: try to set 0 -> 1. On failure, FUTEX_WAIT.
                loop {
                    if stop_requested(stop) {
                        break;
                    }
                    if atom
                        .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
                        .is_ok()
                    {
                        break;
                    }
                    let before_block = Instant::now();
                    unsafe {
                        futex_wait(
                            futex_ptr,
                            1u32, /* expected value (locked) */
                            &FUTEX_WAIT_TIMEOUT,
                        )
                    };
                    wake.push(before_block.elapsed().as_nanos() as u64);
                }
                // Critical section: hold the lock.
                spin_burst(&mut work_units, hold_iters);
                // Release: atomic store with Release ordering ensures
                // critical section work is visible before the unlock.
                atom.store(0, Ordering::Release);
                unsafe { futex_wake(futex_ptr, 1) };
                last_iter_time = Instant::now();
                iterations += 1;
            }
            // Stubs for the 6 new pathology-taxonomy variants. The
            // type-system surface is wired (enum arms, factory
            // methods, name/from_name/group_size/needs_shared_mem);
            // per-variant worker_main bodies land later. Until then,
            // each variant's outer loop spins burst-iter CPU work so
            // a worker that gets dispatched (e.g. via from_name()
            // round-trip tests) still produces a non-zero work_units
            // report rather than silently looping at zero.
            WorkType::ThunderingHerd {
                waiters,
                batches,
                inter_batch_ms,
            } => {
                // Single global futex: every worker in the group
                // shares the same `futex_ptr` because
                // `worker_group_size = waiters + 1` collapses the
                // herd into one group. `pos == 0` is the waker;
                // `pos > 0` are waiters.
                //
                // Waker: increment generation, FUTEX_WAKE(INT_MAX)
                // — broadcasts to every parked waiter
                // simultaneously (`kernel/futex/waitwake.c`'s
                // `futex_wake_op` walks the bucket's plist and
                // wakes up to `nr_wake` callers). We pass
                // `i32::MAX` via `clamp_futex_wake_n(usize::MAX)`
                // so the kernel wakes everyone parked on this
                // word in a single syscall, matching the
                // thundering-herd shape.
                //
                // Waiter: park on the futex, observe generation
                // advance, record wake latency. Same idiom as
                // FutexFanOut waiter; the difference is purely the
                // group shape (single global vs per-group).
                //
                // After the configured number of batches, the
                // waker stops triggering and the waiters drain.
                // STOP from SIGUSR1 unblocks both sides via the
                // FUTEX_WAIT_TIMEOUT poll cycle.
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let is_waker = pos == 0;
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                if is_waker {
                    let mut batches_done: u64 = 0;
                    while batches_done < batches && !stop_requested(stop) {
                        // Inter-batch sleep so waiters re-park on
                        // futex before the next thundering wake.
                        // `nanosleep` blocking ALSO contributes a
                        // wake-latency sample for the waker so its
                        // report carries telemetry comparable to
                        // the waiters'.
                        if inter_batch_ms > 0 {
                            let before_sleep = Instant::now();
                            std::thread::sleep(Duration::from_millis(inter_batch_ms));
                            wake.push(before_sleep.elapsed().as_nanos() as u64);
                        }
                        // Advance generation counter and broadcast
                        // wake. Relaxed ordering matches FutexFanOut
                        // — futex syscall supplies kernel-side
                        // cross-thread ordering for the wake itself.
                        let next = atom.load(Ordering::Relaxed).wrapping_add(1);
                        atom.store(next, Ordering::Relaxed);
                        // Clamp to i32::MAX so the syscall wakes
                        // every parked waiter on the futex word.
                        unsafe { futex_wake(futex_ptr, clamp_futex_wake_n(usize::MAX)) };
                        spin_burst(&mut work_units, 256);
                        batches_done += 1;
                    }
                } else {
                    // Waiter: park, observe advance, record latency.
                    let _ = waiters; // pattern-binding only; size set at spawn time.
                    let expected = atom.load(Ordering::Relaxed);
                    let before_block = Instant::now();
                    loop {
                        if stop_requested(stop) {
                            break;
                        }
                        let cur = atom.load(Ordering::Relaxed);
                        if cur != expected {
                            wake.push(before_block.elapsed().as_nanos() as u64);
                            break;
                        }
                        unsafe { futex_wait(futex_ptr, expected, &FUTEX_WAIT_TIMEOUT) };
                    }
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::WakeChain {
                depth,
                wake: wake_mech,
                work_per_hop,
            } => {
                // Two implementations selected by `wake`:
                //
                // - [`WakeMechanism::Pipe`] (anon-pipe ring): each
                //   stage blocks on `read(read_fd, &mut [0u8; 1], 1)`,
                //   does its CPU burst, then `write(write_fd,
                //   &[0u8; 1], 1)` to wake the next stage. The
                //   kernel routes the write through
                //   `anon_pipe_write` (fs/pipe.c:431-601) →
                //   `wake_up_interruptible_sync_poll`
                //   (include/linux/wait.h:246) →
                //   `__wake_up_sync_key`
                //   (kernel/sched/wait.c:186-193) → `WF_SYNC` is
                //   set in the wake call. Stage 0 bootstraps the
                //   ring on its first iteration with a single
                //   write so stage 1 unblocks.
                //
                // - [`WakeMechanism::Futex`] (futex-word ring):
                //   all `depth` stages share one futex word; the
                //   stage whose `pos` matches the word value is
                //   active, does its CPU burst, advances the
                //   word, and `FUTEX_WAKE(INT_MAX)` broadcasts so
                //   every other stage observes the new value and
                //   either runs or re-parks. No `WF_SYNC`.
                //
                // `worker_group_size = depth` for both paths.
                if depth == 0 {
                    break;
                }
                if matches!(wake_mech, WakeMechanism::Pipe) {
                    let (read_fd, write_fd) = match pipe_fds {
                        Some(p) => p,
                        None => break,
                    };
                    if read_fd < 0 || write_fd < 0 {
                        break;
                    }
                    // Stage 0 is the bootstrap producer on the
                    // first iteration only — it writes one byte
                    // into its own pipe so stage 1 can unblock.
                    // After that, every stage waits for its
                    // predecessor's wake before proceeding.
                    //
                    // pos is the worker's position within its
                    // chain, supplied by the spawn-side futex
                    // tuple. When `chain_pipe_depth` is Some, the
                    // spawn-side always allocates the per-group
                    // futex too, so `futex == None` here means
                    // the spawn-side broke its own invariant —
                    // bail rather than silently treating every
                    // worker as stage 0 (which would have every
                    // worker fire the bootstrap write and stall
                    // the chain).
                    let pos = match futex {
                        Some((_, p)) => p,
                        None => break,
                    };
                    if iterations == 0 && pos == 0 {
                        // Gate the bootstrap write behind a stop
                        // check. If SIGUSR1 fires during spawn,
                        // skipping this write keeps the chain
                        // dormant — every other stage is already
                        // poll-blocking with the same stop check
                        // so the chain unwinds promptly. Without
                        // the gate, a deep chain (depth=64,
                        // work_per_hop=100ms) would burn through a
                        // full ring round-trip (~6.4s) before
                        // observing the stop on its second
                        // iteration.
                        if stop_requested(stop) {
                            iterations += 1;
                            continue;
                        }
                        let one = [0u8; 1];
                        // Bootstrap write: best-effort. EAGAIN is
                        // impossible (a fresh anon pipe has full
                        // buffer); EPIPE means the successor
                        // already exited (shutdown is in-flight)
                        // and the next outer-iteration
                        // stop_requested check unwinds the chain.
                        // Either way the failure mode is "chain
                        // doesn't bootstrap this run" — surfaces
                        // as zero work_units, not a crash.
                        let _ = unsafe {
                            libc::write(write_fd, one.as_ptr() as *const libc::c_void, 1)
                        };
                    }
                    // Stop-pollable read: 100ms poll cadence so
                    // the worker re-checks `stop_requested` even
                    // when the predecessor never wakes us. Mirrors
                    // `pipe_exchange` (the PipeIo/CachePipe wake
                    // helper) verbatim. POLLIN→read 1 byte and
                    // record the wake-latency reservoir;
                    // POLLHUP/POLLERR→break (predecessor's pipe
                    // end is closed, no more wakes will arrive).
                    let before_block = Instant::now();
                    let mut pfd = libc::pollfd {
                        fd: read_fd,
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    let mut got_byte = false;
                    loop {
                        if stop_requested(stop) {
                            break;
                        }
                        let ret = unsafe { libc::poll(&mut pfd, 1, 100) };
                        if ret > 0 {
                            let mut buf = [0u8; 1];
                            let n = unsafe {
                                libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, 1)
                            };
                            wake.push(before_block.elapsed().as_nanos() as u64);
                            if n == 1 {
                                got_byte = true;
                            }
                            break;
                        }
                        if ret < 0 {
                            break;
                        }
                    }
                    if !got_byte {
                        // Either stop fired during the poll loop
                        // or POLLHUP / poll error broke us out
                        // without delivering a byte. Both paths
                        // skip the CPU burst and successor wake;
                        // the next outer loop iteration handles
                        // teardown.
                        if stop_requested(stop) {
                            iterations += 1;
                            continue;
                        }
                        break;
                    }
                    if stop_requested(stop) {
                        iterations += 1;
                        continue;
                    }
                    let work_end = Instant::now() + work_per_hop;
                    while Instant::now() < work_end && !stop_requested(stop) {
                        spin_burst(&mut work_units, 256);
                    }
                    if stop_requested(stop) {
                        iterations += 1;
                        continue;
                    }
                    let one = [0u8; 1];
                    // Chain-advance write: same best-effort policy
                    // as the bootstrap above. EPIPE on a successor
                    // exit unwinds via the outer stop_requested
                    // check; EAGAIN cannot occur on a 1-byte write
                    // to a pipe that the predecessor's poll-then-
                    // read just drained.
                    let _ =
                        unsafe { libc::write(write_fd, one.as_ptr() as *const libc::c_void, 1) };
                    last_iter_time = Instant::now();
                    iterations += 1;
                } else {
                    let (futex_ptr, pos) = match futex {
                        Some(f) => f,
                        None => break,
                    };
                    if pos >= depth {
                        // Defense in depth: surface uses
                        // `worker_group_size = depth`, so the
                        // spawn-side divisibility check
                        // guarantees pos < depth before we get
                        // here. This branch handles only a
                        // programmer bug that bypasses spawn.
                        break;
                    }
                    let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                    let my_stage = pos as u32;
                    let next_stage = ((pos + 1) % depth) as u32;
                    let before_block = Instant::now();
                    loop {
                        if stop_requested(stop) {
                            break;
                        }
                        let cur = atom.load(Ordering::Relaxed);
                        if cur == my_stage {
                            // Our turn. Record blocked-time as
                            // a wake sample. pos == 0 on the
                            // very first iteration sees
                            // `cur == 0` immediately (never
                            // blocked) — `before_block` is
                            // post-spawn, the elapsed time still
                            // captures the spawn-to-first-stage
                            // gap, matching how FutexFanOut
                            // handles its first iteration.
                            wake.push(before_block.elapsed().as_nanos() as u64);
                            break;
                        }
                        unsafe { futex_wait(futex_ptr, cur, &FUTEX_WAIT_TIMEOUT) };
                    }
                    if stop_requested(stop) {
                        iterations += 1;
                        continue;
                    }
                    let work_end = Instant::now() + work_per_hop;
                    while Instant::now() < work_end && !stop_requested(stop) {
                        spin_burst(&mut work_units, 256);
                    }
                    if stop_requested(stop) {
                        iterations += 1;
                        continue;
                    }
                    // Advance to the next stage and wake everyone
                    // parked. Relaxed store: futex syscall provides
                    // the kernel-side cross-thread ordering for the
                    // wake event (matches FutexFanOut's idiom).
                    atom.store(next_stage, Ordering::Relaxed);
                    unsafe { futex_wake(futex_ptr, clamp_futex_wake_n(usize::MAX)) };
                    last_iter_time = Instant::now();
                    iterations += 1;
                }
            }
            WorkType::AsymmetricWaker {
                waker_class,
                wakee_class,
                burst_iters,
            } => {
                // Paired waker/wakee in different scheduling classes.
                // `worker_group_size = 2`, so pos ∈ {0, 1}: pos == 0
                // is the waker, pos == 1 is the wakee. Each holds
                // its own class for the entire run; transition
                // happens once on the first iteration.
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                if !per_pos_policy_applied {
                    let class = if pos == 0 { waker_class } else { wakee_class };
                    // Soft-fail on EPERM (no CAP_SYS_NICE) — same
                    // policy as the apply_nice / set_thread_affinity
                    // sites in worker_main: log and continue with
                    // the inherited class so the test reports
                    // visible failure mode rather than crashing.
                    let _ = set_sched_policy(0, class.to_policy());
                    per_pos_policy_applied = true;
                }
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                if pos == 0 {
                    // Waker: spin to build CPU runtime, then advance
                    // the futex word and FUTEX_WAKE the wakee. The
                    // wakee's wake_latencies_ns reservoir will
                    // capture the wake-affine placement gap on its
                    // side; the waker's reservoir is empty (no
                    // blocking syscall on this side).
                    spin_burst(&mut work_units, burst_iters);
                    let next = atom.load(Ordering::Relaxed).wrapping_add(1);
                    atom.store(next, Ordering::Relaxed);
                    unsafe { futex_wake(futex_ptr, 1) };
                } else {
                    // Wakee: park on the futex word; advance to
                    // user-space when the waker bumps it. Same
                    // observe-then-record pattern as FutexFanOut's
                    // receiver — `before_block` captures the full
                    // wait→wake→reschedule round trip.
                    let expected = atom.load(Ordering::Relaxed);
                    let before_block = Instant::now();
                    loop {
                        if stop_requested(stop) {
                            break;
                        }
                        let cur = atom.load(Ordering::Relaxed);
                        if cur != expected {
                            wake.push(before_block.elapsed().as_nanos() as u64);
                            break;
                        }
                        unsafe { futex_wait(futex_ptr, expected, &FUTEX_WAIT_TIMEOUT) };
                    }
                    // Wakee also burns CPU after wake to test
                    // wake-affine placement under load — without
                    // this the wakee re-parks immediately and the
                    // scheduler never sees concurrent demand.
                    spin_burst(&mut work_units, burst_iters);
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::PriorityInversion {
                high_count,
                medium_count,
                low_count,
                hold_iters,
                work_iters,
                pi_mode,
            } => {
                // Three priority tiers contend on one shared futex
                // word in the same group. `pos` selects tier:
                //   pos < high_count → high (top RT prio)
                //   pos < high_count + medium_count → medium (mid RT)
                //   else → low (lowest RT prio)
                // The classic inversion: `low` holds the lock,
                // `medium` runs at higher prio and preempts `low`,
                // `high` waits on the lock indefinitely.
                //
                // pi_mode:
                //   Pi   → FUTEX_LOCK_PI (rt_mutex PI boost via
                //          kernel/futex/pi.c — kernel boosts the
                //          lock holder to the waiter's prio for
                //          the duration of the hold, breaking the
                //          inversion).
                //   Plain → plain CAS + FUTEX_WAIT/WAKE — the
                //           inversion goes uncorrected.
                //
                // RT priority assignment:
                //   high   → 70  (top)
                //   medium → 50  (middle, between high and low)
                //   low    → 30  (bottom; still RT so it competes
                //                  in the rt class but loses to
                //                  medium under preemption)
                // Picked inside 1..=99 so even a loaded host with
                // an existing kernel-RT task at prio 99 (e.g.
                // migration/N) sees three distinct tiers.
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let high_end = high_count;
                let medium_end = high_count + medium_count;
                let total = high_count + medium_count + low_count;
                if pos >= total {
                    break;
                }
                let (tier_prio, is_low, is_medium) = if pos < high_end {
                    (70u32, false, false)
                } else if pos < medium_end {
                    (50u32, false, true)
                } else {
                    (30u32, true, false)
                };
                if !per_pos_policy_applied {
                    let _ = set_sched_policy(0, SchedPolicy::Fifo(tier_prio));
                    per_pos_policy_applied = true;
                }
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                if is_medium {
                    // Medium: pure CPU spin (no lock). Higher prio
                    // than `low` so it preempts the lock holder.
                    spin_burst(&mut work_units, work_iters);
                } else {
                    // High and low both contend on the lock.
                    spin_burst(&mut work_units, work_iters);
                    match pi_mode {
                        FutexLockMode::Pi => {
                            // FUTEX_LOCK_PI: kernel handles the
                            // CAS atomically, transfers ownership
                            // via the futex word's TID encoding,
                            // and applies PI boost on the holder.
                            // Returns 0 on lock-acquired, -1 on
                            // error or signal.
                            let lock_rc = unsafe {
                                libc::syscall(
                                    libc::SYS_futex,
                                    futex_ptr,
                                    libc::FUTEX_LOCK_PI,
                                    0u32, /* unused for LOCK_PI */
                                    std::ptr::null::<libc::timespec>(),
                                    std::ptr::null::<u32>(),
                                    0u32,
                                )
                            };
                            if lock_rc == 0 {
                                spin_burst(&mut work_units, hold_iters);
                                unsafe {
                                    libc::syscall(
                                        libc::SYS_futex,
                                        futex_ptr,
                                        libc::FUTEX_UNLOCK_PI,
                                        0u32,
                                        std::ptr::null::<libc::timespec>(),
                                        std::ptr::null::<u32>(),
                                        0u32,
                                    );
                                }
                            }
                        }
                        FutexLockMode::Plain => {
                            // Plain spin-then-wait: try CAS 0→1,
                            // FUTEX_WAIT on contention, hold
                            // hold_iters of spin, store 0 + wake
                            // on release. Same idiom as
                            // MutexContention's body.
                            loop {
                                if stop_requested(stop) {
                                    break;
                                }
                                if atom
                                    .compare_exchange_weak(
                                        0,
                                        1,
                                        Ordering::Acquire,
                                        Ordering::Relaxed,
                                    )
                                    .is_ok()
                                {
                                    break;
                                }
                                let before_block = Instant::now();
                                unsafe {
                                    futex_wait(futex_ptr, 1u32, &FUTEX_WAIT_TIMEOUT);
                                }
                                wake.push(before_block.elapsed().as_nanos() as u64);
                            }
                            // Hold critical section. `low` does
                            // hold_iters of spin (the inversion
                            // window); `high` does work_iters
                            // (it just wants to acquire+release).
                            let hold = if is_low { hold_iters } else { work_iters };
                            spin_burst(&mut work_units, hold);
                            atom.store(0, Ordering::Release);
                            unsafe { futex_wake(futex_ptr, 1) };
                        }
                    }
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::ProducerConsumerImbalance {
                producers,
                consumers,
                produce_rate_hz,
                consume_iters,
                queue_depth_target,
            } => {
                // MPMC ring queue in shared memory. Layout:
                //   offset 0  : head     (producer reserve idx, u64)
                //   offset 8  : tail     (consumer read idx, u64)
                //   offset 16 : prod_wake (consumers' "queue drained" futex, u32)
                //   offset 20 : cons_wake (producers' "items available" futex, u32)
                //   offset 24 : pub_head (producer publish idx, u64)
                //   offset 32 : ring[Q] of u64 slots
                // pos < producers → producer; else consumer.
                //
                // Two producer counters split the MPMC protocol:
                //
                //  - `head` (reserve): producers CAS-advance head
                //    from N to N+1 to claim slot N. Two producers
                //    that observed head=N race the CAS; only one
                //    succeeds, the other reloads and re-checks
                //    capacity. The CAS is the atomic capacity check
                //    + reservation, closing the load-then-add
                //    TOCTOU that a separate `head < tail+q` check
                //    + `fetch_add` opens.
                //
                //  - `pub_head` (publish): after writing its slot,
                //    each producer spin-waits for `pub_head ==
                //    slot_avail` (its own reservation index) and
                //    then Release-stores `slot_avail+1`. This
                //    serialises publication into reservation order
                //    — slot N is announced strictly before slot
                //    N+1 — so the Release on the publish store
                //    synchronises-with the consumer's Acquire load
                //    on `pub_head`, making the slot data
                //    happens-before the consumer's read. Without
                //    this serialisation, a producer that reserved
                //    a high slot but hasn't written it yet could
                //    see its head advance be observed by a
                //    consumer that then reads stale memory at the
                //    older tail slot.
                //
                // Consumer reads `pub_head` (not `head`) so an
                // unwritten reservation never appears. Tail uses
                // Acquire/Release for the producer side's
                // queue-not-full check.
                //
                // Producer paces with nanosleep(1s/produce_rate_hz)
                // between pushes. On full queue
                // (head - tail == Q): FUTEX_WAIT on prod_wake
                // (consumers wake it when tail advances). Producer
                // tags items with monotonic counter — content
                // opaque to the workload, only its sequencing
                // matters.
                //
                // Consumer pops one item per loop: if pub_head ==
                // tail, FUTEX_WAIT on cons_wake (producers wake it
                // after pub_head advances). Then spin consume_iters
                // of CPU.
                //
                // Imbalance: when producers * rate > consumers * /
                // (consume_iters work-time), the queue grows toward
                // Q and producers eventually block — pressure-
                // testing scheduler fairness under sustained
                // backpressure (DSQ unbounded growth in scx).
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let total = producers + consumers;
                if pos >= total || queue_depth_target == 0 {
                    break;
                }
                // Clamp matches the spawn-side
                // `futex_region_size_for` clamp so worker and
                // mmap agree on the slot count.
                let q_target_usize = std::cmp::min(queue_depth_target as usize, usize::MAX / 8 - 4);
                let q = q_target_usize as u64;
                if q == 0 {
                    break;
                }
                let base = futex_ptr as *mut u8;
                let head_atom = unsafe { &*(base as *const std::sync::atomic::AtomicU64) };
                let tail_atom = unsafe { &*(base.add(8) as *const std::sync::atomic::AtomicU64) };
                let prod_wake_ptr = unsafe { base.add(16) as *mut u32 };
                let cons_wake_ptr = unsafe { base.add(20) as *mut u32 };
                let pub_head_atom =
                    unsafe { &*(base.add(24) as *const std::sync::atomic::AtomicU64) };
                let ring_base = unsafe { base.add(32) as *mut u64 };
                if pos < producers {
                    // Producer.
                    let mut next_seq: u64 = 0;
                    let pace_ns: u64 = if produce_rate_hz == 0 {
                        0
                    } else {
                        // Per-producer rate; total rate = producers
                        // × produce_rate_hz. Avoid division by
                        // zero with the gate above.
                        1_000_000_000u64 / produce_rate_hz
                    };
                    while !stop_requested(stop) {
                        // CAS-reserve a slot. Each iteration
                        // reloads head and tail; if the queue is
                        // full, FUTEX_WAIT and re-evaluate. When
                        // capacity is available, attempt
                        // `compare_exchange_weak(head, head+1)` —
                        // success means we exclusively own slot
                        // index `head`; failure means another
                        // producer raced ahead (or a spurious
                        // weak-CAS failure) so reload and retry.
                        // The CAS is the atomic capacity-check +
                        // reservation, so two producers cannot
                        // both claim the same slot index nor
                        // overshoot tail+q.
                        let mut slot_avail: u64 = 0;
                        let mut got_slot = false;
                        loop {
                            if stop_requested(stop) {
                                break;
                            }
                            let head = head_atom.load(Ordering::Relaxed);
                            let tail = tail_atom.load(Ordering::Acquire);
                            if head.wrapping_sub(tail) >= q {
                                let prod_wake_atom = unsafe {
                                    &*(prod_wake_ptr as *const std::sync::atomic::AtomicU32)
                                };
                                let expected = prod_wake_atom.load(Ordering::Relaxed);
                                let before_block = Instant::now();
                                unsafe { futex_wait(prod_wake_ptr, expected, &FUTEX_WAIT_TIMEOUT) };
                                wake.push(before_block.elapsed().as_nanos() as u64);
                                continue;
                            }
                            // Try to reserve slot `head`. Acquire
                            // on success orders subsequent slot
                            // write after the CAS; failure path is
                            // Relaxed because we discard the loaded
                            // value and reload from the top.
                            match head_atom.compare_exchange_weak(
                                head,
                                head.wrapping_add(1),
                                Ordering::Acquire,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => {
                                    slot_avail = head;
                                    got_slot = true;
                                    break;
                                }
                                Err(_) => continue,
                            }
                        }
                        if !got_slot || stop_requested(stop) {
                            break;
                        }
                        // Write the reserved slot.
                        let slot_idx = (slot_avail % q) as usize;
                        unsafe {
                            std::ptr::write_volatile(ring_base.add(slot_idx), next_seq);
                        }
                        next_seq = next_seq.wrapping_add(1);
                        // Publish in reservation order. Spin until
                        // every prior reservation has published
                        // (pub_head == slot_avail), then Release-
                        // store slot_avail+1. The serialisation
                        // guarantees consumers observing pub_head >
                        // K see slot K's data: producer J's
                        // Acquire-load of pub_head synchronises-with
                        // producer J-1's Release-store, transitively
                        // chaining producer K's slot write
                        // happens-before producer J's pub_head
                        // Release, and the consumer's Acquire-load
                        // on pub_head synchronises-with the latest
                        // Release. The spin load MUST be Acquire,
                        // not Relaxed: a Relaxed load would break
                        // the transitive happens-before chain
                        // because producer J would not synchronise-
                        // with producer K, leaving producer K's
                        // slot write potentially invisible at
                        // consumer side.
                        while pub_head_atom.load(Ordering::Acquire) != slot_avail {
                            if stop_requested(stop) {
                                break;
                            }
                            std::hint::spin_loop();
                        }
                        if stop_requested(stop) {
                            break;
                        }
                        pub_head_atom.store(slot_avail.wrapping_add(1), Ordering::Release);
                        // Wake one consumer (advance cons_wake counter).
                        let cons_wake_atom =
                            unsafe { &*(cons_wake_ptr as *const std::sync::atomic::AtomicU32) };
                        let cur = cons_wake_atom.load(Ordering::Relaxed);
                        cons_wake_atom.store(cur.wrapping_add(1), Ordering::Relaxed);
                        unsafe { futex_wake(cons_wake_ptr, 1) };
                        work_units = std::hint::black_box(work_units.wrapping_add(1));
                        // Pace.
                        if pace_ns > 0 {
                            let ts = libc::timespec {
                                tv_sec: (pace_ns / 1_000_000_000) as libc::time_t,
                                tv_nsec: (pace_ns % 1_000_000_000) as libc::c_long,
                            };
                            unsafe {
                                libc::nanosleep(&ts, std::ptr::null_mut());
                            }
                        }
                        iterations += 1;
                    }
                } else {
                    // Consumer.
                    while !stop_requested(stop) {
                        // Block on empty queue. Same init/got
                        // pattern as the producer half so the
                        // borrow checker can prove item_idx is
                        // initialized when read. Read pub_head
                        // (publication counter), not head
                        // (reservation counter), so an in-flight
                        // unwritten reservation is invisible.
                        let mut item_idx: u64 = 0;
                        let mut got_item = false;
                        loop {
                            if stop_requested(stop) {
                                break;
                            }
                            let tail = tail_atom.load(Ordering::Relaxed);
                            let pub_head = pub_head_atom.load(Ordering::Acquire);
                            if pub_head != tail {
                                item_idx = tail;
                                got_item = true;
                                break;
                            }
                            let cons_wake_atom =
                                unsafe { &*(cons_wake_ptr as *const std::sync::atomic::AtomicU32) };
                            let expected = cons_wake_atom.load(Ordering::Relaxed);
                            let before_block = Instant::now();
                            unsafe { futex_wait(cons_wake_ptr, expected, &FUTEX_WAIT_TIMEOUT) };
                            wake.push(before_block.elapsed().as_nanos() as u64);
                        }
                        if !got_item || stop_requested(stop) {
                            break;
                        }
                        let slot_idx = (item_idx % q) as usize;
                        let _val = unsafe { std::ptr::read_volatile(ring_base.add(slot_idx)) };
                        // Advance tail with Release so producers
                        // observing tail also see we've finished
                        // reading the slot.
                        tail_atom.store(item_idx.wrapping_add(1), Ordering::Release);
                        // Wake a producer that may be blocked on full queue.
                        let prod_wake_atom =
                            unsafe { &*(prod_wake_ptr as *const std::sync::atomic::AtomicU32) };
                        let cur = prod_wake_atom.load(Ordering::Relaxed);
                        prod_wake_atom.store(cur.wrapping_add(1), Ordering::Relaxed);
                        unsafe { futex_wake(prod_wake_ptr, 1) };
                        // Burn consume_iters of CPU.
                        spin_burst(&mut work_units, consume_iters);
                        iterations += 1;
                    }
                }
                last_iter_time = Instant::now();
            }
            WorkType::RtStarvation {
                rt_workers,
                cfs_workers: _,
                rt_priority,
                burst_iters,
            } => {
                // RT workers (pos < rt_workers) run as SCHED_FIFO
                // at `rt_priority`; CFS workers (pos >= rt_workers)
                // stay on SCHED_NORMAL. Both groups spin burst_iters
                // per outer iteration. The pathology: SCHED_FIFO at
                // any priority preempts SCHED_NORMAL until the kernel's
                // RT throttling kicks in
                // (`sched_rt_period_us`/`sched_rt_runtime_us`); under
                // sched_ext switch-all, ext_sched_class loses to the
                // RT class on the same CPU because dl_sched_class >
                // rt_sched_class > ext_sched_class in the class
                // hierarchy. There is no DL server protecting ext
                // (in contrast to the DL server that throttles RT
                // for fair tasks), so an ext-managed task starves
                // until RT yields. This is the inversion.
                //
                // pos for cfs_workers is implicit (anything >=
                // rt_workers is CFS); _ binds it without warning.
                let (_, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                if !per_pos_policy_applied {
                    if pos < rt_workers {
                        // Clamp at the syscall boundary: kernel
                        // rejects priorities outside 1..=99 with
                        // EINVAL, but we soft-clamp to a sane range
                        // so a programmer typo doesn't kill the
                        // worker.
                        let prio = rt_priority.clamp(1, 99) as u32;
                        let _ = set_sched_policy(0, SchedPolicy::Fifo(prio));
                    } else {
                        let _ = set_sched_policy(0, SchedPolicy::Normal);
                    }
                    per_pos_policy_applied = true;
                }
                spin_burst(&mut work_units, burst_iters);
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::NumaWorkingSetSweep {
                region_kib,
                sweep_period_ms,
                ref target_nodes,
            } => {
                // Per-worker anonymous region, rebound to a
                // rotating NUMA node every `sweep_period_ms`. Each
                // sweep:
                //   1. mbind(MPOL_BIND, MPOL_MF_MOVE) the region to
                //      `target_nodes[(iter + phase) % len]` so the
                //      kernel migrates pages off the current node.
                //   2. Touch every page (via volatile write) so
                //      the migration triggers physical page motion
                //      rather than lazy-reservation.
                //   3. nanosleep(sweep_period_ms) before next bind.
                //
                // Empty target_nodes: no binding, just keep
                // touching the region every iteration for the
                // baseline. Single-node target_nodes: pin once
                // (effectively MPOL_BIND with no rotation).
                //
                // Per-worker phase = tid % len so the cohort
                // doesn't slam the same node simultaneously
                // (matches the "phase offset" doc on the variant).
                //
                // Region allocated lazily on first iteration via
                // `page_fault_region` to reuse the existing
                // PageFaultChurn-style mmap+free idiom (the
                // SpawnGuard does NOT clean per-worker mmaps on
                // exit because they're post-fork; the worker
                // lives until SIGUSR1 and then exits, releasing
                // the mapping).
                let region_size = match region_kib.checked_mul(1024) {
                    Some(v) => v,
                    None => {
                        tracing::warn!(
                            tid,
                            region_kib,
                            "NumaWorkingSetSweep region_kib * 1024 overflowed usize"
                        );
                        break;
                    }
                };
                let (ptr, _) = match page_fault_region {
                    Some(p) => p,
                    None => {
                        let ptr = unsafe {
                            libc::mmap(
                                std::ptr::null_mut(),
                                region_size,
                                libc::PROT_READ | libc::PROT_WRITE,
                                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                                -1,
                                0,
                            )
                        };
                        if ptr == libc::MAP_FAILED {
                            break;
                        }
                        page_fault_region = Some((ptr, region_size));
                        (ptr, region_size)
                    }
                };
                // Rotate target node based on iteration count. The
                // (mask, maxnode) pair for each entry in
                // `target_nodes` is precomputed at worker entry into
                // `numa_sweep_masks` so this hot path avoids both the
                // `[node].into_iter().collect()` BTreeSet build and
                // the `build_nodemask` Vec allocation that dominates
                // the per-iteration cost.
                if !target_nodes.is_empty() {
                    let phase = (tid as usize) % target_nodes.len();
                    let node_idx = ((iterations as usize).wrapping_add(phase)) % target_nodes.len();
                    let (mask, maxnode) = &numa_sweep_masks[node_idx];
                    // MPOL_MF_MOVE = 1 << 1 (include/uapi/linux/mempolicy.h).
                    // MPOL_BIND from libc.
                    const MPOL_MF_MOVE: libc::c_ulong = 1 << 1;
                    // Best-effort migration: a single-node host or
                    // a node missing from the system rejects with
                    // EINVAL; CAP_SYS_NICE absence rejects MOVE
                    // with EPERM. Both leave pages on their current
                    // nodes — the page-touch loop below still runs
                    // and the test surfaces "no cross-node
                    // migration observed" as its visible failure
                    // mode rather than aborting the worker.
                    let _ = unsafe {
                        libc::syscall(
                            libc::SYS_mbind,
                            ptr,
                            region_size as libc::c_ulong,
                            libc::MPOL_BIND as libc::c_ulong,
                            mask.as_ptr(),
                            *maxnode,
                            MPOL_MF_MOVE,
                        )
                    };
                }
                // Touch every page so any migration kicked off
                // by mbind actually moves a referenced page (the
                // kernel only migrates pages the process has
                // accessed). page_count clamped to 1 so a sub-page
                // region is still touched.
                let page_count = (region_size / 4096).max(1);
                for page_idx in 0..page_count {
                    let page_ptr = unsafe { (ptr as *mut u8).add(page_idx * 4096) };
                    unsafe { std::ptr::write_volatile(page_ptr, 1u8) };
                    work_units = std::hint::black_box(work_units.wrapping_add(1));
                }
                if sweep_period_ms > 0 && !stop_requested(stop) {
                    let before_sleep = Instant::now();
                    std::thread::sleep(Duration::from_millis(sweep_period_ms));
                    wake.push(before_sleep.elapsed().as_nanos() as u64);
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::CgroupChurn { groups, cycle_ms } => {
                // Rotate the worker's cgroup membership by writing
                // tid to `wt-cgroup-churn-<i>/cgroup.procs` under the
                // resolved workload cgroup root (default
                // /sys/fs/cgroup/ktstr, or the per-test
                // `workload_root_cgroup`). Drives `sched_move_task`
                // and the `scx_cgroup_move_task` ops callback. The
                // worker auto-creates these leaf cgroups at entry (see
                // the `cgroup_churn_paths` setup); if a create or open
                // fails the worker logs and continues spinning so the
                // variant is observable but does not panic on a
                // misconfigured topology.
                //
                // The path strings and the `tid\n` write payload are
                // pre-formatted at worker entry into
                // `cgroup_churn_paths` / `cgroup_churn_tid_bytes` so
                // the hot path issues no per-iteration heap
                // allocations. The fd cache (`cgroup_churn_files`) is
                // populated lazily on first use and invalidated on
                // any write/open failure: a write to a cached fd
                // whose cgroup was rmdir'd returns -ENODEV (see
                // `cgroup_kn_lock_live` in kernel/cgroup/cgroup.c
                // returning NULL on dead cgroups, surfaced through
                // `__cgroup_procs_write` as -ENODEV), and a recreate
                // at the same path yields a fresh kernfs inode the
                // cached fd never observes. Invalidate-on-error keeps
                // the worker self-healing across mid-test rmdir +
                // recreate sequences.
                let path_count = cgroup_churn_paths.len();
                if path_count == 0 {
                    let _ = groups; // pattern-binding only; pre-compute
                // is gated on this same field so emptiness here
                // means a programmer bug bypassed the gate. Skip
                // the IO step rather than panicking.
                } else {
                    let target_idx = (iterations as usize) % path_count;
                    if cgroup_churn_files[target_idx].is_none() {
                        match std::fs::OpenOptions::new()
                            .write(true)
                            .open(&cgroup_churn_paths[target_idx])
                        {
                            Ok(f) => cgroup_churn_files[target_idx] = Some(f),
                            Err(e) => {
                                tracing::warn!(
                                    ?e,
                                    path = %cgroup_churn_paths[target_idx],
                                    "CgroupChurn open failed"
                                );
                            }
                        }
                    }
                    // `take()` lifts the cached File out of the slot
                    // so the inner write borrows nothing of
                    // `cgroup_churn_files`; success replaces, failure
                    // drops (closing the fd) so the next iteration
                    // re-opens. This keeps the borrow checker happy
                    // and matches the invalidate-on-error contract.
                    if let Some(f) = cgroup_churn_files[target_idx].take() {
                        use std::io::Write;
                        // `&File` implements `Write`; the immutable
                        // borrow of `f` lives only inside this
                        // match arm.
                        match (&f).write_all(&cgroup_churn_tid_bytes) {
                            Ok(()) => {
                                cgroup_churn_files[target_idx] = Some(f);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    ?e,
                                    path = %cgroup_churn_paths[target_idx],
                                    "CgroupChurn write failed; invalidating cached fd"
                                );
                                // Dropping `f` closes the kernfs fd
                                // so the next iteration's lazy-open
                                // observes the recreated kernfs node.
                            }
                        }
                    }
                }
                if cycle_ms > 0 && !stop_requested(stop) {
                    std::thread::sleep(Duration::from_millis(cycle_ms));
                }
                spin_burst(&mut work_units, 256);
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::CgroupAttachStorm { reap, .. } => {
                // Fork a transient child and migrate it — the whole process
                // — into the sibling `<dest>/cgroup.procs` by writing its
                // pid, while the child immediately `_exit`s. The write drives
                // `cgroup_attach_task`'s threadgroup walk →
                // `TRACE_CGROUP_PATH(attach_task)` (kernel/cgroup/cgroup.c);
                // racing the child's teardown is the leader-acquire surface a
                // scheduler's cgroup-attach handler must survive. The dest
                // path is resolved at worker entry; the fd is cached lazily
                // and invalidated on error (mirrors CgroupChurn), and the
                // child pid is formatted into a reused scratch buffer so the
                // hot path allocates nothing after warm-up.
                if !cgroup_attach_storm_writable {
                    // The pre-flight warn already fired (fail-loud per
                    // no-silent-drops). No-op the storm WITHOUT bumping
                    // `work_units` so a caller can detect the vacuous
                    // survive (work_units stays 0); advance `iterations`
                    // and yield so the loop still progresses to the stop
                    // check.
                    unsafe {
                        libc::sched_yield();
                    }
                    iterations += 1;
                } else {
                    let pid = unsafe { libc::fork() };
                    match pid {
                        -1 => {
                            // EAGAIN under fork pressure — sched_yield to back
                            // off, then count the cycle. Bumps the same
                            // work_units + iterations as ForkExit's -1 arm but
                            // adds a yield, since the storm forks at a higher
                            // rate and benefits from ceding the CPU on EAGAIN.
                            unsafe {
                                libc::sched_yield();
                            }
                            work_units = std::hint::black_box(work_units.wrapping_add(1));
                            iterations += 1;
                        }
                        0 => {
                            // Child: async-signal-safe immediate exit, racing
                            // the parent's write + (SigIgn) auto-reap.
                            unsafe { libc::_exit(0) };
                        }
                        child => {
                            // Parent: write the child pid to the sibling dest
                            // `cgroup.procs` (lazy-open + invalidate on error,
                            // like CgroupChurn).
                            if cgroup_attach_storm_file.is_none()
                                && let Some(path) = &cgroup_attach_storm_path
                            {
                                match std::fs::OpenOptions::new().write(true).open(path) {
                                    Ok(f) => cgroup_attach_storm_file = Some(f),
                                    Err(e) => {
                                        tracing::warn!(
                                            ?e,
                                            path = %path.display(),
                                            "CgroupAttachStorm open failed"
                                        );
                                    }
                                }
                            }
                            if let Some(f) = cgroup_attach_storm_file.take() {
                                use std::fmt::Write as _;
                                use std::io::Write as _;
                                cgroup_attach_storm_pid_buf.clear();
                                // Infallible for `String`; reuses the buffer's
                                // retained capacity after the first iteration.
                                let _ = write!(cgroup_attach_storm_pid_buf, "{child}");
                                match (&f).write_all(cgroup_attach_storm_pid_buf.as_bytes()) {
                                    Ok(()) => {
                                        cgroup_attach_storm_file = Some(f);
                                    }
                                    Err(e) if e.raw_os_error() == Some(libc::ESRCH) => {
                                        // Expected ReapMode::SigIgn-race
                                        // outcome: the child `_exit`ed and
                                        // auto-reaped before the migrate write
                                        // resolved its pid, so the kernel
                                        // resolves the pid via
                                        // find_task_by_vpid -> NULL ->
                                        // ERR_PTR(-ESRCH) (kernel/cgroup/
                                        // cgroup.c, propagated through
                                        // __cgroup_procs_write). The dest
                                        // cgroup is still live, so KEEP the
                                        // cached fd (no re-open) and do NOT
                                        // warn — the write losing the race to
                                        // the child exit is the storm working,
                                        // not a failure. The fork+attempt is
                                        // still counted as a cycle below.
                                        cgroup_attach_storm_file = Some(f);
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            ?e,
                                            "CgroupAttachStorm write failed; invalidating cached fd"
                                        );
                                        // A genuine dead fd (e.g. -ENODEV from
                                        // an rmdir'd dest kernfs node, surfaced
                                        // by cgroup_kn_lock_live): dropping `f`
                                        // closes it so the next iteration
                                        // re-opens, picking up a recreate.
                                    }
                                }
                            }
                            match reap {
                                ReapMode::Waitpid => {
                                    // Non-racing control: block until the
                                    // child is reaped, feeding the interval
                                    // into the wake reservoir on the same
                                    // contract as ForkExit's waitpid.
                                    let mut status = 0i32;
                                    let before_wait = Instant::now();
                                    unsafe { libc::waitpid(child, &mut status, 0) };
                                    wake.push(before_wait.elapsed().as_nanos() as u64);
                                }
                                ReapMode::SigIgn => {
                                    // SIGCHLD = SIG_IGN (installed at entry)
                                    // auto-reaps the child; the write above
                                    // races its exit. Nothing to do here.
                                }
                            }
                            work_units = std::hint::black_box(work_units.wrapping_add(1));
                            iterations += 1;
                        }
                    }
                }
            }
            WorkType::SignalStorm {
                signals_per_iter,
                work_iters,
            } => {
                // Paired SIGUSR2 storm. Uses SIGUSR2 (not SIGUSR1)
                // because SIGUSR1 is the framework's Fork-mode stop
                // channel (sigusr1_handler flips STOP); a no-op SIGUSR1
                // handler installed here would clobber the stop handler
                // and leave the worker un-stoppable — it would loop
                // until the killpg/SIGKILL escalation, never returning
                // to write its report (work_units would collect as 0).
                // Each worker installs a no-op SIGUSR2 handler once and
                // exchanges its tid
                // with the partner via the per-pair futex shared
                // region (slot 0 = worker 0's tid, slot 1 = worker
                // 1's tid). Once both slots are populated, each
                // worker fires `signals_per_iter` `tkill` syscalls
                // at the partner per iteration, with a
                // `work_iters` CPU spin between bursts. Exercises
                // `signal_wake_up_state` + `sighand->siglock`.
                //
                // `tkill(tid, sig)` is used (not `kill(tid, sig)`
                // and not `tgkill(tgid, tid, sig)`) so the signal
                // targets the partner thread specifically across
                // all `CloneMode`s. `sys_tkill` calls
                // `do_tkill(0, pid, sig)` → `do_send_specific(0,
                // pid, ...)` (kernel/signal.c), which uses
                // `find_task_by_vpid(pid)` and skips the tgid
                // check (`tgid <= 0`), then delivers via
                // `do_send_sig_info(sig, info, p, PIDTYPE_PID)` —
                // the per-task pending queue, so only the
                // addressed thread can dequeue. `kill(tid, ...)`
                // resolves to the tgid-wide pending queue
                // (`PIDTYPE_TGID`) so siblings could dequeue under
                // Thread/Pcomm modes. `tgkill(self_tgid, partner,
                // ...)` returns `-ESRCH` under Fork mode because
                // each forked worker is its own tgid leader, so
                // the partner's tgid does not match
                // `self_tgid`. `tkill`'s
                // ignore-the-tgid behavior is correct for Fork,
                // Thread, and Pcomm uniformly.
                use std::sync::Once;
                use std::sync::atomic::AtomicU32;
                static SIG_HANDLER_INSTALLED: Once = Once::new();
                SIG_HANDLER_INSTALLED.call_once(|| {
                    extern "C" fn handler(_: libc::c_int) {}
                    let mut sa: libc::sigaction = unsafe { std::mem::zeroed() };
                    sa.sa_sigaction = handler as *const () as usize;
                    sa.sa_flags = libc::SA_RESTART;
                    unsafe {
                        libc::sigemptyset(&mut sa.sa_mask);
                        libc::sigaction(libc::SIGUSR2, &sa, std::ptr::null_mut());
                    }
                });
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                // SAFETY: `futex_ptr` is a stable mmap region the
                // spawn-side allocated for this group; the first 8
                // bytes hold two u32 tid slots (one per pair
                // member). The cast from `*mut u32` to `*mut
                // AtomicU32` is sound because `AtomicU32` and
                // `u32` have the same in-memory layout (atomics
                // doc).
                let slots = futex_ptr as *mut AtomicU32;
                let self_slot_idx = pos & 1;
                let partner_slot_idx = self_slot_idx ^ 1;
                unsafe {
                    (*slots.add(self_slot_idx)).store(tid as u32, Ordering::Release);
                }
                let partner_tid =
                    unsafe { (*slots.add(partner_slot_idx)).load(Ordering::Acquire) as i32 };
                if partner_tid != 0 {
                    for _ in 0..signals_per_iter {
                        // Per-iteration stop check: a hostile or
                        // simply large `signals_per_iter` would
                        // otherwise let the worker keep firing
                        // `tkill`s for the entire burst after the
                        // test timeout fires. Mirrors the Bursty
                        // arm's stop-aware inner loop above.
                        if stop_requested(stop) {
                            break;
                        }
                        // `tkill` delivers via `PIDTYPE_PID` so the
                        // signal hits the addressed thread's
                        // per-task queue, not the tgid-wide queue.
                        // Works across Fork, Thread, and Pcomm
                        // modes (see arm doc above).
                        let rc =
                            unsafe { libc::syscall(libc::SYS_tkill, partner_tid, libc::SIGUSR2) };
                        if rc == -1 {
                            let errno = std::io::Error::last_os_error().raw_os_error();
                            if !tkill_warned {
                                tracing::warn!(errno, partner_tid, "SignalStorm tkill failed");
                                tkill_warned = true;
                            }
                        }
                        work_units = std::hint::black_box(work_units.wrapping_add(1));
                    }
                }
                spin_burst(&mut work_units, work_iters);
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::PreemptStorm {
                cfs_workers: _,
                rt_burst_iters,
                rt_sleep_us,
            } => {
                // Worker 0 in the group runs as SCHED_FIFO at
                // priority 1 with a burst+nanosleep loop; workers
                // 1..=cfs_workers stay on SCHED_NORMAL and spin.
                // Each RT wake (post-nanosleep) hits
                // `wakeup_preempt` → `resched_curr` against the
                // CFS sibling on the same CPU. The
                // `per_pos_policy_applied` stack-local latch (shared
                // with AsymmetricWaker / PriorityInversion) ensures
                // the FIFO promotion runs once per worker. A
                // process-wide static would race in Thread mode where
                // multiple PreemptStorm groups share one process: the
                // first pos==0 thread to swap the static would block
                // every subsequent group's pos==0 thread from
                // applying RT, silently inerting their storm.
                let pos = match futex {
                    Some((_, p)) => p,
                    // Without the per-pos shared region we cannot
                    // distinguish RT from CFS — fall back to all
                    // CFS spinning so the variant is observable
                    // even if spawn-side wiring drops the futex.
                    None => 1,
                };
                let is_rt = pos == 0;
                if is_rt && !per_pos_policy_applied {
                    let param = libc::sched_param { sched_priority: 1 };
                    let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
                    if rc != 0 {
                        tracing::warn!(
                            errno = std::io::Error::last_os_error().raw_os_error(),
                            "PreemptStorm sched_setscheduler(FIFO) failed \
                             (need CAP_SYS_NICE / RLIMIT_RTPRIO)"
                        );
                    }
                    per_pos_policy_applied = true;
                }
                spin_burst(&mut work_units, rt_burst_iters);
                if is_rt && rt_sleep_us > 0 && !stop_requested(stop) {
                    let req = libc::timespec {
                        tv_sec: (rt_sleep_us / 1_000_000) as libc::time_t,
                        tv_nsec: ((rt_sleep_us % 1_000_000) * 1_000) as libc::c_long,
                    };
                    unsafe {
                        libc::clock_nanosleep(libc::CLOCK_MONOTONIC, 0, &req, std::ptr::null_mut());
                    }
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::EpollStorm {
                producers,
                consumers: _,
                events_per_burst,
            } => {
                // Producers `eventfd_write` in bursts; consumers
                // `epoll_wait(maxevents=1)` + read counter + spin.
                // Per-pos role: indices [0, producers) are
                // producers; the rest are consumers. The eventfd
                // and epoll fd are stored in the per-group
                // shared-memory region: u64 slot 0 = eventfd + 1,
                // u64 slot 1 = epoll fd + 1 (the +1 distinguishes
                // "not yet initialised" — value 0 — from a real
                // fd of 0). Worker pos 0 (the first producer)
                // creates them on its first iteration; siblings
                // busy-spin on the slots until they appear.
                use std::sync::atomic::AtomicU64;
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                // SAFETY: spawn-side allocates a shared region
                // sized for at least 16 bytes; reinterpreting the
                // first two u64s as `AtomicU64` is sound (same
                // layout as `u64`).
                let slots = futex_ptr as *mut AtomicU64;
                let efd_slot = unsafe { &*slots };
                let epfd_slot = unsafe { &*slots.add(1) };
                let is_producer = pos < producers;
                if pos == 0 && efd_slot.load(Ordering::Acquire) == 0 {
                    let efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };
                    let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
                    if efd >= 0 && epfd >= 0 {
                        let mut ev = libc::epoll_event {
                            events: libc::EPOLLIN as u32,
                            u64: 0,
                        };
                        unsafe {
                            libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, efd, &mut ev);
                        }
                        efd_slot.store(efd as u64 + 1, Ordering::Release);
                        epfd_slot.store(epfd as u64 + 1, Ordering::Release);
                    }
                }
                let efd_raw = efd_slot.load(Ordering::Acquire);
                let epfd_raw = epfd_slot.load(Ordering::Acquire);
                if efd_raw == 0 || epfd_raw == 0 {
                    spin_burst(&mut work_units, 256);
                } else {
                    let efd = (efd_raw - 1) as libc::c_int;
                    let epfd = (epfd_raw - 1) as libc::c_int;
                    if is_producer {
                        for _ in 0..events_per_burst {
                            let one: u64 = 1;
                            unsafe {
                                libc::write(efd, &one as *const u64 as *const libc::c_void, 8);
                            }
                            work_units = std::hint::black_box(work_units.wrapping_add(1));
                        }
                    } else {
                        let mut ev: libc::epoll_event = unsafe { std::mem::zeroed() };
                        let before_wait = Instant::now();
                        let n = unsafe { libc::epoll_wait(epfd, &mut ev, 1, 100) };
                        if n > 0 {
                            let mut buf = [0u8; 8];
                            unsafe {
                                libc::read(efd, buf.as_mut_ptr() as *mut libc::c_void, 8);
                            }
                            wake.push(before_wait.elapsed().as_nanos() as u64);
                        }
                        spin_burst(&mut work_units, 256);
                    }
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::NumaMigrationChurn { period_ms } => {
                // Read online NUMA nodes once at startup, then
                // rotate sched_setaffinity through one node's CPUs
                // per period. On hosts with one NUMA node this
                // degenerates to re-pinning to the same node.
                //
                // sysfs cpulist format is comma-separated ranges or
                // singletons (`"0-7,16-23"`); the inline parser
                // expands every range into individual CPU ids.
                fn parse_cpulist_inline(s: &str) -> Vec<usize> {
                    let mut out = Vec::new();
                    for part in s.split(',') {
                        let part = part.trim();
                        if part.is_empty() {
                            continue;
                        }
                        if let Some((lo, hi)) = part.split_once('-') {
                            if let (Ok(lo), Ok(hi)) = (lo.parse::<usize>(), hi.parse::<usize>()) {
                                for c in lo..=hi {
                                    out.push(c);
                                }
                            }
                        } else if let Ok(c) = part.parse::<usize>() {
                            out.push(c);
                        }
                    }
                    out
                }
                static NUMA_NODES: std::sync::OnceLock<Vec<Vec<usize>>> =
                    std::sync::OnceLock::new();
                let nodes = NUMA_NODES.get_or_init(|| {
                    let online = std::fs::read_to_string("/sys/devices/system/node/online")
                        .unwrap_or_default();
                    let mut node_cpus: Vec<Vec<usize>> = Vec::new();
                    for part in online.trim().split(',') {
                        if let Some((lo, hi)) = part.split_once('-') {
                            let lo: usize = lo.parse().unwrap_or(0);
                            let hi: usize = hi.parse().unwrap_or(0);
                            for n in lo..=hi {
                                if let Ok(s) = std::fs::read_to_string(format!(
                                    "/sys/devices/system/node/node{}/cpulist",
                                    n
                                )) {
                                    node_cpus.push(parse_cpulist_inline(s.trim()));
                                }
                            }
                        } else if let Ok(n) = part.parse::<usize>()
                            && let Ok(s) = std::fs::read_to_string(format!(
                                "/sys/devices/system/node/node{}/cpulist",
                                n
                            ))
                        {
                            node_cpus.push(parse_cpulist_inline(s.trim()));
                        }
                    }
                    if node_cpus.is_empty() {
                        node_cpus.push(vec![0]);
                    }
                    node_cpus
                });
                let target_node = (iterations as usize) % nodes.len();
                let cpus = &nodes[target_node];
                if !cpus.is_empty() {
                    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
                    unsafe { libc::CPU_ZERO(&mut set) };
                    for &cpu in cpus {
                        if cpu < libc::CPU_SETSIZE as usize {
                            unsafe { libc::CPU_SET(cpu, &mut set) };
                        }
                    }
                    unsafe {
                        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
                    }
                }
                if period_ms > 0 && !stop_requested(stop) {
                    std::thread::sleep(Duration::from_millis(period_ms));
                }
                spin_burst(&mut work_units, 256);
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::IdleChurn {
                burst_duration,
                sleep_duration,
                precise_timing,
            } => {
                // Per-iteration: spin for `burst_duration`, then
                // `nanosleep` for `sleep_duration`. Both fields
                // are pre-validated non-zero at spawn time — the
                // `if let WorkType::IdleChurn { ... }` block
                // inside `WorkloadHandle::spawn`'s per-group
                // setup loop bails on `Duration::ZERO` for either
                // field with an actionable diagnostic before this
                // dispatch arm runs (grep
                // `IdleChurn burst_duration must be > 0`). The
                // loop body therefore always exercises both
                // phases.
                //
                // The nanosleep dequeues the task into
                // TASK_INTERRUPTIBLE; on a CPU with no other
                // runnable tasks the scheduler picks the idle
                // class via `__pick_next_task` →
                // `pick_task_idle` (kernel/sched/idle.c:480).
                // The hrtimer expiry callback `hrtimer_wakeup`
                // fires `wake_up_process` → `try_to_wake_up`
                // and the worker re-runs.
                //
                // Stop discipline: check `stop_requested` at three
                // points — at the start of the iteration, between
                // the burst and the sleep, and after the wake.
                // The middle check ensures a stop signal observed
                // mid-iteration aborts the sleep without
                // initiating it.
                //
                // Burst gating: `Instant`-based deadline matches
                // Bursty / WakeChain. CPU-spin granularity is
                // `spin_burst(256)` so the worker checks
                // `stop_requested` and the deadline at most every
                // 256 iterations of the inner loop.
                //
                // `precise_timing`: opt-in
                // `prctl(PR_SET_TIMERSLACK, 1)` shrinks
                // `current->timer_slack_ns` from the inherited
                // 50µs default to 1ns. The kernel arm at
                // `kernel/sys.c:2645-2653` sets
                // `current->timer_slack_ns = arg2` when `arg2 >
                // 0`; passing `0` is a RESET to
                // `default_timer_slack_ns` (the inherited
                // default), so `1` is the smallest value that
                // actually narrows the slack. RT/DL tasks bypass
                // PR_SET_TIMERSLACK entirely (`if
                // (rt_or_dl_task_policy(current)) break;` at
                // `kernel/sys.c:2646`); the syscall returns 0
                // but the slack stays at 0 (forced by the
                // sched-class entry path at
                // `kernel/sched/syscalls.c:258`), so the call
                // is a harmless no-op for RT IdleChurn workers.
                //
                // The one-shot guard
                // `idle_churn_slack_applied` ensures the prctl
                // fires once per worker, not on every loop
                // iteration. Pre-loop application is impossible
                // here because `worker_main`'s outer loop
                // re-matches `work_type` every iteration; the
                // guard is the cheapest way to keep the
                // single-call discipline without restructuring
                // the dispatch.
                if precise_timing && !idle_churn_slack_applied {
                    // SAFETY: `prctl` is async-signal-unsafe
                    // per signal-safety(7) but this site runs
                    // outside any signal handler context — the
                    // worker's only async signal is SIGUSR1
                    // (stop), and the SIGUSR1 handler does not
                    // call into worker_main. PR_SET_TIMERSLACK
                    // is documented for arg2 = unsigned long;
                    // we pass 1 (smallest value that narrows
                    // slack — see kernel cite above).
                    // Return code is intentionally unused: a
                    // failure leaves the worker on the
                    // inherited slack (50µs default) which
                    // matches the `precise_timing == false`
                    // semantics, so the test still runs and
                    // produces wake-latency samples; only the
                    // slack-floor distribution differs.
                    let _ = unsafe { libc::prctl(libc::PR_SET_TIMERSLACK, 1u64) };
                    idle_churn_slack_applied = true;
                }
                let burst_end = Instant::now() + burst_duration;
                while Instant::now() < burst_end && !stop_requested(stop) {
                    spin_burst(&mut work_units, 256);
                }
                if stop_requested(stop) {
                    iterations += 1;
                    continue;
                }
                let req = libc::timespec {
                    tv_sec: sleep_duration.as_secs() as libc::time_t,
                    tv_nsec: sleep_duration.subsec_nanos() as libc::c_long,
                };
                let before_sleep = Instant::now();
                // SAFETY: `req` is a valid `timespec` populated
                // with non-negative `tv_sec` (from
                // `Duration::as_secs`, u64 → time_t cast safe on
                // all supported targets) and `tv_nsec` in
                // [0, 1_000_000_000) (from `subsec_nanos()`).
                // `rem` parameter is null because we don't
                // resume on EINTR — the SIGUSR1 handler sets STOP
                // and the post-sleep `stop_requested` check
                // exits the outer loop.
                let nanosleep_rc = unsafe { libc::nanosleep(&req, std::ptr::null_mut()) };
                // Bail on EINVAL: the timespec is malformed
                // (negative `tv_sec`, or `tv_nsec` outside
                // [0, 1e9)). Spawn-side validation guarantees
                // non-zero Durations and `subsec_nanos` is
                // always in range, so this branch is only
                // reachable if a future refactor breaks the
                // invariants. EINTR is handled by the post-sleep
                // `stop_requested` check on the next outer-loop
                // iteration; no special EINTR handling here.
                if nanosleep_rc < 0 {
                    let errno = std::io::Error::last_os_error().raw_os_error();
                    if errno == Some(libc::EINVAL) {
                        tracing::error!(
                            errno = errno,
                            "IdleChurn nanosleep returned EINVAL; bailing"
                        );
                        break;
                    }
                }
                // Isolate the scheduler-resume overhead from the
                // sleep request. `before_sleep.elapsed()` measures
                // the FULL nanosleep interval — `sleep_duration` +
                // `current->timer_slack_ns` (default 50µs) +
                // wake-path delay. Subtracting the requested
                // `sleep_duration` leaves the slack + actual
                // try_to_wake_up → on-CPU latency, which is the
                // signal a scheduler-A/B test cares about.
                //
                // saturating_sub guards against the rare case
                // where `elapsed < sleep_duration`. With rem=null,
                // do_nanosleep (kernel/time/hrtimer.c:2115-2148)
                // returns 0 when the hrtimer fires before any
                // signal lands (full sleep elapsed), otherwise
                // returns -ERESTART_RESTARTBLOCK. The syscall
                // layer then either restarts via
                // hrtimer_nanosleep_restart (absolute expiry, no
                // time loss) when the signal handler has
                // SA_RESTART set, or returns -EINTR to userspace
                // immediately at signal-delivery time when it
                // does not — the latter case can leave elapsed <
                // sleep_duration. Other triggers include
                // virtualization-layer clock skew (KVM TSC reads
                // under host frequency scaling) and
                // sub-microsecond measurement-window boundaries
                // where Instant::now's monotonic resolution
                // rounds elapsed down. Saturation produces 0
                // instead of underflowing u64, matching the
                // "no observable resume overhead" interpretation.
                let elapsed = before_sleep.elapsed();
                let resume_overhead = elapsed.saturating_sub(sleep_duration);
                wake.push(resume_overhead.as_nanos() as u64);
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::TimerLatency { interval_us } => {
                // Cyclictest absolute-deadline loop. The deadline accumulates
                // (`timer_next_deadline_ns += interval_ns`, NOT `now + interval`)
                // so a late wake surfaces AS latency instead of shifting the next
                // period out — the coordinated-omission-free measurement
                // cyclictest makes. `interval_us` is validated > 0 at spawn (a
                // zero interval would never advance the deadline). One latency
                // sample per cycle into the DISTINCT `timer` reservoir
                // (timer_latencies_ns), never the shared `wake`.
                //
                // Kernel path: clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)
                // → hrtimer_nanosleep(HRTIMER_MODE_ABS) → do_nanosleep
                // schedule()s the task; on expiry hrtimer_wakeup →
                // wake_up_process → try_to_wake_up re-runs it. The measured
                // latency is the scheduler's wake-to-on-CPU delay for a
                // timer-woken task.
                let interval_ns = interval_us.saturating_mul(1_000);
                // Seed the first absolute deadline one interval out. A failed
                // CLOCK_MONOTONIC read (warned once in clock_gettime_ns) skips
                // this cycle rather than recording a garbage latency; the outer
                // loop retries.
                if timer_next_deadline_ns == 0 {
                    match clock_gettime_ns(libc::CLOCK_MONOTONIC) {
                        Some(now_ns) => {
                            timer_next_deadline_ns = now_ns.saturating_add(interval_ns);
                        }
                        None => {
                            iterations += 1;
                            continue;
                        }
                    }
                }
                let deadline = libc::timespec {
                    tv_sec: (timer_next_deadline_ns / 1_000_000_000) as libc::time_t,
                    tv_nsec: (timer_next_deadline_ns % 1_000_000_000) as libc::c_long,
                };
                // SAFETY: `deadline` is a valid timespec (tv_nsec in [0, 1e9) by
                // construction); rmtp is null because TIMER_ABSTIME sleeps to the
                // absolute deadline (no remaining-time bookkeeping on EINTR — the
                // post-sleep stop check exits the loop).
                let rc = unsafe {
                    libc::clock_nanosleep(
                        libc::CLOCK_MONOTONIC,
                        libc::TIMER_ABSTIME,
                        &deadline,
                        std::ptr::null_mut(),
                    )
                };
                if stop_requested(stop) {
                    iterations += 1;
                    continue;
                }
                // Record latency only on a clean wake (rc == 0). A non-zero rc is
                // EINTR (a signal landed — the stop check above handles
                // shutdown); an interrupted sleep did not reach the deadline, so
                // its "latency" is meaningless and must not enter the histogram.
                if rc == 0 {
                    if let Some(wake_ns) = clock_gettime_ns(libc::CLOCK_MONOTONIC) {
                        // Floor at 0: a wake observed at/before the deadline
                        // (clock granularity) is 0 latency, never a wrapping u64
                        // subtract — the same delta>=0 discipline the schbench
                        // engine uses.
                        timer.push(wake_ns.saturating_sub(timer_next_deadline_ns));
                    }
                }
                // Accumulate the absolute deadline by exactly one interval (never
                // re-base on `now`) — the CO-free invariant.
                timer_next_deadline_ns = timer_next_deadline_ns.saturating_add(interval_ns);
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::AluHot { width } => {
                // Resolve the configured width on first iteration.
                // `Widest` picks the widest variant the host supports;
                // explicit widths that the host can't run downgrade
                // to the next-widest available with a one-shot warn.
                let resolved = *alu_hot_resolved_width.get_or_insert_with(|| {
                    let r = resolve_alu_width(width);
                    if r != width && !matches!(width, AluWidth::Widest) {
                        tracing::warn!(
                            requested = ?width,
                            resolved = ?r,
                            tid,
                            "AluHot width unavailable on this host; downgraded — \
                             see [`AluWidth`] doc for resolution order"
                        );
                    }
                    r
                });
                // ALU_HOT_CHAIN_STEPS sets the per-iteration retire
                // count for each independent multiply chain. 4
                // chains × N steps drives functional-unit pressure
                // without burning excessive time per outer loop
                // iteration — the outer loop checks `stop` between
                // iterations so shutdown latency stays bounded.
                //
                // Sample iteration cost (wall-clock duration of one
                // compute iteration, including any scheduler
                // preemption) into iteration_costs_ns: AluHot never
                // blocks, so the wake_latencies_ns reservoir would
                // not capture preemption inflation here. Variance
                // across samples encodes the scheduler signal.
                let iter_start = Instant::now();
                alu_hot_chain(resolved, ALU_HOT_CHAIN_STEPS, &mut work_units);
                reservoir_push(
                    &mut iteration_costs_ns,
                    &mut iteration_cost_sample_count,
                    iter_start.elapsed().as_nanos() as u64,
                    MAX_WAKE_SAMPLES,
                );
                iterations += 1;
            }
            WorkType::SmtSiblingSpin => {
                // Pure PAUSE-spin to contend for SMT-shared
                // execution resources with the partner worker.
                // The framework expects the test author to pin
                // both group members (group_size = 2) to the SMT
                // siblings of one physical core via
                // [`AffinityIntent::Exact`]; without that, the
                // workload degenerates to two independent
                // SpinWaits and exercises no SMT contention.
                // The variant body itself is identical to
                // SpinWait — the SMT semantics come from the
                // affinity layer, not the work loop.
                //
                // Sample iteration cost (wall-clock duration of one
                // compute iteration, including any scheduler
                // preemption) into iteration_costs_ns:
                // SmtSiblingSpin never blocks. Variance across
                // iterations encodes the SMT-contention signal —
                // sibling activity slows each spin burst measurably.
                let iter_start = Instant::now();
                spin_burst(&mut work_units, 1024);
                reservoir_push(
                    &mut iteration_costs_ns,
                    &mut iteration_cost_sample_count,
                    iter_start.elapsed().as_nanos() as u64,
                    MAX_WAKE_SAMPLES,
                );
                iterations += 1;
            }
            WorkType::IpcVariance {
                hot_iters,
                cold_iters,
                period_iters,
            } => {
                // Allocate the cold-phase working set lazily on
                // first iteration. 512 KB / 8 = 65_536 u64 slots
                // — fits in LLC on most hosts but spills to DRAM
                // on small-LLC SKUs, giving the cold phase a
                // realistic memory-bound IPC profile rather than
                // an LLC-resident degenerate version.
                let buf =
                    ipc_variance_buf.get_or_insert_with(|| vec![0u64; IPC_VARIANCE_REGION_U64]);
                // xorshift64 PRNG for cache-line offset selection.
                // Per-worker seed (tid-based) keeps workers'
                // access patterns independent.
                let xorshift64 = |state: &mut u64| -> u64 {
                    let mut x = *state;
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    *state = x;
                    x
                };
                // Sample iteration cost (wall-clock duration of one
                // compute iteration, including any scheduler
                // preemption) covering the full hot+cold loop into
                // iteration_costs_ns. IpcVariance never blocks; the
                // cost variance across iterations is the
                // load-bearing signal that distinguishes hot- vs
                // cold-phase IPC under different schedulers.
                //
                // The `completed` flag gates reservoir_push: a
                // truncated iteration (loop broke early via
                // stop_requested) is not a full hot+cold cycle
                // and would contaminate the reservoir with a
                // shorter-than-real sample.
                let iter_start = Instant::now();
                let mut completed = true;
                for _ in 0..period_iters {
                    if stop_requested(stop) {
                        completed = false;
                        break;
                    }
                    // Hot phase: `hot_iters` of independent
                    // multiplies. Same shape as the AluHot Scalar
                    // path so cross-variant comparisons of pure
                    // ALU phases are direct.
                    alu_hot_chain(AluWidth::Scalar, hot_iters, &mut work_units);
                    if stop_requested(stop) {
                        completed = false;
                        break;
                    }
                    // Cold phase: `cold_iters` random cache-line
                    // reads. `black_box` on the loaded value
                    // defeats dead-load elimination so the memory
                    // traffic is real, and the wrapping_add(1)
                    // store-back keeps the touched lines dirty
                    // across cold phases — a future scheduler
                    // that profiles memory-stall behaviour can
                    // see a rising dirty-line count. No `unsafe`
                    // is needed: `Vec::<u64>` indexing is the
                    // canonical safe path, and `black_box`
                    // suffices to keep the load observable.
                    let len = buf.len();
                    if len > 0 {
                        for _ in 0..cold_iters {
                            let idx = (xorshift64(&mut ipc_variance_rng) as usize) % len;
                            let cur = std::hint::black_box(buf[idx]);
                            buf[idx] = cur.wrapping_add(1);
                            work_units = std::hint::black_box(work_units.wrapping_add(1));
                        }
                    }
                }
                if completed {
                    reservoir_push(
                        &mut iteration_costs_ns,
                        &mut iteration_cost_sample_count,
                        iter_start.elapsed().as_nanos() as u64,
                        MAX_WAKE_SAMPLES,
                    );
                }
                iterations += 1;
            }
            WorkType::Custom { .. } => unreachable!("handled by early return"),
        }

        // Publish iteration count to shared memory for host-side
        // sampling, gated by IT_SLOT_PUBLISH_INTERVAL so the
        // AtomicU64 store doesn't churn the cache line every
        // iteration on tight variants (SpinWait at ~ns/iter
        // produced one cross-CPU coherence round-trip per spin
        // cycle). SAFETY: alignment + atomic-only-access invariant
        // established at the iter_counters mmap site in
        // `WorkloadHandle::spawn` and carried by the
        // `*mut AtomicU64` type.
        //
        // `now_for_gate` is computed at most once per outer
        // iteration and reused by the work_units%1024 gap
        // accounting below. Computing it lazily (only when at least
        // one consumer needs it) keeps the per-iter clock reads at
        // 1 in the worst case (down from 2 in a naïve "compute
        // both gates eagerly" design).
        let mut now_for_gate: Option<Instant> = None;
        if !iter_slot.is_null() {
            let now = *now_for_gate.get_or_insert_with(Instant::now);
            if now.duration_since(last_iter_slot_publish) >= IT_SLOT_PUBLISH_INTERVAL {
                // Relaxed store: the parent reads this counter via
                // `snapshot_iterations()` with Relaxed ordering only
                // for progress-sampling — no cross-field
                // happens-before edge is required (see that
                // function's ordering rationale).
                unsafe { &*iter_slot }.store(iterations, Ordering::Relaxed);
                last_iter_slot_publish = now;
            }
        }

        // Poll the shared phase epoch every outer iteration (one cheap
        // Relaxed load, amortized over the work-type body) so a blocked or
        // low-throughput backdrop worker that does NOT advance work_units
        // past the next 1024 checkpoint still detects the transition and
        // finalizes its PhaseSlice promptly — otherwise the phase's
        // wall/cpu/iteration deltas, and any inter-step-gap work, would leak
        // into the adjacent slice. The 1024 checkpoint still drives the
        // gap/migration/cpus_used gauges; firing the block on a transition
        // adds one more (more-accurate) gap/migration sample at the boundary.
        let epoch_changed =
            !phase_epoch.is_null() && unsafe { &*phase_epoch }.load(Ordering::Relaxed) != cur_epoch;

        if work_units.is_multiple_of(1024) || epoch_changed {
            let now = *now_for_gate.get_or_insert_with(Instant::now);
            let gap = now.duration_since(last_iter_time).as_nanos() as u64;
            if gap > max_gap_ns {
                max_gap_ns = gap;
                max_gap_cpu = last_cpu;
                max_gap_at_ns = now.duration_since(start).as_nanos() as u64;
            }
            // Per-phase max gap, argmax-paired with the pre-migration CPU
            // exactly like the whole-run pair above (a separate threshold
            // since the per-phase peak resets at each boundary).
            if gap > phase_max_gap_ns {
                phase_max_gap_ns = gap;
                phase_max_gap_cpu = last_cpu;
            }
            last_iter_time = now;

            let cpu = sched_getcpu();
            phase_cpus_used.insert(cpu);
            if cpu != last_cpu {
                migration_count += 1;
                phase_migration_count += 1;
                cpus_used.insert(cpu);
                migrations.push(Migration {
                    at_ns: now.duration_since(start).as_nanos() as u64,
                    from_cpu: last_cpu,
                    to_cpu: cpu,
                });
                last_cpu = cpu;
            }

            // Drain-on-change: when the parent advances `phase_epoch`,
            // finalize the `PhaseSlice` for the phase just ended (counter
            // deltas vs the per-phase baselines; `cpus_used`/`numa_pages`
            // gauges) and re-baseline for the next phase. Gap- and
            // baseline-tagged epochs are emitted as-is; the host discards
            // those (see `PhaseSlice::phase_epoch`).
            if !phase_epoch.is_null() {
                let new_epoch = unsafe { &*phase_epoch }.load(Ordering::Relaxed);
                if new_epoch != cur_epoch {
                    let phase_cpu_end = thread_cpu_time_ns();
                    let phase_schedstat_end = read_schedstat(Some(tid));
                    let phase_vmstat_migrated_end = read_vmstat_numa_pages_migrated();
                    let phase_numa = read_numa_maps_pages();
                    let phase_wall_ns = now.duration_since(phase_start).as_nanos() as u64;
                    phase_slices.push(build_phase_slice(
                        cur_epoch,
                        phase_wall_ns,
                        phase_cpu_start,
                        phase_cpu_end,
                        phase_schedstat_start,
                        phase_schedstat_end,
                        phase_vmstat_migrated_start,
                        phase_vmstat_migrated_end,
                        iterations.saturating_sub(phase_iterations_start),
                        phase_migration_count,
                        phase_max_gap_ns,
                        phase_max_gap_cpu,
                        std::mem::take(&mut phase_cpus_used),
                        phase_numa,
                        wake.drain_phase(),
                        timer.drain_phase(),
                    ));
                    cur_epoch = new_epoch;
                    observed_change = true;
                    phase_start = now;
                    phase_cpu_start = phase_cpu_end;
                    phase_schedstat_start = phase_schedstat_end;
                    phase_vmstat_migrated_start = phase_vmstat_migrated_end;
                    phase_migration_count = 0;
                    phase_iterations_start = iterations;
                    phase_max_gap_ns = 0;
                    phase_max_gap_cpu = last_cpu;
                    // Seed the new phase's cpu gauge with the boundary CPU.
                    // cpus_used is a union gauge, not a counter, so the
                    // one-CPU overlap with the phase just drained (which also
                    // recorded this CPU) is intentional and bounded — both
                    // phases genuinely observed the task on it at the boundary.
                    phase_cpus_used.insert(last_cpu);
                }
            }
        }
    }

    // Reset nice to 0 so report serialization runs at default priority.
    if matches!(work_type, WorkType::NiceSweep) {
        unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 0) };
    }

    // Reset to SCHED_OTHER so report serialization runs at normal policy.
    if matches!(work_type, WorkType::PolicyChurn { .. }) {
        let param = libc::sched_param { sched_priority: 0 };
        unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &param) };
    }

    // io_seq_file (WorkPhase::Io tempfile), io_disk, and io_buf clean
    // themselves up via Drop on [`PhaseIoTempfile`] / [`IoBacking`] /
    // [`DirectIoBuf`] when this function returns: file fd closed,
    // host-side tempfile unlinked, heap buffer freed. Intentionally
    // NOT explicitly `take()`-d here so a panic between this point
    // and the function return still runs Drop.
    //
    // The persistent page_fault_region mmap is NOT freed here: its munmap is
    // deferred until after the NUMA read below, so a policy-governed
    // working-set region (NumaWorkingSetSweep) is still mapped when
    // read_numa_maps_region_pages samples it.

    // Final iteration count store for host-side sampling.
    // SAFETY: same as the iter_slot publish in the outer
    // `worker_main` loop above.
    if !iter_slot.is_null() {
        unsafe { &*iter_slot }.store(iterations, Ordering::Relaxed);
    }

    // Capture one wall instant paired with the thread-CPU sample, BEFORE the
    // /proc reads below, so both the whole-run and the final-phase off_cpu use
    // the same (wall, cpu) snapshot. Re-sampling the phase wall after the
    // /proc reads (a fresh phase_start.elapsed()) would charge their wall time
    // to the final phase's off_cpu without the matching CPU charge, diverging
    // it from the per-transition drain — which pairs wall (`now`) and
    // thread-CPU both before its own /proc reads.
    let end = Instant::now();
    let cpu_time_ns = thread_cpu_time_ns();
    let wall_time_ns = end.duration_since(start).as_nanos() as u64;

    // schedstat snapshot at work-loop end; compute deltas if both
    // snapshots succeeded, else zero (the start-of-loop read already
    // emitted a warning if schedstat is unavailable). Pair the
    // path with the start snapshot — same `tid` so the delta
    // measures the same task.
    let schedstat_end = read_schedstat(Some(tid));
    let (ss_delay_delta, ss_ts_delta, ss_cpu_delta) = match (schedstat_start, schedstat_end) {
        (Some((cpu_s, delay_s, ts_s)), Some((cpu_e, delay_e, ts_e))) => (
            delay_e.saturating_sub(delay_s),
            ts_e.saturating_sub(ts_s),
            cpu_e.saturating_sub(cpu_s),
        ),
        _ => (0, 0, 0),
    };

    // NUMA: read numa_maps and vmstat after workload. For NumaWorkingSetSweep
    // the locality metric must reflect ONLY the policy-governed working-set
    // region — whole-process numa_maps is dominated by the binary/libc/stack
    // pages COW-inherited from the parent BEFORE set_mempolicy ran, which the
    // policy never placed (set_mempolicy governs only future faults). Scope the
    // read to that region's VMA, which is still mapped because its munmap is
    // deferred to just below. Other work types report the whole-process
    // footprint as before.
    let numa_pages = match (&work_type, page_fault_region) {
        (WorkType::NumaWorkingSetSweep { .. }, Some((ptr, _))) => {
            read_numa_maps_region_pages(ptr as u64)
        }
        _ => read_numa_maps_pages(),
    };
    let vmstat_migrated_end = read_vmstat_numa_pages_migrated();
    let vmstat_migrated_delta = vmstat_migrated_end.saturating_sub(vmstat_migrated_start);

    // Free the persistent page_fault_region mmap now that the NUMA read above
    // has sampled it (NumaWorkingSetSweep scopes locality to this VMA).
    if let Some((ptr, size)) = page_fault_region {
        unsafe { libc::munmap(ptr, size) };
    }

    // Final drain: a backdrop worker that observed at least one phase
    // transition has one still-open phase with no closing epoch change.
    // Finalize it here, reusing the whole-run end snapshots computed
    // above (`cpu_time_ns`, `schedstat_end`, `numa_pages`,
    // `vmstat_migrated_end`); `numa_pages` is cloned because the report
    // below moves the original.
    if observed_change {
        let phase_wall_ns = end.duration_since(phase_start).as_nanos() as u64;
        phase_slices.push(build_phase_slice(
            cur_epoch,
            phase_wall_ns,
            phase_cpu_start,
            cpu_time_ns,
            phase_schedstat_start,
            schedstat_end,
            phase_vmstat_migrated_start,
            vmstat_migrated_end,
            iterations.saturating_sub(phase_iterations_start),
            phase_migration_count,
            phase_max_gap_ns,
            phase_max_gap_cpu,
            std::mem::take(&mut phase_cpus_used),
            numa_pages.clone(),
            wake.drain_phase(),
            timer.drain_phase(),
        ));
    }

    WorkerReport {
        tid,
        work_units,
        cpu_time_ns,
        wall_time_ns,
        off_cpu_ns: derive_off_cpu_ns(wall_time_ns, cpu_time_ns),
        migration_count,
        cpus_used,
        migrations,
        max_gap_ms: max_gap_ns / 1_000_000,
        max_gap_cpu,
        max_gap_at_ms: max_gap_at_ns / 1_000_000,
        wake_latencies_ns: wake.run,
        wake_sample_total: wake.run_total,
        iteration_costs_ns,
        iteration_cost_sample_total: iteration_cost_sample_count,
        timer_latencies_ns: timer.run,
        timer_sample_total: timer.run_total,
        iterations,
        schedstat_run_delay_ns: ss_delay_delta,
        schedstat_run_count: ss_ts_delta,
        schedstat_cpu_time_ns: ss_cpu_delta,
        completed: true,
        numa_pages,
        vmstat_numa_pages_migrated: vmstat_migrated_delta,
        // Populated by the sentinel path in `stop_and_collect`; a
        // report emitted from this (live) worker path always carries
        // `None` — the child reached the `f.write_all(&json)` site
        // and handed a complete report back to the parent.
        exit_info: None,
        // `futex` is `Some((ptr, pos))` for several work types and
        // `pos == 0` MEANS DIFFERENT THINGS PER VARIANT:
        //   - FutexFanOut / FanOutCompute: pos == 0 is the
        //     messenger — one worker per group advances the
        //     generation and fans out wakes. Exactly the shape the
        //     WorkerReport doc pins.
        //   - FutexPingPong: pos == 0 is a pair-position flag.
        //     Both workers write+wake symmetrically; neither is a
        //     messenger.
        //   - MutexContention: pos is unused (every contender
        //     competes equally on the same word).
        //   - ThunderingHerd: pos == 0 is the waker; pos > 0 are
        //     waiters parked on the futex. Not a messenger in the
        //     fan-out sense — the waker doesn't carry per-message
        //     state, just kicks the herd.
        //   - WakeChain: pos is the stage index in the chain ring.
        //     The active stage rotates each iteration, so no single
        //     worker is "the messenger" across the run.
        // Gate on the WorkType so only the fanout variants
        // propagate `pos == 0` as `is_messenger`; every other work
        // type lands `false` as the field doc contract requires.
        is_messenger: matches!(
            work_type,
            WorkType::FutexFanOut { .. } | WorkType::FanOutCompute { .. }
        ) && futex.map(|(_, p)| p == 0).unwrap_or(false),
        group_idx,
        affinity_error,
        phase_slices,
    }
}

// =====================================================================
// Workload primitives — DO NOT remove the "weird-looking" constructs
// =====================================================================
//
// The functions below (`spin_burst`, `cache_rmw_loop`,
// `matrix_multiply`, the per-WorkType inline loops in `worker_main`)
// are the kernels of every workload primitive ktstr exposes. They
// look like trivial loops but carry MULTIPLE LAYERS of optimization-
// elimination defenses that a casual reader (or a future maintainer
// running clippy with cleanup intent) might be tempted to remove
// as "redundant ceremony". Each layer is load-bearing:
//
// 1. **`std::hint::black_box(value)`** — a value-elimination
//    barrier. Routing `wrapping_add(1)` results, multiplicand
//    loads, and accumulator updates through `black_box` prevents
//    LLVM from constant-folding, partial-evaluating, or
//    algebraically simplifying the expressions. WITHOUT this,
//    `for _ in 0..count { x = x.wrapping_add(1) }` collapses to
//    `x += count` at `-O2`, defeating the per-iteration timing
//    granularity these workloads need to drive scheduler events.
//
// 2. **`ptr::read_volatile` / `ptr::write_volatile`** — a memory-
//    operation-elimination barrier. `black_box` keeps a value live,
//    but a sufficiently smart pass can still prove the BACKING
//    LOAD/STORE dead and synthesize the bytes from thin air. The
//    workloads' cache-pressure variants depend on actual L1/L2/LLC
//    line traffic — a process-local `Vec<u8>` whose contents no
//    external observer reads is otherwise DCE-eligible. Volatile
//    operations are not eliminable: every access becomes a real
//    `mov` against the actual memory slot.
//
// 3. **Real syscalls** (`futex`, `pipe`, `read`, `write`,
//    `nanosleep`, `sched_yield`, `mmap`, etc.) — opaque to LLVM by
//    construction. The optimizer cannot reason across the
//    user-kernel boundary, so syscall sites act as natural barriers
//    that force surrounding values to materialize. WorkTypes that
//    need scheduler events (`FutexFanOut`, `IoSyncWrite`,
//    `WorkPhase::Yield`)
//    rely on this implicit barrier in addition to the explicit
//    `black_box` / volatile pairs above.
//
// 4. **`#[inline(never)]`** on the workload helpers (`spin_burst`,
//    `cache_rmw_loop`, `matrix_multiply`) — keeps each call a
//    distinct boundary in the IR. Without it, inlining can fuse
//    per-iteration `black_box` increments with the caller's
//    arithmetic, defeating the granularity defense.
//
// Backend assumption: these barriers assume the LLVM backend
// (rustc default for every release toolchain). On the cranelift
// backend, `black_box` is a pure no-op identity function —
// `rustc_codegen_cranelift/src/intrinsics/mod.rs` carries a
// literal `FIXME implement black_box semantics` and just writes
// the value back unchanged. Any build with
// `-Z codegen-backend=cranelift` would silently lose every
// `black_box` barrier in this file. Volatile loads/stores and
// real syscalls survive that backend swap, so the cache-pressure
// and PageFaultChurn variants stay anchored, but every
// `spin_burst` / `matrix_multiply` / `work_units` increment
// would become DCE-eligible. Stick with the LLVM backend for
// release / nextest / `cargo ktstr test` runs.
//
// Future maintainers: if you see code like
// `*work_units = std::hint::black_box(work_units.wrapping_add(1));`
// or `unsafe { ptr::read_volatile(&buf[idx]) }` and your reflex is
// "this can be simplified", STOP. Read this comment block. Each
// of these constructs has a documented function in the workload's
// optimization-resistance contract. Removing one (a) breaks the
// scheduler-event timing the workload claims to produce, (b)
// degrades the cache-pressure traffic, (c) collapses multi-step
// arithmetic into a single fold, OR (d) all three. The breakage
// won't surface as a test failure — it'll surface as silently
// degraded workload realism, which is much harder to debug than
// a panic.

/// CPU spin burst: black_box increment + spin_loop hint, repeated `count` times.
///
/// `#[inline(never)]` is deliberate: when this is inlined into a
/// caller that also does observable work after the loop, LLVM can
/// merge `count`-many `wrapping_add(1)` operations into a single
/// `+ count` operation, defeating the point of the per-iteration
/// `black_box`. Forcing the function out-of-line keeps each
/// iteration's `black_box`-wrapped increment visible as a
/// distinct call-and-return boundary the optimizer cannot fold.
#[inline(never)]
pub(super) fn spin_burst(work_units: &mut u64, count: u64) {
    for _ in 0..count {
        *work_units = std::hint::black_box(work_units.wrapping_add(1));
        std::hint::spin_loop();
    }
}

/// Read the worker's effective cpuset (the CPUs it may run on) via
/// `sched_getaffinity(0)`. Returns the set CPU indices, or an empty
/// vec if the query fails. Shared by the [`WorkType::AffinityChurn`] /
/// [`WorkType::CrossAffinityChurn`] setup and by `WorkerCtx::cpus` so
/// the syscall logic lives in one place.
fn read_effective_cpus() -> Vec<usize> {
    let mut cpu_set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let ret =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut cpu_set) };
    if ret == 0 {
        (0..libc::CPU_SETSIZE as usize)
            .filter(|c| unsafe { libc::CPU_ISSET(*c, &cpu_set) })
            .collect()
    } else {
        Vec::new()
    }
}

/// Parse a cgroup-v2 relative path from the contents of
/// `/proc/self/cgroup` — the single `0::/<rel>` line. Returns the
/// `<rel>` portion (e.g. `/ktstr/cgA`), trimmed; `None` when no `0::`
/// line is present (a cgroup-v1-only host). Pure so it is testable on a
/// literal without a live `/proc`.
fn parse_cgroup_rel(proc_self_cgroup: &str) -> Option<String> {
    proc_self_cgroup
        .lines()
        .find_map(|l| l.strip_prefix("0::").map(|s| s.trim().to_string()))
}

/// The calling task's cgroup-v2 relative path, read from
/// `/proc/self/cgroup`. `None` when the file is unreadable or carries no
/// `0::` line. Single-sources the `0::` parse for [`read_sibling_pids`]
/// and [`read_self_cgroup_dir`].
fn read_self_cgroup_rel() -> Option<String> {
    parse_cgroup_rel(&std::fs::read_to_string("/proc/self/cgroup").ok()?)
}

/// Map a cgroup-v2 relative path (as produced by [`parse_cgroup_rel`])
/// to the absolute cgroup-v2 directory `/sys/fs/cgroup<rel>`. `None` for
/// the root (`rel == "/"`) or an empty `rel` so the bare
/// `/sys/fs/cgroup` root is never mistaken for a dedicated cgroup. Pure
/// so both branches are testable on a literal.
fn cgroup_dir_from_rel(rel: &str) -> Option<std::path::PathBuf> {
    if rel == "/" || rel.is_empty() {
        return None;
    }
    Some(std::path::PathBuf::from(format!("/sys/fs/cgroup{rel}")))
}

/// The calling worker's absolute cgroup-v2 directory
/// (`/sys/fs/cgroup<rel>`), or `None` for the root cgroup (`rel == "/"`)
/// or when cgroup v2 is unavailable. `None` — never a bogus
/// `/sys/fs/cgroup` — so a caller never mistakes the root for a
/// dedicated cgroup. Backs [`WorkerCtx::cgroup_dir`] (the production
/// consumer); `pub(crate)` so the cross-module wiring regression-pin
/// (`spawn::tests_lifecycle::custom_worker_receives_wired_ctx`) can
/// assert the `worker_main` hand-off threads this value through.
pub(crate) fn read_self_cgroup_dir() -> Option<std::path::PathBuf> {
    cgroup_dir_from_rel(&read_self_cgroup_rel()?)
}

/// Resolve the `cgroup.procs` path of a SIBLING cgroup — a child of
/// `cgroup_dir`'s parent named `name` — validating `name` first so a
/// hostile `.`/`..`/absolute/NUL component cannot escape the parent via
/// `Path::join`. `Err(InvalidInput)` for a bad name; `Err(NotFound)`
/// when `cgroup_dir` is `None` (root / non-v2) or has no parent. Pure
/// (no fs access) so both error branches are testable. Single-sources
/// the sibling resolution shared by [`WorkerCtx::open_sibling_cgroup_procs`]
/// (the Custom-worker affordance) and the
/// [`WorkType::CgroupAttachStorm`]
/// built-in dispatch arm so the two address the same target by
/// construction.
pub(crate) fn resolve_sibling_cgroup_procs(
    cgroup_dir: Option<&std::path::Path>,
    name: &str,
) -> std::io::Result<std::path::PathBuf> {
    crate::cgroup::validate_cgroup_name(name)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
    let cg = cgroup_dir.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "open_sibling_cgroup_procs: worker has no resolvable cgroup-v2 \
             dir (run it in a dedicated cgroup, not the root / a non-v2 host)",
        )
    })?;
    let parent = cg.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "open_sibling_cgroup_procs: the worker's cgroup has no parent",
        )
    })?;
    Ok(parent.join(name).join("cgroup.procs"))
}

/// The cgroup name a [`WorkType::CgroupChurn`] worker rotates target `i`
/// through: `wt-cgroup-churn-<i>`. Single-sourced so auto-creation and
/// path-building agree.
fn churn_cgroup_name(i: usize) -> String {
    format!("wt-cgroup-churn-{i}")
}

/// The per-target `cgroup.procs` paths a [`WorkType::CgroupChurn`] worker
/// writes its tid to — `<root>/wt-cgroup-churn-<i>/cgroup.procs` for `i in
/// 0..groups` (at least one). Shared by the worker dispatch and its unit
/// test so the layout under the workload root is single-sourced.
fn churn_cgroup_procs_paths(root: &str, groups: usize) -> Vec<String> {
    (0..groups.max(1))
        .map(|i| format!("{root}/{}/cgroup.procs", churn_cgroup_name(i)))
        .collect()
}

/// True for the `work_units`-vs-`iterations` footgun shape: a `Custom`
/// worker counted work (`work_units > 0`) but left the outer-loop counter
/// at zero (`iterations == 0`), so headline throughput — which reads
/// `WorkerReport::iterations` — reads zero. See the `WorkType::Custom`
/// telemetry contract. Pure so it is testable on a literal report.
fn custom_iterations_telltale(r: &WorkerReport) -> bool {
    r.work_units > 0 && r.iterations == 0
}

/// Warn ONCE per process that a `Custom` worker returned the
/// [`custom_iterations_telltale`] shape. Mirrors the `std::sync::Once`
/// idiom of `warn_setpriority_failed_once`; advisory only — it inspects
/// and prints, never mutates the verbatim report.
fn warn_custom_iterations_zero_once(name: &str) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "workload: Custom worker '{name}' returned work_units>0 but \
             iterations==0; headline throughput (total_iterations / \
             iterations_per_worker / iterations_per_cpu_sec) reads \
             WorkerReport::iterations and will read zero — populate iterations \
             (commonly = work_units). See the WorkType::Custom telemetry contract."
        );
    });
}

/// Read the calling worker's sibling pids from its cgroup's
/// `cgroup.procs`, excluding the caller's own pid. Used by the
/// [`WorkType::CrossAffinityChurn`] arm and by
/// `WorkerCtx::sibling_pids` so a Custom worker gets the same peer
/// set without re-deriving it from `/proc`+`/sys`.
///
/// cgroup v2: `/proc/self/cgroup` is a single `0::/<rel>` line (parsed
/// by [`read_self_cgroup_rel`]); the members live in
/// `/sys/fs/cgroup<rel>/cgroup.procs`. The set is the worker's CGROUP
/// membership, so this is only meaningful in a dedicated cgroup (see the
/// `CrossAffinityChurn` doc). A missing `0::` line OR an unreadable
/// `/proc/self/cgroup` falls back to the root (`/`); only a
/// `cgroup.procs` read error yields an empty set (the caller treats
/// "no siblings" as a no-op).
pub(super) fn read_sibling_pids() -> Vec<libc::pid_t> {
    let me = unsafe { libc::getpid() };
    let rel = read_self_cgroup_rel().unwrap_or_else(|| "/".to_string());
    let procs_path = format!("/sys/fs/cgroup{rel}/cgroup.procs");
    std::fs::read_to_string(&procs_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.trim().parse::<libc::pid_t>().ok())
        .filter(|&pid| pid != me)
        .collect()
}

/// Build two cpuset sub-masks that differ by exactly one CPU, from
/// the worker's effective `cpus`. `mask_a` covers all of `cpus`;
/// `mask_b` drops the last. Toggling between them makes every
/// `sched_setaffinity` a genuine mask change (never `cpumask_equal`
/// to the task's current mask), so the kernel runs the full
/// `set_cpus_allowed` path each iteration rather than early-returning.
/// Returns `None` when `cpus` has fewer than 2 entries — no two
/// distinct masks are possible, so [`WorkType::CrossAffinityChurn`] is
/// a no-op for that worker.
pub(super) fn build_cross_affinity_masks(
    cpus: &[usize],
) -> Option<(libc::cpu_set_t, libc::cpu_set_t)> {
    if cpus.len() < 2 {
        return None;
    }
    let make = |drop_last: bool| -> libc::cpu_set_t {
        let take = if drop_last {
            cpus.len() - 1
        } else {
            cpus.len()
        };
        let mut s: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe { libc::CPU_ZERO(&mut s) };
        for &c in cpus.iter().take(take) {
            unsafe { libc::CPU_SET(c, &mut s) };
        }
        s
    };
    Some((make(false), make(true)))
}

/// Strided read-modify-write over a cache buffer.
///
/// `-O2`/`-O3` are aggressive about eliminating "no-observer" memory
/// traffic on a process-local `Vec<u8>`: nothing outside the worker
/// reads `buf`, so without an explicit barrier LLVM may prove every
/// store dead and collapse the loop body to the `work_units`
/// increment alone. `work_units` flows into a shared iter-slot
/// atomic and the worker report, which keeps THAT dependency live,
/// but that observable flow does not force the independent cache
/// traffic to execute.
///
/// `black_box` on a value defeats VALUE elimination — the load /
/// store has to materialize bytes the optimizer can't reason about
/// — but a sufficiently smart pass can still prove the BACKING
/// memory access dead and replace it with synthesized bytes. To
/// pin the cache-line traffic itself, route the load through
/// `ptr::read_volatile` and the store through `ptr::write_volatile`.
/// Volatile memory operations are not eliminable: each one becomes
/// a real `mov` against the actual buffer slot, which is what the
/// `WorkType::CachePressure` / `CacheYield` / `CachePipe` workloads
/// claim to exercise. The `work_units` bump retains its `black_box`
/// wrap separately to defeat increment-fusion across iterations.
///
/// `#[inline(never)]` matches `spin_burst`'s rationale: forcing
/// out-of-line keeps the per-iteration volatile load/store and
/// `black_box`-wrapped increment visible as distinct boundaries
/// LLVM cannot collapse with surrounding caller arithmetic.
#[inline(never)]
pub(super) fn cache_rmw_loop(buf: &mut [u8], stride: usize, iters: u64, work_units: &mut u64) {
    let len = buf.len();
    let mut idx = 0;
    for _ in 0..iters {
        // SAFETY: `idx` stays in `0..len` (mod by len at the bottom
        // of the loop), so `&buf[idx]` is a valid `&u8` and
        // `&mut buf[idx]` is a valid `&mut u8`. Volatile read/write
        // through these references is sound; volatility just suppresses
        // optimization, it does not change pointer-validity rules.
        let cur = unsafe { std::ptr::read_volatile(&buf[idx]) };
        unsafe { std::ptr::write_volatile(&mut buf[idx], cur.wrapping_add(1)) };
        idx = (idx + stride) % len;
        *work_units = std::hint::black_box(work_units.wrapping_add(1));
    }
}

/// Weyl-sequence increment derived from the golden ratio,
/// `2^64 / phi`. Used wherever the worker needs a per-thread RNG
/// seed or a multiplicative spread constant that avoids the
/// degenerate "all zero" case on tid `0`. Same value glibc's
/// `nrand48` family uses; promoting it to a named constant lets
/// every call site reference one source instead of re-typing the
/// 64-bit literal.
pub(super) const GOLDEN_RATIO_64: u64 = 0x9E37_79B9_7F4A_7C15;

/// Per-iteration multiply-chain step count for [`WorkType::AluHot`].
/// Chosen so a single [`alu_hot_chain`] call retires several thousand
/// real ALU ops before the outer loop checks `stop` — large enough
/// to drive functional-unit pressure, small enough that shutdown
/// latency stays in the millisecond regime even on slow cores.
pub(super) const ALU_HOT_CHAIN_STEPS: u64 = 1024;

/// Cold-phase working-set size in u64 slots for [`WorkType::IpcVariance`].
/// 65_536 × 8 bytes = 512 KiB — large enough to spill out of typical
/// L1 (32 KiB) and L2 (256 KiB), small enough to fit in L3 on most
/// hosts (LLC pressure rather than DRAM-spill on the common case).
pub(super) const IPC_VARIANCE_REGION_U64: usize = 64 * 1024;

/// Resolve a configured [`AluWidth`] to a width the host can actually
/// run. Probes CPU features at call time; never fabricates a higher
/// width than the host supports.
///
/// Resolution order on x86_64: `Amx > Vec512 > Vec256 > Vec128 > Scalar`.
/// On aarch64 only `Vec128` (NEON) is reachable; everything wider
/// downgrades to `Vec128` when NEON is present, otherwise `Scalar`.
/// On any other architecture every width downgrades to `Scalar`.
pub(super) fn resolve_alu_width(requested: AluWidth) -> AluWidth {
    #[cfg(target_arch = "x86_64")]
    {
        // `is_x86_feature_detected!` reads the compiler's runtime
        // feature-detection cache. Order checked: AVX-512F, AVX2,
        // SSE2 (always present on x86_64). AMX detection requires
        // the unstable `x86_amx_intrinsics` feature gate so it is
        // not probed here on stable; an explicit
        // `AluWidth::Amx` request downgrades to the next-widest
        // available variant. See follow-up #309 for the AMX
        // detection path under nightly / future stabilization.
        let widest = if std::is_x86_feature_detected!("avx512f") {
            AluWidth::Vec512
        } else if std::is_x86_feature_detected!("avx2") {
            AluWidth::Vec256
        } else {
            // SSE2 is part of the x86_64 baseline ABI per
            // System V x86_64 ABI; skip the runtime check.
            AluWidth::Vec128
        };
        match requested {
            AluWidth::Widest => widest,
            // AMX is not probed on stable — downgrade to the
            // host's widest stable-detectable variant.
            AluWidth::Amx => widest,
            AluWidth::Vec512 if std::is_x86_feature_detected!("avx512f") => AluWidth::Vec512,
            AluWidth::Vec512 => widest,
            AluWidth::Vec256 if std::is_x86_feature_detected!("avx2") => AluWidth::Vec256,
            AluWidth::Vec256 => widest,
            AluWidth::Vec128 => AluWidth::Vec128,
            AluWidth::Scalar => AluWidth::Scalar,
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON (asimd) is mandatory on AArch64-A profile per the
        // ARMv8-A ABI — no runtime check needed for `Vec128`.
        // Wider variants (Vec256/Vec512/Amx) have no aarch64
        // equivalent in the v8/v9 base ABI; downgrade to `Vec128`.
        match requested {
            AluWidth::Widest | AluWidth::Vec128 => AluWidth::Vec128,
            AluWidth::Vec256 | AluWidth::Vec512 | AluWidth::Amx => AluWidth::Vec128,
            AluWidth::Scalar => AluWidth::Scalar,
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Unsupported architecture — downgrade everything to
        // `Scalar`. Other arches don't surface in the ktstr CI
        // matrix today; this branch keeps compilation honest.
        let _ = requested;
        AluWidth::Scalar
    }
}

/// Run an ALU multiply chain at the given width for `steps`
/// iterations. The chain runs four independent streams in parallel
/// (each its own dependency chain) so a modern out-of-order core
/// can dispatch all four per cycle, sustaining IPC ≥ 2.0 with no
/// data dependency between adjacent multiplies.
///
/// `width` is the resolved variant from [`resolve_alu_width`] — the
/// caller MUST resolve before calling so this function never sees
/// a `Widest` sentinel. The width currently selects the scalar
/// integer multiply for every variant; future work can drop in
/// SIMD intrinsics under each arm. Even the scalar path retires
/// real arithmetic at IPC ≥ 2.0 because of the four-way independence.
///
/// `#[inline(never)]` matches `spin_burst` / `cache_rmw_loop`'s
/// rationale: forcing out-of-line keeps the per-step `black_box`
/// barriers visible to LLVM as distinct call boundaries that
/// can't be merged with surrounding caller arithmetic.
#[inline(never)]
pub(super) fn alu_hot_chain(width: AluWidth, steps: u64, work_units: &mut u64) {
    // Independent multiplier streams. Initial values are
    // black_box'd so the compiler can't fold them at compile
    // time. Per-step constants are large odd primes
    // (golden-ratio-scaled) — no zero or one that would
    // collapse the chain.
    let mut a: u64 = std::hint::black_box(GOLDEN_RATIO_64);
    let mut b: u64 = std::hint::black_box(0xBF58_476D_1CE4_E5B9u64);
    let mut c: u64 = std::hint::black_box(0x94D0_49BB_1331_11EBu64);
    let mut d: u64 = std::hint::black_box(0x2545_F491_4F6C_DD1Du64);
    // Caller MUST resolve `Widest` via `resolve_alu_width` before
    // reaching here. The width parameter currently runs the same
    // scalar multiply path regardless of variant — `width` is
    // retained on the signature so a future SIMD-intrinsic drop
    // can pivot per-arm without changing the call shape. The
    // debug_assert keeps the resolution invariant honest in
    // tests; release builds elide the check.
    debug_assert!(
        !matches!(width, AluWidth::Widest),
        "alu_hot_chain reached with AluWidth::Widest; \
         caller must resolve via resolve_alu_width first",
    );
    for _ in 0..steps {
        a = std::hint::black_box(a).wrapping_mul(0xD134_2543_DE82_EF95);
        b = std::hint::black_box(b).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        c = std::hint::black_box(c).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        d = std::hint::black_box(d).wrapping_mul(0xCA45_4F47_1E40_FE19);
        *work_units = std::hint::black_box(work_units.wrapping_add(1));
    }
    // Final black_box on the chain values keeps the entire
    // multiply chain live: without an observable sink the
    // optimizer could prove the loop dead and elide every
    // iteration. The XOR fold collapses the four streams into
    // one observable value the black_box wraps; the fold itself
    // adds one cycle to the function but is paid once per call,
    // not per step.
    let sink = a ^ b ^ c ^ d;
    let _ = std::hint::black_box(sink);
}

/// Naive matrix multiply: three square matrices of u64, O(n^3).
///
/// The caller owns a `Vec<u64>` of length `3 * size * size` so the
/// storage is naturally 8-byte aligned. An earlier version took a
/// `&mut [u8]` and cast to `*mut u64`, which was UB because a
/// `Vec<u8>` is only 1-byte aligned.
///
/// Optimization-elimination barrier: the slice is `black_box`-clobbered
/// ONCE at entry, the C-region store uses `write_volatile`, and one C value
/// is read back into the observable `work_units`. `matrix_buf` in
/// `worker_main` is a process-local `Vec<u64>` whose C region (the upper
/// third) is NEVER read by `matrix_multiply` or any caller, and the A/B
/// regions stay zero-initialized — so LLVM is otherwise free to (a) prove
/// the loads are zero and fold the multiply to a constant, and (b) mark the
/// store dead and elide the chain feeding it. The entry `black_box` makes
/// the buffer opaque (kills the fold); the `write_volatile` makes the store
/// non-elidable and the read-back gives the chain a load-bearing consumer
/// (kills the DCE) — so the compute path the workload claims to exercise
/// actually executes under `-O2`/`-O3`.
///
/// The barrier is on the SLICE, not per LOAD: a per-load `black_box` forces
/// every operand through a register barrier, blocking the fused memory-operand
/// `imul` and loop unrolling LLVM emits otherwise -- which de-optimized do_work
/// ~1.3x vs schbench's `-O2` `do_some_math`. One entry barrier stays fold-safe
/// while leaving the scalar inner loop intact, matching schbench's compute cost.
/// The loop does NOT auto-vectorize either way (no SIMD `u64` multiply on the
/// `x86-64-v3` target and the serial accumulator blocks SLP -- schbench's `-O2`
/// kernel is likewise scalar `imul`), so the entry barrier costs no SIMD.
///
/// `#[inline(never)]` matches `spin_burst` / `cache_rmw_loop` — the boundary
/// keeps the caller's zero-init from flowing in; the entry `black_box` closes
/// the residual thin-LTO IPO path (`Cargo.toml` `lto = "thin"`).
#[inline(never)]
pub(super) fn matrix_multiply(data: &mut [u64], size: usize, work_units: &mut u64) {
    debug_assert_eq!(data.len(), 3 * size * size);
    let stride = size * size;
    // Clobber the buffer ONCE so the optimizer treats it as opaque and cannot
    // prove the zero-initialized A/B regions are all-zero (which would let it
    // fold the multiply to a constant). #[inline(never)] hides the caller's
    // zero-init across the call boundary; this also closes the residual
    // thin-LTO IPO path (Cargo.toml `lto = "thin"`). Unlike a per-load
    // black_box, one barrier on the slice leaves the inner loop's scalar
    // codegen intact (fused memory-operand `imul`, unrolling) instead of
    // forcing every operand through a register barrier.
    let data = std::hint::black_box(data);
    for i in 0..size {
        for j in 0..size {
            let mut acc: u64 = 0;
            for k in 0..size {
                acc =
                    acc.wrapping_add(data[i * size + k].wrapping_mul(data[stride + k * size + j]));
            }
            // SAFETY: `2 * stride + i * size + j` is in-bounds for a slice of
            // length `3 * stride` whenever `i, j < size`, which the surrounding
            // `for` ranges enforce; the `debug_assert_eq!` above pins the length
            // contract and `u64` is naturally aligned via the `Vec<u64>` backing.
            // The volatile store makes the C-region write non-elidable, so the
            // multiply chain feeding `acc` executes under `-O2`/`-O3` (no later
            // code reads the C region, so a plain store would be DCE-eligible).
            unsafe {
                std::ptr::write_volatile(&mut data[2 * stride + i * size + j] as *mut u64, acc);
            }
        }
    }
    // Defense-in-depth read-back sink: route one C-region value into the
    // observable `work_units` so the multiply chain has a load-bearing consumer
    // beyond the volatile store. `data[2 * stride]` is in-bounds because the
    // caller only invokes matrix_multiply when `matrix_size > 0`.
    *work_units = work_units.wrapping_add(data[2 * stride]);
}

/// Shared-matrix variant of [`matrix_multiply`] for schbench's `--split` shared
/// matrix, which every worker thread writes concurrently. It mirrors what gcc and
/// clang actually emit for schbench's shared `do_some_math` (`m3[..] = 0; for k:
/// m3[..] += m1[..]*m2[..]`): the running sum is kept in a register, but both
/// compilers STORE it to the shared C cell every `k` (`size^3` shared-C stores
/// per matrix, not one) and do NOT reload it. The store is kept because
/// `do_some_math` reads `m1`/`m2`/`m3` as offsets into one base pointer (its
/// `data` param), so neither compiler can prove the `m3` store doesn't alias the
/// next `k`'s `m1`/`m2` loads -- this is in-function aliasing (the SAME
/// `do_some_math` schbench uses for the private matrix; one symbol, three call
/// sites in the gcc binary), not anything specific to the shared global. That
/// per-k store is the source of `--split`'s cross-core cache-line contention, so
/// this kernel accumulates in a register AND stores `acc` to the shared C cell
/// every `k`; storing once per cell would under-generate the contention split
/// exists to model. (The private [`matrix_multiply`] does store once per cell --
/// non-split, ktstr stays within schbench's gcc-vs-clang throughput envelope with
/// no contention to amplify the difference, and a faithful per-k store there
/// would need DCE-defeating volatile writes.) A and B are loaded each `k`
/// (`size^3` shared loads); C is write-only in the loop. All accesses use
/// `Ordering::Relaxed`: on x86-64 each lowers to a plain `MOV` (only read-modify-
/// write atomics take `LOCK`), so the contention matches schbench's plain
/// shared-memory race -- but it is sound (every concurrent access goes through an
/// atomic, so there is no data race / no UB, unlike a plain shared `&mut` buffer;
/// the C result races exactly as schbench's does, which the contention workload
/// does not depend on).
///
/// No `black_box` / `write_volatile` / read-back barrier (unlike
/// [`matrix_multiply`], whose barriers stop the optimizer constant-folding the
/// zero-init A/B and DCE-ing the single C store on the single-thread plain path):
/// an atomic `load(Relaxed)` is un-foldable and an atomic `store(Relaxed)` is
/// un-elidable (a cross-thread-visible side effect), so the per-k C stores cannot
/// be collapsed back to a single store. Neither this nor [`matrix_multiply`]
/// auto-vectorizes (the serial accumulator reduction blocks SLP for both; there
/// is no SIMD `u64` multiply on the `x86-64-v3` target -- verified by disassembly
/// of ktstr and of gcc- and clang-built schbench).
///
/// Takes `&[AtomicU64]` (a shared ref, interior mutability), which is what lets
/// every worker share one `Arc<[AtomicU64]>`. `#[inline(never)]` matches
/// [`matrix_multiply`] so the per-call boundary cost is the same.
#[inline(never)]
pub(super) fn matrix_multiply_shared(
    data: &[core::sync::atomic::AtomicU64],
    size: usize,
    work_units: &mut u64,
) {
    use core::sync::atomic::Ordering::Relaxed;
    debug_assert_eq!(data.len(), 3 * size * size);
    let stride = size * size;
    for i in 0..size {
        for j in 0..size {
            // Mirror what gcc and clang actually emit for schbench's shared
            // `do_some_math` (`m3[..] += m1[..]*m2[..]`): keep the running sum in
            // a register, but STORE it to the shared C cell every k. Both keep the
            // per-k store because m1/m2/m3 are offsets into one base pointer, so
            // neither can prove the m3 store doesn't alias the next k's m1/m2
            // loads (in-function aliasing -- the same do_some_math is used for the
            // private matrix). That per-k store (size^3 shared-C stores per
            // matrix, no reload) is the source of --split's cross-core cache
            // contention, so we store `acc` every k; a single store per cell would
            // under-generate it.
            let c = &data[2 * stride + i * size + j];
            c.store(0, Relaxed);
            let mut acc: u64 = 0;
            for k in 0..size {
                let prod = data[i * size + k]
                    .load(Relaxed)
                    .wrapping_mul(data[stride + k * size + j].load(Relaxed));
                acc = acc.wrapping_add(prod);
                c.store(acc, Relaxed);
            }
        }
    }
    // Route one computed C value into the observable `work_units` for
    // loop-accounting parity with `matrix_multiply` (NOT a DCE barrier -- the
    // per-k atomic stores above are already non-elidable). In-bounds: the caller
    // invokes this only when the shared matrix size is > 0.
    *work_units = work_units.wrapping_add(data[2 * stride].load(Relaxed));
}

/// Write 1 byte to partner, poll for response, read, record wake latency.
pub(super) fn pipe_exchange(read_fd: i32, write_fd: i32, wake: &mut WakeRec, stop: &AtomicBool) {
    unsafe { libc::write(write_fd, b"x".as_ptr() as *const _, 1) };
    let before_block = Instant::now();
    let mut pfd = libc::pollfd {
        fd: read_fd,
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        if stop_requested(stop) {
            break;
        }
        let ret = unsafe { libc::poll(&mut pfd, 1, 100) };
        if ret > 0 {
            let mut byte = [0u8; 1];
            unsafe { libc::read(read_fd, byte.as_mut_ptr() as *mut _, 1) };
            wake.push(before_block.elapsed().as_nanos() as u64);
            break;
        }
        if ret < 0 {
            break;
        }
    }
}

/// Record a wake latency sample using reservoir sampling (Algorithm R).
/// Maintains a uniform random sample of at most `cap` entries from all
/// observed latencies.
///
/// The replacement-index draw uses a thread-local xorshift64 PRNG so
/// the hot path avoids `rand::rng()`'s ChaCha20 block-RNG seeding
/// cost (one syscall on first use plus ChaCha20 block computation
/// every ~64 draws) and the rand crate's documented post-fork
/// seed-correlation hazard. `ThreadRng` is a non-reseeded handle into
/// a per-thread `Rc<UnsafeCell<BlockRng<ReseedingCore>>>`
/// (`rand-0.10/src/rngs/thread.rs`); after `fork(2)` the child
/// inherits the parent's RNG state, so multiple worker children spawned
/// in lock-step would draw identical first samples without an explicit
/// `ThreadRng::reseed()` call (which the rand crate's own doc requires
/// post-fork). xorshift64 with a tid-derived seed sidesteps the issue
/// by giving every worker an independent stream from its first call.
///
/// Modulo bias for `r % count` where `r` is a uniform u64 and
/// `count <= MAX_WAKE_SAMPLES = 100_000`: the bias bound is
/// `count / 2^64 ≈ 5.4e-15`, far below any statistical threshold a
/// reservoir sampler can resolve. Algorithm R only requires that the
/// replacement decision be uniform-ish; this bound is orders of
/// magnitude inside the noise floor.
///
/// The seed is derived from `gettid(2)` × `GOLDEN_RATIO_64` on first
/// use (matching the worker's other PRNG seeds — `io_rng`,
/// `ipc_variance_rng`, `page_fault_rng_state`) so each thread / forked
/// worker produces an independent stream. `cell.get() == 0` is the
/// "uninitialised" sentinel because xorshift64 has 0 as a fixed point.
pub(crate) fn reservoir_push(buf: &mut Vec<u64>, count: &mut u64, sample: u64, cap: usize) {
    *count += 1;
    if buf.len() < cap {
        buf.push(sample);
    } else {
        thread_local! {
            // `Cell<u64>` is enough — xorshift state is a single u64
            // with no Drop. `const { ... }` keeps the initialiser a
            // compile-time constant so first-touch cost on the hot
            // path is bounded to the seed-from-tid step below, not a
            // generic thread-local lazy-init.
            static RESERVOIR_RNG: std::cell::Cell<u64> = const {
                std::cell::Cell::new(0)
            };
        }
        let r = RESERVOIR_RNG.with(|cell| {
            let mut s = cell.get();
            if s == 0 {
                // Lazy seed on first use. Mirrors the per-worker
                // seed pattern in `worker_main` for `io_rng` /
                // `ipc_variance_rng`: tid × golden-ratio Weyl
                // increment, with a non-zero fallback if the
                // multiply happened to land on the xorshift fixed
                // point. SYS_gettid is a vDSO-cached fast path on
                // glibc + recent kernels; the cost is paid once
                // per thread.
                let tid = unsafe { libc::syscall(libc::SYS_gettid) as u64 };
                s = tid.wrapping_mul(GOLDEN_RATIO_64);
                if s == 0 {
                    s = GOLDEN_RATIO_64;
                }
            }
            // xorshift64 — same triple-shift as the inline
            // PageFaultChurn / IpcVariance helpers and the
            // `io::xorshift64` re-export.
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            cell.set(s);
            s
        });
        let idx = (r % *count) as usize;
        if idx < cap {
            buf[idx] = sample;
        }
    }
}

// Scheduler / clock / metric helpers (read_schedstat, parse_schedstat_line,
// clock_gettime_ns, set_sched_policy, etc.) live in sched.rs to keep
// this file under the per-file line budget. Re-imported via
// `use sched::*;` so the dispatch arms reference the items without
// qualification, matching the pre-split call sites.
mod sched;
use sched::*;

#[cfg(test)]
mod tests;
