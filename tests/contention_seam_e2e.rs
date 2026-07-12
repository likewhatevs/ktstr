//! End-to-end fixtures for the host-side contention seam
//! (`apply_contention_verdict`): the tri-state re-evaluation of a guest
//! `max_p99_wake_latency_ns` gate failure against the Body-phase
//! contention witness measured host-side.
//!
//! Both cells run the `scx-ktstr` scheduler and drive the scenario through
//! `execute_steps_with` so the run reaches `LifecycleStage::Body` via
//! `send_scenario_start` — the precondition (together with an scx BPF
//! scheduler under test) for `VmResult::contention_witness` to populate.
//! The workload is `Bursty`, whose burst/sleep worker blocks off-CPU each
//! cycle and records the sleep's wall duration as a wake latency into
//! `WorkerReport::wake_latencies_ns` — the exemplar the p99 gate reads.
//! (SpinWait — the default cgroup work type — and FutexPingPong record no
//! contention-SENSITIVE wake latency: SpinWait never blocks, and a
//! FutexPingPong hand-off completes inside one host time-slice, so its p99
//! stays ~30 µs even at D ≈ 2. Bursty's timer wakeup, by contrast, waits
//! for the host CPU under overcommit, so its p99 inflates — the property
//! (b) needs.)
//!
//! Both branches turn on the same lever — the size of the measured p99
//! relative to `W`, the peak host run-delay over the (tick-quantised,
//! ≥ 2-tick) window `latency_verdict` scores the p99 exemplar against:
//!
//! (a) QUIET-HOST CONFIRMED FAIL — default (non-overcommit) placement, a
//!     single Bursty worker with a large (~500 ms) sleep and a
//!     `max_p99_wake_latency_ns` of 1 ns that always trips. The sole worker
//!     is asleep most of the window so its vCPU is idle and the host accrues
//!     little run-delay against it — `W` stays far below the ~500 ms gap, so
//!     the tri-state CONFIRMS (excess > W) and the failure STAYS. `expect_err`
//!     makes the confirmed-fail the green expected outcome — proving the
//!     confirmed path survives.
//!
//! (b) CONTENDED → INDETERMINATE → PASS — `no_perf_mode` + `cpu_budget = 1`
//!     with 2 vCPUs plus a free-running burner on the sibling vCPU, so the
//!     host time-slices the two vCPU threads (Body dilation D ≈ 2). A single
//!     Bursty worker with a small (~2 ms) sleep sees its wakeup inflated to a
//!     ~22 ms p99 by host wakeup delay; the gate is tuned to trip only under
//!     that inflation, and the witnessed host contention `W` (~300 ms over
//!     the p99-long window) covers the excess, so the seam DEMOTES the
//!     failure to a non-blocking `contention-indeterminate` pass. The test
//!     PASSES with the annotation visible.

use anyhow::Result;
use ktstr::assert::{Assert, AssertResult};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps_with};
use ktstr::test_support::{Scheduler, SchedulerSpec};
use ktstr::workload::{WorkType, WorkloadConfig, WorkloadHandle};
use std::time::Duration;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Build a one-step scenario running `work_type` across `workers` workers
/// in one cgroup, held for the full duration, checked with `checks`. The
/// Bursty work type used here blocks off-CPU each cycle, so every sleep
/// records a wake latency — the exemplar the p99 gate reads (SpinWait, the
/// default, records none).
fn wake_latency_scenario(
    ctx: &Ctx,
    workers: usize,
    work_type: WorkType,
    checks: &Assert,
) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0").workers(workers).work_type(work_type)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(checks))
}

// ===========================================================================
// (a) Quiet-host confirmed fail: measured p99 (~500 ms) >> W → the tri-state
//     confirms, and the failure stays.
// ===========================================================================

/// Bursty sleep per cycle. A single Bursty worker sleeps ~this long each
/// iteration (a large intrinsic off-CPU gap the scheduler "allowed"),
/// recording it as a wake latency. Because the sole worker is asleep for
/// most of the window, its one vCPU is mostly halted — the host accrues
/// little run-delay against it, so the Body-phase contamination bound `W`
/// stays FAR below this gap. The p99 (~this sleep) therefore exceeds `W`
/// and the tri-state CONFIRMS: no witnessed host contention explains a
/// half-second off-CPU gap.
const CONFIRM_SLEEP_MS: u64 = 500;
/// Bursty compute burst per cycle — a short on-CPU slice between sleeps so
/// the worker makes progress and the run is a normal burst/idle cycle
/// rather than a pure sleep.
const CONFIRM_BURST_MS: u64 = 50;

/// A `max_p99_wake_latency_ns` of 1 ns always trips (any wake latency
/// exceeds it). The single Bursty worker's ~500 ms off-CPU gap is an
/// intrinsic workload latency, not host contention: its idle vCPU accrues
/// little host run-delay, so the Body-phase witness bound `W` stays well
/// under 500 ms and the seam CONFIRMS the failure (excess > W), appending
/// the `(contention-checked: ...)` note. `expect_err` inverts that
/// confirmed failure to the green expected outcome, proving the confirmed
/// path survives the seam.
///
/// One vCPU (threads = 1) keeps the witnessed run-delay single-threaded
/// (W is a Σ over vCPU threads, so fewer threads ⇒ smaller W for the same
/// per-CPU steal), widening the excess-over-W margin on a busy host.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    expect_err,
    llcs = 1,
    cores = 1,
    threads = 1,
    duration_s = 8,
    watchdog_timeout_s = 40
)]
fn contention_confirmed_fail_quiet_host(ctx: &Ctx) -> Result<AssertResult> {
    let checks = Assert::default_checks().max_p99_wake_latency_ns(1);
    let work = WorkType::bursty(
        Duration::from_millis(CONFIRM_BURST_MS),
        Duration::from_millis(CONFIRM_SLEEP_MS),
    );
    wake_latency_scenario(ctx, 1, work, &checks)
}

// ===========================================================================
// (b) Contended → indeterminate → pass: host time-slices the vCPUs (D ≈ 2),
//     W covers the excess, the seam demotes to a non-blocking pass.
// ===========================================================================

/// Threshold (ns) for the overcommit cell's p99 wake-latency gate. Sits
/// ABOVE the quiet-host Bursty wakeup (~the sleep itself, a few ms) but
/// BELOW the overcommit-inflated p99 (the sleep plus tens of ms of host
/// wakeup delay when the sibling vCPU holds the one host CPU), so the gate
/// trips ONLY under contention. The excess over this ceiling is pure host
/// wakeup delay, which the Body-phase witness `W` (peak run-delay over the
/// p99-long window, hundreds of ms under D ≈ 2) covers — landing the
/// tri-state on `ContentionIndeterminatePass`.
///
/// 5 ms is empirical (kernel 7.1.3, cpu_budget = 1, 2 vCPUs): the quiet
/// Bursty wakeup floor is ~the 2 ms sleep, while the overcommit-inflated
/// pooled p99 measured ~22 ms (median ~2 ms — most wakeups uncontended, the
/// tail stretched by host wakeup delay), with `W` ~300 ms. 5 ms clears the
/// quiet floor yet the contended p99 overruns it ~4x, so the gate trips
/// under contention and the excess (~17 ms) is covered by `W` with a wide
/// margin.
const OVERCOMMIT_P99_THRESHOLD_NS: u64 = 5_000_000;

/// Bursty measurement worker cadence. Each cycle bursts on-CPU briefly then
/// sleeps this long; the recorded wake latency is the actual wall time of
/// that sleep — on a quiet host ~the sleep, but under overcommit inflated
/// by however long the woken thread waits for the one host CPU (held by the
/// sibling vCPU's burner). Short so the sleep itself contributes little and
/// the inflation is dominated by host wakeup delay.
const OVERCOMMIT_SLEEP_MS: u64 = 2;
const OVERCOMMIT_BURST_MS: u64 = 1;

/// `no_perf_mode` + `cpu_budget = 1` with 2 vCPUs: the two vCPU threads
/// share one host CPU, so the host time-slices them (Body dilation D ≈ 2).
/// A free-running SpinWait burner keeps the sibling vCPU runnable for the
/// whole window, so the one host CPU is always contended — when the Bursty
/// measurement worker's sleep timer fires, its vCPU must wait for the host
/// CPU, inflating the recorded wake latency (Bursty's wakeup is
/// contention-SENSITIVE, unlike FutexPingPong's sub-ms hand-off which the
/// two co-scheduled vCPUs complete inside one host slice). The inflated p99
/// trips the gate in-guest, but the host-side witness (`W` over the p99-long
/// window) covers the excess, so the seam DEMOTES the failure to a
/// `contention-indeterminate` non-blocking pass. The test PASSES (NOT
/// `expect_err`) with the annotation on `info_notes` and echoed to stderr.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    no_perf_mode,
    cpu_budget = 1,
    llcs = 1,
    cores = 1,
    threads = 2,
    duration_s = 12,
    watchdog_timeout_s = 60
)]
fn contention_indeterminate_pass_under_overcommit(ctx: &Ctx) -> Result<AssertResult> {
    // Free-running burner on the sibling vCPU: keeps the second vCPU
    // runnable for the whole window so the one host CPU stays contended and
    // the measurement worker's wakeup is delayed behind it (D ≈ 2).
    let mut burner = WorkloadHandle::spawn(&WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::SpinWait,
        ..Default::default()
    })?;
    burner.start();

    let checks = Assert::default_checks().max_p99_wake_latency_ns(OVERCOMMIT_P99_THRESHOLD_NS);
    let work = WorkType::bursty(
        Duration::from_millis(OVERCOMMIT_BURST_MS),
        Duration::from_millis(OVERCOMMIT_SLEEP_MS),
    );
    let result = wake_latency_scenario(ctx, 1, work, &checks);

    let _ = burner.stop_and_collect();
    result
}
