//! End-to-end verification of the three-tier progress watchdog: fast
//! Tier-1 / Tier-2 kills on injected INFRA-phase wedges, plus the
//! Body-stage exemption that keeps a legitimately-idle test alive.
//!
//! # Progress is milestone-only — why these fixtures work
//!
//! An earlier evidence model counted kernel scheduling noise
//! (ttwu/sched_count deltas) as progress, which made every injected wedge
//! immortal: a live kernel's background kthread wakeups reset the progress
//! clock every tick (measured: a 60 s teardown idle-sleep sailed through
//! untripped). The current model (`src/vmm/freeze_coord/watchdog_step.rs`)
//! anchors progress on MILESTONES (lifecycle stage advances) only, and
//! reasons about a phase's CPU burn and CPU trickle:
//!   - **Tier-1** — the busiest vCPU's `max_vcpu_cpu_in_phase` exceeded the
//!     phase's flat `phase_cpu_budget_ns` without reaching a milestone: a
//!     *spinning* wedge.
//!   - **Tier-2** — an INFRA phase sat past its `phase_wall_backstop_ns`
//!     with live evidence channels, no runnable demand, and its CPU
//!     trickle-stalled (two consecutive 10 s windows under the 1 ms floor):
//!     a *silent idle* wedge.
//!   - **Tier-3** — the dead-man wall deadline, deferring while the cell is
//!     alive (CPU accruing or a recent milestone) and firing only on a dead
//!     monitor or a truly inert cell.
//!
//! Both progress tiers fire ONLY in INFRA stages (Boot / Attach / Dispatch
//! / Teardown); the Body stage is exempt via the `u64::MAX` Body budgets in
//! `test_support::runtime::{phase_cpu_budget_ns, phase_wall_backstop_ns}`.
//!
//! # Fixture mechanism
//!
//! A `#[ktstr_test]` body runs exclusively in the Body stage and the guest
//! teardown path bounds all its own waits, so no workload or scenario op
//! can wedge an INFRA stage. The wedge fixtures instead use a guest-side
//! fault-injection hook (`maybe_inject_teardown_wedge_fault` in
//! `src/vmm/rust_init/init.rs`, sibling of the Boot-phase
//! `KTSTR_FAULT_AP_GAP` hook): the host sets
//! `KTSTR_FAULT_TEARDOWN_WEDGE=idle|spin` on the guest cmdline via the test
//! scheduler's `kargs`; at the end of guest teardown (background threads
//! stopped, `send_exit` not yet sent) the guest advances the host stage to
//! Teardown via `ScenarioEnd` — a milestone, anchoring the in-phase deltas
//! at the wedge start — and then sleeps or spins FOREVER. The watchdog is
//! what ends the VM.
//!
//! The body publishes `AssertResult::pass()` BEFORE the wedge, so the
//! host's parse-success arm records a passing base verdict (`timed_out`
//! alone does not flip a parsed PASS). Each fixture's
//! `post_vm_unconditional` callback then asserts the kill happened AND was
//! fast; a callback `Err` rides the `PostVmAssertionFailure` marker, an
//! unconditional hard `EXIT_FAIL`. So the fixture is GREEN exactly when the
//! wedge dies fast on its progress tier, and RED when the watchdog does not
//! fire or fires only at Tier-3 speed.
//!
//! # Timing bounds (1 vCPU, `no_perf_mode`, `watchdog_timeout_s = 90`)
//!
//! Tier-3 dead-man deadline (`vm_timeout_from_entry`):
//! `max(watchdog_timeout_s, duration_s, 1s) + vm_boot_headroom(1) × 3`
//! = `90s + (10s + 10.15s) × 3` ≈ **150.5 s** — and for the idle shape the
//! deadman additionally waits out its 60 s milestone grace, so a
//! deadman-speed kill can never land under ~150 s.
//!
//!   - **Tier-2 idle**: the wedge starts at ~10-15 s (boot ≈ 5-10 s + 1 s
//!     body + bounded teardown preamble). Firing needs BOTH conjuncts:
//!     `wall_in_phase > 15s` (the Teardown backstop) and trickle-stall =
//!     two consecutive 10 s sub-1 ms windows — worst-case ≈ 10 s (partial
//!     window contaminated by pre-wedge CPU) + 2 × 10 s = 30 s after wedge
//!     start, so the kill lands ≈ 40-50 s total. Asserted `< 90s` — 0.6×
//!     the deadman, generous for host-load jitter yet strictly
//!     discriminating (a deadman kill cannot beat ~150 s).
//!   - **Tier-1 spin**: the Teardown CPU budget is now the flat 8 s
//!     (≤ 12 s if pthread-widened 3/2), charged against the MAX per-vCPU
//!     in-phase burn; a 1-vCPU spinner IS the max and accrues ~1 s CPU per
//!     wall-second, so Tier-1 fires ≈ 8-12 s after the wedge starts —
//!     ≈ 20-27 s total. Asserted `< 60s` (0.4× the deadman): tighter than
//!     the idle bound because Tier-1 needs no trickle windows.
//!
//! The kill CAUSE line (`cause=tier1-cpu-budget` / `cause=tier2-idle-wedge`)
//! goes to the watchdog's stderr dump — captured by nextest, not carried on
//! `VmResult` — so cause-exactness is pinned by the `watchdog_step` unit
//! truth-table; here the discriminator is the injected wedge shape plus the
//! fast-kill timing.
//!
//! # Out of scope here
//!
//! Starvation-immunity (a runnable-but-host-starved cell must NOT be
//! killed) needs real host oversubscription and is covered at acceptance
//! level by the scheduler-storm harness, not per-test.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::test_support::Scheduler;
use std::time::Duration;

/// EEVDF (no userspace scheduler) plus the cmdline token that makes the
/// guest sleep forever in the Teardown stage — the Tier-2 idle-wedge shape.
const TEARDOWN_IDLE_SCHED: Scheduler =
    Scheduler::named("progress_wd_teardown_idle").kargs(&["KTSTR_FAULT_TEARDOWN_WEDGE=idle"]);

/// EEVDF plus the cmdline token that makes the guest busy-spin forever in
/// the Teardown stage — the Tier-1 spinning-wedge shape.
const TEARDOWN_SPIN_SCHED: Scheduler =
    Scheduler::named("progress_wd_teardown_spin").kargs(&["KTSTR_FAULT_TEARDOWN_WEDGE=spin"]);

/// Upper bound on a Tier-2 idle-wedge kill's wall time. Derivation in the
/// module doc: expected ≈ 40-50 s (wedge start + 15 s backstop ∥ ~30 s
/// worst-case trickle latch); the Tier-3 deadman cannot land under ~150 s.
const TIER2_IDLE_BOUND: Duration = Duration::from_secs(90);

/// Upper bound on a Tier-1 spin-wedge kill's wall time. Expected ≈ 20-27 s
/// (wedge start + ≤12 s widened CPU-budget burn); tighter than the idle
/// bound because Tier-1 needs no trickle windows.
const TIER1_SPIN_BOUND: Duration = Duration::from_secs(60);

/// Host-side gate for the idle (Tier-2) fixture: the injected Teardown
/// idle-wedge MUST time out under the watchdog, and fast enough to prove
/// the Tier-2 backstop — not the ~150 s Tier-3 dead-man — killed it. An
/// `Err` rides `PostVmAssertionFailure` → hard fail, never inverted.
fn assert_tier2_idle_fast_kill(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.timed_out,
        "expected the progress watchdog to fire (timed_out) on the injected \
         Teardown idle-wedge, but the run did not time out — Tier-2 \
         (idle-wedge) did not kill the cell"
    );
    anyhow::ensure!(
        result.duration < TIER2_IDLE_BOUND,
        "Teardown idle-wedge timed out but too slowly ({:.1}s >= {:.1}s \
         bound): the kill looks like the Tier-3 dead-man deadline (~150s+), \
         not the Tier-2 wall backstop + trickle-stall latch (~40-50s)",
        result.duration.as_secs_f64(),
        TIER2_IDLE_BOUND.as_secs_f64(),
    );
    Ok(())
}

/// Host-side gate for the spin (Tier-1) fixture: the injected Teardown
/// spin-wedge MUST time out fast enough to prove the Tier-1 CPU budget —
/// not Tier-3 — killed it.
fn assert_tier1_spin_fast_kill(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.timed_out,
        "expected the progress watchdog to fire (timed_out) on the injected \
         Teardown spin-wedge, but the run did not time out — Tier-1 \
         (CPU-budget) did not kill the cell"
    );
    anyhow::ensure!(
        result.duration < TIER1_SPIN_BOUND,
        "Teardown spin-wedge timed out but too slowly ({:.1}s >= {:.1}s \
         bound): the kill looks like the Tier-3 dead-man deadline (~150s+), \
         not the Tier-1 Teardown CPU budget (~8-12s of in-phase burn)",
        result.duration.as_secs_f64(),
        TIER1_SPIN_BOUND.as_secs_f64(),
    );
    Ok(())
}

/// Host-side gate for the Body-exemption control: a guest that sat fully
/// idle for most of its declared duration in the Body stage MUST survive.
/// An `Err` here rides the `PostVmAssertionFailure` marker → hard fail.
fn assert_idle_body_survived(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        !result.timed_out,
        "a fully-idle Body-stage guest was killed under the watchdog — the \
         Body-stage tier exemption regressed. An idle Body (no CPU, no \
         forward progress) is the textbook Tier-2 idle-wedge shape, but the \
         Body stage's u64::MAX wall backstop must exempt it; a kill here \
         means Tier-2 (or Tier-1) is charging the Body stage"
    );
    Ok(())
}

/// Tier-2 (silent idle-wedge): after publishing its PASS, the guest sleeps
/// forever in the Teardown stage — no runnable demand, ISR-only CPU
/// trickle. The watchdog's Tier-2 rule kills it once the trickle-stall
/// discriminator latches (~2×10s windows) past the 15 s Teardown wall
/// backstop — far below the ~150 s Tier-3 deadman. Green iff the kill
/// lands fast (see `assert_tier2_idle_fast_kill`).
///
/// This fixture is the regression test for the trickle floor's currency
/// calibration: under the pthread currency fallback (any host where
/// `perf_event_paranoid=2` rejects the PMU task-clock open in
/// `monitor::perf_counters`), the pthread clock charges the full
/// VM-exit/host path for every residual idle kernel tick — measured
/// 1-10 ms per 10 s window on this exact idle-wedge shape — which sailed
/// over the original PMU-calibrated 1 ms floor and made the wedge
/// immortal to Tier-2 AND the deadman. The currency-dependent floor
/// (`trickle_floor_for_currency` in `watchdog_step.rs`) is what makes
/// this fixture fire; a revert of that calibration turns this test red
/// at nextest's terminate-after.
#[ktstr_test(
    scheduler = TEARDOWN_IDLE_SCHED,
    llcs = 1,
    cores = 1,
    threads = 1,
    memory_mib = 256,
    no_perf_mode,
    duration_s = 1,
    watchdog_timeout_s = 90,
    auto_repro = false,
    post_vm_unconditional = assert_tier2_idle_fast_kill,
)]
fn teardown_idle_wedge_killed_by_tier2(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

/// Tier-1 (spinning wedge): after publishing its PASS, the guest busy-spins
/// forever in the Teardown stage, burning CPU with no milestone. The
/// watchdog's Tier-1 rule kills it once its `max_vcpu_cpu_in_phase` exceeds
/// the flat 8 s Teardown CPU budget (the lone spinning vCPU IS the max) —
/// earlier than Tier-2's backstop+trickle latch and
/// far below Tier-3. Green iff the kill lands fast (see
/// `assert_tier1_spin_fast_kill`).
#[ktstr_test(
    scheduler = TEARDOWN_SPIN_SCHED,
    llcs = 1,
    cores = 1,
    threads = 1,
    memory_mib = 256,
    no_perf_mode,
    duration_s = 1,
    watchdog_timeout_s = 90,
    auto_repro = false,
    post_vm_unconditional = assert_tier1_spin_fast_kill,
)]
fn teardown_spin_wedge_killed_by_tier1(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

/// Body-stage exemption positive control: a guest that sleeps fully idle —
/// off-CPU, no workers, no runnable demand — for its entire declared
/// duration PASSES, proving the watchdog never kills a test that is
/// legitimately quiescent (e.g. a body blocked on a slow event) rather
/// than wedged. `duration_s` is a large fraction of `watchdog_timeout_s`
/// so the idle body is long relative to the run budget;
/// `watchdog_timeout_s` stays comfortably above `duration_s` so the Tier-3
/// dead-man deadline cannot fire on the idle body either. `no_perf_mode`
/// so it runs on any host without a 1:1-pin requirement; EEVDF (no
/// scheduler) keeps the run minimal.
#[ktstr_test(
    llcs = 1,
    cores = 1,
    threads = 1,
    memory_mib = 256,
    no_perf_mode,
    duration_s = 8,
    watchdog_timeout_s = 30,
    auto_repro = false,
    post_vm_unconditional = assert_idle_body_survived,
)]
fn legit_idle_body_survives_body_exemption(ctx: &Ctx) -> Result<AssertResult> {
    // Sit fully idle (off-CPU, no workers, no runnable demand) for the whole
    // declared duration. In an INFRA stage this is a textbook Tier-2
    // idle-wedge; in the Body stage it is a healthy quiescent test and must
    // survive.
    std::thread::sleep(ctx.duration);
    Ok(AssertResult::pass())
}
