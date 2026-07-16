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
//! reasons about a phase's CPU burn and runnable demand:
//!   - **Tier-1** — the busiest vCPU's `max_vcpu_cpu_in_phase` exceeded the
//!     phase's flat `phase_cpu_budget_ns` without reaching a milestone: a
//!     *spinning* wedge.
//!   - **Tier-2** — an INFRA phase sat past its `phase_wall_backstop_ns`
//!     with live evidence channels and no runnable demand: a *silent idle*
//!     wedge. Deliberately NO CPU/trickle conjunct — the runnable conjunct
//!     alone carries the starvation protection (a starved cell WITH work
//!     always shows queued-or-running tasks in its own rq memory), and the
//!     old trickle conjunct was width-broken belt-and-braces (a wide idle
//!     guest's housekeeping burn cleared any 1-vCPU-calibrated floor and
//!     made the wedge immortal). The `*_wide` fixtures (64 vCPUs) are the
//!     regression guard for that.
//!   - **Tier-3** — the dead-man wall deadline, deferring while the cell is
//!     alive (busiest-vCPU CPU trickle above the floor, or a recent
//!     milestone), but only while the current phase remains within the
//!     VM's whole effective-deadline busiest-vCPU CPU budget. This keeps
//!     contention immunity without leaving an active Body livelock to the
//!     outer nextest rail.
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
//! `post_vm_unconditional` callback then asserts the kill happened AND
//! names the expected tier; a callback `Err` rides the
//! `PostVmAssertionFailure` marker, an unconditional hard `EXIT_FAIL`.
//!
//! # Mechanism assertions, not wall bounds
//!
//! These fixtures assert WHICH watchdog rule killed the wedge
//! (`VmResult::watchdog_kill_reason` — the dump's `cause=` line), never
//! how long the kill took on the wall clock. Earlier revisions bounded
//! wall time (`< 90 s` for Tier-2, `< 60 s` for Tier-1) as a proxy for
//! "the fast tier fired, not the deadman" — the disguised-wall-deadline
//! anti-pattern living in the regression tests for the anti-wall-deadline
//! machinery. Real arm64 CI runs falsified the proxy in both directions:
//!
//!   - a 1-vCPU cell at ~14% host CPU share had Tier-1 fire EXACTLY on
//!     its 12 s in-phase CPU budget (`cause=tier1-cpu-budget`,
//!     `max_vcpu_cpu_in_phase=12.07s vs budget=12s`) — the mechanism on
//!     spec — yet took 83 s of wall in-phase and failed the 60 s bound;
//!   - a sibling cell at ~0.3% share sat in `Boot` for 172 s (the guest
//!     never finished booting) and was correctly bounded by the Tier-3
//!     deadman — nothing wedge-related can be asserted about a guest
//!     that never reached the wedge.
//!
//! The verdict logic (`assert_wedge_kill_mechanism`) is therefore:
//!   (a) killed by the expected tier → PASS at any host dilation;
//!   (b) the guest never reached the Teardown wedge phase
//!       (`VmResult::final_guest_phase < Teardown`, the 0.3%-share
//!       shape) → host-side SKIP (`post_vm_skip`): environmental
//!       non-verdict, the injected wedge was never exercised;
//!   (c) reached the wedge but killed by the wrong rule (deadman,
//!       wrong tier, AP kill) → FAIL — the detection regression these
//!       fixtures exist to catch;
//!   (d) not killed at all → FAIL (the watchdog let the wedge escape;
//!       the harness/terminate-after bound is the only outer net).
//!
//! # Timing expectations (documentation only — asserted nowhere)
//!
//! On a healthy, non-oversubscribed host (1 vCPU, `no_perf_mode`,
//! `watchdog_timeout_s = 90`): the Tier-3 dead-man deadline
//! (`vm_timeout_from_entry`) is `max(watchdog_timeout_s, duration_s, 1s)
//! + vm_boot_headroom(1) × 3` = `90s + (10s + 10.15s) × 3` ≈ 150.5 s.
//!   - **Tier-2 idle**: wedge starts at ~10-15 s (boot ≈ 5-10 s + 1 s
//!     body + bounded teardown preamble); Tier-2 fires once
//!     `wall_in_phase > 15s` (the Teardown backstop) → kill ≈ 25-35 s.
//!   - **Tier-1 spin**: the Teardown CPU budget is the flat 8 s (≤ 12 s
//!     pthread-widened 3/2) charged against the MAX per-vCPU in-phase
//!     burn; a 1-vCPU spinner accrues ~1 s CPU per wall-second → kill
//!     ≈ 20-27 s. Under host contention the same CPU budget stretches
//!     across proportionally more wall — by design.
//!
//! # Out of scope here
//!
//! Starvation-immunity (a runnable-but-host-starved cell must NOT be
//! killed) needs real host oversubscription and is covered at acceptance
//! level by the scheduler-storm harness, not per-test.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{
    GuestLifecyclePhase, VmResult, WatchdogKillReason, capture_starvation_witness, post_vm_skip,
};
use ktstr::scenario::Ctx;
use ktstr::test_support::Scheduler;

/// EEVDF (no userspace scheduler) plus the cmdline token that makes the
/// guest sleep forever in the Teardown stage — the Tier-2 idle-wedge shape.
const TEARDOWN_IDLE_SCHED: Scheduler =
    Scheduler::named("progress_wd_teardown_idle").kargs(&["KTSTR_FAULT_TEARDOWN_WEDGE=idle"]);

/// EEVDF plus the cmdline token that makes the guest busy-spin forever in
/// the Teardown stage — the Tier-1 spinning-wedge shape.
const TEARDOWN_SPIN_SCHED: Scheduler =
    Scheduler::named("progress_wd_teardown_spin").kargs(&["KTSTR_FAULT_TEARDOWN_WEDGE=spin"]);

/// Wide-topology (64-vCPU) siblings of the two wedge shapes above. Same
/// fault injectors; the wide topology is the REGRESSION GUARD for Tier-2's
/// width-soundness. When Tier-2 still carried a CPU-trickle conjunct, a
/// wide idle guest's background burn cleared the 1-vCPU-calibrated floor at
/// width (summed: ~64x; even the busiest-single-vCPU windowed accrual reads
/// 20-45 ms/10 s from the housekeeping CPU's timekeeping/RCU duty), so the
/// trickle never latched "stalled" — Tier-2 AND the trickle-gated Tier-3
/// deadman both deferred, and the wide idle wedge escaped to the harness
/// bound. Tier-2 now fires on channels + no-demand + wall backstop alone
/// (the runnable conjunct carries the starvation protection), so the wide
/// idle wedge dies on the same bound as the narrow one. The wide spin wedge
/// stays Tier-1: its lone hot vCPU IS the max-per-vCPU burn, so width does
/// not move Tier-1's flat budget either.
const TEARDOWN_IDLE_WIDE_SCHED: Scheduler =
    Scheduler::named("progress_wd_teardown_idle_wide").kargs(&["KTSTR_FAULT_TEARDOWN_WEDGE=idle"]);
const TEARDOWN_SPIN_WIDE_SCHED: Scheduler =
    Scheduler::named("progress_wd_teardown_spin_wide").kargs(&["KTSTR_FAULT_TEARDOWN_WEDGE=spin"]);

/// Shared mechanism gate for the wedge fixtures — the (a)-(d) verdict
/// logic from the module doc. Asserts WHICH rule killed the run, never
/// how long the kill took (see "Mechanism assertions, not wall bounds").
///
/// Order matters: the never-reached-the-wedge check (b) precedes the
/// expected-tier check (a) so a kill that fired the right rule in the
/// WRONG phase (e.g. Tier-2 on an idle guest still stuck in Boot — Boot
/// is INFRA too, with its own backstop) reads as environmental SKIP
/// rather than a false proof that the injected Teardown wedge was
/// detected. A starved run can also leave the final ledger phase below
/// Teardown when the kill outran the host's consumption of the guest's
/// Teardown frame — likewise a non-verdict, not a regression.
fn assert_wedge_kill_mechanism(
    result: &VmResult,
    expected: WatchdogKillReason,
    wedge_desc: &str,
) -> Result<()> {
    // (d) Not killed at all: the watchdog let a forever-wedge escape to
    // the harness bound — the machinery failed open.
    anyhow::ensure!(
        result.timed_out,
        "expected the progress watchdog to fire (timed_out) on the injected \
         {wedge_desc}, but the run did not time out — the watchdog let the \
         wedge escape"
    );
    // (b) The guest never reached the Teardown wedge phase: the injected
    // fault never executed, so no wedge-detection claim — for or against —
    // is testable. Environmental (observed: an arm64 cell at ~0.3% host
    // CPU share sat in Boot for 172 s; nothing can boot at that share).
    // SKIP, not FAIL: the deadman bounding such a cell is the DESIGNED
    // degradation, not a detection regression.
    if result.final_guest_phase < GuestLifecyclePhase::Teardown {
        return Err(post_vm_skip(format!(
            "guest never reached the Teardown wedge phase \
             (final_guest_phase={:?}, progress_epoch={}, kill={:?} at \
             {:.1}s): the host was too starved for the guest to boot to \
             the injected {wedge_desc} — environmental non-verdict, the \
             wedge mechanism was never exercised",
            result.final_guest_phase,
            result.final_progress_epoch,
            result.watchdog_kill_reason,
            result.duration.as_secs_f64(),
        )));
    }
    // (a) Killed by the expected tier: mechanism proven, at any host
    // dilation — a CPU budget stretches across wall proportionally to
    // starvation by design (observed on-spec at ~14% share: Tier-1 at
    // 12.07s-vs-12s CPU across 83 s of wall).
    if result.watchdog_kill_reason == Some(expected) {
        return Ok(());
    }
    // (a1) Reached the wedge, but the Tier-3 DEADMAN fired instead of the
    // expected tier, UNDER witnessed host contention: the wedge was starved
    // so far below the CPU-accrual rate its tier needs that the wall-clock
    // deadman won the race first. This is the plan's row C — a near-zero-CPU
    // wedge is indistinguishable from a starved-but-healthy cell using guest
    // counters, and the design assigns exactly that case to the deadman. The
    // whole-run dilation reads huge here (a spinner pinned at a few percent
    // host share accrues enormous run_delay vs on_cpu), so the witness fires
    // cleanly. SKIP: the deadman bounding such a wedge is the DESIGNED
    // degradation. A deadman kill on a QUIET host (no witness), or any wrong
    // TIER (Tier-2 for Tier-1, etc.), still falls through to the (c) bail.
    if result.watchdog_kill_reason == Some(WatchdogKillReason::Tier3Deadman)
        && let Some(d) = capture_starvation_witness(result)
    {
        return Err(post_vm_skip(format!(
            "the injected {wedge_desc} was reached but the Tier-3 deadman \
             fired instead of {expected:?} under witnessed host contention \
             (D={d:.2}, duration {:.1}s): the wedge was starved below the \
             CPU-accrual rate its tier needs to win the race to the deadman \
             — environmental non-verdict (row C: the deadman bounding a \
             near-zero-CPU wedge is the designed degradation, not a \
             detection regression)",
            result.duration.as_secs_f64(),
        )));
    }
    // (c) Reached the wedge but the wrong rule killed it (deadman on a quiet
    // host, wrong tier, AP kill, or no reason recorded): the detection
    // regression these fixtures exist to catch.
    anyhow::bail!(
        "the injected {wedge_desc} was reached (final_guest_phase={:?}, \
         progress_epoch={}) but the kill was {:?}, not {expected:?} \
         (duration {:.1}s) — the expected tier failed to detect the wedge",
        result.final_guest_phase,
        result.final_progress_epoch,
        result.watchdog_kill_reason,
        result.duration.as_secs_f64(),
    )
}

/// Host-side gate for the idle (Tier-2) fixtures: the injected Teardown
/// idle-wedge must die to `cause=tier2-idle-wedge`. An `Err` rides
/// `PostVmAssertionFailure` → hard fail, never inverted; the
/// starved-boot shape lands a host-side SKIP instead (see
/// `assert_wedge_kill_mechanism`).
fn assert_tier2_idle_kill(result: &VmResult) -> Result<()> {
    assert_wedge_kill_mechanism(
        result,
        WatchdogKillReason::Tier2IdleWedge,
        "Teardown idle-wedge",
    )
}

/// Host-side gate for the spin (Tier-1) fixtures: the injected Teardown
/// spin-wedge must die to `cause=tier1-cpu-budget`.
fn assert_tier1_spin_kill(result: &VmResult) -> Result<()> {
    assert_wedge_kill_mechanism(
        result,
        WatchdogKillReason::Tier1CpuBudget,
        "Teardown spin-wedge",
    )
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
/// trickle. The watchdog's Tier-2 rule kills it once `wall_in_phase`
/// crosses the 15 s Teardown wall backstop (channels live, nothing
/// runnable). Green iff `cause=tier2-idle-wedge` did the killing (see
/// `assert_tier2_idle_kill`); wall time is documentation, not verdict.
///
/// This fixture pins the Tier-2 conjuncts end-to-end: the guest's
/// evidence channels are live and its rq memory reports nothing
/// queued-or-running, so the idle-in-INFRA wedge dies on the wall backstop
/// alone — no CPU/trickle term can defer it (the old trickle conjunct
/// deferred exactly this shape whenever the measured idle trickle cleared
/// the floor, and was width-broken besides — see `watchdog_step.rs`).
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
    post_vm_unconditional = assert_tier2_idle_kill,
)]
fn teardown_idle_wedge_killed_by_tier2(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

/// Tier-1 (spinning wedge): after publishing its PASS, the guest busy-spins
/// forever in the Teardown stage, burning CPU with no milestone. The
/// watchdog's Tier-1 rule kills it once its `max_vcpu_cpu_in_phase` exceeds
/// the flat 8 s Teardown CPU budget (the lone spinning vCPU IS the max) —
/// earlier than Tier-2's 15 s wall backstop. Green iff
/// `cause=tier1-cpu-budget` did the killing (see `assert_tier1_spin_kill`);
/// under host contention the same CPU budget takes proportionally more
/// wall, which is correct and unasserted.
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
    post_vm_unconditional = assert_tier1_spin_kill,
)]
fn teardown_spin_wedge_killed_by_tier1(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

/// Tier-2 idle wedge at WIDTH (8 llcs × 8 cores = 64 vCPUs) — THE
/// regression test for Tier-2's width-soundness. When Tier-2 still carried
/// a CPU-trickle conjunct, this exact shape was immortal: the 64-vCPU idle
/// guest's background burn cleared the 1-vCPU-calibrated floor at width
/// (measured 20-45 ms per 10 s window on the busiest vCPU alone — the
/// housekeeping CPU's timekeeping/RCU duty scales with vCPU count), so the
/// trickle never latched "stalled" and Tier-2 AND the trickle-gated Tier-3
/// deadman both deferred — the wedge escaped to the harness bound. Tier-2
/// now needs only channels + no-demand + the 15 s Teardown wall backstop,
/// so this wedge dies on the same rule as the 1-vCPU fixture — asserted
/// via `cause=tier2-idle-wedge`, not wall time.
#[ktstr_test(
    scheduler = TEARDOWN_IDLE_WIDE_SCHED,
    llcs = 8,
    cores = 8,
    threads = 1,
    memory_mib = 256,
    no_perf_mode,
    duration_s = 1,
    watchdog_timeout_s = 90,
    auto_repro = false,
    post_vm_unconditional = assert_tier2_idle_kill,
)]
fn teardown_idle_wedge_wide_killed_by_tier2(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

/// Tier-1 spin wedge at WIDTH (64 vCPUs): the control proving width does not
/// move Tier-1. The `spin` fault burns CPU on a SINGLE vCPU, which IS the
/// max-per-vCPU in-phase burn, so Tier-1 fires on its flat Teardown CPU
/// budget exactly as at 1 vCPU — the other 63 idle vCPUs sum into
/// `cpu_ns_now` but never into the width-independent Tier-1 evidence.
/// Asserted via `cause=tier1-cpu-budget`, not wall time.
#[ktstr_test(
    scheduler = TEARDOWN_SPIN_WIDE_SCHED,
    llcs = 8,
    cores = 8,
    threads = 1,
    memory_mib = 256,
    no_perf_mode,
    duration_s = 1,
    watchdog_timeout_s = 90,
    auto_repro = false,
    post_vm_unconditional = assert_tier1_spin_kill,
)]
fn teardown_spin_wedge_wide_killed_by_tier1(_ctx: &Ctx) -> Result<AssertResult> {
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
