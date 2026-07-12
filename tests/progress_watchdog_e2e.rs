//! End-to-end coverage for the progress watchdog, scoped to what a
//! `#[ktstr_test]` can actually exercise on a live guest.
//!
//! # What this file covers, and what it deliberately does not
//!
//! The progress watchdog has three tiers (`src/vmm/freeze_coord/watchdog_step.rs`):
//!   - **Tier-1** (CPU burned past a phase budget with no progress — a
//!     *spinning* wedge),
//!   - **Tier-2** (no progress past a phase wall backstop with no runnable
//!     demand — a *silent idle* wedge), and
//!   - **Tier-3** (the flat dead-man deadline).
//!
//! Tier-1 and Tier-2 fire ONLY in the INFRA lifecycle stages
//! (Boot / Attach / Dispatch / Teardown); both are structurally disabled in
//! the Body stage via the `u64::MAX` Body budgets in
//! `test_support::runtime::{phase_cpu_budget_ns, phase_wall_backstop_ns}`.
//!
//! ## Tier-1 / Tier-2 are NOT reachable from an e2e fixture
//!
//! A `#[ktstr_test]` runs its body exclusively in the Body stage, and every
//! INFRA stage is framework code with no test-body hook; the guest teardown
//! path additionally bounds each of its waits (scheduler reap, probe
//! finalize, trace drain) well under the tier backstops. So no workload,
//! scenario op, or scheduler fixture can wedge an INFRA stage.
//!
//! A guest-side fault injector (a sibling of the Boot-phase
//! `KTSTR_FAULT_AP_GAP` cmdline hook) that made the guest sleep or spin in
//! the Teardown stage was prototyped and measured — it does NOT trip either
//! tier, and cannot. The tiers key on `progress_evidence`
//! (`src/monitor/reader.rs`), which fires on any positive `ttwu_count` /
//! `sched_count` / `pcount` delta or scx-event / local-DSQ delta. A LIVE
//! guest kernel continuously wakes kthreads (RCU, timer wheel, workqueue)
//! and the ktstr guest's own background poll threads run on a ~200 ms
//! cadence, so `ttwu_count` advances within every tier window and the
//! progress epoch is reset long before the 15 s Teardown / 35 s Dispatch
//! backstops. Measured directly: a 60 s teardown idle-sleep exited cleanly
//! at ~65 s with `timed_out == false` — Tier-2 never fired. A
//! CPU-monopolizing spinner does not help: even a `SCHED_FIFO` task that
//! starves every lower-priority thread cannot stop `ttwu_count` (a wakeup
//! enqueues a task and bumps the counter even when that task never gets the
//! CPU), so Tier-1's since-progress accumulator keeps resetting too.
//!
//! This is the tiers' by-design false-positive protection: they must fire
//! only when the guest makes NO scheduling progress at all — a true
//! guest-kernel hang — which cannot be safely injected from a benign test
//! fixture. Tier-1 and Tier-2 are therefore covered at the UNIT level, where
//! the pure `watchdog_step` core is driven through its full truth table
//! (`src/vmm/freeze_coord/watchdog_step.rs` tests: every {phase-class × CPU ×
//! wall × demand × currency × monitor-liveness} row, including the
//! Tier-1/Tier-2 fire rows, the starvation guard, and the cause-tag
//! rendering). The `cause=tier1-cpu-budget` / `cause=tier2-idle-wedge` dump
//! text is likewise asserted there and in the freeze-coord dump tests; it
//! never reaches the host-visible `VmResult`.
//!
//! ## Starvation-immunity (verification-plan item 4)
//!
//! A runnable-but-host-starved cell must NOT be killed. That needs real host
//! oversubscription, which a single lightweight fixture cannot create, so it
//! is covered at acceptance level by the scheduler-storm harness, not here.
//!
//! ## What this file DOES cover: the Body-stage exemption (item 3)
//!
//! [`legit_idle_body_survives_body_exemption`] is the e2e positive control
//! for the Body exemption: a guest that sits fully idle — the exact
//! no-CPU / no-progress shape that Tier-2 is built to catch in an INFRA
//! stage — for its entire declared duration must PASS, because
//! `phase_wall_backstop_ns(Body) == u64::MAX` disables Tier-2 in the Body
//! stage. A regression that let Tier-2 (or Tier-1) charge the Body stage
//! would kill this healthy quiescent test; the `post_vm_unconditional` gate
//! asserts it survived (`!timed_out`).

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;

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

/// Body-stage exemption positive control (verification-plan item 3): a guest
/// that sleeps fully idle — off-CPU, no workers, no runnable demand — for
/// its entire declared duration PASSES, proving the watchdog never kills a
/// test that is legitimately quiescent (e.g. a body blocked on a slow
/// event) rather than wedged. `duration_s` is a large fraction of
/// `watchdog_timeout_s` so the idle body is long relative to the run budget;
/// `watchdog_timeout_s` stays comfortably above `duration_s` so the Tier-3
/// dead-man deadline cannot fire on the idle body either. `no_perf_mode` so
/// it runs on any host without a 1:1-pin requirement; EEVDF (no scheduler)
/// keeps the run minimal.
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
