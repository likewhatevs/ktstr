//! Pure kill-decision core for the progress watchdog.
//!
//! [`watchdog_step`] is the safety heart of the progress-watchdog: given
//! a snapshot of one lifecycle phase's progress evidence, it decides
//! whether that phase has wedged badly enough to evict the cell. It is a
//! pure function of its arguments — no clocks, no atomics, no I/O — so
//! the full kill truth-table can be exercised deterministically in unit
//! tests (the `forge_step` precedent in `monitor::reader`). The watchdog
//! wires it to live [`crate::monitor::ProgressLedger`] reads in a later
//! commit; nothing calls it yet.
//!
//! Two tiers live here:
//!   - Tier-1 (CPU budget): the phase burned more guest CPU than its
//!     [`phase_cpu_budget_ns`] without a progress epoch — a *spinning*
//!     wedge.
//!   - Tier-2 (idle wedge): the phase made no progress for longer than
//!     its [`phase_wall_backstop_ns`] AND the guest shows no runnable
//!     demand — a *silent* wedge.
//!
//! Tier-3 (the guest-derived hard deadline) and the AP-kill path stay in
//! the existing watchdog deadline logic — this module is only the two
//! progress-derived tiers.
#![allow(dead_code)]

use crate::monitor::{CPU_CURRENCY_NONE, CPU_CURRENCY_PTHREAD};
use crate::test_support::runtime::{phase_cpu_budget_ns, phase_wall_backstop_ns};

/// Lifecycle-phase *class* the watchdog treats a phase as, decoupled
/// from the phase id itself. Tier-2 (the wall-time idle backstop) only
/// applies to INFRA phases — Boot/Attach/Dispatch/Teardown, where a lack
/// of forward progress is unambiguously a wedge. A BODY phase is the
/// running workload, which may legitimately sit quiescent, so Tier-2 is
/// off for it. C4 maps [`crate::monitor::LifecycleStage`] onto this;
/// keeping it a separate input lets C3 stay independent of that enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PhaseClass {
    /// Boot / Attach / Dispatch / Teardown — no-progress is a wedge.
    Infra,
    /// The running workload — may sit quiescent; Tier-2 does not apply.
    Body,
}

/// The two progress-tier kill verdicts this module can return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KillDecision {
    /// No progress-tier kill. (Tier-3/AP-kill may still fire elsewhere.)
    None,
    /// Tier-1: guest CPU burned past the phase budget with no progress —
    /// a spinning wedge.
    Tier1CpuBudget,
    /// Tier-2: no progress past the phase wall backstop with no runnable
    /// demand — a silent idle wedge.
    Tier2IdleWedge,
}

/// A guest that is *runnable but starved of host CPU* accrues no
/// meaningful guest CPU time. If it has burned less than this much CPU
/// since the last progress epoch while still showing runnable demand,
/// Tier-2 must NOT kill it: the cell is healthy but host-starved, and
/// evicting it would punish a scheduling victim rather than a wedge. 10
/// ms is comfortably above measurement noise on the per-vCPU CPU-time
/// sources yet far below any real phase budget, so it cleanly separates
/// "made essentially no progress because it got no CPU" from "sat on CPU
/// and still made no progress".
const TIER2_CPU_EPSILON_NS: u64 = 10_000_000;

/// Widen a CPU budget for the trust level of its currency. PMU cycles
/// (`exclude_host`) measure guest CPU directly, so the budget is used
/// as-is. Per-vCPU pthread CPU time additionally charges VM-exit /
/// host-side overhead to the guest, inflating the apparent CPU burn by a
/// documented margin; widen the budget by 3/2 (integer `b + b/2`) so
/// that overhead does not push a healthy phase over its Tier-1 budget.
/// Saturating so a `u64::MAX` (Body sentinel) budget cannot wrap.
const fn widen_budget_for_currency(budget: u64, cpu_currency: u8) -> u64 {
    match cpu_currency {
        // pthread time includes VM-exit overhead — degrade the budget by
        // widening it 3/2 (integer math).
        CPU_CURRENCY_PTHREAD => budget.saturating_add(budget / 2),
        // PMU (or any other trusted source) measures guest cycles
        // directly — no widening.
        _ => budget,
    }
}

/// Decide whether a single lifecycle phase has wedged, from a snapshot
/// of its progress evidence. Pure — see the module doc.
///
/// Arguments:
///   - `phase`: the [`crate::monitor::LifecycleStage`] discriminant
///     (raw `u8` so this fn carries no dependency on that enum).
///   - `class`: [`PhaseClass::Infra`] vs [`PhaseClass::Body`] — gates
///     Tier-2.
///   - `cpu_since_progress_ns`: guest CPU burned since the last progress
///     epoch in this phase.
///   - `wall_since_progress_ns`: wall time since the last progress epoch.
///   - `runnable_demand`: the guest had runnable-but-not-running tasks —
///     evidence it *wants* CPU it isn't getting.
///   - `monitor_live`: the monitor thread is producing fresh ledger
///     writes.
///   - `cpu_currency`: provenance of `cpu_since_progress_ns`
///     ([`CPU_CURRENCY_NONE`]/`PTHREAD`/`PMU`).
///   - `vcpus`: guest vCPU count, scaling the per-phase CPU budget.
pub(crate) fn watchdog_step(
    phase: u8,
    class: PhaseClass,
    cpu_since_progress_ns: u64,
    wall_since_progress_ns: u64,
    runnable_demand: bool,
    monitor_live: bool,
    cpu_currency: u8,
    vcpus: u32,
) -> KillDecision {
    // A dead monitor invalidates BOTH tiers' evidence. Tier-1's CPU
    // signal comes from monitor ledger writes, so a stale monitor makes
    // it untrustworthy. Tier-2 is subtler: `wall_since_progress_ns`
    // measures time since the last *progress epoch*, and progress epochs
    // are only refreshed by monitor/ledger writes — with a dead monitor
    // the epoch can never advance, so `wall_since_progress_ns` grows
    // unboundedly even for a perfectly healthy guest, and Tier-2 would
    // false-fire. Suppress both while the monitor is not live.
    if !monitor_live {
        return KillDecision::None;
    }

    // Tier-1: spinning wedge — guest CPU past the phase budget with no
    // progress. Only valid when the CPU signal has a trusted currency;
    // with CPU_CURRENCY_NONE there is no per-vCPU CPU-time source this
    // tick, so the CPU measurement is meaningless and Tier-1 is off
    // (Tier-2 below is still valid — progress evidence needs no CPU
    // time). Body's budget is the `u64::MAX` sentinel, so Tier-1 is
    // structurally off for Body via the budget table, not a class check.
    if cpu_currency != CPU_CURRENCY_NONE {
        let budget = widen_budget_for_currency(
            phase_cpu_budget_ns(phase, vcpus),
            cpu_currency,
        );
        // Strict `>`: a phase that burned *exactly* its budget has not
        // yet exceeded it.
        if cpu_since_progress_ns > budget {
            return KillDecision::Tier1CpuBudget;
        }
    }

    // Tier-2: silent idle wedge — no progress past the phase wall
    // backstop. Only INFRA phases qualify (a BODY phase may legitimately
    // sit quiescent). Body's backstop is the `u64::MAX` sentinel too, so
    // even a mislabeled Body can never trip the wall comparison.
    //
    // Starvation guard: a guest that is runnable but has accrued
    // essentially no CPU (< TIER2_CPU_EPSILON_NS) is a host-scheduling
    // victim, not a wedge — killing it would evict a healthy starved
    // cell. Suppress Tier-2 in exactly that case. A fully quiesced guest
    // (no runnable demand) is a real idle wedge and DOES fire. A
    // runnable guest that DID burn CPU (>= epsilon) without progress has
    // already spun on-CPU without advancing; letting Tier-2 fire there
    // is redundant with Tier-1 (which usually fires first on the larger
    // CPU budget) and harmless — the design's table only exempts the
    // low-CPU + runnable row.
    let starved_but_runnable =
        runnable_demand && cpu_since_progress_ns < TIER2_CPU_EPSILON_NS;
    if class == PhaseClass::Infra
        && wall_since_progress_ns > phase_wall_backstop_ns(phase)
        && !starved_but_runnable
    {
        return KillDecision::Tier2IdleWedge;
    }

    KillDecision::None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::CPU_CURRENCY_PMU;

    // Lifecycle discriminants (mirror `monitor::LifecycleStage`; C3 is
    // written against the raw ids so it does not depend on that enum).
    const BOOT: u8 = 0;
    const BODY: u8 = 3;

    const S: u64 = 1_000_000_000;

    // Boot(0), 1 vCPU: cpu budget 8_400 ms, wall backstop 45 s.
    fn boot_budget_ns() -> u64 {
        phase_cpu_budget_ns(BOOT, 1)
    }
    fn boot_backstop_ns() -> u64 {
        phase_wall_backstop_ns(BOOT)
    }

    // -- Row A: HIGH C + runnable (spinning wedge) --

    #[test]
    fn row_a_infra_high_cpu_runnable_fires_tier1() {
        // Way over the Boot CPU budget, still runnable, wall irrelevant
        // (0). PMU currency (no widening). Spinning wedge → Tier-1.
        let d = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            boot_budget_ns() + 100 * S,
            0,
            true,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(d, KillDecision::Tier1CpuBudget);
    }

    #[test]
    fn row_a_body_high_cpu_never_fires() {
        // Same spinning shape but the BODY phase: its CPU budget is the
        // u64::MAX sentinel, so `cpu > budget` is unsatisfiable — a body
        // test is *supposed* to burn CPU.
        let d = watchdog_step(
            BODY,
            PhaseClass::Body,
            u64::MAX - 1,
            0,
            true,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(d, KillDecision::None);
    }

    // -- Row B: HIGH C + no runnable demand --

    #[test]
    fn row_b_infra_high_cpu_no_demand_fires_tier1() {
        let d = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            boot_budget_ns() + 100 * S,
            0,
            false,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(d, KillDecision::Tier1CpuBudget);
    }

    // -- Row C: LOW C + runnable + wall exceeded (starved, healthy) --

    #[test]
    fn row_c_infra_low_cpu_runnable_does_not_fire() {
        // No meaningful CPU (< epsilon) but runnable and long past the
        // wall backstop: a host-starved but healthy cell. Neither tier
        // may fire — Tier-1 (below budget) nor Tier-2 (starvation
        // guard).
        let d = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            5_000_000, // 5 ms < TIER2_CPU_EPSILON_NS
            boot_backstop_ns() + S,
            true,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(d, KillDecision::None);
    }

    // -- Row D: LOW C + no runnable demand + wall exceeded (idle wedge) --

    #[test]
    fn row_d_infra_idle_wedge_fires_tier2() {
        let d = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            0,
            boot_backstop_ns() + S,
            false,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(d, KillDecision::Tier2IdleWedge);
    }

    #[test]
    fn row_d_body_idle_never_fires() {
        // Same idle shape but BODY: the wall backstop is the u64::MAX
        // sentinel AND the class is not Infra — either alone suppresses
        // Tier-2.
        let d = watchdog_step(
            BODY,
            PhaseClass::Body,
            0,
            100 * S,
            false,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(d, KillDecision::None);
    }

    // -- Starvation invariant: cpu==0 is never a Tier-1 kill --

    #[test]
    fn zero_cpu_is_never_tier1_at_any_wall() {
        // Runnable, zero CPU, wall far past the backstop: Row-C-shaped —
        // must stay None (starved healthy cell), never Tier-1.
        for wall in [0, boot_backstop_ns() + S, 100 * S] {
            let d = watchdog_step(
                BOOT,
                PhaseClass::Infra,
                0,
                wall,
                true,
                true,
                CPU_CURRENCY_PMU,
                1,
            );
            assert_eq!(d, KillDecision::None, "wall={wall}");
        }
    }

    // -- Monitor liveness gate: dead monitor suppresses both tiers --

    #[test]
    fn dead_monitor_suppresses_everything() {
        // Both a Tier-1-shaped input (high CPU) and a Tier-2-shaped one
        // (idle wedge) must return None while the monitor is not live.
        let tier1_shape = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            boot_budget_ns() + 100 * S,
            0,
            false,
            false, // monitor dead
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(tier1_shape, KillDecision::None);

        let tier2_shape = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            0,
            boot_backstop_ns() + 100 * S,
            false,
            false, // monitor dead
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(tier2_shape, KillDecision::None);
    }

    // -- Currency split: NONE turns Tier-1 off, leaves Tier-2 on --

    #[test]
    fn currency_none_disables_tier1_only() {
        // High CPU but no trusted currency AND wall below the backstop
        // (so Tier-2 can't fire): Tier-1 is off → None.
        let tier1_off = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            boot_budget_ns() + 100 * S,
            0,
            false,
            true,
            CPU_CURRENCY_NONE,
            1,
        );
        assert_eq!(tier1_off, KillDecision::None);

        // Idle wedge with NONE currency: Tier-2 needs no CPU signal, so
        // it still fires (monitor live, no runnable demand, wall past
        // the backstop).
        let tier2_on = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            0,
            boot_backstop_ns() + S,
            false,
            true,
            CPU_CURRENCY_NONE,
            1,
        );
        assert_eq!(tier2_on, KillDecision::Tier2IdleWedge);
    }

    // -- pthread widening: (budget, 1.5·budget] fires only under PMU --

    #[test]
    fn pthread_widening_band_fires_only_under_pmu() {
        let budget = boot_budget_ns();
        // A CPU burn strictly between the raw budget and 1.5× it. Wall
        // 0 and no demand so only Tier-1 is in play.
        let cpu = budget + budget / 4; // within (budget, 1.5·budget)
        let pmu = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            cpu,
            0,
            false,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(
            pmu,
            KillDecision::Tier1CpuBudget,
            "PMU currency uses the raw budget — this burn is over it"
        );
        let pthread = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            cpu,
            0,
            false,
            true,
            CPU_CURRENCY_PTHREAD,
            1,
        );
        assert_eq!(
            pthread,
            KillDecision::None,
            "pthread widens the budget 3/2 — this burn is under the widened budget"
        );
    }

    #[test]
    fn pthread_widening_fires_above_widened_budget() {
        let budget = boot_budget_ns();
        // Above 1.5× the raw budget — over even the widened pthread
        // budget, so Tier-1 fires under pthread currency too.
        let cpu = budget + budget / 2 + S;
        let pthread = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            cpu,
            0,
            false,
            true,
            CPU_CURRENCY_PTHREAD,
            1,
        );
        assert_eq!(pthread, KillDecision::Tier1CpuBudget);
    }

    // -- Budget boundary is strict `>` --

    #[test]
    fn tier1_budget_boundary_is_exclusive() {
        let budget = boot_budget_ns(); // PMU: no widening
        // Exactly at budget → not over → no fire.
        let at = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            budget,
            0,
            false,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(at, KillDecision::None, "cpu == budget is not over budget");
        // One ns over → fire.
        let over = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            budget + 1,
            0,
            false,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(over, KillDecision::Tier1CpuBudget);
    }

    // -- Tier-2 starvation guard hinges on the CPU epsilon, not merely
    //    on runnable demand: a runnable guest that DID burn CPU (>=
    //    epsilon) without progress is not exempt (the design's redundant-
    //    but-harmless case). --

    #[test]
    fn tier2_runnable_above_epsilon_still_fires() {
        let d = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            TIER2_CPU_EPSILON_NS + 1, // just over epsilon, still << budget
            boot_backstop_ns() + S,
            true, // runnable, but not starved (burned CPU)
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(d, KillDecision::Tier2IdleWedge);
    }

    // -- Unknown phase id: never kill --

    #[test]
    fn unknown_phase_never_fires() {
        // Both budget and backstop are the u64::MAX sentinel for an
        // unmodeled phase, so no shape of input can trip either tier.
        let d = watchdog_step(
            7,
            PhaseClass::Infra,
            u64::MAX - 1,
            u64::MAX - 1,
            false,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(d, KillDecision::None);
    }
}
