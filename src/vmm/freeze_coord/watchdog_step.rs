//! Pure kill-decision core for the progress watchdog.
//!
//! [`watchdog_step`] is the safety heart of the progress-watchdog: given
//! a snapshot of one lifecycle phase's evidence, it decides whether that
//! phase has wedged badly enough to evict the cell. It is a pure function
//! of its arguments — no clocks, no atomics, no I/O — so the full kill
//! truth-table can be exercised deterministically in unit tests (the
//! `forge_step` precedent in `monitor::reader`).
//!
//! PROGRESS IS MILESTONE-ONLY (see
//! [`crate::monitor::ProgressLedger::progress_epoch`]): kernel scheduling
//! noise (ttwu/sched_count/pcount/scx events) ticks every 100 ms on any
//! live guest even under a wedged scheduler, so it is NEVER progress. The
//! watchdog therefore does not lean on per-tick progress evidence; it
//! reasons about a phase's CPU burn and CPU trickle instead:
//!   - Tier-1 (CPU budget): the phase burned more guest CPU IN-PHASE than
//!     its [`phase_cpu_budget_ns`] without reaching its milestone — a
//!     *spinning* wedge.
//!   - Tier-2 (idle wedge): an INFRA phase sat past its
//!     [`phase_wall_backstop_ns`] with no runnable demand AND its summed
//!     CPU trickle-stalled (below the ISR-noise floor) — a *silent* wedge.
//!   - Tier-3 (the guest-derived hard deadline) lives in the watchdog
//!     thread ([`super`]); [`deadman_should_fire`] here is its deferral
//!     gate. It fires at the wall deadline only when the monitor is dead
//!     or the cell is inert (CPU trickle-stalled AND no milestone within
//!     the grace) — so a merely starved-but-alive cell outlives the wall
//!     deadline by design and a userspace-wedged cell on a live kernel is
//!     still bounded.
//!
//! The CPU-trickle discriminator ([`CpuTrickleTracker`]) is the load-
//! bearing new idea: a starved-but-alive cell (even at ~40x host dilation)
//! still accrues tens of ms/s of guest CPU, while an idle/wedged guest
//! accrues only microseconds/s of parked-vCPU ISR time — orders of
//! magnitude apart, so a fixed floor separates them cleanly.

use crate::monitor::{CPU_CURRENCY_NONE, CPU_CURRENCY_PTHREAD, LedgerSnapshot, LifecycleStage, StageClass};
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
    /// Tier-2: INFRA phase past its wall backstop, no runnable demand, and
    /// CPU trickle-stalled — a silent idle wedge.
    Tier2IdleWedge,
}

/// Trailing window over which [`CpuTrickleTracker`] measures the guest's
/// summed CPU accrual. 10 s is long enough to average out the 100 ms tick
/// jitter and the bursty ISR pattern on parked vCPUs, yet short enough
/// that the Tier-2 / Tier-3 stall verdicts refresh several times inside
/// the 45 s wall backstop / 60 s deadman grace.
const TRICKLE_STALL_WINDOW_NS: u64 = 10_000_000_000;

/// Minimum summed guest CPU (ns) that must accrue over one
/// [`TRICKLE_STALL_WINDOW_NS`] for the guest to count as "receiving CPU"
/// (NOT trickle-stalled). 1 ms / 10 s discriminates the two populations
/// the watchdog must tell apart: an idle-or-wedged guest still takes timer
/// interrupts on its parked vCPUs and accrues only microseconds/s (tens of
/// µs over the window — below the floor → stalled), while even a cell
/// starved at ~40x host dilation accrues tens of ms/s (hundreds of ms over
/// the window — far above the floor → not stalled). The floor sits in the
/// wide gap between them, so it never misjudges a starved-but-alive cell
/// as wedged nor an idle wedge as alive.
const TRICKLE_FLOOR_NS: u64 = 1_000_000;

/// Widen a CPU budget for the trust level of its currency. The PMU
/// SW task-clock (`exclude_host`) measures guest-only CPU time directly,
/// so the budget is used as-is. Per-vCPU pthread CPU time additionally
/// charges VM-exit / host-side overhead to the guest, inflating the
/// apparent CPU burn by a documented margin; widen the budget by 3/2
/// (integer `b + b/2`) so that overhead does not push a healthy phase
/// over its Tier-1 budget. Saturating so a `u64::MAX` (Body sentinel)
/// budget cannot wrap.
const fn widen_budget_for_currency(budget: u64, cpu_currency: u8) -> u64 {
    match cpu_currency {
        // pthread time includes VM-exit overhead — degrade the budget by
        // widening it 3/2 (integer math).
        CPU_CURRENCY_PTHREAD => budget.saturating_add(budget / 2),
        // PMU (or any other trusted source) measures guest-only CPU time
        // directly — no widening.
        _ => budget,
    }
}

/// Decide whether a single lifecycle phase has wedged, from a snapshot
/// of its evidence. Pure — see the module doc.
///
/// Arguments:
///   - `phase`: the [`crate::monitor::LifecycleStage`] discriminant
///     (raw `u8` so this fn carries no dependency on that enum).
///   - `class`: [`PhaseClass::Infra`] vs [`PhaseClass::Body`] — gates
///     Tier-2.
///   - `cpu_in_phase_ns`: guest CPU burned since this phase was entered
///     (the last milestone). Since progress is milestone-only and a phase
///     entry IS a milestone, this is also "CPU since the last milestone".
///   - `wall_in_phase_ns`: wall time since this phase was entered (the
///     last milestone).
///   - `runnable_demand`: the guest had runnable-but-not-running tasks —
///     an independent Tier-2 exempt (a runnable cell is not an idle wedge;
///     if it is also stalled the Tier-3 deadman bounds it instead).
///   - `channels_live`: the per-CPU GUEST evidence channels resolved this
///     tick (see [`crate::monitor::ProgressLedger::evidence_channels_live`]).
///     Gates Tier-2 — `runnable_demand` is only meaningful when true.
///   - `cpu_trickle_stalled`: the guest's summed CPU accrued below
///     [`TRICKLE_FLOOR_NS`] over the last [`TRICKLE_STALL_WINDOW_NS`]
///     ([`CpuTrickleTracker`]) — i.e. it is receiving essentially no CPU.
///   - `monitor_live`: the monitor thread is producing fresh ledger
///     writes.
///   - `cpu_currency`: provenance of `cpu_in_phase_ns`
///     ([`CPU_CURRENCY_NONE`]/`PTHREAD`/`PMU`).
///   - `vcpus`: guest vCPU count, scaling the per-phase CPU budget.
#[allow(clippy::too_many_arguments)]
pub(crate) fn watchdog_step(
    phase: u8,
    class: PhaseClass,
    cpu_in_phase_ns: u64,
    wall_in_phase_ns: u64,
    runnable_demand: bool,
    channels_live: bool,
    cpu_trickle_stalled: bool,
    monitor_live: bool,
    cpu_currency: u8,
    vcpus: u32,
) -> KillDecision {
    // A dead monitor invalidates BOTH tiers' evidence. Tier-1's CPU signal
    // comes from monitor ledger writes, so a stale monitor makes it
    // untrustworthy. Tier-2's `wall_in_phase_ns` measures time since the
    // last milestone, and milestones are published through the same ledger
    // — with a dead monitor the anchor can never advance, so the wall
    // delta grows unboundedly even for a healthy guest and Tier-2 would
    // false-fire. Suppress both while the monitor is not live; the Tier-3
    // deadman ([`deadman_should_fire`]) takes over that case.
    if !monitor_live {
        return KillDecision::None;
    }

    // Tier-1: spinning wedge — the phase burned its CPU budget IN-PHASE
    // without reaching its milestone. Only valid when the CPU signal has a
    // trusted currency; with CPU_CURRENCY_NONE there is no per-vCPU
    // CPU-time source this tick, so the measurement is meaningless and
    // Tier-1 is off (Tier-2 below needs no CPU currency — it reasons about
    // the trickle bool). Body's budget is the `u64::MAX` sentinel, so
    // Tier-1 is structurally off for Body via the budget table, not a
    // class check.
    if cpu_currency != CPU_CURRENCY_NONE {
        let budget = widen_budget_for_currency(phase_cpu_budget_ns(phase, vcpus), cpu_currency);
        // Strict `>`: a phase that burned *exactly* its budget has not yet
        // exceeded it.
        if cpu_in_phase_ns > budget {
            return KillDecision::Tier1CpuBudget;
        }
    }

    // Tier-2: silent idle wedge — an INFRA phase sat past its wall
    // backstop with no runnable demand while its CPU trickle-stalled. Only
    // INFRA phases qualify (a BODY phase may legitimately sit quiescent);
    // Body's backstop is the `u64::MAX` sentinel too, so a mislabeled Body
    // can never trip the wall comparison. Every conjunct is load-bearing:
    //   - `channels_live`: absence of demand is only EVIDENCE when the
    //     demand channel exists. Before the guest brings up its per-CPU
    //     runqueue structures the monitor reads nothing, so
    //     `runnable_demand` reads false purely because the channel is
    //     blind — the acceptance storm's 33 false kills at the Boot
    //     backstop. Gating on `channels_live` suppresses Tier-2 in that
    //     blind window.
    //   - `!runnable_demand`: a runnable guest is not an idle wedge (it
    //     wants CPU it may just not be getting). Kept as an independent
    //     exempt; if such a cell is also stalled, the Tier-3 deadman
    //     bounds it rather than Tier-2.
    //   - `cpu_trickle_stalled`: the discriminator between a host-starved
    //     but ALIVE cell (accruing tens of ms/s → not stalled → NOT
    //     killed) and a truly idle/wedged guest (ISR-only µs/s → stalled).
    //     This supersedes the old fixed CPU-epsilon guard.
    if class == PhaseClass::Infra
        && channels_live
        && !runnable_demand
        && cpu_trickle_stalled
        && wall_in_phase_ns > phase_wall_backstop_ns(phase)
    {
        return KillDecision::Tier2IdleWedge;
    }

    KillDecision::None
}

/// The widened Tier-1 CPU budget (ns) the watchdog charges `phase` at
/// `vcpus` under CPU-time provenance `cpu_currency` — the SAME number
/// [`watchdog_step`] compares `cpu_in_phase_ns` against. Exposed so the
/// watchdog's failure dump can render `cpu_in_phase vs budget` against the
/// effective (currency-widened) budget rather than the raw
/// [`phase_cpu_budget_ns`].
pub(crate) fn widened_cpu_budget_ns(phase: u8, vcpus: u32, cpu_currency: u8) -> u64 {
    widen_budget_for_currency(phase_cpu_budget_ns(phase, vcpus), cpu_currency)
}

/// Fold one live [`LedgerSnapshot`] into a [`KillDecision`]: the glue
/// between the ledger and the pure [`watchdog_step`]. Derives the two
/// in-phase deltas and maps the lifecycle stage onto a [`PhaseClass`],
/// then defers to `watchdog_step`. Kept pure (the wall clock arrives as
/// `now_wall_ns`, the trickle verdict as `cpu_trickle_stalled`) so the
/// ledger-read → tier-decision path is unit-testable without a live
/// watchdog thread.
///
/// `now_wall_ns` is `run_start`-relative — the SAME `run_start` the
/// monitor/dispatch anchor `wall_ns_at_progress` to — so `wall_in_phase`
/// aligns. Because progress is milestone-only and a phase entry IS a
/// milestone, `cpu_ns_at_phase` and the milestone anchor `cpu_ns_at_progress`
/// coincide; this uses `cpu_ns_at_phase` for the CPU delta and the
/// milestone anchor `wall_ns_at_progress` for the wall delta.
///
/// INITIAL STATE: before the first milestone both anchors are 0, so
/// `cpu_in_phase == cpu_ns_now` and `wall_in_phase == now_wall_ns`. Boot's
/// budgets thus count from run start — intended (Boot owes its first
/// milestone). Saturating subtraction guards the benign torn read where a
/// fresher anchor races an older `now`/`cpu_now` (clamps to 0 — more
/// conservative, never a spurious kill).
pub(crate) fn evaluate_progress(
    snap: &LedgerSnapshot,
    now_wall_ns: u64,
    cpu_trickle_stalled: bool,
    monitor_live: bool,
    vcpus: u32,
) -> KillDecision {
    let cpu_in_phase = snap.cpu_ns_now.saturating_sub(snap.cpu_ns_at_phase);
    let wall_in_phase = now_wall_ns.saturating_sub(snap.wall_ns_at_progress);
    let class = match LifecycleStage::from_u8(snap.phase).class() {
        StageClass::Body => PhaseClass::Body,
        StageClass::Infra => PhaseClass::Infra,
    };
    watchdog_step(
        snap.phase,
        class,
        cpu_in_phase,
        wall_in_phase,
        snap.runnable_demand,
        snap.evidence_channels_live,
        cpu_trickle_stalled,
        monitor_live,
        snap.cpu_currency,
        vcpus,
    )
}

/// Grace window for the Tier-3 deadman (the guest-derived hard deadline):
/// once the wall deadline has elapsed, the deadman fires ONLY IF the
/// monitor is dead OR the cell is inert — CPU trickle-stalled AND no
/// milestone reached within this window.
///
/// TERMINATION STORY (exactly this shape):
///   - A starved-but-alive cell keeps accruing guest CPU (tens of ms/s
///     even at ~40x dilation), so it is NOT trickle-stalled and is
///     DEFERRED past the wall deadline by design. Its outer bound becomes
///     the harness / operator (nextest `terminate-after`), NOT this
///     deadman — a slow cell is not a wedged cell.
///   - An idle userspace wedge on a LIVE kernel (the classic PID-1-blocked
///     shape) trickles below the floor and reaches no milestone, so the
///     deadman fires at the wall deadline — BOUNDED. (Milestone-only
///     progress is what makes this work: kernel scheduling noise would
///     otherwise reset a wall-since-progress clock every tick and defer
///     forever.)
///   - A runnable-piled-up stalled-scx cell has `runnable_demand=true`, so
///     Tier-2 exempts it — but its CPU also trickle-stalls (tasks never
///     run), so this deadman bounds it at the wall deadline.
///   - A spinning wedge never reaches here: Tier-1 bounds it on the CPU
///     budget long before the deadline.
///
/// 60 s is generous against the 100 ms watchdog tick and any legitimate
/// quiet span (so a live cell is never misjudged), yet small against the
/// multi-minute deadlines it defers.
pub(crate) const TIER3_PROGRESS_GRACE_NS: u64 = 60_000_000_000;

/// Decide whether the Tier-3 deadman should FIRE now that the hard wall
/// deadline has already elapsed. Pure — the watchdog supplies the live
/// values (see the caller in `freeze_coord`).
///
///   - `monitor_live`: the monitor is producing fresh ledger writes. When
///     false the progress machinery is dead, so nothing can defer — fire.
///   - `wall_since_milestone_ns`: wall time since the last MILESTONE
///     (`progress_epoch` anchor). Milestone-only, so a live kernel's
///     scheduling noise does NOT reset it.
///   - `cpu_trickle_stalled`: the guest is receiving essentially no CPU
///     ([`CpuTrickleTracker`]). CPU still trickling above the floor = the
///     guest is alive and merely slow → defer.
///
/// Fires iff the monitor is dead, OR the cell is inert: CPU trickle-stalled
/// AND no milestone within the full [`TIER3_PROGRESS_GRACE_NS`].
pub(crate) fn deadman_should_fire(
    monitor_live: bool,
    wall_since_milestone_ns: u64,
    cpu_trickle_stalled: bool,
) -> bool {
    !monitor_live
        || (cpu_trickle_stalled && wall_since_milestone_ns > TIER3_PROGRESS_GRACE_NS)
}

/// Watchdog-thread-local tracker that turns the ledger's monotone summed
/// `cpu_ns_now` into a boolean "the guest is receiving essentially no CPU"
/// (trickle-stalled), the discriminator Tier-2 and the Tier-3 deadman both
/// rely on. Pure state machine — no clocks, no atomics — so its window
/// arithmetic is unit-testable in isolation; the watchdog feeds it
/// `now_ns` (a `run_start`-relative wall reading) each tick.
///
/// Measures accrual over a trailing [`TRICKLE_STALL_WINDOW_NS`]: it holds
/// a `(now_ns, cpu_ns_now)` anchor and, each time at least a full window
/// has elapsed since the anchor, recomputes `stalled = accrued <
/// TRICKLE_FLOOR_NS` and slides the anchor forward. Between window
/// closures it reports the last verdict. Before the FIRST window closes it
/// reports `false` (not stalled) — conservative during the boot CPU ramp,
/// so a cell is never declared stalled before a full window of evidence
/// exists.
pub(crate) struct CpuTrickleTracker {
    anchor_ns: u64,
    anchor_cpu_ns: u64,
    stalled: bool,
    seeded: bool,
}

impl CpuTrickleTracker {
    pub(crate) fn new() -> Self {
        Self {
            anchor_ns: 0,
            anchor_cpu_ns: 0,
            stalled: false,
            seeded: false,
        }
    }

    /// Fold this tick's `(cpu_ns_now, now_ns)` and return whether the
    /// guest is currently trickle-stalled. `cpu_ns_now` is the ledger's
    /// summed per-vCPU CPU time; `now_ns` is a `run_start`-relative wall
    /// reading (monotone across ticks).
    pub(crate) fn observe(&mut self, cpu_ns_now: u64, now_ns: u64) -> bool {
        if !self.seeded {
            self.anchor_ns = now_ns;
            self.anchor_cpu_ns = cpu_ns_now;
            self.seeded = true;
            return false;
        }
        if now_ns.saturating_sub(self.anchor_ns) >= TRICKLE_STALL_WINDOW_NS {
            let accrued = cpu_ns_now.saturating_sub(self.anchor_cpu_ns);
            self.stalled = accrued < TRICKLE_FLOOR_NS;
            self.anchor_ns = now_ns;
            self.anchor_cpu_ns = cpu_ns_now;
        }
        self.stalled
    }
}

/// Consecutive stalled watchdog ticks (frozen `monitor_heartbeat`) before
/// the monitor is declared not-live, which suppresses BOTH progress tiers
/// (see [`watchdog_step`]'s `monitor_live` gate). The watchdog ticks
/// every 100 ms and the monitor samples on the same nominal 100 ms
/// cadence, but under heavy host-compile contention the monitor's tick
/// slips and several watchdog ticks can pass between heartbeat bumps. 20
/// ticks = 2 s is comfortably above any realistic single monitor-interval
/// slip (so a merely slow monitor is never misread as dead) yet short
/// enough that a genuinely wedged monitor disables the CPU/wall tiers
/// within 2 s, leaving only the Tier-3 hard deadline. This is ALSO the
/// natural degraded / no-monitor path: a host where `start_monitor`
/// returned no monitor never bumps the heartbeat, so `monitor_live`
/// latches false here after 2 s and the cell runs Tier-3-only.
pub(crate) const WATCHDOG_MONITOR_LIVENESS_MISS_TICKS: u32 = 20;

/// Watchdog-thread-local tracker for monitor liveness: the monitor reads
/// live iff its `monitor_heartbeat` advanced within the last
/// [`WATCHDOG_MONITOR_LIVENESS_MISS_TICKS`] ticks. A pure state machine —
/// no clocks, no atomics — so the stall/recover behaviour is unit-
/// testable in isolation.
pub(crate) struct MonitorLiveness {
    prev_heartbeat: u64,
    misses: u32,
}

impl MonitorLiveness {
    /// Seed at the heartbeat's zero-init value. Before any monitor tick
    /// the heartbeat is 0; the first `observe` that sees a bumped value
    /// clears the miss counter. If no monitor ever runs the heartbeat
    /// stays 0, so this seeds the natural no-monitor stall.
    pub(crate) fn new() -> Self {
        Self {
            prev_heartbeat: 0,
            misses: 0,
        }
    }

    /// Fold this tick's `monitor_heartbeat` and return whether the monitor
    /// is currently live. An advance resets the miss counter; a frozen
    /// heartbeat increments it, and once it reaches the miss budget the
    /// monitor reads not-live until the heartbeat advances again.
    pub(crate) fn observe(&mut self, heartbeat: u64) -> bool {
        if heartbeat != self.prev_heartbeat {
            self.prev_heartbeat = heartbeat;
            self.misses = 0;
        } else {
            self.misses = self.misses.saturating_add(1);
        }
        self.misses < WATCHDOG_MONITOR_LIVENESS_MISS_TICKS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::CPU_CURRENCY_PMU;

    // Lifecycle discriminants (mirror `monitor::LifecycleStage`; written
    // against the raw ids so this stays independent of that enum).
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

    /// Common Boot/INFRA, monitor-live, 1-vCPU caller. Each row varies the
    /// inputs that matter to it: `cpu_in_phase`, `wall_in_phase`,
    /// `runnable`, `channels`, `trickle_stalled`, `currency`.
    #[allow(clippy::too_many_arguments)]
    fn boot_step(
        cpu_in_phase: u64,
        wall_in_phase: u64,
        runnable: bool,
        channels: bool,
        trickle_stalled: bool,
        currency: u8,
    ) -> KillDecision {
        watchdog_step(
            BOOT,
            PhaseClass::Infra,
            cpu_in_phase,
            wall_in_phase,
            runnable,
            channels,
            trickle_stalled,
            true, // monitor_live
            currency,
            1,
        )
    }

    // ---- Tier-1: spinning wedge (cpu_in_phase over the phase budget) ----

    #[test]
    fn tier1_high_cpu_fires_regardless_of_demand_or_trickle() {
        // Way over the Boot CPU budget → Tier-1, whether the guest looks
        // runnable or not and whatever the trickle/channel state: a
        // spinning wedge is caught on CPU alone.
        for &runnable in &[true, false] {
            for &channels in &[true, false] {
                for &trickle in &[true, false] {
                    let d = boot_step(
                        boot_budget_ns() + 100 * S,
                        0,
                        runnable,
                        channels,
                        trickle,
                        CPU_CURRENCY_PMU,
                    );
                    assert_eq!(
                        d,
                        KillDecision::Tier1CpuBudget,
                        "runnable={runnable} channels={channels} trickle={trickle}"
                    );
                }
            }
        }
    }

    #[test]
    fn tier1_body_high_cpu_never_fires() {
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
            false,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn tier1_budget_boundary_is_exclusive() {
        let budget = boot_budget_ns(); // PMU: no widening
        // Exactly at budget → not over → no fire.
        assert_eq!(
            boot_step(budget, 0, false, true, false, CPU_CURRENCY_PMU),
            KillDecision::None,
            "cpu == budget is not over budget"
        );
        // One ns over → fire.
        assert_eq!(
            boot_step(budget + 1, 0, false, true, false, CPU_CURRENCY_PMU),
            KillDecision::Tier1CpuBudget
        );
    }

    #[test]
    fn zero_cpu_is_never_tier1_at_any_wall() {
        // Zero CPU burned in-phase: never Tier-1, at any wall.
        for wall in [0, boot_backstop_ns() + S, 100 * S] {
            let d = boot_step(0, wall, true, true, false, CPU_CURRENCY_PMU);
            assert_eq!(d, KillDecision::None, "wall={wall}");
        }
    }

    // ---- Tier-2: idle wedge (INFRA + channels + !runnable + trickle-
    //      stalled + wall past backstop) ----

    #[test]
    fn tier2_idle_wedge_fires_when_all_conjuncts_hold() {
        // No runnable demand, channels live, CPU trickle-stalled, past the
        // wall backstop → a silent idle wedge.
        let d = boot_step(0, boot_backstop_ns() + S, false, true, true, CPU_CURRENCY_PMU);
        assert_eq!(d, KillDecision::Tier2IdleWedge);
    }

    #[test]
    fn tier2_starved_but_alive_not_trickle_stalled_does_not_fire() {
        // The KEY revised-design row: no runnable demand and past the wall
        // backstop, BUT the CPU is still trickling above the floor (NOT
        // trickle-stalled) — a host-starved but ALIVE cell. Tier-2 must
        // NOT fire; the deadman defers it too.
        let d = boot_step(5_000_000, boot_backstop_ns() + S, false, true, false, CPU_CURRENCY_PMU);
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn tier2_runnable_is_an_independent_exempt() {
        // Runnable demand present → not an idle wedge even when
        // trickle-stalled and past the backstop (the deadman bounds a
        // runnable-piled-up stalled cell instead).
        let d = boot_step(0, boot_backstop_ns() + S, true, true, true, CPU_CURRENCY_PMU);
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn tier2_requires_live_channels() {
        // Channels DEAD + quiet INFRA past the backstop + trickle-stalled:
        // absence of demand is blind here (the acceptance storm's 33 false
        // kills), so Tier-2 must NOT fire. currency=none (the storm's
        // pre-resolution shape) so Tier-1 is also off → None.
        let d = boot_step(0, boot_backstop_ns() + S, false, false, true, CPU_CURRENCY_NONE);
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn tier2_needs_no_cpu_currency() {
        // Idle wedge under currency=NONE still fires: Tier-2 reasons about
        // the trickle bool + channels, not the CPU sum's currency.
        let d = boot_step(0, boot_backstop_ns() + S, false, true, true, CPU_CURRENCY_NONE);
        assert_eq!(d, KillDecision::Tier2IdleWedge);
    }

    #[test]
    fn tier2_body_idle_never_fires() {
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
            true,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn tier2_below_backstop_does_not_fire() {
        // All Tier-2 conjuncts but the wall is still short of the backstop.
        let d = boot_step(0, boot_backstop_ns() - S, false, true, true, CPU_CURRENCY_PMU);
        assert_eq!(d, KillDecision::None);
    }

    // ---- Channel-independent Tier-1 even when channels are dead ----

    #[test]
    fn tier1_fires_with_dead_channels_cpu_is_host_side() {
        // Channels DEAD but CPU over the widened pthread budget: CPU
        // evidence is host-side (pthread/PMU) and channel-independent, so a
        // spinning boot wedge STILL fires even in the blind early-boot
        // window.
        let over = boot_budget_ns() + boot_budget_ns() / 2 + S; // > widened pthread budget
        let d = boot_step(over, 0, false, false, false, CPU_CURRENCY_PTHREAD);
        assert_eq!(d, KillDecision::Tier1CpuBudget);
    }

    // ---- Monitor-liveness gate: dead monitor suppresses both tiers ----

    #[test]
    fn dead_monitor_suppresses_everything() {
        // A Tier-1-shaped input (high CPU) and a Tier-2-shaped one (idle
        // wedge) both return None while the monitor is not live.
        let tier1_shape = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            boot_budget_ns() + 100 * S,
            0,
            false,
            true,
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
            true,
            true,
            false, // monitor dead
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(tier2_shape, KillDecision::None);
    }

    // ---- pthread widening: (budget, 1.5·budget] fires only under PMU ----

    #[test]
    fn pthread_widening_band_fires_only_under_pmu() {
        let budget = boot_budget_ns();
        let cpu = budget + budget / 4; // within (budget, 1.5·budget)
        assert_eq!(
            boot_step(cpu, 0, false, true, false, CPU_CURRENCY_PMU),
            KillDecision::Tier1CpuBudget,
            "PMU currency uses the raw budget — this burn is over it"
        );
        assert_eq!(
            boot_step(cpu, 0, false, true, false, CPU_CURRENCY_PTHREAD),
            KillDecision::None,
            "pthread widens the budget 3/2 — this burn is under the widened budget"
        );
    }

    #[test]
    fn pthread_widening_fires_above_widened_budget() {
        let budget = boot_budget_ns();
        let cpu = budget + budget / 2 + S; // over even the widened budget
        assert_eq!(
            boot_step(cpu, 0, false, true, false, CPU_CURRENCY_PTHREAD),
            KillDecision::Tier1CpuBudget
        );
    }

    #[test]
    fn widened_cpu_budget_matches_currency_widening() {
        // pthread widens the raw Boot budget 3/2; PMU leaves it raw.
        let raw = boot_budget_ns();
        assert_eq!(widened_cpu_budget_ns(BOOT, 1, CPU_CURRENCY_PMU), raw);
        assert_eq!(
            widened_cpu_budget_ns(BOOT, 1, CPU_CURRENCY_PTHREAD),
            raw + raw / 2
        );
    }

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
            true,
            true,
            CPU_CURRENCY_PMU,
            1,
        );
        assert_eq!(d, KillDecision::None);
    }

    // ---- Tier-3 deadman deferral: deadman_should_fire ----

    #[test]
    fn deadman_dead_monitor_always_fires() {
        // Monitor dead → fire regardless of wall / trickle (the machinery
        // that could defer is gone).
        assert!(deadman_should_fire(false, 0, false));
        assert!(deadman_should_fire(false, 100 * S, true));
    }

    #[test]
    fn deadman_starved_but_alive_defers_forever() {
        // Monitor live, NOT trickle-stalled (CPU still accruing) — a
        // starved-but-alive cell defers past the wall deadline no matter
        // how long since the last milestone. Bounded by the harness, not
        // this deadman.
        assert!(!deadman_should_fire(true, TIER3_PROGRESS_GRACE_NS + S, false));
        assert!(!deadman_should_fire(true, 1_000 * S, false));
    }

    #[test]
    fn deadman_inert_cell_fires_past_grace() {
        // Monitor live, trickle-stalled, and no milestone for longer than
        // the grace → the cell is inert → fire (the idle-userspace-wedge /
        // stalled-scx bound). Strict `>` on the grace boundary.
        assert!(deadman_should_fire(true, TIER3_PROGRESS_GRACE_NS + 1, true));
        assert!(
            !deadman_should_fire(true, TIER3_PROGRESS_GRACE_NS, true),
            "exactly at the grace is not yet past it"
        );
    }

    #[test]
    fn deadman_trickle_stalled_within_grace_defers() {
        // Trickle-stalled but a milestone landed within the grace → defer
        // (a recent milestone means the cell is still advancing its
        // lifecycle).
        assert!(!deadman_should_fire(true, TIER3_PROGRESS_GRACE_NS - S, true));
    }

    // ---- CpuTrickleTracker ----

    #[test]
    fn trickle_reports_false_before_first_window_closes() {
        let mut t = CpuTrickleTracker::new();
        // Seed tick, then sub-window ticks: not enough evidence yet.
        assert!(!t.observe(0, 0));
        assert!(!t.observe(0, S));
        assert!(!t.observe(0, TRICKLE_STALL_WINDOW_NS - S));
    }

    #[test]
    fn trickle_idle_cell_stalls_after_a_window() {
        // Idle/wedged guest: only ISR trickle (10 µs) accrues over the 10 s
        // window — below the 1 ms floor → stalled.
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(0, 0)); // seed
        assert!(
            t.observe(10_000, TRICKLE_STALL_WINDOW_NS),
            "10 µs over the window is below the floor → stalled"
        );
    }

    #[test]
    fn trickle_starved_cell_stays_alive() {
        // Host-starved cell at ~40x dilation still accrues tens of ms/s;
        // over a 10 s window that is hundreds of ms — far above the floor →
        // NOT stalled.
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(0, 0)); // seed
        assert!(
            !t.observe(200_000_000, TRICKLE_STALL_WINDOW_NS),
            "200 ms over the window is above the floor → alive"
        );
    }

    #[test]
    fn trickle_floor_boundary() {
        // Exactly the floor is NOT below the floor → not stalled; one ns
        // under is → stalled.
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(0, 0));
        assert!(!t.observe(TRICKLE_FLOOR_NS, TRICKLE_STALL_WINDOW_NS));

        let mut t2 = CpuTrickleTracker::new();
        assert!(!t2.observe(0, 0));
        assert!(t2.observe(TRICKLE_FLOOR_NS - 1, TRICKLE_STALL_WINDOW_NS));
    }

    #[test]
    fn trickle_holds_verdict_between_window_closures() {
        // After a stalled verdict at one window close, sub-window ticks
        // keep reporting stalled until the next window closes and
        // re-evaluates (here to alive on a fresh accrual).
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(0, 0)); // seed
        assert!(t.observe(0, TRICKLE_STALL_WINDOW_NS)); // first close → stalled
        // Sub-window tick, still stalled (held).
        assert!(t.observe(500_000_000, TRICKLE_STALL_WINDOW_NS + S));
        // Next full window with big accrual → re-evaluates to alive.
        assert!(!t.observe(1_000_000_000, 2 * TRICKLE_STALL_WINDOW_NS + S));
    }

    // ---- Monitor-liveness tracker: stall → not-live after N, recovers ----

    #[test]
    fn monitor_liveness_stalls_after_miss_budget_and_recovers() {
        let n = WATCHDOG_MONITOR_LIVENESS_MISS_TICKS;
        let mut live = MonitorLiveness::new();
        assert!(live.observe(1), "advancing heartbeat is live");
        for i in 1..n {
            assert!(live.observe(1), "stall {i} within budget stays live");
        }
        assert!(
            !live.observe(1),
            "the {n}th consecutive stall crosses the miss budget → not live"
        );
        assert!(live.observe(2), "a heartbeat advance recovers liveness");
    }

    #[test]
    fn monitor_liveness_no_monitor_seed_latches_dead() {
        let mut live = MonitorLiveness::new();
        let n = WATCHDOG_MONITOR_LIVENESS_MISS_TICKS;
        let mut last = true;
        for _ in 0..n {
            last = live.observe(0);
        }
        assert!(!last, "a heartbeat frozen at 0 latches not-live after N ticks");
    }

    // ---- evaluate_progress: ledger snapshot → tier decision glue ----

    #[allow(clippy::too_many_arguments)]
    fn snap(
        phase: u8,
        cpu_ns_now: u64,
        cpu_ns_at_phase: u64,
        wall_ns_at_progress: u64,
        runnable_demand: bool,
        channels_live: bool,
        cpu_currency: u8,
    ) -> LedgerSnapshot {
        LedgerSnapshot {
            phase,
            cpu_ns_now,
            cpu_ns_at_phase,
            wall_ns_at_progress,
            progress_epoch: 0,
            monitor_heartbeat: 0,
            runnable_demand,
            cpu_currency,
            evidence_channels_live: channels_live,
        }
    }

    #[test]
    fn evaluate_progress_infra_idle_wedge_fires_tier2() {
        // INFRA (Boot), live monitor, no milestone (wall_at_progress==0) so
        // wall_in_phase == now past the backstop, no runnable demand,
        // channels live, trickle-stalled → a silent idle wedge.
        let now = boot_backstop_ns() + S;
        let d = evaluate_progress(
            &snap(BOOT, 0, 0, 0, false, true, CPU_CURRENCY_PMU),
            now,
            true, // trickle_stalled
            true, // monitor_live
            1,
        );
        assert_eq!(d, KillDecision::Tier2IdleWedge);
    }

    #[test]
    fn evaluate_progress_channels_dead_suppresses_tier2() {
        // Same idle shape but channels dead (the storm's early-boot blind
        // window) → Tier-2 suppressed → None.
        let now = boot_backstop_ns() + S;
        let d = evaluate_progress(
            &snap(BOOT, 0, 0, 0, false, false, CPU_CURRENCY_PMU),
            now,
            true,
            true,
            1,
        );
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn evaluate_progress_channels_dead_still_fires_tier1() {
        // Storm scenario end-to-end: channels dead, currency=pthread, CPU
        // in-phase over the widened budget → Tier-1 STILL fires (host-side
        // CPU evidence).
        let over = boot_budget_ns() + boot_budget_ns() / 2 + S;
        let d = evaluate_progress(
            &snap(BOOT, over, 0, 0, false, false, CPU_CURRENCY_PTHREAD),
            0,
            false,
            true,
            1,
        );
        assert_eq!(d, KillDecision::Tier1CpuBudget);
    }

    #[test]
    fn evaluate_progress_initial_state_counts_from_run_start() {
        // Before the first milestone both anchors are 0, so a Boot that
        // burned past its CPU budget since run start fires Tier-1.
        let d = evaluate_progress(
            &snap(BOOT, boot_budget_ns() + 100 * S, 0, 0, false, false, CPU_CURRENCY_PMU),
            0,
            false,
            true,
            1,
        );
        assert_eq!(d, KillDecision::Tier1CpuBudget);
    }

    #[test]
    fn evaluate_progress_dead_monitor_suppresses_tiers() {
        let now = boot_backstop_ns() + S;
        let d = evaluate_progress(
            &snap(BOOT, 0, 0, 0, false, true, CPU_CURRENCY_PMU),
            now,
            true,
            false, // monitor dead
            1,
        );
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn evaluate_progress_body_never_fires() {
        let now = 1_000 * S;
        let d = evaluate_progress(
            &snap(BODY, u64::MAX - 1, 0, 0, false, true, CPU_CURRENCY_PMU),
            now,
            true,
            true,
            1,
        );
        assert_eq!(d, KillDecision::None);
    }
}
