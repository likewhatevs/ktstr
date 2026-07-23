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
//! reasons about a phase's CPU burn and runnable demand instead:
//!   - Tier-1 (CPU budget): the phase burned more guest CPU IN-PHASE than
//!     its [`phase_cpu_budget_ns`] without reaching its milestone — a
//!     *spinning* wedge.
//!   - Tier-2 (idle wedge): an INFRA phase sat past its
//!     [`phase_wall_backstop_ns`] with live evidence channels and no
//!     runnable demand — a *silent* wedge. No CPU term: the runnable
//!     conjunct alone carries the starvation protection (see the rule).
//!   - Tier-3 begins at the guest-derived hard wall boundary, but wall age
//!     is not a verdict. [`DeadmanHostService`] keeps explicit per-vCPU host
//!     task/service samples: runnable tasks, sample changes, and delivered
//!     vCPU service defer and re-anchor; only an unchanged blocked set can
//!     spend a finite watchdog-thread CPU-service budget. Explicit monitor
//!     termination and the busiest-vCPU CPU backstop remain authoritative.
//!
//! [`CpuTrickleTracker`] remains diagnostic telemetry. Its calibrated
//! busiest-vCPU window is useful in failure reports, but it is deliberately
//! absent from every kill predicate: wide idle guests and deeply starved
//! live guests overlap too much for a fixed trickle floor to be proof of a
//! wedge.

use crate::monitor::{
    CPU_CURRENCY_NONE, CPU_CURRENCY_PTHREAD, LedgerSnapshot, LifecycleStage, StageClass,
};
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
    /// Tier-2: INFRA phase past its wall backstop with live evidence
    /// channels and no runnable demand — a silent idle wedge.
    Tier2IdleWedge,
}

/// Trailing window over which [`CpuTrickleTracker`] measures the guest's
/// busiest-vCPU CPU accrual. 10 s is long enough to average out the 100 ms
/// tick jitter and the bursty ISR pattern on parked vCPUs while keeping the
/// diagnostic responsive.
const TRICKLE_STALL_WINDOW_NS: u64 = 10_000_000_000;

/// Minimum busiest-vCPU guest CPU (ns) that must accrue over one
/// [`TRICKLE_STALL_WINDOW_NS`] for the guest to count as "receiving CPU"
/// (NOT trickle-stalled) under the PMU currency. 1 ms / 10 s distinguishes
/// two useful diagnostic populations: an
/// idle-or-wedged guest still takes timer interrupts on its parked vCPUs
/// but the PMU SW task-clock (`exclude_host`) counts only guest-mode
/// execution, so its busiest vCPU accrues just the in-guest ISR bodies —
/// tens of µs over the window at 1 vCPU, below the floor → stalled — while
/// even a cell starved at ~40x host dilation lands tens of ms/s on its
/// busiest vCPU (hundreds of ms over the window — far above the floor →
/// not stalled).
///
/// CALIBRATED ON 1-vCPU CELLS. At width the guest housekeeping CPU's
/// timekeeping/RCU duty grows with vCPU count and can lift an idle guest's
/// busiest vCPU over this floor (see [`CpuTrickleTracker`]); the misread
/// is why this signal is diagnostic only.
const TRICKLE_FLOOR_NS: u64 = 1_000_000;

/// [`TRICKLE_FLOOR_NS`]'s sibling for the PTHREAD currency, mirroring the
/// Tier-1 budget's [`widen_budget_for_currency`]: the per-vCPU pthread
/// clock charges the FULL VM-exit/entry/host-ISR path for every residual
/// idle kernel tick (clocksource watchdog, RCU, hrtimers), so an
/// idle-wedged guest's measured trickle is orders of magnitude above the
/// PMU magnitude the 1 ms floor was calibrated for. Measured on an
/// idle-wedged 1-vCPU cell under pthread currency: 1-10 ms per 10 s
/// window — ABOVE the 1 ms floor (which back when trickle also gated
/// Tier-2 made the idle wedge invisible to both consumers); a
/// starved-but-alive cell at even ~100x dilation accrues 100+ ms per
/// window on its busiest vCPU. 25 ms sits in that measured gap with ~2.5x
/// margin in both directions for diagnostics.
///
/// CALIBRATED ON 1-vCPU CELLS (where the summed and busiest-vCPU
/// currencies coincide). At width the housekeeping-CPU duty lifts an idle
/// guest's busiest vCPU to 20-45 ms per window at 64 vCPUs — over this
/// floor, which is why no watchdog verdict consumes this classification.
const TRICKLE_FLOOR_PTHREAD_NS: u64 = 25_000_000;

/// Consecutive sub-floor [`TRICKLE_STALL_WINDOW_NS`] windows required
/// before [`CpuTrickleTracker`] reports `stalled`. A SINGLE 10 s window can
/// legitimately accrue below the floor on a live cell: at a 100-300x
/// scheduling tail a bursty host gap can deny a still-alive guest CPU for
/// the whole window (the trickle-window-misread residual — a cell inert-
/// killed on one stalled window during a ~285x gap). Two CONSECUTIVE
/// windows are 20 s of continuous sub-millisecond accrual, which cleanly
/// separates a genuinely inert guest (µs/s ISR trickle, every window sub-
/// floor) from bursty starvation (some window recovers, resetting the
/// streak) while still bounding a true wedge within ~20 s + the Tier-3
/// grace.
const TRICKLE_STALL_CONSECUTIVE_WINDOWS: u32 = 2;

/// Widen a CPU budget for the trust level of its currency. The PMU
/// SW task-clock (`exclude_host`) measures guest-only CPU time directly,
/// so the budget is used as-is. Per-vCPU pthread CPU time additionally
/// charges VM-exit / host-side overhead to the guest, inflating the
/// apparent CPU burn by a documented margin; widen the budget by 3/2
/// (integer `b + b/2`) so that overhead does not push a healthy phase
/// over its Tier-1 budget. Saturating so a `u64::MAX` (Body sentinel)
/// budget cannot wrap.
pub(crate) const fn widen_budget_for_currency(budget: u64, cpu_currency: u8) -> u64 {
    match cpu_currency {
        // pthread time includes VM-exit overhead — degrade the budget by
        // widening it 3/2 (integer math).
        CPU_CURRENCY_PTHREAD => budget.saturating_add(budget / 2),
        // PMU (or any other trusted source) measures guest-only CPU time
        // directly — no widening.
        _ => budget,
    }
}

/// The trickle-stall floor (ns per [`TRICKLE_STALL_WINDOW_NS`]) for the
/// trust level of the CPU currency — [`widen_budget_for_currency`]'s
/// sibling for the [`CpuTrickleTracker`]. PMU measures guest-only time, so
/// the tight [`TRICKLE_FLOOR_NS`] applies; pthread time inflates an idle
/// guest's trickle with VM-exit overhead, so the widened
/// [`TRICKLE_FLOOR_PTHREAD_NS`] applies (see its doc for the measured
/// populations). `CPU_CURRENCY_NONE` gets the pthread floor too: with no
/// per-vCPU source this tick the readings carry no trusted magnitude, and
/// the wider floor produces the more useful diagnostic classification.
pub(crate) const fn trickle_floor_for_currency(cpu_currency: u8) -> u64 {
    match cpu_currency {
        crate::monitor::CPU_CURRENCY_PMU => TRICKLE_FLOOR_NS,
        _ => TRICKLE_FLOOR_PTHREAD_NS,
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
///   - `max_vcpu_cpu_in_phase_ns`: the MAX over vCPUs of per-vCPU guest CPU
///     burned since this phase was entered (the last milestone) — the
///     monitor's width-independent Tier-1 evidence. A spinning wedge is a
///     hot thread, so its max crosses the phase budget (flat outside Boot);
///     a wide idle guest's summed background burn does NOT (that summed
///     number lives in `cpu_ns_now`, which no longer feeds any tier).
///   - `wall_in_phase_ns`: wall time since this phase was entered (the
///     last milestone).
///   - `runnable_demand`: the guest had queued-or-running tasks anywhere —
///     THE Tier-2 exempt, carrying the starvation protection alone (a
///     runnable cell is not an idle wedge; if it is also stalled the Tier-3
///     deadman / guest scx watchdog bound it instead).
///   - `channels_live`: the per-CPU GUEST evidence channels resolved this
///     tick (see [`crate::monitor::ProgressLedger::evidence_channels_live`]).
///     Gates Tier-2 — `runnable_demand` is only meaningful when true.
///   - `monitor_live`: the monitor thread is producing fresh ledger
///     writes. Gates TIER-1 ONLY (its CPU currency comes from monitor
///     writes); Tier-2 is monitor-live-independent because its wall anchor
///     is dispatch-advanced (see the Tier-2 block).
///   - `cpu_currency`: provenance of `max_vcpu_cpu_in_phase_ns`
///     ([`CPU_CURRENCY_NONE`]/`PTHREAD`/`PMU`).
///   - `vcpus`: total guest width. Only Boot consumes it because the BSP
///     serially initializes every AP; other phase budgets remain flat.
#[allow(clippy::too_many_arguments)]
pub(crate) fn watchdog_step(
    phase: u8,
    class: PhaseClass,
    max_vcpu_cpu_in_phase_ns: u64,
    wall_in_phase_ns: u64,
    runnable_demand: bool,
    channels_live: bool,
    monitor_live: bool,
    cpu_currency: u8,
    vcpus: u32,
) -> KillDecision {
    // Tier-1: spinning wedge — the phase burned its CPU budget IN-PHASE
    // without reaching its milestone. Requires a LIVE monitor: Tier-1's CPU
    // signal (`max_vcpu_cpu_in_phase_ns`) comes from monitor ledger writes,
    // so a stale monitor makes the in-phase burn untrustworthy. Also only
    // valid when the CPU signal has a trusted currency; with
    // CPU_CURRENCY_NONE there is no per-vCPU CPU-time source this tick, so
    // the measurement is meaningless and Tier-1 is off (Tier-2 below carries
    // no CPU term at all). Body's budget is the `u64::MAX` sentinel, so
    // Tier-1 is structurally off for Body via the budget table, not a class
    // check.
    if monitor_live && cpu_currency != CPU_CURRENCY_NONE {
        let budget = widen_budget_for_currency(phase_cpu_budget_ns(phase, vcpus), cpu_currency);
        // Strict `>`: a phase whose busiest vCPU burned *exactly* the budget
        // has not yet exceeded it.
        if max_vcpu_cpu_in_phase_ns > budget {
            return KillDecision::Tier1CpuBudget;
        }
    }

    // Tier-2: silent idle wedge — an INFRA phase sat past its wall
    // backstop, with live evidence channels and no runnable demand. Only
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
    //   - `!runnable_demand`: THE starvation protection, alone. A starved
    //     cell WITH work always shows queued-or-running tasks in its own
    //     rq memory (`nr_running` includes the on-CPU task, and guest
    //     memory is readable regardless of host scheduling), so it is
    //     exempt here; a cell with NOTHING runnable is not starved of
    //     anything, and idle-in-INFRA past the backstop IS the wedge
    //     definition. Host-side Tier-3 task/service evidence and the guest
    //     scx watchdog bound runnable-exempted cells rather than Tier-2.
    //
    // Deliberately NO CPU-trickle conjunct. Trickle here was width-broken
    // belt-and-braces duplicating the runnable protection: the guest
    // housekeeping CPU's timekeeping/RCU tick burns 20-45 ms per 10 s
    // window at 64 vCPUs (measured; scales with width), so no width-stable
    // floor exists — a wide idle wedge read "not stalled". Trickle remains
    // diagnostic telemetry only and cannot influence this verdict.
    //
    // Deliberately NOT gated on `monitor_live` (unlike Tier-1). Tier-2's
    // `wall_in_phase_ns` counts wall since the last MILESTONE anchor
    // (`wall_ns_at_progress`), which the DISPATCH thread advances on guest
    // milestones (`advance_phase` / `record_progress` /
    // `record_boot_progress`) — NOT the monitor (the monitor's per-tick
    // `record_liveness` bumps only the heartbeat; it never touches the
    // anchor). So a monitor whose heartbeat stalled >2 s under host load
    // cannot freeze this anchor: a healthy guest keeps hitting milestones
    // through dispatch and its `wall_in_phase` stays small, while a wedged
    // guest publishes no milestone and the delta grows as intended. The
    // `channels_live` conjunct still gates on last-observed evidence (false
    // through the blind-channel boot window a dead monitor never advanced
    // past), and `!runnable_demand` on the last-observed demand carries the
    // starvation protection. Gating Tier-2 on `monitor_live` turned a
    // monitor stalled under contention into a suppressed idle-wedge verdict,
    // letting a Teardown idle wedge escape every prompt tier (the deadman
    // then defers on the guest's timer-tick CPU trickle) — a dumpless hang
    // to the nextest terminate-after.
    if class == PhaseClass::Infra
        && channels_live
        && !runnable_demand
        && wall_in_phase_ns > phase_wall_backstop_ns(phase, vcpus)
    {
        return KillDecision::Tier2IdleWedge;
    }

    KillDecision::None
}

/// The widened Tier-1 CPU budget (ns) the watchdog charges `phase` under
/// CPU-time provenance `cpu_currency` — the SAME number [`watchdog_step`]
/// compares `max_vcpu_cpu_in_phase_ns` against. Exposed so the watchdog's
/// failure dump can render `max_vcpu_cpu_in_phase vs budget` against the
/// effective (currency-widened) budget rather than the raw
/// [`phase_cpu_budget_ns`]. `vcpus` affects Boot only.
pub(crate) fn widened_cpu_budget_ns(phase: u8, cpu_currency: u8, vcpus: u32) -> u64 {
    widen_budget_for_currency(phase_cpu_budget_ns(phase, vcpus), cpu_currency)
}

/// Width- and dilation-independent CPU backstop for Tier-3 deferral.
///
/// `effective_deadline_budget_ns` is the effective VM deadline expressed
/// as an offset from VM start. Charging that *entire* budget against the
/// busiest single vCPU in the current phase is deliberately generous: a
/// healthy phase cannot consume more single-vCPU CPU than the whole VM was
/// allowed to run. Unlike a second wall deadline, this stays immune to host
/// contention — a starved guest accrues the currency proportionally more
/// slowly. Pthread currency gets the same 3/2 VM-exit-overhead allowance as
/// Tier-1. With no trusted CPU currency the backstop is structurally off.
pub(crate) const fn deadman_cpu_budget_ns(
    effective_deadline_budget_ns: u64,
    cpu_currency: u8,
) -> u64 {
    if cpu_currency == CPU_CURRENCY_NONE {
        u64::MAX
    } else {
        widen_budget_for_currency(effective_deadline_budget_ns, cpu_currency)
    }
}

/// Whether the active-cell Tier-3 CPU backstop has been exhausted.
/// Strict `>` matches Tier-1: exactly at the budget is still permitted.
pub(crate) const fn deadman_cpu_budget_exhausted(
    max_vcpu_cpu_in_phase_ns: u64,
    effective_deadline_budget_ns: u64,
    cpu_currency: u8,
) -> bool {
    max_vcpu_cpu_in_phase_ns > deadman_cpu_budget_ns(effective_deadline_budget_ns, cpu_currency)
}

/// The liveness-INDEPENDENT Tier-3 wall net: the hard multiple of the
/// effective VM deadline that a single phase may sit through — publishing NO
/// milestone and drawing NO tier/deadman kill — before it is declared
/// wedged regardless of any host-service liveness or defer state.
///
/// The starvation-invariant deadman ([`DeadmanHostService`]) re-anchors on
/// every sign of cell liveness — a runnable vCPU, a task-generation change,
/// material vCPU service — so a host-starved-but-HEALTHY cell is never
/// killed for host wall starvation. A WEDGED guest keeps exactly those signs
/// alive forever, though: a spinner stays runnable, an idle guest trickles
/// timer CPU. With both fast tiers suppressible under host load, the deadman
/// alone could then defer indefinitely and let the wedge run to the outer
/// nextest terminate-after with NO ktstr kill and NO failure dump. This net
/// is the terminal guarantee that this cannot happen: it fires the ordinary
/// Tier-3 kill+dump path once a phase has burned [`WALL_NET_DEADLINE_MULT`]x
/// the WHOLE VM's effective-deadline budget without a milestone — which no
/// legitimate phase does (healthy tiers fire in tens of seconds; a genuinely
/// starved cell becomes a witnessed Tier-3 SKIP, the designed degradation).
pub(crate) const WALL_NET_DEADLINE_MULT: u64 = 4;

/// Absolute, dilation-INDEPENDENT ceiling for the wall net, measured on the
/// PROCESS-start clock (nanoseconds since [`crate::vmm::record_process_start`]).
///
/// The [`WALL_NET_DEADLINE_MULT`] term above is measured against
/// `effective_deadline_budget_ns`, which is itself dilation-scaled: the
/// scenario-start watchdog reset arms the deadline to
/// `ScenarioEnd_wall + workload_duration`, and under an extremely
/// oversubscribed host (observed: a coverage lane running guests at ~0.7% of
/// real speed, an 8 s body taking 1190 s of wall) the guest reaches its
/// milestones so late that the effective deadline balloons to ~1000 s, so
/// `4x` lands well past the outer nextest terminate-after — the multiple term
/// alone cannot fire in time and the wedge escapes to a dumpless SIGKILL.
///
/// This ceiling is a FIXED wall bound, independent of any guest-derived or
/// dilation-scaled quantity. CRITICALLY it is measured on the process clock,
/// NOT the VM's `run_start`: nextest's per-test `terminate-after` counts the
/// in-process admission wait, and on a queued lane (ARM: a cell whose
/// VM-relative deadline expired at 126 s had a 672 s nextest wall — a ~546 s
/// queue) the post-admission `run_start.elapsed()` never reaches a
/// ~1000 s ceiling before the 1260 s SIGKILL. Anchoring on the same clock
/// nextest counts makes the ceiling reachable regardless of the queue, with
/// margin for the kill + teardown to complete. Only consulted together with
/// the wedge-signature gate in [`wall_net_tripped`] (a phase stalled past its
/// own wall backstop), so a legitimately-dilated cell still making milestone
/// progress — including a long idle Body, exempt via its `u64::MAX` backstop —
/// is untouched.
pub(crate) const WALL_NET_ABSOLUTE_CEILING_NS: u64 = 1_000_000_000_000; // 1000 s

/// Whether the liveness-independent Tier-3 wall net has tripped. Two terms,
/// either sufficient:
///   - the deadline-relative term: the phase has sat `wall_in_phase_ns`
///     (measured from the last milestone; run-start-relative) past
///     [`WALL_NET_DEADLINE_MULT`]x the effective-deadline budget. A zero budget
///     (deadline unset) disables this term.
///   - the absolute-ceiling term: the process has run past
///     [`WALL_NET_ABSOLUTE_CEILING_NS`] of `now_process_ns` (process-start
///     relative, so it INCLUDES the admission queue and tracks nextest's rail)
///     AND the current phase has stalled past its own `phase_wall_backstop_ns`
///     (the wedge signature; `wall_in_phase_ns` stays VM-relative). This
///     dilation-independent backstop catches a wedge whose effective deadline
///     was inflated past the outer rail by a dilation-scaled reset. The
///     backstop gate keeps it off a Body phase (`u64::MAX` backstop) and off a
///     phase that only just re-anchored a milestone (a legitimately-slow cell
///     making forward progress), so it fires only on a genuinely stalled
///     INFRA phase.
///
/// Strict `>` throughout.
pub(crate) const fn wall_net_tripped(
    wall_in_phase_ns: u64,
    effective_deadline_budget_ns: u64,
    now_process_ns: u64,
    phase_wall_backstop_ns: u64,
) -> bool {
    let deadline_term = effective_deadline_budget_ns != 0
        && wall_in_phase_ns > effective_deadline_budget_ns.saturating_mul(WALL_NET_DEADLINE_MULT);
    let absolute_term =
        now_process_ns > WALL_NET_ABSOLUTE_CEILING_NS && wall_in_phase_ns > phase_wall_backstop_ns;
    deadline_term || absolute_term
}

/// Fold one live [`LedgerSnapshot`] into a [`KillDecision`]: the glue
/// between the ledger and the pure [`watchdog_step`]. Derives the wall
/// in-phase delta and maps the lifecycle stage onto a [`PhaseClass`],
/// then defers to `watchdog_step`. Kept pure (the wall clock arrives as
/// `now_wall_ns`) so the ledger-read → tier-decision path is unit-testable
/// without a live watchdog thread. The ledger's
/// `cpu_trickle_stalled` remains diagnostic-only and is not consumed here.
///
/// `now_wall_ns` is `run_start`-relative — the SAME `run_start` the
/// monitor/dispatch anchor `wall_ns_at_progress` to — so `wall_in_phase`
/// aligns. The Tier-1 CPU evidence is the monitor's
/// `max_vcpu_cpu_in_phase_ns` (already anchored monitor-side to the current
/// phase — see [`crate::monitor::ProgressLedger::max_vcpu_cpu_in_phase_ns`]),
/// used as-is; the wall delta uses the milestone anchor
/// `wall_ns_at_progress`.
///
/// INITIAL STATE: before the monitor's first re-anchor of a fresh phase,
/// `max_vcpu_cpu_in_phase_ns` reads 0 (the monitor re-anchors before
/// computing), and before the first milestone `wall_ns_at_progress` is 0 so
/// `wall_in_phase == now_wall_ns`. Boot's wall backstop thus counts from run
/// start — intended (Boot owes its first milestone). Saturating subtraction
/// guards the benign torn read where a fresher anchor races an older `now`
/// (clamps to 0 — more conservative, never a spurious kill).
pub(crate) fn evaluate_progress(
    snap: &LedgerSnapshot,
    now_wall_ns: u64,
    monitor_live: bool,
    vcpus: u32,
) -> KillDecision {
    let wall_in_phase = now_wall_ns.saturating_sub(snap.wall_ns_at_progress);
    let class = match LifecycleStage::from_u8(snap.phase).class() {
        StageClass::Body => PhaseClass::Body,
        StageClass::Infra => PhaseClass::Infra,
    };
    watchdog_step(
        snap.phase,
        class,
        snap.max_vcpu_cpu_in_phase_ns,
        wall_in_phase,
        snap.runnable_demand,
        snap.evidence_channels_live,
        monitor_live,
        snap.cpu_currency,
        vcpus,
    )
}

/// Delivered watchdog CPU service allowed while every sampled host-vCPU
/// task remains unchanged and non-runnable (or its run state is unknown).
///
/// This is deliberately watchdog-thread CPU time, not wall time. A host
/// scheduler can therefore deschedule both a healthy guest and its
/// watchdog for arbitrarily long without spending the budget. Once the
/// watchdog itself receives two full CPU seconds while repeatedly
/// observing no runnable vCPU, no vCPU service, and no sample change, the
/// observation machinery has had orders of magnitude more service than
/// one `/proc` snapshot needs and the blocked verdict is finite. The
/// budget is only entered after the guest-derived wall boundary; it does
/// not lengthen healthy runs.
pub(crate) const DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS: u64 = 2_000_000_000;

/// Minimum PER-vCPU guest-CPU advance BETWEEN two Tier-3 observations that
/// counts as forward progress and re-anchors the blocked-observer interval;
/// a smaller advance is trickle, not progress, and lets the finite
/// blocked-observer budget keep accruing.
///
/// Consulted (as the per-vCPU MAX advance, see
/// [`DeadmanHostService::max_per_task_advance_ns`]) once every sampled vCPU is
/// non-runnable/unknown; a runnable vCPU already defers unconditionally. An
/// idle guest that has merely HLT-ed still takes its periodic timer / RCU
/// interrupts, nudging a vCPU's cumulative CPU up a few tens to a few hundred
/// microseconds per ~100 ms watchdog observation (measured populations:
/// 1-10 ms per 10 s window at 1 vCPU, 20-45 ms per 10 s on the busiest vCPU at
/// 64 vCPUs — the housekeeping CPU's timekeeping/RCU duty; see
/// [`CpuTrickleTracker`]). MAX, not SUM, is the load-bearing choice: a wide
/// idle guest's diffuse trickle SUMS above any fixed floor at width (the
/// 256-vCPU idle-wedge escape) yet its per-vCPU MAX stays sub-floor exactly
/// like the narrow case, while a genuinely progressing guest always has SOME
/// vCPU crossing the floor. Treating trickle as "service advanced"
/// re-anchored the deadman every observation, so a fully idle wedge never
/// spent the blocked-observer budget. A vCPU doing genuine work accrues on
/// the order of the whole inter-observation interval (tens of ms per 100 ms
/// sample), an order of magnitude above this floor, so a
/// materially-progressing cell still re-anchors and defers. Deliberately
/// generous (1 ms) so a legitimately-slow-but-alive teardown is never charged.
pub(crate) const DEADMAN_BLOCKED_OBSERVER_PROGRESS_FLOOR_NS: u64 = 1_000_000;

/// Host scheduler state of one vCPU task at a Tier-3 observation.
///
/// `Unknown` is intentionally a blocked-class observation rather than a
/// sensor failure: `/proc` can race task publication and exit. A change
/// into or out of `Unknown` re-anchors the blocked-service interval, and a
/// stable unknown sample remains finitely bounded by watchdog CPU service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostVcpuRunState {
    Runnable,
    NonRunnable,
    Unknown,
}

/// One vCPU slot's exact host-side identity and service observation.
///
/// `task_id=None` represents an unpublished vCPU slot. `cpu_ns=None`
/// represents a task whose host CPU clock could not be sampled. Either
/// availability transition changes the sample and re-anchors; stable
/// absence is treated like stable blocked evidence and remains finite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostVcpuTaskSample {
    pub(crate) task_id: Option<u32>,
    pub(crate) cpu_ns: Option<u64>,
    pub(crate) run_state: HostVcpuRunState,
}

/// The watchdog's own `CLOCK_THREAD_CPUTIME_ID` observation.
///
/// Unlike an unavailable vCPU clock, an unavailable watchdog clock cannot
/// degrade to blocked evidence: it is the currency that makes the wait
/// finite without charging host starvation, so its loss has a distinct,
/// deterministic verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeadmanObserverClock {
    Reading(u64),
    Unavailable,
}

/// One pure Tier-3 host-service observation.
///
/// The caller evaluates this only after the guest-derived wall boundary.
/// Wall age, monitor heartbeat age, and guest trickle classification are
/// intentionally absent: none is evidence that a host-starved vCPU task
/// has wedged. `monitor_terminal` and the existing busiest-vCPU CPU
/// backstop remain authoritative and are evaluated before any deferral.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeadmanHostServiceInput<'a> {
    pub(crate) monitor_terminal: bool,
    pub(crate) vcpu_cpu_budget_exhausted: bool,
    pub(crate) observer_cpu: DeadmanObserverClock,
    pub(crate) vcpu_tasks: &'a [HostVcpuTaskSample],
}

/// Why the host-service deadman deferred this observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeadmanHostDefer {
    /// First complete observation establishes both baselines.
    Seeded,
    /// At least one host vCPU task is runnable. Host starvation is not a
    /// guest-wedge verdict, regardless of elapsed wall time.
    Runnable,
    /// The sampled task set, CPU-clock availability, or normalized run
    /// state changed.
    SampleChanged,
    /// A per-task host CPU clock regressed. This commonly means a task was
    /// replaced between observations; re-anchor instead of comparing
    /// across generations.
    VcpuServiceRegressed,
    /// Summed host-vCPU on-CPU service advanced.
    VcpuServiceAdvanced,
    /// The blocked sample stayed identical; this much watchdog CPU
    /// service has been charged against the finite budget.
    Blocked {
        observer_service_ns: u64,
        budget_ns: u64,
    },
}

/// Required-sensor failure detected by the host-service deadman.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeadmanHostSensorFailure {
    ObserverClockUnavailable,
    ObserverClockRegressed { previous_ns: u64, current_ns: u64 },
}

/// A finite Tier-3 verdict produced from host-service evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeadmanHostFire {
    /// The monitor task has terminated; no future evidence can arrive.
    MonitorTerminal,
    /// The existing dilation-immune busiest-vCPU CPU backstop was spent.
    VcpuCpuBudget,
    /// The watchdog received the full blocked-observation service budget
    /// while every vCPU sample remained unchanged and non-runnable/unknown.
    BlockedObserverService {
        observer_service_ns: u64,
        budget_ns: u64,
    },
    /// The observer clock needed to make the blocked wait finite failed.
    SensorFailure(DeadmanHostSensorFailure),
}

/// Result of one [`DeadmanHostService`] observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeadmanHostDecision {
    Defer(DeadmanHostDefer),
    Fire(DeadmanHostFire),
}

#[derive(Debug, Clone)]
struct DeadmanHostBaseline {
    tasks: Vec<HostVcpuTaskSample>,
    summed_cpu_ns: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeadmanHostSampleRelation {
    Unchanged,
    Changed,
    ServiceRegressed,
    ServiceAdvanced,
}

/// Pure Tier-3 host-service state machine.
///
/// Once the wall boundary is reached, feed this tracker exact vCPU-slot
/// samples and the watchdog's own thread-CPU reading. Runnable host tasks
/// and host-vCPU service always defer and re-anchor. So do task-generation,
/// state, clock-availability, and service-regression changes. Only an
/// identical non-runnable/unknown sample charges the observer's delivered
/// CPU service; host wall starvation spends nothing.
///
/// The tracker owns exact per-slot baselines rather than a hash, avoiding
/// both identity aliasing and the false "no progress" result that a summed
/// clock alone could produce when one task advances while another is
/// replaced or regresses.
#[derive(Debug, Clone)]
pub(crate) struct DeadmanHostService {
    baseline: Option<DeadmanHostBaseline>,
    blocked_observer_anchor_ns: Option<u64>,
    last_observer_cpu_ns: Option<u64>,
}

impl DeadmanHostService {
    pub(crate) fn new() -> Self {
        Self {
            baseline: None,
            blocked_observer_anchor_ns: None,
            last_observer_cpu_ns: None,
        }
    }

    /// Forget every Tier-3 host-service anchor.
    ///
    /// An explicit control overlay owns progress while it is active. Merely
    /// skipping observations is insufficient: the watchdog thread's CPU
    /// clock keeps advancing during that interval, so retaining the old
    /// blocked anchor would charge the overlay retroactively on the first
    /// post-overlay observation. Resetting makes the next ordinary tick a
    /// fresh seed, exactly as if the finite wait had not elapsed.
    pub(crate) fn reset(&mut self) {
        self.baseline = None;
        self.blocked_observer_anchor_ns = None;
        self.last_observer_cpu_ns = None;
    }

    /// Fold one host-service observation into a deterministic decision.
    pub(crate) fn observe(&mut self, input: DeadmanHostServiceInput<'_>) -> DeadmanHostDecision {
        self.observe_with_budget(input, DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS)
    }

    /// Fold an observation using a caller-owned blocked-observer service
    /// budget.
    ///
    /// The ordinary Tier-3 deadman uses its short fixed budget through
    /// [`Self::observe`]. Finite control overlays reuse the same
    /// runnable/starvation and task-generation semantics with their own
    /// explicit service budget.
    pub(crate) fn observe_with_budget(
        &mut self,
        input: DeadmanHostServiceInput<'_>,
        blocked_observer_budget_ns: u64,
    ) -> DeadmanHostDecision {
        // These are pre-existing authoritative kill facts. Their precedence
        // also means a failed observer clock cannot hide a completed monitor
        // or an exhausted guest-CPU backstop.
        if input.monitor_terminal {
            return DeadmanHostDecision::Fire(DeadmanHostFire::MonitorTerminal);
        }
        if input.vcpu_cpu_budget_exhausted {
            return DeadmanHostDecision::Fire(DeadmanHostFire::VcpuCpuBudget);
        }

        let observer_cpu_ns = match input.observer_cpu {
            DeadmanObserverClock::Reading(reading) => reading,
            DeadmanObserverClock::Unavailable => {
                return DeadmanHostDecision::Fire(DeadmanHostFire::SensorFailure(
                    DeadmanHostSensorFailure::ObserverClockUnavailable,
                ));
            }
        };

        if let Some(previous_ns) = self.last_observer_cpu_ns
            && observer_cpu_ns < previous_ns
        {
            return DeadmanHostDecision::Fire(DeadmanHostFire::SensorFailure(
                DeadmanHostSensorFailure::ObserverClockRegressed {
                    previous_ns,
                    current_ns: observer_cpu_ns,
                },
            ));
        }
        self.last_observer_cpu_ns = Some(observer_cpu_ns);

        if input
            .vcpu_tasks
            .iter()
            .any(|sample| sample.run_state == HostVcpuRunState::Runnable)
        {
            self.reanchor(input.vcpu_tasks, observer_cpu_ns);
            return DeadmanHostDecision::Defer(DeadmanHostDefer::Runnable);
        }

        let Some(baseline) = &self.baseline else {
            self.reanchor(input.vcpu_tasks, observer_cpu_ns);
            return DeadmanHostDecision::Defer(DeadmanHostDefer::Seeded);
        };
        // The MAX per-vCPU CPU advance since the baseline (not the SUM). A
        // wide idle guest's timer/RCU trickle is diffuse — tens to a few
        // hundred µs on EACH of many vCPUs — so its SUM trivially clears a
        // fixed floor at width and would re-anchor forever (the 256-vCPU idle
        // wedge escape). Its per-vCPU MAX, though, stays sub-floor, exactly
        // like the narrow case; a genuinely progressing guest always has SOME
        // vCPU doing floor-crossing work. Computed here while `baseline` is
        // still borrowed, before the arms may replace `self.baseline`.
        let baseline_max_advance_ns = Self::max_per_task_advance_ns(baseline, input.vcpu_tasks);

        // Charge the finite blocked-observer budget against the existing
        // observer anchor. Shared by the Unchanged and sub-floor-trickle
        // paths: only the watchdog observer clock advances here, so the
        // blocked verdict stays finite.
        let charge_blocked = |observer_service_ns: u64| {
            if observer_service_ns >= blocked_observer_budget_ns {
                DeadmanHostDecision::Fire(DeadmanHostFire::BlockedObserverService {
                    observer_service_ns,
                    budget_ns: blocked_observer_budget_ns,
                })
            } else {
                DeadmanHostDecision::Defer(DeadmanHostDefer::Blocked {
                    observer_service_ns,
                    budget_ns: blocked_observer_budget_ns,
                })
            }
        };

        match Self::relation(baseline, input.vcpu_tasks) {
            DeadmanHostSampleRelation::Changed => {
                self.reanchor(input.vcpu_tasks, observer_cpu_ns);
                DeadmanHostDecision::Defer(DeadmanHostDefer::SampleChanged)
            }
            DeadmanHostSampleRelation::ServiceRegressed => {
                self.reanchor(input.vcpu_tasks, observer_cpu_ns);
                DeadmanHostDecision::Defer(DeadmanHostDefer::VcpuServiceRegressed)
            }
            DeadmanHostSampleRelation::ServiceAdvanced => {
                // Every vCPU here is non-runnable/unknown (a runnable vCPU
                // deferred above). A MATERIAL advance (>= the progress floor)
                // is forward progress: re-anchor both the guest baseline and
                // the blocked-observer interval and defer. A sub-floor
                // advance is idle timer/RCU trickle, NOT progress — refresh
                // the guest CPU baseline so the trickle cannot accumulate
                // across observations, but KEEP the observer anchor so the
                // finite blocked-observer budget keeps accruing (as for
                // Unchanged). Width-independent: the PER-vCPU MAX advance, not
                // the summed one — see DEADMAN_BLOCKED_OBSERVER_PROGRESS_FLOOR_NS.
                if baseline_max_advance_ns >= u128::from(DEADMAN_BLOCKED_OBSERVER_PROGRESS_FLOOR_NS)
                {
                    self.reanchor(input.vcpu_tasks, observer_cpu_ns);
                    DeadmanHostDecision::Defer(DeadmanHostDefer::VcpuServiceAdvanced)
                } else {
                    let anchor_ns = self
                        .blocked_observer_anchor_ns
                        .expect("a deadman host baseline always has an observer anchor");
                    self.baseline = Some(DeadmanHostBaseline {
                        tasks: input.vcpu_tasks.to_vec(),
                        summed_cpu_ns: Self::summed_cpu_ns(input.vcpu_tasks),
                    });
                    charge_blocked(observer_cpu_ns - anchor_ns)
                }
            }
            DeadmanHostSampleRelation::Unchanged => {
                let anchor_ns = self
                    .blocked_observer_anchor_ns
                    .expect("a deadman host baseline always has an observer anchor");
                charge_blocked(observer_cpu_ns - anchor_ns)
            }
        }
    }

    fn reanchor(&mut self, tasks: &[HostVcpuTaskSample], observer_cpu_ns: u64) {
        self.baseline = Some(DeadmanHostBaseline {
            tasks: tasks.to_vec(),
            summed_cpu_ns: Self::summed_cpu_ns(tasks),
        });
        self.blocked_observer_anchor_ns = Some(observer_cpu_ns);
    }

    fn summed_cpu_ns(tasks: &[HostVcpuTaskSample]) -> u128 {
        tasks
            .iter()
            .filter_map(|sample| sample.cpu_ns)
            .map(u128::from)
            .sum()
    }

    /// The MAX over vCPUs of `tasks[i].cpu_ns - baseline.tasks[i].cpu_ns`
    /// (saturating; a missing reading on either side contributes 0). Only
    /// meaningful when `relation` has already established a structural match
    /// (equal length, same per-index `task_id` / `cpu_ns` presence), which the
    /// `ServiceAdvanced` caller guarantees. Width-independent, unlike
    /// [`Self::summed_cpu_ns`]: diffuse per-vCPU idle trickle does not sum into
    /// a false "progress" verdict.
    fn max_per_task_advance_ns(
        baseline: &DeadmanHostBaseline,
        tasks: &[HostVcpuTaskSample],
    ) -> u128 {
        baseline
            .tasks
            .iter()
            .zip(tasks)
            .map(|(before, now)| match (before.cpu_ns, now.cpu_ns) {
                (Some(a), Some(b)) => u128::from(b.saturating_sub(a)),
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }

    fn relation(
        baseline: &DeadmanHostBaseline,
        tasks: &[HostVcpuTaskSample],
    ) -> DeadmanHostSampleRelation {
        if baseline.tasks.len() != tasks.len()
            || baseline.tasks.iter().zip(tasks).any(|(before, now)| {
                before.task_id != now.task_id
                    || before.cpu_ns.is_some() != now.cpu_ns.is_some()
                    || before.run_state != now.run_state
            })
        {
            return DeadmanHostSampleRelation::Changed;
        }

        if baseline
            .tasks
            .iter()
            .zip(tasks)
            .any(|(before, now)| matches!((before.cpu_ns, now.cpu_ns), (Some(a), Some(b)) if b < a))
        {
            return DeadmanHostSampleRelation::ServiceRegressed;
        }

        let summed_cpu_ns = Self::summed_cpu_ns(tasks);
        if summed_cpu_ns > baseline.summed_cpu_ns {
            DeadmanHostSampleRelation::ServiceAdvanced
        } else if summed_cpu_ns < baseline.summed_cpu_ns {
            // The per-task check above normally owns this path. Retain the
            // aggregate guard so future sample representations cannot turn a
            // regression into charged blocked service.
            DeadmanHostSampleRelation::ServiceRegressed
        } else {
            DeadmanHostSampleRelation::Unchanged
        }
    }
}

/// Pure state machine that turns each tick's PER-vCPU cumulative CPU-time
/// readings into a boolean "the guest is receiving essentially no CPU"
/// (trickle-stalled) for diagnostics. No clocks, no atomics — so its window
/// arithmetic is unit-testable in isolation. Driven by the MONITOR (which
/// owns the per-vCPU CPU data), not the watchdog: the
/// monitor feeds it each tick's active-currency per-vCPU cumulatives (the
/// same readings the Tier-1
/// [`crate::monitor::reader::MaxVcpuInPhaseTracker`] sees) and a `run_start`-
/// relative `now_ns`, and publishes the resulting bool to the ledger.
///
/// CURRENCY — THE BUSIEST SINGLE vCPU, computed with PER-vCPU WINDOW
/// ANCHORS. The window's accrual is `max_i(readings[i] - anchor[i])` over
/// the vCPUs, where `anchor[i]` is vCPU `i`'s cumulative at the window
/// START — i.e. the true busiest-single-vCPU CPU burned across the window.
/// This is the least width-sensitive currency available: a 64-vCPU idle
/// guest's diffuse background burn (timer ticks / RCU / IPIs) rotates
/// across vCPUs and SERIALISES — at any tick essentially one vCPU is active
/// — so a summed currency grows ~linearly with count, and a per-TICK max
/// summed over the window grows the same way (`Σ_ticks max_i(Δ_i)` ≈
/// `Σ_ticks Σ_i(Δ_i)`; measured: ~42 ms per 10 s window at 64 vCPUs). The
/// per-vCPU windowed MAX collapses that rotation down to the busiest
/// vCPU's own share. A RESIDUAL width term remains even so — the guest
/// housekeeping CPU's timekeeping/RCU duty grows with width (measured
/// 20-45 ms per 10 s window at 64 vCPUs under pthread currency, vs 1-10 ms
/// at 1 vCPU), which can exceed the 1-vCPU-calibrated floors and misread a
/// wide idle guest as alive. That overlap is why this discriminator is
/// diagnostic-only and absent from both Tier-2 and Tier-3 predicates.
///
/// Measures accrual over a trailing [`TRICKLE_STALL_WINDOW_NS`]: it holds a
/// `(now_ns, anchor[])` window anchor and, each time at least a full window
/// has elapsed, closes the window — classifying `max_i(reading-anchor)` as
/// sub-floor (`< floor_ns`, the caller-supplied currency-dependent floor
/// from [`trickle_floor_for_currency`]) or not — and slides the anchor
/// forward. It reports `stalled` only once
/// [`TRICKLE_STALL_CONSECUTIVE_WINDOWS`] windows have closed sub-floor
/// back-to-back; any window that recovers above the floor resets the
/// streak. Between window closures it reports the last verdict. Before the
/// FIRST window closes it reports `false` (not stalled) — conservative
/// during the boot CPU ramp. A vCPU-count change re-anchors (returns the
/// held verdict) rather than zipping mismatched indices.
pub(crate) struct CpuTrickleTracker {
    anchor_ns: u64,
    /// Per-vCPU cumulative CPU-time (ns) at the current window's START.
    /// Index-stable: vCPU `i` is `anchor[i]`.
    anchor: Vec<u64>,
    /// Held output verdict, refreshed at each window close and reported
    /// between closes.
    stalled: bool,
    /// Consecutive closed windows whose busiest vCPU accrued below the
    /// floor. Any above-floor window resets it to 0; `stalled` latches true
    /// once it reaches [`TRICKLE_STALL_CONSECUTIVE_WINDOWS`].
    consecutive_stalls: u32,
    /// Busiest-vCPU CPU accrued over the most recently CLOSED window (0
    /// before the first close). Not part of the stall decision — surfaced
    /// via [`Self::last_window_accrued`] for the failure dump's
    /// `busiest_vcpu_window` evidence line and the ledger.
    last_window_accrued: u64,
    seeded: bool,
}

impl CpuTrickleTracker {
    pub(crate) fn new() -> Self {
        Self {
            anchor_ns: 0,
            anchor: Vec::new(),
            stalled: false,
            consecutive_stalls: 0,
            last_window_accrued: 0,
            seeded: false,
        }
    }

    /// Fold this tick's per-vCPU cumulative `readings` (active-currency
    /// CPU-time ns, absent → 0, index-stable per vCPU) and `now_ns` (a
    /// `run_start`-relative wall reading, monotone across ticks); return
    /// whether the guest is currently trickle-stalled. `floor_ns` is the
    /// currency-dependent stall floor ([`trickle_floor_for_currency`]) the
    /// closing window's busiest-vCPU accrual is classified against — a
    /// parameter, not captured state, so the tracker stays pure (the ledger
    /// latches the currency on the first tick, so in practice it is constant
    /// per run). Re-anchors (returns the held verdict) on a vCPU-count
    /// change rather than zipping mismatched indices.
    pub(crate) fn observe(&mut self, readings: &[u64], now_ns: u64, floor_ns: u64) -> bool {
        if !self.seeded || self.anchor.len() != readings.len() {
            self.anchor = readings.to_vec();
            self.anchor_ns = now_ns;
            self.seeded = true;
            return self.stalled;
        }
        if now_ns.saturating_sub(self.anchor_ns) >= TRICKLE_STALL_WINDOW_NS {
            // Busiest single vCPU's accrual across the window: the max over
            // vCPUs of each vCPU's OWN window delta (saturating, so an
            // absent/regressed reading contributes 0). Width-independent.
            let accrued = readings
                .iter()
                .zip(&self.anchor)
                .map(|(&r, &a)| r.saturating_sub(a))
                .max()
                .unwrap_or(0);
            self.last_window_accrued = accrued;
            if accrued < floor_ns {
                self.consecutive_stalls = self.consecutive_stalls.saturating_add(1);
            } else {
                self.consecutive_stalls = 0;
            }
            self.stalled = self.consecutive_stalls >= TRICKLE_STALL_CONSECUTIVE_WINDOWS;
            self.anchor.copy_from_slice(readings);
            self.anchor_ns = now_ns;
        }
        self.stalled
    }

    /// Busiest-vCPU CPU (ns) accrued over the most recently closed trickle
    /// window (0 before the first window closes). Dump-only evidence — the
    /// quantity the stall verdict compares against the floor.
    pub(crate) fn last_window_accrued(&self) -> u64 {
        self.last_window_accrued
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
    /// Seed at the heartbeat's zero-init value. Before the monitor thread
    /// first runs (either a pre-sampling setup pulse or a full sample tick)
    /// the heartbeat is 0; the first `observe` that sees a bumped value clears
    /// the miss counter. If no monitor ever runs the heartbeat stays 0, so
    /// this seeds the natural no-monitor stall.
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
    use crate::monitor::{CPU_CURRENCY_NONE, CPU_CURRENCY_PMU, ProgressLedger};

    // Lifecycle discriminants (mirror `monitor::LifecycleStage`; written
    // against the raw ids so this stays independent of that enum).
    const BOOT: u8 = 0;
    const BODY: u8 = 3;
    const TEST_VCPUS: u32 = 1;

    const S: u64 = 1_000_000_000;

    // Boot(0): authoritative boot headroom at one vCPU (20.15 s), wall
    // backstop 45 s (the width-scaled backstop retains this floor).
    fn boot_budget_ns() -> u64 {
        phase_cpu_budget_ns(BOOT, TEST_VCPUS)
    }
    fn boot_backstop_ns() -> u64 {
        phase_wall_backstop_ns(BOOT, TEST_VCPUS)
    }

    /// Common Boot/INFRA, monitor-live caller. Each row varies the inputs
    /// that matter to it: `max_vcpu_cpu_in_phase`, `wall_in_phase`,
    /// `runnable`, `channels`, `currency`.
    fn boot_step(
        max_vcpu_cpu_in_phase: u64,
        wall_in_phase: u64,
        runnable: bool,
        channels: bool,
        currency: u8,
    ) -> KillDecision {
        watchdog_step(
            BOOT,
            PhaseClass::Infra,
            max_vcpu_cpu_in_phase,
            wall_in_phase,
            runnable,
            channels,
            true, // monitor_live
            currency,
            TEST_VCPUS,
        )
    }

    // ---- Tier-1: spinning wedge (cpu_in_phase over the phase budget) ----

    #[test]
    fn tier1_high_cpu_fires_regardless_of_demand_or_channels() {
        // Way over the Boot CPU budget → Tier-1, whether the guest looks
        // runnable or not and whatever the channel state: a spinning wedge
        // is caught on CPU alone.
        for &runnable in &[true, false] {
            for &channels in &[true, false] {
                let d = boot_step(
                    boot_budget_ns() + 100 * S,
                    0,
                    runnable,
                    channels,
                    CPU_CURRENCY_PMU,
                );
                assert_eq!(
                    d,
                    KillDecision::Tier1CpuBudget,
                    "runnable={runnable} channels={channels}"
                );
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
            true,
            CPU_CURRENCY_PMU,
            TEST_VCPUS,
        );
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn tier1_budget_boundary_is_exclusive() {
        let budget = boot_budget_ns(); // PMU: no widening
        // Exactly at budget → not over → no fire.
        assert_eq!(
            boot_step(budget, 0, false, true, CPU_CURRENCY_PMU),
            KillDecision::None,
            "cpu == budget is not over budget"
        );
        // One ns over → fire.
        assert_eq!(
            boot_step(budget + 1, 0, false, true, CPU_CURRENCY_PMU),
            KillDecision::Tier1CpuBudget
        );
    }

    #[test]
    fn tier1_wide_idle_guest_max_stays_tiny_no_fire() {
        // THE WIDE-SMP REGRESSION ROW: a 256-vCPU idle guest in Attach whose
        // 256 vCPUs each burn a diffuse background trickle. The old summed
        // evidence charged ~141 s against a per-vCPU-linear budget and false-
        // fired; the max-per-vCPU evidence is only ~50 ms (one vCPU's share),
        // far under the flat 35 s Attach budget (pthread-widened to 52.5 s) →
        // None. Width no longer moves Tier-1. Wall (30 s) is still under the
        // 40 s Attach backstop, so Tier-2 stays quiet too — a healthy slow
        // attach is not killed.
        const ATTACH: u8 = 1;
        let max_vcpu = 50_000_000; // 50 ms: busiest single vCPU's in-phase burn
        for &currency in &[CPU_CURRENCY_PMU, CPU_CURRENCY_PTHREAD] {
            let d = watchdog_step(
                ATTACH,
                PhaseClass::Infra,
                max_vcpu,
                30 * S, // deep into the phase, short of the 40 s backstop
                false,
                true,
                true,
                currency,
                256,
            );
            assert_eq!(d, KillDecision::None, "currency={currency}");
        }
    }

    /// Regression for the 240-vCPU Cosmos verifier false kill. The old
    /// separately fitted Boot formula produced a 40.425 s pthread budget
    /// and killed a live boot at 40.487 s despite ~190 s remaining on its
    /// existing VM deadline. Boot now reuses the authoritative 56 s
    /// boot-headroom contract, widened to 84 s for pthread currency.
    /// Evidence channels were not yet live, which independently keeps the
    /// wall interval out of Tier-2.
    #[test]
    fn cosmos_240cpu_live_boot_uses_authoritative_headroom() {
        let observed_bsp_cpu = 40_487_683_525;
        let widened = widened_cpu_budget_ns(BOOT, CPU_CURRENCY_PTHREAD, 240);
        assert_eq!(widened, 84_000_000_000);
        assert_eq!(
            watchdog_step(
                BOOT,
                PhaseClass::Infra,
                observed_bsp_cpu,
                126_326_546_832,
                false,
                false,
                true,
                CPU_CURRENCY_PTHREAD,
                240,
            ),
            KillDecision::None,
        );
    }

    #[test]
    fn zero_cpu_is_never_tier1_at_any_wall() {
        // Zero CPU burned in-phase: never Tier-1, at any wall. Runnable
        // demand keeps Tier-2 exempt past the backstop, isolating Tier-1.
        for wall in [0, boot_backstop_ns() + S, 100 * S] {
            let d = boot_step(0, wall, true, true, CPU_CURRENCY_PMU);
            assert_eq!(d, KillDecision::None, "wall={wall}");
        }
    }

    // ---- Tier-2: idle wedge (INFRA + channels + !runnable + wall past
    //      backstop; deliberately NO CPU term — see the rule) ----

    #[test]
    fn tier2_idle_wedge_fires_when_all_conjuncts_hold() {
        // No runnable demand, channels live, past the wall backstop → a
        // silent idle wedge. No CPU/trickle input exists to defer it.
        let d = boot_step(0, boot_backstop_ns() + S, false, true, CPU_CURRENCY_PMU);
        assert_eq!(d, KillDecision::Tier2IdleWedge);
    }

    #[test]
    fn tier2_wide_idle_housekeeping_burn_cannot_defer() {
        // THE WIDE-IDLE REGRESSION ROW (restores approved row D-infra): a
        // 64-vCPU idle wedge whose housekeeping CPU burns 20-45 ms per 10 s
        // window (the measured width residual that kept the old trickle
        // conjunct reading "alive" and made the wedge immortal to Tier-2).
        // With the trickle conjunct gone,
        // that burn shows up only in `max_vcpu_cpu_in_phase` — far under
        // the Tier-1 budget — and Tier-2 fires on the wall backstop alone:
        // nothing runnable + INFRA past backstop IS the wedge.
        let housekeeping_max = 45_000_000; // 45 ms in-phase on the busiest vCPU
        let d = boot_step(
            housekeeping_max,
            boot_backstop_ns() + S,
            false,
            true,
            CPU_CURRENCY_PTHREAD,
        );
        assert_eq!(d, KillDecision::Tier2IdleWedge);
    }

    #[test]
    fn tier2_starved_with_runnable_work_is_exempt() {
        // THE starvation protection, carried by the runnable conjunct
        // alone: a host-starved cell WITH work always shows queued-or-
        // running tasks in its own rq memory (`nr_running` includes the
        // on-CPU task, and guest memory is readable regardless of host
        // scheduling), so it is exempt however long the wall grows. If it
        // is also inert, the Tier-3 deadman / guest scx watchdog bound it
        // — never Tier-2.
        for wall in [boot_backstop_ns() + S, 1_000 * S] {
            let d = boot_step(0, wall, true, true, CPU_CURRENCY_PMU);
            assert_eq!(d, KillDecision::None, "wall={wall}");
        }
    }

    #[test]
    fn tier2_requires_live_channels() {
        // Channels DEAD + quiet INFRA past the backstop: absence of demand
        // is blind here (the acceptance storm's 33 false kills), so Tier-2
        // must NOT fire. currency=none (the storm's pre-resolution shape)
        // so Tier-1 is also off → None.
        let d = boot_step(0, boot_backstop_ns() + S, false, false, CPU_CURRENCY_NONE);
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn tier2_needs_no_cpu_currency() {
        // Idle wedge under currency=NONE still fires: Tier-2 carries no
        // CPU term at all — it reasons about channels + demand + wall.
        let d = boot_step(0, boot_backstop_ns() + S, false, true, CPU_CURRENCY_NONE);
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
            CPU_CURRENCY_PMU,
            TEST_VCPUS,
        );
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn tier2_below_backstop_does_not_fire() {
        // All Tier-2 conjuncts but the wall is still short of the backstop.
        let d = boot_step(0, boot_backstop_ns() - S, false, true, CPU_CURRENCY_PMU);
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
        let d = boot_step(over, 0, false, false, CPU_CURRENCY_PTHREAD);
        assert_eq!(d, KillDecision::Tier1CpuBudget);
    }

    // ---- Monitor-liveness gate: suppresses Tier-1 only, not Tier-2 ----

    #[test]
    fn dead_monitor_suppresses_tier1_but_not_tier2() {
        // Tier-1 depends on the monitor's per-vCPU CPU ledger writes, so a
        // dead monitor suppresses it: a Tier-1-shaped input (high CPU) reads
        // None.
        let tier1_shape = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            boot_budget_ns() + 100 * S,
            0,
            false,
            true,
            false, // monitor dead
            CPU_CURRENCY_PMU,
            TEST_VCPUS,
        );
        assert_eq!(tier1_shape, KillDecision::None);

        // Tier-2's wall anchor is dispatch-advanced, not monitor-advanced, so
        // a dead monitor does NOT suppress it: a Tier-2-shaped idle wedge
        // (channels live, no runnable demand, past the backstop) still fires.
        // This is the fix for the dumpless-hang regression — a monitor
        // stalled >2 s under host load previously disabled the only prompt
        // catch for an idle Teardown wedge.
        let tier2_shape = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            0,
            boot_backstop_ns() + 100 * S,
            false,
            true,
            false, // monitor dead
            CPU_CURRENCY_PMU,
            TEST_VCPUS,
        );
        assert_eq!(tier2_shape, KillDecision::Tier2IdleWedge);
    }

    // ---- pthread widening: (budget, 1.5·budget] fires only under PMU ----

    #[test]
    fn pthread_widening_band_fires_only_under_pmu() {
        let budget = boot_budget_ns();
        let cpu = budget + budget / 4; // within (budget, 1.5·budget)
        assert_eq!(
            boot_step(cpu, 0, false, true, CPU_CURRENCY_PMU),
            KillDecision::Tier1CpuBudget,
            "PMU currency uses the raw budget — this burn is over it"
        );
        assert_eq!(
            boot_step(cpu, 0, false, true, CPU_CURRENCY_PTHREAD),
            KillDecision::None,
            "pthread widens the budget 3/2 — this burn is under the widened budget"
        );
    }

    #[test]
    fn pthread_widening_fires_above_widened_budget() {
        let budget = boot_budget_ns();
        let cpu = budget + budget / 2 + S; // over even the widened budget
        assert_eq!(
            boot_step(cpu, 0, false, true, CPU_CURRENCY_PTHREAD),
            KillDecision::Tier1CpuBudget
        );
    }

    #[test]
    fn widened_cpu_budget_matches_currency_widening() {
        // pthread widens the raw Boot budget 3/2; PMU leaves it raw.
        let raw = boot_budget_ns();
        assert_eq!(
            widened_cpu_budget_ns(BOOT, CPU_CURRENCY_PMU, TEST_VCPUS),
            raw
        );
        assert_eq!(
            widened_cpu_budget_ns(BOOT, CPU_CURRENCY_PTHREAD, TEST_VCPUS),
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
            CPU_CURRENCY_PMU,
            TEST_VCPUS,
        );
        assert_eq!(d, KillDecision::None);
    }

    // ---- Tier-3 host-service deadman ----

    #[test]
    fn deadman_cpu_backstop_is_width_independent_and_currency_widened() {
        let budget = 100 * S;
        assert_eq!(deadman_cpu_budget_ns(budget, CPU_CURRENCY_PMU), budget);
        assert_eq!(
            deadman_cpu_budget_ns(budget, CPU_CURRENCY_PTHREAD),
            budget + budget / 2
        );
        assert_eq!(deadman_cpu_budget_ns(budget, CPU_CURRENCY_NONE), u64::MAX);
        assert!(!deadman_cpu_budget_exhausted(
            budget,
            budget,
            CPU_CURRENCY_PMU
        ));
        assert!(deadman_cpu_budget_exhausted(
            budget + 1,
            budget,
            CPU_CURRENCY_PMU
        ));
        assert!(!deadman_cpu_budget_exhausted(
            budget + 1,
            budget,
            CPU_CURRENCY_PTHREAD
        ));
        assert!(!deadman_cpu_budget_exhausted(
            u64::MAX,
            budget,
            CPU_CURRENCY_NONE
        ));
    }

    fn host_task(
        task_id: Option<u32>,
        cpu_ns: Option<u64>,
        run_state: HostVcpuRunState,
    ) -> HostVcpuTaskSample {
        HostVcpuTaskSample {
            task_id,
            cpu_ns,
            run_state,
        }
    }

    fn host_input(
        observer_cpu_ns: u64,
        vcpu_tasks: &[HostVcpuTaskSample],
    ) -> DeadmanHostServiceInput<'_> {
        DeadmanHostServiceInput {
            monitor_terminal: false,
            vcpu_cpu_budget_exhausted: false,
            observer_cpu: DeadmanObserverClock::Reading(observer_cpu_ns),
            vcpu_tasks,
        }
    }

    #[test]
    fn host_deadman_authoritative_facts_precede_deferral_and_sensor_failure() {
        let runnable = [host_task(Some(11), Some(0), HostVcpuRunState::Runnable)];
        let unavailable = DeadmanHostServiceInput {
            monitor_terminal: true,
            vcpu_cpu_budget_exhausted: true,
            observer_cpu: DeadmanObserverClock::Unavailable,
            vcpu_tasks: &runnable,
        };
        let mut deadman = DeadmanHostService::new();
        assert_eq!(
            deadman.observe(unavailable),
            DeadmanHostDecision::Fire(DeadmanHostFire::MonitorTerminal),
            "terminal monitor is the first authoritative fact"
        );

        let cpu_exhausted = DeadmanHostServiceInput {
            monitor_terminal: false,
            vcpu_cpu_budget_exhausted: true,
            ..unavailable
        };
        assert_eq!(
            deadman.observe(cpu_exhausted),
            DeadmanHostDecision::Fire(DeadmanHostFire::VcpuCpuBudget),
            "the existing CPU backstop is authoritative over host-service deferral"
        );
    }

    #[test]
    fn host_deadman_runnable_vcpu_defers_through_arbitrary_wall_starvation() {
        // Wall time is deliberately not an input. Even enormous gaps between
        // observations cannot become kill evidence while a vCPU host task is
        // runnable; each observation re-anchors the blocked-service budget.
        let mut deadman = DeadmanHostService::new();
        let runnable = [host_task(Some(21), Some(7), HostVcpuRunState::Runnable)];
        for observer_cpu_ns in [
            0,
            DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
            10 * DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
            u64::MAX - 1,
        ] {
            assert_eq!(
                deadman.observe(host_input(observer_cpu_ns, &runnable)),
                DeadmanHostDecision::Defer(DeadmanHostDefer::Runnable)
            );
        }
    }

    #[test]
    fn host_deadman_vcpu_service_progress_restarts_blocked_budget() {
        let mut deadman = DeadmanHostService::new();
        let blocked0 = [host_task(
            Some(31),
            Some(100),
            HostVcpuRunState::NonRunnable,
        )];
        assert_eq!(
            deadman.observe(host_input(10, &blocked0)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Seeded)
        );
        assert_eq!(
            deadman.observe(host_input(
                10 + DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 1,
                &blocked0
            )),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Blocked {
                observer_service_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 1,
                budget_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
            })
        );

        // A MATERIAL advance (>= the progress floor) re-anchors; a sub-floor
        // trickle would not (see `host_deadman_sub_floor_trickle_*`).
        let blocked_progress = [host_task(
            Some(31),
            Some(100 + DEADMAN_BLOCKED_OBSERVER_PROGRESS_FLOOR_NS),
            HostVcpuRunState::NonRunnable,
        )];
        let progress_at = 10 + DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS;
        assert_eq!(
            deadman.observe(host_input(progress_at, &blocked_progress)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::VcpuServiceAdvanced)
        );
        assert_eq!(
            deadman.observe(host_input(
                progress_at + DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 1,
                &blocked_progress
            )),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Blocked {
                observer_service_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 1,
                budget_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
            }),
            "service is charged from the vCPU-progress re-anchor"
        );
    }

    #[test]
    fn host_deadman_stable_blocked_sample_is_finite_at_exact_budget() {
        let mut deadman = DeadmanHostService::new();
        let blocked = [
            host_task(Some(41), Some(1_000), HostVcpuRunState::NonRunnable),
            host_task(Some(42), None, HostVcpuRunState::Unknown),
        ];
        let anchor = 500;
        assert_eq!(
            deadman.observe(host_input(anchor, &blocked)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Seeded)
        );
        assert_eq!(
            deadman.observe(host_input(
                anchor + DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
                &blocked
            )),
            DeadmanHostDecision::Fire(DeadmanHostFire::BlockedObserverService {
                observer_service_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
                budget_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
            }),
            "the exact delivered-service boundary is finite"
        );
    }

    #[test]
    fn host_deadman_sample_changes_each_reanchor_blocked_service() {
        let mut deadman = DeadmanHostService::new();
        let original = [host_task(Some(51), Some(10), HostVcpuRunState::NonRunnable)];
        assert_eq!(
            deadman.observe(host_input(0, &original)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Seeded)
        );

        let replaced = [host_task(Some(52), Some(10), HostVcpuRunState::NonRunnable)];
        assert_eq!(
            deadman.observe(host_input(
                DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 1,
                &replaced
            )),
            DeadmanHostDecision::Defer(DeadmanHostDefer::SampleChanged),
            "task-generation replacement re-anchors"
        );

        let clock_missing = [host_task(Some(52), None, HostVcpuRunState::NonRunnable)];
        let second_anchor = 2 * DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 2;
        assert_eq!(
            deadman.observe(host_input(second_anchor, &clock_missing)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::SampleChanged),
            "vCPU clock availability change re-anchors"
        );

        let state_unknown = [host_task(Some(52), None, HostVcpuRunState::Unknown)];
        let third_anchor = 3 * DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 3;
        assert_eq!(
            deadman.observe(host_input(third_anchor, &state_unknown)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::SampleChanged),
            "normalized blocked-state change re-anchors"
        );
        assert_eq!(
            deadman.observe(host_input(
                third_anchor + DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 1,
                &state_unknown
            )),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Blocked {
                observer_service_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 1,
                budget_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
            }),
            "none of the earlier observer service leaks across re-anchors"
        );
    }

    #[test]
    fn host_deadman_per_task_service_regression_reanchors_even_if_sum_advances() {
        let mut deadman = DeadmanHostService::new();
        let before = [
            host_task(Some(61), Some(100), HostVcpuRunState::NonRunnable),
            host_task(Some(62), Some(100), HostVcpuRunState::NonRunnable),
        ];
        assert!(matches!(
            deadman.observe(host_input(0, &before)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Seeded)
        ));

        // The aggregate advanced (200 -> 250), but task 61 regressed. This
        // is a generation/race boundary, not comparable service progress.
        let raced = [
            host_task(Some(61), Some(50), HostVcpuRunState::NonRunnable),
            host_task(Some(62), Some(200), HostVcpuRunState::NonRunnable),
        ];
        assert_eq!(
            deadman.observe(host_input(
                DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 1,
                &raced
            )),
            DeadmanHostDecision::Defer(DeadmanHostDefer::VcpuServiceRegressed)
        );
    }

    #[test]
    fn host_deadman_summed_multi_vcpu_service_advance_reanchors() {
        let mut deadman = DeadmanHostService::new();
        let before = [
            host_task(Some(71), Some(10), HostVcpuRunState::NonRunnable),
            host_task(Some(72), Some(20), HostVcpuRunState::NonRunnable),
        ];
        assert!(matches!(
            deadman.observe(host_input(0, &before)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Seeded)
        ));
        // A MATERIAL summed advance (>= the progress floor) re-anchors.
        let after = [
            host_task(Some(71), Some(10), HostVcpuRunState::NonRunnable),
            host_task(
                Some(72),
                Some(20 + DEADMAN_BLOCKED_OBSERVER_PROGRESS_FLOOR_NS),
                HostVcpuRunState::NonRunnable,
            ),
        ];
        assert_eq!(
            deadman.observe(host_input(
                DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 1,
                &after
            )),
            DeadmanHostDecision::Defer(DeadmanHostDefer::VcpuServiceAdvanced)
        );
    }

    #[test]
    fn host_deadman_sub_floor_trickle_charges_blocked_budget_and_fires() {
        // THE IDLE-WEDGE FIX: a fully non-runnable guest whose cumulative CPU
        // creeps up by less than the progress floor each observation (an idle
        // guest taking its periodic timer/RCU interrupts) is trickle, NOT
        // progress. It must NOT re-anchor the blocked-observer interval every
        // tick (which previously made the idle wedge immortal to the
        // deadman); the finite blocked-observer budget must keep accruing and
        // ultimately fire.
        let mut deadman = DeadmanHostService::new();
        let trickle_ns = DEADMAN_BLOCKED_OBSERVER_PROGRESS_FLOOR_NS - 1;
        let start = [host_task(Some(91), Some(0), HostVcpuRunState::NonRunnable)];
        assert_eq!(
            deadman.observe(host_input(0, &start)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Seeded)
        );
        // Sub-floor advance halfway through the observer budget: charged, not
        // re-anchored.
        let mid = [host_task(
            Some(91),
            Some(trickle_ns),
            HostVcpuRunState::NonRunnable,
        )];
        let half = DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS / 2;
        assert_eq!(
            deadman.observe(host_input(half, &mid)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Blocked {
                observer_service_ns: half,
                budget_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
            }),
            "sub-floor trickle charges the blocked-observer budget from the \
             original anchor, it does not re-anchor"
        );
        // Another sub-floor advance carries the observer clock past the
        // budget: the blocked verdict is finite despite the perpetual
        // trickle.
        let end = [host_task(
            Some(91),
            Some(2 * trickle_ns),
            HostVcpuRunState::NonRunnable,
        )];
        assert_eq!(
            deadman.observe(host_input(DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS, &end)),
            DeadmanHostDecision::Fire(DeadmanHostFire::BlockedObserverService {
                observer_service_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
                budget_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
            })
        );
    }

    #[test]
    fn host_deadman_wide_idle_trickle_stays_sub_floor_by_max() {
        // THE WIDE-IDLE ESCAPE FIX: a 64-vCPU idle guest whose per-vCPU timer
        // trickle is sub-floor but whose SUM across 64 vCPUs clears the floor
        // many times over. The per-vCPU MAX predicate must keep it classified
        // as blocked (charging the finite budget), not re-anchor forever.
        let n = 64usize;
        let per_vcpu = DEADMAN_BLOCKED_OBSERVER_PROGRESS_FLOOR_NS / 4; // sub-floor
        // Summed advance = 64 * (floor/4) = 16 * floor — far above the floor.
        assert!((n as u64) * per_vcpu > DEADMAN_BLOCKED_OBSERVER_PROGRESS_FLOOR_NS);
        let mut deadman = DeadmanHostService::new();
        let start: Vec<_> = (0..n)
            .map(|i| host_task(Some(i as u32 + 1), Some(0), HostVcpuRunState::NonRunnable))
            .collect();
        assert_eq!(
            deadman.observe(host_input(0, &start)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Seeded)
        );
        // Every vCPU trickled sub-floor; the summed advance is 16x the floor
        // but the per-vCPU MAX is floor/4, so this is blocked, not progress.
        let trickled: Vec<_> = (0..n)
            .map(|i| {
                host_task(
                    Some(i as u32 + 1),
                    Some(per_vcpu),
                    HostVcpuRunState::NonRunnable,
                )
            })
            .collect();
        assert_eq!(
            deadman.observe(host_input(
                DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
                &trickled
            )),
            DeadmanHostDecision::Fire(DeadmanHostFire::BlockedObserverService {
                observer_service_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
                budget_ns: DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS,
            }),
            "wide idle trickle must charge the blocked budget (per-vCPU MAX \
             sub-floor), not re-anchor on the summed advance"
        );
    }

    #[test]
    fn host_deadman_wide_one_hot_vcpu_reanchors() {
        // The control for the width-independent floor: 64 vCPUs, 63 idle but
        // ONE doing floor-crossing work → genuine progress → re-anchor/defer,
        // exactly as a narrow progressing guest would.
        let n = 64usize;
        let mut deadman = DeadmanHostService::new();
        let start: Vec<_> = (0..n)
            .map(|i| host_task(Some(i as u32 + 1), Some(0), HostVcpuRunState::NonRunnable))
            .collect();
        assert_eq!(
            deadman.observe(host_input(0, &start)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Seeded)
        );
        let mut hot = start.clone();
        hot[7] = host_task(
            Some(8),
            Some(DEADMAN_BLOCKED_OBSERVER_PROGRESS_FLOOR_NS),
            HostVcpuRunState::NonRunnable,
        );
        assert_eq!(
            deadman.observe(host_input(DEADMAN_BLOCKED_OBSERVER_CPU_BUDGET_NS - 1, &hot)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::VcpuServiceAdvanced)
        );
    }

    #[test]
    fn wall_net_trips_only_past_the_deadline_multiple() {
        let budget = 150 * S; // a representative effective-deadline budget
        // Isolate the deadline term: now_wall 0 (below ceiling) and an
        // infinite backstop so the absolute term never contributes.
        let deadline_only = |wall_in_phase| wall_net_tripped(wall_in_phase, budget, 0, u64::MAX);
        // Below and at the multiple: not tripped (strict `>`).
        assert!(!deadline_only(budget * WALL_NET_DEADLINE_MULT));
        assert!(!deadline_only(budget));
        // Just past the multiple: tripped.
        assert!(deadline_only(budget * WALL_NET_DEADLINE_MULT + 1));
    }

    #[test]
    fn wall_net_never_trips_on_an_unset_deadline() {
        // A zero budget disables the deadline term; with the process clock below the
        // ceiling the net must never fire however large the in-phase wall.
        assert!(!wall_net_tripped(u64::MAX, 0, 0, u64::MAX));
    }

    #[test]
    fn wall_net_saturates_instead_of_overflowing() {
        // A pathologically large budget must not overflow the multiply.
        assert!(!wall_net_tripped(0, u64::MAX, 0, u64::MAX));
    }

    #[test]
    fn wall_net_absolute_ceiling_fires_past_ceiling_and_backstop() {
        // THE DILATION ESCAPE FIX: under extreme host oversubscription the
        // scenario-start reset inflates the effective deadline to ~the dilated
        // boot wall, so the deadline term (`4x`) lands far past the outer rail
        // and never fires. The absolute ceiling is dilation-independent: once
        // the PROCESS clock (3rd arg — queue-inclusive, tracking nextest's
        // rail; the caller passes process-start-relative wall, which on a
        // queued lane runs hundreds of seconds AHEAD of the VM's own run_start)
        // is past the ceiling AND the phase has stalled past its own wall
        // backstop, the net fires regardless of the (huge) deadline budget.
        let teardown_backstop = 15 * S;
        let huge_budget = 100_000 * S; // 4x is astronomically beyond the rail
        // Just past the ceiling AND past the phase backstop: fires.
        assert!(wall_net_tripped(
            teardown_backstop + 1,
            huge_budget,
            WALL_NET_ABSOLUTE_CEILING_NS + 1,
            teardown_backstop,
        ));
    }

    #[test]
    fn wall_net_absolute_ceiling_needs_both_the_ceiling_and_the_backstop() {
        let teardown_backstop = 15 * S;
        let huge_budget = 100_000 * S;
        // Past the ceiling but the phase only just re-anchored a milestone
        // (in-phase wall below its backstop — a legitimately-slow cell making
        // forward progress): NOT tripped.
        assert!(!wall_net_tripped(
            teardown_backstop,
            huge_budget,
            WALL_NET_ABSOLUTE_CEILING_NS + 1,
            teardown_backstop,
        ));
        // Past the backstop but below the ceiling: NOT tripped.
        assert!(!wall_net_tripped(
            teardown_backstop + 1,
            huge_budget,
            WALL_NET_ABSOLUTE_CEILING_NS,
            teardown_backstop,
        ));
        // A Body phase (u64::MAX backstop) is exempt from the absolute term
        // however long it runs past the ceiling — the idle-Body exemption.
        // budget 0 isolates the absolute term (the deadline term is off), so
        // this pins the backstop gate alone: nothing exceeds a u64::MAX
        // backstop, so the absolute term cannot fire in Body.
        assert!(!wall_net_tripped(
            u64::MAX,
            0,
            WALL_NET_ABSOLUTE_CEILING_NS * 2,
            u64::MAX,
        ));
    }

    #[test]
    fn host_deadman_observer_clock_failures_are_deterministic() {
        let blocked = [host_task(Some(81), Some(0), HostVcpuRunState::NonRunnable)];
        let mut unavailable = DeadmanHostService::new();
        assert_eq!(
            unavailable.observe(DeadmanHostServiceInput {
                monitor_terminal: false,
                vcpu_cpu_budget_exhausted: false,
                observer_cpu: DeadmanObserverClock::Unavailable,
                vcpu_tasks: &blocked,
            }),
            DeadmanHostDecision::Fire(DeadmanHostFire::SensorFailure(
                DeadmanHostSensorFailure::ObserverClockUnavailable
            ))
        );

        let mut regressed = DeadmanHostService::new();
        assert!(matches!(
            regressed.observe(host_input(100, &blocked)),
            DeadmanHostDecision::Defer(DeadmanHostDefer::Seeded)
        ));
        assert_eq!(
            regressed.observe(host_input(99, &blocked)),
            DeadmanHostDecision::Fire(DeadmanHostFire::SensorFailure(
                DeadmanHostSensorFailure::ObserverClockRegressed {
                    previous_ns: 100,
                    current_ns: 99,
                }
            ))
        );
    }

    // ---- CpuTrickleTracker ----

    #[test]
    fn trickle_reports_false_before_first_window_closes() {
        let mut t = CpuTrickleTracker::new();
        // Seed tick, then sub-window ticks: not enough evidence yet.
        assert!(!t.observe(&[0], 0, TRICKLE_FLOOR_NS));
        assert!(!t.observe(&[0], S, TRICKLE_FLOOR_NS));
        assert!(!t.observe(&[0], TRICKLE_STALL_WINDOW_NS - S, TRICKLE_FLOOR_NS));
    }

    // Window length alias for the multi-window trickle tests below.
    const W: u64 = TRICKLE_STALL_WINDOW_NS;

    #[test]
    fn trickle_single_stalled_window_is_not_stalled() {
        // One sub-floor window is NOT enough: a single 10 s window can
        // legitimately accrue below the floor on a live cell at a 100-300x
        // scheduling tail (the trickle-window-misread residual). The streak
        // must reach TRICKLE_STALL_CONSECUTIVE_WINDOWS.
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(&[0], 0, TRICKLE_FLOOR_NS)); // seed
        assert!(
            !t.observe(&[10_000], W, TRICKLE_FLOOR_NS),
            "one sub-floor (10 µs) window must NOT report stalled"
        );
    }

    #[test]
    fn trickle_idle_cell_stalls_after_two_windows() {
        // Idle/wedged guest under PMU currency: only in-guest ISR trickle
        // (10 µs) accrues each 10 s window — below the 1 ms floor. Two
        // CONSECUTIVE sub-floor windows → stalled.
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(&[0], 0, TRICKLE_FLOOR_NS)); // seed
        assert!(
            !t.observe(&[10_000], W, TRICKLE_FLOOR_NS),
            "window 1 sub-floor → streak 1, not yet stalled"
        );
        assert!(
            t.observe(&[20_000], 2 * W, TRICKLE_FLOOR_NS),
            "window 2 consecutively sub-floor → streak 2 → stalled"
        );
    }

    #[test]
    fn trickle_starved_cell_stays_alive() {
        // Host-starved cell at ~40x dilation still accrues tens of ms/s;
        // over a 10 s window that is hundreds of ms — far above the floor →
        // NOT stalled (streak never even reaches 1).
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(&[0], 0, TRICKLE_FLOOR_NS)); // seed
        assert!(
            !t.observe(&[200_000_000], W, TRICKLE_FLOOR_NS),
            "200 ms over the window is above the floor → alive"
        );
        assert!(
            !t.observe(&[400_000_000], 2 * W, TRICKLE_FLOOR_NS),
            "still above the floor a second window → alive"
        );
    }

    #[test]
    fn trickle_interleaved_recovery_resets_streak() {
        // A window that recovers above the floor between two sub-floor
        // windows resets the streak, so bursty starvation never latches.
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(&[0], 0, TRICKLE_FLOOR_NS)); // seed
        assert!(
            !t.observe(&[0], W, TRICKLE_FLOOR_NS),
            "sub-floor → streak 1"
        );
        assert!(
            !t.observe(&[200_000_000], 2 * W, TRICKLE_FLOOR_NS),
            "above floor → streak reset to 0"
        );
        assert!(
            !t.observe(&[200_000_000], 3 * W, TRICKLE_FLOOR_NS),
            "sub-floor again → only streak 1"
        );
        assert!(
            t.observe(&[200_000_000], 4 * W, TRICKLE_FLOOR_NS),
            "second consecutive sub-floor → streak 2 → stalled"
        );
    }

    #[test]
    fn trickle_floor_boundary() {
        // Exactly the floor is NOT below the floor → never accrues a stall,
        // even across two windows.
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(&[0], 0, TRICKLE_FLOOR_NS));
        assert!(!t.observe(&[TRICKLE_FLOOR_NS], W, TRICKLE_FLOOR_NS));
        assert!(!t.observe(&[2 * TRICKLE_FLOOR_NS], 2 * W, TRICKLE_FLOOR_NS));

        // One ns under the floor is sub-floor, but a single window is not
        // yet stalled; a second consecutive sub-floor window crosses the
        // streak.
        let mut t2 = CpuTrickleTracker::new();
        assert!(!t2.observe(&[0], 0, TRICKLE_FLOOR_NS));
        assert!(
            !t2.observe(&[TRICKLE_FLOOR_NS - 1], W, TRICKLE_FLOOR_NS),
            "one sub-floor window not yet stalled"
        );
        assert!(
            t2.observe(&[2 * (TRICKLE_FLOOR_NS - 1)], 2 * W, TRICKLE_FLOOR_NS),
            "second consecutive sub-floor window → stalled"
        );
    }

    #[test]
    fn trickle_holds_verdict_between_window_closures() {
        // After a stalled verdict (two consecutive sub-floor windows),
        // sub-window ticks keep reporting stalled until the next window
        // closes and re-evaluates — here to alive on a fresh accrual, which
        // also resets the streak.
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(&[0], 0, TRICKLE_FLOOR_NS)); // seed
        assert!(!t.observe(&[0], W, TRICKLE_FLOOR_NS)); // window 1 sub-floor → streak 1
        assert!(t.observe(&[0], 2 * W, TRICKLE_FLOOR_NS)); // window 2 sub-floor → stalled
        // Sub-window tick, still stalled (held).
        assert!(t.observe(&[500_000_000], 2 * W + S, TRICKLE_FLOOR_NS));
        // Next full window with big accrual → re-evaluates to alive.
        assert!(!t.observe(&[1_000_000_000], 3 * W + S, TRICKLE_FLOOR_NS));
    }

    // ---- Currency-dependent trickle floor ----

    #[test]
    fn trickle_floor_for_currency_selects_per_currency() {
        // PMU keeps the tight guest-only floor; pthread (and the
        // no-currency tick) get the widened floor that absorbs VM-exit
        // overhead charged to an idle guest.
        assert_eq!(
            trickle_floor_for_currency(CPU_CURRENCY_PMU),
            TRICKLE_FLOOR_NS
        );
        assert_eq!(
            trickle_floor_for_currency(CPU_CURRENCY_PTHREAD),
            TRICKLE_FLOOR_PTHREAD_NS
        );
        assert_eq!(
            trickle_floor_for_currency(CPU_CURRENCY_NONE),
            TRICKLE_FLOOR_PTHREAD_NS
        );
    }

    #[test]
    fn trickle_idle_pthread_accrual_stalls_under_pthread_floor() {
        // The measured 1-vCPU idle-wedge population under pthread currency:
        // 1-10 ms per 10 s window (VM-exit overhead for residual idle
        // kernel ticks). 10 ms/window sits ABOVE the 1 ms PMU floor — the
        // miscalibration that once made an idle wedge invisible to the
        // trickle consumers — but BELOW the 25 ms pthread floor, so two
        // consecutive windows latch stalled and the deadman can fire on an
        // inert cell.
        let f = trickle_floor_for_currency(CPU_CURRENCY_PTHREAD);
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(&[0], 0, f)); // seed
        assert!(
            !t.observe(&[10_000_000], W, f),
            "window 1: 10 ms idle-pthread accrual is sub-floor → streak 1"
        );
        assert!(
            t.observe(&[20_000_000], 2 * W, f),
            "window 2: consecutively sub-floor → stalled"
        );

        // Control: the same 10 ms/window accrual under the PMU floor never
        // stalls — this is exactly the pre-calibration immortality shape.
        let mut pmu = CpuTrickleTracker::new();
        assert!(!pmu.observe(&[0], 0, TRICKLE_FLOOR_NS));
        assert!(!pmu.observe(&[10_000_000], W, TRICKLE_FLOOR_NS));
        assert!(!pmu.observe(&[20_000_000], 2 * W, TRICKLE_FLOOR_NS));
    }

    #[test]
    fn trickle_wide_idle_busiest_vcpu_stays_sub_floor() {
        // ROTATION-COLLAPSE ARITHMETIC: 64 vCPUs whose background burn is
        // SPREAD across them — the sum is ~38 ms per 10 s window (above the
        // 25 ms pthread floor) but each vCPU's OWN window accrual is only
        // ~0.6 ms, so the per-vCPU windowed MAX is sub-floor and two
        // consecutive windows latch stalled. This pins the property that a
        // summed or per-tick-max currency lacks. (A REAL wide idle guest
        // additionally concentrates 20-45 ms of housekeeping duty on one
        // vCPU — over the floor — which is why the trickle verdict gates
        // only the deadman's deferral, never a Tier-2 kill.)
        let f = trickle_floor_for_currency(CPU_CURRENCY_PTHREAD);
        let mut t = CpuTrickleTracker::new();
        let anchor = vec![0u64; 64];
        assert!(!t.observe(&anchor, 0, f)); // seed
        // Window 1: each vCPU advanced a jittered 0.3-0.9 ms (sum ~38 ms,
        // busiest single vCPU 0.9 ms < 25 ms floor).
        let w1: Vec<u64> = (0..64)
            .map(|i| 300_000 + (i as u64 % 7) * 100_000)
            .collect();
        assert!(
            !t.observe(&w1, W, f),
            "window 1 busiest-vCPU sub-floor → streak 1"
        );
        // Window 2: another ~0.3-0.9 ms per vCPU on top → busiest still
        // sub-floor → stalled.
        let w2: Vec<u64> = w1
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 300_000 + (i as u64 % 7) * 100_000)
            .collect();
        assert!(
            t.observe(&w2, 2 * W, f),
            "window 2 consecutively sub-floor → stalled"
        );
    }

    #[test]
    fn trickle_wide_starved_alive_one_hot_vcpu_stays_alive() {
        // A wide but starved-ALIVE guest: most vCPUs idle (µs deltas) but
        // ONE vCPU lands a real 100 ms slice each window — the per-vCPU
        // windowed max surfaces it, so the busiest vCPU is above the floor →
        // never stalled. Width does not hide a live vCPU.
        let f = trickle_floor_for_currency(CPU_CURRENCY_PTHREAD);
        let mut t = CpuTrickleTracker::new();
        let mut cum = vec![0u64; 64];
        assert!(!t.observe(&cum, 0, f)); // seed
        for win in 1..=3u64 {
            for (i, c) in cum.iter_mut().enumerate() {
                *c += if i == 40 { 100_000_000 } else { 20_000 };
            }
            assert!(
                !t.observe(&cum, win * W, f),
                "one hot vCPU at 100 ms/window keeps the busiest above the floor → alive"
            );
        }
    }

    #[test]
    fn trickle_vcpu_count_change_re_anchors() {
        // A vCPU-count change re-anchors (returns the held verdict) rather
        // than zipping mismatched-length slices.
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(&[0, 0], 0, TRICKLE_FLOOR_NS)); // seed at 2 vCPUs
        // Grow to 3 vCPUs mid-run: re-anchor, hold (still not stalled).
        assert!(!t.observe(&[0, 0, 0], W, TRICKLE_FLOOR_NS));
        // From the fresh 3-vCPU anchor, a full sub-floor window streaks.
        assert!(!t.observe(&[10_000, 10_000, 10_000], 2 * W, TRICKLE_FLOOR_NS));
        assert!(t.observe(&[20_000, 20_000, 20_000], 3 * W, TRICKLE_FLOOR_NS));
    }

    #[test]
    fn trickle_starved_pthread_accrual_stays_alive_under_pthread_floor() {
        // The measured starved-but-alive population: 100+ ms per 10 s
        // window even at ~100x dilation — above the 25 ms pthread floor
        // with ~4x margin, so a starved cell is never misjudged as
        // stalled.
        let f = trickle_floor_for_currency(CPU_CURRENCY_PTHREAD);
        let mut t = CpuTrickleTracker::new();
        assert!(!t.observe(&[0], 0, f)); // seed
        assert!(
            !t.observe(&[100_000_000], W, f),
            "100 ms/window starved-alive accrual is above the pthread floor → alive"
        );
        assert!(
            !t.observe(&[200_000_000], 2 * W, f),
            "still above the floor a second window → alive"
        );
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
        assert!(
            !last,
            "a heartbeat frozen at 0 latches not-live after N ticks"
        );
    }

    #[test]
    fn bootstrap_heartbeat_keeps_monitor_live_without_sample_evidence() {
        let ledger = ProgressLedger::default();
        let mut live = MonitorLiveness::new();

        // Run for twice the frozen-heartbeat miss budget. A pre-sampling
        // monitor that is alive and pulsing must never be mistaken for a
        // dead sensor even though it cannot publish guest evidence yet.
        for tick in 1..=2 * WATCHDOG_MONITOR_LIVENESS_MISS_TICKS {
            ledger.record_monitor_heartbeat();
            let snapshot = ledger.snapshot();
            assert!(
                live.observe(snapshot.monitor_heartbeat),
                "bootstrap pulse {tick} must keep the monitor live"
            );
            assert_eq!(snapshot.cpu_currency, CPU_CURRENCY_NONE);
            assert_eq!(snapshot.max_vcpu_cpu_in_phase_ns, 0);
            assert!(!snapshot.cpu_trickle_stalled);
            assert!(!snapshot.runnable_demand);
            assert!(!snapshot.evidence_channels_live);
        }
    }

    // ---- evaluate_progress: ledger snapshot → tier decision glue ----

    fn snap(
        phase: u8,
        max_vcpu_cpu_in_phase_ns: u64,
        wall_ns_at_progress: u64,
        runnable_demand: bool,
        channels_live: bool,
        cpu_currency: u8,
    ) -> LedgerSnapshot {
        LedgerSnapshot {
            phase,
            phase_epoch: 0,
            // evaluate_progress reads only the max-per-vCPU field (Tier-1)
            // and the demand/channel/wall fields (Tier-2); the ledger's
            // cpu_trickle_stalled / cpu_ns_now / busiest_vcpu_window_ns feed
            // the deadman and the dump, not this glue — inert here.
            cpu_ns_now: 0,
            max_vcpu_cpu_in_phase_ns,
            cpu_trickle_stalled: false,
            busiest_vcpu_window_ns: 0,
            wall_ns_at_progress,
            progress_epoch: 0,
            monitor_heartbeat: 0,
            monitor_terminal: false,
            runnable_demand,
            cpu_currency,
            evidence_channels_live: channels_live,
        }
    }

    #[test]
    fn evaluate_progress_infra_idle_wedge_fires_tier2() {
        // INFRA (Boot), live monitor, no milestone (wall_at_progress==0) so
        // wall_in_phase == now past the backstop, no runnable demand,
        // channels live → a silent idle wedge, no CPU term consulted.
        let now = boot_backstop_ns() + S;
        let d = evaluate_progress(
            &snap(BOOT, 0, 0, false, true, CPU_CURRENCY_PMU),
            now,
            true, // monitor_live
            TEST_VCPUS,
        );
        assert_eq!(d, KillDecision::Tier2IdleWedge);
    }

    #[test]
    fn evaluate_progress_channels_dead_suppresses_tier2() {
        // Same idle shape but channels dead (the storm's early-boot blind
        // window) → Tier-2 suppressed → None.
        let now = boot_backstop_ns() + S;
        let d = evaluate_progress(
            &snap(BOOT, 0, 0, false, false, CPU_CURRENCY_PMU),
            now,
            true,
            TEST_VCPUS,
        );
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn evaluate_progress_channels_dead_still_fires_tier1() {
        // Storm scenario end-to-end: channels dead, currency=pthread, the
        // busiest vCPU's in-phase burn over the widened budget → Tier-1
        // STILL fires (host-side CPU evidence).
        let over = boot_budget_ns() + boot_budget_ns() / 2 + S;
        let d = evaluate_progress(
            &snap(BOOT, over, 0, false, false, CPU_CURRENCY_PTHREAD),
            0,
            true,
            TEST_VCPUS,
        );
        assert_eq!(d, KillDecision::Tier1CpuBudget);
    }

    #[test]
    fn evaluate_progress_initial_state_counts_from_run_start() {
        // Before the monitor re-anchors, the max-in-phase reads whatever the
        // monitor last published; a Boot whose busiest vCPU burned past its
        // CPU budget fires Tier-1.
        let d = evaluate_progress(
            &snap(
                BOOT,
                boot_budget_ns() + 100 * S,
                0,
                false,
                false,
                CPU_CURRENCY_PMU,
            ),
            0,
            true,
            TEST_VCPUS,
        );
        assert_eq!(d, KillDecision::Tier1CpuBudget);
    }

    #[test]
    fn evaluate_progress_dead_monitor_suppresses_tier1_only() {
        // A dead monitor suppresses Tier-1 (its CPU currency is monitor-
        // sourced): a Tier-1-shaped snapshot (busiest vCPU over budget) reads
        // None.
        let tier1 = evaluate_progress(
            &snap(
                BOOT,
                boot_budget_ns() + 100 * S,
                0,
                false,
                false,
                CPU_CURRENCY_PMU,
            ),
            0,
            false, // monitor dead
            TEST_VCPUS,
        );
        assert_eq!(tier1, KillDecision::None);

        // But NOT Tier-2: its wall anchor is dispatch-advanced, so an idle
        // wedge (channels live, no demand, past the backstop) still fires
        // through a stalled monitor — the dumpless-hang fix.
        let now = boot_backstop_ns() + S;
        let tier2 = evaluate_progress(
            &snap(BOOT, 0, 0, false, true, CPU_CURRENCY_PMU),
            now,
            false, // monitor dead
            TEST_VCPUS,
        );
        assert_eq!(tier2, KillDecision::Tier2IdleWedge);
    }

    #[test]
    fn evaluate_progress_body_never_fires() {
        let now = 1_000 * S;
        let d = evaluate_progress(
            &snap(BODY, u64::MAX - 1, 0, false, true, CPU_CURRENCY_PMU),
            now,
            true,
            TEST_VCPUS,
        );
        assert_eq!(d, KillDecision::None);
    }
}
