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
//!   - Tier-3 (the guest-derived hard deadline) lives in the watchdog
//!     thread ([`super`]); [`deadman_should_fire`] here is its deferral
//!     gate. It fires at the wall deadline when the monitor is dead, the
//!     cell is inert (CPU trickle-stalled AND no milestone within the
//!     grace), or the current phase has consumed more busiest-vCPU CPU
//!     than the VM's entire effective deadline budget. The last condition
//!     is a dilation-immune bound for an active Body livelock: host
//!     starvation stretches the wall needed to spend the budget, while a
//!     guest that keeps burning CPU can no longer defer forever.
//!
//! The CPU-trickle discriminator ([`CpuTrickleTracker`]) serves the Tier-3
//! deadman's deferral gate ONLY: a starved-but-alive cell (even at ~40x
//! host dilation) still lands tens of ms/s of guest CPU on its busiest
//! vCPU → not stalled → the deadman defers a slow cell rather than
//! false-killing it. It is denominated in the BUSIEST single vCPU's
//! windowed accrual (`max_i(C_i(now) - C_i(window_start))` over the
//! per-vCPU cumulatives), not the summed CPU nor a per-tick max: those grow
//! with vCPU count faster (a wide idle guest's rotating background burn
//! sums ~64x above a per-vCPU floor), while the per-vCPU windowed max
//! collapses the rotation. Computed monitor-side and published as
//! `cpu_trickle_stalled`. It is deliberately NOT a Tier-2 conjunct: the
//! guest housekeeping CPU's timekeeping/RCU tick burns 20-45 ms/10 s at 64
//! vCPUs (scaling with width), so no width-stable floor can separate
//! wide-idle from starved-alive there — the runnable conjunct carries that
//! protection instead (see the Tier-2 rule). The same width residual in the
//! deadman only DEFERS (never kills); its runnable-piled remainder is
//! bounded by the guest scx watchdog (see [`deadman_should_fire`]).

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
/// tick jitter and the bursty ISR pattern on parked vCPUs, yet short enough
/// that the deadman's stall verdict refreshes several times inside its
/// 60 s milestone grace.
const TRICKLE_STALL_WINDOW_NS: u64 = 10_000_000_000;

/// Minimum busiest-vCPU guest CPU (ns) that must accrue over one
/// [`TRICKLE_STALL_WINDOW_NS`] for the guest to count as "receiving CPU"
/// (NOT trickle-stalled) under the PMU currency. 1 ms / 10 s discriminates
/// the two populations the deadman's deferral gate must tell apart: an
/// idle-or-wedged guest still takes timer interrupts on its parked vCPUs
/// but the PMU SW task-clock (`exclude_host`) counts only guest-mode
/// execution, so its busiest vCPU accrues just the in-guest ISR bodies —
/// tens of µs over the window at 1 vCPU, below the floor → stalled — while
/// even a cell starved at ~40x host dilation lands tens of ms/s on its
/// busiest vCPU (hundreds of ms over the window — far above the floor →
/// not stalled → the deadman defers).
///
/// CALIBRATED ON 1-vCPU CELLS. At width the guest housekeeping CPU's
/// timekeeping/RCU duty grows with vCPU count and can lift an idle guest's
/// busiest vCPU over this floor (see [`CpuTrickleTracker`]); the misread
/// only DEFERS the deadman — Tier-2 owns the idle-wedge kill and carries
/// no trickle term — so the 1-vCPU calibration stays safe to keep.
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
/// margin in both directions. The deadman additionally requires a 60 s-stale
/// milestone ([`TIER3_PROGRESS_GRACE_NS`]), so the joint false-kill
/// condition is sub-0.25%-CPU-delivery for 60+ s at the wall deadline — a
/// defensible operator bound.
///
/// CALIBRATED ON 1-vCPU CELLS (where the summed and busiest-vCPU
/// currencies coincide). At width the housekeeping-CPU duty lifts an idle
/// guest's busiest vCPU to 20-45 ms per window at 64 vCPUs — over this
/// floor — so the deadman can misread a wide idle guest as alive and
/// DEFER. That is safe: Tier-2 owns the idle-wedge kill without any
/// trickle term, and the deferral's runnable-piled remainder is bounded by
/// the guest scx watchdog (see [`deadman_should_fire`]).
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

/// The trickle-stall floor (ns per [`TRICKLE_STALL_WINDOW_NS`]) for the
/// trust level of the CPU currency — [`widen_budget_for_currency`]'s
/// sibling for the [`CpuTrickleTracker`]. PMU measures guest-only time, so
/// the tight [`TRICKLE_FLOOR_NS`] applies; pthread time inflates an idle
/// guest's trickle with VM-exit overhead, so the widened
/// [`TRICKLE_FLOOR_PTHREAD_NS`] applies (see its doc for the measured
/// populations). `CPU_CURRENCY_NONE` gets the pthread floor too: with no
/// per-vCPU source this tick the readings carry no trusted magnitude, and
/// the wider floor is the conservative-against-immortality choice — an
/// inert cell must stay deadman-killable — while the deadman's 60 s
/// milestone grace still guards a live cell.
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
///     hot thread, so its max crosses a flat budget; a wide idle guest's
///     summed background burn does NOT (that summed number lives in
///     `cpu_ns_now`, which no longer feeds any tier).
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
///     writes.
///   - `cpu_currency`: provenance of `max_vcpu_cpu_in_phase_ns`
///     ([`CPU_CURRENCY_NONE`]/`PTHREAD`/`PMU`).
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
    // Tier-1 is off (Tier-2 below carries no CPU term at all). Body's
    // budget is the `u64::MAX` sentinel, so Tier-1 is structurally off for
    // Body via the budget table, not a class check.
    if cpu_currency != CPU_CURRENCY_NONE {
        let budget = widen_budget_for_currency(phase_cpu_budget_ns(phase), cpu_currency);
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
    //     definition. If a runnable-exempted cell is also stalled, the
    //     Tier-3 deadman / guest scx watchdog bound it rather than Tier-2.
    //
    // Deliberately NO CPU-trickle conjunct. Trickle here was width-broken
    // belt-and-braces duplicating the runnable protection: the guest
    // housekeeping CPU's timekeeping/RCU tick burns 20-45 ms per 10 s
    // window at 64 vCPUs (measured; scales with width), so no width-stable
    // floor exists — a wide idle wedge read "not stalled" and became
    // immortal to Tier-2 AND the trickle-gated deadman. The trickle
    // discriminator remains the DEADMAN's deferral gate only
    // ([`deadman_should_fire`]), where its residual merely defers.
    if class == PhaseClass::Infra
        && channels_live
        && !runnable_demand
        && wall_in_phase_ns > phase_wall_backstop_ns(phase)
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
/// [`phase_cpu_budget_ns`]. Width-independent — no vCPU count.
pub(crate) fn widened_cpu_budget_ns(phase: u8, cpu_currency: u8) -> u64 {
    widen_budget_for_currency(phase_cpu_budget_ns(phase), cpu_currency)
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

/// Fold one live [`LedgerSnapshot`] into a [`KillDecision`]: the glue
/// between the ledger and the pure [`watchdog_step`]. Derives the wall
/// in-phase delta and maps the lifecycle stage onto a [`PhaseClass`],
/// then defers to `watchdog_step`. Kept pure (the wall clock arrives as
/// `now_wall_ns`) so the ledger-read → tier-decision path is unit-testable
/// without a live watchdog thread. The ledger's `cpu_trickle_stalled` is
/// NOT consumed here — it gates only the Tier-3 deadman
/// ([`deadman_should_fire`]), which the watchdog thread evaluates
/// separately.
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
    )
}

/// Grace window for the Tier-3 deadman (the guest-derived hard deadline):
/// once the wall deadline has elapsed, an otherwise-unbounded active-cell
/// deferral ends when its CPU backstop is exhausted; an inert-cell deferral
/// ends after this milestone grace.
///
/// TERMINATION STORY (exactly this shape):
///   - A starved-but-alive cell keeps accruing guest CPU (tens of ms/s
///     even at ~40x dilation), so it is NOT trickle-stalled and is
///     DEFERRED past the wall deadline by design. It remains bounded by the
///     dilation-immune busiest-vCPU budget: a slow cell spends that budget
///     slowly, while an active livelock eventually exhausts it.
///   - An idle userspace wedge on a LIVE kernel (the classic PID-1-blocked
///     shape) shows no runnable demand, so Tier-2 kills it at the wall
///     backstop long before this deadline; the deadman is its backstop-of-
///     last-resort should Tier-2's channel gate stay blind. (Milestone-only
///     progress is what makes this work: kernel scheduling noise would
///     otherwise reset a wall-since-progress clock every tick and defer
///     forever.)
///   - A runnable-piled-up stalled-scx cell has `runnable_demand=true`, so
///     Tier-2 exempts it — its CPU also trickle-stalls (tasks never run),
///     so this deadman bounds it at the wall deadline. At wide vCPU counts
///     the trickle can misread such a cell as alive (the housekeeping-CPU
///     width residual — see [`CpuTrickleTracker`]) and this deadman then
///     DEFERS; that residual is still bounded, by the guest scx watchdog
///     (`scx_tick` fires `SCX_EXIT_ERROR_STALL` on a stalled scheduler),
///     not by wall deferral here.
///   - An INFRA spinning wedge never reaches here: Tier-1 bounds it on the
///     phase CPU budget long before the deadline. Body deliberately has no
///     Tier-1 budget, so the whole-VM CPU backstop closes that gap here.
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
///     guest is alive and merely slow → defer. This deadman is the trickle
///     discriminator's ONLY consumer (Tier-2 dropped it as width-broken
///     belt-and-braces): here a width residual can only DEFER a kill, and
///     the runnable-piled cell it would defer is bounded by the guest scx
///     watchdog instead.
///   - `max_vcpu_cpu_in_phase_ns`, `effective_deadline_budget_ns`, and
///     `cpu_currency`: the active-cell CPU backstop evidence, budget, and
///     provenance. [`deadman_cpu_budget_exhausted`] owns the comparison and
///     pthread widening.
///
/// Fires iff the monitor is dead, the active-cell CPU backstop is exhausted,
/// OR the cell is inert: CPU trickle-stalled AND no milestone within the full
/// [`TIER3_PROGRESS_GRACE_NS`].
pub(crate) fn deadman_should_fire(
    monitor_live: bool,
    wall_since_milestone_ns: u64,
    cpu_trickle_stalled: bool,
    max_vcpu_cpu_in_phase_ns: u64,
    effective_deadline_budget_ns: u64,
    cpu_currency: u8,
) -> bool {
    !monitor_live
        || deadman_cpu_budget_exhausted(
            max_vcpu_cpu_in_phase_ns,
            effective_deadline_budget_ns,
            cpu_currency,
        )
        || (cpu_trickle_stalled && wall_since_milestone_ns > TIER3_PROGRESS_GRACE_NS)
}

/// Pure state machine that turns each tick's PER-vCPU cumulative CPU-time
/// readings into a boolean "the guest is receiving essentially no CPU"
/// (trickle-stalled) — the Tier-3 deadman's deferral discriminator
/// ([`deadman_should_fire`]), its ONLY consumer. No clocks, no atomics — so
/// its window arithmetic is unit-testable in isolation. Driven by the
/// MONITOR (which owns the per-vCPU CPU data), not the watchdog: the
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
/// wide idle guest as alive. That misread is why this discriminator gates
/// ONLY the deadman's deferral (where it can merely DEFER a kill, bounded
/// elsewhere) and is NOT a Tier-2 conjunct.
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

    // Boot(0): flat cpu budget 15 s, wall backstop 45 s.
    fn boot_budget_ns() -> u64 {
        phase_cpu_budget_ns(BOOT)
    }
    fn boot_backstop_ns() -> u64 {
        phase_wall_backstop_ns(BOOT)
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
            );
            assert_eq!(d, KillDecision::None, "currency={currency}");
        }
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
        // conjunct reading "alive" and made the wedge immortal to Tier-2
        // AND the trickle-gated deadman). With the trickle conjunct gone,
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
            false, // monitor dead
            CPU_CURRENCY_PMU,
        );
        assert_eq!(tier1_shape, KillDecision::None);

        let tier2_shape = watchdog_step(
            BOOT,
            PhaseClass::Infra,
            0,
            boot_backstop_ns() + 100 * S,
            false,
            true,
            false, // monitor dead
            CPU_CURRENCY_PMU,
        );
        assert_eq!(tier2_shape, KillDecision::None);
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
        assert_eq!(widened_cpu_budget_ns(BOOT, CPU_CURRENCY_PMU), raw);
        assert_eq!(
            widened_cpu_budget_ns(BOOT, CPU_CURRENCY_PTHREAD),
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
        );
        assert_eq!(d, KillDecision::None);
    }

    // ---- Tier-3 deadman deferral: deadman_should_fire ----

    #[test]
    fn deadman_dead_monitor_always_fires() {
        // Monitor dead → fire regardless of wall / trickle (the machinery
        // that could defer is gone).
        assert!(deadman_should_fire(
            false,
            0,
            false,
            0,
            100 * S,
            CPU_CURRENCY_PMU
        ));
        assert!(deadman_should_fire(
            false,
            100 * S,
            true,
            0,
            100 * S,
            CPU_CURRENCY_PMU,
        ));
    }

    #[test]
    fn deadman_starved_but_alive_defers_while_cpu_budget_remains() {
        // Monitor live, NOT trickle-stalled (CPU still accruing) — a
        // starved-but-alive cell defers past the wall deadline while it has
        // not spent the dilation-immune CPU backstop.
        assert!(!deadman_should_fire(
            true,
            TIER3_PROGRESS_GRACE_NS + S,
            false,
            99 * S,
            100 * S,
            CPU_CURRENCY_PMU,
        ));
        assert!(!deadman_should_fire(
            true,
            1_000 * S,
            false,
            99 * S,
            100 * S,
            CPU_CURRENCY_PMU,
        ));
    }

    #[test]
    fn deadman_active_cell_fires_when_cpu_budget_is_exhausted() {
        assert!(deadman_should_fire(
            true,
            0,
            false,
            100 * S + 1,
            100 * S,
            CPU_CURRENCY_PMU,
        ));
    }

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

    #[test]
    fn deadman_inert_cell_fires_past_grace() {
        // Monitor live, trickle-stalled, and no milestone for longer than
        // the grace → the cell is inert → fire (the idle-userspace-wedge /
        // stalled-scx bound). Strict `>` on the grace boundary.
        assert!(deadman_should_fire(
            true,
            TIER3_PROGRESS_GRACE_NS + 1,
            true,
            0,
            100 * S,
            CPU_CURRENCY_PMU,
        ));
        assert!(
            !deadman_should_fire(
                true,
                TIER3_PROGRESS_GRACE_NS,
                true,
                0,
                100 * S,
                CPU_CURRENCY_PMU,
            ),
            "exactly at the grace is not yet past it"
        );
    }

    #[test]
    fn deadman_trickle_stalled_within_grace_defers() {
        // Trickle-stalled but a milestone landed within the grace → defer
        // (a recent milestone means the cell is still advancing its
        // lifecycle).
        assert!(!deadman_should_fire(
            true,
            TIER3_PROGRESS_GRACE_NS - S,
            true,
            0,
            100 * S,
            CPU_CURRENCY_PMU,
        ));
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
        );
        assert_eq!(d, KillDecision::Tier2IdleWedge);
    }

    #[test]
    fn evaluate_progress_channels_dead_suppresses_tier2() {
        // Same idle shape but channels dead (the storm's early-boot blind
        // window) → Tier-2 suppressed → None.
        let now = boot_backstop_ns() + S;
        let d = evaluate_progress(&snap(BOOT, 0, 0, false, false, CPU_CURRENCY_PMU), now, true);
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
        );
        assert_eq!(d, KillDecision::Tier1CpuBudget);
    }

    #[test]
    fn evaluate_progress_dead_monitor_suppresses_tiers() {
        let now = boot_backstop_ns() + S;
        let d = evaluate_progress(
            &snap(BOOT, 0, 0, false, true, CPU_CURRENCY_PMU),
            now,
            false, // monitor dead
        );
        assert_eq!(d, KillDecision::None);
    }

    #[test]
    fn evaluate_progress_body_never_fires() {
        let now = 1_000 * S;
        let d = evaluate_progress(
            &snap(BODY, u64::MAX - 1, 0, false, true, CPU_CURRENCY_PMU),
            now,
            true,
        );
        assert_eq!(d, KillDecision::None);
    }
}
