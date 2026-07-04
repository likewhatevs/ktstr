//! State types, constants, and pure helpers shared across the freeze
//! coordinator's call sites.
//!
//! Two kinds of item live here: plain state types/constants (no
//! behaviour) and pure, side-effect-free helpers (deterministic
//! functions of their arguments — no I/O, no guest-memory access).
//! Stateful behaviour (the run-loop closure, freeze rendezvous, guest
//! memory reads) lives in the modules that consume these (the run-loop
//! closure in [`super`], the snapshot request handlers in
//! [`super::snapshot`]). Splitting them out here lets the closure body
//! shrink, lets each consumer share one vocabulary, and makes the pure
//! helpers unit-testable in isolation.
//!
//! Types and constants:
//!
//! * [`FREEZE_RENDEZVOUS_TIMEOUT`] — wall-clock budget for parked-
//!   vCPU rendezvous and the matching post-thaw barrier.
//! * [`BspExitReason`] — diagnostic enum logged when the BSP run
//!   loop breaks.
//! * [`SnapshotRequest`] — typed view of a guest-side
//!   `MSG_TYPE_SNAPSHOT_REQUEST` TLV.
//! * [`FreezeState`] — the dump state machine the run-loop closure
//!   advances on each freeze cycle.
//! * [`PeriodicWindow`] — the resolved periodic-capture window (slice
//!   bounds + scenario-relative offset anchor).
//!
//! Pure helpers:
//!
//! * [`periodic_tag`] — the `"periodic_NNN"` snapshot tag for a
//!   boundary index.
//! * [`compute_periodic_boundaries_ns`] — the boundary timestamps
//!   sliced across a `[window_start, window_end]` span.
//! * [`resolve_periodic_window`] — float/clamp/defer resolution of the
//!   periodic-capture window from the scenario anchor.
//! * [`periodic_accessor_current`] — whether the periodic-capture BPF
//!   accessor is still current for the live scheduler.
//!
//! `FREEZE_RENDEZVOUS_TIMEOUT`, `BspExitReason`, `SnapshotRequest`, and
//! `FreezeState` were extracted (unchanged) from `freeze_coord/mod.rs`
//! (or, for `FreezeState`, the run-loop closure body); `PeriodicWindow`
//! and `resolve_periodic_window` were extracted from the run-loop's
//! inline boundary-build logic so the float/clamp/defer wiring is
//! unit-testable.

use std::time::Duration;

/// Maximum wall-clock duration the freeze coordinator will wait for
/// every vCPU to acknowledge parked state before logging a timeout
/// and giving up on the dump. Well above the worst-case drain-dance
/// and single-iteration park latency on healthy guests; a real
/// timeout indicates a vCPU stuck in KVM_RUN that the
/// `immediate_exit` kick failed to interrupt.
pub(super) const FREEZE_RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(30);

/// Why [`super::KtstrVm::run_bsp_loop`] exited. Logged at break time
/// so an operator reading stderr (`BSP: loop exit reason=...`) can
/// diagnose a `code=-1` exit without correlating to peer-vCPU
/// stderr or `tracing` output.
///
/// Mapping to the BSP loop's exit_code:
///   - [`Shutdown`](Self::Shutdown) → exit_code = 0 (the only path
///     that overwrites the local `-1` sentinel).
///   - Every other variant → exit_code = -1, but
///     [`super::super::KtstrVm::collect_results`] re-derives the
///     final [`super::super::result::VmResult::exit_code`] from the
///     bulk-port `MSG_TYPE_EXIT` payload (or COM2 `KTSTR_EXIT:`
///     sentinel) when either is present, so a `-1` from the BSP
///     run-loop is not authoritative for caller-visible test
///     outcome.
#[derive(Debug, Clone, Copy)]
pub(super) enum BspExitReason {
    /// `kill.load(Acquire)` returned `true` at the top of the loop —
    /// some peer (an AP that observed [`super::exit_dispatch::ExitAction::Shutdown`] or
    /// [`super::exit_dispatch::ExitAction::Fatal`], the panic hook, the monitor thread on
    /// `MSG_TYPE_SCHED_EXIT`, or `collect_results`) flipped the flag.
    /// In particular, on a clean test exit where the kernel's i8042
    /// reset OUT is dispatched to a non-BSP vCPU, the AP path sets
    /// `kill` and the BSP exits via this branch. The default value
    /// for the local — every break path that does not explicitly
    /// reassign falls into this case.
    ExternalKill,
    /// BSP itself observed [`super::exit_dispatch::ExitAction::Shutdown`] from
    /// `classify_exit` (i8042 reset on x86_64, PSCI SystemEvent /
    /// `VcpuExit::Shutdown` on aarch64). The only path that sets
    /// exit_code to 0.
    Shutdown,
    /// BSP itself observed [`super::exit_dispatch::ExitAction::Fatal`] from `classify_exit`
    /// (`VcpuExit::FailEntry` or `VcpuExit::InternalError`). Kill
    /// flag is propagated to peers before break.
    Fatal,
    /// `bsp.run()` returned a non-EINTR/EAGAIN errno. Indicates a
    /// permanent KVM_RUN failure on the BSP vCPU fd.
    RunError,
    /// A guest kernel panic (e.g. OOM → "Kernel panic - not syncing:
    /// Attempted to kill init! exitcode=0x…") was latched on COM1 (the
    /// kernel console) at ingest and observed by the run loop, which breaks
    /// immediately rather than spinning to the watchdog or the 24h
    /// interactive-shell timeout (the gap that turned guest OOM into a
    /// cryptic "test timed out"). Falls under the "every other variant →
    /// exit_code=-1" rule above (no clean MSG_TYPE_EXIT follows a panic);
    /// the latched panic line is logged at break time so the operator sees
    /// the cause.
    GuestPanic,
}

/// Decoded contents of a guest-side `MSG_TYPE_SNAPSHOT_REQUEST` TLV
/// frame consumed from the virtio-console port-1 TX stream by the
/// coordinator's TOKEN_TX handler. The request id is echoed in the
/// matching `MSG_TYPE_SNAPSHOT_REPLY` payload so the guest's blocking
/// reader can pair the reply against its outstanding request; `kind`
/// selects the CAPTURE / WATCH dispatch path and `tag` carries the
/// snapshot name (CAPTURE) or symbol path (WATCH).
pub(super) struct SnapshotRequest {
    pub(super) request_id: u32,
    pub(super) kind: u32,
    pub(super) tag: String,
}

/// Dual-snapshot state machine the freeze coordinator's run-loop
/// advances on each capture cycle. Only the `TookEarly` variant is
/// reachable when `freeze_coord_dual_snapshot` is true; the single-
/// snapshot path drives the same transitions but skips the early
/// branch entirely.
///
/// * [`Idle`](Self::Idle) — no dump captured yet.
/// * [`TookEarly`](Self::TookEarly) — early snapshot captured
///   (dual-snapshot mode only); waiting for the err_exit latch to
///   fire.
/// * [`Done`](Self::Done) — no further late-trigger captures
///   attempted. Reach paths: (1) late Captured → full single or dual
///   SCHEMA JSON on the main dump path; (2) late Degraded → degraded
///   SCHEMA JSON on the main dump path (and, if dual-mode held a
///   Captured early, the early reaches disk at the
///   `early-pre-late-degraded` tagged sibling per
///   [`crate::monitor::dump::SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED`]);
///   (3) late Suppressed → no main-path emit (clean exit, no late
///   failure to dump), but if dual-mode held a Captured early, that
///   early reaches disk at the `early-only-late-suppressed` tagged
///   sibling per
///   [`crate::monitor::dump::SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED`].
///   Every captured snapshot reaches disk regardless of how the late
///   path resolves. Coord idles until kill / bsp_done. The shared
///   terminal semantic is "stop probing the trigger" — what differs
///   across reach paths is which file(s) the operator finds in the
///   dump directory.
///
///   Reach path (4) does NOT pass through `Done` via a late-trigger
///   arm at all: the late trigger never fires (no `err_exit_detected`
///   BPF latch flip for the run), `freeze_state` stays at `Idle` or
///   `TookEarly`, and the coordinator's normal-exit cleanup at the
///   `'coord:` loop tail drains a still-held `early_snapshot` to
///   `early-only-late-never-fired` (per
///   [`crate::monitor::dump::SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED`]).
///   This case is distinct from path (3) — both share the "no main
///   dump, early at tagged sibling" shape, but signal differs: (3)
///   means the late trigger fired and the gate decided clean; (4)
///   means the late trigger never reached terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FreezeState {
    Idle,
    TookEarly,
    Done,
}

/// Format the bridge tag for periodic boundary `idx`. The wire shape
/// — zero-padded 3-digit `"periodic_NNN"` — is documented on
/// [`crate::test_support::KtstrTestEntry::num_snapshots`] and
/// pinned by the unit tests in this module. Pulled out of the inline
/// `format!` site so the format string lives at one location and a
/// future width change does not silently shift between fire path and
/// downstream string-matching tests.
pub(super) fn periodic_tag(idx: u32) -> String {
    format!("periodic_{:03}", idx)
}

/// Compute the absolute boundary timestamps (nanoseconds since
/// `run_start`) at which the freeze coordinator's periodic-capture
/// loop fires
/// `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })`.
/// Boundaries are uniformly
/// sliced across the capturable window `[window_start_ns,
/// window_end_ns]`: a 10 % pre-buffer at the start (ramp-up) and a
/// 10 % post-buffer at the end (ramp-down) keep periodic samples off
/// transient state; the remaining 80 % is divided into
/// `num_snapshots + 1` equal intervals, yielding `num_snapshots`
/// interior boundary points.
///
/// The caller supplies an explicit `[start, end]` span rather than
/// `anchor + duration` so it can CLAMP `window_end_ns` to the workload
/// end (`scenario_anchor + workload_duration`, the anchor being the
/// ScenarioStart stamp or the watchdog_reset-derived attach moment)
/// while letting
/// `window_start_ns` float to the prereq-ready moment on a cold boot.
/// That clamp keeps every boundary inside the workload: a cold-boot
/// `window_start` would otherwise (under the old anchor+duration form)
/// push the window tail past the workload end and sample post-workload
/// idle, diluting window-averaged periodic metrics
/// (`avg_cpu_util_comp_scale`, `avg_task_lat_cri`) toward their idle
/// floor. `avg_cpu_util_comp_scale` is the verdict-affecting one (a
/// LowerBetter first-to-last cpustat-delta RATIO), so under the old form
/// its cold-vs-warm dilution was a systematic single-run perf-delta
/// skew (baseline cold, HEAD warm); `avg_task_lat_cri` is Informational
/// (verdict-inert — its compare-path delta never counts toward the
/// regression verdict), so clamping it is a sampling-fidelity gain, not a
/// verdict fix. The clamp fully removes the idle-spill for both. A
/// second-order residual remains: for `avg_cpu_util_comp_scale` (the
/// ratio) it is nonzero only under a time-varying-IRQ workload; for
/// `avg_task_lat_cri` (a mean over (freeze, task)) it is a shorter/later
/// window sampling a different lat_cri sub-distribution. Both are why
/// cross-run compares of these keys use `--noise-adjust` (see the
/// `fold_util_comp_scale` caveat in `assert::phase_build`). Equivalence
/// to the prior anchor-relative landings holds whenever `window_start ==
/// scenario_anchor` (the common warm-boot ordering, where prereqs are
/// ready by scenario-start) — not a hard guarantee: if prereqs land
/// after the scenario-start stamp, `window_start` floats later and the
/// span is `[max(anchor, prereqs_ready), anchor + d]`.
///
/// Compute in `u128` to guard a hostile or degenerate `window_end` near
/// `u64::MAX`: realistic spans are tiny by comparison (a 10-minute span
/// is ~6e11 ns, whose `window · (i + 1)` is ~3e13, far below `u64::MAX`),
/// but the function takes raw `u64` args and `window · (i + 1)` overflows
/// `u64` only once `window` exceeds ~2.8e17 ns. The final boundary fits
/// in `u64` (saturating cast) for callers that read against the
/// run-loop's `u64` clock.
///
/// # Edge cases
///
/// * `num_snapshots == 0` → empty `Vec` (the disabled case; the
///   run-loop's check skips the branch when the list is empty / unset).
/// * `window_end_ns <= window_start_ns` → empty `Vec`: a degenerate /
///   inverted window (e.g. a cold boot whose prereq-ready moment lands
///   at or after the workload end) has no capturable span, so no
///   boundaries fire. (The old anchor-relative form would instead place
///   all boundaries past the workload end, capturing idle.)
/// * inner `window < num_snapshots + 1` → empty `Vec`: below this
///   SUFFICIENT-monotonicity threshold the per-step lower bound
///   `floor(window/(N+1))` drops to 0, so consecutive offsets are no
///   longer GUARANTEED ≥ 1 ns apart under the truncating divide
///   (`window == N` is in fact still strict but is conservatively
///   rejected; `window < N` genuinely pigeonholes into duplicates). The
///   guard emits none rather than risk duplicate-timestamp boundaries —
///   a positive but pathologically tiny window (a cold boot whose
///   prereq-ready lands within ~`(N+1)/0.8` ns of the clamped end).
///   Unreachable for any realistic seconds-scale window.
/// * `num_snapshots == N` for `N ≥ 1` → `N` boundaries at
///   `window_start + pre + (i + 1) · window / (N + 1)` for `i ∈ 0..N`,
///   where `pre = avail / 10` and `window = avail − 2 · pre`, all
///   truncating integer divides (so `pre` / `window` are the integer
///   `avail/10` and `avail − 2·(avail/10)`, not exact `0.1·` / `0.8·`
///   fractions — see the `odd_ns_rounding` test); `avail = window_end -
///   window_start`. `N == 1` lands at the window midpoint.
///
/// Non-empty results are strictly monotonically increasing (the
/// run-loop's linear scan via `next_periodic_idx` relies on it): the
/// `window < num_snapshots + 1` empty-return enforces the sufficient
/// `window >= num_snapshots + 1` condition that guarantees it.
///
/// The per-boundary interval is validated against a 100 ms floor at
/// config time using the FULL duration (the warm-boot span); on a cold
/// boot the clamped `avail` is smaller, so the actual interval can be
/// tighter than that floor — a sampling-quality, not correctness,
/// limitation (`avail` is not known at validation time).
pub(super) fn compute_periodic_boundaries_ns(
    window_start_ns: u64,
    window_end_ns: u64,
    num_snapshots: u32,
) -> Vec<u64> {
    if num_snapshots == 0 || window_end_ns <= window_start_ns {
        return Vec::new();
    }
    let n = num_snapshots as u128;
    let avail = (window_end_ns - window_start_ns) as u128;
    let pre_buffer = avail / 10;
    let window = avail.saturating_sub(2u128.saturating_mul(pre_buffer));
    // window >= n+1 is a SUFFICIENT condition for strict monotonicity:
    // under the truncating divide consecutive offsets differ by
    // >= floor(window/(n+1)), which is >= 1 exactly when window >= n+1.
    // Below that the per-step lower bound drops to 0, so consecutive
    // offsets are no longer guaranteed distinct (window == n is still
    // strict but conservatively rejected; window < n genuinely
    // pigeonholes into EQUAL values). Return empty rather than risk
    // duplicate-timestamp boundaries, upholding the strict-monotonic
    // contract the run-loop's linear scan relies on. Unreachable for any
    // realistic seconds-scale window.
    if window < n + 1 {
        return Vec::new();
    }
    let mut boundaries: Vec<u64> = Vec::with_capacity(num_snapshots as usize);
    for i in 0..n {
        let offset = pre_buffer.saturating_add(window.saturating_mul(i + 1) / (n + 1));
        let absolute = (window_start_ns as u128).saturating_add(offset);
        boundaries.push(u64::try_from(absolute).unwrap_or(u64::MAX));
    }
    boundaries
}

/// The resolved periodic-capture window: the slice bounds plus the
/// scenario-relative offset anchor. Produced by
/// `resolve_periodic_window` and consumed by
/// `compute_periodic_boundaries_ns` (the bounds) and the coordinator
/// run-loop's boundary bucketing (`anchor_ns`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PeriodicWindow {
    /// Slice start — the LATER of the scenario anchor and the
    /// prereq-ready moment (floats later on a cold boot).
    pub(super) window_start_ns: u64,
    /// Slice end — `scenario_anchor + workload_duration` (saturating),
    /// CLAMPED to the workload end.
    pub(super) window_end_ns: u64,
    /// Scenario-relative offset frame for boundary bucketing — always
    /// the scenario anchor, never `window_start_ns`.
    pub(super) anchor_ns: u64,
}

/// Resolve the periodic-capture window from the resolved scenario
/// anchor, the prereq-ready moment, and the workload duration — or
/// `None` to DEFER.
///
/// Returns `None` when `scenario_anchor == 0` (no anchor resolved yet:
/// neither the guest's `MSG_TYPE_SCENARIO_START` stamp nor the
/// `watchdog_reset` fallback has latched). A 0 anchor would set
/// `window_end = 0 + workload_duration`, INVERTING the clamped window
/// (`window_end < window_start` whenever `prereqs_ready >
/// workload_duration`, the common multi-second-boot case) so the slicer
/// 0-fires — re-introducing the cold-boot 0-sample flake — and it would
/// stamp `anchor_ns = 0`, mis-bucketing every boundary offset. The
/// caller retries once an anchor latches; the prereq-ready stamp is
/// kept across retries so no rework is lost.
///
/// Otherwise:
/// - `window_start_ns = max(scenario_anchor, periodic_prereqs_ready_ns)`
///   — floats to the prereq-ready moment on a cold boot so no boundary
///   lands in the already-elapsed pre-ready span. On a warm boot
///   (`prereqs_ready <= anchor`) it equals the anchor and the span is
///   the full `[anchor, anchor + workload_duration]`.
/// - `window_end_ns = scenario_anchor + workload_duration_ns`
///   (saturating) — CLAMPED to the workload end so a cold-boot-late
///   start cannot push captures into post-workload idle. The clamp is
///   anchored at `scenario_anchor`, NOT `window_start`, so a late
///   prereq-ready shrinks the span from the start side only.
/// - `anchor_ns = scenario_anchor` — the scenario-relative offset
///   frame (`boundary - anchor_ns` is scenario-relative); a
///   window-relative anchor would mis-attribute cold-boot captures by
///   the prereq lag.
pub(super) fn resolve_periodic_window(
    scenario_anchor: u64,
    periodic_prereqs_ready_ns: u64,
    workload_duration_ns: u64,
) -> Option<PeriodicWindow> {
    if scenario_anchor == 0 {
        return None;
    }
    Some(PeriodicWindow {
        window_start_ns: scenario_anchor.max(periodic_prereqs_ready_ns),
        window_end_ns: scenario_anchor.saturating_add(workload_duration_ns),
        anchor_ns: scenario_anchor,
    })
}

/// True when the periodic-capture accessor is still current for the live
/// scheduler. `last_sched_kva` is the `scx_root` value the scx_root watchpoint
/// poll last republished the owned BPF accessor against (0 = detached);
/// `live_sched_kva` is a fresh `*scx_root` read taken AT the periodic boundary.
/// They match only when no scheduler swap has occurred since the last republish.
///
/// When false, the periodic boundary MUST defer: `owned_accessor` is bound to
/// the prior scheduler's BPF object, so a capture reads the PRIOR scheduler's
/// `.bss` (its counters), not the live one's; and once the prior BPF object's
/// maps are released (map refcount -> 0 on scheduler-process exit / fd close —
/// NOT synchronously by `scx_root_disable`, which only unlinks the `scx_sched`
/// and NULLs `*scx_root`), that `.bss` page is recycled/zeroed, surfacing a
/// silent `nr_dispatched=0`. The watchpoint
/// poll (≤ `SCAN_INTERVAL`) re-resolves the accessor for the live scheduler and
/// the next boundary passes. A zero `live_sched_kva` (detached) also defers —
/// there is no scheduler to capture. Pure helper so the
/// `(last_sched_kva, live_sched_kva)` transition matrix is unit-testable without
/// booting a VM, mirroring `republish_watchpoint_on_rebind`'s pure-helper shape.
pub(super) fn periodic_accessor_current(last_sched_kva: u64, live_sched_kva: u64) -> bool {
    live_sched_kva != 0 && live_sched_kva == last_sched_kva
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No scheduler swap since the last watchpoint republish → the accessor is
    /// current, the periodic boundary captures.
    #[test]
    fn periodic_accessor_current_when_root_unchanged() {
        assert!(periodic_accessor_current(
            0xffff_8000_1234_5678,
            0xffff_8000_1234_5678
        ));
    }

    /// scx_root moved (A→B) before the ≤SCAN_INTERVAL watchpoint poll re-resolved
    /// the accessor → stale → defer (the Window-B race).
    #[test]
    fn periodic_accessor_stale_after_swap() {
        assert!(!periodic_accessor_current(
            0xffff_8000_1111_1111,
            0xffff_8000_2222_2222
        ));
    }

    /// A zero live scx_root (scheduler detached) defers regardless of the last
    /// republished value — there is no scheduler to capture; and a fresh attach
    /// (last still 0, live = B) also defers until the accessor re-resolves.
    #[test]
    fn periodic_accessor_detached_or_unresolved_defers() {
        assert!(!periodic_accessor_current(0xffff_8000_1111_1111, 0));
        assert!(!periodic_accessor_current(0, 0));
        assert!(!periodic_accessor_current(0, 0xffff_8000_2222_2222));
    }

    /// Dispatched into the periodic-capture loop's tag-formatting
    /// step. The tag wire format is documented in
    /// [`crate::test_support::entry::KtstrTestEntry::num_snapshots`]:
    /// each interior boundary is stored on the bridge under
    /// `"periodic_NNN"` (zero-padded 3-digit index). Tests assert
    /// the exact string shape so a regression to a different padding
    /// width or separator surfaces here, not in downstream test
    /// authors' string-matching code.
    #[test]
    fn periodic_tag_name_format_low_index() {
        // Index 0 — leftmost slot.
        assert_eq!(periodic_tag(0), "periodic_000");
        // Single-digit, single-digit — pad to 3 chars.
        assert_eq!(periodic_tag(1), "periodic_001");
        assert_eq!(periodic_tag(7), "periodic_007");
    }

    /// Two- and three-digit indices must keep the zero-padding
    /// width at exactly 3 characters. A naive `format!("periodic_{i}")`
    /// without `:03` would surface here as `"periodic_10"` /
    /// `"periodic_64"` — failing the comparison.
    #[test]
    fn periodic_tag_name_format_high_index() {
        assert_eq!(periodic_tag(10), "periodic_010");
        assert_eq!(periodic_tag(63), "periodic_063");
        // MAX_STORED_SNAPSHOTS == 64; the cap rejects N > 64 but
        // index 64 itself is unreachable in practice (boundaries
        // are 0 .. N-1). Pin the format for index 64 anyway so a
        // future cap raise that produced index 64 doesn't surface
        // as a tag-format regression.
        assert_eq!(periodic_tag(64), "periodic_064");
    }

    /// `compute_periodic_boundaries_ns` divides the span
    /// `window_end - window_start` into `num_snapshots + 1` equal
    /// intervals after reserving a 10 % pre-buffer and 10 %
    /// post-buffer. For `N == 1` the lone interior boundary lands at
    /// the span midpoint — `window_start + 0.5 · (window_end -
    /// window_start)`. Pinning the midpoint case proves the formula's
    /// symmetry: a shift of the pre/post buffer balance would surface
    /// here as a non-mid landing point.
    #[test]
    fn compute_periodic_boundaries_ns_n1_lands_at_midpoint() {
        // 10 s window [0, 10 s] → midpoint = 5 s = 5 · 1e9 ns.
        let boundaries = compute_periodic_boundaries_ns(0, 10_000_000_000, 1);
        assert_eq!(
            boundaries.len(),
            1,
            "N=1 must produce exactly one interior boundary",
        );
        assert_eq!(
            boundaries[0], 5_000_000_000,
            "N=1 lands at window_start + 0.5·(window_end - window_start) = 5 s",
        );
    }

    /// `N == 3` lands captures at 0.3, 0.5, 0.7 of the span
    /// `window_end - window_start` when the 10 % pre-buffer + 80 %
    /// usable span + 10 % post-buffer formula is honoured (here the
    /// window is `[0, 10 s]`, so the span equals the workload
    /// duration `d` and the landings are `0.3·d / 0.5·d / 0.7·d`).
    /// The doc on
    /// [`crate::test_support::entry::KtstrTestEntry::num_snapshots`]
    /// pins these exact landing points; regressing the formula
    /// (e.g. dropping the buffer, splitting at fence-post boundaries
    /// instead of equal-interval interior points) would surface
    /// here.
    #[test]
    fn compute_periodic_boundaries_ns_n3_quartile_landings() {
        // 10 s window [0, 10 s]:
        //   pre-buffer = 1 s, window = 8 s, splits = 4 intervals
        //   boundary[0] = 1 s + 1/4 · 8 s = 3 s
        //   boundary[1] = 1 s + 2/4 · 8 s = 5 s
        //   boundary[2] = 1 s + 3/4 · 8 s = 7 s
        let boundaries = compute_periodic_boundaries_ns(0, 10_000_000_000, 3);
        assert_eq!(boundaries.len(), 3, "N=3 must produce 3 boundaries");
        assert_eq!(boundaries[0], 3_000_000_000, "N=3 boundary[0] at 0.3·d");
        assert_eq!(boundaries[1], 5_000_000_000, "N=3 boundary[1] at 0.5·d");
        assert_eq!(boundaries[2], 7_000_000_000, "N=3 boundary[2] at 0.7·d");
    }

    /// Boundaries must be biased by `window_start_ns` — a non-zero
    /// window start must shift every boundary by the same offset. Pins
    /// the absolute-vs-relative contract: the formula is
    /// `window_start + pre_buffer + (i+1) · window / (N+1)`, so a
    /// regression that dropped the window_start add (returning relative
    /// offsets) would surface here as `boundaries[0] != window_start + 5e9`.
    #[test]
    fn compute_periodic_boundaries_ns_respects_window_start_offset() {
        // Window [2 s, 12 s] (start 2 s, end = start + a 10 s span),
        // N=1 → boundary at the window midpoint = 7 s.
        let window_start_ns: u64 = 2_000_000_000;
        let window_end_ns: u64 = 12_000_000_000;
        let boundaries = compute_periodic_boundaries_ns(window_start_ns, window_end_ns, 1);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(
            boundaries[0], 7_000_000_000,
            "window_start(2 s) + midpoint(5 s of the 10 s span) = 7 s",
        );
    }

    /// `N == 0` is the disabled-periodic-capture case. The function
    /// must return an empty `Vec` — NOT a panic, NOT a single
    /// boundary at the midpoint. The freeze coordinator's
    /// per-iteration check uses the boundary list as the gate; an
    /// empty vec means no periodic samples fire.
    #[test]
    fn compute_periodic_boundaries_ns_n0_yields_empty() {
        let boundaries = compute_periodic_boundaries_ns(0, 10_000_000_000, 0);
        assert!(
            boundaries.is_empty(),
            "N=0 must yield no boundaries, got {boundaries:?}",
        );
    }

    /// Boundaries must be strictly monotonically increasing.
    /// Dispatch loop relies on this invariant: the run-loop
    /// scans boundaries linearly via `next_periodic_idx` and
    /// expects `boundaries[i] < boundaries[i+1]`. A regression
    /// that produced duplicate or descending boundaries would
    /// silently fire the same idx twice or stall.
    #[test]
    fn compute_periodic_boundaries_ns_strictly_monotonic() {
        for &n in &[1u32, 2, 3, 7, 16, 32, 64] {
            let boundaries = compute_periodic_boundaries_ns(0, 60_000_000_000, n);
            assert_eq!(
                boundaries.len(),
                n as usize,
                "N={n} must produce exactly {n} boundaries",
            );
            for w in boundaries.windows(2) {
                assert!(
                    w[0] < w[1],
                    "boundaries must be strictly monotonic; got {w:?} for N={n}",
                );
            }
        }
    }

    /// Every boundary must land inside the 10 %–90 % sub-range of the
    /// window `[window_start, window_end]`. The pre-buffer and
    /// post-buffer give the workload ramp-up / ramp-down room
    /// without periodic samples landing on transient state. Pins
    /// the buffer invariant against a regression that removed or
    /// shifted the buffers (which would surface here as a
    /// boundary at < 0.1·avail or > 0.9·avail).
    #[test]
    fn compute_periodic_boundaries_ns_within_buffer_window() {
        // Window [0, 10 s]; avail == the full 10 s span.
        let total_ns: u64 = 10_000_000_000;
        // 10 % pre = 1 s; 90 % post = 9 s. Use inclusive bounds —
        // boundaries land STRICTLY inside the window, never AT
        // the buffer edge (the formula is `+ (i+1) · window /
        // (N+1)` for i ∈ 0..N, so the lowest landing is at
        // `pre + 1·window/(N+1) > pre`).
        for &n in &[1u32, 4, 8] {
            let boundaries = compute_periodic_boundaries_ns(0, total_ns, n);
            for (i, &b) in boundaries.iter().enumerate() {
                assert!(
                    b > total_ns / 10,
                    "boundary {i} ({b}) must be strictly above 10% pre-buffer ({})",
                    total_ns / 10,
                );
                assert!(
                    b < total_ns - total_ns / 10,
                    "boundary {i} ({b}) must be strictly below 90% post-buffer ({})",
                    total_ns - total_ns / 10,
                );
            }
        }
    }

    /// Pin the rounding behaviour for non-round-number durations.
    /// `compute_periodic_boundaries_ns` performs all arithmetic in
    /// `u128` and rounds toward zero on the integer division
    /// `window * (i + 1) / (N + 1)`. A future arithmetic refactor
    /// (rounding to nearest, switching to f64, swapping operand
    /// order) would silently shift periodic-sample landings.
    /// These exact-byte assertions surface that drift here, before
    /// it ships.
    ///
    /// The expected values are computed by hand from the formula
    ///   boundary[i] = window_start + (avail / 10)
    ///                 + (avail - 2 · (avail / 10)) · (i + 1) / (N + 1)
    /// where avail = window_end - window_start; all operations in u128
    /// truncating-divide. Each case is
    /// derived in the test body comment so a future reader can
    /// re-verify without re-running the math.
    #[test]
    fn compute_periodic_boundaries_ns_odd_ns_rounding() {
        // Case 1: 1_000_000_001 ns, N = 3.
        //   pre = 1_000_000_001 / 10 = 100_000_000
        //   window = 1_000_000_001 − 200_000_000 = 800_000_001
        //   i=0: pre + 800_000_001 · 1 / 4 = 100_000_000 + 200_000_000 = 300_000_000
        //   i=1: pre + 800_000_001 · 2 / 4 = 100_000_000 + 400_000_000 = 500_000_000
        //   i=2: pre + 800_000_001 · 3 / 4 = 100_000_000 + 600_000_000 = 700_000_000
        //
        // (`800_000_001 · k / 4` truncates the `+ 1` for k = 1..3
        // because 4 · 200_000_000 = 800_000_000 < 800_000_001.)
        let boundaries = compute_periodic_boundaries_ns(0, 1_000_000_001, 3);
        assert_eq!(
            boundaries,
            vec![300_000_000, 500_000_000, 700_000_000],
            "1_000_000_001 ns, N=3 must round each boundary down to \
             the multiple-of-100_000_000 nearest the truncating-divide \
             result"
        );

        // Case 2: 1_000_000_007 ns, N = 2.
        //   pre = 100_000_000  (since 1_000_000_007 / 10 = 100_000_000)
        //   window = 1_000_000_007 − 200_000_000 = 800_000_007
        //   i=0: 100_000_000 + 800_000_007 · 1 / 3 = 100_000_000 + 266_666_669 = 366_666_669
        //   i=1: 100_000_000 + 800_000_007 · 2 / 3 = 100_000_000 + 533_333_338 = 633_333_338
        //
        // 3 · 266_666_669 = 800_000_007 exactly, so i=0 lands on the
        // exact divide; 3 · 533_333_338 = 1_600_000_014 exactly so
        // i=1 also exact. Pin the values so a future change
        // (e.g. operand-order swap that produces the same exact
        // result vs a `f64` round-to-nearest path that would shift
        // by 1 ns) surfaces here.
        let boundaries2 = compute_periodic_boundaries_ns(0, 1_000_000_007, 2);
        assert_eq!(boundaries2, vec![366_666_669, 633_333_338]);

        // Case 3: 1_000_000_007 ns, N = 4 — exercises the same
        // odd-ns total against a denser boundary grid.
        //   window = 800_000_007, n+1 = 5
        //   i=0: 100_000_000 + 800_000_007 / 5 = 100_000_000 + 160_000_001 = 260_000_001
        //         (5 · 160_000_001 = 800_000_005, remainder 2)
        //   i=1: 100_000_000 + 1_600_000_014 / 5 = 100_000_000 + 320_000_002 = 420_000_002
        //         (5 · 320_000_002 = 1_600_000_010, remainder 4)
        //   i=2: 100_000_000 + 2_400_000_021 / 5 = 100_000_000 + 480_000_004 = 580_000_004
        //         (5 · 480_000_004 = 2_400_000_020, remainder 1)
        //   i=3: 100_000_000 + 3_200_000_028 / 5 = 100_000_000 + 640_000_005 = 740_000_005
        //         (5 · 640_000_005 = 3_200_000_025, remainder 3)
        let boundaries3 = compute_periodic_boundaries_ns(0, 1_000_000_007, 4);
        assert_eq!(
            boundaries3,
            vec![260_000_001, 420_000_002, 580_000_004, 740_000_005]
        );

        // Case 4: window_start offset survives the odd-ns rounding —
        // window_start + computed offset, no extra rounding step.
        //   window_start = 12_345, window_end = 12_345 + 1_000_000_001
        //   (Case 1's span), so avail = 1_000_000_001; each boundary
        //   should be Case 1's boundary + 12_345.
        let boundaries4 = compute_periodic_boundaries_ns(12_345, 1_000_012_346, 3);
        assert_eq!(
            boundaries4,
            vec![300_012_345, 500_012_345, 700_012_345],
            "window_start offset must be added to the truncated boundary, \
             not folded into the truncating-divide"
        );
    }

    /// A degenerate / inverted window — `window_end_ns <=
    /// window_start_ns` — yields no boundaries. This is the cold-boot
    /// case where the prereq-ready moment (window_start) lands at or
    /// after the workload end (window_end): there is no capturable
    /// in-workload span, so no periodic samples fire. The prior
    /// anchor+duration form would instead place every boundary past
    /// the workload end, sampling post-workload idle.
    #[test]
    fn compute_periodic_boundaries_ns_degenerate_window_is_empty() {
        // Zero span (window_start == window_end).
        assert!(
            compute_periodic_boundaries_ns(5_000_000_000, 5_000_000_000, 3).is_empty(),
            "a zero-span window must yield no boundaries",
        );
        // Inverted span (prereq-ready landed past the workload end).
        assert!(
            compute_periodic_boundaries_ns(8_000_000_000, 5_000_000_000, 3).is_empty(),
            "an inverted window must yield no boundaries",
        );
    }

    /// Cold-boot clamp regression pin: with a LATE window_start (the
    /// prereq lag) and window_end clamped to the workload end, every
    /// boundary stays inside `[window_start, window_end]` — none spill
    /// past window_end into post-workload idle. Reproduces the
    /// cold-boot dilution bug: the old anchor+duration form sliced
    /// `[start, start + d]`, so a `window_start = anchor + Δ` pushed the
    /// tail to `anchor + Δ + 0.9 d > anchor + d` (the workload end) for
    /// Δ > 0.1 d, sampling idle and diluting window-averaged metrics.
    #[test]
    fn compute_periodic_boundaries_ns_cold_boot_stays_within_window() {
        // Workload [0, 10 s]; cold-boot prereq-ready at 3 s (Δ = 0.3 d,
        // well past the 0.1 d the old tail would have overrun).
        let window_start_ns: u64 = 3_000_000_000;
        let window_end_ns: u64 = 10_000_000_000;
        for &n in &[1u32, 3, 6, 12] {
            let boundaries = compute_periodic_boundaries_ns(window_start_ns, window_end_ns, n);
            assert_eq!(boundaries.len(), n as usize, "N={n} boundary count");
            for (i, &b) in boundaries.iter().enumerate() {
                assert!(
                    b > window_start_ns,
                    "boundary {i} ({b}) must be after window_start ({window_start_ns})",
                );
                assert!(
                    b < window_end_ns,
                    "boundary {i} ({b}) must be before the clamped window_end \
                     ({window_end_ns}) — never spilling into post-workload idle",
                );
            }
        }
    }

    /// A positive but pathologically tiny window — inner span smaller
    /// than num_snapshots+1 — yields no boundaries (rather than the
    /// duplicate-timestamp boundaries that would break the strict-
    /// monotonic contract the run-loop's linear scan relies on). This is
    /// the cold-boot case where the prereq-ready moment lands within
    /// ~(N+1)/0.8 ns of the clamped workload end. Pins the
    /// `window < num_snapshots + 1` empty-return.
    #[test]
    fn compute_periodic_boundaries_ns_tiny_window_below_min_step_is_empty() {
        // avail=70 -> pre=7, inner window=56 < N+1=65 (the pigeonhole
        // example): empty rather than duplicate-timestamp boundaries.
        assert!(
            compute_periodic_boundaries_ns(0, 70, 64).is_empty(),
            "inner window 56 < 65 must yield no boundaries",
        );
        // avail=50, N=64: pre=5, window=40 < 65 -> empty.
        assert!(compute_periodic_boundaries_ns(0, 50, 64).is_empty());
        // The guard flips at window == N+1; pin BOTH sides at N=64.
        // avail=78 -> pre=7, window=64 == N -> conservatively rejected
        // (empty) even though window==N is itself still strictly monotonic
        // — the guard is a sufficient bound, not the tight one.
        assert!(
            compute_periodic_boundaries_ns(0, 78, 64).is_empty(),
            "window == N (64) is conservatively rejected by the >= N+1 guard",
        );
        // avail=79 -> pre=7, window=65 == N+1 -> the SMALLEST non-empty
        // window; every step is exactly floor(65/65)==1, so the 64
        // boundaries are strictly monotonic at zero slack ([8..=71]).
        let edge = compute_periodic_boundaries_ns(0, 79, 64);
        assert_eq!(
            edge.len(),
            64,
            "window == N+1 (65) is the smallest non-empty"
        );
        assert_eq!(edge[0], 8, "first boundary = pre(7) + 1");
        assert_eq!(edge[63], 71, "last boundary = pre(7) + 64");
        for w in edge.windows(2) {
            assert!(
                w[0] < w[1],
                "window == N+1 must stay strict at zero slack: {edge:?}",
            );
        }
        // Small-N readability: avail=4 -> pre=0, window=4 == N+1 -> unit
        // steps [1, 2, 3]; avail=3 -> window=3 == N -> rejected (empty).
        assert_eq!(
            compute_periodic_boundaries_ns(0, 4, 3),
            vec![1, 2, 3],
            "window == N+1 at N=3 lands [1,2,3] with unit steps",
        );
        assert!(compute_periodic_boundaries_ns(0, 3, 3).is_empty());
    }

    /// Overflow-safety pin for a hostile / degenerate window near
    /// `u64::MAX`. `compute_periodic_boundaries_ns` takes raw `u64`
    /// window bounds (attacker-influenced on a misbehaving guest) and
    /// computes `window · (i + 1)` in `u128`: for a near-`u64::MAX`
    /// span that product reaches ~9.4e20, which OVERFLOWS `u64` (max
    /// ~1.8e19) but fits `u128`. A future refactor narrowing the
    /// arithmetic to `u64` (or dropping the saturating ops) would
    /// panic in debug / wrap in release with NO test failing — the
    /// silent-wrong-answer class the doc's "compute in u128" rationale
    /// guards. Pins len, strict monotonicity, and the in-window bound.
    #[test]
    fn compute_periodic_boundaries_ns_near_u64_max_no_overflow() {
        // Widest possible span [0, u64::MAX], N=64. avail = u64::MAX,
        // inner window ≈ 1.47e19, so window·64 ≈ 9.4e20 overflows u64 —
        // the u128 path is what makes this correct.
        let wide = compute_periodic_boundaries_ns(0, u64::MAX, 64);
        assert_eq!(
            wide.len(),
            64,
            "max-span window must produce all 64 boundaries"
        );
        for (i, &b) in wide.iter().enumerate() {
            assert!(
                b < u64::MAX,
                "boundary {i} ({b}) must stay below window_end (u64::MAX) — \
                 the final saturating cast must not clamp",
            );
        }
        for w in wide.windows(2) {
            assert!(
                w[0] < w[1],
                "max-span boundaries must be strictly monotonic; got {w:?}",
            );
        }

        // window_start near the top of the range: [u64::MAX-79, u64::MAX],
        // N=64 → avail=79, pre=7, inner window=65==N+1 (the smallest
        // non-empty, unit steps). Pins the saturating_add(window_start,
        // offset) at the high end: offsets 8..=71 above window_start, all
        // strictly below window_end.
        let start = u64::MAX - 79;
        let high = compute_periodic_boundaries_ns(start, u64::MAX, 64);
        assert_eq!(high.len(), 64, "window==N+1 at high window_start yields 64");
        assert_eq!(
            high[0],
            start + 8,
            "first boundary = window_start + pre(7) + 1"
        );
        assert_eq!(
            high[63],
            start + 71,
            "last boundary = window_start + pre(7) + 64"
        );
        for (i, &b) in high.iter().enumerate() {
            assert!(
                b < u64::MAX && b > start,
                "boundary {i} ({b}) must stay in (window_start, window_end)",
            );
        }
    }

    /// Warm boot (`prereqs_ready <= anchor`): `resolve_periodic_window`
    /// keeps `window_start == anchor`, so the span is the full
    /// `[anchor, anchor + duration]` matching the prior anchor+duration
    /// form. Pins the float's no-op case + the clamp + the anchor frame.
    #[test]
    fn resolve_periodic_window_warm_boot_starts_at_anchor() {
        let w = resolve_periodic_window(5_000, 3_000, 10_000)
            .expect("non-zero anchor must build a window");
        assert_eq!(
            w.window_start_ns, 5_000,
            "warm boot: window_start == anchor"
        );
        assert_eq!(w.window_end_ns, 15_000, "window_end == anchor + duration");
        assert_eq!(w.anchor_ns, 5_000, "anchor_ns == scenario_anchor");
    }

    /// Cold boot (`prereqs_ready > anchor`): `window_start` floats to
    /// the prereq-ready moment, but `window_end` stays clamped at
    /// `anchor + duration` (NOT `prereqs_ready + duration`) and
    /// `anchor_ns` stays the scenario anchor (NOT `window_start`). Pins
    /// the float, the start-side-only clamp, and the offset frame — the
    /// load-bearing novelty of the change. A regression swapping `max`
    /// for `min`, or anchoring the clamp / offset at `window_start`,
    /// surfaces here.
    #[test]
    fn resolve_periodic_window_cold_boot_floats_start_to_prereqs() {
        let w = resolve_periodic_window(5_000, 8_000, 10_000)
            .expect("non-zero anchor must build a window");
        assert_eq!(
            w.window_start_ns, 8_000,
            "cold boot: window_start floats to prereqs_ready",
        );
        assert_eq!(
            w.window_end_ns, 15_000,
            "window_end stays clamped to anchor + duration, not prereqs + duration",
        );
        assert_eq!(
            w.anchor_ns, 5_000,
            "anchor_ns stays scenario_anchor, not the floated window_start",
        );
    }

    /// `anchor == 0` DEFERS (returns `None`): a 0 anchor would set
    /// `window_end = 0 + duration`, inverting the clamped window and
    /// 0-firing the slicer (re-introducing the cold-boot 0-sample
    /// flake). The run-loop leaves the boundary list unbuilt and retries
    /// once an anchor latches.
    #[test]
    fn resolve_periodic_window_zero_anchor_defers() {
        assert!(
            resolve_periodic_window(0, 8_000, 10_000).is_none(),
            "0 anchor must DEFER, not build an inverted window",
        );
        // Defer holds regardless of prereqs / duration.
        assert!(resolve_periodic_window(0, 0, 0).is_none());
    }

    /// `window_end` saturates rather than wrapping on a hostile
    /// `anchor + duration` overflow. Pins the `saturating_add` against a
    /// refactor to a plain `+` (which would wrap to a tiny value, then
    /// be rejected by the slicer's inverted-window guard — a silent
    /// 0-fire instead of a clamp).
    #[test]
    fn resolve_periodic_window_clamps_end_overflow() {
        let w = resolve_periodic_window(u64::MAX - 5, 0, 100)
            .expect("non-zero anchor must build a window");
        assert_eq!(
            w.window_end_ns,
            u64::MAX,
            "anchor + duration must saturate at u64::MAX, not wrap",
        );
        assert_eq!(
            w.window_start_ns,
            u64::MAX - 5,
            "window_start == anchor (prereqs 0 < anchor)",
        );
        assert_eq!(w.anchor_ns, u64::MAX - 5);
    }

    /// Cold-boot-TOO-LATE composition: when prereqs_ready lands at or
    /// after the clamped workload end, resolve_periodic_window returns
    /// an INVERTED window (window_start > window_end), which
    /// compute_periodic_boundaries_ns then empties — no boundaries
    /// fire. Pins the documented end-to-end behavior (resolve produces
    /// an inverted window → the slicer 0-fires); a regression that
    /// min()-clamped window_start down to window_end in the helper
    /// (silently re-enabling a degenerate fire) would pass every other
    /// resolve test but fail here.
    #[test]
    fn resolve_periodic_window_cold_boot_too_late_yields_no_boundaries() {
        // anchor=5_000, prereqs_ready=20_000, duration=10_000 →
        // window_end = 5_000 + 10_000 = 15_000; window_start =
        // max(5_000, 20_000) = 20_000 > 15_000 (inverted).
        let w = resolve_periodic_window(5_000, 20_000, 10_000)
            .expect("non-zero anchor still builds a (possibly inverted) window");
        assert_eq!(
            w.window_start_ns, 20_000,
            "window_start floats to the late prereqs_ready",
        );
        assert_eq!(
            w.window_end_ns, 15_000,
            "window_end clamps to anchor + duration",
        );
        assert!(
            w.window_start_ns > w.window_end_ns,
            "prereqs past the workload end invert the window",
        );
        assert_eq!(w.anchor_ns, 5_000);
        // The slicer empties an inverted window: no periodic samples fire.
        assert!(
            compute_periodic_boundaries_ns(w.window_start_ns, w.window_end_ns, 3).is_empty(),
            "an inverted window must yield no boundaries (slicer 0-fires)",
        );
    }
}
