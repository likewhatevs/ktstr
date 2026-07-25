//! Grant-flow diagnostics (diagnostic-only, pure observation).
//!
//! Discriminates the C≈1.3-concurrent admission wall between the candidate
//! mechanisms by counting, per process:
//! - `grants_issued`: WAITING→GRANTED transitions the coordinator scan makes.
//! - `grants_reached_held`: grants that converted to a live HELD run claim.
//! - `grants_lost`: grants a cell was told about but did not convert (registry
//!   revoke, stale prefix, or a lost physical probe) — the revoke/thrash churn.
//! - `held_in_flight` / `distinct_held_cpus`: peak count of in-flight HELD
//!   claims and the distinct host CPUs they cover — the ramp indicator.
//! - `discover` count and wall-time: how often the placement contention-bail
//!   holder diagnostic runs and how long it takes. The bail probes only the
//!   contended LLC lock files with a non-blocking flock (no host-global
//!   `/proc/locks` walk), so `discover_ns` staying at syscall scale is the
//!   success metric for that fix.
//!
//! Entirely inert unless `KTSTR_BUILD_DIAGNOSTICS_DIR` is set (CI only): the
//! `note_*` calls are relaxed atomic updates and the expensive inputs
//! (distinct-CPU popcount, discover timing) are computed only behind
//! [`enabled`]. One aggregate line per process lands at
//! `${KTSTR_BUILD_DIAGNOSTICS_DIR}/grant-flow-<pid>.txt`, rewritten on each
//! authoritative grant scan and finally at exit, the same idiom as
//! [`super::exit_timing`] and the coordinator-wake counters. The mid-run
//! rewrite exists because the scan-running orchestrator's atexit line was
//! historically lost, undercounting `grants_issued` to near zero.

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const DIR_ENV: &str = "KTSTR_BUILD_DIAGNOSTICS_DIR";

static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

static GRANTS_ISSUED: AtomicU64 = AtomicU64::new(0);
static GRANTS_REACHED_HELD: AtomicU64 = AtomicU64::new(0);
static GRANTS_LOST: AtomicU64 = AtomicU64::new(0);
// LostGrant cause breakdown, to attribute the grant→commit churn.
static LOST_REVOKED: AtomicU64 = AtomicU64::new(0);
static LOST_SUFFIX_WATERMARK: AtomicU64 = AtomicU64::new(0);
static LOST_PREFIX_EPOCH: AtomicU64 = AtomicU64::new(0);
static LOST_STALE_REGRANT: AtomicU64 = AtomicU64::new(0);
static LOST_PHYSICAL_PROBE: AtomicU64 = AtomicU64::new(0);
// Licensed probe-loss characterization: `rotated` completions replaced their
// designation after the miss (default's opportunistic CPU-EX pin probe
// discards its evidence before rotating, so it lands here); the remaining
// buckets split unchanged completions by the exact rejecting resource and
// the probe mode that failed. `no_evidence` unchanged losses are pure
// requeues without a durable blocker (e.g. lost preparation-token races).
static LOST_PROBE_ROTATED: AtomicU64 = AtomicU64::new(0);
static LOST_PROBE_CPU_EX: AtomicU64 = AtomicU64::new(0);
static LOST_PROBE_CPU_SH: AtomicU64 = AtomicU64::new(0);
static LOST_PROBE_LLC: AtomicU64 = AtomicU64::new(0);
static LOST_PROBE_PERMIT: AtomicU64 = AtomicU64::new(0);
static LOST_PROBE_NO_EVIDENCE: AtomicU64 = AtomicU64::new(0);
// Default-mode opportunistic unshared placement probe outcomes (the
// transient CPU-EX pin over a published SH claim): a miss is only the
// signal to fall back, but each miss burns one full flock-probe cycle.
static DEFAULT_EXACT_HIT: AtomicU64 = AtomicU64::new(0);
static DEFAULT_EXACT_MISS: AtomicU64 = AtomicU64::new(0);
// Suffix-watermark parks at the entry self-demote with a REPLAN wave
// outstanding: the park also shortens the wave's deferred rescan edge
// (load-bearing coordination, not churn). The former per-park O(N) overlap
// classification walk is gone — it materialized every record's encoded watch
// under the registry lock and convoyed CI-scale entrant herds.
static WATERMARK_PARK_WAVE: AtomicU64 = AtomicU64::new(0);
// Dirty-watermark entrants/commits that PROCEEDED because their claim was
// disjoint from every claim accumulated since the last authoritative scan —
// the overlap-tested park converting former blanket parks into progress.
static WATERMARK_PROCEED_DISJOINT: AtomicU64 = AtomicU64::new(0);
// Non-fencing REPLAN claim replacements (registry.rs:16427), split by whether
// the replan reproduced the same claim. A `new == old` replacement still dirties
// the later suffix and parks every junior grant, yet introduces no new fence —
// the guard-headroom measurement for the no-op replan damping fix.
static REPLAN_REPLACE_TOTAL: AtomicU64 = AtomicU64::new(0);
static REPLAN_REPLACE_NOOP: AtomicU64 = AtomicU64::new(0);
// Of the replan replacements that DO change the claim, how many keep the same
// physical placement (CPUs + LLCs) and differ only in permits/mode/class. A
// high share means the suffix-dirty churn is permit-rotation, not a genuine
// placement move — the headroom a placement-scoped damping guard would need.
static REPLAN_REPLACE_PLACEMENT_SAME: AtomicU64 = AtomicU64::new(0);
// Speculative REPLAN callbacks that re-planned/requeued without acquiring — the
// elastic replanner's own churn, distinct from a licensed GRANTED grant loss.
static REPLAN_REQUEUE: AtomicU64 = AtomicU64::new(0);
// Grant-disjointness damping guard outcomes, per site. `skip` means the
// replacement/completion dirtied only the replan watermark word (no junior
// grant parks); `dirty` means it overlapped a live grant charge and dirtied
// the full suffix as before.
static GUARD_REPLACE_SKIP: AtomicU64 = AtomicU64::new(0);
static GUARD_REPLACE_DIRTY: AtomicU64 = AtomicU64::new(0);
static GUARD_COMPLETION_SKIP: AtomicU64 = AtomicU64::new(0);
static GUARD_COMPLETION_DIRTY: AtomicU64 = AtomicU64::new(0);
// Two-tier permit selection: per-candidate grant-charge hits during the
// grant-aware tier, and whether a planning pass had to fall back to the
// grant-blind tier (the fallback binding persistently at wave peaks means the
// residual contention is capacity truth, not planner blindness).
static PERMIT_CANDIDATES_TOTAL: AtomicU64 = AtomicU64::new(0);
static PERMIT_CANDIDATES_GRANT_CHARGED: AtomicU64 = AtomicU64::new(0);
static PERMIT_SELECT_GRANT_AWARE: AtomicU64 = AtomicU64::new(0);
static PERMIT_SELECT_FALLBACK: AtomicU64 = AtomicU64::new(0);
static HELD_IN_FLIGHT_MAX: AtomicU64 = AtomicU64::new(0);
static DISTINCT_HELD_CPUS_MAX: AtomicU64 = AtomicU64::new(0);
static DISCOVER_COUNT: AtomicU64 = AtomicU64::new(0);
static DISCOVER_NS_SUM: AtomicU64 = AtomicU64::new(0);
static DISCOVER_NS_MAX: AtomicU64 = AtomicU64::new(0);
static ATEXIT_REGISTERED: AtomicBool = AtomicBool::new(false);

fn dir() -> Option<&'static PathBuf> {
    DIR.get_or_init(|| {
        std::env::var_os(DIR_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    })
    .as_ref()
}

/// Whether the diagnostics sink is active. Callers gate any input computation
/// that is not itself free (distinct-CPU popcount, discover timing) on this.
pub(crate) fn enabled() -> bool {
    dir().is_some()
}

fn ensure_atexit() {
    if ATEXIT_REGISTERED.swap(true, Ordering::Relaxed) {
        return;
    }
    // SAFETY: `emit` is a plain extern "C" fn with no arguments, registered at
    // most once. `atexit` only stores the pointer.
    unsafe {
        libc::atexit(emit);
    }
}

fn bump_max(cell: &AtomicU64, value: u64) {
    let mut current = cell.load(Ordering::Relaxed);
    while value > current {
        match cell.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

/// One WAITING→GRANTED transition issued by the coordinator scan.
pub(crate) fn note_grant_issued() {
    GRANTS_ISSUED.fetch_add(1, Ordering::Relaxed);
    ensure_atexit();
}

/// One grant that converted into a live HELD run claim.
pub(crate) fn note_grant_reached_held() {
    GRANTS_REACHED_HELD.fetch_add(1, Ordering::Relaxed);
    ensure_atexit();
}

/// One grant a cell received but did not convert to HELD (revoke / stale
/// prefix / lost physical probe) and requeued for.
pub(crate) fn note_grant_lost() {
    GRANTS_LOST.fetch_add(1, Ordering::Relaxed);
    ensure_atexit();
}

/// The coordinator scan revoked this grant (an earlier ticket took its
/// resource): `GRANTED→REVOKED`, acknowledged here.
pub(crate) fn note_lost_revoked() {
    LOST_REVOKED.fetch_add(1, Ordering::Relaxed);
}

/// The suffix watermark parked this grant: some earlier ticket changed
/// (`min_changed_ticket < ticket`), whether or not it was relevant to this
/// claim — and even when the physical acquire had already succeeded.
pub(crate) fn note_lost_suffix_watermark() {
    LOST_SUFFIX_WATERMARK.fetch_add(1, Ordering::Relaxed);
}

/// Count one entry-level suffix-watermark park that coincided with an
/// outstanding REPLAN wave (the park doubles as wave-edge coordination).
pub(crate) fn note_watermark_park(wave_outstanding: bool) {
    if wave_outstanding {
        WATERMARK_PARK_WAVE.fetch_add(1, Ordering::Relaxed);
    }
}

/// One dirty-watermark entrant or commit that proceeded on accumulator
/// disjointness instead of parking.
pub(crate) fn note_watermark_proceed() {
    WATERMARK_PROCEED_DISJOINT.fetch_add(1, Ordering::Relaxed);
}

/// Count one non-fencing REPLAN claim replacement. `noop` flags an identical
/// claim (`new == old`); `placement_same` flags a changed claim that keeps the
/// same CPUs+LLCs (permit/mode/class churn only). Sizes the damping headroom.
pub(crate) fn note_replan_replace(noop: bool, placement_same: bool) {
    REPLAN_REPLACE_TOTAL.fetch_add(1, Ordering::Relaxed);
    if noop {
        REPLAN_REPLACE_NOOP.fetch_add(1, Ordering::Relaxed);
    }
    if placement_same {
        REPLAN_REPLACE_PLACEMENT_SAME.fetch_add(1, Ordering::Relaxed);
    }
}

/// The cached predecessor prefix epoch went stale.
pub(crate) fn note_lost_prefix_epoch() {
    LOST_PREFIX_EPOCH.fetch_add(1, Ordering::Relaxed);
}

/// A newer grant for the same exact claim was published while this callback
/// physically held it, so this one is consumed rather than re-committed.
pub(crate) fn note_lost_stale_regrant() {
    LOST_STALE_REGRANT.fetch_add(1, Ordering::Relaxed);
}

/// One licensed probe loss, pre-classified by [`classify_probe_loss`].
#[derive(Clone, Copy)]
pub(crate) enum ProbeLossClass {
    Rotated,
    CpuExclusive,
    CpuShared,
    Llc,
    Permit,
    NoEvidence,
}

/// Classify a licensed probe loss from its completion shape: whether the
/// completion rotated to a new designation, and the rejecting resource
/// evidence as `(is_cpu, is_permit, wanted_exclusive)` when retained.
pub(crate) fn classify_probe_loss(
    changed: bool,
    evidence: Option<(bool, bool, bool)>,
) -> ProbeLossClass {
    if changed {
        return ProbeLossClass::Rotated;
    }
    match evidence {
        Some((true, _, true)) => ProbeLossClass::CpuExclusive,
        Some((true, _, false)) => ProbeLossClass::CpuShared,
        Some((_, true, _)) => ProbeLossClass::Permit,
        Some((false, false, _)) => ProbeLossClass::Llc,
        None => ProbeLossClass::NoEvidence,
    }
}

/// The physical flock probe lost the resource to a real competing holder.
pub(crate) fn note_lost_physical_probe(class: ProbeLossClass) {
    LOST_PHYSICAL_PROBE.fetch_add(1, Ordering::Relaxed);
    let cell = match class {
        ProbeLossClass::Rotated => &LOST_PROBE_ROTATED,
        ProbeLossClass::CpuExclusive => &LOST_PROBE_CPU_EX,
        ProbeLossClass::CpuShared => &LOST_PROBE_CPU_SH,
        ProbeLossClass::Llc => &LOST_PROBE_LLC,
        ProbeLossClass::Permit => &LOST_PROBE_PERMIT,
        ProbeLossClass::NoEvidence => &LOST_PROBE_NO_EVIDENCE,
    };
    cell.fetch_add(1, Ordering::Relaxed);
}

/// One default-mode opportunistic unshared (CPU-EX pin) placement probe.
pub(crate) fn note_default_exact_probe(hit: bool) {
    if hit {
        DEFAULT_EXACT_HIT.fetch_add(1, Ordering::Relaxed);
    } else {
        DEFAULT_EXACT_MISS.fetch_add(1, Ordering::Relaxed);
    }
}

/// A speculative REPLAN callback requeued without acquiring (normal elastic
/// replanning), distinct from a licensed grant loss.
pub(crate) fn note_replan_requeue() {
    REPLAN_REQUEUE.fetch_add(1, Ordering::Relaxed);
}

/// One NonFencing replacement through the grant-disjointness guard.
/// `skipped` means the main suffix watermark stayed clean.
pub(crate) fn note_replace_guard(skipped: bool) {
    if skipped {
        GUARD_REPLACE_SKIP.fetch_add(1, Ordering::Relaxed);
    } else {
        GUARD_REPLACE_DIRTY.fetch_add(1, Ordering::Relaxed);
    }
}

/// One unchanged REPLAN completion through the grant-disjointness guard.
pub(crate) fn note_completion_guard(skipped: bool) {
    if skipped {
        GUARD_COMPLETION_SKIP.fetch_add(1, Ordering::Relaxed);
    } else {
        GUARD_COMPLETION_DIRTY.fetch_add(1, Ordering::Relaxed);
    }
}

/// One candidate evaluated by the grant-aware permit tier. Only called behind
/// [`enabled`] (per-candidate volume).
pub(crate) fn note_permit_candidate(grant_charged: bool) {
    PERMIT_CANDIDATES_TOTAL.fetch_add(1, Ordering::Relaxed);
    if grant_charged {
        PERMIT_CANDIDATES_GRANT_CHARGED.fetch_add(1, Ordering::Relaxed);
    }
}

/// One two-tier permit selection pass; `fallback` means the grant-free tier
/// found nothing and the grant-blind rerun decided the outcome.
pub(crate) fn note_permit_selection(fallback: bool) {
    if fallback {
        PERMIT_SELECT_FALLBACK.fetch_add(1, Ordering::Relaxed);
    } else {
        PERMIT_SELECT_GRANT_AWARE.fetch_add(1, Ordering::Relaxed);
    }
    ensure_atexit();
}

/// Peak in-flight HELD run claims and the distinct host CPUs they cover,
/// sampled once per authoritative grant scan. Only called behind [`enabled`].
///
/// Also persists the current counter image: the process running the
/// authoritative scans is typically the long-lived orchestrator, whose
/// `atexit` line was historically lost (its `grants_issued` never reached the
/// diagnostics directory, making issued:held ratios meaningless). Each scan
/// rewrites this process's file, so the last image survives however the
/// process ends.
pub(crate) fn note_held_in_flight(held_records: u64, distinct_cpus: u64) {
    bump_max(&HELD_IN_FLIGHT_MAX, held_records);
    bump_max(&DISTINCT_HELD_CPUS_MAX, distinct_cpus);
    ensure_atexit();
    persist_now();
}

/// One placement contention-bail holder diagnostic, and its wall-time. Since
/// the bail now probes only the contended LLC lock files with a non-blocking
/// flock (no host-global `/proc/locks` walk), this both counts how often the
/// diagnostic runs and proves the walk is gone: `discover_ns` should sit at
/// per-needle syscall scale, not the hundreds of milliseconds the seq-file read
/// cost under a many-lane flock storm. Only called behind [`enabled`].
pub(crate) fn note_discover(elapsed_ns: u64) {
    DISCOVER_COUNT.fetch_add(1, Ordering::Relaxed);
    DISCOVER_NS_SUM.fetch_add(elapsed_ns, Ordering::Relaxed);
    bump_max(&DISCOVER_NS_MAX, elapsed_ns);
    ensure_atexit();
}

fn format_line(pid: u32) -> String {
    let discover_count = DISCOVER_COUNT.load(Ordering::Relaxed);
    let discover_ns_mean = DISCOVER_NS_SUM
        .load(Ordering::Relaxed)
        .checked_div(discover_count)
        .unwrap_or(0);
    format!(
        "grant-flow: pid={pid} grants_issued={} grants_reached_held={} grants_lost={} \
         lost_revoked={} lost_suffix_watermark={} lost_prefix_epoch={} \
         lost_stale_regrant={} lost_physical_probe={} replan_requeue={} \
         lost_probe_rotated={} lost_probe_cpu_ex={} lost_probe_cpu_sh={} \
         lost_probe_llc={} lost_probe_permit={} lost_probe_no_evidence={} \
         default_exact_hit={} default_exact_miss={} \
         wmpark_wave={} wmpark_proceed_disjoint={} \
         replan_replace_total={} replan_replace_noop={} replan_replace_placement_same={} \
         guard_replace_skip={} guard_replace_dirty={} \
         guard_completion_skip={} guard_completion_dirty={} \
         permit_candidates_total={} permit_candidates_grant_charged={} \
         permit_select_grant_aware={} permit_select_fallback={} \
         held_in_flight_max={} distinct_held_cpus_max={} discover_count={} \
         discover_ns_mean={discover_ns_mean} discover_ns_max={}\n",
        GRANTS_ISSUED.load(Ordering::Relaxed),
        GRANTS_REACHED_HELD.load(Ordering::Relaxed),
        GRANTS_LOST.load(Ordering::Relaxed),
        LOST_REVOKED.load(Ordering::Relaxed),
        LOST_SUFFIX_WATERMARK.load(Ordering::Relaxed),
        LOST_PREFIX_EPOCH.load(Ordering::Relaxed),
        LOST_STALE_REGRANT.load(Ordering::Relaxed),
        LOST_PHYSICAL_PROBE.load(Ordering::Relaxed),
        REPLAN_REQUEUE.load(Ordering::Relaxed),
        LOST_PROBE_ROTATED.load(Ordering::Relaxed),
        LOST_PROBE_CPU_EX.load(Ordering::Relaxed),
        LOST_PROBE_CPU_SH.load(Ordering::Relaxed),
        LOST_PROBE_LLC.load(Ordering::Relaxed),
        LOST_PROBE_PERMIT.load(Ordering::Relaxed),
        LOST_PROBE_NO_EVIDENCE.load(Ordering::Relaxed),
        DEFAULT_EXACT_HIT.load(Ordering::Relaxed),
        DEFAULT_EXACT_MISS.load(Ordering::Relaxed),
        WATERMARK_PARK_WAVE.load(Ordering::Relaxed),
        WATERMARK_PROCEED_DISJOINT.load(Ordering::Relaxed),
        REPLAN_REPLACE_TOTAL.load(Ordering::Relaxed),
        REPLAN_REPLACE_NOOP.load(Ordering::Relaxed),
        REPLAN_REPLACE_PLACEMENT_SAME.load(Ordering::Relaxed),
        GUARD_REPLACE_SKIP.load(Ordering::Relaxed),
        GUARD_REPLACE_DIRTY.load(Ordering::Relaxed),
        GUARD_COMPLETION_SKIP.load(Ordering::Relaxed),
        GUARD_COMPLETION_DIRTY.load(Ordering::Relaxed),
        PERMIT_CANDIDATES_TOTAL.load(Ordering::Relaxed),
        PERMIT_CANDIDATES_GRANT_CHARGED.load(Ordering::Relaxed),
        PERMIT_SELECT_GRANT_AWARE.load(Ordering::Relaxed),
        PERMIT_SELECT_FALLBACK.load(Ordering::Relaxed),
        HELD_IN_FLIGHT_MAX.load(Ordering::Relaxed),
        DISTINCT_HELD_CPUS_MAX.load(Ordering::Relaxed),
        discover_count,
        DISCOVER_NS_MAX.load(Ordering::Relaxed),
    )
}

/// Rewrite this process's per-pid counter file with the current image. The
/// file holds exactly one line; each persist (mid-run from the scan loop, and
/// the final `atexit` image) replaces the previous one, so readers always see
/// one complete, most-recent line per process.
pub(crate) fn persist_now() {
    let Some(dir) = dir() else {
        return;
    };
    let pid = std::process::id();
    let line = format_line(pid);
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(dir.join(format!("grant-flow-{pid}.txt")))
    {
        let _ = file.write_all(line.as_bytes());
    }
}

extern "C" fn emit() {
    let _ = std::panic::catch_unwind(persist_now);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_shape_has_all_fields() {
        let line = format_line(42);
        for field in [
            "grant-flow: pid=42",
            "grants_issued=",
            "grants_reached_held=",
            "grants_lost=",
            "lost_revoked=",
            "lost_suffix_watermark=",
            "lost_prefix_epoch=",
            "lost_stale_regrant=",
            "lost_physical_probe=",
            "replan_requeue=",
            "lost_probe_rotated=",
            "lost_probe_cpu_ex=",
            "lost_probe_cpu_sh=",
            "lost_probe_llc=",
            "lost_probe_permit=",
            "lost_probe_no_evidence=",
            "default_exact_hit=",
            "default_exact_miss=",
            "wmpark_wave=",
            "wmpark_proceed_disjoint=",
            "replan_replace_total=",
            "replan_replace_noop=",
            "replan_replace_placement_same=",
            "guard_replace_skip=",
            "guard_replace_dirty=",
            "guard_completion_skip=",
            "guard_completion_dirty=",
            "permit_candidates_total=",
            "permit_candidates_grant_charged=",
            "permit_select_grant_aware=",
            "permit_select_fallback=",
            "held_in_flight_max=",
            "distinct_held_cpus_max=",
            "discover_count=",
            "discover_ns_mean=",
            "discover_ns_max=",
        ] {
            assert!(line.contains(field), "missing {field} in {line}");
        }
        assert!(line.ends_with('\n'));
    }

    #[test]
    fn max_accumulator_keeps_the_largest() {
        let cell = AtomicU64::new(0);
        bump_max(&cell, 5);
        bump_max(&cell, 3);
        bump_max(&cell, 9);
        bump_max(&cell, 1);
        assert_eq!(cell.load(Ordering::Relaxed), 9);
    }
}
