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
//! `${KTSTR_BUILD_DIAGNOSTICS_DIR}/grant-flow-<pid>.txt` on exit, the same idiom
//! as [`super::exit_timing`] and the coordinator-wake counters.

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
// Suffix-watermark park breakdown at the entry self-demote (registry.rs:2622),
// to size the relevance-gate headroom vs the REPLAN-wave-coordination floor:
// - no_overlap: the record's claim conflicts with no live claim — a candidate
//   for the relevance-gate to commit instead of park.
// - with_overlap: a real conflict exists — parking is a fairness fence.
// - wave: a REPLAN wave was outstanding at park time — the park also shortens
//   the wave's deferred rescan edge (load-bearing coordination, not churn).
static WATERMARK_PARK_NO_OVERLAP: AtomicU64 = AtomicU64::new(0);
static WATERMARK_PARK_WITH_OVERLAP: AtomicU64 = AtomicU64::new(0);
static WATERMARK_PARK_WAVE: AtomicU64 = AtomicU64::new(0);
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

/// Classify one entry-level suffix-watermark park (registry.rs:2622) by whether
/// the parked claim actually overlaps a live claim and whether a REPLAN wave was
/// outstanding. Sizes the relevance-gate headroom against the wave-coordination
/// floor. Only called behind [`enabled`] (the overlap test is O(resources)).
pub(crate) fn note_watermark_park(overlap: bool, wave_outstanding: bool) {
    if overlap {
        WATERMARK_PARK_WITH_OVERLAP.fetch_add(1, Ordering::Relaxed);
    } else {
        WATERMARK_PARK_NO_OVERLAP.fetch_add(1, Ordering::Relaxed);
    }
    if wave_outstanding {
        WATERMARK_PARK_WAVE.fetch_add(1, Ordering::Relaxed);
    }
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

/// The physical flock probe lost the resource to a real competing holder.
pub(crate) fn note_lost_physical_probe() {
    LOST_PHYSICAL_PROBE.fetch_add(1, Ordering::Relaxed);
}

/// A speculative REPLAN callback requeued without acquiring (normal elastic
/// replanning), distinct from a licensed grant loss.
pub(crate) fn note_replan_requeue() {
    REPLAN_REQUEUE.fetch_add(1, Ordering::Relaxed);
}

/// Peak in-flight HELD run claims and the distinct host CPUs they cover,
/// sampled once per authoritative grant scan. Only called behind [`enabled`].
pub(crate) fn note_held_in_flight(held_records: u64, distinct_cpus: u64) {
    bump_max(&HELD_IN_FLIGHT_MAX, held_records);
    bump_max(&DISTINCT_HELD_CPUS_MAX, distinct_cpus);
    ensure_atexit();
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
         wmpark_no_overlap={} wmpark_with_overlap={} wmpark_wave={} \
         replan_replace_total={} replan_replace_noop={} replan_replace_placement_same={} \
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
        WATERMARK_PARK_NO_OVERLAP.load(Ordering::Relaxed),
        WATERMARK_PARK_WITH_OVERLAP.load(Ordering::Relaxed),
        WATERMARK_PARK_WAVE.load(Ordering::Relaxed),
        REPLAN_REPLACE_TOTAL.load(Ordering::Relaxed),
        REPLAN_REPLACE_NOOP.load(Ordering::Relaxed),
        REPLAN_REPLACE_PLACEMENT_SAME.load(Ordering::Relaxed),
        HELD_IN_FLIGHT_MAX.load(Ordering::Relaxed),
        DISTINCT_HELD_CPUS_MAX.load(Ordering::Relaxed),
        discover_count,
        DISCOVER_NS_MAX.load(Ordering::Relaxed),
    )
}

extern "C" fn emit() {
    let _ = std::panic::catch_unwind(|| {
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
            .append(true)
            .open(dir.join(format!("grant-flow-{pid}.txt")))
        {
            let _ = file.write_all(line.as_bytes());
        }
    });
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
            "wmpark_no_overlap=",
            "wmpark_with_overlap=",
            "wmpark_wave=",
            "replan_replace_total=",
            "replan_replace_noop=",
            "replan_replace_placement_same=",
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
