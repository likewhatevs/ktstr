//! Grant-flow diagnostics (diagnostic-only, pure observation).
//!
//! The slim core that outlived the admission-wall investigation:
//! - `grants_issued`: WAITING→GRANTED transitions the coordinator scan makes.
//! - `grants_reached_held`: records published as a live HELD run claim.
//! - `grants_lost`: licensed grants a cell was told about but did not convert.
//! - `held_in_flight` / `granted_in_flight` / `distinct_held_cpus`: peak count
//!   of in-flight GRANTED+HELD claims, of the GRANTED subset alone, and the
//!   distinct host CPUs they cover — the ramp indicator. Both fence later
//!   tickets identically, so the gap between the two is the fenced footprint
//!   that is not running anything.
//! - `backfill_scans` / `backfill_head_scans` / `backfill_head_age_ns_max` /
//!   `backfill_head_capacity_max`: how often an authoritative scan found a
//!   physically blocked exact head to protect at all, and how much bounded
//!   admission that head had left. The bounded-admission fence can only
//!   withhold a grant on a scan that selected a head.
//! - `discover` count and wall-time: how often the placement contention-bail
//!   holder diagnostic runs and how long it takes. The bail probes only the
//!   contended LLC lock files with a non-blocking flock (no host-global
//!   `/proc/locks` walk), so `discover_ns` staying at syscall scale is the
//!   regression tripwire for that fix.
//!
//! The first three are independent rates, not a conserved flow: read them
//! against each other over time, never as `issued == reached_held + lost`. A
//! preparation intent is granted like any other waiter, but its grant converts
//! to a parked PENDING preparation rather than to HELD, and the later one-shot
//! activation that does publish HELD needs no fresh grant.
//!
//! The per-cause loss/probe/replan/guard/watermark breakdown counters this
//! module once carried answered their questions (the watermark-park and
//! EX-pin-rotation churn producers are fixed) and were removed with them.
//!
//! [`GrantBlock`] is the current breakdown: why a wake failed to convert.
//! `grants_issued` says a grant was licensed and `grants_reached_held` says it
//! converted; the gap between them is unattributable without naming the fence
//! that rejected each candidate.
//!
//! Entirely inert unless `KTSTR_BUILD_DIAGNOSTICS_DIR` is set (CI only): the
//! `note_*` calls are relaxed atomic updates and the expensive inputs
//! (distinct-CPU popcount, discover timing) are computed only behind
//! [`enabled`]. One aggregate line per process lands at
//! `${KTSTR_BUILD_DIAGNOSTICS_DIR}/grant-flow-<pid>.txt`, rewritten
//! periodically by the coordinator loop (outside the registry flock) and
//! finally at exit — the scan-running orchestrator's atexit line alone is
//! lost when that process is killed.

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const DIR_ENV: &str = "KTSTR_BUILD_DIAGNOSTICS_DIR";

static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

static GRANTS_ISSUED: AtomicU64 = AtomicU64::new(0);
static GRANTS_REACHED_HELD: AtomicU64 = AtomicU64::new(0);
static GRANTS_LOST: AtomicU64 = AtomicU64::new(0);
static HELD_IN_FLIGHT_MAX: AtomicU64 = AtomicU64::new(0);
static GRANTED_IN_FLIGHT_MAX: AtomicU64 = AtomicU64::new(0);
static DISTINCT_HELD_CPUS_MAX: AtomicU64 = AtomicU64::new(0);
static BACKFILL_SCANS: AtomicU64 = AtomicU64::new(0);
static BACKFILL_HEAD_SCANS: AtomicU64 = AtomicU64::new(0);
static BACKFILL_HEAD_AGE_NS_MAX: AtomicU64 = AtomicU64::new(0);
static BACKFILL_HEAD_CAPACITY_MAX: AtomicU64 = AtomicU64::new(0);
static DISCOVER_COUNT: AtomicU64 = AtomicU64::new(0);
static DISCOVER_NS_SUM: AtomicU64 = AtomicU64::new(0);
static DISCOVER_NS_MAX: AtomicU64 = AtomicU64::new(0);
static ATEXIT_REGISTERED: AtomicBool = AtomicBool::new(false);
static BLOCKS: [AtomicU64; GrantBlock::COUNT] = [const { AtomicU64::new(0) }; GrantBlock::COUNT];

/// Why a wake failed to convert into a HELD run claim.
///
/// The first four name the fence a single candidate placement hit; a wake
/// evaluates many candidates, so these count candidate rejections, not wakes.
/// [`GrantBlock::Permits`] and [`GrantBlock::NoCandidate`] are per-wake
/// outcomes of the selection as a whole, and [`GrantBlock::Unlicensed`] counts
/// wakes that never carried a grant to lose. Read the two groups separately —
/// like the headline totals above they are independent rates, not a
/// conserved flow.
///
/// The `predecessors_*` variants are sub-counts: each also increments
/// [`GrantBlock::Predecessors`], so they partition it by the resource axis the
/// reservation fenced rather than adding to it.
///
/// The `scan_*` group is a third, independent group counted on the other side
/// of the protocol: one per queued waiter per authoritative grant scan that
/// declined to grant it. Everything above is wake-side, and a waiter the scan
/// never grants is never woken to reject a candidate, so only this group can
/// see why admission itself stalled. `scan_conflict_*` partition
/// [`GrantBlock::ScanConflict`] and `scan_fairness_*` partition
/// [`GrantBlock::ScanFairness`], the same way the `predecessors_*` pair does.
#[derive(Clone, Copy)]
pub(crate) enum GrantBlock {
    /// The wake carried no grant license (a REPLAN re-designation), so no
    /// candidate could convert however free the host was.
    Unlicensed,
    /// The weighted permit pool offered no acceptable set, so the selection
    /// never reached a topology candidate.
    Permits,
    /// Permits resolved but every topology candidate was rejected below.
    NoCandidate,
    /// A candidate resource is the blocker this wake's own probe just met.
    Contended,
    /// An older ticket's reservation covers a candidate resource.
    Predecessors,
    /// A candidate resource is observed held by someone else.
    Busy,
    /// A candidate resource has never been observed. The availability fence
    /// rejects unobserved and held identically, so on a mostly-idle host this
    /// is the cause that looks like contention but is not.
    Unobserved,
    /// Sub-count of [`GrantBlock::Predecessors`]: the reserved resource is a
    /// weighted admission permit. Permit selection walks the whole pool through
    /// the same fence, so a busy pool raises `predecessors` without any
    /// placement being fenced off the host.
    PredecessorsPermit,
    /// Sub-count of [`GrantBlock::Predecessors`]: the reserved resource is an
    /// LLC. The CPU-axis remainder is `predecessors` minus these two.
    PredecessorsLlc,
    /// Scan side: the waiter's exact claim overlaps the live predecessor
    /// prefix. Bounded admission (backfill) is a separate term of the same
    /// decision and cannot lift this one: a real conflict withholds the grant
    /// whether or not a backfill head exists.
    ScanConflict,
    /// Sub-count of [`GrantBlock::ScanConflict`] on the permit axis.
    ScanConflictPermit,
    /// Sub-count of [`GrantBlock::ScanConflict`] on the LLC axis.
    ScanConflictLlc,
    /// Scan side: the waiter's claim is free of predecessors but a resource it
    /// names is observed held, or has never been observed at all.
    ScanUnavailable,
    /// Scan side: the waiter is pinned to a blocker whose serial has not moved.
    ScanBlocker,
    /// Scan side: the bounded-admission (backfill) fence withheld the grant to
    /// keep a physically blocked exact head from starving. This is the only
    /// cause that backfill itself produces.
    ScanFairness,
    /// Sub-count of [`GrantBlock::ScanFairness`]: the head's bounded admission
    /// window has expired, so no replacement bypass work is issued.
    ScanFairnessWindow,
    /// Sub-count of [`GrantBlock::ScanFairness`]: the window is open but the
    /// head's bypass capacity is fully spent by outstanding conflicting work.
    ScanFairnessCapacity,
    /// Scan side: a preparation intent found no free slot in the preparation
    /// token pool.
    ScanPreparationPool,
}

impl GrantBlock {
    const COUNT: usize = 18;

    const ALL: [Self; Self::COUNT] = [
        Self::Unlicensed,
        Self::Permits,
        Self::NoCandidate,
        Self::Contended,
        Self::Predecessors,
        Self::Busy,
        Self::Unobserved,
        Self::PredecessorsPermit,
        Self::PredecessorsLlc,
        Self::ScanConflict,
        Self::ScanConflictPermit,
        Self::ScanConflictLlc,
        Self::ScanUnavailable,
        Self::ScanBlocker,
        Self::ScanFairness,
        Self::ScanFairnessWindow,
        Self::ScanFairnessCapacity,
        Self::ScanPreparationPool,
    ];

    const fn index(self) -> usize {
        self as usize
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Unlicensed => "unlicensed",
            Self::Permits => "permits",
            Self::NoCandidate => "no_candidate",
            Self::Contended => "contended",
            Self::Predecessors => "predecessors",
            Self::Busy => "busy",
            Self::Unobserved => "unobserved",
            Self::PredecessorsPermit => "predecessors_permit",
            Self::PredecessorsLlc => "predecessors_llc",
            Self::ScanConflict => "scan_conflict",
            Self::ScanConflictPermit => "scan_conflict_permit",
            Self::ScanConflictLlc => "scan_conflict_llc",
            Self::ScanUnavailable => "scan_unavailable",
            Self::ScanBlocker => "scan_blocker",
            Self::ScanFairness => "scan_fairness",
            Self::ScanFairnessWindow => "scan_fairness_window",
            Self::ScanFairnessCapacity => "scan_fairness_capacity",
            Self::ScanPreparationPool => "scan_preparation_pool",
        }
    }
}

/// One conversion fence hit. Unlike the `note_*` totals above this fires per
/// CANDIDATE per scan, so it checks [`enabled`] before touching anything —
/// including the `atexit` registration, which a diagnostics-off process has no
/// image to write.
pub(crate) fn note_block(block: GrantBlock) {
    if !enabled() {
        return;
    }
    BLOCKS[block.index()].fetch_add(1, Ordering::Relaxed);
    ensure_atexit();
}

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

/// One licensed grant a cell received but did not convert to HELD and
/// requeued for.
pub(crate) fn note_grant_lost() {
    GRANTS_LOST.fetch_add(1, Ordering::Relaxed);
    ensure_atexit();
}

/// Peak in-flight GRANTED+HELD claims and the distinct host CPUs they cover,
/// sampled once per authoritative grant scan. Only called behind [`enabled`].
/// Deliberately no file IO here — this runs under the registry EX flock; the
/// coordinator loop persists the image periodically outside the lock (the
/// atexit line alone is lost when the scan-running orchestrator is killed).
pub(crate) fn note_held_in_flight(held_records: u64, granted_records: u64, distinct_cpus: u64) {
    bump_max(&HELD_IN_FLIGHT_MAX, held_records);
    bump_max(&GRANTED_IN_FLIGHT_MAX, granted_records);
    bump_max(&DISTINCT_HELD_CPUS_MAX, distinct_cpus);
    ensure_atexit();
}

/// One authoritative grant scan's bounded-admission (backfill) outcome,
/// sampled once per scan behind [`enabled`]. `head` is `None` when the scan
/// found no physically blocked exact head to protect; the fence is then inert
/// for that scan and withholds nothing. Otherwise it carries the head's age
/// inside its bounded admission window and the bypass capacity that window
/// opens. Read `backfill_head_scans` against `backfill_scans` before reading
/// anything into `block_scan_fairness`.
pub(crate) fn note_backfill_scan(head: Option<(u64, u32)>) {
    BACKFILL_SCANS.fetch_add(1, Ordering::Relaxed);
    ensure_atexit();
    let Some((age_ns, capacity)) = head else {
        return;
    };
    BACKFILL_HEAD_SCANS.fetch_add(1, Ordering::Relaxed);
    bump_max(&BACKFILL_HEAD_AGE_NS_MAX, age_ns);
    bump_max(&BACKFILL_HEAD_CAPACITY_MAX, u64::from(capacity));
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
    let blocks = GrantBlock::ALL
        .iter()
        .map(|block| {
            format!(
                " block_{}={}",
                block.label(),
                BLOCKS[block.index()].load(Ordering::Relaxed),
            )
        })
        .collect::<String>();
    format!(
        "grant-flow: pid={pid} grants_issued={} grants_reached_held={} grants_lost={} \
         held_in_flight_max={} granted_in_flight_max={} distinct_held_cpus_max={} \
         backfill_scans={} backfill_head_scans={} backfill_head_age_ns_max={} \
         backfill_head_capacity_max={} discover_count={} \
         discover_ns_mean={discover_ns_mean} discover_ns_max={}{blocks}\n",
        GRANTS_ISSUED.load(Ordering::Relaxed),
        GRANTS_REACHED_HELD.load(Ordering::Relaxed),
        GRANTS_LOST.load(Ordering::Relaxed),
        HELD_IN_FLIGHT_MAX.load(Ordering::Relaxed),
        GRANTED_IN_FLIGHT_MAX.load(Ordering::Relaxed),
        DISTINCT_HELD_CPUS_MAX.load(Ordering::Relaxed),
        BACKFILL_SCANS.load(Ordering::Relaxed),
        BACKFILL_HEAD_SCANS.load(Ordering::Relaxed),
        BACKFILL_HEAD_AGE_NS_MAX.load(Ordering::Relaxed),
        BACKFILL_HEAD_CAPACITY_MAX.load(Ordering::Relaxed),
        discover_count,
        DISCOVER_NS_MAX.load(Ordering::Relaxed),
    )
}

/// Rewrite this process's per-pid counter file with the current image. The
/// file holds exactly one line; each persist (periodic from the coordinator
/// loop, and the final `atexit` image) replaces the previous one, so readers
/// always see one complete, most-recent line per process.
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
            "held_in_flight_max=",
            "granted_in_flight_max=",
            "distinct_held_cpus_max=",
            "backfill_scans=",
            "backfill_head_scans=",
            "backfill_head_age_ns_max=",
            "backfill_head_capacity_max=",
            "discover_count=",
            "discover_ns_mean=",
            "discover_ns_max=",
            "block_unlicensed=",
            "block_permits=",
            "block_no_candidate=",
            "block_contended=",
            "block_predecessors=",
            "block_busy=",
            "block_unobserved=",
            "block_predecessors_permit=",
            "block_predecessors_llc=",
            "block_scan_conflict=",
            "block_scan_conflict_permit=",
            "block_scan_conflict_llc=",
            "block_scan_unavailable=",
            "block_scan_blocker=",
            "block_scan_fairness=",
            "block_scan_fairness_window=",
            "block_scan_fairness_capacity=",
            "block_scan_preparation_pool=",
        ] {
            assert!(line.contains(field), "missing {field} in {line}");
        }
        assert!(line.ends_with('\n'));
    }

    /// Every variant owns a distinct slot, so no two causes alias in the line.
    #[test]
    fn block_variants_index_distinct_slots() {
        let mut indices = GrantBlock::ALL.map(GrantBlock::index);
        indices.sort_unstable();
        assert_eq!(
            indices,
            std::array::from_fn::<_, { GrantBlock::COUNT }, _>(|index| index)
        );
    }

    /// The counters stay untouched with the sink off — the whole breakdown is
    /// as inert on a production run as the totals it sits beside.
    #[test]
    fn blocks_are_inert_without_a_diagnostics_dir() {
        if enabled() {
            return;
        }
        let before = BLOCKS[GrantBlock::Unobserved.index()].load(Ordering::Relaxed);
        note_block(GrantBlock::Unobserved);
        assert_eq!(
            BLOCKS[GrantBlock::Unobserved.index()].load(Ordering::Relaxed),
            before,
        );
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
