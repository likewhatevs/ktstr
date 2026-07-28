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
//! - `held_in_flight_mean` / `granted_in_flight_mean` over `in_flight_samples`
//!   spanning `in_flight_span_ns`: the same two depths averaged rather than
//!   peaked. Little's law closes the mean life of an issued grant against
//!   `grants_issued` — `granted_in_flight_mean * in_flight_span_ns /
//!   grants_issued` — which the peak alone cannot give.
//! - `callback_count` / `callback_ns_mean` / `callback_ns_max`: wall spent
//!   inside a licensed grant callback. Subtracted from the residency above it
//!   leaves the scan-publish-to-owner-wake latency, so the two together say
//!   whether an unconverted grant is stuck waiting to be woken or stuck doing
//!   its own probe.
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
//! # `grants_reached_held` is a denominator, not a throughput
//!
//! Read this before deriving anything from it. A lane admits a fixed set of
//! cells — the suite decides how many, not the admission protocol — and each
//! publishes its HELD run claim exactly once, so `grants_reached_held` is a
//! constant of the workload. It has measured 368-374 on every lane of every
//! configuration this protocol has been through — before and after the
//! granted-entrant park, at 18% and at 29% of grants issued. Admission getting
//! faster cannot raise it; it moves only when a cell fails to run at all.
//!
//! The consequence is that `grants_reached_held / grants_issued` is `1 /
//! grants burned per admission`: a **waste ratio**, not a conversion rate and
//! not throughput. Halving the waste does not run one extra cell. Steering a
//! performance change on it reads a fixed numerator over a noisy denominator.
//!
//! `running_in_flight` is the throughput term. `held_in_flight` counts every
//! in-flight GRANTED+HELD record and `granted_in_flight` the GRANTED subset;
//! both fence later tickets identically, but only the difference is running a
//! VM. That difference against the lane's fixed cell count and mean hold time
//! is what predicts wall clock.
//!
//! # Scan-side counters need their denominator
//!
//! The `scan_*` group fires once per queued waiter per authoritative scan, so
//! its totals scale with how deep the queue was — a lane whose queue is three
//! times deeper reports three times the rejections at identical behavior.
//! `grant_scans` and `records_scanned` ship on this line for exactly that
//! reason: normalize by `records_scanned` before comparing any `scan_*` count
//! across runs. The wake-side counters have a different denominator again
//! (candidates walked, not records scanned), so the two groups are not
//! comparable to each other even within one run.
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
//! - `changed_*`: one sample per authoritative grant scan of the changed-claims
//!   accumulator as the scan found it, taken immediately before the scan resets
//!   it. `changed_dirty_scans` against `changed_scans` says how often the suffix
//!   watermark was armed at all, `changed_saturated_scans` how often the
//!   accumulator lost claim precision outright, and the bit counts how much of
//!   the host the accumulator covered when the scan cleared it. Read them
//!   against `block_park_entry` / `block_park_commit`: a large accumulated
//!   footprint only matters if granted entrants actually park on it.
//!
//! Every `*_mean` field is a PER-PROCESS mean over that process's own sample
//! count, so it is only meaningful re-weighted by the paired count — reading a
//! lane by taking the maximum of the per-process means reports whichever
//! process happened to take two samples, not the lane. Each mean therefore
//! ships beside its raw `*_sum` and its sample count (`discover_count`,
//! `changed_scans`, `in_flight_samples`, `callback_count`), so a reader can sum
//! the numerators and denominators across processes and never has to trust or
//! re-derive a mean.
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
static HELD_IN_FLIGHT_SUM: AtomicU64 = AtomicU64::new(0);
static GRANTED_IN_FLIGHT_SUM: AtomicU64 = AtomicU64::new(0);
static RUNNING_IN_FLIGHT_SUM: AtomicU64 = AtomicU64::new(0);
static RUNNING_IN_FLIGHT_MAX: AtomicU64 = AtomicU64::new(0);
static IN_FLIGHT_SAMPLES: AtomicU64 = AtomicU64::new(0);
static IN_FLIGHT_FIRST_NS: AtomicU64 = AtomicU64::new(0);
static IN_FLIGHT_LAST_NS: AtomicU64 = AtomicU64::new(0);
static CALLBACK_NS_SUM: AtomicU64 = AtomicU64::new(0);
static CALLBACK_NS_MAX: AtomicU64 = AtomicU64::new(0);
static CALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);
static DISCOVER_COUNT: AtomicU64 = AtomicU64::new(0);
static DISCOVER_NS_SUM: AtomicU64 = AtomicU64::new(0);
static DISCOVER_NS_MAX: AtomicU64 = AtomicU64::new(0);
static CHANGED_SCANS: AtomicU64 = AtomicU64::new(0);
static CHANGED_DIRTY_SCANS: AtomicU64 = AtomicU64::new(0);
static CHANGED_SATURATED_SCANS: AtomicU64 = AtomicU64::new(0);
static CHANGED_CPU_BITS_SUM: AtomicU64 = AtomicU64::new(0);
static CHANGED_CPU_BITS_MAX: AtomicU64 = AtomicU64::new(0);
static CHANGED_PERMIT_BITS_SUM: AtomicU64 = AtomicU64::new(0);
static CHANGED_PERMIT_BITS_MAX: AtomicU64 = AtomicU64::new(0);
static CHANGED_LLC_BITS_SUM: AtomicU64 = AtomicU64::new(0);
static CHANGED_LLC_BITS_MAX: AtomicU64 = AtomicU64::new(0);
static CHANGED_WATERMARK_SPAN_MAX: AtomicU64 = AtomicU64::new(0);
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
/// [`GrantBlock::ScanConflict`] the same way the `predecessors_*` pair does.
///
/// The last group — `scan_revoke_*`, `revoked_unwoken`, `park_*` and
/// `prepared` — accounts for what became of an ISSUED grant, one per grant
/// rather than per candidate. Every other outcome of a licensed grant already
/// lands in `grants_reached_held` or `grants_lost`, so these are the terms that
/// close the `grants_issued` books: a grant the scan revoked, a revocation the
/// owner's wake never even observed as GRANTED (which therefore fires no
/// `grants_lost`), a grant the changed-claims fence sent back to WAITING, and a
/// grant that legitimately became a parked PENDING preparation instead of a
/// HELD run claim. `scan_revoke_conflict` and `scan_revoke_unavailable`
/// partition [`GrantBlock::ScanRevoke`]; the permit/LLC pair partitions
/// `scan_revoke_conflict` by axis with the CPU axis as the remainder.
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
    /// prefix.
    ScanConflict,
    /// Sub-count of [`GrantBlock::ScanConflict`] on the permit axis.
    ScanConflictPermit,
    /// Sub-count of [`GrantBlock::ScanConflict`] on the LLC axis.
    ScanConflictLlc,
    /// Scan side: the waiter's claim is free of predecessors but a resource it
    /// names is observed held, or has never been observed at all.
    ScanUnavailable,
    /// Sub-count of [`GrantBlock::ScanUnavailable`]: at least one named
    /// resource has never been observed. This is the opposite cause to the
    /// one below — an unobserved resource may well be free — and the two were
    /// one number until this split, which is the same conflation the wake-side
    /// [`GrantBlock::Unobserved`] / [`GrantBlock::Busy`] pair already avoids.
    ScanUnavailableUnobserved,
    /// Sub-count of [`GrantBlock::ScanUnavailable`]: every named resource has
    /// been observed and at least one is genuinely held.
    ScanUnavailableBusy,
    /// Scan side: the waiter is pinned to a blocker whose serial has not moved.
    ScanBlocker,
    /// Scan side: a preparation intent found no free slot in the preparation
    /// token pool.
    ScanPreparationPool,
    /// Scan side: an authoritative scan revoked an in-flight GRANTED record.
    /// The grant is destroyed before its callback can commit, and whether that
    /// costs a `grants_lost` depends only on whether the owner had already
    /// woken.
    ScanRevoke,
    /// Sub-count of [`GrantBlock::ScanRevoke`]: the revoked grant's claim
    /// conflicts the live predecessor prefix. Only an earlier ticket's
    /// fence-preserving claim enters that prefix, so this names senior work
    /// covering a footprint a junior grant already held.
    ScanRevokeConflict,
    /// Sub-count of [`GrantBlock::ScanRevokeConflict`] on the permit axis.
    ScanRevokeConflictPermit,
    /// Sub-count of [`GrantBlock::ScanRevokeConflict`] on the LLC axis.
    ScanRevokeConflictLlc,
    /// Sub-count of [`GrantBlock::ScanRevoke`]: a resource the grant names is
    /// not observed available. Publishing a blocker witness and releasing a
    /// claim whose callback lost its race both drop the named resources out of
    /// availability until the next observation, so this can fire with no queue
    /// conflict at all.
    ScanRevokeUnavailable,
    /// Owner side: the revocation landed before this ticket's wake read
    /// GRANTED, so it acknowledged REVOKED from an ordinary state read and
    /// never entered the grant callback. This is the silent term — the grant
    /// was issued and destroyed without ever reaching `grants_lost`.
    RevokedUnwoken,
    /// Owner side: the changed-claims fence parked the granted entrant before
    /// its probe ran.
    ParkEntry,
    /// Owner side: the changed-claims fence parked the granted entrant after
    /// its probe ran, discarding that work.
    ParkCommit,
    /// Sub-count of [`GrantBlock::ParkEntry`] and [`GrantBlock::ParkCommit`]:
    /// the accumulator was saturated, so the park is the fail-closed blanket
    /// one rather than a proven claim overlap.
    ParkSaturated,
    /// Owner side: a licensed grant converted into a parked PENDING
    /// preparation. By design it never reaches HELD under this grant — the
    /// later one-shot activation publishes HELD without one — so it is neither
    /// conversion nor loss. The coordinator's own preparation finish carries no
    /// grant and is deliberately not counted.
    Prepared,
}

impl GrantBlock {
    const COUNT: usize = 27;

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
        Self::ScanUnavailableUnobserved,
        Self::ScanUnavailableBusy,
        Self::ScanBlocker,
        Self::ScanPreparationPool,
        Self::ScanRevoke,
        Self::ScanRevokeConflict,
        Self::ScanRevokeConflictPermit,
        Self::ScanRevokeConflictLlc,
        Self::ScanRevokeUnavailable,
        Self::RevokedUnwoken,
        Self::ParkEntry,
        Self::ParkCommit,
        Self::ParkSaturated,
        Self::Prepared,
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
            Self::ScanUnavailableUnobserved => "scan_unavailable_unobserved",
            Self::ScanUnavailableBusy => "scan_unavailable_busy",
            Self::ScanBlocker => "scan_blocker",
            Self::ScanPreparationPool => "scan_preparation_pool",
            Self::ScanRevoke => "scan_revoke",
            Self::ScanRevokeConflict => "scan_revoke_conflict",
            Self::ScanRevokeConflictPermit => "scan_revoke_conflict_permit",
            Self::ScanRevokeConflictLlc => "scan_revoke_conflict_llc",
            Self::ScanRevokeUnavailable => "scan_revoke_unavailable",
            Self::RevokedUnwoken => "revoked_unwoken",
            Self::ParkEntry => "park_entry",
            Self::ParkCommit => "park_commit",
            Self::ParkSaturated => "park_saturated",
            Self::Prepared => "prepared",
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
pub(crate) fn note_held_in_flight(
    held_records: u64,
    granted_records: u64,
    distinct_cpus: u64,
    now_ns: u64,
) {
    bump_max(&HELD_IN_FLIGHT_MAX, held_records);
    bump_max(&GRANTED_IN_FLIGHT_MAX, granted_records);
    bump_max(&DISTINCT_HELD_CPUS_MAX, distinct_cpus);
    // Mean in-flight depth over the sampled span, against `grants_issued`, is
    // the only available reading of how long an issued grant survives before
    // it converts or dies: Little's law closes `residency = mean_granted /
    // (grants_issued / span)`. The peak alone cannot separate "many grants,
    // each resolved fast" from "few grants, each stuck for seconds", and that
    // distinction decides whether the fence a grant holds is a scan-cadence
    // problem or a wake-path one.
    HELD_IN_FLIGHT_SUM.fetch_add(held_records, Ordering::Relaxed);
    GRANTED_IN_FLIGHT_SUM.fetch_add(granted_records, Ordering::Relaxed);
    // The running population, differenced per sample rather than between the
    // two means, so the max is a real simultaneous observation instead of the
    // gap between two peaks that never coincided. `held_records` counts the
    // GRANTED subset too, so this is the HELD-only remainder: the cells that
    // are running a VM rather than fencing the host while they wait to be
    // woken. Saturating because the two inputs are counted from one snapshot
    // but nothing in the type system pins their order.
    let running = held_records.saturating_sub(granted_records);
    RUNNING_IN_FLIGHT_SUM.fetch_add(running, Ordering::Relaxed);
    bump_max(&RUNNING_IN_FLIGHT_MAX, running);
    IN_FLIGHT_SAMPLES.fetch_add(1, Ordering::Relaxed);
    let _ = IN_FLIGHT_FIRST_NS.compare_exchange(0, now_ns, Ordering::Relaxed, Ordering::Relaxed);
    bump_max(&IN_FLIGHT_LAST_NS, now_ns);
    ensure_atexit();
}

/// Wall time one licensed grant's callback spent between the owner observing
/// GRANTED and its terminal outcome. Subtracted from the residency above it
/// leaves the scan-publish-to-owner-wake latency, which is what says whether
/// the coordinator rescans faster than owners can answer.
pub(crate) fn note_grant_callback(elapsed_ns: u64) {
    if !enabled() {
        return;
    }
    CALLBACK_NS_SUM.fetch_add(elapsed_ns, Ordering::Relaxed);
    bump_max(&CALLBACK_NS_MAX, elapsed_ns);
    CALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
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

/// One authoritative scan's view of the changed-claims accumulator, sampled
/// immediately before the scan resets it and only behind [`enabled`]. The bit
/// counts are the accumulated coverage on the CPU, folded-permit and LLC axes;
/// `watermark_span` is how far below the highest live ticket the suffix
/// watermark sits, and is `None` when no dirty event armed it at all. Together
/// they say whether the accumulator narrows the granted-entrant park in
/// practice or has grown to cover the host by the time a scan clears it.
pub(crate) fn note_changed_claims(
    saturated: bool,
    cpu_bits: u64,
    permit_bits: u64,
    llc_bits: u64,
    watermark_span: Option<u64>,
) {
    CHANGED_SCANS.fetch_add(1, Ordering::Relaxed);
    if saturated {
        CHANGED_SATURATED_SCANS.fetch_add(1, Ordering::Relaxed);
    }
    CHANGED_CPU_BITS_SUM.fetch_add(cpu_bits, Ordering::Relaxed);
    bump_max(&CHANGED_CPU_BITS_MAX, cpu_bits);
    CHANGED_PERMIT_BITS_SUM.fetch_add(permit_bits, Ordering::Relaxed);
    bump_max(&CHANGED_PERMIT_BITS_MAX, permit_bits);
    CHANGED_LLC_BITS_SUM.fetch_add(llc_bits, Ordering::Relaxed);
    bump_max(&CHANGED_LLC_BITS_MAX, llc_bits);
    if let Some(span) = watermark_span {
        CHANGED_DIRTY_SCANS.fetch_add(1, Ordering::Relaxed);
        bump_max(&CHANGED_WATERMARK_SPAN_MAX, span);
    }
    ensure_atexit();
}

fn format_line(pid: u32) -> String {
    let discover_count = DISCOVER_COUNT.load(Ordering::Relaxed);
    let discover_ns_mean = DISCOVER_NS_SUM
        .load(Ordering::Relaxed)
        .checked_div(discover_count)
        .unwrap_or(0);
    let changed_scans = CHANGED_SCANS.load(Ordering::Relaxed);
    let changed_mean = |cell: &AtomicU64| {
        cell.load(Ordering::Relaxed)
            .checked_div(changed_scans)
            .unwrap_or(0)
    };
    let changed_cpu_bits_mean = changed_mean(&CHANGED_CPU_BITS_SUM);
    let changed_permit_bits_mean = changed_mean(&CHANGED_PERMIT_BITS_SUM);
    let changed_llc_bits_mean = changed_mean(&CHANGED_LLC_BITS_SUM);
    let in_flight_samples = IN_FLIGHT_SAMPLES.load(Ordering::Relaxed);
    let in_flight_mean = |cell: &AtomicU64| {
        cell.load(Ordering::Relaxed)
            .checked_div(in_flight_samples)
            .unwrap_or(0)
    };
    let held_in_flight_mean = in_flight_mean(&HELD_IN_FLIGHT_SUM);
    let granted_in_flight_mean = in_flight_mean(&GRANTED_IN_FLIGHT_SUM);
    let running_in_flight_mean = in_flight_mean(&RUNNING_IN_FLIGHT_SUM);
    // Read from the coordinator's own scan counters rather than accumulated
    // here, so this line and `coordinator-wakes-<pid>.txt` cannot disagree
    // about the denominator every `scan_*` count below has to be divided by.
    let (grant_scans, records_scanned) =
        crate::vmm::host_topology::protocol::coordinator_scan_stats();
    let in_flight_span_ns = IN_FLIGHT_LAST_NS
        .load(Ordering::Relaxed)
        .saturating_sub(IN_FLIGHT_FIRST_NS.load(Ordering::Relaxed));
    let callback_count = CALLBACK_COUNT.load(Ordering::Relaxed);
    let callback_ns_mean = CALLBACK_NS_SUM
        .load(Ordering::Relaxed)
        .checked_div(callback_count)
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
         discover_count={} discover_ns_mean={discover_ns_mean} discover_ns_max={} \
         changed_scans={changed_scans} changed_dirty_scans={} \
         changed_saturated_scans={} changed_cpu_bits_mean={changed_cpu_bits_mean} \
         changed_cpu_bits_max={} changed_permit_bits_mean={changed_permit_bits_mean} \
         changed_permit_bits_max={} changed_llc_bits_mean={changed_llc_bits_mean} \
         changed_llc_bits_max={} changed_watermark_span_max={} \
         held_in_flight_mean={held_in_flight_mean} \
         granted_in_flight_mean={granted_in_flight_mean} \
         in_flight_samples={in_flight_samples} in_flight_span_ns={in_flight_span_ns} \
         callback_count={callback_count} callback_ns_mean={callback_ns_mean} \
         callback_ns_max={} discover_ns_sum={} changed_cpu_bits_sum={} \
         changed_permit_bits_sum={} changed_llc_bits_sum={} held_in_flight_sum={} \
         granted_in_flight_sum={} callback_ns_sum={} \
         running_in_flight_mean={running_in_flight_mean} \
         running_in_flight_sum={} running_in_flight_max={} \
         grant_scans={grant_scans} records_scanned={records_scanned}{blocks}\n",
        GRANTS_ISSUED.load(Ordering::Relaxed),
        GRANTS_REACHED_HELD.load(Ordering::Relaxed),
        GRANTS_LOST.load(Ordering::Relaxed),
        HELD_IN_FLIGHT_MAX.load(Ordering::Relaxed),
        GRANTED_IN_FLIGHT_MAX.load(Ordering::Relaxed),
        DISTINCT_HELD_CPUS_MAX.load(Ordering::Relaxed),
        discover_count,
        DISCOVER_NS_MAX.load(Ordering::Relaxed),
        CHANGED_DIRTY_SCANS.load(Ordering::Relaxed),
        CHANGED_SATURATED_SCANS.load(Ordering::Relaxed),
        CHANGED_CPU_BITS_MAX.load(Ordering::Relaxed),
        CHANGED_PERMIT_BITS_MAX.load(Ordering::Relaxed),
        CHANGED_LLC_BITS_MAX.load(Ordering::Relaxed),
        CHANGED_WATERMARK_SPAN_MAX.load(Ordering::Relaxed),
        CALLBACK_NS_MAX.load(Ordering::Relaxed),
        DISCOVER_NS_SUM.load(Ordering::Relaxed),
        CHANGED_CPU_BITS_SUM.load(Ordering::Relaxed),
        CHANGED_PERMIT_BITS_SUM.load(Ordering::Relaxed),
        CHANGED_LLC_BITS_SUM.load(Ordering::Relaxed),
        HELD_IN_FLIGHT_SUM.load(Ordering::Relaxed),
        GRANTED_IN_FLIGHT_SUM.load(Ordering::Relaxed),
        CALLBACK_NS_SUM.load(Ordering::Relaxed),
        RUNNING_IN_FLIGHT_SUM.load(Ordering::Relaxed),
        RUNNING_IN_FLIGHT_MAX.load(Ordering::Relaxed),
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
            "discover_count=",
            "discover_ns_mean=",
            "discover_ns_max=",
            "changed_scans=",
            "changed_dirty_scans=",
            "changed_saturated_scans=",
            "changed_cpu_bits_mean=",
            "changed_cpu_bits_max=",
            "changed_permit_bits_mean=",
            "changed_permit_bits_max=",
            "changed_llc_bits_mean=",
            "changed_llc_bits_max=",
            "changed_watermark_span_max=",
            "held_in_flight_mean=",
            "granted_in_flight_mean=",
            "in_flight_samples=",
            "in_flight_span_ns=",
            "callback_count=",
            "callback_ns_mean=",
            "callback_ns_max=",
            "discover_ns_sum=",
            "changed_cpu_bits_sum=",
            "changed_permit_bits_sum=",
            "changed_llc_bits_sum=",
            "held_in_flight_sum=",
            "granted_in_flight_sum=",
            "callback_ns_sum=",
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
            "block_scan_unavailable_unobserved=",
            "block_scan_unavailable_busy=",
            "block_scan_blocker=",
            "block_scan_preparation_pool=",
            "block_scan_revoke=",
            "block_scan_revoke_conflict=",
            "block_scan_revoke_conflict_permit=",
            "block_scan_revoke_conflict_llc=",
            "block_scan_revoke_unavailable=",
            "block_revoked_unwoken=",
            "block_park_entry=",
            "block_park_commit=",
            "block_park_saturated=",
            "block_prepared=",
            "running_in_flight_mean=",
            "running_in_flight_sum=",
            "running_in_flight_max=",
            "grant_scans=",
            "records_scanned=",
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

    /// The running population is the HELD-only remainder, differenced per
    /// sample. Differencing the two published means instead would report a max
    /// that is the gap between two peaks which may never have coincided, and
    /// would underflow rather than clamp if a snapshot ever counted the GRANTED
    /// subset above its own total.
    #[test]
    fn running_in_flight_is_the_held_only_remainder_per_sample() {
        let before_sum = RUNNING_IN_FLIGHT_SUM.load(Ordering::Relaxed);
        let before_max = RUNNING_IN_FLIGHT_MAX.load(Ordering::Relaxed);
        note_held_in_flight(11, 7, 0, 1);
        note_held_in_flight(9, 6, 0, 2);
        note_held_in_flight(5, 9, 0, 3);
        assert_eq!(
            RUNNING_IN_FLIGHT_SUM.load(Ordering::Relaxed) - before_sum,
            7,
            "each sample contributes held minus granted, clamped at zero",
        );
        assert_eq!(
            RUNNING_IN_FLIGHT_MAX
                .load(Ordering::Relaxed)
                .max(before_max),
            before_max.max(4),
            "the max is a single sample's remainder, not the gap between peaks",
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
