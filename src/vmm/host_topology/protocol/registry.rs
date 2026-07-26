//! Fixed-record, crash-recoverable admission registry.
//!
//! Arrival, claim replacement, and normal removal touch one fixed record plus
//! the affected aggregate counters; they never rewrite the other waiters.
//! Records live in fixed-size chunk files which grow to the concurrency
//! high-water mark and are never truncated while another process may still
//! have a futex word mapped.

#[cfg(test)]
use super::CpuExContentionSharedWake;
use super::{
    AdmissionClass, ClaimMode, ClaimSet, ContentionEvidence, ContentionMarker, ContentionSet,
    ResourceKey, interrupted, protocol_dir,
};
use crate::flock::{FlockMode, InterruptibleFlockWaiter, block_flock, try_flock};
use anyhow::{Context, Result};
use memmap2::{Mmap, MmapMut};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::hash::{BuildHasher, Hasher};
use std::os::fd::OwnedFd;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

const MAGIC: u64 = u64::from_be_bytes(*b"KTSTRQ26");
const VERSION: u32 = 26;
#[cfg(test)]
const RETAINED_FUTEX_WAIT_MARKER_ENV: &str = "KTSTR_TEST_RETAINED_FUTEX_WAIT_MARKER";
#[cfg(test)]
const RETAINED_FUTEX_WAIT_GATE_ENV: &str = "KTSTR_TEST_RETAINED_FUTEX_WAIT_GATE";
#[cfg(test)]
thread_local! {
    static GENERATION_WAIT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
const HEADER_FIXED: usize = 272;
const HEADER_ALIGN: usize = 4096;
const RECORD_FIXED: usize = 192;
const RECORD_ALIGN: usize = 64;
const RECORDS_PER_CHUNK: usize = 64;
const NONE_SLOT: u64 = u64::MAX;
const MAX_RESOURCE_BITS: usize = 1 << 20;
const MAX_REGISTRY_SLOTS: u64 = 1 << 16;
const INITIALIZER_PREFIX: &str = ".ktstr-acquire-registry-v26-init-";
const LIVENESS_PREFIX: &str = "ktstr-acquire-v26-slot-";
const LIVENESS_SEPARATOR: &str = "-ticket-";
const LIVENESS_SUFFIX: &str = ".live";
const WAIT_DIAGNOSTIC_BUCKET_SECS: u64 = 30;
const WAIT_DIAGNOSTIC_RING_SLOTS: u64 = 8;
const WAIT_DIAGNOSTIC_MAX_RECORDS: usize = 64;
const WAIT_DIAGNOSTIC_MAX_BYTES: usize = 128 * 1024;

/// Stable identity of one admission protocol instance.
///
/// Test lock prefixes are thread-local, while tickets and held publications
/// are intentionally `Send`.  Resolving paths again on a different thread can
/// therefore redirect an existing owner into an unrelated registry (including
/// the production registry).  Capture the protocol root once, when ownership
/// is created or imported, and scope every later operation to that root.
#[derive(Clone, Debug)]
struct RegistryNamespace {
    protocol_dir: PathBuf,
    #[cfg(test)]
    llc_lock_prefix: Option<String>,
    #[cfg(test)]
    cpu_lock_prefix: Option<String>,
}

thread_local! {
    static ACTIVE_REGISTRY_NAMESPACE: std::cell::RefCell<Option<PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

pub(super) struct RegistryNamespaceGuard {
    previous: Option<PathBuf>,
    #[cfg(test)]
    previous_llc_lock_prefix: Option<String>,
    #[cfg(test)]
    previous_cpu_lock_prefix: Option<String>,
}

impl RegistryNamespace {
    fn resolve() -> Self {
        Self {
            protocol_dir: protocol_dir(),
            #[cfg(test)]
            llc_lock_prefix: super::super::LLC_LOCK_PREFIX_OVERRIDE
                .with(|prefix| prefix.borrow().clone()),
            #[cfg(test)]
            cpu_lock_prefix: super::super::CPU_LOCK_PREFIX_OVERRIDE
                .with(|prefix| prefix.borrow().clone()),
        }
    }

    fn enter(&self) -> RegistryNamespaceGuard {
        let previous = ACTIVE_REGISTRY_NAMESPACE
            .with(|active| active.replace(Some(self.protocol_dir.clone())));
        #[cfg(test)]
        let previous_llc_lock_prefix = super::super::LLC_LOCK_PREFIX_OVERRIDE
            .with(|prefix| prefix.replace(self.llc_lock_prefix.clone()));
        #[cfg(test)]
        let previous_cpu_lock_prefix = super::super::CPU_LOCK_PREFIX_OVERRIDE
            .with(|prefix| prefix.replace(self.cpu_lock_prefix.clone()));
        RegistryNamespaceGuard {
            previous,
            #[cfg(test)]
            previous_llc_lock_prefix,
            #[cfg(test)]
            previous_cpu_lock_prefix,
        }
    }
}

impl Drop for RegistryNamespaceGuard {
    fn drop(&mut self) {
        #[cfg(test)]
        super::super::LLC_LOCK_PREFIX_OVERRIDE.with(|prefix| {
            prefix.replace(self.previous_llc_lock_prefix.take());
        });
        #[cfg(test)]
        super::super::CPU_LOCK_PREFIX_OVERRIDE.with(|prefix| {
            prefix.replace(self.previous_cpu_lock_prefix.take());
        });
        ACTIVE_REGISTRY_NAMESPACE.with(|active| {
            active.replace(self.previous.take());
        });
    }
}

fn active_protocol_dir() -> PathBuf {
    ACTIVE_REGISTRY_NAMESPACE
        .with(|active| active.borrow().clone())
        .unwrap_or_else(protocol_dir)
}

const H_MAGIC: usize = 0;
const H_VERSION: usize = 8;
const H_WORDS: usize = 12;
const H_RECORD_SIZE: usize = 16;
const H_RECORDS_PER_CHUNK: usize = 20;
const H_NEXT_TICKET: usize = 24;
const H_GENERATION: usize = 32;
const H_COORDINATOR: usize = 40;
const H_NEXT_SLOT: usize = 48;
const H_FREE_HEAD: usize = 56;
const H_GLOBAL_SERIAL: usize = 64;
const H_AGGREGATE_DIRTY: usize = 72;
const H_LIVENESS_SWEEP: usize = 80;
const H_LAST_LIVENESS_SWEEP_NS: usize = 88;
const H_CLAIM_EPOCH: usize = 96;
const H_MIN_CHANGED_TICKET: usize = 104;
const H_COORDINATOR_SLOT: usize = 112;
const H_PENDING_FLAGS: usize = 120;
const H_OBSERVATION_REQUEST: usize = 128;
const H_GRANT_SCANS: usize = 136;
const H_ACTIVE_HEAD: usize = 144;
const H_ACTIVE_TAIL: usize = 152;
const H_LIVENESS_RECONCILE_BY_NS: usize = 160;
const H_COORDINATOR_EPOCH: usize = 168;
const H_COORDINATOR_HEARTBEAT_NS: usize = 176;
const H_LAST_PROGRESS_NS: usize = 184;
/// Futex sequence paired with `H_GENERATION`. The generation remains a full
/// u64 diagnostic/epoch while waiters sleep on this non-overlapping u32 word.
const H_GENERATION_WAKE: usize = 192;
/// Ticket which most recently received a speculative REPLAN callback. New
/// callback publications start after this ticket and wrap once, so a stream
/// of changing early tickets cannot monopolize the bounded planner window.
const H_REPLAN_CURSOR: usize = 200;
/// Inclusive ticket high-water admitted under the current non-renewing REPLAN
/// lease. Each registry-writer scan observes a finite active-list snapshot,
/// but later scans may extend this diagnostic high-water while older callbacks
/// remain live. It is not an admission horizon.
const H_REPLAN_HORIZON: usize = 208;
/// Number of published speculative callbacks which have not yet returned to
/// WAITING. This is exact record-state/quarantine accounting only: each
/// callback completion independently publishes a coalesced coordinator rescan
/// edge, and later callbacks may join without waiting for this count to drain.
const H_REPLAN_OUTSTANDING: usize = 216;
/// Monotonic start and deadline of the current REPLAN lease. The first live
/// callback arms it; incremental publications never renew it. Every callback
/// still live at the original deadline is quarantined so continuous arrivals
/// cannot keep a stopped planner alive forever.
const H_REPLAN_WAVE_STARTED_NS: usize = 224;
const H_REPLAN_WAVE_DEADLINE_NS: usize = 232;
/// Immutable maximum number of speculative planner callbacks which this host
/// may execute concurrently. Actual VM admission remains independently
/// bounded by the exact CPU/memory/token claims: this limits only parallel
/// placement computation, whose callbacks otherwise form a process herd.
const H_REPLAN_CAPACITY: usize = 240;
/// Absolute monotonic deadline for flushing a coalesced REPLAN rescan. Unlike
/// the coordinator heartbeat, ordinary event turns never renew this value, so
/// a continuous inotify stream cannot postpone a partial planning wave.
const H_DEFERRED_RESCAN_DEADLINE_NS: usize = 248;
/// Suffix watermark consumed only by unlicensed REPLAN acquire-commits
/// (`dirty_redesignation`). `mark_suffix_dirty` writes it together with
/// [`H_MIN_CHANGED_TICKET`]; the grant-disjointness damping guard writes only
/// this word, so a replacement provably unable to doom any in-flight grant
/// still fences speculative REPLAN commits whose claims carry no grant charge.
const H_MIN_CHANGED_TICKET_REPLAN: usize = 256;
/// Whether the changed-claims accumulator lost precision since the last
/// authoritative scan: a suffix dirty arrived through the conservative
/// conservative `Table::mark_suffix_dirty` entry point (test-only today)
/// with no claim analysis, so every
/// later in-flight grant must park (today's behavior). Reset with the
/// watermark at `finish_claim_scan`. Fail-closed: any writer that cannot
/// classify its change saturates rather than under-parks.
const H_CHANGED_CLAIMS_SATURATED: usize = 264;
const _: () = assert!(H_CHANGED_CLAIMS_SATURATED + std::mem::size_of::<u64>() == HEADER_FIXED);
const _: () = assert!(H_AGGREGATE_DIRTY.is_multiple_of(std::mem::align_of::<AtomicU64>()));
const _: () = assert!(H_GENERATION_WAKE.is_multiple_of(std::mem::align_of::<AtomicU32>()));
const _: () = assert!(H_REPLAN_CURSOR.is_multiple_of(std::mem::align_of::<u64>()));
const _: () = assert!(H_REPLAN_HORIZON.is_multiple_of(std::mem::align_of::<u64>()));
const _: () = assert!(H_REPLAN_OUTSTANDING.is_multiple_of(std::mem::align_of::<u64>()));
const _: () = assert!(H_REPLAN_WAVE_STARTED_NS.is_multiple_of(std::mem::align_of::<u64>()));
const _: () = assert!(H_REPLAN_WAVE_DEADLINE_NS.is_multiple_of(std::mem::align_of::<u64>()));
const _: () = assert!(H_REPLAN_CAPACITY.is_multiple_of(std::mem::align_of::<u64>()));
const _: () = assert!(H_DEFERRED_RESCAN_DEADLINE_NS.is_multiple_of(std::mem::align_of::<u64>()));

const R_STATE: usize = 0;
const R_WAKE: usize = 4;
const R_TICKET: usize = 8;
const R_PID: usize = 16;
const R_WATCH_LLC_MODE: usize = 20;
const R_BLOCKED_SERIAL: usize = 24;
const R_NEXT_FREE: usize = 32;
const R_GRANT_EPOCH: usize = 40;
const R_BLOCK_KIND: usize = 48;
const R_BLOCK_MODE: usize = 52;
const R_BLOCK_INDEX: usize = 56;
/// Resource-improvement serial covered by this record's published prefix and
/// availability snapshot. For WAITING records it is an upper bound consumed
/// by the last completed callback; for GRANTED/REPLAN records it is the
/// callback issuance serial. GRANTED covers only its exact designated claim,
/// while REPLAN covers the complete alternative watch. This distinction lets
/// a relevant newer issuance invalidate an in-flight callback without turning
/// every unrelated alternative release into a global callback revocation.
const R_ISSUE_SERIAL: usize = 64;
const R_REPLAN_CLAIM_EPOCH: usize = 72;
const R_PREV_ACTIVE: usize = 80;
const R_NEXT_ACTIVE: usize = 88;
const R_CLAIM_LLC_MODE: usize = 96;
const R_CLAIM_CPU_MODE: usize = 100;
const R_WATCH_CPU_MODE: usize = 104;
const R_CLAIM_PERMIT_MODE: usize = 108;
/// Publication stamp for the four derived predecessor bitsets. Zero means a
/// writer died before publishing a complete snapshot.
const R_PREFIX_EPOCH: usize = 112;
const R_WATCH_PERMIT_MODE: usize = 120;
const R_CLAIM_CLASS: usize = 124;
const R_WATCH_CLASS: usize = 128;
/// Maximum cooperative CPU-permit units which may be outstanding as backfill
/// while this exact ticket is the oldest physically unavailable candidate.
/// Completed bypass work is subtracted from the live scan, not permanently
/// debited from this capacity.
const R_BACKFILL_CAPACITY: usize = 132;
/// Monotonic timestamp at which this claim first became the oldest physically
/// unavailable admission candidate. Reclaiming completed backfill keeps the
/// host busy only for a bounded interval; after this age expires, new
/// conflicting grants stop and the outstanding wave drains for the head.
const R_BACKFILL_STARTED_NS: usize = 136;
/// Compact, transactionally published scan metadata. Grant scans validate
/// these bounds and identities without rereading every word of the immutable
/// watch or the zero tail of each sparse exact claim. Full record decode and
/// dirty repair recompute the metadata from the authoritative bitmaps.
const R_SCAN_FLAGS: usize = 144;
const R_WATCH_CPU_COUNT: usize = 148;
const R_WATCH_LLC_COUNT: usize = 152;
const R_WATCH_PERMIT_COUNT: usize = 156;
const R_WATCH_COOPERATIVE_PERMIT_COUNT: usize = 160;
const R_CLAIM_CPU_WORD_START: usize = 164;
const R_CLAIM_CPU_WORD_END: usize = 166;
const R_CLAIM_LLC_WORD_START: usize = 168;
const R_CLAIM_LLC_WORD_END: usize = 170;
const R_CLAIM_PERMIT_WORD_START: usize = 172;
const R_CLAIM_PERMIT_WORD_END: usize = 174;
const R_WATCH_IDENTITY: usize = 176;
const R_CLAIM_IDENTITY: usize = 184;
const R_BITS: usize = RECORD_FIXED;

const SCAN_FLAG_FLEXIBLE: u32 = 1 << 0;
const SCAN_FLAG_WATCH_EMPTY: u32 = 1 << 1;
// A preparation intent will consume one scarce preparation-slot token from the
// pool when granted, before it can physically prepare. Run reservations that
// share this registry do not. The bit is a pure function of the watch: a
// preparation intent registers with the token pool unioned into its watch (so
// it also wakes on any token release), while a run reservation never watches
// the token sub-range. `for_claims` derives it and `validate_full` re-derives
// and agrees, so it survives claim rewrites (REPLAN) because the watch is
// immutable across them.
const SCAN_FLAG_PREPARATION_INTENT: u32 = 1 << 2;
const SCAN_FLAGS_VALID: u32 =
    SCAN_FLAG_FLEXIBLE | SCAN_FLAG_WATCH_EMPTY | SCAN_FLAG_PREPARATION_INTENT;
const _: () = assert!(R_WATCH_IDENTITY.is_multiple_of(std::mem::align_of::<u64>()));
const _: () = assert!(R_CLAIM_IDENTITY.is_multiple_of(std::mem::align_of::<u64>()));
const _: () = assert!(R_CLAIM_IDENTITY + std::mem::size_of::<u64>() == R_BITS);

const RB_CLAIM_CPUS: usize = 0;
const RB_CLAIM_LLCS: usize = 1;
const RB_CLAIM_PERMITS: usize = 2;
const RB_WATCH_CPUS: usize = 3;
const RB_WATCH_LLCS: usize = 4;
const RB_WATCH_PERMITS: usize = 5;
const RB_PREFIX_CPU_ANY: usize = 6;
const RB_PREFIX_CPU_EXCLUSIVE: usize = 7;
const RB_PREFIX_LLC_ANY: usize = 8;
const RB_PREFIX_LLC_EXCLUSIVE: usize = 9;
const RECORD_BITMAPS: usize = 10;

const STATE_FREE: u32 = 0;
const STATE_WAITING: u32 = 1;
const STATE_GRANTED: u32 = 2;
const STATE_COORDINATOR: u32 = 3;
const STATE_REPLAN: u32 = 4;
const STATE_HELD: u32 = 5;
/// A live coordinator whose bounded progress lease was transferred to a
/// successor. Its logical reservation is parked until it is elected again;
/// physical flock modes remain the final compatibility fence.
const STATE_COORDINATOR_STANDBY: u32 = 6;
/// A same-PID pre-exec arrival marker. Pending records claim the bounded
/// preparation CPU/memory/token/physical-CPU footprint and retain the selected
/// final intent in their watch, but remain ineligible for coordinator election.
/// Activation atomically replaces both with one complete ready claim/watch
/// after immutable VM artifacts have been prepared.
const STATE_PENDING: u32 = 7;
/// A grant revoked by an authoritative scan while its callback may already be
/// probing outside the registry fence. The old exact claim remains a prefix
/// fence until that ticket acknowledges the revocation, then acknowledgement
/// atomically publishes WAITING + PENDING_RESCAN.
const STATE_REVOKED: u32 = 8;
/// A speculative callback which outlived its finite wave lease. Expiration
/// drains the wave without converting the stale callback into an ordinary
/// waiter. Its owner must acknowledge this state before the ticket may be
/// scanned again, and the old callback publication is rejected on return.
const STATE_REPLAN_EXPIRED: u32 = 9;

/// Whether a record in `state` charges its exact claim into the `C_GRANT_*`
/// planner-occupancy families. REVOKED stays charged: a revoked callback's
/// exact claim fences the scan until acknowledgement and its physical OFDs
/// may still be live, so releasing at the revoke flip would re-blind the
/// planner to a genuinely occupied footprint for the revoke -> ack window.
fn charged_state(state: u32) -> bool {
    matches!(state, STATE_GRANTED | STATE_REVOKED)
}

const BLOCK_NONE: u32 = 0;
const BLOCK_CPU: u32 = 1;
const BLOCK_LLC: u32 = 2;
const BLOCK_PERMIT: u32 = 3;

const PENDING_RESCAN: u64 = 1 << 0;
const PENDING_OBSERVATION: u64 = 1 << 1;
/// A rescan publication which is safe to coalesce while speculative planners
/// from the current wave are live. Urgent structural/resource work remains in
/// `PENDING_RESCAN` and is never delayed by this flag.
const PENDING_REPLAN_RESCAN: u64 = 1 << 2;

const B_CLAIM_CPUS: usize = 0;
const B_CLAIM_CPU_EXCLUSIVE: usize = 1;
const B_CLAIM_LLC_ANY: usize = 2;
const B_CLAIM_LLC_EXCLUSIVE: usize = 3;
const B_WATCH_CPUS: usize = 4;
const B_WATCH_CPU_EXCLUSIVE: usize = 5;
const B_WATCH_LLCS: usize = 6;
const B_WATCH_LLC_EXCLUSIVE: usize = 7;
const B_HELD_CPU_SHARED: usize = 8;
const B_HELD_CPU_EXCLUSIVE: usize = 9;
const B_HELD_LLC_SHARED: usize = 10;
const B_HELD_LLC_EXCLUSIVE: usize = 11;
const B_CPU_KNOWN: usize = 12;
const B_CPU_SH_AVAILABLE: usize = 13;
const B_CPU_EX_AVAILABLE: usize = 14;
const B_LLC_KNOWN: usize = 15;
const B_LLC_SH_AVAILABLE: usize = 16;
const B_LLC_EX_AVAILABLE: usize = 17;
const B_PENDING_CPU_SH: usize = 18;
const B_PENDING_CPU_EX: usize = 19;
const B_PENDING_LLC_SH: usize = 20;
const B_PENDING_LLC_EX: usize = 21;
const B_CANDIDATE_CPU_SH: usize = 22;
const B_CANDIDATE_CPU_EX: usize = 23;
const B_CANDIDATE_LLC_SH: usize = 24;
const B_CANDIDATE_LLC_EX: usize = 25;
/// Since-last-scan changed-claims accumulator: the union of every claim that
/// became newly fenceable while dirtying the main suffix watermark
/// (replacements, promotions with a claim change, preparation transitions,
/// completion/ack publications the pending scan can turn into fences).
/// Grow-only between scans — reset together with the watermark at
/// `finish_claim_scan` — so plain bitset OR suffices, no reference counts.
/// A GRANTED entrant behind the watermark parks only when its claim
/// overlaps this set (mode matrix as `first_conflict`); pure fence
/// retirements (removals, revoke acks) dirty the watermark without joining
/// the set because a shrinking prefix cannot doom an in-flight grant.
const B_CHANGED_CPU_ANY: usize = 26;
const B_CHANGED_CPU_EX: usize = 27;
const B_CHANGED_LLC_ANY: usize = 28;
const B_CHANGED_LLC_EX: usize = 29;
const HEADER_BITMAPS: usize = 30;
/// Per-CPU count of live Build-class exact claims. Preparation placement reads
/// this aggregate once and prefers its complement, keeping immutable-image
/// work off the physical CPUs currently licensed to Cargo/kernel builds while
/// still allowing cooperative overlap when the complement is exhausted.
const C_BUILD_CLAIM_CPUS: usize = 12;
/// Count-only planner-charge families for in-flight grants: the exact claims
/// of every GRANTED and REVOKED record, folded like `adjust_held_counts`
/// (permits land at `permit_resource_index` in the CPU families). These are
/// planner *bias* inputs only — they must never feed `conflicts()`,
/// availability, `exclusive_held`, or the holder counts, all of which stay
/// HELD-only so a granted claim can never become an absolute planner fence.
const C_GRANT_CPU_ANY: usize = 13;
const C_GRANT_CPU_EX: usize = 14;
const C_GRANT_LLC_ANY: usize = 15;
const C_GRANT_LLC_EX: usize = 16;
const AGGREGATE_BITMAPS: usize = 17;

const S_CPU_SH: usize = 0;
const S_CPU_EX: usize = 1;
const S_LLC_SH: usize = 2;
const S_LLC_EX: usize = 3;
const SERIAL_ARRAYS: usize = 4;
const Q_CPU_SH: usize = 0;
const Q_CPU_EX: usize = 1;
const Q_LLC_SH: usize = 2;
const Q_LLC_EX: usize = 3;
const REQUEST_ARRAYS: usize = 4;
const LIVENESS_SWEEP_INTERVAL_NS: u64 = 30_000_000_000;
/// Minimum spacing between liveness sweeps triggered by coordinator-session
/// reconcile requests. See [`Table::request_liveness_reconciliation`].
const LIVENESS_RECONCILE_MIN_INTERVAL_NS: u64 = 5_000_000_000;
/// The elected coordinator alone wakes at this cadence and renews one header
/// word. No planner, observation, grant scan, or liveness sweep is coupled to
/// this tick.
const COORDINATOR_HEARTBEAT_INTERVAL_NS: u64 = 1_000_000_000;
pub(super) const COORDINATOR_HEARTBEAT_INTERVAL: Duration =
    Duration::from_nanos(COORDINATOR_HEARTBEAT_INTERVAL_NS);
/// Backoff between nonblocking shared-read attempts when the writer-intent
/// sidecar is held by a live writer. The read yields to the writer this long,
/// then re-probes, instead of blocking on the mode-inverted sidecar-EX (the
/// turnstile convoy that serialized ~990 pre-exec entrants). The ticket futex
/// cuts the wait short on a grant; the timeout re-probes through a self-election
/// that bumped no futex. It caps the re-probe rate under sustained writer
/// priority — it is not a semantic poll interval, and the read still always
/// returns the authoritative state on the next turnstile gap.
const TURNSTILE_READ_BACKOFF: Duration = Duration::from_millis(2);
const DEFERRED_RESCAN_INTERVAL_NS: u64 = 1_000_000_000;
#[cfg(test)]
thread_local! {
    /// Test-injected replacement for [`DEFERRED_RESCAN_INTERVAL_NS`]. The
    /// deadline exercises arm a fixture deadline exactly one interval ahead
    /// and assert the first completion shortens it; with the production 1s
    /// interval, a test process descheduled for ~1s between arming and the
    /// completion crosses the window and flakes the assertion. Injecting a
    /// longer interval (consistently into the arm site AND the validity
    /// clamps, which flush any deadline further than one interval out)
    /// removes the wall-clock race without weakening what the tests pin.
    static DEFERRED_RESCAN_INTERVAL_OVERRIDE_NS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

fn deferred_rescan_interval_ns() -> u64 {
    #[cfg(test)]
    {
        let injected = DEFERRED_RESCAN_INTERVAL_OVERRIDE_NS.with(std::cell::Cell::get);
        if injected != 0 {
            return injected;
        }
    }
    DEFERRED_RESCAN_INTERVAL_NS
}
/// Short, non-renewing coalescing quantum for authoritative grant-scan edges
/// that arrive in bursts: a GRANTED batch's negative completions (which probe
/// outside registry EX and tend to return together) and HELD releases (a
/// draining herd completes en masse). Absorbing every such edge inside one
/// window lets the coordinator run a single O(N) grant scan per quantum instead
/// of reacquiring EX and rescanning between individual publications.
///
/// 10 ms is chosen from the drain profile: a grant scan and a physical
/// acquire/commit are sub-millisecond, so 10 ms comfortably batches a wave of
/// same-quantum completions while adding latency that is invisible next to the
/// 9–20 s a cell spends resident. It is a re-probe/coalesce cap, not a semantic
/// poll interval — the next scan still grants everything grantable, and any
/// edge that misses the window schedules the following scan.
const GRANT_SCAN_COALESCE_INTERVAL_NS: u64 = 10_000_000;

/// CI-only scan-cost accounting (validation instrumentation), companion to the
/// coordinator-wake counters. `GRANT_SCANS` counts authoritative grant scans
/// this process drove and `RECORDS_SCANNED` accumulates their record counts, so
/// their ratio is the mean scan width. Both are relaxed atomics read only when
/// the diagnostics dir is set — a healthy drain shows
/// `records_scanned/grant_scans` far below the peak herd size.
///
/// `SCANS_COALESCED` counts release edges that joined an already-pending scan
/// instead of opening a fresh window, and is deliberately process-local: the
/// increment lands in whichever process drove the release, while the diagnostic
/// line is written by the process sitting in the coordinator wait, so a
/// persisted value would not describe the run's actual coalescing. It exists for
/// `exercise_release_coalesce_for_tests`, which drives the releases and the
/// scans in one process and reads it directly.
static COORDINATOR_GRANT_SCANS: AtomicU64 = AtomicU64::new(0);
static COORDINATOR_RECORDS_SCANNED: AtomicU64 = AtomicU64::new(0);
static COORDINATOR_SCANS_COALESCED: AtomicU64 = AtomicU64::new(0);

fn note_grant_scan(records: usize) {
    COORDINATOR_GRANT_SCANS.fetch_add(1, Ordering::Relaxed);
    COORDINATOR_RECORDS_SCANNED.fetch_add(records as u64, Ordering::Relaxed);
}

fn note_grant_scan_coalesced() {
    COORDINATOR_SCANS_COALESCED.fetch_add(1, Ordering::Relaxed);
}

/// `(grant_scans, records_scanned)` for the coordinator-wake diagnostic line.
/// Cumulative and process-global, like the wake counters.
pub(super) fn coordinator_scan_stats() -> (u64, u64) {
    (
        COORDINATOR_GRANT_SCANS.load(Ordering::Relaxed),
        COORDINATOR_RECORDS_SCANNED.load(Ordering::Relaxed),
    )
}

/// Eight missed one-second heartbeats are enough to transfer progress to the
/// oldest waiter. The displaced live coordinator is parked, so a false
/// positive under extreme descheduling is safe and does not weaken physical
/// or logical allocation fences.
const COORDINATOR_HEARTBEAT_LEASE_NS: u64 = 8_000_000_000;
/// Keep a physically blocked exact head work-conserving for one coordinator
/// backfill window while bounding starvation. This semantic allocation window
/// remains independent of the much shorter coordinator progress heartbeat.
/// During this interval the backfill capacity limits *currently outstanding*
/// bypass work, so a completed small cell can be replaced instead of
/// manufacturing a low-utilization drain bubble. Once the interval expires,
/// no replacement conflicts are issued and at most one capacity wave remains
/// to drain before the head runs.
const BACKFILL_MAX_AGE_NS: u64 = 120_000_000_000;
/// A speculative callback is user code and may legitimately take much longer
/// than a coordinator heartbeat. Keep its bounded recovery window explicit
/// and independent of coordinator takeover.
const REPLAN_WAVE_LEASE_NS: u64 = 120_000_000_000;

#[cfg(test)]
thread_local! {
    static SHARED_STATE_READS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static SHARED_STATE_RECOVERY_UPGRADES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static TICKET_SHARED_MAPPING_BUILDS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static REGISTRY_EX_ACQUISITIONS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
    static LIVENESS_PROBES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static GRANT_PREFIX_RECORD_READS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static PREFIX_COMPARE_RECORD_READS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static FULL_WATCH_MATERIALIZATIONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static ENCODED_WATCH_SERIAL_WALKS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static SCAN_EXACT_WORD_READS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static SCAN_CLAIM_HEAP_SPILLS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static FULL_PREFIX_SNAPSHOT_PUBLISHES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static ACTIVE_LIST_RECORD_READS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static COORDINATOR_ELECTION_RECORD_READS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    #[cfg(test)]
    static CANCEL_GRANTED_AFTER_COMMIT: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    #[cfg(test)]
    static CANCEL_COORDINATOR_AFTER_COMMIT: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    #[cfg(test)]
    static HELD_DROP_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    #[cfg(test)]
    static NOTIFY_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    #[cfg(test)]
    static NOTIFY_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum State {
    Waiting,
    Granted,
    Replan,
    Coordinator,
    CoordinatorStandby,
}

#[derive(Debug)]
pub(super) struct ScheduleSnapshot {
    pub watch: ClaimSet,
    pub candidate_claim: ClaimSet,
    pub candidate_watch: ClaimSet,
    pub predecessors: AggregateSnapshot,
    pub availability: AvailabilitySnapshot,
    pub commit_token: CoordinatorCommitToken,
    pub should_step: bool,
    pub observation: Option<ObservationRequest>,
    pub liveness_due_in: Duration,
    /// Time until the active coordinator must renew its lightweight progress
    /// heartbeat. This is deliberately independent of semantic retries and
    /// full liveness reconciliation: an idle healthy coordinator only updates
    /// one header word at this deadline.
    pub heartbeat_due_in: Duration,
    /// Absolute, non-renewing flush deadline for coalesced speculative-planner
    /// publications. `None` means there is no deferred rescan work.
    pub deferred_rescan_due_in: Option<Duration>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct HeartbeatStatus {
    pub(super) parked: bool,
    pub(super) rescan_pending: bool,
}

#[derive(Debug, Clone, Copy)]
struct AcknowledgeResult {
    acknowledged: bool,
    notify: bool,
}

impl AcknowledgeResult {
    const UNCHANGED: Self = Self {
        acknowledged: false,
        notify: false,
    };
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CoordinatorCommitToken {
    prefix_epoch: u64,
    coordinator_epoch: u64,
}

pub(super) enum FinishAcquireResult {
    Committed(HeldClaim),
    Stale,
}

pub(super) enum FinishPreparationResult {
    Committed(ClaimSet),
    Stale,
}

enum PendingTransition {
    Committed(ClaimSet),
    Contended(ContentionMarker),
}

#[derive(Clone, Copy)]
enum ReplacementFenceEffect {
    ChangesPredecessorPrefix,
    NonFencing,
}

#[derive(Debug)]
pub(super) struct GrantAttempt<T> {
    pub acquired: Option<T>,
    /// When present, a successful callback owns this physical preparation
    /// claim. Commit publishes that physical footprint as the PENDING claim
    /// and retains the selected run intent in the same record's watch until
    /// exact activation atomically replaces both.
    pub preparation_claim: Option<ClaimSet>,
    /// Physical preparation resource that prevented this selected intent from
    /// entering PENDING. Retained for the mixing-invariant assertions below,
    /// but no longer a wake blocker on the GRANTED path: the grant scan's
    /// preparation-slot pool budget bounds the granted cohort to the free-slot
    /// count, so a granted intent always has a slot to race for and a lost
    /// token race requeues WAITING for the next budget-ordered grant instead of
    /// pinning to one token. (The coordinator's own legitimate pool-full wait
    /// is a separate path; its watch spans the whole pool, so any release wakes
    /// it.) It may lie outside the final-run watch and is never run fairness.
    pub preparation_contention: Option<ContentionEvidence>,
    pub next_claim: ClaimSet,
    pub contention: Option<ContentionEvidence>,
}

#[cfg(test)]
struct DropProbe {
    dropped: std::rc::Rc<std::cell::Cell<bool>>,
    registry_unlocked: std::rc::Rc<std::cell::Cell<bool>>,
}

#[cfg(test)]
impl Drop for DropProbe {
    fn drop(&mut self) {
        self.dropped.set(true);
        self.registry_unlocked.set(
            try_lock_registry_existing_nonblocking(FlockMode::Shared)
                .ok()
                .flatten()
                .is_some(),
        );
    }
}

#[cfg(test)]
fn arm_drop_before_notify_probe(
    payload_dropped: std::rc::Rc<std::cell::Cell<bool>>,
    dropped_at_notify: std::rc::Rc<std::cell::Cell<bool>>,
) {
    arm_notify_hook_for_tests(move || {
        dropped_at_notify.set(payload_dropped.get());
    });
}

#[cfg(test)]
pub(super) fn arm_notify_hook_for_tests(hook: impl FnOnce() + 'static) {
    NOTIFY_HOOK.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "a coordinator-notify hook is already installed on this test thread"
        );
        *slot.borrow_mut() = Some(Box::new(hook));
    });
}

pub(super) enum GrantResult<T> {
    Acquired(T, HeldClaim),
    Prepared(T, ClaimSet),
    Requeued,
    LostGrant,
}

pub(super) enum PendingOneShotResult<T> {
    Acquired(T, HeldClaim),
    Unavailable,
}

pub(super) struct PendingOneShotProbe<'a> {
    table: &'a Table,
    excluded: &'a ClaimSet,
}

impl PendingOneShotProbe<'_> {
    pub(super) fn candidate_ready(&self, candidate: &ClaimSet) -> Result<bool> {
        Ok(!self
            .table
            .claim_conflicts_aggregate_excluding(candidate, self.excluded)?)
    }
}

#[derive(Debug, Clone)]
pub(super) struct ObservationRequest {
    pub cpus: BTreeMap<usize, (Option<u64>, Option<u64>)>,
    pub llcs: BTreeMap<usize, (Option<u64>, Option<u64>)>,
    pub permits: BTreeMap<usize, (Option<u64>, Option<u64>)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CpuAvailability {
    ExclusiveHeld,
    SharedHeld,
    Free,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CpuObservation {
    pub availability: CpuAvailability,
    pub sh_resolved: bool,
    pub ex_resolved: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LlcAvailability {
    ExclusiveHeld,
    SharedHeld,
    Free,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LlcObservation {
    pub availability: LlcAvailability,
    pub sh_resolved: bool,
    pub ex_resolved: bool,
}

#[derive(Debug, Default)]
pub(super) struct AvailabilityObservation {
    pub cpus: BTreeMap<usize, CpuObservation>,
    pub llcs: BTreeMap<usize, LlcObservation>,
    pub permits: BTreeMap<usize, CpuObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockedOn {
    key: ResourceKey,
    mode: FlockMode,
    serial: u64,
}

#[derive(Default)]
struct PossibleReleasePlan {
    cpu_sh: BTreeSet<usize>,
    cpu_ex: BTreeSet<usize>,
    llc_sh: BTreeSet<usize>,
    llc_ex: BTreeSet<usize>,
    permit_sh: BTreeSet<usize>,
    permit_ex: BTreeSet<usize>,
}

impl PossibleReleasePlan {
    fn is_empty(&self) -> bool {
        self.cpu_sh.is_empty()
            && self.cpu_ex.is_empty()
            && self.llc_sh.is_empty()
            && self.llc_ex.is_empty()
            && self.permit_sh.is_empty()
            && self.permit_ex.is_empty()
    }

    fn extend(&mut self, other: Self) {
        self.cpu_sh.extend(other.cpu_sh);
        self.cpu_ex.extend(other.cpu_ex);
        self.llc_sh.extend(other.llc_sh);
        self.llc_ex.extend(other.llc_ex);
        self.permit_sh.extend(other.permit_sh);
        self.permit_ex.extend(other.permit_ex);
    }
}

pub(super) enum FenceResult<T> {
    Fenced,
    Ran { value: T, watched: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AggregateSnapshot {
    bits: usize,
    cpu_any: Vec<u64>,
    cpu_exclusive: Vec<u64>,
    llc_any: Vec<u64>,
    llc_exclusive: Vec<u64>,
    cpu_shared_holders: Vec<u32>,
    cpu_exclusive_holders: Vec<u32>,
    llc_shared_holders: Vec<u32>,
    llc_exclusive_holders: Vec<u32>,
    build_cpu_claims: Vec<u32>,
    /// In-flight grant charge (`C_GRANT_*`): the exact claims of GRANTED and
    /// REVOKED records, permits folded at their CPU-space resource indices.
    /// Planner bias only — deliberately absent from `conflicts` /
    /// `first_conflict` and from every holder/exclusive accessor, so a
    /// granted claim can bias placement but never fence it.
    cpu_grant_any: Vec<u32>,
    cpu_grant_exclusive: Vec<u32>,
    llc_grant_any: Vec<u32>,
    llc_grant_exclusive: Vec<u32>,
}

impl AggregateSnapshot {
    fn empty(layout: HeaderLayout) -> Self {
        Self {
            bits: layout.bits,
            cpu_any: vec![0; layout.words],
            cpu_exclusive: vec![0; layout.words],
            llc_any: vec![0; layout.words],
            llc_exclusive: vec![0; layout.words],
            cpu_shared_holders: vec![0; layout.bits],
            cpu_exclusive_holders: vec![0; layout.bits],
            llc_shared_holders: vec![0; layout.bits],
            llc_exclusive_holders: vec![0; layout.bits],
            build_cpu_claims: vec![0; layout.bits],
            cpu_grant_any: vec![0; layout.bits],
            cpu_grant_exclusive: vec![0; layout.bits],
            llc_grant_any: vec![0; layout.bits],
            llc_grant_exclusive: vec![0; layout.bits],
        }
    }

    pub(super) fn conflicts(&self, candidate: &ClaimSet) -> Result<bool> {
        validate_claim(candidate)?;
        claim_conflicts_bits(
            candidate,
            &self.cpu_any,
            &self.cpu_exclusive,
            &self.llc_any,
            &self.llc_exclusive,
            self.bits,
        )
    }

    pub(super) fn first_conflict(&self, candidate: &ClaimSet) -> Result<Option<ContentionMarker>> {
        validate_claim(candidate)?;
        let cpu_bits = if candidate.cpu_mode == ClaimMode::Exclusive {
            &self.cpu_any
        } else {
            &self.cpu_exclusive
        };
        for &cpu in &candidate.cpus {
            self.ensure_index(cpu, "CPU")?;
            if cpu_bits[cpu / 64] & (1u64 << (cpu % 64)) != 0 {
                return Ok(Some(ContentionMarker {
                    blocker: ResourceKey::Cpu(cpu),
                    mode: match candidate.cpu_mode {
                        ClaimMode::Shared => FlockMode::Shared,
                        ClaimMode::Exclusive => FlockMode::Exclusive,
                    },
                }));
            }
        }

        let permit_bits = if candidate.permit_mode == ClaimMode::Exclusive {
            &self.cpu_any
        } else {
            &self.cpu_exclusive
        };
        for &permit in &candidate.permits {
            let index = permit_resource_index(permit)?;
            self.ensure_index(index, "permit resource")?;
            if permit_bits[index / 64] & (1u64 << (index % 64)) != 0 {
                return Ok(Some(ContentionMarker {
                    blocker: ResourceKey::Permit(permit),
                    mode: match candidate.permit_mode {
                        ClaimMode::Shared => FlockMode::Shared,
                        ClaimMode::Exclusive => FlockMode::Exclusive,
                    },
                }));
            }
        }

        let llc_bits = if candidate.llc_mode == ClaimMode::Exclusive {
            &self.llc_any
        } else {
            &self.llc_exclusive
        };
        for &llc in &candidate.llcs {
            self.ensure_index(llc, "LLC")?;
            if llc_bits[llc / 64] & (1u64 << (llc % 64)) != 0 {
                return Ok(Some(ContentionMarker {
                    blocker: ResourceKey::Llc(llc),
                    mode: match candidate.llc_mode {
                        ClaimMode::Shared => FlockMode::Shared,
                        ClaimMode::Exclusive => FlockMode::Exclusive,
                    },
                }));
            }
        }
        Ok(None)
    }

    pub(super) fn cpu_holder_count(&self, cpu: usize) -> Result<usize> {
        self.ensure_index(cpu, "CPU")?;
        Ok((self.cpu_shared_holders[cpu] as usize)
            .saturating_add(self.cpu_exclusive_holders[cpu] as usize))
    }

    pub(super) fn cpu_exclusive_held(&self, cpu: usize) -> Result<bool> {
        self.ensure_index(cpu, "CPU")?;
        Ok(self.cpu_exclusive_holders[cpu] != 0)
    }

    pub(super) fn llc_holder_count(&self, llc: usize) -> Result<usize> {
        self.ensure_index(llc, "LLC")?;
        Ok((self.llc_shared_holders[llc] as usize)
            .saturating_add(self.llc_exclusive_holders[llc] as usize))
    }

    pub(super) fn llc_exclusive_held(&self, llc: usize) -> Result<bool> {
        self.ensure_index(llc, "LLC")?;
        Ok(self.llc_exclusive_holders[llc] != 0)
    }

    pub(super) fn cpu_build_claimed(&self, cpu: usize) -> Result<bool> {
        self.ensure_index(cpu, "CPU")?;
        Ok(self.build_cpu_claims[cpu] != 0)
    }

    pub(super) fn cpu_grant_count(&self, cpu: usize) -> Result<usize> {
        self.ensure_index(cpu, "CPU")?;
        Ok(self.cpu_grant_any[cpu] as usize)
    }

    pub(super) fn llc_grant_count(&self, llc: usize) -> Result<usize> {
        self.ensure_index(llc, "LLC")?;
        Ok(self.llc_grant_any[llc] as usize)
    }

    /// Whether `candidate` overlaps any in-flight grant charge, using the
    /// same mode matrix as [`Self::first_conflict`]: an exclusive candidate
    /// resource conflicts with any charge, a shared one only with an
    /// exclusive charge. Permits are tested at their folded CPU-space
    /// indices. This is a soft-avoid input for two-tier candidate selection —
    /// never a fence: callers must fall back to a grant-blind pass when no
    /// grant-free candidate satisfies the request.
    pub(super) fn grant_conflicts(&self, candidate: &ClaimSet) -> Result<bool> {
        validate_claim(candidate)?;
        for &cpu in &candidate.cpus {
            self.ensure_index(cpu, "CPU")?;
            let charged = match candidate.cpu_mode {
                ClaimMode::Exclusive => self.cpu_grant_any[cpu],
                ClaimMode::Shared => self.cpu_grant_exclusive[cpu],
            };
            if charged != 0 {
                return Ok(true);
            }
        }
        for &permit in &candidate.permits {
            let index = permit_resource_index(permit)?;
            self.ensure_index(index, "permit resource")?;
            let charged = match candidate.permit_mode {
                ClaimMode::Exclusive => self.cpu_grant_any[index],
                ClaimMode::Shared => self.cpu_grant_exclusive[index],
            };
            if charged != 0 {
                return Ok(true);
            }
        }
        for &llc in &candidate.llcs {
            self.ensure_index(llc, "LLC")?;
            let charged = match candidate.llc_mode {
                ClaimMode::Exclusive => self.llc_grant_any[llc],
                ClaimMode::Shared => self.llc_grant_exclusive[llc],
            };
            if charged != 0 {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn ensure_index(&self, index: usize, kind: &str) -> Result<()> {
        if index >= self.bits {
            anyhow::bail!("{kind} index {index} exceeds queue registry capacity");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(super) struct AvailabilitySnapshot {
    bits: usize,
    cpu_known: Vec<u64>,
    cpu_sh_available: Vec<u64>,
    cpu_ex_available: Vec<u64>,
    llc_known: Vec<u64>,
    llc_sh_available: Vec<u64>,
    llc_ex_available: Vec<u64>,
}

impl AvailabilitySnapshot {
    pub(super) fn allows(&self, candidate: &ClaimSet) -> Result<bool> {
        validate_claim(candidate)?;
        let cpu_available = match candidate.cpu_mode {
            ClaimMode::Shared => &self.cpu_sh_available,
            ClaimMode::Exclusive => &self.cpu_ex_available,
        };
        for &cpu in &candidate.cpus {
            if cpu >= self.bits {
                anyhow::bail!("CPU index {cpu} exceeds queue registry capacity");
            }
            let mask = 1u64 << (cpu % 64);
            if self.cpu_known[cpu / 64] & mask == 0 || cpu_available[cpu / 64] & mask == 0 {
                return Ok(false);
            }
        }
        let permit_available = match candidate.permit_mode {
            ClaimMode::Shared => &self.cpu_sh_available,
            ClaimMode::Exclusive => &self.cpu_ex_available,
        };
        for &permit in &candidate.permits {
            let index = permit_resource_index(permit)?;
            if index >= self.bits {
                anyhow::bail!("permit index {permit} exceeds queue registry capacity");
            }
            let mask = 1u64 << (index % 64);
            if self.cpu_known[index / 64] & mask == 0 || permit_available[index / 64] & mask == 0 {
                return Ok(false);
            }
        }
        let llc_available = match candidate.llc_mode {
            ClaimMode::Shared => &self.llc_sh_available,
            ClaimMode::Exclusive => &self.llc_ex_available,
        };
        for &llc in &candidate.llcs {
            if llc >= self.bits {
                anyhow::bail!("LLC index {llc} exceeds queue registry capacity");
            }
            let mask = 1u64 << (llc % 64);
            if self.llc_known[llc / 64] & mask == 0 || llc_available[llc / 64] & mask == 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[derive(Debug, Clone)]
struct Record {
    slot: u64,
    state: u32,
    ticket: u64,
    pid: u32,
    claim: ClaimSet,
    watch: ClaimSet,
    blocked_on: Option<BlockedOn>,
    issue_serial: u64,
    replan_claim_epoch: u64,
    grant_epoch: u64,
    prefix_epoch: u64,
    backfill_capacity: u32,
    backfill_started_ns: u64,
    prev_active: u64,
    next_active: u64,
}

#[derive(Debug, Clone, Copy)]
struct EncodedWatchModes {
    cpu: ClaimMode,
    llc: ClaimMode,
    permit: ClaimMode,
}

/// Compact identity of one encoded alternative-watch serial query.
///
/// The fixed-seed AHash is persisted with the immutable watch and validated
/// whenever a full record is decoded or repaired. Grant scans therefore avoid
/// copying hundreds of bitmap words merely to memoize identical watches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EncodedWatchSerialMemoKey {
    watch_identity: u64,
    blocked_on: Option<EncodedWatchSerialBlockedOn>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EncodedWatchSerialBlockedOn {
    kind: u32,
    index: usize,
    exclusive: bool,
    serial: u64,
}

/// Read-only resource view shared by the authoritative [`ClaimSet`] and the
/// allocation-free common case used by grant scans.
///
/// Full record decode retains `BTreeSet` semantics for every mutation and
/// repair boundary. A grant scan only needs sorted iteration, membership, and
/// cardinality, so rebuilding one tree node per resource for every record in
/// every scan is pure allocator traffic.
trait ClaimView {
    fn llcs(&self) -> impl Iterator<Item = usize> + '_;
    fn cpus(&self) -> impl Iterator<Item = usize> + '_;
    fn permits(&self) -> impl Iterator<Item = usize> + '_;
    fn llc_len(&self) -> usize;
    fn cpu_len(&self) -> usize;
    fn permit_len(&self) -> usize;
    fn contains_llc(&self, llc: usize) -> bool;
    fn contains_cpu(&self, cpu: usize) -> bool;
    fn contains_permit(&self, permit: usize) -> bool;
    fn llc_mode(&self) -> ClaimMode;
    fn cpu_mode(&self) -> ClaimMode;
    fn permit_mode(&self) -> ClaimMode;
    fn admission_class(&self) -> AdmissionClass;
}

impl ClaimView for ClaimSet {
    fn llcs(&self) -> impl Iterator<Item = usize> + '_ {
        self.llcs.iter().copied()
    }

    fn cpus(&self) -> impl Iterator<Item = usize> + '_ {
        self.cpus.iter().copied()
    }

    fn permits(&self) -> impl Iterator<Item = usize> + '_ {
        self.permits.iter().copied()
    }

    fn llc_len(&self) -> usize {
        self.llcs.len()
    }

    fn cpu_len(&self) -> usize {
        self.cpus.len()
    }

    fn permit_len(&self) -> usize {
        self.permits.len()
    }

    fn contains_llc(&self, llc: usize) -> bool {
        self.llcs.contains(&llc)
    }

    fn contains_cpu(&self, cpu: usize) -> bool {
        self.cpus.contains(&cpu)
    }

    fn contains_permit(&self, permit: usize) -> bool {
        self.permits.contains(&permit)
    }

    fn llc_mode(&self) -> ClaimMode {
        self.llc_mode
    }

    fn cpu_mode(&self) -> ClaimMode {
        self.cpu_mode
    }

    fn permit_mode(&self) -> ClaimMode {
        self.permit_mode
    }

    fn admission_class(&self) -> AdmissionClass {
        self.admission_class
    }
}

const SCAN_INLINE_RESOURCES: usize = 8;

#[derive(Debug, Clone)]
struct ScanClaim {
    llcs: SmallVec<[usize; SCAN_INLINE_RESOURCES]>,
    cpus: SmallVec<[usize; SCAN_INLINE_RESOURCES]>,
    permits: SmallVec<[usize; SCAN_INLINE_RESOURCES]>,
    llc_mode: ClaimMode,
    cpu_mode: ClaimMode,
    permit_mode: ClaimMode,
    admission_class: AdmissionClass,
}

impl ScanClaim {
    fn is_empty(&self) -> bool {
        self.llcs.is_empty() && self.cpus.is_empty() && self.permits.is_empty()
    }
}

impl ClaimView for ScanClaim {
    fn llcs(&self) -> impl Iterator<Item = usize> + '_ {
        self.llcs.iter().copied()
    }

    fn cpus(&self) -> impl Iterator<Item = usize> + '_ {
        self.cpus.iter().copied()
    }

    fn permits(&self) -> impl Iterator<Item = usize> + '_ {
        self.permits.iter().copied()
    }

    fn llc_len(&self) -> usize {
        self.llcs.len()
    }

    fn cpu_len(&self) -> usize {
        self.cpus.len()
    }

    fn permit_len(&self) -> usize {
        self.permits.len()
    }

    fn contains_llc(&self, llc: usize) -> bool {
        self.llcs.contains(&llc)
    }

    fn contains_cpu(&self, cpu: usize) -> bool {
        self.cpus.contains(&cpu)
    }

    fn contains_permit(&self, permit: usize) -> bool {
        self.permits.contains(&permit)
    }

    fn llc_mode(&self) -> ClaimMode {
        self.llc_mode
    }

    fn cpu_mode(&self) -> ClaimMode {
        self.cpu_mode
    }

    fn permit_mode(&self) -> ClaimMode {
        self.permit_mode
    }

    fn admission_class(&self) -> AdmissionClass {
        self.admission_class
    }
}

impl EncodedWatchSerialBlockedOn {
    fn new(blocked: BlockedOn) -> Self {
        let (kind, index) = match blocked.key {
            ResourceKey::Cpu(index) => (BLOCK_CPU, index),
            ResourceKey::Llc(index) => (BLOCK_LLC, index),
            ResourceKey::Permit(index) => (BLOCK_PERMIT, index),
        };
        Self {
            kind,
            index,
            exclusive: blocked.mode == FlockMode::Exclusive,
            serial: blocked.serial,
        }
    }

    fn resource_key(self) -> Result<ResourceKey> {
        match self.kind {
            BLOCK_CPU => Ok(ResourceKey::Cpu(self.index)),
            BLOCK_LLC => Ok(ResourceKey::Llc(self.index)),
            BLOCK_PERMIT => Ok(ResourceKey::Permit(self.index)),
            kind => anyhow::bail!("invalid encoded watch memo blocker kind {kind}"),
        }
    }

    fn mode(self) -> FlockMode {
        if self.exclusive {
            FlockMode::Exclusive
        } else {
            FlockMode::Shared
        }
    }
}

/// The grant scan needs every exact claim, but almost never needs to
/// materialize a ticket's potentially host-wide alternative watch. Keep the
/// scan representation small and retain only the encoded-watch facts needed
/// for selection. Full [`Record`] decoding remains the validation path for
/// callbacks, repair, removal, and diagnostics.
#[derive(Debug, Clone)]
struct ScanRecord {
    slot: u64,
    state: u32,
    ticket: u64,
    claim: ScanClaim,
    watch_modes: EncodedWatchModes,
    watch_identity: u64,
    flexible: bool,
    preparation_intent: bool,
    blocked_on: Option<BlockedOn>,
    external_blocker: Option<ContentionMarker>,
    issue_serial: u64,
    replan_claim_epoch: u64,
    grant_epoch: u64,
    prefix_epoch: u64,
    backfill_capacity: u32,
    backfill_started_ns: u64,
    prev_active: u64,
    next_active: u64,
}

pub(super) struct Ticket {
    namespace: RegistryNamespace,
    slot: u64,
    ticket: u64,
    liveness_path: PathBuf,
    liveness: Option<OwnedFd>,
    shared: Option<TicketSharedMaps>,
    _interrupt_waiter: Option<InterruptibleFlockWaiter>,
    finished: bool,
    // Preparation-slot token sub-range, supplied by the coordinator loop that
    // owns the topology so every scan this ticket drives applies the pool
    // budget uniformly. `None` disables the budget (no preparation capacity, or
    // a ticket that never coordinates a scan — e.g. tests).
    preparation_tokens: Option<std::ops::Range<usize>>,
}

pub(super) enum PendingRegistration {
    Registered(Box<Ticket>),
    Contended(u32),
}

/// One registry-owned publication of a live physical reservation.
///
/// The caller keeps this behind the physical flock set in an outer RAII
/// owner. Its liveness fd lets another process prune the record after a crash;
/// normal teardown removes the exact record synchronously.
pub(super) struct HeldClaim {
    namespace: RegistryNamespace,
    slot: u64,
    ticket: u64,
    liveness_path: PathBuf,
    liveness: Option<OwnedFd>,
}

impl HeldClaim {
    fn from_ticket(ticket: &mut Ticket) -> Result<Self> {
        let liveness = ticket
            .liveness
            .take()
            .ok_or_else(|| anyhow::anyhow!("queue ticket liveness fd disappeared at commit"))?;
        ticket.shared.take();
        ticket._interrupt_waiter.take();
        ticket.finished = true;
        Ok(Self {
            namespace: ticket.namespace.clone(),
            slot: ticket.slot,
            ticket: ticket.ticket,
            liveness_path: ticket.liveness_path.clone(),
            liveness: Some(liveness),
        })
    }

    fn remove_record(&mut self) -> Result<()> {
        let _namespace = self.namespace.enter();
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        if let Some(record) = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
        {
            table.remove_record(&record, false)?;
            table.advance_generation_and_wake_pending()?;
        }
        drop(table);
        drop(_lock);
        Ok(())
    }

    #[cfg(test)]
    fn abandon_for_tests(mut self) {
        // Model abrupt process death: close the authoritative liveness flock
        // without running normal record cleanup or unlinking its inode.
        self.liveness.take();
        std::mem::forget(self);
    }
}

/// Rebuild the sole owner of a PENDING record after a same-PID exec. The
/// inherited liveness fd is checked against both the record identity and the
/// canonical liveness inode. The record's exact claim must still be the
/// transferred physical preparation footprint; its watch retains the selected
/// final intent without sequestering those not-yet-owned run resources.
pub(super) fn import_pending_exec_handoff(
    slot: u64,
    ticket: u64,
    liveness: &OwnedFd,
    preparation_claim: &ClaimSet,
) -> Result<(Ticket, ClaimSet)> {
    use std::os::fd::AsRawFd;

    let namespace = RegistryNamespace::resolve();
    let _namespace = namespace.enter();
    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    let record = table
        .record(slot)?
        .filter(|record| record.ticket == ticket)
        .ok_or_else(|| anyhow::anyhow!("pending exec-handoff ticket {ticket} disappeared"))?;
    anyhow::ensure!(
        record.state == STATE_PENDING,
        "pending exec-handoff ticket {ticket} is in state {}, not PENDING",
        record.state,
    );
    anyhow::ensure!(
        record.pid == std::process::id(),
        "pending exec-handoff ticket {ticket} belongs to PID {}, current PID is {}",
        record.pid,
        std::process::id(),
    );
    anyhow::ensure!(
        record.claim == *preparation_claim,
        "pending exec-handoff ticket {ticket} claim does not match its physical preparation owner",
    );
    anyhow::ensure!(
        claim_covers(&record.watch, preparation_claim),
        "pending exec-handoff preparation resources are not covered by ticket {ticket}'s watch",
    );
    let liveness_path = liveness_path(slot, ticket);
    let actual = File::from(
        liveness
            .try_clone()
            .context("duplicate pending exec-handoff liveness descriptor")?,
    )
    .metadata()
    .context("stat pending exec-handoff liveness descriptor")?;
    let expected = std::fs::metadata(&liveness_path)
        .with_context(|| format!("stat pending liveness path {}", liveness_path.display()))?;
    anyhow::ensure!(
        actual.mode() & libc::S_IFMT == libc::S_IFREG
            && (actual.dev(), actual.ino()) == (expected.dev(), expected.ino()),
        "pending exec-handoff liveness descriptor does not name {}",
        liveness_path.display(),
    );
    // The raw identity is also checked here to make accidental replacement
    // during future refactors obvious in diagnostics.
    anyhow::ensure!(
        liveness.as_raw_fd() >= 0,
        "invalid pending liveness descriptor"
    );
    let shared = table.map_ticket_shared(slot, ticket)?;
    // Keep the inherited owner in the exec-handoff layer until every
    // validation above has succeeded. If import fails, that layer drops the
    // physical preparation owner before this liveness fd; only the completed
    // Ticket needs its own reference to the same authoritative inode.
    let ticket_liveness = liveness
        .try_clone()
        .context("retain pending exec-handoff liveness descriptor")?;
    drop(table);
    drop(_lock);
    Ok((
        Ticket {
            namespace,
            slot,
            ticket,
            liveness_path,
            liveness: Some(ticket_liveness),
            shared: Some(shared),
            _interrupt_waiter: None,
            finished: false,
            preparation_tokens: None,
        },
        record.claim,
    ))
}

impl Drop for HeldClaim {
    fn drop(&mut self) {
        let _namespace = self.namespace.enter();
        #[cfg(test)]
        if let Some(hook) = HELD_DROP_HOOK.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
        if let Err(error) = self.remove_record() {
            tracing::warn!(
                ticket = self.ticket,
                %error,
                "failed to remove held reservation; liveness cleanup will prune it"
            );
        }
        self.liveness.take();
        let _ = std::fs::remove_file(&self.liveness_path);
        notify_coordinator();
    }
}

#[cfg(test)]
pub(super) fn set_held_drop_hook_for_tests(hook: impl FnOnce() + 'static) {
    HELD_DROP_HOOK.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "a HELD drop hook is already installed on this test thread"
        );
        *slot.borrow_mut() = Some(Box::new(hook));
    });
}

#[cfg(test)]
pub(super) fn exercise_held_teardown_notify_count_for_tests() -> Result<u64> {
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let held = publish_acquired(&claim)?;
    let before = NOTIFY_CALLS.with(std::cell::Cell::get);
    drop(held);
    Ok(NOTIFY_CALLS.with(std::cell::Cell::get).wrapping_sub(before))
}

#[cfg(test)]
pub(super) fn abandon_held_for_tests(held: HeldClaim) {
    held.abandon_for_tests();
}

#[cfg(test)]
pub(super) fn exercise_pending_activation_overlap_watch_for_tests() -> Result<(bool, bool, bool)> {
    let initial = ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let mut ticket = match Ticket::register_pending(3, initial.clone())? {
        PendingRegistration::Registered(ticket) => *ticket,
        PendingRegistration::Contended(_) => {
            anyhow::bail!("isolated pending activation test unexpectedly contended")
        }
    };
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        // Model the observer having resolved the PENDING watch before exact
        // activation. Replacing the sole watch must either preserve that
        // observation or publish a fresh observation request.
        table.set_bitmap_bit(B_CPU_KNOWN, 1, true)?;
        table.set_bitmap_bit(B_CPU_SH_AVAILABLE, 1, true)?;
        table.set_bitmap_bit(B_PENDING_CPU_SH, 1, false)?;
        table.set_bitmap_bit(B_CANDIDATE_CPU_SH, 1, true)?;
        table.refresh_pending_observation_flag()?;
    }
    let exact = ClaimSet::with_modes(
        std::iter::empty(),
        [1usize, 2usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    ticket.activate_pending(&initial, exact.clone(), exact, None)?;
    ticket.notify_after_coordinator_payload_drop();
    let result = {
        let _lock = lock_registry_existing(FlockMode::Shared)?;
        let table = Table::open_existing()?;
        (
            table.bitmap_bit(B_WATCH_CPUS, 1)?,
            table.bitmap_bit(B_CANDIDATE_CPU_SH, 1)?,
            table.bitmap_bit(B_PENDING_CPU_SH, 1)?,
        )
    };
    drop(ticket);
    Ok(result)
}

#[cfg(test)]
pub(super) fn register_pending_claim_for_tests(claim: ClaimSet) -> Result<Ticket> {
    match Ticket::register_pending(required_resource_bits(&claim), claim)? {
        PendingRegistration::Registered(ticket) => Ok(*ticket),
        PendingRegistration::Contended(_) => {
            anyhow::bail!("isolated pending-claim test unexpectedly contended")
        }
    }
}

#[cfg(test)]
pub(super) fn registry_ex_acquisition_count_for_tests() -> u64 {
    REGISTRY_EX_ACQUISITIONS.with(std::cell::Cell::get)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileIdentity {
    _device: u64,
    _inode: u64,
    length: u64,
}

impl FileIdentity {
    fn validate_open_path(file: &File, path: &Path, description: &str) -> Result<Self> {
        let opened = file
            .metadata()
            .with_context(|| format!("stat opened {description} {}", path.display()))?;
        let named = std::fs::metadata(path)
            .with_context(|| format!("stat named {description} {}", path.display()))?;
        anyhow::ensure!(
            opened.is_file(),
            "{description} {} is not a regular file",
            path.display(),
        );
        anyhow::ensure!(
            (opened.dev(), opened.ino()) == (named.dev(), named.ino()),
            "opened {description} {} no longer names its canonical inode",
            path.display(),
        );
        Ok(Self {
            _device: opened.dev(),
            _inode: opened.ino(),
            length: opened.len(),
        })
    }
}

/// Read-only MAP_SHARED view retained for one live ticket. The chunk mapping
/// supplies both ordinary state reads and the shared futex address; the header
/// mapping supplies queue epochs and coordinator liveness without reopening or
/// remapping either inode on every wake.
struct TicketSharedMaps {
    header: Mmap,
    chunk: Mmap,
    layout: HeaderLayout,
    record_range: std::ops::Range<usize>,
    header_identity: FileIdentity,
    chunk_identity: FileIdentity,
    wake: *const AtomicU32,
}

// Both mappings are MAP_SHARED and read-only. Non-atomic bytes are read only
// while the registry SH flock excludes every writer; the sole lock-free access
// is the aligned AtomicU32 futex word.
unsafe impl Send for TicketSharedMaps {}

impl TicketSharedMaps {
    fn open(table: &mut Table, slot: u64, ticket: u64) -> Result<Self> {
        let next_slot = table.next_slot()?;
        if slot >= next_slot {
            anyhow::bail!("queue shared-mapping slot {slot} is outside 0..{next_slot}");
        }

        let header_path = header_path();
        let header_file = File::open(&header_path)
            .with_context(|| format!("open queue registry header {}", header_path.display()))?;
        let header_identity =
            FileIdentity::validate_open_path(&header_file, &header_path, "queue registry header")?;
        let header = unsafe { Mmap::map(&header_file) }
            .with_context(|| format!("map queue registry header {}", header_path.display()))?;
        let layout = HeaderLayout::validate(&header)?;
        anyhow::ensure!(
            layout == table.layout,
            "queue registry header layout changed while mapping ticket {ticket}",
        );
        anyhow::ensure!(
            usize::try_from(header_identity.length).is_ok_and(|length| length == header.len()),
            "queue registry header mapping length does not match its inode",
        );
        let mapped_next_slot = read_u64(&header, H_NEXT_SLOT);
        anyhow::ensure!(
            slot < mapped_next_slot && mapped_next_slot <= MAX_REGISTRY_SLOTS,
            "ticket {ticket} slot {slot} is outside mapped registry high-water {mapped_next_slot}",
        );

        let (chunk, record_range) = record_range(slot, layout.record_size)?;
        let chunk_path = chunk_path(chunk);
        let chunk_file = File::open(&chunk_path)
            .with_context(|| format!("open queue registry chunk {}", chunk_path.display()))?;
        let chunk_identity =
            FileIdentity::validate_open_path(&chunk_file, &chunk_path, "queue registry chunk")?;
        let chunk_map = unsafe { Mmap::map(&chunk_file) }
            .with_context(|| format!("map queue registry chunk {}", chunk_path.display()))?;
        anyhow::ensure!(
            usize::try_from(chunk_identity.length).is_ok_and(|length| length == chunk_map.len()),
            "queue registry chunk {chunk} mapping length does not match its inode",
        );
        anyhow::ensure!(
            record_range.end <= chunk_map.len(),
            "queue registry chunk {chunk} is too short for ticket {ticket} slot {slot}: {} bytes < {}",
            chunk_map.len(),
            record_range.end,
        );
        let bytes = &chunk_map[record_range.clone()];
        let mapped_ticket = read_u64(bytes, R_TICKET);
        let state = read_u32(bytes, R_STATE);
        anyhow::ensure!(
            mapped_ticket == ticket && state != STATE_FREE,
            "ticket {ticket} slot {slot} did not name its live record while mapping (ticket={mapped_ticket}, state={state})",
        );
        let wake_offset = record_range
            .start
            .checked_add(R_WAKE)
            .ok_or_else(|| anyhow::anyhow!("queue futex offset overflow"))?;
        let wake = unsafe { chunk_map.as_ptr().add(wake_offset).cast::<AtomicU32>() };
        anyhow::ensure!(
            (wake as usize).is_multiple_of(std::mem::align_of::<AtomicU32>()),
            "ticket {ticket} slot {slot} futex address is not naturally aligned",
        );
        #[cfg(test)]
        TICKET_SHARED_MAPPING_BUILDS.with(|count| count.set(count.get().saturating_add(1)));
        Ok(Self {
            header,
            chunk: chunk_map,
            layout,
            record_range,
            header_identity,
            chunk_identity,
            wake,
        })
    }

    fn validate_header(&self, slot: u64, ticket: u64) -> Result<HeaderLayout> {
        let layout = HeaderLayout::validate(&self.header)?;
        anyhow::ensure!(
            layout == self.layout,
            "queue registry layout changed under live ticket {ticket}",
        );
        anyhow::ensure!(
            usize::try_from(self.header_identity.length)
                .is_ok_and(|length| length == self.header.len()),
            "queue registry header inode length changed under live ticket {ticket}",
        );
        let next_slot = read_u64(&self.header, H_NEXT_SLOT);
        anyhow::ensure!(
            next_slot <= MAX_REGISTRY_SLOTS && slot < next_slot,
            "live queue ticket {ticket} slot {slot} is outside 0..{next_slot}",
        );
        Ok(layout)
    }

    fn record_bytes(&self, slot: u64, ticket: u64) -> Result<&[u8]> {
        anyhow::ensure!(
            self.record_range.end <= self.chunk.len()
                && usize::try_from(self.chunk_identity.length)
                    .is_ok_and(|length| length == self.chunk.len()),
            "queue registry chunk mapping changed under live ticket {ticket}",
        );
        let bytes = &self.chunk[self.record_range.clone()];
        let mapped_ticket = read_u64(bytes, R_TICKET);
        anyhow::ensure!(
            mapped_ticket == ticket,
            "live queue ticket {ticket} disappeared from slot {slot} (found ticket {mapped_ticket})",
        );
        Ok(bytes)
    }

    fn expected(&self) -> u32 {
        // SAFETY: `wake` is aligned, points into `chunk`, and `chunk` outlives
        // every access through it.
        unsafe { (&*self.wake).load(Ordering::Acquire) }
    }

    fn wait(&self, expected: u32, timeout: Duration) -> Result<bool> {
        #[cfg(test)]
        let wait_marker = std::env::var_os(RETAINED_FUTEX_WAIT_MARKER_ENV);
        #[cfg(test)]
        if let Some(marker) = &wait_marker {
            std::fs::write(marker, b"entered").with_context(|| {
                format!(
                    "publish retained-futex wait marker {}",
                    Path::new(marker).display()
                )
            })?;
        }
        #[cfg(test)]
        if let Some(gate) = std::env::var_os(RETAINED_FUTEX_WAIT_GATE_ENV) {
            // Let the cross-process regression deterministically put the
            // publication between the userspace sample and FUTEX_WAIT. This
            // is test-only; production never polls a filesystem gate here.
            while !Path::new(&gate).exists() {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        #[cfg(test)]
        let publish_wait_resolution = || -> Result<()> {
            if let Some(marker) = &wait_marker {
                std::fs::write(marker, b"woken").with_context(|| {
                    format!(
                        "publish retained-futex wake marker {}",
                        Path::new(marker).display()
                    )
                })?;
            }
            Ok(())
        };
        let ts = libc::timespec {
            tv_sec: timeout.as_secs().try_into().unwrap_or(libc::time_t::MAX),
            tv_nsec: timeout.subsec_nanos().into(),
        };
        // SAFETY: the word is a live, aligned u32 in a MAP_SHARED mapping.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_futex,
                self.wake.cast::<u32>(),
                libc::FUTEX_WAIT,
                expected,
                &ts as *const libc::timespec,
                std::ptr::null::<u32>(),
                0u32,
            )
        };
        if rc == 0 {
            #[cfg(test)]
            publish_wait_resolution()?;
            return Ok(false);
        }
        let error = std::io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::ETIMEDOUT) => Ok(true),
            Some(libc::EAGAIN) => {
                // The shared wake word changed after `expected` was sampled
                // but before FUTEX_WAIT entered the kernel. That is the
                // sample-before-wait protocol resolving the same publication,
                // not a missed wake. Keep the test marker faithful to that
                // semantic so descheduling cannot strand its observer.
                #[cfg(test)]
                publish_wait_resolution()?;
                Ok(false)
            }
            Some(libc::EINTR) => Ok(false),
            _ => Err(error.into()),
        }
    }
}

impl Ticket {
    pub(super) fn enter_namespace(&self) -> RegistryNamespaceGuard {
        self.namespace.enter()
    }

    /// Supply the preparation-slot token sub-range so every grant scan this
    /// ticket drives (while it holds the coordinator license) applies the pool
    /// budget. The coordinator loop, which owns the topology, sets this once;
    /// the registry itself keeps treating the indices opaquely.
    pub(super) fn set_preparation_tokens(&mut self, tokens: Option<std::ops::Range<usize>>) {
        self.preparation_tokens = tokens;
    }

    /// Publish a same-PID pre-exec arrival together with its complete bounded
    /// preparation footprint.  The caller already owns the matching physical
    /// permit flocks.  Returning `None` means an older registry claim won the
    /// race; the caller must drop those flocks and try another preparation
    /// candidate.
    pub(super) fn register_pending(
        required_bits: usize,
        claim: ClaimSet,
    ) -> Result<PendingRegistration> {
        Self::register_pending_impl(required_bits, claim, false)?.ok_or_else(|| {
            anyhow::anyhow!("blocking pending registration unexpectedly reported contention")
        })
    }

    /// Attempt to publish a same-PID pre-exec owner without waiting for the
    /// registry writer. `None` means the registry flock was busy; a claim
    /// conflict remains a distinct `PendingRegistration::Contended` result so
    /// the caller may try another physical preparation tuple.
    pub(super) fn try_register_pending(
        required_bits: usize,
        claim: ClaimSet,
    ) -> Result<Option<PendingRegistration>> {
        Self::register_pending_impl(required_bits, claim, true)
    }

    fn register_pending_impl(
        required_bits: usize,
        claim: ClaimSet,
        nonblocking: bool,
    ) -> Result<Option<PendingRegistration>> {
        let namespace = RegistryNamespace::resolve();
        let _namespace = namespace.enter();
        validate_claim(&claim)?;
        anyhow::ensure!(
            !claim.is_empty(),
            "PENDING admission must reserve preparation capacity"
        );
        let watch = claim.clone();
        materialize_claim_paths(&watch)?;
        let _lock = if nonblocking {
            let Some(lock) = try_lock_registry_for_initialization()? else {
                return Ok(None);
            };
            lock
        } else {
            lock_registry_interruptible(None)?
        };
        let mut table = Table::open(required_bits.max(required_resource_bits(&watch)).max(1))?;
        if nonblocking {
            if atomic_u64(&table.header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0 {
                return Ok(None);
            }
        } else {
            table.repair_consistency_if_needed()?;
            table.recover_coordinator_if_dead()?;
        }
        if table.claim_conflicts_aggregate(&claim)? {
            return Ok(Some(PendingRegistration::Contended(
                table.generation_wake(),
            )));
        }
        table.begin_transaction()?;

        let ticket = table.next_ticket()?;
        let slot = table.allocate_slot()?;
        let liveness_path = liveness_path(slot, ticket);
        let liveness = try_flock(&liveness_path, FlockMode::Exclusive)?.ok_or_else(|| {
            anyhow::anyhow!("fresh pending-ticket liveness file is already locked")
        })?;
        let predecessors = table.aggregate_claim_snapshot();
        let newly_watched = table.newly_watched(&watch)?;
        let issue_serial = table.max_watch_serial(&watch)?;
        table.initialize_record(
            slot,
            ticket,
            std::process::id(),
            &claim,
            &watch,
            STATE_PENDING,
            &predecessors,
            table.claim_epoch(),
            issue_serial,
        )?;
        table.append_active(slot)?;
        table.adjust_claim_counts(&claim, true)?;
        table.adjust_watch_counts(&watch, true)?;
        table.mark_observation_modes(&newly_watched)?;
        table.set_next_ticket(
            ticket
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("queue ticket id overflow"))?,
        );
        table.advance_generation()?;
        table.finish_transaction()?;
        let shared = table.map_ticket_shared(slot, ticket)?;
        drop(table);
        drop(_lock);

        Ok(Some(PendingRegistration::Registered(Box::new(Self {
            namespace,
            slot,
            ticket,
            liveness_path,
            liveness: Some(liveness),
            shared: Some(shared),
            _interrupt_waiter: None,
            finished: false,
            preparation_tokens: None,
        }))))
    }

    /// Atomically replace this process's physical-preparation PENDING claim
    /// with one complete schedulable claim. The same ticket and its selected
    /// intent watch remain published throughout the transition.
    pub(super) fn activate_pending(
        &mut self,
        expected_pending: &ClaimSet,
        claim: ClaimSet,
        watch: ClaimSet,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        let _namespace = self.namespace.enter();
        validate_claim(expected_pending)?;
        validate_claim(&claim)?;
        let watch = union_claims(&watch, &claim);
        validate_claim_within_watch(&claim, &watch)?;
        materialize_claim_paths(&watch)?;
        let interrupt_waiter = cancelled
            .map(|_| InterruptibleFlockWaiter::register())
            .transpose()?;

        let _lock = lock_registry_interruptible_existing(cancelled)?;
        let mut table = Table::open(required_resource_bits(&watch))?;
        table.repair_consistency_if_needed()?;
        table.recover_coordinator_if_dead()?;
        let record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("pending ticket {} disappeared", self.ticket))?;
        anyhow::ensure!(
            record.state == STATE_PENDING,
            "pending ticket {} is in state {}, not PENDING",
            self.ticket,
            record.state,
        );
        anyhow::ensure!(
            record.pid == std::process::id(),
            "pending ticket {} belongs to PID {}, current PID is {}",
            self.ticket,
            record.pid,
            std::process::id(),
        );
        anyhow::ensure!(
            record.claim == *expected_pending && claim_covers(&record.watch, expected_pending),
            "pending ticket {} preparation claim/watch changed before exact activation",
            self.ticket,
        );

        let issue_serial = table.max_watch_serial(&watch)?;
        table.begin_transaction()?;
        table.adjust_claim_counts(&record.claim, false)?;
        table.adjust_watch_counts(&record.watch, false)?;
        // Compute the observation transition only after removing the old
        // PENDING watch.  Otherwise an overlapping PENDING -> WAITING watch
        // looks pre-existing here, then drops to zero below and loses its
        // observed mode permanently.
        let newly_watched = table.newly_watched(&watch)?;
        table.adjust_claim_counts(&claim, true)?;
        table.adjust_watch_counts(&watch, true)?;
        table.mark_observation_modes(&newly_watched)?;
        let layout = table.layout;
        {
            let bytes = table
                .record_bytes_mut(self.slot)?
                .ok_or_else(|| anyhow::anyhow!("pending slot {} disappeared", self.slot))?;
            // Invalidate the authoritative record before rewriting its claim
            // and watch. A killed writer then leaves a FREE image which dirty
            // repair can discard instead of an active PENDING record with
            // partially cleared bitsets.
            write_u32(bytes, R_STATE, STATE_FREE);
            clear_record_claim_bits(bytes, layout);
            clear_record_watch_bits(bytes, layout);
            crash_at_for_tests("activate_pending_state_free_before_record");
            write_u32(
                bytes,
                R_CLAIM_LLC_MODE,
                u32::from(claim.llc_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_CLAIM_CPU_MODE,
                u32::from(claim.cpu_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_CLAIM_PERMIT_MODE,
                u32::from(claim.permit_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_WATCH_LLC_MODE,
                u32::from(watch.llc_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_WATCH_CPU_MODE,
                u32::from(watch.cpu_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_WATCH_PERMIT_MODE,
                u32::from(watch.permit_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_CLAIM_CLASS,
                encode_admission_class(claim.admission_class),
            );
            write_u32(
                bytes,
                R_WATCH_CLASS,
                encode_admission_class(watch.admission_class),
            );
            encode_claim(bytes, layout, &claim, &watch)?;
            write_u64(bytes, R_ISSUE_SERIAL, issue_serial);
            write_u64(bytes, R_GRANT_EPOCH, 0);
            write_u64(bytes, R_REPLAN_CLAIM_EPOCH, 0);
            write_u64(bytes, R_PREFIX_EPOCH, 0);
            write_u32(
                bytes,
                R_BACKFILL_CAPACITY,
                backfill_capacity_for_watch(&watch),
            );
            write_u64(bytes, R_BACKFILL_STARTED_NS, 0);
            write_u64(bytes, R_BLOCKED_SERIAL, 0);
            write_u32(bytes, R_BLOCK_KIND, BLOCK_NONE);
            write_u32(bytes, R_BLOCK_MODE, 0);
            write_u64(bytes, R_BLOCK_INDEX, 0);
            // Publish the complete record last.
            write_u32(bytes, R_STATE, STATE_WAITING);
        }
        table.mark_claim_changed_fencing(self.ticket, &claim)?;
        table.advance_generation_and_wake_pending()?;
        table.elect_coordinator_in_transaction()?;
        table.finish_transaction()?;
        self._interrupt_waiter = interrupt_waiter;
        drop(table);
        drop(_lock);
        Ok(())
    }

    /// Consume one PENDING admission with a single nonblocking physical
    /// attempt. The physical preparation claim and selected-intent watch remain
    /// published and the registry EX lock remains held until the attempt either
    /// becomes HELD in this exact slot or is removed synchronously; there is no
    /// WAITING/coordinator state and no release/re-register window.
    pub(super) fn try_activate_pending_once<T>(
        &mut self,
        expected_pending: &ClaimSet,
        attempt: impl FnOnce(&PendingOneShotProbe<'_>) -> Result<Option<(ClaimSet, T)>>,
    ) -> Result<PendingOneShotResult<T>> {
        let _namespace = self.namespace.enter();
        validate_claim(expected_pending)?;
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        table.recover_coordinator_if_dead()?;
        let record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("pending ticket {} disappeared", self.ticket))?;
        anyhow::ensure!(
            record.state == STATE_PENDING,
            "pending ticket {} is in state {}, not PENDING",
            self.ticket,
            record.state,
        );
        anyhow::ensure!(
            record.pid == std::process::id(),
            "pending ticket {} belongs to PID {}, current PID is {}",
            self.ticket,
            record.pid,
            std::process::id(),
        );
        anyhow::ensure!(
            record.claim == *expected_pending && claim_covers(&record.watch, expected_pending),
            "pending ticket {} preparation claim/watch changed before one-shot activation",
            self.ticket,
        );

        let probe = PendingOneShotProbe {
            table: &table,
            excluded: &record.claim,
        };
        let attempted = attempt(&probe);
        let attempted = match attempted {
            Ok(attempted) => attempted,
            Err(error) => {
                table.remove_record(&record, false)?;
                table.advance_generation_and_wake_pending()?;
                self.finish_removed_record();
                drop(table);
                drop(_lock);
                notify_coordinator();
                return Err(error);
            }
        };
        let Some((exact, value)) = attempted else {
            table.remove_record(&record, false)?;
            table.advance_generation_and_wake_pending()?;
            self.finish_removed_record();
            drop(table);
            drop(_lock);
            notify_coordinator();
            return Ok(PendingOneShotResult::Unavailable);
        };

        let commit = validate_claim(&exact).and_then(|()| {
            materialize_claim_paths(&exact)?;
            if table.claim_conflicts_aggregate_excluding(&exact, &record.claim)? {
                Ok(false)
            } else {
                table.promote_record_to_held(&record, &exact, &[])?;
                Ok(true)
            }
        });
        match commit {
            Ok(true) => {
                let held = HeldClaim::from_ticket(self)?;
                drop(table);
                drop(_lock);
                notify_coordinator();
                Ok(PendingOneShotResult::Acquired(value, held))
            }
            Ok(false) => {
                table.remove_record(&record, false)?;
                table.advance_generation_and_wake_pending()?;
                self.finish_removed_record();
                drop(table);
                drop(_lock);
                drop(value);
                notify_coordinator();
                Ok(PendingOneShotResult::Unavailable)
            }
            Err(error) => {
                table.remove_record(&record, false)?;
                table.advance_generation_and_wake_pending()?;
                self.finish_removed_record();
                drop(table);
                drop(_lock);
                drop(value);
                notify_coordinator();
                Err(error)
            }
        }
    }

    pub(super) fn pending_exec_handoff_parts(&self) -> Result<(u64, u64, std::os::fd::RawFd)> {
        use std::os::fd::AsRawFd;
        let _namespace = self.namespace.enter();
        let liveness = self.liveness.as_ref().ok_or_else(|| {
            anyhow::anyhow!("pending admission liveness descriptor was already consumed")
        })?;
        Ok((self.slot, self.ticket, liveness.as_raw_fd()))
    }

    #[cfg(test)]
    pub(super) fn pending_claim_watch_for_tests(&self) -> Result<(ClaimSet, ClaimSet)> {
        let _namespace = self.namespace.enter();
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        let record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("pending test ticket {} disappeared", self.ticket))?;
        anyhow::ensure!(
            record.state == STATE_PENDING,
            "pending test ticket {} is in state {}, not PENDING",
            self.ticket,
            record.state,
        );
        Ok((record.claim, record.watch))
    }

    #[cfg(test)]
    pub(super) fn register(
        claim: ClaimSet,
        watch: ClaimSet,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        Self::register_after_contention(claim, watch, None, cancelled)
    }

    pub(super) fn register_after_contention(
        claim: ClaimSet,
        watch: ClaimSet,
        initial_contention: Option<ContentionSet>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        Self::register_after_contention_with_capacity(
            claim,
            watch,
            initial_contention,
            cancelled,
            0,
        )
    }

    pub(super) fn register_after_contention_with_capacity(
        claim: ClaimSet,
        watch: ClaimSet,
        initial_contention: Option<ContentionSet>,
        cancelled: Option<&AtomicBool>,
        required_bits_hint: usize,
    ) -> Result<Self> {
        let namespace = RegistryNamespace::resolve();
        let _namespace = namespace.enter();
        validate_claim(&claim)?;
        if !watch.llcs.is_empty()
            && !claim.llcs.is_empty()
            && watch.llc_mode != ClaimMode::Exclusive
            && watch.llc_mode != claim.llc_mode
        {
            anyhow::bail!(
                "queue watch LLC mode {:?} does not cover exact claim mode {:?}",
                watch.llc_mode,
                claim.llc_mode
            );
        }
        if !watch.cpus.is_empty()
            && !claim.cpus.is_empty()
            && watch.cpu_mode != ClaimMode::Exclusive
            && watch.cpu_mode != claim.cpu_mode
        {
            anyhow::bail!(
                "queue watch CPU mode {:?} does not cover exact claim mode {:?}",
                watch.cpu_mode,
                claim.cpu_mode
            );
        }
        if !watch.permits.is_empty()
            && !claim.permits.is_empty()
            && watch.permit_mode != ClaimMode::Exclusive
            && watch.permit_mode != claim.permit_mode
        {
            anyhow::bail!(
                "queue watch permit mode {:?} does not cover exact claim mode {:?}",
                watch.permit_mode,
                claim.permit_mode
            );
        }
        let watch = union_claims(&watch, &claim);
        let contention_markers = initial_contention
            .as_ref()
            .map(ContentionSet::marker_vec)
            .unwrap_or_default();
        validate_contention_within_watch(&contention_markers, &watch)?;
        materialize_claim_paths(&watch)?;
        let interrupt_waiter = cancelled
            .map(|_| InterruptibleFlockWaiter::register())
            .transpose()?;

        let _lock = lock_registry_interruptible(cancelled)?;
        let required_bits = required_resource_bits(&watch).max(required_bits_hint);
        let mut table = Table::open(required_bits)?;
        table.repair_consistency_if_needed()?;
        // A dead coordinator cannot consume its own liveness close. Recover
        // and batch-prune it before publishing a later ticket, so registration
        // never leaves a live waiter behind a dead coordinator.
        table.recover_coordinator_if_dead()?;
        table.begin_transaction()?;

        let ticket = table.next_ticket()?;
        let slot = table.allocate_slot()?;
        let liveness_path = liveness_path(slot, ticket);
        let liveness = try_flock(&liveness_path, FlockMode::Exclusive)?
            .ok_or_else(|| anyhow::anyhow!("fresh queue ticket liveness file is already locked"))?;
        let has_predecessor = read_u64(&table.header, H_ACTIVE_HEAD) != NONE_SLOT;
        let conflicts = table.claim_conflicts_aggregate(&claim)?;
        let flexible = claim_is_flexible(&claim, &watch);
        let exact_blocker =
            contention_markers
                .iter()
                .copied()
                .find(|marker| match marker.blocker {
                    ResourceKey::Cpu(index) => claim.cpus.contains(&index),
                    ResourceKey::Llc(index) => claim.llcs.contains(&index),
                    ResourceKey::Permit(index) => claim.permits.contains(&index),
                });
        let contended_exact = exact_blocker.is_some();
        let initial_state = if has_predecessor
            && !contended_exact
            && !conflicts
            && table.claim_availability_compatible(&claim)?
        {
            STATE_GRANTED
        } else {
            STATE_WAITING
        };
        // REPLAN is speculative planning work, not runnable capacity. A
        // flexible ticket which cannot use its exact designation joins
        // WAITING and asks the authoritative coordinator scan to publish it
        // with the rest of the current finite planning wave. Registration
        // itself never bypasses that prefix-validation scan.
        let needs_initial_replan = has_predecessor && flexible && initial_state == STATE_WAITING;
        // A newly appended ticket's predecessor prefix is exactly the global
        // aggregate before its own exact claim is counted.
        let claim_epoch = table.claim_epoch();
        let issue_serial = if initial_state == STATE_GRANTED {
            table.max_watch_serial(&claim)?
        } else {
            table.max_watch_serial(&watch)?
        };
        let predecessors = table.aggregate_claim_snapshot();
        let newly_watched = table.newly_watched(&watch)?;
        let blocked_at = exact_blocker
            .map(|marker| table.blocker_serial(marker.blocker, marker.mode))
            .transpose()?;
        table.initialize_record(
            slot,
            ticket,
            std::process::id(),
            &claim,
            &watch,
            initial_state,
            &predecessors,
            claim_epoch,
            issue_serial,
        )?;
        if needs_initial_replan {
            table.invalidate_record_prefix(slot)?;
        }
        table.append_active(slot)?;
        crash_at_for_tests("register_record_before_counts");
        table.adjust_claim_counts(&claim, true)?;
        table.adjust_watch_counts(&watch, true)?;
        if initial_state == STATE_GRANTED {
            // The born-GRANTED fast path publishes a charged record without
            // passing through `set_record_state`; charge it here, inside the
            // same dirty transaction as its claim counts.
            table.adjust_granted_occupancy(&claim, true)?;
        }
        table.mark_observation_modes(&newly_watched)?;
        table.mark_blockers_unknown(&contention_markers)?;
        if initial_state == STATE_WAITING
            && let Some(marker) = exact_blocker
        {
            table.set_record_blocked(
                slot,
                marker,
                blocked_at.expect("initial blocker serial must accompany evidence"),
            )?;
        }
        table.set_next_ticket(
            ticket
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("queue ticket id overflow"))?,
        );
        table.advance_generation()?;
        if needs_initial_replan {
            table.schedule_deferred_replan_rescan_in_transaction()?;
        }
        if initial_state == STATE_WAITING {
            table.elect_coordinator_in_transaction()?;
        }
        table.finish_transaction()?;
        let shared = table.map_ticket_shared(slot, ticket)?;
        drop(table);
        drop(_lock);
        // UNKNOWN and the initial blocker publication are durable now. Close
        // the writable contention witness before waking the coordinator so it
        // cannot consume this edge while the resource still appears busy.
        drop(initial_contention);
        // A WAITING registration changes the aggregate watch even when its
        // speculative rescan is coalesced. Wake the coordinator to install
        // that watch/observation state, but let the central scan policy retain
        // the deferred edge. An immediately GRANTED tail needs no coordinator
        // work and no transport edge.
        if initial_state == STATE_WAITING {
            notify_coordinator();
        }

        Ok(Self {
            namespace,
            slot,
            ticket,
            liveness_path,
            liveness: Some(liveness),
            shared: Some(shared),
            _interrupt_waiter: interrupt_waiter,
            finished: false,
            preparation_tokens: None,
        })
    }

    pub(super) fn state(&self, cancelled: Option<&AtomicBool>) -> Result<State> {
        self.state_with_recovery(false, cancelled)
    }

    fn state_with_recovery(
        &self,
        allow_stalled_takeover: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<State> {
        let _namespace = self.namespace.enter();
        let _lock = lock_registry_interruptible_existing(cancelled)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        if allow_stalled_takeover {
            table.recover_coordinator_if_stalled()?;
        } else {
            table.recover_coordinator_if_dead()?;
        }
        let mut record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("live queue ticket {} disappeared", self.ticket))?;
        let acknowledgement = match record.state {
            STATE_REVOKED => table.acknowledge_revoked(self.slot, self.ticket, None)?,
            STATE_REPLAN_EXPIRED => {
                table.acknowledge_expired_replan(self.slot, self.ticket, None)?
            }
            _ => AcknowledgeResult::UNCHANGED,
        };
        if acknowledgement.acknowledged {
            record = table
                .record(self.slot)?
                .filter(|record| record.ticket == self.ticket)
                .ok_or_else(|| anyhow::anyhow!("live queue ticket {} disappeared", self.ticket))?;
        }
        let state = match record.state {
            STATE_WAITING => State::Waiting,
            STATE_GRANTED => State::Granted,
            STATE_REPLAN => State::Replan,
            STATE_COORDINATOR => State::Coordinator,
            STATE_COORDINATOR_STANDBY => State::CoordinatorStandby,
            state => anyhow::bail!("queue ticket {} has invalid state {state}", self.ticket),
        };
        drop(table);
        drop(_lock);
        if acknowledgement.notify {
            notify_coordinator();
        }
        Ok(state)
    }

    /// Acquire the registry SHARED lock for a state read WITHOUT joining the
    /// writer-intent turnstile convoy. A blocking SH read takes the sidecar
    /// EXCLUSIVELY (writer_intent_mode inverts SH->EX), so under a live
    /// registry-EX writer every pre-exec entrant's per-tick read serializes on
    /// the turnstile behind the writer's retained sidecar-SH — the observed
    /// ~990-process, 25-minute flock convoy. Instead, probe nonblocking; on a
    /// live writer, YIELD (writer-priority preserved) via a bounded futex-backed
    /// backoff and retry. The turnstile frees whenever the writer finishes its
    /// bounded transaction, so the probe succeeds on the next gap and the read
    /// always returns the authoritative state — no missed election or grant. The
    /// ticket futex wakes the retry immediately on a grant; the short timeout
    /// re-probes through a self-election that bumped no futex. A missing sidecar
    /// falls back to the blocking existing-lock so its authoritative error/None
    /// surfaces exactly as before. Crash-safety is unchanged (same fds, released
    /// by the kernel on death).
    fn lock_registry_shared_yielding(
        &self,
        cancelled: Option<&AtomicBool>,
    ) -> Result<RegistryLock> {
        loop {
            check_cancelled(cancelled)?;
            match probe_lock_registry_existing_nonblocking(FlockMode::Shared)? {
                NonblockingRegistryLock::Acquired(lock) => return Ok(lock),
                NonblockingRegistryLock::Missing => {
                    return normalize_cancellation(
                        lock_registry_existing(FlockMode::Shared),
                        cancelled,
                    );
                }
                NonblockingRegistryLock::Contended => {
                    if let Some(shared) = self.shared.as_ref() {
                        let expected = shared.expected();
                        let _ = shared.wait(expected, TURNSTILE_READ_BACKOFF)?;
                    } else {
                        std::thread::sleep(TURNSTILE_READ_BACKOFF);
                    }
                }
            }
        }
    }

    fn state_shared(
        &self,
        check_coordinator_liveness: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Option<State>> {
        let _namespace = self.namespace.enter();
        check_cancelled(cancelled)?;
        let _lock = self.lock_registry_shared_yielding(cancelled)?;
        #[cfg(test)]
        SHARED_STATE_READS.with(|reads| reads.set(reads.get() + 1));
        let shared = self
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("queue ticket shared mappings were released"))?;
        let layout = shared.validate_header(self.slot, self.ticket)?;
        let header = &shared.header;
        if atomic_u64(header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0 {
            // A killed writer left mutable state to repair. The caller drops
            // this SH lock and upgrades through the rare EX recovery path.
            return Ok(None);
        }
        let next_slot = read_u64(header, H_NEXT_SLOT);
        let bytes = shared.record_bytes(self.slot, self.ticket)?;
        let state = match read_u32(bytes, R_STATE) {
            STATE_WAITING => State::Waiting,
            STATE_GRANTED => State::Granted,
            STATE_REPLAN => State::Replan,
            STATE_COORDINATOR => State::Coordinator,
            STATE_COORDINATOR_STANDBY => State::CoordinatorStandby,
            // Revoked and expired callback publications need one EX
            // acknowledgement before becoming ordinary waiters. REVOKED
            // retires an exact fence; EXPIRED retires a stale planner token.
            STATE_REVOKED | STATE_REPLAN_EXPIRED => return Ok(None),
            state => anyhow::bail!("queue ticket {} has invalid state {state}", self.ticket),
        };
        if check_coordinator_liveness && matches!(state, State::Waiting | State::CoordinatorStandby)
        {
            let now = monotonic_now_ns()?;
            if replan_wave_requires_recovery_from_header(header, now) {
                #[cfg(test)]
                SHARED_STATE_RECOVERY_UPGRADES.with(|upgrades| upgrades.set(upgrades.get() + 1));
                return Ok(None);
            }
            let coordinator = read_u64(header, H_COORDINATOR);
            let coordinator_slot = read_u64(header, H_COORDINATOR_SLOT);
            let progress_is_live = if coordinator == 0 {
                shared_live_inflight_head(header, layout, next_slot)?
            } else {
                coordinator_slot != NONE_SLOT
                    && coordinator_slot < next_slot
                    && ticket_is_live(coordinator_slot, coordinator)?
                    && coordinator_activity_is_fresh(header, now)
            };
            if !progress_is_live {
                // Drop the SH lock before the caller takes the rare EX
                // recovery path. Healthy timeout checks remain readonly and
                // concurrent across every waiter.
                #[cfg(test)]
                SHARED_STATE_RECOVERY_UPGRADES.with(|upgrades| upgrades.set(upgrades.get() + 1));
                return Ok(None);
            }
        }
        Ok(Some(state))
    }

    // The retained-futex-wake regression test observes that a cross-process
    // grant reaches a waiter through its retained read-only MAP_SHARED view.
    // A grant advances the record state and FUTEX_WAKEs the shared word under
    // the same registry lock, so the waiter observes it EITHER by its
    // FUTEX_WAIT resolving (marked inside `RetainedWake::wait`) OR by the next
    // shared state read seeing the resolved state — which happens whenever the
    // grant lands after this call's wait budget elapsed, or before the wait was
    // entered at all. Both are the retained mapping faithfully reflecting the
    // cross-process wake; publish the same marker so the observer is not
    // stranded when coordinator latency shifts the grant out of the FUTEX_WAIT
    // window. Marker-gated (test-only); untouched when the env var is unset.
    #[cfg(test)]
    fn publish_retained_wake_resolution() -> Result<()> {
        if let Some(marker) = std::env::var_os(RETAINED_FUTEX_WAIT_MARKER_ENV) {
            std::fs::write(&marker, b"woken").with_context(|| {
                format!(
                    "publish retained-futex wake resolution marker {}",
                    Path::new(&marker).display()
                )
            })?;
        }
        Ok(())
    }

    pub(super) fn state_or_wait(
        &self,
        timeout: Duration,
        cancelled: Option<&AtomicBool>,
    ) -> Result<State> {
        let _namespace = self.namespace.enter();
        check_cancelled(cancelled)?;
        let wake = self
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("queue ticket shared mappings were released"))?;
        let expected = wake.expected();
        // Sample the futex before the single registry state read. A grant
        // between these operations changes either the observed state or the
        // futex value, so FUTEX_WAIT cannot sleep past it.
        let state = match self.state_shared(false, cancelled)? {
            Some(state) => state,
            None => self.state(cancelled)?,
        };
        if !matches!(state, State::Waiting | State::CoordinatorStandby) {
            #[cfg(test)]
            Self::publish_retained_wake_resolution()?;
            return Ok(state);
        }
        let wait_started = std::time::Instant::now();
        let timed_out = loop {
            let elapsed = wait_started.elapsed();
            if elapsed >= timeout {
                break true;
            }
            let remaining = timeout.saturating_sub(elapsed);
            let wait_for = super::super::reservation_wait_progress_poll()
                .map_or(remaining, |poll| remaining.min(poll));
            let timed_out = wake.wait(expected, wait_for)?;
            check_cancelled(cancelled)?;
            if !timed_out {
                break false;
            }
            super::super::tick_reservation_wait_progress();
        };
        if timed_out {
            // Ordinary wakes stay entirely on SH/read-only mappings. The
            // bounded crash-recovery tick alone pays for an EX coordinator
            // liveness check, so hundreds of futex waiters do not serialize
            // on every predecessor transition.
            let resolved = match self.state_shared(true, cancelled)? {
                Some(state) => state,
                None => self.state_with_recovery(true, cancelled)?,
            };
            // The wait budget elapsed, but a grant that landed meanwhile is
            // observed here through the retained shared mapping just as a
            // FUTEX_WAIT would have resolved it; keep the marker faithful.
            #[cfg(test)]
            if !matches!(resolved, State::Waiting | State::CoordinatorStandby) {
                Self::publish_retained_wake_resolution()?;
            }
            Ok(resolved)
        } else {
            Ok(state)
        }
    }

    pub(super) fn run_granted<T>(
        &mut self,
        cancelled: Option<&AtomicBool>,
        attempt: impl FnOnce(
            &ClaimSet,
            &ClaimSet,
            bool,
            AggregateSnapshot,
            AvailabilitySnapshot,
        ) -> Result<GrantAttempt<T>>,
    ) -> Result<GrantResult<T>> {
        let _namespace = self.namespace.enter();
        let (
            designated,
            watch,
            acquisition_allowed,
            predecessors,
            availability,
            callback_epoch,
            callback_serial,
            callback_snapshot_serial,
            callback_wake,
        ) = {
            let _lock = lock_registry_interruptible_existing(cancelled)?;
            let mut table = Table::open_existing()?;
            table.repair_consistency_if_needed()?;
            if table.expire_replan_wave_if_due()? {
                notify_coordinator();
            }
            let record = table
                .record(self.slot)?
                .filter(|record| record.ticket == self.ticket)
                .ok_or_else(|| anyhow::anyhow!("live queue ticket {} disappeared", self.ticket))?;
            if record.state == STATE_REVOKED {
                let acknowledgement = table.acknowledge_revoked(self.slot, self.ticket, None)?;
                drop(table);
                drop(_lock);
                if acknowledgement.notify {
                    notify_coordinator();
                }
                return Ok(GrantResult::LostGrant);
            }
            if record.state == STATE_REPLAN_EXPIRED {
                let acknowledgement =
                    table.acknowledge_expired_replan(self.slot, self.ticket, None)?;
                drop(table);
                drop(_lock);
                if acknowledgement.notify {
                    notify_coordinator();
                }
                return Ok(GrantResult::LostGrant);
            }
            if !matches!(record.state, STATE_GRANTED | STATE_REPLAN) {
                return Ok(GrantResult::LostGrant);
            }
            // Park a granted entrant behind a dirty suffix only when its
            // claim overlaps something that became newly fenceable since the
            // last authoritative scan: new fences are a subset of the
            // changed-claims accumulator and retirements only shrink
            // prefixes, so a disjoint entrant cannot be doomed by the
            // pending scan and proceeds — its probe runs now and every
            // completion path publishes its own scan edge, superseding the
            // wave-edge shortening a park would have provided.
            if record.state == STATE_GRANTED
                && table.min_changed_ticket() < record.ticket
                && (table.changed_claims_saturated()
                    || table.changed_claims_conflict(&record.claim)?)
            {
                // Never run the O(N) authoritative scan from a callback
                // entrant. An earlier non-fencing REPLAN replacement can
                // dirty this suffix without advancing the global claim
                // epoch, so the suffix watermark plus the changed-claims
                // accumulator is the complete admission fence. Park this
                // unentered token and publish one coalesced edge; the
                // coordinator consumes it in its normal scan turn.
                table.begin_transaction()?;
                table.set_record_state(self.slot, STATE_WAITING)?;
                table.clear_record_blocked(self.slot)?;
                let notify = table.schedule_grant_completion_edge_in_transaction()?;
                table.finish_transaction()?;
                drop(table);
                drop(_lock);
                if notify {
                    notify_coordinator();
                }
                return Ok(GrantResult::LostGrant);
            }
            let state_epoch = if record.state == STATE_GRANTED {
                record.grant_epoch
            } else {
                record.replan_claim_epoch
            };
            let (prefix_epoch, mut predecessors) = table.cached_prefix(record.slot)?;
            if prefix_epoch == 0 || prefix_epoch != state_epoch {
                table.begin_transaction()?;
                let replan_completion = record.state == STATE_REPLAN;
                if replan_completion {
                    table.mark_suffix_dirty_fencing(record.ticket, &record.claim)?;
                }
                table.set_record_state(self.slot, STATE_WAITING)?;
                table.clear_record_blocked(self.slot)?;
                let notify = if replan_completion {
                    table.schedule_replan_completion_edge_in_transaction()?
                } else {
                    table.schedule_grant_completion_edge_in_transaction()?
                };
                table.finish_transaction()?;
                drop(table);
                drop(_lock);
                if notify {
                    notify_coordinator();
                }
                return Ok(GrantResult::LostGrant);
            }
            // A licensed GRANTED ticket's own claim is charged while its
            // callback plans; subtract it so the soft-avoid bias cannot steer
            // the callback off its own designated footprint. REPLAN is never
            // charged and needs no exclusion.
            table.fill_grant_occupancy(
                &mut predecessors,
                (record.state == STATE_GRANTED).then_some(&record.claim),
            )?;
            let availability = table.availability_snapshot();
            let callback_wake = table
                .record_bytes(record.slot)?
                .map(|bytes| read_u32(bytes, R_WAKE))
                .ok_or_else(|| anyhow::anyhow!("live queue ticket {} disappeared", self.ticket))?;
            (
                record.claim,
                record.watch,
                record.state == STATE_GRANTED,
                predecessors,
                availability,
                prefix_epoch,
                record.issue_serial,
                // This upper-bounds every resource serial represented by the
                // callback's availability snapshot without walking a large
                // alternative watch on the live-GRANTED path. If the callback
                // returns to WAITING, any relevant later improvement must have
                // a strictly greater serial and triggers one fresh REPLAN.
                table.global_serial(),
                callback_wake,
            )
        };

        // The exact designated claim remains published while the expensive
        // host snapshot and resource-flock probe run without the registry EX
        // flock. Disjoint grants can therefore probe concurrently, while the
        // aggregate fence still excludes conflicting fast paths.
        let mut result = normalize_cancellation(
            attempt(
                &designated,
                &watch,
                acquisition_allowed,
                predecessors,
                availability,
            ),
            cancelled,
        )?;
        anyhow::ensure!(
            result.acquired.is_some() || result.preparation_claim.is_none(),
            "queue callback returned a preparation claim without physical ownership",
        );
        anyhow::ensure!(
            result.preparation_contention.is_none()
                || (result.acquired.is_none()
                    && result.preparation_claim.is_none()
                    && result.contention.is_none()),
            "queue callback mixed preparation contention with another probe result",
        );

        let lock = lock_registry_interruptible_existing(cancelled)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        if table.expire_replan_wave_if_due()? {
            notify_coordinator();
        }
        let record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("live queue ticket {} disappeared", self.ticket))?;
        // The claim physically backing `result.acquired`: a same-wake
        // re-designation (a no-license REPLAN wake that acquired a different
        // claim than the scan published) holds `next_claim`; every other
        // acquire holds the published designation. Each release site still
        // prefers a preparation footprint when the acquire produced one.
        let acquired_designation = if !acquisition_allowed && result.next_claim != designated {
            result.next_claim.clone()
        } else {
            designated.clone()
        };
        if record.state == STATE_REPLAN_EXPIRED {
            let blocked = if result.acquired.is_none() {
                table.blocked_evidence(
                    result.contention.as_ref(),
                    &watch,
                    callback_snapshot_serial,
                )?
            } else {
                None
            };
            if result.acquired.is_some() {
                // A non-acquiring REPLAN callback must never return physical
                // ownership. Keep this defensive path ordered anyway: make
                // the footprint unknown before dropping opaque caller state,
                // then acknowledge only after those OFDs are gone.
                let released_claim = result
                    .preparation_claim
                    .as_ref()
                    .unwrap_or(&acquired_designation);
                table.begin_transaction()?;
                table.mark_claim_unknown(released_claim)?;
                table.finish_transaction()?;
                drop(table);
                drop(lock);
                drop(result);

                let ack_lock = lock_registry_existing(FlockMode::Exclusive)?;
                let mut table = Table::open_existing()?;
                table.repair_consistency_if_needed()?;
                let _acknowledgement =
                    table.acknowledge_expired_replan(self.slot, self.ticket, None)?;
                drop(table);
                drop(ack_lock);
                // The acknowledgement is ordered after the opaque payload
                // closes. Always target the coordinator here: an already-set
                // RESCAN edge may have been notified while these OFDs were
                // still physically busy.
                notify_coordinator();
            } else {
                // Publish the negative witness before dropping it. The ticket
                // becomes WAITING only after this late callback is known to be
                // unable to publish its stale replacement.
                let acknowledgement =
                    table.acknowledge_expired_replan(self.slot, self.ticket, blocked)?;
                drop(table);
                drop(lock);
                drop(result);
                if acknowledgement.notify {
                    notify_coordinator();
                }
            }
            return Ok(GrantResult::LostGrant);
        }
        let dirty_suffix = acquisition_allowed && table.min_changed_ticket() < record.ticket;
        let dirty_grant = dirty_suffix
            && (table.changed_claims_saturated()
                || table.changed_claims_conflict(&record.claim)?);
        if dirty_grant && record.state == STATE_GRANTED && record.claim == designated {
            let released_acquired = result.acquired.is_some();
            let blocked = if released_acquired {
                None
            } else {
                table.blocked_evidence(
                    result.contention.as_ref(),
                    &watch,
                    callback_snapshot_serial,
                )?
            };
            table.begin_transaction()?;
            table.set_record_state(self.slot, STATE_WAITING)?;
            table.clear_record_blocked(self.slot)?;
            if released_acquired {
                let released_claim = result
                    .preparation_claim
                    .as_ref()
                    .unwrap_or(&acquired_designation);
                table.mark_claim_unknown(released_claim)?;
            } else if let Some((marker, serial, consumed_serial)) = blocked {
                table.set_record_blocked(self.slot, marker, serial)?;
                table.set_record_issue_serial(self.slot, consumed_serial)?;
                table.mark_blocker_unknown(marker)?;
            }
            let notify = table.schedule_grant_completion_edge_in_transaction()?;
            table.finish_transaction()?;
            drop(table);
            drop(lock);
            drop(result);
            // UNKNOWN cannot be observed as physically free until the opaque
            // payload closes. Preserve that drop-before-targeted-notify edge
            // even when RESCAN was already coalesced by another callback.
            if notify || released_acquired {
                notify_coordinator();
            }
            return Ok(GrantResult::LostGrant);
        }
        // Fairness fence for a same-wake re-designation. Its physical acquire
        // ran with the registry unlocked against the predecessor snapshot taken
        // above, so an earlier ticket may have changed its exact claim in that
        // window. Reuse the exact suffix watermark the GRANTED dirty path uses:
        // any earlier ticket's change advances `min_changed_ticket` below this
        // one, so release the just-acquired alternative and requeue rather than
        // committing ahead of an older waiter that may need this rotation.
        // Any unlicensed REPLAN acquire rides the suffix-watermark fence,
        // whether it re-designated onto a different claim OR re-acquired its
        // own freed designation: the candidate loop legitimately selects the
        // cell's own placement when it frees, and that acquire commits ahead of
        // an older waiter just as a re-designation would. Fence on
        // `acquired.is_some()` uniformly rather than on a claim change, so an
        // own-designation acquire behind a dirtied older ticket still releases
        // and requeues. `result.next_claim` equals `designated` in that shape,
        // so the mark_unknown release below targets the correct footprint.
        let dirty_redesignation = !acquisition_allowed
            && record.state == STATE_REPLAN
            && result.acquired.is_some()
            && table.min_changed_ticket_replan() < record.ticket;
        if dirty_redesignation {
            table.begin_transaction()?;
            // REPLAN -> WAITING drains this ring slot through set_record_state.
            table.set_record_state(self.slot, STATE_WAITING)?;
            table.clear_record_blocked(self.slot)?;
            table.mark_claim_unknown(&result.next_claim)?;
            let _edge = table.schedule_replan_completion_edge_in_transaction()?;
            table.finish_transaction()?;
            drop(table);
            drop(lock);
            drop(result);
            // Released capacity cannot be observed until the physical payload
            // closes; target the coordinator after that drop.
            notify_coordinator();
            return Ok(GrantResult::LostGrant);
        }
        if record.state == STATE_REVOKED {
            // The revocation scan deliberately kept this exact claim in every
            // later prefix. If the callback won the physical race, publish an
            // UNKNOWN observation before releasing its OFDs. Only after the
            // opaque payload is gone may acknowledgement remove that fence
            // and request the successor scan.
            let released_acquired = result.acquired.is_some();
            let blocked = if !released_acquired {
                table.blocked_evidence(
                    result.contention.as_ref(),
                    &watch,
                    callback_snapshot_serial,
                )?
            } else {
                None
            };
            if released_acquired {
                let released_claim = result
                    .preparation_claim
                    .as_ref()
                    .unwrap_or(&acquired_designation);
                table.begin_transaction()?;
                table.mark_claim_unknown(released_claim)?;
                table.finish_transaction()?;
            }
            drop(table);
            drop(lock);
            drop(result);

            let ack_lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            table.repair_consistency_if_needed()?;
            let acknowledgement = table.acknowledge_revoked(self.slot, self.ticket, blocked)?;
            drop(table);
            drop(ack_lock);
            if acknowledgement.notify || released_acquired {
                notify_coordinator();
            }
            return Ok(GrantResult::LostGrant);
        }
        let expected_state = if acquisition_allowed {
            STATE_GRANTED
        } else {
            STATE_REPLAN
        };
        let state_epoch = if acquisition_allowed {
            record.grant_epoch
        } else {
            record.replan_claim_epoch
        };
        let prefix_epoch = table.record_prefix_epoch(record.slot)?;
        let epoch_publication_changed =
            state_epoch != callback_epoch || prefix_epoch != callback_epoch;
        let issue_serial_changed = record.issue_serial != callback_serial;
        let stale = epoch_publication_changed || issue_serial_changed;
        let current_wake = table
            .record_bytes(record.slot)?
            .map(|bytes| read_u32(bytes, R_WAKE))
            .ok_or_else(|| anyhow::anyhow!("live queue ticket {} disappeared", self.ticket))?;
        let same_replan_issuance = acquisition_allowed || current_wake == callback_wake;
        let current_designated_contention =
            result
                .contention
                .as_ref()
                .is_some_and(|evidence| match evidence.blocker {
                    ResourceKey::Cpu(index) => {
                        designated.cpus.contains(&index)
                            && ClaimMode::from(evidence.mode) == designated.cpu_mode
                    }
                    ResourceKey::Llc(index) => {
                        designated.llcs.contains(&index)
                            && ClaimMode::from(evidence.mode) == designated.llc_mode
                    }
                    ResourceKey::Permit(index) => {
                        designated.permits.contains(&index)
                            && ClaimMode::from(evidence.mode) == designated.permit_mode
                    }
                });
        // A failed nonblocking flock is authoritative current negative
        // evidence even if a predecessor publication changed while the
        // callback was running. If the same exact grant is still
        // live, commit that contention at the *current* blocker serial and
        // discard any positive alternative selected from the stale snapshot.
        // Keeping the writable witness through publication orders a later
        // re-observation; dropping this result instead immediately regrants a
        // known-stale callback and can create an unbounded probe storm.
        let accept_stale_contention = stale
            && acquisition_allowed
            && record.state == STATE_GRANTED
            && record.claim == designated
            && result.acquired.is_none()
            && current_designated_contention;
        if accept_stale_contention {
            result.next_claim = designated.clone();
        }
        if record.state != expected_state
            || record.claim != designated
            || !same_replan_issuance
            || (acquisition_allowed && stale && !accept_stale_contention)
        {
            // A changed publication token means the coordinator already issued
            // a fresh callback snapshot. Preserve an unrelated new claim, but
            // revoke a same-claim grant below when this callback proved that
            // snapshot stale by physically acquiring the resource.
            let released_acquired = result.acquired.is_some();
            // A newer grant for the same exact claim may have been published
            // while the old callback physically held that claim. It was
            // issued from an availability snapshot the old payload disproves,
            // so consume that grant instead of letting this thread reacquire
            // immediately after dropping the stale payload.
            let invalidate_regrant =
                released_acquired && record.state == STATE_GRANTED && record.claim == designated;
            let return_to_waiting = invalidate_regrant;
            let mut notify_now = false;
            if released_acquired {
                table.begin_transaction()?;
            }
            if return_to_waiting {
                table.set_record_state(self.slot, STATE_WAITING)?;
                table.clear_record_blocked(self.slot)?;
            }
            if released_acquired {
                // The optimistic availability snapshot predates this physical
                // acquisition. Revoke it before releasing the payload so no
                // waiter can be regranted from the stale free snapshot. The
                // coordinator's observation turn publishes the eventual
                // availability/global wake after the payload closes; the
                // callback itself only needs one coalesced rescan edge.
                let released_claim = result
                    .preparation_claim
                    .as_ref()
                    .unwrap_or(&acquired_designation);
                table.mark_claim_unknown(released_claim)?;
                notify_now = table.schedule_grant_completion_edge_in_transaction()?;
                table.finish_transaction()?;
            }
            // Keep the resource unavailable across the registry unlock, then
            // release the stale physical payload before publishing the wake.
            // Dropping an opaque caller-owned T while holding the registry
            // fence could deadlock if its destructor re-enters admission.
            drop(table);
            drop(lock);
            drop(result);
            // A targeted coordinator notification after the payload closes is
            // required even when this completion coalesced into an existing
            // RESCAN edge: the earlier notifier could otherwise probe while
            // these physical OFDs were still held.
            if notify_now || released_acquired {
                notify_coordinator();
            }
            return Ok(GrantResult::LostGrant);
        }
        check_cancelled(cancelled)?;

        if let Some(acquired) = result.acquired.take() {
            // A no-license REPLAN wake may return a payload for EITHER a
            // re-designation (a different claim than the scan published) OR its
            // own designation freeing up first: the acquirer's candidate loop
            // includes the cell's own placement, and `try_acquire_redesignation`
            // committing it via `candidate_ready` is exactly what the
            // authoritative grant would have done. There is nothing to fence on
            // `next_claim` here — the suffix-watermark `dirty_redesignation`
            // check above already released+requeued any unlicensed acquire that
            // raced an older ticket's change. The preparation-claim invariant
            // still holds: a same-wake acquire never carries a preparation
            // footprint (that is only produced by the licensed prepare path).
            anyhow::ensure!(
                acquisition_allowed || result.preparation_claim.is_none(),
                "same-wake re-designation cannot carry a preparation claim for ticket {}",
                self.ticket
            );
            if let Some(preparation_claim) = result.preparation_claim.take() {
                match table.transition_record_to_pending(
                    &record,
                    &designated,
                    &preparation_claim,
                    &[],
                )? {
                    PendingTransition::Committed(pending_claim) => {
                        drop(table);
                        drop(lock);
                        notify_coordinator();
                        return Ok(GrantResult::Prepared(acquired, pending_claim));
                    }
                    PendingTransition::Contended(marker) => {
                        let blocked_at = table.blocker_serial(marker.blocker, marker.mode)?;

                        // A disjoint publication raced the physical
                        // preparation probe. Park on its exact registry
                        // blocker before dropping the stale OFDs; this yields
                        // the scheduling turn instead of immediately
                        // regranting the same intent in a probe storm.
                        table.begin_transaction()?;
                        table.set_record_state(self.slot, STATE_WAITING)?;
                        table.set_record_blocked(self.slot, marker, blocked_at)?;
                        table.set_record_issue_serial(
                            self.slot,
                            callback_snapshot_serial.max(blocked_at),
                        )?;
                        table.mark_claim_unknown(&preparation_claim)?;
                        table.schedule_grant_completion_edge_in_transaction()?;
                        table.finish_transaction()?;
                        drop(table);
                        drop(lock);
                        drop(acquired);
                        // Notify after the physical preparation payload closes.
                        // This is targeted transport, not a generation-futex
                        // broadcast, and remains necessary when RESCAN was
                        // already pending before this transaction.
                        notify_coordinator();
                        return Ok(GrantResult::Requeued);
                    }
                }
            }
            crash_at_for_tests("granted_acquired_before_clear");
            // The physical resource flocks in `acquired` are already held.
            // Convert this exact queue claim into a live HELD publication
            // before exposing success, preserving one uninterrupted registry
            // fence across the state transition. `next_claim` equals the
            // designation on a licensed grant and carries the re-designated
            // claim a REPLAN wake physically acquired; `promote_record_to_held`
            // adjusts the claim counts across that difference.
            table.promote_record_to_held(&record, &result.next_claim, &[])?;
            let held = HeldClaim::from_ticket(self)?;
            drop(table);
            drop(lock);
            notify_coordinator();
            cancel_granted_commit_for_tests(cancelled);
            return Ok(GrantResult::Acquired(acquired, held));
        }

        validate_claim(&result.next_claim)?;
        validate_claim_within_watch(&result.next_claim, &watch)?;
        if let Some(evidence) = result.contention.as_ref() {
            validate_contention_within_watch(&[evidence.marker()], &watch)?;
        }
        let changed = result.next_claim != designated;
        anyhow::ensure!(
            result.preparation_contention.is_none() || !changed,
            "preparation contention cannot replace the final-run designation",
        );
        // Run-claim contention only, for the reason `blocked_evidence`
        // documents. The consumed serial is folded in separately below
        // because the replacement paths take the pin and the serial apart.
        let blocked_evidence = result.contention.as_ref();
        let blocked = if let Some(evidence) = blocked_evidence {
            let marker = evidence.marker();
            Some((marker, table.blocker_serial(marker.blocker, marker.mode)?))
        } else {
            None
        };
        // The availability snapshot covers `callback_snapshot_serial`, while
        // a physical negative probe is authoritative at the resource serial
        // sampled above. Consume both observations together so WAITING does
        // not immediately re-run the same callback solely because the
        // blocker improved before the later physical probe disproved it.
        let consumed_serial = blocked.map_or(callback_snapshot_serial, |(_, serial)| {
            callback_snapshot_serial.max(serial)
        });
        // Every callback completion publishes a coalesced rescan edge in the
        // same transaction as REPLAN -> WAITING. A live REPLAN wave batches
        // only the targeted transport notification: its coordinator heartbeat
        // consumes a partial batch within one second, while the final callback
        // wakes it immediately. The edge and suffix fence remain authoritative
        // throughout.
        let notify_now;
        if changed {
            if acquisition_allowed {
                table.begin_transaction()?;
                let rescan_was_pending =
                    table.pending_flags() & (PENDING_RESCAN | PENDING_REPLAN_RESCAN) != 0;
                table.replace_claim_in_transaction(
                    self.slot,
                    record.ticket,
                    &designated,
                    &result.next_claim,
                    STATE_WAITING,
                    consumed_serial,
                    blocked,
                    false,
                    ReplacementFenceEffect::ChangesPredecessorPrefix,
                )?;
                let schedule_notify = table.schedule_grant_completion_edge_in_transaction()?;
                notify_now = !rescan_was_pending || schedule_notify;
                table.finish_transaction()?;
            } else {
                notify_now = table.replace_replan_claim_and_schedule(
                    self.slot,
                    record.ticket,
                    &designated,
                    &result.next_claim,
                    consumed_serial,
                    blocked,
                    false,
                )?;
            }
        } else {
            table.begin_transaction()?;
            if !acquisition_allowed {
                // Completing an unchanged speculative callback can still turn
                // this previously non-fencing ticket into a predecessor fence
                // when the pending scan grants it. Apply the same
                // grant-disjointness damping as a NonFencing replacement — a
                // fortiori sound with an empty delta: the dirty exists only
                // because completion makes this claim fenceable at the
                // pending scan, and disjointness from every GRANTED/REVOKED
                // charge proves that new fence can doom no in-flight grant.
                // The replan word still fences speculative acquire-commits.
                if table.claim_grant_conflicts_header(&record.claim)? {
                    table.mark_suffix_dirty_fencing(record.ticket, &record.claim)?;
                } else {
                    table.mark_replan_suffix_dirty(record.ticket);
                }
            }
            if let Some(evidence) = blocked_evidence {
                let marker = evidence.marker();
                let blocked_at = table.blocker_serial(marker.blocker, marker.mode)?;
                table.set_record_blocked(self.slot, marker, blocked_at)?;
                // `set_record_blocked` first installs an event-only watch
                // reference when preparation lies outside the final-run
                // watch. The same observation machinery can then persist the
                // UNKNOWN proof without publishing prep as run intent.
                table.mark_blocker_unknown(marker)?;
            } else {
                table.clear_record_blocked(self.slot)?;
            }
            table.set_record_state(self.slot, STATE_WAITING)?;
            table.set_record_issue_serial(self.slot, consumed_serial)?;
            notify_now = if acquisition_allowed {
                table.schedule_grant_completion_edge_in_transaction()?
            } else {
                table.schedule_replan_completion_edge_in_transaction()?
            };
            table.finish_transaction()?;
        }

        drop(table);
        drop(lock);
        // Any contention witness represented by this WAITING publication must
        // close before the sole targeted edge is observable. In particular,
        // preparation contention can live outside the immutable run watch, so
        // there may be no later watched close to repair an early busy sample.
        drop(result);
        if notify_now {
            notify_coordinator();
        }
        Ok(GrantResult::Requeued)
    }

    /// Renew the coordinator progress heartbeat and promote deferred rescan
    /// metadata when needed. Unlike [`Self::schedule`], this path never
    /// observes resources, scans grants, refreshes prefixes, or changes
    /// allocation. If another waiter already displaced this ticket,
    /// `open_coordinator_table` parks it until a later election before the
    /// heartbeat can be renewed.
    pub(super) fn heartbeat(&self, cancelled: Option<&AtomicBool>) -> Result<HeartbeatStatus> {
        self.heartbeat_at(None, cancelled)
    }

    fn heartbeat_at(
        &self,
        now: Option<u64>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<HeartbeatStatus> {
        let _namespace = self.namespace.enter();
        let mut parked = false;
        let mut on_park = || parked = true;
        let (_lock, mut table) = self.open_coordinator_table(cancelled, &mut on_park)?;
        let now = match now {
            Some(now) => now,
            None => monotonic_now_ns()?,
        };
        // A completion can publish immediately after the coordinator captured
        // a snapshot with no deferred deadline and deliberately omit transport
        // notification. The next already-scheduled heartbeat is therefore the
        // causal fallback: promote any deferred edge here rather than waiting
        // for a deadline this loop has not observed yet.
        let rescan_pending = table.promote_deferred_rescan()?;
        table.touch_coordinator_heartbeat_at(now);
        Ok(HeartbeatStatus {
            parked,
            rescan_pending,
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "one coordinator transaction combines independent event, claim, liveness, and cancellation inputs"
    )]
    pub(super) fn schedule(
        &mut self,
        coordinator_claim: Option<&ClaimSet>,
        closed_cpus: &BTreeSet<usize>,
        closed_llcs: &BTreeSet<usize>,
        closed_permits: &BTreeSet<usize>,
        overflow: bool,
        contention: &[ContentionMarker],
        closed_tickets: &[(u64, u64)],
        reobserve_watch: bool,
        reconcile_liveness_after: Option<Duration>,
        force_liveness_maintenance: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ScheduleSnapshot> {
        let _namespace = self.namespace.enter();
        let pure_release_batch = coordinator_claim.is_none()
            && (!closed_cpus.is_empty() || !closed_llcs.is_empty() || !closed_permits.is_empty())
            && !overflow
            && contention.is_empty()
            && closed_tickets.is_empty()
            && !reobserve_watch
            && reconcile_liveness_after.is_none()
            && !force_liveness_maintenance;
        if pure_release_batch
            && let Some(snapshot) = self.try_known_free_release_snapshot(
                closed_cpus,
                closed_llcs,
                closed_permits,
                cancelled,
            )?
        {
            return Ok(snapshot);
        }

        let mut on_park = || {};
        let (_lock, mut table) = self.open_coordinator_table(cancelled, &mut on_park)?;
        table.prune_dead_identities(closed_tickets)?;
        if let Some(delay) = reconcile_liveness_after {
            table.request_liveness_reconciliation(delay)?;
        }
        table.perform_liveness_sweep_if_due(force_liveness_maintenance)?;
        let record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("coordinator ticket {} disappeared", self.ticket))?;
        if record.state != STATE_COORDINATOR
            || table.coordinator_ticket() != self.ticket
            || table.coordinator_slot()? != self.slot
        {
            anyhow::bail!("ticket {} lost the queue coordinator license", self.ticket);
        }
        if let Some(claim) = coordinator_claim {
            validate_claim(claim)?;
            validate_claim_within_watch(claim, &record.watch)?;
        }
        let candidate_claim = coordinator_claim
            .cloned()
            .unwrap_or_else(|| record.claim.clone());
        validate_contention_within_watch(contention, &record.watch)?;
        let event_cpus;
        let event_llcs;
        let event_permits;
        if overflow {
            let aggregate = table.aggregate_watch()?;
            event_cpus = aggregate.cpus;
            event_llcs = aggregate.llcs;
            event_permits = aggregate.permits;
        } else {
            (event_cpus, event_llcs, event_permits) =
                table.watched_subset(closed_cpus, closed_llcs, closed_permits)?;
        }
        let claim_changed = coordinator_claim.is_some_and(|claim| *claim != record.claim);
        let mut release_plan =
            table.possible_release_plan(&event_cpus, &event_llcs, &event_permits)?;
        if reobserve_watch {
            // The coordinator installs its inotify watch before this first
            // schedule call. Re-observe only modes that could have become less
            // restrictive before that watch existed: known-free modes cannot
            // improve, while existing unknown/pending work remains durable.
            let watch = table.aggregate_watch()?;
            release_plan.extend(table.possible_release_plan(
                &watch.cpus,
                &watch.llcs,
                &watch.permits,
            )?);
        }
        let registry_changed = claim_changed || !release_plan.is_empty() || !contention.is_empty();
        if registry_changed {
            table.begin_transaction()?;
            if let Some(claim) = coordinator_claim.filter(|claim| **claim != record.claim) {
                table.replace_claim_in_transaction(
                    self.slot,
                    record.ticket,
                    &record.claim,
                    claim,
                    STATE_COORDINATOR,
                    0,
                    None,
                    false,
                    ReplacementFenceEffect::ChangesPredecessorPrefix,
                )?;
            }
            table.apply_possible_release(&release_plan)?;
            table.mark_blockers_unknown(contention)?;
            if claim_changed {
                table.advance_generation_and_wake_pending()?;
            } else {
                table.advance_generation()?;
            }
            table.finish_transaction()?;
        }
        let should_scan = table.prepare_grant_scan()?;
        let (watch, coordinator_prefix_changed) = if should_scan {
            table.grant_compatible_with_tokens(self.preparation_tokens.as_ref())?
        } else {
            (table.aggregate_watch()?, false)
        };
        let (prefix_epoch, mut predecessors) = table.cached_prefix(self.slot)?;
        // A coordinator record is never grant-charged: no self-exclusion.
        table.fill_grant_occupancy(&mut predecessors, None)?;
        let availability = table.availability_snapshot();
        let observation = table.observation_request()?;
        table.touch_coordinator_heartbeat()?;
        let heartbeat_due_in = table.coordinator_heartbeat_due_in()?;
        let deferred_rescan_due_in = table.deferred_rescan_due_in()?;
        Ok(ScheduleSnapshot {
            watch,
            candidate_claim,
            candidate_watch: record.watch,
            predecessors,
            availability,
            commit_token: CoordinatorCommitToken {
                prefix_epoch,
                coordinator_epoch: table.coordinator_epoch(),
            },
            should_step: coordinator_prefix_changed,
            observation,
            liveness_due_in: table.liveness_due_in()?,
            heartbeat_due_in,
            deferred_rescan_due_in,
        })
    }

    /// Publish one preparation-only blocker for the active coordinator.
    ///
    /// The blocker contributes to aggregate event watching when its resource
    /// is outside the ticket's immutable final-run watch (an overlapping
    /// resource reuses the existing event reference). It never changes that
    /// immutable watch, claim compatibility, or fairness.
    pub(super) fn mark_external_contention(
        &mut self,
        contention: &[ContentionMarker],
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        let _namespace = self.namespace.enter();
        anyhow::ensure!(
            contention.len() == 1,
            "preparation probe must publish exactly one external blocker",
        );
        let marker = contention[0];
        let _lock = lock_registry_interruptible_existing(cancelled)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        let record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("coordinator ticket {} disappeared", self.ticket))?;
        if record.state != STATE_COORDINATOR
            || table.coordinator_ticket() != self.ticket
            || table.coordinator_slot()? != self.slot
        {
            anyhow::bail!("ticket {} lost the queue coordinator license", self.ticket);
        }
        let blocked_at = table.blocker_serial(marker.blocker, marker.mode)?;
        table.begin_transaction()?;
        table.set_record_blocked(self.slot, marker, blocked_at)?;
        table.mark_blocker_unknown(marker)?;
        table.advance_generation()?;
        table.finish_transaction()?;
        Ok(())
    }

    fn open_coordinator_table(
        &self,
        cancelled: Option<&AtomicBool>,
        on_park: &mut impl FnMut(),
    ) -> Result<(RegistryLock, Table)> {
        let _namespace = self.namespace.enter();
        loop {
            let lock = lock_registry_interruptible_existing(cancelled)?;
            let mut table = Table::open_existing()?;
            table.repair_consistency_if_needed()?;
            if table.expire_replan_wave_if_due()? {
                notify_coordinator();
            }
            let record = table
                .record(self.slot)?
                .filter(|record| record.ticket == self.ticket)
                .ok_or_else(|| anyhow::anyhow!("coordinator ticket {} disappeared", self.ticket))?;
            if record.state == STATE_COORDINATOR
                && table.coordinator_ticket() == self.ticket
                && table.coordinator_slot()? == self.slot
            {
                return Ok((lock, table));
            }
            if record.state != STATE_COORDINATOR_STANDBY {
                anyhow::bail!("ticket {} lost the queue coordinator license", self.ticket);
            }
            on_park();
            drop(table);
            drop(lock);
            loop {
                match self.state_or_wait(Duration::from_secs(4), cancelled)? {
                    State::Coordinator => break,
                    State::CoordinatorStandby | State::Waiting => {}
                    State::Granted | State::Replan => {
                        anyhow::bail!(
                            "parked coordinator ticket {} was published as an ordinary callback",
                            self.ticket
                        );
                    }
                }
            }
        }
    }

    fn try_known_free_release_snapshot(
        &self,
        closed_cpus: &BTreeSet<usize>,
        closed_llcs: &BTreeSet<usize>,
        closed_permits: &BTreeSet<usize>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Option<ScheduleSnapshot>> {
        let _namespace = self.namespace.enter();
        check_cancelled(cancelled)?;
        // Same turnstile-yielding read as `state_shared`: the coordinator's
        // known-free fast path must not join the convoy either.
        let _lock = self.lock_registry_shared_yielding(cancelled)?;
        let shared = self
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("queue ticket shared mappings were released"))?;
        let layout = shared.validate_header(self.slot, self.ticket)?;
        let header = &shared.header;
        let now = monotonic_now_ns()?;
        if atomic_u64(header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0
            || read_u64(header, H_PENDING_FLAGS) & !PENDING_REPLAN_RESCAN != 0
            || replan_wave_requires_recovery_from_header(header, now)
            || read_u64(header, H_COORDINATOR) != self.ticket
            || read_u64(header, H_COORDINATOR_SLOT) != self.slot
            || !coordinator_activity_is_fresh(header, now)
        {
            return Ok(None);
        }
        let record_bytes = shared.record_bytes(self.slot, self.ticket)?;
        if read_u32(record_bytes, R_STATE) != STATE_COORDINATOR {
            return Ok(None);
        }

        let watch_llcs = decode_header_bitset(header, layout, B_WATCH_LLCS);
        let (watch_cpus, watch_permits) =
            split_cpu_permit_indices(decode_header_bitset(header, layout, B_WATCH_CPUS));
        let watch_llc_exclusive = decode_header_bitset(header, layout, B_WATCH_LLC_EXCLUSIVE);
        let (watch_cpu_exclusive, watch_permit_exclusive) =
            split_cpu_permit_indices(decode_header_bitset(header, layout, B_WATCH_CPU_EXCLUSIVE));
        let watch = ClaimSet::with_all_claim_modes(
            watch_llcs,
            watch_cpus,
            watch_permits,
            if watch_llc_exclusive.is_empty() {
                ClaimMode::Shared
            } else {
                ClaimMode::Exclusive
            },
            if watch_cpu_exclusive.is_empty() {
                ClaimMode::Shared
            } else {
                ClaimMode::Exclusive
            },
            if watch_permit_exclusive.is_empty() {
                ClaimMode::Shared
            } else {
                ClaimMode::Exclusive
            },
        );
        let watched_cpus = closed_cpus.intersection(&watch.cpus).copied();
        let watched_llcs = closed_llcs.intersection(&watch.llcs).copied();
        let watched_permits = closed_permits.intersection(&watch.permits).copied();
        let cpus_free = watched_cpus.into_iter().all(|cpu| {
            header_bitmap_bit(header, layout, B_CPU_KNOWN, cpu)
                && header_bitmap_bit(header, layout, B_CPU_SH_AVAILABLE, cpu)
                && (!watch_cpu_exclusive.contains(&cpu)
                    || header_bitmap_bit(header, layout, B_CPU_EX_AVAILABLE, cpu))
        });
        let llcs_free = watched_llcs.into_iter().all(|llc| {
            header_bitmap_bit(header, layout, B_LLC_KNOWN, llc)
                && header_bitmap_bit(header, layout, B_LLC_SH_AVAILABLE, llc)
                && (!watch_llc_exclusive.contains(&llc)
                    || header_bitmap_bit(header, layout, B_LLC_EX_AVAILABLE, llc))
        });
        let permits_free = watched_permits.into_iter().all(|permit| {
            let Ok(index) = permit_resource_index(permit) else {
                return false;
            };
            header_bitmap_bit(header, layout, B_CPU_KNOWN, index)
                && header_bitmap_bit(header, layout, B_CPU_SH_AVAILABLE, index)
                && (!watch_permit_exclusive.contains(&permit)
                    || header_bitmap_bit(header, layout, B_CPU_EX_AVAILABLE, index))
        });
        if !cpus_free || !llcs_free || !permits_free {
            return Ok(None);
        }
        check_cancelled(cancelled)?;
        // Only a known-free hit pays the full record decode and the
        // O(bits) snapshot copies below; the common busy-close sample
        // returns above after the cheap header-bit checks.
        let record = decode_record(record_bytes, layout, self.slot)?;
        let record_words = |which| {
            (0..layout.words)
                .map(|word| {
                    read_u64(
                        record_bytes,
                        record_bitset_offset(layout, which) + word * std::mem::size_of::<u64>(),
                    )
                })
                .collect()
        };
        let header_counts = |which| {
            (0..layout.bits)
                .map(|index| {
                    read_u32(
                        header,
                        layout.count_offset(which) + index * std::mem::size_of::<u32>(),
                    )
                })
                .collect()
        };
        let predecessors = AggregateSnapshot {
            bits: layout.bits,
            cpu_any: record_words(RB_PREFIX_CPU_ANY),
            cpu_exclusive: record_words(RB_PREFIX_CPU_EXCLUSIVE),
            llc_any: record_words(RB_PREFIX_LLC_ANY),
            llc_exclusive: record_words(RB_PREFIX_LLC_EXCLUSIVE),
            cpu_shared_holders: vec![0; layout.bits],
            cpu_exclusive_holders: vec![0; layout.bits],
            llc_shared_holders: vec![0; layout.bits],
            llc_exclusive_holders: vec![0; layout.bits],
            build_cpu_claims: vec![0; layout.bits],
            // The coordinator itself is never grant-charged, so its planning
            // view carries the whole live grant charge without exclusion.
            cpu_grant_any: header_counts(C_GRANT_CPU_ANY),
            cpu_grant_exclusive: header_counts(C_GRANT_CPU_EX),
            llc_grant_any: header_counts(C_GRANT_LLC_ANY),
            llc_grant_exclusive: header_counts(C_GRANT_LLC_EX),
        };
        let header_words = |which| {
            (0..layout.words)
                .map(|word| {
                    read_u64(
                        header,
                        layout.bitset_offset(which) + word * std::mem::size_of::<u64>(),
                    )
                })
                .collect()
        };
        let availability = AvailabilitySnapshot {
            bits: layout.bits,
            cpu_known: header_words(B_CPU_KNOWN),
            cpu_sh_available: header_words(B_CPU_SH_AVAILABLE),
            cpu_ex_available: header_words(B_CPU_EX_AVAILABLE),
            llc_known: header_words(B_LLC_KNOWN),
            llc_sh_available: header_words(B_LLC_SH_AVAILABLE),
            llc_ex_available: header_words(B_LLC_EX_AVAILABLE),
        };
        Ok(Some(ScheduleSnapshot {
            watch,
            candidate_claim: record.claim,
            candidate_watch: record.watch,
            predecessors,
            availability,
            commit_token: CoordinatorCommitToken {
                prefix_epoch: read_u64(record_bytes, R_PREFIX_EPOCH),
                coordinator_epoch: read_u64(header, H_COORDINATOR_EPOCH).max(1),
            },
            // The registry already knew these resources were free. Their
            // writable closes cannot improve the coordinator's last planning
            // snapshot and are discarded without another planner pass.
            should_step: false,
            observation: None,
            liveness_due_in: liveness_due_in_from_header(header, now),
            heartbeat_due_in: coordinator_heartbeat_due_in_from_header(header, now),
            deferred_rescan_due_in: deferred_rescan_due_in_from_header(header, now),
        }))
    }

    pub(super) fn apply_observation(
        &mut self,
        request: &ObservationRequest,
        observation: &AvailabilityObservation,
        release_proofs: impl FnOnce(),
        cancelled: Option<&AtomicBool>,
    ) -> Result<ScheduleSnapshot> {
        let _namespace = self.namespace.enter();
        let mut release_proofs = Some(release_proofs);
        let (_lock, mut table) = {
            let mut release_if_parked = || {
                if let Some(release) = release_proofs.take() {
                    release();
                }
            };
            self.open_coordinator_table(cancelled, &mut release_if_parked)?
        };
        let record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("coordinator ticket {} disappeared", self.ticket))?;
        if record.state != STATE_COORDINATOR
            || table.coordinator_ticket() != self.ticket
            || table.coordinator_slot()? != self.slot
        {
            anyhow::bail!("ticket {} lost the queue coordinator license", self.ticket);
        }
        let planner_serial_before = table.max_coordinator_planner_serial(&record)?;
        table.begin_transaction()?;
        let improved = table.apply_observation(request, observation)?;
        if improved {
            // The caller still owns every physical proof flock here. Publish
            // the admission wake while registry EX remains held, then drop
            // those proofs below before any awakened registrant can acquire
            // EX and rebuild its preparation sweep.
            table.advance_generation_and_wake_pending()?;
        }
        table.finish_transaction()?;
        let planner_serial_after = table.max_coordinator_planner_serial(&record)?;
        // Keep the registry EX fence while dropping proof flocks, then grant
        // from the state they proved. A split release/reacquire lets a fast SH
        // fenced acquirer steal the resource between proof and waiter wake.
        if let Some(release) = release_proofs.take() {
            release();
        }
        let (watch, coordinator_prefix_changed) = if table.prepare_grant_scan()? {
            table.grant_compatible_with_tokens(self.preparation_tokens.as_ref())?
        } else {
            (table.aggregate_watch()?, false)
        };
        let (prefix_epoch, mut predecessors) = table.cached_prefix(self.slot)?;
        // A coordinator record is never grant-charged: no self-exclusion.
        table.fill_grant_occupancy(&mut predecessors, None)?;
        let availability = table.availability_snapshot();
        let observation = table.observation_request()?;
        let liveness_due_in = table.liveness_due_in()?;
        table.touch_coordinator_heartbeat()?;
        let heartbeat_due_in = table.coordinator_heartbeat_due_in()?;
        let deferred_rescan_due_in = table.deferred_rescan_due_in()?;
        Ok(ScheduleSnapshot {
            watch,
            candidate_claim: record.claim,
            candidate_watch: record.watch,
            predecessors,
            availability,
            commit_token: CoordinatorCommitToken {
                prefix_epoch,
                coordinator_epoch: table.coordinator_epoch(),
            },
            should_step: planner_serial_after > planner_serial_before || coordinator_prefix_changed,
            observation,
            liveness_due_in,
            heartbeat_due_in,
            deferred_rescan_due_in,
        })
    }

    #[cfg(test)]
    pub(super) fn read_state_shared_for_tests(&self) -> Result<()> {
        self.state_shared(false, None)?
            .ok_or_else(|| anyhow::anyhow!("test state read unexpectedly required recovery"))?;
        Ok(())
    }

    #[cfg(test)]
    fn state_without_recovery_for_tests(&self) -> Result<State> {
        self.state_shared(false, None)?.ok_or_else(|| {
            anyhow::anyhow!("test state observation unexpectedly needed repair or acknowledgement")
        })
    }

    #[cfg(test)]
    pub(super) fn state_or_wait_for_tests(&self) -> Result<()> {
        self.state_or_wait(Duration::ZERO, None).map(|_| ())
    }

    #[cfg(test)]
    fn commit_token_for_tests(&self) -> Result<CoordinatorCommitToken> {
        let _namespace = self.namespace.enter();
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| {
                anyhow::anyhow!("test coordinator ticket {} disappeared", self.ticket)
            })?;
        if record.state != STATE_COORDINATOR
            || table.coordinator_ticket() != self.ticket
            || table.coordinator_slot()? != self.slot
        {
            anyhow::bail!("test ticket {} is not the queue coordinator", self.ticket);
        }
        Ok(CoordinatorCommitToken {
            prefix_epoch: record.prefix_epoch,
            coordinator_epoch: table.coordinator_epoch(),
        })
    }

    /// Synchronously remove this process's own publication.
    ///
    /// Ordinary `Drop` must remain nonblocking so teardown cannot join an EX
    /// convoy. A caller which will immediately perform a fresh nonblocking
    /// admission probe needs a stronger boundary: leaving its dead PENDING
    /// claim for liveness pruning can make that probe fence itself while a
    /// different live ticket keeps the aggregate snapshot authoritative.
    #[cfg(test)]
    pub(super) fn finish(&mut self, cancelled: Option<&AtomicBool>) -> Result<()> {
        let _namespace = self.namespace.enter();
        if self.finished {
            return Ok(());
        }
        // A slot cannot be recycled while this process still has retained
        // views of its header and record.
        self.shared.take();
        // Keep the targeted wake live through this interruptible acquisition.
        // If it fails, Drop disarms the broker before its retrying cleanup.
        self.remove_record_interruptible(cancelled)?;
        self._interrupt_waiter.take();
        self.finished = true;
        self.liveness.take();
        let _ = std::fs::remove_file(&self.liveness_path);
        notify_coordinator();
        Ok(())
    }

    /// Publish the coordinator edge only after caller-owned probe payloads
    /// and temporary proof OFDs have been destroyed. `finish_acquired` and
    /// `finish_preparation` deliberately leave this edge to their caller so a
    /// successor cannot consume UNKNOWN while the corresponding resource is
    /// still physically busy.
    pub(super) fn notify_after_coordinator_payload_drop(&self) {
        let _namespace = self.namespace.enter();
        notify_coordinator();
    }

    pub(super) fn finish_acquired(
        &mut self,
        exact: &ClaimSet,
        commit_token: CoordinatorCommitToken,
        contention: &[ContentionMarker],
        cancelled: Option<&AtomicBool>,
    ) -> Result<FinishAcquireResult> {
        let _namespace = self.namespace.enter();
        if self.finished {
            anyhow::bail!("coordinator ticket was already committed");
        }
        validate_claim(exact)?;
        let _lock = lock_registry_interruptible_existing(cancelled)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        let mut record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("coordinator ticket {} disappeared", self.ticket))?;
        validate_claim_within_watch(exact, &record.watch)?;
        validate_contention_within_watch(contention, &record.watch)?;
        if record.state == STATE_COORDINATOR_STANDBY {
            table.publish_negative_evidence(exact, contention)?;
            drop(table);
            drop(_lock);
            return Ok(FinishAcquireResult::Stale);
        }
        if record.state != STATE_COORDINATOR
            || table.coordinator_ticket() != self.ticket
            || table.coordinator_slot()? != self.slot
        {
            table.publish_negative_evidence(exact, contention)?;
            drop(table);
            drop(_lock);
            anyhow::bail!("ticket {} lost the queue coordinator license", self.ticket);
        }
        let reconciled_dirty_suffix = table.min_changed_ticket() < record.ticket
            || table.pending_flags() & (PENDING_RESCAN | PENDING_REPLAN_RESCAN) != 0;
        if reconciled_dirty_suffix {
            // This is the elected coordinator's authoritative commit path, so
            // reconcile the dirty predecessor suffix here. Exact cached-prefix
            // comparison preserves a disjoint expensive success while still
            // rejecting a genuinely stale payload below.
            table.grant_compatible_with_tokens(self.preparation_tokens.as_ref())?;
            record = table
                .record(self.slot)?
                .filter(|record| record.ticket == self.ticket)
                .ok_or_else(|| anyhow::anyhow!("coordinator ticket {} disappeared", self.ticket))?;
        }
        let refreshed_prefix_compatible = if reconciled_dirty_suffix
            && record.prefix_epoch != 0
            && record.prefix_epoch != commit_token.prefix_epoch
        {
            !table.cached_prefix(record.slot)?.1.conflicts(exact)?
        } else {
            false
        };
        let stale = commit_token.prefix_epoch == 0
            || record.prefix_epoch == 0
            || table.coordinator_epoch() != commit_token.coordinator_epoch
            || record.state != STATE_COORDINATOR
            || (record.prefix_epoch != commit_token.prefix_epoch && !refreshed_prefix_compatible);
        if stale {
            // The physical probe raced an earlier callback that changed or
            // removed its reservation. Preserve this coordinator ticket,
            // publish any exact negative evidence gathered in the same planner
            // turn, and let the next scan refresh its predecessor prefix before
            // probing again.
            table.publish_negative_evidence(exact, contention)?;
            drop(table);
            drop(_lock);
            return Ok(FinishAcquireResult::Stale);
        }
        // Keep the retained per-ticket mappings live until every stale-success
        // check above has passed. The real fds are already held; convert the
        // coordinator record into a crash-recoverable HELD publication before
        // returning them.
        table.promote_record_to_held(&record, exact, contention)?;
        let held = HeldClaim::from_ticket(self)?;
        drop(table);
        drop(_lock);
        cancel_coordinator_commit_for_tests(cancelled);
        Ok(FinishAcquireResult::Committed(held))
    }

    /// Commit a coordinator-selected intent into physical preparation
    /// ownership without ever publishing the final run claim as HELD.
    pub(super) fn finish_preparation(
        &mut self,
        selected_final: &ClaimSet,
        preparation: &ClaimSet,
        commit_token: CoordinatorCommitToken,
        contention: &[ContentionMarker],
        cancelled: Option<&AtomicBool>,
    ) -> Result<FinishPreparationResult> {
        let _namespace = self.namespace.enter();
        if self.finished {
            anyhow::bail!("coordinator ticket was already committed");
        }
        validate_claim(selected_final)?;
        validate_claim(preparation)?;
        let _lock = lock_registry_interruptible_existing(cancelled)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        let mut record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("coordinator ticket {} disappeared", self.ticket))?;
        validate_claim_within_watch(selected_final, &record.watch)?;
        validate_contention_within_watch(contention, &record.watch)?;
        if record.state == STATE_COORDINATOR_STANDBY {
            table.publish_negative_evidence(preparation, contention)?;
            drop(table);
            drop(_lock);
            return Ok(FinishPreparationResult::Stale);
        }
        if record.state != STATE_COORDINATOR
            || table.coordinator_ticket() != self.ticket
            || table.coordinator_slot()? != self.slot
        {
            table.publish_negative_evidence(preparation, contention)?;
            drop(table);
            drop(_lock);
            anyhow::bail!("ticket {} lost the queue coordinator license", self.ticket);
        }
        let reconciled_dirty_suffix = table.min_changed_ticket() < record.ticket
            || table.pending_flags() & (PENDING_RESCAN | PENDING_REPLAN_RESCAN) != 0;
        if reconciled_dirty_suffix {
            table.grant_compatible_with_tokens(self.preparation_tokens.as_ref())?;
            record = table
                .record(self.slot)?
                .filter(|record| record.ticket == self.ticket)
                .ok_or_else(|| anyhow::anyhow!("coordinator ticket {} disappeared", self.ticket))?;
        }
        let refreshed_prefix_compatible = if reconciled_dirty_suffix
            && record.prefix_epoch != 0
            && record.prefix_epoch != commit_token.prefix_epoch
        {
            !table.cached_prefix(record.slot)?.1.conflicts(preparation)?
        } else {
            false
        };
        let stale = commit_token.prefix_epoch == 0
            || record.prefix_epoch == 0
            || table.coordinator_epoch() != commit_token.coordinator_epoch
            || record.state != STATE_COORDINATOR
            || (record.prefix_epoch != commit_token.prefix_epoch && !refreshed_prefix_compatible);
        if stale {
            table.publish_negative_evidence(preparation, contention)?;
            drop(table);
            drop(_lock);
            return Ok(FinishPreparationResult::Stale);
        }
        let pending_claim = match table.transition_record_to_pending(
            &record,
            selected_final,
            preparation,
            contention,
        )? {
            PendingTransition::Committed(pending_claim) => pending_claim,
            PendingTransition::Contended(_) => {
                table.publish_negative_evidence(preparation, contention)?;
                drop(table);
                drop(_lock);
                return Ok(FinishPreparationResult::Stale);
            }
        };
        drop(table);
        drop(_lock);
        Ok(FinishPreparationResult::Committed(pending_claim))
    }

    #[cfg(test)]
    fn remove_record_interruptible(&mut self, cancelled: Option<&AtomicBool>) -> Result<()> {
        let _namespace = self.namespace.enter();
        let _lock = lock_registry_interruptible_existing(cancelled)?;
        self.remove_record_locked()
    }

    #[cfg(test)]
    fn abandon_for_tests(mut self) {
        // Model abrupt owner death: release the shared maps and liveness OFD,
        // but deliberately leave the registry record for coordinator
        // liveness pruning. Marking the Rust value finished prevents Drop's
        // opportunistic nonblocking EX cleanup from hiding that path.
        self.shared.take();
        self._interrupt_waiter.take();
        self.finished = true;
        self.liveness.take();
        let _ = std::fs::remove_file(&self.liveness_path);
        notify_coordinator();
    }

    fn remove_record_locked(&mut self) -> Result<()> {
        let _namespace = self.namespace.enter();
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        if let Some(record) = table.record(self.slot)?
            && record.ticket == self.ticket
        {
            table.remove_record(&record, false)?;
            table.advance_generation_and_wake_pending()?;
        }
        Ok(())
    }

    fn finish_removed_record(&mut self) {
        self.shared.take();
        self._interrupt_waiter.take();
        self.finished = true;
        self.liveness.take();
        let _ = std::fs::remove_file(&self.liveness_path);
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let _namespace = self.namespace.enter();
        if self.finished {
            return;
        }
        self.shared.take();
        // Destruction never joins an EX-flock convoy. Disarm the targeted
        // broker and make one nonblocking cleanup attempt; if another writer
        // owns the registry, closing liveness is the authoritative removal
        // request and the coordinator prunes this exact slot+ticket.
        self._interrupt_waiter.take();
        match try_lock_registry_existing_nonblocking(FlockMode::Exclusive) {
            Ok(Some(_lock)) => {
                if let Err(error) = self.remove_record_locked() {
                    tracing::warn!(
                        ticket = self.ticket,
                        %error,
                        "failed to remove queue ticket; liveness cleanup will prune it"
                    );
                }
            }
            Ok(None) => {}
            Err(error) => {
                tracing::warn!(
                    ticket = self.ticket,
                    %error,
                    "failed to probe queue registry cleanup; liveness cleanup will prune it"
                );
            }
        }
        self.liveness.take();
        let _ = std::fs::remove_file(&self.liveness_path);
        notify_coordinator();
        // Post-release only: names when a ticket record (an intent or a
        // coordinator seat the process still held) is finally retired on the
        // way out, and the `try_lock_registry_existing_nonblocking` above shows
        // whether that retire had to defer to liveness pruning.
        crate::vmm::exit_timing::stamp("ticket_record_retire");
    }
}

#[cfg(test)]
pub(super) fn aggregate_conflicts(candidate: &ClaimSet) -> Result<bool> {
    Ok(matches!(
        with_aggregate_fence(candidate, || Ok(()))?,
        FenceResult::Fenced
    ))
}

fn aggregate_snapshot_impl(
    required: &ClaimSet,
    nonblocking: bool,
) -> Result<Option<AggregateSnapshot>> {
    validate_claim(required)?;
    let required_layout = HeaderLayout::new(required_resource_bits(required))
        .context("aggregate snapshot exceeds admission registry capacity")?;
    loop {
        let shared = if nonblocking {
            match probe_lock_registry_existing_nonblocking(FlockMode::Shared)? {
                NonblockingRegistryLock::Acquired(shared) => shared,
                NonblockingRegistryLock::Missing => {
                    return Ok(Some(AggregateSnapshot::empty(required_layout)));
                }
                NonblockingRegistryLock::Contended => return Ok(None),
            }
        } else {
            let Some(shared) = try_lock_registry_existing(FlockMode::Shared)? else {
                return Ok(Some(AggregateSnapshot::empty(required_layout)));
            };
            shared
        };
        let path = header_path();
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Some(AggregateSnapshot::empty(required_layout)));
            }
            Err(error) => return Err(error).with_context(|| format!("open {}", path.display())),
        };
        if file.metadata()?.len() == 0 {
            return Ok(Some(AggregateSnapshot::empty(required_layout)));
        }
        let map = unsafe { Mmap::map(&file) }
            .with_context(|| format!("map queue aggregate {}", path.display()))?;
        #[cfg(test)]
        AGGREGATE_SNAPSHOT_READS.with(|reads| reads.set(reads.get() + 1));
        let layout = match HeaderLayout::validate(&map) {
            Ok(layout) => layout,
            Err(_) if map.iter().all(|byte| *byte == 0) => {
                // A creator may have died after sizing the inode but before
                // publishing the initialized image. This is still an empty
                // registry, not authority for an observer to choose the
                // process-wide resource width. Keep SH through this read and
                // leave replacement to the first ticket registrant under EX.
                return Ok(Some(AggregateSnapshot::empty(required_layout)));
            }
            Err(error) => return Err(error),
        };
        if required_layout.bits > layout.bits {
            anyhow::bail!(
                "queue registry host layout supports resource indices below {}, but this aggregate snapshot needs {}",
                layout.bits,
                required_layout.bits
            );
        }
        if atomic_u64(&map, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0 {
            drop(map);
            drop(file);
            drop(shared);
            if nonblocking {
                return Ok(None);
            }
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            table.repair_consistency_if_needed()?;
            continue;
        }
        let read_words = |which| {
            (0..layout.words)
                .map(|word| read_u64(&map, layout.bitset_offset(which) + word * 8))
                .collect()
        };
        let read_counts = |which| {
            (0..layout.bits)
                .map(|index| read_u32(&map, layout.count_offset(which) + index * 4))
                .collect()
        };
        let snapshot = AggregateSnapshot {
            bits: layout.bits,
            cpu_any: read_words(B_CLAIM_CPUS),
            cpu_exclusive: read_words(B_CLAIM_CPU_EXCLUSIVE),
            llc_any: read_words(B_CLAIM_LLC_ANY),
            llc_exclusive: read_words(B_CLAIM_LLC_EXCLUSIVE),
            cpu_shared_holders: read_counts(B_HELD_CPU_SHARED),
            cpu_exclusive_holders: read_counts(B_HELD_CPU_EXCLUSIVE),
            llc_shared_holders: read_counts(B_HELD_LLC_SHARED),
            llc_exclusive_holders: read_counts(B_HELD_LLC_EXCLUSIVE),
            build_cpu_claims: read_counts(C_BUILD_CLAIM_CPUS),
            cpu_grant_any: read_counts(C_GRANT_CPU_ANY),
            cpu_grant_exclusive: read_counts(C_GRANT_CPU_EX),
            llc_grant_any: read_counts(C_GRANT_LLC_ANY),
            llc_grant_exclusive: read_counts(C_GRANT_LLC_EX),
        };
        // The common disjoint snapshot is a pure SH/read-only operation. Only
        // a claim that is actually fenced needs any liveness work.
        if !snapshot.conflicts(required)? {
            return Ok(Some(snapshot));
        }
        let coordinator = read_u64(&map, H_COORDINATOR);
        let coordinator_slot = read_u64(&map, H_COORDINATOR_SLOT);
        let next_slot = read_u64(&map, H_NEXT_SLOT);
        let progress_is_live = if coordinator == 0 {
            shared_live_inflight_head(&map, layout, next_slot)?
        } else {
            if coordinator_slot >= next_slot {
                anyhow::bail!(
                    "queue registry v{VERSION} coordinator slot {coordinator_slot} is outside 0..{next_slot}"
                );
            }
            ticket_is_live(coordinator_slot, coordinator)?
        };
        if progress_is_live {
            return Ok(Some(snapshot));
        }
        drop(map);
        drop(file);
        drop(shared);
        if nonblocking {
            return Ok(None);
        }
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        table.recover_coordinator_if_dead()?;
        continue;
    }
}

pub(super) fn aggregate_snapshot(required: &ClaimSet) -> Result<AggregateSnapshot> {
    aggregate_snapshot_impl(required, false)?.ok_or_else(|| {
        anyhow::anyhow!("blocking aggregate snapshot unexpectedly reported registry contention")
    })
}

pub(super) fn try_aggregate_snapshot(required: &ClaimSet) -> Result<Option<AggregateSnapshot>> {
    aggregate_snapshot_impl(required, true)
}

#[cfg(test)]
thread_local! {
    static AGGREGATE_SNAPSHOT_READS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn aggregate_snapshot_read_count_for_tests() -> usize {
    AGGREGATE_SNAPSHOT_READS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(super) fn union_claims_for_tests(a: &ClaimSet, b: &ClaimSet) -> ClaimSet {
    union_claims(a, b)
}

#[cfg(test)]
pub(super) fn round_trip_claim_modes_for_tests(
    claim: &ClaimSet,
    watch: &ClaimSet,
) -> Result<(ClaimSet, ClaimSet)> {
    let max_resource = claim
        .cpus
        .iter()
        .chain(claim.llcs.iter())
        .chain(claim.permits.iter())
        .chain(watch.cpus.iter())
        .chain(watch.llcs.iter())
        .chain(watch.permits.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let layout = HeaderLayout::new(max_resource.saturating_add(1).max(1))?;
    let mut bytes = vec![0u8; layout.record_size];
    write_u32(
        &mut bytes,
        R_CLAIM_LLC_MODE,
        u32::from(claim.llc_mode == ClaimMode::Exclusive),
    );
    write_u32(
        &mut bytes,
        R_CLAIM_CPU_MODE,
        u32::from(claim.cpu_mode == ClaimMode::Exclusive),
    );
    write_u32(
        &mut bytes,
        R_WATCH_LLC_MODE,
        u32::from(watch.llc_mode == ClaimMode::Exclusive),
    );
    write_u32(
        &mut bytes,
        R_WATCH_CPU_MODE,
        u32::from(watch.cpu_mode == ClaimMode::Exclusive),
    );
    write_u32(
        &mut bytes,
        R_CLAIM_PERMIT_MODE,
        u32::from(claim.permit_mode == ClaimMode::Exclusive),
    );
    write_u32(
        &mut bytes,
        R_WATCH_PERMIT_MODE,
        u32::from(watch.permit_mode == ClaimMode::Exclusive),
    );
    write_u32(
        &mut bytes,
        R_CLAIM_CLASS,
        encode_admission_class(claim.admission_class),
    );
    write_u32(
        &mut bytes,
        R_WATCH_CLASS,
        encode_admission_class(watch.admission_class),
    );
    encode_claim(&mut bytes, layout, claim, watch)?;
    let record = decode_record(&bytes, layout, 0)?;
    Ok((record.claim, record.watch))
}

#[cfg(test)]
pub(crate) struct ScanMetadataValidationOutcome {
    pub(crate) layout_words: usize,
    pub(crate) exact_word_reads: usize,
    pub(crate) invalid_span_rejected: bool,
    pub(crate) invalid_exact_identity_rejected: bool,
    pub(crate) invalid_flags_rejected: bool,
    pub(crate) invalid_watch_identity_rejected_by_full_decode: bool,
}

#[cfg(test)]
pub(super) fn exercise_scan_metadata_validation_for_tests() -> Result<ScanMetadataValidationOutcome>
{
    let layout = HeaderLayout::new(4096)?;
    let claim = ClaimSet::with_permits(
        [20usize],
        [128usize, 129],
        1000usize..1005,
        FlockMode::Shared,
        FlockMode::Exclusive,
        FlockMode::Exclusive,
    );
    let watch = ClaimSet::with_permits(
        0usize..24,
        0usize..192,
        0usize..2121,
        FlockMode::Shared,
        FlockMode::Exclusive,
        FlockMode::Exclusive,
    );
    let mut bytes = vec![0u8; layout.record_size];
    write_u32(&mut bytes, R_STATE, STATE_WAITING);
    write_u64(&mut bytes, R_TICKET, 1);
    for (offset, mode) in [
        (R_CLAIM_LLC_MODE, claim.llc_mode),
        (R_CLAIM_CPU_MODE, claim.cpu_mode),
        (R_CLAIM_PERMIT_MODE, claim.permit_mode),
        (R_WATCH_LLC_MODE, watch.llc_mode),
        (R_WATCH_CPU_MODE, watch.cpu_mode),
        (R_WATCH_PERMIT_MODE, watch.permit_mode),
    ] {
        write_u32(&mut bytes, offset, u32::from(mode == ClaimMode::Exclusive));
    }
    write_u32(
        &mut bytes,
        R_CLAIM_CLASS,
        encode_admission_class(claim.admission_class),
    );
    write_u32(
        &mut bytes,
        R_WATCH_CLASS,
        encode_admission_class(watch.admission_class),
    );
    write_u32(
        &mut bytes,
        R_BACKFILL_CAPACITY,
        backfill_capacity_for_watch(&watch),
    );
    encode_claim(&mut bytes, layout, &claim, &watch)?;

    let reads_before = SCAN_EXACT_WORD_READS.with(std::cell::Cell::get);
    decode_scan_record(&bytes, layout, 0)?;
    let exact_word_reads = SCAN_EXACT_WORD_READS
        .with(std::cell::Cell::get)
        .saturating_sub(reads_before);

    let mut invalid_span = bytes.clone();
    write_u16(
        &mut invalid_span,
        R_CLAIM_CPU_WORD_END,
        u16::try_from(layout.words + 1).context("test layout word count does not fit u16")?,
    );
    let invalid_span_rejected = decode_scan_record(&invalid_span, layout, 0).is_err();

    let mut invalid_exact_identity = bytes.clone();
    let exact_identity = read_u64(&invalid_exact_identity, R_CLAIM_IDENTITY);
    write_u64(
        &mut invalid_exact_identity,
        R_CLAIM_IDENTITY,
        exact_identity ^ 1,
    );
    let invalid_exact_identity_rejected =
        decode_scan_record(&invalid_exact_identity, layout, 0).is_err();

    let mut invalid_flags = bytes.clone();
    let flags = read_u32(&invalid_flags, R_SCAN_FLAGS);
    write_u32(&mut invalid_flags, R_SCAN_FLAGS, flags | (1 << 31));
    let invalid_flags_rejected = decode_scan_record(&invalid_flags, layout, 0).is_err();

    let mut invalid_watch_identity = bytes;
    let watch_identity = read_u64(&invalid_watch_identity, R_WATCH_IDENTITY);
    write_u64(
        &mut invalid_watch_identity,
        R_WATCH_IDENTITY,
        watch_identity ^ 1,
    );
    let invalid_watch_identity_rejected_by_full_decode =
        decode_record(&invalid_watch_identity, layout, 0).is_err();

    Ok(ScanMetadataValidationOutcome {
        layout_words: layout.words,
        exact_word_reads,
        invalid_span_rejected,
        invalid_exact_identity_rejected,
        invalid_flags_rejected,
        invalid_watch_identity_rejected_by_full_decode,
    })
}

#[cfg(test)]
pub(super) fn exercise_shared_watch_held_metadata_for_tests() -> Result<bool> {
    let claim = ClaimSet::with_permits(
        [1usize],
        [2usize],
        [3usize],
        FlockMode::Shared,
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let mut ticket = Ticket::register(claim.clone(), claim.clone(), None)?;
    let held_decodes_with_canonical_watch = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(ticket.slot)?
            .ok_or_else(|| anyhow::anyhow!("shared-watch HELD fixture disappeared"))?;
        table.promote_record_to_held(&record, &claim, &[])?;
        table.record(ticket.slot)?.is_some_and(|record| {
            record.state == STATE_HELD
                && record.watch.is_empty()
                && record.watch.cpu_mode == ClaimMode::Exclusive
                && record.watch.llc_mode == ClaimMode::Exclusive
                && record.watch.permit_mode == ClaimMode::Exclusive
        })
    };
    ticket.finish(None)?;
    Ok(held_decodes_with_canonical_watch)
}

#[cfg(test)]
pub(super) fn probe_snapshots_for_tests(
    predecessors: &[ClaimSet],
    cpu_availability: &[(usize, Option<CpuAvailability>)],
    llc_availability: &[(usize, Option<LlcAvailability>)],
) -> Result<(AggregateSnapshot, AvailabilitySnapshot)> {
    let max_resource = predecessors
        .iter()
        .flat_map(|claim| claim.cpus.iter().chain(claim.llcs.iter()))
        .copied()
        .chain(cpu_availability.iter().map(|(index, _)| *index))
        .chain(llc_availability.iter().map(|(index, _)| *index))
        .max()
        .unwrap_or(0);
    let layout = HeaderLayout::new(max_resource.saturating_add(1).max(1))?;
    let mut prefix = AggregateSnapshot::empty(layout);
    for claim in predecessors {
        add_claim_bits(
            claim,
            &mut prefix.cpu_any,
            &mut prefix.cpu_exclusive,
            &mut prefix.llc_any,
            &mut prefix.llc_exclusive,
            layout.bits,
        )?;
    }

    let mut availability = AvailabilitySnapshot {
        bits: layout.bits,
        cpu_known: vec![0; layout.words],
        cpu_sh_available: vec![0; layout.words],
        cpu_ex_available: vec![0; layout.words],
        llc_known: vec![0; layout.words],
        llc_sh_available: vec![0; layout.words],
        llc_ex_available: vec![0; layout.words],
    };
    for &(cpu, state) in cpu_availability {
        set_snapshot_bit(
            &mut availability.cpu_known,
            cpu,
            state.is_some(),
            layout.bits,
        )?;
        set_snapshot_bit(
            &mut availability.cpu_sh_available,
            cpu,
            state.is_some_and(|state| state != CpuAvailability::ExclusiveHeld),
            layout.bits,
        )?;
        set_snapshot_bit(
            &mut availability.cpu_ex_available,
            cpu,
            state == Some(CpuAvailability::Free),
            layout.bits,
        )?;
    }
    for &(llc, state) in llc_availability {
        set_snapshot_bit(
            &mut availability.llc_known,
            llc,
            state.is_some(),
            layout.bits,
        )?;
        set_snapshot_bit(
            &mut availability.llc_sh_available,
            llc,
            state.is_some_and(|state| state != LlcAvailability::ExclusiveHeld),
            layout.bits,
        )?;
        set_snapshot_bit(
            &mut availability.llc_ex_available,
            llc,
            state == Some(LlcAvailability::Free),
            layout.bits,
        )?;
    }
    Ok((prefix, availability))
}

#[cfg(test)]
fn set_snapshot_bit(words: &mut [u64], index: usize, value: bool, bits: usize) -> Result<()> {
    if index >= bits {
        anyhow::bail!("snapshot resource index {index} exceeds {bits} bits");
    }
    let mask = 1u64 << (index % 64);
    if value {
        words[index / 64] |= mask;
    } else {
        words[index / 64] &= !mask;
    }
    Ok(())
}

pub(super) fn with_aggregate_fence<T>(
    candidate: &ClaimSet,
    run: impl FnOnce() -> Result<T>,
) -> Result<FenceResult<T>> {
    validate_claim(candidate)?;
    HeaderLayout::new(required_resource_bits(candidate))
        .context("fast-path claim exceeds admission registry capacity")?;
    let mut run = Some(run);
    let mut speculative = None;
    loop {
        let shared = match try_lock_registry_existing(FlockMode::Shared)? {
            Some(shared) => shared,
            None => {
                // Do not create registry metadata on the no-contention hot
                // path. Acquire the real resource flocks first, then close the
                // creation race by checking the registry lock a second time
                // while those flocks remain held. If a registration appeared,
                // validate it under SH below; a conflicting claim discards the
                // speculative payload and retries through the ticket path.
                speculative = Some(run.take().expect("probe runs once")()?);
                match try_lock_registry_existing(FlockMode::Shared)? {
                    Some(shared) => shared,
                    None => {
                        return Ok(FenceResult::Ran {
                            value: speculative.take().expect("speculative probe completed"),
                            watched: false,
                        });
                    }
                }
            }
        };
        let path = header_path();
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // Keep the shared registry flock across the probe: a ticket
                // registration needs EX and therefore cannot publish a claim
                // in the check-to-flock gap.
                return Ok(FenceResult::Ran {
                    value: match speculative.take() {
                        Some(value) => value,
                        None => run.take().expect("probe runs once")()?,
                    },
                    watched: false,
                });
            }
            Err(error) => return Err(error).with_context(|| format!("open {}", path.display())),
        };
        if file.metadata()?.len() == 0 {
            return Ok(FenceResult::Ran {
                value: match speculative.take() {
                    Some(value) => value,
                    None => run.take().expect("probe runs once")()?,
                },
                watched: false,
            });
        }
        let map = unsafe { Mmap::map(&file) }
            .with_context(|| format!("map queue aggregate {}", path.display()))?;
        let layout = match HeaderLayout::validate(&map) {
            Ok(layout) => layout,
            Err(_) if map.iter().all(|byte| *byte == 0) => {
                // A creator may have died before publishing the initialized
                // inode. An entirely zeroed image is unambiguously
                // unpublished, so treat it like the zero-length case while
                // retaining SH across the physical probe. Only the first
                // ticket registrant may choose and initialize the layout.
                return Ok(FenceResult::Ran {
                    value: match speculative.take() {
                        Some(value) => value,
                        None => run.take().expect("probe runs once")()?,
                    },
                    watched: false,
                });
            }
            Err(error) => return Err(error),
        };
        if atomic_u64(&map, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0 {
            drop(map);
            drop(file);
            drop(shared);
            // A writer died while authoritative records and derived
            // aggregates disagreed. Repair under EX, then retry the SH read.
            let Some(_lock) = try_lock_registry_existing_nonblocking(FlockMode::Exclusive)? else {
                // Another caller won recovery election. This caller already
                // observed a conflicting aggregate, so it can register rather
                // than queueing behind the winner merely to rediscover that.
                return Ok(FenceResult::Fenced);
            };
            let mut table = Table::open_existing()?;
            table.repair_consistency_if_needed()?;
            continue;
        }
        if aggregate_map_conflicts(&map, layout, candidate)? {
            let coordinator = read_u64(&map, H_COORDINATOR);
            let coordinator_slot = read_u64(&map, H_COORDINATOR_SLOT);
            let next_slot = read_u64(&map, H_NEXT_SLOT);
            let live_inflight_head =
                coordinator == 0 && shared_live_inflight_head(&map, layout, next_slot)?;
            if live_inflight_head {
                return Ok(FenceResult::Fenced);
            }
            if coordinator != 0 {
                if coordinator_slot >= next_slot {
                    anyhow::bail!(
                        "queue registry v{VERSION} coordinator slot {coordinator_slot} is outside \
                         0..{next_slot}"
                    );
                }
                if ticket_is_live(coordinator_slot, coordinator)? {
                    return Ok(FenceResult::Fenced);
                }
            }
            drop(map);
            drop(file);
            drop(shared);

            // Coordinator death is the one crash a fast-path fence must repair
            // itself: the sole coordinator cannot consume its own liveness
            // close. Its slot+ticket header identity makes this O(1); all other
            // ticket deaths are consumed event-by-event by the live
            // coordinator, with a rare full maintenance sweep as overflow
            // recovery.
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            table.repair_consistency_if_needed()?;
            table.recover_coordinator_if_dead()?;
            if table.claim_conflicts_aggregate(candidate)? {
                return Ok(FenceResult::Fenced);
            }
            continue;
        }
        let watched = claim_intersects_watch_map(&map, layout, candidate)?;
        let value = match speculative.take() {
            Some(value) => value,
            None => run.take().expect("probe runs once")()?,
        };
        return Ok(FenceResult::Ran { value, watched });
    }
}

/// Persist one compact, cross-process admission snapshot after a coordinator
/// has remained active long enough to be interesting. The diagnostics root is
/// already lifecycle-managed and uploaded by CI. A fixed eight-slot,
/// thirty-second ring keeps even a permanently wedged host bounded. A per-slot
/// flock serializes writers, and the largest live queue observed in each bucket
/// wins so an isolated protocol fixture cannot hide the production queue.
pub(super) fn persist_wait_diagnostic_if_enabled() {
    let Some(root) = std::env::var_os("KTSTR_BUILD_DIAGNOSTICS_DIR")
        .filter(|root| !root.is_empty())
        .map(PathBuf::from)
    else {
        return;
    };
    let unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let bucket = unix_secs / WAIT_DIAGNOSTIC_BUCKET_SECS;
    let _ = persist_wait_diagnostic(&root, bucket, unix_secs);
}

pub(super) fn persist_wait_diagnostic(root: &Path, bucket: u64, unix_secs: u64) -> Result<()> {
    std::fs::create_dir_all(root)
        .with_context(|| format!("create queue diagnostics directory {}", root.display()))?;
    let ring_slot = bucket % WAIT_DIAGNOSTIC_RING_SLOTS;
    let lock_path = root.join(format!("queue-wait-{ring_slot:02}.lock"));
    let Some(_writer) = try_flock(&lock_path, FlockMode::Exclusive)? else {
        return Ok(());
    };
    let output_path = root.join(format!("queue-wait-{ring_slot:02}.txt"));
    let Some(snapshot) = bounded_wait_diagnostic(bucket, unix_secs)? else {
        return Ok(());
    };
    let bucket_header = format!("bucket={bucket}");
    if let Ok(existing) = std::fs::read_to_string(&output_path)
        && existing.lines().next() == Some(bucket_header.as_str())
        && wait_diagnostic_active_records(&existing).unwrap_or(0) >= snapshot.active_records
    {
        return Ok(());
    }
    let mut rendered = snapshot.rendered;
    if rendered.len() > WAIT_DIAGNOSTIC_MAX_BYTES {
        rendered.truncate(WAIT_DIAGNOSTIC_MAX_BYTES);
        rendered.push_str("\ntruncated_bytes=true\n");
    }
    let temp_path = root.join(format!(
        ".queue-wait-{ring_slot:02}-{}.tmp",
        std::process::id()
    ));
    std::fs::write(&temp_path, rendered)
        .with_context(|| format!("write queue diagnostic {}", temp_path.display()))?;
    std::fs::rename(&temp_path, &output_path)
        .with_context(|| format!("publish queue diagnostic {}", output_path.display()))?;
    Ok(())
}

/// Best-effort holder accusation for the exit-timing coordinator-fallback
/// diagnostic: name a holder of each resource in `watched` (the blocked
/// coordinator claim) as `<kind>=<index>:pid=<pid>:ticket=<ticket>:state=<...>`,
/// space-joined and bounded. One nonblocking SH read so it never perturbs
/// coordinator timing — `Ok(None)` when the registry lock is busy/absent/dirty
/// (the fallback tick is still counted, just unnamed).
pub(super) fn resource_holders_nonblocking(watched: &ClaimSet) -> Result<Option<String>> {
    if watched.cpus.is_empty() && watched.llcs.is_empty() && watched.permits.is_empty() {
        return Ok(None);
    }
    let Some(_registry) = try_lock_registry_existing_nonblocking(FlockMode::Shared)? else {
        return Ok(None);
    };
    match File::open(header_path()) {
        Ok(header) => {
            if header.metadata().map(|meta| meta.len()).unwrap_or(0) == 0 {
                return Ok(None);
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("open queue registry for holder accusation"),
    }
    let mut table = Table::open_existing()?;
    if atomic_u64(&table.header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0 {
        return Ok(None);
    }
    let self_pid = std::process::id();
    let next_slot = table.next_slot()?;
    let mut slot = read_u64(&table.header, H_ACTIVE_HEAD);
    let mut visited = 0u64;
    let mut named: Vec<String> = Vec::new();
    const MAX_NAMED: usize = 8;
    while slot != NONE_SLOT && named.len() < MAX_NAMED {
        if visited >= next_slot {
            break;
        }
        let Some(record) = table.record(slot)? else {
            break;
        };
        let next_active = record.next_active;
        if record.pid != self_pid {
            let resource = record
                .claim
                .cpus
                .iter()
                .find(|cpu| watched.cpus.contains(cpu))
                .map(|cpu| format!("cpu={cpu}"))
                .or_else(|| {
                    record
                        .claim
                        .llcs
                        .iter()
                        .find(|llc| watched.llcs.contains(llc))
                        .map(|llc| format!("llc={llc}"))
                })
                .or_else(|| {
                    record
                        .claim
                        .permits
                        .iter()
                        .find(|permit| watched.permits.contains(permit))
                        .map(|permit| format!("permit={permit}"))
                });
            if let Some(resource) = resource {
                named.push(format!(
                    "{resource}:pid={}:ticket={}:state={}",
                    record.pid,
                    record.ticket,
                    record_state_name(record.state),
                ));
            }
        }
        visited += 1;
        slot = next_active;
    }
    Ok((!named.is_empty()).then(|| named.join(" ")))
}

struct WaitDiagnosticSnapshot {
    rendered: String,
    active_records: u64,
}

fn wait_diagnostic_active_records(rendered: &str) -> Option<u64> {
    rendered.split_ascii_whitespace().find_map(|field| {
        field
            .strip_prefix("active_records=")
            .and_then(|count| count.parse().ok())
    })
}

fn bounded_wait_diagnostic(bucket: u64, unix_secs: u64) -> Result<Option<WaitDiagnosticSnapshot>> {
    let Some(_registry) = try_lock_registry_existing_nonblocking(FlockMode::Shared)? else {
        return Ok(None);
    };
    let header = match File::open(header_path()) {
        Ok(header) => header,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("open queue registry for diagnostics"),
    };
    if header.metadata()?.len() == 0 {
        return Ok(None);
    }
    drop(header);
    let mut table = Table::open_existing()?;
    if atomic_u64(&table.header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0 {
        return Ok(Some(WaitDiagnosticSnapshot {
            rendered: format!(
                "bucket={bucket}\ncaptured_unix_secs={unix_secs}\n\
                 registry=dirty active_records=0\n"
            ),
            active_records: 0,
        }));
    }
    let next_slot = table.next_slot()?;
    let mut slot = read_u64(&table.header, H_ACTIVE_HEAD);
    let active_tail = read_u64(&table.header, H_ACTIVE_TAIL);
    let mut visited = 0u64;
    let mut rows = Vec::new();
    while slot != NONE_SLOT {
        if visited >= next_slot {
            if rows.len() < WAIT_DIAGNOSTIC_MAX_RECORDS {
                rows.push("active_list_cycle=true".to_owned());
            }
            break;
        }
        let next_active = if rows.len() < WAIT_DIAGNOSTIC_MAX_RECORDS {
            let Some(record) = table.record(slot)? else {
                rows.push(format!("slot={slot} record=missing"));
                break;
            };
            let state = record_state_name(record.state);
            let watch_serial = table.max_watch_serial(&record.watch)?;
            rows.push(format!(
                "slot={} ticket={} pid={} state={} claim={:?} watch={:?} blocked={:?} \
                 issue_serial={} watch_serial={} grant_epoch={} replan_epoch={} prefix_epoch={} \
                 backfill_capacity={} backfill_started_ns={}",
                record.slot,
                record.ticket,
                record.pid,
                state,
                record.claim,
                record.watch,
                record.blocked_on,
                record.issue_serial,
                watch_serial,
                record.grant_epoch,
                record.replan_claim_epoch,
                record.prefix_epoch,
                record.backfill_capacity,
                record.backfill_started_ns,
            ));
            record.next_active
        } else {
            let Some(bytes) = table.record_bytes(slot)? else {
                break;
            };
            if read_u32(bytes, R_STATE) == STATE_FREE {
                break;
            }
            read_u64(bytes, R_NEXT_ACTIVE)
        };
        visited += 1;
        slot = next_active;
    }
    let active_records = visited;
    let truncated_records = active_records > rows.len() as u64 || slot != NONE_SLOT;
    let now_ns = monotonic_now_ns()?;
    let last_progress_ns = read_u64(&table.header, H_LAST_PROGRESS_NS);
    let stalled_ns = now_ns.saturating_sub(last_progress_ns);
    Ok(Some(WaitDiagnosticSnapshot {
        rendered: format!(
            "bucket={bucket}\ncaptured_unix_secs={unix_secs}\nregistry_version={VERSION} \
         coordinator={} coordinator_slot={} coordinator_epoch={} coordinator_heartbeat_ns={} \
         last_progress_ns={} stalled_ns={} generation={} generation_wake={} claim_epoch={} min_changed_ticket={} \
         min_changed_ticket_replan={} pending_flags={:#x} replan_outstanding={} replan_capacity={} replan_cursor={} replan_horizon={} \
         replan_wave_started_ns={} replan_wave_deadline_ns={} \
         global_serial={} grant_scans={} next_slot={} active_tail={} \
         active_records={} records_rendered={} records_truncated={}\n{}\n",
            table.coordinator_ticket(),
            table.coordinator_slot()?,
            table.coordinator_epoch(),
            read_u64(&table.header, H_COORDINATOR_HEARTBEAT_NS),
            last_progress_ns,
            stalled_ns,
            table.generation(),
            table.generation_wake(),
            table.claim_epoch(),
            table.min_changed_ticket(),
            table.min_changed_ticket_replan(),
            table.pending_flags(),
            table.replan_outstanding(),
            table.replan_capacity(),
            read_u64(&table.header, H_REPLAN_CURSOR),
            read_u64(&table.header, H_REPLAN_HORIZON),
            read_u64(&table.header, H_REPLAN_WAVE_STARTED_NS),
            read_u64(&table.header, H_REPLAN_WAVE_DEADLINE_NS),
            table.global_serial(),
            read_u64(&table.header, H_GRANT_SCANS),
            next_slot,
            active_tail,
            active_records,
            rows.len(),
            truncated_records,
            rows.join("\n"),
        ),
        active_records,
    }))
}

fn record_state_name(state: u32) -> &'static str {
    match state {
        STATE_PENDING => "pending",
        STATE_REVOKED => "revoked",
        STATE_WAITING => "waiting",
        STATE_GRANTED => "granted",
        STATE_COORDINATOR => "coordinator",
        STATE_COORDINATOR_STANDBY => "coordinator-standby",
        STATE_REPLAN => "replan",
        STATE_REPLAN_EXPIRED => "replan-expired",
        STATE_HELD => "held",
        STATE_FREE => "free",
        _ => "invalid",
    }
}

#[cfg(test)]
pub(super) fn snapshot() -> Result<Vec<(u64, u32, ClaimSet)>> {
    let Some(_lock) = try_lock_registry_existing(FlockMode::Exclusive)? else {
        // The uncontended fast path deliberately creates no registry
        // metadata. Test pollers therefore observe a missing lock as the
        // authoritative empty-registry state until the first ticket publishes.
        return Ok(Vec::new());
    };
    let header = match File::open(header_path()) {
        Ok(header) => header,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error).context("open existing queue registry header"),
    };
    if header.metadata()?.len() == 0 {
        return Ok(Vec::new());
    }
    let unpublished = unsafe { Mmap::map(&header) }?;
    if unpublished.iter().all(|byte| *byte == 0) {
        return Ok(Vec::new());
    }
    drop(unpublished);
    drop(header);
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    table.prune_dead()?;
    let mut records = table.records()?;
    records.sort_by_key(|record| record.ticket);
    Ok(records
        .into_iter()
        .map(|record| (record.ticket, record.pid, record.claim))
        .collect())
}

#[cfg(test)]
pub(super) fn registration_batch_kept_initial_coordinator_for_tests(
    initial: &Ticket,
    tickets: &[Ticket],
) -> Result<bool> {
    let _namespace = initial.namespace.enter();
    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    Ok(table.coordinator_ticket() == initial.ticket
        && table.coordinator_slot()? == initial.slot
        && table
            .record(initial.slot)?
            .is_some_and(|record| record.state == STATE_COORDINATOR)
        && tickets.iter().all(|ticket| {
            table
                .record(ticket.slot)
                .ok()
                .flatten()
                .is_some_and(|record| record.state != STATE_COORDINATOR_STANDBY)
        }))
}

#[cfg(test)]
pub(super) fn exercise_replan_capacity_validation_for_tests() -> Result<(bool, bool, bool)> {
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut ticket = Ticket::register(claim.clone(), claim, None)?;
    let (preserved_by_repair, zero_rejected, oversized_rejected) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_replan_capacity_for_tests(3)?;
        atomic_u64(&table.header, H_AGGREGATE_DIRTY).store(1, Ordering::SeqCst);
        table.repair_consistency_if_needed()?;
        let preserved_by_repair = table.replan_capacity() == 3;
        write_u64(&mut table.header, H_REPLAN_CAPACITY, 0);
        let zero_rejected = HeaderLayout::validate(&table.header).is_err();
        write_u64(
            &mut table.header,
            H_REPLAN_CAPACITY,
            u64::try_from(table.layout.bits)
                .context("test registry layout does not fit capacity header")?
                .saturating_add(1),
        );
        let oversized_rejected = HeaderLayout::validate(&table.header).is_err();
        write_u64(&mut table.header, H_REPLAN_CAPACITY, 3);
        (preserved_by_repair, zero_rejected, oversized_rejected)
    };
    ticket.finish(None)?;
    Ok((preserved_by_repair, zero_rejected, oversized_rejected))
}

#[cfg(test)]
pub(super) fn exercise_generation_timeout_takeover_for_tests() -> Result<(bool, bool)> {
    let blocked_claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let disjoint_claim = ClaimSet::new(std::iter::empty(), [2usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(blocked_claim.clone(), blocked_claim.clone(), None)?;
    let mut successor = Ticket::register(blocked_claim.clone(), blocked_claim, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        write_u64(&mut table.header, H_COORDINATOR_HEARTBEAT_NS, 0);
        write_u64(&mut table.header, H_LAST_PROGRESS_NS, 0);
    }

    // A normal mutation may prune a dead identity but must not interpret a
    // stale heartbeat as permission to displace a live pre-loop owner.
    let mut mutation = Ticket::register(disjoint_claim.clone(), disjoint_claim, None)?;
    let (mutation_retained_owner, expected_generation) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        (
            table.coordinator_ticket() == coordinator.ticket
                && table.coordinator_slot()? == coordinator.slot,
            table.generation_wake(),
        )
    };

    // The no-ticket pending-admission fallback has now completed a real
    // bounded wait. Its timeout is the semantic license to transfer the stale
    // live lease to the already-published successor.
    wait_for_generation_change(expected_generation, Duration::ZERO)?;
    let timeout_transferred = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.coordinator_ticket() == successor.ticket
            && table.coordinator_slot()? == successor.slot
            && table
                .record(coordinator.slot)?
                .is_some_and(|record| record.state == STATE_COORDINATOR_STANDBY)
            && table
                .record(successor.slot)?
                .is_some_and(|record| record.state == STATE_COORDINATOR)
    };

    mutation.finish(None)?;
    successor.finish(None)?;
    coordinator.finish(None)?;
    Ok((mutation_retained_owner, timeout_transferred))
}

/// Exercise the three monotonic-worsening publication paths against the real
/// registry lifecycle. Each batch stays live while its counters are sampled,
/// proving additions advance the structural image without changing the futex
/// word; synchronous teardown then proves improvements still broadcast.
#[cfg(test)]
pub(super) fn exercise_quiet_generation_additions_for_tests(
    additions: usize,
) -> Result<QuietGenerationAdditionsOutcome> {
    anyhow::ensure!(additions > 0, "quiet-generation fixture needs additions");

    fn counters() -> Result<(u64, u32)> {
        let _lock = lock_registry_existing(FlockMode::Shared)?;
        let table = Table::open_existing()?;
        Ok((table.generation(), table.generation_wake()))
    }

    fn count_state(tickets: &[Ticket], expected: u32) -> Result<usize> {
        let _lock = lock_registry_existing(FlockMode::Shared)?;
        let mut table = Table::open_existing()?;
        tickets.iter().try_fold(0usize, |count, ticket| {
            let record = table
                .record(ticket.slot)?
                .filter(|record| record.ticket == ticket.ticket)
                .ok_or_else(|| anyhow::anyhow!("quiet-generation ticket disappeared"))?;
            Ok(count + usize::from(record.state == expected))
        })
    }

    // The high anchor fixes the registry layout and keeps one live coordinator
    // throughout. Its immutable watch also keeps the GRANTED fixture's CPU
    // observed, so every addition deterministically takes the same state path
    // instead of the first addition creating a fresh UNKNOWN observation.
    let anchor_claim = ClaimSet::new(std::iter::empty(), [2_200usize], FlockMode::Exclusive);
    let anchor_watch = ClaimSet::new(
        std::iter::empty(),
        [2_198usize, 2_200usize],
        FlockMode::Exclusive,
    );
    let mut anchor = Ticket::register(anchor_claim.clone(), anchor_watch, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, 2_198, true)?;
    }

    let pending_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [2_197usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let pending_before = counters()?;
    let mut pending = Vec::with_capacity(additions);
    for _ in 0..additions {
        match Ticket::register_pending(
            required_resource_bits(&pending_claim),
            pending_claim.clone(),
        )? {
            PendingRegistration::Registered(ticket) => pending.push(*ticket),
            PendingRegistration::Contended(_) => {
                anyhow::bail!("compatible PENDING addition unexpectedly contended")
            }
        }
    }
    let pending_after = counters()?;
    for ticket in &mut pending {
        ticket.finish(None)?;
    }
    let pending_released = counters()?;

    let waiting_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [2_200usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let waiting_before = counters()?;
    let mut waiting = Vec::with_capacity(additions);
    for _ in 0..additions {
        waiting.push(Ticket::register(
            waiting_claim.clone(),
            waiting_claim.clone(),
            None,
        )?);
    }
    let waiting_after = counters()?;
    let waiting_state_count = count_state(&waiting, STATE_WAITING)?;
    for ticket in &mut waiting {
        ticket.finish(None)?;
    }
    let waiting_released = counters()?;

    let granted_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [2_198usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let granted_before = counters()?;
    let mut granted = Vec::with_capacity(additions);
    for _ in 0..additions {
        granted.push(Ticket::register(
            granted_claim.clone(),
            granted_claim.clone(),
            None,
        )?);
    }
    let granted_after = counters()?;
    let granted_state_count = count_state(&granted, STATE_GRANTED)?;
    for ticket in &mut granted {
        ticket.finish(None)?;
    }
    let granted_released = counters()?;

    let held_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [2_199usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let held_before = counters()?;
    let mut held = Vec::with_capacity(additions);
    for _ in 0..additions {
        held.push(publish_acquired(&held_claim)?);
    }
    let held_after = counters()?;
    drop(held);
    let held_released = counters()?;

    anchor.finish(None)?;
    Ok(QuietGenerationAdditionsOutcome {
        additions,
        pending_generation_delta: pending_after.0.saturating_sub(pending_before.0),
        pending_wake_delta: pending_after.1.wrapping_sub(pending_before.1),
        pending_release_wake_delta: pending_released.1.wrapping_sub(pending_after.1),
        waiting_generation_delta: waiting_after.0.saturating_sub(waiting_before.0),
        waiting_wake_delta: waiting_after.1.wrapping_sub(waiting_before.1),
        waiting_release_wake_delta: waiting_released.1.wrapping_sub(waiting_after.1),
        waiting_state_count,
        granted_generation_delta: granted_after.0.saturating_sub(granted_before.0),
        granted_wake_delta: granted_after.1.wrapping_sub(granted_before.1),
        granted_release_wake_delta: granted_released.1.wrapping_sub(granted_after.1),
        granted_state_count,
        held_generation_delta: held_after.0.saturating_sub(held_before.0),
        held_wake_delta: held_after.1.wrapping_sub(held_before.1),
        held_release_wake_delta: held_released.1.wrapping_sub(held_after.1),
    })
}

#[cfg(test)]
pub(super) fn diagnostics_for_tests() -> Result<String> {
    let Some(_lock) = try_lock_registry_existing_nonblocking(FlockMode::Exclusive)? else {
        return Ok(if registry_lock_path().exists() {
            "registry=busy".to_owned()
        } else {
            "registry=absent".to_owned()
        });
    };
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    let records = table.records()?;
    let mut rows = Vec::with_capacity(records.len());
    for record in records {
        let state = match record.state {
            STATE_PENDING => "pending",
            STATE_REVOKED => "revoked",
            STATE_WAITING => "waiting",
            STATE_GRANTED => "granted",
            STATE_COORDINATOR => "coordinator",
            STATE_COORDINATOR_STANDBY => "coordinator-standby",
            STATE_REPLAN => "replan",
            STATE_REPLAN_EXPIRED => "replan-expired",
            STATE_HELD => "held",
            STATE_FREE => "free",
            _ => "invalid",
        };
        let watch_serial = table.max_watch_serial(&record.watch)?;
        rows.push(format!(
            "ticket={} pid={} state={} claim={:?} watch={:?} blocked={:?} \
             issue_serial={} watch_serial={} grant_epoch={} replan_epoch={} prefix_epoch={} \
             backfill_capacity={} backfill_started_ns={}",
            record.ticket,
            record.pid,
            state,
            record.claim,
            record.watch,
            record.blocked_on,
            record.issue_serial,
            watch_serial,
            record.grant_epoch,
            record.replan_claim_epoch,
            record.prefix_epoch,
            record.backfill_capacity,
            record.backfill_started_ns,
        ));
    }
    Ok(format!(
        "coordinator={} coordinator_slot={} coordinator_epoch={} coordinator_heartbeat_ns={} \
         last_progress_ns={} generation={} generation_wake={} claim_epoch={} min_changed_ticket={} \
         min_changed_ticket_replan={} pending_flags={:#x} replan_outstanding={} replan_capacity={} replan_cursor={} \
         replan_horizon={} global_serial={} grant_scans={}; [{}]",
        table.coordinator_ticket(),
        table.coordinator_slot()?,
        table.coordinator_epoch(),
        read_u64(&table.header, H_COORDINATOR_HEARTBEAT_NS),
        read_u64(&table.header, H_LAST_PROGRESS_NS),
        table.generation(),
        table.generation_wake(),
        table.claim_epoch(),
        table.min_changed_ticket(),
        table.min_changed_ticket_replan(),
        table.pending_flags(),
        table.replan_outstanding(),
        table.replan_capacity(),
        read_u64(&table.header, H_REPLAN_CURSOR),
        read_u64(&table.header, H_REPLAN_HORIZON),
        table.global_serial(),
        read_u64(&table.header, H_GRANT_SCANS),
        rows.join("; "),
    ))
}

#[cfg(test)]
pub(crate) struct CoordinatorHeartbeatDeadlineOutcome {
    pub(crate) healthy_retained_before_deadline: bool,
    pub(crate) takeover_at_deadline: bool,
    pub(crate) displaced_coordinator_parked: bool,
    pub(crate) heartbeat_advanced: bool,
    pub(crate) heartbeat_tick_preserved_protocol_state: bool,
}

#[cfg(test)]
pub(crate) struct RepeatedCoordinatorTakeoverOutcome {
    pub(crate) first_header_transferred: bool,
    pub(crate) first_states_coherent: bool,
    pub(crate) first_epoch_advanced: bool,
    pub(crate) first_targeted_wakes: bool,
    pub(crate) first_scan_cleared_suffix: bool,
    pub(crate) second_header_returned: bool,
    pub(crate) second_states_coherent: bool,
    pub(crate) second_epoch_advanced: bool,
    pub(crate) second_targeted_wakes: bool,
    pub(crate) second_watermark_starts_at_older: bool,
    pub(crate) intervening_grant_callback_suppressed: bool,
    pub(crate) intervening_grant_waiting_before_scan: bool,
    pub(crate) intervening_grant_regranted_after_scan: bool,
    pub(crate) state_only_generation_wake_unchanged: bool,
}

#[cfg(test)]
pub(crate) struct FreshWaitingCoordinatorTakeoverOutcome {
    pub(crate) first_transfer_to_b: bool,
    pub(crate) first_states_coherent: bool,
    pub(crate) first_epoch_advanced: bool,
    pub(crate) first_wakes_target_only_a_and_b: bool,
    pub(crate) second_transfer_to_c: bool,
    pub(crate) second_states_coherent: bool,
    pub(crate) second_epoch_advanced: bool,
    pub(crate) second_wakes_target_only_b_and_c: bool,
    pub(crate) state_only_generation_wake_unchanged: bool,
}

#[cfg(test)]
pub(super) fn exercise_coordinator_heartbeat_deadline_for_tests()
-> Result<CoordinatorHeartbeatDeadlineOutcome> {
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(claim.clone(), claim.clone(), None)?;
    let mut successor = Ticket::register(claim.clone(), claim, None)?;

    let protocol_image = |table: &mut Table| -> Result<_> {
        let mut header = table.header.to_vec();
        header[H_COORDINATOR_HEARTBEAT_NS..H_COORDINATOR_HEARTBEAT_NS + 8].fill(0);
        let mut records = Vec::new();
        for slot in 0..table.next_slot()? {
            records.push(
                table
                    .record_bytes(slot)?
                    .ok_or_else(|| anyhow::anyhow!("heartbeat test slot {slot} disappeared"))?
                    .to_vec(),
            );
        }
        Ok((header, records))
    };

    let (signature_before_tick, heartbeat_before, first_tick_at) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        anyhow::ensure!(
            table.coordinator_ticket() == coordinator.ticket
                && table.coordinator_slot()? == coordinator.slot,
            "heartbeat-deadline fixture did not elect its first ticket",
        );
        (
            protocol_image(&mut table)?,
            read_u64(&table.header, H_COORDINATOR_HEARTBEAT_NS),
            monotonic_now_ns()?.saturating_add(1_000_000).max(1),
        )
    };
    let second_tick_at = first_tick_at.saturating_add(COORDINATOR_HEARTBEAT_INTERVAL_NS);
    let first_tick = coordinator.heartbeat_at(Some(first_tick_at), None)?;
    let second_tick = coordinator.heartbeat_at(Some(second_tick_at), None)?;
    let (signature_after_tick, heartbeat_after) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        (
            protocol_image(&mut table)?,
            read_u64(&table.header, H_COORDINATOR_HEARTBEAT_NS),
        )
    };

    let deadline = second_tick_at.saturating_add(COORDINATOR_HEARTBEAT_LEASE_NS);
    let (healthy_retained_before_deadline, takeover_at_deadline, displaced_coordinator_parked) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        // Prove that generic queue progress cannot mask a missed coordinator
        // heartbeat. At the exact heartbeat boundary this progress stamp is
        // still fresh, but the coordinator must nevertheless transfer.
        write_u64(
            &mut table.header,
            H_LAST_PROGRESS_NS,
            deadline.saturating_sub(1),
        );
        table.recover_coordinator_if_stalled_at(deadline.saturating_sub(1))?;
        let healthy = table.coordinator_ticket() == coordinator.ticket
            && table.coordinator_slot()? == coordinator.slot
            && table.record(coordinator.slot)?.is_some_and(|record| {
                record.ticket == coordinator.ticket && record.state == STATE_COORDINATOR
            });

        table.recover_coordinator_if_stalled_at(deadline)?;
        let takeover = table.coordinator_ticket() == successor.ticket
            && table.coordinator_slot()? == successor.slot
            && table.record(successor.slot)?.is_some_and(|record| {
                record.ticket == successor.ticket && record.state == STATE_COORDINATOR
            });
        let parked = table.record(coordinator.slot)?.is_some_and(|record| {
            record.ticket == coordinator.ticket && record.state == STATE_COORDINATOR_STANDBY
        });
        (healthy, takeover, parked)
    };

    successor.finish(None)?;
    coordinator.finish(None)?;
    Ok(CoordinatorHeartbeatDeadlineOutcome {
        healthy_retained_before_deadline,
        takeover_at_deadline,
        displaced_coordinator_parked,
        heartbeat_advanced: !first_tick.parked
            && !second_tick.parked
            && !first_tick.rescan_pending
            && !second_tick.rescan_pending
            && heartbeat_after == second_tick_at
            && heartbeat_after != heartbeat_before,
        heartbeat_tick_preserved_protocol_state: signature_after_tick == signature_before_tick,
    })
}

/// Prefer a fresh WAITING successor over recycling an older live standby.
/// This is the forward-drain policy paired with the standby fallback exercised
/// below: A -> B parks A, then stale B must promote waiting C rather than
/// immediately returning the lease to A.
#[cfg(test)]
pub(super) fn exercise_fresh_waiting_coordinator_takeover_for_tests()
-> Result<FreshWaitingCoordinatorTakeoverOutcome> {
    let claim_a = ClaimSet::new(std::iter::empty(), [11usize], FlockMode::Exclusive);
    let mut coordinator_a = Ticket::register(claim_a.clone(), claim_a, None)?;
    let claim_b = ClaimSet::new(std::iter::empty(), [12usize], FlockMode::Exclusive);
    let mut coordinator_b = Ticket::register(claim_b.clone(), claim_b, None)?;
    let claim_c = ClaimSet::new(std::iter::empty(), [13usize], FlockMode::Exclusive);
    let mut coordinator_c = Ticket::register(claim_c.clone(), claim_c, None)?;

    let (initial_epoch, initial_generation_wake, wakes_before) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        // Registration exercises real liveness recovery. On a heavily
        // descheduled test process, more than one coordinator lease may pass
        // while B and C are being registered, so transactionally restore the
        // synthetic A -> B -> C starting point before testing explicit lease
        // boundaries below. Do not make fixture setup depend on wall time.
        table.begin_transaction()?;
        for ticket in [&coordinator_a, &coordinator_b, &coordinator_c] {
            let record = table
                .record(ticket.slot)?
                .filter(|record| record.ticket == ticket.ticket)
                .ok_or_else(|| anyhow::anyhow!("fresh-waiter fixture ticket disappeared"))?;
            table.set_record_state(record.slot, STATE_WAITING)?;
            table.clear_record_blocked(record.slot)?;
        }
        table.set_coordinator(0, NONE_SLOT)?;
        table.elect_coordinator_in_transaction()?;
        table.finish_transaction()?;
        anyhow::ensure!(
            table.coordinator_ticket() == coordinator_a.ticket
                && table.coordinator_slot()? == coordinator_a.slot
                && table.record(coordinator_a.slot)?.is_some_and(|record| {
                    record.ticket == coordinator_a.ticket && record.state == STATE_COORDINATOR
                })
                && table.record(coordinator_b.slot)?.is_some_and(|record| {
                    record.ticket == coordinator_b.ticket && record.state == STATE_WAITING
                })
                && table.record(coordinator_c.slot)?.is_some_and(|record| {
                    record.ticket == coordinator_c.ticket && record.state == STATE_WAITING
                }),
            "fresh-waiter takeover fixture did not stage A coordinator before B and C waiters",
        );
        let wake = |ticket: &Ticket, label: &str| -> Result<u32> {
            Ok(ticket
                .shared
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("coordinator {label} wake mapping disappeared"))?
                .expected())
        };
        (
            table.coordinator_epoch(),
            table.generation_wake(),
            (
                wake(&coordinator_a, "A")?,
                wake(&coordinator_b, "B")?,
                wake(&coordinator_c, "C")?,
            ),
        )
    };

    let first_now = monotonic_now_ns()?.max(COORDINATOR_HEARTBEAT_LEASE_NS);
    let (first_transfer_to_b, first_states_coherent, first_epoch) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        write_u64(&mut table.header, H_COORDINATOR_HEARTBEAT_NS, 0);
        write_u64(&mut table.header, H_LAST_PROGRESS_NS, 0);
        table.recover_coordinator_if_stalled_at(first_now)?;
        (
            table.coordinator_ticket() == coordinator_b.ticket
                && table.coordinator_slot()? == coordinator_b.slot,
            table.record(coordinator_a.slot)?.is_some_and(|record| {
                record.ticket == coordinator_a.ticket && record.state == STATE_COORDINATOR_STANDBY
            }) && table.record(coordinator_b.slot)?.is_some_and(|record| {
                record.ticket == coordinator_b.ticket && record.state == STATE_COORDINATOR
            }) && table.record(coordinator_c.slot)?.is_some_and(|record| {
                record.ticket == coordinator_c.ticket && record.state == STATE_WAITING
            }),
            table.coordinator_epoch(),
        )
    };
    let wakes_after_first = (
        coordinator_a
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("parked coordinator A wake mapping disappeared"))?
            .expected(),
        coordinator_b
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("promoted coordinator B wake mapping disappeared"))?
            .expected(),
        coordinator_c
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("waiting coordinator C wake mapping disappeared"))?
            .expected(),
    );

    let second_now = first_now.saturating_add(COORDINATOR_HEARTBEAT_LEASE_NS);
    let (second_transfer_to_c, second_states_coherent, second_epoch, final_generation_wake) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        write_u64(&mut table.header, H_COORDINATOR_HEARTBEAT_NS, 0);
        write_u64(&mut table.header, H_LAST_PROGRESS_NS, 0);
        table.recover_coordinator_if_stalled_at(second_now)?;
        (
            table.coordinator_ticket() == coordinator_c.ticket
                && table.coordinator_slot()? == coordinator_c.slot,
            table.record(coordinator_a.slot)?.is_some_and(|record| {
                record.ticket == coordinator_a.ticket && record.state == STATE_COORDINATOR_STANDBY
            }) && table.record(coordinator_b.slot)?.is_some_and(|record| {
                record.ticket == coordinator_b.ticket && record.state == STATE_COORDINATOR_STANDBY
            }) && table.record(coordinator_c.slot)?.is_some_and(|record| {
                record.ticket == coordinator_c.ticket && record.state == STATE_COORDINATOR
            }),
            table.coordinator_epoch(),
            table.generation_wake(),
        )
    };
    let wakes_after_second = (
        coordinator_a
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("standby coordinator A wake mapping disappeared"))?
            .expected(),
        coordinator_b
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("parked coordinator B wake mapping disappeared"))?
            .expected(),
        coordinator_c
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("promoted coordinator C wake mapping disappeared"))?
            .expected(),
    );

    coordinator_c.finish(None)?;
    coordinator_b.finish(None)?;
    coordinator_a.finish(None)?;
    Ok(FreshWaitingCoordinatorTakeoverOutcome {
        first_transfer_to_b,
        first_states_coherent,
        first_epoch_advanced: first_epoch > initial_epoch,
        first_wakes_target_only_a_and_b: wakes_after_first.0 != wakes_before.0
            && wakes_after_first.1 != wakes_before.1
            && wakes_after_first.2 == wakes_before.2,
        second_transfer_to_c,
        second_states_coherent,
        second_epoch_advanced: second_epoch > first_epoch,
        second_wakes_target_only_b_and_c: wakes_after_second.0 == wakes_after_first.0
            && wakes_after_second.1 != wakes_after_first.1
            && wakes_after_second.2 != wakes_after_first.2,
        state_only_generation_wake_unchanged: final_generation_wake == initial_generation_wake,
    })
}

/// A dead WAITING record retains ticket priority until liveness recovery sees
/// it, but it must never receive a live coordinator lease ahead of the next
/// live waiter.
#[cfg(test)]
pub(super) fn exercise_dead_waiter_takeover_skip_for_tests() -> Result<(bool, bool, bool)> {
    let claim_a = ClaimSet::new(std::iter::empty(), [21usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(claim_a.clone(), claim_a, None)?;
    let dead_claim = ClaimSet::new(std::iter::empty(), [22usize], FlockMode::Exclusive);
    let dead = Ticket::register(dead_claim.clone(), dead_claim, None)?;
    let dead_slot = dead.slot;
    let dead_ticket = dead.ticket;
    let live_claim = ClaimSet::new(std::iter::empty(), [23usize], FlockMode::Exclusive);
    let mut live = Ticket::register(live_claim.clone(), live_claim, None)?;
    dead.abandon_for_tests();

    let wake_a_before = coordinator
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("dead-skip coordinator wake mapping disappeared"))?
        .expected();
    let wake_live_before = live
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("dead-skip live waiter wake mapping disappeared"))?
        .expected();
    let (transferred_to_live, coherent_states, dead_skipped) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        anyhow::ensure!(
            table.coordinator_ticket() == coordinator.ticket
                && table.record(dead_slot)?.is_some_and(|record| {
                    record.ticket == dead_ticket && record.state == STATE_WAITING
                })
                && table.record(live.slot)?.is_some_and(|record| {
                    record.ticket == live.ticket && record.state == STATE_WAITING
                }),
            "dead-skip fixture did not stage a dead waiter before its live successor",
        );
        write_u64(&mut table.header, H_COORDINATOR_HEARTBEAT_NS, 0);
        write_u64(&mut table.header, H_LAST_PROGRESS_NS, 0);
        table.recover_coordinator_if_stalled_at(
            monotonic_now_ns()?.max(COORDINATOR_HEARTBEAT_LEASE_NS),
        )?;
        let transferred_to_live =
            table.coordinator_ticket() == live.ticket && table.coordinator_slot()? == live.slot;
        let coherent_states = table.record(coordinator.slot)?.is_some_and(|record| {
            record.ticket == coordinator.ticket && record.state == STATE_COORDINATOR_STANDBY
        }) && table.record(live.slot)?.is_some_and(|record| {
            record.ticket == live.ticket && record.state == STATE_COORDINATOR
        });
        let dead_skipped = table.coordinator_ticket() != dead_ticket
            && table
                .record(dead_slot)?
                .is_none_or(|record| record.state != STATE_COORDINATOR);
        table.prune_dead()?;
        (transferred_to_live, coherent_states, dead_skipped)
    };
    let targeted_wakes = coordinator
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("parked dead-skip coordinator wake mapping disappeared"))?
        .expected()
        != wake_a_before
        && live
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("promoted live waiter wake mapping disappeared"))?
            .expected()
            != wake_live_before;

    live.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        transferred_to_live && coherent_states,
        dead_skipped,
        targeted_wakes,
    ))
}

/// Exercise two consecutive live-coordinator lease transfers. Ticket C is
/// deliberately registered between A and B and kept GRANTED: B -> A must
/// dirty the suffix from the older successor A, not merely from the displaced
/// B, or C could enter with a predecessor snapshot from the previous lease.
#[cfg(test)]
pub(super) fn exercise_repeated_coordinator_takeover_for_tests()
-> Result<RepeatedCoordinatorTakeoverOutcome> {
    // A's claim deliberately OVERLAPS the intervening C grant (CPU 2): while
    // A is coordinator its claim never fences (CPU 1 stays busy, so A is not
    // acquisition-viable), and while A is standby it never fences either —
    // so C is granted on CPU 2 with A's claim absent from its prefix. The
    // second lease transfer re-fences A's claim, a genuinely NEW fence
    // against C's grant token, which the changed-claims accumulator must
    // recognize as an overlap and park C until the authoritative scan.
    let coordinator_a_claim =
        ClaimSet::new(std::iter::empty(), [1usize, 2usize], FlockMode::Exclusive);
    let mut coordinator_a =
        Ticket::register(coordinator_a_claim.clone(), coordinator_a_claim, None)?;
    let intervening_claim = ClaimSet::new(std::iter::empty(), [2usize], FlockMode::Exclusive);
    let mut intervening =
        Ticket::register(intervening_claim.clone(), intervening_claim.clone(), None)?;
    let coordinator_b_claim = ClaimSet::new(std::iter::empty(), [3usize], FlockMode::Exclusive);
    let mut coordinator_b =
        Ticket::register(coordinator_b_claim.clone(), coordinator_b_claim, None)?;

    let (initial_epoch, initial_generation_wake, wake_a_before, wake_b_before) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        set_cpu_free_for_tests(&mut table, 1, false)?;
        set_cpu_free_for_tests(&mut table, 2, true)?;
        set_cpu_free_for_tests(&mut table, 3, false)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table.coordinator_ticket() == coordinator_a.ticket
                && table.coordinator_slot()? == coordinator_a.slot
                && table.record(coordinator_a.slot)?.is_some_and(|record| {
                    record.ticket == coordinator_a.ticket && record.state == STATE_COORDINATOR
                })
                && table.record(intervening.slot)?.is_some_and(|record| {
                    record.ticket == intervening.ticket && record.state == STATE_GRANTED
                })
                && table.record(coordinator_b.slot)?.is_some_and(|record| {
                    record.ticket == coordinator_b.ticket && record.state == STATE_WAITING
                }),
            "repeated-takeover fixture did not stage A coordinator, intervening C grant, and B waiter",
        );
        let wake_a = coordinator_a
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("coordinator A wake mapping disappeared"))?
            .expected();
        let wake_b = coordinator_b
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("coordinator B wake mapping disappeared"))?
            .expected();
        (
            table.coordinator_epoch(),
            table.generation_wake(),
            wake_a,
            wake_b,
        )
    };

    let first_now = monotonic_now_ns()?.max(COORDINATOR_HEARTBEAT_LEASE_NS);
    let (first_header_transferred, first_states_coherent, first_epoch, first_suffix_clear) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        write_u64(&mut table.header, H_COORDINATOR_HEARTBEAT_NS, 0);
        write_u64(&mut table.header, H_LAST_PROGRESS_NS, 0);
        table.recover_coordinator_if_stalled_at(first_now)?;
        let first_header_transferred = table.coordinator_ticket() == coordinator_b.ticket
            && table.coordinator_slot()? == coordinator_b.slot;
        let first_states_coherent = table.record(coordinator_a.slot)?.is_some_and(|record| {
            record.ticket == coordinator_a.ticket && record.state == STATE_COORDINATOR_STANDBY
        }) && table.record(coordinator_b.slot)?.is_some_and(|record| {
            record.ticket == coordinator_b.ticket && record.state == STATE_COORDINATOR
        }) && table.record(intervening.slot)?.is_some_and(|record| {
            record.ticket == intervening.ticket && record.state == STATE_GRANTED
        });
        let first_epoch = table.coordinator_epoch();

        // Clear the first transfer's suffix edge and refresh C's grant token.
        // The second transfer must independently dirty C from A's older ticket.
        set_cpu_free_for_tests(&mut table, 2, true)?;
        table.grant_compatible()?;
        let first_suffix_clear = table.min_changed_ticket() == u64::MAX
            && table.pending_flags() & PENDING_RESCAN == 0
            && table.record(intervening.slot)?.is_some_and(|record| {
                record.ticket == intervening.ticket && record.state == STATE_GRANTED
            });
        (
            first_header_transferred,
            first_states_coherent,
            first_epoch,
            first_suffix_clear,
        )
    };
    let wake_a_after_first = coordinator_a
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("parked coordinator A wake mapping disappeared"))?
        .expected();
    let wake_b_after_first = coordinator_b
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("promoted coordinator B wake mapping disappeared"))?
        .expected();

    let second_now = first_now.saturating_add(COORDINATOR_HEARTBEAT_LEASE_NS);
    let (
        second_header_returned,
        second_states_coherent,
        second_epoch,
        second_watermark_starts_at_older,
    ) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        write_u64(&mut table.header, H_COORDINATOR_HEARTBEAT_NS, 0);
        write_u64(&mut table.header, H_LAST_PROGRESS_NS, 0);
        table.recover_coordinator_if_stalled_at(second_now)?;
        (
            table.coordinator_ticket() == coordinator_a.ticket
                && table.coordinator_slot()? == coordinator_a.slot,
            table.record(coordinator_a.slot)?.is_some_and(|record| {
                record.ticket == coordinator_a.ticket && record.state == STATE_COORDINATOR
            }) && table.record(coordinator_b.slot)?.is_some_and(|record| {
                record.ticket == coordinator_b.ticket && record.state == STATE_COORDINATOR_STANDBY
            }) && table.record(intervening.slot)?.is_some_and(|record| {
                record.ticket == intervening.ticket && record.state == STATE_GRANTED
            }),
            table.coordinator_epoch(),
            table.min_changed_ticket() == coordinator_a.ticket
                && table.pending_flags() & PENDING_RESCAN != 0,
        )
    };
    let wake_a_after_second = coordinator_a
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("restored coordinator A wake mapping disappeared"))?
        .expected();
    let wake_b_after_second = coordinator_b
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("parked coordinator B wake mapping disappeared"))?
        .expected();

    let mut intervening_callback_ran = false;
    let intervening_result = intervening.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            intervening_callback_ran = true;
            anyhow::ensure!(
                acquisition_allowed && current == &intervening_claim,
                "intervening grant received the wrong repeated-takeover publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: current.clone(),
                contention: None,
            })
        },
    )?;
    let intervening_grant_callback_suppressed =
        !intervening_callback_ran && matches!(intervening_result, GrantResult::LostGrant);
    let (
        intervening_grant_waiting_before_scan,
        intervening_grant_regranted_after_scan,
        final_generation_wake,
    ) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let waiting = table.record(intervening.slot)?.is_some_and(|record| {
            record.ticket == intervening.ticket && record.state == STATE_WAITING
        });
        set_cpu_free_for_tests(&mut table, 2, true)?;
        table.grant_compatible()?;
        let regranted = table.record(intervening.slot)?.is_some_and(|record| {
            record.ticket == intervening.ticket && record.state == STATE_GRANTED
        });
        (waiting, regranted, table.generation_wake())
    };

    coordinator_b.finish(None)?;
    intervening.finish(None)?;
    coordinator_a.finish(None)?;
    Ok(RepeatedCoordinatorTakeoverOutcome {
        first_header_transferred,
        first_states_coherent,
        first_epoch_advanced: first_epoch > initial_epoch,
        first_targeted_wakes: wake_a_after_first != wake_a_before
            && wake_b_after_first != wake_b_before,
        first_scan_cleared_suffix: first_suffix_clear,
        second_header_returned,
        second_states_coherent,
        second_epoch_advanced: second_epoch > first_epoch,
        second_targeted_wakes: wake_a_after_second != wake_a_after_first
            && wake_b_after_second != wake_b_after_first,
        second_watermark_starts_at_older,
        intervening_grant_callback_suppressed,
        intervening_grant_waiting_before_scan,
        intervening_grant_regranted_after_scan,
        state_only_generation_wake_unchanged: final_generation_wake == initial_generation_wake,
    })
}

#[cfg(test)]
pub(super) fn expire_coordinator_lease_for_tests() -> Result<()> {
    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    anyhow::ensure!(
        table.coordinator_ticket() != 0,
        "cannot expire an absent test coordinator"
    );
    write_u64(&mut table.header, H_COORDINATOR_HEARTBEAT_NS, 0);
    write_u64(&mut table.header, H_LAST_PROGRESS_NS, 0);
    Ok(())
}

/// Transfer the current coordinator to a live waiter while leaving an already
/// pending rescan edge in place. Optionally demote the displaced ticket from
/// STANDBY to WAITING to exercise the lost-license error path rather than the
/// ordinary parked-coordinator stale path.
#[cfg(test)]
pub(super) fn force_coordinator_commit_race_for_tests(lost_license: bool) -> Result<bool> {
    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    let displaced_ticket = table.coordinator_ticket();
    let displaced_slot = table.coordinator_slot()?;
    anyhow::ensure!(
        displaced_ticket != 0 && displaced_slot != NONE_SLOT,
        "coordinator commit-race fixture has no active coordinator",
    );
    table.set_pending_flag(PENDING_RESCAN);
    write_u64(&mut table.header, H_COORDINATOR_HEARTBEAT_NS, 0);
    write_u64(&mut table.header, H_LAST_PROGRESS_NS, 0);
    table.recover_coordinator_if_stalled_at(
        monotonic_now_ns()?.saturating_add(COORDINATOR_HEARTBEAT_LEASE_NS),
    )?;
    anyhow::ensure!(
        table.coordinator_ticket() != displaced_ticket
            && table.record(displaced_slot)?.is_some_and(|record| {
                record.ticket == displaced_ticket && record.state == STATE_COORDINATOR_STANDBY
            }),
        "coordinator commit-race fixture did not transfer the active lease",
    );
    if lost_license {
        table.begin_transaction()?;
        table.set_record_state(displaced_slot, STATE_WAITING)?;
        table.finish_transaction()?;
    }
    Ok(table.pending_flags() & PENDING_RESCAN != 0)
}

/// Invalidate only the caller's coordinator commit token, leaving the same
/// ticket elected so the coordinator loop must take its stale retry path.
#[cfg(test)]
pub(super) fn invalidate_coordinator_commit_token_for_tests() -> Result<()> {
    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    anyhow::ensure!(
        table.coordinator_ticket() != 0 && table.coordinator_slot()? != NONE_SLOT,
        "coordinator token invalidation fixture has no elected coordinator",
    );
    let next = table
        .coordinator_epoch()
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("coordinator token invalidation exhausted the epoch"))?;
    table.begin_transaction()?;
    write_u64(&mut table.header, H_COORDINATOR_EPOCH, next);
    table.finish_transaction()
}

#[cfg(test)]
pub(super) fn exercise_stalled_takeover_notification_for_tests(
    watch: &super::LockDirWatch,
) -> Result<(bool, bool, bool, bool)> {
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(claim.clone(), claim.clone(), None)?;
    let mut successor = Ticket::register(claim.clone(), claim, None)?;

    // Install the real watch before this helper is called, then discard every
    // registration edge. The only event observed below must come from the
    // stalled-transfer transaction itself.
    watch.drain()?;
    let wake_before = coordinator
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("stalled coordinator wake mapping disappeared"))?
        .expected();
    let (coordinator_parked, successor_promoted) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        anyhow::ensure!(
            table.coordinator_ticket() == coordinator.ticket
                && table.coordinator_slot()? == coordinator.slot,
            "stalled-takeover fixture did not elect its first ticket",
        );
        write_u64(&mut table.header, H_COORDINATOR_HEARTBEAT_NS, 0);
        write_u64(&mut table.header, H_LAST_PROGRESS_NS, 0);
        table.recover_coordinator_if_stalled()?;
        (
            table.record(coordinator.slot)?.is_some_and(|record| {
                record.ticket == coordinator.ticket && record.state == STATE_COORDINATOR_STANDBY
            }),
            table.record(successor.slot)?.is_some_and(|record| {
                record.ticket == successor.ticket
                    && record.state == STATE_COORDINATOR
                    && table.coordinator_ticket() == successor.ticket
            }),
        )
    };
    let wake_after = coordinator
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("parked coordinator wake mapping disappeared"))?
        .expected();
    let notified = watch.drain()?.contains_registry_notify();

    successor.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        wake_after != wake_before,
        coordinator_parked,
        successor_promoted,
        notified,
    ))
}

#[cfg(test)]
pub(super) fn exercise_dirty_repair_notification_for_tests(
    watch: &super::LockDirWatch,
) -> Result<(bool, bool, bool, bool)> {
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(claim.clone(), claim, None)?;

    watch.drain()?;
    let wake_before = coordinator
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("repair coordinator wake mapping disappeared"))?
        .expected();
    let (repair_clean, coordinator_restored) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        anyhow::ensure!(
            table.coordinator_ticket() == coordinator.ticket
                && table.coordinator_slot()? == coordinator.slot,
            "dirty-repair fixture did not elect its ticket",
        );
        // Model a writer dying after parking the coordinator record but
        // before publishing a matching header and notification edge.
        atomic_u64(&table.header, H_AGGREGATE_DIRTY).store(1, Ordering::SeqCst);
        table.set_record_state(coordinator.slot, STATE_COORDINATOR_STANDBY)?;
        table.repair_consistency_if_needed()?;
        (
            atomic_u64(&table.header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) == 0,
            table.record(coordinator.slot)?.is_some_and(|record| {
                record.ticket == coordinator.ticket
                    && record.state == STATE_COORDINATOR
                    && table.coordinator_ticket() == coordinator.ticket
            }),
        )
    };
    let wake_after = coordinator
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("repaired coordinator wake mapping disappeared"))?
        .expected();
    let notified = watch.drain()?.contains_registry_notify();

    coordinator.finish(None)?;
    Ok((
        repair_clean,
        coordinator_restored,
        wake_after != wake_before,
        notified,
    ))
}

#[cfg(test)]
pub(super) fn exercise_clean_coordinator_mismatch_recovery_for_tests() -> Result<()> {
    let first_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        FlockMode::Exclusive,
        FlockMode::Exclusive,
    );
    let first = Ticket::register(first_claim.clone(), first_claim, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        let slot = table.coordinator_slot()?;
        anyhow::ensure!(
            table.coordinator_ticket() == first.ticket && slot == first.slot,
            "test setup did not elect the first ticket"
        );
        // Model the observed CI image: the transaction marker is clean, but
        // the header still names a record whose state no longer grants the
        // coordinator license.
        table.set_record_state(slot, STATE_WAITING)?;
        atomic_u64(&table.header, H_AGGREGATE_DIRTY).store(0, Ordering::SeqCst);
    }

    let second_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [2usize],
        FlockMode::Exclusive,
        FlockMode::Exclusive,
    );
    let second = Ticket::register(second_claim.clone(), second_claim, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let coordinator = table.coordinator_ticket();
        let slot = table.coordinator_slot()?;
        let record = table
            .record(slot)?
            .filter(|record| record.ticket == coordinator && record.state == STATE_COORDINATOR)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "coordinator recovery left ticket={coordinator}, slot={slot} incoherent"
                )
            })?;
        anyhow::ensure!(
            record.ticket == first.ticket,
            "record repair unexpectedly replaced the live first coordinator"
        );
    }
    drop(second);
    drop(first);
    Ok(())
}

#[cfg(test)]
pub(super) fn churn_registry_generation_for_tests(rounds: usize) -> Result<()> {
    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    for _ in 0..rounds {
        // Registration and cancellation both advance the structural
        // generation. This deliberately exercises that generic churn without
        // manufacturing coordinator or runnable-queue progress.
        table.advance_generation()?;
    }
    Ok(())
}

/// No-convoy proof: `n` concurrent entrants each register a distinct ticket and
/// perform their post-registration `state_or_wait` reads while a background
/// writer continuously churns registry-EX (retaining the writer-intent
/// sidecar-SH). Under the old BLOCKING shared read every entrant serializes on
/// the mode-inverted sidecar-EX behind the churning writer — the turnstile
/// convoy. The yielding read makes each entrant catch a gap and complete. All
/// `n` must finish their reads well inside `deadline`; returns `(all_ok,
/// elapsed)`.
#[cfg(test)]
pub(super) fn exercise_n_entrants_read_under_churn_for_tests(
    n: usize,
    reads_each: usize,
    llc_prefix: Option<String>,
    cpu_prefix: Option<String>,
    deadline: Duration,
) -> Result<(bool, Duration)> {
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Barrier};

    fn install(llc: &Option<String>, cpu: &Option<String>) {
        super::super::LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc.clone());
        super::super::CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu.clone());
    }
    install(&llc_prefix, &cpu_prefix);

    let stop = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(AtomicUsize::new(0));
    let start_gate = Arc::new(Barrier::new(n + 1));

    // Background writer: churn registry-EX (each hold retains the sidecar-SH the
    // reader's blocking path would convoy behind).
    let writer = {
        let stop = Arc::clone(&stop);
        let llc = llc_prefix.clone();
        let cpu = cpu_prefix.clone();
        std::thread::spawn(move || {
            install(&llc, &cpu);
            while !stop.load(Ordering::Relaxed) {
                if let Ok(lock) = hold_registry_exclusive_for_tests() {
                    std::thread::sleep(Duration::from_millis(1));
                    drop(lock);
                }
                std::thread::sleep(Duration::from_micros(200));
            }
        })
    };

    let mut readers = Vec::with_capacity(n);
    for index in 0..n {
        let llc = llc_prefix.clone();
        let cpu = cpu_prefix.clone();
        let completed = Arc::clone(&completed);
        let start_gate = Arc::clone(&start_gate);
        readers.push(std::thread::spawn(move || -> Result<()> {
            install(&llc, &cpu);
            let claim = ClaimSet::new(std::iter::empty(), [700 + index], FlockMode::Exclusive);
            let mut ticket = Ticket::register(claim.clone(), claim, None)?;
            start_gate.wait();
            for _ in 0..reads_each {
                // The pre-exec wait loop's read: must yield to the churning
                // writer and complete, not convoy.
                let _ = ticket.state_or_wait(Duration::from_millis(2), None)?;
            }
            ticket.finish(None)?;
            completed.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }));
    }
    start_gate.wait();
    let start = std::time::Instant::now();
    while completed.load(Ordering::Relaxed) < n && start.elapsed() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    let elapsed = start.elapsed();
    let all_completed = completed.load(Ordering::Relaxed) == n;
    stop.store(true, Ordering::Relaxed);
    let mut ok = all_completed;
    for reader in readers {
        match reader.join() {
            Ok(Ok(())) => {}
            _ => ok = false,
        }
    }
    let _ = writer.join();
    Ok((ok, elapsed))
}

#[cfg(test)]
pub(super) fn hold_registry_shared_for_tests() -> Result<RegistryLock> {
    lock_registry_existing(FlockMode::Shared)
}

#[cfg(test)]
pub(super) fn hold_registry_exclusive_for_tests() -> Result<RegistryLock> {
    lock_registry_interruptible(None)
}

#[cfg(test)]
pub(super) fn hold_registry_exclusive_after_intent_for_tests(
    on_intent: impl FnOnce() -> Result<()>,
) -> Result<RegistryLock> {
    try_lock_registry_existing_with_writer_intent_hook(FlockMode::Exclusive, on_intent)?.ok_or_else(
        || anyhow::anyhow!("existing admission registry disappeared after test initialization"),
    )
}

#[cfg(test)]
pub(super) fn exercise_writer_intent_initialization_race_for_tests() -> Result<bool> {
    std::fs::create_dir_all(registry_data_dir())?;
    std::fs::create_dir_all(event_dir())?;
    anyhow::ensure!(
        !registry_writer_intent_path().exists() && !registry_lock_path().exists(),
        "writer-intent initialization-race fixture did not start empty",
    );
    let observed = open_existing_writer_intent_after_initial_miss(|| {
        // Model the winning initializer completing after this observer's first
        // sidecar open but before its registry.lock existence check.
        crate::flock::materialize(registry_writer_intent_path())?;
        crate::flock::materialize(registry_lock_path())
    })?;
    Ok(observed.is_some())
}

#[cfg(test)]
pub(super) fn try_hold_registry_shared_for_tests() -> Result<Option<RegistryLock>> {
    try_lock_registry_existing_nonblocking(FlockMode::Shared)
}

#[cfg(test)]
pub(super) fn try_hold_registry_exclusive_for_tests() -> Result<Option<RegistryLock>> {
    try_lock_registry_existing_nonblocking(FlockMode::Exclusive)
}

#[cfg(test)]
pub(super) fn shared_state_read_count_for_tests() -> usize {
    SHARED_STATE_READS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(super) fn ticket_shared_mapping_build_count_for_tests() -> usize {
    TICKET_SHARED_MAPPING_BUILDS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(super) fn exercise_retained_shared_publication_for_tests() -> Result<(bool, bool, bool)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let waiting_claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut waiting = Ticket::register(waiting_claim.clone(), waiting_claim, None)?;
    let mappings_after_registration = ticket_shared_mapping_build_count_for_tests();
    let initially_waiting = waiting.state_shared(false, None)? == Some(State::Waiting);
    let wake_before = waiting
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("WAITING ticket shared mappings disappeared"))?
        .expected();

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, 1, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
    }

    let wake_after = waiting
        .shared
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("granted ticket shared mappings disappeared"))?
        .expected();
    let publication_visible = waiting.state_shared(false, None)? == Some(State::Granted);
    let mapping_reused =
        ticket_shared_mapping_build_count_for_tests() == mappings_after_registration;

    waiting.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        initially_waiting && publication_visible,
        wake_after != wake_before,
        mapping_reused,
    ))
}

#[cfg(test)]
pub(super) fn exercise_retained_mapping_slot_reuse_for_tests() -> Result<(bool, bool)> {
    let claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let mut first = Ticket::register(claim.clone(), claim.clone(), None)?;
    let first_slot = first.slot;
    let first_ticket = first.ticket;
    let stale = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.map_ticket_shared(first_slot, first_ticket)?
    };
    first.finish(None)?;

    let mut replacement = Ticket::register(claim.clone(), claim, None)?;
    let slot_reused = replacement.slot == first_slot && replacement.ticket != first_ticket;
    let stale_mapping_rejected = stale.record_bytes(first_slot, first_ticket).is_err();

    replacement.finish(None)?;
    Ok((slot_reused, stale_mapping_rejected))
}

#[cfg(test)]
pub(super) fn resource_epoch_for_tests() -> Result<u64> {
    let _lock = lock_registry_existing(FlockMode::Shared)?;
    let path = header_path();
    let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let map = unsafe { Mmap::map(&file) }
        .with_context(|| format!("map admission registry header {}", path.display()))?;
    HeaderLayout::validate(&map)?;
    Ok(read_u64(&map, H_GLOBAL_SERIAL).max(1))
}

#[cfg(test)]
pub(super) fn ticket_blocked_at_current_serial_for_tests(pid: u32) -> Result<bool> {
    let Some(_lock) = try_lock_registry_existing(FlockMode::Exclusive)? else {
        return Ok(false);
    };
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    let Some(record) = table
        .records()?
        .into_iter()
        .find(|record| record.pid == pid)
    else {
        return Ok(false);
    };
    let Some(blocked) = record.blocked_on else {
        return Ok(false);
    };
    let blocker_serial = table.blocker_serial(blocked.key, blocked.mode)?;
    let callback_serial = table.max_watch_serial(&record.watch)?.max(blocker_serial);
    Ok(record.state == STATE_WAITING
        && blocked.serial == blocker_serial
        && record.issue_serial >= callback_serial)
}

#[cfg(test)]
pub(super) fn ticket_is_waiting_for_tests(pid: u32) -> Result<bool> {
    let Some(_lock) = try_lock_registry_existing_nonblocking(FlockMode::Shared)? else {
        return Ok(false);
    };
    let mut table = Table::open_existing()?;
    Ok(table
        .records()?
        .into_iter()
        .find(|record| record.pid == pid)
        .is_some_and(|record| record.state == STATE_WAITING))
}

#[cfg(test)]
pub(super) fn ticket_is_granted_for_tests(pid: u32) -> Result<bool> {
    let Some(_lock) = try_lock_registry_existing_nonblocking(FlockMode::Shared)? else {
        return Ok(false);
    };
    let mut table = Table::open_existing()?;
    Ok(table
        .records()?
        .into_iter()
        .find(|record| record.pid == pid)
        .is_some_and(|record| record.state == STATE_GRANTED))
}

#[cfg(test)]
pub(super) fn ticket_is_revoked_for_tests(pid: u32) -> Result<bool> {
    let Some(_lock) = try_lock_registry_existing_nonblocking(FlockMode::Shared)? else {
        return Ok(false);
    };
    let mut table = Table::open_existing()?;
    Ok(table
        .records()?
        .into_iter()
        .find(|record| record.pid == pid)
        .is_some_and(|record| record.state == STATE_REVOKED))
}

#[cfg(test)]
pub(super) fn coordinator_liveness_probe_for_tests() -> Result<((u64, u64), bool)> {
    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    let ticket = table.coordinator_ticket();
    let slot = table.coordinator_slot()?;
    if ticket == 0 || slot == NONE_SLOT {
        anyhow::bail!("liveness probe test needs a live coordinator");
    }
    Ok(((slot, ticket), ticket_is_live(slot, ticket)?))
}

#[cfg(test)]
pub(super) fn missing_liveness_probe_does_not_create_for_tests() -> Result<bool> {
    let slot = MAX_REGISTRY_SLOTS - 1;
    let ticket = u64::MAX - 1;
    let path = liveness_path(slot, ticket);
    let _ = std::fs::remove_file(&path);
    Ok(!ticket_is_live(slot, ticket)? && !path.exists())
}

#[cfg(test)]
pub(super) fn active_free_head_is_rejected_for_tests() -> Result<()> {
    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    let active_slot = table
        .records()?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("free-list corruption test needs one active record"))?
        .slot;
    let previous = read_u64(&table.header, H_FREE_HEAD);
    write_u64(&mut table.header, H_FREE_HEAD, active_slot);
    let result = table.allocate_slot();
    write_u64(&mut table.header, H_FREE_HEAD, previous);
    match result {
        Err(error)
            if error.to_string().contains("free-list head")
                && error.to_string().contains("active or malformed") =>
        {
            Ok(())
        }
        Err(error) => Err(error)
            .context("malformed free-list head failed, but without identifying the active record"),
        Ok(slot) => anyhow::bail!(
            "malformed free-list head reused active registry slot {slot} instead of failing closed"
        ),
    }
}

#[cfg(test)]
pub(super) fn cancel_granted_after_commit_for_tests() {
    CANCEL_GRANTED_AFTER_COMMIT.with(|armed| armed.set(true));
}

#[cfg(test)]
pub(super) fn cancel_coordinator_after_commit_for_tests() {
    CANCEL_COORDINATOR_AFTER_COMMIT.with(|armed| armed.set(true));
}

#[cfg(test)]
fn set_cpu_availability_for_tests(
    table: &mut Table,
    cpu: usize,
    availability: CpuAvailability,
) -> Result<()> {
    table.set_bitmap_bit(B_CPU_KNOWN, cpu, true)?;
    table.set_bitmap_bit(
        B_CPU_SH_AVAILABLE,
        cpu,
        availability != CpuAvailability::ExclusiveHeld,
    )?;
    table.set_bitmap_bit(
        B_CPU_EX_AVAILABLE,
        cpu,
        availability == CpuAvailability::Free,
    )?;
    table.set_bitmap_bit(B_PENDING_CPU_SH, cpu, false)?;
    table.set_bitmap_bit(B_PENDING_CPU_EX, cpu, false)?;
    table.set_bitmap_bit(B_CANDIDATE_CPU_SH, cpu, false)?;
    table.set_bitmap_bit(B_CANDIDATE_CPU_EX, cpu, false)?;
    table.set_resource_request(Q_CPU_SH, cpu, 0)?;
    table.set_resource_request(Q_CPU_EX, cpu, 0)
}

#[cfg(test)]
fn set_cpu_free_for_tests(table: &mut Table, cpu: usize, free: bool) -> Result<()> {
    set_cpu_availability_for_tests(
        table,
        cpu,
        if free {
            CpuAvailability::Free
        } else {
            CpuAvailability::ExclusiveHeld
        },
    )
}

#[cfg(test)]
pub(super) fn exercise_resource_weighted_backfill_accounting_for_tests() -> (u32, u32, u32, u32) {
    let cooperative_end = super::super::cooperative_cpu_permit_end();
    let heavy_units = cooperative_end.min(4);
    let watch = ClaimSet::with_permits(
        std::iter::empty(),
        [0usize],
        0..cooperative_end,
        FlockMode::Shared,
        FlockMode::Shared,
        FlockMode::Exclusive,
    );
    let light = ClaimSet::with_permits(
        std::iter::empty(),
        [0usize],
        [0usize],
        FlockMode::Shared,
        FlockMode::Shared,
        FlockMode::Exclusive,
    );
    let heavy = ClaimSet::with_permits(
        std::iter::empty(),
        [0usize],
        0..heavy_units,
        FlockMode::Shared,
        FlockMode::Shared,
        FlockMode::Exclusive,
    );
    let non_cooperative = ClaimSet::with_permits(
        std::iter::empty(),
        [0usize, 1usize],
        [cooperative_end],
        FlockMode::Shared,
        FlockMode::Shared,
        FlockMode::Exclusive,
    );
    (
        backfill_capacity_for_watch(&watch),
        backfill_cost_for_claim(&light),
        backfill_cost_for_claim(&heavy),
        backfill_cost_for_claim(&non_cooperative),
    )
}

#[cfg(test)]
pub(crate) struct PreparationPoolBudgetOutcome {
    pub(crate) pool: usize,
    pub(crate) waiters: usize,
    pub(crate) granted_in_one_scan: usize,
    pub(crate) oldest_prefix_granted: bool,
    pub(crate) surplus_none_granted: bool,
    pub(crate) surplus_none_pinned: bool,
}

/// Work-conservation and FIFO for the preparation-slot pool: a single scan
/// over `WAITERS` compatible preparation intents with a `POOL`-slot budget must
/// grant exactly `min(WAITERS, POOL)` of them, the oldest by ticket, and leave
/// the surplus WAITING without pinning any to a token.
#[cfg(test)]
pub(super) fn exercise_preparation_pool_budget_for_tests() -> Result<PreparationPoolBudgetOutcome> {
    const POOL: usize = 2;
    const WAITERS: usize = 5;
    let tokens = super::super::preparation_token_range()?;
    anyhow::ensure!(
        tokens.len() >= POOL,
        "host preparation pool {} too small for budget test",
        tokens.len(),
    );
    let pool = tokens.start..tokens.start + POOL;
    let run_cpu = 7usize;
    let coord_claim = ClaimSet::new(std::iter::empty(), [3usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coord_claim.clone(), coord_claim, None)?;
    // The run claim carries no token; unioning one pool token into the watch is
    // what marks the record a token-consuming preparation intent for the scan.
    let waiter_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [run_cpu],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let waiter_watch = ClaimSet::with_permits(
        std::iter::empty(),
        [run_cpu],
        [tokens.start],
        FlockMode::Shared,
        FlockMode::Shared,
        FlockMode::Exclusive,
    );
    let mut waiters = (0..WAITERS)
        .map(|_| Ticket::register(waiter_claim.clone(), waiter_watch.clone(), None))
        .collect::<Result<Vec<_>>>()?;

    let (granted, oldest_prefix, surplus_none, surplus_unpinned) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.restage_coordinator_for_tests(&coordinator, waiters.iter())?;
        set_cpu_free_for_tests(&mut table, run_cpu, true)?;
        set_cpu_free_for_tests(&mut table, 3, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), Some(&pool))?;

        let mut granted = 0usize;
        for waiter in &waiters {
            if table
                .record(waiter.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED)
            {
                granted += 1;
            }
        }
        let mut oldest_prefix = true;
        for waiter in &waiters[..POOL] {
            oldest_prefix &= table
                .record(waiter.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED);
        }
        let mut surplus_none = true;
        let mut surplus_unpinned = true;
        for waiter in &waiters[POOL..] {
            let record = table
                .record(waiter.slot)?
                .ok_or_else(|| anyhow::anyhow!("budget-blocked waiter disappeared"))?;
            surplus_none &= record.state != STATE_GRANTED;
            surplus_unpinned &= record.blocked_on.is_none();
        }
        (granted, oldest_prefix, surplus_none, surplus_unpinned)
    };
    for waiter in &mut waiters {
        waiter.finish(None)?;
    }
    coordinator.finish(None)?;
    exercise_preparation_pool_revoked_cohort_for_tests()?;
    Ok(PreparationPoolBudgetOutcome {
        pool: POOL,
        waiters: WAITERS,
        granted_in_one_scan: granted,
        oldest_prefix_granted: oldest_prefix,
        surplus_none_granted: surplus_none,
        surplus_none_pinned: surplus_unpinned,
    })
}

/// Budget accounting for a grant revoked inside the scan that observes it: the
/// revoked callback may already own the token it raced, so its slot stays
/// charged until its own REVOKED acknowledgement retires the record. A
/// disjoint successor must therefore stay WAITING in that same scan and win
/// the slot only once the revoked intent is gone. Asserted in place because
/// the exercise is a phase of the pool-budget test, not a separate outcome.
#[cfg(test)]
fn exercise_preparation_pool_revoked_cohort_for_tests() -> Result<()> {
    let tokens = super::super::preparation_token_range()?;
    anyhow::ensure!(
        !tokens.is_empty(),
        "host preparation pool is empty for the revoked-cohort test",
    );
    let pool = tokens.start..tokens.start + 1;
    let victim_cpu = 6usize;
    let successor_cpu = 8usize;
    let coord_claim = ClaimSet::new(std::iter::empty(), [3usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coord_claim.clone(), coord_claim, None)?;
    // Disjoint run CPUs: withdrawing the victim's CPU revokes its grant while
    // leaving the successor physically viable, so only the pool budget can
    // hold the successor back.
    let register_intent = |cpu: usize| -> Result<Ticket> {
        let claim = ClaimSet::with_modes(
            std::iter::empty(),
            [cpu],
            FlockMode::Shared,
            FlockMode::Shared,
        );
        let watch = ClaimSet::with_permits(
            std::iter::empty(),
            [cpu],
            [tokens.start],
            FlockMode::Shared,
            FlockMode::Shared,
            FlockMode::Exclusive,
        );
        Ticket::register(claim, watch, None)
    };
    let mut victim = register_intent(victim_cpu)?;
    let mut successor = register_intent(successor_cpu)?;

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.restage_coordinator_for_tests(&coordinator, [&victim, &successor].into_iter())?;
        set_cpu_free_for_tests(&mut table, victim_cpu, true)?;
        set_cpu_free_for_tests(&mut table, successor_cpu, true)?;
        set_cpu_free_for_tests(&mut table, 3, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), Some(&pool))?;
        anyhow::ensure!(
            table
                .record(victim.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED),
            "the oldest preparation intent must first win the single pool slot",
        );
        anyhow::ensure!(
            table
                .record(successor.slot)?
                .is_some_and(|record| record.state != STATE_GRANTED),
            "the successor must not be granted while the single pool slot is taken",
        );
    }

    // Withdraw the victim's run CPU so the next scan revokes its grant.
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, victim_cpu, false)?;
        set_cpu_free_for_tests(&mut table, successor_cpu, true)?;
        set_cpu_free_for_tests(&mut table, 3, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), Some(&pool))?;
        anyhow::ensure!(
            table
                .record(victim.slot)?
                .is_some_and(|record| record.state == STATE_REVOKED),
            "withdrawing the granted intent's run CPU must revoke its grant",
        );
        anyhow::ensure!(
            table
                .record(successor.slot)?
                .is_some_and(|record| record.state != STATE_GRANTED),
            "a grant revoked in this scan still holds its pool slot; the successor must not be granted against it",
        );
    }

    // Retiring the revoked intent frees the slot: the same successor, under
    // the same host availability, is granted by the next scan.
    victim.finish(None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, victim_cpu, false)?;
        set_cpu_free_for_tests(&mut table, successor_cpu, true)?;
        set_cpu_free_for_tests(&mut table, 3, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), Some(&pool))?;
        anyhow::ensure!(
            table
                .record(successor.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED),
            "the successor must win the pool slot once the revoked intent retires",
        );
    }
    successor.finish(None)?;
    coordinator.finish(None)?;
    Ok(())
}

#[cfg(test)]
pub(crate) struct PreparationPoolStarvationOutcome {
    pub(crate) blocked_while_full: usize,
    pub(crate) oldest_wins_freed_slot: bool,
    pub(crate) newcomer_still_blocked: bool,
}

/// FIFO starvation-freedom under churn: with a full one-slot pool, the oldest
/// waiter and a newcomer both wait; when the held slot frees, the very next
/// scan grants the oldest — never the newcomer — in bounded time (one scan).
#[cfg(test)]
pub(super) fn exercise_preparation_pool_starvation_for_tests()
-> Result<PreparationPoolStarvationOutcome> {
    let tokens = super::super::preparation_token_range()?;
    anyhow::ensure!(
        tokens.len() >= 2,
        "host preparation pool {} too small for starvation test",
        tokens.len(),
    );
    let pool = tokens.start..tokens.start + 1;
    let held_token = tokens.start;
    let run_cpu = 9usize;
    let coord_claim = ClaimSet::new(std::iter::empty(), [4usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coord_claim.clone(), coord_claim, None)?;
    // A PENDING holder occupies the single pool slot: its acquired token is in
    // its claim, so the scan counts it against the budget.
    let holder_claim = ClaimSet::with_permits(
        std::iter::empty(),
        std::iter::empty(),
        [held_token],
        FlockMode::Shared,
        FlockMode::Shared,
        FlockMode::Exclusive,
    );
    let mut holder = Ticket::register(holder_claim.clone(), holder_claim.clone(), None)?;
    let waiter_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [run_cpu],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let waiter_watch = ClaimSet::with_permits(
        std::iter::empty(),
        [run_cpu],
        [tokens.start],
        FlockMode::Shared,
        FlockMode::Shared,
        FlockMode::Exclusive,
    );
    // `oldest` registers before `newcomer`, so it holds the lower ticket.
    let mut oldest = Ticket::register(waiter_claim.clone(), waiter_watch.clone(), None)?;
    let mut newcomer = Ticket::register(waiter_claim.clone(), waiter_watch.clone(), None)?;

    let blocked_while_full = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.restage_coordinator_for_tests(
            &coordinator,
            [&holder, &oldest, &newcomer].into_iter(),
        )?;
        table.set_record_state(holder.slot, STATE_PENDING)?;
        set_cpu_free_for_tests(&mut table, run_cpu, true)?;
        set_cpu_free_for_tests(&mut table, 4, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), Some(&pool))?;
        let mut blocked = 0usize;
        for waiter in [&oldest, &newcomer] {
            if table
                .record(waiter.slot)?
                .is_some_and(|record| record.state != STATE_GRANTED)
            {
                blocked += 1;
            }
        }
        blocked
    };

    // Free the single slot by retiring the holder, then rescan.
    holder.finish(None)?;
    let (oldest_wins, newcomer_blocked) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, run_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), Some(&pool))?;
        let oldest_wins = table
            .record(oldest.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED);
        let newcomer_blocked = table
            .record(newcomer.slot)?
            .is_some_and(|record| record.state != STATE_GRANTED);
        (oldest_wins, newcomer_blocked)
    };
    oldest.finish(None)?;
    newcomer.finish(None)?;
    coordinator.finish(None)?;
    Ok(PreparationPoolStarvationOutcome {
        blocked_while_full,
        oldest_wins_freed_slot: oldest_wins,
        newcomer_still_blocked: newcomer_blocked,
    })
}

#[cfg(test)]
pub(crate) struct PreparationPoolCrashRecoveryOutcome {
    pub(crate) victim_granted: bool,
    pub(crate) successor_blocked_before_prune: bool,
    pub(crate) successor_granted_after_prune: bool,
}

/// Crash-safety of a budgeted slot: a preparation intent granted but not yet
/// confirmed (GRANTED, still racing) holds a slot in the budget. If its process
/// dies, the slot must not leak — its dead record keeps charging the budget only
/// until liveness pruning removes it, after which the next scan hands the slot
/// to a waiting successor.
#[cfg(test)]
pub(super) fn exercise_preparation_pool_crash_recovery_for_tests()
-> Result<PreparationPoolCrashRecoveryOutcome> {
    let tokens = super::super::preparation_token_range()?;
    anyhow::ensure!(
        tokens.len() >= 2,
        "host preparation pool {} too small for crash-recovery test",
        tokens.len(),
    );
    let pool = tokens.start..tokens.start + 1;
    let run_cpu = 11usize;
    let coord_claim = ClaimSet::new(std::iter::empty(), [5usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coord_claim.clone(), coord_claim, None)?;
    let waiter_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [run_cpu],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let waiter_watch = ClaimSet::with_permits(
        std::iter::empty(),
        [run_cpu],
        [tokens.start],
        FlockMode::Shared,
        FlockMode::Shared,
        FlockMode::Exclusive,
    );
    let victim = Ticket::register(waiter_claim.clone(), waiter_watch.clone(), None)?;
    let mut successor = Ticket::register(waiter_claim.clone(), waiter_watch.clone(), None)?;

    let (victim_granted, successor_blocked) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.restage_coordinator_for_tests(&coordinator, [&victim, &successor].into_iter())?;
        set_cpu_free_for_tests(&mut table, run_cpu, true)?;
        set_cpu_free_for_tests(&mut table, 5, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), Some(&pool))?;
        let victim_granted = table
            .record(victim.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED);
        let successor_blocked = table
            .record(successor.slot)?
            .is_some_and(|record| record.state != STATE_GRANTED);
        (victim_granted, successor_blocked)
    };

    // Model abrupt death of the granted-but-unconfirmed intent.
    victim.abandon_for_tests();

    // Its dead GRANTED record still charges the slot until it is pruned.
    let successor_blocked_before_prune = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, run_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), Some(&pool))?;
        table
            .record(successor.slot)?
            .is_some_and(|record| record.state != STATE_GRANTED)
    };

    // Liveness pruning reclaims the dead slot; the next scan grants the successor.
    let successor_granted_after_prune = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.prune_dead()?;
        set_cpu_free_for_tests(&mut table, run_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), Some(&pool))?;
        table
            .record(successor.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED)
    };
    successor.finish(None)?;
    coordinator.finish(None)?;
    Ok(PreparationPoolCrashRecoveryOutcome {
        victim_granted,
        successor_blocked_before_prune: successor_blocked && successor_blocked_before_prune,
        successor_granted_after_prune,
    })
}

#[cfg(test)]
pub(crate) struct WorkConservingBackfillOutcome {
    pub(crate) conflicting_grants: usize,
    pub(crate) conflicting_waiters: usize,
    pub(crate) disjoint_grants: usize,
    pub(crate) refilled_after_completion: bool,
    pub(crate) expired_head_stops_refill: bool,
    pub(crate) wide_wins: bool,
    pub(crate) racer_revoked_without_placement_damage: bool,
    pub(crate) stale_callback_suppressed: bool,
}

#[cfg(test)]
pub(super) fn exercise_work_conserving_backfill_for_tests() -> Result<WorkConservingBackfillOutcome>
{
    const TEST_CAPACITY: u32 = 3;
    const CONFLICTING: usize = TEST_CAPACITY as usize + 2;
    const DISJOINT: usize = TEST_CAPACITY as usize + 5;

    let coordinator_claim = ClaimSet::new(std::iter::empty(), [3usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let wide = ClaimSet::with_permits(
        std::iter::empty(),
        [0usize, 1usize],
        0..TEST_CAPACITY as usize,
        FlockMode::Exclusive,
        FlockMode::Exclusive,
        FlockMode::Exclusive,
    );
    let mut wide_ticket = Ticket::register(wide.clone(), wide, None)?;
    let conflicting_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [0usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let mut conflicting = (0..CONFLICTING)
        .map(|_| Ticket::register(conflicting_claim.clone(), conflicting_claim.clone(), None))
        .collect::<Result<Vec<_>>>()?;
    let disjoint_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [2usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let mut disjoint = (0..DISJOINT)
        .map(|_| Ticket::register(disjoint_claim.clone(), disjoint_claim.clone(), None))
        .collect::<Result<Vec<_>>>()?;
    let count_state = |table: &mut Table, tickets: &[Ticket], state| -> Result<usize> {
        tickets.iter().try_fold(0usize, |count, ticket| {
            Ok(count
                + if table
                    .record(ticket.slot)?
                    .is_some_and(|record| record.state == state)
                {
                    1
                } else {
                    0
                })
        })
    };

    let (initial_grants, initial_waiters, disjoint_grants, backfill_started_ns) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.restage_coordinator_for_tests(
            &coordinator,
            std::iter::once(&wide_ticket)
                .chain(conflicting.iter())
                .chain(disjoint.iter()),
        )?;
        table.set_record_backfill_capacity(wide_ticket.slot, TEST_CAPACITY)?;
        let scan_now_ns = monotonic_now_ns()?.max(1);
        let stale_future_ns = scan_now_ns
            .checked_add(BACKFILL_MAX_AGE_NS)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("synthetic future backfill epoch overflow"))?;
        table.set_record_backfill_started_ns(wide_ticket.slot, stale_future_ns)?;
        set_cpu_free_for_tests(&mut table, 0, true)?;
        set_cpu_free_for_tests(&mut table, 1, false)?;
        set_cpu_free_for_tests(&mut table, 2, true)?;
        set_cpu_free_for_tests(&mut table, 3, true)?;
        for permit in 0..TEST_CAPACITY as usize {
            set_cpu_free_for_tests(&mut table, permit_resource_index(permit)?, true)?;
        }
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(scan_now_ns, None)?;

        let head = table
            .record(wide_ticket.slot)?
            .ok_or_else(|| anyhow::anyhow!("wide backfill head disappeared"))?;
        anyhow::ensure!(
            head.backfill_capacity == TEST_CAPACITY,
            "wide backfill head mutated its {}-unit capacity to {}",
            TEST_CAPACITY,
            head.backfill_capacity,
        );
        anyhow::ensure!(
            head.backfill_started_ns == scan_now_ns,
            "wide backfill head did not normalize and persist its bounded-admission start: {} != {scan_now_ns}",
            head.backfill_started_ns,
        );
        (
            count_state(&mut table, &conflicting, STATE_GRANTED)?,
            count_state(&mut table, &conflicting, STATE_WAITING)?,
            count_state(&mut table, &disjoint, STATE_GRANTED)?,
            head.backfill_started_ns,
        )
    };

    // Remove one member of the first bypass wave. The next scan must reclaim
    // that live capacity and admit the oldest waiting replacement rather than
    // entering a permanent low-utilization drain.
    conflicting[0].finish(None)?;
    let refilled_after_completion = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_availability_for_tests(&mut table, 0, CpuAvailability::SharedHeld)?;
        set_cpu_free_for_tests(&mut table, 1, false)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(backfill_started_ns.saturating_add(1), None)?;
        count_state(&mut table, &conflicting, STATE_GRANTED)? == TEST_CAPACITY as usize
            && table
                .record(conflicting[TEST_CAPACITY as usize].slot)?
                .is_some_and(|record| record.state == STATE_GRANTED)
            && table
                .record(wide_ticket.slot)?
                .is_some_and(|record| record.backfill_capacity == TEST_CAPACITY)
    };

    // Once the same head's admission age expires, removing another member may
    // not be replaced. Existing bypass work drains naturally, bounding the
    // delay before an exclusive performance-mode head can run.
    conflicting[1].finish(None)?;
    let expired_now_ns = backfill_started_ns
        .checked_add(BACKFILL_MAX_AGE_NS)
        .ok_or_else(|| anyhow::anyhow!("synthetic backfill age overflow"))?;
    let expired_head_stops_refill = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_availability_for_tests(&mut table, 0, CpuAvailability::SharedHeld)?;
        set_cpu_free_for_tests(&mut table, 1, false)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(expired_now_ns, None)?;
        count_state(&mut table, &conflicting, STATE_GRANTED)? == TEST_CAPACITY as usize - 1
            && table
                .record(conflicting[CONFLICTING - 1].slot)?
                .is_some_and(|record| record.state == STATE_WAITING)
    };

    let racer_index = CONFLICTING - 1;
    let (wide_wins, racer_revoked, disjoint_preserved) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        // Model the remaining admitted burst having released. The extra GRANTED record
        // models a callback issued from the old availability snapshot just as
        // the wide head becomes viable.
        for ticket in &conflicting[2..racer_index] {
            if table.record(ticket.slot)?.is_some() {
                table.set_record_state(ticket.slot, STATE_WAITING)?;
            }
        }
        table.set_record_state(conflicting[racer_index].slot, STATE_GRANTED)?;
        set_cpu_free_for_tests(&mut table, 0, true)?;
        set_cpu_free_for_tests(&mut table, 1, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(expired_now_ns, None)?;
        (
            table
                .record(wide_ticket.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED),
            table
                .record(conflicting[racer_index].slot)?
                .is_some_and(|record| record.state == STATE_REVOKED),
            disjoint.iter().all(|ticket| {
                table
                    .record(ticket.slot)
                    .ok()
                    .flatten()
                    .is_some_and(|record| record.state == STATE_GRANTED)
            }),
        )
    };
    let mut racer_callbacks = 0usize;
    let racer_result = conflicting[racer_index].run_granted(
        None,
        |_designated, _watch, _allowed, _predecessors, _availability| {
            racer_callbacks += 1;
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: conflicting_claim.clone(),
                contention: None,
            })
        },
    )?;
    let revoked_ack_published = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table
            .record(conflicting[racer_index].slot)?
            .is_some_and(|record| record.state == STATE_WAITING)
            && table.pending_flags() & PENDING_RESCAN != 0
    };
    let stale_callback_suppressed = matches!(racer_result, GrantResult::LostGrant)
        && racer_callbacks == 0
        && revoked_ack_published;

    for ticket in &mut disjoint {
        ticket.finish(None)?;
    }
    for ticket in &mut conflicting {
        ticket.finish(None)?;
    }
    wide_ticket.finish(None)?;
    coordinator.finish(None)?;
    Ok(WorkConservingBackfillOutcome {
        conflicting_grants: initial_grants,
        conflicting_waiters: initial_waiters,
        disjoint_grants,
        refilled_after_completion,
        expired_head_stops_refill,
        wide_wins,
        racer_revoked_without_placement_damage: racer_revoked && disjoint_preserved,
        stale_callback_suppressed,
    })
}

#[cfg(test)]
pub(super) fn exercise_granted_only_drain_election_reads_for_tests(
    waiters: usize,
) -> Result<usize> {
    if waiters == 0 {
        return Ok(0);
    }
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut tickets = Vec::with_capacity(waiters);
    for _ in 0..waiters {
        tickets.push(Ticket::register(claim.clone(), claim.clone(), None)?);
    }
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.begin_transaction()?;
        table.set_coordinator(0, NONE_SLOT)?;
        for ticket in &tickets {
            table.set_record_state(ticket.slot, STATE_GRANTED)?;
        }
        table.finish_transaction()?;
    }

    let reads_before = COORDINATOR_ELECTION_RECORD_READS.with(std::cell::Cell::get);
    while let Some(ticket) = tickets.pop() {
        drop(ticket);
    }
    Ok(COORDINATOR_ELECTION_RECORD_READS.with(std::cell::Cell::get) - reads_before)
}

#[cfg(test)]
pub(super) fn exercise_known_free_close_storm_for_tests(
    closes: usize,
) -> Result<(usize, u64, usize, u64, u64)> {
    exercise_known_free_close_storm_impl(closes, false)
}

#[cfg(test)]
pub(super) fn exercise_stale_heartbeat_known_free_close_for_tests()
-> Result<(usize, u64, usize, u64, u64)> {
    exercise_known_free_close_storm_impl(1, true)
}

#[cfg(test)]
fn exercise_known_free_close_storm_impl(
    closes: usize,
    force_stale_heartbeat: bool,
) -> Result<(usize, u64, usize, u64, u64)> {
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut ticket = Ticket::register(claim.clone(), claim, None)?;
    let empty = BTreeSet::new();
    let initial = ticket.schedule(
        None,
        &empty,
        &empty,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let request = initial
        .observation
        .ok_or_else(|| anyhow::anyhow!("first watch must request an observation"))?;
    let mut observation = AvailabilityObservation::default();
    observation.cpus.insert(
        1,
        CpuObservation {
            availability: CpuAvailability::Free,
            sh_resolved: true,
            ex_resolved: true,
        },
    );
    let initial = ticket.apply_observation(&request, &observation, || {}, None)?;
    if !initial.should_step {
        anyhow::bail!("first free observation must publish one persisted improvement");
    }
    if force_stale_heartbeat {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        anyhow::ensure!(
            table.coordinator_ticket() == ticket.ticket && table.coordinator_slot()? == ticket.slot,
            "test ticket lost coordinator ownership before heartbeat expiry",
        );
        write_u64(&mut table.header, H_COORDINATOR_HEARTBEAT_NS, 0);
    }
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let generation_before = diagnostic_counter_for_tests(H_GENERATION)?;
    let ex_before = REGISTRY_EX_ACQUISITIONS.with(std::cell::Cell::get);
    let mut observations = 0;
    let mut planner_steps = 0;
    let closed = BTreeSet::from([1usize, usize::MAX]);
    let closed_unwatched = BTreeSet::from([usize::MAX]);
    for _ in 0..closes {
        let snapshot = ticket.schedule(
            None,
            &closed,
            &closed_unwatched,
            &closed_unwatched,
            false,
            &[],
            &[],
            false,
            None,
            false,
            None,
        )?;
        observations += usize::from(snapshot.observation.is_some());
        planner_steps += usize::from(snapshot.should_step);
    }
    let scans = diagnostic_counter_for_tests(H_GRANT_SCANS)? - scans_before;
    let generation = diagnostic_counter_for_tests(H_GENERATION)? - generation_before;
    let ex_acquisitions = REGISTRY_EX_ACQUISITIONS.with(std::cell::Cell::get) - ex_before;
    ticket.finish(None)?;
    Ok((
        observations,
        scans,
        planner_steps,
        generation,
        ex_acquisitions,
    ))
}

#[cfg(test)]
pub(super) fn exercise_llc_sh_only_shared_to_free_close_for_tests() -> Result<(bool, bool, u64, u64)>
{
    let claim = ClaimSet::new([1usize], std::iter::empty(), FlockMode::Shared);
    let mut ticket = Ticket::register(claim.clone(), claim, None)?;
    let empty = BTreeSet::new();
    let initial = ticket.schedule(
        None,
        &empty,
        &empty,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let request = initial
        .observation
        .ok_or_else(|| anyhow::anyhow!("first SH-only LLC watch needs an observation"))?;
    let requested_modes = request
        .llcs
        .get(&1)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("SH-only LLC request omitted its resource"))?;
    if requested_modes.0.is_none() || requested_modes.1.is_some() {
        anyhow::bail!("SH-only LLC watch requested the wrong modes: {requested_modes:?}");
    }
    let mut shared = AvailabilityObservation::default();
    shared.llcs.insert(
        1,
        LlcObservation {
            availability: LlcAvailability::SharedHeld,
            sh_resolved: true,
            ex_resolved: true,
        },
    );
    let initial = ticket.apply_observation(&request, &shared, || {}, None)?;
    if !initial.should_step {
        anyhow::bail!("initial SharedHeld observation must establish SH availability");
    }
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let generation_before = diagnostic_counter_for_tests(H_GENERATION)?;
    let closed = BTreeSet::from([1usize]);
    let after_close = ticket.schedule(
        None,
        &empty,
        &closed,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let scans = diagnostic_counter_for_tests(H_GRANT_SCANS)? - scans_before;
    let generations = diagnostic_counter_for_tests(H_GENERATION)? - generation_before;
    let observation_requested = after_close.observation.is_some();
    let planner_step = after_close.should_step;
    ticket.finish(None)?;
    Ok((observation_requested, planner_step, scans, generations))
}

#[cfg(test)]
pub(super) fn exercise_busy_to_free_close_for_tests() -> Result<(usize, u64, usize, u32, u32)> {
    fn generation_wake() -> Result<u32> {
        let _lock = lock_registry_existing(FlockMode::Shared)?;
        let table = Table::open_existing()?;
        Ok(table.generation_wake())
    }

    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut ticket = Ticket::register(claim.clone(), claim, None)?;
    let empty = BTreeSet::new();
    let initial = ticket.schedule(
        None,
        &empty,
        &empty,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let request = initial
        .observation
        .ok_or_else(|| anyhow::anyhow!("first watch must request an observation"))?;
    let mut busy = AvailabilityObservation::default();
    busy.cpus.insert(
        1,
        CpuObservation {
            availability: CpuAvailability::ExclusiveHeld,
            sh_resolved: true,
            ex_resolved: true,
        },
    );
    let initial = ticket.apply_observation(&request, &busy, || {}, None)?;
    if initial.should_step {
        anyhow::bail!("first busy observation must not report usable capacity");
    }
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let wake_before_schedule = generation_wake()?;
    let closed = BTreeSet::from([1usize]);
    let pending = ticket.schedule(
        None,
        &closed,
        &empty,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let wake_after_schedule = generation_wake()?;
    let request = pending
        .observation
        .ok_or_else(|| anyhow::anyhow!("busy resource close must request an observation"))?;
    let mut free = AvailabilityObservation::default();
    free.cpus.insert(
        1,
        CpuObservation {
            availability: CpuAvailability::Free,
            sh_resolved: true,
            ex_resolved: true,
        },
    );
    let improved = ticket.apply_observation(&request, &free, || {}, None)?;
    let wake_after_observation = generation_wake()?;
    let observations = 1;
    let planner_steps = usize::from(improved.should_step);
    let scans = diagnostic_counter_for_tests(H_GRANT_SCANS)? - scans_before;
    ticket.finish(None)?;
    Ok((
        observations,
        scans,
        planner_steps,
        wake_after_schedule.wrapping_sub(wake_before_schedule),
        wake_after_observation.wrapping_sub(wake_after_schedule),
    ))
}

#[cfg(test)]
pub(super) fn exercise_llc_ex_contention_shared_wake_for_tests() -> Result<(u64, bool, bool, bool)>
{
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let coordinator_watch = ClaimSet::new([1usize], [0usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim, coordinator_watch, None)?;
    let shared_claim = ClaimSet::new([1usize], std::iter::empty(), FlockMode::Shared);
    let mut shared = Ticket::register(shared_claim.clone(), shared_claim, None)?;
    let exclusive_claim = ClaimSet::new([1usize], std::iter::empty(), FlockMode::Exclusive);
    let mut exclusive = Ticket::register(exclusive_claim.clone(), exclusive_claim, None)?;

    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    table.set_record_state(shared.slot, STATE_WAITING)?;
    table.clear_record_blocked(shared.slot)?;
    table.set_record_state(exclusive.slot, STATE_WAITING)?;
    table.clear_record_blocked(exclusive.slot)?;
    table.set_bitmap_bit(B_LLC_KNOWN, 1, true)?;
    table.set_bitmap_bit(B_LLC_SH_AVAILABLE, 1, true)?;
    table.set_bitmap_bit(B_LLC_EX_AVAILABLE, 1, false)?;
    table.set_bitmap_bit(B_PENDING_LLC_SH, 1, false)?;
    table.set_bitmap_bit(B_PENDING_LLC_EX, 1, false)?;
    table.set_bitmap_bit(B_CANDIDATE_LLC_SH, 1, false)?;
    table.set_bitmap_bit(B_CANDIDATE_LLC_EX, 1, false)?;
    write_u64(&mut table.header, H_PENDING_FLAGS, 0);

    table.begin_transaction()?;
    table.mark_blocker_unknown(ContentionMarker {
        blocker: ResourceKey::Llc(1),
        mode: FlockMode::Exclusive,
    })?;
    table.set_pending_flag(PENDING_RESCAN);
    table.finish_transaction()?;
    table.grant_compatible()?;
    if table
        .record(shared.slot)?
        .is_none_or(|record| record.state != STATE_WAITING)
        || table
            .record(exclusive.slot)?
            .is_none_or(|record| record.state != STATE_WAITING)
    {
        anyhow::bail!("unknown LLC modes must keep both SH and EX tickets waiting");
    }

    let request = table
        .observation_request()?
        .ok_or_else(|| anyhow::anyhow!("failed EX probe must request an LLC observation"))?;
    let scans_before = read_u64(&table.header, H_GRANT_SCANS);
    drop(table);
    drop(_lock);

    let mut observation = AvailabilityObservation::default();
    observation.llcs.insert(
        1,
        LlcObservation {
            availability: LlcAvailability::SharedHeld,
            sh_resolved: true,
            ex_resolved: true,
        },
    );
    let snapshot = coordinator.apply_observation(&request, &observation, || {}, None)?;
    if snapshot.should_step {
        anyhow::bail!(
            "an SH-only LLC improvement must not rerun the incompatible EX coordinator planner"
        );
    }

    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    let scans = read_u64(&table.header, H_GRANT_SCANS) - scans_before;
    let shared_granted = table
        .record(shared.slot)?
        .is_some_and(|record| record.state == STATE_GRANTED);
    let exclusive_waiting = table
        .record(exclusive.slot)?
        .is_some_and(|record| record.state == STATE_WAITING);
    drop(table);
    drop(_lock);

    shared.finish(None)?;
    exclusive.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        scans,
        shared_granted,
        exclusive_waiting,
        !snapshot.should_step,
    ))
}

#[cfg(test)]
pub(super) fn exercise_cpu_ex_contention_shared_wake_for_tests() -> Result<CpuExContentionSharedWake>
{
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let coordinator_watch =
        ClaimSet::new(std::iter::empty(), [0usize, 1usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim, coordinator_watch, None)?;
    let shared_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let mut shared = Ticket::register(shared_claim.clone(), shared_claim, None)?;
    let exclusive_claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut exclusive = Ticket::register(exclusive_claim.clone(), exclusive_claim, None)?;

    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    table.set_record_state(shared.slot, STATE_WAITING)?;
    table.clear_record_blocked(shared.slot)?;
    table.set_record_state(exclusive.slot, STATE_WAITING)?;
    table.clear_record_blocked(exclusive.slot)?;
    set_cpu_availability_for_tests(&mut table, 1, CpuAvailability::SharedHeld)?;
    write_u64(&mut table.header, H_PENDING_FLAGS, 0);

    table.begin_transaction()?;
    table.mark_blocker_unknown(ContentionMarker {
        blocker: ResourceKey::Cpu(1),
        mode: FlockMode::Exclusive,
    })?;
    table.set_pending_flag(PENDING_RESCAN);
    table.finish_transaction()?;
    table.grant_compatible()?;
    if table
        .record(shared.slot)?
        .is_none_or(|record| record.state != STATE_WAITING)
        || table
            .record(exclusive.slot)?
            .is_none_or(|record| record.state != STATE_WAITING)
    {
        anyhow::bail!("unknown CPU modes must keep both SH and EX tickets waiting");
    }

    let request = table
        .observation_request()?
        .ok_or_else(|| anyhow::anyhow!("failed CPU EX probe must request an observation"))?;
    let scans_before = read_u64(&table.header, H_GRANT_SCANS);
    let sh_serial_before = table.resource_serial(S_CPU_SH, 1)?;
    let ex_serial_before = table.resource_serial(S_CPU_EX, 1)?;
    let shared_wake_before = atomic_u32(
        table
            .record_bytes(shared.slot)?
            .ok_or_else(|| anyhow::anyhow!("shared CPU ticket disappeared"))?,
        R_WAKE,
    )
    .load(Ordering::Acquire);
    let exclusive_wake_before = atomic_u32(
        table
            .record_bytes(exclusive.slot)?
            .ok_or_else(|| anyhow::anyhow!("exclusive CPU ticket disappeared"))?,
        R_WAKE,
    )
    .load(Ordering::Acquire);
    let mut observation = AvailabilityObservation::default();
    observation.cpus.insert(
        1,
        CpuObservation {
            availability: CpuAvailability::SharedHeld,
            sh_resolved: true,
            ex_resolved: true,
        },
    );
    drop(table);
    drop(_lock);

    let snapshot = coordinator.apply_observation(&request, &observation, || {}, None)?;
    if snapshot.should_step {
        anyhow::bail!(
            "an SH-only CPU improvement must not rerun the incompatible EX coordinator planner"
        );
    }

    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    let scans = read_u64(&table.header, H_GRANT_SCANS) - scans_before;
    let shared_granted = table
        .record(shared.slot)?
        .is_some_and(|record| record.state == STATE_GRANTED);
    let exclusive_waiting = table
        .record(exclusive.slot)?
        .is_some_and(|record| record.state == STATE_WAITING);
    let sh_serial_advanced = table.resource_serial(S_CPU_SH, 1)? > sh_serial_before;
    let ex_serial_unchanged = table.resource_serial(S_CPU_EX, 1)? == ex_serial_before;
    let shared_woke = atomic_u32(
        table
            .record_bytes(shared.slot)?
            .ok_or_else(|| anyhow::anyhow!("shared CPU ticket disappeared after grant"))?,
        R_WAKE,
    )
    .load(Ordering::Acquire)
        != shared_wake_before;
    let exclusive_not_woken = atomic_u32(
        table
            .record_bytes(exclusive.slot)?
            .ok_or_else(|| anyhow::anyhow!("exclusive CPU ticket disappeared after scan"))?,
        R_WAKE,
    )
    .load(Ordering::Acquire)
        == exclusive_wake_before;
    drop(table);
    drop(_lock);

    shared.finish(None)?;
    exclusive.finish(None)?;
    coordinator.finish(None)?;
    Ok(CpuExContentionSharedWake {
        scans,
        shared_granted,
        exclusive_waiting,
        sh_serial_advanced,
        ex_serial_unchanged,
        shared_woke,
        exclusive_not_woken,
        coordinator_did_not_replan: !snapshot.should_step,
    })
}

#[cfg(test)]
pub(super) fn exercise_coordinator_turnover_for_tests(
    coordinators: usize,
) -> Result<(u64, usize, u64, bool)> {
    if coordinators == 0 {
        anyhow::bail!("coordinator-turnover exercise needs at least one ticket");
    }
    let claim = ClaimSet::new(std::iter::empty(), [1usize, 2usize], FlockMode::Exclusive);
    let mut tickets = Vec::with_capacity(coordinators);
    for _ in 0..coordinators {
        tickets.push(Ticket::register(claim.clone(), claim.clone(), None)?);
    }

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        let (coordinator, waiters) = tickets
            .split_first()
            .ok_or_else(|| anyhow::anyhow!("coordinator-turnover ticket list disappeared"))?;
        table.restage_coordinator_for_tests(coordinator, waiters)?;
        for (cpu, available) in [(1usize, false), (2usize, true)] {
            set_cpu_free_for_tests(&mut table, cpu, available)?;
        }
        write_u64(&mut table.header, H_PENDING_FLAGS, 0);
        write_u64(&mut table.header, H_LAST_LIVENESS_SWEEP_NS, 0);
    }

    let sweeps_before = diagnostic_counter_for_tests(H_LIVENESS_SWEEP)?;
    let observations_before = diagnostic_counter_for_tests(H_OBSERVATION_REQUEST)?;
    let probes_before = LIVENESS_PROBES.with(std::cell::Cell::get);
    let empty = BTreeSet::new();
    let mut shared_reconcile_by = None;
    let mut reconcile_deadline_coalesced = true;
    for (index, ticket) in tickets.iter_mut().enumerate() {
        let snapshot = ticket.schedule(
            None,
            &empty,
            &empty,
            &empty,
            false,
            &[],
            &[],
            true,
            Some(Duration::from_secs(60 * 60)),
            false,
            None,
        )?;
        let reconcile_by = diagnostic_counter_for_tests(H_LIVENESS_RECONCILE_BY_NS)?;
        if index == 0 {
            // The deliberately due periodic sweep consumes the first
            // coordinator's request.
            reconcile_deadline_coalesced &= reconcile_by == 0;
        } else if let Some(shared) = shared_reconcile_by {
            // Every later handoff requests "one hour from now". The shared
            // deadline must remain the first request rather than sliding
            // forward once per coordinator.
            reconcile_deadline_coalesced &= reconcile_by == shared;
        } else {
            reconcile_deadline_coalesced &= reconcile_by != 0;
            shared_reconcile_by = Some(reconcile_by);
        }
        let request = snapshot
            .observation
            .ok_or_else(|| anyhow::anyhow!("known-busy CPU must remain observable"))?;
        if !request.cpus.contains_key(&1) || request.cpus.contains_key(&2) {
            anyhow::bail!(
                "post-watch reobserve must request only unavailable CPU 1, got {:?}",
                request.cpus.keys().collect::<Vec<_>>()
            );
        }
        ticket.finish(None)?;
    }
    let sweeps = diagnostic_counter_for_tests(H_LIVENESS_SWEEP)? - sweeps_before;
    let probes = LIVENESS_PROBES.with(std::cell::Cell::get) - probes_before;
    let observation_requests =
        diagnostic_counter_for_tests(H_OBSERVATION_REQUEST)? - observations_before;
    Ok((
        sweeps,
        probes,
        observation_requests,
        reconcile_deadline_coalesced,
    ))
}

#[cfg(test)]
pub(super) fn exercise_exact_commit_scan_elision_for_tests(commits: usize) -> Result<u64> {
    if commits == 0 {
        anyhow::bail!("exact-commit exercise needs at least one waiter");
    }
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let mut coordinator =
        Ticket::register(coordinator_claim.clone(), coordinator_claim.clone(), None)?;
    let mut waiters = Vec::with_capacity(commits);
    for index in 0..commits {
        let claim = ClaimSet::new(std::iter::empty(), [index + 1], FlockMode::Exclusive);
        waiters.push((Ticket::register(claim.clone(), claim.clone(), None)?, claim));
    }
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        table.restage_coordinator_for_tests(
            &coordinator,
            waiters.iter().map(|(ticket, _)| ticket),
        )?;
        let claim_epoch = table.claim_epoch();
        let prefix_bits = table.layout.bits;
        set_cpu_free_for_tests(&mut table, 0, false)?;
        let mut prefix = AggregateSnapshot::empty(table.layout);
        add_claim_bits(
            &coordinator_claim,
            &mut prefix.cpu_any,
            &mut prefix.cpu_exclusive,
            &mut prefix.llc_any,
            &mut prefix.llc_exclusive,
            prefix_bits,
        )?;
        for (ticket, claim) in &waiters {
            let record = table
                .record(ticket.slot)?
                .ok_or_else(|| anyhow::anyhow!("exact-commit waiter disappeared"))?;
            let issue_serial = table.max_watch_serial(&record.watch)?;
            table.publish_prefix(
                ticket.slot,
                &prefix,
                R_GRANT_EPOCH,
                claim_epoch,
                issue_serial,
            )?;
            table.set_record_state(ticket.slot, STATE_GRANTED)?;
            for &cpu in &claim.cpus {
                set_cpu_free_for_tests(&mut table, cpu, true)?;
            }
            add_claim_bits(
                claim,
                &mut prefix.cpu_any,
                &mut prefix.cpu_exclusive,
                &mut prefix.llc_any,
                &mut prefix.llc_exclusive,
                prefix_bits,
            )?;
        }
        write_u64(&mut table.header, H_PENDING_FLAGS, 0);
        table.finish_claim_scan();
    }

    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let empty = BTreeSet::new();
    let mut held = Vec::with_capacity(commits);
    for (ticket, claim) in &mut waiters {
        // Keep this helper falsifiable against the old implementation, whose
        // prior acquired commit advanced the epoch for every later grant.
        {
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            let claim_epoch = table.claim_epoch();
            table.set_record_state(ticket.slot, STATE_GRANTED)?;
            table.set_record_grant_epoch(ticket.slot, claim_epoch)?;
            table.finish_claim_scan();
        }
        let result = ticket.run_granted(
            None,
            |designated, _watch, acquisition_allowed, _predecessors, _availability| {
                if !acquisition_allowed || designated != claim {
                    anyhow::bail!("exact-commit waiter lost its prepared grant");
                }
                Ok(GrantAttempt {
                    acquired: Some(()),
                    preparation_claim: None,
                    preparation_contention: None,
                    next_claim: designated.clone(),
                    contention: None,
                })
            },
        )?;
        held.push(match result {
            GrantResult::Acquired((), held) => held,
            _ => anyhow::bail!("exact-commit waiter did not commit its prepared acquisition"),
        });
        coordinator.schedule(
            None,
            &empty,
            &empty,
            &empty,
            false,
            &[],
            &[],
            false,
            None,
            false,
            None,
        )?;
    }
    let scans = diagnostic_counter_for_tests(H_GRANT_SCANS)? - scans_before;
    drop(held);
    coordinator.finish(None)?;
    Ok(scans)
}

#[cfg(test)]
pub(super) fn exercise_mismatched_commit_rescan_for_tests() -> Result<(u64, bool)> {
    let old = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let watch = ClaimSet::new(std::iter::empty(), [1usize, 2usize], FlockMode::Exclusive);
    let mut first = Ticket::register(old, watch, None)?;
    let middle_claim = ClaimSet::new(std::iter::empty(), [3usize], FlockMode::Exclusive);
    let mut middle = Ticket::register(middle_claim.clone(), middle_claim, None)?;
    let later_claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut later = Ticket::register(later_claim.clone(), later_claim.clone(), None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_record_state(middle.slot, STATE_WAITING)?;
        table.set_record_state(later.slot, STATE_WAITING)?;
        for (cpu, available) in [(1usize, true), (2usize, true), (3usize, false)] {
            set_cpu_free_for_tests(&mut table, cpu, available)?;
        }
        write_u64(&mut table.header, H_PENDING_FLAGS, 0);
        table.finish_claim_scan();
    }
    let exact = ClaimSet::new(std::iter::empty(), [2usize], FlockMode::Exclusive);
    let commit_token = first.commit_token_for_tests()?;
    first.finish_acquired(&exact, commit_token, &[], None)?;
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let empty = BTreeSet::new();
    middle.schedule(
        None,
        &empty,
        &empty,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let scans = diagnostic_counter_for_tests(H_GRANT_SCANS)? - scans_before;
    let later_granted = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table
            .record(later.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED)
    };
    later.finish(None)?;
    middle.finish(None)?;
    Ok((scans, later_granted))
}

#[cfg(test)]
pub(super) fn exercise_superset_commit_rescan_for_tests() -> Result<(u64, bool)> {
    let old = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let watch = ClaimSet::new(std::iter::empty(), [1usize, 2usize], FlockMode::Exclusive);
    let mut first = Ticket::register(old, watch, None)?;
    let middle_claim = ClaimSet::new(std::iter::empty(), [3usize], FlockMode::Exclusive);
    let mut middle = Ticket::register(middle_claim.clone(), middle_claim, None)?;
    let later_claim = ClaimSet::new(std::iter::empty(), [2usize], FlockMode::Exclusive);
    let mut later = Ticket::register(later_claim.clone(), later_claim.clone(), None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let claim_epoch = table.claim_epoch();
        table.set_record_state(middle.slot, STATE_WAITING)?;
        table.set_record_state(later.slot, STATE_GRANTED)?;
        table.set_record_grant_epoch(later.slot, claim_epoch)?;
        for (cpu, available) in [(1usize, false), (2usize, true), (3usize, false)] {
            set_cpu_free_for_tests(&mut table, cpu, available)?;
        }
        write_u64(&mut table.header, H_PENDING_FLAGS, 0);
        table.finish_claim_scan();
    }
    let exact = ClaimSet::new(std::iter::empty(), [1usize, 2usize], FlockMode::Exclusive);
    let commit_token = first.commit_token_for_tests()?;
    first.finish_acquired(&exact, commit_token, &[], None)?;
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let empty = BTreeSet::new();
    middle.schedule(
        None,
        &empty,
        &empty,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let scans = diagnostic_counter_for_tests(H_GRANT_SCANS)? - scans_before;
    let later_revoked = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table
            .record(later.slot)?
            .is_some_and(|record| record.state == STATE_REVOKED)
    };
    let mut stale_callback_ran = false;
    let revoked_result = later.run_granted(
        None,
        |_designated, _watch, _allowed, _predecessors, _availability| {
            stale_callback_ran = true;
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: later_claim.clone(),
                contention: None,
            })
        },
    )?;
    let revoked_ack_published = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table
            .record(later.slot)?
            .is_some_and(|record| record.state == STATE_WAITING)
            && table.pending_flags() & PENDING_RESCAN != 0
    };
    later.finish(None)?;
    middle.finish(None)?;
    Ok((
        scans,
        later_revoked
            && matches!(revoked_result, GrantResult::LostGrant)
            && !stale_callback_ran
            && revoked_ack_published,
    ))
}

#[cfg(test)]
pub(super) fn exercise_shared_commit_improvement_for_tests() -> Result<(u64, bool)> {
    let shared = ClaimSet::new([1usize], std::iter::empty(), FlockMode::Shared);
    let mut first = Ticket::register(shared.clone(), shared, None)?;
    let middle_claim = ClaimSet::new(std::iter::empty(), [3usize], FlockMode::Exclusive);
    let mut middle = Ticket::register(middle_claim.clone(), middle_claim, None)?;
    let later_claim = ClaimSet::new([1usize], std::iter::empty(), FlockMode::Shared);
    let mut later = Ticket::register(later_claim.clone(), later_claim, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_record_state(middle.slot, STATE_WAITING)?;
        table.set_record_state(later.slot, STATE_WAITING)?;
        table.set_bitmap_bit(B_LLC_KNOWN, 1, true)?;
        table.set_bitmap_bit(B_LLC_SH_AVAILABLE, 1, false)?;
        table.set_bitmap_bit(B_LLC_EX_AVAILABLE, 1, false)?;
        table.set_bitmap_bit(B_PENDING_LLC_SH, 1, false)?;
        table.set_bitmap_bit(B_PENDING_LLC_EX, 1, false)?;
        table.set_bitmap_bit(B_CANDIDATE_LLC_SH, 1, false)?;
        table.set_bitmap_bit(B_CANDIDATE_LLC_EX, 1, false)?;
        set_cpu_free_for_tests(&mut table, 3, false)?;
        write_u64(&mut table.header, H_PENDING_FLAGS, 0);
        table.finish_claim_scan();
    }
    let exact = ClaimSet::new([1usize], std::iter::empty(), FlockMode::Shared);
    let commit_token = first.commit_token_for_tests()?;
    first.finish_acquired(&exact, commit_token, &[], None)?;
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let empty = BTreeSet::new();
    middle.schedule(
        None,
        &empty,
        &empty,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let scans = diagnostic_counter_for_tests(H_GRANT_SCANS)? - scans_before;
    let later_granted = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table
            .record(later.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED)
    };
    later.finish(None)?;
    middle.finish(None)?;
    Ok((scans, later_granted))
}

#[cfg(test)]
pub(super) fn exercise_cpu_shared_commit_improvement_for_tests() -> Result<(u64, bool)> {
    let shared = ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let mut first = Ticket::register(shared.clone(), shared.clone(), None)?;
    let middle_claim = ClaimSet::new(std::iter::empty(), [3usize], FlockMode::Exclusive);
    let mut middle = Ticket::register(middle_claim.clone(), middle_claim, None)?;
    let later_claim = shared.clone();
    let mut later = Ticket::register(later_claim.clone(), later_claim, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_record_state(middle.slot, STATE_WAITING)?;
        table.set_record_state(later.slot, STATE_WAITING)?;
        set_cpu_availability_for_tests(&mut table, 1, CpuAvailability::ExclusiveHeld)?;
        set_cpu_free_for_tests(&mut table, 3, false)?;
        write_u64(&mut table.header, H_PENDING_FLAGS, 0);
        table.finish_claim_scan();
    }
    let commit_token = first.commit_token_for_tests()?;
    first.finish_acquired(&shared, commit_token, &[], None)?;
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let empty = BTreeSet::new();
    middle.schedule(
        None,
        &empty,
        &empty,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let scans = diagnostic_counter_for_tests(H_GRANT_SCANS)? - scans_before;
    let later_granted = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table
            .record(later.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED)
    };
    later.finish(None)?;
    middle.finish(None)?;
    Ok((scans, later_granted))
}

#[cfg(test)]
pub(super) fn exercise_cpu_mode_repair_for_tests() -> Result<(bool, bool, bool, bool)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let flexible_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let flexible_watch = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut flexible = Ticket::register(flexible_claim.clone(), flexible_watch.clone(), None)?;
    let fixed_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [2usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let mut fixed = Ticket::register(fixed_claim.clone(), fixed_claim.clone(), None)?;

    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.set_record_state(flexible.slot, STATE_WAITING)?;
    table.set_record_state(fixed.slot, STATE_WAITING)?;
    // Model a writer dying after marking the aggregate transaction dirty.
    // Recovery must rebuild from the record's independently encoded exact and
    // watch CPU modes.
    table.begin_transaction()?;
    table.repair_consistency_if_needed()?;
    let observation = table
        .observation_request()?
        .ok_or_else(|| anyhow::anyhow!("dirty repair omitted its availability observation"))?;
    let flexible_modes = observation
        .cpus
        .get(&1)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("dirty repair omitted the flexible CPU watch"))?;
    let fixed_modes = observation
        .cpus
        .get(&2)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("dirty repair omitted the fixed CPU watch"))?;
    if flexible_modes.0.is_none()
        || flexible_modes.1.is_none()
        || fixed_modes.0.is_none()
        || fixed_modes.1.is_some()
    {
        anyhow::bail!(
            "dirty repair reconstructed the wrong CPU watch modes: \
             flexible={flexible_modes:?}, fixed={fixed_modes:?}"
        );
    }
    let flexible_record = table
        .record(flexible.slot)?
        .ok_or_else(|| anyhow::anyhow!("flexible CPU record disappeared during repair"))?;
    let fixed_record = table
        .record(fixed.slot)?
        .ok_or_else(|| anyhow::anyhow!("fixed CPU record disappeared during repair"))?;
    let flexible_preserved = flexible_record.claim == flexible_claim
        && flexible_record.watch == flexible_watch
        && flexible_record.state == STATE_WAITING
        && flexible_record.prefix_epoch == 0;
    let fixed_preserved = fixed_record.claim == fixed_claim
        && fixed_record.watch == fixed_claim
        && fixed_record.state == STATE_WAITING;
    let flexible_still_flexible = claim_is_flexible(&flexible_record.claim, &flexible_record.watch);
    let fixed_still_fixed = !claim_is_flexible(&fixed_record.claim, &fixed_record.watch);
    drop(table);
    drop(_lock);

    fixed.finish(None)?;
    flexible.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        flexible_preserved,
        flexible_still_flexible,
        fixed_preserved,
        fixed_still_fixed,
    ))
}

#[cfg(test)]
pub(crate) struct ReplanTokenWaveOutcome {
    pub(crate) registration_waiting: bool,
    pub(crate) exact_grants: usize,
    pub(crate) initial_replans: usize,
    pub(crate) initial_wakes: usize,
    pub(crate) initial_prefix_comparisons: usize,
    pub(crate) initial_full_watch_materializations: usize,
    pub(crate) initial_encoded_watch_serial_walks: usize,
    pub(crate) memo_identical_waiters: usize,
    pub(crate) memo_identical_replans: usize,
    pub(crate) memo_identical_serial_walks: usize,
    pub(crate) memo_identical_layout_words: usize,
    pub(crate) memo_identical_exact_word_reads: usize,
    pub(crate) memo_identical_claim_heap_spills: usize,
    pub(crate) memo_mixed_waiters: usize,
    pub(crate) memo_mixed_replans: usize,
    pub(crate) memo_mixed_serial_walks: usize,
    pub(crate) memo_guard_waiters: usize,
    pub(crate) memo_guard_initial_capacity: usize,
    pub(crate) memo_guard_initial_replans: usize,
    pub(crate) memo_guard_saturated_replans: usize,
    pub(crate) memo_guard_saturated_serial_walks: usize,
    pub(crate) memo_guard_closed_replans: usize,
    pub(crate) memo_guard_closed_serial_walks: usize,
    pub(crate) memo_guard_opened_replans: usize,
    pub(crate) memo_guard_opened_serial_walks: usize,
    pub(crate) initial_full_prefix_snapshot_publishes: usize,
    pub(crate) repeated_replans: usize,
    pub(crate) repeated_wakes: usize,
    pub(crate) fixed_waiter_granted: bool,
    pub(crate) fixed_waiter_woken: bool,
    pub(crate) fixed_scan_replans: usize,
    pub(crate) fixed_scan_replan_wakes: usize,
    pub(crate) callback_requeued: bool,
    pub(crate) callback_prefix_reads: usize,
    pub(crate) callback_active_reads: usize,
    pub(crate) mixed_age_old_replanned: bool,
    pub(crate) mixed_age_old_woken: bool,
    pub(crate) mixed_age_late_replanned: bool,
    pub(crate) mixed_age_late_woken: bool,
    pub(crate) mixed_age_repeated_replans: usize,
    pub(crate) mixed_age_repeated_wakes: usize,
    pub(crate) mixed_age_stragglers_remaining: usize,
    pub(crate) mixed_age_outstanding: usize,
    pub(crate) mixed_age_horizon_extended: bool,
    pub(crate) mixed_age_deadline_preserved: bool,
    pub(crate) mixed_age_repeated_outstanding: usize,
    pub(crate) mixed_age_callbacks_drained: bool,
    pub(crate) mixed_age_clock_cleared: bool,
    pub(crate) mixed_age_post_drain_replans: usize,
    pub(crate) mixed_age_post_drain_wakes: usize,
}

#[cfg(test)]
struct EncodedWatchSerialMemoCaseOutcome {
    initial_replans: usize,
    initial_serial_walks: usize,
    layout_words: usize,
    initial_exact_word_reads: usize,
    initial_claim_heap_spills: usize,
    saturated_replans: usize,
    saturated_serial_walks: usize,
    closed_replans: usize,
    closed_serial_walks: usize,
    opened_replans: usize,
    opened_serial_walks: usize,
}

#[cfg(test)]
fn exercise_encoded_watch_serial_memo_case_for_tests(
    waiter_count: usize,
    mixed_keys: bool,
    initial_capacity: usize,
) -> Result<EncodedWatchSerialMemoCaseOutcome> {
    anyhow::ensure!(waiter_count >= 6, "watch memo exercise needs six waiters");
    anyhow::ensure!(
        initial_capacity != 0 && initial_capacity <= waiter_count,
        "watch memo exercise capacity must fit its waiter count",
    );
    let coordinator_cpu = 2usize;
    let common_cpu = 20usize;
    let blocker_a_cpu = 21usize;
    let blocker_b_cpu = 22usize;
    let distinct_cpu = 23usize;
    let first_waiter_cpu = 100usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;

    let base_watch = ClaimSet::new(
        std::iter::empty(),
        (first_waiter_cpu..first_waiter_cpu + waiter_count).chain([
            common_cpu,
            blocker_a_cpu,
            blocker_b_cpu,
        ]),
        FlockMode::Exclusive,
    );
    let mut waiters = Vec::with_capacity(waiter_count);
    for index in 0..waiter_count {
        let claim = ClaimSet::new(
            std::iter::empty(),
            [first_waiter_cpu + index],
            FlockMode::Exclusive,
        );
        let mut watch = base_watch.clone();
        if mixed_keys && index == 2 {
            watch.cpus.insert(distinct_cpu);
        }
        let blocker = if mixed_keys {
            match index {
                3 | 4 => Some(ContentionMarker {
                    blocker: ResourceKey::Cpu(blocker_a_cpu),
                    mode: FlockMode::Exclusive,
                }),
                5 => Some(ContentionMarker {
                    blocker: ResourceKey::Cpu(blocker_b_cpu),
                    mode: FlockMode::Exclusive,
                }),
                _ => None,
            }
        } else {
            None
        };
        waiters.push((Ticket::register(claim, watch, None)?, blocker));
    }

    let outcome = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_replan_capacity_for_tests(initial_capacity)?;
        table.restage_coordinator_for_tests(
            &coordinator,
            waiters.iter().map(|(ticket, _)| ticket),
        )?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        for cpu in &base_watch.cpus {
            set_cpu_free_for_tests(&mut table, *cpu, false)?;
        }
        if mixed_keys {
            set_cpu_free_for_tests(&mut table, distinct_cpu, false)?;
        }
        for (ticket, blocker) in &waiters {
            if let Some(blocker) = blocker {
                table.set_record_blocked(ticket.slot, *blocker, 0)?;
            }
        }
        table.stamp_resource_improvement(S_CPU_EX, common_cpu)?;
        table.set_pending_flag(PENDING_RESCAN);
        let walks_before = ENCODED_WATCH_SERIAL_WALKS.with(std::cell::Cell::get);
        let exact_reads_before = SCAN_EXACT_WORD_READS.with(std::cell::Cell::get);
        let heap_spills_before = SCAN_CLAIM_HEAP_SPILLS.with(std::cell::Cell::get);
        table.grant_compatible()?;
        let initial_serial_walks = ENCODED_WATCH_SERIAL_WALKS
            .with(std::cell::Cell::get)
            .saturating_sub(walks_before);
        let initial_exact_word_reads = SCAN_EXACT_WORD_READS
            .with(std::cell::Cell::get)
            .saturating_sub(exact_reads_before);
        let initial_claim_heap_spills = SCAN_CLAIM_HEAP_SPILLS
            .with(std::cell::Cell::get)
            .saturating_sub(heap_spills_before);
        let count_replans = |table: &mut Table| -> Result<usize> {
            waiters.iter().try_fold(0usize, |count, (ticket, _)| {
                Ok(count
                    + usize::from(
                        table
                            .record(ticket.slot)?
                            .is_some_and(|record| record.state == STATE_REPLAN),
                    ))
            })
        };
        let initial_replans = count_replans(&mut table)?;

        let mut saturated_replans = initial_replans;
        let mut saturated_serial_walks = 0usize;
        let mut closed_replans = initial_replans;
        let mut closed_serial_walks = 0usize;
        let mut opened_replans = initial_replans;
        let mut opened_serial_walks = 0usize;
        if initial_capacity < waiter_count {
            table.set_pending_flag(PENDING_RESCAN);
            let walks_before = ENCODED_WATCH_SERIAL_WALKS.with(std::cell::Cell::get);
            table.grant_compatible()?;
            saturated_serial_walks = ENCODED_WATCH_SERIAL_WALKS
                .with(std::cell::Cell::get)
                .saturating_sub(walks_before);
            saturated_replans = count_replans(&mut table)?;

            table.set_replan_capacity_for_tests(waiter_count)?;
            let closed_now = monotonic_now_ns()?.max(2);
            write_u64(&mut table.header, H_REPLAN_WAVE_STARTED_NS, closed_now - 1);
            write_u64(&mut table.header, H_REPLAN_WAVE_DEADLINE_NS, closed_now);
            table.set_pending_flag(PENDING_RESCAN);
            let walks_before = ENCODED_WATCH_SERIAL_WALKS.with(std::cell::Cell::get);
            table.grant_compatible_at(closed_now, None)?;
            closed_serial_walks = ENCODED_WATCH_SERIAL_WALKS
                .with(std::cell::Cell::get)
                .saturating_sub(walks_before);
            closed_replans = count_replans(&mut table)?;

            let opened_now = closed_now.saturating_add(1);
            table.arm_replan_wave_at(opened_now);
            table.set_pending_flag(PENDING_RESCAN);
            let walks_before = ENCODED_WATCH_SERIAL_WALKS.with(std::cell::Cell::get);
            table.grant_compatible_at(opened_now, None)?;
            opened_serial_walks = ENCODED_WATCH_SERIAL_WALKS
                .with(std::cell::Cell::get)
                .saturating_sub(walks_before);
            opened_replans = count_replans(&mut table)?;
        }
        EncodedWatchSerialMemoCaseOutcome {
            initial_replans,
            initial_serial_walks,
            layout_words: table.layout.words,
            initial_exact_word_reads,
            initial_claim_heap_spills,
            saturated_replans,
            saturated_serial_walks,
            closed_replans,
            closed_serial_walks,
            opened_replans,
            opened_serial_walks,
        }
    };

    for (ticket, _) in waiters.iter_mut().rev() {
        ticket.finish(None)?;
    }
    coordinator.finish(None)?;
    Ok(outcome)
}

#[cfg(test)]
pub(crate) struct ReplanChangedWaveOutcome {
    pub(crate) callbacks: usize,
    pub(crate) intermediate_notify_delta: u64,
    pub(crate) intermediate_scan_delta: u64,
    pub(crate) intermediate_generation_wake_delta: u32,
    pub(crate) intermediate_rescan_coalesced: bool,
    pub(crate) final_scan_delta_before_authoritative: u64,
    pub(crate) final_notify_delta: u64,
    pub(crate) final_generation_wake_delta: u32,
    pub(crate) final_rescan_edge: bool,
    pub(crate) authoritative_scan_delta: u64,
    pub(crate) authoritative_flags_clear: bool,
    pub(crate) replacements_preserved: bool,
}

#[cfg(test)]
pub(crate) struct QuietGenerationAdditionsOutcome {
    pub(crate) additions: usize,
    pub(crate) pending_generation_delta: u64,
    pub(crate) pending_wake_delta: u32,
    pub(crate) pending_release_wake_delta: u32,
    pub(crate) waiting_generation_delta: u64,
    pub(crate) waiting_wake_delta: u32,
    pub(crate) waiting_release_wake_delta: u32,
    pub(crate) waiting_state_count: usize,
    pub(crate) granted_generation_delta: u64,
    pub(crate) granted_wake_delta: u32,
    pub(crate) granted_release_wake_delta: u32,
    pub(crate) granted_state_count: usize,
    pub(crate) held_generation_delta: u64,
    pub(crate) held_wake_delta: u32,
    pub(crate) held_release_wake_delta: u32,
}

#[cfg(test)]
pub(crate) struct DeferredRescanPolicyOutcome {
    pub(crate) registration_and_teardown_coalesced: bool,
    pub(crate) known_free_release_preserved_deferred_fast_path: bool,
    pub(crate) ordinary_turn_preserved_deadline: bool,
    pub(crate) heartbeat_promoted_before_deadline: bool,
    pub(crate) observation_survived_promotion_and_scan: bool,
    pub(crate) exact_deadline_promoted: bool,
    pub(crate) final_drain_promoted: bool,
}

#[cfg(test)]
pub(crate) struct GrantCompletionBatchOutcome {
    pub(crate) first_completion_notified: bool,
    pub(crate) later_deferred_deadline_shortened: bool,
    pub(crate) second_completion_coalesced: bool,
    pub(crate) deadline_was_not_renewed: bool,
    pub(crate) no_scan_before_deadline: bool,
    pub(crate) one_scan_at_deadline: bool,
}

#[cfg(test)]
pub(crate) struct ReleaseCoalesceOutcome {
    /// The `holders` releases armed exactly one deferred edge (not one urgent
    /// scan each), and all but the first were counted as coalesced.
    pub(crate) releases_coalesced_into_one_edge: bool,
    /// No authoritative scan ran before the coalescing deadline, so the herd of
    /// releases did not each drive an O(N) scan.
    pub(crate) no_scan_before_deadline: bool,
    /// One scan at the deadline granted every waiter the releases freed —
    /// work-conserving, nothing left grantable.
    pub(crate) one_scan_at_deadline_grants_all: bool,
    /// A release arriving after that scan cleared the edge re-armed the next
    /// scan (the no-lost-edge handshake).
    pub(crate) post_scan_release_reschedules: bool,
}

#[cfg(test)]
pub(crate) struct BoundedReplanWindowOutcome {
    pub(crate) capacity: usize,
    pub(crate) peak_outstanding: usize,
    pub(crate) slices: Vec<Vec<usize>>,
    pub(crate) disjoint_exact_granted: bool,
}

#[cfg(test)]
pub(crate) struct ReplanStragglerProgressOutcome {
    pub(crate) completion_requeued: bool,
    pub(crate) callback_notify_delta: u64,
    pub(crate) deferred_not_due_early: bool,
    pub(crate) deferred_due_at_deadline: bool,
    pub(crate) callback_scan_delta: u64,
    pub(crate) callback_generation_wake_delta: u32,
    pub(crate) edge_coalesced_with_straggler: bool,
    pub(crate) later_grant_fenced_until_scan: bool,
    pub(crate) later_grant_demotion_shortened_deferred_edge: bool,
    pub(crate) later_grant_demotion_notified_once: bool,
    pub(crate) later_grant_demotion_deadline_exact: bool,
    pub(crate) later_conflicting_grant_not_regranted: bool,
    pub(crate) authoritative_scan_delta: u64,
    pub(crate) completed_replacement_granted: bool,
    pub(crate) straggler_still_replan: bool,
    pub(crate) wave_deadline_not_reached: bool,
}

#[cfg(test)]
pub(crate) struct PendingReplanGrantRaceOutcome {
    pub(crate) entry_callback_suppressed: bool,
    pub(crate) entry_conflict_blocked_after_scan: bool,
    pub(crate) entry_authoritative_scan_delta: u64,
    pub(crate) commit_callback_entered: bool,
    pub(crate) commit_rejected: bool,
    pub(crate) commit_conflict_blocked_after_scan: bool,
    pub(crate) commit_authoritative_scan_delta: u64,
}

#[cfg(test)]
pub(crate) struct CoordinatorPendingReplanOutcome {
    pub(crate) acquisition_conflict_rejected: bool,
    pub(crate) acquisition_disjoint_preserved: bool,
    pub(crate) preparation_conflict_rejected: bool,
    pub(crate) preparation_disjoint_preserved: bool,
    pub(crate) acquisition_conflict_scan_delta: u64,
    pub(crate) acquisition_disjoint_scan_delta: u64,
    pub(crate) preparation_conflict_scan_delta: u64,
    pub(crate) preparation_disjoint_scan_delta: u64,
}

#[cfg(test)]
pub(crate) struct ReplanWaveExpiryOutcome {
    pub(crate) incremental_late_published: bool,
    pub(crate) incremental_horizon_extended: bool,
    pub(crate) incremental_deadline_preserved: bool,
    pub(crate) ordinary_wave_not_expired_early: bool,
    pub(crate) completed_replacement_preserved: bool,
    pub(crate) stragglers_quarantined: bool,
    pub(crate) wave_drained_and_clock_cleared: bool,
    pub(crate) expiration_avoids_global_generation_wake: bool,
    pub(crate) completed_replacement_granted: bool,
    pub(crate) expired_tickets_not_reissued: bool,
    pub(crate) late_completion_rejected: bool,
    pub(crate) late_replacement_rejected: bool,
    pub(crate) completion_acknowledged_waiting: bool,
    pub(crate) entry_callback_suppressed: bool,
    pub(crate) entry_acknowledged_waiting: bool,
    pub(crate) acknowledgement_rescan_edge: bool,
}

#[cfg(test)]
pub(crate) struct ReplanCrashRepairOutcome {
    pub(crate) dirty_repair_completed: bool,
    pub(crate) repair_generation_advanced: bool,
    pub(crate) repair_generation_woke: bool,
    pub(crate) torn_callbacks_demoted: bool,
    pub(crate) cursor_preserved: bool,
    pub(crate) horizon_preserved: bool,
    pub(crate) all_eligible_recovered: bool,
    pub(crate) recovered_replans: usize,
    pub(crate) recovered_wakes: usize,
    pub(crate) repeated_replans: usize,
    pub(crate) repeated_wakes: usize,
}

/// Model a process dying at `replan_state_and_cursor_before_wake` while the
/// registry flock is transferred to its successor. The interrupted writer has
/// published the callback prefix, REPLAN state, finite-round horizon, and
/// cursor, but neither the futex wake nor the clean transaction marker.
#[cfg(test)]
pub(super) fn exercise_replan_crash_repair_for_tests() -> Result<ReplanCrashRepairOutcome> {
    let coordinator_cpu = 40usize;
    let common_cpu = 41usize;
    let first_cpu = 42usize;
    let second_cpu = 43usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let first_claim = ClaimSet::new(std::iter::empty(), [first_cpu], FlockMode::Exclusive);
    let first_watch = ClaimSet::new(
        std::iter::empty(),
        [first_cpu, common_cpu],
        FlockMode::Exclusive,
    );
    let mut first = Ticket::register(first_claim, first_watch, None)?;
    let second_claim = ClaimSet::new(std::iter::empty(), [second_cpu], FlockMode::Exclusive);
    let second_watch = ClaimSet::new(
        std::iter::empty(),
        [second_cpu, common_cpu],
        FlockMode::Exclusive,
    );
    let mut second = Ticket::register(second_claim, second_watch, None)?;

    let outcome = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        for cpu in [coordinator_cpu, common_cpu, first_cpu, second_cpu] {
            set_cpu_free_for_tests(&mut table, cpu, false)?;
        }
        let first_record = table
            .record(first.slot)?
            .ok_or_else(|| anyhow::anyhow!("first torn REPLAN ticket disappeared"))?;
        let second_record = table
            .record(second.slot)?
            .ok_or_else(|| anyhow::anyhow!("second torn REPLAN ticket disappeared"))?;
        anyhow::ensure!(
            first_record.state == STATE_WAITING && second_record.state == STATE_WAITING,
            "flexible crash-repair tickets did not begin WAITING",
        );
        let wake = |table: &mut Table, slot: u64| -> Result<u32> {
            let bytes = table
                .record_bytes(slot)?
                .ok_or_else(|| anyhow::anyhow!("crash-repair slot {slot} disappeared"))?;
            Ok(read_u32(bytes, R_WAKE))
        };
        let first_wake_before = wake(&mut table, first.slot)?;
        let second_wake_before = wake(&mut table, second.slot)?;
        let expected_cursor = first.ticket;
        let expected_horizon = read_u64(&table.header, H_NEXT_TICKET).saturating_sub(1);
        let generation_before = table.generation();
        let generation_wake_before = table.generation_wake();

        table.begin_transaction()?;
        let prefix = AggregateSnapshot::empty(table.layout);
        let epoch = table.claim_epoch();
        let mut issue_serial = table.max_watch_serial(&first_record.watch)?;
        if let Some(blocked) = first_record.blocked_on {
            issue_serial = issue_serial.max(table.blocker_serial(blocked.key, blocked.mode)?);
        }
        table.publish_prefix(
            first.slot,
            &prefix,
            R_REPLAN_CLAIM_EPOCH,
            epoch,
            issue_serial,
        )?;
        table.set_record_state(first.slot, STATE_REPLAN)?;
        table.clear_record_blocked(first.slot)?;
        write_u64(&mut table.header, H_REPLAN_HORIZON, expected_horizon);
        write_u64(&mut table.header, H_REPLAN_CURSOR, expected_cursor);
        crash_at_for_tests("replan_state_and_cursor_before_wake");

        table.repair_consistency_if_needed()?;
        let dirty_repair_completed =
            atomic_u64(&table.header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) == 0;
        let repair_generation_advanced = table.generation() > generation_before;
        let repair_generation_woke = table.generation_wake() != generation_wake_before;
        let repaired_records = [first.slot, second.slot]
            .into_iter()
            .map(|slot| {
                table
                    .record(slot)?
                    .ok_or_else(|| anyhow::anyhow!("repaired REPLAN slot {slot} disappeared"))
            })
            .collect::<Result<Vec<_>>>()?;
        let torn_callbacks_demoted = repaired_records
            .iter()
            .all(|record| record.state == STATE_WAITING && record.prefix_epoch == 0);
        let cursor_preserved = read_u64(&table.header, H_REPLAN_CURSOR) == expected_cursor;
        let horizon_preserved = read_u64(&table.header, H_REPLAN_HORIZON) == expected_horizon
            && expected_horizon >= expected_cursor;
        anyhow::ensure!(
            wake(&mut table, first.slot)? == first_wake_before
                && wake(&mut table, second.slot)? == second_wake_before,
            "dirty repair woke a speculative callback before republishing its token",
        );

        table.grant_compatible()?;
        let recovered_records = [first.slot, second.slot]
            .into_iter()
            .map(|slot| {
                table
                    .record(slot)?
                    .ok_or_else(|| anyhow::anyhow!("recovered REPLAN slot {slot} disappeared"))
            })
            .collect::<Result<Vec<_>>>()?;
        let recovered_replans = recovered_records
            .iter()
            .filter(|record| record.state == STATE_REPLAN)
            .count();
        let all_eligible_recovered = recovered_records
            .iter()
            .all(|record| record.state == STATE_REPLAN);
        let recovered_wakes = usize::from(wake(&mut table, first.slot)? != first_wake_before)
            + usize::from(wake(&mut table, second.slot)? != second_wake_before);

        let first_wake_after_recovery = wake(&mut table, first.slot)?;
        let second_wake_after_recovery = wake(&mut table, second.slot)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let repeated_replans = [first.slot, second.slot]
            .into_iter()
            .map(|slot| table.record(slot))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .filter(|record| record.state == STATE_REPLAN)
            .count();
        let repeated_wakes =
            usize::from(wake(&mut table, first.slot)? != first_wake_after_recovery)
                + usize::from(wake(&mut table, second.slot)? != second_wake_after_recovery);

        ReplanCrashRepairOutcome {
            dirty_repair_completed,
            repair_generation_advanced,
            repair_generation_woke,
            torn_callbacks_demoted,
            cursor_preserved,
            horizon_preserved,
            all_eligible_recovered,
            recovered_replans,
            recovered_wakes,
            repeated_replans,
            repeated_wakes,
        }
    };

    second.finish(None)?;
    first.finish(None)?;
    coordinator.finish(None)?;
    Ok(outcome)
}

#[cfg(test)]
pub(super) fn exercise_intrascan_fence_epoch_for_tests() -> Result<(bool, bool, bool)> {
    let coordinator_cpu = 50usize;
    let earlier_cpu = 51usize;
    let later_cpu = 52usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let earlier_claim = ClaimSet::new(std::iter::empty(), [earlier_cpu], FlockMode::Exclusive);
    let mut earlier = Ticket::register(earlier_claim.clone(), earlier_claim, None)?;
    let later_claim = ClaimSet::new(std::iter::empty(), [later_cpu], FlockMode::Exclusive);
    let later_watch = ClaimSet::new(
        std::iter::empty(),
        [earlier_cpu, later_cpu],
        FlockMode::Exclusive,
    );
    let mut later = Ticket::register(later_claim.clone(), later_watch, None)?;

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        for cpu in [coordinator_cpu, earlier_cpu, later_cpu] {
            set_cpu_free_for_tests(&mut table, cpu, false)?;
        }
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table
                .record(later.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN),
            "intra-scan epoch setup did not publish the later REPLAN token",
        );
    }

    let mut earlier_granted = false;
    let mut publication_changed = false;
    let later_slot = later.slot;
    let result = later.run_granted(
        None,
        |designated, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && designated == &later_claim,
                "intra-scan epoch test entered the wrong callback",
            );
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            let epoch_before = table
                .record(later_slot)?
                .ok_or_else(|| anyhow::anyhow!("later REPLAN disappeared before refresh"))?
                .replan_claim_epoch;
            set_cpu_free_for_tests(&mut table, earlier_cpu, true)?;
            table.set_pending_flag(PENDING_RESCAN);
            table.grant_compatible()?;
            let earlier_record = table
                .record(earlier.slot)?
                .ok_or_else(|| anyhow::anyhow!("earlier fixed waiter disappeared"))?;
            let later_record = table
                .record(later_slot)?
                .ok_or_else(|| anyhow::anyhow!("later REPLAN disappeared after refresh"))?;
            earlier_granted = earlier_record.state == STATE_GRANTED;
            publication_changed = later_record.state == STATE_REPLAN
                && later_record.replan_claim_epoch != epoch_before
                && later_record.prefix_epoch == later_record.replan_claim_epoch;
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: designated.clone(),
                contention: None,
            })
        },
    )?;
    // This fixture owns a synthetic coordinator without its production
    // heartbeat loop. Inspect the callback result without `Ticket::state`,
    // whose deliberately mutating liveness recovery may elect this waiter if
    // an overloaded test process is descheduled beyond the coordinator lease.
    let completion_accepted_for_revalidation = matches!(result, GrantResult::Requeued) && {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table
            .record(later.slot)?
            .is_some_and(|record| record.ticket == later.ticket && record.state == STATE_WAITING)
    };

    later.finish(None)?;
    earlier.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        earlier_granted,
        publication_changed,
        completion_accepted_for_revalidation,
    ))
}

#[cfg(test)]
pub(super) fn exercise_grant_scan_crash_fence_for_tests() -> Result<(bool, bool, bool)> {
    let coordinator_cpu = 60usize;
    let replacement_cpu = 61usize;
    let initial_cpu = 62usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let initial_claim = ClaimSet::new(std::iter::empty(), [initial_cpu], FlockMode::Exclusive);
    let flexible_watch = ClaimSet::new(
        std::iter::empty(),
        [replacement_cpu, initial_cpu],
        FlockMode::Exclusive,
    );
    let mut earlier = Ticket::register(initial_claim.clone(), flexible_watch, None)?;
    let later_claim = ClaimSet::new(std::iter::empty(), [replacement_cpu], FlockMode::Exclusive);
    let mut later = Ticket::register(later_claim.clone(), later_claim.clone(), None)?;

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, initial_cpu, false)?;
        set_cpu_free_for_tests(&mut table, replacement_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table
                .record(earlier.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN)
                && table
                    .record(later.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED),
            "grant-crash setup did not publish earlier REPLAN plus later GRANTED",
        );
    }
    let replacement = later_claim.clone();
    let replan_result = earlier.run_granted(
        None,
        |designated, _watch, allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !allowed && designated == &initial_claim,
                "grant-crash setup entered the wrong REPLAN callback",
            );
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: replacement,
                contention: None,
            })
        },
    )?;
    anyhow::ensure!(
        matches!(replan_result, GrantResult::Requeued),
        "earlier REPLAN did not publish its conflicting WAITING replacement",
    );

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(earlier.slot)?
            .filter(|record| record.state == STATE_WAITING && record.claim == later_claim)
            .ok_or_else(|| anyhow::anyhow!("earlier conflicting replacement disappeared"))?;
        table.begin_transaction()?;
        let prefix = AggregateSnapshot::empty(table.layout);
        let epoch = table.claim_epoch();
        let issue_serial = table.max_watch_serial(&record.claim)?;
        table.publish_prefix(record.slot, &prefix, R_GRANT_EPOCH, epoch, issue_serial)?;
        table.set_record_state(record.slot, STATE_GRANTED)?;
        table.clear_record_blocked(record.slot)?;
        crash_at_for_tests("grant_state_before_wake");
        // Deliberately omit both the wake and the later-GRANTED revocation:
        // this is the exact torn middle of the authoritative scan.
    }

    let later_slot = later.slot;
    let mut stale_later_callback_ran = false;
    let later_result: GrantResult<()> = later.run_granted(
        None,
        |_designated, _watch, _allowed, _predecessors, _availability| {
            stale_later_callback_ran = true;
            anyhow::bail!("later stale GRANTED callback entered after scanner death")
        },
    )?;
    let stale_later_rejected =
        !stale_later_callback_ran && matches!(later_result, GrantResult::LostGrant);

    let (later_demoted, earlier_regranted) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let later_demoted = table
            .record(later_slot)?
            .is_some_and(|record| record.state == STATE_WAITING);
        set_cpu_free_for_tests(&mut table, replacement_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let earlier_regranted = table
            .record(earlier.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED);
        (later_demoted, earlier_regranted)
    };

    later.finish(None)?;
    earlier.finish(None)?;
    coordinator.finish(None)?;
    Ok((stale_later_rejected, later_demoted, earlier_regranted))
}

/// Exercise a deliberately tiny planner window while every completed early
/// callback becomes eligible again. The cyclic cursor must visit the later
/// tickets instead of letting that changing prefix monopolize each refill.
#[cfg(test)]
pub(super) fn exercise_bounded_replan_window_for_tests(
    capacity: usize,
    waiter_count: usize,
) -> Result<BoundedReplanWindowOutcome> {
    anyhow::ensure!(capacity >= 2, "bounded REPLAN fixture needs capacity >= 2");
    anyhow::ensure!(
        waiter_count > capacity.saturating_mul(2),
        "bounded REPLAN fixture needs more than two complete windows",
    );
    let coordinator_cpu = 2_000usize;
    let first_waiter_cpu = 2_010usize;
    let common_cpu = first_waiter_cpu + waiter_count + 1;
    let fixed_cpu = common_cpu + 1;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let mut claims = Vec::with_capacity(waiter_count);
    let mut waiters = Vec::with_capacity(waiter_count);
    for index in 0..waiter_count {
        let claim = ClaimSet::new(
            std::iter::empty(),
            [first_waiter_cpu + index],
            FlockMode::Exclusive,
        );
        let watch = ClaimSet::new(
            std::iter::empty(),
            [first_waiter_cpu + index, common_cpu],
            FlockMode::Exclusive,
        );
        waiters.push(Ticket::register(claim.clone(), watch, None)?);
        claims.push(claim);
    }
    let fixed_claim = ClaimSet::new(std::iter::empty(), [fixed_cpu], FlockMode::Exclusive);
    let mut fixed = Ticket::register(fixed_claim.clone(), fixed_claim, None)?;

    let mut slices = Vec::new();
    let mut peak_outstanding = 0usize;
    let disjoint_exact_granted;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_replan_capacity_for_tests(capacity)?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, common_cpu, false)?;
        set_cpu_free_for_tests(&mut table, fixed_cpu, true)?;
        for index in 0..waiter_count {
            set_cpu_free_for_tests(&mut table, first_waiter_cpu + index, false)?;
        }
        table.stamp_resource_improvement(S_CPU_EX, common_cpu)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let selected = waiters
            .iter()
            .enumerate()
            .filter_map(|(index, ticket)| {
                table
                    .record(ticket.slot)
                    .ok()
                    .flatten()
                    .is_some_and(|record| record.state == STATE_REPLAN)
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        peak_outstanding = peak_outstanding.max(
            usize::try_from(table.replan_outstanding())
                .context("initial bounded REPLAN count does not fit usize")?,
        );
        slices.push(selected);
        disjoint_exact_granted = table
            .record(fixed.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED);
    }

    let window_count = waiter_count.div_ceil(capacity);
    while slices.len() < window_count {
        let selected = slices
            .last()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("bounded REPLAN fixture published no first window"))?;
        anyhow::ensure!(
            !selected.is_empty() && selected.len() <= capacity,
            "bounded REPLAN fixture published malformed slice {selected:?}",
        );
        for index in selected {
            let designated = claims[index].clone();
            let result = waiters[index].run_granted(
                None,
                |current, _watch, acquisition_allowed, _predecessors, _availability| {
                    anyhow::ensure!(
                        !acquisition_allowed && current == &designated,
                        "bounded REPLAN callback received an exact acquisition license",
                    );
                    Ok(GrantAttempt::<()> {
                        acquired: None,
                        preparation_claim: None,
                        preparation_contention: None,
                        next_claim: designated.clone(),
                        contention: None,
                    })
                },
            )?;
            anyhow::ensure!(
                matches!(result, GrantResult::Requeued),
                "bounded REPLAN callback did not return to WAITING",
            );
        }
        let next = {
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            table.stamp_resource_improvement(S_CPU_EX, common_cpu)?;
            table.set_pending_flag(PENDING_RESCAN);
            table.grant_compatible()?;
            peak_outstanding = peak_outstanding.max(
                usize::try_from(table.replan_outstanding())
                    .context("refilled bounded REPLAN count does not fit usize")?,
            );
            waiters
                .iter()
                .enumerate()
                .filter_map(|(index, ticket)| {
                    table
                        .record(ticket.slot)
                        .ok()
                        .flatten()
                        .is_some_and(|record| record.state == STATE_REPLAN)
                        .then_some(index)
                })
                .collect()
        };
        slices.push(next);
    }

    fixed.finish(None)?;
    for waiter in waiters.iter_mut().rev() {
        waiter.finish(None)?;
    }
    coordinator.finish(None)?;
    Ok(BoundedReplanWindowOutcome {
        capacity,
        peak_outstanding,
        slices,
        disjoint_exact_granted,
    })
}

#[cfg(test)]
pub(super) fn exercise_replan_token_wave_for_tests(
    waiter_count: usize,
) -> Result<ReplanTokenWaveOutcome> {
    if waiter_count < 4 {
        anyhow::bail!("REPLAN wave exercise needs at least four waiters");
    }
    let coordinator_cpu = 10usize;
    let common_cpu = 20usize;
    let first_waiter_cpu = 100usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator =
        Ticket::register(coordinator_claim.clone(), coordinator_claim.clone(), None)?;
    let mut waiters = Vec::with_capacity(waiter_count);
    for index in 0..waiter_count {
        let claim = ClaimSet::new(
            std::iter::empty(),
            [first_waiter_cpu + index],
            FlockMode::Exclusive,
        );
        let watch = ClaimSet::new(
            std::iter::empty(),
            [first_waiter_cpu + index, common_cpu],
            FlockMode::Exclusive,
        );
        waiters.push((Some(Ticket::register(claim.clone(), watch, None)?), claim));
    }

    let exact_grant_indices: BTreeSet<_> = (0..waiter_count).step_by(16).collect();
    let record_wake = |table: &mut Table, slot: u64| -> Result<u32> {
        let bytes = table
            .record_bytes(slot)?
            .ok_or_else(|| anyhow::anyhow!("REPLAN wave slot {slot} disappeared"))?;
        Ok(read_u32(bytes, R_WAKE))
    };
    let (
        registration_waiting,
        exact_grants,
        initial_replans,
        initial_wakes,
        initial_prefix_comparisons,
        initial_full_watch_materializations,
        initial_encoded_watch_serial_walks,
        initial_full_prefix_snapshot_publishes,
        initial_replan_indices,
    ) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_replan_capacity_for_tests(waiter_count)?;
        table.restage_coordinator_for_tests(
            &coordinator,
            waiters
                .iter()
                .map(|(ticket, _)| ticket.as_ref().expect("live REPLAN wave ticket")),
        )?;
        let registration_waiting = waiters.iter().all(|(ticket, _)| {
            ticket.as_ref().is_some_and(|ticket| {
                table
                    .record(ticket.slot)
                    .ok()
                    .flatten()
                    .is_some_and(|record| record.state == STATE_WAITING)
            })
        });
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, common_cpu, false)?;
        for index in 0..waiter_count {
            set_cpu_free_for_tests(
                &mut table,
                first_waiter_cpu + index,
                exact_grant_indices.contains(&index),
            )?;
        }
        let wakes_before = waiters
            .iter()
            .map(|(ticket, _)| {
                record_wake(
                    &mut table,
                    ticket.as_ref().expect("live REPLAN wave ticket").slot,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        table.stamp_resource_improvement(S_CPU_EX, common_cpu)?;
        table.set_pending_flag(PENDING_RESCAN);
        let prefix_comparisons_before = PREFIX_COMPARE_RECORD_READS.with(std::cell::Cell::get);
        let full_watch_materializations_before =
            FULL_WATCH_MATERIALIZATIONS.with(std::cell::Cell::get);
        let encoded_watch_serial_walks_before =
            ENCODED_WATCH_SERIAL_WALKS.with(std::cell::Cell::get);
        let full_prefix_snapshot_publishes_before =
            FULL_PREFIX_SNAPSHOT_PUBLISHES.with(std::cell::Cell::get);
        table.grant_compatible()?;
        let initial_prefix_comparisons =
            PREFIX_COMPARE_RECORD_READS.with(std::cell::Cell::get) - prefix_comparisons_before;
        let initial_full_watch_materializations = FULL_WATCH_MATERIALIZATIONS
            .with(std::cell::Cell::get)
            - full_watch_materializations_before;
        let initial_encoded_watch_serial_walks = ENCODED_WATCH_SERIAL_WALKS
            .with(std::cell::Cell::get)
            - encoded_watch_serial_walks_before;
        let initial_full_prefix_snapshot_publishes = FULL_PREFIX_SNAPSHOT_PUBLISHES
            .with(std::cell::Cell::get)
            - full_prefix_snapshot_publishes_before;
        let records = waiters
            .iter()
            .map(|(ticket, _)| {
                table
                    .record(ticket.as_ref().expect("live REPLAN wave ticket").slot)?
                    .ok_or_else(|| anyhow::anyhow!("REPLAN wave ticket disappeared after scan"))
            })
            .collect::<Result<Vec<_>>>()?;
        let exact_grants = records
            .iter()
            .filter(|record| record.state == STATE_GRANTED)
            .count();
        let replans: Vec<_> = records
            .iter()
            .enumerate()
            .filter_map(|(index, record)| (record.state == STATE_REPLAN).then_some(index))
            .collect();
        let initial_wakes = waiters
            .iter()
            .zip(wakes_before)
            .filter_map(|((ticket, _), before)| {
                let slot = ticket.as_ref().expect("live REPLAN wave ticket").slot;
                record_wake(&mut table, slot)
                    .ok()
                    .filter(|after| *after != before)
            })
            .count();
        (
            registration_waiting,
            exact_grants,
            replans.len(),
            initial_wakes,
            initial_prefix_comparisons,
            initial_full_watch_materializations,
            initial_encoded_watch_serial_walks,
            initial_full_prefix_snapshot_publishes,
            replans,
        )
    };
    let callback_index = initial_replan_indices
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("scan issued no REPLAN callbacks"))?;

    let (repeated_replans, repeated_wakes) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let wakes_before = initial_replan_indices
            .iter()
            .map(|index| {
                record_wake(
                    &mut table,
                    waiters[*index]
                        .0
                        .as_ref()
                        .expect("live REPLAN wave ticket")
                        .slot,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let replans = initial_replan_indices
            .iter()
            .filter(|index| {
                let ticket = waiters[**index]
                    .0
                    .as_ref()
                    .expect("live REPLAN wave ticket");
                table
                    .record(ticket.slot)
                    .ok()
                    .flatten()
                    .is_some_and(|record| record.state == STATE_REPLAN)
            })
            .count();
        let wakes = initial_replan_indices
            .iter()
            .zip(wakes_before)
            .filter_map(|(index, before)| {
                let slot = waiters[*index]
                    .0
                    .as_ref()
                    .expect("live REPLAN wave ticket")
                    .slot;
                record_wake(&mut table, slot)
                    .ok()
                    .filter(|after| *after != before)
            })
            .count();
        (replans, wakes)
    };

    // A fixed exact waiter remains independent of the speculative wave. It
    // can become runnable without republishing or re-waking any live REPLAN
    // callback.
    let fixed_cpu = first_waiter_cpu + waiter_count + 10;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, fixed_cpu, false)?;
    }
    let fixed_claim = ClaimSet::new(std::iter::empty(), [fixed_cpu], FlockMode::Exclusive);
    let mut fixed = Ticket::register(fixed_claim.clone(), fixed_claim, None)?;
    let (fixed_waiter_granted, fixed_waiter_woken, fixed_scan_replans, fixed_scan_replan_wakes) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        anyhow::ensure!(
            table
                .record(fixed.slot)?
                .is_some_and(|record| record.state == STATE_WAITING),
            "blocked fixed waiter did not register WAITING",
        );
        let fixed_wake_before = record_wake(&mut table, fixed.slot)?;
        let replan_wakes_before = initial_replan_indices
            .iter()
            .map(|index| {
                let slot = waiters[*index]
                    .0
                    .as_ref()
                    .expect("live REPLAN wave ticket")
                    .slot;
                Ok((slot, record_wake(&mut table, slot)?))
            })
            .collect::<Result<Vec<_>>>()?;
        set_cpu_free_for_tests(&mut table, fixed_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let fixed_waiter_granted = table
            .record(fixed.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED);
        let fixed_waiter_woken = record_wake(&mut table, fixed.slot)? != fixed_wake_before;
        let fixed_scan_replans = initial_replan_indices
            .iter()
            .filter(|index| {
                let slot = waiters[**index]
                    .0
                    .as_ref()
                    .expect("live REPLAN wave ticket")
                    .slot;
                table
                    .record(slot)
                    .ok()
                    .flatten()
                    .is_some_and(|record| record.state == STATE_REPLAN)
            })
            .count();
        let fixed_scan_replan_wakes = replan_wakes_before
            .into_iter()
            .filter_map(|(slot, before)| {
                record_wake(&mut table, slot)
                    .ok()
                    .filter(|after| *after != before)
            })
            .count();
        (
            fixed_waiter_granted,
            fixed_waiter_woken,
            fixed_scan_replans,
            fixed_scan_replan_wakes,
        )
    };

    let active_reads_before = ACTIVE_LIST_RECORD_READS.with(std::cell::Cell::get);
    let prefix_reads_before = GRANT_PREFIX_RECORD_READS.with(std::cell::Cell::get);
    let callback_claim = waiters[callback_index].1.clone();
    let callback_result = waiters[callback_index]
        .0
        .as_mut()
        .expect("selected REPLAN callback remains live")
        .run_granted(
            None,
            |designated, _watch, acquisition_allowed, _predecessors, _availability| {
                anyhow::ensure!(
                    !acquisition_allowed && designated == &callback_claim,
                    "REPLAN callback received an acquisition license or wrong designation",
                );
                Ok(GrantAttempt::<()> {
                    acquired: None,
                    preparation_claim: None,
                    preparation_contention: None,
                    next_claim: designated.clone(),
                    contention: None,
                })
            },
        )?;
    let callback_requeued = matches!(callback_result, GrantResult::Requeued);
    let callback_active_reads =
        ACTIVE_LIST_RECORD_READS.with(std::cell::Cell::get) - active_reads_before;
    let callback_prefix_reads =
        GRANT_PREFIX_RECORD_READS.with(std::cell::Cell::get) - prefix_reads_before;

    // The completed callback is older than the newly registered flexible
    // ticket. A subsequent relevant observation must publish both immediately
    // even while the original finite wave retains live stragglers. Extending
    // that wave moves only its diagnostic ticket horizon; every callback still
    // shares the original bounded lease deadline.
    let late_cpu = fixed_cpu + 1;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, late_cpu, false)?;
    }
    let late_claim = ClaimSet::new(std::iter::empty(), [late_cpu], FlockMode::Exclusive);
    let late_watch = ClaimSet::new(
        std::iter::empty(),
        [late_cpu, common_cpu],
        FlockMode::Exclusive,
    );
    let mut late = Ticket::register(late_claim.clone(), late_watch, None)?;
    let (
        mixed_age_old_replanned,
        mixed_age_old_woken,
        mixed_age_late_replanned,
        mixed_age_late_woken,
        mixed_age_stragglers_remaining,
        mixed_age_outstanding,
        mixed_age_horizon_extended,
        mixed_age_deadline_preserved,
    ) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let old = waiters[callback_index]
            .0
            .as_ref()
            .expect("old pre-horizon callback remains live");
        anyhow::ensure!(old.ticket < late.ticket, "late ticket was not post-horizon");
        let old_wake_before = record_wake(&mut table, old.slot)?;
        let late_wake_before = record_wake(&mut table, late.slot)?;
        let horizon_before = read_u64(&table.header, H_REPLAN_HORIZON);
        let deadline_before = table.replan_wave_deadline_ns();
        table.stamp_resource_improvement(S_CPU_EX, common_cpu)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let mixed_age_old_replanned = table
            .record(old.slot)?
            .is_some_and(|record| record.state == STATE_REPLAN);
        let mixed_age_late_replanned = table
            .record(late.slot)?
            .is_some_and(|record| record.state == STATE_REPLAN);
        let mixed_age_stragglers_remaining = initial_replan_indices
            .iter()
            .copied()
            .filter(|index| *index != callback_index)
            .filter(|index| {
                waiters[*index]
                    .0
                    .as_ref()
                    .and_then(|ticket| table.record(ticket.slot).ok().flatten())
                    .is_some_and(|record| record.state == STATE_REPLAN)
            })
            .count();
        (
            mixed_age_old_replanned,
            record_wake(&mut table, old.slot)? != old_wake_before,
            mixed_age_late_replanned,
            record_wake(&mut table, late.slot)? != late_wake_before,
            mixed_age_stragglers_remaining,
            usize::try_from(table.replan_outstanding())
                .context("mixed-age REPLAN count does not fit usize")?,
            horizon_before < late.ticket
                && read_u64(&table.header, H_REPLAN_HORIZON) >= late.ticket,
            deadline_before != 0 && table.replan_wave_deadline_ns() == deadline_before,
        )
    };

    let (mixed_age_repeated_replans, mixed_age_repeated_wakes, mixed_age_repeated_outstanding) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let old = waiters[callback_index]
            .0
            .as_ref()
            .expect("old mixed-age callback remains live");
        let old_wake_before = record_wake(&mut table, old.slot)?;
        let late_wake_before = record_wake(&mut table, late.slot)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let repeated_replans = usize::from(
            table
                .record(old.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN),
        ) + usize::from(
            table
                .record(late.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN),
        );
        let repeated_wakes = usize::from(record_wake(&mut table, old.slot)? != old_wake_before)
            + usize::from(record_wake(&mut table, late.slot)? != late_wake_before);
        (
            repeated_replans,
            repeated_wakes,
            usize::try_from(table.replan_outstanding())
                .context("repeated mixed-age REPLAN count does not fit usize")?,
        )
    };

    // Complete the original stragglers, followed by the two callbacks which
    // joined incrementally. Entry snapshots consume every observation already
    // published before they returned; draining the wave therefore clears its
    // clock, and the coalesced post-completion scan has no reason to republish
    // either callback without a fresh input change.
    for index in initial_replan_indices
        .iter()
        .copied()
        .filter(|index| *index != callback_index)
    {
        let designated = waiters[index].1.clone();
        let result = waiters[index]
            .0
            .as_mut()
            .expect("remaining first-wave callback remains live")
            .run_granted(
                None,
                |current, _watch, acquisition_allowed, _predecessors, _availability| {
                    anyhow::ensure!(
                        !acquisition_allowed && current == &designated,
                        "remaining first-wave callback received the wrong publication",
                    );
                    Ok(GrantAttempt::<()> {
                        acquired: None,
                        preparation_claim: None,
                        preparation_contention: None,
                        next_claim: current.clone(),
                        contention: None,
                    })
                },
            )?;
        anyhow::ensure!(
            matches!(result, GrantResult::Requeued),
            "remaining first-wave callback did not return to WAITING",
        );
    }
    let old_result = waiters[callback_index]
        .0
        .as_mut()
        .expect("incrementally republished old callback remains live")
        .run_granted(
            None,
            |current, _watch, acquisition_allowed, _predecessors, _availability| {
                anyhow::ensure!(
                    !acquisition_allowed && current == &callback_claim,
                    "incrementally republished old callback received the wrong publication",
                );
                Ok(GrantAttempt::<()> {
                    acquired: None,
                    preparation_claim: None,
                    preparation_contention: None,
                    next_claim: current.clone(),
                    contention: None,
                })
            },
        )?;
    anyhow::ensure!(
        matches!(old_result, GrantResult::Requeued),
        "incrementally republished old callback did not return to WAITING",
    );
    let late_result = late.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &late_claim,
                "incrementally published late callback received the wrong publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: current.clone(),
                contention: None,
            })
        },
    )?;
    anyhow::ensure!(
        matches!(late_result, GrantResult::Requeued),
        "incrementally published late callback did not return to WAITING",
    );

    let (
        mixed_age_callbacks_drained,
        mixed_age_clock_cleared,
        mixed_age_post_drain_replans,
        mixed_age_post_drain_wakes,
    ) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let old = waiters[callback_index]
            .0
            .as_ref()
            .expect("drained old mixed-age callback remains live");
        let old_wake_before = record_wake(&mut table, old.slot)?;
        let late_wake_before = record_wake(&mut table, late.slot)?;
        table.grant_compatible()?;
        let post_drain_replans = usize::from(
            table
                .record(old.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN),
        ) + usize::from(
            table
                .record(late.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN),
        );
        let post_drain_wakes = usize::from(record_wake(&mut table, old.slot)? != old_wake_before)
            + usize::from(record_wake(&mut table, late.slot)? != late_wake_before);
        (
            table.replan_outstanding() == 0,
            table.replan_wave_started_ns() == 0 && table.replan_wave_deadline_ns() == 0,
            post_drain_replans,
            post_drain_wakes,
        )
    };

    late.finish(None)?;
    fixed.finish(None)?;
    for (ticket, _) in waiters.iter_mut().rev() {
        if let Some(ticket) = ticket.as_mut() {
            ticket.finish(None)?;
        }
    }
    coordinator.finish(None)?;

    let memo_identical_waiters = 64usize;
    let memo_identical = exercise_encoded_watch_serial_memo_case_for_tests(
        memo_identical_waiters,
        false,
        memo_identical_waiters,
    )?;
    let memo_identical_replans = memo_identical.initial_replans;
    let memo_identical_serial_walks = memo_identical.initial_serial_walks;
    let memo_identical_layout_words = memo_identical.layout_words;
    let memo_identical_exact_word_reads = memo_identical.initial_exact_word_reads;
    let memo_identical_claim_heap_spills = memo_identical.initial_claim_heap_spills;
    let memo_mixed_waiters = 8usize;
    let memo_mixed = exercise_encoded_watch_serial_memo_case_for_tests(
        memo_mixed_waiters,
        true,
        memo_mixed_waiters,
    )?;
    let memo_mixed_replans = memo_mixed.initial_replans;
    let memo_mixed_serial_walks = memo_mixed.initial_serial_walks;
    let memo_guard_waiters = 8usize;
    let memo_guard_initial_capacity = 2usize;
    let memo_guard = exercise_encoded_watch_serial_memo_case_for_tests(
        memo_guard_waiters,
        false,
        memo_guard_initial_capacity,
    )?;

    Ok(ReplanTokenWaveOutcome {
        registration_waiting,
        exact_grants,
        initial_replans,
        initial_wakes,
        initial_prefix_comparisons,
        initial_full_watch_materializations,
        initial_encoded_watch_serial_walks,
        memo_identical_waiters,
        memo_identical_replans,
        memo_identical_serial_walks,
        memo_identical_layout_words,
        memo_identical_exact_word_reads,
        memo_identical_claim_heap_spills,
        memo_mixed_waiters,
        memo_mixed_replans,
        memo_mixed_serial_walks,
        memo_guard_waiters,
        memo_guard_initial_capacity,
        memo_guard_initial_replans: memo_guard.initial_replans,
        memo_guard_saturated_replans: memo_guard.saturated_replans,
        memo_guard_saturated_serial_walks: memo_guard.saturated_serial_walks,
        memo_guard_closed_replans: memo_guard.closed_replans,
        memo_guard_closed_serial_walks: memo_guard.closed_serial_walks,
        memo_guard_opened_replans: memo_guard.opened_replans,
        memo_guard_opened_serial_walks: memo_guard.opened_serial_walks,
        initial_full_prefix_snapshot_publishes,
        repeated_replans,
        repeated_wakes,
        fixed_waiter_granted,
        fixed_waiter_woken,
        fixed_scan_replans,
        fixed_scan_replan_wakes,
        callback_requeued,
        callback_prefix_reads,
        callback_active_reads,
        mixed_age_old_replanned,
        mixed_age_old_woken,
        mixed_age_late_replanned,
        mixed_age_late_woken,
        mixed_age_repeated_replans,
        mixed_age_repeated_wakes,
        mixed_age_stragglers_remaining,
        mixed_age_outstanding,
        mixed_age_horizon_extended,
        mixed_age_deadline_preserved,
        mixed_age_repeated_outstanding,
        mixed_age_callbacks_drained,
        mixed_age_clock_cleared,
        mixed_age_post_drain_replans,
        mixed_age_post_drain_wakes,
    })
}

/// Complete a large finite REPLAN wave. Every callback publishes the same
/// coalesced rescan edge; none performs the authoritative O(N) scan or a
/// global generation wake while the coordinator remains live.
#[cfg(test)]
pub(super) fn exercise_changed_replan_wave_completions_for_tests(
    callback_count: usize,
) -> Result<ReplanChangedWaveOutcome> {
    if callback_count < 2 {
        anyhow::bail!("changed REPLAN wave exercise needs at least two callbacks");
    }
    let coordinator_cpu = 900usize;
    let designated_base = 1_000usize;
    let replacement_base = designated_base + callback_count;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let mut callbacks = Vec::with_capacity(callback_count);
    for index in 0..callback_count {
        let designated = ClaimSet::new(
            std::iter::empty(),
            [designated_base + index],
            FlockMode::Exclusive,
        );
        let replacement = ClaimSet::new(
            std::iter::empty(),
            [replacement_base + index],
            FlockMode::Exclusive,
        );
        let watch = ClaimSet::new(
            std::iter::empty(),
            [designated_base + index, replacement_base + index],
            FlockMode::Exclusive,
        );
        callbacks.push((
            Ticket::register(designated.clone(), watch, None)?,
            designated,
            replacement,
        ));
    }

    let (scans_before, generation_wake_before, notify_before) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_replan_capacity_for_tests(callback_count)?;
        table.restage_coordinator_for_tests(
            &coordinator,
            callbacks.iter().map(|(ticket, _, _)| ticket),
        )?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        for index in 0..callback_count {
            set_cpu_free_for_tests(&mut table, designated_base + index, false)?;
            set_cpu_free_for_tests(&mut table, replacement_base + index, false)?;
        }
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table.replan_outstanding() == callback_count as u64
                && callbacks.iter().all(|(ticket, _, _)| {
                    table
                        .record(ticket.slot)
                        .ok()
                        .flatten()
                        .is_some_and(|record| record.state == STATE_REPLAN)
                }),
            "changed REPLAN wave did not publish every callback",
        );
        (
            read_u64(&table.header, H_GRANT_SCANS),
            table.generation_wake(),
            NOTIFY_CALLS.with(std::cell::Cell::get),
        )
    };

    for (ticket, designated, replacement) in callbacks.iter_mut().take(callback_count - 1) {
        let designated = designated.clone();
        let replacement = replacement.clone();
        let result = ticket.run_granted(
            None,
            |current, _watch, acquisition_allowed, _predecessors, _availability| {
                anyhow::ensure!(
                    !acquisition_allowed && current == &designated,
                    "changed REPLAN callback received the wrong publication",
                );
                Ok(GrantAttempt::<()> {
                    acquired: None,
                    preparation_claim: None,
                    preparation_contention: None,
                    next_claim: replacement,
                    contention: None,
                })
            },
        )?;
        anyhow::ensure!(
            matches!(result, GrantResult::Requeued),
            "changed REPLAN callback did not return to WAITING",
        );
    }

    let (
        intermediate_notify_delta,
        intermediate_scan_delta,
        intermediate_generation_wake_delta,
        intermediate_rescan_coalesced,
    ) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        (
            NOTIFY_CALLS
                .with(std::cell::Cell::get)
                .wrapping_sub(notify_before),
            read_u64(&table.header, H_GRANT_SCANS).wrapping_sub(scans_before),
            table.generation_wake().wrapping_sub(generation_wake_before),
            table.replan_outstanding() == 1
                && table.pending_flags() & PENDING_REPLAN_RESCAN != 0
                && table.pending_flags() & PENDING_RESCAN == 0
                && table.deferred_rescan_deadline_ns() != 0,
        )
    };

    let last = callback_count - 1;
    let designated = callbacks[last].1.clone();
    let replacement = callbacks[last].2.clone();
    let result = callbacks[last].0.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &designated,
                "final changed REPLAN callback received the wrong publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: replacement,
                contention: None,
            })
        },
    )?;
    anyhow::ensure!(
        matches!(result, GrantResult::Requeued),
        "final changed REPLAN callback did not return to WAITING",
    );

    let (
        final_scan_delta_before_authoritative,
        final_notify_delta,
        final_generation_wake_delta,
        final_rescan_edge,
        authoritative_scan_delta,
        authoritative_flags_clear,
        replacements_preserved,
    ) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let final_scan_delta_before_authoritative =
            read_u64(&table.header, H_GRANT_SCANS).wrapping_sub(scans_before);
        let final_notify_delta = NOTIFY_CALLS
            .with(std::cell::Cell::get)
            .wrapping_sub(notify_before);
        let final_generation_wake_delta =
            table.generation_wake().wrapping_sub(generation_wake_before);
        let final_rescan_edge = table.replan_outstanding() == 0
            && table.pending_flags() & PENDING_RESCAN != 0
            && table.pending_flags() & PENDING_REPLAN_RESCAN == 0
            && table.deferred_rescan_deadline_ns() == 0;
        table.grant_compatible()?;
        let authoritative_scan_delta =
            read_u64(&table.header, H_GRANT_SCANS).wrapping_sub(scans_before);
        let authoritative_flags_clear =
            table.pending_flags() & (PENDING_RESCAN | PENDING_REPLAN_RESCAN) == 0
                && table.deferred_rescan_deadline_ns() == 0;
        let replacements_preserved = callbacks.iter().all(|(ticket, _, replacement)| {
            table
                .record(ticket.slot)
                .ok()
                .flatten()
                .is_some_and(|record| record.state == STATE_WAITING && record.claim == *replacement)
        });
        (
            final_scan_delta_before_authoritative,
            final_notify_delta,
            final_generation_wake_delta,
            final_rescan_edge,
            authoritative_scan_delta,
            authoritative_flags_clear,
            replacements_preserved,
        )
    };

    for (ticket, _, _) in callbacks.iter_mut().rev() {
        ticket.finish(None)?;
    }
    coordinator.finish(None)?;
    Ok(ReplanChangedWaveOutcome {
        callbacks: callback_count,
        intermediate_notify_delta,
        intermediate_scan_delta,
        intermediate_generation_wake_delta,
        intermediate_rescan_coalesced,
        final_scan_delta_before_authoritative,
        final_notify_delta,
        final_generation_wake_delta,
        final_rescan_edge,
        authoritative_scan_delta,
        authoritative_flags_clear,
        replacements_preserved,
    })
}

/// Register a coordinator plus one flexible waiter (designated `designated_cpu`,
/// alternative `replacement_cpu`, host-wide watch over both) and drive one
/// authoritative scan that leaves the waiter in REPLAN because its designation
/// is unavailable. Shared setup for the same-wake re-designation exercises.
#[cfg(test)]
fn stage_same_wake_replan_waiter(
    coordinator_cpu: usize,
    designated_cpu: usize,
    replacement_cpu: usize,
    replacement_free: bool,
) -> Result<(Ticket, Ticket, ClaimSet, ClaimSet, u64)> {
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let designated = ClaimSet::new(std::iter::empty(), [designated_cpu], FlockMode::Exclusive);
    let replacement = ClaimSet::new(std::iter::empty(), [replacement_cpu], FlockMode::Exclusive);
    let watch = ClaimSet::new(
        std::iter::empty(),
        [designated_cpu, replacement_cpu],
        FlockMode::Exclusive,
    );
    let waiter = Ticket::register(designated.clone(), watch, None)?;
    let scans_before = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_replan_capacity_for_tests(1)?;
        table.restage_coordinator_for_tests(&coordinator, std::iter::once(&waiter))?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, designated_cpu, false)?;
        set_cpu_free_for_tests(&mut table, replacement_cpu, replacement_free)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table.replan_outstanding() == 1
                && table
                    .record(waiter.slot)?
                    .is_some_and(|record| record.state == STATE_REPLAN),
            "same-wake exercise did not REPLAN its blocked flexible waiter",
        );
        read_u64(&table.header, H_GRANT_SCANS)
    };
    Ok((coordinator, waiter, designated, replacement, scans_before))
}

/// Outcome of the same-wake re-designation grant exercise.
#[cfg(test)]
pub(crate) struct SameWakeRedesignationOutcome {
    pub granted: bool,
    pub held_claim_is_replacement: bool,
    pub replan_drained: bool,
    pub no_second_grant_scan: bool,
}

/// A REPLAN wake whose designation is blocked but whose watch alternative is
/// free acquires that alternative in the same wake: the record commits HELD with
/// the re-designated claim, its ring slot drains, and NO additional
/// authoritative scan runs between the REPLAN publication and HELD.
#[cfg(test)]
pub(super) fn exercise_same_wake_redesignation_grant_for_tests()
-> Result<SameWakeRedesignationOutcome> {
    let (mut coordinator, mut waiter, designated, replacement, scans_before) =
        stage_same_wake_replan_waiter(900, 1_000, 1_001, true)?;
    let designated_for_wake = designated.clone();
    let replacement_for_wake = replacement.clone();
    let result = waiter.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &designated_for_wake,
                "same-wake callback received the wrong publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: Some(()),
                preparation_claim: None,
                preparation_contention: None,
                next_claim: replacement_for_wake.clone(),
                contention: None,
            })
        },
    )?;
    let held = match result {
        GrantResult::Acquired((), held) => held,
        _ => anyhow::bail!("same-wake re-designation did not commit acquisition"),
    };
    let (held_claim_is_replacement, replan_drained, no_second_grant_scan) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(waiter.slot)?
            .ok_or_else(|| anyhow::anyhow!("same-wake held record disappeared"))?;
        (
            record.state == STATE_HELD && record.claim == replacement,
            table.replan_outstanding() == 0,
            read_u64(&table.header, H_GRANT_SCANS) == scans_before,
        )
    };
    drop(held);
    coordinator.finish(None)?;
    waiter.finish(None)?;
    Ok(SameWakeRedesignationOutcome {
        granted: true,
        held_claim_is_replacement,
        replan_drained,
        no_second_grant_scan,
    })
}

/// When no watch alternative is free the REPLAN callback cannot complete in the
/// wake and falls back to publishing a WAITING replacement (`Requeued`), leaving
/// its ring slot drained for a later authoritative scan — the pre-existing
/// re-plan contract, preserved.
#[cfg(test)]
pub(super) fn exercise_same_wake_redesignation_fallback_for_tests() -> Result<(bool, bool, bool)> {
    let (mut coordinator, mut waiter, designated, replacement, _scans_before) =
        stage_same_wake_replan_waiter(900, 1_000, 1_001, false)?;
    let designated_for_wake = designated.clone();
    let replacement_for_wake = replacement.clone();
    let result = waiter.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &designated_for_wake,
                "fallback callback received the wrong publication",
            );
            // Nothing free: reserve the alternative, exactly as the acquirer's
            // fallback does after a failed re-designation probe.
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: replacement_for_wake.clone(),
                contention: None,
            })
        },
    )?;
    let requeued = matches!(result, GrantResult::Requeued);
    let (waiting_replacement, replan_drained) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(waiter.slot)?
            .ok_or_else(|| anyhow::anyhow!("fallback waiter record disappeared"))?;
        (
            record.state == STATE_WAITING && record.claim == replacement,
            table.replan_outstanding() == 0,
        )
    };
    coordinator.finish(None)?;
    waiter.finish(None)?;
    Ok((requeued, waiting_replacement, replan_drained))
}

/// A same-wake re-designation that physically acquired an alternative against a
/// predecessor snapshot which an older ticket has since invalidated must NOT
/// overtake that older ticket. The scan's exact suffix watermark
/// (`min_changed_ticket`) below this ticket forces the acquired alternative to
/// be released and the ticket requeued, exactly as the GRANTED dirty path does.
#[cfg(test)]
pub(super) fn exercise_same_wake_redesignation_older_fence_for_tests() -> Result<(bool, bool, bool)>
{
    let (mut coordinator, mut waiter, designated, replacement, _scans_before) =
        stage_same_wake_replan_waiter(900, 1_000, 1_001, true)?;
    // Simulate an older ticket changing its exact claim after this callback's
    // predecessor snapshot: advance the suffix watermark below this ticket
    // without touching the claim epoch, so the callback still runs and this
    // ticket's own dirty fence — not the epoch staleness guard — releases it.
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.mark_suffix_dirty(coordinator.ticket);
        anyhow::ensure!(
            table.min_changed_ticket() < waiter.ticket,
            "older-ticket fence exercise did not stage a preceding change",
        );
    }
    let designated_for_wake = designated.clone();
    let replacement_for_wake = replacement.clone();
    let result = waiter.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &designated_for_wake,
                "older-fence callback received the wrong publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: Some(()),
                preparation_claim: None,
                preparation_contention: None,
                next_claim: replacement_for_wake.clone(),
                contention: None,
            })
        },
    )?;
    // Released, not granted: the fence returns LostGrant and the record stays on
    // its published designation rather than committing the alternative.
    let released = matches!(result, GrantResult::LostGrant);
    let (not_committed, replan_drained) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(waiter.slot)?
            .ok_or_else(|| anyhow::anyhow!("older-fence waiter record disappeared"))?;
        (
            record.state != STATE_HELD && record.claim != replacement,
            table.replan_outstanding() == 0,
        )
    };
    coordinator.finish(None)?;
    waiter.finish(None)?;
    Ok((released, not_committed, replan_drained))
}

/// Outcome of the own-designation REPLAN acquire exercise (items 2/3).
#[cfg(test)]
pub(crate) struct SameWakeOwnDesignationOutcome {
    /// Unfenced: committed HELD on the cell's OWN designation.
    pub granted: bool,
    pub held_on_designation: bool,
    /// Fenced: the suffix watermark released + requeued instead of committing.
    pub released_by_fence: bool,
    pub not_committed: bool,
    pub replan_drained: bool,
}

/// A no-license REPLAN wake whose OWN designation frees first: the acquirer's
/// candidate loop legitimately selects its own placement, so the callback
/// returns an acquired payload with `next_claim == designated`. This must NOT
/// trip the "replan-only wake returned an acquired payload without
/// re-designation" assertion (it was too narrow); the record commits HELD on the
/// designation. The `fenced` variant stages an older ticket's suffix change so
/// the same unlicensed acquire — own-designation and all — is released and
/// requeued by the suffix watermark rather than overtaking the older waiter.
#[cfg(test)]
pub(super) fn exercise_same_wake_own_designation_grant_for_tests(
    fenced: bool,
) -> Result<SameWakeOwnDesignationOutcome> {
    // `replacement_free = false`: the only capacity this waiter can take is its
    // own designation, which frees below after the REPLAN publication.
    let (mut coordinator, mut waiter, designated, _replacement, _scans_before) =
        stage_same_wake_replan_waiter(900, 1_000, 1_001, false)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, 1_000, true)?;
        if fenced {
            // An older ticket changed its exact claim after this callback's
            // predecessor snapshot: advance the suffix watermark below this
            // ticket without touching the claim epoch.
            table.mark_suffix_dirty(coordinator.ticket);
            anyhow::ensure!(
                table.min_changed_ticket() < waiter.ticket,
                "own-designation fence exercise did not stage a preceding change",
            );
        }
    }
    let designated_for_wake = designated.clone();
    let result = waiter.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &designated_for_wake,
                "own-designation callback received the wrong publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: Some(()),
                preparation_claim: None,
                preparation_contention: None,
                // The candidate loop selected the cell's OWN designation.
                next_claim: designated_for_wake.clone(),
                contention: None,
            })
        },
    )?;
    let outcome = if fenced {
        let released_by_fence = matches!(result, GrantResult::LostGrant);
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(waiter.slot)?
            .ok_or_else(|| anyhow::anyhow!("own-designation fenced record disappeared"))?;
        SameWakeOwnDesignationOutcome {
            granted: false,
            held_on_designation: false,
            released_by_fence,
            not_committed: record.state != STATE_HELD,
            replan_drained: table.replan_outstanding() == 0,
        }
    } else {
        let held = match result {
            GrantResult::Acquired((), held) => held,
            _ => anyhow::bail!("own-designation acquire did not commit HELD"),
        };
        let (held_on_designation, replan_drained) = {
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            let record = table
                .record(waiter.slot)?
                .ok_or_else(|| anyhow::anyhow!("own-designation held record disappeared"))?;
            (
                record.state == STATE_HELD && record.claim == designated,
                table.replan_outstanding() == 0,
            )
        };
        drop(held);
        SameWakeOwnDesignationOutcome {
            granted: true,
            held_on_designation,
            released_by_fence: false,
            not_committed: false,
            replan_drained,
        }
    };
    coordinator.finish(None)?;
    waiter.finish(None)?;
    Ok(outcome)
}

/// A REPLAN wave that expires while a same-wake re-designation callback holds a
/// freshly-acquired alternative must RELEASE that physical alternative, not
/// silently commit it: the expired-replan path marks the acquired footprint
/// unknown and returns the ticket to WAITING.
#[cfg(test)]
pub(super) fn exercise_same_wake_redesignation_expired_release_for_tests()
-> Result<(bool, bool, bool)> {
    let (mut coordinator, mut waiter, designated, replacement, _scans_before) =
        stage_same_wake_replan_waiter(900, 1_000, 1_001, true)?;
    let designated_for_wake = designated.clone();
    let replacement_for_wake = replacement.clone();
    let result = waiter.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &designated_for_wake,
                "expired-release callback received the wrong publication",
            );
            // Expire the live wave mid-acquire: the physical alternative is held
            // (below), but the registry re-lock will quarantine this callback.
            {
                let _lock = lock_registry_existing(FlockMode::Exclusive)?;
                let mut table = Table::open_existing()?;
                // Drive the live wave's clock fully into the past (valid and
                // due) so the registry re-lock quarantines this callback.
                write_u64(&mut table.header, H_REPLAN_WAVE_STARTED_NS, 1);
                write_u64(&mut table.header, H_REPLAN_WAVE_DEADLINE_NS, 1);
            }
            Ok(GrantAttempt::<()> {
                acquired: Some(()),
                preparation_claim: None,
                preparation_contention: None,
                next_claim: replacement_for_wake.clone(),
                contention: None,
            })
        },
    )?;
    let lost_grant = matches!(result, GrantResult::LostGrant);
    let (released_to_waiting, replan_drained) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(waiter.slot)?
            .ok_or_else(|| anyhow::anyhow!("expired-release waiter record disappeared"))?;
        (
            record.state == STATE_WAITING && record.claim != replacement,
            table.replan_outstanding() == 0,
        )
    };
    coordinator.finish(None)?;
    waiter.finish(None)?;
    Ok((lost_grant, released_to_waiting, replan_drained))
}

/// Prove that a physical GRANTED callback batch publishes one short absolute
/// rescan deadline instead of handing registry EX back to the coordinator
/// between individual negative completions.
#[cfg(test)]
pub(super) fn exercise_grant_completion_batch_for_tests() -> Result<GrantCompletionBatchOutcome> {
    // Deflake: the fixture arms its speculative deadline one deferred-rescan
    // interval ahead and asserts the first completion shortens it. Inject a
    // five-minute interval so test-process descheduling cannot cross the
    // window (see `DEFERRED_RESCAN_INTERVAL_OVERRIDE_NS`).
    struct IntervalGuard;
    impl Drop for IntervalGuard {
        fn drop(&mut self) {
            DEFERRED_RESCAN_INTERVAL_OVERRIDE_NS.with(|cell| cell.set(0));
        }
    }
    DEFERRED_RESCAN_INTERVAL_OVERRIDE_NS.with(|cell| cell.set(300_000_000_000));
    let _interval_guard = IntervalGuard;
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [1_750usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        for cpu in [1_750usize, 1_751, 1_752] {
            set_cpu_free_for_tests(&mut table, cpu, true)?;
        }
    }
    let first_claim = ClaimSet::new(std::iter::empty(), [1_751usize], FlockMode::Exclusive);
    let second_claim = ClaimSet::new(std::iter::empty(), [1_752usize], FlockMode::Exclusive);
    let mut first = Ticket::register(first_claim.clone(), first_claim.clone(), None)?;
    let mut second = Ticket::register(second_claim.clone(), second_claim.clone(), None)?;
    let notify_before = NOTIFY_CALLS.with(std::cell::Cell::get);
    let (scans_before, speculative_deadline) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        anyhow::ensure!(
            table.replan_outstanding() == 0
                && table
                    .record(first.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED)
                && table
                    .record(second.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED),
            "grant-completion fixture did not publish both exact callbacks",
        );
        let speculative_deadline = monotonic_now_ns()?
            .max(1)
            .saturating_add(deferred_rescan_interval_ns());
        table.begin_transaction()?;
        table.set_pending_flag(PENDING_REPLAN_RESCAN);
        write_u64(
            &mut table.header,
            H_DEFERRED_RESCAN_DEADLINE_NS,
            speculative_deadline,
        );
        table.finish_transaction()?;
        (read_u64(&table.header, H_GRANT_SCANS), speculative_deadline)
    };

    let mut first_deadline = 0;
    let mut first_completion_notified = false;
    let mut later_deferred_deadline_shortened = false;
    for (index, (ticket, expected)) in [(&mut first, &first_claim), (&mut second, &second_claim)]
        .into_iter()
        .enumerate()
    {
        let result = ticket.run_granted(
            None,
            |designated, _watch, acquisition_allowed, _predecessors, _availability| {
                anyhow::ensure!(
                    acquisition_allowed && designated == expected,
                    "grant-completion fixture received the wrong callback publication",
                );
                Ok(GrantAttempt::<()> {
                    acquired: None,
                    preparation_claim: None,
                    preparation_contention: None,
                    next_claim: designated.clone(),
                    contention: None,
                })
            },
        )?;
        anyhow::ensure!(
            matches!(result, GrantResult::Requeued),
            "negative exact callback did not return to WAITING",
        );
        if index == 0 {
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let table = Table::open_existing()?;
            first_deadline = table.deferred_rescan_deadline_ns();
            first_completion_notified = first_deadline != 0
                && NOTIFY_CALLS
                    .with(std::cell::Cell::get)
                    .wrapping_sub(notify_before)
                    == 1;
            later_deferred_deadline_shortened = first_deadline < speculative_deadline;
        }
    }

    let (
        second_completion_coalesced,
        deadline_was_not_renewed,
        no_scan_before_deadline,
        one_scan_at_deadline,
    ) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let deadline = table.deferred_rescan_deadline_ns();
        let notify_delta = NOTIFY_CALLS
            .with(std::cell::Cell::get)
            .wrapping_sub(notify_before);
        let pending = table.pending_flags() & PENDING_REPLAN_RESCAN != 0
            && table.pending_flags() & PENDING_RESCAN == 0
            && deadline != 0;
        let before = !table.prepare_grant_scan_at(deadline.saturating_sub(1))?
            && read_u64(&table.header, H_GRANT_SCANS) == scans_before;
        let due = table.prepare_grant_scan_at(deadline)?;
        if due {
            table.grant_compatible_at(deadline, None)?;
        }
        let scan_delta = read_u64(&table.header, H_GRANT_SCANS).wrapping_sub(scans_before);
        (
            pending && notify_delta == 1,
            pending && deadline == first_deadline,
            before,
            due && scan_delta == 1
                && table.pending_flags() & (PENDING_RESCAN | PENDING_REPLAN_RESCAN) == 0,
        )
    };

    second.finish(None)?;
    first.finish(None)?;
    coordinator.finish(None)?;
    Ok(GrantCompletionBatchOutcome {
        first_completion_notified,
        later_deferred_deadline_shortened,
        second_completion_coalesced,
        deadline_was_not_renewed,
        no_scan_before_deadline,
        one_scan_at_deadline,
    })
}

/// Change A: prove a herd of HELD releases coalesces into one authoritative
/// grant scan per quantum instead of one O(N) scan per release, that the single
/// scan still grants everything the releases freed, and that a release after the
/// scan re-arms the next one (no lost edge). No sleeping: the armed absolute
/// deadline is read back and probed on either side.
#[cfg(test)]
pub(super) fn exercise_release_coalesce_for_tests() -> Result<ReleaseCoalesceOutcome> {
    const K: usize = 4;
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [1_770usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let cpus: [usize; K] = std::array::from_fn(|index| 1_771 + index);

    // A HELD holder occupies each contended CPU; a waiter behind it registers
    // WAITING because the aggregate is already busy there.
    let mut holders = Vec::new();
    let mut waiters = Vec::new();
    for &cpu in &cpus {
        let claim = ClaimSet::new(std::iter::empty(), [cpu], FlockMode::Exclusive);
        holders.push(publish_acquired(&claim)?);
        waiters.push(Ticket::register(claim.clone(), claim, None)?);
    }

    // Consume the registration/init edge so the coalescing test starts from a
    // clean scan state (fresh tables publish PENDING_RESCAN). The holders are
    // still HELD, so no waiter is granted here.
    let (scans_before, coalesced_before) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.grant_compatible()?;
        for (holder, waiter) in holders.iter().zip(&waiters) {
            anyhow::ensure!(
                table
                    .record(holder.slot)?
                    .is_some_and(|record| record.state == STATE_HELD)
                    && table
                        .record(waiter.slot)?
                        .is_some_and(|record| record.state == STATE_WAITING),
                "release-coalesce fixture did not publish HELD holder + WAITING waiter",
            );
        }
        anyhow::ensure!(
            table.pending_flags() & (PENDING_RESCAN | PENDING_REPLAN_RESCAN) == 0,
            "release-coalesce fixture did not reach a clean post-scan edge state",
        );
        (
            read_u64(&table.header, H_GRANT_SCANS),
            COORDINATOR_SCANS_COALESCED.load(Ordering::Relaxed),
        )
    };

    // Release every holder within one quantum. The HeldClaim drop path takes the
    // registry lock itself, so drop without holding it.
    for holder in holders.drain(..) {
        drop(holder);
    }

    let (deadline, releases_coalesced_into_one_edge, no_scan_before_deadline) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let deadline = table.deferred_rescan_deadline_ns();
        let one_deferred_edge = table.pending_flags() & PENDING_RESCAN == 0
            && table.pending_flags() & PENDING_REPLAN_RESCAN != 0
            && deadline != 0
            && read_u64(&table.header, H_GRANT_SCANS) == scans_before
            && COORDINATOR_SCANS_COALESCED
                .load(Ordering::Relaxed)
                .wrapping_sub(coalesced_before)
                == K as u64 - 1;
        // The deferred edge is not due one nanosecond early: no scan yet.
        let quiet_before_deadline = !table.prepare_grant_scan_at(deadline.saturating_sub(1))?
            && read_u64(&table.header, H_GRANT_SCANS) == scans_before;
        (deadline, one_deferred_edge, quiet_before_deadline)
    };

    let one_scan_at_deadline_grants_all = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        // Model the coordinator's release observation making the freed CPUs
        // available before its due scan grants the waiters.
        for &cpu in &cpus {
            set_cpu_free_for_tests(&mut table, cpu, true)?;
        }
        let promoted = table.prepare_grant_scan_at(deadline)?;
        table.grant_compatible_at(deadline, None)?;
        let one_scan = read_u64(&table.header, H_GRANT_SCANS) == scans_before + 1;
        let all_granted = waiters.iter().try_fold(true, |acc, waiter| {
            Ok::<_, anyhow::Error>(
                acc && table
                    .record(waiter.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED),
            )
        })?;
        let edge_cleared = table.pending_flags() & (PENDING_RESCAN | PENDING_REPLAN_RESCAN) == 0
            && table.deferred_rescan_deadline_ns() == 0;
        promoted && one_scan && all_granted && edge_cleared
    };

    // Handshake: an edge arriving after the scan cleared the deferred edge must
    // schedule the next scan rather than be lost. The scan above left both flags
    // clear; one more coalesced release edge re-arms the deferred deadline.
    let post_scan_release_reschedules = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let started_clear = table.pending_flags() & (PENDING_RESCAN | PENDING_REPLAN_RESCAN) == 0;
        table.begin_transaction()?;
        table.schedule_coalesced_grant_rescan_in_transaction()?;
        table.finish_transaction()?;
        started_clear
            && table.pending_flags() & PENDING_REPLAN_RESCAN != 0
            && table.deferred_rescan_deadline_ns() != 0
    };

    for waiter in waiters.iter_mut().rev() {
        waiter.finish(None)?;
    }
    coordinator.finish(None)?;
    Ok(ReleaseCoalesceOutcome {
        releases_coalesced_into_one_edge,
        no_scan_before_deadline,
        one_scan_at_deadline_grants_all,
        post_scan_release_reschedules,
    })
}

/// Exercise every deferred-rescan consumer with one real live REPLAN owner.
/// Registrations and removals create ordinary event traffic, while injected
/// monotonic times prove both the heartbeat causal fallback and exact absolute
/// deadline without sleeping.
#[cfg(test)]
pub(super) fn exercise_deferred_rescan_policy_for_tests() -> Result<DeferredRescanPolicyOutcome> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [1_800usize], FlockMode::Exclusive);
    let mut coordinator =
        Ticket::register(coordinator_claim.clone(), coordinator_claim.clone(), None)?;
    let designated = ClaimSet::new(std::iter::empty(), [1_801usize], FlockMode::Exclusive);
    let replacement = ClaimSet::new(std::iter::empty(), [1_802usize], FlockMode::Exclusive);
    let watch = ClaimSet::new(
        std::iter::empty(),
        [1_801usize, 1_802usize],
        FlockMode::Exclusive,
    );
    let mut callback = Ticket::register(designated.clone(), watch, None)?;
    let scans_before = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_replan_capacity_for_tests(1)?;
        for cpu in [1_800usize, 1_801, 1_802] {
            set_cpu_free_for_tests(&mut table, cpu, false)?;
        }
        table.set_urgent_rescan();
        table.grant_compatible()?;
        anyhow::ensure!(
            table.replan_outstanding() == 1
                && table
                    .record(callback.slot)?
                    .is_some_and(|record| record.state == STATE_REPLAN),
            "deferred-rescan fixture did not publish its live REPLAN callback",
        );
        read_u64(&table.header, H_GRANT_SCANS)
    };

    let mut arrivals = Vec::new();
    for index in 0..24usize {
        let exact_cpu = 1_900 + index * 2;
        let alternative_cpu = exact_cpu + 1;
        let exact = ClaimSet::new(std::iter::empty(), [exact_cpu], FlockMode::Exclusive);
        let watch = ClaimSet::new(
            std::iter::empty(),
            [exact_cpu, alternative_cpu],
            FlockMode::Exclusive,
        );
        arrivals.push(Ticket::register(exact, watch, None)?);
    }
    let (first_deadline, registration_coalesced) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        (
            table.deferred_rescan_deadline_ns(),
            table.pending_flags() & PENDING_REPLAN_RESCAN != 0
                && table.pending_flags() & PENDING_RESCAN == 0
                && read_u64(&table.header, H_GRANT_SCANS) == scans_before,
        )
    };
    for ticket in arrivals.iter_mut().take(12) {
        ticket.finish(None)?;
    }
    let registration_and_teardown_coalesced = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        registration_coalesced
            && table.deferred_rescan_deadline_ns() == first_deadline
            && read_u64(&table.header, H_GRANT_SCANS) == scans_before
    };

    // A deferred edge is not structural dirty work. Prove through the public
    // coordinator turn that a known-free release remains SH-only and carries
    // the absolute deadline forward for the protocol loop to honor.
    let (fast_path_deadline, fast_path_scans_before) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.begin_transaction()?;
        set_cpu_free_for_tests(&mut table, 1_800, true)?;
        table.clear_pending_flag(PENDING_RESCAN | PENDING_OBSERVATION);
        if table.pending_flags() & PENDING_REPLAN_RESCAN == 0 {
            table.schedule_deferred_replan_rescan_in_transaction()?;
        }
        table.finish_transaction()?;
        (
            table.deferred_rescan_deadline_ns(),
            read_u64(&table.header, H_GRANT_SCANS),
        )
    };
    // Deflake: the known-free SH fast path requires a fresh coordinator
    // activity lease. Renew the heartbeat word immediately before the
    // release turn so a descheduled test process cannot stale the lease
    // between fixture setup and this assertion — the EX fallback it would
    // otherwise take is correct production behavior, but not what this pin
    // tests. A raw header touch (not `Ticket::heartbeat`, whose loop also
    // promotes deferred edges) leaves the deferred state this test asserts
    // on untouched; it takes the registry EX flock itself, so the
    // EX-acquisition counter is sampled only afterwards.
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.touch_coordinator_heartbeat()?;
    }
    let ex_before = REGISTRY_EX_ACQUISITIONS.with(std::cell::Cell::get);
    let released_cpu = BTreeSet::from([1_800usize]);
    let empty = BTreeSet::new();
    let release_snapshot = coordinator.schedule(
        None,
        &released_cpu,
        &empty,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let ex_after = REGISTRY_EX_ACQUISITIONS.with(std::cell::Cell::get);
    let known_free_release_preserved_deferred_fast_path = {
        let _lock = lock_registry_existing(FlockMode::Shared)?;
        let table = Table::open_existing()?;
        ex_after == ex_before
            && read_u64(&table.header, H_GRANT_SCANS) == fast_path_scans_before
            && table.pending_flags() & PENDING_REPLAN_RESCAN != 0
            && table.pending_flags() & (PENDING_RESCAN | PENDING_OBSERVATION) == 0
            && table.deferred_rescan_deadline_ns() == fast_path_deadline
            && fast_path_deadline != 0
            && release_snapshot.deferred_rescan_due_in.is_some()
            && release_snapshot.observation.is_none()
            && !release_snapshot.should_step
    };

    let ordinary_turn_preserved_deadline = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        !table.prepare_grant_scan_at(first_deadline.saturating_sub(1))?
            && table.deferred_rescan_deadline_ns() == first_deadline
            && table.deferred_rescan_due_in_at(first_deadline.saturating_sub(1))
                == Some(Duration::from_nanos(1))
            && read_u64(&table.header, H_GRANT_SCANS) == scans_before
    };

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_pending_flag(PENDING_OBSERVATION);
    }
    let heartbeat = coordinator.heartbeat_at(Some(first_deadline.saturating_sub(1)), None)?;
    let (heartbeat_promoted_before_deadline, observation_survived_promotion_and_scan) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let promoted = heartbeat.rescan_pending
            && table.pending_flags() & PENDING_RESCAN != 0
            && table.pending_flags() & PENDING_REPLAN_RESCAN == 0
            && table.deferred_rescan_deadline_ns() == 0
            && read_u64(&table.header, H_GRANT_SCANS) == scans_before;
        table.grant_compatible_at(first_deadline.saturating_sub(1), None)?;
        (
            promoted,
            table.pending_flags() & PENDING_OBSERVATION != 0
                && table.pending_flags() & (PENDING_RESCAN | PENDING_REPLAN_RESCAN) == 0,
        )
    };

    let exact_deadline_promoted = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.begin_transaction()?;
        table.schedule_deferred_replan_rescan_in_transaction()?;
        table.finish_transaction()?;
        let deadline = table.deferred_rescan_deadline_ns();
        let before = !table.prepare_grant_scan_at(deadline.saturating_sub(1))?;
        let due = table.prepare_grant_scan_at(deadline)?;
        let promoted = table.pending_flags() & PENDING_RESCAN != 0
            && table.pending_flags() & PENDING_REPLAN_RESCAN == 0;
        table.grant_compatible_at(deadline, None)?;
        before && due && promoted
    };

    let result = callback.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &designated,
                "deferred-rescan fixture received the wrong callback publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: replacement,
                contention: None,
            })
        },
    )?;
    let final_drain_promoted = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        matches!(result, GrantResult::Requeued)
            && table.replan_outstanding() == 0
            && table.pending_flags() & PENDING_RESCAN != 0
            && table.pending_flags() & PENDING_REPLAN_RESCAN == 0
            && table.deferred_rescan_deadline_ns() == 0
    };

    for ticket in arrivals.iter_mut().rev() {
        if !ticket.finished {
            ticket.finish(None)?;
        }
    }
    callback.finish(None)?;
    coordinator.finish(None)?;
    Ok(DeferredRescanPolicyOutcome {
        registration_and_teardown_coalesced,
        known_free_release_preserved_deferred_fast_path,
        ordinary_turn_preserved_deadline,
        heartbeat_promoted_before_deadline,
        observation_survived_promotion_and_scan,
        exact_deadline_promoted,
        final_drain_promoted,
    })
}

/// Prove that an unrelated straggler is accounting/quarantine state, not a
/// grant barrier. The first callback's replacement must become runnable in a
/// normal coordinator scan while the second callback is still in REPLAN and
/// the finite-wave deadline remains in the future.
#[cfg(test)]
pub(super) fn exercise_replan_straggler_progress_for_tests()
-> Result<ReplanStragglerProgressOutcome> {
    let coordinator_cpu = 1_200usize;
    let first_designated_cpu = 1_201usize;
    let first_replacement_cpu = 1_202usize;
    let straggler_designated_cpu = 1_203usize;
    let straggler_replacement_cpu = 1_204usize;
    // The later grant lands exactly on the completion's replacement CPU, so
    // the grant-disjointness damping guard must dirty the full suffix and
    // preserve the old fencing/demotion behavior. The grant-disjoint shape
    // (main watermark stays clean, junior commits) is pinned separately by
    // `exercise_grant_disjoint_completion_for_tests`.
    let later_grant_cpu = first_replacement_cpu;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let first_designated = ClaimSet::new(
        std::iter::empty(),
        [first_designated_cpu],
        FlockMode::Exclusive,
    );
    let first_replacement = ClaimSet::new(
        std::iter::empty(),
        [first_replacement_cpu],
        FlockMode::Exclusive,
    );
    let first_watch = ClaimSet::new(
        std::iter::empty(),
        [first_designated_cpu, first_replacement_cpu],
        FlockMode::Exclusive,
    );
    let mut first = Ticket::register(first_designated.clone(), first_watch, None)?;
    let straggler_designated = ClaimSet::new(
        std::iter::empty(),
        [straggler_designated_cpu],
        FlockMode::Exclusive,
    );
    let straggler_watch = ClaimSet::new(
        std::iter::empty(),
        [straggler_designated_cpu, straggler_replacement_cpu],
        FlockMode::Exclusive,
    );
    let mut straggler = Ticket::register(straggler_designated.clone(), straggler_watch, None)?;
    let later_grant_claim =
        ClaimSet::new(std::iter::empty(), [later_grant_cpu], FlockMode::Exclusive);
    let mut later_grant =
        Ticket::register(later_grant_claim.clone(), later_grant_claim.clone(), None)?;

    let (scans_before, generation_wake_before, wave_deadline_ns, notify_before) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        for cpu in [
            coordinator_cpu,
            first_designated_cpu,
            first_replacement_cpu,
            straggler_designated_cpu,
            straggler_replacement_cpu,
        ] {
            set_cpu_free_for_tests(&mut table, cpu, false)?;
        }
        set_cpu_free_for_tests(&mut table, later_grant_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), None)?;
        anyhow::ensure!(
            table.replan_outstanding() == 2
                && table
                    .record(first.slot)?
                    .is_some_and(|record| record.state == STATE_REPLAN)
                && table
                    .record(straggler.slot)?
                    .is_some_and(|record| record.state == STATE_REPLAN)
                && table
                    .record(later_grant.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED),
            "straggler progress fixture did not publish both callbacks and its conflicting later grant",
        );
        (
            read_u64(&table.header, H_GRANT_SCANS),
            table.generation_wake(),
            table.replan_wave_deadline_ns(),
            NOTIFY_CALLS.with(std::cell::Cell::get),
        )
    };

    let callback_result = first.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &first_designated,
                "completed straggler fixture callback received the wrong publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: first_replacement.clone(),
                contention: None,
            })
        },
    )?;
    let completion_requeued = matches!(callback_result, GrantResult::Requeued);
    let callback_notify_delta = NOTIFY_CALLS
        .with(std::cell::Cell::get)
        .wrapping_sub(notify_before);
    let (
        deferred_not_due_early,
        deferred_due_at_deadline,
        deferred_deadline_ns,
        edge_coalesced_with_straggler,
    ) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let deadline = table.deferred_rescan_deadline_ns();
        (
            deadline != 0 && !table.deferred_rescan_due_at(deadline.saturating_sub(1)),
            deadline != 0 && table.deferred_rescan_due_at(deadline),
            deadline,
            table.replan_outstanding() == 1
                && table.pending_flags() & PENDING_REPLAN_RESCAN != 0
                && table.pending_flags() & PENDING_RESCAN == 0
                && table
                    .record(straggler.slot)?
                    .is_some_and(|record| record.state == STATE_REPLAN),
        )
    };

    let mut dirty_later_callback_ran = false;
    let dirty_later_result = later_grant.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            dirty_later_callback_ran = true;
            anyhow::ensure!(
                acquisition_allowed && current == &later_grant_claim,
                "later disjoint grant received the wrong publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: later_grant_claim.clone(),
                contention: None,
            })
        },
    )?;
    let later_grant_fenced_until_scan =
        !dirty_later_callback_ran && matches!(dirty_later_result, GrantResult::LostGrant);

    let (
        callback_scan_delta,
        callback_generation_wake_delta,
        later_grant_demotion_shortened_deferred_edge,
        later_grant_demotion_notified_once,
        later_grant_demotion_deadline_exact,
        authoritative_scan_delta,
        completed_replacement_granted,
        straggler_still_replan,
        wave_deadline_not_reached,
    ) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let callback_scan_delta = read_u64(&table.header, H_GRANT_SCANS).wrapping_sub(scans_before);
        let callback_generation_wake_delta =
            table.generation_wake().wrapping_sub(generation_wake_before);
        let shortened_deadline_ns = table.deferred_rescan_deadline_ns();
        let later_grant_demotion_shortened_deferred_edge = table.replan_outstanding() == 1
            && table.pending_flags() & PENDING_REPLAN_RESCAN != 0
            && table.pending_flags() & PENDING_RESCAN == 0
            && shortened_deadline_ns != 0
            && shortened_deadline_ns < deferred_deadline_ns
            && table
                .record(straggler.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN);
        let later_grant_demotion_notified_once = NOTIFY_CALLS
            .with(std::cell::Cell::get)
            .wrapping_sub(notify_before)
            == 1;
        let later_grant_demotion_deadline_exact = shortened_deadline_ns != 0
            && !table.deferred_rescan_due_at(shortened_deadline_ns.saturating_sub(1))
            && table.deferred_rescan_due_at(shortened_deadline_ns);
        set_cpu_free_for_tests(&mut table, first_replacement_cpu, true)?;
        set_cpu_free_for_tests(&mut table, later_grant_cpu, true)?;
        anyhow::ensure!(
            table.prepare_grant_scan_at(shortened_deadline_ns)?,
            "shortened dirty-grant deadline did not promote its deferred scan",
        );
        table.grant_compatible_at(shortened_deadline_ns, None)?;
        let authoritative_scan_delta =
            read_u64(&table.header, H_GRANT_SCANS).wrapping_sub(scans_before);
        let completed_replacement_granted = table.record(first.slot)?.is_some_and(|record| {
            record.state == STATE_GRANTED && record.claim == first_replacement
        });
        let straggler_still_replan = table
            .record(straggler.slot)?
            .is_some_and(|record| record.state == STATE_REPLAN)
            && table.replan_outstanding() == 1;
        let wave_deadline_not_reached = table.replan_wave_deadline_ns() == wave_deadline_ns
            && shortened_deadline_ns < wave_deadline_ns;
        (
            callback_scan_delta,
            callback_generation_wake_delta,
            later_grant_demotion_shortened_deferred_edge,
            later_grant_demotion_notified_once,
            later_grant_demotion_deadline_exact,
            authoritative_scan_delta,
            completed_replacement_granted,
            straggler_still_replan,
            wave_deadline_not_reached,
        )
    };

    let mut later_callback_ran = false;
    let later_result = later_grant.run_granted(
        None,
        |_current, _watch, _acquisition_allowed, _predecessors, _availability| {
            later_callback_ran = true;
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: later_grant_claim.clone(),
                contention: None,
            })
        },
    )?;
    // The authoritative scan granted the SENIOR completed replacement onto
    // this CPU, so the demoted junior must stay WAITING — never regranted
    // ahead of the conflicting senior.
    let later_conflicting_grant_not_regranted =
        !later_callback_ran && matches!(later_result, GrantResult::LostGrant);

    later_grant.finish(None)?;
    straggler.finish(None)?;
    first.finish(None)?;
    coordinator.finish(None)?;
    Ok(ReplanStragglerProgressOutcome {
        completion_requeued,
        callback_notify_delta,
        deferred_not_due_early,
        deferred_due_at_deadline,
        callback_scan_delta,
        callback_generation_wake_delta,
        edge_coalesced_with_straggler,
        later_grant_fenced_until_scan,
        later_grant_demotion_shortened_deferred_edge,
        later_grant_demotion_notified_once,
        later_grant_demotion_deadline_exact,
        later_conflicting_grant_not_regranted,
        authoritative_scan_delta,
        completed_replacement_granted,
        straggler_still_replan,
        wave_deadline_not_reached,
    })
}

/// Compare every `C_GRANT_*` header count array against a recomputation from
/// the live records (charged set = {GRANTED, REVOKED}). The strongest leak
/// tripwire available to tests: any unbalanced charge site shows up as a
/// per-resource mismatch here even before an adjust helper under/overflows.
#[cfg(test)]
fn grant_charge_derived_matches(table: &mut Table) -> Result<()> {
    let bits = table.layout.bits;
    let mut cpu_any = vec![0u32; bits];
    let mut cpu_ex = vec![0u32; bits];
    let mut llc_any = vec![0u32; bits];
    let mut llc_ex = vec![0u32; bits];
    for record in table.records()? {
        if !charged_state(record.state) {
            continue;
        }
        let claim = &record.claim;
        for &cpu in &claim.cpus {
            cpu_any[cpu] += 1;
            if claim.cpu_mode == ClaimMode::Exclusive {
                cpu_ex[cpu] += 1;
            }
        }
        for &permit in &claim.permits {
            let index = permit_resource_index(permit)?;
            cpu_any[index] += 1;
            if claim.permit_mode == ClaimMode::Exclusive {
                cpu_ex[index] += 1;
            }
        }
        for &llc in &claim.llcs {
            llc_any[llc] += 1;
            if claim.llc_mode == ClaimMode::Exclusive {
                llc_ex[llc] += 1;
            }
        }
    }
    for (name, which, expected) in [
        ("C_GRANT_CPU_ANY", C_GRANT_CPU_ANY, &cpu_any),
        ("C_GRANT_CPU_EX", C_GRANT_CPU_EX, &cpu_ex),
        ("C_GRANT_LLC_ANY", C_GRANT_LLC_ANY, &llc_any),
        ("C_GRANT_LLC_EX", C_GRANT_LLC_EX, &llc_ex),
    ] {
        let actual = table.header_counts(which);
        for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            anyhow::ensure!(
                actual == expected,
                "grant charge {name}[{index}] is {actual}, derived from records: {expected}",
            );
        }
    }
    Ok(())
}

#[cfg(test)]
fn grant_charge_total(table: &Table) -> u64 {
    table
        .header_counts(C_GRANT_CPU_ANY)
        .iter()
        .chain(table.header_counts(C_GRANT_LLC_ANY).iter())
        .map(|&count| u64::from(count))
        .sum()
}

/// Assert aggregate == derived for the grant-charge families in the shared
/// registry image, running dirty repair first exactly like any entrant.
#[cfg(test)]
pub(super) fn grant_charge_matches_derived_for_tests() -> Result<()> {
    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    grant_charge_derived_matches(&mut table)
}

#[cfg(test)]
pub(crate) struct GrantedChargeLifecycleOutcome {
    pub(crate) scan_granted_charged: bool,
    pub(crate) pending_uncharged: bool,
    pub(crate) born_granted_charged: bool,
    pub(crate) held_uncharged: bool,
    pub(crate) drained_to_zero: bool,
}

/// Drive one grant through GRANTED -> PENDING (preparation) -> removal and a
/// second through born-GRANTED -> HELD -> removal, auditing the `C_GRANT_*`
/// charge against the derived truth at every stage. Pins the
/// GRANTED->PENDING release (the old claim, not the preparation claim), the
/// born-GRANTED registration charge, and the unconditional promote release.
#[cfg(test)]
pub(super) fn exercise_granted_charge_lifecycle_for_tests() -> Result<GrantedChargeLifecycleOutcome>
{
    let coordinator_cpu = 2_300usize;
    let scan_cpu = 2_301usize;
    let prep_cpu = 2_302usize;
    let born_cpu = 2_303usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let scan_claim = ClaimSet::new(std::iter::empty(), [scan_cpu], FlockMode::Exclusive);
    let mut scan_ticket = Ticket::register(scan_claim.clone(), scan_claim.clone(), None)?;
    let scan_granted_charged = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, scan_cpu, true)?;
        set_cpu_free_for_tests(&mut table, prep_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), None)?;
        anyhow::ensure!(
            table
                .record(scan_ticket.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED),
            "charge lifecycle fixture did not grant its scan waiter",
        );
        grant_charge_derived_matches(&mut table)?;
        grant_charge_total(&table) == 1
    };
    let preparation = ClaimSet::new(std::iter::empty(), [prep_cpu], FlockMode::Exclusive);
    let scan_claim_for_wake = scan_claim.clone();
    let preparation_for_wake = preparation.clone();
    let result = scan_ticket.run_granted(None, |current, _watch, acquisition_allowed, _, _| {
        anyhow::ensure!(
            acquisition_allowed && current == &scan_claim_for_wake,
            "charge lifecycle callback received the wrong publication",
        );
        Ok(GrantAttempt::<()> {
            acquired: Some(()),
            preparation_claim: Some(preparation_for_wake.clone()),
            preparation_contention: None,
            next_claim: scan_claim_for_wake.clone(),
            contention: None,
        })
    })?;
    anyhow::ensure!(
        matches!(result, GrantResult::Prepared((), _)),
        "charge lifecycle preparation grant did not reach PENDING",
    );
    let pending_uncharged = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        anyhow::ensure!(
            table
                .record(scan_ticket.slot)?
                .is_some_and(|record| record.state == STATE_PENDING),
            "charge lifecycle preparation grant is not PENDING",
        );
        grant_charge_derived_matches(&mut table)?;
        grant_charge_total(&table) == 0
    };
    scan_ticket.finish(None)?;
    // The PENDING removal dirtied the suffix watermark; run one scan so the
    // born-GRANTED entrant below is not parked by unrelated churn.
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, born_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), None)?;
    }
    let born_claim = ClaimSet::new(std::iter::empty(), [born_cpu], FlockMode::Exclusive);
    let mut born = Ticket::register(born_claim.clone(), born_claim.clone(), None)?;
    let born_granted_charged = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        anyhow::ensure!(
            table
                .record(born.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED),
            "registration did not take the born-GRANTED fast path",
        );
        grant_charge_derived_matches(&mut table)?;
        grant_charge_total(&table) == 1
    };
    let born_claim_for_wake = born_claim.clone();
    let result = born.run_granted(None, |current, _watch, acquisition_allowed, _, _| {
        anyhow::ensure!(
            acquisition_allowed && current == &born_claim_for_wake,
            "born-GRANTED callback received the wrong publication",
        );
        Ok(GrantAttempt::<()> {
            acquired: Some(()),
            preparation_claim: None,
            preparation_contention: None,
            next_claim: born_claim_for_wake.clone(),
            contention: None,
        })
    })?;
    let held = match result {
        GrantResult::Acquired((), held) => held,
        _ => anyhow::bail!("born-GRANTED acquire did not commit HELD"),
    };
    let held_uncharged = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        grant_charge_derived_matches(&mut table)?;
        grant_charge_total(&table) == 0
    };
    drop(held);
    coordinator.finish(None)?;
    let drained_to_zero = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        grant_charge_derived_matches(&mut table)?;
        grant_charge_total(&table) == 0
    };
    Ok(GrantedChargeLifecycleOutcome {
        scan_granted_charged,
        pending_uncharged,
        born_granted_charged,
        held_uncharged,
        drained_to_zero,
    })
}

#[cfg(test)]
pub(crate) struct GrantChargeRevokeAckOutcome {
    pub(crate) junior_granted_charged: bool,
    pub(crate) replacement_dirtied_watermark: bool,
    pub(crate) senior_granted_and_junior_revoked: bool,
    pub(crate) revoked_stays_charged: bool,
    pub(crate) ack_releases_charge: bool,
}

/// End-to-end ticket-order win on a grant-charged resource: a senior REPLAN
/// publishes a replacement overlapping a junior's grant (the guard dirties
/// the suffix), the next authoritative scan grants the senior and revokes the
/// junior, and the revoked claim stays charged until its acknowledgement.
#[cfg(test)]
pub(super) fn exercise_grant_charge_revoke_ack_for_tests() -> Result<GrantChargeRevokeAckOutcome> {
    let coordinator_cpu = 2_400usize;
    let senior_designated_cpu = 2_401usize;
    let contended_cpu = 2_402usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let senior_designated = ClaimSet::new(
        std::iter::empty(),
        [senior_designated_cpu],
        FlockMode::Exclusive,
    );
    let senior_watch = ClaimSet::new(
        std::iter::empty(),
        [senior_designated_cpu, contended_cpu],
        FlockMode::Exclusive,
    );
    let mut senior = Ticket::register(senior_designated.clone(), senior_watch, None)?;
    let junior_claim = ClaimSet::new(std::iter::empty(), [contended_cpu], FlockMode::Exclusive);
    let mut junior = Ticket::register(junior_claim.clone(), junior_claim.clone(), None)?;
    let junior_granted_charged = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, senior_designated_cpu, false)?;
        set_cpu_free_for_tests(&mut table, contended_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), None)?;
        anyhow::ensure!(
            table
                .record(senior.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN)
                && table
                    .record(junior.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED),
            "revoke-ack fixture did not publish its senior callback and junior grant",
        );
        grant_charge_derived_matches(&mut table)?;
        grant_charge_total(&table) == 1
    };
    let replacement = junior_claim.clone();
    let senior_designated_for_wake = senior_designated.clone();
    let replacement_for_wake = replacement.clone();
    let result = senior.run_granted(None, |current, _watch, acquisition_allowed, _, _| {
        anyhow::ensure!(
            !acquisition_allowed && current == &senior_designated_for_wake,
            "revoke-ack senior callback received the wrong publication",
        );
        Ok(GrantAttempt::<()> {
            acquired: None,
            preparation_claim: None,
            preparation_contention: None,
            next_claim: replacement_for_wake.clone(),
            contention: None,
        })
    })?;
    anyhow::ensure!(
        matches!(result, GrantResult::Requeued),
        "revoke-ack senior completion did not requeue",
    );
    let (replacement_dirtied_watermark, senior_granted_and_junior_revoked, revoked_stays_charged) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        // The replacement overlaps the junior grant's charge, so the guard
        // must have dirtied the full suffix (both watermark words).
        let replacement_dirtied_watermark = table.min_changed_ticket() < junior.ticket
            && table.min_changed_ticket_replan() < junior.ticket;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), None)?;
        let senior_granted_and_junior_revoked = table
            .record(senior.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED && record.claim == replacement)
            && table
                .record(junior.slot)?
                .is_some_and(|record| record.state == STATE_REVOKED);
        grant_charge_derived_matches(&mut table)?;
        // Senior GRANTED and junior REVOKED both charge the contended CPU.
        let revoked_stays_charged = grant_charge_total(&table) == 2;
        (
            replacement_dirtied_watermark,
            senior_granted_and_junior_revoked,
            revoked_stays_charged,
        )
    };
    // The junior's next state read acknowledges the revocation.
    let junior_state = junior.state(None)?;
    let ack_releases_charge = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        anyhow::ensure!(
            junior_state == State::Waiting
                && table
                    .record(junior.slot)?
                    .is_some_and(|record| record.state == STATE_WAITING),
            "revoked junior did not acknowledge back to WAITING",
        );
        grant_charge_derived_matches(&mut table)?;
        grant_charge_total(&table) == 1
    };
    junior.finish(None)?;
    senior.finish(None)?;
    coordinator.finish(None)?;
    Ok(GrantChargeRevokeAckOutcome {
        junior_granted_charged,
        replacement_dirtied_watermark,
        senior_granted_and_junior_revoked,
        revoked_stays_charged,
        ack_releases_charge,
    })
}

/// Crash a writer mid-transaction with a live GRANTED and a live REVOKED
/// record, then prove dirty repair rebuilds the grant charge without
/// underflow: the demotion loop releases the GRANTED record's charge through
/// the `set_record_state` chokepoint over the just-zeroed count arrays, and
/// the surviving REVOKED fence stays charged.
#[cfg(test)]
pub(super) fn exercise_dirty_repair_grant_charges_for_tests() -> Result<(bool, bool)> {
    let coordinator_cpu = 2_500usize;
    let granted_cpu = 2_501usize;
    let revoked_cpu = 2_502usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let granted_claim = ClaimSet::new(std::iter::empty(), [granted_cpu], FlockMode::Exclusive);
    let mut granted = Ticket::register(granted_claim.clone(), granted_claim.clone(), None)?;
    let revoked_claim = ClaimSet::new(std::iter::empty(), [revoked_cpu], FlockMode::Exclusive);
    let mut revoked = Ticket::register(revoked_claim.clone(), revoked_claim.clone(), None)?;
    let (repaired_without_underflow, revoked_charge_preserved) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, granted_cpu, true)?;
        set_cpu_free_for_tests(&mut table, revoked_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), None)?;
        anyhow::ensure!(
            table
                .record(granted.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED)
                && table
                    .record(revoked.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED),
            "dirty-repair charge fixture did not grant both waiters",
        );
        table.begin_transaction()?;
        table.set_record_state(revoked.slot, STATE_REVOKED)?;
        table.finish_transaction()?;
        // Model a writer dying mid-transaction after the states were
        // published: the dirty marker alone forces a full rebuild.
        atomic_u64(&table.header, H_AGGREGATE_DIRTY).store(1, Ordering::SeqCst);
        table.repair_consistency_if_needed()?;
        grant_charge_derived_matches(&mut table)?;
        let repaired = table
            .record(granted.slot)?
            .is_some_and(|record| record.state == STATE_WAITING)
            && table
                .record(revoked.slot)?
                .is_some_and(|record| record.state == STATE_REVOKED);
        (repaired, grant_charge_total(&table) == 1)
    };
    revoked.finish(None)?;
    granted.finish(None)?;
    coordinator.finish(None)?;
    Ok((repaired_without_underflow, revoked_charge_preserved))
}

/// A granted-EX claim must stay planner *bias*: it appears in the grant
/// counts but never in `exclusive_held` or the holder counts, so it can
/// never become an absolute placement fence.
#[cfg(test)]
pub(super) fn exercise_exclusive_grant_bias_only_for_tests() -> Result<(bool, bool, bool, bool)> {
    let coordinator_cpu = 2_600usize;
    let granted_cpu = 2_601usize;
    let granted_llc = 9usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, granted_cpu, true)?;
        table.set_bitmap_bit(B_LLC_KNOWN, granted_llc, true)?;
        table.set_bitmap_bit(B_LLC_SH_AVAILABLE, granted_llc, true)?;
        table.set_bitmap_bit(B_LLC_EX_AVAILABLE, granted_llc, true)?;
    }
    let granted_claim = ClaimSet::with_modes(
        [granted_llc],
        [granted_cpu],
        FlockMode::Exclusive,
        FlockMode::Exclusive,
    );
    let mut granted = Ticket::register(granted_claim.clone(), granted_claim, None)?;
    let outcome = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        anyhow::ensure!(
            table
                .record(granted.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED),
            "exclusive-bias fixture did not take the born-GRANTED fast path",
        );
        let snapshot = table.aggregate_claim_snapshot();
        (
            !snapshot.cpu_exclusive_held(granted_cpu)?,
            !snapshot.llc_exclusive_held(granted_llc)?,
            snapshot.cpu_grant_count(granted_cpu)? == 1,
            snapshot.llc_grant_count(granted_llc)? == 1,
        )
    };
    granted.finish(None)?;
    coordinator.finish(None)?;
    Ok(outcome)
}

#[cfg(test)]
pub(crate) struct GrantDisjointCompletionOutcome {
    pub(crate) main_watermark_clean: bool,
    pub(crate) replan_watermark_dirty: bool,
    pub(crate) junior_committed_held_during_wave: bool,
    pub(crate) straggler_same_wake_fenced: bool,
    pub(crate) straggler_requeued_waiting: bool,
}

/// The guard-skip canary: a NonFencing replacement disjoint from every grant
/// charge leaves the GRANTED-facing watermark clean, so the in-flight
/// disjoint junior grant enters and commits to HELD while the wave is still
/// outstanding — while the always-dirtied replan word still releases and
/// requeues an unlicensed same-wake REPLAN acquire (the split-watermark
/// inversion fix). Disjoint grants forgo park-driven deferred-edge
/// shortening and ride the batched rescan deadline instead.
#[cfg(test)]
pub(super) fn exercise_grant_disjoint_completion_for_tests()
-> Result<GrantDisjointCompletionOutcome> {
    let coordinator_cpu = 2_700usize;
    let first_designated_cpu = 2_701usize;
    let first_replacement_cpu = 2_702usize;
    let straggler_designated_cpu = 2_703usize;
    let straggler_replacement_cpu = 2_704usize;
    let later_grant_cpu = 2_705usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let first_designated = ClaimSet::new(
        std::iter::empty(),
        [first_designated_cpu],
        FlockMode::Exclusive,
    );
    let first_replacement = ClaimSet::new(
        std::iter::empty(),
        [first_replacement_cpu],
        FlockMode::Exclusive,
    );
    let first_watch = ClaimSet::new(
        std::iter::empty(),
        [first_designated_cpu, first_replacement_cpu],
        FlockMode::Exclusive,
    );
    let mut first = Ticket::register(first_designated.clone(), first_watch, None)?;
    let straggler_designated = ClaimSet::new(
        std::iter::empty(),
        [straggler_designated_cpu],
        FlockMode::Exclusive,
    );
    let straggler_replacement = ClaimSet::new(
        std::iter::empty(),
        [straggler_replacement_cpu],
        FlockMode::Exclusive,
    );
    let straggler_watch = ClaimSet::new(
        std::iter::empty(),
        [straggler_designated_cpu, straggler_replacement_cpu],
        FlockMode::Exclusive,
    );
    let mut straggler = Ticket::register(straggler_designated.clone(), straggler_watch, None)?;
    let later_grant_claim =
        ClaimSet::new(std::iter::empty(), [later_grant_cpu], FlockMode::Exclusive);
    let mut later_grant =
        Ticket::register(later_grant_claim.clone(), later_grant_claim.clone(), None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        for cpu in [
            coordinator_cpu,
            first_designated_cpu,
            first_replacement_cpu,
            straggler_designated_cpu,
            straggler_replacement_cpu,
        ] {
            set_cpu_free_for_tests(&mut table, cpu, false)?;
        }
        set_cpu_free_for_tests(&mut table, later_grant_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), None)?;
        anyhow::ensure!(
            table.replan_outstanding() == 2
                && table
                    .record(first.slot)?
                    .is_some_and(|record| record.state == STATE_REPLAN)
                && table
                    .record(straggler.slot)?
                    .is_some_and(|record| record.state == STATE_REPLAN)
                && table
                    .record(later_grant.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED),
            "disjoint-completion fixture did not publish both callbacks and its later grant",
        );
    }
    let first_designated_for_wake = first_designated.clone();
    let first_replacement_for_wake = first_replacement.clone();
    let result = first.run_granted(None, |current, _watch, acquisition_allowed, _, _| {
        anyhow::ensure!(
            !acquisition_allowed && current == &first_designated_for_wake,
            "disjoint-completion callback received the wrong publication",
        );
        Ok(GrantAttempt::<()> {
            acquired: None,
            preparation_claim: None,
            preparation_contention: None,
            next_claim: first_replacement_for_wake.clone(),
            contention: None,
        })
    })?;
    anyhow::ensure!(
        matches!(result, GrantResult::Requeued),
        "disjoint-completion replacement did not requeue",
    );
    let (main_watermark_clean, replan_watermark_dirty) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        (
            table.min_changed_ticket() == u64::MAX,
            table.min_changed_ticket_replan() <= first.ticket,
        )
    };
    let later_grant_claim_for_wake = later_grant_claim.clone();
    let mut later_callback_ran = false;
    let later_result = later_grant.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            later_callback_ran = true;
            anyhow::ensure!(
                acquisition_allowed && current == &later_grant_claim_for_wake,
                "disjoint later grant received the wrong publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: Some(()),
                preparation_claim: None,
                preparation_contention: None,
                next_claim: later_grant_claim_for_wake.clone(),
                contention: None,
            })
        },
    )?;
    let held = match later_result {
        GrantResult::Acquired((), held) => Some(held),
        _ => None,
    };
    let junior_committed_held_during_wave = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        later_callback_ran
            && held.is_some()
            && table.replan_outstanding() == 1
            && table
                .record(later_grant.slot)?
                .is_some_and(|record| record.state == STATE_HELD)
            && table
                .record(straggler.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN)
    };
    // The unlicensed straggler acquires physically in the same wake. The
    // replan watermark word — dirtied by the guard-skipped replacement whose
    // claim it cannot see in any charge — must release and requeue it.
    let straggler_designated_for_wake = straggler_designated.clone();
    let straggler_replacement_for_wake = straggler_replacement.clone();
    let mut straggler_callback_ran = false;
    let straggler_result = straggler.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            straggler_callback_ran = true;
            anyhow::ensure!(
                !acquisition_allowed && current == &straggler_designated_for_wake,
                "straggler same-wake callback received the wrong publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: Some(()),
                preparation_claim: None,
                preparation_contention: None,
                next_claim: straggler_replacement_for_wake.clone(),
                contention: None,
            })
        },
    )?;
    let straggler_same_wake_fenced =
        straggler_callback_ran && matches!(straggler_result, GrantResult::LostGrant);
    let straggler_requeued_waiting = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.record(straggler.slot)?.is_some_and(|record| {
            record.state == STATE_WAITING && record.claim == straggler_designated
        }) && table.replan_outstanding() == 0
    };
    drop(held);
    later_grant.finish(None)?;
    straggler.finish(None)?;
    first.finish(None)?;
    coordinator.finish(None)?;
    Ok(GrantDisjointCompletionOutcome {
        main_watermark_clean,
        replan_watermark_dirty,
        junior_committed_held_during_wave,
        straggler_same_wake_fenced,
        straggler_requeued_waiting,
    })
}

/// Site-B guard (unchanged REPLAN completion): grant-disjoint completions
/// leave the main watermark clean and the junior grant enters; overlapping
/// completions dirty the full suffix and park the junior.
#[cfg(test)]
pub(super) fn exercise_unchanged_completion_guard_for_tests(
    conflicting: bool,
) -> Result<(bool, bool)> {
    let base = if conflicting { 2_800usize } else { 2_820usize };
    let coordinator_cpu = base;
    let shared_cpu = base + 1;
    let busy_cpu = base + 2;
    let junior_cpu = if conflicting { shared_cpu } else { base + 3 };
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    // The senior's exact claim spans a free CPU (the potential overlap) and a
    // busy one, so the wave publishes it as a speculative callback while a
    // junior grant can still land on the free CPU (WAITING/REPLAN claims do
    // not fence the grant scan).
    let senior_claim = ClaimSet::new(
        std::iter::empty(),
        [shared_cpu, busy_cpu],
        FlockMode::Exclusive,
    );
    let senior_watch = ClaimSet::new(
        std::iter::empty(),
        [shared_cpu, busy_cpu, base + 4],
        FlockMode::Exclusive,
    );
    let mut senior = Ticket::register(senior_claim.clone(), senior_watch, None)?;
    let junior_claim = ClaimSet::new(std::iter::empty(), [junior_cpu], FlockMode::Exclusive);
    let mut junior = Ticket::register(junior_claim.clone(), junior_claim.clone(), None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, shared_cpu, true)?;
        set_cpu_free_for_tests(&mut table, busy_cpu, false)?;
        set_cpu_free_for_tests(&mut table, junior_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), None)?;
        anyhow::ensure!(
            table
                .record(senior.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN)
                && table
                    .record(junior.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED),
            "unchanged-completion fixture did not publish its callback and junior grant",
        );
    }
    let senior_claim_for_wake = senior_claim.clone();
    let result = senior.run_granted(None, |current, _watch, acquisition_allowed, _, _| {
        anyhow::ensure!(
            !acquisition_allowed && current == &senior_claim_for_wake,
            "unchanged-completion callback received the wrong publication",
        );
        Ok(GrantAttempt::<()> {
            acquired: None,
            preparation_claim: None,
            preparation_contention: None,
            next_claim: senior_claim_for_wake.clone(),
            contention: None,
        })
    })?;
    anyhow::ensure!(
        matches!(result, GrantResult::Requeued),
        "unchanged completion did not requeue",
    );
    let guarded_as_expected = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        let main_dirty = table.min_changed_ticket() < junior.ticket;
        let replan_dirty = table.min_changed_ticket_replan() < junior.ticket;
        replan_dirty && main_dirty == conflicting
    };
    let junior_claim_for_wake = junior_claim.clone();
    let mut junior_callback_ran = false;
    let junior_result = later_grant_probe(
        &mut junior,
        &junior_claim_for_wake,
        &mut junior_callback_ran,
    )?;
    let junior_behaved_as_expected = if conflicting {
        !junior_callback_ran && matches!(junior_result, GrantResult::LostGrant)
    } else {
        junior_callback_ran && matches!(junior_result, GrantResult::Requeued)
    };
    junior.finish(None)?;
    senior.finish(None)?;
    coordinator.finish(None)?;
    Ok((guarded_as_expected, junior_behaved_as_expected))
}

/// One non-acquiring probe of a GRANTED junior: records whether its callback
/// entered, returning the raw grant result.
#[cfg(test)]
fn later_grant_probe(
    junior: &mut Ticket,
    claim: &ClaimSet,
    callback_ran: &mut bool,
) -> Result<GrantResult<()>> {
    let claim = claim.clone();
    junior.run_granted(None, |_current, _watch, _acquisition_allowed, _, _| {
        *callback_ran = true;
        Ok(GrantAttempt::<()> {
            acquired: None,
            preparation_claim: None,
            preparation_contention: None,
            next_claim: claim.clone(),
            contention: None,
        })
    })
}

/// Site-A guard, kept-resource inversion: a changed replacement that KEEPS a
/// resource a junior grant sits on must dirty the full suffix — delta
/// cleanliness is not disjointness.
#[cfg(test)]
pub(super) fn exercise_replacement_kept_overlap_guard_for_tests() -> Result<(bool, bool)> {
    let coordinator_cpu = 2_900usize;
    let busy_cpu = 2_901usize;
    let kept_cpu = 2_902usize;
    let fresh_cpu = 2_903usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let senior_designated = ClaimSet::new(
        std::iter::empty(),
        [busy_cpu, kept_cpu],
        FlockMode::Exclusive,
    );
    let senior_replacement = ClaimSet::new(
        std::iter::empty(),
        [kept_cpu, fresh_cpu],
        FlockMode::Exclusive,
    );
    let senior_watch = ClaimSet::new(
        std::iter::empty(),
        [busy_cpu, kept_cpu, fresh_cpu],
        FlockMode::Exclusive,
    );
    let mut senior = Ticket::register(senior_designated.clone(), senior_watch, None)?;
    let junior_claim = ClaimSet::new(std::iter::empty(), [kept_cpu], FlockMode::Exclusive);
    let mut junior = Ticket::register(junior_claim.clone(), junior_claim.clone(), None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, busy_cpu, false)?;
        set_cpu_free_for_tests(&mut table, kept_cpu, true)?;
        set_cpu_free_for_tests(&mut table, fresh_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), None)?;
        anyhow::ensure!(
            table
                .record(senior.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN)
                && table
                    .record(junior.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED),
            "kept-overlap fixture did not publish its callback and junior grant",
        );
    }
    let senior_designated_for_wake = senior_designated.clone();
    let senior_replacement_for_wake = senior_replacement.clone();
    let result = senior.run_granted(None, |current, _watch, acquisition_allowed, _, _| {
        anyhow::ensure!(
            !acquisition_allowed && current == &senior_designated_for_wake,
            "kept-overlap callback received the wrong publication",
        );
        Ok(GrantAttempt::<()> {
            acquired: None,
            preparation_claim: None,
            preparation_contention: None,
            next_claim: senior_replacement_for_wake.clone(),
            contention: None,
        })
    })?;
    anyhow::ensure!(
        matches!(result, GrantResult::Requeued),
        "kept-overlap replacement did not requeue",
    );
    let both_words_dirty = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        table.min_changed_ticket() < junior.ticket
            && table.min_changed_ticket_replan() < junior.ticket
    };
    let mut junior_callback_ran = false;
    let junior_result = later_grant_probe(&mut junior, &junior_claim, &mut junior_callback_ran)?;
    let junior_parked = !junior_callback_ran && matches!(junior_result, GrantResult::LostGrant);
    junior.finish(None)?;
    senior.finish(None)?;
    coordinator.finish(None)?;
    Ok((both_words_dirty, junior_parked))
}

#[cfg(test)]
pub(crate) struct DisjointEntrantProceedsOutcome {
    pub(crate) watermark_dirty_below_juniors: bool,
    pub(crate) overlapping_junior_parked: bool,
    pub(crate) disjoint_junior_committed_held: bool,
}

/// The overlap-tested entry park: one senior replacement dirties the suffix
/// with a claim overlapping junior1's grant. Junior1 must still park at
/// entry (fairness for the genuine overlap), while junior2 — granted a
/// claim disjoint from everything accumulated since the last scan —
/// proceeds through BOTH the entry test and the acquired-commit test and
/// converts to HELD, instead of the blanket park both juniors got before
/// the changed-claims accumulator.
#[cfg(test)]
pub(super) fn exercise_disjoint_entrant_proceeds_for_tests()
-> Result<DisjointEntrantProceedsOutcome> {
    let coordinator_cpu = 3_000usize;
    let senior_designated_cpu = 3_001usize;
    let contended_cpu = 3_002usize;
    let disjoint_cpu = 3_003usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let senior_designated = ClaimSet::new(
        std::iter::empty(),
        [senior_designated_cpu],
        FlockMode::Exclusive,
    );
    let senior_watch = ClaimSet::new(
        std::iter::empty(),
        [senior_designated_cpu, contended_cpu],
        FlockMode::Exclusive,
    );
    let mut senior = Ticket::register(senior_designated.clone(), senior_watch, None)?;
    let junior1_claim = ClaimSet::new(std::iter::empty(), [contended_cpu], FlockMode::Exclusive);
    let mut junior1 = Ticket::register(junior1_claim.clone(), junior1_claim.clone(), None)?;
    let junior2_claim = ClaimSet::new(std::iter::empty(), [disjoint_cpu], FlockMode::Exclusive);
    let mut junior2 = Ticket::register(junior2_claim.clone(), junior2_claim.clone(), None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, senior_designated_cpu, false)?;
        set_cpu_free_for_tests(&mut table, contended_cpu, true)?;
        set_cpu_free_for_tests(&mut table, disjoint_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), None)?;
        anyhow::ensure!(
            table
                .record(senior.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN)
                && table
                    .record(junior1.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED)
                && table
                    .record(junior2.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED),
            "disjoint-entrant fixture did not publish its callback and grants",
        );
    }
    let senior_designated_for_wake = senior_designated.clone();
    let replacement_for_wake = junior1_claim.clone();
    let result = senior.run_granted(None, |current, _watch, acquisition_allowed, _, _| {
        anyhow::ensure!(
            !acquisition_allowed && current == &senior_designated_for_wake,
            "disjoint-entrant senior callback received the wrong publication",
        );
        Ok(GrantAttempt::<()> {
            acquired: None,
            preparation_claim: None,
            preparation_contention: None,
            next_claim: replacement_for_wake.clone(),
            contention: None,
        })
    })?;
    anyhow::ensure!(
        matches!(result, GrantResult::Requeued),
        "disjoint-entrant senior completion did not requeue",
    );
    let watermark_dirty_below_juniors = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        table.min_changed_ticket() < junior1.ticket && !table.changed_claims_saturated()
    };
    let mut junior1_callback_ran = false;
    let junior1_result =
        later_grant_probe(&mut junior1, &junior1_claim, &mut junior1_callback_ran)?;
    let overlapping_junior_parked = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        !junior1_callback_ran
            && matches!(junior1_result, GrantResult::LostGrant)
            && table
                .record(junior1.slot)?
                .is_some_and(|record| record.state == STATE_WAITING)
    };
    let junior2_claim_for_wake = junior2_claim.clone();
    let mut junior2_callback_ran = false;
    let junior2_result = junior2.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            junior2_callback_ran = true;
            anyhow::ensure!(
                acquisition_allowed && current == &junior2_claim_for_wake,
                "disjoint junior grant received the wrong publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: Some(()),
                preparation_claim: None,
                preparation_contention: None,
                next_claim: junior2_claim_for_wake.clone(),
                contention: None,
            })
        },
    )?;
    let held = match junior2_result {
        GrantResult::Acquired((), held) => Some(held),
        _ => None,
    };
    let disjoint_junior_committed_held = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        junior2_callback_ran
            && held.is_some()
            && table
                .record(junior2.slot)?
                .is_some_and(|record| record.state == STATE_HELD)
    };
    drop(held);
    junior2.finish(None)?;
    junior1.finish(None)?;
    senior.finish(None)?;
    coordinator.finish(None)?;
    Ok(DisjointEntrantProceedsOutcome {
        watermark_dirty_below_juniors,
        overlapping_junior_parked,
        disjoint_junior_committed_held,
    })
}

#[cfg(test)]
fn exercise_pending_replan_grant_race_case(
    offset: usize,
    complete_during_callback: bool,
) -> Result<(bool, bool, bool, u64)> {
    let coordinator_cpu = offset;
    let earlier_designated_cpu = offset + 1;
    let conflicting_cpu = offset + 2;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let earlier_designated = ClaimSet::new(
        std::iter::empty(),
        [earlier_designated_cpu],
        FlockMode::Exclusive,
    );
    let earlier_replacement =
        ClaimSet::new(std::iter::empty(), [conflicting_cpu], FlockMode::Exclusive);
    let earlier_watch = ClaimSet::new(
        std::iter::empty(),
        [earlier_designated_cpu, conflicting_cpu],
        FlockMode::Exclusive,
    );
    let mut earlier = Ticket::register(earlier_designated.clone(), earlier_watch, None)?;
    let later_claim = ClaimSet::new(std::iter::empty(), [conflicting_cpu], FlockMode::Exclusive);
    let mut later = Ticket::register(later_claim.clone(), later_claim.clone(), None)?;

    let scans_before = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, earlier_designated_cpu, false)?;
        set_cpu_free_for_tests(&mut table, conflicting_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table
                .record(earlier.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN)
                && table
                    .record(later.slot)?
                    .is_some_and(|record| record.state == STATE_GRANTED),
            "pending-edge race fixture did not publish REPLAN before its conflicting later grant",
        );
        read_u64(&table.header, H_GRANT_SCANS)
    };

    let mut callback_entered = false;
    let (later_result, earlier_completed) = if complete_during_callback {
        let mut earlier_completed = false;
        let later_result = later.run_granted(
            None,
            |current, _watch, acquisition_allowed, _predecessors, _availability| {
                callback_entered = true;
                anyhow::ensure!(
                    acquisition_allowed && current == &later_claim,
                    "pending-edge completion fixture received the wrong later grant",
                );
                let result = earlier.run_granted(
                    None,
                    |current, _watch, acquisition_allowed, _predecessors, _availability| {
                        anyhow::ensure!(
                            !acquisition_allowed && current == &earlier_designated,
                            "pending-edge completion fixture received the wrong earlier REPLAN",
                        );
                        Ok(GrantAttempt::<()> {
                            acquired: None,
                            preparation_claim: None,
                            preparation_contention: None,
                            next_claim: earlier_replacement.clone(),
                            contention: None,
                        })
                    },
                )?;
                earlier_completed = matches!(result, GrantResult::Requeued);
                Ok(GrantAttempt {
                    acquired: Some(()),
                    preparation_claim: None,
                    preparation_contention: None,
                    next_claim: current.clone(),
                    contention: None,
                })
            },
        )?;
        (later_result, earlier_completed)
    } else {
        let result = earlier.run_granted(
            None,
            |current, _watch, acquisition_allowed, _predecessors, _availability| {
                anyhow::ensure!(
                    !acquisition_allowed && current == &earlier_designated,
                    "pending-edge entry fixture received the wrong earlier REPLAN",
                );
                Ok(GrantAttempt::<()> {
                    acquired: None,
                    preparation_claim: None,
                    preparation_contention: None,
                    next_claim: earlier_replacement.clone(),
                    contention: None,
                })
            },
        )?;
        let earlier_completed = matches!(result, GrantResult::Requeued);
        let later_result = later.run_granted(
            None,
            |current, _watch, acquisition_allowed, _predecessors, _availability| {
                callback_entered = true;
                anyhow::ensure!(
                    acquisition_allowed && current == &later_claim,
                    "pending-edge entry fixture received the wrong later grant",
                );
                Ok(GrantAttempt {
                    acquired: Some(()),
                    preparation_claim: None,
                    preparation_contention: None,
                    next_claim: current.clone(),
                    contention: None,
                })
            },
        )?;
        (later_result, earlier_completed)
    };
    let rejected = match later_result {
        GrantResult::LostGrant => true,
        GrantResult::Acquired((), held) => {
            drop(held);
            false
        }
        GrantResult::Prepared((), _) | GrantResult::Requeued => false,
    };

    let (conflict_blocked_after_scan, scan_delta) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        // A rejected physical result publishes UNKNOWN until its opaque
        // payload closes. Model the coordinator's post-drop observation before
        // consuming the authoritative edge.
        set_cpu_free_for_tests(&mut table, conflicting_cpu, true)?;
        table.grant_compatible()?;
        let earlier_granted = table
            .record(earlier.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED);
        let later_blocked = table
            .record(later.slot)?
            .is_some_and(|record| record.state != STATE_GRANTED);
        (
            earlier_granted && later_blocked,
            read_u64(&table.header, H_GRANT_SCANS).wrapping_sub(scans_before),
        )
    };

    later.finish(None)?;
    earlier.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        callback_entered,
        earlier_completed && rejected,
        conflict_blocked_after_scan,
        scan_delta,
    ))
}

/// A completed non-fencing replacement dirties the suffix even though it does
/// not advance the public claim epoch. A later exact grant must therefore be
/// rejected both before callback entry and at the callback commit boundary.
#[cfg(test)]
pub(super) fn exercise_pending_replan_grant_races_for_tests()
-> Result<PendingReplanGrantRaceOutcome> {
    let entry = exercise_pending_replan_grant_race_case(1_300, false)?;
    let commit = exercise_pending_replan_grant_race_case(1_310, true)?;
    Ok(PendingReplanGrantRaceOutcome {
        entry_callback_suppressed: !entry.0 && entry.1,
        entry_conflict_blocked_after_scan: entry.2,
        entry_authoritative_scan_delta: entry.3,
        commit_callback_entered: commit.0,
        commit_rejected: commit.1,
        commit_conflict_blocked_after_scan: commit.2,
        commit_authoritative_scan_delta: commit.3,
    })
}

/// Complete the last live callback after its coordinator disappeared while a
/// RESCAN edge was already pending. Election must run after REPLAN -> WAITING
/// publication even though no new transport edge is needed.
#[cfg(test)]
pub(super) fn exercise_replan_completion_election_for_tests() -> Result<(bool, u64)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [1_250usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let designated = ClaimSet::new(std::iter::empty(), [1_251usize], FlockMode::Exclusive);
    let replacement = ClaimSet::new(std::iter::empty(), [1_252usize], FlockMode::Exclusive);
    let watch = ClaimSet::new(
        std::iter::empty(),
        [1_251usize, 1_252usize],
        FlockMode::Exclusive,
    );
    let mut callback = Ticket::register(designated.clone(), watch, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        for cpu in [1_250usize, 1_251, 1_252] {
            set_cpu_free_for_tests(&mut table, cpu, false)?;
        }
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table
                .record(callback.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN),
            "completion-election fixture did not publish REPLAN",
        );
    }
    coordinator.finish(None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        anyhow::ensure!(
            table.coordinator_ticket() == 0 && table.pending_flags() & PENDING_RESCAN != 0,
            "completion-election fixture retained a coordinator or lost its pending edge",
        );
    }
    let notify_before = NOTIFY_CALLS.with(std::cell::Cell::get);
    let result = callback.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &designated,
                "completion-election fixture received the wrong callback",
            );
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: replacement.clone(),
                contention: None,
            })
        },
    )?;
    let elected = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        matches!(result, GrantResult::Requeued)
            && table.coordinator_ticket() == callback.ticket
            && table.record(callback.slot)?.is_some_and(|record| {
                record.state == STATE_COORDINATOR && record.claim == replacement
            })
    };
    let notify_delta = NOTIFY_CALLS
        .with(std::cell::Cell::get)
        .wrapping_sub(notify_before);
    callback.finish(None)?;
    Ok((elected, notify_delta))
}

/// Drive a finite wave across its lease boundary without sleeping. One
/// callback publishes a completed replacement, one returns after the wave was
/// quarantined, and one has not entered yet. This covers both acknowledgement
/// paths while proving that expiration unblocks the completed work first.
#[cfg(test)]
pub(super) fn exercise_replan_wave_expiry_for_tests() -> Result<ReplanWaveExpiryOutcome> {
    let coordinator_cpu = 1_300usize;
    let designated_cpus = [1_301usize, 1_303, 1_305];
    let replacement_cpus = [1_302usize, 1_304, 1_306];
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let mut callbacks = Vec::new();
    for (&designated_cpu, &replacement_cpu) in designated_cpus.iter().zip(replacement_cpus.iter()) {
        let designated = ClaimSet::new(std::iter::empty(), [designated_cpu], FlockMode::Exclusive);
        let replacement =
            ClaimSet::new(std::iter::empty(), [replacement_cpu], FlockMode::Exclusive);
        let watch = ClaimSet::new(
            std::iter::empty(),
            [designated_cpu, replacement_cpu],
            FlockMode::Exclusive,
        );
        callbacks.push((
            Ticket::register(designated.clone(), watch, None)?,
            designated,
            replacement,
        ));
    }

    let wave_started_ns = monotonic_now_ns()?.max(1);
    let wave_deadline_ns = wave_started_ns.saturating_add(REPLAN_WAVE_LEASE_NS);
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        for cpu in designated_cpus.into_iter().chain(replacement_cpus) {
            set_cpu_free_for_tests(&mut table, cpu, false)?;
        }
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(wave_started_ns, None)?;
        anyhow::ensure!(
            table.replan_outstanding() == callbacks.len() as u64
                && table.replan_wave_started_ns() == wave_started_ns
                && table.replan_wave_deadline_ns() == wave_deadline_ns
                && callbacks.iter().all(|(ticket, _, _)| {
                    table
                        .record(ticket.slot)
                        .ok()
                        .flatten()
                        .is_some_and(|record| record.state == STATE_REPLAN)
                }),
            "expiry fixture did not publish one coherent finite REPLAN wave",
        );
    }

    let completed_designated = callbacks[0].1.clone();
    let completed_replacement = callbacks[0].2.clone();
    let completed = callbacks[0].0.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &completed_designated,
                "completed expiry callback received the wrong publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: completed_replacement.clone(),
                contention: None,
            })
        },
    )?;
    anyhow::ensure!(
        matches!(completed, GrantResult::Requeued),
        "completed expiry callback did not return to WAITING",
    );

    // Join one post-horizon waiter to the still-live wave. The extension must
    // share the original deadline: continuous arrivals may increase useful
    // planner parallelism, but cannot renew a stuck callback's lease.
    let incremental_designated_cpu = 1_307usize;
    let incremental_alternative_cpu = 1_308usize;
    let incremental_designated = ClaimSet::new(
        std::iter::empty(),
        [incremental_designated_cpu],
        FlockMode::Exclusive,
    );
    let incremental_watch = ClaimSet::new(
        std::iter::empty(),
        [incremental_designated_cpu, incremental_alternative_cpu],
        FlockMode::Exclusive,
    );
    let mut incremental_late = Ticket::register(incremental_designated, incremental_watch, None)?;
    let (incremental_late_published, incremental_horizon_extended, incremental_deadline_preserved) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, incremental_designated_cpu, false)?;
        set_cpu_free_for_tests(&mut table, incremental_alternative_cpu, false)?;
        let horizon_before = read_u64(&table.header, H_REPLAN_HORIZON);
        table.stamp_resource_improvement(S_CPU_EX, incremental_alternative_cpu)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(wave_started_ns.saturating_add(1), None)?;
        (
            table.replan_outstanding() == 3
                && table
                    .record(incremental_late.slot)?
                    .is_some_and(|record| record.state == STATE_REPLAN),
            horizon_before < incremental_late.ticket
                && read_u64(&table.header, H_REPLAN_HORIZON) >= incremental_late.ticket,
            table.replan_wave_started_ns() == wave_started_ns
                && table.replan_wave_deadline_ns() == wave_deadline_ns,
        )
    };

    let mut ordinary_wave_not_expired_early = false;
    let mut completed_replacement_preserved = false;
    let mut stragglers_quarantined = false;
    let mut wave_drained_and_clock_cleared = false;
    let mut expiration_avoids_global_generation_wake = false;
    let mut completed_replacement_granted = false;
    let mut expired_tickets_not_reissued = false;
    let mut generation_after_expiration_scan = 0u32;
    let completed_slot = callbacks[0].0.slot;
    let late_slot = callbacks[1].0.slot;
    let entry_slot = callbacks[2].0.slot;
    let straggler_slots = [late_slot, entry_slot, incremental_late.slot];
    let completed_replacement_for_check = callbacks[0].2.clone();
    let late_designated = callbacks[1].1.clone();
    let late_replacement = callbacks[1].2.clone();
    let late_completion = callbacks[1].0.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &late_designated,
                "late expiry callback received the wrong publication",
            );
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            let generation_before_expiration = table.generation_wake();
            let expired_early =
                table.expire_replan_wave_if_due_at(wave_deadline_ns.saturating_sub(1))?;
            ordinary_wave_not_expired_early = !expired_early
                && table.replan_outstanding() == 3
                && straggler_slots.iter().all(|slot| {
                    table
                        .record(*slot)
                        .ok()
                        .flatten()
                        .is_some_and(|record| record.state == STATE_REPLAN)
                });

            anyhow::ensure!(
                table.expire_replan_wave_if_due_at(wave_deadline_ns)?,
                "due REPLAN wave did not expire",
            );
            let completed_record = table
                .record(completed_slot)?
                .ok_or_else(|| anyhow::anyhow!("completed replacement disappeared"))?;
            completed_replacement_preserved = completed_record.state == STATE_WAITING
                && completed_record.claim == completed_replacement_for_check;
            stragglers_quarantined = straggler_slots.iter().all(|slot| {
                table
                    .record(*slot)
                    .ok()
                    .flatten()
                    .is_some_and(|record| record.state == STATE_REPLAN_EXPIRED)
            });
            wave_drained_and_clock_cleared = table.replan_outstanding() == 0
                && table.replan_wave_started_ns() == 0
                && table.replan_wave_deadline_ns() == 0
                && table.pending_flags() & PENDING_RESCAN != 0;
            expiration_avoids_global_generation_wake = table
                .generation_wake()
                .wrapping_sub(generation_before_expiration)
                == 0;

            set_cpu_free_for_tests(&mut table, replacement_cpus[0], true)?;
            table.grant_compatible_at(wave_deadline_ns, None)?;
            completed_replacement_granted = table.record(completed_slot)?.is_some_and(|record| {
                record.state == STATE_GRANTED && record.claim == completed_replacement_for_check
            });
            expired_tickets_not_reissued = straggler_slots.iter().all(|slot| {
                table
                    .record(*slot)
                    .ok()
                    .flatten()
                    .is_some_and(|record| record.state == STATE_REPLAN_EXPIRED)
            });
            generation_after_expiration_scan = table.generation_wake();
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: late_replacement.clone(),
                contention: None,
            })
        },
    )?;
    let late_completion_rejected = matches!(late_completion, GrantResult::LostGrant);

    let (late_replacement_rejected, completion_acknowledged_waiting) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(callbacks[1].0.slot)?
            .ok_or_else(|| anyhow::anyhow!("late callback ticket disappeared after ack"))?;
        (
            record.claim == callbacks[1].1 && record.claim != callbacks[1].2,
            record.state == STATE_WAITING && record.prefix_epoch == 0,
        )
    };

    let mut entry_callback_ran = false;
    let entry_result = callbacks[2].0.run_granted(
        None,
        |_current,
         _watch,
         _acquisition_allowed,
         _predecessors,
         _availability|
         -> Result<GrantAttempt<()>> {
            entry_callback_ran = true;
            anyhow::bail!("expired REPLAN callback entered stale user code")
        },
    )?;
    let entry_callback_suppressed =
        matches!(entry_result, GrantResult::LostGrant) && !entry_callback_ran;
    let (entry_acknowledged_waiting, acknowledgement_rescan_edge) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(callbacks[2].0.slot)?
            .ok_or_else(|| anyhow::anyhow!("unentered callback ticket disappeared after ack"))?;
        (
            record.state == STATE_WAITING && record.prefix_epoch == 0,
            table.pending_flags() & PENDING_RESCAN != 0
                && table
                    .generation_wake()
                    .wrapping_sub(generation_after_expiration_scan)
                    == 0,
        )
    };

    incremental_late.finish(None)?;
    for (ticket, _, _) in callbacks.iter_mut().rev() {
        ticket.finish(None)?;
    }
    coordinator.finish(None)?;
    Ok(ReplanWaveExpiryOutcome {
        incremental_late_published,
        incremental_horizon_extended,
        incremental_deadline_preserved,
        ordinary_wave_not_expired_early,
        completed_replacement_preserved,
        stragglers_quarantined,
        wave_drained_and_clock_cleared,
        expiration_avoids_global_generation_wake,
        completed_replacement_granted,
        expired_tickets_not_reissued,
        late_completion_rejected,
        late_replacement_rejected,
        completion_acknowledged_waiting,
        entry_callback_suppressed,
        entry_acknowledged_waiting,
        acknowledgement_rescan_edge,
    })
}

/// Model a writer dying after decrementing the final outstanding count but
/// before publishing the corresponding record state. The retained deadline
/// must let dirty repair recognize and quarantine the still-REPLAN record.
#[cfg(test)]
pub(super) fn exercise_replan_expiry_publication_crash_for_tests() -> Result<bool> {
    let coordinator_cpu = 1_310usize;
    let callback_cpu = 1_311usize;
    let alternative_cpu = 1_312usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let callback_claim = ClaimSet::new(std::iter::empty(), [callback_cpu], FlockMode::Exclusive);
    let callback_watch = ClaimSet::new(
        std::iter::empty(),
        [callback_cpu, alternative_cpu],
        FlockMode::Exclusive,
    );
    let mut callback = Ticket::register(callback_claim, callback_watch, None)?;

    let publication_torn_with_deadline = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, callback_cpu, false)?;
        set_cpu_free_for_tests(&mut table, alternative_cpu, false)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible_at(monotonic_now_ns()?.max(1), None)?;
        anyhow::ensure!(
            table
                .record(callback.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN)
                && table.replan_outstanding() == 1,
            "crash-order fixture did not publish its REPLAN callback",
        );

        table.begin_transaction()?;
        write_u64(&mut table.header, H_REPLAN_WAVE_STARTED_NS, 1);
        write_u64(&mut table.header, H_REPLAN_WAVE_DEADLINE_NS, 1);
        table.transition_replan_state(STATE_REPLAN, STATE_REPLAN_EXPIRED)?;
        table.replan_outstanding() == 0
            && table.replan_wave_deadline_ns() == 1
            && table
                .record(callback.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN)
            && atomic_u64(&table.header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0
    };

    let repaired_to_expired = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        table
            .record(callback.slot)?
            .is_some_and(|record| record.state == STATE_REPLAN_EXPIRED)
            && table.replan_outstanding() == 0
            && table.replan_wave_started_ns() == 0
            && table.replan_wave_deadline_ns() == 0
            && atomic_u64(&table.header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) == 0
    };

    callback.finish(None)?;
    coordinator.finish(None)?;
    Ok(publication_torn_with_deadline && repaired_to_expired)
}

#[cfg(test)]
pub(crate) struct GranularPrefixInvalidationOutcome {
    pub(crate) coordinator_preserved: bool,
    pub(crate) granted_preserved: bool,
    pub(crate) replan_preserved: bool,
    pub(crate) waiting_preserved: bool,
    pub(crate) granted_refreshed: bool,
    pub(crate) replan_refreshed: bool,
    pub(crate) waiting_replanned: bool,
    pub(crate) entry_unchanged_deferred: bool,
    pub(crate) entry_changed_deferred: bool,
    pub(crate) completion_unchanged_kept: bool,
    pub(crate) completion_changed_deferred: bool,
    pub(crate) coordinator_completion_unchanged_kept: bool,
    pub(crate) coordinator_completion_disjoint_change_kept: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CallbackPublicationToken {
    state: u32,
    grant_epoch: u64,
    replan_epoch: u64,
    prefix_epoch: u64,
    issue_serial: u64,
    blocked_on: Option<BlockedOn>,
}

#[cfg(test)]
fn callback_publication_token(record: &Record) -> CallbackPublicationToken {
    CallbackPublicationToken {
        state: record.state,
        grant_epoch: record.grant_epoch,
        replan_epoch: record.replan_claim_epoch,
        prefix_epoch: record.prefix_epoch,
        issue_serial: record.issue_serial,
        blocked_on: record.blocked_on,
    }
}

#[cfg(test)]
fn exercise_granular_prefix_invalidation_case(
    offset: usize,
    target_state: u32,
) -> Result<(bool, bool, bool)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [offset + 10], FlockMode::Exclusive);
    let mut coordinator =
        Ticket::register(coordinator_claim.clone(), coordinator_claim.clone(), None)?;
    let predecessor_watch = ClaimSet::with_modes(
        std::iter::empty(),
        [offset, offset + 4, offset + 5, offset + 40],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let predecessor_a = ClaimSet::with_modes(
        std::iter::empty(),
        [offset],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let predecessor_b = ClaimSet::with_modes(
        std::iter::empty(),
        [offset + 4],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let predecessor_c = ClaimSet::with_modes(
        std::iter::empty(),
        [offset + 5],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let mut changing = Ticket::register(predecessor_a.clone(), predecessor_watch, None)?;
    let mut duplicate_a = Ticket::register(predecessor_a.clone(), predecessor_a.clone(), None)?;
    let mut duplicate_b = Ticket::register(predecessor_b.clone(), predecessor_b.clone(), None)?;
    let target_claim = ClaimSet::new(std::iter::empty(), [offset + 20], FlockMode::Exclusive);
    let target_watch = if target_state == STATE_GRANTED {
        target_claim.clone()
    } else {
        ClaimSet::new(
            std::iter::empty(),
            [offset + 20, offset + 21],
            FlockMode::Exclusive,
        )
    };
    let mut target = Ticket::register(target_claim.clone(), target_watch, None)?;

    let (coordinator_before, target_before) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        // Registration is a real liveness path: a sufficiently descheduled
        // fixture may legitimately transfer the coordinator heartbeat lease
        // while the remaining synthetic tickets are being constructed.
        // Restage the complete synthetic image transactionally so overwriting
        // a promoted predecessor cannot leave the coordinator header pointing
        // at a non-coordinator record.
        table.begin_transaction()?;
        for ticket in [&changing, &duplicate_a, &duplicate_b] {
            table.set_record_state(ticket.slot, STATE_PENDING)?;
            table.clear_record_blocked(ticket.slot)?;
        }
        set_cpu_free_for_tests(&mut table, offset + 10, true)?;
        set_cpu_free_for_tests(&mut table, offset + 20, target_state == STATE_GRANTED)?;
        set_cpu_free_for_tests(&mut table, offset + 21, false)?;
        table.set_record_state(
            target.slot,
            if matches!(target_state, STATE_REPLAN | STATE_WAITING) {
                STATE_REPLAN
            } else {
                STATE_WAITING
            },
        )?;
        table.clear_record_blocked(target.slot)?;
        table.set_record_state(coordinator.slot, STATE_WAITING)?;
        table.clear_record_blocked(coordinator.slot)?;
        table.set_coordinator(0, NONE_SLOT)?;
        table.elect_coordinator_in_transaction()?;
        table.set_pending_flag(PENDING_RESCAN);
        table.finish_transaction()?;
        table.grant_compatible()?;
        if target_state == STATE_WAITING {
            // Seed a valid speculative predecessor publication, then model a
            // completed unchanged callback. REPLAN and WAITING are both
            // non-fencing, so preserving the cached prefix is exactly the
            // production fast path exercised by the following mutations.
            anyhow::ensure!(
                table
                    .record(target.slot)?
                    .is_some_and(|record| record.state == STATE_REPLAN),
                "granular invalidation WAITING target was not seeded through REPLAN",
            );
            table.set_record_state(target.slot, STATE_WAITING)?;
        }
        let prepared = table
            .record(target.slot)?
            .ok_or_else(|| anyhow::anyhow!("granular invalidation target disappeared"))?;
        let expected_prepared = target_state;
        anyhow::ensure!(
            prepared.state == expected_prepared,
            "granular invalidation target prepared as {}, expected {expected_prepared}",
            prepared.state,
        );
        if target_state != STATE_GRANTED {
            let blocker = ContentionMarker {
                blocker: ResourceKey::Cpu(offset + 40),
                mode: FlockMode::Exclusive,
            };
            let blocked_at = table.blocker_serial(blocker.blocker, blocker.mode)?;
            table.set_record_blocked(target.slot, blocker, blocked_at)?;
        }
        let coordinator = table
            .record(coordinator.slot)?
            .ok_or_else(|| anyhow::anyhow!("granular invalidation coordinator disappeared"))?;
        let target = table
            .record(target.slot)?
            .ok_or_else(|| anyhow::anyhow!("granular invalidation target disappeared"))?;
        (
            callback_publication_token(&coordinator),
            callback_publication_token(&target),
        )
    };

    let (coordinator_same, target_same) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let changing_record = table
            .record(changing.slot)?
            .ok_or_else(|| anyhow::anyhow!("changing predecessor disappeared"))?;
        let issue_serial = table.max_watch_serial(&changing_record.watch)?;
        table.replace_claim(
            changing.slot,
            changing.ticket,
            &predecessor_a,
            &predecessor_b,
            STATE_PENDING,
            issue_serial,
            None,
            false,
        )?;
        table.grant_compatible()?;
        let coordinator = table
            .record(coordinator.slot)?
            .ok_or_else(|| anyhow::anyhow!("granular invalidation coordinator disappeared"))?;
        let target = table
            .record(target.slot)?
            .ok_or_else(|| anyhow::anyhow!("granular invalidation target disappeared"))?;
        (
            callback_publication_token(&coordinator),
            callback_publication_token(&target),
        )
    };

    let (coordinator_changed, target_changed) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let changing_record = table
            .record(changing.slot)?
            .ok_or_else(|| anyhow::anyhow!("changing predecessor disappeared"))?;
        let issue_serial = table.max_watch_serial(&changing_record.watch)?;
        table.replace_claim(
            changing.slot,
            changing.ticket,
            &predecessor_b,
            &predecessor_c,
            STATE_PENDING,
            issue_serial,
            None,
            false,
        )?;
        table.grant_compatible()?;
        let coordinator = table
            .record(coordinator.slot)?
            .ok_or_else(|| anyhow::anyhow!("granular invalidation coordinator disappeared"))?;
        let target = table
            .record(target.slot)?
            .ok_or_else(|| anyhow::anyhow!("granular invalidation target disappeared"))?;
        (
            callback_publication_token(&coordinator),
            callback_publication_token(&target),
        )
    };

    let coordinator_preserved =
        coordinator_same == coordinator_before && coordinator_changed == coordinator_before;
    let same_prefix_preserved = target_same == target_before;
    let real_prefix_invalidated = match target_state {
        STATE_GRANTED | STATE_REPLAN => {
            target_changed.state == target_state && target_changed != target_same
        }
        STATE_WAITING => target_changed.state == STATE_REPLAN && target_changed != target_same,
        _ => anyhow::bail!("unsupported granular invalidation target state {target_state}"),
    };

    target.finish(None)?;
    duplicate_b.finish(None)?;
    duplicate_a.finish(None)?;
    changing.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        coordinator_preserved,
        same_prefix_preserved,
        real_prefix_invalidated,
    ))
}

#[cfg(test)]
fn exercise_callback_suffix_reconciliation_case(
    offset: usize,
    change_before_callback: bool,
    real_prefix_change: bool,
) -> Result<bool> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [offset + 10], FlockMode::Exclusive);
    let mut coordinator =
        Ticket::register(coordinator_claim.clone(), coordinator_claim.clone(), None)?;
    let predecessor_a = ClaimSet::with_modes(
        std::iter::empty(),
        [offset],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let predecessor_b = ClaimSet::with_modes(
        std::iter::empty(),
        [offset + 4],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let predecessor_c = ClaimSet::with_modes(
        std::iter::empty(),
        [offset + 5],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let predecessor_watch = ClaimSet::with_modes(
        std::iter::empty(),
        [offset, offset + 4, offset + 5],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let mut changing = Ticket::register(predecessor_a.clone(), predecessor_watch, None)?;
    let mut duplicate_a = Ticket::register(predecessor_a.clone(), predecessor_a.clone(), None)?;
    let mut duplicate_b = Ticket::register(predecessor_b.clone(), predecessor_b.clone(), None)?;
    let designated = ClaimSet::new(std::iter::empty(), [offset + 20], FlockMode::Exclusive);
    let watch = ClaimSet::new(
        std::iter::empty(),
        [offset + 20, offset + 21],
        FlockMode::Exclusive,
    );
    let mut target = Ticket::register(designated.clone(), watch, None)?;

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        // As above, ticket construction may outlive the real coordinator
        // lease on a descheduled runner. Rebuild the intended synthetic
        // coordinator and predecessor states as one coherent publication.
        table.begin_transaction()?;
        for ticket in [&changing, &duplicate_a, &duplicate_b] {
            table.set_record_state(ticket.slot, STATE_PENDING)?;
            table.clear_record_blocked(ticket.slot)?;
        }
        set_cpu_free_for_tests(&mut table, offset + 10, true)?;
        set_cpu_free_for_tests(&mut table, offset + 20, false)?;
        set_cpu_free_for_tests(&mut table, offset + 21, false)?;
        table.set_record_state(target.slot, STATE_REPLAN)?;
        table.clear_record_blocked(target.slot)?;
        table.set_record_state(coordinator.slot, STATE_WAITING)?;
        table.clear_record_blocked(coordinator.slot)?;
        table.set_coordinator(0, NONE_SLOT)?;
        table.elect_coordinator_in_transaction()?;
        table.set_pending_flag(PENDING_RESCAN);
        table.finish_transaction()?;
        table.grant_compatible()?;
        anyhow::ensure!(
            table
                .record(target.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN),
            "callback reconciliation target did not start in REPLAN",
        );
    }

    let replacement = if real_prefix_change {
        predecessor_c.clone()
    } else {
        predecessor_b.clone()
    };
    let publish_change = || -> Result<()> {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(changing.slot)?
            .ok_or_else(|| anyhow::anyhow!("callback changing predecessor disappeared"))?;
        let issue_serial = table.max_watch_serial(&record.watch)?;
        table.replace_claim(
            changing.slot,
            changing.ticket,
            &predecessor_a,
            &replacement,
            STATE_PENDING,
            issue_serial,
            None,
            false,
        )
    };
    if change_before_callback {
        publish_change()?;
    }

    let changed_candidate = ClaimSet::new(std::iter::empty(), [offset + 5], FlockMode::Exclusive);
    let mut callback_ran = false;
    let mut callback_saw_changed_prefix = false;
    let result = target.run_granted(
        None,
        |current, _watch, acquisition_allowed, predecessors, _availability| {
            callback_ran = true;
            anyhow::ensure!(
                !acquisition_allowed && current == &designated,
                "callback reconciliation received the wrong REPLAN publication",
            );
            callback_saw_changed_prefix = predecessors.conflicts(&changed_candidate)?;
            if !change_before_callback {
                publish_change()?;
            }
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: current.clone(),
                contention: None,
            })
        },
    )?;
    // REPLAN is a coherent non-acquiring snapshot. An older prefix may choose
    // one WAITING replacement exactly once; the deferred authoritative scan,
    // not callback discard/retry, reconciles either aggregate-equivalent or
    // genuinely changed predecessor input.
    let target_waiting = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table
            .record(target.slot)?
            .is_some_and(|record| record.ticket == target.ticket && record.state == STATE_WAITING)
    };
    let expected = callback_ran
        && !callback_saw_changed_prefix
        && matches!(result, GrantResult::Requeued)
        && target_waiting;

    target.finish(None)?;
    duplicate_b.finish(None)?;
    duplicate_a.finish(None)?;
    changing.finish(None)?;
    coordinator.finish(None)?;
    Ok(expected)
}

#[cfg(test)]
fn exercise_coordinator_suffix_reconciliation_case(
    offset: usize,
    real_prefix_change: bool,
) -> Result<bool> {
    let predecessor_a = ClaimSet::with_modes(
        std::iter::empty(),
        [offset],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let predecessor_b = ClaimSet::with_modes(
        std::iter::empty(),
        [offset + 1],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let predecessor_c = ClaimSet::with_modes(
        std::iter::empty(),
        [offset + 2],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let predecessor_watch = ClaimSet::with_modes(
        std::iter::empty(),
        [offset, offset + 1, offset + 2],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let mut changing = Ticket::register(predecessor_a.clone(), predecessor_watch, None)?;
    let mut duplicate_a = Ticket::register(predecessor_a.clone(), predecessor_a.clone(), None)?;
    let mut duplicate_b = Ticket::register(predecessor_b.clone(), predecessor_b.clone(), None)?;
    let target_claim = ClaimSet::new(std::iter::empty(), [offset + 10], FlockMode::Exclusive);
    let mut target = Ticket::register(target_claim.clone(), target_claim.clone(), None)?;

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.begin_transaction()?;
        for predecessor in [&changing, &duplicate_a, &duplicate_b] {
            table.set_record_state(predecessor.slot, STATE_PENDING)?;
            table.clear_record_blocked(predecessor.slot)?;
        }
        table.set_record_state(target.slot, STATE_WAITING)?;
        table.clear_record_blocked(target.slot)?;
        table.set_coordinator(0, NONE_SLOT)?;
        table.elect_coordinator_in_transaction()?;
        table.set_pending_flag(PENDING_RESCAN);
        table.finish_transaction()?;
        table.grant_compatible()?;
        anyhow::ensure!(
            table.record(target.slot)?.is_some_and(|record| {
                record.state == STATE_COORDINATOR && record.prefix_epoch != 0
            }),
            "granular coordinator target did not receive a coherent prefix",
        );
    }
    let token = target.commit_token_for_tests()?;
    let replacement = if real_prefix_change {
        &predecessor_c
    } else {
        &predecessor_b
    };
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(changing.slot)?
            .ok_or_else(|| anyhow::anyhow!("coordinator predecessor disappeared"))?;
        let issue_serial = table.max_watch_serial(&record.watch)?;
        table.replace_claim(
            changing.slot,
            changing.ticket,
            &predecessor_a,
            replacement,
            STATE_PENDING,
            issue_serial,
            None,
            false,
        )?;
    }

    let result = target.finish_acquired(&target_claim, token, &[], None)?;
    let kept = match result {
        FinishAcquireResult::Committed(held) => {
            drop(held);
            true
        }
        FinishAcquireResult::Stale => {
            target.finish(None)?;
            false
        }
    };
    duplicate_b.finish(None)?;
    duplicate_a.finish(None)?;
    changing.finish(None)?;
    Ok(kept)
}

#[cfg(test)]
fn exercise_coordinator_pending_replan_case(
    offset: usize,
    preparation_commit: bool,
    conflicting: bool,
) -> Result<(bool, u64)> {
    let coordinator_cpu = offset;
    let earlier_designated_cpu = offset + 1;
    let disjoint_replacement_cpu = offset + 2;
    let selected_final_cpu = offset + 3;
    let preparation_cpu = offset + 4;
    let replacement_cpu = if conflicting {
        if preparation_commit {
            preparation_cpu
        } else {
            selected_final_cpu
        }
    } else {
        disjoint_replacement_cpu
    };

    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let earlier_designated = ClaimSet::new(
        std::iter::empty(),
        [earlier_designated_cpu],
        FlockMode::Exclusive,
    );
    let earlier_replacement =
        ClaimSet::new(std::iter::empty(), [replacement_cpu], FlockMode::Exclusive);
    let earlier_watch = ClaimSet::new(
        std::iter::empty(),
        [earlier_designated_cpu, replacement_cpu],
        FlockMode::Exclusive,
    );
    let mut earlier = Ticket::register(earlier_designated.clone(), earlier_watch, None)?;
    let selected_final = ClaimSet::new(
        std::iter::empty(),
        [selected_final_cpu],
        FlockMode::Exclusive,
    );
    let mut target = Ticket::register(selected_final.clone(), selected_final.clone(), None)?;
    let preparation = ClaimSet::new(std::iter::empty(), [preparation_cpu], FlockMode::Exclusive);

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, earlier_designated_cpu, false)?;
        set_cpu_free_for_tests(&mut table, selected_final_cpu, false)?;
        set_cpu_free_for_tests(&mut table, preparation_cpu, true)?;
        set_cpu_free_for_tests(&mut table, disjoint_replacement_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table
                .record(earlier.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN)
                && table
                    .record(target.slot)?
                    .is_some_and(|record| record.state == STATE_WAITING),
            "coordinator pending-edge fixture did not stage REPLAN before a waiting successor",
        );
    }
    coordinator.finish(None)?;

    let scans_before = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, selected_final_cpu, true)?;
        set_cpu_free_for_tests(&mut table, preparation_cpu, true)?;
        set_cpu_free_for_tests(&mut table, replacement_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table.record(target.slot)?.is_some_and(|record| {
                record.state == STATE_COORDINATOR && record.prefix_epoch != 0
            }) && table
                .record(earlier.slot)?
                .is_some_and(|record| record.state == STATE_REPLAN),
            "coordinator pending-edge fixture did not publish a coherent commit token",
        );
        read_u64(&table.header, H_GRANT_SCANS)
    };
    let token = target.commit_token_for_tests()?;

    let completion = earlier.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                !acquisition_allowed && current == &earlier_designated,
                "coordinator pending-edge fixture received the wrong REPLAN publication",
            );
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: earlier_replacement.clone(),
                contention: None,
            })
        },
    )?;
    anyhow::ensure!(
        matches!(completion, GrantResult::Requeued),
        "coordinator pending-edge fixture did not publish its replacement",
    );

    let matched = if preparation_commit {
        match target.finish_preparation(&selected_final, &preparation, token, &[], None)? {
            FinishPreparationResult::Committed(_) if !conflicting => true,
            FinishPreparationResult::Stale if conflicting => true,
            FinishPreparationResult::Committed(_) | FinishPreparationResult::Stale => false,
        }
    } else {
        match target.finish_acquired(&selected_final, token, &[], None)? {
            FinishAcquireResult::Committed(held) if !conflicting => {
                drop(held);
                true
            }
            FinishAcquireResult::Stale if conflicting => true,
            FinishAcquireResult::Committed(held) => {
                drop(held);
                false
            }
            FinishAcquireResult::Stale => false,
        }
    };
    let scan_delta = diagnostic_counter_for_tests(H_GRANT_SCANS)?.wrapping_sub(scans_before);

    target.finish(None)?;
    earlier.finish(None)?;
    Ok((matched, scan_delta))
}

#[cfg(test)]
pub(super) fn exercise_coordinator_pending_replan_for_tests()
-> Result<CoordinatorPendingReplanOutcome> {
    let acquisition_conflict = exercise_coordinator_pending_replan_case(1_400, false, true)?;
    let acquisition_disjoint = exercise_coordinator_pending_replan_case(1_410, false, false)?;
    let preparation_conflict = exercise_coordinator_pending_replan_case(1_420, true, true)?;
    let preparation_disjoint = exercise_coordinator_pending_replan_case(1_430, true, false)?;
    Ok(CoordinatorPendingReplanOutcome {
        acquisition_conflict_rejected: acquisition_conflict.0,
        acquisition_disjoint_preserved: acquisition_disjoint.0,
        preparation_conflict_rejected: preparation_conflict.0,
        preparation_disjoint_preserved: preparation_disjoint.0,
        acquisition_conflict_scan_delta: acquisition_conflict.1,
        acquisition_disjoint_scan_delta: acquisition_disjoint.1,
        preparation_conflict_scan_delta: preparation_conflict.1,
        preparation_disjoint_scan_delta: preparation_disjoint.1,
    })
}

#[cfg(test)]
pub(super) fn exercise_granular_prefix_invalidation_for_tests()
-> Result<GranularPrefixInvalidationOutcome> {
    let granted = exercise_granular_prefix_invalidation_case(100, STATE_GRANTED)?;
    let replan = exercise_granular_prefix_invalidation_case(200, STATE_REPLAN)?;
    let waiting = exercise_granular_prefix_invalidation_case(300, STATE_WAITING)?;
    let entry_unchanged = exercise_callback_suffix_reconciliation_case(400, true, false)?;
    let entry_changed = exercise_callback_suffix_reconciliation_case(500, true, true)?;
    let completion_unchanged = exercise_callback_suffix_reconciliation_case(600, false, false)?;
    let completion_changed = exercise_callback_suffix_reconciliation_case(700, false, true)?;
    let coordinator_completion_unchanged =
        exercise_coordinator_suffix_reconciliation_case(750, false)?;
    let coordinator_completion_changed =
        exercise_coordinator_suffix_reconciliation_case(770, true)?;
    Ok(GranularPrefixInvalidationOutcome {
        coordinator_preserved: granted.0 && replan.0 && waiting.0,
        granted_preserved: granted.1,
        replan_preserved: replan.1,
        waiting_preserved: waiting.1,
        granted_refreshed: granted.2,
        replan_refreshed: replan.2,
        waiting_replanned: waiting.2,
        entry_unchanged_deferred: entry_unchanged,
        entry_changed_deferred: entry_changed,
        completion_unchanged_kept: completion_unchanged,
        completion_changed_deferred: completion_changed,
        coordinator_completion_unchanged_kept: coordinator_completion_unchanged,
        coordinator_completion_disjoint_change_kept: coordinator_completion_changed,
    })
}

#[cfg(test)]
pub(super) fn exercise_granted_serial_scope_for_tests() -> Result<(bool, bool, bool, bool)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [810usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let designated = ClaimSet::new(std::iter::empty(), [800usize], FlockMode::Exclusive);
    let watch = ClaimSet::new(
        std::iter::empty(),
        [800usize, 801usize],
        FlockMode::Exclusive,
    );
    let mut target = Ticket::register(designated.clone(), watch, None)?;
    let before = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, 800, true)?;
        set_cpu_free_for_tests(&mut table, 801, false)?;
        table.set_record_state(target.slot, STATE_WAITING)?;
        table.clear_record_blocked(target.slot)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let record = table
            .record(target.slot)?
            .filter(|record| record.state == STATE_GRANTED)
            .ok_or_else(|| anyhow::anyhow!("serial-scope target was not granted"))?;
        callback_publication_token(&record)
    };

    let after_alternatives = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        for _ in 0..32 {
            table.stamp_resource_improvement(S_CPU_EX, 801)?;
            table.set_pending_flag(PENDING_RESCAN);
            table.grant_compatible()?;
        }
        callback_publication_token(
            &table
                .record(target.slot)?
                .ok_or_else(|| anyhow::anyhow!("serial-scope target disappeared"))?,
        )
    };
    let after_designated = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.stamp_resource_improvement(S_CPU_EX, 800)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        callback_publication_token(
            &table
                .record(target.slot)?
                .ok_or_else(|| anyhow::anyhow!("serial-scope target disappeared"))?,
        )
    };
    let result = target.run_granted(
        None,
        |current, _watch, allowed, _predecessors, _availability| {
            anyhow::ensure!(
                allowed && current == &designated,
                "serial-scope grant changed"
            );
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            table.stamp_resource_improvement(S_CPU_EX, 801)?;
            table.set_pending_flag(PENDING_RESCAN);
            table.grant_compatible()?;
            Ok(GrantAttempt {
                acquired: Some(()),
                preparation_claim: None,
                preparation_contention: None,
                next_claim: current.clone(),
                contention: None,
            })
        },
    )?;
    let physical_commit_kept = match result {
        GrantResult::Acquired((), held) => {
            drop(held);
            true
        }
        _ => false,
    };
    coordinator.finish(None)?;

    let herd_coordinator_claim =
        ClaimSet::new(std::iter::empty(), [950usize], FlockMode::Exclusive);
    let mut herd_coordinator =
        Ticket::register(herd_coordinator_claim.clone(), herd_coordinator_claim, None)?;
    let alternative = 940usize;
    let mut herd = Vec::new();
    for cpu in 900usize..912 {
        let claim = ClaimSet::new(std::iter::empty(), [cpu], FlockMode::Exclusive);
        let watch = ClaimSet::new(std::iter::empty(), [cpu, alternative], FlockMode::Exclusive);
        herd.push(Ticket::register(claim, watch, None)?);
    }
    let before_herd = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        for (index, ticket) in herd.iter().enumerate() {
            set_cpu_free_for_tests(&mut table, 900 + index, index.is_multiple_of(2))?;
            table.set_record_state(ticket.slot, STATE_WAITING)?;
            table.clear_record_blocked(ticket.slot)?;
        }
        set_cpu_free_for_tests(&mut table, alternative, false)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        herd.iter()
            .enumerate()
            .map(|(index, ticket)| {
                let expected = if index.is_multiple_of(2) {
                    STATE_GRANTED
                } else {
                    STATE_REPLAN
                };
                let record = table
                    .record(ticket.slot)?
                    .filter(|record| record.state == expected)
                    .ok_or_else(|| {
                        anyhow::anyhow!("serial-scope herd member had the wrong callback state")
                    })?;
                Ok(callback_publication_token(&record))
            })
            .collect::<Result<Vec<_>>>()?
    };
    let after_herd = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        for _ in 0..16 {
            table.stamp_resource_improvement(S_CPU_EX, alternative)?;
            table.set_pending_flag(PENDING_RESCAN);
            table.grant_compatible()?;
        }
        herd.iter()
            .map(|ticket| {
                table
                    .record(ticket.slot)?
                    .map(|record| callback_publication_token(&record))
                    .ok_or_else(|| anyhow::anyhow!("serial-scope herd member disappeared"))
            })
            .collect::<Result<Vec<_>>>()?
    };
    let herd_preserved = before_herd == after_herd;
    for ticket in herd.iter_mut().rev() {
        ticket.finish(None)?;
    }
    herd_coordinator.finish(None)?;

    Ok((
        before == after_alternatives,
        after_alternatives == after_designated,
        physical_commit_kept,
        herd_preserved,
    ))
}

#[cfg(test)]
pub(crate) struct RevocationAckOutcome {
    pub(crate) before_entry_acked: bool,
    pub(crate) during_callback_acked: bool,
    pub(crate) later_publication_preserved: bool,
    pub(crate) successor_rescan_published: bool,
    pub(crate) flexible_replanned_without_serial_churn: bool,
}

#[cfg(test)]
fn exercise_revocation_ack_case(
    offset: usize,
    during_callback: bool,
) -> Result<(bool, bool, bool)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [offset], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let claim = ClaimSet::new(std::iter::empty(), [offset + 1], FlockMode::Exclusive);
    let mut granted = Ticket::register(claim.clone(), claim.clone(), None)?;
    let later_watch = ClaimSet::new(
        std::iter::empty(),
        [offset + 1, offset + 2],
        FlockMode::Exclusive,
    );
    let mut later = Ticket::register(claim.clone(), later_watch, None)?;
    let granted_slot = granted.slot;
    let later_slot = later.slot;
    let later_before = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, offset, true)?;
        set_cpu_free_for_tests(&mut table, offset + 1, true)?;
        set_cpu_free_for_tests(&mut table, offset + 2, false)?;
        table.set_record_state(granted.slot, STATE_WAITING)?;
        table.clear_record_blocked(granted.slot)?;
        table.set_record_state(later.slot, STATE_REPLAN)?;
        table.clear_record_blocked(later.slot)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table
                .record(granted_slot)?
                .is_some_and(|record| record.state == STATE_GRANTED),
            "revocation target was not granted",
        );
        let record = table
            .record(later_slot)?
            .filter(|record| record.state == STATE_REPLAN)
            .ok_or_else(|| anyhow::anyhow!("revocation successor was not in REPLAN"))?;
        callback_publication_token(&record)
    };

    let revoke = || -> Result<bool> {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, offset + 1, false)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let target_revoked = table
            .record(granted_slot)?
            .is_some_and(|record| record.state == STATE_REVOKED);
        let later_after = table
            .record(later_slot)?
            .map(|record| callback_publication_token(&record))
            .ok_or_else(|| anyhow::anyhow!("revocation successor disappeared"))?;
        Ok(target_revoked && later_after == later_before)
    };

    let mut later_preserved = true;
    if !during_callback {
        later_preserved = revoke()?;
    }
    let result = granted.run_granted(
        None,
        |current, _watch, allowed, _predecessors, _availability| {
            anyhow::ensure!(
                allowed && current == &claim,
                "revoked callback changed grant"
            );
            later_preserved &= revoke()?;
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: current.clone(),
                contention: None,
            })
        },
    )?;
    let (acked, rescan) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let acked = table
            .record(granted_slot)?
            .is_some_and(|record| record.state == STATE_WAITING);
        (acked, table.pending_flags() & PENDING_RESCAN != 0)
    };

    later.finish(None)?;
    granted.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        matches!(result, GrantResult::LostGrant) && acked,
        later_preserved,
        rescan,
    ))
}

#[cfg(test)]
fn exercise_flexible_revocation_replan_case(offset: usize) -> Result<bool> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [offset], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let exact = ClaimSet::new(std::iter::empty(), [offset + 1], FlockMode::Exclusive);
    let watch = ClaimSet::new(
        std::iter::empty(),
        [offset + 1, offset + 2],
        FlockMode::Exclusive,
    );
    let mut target = Ticket::register(exact, watch, None)?;

    let (issue_serial, global_serial) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, offset, true)?;
        set_cpu_free_for_tests(&mut table, offset + 1, true)?;
        set_cpu_free_for_tests(&mut table, offset + 2, true)?;
        table.set_record_state(target.slot, STATE_WAITING)?;
        table.clear_record_blocked(target.slot)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let issue_serial = table
            .record(target.slot)?
            .filter(|record| record.state == STATE_GRANTED)
            .ok_or_else(|| anyhow::anyhow!("flexible revocation target was not granted"))?
            .issue_serial;
        let global_serial = table.global_serial();

        // This is a deterioration of only the exact designation. It must not
        // synthesize the improvement serial that the regression is meant to
        // avoid relying on; the already-free alternative remains unchanged.
        set_cpu_free_for_tests(&mut table, offset + 1, false)?;
        anyhow::ensure!(
            table.global_serial() == global_serial,
            "making the exact designation unavailable changed the global improvement serial",
        );
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table
                .record(target.slot)?
                .is_some_and(|record| record.state == STATE_REVOKED),
            "unavailable exact designation did not revoke its flexible grant",
        );
        anyhow::ensure!(
            table.global_serial() == global_serial,
            "revoking the exact designation changed the global improvement serial",
        );
        (issue_serial, global_serial)
    };

    anyhow::ensure!(
        target.state(None)? == State::Waiting,
        "flexible revoked target did not acknowledge to WAITING",
    );
    let replanned_without_serial_churn = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let acknowledged = table
            .record(target.slot)?
            .filter(|record| record.state == STATE_WAITING)
            .ok_or_else(|| anyhow::anyhow!("flexible revoked target disappeared after ACK"))?;
        anyhow::ensure!(
            acknowledged.prefix_epoch == 0,
            "flexible revoked target retained its stale GRANTED prefix after ACK",
        );
        anyhow::ensure!(
            acknowledged.issue_serial == issue_serial
                && table.global_serial() == global_serial
                && !table.bitmap_bit(B_CPU_EX_AVAILABLE, offset + 1)?
                && table.bitmap_bit(B_CPU_EX_AVAILABLE, offset + 2)?,
            "flexible revoked target's no-churn test inputs changed before rescan",
        );
        let wake_before = target
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("flexible revoked target wake mapping disappeared"))?
            .expected();
        table.grant_compatible()?;
        let wake_after = target
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("flexible revoked target wake mapping disappeared"))?
            .expected();
        table.record(target.slot)?.is_some_and(|record| {
            record.state == STATE_REPLAN && record.issue_serial == issue_serial
        }) && table.global_serial() == global_serial
            && wake_after == wake_before.wrapping_add(1)
    };

    target.finish(None)?;
    coordinator.finish(None)?;
    Ok(replanned_without_serial_churn)
}

#[cfg(test)]
pub(super) fn exercise_revocation_ack_for_tests() -> Result<RevocationAckOutcome> {
    let before = exercise_revocation_ack_case(1000, false)?;
    let during = exercise_revocation_ack_case(1010, true)?;
    let flexible_replanned_without_serial_churn = exercise_flexible_revocation_replan_case(1020)?;
    Ok(RevocationAckOutcome {
        before_entry_acked: before.0,
        during_callback_acked: during.0,
        later_publication_preserved: before.1 && during.1,
        successor_rescan_published: before.2 && during.2,
        flexible_replanned_without_serial_churn,
    })
}

#[cfg(test)]
pub(super) fn exercise_revoked_owner_death_for_tests() -> Result<(bool, bool, bool, bool)> {
    let coordinator_cpu = 1030usize;
    let contested_cpu = 1031usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let contested_claim = ClaimSet::new(std::iter::empty(), [contested_cpu], FlockMode::Exclusive);
    let mut revoked = Some(Ticket::register(
        contested_claim.clone(),
        contested_claim.clone(),
        None,
    )?);
    let mut successor = Ticket::register(contested_claim.clone(), contested_claim.clone(), None)?;
    let revoked_slot = revoked.as_ref().expect("live revocation target").slot;
    let revoked_ticket = revoked.as_ref().expect("live revocation target").ticket;

    let (live_fence_preserved, successor_wake_before) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, contested_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table
                .record(revoked_slot)?
                .is_some_and(|record| record.state == STATE_GRANTED),
            "revoked-owner test target was not initially granted",
        );
        set_cpu_free_for_tests(&mut table, contested_cpu, false)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table
                .record(revoked_slot)?
                .is_some_and(|record| record.state == STATE_REVOKED),
            "revoked-owner test target was not revoked",
        );
        set_cpu_free_for_tests(&mut table, contested_cpu, true)?;
        table.prune_dead_identities(&[(revoked_slot, revoked_ticket)])?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let live_fence_preserved = table
            .record(revoked_slot)?
            .is_some_and(|record| record.state == STATE_REVOKED)
            && table
                .record(successor.slot)?
                .is_some_and(|record| record.state == STATE_WAITING);
        let wake = successor
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("successor wake mapping disappeared"))?
            .expected();
        (live_fence_preserved, wake)
    };

    revoked
        .take()
        .expect("live revocation target")
        .abandon_for_tests();
    let (dead_record_removed, successor_granted, successor_woken) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.prune_dead_identities(&[(revoked_slot, revoked_ticket)])?;
        table.grant_compatible()?;
        let wake_after = successor
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("successor wake mapping disappeared"))?
            .expected();
        (
            table.record(revoked_slot)?.is_none(),
            table
                .record(successor.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED),
            wake_after != successor_wake_before,
        )
    };

    successor.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        live_fence_preserved,
        dead_record_removed,
        successor_granted,
        successor_woken,
    ))
}

#[cfg(test)]
pub(super) fn exercise_revoke_crash_repair_for_tests() -> Result<(bool, bool, bool, bool)> {
    let coordinator_cpu = 1040usize;
    let contested_cpu = 1041usize;
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [coordinator_cpu], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let contested_claim = ClaimSet::new(std::iter::empty(), [contested_cpu], FlockMode::Exclusive);
    let mut revoked = Ticket::register(contested_claim.clone(), contested_claim.clone(), None)?;
    let mut successor = Ticket::register(contested_claim.clone(), contested_claim.clone(), None)?;

    let (repair_preserved_revoked, repair_woke_owner) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, coordinator_cpu, false)?;
        set_cpu_free_for_tests(&mut table, contested_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        anyhow::ensure!(
            table
                .record(revoked.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED),
            "revoke-crash target was not initially granted",
        );
        let wake_before = revoked
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("revoked wake mapping disappeared"))?
            .expected();
        table.begin_transaction()?;
        table.set_record_state(revoked.slot, STATE_REVOKED)?;
        table.clear_record_blocked(revoked.slot)?;
        crash_at_for_tests("revoke_state_before_wake");
        table.repair_consistency_if_needed()?;
        let wake_after = revoked
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("revoked wake mapping disappeared"))?
            .expected();
        (
            table
                .record(revoked.slot)?
                .is_some_and(|record| record.state == STATE_REVOKED)
                && table
                    .record(successor.slot)?
                    .is_some_and(|record| record.state == STATE_WAITING),
            wake_after != wake_before,
        )
    };

    let mut callback_ran = false;
    let revoked_result: GrantResult<()> = revoked.run_granted(
        None,
        |_designated, _watch, _allowed, _predecessors, _availability| {
            callback_ran = true;
            anyhow::bail!("a repaired REVOKED owner entered its stale callback")
        },
    )?;
    let acked_without_callback = !callback_ran && matches!(revoked_result, GrantResult::LostGrant);
    revoked.finish(None)?;

    let successor_progressed = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        // Dirty repair deliberately invalidates every watched availability
        // proof. Model the coordinator's post-repair observation before
        // requiring the successor grant.
        set_cpu_free_for_tests(&mut table, contested_cpu, true)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        table
            .record(successor.slot)?
            .is_some_and(|record| record.state == STATE_GRANTED)
    };
    successor.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        repair_preserved_revoked,
        repair_woke_owner,
        acked_without_callback,
        successor_progressed,
    ))
}

/// Reproduce the small-queue state seen in the CI artifacts: one coordinator
/// has already consumed all prior work, then an exact ticket registers as
/// WAITING before any scan has run. A free physical observation must durably
/// request the scan that grants and wakes that ticket.
#[cfg(test)]
pub(super) fn exercise_waiting_release_wake_for_tests() -> Result<(bool, bool, bool, bool)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [40usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, 40, false)?;
        table.refresh_pending_observation_flag()?;
        write_u64(&mut table.header, H_PENDING_FLAGS, 0);
        table.finish_claim_scan();
    }
    let waiting_claim = ClaimSet::new(std::iter::empty(), [41usize], FlockMode::Exclusive);
    let mut waiting = Ticket::register(waiting_claim.clone(), waiting_claim, None)?;

    let (waiting_without_scan, observation_scheduled, granted, futex_woken) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let scans_before = read_u64(&table.header, H_GRANT_SCANS);
        let wake_before = waiting
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("WAITING wake mapping disappeared"))?
            .expected();
        let waiting_without_scan = scans_before == 0
            && table.min_changed_ticket() == u64::MAX
            && table.pending_flags() & PENDING_RESCAN == 0
            && table
                .record(waiting.slot)?
                .is_some_and(|record| record.state == STATE_WAITING);

        let request = table
            .observation_request()?
            .ok_or_else(|| anyhow::anyhow!("WAITING registration requested no observation"))?;
        let mut observation = AvailabilityObservation::default();
        observation.cpus.insert(
            41,
            CpuObservation {
                availability: CpuAvailability::Free,
                sh_resolved: true,
                ex_resolved: true,
            },
        );
        let improved = table.apply_observation(&request, &observation)?;
        let observation_scheduled = improved && table.pending_flags() & PENDING_RESCAN != 0;
        table.grant_compatible()?;
        let wake_after = waiting
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("WAITING wake mapping disappeared"))?
            .expected();
        (
            waiting_without_scan,
            observation_scheduled,
            table
                .record(waiting.slot)?
                .is_some_and(|record| record.state == STATE_GRANTED),
            wake_after != wake_before,
        )
    };

    waiting.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        waiting_without_scan,
        observation_scheduled,
        granted,
        futex_woken,
    ))
}

#[cfg(test)]
pub(super) fn exercise_one_shot_replacement_for_tests()
-> Result<(usize, bool, bool, bool, bool, usize)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let mut coordinator =
        Ticket::register(coordinator_claim.clone(), coordinator_claim.clone(), None)?;
    let next_claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let watch = ClaimSet::new(std::iter::empty(), [0usize, 1usize], FlockMode::Exclusive);
    let mut waiter = Ticket::register(coordinator_claim.clone(), watch, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.grant_compatible()?;
    }
    if waiter.state_without_recovery_for_tests()? != State::Replan {
        anyhow::bail!("one-shot replacement waiter did not receive the REPLAN token");
    }

    let active_reads_before = ACTIVE_LIST_RECORD_READS.with(std::cell::Cell::get);
    let mut callbacks = 0usize;
    let mut acquisition_allowed = true;
    let result = waiter.run_granted(
        None,
        |designated, _watch, allowed, _predecessors, _availability| {
            callbacks += 1;
            acquisition_allowed = allowed;
            if designated != &coordinator_claim {
                anyhow::bail!("one-shot callback received the wrong designation");
            }
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: next_claim.clone(),
                contention: None,
            })
        },
    )?;
    let active_reads = ACTIVE_LIST_RECORD_READS.with(std::cell::Cell::get) - active_reads_before;
    let (waiting, replaced, rescan_pending) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(waiter.slot)?
            .ok_or_else(|| anyhow::anyhow!("one-shot replacement waiter disappeared"))?;
        (
            record.state == STATE_WAITING,
            record.claim == next_claim,
            table.pending_flags() & PENDING_RESCAN != 0,
        )
    };
    waiter.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        callbacks,
        matches!(result, GrantResult::Requeued) && !acquisition_allowed,
        waiting,
        replaced,
        rescan_pending,
        active_reads,
    ))
}

#[cfg(test)]
pub(super) fn exercise_prefix_epoch_validation_for_tests() -> Result<(usize, bool, bool)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let mut coordinator =
        Ticket::register(coordinator_claim.clone(), coordinator_claim.clone(), None)?;
    let watch = ClaimSet::new(std::iter::empty(), [0usize, 1usize], FlockMode::Exclusive);
    let mut waiter = Ticket::register(coordinator_claim, watch, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.grant_compatible()?;
    }
    if waiter.state_without_recovery_for_tests()? != State::Replan {
        anyhow::bail!("epoch validation waiter did not receive the REPLAN token");
    }

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let bytes = table
            .record_bytes_mut(waiter.slot)?
            .ok_or_else(|| anyhow::anyhow!("epoch validation waiter disappeared"))?;
        write_u64(bytes, R_PREFIX_EPOCH, 0);
    }
    let mut callbacks = 0usize;
    let torn = waiter.run_granted(
        None,
        |_designated, _watch, _allowed, _predecessors, _availability| {
            callbacks += 1;
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: ClaimSet::default(),
                contention: None,
            })
        },
    )?;
    let torn_demoted = matches!(torn, GrantResult::LostGrant)
        && waiter.state_without_recovery_for_tests()? == State::Waiting;
    let empty = BTreeSet::new();
    coordinator.schedule(
        None,
        &empty,
        &empty,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let torn_rejected = torn_demoted && waiter.state_without_recovery_for_tests()? == State::Replan;

    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let (_, prefix) = table.cached_prefix(waiter.slot)?;
        let epoch = table.claim_epoch();
        let record = table
            .record(waiter.slot)?
            .ok_or_else(|| anyhow::anyhow!("epoch validation waiter disappeared"))?;
        let issue_serial = table.max_watch_serial(&record.watch)?;
        table.publish_prefix(
            waiter.slot,
            &prefix,
            R_REPLAN_CLAIM_EPOCH,
            epoch,
            issue_serial,
        )?;
        table.set_record_state(waiter.slot, STATE_REPLAN)?;
        table.finish_claim_scan();
        let bytes = table
            .record_bytes_mut(waiter.slot)?
            .ok_or_else(|| anyhow::anyhow!("epoch validation waiter disappeared"))?;
        write_u64(
            bytes,
            R_REPLAN_CLAIM_EPOCH,
            epoch
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("epoch validation stamp exhausted"))?,
        );
    }
    let stale = waiter.run_granted(
        None,
        |_designated, _watch, _allowed, _predecessors, _availability| {
            callbacks += 1;
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: ClaimSet::default(),
                contention: None,
            })
        },
    )?;
    let stale_rejected = matches!(stale, GrantResult::LostGrant)
        && waiter.state_without_recovery_for_tests()? == State::Waiting;

    waiter.finish(None)?;
    coordinator.finish(None)?;
    Ok((callbacks, torn_rejected, stale_rejected))
}

#[cfg(test)]
pub(super) fn exercise_prefix_order_and_repair_for_tests() -> Result<(bool, bool, bool, bool)> {
    let predecessor_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        FlockMode::Exclusive,
        FlockMode::Shared,
    );
    let mut predecessor =
        Ticket::register(predecessor_claim.clone(), predecessor_claim.clone(), None)?;
    let target_claim = ClaimSet::new(std::iter::empty(), [2usize], FlockMode::Exclusive);
    let target_watch = ClaimSet::new(
        std::iter::empty(),
        [1usize, 2usize, 3usize],
        FlockMode::Exclusive,
    );
    let mut target = Ticket::register(target_claim.clone(), target_watch, None)?;
    let successor_claim = ClaimSet::new(std::iter::empty(), [3usize], FlockMode::Exclusive);
    let mut successor = Ticket::register(successor_claim.clone(), successor_claim.clone(), None)?;
    let shared_on_predecessor = ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        FlockMode::Exclusive,
        FlockMode::Shared,
    );
    let exclusive_on_predecessor =
        ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);

    let (initial_modes_correct, initial_excludes_successor, repaired_modes_correct, repaired_order) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let target_record = table
            .record(target.slot)?
            .ok_or_else(|| anyhow::anyhow!("target record disappeared"))?;
        let (target_epoch, target_prefix) = table.cached_prefix(target.slot)?;
        let initial_modes_correct = target_epoch == 0
            && target_record.state == STATE_WAITING
            && !target_prefix.conflicts(&shared_on_predecessor)?
            && target_prefix.conflicts(&exclusive_on_predecessor)?;
        let initial_excludes_successor = !target_prefix.conflicts(&successor_claim)?;

        // Model a writer dying with an invalid derived cache. Repair must
        // regenerate every ticket's prefix in ticket order before clearing
        // the dirty marker.
        atomic_u64(&table.header, H_AGGREGATE_DIRTY).store(1, Ordering::SeqCst);
        table.repair_consistency_if_needed()?;
        let target_record = table
            .record(target.slot)?
            .ok_or_else(|| anyhow::anyhow!("target record disappeared after repair"))?;
        let successor_record = table
            .record(successor.slot)?
            .ok_or_else(|| anyhow::anyhow!("successor record disappeared after repair"))?;
        let (target_epoch, target_prefix) = table.cached_prefix(target.slot)?;
        let (successor_epoch, successor_prefix) = table.cached_prefix(successor.slot)?;
        let repaired_modes_correct = target_epoch == 0
            && target_record.state == STATE_WAITING
            && !target_prefix.conflicts(&shared_on_predecessor)?
            && target_prefix.conflicts(&exclusive_on_predecessor)?;
        let repaired_order = successor_epoch == successor_record.replan_claim_epoch
            && !target_prefix.conflicts(&successor_claim)?
            && successor_prefix.conflicts(&target_claim)?
            && successor_prefix.conflicts(&exclusive_on_predecessor)?;
        (
            initial_modes_correct,
            initial_excludes_successor,
            repaired_modes_correct,
            repaired_order,
        )
    };

    successor.finish(None)?;
    target.finish(None)?;
    predecessor.finish(None)?;
    Ok((
        initial_modes_correct,
        initial_excludes_successor,
        repaired_modes_correct,
        repaired_order,
    ))
}

#[cfg(test)]
pub(super) fn exercise_prefix_refresh_after_predecessor_release_for_tests()
-> Result<(bool, bool, bool, bool)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let mut coordinator =
        Ticket::register(coordinator_claim.clone(), coordinator_claim.clone(), None)?;
    let predecessor_claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut predecessor =
        Ticket::register(predecessor_claim.clone(), predecessor_claim.clone(), None)?;
    let designated = ClaimSet::new(std::iter::empty(), [2usize], FlockMode::Exclusive);
    let watch = ClaimSet::new(std::iter::empty(), [1usize, 2usize], FlockMode::Exclusive);
    let mut waiter = Ticket::register(designated.clone(), watch, None)?;

    // Give the fixed predecessor an exact grant, then commit it as a real
    // holder. Acquired removal deliberately does not advance the claim epoch:
    // the real flock carries the same reservation until its release.
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, 1, true)?;
        let claim_epoch = table.claim_epoch();
        let (_, prefix) = table.cached_prefix(predecessor.slot)?;
        let record = table
            .record(predecessor.slot)?
            .ok_or_else(|| anyhow::anyhow!("release-prefix predecessor disappeared"))?;
        let issue_serial = table.max_watch_serial(&record.watch)?;
        table.publish_prefix(
            predecessor.slot,
            &prefix,
            R_GRANT_EPOCH,
            claim_epoch,
            issue_serial,
        )?;
        table.set_record_state(predecessor.slot, STATE_GRANTED)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
    }
    if waiter.state_without_recovery_for_tests()? != State::Replan {
        anyhow::bail!("release-prefix waiter did not receive the REPLAN token");
    }
    let acquired = predecessor.run_granted(
        None,
        |designated, _watch, allowed, _predecessors, _availability| {
            if !allowed || designated != &predecessor_claim {
                anyhow::bail!("release-prefix predecessor lost its exact grant");
            }
            Ok(GrantAttempt {
                acquired: Some(()),
                preparation_claim: None,
                preparation_contention: None,
                next_claim: designated.clone(),
                contention: None,
            })
        },
    )?;
    let held = match acquired {
        GrantResult::Acquired((), held) => held,
        _ => anyhow::bail!("release-prefix predecessor did not commit acquisition"),
    };

    let candidate = predecessor_claim;
    let stale_before = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let (_, before) = table.cached_prefix(waiter.slot)?;
        before.conflicts(&candidate)?
    };

    // Release the acquired predecessor before publishing the corresponding
    // physical-lock improvement. The HELD record must remain live through the
    // stale-prefix sample above; discarding it at the acquisition match would
    // test a second synthetic release instead of the production lifecycle.
    drop(held);
    let (refreshed_prefix, refreshed_publication) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, 1, true)?;
        table.stamp_resource_improvement(S_CPU_EX, 1)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;

        let record = table
            .record(waiter.slot)?
            .ok_or_else(|| anyhow::anyhow!("release-prefix waiter disappeared"))?;
        let (_, after) = table.cached_prefix(waiter.slot)?;
        (
            !after.conflicts(&candidate)?,
            record.prefix_epoch == record.replan_claim_epoch,
        )
    };

    let mut candidate_ready = false;
    let result = waiter.run_granted(
        None,
        |_designated, _watch, allowed, predecessors, availability| {
            if allowed {
                anyhow::bail!("release-prefix REPLAN callback received an acquire license");
            }
            candidate_ready =
                !predecessors.conflicts(&candidate)? && availability.allows(&candidate)?;
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: candidate.clone(),
                contention: None,
            })
        },
    )?;
    let replacement_committed = matches!(result, GrantResult::Requeued) && {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table
            .record(waiter.slot)?
            .is_some_and(|record| record.claim == candidate && record.state == STATE_WAITING)
    };

    waiter.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        stale_before && refreshed_prefix,
        refreshed_publication,
        candidate_ready,
        replacement_committed,
    ))
}

/// Reproduce a predecessor release after the coordinator planner returned
/// WAITING but before that WAITING publication takes the registry fence.
#[cfg(test)]
pub(super) fn exercise_waiting_publication_release_progress_for_tests()
-> Result<(bool, bool, bool, bool)> {
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut predecessor = Ticket::register(claim.clone(), claim.clone(), None)?;
    let mut coordinator = Ticket::register(claim.clone(), claim.clone(), None)?;

    let commit_token = predecessor.commit_token_for_tests()?;
    let held = match predecessor.finish_acquired(&claim, commit_token, &[], None)? {
        FinishAcquireResult::Committed(held) => held,
        FinishAcquireResult::Stale => {
            anyhow::bail!("release-progress predecessor acquisition unexpectedly became stale")
        }
    };

    let stale_prefix = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(coordinator.slot)?
            .filter(|record| {
                record.ticket == coordinator.ticket && record.state == STATE_COORDINATOR
            })
            .ok_or_else(|| anyhow::anyhow!("release-progress successor was not coordinator"))?;
        let (_, prefix) = table.cached_prefix(record.slot)?;
        let stale_prefix = prefix.conflicts(&claim)?;

        // Model the CI ordering: the physical close was already proven free
        // while the conservative HELD record remained published. Its removal
        // must make progress through the prefix rescan alone, without another
        // lock-close observation.
        set_cpu_free_for_tests(&mut table, 1, true)?;
        write_u64(&mut table.header, H_PENDING_FLAGS, 0);
        table.finish_claim_scan();
        stale_prefix
    };

    drop(held);
    // Change A: a HELD removal coalesces its authoritative rescan behind the
    // short quantum rather than publishing an urgent scan. Availability was
    // already free, so no further lock-close will arrive — progress must come
    // from this durable deferred edge plus the coordinator's own deferred timer
    // alone, never a second external observation.
    let (release_published_without_observation, deadline) = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        let deadline = table.deferred_rescan_deadline_ns();
        (
            table.pending_flags() & PENDING_REPLAN_RESCAN != 0
                && table.pending_flags() & PENDING_RESCAN == 0
                && table.pending_flags() & PENDING_OBSERVATION == 0
                && deadline != 0,
            deadline,
        )
    };
    // Model the coordinator's blocking wait timing out at its own deferred
    // deadline — an internal timer, not another lock-close event.
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        anyhow::ensure!(
            table.prepare_grant_scan_at(deadline)?,
            "the coalesced release edge must promote at its own deferred deadline",
        );
    }

    let empty = BTreeSet::new();
    let snapshot = coordinator.schedule(
        Some(&claim),
        &empty,
        &empty,
        &empty,
        false,
        &[],
        &[],
        false,
        None,
        false,
        None,
    )?;
    let prefix_refreshed = !snapshot.predecessors.conflicts(&claim)?;
    let immediate_step_without_observation = snapshot.should_step && snapshot.observation.is_none();

    coordinator.finish(None)?;
    Ok((
        stale_prefix,
        release_published_without_observation,
        prefix_refreshed,
        immediate_step_without_observation,
    ))
}

#[cfg(test)]
pub(super) fn exercise_issue_serial_race_for_tests() -> Result<(bool, bool, bool, bool)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let mut coordinator =
        Ticket::register(coordinator_claim.clone(), coordinator_claim.clone(), None)?;
    let designated = ClaimSet::new(std::iter::empty(), [2usize], FlockMode::Exclusive);
    let candidate = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let watch = ClaimSet::new(std::iter::empty(), [1usize, 2usize], FlockMode::Exclusive);
    let mut waiter = Ticket::register(designated.clone(), watch, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, 1, false)?;
        set_cpu_free_for_tests(&mut table, 2, false)?;
        table.grant_compatible()?;
    }
    if waiter.state_without_recovery_for_tests()? != State::Replan {
        anyhow::bail!("issue-serial waiter did not receive the REPLAN token");
    }

    let mut stale_snapshot_published_once = false;
    let first = waiter.run_granted(
        None,
        |_designated, _watch, allowed, predecessors, availability| {
            if allowed {
                anyhow::bail!("issue-serial REPLAN callback received an acquire license");
            }
            if predecessors.conflicts(&candidate)? || availability.allows(&candidate)? {
                anyhow::bail!(
                    "issue-serial callback did not start from the expected busy snapshot"
                );
            }
            {
                let _lock = lock_registry_existing(FlockMode::Exclusive)?;
                let mut table = Table::open_existing()?;
                set_cpu_free_for_tests(&mut table, 1, true)?;
                table.stamp_resource_improvement(S_CPU_EX, 1)?;
                table.set_pending_flag(PENDING_RESCAN);
                table.grant_compatible()?;
            }
            stale_snapshot_published_once = true;
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: designated.clone(),
                contention: None,
            })
        },
    )?;
    stale_snapshot_published_once &= matches!(first, GrantResult::Requeued)
        && waiter.state_without_recovery_for_tests()? == State::Waiting;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        anyhow::ensure!(
            table.pending_flags() & PENDING_RESCAN != 0,
            "REPLAN completion did not preserve the mid-callback improvement watermark",
        );
        table.grant_compatible()?;
    }
    anyhow::ensure!(
        waiter.state_without_recovery_for_tests()? == State::Replan,
        "WAITING callback did not replan from its unconsumed improvement",
    );

    let mut fresh_snapshot_seen = false;
    let second = waiter.run_granted(
        None,
        |_designated, _watch, allowed, predecessors, availability| {
            if allowed {
                anyhow::bail!("fresh issue-serial REPLAN received an acquire license");
            }
            fresh_snapshot_seen =
                !predecessors.conflicts(&candidate)? && availability.allows(&candidate)?;
            if fresh_snapshot_seen {
                // The selected alternative can become unavailable after the
                // callback snapshot. Publishing it once as WAITING is safe;
                // the mandatory successor scan must not grant it.
                let _lock = lock_registry_existing(FlockMode::Exclusive)?;
                let mut table = Table::open_existing()?;
                set_cpu_free_for_tests(&mut table, 1, false)?;
            }
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: candidate.clone(),
                contention: None,
            })
        },
    )?;
    let replacement_committed_and_revalidated = matches!(second, GrantResult::Requeued) && {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let published = table
            .record(waiter.slot)?
            .is_some_and(|record| record.claim == candidate && record.state == STATE_WAITING);
        table.grant_compatible()?;
        published
            && table
                .record(waiter.slot)?
                .is_some_and(|record| record.claim == candidate && record.state == STATE_WAITING)
    };
    let serial_consumed_only_by_fresh_snapshot = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        let record = table
            .record(waiter.slot)?
            .ok_or_else(|| anyhow::anyhow!("issue-serial waiter disappeared"))?;
        record.issue_serial >= table.max_watch_serial(&record.watch)?
    };

    waiter.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        stale_snapshot_published_once,
        fresh_snapshot_seen,
        replacement_committed_and_revalidated,
        serial_consumed_only_by_fresh_snapshot,
    ))
}

#[cfg(test)]
pub(super) fn exercise_stale_acquired_release_order_for_tests()
-> Result<(bool, bool, bool, bool, bool, bool)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let coordinator_watch =
        ClaimSet::new(std::iter::empty(), [0usize, 1usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_watch, None)?;
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut waiter = Ticket::register(claim.clone(), claim.clone(), None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, 1, true)?;
        table.set_record_state(waiter.slot, STATE_WAITING)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
    }
    if waiter.state_without_recovery_for_tests()? != State::Granted {
        anyhow::bail!("stale-acquired exercise failed to prepare a live grant");
    }

    let payload_dropped = std::rc::Rc::new(std::cell::Cell::new(false));
    let registry_unlocked_at_drop = std::rc::Rc::new(std::cell::Cell::new(false));
    let dropped_at_notify = std::rc::Rc::new(std::cell::Cell::new(false));
    let hook_payload_dropped = std::rc::Rc::clone(&payload_dropped);
    let hook_observation = std::rc::Rc::clone(&dropped_at_notify);
    NOTIFY_HOOK.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "a coordinator-notify hook is already installed on this test thread"
        );
        *slot.borrow_mut() = Some(Box::new(move || {
            hook_observation.set(hook_payload_dropped.get());
        }));
    });

    let coordinator_slot = coordinator.slot;
    let coordinator_ticket = coordinator.ticket;
    let result = waiter.run_granted(
        None,
        |designated, _watch, allowed, _predecessors, _availability| {
            if !allowed {
                anyhow::bail!("stale-acquired exercise lost its acquisition license");
            }
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            let record = table
                .record(coordinator_slot)?
                .filter(|record| record.ticket == coordinator_ticket)
                .ok_or_else(|| anyhow::anyhow!("stale-acquired coordinator disappeared"))?;
            table.replace_claim(
                coordinator_slot,
                coordinator_ticket,
                &coordinator_claim,
                &claim,
                STATE_COORDINATOR,
                record.issue_serial,
                None,
                false,
            )?;
            table.advance_generation_and_wake_pending()?;
            table.grant_compatible()?;
            Ok(GrantAttempt {
                acquired: Some(DropProbe {
                    dropped: std::rc::Rc::clone(&payload_dropped),
                    registry_unlocked: std::rc::Rc::clone(&registry_unlocked_at_drop),
                }),
                preparation_claim: None,
                preparation_contention: None,
                next_claim: designated.clone(),
                contention: None,
            })
        },
    )?;
    let lost_grant = matches!(result, GrantResult::LostGrant);
    // Do not turn this assertion into an unrelated coordinator-liveness
    // operation. The synthetic coordinator has no heartbeat loop, and an ARM
    // CI deschedule longer than the lease can otherwise elect this correctly
    // revoked waiter while `Ticket::state` is only trying to observe it.
    let regrant_revoked = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table
            .record(waiter.slot)?
            .is_some_and(|record| record.ticket == waiter.ticket && record.state == STATE_WAITING)
    };
    let payload_dropped = payload_dropped.get();
    let registry_unlocked_at_drop = registry_unlocked_at_drop.get();
    let dropped_at_notify = dropped_at_notify.get();
    let observation_requested = {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let table = Table::open_existing()?;
        table.observation_request()?.is_some()
    };

    waiter.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        lost_grant,
        regrant_revoked,
        payload_dropped,
        registry_unlocked_at_drop,
        dropped_at_notify,
        observation_requested,
    ))
}

#[cfg(test)]
fn exercise_acknowledgement_payload_notify_case(
    expired_replan: bool,
) -> Result<(bool, bool, bool)> {
    let base = if expired_replan { 100usize } else { 110usize };
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [base], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let designated = ClaimSet::new(std::iter::empty(), [base + 1], FlockMode::Exclusive);
    let watch = if expired_replan {
        ClaimSet::new(
            std::iter::empty(),
            [base + 1, base + 2],
            FlockMode::Exclusive,
        )
    } else {
        designated.clone()
    };
    let mut callback = Ticket::register(designated.clone(), watch, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        set_cpu_free_for_tests(&mut table, base, false)?;
        set_cpu_free_for_tests(&mut table, base + 1, !expired_replan)?;
        if expired_replan {
            set_cpu_free_for_tests(&mut table, base + 2, false)?;
        }
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
        let expected = if expired_replan {
            STATE_REPLAN
        } else {
            STATE_GRANTED
        };
        anyhow::ensure!(
            table
                .record(callback.slot)?
                .is_some_and(|record| record.state == expected),
            "acknowledgement payload fixture did not publish {}",
            record_state_name(expected),
        );
    }

    let payload_dropped = std::rc::Rc::new(std::cell::Cell::new(false));
    let registry_unlocked = std::rc::Rc::new(std::cell::Cell::new(false));
    let dropped_at_notify = std::rc::Rc::new(std::cell::Cell::new(false));
    arm_drop_before_notify_probe(
        std::rc::Rc::clone(&payload_dropped),
        std::rc::Rc::clone(&dropped_at_notify),
    );
    let callback_slot = callback.slot;
    let callback_ticket = callback.ticket;
    let result = callback.run_granted(
        None,
        |current, _watch, acquisition_allowed, _predecessors, _availability| {
            anyhow::ensure!(
                acquisition_allowed != expired_replan && current == &designated,
                "acknowledgement payload fixture received the wrong callback",
            );
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            let old_state = if expired_replan {
                STATE_REPLAN
            } else {
                STATE_GRANTED
            };
            let new_state = if expired_replan {
                STATE_REPLAN_EXPIRED
            } else {
                STATE_REVOKED
            };
            table.begin_transaction()?;
            let record = table
                .record(callback_slot)?
                .filter(|record| record.ticket == callback_ticket && record.state == old_state)
                .ok_or_else(|| anyhow::anyhow!("acknowledgement payload callback changed"))?;
            table.set_record_state(record.slot, new_state)?;
            // Force acknowledgement to coalesce with an existing edge. The
            // physical-drop path must still issue its targeted notification.
            table.set_pending_flag(PENDING_RESCAN);
            table.finish_transaction()?;
            Ok(GrantAttempt {
                acquired: Some(DropProbe {
                    dropped: std::rc::Rc::clone(&payload_dropped),
                    registry_unlocked: std::rc::Rc::clone(&registry_unlocked),
                }),
                preparation_claim: None,
                preparation_contention: None,
                next_claim: current.clone(),
                contention: None,
            })
        },
    )?;
    let ordered = matches!(result, GrantResult::LostGrant)
        && payload_dropped.get()
        && registry_unlocked.get()
        && dropped_at_notify.get();
    let payload_dropped = payload_dropped.get();
    let registry_unlocked = registry_unlocked.get();
    callback.finish(None)?;
    coordinator.finish(None)?;
    Ok((ordered, payload_dropped, registry_unlocked))
}

/// Both late acknowledgement shapes must preserve opaque destruction before
/// targeted coordinator notification even when RESCAN was already pending.
#[cfg(test)]
pub(super) fn exercise_acknowledgement_payload_notify_order_for_tests()
-> Result<(bool, bool, bool, bool, bool, bool)> {
    let expired = exercise_acknowledgement_payload_notify_case(true)?;
    let revoked = exercise_acknowledgement_payload_notify_case(false)?;
    Ok((
        expired.0, expired.1, expired.2, revoked.0, revoked.1, revoked.2,
    ))
}

#[cfg(test)]
pub(super) fn exercise_stale_contention_commit_for_tests() -> Result<(bool, bool, bool, bool)> {
    let coordinator_claim = ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let coordinator_next = ClaimSet::new(std::iter::empty(), [3usize], FlockMode::Exclusive);
    let coordinator_watch =
        ClaimSet::new(std::iter::empty(), [0usize, 3usize], FlockMode::Exclusive);
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_watch, None)?;
    let designated = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let alternative = ClaimSet::new(std::iter::empty(), [2usize], FlockMode::Exclusive);
    let watch = ClaimSet::new(std::iter::empty(), [1usize, 2usize], FlockMode::Exclusive);
    let mut waiter = Ticket::register(designated.clone(), watch, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        set_cpu_free_for_tests(&mut table, 1, true)?;
        set_cpu_free_for_tests(&mut table, 2, true)?;
        set_cpu_free_for_tests(&mut table, 3, true)?;
        table.set_record_state(waiter.slot, STATE_WAITING)?;
        table.clear_record_blocked(waiter.slot)?;
        table.set_pending_flag(PENDING_RESCAN);
        table.grant_compatible()?;
    }
    if waiter.state_without_recovery_for_tests()? != State::Granted {
        anyhow::bail!("stale-contention exercise failed to grant the disjoint waiter");
    }

    let coordinator_slot = coordinator.slot;
    let coordinator_ticket = coordinator.ticket;
    let result = waiter.run_granted(
        None,
        |_designated, _watch, allowed, _predecessors, _availability| {
            if !allowed {
                anyhow::bail!("stale-contention exercise lost its acquisition license");
            }
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            let record = table
                .record(coordinator_slot)?
                .filter(|record| record.ticket == coordinator_ticket)
                .ok_or_else(|| anyhow::anyhow!("stale-contention coordinator disappeared"))?;
            table.replace_claim(
                coordinator_slot,
                coordinator_ticket,
                &coordinator_claim,
                &coordinator_next,
                STATE_COORDINATOR,
                record.issue_serial,
                None,
                false,
            )?;
            // The callback's later physical negative observation must consume
            // this mid-callback improvement instead of immediately waking
            // itself for the same already-disproved state.
            table.stamp_resource_improvement(S_CPU_EX, 1)?;
            table.advance_generation_and_wake_pending()?;
            table.grant_compatible()?;
            Ok(GrantAttempt::<()> {
                acquired: None,
                preparation_claim: None,
                preparation_contention: None,
                next_claim: alternative.clone(),
                contention: Some(ContentionEvidence {
                    blocker: ResourceKey::Cpu(1),
                    mode: FlockMode::Exclusive,
                    _witness: File::open("/dev/null")?.into(),
                }),
            })
        },
    )?;

    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    let record = table
        .record(waiter.slot)?
        .ok_or_else(|| anyhow::anyhow!("stale-contention waiter disappeared"))?;
    let exact_preserved = record.claim == designated;
    let blocked_current = record.blocked_on.is_some_and(|blocked| {
        blocked.key == ResourceKey::Cpu(1)
            && blocked.mode == FlockMode::Exclusive
            && table
                .blocker_serial(blocked.key, blocked.mode)
                .is_ok_and(|serial| serial == blocked.serial)
    });
    table.grant_compatible()?;
    let stayed_waiting = table
        .record(waiter.slot)?
        .is_some_and(|record| record.state == STATE_WAITING);
    drop(table);
    drop(_lock);

    waiter.finish(None)?;
    coordinator.finish(None)?;
    Ok((
        matches!(result, GrantResult::Requeued),
        exact_preserved,
        blocked_current,
        stayed_waiting,
    ))
}

#[cfg(test)]
fn diagnostic_counter_for_tests(offset: usize) -> Result<u64> {
    let _lock = lock_registry_existing(FlockMode::Shared)?;
    let file = File::open(header_path())?;
    let map = unsafe { Mmap::map(&file) }?;
    HeaderLayout::validate(&map)?;
    Ok(read_u64(&map, offset))
}

#[cfg(test)]
pub(super) fn defer_liveness_maintenance_for_tests() -> Result<()> {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let _lock = loop {
        if let Some(lock) = try_lock_registry_existing_nonblocking(FlockMode::Exclusive)? {
            break lock;
        }
        if std::time::Instant::now() >= deadline {
            anyhow::bail!("timed out taking admission registry lock to defer liveness maintenance");
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    write_u64(
        &mut table.header,
        H_LAST_LIVENESS_SWEEP_NS,
        monotonic_now_ns()?,
    );
    write_u64(&mut table.header, H_LIVENESS_RECONCILE_BY_NS, 0);
    Ok(())
}

#[cfg(test)]
pub(super) fn initializer_temp_count_for_tests() -> Result<usize> {
    let dir = registry_data_dir();
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("scan admission registry directory {}", dir.display()));
        }
    };
    let mut count = 0;
    for entry in entries {
        if entry?
            .file_name()
            .to_string_lossy()
            .starts_with(INITIALIZER_PREFIX)
        {
            count += 1;
        }
    }
    Ok(count)
}

#[cfg(test)]
pub(super) fn observer_preserves_uninitialized_header_for_tests() -> Result<bool> {
    std::fs::create_dir_all(registry_data_dir())?;
    std::fs::create_dir_all(event_dir())?;
    crate::flock::materialize(registry_writer_intent_path())?;
    crate::flock::materialize(registry_lock_path())?;
    let missing_snapshot = snapshot()?;
    let missing_preserved = !header_path().exists();

    crate::flock::materialize(header_path())?;
    let zero_snapshot = snapshot()?;
    let zero_preserved = std::fs::metadata(header_path())?.len() == 0;

    let zero_header = OpenOptions::new()
        .write(true)
        .open(header_path())
        .context("open zeroed observer-test header")?;
    zero_header.set_len(HEADER_ALIGN as u64)?;
    drop(zero_header);
    let zeroed_snapshot = snapshot()?;
    let zeroed_preserved = std::fs::read(header_path())?
        .into_iter()
        .all(|byte| byte == 0);
    Ok(missing_snapshot.is_empty()
        && zero_snapshot.is_empty()
        && zeroed_snapshot.is_empty()
        && missing_preserved
        && zero_preserved
        && zeroed_preserved)
}

#[cfg(test)]
pub(super) fn prepare_zeroed_uninitialized_header_for_tests() -> Result<()> {
    std::fs::create_dir_all(registry_data_dir())?;
    std::fs::create_dir_all(event_dir())?;
    crate::flock::materialize(registry_writer_intent_path())?;
    crate::flock::materialize(registry_lock_path())?;
    let header = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(header_path())
        .context("create zeroed observer-test header")?;
    header.set_len(HEADER_ALIGN as u64)?;
    Ok(())
}

fn validate_claim(claim: &ClaimSet) -> Result<()> {
    if claim.is_empty() {
        anyhow::bail!("queue claims must designate a non-empty exact reservation");
    }
    Ok(())
}

const CLAIM_DIAGNOSTIC_INDEX_LIMIT: usize = 8;

fn bounded_index_set(indices: &BTreeSet<usize>) -> String {
    let mut preview = indices
        .iter()
        .take(CLAIM_DIAGNOSTIC_INDEX_LIMIT)
        .map(usize::to_string)
        .collect::<Vec<_>>();
    if indices.len() > CLAIM_DIAGNOSTIC_INDEX_LIMIT {
        preview.push(format!(
            "... +{}",
            indices.len() - CLAIM_DIAGNOSTIC_INDEX_LIMIT
        ));
    }
    format!("len={} [{}]", indices.len(), preview.join(", "))
}

fn bounded_claim(claim: &ClaimSet) -> String {
    format!(
        "class={:?} cpus({}) llcs({}) permits({}) \
         modes(cpu={:?}, llc={:?}, permit={:?})",
        claim.admission_class,
        bounded_index_set(&claim.cpus),
        bounded_index_set(&claim.llcs),
        bounded_index_set(&claim.permits),
        claim.cpu_mode,
        claim.llc_mode,
        claim.permit_mode,
    )
}

pub(super) fn validate_claim_within_watch(claim: &ClaimSet, watch: &ClaimSet) -> Result<()> {
    let mode_within = |claim_empty, watch_empty, claim_mode, watch_mode| {
        claim_empty || watch_empty || watch_mode == ClaimMode::Exclusive || claim_mode == watch_mode
    };
    let admission_within = matches!(
        (watch.admission_class, claim.admission_class),
        (AdmissionClass::Ordinary, AdmissionClass::Ordinary)
            | (AdmissionClass::Ordinary, AdmissionClass::DefaultBorrow)
            | (AdmissionClass::DefaultBorrow, AdmissionClass::Ordinary)
            | (AdmissionClass::DefaultBorrow, AdmissionClass::DefaultBorrow)
            | (AdmissionClass::Build, AdmissionClass::Build)
    );
    let mut violations = Vec::new();
    if !claim.cpus.is_subset(&watch.cpus) {
        violations.push("CPU set");
    }
    if !claim.llcs.is_subset(&watch.llcs) {
        violations.push("LLC set");
    }
    if !claim.permits.is_subset(&watch.permits) {
        violations.push("permit set");
    }
    if !mode_within(
        claim.cpus.is_empty(),
        watch.cpus.is_empty(),
        claim.cpu_mode,
        watch.cpu_mode,
    ) {
        violations.push("CPU mode");
    }
    if !mode_within(
        claim.llcs.is_empty(),
        watch.llcs.is_empty(),
        claim.llc_mode,
        watch.llc_mode,
    ) {
        violations.push("LLC mode");
    }
    if !mode_within(
        claim.permits.is_empty(),
        watch.permits.is_empty(),
        claim.permit_mode,
        watch.permit_mode,
    ) {
        violations.push("permit mode");
    }
    if !admission_within {
        violations.push("admission class");
    }
    if !violations.is_empty() {
        anyhow::bail!(
            "queue claim is outside its immutable watch set ({violations}): claim={claim}, \
             watch={watch}",
            violations = violations.join(", "),
            claim = bounded_claim(claim),
            watch = bounded_claim(watch),
        );
    }
    Ok(())
}

fn validate_contention_within_watch(
    contention: &[ContentionMarker],
    watch: &ClaimSet,
) -> Result<()> {
    for marker in contention {
        let valid = contention_marker_within_watch(*marker, watch);
        if !valid {
            anyhow::bail!(
                "queue contention marker is outside its immutable watch set: \
                 marker={marker:?}, watch={watch}",
                watch = bounded_claim(watch),
            );
        }
    }
    Ok(())
}

fn contention_marker_within_watch(marker: ContentionMarker, watch: &ClaimSet) -> bool {
    match marker.blocker {
        ResourceKey::Cpu(index) => {
            watch.cpus.contains(&index)
                && (watch.cpu_mode == ClaimMode::Exclusive
                    || ClaimMode::from(marker.mode) == watch.cpu_mode)
        }
        ResourceKey::Llc(index) => {
            watch.llcs.contains(&index)
                && (watch.llc_mode == ClaimMode::Exclusive
                    || ClaimMode::from(marker.mode) == watch.llc_mode)
        }
        ResourceKey::Permit(index) => {
            watch.permits.contains(&index)
                && (watch.permit_mode == ClaimMode::Exclusive
                    || ClaimMode::from(marker.mode) == watch.permit_mode)
        }
    }
}

fn union_claims(a: &ClaimSet, b: &ClaimSet) -> ClaimSet {
    a.union_envelope(b)
}

fn pending_intent_watch(selected_final: &ClaimSet, preparation: &ClaimSet) -> ClaimSet {
    let mut watch = selected_final.union_envelope(preparation);
    // Preparation borrows cooperative capacity transiently. Preserve the
    // selected run's fairness identity in the intent envelope; the exact claim
    // independently describes only the physically owned preparation tuple.
    watch.admission_class = selected_final.admission_class;
    watch
}

fn claim_covers(cover: &ClaimSet, claim: &ClaimSet) -> bool {
    let mode_covers = |claim_empty: bool, cover_mode, claim_mode| {
        claim_empty || cover_mode == ClaimMode::Exclusive || cover_mode == claim_mode
    };
    claim.cpus.is_subset(&cover.cpus)
        && claim.llcs.is_subset(&cover.llcs)
        && claim.permits.is_subset(&cover.permits)
        && mode_covers(claim.cpus.is_empty(), cover.cpu_mode, claim.cpu_mode)
        && mode_covers(claim.llcs.is_empty(), cover.llc_mode, claim.llc_mode)
        && mode_covers(
            claim.permits.is_empty(),
            cover.permit_mode,
            claim.permit_mode,
        )
}

fn claim_is_flexible(claim: &ClaimSet, watch: &ClaimSet) -> bool {
    claim.cpus != watch.cpus
        || claim.llcs != watch.llcs
        || claim.cpu_mode != watch.cpu_mode
        || claim.llc_mode != watch.llc_mode
        || claim.permits != watch.permits
        || claim.permit_mode != watch.permit_mode
        || claim.admission_class != watch.admission_class
}

fn materialize_claim_paths(claim: &ClaimSet) -> Result<()> {
    for &llc in &claim.llcs {
        materialize_if_missing(super::super::llc_lock_path(llc))?;
    }
    for &cpu in &claim.cpus {
        materialize_if_missing(super::super::cpu_lock_path(cpu))?;
    }
    for &permit in &claim.permits {
        materialize_if_missing(super::super::permit_lock_path(permit))?;
    }
    Ok(())
}

fn materialize_if_missing(path: impl AsRef<std::path::Path>) -> Result<()> {
    let path = path.as_ref();
    match std::fs::metadata(path) {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            crate::flock::materialize(path)
        }
        Err(error) => Err(error)
            .with_context(|| format!("inspect admission resource lock {}", path.display())),
    }
}

fn required_resource_bits(claim: &ClaimSet) -> usize {
    let physical_bits = claim
        .llcs
        .iter()
        .chain(&claim.cpus)
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let permit_bits = claim
        .permits
        .iter()
        .copied()
        .max()
        .map(|permit| {
            host_cpu_resource_bits()
                .saturating_add(permit)
                .saturating_add(1)
        })
        .unwrap_or(0);
    // The v26 mapping is deliberately overprovisioned once. It never needs a
    // migration/grow protocol when the first low-index ticket is followed by a
    // valid sparse CPU/LLC/permit index on the same host. Permit indices occupy
    // a disjoint internal bit range immediately after possible host CPUs.
    physical_bits
        .max(permit_bits)
        .max(
            host_cpu_resource_bits()
                .saturating_mul(2)
                .min(MAX_RESOURCE_BITS),
        )
        .max(4096)
}

pub(super) fn required_bits_for_claim(claim: &ClaimSet) -> usize {
    required_resource_bits(claim)
}

pub(super) fn required_bits_for_permit_index(max_permit: usize) -> usize {
    host_cpu_resource_bits()
        .saturating_add(max_permit)
        .saturating_add(1)
        .max(
            host_cpu_resource_bits()
                .saturating_mul(2)
                .min(MAX_RESOURCE_BITS),
        )
        .clamp(4096, MAX_RESOURCE_BITS)
}

fn permit_resource_index(permit: usize) -> Result<usize> {
    let index = host_cpu_resource_bits()
        .checked_add(permit)
        .ok_or_else(|| anyhow::anyhow!("permit resource index overflow"))?;
    if index >= MAX_RESOURCE_BITS {
        anyhow::bail!(
            "permit index {permit} maps to resource index {index}, exceeding registry capacity"
        );
    }
    Ok(index)
}

fn split_cpu_permit_indices(indices: BTreeSet<usize>) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let base = host_cpu_resource_bits();
    let mut cpus = BTreeSet::new();
    let mut permits = BTreeSet::new();
    for index in indices {
        if index < base {
            cpus.insert(index);
        } else {
            permits.insert(index - base);
        }
    }
    (cpus, permits)
}

fn host_cpu_resource_bits() -> usize {
    static BITS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BITS.get_or_init(|| {
        std::fs::read_to_string("/sys/devices/system/cpu/possible")
            .ok()
            .and_then(|text| {
                text.trim()
                    .split(',')
                    .filter_map(|range| {
                        range
                            .rsplit_once('-')
                            .map_or(range, |(_, end)| end)
                            .parse()
                            .ok()
                    })
                    .max()
            })
            .unwrap_or(63usize)
            .saturating_add(1)
    })
}

fn host_planner_capacity() -> usize {
    static CAPACITY: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAPACITY.get_or_init(|| {
        std::fs::read_to_string("/sys/devices/system/cpu/possible")
            .ok()
            .and_then(|text| {
                text.trim().split(',').try_fold(0usize, |total, range| {
                    let (start, end) = range
                        .split_once('-')
                        .map_or((range, range), |(start, end)| (start, end));
                    let start = start.parse::<usize>().ok()?;
                    let end = end.parse::<usize>().ok()?;
                    total.checked_add(end.checked_sub(start)?.checked_add(1)?)
                })
            })
            .or_else(|| std::thread::available_parallelism().ok().map(usize::from))
            .unwrap_or(1)
            .max(1)
    })
}

fn registry_lock_path() -> PathBuf {
    registry_data_dir().join("registry.lock")
}

fn registry_writer_intent_path() -> PathBuf {
    registry_data_dir().join("registry.turnstile")
}

fn header_path() -> PathBuf {
    registry_data_dir().join("registry.map")
}

fn notify_path() -> PathBuf {
    event_dir().join("notify")
}

fn registry_data_dir() -> PathBuf {
    active_protocol_dir().join("ktstr-acquire-registry-v26")
}

pub(super) fn event_dir() -> PathBuf {
    active_protocol_dir().join("ktstr-acquire-events-v26")
}

#[cfg(test)]
pub(super) fn protocol_dir_path() -> PathBuf {
    active_protocol_dir()
}

pub(super) fn notify_basename() -> Result<std::ffi::OsString> {
    notify_path()
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .ok_or_else(|| anyhow::anyhow!("queue registry notify path has no basename"))
}

fn chunk_path(chunk: u64) -> PathBuf {
    registry_data_dir().join(format!("{chunk:08}.chunk"))
}

fn liveness_path(slot: u64, ticket: u64) -> PathBuf {
    event_dir().join(format!(
        "{LIVENESS_PREFIX}{slot:020}{LIVENESS_SEPARATOR}{ticket:020}{LIVENESS_SUFFIX}"
    ))
}

pub(super) fn parse_liveness_basename(name: &std::ffi::OsStr) -> Option<(u64, u64)> {
    let name = name.to_str()?;
    let payload = name
        .strip_prefix(LIVENESS_PREFIX)?
        .strip_suffix(LIVENESS_SUFFIX)?;
    let (slot, ticket) = payload.split_once(LIVENESS_SEPARATOR)?;
    Some((slot.parse().ok()?, ticket.parse().ok()?))
}

/// Sleep until a registry mutation can change a rejected PENDING placement.
/// Sampling occurs while holding the registry SH flock; the futex value then
/// closes the unlock-to-wait race without polling or repeatedly rebuilding
/// heavyweight preparation state.
pub(super) fn wait_for_generation_change(expected: u32, timeout: Duration) -> Result<()> {
    #[cfg(test)]
    GENERATION_WAIT_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));
    let lock = lock_registry_existing(FlockMode::Shared)?;
    let file = File::open(header_path()).context("open admission generation futex")?;
    let map = unsafe { Mmap::map(&file) }.context("map admission generation futex")?;
    HeaderLayout::validate(&map)?;
    let wake = atomic_u32(&map, H_GENERATION_WAKE);
    if wake.load(Ordering::Acquire) != expected {
        return Ok(());
    }
    drop(lock);

    let ts = libc::timespec {
        tv_sec: timeout.as_secs().try_into().unwrap_or(libc::time_t::MAX),
        tv_nsec: timeout.subsec_nanos().into(),
    };
    // SAFETY: `wake` is an aligned u32 in a live MAP_SHARED mapping. A change
    // between the sample and this call yields EAGAIN instead of sleeping past
    // the mutation.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_futex,
            (wake as *const AtomicU32).cast::<u32>(),
            libc::FUTEX_WAIT,
            expected,
            &ts as *const libc::timespec,
            std::ptr::null::<u32>(),
            0u32,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::EAGAIN) | Some(libc::EINTR) => Ok(()),
        Some(libc::ETIMEDOUT) => {
            // This caller owns no queue ticket, so it cannot use the ordinary
            // waiter futex path to recover a live coordinator which stopped
            // making progress. The completed bounded wait is the sole license
            // for a semantic lease transfer; registration itself performs
            // dead-liveness repair only and never churns a merely descheduled
            // pre-loop owner.
            drop(map);
            drop(file);
            let _lock = lock_registry_existing(FlockMode::Exclusive)?;
            let mut table = Table::open_existing()?;
            table.repair_consistency_if_needed()?;
            table.recover_coordinator_if_stalled()
        }
        _ => Err(error).context("wait for admission registry generation change"),
    }
}

#[cfg(test)]
pub(super) fn reset_generation_wait_calls_for_tests() {
    GENERATION_WAIT_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
pub(super) fn generation_wait_calls_for_tests() -> usize {
    GENERATION_WAIT_CALLS.with(std::cell::Cell::get)
}

/// RAII ownership of the authoritative registry flock.
///
/// An exclusive registry owner also retains one shared lock on the writer-
/// intent sidecar. Readers take that sidecar exclusively only while acquiring
/// their shared registry lock, so ordinary readers still overlap after
/// admission while any announced writer closes the reader-barge window.
pub(crate) struct RegistryLock {
    target: Option<OwnedFd>,
    writer_intent: Option<OwnedFd>,
}

impl RegistryLock {
    fn new(target: OwnedFd, writer_intent: Option<OwnedFd>) -> Self {
        Self {
            target: Some(target),
            writer_intent,
        }
    }
}

impl Drop for RegistryLock {
    fn drop(&mut self) {
        // The target must unlock before writer intent. Otherwise a reader can
        // pass the sidecar while this writer still owns registry.lock EX.
        drop(self.target.take());
        drop(self.writer_intent.take());
    }
}

fn writer_intent_mode(mode: FlockMode) -> FlockMode {
    match mode {
        FlockMode::Exclusive => FlockMode::Shared,
        FlockMode::Shared => FlockMode::Exclusive,
    }
}

fn finish_registry_lock(
    registry: OwnedFd,
    writer_intent: OwnedFd,
    mode: FlockMode,
) -> RegistryLock {
    let writer_intent = if mode == FlockMode::Exclusive {
        Some(writer_intent)
    } else {
        drop(writer_intent);
        None
    };
    RegistryLock::new(registry, writer_intent)
}

fn note_registry_ex_acquisition(_mode: FlockMode) {
    #[cfg(test)]
    if _mode == FlockMode::Exclusive {
        REGISTRY_EX_ACQUISITIONS.with(|count| count.set(count.get().saturating_add(1)));
    }
}

fn open_existing_lock(path: &Path, what: &str) -> Result<Option<File>> {
    match OpenOptions::new().read(true).write(true).open(path) {
        Ok(file) => Ok(Some(file)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => {
            Err(error).with_context(|| format!("open existing {what} {}", path.display()))
        }
    }
}

fn open_existing_writer_intent_after_initial_miss(
    after_initial_miss: impl FnOnce() -> Result<()>,
) -> Result<Option<File>> {
    let path = registry_writer_intent_path();
    let writer_intent = open_existing_lock(&path, "admission registry writer-intent gate")?;
    if writer_intent.is_some() {
        return Ok(writer_intent);
    }
    after_initial_miss()?;
    if !registry_lock_path().exists() {
        return Ok(None);
    }
    // A correct initializer materializes and locks the sidecar before it
    // creates registry.lock. The first open and the registry existence test
    // are not atomic, though: an observer can miss the sidecar immediately
    // before a concurrent initializer publishes both paths. Reopen once after
    // observing registry.lock so that valid transition does not masquerade as
    // an incompatible protocol instance. A second miss is genuinely invalid;
    // recreating the sidecar here could split live participants across inodes.
    open_existing_lock(&path, "admission registry writer-intent gate")?.map_or_else(
        || {
            anyhow::bail!(
                "admission registry lock {} exists without its v{VERSION} writer-intent gate {}",
                registry_lock_path().display(),
                path.display(),
            )
        },
        |writer_intent| Ok(Some(writer_intent)),
    )
}

fn open_existing_writer_intent() -> Result<Option<File>> {
    open_existing_writer_intent_after_initial_miss(|| Ok(()))
}

fn flock_open_file(file: &File, path: &Path, mode: FlockMode, nonblocking: bool) -> Result<bool> {
    use rustix::fs::{FlockOperation, flock};

    let operation = match (mode, nonblocking) {
        (FlockMode::Exclusive, false) => FlockOperation::LockExclusive,
        (FlockMode::Shared, false) => FlockOperation::LockShared,
        (FlockMode::Exclusive, true) => FlockOperation::NonBlockingLockExclusive,
        (FlockMode::Shared, true) => FlockOperation::NonBlockingLockShared,
    };
    match flock(file, operation) {
        Ok(()) => Ok(true),
        Err(error) if nonblocking && error == rustix::io::Errno::WOULDBLOCK => Ok(false),
        Err(errno) => Err(std::io::Error::from_raw_os_error(errno.raw_os_error()))
            .with_context(|| format!("lock admission registry file {}", path.display())),
    }
}

fn lock_registry_for_initialization() -> Result<RegistryLock> {
    std::fs::create_dir_all(registry_data_dir())?;
    std::fs::create_dir_all(event_dir())?;
    // Materialization order is protocol: once registry.lock is nameable, every
    // v26 entrant must also be able to join the writer-intent gate ahead of it.
    let writer_intent = block_flock(registry_writer_intent_path(), FlockMode::Shared)?;
    let registry = block_flock(registry_lock_path(), FlockMode::Exclusive)?;
    let lock = finish_registry_lock(registry, writer_intent, FlockMode::Exclusive);
    note_registry_ex_acquisition(FlockMode::Exclusive);
    Ok(lock)
}

fn lock_registry_existing(mode: FlockMode) -> Result<RegistryLock> {
    try_lock_registry_existing(mode)?.ok_or_else(|| {
        anyhow::anyhow!(
            "existing admission registry lock {} disappeared",
            registry_lock_path().display()
        )
    })
}

fn try_lock_registry_existing(mode: FlockMode) -> Result<Option<RegistryLock>> {
    try_lock_registry_existing_with_writer_intent_hook(mode, || Ok(()))
}

fn try_lock_registry_existing_with_writer_intent_hook(
    mode: FlockMode,
    on_writer_intent: impl FnOnce() -> Result<()>,
) -> Result<Option<RegistryLock>> {
    let Some(writer_intent) = open_existing_writer_intent()? else {
        return Ok(None);
    };
    let writer_intent_path = registry_writer_intent_path();
    flock_open_file(
        &writer_intent,
        &writer_intent_path,
        writer_intent_mode(mode),
        false,
    )?;
    on_writer_intent()?;

    let registry_path = registry_lock_path();
    let Some(registry) = open_existing_lock(&registry_path, "admission registry lock")? else {
        return Ok(None);
    };
    flock_open_file(&registry, &registry_path, mode, false)?;
    let lock = finish_registry_lock(registry.into(), writer_intent.into(), mode);
    note_registry_ex_acquisition(mode);
    // A writer announces intent with sidecar SH before waiting for target EX
    // and retains both locks. Every later reader needs sidecar EX before target
    // SH, so it cannot pass any announced writer. Multiple writers may announce
    // concurrently, which also leaves no reader admission gap between queued
    // writers. This does not claim strict arrival FIFO: the kernel may choose
    // either entrant during the initial transition before intent is held.
    Ok(Some(lock))
}

enum NonblockingRegistryLock {
    Acquired(RegistryLock),
    Missing,
    Contended,
}

fn probe_lock_registry_existing_nonblocking(mode: FlockMode) -> Result<NonblockingRegistryLock> {
    let Some(writer_intent) = open_existing_writer_intent()? else {
        return Ok(NonblockingRegistryLock::Missing);
    };
    let writer_intent_path = registry_writer_intent_path();
    if !flock_open_file(
        &writer_intent,
        &writer_intent_path,
        writer_intent_mode(mode),
        true,
    )? {
        return Ok(NonblockingRegistryLock::Contended);
    }

    let registry_path = registry_lock_path();
    let Some(registry) = open_existing_lock(&registry_path, "admission registry lock")? else {
        return Ok(NonblockingRegistryLock::Missing);
    };
    if !flock_open_file(&registry, &registry_path, mode, true)? {
        return Ok(NonblockingRegistryLock::Contended);
    }
    let lock = finish_registry_lock(registry.into(), writer_intent.into(), mode);
    note_registry_ex_acquisition(mode);
    Ok(NonblockingRegistryLock::Acquired(lock))
}

fn try_lock_registry_existing_nonblocking(mode: FlockMode) -> Result<Option<RegistryLock>> {
    Ok(match probe_lock_registry_existing_nonblocking(mode)? {
        NonblockingRegistryLock::Acquired(lock) => Some(lock),
        NonblockingRegistryLock::Missing | NonblockingRegistryLock::Contended => None,
    })
}

fn try_lock_registry_for_initialization() -> Result<Option<RegistryLock>> {
    std::fs::create_dir_all(registry_data_dir())?;
    std::fs::create_dir_all(event_dir())?;
    let Some(writer_intent) = try_flock(registry_writer_intent_path(), FlockMode::Shared)? else {
        return Ok(None);
    };
    let Some(registry) = try_flock(registry_lock_path(), FlockMode::Exclusive)? else {
        return Ok(None);
    };
    let lock = finish_registry_lock(registry, writer_intent, FlockMode::Exclusive);
    note_registry_ex_acquisition(FlockMode::Exclusive);
    Ok(Some(lock))
}

fn check_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        Err(interrupted())
    } else {
        Ok(())
    }
}

fn normalize_cancellation<T>(result: Result<T>, cancelled: Option<&AtomicBool>) -> Result<T> {
    match result {
        Ok(value) => {
            check_cancelled(cancelled)?;
            Ok(value)
        }
        Err(error) => {
            check_cancelled(cancelled)?;
            Err(error)
        }
    }
}

fn lock_registry_interruptible(cancelled: Option<&AtomicBool>) -> Result<RegistryLock> {
    check_cancelled(cancelled)?;
    let lock = match try_lock_registry_existing(FlockMode::Exclusive)? {
        Some(lock) => lock,
        None => lock_registry_for_initialization()?,
    };
    normalize_cancellation(Ok(lock), cancelled)
}

fn lock_registry_interruptible_existing(cancelled: Option<&AtomicBool>) -> Result<RegistryLock> {
    check_cancelled(cancelled)?;
    normalize_cancellation(lock_registry_existing(FlockMode::Exclusive), cancelled)
}

fn notify_coordinator() {
    #[cfg(test)]
    NOTIFY_CALLS.with(|calls| calls.set(calls.get().wrapping_add(1)));
    #[cfg(test)]
    if let Some(hook) = NOTIFY_HOOK.with(|slot| slot.borrow_mut().take()) {
        hook();
    }
    let path = notify_path();
    if let Err(error) = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
    {
        tracing::warn!(
            path = %path.display(),
            %error,
            "failed to notify the host-admission coordinator; the bounded recovery wake remains active",
        );
    }
}

pub(super) fn publish_acquired(claim: &ClaimSet) -> Result<HeldClaim> {
    let namespace = RegistryNamespace::resolve();
    let _namespace = namespace.enter();
    validate_claim(claim)?;
    materialize_claim_paths(claim)?;
    let _lock = lock_registry_interruptible(None)?;
    let required_bits = required_resource_bits(claim);
    let mut table = Table::open(required_bits)?;
    table.repair_consistency_if_needed()?;
    table.recover_coordinator_if_dead()?;
    let held = publish_acquired_in_table(&mut table, claim, namespace)?;
    drop(table);
    drop(_lock);
    notify_coordinator();
    Ok(held)
}

fn publish_acquired_in_table(
    table: &mut Table,
    claim: &ClaimSet,
    namespace: RegistryNamespace,
) -> Result<HeldClaim> {
    table.begin_transaction()?;

    let ticket = table.next_ticket()?;
    let slot = table.allocate_slot()?;
    let liveness_path = liveness_path(slot, ticket);
    let liveness = try_flock(&liveness_path, FlockMode::Exclusive)?
        .ok_or_else(|| anyhow::anyhow!("fresh held-claim liveness file is already locked"))?;
    let claim_epoch = table.claim_epoch();
    let predecessors = table.aggregate_claim_snapshot();
    table.initialize_record(
        slot,
        ticket,
        std::process::id(),
        claim,
        &ClaimSet::default(),
        STATE_HELD,
        &predecessors,
        claim_epoch,
        0,
    )?;
    table.append_active(slot)?;
    table.adjust_claim_counts(claim, true)?;
    table.adjust_held_counts(claim, true)?;
    table.publish_claim_busy(claim)?;
    table.set_next_ticket(
        ticket
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("queue ticket id overflow"))?,
    );
    table.mark_claim_changed_fencing(ticket, claim)?;
    table.advance_generation()?;
    table.finish_transaction()?;
    Ok(HeldClaim {
        namespace,
        slot,
        ticket,
        liveness_path,
        liveness: Some(liveness),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeaderLayout {
    words: usize,
    bits: usize,
    record_size: usize,
    header_size: usize,
}

impl HeaderLayout {
    fn new(bits: usize) -> Result<Self> {
        if bits == 0 || bits > MAX_RESOURCE_BITS {
            anyhow::bail!(
                "queue registry resource capacity {bits} is outside the supported range 1..={MAX_RESOURCE_BITS}"
            );
        }
        let words = bits
            .checked_add(63)
            .ok_or_else(|| anyhow::anyhow!("queue registry resource word count overflow"))?
            / 64;
        let bits = words
            .checked_mul(64)
            .ok_or_else(|| anyhow::anyhow!("queue registry resource capacity overflow"))?;
        let bitset_bytes = words
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| anyhow::anyhow!("queue registry bitset size overflow"))?;
        let count_bytes = bits
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| anyhow::anyhow!("queue registry counter size overflow"))?;
        let serial_bytes = bits
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| anyhow::anyhow!("queue registry serial array size overflow"))?;
        let payload = HEADER_FIXED
            .checked_add(
                bitset_bytes
                    .checked_mul(HEADER_BITMAPS)
                    .ok_or_else(|| anyhow::anyhow!("queue registry header size overflow"))?,
            )
            .and_then(|size| size.checked_add(count_bytes.checked_mul(AGGREGATE_BITMAPS)?))
            .and_then(|size| size.checked_add(serial_bytes.checked_mul(SERIAL_ARRAYS)?))
            .and_then(|size| size.checked_add(serial_bytes.checked_mul(REQUEST_ARRAYS)?))
            .ok_or_else(|| anyhow::anyhow!("queue registry header size overflow"))?;
        let header_size = align_up_checked(payload, HEADER_ALIGN)?;
        let record_payload = RECORD_FIXED
            .checked_add(
                bitset_bytes
                    .checked_mul(RECORD_BITMAPS)
                    .ok_or_else(|| anyhow::anyhow!("queue registry record size overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("queue registry record size overflow"))?;
        let record_size = align_up_checked(record_payload, RECORD_ALIGN)?;
        u32::try_from(words).context("queue registry word count does not fit its header field")?;
        u32::try_from(record_size)
            .context("queue registry record size does not fit its header field")?;
        Ok(Self {
            words,
            bits,
            record_size,
            header_size,
        })
    }

    fn validate(map: &[u8]) -> Result<Self> {
        if map.len() < HEADER_FIXED {
            anyhow::bail!(
                "queue registry v{VERSION} header is truncated: {} bytes < {HEADER_FIXED}",
                map.len()
            );
        }
        if read_u64(map, H_MAGIC) != MAGIC || read_u32(map, H_VERSION) != VERSION {
            anyhow::bail!(
                "queue registry has unsupported magic {:#018x} or version {} (expected v{VERSION})",
                read_u64(map, H_MAGIC),
                read_u32(map, H_VERSION)
            );
        }
        let words = read_u32(map, H_WORDS) as usize;
        if words == 0 {
            anyhow::bail!("queue registry v{VERSION} header has zero resource words");
        }
        let bits = words
            .checked_mul(64)
            .ok_or_else(|| anyhow::anyhow!("queue registry resource capacity overflow"))?;
        let expected = Self::new(bits).context("invalid queue registry resource layout")?;
        if read_u32(map, H_RECORD_SIZE) as usize != expected.record_size
            || read_u32(map, H_RECORDS_PER_CHUNK) as usize != RECORDS_PER_CHUNK
            || map.len() < expected.header_size
        {
            anyhow::bail!(
                "queue registry v{VERSION} stride or chunk layout is malformed: \
                 record_size={}, records_per_chunk={}, file_size={}, \
                 expected_record_size={}, expected_records_per_chunk={RECORDS_PER_CHUNK}, \
                 minimum_file_size={}",
                read_u32(map, H_RECORD_SIZE),
                read_u32(map, H_RECORDS_PER_CHUNK),
                map.len(),
                expected.record_size,
                expected.header_size
            );
        }
        let replan_capacity = read_u64(map, H_REPLAN_CAPACITY);
        if replan_capacity == 0
            || usize::try_from(replan_capacity)
                .ok()
                .is_none_or(|capacity| capacity > expected.bits)
        {
            anyhow::bail!(
                "queue registry v{VERSION} planner capacity {replan_capacity} is outside 1..={}",
                expected.bits,
            );
        }
        Ok(expected)
    }

    fn bitset_offset(self, which: usize) -> usize {
        HEADER_FIXED + which * self.words * 8
    }

    fn count_offset(self, which: usize) -> usize {
        HEADER_FIXED + self.words * 8 * HEADER_BITMAPS + which * self.bits * 4
    }

    fn serial_offset(self, which: usize) -> usize {
        HEADER_FIXED
            + self.words * 8 * HEADER_BITMAPS
            + self.bits * 4 * AGGREGATE_BITMAPS
            + which * self.bits * 8
    }

    fn request_offset(self, which: usize) -> usize {
        HEADER_FIXED
            + self.words * 8 * HEADER_BITMAPS
            + self.bits * 4 * AGGREGATE_BITMAPS
            + self.bits * 8 * SERIAL_ARRAYS
            + which * self.bits * 8
    }
}

struct Table {
    header: MmapMut,
    layout: HeaderLayout,
    chunks: BTreeMap<u64, MmapMut>,
}

impl Table {
    fn open(required_bits: usize) -> Result<Self> {
        let path = header_path();
        crate::flock::materialize(&path)?;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("open queue registry {}", path.display()))?;
        let len = usize::try_from(file.metadata()?.len())
            .context("queue registry header length does not fit this process")?;
        if len == 0 {
            let layout = HeaderLayout::new(required_bits)?;
            drop(file);
            // This is the only path that creates an initializer. A process
            // killed before tempfile::persist cannot run TempPath::drop, so
            // reclaim unreachable initializer inodes here rather than scanning
            // the shared lock directory on every registry operation.
            remove_stale_initializers()?;
            crate::flock::materialize(notify_path())?;
            file = initialize_header_file(&path, layout)?;
        } else {
            let prefix = unsafe { Mmap::map(&file) }?;
            let layout = match HeaderLayout::validate(&prefix) {
                Ok(layout) => layout,
                Err(_) if prefix.iter().all(|byte| *byte == 0) => {
                    let layout = HeaderLayout::new(required_bits)?;
                    drop(prefix);
                    drop(file);
                    remove_stale_initializers()?;
                    crate::flock::materialize(notify_path())?;
                    file = initialize_header_file(&path, layout)?;
                    layout
                }
                Err(error) => return Err(error),
            };
            if required_bits > layout.bits {
                anyhow::bail!(
                    "queue registry host layout supports resource indices below {}, \
                     but this claim needs {}",
                    layout.bits,
                    required_bits
                );
            }
        }
        let header = unsafe { MmapMut::map_mut(&file) }?;
        let validated = HeaderLayout::validate(&header)?;
        Ok(Self {
            header,
            layout: validated,
            chunks: BTreeMap::new(),
        })
    }

    fn open_existing() -> Result<Self> {
        let path = header_path();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("open existing queue registry {}", path.display()))?;
        if file.metadata()?.len() == 0 {
            anyhow::bail!(
                "existing queue registry {} has no initialized header",
                path.display()
            );
        }
        let header = unsafe { MmapMut::map_mut(&file) }?;
        let layout = HeaderLayout::validate(&header)?;
        Ok(Self {
            header,
            layout,
            chunks: BTreeMap::new(),
        })
    }

    fn next_ticket(&self) -> Result<u64> {
        let ticket = read_u64(&self.header, H_NEXT_TICKET).max(1);
        if ticket == u64::MAX {
            anyhow::bail!("queue ticket id overflow");
        }
        Ok(ticket)
    }

    fn set_next_ticket(&mut self, ticket: u64) {
        write_u64(&mut self.header, H_NEXT_TICKET, ticket);
    }

    fn generation(&self) -> u64 {
        read_u64(&self.header, H_GENERATION)
    }

    fn generation_wake(&self) -> u32 {
        atomic_u32(&self.header, H_GENERATION_WAKE).load(Ordering::Acquire)
    }

    /// Advance the structural image version without waking pending-admission
    /// futex sleepers. This is valid when aggregate exact claims and physical
    /// availability cannot become less restrictive. Additions, queue-state
    /// publications, and negative observations can invalidate work, but cannot
    /// make a rejected physical preparation placement become runnable.
    fn advance_generation(&mut self) -> Result<()> {
        let next = self
            .generation()
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("queue registry generation exhausted"))?;
        write_u64(&mut self.header, H_GENERATION, next);
        Ok(())
    }

    /// Advance the structural image and wake every pending-admission sleeper.
    /// Exact-claim releases/replacements and physical-release observations can
    /// each make a different preparation candidate runnable; repair can expose
    /// either kind of improvement, so all such paths retain the broadcast.
    fn advance_generation_and_wake_pending(&mut self) -> Result<()> {
        self.advance_generation()?;
        let wake = atomic_u32(&self.header, H_GENERATION_WAKE);
        wake.fetch_add(1, Ordering::Release);
        // SAFETY: this is an aligned AtomicU32 in a MAP_SHARED registry
        // mapping. Wake every pending registrant because each may discover a
        // different physical preparation candidate made runnable by this
        // improvement when it rebuilds its probe after waking.
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                (wake as *const AtomicU32).cast::<u32>(),
                libc::FUTEX_WAKE,
                i32::MAX,
                std::ptr::null::<libc::timespec>(),
                std::ptr::null::<u32>(),
                0u32,
            );
        }
        Ok(())
    }

    fn note_queue_progress(&mut self) -> Result<()> {
        write_u64(&mut self.header, H_LAST_PROGRESS_NS, monotonic_now_ns()?);
        Ok(())
    }

    fn global_serial(&self) -> u64 {
        read_u64(&self.header, H_GLOBAL_SERIAL).max(1)
    }

    fn next_global_serial(&mut self) -> Result<u64> {
        let next = self
            .global_serial()
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("queue registry resource serial exhausted"))?;
        write_u64(&mut self.header, H_GLOBAL_SERIAL, next);
        Ok(next)
    }

    fn pending_flags(&self) -> u64 {
        read_u64(&self.header, H_PENDING_FLAGS)
    }

    fn replan_outstanding(&self) -> u64 {
        read_u64(&self.header, H_REPLAN_OUTSTANDING)
    }

    fn replan_capacity(&self) -> usize {
        usize::try_from(read_u64(&self.header, H_REPLAN_CAPACITY))
            .expect("validated queue registry planner capacity fits usize")
    }

    #[cfg(test)]
    fn set_replan_capacity_for_tests(&mut self, capacity: usize) -> Result<()> {
        anyhow::ensure!(
            capacity != 0 && capacity <= self.layout.bits,
            "test planner capacity {capacity} is outside 1..={}",
            self.layout.bits,
        );
        let encoded =
            u64::try_from(capacity).context("test planner capacity does not fit header")?;
        anyhow::ensure!(
            self.replan_outstanding() <= encoded,
            "cannot shrink test planner capacity below outstanding callbacks",
        );
        write_u64(&mut self.header, H_REPLAN_CAPACITY, encoded);
        Ok(())
    }

    fn replan_wave_started_ns(&self) -> u64 {
        read_u64(&self.header, H_REPLAN_WAVE_STARTED_NS)
    }

    fn replan_wave_deadline_ns(&self) -> u64 {
        read_u64(&self.header, H_REPLAN_WAVE_DEADLINE_NS)
    }

    fn arm_replan_wave_at(&mut self, now: u64) {
        let started = now.max(1);
        write_u64(&mut self.header, H_REPLAN_WAVE_STARTED_NS, started);
        write_u64(
            &mut self.header,
            H_REPLAN_WAVE_DEADLINE_NS,
            started.saturating_add(REPLAN_WAVE_LEASE_NS),
        );
    }

    fn clear_replan_wave_clock(&mut self) {
        write_u64(&mut self.header, H_REPLAN_WAVE_STARTED_NS, 0);
        write_u64(&mut self.header, H_REPLAN_WAVE_DEADLINE_NS, 0);
    }

    fn replan_wave_clock_valid_at(&self, now: u64) -> bool {
        let started = self.replan_wave_started_ns();
        let deadline = self.replan_wave_deadline_ns();
        started != 0 && started <= now && deadline >= started
    }

    fn replan_wave_due_at(&self, now: u64) -> bool {
        self.replan_outstanding() != 0
            && self.replan_wave_clock_valid_at(now)
            && self.replan_wave_deadline_ns() <= now
    }

    fn transition_replan_state(&mut self, old: u32, new: u32) -> Result<()> {
        if (old == STATE_REPLAN) == (new == STATE_REPLAN) {
            return Ok(());
        }
        let outstanding = self.replan_outstanding();
        let next = if new == STATE_REPLAN {
            outstanding.checked_add(1).ok_or_else(|| {
                anyhow::anyhow!("queue registry speculative callback count overflow")
            })?
        } else {
            outstanding.checked_sub(1).ok_or_else(|| {
                anyhow::anyhow!("queue registry speculative callback count underflow")
            })?
        };
        write_u64(&mut self.header, H_REPLAN_OUTSTANDING, next);
        Ok(())
    }

    fn finish_replan_state_publication(&mut self, old: u32, new: u32) {
        if old == STATE_REPLAN && new != STATE_REPLAN && self.replan_outstanding() == 0 {
            // Keep the deadline until after the last record state is durable.
            // Dirty repair can then distinguish a crash in the count-to-state
            // gap and quarantine a due callback instead of republishing it.
            self.clear_replan_wave_clock();
        }
    }

    /// Quarantine every callback which survived past the finite wave lease.
    /// Completed WAITING replacements are deliberately untouched. The single
    /// transaction drains the derived count, preserves one coalesced public
    /// rescan edge, and wakes each stale callback owner so it can acknowledge
    /// before this ticket is eligible again.
    fn expire_replan_wave_if_due_at(&mut self, now: u64) -> Result<bool> {
        let now = now.max(1);
        if self.replan_outstanding() == 0 {
            if self.replan_wave_started_ns() != 0 || self.replan_wave_deadline_ns() != 0 {
                self.begin_transaction()?;
                if self.replan_outstanding() == 0 {
                    self.clear_replan_wave_clock();
                }
                self.finish_transaction()?;
            }
            return Ok(false);
        }
        if !self.replan_wave_clock_valid_at(now) {
            // Monotonic time can move backwards only across boot. A torn
            // clock also cannot justify immediate quarantine, so give the
            // surviving live wave one complete lease from this observation.
            self.begin_transaction()?;
            if self.replan_outstanding() != 0 && !self.replan_wave_clock_valid_at(now) {
                self.arm_replan_wave_at(now);
            }
            self.finish_transaction()?;
            return Ok(false);
        }
        if !self.replan_wave_due_at(now) {
            return Ok(false);
        }

        self.begin_transaction()?;
        if !self.replan_wave_due_at(now) {
            self.finish_transaction()?;
            return Ok(false);
        }
        let expected = self.replan_outstanding();
        let records = self.scan_records()?;
        let mut expired = 0u64;
        for record in records.iter().filter(|record| record.state == STATE_REPLAN) {
            self.set_record_state(record.slot, STATE_REPLAN_EXPIRED)?;
            self.clear_record_blocked(record.slot)?;
            self.invalidate_record_prefix(record.slot)?;
            self.wake_slot(record.slot)?;
            expired = expired
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("queue registry expired callback count overflow"))?;
        }
        anyhow::ensure!(
            expired == expected && self.replan_outstanding() == 0,
            "queue registry REPLAN wave count mismatch during expiration: header={expected}, records={expired}",
        );
        self.clear_replan_wave_clock();
        self.schedule_rescan_edge_in_transaction()?;
        self.note_queue_progress()?;
        self.finish_transaction()?;
        Ok(true)
    }

    fn expire_replan_wave_if_due(&mut self) -> Result<bool> {
        if self.replan_outstanding() == 0
            && self.replan_wave_started_ns() == 0
            && self.replan_wave_deadline_ns() == 0
        {
            return Ok(false);
        }
        self.expire_replan_wave_if_due_at(monotonic_now_ns()?)
    }

    fn set_pending_flag(&mut self, flag: u64) {
        let flags = self.pending_flags() | flag;
        write_u64(&mut self.header, H_PENDING_FLAGS, flags);
    }

    fn clear_pending_flag(&mut self, flag: u64) {
        let flags = self.pending_flags() & !flag;
        write_u64(&mut self.header, H_PENDING_FLAGS, flags);
    }

    fn deferred_rescan_deadline_ns(&self) -> u64 {
        read_u64(&self.header, H_DEFERRED_RESCAN_DEADLINE_NS)
    }

    fn clear_deferred_rescan(&mut self) {
        self.clear_pending_flag(PENDING_REPLAN_RESCAN);
        write_u64(&mut self.header, H_DEFERRED_RESCAN_DEADLINE_NS, 0);
    }

    fn set_urgent_rescan(&mut self) {
        self.clear_deferred_rescan();
        self.set_pending_flag(PENDING_RESCAN);
    }

    fn deferred_rescan_due_at(&self, now: u64) -> bool {
        if self.pending_flags() & PENDING_REPLAN_RESCAN == 0 {
            return false;
        }
        let deadline = self.deferred_rescan_deadline_ns();
        // Zero is a torn/legacy publication. A deadline more than one complete
        // interval ahead can only have crossed a monotonic-clock epoch. Both
        // cases flush now instead of stranding authoritative work.
        deadline == 0
            || deadline <= now
            || deadline > now.saturating_add(deferred_rescan_interval_ns())
    }

    fn deferred_rescan_due_in_at(&self, now: u64) -> Option<Duration> {
        if self.pending_flags() & PENDING_REPLAN_RESCAN == 0 {
            return None;
        }
        if self.deferred_rescan_due_at(now) {
            return Some(Duration::ZERO);
        }
        Some(Duration::from_nanos(
            self.deferred_rescan_deadline_ns().saturating_sub(now),
        ))
    }

    fn deferred_rescan_due_in(&self) -> Result<Option<Duration>> {
        Ok(self.deferred_rescan_due_in_at(monotonic_now_ns()?))
    }

    fn prepare_grant_scan(&mut self) -> Result<bool> {
        self.prepare_grant_scan_at(monotonic_now_ns()?)
    }

    fn prepare_grant_scan_at(&mut self, now: u64) -> Result<bool> {
        if self.pending_flags() & PENDING_RESCAN != 0 {
            return Ok(true);
        }
        if self.pending_flags() & PENDING_REPLAN_RESCAN == 0 || !self.deferred_rescan_due_at(now) {
            return Ok(false);
        }

        self.promote_deferred_rescan()
    }

    fn promote_deferred_rescan(&mut self) -> Result<bool> {
        if self.pending_flags() & PENDING_RESCAN != 0 {
            return Ok(true);
        }
        if self.pending_flags() & PENDING_REPLAN_RESCAN == 0 {
            return Ok(false);
        }
        // Promote the edge in its own crash-recoverable transaction. An
        // interrupted coordinator therefore leaves either the durable
        // deferred edge or a durable urgent edge; it can never clear the only
        // publication in the gap before `grant_compatible` starts its scan.
        self.begin_transaction()?;
        if self.pending_flags() & PENDING_RESCAN == 0
            && self.pending_flags() & PENDING_REPLAN_RESCAN != 0
        {
            self.set_urgent_rescan();
        }
        self.finish_transaction()?;
        Ok(self.pending_flags() & PENDING_RESCAN != 0)
    }

    /// Publish one coordinator-consumed rescan edge while registry EX and a
    /// dirty transaction are held. Concurrent callback completions serialize
    /// here: only the first clear -> set transition requests a new transport
    /// notification, while later completions join the same pending scan. A
    /// missing coordinator is independently elected and target-woken even if
    /// the edge was already set. A completion which arrives after the scan
    /// clears the bit publishes the next transport edge.
    fn schedule_rescan_edge_in_transaction(&mut self) -> Result<bool> {
        let new_edge = self.pending_flags() & PENDING_RESCAN == 0;
        // Urgent work subsumes every deferred mutation visible under this EX
        // fence. Removing the deferred deadline here prevents a stale timer
        // from manufacturing a second scan after the urgent one completes.
        if new_edge {
            self.set_urgent_rescan();
        } else {
            self.clear_deferred_rescan();
        }
        let coordinator_before = self.coordinator_ticket();
        if self.coordinator_ticket() == 0 {
            // Election targets and wakes exactly one waiter, and publishes the
            // structural generation itself. A live coordinator instead wakes
            // through the single post-commit notify edge below.
            self.elect_coordinator_in_transaction()?;
        }
        let coordinator_elected = coordinator_before == 0 && self.coordinator_ticket() != 0;
        Ok(new_edge || coordinator_elected)
    }

    /// Coalesce a tail registration or REPLAN -> WAITING publication into the
    /// current speculative wave. The first deferred edge arms one absolute
    /// deadline; later producers never renew it. Without a live wave or
    /// coordinator, ordinary urgent publication/election preserves progress.
    fn schedule_deferred_replan_rescan_in_transaction(&mut self) -> Result<bool> {
        if self.replan_outstanding() == 0 || self.coordinator_ticket() == 0 {
            return self.schedule_rescan_edge_in_transaction();
        }
        if self.pending_flags() & PENDING_RESCAN != 0 {
            return Ok(false);
        }
        if self.pending_flags() & PENDING_REPLAN_RESCAN == 0 {
            let now = monotonic_now_ns()?.max(1);
            self.set_pending_flag(PENDING_REPLAN_RESCAN);
            write_u64(
                &mut self.header,
                H_DEFERRED_RESCAN_DEADLINE_NS,
                now.saturating_add(deferred_rescan_interval_ns()),
            );
        }
        Ok(false)
    }

    /// Publish the authoritative scan edge for one REPLAN -> WAITING
    /// completion without waking the same live coordinator once per callback.
    /// The coordinator's next heartbeat promotes an unnotified partial wave;
    /// draining the wave or electing a missing coordinator remains an immediate
    /// targeted wake even when another completion already set the deferred edge.
    fn schedule_replan_completion_edge_in_transaction(&mut self) -> Result<bool> {
        let drained = self.replan_outstanding() == 0;
        let notify = self.schedule_deferred_replan_rescan_in_transaction()?;
        // A drained wave is always promoted to the urgent transport above.
        // Return it explicitly even when an older urgent edge already exists:
        // the final callback is the drop-order boundary which guarantees its
        // payload/proof OFDs have closed before the coordinator samples them.
        Ok(drained || notify)
    }

    /// Batch GRANTED -> WAITING callback completions behind one short absolute
    /// deadline. Unlike speculative REPLAN batching, the first completion is
    /// transported immediately so an idle coordinator installs the timer;
    /// subsequent completions join the same edge without renewing it.
    fn schedule_grant_completion_edge_in_transaction(&mut self) -> Result<bool> {
        if self.coordinator_ticket() == 0 {
            return self.schedule_rescan_edge_in_transaction();
        }
        if self.pending_flags() & PENDING_RESCAN != 0 {
            return Ok(false);
        }
        let now = monotonic_now_ns()?.max(1);
        let deadline = now.saturating_add(GRANT_SCAN_COALESCE_INTERVAL_NS);
        if self.pending_flags() & PENDING_REPLAN_RESCAN != 0 {
            let current = self.deferred_rescan_deadline_ns();
            if current == 0 || deadline < current {
                write_u64(&mut self.header, H_DEFERRED_RESCAN_DEADLINE_NS, deadline);
                // A speculative partial wave may have omitted transport for
                // its later deadline. Publish the shortened hard boundary.
                return Ok(true);
            }
            return Ok(false);
        }
        self.set_pending_flag(PENDING_REPLAN_RESCAN);
        write_u64(&mut self.header, H_DEFERRED_RESCAN_DEADLINE_NS, deadline);
        Ok(true)
    }

    /// Coalesce a HELD-release grant rescan behind the same short
    /// [`GRANT_SCAN_COALESCE_INTERVAL_NS`] window the GRANTED batch drain uses,
    /// so a herd of completing cells drives one O(N) coordinator scan per
    /// quantum instead of one scan per release. A release only relaxes the
    /// prefix and its suffix fence is published unconditionally by the caller
    /// (`mark_claim_changed_metadata`), so deferring just the *scan* costs at
    /// most one quantum of grant latency — invisible next to a cell's whole run.
    ///
    /// This generalizes the record's own pre-existing "coalesce the O(N) scan
    /// while a speculative wave is live" intent to hold unconditionally: an
    /// already-armed shorter urgent/deferred edge is never lengthened (shortest
    /// deadline wins, matching the grant-completion batch). With no coordinator
    /// it keeps the historical urgent publication without an active-list
    /// election search — future registration or crash repair owns election,
    /// exactly as the pre-coalesce release did. `coalesced` reports whether this
    /// edge joined an already-pending scan rather than opening a fresh window,
    /// for the scan-cost diagnostic.
    fn schedule_coalesced_grant_rescan_in_transaction(&mut self) -> Result<()> {
        if self.coordinator_ticket() == 0 {
            self.set_urgent_rescan();
            return Ok(());
        }
        if self.pending_flags() & PENDING_RESCAN != 0 {
            note_grant_scan_coalesced();
            return Ok(());
        }
        let now = monotonic_now_ns()?.max(1);
        let deadline = now.saturating_add(GRANT_SCAN_COALESCE_INTERVAL_NS);
        if self.pending_flags() & PENDING_REPLAN_RESCAN != 0 {
            let current = self.deferred_rescan_deadline_ns();
            if current == 0 || deadline < current {
                write_u64(&mut self.header, H_DEFERRED_RESCAN_DEADLINE_NS, deadline);
            }
            note_grant_scan_coalesced();
            return Ok(());
        }
        self.set_pending_flag(PENDING_REPLAN_RESCAN);
        write_u64(&mut self.header, H_DEFERRED_RESCAN_DEADLINE_NS, deadline);
        Ok(())
    }

    fn observation_request_serial(&self) -> u64 {
        read_u64(&self.header, H_OBSERVATION_REQUEST).max(1)
    }

    fn bump_observation_request(&mut self) -> Result<u64> {
        let next = self
            .observation_request_serial()
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("queue registry observation serial exhausted"))?;
        write_u64(&mut self.header, H_OBSERVATION_REQUEST, next);
        self.set_pending_flag(PENDING_OBSERVATION);
        Ok(next)
    }

    fn resource_request(&self, which: usize, index: usize) -> Result<u64> {
        if index >= self.layout.bits {
            anyhow::bail!("resource index {index} exceeds queue registry capacity");
        }
        Ok(read_u64(
            &self.header,
            self.layout.request_offset(which) + index * 8,
        ))
    }

    fn set_resource_request(&mut self, which: usize, index: usize, serial: u64) -> Result<()> {
        if index >= self.layout.bits {
            anyhow::bail!("resource index {index} exceeds queue registry capacity");
        }
        write_u64(
            &mut self.header,
            self.layout.request_offset(which) + index * 8,
            serial,
        );
        Ok(())
    }

    fn bump_grant_scans(&mut self) {
        let next = read_u64(&self.header, H_GRANT_SCANS).wrapping_add(1);
        write_u64(&mut self.header, H_GRANT_SCANS, next);
    }

    fn claim_epoch(&self) -> u64 {
        read_u64(&self.header, H_CLAIM_EPOCH).max(1)
    }

    fn advance_claim_epoch(&mut self) -> Result<u64> {
        let next = self
            .claim_epoch()
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("queue registry claim epoch exhausted"))?;
        write_u64(&mut self.header, H_CLAIM_EPOCH, next);
        Ok(next)
    }

    fn min_changed_ticket(&self) -> u64 {
        read_u64(&self.header, H_MIN_CHANGED_TICKET)
    }

    /// The watermark consumed by unlicensed REPLAN acquire-commits: the
    /// minimum of the GRANTED-facing word and the always-dirtied replan word.
    /// A guard-skipped replacement is invisible to GRANTED entrants (its
    /// grant-disjointness proof covers them) but must still fence speculative
    /// commits whose claims carry no grant charge.
    fn min_changed_ticket_replan(&self) -> u64 {
        self.min_changed_ticket()
            .min(read_u64(&self.header, H_MIN_CHANGED_TICKET_REPLAN))
    }

    /// Dirty the suffix conservatively: no accompanying claim analysis, so
    /// saturate the changed-claims accumulator and park every later
    /// in-flight grant, exactly the pre-accumulator behavior. Every
    /// production site classifies itself through the `_fencing` /
    /// `_relaxation` variants; this stays the fail-closed entry point for
    /// fixtures (and for any future site that cannot classify its change).
    #[cfg(test)]
    fn mark_suffix_dirty(&mut self, ticket: u64) {
        write_u64(&mut self.header, H_CHANGED_CLAIMS_SATURATED, 1);
        self.mark_suffix_dirty_relaxation(ticket);
    }

    /// Dirty the suffix for a claim that becomes fenceable at (or before)
    /// the pending scan, folding it into the changed-claims accumulator so
    /// only the later grants that actually overlap it park at entry/commit.
    fn mark_suffix_dirty_fencing(&mut self, ticket: u64, claim: &impl ClaimView) -> Result<()> {
        self.mark_changed_claim(claim)?;
        self.mark_suffix_dirty_relaxation(ticket);
        Ok(())
    }

    /// Dirty the suffix for a pure fence retirement (record removal, revoke
    /// acknowledgement): the event only shrinks later tickets' predecessor
    /// prefixes, so no in-flight grant can be doomed by it and nothing joins
    /// the accumulator — disjointly granted juniors proceed. The watermark
    /// still moves: the epoch/scan bookkeeping and the unlicensed-REPLAN
    /// fence (the replan word) keep their existing consumers.
    fn mark_suffix_dirty_relaxation(&mut self, ticket: u64) {
        let minimum = self.min_changed_ticket().min(ticket);
        write_u64(&mut self.header, H_MIN_CHANGED_TICKET, minimum);
        self.mark_replan_suffix_dirty(ticket);
    }

    /// Fold one newly-fenceable claim into the changed-claims accumulator
    /// (permits at their folded CPU-space indices, exclusive modes into the
    /// EX maps). Grow-only until `finish_claim_scan` resets the set.
    fn mark_changed_claim(&mut self, claim: &impl ClaimView) -> Result<()> {
        for cpu in claim.cpus() {
            self.set_bitmap_bit(B_CHANGED_CPU_ANY, cpu, true)?;
            if claim.cpu_mode() == ClaimMode::Exclusive {
                self.set_bitmap_bit(B_CHANGED_CPU_EX, cpu, true)?;
            }
        }
        for permit in claim.permits() {
            let index = permit_resource_index(permit)?;
            self.set_bitmap_bit(B_CHANGED_CPU_ANY, index, true)?;
            if claim.permit_mode() == ClaimMode::Exclusive {
                self.set_bitmap_bit(B_CHANGED_CPU_EX, index, true)?;
            }
        }
        for llc in claim.llcs() {
            self.set_bitmap_bit(B_CHANGED_LLC_ANY, llc, true)?;
            if claim.llc_mode() == ClaimMode::Exclusive {
                self.set_bitmap_bit(B_CHANGED_LLC_EX, llc, true)?;
            }
        }
        Ok(())
    }

    fn changed_claims_saturated(&self) -> bool {
        read_u64(&self.header, H_CHANGED_CLAIMS_SATURATED) != 0
    }

    /// Whether `claim` overlaps any claim accumulated since the last
    /// authoritative scan, with the `first_conflict` mode matrix: an
    /// exclusive resource conflicts with any accumulated bit, a shared one
    /// only with an accumulated exclusive bit. O(claim) header bit reads —
    /// never a record walk.
    fn changed_claims_conflict(&self, claim: &ClaimSet) -> Result<bool> {
        for &cpu in &claim.cpus {
            let which = match claim.cpu_mode {
                ClaimMode::Exclusive => B_CHANGED_CPU_ANY,
                ClaimMode::Shared => B_CHANGED_CPU_EX,
            };
            if self.bitmap_bit(which, cpu)? {
                return Ok(true);
            }
        }
        for &permit in &claim.permits {
            let index = permit_resource_index(permit)?;
            let which = match claim.permit_mode {
                ClaimMode::Exclusive => B_CHANGED_CPU_ANY,
                ClaimMode::Shared => B_CHANGED_CPU_EX,
            };
            if self.bitmap_bit(which, index)? {
                return Ok(true);
            }
        }
        for &llc in &claim.llcs {
            let which = match claim.llc_mode {
                ClaimMode::Exclusive => B_CHANGED_LLC_ANY,
                ClaimMode::Shared => B_CHANGED_LLC_EX,
            };
            if self.bitmap_bit(which, llc)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn mark_replan_suffix_dirty(&mut self, ticket: u64) {
        let minimum = read_u64(&self.header, H_MIN_CHANGED_TICKET_REPLAN).min(ticket);
        write_u64(&mut self.header, H_MIN_CHANGED_TICKET_REPLAN, minimum);
    }

    /// Claim-epoch advance plus suffix dirty for a change whose
    /// newly-fenceable claim is known: accumulate it instead of saturating
    /// the accumulator.
    fn mark_claim_changed_fencing(&mut self, ticket: u64, claim: &impl ClaimView) -> Result<()> {
        self.advance_claim_epoch()?;
        self.mark_suffix_dirty_fencing(ticket, claim)?;
        self.set_urgent_rescan();
        Ok(())
    }

    /// Claim-epoch advance plus suffix dirty for a pure fence retirement,
    /// without the urgent-rescan edge (callers coalesce their own).
    fn mark_claim_changed_relaxation_metadata(&mut self, ticket: u64) -> Result<()> {
        self.advance_claim_epoch()?;
        self.mark_suffix_dirty_relaxation(ticket);
        Ok(())
    }

    /// [`Self::mark_claim_changed_relaxation_metadata`] plus the urgent
    /// rescan edge.
    fn mark_claim_changed_relaxation(&mut self, ticket: u64) -> Result<()> {
        self.mark_claim_changed_relaxation_metadata(ticket)?;
        self.set_urgent_rescan();
        Ok(())
    }

    fn finish_claim_scan(&mut self) {
        write_u64(&mut self.header, H_MIN_CHANGED_TICKET, u64::MAX);
        write_u64(&mut self.header, H_MIN_CHANGED_TICKET_REPLAN, u64::MAX);
        write_u64(&mut self.header, H_CHANGED_CLAIMS_SATURATED, 0);
        for which in [
            B_CHANGED_CPU_ANY,
            B_CHANGED_CPU_EX,
            B_CHANGED_LLC_ANY,
            B_CHANGED_LLC_EX,
        ] {
            let offset = self.layout.bitset_offset(which);
            self.header[offset..offset + self.layout.words * 8].fill(0);
        }
    }

    fn liveness_sweep(&self) -> u64 {
        read_u64(&self.header, H_LIVENESS_SWEEP)
    }

    fn bump_liveness_sweep(&mut self) {
        let next = self.liveness_sweep().wrapping_add(1);
        write_u64(&mut self.header, H_LIVENESS_SWEEP, next);
    }

    fn liveness_due_in(&self) -> Result<Duration> {
        Ok(liveness_due_in_from_header(
            &self.header,
            monotonic_now_ns()?,
        ))
    }

    /// Arrange one full liveness reconciliation after a newly elected
    /// coordinator has installed its inotify watch.
    ///
    /// A ticket can die after election but before that watch exists, so its
    /// CLOSE_WRITE is not observable. The deadline is shared by the whole
    /// registry and may only move earlier: a rapid chain of coordinators
    /// therefore coalesces into one O(N) sweep instead of performing one sweep
    /// per handoff or postponing recovery indefinitely.
    fn request_liveness_reconciliation(&mut self, delay: Duration) -> Result<()> {
        let now = monotonic_now_ns()?;
        let delay_ns = u64::try_from(delay.as_nanos()).unwrap_or(u64::MAX);
        // Rate-limit reconcile-triggered sweeps: every coordinator session
        // requests one, and a session churn storm would otherwise run the
        // O(records) prune (full liveness probe per record, under the
        // registry EX flock) every ~500ms. Clamping the deadline to a
        // minimum spacing after the LAST sweep keeps the pre-watch
        // death-recovery guarantee (the request is never dropped, only
        // deferred) while bounding the sweep rate well inside the ordinary
        // 30s liveness cadence.
        let floor = read_u64(&self.header, H_LAST_LIVENESS_SWEEP_NS)
            .saturating_add(LIVENESS_RECONCILE_MIN_INTERVAL_NS);
        let requested = now.saturating_add(delay_ns).max(floor).max(1);
        let current = read_u64(&self.header, H_LIVENESS_RECONCILE_BY_NS);
        if current == 0 || requested < current {
            write_u64(&mut self.header, H_LIVENESS_RECONCILE_BY_NS, requested);
        }
        Ok(())
    }

    fn perform_liveness_sweep_if_due(&mut self, force: bool) -> Result<Duration> {
        let now = monotonic_now_ns()?;
        let last = read_u64(&self.header, H_LAST_LIVENESS_SWEEP_NS);
        let reconcile_by = read_u64(&self.header, H_LIVENESS_RECONCILE_BY_NS);
        let reconcile_due = reconcile_by != 0 && reconcile_by <= now;
        if force || liveness_due_in_from_last(last, now).is_zero() || reconcile_due {
            self.prune_dead()?;
            write_u64(
                &mut self.header,
                H_LAST_LIVENESS_SWEEP_NS,
                monotonic_now_ns()?,
            );
            write_u64(&mut self.header, H_LIVENESS_RECONCILE_BY_NS, 0);
            self.bump_liveness_sweep();
        }
        self.liveness_due_in()
    }

    fn coordinator_ticket(&self) -> u64 {
        read_u64(&self.header, H_COORDINATOR)
    }

    fn coordinator_slot(&self) -> Result<u64> {
        let slot = read_u64(&self.header, H_COORDINATOR_SLOT);
        if slot != NONE_SLOT && slot >= self.next_slot()? {
            anyhow::bail!(
                "queue registry v{VERSION} coordinator slot {slot} is outside the active high-water mark"
            );
        }
        Ok(slot)
    }

    fn coordinator_epoch(&self) -> u64 {
        read_u64(&self.header, H_COORDINATOR_EPOCH).max(1)
    }

    fn touch_coordinator_heartbeat(&mut self) -> Result<()> {
        self.touch_coordinator_heartbeat_at(monotonic_now_ns()?);
        Ok(())
    }

    fn touch_coordinator_heartbeat_at(&mut self, now: u64) {
        write_u64(&mut self.header, H_COORDINATOR_HEARTBEAT_NS, now.max(1));
    }

    fn coordinator_heartbeat_due_in(&self) -> Result<Duration> {
        Ok(coordinator_heartbeat_due_in_from_header(
            &self.header,
            monotonic_now_ns()?,
        ))
    }

    fn set_coordinator(&mut self, ticket: u64, slot: u64) -> Result<()> {
        if (ticket == 0) != (slot == NONE_SLOT) {
            anyhow::bail!(
                "queue registry v{VERSION} coordinator identity is incomplete: ticket={ticket}, slot={slot}"
            );
        }
        write_u64(&mut self.header, H_COORDINATOR, ticket);
        write_u64(&mut self.header, H_COORDINATOR_SLOT, slot);
        if ticket == 0 {
            write_u64(&mut self.header, H_COORDINATOR_HEARTBEAT_NS, 0);
        } else {
            let epoch = self
                .coordinator_epoch()
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("queue coordinator epoch exhausted"))?;
            write_u64(&mut self.header, H_COORDINATOR_EPOCH, epoch);
            self.touch_coordinator_heartbeat()?;
            self.note_queue_progress()?;
        }
        Ok(())
    }

    fn next_slot(&self) -> Result<u64> {
        let next_slot = read_u64(&self.header, H_NEXT_SLOT);
        if next_slot > MAX_REGISTRY_SLOTS {
            anyhow::bail!(
                "queue registry v{VERSION} next-slot value {next_slot} exceeds the supported maximum {MAX_REGISTRY_SLOTS}"
            );
        }
        Ok(next_slot)
    }

    fn allocate_slot(&mut self) -> Result<u64> {
        let next_slot = self.next_slot()?;
        let free = read_u64(&self.header, H_FREE_HEAD);
        if free != NONE_SLOT {
            if free >= next_slot {
                anyhow::bail!(
                    "queue registry v{VERSION} free-list head {free} is outside 0..{next_slot}"
                );
            }
            let bytes = self.record_bytes(free)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "queue registry v{VERSION} free-list head {free} disappeared from 0..{next_slot}"
                )
            })?;
            let state = read_u32(bytes, R_STATE);
            let ticket = read_u64(bytes, R_TICKET);
            if state != STATE_FREE || ticket != 0 {
                anyhow::bail!(
                    "queue registry v{VERSION} free-list head {free} names an active or malformed \
                     record (state={state}, ticket={ticket})"
                );
            }
            let next = read_u64(bytes, R_NEXT_FREE);
            if next == free {
                anyhow::bail!(
                    "queue registry v{VERSION} free-list head {free} contains a self-cycle"
                );
            }
            if next != NONE_SLOT && next >= next_slot {
                anyhow::bail!(
                    "queue registry v{VERSION} free-list successor {next} is outside 0..{next_slot}"
                );
            }
            if next != NONE_SLOT {
                let successor = self.record_bytes(next)?.ok_or_else(|| {
                    anyhow::anyhow!(
                        "queue registry v{VERSION} free-list successor {next} disappeared from \
                         0..{next_slot}"
                    )
                })?;
                let successor_state = read_u32(successor, R_STATE);
                let successor_ticket = read_u64(successor, R_TICKET);
                if successor_state != STATE_FREE || successor_ticket != 0 {
                    anyhow::bail!(
                        "queue registry v{VERSION} free-list successor {next} names an active or \
                         malformed record (state={successor_state}, ticket={successor_ticket})"
                    );
                }
            }
            write_u64(&mut self.header, H_FREE_HEAD, next);
            return Ok(free);
        }
        let slot = next_slot;
        if slot >= MAX_REGISTRY_SLOTS {
            anyhow::bail!(
                "queue registry reached its supported high-water mark of {MAX_REGISTRY_SLOTS} concurrent slots"
            );
        }
        self.ensure_chunk(slot / RECORDS_PER_CHUNK as u64)?;
        let next = slot
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("queue registry slot counter overflow"))?;
        write_u64(&mut self.header, H_NEXT_SLOT, next);
        Ok(slot)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "record publication writes one fixed-layout queue record and its publication epochs as a unit"
    )]
    fn initialize_record(
        &mut self,
        slot: u64,
        ticket: u64,
        pid: u32,
        claim: &ClaimSet,
        watch: &ClaimSet,
        publish_state: u32,
        predecessors: &AggregateSnapshot,
        claim_epoch: u64,
        issue_serial: u64,
    ) -> Result<()> {
        let layout = self.layout;
        let bytes = self
            .record_bytes_mut(slot)?
            .ok_or_else(|| anyhow::anyhow!("new queue slot {slot} was not materialized"))?;
        let wake = atomic_u32_mut(bytes, R_WAKE)
            .load(Ordering::Relaxed)
            .wrapping_add(1);
        bytes.fill(0);
        atomic_u32_mut(bytes, R_WAKE).store(wake, Ordering::Release);
        write_u64(bytes, R_TICKET, ticket);
        write_u32(bytes, R_PID, pid);
        write_u32(
            bytes,
            R_WATCH_LLC_MODE,
            u32::from(watch.llc_mode == ClaimMode::Exclusive),
        );
        write_u32(
            bytes,
            R_CLAIM_LLC_MODE,
            u32::from(claim.llc_mode == ClaimMode::Exclusive),
        );
        write_u32(
            bytes,
            R_CLAIM_CPU_MODE,
            u32::from(claim.cpu_mode == ClaimMode::Exclusive),
        );
        write_u32(
            bytes,
            R_WATCH_CPU_MODE,
            u32::from(watch.cpu_mode == ClaimMode::Exclusive),
        );
        write_u32(
            bytes,
            R_CLAIM_PERMIT_MODE,
            u32::from(claim.permit_mode == ClaimMode::Exclusive),
        );
        write_u32(
            bytes,
            R_WATCH_PERMIT_MODE,
            u32::from(watch.permit_mode == ClaimMode::Exclusive),
        );
        write_u32(
            bytes,
            R_CLAIM_CLASS,
            encode_admission_class(claim.admission_class),
        );
        write_u32(
            bytes,
            R_WATCH_CLASS,
            encode_admission_class(watch.admission_class),
        );
        write_u32(
            bytes,
            R_BACKFILL_CAPACITY,
            backfill_capacity_for_watch(watch),
        );
        write_u64(bytes, R_BACKFILL_STARTED_NS, 0);
        write_u64(bytes, R_BLOCKED_SERIAL, 0);
        write_u64(bytes, R_NEXT_FREE, NONE_SLOT);
        write_u64(bytes, R_GRANT_EPOCH, 0);
        write_u32(bytes, R_BLOCK_KIND, BLOCK_NONE);
        write_u32(bytes, R_BLOCK_MODE, 0);
        write_u64(bytes, R_BLOCK_INDEX, 0);
        write_u64(bytes, R_ISSUE_SERIAL, issue_serial);
        write_u64(bytes, R_REPLAN_CLAIM_EPOCH, 0);
        write_u64(bytes, R_PREV_ACTIVE, NONE_SLOT);
        write_u64(bytes, R_NEXT_ACTIVE, NONE_SLOT);
        encode_claim(bytes, layout, claim, watch)?;
        let state_epoch_offset = if publish_state == STATE_GRANTED {
            R_GRANT_EPOCH
        } else {
            R_REPLAN_CLAIM_EPOCH
        };
        write_u64(bytes, state_epoch_offset, claim_epoch);
        encode_prefix(bytes, layout, predecessors)?;
        // The record itself is still FREE, so this final derived-cache stamp
        // precedes (and is covered by) the authoritative state publication.
        write_u64(bytes, R_PREFIX_EPOCH, claim_epoch);
        crash_at_for_tests("register_record_before_state_publish");
        // Publish the active state last. A killed writer therefore leaves
        // either a complete active record or a FREE record which dirty
        // recovery can discard; no partial claim becomes authoritative.
        write_u32(bytes, R_STATE, publish_state);
        Ok(())
    }

    fn append_active(&mut self, slot: u64) -> Result<()> {
        let head = read_u64(&self.header, H_ACTIVE_HEAD);
        let tail = read_u64(&self.header, H_ACTIVE_TAIL);
        if (head == NONE_SLOT) != (tail == NONE_SLOT) {
            anyhow::bail!(
                "queue registry v{VERSION} active-list endpoints are incomplete: head={head}, tail={tail}"
            );
        }
        {
            let bytes = self
                .record_bytes_mut(slot)?
                .ok_or_else(|| anyhow::anyhow!("new queue slot {slot} disappeared"))?;
            write_u64(bytes, R_PREV_ACTIVE, tail);
            write_u64(bytes, R_NEXT_ACTIVE, NONE_SLOT);
        }
        if tail == NONE_SLOT {
            write_u64(&mut self.header, H_ACTIVE_HEAD, slot);
        } else {
            let bytes = self.record_bytes_mut(tail)?.ok_or_else(|| {
                anyhow::anyhow!("queue active-list tail {tail} disappeared during append")
            })?;
            write_u64(bytes, R_NEXT_ACTIVE, slot);
        }
        write_u64(&mut self.header, H_ACTIVE_TAIL, slot);
        Ok(())
    }

    fn unlink_active(&mut self, record: &Record) -> Result<()> {
        if record.prev_active == NONE_SLOT {
            write_u64(&mut self.header, H_ACTIVE_HEAD, record.next_active);
        } else {
            let bytes = self.record_bytes_mut(record.prev_active)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "queue active predecessor {} disappeared",
                    record.prev_active
                )
            })?;
            write_u64(bytes, R_NEXT_ACTIVE, record.next_active);
        }
        if record.next_active == NONE_SLOT {
            write_u64(&mut self.header, H_ACTIVE_TAIL, record.prev_active);
        } else {
            let bytes = self.record_bytes_mut(record.next_active)?.ok_or_else(|| {
                anyhow::anyhow!("queue active successor {} disappeared", record.next_active)
            })?;
            write_u64(bytes, R_PREV_ACTIVE, record.prev_active);
        }
        Ok(())
    }

    /// Bytes of a slot that holds a record in a valid active state.
    ///
    /// Single-sources the state whitelist for both decode paths; every
    /// non-FREE state must appear here or that decode silently rejects
    /// live records.
    #[inline]
    fn active_record_bytes(&mut self, slot: u64) -> Result<Option<&[u8]>> {
        let Some(bytes) = self.record_bytes(slot)? else {
            return Ok(None);
        };
        let state = read_u32(bytes, R_STATE);
        if state == STATE_FREE {
            return Ok(None);
        }
        if !matches!(
            state,
            STATE_WAITING
                | STATE_GRANTED
                | STATE_REPLAN
                | STATE_COORDINATOR
                | STATE_HELD
                | STATE_COORDINATOR_STANDBY
                | STATE_PENDING
                | STATE_REVOKED
                | STATE_REPLAN_EXPIRED
        ) {
            anyhow::bail!("queue registry v{VERSION} slot {slot} has invalid state {state}");
        }
        Ok(Some(bytes))
    }

    fn record(&mut self, slot: u64) -> Result<Option<Record>> {
        let layout = self.layout;
        let Some(bytes) = self.active_record_bytes(slot)? else {
            return Ok(None);
        };
        let record = decode_record(bytes, layout, slot)?;
        if record.ticket == 0
            || record.claim.is_empty()
            || (record.state == STATE_PENDING && record.watch.is_empty())
        {
            anyhow::bail!(
                "queue registry v{VERSION} slot {slot} is active with ticket {} and an {} claim",
                record.ticket,
                if record.claim.is_empty() {
                    "empty"
                } else {
                    "invalid"
                }
            );
        }
        Ok(Some(record))
    }

    fn scan_record(&mut self, slot: u64) -> Result<Option<ScanRecord>> {
        let layout = self.layout;
        let Some(bytes) = self.active_record_bytes(slot)? else {
            return Ok(None);
        };
        let record = decode_scan_record(bytes, layout, slot)?;
        if record.ticket == 0 || record.claim.is_empty() {
            anyhow::bail!(
                "queue registry v{VERSION} slot {slot} is active with ticket {} and an empty claim",
                record.ticket,
            );
        }
        Ok(Some(record))
    }

    /// Traverse the authoritative active list without expanding every
    /// alternative watch into thousands of BTree nodes. Strict ticket order
    /// already proves uniqueness, so this also avoids a redundant ticket set.
    fn scan_records(&mut self) -> Result<Vec<ScanRecord>> {
        let mut records = Vec::new();
        let next_slot = self.next_slot()?;
        let head = read_u64(&self.header, H_ACTIVE_HEAD);
        let tail = read_u64(&self.header, H_ACTIVE_TAIL);
        if (head == NONE_SLOT) != (tail == NONE_SLOT) {
            anyhow::bail!(
                "queue registry v{VERSION} active-list endpoints are incomplete: head={head}, tail={tail}"
            );
        }
        let mut slot = head;
        let mut previous = NONE_SLOT;
        let mut previous_ticket = 0;
        while slot != NONE_SLOT {
            if records.len() >= usize::try_from(next_slot).unwrap_or(usize::MAX) {
                anyhow::bail!("queue registry v{VERSION} active list contains a cycle");
            }
            #[cfg(test)]
            ACTIVE_LIST_RECORD_READS.with(|reads| reads.set(reads.get() + 1));
            let record = self.scan_record(slot)?.ok_or_else(|| {
                anyhow::anyhow!("queue registry v{VERSION} active slot {slot} is free")
            })?;
            if record.prev_active != previous {
                anyhow::bail!(
                    "queue registry v{VERSION} active slot {slot} has prev={}, expected {previous}",
                    record.prev_active
                );
            }
            if record.ticket <= previous_ticket {
                anyhow::bail!(
                    "queue registry v{VERSION} active tickets are not strictly increasing at {}",
                    record.ticket
                );
            }
            previous = slot;
            previous_ticket = record.ticket;
            slot = record.next_active;
            records.push(record);
        }
        if previous != tail {
            anyhow::bail!(
                "queue registry v{VERSION} active-list tail is {tail}, traversal ended at {previous}"
            );
        }
        Ok(records)
    }

    fn record_identity_is(&mut self, slot: u64, ticket: u64, state: u32) -> Result<bool> {
        let Some(bytes) = self.record_bytes(slot)? else {
            return Ok(false);
        };
        Ok(read_u64(bytes, R_TICKET) == ticket && read_u32(bytes, R_STATE) == state)
    }

    /// Strict ticket order proves uniqueness, so no ticket set is needed here.
    fn records(&mut self) -> Result<Vec<Record>> {
        let mut records = Vec::new();
        let next_slot = self.next_slot()?;
        let head = read_u64(&self.header, H_ACTIVE_HEAD);
        let tail = read_u64(&self.header, H_ACTIVE_TAIL);
        if (head == NONE_SLOT) != (tail == NONE_SLOT) {
            anyhow::bail!(
                "queue registry v{VERSION} active-list endpoints are incomplete: head={head}, tail={tail}"
            );
        }
        let mut slot = head;
        let mut previous = NONE_SLOT;
        let mut previous_ticket = 0;
        while slot != NONE_SLOT {
            if records.len() >= usize::try_from(next_slot).unwrap_or(usize::MAX) {
                anyhow::bail!("queue registry v{VERSION} active list contains a cycle");
            }
            #[cfg(test)]
            ACTIVE_LIST_RECORD_READS.with(|reads| reads.set(reads.get() + 1));
            let record = self.record(slot)?.ok_or_else(|| {
                anyhow::anyhow!("queue registry v{VERSION} active slot {slot} is free")
            })?;
            if record.prev_active != previous {
                anyhow::bail!(
                    "queue registry v{VERSION} active slot {slot} has prev={}, expected {previous}",
                    record.prev_active
                );
            }
            if record.ticket <= previous_ticket {
                anyhow::bail!(
                    "queue registry v{VERSION} active tickets are not strictly increasing at {}",
                    record.ticket
                );
            }
            previous = slot;
            previous_ticket = record.ticket;
            slot = record.next_active;
            records.push(record);
        }
        if previous != tail {
            anyhow::bail!(
                "queue registry v{VERSION} active-list tail is {tail}, traversal ended at {previous}"
            );
        }
        Ok(records)
    }

    fn aggregate_claim_snapshot(&self) -> AggregateSnapshot {
        AggregateSnapshot {
            bits: self.layout.bits,
            cpu_any: self.header_words(B_CLAIM_CPUS),
            cpu_exclusive: self.header_words(B_CLAIM_CPU_EXCLUSIVE),
            llc_any: self.header_words(B_CLAIM_LLC_ANY),
            llc_exclusive: self.header_words(B_CLAIM_LLC_EXCLUSIVE),
            cpu_shared_holders: self.header_counts(B_HELD_CPU_SHARED),
            cpu_exclusive_holders: self.header_counts(B_HELD_CPU_EXCLUSIVE),
            llc_shared_holders: self.header_counts(B_HELD_LLC_SHARED),
            llc_exclusive_holders: self.header_counts(B_HELD_LLC_EXCLUSIVE),
            build_cpu_claims: self.header_counts(C_BUILD_CLAIM_CPUS),
            cpu_grant_any: self.header_counts(C_GRANT_CPU_ANY),
            cpu_grant_exclusive: self.header_counts(C_GRANT_CPU_EX),
            llc_grant_any: self.header_counts(C_GRANT_LLC_ANY),
            llc_grant_exclusive: self.header_counts(C_GRANT_LLC_EX),
        }
    }

    fn availability_snapshot(&self) -> AvailabilitySnapshot {
        AvailabilitySnapshot {
            bits: self.layout.bits,
            cpu_known: self.header_words(B_CPU_KNOWN),
            cpu_sh_available: self.header_words(B_CPU_SH_AVAILABLE),
            cpu_ex_available: self.header_words(B_CPU_EX_AVAILABLE),
            llc_known: self.header_words(B_LLC_KNOWN),
            llc_sh_available: self.header_words(B_LLC_SH_AVAILABLE),
            llc_ex_available: self.header_words(B_LLC_EX_AVAILABLE),
        }
    }

    fn header_words(&self, which: usize) -> Vec<u64> {
        (0..self.layout.words)
            .map(|word| {
                read_u64(
                    &self.header,
                    self.layout.bitset_offset(which) + word * std::mem::size_of::<u64>(),
                )
            })
            .collect()
    }

    fn header_counts(&self, which: usize) -> Vec<u32> {
        (0..self.layout.bits)
            .map(|index| {
                read_u32(
                    &self.header,
                    self.layout.count_offset(which) + index * std::mem::size_of::<u32>(),
                )
            })
            .collect()
    }

    /// Copy the scan-time predecessor reservation cached in this ticket's
    /// record. This is O(host words), independent of queue depth.
    fn cached_prefix(&mut self, slot: u64) -> Result<(u64, AggregateSnapshot)> {
        let layout = self.layout;
        let bytes = self
            .record_bytes(slot)?
            .ok_or_else(|| anyhow::anyhow!("queue slot {slot} disappeared during prefix read"))?;
        #[cfg(test)]
        GRANT_PREFIX_RECORD_READS.with(|reads| reads.set(reads.get() + 1));
        let read_words = |which| {
            (0..layout.words)
                .map(|word| {
                    read_u64(
                        bytes,
                        record_bitset_offset(layout, which) + word * std::mem::size_of::<u64>(),
                    )
                })
                .collect()
        };
        Ok((
            read_u64(bytes, R_PREFIX_EPOCH),
            AggregateSnapshot {
                bits: layout.bits,
                cpu_any: read_words(RB_PREFIX_CPU_ANY),
                cpu_exclusive: read_words(RB_PREFIX_CPU_EXCLUSIVE),
                llc_any: read_words(RB_PREFIX_LLC_ANY),
                llc_exclusive: read_words(RB_PREFIX_LLC_EXCLUSIVE),
                cpu_shared_holders: vec![0; layout.bits],
                cpu_exclusive_holders: vec![0; layout.bits],
                llc_shared_holders: vec![0; layout.bits],
                llc_exclusive_holders: vec![0; layout.bits],
                build_cpu_claims: vec![0; layout.bits],
                cpu_grant_any: vec![0; layout.bits],
                cpu_grant_exclusive: vec![0; layout.bits],
                llc_grant_any: vec![0; layout.bits],
                llc_grant_exclusive: vec![0; layout.bits],
            },
        ))
    }

    /// Whether `claim` overlaps any live `C_GRANT_*` charge (GRANTED and
    /// REVOKED records), read directly from the header count arrays under the
    /// registry EX transaction — O(claim), no snapshot copy, no TOCTOU. Mode
    /// matrix matches [`AggregateSnapshot::grant_conflicts`].
    fn claim_grant_conflicts_header(&self, claim: &ClaimSet) -> Result<bool> {
        let count = |which: usize, index: usize| -> Result<u32> {
            if index >= self.layout.bits {
                anyhow::bail!("resource index {index} exceeds queue registry capacity");
            }
            Ok(read_u32(
                &self.header,
                self.layout.count_offset(which) + index * std::mem::size_of::<u32>(),
            ))
        };
        for &cpu in &claim.cpus {
            let which = match claim.cpu_mode {
                ClaimMode::Exclusive => C_GRANT_CPU_ANY,
                ClaimMode::Shared => C_GRANT_CPU_EX,
            };
            if count(which, cpu)? != 0 {
                return Ok(true);
            }
        }
        for &permit in &claim.permits {
            let index = permit_resource_index(permit)?;
            let which = match claim.permit_mode {
                ClaimMode::Exclusive => C_GRANT_CPU_ANY,
                ClaimMode::Shared => C_GRANT_CPU_EX,
            };
            if count(which, index)? != 0 {
                return Ok(true);
            }
        }
        for &llc in &claim.llcs {
            let which = match claim.llc_mode {
                ClaimMode::Exclusive => C_GRANT_LLC_ANY,
                ClaimMode::Shared => C_GRANT_LLC_EX,
            };
            if count(which, llc)? != 0 {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Copy the live `C_GRANT_*` charge into a planning snapshot, optionally
    /// subtracting one record's own charged claim (the
    /// `claim_conflicts_aggregate_excluding` count-subtraction pattern): a
    /// licensed GRANTED callback replanning must not soft-avoid its own
    /// designated footprint.
    fn fill_grant_occupancy(
        &self,
        snapshot: &mut AggregateSnapshot,
        exclude: Option<&ClaimSet>,
    ) -> Result<()> {
        anyhow::ensure!(
            snapshot.bits == self.layout.bits,
            "queue grant-occupancy snapshot layout does not match this registry",
        );
        snapshot.cpu_grant_any = self.header_counts(C_GRANT_CPU_ANY);
        snapshot.cpu_grant_exclusive = self.header_counts(C_GRANT_CPU_EX);
        snapshot.llc_grant_any = self.header_counts(C_GRANT_LLC_ANY);
        snapshot.llc_grant_exclusive = self.header_counts(C_GRANT_LLC_EX);
        let Some(claim) = exclude else {
            return Ok(());
        };
        let subtract = |counts: &mut [u32], index: usize| {
            let count = counts.get_mut(index).ok_or_else(|| {
                anyhow::anyhow!("resource index {index} exceeds queue registry capacity")
            })?;
            *count = count.checked_sub(1).ok_or_else(|| {
                anyhow::anyhow!("queue grant charge omitted the excluded claim at resource {index}")
            })?;
            Ok::<_, anyhow::Error>(())
        };
        for &cpu in &claim.cpus {
            subtract(&mut snapshot.cpu_grant_any, cpu)?;
            if claim.cpu_mode == ClaimMode::Exclusive {
                subtract(&mut snapshot.cpu_grant_exclusive, cpu)?;
            }
        }
        for &permit in &claim.permits {
            let index = permit_resource_index(permit)?;
            subtract(&mut snapshot.cpu_grant_any, index)?;
            if claim.permit_mode == ClaimMode::Exclusive {
                subtract(&mut snapshot.cpu_grant_exclusive, index)?;
            }
        }
        for &llc in &claim.llcs {
            subtract(&mut snapshot.llc_grant_any, llc)?;
            if claim.llc_mode == ClaimMode::Exclusive {
                subtract(&mut snapshot.llc_grant_exclusive, llc)?;
            }
        }
        Ok(())
    }

    fn cached_prefix_matches_words(
        &mut self,
        slot: u64,
        cpu_any: &[u64],
        cpu_exclusive: &[u64],
        llc_any: &[u64],
        llc_exclusive: &[u64],
    ) -> Result<bool> {
        #[cfg(test)]
        PREFIX_COMPARE_RECORD_READS.with(|reads| reads.set(reads.get() + 1));
        let layout = self.layout;
        if [cpu_any, cpu_exclusive, llc_any, llc_exclusive]
            .iter()
            .any(|words| words.len() != layout.words)
        {
            anyhow::bail!("queue prefix comparison used mismatched registry words");
        }
        let bytes = self.record_bytes(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during prefix comparison")
        })?;
        for (which, expected) in [
            (RB_PREFIX_CPU_ANY, cpu_any),
            (RB_PREFIX_CPU_EXCLUSIVE, cpu_exclusive),
            (RB_PREFIX_LLC_ANY, llc_any),
            (RB_PREFIX_LLC_EXCLUSIVE, llc_exclusive),
        ] {
            let offset = record_bitset_offset(layout, which);
            for (word, expected) in expected.iter().copied().enumerate() {
                if read_u64(bytes, offset + word * std::mem::size_of::<u64>()) != expected {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn record_prefix_epoch(&mut self, slot: u64) -> Result<u64> {
        let bytes = self.record_bytes(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during prefix epoch read")
        })?;
        Ok(read_u64(bytes, R_PREFIX_EPOCH))
    }

    fn invalidate_record_prefix(&mut self, slot: u64) -> Result<()> {
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during prefix invalidation")
        })?;
        write_u64(bytes, R_PREFIX_EPOCH, 0);
        Ok(())
    }

    #[cfg(test)]
    fn set_record_backfill_capacity(&mut self, slot: u64, capacity: u32) -> Result<()> {
        let maximum = self
            .record(slot)?
            .map(|record| backfill_capacity_for_watch(&record.watch))
            .ok_or_else(|| {
                anyhow::anyhow!("queue slot {slot} disappeared during backfill-capacity update")
            })?;
        if capacity > maximum {
            anyhow::bail!(
                "queue backfill capacity {capacity} exceeds the ticket's resource-weighted maximum \
                 {maximum}"
            );
        }
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during backfill-capacity update")
        })?;
        write_u32(bytes, R_BACKFILL_CAPACITY, capacity);
        Ok(())
    }

    fn set_record_backfill_started_ns(&mut self, slot: u64, started_ns: u64) -> Result<()> {
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during backfill-age update")
        })?;
        write_u64(bytes, R_BACKFILL_STARTED_NS, started_ns);
        Ok(())
    }

    /// Publish one complete predecessor cache. The epoch is invalidated first
    /// and is the last word published, so a killed writer cannot expose torn
    /// bitsets as authoritative callback input.
    fn publish_prefix(
        &mut self,
        slot: u64,
        prefix: &AggregateSnapshot,
        state_epoch_offset: usize,
        epoch: u64,
        issue_serial: u64,
    ) -> Result<()> {
        if epoch == 0 || prefix.bits != self.layout.bits {
            anyhow::bail!("invalid queue prefix publication epoch or layout");
        }
        #[cfg(test)]
        FULL_PREFIX_SNAPSHOT_PUBLISHES.with(|count| count.set(count.get().saturating_add(1)));
        self.publish_prefix_words(
            slot,
            &prefix.cpu_any,
            &prefix.cpu_exclusive,
            &prefix.llc_any,
            &prefix.llc_exclusive,
            state_epoch_offset,
            epoch,
            issue_serial,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "prefix publication writes four encoded predecessor sets and one callback token"
    )]
    fn publish_prefix_words(
        &mut self,
        slot: u64,
        cpu_any: &[u64],
        cpu_exclusive: &[u64],
        llc_any: &[u64],
        llc_exclusive: &[u64],
        state_epoch_offset: usize,
        epoch: u64,
        issue_serial: u64,
    ) -> Result<()> {
        if epoch == 0
            || [cpu_any, cpu_exclusive, llc_any, llc_exclusive]
                .iter()
                .any(|words| words.len() != self.layout.words)
        {
            anyhow::bail!("invalid queue prefix publication epoch or layout");
        }
        let layout = self.layout;
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during prefix publish")
        })?;
        write_u64(bytes, R_PREFIX_EPOCH, 0);
        crash_at_for_tests("prefix_invalidated_before_copy");
        for (which, words) in [
            (RB_PREFIX_CPU_ANY, cpu_any),
            (RB_PREFIX_CPU_EXCLUSIVE, cpu_exclusive),
            (RB_PREFIX_LLC_ANY, llc_any),
            (RB_PREFIX_LLC_EXCLUSIVE, llc_exclusive),
        ] {
            let offset = record_bitset_offset(layout, which);
            for (word, value) in words.iter().copied().enumerate() {
                write_u64(bytes, offset + word * std::mem::size_of::<u64>(), value);
            }
        }
        crash_at_for_tests("prefix_copied_before_epoch");
        write_u64(bytes, state_epoch_offset, epoch);
        write_u64(bytes, R_ISSUE_SERIAL, issue_serial);
        write_u64(bytes, R_PREFIX_EPOCH, epoch);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "claim replacement carries the complete transactional record transition"
    )]
    #[cfg(test)]
    fn replace_claim(
        &mut self,
        slot: u64,
        ticket: u64,
        old: &ClaimSet,
        new: &ClaimSet,
        publish_state: u32,
        issue_serial: u64,
        blocked: Option<(ContentionMarker, u64)>,
        persist_blocker: bool,
    ) -> Result<()> {
        self.begin_transaction()?;
        self.replace_claim_in_transaction(
            slot,
            ticket,
            old,
            new,
            publish_state,
            issue_serial,
            blocked,
            persist_blocker,
            ReplacementFenceEffect::ChangesPredecessorPrefix,
        )?;
        self.finish_transaction()?;
        Ok(())
    }

    /// Publish a replacement selected by one non-acquiring REPLAN callback.
    /// Both the old REPLAN and new WAITING record are non-fencing. Publish the
    /// rescan edge atomically, while batching its transport notification with
    /// the rest of the live speculative wave.
    #[expect(
        clippy::too_many_arguments,
        reason = "speculative replacement carries the complete record transition"
    )]
    fn replace_replan_claim_and_schedule(
        &mut self,
        slot: u64,
        ticket: u64,
        old: &ClaimSet,
        new: &ClaimSet,
        issue_serial: u64,
        blocked: Option<(ContentionMarker, u64)>,
        persist_blocker: bool,
    ) -> Result<bool> {
        self.begin_transaction()?;
        self.replace_claim_in_transaction(
            slot,
            ticket,
            old,
            new,
            STATE_WAITING,
            issue_serial,
            blocked,
            persist_blocker,
            ReplacementFenceEffect::NonFencing,
        )?;
        // Publish WAITING before election so a completion which outlived its
        // coordinator is itself eligible to take over immediately.
        let notify = self.schedule_replan_completion_edge_in_transaction()?;
        self.finish_transaction()?;
        Ok(notify)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "claim replacement carries the complete transactional record transition"
    )]
    fn replace_claim_in_transaction(
        &mut self,
        slot: u64,
        ticket: u64,
        old: &ClaimSet,
        new: &ClaimSet,
        publish_state: u32,
        issue_serial: u64,
        blocked: Option<(ContentionMarker, u64)>,
        persist_blocker: bool,
        fence_effect: ReplacementFenceEffect,
    ) -> Result<()> {
        let prior = self
            .record(slot)?
            .filter(|record| record.ticket == ticket)
            .ok_or_else(|| anyhow::anyhow!("queue slot {slot} disappeared during replacement"))?;
        if matches!(fence_effect, ReplacementFenceEffect::NonFencing) {
            anyhow::ensure!(
                prior.state == STATE_REPLAN && publish_state == STATE_WAITING,
                "non-fencing replacement requires REPLAN -> WAITING, got {} -> {} for ticket {ticket}",
                record_state_name(prior.state),
                record_state_name(publish_state),
            );
        }
        let replan_claim_epoch = prior.replan_claim_epoch;
        self.clear_record_blocked(slot)?;
        self.adjust_claim_counts(old, false)?;
        self.adjust_claim_counts(new, true)?;
        debug_assert!(
            !charged_state(publish_state),
            "claim replacement cannot publish a grant-charged state directly",
        );
        if charged_state(prior.state) && !charged_state(publish_state) {
            // A GRANTED record replacing its claim (the licensed-replan
            // ChangesPredecessorPrefix path) rewrites claim bytes before the
            // state flip, so the chokepoint cannot see the OLD claim. Release
            // the old charge here; the published state is never charged.
            self.adjust_granted_occupancy(old, false)?;
        }
        crash_at_for_tests("replace_counts_before_record");
        let layout = self.layout;
        {
            let bytes = self
                .record_bytes_mut(slot)?
                .ok_or_else(|| anyhow::anyhow!("queue slot {slot} disappeared"))?;
            write_u32(bytes, R_STATE, STATE_FREE);
            clear_record_claim_bits(bytes, layout);
            // Immutable watch modes cover every alternative. Exact modes
            // follow the designated candidate independently for each resource
            // class.
            write_u32(
                bytes,
                R_CLAIM_LLC_MODE,
                u32::from(new.llc_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_CLAIM_CPU_MODE,
                u32::from(new.cpu_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_CLAIM_PERMIT_MODE,
                u32::from(new.permit_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_CLAIM_CLASS,
                encode_admission_class(new.admission_class),
            );
            encode_exact_claim(bytes, layout, new)?;
            ScanMetadata::for_claims(new, &prior.watch)?.write(bytes);
            write_u64(bytes, R_BLOCKED_SERIAL, 0);
            write_u32(bytes, R_BLOCK_KIND, BLOCK_NONE);
            write_u32(bytes, R_BLOCK_MODE, 0);
            write_u64(bytes, R_BLOCK_INDEX, 0);
            write_u64(bytes, R_ISSUE_SERIAL, issue_serial);
            write_u64(bytes, R_REPLAN_CLAIM_EPOCH, replan_claim_epoch);
            write_u64(bytes, R_BACKFILL_STARTED_NS, 0);
        }
        if persist_blocker && let Some((evidence, serial)) = blocked {
            self.set_record_blocked(slot, evidence, serial)?;
            self.mark_blocker_unknown(evidence)?;
        }
        self.transition_replan_state(prior.state, publish_state)?;
        match fence_effect {
            ReplacementFenceEffect::ChangesPredecessorPrefix => {
                self.mark_claim_changed_fencing(ticket, new)?;
            }
            ReplacementFenceEffect::NonFencing => {
                // REPLAN and WAITING are both non-fencing, so this replacement
                // must not invalidate every live callback through the global
                // claim epoch. It can become a new predecessor fence in the
                // next authoritative scan, though: dirty precisely the later
                // ticket suffix so no old GRANTED callback can enter or commit
                // across that scan edge.
                //
                // Grant-disjointness damping: the GRANTED consumers of the
                // main watermark exist so an in-flight grant cannot enter or
                // commit across a scan edge that could revoke it. The set of
                // claims a pending scan can newly conflict with is exactly
                // the fence-preserving states {GRANTED, REVOKED, HELD}; a
                // replacement whose FULL new claim (not the delta — a kept
                // permit overlapping a junior grant must still park it) is
                // disjoint from every GRANTED and REVOKED charge can doom no
                // in-flight grant, and HELD overlap cannot revoke a grant
                // either, so parking juniors would serve neither the proof
                // nor ordering. Unlicensed REPLAN acquire-commits carry no
                // grant charge and stay fenced through the always-dirtied
                // replan word.
                if self.claim_grant_conflicts_header(new)? {
                    self.mark_suffix_dirty_fencing(ticket, new)?;
                } else {
                    self.mark_replan_suffix_dirty(ticket);
                }
            }
        }
        crash_at_for_tests("replace_record_before_state_publish");
        let bytes = self
            .record_bytes_mut(slot)?
            .ok_or_else(|| anyhow::anyhow!("queue slot {slot} disappeared"))?;
        write_u32(bytes, R_STATE, publish_state);
        self.finish_replan_state_publication(prior.state, publish_state);
        // Replacing a publication which participates in predecessor prefixes
        // advances the suffix watermark even though its new state is WAITING.
        // REPLAN and its completed WAITING replacement are both non-fencing;
        // their next authoritative scan introduces any new fence and revokes
        // conflicting later grants atomically.
        Ok(())
    }

    fn promote_record_to_held(
        &mut self,
        record: &Record,
        exact: &ClaimSet,
        contention: &[ContentionMarker],
    ) -> Result<()> {
        validate_claim(exact)?;
        self.begin_transaction()?;
        self.mark_blockers_unknown(contention)?;
        self.clear_record_blocked(record.slot)?;
        // A same-wake re-designation promotes directly from REPLAN to HELD, so
        // this drains its speculative-callback ring slot in the same
        // transaction that publishes ownership. Non-REPLAN callers (GRANTED /
        // COORDINATOR promotions) take the no-op early return.
        self.transition_replan_state(record.state, STATE_HELD)?;
        let claim_changed = record.claim != *exact;
        if claim_changed {
            self.adjust_claim_counts(&record.claim, false)?;
            self.adjust_claim_counts(exact, true)?;
        }
        if charged_state(record.state) {
            // HELD is deliberately uncharged (the held counts take over), and
            // the promotion may rewrite claim bytes, so release the OLD claim
            // charge explicitly rather than through the chokepoint.
            self.adjust_granted_occupancy(&record.claim, false)?;
        }
        self.adjust_watch_counts(&record.watch, false)?;
        self.adjust_held_counts(exact, true)?;
        self.publish_claim_busy(exact)?;
        crash_at_for_tests("held_counts_before_record");
        let layout = self.layout;
        let mut held_watch = record.watch.clone();
        held_watch.cpus.clear();
        held_watch.llcs.clear();
        held_watch.permits.clear();
        held_watch.cpu_mode = ClaimMode::Exclusive;
        held_watch.llc_mode = ClaimMode::Exclusive;
        held_watch.permit_mode = ClaimMode::Exclusive;
        {
            let bytes = self
                .record_bytes_mut(record.slot)?
                .ok_or_else(|| anyhow::anyhow!("queue slot {} disappeared", record.slot))?;
            // HELD removes the immutable callback watch even when its exact
            // claim is unchanged. Publish FREE before either rewrite so dirty
            // repair never decodes a half-HELD active record.
            write_u32(bytes, R_STATE, STATE_FREE);
            if claim_changed {
                clear_record_claim_bits(bytes, layout);
            }
            clear_record_watch_bits(bytes, layout);
            write_u32(bytes, R_WATCH_LLC_MODE, 1);
            write_u32(bytes, R_WATCH_CPU_MODE, 1);
            write_u32(bytes, R_WATCH_PERMIT_MODE, 1);
            crash_at_for_tests("promote_held_state_free_before_record");
            if claim_changed {
                write_u32(
                    bytes,
                    R_CLAIM_LLC_MODE,
                    u32::from(exact.llc_mode == ClaimMode::Exclusive),
                );
                write_u32(
                    bytes,
                    R_CLAIM_CPU_MODE,
                    u32::from(exact.cpu_mode == ClaimMode::Exclusive),
                );
                write_u32(
                    bytes,
                    R_CLAIM_PERMIT_MODE,
                    u32::from(exact.permit_mode == ClaimMode::Exclusive),
                );
                write_u32(
                    bytes,
                    R_CLAIM_CLASS,
                    encode_admission_class(exact.admission_class),
                );
                encode_exact_claim(bytes, layout, exact)?;
            }
            ScanMetadata::for_claims(exact, &held_watch)?.write(bytes);
            write_u64(bytes, R_BLOCKED_SERIAL, 0);
            write_u32(bytes, R_BLOCK_KIND, BLOCK_NONE);
            write_u32(bytes, R_BLOCK_MODE, 0);
            write_u64(bytes, R_BLOCK_INDEX, 0);
            write_u32(bytes, R_STATE, STATE_HELD);
        }
        self.finish_replan_state_publication(record.state, STATE_HELD);
        crate::vmm::grant_flow::note_grant_reached_held();
        if claim_changed {
            self.mark_claim_changed_fencing(record.ticket, exact)?;
        }
        if self.coordinator_ticket() == record.ticket {
            self.set_coordinator(0, NONE_SLOT)?;
            self.elect_coordinator_in_transaction()?;
        }
        if claim_changed {
            self.advance_generation_and_wake_pending()?;
        } else {
            self.advance_generation()?;
        }
        self.finish_transaction()?;
        Ok(())
    }

    /// Atomically park a selected run ticket on the physical preparation
    /// footprint acquired by its callback.
    ///
    /// PENDING claims exactly the resources backed by live preparation OFDs.
    /// The selected final intent remains in the record's watch and retains the
    /// same ticket order, but does not sequester CPUs, LLCs, or run permits
    /// while immutable artifacts are prepared. Exact activation replaces both
    /// fields in one transaction without a release/re-register window.
    fn transition_record_to_pending(
        &mut self,
        record: &Record,
        selected_final: &ClaimSet,
        preparation: &ClaimSet,
        contention: &[ContentionMarker],
    ) -> Result<PendingTransition> {
        validate_claim(selected_final)?;
        validate_claim(preparation)?;
        validate_claim_within_watch(selected_final, &record.watch)?;
        anyhow::ensure!(
            !preparation.is_empty(),
            "granted intent produced an empty preparation claim",
        );
        let pending_claim = preparation.clone();
        let pending_watch = pending_intent_watch(selected_final, preparation);
        validate_claim(&pending_claim)?;
        validate_claim_within_watch(&pending_claim, &pending_watch)?;
        materialize_claim_paths(&pending_watch)?;
        // The ticket already earned this grant against its predecessor prefix.
        // Later tickets may publish while its physical preparation probe runs,
        // but they cannot retroactively veto the older grant. Recompute the
        // prefix under the commit lock and fence only genuine predecessors.
        let predecessors = self.cached_prefix(record.slot)?.1;
        if let Some(marker) = predecessors.first_conflict(&pending_claim)? {
            return Ok(PendingTransition::Contended(marker));
        }

        self.begin_transaction()?;
        self.mark_blockers_unknown(contention)?;
        self.clear_record_blocked(record.slot)?;
        self.adjust_claim_counts(&record.claim, false)?;
        if charged_state(record.state) {
            // Release the OLD (granted) claim's charge, not the preparation
            // claim: PENDING is deliberately uncharged, and this transition
            // rewrites claim bytes so the chokepoint cannot balance it.
            self.adjust_granted_occupancy(&record.claim, false)?;
        }
        self.adjust_watch_counts(&record.watch, false)?;
        let newly_watched = self.newly_watched(&pending_watch)?;
        self.adjust_claim_counts(&pending_claim, true)?;
        self.adjust_watch_counts(&pending_watch, true)?;
        self.mark_observation_modes(&newly_watched)?;
        let issue_serial = self.max_watch_serial(&pending_watch)?;
        let layout = self.layout;
        {
            let bytes = self
                .record_bytes_mut(record.slot)?
                .ok_or_else(|| anyhow::anyhow!("queue slot {} disappeared", record.slot))?;
            write_u32(bytes, R_STATE, STATE_FREE);
            clear_record_claim_bits(bytes, layout);
            clear_record_watch_bits(bytes, layout);
            write_u32(
                bytes,
                R_CLAIM_LLC_MODE,
                u32::from(pending_claim.llc_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_CLAIM_CPU_MODE,
                u32::from(pending_claim.cpu_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_CLAIM_PERMIT_MODE,
                u32::from(pending_claim.permit_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_WATCH_LLC_MODE,
                u32::from(pending_watch.llc_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_WATCH_CPU_MODE,
                u32::from(pending_watch.cpu_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_WATCH_PERMIT_MODE,
                u32::from(pending_watch.permit_mode == ClaimMode::Exclusive),
            );
            write_u32(
                bytes,
                R_CLAIM_CLASS,
                encode_admission_class(pending_claim.admission_class),
            );
            write_u32(
                bytes,
                R_WATCH_CLASS,
                encode_admission_class(pending_watch.admission_class),
            );
            encode_claim(bytes, layout, &pending_claim, &pending_watch)?;
            write_u64(bytes, R_ISSUE_SERIAL, issue_serial);
            write_u64(bytes, R_GRANT_EPOCH, 0);
            write_u64(bytes, R_REPLAN_CLAIM_EPOCH, 0);
            write_u64(bytes, R_PREFIX_EPOCH, 0);
            write_u32(
                bytes,
                R_BACKFILL_CAPACITY,
                backfill_capacity_for_watch(&pending_watch),
            );
            write_u64(bytes, R_BACKFILL_STARTED_NS, 0);
            write_u64(bytes, R_BLOCKED_SERIAL, 0);
            write_u32(bytes, R_BLOCK_KIND, BLOCK_NONE);
            write_u32(bytes, R_BLOCK_MODE, 0);
            write_u64(bytes, R_BLOCK_INDEX, 0);
            write_u32(bytes, R_STATE, STATE_PENDING);
        }
        self.mark_claim_changed_fencing(record.ticket, &pending_claim)?;
        if self.coordinator_ticket() == record.ticket {
            self.set_coordinator(0, NONE_SLOT)?;
            self.elect_coordinator_in_transaction()?;
        }
        self.advance_generation_and_wake_pending()?;
        self.finish_transaction()?;
        Ok(PendingTransition::Committed(pending_claim))
    }

    fn remove_record(&mut self, record: &Record, acquired: bool) -> Result<()> {
        let removed_coordinator = self.coordinator_ticket() == record.ticket;
        self.begin_transaction()?;
        if acquired {
            self.publish_claim_busy(&record.claim)?;
        }
        self.remove_record_in_transaction(record, acquired)?;
        if !acquired {
            // Removal is a monotonic prefix relaxation. Preserve its claim
            // epoch/suffix watermark immediately (without joining the
            // changed-claims accumulator — a shrinking prefix dooms no
            // in-flight grant), but coalesce the O(N) scan behind one short
            // quantum. Physical release proof is independently observed and
            // promotes urgent work.
            self.mark_claim_changed_relaxation_metadata(record.ticket)?;
        }
        crash_at_for_tests("remove_record_before_election");
        if removed_coordinator {
            self.elect_coordinator_in_transaction()?;
        }
        if !acquired {
            if removed_coordinator {
                // The explicit election above already searched once. Publish
                // the urgent scan edge without repeating the full active-list
                // walk when that search found no successor.
                self.set_urgent_rescan();
            } else if matches!(record.state, STATE_GRANTED | STATE_REVOKED | STATE_HELD) {
                // A real resource release. A draining herd completes en masse,
                // and one urgent scan per completion makes the coordinator run
                // an O(N) grant scan per release (O(N^2) drain). Coalesce these
                // behind the shared short quantum so a burst drives one scan,
                // whether or not a speculative wave is live (shortest deadline
                // wins — a freed resource should rescan promptly, matching the
                // grant-completion batch). Work-conserving: the pending scan
                // still grants everything grantable, and the per-release
                // coordinator notify guarantees a release arriving after a scan
                // cleared the edge schedules the next one.
                self.schedule_coalesced_grant_rescan_in_transaction()?;
            } else if self.replan_outstanding() != 0 && self.coordinator_ticket() != 0 {
                // A non-resource teardown (a WAITING/REPLAN ticket cancelling)
                // frees only a watch, so it rides the live speculative wave
                // without accelerating it.
                self.schedule_deferred_replan_rescan_in_transaction()?;
            } else {
                // Preserve the old callback-only drain invariant: publishing
                // urgent work does not need to search a shrinking active list
                // for a coordinator which is known not to exist. Normal drop
                // still emits the transport edge, while future registration
                // or crash repair owns coordinator election.
                self.set_urgent_rescan();
            }
        }
        self.finish_transaction()?;
        Ok(())
    }

    fn remove_record_in_transaction(&mut self, record: &Record, acquired: bool) -> Result<()> {
        if !acquired && matches!(record.state, STATE_GRANTED | STATE_REVOKED | STATE_HELD) {
            // A granted callback probes without EX and may die while
            // holding its exact fds, before committing acquired publication.
            // REVOKED deliberately covers the same in-flight window. A HELD
            // owner is pruned only after its physical fds have closed (normal
            // RAII teardown or process death), so all three states may
            // represent a genuine compatibility improvement.
            self.mark_possible_release(
                &record.claim.cpus,
                &record.claim.llcs,
                &record.claim.permits,
            )?;
        }
        self.clear_record_blocked(record.slot)?;
        self.adjust_claim_counts(&record.claim, false)?;
        if charged_state(record.state) {
            // Drop, cancel, and crash prune all remove charged records
            // without a demoting state flip; release the grant charge in the
            // same transaction that frees the record.
            self.adjust_granted_occupancy(&record.claim, false)?;
        }
        if record.state == STATE_HELD {
            self.adjust_held_counts(&record.claim, false)?;
        } else {
            self.adjust_watch_counts(&record.watch, false)?;
        }
        crash_at_for_tests("remove_counts_before_free");
        if self.coordinator_ticket() == record.ticket {
            self.set_coordinator(0, NONE_SLOT)?;
        }
        self.unlink_active(record)?;
        let free_head = read_u64(&self.header, H_FREE_HEAD);
        self.transition_replan_state(record.state, STATE_FREE)?;
        let bytes = self
            .record_bytes_mut(record.slot)?
            .ok_or_else(|| anyhow::anyhow!("queue slot {} disappeared", record.slot))?;
        write_u32(bytes, R_STATE, STATE_FREE);
        write_u64(bytes, R_TICKET, 0);
        write_u64(bytes, R_NEXT_FREE, free_head);
        write_u64(&mut self.header, H_FREE_HEAD, record.slot);
        self.finish_replan_state_publication(record.state, STATE_FREE);
        Ok(())
    }

    /// Decode only the exact claim of one record: modes plus the `RB_CLAIM_*`
    /// bitsets. Deliberately skips the watch — expanding an alternative watch
    /// into BTree nodes on every state flip would materialize thousands of
    /// nodes per scan (the lazy-watch invariant the active-list traversal
    /// exists to preserve).
    fn record_charge_claim(&mut self, slot: u64) -> Result<ClaimSet> {
        let layout = self.layout;
        let bytes = self
            .record_bytes(slot)?
            .ok_or_else(|| anyhow::anyhow!("queue slot {slot} disappeared during charge read"))?;
        let llc_mode = decode_mode(bytes, R_CLAIM_LLC_MODE, slot, "exact LLC claim")?;
        let cpu_mode = decode_mode(bytes, R_CLAIM_CPU_MODE, slot, "exact CPU claim")?;
        let permit_mode = decode_mode(bytes, R_CLAIM_PERMIT_MODE, slot, "exact permit claim")?;
        Ok(ClaimSet::with_all_claim_modes(
            decode_bitset(bytes, record_bitset_offset(layout, RB_CLAIM_LLCS), layout)?,
            decode_bitset(bytes, record_bitset_offset(layout, RB_CLAIM_CPUS), layout)?,
            decode_bitset(
                bytes,
                record_bitset_offset(layout, RB_CLAIM_PERMITS),
                layout,
            )?,
            llc_mode,
            cpu_mode,
            permit_mode,
        ))
    }

    fn set_record_state(&mut self, slot: u64, state: u32) -> Result<()> {
        let old = self
            .record_bytes(slot)?
            .map(|bytes| read_u32(bytes, R_STATE))
            .ok_or_else(|| anyhow::anyhow!("queue slot {slot} disappeared during state update"))?;
        self.transition_replan_state(old, state)?;
        // Grant-charge chokepoint. Every state-only transition into or out of
        // the charged set {GRANTED, REVOKED} flows through here, so charging
        // is keyed purely on the persisted claim bytes at flip time. Callers
        // must not mutate a record's claim bytes before this state flip in
        // the same transaction — replace/promote/pending rewrite claims and
        // therefore handle their own charge swaps explicitly, bypassing this
        // path. GRANTED -> REVOKED stays a no-op (both charged).
        if charged_state(old) != charged_state(state) {
            let claim = self.record_charge_claim(slot)?;
            self.adjust_granted_occupancy(&claim, charged_state(state))?;
        }
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during state publication")
        })?;
        write_u32(bytes, R_STATE, state);
        self.finish_replan_state_publication(old, state);
        Ok(())
    }

    /// Retire the exact predecessor fence left by a revoked callback.
    ///
    /// The REVOKED publication and this acknowledgement are deliberately two
    /// transactions: the callback may own real OFDs while outside registry
    /// EX. Only its own next state read (or rejected run entry) can prove that
    /// it will not begin that stale probe. Publishing WAITING and RESCAN in one
    /// dirty transaction makes the successor scan crash-recoverable.
    fn acknowledge_revoked(
        &mut self,
        slot: u64,
        ticket: u64,
        blocked: Option<(ContentionMarker, u64, u64)>,
    ) -> Result<AcknowledgeResult> {
        let Some(record) = self.record(slot)?.filter(|record| record.ticket == ticket) else {
            return Ok(AcknowledgeResult::UNCHANGED);
        };
        if record.state != STATE_REVOKED {
            return Ok(AcknowledgeResult::UNCHANGED);
        }
        self.begin_transaction()?;
        let record = self
            .record(slot)?
            .filter(|record| record.ticket == ticket && record.state == STATE_REVOKED)
            .ok_or_else(|| anyhow::anyhow!("revoked queue ticket {ticket} changed during ack"))?;
        self.set_record_state(record.slot, STATE_WAITING)?;
        self.clear_record_blocked(record.slot)?;
        if claim_is_flexible(&record.claim, &record.watch) {
            // Revocation can represent a deterioration of only this exact
            // designation, which deliberately does not advance an
            // improvement serial. The old GRANTED predecessor cache may
            // still match after its fence is acknowledged. Force the fresh
            // WAITING ticket to reconsider an already-available alternative
            // without requiring unrelated queue or resource churn.
            self.invalidate_record_prefix(record.slot)?;
        }
        if let Some((marker, serial, consumed_serial)) = blocked {
            self.set_record_blocked(record.slot, marker, serial)?;
            self.set_record_issue_serial(record.slot, consumed_serial)?;
            self.mark_blocker_unknown(marker)?;
        }
        // Removing the revoked exact fence changes every later ticket's
        // predecessor prefix. Give the successor scan a new publication epoch
        // so an already-running callback cannot mistake a refreshed, different
        // prefix for its original token. The change is a pure relaxation —
        // the prefix only shrinks — so it does not join the changed-claims
        // accumulator and dooms no in-flight grant.
        let notify = self.schedule_rescan_edge_in_transaction()?;
        self.mark_claim_changed_relaxation(record.ticket)?;
        self.finish_transaction()?;
        Ok(AcknowledgeResult {
            acknowledged: true,
            notify,
        })
    }

    /// Make one expired speculative ticket eligible only after its owner has
    /// observed the quarantine. Expiration is non-fencing, so unlike REVOKED
    /// acknowledgement this does not publish a claim change. Invalidating the
    /// ticket's own callback token plus one rescan edge makes it reconsider
    /// the complete immutable watch. The pending scan can turn this ticket
    /// into a new predecessor fence, so later callbacks are fenced by the
    /// non-epoch suffix watermark until that scan completes.
    fn acknowledge_expired_replan(
        &mut self,
        slot: u64,
        ticket: u64,
        blocked: Option<(ContentionMarker, u64, u64)>,
    ) -> Result<AcknowledgeResult> {
        let Some(record) = self.record(slot)?.filter(|record| record.ticket == ticket) else {
            return Ok(AcknowledgeResult::UNCHANGED);
        };
        if record.state != STATE_REPLAN_EXPIRED {
            return Ok(AcknowledgeResult::UNCHANGED);
        }
        self.begin_transaction()?;
        let record = self
            .record(slot)?
            .filter(|record| record.ticket == ticket && record.state == STATE_REPLAN_EXPIRED)
            .ok_or_else(|| {
                anyhow::anyhow!("expired REPLAN queue ticket {ticket} changed during ack")
            })?;
        self.mark_suffix_dirty_fencing(record.ticket, &record.claim)?;
        self.set_record_state(record.slot, STATE_WAITING)?;
        self.clear_record_blocked(record.slot)?;
        self.invalidate_record_prefix(record.slot)?;
        if let Some((marker, serial, consumed_serial)) = blocked {
            self.set_record_blocked(record.slot, marker, serial)?;
            self.set_record_issue_serial(record.slot, consumed_serial)?;
            self.mark_blocker_unknown(marker)?;
        }
        let notify = self.schedule_rescan_edge_in_transaction()?;
        self.finish_transaction()?;
        Ok(AcknowledgeResult {
            acknowledged: true,
            notify,
        })
    }

    fn clear_record_blocked(&mut self, slot: u64) -> Result<()> {
        let record = self.record(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during blocker update")
        })?;
        let external = Self::external_blocker(&record);
        let owns_transaction = external.is_some()
            && atomic_u64(&self.header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) == 0;
        if owns_transaction {
            self.begin_transaction()?;
        }
        if let Some(marker) = external {
            self.adjust_external_event_watch(marker, false)?;
        }
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during blocker update")
        })?;
        write_u64(bytes, R_BLOCKED_SERIAL, 0);
        write_u32(bytes, R_BLOCK_KIND, BLOCK_NONE);
        write_u32(bytes, R_BLOCK_MODE, 0);
        write_u64(bytes, R_BLOCK_INDEX, 0);
        if owns_transaction {
            self.finish_transaction()?;
        }
        Ok(())
    }

    /// Grant scans already decoded whether the blocker contributes a separate
    /// event-watch reference. Clear it without expanding the record's full
    /// immutable watch a second time.
    fn clear_record_blocked_known(
        &mut self,
        slot: u64,
        external: Option<ContentionMarker>,
    ) -> Result<()> {
        if let Some(marker) = external {
            self.adjust_external_event_watch(marker, false)?;
        }
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during blocker update")
        })?;
        write_u64(bytes, R_BLOCKED_SERIAL, 0);
        write_u32(bytes, R_BLOCK_KIND, BLOCK_NONE);
        write_u32(bytes, R_BLOCK_MODE, 0);
        write_u64(bytes, R_BLOCK_INDEX, 0);
        Ok(())
    }

    fn set_record_blocked(
        &mut self,
        slot: u64,
        evidence: ContentionMarker,
        serial: u64,
    ) -> Result<()> {
        let record = self.record(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during blocker update")
        })?;
        let old_external = Self::external_blocker(&record);
        let new_external =
            (!contention_marker_within_watch(evidence, &record.watch)).then_some(evidence);
        let owns_transaction = old_external != new_external
            && atomic_u64(&self.header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) == 0;
        if owns_transaction {
            self.begin_transaction()?;
        }
        if old_external != new_external {
            if let Some(marker) = old_external {
                self.adjust_external_event_watch(marker, false)?;
            }
            if let Some(marker) = new_external {
                self.adjust_external_event_watch(marker, true)?;
            }
        }
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during blocker update")
        })?;
        write_u64(bytes, R_BLOCKED_SERIAL, serial);
        let (kind, index) = match evidence.blocker {
            ResourceKey::Cpu(index) => (BLOCK_CPU, index),
            ResourceKey::Llc(index) => (BLOCK_LLC, index),
            ResourceKey::Permit(index) => (BLOCK_PERMIT, index),
        };
        write_u32(bytes, R_BLOCK_KIND, kind);
        write_u32(
            bytes,
            R_BLOCK_MODE,
            u32::from(evidence.mode == FlockMode::Exclusive),
        );
        write_u64(
            bytes,
            R_BLOCK_INDEX,
            u64::try_from(index).context("blocked resource index does not fit registry record")?,
        );
        if owns_transaction {
            self.finish_transaction()?;
        }
        Ok(())
    }

    fn external_blocker(record: &Record) -> Option<ContentionMarker> {
        record.blocked_on.and_then(|blocked| {
            let marker = ContentionMarker {
                blocker: blocked.key,
                mode: blocked.mode,
            };
            (!contention_marker_within_watch(marker, &record.watch)).then_some(marker)
        })
    }

    fn adjust_external_event_watch(&mut self, marker: ContentionMarker, add: bool) -> Result<()> {
        let mut watch = ClaimSet::default();
        match marker.blocker {
            ResourceKey::Cpu(cpu) => {
                watch.cpus.insert(cpu);
                watch.cpu_mode = ClaimMode::from(marker.mode);
            }
            ResourceKey::Llc(llc) => {
                watch.llcs.insert(llc);
                watch.llc_mode = ClaimMode::from(marker.mode);
            }
            ResourceKey::Permit(permit) => {
                watch.permits.insert(permit);
                watch.permit_mode = ClaimMode::from(marker.mode);
            }
        }
        self.adjust_watch_counts(&watch, add)
    }

    fn set_record_issue_serial(&mut self, slot: u64, serial: u64) -> Result<()> {
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during issue-serial update")
        })?;
        write_u64(bytes, R_ISSUE_SERIAL, serial);
        Ok(())
    }

    #[cfg(test)]
    fn set_record_grant_epoch(&mut self, slot: u64, epoch: u64) -> Result<()> {
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during grant-epoch update")
        })?;
        write_u64(bytes, R_GRANT_EPOCH, epoch);
        Ok(())
    }

    fn wake_slot(&mut self, slot: u64) -> Result<()> {
        let bytes = self
            .record_bytes_mut(slot)?
            .ok_or_else(|| anyhow::anyhow!("queue slot {slot} disappeared during wake"))?;
        let atom = atomic_u32_mut(bytes, R_WAKE);
        atom.fetch_add(1, Ordering::Release);
        // SAFETY: the address is an aligned shared futex word.  Waking while
        // the registry flock is held closes grant-before-wake crash gaps.
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                (atom as *const AtomicU32).cast::<u32>(),
                libc::FUTEX_WAKE,
                1i32,
                std::ptr::null::<libc::timespec>(),
                std::ptr::null::<u32>(),
                0u32,
            );
        }
        Ok(())
    }

    fn map_ticket_shared(&mut self, slot: u64, ticket: u64) -> Result<TicketSharedMaps> {
        TicketSharedMaps::open(self, slot, ticket)
    }

    #[cfg(test)]
    fn grant_compatible(&mut self) -> Result<(ClaimSet, bool)> {
        self.grant_compatible_at(monotonic_now_ns()?.max(1), None)
    }

    fn grant_compatible_with_tokens(
        &mut self,
        preparation_tokens: Option<&std::ops::Range<usize>>,
    ) -> Result<(ClaimSet, bool)> {
        self.grant_compatible_at(monotonic_now_ns()?.max(1), preparation_tokens)
    }

    // `preparation_tokens` is the preparation-slot permit sub-range, passed in
    // by the caller that owns the topology so the registry stays index-opaque
    // and does its pool arithmetic on given budget data. `None` disables the
    // pool budget (non-preparation callers and tests).
    fn grant_compatible_at(
        &mut self,
        backfill_now_ns: u64,
        preparation_tokens: Option<&std::ops::Range<usize>>,
    ) -> Result<(ClaimSet, bool)> {
        struct BackfillHead {
            claim: ScanClaim,
            available: u32,
            admission_open: bool,
        }

        struct ReplanCandidate {
            slot: u64,
            ticket: u64,
            waiting_serial: u64,
            external_blocker: Option<ContentionMarker>,
            cpu_any: Vec<u64>,
            cpu_exclusive: Vec<u64>,
            llc_any: Vec<u64>,
            llc_exclusive: Vec<u64>,
        }

        // The scan publishes a mutually dependent set of prefix caches,
        // grants/revocations, REPLAN cursor state, and wakes. A scanner dying
        // after activating an earlier fence but before revoking a conflicting
        // later grant must not leave that later callback runnable from a stale
        // clean snapshot. Dirty recovery conservatively demotes every
        // uncommitted callback publication before any successor can enter it.
        self.begin_transaction()?;
        let records = self.scan_records()?;
        let backfill_now_ns = backfill_now_ns.max(1);
        if self.replan_outstanding() != 0 && !self.replan_wave_clock_valid_at(backfill_now_ns) {
            self.arm_replan_wave_at(backfill_now_ns);
        } else if self.replan_outstanding() == 0 {
            self.clear_replan_wave_clock();
        }
        let mut cpu_any = vec![0u64; self.layout.words];
        let mut cpu_exclusive = vec![0u64; self.layout.words];
        let mut llc_any = vec![0u64; self.layout.words];
        let mut llc_exclusive = vec![0u64; self.layout.words];
        let claim_epoch = self.claim_epoch();
        let mut scan_publication_epoch = claim_epoch;
        let coordinator_ticket = self.coordinator_ticket();
        // Preparation-slot pool budget. The scan grants at most the currently
        // free token count, in ticket order, so the oldest eligible waiter is
        // guaranteed the next freed slot and the granted cohort never exceeds
        // the pool it will physically race (so no granted waiter loses the race
        // and needs a per-token blocked pin). `preparation_tokens_consumed`
        // accrues tokens already held by fenced PENDING records — their
        // acquired token is in the claim, so `add_claim_bits` and this counter
        // see it — plus one per preparation grant this scan, because a WAITING
        // intent's run claim carries no token yet and the scan must count its
        // own not-yet-physical grants. The coordinator is the privileged FIFO
        // head: if it is itself a token-needing preparation intent still racing
        // (not yet PENDING/HELD), one slot is reserved out of the budget so a
        // junior budgeted waiter can never win the slot ahead of it.
        let preparation_pool = preparation_tokens.map_or(0, |range| range.len());
        // Tokens already held by fenced PENDING/REVOKED records vs.
        // reservations for this scan's own grants.
        let mut prep_pending_tokens = 0usize;
        let mut prep_granted_tokens = 0usize;
        let coordinator_reserve = usize::from(
            preparation_tokens.is_some()
                && records.iter().any(|record| {
                    record.ticket == coordinator_ticket
                        && record.preparation_intent
                        && !matches!(record.state, STATE_PENDING | STATE_HELD)
                }),
        );
        let preparation_tokens_in_claim = |claim: &ScanClaim| -> usize {
            preparation_tokens.map_or(0, |range| {
                claim
                    .permits()
                    .filter(|permit| range.contains(permit))
                    .count()
            })
        };
        let global_serial = self.global_serial();
        // REPLAN is a speculative planner callback, not a resource claim. A
        // host-width window bounds concurrent placement computation while
        // exact grants and admitted VMs retain their independent resource
        // capacity. Each callback completion publishes one coalesced rescan
        // edge, so every scan refills the available portion of this window.
        let replan_outstanding = usize::try_from(self.replan_outstanding())
            .context("queue registry speculative callback count does not fit usize")?;
        let replan_capacity = self.replan_capacity();
        anyhow::ensure!(
            replan_outstanding <= replan_capacity,
            "queue registry has {replan_outstanding} speculative callbacks for planner capacity {replan_capacity}",
        );
        let replan_slots = replan_capacity - replan_outstanding;
        let replan_cursor = read_u64(&self.header, H_REPLAN_CURSOR);
        let replan_lease_active = replan_outstanding != 0;
        // Production entry expires a due lease before this scan. Preserve the
        // boundary if the clock crosses its deadline between those two
        // monotonic samples instead of publishing immediately-expired work.
        let replan_publication_open =
            !replan_lease_active || !self.replan_wave_due_at(backfill_now_ns);
        let mut replan_batch_started = false;
        let mut replan_wake_slots = Vec::new();
        // Records are scanned in ticket order because their predecessor
        // prefixes are order-sensitive. Retain the cyclic tail first and only
        // enough wrapped-head fallbacks to fill the remaining window. REPLAN
        // is non-fencing, so deferring these state publications until after
        // the scan cannot alter any later ticket's compatibility decision.
        let mut replan_tail = Vec::<ReplanCandidate>::new();
        let mut replan_wrapped_head = Vec::<ReplanCandidate>::new();
        let mut changed = false;
        let mut coordinator_prefix_changed = false;
        let mut backfill_head: Option<BackfillHead> = None;
        // Generated cells overwhelmingly share one host-wide alternative
        // watch. The exact key keeps collision handling semantic while making
        // their resource-serial lookup a once-per-scan cost.
        let mut encoded_watch_serial_memo = BTreeMap::<EncodedWatchSerialMemoKey, u64>::new();
        self.bump_grant_scans();
        note_grant_scan(records.len());
        // Grant-flow ramp gauge (diagnostics-only): count in-flight HELD/GRANTED
        // run claims and the distinct host CPUs they cover this scan. Gated so a
        // disabled sink pays no per-record accounting.
        let sample_held_in_flight = crate::vmm::grant_flow::enabled();
        let mut held_in_flight = 0u64;
        let mut held_cpu_bits = if sample_held_in_flight {
            vec![0u64; self.layout.words]
        } else {
            Vec::new()
        };

        for record in records {
            // An expired planner callback is quarantined until its own owner
            // acknowledges it. It contributes neither a predecessor fence
            // nor coordinator/progress ownership and cannot be republished by
            // an intervening authoritative scan.
            if record.state == STATE_REPLAN_EXPIRED {
                if record.backfill_started_ns != 0 {
                    self.set_record_backfill_started_ns(record.slot, 0)?;
                    changed = true;
                }
                continue;
            }
            // PENDING does not participate in coordinator election, but its
            // bounded preparation footprint is a real predecessor claim.
            // REVOKED likewise remains an exact predecessor fence until its
            // own callback acknowledges that no stale physical probe can
            // begin or that every acquired payload has already been dropped.
            if matches!(record.state, STATE_PENDING | STATE_REVOKED) {
                if let Some(head) = backfill_head
                    .as_mut()
                    .filter(|head| claims_conflict(&head.claim, &record.claim))
                {
                    head.available = head
                        .available
                        .saturating_sub(backfill_cost_for_claim(&record.claim));
                }
                add_claim_bits(
                    &record.claim,
                    &mut cpu_any,
                    &mut cpu_exclusive,
                    &mut llc_any,
                    &mut llc_exclusive,
                    self.layout.bits,
                )?;
                // A PENDING record physically holds its preparation token, and
                // that token is in its claim, so count it directly. A REVOKED
                // preparation grant's callback may still own a token it raced
                // before acknowledging the revoke (its run claim no longer
                // names one), so charge its slot conservatively to keep the
                // budget from over-granting against an in-flight release.
                prep_pending_tokens += if record.state == STATE_PENDING {
                    preparation_tokens_in_claim(&record.claim)
                } else if record.preparation_intent {
                    1
                } else {
                    0
                };
                if record.backfill_started_ns != 0 {
                    self.set_record_backfill_started_ns(record.slot, 0)?;
                    changed = true;
                }
                continue;
            }
            let conflict = claim_conflicts_bits(
                &record.claim,
                &cpu_any,
                &cpu_exclusive,
                &llc_any,
                &llc_exclusive,
                self.layout.bits,
            )?;
            let flexible = record.flexible;
            let availability_compatible = self.claim_availability_compatible(&record.claim)?;
            let blocker_ready = match record.blocked_on {
                None => true,
                Some(blocked) => {
                    let still_designated = match blocked.key {
                        ResourceKey::Cpu(index) => record.claim.cpus.contains(&index),
                        ResourceKey::Llc(index) => record.claim.llcs.contains(&index),
                        ResourceKey::Permit(index) => record.claim.permits.contains(&index),
                    };
                    let external_preparation = record.external_blocker.is_some();
                    (!still_designated && !external_preparation)
                        || self.blocker_serial(blocked.key, blocked.mode)? > blocked.serial
                }
            };
            let prefix_invalid = record.prefix_epoch == 0
                || match record.state {
                    STATE_GRANTED => record.prefix_epoch != record.grant_epoch,
                    STATE_REPLAN => record.prefix_epoch != record.replan_claim_epoch,
                    STATE_HELD => false,
                    // WAITING may have returned from either kind of callback.
                    // Its predecessor prefix remains valid because replacing
                    // this ticket's own exact claim cannot alter its prefix.
                    STATE_WAITING | STATE_COORDINATOR | STATE_COORDINATOR_STANDBY => {
                        record.prefix_epoch != record.grant_epoch
                            && record.prefix_epoch != record.replan_claim_epoch
                    }
                    _ => true,
                };
            let consider_waiting_replan =
                record.state == STATE_WAITING && record.ticket != coordinator_ticket && flexible;
            // Existing callback/coordinator publications must compare their
            // complete predecessor prefix. Fence participation can change
            // inside this very scan (for example WAITING -> GRANTED) without
            // an exact-claim mutation watermark. An invalid epoch already
            // forces refresh and needs no comparison. WAITING candidates are
            // handled lazily after exact viability, so runnable flexible work
            // never pays this speculative cost.
            let published_prefix_matches = if !prefix_invalid
                && matches!(
                    record.state,
                    STATE_GRANTED | STATE_REPLAN | STATE_COORDINATOR
                ) {
                self.cached_prefix_matches_words(
                    record.slot,
                    &cpu_any,
                    &cpu_exclusive,
                    &llc_any,
                    &llc_exclusive,
                )?
            } else {
                true
            };
            // One complete cooperative-capacity wave may remain outstanding
            // behind the oldest unavailable head. Charge resource units, not
            // callbacks: a herd of small cells can therefore fill the same
            // weighted permit pool that normally bounds it. Recompute the
            // live debit from PENDING/GRANTED/HELD records on every scan so a
            // completed bypass claim immediately makes room for replacement
            // work while this head's bounded admission interval remains open.
            let backfill_cost = backfill_head.as_ref().and_then(|head| {
                claims_conflict(&head.claim, &record.claim)
                    .then(|| backfill_cost_for_claim(&record.claim))
            });
            if matches!(record.state, STATE_GRANTED | STATE_HELD)
                && let (Some(head), Some(cost)) = (backfill_head.as_mut(), backfill_cost)
            {
                head.available = head.available.saturating_sub(cost);
            }
            let fairness_blocked = match (backfill_head.as_ref(), backfill_cost) {
                (Some(head), Some(cost)) => !head.admission_open || cost > head.available,
                _ => false,
            };
            let acquisition_viable =
                !conflict && availability_compatible && blocker_ready && !fairness_blocked;
            // Preparation-slot pool gate. A WAITING preparation intent may be
            // granted only while a free token remains after tokens held by
            // fenced PENDING records, this scan's earlier preparation grants,
            // and the coordinator's reserved head slot. Ticket order makes the
            // budget FIFO: when a token frees, the next scan grants the oldest
            // still-blocked preparation intent first. A pool-blocked intent
            // stays plain WAITING — never blocked-pinned — and its now-in-watch
            // token pool wakes it on any release, so the next scan re-evaluates
            // it. Run reservations (not preparation intents) never gate here.
            let preparation_pool_blocked = record.preparation_intent
                && preparation_tokens.is_some()
                && prep_pending_tokens + prep_granted_tokens + coordinator_reserve
                    >= preparation_pool;
            let mut scan_state = record.state;
            if record.state == STATE_COORDINATOR {
                // A coordinator can now sit behind live GRANTED/REPLAN
                // callbacks. Keep its cached predecessor prefix synchronized
                // by the same authoritative scan that refreshes callback
                // prefixes. In particular, a predecessor that commits and
                // later releases must disappear from this cache before the
                // coordinator can retry the formerly-conflicting target.
                if prefix_invalid || !published_prefix_matches {
                    if !published_prefix_matches && scan_publication_epoch == claim_epoch {
                        scan_publication_epoch = self.advance_claim_epoch()?;
                    }
                    coordinator_prefix_changed |= !published_prefix_matches;
                    self.publish_prefix_words(
                        record.slot,
                        &cpu_any,
                        &cpu_exclusive,
                        &llc_any,
                        &llc_exclusive,
                        R_GRANT_EPOCH,
                        scan_publication_epoch,
                        record.issue_serial,
                    )?;
                }
            } else if record.state == STATE_REPLAN && (prefix_invalid || !published_prefix_matches)
            {
                if !published_prefix_matches && scan_publication_epoch == claim_epoch {
                    scan_publication_epoch = self.advance_claim_epoch()?;
                }
                // The callback runs outside the registry fence. Refresh its
                // token only when an input it can observe actually changed.
                // A global claim epoch alone is not an input: the exact cached
                // predecessor bitsets above prove whether the suffix change
                // reached this ticket. Resource improvements do not rewrite a
                // live REPLAN: it publishes one WAITING choice, whose next
                // scan validates exact viability and consumes any unseen
                // alternative improvement.
                self.publish_prefix_words(
                    record.slot,
                    &cpu_any,
                    &cpu_exclusive,
                    &llc_any,
                    &llc_exclusive,
                    R_REPLAN_CLAIM_EPOCH,
                    scan_publication_epoch,
                    record.issue_serial,
                )?;
                changed = true;
            } else if record.state == STATE_GRANTED && (conflict || !availability_compatible) {
                // An earlier ticket may have changed its exact alternative
                // or committed a newly-busy resource after this grant was
                // issued. Revoke the stale grant before it can race into the
                // real flock probe.
                self.set_record_state(record.slot, STATE_REVOKED)?;
                self.clear_record_blocked_known(record.slot, record.external_blocker)?;
                crash_at_for_tests("revoke_state_before_wake");
                self.wake_slot(record.slot)?;
                changed = true;
                scan_state = STATE_REVOKED;
                // Keep the old grant in this and every intervening scan. Its
                // callback may already own the real flock outside registry
                // EX; only that ticket's explicit REVOKED acknowledgement may
                // publish WAITING + RESCAN and retire this fence.
            } else if record.state == STATE_WAITING
                && record.ticket != coordinator_ticket
                && acquisition_viable
                && !preparation_pool_blocked
            {
                // Charge only an actual new conflicting grant. Disjoint work
                // is unbounded; existing PENDING/GRANTED/HELD callbacks were
                // charged above from the authoritative record states.
                if let (Some(head), Some(cost)) = (backfill_head.as_mut(), backfill_cost) {
                    debug_assert!(head.admission_open && cost <= head.available);
                    head.available -= cost;
                }
                self.publish_prefix_words(
                    record.slot,
                    &cpu_any,
                    &cpu_exclusive,
                    &llc_any,
                    &llc_exclusive,
                    R_GRANT_EPOCH,
                    scan_publication_epoch,
                    self.max_watch_serial(&record.claim)?,
                )?;
                self.set_record_state(record.slot, STATE_GRANTED)?;
                self.clear_record_blocked_known(record.slot, record.external_blocker)?;
                crash_at_for_tests("grant_state_before_wake");
                self.wake_slot(record.slot)?;
                crate::vmm::grant_flow::note_grant_issued();
                changed = true;
                scan_state = STATE_GRANTED;
            } else if consider_waiting_replan
                && !preparation_pool_blocked
                && replan_publication_open
                && replan_slots != 0
            {
                // A pool-blocked preparation intent is not replanned: the pool,
                // not its CPU/LLC placement, is the binding constraint, so
                // replanning would only churn. Left plain WAITING, it is granted
                // directly by the next scan once a slot frees, in ticket order.
                //
                // The global serial is an O(1) rejection filter. Walk this
                // ticket's encoded alternative watch only when a newer global
                // observation could actually be relevant. Every eligible
                // ticket may join the bounded parallel publication wave.
                let waiting_serial = if global_serial > record.issue_serial {
                    self.memoized_encoded_watch_serial(
                        &mut encoded_watch_serial_memo,
                        record.slot,
                        record.watch_modes,
                        record.watch_identity,
                        record.blocked_on,
                    )?
                } else {
                    record.issue_serial
                };
                let relevant_resource_change = waiting_serial > record.issue_serial;
                let waiting_prefix_matches = if !relevant_resource_change && !prefix_invalid {
                    self.cached_prefix_matches_words(
                        record.slot,
                        &cpu_any,
                        &cpu_exclusive,
                        &llc_any,
                        &llc_exclusive,
                    )?
                } else {
                    true
                };
                if relevant_resource_change || prefix_invalid || !waiting_prefix_matches {
                    if record.ticket > replan_cursor {
                        if replan_tail.len() < replan_slots {
                            replan_tail.push(ReplanCandidate {
                                slot: record.slot,
                                ticket: record.ticket,
                                waiting_serial,
                                external_blocker: record.external_blocker,
                                cpu_any: cpu_any.clone(),
                                cpu_exclusive: cpu_exclusive.clone(),
                                llc_any: llc_any.clone(),
                                llc_exclusive: llc_exclusive.clone(),
                            });
                            replan_wrapped_head
                                .truncate(replan_slots.saturating_sub(replan_tail.len()));
                        }
                    } else if replan_wrapped_head.len() < replan_slots {
                        replan_wrapped_head.push(ReplanCandidate {
                            slot: record.slot,
                            ticket: record.ticket,
                            waiting_serial,
                            external_blocker: record.external_blocker,
                            cpu_any: cpu_any.clone(),
                            cpu_exclusive: cpu_exclusive.clone(),
                            llc_any: llc_any.clone(),
                            llc_exclusive: llc_exclusive.clone(),
                        });
                    }
                }
            } else if record.state == STATE_GRANTED && (prefix_invalid || !published_prefix_matches)
            {
                if !published_prefix_matches && scan_publication_epoch == claim_epoch {
                    scan_publication_epoch = self.advance_claim_epoch()?;
                }
                // This grant remains valid, but its callback must observe each
                // real predecessor-prefix change. Resource improvements cannot
                // invalidate the authoritative physical exact probe.
                self.publish_prefix_words(
                    record.slot,
                    &cpu_any,
                    &cpu_exclusive,
                    &llc_any,
                    &llc_exclusive,
                    R_GRANT_EPOCH,
                    scan_publication_epoch,
                    record.issue_serial,
                )?;
            }

            // Only real/published owners and exact claims which can run now
            // reserve resources from later queue records. An unavailable wide
            // WAITING head no longer drains unrelated host capacity merely by
            // being earlier in ticket order. REPLAN has no acquisition license
            // and therefore does not fence: its callback can only publish a
            // WAITING replacement, which forces a fresh authoritative scan
            // before that replacement may acquire anything.
            if record.state == STATE_COORDINATOR
                && acquisition_viable
                && let (Some(head), Some(cost)) = (backfill_head.as_mut(), backfill_cost)
            {
                debug_assert!(head.admission_open && cost <= head.available);
                head.available -= cost;
            }

            let preserves_fence = matches!(scan_state, STATE_GRANTED | STATE_REVOKED | STATE_HELD)
                || (scan_state == STATE_COORDINATOR && acquisition_viable);
            if sample_held_in_flight && matches!(scan_state, STATE_GRANTED | STATE_HELD) {
                held_in_flight += 1;
                for cpu in record.claim.cpus() {
                    if cpu < self.layout.bits {
                        held_cpu_bits[cpu / 64] |= 1u64 << (cpu % 64);
                    }
                }
            }
            let mut selected_backfill_head = false;
            if preserves_fence {
                add_claim_bits(
                    &record.claim,
                    &mut cpu_any,
                    &mut cpu_exclusive,
                    &mut llc_any,
                    &mut llc_exclusive,
                    self.layout.bits,
                )?;
                // A GRANTED preparation intent — whether granted in an earlier
                // scan and still racing, or promoted from WAITING just above —
                // reserves one pool token it has not yet turned into a PENDING
                // claim. Charge it here (its run claim names no token) so a
                // junior waiter later in ticket order cannot be granted against
                // a slot this cohort is already racing for. A grant revoked
                // just above is charged the same way: its callback may already
                // own the token it raced, and only that ticket's own REVOKED
                // acknowledgement releases it — the same reason a record which
                // entered this scan REVOKED is charged above. HELD has released
                // its token; the coordinator is charged via the head reserve.
                if matches!(scan_state, STATE_GRANTED | STATE_REVOKED) && record.preparation_intent
                {
                    prep_granted_tokens += 1;
                }
            } else if backfill_head.is_none()
                && !conflict
                && (!availability_compatible || !blocker_ready)
                && (scan_state == STATE_COORDINATOR || (scan_state == STATE_WAITING && !flexible))
            {
                // Protect only the oldest physically blocked exact candidate.
                // Once it runs, queue order inductively gives the next blocked
                // record its own bounded work-conserving interval. This avoids
                // an O(N²) list of fairness barriers while still bounding
                // starvation.
                let started_ns = if record.backfill_started_ns == 0
                    || record.backfill_started_ns > backfill_now_ns
                {
                    self.set_record_backfill_started_ns(record.slot, backfill_now_ns)?;
                    changed = true;
                    backfill_now_ns
                } else {
                    record.backfill_started_ns
                };
                backfill_head = Some(BackfillHead {
                    claim: record.claim.clone(),
                    available: record.backfill_capacity,
                    admission_open: backfill_now_ns - started_ns < BACKFILL_MAX_AGE_NS,
                });
                selected_backfill_head = true;
            }
            if !selected_backfill_head && record.backfill_started_ns != 0 {
                self.set_record_backfill_started_ns(record.slot, 0)?;
                changed = true;
            }
        }
        for candidate in replan_tail.into_iter().chain(replan_wrapped_head) {
            if !replan_batch_started {
                // Never renew a live lease: its original deadline bounds both
                // an old straggler and all incrementally published work.
                if !replan_lease_active {
                    self.arm_replan_wave_at(backfill_now_ns);
                    write_u64(&mut self.header, H_REPLAN_HORIZON, candidate.ticket);
                }
                replan_batch_started = true;
            }
            self.publish_prefix_words(
                candidate.slot,
                &candidate.cpu_any,
                &candidate.cpu_exclusive,
                &candidate.llc_any,
                &candidate.llc_exclusive,
                R_REPLAN_CLAIM_EPOCH,
                scan_publication_epoch,
                candidate.waiting_serial,
            )?;
            self.set_record_state(candidate.slot, STATE_REPLAN)?;
            self.clear_record_blocked_known(candidate.slot, candidate.external_blocker)?;
            write_u64(&mut self.header, H_REPLAN_CURSOR, candidate.ticket);
            let horizon = read_u64(&self.header, H_REPLAN_HORIZON).max(candidate.ticket);
            write_u64(&mut self.header, H_REPLAN_HORIZON, horizon);
            crash_at_for_tests("replan_state_and_cursor_before_wake");
            replan_wake_slots.push(candidate.slot);
            changed = true;
        }
        // Publish the complete scan batch before making its workers runnable.
        // This keeps awakened planners from convoying on the writer while it
        // is still copying later prefixes. A crash anywhere before the clean
        // marker leaves the transaction dirty; repair demotes every
        // ambiguously delivered callback before any completion can commit.
        if replan_batch_started {
            crash_at_for_tests("replan_wave_published_before_wake");
        }
        for slot in replan_wake_slots {
            self.wake_slot(slot)?;
        }
        if changed {
            self.advance_generation()?;
            self.note_queue_progress()?;
        }
        if sample_held_in_flight {
            let distinct_cpus = held_cpu_bits
                .iter()
                .map(|word| word.count_ones() as u64)
                .sum();
            crate::vmm::grant_flow::note_held_in_flight(held_in_flight, distinct_cpus);
        }
        self.finish_claim_scan();
        self.clear_pending_flag(PENDING_RESCAN | PENDING_REPLAN_RESCAN);
        write_u64(&mut self.header, H_DEFERRED_RESCAN_DEADLINE_NS, 0);
        let watch = self.aggregate_watch()?;
        self.finish_transaction()?;
        Ok((watch, coordinator_prefix_changed))
    }

    fn prune_dead(&mut self) -> Result<()> {
        // Victim detection needs only identities; walk the lazy scan records
        // so a large live population does not pay a full watch decode per
        // record on every sweep. Only the dead few are fully re-decoded
        // inside the removal transaction below.
        let records = self.scan_records()?;
        let mut dead = Vec::new();
        for record in records {
            if !ticket_is_live(record.slot, record.ticket)? {
                dead.push((record.slot, record.ticket));
            }
        }
        if dead.is_empty() {
            return Ok(());
        }

        // Full maintenance is the overflow recovery path and may discover a large dead
        // prefix after job cancellation. Free the whole batch in one dirty
        // transaction, then scan/elect once instead of repeatedly electing the
        // next corpse and degenerating to O(N²).
        let removed_coordinator = dead
            .iter()
            .any(|&(_, ticket)| ticket == self.coordinator_ticket());
        self.begin_transaction()?;
        for &(slot, ticket) in &dead {
            // Each unlink mutates its neighbours' prev/next fields. Re-read
            // the next victim so a cloned stale link cannot write through an
            // already-freed predecessor and corrupt the active/free lists.
            if let Some(current) = self
                .record(slot)?
                .filter(|current| current.ticket == ticket)
            {
                self.remove_record_in_transaction(&current, false)?;
            }
        }
        if let Some(first) = dead.iter().map(|&(_, ticket)| ticket).min() {
            self.mark_claim_changed_relaxation_metadata(first)?;
        }
        crash_at_for_tests("remove_record_before_election");
        self.elect_coordinator_in_transaction()?;
        if removed_coordinator {
            self.set_urgent_rescan();
        } else if self.replan_outstanding() != 0 && self.coordinator_ticket() != 0 {
            self.schedule_deferred_replan_rescan_in_transaction()?;
        } else {
            self.set_urgent_rescan();
        }
        self.advance_generation_and_wake_pending()?;
        self.finish_transaction()?;
        for (slot, ticket) in dead {
            let _ = std::fs::remove_file(liveness_path(slot, ticket));
        }
        Ok(())
    }

    fn prune_dead_identities(&mut self, identities: &[(u64, u64)]) -> Result<()> {
        let mut dead = Vec::new();
        for &(slot, ticket) in identities {
            let Some(record) = self.record(slot)? else {
                continue;
            };
            if record.ticket == ticket && !ticket_is_live(slot, ticket)? {
                dead.push(record);
            }
        }
        if dead.is_empty() {
            return Ok(());
        }
        let removed_coordinator = dead
            .iter()
            .any(|record| record.ticket == self.coordinator_ticket());
        self.begin_transaction()?;
        for record in &dead {
            if let Some(current) = self
                .record(record.slot)?
                .filter(|current| current.ticket == record.ticket)
            {
                self.remove_record_in_transaction(&current, false)?;
            }
        }
        if let Some(first) = dead.iter().map(|record| record.ticket).min() {
            self.mark_claim_changed_relaxation_metadata(first)?;
        }
        self.elect_coordinator_in_transaction()?;
        if removed_coordinator {
            self.set_urgent_rescan();
        } else if self.replan_outstanding() != 0 && self.coordinator_ticket() != 0 {
            self.schedule_deferred_replan_rescan_in_transaction()?;
        } else {
            self.set_urgent_rescan();
        }
        self.advance_generation_and_wake_pending()?;
        self.finish_transaction()?;
        for record in dead {
            let _ = std::fs::remove_file(liveness_path(record.slot, record.ticket));
        }
        Ok(())
    }

    fn recover_coordinator_if_dead(&mut self) -> Result<()> {
        self.recover_coordinator_at(monotonic_now_ns()?, false)
    }

    fn recover_coordinator_if_stalled(&mut self) -> Result<()> {
        self.recover_coordinator_at(monotonic_now_ns()?, true)
    }

    #[cfg(test)]
    fn recover_coordinator_if_stalled_at(&mut self, now: u64) -> Result<()> {
        self.recover_coordinator_at(now, true)
    }

    fn recover_coordinator_at(&mut self, now: u64, allow_stalled_takeover: bool) -> Result<()> {
        if self.expire_replan_wave_if_due_at(now)? {
            // Fallback recovery may run in an ordinary ticket after a timed
            // futex wait. Wake the coordinator's real inotify transport as
            // well as the registry futexes published by expiration.
            notify_coordinator();
        }
        let mut coordinator = self.coordinator_ticket();
        let mut slot = self.coordinator_slot()?;
        if coordinator == 0 {
            let head = read_u64(&self.header, H_ACTIVE_HEAD);
            if head != NONE_SLOT
                && let Some(record) = self.record(head)?
                && !ticket_is_live(record.slot, record.ticket)?
            {
                // A cancelled process wave commonly leaves several consecutive
                // in-flight/dead heads. Batch every corpse and elect once.
                self.prune_dead()?;
            }
            // Clean transactions elect synchronously whenever a record enters
            // WAITING. With no coordinator, a live callback-only list therefore
            // contains no work that an election scan could discover.
            return Ok(());
        }
        let mut record = self
            .record(slot)?
            .filter(|record| record.ticket == coordinator && record.state == STATE_COORDINATOR);
        if record.is_none() {
            // The record table is authoritative. A clean header/record
            // mismatch must not permanently poison admission, so defensively
            // rebuild the active/free lists, aggregates, coordinator header,
            // and record states together. Current v26 publication validates
            // this pair before clearing the dirty bit.
            self.repair_consistency()?;
            coordinator = self.coordinator_ticket();
            slot = self.coordinator_slot()?;
            if coordinator == 0 {
                return Ok(());
            }
            record = self
                .record(slot)?
                .filter(|record| record.ticket == coordinator && record.state == STATE_COORDINATOR);
        }
        let record = record.ok_or_else(|| {
            anyhow::anyhow!(
                "queue registry v{VERSION} coordinator header ticket={coordinator}, \
                     slot={slot} does not name its coordinator record"
            )
        })?;
        if !ticket_is_live(slot, coordinator)? {
            // Coordinator death commonly accompanies a cancelled job with a
            // dead prefix. This is a rare recovery scan: batch every dead
            // record and elect one live survivor once, rather than peeling and
            // re-electing dead coordinators one O(N) scan at a time.
            self.prune_dead()?;
            return self.elect_coordinator();
        }

        // Ordinary registry mutations repair a coordinator which actually
        // died, but never transfer a merely stale live lease. Only a waiting
        // ticket's bounded recovery timeout may make that semantic decision.
        // This keeps a large same-thread registration batch from repeatedly
        // displacing its elected owner before any ticket has entered the
        // coordinator loop, while preserving the same eight-second recovery
        // bound once a successor is genuinely waiting for progress.
        if !allow_stalled_takeover {
            return Ok(());
        }

        if coordinator_activity_is_fresh(&self.header, now) {
            return Ok(());
        }
        let mut candidates = self.records()?;
        candidates.retain(|candidate| {
            matches!(candidate.state, STATE_WAITING | STATE_COORDINATOR_STANDBY)
        });
        // Drain never-tried waiters before recycling a coordinator which
        // already missed its lease. Once no WAITING ticket remains, the
        // oldest standby is the progress fallback instead of a terminal
        // two-ticket wedge.
        candidates.sort_by_key(|candidate| (candidate.state != STATE_WAITING, candidate.ticket));
        let mut successor = None;
        for candidate in candidates {
            // Do not spend another bounded lease and repair turn transferring
            // the header to an identity whose liveness OFD is already gone.
            // A death after this probe remains covered by ordinary coordinator
            // recovery on the next waiter tick.
            if ticket_is_live(candidate.slot, candidate.ticket)? {
                successor = Some(candidate);
                break;
            }
        }
        let Some(successor) = successor else {
            return Ok(());
        };

        // A live process can stop advancing between any two userspace
        // instructions (or remain indefinitely descheduled under a storm).
        // Transfer the bounded coordinator lease to the oldest waiter. The old
        // ticket stays live but parked and is eligible for a later election;
        // its logical prefix reservation is suspended while parked so the new
        // coordinator can actually drain compatible work.
        self.begin_transaction()?;
        let current = self
            .record(slot)?
            .filter(|current| current.ticket == record.ticket && current.state == STATE_COORDINATOR)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "queue coordinator {} changed during stalled-lease recovery",
                    record.ticket
                )
            })?;
        let successor = self
            .record(successor.slot)?
            .filter(|candidate| {
                candidate.ticket == successor.ticket
                    && matches!(candidate.state, STATE_WAITING | STATE_COORDINATOR_STANDBY)
            })
            .ok_or_else(|| anyhow::anyhow!("queue takeover successor changed during recovery"))?;
        if coordinator_activity_is_fresh(&self.header, now) {
            self.finish_transaction()?;
            return Ok(());
        }
        let changed_suffix = current.ticket.min(successor.ticket);
        self.set_record_state(current.slot, STATE_COORDINATOR_STANDBY)?;
        self.set_coordinator(successor.ticket, successor.slot)?;
        self.set_record_state(successor.slot, STATE_COORDINATOR)?;
        self.clear_record_blocked(successor.slot)?;
        // Transfer removes the displaced coordinator's potential fence and
        // adds the successor's. A repeated takeover can promote an older
        // standby ticket, so dirty from the earlier identity rather than
        // assuming every successor lies later in queue order. Only the
        // successor's claim is newly fenceable; the displaced side is a
        // relaxation.
        self.mark_claim_changed_fencing(changed_suffix, &successor.claim)?;
        self.advance_generation()?;
        self.wake_slot(current.slot)?;
        self.wake_slot(successor.slot)?;
        self.finish_transaction()?;
        // The displaced coordinator may already be blocked in its real
        // inotify wait. Its slot futex wakes state_or_wait, but cannot wake
        // that directory poll; publish the same event edge ordinary queue
        // mutations use so it observes STANDBY and parks immediately.
        notify_coordinator();
        Ok(())
    }

    fn elect_coordinator(&mut self) -> Result<()> {
        if self.coordinator_ticket() != 0 {
            return Ok(());
        }
        self.begin_transaction()?;
        self.elect_coordinator_in_transaction()?;
        self.finish_transaction()?;
        Ok(())
    }

    fn elect_coordinator_in_transaction(&mut self) -> Result<()> {
        if self.coordinator_ticket() != 0 {
            return Ok(());
        }
        let next_slot = self.next_slot()?;
        let mut slot = read_u64(&self.header, H_ACTIVE_HEAD);
        let mut visited = 0u64;
        while slot != NONE_SLOT {
            if visited >= next_slot {
                anyhow::bail!(
                    "queue registry v{VERSION} active list contains a cycle during coordinator election"
                );
            }
            #[cfg(test)]
            COORDINATOR_ELECTION_RECORD_READS
                .with(|reads| reads.set(reads.get().saturating_add(1)));
            // Election needs only state/ticket/links; the lazy scan decode
            // avoids materializing every candidate's alternative watch under
            // the registry EX flock.
            let record = self.scan_record(slot)?.ok_or_else(|| {
                anyhow::anyhow!("queue active slot {slot} disappeared during coordinator election")
            })?;
            if matches!(record.state, STATE_WAITING | STATE_COORDINATOR_STANDBY) {
                self.set_coordinator(record.ticket, record.slot)?;
                crash_at_for_tests("elect_header_before_state");
                self.set_record_state(record.slot, STATE_COORDINATOR)?;
                self.clear_record_blocked(record.slot)?;
                self.wake_slot(record.slot)?;
                self.advance_generation()?;
                return Ok(());
            }
            slot = record.next_active;
            visited += 1;
        }
        Ok(())
    }

    /// Restore a synthetic test queue after registration has exercised real
    /// wall-clock coordinator recovery. Large fixtures may take longer than a
    /// production lease merely to append their records on an oversubscribed
    /// host; their explicit scan assertions must start from the intended
    /// oldest coordinator rather than whichever waiter registration happened
    /// to promote. Keep this test-only and transactional so the published
    /// header and record state are never incoherent.
    #[cfg(test)]
    fn restage_coordinator_for_tests<'a>(
        &mut self,
        coordinator: &Ticket,
        waiters: impl IntoIterator<Item = &'a Ticket>,
    ) -> Result<()> {
        self.begin_transaction()?;
        let record = self
            .record(coordinator.slot)?
            .filter(|record| record.ticket == coordinator.ticket)
            .ok_or_else(|| anyhow::anyhow!("synthetic coordinator ticket disappeared"))?;
        self.set_record_state(record.slot, STATE_WAITING)?;
        self.clear_record_blocked(record.slot)?;
        for waiter in waiters {
            let record = self
                .record(waiter.slot)?
                .filter(|record| record.ticket == waiter.ticket)
                .ok_or_else(|| anyhow::anyhow!("synthetic waiter ticket disappeared"))?;
            self.set_record_state(record.slot, STATE_WAITING)?;
            self.clear_record_blocked(record.slot)?;
        }
        self.set_coordinator(0, NONE_SLOT)?;
        self.elect_coordinator_in_transaction()?;
        anyhow::ensure!(
            self.coordinator_ticket() == coordinator.ticket
                && self.coordinator_slot()? == coordinator.slot,
            "synthetic queue did not re-elect its oldest intended coordinator",
        );
        self.finish_transaction()
    }

    fn begin_transaction(&mut self) -> Result<()> {
        self.repair_consistency_if_needed()?;
        atomic_u64(&self.header, H_AGGREGATE_DIRTY).store(1, Ordering::SeqCst);
        Ok(())
    }

    fn finish_transaction(&mut self) -> Result<()> {
        let coordinator = self.coordinator_ticket();
        let slot = self.coordinator_slot()?;
        if coordinator == 0 {
            anyhow::ensure!(
                slot == NONE_SLOT,
                "queue registry v{VERSION} cannot publish an empty coordinator ticket with slot {slot}"
            );
        } else {
            anyhow::ensure!(
                self.record_identity_is(slot, coordinator, STATE_COORDINATOR)?,
                "queue registry v{VERSION} cannot publish coordinator header ticket={coordinator}, \
                 slot={slot} without its coordinator record"
            );
        }
        atomic_u64(&self.header, H_AGGREGATE_DIRTY).store(0, Ordering::SeqCst);
        Ok(())
    }

    fn bitmap_bit(&self, which: usize, index: usize) -> Result<bool> {
        if index >= self.layout.bits {
            anyhow::bail!("resource index {index} exceeds queue registry capacity");
        }
        let offset = self.layout.bitset_offset(which) + index / 64 * 8;
        Ok(read_u64(&self.header, offset) & (1u64 << (index % 64)) != 0)
    }

    fn set_bitmap_bit(&mut self, which: usize, index: usize, value: bool) -> Result<()> {
        if index >= self.layout.bits {
            anyhow::bail!("resource index {index} exceeds queue registry capacity");
        }
        let offset = self.layout.bitset_offset(which) + index / 64 * 8;
        let old = read_u64(&self.header, offset);
        let mask = 1u64 << (index % 64);
        write_u64(
            &mut self.header,
            offset,
            if value { old | mask } else { old & !mask },
        );
        Ok(())
    }

    fn bitmap_indices(&self, which: usize) -> BTreeSet<usize> {
        let mut indices = BTreeSet::new();
        for word_index in 0..self.layout.words {
            let mut word = read_u64(
                &self.header,
                self.layout.bitset_offset(which) + word_index * 8,
            );
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                indices.insert(word_index * 64 + bit);
                word &= word - 1;
            }
        }
        indices
    }

    fn resource_serial(&self, which: usize, index: usize) -> Result<u64> {
        if index >= self.layout.bits {
            anyhow::bail!("resource index {index} exceeds queue registry capacity");
        }
        Ok(read_u64(
            &self.header,
            self.layout.serial_offset(which) + index * 8,
        ))
    }

    fn stamp_resource_improvement(&mut self, which: usize, index: usize) -> Result<u64> {
        let serial = self.next_global_serial()?;
        write_u64(
            &mut self.header,
            self.layout.serial_offset(which) + index * 8,
            serial,
        );
        Ok(serial)
    }

    fn claim_availability_compatible(&self, claim: &impl ClaimView) -> Result<bool> {
        for cpu in claim.cpus() {
            if !self.bitmap_bit(B_CPU_KNOWN, cpu)? {
                return Ok(false);
            }
            let available = match claim.cpu_mode() {
                ClaimMode::Shared => self.bitmap_bit(B_CPU_SH_AVAILABLE, cpu)?,
                ClaimMode::Exclusive => self.bitmap_bit(B_CPU_EX_AVAILABLE, cpu)?,
            };
            if !available {
                return Ok(false);
            }
        }
        for permit in claim.permits() {
            let index = permit_resource_index(permit)?;
            if !self.bitmap_bit(B_CPU_KNOWN, index)? {
                return Ok(false);
            }
            let available = match claim.permit_mode() {
                ClaimMode::Shared => self.bitmap_bit(B_CPU_SH_AVAILABLE, index)?,
                ClaimMode::Exclusive => self.bitmap_bit(B_CPU_EX_AVAILABLE, index)?,
            };
            if !available {
                return Ok(false);
            }
        }
        for llc in claim.llcs() {
            if !self.bitmap_bit(B_LLC_KNOWN, llc)? {
                return Ok(false);
            }
            let available = match claim.llc_mode() {
                ClaimMode::Shared => self.bitmap_bit(B_LLC_SH_AVAILABLE, llc)?,
                ClaimMode::Exclusive => self.bitmap_bit(B_LLC_EX_AVAILABLE, llc)?,
            };
            if !available {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn claim_conflicts_aggregate(&self, claim: &ClaimSet) -> Result<bool> {
        let read_words = |which| {
            (0..self.layout.words)
                .map(|word| read_u64(&self.header, self.layout.bitset_offset(which) + word * 8))
                .collect::<Vec<_>>()
        };
        claim_conflicts_bits(
            claim,
            &read_words(B_CLAIM_CPUS),
            &read_words(B_CLAIM_CPU_EXCLUSIVE),
            &read_words(B_CLAIM_LLC_ANY),
            &read_words(B_CLAIM_LLC_EXCLUSIVE),
            self.layout.bits,
        )
    }

    /// Test one candidate against every live claim except one exact record
    /// already owned by this caller. Reference counts make this O(resources)
    /// without mutating the authoritative aggregate or scanning records.
    fn claim_conflicts_aggregate_excluding(
        &self,
        candidate: &ClaimSet,
        excluded: &ClaimSet,
    ) -> Result<bool> {
        Ok(self
            .first_claim_conflict_aggregate_excluding(candidate, excluded)?
            .is_some())
    }

    fn first_claim_conflict_aggregate_excluding(
        &self,
        candidate: &ClaimSet,
        excluded: &ClaimSet,
    ) -> Result<Option<ContentionMarker>> {
        validate_claim(candidate)?;
        let count_after_excluding = |which: usize, index: usize, contributes: bool| {
            if index >= self.layout.bits {
                anyhow::bail!("resource index {index} exceeds queue registry capacity");
            }
            let count = read_u32(
                &self.header,
                self.layout.count_offset(which) + index * std::mem::size_of::<u32>(),
            );
            if contributes {
                count.checked_sub(1).ok_or_else(|| {
                    anyhow::anyhow!(
                        "queue aggregate omitted the excluded claim at resource {index}"
                    )
                })
            } else {
                Ok(count)
            }
        };

        let cpu_which = if candidate.cpu_mode == ClaimMode::Exclusive {
            B_CLAIM_CPUS
        } else {
            B_CLAIM_CPU_EXCLUSIVE
        };
        for &cpu in &candidate.cpus {
            let contributes = excluded.cpus.contains(&cpu)
                && (cpu_which == B_CLAIM_CPUS || excluded.cpu_mode == ClaimMode::Exclusive);
            if count_after_excluding(cpu_which, cpu, contributes)? != 0 {
                return Ok(Some(ContentionMarker {
                    blocker: ResourceKey::Cpu(cpu),
                    mode: match candidate.cpu_mode {
                        ClaimMode::Shared => FlockMode::Shared,
                        ClaimMode::Exclusive => FlockMode::Exclusive,
                    },
                }));
            }
        }

        let permit_which = if candidate.permit_mode == ClaimMode::Exclusive {
            B_CLAIM_CPUS
        } else {
            B_CLAIM_CPU_EXCLUSIVE
        };
        for &permit in &candidate.permits {
            let index = permit_resource_index(permit)?;
            let contributes = excluded.permits.contains(&permit)
                && (permit_which == B_CLAIM_CPUS || excluded.permit_mode == ClaimMode::Exclusive);
            if count_after_excluding(permit_which, index, contributes)? != 0 {
                return Ok(Some(ContentionMarker {
                    blocker: ResourceKey::Permit(permit),
                    mode: match candidate.permit_mode {
                        ClaimMode::Shared => FlockMode::Shared,
                        ClaimMode::Exclusive => FlockMode::Exclusive,
                    },
                }));
            }
        }

        let llc_which = if candidate.llc_mode == ClaimMode::Exclusive {
            B_CLAIM_LLC_ANY
        } else {
            B_CLAIM_LLC_EXCLUSIVE
        };
        for &llc in &candidate.llcs {
            let contributes = excluded.llcs.contains(&llc)
                && (llc_which == B_CLAIM_LLC_ANY || excluded.llc_mode == ClaimMode::Exclusive);
            if count_after_excluding(llc_which, llc, contributes)? != 0 {
                return Ok(Some(ContentionMarker {
                    blocker: ResourceKey::Llc(llc),
                    mode: match candidate.llc_mode {
                        ClaimMode::Shared => FlockMode::Shared,
                        ClaimMode::Exclusive => FlockMode::Exclusive,
                    },
                }));
            }
        }
        Ok(None)
    }

    fn blocker_serial(&self, key: ResourceKey, mode: FlockMode) -> Result<u64> {
        match (key, mode) {
            (ResourceKey::Cpu(index), FlockMode::Shared) => self.resource_serial(S_CPU_SH, index),
            (ResourceKey::Cpu(index), FlockMode::Exclusive) => {
                self.resource_serial(S_CPU_EX, index)
            }
            (ResourceKey::Llc(index), FlockMode::Shared) => self.resource_serial(S_LLC_SH, index),
            (ResourceKey::Llc(index), FlockMode::Exclusive) => {
                self.resource_serial(S_LLC_EX, index)
            }
            (ResourceKey::Permit(index), FlockMode::Shared) => {
                self.resource_serial(S_CPU_SH, permit_resource_index(index)?)
            }
            (ResourceKey::Permit(index), FlockMode::Exclusive) => {
                self.resource_serial(S_CPU_EX, permit_resource_index(index)?)
            }
        }
    }

    /// Turn a callback's negative physical evidence into the registry pin a
    /// requeue publishes: the blocker, the serial it was observed at, and the
    /// consumed serial that folds in the callback's availability snapshot.
    /// Only CPU/LLC run-claim contention pins a record: a lost
    /// preparation-token race no longer blocks, because the grant scan's pool
    /// budget keeps the granted cohort within the free-slot count, so a
    /// granted intent has a slot to race for and any transient miss simply
    /// requeues WAITING for the next budget-ordered grant. (See
    /// `preparation_contention`.)
    fn blocked_evidence(
        &self,
        contention: Option<&ContentionEvidence>,
        watch: &ClaimSet,
        callback_snapshot_serial: u64,
    ) -> Result<Option<(ContentionMarker, u64, u64)>> {
        let Some(evidence) = contention else {
            return Ok(None);
        };
        let marker = evidence.marker();
        validate_contention_within_watch(&[marker], watch)?;
        let serial = self.blocker_serial(marker.blocker, marker.mode)?;
        Ok(Some((marker, serial, callback_snapshot_serial.max(serial))))
    }

    fn max_watch_serial(&self, watch: &impl ClaimView) -> Result<u64> {
        let mut serial = 0;
        for cpu in watch.cpus() {
            serial = serial.max(self.resource_serial(S_CPU_SH, cpu)?);
            if watch.cpu_mode() == ClaimMode::Exclusive {
                serial = serial.max(self.resource_serial(S_CPU_EX, cpu)?);
            }
        }
        for llc in watch.llcs() {
            serial = serial.max(self.resource_serial(S_LLC_SH, llc)?);
            if watch.llc_mode() == ClaimMode::Exclusive {
                serial = serial.max(self.resource_serial(S_LLC_EX, llc)?);
            }
        }
        for permit in watch.permits() {
            let index = permit_resource_index(permit)?;
            serial = serial.max(self.resource_serial(S_CPU_SH, index)?);
            if watch.permit_mode() == ClaimMode::Exclusive {
                serial = serial.max(self.resource_serial(S_CPU_EX, index)?);
            }
        }
        Ok(serial)
    }

    /// Return one exact encoded-watch serial query, memoized for the duration
    /// of a single grant scan. Registry EX keeps every resource serial stable
    /// for that duration. The persisted fixed-seed watch identity includes the
    /// complete immutable watch contents, modes, and admission class; blocker
    /// identity and observation serial complete the key. Each record's issue
    /// serial remains outside the memo and is compared independently with the
    /// returned maximum.
    fn memoized_encoded_watch_serial(
        &mut self,
        memo: &mut BTreeMap<EncodedWatchSerialMemoKey, u64>,
        slot: u64,
        modes: EncodedWatchModes,
        watch_identity: u64,
        blocked_on: Option<BlockedOn>,
    ) -> Result<u64> {
        let key = EncodedWatchSerialMemoKey {
            watch_identity,
            blocked_on: blocked_on.map(EncodedWatchSerialBlockedOn::new),
        };
        if let Some(&serial) = memo.get(&key) {
            return Ok(serial);
        }
        let serial = self.max_encoded_watch_serial(slot, modes, key.blocked_on)?;
        memo.insert(key, serial);
        Ok(serial)
    }

    /// Walk one unique encoded alternative watch's set bits. The per-scan memo
    /// makes a generated-cell storm pay this host-width cost once per exact
    /// watch/blocker identity rather than once per waiter.
    fn max_encoded_watch_serial(
        &mut self,
        slot: u64,
        modes: EncodedWatchModes,
        blocked_on: Option<EncodedWatchSerialBlockedOn>,
    ) -> Result<u64> {
        #[cfg(test)]
        ENCODED_WATCH_SERIAL_WALKS.with(|count| count.set(count.get().saturating_add(1)));
        let mut serial = 0u64;
        for word_index in 0..self.layout.words {
            let mut word = self.encoded_watch_word(slot, RB_WATCH_CPUS, word_index)?;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let index = word_index * 64 + bit;
                serial = serial.max(self.resource_serial(S_CPU_SH, index)?);
                if modes.cpu == ClaimMode::Exclusive {
                    serial = serial.max(self.resource_serial(S_CPU_EX, index)?);
                }
                word &= word - 1;
            }
        }
        for word_index in 0..self.layout.words {
            let mut word = self.encoded_watch_word(slot, RB_WATCH_LLCS, word_index)?;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let index = word_index * 64 + bit;
                serial = serial.max(self.resource_serial(S_LLC_SH, index)?);
                if modes.llc == ClaimMode::Exclusive {
                    serial = serial.max(self.resource_serial(S_LLC_EX, index)?);
                }
                word &= word - 1;
            }
        }
        for word_index in 0..self.layout.words {
            let mut word = self.encoded_watch_word(slot, RB_WATCH_PERMITS, word_index)?;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let permit = word_index * 64 + bit;
                let index = permit_resource_index(permit)?;
                serial = serial.max(self.resource_serial(S_CPU_SH, index)?);
                if modes.permit == ClaimMode::Exclusive {
                    serial = serial.max(self.resource_serial(S_CPU_EX, index)?);
                }
                word &= word - 1;
            }
        }
        if let Some(blocked) = blocked_on {
            serial = serial.max(self.blocker_serial(blocked.resource_key()?, blocked.mode())?);
        }
        Ok(serial)
    }

    fn encoded_watch_word(&mut self, slot: u64, which: usize, word: usize) -> Result<u64> {
        let layout = self.layout;
        let bytes = self.record_bytes(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during encoded watch scan")
        })?;
        Ok(read_u64(
            bytes,
            record_bitset_offset(layout, which) + word * std::mem::size_of::<u64>(),
        ))
    }

    /// Highest compatibility serial that can make the coordinator's current
    /// exact designation runnable.
    ///
    /// An EX contention deliberately observes both SH and EX compatibility:
    /// resolving it as SharedHeld must wake compatible SH tickets. That SH
    /// improvement does not, however, make an EX coordinator runnable. Keep
    /// the broad publication serial above for callback invalidation, while
    /// planner wakes follow the modes of the exact claim currently published
    /// by the coordinator.
    fn max_planner_watch_serial(&self, watch: &ClaimSet, claim: &ClaimSet) -> Result<u64> {
        let mut serial = 0;
        let cpu_serial = match claim.cpu_mode {
            ClaimMode::Shared => S_CPU_SH,
            ClaimMode::Exclusive => S_CPU_EX,
        };
        for &cpu in &watch.cpus {
            serial = serial.max(self.resource_serial(cpu_serial, cpu)?);
        }
        let llc_serial = match claim.llc_mode {
            ClaimMode::Shared => S_LLC_SH,
            ClaimMode::Exclusive => S_LLC_EX,
        };
        for &llc in &watch.llcs {
            serial = serial.max(self.resource_serial(llc_serial, llc)?);
        }
        let permit_serial = match claim.permit_mode {
            ClaimMode::Shared => S_CPU_SH,
            ClaimMode::Exclusive => S_CPU_EX,
        };
        for &permit in &watch.permits {
            serial =
                serial.max(self.resource_serial(permit_serial, permit_resource_index(permit)?)?);
        }
        Ok(serial)
    }

    /// Highest serial that can make the active coordinator's current planner
    /// turn runnable. Preparation blockers may intentionally live outside the
    /// immutable final-run watch, so include their exact observation serial in
    /// addition to the ordinary designation watch.
    fn max_coordinator_planner_serial(&self, record: &Record) -> Result<u64> {
        let mut serial = self.max_planner_watch_serial(&record.watch, &record.claim)?;
        if let Some(blocked) = record.blocked_on {
            serial = serial.max(self.blocker_serial(blocked.key, blocked.mode)?);
        }
        Ok(serial)
    }

    fn mark_unknown(
        &mut self,
        cpus: &BTreeSet<usize>,
        llcs: &BTreeSet<usize>,
        permits: &BTreeSet<usize>,
    ) -> Result<bool> {
        let plan = self.watched_observation_plan(cpus, llcs, permits)?;
        self.mark_observation_modes(&plan)
    }

    fn mark_claim_unknown(&mut self, claim: &ClaimSet) -> Result<bool> {
        self.mark_unknown(&claim.cpus, &claim.llcs, &claim.permits)
    }

    fn mark_observation_modes(&mut self, plan: &PossibleReleasePlan) -> Result<bool> {
        if plan.is_empty() {
            return Ok(false);
        }
        let request = self.bump_observation_request()?;
        for &cpu in &plan.cpu_sh {
            self.set_bitmap_bit(B_CANDIDATE_CPU_SH, cpu, true)?;
            self.set_bitmap_bit(B_CPU_SH_AVAILABLE, cpu, false)?;
            self.set_bitmap_bit(B_PENDING_CPU_SH, cpu, true)?;
            self.set_resource_request(Q_CPU_SH, cpu, request)?;
        }
        for &cpu in &plan.cpu_ex {
            self.set_bitmap_bit(B_CANDIDATE_CPU_EX, cpu, true)?;
            self.set_bitmap_bit(B_CPU_EX_AVAILABLE, cpu, false)?;
            self.set_bitmap_bit(B_PENDING_CPU_EX, cpu, true)?;
            self.set_resource_request(Q_CPU_EX, cpu, request)?;
        }
        for &llc in &plan.llc_sh {
            self.set_bitmap_bit(B_CANDIDATE_LLC_SH, llc, true)?;
            self.set_bitmap_bit(B_LLC_SH_AVAILABLE, llc, false)?;
            self.set_bitmap_bit(B_PENDING_LLC_SH, llc, true)?;
            self.set_resource_request(Q_LLC_SH, llc, request)?;
        }
        for &llc in &plan.llc_ex {
            self.set_bitmap_bit(B_CANDIDATE_LLC_EX, llc, true)?;
            self.set_bitmap_bit(B_LLC_EX_AVAILABLE, llc, false)?;
            self.set_bitmap_bit(B_PENDING_LLC_EX, llc, true)?;
            self.set_resource_request(Q_LLC_EX, llc, request)?;
        }
        for &permit in &plan.permit_sh {
            let index = permit_resource_index(permit)?;
            self.set_bitmap_bit(B_CANDIDATE_CPU_SH, index, true)?;
            self.set_bitmap_bit(B_CPU_SH_AVAILABLE, index, false)?;
            self.set_bitmap_bit(B_PENDING_CPU_SH, index, true)?;
            self.set_resource_request(Q_CPU_SH, index, request)?;
        }
        for &permit in &plan.permit_ex {
            let index = permit_resource_index(permit)?;
            self.set_bitmap_bit(B_CANDIDATE_CPU_EX, index, true)?;
            self.set_bitmap_bit(B_CPU_EX_AVAILABLE, index, false)?;
            self.set_bitmap_bit(B_PENDING_CPU_EX, index, true)?;
            self.set_resource_request(Q_CPU_EX, index, request)?;
        }
        Ok(true)
    }

    fn watched_observation_plan(
        &self,
        cpus: &BTreeSet<usize>,
        llcs: &BTreeSet<usize>,
        permits: &BTreeSet<usize>,
    ) -> Result<PossibleReleasePlan> {
        let mut plan = PossibleReleasePlan::default();
        for &cpu in cpus {
            if self.bitmap_bit(B_WATCH_CPUS, cpu)? {
                plan.cpu_sh.insert(cpu);
            }
            if self.bitmap_bit(B_WATCH_CPU_EXCLUSIVE, cpu)? {
                plan.cpu_ex.insert(cpu);
            }
        }
        for &llc in llcs {
            if self.bitmap_bit(B_WATCH_LLCS, llc)? {
                plan.llc_sh.insert(llc);
            }
            if self.bitmap_bit(B_WATCH_LLC_EXCLUSIVE, llc)? {
                plan.llc_ex.insert(llc);
            }
        }
        for &permit in permits {
            let index = permit_resource_index(permit)?;
            if self.bitmap_bit(B_WATCH_CPUS, index)? {
                plan.permit_sh.insert(permit);
            }
            if self.bitmap_bit(B_WATCH_CPU_EXCLUSIVE, index)? {
                plan.permit_ex.insert(permit);
            }
        }
        Ok(plan)
    }

    /// Record a close that can only make an existing holder state less
    /// restrictive. Already-usable modes stay authoritative and generate no
    /// observation work: free resources are no-ops, while a resource observed
    /// shared-held rechecks EX only. This is the high-rate path for failed
    /// all-or-none probe prefixes, whose writable fd closes must not
    /// manufacture a procfs/scan/planner storm.
    fn possible_release_plan(
        &self,
        cpus: &BTreeSet<usize>,
        llcs: &BTreeSet<usize>,
        permits: &BTreeSet<usize>,
    ) -> Result<PossibleReleasePlan> {
        let mut plan = PossibleReleasePlan::default();
        for &cpu in cpus {
            let known = self.bitmap_bit(B_CPU_KNOWN, cpu)?;
            let sh_pending = self.bitmap_bit(B_PENDING_CPU_SH, cpu)?;
            let ex_pending = self.bitmap_bit(B_PENDING_CPU_EX, cpu)?;
            let sh_available = known && self.bitmap_bit(B_CPU_SH_AVAILABLE, cpu)?;
            let ex_available = known && self.bitmap_bit(B_CPU_EX_AVAILABLE, cpu)?;
            if self.bitmap_bit(B_WATCH_CPUS, cpu)? && !sh_available && !sh_pending {
                plan.cpu_sh.insert(cpu);
            }
            if self.bitmap_bit(B_WATCH_CPU_EXCLUSIVE, cpu)? && !ex_available && !ex_pending {
                plan.cpu_ex.insert(cpu);
            }
        }
        for &llc in llcs {
            let known = self.bitmap_bit(B_LLC_KNOWN, llc)?;
            let sh_pending = self.bitmap_bit(B_PENDING_LLC_SH, llc)?;
            let ex_pending = self.bitmap_bit(B_PENDING_LLC_EX, llc)?;
            let sh_available = known && self.bitmap_bit(B_LLC_SH_AVAILABLE, llc)?;
            let ex_available = known && self.bitmap_bit(B_LLC_EX_AVAILABLE, llc)?;
            if self.bitmap_bit(B_WATCH_LLCS, llc)? && !sh_available && !sh_pending {
                plan.llc_sh.insert(llc);
            }
            if self.bitmap_bit(B_WATCH_LLC_EXCLUSIVE, llc)? && !ex_available && !ex_pending {
                plan.llc_ex.insert(llc);
            }
        }
        for &permit in permits {
            let index = permit_resource_index(permit)?;
            let known = self.bitmap_bit(B_CPU_KNOWN, index)?;
            let sh_pending = self.bitmap_bit(B_PENDING_CPU_SH, index)?;
            let ex_pending = self.bitmap_bit(B_PENDING_CPU_EX, index)?;
            let sh_available = known && self.bitmap_bit(B_CPU_SH_AVAILABLE, index)?;
            let ex_available = known && self.bitmap_bit(B_CPU_EX_AVAILABLE, index)?;
            if self.bitmap_bit(B_WATCH_CPUS, index)? && !sh_available && !sh_pending {
                plan.permit_sh.insert(permit);
            }
            if self.bitmap_bit(B_WATCH_CPU_EXCLUSIVE, index)? && !ex_available && !ex_pending {
                plan.permit_ex.insert(permit);
            }
        }
        Ok(plan)
    }

    fn apply_possible_release(&mut self, plan: &PossibleReleasePlan) -> Result<bool> {
        if plan.is_empty() {
            return Ok(false);
        }
        let request = self.bump_observation_request()?;
        for &cpu in &plan.cpu_sh {
            self.set_bitmap_bit(B_CANDIDATE_CPU_SH, cpu, true)?;
            self.set_bitmap_bit(B_PENDING_CPU_SH, cpu, true)?;
            self.set_resource_request(Q_CPU_SH, cpu, request)?;
        }
        for &cpu in &plan.cpu_ex {
            self.set_bitmap_bit(B_CANDIDATE_CPU_EX, cpu, true)?;
            self.set_bitmap_bit(B_PENDING_CPU_EX, cpu, true)?;
            self.set_resource_request(Q_CPU_EX, cpu, request)?;
        }
        for &llc in &plan.llc_sh {
            self.set_bitmap_bit(B_CANDIDATE_LLC_SH, llc, true)?;
            self.set_bitmap_bit(B_PENDING_LLC_SH, llc, true)?;
            self.set_resource_request(Q_LLC_SH, llc, request)?;
        }
        for &llc in &plan.llc_ex {
            self.set_bitmap_bit(B_CANDIDATE_LLC_EX, llc, true)?;
            self.set_bitmap_bit(B_PENDING_LLC_EX, llc, true)?;
            self.set_resource_request(Q_LLC_EX, llc, request)?;
        }
        for &permit in &plan.permit_sh {
            let index = permit_resource_index(permit)?;
            self.set_bitmap_bit(B_CANDIDATE_CPU_SH, index, true)?;
            self.set_bitmap_bit(B_PENDING_CPU_SH, index, true)?;
            self.set_resource_request(Q_CPU_SH, index, request)?;
        }
        for &permit in &plan.permit_ex {
            let index = permit_resource_index(permit)?;
            self.set_bitmap_bit(B_CANDIDATE_CPU_EX, index, true)?;
            self.set_bitmap_bit(B_PENDING_CPU_EX, index, true)?;
            self.set_resource_request(Q_CPU_EX, index, request)?;
        }
        Ok(true)
    }

    fn mark_possible_release(
        &mut self,
        cpus: &BTreeSet<usize>,
        llcs: &BTreeSet<usize>,
        permits: &BTreeSet<usize>,
    ) -> Result<bool> {
        let plan = self.possible_release_plan(cpus, llcs, permits)?;
        self.apply_possible_release(&plan)
    }

    fn mark_blocker_unknown(&mut self, evidence: ContentionMarker) -> Result<()> {
        let (cpus, llcs, permits) = match evidence.blocker {
            ResourceKey::Cpu(index) => (BTreeSet::from([index]), BTreeSet::new(), BTreeSet::new()),
            ResourceKey::Llc(index) => (BTreeSet::new(), BTreeSet::from([index]), BTreeSet::new()),
            ResourceKey::Permit(index) => {
                (BTreeSet::new(), BTreeSet::new(), BTreeSet::from([index]))
            }
        };
        self.mark_unknown(&cpus, &llcs, &permits)?;
        Ok(())
    }

    fn mark_blockers_unknown(&mut self, evidence: &[ContentionMarker]) -> Result<()> {
        if evidence.is_empty() {
            return Ok(());
        }
        let mut cpus = BTreeSet::new();
        let mut llcs = BTreeSet::new();
        let mut permits = BTreeSet::new();
        for marker in evidence {
            match marker.blocker {
                ResourceKey::Cpu(index) => {
                    cpus.insert(index);
                }
                ResourceKey::Llc(index) => {
                    llcs.insert(index);
                }
                ResourceKey::Permit(index) => {
                    permits.insert(index);
                }
            }
        }
        self.mark_unknown(&cpus, &llcs, &permits).map(|_| ())
    }

    /// Retire a physical probe that produced no publishable ownership: the
    /// probe footprint and every blocker it observed become UNKNOWN in one
    /// transaction that also advances the generation. Every stale, contended,
    /// or license-losing exit from the commit paths publishes exactly this
    /// before releasing the registry fence.
    fn publish_negative_evidence(
        &mut self,
        footprint: &ClaimSet,
        contention: &[ContentionMarker],
    ) -> Result<()> {
        self.begin_transaction()?;
        self.mark_claim_unknown(footprint)?;
        self.mark_blockers_unknown(contention)?;
        self.advance_generation()?;
        self.finish_transaction()
    }

    fn publish_claim_busy(&mut self, claim: &ClaimSet) -> Result<()> {
        let mut invalidated_observation = false;
        let mut compatibility_improved = false;
        for &cpu in &claim.cpus {
            let known = self.bitmap_bit(B_CPU_KNOWN, cpu)?;
            let sh_available = known && self.bitmap_bit(B_CPU_SH_AVAILABLE, cpu)?;
            let pending = self.bitmap_bit(B_PENDING_CPU_SH, cpu)?
                || self.bitmap_bit(B_PENDING_CPU_EX, cpu)?;
            invalidated_observation |= pending;
            if claim.cpu_mode == ClaimMode::Shared && !sh_available {
                self.stamp_resource_improvement(S_CPU_SH, cpu)?;
                compatibility_improved = true;
            }
            self.set_bitmap_bit(B_CPU_KNOWN, cpu, true)?;
            self.set_bitmap_bit(B_CPU_SH_AVAILABLE, cpu, claim.cpu_mode == ClaimMode::Shared)?;
            self.set_bitmap_bit(B_CPU_EX_AVAILABLE, cpu, false)?;
            self.set_bitmap_bit(B_PENDING_CPU_SH, cpu, false)?;
            self.set_bitmap_bit(B_PENDING_CPU_EX, cpu, false)?;
            self.set_bitmap_bit(B_CANDIDATE_CPU_SH, cpu, false)?;
            self.set_bitmap_bit(B_CANDIDATE_CPU_EX, cpu, false)?;
        }
        for &permit in &claim.permits {
            let index = permit_resource_index(permit)?;
            let known = self.bitmap_bit(B_CPU_KNOWN, index)?;
            let sh_available = known && self.bitmap_bit(B_CPU_SH_AVAILABLE, index)?;
            let pending = self.bitmap_bit(B_PENDING_CPU_SH, index)?
                || self.bitmap_bit(B_PENDING_CPU_EX, index)?;
            invalidated_observation |= pending;
            if claim.permit_mode == ClaimMode::Shared && !sh_available {
                self.stamp_resource_improvement(S_CPU_SH, index)?;
                compatibility_improved = true;
            }
            self.set_bitmap_bit(B_CPU_KNOWN, index, true)?;
            self.set_bitmap_bit(
                B_CPU_SH_AVAILABLE,
                index,
                claim.permit_mode == ClaimMode::Shared,
            )?;
            self.set_bitmap_bit(B_CPU_EX_AVAILABLE, index, false)?;
            self.set_bitmap_bit(B_PENDING_CPU_SH, index, false)?;
            self.set_bitmap_bit(B_PENDING_CPU_EX, index, false)?;
            self.set_bitmap_bit(B_CANDIDATE_CPU_SH, index, false)?;
            self.set_bitmap_bit(B_CANDIDATE_CPU_EX, index, false)?;
        }
        for &llc in &claim.llcs {
            let known = self.bitmap_bit(B_LLC_KNOWN, llc)?;
            let sh_available = known && self.bitmap_bit(B_LLC_SH_AVAILABLE, llc)?;
            let pending = self.bitmap_bit(B_PENDING_LLC_SH, llc)?
                || self.bitmap_bit(B_PENDING_LLC_EX, llc)?;
            invalidated_observation |= pending;
            if claim.llc_mode == ClaimMode::Shared && !sh_available {
                // A real held SH fd is an authoritative compatibility
                // improvement over unknown or EX-held state even when an
                // earlier observation already consumed the candidate bit.
                // The generic removal scan no longer masks this transition.
                self.stamp_resource_improvement(S_LLC_SH, llc)?;
                compatibility_improved = true;
            }
            self.set_bitmap_bit(B_LLC_KNOWN, llc, true)?;
            self.set_bitmap_bit(B_LLC_SH_AVAILABLE, llc, claim.llc_mode == ClaimMode::Shared)?;
            self.set_bitmap_bit(B_LLC_EX_AVAILABLE, llc, false)?;
            self.set_bitmap_bit(B_PENDING_LLC_SH, llc, false)?;
            self.set_bitmap_bit(B_PENDING_LLC_EX, llc, false)?;
            self.set_bitmap_bit(B_CANDIDATE_LLC_SH, llc, false)?;
            self.set_bitmap_bit(B_CANDIDATE_LLC_EX, llc, false)?;
        }
        if invalidated_observation {
            self.bump_observation_request()?;
        }
        if compatibility_improved {
            self.set_urgent_rescan();
        }
        self.refresh_pending_observation_flag()?;
        Ok(())
    }

    fn refresh_pending_observation_flag(&mut self) -> Result<()> {
        let pending = (0..self.layout.words).any(|word| {
            read_u64(
                &self.header,
                self.layout.bitset_offset(B_PENDING_CPU_SH) + word * 8,
            ) != 0
                || read_u64(
                    &self.header,
                    self.layout.bitset_offset(B_PENDING_CPU_EX) + word * 8,
                ) != 0
                || read_u64(
                    &self.header,
                    self.layout.bitset_offset(B_PENDING_LLC_SH) + word * 8,
                ) != 0
                || read_u64(
                    &self.header,
                    self.layout.bitset_offset(B_PENDING_LLC_EX) + word * 8,
                ) != 0
        });
        if pending {
            self.set_pending_flag(PENDING_OBSERVATION);
        } else {
            self.clear_pending_flag(PENDING_OBSERVATION);
        }
        Ok(())
    }

    fn observation_request(&self) -> Result<Option<ObservationRequest>> {
        if self.pending_flags() & PENDING_OBSERVATION == 0 {
            return Ok(None);
        }
        let mut cpus = BTreeMap::new();
        let mut llcs = BTreeMap::new();
        let mut permits = BTreeMap::new();
        let pending_cpu_sh = self.bitmap_indices(B_PENDING_CPU_SH);
        let pending_cpu_ex = self.bitmap_indices(B_PENDING_CPU_EX);
        for index in pending_cpu_sh.union(&pending_cpu_ex).copied() {
            let sh = pending_cpu_sh
                .contains(&index)
                .then(|| self.resource_request(Q_CPU_SH, index))
                .transpose()?;
            let ex = pending_cpu_ex
                .contains(&index)
                .then(|| self.resource_request(Q_CPU_EX, index))
                .transpose()?;
            if index < host_cpu_resource_bits() {
                cpus.insert(index, (sh, ex));
            } else {
                permits.insert(index - host_cpu_resource_bits(), (sh, ex));
            }
        }
        let pending_sh = self.bitmap_indices(B_PENDING_LLC_SH);
        let pending_ex = self.bitmap_indices(B_PENDING_LLC_EX);
        for index in pending_sh.union(&pending_ex).copied() {
            let sh = pending_sh
                .contains(&index)
                .then(|| self.resource_request(Q_LLC_SH, index))
                .transpose()?;
            let ex = pending_ex
                .contains(&index)
                .then(|| self.resource_request(Q_LLC_EX, index))
                .transpose()?;
            llcs.insert(index, (sh, ex));
        }
        if cpus.is_empty() && llcs.is_empty() && permits.is_empty() {
            return Ok(None);
        }
        Ok(Some(ObservationRequest {
            cpus,
            llcs,
            permits,
        }))
    }

    fn apply_observation(
        &mut self,
        request: &ObservationRequest,
        observation: &AvailabilityObservation,
    ) -> Result<bool> {
        let mut improved = false;
        for (&cpu, &(sh_request, ex_request)) in &request.cpus {
            let Some(observed) = observation.cpus.get(&cpu).copied() else {
                continue;
            };
            let sh_matches = if let Some(serial) = sh_request {
                observed.sh_resolved
                    && self.bitmap_bit(B_PENDING_CPU_SH, cpu)?
                    && self.resource_request(Q_CPU_SH, cpu)? == serial
            } else {
                false
            };
            let ex_matches = if let Some(serial) = ex_request {
                observed.ex_resolved
                    && self.bitmap_bit(B_PENDING_CPU_EX, cpu)?
                    && self.resource_request(Q_CPU_EX, cpu)? == serial
            } else {
                false
            };
            if !sh_matches && !ex_matches {
                continue;
            }
            let availability = observed.availability;
            let sh_available = availability != CpuAvailability::ExclusiveHeld;
            let ex_available = availability == CpuAvailability::Free;
            let sh_candidate = self.bitmap_bit(B_CANDIDATE_CPU_SH, cpu)?;
            let ex_candidate = self.bitmap_bit(B_CANDIDATE_CPU_EX, cpu)?;
            self.set_bitmap_bit(B_CPU_KNOWN, cpu, true)?;
            if sh_matches {
                self.set_bitmap_bit(B_CPU_SH_AVAILABLE, cpu, sh_available)?;
                self.set_bitmap_bit(B_PENDING_CPU_SH, cpu, false)?;
                self.set_bitmap_bit(B_CANDIDATE_CPU_SH, cpu, false)?;
            }
            if ex_matches {
                self.set_bitmap_bit(B_CPU_EX_AVAILABLE, cpu, ex_available)?;
                self.set_bitmap_bit(B_PENDING_CPU_EX, cpu, false)?;
                self.set_bitmap_bit(B_CANDIDATE_CPU_EX, cpu, false)?;
            }
            if sh_matches && sh_candidate && sh_available {
                self.stamp_resource_improvement(S_CPU_SH, cpu)?;
                improved = true;
            }
            if ex_matches && ex_candidate && ex_available {
                self.stamp_resource_improvement(S_CPU_EX, cpu)?;
                improved = true;
            }
        }
        for (&permit, &(sh_request, ex_request)) in &request.permits {
            let index = permit_resource_index(permit)?;
            let Some(observed) = observation.permits.get(&permit).copied() else {
                continue;
            };
            let sh_matches = if let Some(serial) = sh_request {
                observed.sh_resolved
                    && self.bitmap_bit(B_PENDING_CPU_SH, index)?
                    && self.resource_request(Q_CPU_SH, index)? == serial
            } else {
                false
            };
            let ex_matches = if let Some(serial) = ex_request {
                observed.ex_resolved
                    && self.bitmap_bit(B_PENDING_CPU_EX, index)?
                    && self.resource_request(Q_CPU_EX, index)? == serial
            } else {
                false
            };
            if !sh_matches && !ex_matches {
                continue;
            }
            let availability = observed.availability;
            let sh_available = availability != CpuAvailability::ExclusiveHeld;
            let ex_available = availability == CpuAvailability::Free;
            let sh_candidate = self.bitmap_bit(B_CANDIDATE_CPU_SH, index)?;
            let ex_candidate = self.bitmap_bit(B_CANDIDATE_CPU_EX, index)?;
            self.set_bitmap_bit(B_CPU_KNOWN, index, true)?;
            if sh_matches {
                self.set_bitmap_bit(B_CPU_SH_AVAILABLE, index, sh_available)?;
                self.set_bitmap_bit(B_PENDING_CPU_SH, index, false)?;
                self.set_bitmap_bit(B_CANDIDATE_CPU_SH, index, false)?;
            }
            if ex_matches {
                self.set_bitmap_bit(B_CPU_EX_AVAILABLE, index, ex_available)?;
                self.set_bitmap_bit(B_PENDING_CPU_EX, index, false)?;
                self.set_bitmap_bit(B_CANDIDATE_CPU_EX, index, false)?;
            }
            if sh_matches && sh_candidate && sh_available {
                self.stamp_resource_improvement(S_CPU_SH, index)?;
                improved = true;
            }
            if ex_matches && ex_candidate && ex_available {
                self.stamp_resource_improvement(S_CPU_EX, index)?;
                improved = true;
            }
        }
        for (&llc, &(sh_request, ex_request)) in &request.llcs {
            let Some(observed) = observation.llcs.get(&llc).copied() else {
                continue;
            };
            let sh_matches = if let Some(serial) = sh_request {
                observed.sh_resolved
                    && self.bitmap_bit(B_PENDING_LLC_SH, llc)?
                    && self.resource_request(Q_LLC_SH, llc)? == serial
            } else {
                false
            };
            let ex_matches = if let Some(serial) = ex_request {
                observed.ex_resolved
                    && self.bitmap_bit(B_PENDING_LLC_EX, llc)?
                    && self.resource_request(Q_LLC_EX, llc)? == serial
            } else {
                false
            };
            if !sh_matches && !ex_matches {
                continue;
            }
            let availability = observed.availability;
            let sh_available = availability != LlcAvailability::ExclusiveHeld;
            let ex_available = availability == LlcAvailability::Free;
            let sh_candidate = self.bitmap_bit(B_CANDIDATE_LLC_SH, llc)?;
            let ex_candidate = self.bitmap_bit(B_CANDIDATE_LLC_EX, llc)?;
            self.set_bitmap_bit(B_LLC_KNOWN, llc, true)?;
            if sh_matches {
                self.set_bitmap_bit(B_LLC_SH_AVAILABLE, llc, sh_available)?;
                self.set_bitmap_bit(B_PENDING_LLC_SH, llc, false)?;
                self.set_bitmap_bit(B_CANDIDATE_LLC_SH, llc, false)?;
            }
            if ex_matches {
                self.set_bitmap_bit(B_LLC_EX_AVAILABLE, llc, ex_available)?;
                self.set_bitmap_bit(B_PENDING_LLC_EX, llc, false)?;
                self.set_bitmap_bit(B_CANDIDATE_LLC_EX, llc, false)?;
            }
            if sh_matches && sh_candidate && sh_available {
                self.stamp_resource_improvement(S_LLC_SH, llc)?;
                improved = true;
            }
            if ex_matches && ex_candidate && ex_available {
                self.stamp_resource_improvement(S_LLC_EX, llc)?;
                improved = true;
            }
        }
        self.refresh_pending_observation_flag()?;
        if improved {
            self.set_urgent_rescan();
        }
        Ok(improved)
    }

    fn adjust_claim_counts(&mut self, claim: &ClaimSet, add: bool) -> Result<()> {
        for &cpu in &claim.cpus {
            self.adjust_aggregate_bit(B_CLAIM_CPUS, cpu, add)?;
            if claim.cpu_mode == ClaimMode::Exclusive {
                self.adjust_aggregate_bit(B_CLAIM_CPU_EXCLUSIVE, cpu, add)?;
            }
            if claim.admission_class == AdmissionClass::Build {
                self.adjust_aggregate_count(C_BUILD_CLAIM_CPUS, cpu, add)?;
            }
        }
        for &permit in &claim.permits {
            let index = permit_resource_index(permit)?;
            self.adjust_aggregate_bit(B_CLAIM_CPUS, index, add)?;
            if claim.permit_mode == ClaimMode::Exclusive {
                self.adjust_aggregate_bit(B_CLAIM_CPU_EXCLUSIVE, index, add)?;
            }
        }
        for &llc in &claim.llcs {
            self.adjust_aggregate_bit(B_CLAIM_LLC_ANY, llc, add)?;
            if claim.llc_mode == ClaimMode::Exclusive {
                self.adjust_aggregate_bit(B_CLAIM_LLC_EXCLUSIVE, llc, add)?;
            }
        }
        Ok(())
    }

    fn adjust_held_counts(&mut self, claim: &ClaimSet, add: bool) -> Result<()> {
        let cpu_which = match claim.cpu_mode {
            ClaimMode::Shared => B_HELD_CPU_SHARED,
            ClaimMode::Exclusive => B_HELD_CPU_EXCLUSIVE,
        };
        for &cpu in &claim.cpus {
            self.adjust_aggregate_bit(cpu_which, cpu, add)?;
        }
        let permit_which = match claim.permit_mode {
            ClaimMode::Shared => B_HELD_CPU_SHARED,
            ClaimMode::Exclusive => B_HELD_CPU_EXCLUSIVE,
        };
        for &permit in &claim.permits {
            self.adjust_aggregate_bit(permit_which, permit_resource_index(permit)?, add)?;
        }
        let llc_which = match claim.llc_mode {
            ClaimMode::Shared => B_HELD_LLC_SHARED,
            ClaimMode::Exclusive => B_HELD_LLC_EXCLUSIVE,
        };
        for &llc in &claim.llcs {
            self.adjust_aggregate_bit(llc_which, llc, add)?;
        }
        Ok(())
    }

    /// Mirror of [`Self::adjust_held_counts`] over the count-only
    /// `C_GRANT_*` families: the planner-visible charge of one in-flight
    /// grant (a GRANTED or REVOKED record's exact claim). Checked
    /// over/underflow in `adjust_aggregate_count` is the leak tripwire —
    /// every charge site must balance exactly or the next adjustment fails.
    fn adjust_granted_occupancy(&mut self, claim: &impl ClaimView, add: bool) -> Result<()> {
        for cpu in claim.cpus() {
            self.adjust_aggregate_count(C_GRANT_CPU_ANY, cpu, add)?;
            if claim.cpu_mode() == ClaimMode::Exclusive {
                self.adjust_aggregate_count(C_GRANT_CPU_EX, cpu, add)?;
            }
        }
        for permit in claim.permits() {
            let index = permit_resource_index(permit)?;
            self.adjust_aggregate_count(C_GRANT_CPU_ANY, index, add)?;
            if claim.permit_mode() == ClaimMode::Exclusive {
                self.adjust_aggregate_count(C_GRANT_CPU_EX, index, add)?;
            }
        }
        for llc in claim.llcs() {
            self.adjust_aggregate_count(C_GRANT_LLC_ANY, llc, add)?;
            if claim.llc_mode() == ClaimMode::Exclusive {
                self.adjust_aggregate_count(C_GRANT_LLC_EX, llc, add)?;
            }
        }
        Ok(())
    }

    fn adjust_watch_counts(&mut self, watch: &ClaimSet, add: bool) -> Result<()> {
        for &cpu in &watch.cpus {
            self.adjust_aggregate_bit(B_WATCH_CPUS, cpu, add)?;
            if watch.cpu_mode == ClaimMode::Exclusive {
                self.adjust_aggregate_bit(B_WATCH_CPU_EXCLUSIVE, cpu, add)?;
            }
            if !add && !self.bitmap_bit(B_WATCH_CPU_EXCLUSIVE, cpu)? {
                self.set_bitmap_bit(B_CPU_EX_AVAILABLE, cpu, false)?;
                self.set_bitmap_bit(B_PENDING_CPU_EX, cpu, false)?;
                self.set_bitmap_bit(B_CANDIDATE_CPU_EX, cpu, false)?;
                self.set_resource_request(Q_CPU_EX, cpu, 0)?;
            }
            if !add && !self.bitmap_bit(B_WATCH_CPUS, cpu)? {
                self.set_bitmap_bit(B_CPU_KNOWN, cpu, false)?;
                self.set_bitmap_bit(B_CPU_SH_AVAILABLE, cpu, false)?;
                self.set_bitmap_bit(B_PENDING_CPU_SH, cpu, false)?;
                self.set_bitmap_bit(B_CANDIDATE_CPU_SH, cpu, false)?;
                self.set_resource_request(Q_CPU_SH, cpu, 0)?;
            }
        }
        for &permit in &watch.permits {
            let index = permit_resource_index(permit)?;
            self.adjust_aggregate_bit(B_WATCH_CPUS, index, add)?;
            if watch.permit_mode == ClaimMode::Exclusive {
                self.adjust_aggregate_bit(B_WATCH_CPU_EXCLUSIVE, index, add)?;
            }
            if !add && !self.bitmap_bit(B_WATCH_CPU_EXCLUSIVE, index)? {
                self.set_bitmap_bit(B_CPU_EX_AVAILABLE, index, false)?;
                self.set_bitmap_bit(B_PENDING_CPU_EX, index, false)?;
                self.set_bitmap_bit(B_CANDIDATE_CPU_EX, index, false)?;
                self.set_resource_request(Q_CPU_EX, index, 0)?;
            }
            if !add && !self.bitmap_bit(B_WATCH_CPUS, index)? {
                self.set_bitmap_bit(B_CPU_KNOWN, index, false)?;
                self.set_bitmap_bit(B_CPU_SH_AVAILABLE, index, false)?;
                self.set_bitmap_bit(B_PENDING_CPU_SH, index, false)?;
                self.set_bitmap_bit(B_CANDIDATE_CPU_SH, index, false)?;
                self.set_resource_request(Q_CPU_SH, index, 0)?;
            }
        }
        for &llc in &watch.llcs {
            self.adjust_aggregate_bit(B_WATCH_LLCS, llc, add)?;
            if watch.llc_mode == ClaimMode::Exclusive {
                self.adjust_aggregate_bit(B_WATCH_LLC_EXCLUSIVE, llc, add)?;
            }
            if !add && !self.bitmap_bit(B_WATCH_LLC_EXCLUSIVE, llc)? {
                self.set_bitmap_bit(B_LLC_EX_AVAILABLE, llc, false)?;
                self.set_bitmap_bit(B_PENDING_LLC_EX, llc, false)?;
                self.set_bitmap_bit(B_CANDIDATE_LLC_EX, llc, false)?;
                self.set_resource_request(Q_LLC_EX, llc, 0)?;
            }
            if !add && !self.bitmap_bit(B_WATCH_LLCS, llc)? {
                self.set_bitmap_bit(B_LLC_KNOWN, llc, false)?;
                self.set_bitmap_bit(B_LLC_SH_AVAILABLE, llc, false)?;
                self.set_bitmap_bit(B_PENDING_LLC_SH, llc, false)?;
                self.set_bitmap_bit(B_CANDIDATE_LLC_SH, llc, false)?;
                self.set_resource_request(Q_LLC_SH, llc, 0)?;
            }
        }
        if !add {
            self.refresh_pending_observation_flag()?;
        }
        Ok(())
    }

    fn newly_watched(&self, watch: &ClaimSet) -> Result<PossibleReleasePlan> {
        let mut plan = PossibleReleasePlan::default();
        for &cpu in &watch.cpus {
            let offset = self.layout.count_offset(B_WATCH_CPUS) + cpu * 4;
            if read_u32(&self.header, offset) == 0 {
                plan.cpu_sh.insert(cpu);
            }
            if watch.cpu_mode == ClaimMode::Exclusive {
                let offset = self.layout.count_offset(B_WATCH_CPU_EXCLUSIVE) + cpu * 4;
                if read_u32(&self.header, offset) == 0 {
                    plan.cpu_ex.insert(cpu);
                }
            }
        }
        for &permit in &watch.permits {
            let index = permit_resource_index(permit)?;
            let offset = self.layout.count_offset(B_WATCH_CPUS) + index * 4;
            if read_u32(&self.header, offset) == 0 {
                plan.permit_sh.insert(permit);
            }
            if watch.permit_mode == ClaimMode::Exclusive {
                let offset = self.layout.count_offset(B_WATCH_CPU_EXCLUSIVE) + index * 4;
                if read_u32(&self.header, offset) == 0 {
                    plan.permit_ex.insert(permit);
                }
            }
        }
        for &llc in &watch.llcs {
            let offset = self.layout.count_offset(B_WATCH_LLCS) + llc * 4;
            if read_u32(&self.header, offset) == 0 {
                plan.llc_sh.insert(llc);
            }
            if watch.llc_mode == ClaimMode::Exclusive {
                let offset = self.layout.count_offset(B_WATCH_LLC_EXCLUSIVE) + llc * 4;
                if read_u32(&self.header, offset) == 0 {
                    plan.llc_ex.insert(llc);
                }
            }
        }
        Ok(plan)
    }

    fn aggregate_watch(&self) -> Result<ClaimSet> {
        let llcs = self.bitmap_indices(B_WATCH_LLCS);
        let (cpus, permits) = split_cpu_permit_indices(self.bitmap_indices(B_WATCH_CPUS));
        let (exclusive_cpus, exclusive_permits) =
            split_cpu_permit_indices(self.bitmap_indices(B_WATCH_CPU_EXCLUSIVE));
        let llc_mode = if self.bitmap_indices(B_WATCH_LLC_EXCLUSIVE).is_empty() {
            ClaimMode::Shared
        } else {
            ClaimMode::Exclusive
        };
        let cpu_mode = if exclusive_cpus.is_empty() {
            ClaimMode::Shared
        } else {
            ClaimMode::Exclusive
        };
        let permit_mode = if exclusive_permits.is_empty() {
            ClaimMode::Shared
        } else {
            ClaimMode::Exclusive
        };
        Ok(ClaimSet::with_all_claim_modes(
            llcs,
            cpus,
            permits,
            llc_mode,
            cpu_mode,
            permit_mode,
        ))
    }

    fn watched_subset(
        &self,
        cpus: &BTreeSet<usize>,
        llcs: &BTreeSet<usize>,
        permits: &BTreeSet<usize>,
    ) -> Result<(BTreeSet<usize>, BTreeSet<usize>, BTreeSet<usize>)> {
        // Release notifications are observations, not claims. They may cover
        // resources outside this registry layout (including resources no live
        // ticket has ever watched), so intersect with the valid aggregate
        // watch before touching layout-indexed bitmaps. The SH fast path does
        // the same intersection; falling back after a stale coordinator
        // heartbeat must preserve that behavior rather than rejecting an
        // otherwise irrelevant event index.
        let watch_llcs = self.bitmap_indices(B_WATCH_LLCS);
        let (watch_cpus, watch_permits) =
            split_cpu_permit_indices(self.bitmap_indices(B_WATCH_CPUS));
        Ok((
            cpus.intersection(&watch_cpus).copied().collect(),
            llcs.intersection(&watch_llcs).copied().collect(),
            permits.intersection(&watch_permits).copied().collect(),
        ))
    }

    fn adjust_aggregate_bit(&mut self, which: usize, bit: usize, add: bool) -> Result<()> {
        if bit >= self.layout.bits {
            anyhow::bail!("resource index {bit} exceeds queue registry capacity");
        }
        let count_offset = self.layout.count_offset(which) + bit * 4;
        let old = read_u32(&self.header, count_offset);
        let new = if add {
            old.checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("queue aggregate reference count overflow"))?
        } else {
            old.checked_sub(1).ok_or_else(|| {
                anyhow::anyhow!("queue aggregate reference count underflow at bit {bit}")
            })?
        };
        write_u32(&mut self.header, count_offset, new);
        let word_offset = self.layout.bitset_offset(which) + (bit / 64) * 8;
        let mut word = read_u64(&self.header, word_offset);
        if new == 0 {
            word &= !(1u64 << (bit % 64));
        } else {
            word |= 1u64 << (bit % 64);
        }
        write_u64(&mut self.header, word_offset, word);
        Ok(())
    }

    fn adjust_aggregate_count(&mut self, which: usize, bit: usize, add: bool) -> Result<()> {
        if bit >= self.layout.bits {
            anyhow::bail!("resource index {bit} exceeds queue registry capacity");
        }
        let count_offset = self.layout.count_offset(which) + bit * 4;
        let old = read_u32(&self.header, count_offset);
        let new = if add {
            old.checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("queue aggregate reference count overflow"))?
        } else {
            old.checked_sub(1).ok_or_else(|| {
                anyhow::anyhow!("queue aggregate reference count underflow at bit {bit}")
            })?
        };
        write_u32(&mut self.header, count_offset, new);
        Ok(())
    }

    fn repair_consistency_if_needed(&mut self) -> Result<()> {
        if atomic_u64(&self.header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0 {
            self.repair_consistency()?;
        }
        Ok(())
    }

    fn repair_consistency(&mut self) -> Result<()> {
        atomic_u64(&self.header, H_AGGREGATE_DIRTY).store(1, Ordering::SeqCst);
        let repair_now = monotonic_now_ns()?.max(1);
        let replan_wave_started = self.replan_wave_started_ns();
        let replan_wave_deadline = self.replan_wave_deadline_ns();
        let expire_replan_wave = replan_wave_started != 0
            && replan_wave_started <= repair_now
            && replan_wave_deadline >= replan_wave_started
            && replan_wave_deadline <= repair_now;
        let next_slot = self.next_slot()?;
        let mut records = Vec::new();
        let mut tickets = BTreeSet::new();
        let mut free_head = NONE_SLOT;
        let mut max_ticket = 0u64;
        for slot in (0..next_slot).rev() {
            let record = self.record(slot)?;
            let valid = record.as_ref().is_some_and(|record| {
                record.ticket != 0
                    && !record.claim.is_empty()
                    && (record.state != STATE_PENDING || !record.watch.is_empty())
            });
            if valid {
                let record = record.expect("checked above");
                if !tickets.insert(record.ticket) {
                    anyhow::bail!(
                        "cannot repair queue registry v{VERSION}: duplicate active ticket {}",
                        record.ticket
                    );
                }
                max_ticket = max_ticket.max(record.ticket);
                records.push(record);
            } else {
                let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
                    anyhow::anyhow!("queue slot {slot} disappeared during repair")
                })?;
                write_u32(bytes, R_STATE, STATE_FREE);
                write_u64(bytes, R_TICKET, 0);
                write_u64(bytes, R_NEXT_FREE, free_head);
                free_head = slot;
            }
        }
        write_u64(&mut self.header, H_FREE_HEAD, free_head);
        let next_ticket = read_u64(&self.header, H_NEXT_TICKET)
            .max(max_ticket.saturating_add(1))
            .max(1);
        write_u64(&mut self.header, H_NEXT_TICKET, next_ticket);
        let ticket_high_water = next_ticket.saturating_sub(1);
        let replan_cursor = read_u64(&self.header, H_REPLAN_CURSOR).min(ticket_high_water);
        let replan_horizon = read_u64(&self.header, H_REPLAN_HORIZON)
            .min(ticket_high_water)
            .max(replan_cursor);
        write_u64(&mut self.header, H_REPLAN_HORIZON, replan_horizon);
        write_u64(&mut self.header, H_REPLAN_CURSOR, replan_cursor);
        // Rebuild the derived in-flight count from the records which survived
        // validation. Subsequent demotion through `set_record_state` drains it
        // to zero before the repaired image is declared clean.
        write_u64(
            &mut self.header,
            H_REPLAN_OUTSTANDING,
            records
                .iter()
                .filter(|record| record.state == STATE_REPLAN)
                .count() as u64,
        );

        for which in 0..HEADER_BITMAPS {
            let bitset = self.layout.bitset_offset(which);
            self.header[bitset..bitset + self.layout.words * 8].fill(0);
        }
        for which in 0..AGGREGATE_BITMAPS {
            let counts = self.layout.count_offset(which);
            self.header[counts..counts + self.layout.bits * 4].fill(0);
        }
        for which in 0..REQUEST_ARRAYS {
            let requests = self.layout.request_offset(which);
            self.header[requests..requests + self.layout.bits * 8].fill(0);
        }
        write_u64(&mut self.header, H_PENDING_FLAGS, PENDING_RESCAN);
        write_u64(&mut self.header, H_DEFERRED_RESCAN_DEADLINE_NS, 0);
        for record in &records {
            self.adjust_claim_counts(&record.claim, true)?;
            if charged_state(record.state) {
                // Recharge surviving GRANTED/REVOKED records over the
                // just-zeroed count arrays. Mandatory: the demotion loop
                // below releases through the `set_record_state` chokepoint
                // and would underflow without this recompute.
                self.adjust_granted_occupancy(&record.claim, true)?;
            }
            if record.state == STATE_HELD {
                self.adjust_held_counts(&record.claim, true)?;
            } else {
                self.adjust_watch_counts(&record.watch, true)?;
            }
            // An interrupted transaction may leave a persisted preparation
            // blocker outside the immutable final-run watch. Reconstruct its
            // event-only reference so the ordinary blocker clearing below can
            // balance it without underflowing the aggregate watch counts.
            if let Some(marker) = Self::external_blocker(record) {
                self.adjust_external_event_watch(marker, true)?;
            }
        }
        records.sort_by_key(|record| record.ticket);
        write_u64(
            &mut self.header,
            H_ACTIVE_HEAD,
            records.first().map_or(NONE_SLOT, |record| record.slot),
        );
        write_u64(
            &mut self.header,
            H_ACTIVE_TAIL,
            records.last().map_or(NONE_SLOT, |record| record.slot),
        );
        for (index, record) in records.iter().enumerate() {
            let previous = index
                .checked_sub(1)
                .map_or(NONE_SLOT, |previous| records[previous].slot);
            let next = records.get(index + 1).map_or(NONE_SLOT, |next| next.slot);
            let bytes = self
                .record_bytes_mut(record.slot)?
                .ok_or_else(|| anyhow::anyhow!("queue slot {} disappeared", record.slot))?;
            write_u64(bytes, R_PREV_ACTIVE, previous);
            write_u64(bytes, R_NEXT_ACTIVE, next);
        }
        let watched_cpus = records
            .iter()
            .flat_map(|record| record.watch.cpus.iter().copied())
            .collect();
        let watched_llcs = records
            .iter()
            .flat_map(|record| record.watch.llcs.iter().copied())
            .collect();
        let watched_permits = records
            .iter()
            .flat_map(|record| record.watch.permits.iter().copied())
            .collect();
        self.mark_unknown(&watched_cpus, &watched_llcs, &watched_permits)?;
        self.set_urgent_rescan();
        let previous = self.coordinator_ticket();
        let coordinator = records
            .iter()
            .find(|record| {
                record.ticket == previous
                    && !matches!(record.state, STATE_HELD | STATE_PENDING | STATE_REVOKED)
                    && record.state != STATE_REPLAN_EXPIRED
                    && !(expire_replan_wave && record.state == STATE_REPLAN)
            })
            .or_else(|| {
                records
                    .iter()
                    .filter(|record| {
                        !matches!(record.state, STATE_HELD | STATE_PENDING | STATE_REVOKED)
                            && record.state != STATE_REPLAN_EXPIRED
                            && !(expire_replan_wave && record.state == STATE_REPLAN)
                    })
                    .min_by_key(|record| record.ticket)
            });
        let (coordinator, coordinator_slot) = coordinator
            .map(|record| (record.ticket, record.slot))
            .unwrap_or((0, NONE_SLOT));
        self.set_coordinator(coordinator, coordinator_slot)?;
        let repair_claim_epoch = self.advance_claim_epoch()?;
        let mut prefix = AggregateSnapshot::empty(self.layout);
        let prefix_bits = prefix.bits;
        for record in &records {
            if record.state == STATE_REPLAN_EXPIRED
                || (expire_replan_wave && record.state == STATE_REPLAN)
            {
                self.set_record_state(record.slot, STATE_REPLAN_EXPIRED)?;
                self.clear_record_blocked(record.slot)?;
                self.invalidate_record_prefix(record.slot)?;
                continue;
            }
            if matches!(record.state, STATE_PENDING | STATE_REVOKED) {
                self.set_record_state(record.slot, record.state)?;
                self.clear_record_blocked(record.slot)?;
                add_claim_bits(
                    &record.claim,
                    &mut prefix.cpu_any,
                    &mut prefix.cpu_exclusive,
                    &mut prefix.llc_any,
                    &mut prefix.llc_exclusive,
                    prefix_bits,
                )?;
                continue;
            }
            if record.state == STATE_HELD {
                self.set_record_state(record.slot, STATE_HELD)?;
                self.clear_record_blocked(record.slot)?;
                add_claim_bits(
                    &record.claim,
                    &mut prefix.cpu_any,
                    &mut prefix.cpu_exclusive,
                    &mut prefix.llc_any,
                    &mut prefix.llc_exclusive,
                    prefix_bits,
                )?;
                continue;
            }
            let issue_serial = self.max_watch_serial(&record.watch)?;
            let state = if record.ticket == coordinator {
                STATE_COORDINATOR
            } else if record.state == STATE_COORDINATOR_STANDBY {
                STATE_COORDINATOR_STANDBY
            } else {
                STATE_WAITING
            };
            self.publish_prefix(
                record.slot,
                &prefix,
                R_REPLAN_CLAIM_EPOCH,
                repair_claim_epoch,
                issue_serial,
            )?;
            self.set_record_state(record.slot, state)?;
            self.clear_record_blocked(record.slot)?;
            if state == STATE_WAITING && claim_is_flexible(&record.claim, &record.watch) {
                // Recovery cannot prove which speculative callbacks reached
                // user code before the interrupted transaction. Rebuild all
                // flexible tickets as ordinary waiters and let the pending
                // authoritative scan publish one fresh finite REPLAN wave.
                self.invalidate_record_prefix(record.slot)?;
            }
            if state != STATE_COORDINATOR_STANDBY {
                add_claim_bits(
                    &record.claim,
                    &mut prefix.cpu_any,
                    &mut prefix.cpu_exclusive,
                    &mut prefix.llc_any,
                    &mut prefix.llc_exclusive,
                    prefix_bits,
                )?;
            }
        }
        anyhow::ensure!(
            self.replan_outstanding() == 0,
            "queue repair left speculative callbacks outstanding after demotion",
        );
        self.clear_replan_wave_clock();
        // Any interrupted mutation invalidates outstanding grants. They were
        // demoted above; a fresh coordinator pass will stamp the new epoch.
        self.finish_claim_scan();
        // Repair can unblock pending-registration waiters even when there is
        // no coordinator record to slot-wake (for example a registry made
        // entirely of PENDING/HELD/REVOKED records). Publish the structural
        // generation and wake every such waiter before declaring the rebuilt
        // image clean.
        self.advance_generation_and_wake_pending()?;
        // A torn GRANTED -> REVOKED publication may have died before its
        // targeted wake. REVOKED remains a predecessor fence through repair,
        // so wake every such owner to force prompt acknowledgement rather
        // than stranding successors behind a sleeping fence.
        for record in records
            .iter()
            .filter(|record| record.state == STATE_REVOKED)
        {
            self.wake_slot(record.slot)?;
        }
        for record in records.iter().filter(|record| {
            record.state == STATE_REPLAN_EXPIRED
                || (expire_replan_wave && record.state == STATE_REPLAN)
        }) {
            self.wake_slot(record.slot)?;
        }
        if let Some(record) = records.iter().find(|record| record.ticket == coordinator) {
            self.wake_slot(record.slot)?;
        }
        atomic_u64(&self.header, H_AGGREGATE_DIRTY).store(0, Ordering::SeqCst);
        // A killed writer may have torn coordinator/standby state after the
        // last event edge. Once the repaired image is clean, always wake the
        // real inotify transport as well as targeted slot futexes so no
        // coordinator remains parked until the long recovery fallback.
        notify_coordinator();
        Ok(())
    }

    fn ensure_chunk(&mut self, chunk: u64) -> Result<()> {
        let record_size = self.layout.record_size;
        if let std::collections::btree_map::Entry::Vacant(entry) = self.chunks.entry(chunk) {
            let file = open_chunk_file(chunk, record_size)?;
            let map = unsafe { MmapMut::map_mut(&file) }?;
            entry.insert(map);
        }
        Ok(())
    }

    fn record_bytes(&mut self, slot: u64) -> Result<Option<&[u8]>> {
        let next_slot = self.next_slot()?;
        if slot >= next_slot {
            return Ok(None);
        }
        let (chunk, range) = record_range(slot, self.layout.record_size)?;
        self.ensure_chunk(chunk)?;
        let map = self.chunks.get(&chunk).expect("chunk inserted");
        if range.end > map.len() {
            anyhow::bail!(
                "queue registry chunk {chunk} is too short for slot {slot}: {} bytes < {}",
                map.len(),
                range.end
            );
        }
        Ok(Some(&map[range]))
    }

    fn record_bytes_mut(&mut self, slot: u64) -> Result<Option<&mut [u8]>> {
        let next_slot = self.next_slot()?;
        if slot >= next_slot {
            return Ok(None);
        }
        let (chunk, range) = record_range(slot, self.layout.record_size)?;
        self.ensure_chunk(chunk)?;
        let map = self.chunks.get_mut(&chunk).expect("chunk inserted");
        if range.end > map.len() {
            anyhow::bail!(
                "queue registry chunk {chunk} is too short for slot {slot}: {} bytes < {}",
                map.len(),
                range.end
            );
        }
        Ok(Some(&mut map[range]))
    }
}

fn initialize_header_file(path: &PathBuf, layout: HeaderLayout) -> Result<File> {
    let initializer = tempfile::Builder::new()
        .prefix(INITIALIZER_PREFIX)
        .tempfile_in(registry_data_dir())
        .context("create queue registry initializer")?;
    let file = initializer.as_file();
    file.set_len(
        u64::try_from(layout.header_size)
            .context("queue registry header size does not fit a file length")?,
    )?;
    {
        let mut header = unsafe { MmapMut::map_mut(file) }?;
        header.fill(0);
        write_u32(&mut header, H_VERSION, VERSION);
        write_u32(&mut header, H_WORDS, layout.words as u32);
        write_u32(&mut header, H_RECORD_SIZE, layout.record_size as u32);
        write_u32(&mut header, H_RECORDS_PER_CHUNK, RECORDS_PER_CHUNK as u32);
        write_u64(&mut header, H_NEXT_TICKET, 1);
        write_u64(&mut header, H_FREE_HEAD, NONE_SLOT);
        write_u64(&mut header, H_GLOBAL_SERIAL, 1);
        write_u64(&mut header, H_CLAIM_EPOCH, 1);
        write_u64(&mut header, H_COORDINATOR_EPOCH, 1);
        write_u64(&mut header, H_MIN_CHANGED_TICKET, u64::MAX);
        write_u64(&mut header, H_MIN_CHANGED_TICKET_REPLAN, u64::MAX);
        write_u64(&mut header, H_COORDINATOR_SLOT, NONE_SLOT);
        write_u64(&mut header, H_OBSERVATION_REQUEST, 1);
        write_u64(&mut header, H_ACTIVE_HEAD, NONE_SLOT);
        write_u64(&mut header, H_ACTIVE_TAIL, NONE_SLOT);
        write_u64(&mut header, H_REPLAN_CURSOR, 0);
        write_u64(&mut header, H_REPLAN_HORIZON, 0);
        write_u64(&mut header, H_REPLAN_OUTSTANDING, 0);
        write_u64(&mut header, H_REPLAN_WAVE_STARTED_NS, 0);
        write_u64(&mut header, H_REPLAN_WAVE_DEADLINE_NS, 0);
        write_u64(&mut header, H_DEFERRED_RESCAN_DEADLINE_NS, 0);
        write_u64(
            &mut header,
            H_REPLAN_CAPACITY,
            u64::try_from(host_planner_capacity())
                .context("host planner capacity does not fit queue header")?,
        );
        write_u64(&mut header, H_LAST_PROGRESS_NS, monotonic_now_ns()?);
        // Magic is published last in an inode nobody else can name. The
        // registry path changes only after the complete mapping is flushed.
        write_u64(&mut header, H_MAGIC, MAGIC);
        header.flush()?;
    }
    crash_at_for_tests("initialize_before_publish");
    initializer.persist(path).map_err(|error| {
        anyhow::anyhow!(
            "publish initialized queue registry {}: {}",
            path.display(),
            error.error
        )
    })
}

fn remove_stale_initializers() -> Result<()> {
    let dir = registry_data_dir();
    match std::fs::read_dir(&dir) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry.with_context(|| {
                    format!(
                        "read admission registry directory entry in {}",
                        dir.display()
                    )
                })?;
                if !entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(INITIALIZER_PREFIX)
                {
                    continue;
                }
                match std::fs::remove_file(entry.path()) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!(
                                "remove stale admission registry initializer {}",
                                entry.path().display()
                            )
                        });
                    }
                }
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("scan admission registry directory {}", dir.display())),
    }
}

fn record_range(slot: u64, record_size: usize) -> Result<(u64, std::ops::Range<usize>)> {
    if slot >= MAX_REGISTRY_SLOTS {
        anyhow::bail!(
            "queue registry slot {slot} exceeds the supported maximum {}",
            MAX_REGISTRY_SLOTS - 1
        );
    }
    let chunk = slot / RECORDS_PER_CHUNK as u64;
    let index = usize::try_from(slot % RECORDS_PER_CHUNK as u64)
        .context("queue registry record index does not fit this process")?;
    let start = index
        .checked_mul(record_size)
        .ok_or_else(|| anyhow::anyhow!("queue registry record offset overflow"))?;
    let end = start
        .checked_add(record_size)
        .ok_or_else(|| anyhow::anyhow!("queue registry record end overflow"))?;
    Ok((chunk, start..end))
}

fn open_chunk_file(chunk: u64, record_size: usize) -> Result<File> {
    let max_chunk = (MAX_REGISTRY_SLOTS - 1) / RECORDS_PER_CHUNK as u64;
    if chunk > max_chunk {
        anyhow::bail!(
            "queue registry chunk index {chunk} exceeds the supported maximum {max_chunk}"
        );
    }
    let path = chunk_path(chunk);
    crate::flock::materialize(&path)?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .with_context(|| format!("open queue registry chunk {}", path.display()))?;
    let required = record_size
        .checked_mul(RECORDS_PER_CHUNK)
        .ok_or_else(|| anyhow::anyhow!("queue registry chunk size overflow"))
        .and_then(|bytes| {
            u64::try_from(bytes).context("queue registry chunk size does not fit a file length")
        })?;
    let len = file.metadata()?.len();
    if len == 0 {
        file.set_len(required)?;
    } else if len < required {
        anyhow::bail!(
            "queue registry chunk {} is truncated: {len} < {required}",
            path.display()
        );
    }
    Ok(file)
}

fn ticket_is_live(slot: u64, ticket: u64) -> Result<bool> {
    use rustix::fs::{FlockOperation, flock};

    #[cfg(test)]
    LIVENESS_PROBES.with(|probes| probes.set(probes.get().saturating_add(1)));
    let path = liveness_path(slot, ticket);
    let file = match OpenOptions::new().read(true).open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("open admission ticket liveness {}", path.display()));
        }
    };
    match flock(&file, FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => Ok(false),
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => Ok(true),
        Err(errno) => Err(std::io::Error::from_raw_os_error(errno.raw_os_error()))
            .with_context(|| format!("probe admission ticket liveness {}", path.display())),
    }
}

/// With no elected coordinator, the active head may legitimately be preparing
/// a PENDING run, executing a GRANTED acquisition or REPLAN callback, or
/// holding its acquired resources. That in-flight record is the authoritative
/// progress owner; treating coordinator==0 as failure convoys every timeout
/// and conflicting fast probe through the EX recovery path.
fn shared_live_inflight_head(header: &[u8], layout: HeaderLayout, next_slot: u64) -> Result<bool> {
    let slot = read_u64(header, H_ACTIVE_HEAD);
    if slot == NONE_SLOT {
        return Ok(false);
    }
    if slot >= next_slot {
        anyhow::bail!(
            "queue registry v{VERSION} active head slot {slot} is outside 0..{next_slot}"
        );
    }
    let (chunk, range) = record_range(slot, layout.record_size)?;
    let path = chunk_path(chunk);
    let file = File::open(&path)
        .with_context(|| format!("open queue registry chunk {}", path.display()))?;
    let map = unsafe { Mmap::map(&file) }
        .with_context(|| format!("map queue registry chunk {}", path.display()))?;
    if range.end > map.len() {
        anyhow::bail!("queue registry chunk {chunk} is too short for active head slot {slot}");
    }
    let bytes = &map[range];
    let state = read_u32(bytes, R_STATE);
    if !matches!(
        state,
        STATE_PENDING | STATE_GRANTED | STATE_REVOKED | STATE_REPLAN | STATE_HELD
    ) {
        return Ok(false);
    }
    let ticket = read_u64(bytes, R_TICKET);
    Ok(ticket != 0 && ticket_is_live(slot, ticket)?)
}

fn decode_mode(bytes: &[u8], offset: usize, slot: u64, label: &str) -> Result<ClaimMode> {
    match read_u32(bytes, offset) {
        0 => Ok(ClaimMode::Shared),
        1 => Ok(ClaimMode::Exclusive),
        mode => {
            anyhow::bail!("queue registry v{VERSION} slot {slot} has invalid {label} mode {mode}")
        }
    }
}

fn encode_admission_class(class: AdmissionClass) -> u32 {
    match class {
        AdmissionClass::Ordinary => 0,
        AdmissionClass::DefaultBorrow => 1,
        AdmissionClass::Build => 2,
    }
}

fn decode_admission_class(
    bytes: &[u8],
    offset: usize,
    slot: u64,
    label: &str,
) -> Result<AdmissionClass> {
    match read_u32(bytes, offset) {
        0 => Ok(AdmissionClass::Ordinary),
        1 => Ok(AdmissionClass::DefaultBorrow),
        2 => Ok(AdmissionClass::Build),
        class => anyhow::bail!(
            "queue registry v{VERSION} slot {slot} has invalid {label} admission class {class}"
        ),
    }
}

fn decode_blocked_on(bytes: &[u8], slot: u64) -> Result<Option<BlockedOn>> {
    match read_u32(bytes, R_BLOCK_KIND) {
        BLOCK_NONE => Ok(None),
        kind @ (BLOCK_CPU | BLOCK_LLC | BLOCK_PERMIT) => {
            let index = usize::try_from(read_u64(bytes, R_BLOCK_INDEX))
                .context("blocked resource index does not fit this process")?;
            let key = match kind {
                BLOCK_CPU => ResourceKey::Cpu(index),
                BLOCK_LLC => ResourceKey::Llc(index),
                BLOCK_PERMIT => ResourceKey::Permit(index),
                _ => unreachable!("block kind matched above"),
            };
            let mode = match read_u32(bytes, R_BLOCK_MODE) {
                0 => FlockMode::Shared,
                1 => FlockMode::Exclusive,
                mode => anyhow::bail!(
                    "queue registry v{VERSION} slot {slot} has invalid blocker mode {mode}"
                ),
            };
            Ok(Some(BlockedOn {
                key,
                mode,
                serial: read_u64(bytes, R_BLOCKED_SERIAL),
            }))
        }
        kind => {
            anyhow::bail!("queue registry v{VERSION} slot {slot} has invalid blocker kind {kind}")
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BitsetWordSpan {
    start: u16,
    end: u16,
}

impl BitsetWordSpan {
    fn for_indices(indices: &BTreeSet<usize>) -> Result<Self> {
        let Some(first) = indices.iter().next().copied() else {
            return Ok(Self { start: 0, end: 0 });
        };
        let last = indices
            .iter()
            .next_back()
            .copied()
            .expect("non-empty exact bitset has a last index");
        Ok(Self {
            start: u16::try_from(first / 64)
                .context("exact claim first nonzero word does not fit u16")?,
            end: u16::try_from(last / 64 + 1)
                .context("exact claim last nonzero word does not fit u16")?,
        })
    }

    fn validate(self, layout: HeaderLayout, slot: u64, class: &str) -> Result<()> {
        let start = usize::from(self.start);
        let end = usize::from(self.end);
        if start > end || end > layout.words || (start == end && start != 0) {
            anyhow::bail!(
                "queue registry v{VERSION} slot {slot} has invalid {class} exact-word span {start}..{end} for {} words",
                layout.words,
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScanMetadata {
    flags: u32,
    watch_cpus: u32,
    watch_llcs: u32,
    watch_permits: u32,
    watch_cooperative_permits: u32,
    claim_cpu_span: BitsetWordSpan,
    claim_llc_span: BitsetWordSpan,
    claim_permit_span: BitsetWordSpan,
    watch_identity: u64,
    claim_identity: u64,
}

fn fixed_claim_identity(claim: &impl ClaimView) -> u64 {
    fn hash_u64(hasher: &mut ahash::AHasher, value: u64) {
        hasher.write(&value.to_le_bytes());
    }

    fn hash_indices(hasher: &mut ahash::AHasher, len: usize, indices: impl Iterator<Item = usize>) {
        hash_u64(hasher, len as u64);
        for index in indices {
            hash_u64(hasher, index as u64);
        }
    }

    let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
    hash_indices(&mut hasher, claim.cpu_len(), claim.cpus());
    hash_indices(&mut hasher, claim.llc_len(), claim.llcs());
    hash_indices(&mut hasher, claim.permit_len(), claim.permits());
    hasher.write(&[
        u8::from(claim.cpu_mode() == ClaimMode::Exclusive),
        u8::from(claim.llc_mode() == ClaimMode::Exclusive),
        u8::from(claim.permit_mode() == ClaimMode::Exclusive),
    ]);
    hasher.write(&encode_admission_class(claim.admission_class()).to_le_bytes());
    hasher.finish()
}

fn scan_watch_modes(watch: &ClaimSet) -> EncodedWatchModes {
    EncodedWatchModes {
        cpu: if watch.cpus.is_empty() {
            ClaimMode::Exclusive
        } else {
            watch.cpu_mode
        },
        llc: if watch.llcs.is_empty() {
            ClaimMode::Exclusive
        } else {
            watch.llc_mode
        },
        permit: if watch.permits.is_empty() {
            ClaimMode::Exclusive
        } else {
            watch.permit_mode
        },
    }
}

fn scan_claim_is_flexible(claim: &ClaimSet, watch: &ClaimSet) -> bool {
    let modes = scan_watch_modes(watch);
    claim.cpus != watch.cpus
        || claim.llcs != watch.llcs
        || claim.permits != watch.permits
        || claim.cpu_mode != modes.cpu
        || claim.llc_mode != modes.llc
        || claim.permit_mode != modes.permit
        || claim.admission_class != watch.admission_class
}

impl ScanMetadata {
    fn for_claims(claim: &ClaimSet, watch: &ClaimSet) -> Result<Self> {
        let watch_empty =
            watch.cpus.is_empty() && watch.llcs.is_empty() && watch.permits.is_empty();
        let mut flags = 0;
        if scan_claim_is_flexible(claim, watch) {
            flags |= SCAN_FLAG_FLEXIBLE;
        }
        if watch_empty {
            flags |= SCAN_FLAG_WATCH_EMPTY;
        }
        // A preparation intent unions the token pool into its watch; a run
        // reservation never watches the token sub-range. The pool is the
        // topmost permit sub-range, so any watched permit at or above its base
        // marks this record as a token-consuming preparation intent. This is
        // the same kind of topology-boundary lookup `watch_cooperative_permits`
        // below already performs, and it is a pure function of `watch`.
        if watch
            .permits
            .range(super::super::preparation_token_pool_start()?..)
            .next()
            .is_some()
        {
            flags |= SCAN_FLAG_PREPARATION_INTENT;
        }
        Ok(Self {
            flags,
            watch_cpus: u32::try_from(watch.cpus.len())
                .context("watch CPU cardinality does not fit u32")?,
            watch_llcs: u32::try_from(watch.llcs.len())
                .context("watch LLC cardinality does not fit u32")?,
            watch_permits: u32::try_from(watch.permits.len())
                .context("watch permit cardinality does not fit u32")?,
            watch_cooperative_permits: u32::try_from(
                watch
                    .permits
                    .range(..super::super::cooperative_cpu_permit_end())
                    .count(),
            )
            .context("cooperative watch permit cardinality does not fit u32")?,
            claim_cpu_span: BitsetWordSpan::for_indices(&claim.cpus)?,
            claim_llc_span: BitsetWordSpan::for_indices(&claim.llcs)?,
            claim_permit_span: BitsetWordSpan::for_indices(&claim.permits)?,
            watch_identity: fixed_claim_identity(watch),
            claim_identity: fixed_claim_identity(claim),
        })
    }

    fn read(bytes: &[u8], layout: HeaderLayout, slot: u64) -> Result<Self> {
        let metadata = Self {
            flags: read_u32(bytes, R_SCAN_FLAGS),
            watch_cpus: read_u32(bytes, R_WATCH_CPU_COUNT),
            watch_llcs: read_u32(bytes, R_WATCH_LLC_COUNT),
            watch_permits: read_u32(bytes, R_WATCH_PERMIT_COUNT),
            watch_cooperative_permits: read_u32(bytes, R_WATCH_COOPERATIVE_PERMIT_COUNT),
            claim_cpu_span: BitsetWordSpan {
                start: read_u16(bytes, R_CLAIM_CPU_WORD_START),
                end: read_u16(bytes, R_CLAIM_CPU_WORD_END),
            },
            claim_llc_span: BitsetWordSpan {
                start: read_u16(bytes, R_CLAIM_LLC_WORD_START),
                end: read_u16(bytes, R_CLAIM_LLC_WORD_END),
            },
            claim_permit_span: BitsetWordSpan {
                start: read_u16(bytes, R_CLAIM_PERMIT_WORD_START),
                end: read_u16(bytes, R_CLAIM_PERMIT_WORD_END),
            },
            watch_identity: read_u64(bytes, R_WATCH_IDENTITY),
            claim_identity: read_u64(bytes, R_CLAIM_IDENTITY),
        };
        if metadata.flags & !SCAN_FLAGS_VALID != 0 {
            anyhow::bail!(
                "queue registry v{VERSION} slot {slot} has unknown scan flags {:#x}",
                metadata.flags,
            );
        }
        for (class, count) in [
            ("CPU", metadata.watch_cpus),
            ("LLC", metadata.watch_llcs),
            ("permit", metadata.watch_permits),
        ] {
            if usize::try_from(count).unwrap_or(usize::MAX) > layout.bits {
                anyhow::bail!(
                    "queue registry v{VERSION} slot {slot} has {count} watched {class} resources for {} bits",
                    layout.bits,
                );
            }
        }
        if metadata.watch_cooperative_permits > metadata.watch_permits
            || usize::try_from(metadata.watch_cooperative_permits).unwrap_or(usize::MAX)
                > super::super::cooperative_cpu_permit_end()
        {
            anyhow::bail!(
                "queue registry v{VERSION} slot {slot} has invalid cooperative watch permit count {} of {}",
                metadata.watch_cooperative_permits,
                metadata.watch_permits,
            );
        }
        let watch_empty =
            metadata.watch_cpus == 0 && metadata.watch_llcs == 0 && metadata.watch_permits == 0;
        if watch_empty != (metadata.flags & SCAN_FLAG_WATCH_EMPTY != 0) {
            anyhow::bail!(
                "queue registry v{VERSION} slot {slot} has inconsistent empty-watch scan metadata"
            );
        }
        metadata.claim_cpu_span.validate(layout, slot, "CPU")?;
        metadata.claim_llc_span.validate(layout, slot, "LLC")?;
        metadata
            .claim_permit_span
            .validate(layout, slot, "permit")?;
        Ok(metadata)
    }

    fn write(self, bytes: &mut [u8]) {
        write_u32(bytes, R_SCAN_FLAGS, self.flags);
        write_u32(bytes, R_WATCH_CPU_COUNT, self.watch_cpus);
        write_u32(bytes, R_WATCH_LLC_COUNT, self.watch_llcs);
        write_u32(bytes, R_WATCH_PERMIT_COUNT, self.watch_permits);
        write_u32(
            bytes,
            R_WATCH_COOPERATIVE_PERMIT_COUNT,
            self.watch_cooperative_permits,
        );
        write_u16(bytes, R_CLAIM_CPU_WORD_START, self.claim_cpu_span.start);
        write_u16(bytes, R_CLAIM_CPU_WORD_END, self.claim_cpu_span.end);
        write_u16(bytes, R_CLAIM_LLC_WORD_START, self.claim_llc_span.start);
        write_u16(bytes, R_CLAIM_LLC_WORD_END, self.claim_llc_span.end);
        write_u16(
            bytes,
            R_CLAIM_PERMIT_WORD_START,
            self.claim_permit_span.start,
        );
        write_u16(bytes, R_CLAIM_PERMIT_WORD_END, self.claim_permit_span.end);
        write_u64(bytes, R_WATCH_IDENTITY, self.watch_identity);
        write_u64(bytes, R_CLAIM_IDENTITY, self.claim_identity);
    }

    fn validate_full(
        self,
        layout: HeaderLayout,
        slot: u64,
        claim: &ClaimSet,
        watch: &ClaimSet,
    ) -> Result<()> {
        let expected = Self::for_claims(claim, watch)?;
        if self != expected {
            anyhow::bail!(
                "queue registry v{VERSION} slot {slot} scan metadata does not match its authoritative claim/watch: stored={self:?}, expected={expected:?}"
            );
        }
        self.claim_cpu_span.validate(layout, slot, "CPU")?;
        self.claim_llc_span.validate(layout, slot, "LLC")?;
        self.claim_permit_span.validate(layout, slot, "permit")?;
        Ok(())
    }
}

fn decode_bitset_span(
    bytes: &[u8],
    layout: HeaderLayout,
    which: usize,
    span: BitsetWordSpan,
    slot: u64,
    class: &str,
) -> Result<SmallVec<[usize; SCAN_INLINE_RESOURCES]>> {
    span.validate(layout, slot, class)?;
    let mut out = SmallVec::new();
    let start = usize::from(span.start);
    let end = usize::from(span.end);
    let offset = record_bitset_offset(layout, which);
    for word_index in start..end {
        #[cfg(test)]
        SCAN_EXACT_WORD_READS.with(|reads| reads.set(reads.get().saturating_add(1)));
        let mut word = read_u64(bytes, offset + word_index * std::mem::size_of::<u64>());
        if (word_index == start || word_index + 1 == end) && word == 0 {
            anyhow::bail!(
                "queue registry v{VERSION} slot {slot} has zero boundary word in {class} exact-word span {start}..{end}"
            );
        }
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            out.push(word_index * 64 + bit);
            word &= word - 1;
        }
    }
    Ok(out)
}

fn encoded_bitset_contains(bytes: &[u8], layout: HeaderLayout, which: usize, index: usize) -> bool {
    if index >= layout.bits {
        return false;
    }
    let offset = record_bitset_offset(layout, which) + index / 64 * std::mem::size_of::<u64>();
    read_u64(bytes, offset) & (1u64 << (index % 64)) != 0
}

fn contention_marker_within_encoded_watch(
    marker: ContentionMarker,
    bytes: &[u8],
    layout: HeaderLayout,
    modes: EncodedWatchModes,
) -> bool {
    let (which, index, mode) = match marker.blocker {
        ResourceKey::Cpu(index) => (RB_WATCH_CPUS, index, modes.cpu),
        ResourceKey::Llc(index) => (RB_WATCH_LLCS, index, modes.llc),
        ResourceKey::Permit(index) => (RB_WATCH_PERMITS, index, modes.permit),
    };
    encoded_bitset_contains(bytes, layout, which, index)
        && (mode == ClaimMode::Exclusive || ClaimMode::from(marker.mode) == mode)
}

fn decode_scan_record(bytes: &[u8], layout: HeaderLayout, slot: u64) -> Result<ScanRecord> {
    let claim_llc_mode = decode_mode(bytes, R_CLAIM_LLC_MODE, slot, "exact LLC claim")?;
    let claim_cpu_mode = decode_mode(bytes, R_CLAIM_CPU_MODE, slot, "exact CPU claim")?;
    let claim_permit_mode = decode_mode(bytes, R_CLAIM_PERMIT_MODE, slot, "exact permit claim")?;
    let watch_llc_mode = decode_mode(bytes, R_WATCH_LLC_MODE, slot, "LLC watch")?;
    let watch_cpu_mode = decode_mode(bytes, R_WATCH_CPU_MODE, slot, "CPU watch")?;
    let watch_permit_mode = decode_mode(bytes, R_WATCH_PERMIT_MODE, slot, "permit watch")?;
    let claim_class = decode_admission_class(bytes, R_CLAIM_CLASS, slot, "exact claim")?;
    let metadata = ScanMetadata::read(bytes, layout, slot)?;
    let llcs = decode_bitset_span(
        bytes,
        layout,
        RB_CLAIM_LLCS,
        metadata.claim_llc_span,
        slot,
        "LLC",
    )?;
    let cpus = decode_bitset_span(
        bytes,
        layout,
        RB_CLAIM_CPUS,
        metadata.claim_cpu_span,
        slot,
        "CPU",
    )?;
    let permits = decode_bitset_span(
        bytes,
        layout,
        RB_CLAIM_PERMITS,
        metadata.claim_permit_span,
        slot,
        "permit",
    )?;
    let claim = ScanClaim {
        llc_mode: if llcs.is_empty() {
            ClaimMode::Exclusive
        } else {
            claim_llc_mode
        },
        cpu_mode: if cpus.is_empty() {
            ClaimMode::Exclusive
        } else {
            claim_cpu_mode
        },
        permit_mode: if permits.is_empty() {
            ClaimMode::Exclusive
        } else {
            claim_permit_mode
        },
        llcs,
        cpus,
        permits,
        admission_class: claim_class,
    };
    #[cfg(test)]
    if claim.llcs.spilled() || claim.cpus.spilled() || claim.permits.spilled() {
        SCAN_CLAIM_HEAP_SPILLS.with(|spills| spills.set(spills.get().saturating_add(1)));
    }
    if fixed_claim_identity(&claim) != metadata.claim_identity {
        anyhow::bail!(
            "queue registry v{VERSION} slot {slot} sparse exact claim does not match its persisted identity"
        );
    }
    let watch_cpus =
        usize::try_from(metadata.watch_cpus).context("watch CPU cardinality does not fit usize")?;
    let watch_llcs =
        usize::try_from(metadata.watch_llcs).context("watch LLC cardinality does not fit usize")?;
    let watch_permits = usize::try_from(metadata.watch_permits)
        .context("watch permit cardinality does not fit usize")?;
    let watch_modes = EncodedWatchModes {
        cpu: if watch_cpus == 0 {
            ClaimMode::Exclusive
        } else {
            watch_cpu_mode
        },
        llc: if watch_llcs == 0 {
            ClaimMode::Exclusive
        } else {
            watch_llc_mode
        },
        permit: if watch_permits == 0 {
            ClaimMode::Exclusive
        } else {
            watch_permit_mode
        },
    };
    let flexible = metadata.flags & SCAN_FLAG_FLEXIBLE != 0;
    let preparation_intent = metadata.flags & SCAN_FLAG_PREPARATION_INTENT != 0;
    let watch_empty = metadata.flags & SCAN_FLAG_WATCH_EMPTY != 0;
    if read_u32(bytes, R_STATE) == STATE_PENDING && watch_empty {
        anyhow::bail!("queue registry v{VERSION} slot {slot} is PENDING with an empty watch");
    }
    let cooperative_permits = usize::try_from(metadata.watch_cooperative_permits)
        .context("cooperative watch permit cardinality does not fit usize")?;
    let maximum_backfill_capacity =
        u32::try_from(cooperative_permits.max(watch_cpus.max(watch_llcs).max(1)))
            .unwrap_or(u32::MAX);
    let backfill_capacity = read_u32(bytes, R_BACKFILL_CAPACITY);
    if backfill_capacity > maximum_backfill_capacity {
        anyhow::bail!(
            "queue registry v{VERSION} slot {slot} has invalid backfill capacity \
             {backfill_capacity} > {maximum_backfill_capacity}"
        );
    }
    let blocked_on = decode_blocked_on(bytes, slot)?;
    let external_blocker = blocked_on.and_then(|blocked| {
        let marker = ContentionMarker {
            blocker: blocked.key,
            mode: blocked.mode,
        };
        (!contention_marker_within_encoded_watch(marker, bytes, layout, watch_modes))
            .then_some(marker)
    });
    Ok(ScanRecord {
        slot,
        state: read_u32(bytes, R_STATE),
        ticket: read_u64(bytes, R_TICKET),
        claim,
        watch_modes,
        watch_identity: metadata.watch_identity,
        flexible,
        preparation_intent,
        blocked_on,
        external_blocker,
        issue_serial: read_u64(bytes, R_ISSUE_SERIAL),
        replan_claim_epoch: read_u64(bytes, R_REPLAN_CLAIM_EPOCH),
        grant_epoch: read_u64(bytes, R_GRANT_EPOCH),
        prefix_epoch: read_u64(bytes, R_PREFIX_EPOCH),
        backfill_capacity,
        backfill_started_ns: read_u64(bytes, R_BACKFILL_STARTED_NS),
        prev_active: read_u64(bytes, R_PREV_ACTIVE),
        next_active: read_u64(bytes, R_NEXT_ACTIVE),
    })
}

fn decode_record(bytes: &[u8], layout: HeaderLayout, slot: u64) -> Result<Record> {
    let watch_llc_mode = decode_mode(bytes, R_WATCH_LLC_MODE, slot, "LLC watch")?;
    let claim_llc_mode = decode_mode(bytes, R_CLAIM_LLC_MODE, slot, "exact LLC claim")?;
    let watch_cpu_mode = decode_mode(bytes, R_WATCH_CPU_MODE, slot, "CPU watch")?;
    let claim_cpu_mode = decode_mode(bytes, R_CLAIM_CPU_MODE, slot, "exact CPU claim")?;
    let watch_permit_mode = decode_mode(bytes, R_WATCH_PERMIT_MODE, slot, "permit watch")?;
    let claim_permit_mode = decode_mode(bytes, R_CLAIM_PERMIT_MODE, slot, "exact permit claim")?;
    let claim = ClaimSet::with_all_claim_modes(
        decode_bitset(bytes, record_bitset_offset(layout, RB_CLAIM_LLCS), layout)?,
        decode_bitset(bytes, record_bitset_offset(layout, RB_CLAIM_CPUS), layout)?,
        decode_bitset(
            bytes,
            record_bitset_offset(layout, RB_CLAIM_PERMITS),
            layout,
        )?,
        claim_llc_mode,
        claim_cpu_mode,
        claim_permit_mode,
    )
    .with_admission_class(decode_admission_class(
        bytes,
        R_CLAIM_CLASS,
        slot,
        "exact claim",
    )?);
    #[cfg(test)]
    FULL_WATCH_MATERIALIZATIONS.with(|count| count.set(count.get().saturating_add(1)));
    let watch = ClaimSet::with_all_claim_modes(
        decode_bitset(bytes, record_bitset_offset(layout, RB_WATCH_LLCS), layout)?,
        decode_bitset(bytes, record_bitset_offset(layout, RB_WATCH_CPUS), layout)?,
        decode_bitset(
            bytes,
            record_bitset_offset(layout, RB_WATCH_PERMITS),
            layout,
        )?,
        watch_llc_mode,
        watch_cpu_mode,
        watch_permit_mode,
    )
    .with_admission_class(decode_admission_class(bytes, R_WATCH_CLASS, slot, "watch")?);
    ScanMetadata::read(bytes, layout, slot)?.validate_full(layout, slot, &claim, &watch)?;
    let blocked_on = decode_blocked_on(bytes, slot)?;
    let backfill_capacity = read_u32(bytes, R_BACKFILL_CAPACITY);
    let maximum_backfill_capacity = backfill_capacity_for_watch(&watch);
    if backfill_capacity > maximum_backfill_capacity {
        anyhow::bail!(
            "queue registry v{VERSION} slot {slot} has invalid backfill capacity \
             {backfill_capacity} > {maximum_backfill_capacity}"
        );
    }
    Ok(Record {
        slot,
        state: read_u32(bytes, R_STATE),
        ticket: read_u64(bytes, R_TICKET),
        pid: read_u32(bytes, R_PID),
        claim,
        watch,
        blocked_on,
        issue_serial: read_u64(bytes, R_ISSUE_SERIAL),
        replan_claim_epoch: read_u64(bytes, R_REPLAN_CLAIM_EPOCH),
        grant_epoch: read_u64(bytes, R_GRANT_EPOCH),
        prefix_epoch: read_u64(bytes, R_PREFIX_EPOCH),
        backfill_capacity,
        backfill_started_ns: read_u64(bytes, R_BACKFILL_STARTED_NS),
        prev_active: read_u64(bytes, R_PREV_ACTIVE),
        next_active: read_u64(bytes, R_NEXT_ACTIVE),
    })
}

fn encode_claim(
    bytes: &mut [u8],
    layout: HeaderLayout,
    claim: &ClaimSet,
    watch: &ClaimSet,
) -> Result<()> {
    encode_exact_claim(bytes, layout, claim)?;
    encode_bitset(
        bytes,
        record_bitset_offset(layout, RB_WATCH_CPUS),
        layout,
        &watch.cpus,
    )?;
    encode_bitset(
        bytes,
        record_bitset_offset(layout, RB_WATCH_LLCS),
        layout,
        &watch.llcs,
    )?;
    encode_bitset(
        bytes,
        record_bitset_offset(layout, RB_WATCH_PERMITS),
        layout,
        &watch.permits,
    )?;
    ScanMetadata::for_claims(claim, watch)?.write(bytes);
    Ok(())
}

fn encode_exact_claim(bytes: &mut [u8], layout: HeaderLayout, claim: &ClaimSet) -> Result<()> {
    encode_bitset(
        bytes,
        record_bitset_offset(layout, RB_CLAIM_CPUS),
        layout,
        &claim.cpus,
    )?;
    encode_bitset(
        bytes,
        record_bitset_offset(layout, RB_CLAIM_LLCS),
        layout,
        &claim.llcs,
    )?;
    encode_bitset(
        bytes,
        record_bitset_offset(layout, RB_CLAIM_PERMITS),
        layout,
        &claim.permits,
    )
}

fn encode_prefix(bytes: &mut [u8], layout: HeaderLayout, prefix: &AggregateSnapshot) -> Result<()> {
    if prefix.bits != layout.bits {
        anyhow::bail!("queue predecessor prefix layout does not match its record");
    }
    for (which, words) in [
        (RB_PREFIX_CPU_ANY, prefix.cpu_any.as_slice()),
        (RB_PREFIX_CPU_EXCLUSIVE, prefix.cpu_exclusive.as_slice()),
        (RB_PREFIX_LLC_ANY, prefix.llc_any.as_slice()),
        (RB_PREFIX_LLC_EXCLUSIVE, prefix.llc_exclusive.as_slice()),
    ] {
        let offset = record_bitset_offset(layout, which);
        for (word, value) in words.iter().copied().enumerate() {
            write_u64(bytes, offset + word * std::mem::size_of::<u64>(), value);
        }
    }
    Ok(())
}

fn clear_record_claim_bits(bytes: &mut [u8], layout: HeaderLayout) {
    let start = record_bitset_offset(layout, RB_CLAIM_CPUS);
    let end = record_bitset_offset(layout, RB_WATCH_CPUS);
    bytes[start..end].fill(0);
}

fn clear_record_watch_bits(bytes: &mut [u8], layout: HeaderLayout) {
    // The capacity is derived from the immutable watch. Reset it before
    // publishing an empty watch so every intermediate record remains
    // decodable if a coordinator scan observes this transition. Activation
    // overwrites it from the replacement watch; HELD records keep the
    // canonical empty-watch capacity.
    write_u32(bytes, R_BACKFILL_CAPACITY, 1);
    write_u64(bytes, R_BACKFILL_STARTED_NS, 0);
    let start = record_bitset_offset(layout, RB_WATCH_CPUS);
    let end = record_bitset_offset(layout, RB_PREFIX_CPU_ANY);
    bytes[start..end].fill(0);
}

fn record_bitset_offset(layout: HeaderLayout, which: usize) -> usize {
    debug_assert!(which < RECORD_BITMAPS);
    R_BITS + which * layout.words * std::mem::size_of::<u64>()
}

fn encode_bitset(
    bytes: &mut [u8],
    offset: usize,
    layout: HeaderLayout,
    indices: &BTreeSet<usize>,
) -> Result<()> {
    for &index in indices {
        if index >= layout.bits {
            anyhow::bail!("resource index {index} exceeds queue registry capacity");
        }
        let word_offset = offset + index / 64 * 8;
        let word = read_u64(bytes, word_offset) | (1u64 << (index % 64));
        write_u64(bytes, word_offset, word);
    }
    Ok(())
}

fn decode_bitset(bytes: &[u8], offset: usize, layout: HeaderLayout) -> Result<BTreeSet<usize>> {
    let mut out = BTreeSet::new();
    for word_index in 0..layout.words {
        let mut word = read_u64(bytes, offset + word_index * 8);
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            out.insert(word_index * 64 + bit);
            word &= word - 1;
        }
    }
    Ok(out)
}

fn decode_header_bitset(header: &[u8], layout: HeaderLayout, which: usize) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    for word_index in 0..layout.words {
        let mut word = read_u64(
            header,
            layout.bitset_offset(which) + word_index * std::mem::size_of::<u64>(),
        );
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            out.insert(word_index * 64 + bit);
            word &= word - 1;
        }
    }
    out
}

fn header_bitmap_bit(header: &[u8], layout: HeaderLayout, which: usize, index: usize) -> bool {
    if index >= layout.bits {
        return false;
    }
    let offset = layout.bitset_offset(which) + index / 64 * std::mem::size_of::<u64>();
    read_u64(header, offset) & (1u64 << (index % 64)) != 0
}

fn claim_conflicts_bits(
    claim: &impl ClaimView,
    cpu_any: &[u64],
    cpu_exclusive: &[u64],
    llc_any: &[u64],
    llc_exclusive: &[u64],
    bits: usize,
) -> Result<bool> {
    let cpu_fence = if claim.cpu_mode() == ClaimMode::Exclusive {
        cpu_any
    } else {
        cpu_exclusive
    };
    for cpu in claim.cpus() {
        if cpu >= bits {
            anyhow::bail!("CPU index {cpu} exceeds queue registry capacity");
        }
        if cpu_fence[cpu / 64] & (1u64 << (cpu % 64)) != 0 {
            return Ok(true);
        }
    }
    let permit_fence = if claim.permit_mode() == ClaimMode::Exclusive {
        cpu_any
    } else {
        cpu_exclusive
    };
    for permit in claim.permits() {
        let index = permit_resource_index(permit)?;
        if index >= bits {
            anyhow::bail!("permit index {permit} exceeds queue registry capacity");
        }
        if permit_fence[index / 64] & (1u64 << (index % 64)) != 0 {
            return Ok(true);
        }
    }
    let llc_fence = if claim.llc_mode() == ClaimMode::Exclusive {
        llc_any
    } else {
        llc_exclusive
    };
    for llc in claim.llcs() {
        if llc >= bits {
            anyhow::bail!("LLC index {llc} exceeds queue registry capacity");
        }
        if llc_fence[llc / 64] & (1u64 << (llc % 64)) != 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn claims_conflict(a: &impl ClaimView, b: &impl ClaimView) -> bool {
    let incompatible = |a_mode: ClaimMode, b_mode: ClaimMode| {
        a_mode == ClaimMode::Exclusive || b_mode == ClaimMode::Exclusive
    };
    (incompatible(a.cpu_mode(), b.cpu_mode()) && a.cpus().any(|cpu| b.contains_cpu(cpu)))
        || (incompatible(a.permit_mode(), b.permit_mode())
            && a.permits().any(|permit| b.contains_permit(permit)))
        || (incompatible(a.llc_mode(), b.llc_mode()) && a.llcs().any(|llc| b.contains_llc(llc)))
}

/// Measure one backfill wave in the same CPU-permit units that bound
/// cooperative VM oversubscription. A production VM watch contains the whole
/// cooperative permit pool, so this capacity cannot stop admission before that
/// pool itself is full. CPU/LLC width is the fallback for build claims and
/// synthetic/test claims which use another permit namespace.
fn backfill_capacity_for_watch(watch: &ClaimSet) -> u32 {
    let cooperative_end = super::super::cooperative_cpu_permit_end();
    let permit_units = watch.permits.range(..cooperative_end).count();
    let physical_units = watch.cpus.len().max(watch.llcs.len()).max(1);
    u32::try_from(permit_units.max(physical_units)).unwrap_or(u32::MAX)
}

fn backfill_cost_for_claim(claim: &impl ClaimView) -> u32 {
    let cooperative_end = super::super::cooperative_cpu_permit_end();
    let permit_units = claim
        .permits()
        .filter(|permit| *permit < cooperative_end)
        .count();
    let physical_units = claim.cpu_len().max(claim.llc_len()).max(1);
    u32::try_from(if permit_units == 0 {
        physical_units
    } else {
        permit_units
    })
    .unwrap_or(u32::MAX)
}

fn add_claim_bits(
    claim: &impl ClaimView,
    cpu_any: &mut [u64],
    cpu_exclusive: &mut [u64],
    llc_any: &mut [u64],
    llc_exclusive: &mut [u64],
    bits: usize,
) -> Result<()> {
    for cpu in claim.cpus() {
        if cpu >= bits {
            anyhow::bail!("CPU index {cpu} exceeds queue registry capacity");
        }
        cpu_any[cpu / 64] |= 1u64 << (cpu % 64);
        if claim.cpu_mode() == ClaimMode::Exclusive {
            cpu_exclusive[cpu / 64] |= 1u64 << (cpu % 64);
        }
    }
    for permit in claim.permits() {
        let index = permit_resource_index(permit)?;
        if index >= bits {
            anyhow::bail!("permit index {permit} exceeds queue registry capacity");
        }
        cpu_any[index / 64] |= 1u64 << (index % 64);
        if claim.permit_mode() == ClaimMode::Exclusive {
            cpu_exclusive[index / 64] |= 1u64 << (index % 64);
        }
    }
    for llc in claim.llcs() {
        if llc >= bits {
            anyhow::bail!("LLC index {llc} exceeds queue registry capacity");
        }
        llc_any[llc / 64] |= 1u64 << (llc % 64);
        if claim.llc_mode() == ClaimMode::Exclusive {
            llc_exclusive[llc / 64] |= 1u64 << (llc % 64);
        }
    }
    Ok(())
}

fn aggregate_map_conflicts(map: &[u8], layout: HeaderLayout, candidate: &ClaimSet) -> Result<bool> {
    let cpu_which = if candidate.cpu_mode == ClaimMode::Exclusive {
        B_CLAIM_CPUS
    } else {
        B_CLAIM_CPU_EXCLUSIVE
    };
    for &cpu in &candidate.cpus {
        if cpu >= layout.bits {
            anyhow::bail!(
                "CPU index {cpu} exceeds queue registry v{VERSION} capacity {}",
                layout.bits
            );
        }
        let word = read_u64(map, layout.bitset_offset(cpu_which) + cpu / 64 * 8);
        if word & (1u64 << (cpu % 64)) != 0 {
            return Ok(true);
        }
    }
    let permit_which = if candidate.permit_mode == ClaimMode::Exclusive {
        B_CLAIM_CPUS
    } else {
        B_CLAIM_CPU_EXCLUSIVE
    };
    for &permit in &candidate.permits {
        let index = permit_resource_index(permit)?;
        if index >= layout.bits {
            anyhow::bail!(
                "permit index {permit} exceeds queue registry v{VERSION} capacity {}",
                layout.bits
            );
        }
        let word = read_u64(map, layout.bitset_offset(permit_which) + index / 64 * 8);
        if word & (1u64 << (index % 64)) != 0 {
            return Ok(true);
        }
    }
    let which = if candidate.llc_mode == ClaimMode::Exclusive {
        B_CLAIM_LLC_ANY
    } else {
        B_CLAIM_LLC_EXCLUSIVE
    };
    for &llc in &candidate.llcs {
        if llc >= layout.bits {
            anyhow::bail!(
                "LLC index {llc} exceeds queue registry v{VERSION} capacity {}",
                layout.bits
            );
        }
        let word = read_u64(map, layout.bitset_offset(which) + llc / 64 * 8);
        if word & (1u64 << (llc % 64)) != 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn claim_intersects_watch_map(map: &[u8], layout: HeaderLayout, claim: &ClaimSet) -> Result<bool> {
    for (&index, which) in claim
        .cpus
        .iter()
        .map(|index| (index, B_WATCH_CPUS))
        .chain(claim.llcs.iter().map(|index| (index, B_WATCH_LLCS)))
    {
        if index >= layout.bits {
            continue;
        }
        let word = read_u64(map, layout.bitset_offset(which) + index / 64 * 8);
        if word & (1u64 << (index % 64)) != 0 {
            return Ok(true);
        }
    }
    for &permit in &claim.permits {
        let index = permit_resource_index(permit)?;
        if index >= layout.bits {
            continue;
        }
        let word = read_u64(map, layout.bitset_offset(B_WATCH_CPUS) + index / 64 * 8);
        if word & (1u64 << (index % 64)) != 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn monotonic_now_ns() -> Result<u64> {
    let mut timestamp: libc::timespec = unsafe { std::mem::zeroed() };
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut timestamp) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let seconds =
        u64::try_from(timestamp.tv_sec).context("CLOCK_MONOTONIC returned negative seconds")?;
    let nanoseconds = u64::try_from(timestamp.tv_nsec)
        .context("CLOCK_MONOTONIC returned negative nanoseconds")?;
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanoseconds))
        .ok_or_else(|| anyhow::anyhow!("CLOCK_MONOTONIC nanosecond value overflow"))
}

fn coordinator_activity_is_fresh(header: &[u8], now: u64) -> bool {
    let heartbeat = read_u64(header, H_COORDINATOR_HEARTBEAT_NS);
    heartbeat != 0 && heartbeat <= now && now - heartbeat < COORDINATOR_HEARTBEAT_LEASE_NS
}

fn coordinator_heartbeat_due_in_from_header(header: &[u8], now: u64) -> Duration {
    let heartbeat = read_u64(header, H_COORDINATOR_HEARTBEAT_NS);
    if heartbeat == 0 || heartbeat > now {
        return Duration::ZERO;
    }
    Duration::from_nanos(COORDINATOR_HEARTBEAT_INTERVAL_NS.saturating_sub(now - heartbeat))
}

fn deferred_rescan_due_in_from_header(header: &[u8], now: u64) -> Option<Duration> {
    if read_u64(header, H_PENDING_FLAGS) & PENDING_REPLAN_RESCAN == 0 {
        return None;
    }
    let deadline = read_u64(header, H_DEFERRED_RESCAN_DEADLINE_NS);
    if deadline == 0
        || deadline <= now
        || deadline > now.saturating_add(deferred_rescan_interval_ns())
    {
        Some(Duration::ZERO)
    } else {
        Some(Duration::from_nanos(deadline - now))
    }
}

fn liveness_due_in_from_last(last: u64, now: u64) -> Duration {
    if last == 0 || last > now {
        return Duration::ZERO;
    }
    Duration::from_nanos(LIVENESS_SWEEP_INTERVAL_NS.saturating_sub(now - last))
}

fn replan_wave_due_in_from_header(header: &[u8], now: u64) -> Option<Duration> {
    if read_u64(header, H_REPLAN_OUTSTANDING) == 0 {
        return None;
    }
    let started = read_u64(header, H_REPLAN_WAVE_STARTED_NS);
    let deadline = read_u64(header, H_REPLAN_WAVE_DEADLINE_NS);
    if started == 0 || started > now || deadline < started {
        return Some(Duration::ZERO);
    }
    Some(Duration::from_nanos(deadline.saturating_sub(now)))
}

fn replan_wave_requires_recovery_from_header(header: &[u8], now: u64) -> bool {
    replan_wave_due_in_from_header(header, now).is_some_and(|duration| duration.is_zero())
}

fn liveness_due_in_from_header(header: &[u8], now: u64) -> Duration {
    let periodic = liveness_due_in_from_last(read_u64(header, H_LAST_LIVENESS_SWEEP_NS), now);
    let reconcile_by = read_u64(header, H_LIVENESS_RECONCILE_BY_NS);
    let liveness = if reconcile_by == 0 {
        periodic
    } else {
        periodic.min(Duration::from_nanos(reconcile_by.saturating_sub(now)))
    };
    if let Some(replan) = replan_wave_due_in_from_header(header, now) {
        liveness.min(replan)
    } else {
        liveness
    }
}

fn align_up_checked(value: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        anyhow::bail!("invalid queue registry alignment {alignment}");
    }
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| anyhow::anyhow!("queue registry aligned size overflow"))
}

#[cfg(test)]
fn crash_at_for_tests(point: &str) {
    if std::env::var_os("KTSTR_TEST_REGISTRY_CRASH_POINT").as_deref()
        == Some(std::ffi::OsStr::new(point))
    {
        // `_exit` deliberately bypasses Rust destructors so the test observes
        // the same mmap/flock crash boundary as an abruptly killed cell.
        unsafe { libc::_exit(86) }
    }
}

#[cfg(not(test))]
fn crash_at_for_tests(_point: &str) {}

#[cfg(test)]
fn cancel_granted_commit_for_tests(cancelled: Option<&AtomicBool>) {
    if CANCEL_GRANTED_AFTER_COMMIT.with(|armed| armed.replace(false)) {
        cancelled
            .expect("granted after-commit cancellation hook needs a cancellation flag")
            .store(true, Ordering::SeqCst);
    }
}

#[cfg(not(test))]
fn cancel_granted_commit_for_tests(_cancelled: Option<&AtomicBool>) {}

#[cfg(test)]
fn cancel_coordinator_commit_for_tests(cancelled: Option<&AtomicBool>) {
    if CANCEL_COORDINATOR_AFTER_COMMIT.with(|armed| armed.replace(false)) {
        cancelled
            .expect("coordinator after-commit cancellation hook needs a cancellation flag")
            .store(true, Ordering::SeqCst);
    }
}

#[cfg(not(test))]
fn cancel_coordinator_commit_for_tests(_cancelled: Option<&AtomicBool>) {}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_ne_bytes(bytes[offset..offset + 4].try_into().expect("u32 field"))
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_ne_bytes(bytes[offset..offset + 2].try_into().expect("u16 field"))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_ne_bytes());
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_ne_bytes(bytes[offset..offset + 8].try_into().expect("u64 field"))
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_ne_bytes());
}

fn atomic_u32(bytes: &[u8], offset: usize) -> &AtomicU32 {
    assert_eq!(
        (bytes.as_ptr() as usize + offset) % std::mem::align_of::<AtomicU32>(),
        0
    );
    // SAFETY: alignment is asserted and the mapped field has AtomicU32
    // storage for the mapping's entire lifetime.
    unsafe { &*bytes.as_ptr().add(offset).cast::<AtomicU32>() }
}

fn atomic_u32_mut(bytes: &mut [u8], offset: usize) -> &AtomicU32 {
    atomic_u32(bytes, offset)
}

fn atomic_u64(bytes: &[u8], offset: usize) -> &AtomicU64 {
    assert_eq!(
        (bytes.as_ptr() as usize + offset) % std::mem::align_of::<AtomicU64>(),
        0
    );
    // SAFETY: the mmap base and field offset are naturally aligned, and this
    // header word is accessed atomically for the mapping's entire lifetime.
    unsafe { &*bytes.as_ptr().add(offset).cast::<AtomicU64>() }
}
