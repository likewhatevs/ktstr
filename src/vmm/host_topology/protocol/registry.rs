//! Fixed-record, crash-recoverable admission registry.
//!
//! Arrival, claim replacement, and normal removal touch one fixed record plus
//! the affected aggregate counters; they never rewrite the other waiters.
//! Records live in fixed-size chunk files which grow to the concurrency
//! high-water mark and are never truncated while another process may still
//! have a futex word mapped.

use super::{
    ClaimMode, ClaimSet, ContentionEvidence, ContentionMarker, ContentionSet, ResourceKey,
    interrupted, protocol_dir,
};
use crate::flock::{FlockMode, InterruptibleFlockWaiter, block_flock, try_flock};
use anyhow::{Context, Result};
use memmap2::{Mmap, MmapMut};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

const MAGIC: u64 = 0x4b54_5354_5251_3035; // "KTSTRQ05"
const VERSION: u32 = 5;
const HEADER_FIXED: usize = 256;
const HEADER_ALIGN: usize = 4096;
const RECORD_FIXED: usize = 128;
const RECORD_ALIGN: usize = 64;
const RECORDS_PER_CHUNK: usize = 64;
const NONE_SLOT: u64 = u64::MAX;
const MAX_RESOURCE_BITS: usize = 1 << 20;
const MAX_REGISTRY_SLOTS: u64 = 1 << 16;
const INITIALIZER_PREFIX: &str = ".ktstr-acquire-registry-v5-init-";
const LIVENESS_PREFIX: &str = "ktstr-acquire-v5-slot-";
const LIVENESS_SEPARATOR: &str = "-ticket-";
const LIVENESS_SUFFIX: &str = ".live";

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
const _: () = assert!(H_AGGREGATE_DIRTY % std::mem::align_of::<AtomicU64>() == 0);

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
const R_REPLAN_SERIAL: usize = 64;
const R_REPLAN_CLAIM_EPOCH: usize = 72;
const R_PREV_ACTIVE: usize = 80;
const R_NEXT_ACTIVE: usize = 88;
const R_CLAIM_LLC_MODE: usize = 96;
const R_CLAIM_CPU_MODE: usize = 100;
const R_WATCH_CPU_MODE: usize = 104;
const R_BITS: usize = RECORD_FIXED;

const STATE_FREE: u32 = 0;
const STATE_WAITING: u32 = 1;
const STATE_GRANTED: u32 = 2;
const STATE_COORDINATOR: u32 = 3;
const STATE_REPLAN: u32 = 4;

const BLOCK_NONE: u32 = 0;
const BLOCK_CPU: u32 = 1;
const BLOCK_LLC: u32 = 2;

const PENDING_RESCAN: u64 = 1 << 0;
const PENDING_OBSERVATION: u64 = 1 << 1;

const B_CLAIM_CPUS: usize = 0;
const B_CLAIM_CPU_EXCLUSIVE: usize = 1;
const B_CLAIM_LLC_ANY: usize = 2;
const B_CLAIM_LLC_EXCLUSIVE: usize = 3;
const B_WATCH_CPUS: usize = 4;
const B_WATCH_CPU_EXCLUSIVE: usize = 5;
const B_WATCH_LLCS: usize = 6;
const B_WATCH_LLC_EXCLUSIVE: usize = 7;
const B_CPU_KNOWN: usize = 8;
const B_CPU_SH_AVAILABLE: usize = 9;
const B_CPU_EX_AVAILABLE: usize = 10;
const B_LLC_KNOWN: usize = 11;
const B_LLC_SH_AVAILABLE: usize = 12;
const B_LLC_EX_AVAILABLE: usize = 13;
const B_PENDING_CPU_SH: usize = 14;
const B_PENDING_CPU_EX: usize = 15;
const B_PENDING_LLC_SH: usize = 16;
const B_PENDING_LLC_EX: usize = 17;
const B_CANDIDATE_CPU_SH: usize = 18;
const B_CANDIDATE_CPU_EX: usize = 19;
const B_CANDIDATE_LLC_SH: usize = 20;
const B_CANDIDATE_LLC_EX: usize = 21;
const HEADER_BITMAPS: usize = 22;
const AGGREGATE_BITMAPS: usize = 8;

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

#[cfg(test)]
thread_local! {
    static SHARED_STATE_READS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static SHARED_STATE_RECOVERY_UPGRADES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static REGISTRY_EX_ACQUISITIONS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
    static LIVENESS_PROBES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static CLAIMS_BEFORE_RECORD_READS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    #[cfg(test)]
    static CANCEL_GRANTED_AFTER_COMMIT: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    #[cfg(test)]
    static CANCEL_COORDINATOR_AFTER_COMMIT: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum State {
    Waiting,
    Granted,
    Replan,
    Coordinator,
}

#[derive(Debug)]
pub(super) struct ScheduleSnapshot {
    pub watch: ClaimSet,
    pub should_step: bool,
    pub observation: Option<ObservationRequest>,
    pub liveness_due_in: Duration,
}

#[derive(Debug)]
pub(super) struct GrantAttempt<T> {
    pub acquired: Option<T>,
    pub next_claim: ClaimSet,
    pub contention: Option<ContentionEvidence>,
}

#[derive(Debug)]
pub(super) enum GrantResult<T> {
    Acquired(T),
    Requeued,
    LostGrant,
}

#[derive(Debug, Clone)]
pub(super) struct ObservationRequest {
    pub cpus: BTreeMap<usize, (Option<u64>, Option<u64>)>,
    pub llcs: BTreeMap<usize, (Option<u64>, Option<u64>)>,
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
}

impl PossibleReleasePlan {
    fn is_empty(&self) -> bool {
        self.cpu_sh.is_empty()
            && self.cpu_ex.is_empty()
            && self.llc_sh.is_empty()
            && self.llc_ex.is_empty()
    }

    fn extend(&mut self, other: Self) {
        self.cpu_sh.extend(other.cpu_sh);
        self.cpu_ex.extend(other.cpu_ex);
        self.llc_sh.extend(other.llc_sh);
        self.llc_ex.extend(other.llc_ex);
    }
}

pub(super) enum FenceResult<T> {
    Fenced,
    Ran { value: T, watched: bool },
}

pub(super) struct AggregateSnapshot {
    bits: usize,
    cpu_any: Vec<u64>,
    cpu_exclusive: Vec<u64>,
    llc_any: Vec<u64>,
    llc_exclusive: Vec<u64>,
}

impl AggregateSnapshot {
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
}

#[derive(Debug, Clone)]
struct Record {
    slot: u64,
    state: u32,
    ticket: u64,
    #[cfg(test)]
    pid: u32,
    claim: ClaimSet,
    watch: ClaimSet,
    blocked_on: Option<BlockedOn>,
    replan_serial: u64,
    replan_claim_epoch: u64,
    grant_epoch: u64,
    prev_active: u64,
    next_active: u64,
}

pub(super) struct Ticket {
    slot: u64,
    ticket: u64,
    liveness_path: PathBuf,
    liveness: Option<OwnedFd>,
    wake: Option<FutexSlot>,
    _interrupt_waiter: Option<InterruptibleFlockWaiter>,
    attempt_serial: u64,
    attempt_claim_epoch: u64,
    attempted_claims: Vec<ClaimSet>,
    finished: bool,
}

struct FutexSlot {
    _map: MmapMut,
    ptr: *mut AtomicU32,
}

// The mapping is process-shared and the only concurrently accessed word is an
// aligned AtomicU32.  The remaining record bytes are accessed under the
// registry flock.
unsafe impl Send for FutexSlot {}

impl FutexSlot {
    fn expected(&self) -> u32 {
        // SAFETY: `ptr` is aligned, points into `_map`, and `_map` outlives
        // every access through it.
        unsafe { (&*self.ptr).load(Ordering::Acquire) }
    }

    fn wait(&self, expected: u32, timeout: Duration) -> Result<bool> {
        let ts = libc::timespec {
            tv_sec: timeout.as_secs().try_into().unwrap_or(libc::time_t::MAX),
            tv_nsec: timeout.subsec_nanos().into(),
        };
        // SAFETY: the word is a live, aligned u32 in a MAP_SHARED mapping.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_futex,
                self.ptr.cast::<u32>(),
                libc::FUTEX_WAIT,
                expected,
                &ts as *const libc::timespec,
                std::ptr::null::<u32>(),
                0u32,
            )
        };
        if rc == 0 {
            return Ok(false);
        }
        let error = std::io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::ETIMEDOUT) => Ok(true),
            Some(libc::EAGAIN) | Some(libc::EINTR) => Ok(false),
            _ => Err(error.into()),
        }
    }
}

impl Ticket {
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
        let required_bits = required_resource_bits(&watch);
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
                });
        let contended_exact = exact_blocker.is_some();
        let initial_state = if has_predecessor
            && !contended_exact
            && !conflicts
            && table.claim_availability_compatible(&claim)?
        {
            STATE_GRANTED
        } else if has_predecessor && flexible {
            STATE_REPLAN
        } else {
            STATE_WAITING
        };
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
        )?;
        table.append_active(slot)?;
        crash_at_for_tests("register_record_before_counts");
        table.adjust_claim_counts(&claim, true)?;
        table.adjust_watch_counts(&watch, true)?;
        table.mark_observation_modes(&newly_watched)?;
        table.mark_blockers_unknown(&contention_markers)?;
        if initial_state == STATE_WAITING {
            if let Some(marker) = exact_blocker {
                table.set_record_blocked(
                    slot,
                    marker,
                    blocked_at.expect("initial blocker serial must accompany evidence"),
                )?;
            }
        }
        if initial_state == STATE_GRANTED {
            let claim_epoch = table.claim_epoch();
            table.set_record_grant_epoch(slot, claim_epoch)?;
        }
        table.set_next_ticket(
            ticket
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("queue ticket id overflow"))?,
        );
        table.bump_generation()?;
        table.elect_coordinator_in_transaction()?;
        table.finish_transaction();
        let wake = table.map_futex(slot)?;
        drop(table);
        drop(_lock);
        notify_coordinator();

        Ok(Self {
            slot,
            ticket,
            liveness_path,
            liveness: Some(liveness),
            wake: Some(wake),
            _interrupt_waiter: interrupt_waiter,
            attempt_serial: 0,
            attempt_claim_epoch: 0,
            attempted_claims: Vec::new(),
            finished: false,
        })
    }

    pub(super) fn state(&self, cancelled: Option<&AtomicBool>) -> Result<State> {
        let _lock = lock_registry_interruptible_existing(cancelled)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        table.recover_coordinator_if_dead()?;
        let record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("live queue ticket {} disappeared", self.ticket))?;
        match record.state {
            STATE_WAITING => Ok(State::Waiting),
            STATE_GRANTED => Ok(State::Granted),
            STATE_REPLAN => Ok(State::Replan),
            STATE_COORDINATOR => Ok(State::Coordinator),
            state => anyhow::bail!("queue ticket {} has invalid state {state}", self.ticket),
        }
    }

    fn state_shared(
        &self,
        check_coordinator_liveness: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Option<State>> {
        check_cancelled(cancelled)?;
        let _lock = normalize_cancellation(lock_registry_existing(FlockMode::Shared), cancelled)?;
        #[cfg(test)]
        SHARED_STATE_READS.with(|reads| reads.set(reads.get() + 1));
        let path = header_path();
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let header = unsafe { Mmap::map(&file) }
            .with_context(|| format!("map admission registry header {}", path.display()))?;
        let layout = HeaderLayout::validate(&header)?;
        if atomic_u64(&header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0 {
            // A killed writer left mutable state to repair. The caller drops
            // this SH mapping and upgrades through the rare EX recovery path.
            return Ok(None);
        }
        let next_slot = read_u64(&header, H_NEXT_SLOT);
        if next_slot > MAX_REGISTRY_SLOTS {
            anyhow::bail!(
                "queue registry v{VERSION} next-slot value {next_slot} exceeds the supported maximum {MAX_REGISTRY_SLOTS}"
            );
        }
        if self.slot >= next_slot {
            anyhow::bail!(
                "live queue ticket {} slot {} is outside 0..{next_slot}",
                self.ticket,
                self.slot
            );
        }
        let (chunk, range) = record_range(self.slot, layout.record_size)?;
        let chunk_path = chunk_path(chunk);
        let chunk_file = File::open(&chunk_path)
            .with_context(|| format!("open queue registry chunk {}", chunk_path.display()))?;
        let chunk_map = unsafe { Mmap::map(&chunk_file) }
            .with_context(|| format!("map queue registry chunk {}", chunk_path.display()))?;
        if range.end > chunk_map.len() {
            anyhow::bail!(
                "queue registry chunk {chunk} is too short for slot {}: {} bytes < {}",
                self.slot,
                chunk_map.len(),
                range.end
            );
        }
        let bytes = &chunk_map[range];
        let ticket = read_u64(bytes, R_TICKET);
        if ticket != self.ticket {
            anyhow::bail!(
                "live queue ticket {} disappeared from slot {} (found ticket {ticket})",
                self.ticket,
                self.slot
            );
        }
        let state = match read_u32(bytes, R_STATE) {
            STATE_WAITING => State::Waiting,
            STATE_GRANTED => State::Granted,
            STATE_REPLAN => State::Replan,
            STATE_COORDINATOR => State::Coordinator,
            state => anyhow::bail!("queue ticket {} has invalid state {state}", self.ticket),
        };
        if check_coordinator_liveness && state == State::Waiting {
            let coordinator = read_u64(&header, H_COORDINATOR);
            let coordinator_slot = read_u64(&header, H_COORDINATOR_SLOT);
            let progress_is_live = if coordinator == 0 {
                shared_live_inflight_head(&header, layout, next_slot)?
            } else {
                coordinator_slot != NONE_SLOT
                    && coordinator_slot < next_slot
                    && ticket_is_live(coordinator_slot, coordinator)?
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

    pub(super) fn state_or_wait(
        &self,
        timeout: Duration,
        cancelled: Option<&AtomicBool>,
    ) -> Result<State> {
        check_cancelled(cancelled)?;
        let wake = self
            .wake
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("queue ticket futex mapping was released"))?;
        let expected = wake.expected();
        // Sample the futex before the single registry state read. A grant
        // between these operations changes either the observed state or the
        // futex value, so FUTEX_WAIT cannot sleep past it.
        let state = match self.state_shared(false, cancelled)? {
            Some(state) => state,
            None => self.state(cancelled)?,
        };
        if state != State::Waiting {
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
            match self.state_shared(true, cancelled)? {
                Some(state) => Ok(state),
                None => self.state(cancelled),
            }
        } else {
            Ok(State::Waiting)
        }
    }

    pub(super) fn run_granted<T>(
        &mut self,
        cancelled: Option<&AtomicBool>,
        attempt: impl FnOnce(&ClaimSet, bool, AggregateSnapshot) -> Result<GrantAttempt<T>>,
    ) -> Result<GrantResult<T>> {
        let (designated, watch, acquisition_allowed, predecessors) = {
            let _lock = lock_registry_interruptible_existing(cancelled)?;
            let mut table = Table::open_existing()?;
            table.repair_consistency_if_needed()?;
            let record = table
                .record(self.slot)?
                .filter(|record| record.ticket == self.ticket)
                .ok_or_else(|| anyhow::anyhow!("live queue ticket {} disappeared", self.ticket))?;
            if !matches!(record.state, STATE_GRANTED | STATE_REPLAN) {
                return Ok(GrantResult::LostGrant);
            }
            let claim_epoch = table.claim_epoch();
            if record.state == STATE_GRANTED && record.grant_epoch != claim_epoch {
                if table.min_changed_ticket() < record.ticket {
                    table.set_record_state(self.slot, STATE_WAITING)?;
                    table.clear_record_blocked(self.slot)?;
                    table.set_pending_flag(PENDING_RESCAN);
                    table.bump_generation()?;
                    table.elect_coordinator()?;
                    drop(table);
                    drop(_lock);
                    notify_coordinator();
                    return Ok(GrantResult::LostGrant);
                }
                // Only later tickets changed. Their reservations cannot
                // invalidate this earlier grant, so refresh its stamp without
                // a registry scan.
                table.set_record_grant_epoch(self.slot, claim_epoch)?;
            }
            let watch_serial = table.max_watch_serial(&record.watch)?;
            let prefix_changed = self.attempt_claim_epoch != record.replan_claim_epoch;
            if self.attempt_serial != watch_serial || prefix_changed {
                self.attempt_serial = watch_serial;
                self.attempted_claims.clear();
            }
            self.attempt_claim_epoch = record.replan_claim_epoch;
            if !self.attempted_claims.contains(&record.claim) {
                self.attempted_claims.push(record.claim.clone());
            }
            let predecessors = table.claims_before(record.ticket)?;
            (
                record.claim,
                record.watch,
                record.state == STATE_GRANTED,
                predecessors,
            )
        };

        // The exact designated claim remains published while the expensive
        // host snapshot and resource-flock probe run without the registry EX
        // flock. Disjoint grants can therefore probe concurrently, while the
        // aggregate fence still excludes conflicting fast paths.
        let mut result = normalize_cancellation(
            attempt(&designated, acquisition_allowed, predecessors),
            cancelled,
        )?;

        let lock = lock_registry_interruptible_existing(cancelled)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        let record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("live queue ticket {} disappeared", self.ticket))?;
        let claim_epoch = table.claim_epoch();
        let stale = acquisition_allowed
            && record.grant_epoch != claim_epoch
            && table.min_changed_ticket() < record.ticket;
        let replan_stale =
            !acquisition_allowed && record.replan_claim_epoch != self.attempt_claim_epoch;
        let expected_state = if acquisition_allowed {
            STATE_GRANTED
        } else {
            STATE_REPLAN
        };
        if record.state != expected_state || record.claim != designated || stale || replan_stale {
            if record.state == STATE_GRANTED && stale {
                table.set_record_state(self.slot, STATE_WAITING)?;
                table.clear_record_blocked(self.slot)?;
                table.set_pending_flag(PENDING_RESCAN);
                table.bump_generation()?;
                table.elect_coordinator()?;
            }
            drop(table);
            drop(lock);
            if stale || replan_stale {
                notify_coordinator();
            }
            return Ok(GrantResult::LostGrant);
        }
        if acquisition_allowed && record.grant_epoch != claim_epoch {
            table.set_record_grant_epoch(self.slot, claim_epoch)?;
        }
        check_cancelled(cancelled)?;

        if let Some(acquired) = result.acquired.take() {
            if !acquisition_allowed {
                anyhow::bail!(
                    "replan-only queue wake returned an acquired payload for ticket {}",
                    self.ticket
                );
            }
            crash_at_for_tests("granted_acquired_before_clear");
            // Unmap the per-slot futex before making the slot recyclable.
            self.wake.take();
            // The real resource flocks in `acquired` are already held. Remove
            // the published exact claim under the same EX fence, so a
            // conflicting fast path observes either the claim or those flocks,
            // never a check-to-acquire gap between them.
            table.remove_record(&record, true)?;
            table.bump_generation()?;
            self._interrupt_waiter.take();
            self.finished = true;
            self.liveness.take();
            drop(table);
            drop(lock);
            let _ = std::fs::remove_file(&self.liveness_path);
            notify_coordinator();
            cancel_granted_commit_for_tests(cancelled);
            return Ok(GrantResult::Acquired(acquired));
        }

        validate_claim(&result.next_claim)?;
        validate_claim_within_watch(&result.next_claim, &watch)?;
        if let Some(evidence) = result.contention.as_ref() {
            validate_contention_within_watch(&[evidence.marker()], &watch)?;
        }
        let current_watch_serial = table.max_watch_serial(&watch)?;
        let prefix_changed = record.replan_claim_epoch != self.attempt_claim_epoch;
        if current_watch_serial != self.attempt_serial || prefix_changed {
            self.attempt_serial = current_watch_serial;
            self.attempted_claims.clear();
            self.attempted_claims.push(designated.clone());
        }
        self.attempt_claim_epoch = record.replan_claim_epoch;
        let cycle_exhausted = self.attempted_claims.contains(&result.next_claim);
        let changed = result.next_claim != designated;
        let blocked = if let Some(evidence) = result.contention.as_ref() {
            let marker = evidence.marker();
            Some((marker, table.blocker_serial(marker.blocker, marker.mode)?))
        } else {
            None
        };
        if changed {
            let publish_state = if cycle_exhausted {
                STATE_WAITING
            } else {
                STATE_REPLAN
            };
            table.replace_claim(
                self.slot,
                record.ticket,
                &designated,
                &result.next_claim,
                publish_state,
                if cycle_exhausted {
                    current_watch_serial
                } else {
                    0
                },
                blocked,
                false,
            )?;
            table.wake_slot(self.slot)?;
            if publish_state == STATE_WAITING {
                table.elect_coordinator()?;
            }
        } else {
            table.begin_transaction()?;
            if let Some(evidence) = result.contention.as_ref() {
                let marker = evidence.marker();
                let blocked_at = table.blocker_serial(marker.blocker, marker.mode)?;
                table.mark_blocker_unknown(marker)?;
                table.set_record_blocked(self.slot, marker, blocked_at)?;
            } else {
                table.clear_record_blocked(self.slot)?;
            }
            table.set_record_state(self.slot, STATE_WAITING)?;
            table.set_record_replan_serial(self.slot, current_watch_serial)?;
            table.set_pending_flag(PENDING_RESCAN);
            table.elect_coordinator_in_transaction()?;
            table.finish_transaction();
        }
        table.bump_generation()?;
        drop(table);
        drop(lock);
        notify_coordinator();
        Ok(GrantResult::Requeued)
    }

    pub(super) fn schedule(
        &mut self,
        coordinator_claim: Option<&ClaimSet>,
        closed_cpus: &BTreeSet<usize>,
        closed_llcs: &BTreeSet<usize>,
        overflow: bool,
        newly_held: Option<&ClaimSet>,
        contention: &[ContentionMarker],
        abandoned: Option<&ClaimSet>,
        closed_tickets: &[(u64, u64)],
        reobserve_watch: bool,
        reconcile_liveness_after: Option<Duration>,
        force_liveness_maintenance: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ScheduleSnapshot> {
        let pure_release_batch = coordinator_claim.is_none()
            && (!closed_cpus.is_empty() || !closed_llcs.is_empty())
            && !overflow
            && newly_held.is_none()
            && contention.is_empty()
            && abandoned.is_none()
            && closed_tickets.is_empty()
            && !reobserve_watch
            && reconcile_liveness_after.is_none()
            && !force_liveness_maintenance;
        if pure_release_batch
            && let Some(snapshot) =
                self.try_known_free_release_snapshot(closed_cpus, closed_llcs, cancelled)?
        {
            return Ok(snapshot);
        }

        let _lock = lock_registry_interruptible_existing(cancelled)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        table.prune_dead_identities(closed_tickets)?;
        if let Some(delay) = reconcile_liveness_after {
            table.request_liveness_reconciliation(delay)?;
        }
        let liveness_due_in = table.perform_liveness_sweep_if_due(force_liveness_maintenance)?;
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
        if let Some(claim) = newly_held {
            if !claim.is_empty() {
                validate_claim(claim)?;
                validate_claim_within_watch(claim, &record.watch)?;
            }
        }
        if let Some(claim) = abandoned {
            if !claim.is_empty() {
                validate_claim(claim)?;
                validate_claim_within_watch(claim, &record.watch)?;
            }
        }
        validate_contention_within_watch(contention, &record.watch)?;
        let mut event_cpus;
        let mut event_llcs;
        if overflow {
            let aggregate = table.aggregate_watch()?;
            event_cpus = aggregate.cpus;
            event_llcs = aggregate.llcs;
        } else {
            (event_cpus, event_llcs) = table.watched_subset(closed_cpus, closed_llcs)?;
        }
        if let Some(released) = abandoned {
            event_cpus.extend(released.cpus.iter().copied());
            event_llcs.extend(released.llcs.iter().copied());
        }
        let claim_changed = coordinator_claim.is_some_and(|claim| *claim != record.claim);
        let publish_busy = newly_held.is_some_and(|claim| !claim.is_empty());
        let mut release_plan = table.possible_release_plan(&event_cpus, &event_llcs)?;
        if reobserve_watch {
            // The coordinator installs its inotify watch before this first
            // schedule call. Re-observe only modes that could have become less
            // restrictive before that watch existed: known-free modes cannot
            // improve, while existing unknown/pending work remains durable.
            let watch = table.aggregate_watch()?;
            release_plan.extend(table.possible_release_plan(&watch.cpus, &watch.llcs)?);
        }
        let registry_changed =
            claim_changed || publish_busy || !release_plan.is_empty() || !contention.is_empty();
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
                )?;
            }
            if let Some(claim) = newly_held.filter(|claim| !claim.is_empty()) {
                table.publish_claim_busy(claim)?;
            }
            table.apply_possible_release(&release_plan)?;
            table.mark_blockers_unknown(contention)?;
            table.bump_generation()?;
            table.finish_transaction();
        }
        let should_scan = table.pending_flags() & PENDING_RESCAN != 0;
        let watch = if should_scan {
            table.grant_compatible()?
        } else {
            table.aggregate_watch()?
        };
        Ok(ScheduleSnapshot {
            watch,
            should_step: false,
            observation: table.observation_request()?,
            liveness_due_in,
        })
    }

    fn try_known_free_release_snapshot(
        &self,
        closed_cpus: &BTreeSet<usize>,
        closed_llcs: &BTreeSet<usize>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Option<ScheduleSnapshot>> {
        check_cancelled(cancelled)?;
        let _lock = normalize_cancellation(lock_registry_existing(FlockMode::Shared), cancelled)?;
        let file = File::open(header_path())?;
        let header = unsafe { Mmap::map(&file) }?;
        let layout = HeaderLayout::validate(&header)?;
        if atomic_u64(&header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0
            || read_u64(&header, H_PENDING_FLAGS) != 0
            || read_u64(&header, H_COORDINATOR) != self.ticket
            || read_u64(&header, H_COORDINATOR_SLOT) != self.slot
        {
            return Ok(None);
        }
        let next_slot = read_u64(&header, H_NEXT_SLOT);
        if self.slot >= next_slot {
            anyhow::bail!(
                "coordinator ticket {} slot {} is outside 0..{next_slot}",
                self.ticket,
                self.slot
            );
        }
        let (chunk, range) = record_range(self.slot, layout.record_size)?;
        let chunk_file = File::open(chunk_path(chunk))?;
        let chunk_map = unsafe { Mmap::map(&chunk_file) }?;
        if range.end > chunk_map.len() {
            anyhow::bail!(
                "queue registry chunk {chunk} is too short for coordinator slot {}",
                self.slot
            );
        }
        let record = &chunk_map[range];
        if read_u64(record, R_TICKET) != self.ticket
            || read_u32(record, R_STATE) != STATE_COORDINATOR
        {
            return Ok(None);
        }

        let watch_llcs = decode_header_bitset(&header, layout, B_WATCH_LLCS);
        let watch_cpus = decode_header_bitset(&header, layout, B_WATCH_CPUS);
        let watch_llc_exclusive =
            decode_header_bitset(&header, layout, B_WATCH_LLC_EXCLUSIVE);
        let watch_cpu_exclusive =
            decode_header_bitset(&header, layout, B_WATCH_CPU_EXCLUSIVE);
        let watch = ClaimSet::with_claim_modes(
            watch_llcs,
            watch_cpus,
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
        );
        let watched_cpus = closed_cpus.intersection(&watch.cpus).copied();
        let watched_llcs = closed_llcs.intersection(&watch.llcs).copied();
        let cpus_free = watched_cpus.into_iter().all(|cpu| {
            header_bitmap_bit(&header, layout, B_CPU_KNOWN, cpu)
                && header_bitmap_bit(&header, layout, B_CPU_SH_AVAILABLE, cpu)
                && (!watch_cpu_exclusive.contains(&cpu)
                    || header_bitmap_bit(&header, layout, B_CPU_EX_AVAILABLE, cpu))
        });
        let llcs_free = watched_llcs.into_iter().all(|llc| {
            header_bitmap_bit(&header, layout, B_LLC_KNOWN, llc)
                && header_bitmap_bit(&header, layout, B_LLC_SH_AVAILABLE, llc)
                && (!watch_llc_exclusive.contains(&llc)
                    || header_bitmap_bit(&header, layout, B_LLC_EX_AVAILABLE, llc))
        });
        if !cpus_free || !llcs_free {
            return Ok(None);
        }
        check_cancelled(cancelled)?;
        Ok(Some(ScheduleSnapshot {
            watch,
            should_step: false,
            observation: None,
            liveness_due_in: liveness_due_in_from_header(
                &header,
                monotonic_now_ns()?,
            ),
        }))
    }

    pub(super) fn apply_observation(
        &mut self,
        request: &ObservationRequest,
        observation: &AvailabilityObservation,
        release_proofs: impl FnOnce(),
        cancelled: Option<&AtomicBool>,
    ) -> Result<ScheduleSnapshot> {
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
        let watch_serial_before = table.max_watch_serial(&record.watch)?;
        table.begin_transaction()?;
        table.apply_observation(request, observation)?;
        table.finish_transaction();
        let watch_serial_after = table.max_watch_serial(&record.watch)?;
        // Keep the registry EX fence while dropping proof flocks, then grant
        // from the state they proved. A split release/reacquire lets a fast SH
        // fenced acquirer steal the resource between proof and waiter wake.
        release_proofs();
        let watch = if table.pending_flags() & PENDING_RESCAN != 0 {
            table.grant_compatible()?
        } else {
            table.aggregate_watch()?
        };
        Ok(ScheduleSnapshot {
            watch,
            should_step: watch_serial_after > watch_serial_before,
            observation: table.observation_request()?,
            liveness_due_in: table.liveness_due_in()?,
        })
    }

    #[cfg(test)]
    pub(super) fn read_state_shared_for_tests(&self) -> Result<()> {
        self.state_shared(false, None)?
            .ok_or_else(|| anyhow::anyhow!("test state read unexpectedly required recovery"))?;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn finish(&mut self, cancelled: Option<&AtomicBool>) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        // A slot cannot be recycled while this process still has its futex
        // word mapped.
        self.wake.take();
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

    pub(super) fn finish_acquired(
        &mut self,
        exact: &ClaimSet,
        abandoned: &ClaimSet,
        contention: &[ContentionMarker],
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        validate_claim(exact)?;
        if !abandoned.is_empty() {
            validate_claim(abandoned)?;
        }
        self.wake.take();
        let _lock = lock_registry_interruptible_existing(cancelled)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        let record = table
            .record(self.slot)?
            .filter(|record| record.ticket == self.ticket)
            .ok_or_else(|| anyhow::anyhow!("coordinator ticket {} disappeared", self.ticket))?;
        validate_claim_within_watch(exact, &record.watch)?;
        if !abandoned.is_empty() {
            validate_claim_within_watch(abandoned, &record.watch)?;
        }
        validate_contention_within_watch(contention, &record.watch)?;
        let claim_unchanged = &record.claim == exact;
        table.begin_transaction()?;
        table.publish_claim_busy(exact)?;
        table.mark_possible_release(&abandoned.cpus, &abandoned.llcs)?;
        table.mark_blockers_unknown(contention)?;
        table.remove_record_in_transaction(&record, true)?;
        if !claim_unchanged {
            // Any alternative change invalidates later grants: omitted
            // resources may become runnable, while added resources have
            // become authoritatively busy under the committed real flocks.
            table.mark_claim_changed(record.ticket)?;
        }
        table.elect_coordinator_in_transaction()?;
        if !claim_unchanged {
            table.set_pending_flag(PENDING_RESCAN);
        }
        table.bump_generation()?;
        table.finish_transaction();
        self._interrupt_waiter.take();
        self.finished = true;
        self.liveness.take();
        drop(table);
        drop(_lock);
        let _ = std::fs::remove_file(&self.liveness_path);
        notify_coordinator();
        cancel_coordinator_commit_for_tests(cancelled);
        Ok(())
    }

    #[cfg(test)]
    fn remove_record_interruptible(&mut self, cancelled: Option<&AtomicBool>) -> Result<()> {
        let _lock = lock_registry_interruptible_existing(cancelled)?;
        self.remove_record_locked()
    }

    fn remove_record_locked(&mut self) -> Result<()> {
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        if let Some(record) = table.record(self.slot)?
            && record.ticket == self.ticket
        {
            table.remove_record(&record, false)?;
            table.bump_generation()?;
        }
        Ok(())
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        self.wake.take();
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
    }
}

#[cfg(test)]
pub(super) fn aggregate_conflicts(candidate: &ClaimSet) -> Result<bool> {
    Ok(matches!(
        with_aggregate_fence(candidate, || Ok(()))?,
        FenceResult::Fenced
    ))
}

pub(super) fn aggregate_snapshot(required: &ClaimSet) -> Result<AggregateSnapshot> {
    validate_claim(required)?;
    let required_layout = HeaderLayout::new(required_resource_bits(required))
        .context("aggregate snapshot exceeds admission registry capacity")?;
    loop {
        let Some(shared) = try_lock_registry_existing(FlockMode::Shared)? else {
            return Ok(AggregateSnapshot {
                bits: required_layout.bits,
                cpu_any: vec![0; required_layout.words],
                cpu_exclusive: vec![0; required_layout.words],
                llc_any: vec![0; required_layout.words],
                llc_exclusive: vec![0; required_layout.words],
            });
        };
        let path = header_path();
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(AggregateSnapshot {
                    bits: required_layout.bits,
                    cpu_any: vec![0; required_layout.words],
                    cpu_exclusive: vec![0; required_layout.words],
                    llc_any: vec![0; required_layout.words],
                    llc_exclusive: vec![0; required_layout.words],
                });
            }
            Err(error) => return Err(error).with_context(|| format!("open {}", path.display())),
        };
        if file.metadata()?.len() == 0 {
            return Ok(AggregateSnapshot {
                bits: required_layout.bits,
                cpu_any: vec![0; required_layout.words],
                cpu_exclusive: vec![0; required_layout.words],
                llc_any: vec![0; required_layout.words],
                llc_exclusive: vec![0; required_layout.words],
            });
        }
        let map = unsafe { Mmap::map(&file) }
            .with_context(|| format!("map queue aggregate {}", path.display()))?;
        #[cfg(test)]
        AGGREGATE_SNAPSHOT_READS.with(|reads| reads.set(reads.get() + 1));
        let layout = match HeaderLayout::validate(&map) {
            Ok(layout) => layout,
            Err(_) if map.iter().all(|byte| *byte == 0) => {
                drop(map);
                drop(file);
                drop(shared);
                let _lock = lock_registry_existing(FlockMode::Exclusive)?;
                let mut table = Table::open(required_layout.bits)?;
                table.repair_consistency_if_needed()?;
                continue;
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
        let snapshot = AggregateSnapshot {
            bits: layout.bits,
            cpu_any: read_words(B_CLAIM_CPUS),
            cpu_exclusive: read_words(B_CLAIM_CPU_EXCLUSIVE),
            llc_any: read_words(B_CLAIM_LLC_ANY),
            llc_exclusive: read_words(B_CLAIM_LLC_EXCLUSIVE),
        };
        // The common disjoint snapshot is a pure SH/read-only operation. Only
        // a claim that is actually fenced needs any liveness work.
        if !snapshot.conflicts(required)? {
            return Ok(snapshot);
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
            return Ok(snapshot);
        }
        drop(map);
        drop(file);
        drop(shared);
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        table.recover_coordinator_if_dead()?;
        continue;
    }
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
        .chain(watch.cpus.iter())
        .chain(watch.llcs.iter())
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
    encode_claim(&mut bytes, layout, claim, watch)?;
    let record = decode_record(&bytes, layout, 0)?;
    Ok((record.claim, record.watch))
}

#[cfg(test)]
pub(super) fn shared_state_recovery_upgrade_count_for_tests() -> usize {
    SHARED_STATE_RECOVERY_UPGRADES.with(std::cell::Cell::get)
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
                drop(map);
                drop(file);
                drop(shared);
                // A creator may have died before publishing the initialized
                // inode. Only an entirely zeroed image is unambiguously
                // unpublished; malformed nonzero v5 state is never replaced.
                let _lock = lock_registry_existing(FlockMode::Exclusive)?;
                let mut table = Table::open(required_resource_bits(candidate))?;
                table.repair_consistency_if_needed()?;
                continue;
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
            if coordinator == 0 {
                if shared_live_inflight_head(&map, layout, next_slot)? {
                    return Ok(FenceResult::Fenced);
                }
            } else {
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

#[cfg(test)]
pub(super) fn snapshot() -> Result<Vec<(u64, u32, ClaimSet)>> {
    let _lock = lock_registry_existing(FlockMode::Exclusive)?;
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
pub(super) fn hold_registry_exclusive_for_tests() -> Result<OwnedFd> {
    lock_registry_existing(FlockMode::Exclusive)
}

#[cfg(test)]
pub(super) fn hold_registry_shared_for_tests() -> Result<OwnedFd> {
    lock_registry_existing(FlockMode::Shared)
}

#[cfg(test)]
pub(super) fn shared_state_read_count_for_tests() -> usize {
    SHARED_STATE_READS.with(std::cell::Cell::get)
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
pub(super) fn exercise_known_free_close_storm_for_tests(
    closes: usize,
) -> Result<(usize, u64, usize, u64, u64)> {
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut ticket = Ticket::register(claim.clone(), claim, None)?;
    let empty = BTreeSet::new();
    let initial = ticket.schedule(
        None,
        &empty,
        &empty,
        false,
        None,
        &[],
        None,
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
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let generation_before = diagnostic_counter_for_tests(H_GENERATION)?;
    let ex_before = REGISTRY_EX_ACQUISITIONS.with(std::cell::Cell::get);
    let mut observations = 0;
    let mut planner_steps = 0;
    let closed = BTreeSet::from([1usize, usize::MAX]);
    for _ in 0..closes {
        let snapshot = ticket.schedule(
            None,
            &closed,
            &empty,
            false,
            None,
            &[],
            None,
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
pub(super) fn exercise_llc_sh_only_shared_to_free_close_for_tests(
) -> Result<(bool, bool, u64, u64)> {
    let claim = ClaimSet::new(
        [1usize],
        std::iter::empty(),
        FlockMode::Shared,
    );
    let mut ticket = Ticket::register(claim.clone(), claim, None)?;
    let empty = BTreeSet::new();
    let initial = ticket.schedule(
        None,
        &empty,
        &empty,
        false,
        None,
        &[],
        None,
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
        anyhow::bail!(
            "SH-only LLC watch requested the wrong modes: {requested_modes:?}"
        );
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
        false,
        None,
        &[],
        None,
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
pub(super) fn exercise_busy_to_free_close_for_tests() -> Result<(usize, u64, usize)> {
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut ticket = Ticket::register(claim.clone(), claim, None)?;
    let empty = BTreeSet::new();
    let initial = ticket.schedule(
        None,
        &empty,
        &empty,
        false,
        None,
        &[],
        None,
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
    let closed = BTreeSet::from([1usize]);
    let pending = ticket.schedule(
        None,
        &closed,
        &empty,
        false,
        None,
        &[],
        None,
        &[],
        false,
        None,
        false,
        None,
    )?;
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
    let observations = 1;
    let planner_steps = usize::from(improved.should_step);
    let scans = diagnostic_counter_for_tests(H_GRANT_SCANS)? - scans_before;
    ticket.finish(None)?;
    Ok((observations, scans, planner_steps))
}

#[cfg(test)]
pub(super) fn exercise_llc_ex_contention_shared_wake_for_tests() -> Result<(u64, bool, bool)> {
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
    table.finish_transaction();
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
    let mut observation = AvailabilityObservation::default();
    observation.llcs.insert(
        1,
        LlcObservation {
            availability: LlcAvailability::SharedHeld,
            sh_resolved: true,
            ex_resolved: true,
        },
    );
    table.begin_transaction()?;
    let improved = table.apply_observation(&request, &observation)?;
    table.finish_transaction();
    if !improved || table.pending_flags() & PENDING_RESCAN == 0 {
        anyhow::bail!("shared-held observation after EX contention must publish an SH improvement");
    }
    table.grant_compatible()?;
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
    Ok((scans, shared_granted, exclusive_waiting))
}

#[cfg(test)]
pub(super) fn exercise_cpu_ex_contention_shared_wake_for_tests(
) -> Result<(u64, bool, bool, bool, bool, bool, bool)> {
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
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
    let exclusive_claim =
        ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut exclusive =
        Ticket::register(exclusive_claim.clone(), exclusive_claim, None)?;

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
    table.finish_transaction();
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
    table.begin_transaction()?;
    let improved = table.apply_observation(&request, &observation)?;
    table.finish_transaction();
    if !improved || table.pending_flags() & PENDING_RESCAN == 0 {
        anyhow::bail!(
            "shared-held CPU observation after EX contention must publish an SH improvement"
        );
    }
    table.grant_compatible()?;
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
    Ok((
        scans,
        shared_granted,
        exclusive_waiting,
        sh_serial_advanced,
        ex_serial_unchanged,
        shared_woke,
        exclusive_not_woken,
    ))
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
        for (index, ticket) in tickets.iter().enumerate() {
            table.set_record_state(
                ticket.slot,
                if index == 0 {
                    STATE_COORDINATOR
                } else {
                    STATE_WAITING
                },
            )?;
            table.clear_record_blocked(ticket.slot)?;
        }
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
            false,
            None,
            &[],
            None,
            &[],
            true,
            Some(Duration::from_secs(60 * 60)),
            false,
            None,
        )?;
        let reconcile_by =
            diagnostic_counter_for_tests(H_LIVENESS_RECONCILE_BY_NS)?;
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
    let mut coordinator = Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let mut waiters = Vec::with_capacity(commits);
    for index in 0..commits {
        let claim = ClaimSet::new(std::iter::empty(), [index + 1], FlockMode::Exclusive);
        waiters.push((Ticket::register(claim.clone(), claim.clone(), None)?, claim));
    }
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.repair_consistency_if_needed()?;
        let claim_epoch = table.claim_epoch();
        set_cpu_free_for_tests(&mut table, 0, false)?;
        for (ticket, claim) in &waiters {
            table.set_record_state(ticket.slot, STATE_GRANTED)?;
            table.set_record_grant_epoch(ticket.slot, claim_epoch)?;
            for &cpu in &claim.cpus {
                set_cpu_free_for_tests(&mut table, cpu, true)?;
            }
        }
        write_u64(&mut table.header, H_PENDING_FLAGS, 0);
        table.finish_claim_scan();
    }

    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let empty = BTreeSet::new();
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
            |designated, acquisition_allowed, _predecessors| {
            if !acquisition_allowed || designated != claim {
                anyhow::bail!("exact-commit waiter lost its prepared grant");
            }
            Ok(GrantAttempt {
                acquired: Some(()),
                next_claim: designated.clone(),
                contention: None,
            })
            },
        )?;
        if !matches!(result, GrantResult::Acquired(())) {
            anyhow::bail!("exact-commit waiter did not commit its prepared acquisition");
        }
        coordinator.schedule(
            None,
            &empty,
            &empty,
            false,
            None,
            &[],
            None,
            &[],
            false,
            None,
            false,
            None,
        )?;
    }
    let scans = diagnostic_counter_for_tests(H_GRANT_SCANS)? - scans_before;
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
    let mut later = Ticket::register(later_claim.clone(), later_claim, None)?;
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
    first.finish_acquired(&exact, &ClaimSet::default(), &[], None)?;
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let empty = BTreeSet::new();
    middle.schedule(
        None,
        &empty,
        &empty,
        false,
        None,
        &[],
        None,
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
    let mut later = Ticket::register(later_claim.clone(), later_claim, None)?;
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
    first.finish_acquired(&exact, &ClaimSet::default(), &[], None)?;
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let empty = BTreeSet::new();
    middle.schedule(
        None,
        &empty,
        &empty,
        false,
        None,
        &[],
        None,
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
            .is_some_and(|record| record.state == STATE_WAITING)
    };
    later.finish(None)?;
    middle.finish(None)?;
    Ok((scans, later_revoked))
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
    first.finish_acquired(&exact, &ClaimSet::default(), &[], None)?;
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let empty = BTreeSet::new();
    middle.schedule(
        None,
        &empty,
        &empty,
        false,
        None,
        &[],
        None,
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
    let middle_claim =
        ClaimSet::new(std::iter::empty(), [3usize], FlockMode::Exclusive);
    let mut middle = Ticket::register(middle_claim.clone(), middle_claim, None)?;
    let later_claim = shared.clone();
    let mut later = Ticket::register(later_claim.clone(), later_claim, None)?;
    {
        let _lock = lock_registry_existing(FlockMode::Exclusive)?;
        let mut table = Table::open_existing()?;
        table.set_record_state(middle.slot, STATE_WAITING)?;
        table.set_record_state(later.slot, STATE_WAITING)?;
        set_cpu_availability_for_tests(
            &mut table,
            1,
            CpuAvailability::ExclusiveHeld,
        )?;
        set_cpu_free_for_tests(&mut table, 3, false)?;
        write_u64(&mut table.header, H_PENDING_FLAGS, 0);
        table.finish_claim_scan();
    }
    first.finish_acquired(&shared, &ClaimSet::default(), &[], None)?;
    let scans_before = diagnostic_counter_for_tests(H_GRANT_SCANS)?;
    let empty = BTreeSet::new();
    middle.schedule(
        None,
        &empty,
        &empty,
        false,
        None,
        &[],
        None,
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
    let coordinator_claim =
        ClaimSet::new(std::iter::empty(), [0usize], FlockMode::Exclusive);
    let mut coordinator =
        Ticket::register(coordinator_claim.clone(), coordinator_claim, None)?;
    let flexible_claim = ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let flexible_watch =
        ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut flexible =
        Ticket::register(flexible_claim.clone(), flexible_watch.clone(), None)?;
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
        && flexible_record.state == STATE_REPLAN;
    let fixed_preserved = fixed_record.claim == fixed_claim
        && fixed_record.watch == fixed_claim
        && fixed_record.state == STATE_WAITING;
    let flexible_still_flexible =
        claim_is_flexible(&flexible_record.claim, &flexible_record.watch);
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
        if let Some(lock) =
            try_lock_registry_existing_nonblocking(FlockMode::Exclusive)?
        {
            break lock;
        }
        if std::time::Instant::now() >= deadline {
            anyhow::bail!(
                "timed out taking admission registry lock to defer liveness maintenance"
            );
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

fn validate_claim(claim: &ClaimSet) -> Result<()> {
    if claim.is_empty() {
        anyhow::bail!("queue claims must designate a non-empty exact reservation");
    }
    Ok(())
}

fn validate_claim_within_watch(claim: &ClaimSet, watch: &ClaimSet) -> Result<()> {
    if !claim.cpus.is_subset(&watch.cpus)
        || !claim.llcs.is_subset(&watch.llcs)
        || (!claim.cpus.is_empty()
            && !watch.cpus.is_empty()
            && watch.cpu_mode != ClaimMode::Exclusive
            && claim.cpu_mode != watch.cpu_mode)
        || (!claim.llcs.is_empty()
            && !watch.llcs.is_empty()
            && watch.llc_mode != ClaimMode::Exclusive
            && claim.llc_mode != watch.llc_mode)
    {
        anyhow::bail!(
            "queue claim is outside its immutable watch set: claim={claim:?}, watch={watch:?}"
        );
    }
    Ok(())
}

fn validate_contention_within_watch(
    contention: &[ContentionMarker],
    watch: &ClaimSet,
) -> Result<()> {
    for marker in contention {
        let valid = match marker.blocker {
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
        };
        if !valid {
            anyhow::bail!(
                "queue contention marker is outside its immutable watch set: \
                 marker={marker:?}, watch={watch:?}"
            );
        }
    }
    Ok(())
}

fn union_claims(a: &ClaimSet, b: &ClaimSet) -> ClaimSet {
    a.union_envelope(b)
}

fn claim_is_flexible(claim: &ClaimSet, watch: &ClaimSet) -> bool {
    claim.cpus != watch.cpus
        || claim.llcs != watch.llcs
        || claim.cpu_mode != watch.cpu_mode
        || claim.llc_mode != watch.llc_mode
}

fn materialize_claim_paths(claim: &ClaimSet) -> Result<()> {
    for &llc in &claim.llcs {
        materialize_if_missing(super::super::llc_lock_path(llc))?;
    }
    for &cpu in &claim.cpus {
        materialize_if_missing(super::super::cpu_lock_path(cpu))?;
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
    let claim_bits = claim
        .llcs
        .iter()
        .chain(&claim.cpus)
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    // The v5 mapping is deliberately overprovisioned once. It never needs a
    // migration/grow protocol when the first low-index ticket is followed by a
    // valid sparse CPU/LLC index on the same host.
    claim_bits.max(host_cpu_resource_bits()).max(4096)
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

fn registry_lock_path() -> PathBuf {
    registry_data_dir().join("registry.lock")
}

fn header_path() -> PathBuf {
    registry_data_dir().join("registry.map")
}

fn notify_path() -> PathBuf {
    event_dir().join("notify")
}

fn registry_data_dir() -> PathBuf {
    protocol_dir().join("ktstr-acquire-registry-v5")
}

pub(super) fn event_dir() -> PathBuf {
    protocol_dir().join("ktstr-acquire-events-v5")
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

fn lock_registry_for_initialization() -> Result<OwnedFd> {
    std::fs::create_dir_all(registry_data_dir())?;
    std::fs::create_dir_all(event_dir())?;
    block_flock(registry_lock_path(), FlockMode::Exclusive)
}

fn lock_registry_existing(mode: FlockMode) -> Result<OwnedFd> {
    try_lock_registry_existing(mode)?.ok_or_else(|| {
        anyhow::anyhow!(
            "existing admission registry lock {} disappeared",
            registry_lock_path().display()
        )
    })
}

fn try_lock_registry_existing(mode: FlockMode) -> Result<Option<OwnedFd>> {
    use rustix::fs::{FlockOperation, flock};

    let path = registry_lock_path();
    let file = match OpenOptions::new().read(true).write(true).open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| {
                format!("open existing admission registry lock {}", path.display())
            });
        }
    };
    let operation = match mode {
        FlockMode::Exclusive => FlockOperation::LockExclusive,
        FlockMode::Shared => FlockOperation::LockShared,
    };
    flock(&file, operation)
        .map_err(|errno| std::io::Error::from_raw_os_error(errno.raw_os_error()))
        .with_context(|| format!("lock existing admission registry {}", path.display()))?;
    #[cfg(test)]
    if mode == FlockMode::Exclusive {
        REGISTRY_EX_ACQUISITIONS.with(|count| count.set(count.get().saturating_add(1)));
    }
    Ok(Some(file.into()))
}

fn try_lock_registry_existing_nonblocking(mode: FlockMode) -> Result<Option<OwnedFd>> {
    use rustix::fs::{FlockOperation, flock};

    let path = registry_lock_path();
    let file = match OpenOptions::new().read(true).write(true).open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| {
                format!("open existing admission registry lock {}", path.display())
            });
        }
    };
    let operation = match mode {
        FlockMode::Exclusive => FlockOperation::NonBlockingLockExclusive,
        FlockMode::Shared => FlockOperation::NonBlockingLockShared,
    };
    match flock(&file, operation) {
        Ok(()) => {
            #[cfg(test)]
            if mode == FlockMode::Exclusive {
                REGISTRY_EX_ACQUISITIONS.with(|count| count.set(count.get().saturating_add(1)));
            }
            Ok(Some(file.into()))
        }
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => Ok(None),
        Err(errno) => Err(std::io::Error::from_raw_os_error(errno.raw_os_error()))
            .with_context(|| format!("lock existing admission registry {}", path.display())),
    }
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

fn lock_registry_interruptible(cancelled: Option<&AtomicBool>) -> Result<OwnedFd> {
    check_cancelled(cancelled)?;
    let lock = match try_lock_registry_existing(FlockMode::Exclusive)? {
        Some(lock) => lock,
        None => lock_registry_for_initialization()?,
    };
    normalize_cancellation(Ok(lock), cancelled)
}

fn lock_registry_interruptible_existing(cancelled: Option<&AtomicBool>) -> Result<OwnedFd> {
    check_cancelled(cancelled)?;
    normalize_cancellation(lock_registry_existing(FlockMode::Exclusive), cancelled)
}

fn notify_coordinator() {
    let _ = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(notify_path());
}

pub(super) fn publish_acquired(claim: &ClaimSet) -> Result<()> {
    validate_claim(claim)?;
    let Some(shared) = try_lock_registry_existing(FlockMode::Shared)? else {
        return Ok(());
    };
    let path = header_path();
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error).with_context(|| format!("open {}", path.display())),
    };
    if file.metadata()?.len() == 0 {
        return Ok(());
    }
    let map = unsafe { Mmap::map(&file) }
        .with_context(|| format!("map queue aggregate {}", path.display()))?;
    let layout = HeaderLayout::validate(&map)?;
    let dirty = atomic_u64(&map, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0;
    let watched = claim_intersects_watch_map(&map, layout, claim)?;
    if !dirty && !watched {
        return Ok(());
    }
    drop(map);
    drop(file);
    drop(shared);

    let Some(_lock) = try_lock_registry_existing(FlockMode::Exclusive)? else {
        return Ok(());
    };
    let mut table = Table::open_existing()?;
    table.repair_consistency_if_needed()?;
    let watched = table.watched_intersection(claim)?;
    if watched.is_empty() {
        return Ok(());
    }
    table.begin_transaction()?;
    table.publish_claim_busy(&watched)?;
    table.finish_transaction();
    drop(table);
    drop(_lock);
    notify_coordinator();
    Ok(())
}

#[derive(Clone, Copy)]
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
                    .checked_mul(4)
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
        Self::open(1)
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

    fn bump_generation(&mut self) -> Result<()> {
        let next = self
            .generation()
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("queue registry generation exhausted"))?;
        write_u64(&mut self.header, H_GENERATION, next);
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

    fn set_pending_flag(&mut self, flag: u64) {
        let flags = self.pending_flags() | flag;
        write_u64(&mut self.header, H_PENDING_FLAGS, flags);
    }

    fn clear_pending_flag(&mut self, flag: u64) {
        let flags = self.pending_flags() & !flag;
        write_u64(&mut self.header, H_PENDING_FLAGS, flags);
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

    fn mark_claim_changed(&mut self, ticket: u64) -> Result<()> {
        self.advance_claim_epoch()?;
        let minimum = self.min_changed_ticket().min(ticket);
        write_u64(&mut self.header, H_MIN_CHANGED_TICKET, minimum);
        self.set_pending_flag(PENDING_RESCAN);
        Ok(())
    }

    fn finish_claim_scan(&mut self) {
        write_u64(&mut self.header, H_MIN_CHANGED_TICKET, u64::MAX);
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
        let requested = now.saturating_add(delay_ns).max(1);
        let current = read_u64(&self.header, H_LIVENESS_RECONCILE_BY_NS);
        if current == 0 || requested < current {
            write_u64(
                &mut self.header,
                H_LIVENESS_RECONCILE_BY_NS,
                requested,
            );
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

    fn set_coordinator(&mut self, ticket: u64, slot: u64) -> Result<()> {
        if (ticket == 0) != (slot == NONE_SLOT) {
            anyhow::bail!(
                "queue registry v{VERSION} coordinator identity is incomplete: ticket={ticket}, slot={slot}"
            );
        }
        write_u64(&mut self.header, H_COORDINATOR, ticket);
        write_u64(&mut self.header, H_COORDINATOR_SLOT, slot);
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

    fn initialize_record(
        &mut self,
        slot: u64,
        ticket: u64,
        pid: u32,
        claim: &ClaimSet,
        watch: &ClaimSet,
        publish_state: u32,
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
        write_u64(bytes, R_BLOCKED_SERIAL, 0);
        write_u64(bytes, R_NEXT_FREE, NONE_SLOT);
        write_u64(bytes, R_GRANT_EPOCH, 0);
        write_u32(bytes, R_BLOCK_KIND, BLOCK_NONE);
        write_u32(bytes, R_BLOCK_MODE, 0);
        write_u64(bytes, R_BLOCK_INDEX, 0);
        write_u64(bytes, R_REPLAN_SERIAL, 0);
        write_u64(bytes, R_REPLAN_CLAIM_EPOCH, 0);
        write_u64(bytes, R_PREV_ACTIVE, NONE_SLOT);
        write_u64(bytes, R_NEXT_ACTIVE, NONE_SLOT);
        encode_claim(bytes, layout, claim, watch)?;
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

    fn record(&mut self, slot: u64) -> Result<Option<Record>> {
        let layout = self.layout;
        let Some(bytes) = self.record_bytes(slot)? else {
            return Ok(None);
        };
        let state = read_u32(bytes, R_STATE);
        if state == STATE_FREE {
            return Ok(None);
        }
        if !matches!(
            state,
            STATE_WAITING | STATE_GRANTED | STATE_REPLAN | STATE_COORDINATOR
        ) {
            anyhow::bail!("queue registry v{VERSION} slot {slot} has invalid state {state}");
        }
        let record = decode_record(bytes, layout, slot)?;
        if record.ticket == 0 || record.claim.is_empty() {
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

    fn records(&mut self) -> Result<Vec<Record>> {
        let mut records = Vec::new();
        let mut tickets = BTreeSet::new();
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
            if !tickets.insert(record.ticket) {
                anyhow::bail!(
                    "queue registry v{VERSION} contains duplicate active ticket {}",
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

    /// Aggregate every exact reservation preceding `ticket` in arrival order.
    ///
    /// A flexible waiter replans outside the registry lock. Giving that
    /// callback the immutable prefix lets it avoid an earlier reservation
    /// which has not acquired its real flock yet; `/proc/locks` alone cannot
    /// reveal such a claim. The returned shape is the same compact bitset
    /// snapshot used by the global fast-path fence.
    fn claims_before(&mut self, ticket: u64) -> Result<AggregateSnapshot> {
        let mut cpu_any = vec![0u64; self.layout.words];
        let mut cpu_exclusive = vec![0u64; self.layout.words];
        let mut llc_any = vec![0u64; self.layout.words];
        let mut llc_exclusive = vec![0u64; self.layout.words];
        let next_slot = self.next_slot()?;
        let mut slot = read_u64(&self.header, H_ACTIVE_HEAD);
        let mut previous = NONE_SLOT;
        let mut previous_ticket = 0;
        let mut visited = 0usize;
        let mut found = false;
        while slot != NONE_SLOT {
            if visited >= usize::try_from(next_slot).unwrap_or(usize::MAX) {
                anyhow::bail!("queue registry v{VERSION} active list contains a cycle");
            }
            #[cfg(test)]
            CLAIMS_BEFORE_RECORD_READS.with(|reads| reads.set(reads.get() + 1));
            let bytes = self.record_bytes(slot)?.ok_or_else(|| {
                anyhow::anyhow!("queue registry v{VERSION} active slot {slot} is free")
            })?;
            let state = read_u32(bytes, R_STATE);
            if !matches!(
                state,
                STATE_WAITING | STATE_GRANTED | STATE_REPLAN | STATE_COORDINATOR
            ) {
                anyhow::bail!(
                    "queue registry v{VERSION} active slot {slot} has invalid state {state}"
                );
            }
            let current_ticket = read_u64(bytes, R_TICKET);
            let prev_active = read_u64(bytes, R_PREV_ACTIVE);
            let next_active = read_u64(bytes, R_NEXT_ACTIVE);
            if prev_active != previous {
                anyhow::bail!(
                    "queue registry v{VERSION} active slot {slot} has prev={prev_active}, expected {previous}"
                );
            }
            if current_ticket <= previous_ticket {
                anyhow::bail!(
                    "queue registry v{VERSION} active tickets are not strictly increasing at {current_ticket}"
                );
            }
            if current_ticket == ticket {
                found = true;
                break;
            }
            if current_ticket > ticket {
                anyhow::bail!(
                    "queue ticket {ticket} is missing before active ticket {current_ticket}"
                );
            }
            let llc_mode = read_u32(bytes, R_CLAIM_LLC_MODE);
            if llc_mode > 1 {
                anyhow::bail!(
                    "queue registry v{VERSION} slot {slot} has invalid LLC claim mode {llc_mode}"
                );
            }
            let cpu_mode = read_u32(bytes, R_CLAIM_CPU_MODE);
            if cpu_mode > 1 {
                anyhow::bail!(
                    "queue registry v{VERSION} slot {slot} has invalid CPU claim mode {cpu_mode}"
                );
            }
            for word in 0..self.layout.words {
                let cpus = read_u64(bytes, R_BITS + word * 8);
                cpu_any[word] |= cpus;
                if cpu_mode == 1 {
                    cpu_exclusive[word] |= cpus;
                }
                let llcs = read_u64(bytes, R_BITS + self.layout.words * 8 + word * 8);
                llc_any[word] |= llcs;
                if llc_mode == 1 {
                    llc_exclusive[word] |= llcs;
                }
            }
            previous = slot;
            previous_ticket = current_ticket;
            slot = next_active;
            visited += 1;
        }
        if !found {
            anyhow::bail!("queue ticket {ticket} is absent from the active list");
        }
        Ok(AggregateSnapshot {
            bits: self.layout.bits,
            cpu_any,
            cpu_exclusive,
            llc_any,
            llc_exclusive,
        })
    }

    fn replace_claim(
        &mut self,
        slot: u64,
        ticket: u64,
        old: &ClaimSet,
        new: &ClaimSet,
        publish_state: u32,
        replan_serial: u64,
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
            replan_serial,
            blocked,
            persist_blocker,
        )?;
        self.finish_transaction();
        Ok(())
    }

    fn replace_claim_in_transaction(
        &mut self,
        slot: u64,
        ticket: u64,
        old: &ClaimSet,
        new: &ClaimSet,
        publish_state: u32,
        replan_serial: u64,
        blocked: Option<(ContentionMarker, u64)>,
        persist_blocker: bool,
    ) -> Result<()> {
        let replan_claim_epoch = self
            .record(slot)?
            .filter(|record| record.ticket == ticket)
            .ok_or_else(|| anyhow::anyhow!("queue slot {slot} disappeared during replacement"))?
            .replan_claim_epoch;
        if let Some((evidence, _)) = blocked {
            self.mark_blocker_unknown(evidence)?;
        }
        self.adjust_claim_counts(old, false)?;
        self.adjust_claim_counts(new, true)?;
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
            encode_exact_claim(bytes, layout, new)?;
            write_u64(bytes, R_BLOCKED_SERIAL, 0);
            write_u32(bytes, R_BLOCK_KIND, BLOCK_NONE);
            write_u32(bytes, R_BLOCK_MODE, 0);
            write_u64(bytes, R_BLOCK_INDEX, 0);
            write_u64(bytes, R_REPLAN_SERIAL, replan_serial);
            write_u64(bytes, R_REPLAN_CLAIM_EPOCH, replan_claim_epoch);
            if persist_blocker && let Some((evidence, serial)) = blocked {
                let (kind, index) = match evidence.blocker {
                    ResourceKey::Cpu(index) => (BLOCK_CPU, index),
                    ResourceKey::Llc(index) => (BLOCK_LLC, index),
                };
                write_u64(bytes, R_BLOCKED_SERIAL, serial);
                write_u32(bytes, R_BLOCK_KIND, kind);
                write_u32(
                    bytes,
                    R_BLOCK_MODE,
                    u32::from(evidence.mode == FlockMode::Exclusive),
                );
                write_u64(
                    bytes,
                    R_BLOCK_INDEX,
                    u64::try_from(index)
                        .context("blocked resource index does not fit registry record")?,
                );
            }
        }
        self.mark_claim_changed(ticket)?;
        crash_at_for_tests("replace_record_before_state_publish");
        let bytes = self
            .record_bytes_mut(slot)?
            .ok_or_else(|| anyhow::anyhow!("queue slot {slot} disappeared"))?;
        write_u32(bytes, R_STATE, publish_state);
        // Existing-ticket claim changes invalidate only grants at or after
        // this ticket. The coordinator's next O(N) pass revalidates that
        // suffix; earlier granted waiters refresh their epoch in O(1).
        Ok(())
    }

    fn remove_record(&mut self, record: &Record, acquired: bool) -> Result<()> {
        self.begin_transaction()?;
        if acquired {
            self.publish_claim_busy(&record.claim)?;
        }
        self.remove_record_in_transaction(record, acquired)?;
        if !acquired {
            self.mark_claim_changed(record.ticket)?;
        }
        crash_at_for_tests("remove_record_before_election");
        self.elect_coordinator_in_transaction()?;
        if !acquired {
            self.set_pending_flag(PENDING_RESCAN);
        }
        self.finish_transaction();
        Ok(())
    }

    fn remove_record_in_transaction(&mut self, record: &Record, acquired: bool) -> Result<()> {
        if !acquired {
            if self.coordinator_ticket() == record.ticket {
                // The coordinator may have acquired partials for a freshly
                // planned alternative but crashed before publishing that exact
                // claim. Its immutable watch is the only sound crash envelope.
                self.mark_possible_release(&record.watch.cpus, &record.watch.llcs)?;
            } else if record.state == STATE_GRANTED {
                // A granted callback probes without EX and may die while
                // holding its exact fds, before committing acquired removal.
                self.mark_possible_release(&record.claim.cpus, &record.claim.llcs)?;
            }
        }
        self.adjust_claim_counts(&record.claim, false)?;
        self.adjust_watch_counts(&record.watch, false)?;
        crash_at_for_tests("remove_counts_before_free");
        if self.coordinator_ticket() == record.ticket {
            self.set_coordinator(0, NONE_SLOT)?;
        }
        self.unlink_active(record)?;
        let free_head = read_u64(&self.header, H_FREE_HEAD);
        let bytes = self
            .record_bytes_mut(record.slot)?
            .ok_or_else(|| anyhow::anyhow!("queue slot {} disappeared", record.slot))?;
        write_u32(bytes, R_STATE, STATE_FREE);
        write_u64(bytes, R_TICKET, 0);
        write_u64(bytes, R_NEXT_FREE, free_head);
        write_u64(&mut self.header, H_FREE_HEAD, record.slot);
        Ok(())
    }

    fn set_record_state(&mut self, slot: u64, state: u32) -> Result<()> {
        let bytes = self
            .record_bytes_mut(slot)?
            .ok_or_else(|| anyhow::anyhow!("queue slot {slot} disappeared during state update"))?;
        write_u32(bytes, R_STATE, state);
        Ok(())
    }

    fn clear_record_blocked(&mut self, slot: u64) -> Result<()> {
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
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during blocker update")
        })?;
        write_u64(bytes, R_BLOCKED_SERIAL, serial);
        let (kind, index) = match evidence.blocker {
            ResourceKey::Cpu(index) => (BLOCK_CPU, index),
            ResourceKey::Llc(index) => (BLOCK_LLC, index),
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
        Ok(())
    }

    fn set_record_replan_serial(&mut self, slot: u64, serial: u64) -> Result<()> {
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during replan-serial update")
        })?;
        write_u64(bytes, R_REPLAN_SERIAL, serial);
        Ok(())
    }

    fn set_record_replan_claim_epoch(&mut self, slot: u64, epoch: u64) -> Result<()> {
        let bytes = self.record_bytes_mut(slot)?.ok_or_else(|| {
            anyhow::anyhow!("queue slot {slot} disappeared during replan-epoch update")
        })?;
        write_u64(bytes, R_REPLAN_CLAIM_EPOCH, epoch);
        Ok(())
    }

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

    fn map_futex(&mut self, slot: u64) -> Result<FutexSlot> {
        let next_slot = self.next_slot()?;
        if slot >= next_slot {
            anyhow::bail!("queue futex slot {slot} is outside 0..{next_slot}");
        }
        let (chunk, range) = record_range(slot, self.layout.record_size)?;
        let file = open_chunk_file(chunk, self.layout.record_size)?;
        let mut map = unsafe { MmapMut::map_mut(&file) }?;
        if range.end > map.len() {
            anyhow::bail!(
                "queue registry chunk {chunk} is too short for futex slot {slot}: {} bytes < {}",
                map.len(),
                range.end
            );
        }
        let offset = range
            .start
            .checked_add(R_WAKE)
            .ok_or_else(|| anyhow::anyhow!("queue futex offset overflow"))?;
        let ptr = unsafe { map.as_mut_ptr().add(offset).cast::<AtomicU32>() };
        Ok(FutexSlot { _map: map, ptr })
    }

    fn grant_compatible(&mut self) -> Result<ClaimSet> {
        let records = self.records()?;
        let mut cpu_any = vec![0u64; self.layout.words];
        let mut cpu_exclusive = vec![0u64; self.layout.words];
        let mut llc_any = vec![0u64; self.layout.words];
        let mut llc_exclusive = vec![0u64; self.layout.words];
        let claim_epoch = self.claim_epoch();
        let mut changed = false;
        self.bump_grant_scans();

        for record in records {
            let conflict = claim_conflicts_bits(
                &record.claim,
                &cpu_any,
                &cpu_exclusive,
                &llc_any,
                &llc_exclusive,
                self.layout.bits,
            )?;
            let flexible = claim_is_flexible(&record.claim, &record.watch);
            let availability_compatible = self.claim_availability_compatible(&record.claim)?;
            let blocker_ready = match record.blocked_on {
                None => true,
                Some(blocked) => {
                    let still_designated = match blocked.key {
                        ResourceKey::Cpu(index) => record.claim.cpus.contains(&index),
                        ResourceKey::Llc(index) => record.claim.llcs.contains(&index),
                    };
                    !still_designated
                        || self.blocker_serial(blocked.key, blocked.mode)? > blocked.serial
                }
            };
            let replan_invalidated = record.state == STATE_REPLAN
                && self.min_changed_ticket() < record.ticket
                && record.replan_claim_epoch != claim_epoch;
            if replan_invalidated {
                // The callback runs outside the registry fence. Stamp its
                // record before clearing the global suffix invalidation so its
                // post-callback validation cannot commit an obsolete plan.
                self.set_record_replan_claim_epoch(record.slot, claim_epoch)?;
                self.wake_slot(record.slot)?;
                changed = true;
            } else if record.state == STATE_GRANTED && (conflict || !availability_compatible) {
                // An earlier ticket may have changed its exact alternative
                // or committed a newly-busy resource after this grant was
                // issued. Revoke the stale grant before it can race into the
                // real flock probe.
                self.set_record_state(record.slot, STATE_WAITING)?;
                self.clear_record_blocked(record.slot)?;
                changed = true;
            } else if record.state == STATE_WAITING
                && record.ticket != self.coordinator_ticket()
                && !conflict
                && availability_compatible
                && blocker_ready
            {
                self.set_record_state(record.slot, STATE_GRANTED)?;
                self.clear_record_blocked(record.slot)?;
                self.set_record_grant_epoch(record.slot, claim_epoch)?;
                crash_at_for_tests("grant_state_before_wake");
                self.wake_slot(record.slot)?;
                changed = true;
            } else if record.state == STATE_WAITING
                && record.ticket != self.coordinator_ticket()
                && flexible
            {
                let watch_serial = self.max_watch_serial(&record.watch)?;
                let earlier_claim_changed = record.replan_claim_epoch != claim_epoch
                    && self.min_changed_ticket() < record.ticket;
                if watch_serial > record.replan_serial || earlier_claim_changed {
                    self.set_record_state(record.slot, STATE_REPLAN)?;
                    self.clear_record_blocked(record.slot)?;
                    if earlier_claim_changed {
                        self.set_record_replan_claim_epoch(record.slot, claim_epoch)?;
                    }
                    self.wake_slot(record.slot)?;
                    changed = true;
                }
            } else if record.state == STATE_GRANTED && record.grant_epoch != claim_epoch {
                // This grant precedes every changed ticket and remains valid.
                self.set_record_grant_epoch(record.slot, claim_epoch)?;
            }
            add_claim_bits(
                &record.claim,
                &mut cpu_any,
                &mut cpu_exclusive,
                &mut llc_any,
                &mut llc_exclusive,
                self.layout.bits,
            )?;
        }
        if changed {
            self.bump_generation()?;
        }
        self.finish_claim_scan();
        self.clear_pending_flag(PENDING_RESCAN);
        self.aggregate_watch()
    }

    fn prune_dead(&mut self) -> Result<()> {
        let records = self.records()?;
        let mut dead = Vec::new();
        for record in records {
            if !ticket_is_live(record.slot, record.ticket)? {
                dead.push(record);
            }
        }
        if dead.is_empty() {
            return Ok(());
        }

        // Full maintenance is the overflow recovery path and may discover a large dead
        // prefix after job cancellation. Free the whole batch in one dirty
        // transaction, then scan/elect once instead of repeatedly electing the
        // next corpse and degenerating to O(N²).
        self.begin_transaction()?;
        for record in &dead {
            // Each unlink mutates its neighbours' prev/next fields. Re-read
            // the next victim so a cloned stale link cannot write through an
            // already-freed predecessor and corrupt the active/free lists.
            if let Some(current) = self
                .record(record.slot)?
                .filter(|current| current.ticket == record.ticket)
            {
                self.remove_record_in_transaction(&current, false)?;
            }
        }
        if let Some(first) = dead.iter().map(|record| record.ticket).min() {
            self.mark_claim_changed(first)?;
        }
        crash_at_for_tests("remove_record_before_election");
        self.elect_coordinator_in_transaction()?;
        self.bump_generation()?;
        self.finish_transaction();
        for record in dead {
            let _ = std::fs::remove_file(liveness_path(record.slot, record.ticket));
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
            self.mark_claim_changed(first)?;
        }
        self.elect_coordinator_in_transaction()?;
        self.bump_generation()?;
        self.finish_transaction();
        for record in dead {
            let _ = std::fs::remove_file(liveness_path(record.slot, record.ticket));
        }
        Ok(())
    }

    fn recover_coordinator_if_dead(&mut self) -> Result<()> {
        let coordinator = self.coordinator_ticket();
        let slot = self.coordinator_slot()?;
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
            return self.elect_coordinator();
        }
        if coordinator != 0 {
            let _record = self
                .record(slot)?
                .filter(|record| record.ticket == coordinator && record.state == STATE_COORDINATOR)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "queue registry v{VERSION} coordinator header ticket={coordinator}, \
                         slot={slot} does not name its coordinator record"
                    )
                })?;
            if ticket_is_live(slot, coordinator)? {
                return Ok(());
            }
            // Coordinator death commonly accompanies a cancelled job with a
            // dead prefix. This is a rare recovery scan: batch every dead
            // record and elect one live survivor once, rather than peeling and
            // re-electing dead coordinators one O(N) scan at a time.
            self.prune_dead()?;
        }
        self.elect_coordinator()
    }

    fn elect_coordinator(&mut self) -> Result<()> {
        if self.coordinator_ticket() != 0 {
            return Ok(());
        }
        self.begin_transaction()?;
        self.elect_coordinator_in_transaction()?;
        self.finish_transaction();
        Ok(())
    }

    fn elect_coordinator_in_transaction(&mut self) -> Result<()> {
        if self.coordinator_ticket() != 0 {
            return Ok(());
        }
        let head = read_u64(&self.header, H_ACTIVE_HEAD);
        if head != NONE_SLOT {
            let record = self.record(head)?.ok_or_else(|| {
                anyhow::anyhow!("queue active-list head {head} does not name an active record")
            })?;
            if record.state != STATE_WAITING {
                return Ok(());
            }
            self.set_coordinator(record.ticket, record.slot)?;
            crash_at_for_tests("elect_header_before_state");
            self.set_record_state(record.slot, STATE_COORDINATOR)?;
            self.clear_record_blocked(record.slot)?;
            self.wake_slot(record.slot)?;
            self.bump_generation()?;
        }
        Ok(())
    }

    fn begin_transaction(&mut self) -> Result<()> {
        self.repair_consistency_if_needed()?;
        atomic_u64(&self.header, H_AGGREGATE_DIRTY).store(1, Ordering::SeqCst);
        Ok(())
    }

    fn finish_transaction(&mut self) {
        atomic_u64(&self.header, H_AGGREGATE_DIRTY).store(0, Ordering::SeqCst);
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

    fn claim_availability_compatible(&self, claim: &ClaimSet) -> Result<bool> {
        for &cpu in &claim.cpus {
            if !self.bitmap_bit(B_CPU_KNOWN, cpu)? {
                return Ok(false);
            }
            let available = match claim.cpu_mode {
                ClaimMode::Shared => self.bitmap_bit(B_CPU_SH_AVAILABLE, cpu)?,
                ClaimMode::Exclusive => self.bitmap_bit(B_CPU_EX_AVAILABLE, cpu)?,
            };
            if !available {
                return Ok(false);
            }
        }
        for &llc in &claim.llcs {
            if !self.bitmap_bit(B_LLC_KNOWN, llc)? {
                return Ok(false);
            }
            let available = match claim.llc_mode {
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

    fn blocker_serial(&self, key: ResourceKey, mode: FlockMode) -> Result<u64> {
        match (key, mode) {
            (ResourceKey::Cpu(index), FlockMode::Shared) => {
                self.resource_serial(S_CPU_SH, index)
            }
            (ResourceKey::Cpu(index), FlockMode::Exclusive) => {
                self.resource_serial(S_CPU_EX, index)
            }
            (ResourceKey::Llc(index), FlockMode::Shared) => self.resource_serial(S_LLC_SH, index),
            (ResourceKey::Llc(index), FlockMode::Exclusive) => {
                self.resource_serial(S_LLC_EX, index)
            }
        }
    }

    fn max_watch_serial(&self, watch: &ClaimSet) -> Result<u64> {
        let mut serial = 0;
        for &cpu in &watch.cpus {
            serial = serial.max(self.resource_serial(S_CPU_SH, cpu)?);
            if watch.cpu_mode == ClaimMode::Exclusive {
                serial = serial.max(self.resource_serial(S_CPU_EX, cpu)?);
            }
        }
        for &llc in &watch.llcs {
            serial = serial.max(self.resource_serial(S_LLC_SH, llc)?);
            if watch.llc_mode == ClaimMode::Exclusive {
                serial = serial.max(self.resource_serial(S_LLC_EX, llc)?);
            }
        }
        Ok(serial)
    }

    fn mark_unknown(
        &mut self,
        cpus: &BTreeSet<usize>,
        llcs: &BTreeSet<usize>,
    ) -> Result<bool> {
        let plan = self.watched_observation_plan(cpus, llcs)?;
        self.mark_observation_modes(&plan)
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
        Ok(true)
    }

    fn watched_observation_plan(
        &self,
        cpus: &BTreeSet<usize>,
        llcs: &BTreeSet<usize>,
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
        Ok(true)
    }

    fn mark_possible_release(
        &mut self,
        cpus: &BTreeSet<usize>,
        llcs: &BTreeSet<usize>,
    ) -> Result<bool> {
        let plan = self.possible_release_plan(cpus, llcs)?;
        self.apply_possible_release(&plan)
    }

    fn mark_blocker_unknown(&mut self, evidence: ContentionMarker) -> Result<()> {
        let (cpus, llcs) = match evidence.blocker {
            ResourceKey::Cpu(index) => (BTreeSet::from([index]), BTreeSet::new()),
            ResourceKey::Llc(index) => (BTreeSet::new(), BTreeSet::from([index])),
        };
        self.mark_unknown(&cpus, &llcs)?;
        Ok(())
    }

    fn mark_blockers_unknown(&mut self, evidence: &[ContentionMarker]) -> Result<()> {
        if evidence.is_empty() {
            return Ok(());
        }
        let mut cpus = BTreeSet::new();
        let mut llcs = BTreeSet::new();
        for marker in evidence {
            match marker.blocker {
                ResourceKey::Cpu(index) => {
                    cpus.insert(index);
                }
                ResourceKey::Llc(index) => {
                    llcs.insert(index);
                }
            }
        }
        self.mark_unknown(&cpus, &llcs).map(|_| ())
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
            self.set_bitmap_bit(
                B_CPU_SH_AVAILABLE,
                cpu,
                claim.cpu_mode == ClaimMode::Shared,
            )?;
            self.set_bitmap_bit(B_CPU_EX_AVAILABLE, cpu, false)?;
            self.set_bitmap_bit(B_PENDING_CPU_SH, cpu, false)?;
            self.set_bitmap_bit(B_PENDING_CPU_EX, cpu, false)?;
            self.set_bitmap_bit(B_CANDIDATE_CPU_SH, cpu, false)?;
            self.set_bitmap_bit(B_CANDIDATE_CPU_EX, cpu, false)?;
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
            self.set_bitmap_bit(
                B_LLC_SH_AVAILABLE,
                llc,
                claim.llc_mode == ClaimMode::Shared,
            )?;
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
            self.set_pending_flag(PENDING_RESCAN);
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
            cpus.insert(index, (sh, ex));
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
        if cpus.is_empty() && llcs.is_empty() {
            return Ok(None);
        }
        Ok(Some(ObservationRequest { cpus, llcs }))
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
            self.set_pending_flag(PENDING_RESCAN);
        }
        Ok(improved)
    }

    fn adjust_claim_counts(&mut self, claim: &ClaimSet, add: bool) -> Result<()> {
        for &cpu in &claim.cpus {
            self.adjust_aggregate_bit(B_CLAIM_CPUS, cpu, add)?;
            if claim.cpu_mode == ClaimMode::Exclusive {
                self.adjust_aggregate_bit(B_CLAIM_CPU_EXCLUSIVE, cpu, add)?;
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
                let offset =
                    self.layout.count_offset(B_WATCH_CPU_EXCLUSIVE) + cpu * 4;
                if read_u32(&self.header, offset) == 0 {
                    plan.cpu_ex.insert(cpu);
                }
            }
        }
        for &llc in &watch.llcs {
            let offset = self.layout.count_offset(B_WATCH_LLCS) + llc * 4;
            if read_u32(&self.header, offset) == 0 {
                plan.llc_sh.insert(llc);
            }
            if watch.llc_mode == ClaimMode::Exclusive {
                let offset =
                    self.layout.count_offset(B_WATCH_LLC_EXCLUSIVE) + llc * 4;
                if read_u32(&self.header, offset) == 0 {
                    plan.llc_ex.insert(llc);
                }
            }
        }
        Ok(plan)
    }

    fn aggregate_watch(&self) -> Result<ClaimSet> {
        let llcs = self.bitmap_indices(B_WATCH_LLCS);
        let cpus = self.bitmap_indices(B_WATCH_CPUS);
        let llc_mode = if self.bitmap_indices(B_WATCH_LLC_EXCLUSIVE).is_empty() {
            ClaimMode::Shared
        } else {
            ClaimMode::Exclusive
        };
        let cpu_mode = if self.bitmap_indices(B_WATCH_CPU_EXCLUSIVE).is_empty() {
            ClaimMode::Shared
        } else {
            ClaimMode::Exclusive
        };
        Ok(ClaimSet::with_claim_modes(
            llcs,
            cpus,
            llc_mode,
            cpu_mode,
        ))
    }

    fn watched_intersection(&self, claim: &ClaimSet) -> Result<ClaimSet> {
        let mut cpus = BTreeSet::new();
        let mut llcs = BTreeSet::new();
        for &cpu in &claim.cpus {
            if cpu < self.layout.bits && self.bitmap_bit(B_WATCH_CPUS, cpu)? {
                cpus.insert(cpu);
            }
        }
        for &llc in &claim.llcs {
            if llc < self.layout.bits && self.bitmap_bit(B_WATCH_LLCS, llc)? {
                llcs.insert(llc);
            }
        }
        Ok(ClaimSet::with_claim_modes(
            llcs,
            cpus,
            claim.llc_mode,
            claim.cpu_mode,
        ))
    }

    fn watched_subset(
        &self,
        cpus: &BTreeSet<usize>,
        llcs: &BTreeSet<usize>,
    ) -> Result<(BTreeSet<usize>, BTreeSet<usize>)> {
        let mut watched_cpus = BTreeSet::new();
        let mut watched_llcs = BTreeSet::new();
        for &cpu in cpus {
            if self.bitmap_bit(B_WATCH_CPUS, cpu)? {
                watched_cpus.insert(cpu);
            }
        }
        for &llc in llcs {
            if self.bitmap_bit(B_WATCH_LLCS, llc)? {
                watched_llcs.insert(llc);
            }
        }
        Ok((watched_cpus, watched_llcs))
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

    fn repair_consistency_if_needed(&mut self) -> Result<()> {
        if atomic_u64(&self.header, H_AGGREGATE_DIRTY).load(Ordering::SeqCst) != 0 {
            self.repair_consistency()?;
        }
        Ok(())
    }

    fn repair_consistency(&mut self) -> Result<()> {
        atomic_u64(&self.header, H_AGGREGATE_DIRTY).store(1, Ordering::SeqCst);
        let next_slot = self.next_slot()?;
        let mut records = Vec::new();
        let mut tickets = BTreeSet::new();
        let mut free_head = NONE_SLOT;
        let mut max_ticket = 0u64;
        for slot in (0..next_slot).rev() {
            let record = self.record(slot)?;
            let valid = record
                .as_ref()
                .is_some_and(|record| record.ticket != 0 && !record.claim.is_empty());
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
        for record in &records {
            self.adjust_claim_counts(&record.claim, true)?;
            self.adjust_watch_counts(&record.watch, true)?;
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
        self.mark_unknown(&watched_cpus, &watched_llcs)?;
        self.set_pending_flag(PENDING_RESCAN);
        let previous = self.coordinator_ticket();
        let coordinator = records
            .iter()
            .find(|record| record.ticket == previous)
            .or_else(|| records.iter().min_by_key(|record| record.ticket));
        let (coordinator, coordinator_slot) = coordinator
            .map(|record| (record.ticket, record.slot))
            .unwrap_or((0, NONE_SLOT));
        self.set_coordinator(coordinator, coordinator_slot)?;
        let repair_claim_epoch = self.advance_claim_epoch()?;
        for record in &records {
            let state = if record.ticket == coordinator {
                STATE_COORDINATOR
            } else if claim_is_flexible(&record.claim, &record.watch) {
                STATE_REPLAN
            } else {
                STATE_WAITING
            };
            self.set_record_state(record.slot, state)?;
            self.clear_record_blocked(record.slot)?;
            self.set_record_replan_serial(record.slot, 0)?;
            self.set_record_replan_claim_epoch(record.slot, repair_claim_epoch)?;
            if state == STATE_REPLAN {
                self.wake_slot(record.slot)?;
            }
        }
        // Any interrupted mutation invalidates outstanding grants. They were
        // demoted above; a fresh coordinator pass will stamp the new epoch.
        self.finish_claim_scan();
        if let Some(record) = records.iter().find(|record| record.ticket == coordinator) {
            self.wake_slot(record.slot)?;
        }
        atomic_u64(&self.header, H_AGGREGATE_DIRTY).store(0, Ordering::SeqCst);
        Ok(())
    }

    fn ensure_chunk(&mut self, chunk: u64) -> Result<()> {
        if self.chunks.contains_key(&chunk) {
            return Ok(());
        }
        let file = open_chunk_file(chunk, self.layout.record_size)?;
        let map = unsafe { MmapMut::map_mut(&file) }?;
        self.chunks.insert(chunk, map);
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
        write_u64(&mut header, H_MIN_CHANGED_TICKET, u64::MAX);
        write_u64(&mut header, H_COORDINATOR_SLOT, NONE_SLOT);
        write_u64(&mut header, H_OBSERVATION_REQUEST, 1);
        write_u64(&mut header, H_ACTIVE_HEAD, NONE_SLOT);
        write_u64(&mut header, H_ACTIVE_TAIL, NONE_SLOT);
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

/// With no elected coordinator, the active head may legitimately be executing
/// a GRANTED acquisition or a REPLAN callback. That in-flight record is the
/// authoritative progress owner; treating coordinator==0 as failure convoys
/// every timeout and conflicting fast probe through the EX recovery path.
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
    if !matches!(state, STATE_GRANTED | STATE_REPLAN) {
        return Ok(false);
    }
    let ticket = read_u64(bytes, R_TICKET);
    Ok(ticket != 0 && ticket_is_live(slot, ticket)?)
}

fn decode_mode(bytes: &[u8], offset: usize, slot: u64, label: &str) -> Result<ClaimMode> {
    match read_u32(bytes, offset) {
        0 => ClaimMode::Shared,
        1 => ClaimMode::Exclusive,
        mode => {
            anyhow::bail!(
                "queue registry v{VERSION} slot {slot} has invalid {label} mode {mode}"
            )
        }
    }
}

fn decode_record(bytes: &[u8], layout: HeaderLayout, slot: u64) -> Result<Record> {
    let watch_llc_mode = decode_mode(bytes, R_WATCH_LLC_MODE, slot, "LLC watch")?;
    let claim_llc_mode = decode_mode(bytes, R_CLAIM_LLC_MODE, slot, "exact LLC claim")?;
    let watch_cpu_mode = decode_mode(bytes, R_WATCH_CPU_MODE, slot, "CPU watch")?;
    let claim_cpu_mode = decode_mode(bytes, R_CLAIM_CPU_MODE, slot, "exact CPU claim")?;
    let claim = ClaimSet::with_claim_modes(
        decode_bitset(bytes, R_BITS + layout.words * 8, layout)?,
        decode_bitset(bytes, R_BITS, layout)?,
        claim_llc_mode,
        claim_cpu_mode,
    );
    let watch = ClaimSet::with_claim_modes(
        decode_bitset(bytes, R_BITS + layout.words * 24, layout)?,
        decode_bitset(bytes, R_BITS + layout.words * 16, layout)?,
        watch_llc_mode,
        watch_cpu_mode,
    );
    let blocked_on = match read_u32(bytes, R_BLOCK_KIND) {
        BLOCK_NONE => None,
        kind @ (BLOCK_CPU | BLOCK_LLC) => {
            let index = usize::try_from(read_u64(bytes, R_BLOCK_INDEX))
                .context("blocked resource index does not fit this process")?;
            let key = if kind == BLOCK_CPU {
                ResourceKey::Cpu(index)
            } else {
                ResourceKey::Llc(index)
            };
            let mode = match read_u32(bytes, R_BLOCK_MODE) {
                0 => FlockMode::Shared,
                1 => FlockMode::Exclusive,
                mode => anyhow::bail!(
                    "queue registry v{VERSION} slot {slot} has invalid blocker mode {mode}"
                ),
            };
            Some(BlockedOn {
                key,
                mode,
                serial: read_u64(bytes, R_BLOCKED_SERIAL),
            })
        }
        kind => {
            anyhow::bail!("queue registry v{VERSION} slot {slot} has invalid blocker kind {kind}")
        }
    };
    Ok(Record {
        slot,
        state: read_u32(bytes, R_STATE),
        ticket: read_u64(bytes, R_TICKET),
        #[cfg(test)]
        pid: read_u32(bytes, R_PID),
        claim,
        watch,
        blocked_on,
        replan_serial: read_u64(bytes, R_REPLAN_SERIAL),
        replan_claim_epoch: read_u64(bytes, R_REPLAN_CLAIM_EPOCH),
        grant_epoch: read_u64(bytes, R_GRANT_EPOCH),
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
    encode_bitset(bytes, R_BITS + layout.words * 16, layout, &watch.cpus)?;
    encode_bitset(bytes, R_BITS + layout.words * 24, layout, &watch.llcs)
}

fn encode_exact_claim(bytes: &mut [u8], layout: HeaderLayout, claim: &ClaimSet) -> Result<()> {
    encode_bitset(bytes, R_BITS, layout, &claim.cpus)?;
    encode_bitset(bytes, R_BITS + layout.words * 8, layout, &claim.llcs)
}

fn clear_record_claim_bits(bytes: &mut [u8], layout: HeaderLayout) {
    bytes[R_BITS..R_BITS + layout.words * 16].fill(0);
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
    claim: &ClaimSet,
    cpu_any: &[u64],
    cpu_exclusive: &[u64],
    llc_any: &[u64],
    llc_exclusive: &[u64],
    bits: usize,
) -> Result<bool> {
    let cpu_fence = if claim.cpu_mode == ClaimMode::Exclusive {
        cpu_any
    } else {
        cpu_exclusive
    };
    for &cpu in &claim.cpus {
        if cpu >= bits {
            anyhow::bail!("CPU index {cpu} exceeds queue registry capacity");
        }
        if cpu_fence[cpu / 64] & (1u64 << (cpu % 64)) != 0 {
            return Ok(true);
        }
    }
    let llc_fence = if claim.llc_mode == ClaimMode::Exclusive {
        llc_any
    } else {
        llc_exclusive
    };
    for &llc in &claim.llcs {
        if llc >= bits {
            anyhow::bail!("LLC index {llc} exceeds queue registry capacity");
        }
        if llc_fence[llc / 64] & (1u64 << (llc % 64)) != 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn add_claim_bits(
    claim: &ClaimSet,
    cpu_any: &mut [u64],
    cpu_exclusive: &mut [u64],
    llc_any: &mut [u64],
    llc_exclusive: &mut [u64],
    bits: usize,
) -> Result<()> {
    for &cpu in &claim.cpus {
        if cpu >= bits {
            anyhow::bail!("CPU index {cpu} exceeds queue registry capacity");
        }
        cpu_any[cpu / 64] |= 1u64 << (cpu % 64);
        if claim.cpu_mode == ClaimMode::Exclusive {
            cpu_exclusive[cpu / 64] |= 1u64 << (cpu % 64);
        }
    }
    for &llc in &claim.llcs {
        if llc >= bits {
            anyhow::bail!("LLC index {llc} exceeds queue registry capacity");
        }
        llc_any[llc / 64] |= 1u64 << (llc % 64);
        if claim.llc_mode == ClaimMode::Exclusive {
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

fn liveness_due_in_from_last(last: u64, now: u64) -> Duration {
    if last == 0 || last > now {
        return Duration::ZERO;
    }
    Duration::from_nanos(LIVENESS_SWEEP_INTERVAL_NS.saturating_sub(now - last))
}

fn liveness_due_in_from_header(header: &[u8], now: u64) -> Duration {
    let periodic = liveness_due_in_from_last(
        read_u64(header, H_LAST_LIVENESS_SWEEP_NS),
        now,
    );
    let reconcile_by = read_u64(header, H_LIVENESS_RECONCILE_BY_NS);
    if reconcile_by == 0 {
        periodic
    } else {
        periodic.min(Duration::from_nanos(reconcile_by.saturating_sub(now)))
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
