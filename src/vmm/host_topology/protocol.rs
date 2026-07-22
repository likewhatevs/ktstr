//! Cross-process host-resource admission under nextest.
//!
//! Every ktstr process sharing a lock directory participates in one v19
//! fixed-record mmap registry. A ticket publishes one exact, non-empty CPU/LLC
//! reservation claim plus the resources its planner may watch. Claims preserve
//! the resource-lock semantics exactly: CPU and LLC claims independently use
//! the same SH/EX compatibility matrix as `flock`.
//!
//! Admission is work-conserving without weakening those reservations. One
//! elected coordinator scans tickets in monotonic order and grants a waiter
//! when its exact claim is compatible with every earlier live claim. Thus an
//! incompatible predecessor remains a hard fence, while fully disjoint work can
//! pass it. Coordinator probes are all-or-nothing: a failed target releases
//! every real lock acquired by that attempt before waiting, so an exact queue
//! reservation never sequesters an otherwise usable subset of the host.
//!
//! Nonqueued acquisition remains all-or-nothing. It holds the registry's shared
//! fence from the aggregate-claim check through the real nonblocking flock
//! probe, so a ticket cannot publish in the check-to-acquire gap. A failed probe
//! releases every lock it acquired.
//!
//! Waiting scales with tickets rather than threads or directory watches. Each
//! ordinary waiter sleeps on its own shared futex word; the sole coordinator
//! owns one filtered inotify watch and re-plans against live holders after a
//! relevant resource release or registry notification. Per-ticket liveness
//! flocks make crash pruning authoritative, and dirty transactions rebuild
//! derived aggregates from fixed records after an interrupted mutation.
//!
//! Elapsed time never revokes a live holder's reservation. Fallback wakes exist
//! only to recover a missed event or detect a crashed process; guest and process
//! lifecycle mechanisms remain responsible for genuinely wedged work.

use anyhow::Result;
use std::collections::BTreeSet;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::flock::{FlockMode, TryFlockOutcome, try_flock_with_witness};

mod exec_handoff;
mod registry;

#[cfg(test)]
pub(crate) use exec_handoff::EXEC_HANDOFF_ENV;
pub(crate) use exec_handoff::{prepare_pending_exec_handoff, take_pending_exec_handoff};

/// Stable identity of one host reservation lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ResourceKey {
    Llc(usize),
    Cpu(usize),
    Permit(usize),
}

/// Registry-visible intent attached to a complete resource claim.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum AdmissionClass {
    #[default]
    Ordinary,
    DefaultBorrow,
    Build,
}

/// One canonical host-resource lock together with the exact registry identity
/// it represents. Keeping the identity attached avoids reparsing test-specific
/// lock paths and lets coordinator attempts publish mode-correct state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResourceLock {
    pub(crate) path: String,
    pub(crate) mode: FlockMode,
    pub(crate) resource: ResourceKey,
}

/// Exact evidence retained from a failed nonblocking resource probe.
pub(crate) struct ContentionEvidence {
    pub(crate) blocker: ResourceKey,
    pub(crate) mode: FlockMode,
    pub(crate) _witness: OwnedFd,
}

impl std::fmt::Debug for ContentionEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ContentionEvidence")
            .field("blocker", &self.blocker)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContentionMarker {
    pub(crate) blocker: ResourceKey,
    pub(crate) mode: FlockMode,
}

#[cfg(test)]
pub(crate) struct CpuExContentionSharedWake {
    pub(crate) scans: u64,
    pub(crate) shared_granted: bool,
    pub(crate) exclusive_waiting: bool,
    pub(crate) sh_serial_advanced: bool,
    pub(crate) ex_serial_unchanged: bool,
    pub(crate) shared_woke: bool,
    pub(crate) exclusive_not_woken: bool,
    pub(crate) coordinator_did_not_replan: bool,
}

impl Ord for ContentionMarker {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.blocker, u8::from(self.mode == FlockMode::Exclusive))
            .cmp(&(other.blocker, u8::from(other.mode == FlockMode::Exclusive)))
    }
}

impl PartialOrd for ContentionMarker {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl ContentionEvidence {
    pub(crate) fn marker(&self) -> ContentionMarker {
        ContentionMarker {
            blocker: self.blocker,
            mode: self.mode,
        }
    }
}

#[derive(Default)]
pub(crate) struct ContentionSet {
    markers: BTreeSet<ContentionMarker>,
    witness: Option<ContentionEvidence>,
}

impl ContentionSet {
    pub(crate) fn insert(&mut self, evidence: ContentionEvidence) {
        self.markers.insert(evidence.marker());
        // Keep every exact blocker marker but only one writable witness. One
        // post-publication CLOSE_WRITE is sufficient for ordering, while the
        // bounded fd count matters on hosts with hundreds of alternatives.
        if self.witness.is_none() {
            self.witness = Some(evidence);
        }
    }

    pub(crate) fn marker_vec(&self) -> Vec<ContentionMarker> {
        self.markers.iter().copied().collect()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }
}

impl From<ContentionEvidence> for ContentionSet {
    fn from(evidence: ContentionEvidence) -> Self {
        let mut set = Self::default();
        set.insert(evidence);
        set
    }
}

pub(crate) enum ProbeOutcome<T> {
    Acquired(T),
    Contended(ContentionEvidence),
    Unavailable,
}

pub(crate) trait IntoProbeOutcome<T> {
    fn into_probe_outcome(self) -> ProbeOutcome<T>;
}

impl<T> IntoProbeOutcome<T> for ProbeOutcome<T> {
    fn into_probe_outcome(self) -> ProbeOutcome<T> {
        self
    }
}

impl<T> IntoProbeOutcome<T> for Option<T> {
    fn into_probe_outcome(self) -> ProbeOutcome<T> {
        self.map_or(ProbeOutcome::Unavailable, ProbeOutcome::Acquired)
    }
}

/// Fallback wake for the coordinator's inotify sleep, bounding the staleness
/// window if an event is missed (e.g. a release racing the
/// drain-before-sleep gap). The real wake is the inotify event.
const COORDINATOR_WAKE_FALLBACK: Duration = Duration::from_secs(30);
/// A failed authoritative availability proof stays pending in the registry.
/// Retry it at a deliberately slow cadence; exact lock-close watches wake the
/// coordinator immediately when the holder actually changes.
const OBSERVATION_RETRY_FALLBACK: Duration = Duration::from_secs(5);
/// Claim-specific waiter watches should receive every relevant release or
/// predecessor-state transition. This long, per-ticket-staggered fallback is
/// only a missed-event recovery tick; it avoids turning hundreds of queued
/// cells into a synchronized `/proc/locks` polling herd.
const WAITER_CRASH_RECOVERY_BASE: Duration = Duration::from_secs(3);
const WAIT_DIAGNOSTIC_INITIAL_DELAY: Duration = Duration::from_secs(10);
const WAIT_DIAGNOSTIC_INTERVAL: Duration = Duration::from_secs(30);
/// Bound the watch-install handoff gap without making every short-lived
/// coordinator scan the full registry. The first coordinator that remains
/// active past this shared, non-postponable deadline performs one sweep;
/// ordinary liveness closes after watch installation remain event-driven.
const PREWATCH_LIVENESS_RECONCILE_DELAY: Duration = Duration::from_millis(500);
/// Bound one coordinator turn's inotify consumption. The kernel API returns
/// at most 4 KiB per read, so four reads and 1,024 decoded events are matching
/// hard caps; the wall-time cap also protects unusually expensive name
/// classification. Remaining events are processed after one registry turn.
const INOTIFY_TURN_MAX_READS: usize = 4;
const INOTIFY_TURN_MAX_EVENTS: usize = 1_024;
const INOTIFY_TURN_MAX_TIME: Duration = Duration::from_millis(1);
/// Explicit cfg(test)-only retry transport cadence. This is a semantic
/// coordinator retry deadline, not a short slice of the 30-second real-inotify
/// deadline: each expiry runs schedule + replan again.
#[cfg(test)]
const TEST_RETRY_WAKE_INTERVAL: Duration = Duration::from_millis(8);
#[cfg(test)]
const TEST_RETRY_WAKE_MARKER: &str = ".ktstr-test-retry-wake";

/// Directory the protocol files live in — derived from the LLC
/// lockfile path so the test-only lock-prefix override isolates the
/// current registry files into the same per-test tempdir as the LLC locks
/// they coordinate.
fn protocol_dir() -> PathBuf {
    Path::new(&super::llc_lock_path(0))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

/// Sharing mode of one resource class in an exact [`ClaimSet`].
///
/// Its compatibility rules exactly match `flock`: a shared claim only fences
/// an exclusive requester, while an exclusive claim fences both shared and
/// exclusive requesters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum ClaimMode {
    #[default]
    Exclusive,
    Shared,
}

impl From<FlockMode> for ClaimMode {
    fn from(mode: FlockMode) -> Self {
        match mode {
            FlockMode::Exclusive => Self::Exclusive,
            FlockMode::Shared => Self::Shared,
        }
    }
}

/// One ticket's exact LLC and CPU reservation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ClaimSet {
    pub llcs: BTreeSet<usize>,
    pub cpus: BTreeSet<usize>,
    pub permits: BTreeSet<usize>,
    pub llc_mode: ClaimMode,
    pub cpu_mode: ClaimMode,
    pub permit_mode: ClaimMode,
    pub admission_class: AdmissionClass,
}

impl ClaimSet {
    fn from_all_claim_modes(
        llcs: impl IntoIterator<Item = usize>,
        cpus: impl IntoIterator<Item = usize>,
        permits: impl IntoIterator<Item = usize>,
        llc_mode: ClaimMode,
        cpu_mode: ClaimMode,
        permit_mode: ClaimMode,
    ) -> Self {
        let llcs = llcs.into_iter().collect::<BTreeSet<_>>();
        let cpus = cpus.into_iter().collect::<BTreeSet<_>>();
        let permits = permits.into_iter().collect::<BTreeSet<_>>();
        Self {
            llc_mode: if llcs.is_empty() {
                ClaimMode::Exclusive
            } else {
                llc_mode
            },
            cpu_mode: if cpus.is_empty() {
                ClaimMode::Exclusive
            } else {
                cpu_mode
            },
            permit_mode: if permits.is_empty() {
                ClaimMode::Exclusive
            } else {
                permit_mode
            },
            llcs,
            cpus,
            permits,
            admission_class: AdmissionClass::Ordinary,
        }
    }

    pub(super) fn with_all_claim_modes(
        llcs: impl IntoIterator<Item = usize>,
        cpus: impl IntoIterator<Item = usize>,
        permits: impl IntoIterator<Item = usize>,
        llc_mode: ClaimMode,
        cpu_mode: ClaimMode,
        permit_mode: ClaimMode,
    ) -> Self {
        Self::from_all_claim_modes(llcs, cpus, permits, llc_mode, cpu_mode, permit_mode)
    }

    pub(crate) fn new(
        llcs: impl IntoIterator<Item = usize>,
        cpus: impl IntoIterator<Item = usize>,
        llc_mode: FlockMode,
    ) -> Self {
        Self::from_all_claim_modes(
            llcs,
            cpus,
            std::iter::empty(),
            llc_mode.into(),
            ClaimMode::Exclusive,
            ClaimMode::Exclusive,
        )
    }

    pub(crate) fn with_modes(
        llcs: impl IntoIterator<Item = usize>,
        cpus: impl IntoIterator<Item = usize>,
        llc_mode: FlockMode,
        cpu_mode: FlockMode,
    ) -> Self {
        Self::from_all_claim_modes(
            llcs,
            cpus,
            std::iter::empty(),
            llc_mode.into(),
            cpu_mode.into(),
            ClaimMode::Exclusive,
        )
    }

    pub(crate) fn with_permits(
        llcs: impl IntoIterator<Item = usize>,
        cpus: impl IntoIterator<Item = usize>,
        permits: impl IntoIterator<Item = usize>,
        llc_mode: FlockMode,
        cpu_mode: FlockMode,
        permit_mode: FlockMode,
    ) -> Self {
        Self::from_all_claim_modes(
            llcs,
            cpus,
            permits,
            llc_mode.into(),
            cpu_mode.into(),
            permit_mode.into(),
        )
    }

    pub(crate) fn with_admission_class(mut self, admission_class: AdmissionClass) -> Self {
        self.admission_class = admission_class;
        self
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.llcs.is_empty() && self.cpus.is_empty() && self.permits.is_empty()
    }

    /// Union two alternative footprints while preserving the strongest mode
    /// actually present in each resource class. The canonical mode attached
    /// to an empty class is ignored.
    pub(crate) fn union_envelope(&self, other: &Self) -> Self {
        let strongest = |a_empty: bool, a_mode, b_empty: bool, b_mode| match (a_empty, b_empty) {
            (true, true) => ClaimMode::Exclusive,
            (true, false) => b_mode,
            (false, true) => a_mode,
            (false, false) if a_mode == ClaimMode::Exclusive || b_mode == ClaimMode::Exclusive => {
                ClaimMode::Exclusive
            }
            (false, false) => ClaimMode::Shared,
        };
        Self::from_all_claim_modes(
            self.llcs.union(&other.llcs).copied(),
            self.cpus.union(&other.cpus).copied(),
            self.permits.union(&other.permits).copied(),
            strongest(
                self.llcs.is_empty(),
                self.llc_mode,
                other.llcs.is_empty(),
                other.llc_mode,
            ),
            strongest(
                self.cpus.is_empty(),
                self.cpu_mode,
                other.cpus.is_empty(),
                other.cpu_mode,
            ),
            strongest(
                self.permits.is_empty(),
                self.permit_mode,
                other.permits.is_empty(),
                other.permit_mode,
            ),
        )
        .with_admission_class(self.admission_class.max(other.admission_class))
    }

    /// Whether `request_mode` on `llc_idx` is incompatible with this live
    /// ticket claim. This is the same compatibility matrix as `flock`.
    #[cfg(test)]
    pub(crate) fn conflicts_with_llc(&self, llc_idx: usize, request_mode: FlockMode) -> bool {
        self.llcs.contains(&llc_idx)
            && matches!(
                (self.llc_mode, request_mode),
                (ClaimMode::Exclusive, _) | (ClaimMode::Shared, FlockMode::Exclusive)
            )
    }

    /// Whether two complete reservation claims are incompatible.
    ///
    /// CPU and LLC compatibility independently follow the flock SH/EX matrix.
    #[cfg(test)]
    pub(crate) fn conflicts_with(&self, other: &Self) -> bool {
        if self.cpus.iter().any(|cpu| other.cpus.contains(cpu))
            && matches!(
                (self.cpu_mode, other.cpu_mode),
                (ClaimMode::Exclusive, _) | (_, ClaimMode::Exclusive)
            )
        {
            return true;
        }
        if self
            .permits
            .iter()
            .any(|permit| other.permits.contains(permit))
            && matches!(
                (self.permit_mode, other.permit_mode),
                (ClaimMode::Exclusive, _) | (_, ClaimMode::Exclusive)
            )
        {
            return true;
        }
        self.llcs.iter().any(|llc| {
            other.llcs.contains(llc)
                && matches!(
                    (self.llc_mode, other.llc_mode),
                    (ClaimMode::Exclusive, _) | (_, ClaimMode::Exclusive)
                )
        })
    }
}

/// Whether a fast-path reservation conflicts with an exact live ticket claim.
/// Three registry aggregates answer the common case in O(host bitset) rather
/// than scanning O(waiters).
#[cfg(test)]
pub(crate) fn registered_claim_conflicts(candidate: &ClaimSet) -> Result<bool> {
    registry::aggregate_conflicts(candidate)
}

pub(crate) struct RegisteredClaimSnapshot {
    inner: registry::AggregateSnapshot,
}

impl RegisteredClaimSnapshot {
    pub(crate) fn conflicts(&self, candidate: &ClaimSet) -> Result<bool> {
        self.inner.conflicts(candidate)
    }

    pub(crate) fn cpu_holder_count(&self, cpu: usize) -> Result<usize> {
        self.inner.cpu_holder_count(cpu)
    }

    pub(crate) fn cpu_exclusive_held(&self, cpu: usize) -> Result<bool> {
        self.inner.cpu_exclusive_held(cpu)
    }

    pub(crate) fn cpu_build_claimed(&self, cpu: usize) -> Result<bool> {
        self.inner.cpu_build_claimed(cpu)
    }

    pub(crate) fn llc_holder_count(&self, llc: usize) -> Result<usize> {
        self.inner.llc_holder_count(llc)
    }

    pub(crate) fn llc_exclusive_held(&self, llc: usize) -> Result<bool> {
        self.inner.llc_exclusive_held(llc)
    }
}

/// Copy the aggregate reservation bitsets once for a whole planning pass.
/// Candidate filtering after this call is process-local; the final selected
/// claim still enters [`with_registry_fence`] for authoritative admission.
pub(crate) fn registered_claim_snapshot(required: &ClaimSet) -> Result<RegisteredClaimSnapshot> {
    Ok(RegisteredClaimSnapshot {
        inner: registry::aggregate_snapshot(required)?,
    })
}

/// Copy aggregate reservation state only when the registry's shared fence is
/// immediately available. Interactive admission uses this to preserve its
/// no-wait contract under concurrent queue publication or repair.
pub(crate) fn try_registered_claim_snapshot(
    required: &ClaimSet,
) -> Result<Option<RegisteredClaimSnapshot>> {
    Ok(registry::try_aggregate_snapshot(required)?.map(|inner| RegisteredClaimSnapshot { inner }))
}

#[cfg(test)]
pub(crate) fn aggregate_snapshot_read_count_for_tests() -> usize {
    registry::aggregate_snapshot_read_count_for_tests()
}

#[cfg(test)]
pub(crate) fn union_claims_for_tests(a: &ClaimSet, b: &ClaimSet) -> ClaimSet {
    registry::union_claims_for_tests(a, b)
}

#[cfg(test)]
pub(crate) fn round_trip_claim_modes_for_tests(
    claim: &ClaimSet,
    watch: &ClaimSet,
) -> Result<(ClaimSet, ClaimSet)> {
    registry::round_trip_claim_modes_for_tests(claim, watch)
}

pub(crate) enum RegistryFence<T> {
    Fenced,
    Ran {
        value: T,
        #[allow(dead_code)]
        watched: bool,
    },
}

/// Run one all-or-nothing nonblocking resource probe while holding the
/// registry's shared fence. Ticket publication and claim replacement require
/// EX, so no earlier exact claim can appear between the aggregate check and
/// the actual flock acquisition.
pub(crate) fn with_registry_fence<T>(
    candidate: &ClaimSet,
    run: impl FnOnce() -> Result<T>,
) -> Result<RegistryFence<T>> {
    match registry::with_aggregate_fence(candidate, run) {
        Ok(registry::FenceResult::Fenced) => Ok(RegistryFence::Fenced),
        Ok(registry::FenceResult::Ran { value, watched }) => {
            Ok(RegistryFence::Ran { value, watched })
        }
        Err(error) => Err(error),
    }
}

/// One coordinator-owned wake transport.
///
/// Production always constructs `RealInotify` and propagates initialization
/// failure. TestRetry exists only in cfg(test), and only when a unique
/// LockPrefixesGuard marker has been installed explicitly.
pub(crate) struct LockDirWatch {
    backend: LockDirWatchBackend,
}

enum LockDirWatchBackend {
    RealInotify(RealInotifyWake),
    #[cfg(test)]
    TestRetry,
}

/// Real coordinator-owned inotify watch over the lock directory.
///
/// Only resource-lock closes and the explicit registry notification are
/// actionable. Registry header/chunk/liveness opens are coordinator-internal
/// traffic; filtering their close events is essential because a generation
/// read immediately before `poll` would otherwise wake the coordinator itself
/// forever.
struct RealInotifyWake {
    ino: nix::sys::inotify::Inotify,
    event_wd: nix::sys::inotify::WatchDescriptor,
    resource_watches: std::collections::BTreeMap<nix::sys::inotify::WatchDescriptor, ResourceWatch>,
    notify_name: std::ffi::OsString,
}

#[derive(Default)]
struct ResourceWatch {
    llc_prefix: Option<std::ffi::OsString>,
    cpu_prefix: Option<std::ffi::OsString>,
    permit_prefix: Option<std::ffi::OsString>,
}

#[derive(Default)]
pub(crate) struct LockDirEvents {
    llc_closes: BTreeSet<usize>,
    cpu_closes: BTreeSet<usize>,
    permit_closes: BTreeSet<usize>,
    registry_notify: bool,
    liveness_closes: BTreeSet<(u64, u64)>,
    overflow: bool,
    backlog: bool,
}

impl LockDirEvents {
    fn merge(&mut self, other: Self) {
        self.llc_closes.extend(other.llc_closes);
        self.cpu_closes.extend(other.cpu_closes);
        self.permit_closes.extend(other.permit_closes);
        self.registry_notify |= other.registry_notify;
        self.liveness_closes.extend(other.liveness_closes);
        self.overflow |= other.overflow;
        self.backlog |= other.backlog;
    }

    fn is_actionable(&self) -> bool {
        !self.llc_closes.is_empty()
            || !self.cpu_closes.is_empty()
            || !self.permit_closes.is_empty()
            || self.registry_notify
            || !self.liveness_closes.is_empty()
            || self.overflow
            || self.backlog
    }

    #[cfg(test)]
    pub(crate) fn contains_liveness(&self, identity: (u64, u64)) -> bool {
        self.liveness_closes.contains(&identity)
    }

    #[cfg(test)]
    pub(crate) fn contains_registry_notify(&self) -> bool {
        self.registry_notify
    }

    #[cfg(test)]
    pub(crate) fn has_backlog(&self) -> bool {
        self.backlog
    }

    #[cfg(test)]
    pub(crate) fn overflowed(&self) -> bool {
        self.overflow
    }
}

impl RealInotifyWake {
    /// Watch writable closes only. Resource holders, registry notifications,
    /// and liveness owners all use O_RDWR/O_WRONLY descriptors. Liveness
    /// verification deliberately uses O_RDONLY, so omitting CLOSE_NOWRITE
    /// prevents those probes from entering the inotify queue at all.
    fn new() -> Result<Self> {
        use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
        let llc_path = super::llc_lock_path(0);
        let cpu_path = super::cpu_lock_path(0);
        let permit_path = super::permit_lock_path(0);
        let event_dir = registry::event_dir();
        let llc_dir = Path::new(&llc_path)
            .parent()
            .ok_or_else(|| anyhow::anyhow!("LLC resource lock path has no parent: {llc_path}"))?
            .to_path_buf();
        let cpu_dir = Path::new(&cpu_path)
            .parent()
            .ok_or_else(|| anyhow::anyhow!("CPU resource lock path has no parent: {cpu_path}"))?
            .to_path_buf();
        let permit_dir = Path::new(&permit_path)
            .parent()
            .ok_or_else(|| {
                anyhow::anyhow!("permit resource lock path has no parent: {permit_path}")
            })?
            .to_path_buf();
        if event_dir == llc_dir || event_dir == cpu_dir || event_dir == permit_dir {
            anyhow::bail!(
                "admission event directory must be distinct from resource lock directories"
            );
        }
        let ino = Inotify::init(InitFlags::IN_NONBLOCK | InitFlags::IN_CLOEXEC)?;
        std::fs::create_dir_all(&event_dir)?;
        let event_wd = ino.add_watch(&event_dir, AddWatchFlags::IN_CLOSE_WRITE)?;

        let mut resources = std::collections::BTreeMap::<PathBuf, ResourceWatch>::new();
        resources.entry(llc_dir).or_default().llc_prefix =
            Some(resource_basename_prefix(&llc_path)?);
        resources.entry(cpu_dir).or_default().cpu_prefix =
            Some(resource_basename_prefix(&cpu_path)?);
        resources.entry(permit_dir).or_default().permit_prefix =
            Some(resource_basename_prefix(&permit_path)?);
        let mut resource_watches = std::collections::BTreeMap::new();
        for (dir, resource) in resources {
            std::fs::create_dir_all(&dir)?;
            let wd = ino.add_watch(&dir, AddWatchFlags::IN_CLOSE_WRITE)?;
            resource_watches.insert(wd, resource);
        }
        Ok(RealInotifyWake {
            ino,
            event_wd,
            resource_watches,
            notify_name: registry::notify_basename()?,
        })
    }

    /// Drain one bounded turn of queued events without blocking.
    /// The coordinator calls this right before sleeping so its own
    /// open/close churn from the attempt it just made does not wake
    /// it straight back up (a self-wake busy loop). An external
    /// release slipping into the drain-to-sleep gap is caught by the
    /// [`COORDINATOR_WAKE_FALLBACK`] tick.
    pub(crate) fn drain(&self, watched: &ClaimSet) -> Result<LockDirEvents> {
        self.read_bounded(watched)
    }

    fn read_bounded(&self, watched: &ClaimSet) -> Result<LockDirEvents> {
        let mut batch = LockDirEvents::default();
        let started = std::time::Instant::now();
        let mut reads = 0usize;
        let mut decoded = 0usize;
        loop {
            if reads >= INOTIFY_TURN_MAX_READS
                || decoded >= INOTIFY_TURN_MAX_EVENTS
                || started.elapsed() >= INOTIFY_TURN_MAX_TIME
            {
                // Budget exhaustion is not kernel queue overflow. It merely
                // asks the coordinator to run one scheduling turn before
                // consuming the next bounded chunk.
                batch.backlog = true;
                break;
            }
            match self.ino.read_events() {
                Ok(events) if events.is_empty() => break,
                Ok(events) => {
                    reads += 1;
                    decoded = decoded.saturating_add(events.len());
                    self.classify(events, watched, &mut batch);
                }
                Err(nix::errno::Errno::EAGAIN) => break,
                Err(error) => return Err(error.into()),
            }
        }
        Ok(batch)
    }

    /// Sleep until a resource release/registry notification arrives or
    /// `timeout` passes. Protocol-internal close events are consumed without
    /// returning so coordinator bookkeeping cannot create a self-wake loop.
    pub(crate) fn wait(
        &self,
        timeout: Duration,
        watched: &ClaimSet,
    ) -> Result<Option<LockDirEvents>> {
        use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
        use std::os::fd::AsFd;
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            let mut fds = [PollFd::new(self.ino.as_fd(), PollFlags::POLLIN)];
            let ms =
                u16::try_from(remaining.as_millis().clamp(1, u16::MAX as u128)).unwrap_or(u16::MAX);
            if poll(&mut fds, PollTimeout::from(ms))? == 0 {
                return Ok(None);
            }
            let batch = self.read_bounded(watched)?;
            if batch.is_actionable() {
                return Ok(Some(batch));
            }
        }
    }

    fn classify(
        &self,
        events: Vec<nix::sys::inotify::InotifyEvent>,
        watched: &ClaimSet,
        batch: &mut LockDirEvents,
    ) {
        use nix::sys::inotify::AddWatchFlags;

        for event in events {
            if event.mask.contains(AddWatchFlags::IN_Q_OVERFLOW) {
                batch.overflow = true;
            }
            let Some(name) = event.name.as_deref() else {
                continue;
            };
            if event.wd == self.event_wd {
                if name == self.notify_name {
                    batch.registry_notify = true;
                } else if event.mask.contains(AddWatchFlags::IN_CLOSE_WRITE)
                    && let Some(identity) = registry::parse_liveness_basename(name)
                {
                    // The owner holds an O_RDWR fd, so owner death/finish
                    // yields CLOSE_WRITE. O(1) liveness probes deliberately
                    // use O_RDONLY and yield CLOSE_NOWRITE.
                    batch.liveness_closes.insert(identity);
                }
                continue;
            }
            let Some(resource) = self.resource_watches.get(&event.wd) else {
                continue;
            };
            if let Some(index) = resource
                .llc_prefix
                .as_deref()
                .and_then(|prefix| resource_index(name, prefix))
                .filter(|index| watched.llcs.contains(index))
            {
                batch.llc_closes.insert(index);
            }
            if let Some(index) = resource
                .cpu_prefix
                .as_deref()
                .and_then(|prefix| resource_index(name, prefix))
                .filter(|index| watched.cpus.contains(index))
            {
                batch.cpu_closes.insert(index);
            }
            if let Some(index) = resource
                .permit_prefix
                .as_deref()
                .and_then(|prefix| resource_index(name, prefix))
                .filter(|index| watched.permits.contains(index))
            {
                batch.permit_closes.insert(index);
            }
        }
    }
}

impl LockDirWatch {
    pub(crate) fn new() -> Result<Self> {
        #[cfg(test)]
        if test_retry_wake_marker_path().is_file() {
            return Ok(Self {
                backend: LockDirWatchBackend::TestRetry,
            });
        }
        Self::new_real_wake()
    }

    fn new_real_wake() -> Result<Self> {
        RealInotifyWake::new().map(|wake| Self {
            backend: LockDirWatchBackend::RealInotify(wake),
        })
    }

    /// Dedicated real-inotify constructor for its contract tests. It bypasses
    /// the cfg(test) retry marker and remains fail-fast.
    #[cfg(test)]
    pub(crate) fn new_real_for_tests() -> Result<Self> {
        Self::new_real_wake()
    }

    pub(crate) fn drain(&self, watched: &ClaimSet) -> Result<LockDirEvents> {
        match &self.backend {
            LockDirWatchBackend::RealInotify(wake) => wake.drain(watched),
            #[cfg(test)]
            LockDirWatchBackend::TestRetry => Ok(LockDirEvents::default()),
        }
    }

    pub(crate) fn wait(
        &self,
        timeout: Duration,
        watched: &ClaimSet,
    ) -> Result<Option<LockDirEvents>> {
        match &self.backend {
            LockDirWatchBackend::RealInotify(wake) => wake.wait(timeout, watched),
            #[cfg(test)]
            LockDirWatchBackend::TestRetry => {
                std::thread::sleep(timeout);
                Ok(None)
            }
        }
    }

    fn semantic_retry_interval(&self, observation_pending: bool) -> Duration {
        match &self.backend {
            LockDirWatchBackend::RealInotify(_) => {
                if observation_pending {
                    OBSERVATION_RETRY_FALLBACK
                } else {
                    COORDINATOR_WAKE_FALLBACK
                }
            }
            #[cfg(test)]
            LockDirWatchBackend::TestRetry => TEST_RETRY_WAKE_INTERVAL,
        }
    }
}

#[cfg(test)]
fn test_retry_wake_marker_path() -> PathBuf {
    registry::protocol_dir_path().join(TEST_RETRY_WAKE_MARKER)
}

#[cfg(test)]
pub(crate) fn test_retry_wake_marker_path_for_tests() -> PathBuf {
    test_retry_wake_marker_path()
}

/// Publish mode-correct holder state while the corresponding real flocks are
/// still live. This is an authoritative state transition, not an event hint.
pub(crate) fn publish_acquired<T>(claim: &ClaimSet, value: T) -> Result<Acquired<T>> {
    registry::publish_acquired(claim).map(|held| Acquired::tracked(value, held))
}

fn resource_basename_prefix(path: &str) -> Result<std::ffi::OsString> {
    let basename = Path::new(path)
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| anyhow::anyhow!("resource lock path has no UTF-8 basename: {path}"))?;
    let prefix = basename.strip_suffix("0.lock").ok_or_else(|| {
        anyhow::anyhow!("resource lock basename does not end in an indexed lock name: {basename}")
    })?;
    Ok(prefix.into())
}

fn resource_index(name: &std::ffi::OsStr, prefix: &std::ffi::OsStr) -> Option<usize> {
    let name = name.to_str()?;
    let prefix = prefix.to_str()?;
    name.strip_prefix(prefix)?
        .strip_suffix(".lock")?
        .parse()
        .ok()
}

fn interrupted() -> anyhow::Error {
    std::io::Error::new(
        std::io::ErrorKind::Interrupted,
        "ktstr resource acquisition interrupted",
    )
    .into()
}

fn check_interrupted(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        Err(interrupted())
    } else {
        Ok(())
    }
}

/// Prefer the caller's authoritative cancellation verdict over an incidental
/// syscall error (most commonly the `EINTR` produced by the private wake).
fn check_result<T>(result: Result<T>, cancelled: Option<&AtomicBool>) -> Result<T> {
    match result {
        Ok(value) => {
            check_interrupted(cancelled)?;
            Ok(value)
        }
        Err(error) => {
            check_interrupted(cancelled)?;
            Err(error)
        }
    }
}

pub(crate) struct GrantedProbe {
    designated: ClaimSet,
    watch: ClaimSet,
    next_claim: ClaimSet,
    acquisition_allowed: bool,
    contention: Option<ContentionEvidence>,
    predecessors: registry::AggregateSnapshot,
    availability: registry::AvailabilitySnapshot,
    reusable_permits: Vec<(usize, OwnedFd)>,
}

impl GrantedProbe {
    /// A grant covers exactly the ticket's designated reservation. Alternative
    /// plans become the next designation and are considered by the coordinator
    /// on the following scheduling pass.
    pub(crate) fn allows(&self, candidate: &ClaimSet) -> bool {
        *candidate == self.designated
    }

    pub(crate) fn designated(&self) -> &ClaimSet {
        &self.designated
    }

    pub(crate) fn clone_reusable_permits(&self) -> Result<Vec<(usize, OwnedFd)>> {
        self.reusable_permits
            .iter()
            .map(|(permit, fd)| Ok((*permit, fd.try_clone()?)))
            .collect()
    }

    /// Whether `candidate` conflicts with any exact claim preceding this
    /// ticket. Flexible planners must consult this in addition to live flock
    /// holders: an earlier waiter reserves its designated resources before it
    /// acquires the corresponding kernel flocks, so `/proc/locks` cannot see
    /// the full admission prefix.
    pub(crate) fn conflicts_with_predecessors(&self, candidate: &ClaimSet) -> Result<bool> {
        self.predecessors.conflicts(candidate)
    }

    /// Whether one complete alternative is eligible to become this ticket's
    /// next exact designation. This is a scan-time hint only: the coordinator
    /// revalidates the published exact claim before granting its real probe.
    pub(crate) fn candidate_ready(&self, candidate: &ClaimSet) -> Result<bool> {
        registry::validate_claim_within_watch(candidate, &self.watch)?;
        // A failed SH probe proves an EX holder and blocks either candidate
        // mode. A failed EX probe may have met only an SH holder, so retain an
        // SH alternative; the coordinator still revalidates it before grant.
        let just_contended =
            self.contention
                .as_ref()
                .is_some_and(|evidence| match evidence.blocker {
                    ResourceKey::Cpu(cpu) => {
                        candidate.cpus.contains(&cpu)
                            && (evidence.mode == FlockMode::Shared
                                || candidate.cpu_mode == ClaimMode::Exclusive)
                    }
                    ResourceKey::Llc(llc) => {
                        candidate.llcs.contains(&llc)
                            && (evidence.mode == FlockMode::Shared
                                || candidate.llc_mode == ClaimMode::Exclusive)
                    }
                    ResourceKey::Permit(permit) => {
                        candidate.permits.contains(&permit)
                            && (evidence.mode == FlockMode::Shared
                                || candidate.permit_mode == ClaimMode::Exclusive)
                    }
                });
        Ok(!just_contended
            && !self.predecessors.conflicts(candidate)?
            && self.availability.allows(candidate)?)
    }

    pub(crate) fn candidate_holder_pressure(&self, candidate: &ClaimSet) -> Result<usize> {
        let cpu = candidate.cpus.iter().try_fold(0usize, |total, &cpu| {
            Ok::<_, anyhow::Error>(total.saturating_add(self.predecessors.cpu_holder_count(cpu)?))
        })?;
        candidate.llcs.iter().try_fold(cpu, |total, &llc| {
            Ok::<_, anyhow::Error>(total.saturating_add(self.predecessors.llc_holder_count(llc)?))
        })
    }

    pub(crate) fn try_acquire<T, O: IntoProbeOutcome<T>>(
        &mut self,
        candidate: &ClaimSet,
        acquire: impl FnOnce() -> Result<O>,
    ) -> Result<Option<T>> {
        self.try_acquire_impl(candidate, acquire, true)
    }

    /// Probe default mode's preferred unshared placement without turning a
    /// miss into durable queue contention. The published claim deliberately
    /// remains CPU-SH: a failed CPU-EX probe is only the signal to try the
    /// shared fallback, not a blocker that the registry must wait to clear.
    pub(crate) fn try_acquire_default_exact<T, O: IntoProbeOutcome<T>>(
        &mut self,
        candidate: &ClaimSet,
        acquire: impl FnOnce() -> Result<O>,
    ) -> Result<Option<T>> {
        anyhow::ensure!(
            candidate.cpu_mode == ClaimMode::Shared,
            "default exact probe requires a shared published CPU claim",
        );
        anyhow::ensure!(
            candidate.llc_mode == ClaimMode::Shared,
            "default exact probe requires a shared published LLC claim",
        );
        anyhow::ensure!(
            candidate.permit_mode == ClaimMode::Exclusive,
            "default exact probe requires exclusive weighted permits",
        );
        self.try_acquire_impl(candidate, acquire, false)
    }

    fn try_acquire_impl<T, O: IntoProbeOutcome<T>>(
        &mut self,
        candidate: &ClaimSet,
        acquire: impl FnOnce() -> Result<O>,
        retain_contention: bool,
    ) -> Result<Option<T>> {
        if !self.acquisition_allowed || !self.allows(candidate) {
            return Ok(None);
        }
        match acquire()?.into_probe_outcome() {
            ProbeOutcome::Acquired(value) => Ok(Some(value)),
            ProbeOutcome::Contended(evidence) => {
                if retain_contention {
                    self.contention = Some(evidence);
                }
                Ok(None)
            }
            ProbeOutcome::Unavailable => Ok(None),
        }
    }

    pub(crate) fn reserve(&mut self, candidate: &ClaimSet) -> Result<()> {
        if candidate.is_empty() {
            anyhow::bail!("a queued waiter cannot reserve an empty claim");
        }
        self.next_claim = candidate.clone();
        Ok(())
    }
}

pub(crate) struct CoordinatorTicket {
    ticket: registry::Ticket,
    preparation: Option<super::PreparationPermit>,
    /// Extra release events relevant only while selecting physical
    /// preparation. This envelope is never published as a claim/watch.
    preparation_watch: Option<ClaimSet>,
}

#[cfg(test)]
impl CoordinatorTicket {
    pub(crate) fn read_state_shared_for_tests(&self) -> Result<()> {
        self.ticket.read_state_shared_for_tests()
    }
}

/// A physical reservation and its registry publication.
///
/// Drop releases the physical value first (normally a complete flock set),
/// then removes the HELD registry record and notifies waiters. This preserves
/// conservative ordering in both directions: acquisition publishes only after
/// the real flocks exist, while teardown never advertises free capacity before
/// those flocks close.
pub(crate) struct Acquired<T> {
    value: Option<T>,
    held: Option<registry::HeldClaim>,
}

impl<T> Acquired<T> {
    fn tracked(value: T, held: registry::HeldClaim) -> Self {
        Self {
            value: Some(value),
            held: Some(held),
        }
    }

    pub(crate) fn untracked(value: T) -> Self {
        Self {
            value: Some(value),
            held: None,
        }
    }

    pub(crate) fn split_map<R, U>(mut self, map: impl FnOnce(T) -> (R, U)) -> (R, Acquired<U>) {
        let value = self
            .value
            .take()
            .expect("acquired reservation payload was already consumed");
        let held = self.held.take();
        let (result, value) = map(value);
        (
            result,
            Acquired {
                value: Some(value),
                held,
            },
        )
    }

    #[cfg(test)]
    pub(crate) fn abandon_publication_for_tests(mut self) {
        // Match process teardown ordering: physical flocks disappear first,
        // then the liveness fd closes while the HELD record remains for a
        // different participant to prune.
        drop(self.value.take());
        if let Some(held) = self.held.take() {
            registry::abandon_held_for_tests(held);
        }
    }
}

impl<T> std::ops::Deref for Acquired<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
            .as_ref()
            .expect("acquired reservation payload was already consumed")
    }
}

impl<T> std::ops::DerefMut for Acquired<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value
            .as_mut()
            .expect("acquired reservation payload was already consumed")
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Acquired<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Acquired")
            .field("value", &self.value)
            .field("registry_held", &self.held.is_some())
            .finish()
    }
}

impl<T> Drop for Acquired<T> {
    fn drop(&mut self) {
        // Do not rely on field declaration order: the physical reservation
        // must disappear before its registry record can advertise a release.
        drop(self.value.take());
        drop(self.held.take());
    }
}

#[cfg(test)]
pub(crate) fn set_held_drop_hook_for_tests(hook: impl FnOnce() + 'static) {
    registry::set_held_drop_hook_for_tests(hook);
}

pub(crate) enum TicketWork<T> {
    Acquired(Acquired<T>),
    Coordinator(Box<CoordinatorTicket>),
}

/// Same-PID pre-exec admission identity. Its registry claim continuously
/// publishes every physically held preparation resource, while its watch keeps
/// the selected final intent attached to the same ordered ticket. Activation
/// atomically replaces both with the exact run claim/watch.
pub(crate) struct PendingAdmission {
    ticket: Option<registry::Ticket>,
    preparation: Option<super::PreparationPermit>,
    pending_claim: Option<ClaimSet>,
}

impl Drop for PendingAdmission {
    fn drop(&mut self) {
        // Keep the authoritative registry publication in place until every
        // physical preparation OFD has been released. Otherwise ordinary
        // field-order destruction removes the ticket first and briefly leaves
        // unregistered CPU/permit contention behind on error and cancellation
        // paths.
        drop(self.preparation.take());
        drop(self.ticket.take());
    }
}

/// The only probe surface available while atomically consuming a PENDING
/// admission. It can test candidates against external registry claims and
/// reuse the preparation owner's permit OFDs, but cannot queue or retry.
pub(crate) struct PendingOneShotProbe<'a> {
    registry: &'a registry::PendingOneShotProbe<'a>,
    reusable_permits: &'a [(usize, OwnedFd)],
}

impl PendingOneShotProbe<'_> {
    pub(crate) fn candidate_ready(&self, candidate: &ClaimSet) -> Result<bool> {
        self.registry.candidate_ready(candidate)
    }

    pub(crate) fn clone_reusable_permits(&self) -> Result<Vec<(usize, OwnedFd)>> {
        self.reusable_permits
            .iter()
            .map(|(permit, fd)| Ok((*permit, fd.try_clone()?)))
            .collect()
    }

    pub(crate) fn try_acquire<T>(
        &self,
        candidate: &ClaimSet,
        acquire: impl FnOnce() -> Result<ProbeOutcome<T>>,
    ) -> Result<Option<T>> {
        if !self.candidate_ready(candidate)? {
            return Ok(None);
        }
        Ok(match acquire()? {
            ProbeOutcome::Acquired(value) => Some(value),
            ProbeOutcome::Contended(_) | ProbeOutcome::Unavailable => None,
        })
    }
}

impl PendingAdmission {
    pub(crate) fn exec_handoff_parts(&self) -> Result<(u64, u64, std::os::fd::RawFd)> {
        self.ticket
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("pending admission was already consumed"))?
            .pending_exec_handoff_parts()
    }

    pub(crate) fn preparation_handoff_parts(
        &self,
    ) -> Result<(usize, Vec<(usize, std::os::fd::RawFd)>)> {
        use std::os::fd::AsRawFd;
        let preparation = self
            .preparation
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("pending preparation permit was already consumed"))?;
        Ok((
            preparation.index,
            preparation
                .permit_fds
                .iter()
                .map(|(permit, fd)| (*permit, fd.as_raw_fd()))
                .collect(),
        ))
    }

    pub(crate) fn preparation_affinity_handoff_parts(
        &self,
    ) -> Result<(usize, std::os::fd::RawFd, &[usize])> {
        self.preparation
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("pending preparation permit was already consumed"))?
            .affinity_handoff_parts()
    }

    pub(crate) fn restore_preparation_affinity(&mut self) -> Result<()> {
        self.preparation
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("pending preparation permit was already consumed"))?
            .restore_affinity()
    }

    #[cfg(test)]
    pub(crate) fn pending_claim_watch_for_tests(&self) -> Result<(ClaimSet, ClaimSet)> {
        self.ticket
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("pending admission was already consumed"))?
            .pending_claim_watch_for_tests()
    }

    /// Complete immutable preparation without consuming or weakening the
    /// PENDING ticket. Affinity is restored, while the physical preparation
    /// claim, selected-intent watch, and all preparation OFDs remain intact
    /// until exact activation replaces them atomically.
    pub(crate) fn finish_preparation(&mut self) -> Result<()> {
        self.ticket
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("pending admission was already consumed"))?;
        self.pending_claim
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("pending admission claim was already consumed"))?;
        let preparation = self
            .preparation
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("pending preparation permit was already consumed"))?;
        preparation.restore_affinity()?;
        Ok(())
    }

    pub(crate) fn preparation_cpu_permits(&self) -> &[usize] {
        self.preparation
            .as_ref()
            .map_or(&[], |preparation| preparation.cpu_permits.as_slice())
    }

    pub(crate) fn preparation_memory_permits(&self) -> &[usize] {
        self.preparation
            .as_ref()
            .map_or(&[], |preparation| preparation.memory_permits.as_slice())
    }

    fn take_parts(&mut self) -> Result<(registry::Ticket, super::PreparationPermit, ClaimSet)> {
        let ticket = self
            .ticket
            .take()
            .ok_or_else(|| anyhow::anyhow!("pending admission was already consumed"))?;
        let preparation = self
            .preparation
            .take()
            .ok_or_else(|| anyhow::anyhow!("pending preparation permit was already consumed"))?;
        let pending_claim = self
            .pending_claim
            .take()
            .ok_or_else(|| anyhow::anyhow!("pending admission claim was already consumed"))?;
        Ok((ticket, preparation, pending_claim))
    }

    fn from_imported_ticket(
        ticket: registry::Ticket,
        preparation: super::PreparationPermit,
        pending_claim: ClaimSet,
    ) -> Self {
        Self {
            ticket: Some(ticket),
            preparation: Some(preparation),
            pending_claim: Some(pending_claim),
        }
    }
}

fn pending_admission_from_parts(
    ticket: registry::Ticket,
    preparation: super::PreparationPermit,
    pending_claim: ClaimSet,
) -> Result<PendingAdmission> {
    let mut pending = PendingAdmission {
        ticket: Some(ticket),
        preparation: Some(preparation),
        pending_claim: Some(pending_claim),
    };
    pending
        .preparation
        .as_mut()
        .expect("fresh pending admission owns preparation")
        .constrain_affinity()?;
    Ok(pending)
}

/// Publish a lightweight exact/flexible run intent before acquiring any
/// physical preparation capacity. The ordinary queue selects among all
/// visible intents. A selected callback probes the preparation pool exactly
/// once and either commits the physical preparation tuple under the same
/// selected-intent ticket or revokes/requeues the selection.
pub(crate) fn register_intent_for_preparation(
    initial_claim: ClaimSet,
    watch: ClaimSet,
    mut granted_candidate: impl FnMut(&GrantedProbe) -> Result<Option<ClaimSet>>,
    mut coordinator_candidate: impl FnMut(&HeldLocks) -> Result<Option<ClaimSet>>,
) -> Result<PendingAdmission> {
    let preparation_watch = super::preparation_resource_watch()?;
    let host_allowed = super::host_allowed_cpus();
    anyhow::ensure!(
        !host_allowed.is_empty(),
        "could not determine allowed CPU set for selected-intent preparation admission",
    );
    registry::validate_claim_within_watch(&initial_claim, &watch)?;
    let mut ticket = registry::Ticket::register_after_contention_with_capacity(
        initial_claim.clone(),
        watch,
        None,
        None,
        registry::required_bits_for_claim(&preparation_watch),
    )?;
    let mut rotation_bias = 0usize;
    loop {
        super::tick_reservation_wait_progress();
        match ticket.state_or_wait(WAITER_CRASH_RECOVERY_BASE, None)? {
            registry::State::Granted | registry::State::Replan => {
                let result = ticket.run_granted(
                    None,
                    |designated, watch, acquisition_allowed, predecessors, availability| {
                        let mut probe = GrantedProbe {
                            designated: designated.clone(),
                            watch: watch.clone(),
                            next_claim: designated.clone(),
                            acquisition_allowed,
                            contention: None,
                            predecessors,
                            availability,
                            reusable_permits: Vec::new(),
                        };
                        let selected = granted_candidate(&probe)?;
                        if !acquisition_allowed {
                            if let Some(candidate) = selected {
                                probe.reserve(&candidate)?;
                            }
                            return Ok(registry::GrantAttempt {
                                acquired: None,
                                preparation_claim: None,
                                preparation_contention: None,
                                next_claim: probe.next_claim,
                                contention: None,
                            });
                        }
                        let Some(candidate) = selected else {
                            return Ok(registry::GrantAttempt {
                                acquired: None,
                                preparation_claim: None,
                                preparation_contention: None,
                                next_claim: probe.next_claim,
                                contention: None,
                            });
                        };
                        if candidate != *designated {
                            probe.reserve(&candidate)?;
                            return Ok(registry::GrantAttempt {
                                acquired: None,
                                preparation_claim: None,
                                preparation_contention: None,
                                next_claim: probe.next_claim,
                                contention: None,
                            });
                        }

                        let preparation_cpus =
                            super::preparation_affinity_candidates(&designated.cpus, &host_allowed);
                        anyhow::ensure!(
                            !preparation_cpus.is_empty(),
                            "selected final claim has no CPU in the process cpuset",
                        );
                        let selected = super::try_preparation_candidates_once(
                            rotation_bias,
                            &preparation_cpus,
                            |preparation, claim| {
                                Ok(super::PreparationCandidateDecision::Accepted((
                                    preparation,
                                    claim,
                                )))
                            },
                        )?;
                        rotation_bias = rotation_bias.wrapping_add(1);
                        let (acquired, preparation_claim, preparation_contention) = match selected {
                            super::PreparationProbe::Acquired((preparation, claim)) => {
                                (Some(preparation), Some(claim), None)
                            }
                            super::PreparationProbe::Contended(evidence) => {
                                (None, None, Some(evidence))
                            }
                            super::PreparationProbe::Unavailable => (None, None, None),
                        };
                        Ok(registry::GrantAttempt {
                            acquired,
                            preparation_claim,
                            preparation_contention,
                            next_claim: probe.next_claim,
                            contention: None,
                        })
                    },
                )?;
                match result {
                    registry::GrantResult::Prepared(preparation, pending_claim) => {
                        return pending_admission_from_parts(ticket, preparation, pending_claim);
                    }
                    registry::GrantResult::Acquired(_, _) => {
                        unreachable!("intent callback published its run claim as HELD")
                    }
                    registry::GrantResult::Requeued | registry::GrantResult::LostGrant => {}
                }
            }
            registry::State::Coordinator => {
                let coordinator = CoordinatorTicket {
                    ticket,
                    preparation: None,
                    preparation_watch: Some(preparation_watch.clone()),
                };
                let mut step = |held: &mut HeldLocks| {
                    let Some(selected) = coordinator_candidate(held)? else {
                        return Ok(CoordinatorStep::<()>::Waiting {
                            claim: held.designated()?.clone(),
                        });
                    };
                    if !held.candidate_ready(&selected)? {
                        return Ok(CoordinatorStep::<()>::Waiting { claim: selected });
                    }
                    let preparation_cpus =
                        super::preparation_affinity_candidates(&selected.cpus, &host_allowed);
                    anyhow::ensure!(
                        !preparation_cpus.is_empty(),
                        "selected final claim has no CPU in the process cpuset",
                    );
                    let prepared = super::try_preparation_candidates_once(
                        rotation_bias,
                        &preparation_cpus,
                        |preparation, claim| {
                            Ok(super::PreparationCandidateDecision::Accepted((
                                preparation,
                                claim,
                            )))
                        },
                    )?;
                    rotation_bias = rotation_bias.wrapping_add(1);
                    Ok(match prepared {
                        super::PreparationProbe::Acquired((preparation, claim)) => {
                            CoordinatorStep::Prepare {
                                final_claim: selected,
                                preparation_claim: claim,
                                preparation: Box::new(preparation),
                            }
                        }
                        super::PreparationProbe::Contended(evidence) => {
                            held.record_external_contention(evidence);
                            CoordinatorStep::Waiting { claim: selected }
                        }
                        super::PreparationProbe::Unavailable => {
                            CoordinatorStep::Waiting { claim: selected }
                        }
                    })
                };
                return match acquire_as_coordinator(coordinator, &mut step)? {
                    CoordinatorOutcome::Prepared(pending) => Ok(*pending),
                    CoordinatorOutcome::Acquired(_) => {
                        unreachable!("intent coordinator published its run claim as HELD")
                    }
                    CoordinatorOutcome::Aborted { reason } => {
                        Err(anyhow::Error::new(super::ResourceContention { reason }))
                    }
                };
            }
            registry::State::Waiting => {}
            registry::State::CoordinatorStandby => {
                anyhow::bail!("new intent entered coordinator standby before ownership handoff")
            }
        }
    }
}

/// Publish one bounded preparation owner without sleeping.
///
/// Every physical preparation slot is considered once. A physical-capacity
/// miss or a conflicting older registry claim returns `None`; unlike normal
/// generated-test pre-admission, this path never waits for either to change.
pub(crate) fn try_register_pending_admission(
    max_permit_index: usize,
) -> Result<Option<PendingAdmission>> {
    let required = registry::required_bits_for_permit_index(max_permit_index);
    let allowed = super::host_allowed_cpus();
    match super::try_preparation_candidates_once(0, &allowed, |preparation, claim| {
        let pending_claim = claim.clone();
        Ok(
            match registry::Ticket::try_register_pending(required, claim)? {
                Some(registry::PendingRegistration::Registered(ticket)) => {
                    super::PreparationCandidateDecision::Accepted(pending_admission_from_parts(
                        ticket,
                        preparation,
                        pending_claim,
                    )?)
                }
                Some(registry::PendingRegistration::Contended(_)) => {
                    drop(preparation);
                    super::PreparationCandidateDecision::Retry
                }
                None => {
                    drop(preparation);
                    super::PreparationCandidateDecision::Contended
                }
            },
        )
    })? {
        super::PreparationProbe::Acquired(pending) => Ok(Some(pending)),
        super::PreparationProbe::Contended(_) | super::PreparationProbe::Unavailable => Ok(None),
    }
}

pub(crate) fn register_pending_admission(max_permit_index: usize) -> Result<PendingAdmission> {
    let required = registry::required_bits_for_permit_index(max_permit_index);
    let mut rotation_bias = 0usize;
    loop {
        // Scan every preparation slot before waiting.  Publication occurs
        // under the registry EX lock; if an older READY claim appeared after
        // the physical probe, registration returns None and all flocks are
        // dropped before the next rotated scan.
        let (preparation, claim) = super::acquire_preparation_permit(rotation_bias)?;
        match registry::Ticket::register_pending(required, claim.clone())? {
            registry::PendingRegistration::Registered(ticket) => {
                return pending_admission_from_parts(ticket, preparation, claim);
            }
            registry::PendingRegistration::Contended(generation) => {
                drop(preparation);
                registry::wait_for_generation_change(generation, Duration::from_secs(2))?;
            }
        }
        rotation_bias = rotation_bias.wrapping_add(1);
    }
}

#[cfg(test)]
pub(crate) struct PreparationContinuityForTests {
    pub(crate) pending: PendingAdmission,
    pub(crate) ticket: u64,
    pub(crate) affinity_cpu: usize,
    pub(crate) cpu_permits: Vec<usize>,
    pub(crate) memory_permits: Vec<usize>,
    pub(crate) token_permit: usize,
    pub(crate) pending_claim: ClaimSet,
}

#[cfg(test)]
pub(crate) fn exercise_preparation_continuity_for_tests() -> Result<PreparationContinuityForTests> {
    let (preparation, active) = super::acquire_preparation_permit(0)?;
    let affinity_cpu = preparation.affinity_cpu;
    let cpu_permits = preparation.cpu_permits.clone();
    let memory_permits = preparation.memory_permits.clone();
    let token_permit = preparation.token_permit;
    let required =
        registry::required_bits_for_permit_index(super::admission_resource_capacity_hint()?);
    let ticket = match registry::Ticket::register_pending(required, active.clone())? {
        registry::PendingRegistration::Registered(ticket) => ticket,
        registry::PendingRegistration::Contended(_) => {
            anyhow::bail!("isolated preparation transition unexpectedly contended")
        }
    };
    // Deliberately bypass affinity constraining: this helper exercises
    // completion-time claim/OFD continuity without perturbing sibling threads.
    let mut pending = PendingAdmission {
        ticket: Some(ticket),
        preparation: Some(preparation),
        pending_claim: Some(active),
    };
    let ticket_id = pending
        .ticket
        .as_ref()
        .expect("fresh test pending ticket")
        .pending_exec_handoff_parts()?
        .1;
    pending.finish_preparation()?;
    let pending_claim = pending
        .preparation
        .as_ref()
        .expect("completed test preparation")
        .claim();
    Ok(PreparationContinuityForTests {
        pending,
        ticket: ticket_id,
        affinity_cpu,
        cpu_permits,
        memory_permits,
        token_permit,
        pending_claim,
    })
}

#[cfg(test)]
pub(crate) fn ticket_registry_snapshot_for_tests() -> Result<Vec<(u64, u32, ClaimSet)>> {
    registry::snapshot()
}

#[cfg(test)]
pub(crate) fn ticket_registry_diagnostics_for_tests() -> Result<String> {
    registry::diagnostics_for_tests()
}

#[cfg(test)]
pub(crate) fn persist_wait_diagnostic_for_tests(root: &std::path::Path, bucket: u64) -> Result<()> {
    registry::persist_wait_diagnostic(root, bucket, bucket * 30)
}

#[cfg(test)]
pub(crate) fn exercise_pending_activation_overlap_watch_for_tests() -> Result<(bool, bool, bool)> {
    registry::exercise_pending_activation_overlap_watch_for_tests()
}

#[cfg(test)]
pub(crate) struct PendingClaimForTests {
    ticket: Option<registry::Ticket>,
}

#[cfg(test)]
impl PendingClaimForTests {
    pub(crate) fn retire_synchronously(mut self) -> Result<()> {
        self.ticket
            .as_mut()
            .expect("test pending claim was already consumed")
            .finish(None)
    }

    pub(crate) fn try_activate_once<T>(
        mut self,
        exact: ClaimSet,
        attempt: impl FnOnce() -> Result<Option<T>>,
    ) -> Result<Option<Acquired<T>>> {
        let mut ticket = self
            .ticket
            .take()
            .expect("test pending claim was already consumed");
        let expected = ticket_registry_snapshot_for_tests()?
            .into_iter()
            .find(|(candidate, _, _)| {
                ticket
                    .pending_exec_handoff_parts()
                    .is_ok_and(|(_, owned, _)| *candidate == owned)
            })
            .map(|(_, _, claim)| claim)
            .ok_or_else(|| anyhow::anyhow!("test PENDING claim disappeared"))?;
        let activated = ticket.try_activate_pending_once(&expected, |probe| {
            if !probe.candidate_ready(&exact)? {
                return Ok(None);
            }
            Ok(attempt()?.map(|value| (exact, value)))
        })?;
        Ok(match activated {
            registry::PendingOneShotResult::Acquired(value, held) => {
                Some(Acquired::tracked(value, held))
            }
            registry::PendingOneShotResult::Unavailable => None,
        })
    }
}

#[cfg(test)]
pub(crate) fn register_pending_claim_for_tests(claim: ClaimSet) -> Result<PendingClaimForTests> {
    Ok(PendingClaimForTests {
        ticket: Some(registry::register_pending_claim_for_tests(claim)?),
    })
}

#[cfg(test)]
pub(crate) fn registry_ex_acquisition_count_for_tests() -> u64 {
    registry::registry_ex_acquisition_count_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_resource_weighted_backfill_accounting_for_tests() -> (u32, u32, u32, u32) {
    registry::exercise_resource_weighted_backfill_accounting_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_work_conserving_backfill_for_tests()
-> Result<registry::WorkConservingBackfillOutcome> {
    registry::exercise_work_conserving_backfill_for_tests()
}

#[cfg(test)]
pub(crate) fn expire_coordinator_lease_for_tests() -> Result<()> {
    registry::expire_coordinator_lease_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_stalled_takeover_notification_for_tests(
    watch: &LockDirWatch,
) -> Result<(bool, bool, bool, bool)> {
    registry::exercise_stalled_takeover_notification_for_tests(watch)
}

#[cfg(test)]
pub(crate) fn exercise_dirty_repair_notification_for_tests(
    watch: &LockDirWatch,
) -> Result<(bool, bool, bool, bool)> {
    registry::exercise_dirty_repair_notification_for_tests(watch)
}

#[cfg(test)]
pub(crate) fn exercise_clean_coordinator_mismatch_recovery_for_tests() -> Result<()> {
    registry::exercise_clean_coordinator_mismatch_recovery_for_tests()
}

#[cfg(test)]
pub(crate) fn churn_registry_generation_for_tests(rounds: usize) -> Result<()> {
    registry::churn_registry_generation_for_tests(rounds)
}

#[cfg(test)]
pub(crate) fn active_free_head_is_rejected_for_tests() -> Result<()> {
    registry::active_free_head_is_rejected_for_tests()
}

#[cfg(test)]
pub(crate) fn cancel_granted_after_commit_for_tests() {
    registry::cancel_granted_after_commit_for_tests();
}

#[cfg(test)]
pub(crate) fn cancel_coordinator_after_commit_for_tests() {
    registry::cancel_coordinator_after_commit_for_tests();
}

#[cfg(test)]
pub(crate) fn exercise_known_free_close_storm_for_tests(
    closes: usize,
) -> Result<(usize, u64, usize, u64, u64)> {
    registry::exercise_known_free_close_storm_for_tests(closes)
}

#[cfg(test)]
pub(crate) fn exercise_stale_heartbeat_known_free_close_for_tests()
-> Result<(usize, u64, usize, u64, u64)> {
    registry::exercise_stale_heartbeat_known_free_close_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_llc_sh_only_shared_to_free_close_for_tests() -> Result<(bool, bool, u64, u64)>
{
    registry::exercise_llc_sh_only_shared_to_free_close_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_busy_to_free_close_for_tests() -> Result<(usize, u64, usize)> {
    registry::exercise_busy_to_free_close_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_llc_ex_contention_shared_wake_for_tests() -> Result<(u64, bool, bool, bool)>
{
    registry::exercise_llc_ex_contention_shared_wake_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_cpu_ex_contention_shared_wake_for_tests() -> Result<CpuExContentionSharedWake>
{
    registry::exercise_cpu_ex_contention_shared_wake_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_coordinator_turnover_for_tests(
    coordinators: usize,
) -> Result<(u64, usize, u64, bool)> {
    registry::exercise_coordinator_turnover_for_tests(coordinators)
}

#[cfg(test)]
pub(crate) fn defer_liveness_maintenance_for_tests() -> Result<()> {
    registry::defer_liveness_maintenance_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_exact_commit_scan_elision_for_tests(commits: usize) -> Result<u64> {
    registry::exercise_exact_commit_scan_elision_for_tests(commits)
}

#[cfg(test)]
pub(crate) fn exercise_mismatched_commit_rescan_for_tests() -> Result<(u64, bool)> {
    registry::exercise_mismatched_commit_rescan_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_superset_commit_rescan_for_tests() -> Result<(u64, bool)> {
    registry::exercise_superset_commit_rescan_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_shared_commit_improvement_for_tests() -> Result<(u64, bool)> {
    registry::exercise_shared_commit_improvement_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_cpu_shared_commit_improvement_for_tests() -> Result<(u64, bool)> {
    registry::exercise_cpu_shared_commit_improvement_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_cpu_mode_repair_for_tests() -> Result<(bool, bool, bool, bool)> {
    registry::exercise_cpu_mode_repair_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_replan_token_wave_for_tests(
    waiters: usize,
) -> Result<registry::ReplanTokenWaveOutcome> {
    registry::exercise_replan_token_wave_for_tests(waiters)
}

#[cfg(test)]
pub(crate) fn exercise_replan_crash_repair_for_tests() -> Result<registry::ReplanCrashRepairOutcome>
{
    registry::exercise_replan_crash_repair_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_intrascan_fence_epoch_for_tests() -> Result<(bool, bool, bool)> {
    registry::exercise_intrascan_fence_epoch_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_grant_scan_crash_fence_for_tests() -> Result<(bool, bool, bool)> {
    registry::exercise_grant_scan_crash_fence_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_granular_prefix_invalidation_for_tests()
-> Result<registry::GranularPrefixInvalidationOutcome> {
    registry::exercise_granular_prefix_invalidation_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_granted_serial_scope_for_tests() -> Result<(bool, bool, bool, bool)> {
    registry::exercise_granted_serial_scope_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_revocation_ack_for_tests() -> Result<registry::RevocationAckOutcome> {
    registry::exercise_revocation_ack_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_revoked_owner_death_for_tests() -> Result<(bool, bool, bool, bool)> {
    registry::exercise_revoked_owner_death_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_revoke_crash_repair_for_tests() -> Result<(bool, bool, bool, bool)> {
    registry::exercise_revoke_crash_repair_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_waiting_release_wake_for_tests() -> Result<(bool, bool, bool, bool)> {
    registry::exercise_waiting_release_wake_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_one_shot_replacement_for_tests()
-> Result<(usize, bool, bool, bool, bool, usize)> {
    registry::exercise_one_shot_replacement_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_prefix_epoch_validation_for_tests() -> Result<(usize, bool, bool)> {
    registry::exercise_prefix_epoch_validation_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_prefix_order_and_repair_for_tests() -> Result<(bool, bool, bool, bool)> {
    registry::exercise_prefix_order_and_repair_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_prefix_refresh_after_predecessor_release_for_tests()
-> Result<(bool, bool, bool, bool)> {
    registry::exercise_prefix_refresh_after_predecessor_release_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_waiting_publication_release_progress_for_tests()
-> Result<(bool, bool, bool, bool)> {
    registry::exercise_waiting_publication_release_progress_for_tests()
}

#[cfg(test)]
pub(crate) fn waiting_publication_requires_immediate_turn_for_tests(
    should_step: bool,
    observation_pending: bool,
) -> bool {
    waiting_publication_requires_immediate_turn(should_step, observation_pending)
}

#[cfg(test)]
pub(crate) fn exercise_issue_serial_race_for_tests() -> Result<(bool, bool, bool, bool)> {
    registry::exercise_issue_serial_race_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_stale_acquired_release_order_for_tests()
-> Result<(bool, bool, bool, bool, bool, bool)> {
    registry::exercise_stale_acquired_release_order_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_stale_contention_commit_for_tests() -> Result<(bool, bool, bool, bool)> {
    registry::exercise_stale_contention_commit_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_candidate_ready_matrix_for_tests() -> Result<()> {
    let build_watch = ClaimSet::with_permits(
        0usize..96,
        0usize..192,
        0usize..231,
        FlockMode::Shared,
        FlockMode::Shared,
        FlockMode::Exclusive,
    )
    .with_admission_class(AdmissionClass::Build);
    let ordinary_candidate = ClaimSet::with_modes(
        std::iter::empty(),
        [0usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let error = registry::validate_claim_within_watch(&ordinary_candidate, &build_watch)
        .expect_err("an ordinary physical probe must not enter a build watch");
    let diagnostic = error.to_string();
    anyhow::ensure!(
        diagnostic.contains("(admission class)")
            && diagnostic.contains("cpus(len=192 [")
            && diagnostic.contains("permits(len=231 [")
            && diagnostic.len() < 1_024,
        "immutable-watch diagnostics must identify the invariant without dumping host-sized sets: {diagnostic}",
    );
    registry::validate_claim_within_watch(
        &ordinary_candidate.with_admission_class(AdmissionClass::Build),
        &build_watch,
    )?;

    let predecessor =
        ClaimSet::with_modes([1usize], [1usize], FlockMode::Shared, FlockMode::Shared);
    let (predecessors, availability) = registry::probe_snapshots_for_tests(
        &[predecessor],
        &[
            (1, Some(registry::CpuAvailability::Free)),
            (2, Some(registry::CpuAvailability::SharedHeld)),
            (3, Some(registry::CpuAvailability::ExclusiveHeld)),
            (4, None),
            (5, Some(registry::CpuAvailability::Free)),
        ],
        &[
            (1, Some(registry::LlcAvailability::Free)),
            (2, Some(registry::LlcAvailability::SharedHeld)),
            (3, Some(registry::LlcAvailability::ExclusiveHeld)),
            (4, None),
            (5, Some(registry::LlcAvailability::Free)),
        ],
    )?;
    let watch = ClaimSet::with_modes(
        1usize..=5,
        1usize..=5,
        FlockMode::Exclusive,
        FlockMode::Exclusive,
    );
    let designated = ClaimSet::with_modes(
        std::iter::empty(),
        [5usize],
        FlockMode::Exclusive,
        FlockMode::Shared,
    );
    let mut probe = GrantedProbe {
        designated: designated.clone(),
        watch,
        next_claim: designated.clone(),
        acquisition_allowed: false,
        contention: None,
        predecessors,
        availability,
        reusable_permits: Vec::new(),
    };
    let cpu =
        |index, mode| ClaimSet::with_modes(std::iter::empty(), [index], FlockMode::Exclusive, mode);
    let llc =
        |index, mode| ClaimSet::with_modes([index], std::iter::empty(), mode, FlockMode::Exclusive);

    anyhow::ensure!(
        probe.candidate_ready(&cpu(1, FlockMode::Shared))?
            && !probe.candidate_ready(&cpu(1, FlockMode::Exclusive))?
            && probe.candidate_ready(&llc(1, FlockMode::Shared))?
            && !probe.candidate_ready(&llc(1, FlockMode::Exclusive))?,
        "SH predecessors must admit SH but reject EX candidates independently for CPU and LLC"
    );
    anyhow::ensure!(
        probe.candidate_ready(&cpu(2, FlockMode::Shared))?
            && !probe.candidate_ready(&cpu(2, FlockMode::Exclusive))?
            && probe.candidate_ready(&llc(2, FlockMode::Shared))?
            && !probe.candidate_ready(&llc(2, FlockMode::Exclusive))?,
        "shared-held availability must admit SH and reject EX candidates"
    );
    anyhow::ensure!(
        !probe.candidate_ready(&cpu(3, FlockMode::Shared))?
            && !probe.candidate_ready(&cpu(3, FlockMode::Exclusive))?
            && !probe.candidate_ready(&llc(3, FlockMode::Shared))?
            && !probe.candidate_ready(&llc(3, FlockMode::Exclusive))?
            && !probe.candidate_ready(&cpu(4, FlockMode::Shared))?
            && !probe.candidate_ready(&llc(4, FlockMode::Shared))?,
        "exclusive-held and unknown resources must reject every candidate mode"
    );
    anyhow::ensure!(
        probe
            .candidate_ready(&cpu(6, FlockMode::Exclusive))
            .is_err(),
        "a candidate outside the immutable watch must be rejected"
    );

    probe.contention = Some(ContentionEvidence {
        blocker: ResourceKey::Cpu(5),
        mode: FlockMode::Shared,
        _witness: std::fs::File::open("/dev/null")?.into(),
    });
    anyhow::ensure!(
        !probe.candidate_ready(&designated)?
            && probe.candidate_ready(&llc(5, FlockMode::Exclusive))?,
        "failed SH contention must suppress every mode on only that resource"
    );
    probe.contention = Some(ContentionEvidence {
        blocker: ResourceKey::Cpu(5),
        mode: FlockMode::Exclusive,
        _witness: std::fs::File::open("/dev/null")?.into(),
    });
    anyhow::ensure!(
        probe.candidate_ready(&designated)?
            && !probe.candidate_ready(&cpu(5, FlockMode::Exclusive))?,
        "failed EX contention must retain a compatible SH alternative while suppressing EX"
    );
    probe.contention = None;

    let mut ran = false;
    let acquired = probe.try_acquire(&designated, || {
        ran = true;
        Ok::<Option<()>, anyhow::Error>(Some(()))
    })?;
    anyhow::ensure!(
        acquired.is_none() && !ran,
        "REPLAN must never run even its designated acquisition"
    );
    probe.acquisition_allowed = true;
    let acquired = probe.try_acquire(&cpu(2, FlockMode::Shared), || {
        ran = true;
        Ok::<Option<()>, anyhow::Error>(Some(()))
    })?;
    anyhow::ensure!(
        acquired.is_none() && !ran,
        "a GRANTED callback must not directly acquire an alternate candidate"
    );
    Ok(())
}

#[cfg(test)]
pub(crate) fn registry_initializer_temp_count_for_tests() -> Result<usize> {
    registry::initializer_temp_count_for_tests()
}

#[cfg(test)]
pub(crate) fn observer_preserves_uninitialized_header_for_tests() -> Result<bool> {
    registry::observer_preserves_uninitialized_header_for_tests()
}

#[cfg(test)]
pub(crate) fn prepare_zeroed_uninitialized_header_for_tests() -> Result<()> {
    registry::prepare_zeroed_uninitialized_header_for_tests()
}

#[cfg(test)]
pub(crate) fn hold_registry_shared_for_tests() -> Result<OwnedFd> {
    registry::hold_registry_shared_for_tests()
}

#[cfg(test)]
pub(crate) fn hold_registry_exclusive_for_tests() -> Result<OwnedFd> {
    registry::hold_registry_exclusive_for_tests()
}

#[cfg(test)]
pub(crate) fn shared_state_read_count_for_tests() -> usize {
    registry::shared_state_read_count_for_tests()
}

#[cfg(test)]
pub(crate) fn resource_epoch_for_tests() -> Result<u64> {
    registry::resource_epoch_for_tests()
}

#[cfg(test)]
pub(crate) fn ticket_blocked_at_current_serial_for_tests(pid: u32) -> Result<bool> {
    registry::ticket_blocked_at_current_serial_for_tests(pid)
}

#[cfg(test)]
pub(crate) fn ticket_is_waiting_for_tests(pid: u32) -> Result<bool> {
    registry::ticket_is_waiting_for_tests(pid)
}

#[cfg(test)]
pub(crate) fn ticket_is_revoked_for_tests(pid: u32) -> Result<bool> {
    registry::ticket_is_revoked_for_tests(pid)
}

#[cfg(test)]
pub(crate) fn coordinator_liveness_probe_for_tests() -> Result<((u64, u64), bool)> {
    registry::coordinator_liveness_probe_for_tests()
}

#[cfg(test)]
pub(crate) fn missing_liveness_probe_does_not_create_for_tests() -> Result<bool> {
    registry::missing_liveness_probe_does_not_create_for_tests()
}

#[cfg(test)]
pub(crate) fn registry_event_dir_for_tests() -> PathBuf {
    registry::event_dir()
}

#[cfg(test)]
pub(crate) fn exercise_registry_high_water_for_tests(waiters: usize) -> Result<usize> {
    let claim = ClaimSet::new(std::iter::empty(), [1usize], FlockMode::Exclusive);
    let mut tickets = Vec::with_capacity(waiters);
    for _ in 0..waiters {
        tickets.push(registry::Ticket::register(
            claim.clone(),
            claim.clone(),
            None,
        )?);
    }
    let snapshot = registry::snapshot()?;
    if let Some(coordinator) = tickets.first_mut() {
        coordinator.schedule(
            Some(&claim),
            &BTreeSet::new(),
            &BTreeSet::new(),
            &BTreeSet::new(),
            false,
            &[],
            &[],
            false,
            None,
            false,
            None,
        )?;
    }
    // Unmap every non-coordinator futex before its slot reaches the freelist;
    // release the coordinator last so the test does not manufacture 499
    // leadership handoffs while checking chunk growth.
    while tickets.len() > 1 {
        tickets.pop();
    }
    drop(tickets);
    Ok(snapshot.len())
}

#[cfg(test)]
pub(crate) fn exercise_granted_only_drain_election_reads_for_tests(
    waiters: usize,
) -> Result<usize> {
    registry::exercise_granted_only_drain_election_reads_for_tests(waiters)
}

/// Register one exact, non-empty priority claim in the fixed-record registry.
/// Only the elected coordinator scans records and routes grants; ordinary
/// waiters sleep on their own shared futex word and therefore add neither a
/// helper thread nor an inotify instance to a storm.
pub(crate) fn register_ticket_or_acquire<T>(
    initial_claim: ClaimSet,
    watch_claim: ClaimSet,
    cancelled: Option<&AtomicBool>,
    try_acquire: impl FnMut(&mut GrantedProbe) -> Result<Option<T>>,
) -> Result<TicketWork<T>> {
    register_ticket_or_acquire_impl(initial_claim, watch_claim, None, cancelled, try_acquire)
}

pub(crate) fn register_ticket_after_contention<T>(
    initial_claim: ClaimSet,
    watch_claim: ClaimSet,
    contention: ContentionEvidence,
    cancelled: Option<&AtomicBool>,
    try_acquire: impl FnMut(&mut GrantedProbe) -> Result<Option<T>>,
) -> Result<TicketWork<T>> {
    register_ticket_or_acquire_impl(
        initial_claim,
        watch_claim,
        Some(contention.into()),
        cancelled,
        try_acquire,
    )
}

pub(crate) fn register_ticket_after_contentions<T>(
    initial_claim: ClaimSet,
    watch_claim: ClaimSet,
    contention: ContentionSet,
    cancelled: Option<&AtomicBool>,
    try_acquire: impl FnMut(&mut GrantedProbe) -> Result<Option<T>>,
) -> Result<TicketWork<T>> {
    register_ticket_or_acquire_impl(
        initial_claim,
        watch_claim,
        Some(contention),
        cancelled,
        try_acquire,
    )
}

fn register_ticket_or_acquire_impl<T>(
    initial_claim: ClaimSet,
    watch_claim: ClaimSet,
    initial_contention: Option<ContentionSet>,
    cancelled: Option<&AtomicBool>,
    mut try_acquire: impl FnMut(&mut GrantedProbe) -> Result<Option<T>>,
) -> Result<TicketWork<T>> {
    check_interrupted(cancelled)?;
    let ticket = check_result(
        registry::Ticket::register_after_contention(
            initial_claim,
            watch_claim,
            initial_contention,
            cancelled,
        ),
        cancelled,
    )?;
    drive_registered_ticket(ticket, cancelled, &mut try_acquire, None)
}

/// Consume a same-PID PENDING record and publish the complete ready claim in
/// that exact slot before entering the ordinary grant/coordinator loop.
pub(crate) fn activate_pending_ticket<T>(
    mut pending: PendingAdmission,
    initial_claim: ClaimSet,
    watch_claim: ClaimSet,
    cancelled: Option<&AtomicBool>,
    mut try_acquire: impl FnMut(&mut GrantedProbe) -> Result<Option<T>>,
) -> Result<TicketWork<T>> {
    check_interrupted(cancelled)?;
    let (mut ticket, mut preparation, pending_claim) = pending.take_parts()?;
    anyhow::ensure!(
        !preparation.affinity_constrained,
        "preparation affinity must be restored before exact activation",
    );
    check_result(
        ticket.activate_pending(&pending_claim, initial_claim, watch_claim, cancelled),
        cancelled,
    )?;
    preparation.release_resources_for_exact()?;
    // Keep only the preparation token until exact physical ownership is
    // published. It bounds resident prepared processes without causing the
    // exact claim to self-contend on its own CPU or memory resources.
    drive_registered_ticket(ticket, cancelled, &mut try_acquire, Some(preparation))
}

/// Consume a PENDING pre-exec owner with exactly one nonblocking planning
/// callback. The preparation flocks and registry record remain continuous
/// until an acquired exact payload has replaced them in the same slot; a miss
/// removes the record synchronously and never enters the queue protocol.
pub(crate) fn try_activate_pending_once<T>(
    mut pending: PendingAdmission,
    attempt: impl FnOnce(&PendingOneShotProbe<'_>) -> Result<Option<(ClaimSet, T)>>,
) -> Result<Option<Acquired<T>>> {
    let (mut ticket, preparation, pending_claim) = pending.take_parts()?;
    anyhow::ensure!(
        !preparation.affinity_constrained,
        "preparation affinity must be restored before one-shot activation",
    );
    let reusable_permits = preparation.clone_permit_fds()?;
    let activated = ticket.try_activate_pending_once(&pending_claim, move |registry| {
        let result = {
            let probe = PendingOneShotProbe {
                registry,
                reusable_permits: &reusable_permits,
            };
            attempt(&probe)
        };
        // The acquired payload owns clones of every reused OFD it needs. Drop
        // both preparation copies while registry EX still excludes new
        // publications, before the record is promoted to exact HELD.
        drop(reusable_permits);
        drop(preparation);
        result
    })?;
    Ok(match activated {
        registry::PendingOneShotResult::Acquired(value, held) => {
            Some(Acquired::tracked(value, held))
        }
        registry::PendingOneShotResult::Unavailable => None,
    })
}

fn drive_registered_ticket<T>(
    mut ticket: registry::Ticket,
    cancelled: Option<&AtomicBool>,
    try_acquire: &mut impl FnMut(&mut GrantedProbe) -> Result<Option<T>>,
    mut preparation: Option<super::PreparationPermit>,
) -> Result<TicketWork<T>> {
    let stagger = Duration::from_millis((std::process::id() as u64 * 37) % 1000);
    loop {
        super::tick_reservation_wait_progress();
        check_interrupted(cancelled)?;
        match check_result(
            ticket.state_or_wait(WAITER_CRASH_RECOVERY_BASE + stagger, cancelled),
            cancelled,
        )? {
            registry::State::Coordinator => {
                return Ok(TicketWork::Coordinator(Box::new(CoordinatorTicket {
                    ticket,
                    preparation,
                    preparation_watch: None,
                })));
            }
            registry::State::Granted | registry::State::Replan => {
                let reusable_permits = preparation
                    .as_ref()
                    .map(super::PreparationPermit::clone_permit_fds)
                    .transpose()?
                    .unwrap_or_default();
                let result = ticket.run_granted(
                    cancelled,
                    |designated, watch, acquisition_allowed, predecessors, availability| {
                        let mut probe = GrantedProbe {
                            designated: designated.clone(),
                            watch: watch.clone(),
                            next_claim: designated.clone(),
                            acquisition_allowed,
                            contention: None,
                            predecessors,
                            availability,
                            reusable_permits,
                        };
                        let acquired = try_acquire(&mut probe)?;
                        Ok(registry::GrantAttempt {
                            acquired,
                            preparation_claim: None,
                            preparation_contention: None,
                            next_claim: probe.next_claim,
                            contention: probe.contention,
                        })
                    },
                );
                let result = match result {
                    Ok(registry::GrantResult::Acquired(acquired, held)) => {
                        // Removing the ticket while publishing the acquired
                        // real flocks is the terminal commit point.
                        // Cancellation that arrives after it must not turn
                        // success into an unwind that drops those flocks behind
                        // the sole observer.
                        let acquired = Acquired::tracked(acquired, held);
                        drop(preparation.take());
                        return Ok(TicketWork::Acquired(acquired));
                    }
                    Ok(registry::GrantResult::Prepared(_, _)) => {
                        unreachable!("ordinary run acquisition committed preparation ownership")
                    }
                    result => result,
                };
                let result = check_result(result, cancelled)?;
                match result {
                    registry::GrantResult::Acquired(_, _) => {
                        unreachable!("terminal acquisition returned above")
                    }
                    registry::GrantResult::Prepared(_, _) => {
                        unreachable!("prepared result returned through ordinary ticket drive")
                    }
                    registry::GrantResult::Requeued | registry::GrantResult::LostGrant => continue,
                }
            }
            registry::State::Waiting => {}
            registry::State::CoordinatorStandby => {
                anyhow::bail!(
                    "new queue ticket entered coordinator standby before owning the coordinator loop"
                );
            }
        }
    }
}

/// One coordinator's live planning snapshot and exact contention evidence.
///
/// Physical resource probes are deliberately attempt-local: success transfers
/// the complete fd set to the caller, and failure drops the complete prefix.
#[derive(Default)]
pub(crate) struct HeldLocks {
    contention: ContentionSet,
    preparation_contention: ContentionSet,
    designation: Option<ClaimSet>,
    watch: Option<ClaimSet>,
    predecessors: Option<registry::AggregateSnapshot>,
    availability: Option<registry::AvailabilitySnapshot>,
    commit_token: Option<registry::CoordinatorCommitToken>,
    preparation: Option<super::PreparationPermit>,
}

impl HeldLocks {
    fn install_schedule_snapshot(&mut self, snapshot: &registry::ScheduleSnapshot) {
        self.designation = Some(snapshot.candidate_claim.clone());
        self.watch = Some(snapshot.candidate_watch.clone());
        self.predecessors = Some(snapshot.predecessors.clone());
        self.availability = Some(snapshot.availability.clone());
        self.commit_token = Some(snapshot.commit_token);
    }

    /// Whether one complete alternative is ready according to the same
    /// mode-aware registry snapshot used by granted waiter callbacks.
    ///
    /// Predecessor reservations remain an independent hard fence.
    pub(crate) fn candidate_ready(&self, candidate: &ClaimSet) -> Result<bool> {
        let watch = self
            .watch
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("coordinator availability snapshot is missing"))?;
        let predecessors = self
            .predecessors
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("coordinator predecessor snapshot is missing"))?;
        let availability = self
            .availability
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("coordinator availability snapshot is missing"))?;
        registry::validate_claim_within_watch(candidate, watch)?;
        if predecessors.conflicts(candidate)? {
            return Ok(false);
        }

        availability.allows(candidate)
    }

    fn designated(&self) -> Result<&ClaimSet> {
        self.designation
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("coordinator designation snapshot is missing"))
    }

    pub(crate) fn candidate_holder_pressure(&self, candidate: &ClaimSet) -> Result<usize> {
        let predecessors = self
            .predecessors
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("coordinator predecessor snapshot is missing"))?;
        let cpu = candidate.cpus.iter().try_fold(0usize, |total, &cpu| {
            Ok::<_, anyhow::Error>(total.saturating_add(predecessors.cpu_holder_count(cpu)?))
        })?;
        candidate.llcs.iter().try_fold(cpu, |total, &llc| {
            Ok::<_, anyhow::Error>(total.saturating_add(predecessors.llc_holder_count(llc)?))
        })
    }

    /// Retain an exact writable witness from a physical probe performed by a
    /// helper outside the ordinary topology-lock target. Preparation tokens
    /// use the same queue blocker machinery, so their close event can park and
    /// wake the coordinator without a polling loop.
    fn record_external_contention(&mut self, evidence: ContentionEvidence) {
        self.preparation_contention.insert(evidence);
    }

    pub(crate) fn clone_reusable_permits(&self) -> Result<Vec<(usize, OwnedFd)>> {
        self.preparation
            .as_ref()
            .map(super::PreparationPermit::clone_permit_fds)
            .transpose()
            .map(Option::unwrap_or_default)
    }

    /// Probe one exact physical target only when the current mode-aware
    /// predecessor and availability snapshots license its claim.
    pub(crate) fn probe_complete_if_ready(
        &mut self,
        claim: &ClaimSet,
        target: &[ResourceLock],
    ) -> Result<Option<Vec<OwnedFd>>> {
        validate_probe_target(claim, target)?;
        if !self.candidate_ready(claim)? {
            return Ok(None);
        }
        self.probe_complete_inner(target, true)
    }

    /// Probe default mode's opportunistic exact placement. Its published
    /// registry claim is deliberately CPU-SH so later default/no-perf work may
    /// overlap it, while this one transient probe asks the physical CPU flock
    /// for EX to establish that the placement was unshared at selection time.
    /// Keep this exception explicit rather than weakening target validation
    /// for every coordinator path.
    pub(crate) fn probe_default_exact_if_ready(
        &mut self,
        published_claim: &ClaimSet,
        target: &[ResourceLock],
    ) -> Result<Option<Vec<OwnedFd>>> {
        anyhow::ensure!(
            published_claim.cpu_mode == ClaimMode::Shared,
            "default exact probe requires a shared published CPU claim",
        );
        anyhow::ensure!(
            published_claim.llc_mode == ClaimMode::Shared,
            "default exact probe requires a shared published LLC claim",
        );
        anyhow::ensure!(
            published_claim.permit_mode == ClaimMode::Exclusive,
            "default exact probe requires exclusive weighted permits",
        );
        // The complete published CPU-SH footprint includes service headroom.
        // Only mapped vCPUs are transiently probed EX; normalize those modes
        // to SH solely for exact resource-set/path validation.
        let mut published_target = target.to_vec();
        for lock in &mut published_target {
            if matches!(lock.resource, ResourceKey::Cpu(_)) {
                anyhow::ensure!(
                    matches!(lock.mode, FlockMode::Shared | FlockMode::Exclusive),
                    "default exact probe target has an invalid CPU flock mode",
                );
                lock.mode = FlockMode::Shared;
            }
        }
        anyhow::ensure!(
            target.iter().any(|lock| {
                matches!(lock.resource, ResourceKey::Cpu(_)) && lock.mode == FlockMode::Exclusive
            }),
            "default exact probe requires at least one exclusive mapped vCPU",
        );
        validate_probe_target(published_claim, &published_target)?;
        if !self.candidate_ready(published_claim)? {
            return Ok(None);
        }
        self.probe_complete_inner(target, false)
    }

    fn commit_token(&self) -> Result<registry::CoordinatorCommitToken> {
        self.commit_token
            .ok_or_else(|| anyhow::anyhow!("coordinator commit snapshot is missing"))
    }

    /// All-or-nothing nonblocking probe of `candidate`.
    ///
    /// On success the complete fd set is returned in candidate order. On
    /// failure every fd acquired by this attempt is dropped before returning,
    /// while exact blocker evidence remains live until the registry update.
    #[cfg(test)]
    pub(crate) fn probe_complete(
        &mut self,
        candidate: &[ResourceLock],
    ) -> Result<Option<Vec<OwnedFd>>> {
        self.probe_complete_inner(candidate, true)
    }

    fn probe_complete_inner(
        &mut self,
        candidate: &[ResourceLock],
        retain_contention: bool,
    ) -> Result<Option<Vec<OwnedFd>>> {
        let mut fresh = Vec::with_capacity(candidate.len());
        let mut reusable = self
            .clone_reusable_permits()?
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        for lock in candidate {
            if let ResourceKey::Permit(permit) = lock.resource
                && let Some(fd) = reusable.remove(&permit)
            {
                fresh.push(fd);
                continue;
            }
            match try_flock_with_witness(&lock.path, lock.mode)? {
                TryFlockOutcome::Acquired(fd) => fresh.push(fd),
                TryFlockOutcome::Contended(witness) => {
                    drop(fresh);
                    let evidence = ContentionEvidence {
                        blocker: lock.resource,
                        mode: lock.mode,
                        _witness: witness,
                    };
                    if retain_contention {
                        self.record_contention(evidence);
                    }
                    return Ok(None);
                }
            }
        }
        Ok(Some(fresh))
    }

    fn record_contention(&mut self, evidence: ContentionEvidence) {
        self.contention.insert(evidence);
    }

    #[cfg(test)]
    pub(crate) fn contention_markers_for_tests(&self) -> Vec<ContentionMarker> {
        self.contention.marker_vec()
    }

    fn take_contention(&mut self) -> ContentionSet {
        std::mem::take(&mut self.contention)
    }

    fn take_preparation_contention(&mut self) -> ContentionSet {
        std::mem::take(&mut self.preparation_contention)
    }
}

fn validate_probe_target(claim: &ClaimSet, target: &[ResourceLock]) -> Result<()> {
    let mut cpus = BTreeSet::new();
    let mut llcs = BTreeSet::new();
    let mut permits = BTreeSet::new();
    for lock in target {
        match lock.resource {
            ResourceKey::Cpu(cpu) => {
                anyhow::ensure!(
                    cpus.insert(cpu),
                    "coordinator probe target repeats CPU {cpu}"
                );
                anyhow::ensure!(
                    ClaimMode::from(lock.mode) == claim.cpu_mode,
                    "coordinator probe target CPU {cpu} mode {:?} does not match claim mode {:?}",
                    lock.mode,
                    claim.cpu_mode,
                );
                anyhow::ensure!(
                    lock.path == super::cpu_lock_path(cpu),
                    "coordinator probe target CPU {cpu} uses noncanonical lock path {}",
                    lock.path,
                );
            }
            ResourceKey::Llc(llc) => {
                anyhow::ensure!(
                    llcs.insert(llc),
                    "coordinator probe target repeats LLC {llc}"
                );
                anyhow::ensure!(
                    ClaimMode::from(lock.mode) == claim.llc_mode,
                    "coordinator probe target LLC {llc} mode {:?} does not match claim mode {:?}",
                    lock.mode,
                    claim.llc_mode,
                );
                anyhow::ensure!(
                    lock.path == super::llc_lock_path(llc),
                    "coordinator probe target LLC {llc} uses noncanonical lock path {}",
                    lock.path,
                );
            }
            ResourceKey::Permit(permit) => {
                anyhow::ensure!(
                    permits.insert(permit),
                    "coordinator probe target repeats permit {permit}"
                );
                anyhow::ensure!(
                    lock.mode == FlockMode::Exclusive,
                    "coordinator probe target permit {permit} must use an exclusive physical flock",
                );
                anyhow::ensure!(
                    claim.permit_mode == ClaimMode::Exclusive,
                    "coordinator physical probe cannot materialize shared permit claim {permit}",
                );
                anyhow::ensure!(
                    lock.path == super::permit_lock_path(permit),
                    "coordinator probe target permit {permit} uses noncanonical lock path {}",
                    lock.path,
                );
            }
        }
    }
    anyhow::ensure!(
        cpus == claim.cpus && llcs == claim.llcs && permits == claim.permits,
        "coordinator physical probe target does not exactly match its registry claim"
    );
    Ok(())
}

/// One coordinator-loop iteration's verdict, produced by the caller's step
/// closure (which owns the path-specific planning + probing logic).
pub(in crate::vmm) enum CoordinatorStep<T> {
    /// Acquisition complete. `claim` names the exact resource fds carried by
    /// `value`, which may differ from the coordinator's previously published
    /// planning alternative.
    Complete { claim: ClaimSet, value: T },
    /// The coordinator selected this run intent and acquired one bounded
    /// physical preparation tuple. Commit publishes the physical tuple as the
    /// PENDING claim and retains the selected final footprint in its watch; it
    /// does not publish the final run claim as HELD.
    Prepare {
        final_claim: ClaimSet,
        preparation_claim: ClaimSet,
        preparation: Box<super::PreparationPermit>,
    },
    /// Still waiting. `claim` is the freshly planned target to publish
    /// before sleeping for the next release event.
    Waiting { claim: ClaimSet },
    /// The step decided acquisition cannot proceed at all (e.g. no
    /// plannable candidate remains). Terminal; not a timeout.
    Abort { reason: String },
}

/// Outcome of [`acquire_as_coordinator`].
pub(in crate::vmm) enum CoordinatorOutcome<T> {
    Acquired(Acquired<T>),
    Prepared(Box<PendingAdmission>),
    Aborted { reason: String },
}

struct HolderObserver {
    proof_files: std::collections::BTreeMap<ResourceKey, std::fs::File>,
    proof_locks: BTreeSet<ResourceKey>,
}

#[cfg(test)]
thread_local! {
    static FORCE_HOLDER_OBSERVER_UNAVAILABLE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn force_holder_observer_unavailable_for_tests() {
    FORCE_HOLDER_OBSERVER_UNAVAILABLE.with(|forced| forced.set(true));
}

#[cfg(test)]
fn forced_unavailable_for_tests() -> bool {
    FORCE_HOLDER_OBSERVER_UNAVAILABLE.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn forced_unavailable_for_tests() -> bool {
    false
}

impl HolderObserver {
    fn new() -> Self {
        Self {
            proof_files: std::collections::BTreeMap::new(),
            proof_locks: BTreeSet::new(),
        }
    }

    fn observe(
        &mut self,
        request: &registry::ObservationRequest,
    ) -> registry::AvailabilityObservation {
        self.release_proofs();
        if forced_unavailable_for_tests() {
            return registry::AvailabilityObservation::default();
        }
        match self.observe_with_proofs(request) {
            Ok(observation) => observation,
            Err(error) => {
                tracing::debug!(
                    %error,
                    "cannot prove pending reservation availability; keeping durable observation work pending"
                );
                registry::AvailabilityObservation::default()
            }
        }
    }

    fn proof_file(&mut self, key: ResourceKey) -> Result<&std::fs::File> {
        if let std::collections::btree_map::Entry::Vacant(entry) = self.proof_files.entry(key) {
            let path = match key {
                ResourceKey::Llc(index) => super::llc_lock_path(index),
                ResourceKey::Cpu(index) => super::cpu_lock_path(index),
                ResourceKey::Permit(index) => super::permit_lock_path(index),
            };
            let file = std::fs::OpenOptions::new().read(true).open(&path)?;
            entry.insert(file);
        }
        Ok(&self.proof_files[&key])
    }

    fn try_proof(&mut self, key: ResourceKey, mode: FlockMode) -> Result<bool> {
        use rustix::fs::{FlockOperation, flock};
        let operation = match mode {
            FlockMode::Exclusive => FlockOperation::NonBlockingLockExclusive,
            FlockMode::Shared => FlockOperation::NonBlockingLockShared,
        };
        match flock(self.proof_file(key)?, operation) {
            Ok(()) => {
                self.proof_locks.insert(key);
                Ok(true)
            }
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => Ok(false),
            Err(error) => Err(std::io::Error::from_raw_os_error(error.raw_os_error()).into()),
        }
    }

    fn observe_with_proofs(
        &mut self,
        request: &registry::ObservationRequest,
    ) -> Result<registry::AvailabilityObservation> {
        let mut observation = registry::AvailabilityObservation::default();
        for &cpu in request.cpus.keys() {
            let key = ResourceKey::Cpu(cpu);
            if self.try_proof(key, FlockMode::Exclusive)? {
                observation.cpus.insert(
                    cpu,
                    registry::CpuObservation {
                        availability: registry::CpuAvailability::Free,
                        sh_resolved: true,
                        ex_resolved: true,
                    },
                );
            } else if self.try_proof(key, FlockMode::Shared)? {
                observation.cpus.insert(
                    cpu,
                    registry::CpuObservation {
                        availability: registry::CpuAvailability::SharedHeld,
                        sh_resolved: true,
                        ex_resolved: true,
                    },
                );
            } else {
                observation.cpus.insert(
                    cpu,
                    registry::CpuObservation {
                        availability: registry::CpuAvailability::ExclusiveHeld,
                        sh_resolved: true,
                        ex_resolved: true,
                    },
                );
            }
        }
        for &llc in request.llcs.keys() {
            let key = ResourceKey::Llc(llc);
            if self.try_proof(key, FlockMode::Exclusive)? {
                observation.llcs.insert(
                    llc,
                    registry::LlcObservation {
                        availability: registry::LlcAvailability::Free,
                        sh_resolved: true,
                        ex_resolved: true,
                    },
                );
            } else if self.try_proof(key, FlockMode::Shared)? {
                observation.llcs.insert(
                    llc,
                    registry::LlcObservation {
                        availability: registry::LlcAvailability::SharedHeld,
                        sh_resolved: true,
                        ex_resolved: true,
                    },
                );
            } else {
                observation.llcs.insert(
                    llc,
                    registry::LlcObservation {
                        availability: registry::LlcAvailability::ExclusiveHeld,
                        sh_resolved: true,
                        ex_resolved: true,
                    },
                );
            }
        }
        for &permit in request.permits.keys() {
            let key = ResourceKey::Permit(permit);
            let availability = if self.try_proof(key, FlockMode::Exclusive)? {
                registry::CpuAvailability::Free
            } else {
                registry::CpuAvailability::ExclusiveHeld
            };
            observation.permits.insert(
                permit,
                registry::CpuObservation {
                    availability,
                    sh_resolved: true,
                    ex_resolved: true,
                },
            );
        }
        Ok(observation)
    }

    fn release_proofs(&mut self) {
        use rustix::fs::{FlockOperation, flock};
        for key in std::mem::take(&mut self.proof_locks) {
            if let Some(file) = self.proof_files.get(&key) {
                let _ = flock(file, FlockOperation::Unlock);
            }
        }
    }
}

/// Run the elected coordinator loop. The step closure re-plans from live holder
/// state after each relevant wake; the ticket's exact claim is updated before
/// compatible successors are granted.
pub(in crate::vmm) fn acquire_as_coordinator<T>(
    coordinator: impl Into<Box<CoordinatorTicket>>,
    step: impl FnMut(&mut HeldLocks) -> Result<CoordinatorStep<T>>,
) -> Result<CoordinatorOutcome<T>> {
    acquire_as_coordinator_impl(*coordinator.into(), None, step)
}

/// Cancellation-aware variant of [`acquire_as_coordinator`].
///
/// The ticket keeps its private wake registration alive across the
/// waiter-to-coordinator transition, so cancellation interrupts inotify rather
/// than waiting for the fallback tick.
pub(in crate::vmm) fn acquire_as_coordinator_interruptible<T>(
    coordinator: impl Into<Box<CoordinatorTicket>>,
    cancelled: &AtomicBool,
    step: impl FnMut(&mut HeldLocks) -> Result<CoordinatorStep<T>>,
) -> Result<CoordinatorOutcome<T>> {
    acquire_as_coordinator_impl(*coordinator.into(), Some(cancelled), step)
}

fn acquire_as_coordinator_impl<T>(
    mut coordinator: CoordinatorTicket,
    cancelled: Option<&AtomicBool>,
    mut step: impl FnMut(&mut HeldLocks) -> Result<CoordinatorStep<T>>,
) -> Result<CoordinatorOutcome<T>> {
    let _namespace = coordinator.ticket.enter_namespace();
    check_interrupted(cancelled)?;
    let watch = check_result(LockDirWatch::new(), cancelled)?;
    check_interrupted(cancelled)?;
    let mut held = HeldLocks {
        preparation: coordinator.preparation.take(),
        ..HeldLocks::default()
    };
    let preparation_watch = coordinator.preparation_watch.clone();
    let event_watch = |watch: ClaimSet| {
        preparation_watch
            .as_ref()
            .map_or(watch.clone(), |preparation| {
                watch.union_envelope(preparation)
            })
    };
    let mut watched_resources = ClaimSet::default();
    let mut observer = HolderObserver::new();
    let mut first = true;
    let mut force_step = false;
    let mut retry_due = false;
    let mut retry_deadline = std::time::Instant::now() + COORDINATOR_WAKE_FALLBACK;
    let mut wait_diagnostic_deadline = std::time::Instant::now() + WAIT_DIAGNOSTIC_INITIAL_DELAY;
    let mut pending_events = LockDirEvents::default();
    let outcome = loop {
        super::tick_reservation_wait_progress();
        let diagnostic_now = std::time::Instant::now();
        if diagnostic_now >= wait_diagnostic_deadline {
            registry::persist_wait_diagnostic_if_enabled();
            wait_diagnostic_deadline = diagnostic_now + WAIT_DIAGNOSTIC_INTERVAL;
        }
        check_interrupted(cancelled)?;
        pending_events.merge(check_result(watch.drain(&watched_resources), cancelled)?);
        let drain_backlog = pending_events.backlog;
        let closed_tickets: Vec<_> = pending_events.liveness_closes.iter().copied().collect();
        let mut snapshot = check_result(
            coordinator.ticket.schedule(
                None,
                &pending_events.cpu_closes,
                &pending_events.llc_closes,
                &pending_events.permit_closes,
                pending_events.overflow,
                &[],
                &closed_tickets,
                first || retry_due,
                first.then_some(PREWATCH_LIVENESS_RECONCILE_DELAY),
                pending_events.overflow,
                cancelled,
            ),
            cancelled,
        )?;
        retry_due = false;
        let mut liveness_deadline = std::time::Instant::now() + snapshot.liveness_due_in;
        watched_resources = event_watch(snapshot.watch.clone());
        let mut should_step = first || force_step || snapshot.should_step;
        force_step = false;
        if let Some(request) = snapshot.observation.take() {
            let observation = observer.observe(&request);
            snapshot = check_result(
                coordinator.ticket.apply_observation(
                    &request,
                    &observation,
                    || observer.release_proofs(),
                    cancelled,
                ),
                cancelled,
            )?;
            liveness_deadline = std::time::Instant::now() + snapshot.liveness_due_in;
            watched_resources = event_watch(snapshot.watch.clone());
            should_step |= snapshot.should_step;
        }
        held.install_schedule_snapshot(&snapshot);
        let observation_pending = snapshot.observation.is_some();
        pending_events = LockDirEvents::default();

        if should_step {
            let next = match step(&mut held) {
                Ok(next) => next,
                Err(error) => {
                    check_interrupted(cancelled)?;
                    return Err(error);
                }
            };
            check_interrupted(cancelled)?;
            match next {
                CoordinatorStep::Complete { claim, value } => {
                    check_interrupted(cancelled)?;
                    let contention = held.take_contention();
                    let preparation_contention = held.take_preparation_contention();
                    anyhow::ensure!(
                        preparation_contention.is_empty(),
                        "coordinator completed while retaining preparation contention",
                    );
                    let markers = contention.marker_vec();
                    let commit_token = held.commit_token()?;
                    // `finish_acquired` publishes the held flocks and removes
                    // the coordinator record atomically. Once it commits,
                    // cancellation is for the caller's next lifecycle phase—
                    // not grounds to roll back this committed acquire.
                    match coordinator.ticket.finish_acquired(
                        &claim,
                        commit_token,
                        &markers,
                        cancelled,
                    )? {
                        registry::FinishAcquireResult::Committed(publication) => {
                            drop(contention);
                            drop(preparation_contention);
                            drop(held.preparation.take());
                            break CoordinatorOutcome::Acquired(Acquired::tracked(
                                value,
                                publication,
                            ));
                        }
                        registry::FinishAcquireResult::Stale => {
                            // An earlier callback changed its reservation after
                            // this planner snapshot. Drop the stale physical fd
                            // set, retain the coordinator ticket, and force one
                            // fresh planner turn after the pending prefix scan.
                            drop(value);
                            drop(contention);
                            drop(preparation_contention);
                            first = false;
                            force_step = true;
                            continue;
                        }
                    }
                }
                CoordinatorStep::Prepare {
                    final_claim,
                    preparation_claim,
                    preparation,
                } => {
                    check_interrupted(cancelled)?;
                    let preparation_contention = held.take_preparation_contention();
                    anyhow::ensure!(
                        preparation_contention.is_empty(),
                        "coordinator prepared while retaining preparation contention",
                    );
                    let commit_token = held.commit_token()?;
                    match coordinator.ticket.finish_preparation(
                        &final_claim,
                        &preparation_claim,
                        commit_token,
                        cancelled,
                    )? {
                        registry::FinishPreparationResult::Committed(pending_claim) => {
                            drop(held.take_contention());
                            drop(preparation_contention);
                            drop(held.preparation.take());
                            let pending = pending_admission_from_parts(
                                coordinator.ticket,
                                *preparation,
                                pending_claim,
                            )?;
                            break CoordinatorOutcome::Prepared(Box::new(pending));
                        }
                        registry::FinishPreparationResult::Stale => {
                            drop(preparation);
                            drop(held.take_contention());
                            drop(preparation_contention);
                            first = false;
                            force_step = true;
                            continue;
                        }
                    }
                }
                CoordinatorStep::Abort { reason } => {
                    break CoordinatorOutcome::Aborted { reason };
                }
                CoordinatorStep::Waiting { claim } => {
                    if claim.is_empty() {
                        return Err(anyhow::anyhow!(
                            "registry coordinator produced an empty priority claim"
                        ));
                    }
                    // Waiting publishes the exact reservation while every
                    // incomplete physical probe has already released its fds.
                    let contention = held.take_contention();
                    let markers = contention.marker_vec();
                    let preparation_contention = held.take_preparation_contention();
                    let preparation_markers = preparation_contention.marker_vec();
                    let snapshot = check_result(
                        coordinator.ticket.schedule(
                            Some(&claim),
                            &BTreeSet::new(),
                            &BTreeSet::new(),
                            &BTreeSet::new(),
                            false,
                            &markers,
                            &closed_tickets,
                            false,
                            None,
                            false,
                            cancelled,
                        ),
                        cancelled,
                    )?;
                    drop(contention);
                    if !preparation_markers.is_empty() {
                        check_result(
                            coordinator
                                .ticket
                                .mark_external_contention(&preparation_markers, cancelled),
                            cancelled,
                        )?;
                        // Publish the event-only blocker before closing its
                        // writable witness. The next turn observes current
                        // holder state once, then sleeps until a real release
                        // instead of immediately re-probing in a hot loop.
                        drop(preparation_contention);
                        first = false;
                        pending_events = LockDirEvents::default();
                        continue;
                    }
                    drop(preparation_contention);
                    let observe_before_sleep = snapshot.observation.is_some();
                    let retry_before_sleep = waiting_publication_requires_immediate_turn(
                        snapshot.should_step,
                        observe_before_sleep,
                    );
                    liveness_deadline = std::time::Instant::now() + snapshot.liveness_due_in;
                    watched_resources = event_watch(snapshot.watch);
                    if retry_before_sleep {
                        // A predecessor can release after this coordinator's
                        // planner callback but before the WAITING publication
                        // above takes the registry fence. That schedule turn
                        // consumes PENDING_RESCAN and refreshes the cached
                        // prefix. Preserve its one-shot progress signal instead
                        // of sleeping after the notification was already
                        // consumed. The next outer turn installs a coherent
                        // snapshot before invoking the planner again.
                        force_step |= snapshot.should_step;
                        first = false;
                        pending_events = LockDirEvents::default();
                        continue;
                    }
                }
            }
        }
        first = false;
        check_interrupted(cancelled)?;
        if drain_backlog {
            // Fairly interleave event consumption with registry/grant work.
            // This path never sleeps while the kernel queue may still contain
            // a bounded-drain remainder.
            continue;
        }
        let retry_interval = watch.semantic_retry_interval(observation_pending);
        let now = std::time::Instant::now();
        retry_deadline = retry_deadline.min(now + retry_interval);
        let wake_deadline = retry_deadline.min(liveness_deadline);
        loop {
            let wait_now = std::time::Instant::now();
            let semantic_wait = wake_deadline.saturating_duration_since(wait_now);
            let syscall_wait = super::reservation_wait_progress_poll()
                .map_or(semantic_wait, |poll| semantic_wait.min(poll));
            match check_result(watch.wait(syscall_wait, &watched_resources), cancelled)? {
                Some(events) => {
                    pending_events.merge(events);
                    break;
                }
                None => {
                    super::tick_reservation_wait_progress();
                    if std::time::Instant::now() < wake_deadline {
                        // This was only a synchronous progress slice. Remain
                        // inside the same semantic watch wait: do not take the
                        // registry lock or run another schedule pass.
                        continue;
                    }
                    // A global liveness deadline is persisted in the registry,
                    // so rapid coordinator handoff cannot postpone it. If that
                    // deadline alone woke us, the next schedule performs the
                    // due sweep without also manufacturing a whole-watch retry.
                    retry_due = liveness_deadline >= retry_deadline;
                    if retry_due {
                        retry_deadline = std::time::Instant::now() + retry_interval;
                    }
                    break;
                }
            }
        }
    };
    Ok(outcome)
}

fn waiting_publication_requires_immediate_turn(
    should_step: bool,
    observation_pending: bool,
) -> bool {
    should_step || observation_pending
}

/// Canonical global lock order for a resource set: LLC locks by
/// ascending index, then CPU locks by ascending index. Every
/// acquirer — fast path and coordinator alike — walks locks in this order,
/// keeping each all-or-nothing probe deterministic and preventing overlapping
/// attempts from half-blocking each other.
#[cfg(test)]
pub(crate) fn canonical_lock_order(
    llc_indices: &[usize],
    llc_mode: FlockMode,
    cpus: &[usize],
) -> Vec<ResourceLock> {
    canonical_lock_order_with_modes(llc_indices, llc_mode, cpus, FlockMode::Exclusive)
}

pub(crate) fn canonical_lock_order_with_modes(
    llc_indices: &[usize],
    llc_mode: FlockMode,
    cpus: &[usize],
    cpu_mode: FlockMode,
) -> Vec<ResourceLock> {
    canonical_lock_order_with_permits(llc_indices, llc_mode, cpus, cpu_mode, &[])
}

pub(crate) fn canonical_lock_order_with_permits(
    llc_indices: &[usize],
    llc_mode: FlockMode,
    cpus: &[usize],
    cpu_mode: FlockMode,
    permits: &[usize],
) -> Vec<ResourceLock> {
    let mut llcs: Vec<usize> = llc_indices.to_vec();
    llcs.sort_unstable();
    llcs.dedup();
    let mut cpu_sorted: Vec<usize> = cpus.to_vec();
    cpu_sorted.sort_unstable();
    cpu_sorted.dedup();
    let mut permit_sorted = permits.to_vec();
    permit_sorted.sort_unstable();
    permit_sorted.dedup();
    let mut out = Vec::with_capacity(llcs.len() + cpu_sorted.len() + permit_sorted.len());
    for idx in llcs {
        out.push(ResourceLock {
            path: super::llc_lock_path(idx),
            mode: llc_mode,
            resource: ResourceKey::Llc(idx),
        });
    }
    for cpu in cpu_sorted {
        out.push(ResourceLock {
            path: super::cpu_lock_path(cpu),
            mode: cpu_mode,
            resource: ResourceKey::Cpu(cpu),
        });
    }
    for permit in permit_sorted {
        out.push(ResourceLock {
            path: super::permit_lock_path(permit),
            mode: FlockMode::Exclusive,
            resource: ResourceKey::Permit(permit),
        });
    }
    out
}
