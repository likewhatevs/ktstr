//! Cross-process host-resource admission under nextest.
//!
//! Every ktstr process sharing a lock directory participates in one v6
//! fixed-record mmap registry. A ticket publishes one exact, non-empty CPU/LLC
//! reservation claim plus the resources its planner may watch. Claims preserve
//! the resource-lock semantics exactly: CPU and LLC claims independently use
//! the same SH/EX compatibility matrix as `flock`.
//!
//! Admission is work-conserving without weakening those reservations. One
//! elected coordinator scans tickets in monotonic order and grants a waiter
//! when its exact claim is compatible with every earlier live claim. Thus an
//! incompatible predecessor remains a hard fence, while fully disjoint work can
//! pass it. Only the coordinator may retain partial resource locks while
//! waiting, which preserves the no-deadlock invariant.
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

mod registry;

/// Stable identity of one host reservation lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ResourceKey {
    Llc(usize),
    Cpu(usize),
}

/// One canonical host-resource lock together with the exact registry identity
/// it represents. Keeping the identity attached avoids reparsing test-specific
/// lock paths and lets partial coordinator holds publish mode-correct state.
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
/// Bound the watch-install handoff gap without making every short-lived
/// coordinator scan the full registry. The first coordinator that remains
/// active past this shared, non-postponable deadline performs one sweep;
/// ordinary liveness closes after watch installation remain event-driven.
const PREWATCH_LIVENESS_RECONCILE_DELAY: Duration = Duration::from_millis(500);
/// Explicit cfg(test)-only retry transport cadence. This is a semantic
/// coordinator retry deadline, not a short slice of the 30-second real-inotify
/// deadline: each expiry runs schedule + replan again.
#[cfg(test)]
const TEST_RETRY_WAKE_INTERVAL: Duration = Duration::from_millis(8);
#[cfg(test)]
const TEST_RETRY_WAKE_MARKER: &str = ".ktstr-test-retry-wake";

/// Directory the protocol files live in — derived from the LLC
/// lockfile path so the test-only lock-prefix override isolates the
/// v6 registry files into the same per-test tempdir as the LLC locks
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
    pub llc_mode: ClaimMode,
    pub cpu_mode: ClaimMode,
}

impl ClaimSet {
    fn from_claim_modes(
        llcs: impl IntoIterator<Item = usize>,
        cpus: impl IntoIterator<Item = usize>,
        llc_mode: ClaimMode,
        cpu_mode: ClaimMode,
    ) -> Self {
        let llcs = llcs.into_iter().collect::<BTreeSet<_>>();
        let cpus = cpus.into_iter().collect::<BTreeSet<_>>();
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
            llcs,
            cpus,
        }
    }

    pub(super) fn with_claim_modes(
        llcs: impl IntoIterator<Item = usize>,
        cpus: impl IntoIterator<Item = usize>,
        llc_mode: ClaimMode,
        cpu_mode: ClaimMode,
    ) -> Self {
        Self::from_claim_modes(llcs, cpus, llc_mode, cpu_mode)
    }

    pub(crate) fn new(
        llcs: impl IntoIterator<Item = usize>,
        cpus: impl IntoIterator<Item = usize>,
        llc_mode: FlockMode,
    ) -> Self {
        Self::from_claim_modes(llcs, cpus, llc_mode.into(), ClaimMode::Exclusive)
    }

    pub(crate) fn with_modes(
        llcs: impl IntoIterator<Item = usize>,
        cpus: impl IntoIterator<Item = usize>,
        llc_mode: FlockMode,
        cpu_mode: FlockMode,
    ) -> Self {
        Self::from_claim_modes(llcs, cpus, llc_mode.into(), cpu_mode.into())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.llcs.is_empty() && self.cpus.is_empty()
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
        Self::from_claim_modes(
            self.llcs.union(&other.llcs).copied(),
            self.cpus.union(&other.cpus).copied(),
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
        )
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
}

/// Copy the aggregate reservation bitsets once for a whole planning pass.
/// Candidate filtering after this call is process-local; the final selected
/// claim still enters [`with_registry_fence`] for authoritative admission.
pub(crate) fn registered_claim_snapshot(required: &ClaimSet) -> Result<RegisteredClaimSnapshot> {
    Ok(RegisteredClaimSnapshot {
        inner: registry::aggregate_snapshot(required)?,
    })
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
    Ran { value: T, watched: bool },
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
}

#[derive(Default)]
pub(crate) struct LockDirEvents {
    llc_closes: BTreeSet<usize>,
    cpu_closes: BTreeSet<usize>,
    registry_notify: bool,
    liveness_closes: BTreeSet<(u64, u64)>,
    overflow: bool,
}

impl LockDirEvents {
    fn merge(&mut self, other: Self) {
        self.llc_closes.extend(other.llc_closes);
        self.cpu_closes.extend(other.cpu_closes);
        self.registry_notify |= other.registry_notify;
        self.liveness_closes.extend(other.liveness_closes);
        self.overflow |= other.overflow;
    }

    fn is_actionable(&self) -> bool {
        !self.llc_closes.is_empty()
            || !self.cpu_closes.is_empty()
            || self.registry_notify
            || !self.liveness_closes.is_empty()
            || self.overflow
    }

    #[cfg(test)]
    pub(crate) fn contains_liveness(&self, identity: (u64, u64)) -> bool {
        self.liveness_closes.contains(&identity)
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
        let event_dir = registry::event_dir();
        let llc_dir = Path::new(&llc_path)
            .parent()
            .ok_or_else(|| anyhow::anyhow!("LLC resource lock path has no parent: {llc_path}"))?
            .to_path_buf();
        let cpu_dir = Path::new(&cpu_path)
            .parent()
            .ok_or_else(|| anyhow::anyhow!("CPU resource lock path has no parent: {cpu_path}"))?
            .to_path_buf();
        if event_dir == llc_dir || event_dir == cpu_dir {
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

    /// Drain any queued events without blocking, discarding them.
    /// The coordinator calls this right before sleeping so its own
    /// open/close churn from the attempt it just made does not wake
    /// it straight back up (a self-wake busy loop). An external
    /// release slipping into the drain-to-sleep gap is caught by the
    /// [`COORDINATOR_WAKE_FALLBACK`] tick.
    pub(crate) fn drain(&self, watched: &ClaimSet) -> Result<LockDirEvents> {
        let mut batch = LockDirEvents::default();
        loop {
            match self.ino.read_events() {
                Ok(events) if events.is_empty() => break,
                Ok(events) => self.classify(events, watched, &mut batch),
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
            let events = match self.ino.read_events() {
                Ok(events) => events,
                Err(nix::errno::Errno::EAGAIN) => continue,
                Err(error) => return Err(error.into()),
            };
            let mut batch = LockDirEvents::default();
            self.classify(events, watched, &mut batch);
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
    protocol_dir().join(TEST_RETRY_WAKE_MARKER)
}

#[cfg(test)]
pub(crate) fn test_retry_wake_marker_path_for_tests() -> PathBuf {
    test_retry_wake_marker_path()
}

/// Publish mode-correct holder state while the corresponding real flocks are
/// still live. This is an authoritative state transition, not an event hint.
pub(crate) fn publish_acquired(claim: &ClaimSet) -> Result<()> {
    registry::publish_acquired(claim)
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
                });
        Ok(!just_contended
            && !self.predecessors.conflicts(candidate)?
            && self.availability.allows(candidate)?)
    }

    pub(crate) fn try_acquire<T, O: IntoProbeOutcome<T>>(
        &mut self,
        candidate: &ClaimSet,
        acquire: impl FnOnce() -> Result<O>,
    ) -> Result<Option<T>> {
        if !self.acquisition_allowed || !self.allows(candidate) {
            return Ok(None);
        }
        match acquire()?.into_probe_outcome() {
            ProbeOutcome::Acquired(value) => Ok(Some(value)),
            ProbeOutcome::Contended(evidence) => {
                self.contention = Some(evidence);
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
}

#[cfg(test)]
impl CoordinatorTicket {
    pub(crate) fn read_state_shared_for_tests(&self) -> Result<()> {
        self.ticket.read_state_shared_for_tests()
    }
}

pub(crate) enum TicketWork<T> {
    Acquired(T),
    Coordinator(CoordinatorTicket),
}

#[cfg(test)]
pub(crate) fn ticket_registry_snapshot_for_tests() -> Result<Vec<(u64, u32, ClaimSet)>> {
    registry::snapshot()
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
pub(crate) fn exercise_llc_sh_only_shared_to_free_close_for_tests() -> Result<(bool, bool, u64, u64)>
{
    registry::exercise_llc_sh_only_shared_to_free_close_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_busy_to_free_close_for_tests() -> Result<(usize, u64, usize)> {
    registry::exercise_busy_to_free_close_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_llc_ex_contention_shared_wake_for_tests() -> Result<(u64, bool, bool)> {
    registry::exercise_llc_ex_contention_shared_wake_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_cpu_ex_contention_shared_wake_for_tests()
-> Result<(u64, bool, bool, bool, bool, bool, bool)> {
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
pub(crate) fn exercise_prefix_callback_scaling_for_tests(
    waiters: usize,
) -> Result<(usize, usize, usize)> {
    registry::exercise_prefix_callback_scaling_for_tests(waiters)
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
pub(crate) fn exercise_issue_serial_race_for_tests() -> Result<(bool, bool, bool, bool)> {
    registry::exercise_issue_serial_race_for_tests()
}

#[cfg(test)]
pub(crate) fn exercise_candidate_ready_matrix_for_tests() -> Result<()> {
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
pub(crate) fn claim_from_resource_modes_for_tests(
    resources: impl IntoIterator<Item = (ResourceKey, FlockMode)>,
) -> Result<ClaimSet> {
    claim_from_resource_modes(resources.into_iter().collect())
}

#[cfg(test)]
pub(crate) fn registry_initializer_temp_count_for_tests() -> Result<usize> {
    registry::initializer_temp_count_for_tests()
}

#[cfg(test)]
pub(crate) fn hold_registry_shared_for_tests() -> Result<OwnedFd> {
    registry::hold_registry_shared_for_tests()
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
    // Unmap every non-coordinator futex before its slot reaches the freelist;
    // release the coordinator last so the test does not manufacture 499
    // leadership handoffs while checking chunk growth.
    while tickets.len() > 1 {
        tickets.pop();
    }
    drop(tickets);
    Ok(snapshot.len())
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
    register_ticket_or_acquire_impl(
        initial_claim,
        watch_claim,
        None,
        None::<fn()>,
        cancelled,
        try_acquire,
    )
}

/// Register the ticket, then run `after_publish` before inspecting its state or
/// invoking an acquisition callback. The hook is the acquire-before-release
/// handoff point for reservations inherited from an earlier phase: the new
/// exact record is already durable, so releasing the old flocks cannot create
/// an unclaimed interval.
pub(crate) fn register_ticket_or_acquire_after_publish<T>(
    initial_claim: ClaimSet,
    watch_claim: ClaimSet,
    cancelled: Option<&AtomicBool>,
    after_publish: impl FnOnce(),
    try_acquire: impl FnMut(&mut GrantedProbe) -> Result<Option<T>>,
) -> Result<TicketWork<T>> {
    register_ticket_or_acquire_impl(
        initial_claim,
        watch_claim,
        None,
        Some(after_publish),
        cancelled,
        try_acquire,
    )
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
        None::<fn()>,
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
        None::<fn()>,
        cancelled,
        try_acquire,
    )
}

pub(crate) fn register_ticket_after_contentions_and_publish<T>(
    initial_claim: ClaimSet,
    watch_claim: ClaimSet,
    contention: ContentionSet,
    cancelled: Option<&AtomicBool>,
    after_publish: impl FnOnce(),
    try_acquire: impl FnMut(&mut GrantedProbe) -> Result<Option<T>>,
) -> Result<TicketWork<T>> {
    register_ticket_or_acquire_impl(
        initial_claim,
        watch_claim,
        Some(contention),
        Some(after_publish),
        cancelled,
        try_acquire,
    )
}

fn register_ticket_or_acquire_impl<T, H>(
    initial_claim: ClaimSet,
    watch_claim: ClaimSet,
    initial_contention: Option<ContentionSet>,
    after_publish: Option<H>,
    cancelled: Option<&AtomicBool>,
    mut try_acquire: impl FnMut(&mut GrantedProbe) -> Result<Option<T>>,
) -> Result<TicketWork<T>>
where
    H: FnOnce(),
{
    check_interrupted(cancelled)?;
    let mut ticket = check_result(
        registry::Ticket::register_after_contention(
            initial_claim,
            watch_claim,
            initial_contention,
            cancelled,
        ),
        cancelled,
    )?;
    if let Some(after_publish) = after_publish {
        after_publish();
    }
    let stagger = Duration::from_millis((std::process::id() as u64 * 37) % 1000);
    loop {
        super::tick_reservation_wait_progress();
        check_interrupted(cancelled)?;
        match check_result(
            ticket.state_or_wait(WAITER_CRASH_RECOVERY_BASE + stagger, cancelled),
            cancelled,
        )? {
            registry::State::Coordinator => {
                return Ok(TicketWork::Coordinator(CoordinatorTicket { ticket }));
            }
            registry::State::Granted | registry::State::Replan => {
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
                        };
                        let acquired = try_acquire(&mut probe)?;
                        Ok(registry::GrantAttempt {
                            acquired,
                            next_claim: probe.next_claim,
                            contention: probe.contention,
                        })
                    },
                );
                let result = match result {
                    Ok(registry::GrantResult::Acquired(acquired)) => {
                        // Removing the ticket while publishing the acquired
                        // real flocks is the terminal commit point.
                        // Cancellation that arrives after it must not turn
                        // success into an unwind that drops those flocks behind
                        // the sole observer.
                        return Ok(TicketWork::Acquired(acquired));
                    }
                    result => result,
                };
                let result = check_result(result, cancelled)?;
                match result {
                    registry::GrantResult::Acquired(_) => {
                        unreachable!("terminal acquisition returned above")
                    }
                    registry::GrantResult::Requeued | registry::GrantResult::LostGrant => continue,
                }
            }
            registry::State::Waiting => {}
        }
    }
}

/// The coordinator's accumulated partial holds, keyed by lockfile path.
/// Only the registry-elected coordinator may retain these while waiting.
#[derive(Default)]
pub(crate) struct HeldLocks {
    map: std::collections::BTreeMap<String, HeldLock>,
    newly_held: std::collections::BTreeMap<ResourceKey, FlockMode>,
    abandoned_resources: std::collections::BTreeMap<ResourceKey, FlockMode>,
    abandoned_fds: Vec<HeldLock>,
    contention: ContentionSet,
    watch: Option<ClaimSet>,
    predecessors: Option<registry::AggregateSnapshot>,
    availability: Option<registry::AvailabilitySnapshot>,
}

struct HeldLock {
    fd: OwnedFd,
    mode: FlockMode,
    resource: ResourceKey,
}

struct HeldRegistryUpdate {
    newly_held: ClaimSet,
    abandoned: ClaimSet,
    contention: ContentionSet,
    // These descriptors intentionally outlive the registry transaction that
    // marks their resources UNKNOWN. Dropping them sooner recreates the
    // close-before-publication lost-wake race.
    _abandoned_fds: Vec<HeldLock>,
}

impl HeldLocks {
    fn install_schedule_snapshot(&mut self, snapshot: &registry::ScheduleSnapshot) {
        self.watch = Some(snapshot.candidate_watch.clone());
        self.predecessors = Some(snapshot.predecessors.clone());
        self.availability = Some(snapshot.availability.clone());
    }

    /// Whether one complete alternative is ready according to the same
    /// mode-aware registry snapshot used by granted waiter callbacks.
    ///
    /// Exact coordinator-held locks are removed from the live-holder query:
    /// once published they correctly appear unavailable globally, but they are
    /// already usable by this coordinator. Predecessor reservations remain an
    /// independent hard fence.
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

        let mut unheld = candidate.clone();
        unheld.cpus.retain(|&cpu| {
            !self.map.values().any(|held| {
                held.resource == ResourceKey::Cpu(cpu)
                    && held.mode
                        == match candidate.cpu_mode {
                            ClaimMode::Exclusive => FlockMode::Exclusive,
                            ClaimMode::Shared => FlockMode::Shared,
                        }
            })
        });
        unheld.llcs.retain(|&llc| {
            !self.map.values().any(|held| {
                held.resource == ResourceKey::Llc(llc)
                    && held.mode
                        == match candidate.llc_mode {
                            ClaimMode::Exclusive => FlockMode::Exclusive,
                            ClaimMode::Shared => FlockMode::Shared,
                        }
            })
        });
        if unheld.is_empty() {
            Ok(true)
        } else {
            availability.allows(&unheld)
        }
    }

    /// Stage every partial not present with the exact same mode in `target`.
    /// Staged fds remain locked until the registry publishes the corresponding
    /// UNKNOWN transition.
    pub(crate) fn retain_target(&mut self, target: &[ResourceLock]) {
        let keep: std::collections::BTreeMap<_, _> = target
            .iter()
            .map(|lock| (lock.path.as_str(), lock.mode))
            .collect();
        let abandoned: Vec<_> = self
            .map
            .iter()
            .filter(|(path, held)| keep.get(path.as_str()).copied() != Some(held.mode))
            .map(|(path, _)| path.clone())
            .collect();
        for path in abandoned {
            let held = self
                .map
                .remove(&path)
                .expect("staged coordinator hold disappeared");
            self.newly_held.remove(&held.resource);
            self.abandoned_resources.insert(held.resource, held.mode);
            self.abandoned_fds.push(held);
        }
    }

    /// Non-blocking sweep: acquire, in the given (canonical) order,
    /// every lock in `target` not already held. Returns how many NEW
    /// locks were gained. Never blocks — the coordinator's waiting happens
    /// on the inotify watch, not inside flock.
    pub(crate) fn sweep(&mut self, target: &[ResourceLock]) -> Result<usize> {
        let mut gained = 0;
        for lock in target {
            if self
                .map
                .get(&lock.path)
                .is_some_and(|held| held.mode == lock.mode)
            {
                continue;
            }
            // A differently-mode-held path cannot be upgraded while its old
            // open-file description remains locked. The next retain_target
            // publishes and releases it before a later sweep retries.
            let std::collections::btree_map::Entry::Vacant(entry) =
                self.map.entry(lock.path.clone())
            else {
                continue;
            };
            match try_flock_with_witness(&lock.path, lock.mode)? {
                TryFlockOutcome::Acquired(fd) => {
                    entry.insert(HeldLock {
                            fd,
                            mode: lock.mode,
                            resource: lock.resource,
                        });
                    self.newly_held.insert(lock.resource, lock.mode);
                    gained += 1;
                }
                TryFlockOutcome::Contended(witness) => {
                    drop(entry);
                    self.record_contention(ContentionEvidence {
                        blocker: lock.resource,
                        mode: lock.mode,
                        _witness: witness,
                    });
                }
            }
        }
        Ok(gained)
    }

    /// All-or-nothing probe of `candidate` reusing held locks: any
    /// lock already held counts; missing ones are tried non-blocking.
    /// On success the candidate's fds are REMOVED from the held map
    /// and returned in candidate order. On failure the newly taken
    /// fds are released and previously held ones stay — the probe
    /// must not regress accumulated progress.
    pub(crate) fn probe_complete(
        &mut self,
        candidate: &[ResourceLock],
    ) -> Result<Option<Vec<OwnedFd>>> {
        let mut fresh: Vec<(String, HeldLock)> = Vec::new();
        for lock in candidate {
            match self.map.get(&lock.path) {
                Some(held) if held.mode == lock.mode => continue,
                Some(_) => {
                    drop(fresh);
                    return Ok(None);
                }
                None => {}
            }
            match try_flock_with_witness(&lock.path, lock.mode)? {
                TryFlockOutcome::Acquired(fd) => fresh.push((
                    lock.path.clone(),
                    HeldLock {
                        fd,
                        mode: lock.mode,
                        resource: lock.resource,
                    },
                )),
                TryFlockOutcome::Contended(witness) => {
                    drop(fresh);
                    self.record_contention(ContentionEvidence {
                        blocker: lock.resource,
                        mode: lock.mode,
                        _witness: witness,
                    });
                    return Ok(None);
                }
            }
        }
        for (path, held) in fresh {
            self.newly_held.insert(held.resource, held.mode);
            self.map.insert(path, held);
        }
        let mut out = Vec::with_capacity(candidate.len());
        for lock in candidate {
            out.push(
                self.map
                    .remove(&lock.path)
                    .expect("probe_complete: candidate path must be held")
                    .fd,
            );
        }
        Ok(Some(out))
    }

    /// Whether every lock in `target` is currently held.
    pub(crate) fn covers(&self, target: &[ResourceLock]) -> bool {
        target.iter().all(|lock| {
            self.map
                .get(&lock.path)
                .is_some_and(|held| held.mode == lock.mode)
        })
    }

    /// Take the fds for `target` out of the map, in target order.
    /// Caller must have verified [`Self::covers`].
    pub(crate) fn take(&mut self, target: &[ResourceLock]) -> Vec<OwnedFd> {
        target
            .iter()
            .map(|lock| {
                self.map
                    .remove(&lock.path)
                    .expect("take: target path must be held")
                    .fd
            })
            .collect()
    }

    /// Paths currently held (for plan_live's overlap preference).
    pub(crate) fn held_paths(&self) -> BTreeSet<String> {
        self.map.keys().cloned().collect()
    }

    fn record_contention(&mut self, evidence: ContentionEvidence) {
        self.contention.insert(evidence);
    }

    fn take_registry_update(&mut self) -> Result<HeldRegistryUpdate> {
        Ok(HeldRegistryUpdate {
            newly_held: claim_from_resource_modes(std::mem::take(&mut self.newly_held))?,
            abandoned: claim_from_resource_modes(std::mem::take(&mut self.abandoned_resources))?,
            contention: std::mem::take(&mut self.contention),
            _abandoned_fds: std::mem::take(&mut self.abandoned_fds),
        })
    }
}

fn claim_from_resource_modes(
    resources: std::collections::BTreeMap<ResourceKey, FlockMode>,
) -> Result<ClaimSet> {
    let mut cpus = BTreeSet::new();
    let mut llcs = BTreeSet::new();
    let mut cpu_mode = None;
    let mut llc_mode = None;
    for (resource, mode) in resources {
        match resource {
            ResourceKey::Cpu(cpu) => {
                if cpu_mode.is_some_and(|existing| existing != mode) {
                    anyhow::bail!("coordinator accumulated mixed CPU lock modes");
                }
                cpu_mode = Some(mode);
                cpus.insert(cpu);
            }
            ResourceKey::Llc(llc) => {
                if llc_mode.is_some_and(|existing| existing != mode) {
                    anyhow::bail!("coordinator accumulated mixed LLC lock modes");
                }
                llc_mode = Some(mode);
                llcs.insert(llc);
            }
        }
    }
    Ok(ClaimSet::with_modes(
        llcs,
        cpus,
        llc_mode.unwrap_or(FlockMode::Shared),
        cpu_mode.unwrap_or(FlockMode::Shared),
    ))
}

/// One coordinator-loop iteration's verdict, produced by the caller's step
/// closure (which owns the path-specific planning + probing logic).
pub(crate) enum CoordinatorStep<T> {
    /// Acquisition complete. `claim` names the exact resource fds carried by
    /// `value`, which may differ from the coordinator's previously published
    /// planning alternative.
    Complete { claim: ClaimSet, value: T },
    /// Still waiting. `claim` is the freshly planned target to publish
    /// before sleeping for the next release event.
    Waiting { claim: ClaimSet },
    /// The step decided acquisition cannot proceed at all (e.g. no
    /// plannable candidate remains). Terminal; not a timeout.
    Abort { reason: String },
}

/// Outcome of [`acquire_as_coordinator`].
pub(crate) enum CoordinatorOutcome<T> {
    Acquired(T),
    Aborted { reason: String },
}

struct HolderObserver {
    mountinfo: Option<String>,
    needles: std::collections::BTreeMap<(bool, usize), String>,
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

impl HolderObserver {
    fn new() -> Self {
        #[cfg(test)]
        let forced_unavailable = FORCE_HOLDER_OBSERVER_UNAVAILABLE.with(std::cell::Cell::get);
        #[cfg(not(test))]
        let forced_unavailable = false;
        Self {
            mountinfo: if forced_unavailable {
                None
            } else {
                crate::flock::read_mountinfo().ok()
            },
            needles: std::collections::BTreeMap::new(),
            proof_files: std::collections::BTreeMap::new(),
            proof_locks: BTreeSet::new(),
        }
    }

    fn observe(
        &mut self,
        request: &registry::ObservationRequest,
    ) -> registry::AvailabilityObservation {
        self.release_proofs();
        match self.observe_proc(request) {
            Ok(observation) => observation,
            Err(error) => {
                tracing::debug!(
                    %error,
                    "cannot observe reservation modes through procfs; using retained read-only flock proofs"
                );
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
        }
    }

    fn observe_proc(
        &mut self,
        request: &registry::ObservationRequest,
    ) -> Result<registry::AvailabilityObservation> {
        let mountinfo = self
            .mountinfo
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("/proc/self/mountinfo was unavailable"))?;
        for &llc in request.llcs.keys() {
            if let std::collections::btree_map::Entry::Vacant(entry) =
                self.needles.entry((false, llc))
            {
                let path = PathBuf::from(super::llc_lock_path(llc));
                let needle =
                    crate::flock::mountinfo::needle_from_path_with_mountinfo(&path, mountinfo)?;
                entry.insert(needle);
            }
        }
        for &cpu in request.cpus.keys() {
            if let std::collections::btree_map::Entry::Vacant(entry) =
                self.needles.entry((true, cpu))
            {
                let path = PathBuf::from(super::cpu_lock_path(cpu));
                let needle =
                    crate::flock::mountinfo::needle_from_path_with_mountinfo(&path, mountinfo)?;
                entry.insert(needle);
            }
        }
        let resource_needles: BTreeSet<String> = request
            .llcs
            .keys()
            .filter_map(|llc| self.needles.get(&(false, *llc)).cloned())
            .chain(
                request
                    .cpus
                    .keys()
                    .filter_map(|cpu| self.needles.get(&(true, *cpu)).cloned()),
            )
            .collect();
        let summaries = crate::flock::read_flock_mode_summaries(&resource_needles)?;
        let mut observation = registry::AvailabilityObservation::default();
        for &cpu in request.cpus.keys() {
            let summary = summaries[&self.needles[&(true, cpu)]];
            let availability = if summary.exclusive_holder {
                registry::CpuAvailability::ExclusiveHeld
            } else if summary.any_holder {
                registry::CpuAvailability::SharedHeld
            } else {
                registry::CpuAvailability::Free
            };
            observation.cpus.insert(
                cpu,
                registry::CpuObservation {
                    availability,
                    sh_resolved: true,
                    ex_resolved: true,
                },
            );
        }
        for &llc in request.llcs.keys() {
            let summary = summaries[&self.needles[&(false, llc)]];
            let availability = if summary.exclusive_holder {
                registry::LlcAvailability::ExclusiveHeld
            } else if summary.any_holder {
                registry::LlcAvailability::SharedHeld
            } else {
                registry::LlcAvailability::Free
            };
            observation.llcs.insert(
                llc,
                registry::LlcObservation {
                    availability,
                    sh_resolved: true,
                    ex_resolved: true,
                },
            );
        }
        Ok(observation)
    }

    fn proof_file(&mut self, key: ResourceKey) -> Result<&std::fs::File> {
        if let std::collections::btree_map::Entry::Vacant(entry) =
            self.proof_files.entry(key)
        {
            let path = match key {
                ResourceKey::Llc(index) => super::llc_lock_path(index),
                ResourceKey::Cpu(index) => super::cpu_lock_path(index),
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
                        ex_resolved: false,
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
                        ex_resolved: false,
                    },
                );
            }
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
pub(crate) fn acquire_as_coordinator<T>(
    coordinator: CoordinatorTicket,
    step: impl FnMut(&mut HeldLocks) -> Result<CoordinatorStep<T>>,
) -> Result<CoordinatorOutcome<T>> {
    acquire_as_coordinator_impl(coordinator, None, step)
}

/// Cancellation-aware variant of [`acquire_as_coordinator`].
///
/// The ticket keeps its private wake registration alive across the
/// waiter-to-coordinator transition, so cancellation interrupts inotify rather
/// than waiting for the fallback tick.
pub(crate) fn acquire_as_coordinator_interruptible<T>(
    coordinator: CoordinatorTicket,
    cancelled: &AtomicBool,
    step: impl FnMut(&mut HeldLocks) -> Result<CoordinatorStep<T>>,
) -> Result<CoordinatorOutcome<T>> {
    acquire_as_coordinator_impl(coordinator, Some(cancelled), step)
}

fn acquire_as_coordinator_impl<T>(
    mut coordinator: CoordinatorTicket,
    cancelled: Option<&AtomicBool>,
    mut step: impl FnMut(&mut HeldLocks) -> Result<CoordinatorStep<T>>,
) -> Result<CoordinatorOutcome<T>> {
    check_interrupted(cancelled)?;
    let watch = check_result(LockDirWatch::new(), cancelled)?;
    check_interrupted(cancelled)?;
    let mut held = HeldLocks::default();
    let mut watched_resources = ClaimSet::default();
    let mut observer = HolderObserver::new();
    let mut first = true;
    let mut retry_due = false;
    let mut retry_deadline = std::time::Instant::now() + COORDINATOR_WAKE_FALLBACK;
    let mut pending_events = LockDirEvents::default();
    let outcome = loop {
        super::tick_reservation_wait_progress();
        check_interrupted(cancelled)?;
        pending_events.merge(check_result(watch.drain(&watched_resources), cancelled)?);
        let closed_tickets: Vec<_> = pending_events.liveness_closes.iter().copied().collect();
        let mut snapshot = check_result(
            coordinator.ticket.schedule(
                None,
                &pending_events.cpu_closes,
                &pending_events.llc_closes,
                pending_events.overflow,
                None,
                &[],
                None,
                &closed_tickets,
                first || retry_due,
                first.then_some(PREWATCH_LIVENESS_RECONCILE_DELAY),
                pending_events.overflow,
                cancelled,
            ),
            cancelled,
        )?;
        retry_due = false;
        let mut liveness_due_in = snapshot.liveness_due_in;
        watched_resources = snapshot.watch.clone();
        let mut should_step = first || snapshot.should_step;
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
            liveness_due_in = snapshot.liveness_due_in;
            watched_resources = snapshot.watch.clone();
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
                    held.retain_target(&[]);
                    let update = held.take_registry_update()?;
                    let markers = update.contention.marker_vec();
                    // `finish_acquired` publishes the held flocks and removes
                    // the coordinator record atomically. Once it returns
                    // success, cancellation is for the caller's next lifecycle
                    // phase—not grounds to roll back this committed acquire.
                    coordinator.ticket.finish_acquired(
                        &claim,
                        &update.abandoned,
                        &markers,
                        cancelled,
                    )?;
                    drop(update);
                    break CoordinatorOutcome::Acquired(value);
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
                    let update = held.take_registry_update()?;
                    let markers = update.contention.marker_vec();
                    let snapshot = check_result(
                        coordinator.ticket.schedule(
                            Some(&claim),
                            &BTreeSet::new(),
                            &BTreeSet::new(),
                            false,
                            Some(&update.newly_held),
                            &markers,
                            Some(&update.abandoned),
                            &closed_tickets,
                            false,
                            None,
                            false,
                            cancelled,
                        ),
                        cancelled,
                    )?;
                    drop(update);
                    let observe_before_sleep = snapshot.observation.is_some();
                    liveness_due_in = snapshot.liveness_due_in;
                    watched_resources = snapshot.watch;
                    if observe_before_sleep {
                        first = false;
                        pending_events = LockDirEvents::default();
                        continue;
                    }
                }
            }
        }
        first = false;
        check_interrupted(cancelled)?;
        let retry_interval = watch.semantic_retry_interval(observation_pending);
        let now = std::time::Instant::now();
        retry_deadline = retry_deadline.min(now + retry_interval);
        let liveness_deadline = now + liveness_due_in;
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

/// Canonical global lock order for a resource set: LLC locks by
/// ascending index, then CPU locks by ascending index. Every
/// acquirer — fast path and coordinator alike — walks locks in this order,
/// which is what lets the coordinator hold partials
/// safely and keeps overlapping fast-path sets from half-blocking
/// each other.
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
    let mut llcs: Vec<usize> = llc_indices.to_vec();
    llcs.sort_unstable();
    llcs.dedup();
    let mut cpu_sorted: Vec<usize> = cpus.to_vec();
    cpu_sorted.sort_unstable();
    cpu_sorted.dedup();
    let mut out = Vec::with_capacity(llcs.len() + cpu_sorted.len());
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
    out
}
