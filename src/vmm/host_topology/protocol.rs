//! Cross-process host-resource admission under nextest.
//!
//! Every ktstr process sharing a lock directory participates in one v26
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
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use crate::flock::{FlockMode, TryFlockOutcome, try_flock_with_witness};

mod exec_handoff;
mod registry;

#[cfg(test)]
pub(crate) use exec_handoff::EXEC_HANDOFF_ENV;
pub(crate) use exec_handoff::{prepare_pending_exec_handoff, take_pending_exec_handoff};

#[cfg(test)]
pub(crate) use registry::{
    active_free_head_is_rejected_for_tests, aggregate_conflicts as registered_claim_conflicts,
    aggregate_snapshot_read_count_for_tests, churn_registry_generation_for_tests,
    coordinator_liveness_probe_for_tests, defer_liveness_maintenance_for_tests,
    diagnostics_for_tests as ticket_registry_diagnostics_for_tests,
    exercise_acknowledgement_payload_notify_order_for_tests,
    exercise_bounded_replan_window_for_tests, exercise_busy_to_free_close_for_tests,
    exercise_changed_claim_coverage_for_tests, exercise_changed_replan_wave_completions_for_tests,
    exercise_clean_coordinator_mismatch_recovery_for_tests,
    exercise_coordinator_heartbeat_deadline_for_tests,
    exercise_coordinator_pending_replan_for_tests, exercise_coordinator_turnover_for_tests,
    exercise_cpu_ex_contention_shared_wake_for_tests, exercise_cpu_mode_repair_for_tests,
    exercise_cpu_shared_commit_improvement_for_tests, exercise_dead_waiter_takeover_skip_for_tests,
    exercise_deferred_rescan_policy_for_tests, exercise_dirty_repair_grant_charges_for_tests,
    exercise_dirty_repair_notification_for_tests, exercise_disjoint_entrant_proceeds_for_tests,
    exercise_exact_commit_scan_elision_for_tests, exercise_exclusive_grant_bias_only_for_tests,
    exercise_fresh_waiting_coordinator_takeover_for_tests,
    exercise_generation_timeout_takeover_for_tests, exercise_grant_charge_revoke_ack_for_tests,
    exercise_grant_completion_batch_for_tests, exercise_grant_disjoint_completion_for_tests,
    exercise_grant_scan_crash_fence_for_tests, exercise_granted_charge_lifecycle_for_tests,
    exercise_granted_only_drain_election_reads_for_tests, exercise_granted_serial_scope_for_tests,
    exercise_granular_prefix_invalidation_for_tests, exercise_held_teardown_notify_count_for_tests,
    exercise_intrascan_fence_epoch_for_tests, exercise_issue_serial_race_for_tests,
    exercise_junior_dirty_park_for_tests, exercise_known_free_close_storm_for_tests,
    exercise_llc_ex_contention_shared_wake_for_tests,
    exercise_llc_sh_only_shared_to_free_close_for_tests,
    exercise_mismatched_commit_rescan_for_tests, exercise_n_entrants_read_under_churn_for_tests,
    exercise_one_shot_replacement_for_tests, exercise_pending_activation_overlap_watch_for_tests,
    exercise_pending_replan_grant_races_for_tests,
    exercise_permit_grant_charge_avoidance_for_tests, exercise_prefix_epoch_validation_for_tests,
    exercise_prefix_order_and_repair_for_tests,
    exercise_prefix_refresh_after_predecessor_release_for_tests,
    exercise_preparation_pool_budget_for_tests, exercise_preparation_pool_crash_recovery_for_tests,
    exercise_preparation_pool_starvation_for_tests, exercise_quiet_generation_additions_for_tests,
    exercise_release_coalesce_for_tests, exercise_repeated_coordinator_takeover_for_tests,
    exercise_replacement_kept_overlap_guard_for_tests,
    exercise_replan_capacity_validation_for_tests, exercise_replan_completion_election_for_tests,
    exercise_replan_crash_repair_for_tests, exercise_replan_expiry_publication_crash_for_tests,
    exercise_replan_straggler_progress_for_tests, exercise_replan_token_wave_for_tests,
    exercise_replan_wave_expiry_for_tests, exercise_retained_mapping_slot_reuse_for_tests,
    exercise_retained_shared_publication_for_tests, exercise_revocation_ack_for_tests,
    exercise_revoke_crash_repair_for_tests, exercise_revoked_owner_death_for_tests,
    exercise_same_wake_own_designation_grant_for_tests,
    exercise_same_wake_redesignation_expired_release_for_tests,
    exercise_same_wake_redesignation_fallback_for_tests,
    exercise_same_wake_redesignation_grant_for_tests,
    exercise_same_wake_redesignation_older_fence_for_tests,
    exercise_scan_metadata_validation_for_tests, exercise_shared_commit_improvement_for_tests,
    exercise_shared_watch_held_metadata_for_tests, exercise_stale_acquired_release_order_for_tests,
    exercise_stale_contention_commit_for_tests,
    exercise_stale_heartbeat_known_free_close_for_tests,
    exercise_stalled_takeover_notification_for_tests, exercise_superset_commit_rescan_for_tests,
    exercise_unchanged_completion_guard_for_tests,
    exercise_waiting_publication_release_progress_for_tests,
    exercise_waiting_release_wake_for_tests, exercise_writer_intent_initialization_race_for_tests,
    expire_coordinator_lease_for_tests, generation_wait_calls_for_tests,
    grant_charge_matches_derived_for_tests, hold_registry_exclusive_after_intent_for_tests,
    hold_registry_exclusive_for_tests, hold_registry_shared_for_tests,
    initializer_temp_count_for_tests as registry_initializer_temp_count_for_tests,
    missing_liveness_probe_does_not_create_for_tests,
    observer_preserves_uninitialized_header_for_tests,
    prepare_zeroed_uninitialized_header_for_tests, registry_ex_acquisition_count_for_tests,
    resource_epoch_for_tests, round_trip_claim_modes_for_tests, shared_state_read_count_for_tests,
    snapshot as ticket_registry_snapshot_for_tests, ticket_blocked_at_current_serial_for_tests,
    ticket_is_granted_for_tests, ticket_is_revoked_for_tests, ticket_is_waiting_for_tests,
    ticket_shared_mapping_build_count_for_tests, try_hold_registry_exclusive_for_tests,
    try_hold_registry_shared_for_tests, union_claims_for_tests,
};

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

/// Shared ownership of one long-lived host-admission flock.
///
/// Linux reports `IN_CLOSE_WRITE` before implicit flock teardown during the
/// final `close(2)`. Explicitly unlocking while the descriptor is still open
/// reverses that ordering for admission owners: every writable close event is
/// emitted only after the resource is physically available. Clones share the
/// same userspace ownership boundary, so the final logical owner is known and
/// performs the one explicit unlock before the underlying fd closes.
#[derive(Clone)]
pub(crate) struct AdmissionFlock(Arc<AdmissionFlockInner>);

struct AdmissionFlockInner {
    fd: OwnedFd,
}

#[cfg(test)]
thread_local! {
    static ADMISSION_FLOCK_UNLOCK_HOOK:
        std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_admission_flock_unlock_hook_for_tests(hook: impl FnOnce() + 'static) {
    ADMISSION_FLOCK_UNLOCK_HOOK.with(|slot| {
        let previous = slot.borrow_mut().replace(Box::new(hook));
        assert!(
            previous.is_none(),
            "admission-flock unlock test hook was already installed",
        );
    });
}

#[cfg(test)]
fn run_admission_flock_unlock_hook_for_tests() {
    ADMISSION_FLOCK_UNLOCK_HOOK.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
}

#[cfg(not(test))]
fn run_admission_flock_unlock_hook_for_tests() {}

impl AdmissionFlock {
    pub(crate) fn from_acquired(fd: OwnedFd) -> Self {
        Self(Arc::new(AdmissionFlockInner { fd }))
    }

    /// Preserve the fallible `OwnedFd::try_clone` call shape at reuse sites,
    /// while sharing the one physical lock owner and unlock-before-close edge.
    pub(crate) fn try_clone(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

impl AsFd for AdmissionFlock {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.fd.as_fd()
    }
}

impl AsRawFd for AdmissionFlock {
    fn as_raw_fd(&self) -> RawFd {
        self.0.fd.as_raw_fd()
    }
}

impl std::fmt::Debug for AdmissionFlock {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AdmissionFlock")
            .field("fd", &self.as_raw_fd())
            .field("owners", &Arc::strong_count(&self.0))
            .finish()
    }
}

impl Drop for AdmissionFlockInner {
    fn drop(&mut self) {
        use rustix::fs::{FlockOperation, flock};

        loop {
            match flock(&self.fd, FlockOperation::Unlock) {
                Ok(()) => break,
                Err(error) if error == rustix::io::Errno::INTR => continue,
                Err(error) => {
                    tracing::error!(
                        fd = self.fd.as_raw_fd(),
                        %error,
                        "failed to explicitly unlock host-admission flock before close",
                    );
                    break;
                }
            }
        }
        run_admission_flock_unlock_hook_for_tests();
        // Post-release only (inert before this cell's run-claim release), so
        // this stamps the final drop of any admission resource retained past
        // release — the retained preparation permits behind the persisting
        // permit-lock files. `fd` closes only after this Drop returns.
        crate::vmm::exit_timing::stamp("admission_flock_release");
    }
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
/// predecessor-state transition. This long, per-process-staggered fallback is
/// only a missed-event recovery tick; it avoids turning hundreds of queued
/// cells into a synchronized `/proc/locks` polling herd.
const WAITER_CRASH_RECOVERY_BASE: Duration = Duration::from_secs(3);

/// The recovery deadline a waiting ticket sleeps on. The per-process offset is
/// what breaks the herd, so every waiter loop takes the deadline from here
/// rather than staggering (or forgetting to stagger) locally.
fn waiter_crash_recovery_fallback() -> Duration {
    WAITER_CRASH_RECOVERY_BASE + Duration::from_millis((std::process::id() as u64 * 37) % 1000)
}
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

/// CI-only coordinator wake-source accounting (validation instrumentation).
///
/// Process-global cumulative counts of why the coordinator's blocking watch
/// returned. `EVENT` = a real inotify resource/notify wake (the work-conserving
/// path); `RETRY_TIMEOUT` = the semantic retry deadline expired with nothing
/// event-driven to do, of which `FALLBACK` is the subset that rode the full
/// [`COORDINATOR_WAKE_FALLBACK`] tick (no observation pending). Under a healthy
/// event-driven drain `FALLBACK` should stay ~0 while cells are still queued;
/// its only legitimate use is the rare missed-event backstop. Dumped to
/// `${KTSTR_BUILD_DIAGNOSTICS_DIR}/coordinator-wakes-<pid>.txt` (truncating, so
/// the file always holds the latest cumulative totals for that pid) exactly like
/// the admission-timing / queue-wait diagnostics; untouched when the env var is
/// unset, so local and production runs pay only two relaxed atomic increments.
static COORDINATOR_EVENT_WAKES: AtomicU64 = AtomicU64::new(0);
static COORDINATOR_RETRY_TIMEOUT_WAKES: AtomicU64 = AtomicU64::new(0);
static COORDINATOR_FALLBACK_WAKES: AtomicU64 = AtomicU64::new(0);

/// Cumulative coordinator inotify event-wakes for this process. The herd
/// benchmark samples it around a run to report event-wakes per grant, so a
/// watch-traffic regression (for example from widening what a coordinator
/// watches) is visible rather than assumed harmless.
#[cfg(test)]
pub(crate) fn coordinator_event_wakes_for_tests() -> u64 {
    COORDINATOR_EVENT_WAKES.load(Ordering::Relaxed)
}

fn persist_coordinator_wake_stats_if_enabled() {
    let Some(root) = std::env::var_os("KTSTR_BUILD_DIAGNOSTICS_DIR")
        .filter(|root| !root.is_empty())
        .map(PathBuf::from)
    else {
        return;
    };
    let event = COORDINATOR_EVENT_WAKES.load(Ordering::Relaxed);
    let retry = COORDINATOR_RETRY_TIMEOUT_WAKES.load(Ordering::Relaxed);
    let fallback = COORDINATOR_FALLBACK_WAKES.load(Ordering::Relaxed);
    let (grant_scans, records_scanned) = registry::coordinator_scan_stats();
    let pid = std::process::id();
    let line = format!(
        "coordinator-wakes: pid={pid} event={event} retry_timeout={retry} fallback={fallback} \
         grant_scans={grant_scans} records_scanned={records_scanned}\n"
    );
    if std::fs::create_dir_all(&root).is_err() {
        return;
    }
    let temp = root.join(format!(".coordinator-wakes-{pid}.tmp"));
    if std::fs::write(&temp, &line).is_ok() {
        let _ = std::fs::rename(&temp, root.join(format!("coordinator-wakes-{pid}.txt")));
    }
}

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
    /// Test-only since the entry-park overlap diagnostic was removed; the
    /// mode-matrix tests remain its consumers.
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

    /// Count of in-flight grant charges (GRANTED/REVOKED records) covering
    /// this CPU. Planner rank-key bias only — never a fence, and never folded
    /// into the holder counts above.
    pub(crate) fn cpu_grant_count(&self, cpu: usize) -> Result<usize> {
        self.inner.cpu_grant_count(cpu)
    }

    /// Count of in-flight grant charges covering this LLC. Bias only.
    pub(crate) fn llc_grant_count(&self, llc: usize) -> Result<usize> {
        self.inner.llc_grant_count(llc)
    }

    /// Whether `candidate` overlaps any in-flight grant charge, permits
    /// included — they are charged at their folded CPU-space indices, so the
    /// snapshot this is called on must have been taken over a watch naming
    /// the permits being tested. Same soft-avoid-only contract as
    /// [`GrantedProbe::grant_conflicts`].
    pub(crate) fn grant_conflicts(&self, candidate: &ClaimSet) -> Result<bool> {
        self.inner.grant_conflicts(candidate)
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
///
/// Every writable (`IN_CLOSE_WRITE`) resource-lock close is reported — the
/// classifier is handed no watch set to filter by, so drain/wait take none
/// either. A close on a resource the coordinator is not presently focused on
/// can still unblock a *disjoint* waiter, and the release edge is single-shot:
/// a scan that armed a narrow watch after that edge already fired would never
/// see it and would ride the [`COORDINATOR_WAKE_FALLBACK`] tick instead of
/// granting immediately.
/// Reporting every real release keeps the wake work-conserving. The writable
/// closes this queue sees are holder releases plus the contention witnesses
/// `try_flock_with_witness` deliberately retains — those fds are `O_RDWR` too,
/// and their close is exactly the edge a blocked waiter orders after publishing
/// its state, so reporting them is the point rather than a leak. What stays out
/// of the queue is pure observation: availability proofs open the lock file
/// read-only (`HolderObserver::proof_file`, `resource_flock_is_free`), so their
/// `IN_CLOSE_NOWRITE` cannot wake the coordinator that took them.
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
    pub(crate) fn contains_cpu_close(&self, cpu: usize) -> bool {
        self.cpu_closes.contains(&cpu)
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
    pub(crate) fn drain(&self) -> Result<LockDirEvents> {
        self.read_bounded()
    }

    fn read_bounded(&self) -> Result<LockDirEvents> {
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
                    self.classify(events, &mut batch);
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
    pub(crate) fn wait(&self, timeout: Duration) -> Result<Option<LockDirEvents>> {
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
            let batch = self.read_bounded()?;
            if batch.is_actionable() {
                return Ok(Some(batch));
            }
        }
    }

    /// Resource-lock closes are reported regardless of the caller's current
    /// focus set so a disjoint waiter's release edge is never discarded (see
    /// the type doc), which is why no watch set reaches here.
    fn classify(&self, events: Vec<nix::sys::inotify::InotifyEvent>, batch: &mut LockDirEvents) {
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
            {
                batch.llc_closes.insert(index);
            }
            if let Some(index) = resource
                .cpu_prefix
                .as_deref()
                .and_then(|prefix| resource_index(name, prefix))
            {
                batch.cpu_closes.insert(index);
            }
            if let Some(index) = resource
                .permit_prefix
                .as_deref()
                .and_then(|prefix| resource_index(name, prefix))
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

    pub(crate) fn drain(&self) -> Result<LockDirEvents> {
        match &self.backend {
            LockDirWatchBackend::RealInotify(wake) => wake.drain(),
            #[cfg(test)]
            LockDirWatchBackend::TestRetry => Ok(LockDirEvents::default()),
        }
    }

    pub(crate) fn wait(&self, timeout: Duration) -> Result<Option<LockDirEvents>> {
        match &self.backend {
            LockDirWatchBackend::RealInotify(wake) => wake.wait(timeout),
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

/// Start the grant-callback clock only when the diagnostics sink is live, so a
/// production wake pays neither the clock read nor the accounting.
fn grant_callback_clock() -> Option<std::time::Instant> {
    crate::vmm::grant_flow::enabled().then(std::time::Instant::now)
}

/// Charge one licensed grant's callback wall to the grant-flow image. An
/// unlicensed REPLAN completion never held a grant, so it is excluded here for
/// the same reason it is excluded from `grants_lost`.
fn note_grant_callback_elapsed(licensed_grant: bool, started: Option<std::time::Instant>) {
    if let Some(started) = started.filter(|_| licensed_grant) {
        crate::vmm::grant_flow::note_grant_callback(
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
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
    reusable_permits: Vec<(usize, AdmissionFlock)>,
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

    pub(crate) fn clone_reusable_permits(&self) -> Result<Vec<(usize, AdmissionFlock)>> {
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
        // Evaluated as the original short-circuit chain, so a candidate pays
        // for exactly the fences it reaches; the first one that rejects it is
        // the one attributed.
        use crate::vmm::grant_flow::{GrantBlock, note_block};
        if just_contended {
            note_block(GrantBlock::Contended);
            return Ok(false);
        }
        // `first_conflict` costs exactly what `conflicts` does and names the
        // fenced resource, so the axis split is free: permit-pool selection
        // walks every pool entry through this fence, and a rejection there says
        // the admission budget is spent, not that the host has no room.
        if let Some(marker) = self.predecessors.first_conflict(candidate)? {
            note_block(GrantBlock::Predecessors);
            match marker.blocker {
                ResourceKey::Permit(_) => note_block(GrantBlock::PredecessorsPermit),
                ResourceKey::Llc(_) => note_block(GrantBlock::PredecessorsLlc),
                ResourceKey::Cpu(_) => {}
            }
            return Ok(false);
        }
        if !self.availability.allows(candidate)? {
            // `allows` rejects an unobserved resource exactly like a held one.
            // Splitting them is the whole point of the breakdown, and the
            // extra pass runs only for the diagnostics sink.
            if crate::vmm::grant_flow::enabled() {
                note_block(if self.availability.unobserved(candidate)? {
                    GrantBlock::Unobserved
                } else {
                    GrantBlock::Busy
                });
            }
            return Ok(false);
        }
        Ok(true)
    }

    pub(crate) fn candidate_holder_pressure(&self, candidate: &ClaimSet) -> Result<usize> {
        let cpu = candidate.cpus.iter().try_fold(0usize, |total, &cpu| {
            Ok::<_, anyhow::Error>(total.saturating_add(self.predecessors.cpu_holder_count(cpu)?))
        })?;
        candidate.llcs.iter().try_fold(cpu, |total, &llc| {
            Ok::<_, anyhow::Error>(total.saturating_add(self.predecessors.llc_holder_count(llc)?))
        })
    }

    /// Whether `candidate` overlaps any in-flight grant charge other than
    /// this ticket's own (a licensed GRANTED callback's charge is subtracted
    /// when the snapshot is captured). Soft-avoid input for the grant-aware
    /// selection tier only — callers MUST rerun grant-blind when no
    /// grant-free candidate exists, or the permit axis livelocks: a senior
    /// that never publishes an overlapping claim never triggers the scan's
    /// ticket-order revoke, while fast-path juniors re-grab freed permits.
    pub(crate) fn grant_conflicts(&self, candidate: &ClaimSet) -> Result<bool> {
        self.predecessors.grant_conflicts(candidate)
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
    /// remains CPU-SH: a failed CPU-EX probe is only the signal to take the
    /// same-wake shared fallback on the same designated placement, not a
    /// blocker that the registry must wait to clear.
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

    /// Same-wake re-designation for a flexible waiter woken to re-plan.
    ///
    /// A REPLAN wake carries no grant license (`acquisition_allowed` is false),
    /// so [`Self::try_acquire`] is inert and the closure would otherwise only
    /// [`Self::reserve`] a replacement and wait for a second authoritative-scan
    /// grant cycle. When an alternative candidate is free and does not conflict
    /// with any older ticket's reservation, acquire it physically now and let
    /// the wake publish HELD directly. The fence is exactly the one an ordinary
    /// grant applies ([`Self::candidate_ready`]: predecessor prefix +
    /// mode-aware availability), so an earlier ticket is never overtaken. A
    /// licensed GRANTED wake is unaffected and keeps acquiring only its own
    /// designation through [`Self::try_acquire`].
    pub(crate) fn try_acquire_redesignation<T, O: IntoProbeOutcome<T>>(
        &mut self,
        candidate: &ClaimSet,
        acquire: impl FnOnce() -> Result<O>,
    ) -> Result<Option<T>> {
        if self.acquisition_allowed {
            // A licensed grant acquires its designation through `try_acquire`;
            // re-designation is exclusively the no-license REPLAN optimisation.
            return Ok(None);
        }
        if !self.candidate_ready(candidate)? {
            return Ok(None);
        }
        match acquire()?.into_probe_outcome() {
            ProbeOutcome::Acquired(value) => {
                // The wake publishes HELD with the claim we physically hold.
                self.next_claim = candidate.clone();
                Ok(Some(value))
            }
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

    pub(crate) fn state_or_wait_for_tests(&self) -> Result<()> {
        self.ticket.state_or_wait_for_tests()
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
        // Post-release only: a PENDING intent that survives its cell's run to
        // retire here at exit is the intent-registration retirement the exit
        // seam is suspected to serialize on.
        crate::vmm::exit_timing::stamp("pending_admission_retire");
    }
}

/// The only probe surface available while atomically consuming a PENDING
/// admission. It can test candidates against external registry claims and
/// reuse the preparation owner's permit OFDs, but cannot queue or retry.
pub(crate) struct PendingOneShotProbe<'a> {
    registry: &'a registry::PendingOneShotProbe<'a>,
    reusable_permits: &'a [(usize, AdmissionFlock)],
}

impl PendingOneShotProbe<'_> {
    pub(crate) fn candidate_ready(&self, candidate: &ClaimSet) -> Result<bool> {
        self.registry.candidate_ready(candidate)
    }

    pub(crate) fn clone_reusable_permits(&self) -> Result<Vec<(usize, AdmissionFlock)>> {
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
    // Union the token pool into the registered watch. This is what marks the
    // record a preparation intent for the grant scan's pool budget, and it is
    // what makes the intent wake on any token release so the per-token blocked
    // pin is no longer needed to recover a freed slot.
    let watch = watch.union_envelope(&super::preparation_token_pool_watch()?);
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
        match ticket.state_or_wait(waiter_crash_recovery_fallback(), None)? {
            grant_or_replan @ (registry::State::Granted | registry::State::Replan) => {
                let licensed_grant = matches!(grant_or_replan, registry::State::Granted);
                let callback_started = grant_callback_clock();
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
                            crate::vmm::grant_flow::note_block(
                                crate::vmm::grant_flow::GrantBlock::Unlicensed,
                            );
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
                            super::PreparationProbe::RegistryContended(_)
                            | super::PreparationProbe::Unavailable => (None, None, None),
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
                note_grant_callback_elapsed(licensed_grant, callback_started);
                match result {
                    registry::GrantResult::Prepared(preparation, pending_claim) => {
                        return pending_admission_from_parts(ticket, preparation, pending_claim);
                    }
                    registry::GrantResult::Acquired(_, _) => {
                        unreachable!("intent callback published its run claim as HELD")
                    }
                    registry::GrantResult::Requeued | registry::GrantResult::LostGrant => {
                        // Same rule as the ordinary ticket drive: a licensed
                        // GRANTED grant that failed to reach preparation is
                        // headline churn, while an unlicensed REPLAN completion
                        // never held a grant to lose.
                        if licensed_grant {
                            crate::vmm::grant_flow::note_grant_lost();
                        }
                    }
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
                        super::PreparationProbe::RegistryContended(_)
                        | super::PreparationProbe::Unavailable => {
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
                // A process can be descheduled after registration elects it
                // but before it enters the coordinator loop. The heartbeat
                // lease deliberately transfers progress to a successor in
                // that case. Stay parked as an ordinary live ticket: a later
                // election may return this intent to COORDINATOR once the
                // successor finishes or is itself displaced.
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
                        *ticket,
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
        super::PreparationProbe::Contended(_)
        | super::PreparationProbe::RegistryContended(_)
        | super::PreparationProbe::Unavailable => Ok(None),
    }
}

pub(crate) fn register_pending_admission(max_permit_index: usize) -> Result<PendingAdmission> {
    let required = registry::required_bits_for_permit_index(max_permit_index);
    let allowed = super::host_allowed_cpus();
    anyhow::ensure!(
        !allowed.is_empty(),
        "could not determine allowed CPU set for preparation admission",
    );
    let mut rotation_bias = 0usize;
    loop {
        // Physical probing and registry publication form one complete sweep.
        // A logically fenced tuple drops every OFD and advances to the next
        // immediately available tuple; only a fully exhausted sweep may park.
        // The first sampled generation is retained so an improvement racing
        // any later probe makes the subsequent futex check return EAGAIN.
        let probe =
            super::try_preparation_candidates_once_waiting(
                rotation_bias,
                &allowed,
                |preparation, claim| {
                    let pending_claim = claim.clone();
                    Ok(match registry::Ticket::register_pending(required, claim)? {
                        registry::PendingRegistration::Registered(ticket) => {
                            super::PreparationCandidateDecision::Accepted(
                                pending_admission_from_parts(*ticket, preparation, pending_claim)?,
                            )
                        }
                        registry::PendingRegistration::Contended(generation) => {
                            drop(preparation);
                            super::PreparationCandidateDecision::RegistryContended(generation)
                        }
                    })
                },
            )?;
        match probe {
            super::PreparationProbe::Acquired(pending) => return Ok(pending),
            super::PreparationProbe::Contended(evidence) => {
                super::wait_for_preparation_contention(evidence, Duration::from_secs(2))?;
            }
            super::PreparationProbe::RegistryContended(generation) => {
                registry::wait_for_generation_change(generation, Duration::from_secs(2))?;
            }
            super::PreparationProbe::Unavailable => anyhow::bail!(
                "preparation candidate sweep found neither a runnable tuple nor a waitable blocker"
            ),
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
    let allowed = super::host_allowed_cpus();
    anyhow::ensure!(
        !allowed.is_empty(),
        "could not determine allowed CPU set for preparation continuity test",
    );
    let probe =
        super::try_preparation_candidates_once_waiting(0, &allowed, |preparation, active| {
            Ok(super::PreparationCandidateDecision::Accepted((
                preparation,
                active,
            )))
        })?;
    let super::PreparationProbe::Acquired((preparation, active)) = probe else {
        anyhow::bail!("isolated preparation continuity test found no runnable tuple")
    };
    let affinity_cpu = preparation.affinity_cpu;
    let cpu_permits = preparation.cpu_permits.clone();
    let memory_permits = preparation.memory_permits.clone();
    let token_permit = preparation.token_permit;
    let required =
        registry::required_bits_for_permit_index(super::admission_resource_capacity_hint()?);
    let ticket = match registry::Ticket::register_pending(required, active.clone())? {
        registry::PendingRegistration::Registered(ticket) => *ticket,
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
pub(crate) fn persist_wait_diagnostic_for_tests(root: &std::path::Path, bucket: u64) -> Result<()> {
    registry::persist_wait_diagnostic(root, bucket, bucket * 30)
}

#[cfg(test)]
pub(crate) struct PendingClaimForTests {
    ticket: Option<registry::Ticket>,
}

#[cfg(test)]
impl PendingClaimForTests {
    pub(crate) fn activate_for_tests(
        mut self,
        expected_pending: ClaimSet,
        claim: ClaimSet,
        watch: ClaimSet,
    ) -> Result<()> {
        self.ticket
            .as_mut()
            .expect("test pending claim was already consumed")
            .activate_pending(&expected_pending, claim, watch, None)
    }

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
pub(crate) fn reset_generation_wait_calls_for_tests() {
    registry::reset_generation_wait_calls_for_tests();
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
pub(crate) struct LevelProbeRecoveryOutcome {
    /// The release fired before the watch existed, so its inotify edge is lost:
    /// the drain after arming is empty.
    pub prewatch_release_left_no_event: bool,
    /// A bounded blocking wait times out — without the level probe only the 30s
    /// fallback would recover.
    pub blocking_wait_times_out: bool,
    /// The level probe recovers the free-state the lost edge would have carried.
    pub probe_recovers_free: bool,
    /// `probe_newly_watched_free` synthesizes the missing close so the loop
    /// re-scans instead of sleeping.
    pub synthesized_close_present: bool,
}

/// Arm-after-release: a resource freed BEFORE the coordinator created its watch.
/// The inotify edge is lost (edge-triggered; add_watch reports no existing
/// state), so drain/wait see nothing — the exact 30s-fallback trap. The level
/// probe recovers the free-state and synthesizes the close.
#[cfg(test)]
pub(crate) fn exercise_level_probe_recovers_prewatch_free() -> Result<LevelProbeRecoveryOutcome> {
    let cpu = 7usize;
    let path = super::cpu_lock_path(cpu);
    {
        let holder = crate::flock::try_flock(&path, FlockMode::Exclusive)?
            .ok_or_else(|| anyhow::anyhow!("fresh resource must acquire"))?;
        drop(holder); // release BEFORE any watch exists: the lost edge.
    }
    let watch = LockDirWatch::new_real_for_tests()?;
    let prewatch_release_left_no_event = !watch.drain()?.contains_cpu_close(cpu);
    let blocking_wait_times_out = watch.wait(std::time::Duration::from_millis(150))?.is_none();
    let probe_recovers_free = resource_flock_is_free(&path);
    let new_cpus: BTreeSet<usize> = std::iter::once(cpu).collect();
    let synthesized = probe_newly_watched_free(&new_cpus, &BTreeSet::new(), &BTreeSet::new());
    Ok(LevelProbeRecoveryOutcome {
        prewatch_release_left_no_event,
        blocking_wait_times_out,
        probe_recovers_free,
        synthesized_close_present: synthesized.contains_cpu_close(cpu),
    })
}

/// Head-of-line promotion: two resources newly enter the watch set — one whose
/// release predated the watch (free), one still held. The probe synthesizes a
/// close for ONLY the free one, so the coordinator re-scans for the promotable
/// waiter without a spurious wake for the still-blocked resource.
#[cfg(test)]
pub(crate) fn exercise_level_probe_head_of_line_promotion() -> Result<(bool, bool)> {
    let free_cpu = 8usize;
    let held_cpu = 9usize;
    let held = crate::flock::try_flock(super::cpu_lock_path(held_cpu), FlockMode::Exclusive)?
        .ok_or_else(|| anyhow::anyhow!("held cpu must acquire"))?;
    // Materialize `free_cpu`'s lockfile and leave it free (edge predated watch).
    let _ = crate::flock::try_flock(super::cpu_lock_path(free_cpu), FlockMode::Exclusive)?;
    let new_cpus: BTreeSet<usize> = [free_cpu, held_cpu].into_iter().collect();
    let synthesized = probe_newly_watched_free(&new_cpus, &BTreeSet::new(), &BTreeSet::new());
    let only_free_synthesized =
        synthesized.contains_cpu_close(free_cpu) && !synthesized.contains_cpu_close(held_cpu);
    drop(held);
    Ok((only_free_synthesized, synthesized.is_actionable()))
}

#[cfg(test)]
pub(crate) fn waiting_publication_requires_immediate_turn_for_tests(
    should_step: bool,
    observation_pending: bool,
) -> bool {
    waiting_publication_requires_immediate_turn(should_step, observation_pending)
}

#[cfg(test)]
pub(crate) struct CoordinatorPayloadNotifyCase {
    pub(crate) commit_terminated: bool,
    pub(crate) rescan_was_already_pending: bool,
    pub(crate) payload_released: bool,
    pub(crate) payload_released_at_notify: bool,
}

#[cfg(test)]
pub(crate) struct CoordinatorPayloadNotifyOutcome {
    pub(crate) complete_standby: CoordinatorPayloadNotifyCase,
    pub(crate) complete_lost_license: CoordinatorPayloadNotifyCase,
    pub(crate) prepare_standby: CoordinatorPayloadNotifyCase,
    pub(crate) prepare_lost_license: CoordinatorPayloadNotifyCase,
}

#[cfg(test)]
struct CoordinatorOpaqueDropProbe {
    dropped: std::rc::Rc<std::cell::Cell<bool>>,
}

#[cfg(test)]
impl Drop for CoordinatorOpaqueDropProbe {
    fn drop(&mut self) {
        self.dropped.set(true);
    }
}

#[cfg(test)]
fn exercise_coordinator_complete_notify_case(
    cpu: usize,
    lost_license: bool,
) -> Result<CoordinatorPayloadNotifyCase> {
    let claim = ClaimSet::new(std::iter::empty(), [cpu], FlockMode::Exclusive);
    let coordinator_ticket = registry::Ticket::register(claim.clone(), claim.clone(), None)?;
    let mut successor = registry::Ticket::register(claim.clone(), claim.clone(), None)?;
    let coordinator = Box::new(CoordinatorTicket {
        ticket: coordinator_ticket,
        preparation: None,
        preparation_watch: None,
    });
    let cancelled = std::sync::Arc::new(AtomicBool::new(false));
    let hook_cancelled = std::sync::Arc::clone(&cancelled);
    let payload_dropped = std::rc::Rc::new(std::cell::Cell::new(false));
    let step_payload_dropped = std::rc::Rc::clone(&payload_dropped);
    let hook_payload_dropped = std::rc::Rc::clone(&payload_dropped);
    let payload_dropped_at_notify = std::rc::Rc::new(std::cell::Cell::new(false));
    let hook_payload_dropped_at_notify = std::rc::Rc::clone(&payload_dropped_at_notify);
    let rescan_was_already_pending = std::rc::Rc::new(std::cell::Cell::new(false));
    let step_rescan_was_already_pending = std::rc::Rc::clone(&rescan_was_already_pending);
    let mut stepped = false;
    let result = acquire_as_coordinator_interruptible(coordinator, &cancelled, move |_| {
        anyhow::ensure!(
            !stepped,
            "coordinator Complete race ran more than one planner step"
        );
        stepped = true;
        step_rescan_was_already_pending.set(registry::force_coordinator_commit_race_for_tests(
            lost_license,
        )?);
        let hook_cancelled = std::sync::Arc::clone(&hook_cancelled);
        let hook_payload_dropped = std::rc::Rc::clone(&hook_payload_dropped);
        let hook_payload_dropped_at_notify = std::rc::Rc::clone(&hook_payload_dropped_at_notify);
        registry::arm_notify_hook_for_tests(move || {
            hook_payload_dropped_at_notify.set(hook_payload_dropped.get());
            hook_cancelled.store(true, Ordering::Release);
        });
        Ok(CoordinatorStep::Complete {
            claim: claim.clone(),
            value: CoordinatorOpaqueDropProbe {
                dropped: std::rc::Rc::clone(&step_payload_dropped),
            },
        })
    });
    let outcome = CoordinatorPayloadNotifyCase {
        commit_terminated: result.is_err(),
        rescan_was_already_pending: rescan_was_already_pending.get(),
        payload_released: payload_dropped.get(),
        payload_released_at_notify: payload_dropped_at_notify.get(),
    };
    successor.finish(None)?;
    Ok(outcome)
}

#[cfg(test)]
fn exercise_coordinator_prepare_notify_case(
    permit: usize,
    lost_license: bool,
) -> Result<CoordinatorPayloadNotifyCase> {
    let path = super::permit_lock_path(permit);
    let permit_fd = crate::flock::try_flock(&path, FlockMode::Exclusive)?
        .ok_or_else(|| anyhow::anyhow!("coordinator Prepare test permit {permit} is busy"))?;
    let preparation = super::PreparationPermit {
        index: permit,
        token_permit: permit,
        cpu_permits: Vec::new(),
        memory_permits: Vec::new(),
        permit_fds: vec![(permit, AdmissionFlock::from_acquired(permit_fd))],
        affinity_lock: None,
        affinity_cpu: 0,
        original_affinity: Vec::new(),
        affinity_constrained: false,
    };
    let claim = preparation.claim();
    let coordinator_ticket = registry::Ticket::register(claim.clone(), claim.clone(), None)?;
    let mut successor = registry::Ticket::register(claim.clone(), claim.clone(), None)?;
    let coordinator = Box::new(CoordinatorTicket {
        ticket: coordinator_ticket,
        preparation: None,
        preparation_watch: None,
    });
    let cancelled = std::sync::Arc::new(AtomicBool::new(false));
    let hook_cancelled = std::sync::Arc::clone(&cancelled);
    let payload_released_at_notify = std::rc::Rc::new(std::cell::Cell::new(false));
    let hook_payload_released_at_notify = std::rc::Rc::clone(&payload_released_at_notify);
    let rescan_was_already_pending = std::rc::Rc::new(std::cell::Cell::new(false));
    let step_rescan_was_already_pending = std::rc::Rc::clone(&rescan_was_already_pending);
    let mut preparation = Some(Box::new(preparation));
    let mut stepped = false;
    let hook_path = path.clone();
    let result: Result<CoordinatorOutcome<()>> =
        acquire_as_coordinator_interruptible(coordinator, &cancelled, move |_| {
            anyhow::ensure!(
                !stepped,
                "coordinator Prepare race ran more than one planner step"
            );
            stepped = true;
            step_rescan_was_already_pending.set(registry::force_coordinator_commit_race_for_tests(
                lost_license,
            )?);
            let hook_cancelled = std::sync::Arc::clone(&hook_cancelled);
            let hook_payload_released_at_notify =
                std::rc::Rc::clone(&hook_payload_released_at_notify);
            let hook_path = hook_path.clone();
            registry::arm_notify_hook_for_tests(move || {
                let released = crate::flock::try_flock(&hook_path, FlockMode::Exclusive)
                    .ok()
                    .flatten();
                hook_payload_released_at_notify.set(released.is_some());
                drop(released);
                hook_cancelled.store(true, Ordering::Release);
            });
            Ok(CoordinatorStep::Prepare {
                final_claim: claim.clone(),
                preparation_claim: claim.clone(),
                preparation: preparation
                    .take()
                    .expect("coordinator Prepare race payload was already consumed"),
            })
        });
    let released_after_commit = crate::flock::try_flock(&path, FlockMode::Exclusive)?;
    let outcome = CoordinatorPayloadNotifyCase {
        commit_terminated: result.is_err(),
        rescan_was_already_pending: rescan_was_already_pending.get(),
        payload_released: released_after_commit.is_some(),
        payload_released_at_notify: payload_released_at_notify.get(),
    };
    drop(released_after_commit);
    successor.finish(None)?;
    Ok(outcome)
}

/// Coordinator commit rejection must preserve the same opaque-payload
/// destruction boundary as ordinary granted callbacks. Exercise both the
/// parked-coordinator Stale return and the incoherent-license error while a
/// rescan edge is already coalesced, for exact Complete and Prepare payloads.
#[cfg(test)]
pub(crate) fn exercise_coordinator_payload_notify_order_for_tests()
-> Result<CoordinatorPayloadNotifyOutcome> {
    Ok(CoordinatorPayloadNotifyOutcome {
        complete_standby: exercise_coordinator_complete_notify_case(201, false)?,
        complete_lost_license: exercise_coordinator_complete_notify_case(202, true)?,
        prepare_standby: exercise_coordinator_prepare_notify_case(203, false)?,
        prepare_lost_license: exercise_coordinator_prepare_notify_case(204, true)?,
    })
}

#[cfg(test)]
pub(crate) struct StaleCoordinatorPreparationCase {
    pub(crate) token_retained_on_retry: bool,
    pub(crate) attempt_released_on_retry: bool,
    pub(crate) aborted: bool,
    pub(crate) token_released_on_exit: bool,
}

#[cfg(test)]
pub(crate) struct StaleCoordinatorPreparationOutcome {
    pub(crate) complete: StaleCoordinatorPreparationCase,
    pub(crate) prepare: StaleCoordinatorPreparationCase,
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum StaleCoordinatorAttempt {
    Complete,
    Prepare,
}

#[cfg(test)]
fn token_only_preparation_for_tests(token: usize) -> Result<super::PreparationPermit> {
    let path = super::permit_lock_path(token);
    let token_fd = crate::flock::try_flock(&path, FlockMode::Exclusive)?
        .ok_or_else(|| anyhow::anyhow!("coordinator retention test token {token} is busy"))?;
    Ok(super::PreparationPermit {
        index: token,
        token_permit: token,
        cpu_permits: Vec::new(),
        memory_permits: Vec::new(),
        permit_fds: vec![(token, AdmissionFlock::from_acquired(token_fd))],
        affinity_lock: None,
        affinity_cpu: 0,
        original_affinity: Vec::new(),
        affinity_constrained: false,
    })
}

#[cfg(test)]
fn exercise_stale_coordinator_preparation_case(
    retained_token: usize,
    claim_cpu: usize,
    attempt_token: usize,
    attempt: StaleCoordinatorAttempt,
) -> Result<StaleCoordinatorPreparationCase> {
    let retained_path = super::permit_lock_path(retained_token);
    let attempt_path = super::permit_lock_path(attempt_token);
    let retained_preparation = token_only_preparation_for_tests(retained_token)?;
    let mut attempt_preparation = match attempt {
        StaleCoordinatorAttempt::Complete => None,
        StaleCoordinatorAttempt::Prepare => {
            Some(Box::new(token_only_preparation_for_tests(attempt_token)?))
        }
    };
    let claim = ClaimSet::new(std::iter::empty(), [claim_cpu], FlockMode::Exclusive);
    let ticket = registry::Ticket::register(claim.clone(), claim.clone(), None)?;
    let coordinator = Box::new(CoordinatorTicket {
        ticket,
        preparation: Some(retained_preparation),
        preparation_watch: None,
    });
    let mut steps = 0usize;
    let mut token_retained_on_retry = false;
    let mut attempt_released_on_retry = matches!(attempt, StaleCoordinatorAttempt::Complete);
    let outcome = acquire_as_coordinator(coordinator, |held| {
        steps += 1;
        if steps == 1 {
            registry::invalidate_coordinator_commit_token_for_tests()?;
            return Ok(match attempt {
                StaleCoordinatorAttempt::Complete => CoordinatorStep::Complete {
                    claim: claim.clone(),
                    value: (),
                },
                StaleCoordinatorAttempt::Prepare => {
                    let preparation = attempt_preparation
                        .take()
                        .expect("stale Prepare payload was already consumed");
                    CoordinatorStep::Prepare {
                        final_claim: claim.clone(),
                        preparation_claim: preparation.claim(),
                        preparation,
                    }
                }
            });
        }
        let reusable = held.clone_reusable_permits()?;
        token_retained_on_retry = reusable.len() == 1 && reusable[0].0 == retained_token;
        if matches!(attempt, StaleCoordinatorAttempt::Prepare) {
            let released = crate::flock::try_flock(&attempt_path, FlockMode::Exclusive)?;
            attempt_released_on_retry = released.is_some();
            drop(released);
        }
        Ok(CoordinatorStep::Abort {
            reason: "stale preparation retention observed".to_owned(),
        })
    })?;
    let aborted = matches!(outcome, CoordinatorOutcome::Aborted { .. });
    let released = crate::flock::try_flock(&retained_path, FlockMode::Exclusive)?;
    let token_released_on_exit = released.is_some();
    drop(released);
    Ok(StaleCoordinatorPreparationCase {
        token_retained_on_retry,
        attempt_released_on_retry,
        aborted,
        token_released_on_exit,
    })
}

/// A stale physical attempt is temporary: retain the private preparation
/// token across its forced fresh planner turn, drop only attempt-local payloads,
/// then release the token when the coordinator exits without committing exact
/// HELD ownership. Exercise both exact Complete and new Prepare attempts.
#[cfg(test)]
pub(crate) fn exercise_stale_coordinator_preparation_retention_for_tests()
-> Result<StaleCoordinatorPreparationOutcome> {
    Ok(StaleCoordinatorPreparationOutcome {
        complete: exercise_stale_coordinator_preparation_case(
            205,
            206,
            207,
            StaleCoordinatorAttempt::Complete,
        )?,
        prepare: exercise_stale_coordinator_preparation_case(
            208,
            209,
            210,
            StaleCoordinatorAttempt::Prepare,
        )?,
    })
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
            (1, Some(registry::ResourceAvailability::Free)),
            (2, Some(registry::ResourceAvailability::SharedHeld)),
            (3, Some(registry::ResourceAvailability::ExclusiveHeld)),
            (4, None),
            (5, Some(registry::ResourceAvailability::Free)),
        ],
        &[
            (1, Some(registry::ResourceAvailability::Free)),
            (2, Some(registry::ResourceAvailability::SharedHeld)),
            (3, Some(registry::ResourceAvailability::ExclusiveHeld)),
            (4, None),
            (5, Some(registry::ResourceAvailability::Free)),
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
        !probe.availability.unobserved(&cpu(3, FlockMode::Shared))?
            && probe.availability.unobserved(&cpu(4, FlockMode::Shared))?
            && !probe.availability.unobserved(&llc(3, FlockMode::Shared))?
            && probe.availability.unobserved(&llc(4, FlockMode::Shared))?,
        "the grant-flow split must separate a genuinely held resource from an unobserved one \
         that `allows` rejects identically"
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
    if let Some(initial) = tickets.first() {
        anyhow::ensure!(
            registry::registration_batch_kept_initial_coordinator_for_tests(initial, &tickets)?,
            "serial registration displaced its live initial coordinator before any ticket was driven",
        );
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
    if let Err(error) = preparation.release_resources_for_exact() {
        // Activation already published the exact record. Ensure every
        // remaining preparation OFD closes before exposing its sole
        // coordinator edge, even on a partial teardown error.
        drop(preparation);
        ticket.notify_after_coordinator_payload_drop();
        return Err(error);
    }
    // Release the preparation token here, at image-resident. The token's
    // designed purpose is bounding concurrent build-phase memory; activation
    // runs only after the build completed and its image was atomically
    // published to the CAS (a complete, sealed object made visible by rename,
    // governed by the CAS's own liveness/GC), so that purpose is over. Post-
    // build residency is CAS-file durability plus a small process footprint,
    // and the guest's memory is bounded by the run claim acquired below —
    // nothing here still needs the token's exclusivity. Holding it through the
    // run-claim wait only coupled preparation admission to run throughput,
    // starving the pool; releasing it now frees the slot for the next entrant
    // while this process waits for its run resources. The one-shot activation
    // path (`try_activate_pending_once`) already drops the token at this point.
    drop(preparation);
    ticket.notify_after_coordinator_payload_drop();
    drive_registered_ticket(ticket, cancelled, &mut try_acquire, None)
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
    loop {
        super::tick_reservation_wait_progress();
        check_interrupted(cancelled)?;
        match check_result(
            ticket.state_or_wait(waiter_crash_recovery_fallback(), cancelled),
            cancelled,
        )? {
            registry::State::Coordinator => {
                return Ok(TicketWork::Coordinator(Box::new(CoordinatorTicket {
                    ticket,
                    preparation,
                    preparation_watch: None,
                })));
            }
            grant_or_replan @ (registry::State::Granted | registry::State::Replan) => {
                let licensed_grant = matches!(grant_or_replan, registry::State::Granted);
                let reusable_permits = preparation
                    .as_ref()
                    .map(super::PreparationPermit::clone_permit_fds)
                    .transpose()?
                    .unwrap_or_default();
                let callback_started = grant_callback_clock();
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
                note_grant_callback_elapsed(licensed_grant, callback_started);
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
                    registry::GrantResult::Requeued | registry::GrantResult::LostGrant => {
                        // Count only a licensed GRANTED grant that failed to
                        // convert to a live HELD claim (registry revoke, stale
                        // prefix, or a lost physical probe) as headline churn. An
                        // unlicensed REPLAN completion that re-plans without
                        // acquiring never held a grant to lose, so it must not
                        // inflate the grants_lost headline. Then requeue either
                        // way.
                        if licensed_grant {
                            crate::vmm::grant_flow::note_grant_lost();
                        }
                        continue;
                    }
                }
            }
            registry::State::Waiting => {}
            registry::State::CoordinatorStandby => {
                // Registration and coordinator-loop ownership are separated
                // by arbitrary userspace scheduling. If the heartbeat lease
                // expires in that interval, the registry safely parks this
                // ticket and transfers progress. Keep waiting for a future
                // election instead of treating the intended takeover state
                // as a malformed handoff.
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

    /// Whether `candidate` overlaps any in-flight grant charge. The
    /// coordinator record is never charged, so no self-exclusion applies.
    /// Same soft-avoid-only contract as [`GrantedProbe::grant_conflicts`].
    pub(crate) fn grant_conflicts(&self, candidate: &ClaimSet) -> Result<bool> {
        self.predecessors
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("coordinator predecessor snapshot is missing"))?
            .grant_conflicts(candidate)
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

    pub(crate) fn clone_reusable_permits(&self) -> Result<Vec<(usize, AdmissionFlock)>> {
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
    ) -> Result<Option<Vec<AdmissionFlock>>> {
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
    ) -> Result<Option<Vec<AdmissionFlock>>> {
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
    ) -> Result<Option<Vec<AdmissionFlock>>> {
        self.probe_complete_inner(candidate, true)
    }

    fn probe_complete_inner(
        &mut self,
        candidate: &[ResourceLock],
        retain_contention: bool,
    ) -> Result<Option<Vec<AdmissionFlock>>> {
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
                TryFlockOutcome::Acquired(fd) => fresh.push(AdmissionFlock::from_acquired(fd)),
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

    /// The proof file for `key`, or `None` when its lockfile does not exist.
    ///
    /// A resource lockfile is created by the first process to flock it
    /// (`open_lockfile` opens `O_CREAT`) and is never unlinked, so a missing
    /// path means no process has ever held that resource. This probe must not
    /// create it: the open is deliberately read-only so its close is
    /// `IN_CLOSE_NOWRITE` and stays out of the coordinator's watch queue (see
    /// [`RealInotifyWake`]), and creating a lockfile per pool index would emit
    /// exactly the `IN_CLOSE_WRITE` storm that filter exists to avoid.
    fn proof_file(&mut self, key: ResourceKey) -> Result<Option<&std::fs::File>> {
        if let std::collections::btree_map::Entry::Vacant(entry) = self.proof_files.entry(key) {
            let path = match key {
                ResourceKey::Llc(index) => super::llc_lock_path(index),
                ResourceKey::Cpu(index) => super::cpu_lock_path(index),
                ResourceKey::Permit(index) => super::permit_lock_path(index),
            };
            match std::fs::OpenOptions::new().read(true).open(&path) {
                Ok(file) => {
                    entry.insert(file);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => return Err(error.into()),
            }
        }
        Ok(Some(&self.proof_files[&key]))
    }

    fn try_proof(&mut self, key: ResourceKey, mode: FlockMode) -> Result<bool> {
        use rustix::fs::{FlockOperation, flock};
        let operation = match mode {
            FlockMode::Exclusive => FlockOperation::NonBlockingLockExclusive,
            FlockMode::Shared => FlockOperation::NonBlockingLockShared,
        };
        let Some(file) = self.proof_file(key)? else {
            // Never created, so never held: every mode is available. There is
            // no proof flock to retain across the publication, so this one
            // observation rests on the pending/serial invalidation instead —
            // an acquirer that takes the resource in the meantime creates the
            // lockfile, publishes BUSY, and clears the pending bit this
            // observation must still match to be applied.
            return Ok(true);
        };
        let acquired = match flock(file, operation) {
            Ok(()) => true,
            Err(error) if error == rustix::io::Errno::WOULDBLOCK => false,
            Err(error) => {
                return Err(std::io::Error::from_raw_os_error(error.raw_os_error()).into());
            }
        };
        if acquired {
            self.proof_locks.insert(key);
        }
        Ok(acquired)
    }

    /// Classify one resource by the strongest lock its proof file still
    /// grants. Only valid where a holder may take the lock shared — see the
    /// permit loop in `observe_with_proofs` for the exclusive-only case.
    fn classify_proof(&mut self, key: ResourceKey) -> Result<registry::ResourceObservation> {
        let availability = if self.try_proof(key, FlockMode::Exclusive)? {
            registry::ResourceAvailability::Free
        } else if self.try_proof(key, FlockMode::Shared)? {
            registry::ResourceAvailability::SharedHeld
        } else {
            registry::ResourceAvailability::ExclusiveHeld
        };
        Ok(registry::ResourceObservation {
            availability,
            sh_resolved: true,
            ex_resolved: true,
        })
    }

    fn observe_with_proofs(
        &mut self,
        request: &registry::ObservationRequest,
    ) -> Result<registry::AvailabilityObservation> {
        let mut observation = registry::AvailabilityObservation::default();
        for &cpu in request.cpus.keys() {
            let observed = self.classify_proof(ResourceKey::Cpu(cpu))?;
            observation.cpus.insert(cpu, observed);
        }
        for &llc in request.llcs.keys() {
            let observed = self.classify_proof(ResourceKey::Llc(llc))?;
            observation.llcs.insert(llc, observed);
        }
        // Permits are only ever flocked exclusive, so a failed exclusive probe
        // means held, not shared-held; routing them through `classify_proof`
        // would report SharedHeld and make a held permit look available.
        for &permit in request.permits.keys() {
            let key = ResourceKey::Permit(permit);
            let availability = if self.try_proof(key, FlockMode::Exclusive)? {
                registry::ResourceAvailability::Free
            } else {
                registry::ResourceAvailability::ExclusiveHeld
            };
            observation.permits.insert(
                permit,
                registry::ResourceObservation {
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

/// Observe one batch that names a resource whose lockfile exists beside one
/// whose lockfile was never created.
///
/// The coordinator's re-observation batch spans the whole aggregate watch, and
/// a registered VM watch names every permit in the pool — but a permit lockfile
/// exists only once some process has flocked that index. A batch that gives up
/// on the first missing path resolves nothing at all, leaving every index it
/// covered UNKNOWN with its pending bit set, which `allows` then rejects
/// exactly like a held resource.
#[cfg(test)]
pub(crate) fn exercise_missing_lockfile_observation_for_tests() -> Result<()> {
    let present_cpu = 0usize;
    let never_created_permit = 7usize;
    crate::flock::materialize(super::cpu_lock_path(present_cpu))?;
    let request = registry::ObservationRequest {
        cpus: std::collections::BTreeMap::from([(present_cpu, (Some(1), Some(1)))]),
        llcs: std::collections::BTreeMap::new(),
        permits: std::collections::BTreeMap::from([(never_created_permit, (None, Some(1)))]),
    };
    let mut observer = HolderObserver::new();
    let observation = observer.observe(&request);
    observer.release_proofs();
    anyhow::ensure!(
        observation
            .cpus
            .get(&present_cpu)
            .is_some_and(|observed| observed.availability == registry::ResourceAvailability::Free),
        "a resolvable probe must survive a batch mate whose lockfile is missing",
    );
    anyhow::ensure!(
        observation
            .permits
            .get(&never_created_permit)
            .is_some_and(|observed| observed.availability == registry::ResourceAvailability::Free),
        "a permit lockfile that was never created names a permit nobody has ever held",
    );
    Ok(())
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

/// Drive a run-acquisition coordinator to its terminal outcome.
///
/// Run acquirers never return `CoordinatorStep::Prepare` from their step
/// closure, so `Prepared` is unreachable for them; the preparation
/// coordinator (`acquire_pending_admission`) handles that outcome itself and
/// must not use this helper. `what` names the acquirer in the panic.
pub(in crate::vmm) fn finish_run_coordinator<T>(
    coordinator: Box<CoordinatorTicket>,
    cancelled: Option<&AtomicBool>,
    step: impl FnMut(&mut HeldLocks) -> Result<CoordinatorStep<T>>,
    what: &'static str,
) -> Result<Acquired<T>> {
    let outcome = match cancelled {
        Some(cancelled) => acquire_as_coordinator_interruptible(coordinator, cancelled, step)?,
        None => acquire_as_coordinator(coordinator, step)?,
    };
    match outcome {
        CoordinatorOutcome::Acquired(acquired) => Ok(acquired),
        CoordinatorOutcome::Prepared(_) => {
            unreachable!("{what} run coordinator prepared a VM intent")
        }
        CoordinatorOutcome::Aborted { reason } => {
            Err(anyhow::Error::new(super::ResourceContention { reason }))
        }
    }
}

/// Render a coordinator's WATCH ENVELOPE compactly for the exit-timing
/// fallback accusation: `cpu=<i>,…,llc=<i>,…,permit=<i>,…`, each set capped so
/// a wide claim cannot produce an unbounded line.
///
/// This is the union over every candidate placement the waiter would accept,
/// not the subset that actually blocked it — the fallback tick knows only that
/// nothing in the envelope woke it. Because the sets are sorted and truncated,
/// an envelope over a wide host renders as its lowest indices; a `<kind>+<n>`
/// part marks what was dropped so the prefix is not misread as a chosen
/// placement.
fn format_watched_resources(claim: &ClaimSet) -> String {
    const MAX_PER_KIND: usize = 8;
    let mut parts = Vec::new();
    let mut render = |label: &str, resources: &BTreeSet<usize>| {
        for resource in resources.iter().take(MAX_PER_KIND) {
            parts.push(format!("{label}={resource}"));
        }
        if let Some(dropped) = resources.len().checked_sub(MAX_PER_KIND).filter(|n| *n > 0) {
            parts.push(format!("{label}+{dropped}"));
        }
    };
    render("cpu", &claim.cpus);
    render("llc", &claim.llcs);
    render("permit", &claim.permits);
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(",")
    }
}

#[cfg(test)]
pub(crate) fn format_watched_resources_for_tests(claim: &ClaimSet) -> String {
    format_watched_resources(claim)
}

/// Nonblocking physical probe of one resource lockfile: `true` iff it is
/// exclusively acquirable right now (no holder at all). Opens `O_RDONLY` so the
/// immediate release on drop emits `IN_CLOSE_NOWRITE` — invisible to the
/// coordinator's own inotify watch, so a probe never self-wakes. The fd is held
/// only for this call (dropped before return), so a peer's observation sees at
/// most a microsecond of transient EX contention, self-corrected on its next
/// nonblocking turn. A missing/unopenable lockfile returns `false`: not provably
/// free, so leave it to the record-based notify path rather than synthesize.
fn resource_flock_is_free(path: &str) -> bool {
    use rustix::fs::{FlockOperation, flock};
    let Ok(file) = std::fs::OpenOptions::new().read(true).open(path) else {
        return false;
    };
    matches!(
        flock(&file, FlockOperation::NonBlockingLockExclusive),
        Ok(())
    )
}

/// Synthesize the missing close events for resources that NEWLY entered the
/// coordinator's watch set this turn and are already physically free. Probes
/// only the passed (new-entrant) indices — bounded per turn, never a full-
/// registry scan — and returns their closes as a [`LockDirEvents`] to merge
/// into the pending batch.
fn probe_newly_watched_free(
    new_cpus: &BTreeSet<usize>,
    new_llcs: &BTreeSet<usize>,
    new_permits: &BTreeSet<usize>,
) -> LockDirEvents {
    let mut synthesized = LockDirEvents::default();
    for &cpu in new_cpus {
        if resource_flock_is_free(&super::cpu_lock_path(cpu)) {
            synthesized.cpu_closes.insert(cpu);
        }
    }
    for &llc in new_llcs {
        if resource_flock_is_free(&super::llc_lock_path(llc)) {
            synthesized.llc_closes.insert(llc);
        }
    }
    for &permit in new_permits {
        if resource_flock_is_free(&super::permit_lock_path(permit)) {
            synthesized.permit_closes.insert(permit);
        }
    }
    synthesized
}

fn acquire_as_coordinator_impl<T>(
    mut coordinator: CoordinatorTicket,
    cancelled: Option<&AtomicBool>,
    mut step: impl FnMut(&mut HeldLocks) -> Result<CoordinatorStep<T>>,
) -> Result<CoordinatorOutcome<T>> {
    let _namespace = coordinator.ticket.enter_namespace();
    // The coordinator loop owns the topology, so hand the registry the
    // preparation-slot token range as opaque budget data. Every grant scan this
    // coordinator drives then bounds preparation grants to the free-slot count
    // in ticket order. `None` on a host without preparation capacity leaves the
    // budget disabled — no preparation intents can exist there to gate.
    coordinator
        .ticket
        .set_preparation_tokens(super::preparation_token_range().ok());
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
    // Re-armed from the schedule snapshot at the top of every turn, before any
    // read. It drives the level probe and the fallback diagnostic only — the
    // wake transport itself is unfiltered.
    let mut watched_resources;
    // Resources already level-probed for pre-watch free-state (see the probe
    // before the blocking wait). Accumulates so each resource is physically
    // probed at most once per coordinator session — the guarantee that bounds
    // the level-trigger recovery and makes it terminating.
    let mut level_checked = ClaimSet::default();
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
            persist_coordinator_wake_stats_if_enabled();
            // Periodic grant-flow image from the scan-running process, on
            // the same cadence and — unlike the old in-scan persist — always
            // outside the registry EX flock, so diagnostic file IO can never
            // extend the global admission critical section.
            crate::vmm::grant_flow::persist_now();
            wait_diagnostic_deadline = diagnostic_now + WAIT_DIAGNOSTIC_INTERVAL;
        }
        check_interrupted(cancelled)?;
        pending_events.merge(check_result(watch.drain(), cancelled)?);
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
        let mut heartbeat_deadline = std::time::Instant::now() + snapshot.heartbeat_due_in;
        let mut deferred_rescan_deadline = snapshot
            .deferred_rescan_due_in
            .map(|due_in| std::time::Instant::now() + due_in);
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
            heartbeat_deadline = std::time::Instant::now() + snapshot.heartbeat_due_in;
            deferred_rescan_deadline = snapshot
                .deferred_rescan_due_in
                .map(|due_in| std::time::Instant::now() + due_in);
            watched_resources = event_watch(snapshot.watch.clone());
            should_step |= snapshot.should_step;
        }
        if heartbeat_deadline <= std::time::Instant::now() {
            // A stream of known-free close events can keep the coordinator
            // out of the blocking wait indefinitely. Renew here as well as in
            // the timeout branch so event backlog cannot make an actively
            // draining coordinator appear stalled.
            let heartbeat = check_result(coordinator.ticket.heartbeat(cancelled), cancelled)?;
            heartbeat_deadline =
                std::time::Instant::now() + registry::COORDINATOR_HEARTBEAT_INTERVAL;
            if heartbeat.parked {
                first = false;
                force_step = true;
                continue;
            }
            if heartbeat.rescan_pending {
                first = false;
                continue;
            }
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
                    let finish = coordinator.ticket.finish_acquired(
                        &claim,
                        commit_token,
                        &markers,
                        cancelled,
                    );
                    match finish {
                        Ok(registry::FinishAcquireResult::Committed(publication)) => {
                            drop(contention);
                            drop(preparation_contention);
                            drop(held.preparation.take());
                            coordinator.ticket.notify_after_coordinator_payload_drop();
                            break CoordinatorOutcome::Acquired(Acquired::tracked(
                                value,
                                publication,
                            ));
                        }
                        Ok(registry::FinishAcquireResult::Stale) => {
                            // An earlier callback changed its reservation after
                            // this planner snapshot. Drop the stale physical fd
                            // set, retain the coordinator ticket and its private
                            // preparation token, and force one fresh planner turn
                            // after the pending prefix scan. The token continues
                            // to bound this resident prepared process until exact
                            // HELD publication or terminal coordinator teardown.
                            drop(value);
                            drop(contention);
                            drop(preparation_contention);
                            coordinator.ticket.notify_after_coordinator_payload_drop();
                            first = false;
                            force_step = true;
                            continue;
                        }
                        Err(error) => {
                            drop(value);
                            drop(contention);
                            drop(preparation_contention);
                            drop(held.preparation.take());
                            coordinator.ticket.notify_after_coordinator_payload_drop();
                            check_interrupted(cancelled)?;
                            return Err(error);
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
                    let contention = held.take_contention();
                    let markers = contention.marker_vec();
                    let commit_token = held.commit_token()?;
                    let finish = coordinator.ticket.finish_preparation(
                        &final_claim,
                        &preparation_claim,
                        commit_token,
                        &markers,
                        cancelled,
                    );
                    match finish {
                        Ok(registry::FinishPreparationResult::Committed(pending_claim)) => {
                            drop(contention);
                            drop(preparation_contention);
                            drop(held.preparation.take());
                            coordinator.ticket.notify_after_coordinator_payload_drop();
                            let pending = pending_admission_from_parts(
                                coordinator.ticket,
                                *preparation,
                                pending_claim,
                            )?;
                            break CoordinatorOutcome::Prepared(Box::new(pending));
                        }
                        Ok(registry::FinishPreparationResult::Stale) => {
                            // The new preparation attempt is stale, but the
                            // private token inherited from exact activation
                            // still bounds this resident prepared process across
                            // the forced fresh planner turn.
                            drop(preparation);
                            drop(contention);
                            drop(preparation_contention);
                            coordinator.ticket.notify_after_coordinator_payload_drop();
                            first = false;
                            force_step = true;
                            continue;
                        }
                        Err(error) => {
                            drop(preparation);
                            drop(contention);
                            drop(preparation_contention);
                            drop(held.preparation.take());
                            coordinator.ticket.notify_after_coordinator_payload_drop();
                            check_interrupted(cancelled)?;
                            return Err(error);
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
                    heartbeat_deadline = std::time::Instant::now() + snapshot.heartbeat_due_in;
                    deferred_rescan_deadline = snapshot
                        .deferred_rescan_due_in
                        .map(|due_in| std::time::Instant::now() + due_in);
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
        // Level-trigger recovery for the edge-triggered inotify watch. `watch`
        // was created (LockDirWatch::new) at coordinator entry; inotify reports
        // only closes that fire while a directory is watched and never the
        // existing free-state at add_watch time. A release that landed BEFORE a
        // resource entered this coordinator's concern — a fresh head-of-line
        // promotion, or a coordinator handoff whose predecessor freed during the
        // election gap — left no queued event, so the classify broadening
        // (which reports every close after the watch exists) still cannot see
        // it, and only the 30s fallback would recover it. Before sleeping,
        // physically probe every resource that NEWLY entered `watched_resources`
        // this turn (relative to `level_checked`); any already-free entrant
        // synthesizes its own close so the coordinator re-scans instead of
        // sleeping on an edge that already fired. This is done once here rather
        // than at each `event_watch` arming because `watched_resources` holds
        // the final armed set for the turn regardless of which arm produced it,
        // and this is the only point that actually blocks.
        //
        // Bounded + terminating: only the new entrants are probed (never an
        // O(all-resources) scan), and `level_checked` accumulates so each
        // resource is probed at most once per session. A synthesized close
        // forces one more schedule turn; if that scan grants nothing,
        // `watched_resources` is unchanged, so the next turn has no new entrants
        // (they are all in `level_checked`) and the loop sleeps normally — no
        // busy loop. A resource genuinely held at probe time is left to the
        // ordinary post-watch inotify close / record-removal notify, both
        // reliable once the watch exists.
        {
            let new_cpus: BTreeSet<usize> = watched_resources
                .cpus
                .difference(&level_checked.cpus)
                .copied()
                .collect();
            let new_llcs: BTreeSet<usize> = watched_resources
                .llcs
                .difference(&level_checked.llcs)
                .copied()
                .collect();
            let new_permits: BTreeSet<usize> = watched_resources
                .permits
                .difference(&level_checked.permits)
                .copied()
                .collect();
            if !new_cpus.is_empty() || !new_llcs.is_empty() || !new_permits.is_empty() {
                level_checked.cpus.extend(new_cpus.iter().copied());
                level_checked.llcs.extend(new_llcs.iter().copied());
                level_checked.permits.extend(new_permits.iter().copied());
                let synthesized = probe_newly_watched_free(&new_cpus, &new_llcs, &new_permits);
                if synthesized.is_actionable() {
                    pending_events.merge(synthesized);
                    continue;
                }
            }
        }
        let retry_interval = watch.semantic_retry_interval(observation_pending);
        let now = std::time::Instant::now();
        retry_deadline = retry_deadline.min(now + retry_interval);
        loop {
            let mut wake_deadline = retry_deadline
                .min(liveness_deadline)
                .min(heartbeat_deadline);
            if let Some(deferred) = deferred_rescan_deadline {
                wake_deadline = wake_deadline.min(deferred);
            }
            let wait_now = std::time::Instant::now();
            let semantic_wait = wake_deadline.saturating_duration_since(wait_now);
            let syscall_wait = super::reservation_wait_progress_poll()
                .map_or(semantic_wait, |poll| semantic_wait.min(poll));
            // A process that has already released its run claim yet is blocking
            // here is the exit-seam suspect. The stamp's own relaxed-load gate
            // keeps pre-release and sink-unset sleeps free.
            crate::vmm::exit_timing::stamp("coordinator_watch_wait");
            match check_result(watch.wait(syscall_wait), cancelled)? {
                Some(events) => {
                    COORDINATOR_EVENT_WAKES.fetch_add(1, Ordering::Relaxed);
                    pending_events.merge(events);
                    break;
                }
                None => {
                    super::tick_reservation_wait_progress();
                    let now = std::time::Instant::now();
                    if now < wake_deadline {
                        // This was only a synchronous progress slice. Remain
                        // inside the same semantic watch wait: do not take the
                        // registry lock or run another schedule pass.
                        continue;
                    }
                    if deferred_rescan_deadline.is_some_and(|deadline| deadline <= now) {
                        // This absolute persisted deadline is never renewed by
                        // ordinary event turns. Re-enter schedule directly so
                        // it atomically promotes and consumes the deferred
                        // edge; do not disguise this semantic flush as a
                        // lightweight heartbeat renewal.
                        break;
                    }
                    if heartbeat_deadline <= now && retry_deadline > now && liveness_deadline > now
                    {
                        // An idle healthy coordinator renews only one header
                        // word. Do not turn this progress deadline into a
                        // semantic retry, resource re-observation, grant scan,
                        // or planner callback. If the ticket was displaced
                        // while asleep, heartbeat() parks it until a later
                        // election; then refresh the complete schedule before
                        // invoking the planner again.
                        let heartbeat =
                            check_result(coordinator.ticket.heartbeat(cancelled), cancelled)?;
                        heartbeat_deadline =
                            std::time::Instant::now() + registry::COORDINATOR_HEARTBEAT_INTERVAL;
                        if heartbeat.parked {
                            force_step = true;
                            break;
                        }
                        if heartbeat.rescan_pending {
                            break;
                        }
                        continue;
                    }
                    // A global liveness deadline is persisted in the registry,
                    // so rapid coordinator handoff cannot postpone it. If that
                    // deadline alone woke us, the next schedule performs the
                    // due sweep without also manufacturing a whole-watch retry.
                    retry_due = liveness_deadline >= retry_deadline;
                    if retry_due {
                        // The semantic retry deadline (not a live inotify edge)
                        // woke the coordinator: it is the missed-event backstop.
                        // `!observation_pending` means the full
                        // `COORDINATOR_WAKE_FALLBACK` interval elapsed rather
                        // than the shorter observation retry — the tick Gate 3
                        // asserts stays ~0 while cells are still queued.
                        COORDINATOR_RETRY_TIMEOUT_WAKES.fetch_add(1, Ordering::Relaxed);
                        if !observation_pending {
                            COORDINATOR_FALLBACK_WAKES.fetch_add(1, Ordering::Relaxed);
                            // Accusation: a full COORDINATOR_WAKE_FALLBACK tick
                            // rather than a live edge means a blocked claim's
                            // release wake was missed. Name the WATCHED
                            // envelope (the tick cannot know which subset of it
                            // blocked) and a holder within it, so every 30s
                            // tick in CI is attributable. Best-effort, one
                            // nonblocking SH read; never perturbs the wait.
                            if !watched_resources.cpus.is_empty()
                                || !watched_resources.llcs.is_empty()
                                || !watched_resources.permits.is_empty()
                            {
                                let blocked_on = format_watched_resources(&watched_resources);
                                let holder =
                                    registry::resource_holders_nonblocking(&watched_resources)
                                        .ok()
                                        .flatten()
                                        .unwrap_or_else(|| "unknown".to_string());
                                crate::vmm::exit_timing::stamp_fallback_block(&blocked_on, &holder);
                            }
                        }
                        retry_deadline = std::time::Instant::now() + retry_interval;
                    }
                    break;
                }
            }
        }
    };
    persist_coordinator_wake_stats_if_enabled();
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
