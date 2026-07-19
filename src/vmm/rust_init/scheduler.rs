//! sched_ext scheduler spawn, attach polling, and bounded child reap.
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;
use crate::scenario::ops::ScxState;

const SYSFS_SCHED_EXT_STATE: &str = "/sys/kernel/sched_ext/state";

/// The guest attach observer has no local time budget.
///
/// Guest clocks cannot distinguish runnable guest work from host
/// descheduling on every supported architecture (notably arm64 without
/// PVTIME). The host attach-attempt watchdog owns the single authoritative
/// max-vCPU service budget and wakes this observer through its cancellation
/// event source. Keeping the guest loop purely event-driven makes the same
/// rule apply at boot and during Attach/Replace/Restart operations.
const SCHED_ATTACH_OBSERVER_CADENCE: std::time::Duration = std::time::Duration::from_millis(50);

/// Retransmission cadence for a boundary whose exact host ACK has not arrived.
///
/// This is not a guest verdict timeout: retries continue indefinitely. The
/// host max-vCPU watchdog remains the only attach budget authority.
const ATTACH_BOUNDARY_RETRY_CADENCE: std::time::Duration = std::time::Duration::from_millis(250);

static NEXT_ATTACH_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq, Default)]
enum AttachControlReaderHealth {
    #[default]
    NotStarted,
    Live,
    Terminal(String),
}

#[derive(Debug, Default)]
struct AttachControlSnapshot {
    active_generation: Option<u64>,
    started_acked: bool,
    finished_acked: bool,
    cancellation: Option<crate::vmm::wire::AttachCancelCause>,
    reader: AttachControlReaderHealth,
}

impl AttachControlSnapshot {
    fn register_generation(&mut self, generation: u64) -> Result<(), String> {
        if generation == 0 {
            return Err("register zero scheduler-attach generation".into());
        }
        if let Some(active) = self.active_generation {
            return Err(format!(
                "scheduler-attach generation {active} is already active; \
                 cannot register concurrent generation {generation}"
            ));
        }
        self.active_generation = Some(generation);
        self.started_acked = false;
        self.finished_acked = false;
        self.cancellation = None;
        Ok(())
    }

    fn clear_generation(&mut self, generation: u64) {
        if self.active_generation != Some(generation) {
            return;
        }
        self.active_generation = None;
        self.started_acked = false;
        self.finished_acked = false;
        self.cancellation = None;
    }

    fn observe_packet(&mut self, packet: crate::vmm::wire::AttachControlPacket) -> bool {
        if self.active_generation != Some(packet.generation()) {
            return false;
        }
        use crate::vmm::wire::AttachControlPacket;
        match packet {
            AttachControlPacket::StartedAck { .. } => {
                self.started_acked = true;
            }
            AttachControlPacket::FinishedAck { .. } => {
                self.finished_acked = true;
            }
            AttachControlPacket::Cancel { cause, .. } => {
                self.cancellation = Some(cause);
            }
        }
        true
    }

    fn view(&self, generation: u64) -> AttachControlView {
        let exact = self.active_generation == Some(generation);
        AttachControlView {
            started_acked: exact && self.started_acked,
            finished_acked: exact && self.finished_acked,
            cancellation: exact.then_some(self.cancellation).flatten(),
            reader_error: match &self.reader {
                AttachControlReaderHealth::NotStarted => {
                    Some("hvc0 scheduler-attach control reader has not started".into())
                }
                AttachControlReaderHealth::Live => None,
                AttachControlReaderHealth::Terminal(reason) => Some(reason.clone()),
            },
        }
    }
}

#[derive(Debug)]
struct AttachControlView {
    started_acked: bool,
    finished_acked: bool,
    cancellation: Option<crate::vmm::wire::AttachCancelCause>,
    reader_error: Option<String>,
}

struct AttachControlState {
    snapshot: std::sync::Mutex<AttachControlSnapshot>,
    eventfd: OwnedFd,
}

static ATTACH_CONTROL_STATE: OnceLock<Result<AttachControlState, String>> = OnceLock::new();

fn attach_control_state() -> Result<&'static AttachControlState, String> {
    ATTACH_CONTROL_STATE
        .get_or_init(|| {
            // SAFETY: eventfd has no pointer arguments. The returned
            // descriptor is wrapped exactly once below.
            let raw = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
            if raw < 0 {
                return Err(format!(
                    "create scheduler-attach cancellation eventfd: {}",
                    std::io::Error::last_os_error()
                ));
            }
            Ok(AttachControlState {
                snapshot: std::sync::Mutex::new(AttachControlSnapshot::default()),
                // SAFETY: eventfd returned this fresh descriptor and ownership
                // transfers to the process-wide attach control state.
                eventfd: unsafe { OwnedFd::from_raw_fd(raw) },
            })
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn wake_attach_control(state: &AttachControlState) -> Result<(), String> {
    let value = 1u64.to_ne_bytes();
    // SAFETY: state.eventfd is a live nonblocking counter-mode eventfd and
    // eventfd writes require exactly one u64.
    let written = unsafe {
        libc::write(
            state.eventfd.as_raw_fd(),
            value.as_ptr().cast::<libc::c_void>(),
            value.len(),
        )
    };
    if written == value.len() as isize {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if error.kind() == std::io::ErrorKind::WouldBlock {
        return Ok(());
    }
    Err(format!("wake scheduler-attach control eventfd: {error}"))
}

fn deliver_scheduler_attach_control(
    packet: crate::vmm::wire::AttachControlPacket,
) -> Result<(), String> {
    if packet.generation() == 0 {
        return Err("deliver zero scheduler-attach control generation".into());
    }
    let state = attach_control_state()?;
    {
        let mut snapshot = state
            .snapshot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        snapshot.observe_packet(packet);
    }
    wake_attach_control(state)
}

fn register_attach_control_generation(generation: u64) -> Result<(), String> {
    let state = attach_control_state()?;
    let mut snapshot = state
        .snapshot
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    snapshot.register_generation(generation)
}

fn clear_attach_control_generation(generation: u64) {
    let Ok(state) = attach_control_state() else {
        return;
    };
    state
        .snapshot
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .clear_generation(generation);
}

pub(crate) fn acknowledge_scheduler_attach_started(generation: u64) -> Result<(), String> {
    deliver_scheduler_attach_control(crate::vmm::wire::AttachControlPacket::StartedAck {
        generation,
    })
}

pub(crate) fn acknowledge_scheduler_attach_finished(generation: u64) -> Result<(), String> {
    deliver_scheduler_attach_control(crate::vmm::wire::AttachControlPacket::FinishedAck {
        generation,
    })
}

/// Deliver a typed host attach-budget cancellation to the exact guest attempt.
pub(crate) fn cancel_scheduler_attach(
    generation: u64,
    cause: crate::vmm::wire::AttachCancelCause,
) -> Result<(), String> {
    deliver_scheduler_attach_control(crate::vmm::wire::AttachControlPacket::Cancel {
        generation,
        cause,
    })
}

/// Publish control-reader health before any scheduler attempt can begin.
pub(crate) fn scheduler_attach_control_reader_started() -> Result<(), String> {
    let state = attach_control_state()?;
    let mut snapshot = state
        .snapshot
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    match &snapshot.reader {
        AttachControlReaderHealth::NotStarted => {
            snapshot.reader = AttachControlReaderHealth::Live;
            Ok(())
        }
        AttachControlReaderHealth::Live => {
            Err("hvc0 scheduler-attach control reader already started".into())
        }
        AttachControlReaderHealth::Terminal(reason) => Err(format!(
            "hvc0 scheduler-attach control reader already terminated: {reason}"
        )),
    }
}

/// Mark the hvc0 control reader terminal and wake every attach ACK/observer
/// wait. The reason is retained so a wake cannot degrade into a generic hang.
pub(crate) fn scheduler_attach_control_reader_terminated(
    reason: impl Into<String>,
) -> Result<(), String> {
    let state = attach_control_state()?;
    {
        let mut snapshot = state
            .snapshot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        snapshot.reader = AttachControlReaderHealth::Terminal(reason.into());
    }
    wake_attach_control(state)
}

fn attach_control_view(generation: u64) -> Result<AttachControlView, String> {
    let state = attach_control_state()?;
    let snapshot = state
        .snapshot
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    Ok(snapshot.view(generation))
}

fn wait_attach_control_wake(fd: &OwnedFd, timeout: std::time::Duration) -> Result<(), String> {
    let timeout_ms = timeout
        .as_millis()
        .saturating_add(u128::from(
            !timeout.subsec_nanos().is_multiple_of(1_000_000),
        ))
        .min(i32::MAX as u128) as libc::c_int;
    let mut pfd = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: fd is owned for the duration of poll and poll only writes
    // `revents`.
    let rc = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
    if rc < 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::Interrupted {
            return Ok(());
        }
        return Err(format!("scheduler-attach control poll failed: {error}"));
    }
    if pfd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
        return Err(format!(
            "scheduler-attach control eventfd returned terminal events {:#x}",
            pfd.revents
        ));
    }
    if pfd.revents & libc::POLLIN == 0 {
        return Ok(());
    }

    let mut value = [0u8; 8];
    // SAFETY: fd is a live counter-mode eventfd and the destination is exactly
    // one u64 wide.
    let read = unsafe {
        libc::read(
            fd.as_raw_fd(),
            value.as_mut_ptr().cast::<libc::c_void>(),
            value.len(),
        )
    };
    if read == value.len() as isize {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if read < 0 && error.kind() == std::io::ErrorKind::WouldBlock {
        return Ok(());
    }
    Err(if read < 0 {
        format!("drain scheduler-attach control eventfd: {error}")
    } else {
        format!("short scheduler-attach control eventfd read: {read}")
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttachTerminalPhase {
    Open,
    Finishing,
    FinishedAcked,
    Settled,
}

impl AttachTerminalPhase {
    const fn drop_retry(self) -> Option<crate::vmm::wire::AttachAttemptTransition> {
        match self {
            Self::Open | Self::Finishing => {
                Some(crate::vmm::wire::AttachAttemptTransition::Finished)
            }
            Self::FinishedAcked => Some(crate::vmm::wire::AttachAttemptTransition::Settled),
            Self::Settled => None,
        }
    }
}

struct SchedulerAttachAttempt {
    generation: u64,
    kind: crate::vmm::wire::AttachAttemptKind,
    control_fd: OwnedFd,
    _bulk_lifecycle_priority: crate::vmm::guest_comms::BulkLifecyclePriorityGuard,
    started_published: bool,
    terminal_phase: AttachTerminalPhase,
    cancellation: Option<crate::vmm::wire::AttachCancelCause>,
}

impl SchedulerAttachAttempt {
    fn begin(kind: crate::vmm::wire::AttachAttemptKind) -> Result<Self, String> {
        let generation = NEXT_ATTACH_GENERATION.fetch_add(1, Ordering::AcqRel);
        if generation == 0 {
            return Err("scheduler-attach generation counter wrapped".into());
        }
        let state = attach_control_state()?;
        let control_fd = state
            .eventfd
            .try_clone()
            .map_err(|error| format!("clone scheduler-attach control eventfd: {error}"))?;
        register_attach_control_generation(generation)?;
        let bulk_lifecycle_priority = crate::vmm::guest_comms::reserve_bulk_lifecycle_priority();
        let mut attempt = Self {
            generation,
            kind,
            control_fd,
            _bulk_lifecycle_priority: bulk_lifecycle_priority,
            started_published: false,
            terminal_phase: AttachTerminalPhase::Open,
            cancellation: None,
        };
        if !attempt.emit(crate::vmm::wire::AttachAttemptTransition::Started) {
            return Err(format!(
                "publish scheduler-attach start for generation {generation}"
            ));
        }
        attempt.started_published = true;
        if let Err(error) = attempt.await_started_ack() {
            let close = attempt.close_terminal();
            return Err(match close {
                Ok(()) => error,
                Err(close_error) => {
                    format!("{error}; scheduler-attach terminal close failed: {close_error}")
                }
            });
        }
        Ok(attempt)
    }

    fn emit(&self, transition: crate::vmm::wire::AttachAttemptTransition) -> bool {
        crate::vmm::guest_comms::send_attach_attempt(crate::vmm::wire::AttachAttemptEvent {
            transition,
            kind: self.kind,
            generation: self.generation,
        })
    }

    fn cancellation_error(&self, cause: crate::vmm::wire::AttachCancelCause) -> String {
        format!(
            "scheduler-attach generation {} cancelled: cause={}",
            self.generation,
            cause.label()
        )
    }

    fn await_started_ack(&mut self) -> Result<(), String> {
        loop {
            let view = attach_control_view(self.generation)?;
            if let Some(cause) = view.cancellation {
                self.cancellation = Some(cause);
                return Err(self.cancellation_error(cause));
            }
            if view.started_acked {
                return Ok(());
            }
            if let Some(reason) = view.reader_error {
                return Err(format!(
                    "scheduler-attach Started ACK reader terminated for generation {}: {reason}",
                    self.generation
                ));
            }

            wait_attach_control_wake(&self.control_fd, ATTACH_BOUNDARY_RETRY_CADENCE)?;
            if !self.emit(crate::vmm::wire::AttachAttemptTransition::Started) {
                tracing::warn!(
                    generation = self.generation,
                    kind = ?self.kind,
                    "retry of scheduler-attach Started boundary was not queued"
                );
            }
        }
    }

    /// Publish Finished and wait event-driven for the exact host ACK. The
    /// attempt deliberately remains in `Finishing` after this returns; the
    /// owner commit must happen before [`Self::settle`].
    fn await_finished_ack(&mut self) -> Result<(), String> {
        match self.terminal_phase {
            AttachTerminalPhase::Open => {
                self.terminal_phase = AttachTerminalPhase::Finishing;
                if !self.emit(crate::vmm::wire::AttachAttemptTransition::Finished) {
                    tracing::warn!(
                        generation = self.generation,
                        kind = ?self.kind,
                        "initial scheduler-attach Finished boundary was not queued"
                    );
                }
            }
            AttachTerminalPhase::Finishing => {}
            AttachTerminalPhase::FinishedAcked => {
                return self
                    .cancellation
                    .map_or(Ok(()), |cause| Err(self.cancellation_error(cause)));
            }
            AttachTerminalPhase::Settled => {
                return Err("scheduler attach attempt already settled".into());
            }
        }

        loop {
            let view = attach_control_view(self.generation)?;
            if let Some(cause) = view.cancellation {
                self.cancellation = Some(cause);
            }
            if view.finished_acked {
                self.terminal_phase = AttachTerminalPhase::FinishedAcked;
                return self
                    .cancellation
                    .map_or(Ok(()), |cause| Err(self.cancellation_error(cause)));
            }
            if let Some(reason) = view.reader_error {
                return Err(format!(
                    "scheduler-attach Finished ACK reader terminated for generation {}: {reason}",
                    self.generation
                ));
            }

            wait_attach_control_wake(&self.control_fd, ATTACH_BOUNDARY_RETRY_CADENCE)?;
            if !self.emit(crate::vmm::wire::AttachAttemptTransition::Finished) {
                tracing::warn!(
                    generation = self.generation,
                    kind = ?self.kind,
                    "retry of scheduler-attach Finished boundary was not queued"
                );
            }
        }
    }

    /// Reliably queue Settled after FinishedAck. Port-1 writes are FIFO, so
    /// every later guest frame is necessarily ordered after this close.
    fn settle(mut self) -> Result<(), String> {
        if self.terminal_phase != AttachTerminalPhase::FinishedAcked {
            return Err(format!(
                "settle scheduler-attach generation {} before Finished ACK",
                self.generation
            ));
        }
        loop {
            if self.emit(crate::vmm::wire::AttachAttemptTransition::Settled) {
                self.terminal_phase = AttachTerminalPhase::Settled;
                return Ok(());
            }
            tracing::warn!(
                generation = self.generation,
                kind = ?self.kind,
                "retrying scheduler-attach Settled boundary"
            );
            wait_attach_control_wake(&self.control_fd, ATTACH_BOUNDARY_RETRY_CADENCE)?;
        }
    }

    fn complete_terminal(mut self, surface_cancellation: bool) -> Result<(), String> {
        let ack = self.await_finished_ack();
        if self.terminal_phase != AttachTerminalPhase::FinishedAcked {
            return ack;
        }
        // FinishedAck is exact, so the only error paired with an acknowledged
        // terminal phase is a typed host cancellation. Failure cleanup already
        // owns that cause and only needs proof the overlay closed; success
        // commit must preserve it and fail.
        let outcome_error = ack.err();
        let settled = self.settle();
        settled?;
        if surface_cancellation {
            outcome_error.map_or(Ok(()), Err)
        } else {
            Ok(())
        }
    }

    /// Complete the three-way terminal protocol on every non-success path:
    /// Finished → exact FinishedAck → Settled. The failure path already owns
    /// any typed cancellation cause, so successful closure itself returns Ok.
    fn close_terminal(self) -> Result<(), String> {
        self.complete_terminal(false)
    }
}

impl Drop for SchedulerAttachAttempt {
    fn drop(&mut self) {
        // The host treats both terminal boundaries as generation-tagged,
        // idempotent state transitions. Drop emits exactly one best-effort
        // retry appropriate to the phase: Finished before its ACK, Settled
        // after the ACK. It never manufactures a new generation or reanchors
        // a duplicate.
        if self.started_published
            && let Some(transition) = self.terminal_phase.drop_retry()
        {
            let sent = self.emit(transition);
            if !sent {
                tracing::error!(
                    generation = self.generation,
                    kind = ?self.kind,
                    ?transition,
                    "failed to publish idempotent scheduler-attach terminal boundary from Drop"
                );
            }
        }
        clear_attach_control_generation(self.generation);
    }
}

/// Outcome of [`poll_scx_attached`].
#[derive(Debug, PartialEq, Eq)]
enum ScxAttachStatus {
    /// sched_ext is registered AND fully enabled: `root/ops` is non-empty
    /// AND `/sys/kernel/sched_ext/state` reads `enabled` — the scheduler
    /// finished `scx_enable` (ops.init ran, every task was initialized, the
    /// kernel set `SCX_ENABLED`), not merely registered. See
    /// `scx_attach_ready`.
    Attached,
    /// The scheduler process exited at any point before attach completed.
    /// The pidfd is watched in the same poll set as sched_ext state, so a
    /// verifier rejection after the former one-second liveness gate is
    /// reported immediately instead of waiting for an unrelated timeout.
    Died,
    /// Attach made a terminal backwards transition (registered/enabling to
    /// disabling/disabled).
    Rejected(AttachDiagnostic),
    /// The host's max-vCPU attach-service budget expired for this exact
    /// attach generation.
    Cancelled(AttachDiagnostic, crate::vmm::wire::AttachCancelCause),
    /// The stable `/sys/kernel/sched_ext/state` node is absent. A missing
    /// dynamic `root/ops` node is normal before registration and never maps
    /// to this variant.
    SysfsAbsent(AttachDiagnostic),
    /// A required observer primitive failed. This is separate from scheduler
    /// attach failure so diagnostics name the broken pidfd/sysfs/inotify/clock
    /// source rather than inventing a sched_ext state.
    ObserverError(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StateObservation {
    Value(ScxState),
    Missing,
    Invalid(String),
    Io(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OpsObservation {
    Missing,
    Empty,
    Named(String),
    Io(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ScxAttachSnapshot {
    state: StateObservation,
    ops: OpsObservation,
}

impl ScxAttachSnapshot {
    fn state_label(&self) -> String {
        match &self.state {
            StateObservation::Value(ScxState::Enabling) => "enabling".into(),
            StateObservation::Value(ScxState::Enabled) => "enabled".into(),
            StateObservation::Value(ScxState::Disabling) => "disabling".into(),
            StateObservation::Value(ScxState::Disabled) => "disabled".into(),
            StateObservation::Missing => "missing".into(),
            StateObservation::Invalid(raw) => format!("invalid({raw:?})"),
            StateObservation::Io(error) => format!("io-error({error})"),
        }
    }

    fn ops_label(&self) -> String {
        match &self.ops {
            OpsObservation::Missing => "missing".into(),
            OpsObservation::Empty => "empty".into(),
            OpsObservation::Named(name) => format!("named({name:?})"),
            OpsObservation::Io(error) => format!("io-error({error})"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum AttachProgress {
    Loading,
    Registered,
    Enabling,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AttachDiagnostic {
    snapshot: ScxAttachSnapshot,
    wall: std::time::Duration,
}

impl AttachDiagnostic {
    fn reason(&self, cause: &str) -> String {
        format!(
            "cause={cause} state={} ops={} wall_ms={}",
            self.snapshot.state_label(),
            self.snapshot.ops_label(),
            self.wall.as_millis(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AttachProbe {
    Ready,
    Pending(AttachProgress),
    Rejected,
    SysfsAbsent,
    ObserverError(String),
}

/// Interpret one coherent sched_ext snapshot.
///
/// `root/ops == ENOENT` is the normal pre-registration state. The stable
/// `state` node, not the lazy scheduler kobject, establishes whether sched_ext
/// exists. A backwards state after registration/enabling is terminal; waiting
/// longer cannot turn a rejected enable into the scheduler we spawned.
fn classify_attach_snapshot(
    snapshot: &ScxAttachSnapshot,
    high_water: AttachProgress,
) -> AttachProbe {
    match &snapshot.state {
        StateObservation::Missing => return AttachProbe::SysfsAbsent,
        StateObservation::Invalid(raw) => {
            return AttachProbe::ObserverError(format!(
                "invalid {SYSFS_SCHED_EXT_STATE} value {raw:?}"
            ));
        }
        StateObservation::Io(error) => {
            return AttachProbe::ObserverError(format!("read {SYSFS_SCHED_EXT_STATE}: {error}"));
        }
        StateObservation::Value(_) => {}
    }
    if let OpsObservation::Io(error) = &snapshot.ops {
        return AttachProbe::ObserverError(format!("read {SYSFS_SCHED_EXT_ROOT_OPS}: {error}"));
    }

    let state = match snapshot.state {
        StateObservation::Value(state) => state,
        _ => unreachable!("terminal state observations returned above"),
    };
    let registered = matches!(snapshot.ops, OpsObservation::Named(_));
    if let OpsObservation::Named(name) = &snapshot.ops
        && scx_attach_ready(name, Some(state))
    {
        return AttachProbe::Ready;
    }

    match state {
        ScxState::Disabling => AttachProbe::Rejected,
        ScxState::Disabled if high_water >= AttachProgress::Enabling => AttachProbe::Rejected,
        ScxState::Disabled if registered => AttachProbe::Pending(AttachProgress::Registered),
        ScxState::Enabling | ScxState::Enabled => AttachProbe::Pending(AttachProgress::Enabling),
        ScxState::Disabled => AttachProbe::Pending(AttachProgress::Loading),
    }
}

trait AttachWaitIo {
    fn child_exited(&mut self) -> Result<bool, String>;
    fn cancelled(&mut self) -> Result<Option<crate::vmm::wire::AttachCancelCause>, String>;
    fn control_reader_error(&mut self) -> Result<Option<String>, String>;
    fn snapshot(&mut self) -> Result<ScxAttachSnapshot, String>;
    fn wall_elapsed(&mut self) -> std::time::Duration;
    fn wait(&mut self, timeout: std::time::Duration) -> Result<(), String>;
}

/// Observer loop split from Linux fd setup so process/state/cancellation
/// precedence is deterministic under unit tests.
fn wait_scx_attach_with(io: &mut impl AttachWaitIo) -> ScxAttachStatus {
    let mut high_water = AttachProgress::Loading;

    loop {
        // Death wins over a stale `enabled` snapshot. Check both around the
        // sysfs reads because the child can exit while PID 1 is reading.
        match io.child_exited() {
            Ok(true) => return ScxAttachStatus::Died,
            Ok(false) => {}
            Err(error) => return ScxAttachStatus::ObserverError(error),
        }
        let snapshot = match io.snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => return ScxAttachStatus::ObserverError(error),
        };
        match io.child_exited() {
            Ok(true) => return ScxAttachStatus::Died,
            Ok(false) => {}
            Err(error) => return ScxAttachStatus::ObserverError(error),
        }

        let diagnostic = AttachDiagnostic {
            snapshot: snapshot.clone(),
            wall: io.wall_elapsed(),
        };

        // The source-of-truth predicate is evaluated before either budget.
        // Thus an attach that completed while PID 1 was descheduled across a
        // boundary is success, not a false timeout.
        match classify_attach_snapshot(&snapshot, high_water) {
            AttachProbe::Ready => return ScxAttachStatus::Attached,
            AttachProbe::Rejected => return ScxAttachStatus::Rejected(diagnostic),
            AttachProbe::SysfsAbsent => {
                return ScxAttachStatus::SysfsAbsent(diagnostic);
            }
            AttachProbe::ObserverError(error) => {
                return ScxAttachStatus::ObserverError(error);
            }
            AttachProbe::Pending(progress) => {
                if progress > high_water {
                    high_water = progress;
                }
            }
        }

        match io.cancelled() {
            Ok(Some(cause)) => return ScxAttachStatus::Cancelled(diagnostic, cause),
            Ok(None) => {}
            Err(error) => return ScxAttachStatus::ObserverError(error),
        }
        match io.control_reader_error() {
            Ok(Some(error)) => return ScxAttachStatus::ObserverError(error),
            Ok(None) => {}
            Err(error) => return ScxAttachStatus::ObserverError(error),
        }

        if let Err(error) = io.wait(SCHED_ATTACH_OBSERVER_CADENCE) {
            return ScxAttachStatus::ObserverError(error);
        }
    }
}

struct LinuxAttachWait {
    attempt: SchedulerAttachAttempt,
    pidfd: OwnedFd,
    state_file: fs::File,
    inotify: nix::sys::inotify::Inotify,
    wall_start: std::time::Instant,
}

#[cfg(test)]
fn open_scheduler_pidfd(pid: u32) -> Result<OwnedFd, ScxAttachStatus> {
    let raw_pidfd =
        unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0u32) as libc::c_int };
    if raw_pidfd < 0 {
        let error = std::io::Error::last_os_error();
        return if error.raw_os_error() == Some(libc::ESRCH) {
            Err(ScxAttachStatus::Died)
        } else {
            Err(ScxAttachStatus::ObserverError(format!(
                "pidfd_open({pid}) failed: {error}"
            )))
        };
    }
    // SAFETY: pidfd_open returned a new descriptor owned by this value.
    Ok(unsafe { OwnedFd::from_raw_fd(raw_pidfd) })
}

impl LinuxAttachWait {
    fn new(
        pidfd: OwnedFd,
        attempt: SchedulerAttachAttempt,
    ) -> Result<Self, (ScxAttachStatus, OwnedFd, SchedulerAttachAttempt)> {
        let state_file = match fs::File::open(SYSFS_SCHED_EXT_STATE) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let snapshot = ScxAttachSnapshot {
                    state: StateObservation::Missing,
                    ops: OpsObservation::Missing,
                };
                return Err((
                    ScxAttachStatus::SysfsAbsent(AttachDiagnostic {
                        snapshot,
                        wall: std::time::Duration::ZERO,
                    }),
                    pidfd,
                    attempt,
                ));
            }
            Err(error) => {
                return Err((
                    ScxAttachStatus::ObserverError(format!(
                        "open {SYSFS_SCHED_EXT_STATE}: {error}"
                    )),
                    pidfd,
                    attempt,
                ));
            }
        };

        use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
        let inotify = match Inotify::init(InitFlags::IN_NONBLOCK | InitFlags::IN_CLOEXEC) {
            Ok(inotify) => inotify,
            Err(error) => {
                return Err((
                    ScxAttachStatus::ObserverError(format!("inotify_init1 for sched_ext: {error}")),
                    pidfd,
                    attempt,
                ));
            }
        };
        if let Err(error) = inotify.add_watch(
            "/sys/kernel/sched_ext/",
            AddWatchFlags::IN_CREATE
                | AddWatchFlags::IN_MOVED_TO
                | AddWatchFlags::IN_DELETE
                | AddWatchFlags::IN_MOVED_FROM,
        ) {
            return Err((
                ScxAttachStatus::ObserverError(format!(
                    "inotify_add_watch(/sys/kernel/sched_ext): {error}"
                )),
                pidfd,
                attempt,
            ));
        }

        Ok(Self {
            attempt,
            pidfd,
            state_file,
            inotify,
            wall_start: std::time::Instant::now(),
        })
    }

    fn into_parts(self) -> (OwnedFd, SchedulerAttachAttempt) {
        (self.pidfd, self.attempt)
    }
}

fn read_state_observation(file: &mut fs::File) -> StateObservation {
    use std::io::{Seek, SeekFrom};
    if let Err(error) = file.seek(SeekFrom::Start(0)) {
        return StateObservation::Io(error.to_string());
    }
    let mut value = String::with_capacity(16);
    if let Err(error) = file.read_to_string(&mut value) {
        return StateObservation::Io(error.to_string());
    }
    match value.trim() {
        "enabling" => StateObservation::Value(ScxState::Enabling),
        "enabled" => StateObservation::Value(ScxState::Enabled),
        "disabling" => StateObservation::Value(ScxState::Disabling),
        "disabled" => StateObservation::Value(ScxState::Disabled),
        raw => StateObservation::Invalid(raw.to_string()),
    }
}

fn read_ops_observation() -> OpsObservation {
    let mut value = String::with_capacity(64);
    match fs::File::open(SYSFS_SCHED_EXT_ROOT_OPS) {
        Ok(mut file) => match file.read_to_string(&mut value) {
            Ok(_) if value.trim().is_empty() => OpsObservation::Empty,
            Ok(_) => OpsObservation::Named(value.trim().to_string()),
            Err(error) => OpsObservation::Io(error.to_string()),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => OpsObservation::Missing,
        Err(error) => OpsObservation::Io(error.to_string()),
    }
}

impl AttachWaitIo for LinuxAttachWait {
    fn child_exited(&mut self) -> Result<bool, String> {
        let mut pfd = libc::pollfd {
            fd: self.pidfd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one live pidfd; zero timeout makes this a nonblocking truth
        // probe.
        let rc = unsafe { libc::poll(&mut pfd, 1, 0) };
        if rc < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                return Ok(false);
            }
            return Err(format!("nonblocking pidfd poll failed: {error}"));
        }
        if pfd.revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
            return Err(format!(
                "pidfd poll returned unexpected events {:#x}",
                pfd.revents
            ));
        }
        Ok(pfd.revents & (libc::POLLIN | libc::POLLHUP) != 0)
    }

    fn cancelled(&mut self) -> Result<Option<crate::vmm::wire::AttachCancelCause>, String> {
        Ok(attach_control_view(self.attempt.generation)?.cancellation)
    }

    fn control_reader_error(&mut self) -> Result<Option<String>, String> {
        Ok(attach_control_view(self.attempt.generation)?.reader_error)
    }

    fn snapshot(&mut self) -> Result<ScxAttachSnapshot, String> {
        Ok(ScxAttachSnapshot {
            state: read_state_observation(&mut self.state_file),
            ops: read_ops_observation(),
        })
    }

    fn wall_elapsed(&mut self) -> std::time::Duration {
        self.wall_start.elapsed()
    }

    fn wait(&mut self, timeout: std::time::Duration) -> Result<(), String> {
        let timeout_ms = timeout
            .as_millis()
            .saturating_add(u128::from(
                !timeout.subsec_nanos().is_multiple_of(1_000_000),
            ))
            .min(i32::MAX as u128) as libc::c_int;
        let mut pfds = [
            libc::pollfd {
                fd: self.pidfd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: self.state_file.as_raw_fd(),
                events: libc::POLLPRI,
                revents: 0,
            },
            libc::pollfd {
                fd: self.inotify.as_fd().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: self.attempt.control_fd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        // SAFETY: all descriptors are owned by self for the duration of
        // poll; poll only writes the revents fields.
        let rc = unsafe { libc::poll(pfds.as_mut_ptr(), pfds.len() as _, timeout_ms) };
        if rc < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                return Ok(());
            }
            return Err(format!("sched_ext attach poll failed: {error}"));
        }
        if pfds[0].revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
            return Err(format!(
                "pidfd returned unexpected poll events {:#x}",
                pfds[0].revents
            ));
        }
        // kernfs conventionally reports a notified attribute as
        // POLLPRI|POLLERR, so POLLERR is a wake bit for this fd, not a broken
        // source. POLLNVAL still means our stable state fd became invalid.
        if pfds[1].revents & (libc::POLLHUP | libc::POLLNVAL) != 0 {
            return Err(format!(
                "sched_ext state returned terminal poll events {:#x}",
                pfds[1].revents
            ));
        }
        if pfds[2].revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            return Err(format!(
                "sched_ext inotify returned unexpected poll events {:#x}",
                pfds[2].revents
            ));
        }
        if pfds[2].revents & libc::POLLIN != 0 {
            match self.inotify.read_events() {
                Ok(_) | Err(nix::errno::Errno::EAGAIN) => {}
                Err(error) => {
                    return Err(format!("drain sched_ext inotify events: {error}"));
                }
            }
        }
        if pfds[3].revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            return Err(format!(
                "scheduler-attach cancellation eventfd returned unexpected poll events {:#x}",
                pfds[3].revents
            ));
        }
        if pfds[3].revents & libc::POLLIN != 0 {
            let mut value = [0u8; 8];
            // SAFETY: control_fd is a live counter-mode eventfd and the
            // destination is exactly one u64 wide.
            let read = unsafe {
                libc::read(
                    self.attempt.control_fd.as_raw_fd(),
                    value.as_mut_ptr().cast::<libc::c_void>(),
                    value.len(),
                )
            };
            if read < 0 && std::io::Error::last_os_error().kind() != std::io::ErrorKind::WouldBlock
            {
                return Err(format!(
                    "drain scheduler-attach control eventfd: {}",
                    std::io::Error::last_os_error()
                ));
            }
        }
        Ok(())
    }
}

/// True when sched_ext is both REGISTERED (`root/ops` content non-empty) and
/// fully ENABLED (`/sys/kernel/sched_ext/state` reads "enabled", i.e. `state`
/// is `Some(ScxState::Enabled)`).
///
/// The kernel adds the `root` kobject — making `root/ops` non-empty — EARLY in
/// `scx_root_enable_workfn` (kernel/sched/ext.c), BEFORE it runs `ops.init`,
/// initializes every task, and performs the final `SCX_ENABLING` ->
/// `SCX_ENABLED` transition (`scx_tryset_enable_state`). Gating attach on
/// `root/ops` alone therefore reports "attached" while the scheduler is still
/// ENABLING, letting the workload (and the host monitor's first reads) run
/// against a half-up scheduler — a silent wrong result. Requiring
/// `state == Some(ScxState::Enabled)` closes that race; it also rejects a
/// registered-but-FAILED enable, where the state goes `ENABLING` ->
/// `DISABLING` -> `DISABLED` and the ops-only check would mis-report attached.
///
/// Pure over its inputs so the predicate is unit-testable without a live scx
/// scheduler; `poll_scx_attached` supplies the live `root/ops` contents and the
/// `scx_state` reading.
pub(crate) fn scx_attach_ready(root_ops_contents: &str, state: Option<ScxState>) -> bool {
    !root_ops_contents.trim().is_empty() && state == Some(ScxState::Enabled)
}

/// Wait for scheduler attach using pidfd + sched_ext event sources.
///
/// Returns [`ScxAttachStatus::Attached`] once `root/ops` is non-empty and
/// `/sys/kernel/sched_ext/state` reads `enabled`. The same poll set watches the
/// scheduler pidfd, the stable state attribute, and creation/removal under the
/// sched_ext directory. A 50ms cadence covers state transitions which kernels
/// do not `sysfs_notify`.
///
/// The sysfs path is built in two steps by the kernel:
/// - `kernel/sched/ext.c` creates the `sched_ext` kset under
///   `kernel_kobj` via `kset_create_and_add("sched_ext", ...)` in
///   the scx init path, giving `/sys/kernel/sched_ext/`.
/// - Each `struct scx_sched` allocation assigns `sch->kobj.kset =
///   scx_kset` then calls `kobject_init_and_add(..., NULL, "root")`
///   (or `"sub-%llu"` when `CONFIG_EXT_SUB_SCHED` and a parent is
///   present), yielding `/sys/kernel/sched_ext/root/`. The `ops`
///   attribute is registered on `scx_ktype` via `scx_sched_groups`;
///   `scx_attr_ops_show` emits `sch->ops.name` through `sysfs_emit`.
///
/// Semantics: a non-empty `root/ops` proves the scheduler completed
/// `scx_alloc_and_add_sched` (scx_sched allocated, `sch->ops = *ops`
/// copied the ops including `name`, kobject registered with the kset),
/// but the kobject add happens BEFORE `ops.init`, per-task init, and the
/// `SCX_ENABLED` transition — so a non-empty `root/ops` does NOT prove
/// the scheduler is live. This poll therefore ALSO requires
/// `/sys/kernel/sched_ext/state` == `enabled`, so a
/// returned `Attached` means the BPF enable completed and the workload
/// can run against a fully-up scheduler. Before that point, pidfd readiness
/// wins over a stale sysfs snapshot and reports the scheduler's actual exit.
///
/// The host's attach-attempt overlay owns the max-vCPU service budget. This
/// observer waits only for semantic sysfs edges, scheduler death, or the exact
/// generation's cancellation edge, so host descheduling under a verifier storm
/// cannot consume a second guest-local timeout.
fn poll_scx_attached(
    pidfd: OwnedFd,
    attempt: SchedulerAttachAttempt,
) -> (ScxAttachStatus, OwnedFd, SchedulerAttachAttempt) {
    let mut io = match LinuxAttachWait::new(pidfd, attempt) {
        Ok(io) => io,
        Err(outcome) => return outcome,
    };
    let status = wait_scx_attach_with(&mut io);
    let (pidfd, attempt) = io.into_parts();
    (status, pidfd, attempt)
}

/// Test-only pin for the standalone pidfd liveness primitive which preceded
/// the unified production attach wait. Keeping this small contract exercised
/// guards SIGCHLD=SIG_IGN behavior without reintroducing the old sequential
/// one-second gate into scheduler startup.
#[cfg(test)]
#[derive(Debug)]
pub(crate) enum StartupStatus {
    Died,
    Alive,
}

#[cfg(test)]
pub(crate) fn poll_startup(
    child: &mut Child,
    interval: std::time::Duration,
    timeout: std::time::Duration,
) -> StartupStatus {
    let pid = child.id();
    // SAFETY: pidfd_open on the child we just spawned, with flags zero.
    let pidfd =
        unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::c_int, 0u32) as libc::c_int };
    if pidfd < 0 {
        return if poll_proc_pid_absent(pid, interval, timeout) {
            StartupStatus::Died
        } else {
            StartupStatus::Alive
        };
    }

    let deadline = std::time::Instant::now() + timeout;
    let result = loop {
        let now = std::time::Instant::now();
        if now >= deadline {
            break if proc_pid_alive(pid) {
                StartupStatus::Alive
            } else {
                StartupStatus::Died
            };
        }
        let remaining = deadline - now;
        let timeout_ms = remaining
            .as_millis()
            .saturating_add(u128::from(
                !remaining.subsec_nanos().is_multiple_of(1_000_000),
            ))
            .min(i32::MAX as u128) as libc::c_int;
        let mut pfd = libc::pollfd {
            fd: pidfd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one live pidfd; poll only writes `revents`.
        let rc = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
        if rc > 0 && pfd.revents & (libc::POLLIN | libc::POLLHUP) != 0 {
            break StartupStatus::Died;
        }
        if rc < 0 && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
            panic!(
                "test pidfd liveness poll failed: {}",
                std::io::Error::last_os_error()
            );
        }
    };
    // SAFETY: this function exclusively owns the raw pidfd.
    unsafe {
        libc::close(pidfd);
    }
    result
}

/// Probe-pipeline drain handles passed to [`start_scheduler`] so the
/// early-bail paths (scheduler Died, not Attached, spawn Err) can
/// flush probe output before calling `force_reboot()`, and to the hvc0
/// graceful-shutdown handler so it can drain on the watchdog
/// soft-shutdown path. The success path's drain runs in
/// [`super::dump::start_pending_sched_exit_monitor`] instead — it sees the scheduler exit
/// notification and waits on `output_done` there. The payload travels
/// the virtio bulk port (the probe thread `println!`s it to stdout,
/// which `redirect_stdio_to_bulk_port` ships over the bulk port), NOT
/// COM2; the host recovers it when it drains that port.
pub(crate) struct ProbeDrain {
    /// Probe-thread stop request. Setting this wakes the probe
    /// thread out of its ring-buffer poll loop; the thread then
    /// emits its payload and sets `output_done`.
    pub(crate) stop: Arc<AtomicBool>,
    /// One-shot signal: set by the probe thread after `println!`ing
    /// `PROBE_OUTPUT_END` (the payload goes to stdout → the bulk-port
    /// forwarder, not COM2). Waited on with a BOUNDED cap on both drain
    /// paths: the early-bail paths cap at `PROBE_DRAIN_GRACE` (30s, the
    /// host's VM-deadline grace — they have no watchdog window of their own);
    /// the hvc0 shutdown drain uses a tighter bound fitting the 3s watchdog
    /// window. On the cap the host recovers the partial payload
    /// (`extract_probe_output`).
    pub(crate) output_done: Arc<crate::sync::Latch>,
}

/// Drain the probe pipeline: signal stop, then wait (bounded by `timeout`) for
/// `output_done`. Called from each early-bail path in [`start_scheduler`] before
/// `force_reboot()` so the probe payload (or the diagnostic-only payload the
/// probe thread emits on a forced stop) reaches the host — emitted over the
/// virtio bulk port by the stdout forwarder, recovered when the host drains it.
/// Returns `true` if `output_done` was observed within `timeout` (or no probe
/// stack was supplied — a no-op), `false` on the cap. Bounded so a hung probe
/// thread cannot wedge the bail until the coarse outer VM wall-clock — the prior
/// unbounded `output_done.wait()` relied on that wall-clock as its only net.
///
/// The early-bail callers pass the auto-repro probe-drain budget
/// ([`crate::test_support::PROBE_DRAIN_GRACE`], 30s) — the same grace the
/// host already adds to the VM deadline for the probe drain, so the guest cap
/// stays within the host's window (over-sized on success: the guest force-reboots
/// the moment the drain completes). The hvc0 soft-shutdown drain
/// (`dump::drain_probe_for_shutdown`) uses its own tighter bound that
/// fits the 3 s watchdog window — a different outer budget, so NOT shared. On the
/// cap the payload may be truncated, but it is loudly absent: the host's
/// `extract_probe_output` recovers the complete events captured before the
/// terminator and persists the partial.
///
/// `drain` is `None` when no probe stack was supplied — every caller is a
/// no-op in that case.
pub(crate) fn drain_probe_pipeline(
    drain: Option<&ProbeDrain>,
    timeout: std::time::Duration,
) -> bool {
    let Some(d) = drain else { return true };
    d.stop.store(true, Ordering::Release);
    let drained = d.output_done.wait_timeout(timeout);
    if !drained {
        tracing::warn!(
            ?timeout,
            "probe drain hit the cap before PROBE_OUTPUT_END; the probe payload may be \
             truncated (the host recovers the partial)"
        );
    }
    drained
}

/// Wait up to `timeout` for `child` to exit (evented via `pidfd_open` +
/// `poll`), then reap it. Does NOT send a signal — callers drive the
/// exit: the crash-grace caller calls this BEFORE `child.kill()` (giving
/// a self-unregistering scheduler a chance to exit on its own within the
/// grace), and the post-grace caller calls it AFTER `child.kill()` (to
/// reap the pending SIGKILL). Returns the real terminal status iff reaped
/// within the window; on timeout the child is left for the VM reboot to
/// reap — teardown must not block unboundedly on a wedged process (see
/// [`SCHED_REAP_TIMEOUT`]).
#[cfg(test)]
pub(crate) fn reap_child_bounded_status(
    child: &mut std::process::Child,
    timeout: std::time::Duration,
) -> Option<std::process::ExitStatus> {
    // Fast path: already exited (e.g. a clean scheduler that took the
    // SIGKILL immediately).
    if let Ok(Some(status)) = child.try_wait() {
        return Some(status);
    }
    match crate::sync::pidfd_poll_exited(child.id() as libc::pid_t, timeout) {
        // Readable => zombie => the reap is now non-blocking.
        crate::sync::PidfdWait::Exited => child.wait().ok(),
        // Timed out: still alive — leave it for the VM reboot.
        crate::sync::PidfdWait::TimedOut => None,
        // pidfd_open failed (ESRCH/gone or env defect): one non-blocking
        // reap attempt, then give up to the reboot.
        crate::sync::PidfdWait::NoPidfd => child.try_wait().ok().flatten(),
    }
}

/// Start the boot scheduler binary if it exists. Thin wrapper around
/// [`spawn_scheduler_from_paths`] supplying the boot-time paths
/// (`/scheduler` + `/sched_args` + `/tmp/sched.log`). The returned object
/// retains the exact attach pidfd for the caller to transfer into the exit
/// monitor before publishing attach success.
///
/// Mid-experiment scheduler-lifecycle Op dispatch
/// ([`Op::AttachScheduler`](crate::scenario::ops::Op::AttachScheduler) /
/// [`Op::ReplaceScheduler`](crate::scenario::ops::Op::ReplaceScheduler))
/// calls [`spawn_scheduler_from_paths`] directly with paths under
/// `/staging/schedulers/<name>/` so swap binaries don't shadow the
/// boot slot.
#[tracing::instrument(skip(probe_drain))]
pub(crate) fn start_scheduler(probe_drain: Option<ProbeDrain>) -> Option<SpawnedScheduler> {
    spawn_scheduler_from_paths("/scheduler", "/sched_args", "/tmp/sched.log", probe_drain)
}

/// Failure modes for [`try_spawn_scheduler`]. Distinct variants
/// per the three observable failure points in the spawn pipeline
/// so callers can branch on the specific outcome — the boot path
/// uniformly responds with dump + lifecycle + force_reboot, while
/// the scheduler-lifecycle Op dispatch path surfaces each variant
/// as an actionable test-failure diagnostic via the per-variant
/// `Display` text.
#[derive(Debug)]
pub(crate) enum SpawnSchedulerError {
    /// `Command::spawn` returned `Err` — fork/exec failed at the
    /// kernel boundary (ENOMEM, EACCES on the binary, EAGAIN from
    /// rlimit). Carries the underlying `io::Error` so the boot
    /// path can synthesize a `SCHED_OUTPUT_START / END`-framed
    /// log payload via `send_synthetic_sched_output_before_terminal`.
    SpawnFailed(std::io::Error),

    /// The shared attach wait observed the process pidfd become readable
    /// before attach completed — typical for a BPF verifier rejection,
    /// argv validation failure, or userspace crash.
    /// `log_path` is the file the spawn helper wrote
    /// stdout+stderr into; callers use it for `dump_sched_output`.
    ///
    /// The provisional process is never published as the current scheduler;
    /// its RAII owner observes the already-terminal exact pidfd.
    StartupDied { log_path: String },

    /// Process remained alive but did not attach. `reason` is a structured
    /// diagnostic containing cause, terminal state/ops observations, and the
    /// wall/service counters. The caller uses `log_path` to surface the
    /// scheduler's own diagnostic output.
    ///
    /// The provisional process is SIGKILLed through its retained pidfd and
    /// reaped within the finite scheduler cleanup allowance; a pathological
    /// uninterruptible straggler is left to the imminent VM reboot.
    NotAttached { reason: String, log_path: String },
}

impl std::fmt::Display for SpawnSchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SpawnFailed(e) => {
                write!(f, "Command::spawn failed: {e}")
            }
            Self::StartupDied { log_path } => {
                write!(
                    f,
                    "scheduler exited before sched_ext attach completed. The pidfd \
                     is watched continuously through BPF open/load/enable, so this \
                     reports the exit at the point it happened. Common causes are \
                     BPF verifier rejection (look for 'libbpf' / 'verifier' in the \
                     log), a scheduler userspace crash, or argv validation failure. \
                     Log captured at {log_path}; no current-process owner was \
                     published before this error surfaced."
                )
            }
            Self::NotAttached { reason, log_path } => {
                write!(
                    f,
                    "scheduler stayed alive but did not complete sched_ext attach: \
                     {reason}. The host charges the attempt in max-vCPU service, so \
                     host descheduling does not consume the allowance. Log captured \
                     at {log_path}; the framework terminated the unattached process \
                     through its pinned pidfd before surfacing this error; no \
                     current-process owner was published."
                )
            }
        }
    }
}

impl std::error::Error for SpawnSchedulerError {}

/// Scheduler spawn whose process identity is pinned from the attach observer
/// through exit-monitor installation.
pub(crate) struct SpawnedScheduler {
    child: Option<Child>,
    pub(crate) log_path: String,
    pidfd: Option<OwnedFd>,
    attach_attempt: Option<SchedulerAttachAttempt>,
    cleanup_wait_exhausted: bool,
}

impl SpawnedScheduler {
    pub(crate) fn child_id(&self) -> Result<u32, String> {
        self.child
            .as_ref()
            .map(Child::id)
            .ok_or_else(|| "scheduler child ownership was already transferred".to_string())
    }

    pub(crate) fn attach_generation(&self) -> Result<u64, String> {
        self.attach_attempt
            .as_ref()
            .map(|attempt| attempt.generation)
            .ok_or_else(|| "scheduler attach attempt already finished".to_string())
    }

    /// Duplicate the already-open pidfd for monitor ownership. This is a
    /// `dup`, not a second `pidfd_open`; `self` retains the original until
    /// monitor installation and completion publication both succeed.
    pub(crate) fn clone_pidfd(&self) -> Result<OwnedFd, String> {
        self.pidfd
            .as_ref()
            .ok_or_else(|| "scheduler pidfd is unavailable".to_string())?
            .try_clone()
            .map_err(|error| format!("duplicate scheduler pidfd for exit monitor: {error}"))
    }

    pub(crate) fn terminate_after_monitor_failure(&mut self) -> Result<(), String> {
        let result = self.terminate_provisional_process();
        let attach_close = self.close_failed_attach_attempt();
        match (result, attach_close) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(cleanup), Ok(())) => Err(cleanup),
            (Ok(()), Err(close)) => Err(format!("scheduler-attach terminal close failed: {close}")),
            (Err(cleanup), Err(close)) => Err(format!(
                "{cleanup}; scheduler-attach terminal close failed: {close}"
            )),
        }
    }

    pub(crate) fn terminate_provisional_process(&mut self) -> Result<(), String> {
        if self.cleanup_wait_exhausted {
            return Err(
                "provisional scheduler already exhausted its bounded exact cleanup wait"
                    .to_string(),
            );
        }
        let result = match (self.child.as_mut(), self.pidfd.as_ref()) {
            (Some(child), Some(pidfd)) => terminate_scheduler_via_pidfd(child, pidfd),
            (None, _) => Ok(()),
            (Some(_), None) => Err("scheduler pidfd is unavailable".to_string()),
        };
        self.cleanup_wait_exhausted = result.is_err();
        if result.is_ok() {
            self.child.take();
            self.pidfd.take();
        }
        result
    }

    /// Revalidate the retained process identity after the monitor thread is
    /// installed and immediately before attach completion is published.
    pub(crate) fn confirm_alive_after_monitor_install(&self) -> Result<(), String> {
        let pidfd = self
            .pidfd
            .as_ref()
            .ok_or_else(|| "scheduler pidfd is unavailable".to_string())?;
        let mut pfd = libc::pollfd {
            fd: pidfd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one retained pidfd and a nonblocking readiness probe.
        let rc = unsafe { libc::poll(&mut pfd, 1, 0) };
        if rc < 0 {
            return Err(format!(
                "post-monitor scheduler pidfd poll: {}",
                std::io::Error::last_os_error()
            ));
        }
        if pfd.revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
            return Err(format!(
                "post-monitor scheduler pidfd returned invalid events {:#x}",
                pfd.revents
            ));
        }
        if pfd.revents & (libc::POLLIN | libc::POLLHUP) != 0 {
            return Err("scheduler exited before attach completion publication".into());
        }
        Ok(())
    }

    /// Publish Finished and wait for its exact host ACK while retaining the
    /// separately service-accounted Finishing overlay. A later owner-commit
    /// step calls [`Self::settle_attach_attempt`] only after publishing the
    /// final process ownership state.
    pub(crate) fn await_attach_finished_ack(&mut self) -> Result<(), String> {
        self.attach_attempt
            .as_mut()
            .ok_or_else(|| "scheduler attach attempt already finished".to_string())?
            .await_finished_ack()
    }

    /// Reliably publish Settled after the exact Finished ACK was consumed.
    pub(crate) fn settle_attach_attempt(&mut self) -> Result<(), String> {
        self.attach_attempt
            .take()
            .ok_or_else(|| "scheduler attach attempt already finished".to_string())?
            .settle()
    }

    /// Close an attach attempt owned by a scheduler path which cannot commit
    /// success (monitor handoff failure, final pidfd failure, or caller abort).
    pub(crate) fn close_failed_attach_attempt(&mut self) -> Result<(), String> {
        match self.attach_attempt.take() {
            Some(attempt) => attempt.close_terminal(),
            None => Ok(()),
        }
    }

    pub(crate) fn finished_ack_consumed(&self) -> bool {
        self.attach_attempt
            .as_ref()
            .is_some_and(|attempt| attempt.terminal_phase == AttachTerminalPhase::FinishedAcked)
    }

    /// Transfer the exact child and original pidfd into the single live owner.
    /// The attach-attempt token deliberately remains in `self` until the new
    /// owner is published and its pending monitor is committed.
    pub(crate) fn take_current_process(
        &mut self,
        scheduler: Option<&'static crate::test_support::SchedulerSpec>,
        monitor: SchedExitStop,
    ) -> Result<CurrentSchedulerProcess, String> {
        let generation = self.attach_generation()?;
        if self.child.is_none() {
            return Err("scheduler child ownership was already transferred".to_string());
        }
        if self.pidfd.is_none() {
            return Err("scheduler pidfd ownership was already transferred".to_string());
        }
        let child = self.child.take().expect("child presence validated above");
        let pidfd = self.pidfd.take().expect("pidfd presence validated above");
        Ok(CurrentSchedulerProcess {
            generation,
            child,
            pidfd,
            log_path: self.log_path.clone(),
            scheduler,
            monitor: Some(monitor),
            drop_reap_exhausted: false,
        })
    }
}

impl Drop for SpawnedScheduler {
    fn drop(&mut self) {
        // A prior cleanup timeout consumed the one finite wait allowance.
        // Preserve exact targeting with a final nonblocking pidfd signal, but
        // never turn RAII disposal into a second teardown-sized stall.
        let cleanup = match (
            self.cleanup_wait_exhausted,
            self.child.as_mut(),
            self.pidfd.as_ref(),
        ) {
            (true, Some(child), Some(pidfd)) => {
                let _ = pidfd_send_signal(pidfd, libc::SIGKILL);
                let _ = child.try_wait();
                Ok(())
            }
            (false, Some(child), Some(pidfd)) => terminate_scheduler_via_pidfd(child, pidfd),
            // Construction publishes Child and pidfd as one race-free pair.
            // Seeing only the Child would mean a future edit split that
            // invariant; never fall back to signaling its numeric pid.
            (_, Some(_), None) => Err(
                "provisional scheduler invariant broken: child exists without original pidfd"
                    .to_string(),
            ),
            (_, None, _) => Ok(()),
        };
        if let Err(error) = cleanup {
            tracing::error!(
                error = %error,
                log_path = self.log_path,
                "failed to clean provisional scheduler process from Drop"
            );
        }
        self.child.take();
        self.pidfd.take();
        // SchedulerAttachAttempt::Drop performs the deliberately bounded,
        // idempotent one-shot emergency boundary. Normal paths explicitly
        // close the full Finished/Ack/Settled protocol before reaching Drop.
        self.attach_attempt.take();
    }
}

/// Live stdio-forwarder threads still draining a scheduler child's
/// pipes, keyed by the log path they append to.
///
/// Before the live-streaming split the child wrote `/tmp/sched.log`
/// directly, so reaping the child alone guaranteed a complete file.
/// With the forwarders in between, the file (and the live stream)
/// trails the pipe until each thread hits EOF —
/// [`wait_sched_forwarders_drained`] lets the dump paths restore the
/// old completeness guarantee by waiting (bounded) on this registry
/// before reading the file.
struct SchedulerForwarder {
    handle: std::thread::JoinHandle<()>,
    drained: Arc<crate::sync::Latch>,
}

impl SchedulerForwarder {
    fn join(self) {
        let _ = self.handle.join();
    }
}

struct ForwarderDrainOnDrop(Arc<crate::sync::Latch>);

impl Drop for ForwarderDrainOnDrop {
    fn drop(&mut self) {
        self.0.set();
    }
}

/// One scheduler-child output pipe captured by a live forwarder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SchedulerOutputStream {
    Stdout,
    Stderr,
}

impl SchedulerOutputStream {
    fn replay_suffix(self) -> &'static str {
        match self {
            Self::Stdout => ".ktstr-stdout-replay",
            Self::Stderr => ".ktstr-stderr-replay",
        }
    }
}

/// Completed per-stream capture retained only when at least one live frame
/// failed to reach the host.
///
/// Appending a suffix to the merged-log path keeps every scheduler invocation
/// in the same lifecycle-managed directory without assuming that the path is
/// valid UTF-8 beyond the `log_path` contract.
pub(crate) fn scheduler_stream_replay_path(
    log_path: &str,
    stream: SchedulerOutputStream,
) -> std::path::PathBuf {
    let mut path = std::ffi::OsString::from(log_path);
    path.push(stream.replay_suffix());
    path.into()
}

fn scheduler_stream_pending_path(
    log_path: &str,
    stream: SchedulerOutputStream,
) -> std::path::PathBuf {
    let mut path = scheduler_stream_replay_path(log_path, stream).into_os_string();
    path.push(".pending");
    path.into()
}

fn remove_scheduler_stream_capture_files(log_path: &str) {
    for stream in [SchedulerOutputStream::Stdout, SchedulerOutputStream::Stderr] {
        let _ = fs::remove_file(scheduler_stream_replay_path(log_path, stream));
        let _ = fs::remove_file(scheduler_stream_pending_path(log_path, stream));
    }
}

/// A forwarder writes every byte to a pending sidecar while attempting its
/// best-effort live publication. At EOF it atomically promotes the file only
/// when live publication was incomplete. A crash, read error, or sidecar write
/// error therefore cannot masquerade as an authoritative complete replay.
struct SchedulerStreamCapture {
    file: fs::File,
    pending_path: std::path::PathBuf,
    replay_path: std::path::PathBuf,
}

impl SchedulerStreamCapture {
    fn create(log_path: &str, stream: SchedulerOutputStream) -> std::io::Result<Self> {
        let pending_path = scheduler_stream_pending_path(log_path, stream);
        let replay_path = scheduler_stream_replay_path(log_path, stream);
        let file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&pending_path)?;
        Ok(Self {
            file,
            pending_path,
            replay_path,
        })
    }

    fn finish(mut self, replay_needed: bool) {
        let flushed = !replay_needed || self.file.flush().is_ok();
        drop(self.file);
        if replay_needed && flushed {
            if let Err(error) = fs::rename(&self.pending_path, &self.replay_path) {
                tracing::warn!(
                    pending_path = %self.pending_path.display(),
                    replay_path = %self.replay_path.display(),
                    %error,
                    "failed to publish complete scheduler-stream replay"
                );
                let _ = fs::remove_file(&self.pending_path);
            }
        } else {
            let _ = fs::remove_file(&self.pending_path);
        }
    }
}

static SCHED_FWD_THREADS: OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, Vec<SchedulerForwarder>>>,
> = OnceLock::new();

pub(crate) const SCHED_FORWARDER_DRAIN_BOUND: std::time::Duration =
    std::time::Duration::from_secs(5);

/// Wait (bounded) for the stdio forwarders registered against
/// `log_path` to finish draining the dead child's pipes, so the merged log
/// file and any completion-needed per-stream replay carry the child's full
/// output. Callers run AFTER the child was reaped, so EOF is already pending
/// on both pipes and the wait is normally an immediate latch observation; the
/// bound only trips if a forwarder wedges
/// (e.g. blocked on virtio backpressure the host never drains). On that
/// bound the handles remain registered and `false` makes the log transaction
/// retryable; it never frames a knowingly incomplete file. No-op success when
/// no forwarder was registered for the path (file-only fallback wiring, or a
/// completed prior drain).
pub(crate) fn wait_sched_forwarders_drained(log_path: &str, deadline: std::time::Instant) -> bool {
    let Some(map) = SCHED_FWD_THREADS.get() else {
        return true;
    };
    let drained = {
        let registry = map
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(forwarders) = registry.get(log_path) else {
            return true;
        };
        forwarders
            .iter()
            .map(|forwarder| forwarder.drained.clone())
            .collect::<Vec<_>>()
    };
    for completion in drained {
        if !completion.wait_until(deadline) {
            tracing::warn!(
                log_path,
                "scheduler stdio forwarders still draining at dump time; \
                 retaining their handles so a later dump can retry"
            );
            return false;
        }
    }

    let handles = map
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(log_path)
        .unwrap_or_default();
    for handle in handles {
        handle.join();
    }
    true
}

/// Forwarder thread for one scheduler-child pipe: read chunks from
/// `read_end` until EOF (child closed the write end), and for each
/// chunk (a) append it to the merged `/tmp/sched.log` clone `log`,
/// (b) append it to a private per-stream pending sidecar, and (c) ship it
/// live to the host via `sender`.
///
/// The `log` append preserves the pre-streaming merged-file view: both
/// forwarders hold clones of the SAME open file description (from
/// `File::try_clone`), so their appends share one offset and interleave
/// in read order exactly as the child's two direct-to-file fds did.
/// The `sender` ship is best-effort (a not-yet-open bulk port or lifecycle
/// reservation drops the chunk). If every live send succeeds, the sidecar is
/// deleted at EOF and terminal publication adds no redundant stream traffic.
/// If any live send fails, a complete sidecar is atomically retained for the
/// terminal replay path. Neither a file write error nor a send failure stops
/// the drain — the thread reads to EOF so the pipe never wedges the child on
/// backpressure.
///
/// The caller registers the returned handle only after both stdout and stderr
/// readers have spawned. This all-or-fallback handoff prevents a child from
/// inheriting a pipe whose only reader failed to start (and then dying from
/// SIGPIPE on its first diagnostic write).
fn spawn_sched_log_forwarder<S>(
    mut read_end: fs::File,
    mut log: Option<fs::File>,
    mut capture: Option<SchedulerStreamCapture>,
    name: &'static str,
    sender: S,
    force_spawn_failure: bool,
) -> std::io::Result<SchedulerForwarder>
where
    S: Fn(&[u8]) -> bool + Send + 'static,
{
    if force_spawn_failure {
        return Err(std::io::Error::other(
            "injected scheduler forwarder spawn failure",
        ));
    }
    let drained = Arc::new(crate::sync::Latch::new());
    let drain_on_drop = ForwarderDrainOnDrop(drained.clone());
    let handle = std::thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let _drain_on_drop = drain_on_drop;
            let mut buf = [0u8; STDIO_CHUNK_BYTES];
            let mut reached_eof = false;
            let mut capture_complete = capture.is_some();
            let mut live_publication_complete = true;
            loop {
                match read_end.read(&mut buf) {
                    Ok(0) => {
                        reached_eof = true;
                        break;
                    }
                    Ok(n) => {
                        if let Some(f) = log.as_mut() {
                            let _ = f.write_all(&buf[..n]);
                        }
                        if capture_complete
                            && capture
                                .as_mut()
                                .is_some_and(|capture| capture.file.write_all(&buf[..n]).is_err())
                        {
                            capture_complete = false;
                        }
                        if !sender(&buf[..n]) {
                            live_publication_complete = false;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
            if let Some(capture) = capture {
                capture.finish(reached_eof && capture_complete && !live_publication_complete);
            }
        })?;
    Ok(SchedulerForwarder { handle, drained })
}

/// Build the `(stdout, stderr)` [`Stdio`] pair for the scheduler child,
/// wiring each stream through a live-streaming forwarder thread (pipe →
/// merged log append + bulk-port ship).
///
/// On ANY pipe-setup failure, falls back to handing the child the
/// merged log-file fds directly (the pre-streaming file-only wiring),
/// or [`Stdio::null`] when no log file could be created. The streaming
/// plumbing must never fail the spawn — a lost live stream degrades to
/// the prior teardown-only dump, it does not abort the scheduler.
fn sched_child_stdio(log_file: Option<&fs::File>, log_path: &str) -> (Stdio, Stdio) {
    sched_child_stdio_inner(log_file, log_path, None)
}

fn sched_child_stdio_inner(
    log_file: Option<&fs::File>,
    log_path: &str,
    fail_forwarder: Option<usize>,
) -> (Stdio, Stdio) {
    if let Some(streamed) = try_stream_sched_stdio_inner(log_file, log_path, fail_forwarder) {
        return streamed;
    }
    // Fallback: file-only (both fds share the merged log's open file
    // description, as before), or null when no log file exists.
    let stdout = match log_file.and_then(|f| f.try_clone().ok()) {
        Some(f) => Stdio::from(f),
        None => Stdio::null(),
    };
    let stderr = match log_file.and_then(|f| f.try_clone().ok()) {
        Some(f) => Stdio::from(f),
        None => Stdio::null(),
    };
    (stdout, stderr)
}

/// Try the streaming wiring: two pipes, two forwarder threads, and the
/// two pipe write ends handed back as the child's `(stdout, stderr)`.
/// Returns `None` if either pipe could not be created so the caller
/// falls back to the file-only wiring.
///
/// The child receives the pipe WRITE ends; after `Command::spawn` the
/// parent's copies are closed by the spawn machinery, so the child is
/// the sole holder and the forwarders' `read`s see EOF on child exit.
fn try_stream_sched_stdio_inner(
    log_file: Option<&fs::File>,
    log_path: &str,
    fail_forwarder: Option<usize>,
) -> Option<(Stdio, Stdio)> {
    // A replay path is authoritative only after its matching forwarder
    // atomically promotes a complete pending capture. Remove stale artifacts
    // before any setup branch can fall back to direct file descriptors.
    remove_scheduler_stream_capture_files(log_path);
    let (stdout_r, stdout_w) = super::modes::make_pipe()?;
    let (stderr_r, stderr_w) = super::modes::make_pipe()?;
    let captures = match (
        SchedulerStreamCapture::create(log_path, SchedulerOutputStream::Stdout),
        SchedulerStreamCapture::create(log_path, SchedulerOutputStream::Stderr),
    ) {
        (Ok(stdout), Ok(stderr)) => (Some(stdout), Some(stderr)),
        (stdout, stderr) => {
            let error = stdout
                .as_ref()
                .err()
                .or_else(|| stderr.as_ref().err())
                .expect("one scheduler-stream capture creation failed");
            tracing::warn!(
                log_path,
                %error,
                "scheduler stream sidecar setup failed; retaining live and merged output"
            );
            drop(stdout);
            drop(stderr);
            remove_scheduler_stream_capture_files(log_path);
            (None, None)
        }
    };
    let (stdout_capture, stderr_capture) = captures;
    // Clone the merged log file per forwarder so both append through
    // the SAME open file description (shared offset → interleave in
    // read order). A clone failure just drops the file append for that
    // stream; the live bulk-port ship still runs.
    let stdout_log = log_file.and_then(|f| f.try_clone().ok());
    let stderr_log = log_file.and_then(|f| f.try_clone().ok());
    let stdout_handle = match spawn_sched_log_forwarder(
        stdout_r,
        stdout_log,
        stdout_capture,
        "ktstr-sched-stdout-fwd",
        crate::vmm::guest_comms::try_send_sched_stdout_chunk,
        fail_forwarder == Some(0),
    ) {
        Ok(handle) => handle,
        Err(error) => {
            tracing::warn!(
                log_path,
                error = %error,
                "scheduler stdout forwarder failed to spawn; using file-only stdio"
            );
            remove_scheduler_stream_capture_files(log_path);
            return None;
        }
    };
    let stderr_handle = match spawn_sched_log_forwarder(
        stderr_r,
        stderr_log,
        stderr_capture,
        "ktstr-sched-stderr-fwd",
        crate::vmm::guest_comms::try_send_sched_stderr_chunk,
        fail_forwarder == Some(1),
    ) {
        Ok(handle) => handle,
        Err(error) => {
            // Closing both write ends produces EOF for the one live reader
            // before joining it. Only then may the caller clone the regular
            // file descriptors for its fallback child wiring.
            drop(stdout_w);
            drop(stderr_w);
            stdout_handle.join();
            remove_scheduler_stream_capture_files(log_path);
            tracing::warn!(
                log_path,
                error = %error,
                "scheduler stderr forwarder failed to spawn; using file-only stdio"
            );
            return None;
        }
    };
    SCHED_FWD_THREADS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry(log_path.to_string())
        .or_default()
        .extend([stdout_handle, stderr_handle]);
    Some((Stdio::from(stdout_w), Stdio::from(stderr_w)))
}

/// Pure spawn helper — runs the spawn → poll-startup → poll-attached
/// pipeline and returns a `Result` so callers can choose how to
/// handle each failure mode. The boot path uniformly responds with
/// `dump_sched_output_before_terminal` + `send_lifecycle` +
/// `force_reboot`; the scheduler-lifecycle Op dispatch surfaces each
/// `Err` variant as a typed test-failure rather than rebooting the VM.
///
/// `Ok(None)` means the binary file is missing — the caller decides
/// whether that is a degenerate-but-acceptable state (boot path:
/// no scheduler configured) or a hard error (Op dispatch:
/// staging pipeline mis-packed). Per the [`SpawnSchedulerError`]
/// doc, the three failure variants each carry the context the
/// boot path needs (log_path for dump, io::Error for spawn
/// failure) so the wrapper can preserve the prior boot-failure
/// semantics without changes to host-side diagnostics.
pub(crate) fn try_spawn_scheduler(
    binary_path: &str,
    args_path: &str,
    log_path: &str,
    attach_kind: crate::vmm::wire::AttachAttemptKind,
) -> Result<Option<SpawnedScheduler>, SpawnSchedulerError> {
    if !Path::new(binary_path).exists() {
        return Ok(None);
    }

    let sched_args = fs::read_to_string(args_path)
        .unwrap_or_default()
        .trim()
        .to_string();
    let args: Vec<&str> = if sched_args.is_empty() {
        vec![]
    } else {
        sched_args.split_whitespace().collect()
    };

    let log_file = fs::File::create(log_path).ok();
    // Wire the child's stdout/stderr through bulk-port forwarder threads
    // that BOTH append each read chunk to the merged `/tmp/sched.log`
    // AND ship it live to the host (MSG_TYPE_SCHED_STDOUT /
    // MSG_TYPE_SCHED_STDERR). The live stream means the scheduler's
    // output survives a watchdog timeout that never reaches the
    // teardown `dump_sched_output`; the file append keeps the merged
    // view `dump_sched_output`, the boot-failure dumps, and the
    // `SCHED_OUTPUT_*` marker parsing all read unchanged.
    let (stdout, stderr) = sched_child_stdio(log_file.as_ref(), log_path);

    // Build RUST_LOG for the scheduler: append libbpf noise
    // suppression to whatever the guest already has. libbpf
    // emits debug/info messages through the `log` crate via
    // scx_utils::libbpf_logger; raising its threshold to warn
    // keeps scheduler output readable.
    let sched_rust_log = match std::env::var("RUST_LOG") {
        Ok(existing) => format!("{existing},scx_utils::libbpf_logger=warn"),
        Err(_) => "info,scx_utils::libbpf_logger=warn".to_string(),
    };

    let attempt = SchedulerAttachAttempt::begin(attach_kind).map_err(|error| {
        SpawnSchedulerError::SpawnFailed(std::io::Error::other(format!(
            "prepare scheduler attach observer: {error}"
        )))
    })?;

    let mut command = Command::new(binary_path);
    command
        .args(&args)
        .env("RUST_LOG", &sched_rust_log)
        .stdout(stdout)
        .stderr(stderr);
    let (child, pidfd) = match spawn_with_pidfd(&mut command) {
        Ok(process) => process,
        Err(error) => {
            let kind = error.kind();
            let close = attempt.close_terminal();
            let error = match close {
                Ok(()) => error,
                Err(close_error) => std::io::Error::new(
                    kind,
                    format!("{error}; scheduler-attach terminal close failed: {close_error}"),
                ),
            };
            return Err(SpawnSchedulerError::SpawnFailed(error));
        }
    };

    // From this edge onward every early return and unwind is owned by the
    // same provisional RAII record. It cannot publish into the live owner
    // until attach, FinishedAck, and monitor commit all succeed.
    let mut provisional = SpawnedScheduler {
        child: Some(child),
        log_path: log_path.to_string(),
        pidfd: Some(pidfd),
        attach_attempt: Some(attempt),
        cleanup_wait_exhausted: false,
    };
    let child_id = provisional
        .child_id()
        .expect("new provisional spawn owns its child");

    let attempt = provisional
        .attach_attempt
        .take()
        .expect("provisional spawn owns attach attempt");
    let pidfd = provisional
        .pidfd
        .take()
        .expect("race-free spawn owns its original pidfd");
    let (status, pidfd, attempt) = poll_scx_attached(pidfd, attempt);
    // Restore the observer's exact descriptor to the provisional owner before
    // branching. Every terminal/error return then has the same RAII cleanup
    // identity, including scheduler-death and bounded-cleanup failures.
    provisional.pidfd = Some(pidfd);
    match status {
        ScxAttachStatus::Attached => {
            provisional.attach_attempt = Some(attempt);
            Ok(Some(provisional))
        }
        ScxAttachStatus::Died => {
            if let Err(error) = attempt.close_terminal() {
                tracing::error!(
                    error = %error,
                    "failed to close scheduler-attach generation after scheduler death"
                );
            }
            Err(SpawnSchedulerError::StartupDied {
                log_path: log_path.to_string(),
            })
        }
        status => {
            let mut reason = match status {
                ScxAttachStatus::Rejected(diagnostic) => diagnostic.reason("rejected-enable"),
                ScxAttachStatus::Cancelled(diagnostic, cause) => diagnostic.reason(cause.label()),
                ScxAttachStatus::SysfsAbsent(diagnostic) => diagnostic.reason("sysfs-absent"),
                ScxAttachStatus::ObserverError(error) => {
                    format!("cause=observer-error detail={error:?}")
                }
                ScxAttachStatus::Attached | ScxAttachStatus::Died => unreachable!(),
            };
            tracing::warn!(
                pid = child_id,
                reason,
                "scheduler did not complete sched_ext attach"
            );

            // The process is still alive but not attached. Signal and wait on
            // the exact pidfd retained by the observer so PID reuse can never
            // redirect cleanup to another process.
            if let Err(cleanup_error) = provisional.terminate_provisional_process() {
                tracing::error!(
                    pid = child_id,
                    error = %cleanup_error,
                    "failed to prove unattached scheduler cleanup through pidfd"
                );
                reason.push_str(&format!(" cleanup_error={cleanup_error:?}"));
            }
            if let Err(close_error) = attempt.close_terminal() {
                reason.push_str(&format!(" attach_close_error={close_error:?}"));
            }
            Err(SpawnSchedulerError::NotAttached {
                reason,
                log_path: log_path.to_string(),
            })
        }
    }
}

/// Spawn a scheduler binary with BOOT failure semantics: on any
/// failure mode reported by [`try_spawn_scheduler`], dump the
/// scheduler log via the bulk data port, signal
/// `LifecyclePhase::SchedulerDied` / `SchedulerNotAttached` over
/// guest_comms, send an exit code, drain the probe pipeline, and
/// call [`force_reboot`]. Used by the boot wrapper
/// [`start_scheduler`] where a missing or broken scheduler is a
/// terminal condition.
///
/// Mid-experiment scheduler-lifecycle Op dispatch should call
/// [`try_spawn_scheduler`] directly and surface failures as
/// typed test-failure diagnostics instead of rebooting the VM.
///
/// `Ok(None)` from `try_spawn_scheduler` (binary missing) returns
/// `None` — preserves the prior contract where an absent
/// `/scheduler` is "no scheduler configured" rather than a
/// failure.
///
/// `probe_drain` is consumed only on the force_reboot paths; the
/// Ok-success path leaves it for the caller to drop normally.
///
/// # Path contract
///
/// `binary_path` is checked for existence inline; an absent
/// binary returns `(None, None)`. `args_path` may be absent (an
/// empty file produces empty argv). `log_path` is created or
/// truncated at function entry.
#[tracing::instrument(skip(probe_drain), fields(binary = %binary_path))]
pub(crate) fn spawn_scheduler_from_paths(
    binary_path: &str,
    args_path: &str,
    log_path: &str,
    probe_drain: Option<ProbeDrain>,
) -> Option<SpawnedScheduler> {
    match try_spawn_scheduler(
        binary_path,
        args_path,
        log_path,
        crate::vmm::wire::AttachAttemptKind::Boot,
    ) {
        Ok(None) => None,
        // The caller installs the exit monitor from `spawned.pidfd` before
        // emitting SchedulerAttached. This wrapper deliberately cannot
        // publish a success that is not yet continuously monitored.
        Ok(Some(spawned)) => Some(spawned),
        Err(SpawnSchedulerError::SpawnFailed(e)) => {
            tracing::error!(err = %e, "ktstr-init: spawn scheduler failed");
            // Synthesize a minimal sched-log payload framed by
            // the existing SCHED_OUTPUT_START/END markers so the
            // host's `parse_sched_output` returns the spawn-
            // failure diagnostic exactly as the prior COM2 path
            // did.
            let _ = send_synthetic_sched_output_before_terminal(&format!("failed to spawn: {e}"));
            crate::vmm::guest_comms::send_lifecycle(
                crate::vmm::wire::LifecyclePhase::SchedulerDied,
                "",
            );
            crate::vmm::guest_comms::send_exit(1);
            // Drain the probe pipeline before reboot so PROBE_OUTPUT_END
            // is emitted over the bulk port ahead of force_reboot.
            // No-op when no probe stack was supplied.
            drain_probe_pipeline(probe_drain.as_ref(), crate::test_support::PROBE_DRAIN_GRACE);
            force_reboot();
        }
        Err(SpawnSchedulerError::StartupDied { log_path }) => {
            // Scheduler died during startup. Dump the scheduler
            // log via the bulk data port — the
            // SCHED_OUTPUT_START / SCHED_OUTPUT_END markers
            // travel verbatim inside the chunk bytes so the
            // host's `parse_sched_output` walker keeps working
            // unchanged.
            let _ = dump_sched_output_before_terminal(&log_path);
            crate::vmm::guest_comms::send_lifecycle(
                crate::vmm::wire::LifecyclePhase::SchedulerDied,
                "",
            );
            crate::vmm::guest_comms::send_exit(1);
            drain_probe_pipeline(probe_drain.as_ref(), crate::test_support::PROBE_DRAIN_GRACE);
            force_reboot();
        }
        Err(SpawnSchedulerError::NotAttached { reason, log_path }) => {
            let _ = dump_sched_output_before_terminal(&log_path);
            crate::vmm::guest_comms::send_lifecycle(
                crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
                &reason,
            );
            crate::vmm::guest_comms::send_exit(1);
            drain_probe_pipeline(probe_drain.as_ref(), crate::test_support::PROBE_DRAIN_GRACE);
            force_reboot();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn forwarder_retains_replay_only_after_a_live_publication_drop() {
        let dir = tempfile::tempdir().expect("create scheduler replay tempdir");
        let payload = b"one pipe-read chunk\n";

        for live_send_succeeds in [true, false] {
            let log_path = dir
                .path()
                .join(if live_send_succeeds {
                    "all-live.log"
                } else {
                    "one-drop.log"
                })
                .to_string_lossy()
                .into_owned();
            remove_scheduler_stream_capture_files(&log_path);
            let capture = SchedulerStreamCapture::create(&log_path, SchedulerOutputStream::Stdout)
                .expect("create pending stdout capture");
            let replay_path =
                scheduler_stream_replay_path(&log_path, SchedulerOutputStream::Stdout);
            let pending_path =
                scheduler_stream_pending_path(&log_path, SchedulerOutputStream::Stdout);
            let (read_end, mut write_end) =
                super::super::modes::make_pipe().expect("create forwarder pipe");
            let calls = Arc::new(AtomicUsize::new(0));
            let sender_calls = calls.clone();
            let forwarder = spawn_sched_log_forwarder(
                read_end,
                None,
                Some(capture),
                "ktstr-sched-replay-test",
                move |_| {
                    let call = sender_calls.fetch_add(1, Ordering::SeqCst);
                    live_send_succeeds || call != 0
                },
                false,
            )
            .expect("spawn scheduler test forwarder");
            write_end
                .write_all(payload)
                .expect("write one atomic scheduler chunk");
            drop(write_end);
            forwarder.join();

            assert!(
                calls.load(Ordering::SeqCst) >= 1,
                "the scheduler pipe must exercise at least one live frame"
            );
            assert!(
                !pending_path.exists(),
                "a completed forwarder must atomically resolve its pending capture"
            );
            if live_send_succeeds {
                assert!(
                    !replay_path.exists(),
                    "fully published live output must add no terminal replay"
                );
            } else {
                assert_eq!(
                    fs::read(&replay_path).expect("read retained stdout replay"),
                    payload,
                    "one dropped live frame must retain the complete stream"
                );
            }
        }
    }

    #[test]
    fn either_forwarder_spawn_failure_uses_file_fds_without_sigpipe() {
        let dir = tempfile::tempdir().expect("create scheduler stdio tempdir");
        for failed_reader in [0, 1] {
            let path = dir.path().join(format!("sched-{failed_reader}.log"));
            let log = fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(&path)
                .expect("open scheduler log");
            let path_string = path.to_string_lossy().into_owned();
            let (stdout, stderr) =
                sched_child_stdio_inner(Some(&log), &path_string, Some(failed_reader));

            let status = Command::new("/bin/sh")
                .args(["-c", "printf 'fallback survived\\n' >&2"])
                .stdout(stdout)
                .stderr(stderr)
                .status()
                .expect("spawn fallback-wired child");
            assert!(
                status.success(),
                "reader {failed_reader} spawn failure left pipe-backed child stdio: {status}"
            );
            drop(log);
            assert_eq!(
                fs::read(&path).expect("read scheduler fallback log"),
                b"fallback survived\n"
            );
            assert!(
                SCHED_FWD_THREADS
                    .get()
                    .and_then(|threads| {
                        threads
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .get(&path_string)
                            .map(Vec::len)
                    })
                    .is_none(),
                "partial forwarder setup must not publish a drain registry entry"
            );
            for stream in [SchedulerOutputStream::Stdout, SchedulerOutputStream::Stderr] {
                assert!(
                    !scheduler_stream_replay_path(&path_string, stream).exists()
                        && !scheduler_stream_pending_path(&path_string, stream).exists(),
                    "file-only fallback must not expose an authoritative stream capture"
                );
            }
        }
    }

    #[test]
    fn forwarder_drain_wait_uses_completion_event_and_shared_deadline() {
        let path = "/tmp/ktstr-forwarder-latch-regression.log";
        let completion = Arc::new(crate::sync::Latch::new());
        let thread_completion = completion.clone();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let _signal_on_exit = ForwarderDrainOnDrop(thread_completion);
            release_rx.recv().expect("release synthetic forwarder");
        });
        SCHED_FWD_THREADS
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                path.to_string(),
                vec![SchedulerForwarder {
                    handle,
                    drained: completion,
                }],
            );

        assert!(
            !wait_sched_forwarders_drained(path, std::time::Instant::now()),
            "an expired shared deadline must return without polling"
        );
        assert!(
            SCHED_FWD_THREADS
                .get()
                .expect("forwarder registry initialized")
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains_key(path),
            "a timeout must retain the join handle for a later retry"
        );

        release_tx.send(()).expect("release synthetic forwarder");
        assert!(wait_sched_forwarders_drained(
            path,
            std::time::Instant::now() + std::time::Duration::from_secs(2)
        ));
        assert!(
            !SCHED_FWD_THREADS
                .get()
                .expect("forwarder registry initialized")
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains_key(path),
            "completed forwarders must be joined and removed"
        );
    }

    #[test]
    fn attach_control_snapshot_matches_exact_generation_and_retains_cause() {
        let mut snapshot = AttachControlSnapshot {
            reader: AttachControlReaderHealth::Live,
            ..AttachControlSnapshot::default()
        };
        snapshot.register_generation(7).unwrap();
        assert!(
            snapshot.observe_packet(crate::vmm::wire::AttachControlPacket::StartedAck {
                generation: 7,
            })
        );
        assert!(
            snapshot.observe_packet(crate::vmm::wire::AttachControlPacket::FinishedAck {
                generation: 7,
            })
        );
        assert!(
            snapshot.observe_packet(crate::vmm::wire::AttachControlPacket::Cancel {
                generation: 7,
                cause: crate::vmm::wire::AttachCancelCause::ServiceBudget,
            })
        );

        let exact = snapshot.view(7);
        assert!(exact.started_acked);
        assert!(exact.finished_acked);
        assert_eq!(
            exact.cancellation,
            Some(crate::vmm::wire::AttachCancelCause::ServiceBudget)
        );
        assert_eq!(exact.reader_error, None);

        let stale = snapshot.view(6);
        assert!(!stale.started_acked);
        assert!(!stale.finished_acked);
        assert_eq!(stale.cancellation, None);

        assert_eq!(
            snapshot.register_generation(8),
            Err(
                "scheduler-attach generation 7 is already active; cannot register concurrent \
                 generation 8"
                    .into()
            )
        );
    }

    #[test]
    fn wrong_generation_control_is_discarded_not_cached_for_a_future_attempt() {
        let mut snapshot = AttachControlSnapshot {
            reader: AttachControlReaderHealth::Live,
            ..AttachControlSnapshot::default()
        };
        snapshot.register_generation(20).unwrap();
        assert!(
            !snapshot.observe_packet(crate::vmm::wire::AttachControlPacket::StartedAck {
                generation: 21
            })
        );
        assert!(
            !snapshot.observe_packet(crate::vmm::wire::AttachControlPacket::Cancel {
                generation: 19,
                cause: crate::vmm::wire::AttachCancelCause::ServiceBudget,
            })
        );
        assert!(!snapshot.view(20).started_acked);
        assert_eq!(snapshot.view(20).cancellation, None);

        snapshot.clear_generation(20);
        snapshot.register_generation(21).unwrap();
        assert!(
            !snapshot.view(21).started_acked,
            "future ACK from generation 20 was cached into generation 21"
        );
        assert_eq!(snapshot.view(21).cancellation, None);
    }

    #[test]
    fn attach_control_terminal_health_is_visible_to_every_wait() {
        let snapshot = AttachControlSnapshot {
            reader: AttachControlReaderHealth::Terminal("hvc0 control stream disconnected".into()),
            ..AttachControlSnapshot::default()
        };
        assert_eq!(
            snapshot.view(1).reader_error,
            Some("hvc0 control stream disconnected".into())
        );
    }

    #[test]
    fn attach_drop_retry_is_phase_specific_and_idempotent() {
        use crate::vmm::wire::AttachAttemptTransition;
        assert_eq!(
            AttachTerminalPhase::Open.drop_retry(),
            Some(AttachAttemptTransition::Finished)
        );
        assert_eq!(
            AttachTerminalPhase::Finishing.drop_retry(),
            Some(AttachAttemptTransition::Finished)
        );
        assert_eq!(
            AttachTerminalPhase::FinishedAcked.drop_retry(),
            Some(AttachAttemptTransition::Settled)
        );
        assert_eq!(AttachTerminalPhase::Settled.drop_retry(), None);
    }

    fn snapshot(state: ScxState, ops: OpsObservation) -> ScxAttachSnapshot {
        ScxAttachSnapshot {
            state: StateObservation::Value(state),
            ops,
        }
    }

    fn loading_snapshot() -> ScxAttachSnapshot {
        snapshot(ScxState::Disabled, OpsObservation::Missing)
    }

    fn enabling_snapshot() -> ScxAttachSnapshot {
        snapshot(ScxState::Enabling, OpsObservation::Named("test_ops".into()))
    }

    fn attached_snapshot() -> ScxAttachSnapshot {
        snapshot(ScxState::Enabled, OpsObservation::Named("test_ops".into()))
    }

    #[derive(Clone)]
    struct FakeAttachFrame {
        wall: std::time::Duration,
        exited: bool,
        cancelled: Option<crate::vmm::wire::AttachCancelCause>,
        control_reader_error: Option<String>,
        snapshot: ScxAttachSnapshot,
    }

    impl FakeAttachFrame {
        fn new(wall_s: f64, exited: bool, cancelled: bool, snapshot: ScxAttachSnapshot) -> Self {
            Self {
                wall: std::time::Duration::from_secs_f64(wall_s),
                exited,
                cancelled: cancelled.then_some(crate::vmm::wire::AttachCancelCause::ServiceBudget),
                control_reader_error: None,
                snapshot,
            }
        }

        fn with_control_reader_error(mut self, error: impl Into<String>) -> Self {
            self.control_reader_error = Some(error.into());
            self
        }
    }

    struct FakeAttachIo {
        frames: Vec<FakeAttachFrame>,
        index: usize,
        wait_error: Option<String>,
    }

    impl FakeAttachIo {
        fn new(frames: Vec<FakeAttachFrame>) -> Self {
            assert!(!frames.is_empty());
            Self {
                frames,
                index: 0,
                wait_error: None,
            }
        }

        fn frame(&self) -> &FakeAttachFrame {
            &self.frames[self.index]
        }
    }

    impl AttachWaitIo for FakeAttachIo {
        fn child_exited(&mut self) -> Result<bool, String> {
            Ok(self.frame().exited)
        }

        fn cancelled(&mut self) -> Result<Option<crate::vmm::wire::AttachCancelCause>, String> {
            Ok(self.frame().cancelled)
        }

        fn control_reader_error(&mut self) -> Result<Option<String>, String> {
            Ok(self.frame().control_reader_error.clone())
        }

        fn snapshot(&mut self) -> Result<ScxAttachSnapshot, String> {
            Ok(self.frame().snapshot.clone())
        }

        fn wall_elapsed(&mut self) -> std::time::Duration {
            self.frame().wall
        }

        fn wait(&mut self, _timeout: std::time::Duration) -> Result<(), String> {
            if let Some(error) = self.wait_error.take() {
                return Err(error);
            }
            assert!(
                self.index + 1 < self.frames.len(),
                "fake attach wait exhausted without a terminal frame"
            );
            self.index += 1;
            Ok(())
        }
    }

    /// Guest wall time never consumes attach allowance. The host's
    /// max-single-vCPU overlay is the only budget authority.
    #[test]
    fn attach_wait_ignores_guest_wall_dilation() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, false, false, loading_snapshot()),
            FakeAttachFrame::new(120.0, false, false, loading_snapshot()),
            FakeAttachFrame::new(600.0, false, false, enabling_snapshot()),
            FakeAttachFrame::new(601.0, false, false, attached_snapshot()),
        ]);
        assert_eq!(wait_scx_attach_with(&mut io), ScxAttachStatus::Attached);
    }

    /// Readiness is sampled before cancellation, so an attach confirmed on the
    /// same host-service edge wins and the host may clear the attempt normally.
    #[test]
    fn attach_ready_wins_over_simultaneous_cancel() {
        let mut io = FakeAttachIo::new(vec![FakeAttachFrame::new(
            35.0,
            false,
            true,
            attached_snapshot(),
        )]);
        assert_eq!(wait_scx_attach_with(&mut io), ScxAttachStatus::Attached);
    }

    /// The pidfd remains in the wait through the whole attach, not only the
    /// former one-second startup window.
    #[test]
    fn scheduler_death_after_old_startup_gate_is_immediate() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, false, false, loading_snapshot()),
            FakeAttachFrame::new(2.0, true, false, loading_snapshot()),
        ]);
        assert_eq!(wait_scx_attach_with(&mut io), ScxAttachStatus::Died);
    }

    /// A dead child wins over a stale `enabled` sysfs snapshot.
    #[test]
    fn simultaneous_death_and_ready_prefers_death() {
        let mut io = FakeAttachIo::new(vec![FakeAttachFrame::new(
            2.0,
            true,
            true,
            attached_snapshot(),
        )]);
        assert_eq!(wait_scx_attach_with(&mut io), ScxAttachStatus::Died);
    }

    /// The dynamic root kobject is intentionally absent before registration;
    /// a readable stable state node proves sched_ext exists.
    #[test]
    fn missing_root_ops_is_loading_not_sysfs_absent() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, false, false, loading_snapshot()),
            FakeAttachFrame::new(40.0, false, true, loading_snapshot()),
        ]);
        let status = wait_scx_attach_with(&mut io);
        assert!(
            matches!(
                status,
                ScxAttachStatus::Cancelled(_, crate::vmm::wire::AttachCancelCause::ServiceBudget)
            ),
            "missing lazy root/ops was misclassified: {status:?}"
        );
    }

    /// Only absence of the stable sched_ext state node means sched_ext sysfs is
    /// absent.
    #[test]
    fn missing_state_is_sysfs_absent() {
        let mut io = FakeAttachIo::new(vec![FakeAttachFrame::new(
            0.0,
            false,
            false,
            ScxAttachSnapshot {
                state: StateObservation::Missing,
                ops: OpsObservation::Missing,
            },
        )]);
        assert!(matches!(
            wait_scx_attach_with(&mut io),
            ScxAttachStatus::SysfsAbsent(_)
        ));
    }

    /// A host cancellation is generation-specific and terminates a pending
    /// semantic attach without inventing a guest clock verdict.
    #[test]
    fn host_service_cancel_terminates_pending_attach() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, false, false, loading_snapshot()),
            FakeAttachFrame::new(400.0, false, false, enabling_snapshot()),
            FakeAttachFrame::new(401.0, false, true, enabling_snapshot()),
        ]);
        assert!(matches!(
            wait_scx_attach_with(&mut io),
            ScxAttachStatus::Cancelled(_, crate::vmm::wire::AttachCancelCause::ServiceBudget)
        ));
    }

    #[test]
    fn pidfd_cleanup_kills_only_the_pinned_scheduler() {
        let mut scheduler = Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn scheduler stand-in");
        let mut bystander = Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn bystander");
        let pidfd = open_scheduler_pidfd(scheduler.id()).expect("open scheduler pidfd");

        terminate_scheduler_via_pidfd(&mut scheduler, &pidfd)
            .expect("terminate scheduler through its pidfd");
        assert!(
            bystander.try_wait().expect("query bystander").is_none(),
            "pidfd cleanup affected a process outside the pinned identity"
        );

        bystander.kill().expect("kill bystander");
        bystander.wait().expect("reap bystander");
    }

    #[test]
    fn event_source_error_is_not_a_timeout() {
        let mut io = FakeAttachIo::new(vec![FakeAttachFrame::new(
            0.0,
            false,
            false,
            loading_snapshot(),
        )]);
        io.wait_error = Some("poll source failed".into());
        assert_eq!(
            wait_scx_attach_with(&mut io),
            ScxAttachStatus::ObserverError("poll source failed".into())
        );
    }

    #[test]
    fn terminal_control_reader_wakes_pending_attach_as_observer_error() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, false, false, loading_snapshot())
                .with_control_reader_error("hvc0 control reader disconnected"),
        ]);
        assert_eq!(
            wait_scx_attach_with(&mut io),
            ScxAttachStatus::ObserverError("hvc0 control reader disconnected".into())
        );
    }

    #[test]
    fn backwards_enable_transition_is_rejected_immediately() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, false, false, enabling_snapshot()),
            FakeAttachFrame::new(1.0, false, false, loading_snapshot()),
        ]);
        assert!(matches!(
            wait_scx_attach_with(&mut io),
            ScxAttachStatus::Rejected(_)
        ));
    }

    /// `scx_attach_ready` (the attach predicate) is true ONLY when the
    /// scheduler is both registered (non-empty `root/ops`) AND fully enabled
    /// (`state == Enabled`). A registered-but-still-ENABLING scheduler — the
    /// race window where the kernel added `root/ops` but has not yet set
    /// `SCX_ENABLED` — is NOT reported attached; nor is a failed enable
    /// (`Disabling`/`Disabled`) or a missing state read. Exhaustive over the
    /// state axis and the empty/non-empty `root/ops` axis.
    #[test]
    fn scx_attach_ready_requires_registered_and_enabled() {
        // Registered + enabled -> the one true case.
        assert!(scx_attach_ready("ktstr_ops\n", Some(ScxState::Enabled)));
        // Registered but mid-enable / failed-enable / absent state -> NOT ready
        // (the race + failed-enable cases the old ops-only predicate mis-reported
        // as attached).
        assert!(!scx_attach_ready("ktstr_ops\n", Some(ScxState::Enabling)));
        assert!(!scx_attach_ready("ktstr_ops\n", Some(ScxState::Disabling)));
        assert!(!scx_attach_ready("ktstr_ops\n", Some(ScxState::Disabled)));
        assert!(!scx_attach_ready("ktstr_ops\n", None));
        // Not registered (empty / whitespace-only root/ops) -> NOT ready even
        // when state reads enabled (a stale/foreign enable without our ops).
        assert!(!scx_attach_ready("", Some(ScxState::Enabled)));
        assert!(!scx_attach_ready("   \n", Some(ScxState::Enabled)));
        assert!(!scx_attach_ready("", None));
    }

    fn probe_drain() -> ProbeDrain {
        ProbeDrain {
            stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            output_done: std::sync::Arc::new(crate::sync::Latch::new()),
        }
    }

    /// The early-bail drain is BOUNDED: an unset `output_done` (a hung probe
    /// thread) returns `false` at the cap instead of blocking forever (the prior
    /// `output_done.wait()` was unbounded). Asserts `stop` was
    /// signaled and the call returned well inside the cap (a short test cap so
    /// the test is fast).
    #[test]
    fn drain_probe_pipeline_caps_on_unset_output_done() {
        let pd = probe_drain();
        let t0 = std::time::Instant::now();
        let drained = drain_probe_pipeline(Some(&pd), std::time::Duration::from_millis(20));
        assert!(
            !drained,
            "unset output_done -> cap-hit -> false (not a hang)"
        );
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(2),
            "returned at the cap, did not wait unbounded"
        );
        assert!(
            pd.stop.load(std::sync::atomic::Ordering::Acquire),
            "stop signaled to wake the probe thread"
        );
    }

    /// `output_done` already set (PROBE_OUTPUT_END emitted) -> returns `true`
    /// immediately, well inside the cap: the drain-until-marker (early-return)
    /// half is preserved. A 60s cap we never approach pins "returns on the
    /// marker, not the cap".
    #[test]
    fn drain_probe_pipeline_returns_early_on_output_done() {
        let pd = probe_drain();
        pd.output_done.set();
        let t0 = std::time::Instant::now();
        let drained = drain_probe_pipeline(Some(&pd), std::time::Duration::from_secs(60));
        assert!(drained, "set output_done -> true");
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(1),
            "returned on the marker, not after the 60s cap"
        );
    }

    /// No probe stack -> no-op, returns `true` (every caller is a no-op).
    #[test]
    fn drain_probe_pipeline_none_is_noop() {
        assert!(drain_probe_pipeline(
            None,
            std::time::Duration::from_millis(1)
        ));
    }
}
