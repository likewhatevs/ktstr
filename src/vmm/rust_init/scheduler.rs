//! sched_ext scheduler spawn, attach polling, and bounded child reap.
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;
use crate::scenario::ops::ScxState;

const SYSFS_SCHED_EXT_STATE: &str = "/sys/kernel/sched_ext/state";

/// Scheduler CPU service allowed between semantic attach-progress edges.
///
/// This is deliberately CPU time, not PID 1 wall time. Under a verifier
/// storm a scheduler can be runnable but receive a small fraction of a host
/// CPU for many wall seconds; charging that descheduled time made the old
/// fixed 10-second window reject healthy cells. Thirty-five seconds matches
/// the host Attach-stage service allowance while remaining far above a
/// normal BPF open/load/attach.
const SCHED_ATTACH_SERVICE_BUDGET: std::time::Duration = std::time::Duration::from_secs(35);

/// Last-resort guard for a scheduler that consumes no service and makes no
/// progress (for example, a permanently blocked userspace loader). The host's
/// phase-aware watchdog remains authoritative; this guard is intentionally
/// much wider than the old 10-second wall window.
const SCHED_ATTACH_WALL_GUARD: std::time::Duration = std::time::Duration::from_secs(90);

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
    /// No semantic progress consumed the allowed scheduler CPU service.
    ServiceBudgetExceeded(AttachDiagnostic),
    /// Last-resort wall guard fired while service remained below its budget.
    WallGuardExceeded(AttachDiagnostic),
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
    service: std::time::Duration,
}

impl AttachDiagnostic {
    fn reason(&self, cause: &str) -> String {
        format!(
            "cause={cause} state={} ops={} wall_ms={} service_ms={}",
            self.snapshot.state_label(),
            self.snapshot.ops_label(),
            self.wall.as_millis(),
            self.service.as_millis(),
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

#[derive(Debug, Clone, Copy)]
struct AttachWaitPolicy {
    cadence: std::time::Duration,
    service_budget: std::time::Duration,
    wall_guard: std::time::Duration,
}

trait AttachWaitIo {
    fn child_exited(&mut self) -> Result<bool, String>;
    fn snapshot(&mut self) -> Result<ScxAttachSnapshot, String>;
    fn service_elapsed(&mut self) -> Result<std::time::Duration, String>;
    fn wall_elapsed(&mut self) -> std::time::Duration;
    fn wait(&mut self, timeout: std::time::Duration) -> Result<(), String>;
}

/// Policy loop split from Linux fd setup so deadline/process/state precedence is
/// deterministic under unit tests.
fn wait_scx_attach_with(io: &mut impl AttachWaitIo, policy: AttachWaitPolicy) -> ScxAttachStatus {
    let mut high_water = AttachProgress::Loading;
    let mut service_checkpoint = match io.service_elapsed() {
        Ok(service) => service,
        Err(error) => return ScxAttachStatus::ObserverError(error),
    };
    let mut last_service = service_checkpoint;

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

        let wall = io.wall_elapsed();
        let service = match io.service_elapsed() {
            Ok(service) => service,
            Err(error) => {
                // A clock lookup can race process exit. Re-check pidfd before
                // classifying it as an observer failure.
                return match io.child_exited() {
                    Ok(true) => ScxAttachStatus::Died,
                    Ok(false) => ScxAttachStatus::ObserverError(error),
                    Err(pidfd_error) => ScxAttachStatus::ObserverError(format!(
                        "{error}; pidfd recheck also failed: {pidfd_error}"
                    )),
                };
            }
        };
        if service < last_service {
            return ScxAttachStatus::ObserverError(format!(
                "scheduler CPU service clock regressed from {:?} to {:?}",
                last_service, service
            ));
        }
        last_service = service;
        let diagnostic = AttachDiagnostic {
            snapshot: snapshot.clone(),
            wall,
            service,
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
                    service_checkpoint = service;
                }
            }
        }

        if service.saturating_sub(service_checkpoint) >= policy.service_budget {
            return ScxAttachStatus::ServiceBudgetExceeded(diagnostic);
        }
        if wall >= policy.wall_guard {
            return ScxAttachStatus::WallGuardExceeded(diagnostic);
        }

        let remaining_wall = policy.wall_guard.saturating_sub(wall);
        if let Err(error) = io.wait(policy.cadence.min(remaining_wall)) {
            return ScxAttachStatus::ObserverError(error);
        }
    }
}

struct LinuxAttachWait {
    pidfd: OwnedFd,
    state_file: fs::File,
    inotify: nix::sys::inotify::Inotify,
    service_clock: libc::clockid_t,
    service_start: std::time::Duration,
    wall_start: std::time::Instant,
}

impl LinuxAttachWait {
    fn new(pid: u32) -> Result<Self, ScxAttachStatus> {
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
        let pidfd = unsafe { OwnedFd::from_raw_fd(raw_pidfd) };

        let state_file = match fs::File::open(SYSFS_SCHED_EXT_STATE) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let snapshot = ScxAttachSnapshot {
                    state: StateObservation::Missing,
                    ops: OpsObservation::Missing,
                };
                return Err(ScxAttachStatus::SysfsAbsent(AttachDiagnostic {
                    snapshot,
                    wall: std::time::Duration::ZERO,
                    service: std::time::Duration::ZERO,
                }));
            }
            Err(error) => {
                return Err(ScxAttachStatus::ObserverError(format!(
                    "open {SYSFS_SCHED_EXT_STATE}: {error}"
                )));
            }
        };

        use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
        let inotify =
            Inotify::init(InitFlags::IN_NONBLOCK | InitFlags::IN_CLOEXEC).map_err(|error| {
                ScxAttachStatus::ObserverError(format!("inotify_init1 for sched_ext: {error}"))
            })?;
        inotify
            .add_watch(
                "/sys/kernel/sched_ext/",
                AddWatchFlags::IN_CREATE
                    | AddWatchFlags::IN_MOVED_TO
                    | AddWatchFlags::IN_DELETE
                    | AddWatchFlags::IN_MOVED_FROM,
            )
            .map_err(|error| {
                ScxAttachStatus::ObserverError(format!(
                    "inotify_add_watch(/sys/kernel/sched_ext): {error}"
                ))
            })?;

        let mut service_clock: libc::clockid_t = 0;
        // SAFETY: `service_clock` is a valid output pointer and pid identifies
        // the child just spawned by this process.
        let rc = unsafe { libc::clock_getcpuclockid(pid as libc::pid_t, &mut service_clock) };
        if rc != 0 {
            return if rc == libc::ESRCH {
                Err(ScxAttachStatus::Died)
            } else {
                Err(ScxAttachStatus::ObserverError(format!(
                    "clock_getcpuclockid({pid}) failed: {}",
                    std::io::Error::from_raw_os_error(rc)
                )))
            };
        }
        let service_start = match read_cpu_clock(service_clock) {
            Ok(service) => service,
            Err(error) if error.raw_os_error() == Some(libc::ESRCH) => {
                return Err(ScxAttachStatus::Died);
            }
            Err(error) => {
                return Err(ScxAttachStatus::ObserverError(format!(
                    "read scheduler CPU service clock: {error}"
                )));
            }
        };

        Ok(Self {
            pidfd,
            state_file,
            inotify,
            service_clock,
            service_start,
            wall_start: std::time::Instant::now(),
        })
    }
}

fn read_cpu_clock(clock: libc::clockid_t) -> Result<std::time::Duration, std::io::Error> {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `value` is a valid output pointer and `clock` came from
    // clock_getcpuclockid.
    if unsafe { libc::clock_gettime(clock, &mut value) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    if value.tv_sec < 0 || value.tv_nsec < 0 {
        return Err(std::io::Error::other(
            "scheduler CPU clock returned a negative value",
        ));
    }
    Ok(std::time::Duration::new(
        value.tv_sec as u64,
        value.tv_nsec as u32,
    ))
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

    fn snapshot(&mut self) -> Result<ScxAttachSnapshot, String> {
        Ok(ScxAttachSnapshot {
            state: read_state_observation(&mut self.state_file),
            ops: read_ops_observation(),
        })
    }

    fn service_elapsed(&mut self) -> Result<std::time::Duration, String> {
        let now = read_cpu_clock(self.service_clock)
            .map_err(|error| format!("read scheduler CPU service clock: {error}"))?;
        now.checked_sub(self.service_start)
            .ok_or_else(|| "scheduler CPU service clock regressed below its baseline".into())
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
        ];
        // SAFETY: all three descriptors are owned by self for the duration of
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
        if pfds[1].revents & libc::POLLNVAL != 0 {
            return Err(format!(
                "sched_ext state returned invalid-fd poll events {:#x}",
                pfds[1].revents
            ));
        }
        if pfds[2].revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
            return Err(format!(
                "sched_ext inotify returned unexpected poll events {:#x}",
                pfds[2].revents
            ));
        }
        if pfds[2].revents & libc::POLLIN != 0 {
            self.inotify
                .read_events()
                .map_err(|error| format!("drain sched_ext inotify events: {error}"))?;
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
/// The progress budget is denominated in scheduler process CPU service, not
/// guest wall time. Host descheduling under a 52-cell storm therefore cannot
/// consume the attach allowance.
fn poll_scx_attached(pid: u32) -> ScxAttachStatus {
    let mut io = match LinuxAttachWait::new(pid) {
        Ok(io) => io,
        Err(status) => return status,
    };
    wait_scx_attach_with(
        &mut io,
        AttachWaitPolicy {
            cadence: std::time::Duration::from_millis(50),
            service_budget: SCHED_ATTACH_SERVICE_BUDGET,
            wall_guard: SCHED_ATTACH_WALL_GUARD,
        },
    )
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
/// [`start_sched_exit_monitor`] instead — it sees the scheduler exit
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
fn drain_probe_pipeline(drain: Option<&ProbeDrain>, timeout: std::time::Duration) -> bool {
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
/// (`/scheduler` + `/sched_args` + `/tmp/sched.log`). Returns the
/// child process and the path to its log file.
///
/// Mid-experiment scheduler-lifecycle Op dispatch
/// ([`Op::AttachScheduler`](crate::scenario::ops::Op::AttachScheduler) /
/// [`Op::ReplaceScheduler`](crate::scenario::ops::Op::ReplaceScheduler))
/// calls [`spawn_scheduler_from_paths`] directly with paths under
/// `/staging/schedulers/<name>/` so swap binaries don't shadow the
/// boot slot.
#[tracing::instrument(skip(probe_drain))]
pub(crate) fn start_scheduler(probe_drain: Option<ProbeDrain>) -> (Option<Child>, Option<String>) {
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
    /// log payload via `send_sched_log_text`.
    SpawnFailed(std::io::Error),

    /// The shared attach wait observed the process pidfd become readable
    /// before attach completed — typical for a BPF verifier rejection,
    /// argv validation failure, or userspace crash.
    /// `log_path` is the file the spawn helper wrote
    /// stdout+stderr into; callers use it for `dump_sched_output`.
    ///
    /// **Post-mortem state guarantee.** [`try_spawn_scheduler`]
    /// returns this variant only AFTER clearing [`SCHED_PID`] to 0
    /// (the dead pid was published optimistically at spawn so the
    /// sched_exit_monitor caller path could install against a known
    /// id; the StartupDied branch never gets that far so the spawn
    /// helper owns the rollback). The dead child is auto-reaped by
    /// the kernel because PID 1 runs SIGCHLD=`SIG_IGN`; the attach wait
    /// observes pidfd POLLIN and does not reap. No manual
    /// `wait`/`try_wait` is required by the caller.
    StartupDied { log_path: String },

    /// Process remained alive but did not attach. `reason` is a structured
    /// diagnostic containing cause, terminal state/ops observations, and the
    /// wall/service counters. The caller uses `log_path` to surface the
    /// scheduler's own diagnostic output.
    ///
    /// **Post-mortem state guarantee.** [`try_spawn_scheduler`]
    /// returns this variant only AFTER SIGKILLing the orphan
    /// process (which is alive but not bound to scx, so it would
    /// otherwise keep running and could late-bind on the next
    /// scheduler attempt) and waiting on it via `child.wait()` to
    /// reap the zombie, plus clearing [`SCHED_PID`] to 0. No manual
    /// cleanup required by the caller.
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
                     Log captured at {log_path}; SCHED_PID was cleared before this \
                     error surfaced."
                )
            }
            Self::NotAttached { reason, log_path } => {
                write!(
                    f,
                    "scheduler stayed alive but did not complete sched_ext attach: \
                     {reason}. Attach progress is charged in scheduler CPU service, \
                     so host descheduling does not consume the allowance. Log \
                     captured at {log_path}; the framework terminated the \
                     unattached process and cleared SCHED_PID before surfacing this \
                     error."
                )
            }
        }
    }
}

impl std::error::Error for SpawnSchedulerError {}

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
static SCHED_FWD_THREADS: OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, Vec<std::thread::JoinHandle<()>>>>,
> = OnceLock::new();

/// Wait (bounded) for the stdio forwarders registered against
/// `log_path` to finish draining the dead child's pipes, so the merged
/// log file and the live `SchedStdout`/`SchedStderr` streams both carry
/// the child's complete output. Callers run AFTER the child was
/// reaped, so EOF is already pending on both pipes and the wait is
/// normally a few polls; the bound only trips if a forwarder wedges
/// (e.g. blocked on virtio backpressure the host never drains), in
/// which case the dump proceeds with whatever reached the file — the
/// pre-streaming behavior for a torn-down VM. No-op when no forwarder
/// was registered for the path (file-only fallback wiring, or a
/// second dump of the same path).
pub(crate) fn wait_sched_forwarders_drained(log_path: &str) {
    use std::time::{Duration, Instant};
    const DRAIN_BOUND: Duration = Duration::from_secs(5);
    let Some(map) = SCHED_FWD_THREADS.get() else {
        return;
    };
    let Some(handles) = map.lock().unwrap().remove(log_path) else {
        return;
    };
    let deadline = Instant::now() + DRAIN_BOUND;
    for h in handles {
        while !h.is_finished() {
            if Instant::now() >= deadline {
                tracing::warn!(
                    log_path,
                    "scheduler stdio forwarders still draining at dump time; \
                     the dumped log may be missing the child's final output"
                );
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let _ = h.join();
    }
}

/// Forwarder thread for one scheduler-child pipe: read chunks from
/// `read_end` until EOF (child closed the write end), and for each
/// chunk (a) append it to the merged `/tmp/sched.log` clone `log` and
/// (b) ship it live to the host via `sender`.
///
/// The `log` append preserves the pre-streaming merged-file view: both
/// forwarders hold clones of the SAME open file description (from
/// `File::try_clone`), so their appends share one offset and interleave
/// in read order exactly as the child's two direct-to-file fds did.
/// The `sender` ship is best-effort (a not-yet-open bulk port drops the
/// chunk); neither a file write error nor a send failure stops the
/// drain — the thread reads to EOF so the pipe never wedges the child
/// on backpressure.
///
/// The handle is registered in [`SCHED_FWD_THREADS`] under `log_path`
/// so the dump paths can wait for the drain to finish
/// ([`wait_sched_forwarders_drained`]); a thread-spawn failure
/// registers nothing and the stream is silently absent, same as the
/// stdio forwarders in [`super::modes::redirect_stdio_to_bulk_port`].
fn spawn_sched_log_forwarder(
    mut read_end: fs::File,
    mut log: Option<fs::File>,
    log_path: &str,
    name: &'static str,
    sender: fn(&[u8]) -> bool,
) {
    let spawned = std::thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let mut buf = [0u8; STDIO_CHUNK_BYTES];
            loop {
                match read_end.read(&mut buf) {
                    Ok(0) => break, // EOF — child closed its stdout/stderr.
                    Ok(n) => {
                        if let Some(f) = log.as_mut() {
                            let _ = f.write_all(&buf[..n]);
                        }
                        let _ = sender(&buf[..n]);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
        });
    if let Ok(handle) = spawned {
        SCHED_FWD_THREADS
            .get_or_init(Default::default)
            .lock()
            .unwrap()
            .entry(log_path.to_string())
            .or_default()
            .push(handle);
    }
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
    if let Some(streamed) = try_stream_sched_stdio(log_file, log_path) {
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
fn try_stream_sched_stdio(log_file: Option<&fs::File>, log_path: &str) -> Option<(Stdio, Stdio)> {
    let (stdout_r, stdout_w) = super::modes::make_pipe()?;
    let (stderr_r, stderr_w) = super::modes::make_pipe()?;
    // Clone the merged log file per forwarder so both append through
    // the SAME open file description (shared offset → interleave in
    // read order). A clone failure just drops the file append for that
    // stream; the live bulk-port ship still runs.
    let stdout_log = log_file.and_then(|f| f.try_clone().ok());
    let stderr_log = log_file.and_then(|f| f.try_clone().ok());
    spawn_sched_log_forwarder(
        stdout_r,
        stdout_log,
        log_path,
        "ktstr-sched-stdout-fwd",
        crate::vmm::guest_comms::send_sched_stdout_chunk,
    );
    spawn_sched_log_forwarder(
        stderr_r,
        stderr_log,
        log_path,
        "ktstr-sched-stderr-fwd",
        crate::vmm::guest_comms::send_sched_stderr_chunk,
    );
    Some((Stdio::from(stdout_w), Stdio::from(stderr_w)))
}

/// Pure spawn helper — runs the spawn → poll-startup → poll-attached
/// pipeline and returns a `Result` so callers can choose how to
/// handle each failure mode. The boot path uniformly responds with
/// `dump_sched_output` + `send_lifecycle` + `force_reboot`; the
/// scheduler-lifecycle Op dispatch surfaces each `Err` variant as
/// a typed test-failure rather than rebooting the VM.
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
) -> Result<Option<(Child, String)>, SpawnSchedulerError> {
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

    let mut child = Command::new(binary_path)
        .args(&args)
        .env("RUST_LOG", &sched_rust_log)
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
        .map_err(SpawnSchedulerError::SpawnFailed)?;

    // Publish the scheduler PID via the [`SCHED_PID`] atomic side
    // channel — readers retrieve it through [`sched_pid`]. The
    // previous implementation called `std::env::set_var("SCHED_PID",
    // ...)` here, but the Phase A probe thread spawned earlier in
    // `ktstr_guest_init` (`start_probe_phase_a`) is alive at this
    // point, so mutating glibc's global `__environ` array races
    // with the probe thread's potential `getenv`/`execve` traffic
    // — documented UB on Linux. The atomic store is data-race-free
    // and the published value reaches readers via the same
    // `Acquire`/`Release` synchronisation the [`sched_pid`] reader
    // uses.
    //
    // The `child.id()` value fits in `i32` because Linux pids are
    // `pid_t` (signed 32-bit on every supported arch).
    // `kernel.pid_max` is a 22-bit limit by default and the kernel
    // never returns negative pids from `fork(2)`, so the cast is
    // exact.
    SCHED_PID.store(child.id() as i32, Ordering::Release);

    let status = poll_scx_attached(child.id());
    match status {
        ScxAttachStatus::Attached => Ok(Some((child, log_path.to_string()))),
        ScxAttachStatus::Died => {
            // The dead child is auto-reaped under SIGCHLD=SIG_IGN. SCHED_PID
            // was published optimistically at spawn; clear it before any
            // caller can install an exit monitor against the stale pid.
            SCHED_PID.store(0, Ordering::Release);
            Err(SpawnSchedulerError::StartupDied {
                log_path: log_path.to_string(),
            })
        }
        status => {
            let reason = match status {
                ScxAttachStatus::Rejected(diagnostic) => diagnostic.reason("rejected-enable"),
                ScxAttachStatus::ServiceBudgetExceeded(diagnostic) => {
                    diagnostic.reason("service-budget")
                }
                ScxAttachStatus::WallGuardExceeded(diagnostic) => diagnostic.reason("wall-guard"),
                ScxAttachStatus::SysfsAbsent(diagnostic) => diagnostic.reason("sysfs-absent"),
                ScxAttachStatus::ObserverError(error) => {
                    format!("cause=observer-error detail={error:?}")
                }
                ScxAttachStatus::Attached | ScxAttachStatus::Died => unreachable!(),
            };
            tracing::warn!(
                pid = child.id(),
                reason,
                "scheduler did not complete sched_ext attach"
            );

            // The process is still alive but not attached. Kill it
            // deterministically so it cannot late-bind and pollute a later
            // Attach/Replace operation.
            let pid = child.id() as libc::pid_t;
            unsafe {
                let _ = libc::kill(pid, libc::SIGKILL);
            }
            let _ = child.wait();
            SCHED_PID.store(0, Ordering::Release);
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
/// `(None, None)` — preserves the prior contract where an absent
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
) -> (Option<Child>, Option<String>) {
    match try_spawn_scheduler(binary_path, args_path, log_path) {
        Ok(None) => (None, None),
        Ok(Some((child, log))) => {
            // Attach is DEFINITIVELY confirmed here and only here on the
            // boot path: `try_spawn_scheduler` returns `Ok(Some(..))` only
            // after `poll_scx_attached` observed `root/ops` registered with
            // `state == Enabled` AND the child is alive. Emit
            // `SchedulerAttached` so the host arms the progress watchdog's
            // workload deadline from this confirmed-attach moment. NOT
            // emitted on `Ok(None)` (no scheduler configured / EEVDF run —
            // binary absent) nor on any `Err` arm (Died / NotAttached each
            // send their own frame + force_reboot below), so the frame's
            // presence is an unambiguous live-scheduler attach proof. The
            // Op-dispatch re-attach path calls `try_spawn_scheduler`
            // directly and emits no lifecycle frame, matching the
            // Died/NotAttached asymmetry — this boot wrapper is the sole
            // lifecycle-emitting attach site.
            crate::vmm::guest_comms::send_lifecycle(
                crate::vmm::wire::LifecyclePhase::SchedulerAttached,
                "",
            );
            (Some(child), Some(log))
        }
        Err(SpawnSchedulerError::SpawnFailed(e)) => {
            tracing::error!(err = %e, "ktstr-init: spawn scheduler failed");
            // Synthesize a minimal sched-log payload framed by
            // the existing SCHED_OUTPUT_START/END markers so the
            // host's `parse_sched_output` returns the spawn-
            // failure diagnostic exactly as the prior COM2 path
            // did.
            crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_START.as_bytes());
            send_sched_log_text(&format!("failed to spawn: {e}"));
            crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_END.as_bytes());
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
            dump_sched_output(&log_path);
            crate::vmm::guest_comms::send_lifecycle(
                crate::vmm::wire::LifecyclePhase::SchedulerDied,
                "",
            );
            crate::vmm::guest_comms::send_exit(1);
            drain_probe_pipeline(probe_drain.as_ref(), crate::test_support::PROBE_DRAIN_GRACE);
            force_reboot();
        }
        Err(SpawnSchedulerError::NotAttached { reason, log_path }) => {
            dump_sched_output(&log_path);
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
        service: std::time::Duration,
        exited: bool,
        snapshot: ScxAttachSnapshot,
    }

    impl FakeAttachFrame {
        fn new(wall_s: f64, service_s: f64, exited: bool, snapshot: ScxAttachSnapshot) -> Self {
            Self {
                wall: std::time::Duration::from_secs_f64(wall_s),
                service: std::time::Duration::from_secs_f64(service_s),
                exited,
                snapshot,
            }
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

        fn snapshot(&mut self) -> Result<ScxAttachSnapshot, String> {
            Ok(self.frame().snapshot.clone())
        }

        fn service_elapsed(&mut self) -> Result<std::time::Duration, String> {
            Ok(self.frame().service)
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

    fn test_policy() -> AttachWaitPolicy {
        AttachWaitPolicy {
            cadence: std::time::Duration::from_millis(50),
            service_budget: std::time::Duration::from_secs(35),
            wall_guard: std::time::Duration::from_secs(90),
        }
    }

    /// Host starvation no longer consumes attach allowance: more than the old
    /// ten wall seconds may pass while scheduler CPU service remains small.
    #[test]
    fn attach_wait_charges_service_not_wall() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, 0.0, false, loading_snapshot()),
            FakeAttachFrame::new(12.0, 1.0, false, loading_snapshot()),
            FakeAttachFrame::new(20.0, 2.0, false, enabling_snapshot()),
            FakeAttachFrame::new(21.0, 2.5, false, attached_snapshot()),
        ]);
        assert_eq!(
            wait_scx_attach_with(&mut io, test_policy()),
            ScxAttachStatus::Attached
        );
    }

    /// Readiness is sampled before the wall guard, so an attach that completed
    /// while PID 1 was descheduled across the exact boundary wins.
    #[test]
    fn attach_at_wall_guard_is_success() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, 0.0, false, loading_snapshot()),
            FakeAttachFrame::new(90.0, 1.0, false, attached_snapshot()),
        ]);
        assert_eq!(
            wait_scx_attach_with(&mut io, test_policy()),
            ScxAttachStatus::Attached
        );
    }

    /// The pidfd remains in the wait through the whole attach, not only the
    /// former one-second startup window.
    #[test]
    fn scheduler_death_after_old_startup_gate_is_immediate() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, 0.0, false, loading_snapshot()),
            FakeAttachFrame::new(2.0, 0.2, true, loading_snapshot()),
        ]);
        assert_eq!(
            wait_scx_attach_with(&mut io, test_policy()),
            ScxAttachStatus::Died
        );
    }

    /// A dead child wins over a stale `enabled` sysfs snapshot.
    #[test]
    fn simultaneous_death_and_ready_prefers_death() {
        let mut io = FakeAttachIo::new(vec![FakeAttachFrame::new(
            2.0,
            0.2,
            true,
            attached_snapshot(),
        )]);
        assert_eq!(
            wait_scx_attach_with(&mut io, test_policy()),
            ScxAttachStatus::Died
        );
    }

    /// The dynamic root kobject is intentionally absent before registration;
    /// a readable stable state node proves sched_ext exists.
    #[test]
    fn missing_root_ops_is_loading_not_sysfs_absent() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, 0.0, false, loading_snapshot()),
            FakeAttachFrame::new(40.0, 35.0, false, loading_snapshot()),
        ]);
        let status = wait_scx_attach_with(&mut io, test_policy());
        assert!(
            matches!(status, ScxAttachStatus::ServiceBudgetExceeded(_)),
            "missing lazy root/ops was misclassified: {status:?}"
        );
    }

    /// Only absence of the stable sched_ext state node means sched_ext sysfs is
    /// absent.
    #[test]
    fn missing_state_is_sysfs_absent() {
        let mut io = FakeAttachIo::new(vec![FakeAttachFrame::new(
            0.0,
            0.0,
            false,
            ScxAttachSnapshot {
                state: StateObservation::Missing,
                ops: OpsObservation::Missing,
            },
        )]);
        assert!(matches!(
            wait_scx_attach_with(&mut io, test_policy()),
            ScxAttachStatus::SysfsAbsent(_)
        ));
    }

    /// A semantic forward edge resets the per-progress service checkpoint.
    #[test]
    fn attach_progress_resets_service_budget() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, 0.0, false, loading_snapshot()),
            FakeAttachFrame::new(34.0, 34.0, false, loading_snapshot()),
            FakeAttachFrame::new(35.0, 34.5, false, enabling_snapshot()),
            FakeAttachFrame::new(60.0, 60.0, false, enabling_snapshot()),
            FakeAttachFrame::new(61.0, 60.5, false, attached_snapshot()),
        ]);
        assert_eq!(
            wait_scx_attach_with(&mut io, test_policy()),
            ScxAttachStatus::Attached
        );
    }

    #[test]
    fn service_clock_regression_is_observer_error() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, 1.0, false, loading_snapshot()),
            FakeAttachFrame::new(1.0, 0.5, false, loading_snapshot()),
        ]);
        assert!(matches!(
            wait_scx_attach_with(&mut io, test_policy()),
            ScxAttachStatus::ObserverError(ref error) if error.contains("regressed")
        ));
    }

    #[test]
    fn event_source_error_is_not_a_timeout() {
        let mut io = FakeAttachIo::new(vec![FakeAttachFrame::new(
            0.0,
            0.0,
            false,
            loading_snapshot(),
        )]);
        io.wait_error = Some("poll source failed".into());
        assert_eq!(
            wait_scx_attach_with(&mut io, test_policy()),
            ScxAttachStatus::ObserverError("poll source failed".into())
        );
    }

    #[test]
    fn backwards_enable_transition_is_rejected_immediately() {
        let mut io = FakeAttachIo::new(vec![
            FakeAttachFrame::new(0.0, 0.0, false, enabling_snapshot()),
            FakeAttachFrame::new(1.0, 1.0, false, loading_snapshot()),
        ]);
        assert!(matches!(
            wait_scx_attach_with(&mut io, test_policy()),
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
