//! Interrupt handling and child-process execution for cargo-ktstr.
//!
//! One [`InterruptGuard`] spans the CLI lifetime, with an explicit phase
//! boundary. During preflight ([`EARLY`]) a signal restores its default
//! disposition and terminates immediately: no cleanup-owned state exists yet,
//! so metadata, network, and kernel resolution retain ordinary interrupt
//! behavior. Immediately before the first reservation/result-dir/checkout,
//! [`enter_cleanup_phase`] crosses `EARLY -> TRANSITION -> CLEANUP`. Thereafter
//! SIGINT/SIGTERM is recorded and forwarded, the parent survives through RAII
//! cleanup, and the first signal is re-raised.
//!
//! Before ordinary CLI initialization, the externally visible process starts
//! a new-process-group subreaper and re-execs the real cargo-ktstr worker below
//! it. The launcher remains in its caller's process group, forwards terminal
//! signals, and owns a dedicated control pipe. Unexpected launcher EOF or a
//! worker exit without an exact clean-exit record makes the subreaper kill and
//! reap its complete descendant closure. Process-group changes, `setsid`, an
//! erased environment, and double-forking therefore cannot detach work from an
//! abruptly lost cargo-ktstr owner.
//!
//! Cleanup-phase commands within the worker retain the process-group anchor
//! used for signal forwarding and bounded output-tail accounting. The startup
//! subreaper is the ownership backstop outside that group: it keeps all
//! arbitrary `Command` spawning on Rust's normal path while covering escaped
//! descendants uniformly.
//!
//! The active handoff is `IDLE -> SPAWNING -> anchor-pgid -> REAPING -> IDLE`.
//! Signals handled during SPAWNING are counted by kind and replayed after
//! publication (including repeated Ctrl-C). At teardown, REAPING prevents new
//! readers, in-flight handlers drain while the anchor still pins its pgid, and
//! only then is the anchor released/reaped. A handler therefore cannot resume
//! a stale `kill(-pgid)` after reuse.

use std::collections::{HashMap, HashSet};
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::os::unix::fs::MetadataExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{
    Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Output, Stdio,
};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicUsize, Ordering};
use std::time::Duration;
use std::time::Instant;

const IDLE: libc::pid_t = 0;
const SPAWNING: libc::pid_t = -1;
const REAPING: libc::pid_t = -2;

const EARLY: u8 = 0;
const TRANSITION: u8 = 1;
const CLEANUP: u8 = 2;

const ANCHOR_MODE_ENV: &str = "__KTSTR_PROCESS_GROUP_ANCHOR";
const STARTUP_ROLE_ENV: &str = "__KTSTR_STARTUP_ROLE";
const STARTUP_ROLE_SUPERVISOR: &str = "supervisor-v1";
const STARTUP_ROLE_WORKER: &str = "worker-v1";
const STARTUP_TOKEN_ENV: &str = "__KTSTR_STARTUP_TOKEN";
const STARTUP_PARENT_PID_ENV: &str = "__KTSTR_STARTUP_PARENT_PID";
const STARTUP_OWNER_FD_ENV: &str = "__KTSTR_STARTUP_OWNER_FD";
const STARTUP_REPORT_FD_ENV: &str = "__KTSTR_STARTUP_REPORT_FD";
const STARTUP_WORKER_FD_ENV: &str = "__KTSTR_STARTUP_WORKER_FD";

#[cfg(test)]
const STARTUP_TEST_MODE_ENV: &str = "__KTSTR_TEST_STARTUP_MODE";
#[cfg(test)]
const STARTUP_TEST_PIDFILE_ENV: &str = "__KTSTR_TEST_STARTUP_PIDFILE";

const STARTUP_TOKEN_BYTES: usize = 16;
const STARTUP_OWNER_HELLO_MAGIC: &[u8; 8] = b"KOWNR01\0";
const STARTUP_WORKER_READY_MAGIC: &[u8; 8] = b"KWRDY01\0";
const STARTUP_WORKER_CLEAN_MAGIC: &[u8; 8] = b"KCLEN01\0";
const STARTUP_REPORT_READY_MAGIC: &[u8; 8] = b"KSRDY01\0";
const STARTUP_REPORT_STATUS_MAGIC: &[u8; 8] = b"KSTAT01\0";
const STARTUP_OWNER_ACK_MAGIC: &[u8; 8] = b"KACKN01\0";

#[cfg(test)]
const STARTUP_TEST_OBSERVATION_BACKSTOP: Duration = Duration::from_secs(30);
#[cfg(test)]
const STARTUP_TEST_PROMPT_BACKSTOP: Duration = Duration::from_secs(5);
#[cfg(not(test))]
const STARTUP_READY_BACKSTOP: Duration = Duration::from_secs(30);
#[cfg(test)]
const STARTUP_READY_BACKSTOP: Duration = STARTUP_TEST_OBSERVATION_BACKSTOP;
#[cfg(not(test))]
const STARTUP_REAP_BACKSTOP: Duration = Duration::from_secs(30);
#[cfg(test)]
const STARTUP_REAP_BACKSTOP: Duration = STARTUP_TEST_OBSERVATION_BACKSTOP;
#[cfg(not(test))]
const STARTUP_COMMIT_BACKSTOP: Duration = Duration::from_secs(30);
#[cfg(test)]
const STARTUP_COMMIT_BACKSTOP: Duration = STARTUP_TEST_OBSERVATION_BACKSTOP;
const STARTUP_POLL_INTERVAL: Duration = Duration::from_millis(10);
const STARTUP_SIGNAL_READY_GRACE: Duration = Duration::from_secs(1);

/// Post-leader CPU service a residual child group may consume before it is
/// conclusively a leak. Host descheduling consumes no budget.
#[cfg(not(test))]
const GROUP_TAIL_SERVICE_BUDGET_NS: u128 = 2_000_000_000;
#[cfg(test)]
const GROUP_TAIL_SERVICE_BUDGET_NS: u128 = 500_000_000;

/// Absolute starvation backstop for a residual child group.
#[cfg(not(test))]
const GROUP_TAIL_WALL_BACKSTOP: Duration = Duration::from_secs(30);
#[cfg(test)]
const GROUP_TAIL_WALL_BACKSTOP: Duration = Duration::from_secs(3);

const GROUP_SCAN_INTERVAL: Duration = Duration::from_millis(10);
#[cfg(not(test))]
const FORCED_GROUP_REAP_BACKSTOP: Duration = Duration::from_secs(30);
#[cfg(test)]
const FORCED_GROUP_REAP_BACKSTOP: Duration = Duration::from_secs(2);
#[cfg(not(test))]
const ANCHOR_REAP_BACKSTOP: Duration = Duration::from_secs(30);
#[cfg(test)]
const ANCHOR_REAP_BACKSTOP: Duration = Duration::from_secs(2);
#[cfg(not(test))]
const ANCHOR_COOPERATIVE_EXIT_GRACE: Duration = Duration::from_secs(1);
#[cfg(test)]
const ANCHOR_COOPERATIVE_EXIT_GRACE: Duration = Duration::from_millis(200);
#[cfg(not(test))]
const HANDLER_DRAIN_BACKSTOP: Duration = Duration::from_secs(30);
#[cfg(test)]
const HANDLER_DRAIN_BACKSTOP: Duration = Duration::from_secs(2);
const CHILD_OWNERSHIP_FAIL_CLOSED_EXIT_CODE: libc::c_int = 70;
const CHILD_OWNERSHIP_FAIL_CLOSED_DIAGNOSTIC: &[u8] =
    b"cargo ktstr fatal: child process ownership could not be closed safely; terminating instead of detaching work\n";

/// First SIGINT/SIGTERM caught by the current guard; zero means none.
static CAUGHT_SIGNAL: AtomicI32 = AtomicI32::new(0);

/// Non-zero child exit requested by a dispatcher that must remain silent.
///
/// The top-level owner consumes this only after restoring signal dispositions
/// and giving a caught signal first priority. This avoids a dispatcher-local
/// `process::exit` bypassing result-dir/shared-memory cleanup or signal
/// re-raising.
static DEFERRED_EXIT_CODE: AtomicI32 = AtomicI32::new(0);

/// Current CLI signal phase.
static SIGNAL_PHASE: AtomicU8 = AtomicU8::new(EARLY);

/// The active child process group, or one of [`IDLE`] / [`SPAWNING`].
///
/// Only normal-context runner code changes this slot. The signal handler reads
/// it and may signal a published positive pgid.
static ACTIVE_CHILD_GROUP: AtomicI32 = AtomicI32::new(IDLE);

/// Number of handlers that may have loaded a published pgid and not yet
/// completed their `kill(2)`.
///
/// The reaper changes `pgid -> REAPING`, then drains this count before the
/// real wait/reap. A handler that starts later sees REAPING and never obtains
/// the old pgid; a handler already counted must finish while the leader is
/// still an unreaped zombie.
static HANDLERS_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// Signal deliveries observed by handlers that loaded [`SPAWNING`].
///
/// Separate counters preserve repeated deliveries as well as distinct signal
/// kinds: two Ctrl-Cs during the handoff must become two child-group signals,
/// while [`CAUGHT_SIGNAL`] remains first-only for the parent's eventual
/// re-raise.
static PENDING_HANDOFF_SIGINT: AtomicUsize = AtomicUsize::new(0);
static PENDING_HANDOFF_SIGTERM: AtomicUsize = AtomicUsize::new(0);

/// Whether a guard currently owns the process dispositions.
static GUARD_ACTIVE: AtomicBool = AtomicBool::new(false);

/// A checkout and other in-process work can poll this to stop promptly.
pub(crate) static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Startup-launcher signal target. The launcher is single-threaded, but a
/// signal may arrive at any instruction after its handler is installed.
static STARTUP_SUPERVISOR_GROUP: AtomicI32 = AtomicI32::new(IDLE);
static STARTUP_CAUGHT_SIGNAL: AtomicI32 = AtomicI32::new(0);
static STARTUP_PENDING_SIGINT: AtomicUsize = AtomicUsize::new(0);
static STARTUP_PENDING_SIGTERM: AtomicUsize = AtomicUsize::new(0);

/// Clean-intent pipe retained only by the real worker.
///
/// The descriptor is made CLOEXEC immediately after hidden-role validation,
/// so arbitrary commands cannot accidentally keep the protocol alive.
static STARTUP_WORKER_CLEAN_FD: AtomicI32 = AtomicI32::new(-1);

#[cfg(test)]
static TEST_PAUSE_HANDLER_AFTER_LOAD: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static TEST_HANDLER_LOADED: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static TEST_RELEASE_HANDLER: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static TEST_PAUSE_HANDLER_BEFORE_RECORD: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static TEST_HANDLER_BEFORE_RECORD: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static TEST_RELEASE_HANDLER_BEFORE_RECORD: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static TEST_FORWARDED_SIGINT: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static TEST_FORWARDED_SIGTERM: AtomicUsize = AtomicUsize::new(0);

/// Handle SIGINT/SIGTERM according to the current CLI phase.
///
/// Atomic operations and `kill(2)` are async-signal-safe on the supported
/// Linux targets.
extern "C" fn handler(sig: libc::c_int) {
    HANDLERS_IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
    let phase = SIGNAL_PHASE.load(Ordering::SeqCst);

    // EARLY owns no cleanup state. Terminate with normal signal semantics
    // instead of catching a signal around potentially blocking preflight.
    // Do not decrement HANDLERS_IN_FLIGHT: a concurrent phase transition must
    // remain parked until this process has terminated.
    if phase == EARLY {
        terminate_from_handler(sig);
    }

    #[cfg(test)]
    if TEST_PAUSE_HANDLER_BEFORE_RECORD.load(Ordering::SeqCst) {
        TEST_HANDLER_BEFORE_RECORD.store(true, Ordering::SeqCst);
        while !TEST_RELEASE_HANDLER_BEFORE_RECORD.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }
    }

    record_signal(sig);
    let waiter_id = ktstr::flock::interruptible_flock_waiter_id();
    ktstr::flock::wake_interruptible_flock_waiter(waiter_id);
    let pgid = ACTIVE_CHILD_GROUP.load(Ordering::SeqCst);
    #[cfg(test)]
    if TEST_PAUSE_HANDLER_AFTER_LOAD.load(Ordering::SeqCst) {
        TEST_HANDLER_LOADED.store(true, Ordering::SeqCst);
        while !TEST_RELEASE_HANDLER.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }
    }
    if pgid == SPAWNING {
        match sig {
            libc::SIGINT => {
                PENDING_HANDOFF_SIGINT.fetch_add(1, Ordering::SeqCst);
            }
            libc::SIGTERM => {
                PENDING_HANDOFF_SIGTERM.fetch_add(1, Ordering::SeqCst);
            }
            _ => {}
        }
    } else if pgid > 0 {
        // SAFETY: a negative pid addresses a process group. `kill` is
        // async-signal-safe and does not borrow Rust state.
        unsafe {
            libc::kill(-pgid, sig);
        }
    }
    HANDLERS_IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
}

fn record_signal(sig: libc::c_int) {
    let _ = CAUGHT_SIGNAL.compare_exchange(0, sig, Ordering::SeqCst, Ordering::SeqCst);
    INTERRUPTED.store(true, Ordering::SeqCst);
    ktstr::fetch::set_git_operation_interrupted(true);
}

fn terminate_from_handler(sig: libc::c_int) -> ! {
    // SAFETY: sigaction, sigprocmask, getpid, kill, and _exit are
    // async-signal-safe. Unblocking after installing SIG_DFL makes the
    // self-directed delivery terminate synchronously; _exit is the fallback.
    unsafe {
        let mut dfl: libc::sigaction = std::mem::zeroed();
        dfl.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut dfl.sa_mask);
        libc::sigaction(sig, &dfl, std::ptr::null_mut());

        let mut unblock: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut unblock);
        libc::sigaddset(&mut unblock, sig);
        libc::sigprocmask(libc::SIG_UNBLOCK, &unblock, std::ptr::null_mut());
        libc::kill(libc::getpid(), sig);
        libc::_exit(128 + sig);
    }
}

#[cfg(test)]
pub(crate) fn record_for_test(sig: libc::c_int) {
    record_signal(sig);
}

#[cfg(test)]
fn active_group_for_test() -> libc::pid_t {
    ACTIVE_CHILD_GROUP.load(Ordering::SeqCst)
}

#[cfg(test)]
pub(crate) fn test_serial_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(
        ACTIVE_CHILD_GROUP.load(Ordering::SeqCst),
        IDLE,
        "a prior signal-runner test leaked its active child slot",
    );
    CAUGHT_SIGNAL.store(0, Ordering::SeqCst);
    DEFERRED_EXIT_CODE.store(0, Ordering::SeqCst);
    INTERRUPTED.store(false, Ordering::SeqCst);
    ktstr::fetch::set_git_operation_interrupted(false);
    SIGNAL_PHASE.store(EARLY, Ordering::SeqCst);
    HANDLERS_IN_FLIGHT.store(0, Ordering::SeqCst);
    clear_pending_handoff_signals();
    TEST_PAUSE_HANDLER_AFTER_LOAD.store(false, Ordering::SeqCst);
    TEST_HANDLER_LOADED.store(false, Ordering::SeqCst);
    TEST_RELEASE_HANDLER.store(false, Ordering::SeqCst);
    TEST_PAUSE_HANDLER_BEFORE_RECORD.store(false, Ordering::SeqCst);
    TEST_HANDLER_BEFORE_RECORD.store(false, Ordering::SeqCst);
    TEST_RELEASE_HANDLER_BEFORE_RECORD.store(false, Ordering::SeqCst);
    TEST_FORWARDED_SIGINT.store(0, Ordering::SeqCst);
    TEST_FORWARDED_SIGTERM.store(0, Ordering::SeqCst);
    guard
}

/// Owns the temporary SIGINT/SIGTERM dispositions.
pub(crate) struct InterruptGuard {
    prev_sigint: libc::sigaction,
    prev_sigterm: libc::sigaction,
}

/// Resolve every libc PLT entry that the async handler can reach before either
/// caught disposition becomes visible.
///
/// `sigaction` / `sigemptyset` are exercised by installation itself, and the
/// eventfd broker separately primes `write(2)` before publishing its fd. These
/// no-op calls cover the remaining EARLY/cleanup handler surface so its first
/// delivery cannot enter the dynamic loader's lazy-binding path.
fn prime_handler_libc_calls() {
    // SAFETY: all pointers name initialized local signal sets. Blocking an
    // empty set changes no mask, signal 0 performs permission/existence
    // checking without delivering a signal, and this runs in normal context
    // before cargo-ktstr installs either handler.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        assert_eq!(libc::sigemptyset(&mut set), 0, "prime sigemptyset");
        assert_eq!(
            libc::sigaddset(&mut set, libc::SIGINT),
            0,
            "prime sigaddset",
        );
        assert_eq!(libc::sigemptyset(&mut set), 0, "reset prime signal set");
        assert_eq!(
            libc::sigprocmask(libc::SIG_BLOCK, &set, std::ptr::null_mut()),
            0,
            "prime sigprocmask",
        );
        let pid = libc::getpid();
        assert!(pid > 0, "prime getpid");
        assert_eq!(libc::kill(pid, 0), 0, "prime kill");
    }
}

impl InterruptGuard {
    /// Install the forwarding handler and start a fresh first-signal epoch.
    pub(crate) fn install() -> Self {
        prime_handler_libc_calls();
        if ACTIVE_CHILD_GROUP.load(Ordering::SeqCst) != IDLE {
            fail_closed_child_ownership();
        }
        let was_active = GUARD_ACTIVE.swap(true, Ordering::SeqCst);
        assert!(!was_active, "only one InterruptGuard may be active",);
        CAUGHT_SIGNAL.store(0, Ordering::SeqCst);
        DEFERRED_EXIT_CODE.store(0, Ordering::SeqCst);
        INTERRUPTED.store(false, Ordering::SeqCst);
        ktstr::fetch::set_git_operation_interrupted(false);
        SIGNAL_PHASE.store(EARLY, Ordering::SeqCst);

        // SAFETY: `handler` has the signal-handler ABI and performs only
        // async-signal-safe operations. The out-pointers are valid.
        let installed = unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = handler as *const () as usize;
            libc::sigemptyset(&mut action.sa_mask);
            action.sa_flags = 0;
            install_signal_pair(libc::SIGINT, libc::SIGTERM, &action)
        };
        match installed {
            Ok((prev_sigint, prev_sigterm)) => Self {
                prev_sigint,
                prev_sigterm,
            },
            Err(error) => {
                GUARD_ACTIVE.store(false, Ordering::SeqCst);
                panic!("install cargo-ktstr signal handlers: {error}");
            }
        }
    }

    /// The first signal caught during this guard's lifetime.
    #[cfg(test)]
    pub(crate) fn interrupted(&self) -> Option<libc::c_int> {
        caught()
    }
}

/// Install two dispositions transactionally.
///
/// If the second install fails, the first is restored before the error is
/// returned. A caller can therefore unwind without retaining a half-installed
/// handler or ever constructing an [`InterruptGuard`] from uninitialized saved
/// dispositions.
unsafe fn install_signal_pair(
    first_signal: libc::c_int,
    second_signal: libc::c_int,
    action: &libc::sigaction,
) -> io::Result<(libc::sigaction, libc::sigaction)> {
    // SAFETY: callers supply a fully initialized action. Saved-disposition
    // pointers name initialized storage for each successful syscall.
    unsafe {
        let mut first_previous: libc::sigaction = std::mem::zeroed();
        if libc::sigaction(first_signal, action, &mut first_previous) != 0 {
            return Err(io::Error::last_os_error());
        }

        let mut second_previous: libc::sigaction = std::mem::zeroed();
        if libc::sigaction(second_signal, action, &mut second_previous) != 0 {
            let install_error = io::Error::last_os_error();
            if libc::sigaction(first_signal, &first_previous, std::ptr::null_mut()) != 0 {
                fail_closed_child_ownership();
            }
            return Err(install_error);
        }
        Ok((first_previous, second_previous))
    }
}

/// Cross from preflight's terminate-immediately phase into cleanup ownership.
///
/// The TRANSITION state plus handler drain closes the race where a handler
/// loaded EARLY just before the caller acquired its first resource.
pub(crate) fn enter_cleanup_phase() -> io::Result<()> {
    match SIGNAL_PHASE.compare_exchange(EARLY, TRANSITION, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(EARLY) => {
            // Establish the normal-context wake broker before publishing
            // CLEANUP. No reservation can exist before this boundary, and a
            // handler that lands during TRANSITION merely records an
            // interruption that makes this function fail after publication.
            let broker_result = ktstr::flock::start_interruptible_flock_broker()
                .map_err(|error| io::Error::other(format!("{error:#}")));
            drain_handlers_in_flight_bounded(None);
            SIGNAL_PHASE.store(CLEANUP, Ordering::SeqCst);
            broker_result?;
        }
        Err(TRANSITION) => {
            while SIGNAL_PHASE.load(Ordering::SeqCst) == TRANSITION {
                std::hint::spin_loop();
            }
        }
        Err(CLEANUP) => {}
        Err(other) => panic!("invalid cargo-ktstr signal phase {other}"),
        Ok(_) => unreachable!(),
    }

    if caught().is_some() {
        Err(interrupted_error())
    } else {
        Ok(())
    }
}

pub(crate) fn cleanup_phase_active() -> bool {
    SIGNAL_PHASE.load(Ordering::SeqCst) == CLEANUP
}

/// The first signal caught in the current/most-recent guard epoch.
pub(crate) fn caught() -> Option<libc::c_int> {
    match CAUGHT_SIGNAL.load(Ordering::SeqCst) {
        0 => None,
        sig => Some(sig),
    }
}

pub(crate) fn defer_exit_code(code: i32) {
    assert_ne!(code, 0, "a deferred process exit must be non-zero");
    let prior = DEFERRED_EXIT_CODE.compare_exchange(0, code, Ordering::SeqCst, Ordering::SeqCst);
    assert!(
        prior.is_ok() || prior == Err(code),
        "conflicting deferred process exits: existing={prior:?}, requested={code}",
    );
}

pub(crate) fn take_deferred_exit_code() -> Option<i32> {
    match DEFERRED_EXIT_CODE.swap(0, Ordering::SeqCst) {
        0 => None,
        code => Some(code),
    }
}

/// Restore the saved dispositions and close the final restore/read race.
pub(crate) fn restore_and_caught(guard: InterruptGuard) -> Option<libc::c_int> {
    drop(guard);
    caught()
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        let active = ACTIVE_CHILD_GROUP.load(Ordering::SeqCst);
        if active != IDLE {
            // Dropping the dispositions with a published pgid would let a
            // later child reuse that numeric group while an old handler epoch
            // still owns it. If the exact group remains published, stop its
            // work before the fail-closed process exit.
            if active > 0 {
                // SAFETY: a positive slot is an anchor-pinned pgid published
                // by this guard's child runner.
                unsafe {
                    libc::kill(-active, libc::SIGKILL);
                }
            }
            fail_closed_child_ownership();
        }

        // No cleanup-owned waiter remains when the top-level guard drops.
        // Hide and join the broker while our handler is still installed, so a
        // late handler can observe only a closed broker epoch.
        ktstr::flock::stop_interruptible_flock_broker();

        // SAFETY: these are the dispositions returned by `sigaction` during
        // installation and both pointers remain valid for each call.
        let (int_rc, term_rc) = unsafe {
            (
                libc::sigaction(libc::SIGINT, &self.prev_sigint, std::ptr::null_mut()),
                libc::sigaction(libc::SIGTERM, &self.prev_sigterm, std::ptr::null_mut()),
            )
        };
        if int_rc != 0 || term_rc != 0 {
            fail_closed_child_ownership();
        }

        // Restoring both dispositions prevents new entries into `handler`.
        // Drain an entry that began just before restoration only afterwards:
        // it may not have published CAUGHT_SIGNAL yet. This closes the final
        // restore/read race for `restore_and_caught`.
        drain_handlers_in_flight_bounded(None);
        SIGNAL_PHASE.store(EARLY, Ordering::SeqCst);
        GUARD_ACTIVE.store(false, Ordering::SeqCst);
    }
}

/// Reset the first caught signal to its default disposition and re-raise it.
///
/// Call only after cleanup and after dropping the [`InterruptGuard`].
pub(crate) fn reraise(sig: libc::c_int) -> ! {
    // SAFETY: SIG_DFL is a valid disposition and this runs in normal context.
    unsafe {
        let mut dfl: libc::sigaction = std::mem::zeroed();
        dfl.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut dfl.sa_mask);
        libc::sigaction(sig, &dfl, std::ptr::null_mut());
        libc::raise(sig);
    }
    std::process::exit(128 + sig)
}

extern "C" fn startup_launcher_handler(sig: libc::c_int) {
    let _ =
        STARTUP_CAUGHT_SIGNAL.compare_exchange(0, sig, Ordering::SeqCst, Ordering::SeqCst);
    let pgid = STARTUP_SUPERVISOR_GROUP.load(Ordering::SeqCst);
    if pgid == SPAWNING {
        match sig {
            libc::SIGINT => {
                STARTUP_PENDING_SIGINT.fetch_add(1, Ordering::SeqCst);
            }
            libc::SIGTERM => {
                STARTUP_PENDING_SIGTERM.fetch_add(1, Ordering::SeqCst);
            }
            _ => {}
        }
    } else if pgid > 0 {
        // SAFETY: the launcher publishes only the live, unreaped supervisor's
        // verified process-group id.
        unsafe {
            libc::kill(-pgid, sig);
        }
    }
}

extern "C" fn startup_supervisor_handler(_sig: libc::c_int) {}

struct StartupLauncherSignals {
    previous_mask: libc::sigset_t,
    previous_sigint: libc::sigaction,
    previous_sigterm: libc::sigaction,
    previous_sigchld: libc::sigaction,
    restored: bool,
}

impl StartupLauncherSignals {
    fn install_for_handoff() -> io::Result<Self> {
        // SAFETY: startup is single-threaded. Blocking before installing the
        // handler closes the disposition swap; the mask is restored
        // immediately afterwards so the handler can count every delivery
        // until authenticated worker readiness.
        unsafe {
            let mut terminal: libc::sigset_t = std::mem::zeroed();
            if libc::sigemptyset(&mut terminal) != 0
                || libc::sigaddset(&mut terminal, libc::SIGINT) != 0
                || libc::sigaddset(&mut terminal, libc::SIGTERM) != 0
            {
                return Err(io::Error::last_os_error());
            }
            let mut previous_mask: libc::sigset_t = std::mem::zeroed();
            if libc::sigprocmask(libc::SIG_BLOCK, &terminal, &mut previous_mask) != 0 {
                return Err(io::Error::last_os_error());
            }

            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = startup_launcher_handler as usize;
            if libc::sigemptyset(&mut action.sa_mask) != 0
                || libc::sigaddset(&mut action.sa_mask, libc::SIGINT) != 0
                || libc::sigaddset(&mut action.sa_mask, libc::SIGTERM) != 0
            {
                let error = io::Error::last_os_error();
                libc::sigprocmask(libc::SIG_SETMASK, &previous_mask, std::ptr::null_mut());
                return Err(error);
            }
            let mut previous_sigint: libc::sigaction = std::mem::zeroed();
            if libc::sigaction(libc::SIGINT, &action, &mut previous_sigint) != 0 {
                let error = io::Error::last_os_error();
                libc::sigprocmask(libc::SIG_SETMASK, &previous_mask, std::ptr::null_mut());
                return Err(error);
            }
            let mut previous_sigterm: libc::sigaction = std::mem::zeroed();
            if libc::sigaction(libc::SIGTERM, &action, &mut previous_sigterm) != 0 {
                let error = io::Error::last_os_error();
                libc::sigaction(libc::SIGINT, &previous_sigint, std::ptr::null_mut());
                libc::sigprocmask(libc::SIG_SETMASK, &previous_mask, std::ptr::null_mut());
                return Err(error);
            }
            let mut child_action: libc::sigaction = std::mem::zeroed();
            child_action.sa_sigaction = libc::SIG_DFL;
            if libc::sigemptyset(&mut child_action.sa_mask) != 0 {
                let error = io::Error::last_os_error();
                libc::sigaction(libc::SIGINT, &previous_sigint, std::ptr::null_mut());
                libc::sigaction(libc::SIGTERM, &previous_sigterm, std::ptr::null_mut());
                libc::sigprocmask(libc::SIG_SETMASK, &previous_mask, std::ptr::null_mut());
                return Err(error);
            }
            let mut previous_sigchld: libc::sigaction = std::mem::zeroed();
            if libc::sigaction(libc::SIGCHLD, &child_action, &mut previous_sigchld) != 0 {
                let error = io::Error::last_os_error();
                libc::sigaction(libc::SIGINT, &previous_sigint, std::ptr::null_mut());
                libc::sigaction(libc::SIGTERM, &previous_sigterm, std::ptr::null_mut());
                libc::sigprocmask(libc::SIG_SETMASK, &previous_mask, std::ptr::null_mut());
                return Err(error);
            }
            STARTUP_CAUGHT_SIGNAL.store(0, Ordering::SeqCst);
            STARTUP_PENDING_SIGINT.store(0, Ordering::SeqCst);
            STARTUP_PENDING_SIGTERM.store(0, Ordering::SeqCst);
            STARTUP_SUPERVISOR_GROUP.store(SPAWNING, Ordering::SeqCst);
            let installed = Self {
                previous_mask,
                previous_sigint,
                previous_sigterm,
                previous_sigchld,
                restored: false,
            };
            if libc::sigprocmask(
                libc::SIG_SETMASK,
                &installed.previous_mask,
                std::ptr::null_mut(),
            ) != 0
            {
                let error = io::Error::last_os_error();
                libc::sigaction(
                    libc::SIGINT,
                    &installed.previous_sigint,
                    std::ptr::null_mut(),
                );
                libc::sigaction(
                    libc::SIGTERM,
                    &installed.previous_sigterm,
                    std::ptr::null_mut(),
                );
                libc::sigaction(
                    libc::SIGCHLD,
                    &installed.previous_sigchld,
                    std::ptr::null_mut(),
                );
                STARTUP_SUPERVISOR_GROUP.store(IDLE, Ordering::SeqCst);
                return Err(error);
            }
            Ok(installed)
        }
    }

    fn publish_and_replay(&self, pgid: libc::pid_t) -> io::Result<()> {
        if STARTUP_SUPERVISOR_GROUP.compare_exchange(
            SPAWNING,
            pgid,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) != Ok(SPAWNING)
        {
            return Err(io::Error::other(
                "startup supervisor publication state is corrupt",
            ));
        }
        let mut pending_int = STARTUP_PENDING_SIGINT.swap(0, Ordering::SeqCst);
        let mut pending_term = STARTUP_PENDING_SIGTERM.swap(0, Ordering::SeqCst);
        let first = STARTUP_CAUGHT_SIGNAL.load(Ordering::SeqCst);
        // Preserve both distinct kinds before replaying repeats. Exact
        // inter-kind order beyond the first is not representable without an
        // unbounded signal-handler queue.
        match first {
            libc::SIGINT if pending_int > 0 => {
                unsafe {
                    libc::kill(-pgid, libc::SIGINT);
                }
                pending_int -= 1;
                for _ in 0..pending_term {
                    unsafe {
                        libc::kill(-pgid, libc::SIGTERM);
                    }
                }
                pending_term = 0;
            }
            libc::SIGTERM if pending_term > 0 => {
                unsafe {
                    libc::kill(-pgid, libc::SIGTERM);
                }
                pending_term -= 1;
                for _ in 0..pending_int {
                    unsafe {
                        libc::kill(-pgid, libc::SIGINT);
                    }
                }
                pending_int = 0;
            }
            _ => {}
        }
        for _ in 0..pending_int {
            unsafe {
                libc::kill(-pgid, libc::SIGINT);
            }
        }
        for _ in 0..pending_term {
            unsafe {
                libc::kill(-pgid, libc::SIGTERM);
            }
        }
        Ok(())
    }

    fn hide(&self, pgid: libc::pid_t) -> io::Result<()> {
        if STARTUP_SUPERVISOR_GROUP.compare_exchange(
            pgid,
            REAPING,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) != Ok(pgid)
        {
            return Err(io::Error::other(
                "startup supervisor retirement state is corrupt",
            ));
        }
        Ok(())
    }

    fn restore(&mut self) -> io::Result<Option<libc::c_int>> {
        if self.restored {
            let signal = STARTUP_CAUGHT_SIGNAL.load(Ordering::SeqCst);
            return Ok((signal != 0).then_some(signal));
        }
        // SAFETY: both actions and the mask were returned by successful
        // installation calls in this process.
        unsafe {
            if libc::sigaction(
                libc::SIGINT,
                &self.previous_sigint,
                std::ptr::null_mut(),
            ) != 0
                || libc::sigaction(
                    libc::SIGTERM,
                    &self.previous_sigterm,
                    std::ptr::null_mut(),
                ) != 0
                || libc::sigaction(
                    libc::SIGCHLD,
                    &self.previous_sigchld,
                    std::ptr::null_mut(),
                ) != 0
                || libc::sigprocmask(
                    libc::SIG_SETMASK,
                    &self.previous_mask,
                    std::ptr::null_mut(),
                ) != 0
            {
                return Err(io::Error::last_os_error());
            }
        }
        STARTUP_SUPERVISOR_GROUP.store(IDLE, Ordering::SeqCst);
        STARTUP_PENDING_SIGINT.store(0, Ordering::SeqCst);
        STARTUP_PENDING_SIGTERM.store(0, Ordering::SeqCst);
        self.restored = true;
        Ok(match STARTUP_CAUGHT_SIGNAL.load(Ordering::SeqCst) {
            0 => None,
            signal => Some(signal),
        })
    }
}

impl Drop for StartupLauncherSignals {
    fn drop(&mut self) {
        if !self.restored && self.restore().is_err() {
            fail_closed_child_ownership();
        }
    }
}

fn startup_pipe() -> io::Result<(OwnedFd, OwnedFd)> {
    let mut fds = [-1; 2];
    // SAFETY: pipe2 initializes both entries on success.
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: pipe2 returned two fresh descriptors with unique ownership.
    Ok(unsafe {
        (
            OwnedFd::from_raw_fd(fds[0]),
            OwnedFd::from_raw_fd(fds[1]),
        )
    })
}

fn set_fd_cloexec(fd: libc::c_int, enabled: bool) -> io::Result<()> {
    // SAFETY: fcntl reads and updates descriptor flags only.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    let wanted = if enabled {
        flags | libc::FD_CLOEXEC
    } else {
        flags & !libc::FD_CLOEXEC
    };
    // SAFETY: wanted preserves all flags except FD_CLOEXEC.
    if unsafe { libc::fcntl(fd, libc::F_SETFD, wanted) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn validate_startup_pipe(
    fd: libc::c_int,
    access: libc::c_int,
    label: &str,
) -> io::Result<()> {
    // SAFETY: fstat initializes stat for a live descriptor.
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if stat.st_mode & libc::S_IFMT != libc::S_IFIFO {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{label} is not a pipe"),
        ));
    }
    // SAFETY: F_GETFL does not mutate descriptor state.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    if flags & libc::O_ACCMODE != access {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{label} has the wrong access direction"),
        ));
    }
    Ok(())
}

fn startup_token() -> io::Result<[u8; STARTUP_TOKEN_BYTES]> {
    let mut token = [0_u8; STARTUP_TOKEN_BYTES];
    let mut filled = 0;
    while filled < token.len() {
        // SAFETY: getrandom receives the uninitialized suffix of a live byte
        // array and writes at most its supplied length.
        let result = unsafe {
            libc::syscall(
                libc::SYS_getrandom,
                token[filled..].as_mut_ptr(),
                token.len() - filled,
                0_u32,
            )
        };
        if result > 0 {
            filled += result as usize;
            continue;
        }
        if result < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
            continue;
        }
        return Err(if result == 0 {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "getrandom returned no startup-token bytes",
            )
        } else {
            io::Error::last_os_error()
        });
    }
    Ok(token)
}

fn encode_startup_token(token: &[u8; STARTUP_TOKEN_BYTES]) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(STARTUP_TOKEN_BYTES * 2);
    for byte in token {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn decode_startup_token(value: &std::ffi::OsStr) -> io::Result<[u8; STARTUP_TOKEN_BYTES]> {
    let value = value
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "startup token is not UTF-8"))?;
    if value.len() != STARTUP_TOKEN_BYTES * 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "startup token has the wrong length",
        ));
    }
    let mut token = [0_u8; STARTUP_TOKEN_BYTES];
    for (index, byte) in token.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "startup token is not hexadecimal",
            )
        })?;
    }
    Ok(token)
}

fn startup_env_token() -> io::Result<[u8; STARTUP_TOKEN_BYTES]> {
    let value = std::env::var_os(STARTUP_TOKEN_ENV).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "hidden startup role is missing its token",
        )
    })?;
    decode_startup_token(&value)
}

fn startup_env_parent() -> io::Result<libc::pid_t> {
    std::env::var(STARTUP_PARENT_PID_ENV)
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "hidden startup role is missing its parent pid",
            )
        })?
        .parse()
        .ok()
        .filter(|pid| *pid > 0)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "hidden startup role has an invalid parent pid",
            )
        })
}

fn take_startup_fd(
    name: &str,
    access: libc::c_int,
    label: &str,
) -> io::Result<OwnedFd> {
    let raw = std::env::var(name)
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("hidden startup role is missing {label}"),
            )
        })?
        .parse::<libc::c_int>()
        .ok()
        .filter(|fd| *fd > libc::STDERR_FILENO)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("hidden startup role has an invalid {label}"),
            )
        })?;
    // Restore CLOEXEC before any further hidden-role work, so an error path
    // cannot leak this temporarily inheritable descriptor through another
    // exec.
    set_fd_cloexec(raw, true)?;
    validate_startup_pipe(raw, access, label)?;
    // SAFETY: the hidden protocol gives this process sole ownership of raw.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

fn clear_startup_role_env() {
    // SAFETY: all hidden-role entry points run before ordinary initialization
    // or thread creation.
    unsafe {
        std::env::remove_var(STARTUP_ROLE_ENV);
        std::env::remove_var(STARTUP_TOKEN_ENV);
        std::env::remove_var(STARTUP_PARENT_PID_ENV);
        std::env::remove_var(STARTUP_OWNER_FD_ENV);
        std::env::remove_var(STARTUP_REPORT_FD_ENV);
        std::env::remove_var(STARTUP_WORKER_FD_ENV);
    }
}

fn startup_command_for_self_with_args(
    mut args: impl Iterator<Item = std::ffi::OsString>,
) -> io::Result<Command> {
    let arg0 = args
        .next()
        .ok_or_else(|| io::Error::other("cargo-ktstr argv is empty"))?;
    let mut command = Command::new("/proc/self/exe");
    command.arg0(arg0).args(args);
    command.env_remove(STARTUP_ROLE_ENV);
    command.env_remove(STARTUP_TOKEN_ENV);
    command.env_remove(STARTUP_PARENT_PID_ENV);
    command.env_remove(STARTUP_OWNER_FD_ENV);
    command.env_remove(STARTUP_REPORT_FD_ENV);
    command.env_remove(STARTUP_WORKER_FD_ENV);
    Ok(command)
}

fn startup_command_for_self() -> io::Result<Command> {
    startup_command_for_self_with_args(std::env::args_os())
}

#[cfg(test)]
fn startup_test_mode_is(expected: &str) -> bool {
    std::env::var(STARTUP_TEST_MODE_ENV).as_deref() == Ok(expected)
}

#[cfg(test)]
fn startup_test_checkpoint_path(phase: &str, suffix: &str) -> io::Result<std::path::PathBuf> {
    let mut path = std::path::PathBuf::from(
        std::env::var_os(STARTUP_TEST_PIDFILE_ENV).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "startup test checkpoint has no pidfile",
            )
        })?,
    );
    path.set_extension(format!("{phase}.{suffix}"));
    Ok(path)
}

#[cfg(test)]
fn startup_test_checkpoint_if(
    mode: &str,
    phase: &str,
    wait_for_release: bool,
) -> io::Result<()> {
    if !startup_test_mode_is(mode) {
        return Ok(());
    }
    let marker = startup_test_checkpoint_path(phase, "ready")?;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(marker)?;
    writeln!(file, "{}", unsafe { libc::getpid() })?;
    file.flush()?;
    if !wait_for_release {
        return Ok(());
    }
    let release = startup_test_checkpoint_path(phase, "release")?;
    let deadline = Instant::now() + STARTUP_READY_BACKSTOP;
    while !release.exists() {
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("startup test checkpoint {phase} was not released"),
            ));
        }
        std::thread::sleep(STARTUP_POLL_INTERVAL);
    }
    Ok(())
}

#[cfg(test)]
fn inject_startup_exec_failure_if(command: &mut Command, mode: &str) {
    if !startup_test_mode_is(mode) {
        return;
    }
    // SAFETY: this test-only callback constructs a fixed raw OS error and
    // performs no allocation or other non-async-signal-safe work after fork.
    unsafe {
        command.pre_exec(|| Err(io::Error::from_raw_os_error(libc::EIO)));
    }
}

fn write_startup_frame(fd: libc::c_int, mut frame: &[u8]) -> io::Result<()> {
    while !frame.is_empty() {
        // SAFETY: frame points to initialized bytes and fd is the validated
        // write end of one private startup pipe.
        let written = unsafe { libc::write(fd, frame.as_ptr().cast(), frame.len()) };
        if written > 0 {
            frame = &frame[written as usize..];
            continue;
        }
        if written < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
            continue;
        }
        return Err(if written == 0 {
            io::Error::new(
                io::ErrorKind::WriteZero,
                "startup protocol pipe accepted no bytes",
            )
        } else {
            io::Error::last_os_error()
        });
    }
    Ok(())
}

fn drain_startup_read(
    fd: libc::c_int,
    buffer: &mut Vec<u8>,
) -> io::Result<bool> {
    let mut chunk = [0_u8; 256];
    loop {
        // SAFETY: chunk is writable and fd is a validated read end.
        let read = unsafe { libc::read(fd, chunk.as_mut_ptr().cast(), chunk.len()) };
        if read > 0 {
            buffer.extend_from_slice(&chunk[..read as usize]);
            continue;
        }
        if read == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        if error.kind() == io::ErrorKind::WouldBlock {
            return Ok(false);
        }
        return Err(error);
    }
}

fn poll_startup_fds(fds: &mut [libc::pollfd], timeout: Duration) -> io::Result<()> {
    // SAFETY: fds is initialized and timeout_ms always returns a finite value.
    let result = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, timeout_ms(timeout)) };
    if result >= 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::Interrupted {
        Ok(())
    } else {
        Err(error)
    }
}

fn prefixed_token_frame(
    magic: &[u8; 8],
    token: &[u8; STARTUP_TOKEN_BYTES],
) -> [u8; 8 + STARTUP_TOKEN_BYTES] {
    let mut frame = [0_u8; 8 + STARTUP_TOKEN_BYTES];
    frame[..8].copy_from_slice(magic);
    frame[8..].copy_from_slice(token);
    frame
}

fn consume_exact_frame(buffer: &mut Vec<u8>, frame: &[u8]) -> io::Result<bool> {
    let shared = buffer.len().min(frame.len());
    if buffer[..shared] != frame[..shared] {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "startup protocol frame is malformed",
        ));
    }
    if buffer.len() < frame.len() {
        return Ok(false);
    }
    buffer.drain(..frame.len());
    Ok(true)
}

fn initialize_startup_supervisor_signal_state() -> io::Result<()> {
    // A caught disposition, unlike SIG_IGN, resets to SIG_DFL across the
    // worker exec. The supervisor itself remains alive when the launcher
    // forwards a terminal signal to the whole new process group.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = startup_supervisor_handler as usize;
        if libc::sigemptyset(&mut action.sa_mask) != 0
            || libc::sigaddset(&mut action.sa_mask, libc::SIGINT) != 0
            || libc::sigaddset(&mut action.sa_mask, libc::SIGTERM) != 0
            || libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut()) != 0
            || libc::sigaction(libc::SIGTERM, &action, std::ptr::null_mut()) != 0
        {
            return Err(io::Error::last_os_error());
        }
        let mut child_action: libc::sigaction = std::mem::zeroed();
        child_action.sa_sigaction = libc::SIG_DFL;
        if libc::sigemptyset(&mut child_action.sa_mask) != 0
            || libc::sigaction(libc::SIGCHLD, &child_action, std::ptr::null_mut()) != 0
        {
            return Err(io::Error::last_os_error());
        }
        let mut terminal: libc::sigset_t = std::mem::zeroed();
        if libc::sigemptyset(&mut terminal) != 0
            || libc::sigaddset(&mut terminal, libc::SIGINT) != 0
            || libc::sigaddset(&mut terminal, libc::SIGTERM) != 0
            || libc::sigprocmask(libc::SIG_UNBLOCK, &terminal, std::ptr::null_mut()) != 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

fn block_startup_terminal_signals() -> io::Result<libc::sigset_t> {
    unsafe {
        let mut terminal: libc::sigset_t = std::mem::zeroed();
        if libc::sigemptyset(&mut terminal) != 0
            || libc::sigaddset(&mut terminal, libc::SIGINT) != 0
            || libc::sigaddset(&mut terminal, libc::SIGTERM) != 0
        {
            return Err(io::Error::last_os_error());
        }
        let mut previous: libc::sigset_t = std::mem::zeroed();
        if libc::sigprocmask(libc::SIG_BLOCK, &terminal, &mut previous) != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(previous)
    }
}

fn restore_startup_signal_mask(previous: &libc::sigset_t) -> io::Result<()> {
    if unsafe { libc::sigprocmask(libc::SIG_SETMASK, previous, std::ptr::null_mut()) } != 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn initialize_startup_worker_signal_state() -> io::Result<()> {
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = libc::SIG_DFL;
        if libc::sigemptyset(&mut action.sa_mask) != 0
            || libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut()) != 0
            || libc::sigaction(libc::SIGTERM, &action, std::ptr::null_mut()) != 0
        {
            return Err(io::Error::last_os_error());
        }
        let mut terminal: libc::sigset_t = std::mem::zeroed();
        if libc::sigemptyset(&mut terminal) != 0
            || libc::sigaddset(&mut terminal, libc::SIGINT) != 0
            || libc::sigaddset(&mut terminal, libc::SIGTERM) != 0
            || libc::sigprocmask(libc::SIG_UNBLOCK, &terminal, std::ptr::null_mut()) != 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

fn startup_wait_child_bounded(child: &mut Child, timeout: Duration) -> io::Result<ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("startup supervisor {} did not exit", child.id()),
            ));
        }
        std::thread::sleep(STARTUP_POLL_INTERVAL);
    }
}

fn relay_raw_wait_status(raw: libc::c_int) -> ! {
    if libc::WIFEXITED(raw) {
        std::process::exit(libc::WEXITSTATUS(raw));
    }
    if libc::WIFSIGNALED(raw) {
        let signal = libc::WTERMSIG(raw);
        // SAFETY: restore an ordinary terminating signal disposition and
        // unblock it before exact self-delivery.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = libc::SIG_DFL;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(signal, &action, std::ptr::null_mut());
            let mut unblock: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut unblock);
            libc::sigaddset(&mut unblock, signal);
            libc::sigprocmask(libc::SIG_UNBLOCK, &unblock, std::ptr::null_mut());
            libc::kill(libc::getpid(), signal);
            libc::_exit(128 + signal);
        }
    }
    fail_closed_child_ownership()
}

fn startup_launcher_fail(
    mut signals: StartupLauncherSignals,
    owner: Option<OwnedFd>,
    mut supervisor: Option<&mut Child>,
    error: io::Error,
) -> ! {
    // Closing the sole owner writer is itself the fail-closed instruction.
    // Never SIGKILL the subreaper on a launcher timeout: it must remain alive
    // until its descendant closure has been drained.
    drop(owner);
    if let Some(child) = supervisor.as_mut() {
        let _ = startup_wait_child_bounded(child, STARTUP_REAP_BACKSTOP);
    }
    let caught = signals.restore().unwrap_or(None);
    eprintln!("cargo ktstr fatal: startup supervision failed: {error}");
    if let Some(signal) = caught {
        relay_raw_wait_status(signal)
    }
    std::process::exit(CHILD_OWNERSHIP_FAIL_CLOSED_EXIT_CODE)
}

fn startup_launcher_read_ready(
    report: &OwnedFd,
    token: &[u8; STARTUP_TOKEN_BYTES],
    buffer: &mut Vec<u8>,
) -> io::Result<()> {
    let wanted = prefixed_token_frame(STARTUP_REPORT_READY_MAGIC, token);
    let ready_deadline = Instant::now() + STARTUP_READY_BACKSTOP;
    let mut signal_deadline = None;
    loop {
        let eof = drain_startup_read(report.as_raw_fd(), buffer)?;
        if consume_exact_frame(buffer, &wanted)? {
            return Ok(());
        }
        if eof {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "startup supervisor exited before readiness",
            ));
        }
        let now = Instant::now();
        if STARTUP_CAUGHT_SIGNAL.load(Ordering::SeqCst) != 0 {
            let deadline = signal_deadline
                .get_or_insert_with(|| now + STARTUP_SIGNAL_READY_GRACE);
            if now >= *deadline {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "terminal signal caught while the startup worker remained unready; \
                     closing the owner channel for fail-closed subtree teardown",
                ));
            }
        }
        if now >= ready_deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "startup supervisor did not become ready before its bounded protocol deadline",
            ));
        }
        let mut pollfd = [libc::pollfd {
            fd: report.as_raw_fd(),
            events: libc::POLLIN | libc::POLLHUP | libc::POLLERR,
            revents: 0,
        }];
        let poll_deadline = signal_deadline
            .map_or(ready_deadline, |deadline| deadline.min(ready_deadline));
        let wait = poll_deadline
            .saturating_duration_since(now)
            .min(STARTUP_POLL_INTERVAL);
        poll_startup_fds(&mut pollfd, wait)?;
    }
}

fn startup_launcher_read_status(
    report: &OwnedFd,
    buffer: &mut Vec<u8>,
) -> io::Result<libc::c_int> {
    loop {
        let eof = drain_startup_read(report.as_raw_fd(), buffer)?;
        let shared = buffer.len().min(STARTUP_REPORT_STATUS_MAGIC.len());
        if buffer[..shared] != STARTUP_REPORT_STATUS_MAGIC[..shared] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "startup supervisor emitted a malformed status frame",
            ));
        }
        if buffer.len() >= 12 {
            let raw = libc::c_int::from_be_bytes(buffer[8..12].try_into().unwrap());
            buffer.drain(..12);
            if !buffer.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "startup supervisor emitted trailing status data",
                ));
            }
            return Ok(raw);
        }
        if eof {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "startup supervisor exited before reporting worker status",
            ));
        }
        let mut pollfd = [libc::pollfd {
            fd: report.as_raw_fd(),
            events: libc::POLLIN | libc::POLLHUP | libc::POLLERR,
            revents: 0,
        }];
        poll_startup_fds(&mut pollfd, STARTUP_POLL_INTERVAL)?;
    }
}

fn startup_launcher_main() -> ! {
    let token = startup_token().unwrap_or_else(|error| {
        eprintln!("cargo ktstr fatal: create startup token: {error}");
        std::process::exit(CHILD_OWNERSHIP_FAIL_CLOSED_EXIT_CODE);
    });
    let (owner_read, owner_write) = startup_pipe().unwrap_or_else(|error| {
        eprintln!("cargo ktstr fatal: create startup owner pipe: {error}");
        std::process::exit(CHILD_OWNERSHIP_FAIL_CLOSED_EXIT_CODE);
    });
    let (report_read, report_write) = startup_pipe().unwrap_or_else(|error| {
        eprintln!("cargo ktstr fatal: create startup report pipe: {error}");
        std::process::exit(CHILD_OWNERSHIP_FAIL_CLOSED_EXIT_CODE);
    });
    let mut signals = StartupLauncherSignals::install_for_handoff().unwrap_or_else(|error| {
        eprintln!("cargo ktstr fatal: install startup signal relay: {error}");
        std::process::exit(CHILD_OWNERSHIP_FAIL_CLOSED_EXIT_CODE);
    });

    let owner_read_raw = owner_read.as_raw_fd();
    let report_write_raw = report_write.as_raw_fd();
    let mut command = match startup_command_for_self() {
        Ok(command) => command,
        Err(error) => startup_launcher_fail(signals, Some(owner_write), None, error),
    };
    command
        .env(STARTUP_ROLE_ENV, STARTUP_ROLE_SUPERVISOR)
        .env(STARTUP_TOKEN_ENV, encode_startup_token(&token))
        .env(STARTUP_PARENT_PID_ENV, std::process::id().to_string())
        .env(STARTUP_OWNER_FD_ENV, owner_read_raw.to_string())
        .env(STARTUP_REPORT_FD_ENV, report_write_raw.to_string())
        .process_group(0);
    // SAFETY: startup is single-threaded. The closure performs only fcntl and
    // constructs raw-OS errors before the exact /proc/self/exe exec.
    unsafe {
        command.pre_exec(move || {
            set_fd_cloexec(owner_read_raw, false)?;
            set_fd_cloexec(report_write_raw, false)
        });
    }
    #[cfg(test)]
    inject_startup_exec_failure_if(&mut command, "supervisor-exec-failure");
    let mut supervisor = match command.spawn() {
        Ok(child) => child,
        Err(error) => startup_launcher_fail(signals, Some(owner_write), None, error),
    };
    drop(owner_read);
    drop(report_write);

    let pgid = match libc::pid_t::try_from(supervisor.id())
        .ok()
        .filter(|pid| *pid > 0)
    {
        Some(pgid) => pgid,
        None => startup_launcher_fail(
                signals,
                Some(owner_write),
                Some(&mut supervisor),
                io::Error::other("startup supervisor has an invalid pid"),
            ),
    };
    // SAFETY: the unreaped Child pins this exact pid.
    let actual_pgid = unsafe { libc::getpgid(pgid) };
    if actual_pgid != pgid {
        startup_launcher_fail(
            signals,
            Some(owner_write),
            Some(&mut supervisor),
            io::Error::other(format!(
                "startup supervisor process group was not established (pid {pgid}, pgid {actual_pgid})",
            )),
        );
    }
    let hello = prefixed_token_frame(STARTUP_OWNER_HELLO_MAGIC, &token);
    if let Err(error) = write_startup_frame(owner_write.as_raw_fd(), &hello) {
        startup_launcher_fail(
            signals,
            Some(owner_write),
            Some(&mut supervisor),
            error,
        );
    }
    if let Err(error) = set_nonblocking(report_read.as_raw_fd()) {
        startup_launcher_fail(
            signals,
            Some(owner_write),
            Some(&mut supervisor),
            error,
        );
    }

    let mut report = Vec::new();
    if let Err(error) = startup_launcher_read_ready(&report_read, &token, &mut report) {
        startup_launcher_fail(
            signals,
            Some(owner_write),
            Some(&mut supervisor),
            error,
        );
    }
    // Keep the handler in SPAWNING through authenticated worker readiness.
    // Every terminal delivery in the entire L->S->W handoff is counted and
    // replayed only after W has restored its own default dispositions and
    // unblocked the signals.
    if let Err(error) = signals.publish_and_replay(pgid) {
        startup_launcher_fail(
            signals,
            Some(owner_write),
            Some(&mut supervisor),
            error,
        );
    }

    let raw = match startup_launcher_read_status(&report_read, &mut report) {
        Ok(raw) => raw,
        Err(error) => startup_launcher_fail(
            signals,
            Some(owner_write),
            Some(&mut supervisor),
            error,
        ),
    };
    #[cfg(test)]
    if let Err(error) =
        startup_test_checkpoint_if("owner-death-pre-ack", "pre-ack", true)
    {
        startup_launcher_fail(
            signals,
            Some(owner_write),
            Some(&mut supervisor),
            error,
        );
    }

    let mut ack = [0_u8; 12];
    ack[..8].copy_from_slice(STARTUP_OWNER_ACK_MAGIC);
    ack[8..].copy_from_slice(&raw.to_be_bytes());
    if let Err(error) = write_startup_frame(owner_write.as_raw_fd(), &ack) {
        startup_launcher_fail(
            signals,
            Some(owner_write),
            Some(&mut supervisor),
            error,
        );
    }
    if let Err(error) = signals.hide(pgid) {
        startup_launcher_fail(
            signals,
            Some(owner_write),
            Some(&mut supervisor),
            error,
        );
    }
    let supervisor_status =
        match startup_wait_child_bounded(&mut supervisor, STARTUP_REAP_BACKSTOP) {
            Ok(status) => status,
            Err(error) => {
                startup_launcher_fail(signals, Some(owner_write), None, error);
            }
        };
    if supervisor_status.into_raw() != raw {
        startup_launcher_fail(
            signals,
            Some(owner_write),
            None,
            io::Error::other(format!(
                "startup supervisor status {supervisor_status} did not match worker status {}",
                ExitStatus::from_raw(raw),
            )),
        );
    }
    drop(owner_write);
    drop(report_read);
    let caught = signals.restore().unwrap_or_else(|error| {
        eprintln!("cargo ktstr fatal: restore startup signal relay: {error}");
        std::process::exit(CHILD_OWNERSHIP_FAIL_CLOSED_EXIT_CODE);
    });
    if let Some(signal) = caught {
        relay_raw_wait_status(signal);
    }
    relay_raw_wait_status(raw)
}

fn record_startup_worker_status(
    worker_pid: libc::pid_t,
    reaped_pid: libc::pid_t,
    raw: libc::c_int,
    worker_status: &mut Option<libc::c_int>,
) {
    if reaped_pid == worker_pid && worker_status.is_none() {
        *worker_status = Some(raw);
    }
}

fn startup_reap_nonblocking(
    worker_pid: libc::pid_t,
    worker_status: &mut Option<libc::c_int>,
) -> io::Result<bool> {
    loop {
        let mut raw = 0;
        // SAFETY: -1 selects only direct children of this subreaper.
        let pid = unsafe { libc::waitpid(-1, &mut raw, libc::WNOHANG) };
        if pid > 0 {
            record_startup_worker_status(worker_pid, pid, raw, worker_status);
            continue;
        }
        if pid == 0 {
            return Ok(false);
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        if error.raw_os_error() == Some(libc::ECHILD) {
            return Ok(true);
        }
        return Err(error);
    }
}

fn startup_direct_children() -> io::Result<Vec<libc::pid_t>> {
    // Production S is single-threaded, while the nested regression invokes S
    // from a libtest worker thread. Enumerating every task is correct in both
    // cases and also covers a future supervisor implementation that creates a
    // helper thread: Linux records children against the task that forked them.
    let mut children = HashSet::new();
    for task in std::fs::read_dir("/proc/self/task")? {
        let task = task?;
        let path = task.path().join("children");
        let contents = match std::fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(error) if process_gone(&error) => continue,
            Err(error) => return Err(error),
        };
        for pid in contents.split_ascii_whitespace() {
            let pid = pid.parse::<libc::pid_t>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "kernel exposed an invalid direct-child pid",
                )
            })?;
            children.insert(pid);
        }
    }
    let mut children: Vec<_> = children.into_iter().collect();
    children.sort_unstable();
    Ok(children)
}

fn startup_process_parent(pid: libc::pid_t) -> io::Result<Option<libc::pid_t>> {
    let stat = match std::fs::read(format!("/proc/{pid}/stat")) {
        Ok(stat) => stat,
        Err(error) if process_gone(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    let comm_end = stat
        .iter()
        .rposition(|byte| *byte == b')')
        .ok_or_else(|| invalid_proc_stat(pid, "comm boundary"))?;
    let fields: Vec<_> = stat[comm_end + 1..]
        .split(|byte| byte.is_ascii_whitespace())
        .filter(|field| !field.is_empty())
        .collect();
    let ppid = fields
        .get(1)
        .and_then(|field| parse_proc_stat_number::<libc::pid_t>(field))
        .ok_or_else(|| invalid_proc_stat(pid, "parent pid"))?;
    Ok(Some(ppid))
}

fn startup_kill_direct_child(pid: libc::pid_t) -> io::Result<()> {
    // SAFETY: pidfd_open returns a fresh exact process reference.
    let raw =
        unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0_u32) as libc::c_int };
    if raw < 0 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(());
        }
        return Err(error);
    }
    // SAFETY: pidfd_open returned a fresh descriptor.
    let pidfd = unsafe { OwnedFd::from_raw_fd(raw) };
    let own_pid = unsafe { libc::getpid() };
    if startup_process_parent(pid)? != Some(own_pid) {
        return Ok(());
    }
    // SAFETY: exact pidfd, SIGKILL, null siginfo, flags=0.
    let result = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            pidfd.as_raw_fd(),
            libc::SIGKILL,
            std::ptr::null::<libc::siginfo_t>(),
            0_u32,
        )
    };
    if result == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error)
    }
}

fn startup_kill_and_reap_descendants(
    worker_pid: libc::pid_t,
    worker_status: &mut Option<libc::c_int>,
) -> io::Result<libc::c_int> {
    let mut deadline = Instant::now() + STARTUP_REAP_BACKSTOP;
    loop {
        if startup_reap_nonblocking(worker_pid, worker_status)? {
            return (*worker_status).ok_or_else(|| {
                io::Error::other("startup worker disappeared without a wait status")
            });
        }
        for pid in startup_direct_children()? {
            startup_kill_direct_child(pid)?;
        }
        if Instant::now() >= deadline {
            let _ = std::io::stderr().write_all(
                b"cargo ktstr fatal: startup descendant cleanup exceeded its reap backstop; continuing fail-closed drain\n",
            );
            deadline = Instant::now() + STARTUP_REAP_BACKSTOP;
        }
        std::thread::sleep(STARTUP_POLL_INTERVAL);
    }
}

fn startup_drain_descendants_fail_closed(
    worker_pid: libc::pid_t,
    mut worker_status: Option<libc::c_int>,
) -> libc::c_int {
    loop {
        match startup_kill_and_reap_descendants(worker_pid, &mut worker_status) {
            Ok(raw) => return raw,
            Err(error) => {
                let diagnostic =
                    format!("cargo ktstr fatal: retrying startup descendant drain: {error}\n");
                let _ = std::io::stderr().write_all(diagnostic.as_bytes());
                // A previously reaped worker remains represented by its saved
                // status while transient procfs/pidfd failures are retried.
                if worker_status.is_none() {
                    let mut raw = 0;
                    let waited = unsafe { libc::waitpid(worker_pid, &mut raw, libc::WNOHANG) };
                    if waited == worker_pid {
                        worker_status = Some(raw);
                    }
                }
                std::thread::sleep(STARTUP_POLL_INTERVAL);
            }
        }
    }
}

struct StartupSubtreeOwner {
    worker_pid: libc::pid_t,
    worker_status: Option<libc::c_int>,
    armed: bool,
}

impl StartupSubtreeOwner {
    fn new(worker_pid: libc::pid_t) -> Self {
        Self {
            worker_pid,
            worker_status: None,
            armed: true,
        }
    }

    fn dirty_drain(&mut self) -> libc::c_int {
        let raw = startup_drain_descendants_fail_closed(self.worker_pid, self.worker_status);
        self.worker_status = Some(raw);
        self.armed = false;
        raw
    }

    fn release(&mut self) {
        self.armed = false;
    }
}

impl Drop for StartupSubtreeOwner {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.dirty_drain();
        }
    }
}

fn startup_wait_owner_hello(
    owner: &OwnedFd,
    token: &[u8; STARTUP_TOKEN_BYTES],
) -> io::Result<()> {
    set_nonblocking(owner.as_raw_fd())?;
    let wanted = prefixed_token_frame(STARTUP_OWNER_HELLO_MAGIC, token);
    let deadline = Instant::now() + STARTUP_READY_BACKSTOP;
    let mut buffer = Vec::new();
    loop {
        let eof = drain_startup_read(owner.as_raw_fd(), &mut buffer)?;
        if consume_exact_frame(&mut buffer, &wanted)? {
            if !buffer.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "startup owner sent data before supervisor readiness",
                ));
            }
            return Ok(());
        }
        if eof {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "startup owner disappeared before authentication",
            ));
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "startup owner did not authenticate",
            ));
        }
        let mut pollfd = [libc::pollfd {
            fd: owner.as_raw_fd(),
            events: libc::POLLIN | libc::POLLHUP | libc::POLLERR,
            revents: 0,
        }];
        poll_startup_fds(&mut pollfd, STARTUP_POLL_INTERVAL)?;
    }
}

fn startup_wait_owner_ack(
    owner: &OwnedFd,
    buffer: &mut Vec<u8>,
    raw: libc::c_int,
) -> io::Result<()> {
    let mut wanted = [0_u8; 12];
    wanted[..8].copy_from_slice(STARTUP_OWNER_ACK_MAGIC);
    wanted[8..].copy_from_slice(&raw.to_be_bytes());
    let deadline = Instant::now() + STARTUP_COMMIT_BACKSTOP;
    loop {
        let eof = drain_startup_read(owner.as_raw_fd(), buffer)?;
        if eof {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "startup owner disappeared before acknowledging status",
            ));
        }
        if consume_exact_frame(buffer, &wanted)? {
            if !buffer.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "startup owner sent trailing acknowledgement data",
                ));
            }
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "startup owner did not acknowledge worker status",
            ));
        }
        let mut pollfd = [libc::pollfd {
            fd: owner.as_raw_fd(),
            events: libc::POLLIN | libc::POLLHUP | libc::POLLERR,
            revents: 0,
        }];
        poll_startup_fds(&mut pollfd, STARTUP_POLL_INTERVAL)?;
    }
}

fn startup_spawn_worker(
    token: &[u8; STARTUP_TOKEN_BYTES],
    supervisor_pid: libc::pid_t,
) -> io::Result<(OwnedFd, StartupSubtreeOwner)> {
    let (worker_read, worker_write) = startup_pipe()?;
    set_nonblocking(worker_read.as_raw_fd())?;
    let worker_write_raw = worker_write.as_raw_fd();
    let mut command = startup_command_for_self()?;
    command
        .env(STARTUP_ROLE_ENV, STARTUP_ROLE_WORKER)
        .env(STARTUP_TOKEN_ENV, encode_startup_token(token))
        .env(STARTUP_PARENT_PID_ENV, supervisor_pid.to_string())
        .env(STARTUP_WORKER_FD_ENV, worker_write_raw.to_string());
    // SAFETY: the supervisor is single-threaded and the callback performs one
    // fcntl operation before exact self exec.
    unsafe {
        command.pre_exec(move || set_fd_cloexec(worker_write_raw, false));
    }
    #[cfg(test)]
    inject_startup_exec_failure_if(&mut command, "worker-exec-failure");
    let previous_mask = block_startup_terminal_signals()?;
    let mut worker = match command.spawn() {
        Ok(worker) => worker,
        Err(error) => {
            restore_startup_signal_mask(&previous_mask)?;
            return Err(error);
        }
    };
    let worker_pid = match libc::pid_t::try_from(worker.id())
        .ok()
        .filter(|pid| *pid > 0)
    {
        Some(pid) => pid,
        None => {
            let _ = terminate_direct_child_bounded(&mut worker);
            restore_startup_signal_mask(&previous_mask)?;
            return Err(io::Error::other("startup worker has an invalid pid"));
        }
    };
    drop(worker);
    drop(worker_write);
    let subtree = StartupSubtreeOwner::new(worker_pid);
    restore_startup_signal_mask(&previous_mask)?;
    Ok((worker_read, subtree))
}

fn startup_supervisor_inner() -> io::Result<libc::c_int> {
    let token = startup_env_token()?;
    let declared_parent = startup_env_parent()?;
    let owner = take_startup_fd(
        STARTUP_OWNER_FD_ENV,
        libc::O_RDONLY,
        "startup owner pipe",
    )?;
    let report = take_startup_fd(
        STARTUP_REPORT_FD_ENV,
        libc::O_WRONLY,
        "startup report pipe",
    )?;
    clear_startup_role_env();
    if unsafe { libc::getppid() } != declared_parent {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "startup supervisor was not re-execed by its declared parent",
        ));
    }
    initialize_startup_supervisor_signal_state()?;
    // SAFETY: this single-threaded process becomes the reparenting boundary
    // before it creates the worker.
    if unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) } != 0 {
        return Err(io::Error::last_os_error());
    }
    startup_wait_owner_hello(&owner, &token)?;

    let own_pid = unsafe { libc::getpid() };
    #[cfg(test)]
    startup_test_checkpoint_if("worker-exec-failure", "worker-exec", true)?;
    let (worker_read, mut subtree) = startup_spawn_worker(&token, own_pid)?;
    let worker_pid = subtree.worker_pid;

    let ready_frame = prefixed_token_frame(STARTUP_WORKER_READY_MAGIC, &token);
    let report_ready = prefixed_token_frame(STARTUP_REPORT_READY_MAGIC, &token);
    let mut owner_buffer = Vec::new();
    let mut worker_buffer = Vec::new();
    let mut worker_ready = false;
    let mut ready_reported = false;
    let mut clean_code = None::<u8>;
    let mut worker_eof = false;
    let ready_deadline = Instant::now() + STARTUP_READY_BACKSTOP;
    let mut clean_deadline = None::<Instant>;

    loop {
        let owner_eof = drain_startup_read(owner.as_raw_fd(), &mut owner_buffer)?;
        if owner_eof {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "startup owner disappeared before commit",
            ));
        }
        if !owner_buffer.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "startup owner sent a premature or malformed frame",
            ));
        }

        worker_eof |= drain_startup_read(worker_read.as_raw_fd(), &mut worker_buffer)?;
        if !worker_ready {
            match consume_exact_frame(&mut worker_buffer, &ready_frame) {
                Ok(true) => {
                    worker_ready = true;
                    // SAFETY: worker_pid remains an unreaped direct child.
                    let worker_pgid = unsafe { libc::getpgid(worker_pid) };
                    if worker_pgid != own_pid {
                        return Err(io::Error::other(format!(
                            "startup worker process group mismatch (pid {worker_pid}, pgid {worker_pgid}, expected {own_pid})",
                        )));
                    }
                    #[cfg(test)]
                    {
                        startup_test_checkpoint_if(
                            "malformed-report",
                            "report-fault",
                            true,
                        )?;
                        startup_test_checkpoint_if(
                            "truncated-report",
                            "report-fault",
                            true,
                        )?;
                        if startup_test_mode_is("malformed-report") {
                            write_startup_frame(report.as_raw_fd(), &[b'X'; 24])?;
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "injected malformed startup report",
                            ));
                        }
                        if startup_test_mode_is("truncated-report") {
                            write_startup_frame(
                                report.as_raw_fd(),
                                &report_ready[..4],
                            )?;
                            return Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                "injected truncated startup report",
                            ));
                        }
                    }
                    write_startup_frame(report.as_raw_fd(), &report_ready)?;
                    ready_reported = true;
                }
                Ok(false) => {}
                Err(error) => return Err(error),
            }
        }
        if worker_ready && clean_code.is_none() {
            let shared = worker_buffer.len().min(STARTUP_WORKER_CLEAN_MAGIC.len());
            if worker_buffer[..shared] != STARTUP_WORKER_CLEAN_MAGIC[..shared] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "startup worker emitted a malformed clean frame",
                ));
            }
        }
        if worker_ready && clean_code.is_none() && worker_buffer.len() >= 9 {
            if &worker_buffer[..8] != STARTUP_WORKER_CLEAN_MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "startup worker emitted a malformed clean frame",
                ));
            }
            clean_code = Some(worker_buffer[8]);
            clean_deadline = Some(Instant::now() + STARTUP_COMMIT_BACKSTOP);
            worker_buffer.drain(..9);
        }
        if clean_code.is_some() && !worker_buffer.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "startup worker emitted duplicate or trailing protocol data",
            ));
        }
        if worker_eof && clean_code.is_none() && subtree.armed {
            let _ = subtree.dirty_drain();
        }

        if subtree.worker_status.is_none() {
            let mut raw = 0;
            // SAFETY: worker_pid is the exact direct child and WNOHANG is
            // non-blocking.
            let waited = unsafe { libc::waitpid(worker_pid, &mut raw, libc::WNOHANG) };
            if waited == worker_pid {
                subtree.worker_status = Some(raw);
            } else if waited < 0 {
                let error = io::Error::last_os_error();
                if error.kind() != io::ErrorKind::Interrupted {
                    return Err(error);
                }
            }
        }

        if !worker_ready && Instant::now() >= ready_deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "startup worker did not become ready",
            ));
        }
        if subtree.worker_status.is_none()
            && clean_deadline.is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "startup worker did not exit after its clean record",
            ));
        }

        if let Some(raw) = subtree.worker_status {
            let matched_clean = worker_ready
                && worker_buffer.is_empty()
                && clean_code.is_some_and(|code| {
                    libc::WIFEXITED(raw) && libc::WEXITSTATUS(raw) == i32::from(code)
                });
            let raw = if matched_clean {
                raw
            } else if subtree.armed {
                subtree.dirty_drain()
            } else {
                raw
            };
            if !ready_reported {
                return Err(io::Error::other(
                    "startup worker exited before readiness",
                ));
            }
            let mut status_frame = [0_u8; 12];
            status_frame[..8].copy_from_slice(STARTUP_REPORT_STATUS_MAGIC);
            status_frame[8..].copy_from_slice(&raw.to_be_bytes());
            write_startup_frame(report.as_raw_fd(), &status_frame)?;

            startup_wait_owner_ack(&owner, &mut owner_buffer, raw)?;
            subtree.release();
            #[cfg(test)]
            startup_test_checkpoint_if(
                "owner-death-post-ack",
                "post-ack",
                true,
            )?;
            return Ok(raw);
        }

        let mut pollfds = [
            libc::pollfd {
                fd: owner.as_raw_fd(),
                events: libc::POLLIN | libc::POLLHUP | libc::POLLERR,
                revents: 0,
            },
            libc::pollfd {
                fd: worker_read.as_raw_fd(),
                events: libc::POLLIN | libc::POLLHUP | libc::POLLERR,
                revents: 0,
            },
        ];
        poll_startup_fds(&mut pollfds, STARTUP_POLL_INTERVAL)?;
    }
}

fn startup_supervisor_main() -> ! {
    match startup_supervisor_inner() {
        Ok(raw) => relay_raw_wait_status(raw),
        Err(error) => {
            let diagnostic = format!("cargo ktstr fatal: startup supervisor failed: {error}\n");
            // The ordinary write is acceptable here: hidden startup remains
            // single-threaded and is about to use _exit.
            let _ = std::io::stderr().write_all(diagnostic.as_bytes());
            unsafe { libc::_exit(CHILD_OWNERSHIP_FAIL_CLOSED_EXIT_CODE) }
        }
    }
}

fn startup_worker_bootstrap() -> io::Result<()> {
    let token = startup_env_token()?;
    let declared_parent = startup_env_parent()?;
    let control = take_startup_fd(
        STARTUP_WORKER_FD_ENV,
        libc::O_WRONLY,
        "startup worker control pipe",
    )?;
    clear_startup_role_env();
    if unsafe { libc::getppid() } != declared_parent {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "startup worker was not re-execed by its declared supervisor",
        ));
    }
    initialize_startup_worker_signal_state()?;
    let ready = prefixed_token_frame(STARTUP_WORKER_READY_MAGIC, &token);
    write_startup_frame(control.as_raw_fd(), &ready)?;
    let raw = control.into_raw_fd();
    if STARTUP_WORKER_CLEAN_FD.compare_exchange(
        -1,
        raw,
        Ordering::SeqCst,
        Ordering::SeqCst,
    ) != Ok(-1)
    {
        // SAFETY: raw remains uniquely owned after into_raw_fd.
        unsafe {
            libc::close(raw);
        }
        return Err(io::Error::other(
            "startup worker control descriptor is already installed",
        ));
    }
    Ok(())
}

/// Enter startup supervisor/worker roles or launch the supervised worker.
///
/// A normal invocation never returns from its thin launcher. The exact
/// re-execed worker validates its private descriptor and returns into ordinary
/// cargo-ktstr initialization.
pub(crate) fn run_startup_supervision() {
    match std::env::var(STARTUP_ROLE_ENV) {
        Ok(role) if role == STARTUP_ROLE_SUPERVISOR => startup_supervisor_main(),
        Ok(role) if role == STARTUP_ROLE_WORKER => {
            if startup_worker_bootstrap().is_err() {
                fail_closed_child_ownership();
            }
        }
        Ok(_) => fail_closed_child_ownership(),
        Err(std::env::VarError::NotPresent) => startup_launcher_main(),
        Err(std::env::VarError::NotUnicode(_)) => fail_closed_child_ownership(),
    }
}

/// Commit the worker's intended normal exit after all ordinary cleanup.
///
/// The supervisor releases any deliberately detached descendants only if the
/// subsequently observed wait status is WIFEXITED with this exact byte.
pub(crate) fn commit_startup_worker_exit(code: u8) {
    let fd = STARTUP_WORKER_CLEAN_FD.swap(-1, Ordering::SeqCst);
    if fd < 0 {
        fail_closed_child_ownership();
    }
    let mut frame = [0_u8; 9];
    frame[..8].copy_from_slice(STARTUP_WORKER_CLEAN_MAGIC);
    frame[8] = code;
    if write_startup_frame(fd, &frame).is_err() {
        unsafe {
            libc::close(fd);
        }
        fail_closed_child_ownership();
    }
    // SAFETY: swap transferred the unique descriptor ownership here.
    unsafe {
        libc::close(fd);
    }
}

/// Enter the hidden process-group anchor mode.
///
/// Called as the first line of cargo-ktstr's `main`. A normal invocation
/// returns false. An anchor invocation acknowledges readiness on stdout,
/// ignores SIGINT/SIGTERM, and blocks until its stdin control pipe reaches
/// EOF, then returns true so main exits cleanly.
pub(crate) fn run_anchor_mode_if_requested() -> bool {
    if std::env::var_os(ANCHOR_MODE_ENV).is_none() {
        return false;
    }

    // The ready byte is the parent's proof that this process can pin the pgid
    // across forwarded terminal signals. Never acknowledge readiness after a
    // partial disposition or mask install.
    if initialize_anchor_signal_state().is_err() {
        fail_closed_child_ownership();
    }

    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    if stdout
        .write_all(b"R")
        .and_then(|()| stdout.flush())
        .is_err()
    {
        return true;
    }
    drop(stdout);

    let mut stdin = std::io::stdin().lock();
    let mut byte = [0_u8; 64];
    loop {
        match stdin.read(&mut byte) {
            Ok(0) => break,
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
    true
}

fn initialize_anchor_signal_state() -> io::Result<()> {
    initialize_anchor_signal_state_with(
        |signal| {
            // SAFETY: the anchor is a newly exec'd, single-threaded process.
            // The initialized action ignores one valid terminal signal.
            unsafe {
                let mut ignore: libc::sigaction = std::mem::zeroed();
                ignore.sa_sigaction = libc::SIG_IGN;
                if libc::sigemptyset(&mut ignore.sa_mask) != 0 {
                    return Err(io::Error::last_os_error());
                }
                if libc::sigaction(signal, &ignore, std::ptr::null_mut()) != 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(())
        },
        || {
            // SAFETY: this process is still single-threaded and both set
            // operations receive initialized storage.
            unsafe {
                let mut unblock: libc::sigset_t = std::mem::zeroed();
                if libc::sigemptyset(&mut unblock) != 0
                    || libc::sigaddset(&mut unblock, libc::SIGINT) != 0
                    || libc::sigaddset(&mut unblock, libc::SIGTERM) != 0
                {
                    return Err(io::Error::last_os_error());
                }
                if libc::sigprocmask(libc::SIG_UNBLOCK, &unblock, std::ptr::null_mut()) != 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(())
        },
    )
}

fn initialize_anchor_signal_state_with(
    mut ignore_signal: impl FnMut(libc::c_int) -> io::Result<()>,
    unblock_terminal_signals: impl FnOnce() -> io::Result<()>,
) -> io::Result<()> {
    ignore_signal(libc::SIGINT)?;
    ignore_signal(libc::SIGTERM)?;
    unblock_terminal_signals()
}

/// Run a single-use command and return its exit status.
pub(crate) fn run_status(mut command: Command) -> io::Result<ExitStatus> {
    if !runner_enabled() {
        return command.status();
    }
    run_status_with_handoff(command, || {})
}

/// Run a single-use command, capturing stdout and stderr.
pub(crate) fn run_output(mut command: Command) -> io::Result<Output> {
    if !runner_enabled() {
        return command.output();
    }
    run_capture(command, true)
}

/// Run a single-use command, capturing stdout while leaving
/// stderr at the command's configured/default destination.
pub(crate) fn run_stdout(mut command: Command) -> io::Result<Output> {
    if !runner_enabled() {
        return command.output();
    }
    run_capture(command, false)
}

/// Observe captured child pipes while retaining their exact byte streams.
///
/// The observer runs synchronously on the child-owning thread. Its periodic
/// tick remains live while the command is silent, after the command leader
/// closes its pipes, and while same-process-group descendants finish. No
/// drain thread outlives this call, and every error path kills/reaps the
/// published child group before returning.
pub(crate) trait StdoutObserver {
    /// Inspect newly read bytes. The runner appends them to its returned
    /// [`Output::stdout`] before making this call.
    fn observe_stdout(&mut self, bytes: &[u8]);

    /// Inspect newly read stderr bytes. The runner appends them to its returned
    /// [`Output::stderr`] before making this call. Stdout-only runners never
    /// invoke this method.
    fn observe_stderr(&mut self, _bytes: &[u8]) {}

    /// Advance time-based reporting.
    fn tick(&mut self);

    /// Maximum time the runner may sleep before calling [`Self::tick`] again.
    fn next_tick_in(&self) -> Duration;

    /// Report the command's ordinary process exit, including non-zero exits.
    fn finished(&mut self, status: &ExitStatus);

    /// Report a spawn, pipe, poll, wait, or process-group ownership failure.
    fn failed(&mut self, error: &io::Error);
}

/// Run a command with inherited/configured stderr and incrementally observed,
/// exactly preserved stdout.
///
/// Unlike [`run_stdout`], this never waits for process exit before consuming
/// stdout. A silent child therefore continues to drive observer heartbeats,
/// while a chatty Cargo JSON producer cannot fill its pipe and deadlock.
pub(crate) fn run_stdout_observed<O>(command: Command, mut observer: O) -> io::Result<Output>
where
    O: StdoutObserver,
{
    run_observed(command, &mut observer, false)
}

/// Run a command with incrementally observed, exactly preserved stdout and
/// stderr.
///
/// Both pipes are drained synchronously on the child-owning thread. The
/// observer can tee stderr while retaining the exact child byte stream in the
/// returned [`Output`].
pub(crate) fn run_output_observed<O>(command: Command, mut observer: O) -> io::Result<Output>
where
    O: StdoutObserver,
{
    run_observed(command, &mut observer, true)
}

fn run_observed<O>(
    mut command: Command,
    observer: &mut O,
    capture_stderr: bool,
) -> io::Result<Output>
where
    O: StdoutObserver,
{
    command.stdout(Stdio::piped());
    if capture_stderr {
        command.stderr(Stdio::piped());
    }
    let result = if runner_enabled() {
        run_observed_group(command, observer, capture_stderr)
    } else {
        run_observed_direct(command, observer, capture_stderr)
    };
    match &result {
        Ok(output) => observer.finished(&output.status),
        Err(error) => observer.failed(error),
    }
    result
}

fn runner_enabled() -> bool {
    GUARD_ACTIVE.load(Ordering::SeqCst) && cleanup_phase_active()
}

fn interrupted_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::Interrupted,
        "interrupt caught before child process could be started",
    )
}

struct GroupAnchor {
    child: Child,
    control: Option<ChildStdin>,
    pgid: libc::pid_t,
}

struct GroupChild {
    child: Child,
    anchor: GroupAnchor,
    armed: bool,
    published: bool,
    known_members: HashSet<libc::pid_t>,
}

/// Restore the spawning thread's original mask on every return path.
struct SignalMaskGuard {
    previous: libc::sigset_t,
}

impl SignalMaskGuard {
    fn unblock_for_spawn() -> io::Result<Self> {
        // SAFETY: valid signal-set pointers; pthread_sigmask changes only the
        // calling thread, which is exactly the mask the child inherits.
        unsafe {
            let mut set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            libc::sigaddset(&mut set, libc::SIGINT);
            libc::sigaddset(&mut set, libc::SIGTERM);
            let mut previous: libc::sigset_t = std::mem::zeroed();
            let rc = libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, &mut previous);
            if rc != 0 {
                return Err(io::Error::from_raw_os_error(rc));
            }
            Ok(Self { previous })
        }
    }
}

impl Drop for SignalMaskGuard {
    fn drop(&mut self) {
        // SAFETY: `previous` was returned by pthread_sigmask for this thread.
        unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, &self.previous, std::ptr::null_mut());
        }
    }
}

fn spawn_with_unblocked_signals(command: &mut Command) -> io::Result<Child> {
    let _mask = SignalMaskGuard::unblock_for_spawn()?;
    command.spawn()
}

/// Own a spawned child across fallible setup and unwinding.
///
/// `std::process::Child` detaches on Drop. Every post-spawn path therefore
/// remains armed until the child has either moved into a stronger owner or
/// been synchronously observed as reaped.
struct ArmedChild {
    child: Option<Child>,
}

impl ArmedChild {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn child(&self) -> &Child {
        self.child.as_ref().expect("armed child remains present")
    }

    fn child_mut(&mut self) -> &mut Child {
        self.child.as_mut().expect("armed child remains present")
    }

    fn into_child(mut self) -> Child {
        self.child.take().expect("armed child remains present")
    }

    fn disarm_reaped(&mut self) {
        self.child.take();
    }
}

impl Drop for ArmedChild {
    fn drop(&mut self) {
        let Some(child) = &mut self.child else {
            return;
        };
        if terminate_direct_child_bounded(child).is_err() {
            fail_closed_child_ownership();
        }
    }
}

fn anchor_command() -> io::Result<Command> {
    #[cfg(not(test))]
    {
        // Re-exec the inode that is running this process. The on-disk Cargo
        // target path can be atomically replaced by a concurrent rebuild,
        // which would otherwise let an older parent start a newer binary's
        // private anchor protocol.
        let mut command = Command::new("/proc/self/exe");
        command.env(ANCHOR_MODE_ENV, "1");
        Ok(command)
    }
    #[cfg(test)]
    {
        // A libtest binary does not enter cargo-ktstr's main, so use a shell
        // composed entirely of builtins as the test anchor.
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg(
            "trap '' INT TERM; printf R; \
             while IFS= read -r _ktstr_anchor_line; do :; done",
        );
        Ok(command)
    }
}

fn spawn_anchor() -> io::Result<GroupAnchor> {
    let mut command = anchor_command()?;
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0);
    let mut owner = ArmedChild::new(spawn_with_unblocked_signals(&mut command)?);
    let pgid = libc::pid_t::try_from(owner.child().id())
        .ok()
        .filter(|pid| *pid > 0)
        .ok_or_else(|| io::Error::other("anchor child id is not a valid process-group id"))?;
    let control = owner
        .child_mut()
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("anchor stdin control pipe missing"))?;
    let mut ready = owner
        .child_mut()
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("anchor stdout ready pipe missing"))?;
    let mut anchor = GroupAnchor {
        child: owner.into_child(),
        control: Some(control),
        pgid,
    };
    let byte = match read_anchor_ready(&mut ready, pgid) {
        Ok(byte) => byte,
        Err(error) => {
            if release_anchor(&mut anchor).is_err() {
                fail_closed_child_ownership();
            }
            return Err(error);
        }
    };
    if byte != b'R' {
        if release_anchor(&mut anchor).is_err() {
            fail_closed_child_ownership();
        }
        return Err(io::Error::other("anchor emitted an invalid ready byte"));
    }
    // SAFETY: `pgid` is the live anchor's positive child pid. Readiness means
    // it is already executing its blocking anchor mode.
    let actual_pgid = unsafe { libc::getpgid(pgid) };
    if actual_pgid != pgid {
        if release_anchor(&mut anchor).is_err() {
            fail_closed_child_ownership();
        }
        return Err(io::Error::other(format!(
            "anchor process group was not established (pid {pgid}, pgid {actual_pgid})",
        )));
    }
    Ok(anchor)
}

fn read_anchor_ready(ready: &mut ChildStdout, anchor_pid: libc::pid_t) -> io::Result<u8> {
    set_nonblocking(ready.as_raw_fd())?;
    // The exact unreaped child pid cannot be reused. A pre-ready SIGSTOP must
    // not turn readiness into an unbounded pipe read.
    // SAFETY: signal the exact child pid only.
    let _ = unsafe { libc::kill(anchor_pid, libc::SIGCONT) };
    let deadline = Instant::now() + ANCHOR_REAP_BACKSTOP;
    let mut byte = [0_u8; 1];
    loop {
        match ready.read(&mut byte) {
            Ok(1) => return Ok(byte[0]),
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "process-group anchor closed its ready pipe",
                ));
            }
            Ok(_) => unreachable!("one-byte anchor ready read"),
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
                ) => {}
            Err(error) => return Err(error),
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "process-group anchor did not become ready",
            ));
        }
        std::thread::sleep(GROUP_SCAN_INTERVAL);
    }
}

fn release_anchor(anchor: &mut GroupAnchor) -> io::Result<ExitStatus> {
    drop(anchor.control.take());
    // A child can stop the process-group leader before exiting. Continue the
    // exact anchor pid so closing the control pipe can be observed.
    // SAFETY: the unreaped Child pins this exact pid against reuse.
    let continued = unsafe { libc::kill(anchor.pgid, libc::SIGCONT) };
    if continued != 0 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            return Err(error);
        }
    }

    let started = Instant::now();
    let deadline = started + ANCHOR_REAP_BACKSTOP;
    let kill_at = started + ANCHOR_COOPERATIVE_EXIT_GRACE;
    let mut killed = false;
    loop {
        if let Some(status) = anchor.child.try_wait()? {
            return Ok(status);
        }
        let now = Instant::now();
        if !killed && now >= kill_at {
            // Child::kill targets the exact unreaped child pid, not a numeric
            // lookup that could race reuse.
            if let Err(error) = anchor.child.kill() {
                if let Some(status) = anchor.child.try_wait()? {
                    return Ok(status);
                }
                return Err(error);
            }
            killed = true;
        }
        if now >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("process-group anchor {} did not exit", anchor.pgid),
            ));
        }
        std::thread::sleep(GROUP_SCAN_INTERVAL);
    }
}

fn terminate_direct_child_bounded(child: &mut Child) -> io::Result<()> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }
    if let Err(error) = child.kill() {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        return Err(error);
    }

    let deadline = Instant::now() + FORCED_GROUP_REAP_BACKSTOP;
    loop {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "direct child {} survived SIGKILL without becoming reapable",
                    child.id(),
                ),
            ));
        }
        std::thread::sleep(GROUP_SCAN_INTERVAL);
    }
}

fn abandon_spawning() {
    let transitioned =
        ACTIVE_CHILD_GROUP.compare_exchange(SPAWNING, REAPING, Ordering::SeqCst, Ordering::SeqCst);
    if transitioned != Ok(SPAWNING) {
        fail_closed_child_ownership();
    }
    drain_handlers_in_flight_bounded(None);
    clear_pending_handoff_signals();
    ACTIVE_CHILD_GROUP.store(IDLE, Ordering::SeqCst);
}

fn abandon_spawning_anchor(anchor: &mut GroupAnchor) {
    let transitioned =
        ACTIVE_CHILD_GROUP.compare_exchange(SPAWNING, REAPING, Ordering::SeqCst, Ordering::SeqCst);
    if transitioned != Ok(SPAWNING) {
        let _ = anchor.child.kill();
        fail_closed_child_ownership();
    }
    drain_handlers_in_flight_bounded(Some(anchor.pgid));
    clear_pending_handoff_signals();
    if release_anchor(anchor).is_err() {
        fail_closed_child_ownership();
    }
    ACTIVE_CHILD_GROUP.store(IDLE, Ordering::SeqCst);
}

/// Claim the global handoff slot, create the group anchor, spawn the real
/// command into that group, run the test seam, then publish the anchor pgid.
fn spawn_group<F>(mut command: Command, handoff: F) -> io::Result<GroupChild>
where
    F: FnOnce(),
{
    if caught().is_some() {
        return Err(interrupted_error());
    }
    ACTIVE_CHILD_GROUP
        .compare_exchange(IDLE, SPAWNING, Ordering::SeqCst, Ordering::SeqCst)
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                "another cargo-ktstr child process is already active",
            )
        })?;
    if caught().is_some() {
        abandon_spawning();
        return Err(interrupted_error());
    }

    let mut anchor = match spawn_anchor() {
        Ok(anchor) => anchor,
        Err(error) => {
            abandon_spawning();
            return Err(error);
        }
    };
    if caught().is_some() {
        abandon_spawning_anchor(&mut anchor);
        return Err(interrupted_error());
    }

    command.process_group(anchor.pgid);
    let child = match spawn_with_unblocked_signals(&mut command) {
        Ok(child) => child,
        Err(error) => {
            abandon_spawning_anchor(&mut anchor);
            return Err(error);
        }
    };

    let child_pid = libc::pid_t::try_from(child.id()).ok();
    let mut group = GroupChild {
        child,
        anchor,
        armed: true,
        published: false,
        known_members: child_pid.into_iter().collect(),
    };

    // Deterministic test seam: the armed owner already covers unwinding, the
    // real child exists in the anchor group, and the signal handler still
    // observes SPAWNING.
    handoff();

    let prior = ACTIVE_CHILD_GROUP.compare_exchange(
        SPAWNING,
        group.anchor.pgid,
        Ordering::SeqCst,
        Ordering::SeqCst,
    );
    if prior != Ok(SPAWNING) {
        // Ownership state is corrupt, so normal cleanup cannot safely perform
        // another state transition. Kill the entire known group while its
        // anchor still pins the pgid, then terminate this process.
        // SAFETY: negative pgid targets this freshly created group.
        unsafe {
            libc::kill(-group.anchor.pgid, libc::SIGKILL);
        }
        fail_closed_child_ownership();
    }
    group.published = true;

    // A handler that loaded SPAWNING is counted until it publishes its pending
    // signal. New handlers see the positive anchor pgid and forward directly.
    drain_handlers_in_flight_bounded(Some(group.anchor.pgid));
    replay_handoff_signals(group.anchor.pgid);
    Ok(group)
}

fn replay_handoff_signals(pgid: libc::pid_t) {
    let mut pending_int = PENDING_HANDOFF_SIGINT.swap(0, Ordering::SeqCst);
    let mut pending_term = PENDING_HANDOFF_SIGTERM.swap(0, Ordering::SeqCst);
    let first = CAUGHT_SIGNAL.load(Ordering::SeqCst);

    // Preserve the first delivery, then prioritize the other distinct kind as
    // an escalation before replaying repeats. Exact later inter-kind ordering
    // is not representable without an unbounded signal-handler queue, but no
    // handler-observed delivery is lost.
    match first {
        libc::SIGINT if pending_int > 0 => {
            signal_group(pgid, libc::SIGINT);
            pending_int -= 1;
            forward_n(pgid, libc::SIGTERM, pending_term);
            pending_term = 0;
        }
        libc::SIGTERM if pending_term > 0 => {
            signal_group(pgid, libc::SIGTERM);
            pending_term -= 1;
            forward_n(pgid, libc::SIGINT, pending_int);
            pending_int = 0;
        }
        _ => {}
    }
    forward_n(pgid, libc::SIGINT, pending_int);
    forward_n(pgid, libc::SIGTERM, pending_term);
}

fn forward_n(pgid: libc::pid_t, sig: libc::c_int, count: usize) {
    for _ in 0..count {
        signal_group(pgid, sig);
    }
}

fn clear_pending_handoff_signals() {
    PENDING_HANDOFF_SIGINT.store(0, Ordering::SeqCst);
    PENDING_HANDOFF_SIGTERM.store(0, Ordering::SeqCst);
}

fn signal_group(pgid: libc::pid_t, sig: libc::c_int) {
    #[cfg(test)]
    match sig {
        libc::SIGINT => {
            TEST_FORWARDED_SIGINT.fetch_add(1, Ordering::SeqCst);
        }
        libc::SIGTERM => {
            TEST_FORWARDED_SIGTERM.fetch_add(1, Ordering::SeqCst);
        }
        _ => {}
    }
    // SAFETY: positive `pgid` came from Child::id after setpgid(0, 0);
    // negation selects that process group. ESRCH is an ordinary exit race.
    unsafe {
        libc::kill(-pgid, sig);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ProcessIdentity {
    pid: libc::pid_t,
    starttime_ticks: u64,
}

#[derive(Clone, Copy, Debug)]
struct GroupMember {
    identity: ProcessIdentity,
    pgrp: libc::pid_t,
    cpu_ticks: u64,
    state: u8,
}

fn invalid_proc_stat(pid: libc::pid_t, field: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("/proc/{pid}/stat has invalid {field}"),
    )
}

fn process_gone(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::ENOENT) | Some(libc::ESRCH)
    )
}

fn read_process_stat(pid: libc::pid_t) -> io::Result<Option<GroupMember>> {
    let stat = match std::fs::read(format!("/proc/{pid}/stat")) {
        Ok(stat) => stat,
        Err(error) if process_gone(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    parse_process_stat(pid, &stat).map(Some)
}

fn parse_process_stat(pid: libc::pid_t, stat: &[u8]) -> io::Result<GroupMember> {
    // comm is an arbitrary byte string enclosed in parentheses and can itself
    // contain ')' or non-UTF-8 bytes. Find the last delimiter and parse only
    // the documented ASCII fields which follow it.
    let comm_end = stat
        .iter()
        .rposition(|byte| *byte == b')')
        .ok_or_else(|| invalid_proc_stat(pid, "comm boundary"))?;
    let fields: Vec<_> = stat[comm_end + 1..]
        .split(|byte| byte.is_ascii_whitespace())
        .filter(|field| !field.is_empty())
        .collect();
    if fields.len() <= 19 {
        return Err(invalid_proc_stat(pid, "field count"));
    }
    let state = fields[0]
        .first()
        .copied()
        .ok_or_else(|| invalid_proc_stat(pid, "state"))?;
    let pgrp = parse_proc_stat_number::<libc::pid_t>(fields[2])
        .ok_or_else(|| invalid_proc_stat(pid, "process group"))?;
    let utime = parse_proc_stat_number::<u64>(fields[11])
        .ok_or_else(|| invalid_proc_stat(pid, "utime"))?;
    let stime = parse_proc_stat_number::<u64>(fields[12])
        .ok_or_else(|| invalid_proc_stat(pid, "stime"))?;
    let starttime_ticks = parse_proc_stat_number::<u64>(fields[19])
        .ok_or_else(|| invalid_proc_stat(pid, "starttime"))?;
    Ok(GroupMember {
        identity: ProcessIdentity {
            pid,
            starttime_ticks,
        },
        pgrp,
        cpu_ticks: utime.saturating_add(stime),
        state,
    })
}

fn parse_proc_stat_number<T>(field: &[u8]) -> Option<T>
where
    T: std::str::FromStr,
{
    std::str::from_utf8(field)
        .ok()?
        .parse::<T>()
        .ok()
}

fn unreadable_proc_entry_is_unrelated(
    entry: &std::fs::DirEntry,
    pid: libc::pid_t,
    known_members: &HashSet<libc::pid_t>,
) -> io::Result<bool> {
    if known_members.contains(&pid) {
        return Ok(false);
    }
    let metadata = match entry.metadata() {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(true),
        Err(error) => return Err(error),
    };
    // SAFETY: geteuid has no arguments and cannot fail.
    let own_euid = unsafe { libc::geteuid() };
    Ok(proc_owner_is_unrelated(
        pid,
        known_members,
        metadata.uid(),
        own_euid,
    ))
}

fn proc_owner_is_unrelated(
    pid: libc::pid_t,
    known_members: &HashSet<libc::pid_t>,
    owner_uid: libc::uid_t,
    own_euid: libc::uid_t,
) -> bool {
    !known_members.contains(&pid) && owner_uid != own_euid
}

fn process_group_members(
    pgid: libc::pid_t,
    known_members: &HashSet<libc::pid_t>,
) -> io::Result<Vec<GroupMember>> {
    let mut members = Vec::new();
    for entry in std::fs::read_dir("/proc")? {
        let entry = entry?;
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<libc::pid_t>().ok())
        else {
            continue;
        };
        if pid == pgid {
            continue;
        }
        let member = match read_process_stat(pid) {
            Ok(Some(member)) => member,
            Ok(None) => continue,
            Err(error)
                if error.kind() == io::ErrorKind::PermissionDenied
                    && unreadable_proc_entry_is_unrelated(&entry, pid, known_members)? =>
            {
                // hidepid may expose a numeric directory while denying stat
                // for another uid. Such a process cannot join this caller's
                // group without first becoming observable. A known group pid
                // or same-euid pid remains a hard error above.
                continue;
            }
            Err(error) => return Err(error),
        };
        if member.pgrp == pgid && member.state != b'Z' && member.state != b'X' {
            members.push(member);
        }
    }
    members.sort_by_key(|member| member.identity.pid);
    Ok(members)
}

fn snapshot_group_members(group: &mut GroupChild) -> io::Result<Vec<GroupMember>> {
    let members = process_group_members(group.anchor.pgid, &group.known_members)?;
    group
        .known_members
        .extend(members.iter().map(|member| member.identity.pid));
    Ok(members)
}

struct GroupTailBudget {
    started: Instant,
    ticks_per_second: u64,
    prior_ticks: HashMap<ProcessIdentity, u64>,
    delivered_ticks: u128,
}

impl GroupTailBudget {
    fn start(now: Instant, members: &[GroupMember]) -> io::Result<Self> {
        // SAFETY: sysconf with _SC_CLK_TCK has no pointers or side effects.
        let ticks_per_second = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        if ticks_per_second <= 0 {
            return Err(io::Error::other(
                "sysconf(_SC_CLK_TCK) failed for child-group service accounting",
            ));
        }
        Ok(Self {
            started: now,
            ticks_per_second: ticks_per_second as u64,
            prior_ticks: members
                .iter()
                .map(|member| (member.identity, member.cpu_ticks))
                .collect(),
            delivered_ticks: 0,
        })
    }

    fn observe(&mut self, members: &[GroupMember]) {
        let mut current = HashMap::with_capacity(members.len());
        for member in members {
            if let Some(previous) = self.prior_ticks.get(&member.identity) {
                self.delivered_ticks = self
                    .delivered_ticks
                    .saturating_add(member.cpu_ticks.saturating_sub(*previous) as u128);
            }
            current.insert(member.identity, member.cpu_ticks);
        }
        self.prior_ticks = current;
    }

    fn service_exhausted(&self) -> bool {
        self.delivered_ticks.saturating_mul(1_000_000_000)
            > GROUP_TAIL_SERVICE_BUDGET_NS.saturating_mul(self.ticks_per_second as u128)
    }

    fn wall_exhausted(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.started) > GROUP_TAIL_WALL_BACKSTOP
    }
}

fn observer_sleep<O: StdoutObserver>(observer: &O, maximum: Duration) -> Duration {
    observer
        .next_tick_in()
        .min(maximum)
        .max(Duration::from_millis(1))
}

/// Move the anchor pgid into REAPING, drain old handler readers, then close
/// the control pipe and reap the anchor before exposing IDLE.
fn finish_anchor(anchor: &mut GroupAnchor) -> io::Result<()> {
    let transitioned = ACTIVE_CHILD_GROUP.compare_exchange(
        anchor.pgid,
        REAPING,
        Ordering::SeqCst,
        Ordering::SeqCst,
    );
    if transitioned != Ok(anchor.pgid) {
        fail_closed_child_ownership();
    }
    drain_handlers_in_flight_bounded(Some(anchor.pgid));
    if release_anchor(anchor).is_err() {
        fail_closed_child_ownership();
    }
    ACTIVE_CHILD_GROUP.store(IDLE, Ordering::SeqCst);
    Ok(())
}

fn open_verified_member_pidfd(
    member: GroupMember,
    pgid: libc::pid_t,
) -> io::Result<Option<OwnedFd>> {
    // SAFETY: pidfd_open takes an integer pid and flags=0, returning a new fd.
    let raw =
        unsafe { libc::syscall(libc::SYS_pidfd_open, member.identity.pid, 0_u32) as libc::c_int };
    if raw < 0 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(None);
        }
        return Err(error);
    }
    // SAFETY: pidfd_open returned a fresh descriptor owned by this function.
    let pidfd = unsafe { OwnedFd::from_raw_fd(raw) };
    let Some(current) = read_process_stat(member.identity.pid)? else {
        return Ok(None);
    };
    if current.identity != member.identity
        || current.pgrp != pgid
        || current.state == b'Z'
        || current.state == b'X'
    {
        return Ok(None);
    }
    Ok(Some(pidfd))
}

fn signal_member_pidfd(pidfd: &OwnedFd, signal: libc::c_int) -> io::Result<()> {
    // SAFETY: pidfd_send_signal receives a live pidfd, a valid signal, and
    // null siginfo/flags as documented.
    let result = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            pidfd.as_raw_fd(),
            signal,
            std::ptr::null::<libc::siginfo_t>(),
            0_u32,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(error)
        }
    }
}

fn force_reap_group_members(group: &mut GroupChild) -> io::Result<()> {
    let pgid = group.anchor.pgid;
    let deadline = Instant::now() + FORCED_GROUP_REAP_BACKSTOP;
    let mut empty_snapshots = 0;
    loop {
        let members = snapshot_group_members(group)?;
        if members.is_empty() {
            empty_snapshots += 1;
            if empty_snapshots == 2 {
                return Ok(());
            }
        } else {
            empty_snapshots = 0;
            for member in members {
                if let Some(pidfd) = open_verified_member_pidfd(member, pgid)? {
                    signal_member_pidfd(&pidfd, libc::SIGKILL)?;
                }
            }
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("child process group {pgid} survived SIGKILL"),
            ));
        }
        std::thread::sleep(GROUP_SCAN_INTERVAL);
    }
}

/// Drain signal handlers that may still own a published child-group value.
///
/// Every caller runs in normal context after first hiding the state that the
/// handler was allowed to load. A handler is expected to consume only a few
/// lock-free operations and syscalls, but scheduler starvation, ptrace, or a
/// kernel fault must not turn cleanup into an unbounded spin. On expiry the
/// caller kills its still-anchor-pinned group, when one exists, and terminates
/// the parent rather than detaching work under ambiguous ownership.
fn drain_handlers_in_flight_bounded(owned_pgid: Option<libc::pid_t>) {
    if handlers_in_flight_drain_within(HANDLER_DRAIN_BACKSTOP) {
        return;
    }
    if let Some(pgid) = owned_pgid.filter(|pgid| *pgid > 0) {
        // SAFETY: callers pass only the live anchor-pinned group whose
        // publication they just hid from new handler entries.
        unsafe {
            libc::kill(-pgid, libc::SIGKILL);
        }
    }
    fail_closed_child_ownership();
}

fn handlers_in_flight_drain_within(timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if HANDLERS_IN_FLIGHT.load(Ordering::SeqCst) == 0 {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::yield_now();
    }
}

#[cold]
fn fail_closed_child_ownership() -> ! {
    // SAFETY: async-signal-safe, allocation-free last resort. Returning after
    // losing exact ownership would permit pgid reuse under a stale handler.
    unsafe {
        let flags = libc::fcntl(libc::STDERR_FILENO, libc::F_GETFL);
        if flags >= 0 {
            let _ = libc::fcntl(libc::STDERR_FILENO, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }
        let _ = libc::write(
            libc::STDERR_FILENO,
            CHILD_OWNERSHIP_FAIL_CLOSED_DIAGNOSTIC.as_ptr().cast(),
            CHILD_OWNERSHIP_FAIL_CLOSED_DIAGNOSTIC.len(),
        );
        libc::_exit(CHILD_OWNERSHIP_FAIL_CLOSED_EXIT_CODE);
    }
}

fn terminate_group(group: &mut GroupChild) {
    if !group.armed {
        return;
    }
    let expected = if group.published {
        group.anchor.pgid
    } else {
        SPAWNING
    };
    if ACTIVE_CHILD_GROUP.compare_exchange(expected, REAPING, Ordering::SeqCst, Ordering::SeqCst)
        != Ok(expected)
    {
        // The known group is still exact even though the ownership state is
        // corrupt. Prevent descendants surviving the fail-closed parent.
        // SAFETY: negative pgid targets this anchor-pinned group.
        unsafe {
            libc::kill(-group.anchor.pgid, libc::SIGKILL);
        }
        fail_closed_child_ownership();
    }
    drain_handlers_in_flight_bounded(Some(group.anchor.pgid));
    if !group.published {
        clear_pending_handoff_signals();
    }

    // The anchor ignores TERM and therefore continues to pin the pgid while
    // exact pidfds kill every non-anchor member. Never SIGKILL the group as a
    // whole: that would turn the ownership anchor into a zombie before the
    // rescan fence completed.
    signal_group(group.anchor.pgid, libc::SIGTERM);
    if force_reap_group_members(group).is_err() {
        fail_closed_child_ownership();
    }
    let leader_deadline = Instant::now() + FORCED_GROUP_REAP_BACKSTOP;
    loop {
        match group.child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < leader_deadline => {
                std::thread::sleep(GROUP_SCAN_INTERVAL);
            }
            _ => fail_closed_child_ownership(),
        }
    }
    if release_anchor(&mut group.anchor).is_err() {
        fail_closed_child_ownership();
    }
    ACTIVE_CHILD_GROUP.store(IDLE, Ordering::SeqCst);
    group.armed = false;
}

fn finish_group(group: &mut GroupChild) -> io::Result<()> {
    if !group.armed || !group.published {
        fail_closed_child_ownership();
    }
    finish_anchor(&mut group.anchor)?;
    group.armed = false;
    Ok(())
}

impl Drop for GroupChild {
    fn drop(&mut self) {
        if self.armed {
            terminate_group(self);
        }
    }
}

fn run_status_with_handoff<F>(command: Command, handoff: F) -> io::Result<ExitStatus>
where
    F: FnOnce(),
{
    let mut group = spawn_group(command, handoff)?;
    let mut observer = SilentObserver;
    let status = match drain_group_capture(&mut group, CaptureStreams::empty(), &mut observer) {
        Ok(capture) => capture.status,
        Err(error) => {
            terminate_group(&mut group);
            return Err(error);
        }
    };
    finish_group(&mut group)?;
    Ok(status)
}

struct SilentObserver;

impl StdoutObserver for SilentObserver {
    fn observe_stdout(&mut self, _bytes: &[u8]) {}

    fn tick(&mut self) {}

    fn next_tick_in(&self) -> Duration {
        Duration::from_secs(1)
    }

    fn finished(&mut self, _status: &ExitStatus) {}

    fn failed(&mut self, _error: &io::Error) {}
}

struct CaptureStreams {
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
    stdout_bytes: Vec<u8>,
    stderr_bytes: Vec<u8>,
}

impl CaptureStreams {
    fn empty() -> Self {
        Self {
            stdout: None,
            stderr: None,
            stdout_bytes: Vec::new(),
            stderr_bytes: Vec::new(),
        }
    }

    fn new(stdout: ChildStdout, stderr: Option<ChildStderr>) -> io::Result<Self> {
        set_nonblocking(stdout.as_raw_fd())?;
        if let Some(stderr) = &stderr {
            set_nonblocking(stderr.as_raw_fd())?;
        }
        Ok(Self {
            stdout: Some(stdout),
            stderr,
            stdout_bytes: Vec::new(),
            stderr_bytes: Vec::new(),
        })
    }

    fn has_open_pipe(&self) -> bool {
        self.stdout.is_some() || self.stderr.is_some()
    }

    fn poll_once<O: StdoutObserver>(
        &mut self,
        timeout: Duration,
        observer: &mut O,
    ) -> io::Result<()> {
        let mut fds = [
            libc::pollfd {
                fd: self.stdout.as_ref().map_or(-1, |stdout| stdout.as_raw_fd()),
                events: libc::POLLIN | libc::POLLHUP | libc::POLLERR,
                revents: 0,
            },
            libc::pollfd {
                fd: self.stderr.as_ref().map_or(-1, |stderr| stderr.as_raw_fd()),
                events: libc::POLLIN | libc::POLLHUP | libc::POLLERR,
                revents: 0,
            },
        ];
        // SAFETY: `fds` is a pair of initialized pollfd entries. Negative fds
        // are ignored by poll; the timeout is finite.
        let result = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, timeout_ms(timeout)) };
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                return Ok(());
            }
            return Err(error);
        }
        if result == 0 {
            return Ok(());
        }
        if fds
            .iter()
            .any(|pollfd| pollfd.revents & libc::POLLNVAL != 0)
        {
            return Err(io::Error::other(
                "cargo-ktstr child output pipe became invalid while polling",
            ));
        }

        let mut buffer = [0_u8; 64 * 1024];
        if fds[0].revents != 0 {
            read_pipe_once(
                &mut self.stdout,
                &mut self.stdout_bytes,
                &mut buffer,
                |bytes| observer.observe_stdout(bytes),
            )?;
        }
        if fds[1].revents != 0 {
            read_pipe_once(
                &mut self.stderr,
                &mut self.stderr_bytes,
                &mut buffer,
                |bytes| observer.observe_stderr(bytes),
            )?;
        }
        Ok(())
    }

    /// Drain bytes already queued at the exact point the anchored group
    /// becomes quiescent, then close any still-open pipe. A remaining writer
    /// has escaped the owned process group; later output is not ours to wait
    /// for and cannot hold this invocation hostage.
    fn finish_owned_prefix<O: StdoutObserver>(&mut self, observer: &mut O) -> io::Result<()> {
        if let Some(stdout) = &mut self.stdout {
            let available = pending_pipe_bytes(stdout.as_raw_fd())?;
            drain_exact_available(stdout, &mut self.stdout_bytes, available, |bytes| {
                observer.observe_stdout(bytes)
            })?;
        }
        self.stdout.take();

        if let Some(stderr) = &mut self.stderr {
            let available = pending_pipe_bytes(stderr.as_raw_fd())?;
            drain_exact_available(stderr, &mut self.stderr_bytes, available, |bytes| {
                observer.observe_stderr(bytes)
            })?;
        }
        self.stderr.take();
        Ok(())
    }

    fn into_output(self, status: ExitStatus) -> Output {
        Output {
            status,
            stdout: self.stdout_bytes,
            stderr: self.stderr_bytes,
        }
    }
}

fn set_nonblocking(fd: libc::c_int) -> io::Result<()> {
    // SAFETY: fcntl operates on a live pipe fd owned by the caller.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: preserve all existing status flags and add O_NONBLOCK.
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn pending_pipe_bytes(fd: libc::c_int) -> io::Result<usize> {
    let mut pending: libc::c_int = 0;
    // SAFETY: FIONREAD writes one integer to the valid out-pointer.
    if unsafe { libc::ioctl(fd, libc::FIONREAD, &mut pending) } < 0 {
        return Err(io::Error::last_os_error());
    }
    usize::try_from(pending)
        .map_err(|_| io::Error::other("child output pipe reported a negative byte count"))
}

fn read_pipe_once<R: Read>(
    reader: &mut Option<R>,
    output: &mut Vec<u8>,
    buffer: &mut [u8],
    mut observed: impl FnMut(&[u8]),
) -> io::Result<()> {
    let Some(pipe) = reader.as_mut() else {
        return Ok(());
    };
    match pipe.read(buffer) {
        Ok(0) => {
            reader.take();
            Ok(())
        }
        Ok(read) => {
            output.extend_from_slice(&buffer[..read]);
            observed(&buffer[..read]);
            Ok(())
        }
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
            ) =>
        {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn drain_exact_available<R: Read>(
    reader: &mut R,
    output: &mut Vec<u8>,
    mut remaining: usize,
    mut observed: impl FnMut(&[u8]),
) -> io::Result<()> {
    let mut buffer = [0_u8; 64 * 1024];
    while remaining != 0 {
        let wanted = remaining.min(buffer.len());
        match reader.read(&mut buffer[..wanted]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "child output pipe closed before its FIONREAD snapshot was drained",
                ));
            }
            Ok(read) => {
                output.extend_from_slice(&buffer[..read]);
                observed(&buffer[..read]);
                remaining -= read;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                return Err(io::Error::other(
                    "child output pipe lost bytes reported by FIONREAD",
                ));
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn timeout_ms(duration: Duration) -> libc::c_int {
    let partial_millisecond = if duration.subsec_nanos() % 1_000_000 != 0 {
        1
    } else {
        0
    };
    let millis = duration
        .as_millis()
        .saturating_add(partial_millisecond)
        .max(1);
    millis
        .min(libc::c_int::MAX as u128)
        .try_into()
        .unwrap_or(libc::c_int::MAX)
}

struct GroupCapture {
    status: ExitStatus,
    streams: CaptureStreams,
}

fn drain_group_capture<O: StdoutObserver>(
    group: &mut GroupChild,
    mut streams: CaptureStreams,
    observer: &mut O,
) -> io::Result<GroupCapture> {
    let mut leader_status = None;
    let mut budget = None::<GroupTailBudget>;
    let mut next_group_scan = Instant::now();
    let mut empty_snapshots = 0;

    loop {
        observer.tick();
        let now = Instant::now();
        if leader_status.is_none() {
            leader_status = group.child.try_wait()?;
            if leader_status.is_some() {
                next_group_scan = now;
            }
        }

        if leader_status.is_some() && now >= next_group_scan {
            let members = snapshot_group_members(group)?;
            match &mut budget {
                Some(budget) => budget.observe(&members),
                None => budget = Some(GroupTailBudget::start(now, &members)?),
            }

            if members.is_empty() {
                empty_snapshots += 1;
                if empty_snapshots == 2 {
                    streams.finish_owned_prefix(observer)?;
                    return Ok(GroupCapture {
                        status: leader_status.expect("leader status present"),
                        streams,
                    });
                }
            } else {
                empty_snapshots = 0;
            }

            let budget = budget.as_ref().expect("tail budget initialized");
            if budget.service_exhausted() || budget.wall_exhausted(now) {
                // Give cooperative teardown one TERM edge before the caller's
                // exact pidfd-verified SIGKILL sweep.
                signal_group(group.anchor.pgid, libc::SIGTERM);
                let cause = if budget.service_exhausted() {
                    "CPU-service budget"
                } else {
                    "wall backstop"
                };
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "child process group {} exceeded its post-leader {cause}",
                        group.anchor.pgid,
                    ),
                ));
            }
            next_group_scan = now + GROUP_SCAN_INTERVAL;
        }

        let maximum_wait = if streams.has_open_pipe() {
            Duration::from_millis(100)
        } else {
            Duration::from_millis(20)
        };
        let mut wait = observer_sleep(observer, maximum_wait);
        if leader_status.is_some() {
            wait = wait.min(
                next_group_scan
                    .saturating_duration_since(Instant::now())
                    .max(Duration::from_millis(1)),
            );
            if let Some(budget) = &budget {
                wait = wait.min(
                    GROUP_TAIL_WALL_BACKSTOP
                        .saturating_sub(budget.started.elapsed())
                        .max(Duration::from_millis(1)),
                );
            }
        }
        streams.poll_once(wait, observer)?;
    }
}

fn drain_direct_capture<O: StdoutObserver>(
    child: &mut Child,
    mut streams: CaptureStreams,
    observer: &mut O,
) -> io::Result<Output> {
    loop {
        observer.tick();
        if let Some(status) = child.try_wait()? {
            streams.finish_owned_prefix(observer)?;
            return Ok(streams.into_output(status));
        }
        let maximum_wait = if streams.has_open_pipe() {
            Duration::from_millis(100)
        } else {
            Duration::from_millis(20)
        };
        streams.poll_once(observer_sleep(observer, maximum_wait), observer)?;
    }
}

fn run_observed_group<O: StdoutObserver>(
    command: Command,
    observer: &mut O,
    capture_stderr: bool,
) -> io::Result<Output> {
    let mut group = spawn_group(command, || {})?;
    let stdout = match group.child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            terminate_group(&mut group);
            return Err(io::Error::other(
                "cargo-ktstr child stdout was not configured as a pipe",
            ));
        }
    };
    let stderr = if capture_stderr {
        match group.child.stderr.take() {
            Some(stderr) => Some(stderr),
            None => {
                terminate_group(&mut group);
                return Err(io::Error::other(
                    "cargo-ktstr child stderr was not configured as a pipe",
                ));
            }
        }
    } else {
        None
    };
    let streams = match CaptureStreams::new(stdout, stderr) {
        Ok(streams) => streams,
        Err(error) => {
            terminate_group(&mut group);
            return Err(error);
        }
    };
    let capture = match drain_group_capture(&mut group, streams, observer) {
        Ok(capture) => capture,
        Err(error) => {
            terminate_group(&mut group);
            return Err(error);
        }
    };
    finish_group(&mut group)?;
    Ok(capture.streams.into_output(capture.status))
}

fn run_observed_direct<O: StdoutObserver>(
    mut command: Command,
    observer: &mut O,
    capture_stderr: bool,
) -> io::Result<Output> {
    let mut owner = ArmedChild::new(command.spawn()?);
    let stdout = match owner.child_mut().stdout.take() {
        Some(stdout) => stdout,
        None => {
            return Err(io::Error::other(
                "cargo-ktstr child stdout was not configured as a pipe",
            ));
        }
    };
    let stderr = if capture_stderr {
        match owner.child_mut().stderr.take() {
            Some(stderr) => Some(stderr),
            None => {
                return Err(io::Error::other(
                    "cargo-ktstr child stderr was not configured as a pipe",
                ));
            }
        }
    } else {
        None
    };
    let streams = match CaptureStreams::new(stdout, stderr) {
        Ok(streams) => streams,
        Err(error) => return Err(error),
    };
    let output = drain_direct_capture(owner.child_mut(), streams, observer)?;
    owner.disarm_reaped();
    Ok(output)
}

fn run_capture(mut command: Command, capture_stderr: bool) -> io::Result<Output> {
    command.stdout(Stdio::piped());
    if capture_stderr {
        command.stderr(Stdio::piped());
    }

    let mut group = spawn_group(command, || {})?;
    let stdout = match group.child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            terminate_group(&mut group);
            return Err(io::Error::other(
                "cargo-ktstr child stdout was not configured as a pipe",
            ));
        }
    };
    let stderr = if capture_stderr {
        match group.child.stderr.take() {
            Some(stderr) => Some(stderr),
            None => {
                terminate_group(&mut group);
                return Err(io::Error::other(
                    "cargo-ktstr child stderr was not configured as a pipe",
                ));
            }
        }
    } else {
        None
    };
    let streams = match CaptureStreams::new(stdout, stderr) {
        Ok(streams) => streams,
        Err(error) => {
            terminate_group(&mut group);
            return Err(error);
        }
    };
    let mut observer = SilentObserver;
    let capture = match drain_group_capture(&mut group, streams, &mut observer) {
        Ok(capture) => capture,
        Err(error) => {
            terminate_group(&mut group);
            return Err(error);
        }
    };
    finish_group(&mut group)?;
    Ok(capture.streams.into_output(capture.status))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::sync::{Arc, Mutex, mpsc};
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    const STARTUP_TEST_OWNER_ENV: &str = "__KTSTR_TEST_STARTUP_OWNER";

    fn test_pidfd_open(pid: libc::pid_t) -> OwnedFd {
        let raw =
            unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0_u32) as libc::c_int };
        assert!(
            raw >= 0,
            "pidfd_open({pid}) failed: {}",
            io::Error::last_os_error(),
        );
        unsafe { OwnedFd::from_raw_fd(raw) }
    }

    fn test_pidfd_signal(pidfd: &OwnedFd, signal: libc::c_int) {
        let result = unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                pidfd.as_raw_fd(),
                signal,
                std::ptr::null::<libc::siginfo_t>(),
                0_u32,
            )
        };
        assert_eq!(
            result,
            0,
            "pidfd signal {signal} failed: {}",
            io::Error::last_os_error(),
        );
    }

    fn test_pidfd_kill(pidfd: &OwnedFd) {
        test_pidfd_signal(pidfd, libc::SIGKILL);
    }

    fn test_pidfd_exited(pidfd: &OwnedFd) -> bool {
        let mut pollfd = libc::pollfd {
            fd: pidfd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let result = unsafe { libc::poll(&mut pollfd, 1, 0) };
        result == 1 && pollfd.revents & libc::POLLIN != 0
    }

    fn append_startup_test_pid(pidfile: &std::path::Path, pid: libc::pid_t) {
        use std::fs::OpenOptions;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(pidfile)
            .expect("open startup supervision pidfile");
        writeln!(file, "{pid}").expect("append startup supervision pid");
        file.flush().expect("flush startup supervision pid");
    }

    fn fork_startup_test_pause_child(pidfile: &std::path::Path) -> libc::pid_t {
        // SAFETY: the child performs only pause/_exit after fork.
        let child = unsafe { libc::fork() };
        assert!(child >= 0, "fork startup test pause child");
        if child == 0 {
            loop {
                unsafe {
                    libc::pause();
                }
            }
        }
        append_startup_test_pid(pidfile, child);
        child
    }

    fn spawn_startup_test_exec_child(pidfile: &std::path::Path) -> libc::pid_t {
        let child = Command::new("/bin/sleep")
            .arg("300")
            .spawn()
            .expect("spawn startup test exec child");
        let pid = child.id() as libc::pid_t;
        append_startup_test_pid(pidfile, pid);
        drop(child);
        pid
    }

    fn take_startup_test_worker_control() -> OwnedFd {
        let raw = STARTUP_WORKER_CLEAN_FD.swap(-1, Ordering::SeqCst);
        assert!(raw >= 0, "startup worker control descriptor is installed");
        // SAFETY: the atomic transfer gives this helper sole ownership.
        unsafe { OwnedFd::from_raw_fd(raw) }
    }

    fn assert_startup_context_fidelity() {
        use std::os::unix::ffi::OsStrExt;

        assert_eq!(
            std::env::args_os()
                .next()
                .expect("startup context argv0")
                .as_bytes(),
            b"ktstr-context-argv0",
        );
        assert_eq!(
            std::env::var_os(std::ffi::OsStr::from_bytes(
                b"KTSTR_CONTEXT_\xff",
            ))
            .as_deref()
            .map(|value| value.as_bytes()),
            Some(b"value_\xfe".as_slice()),
        );
        assert_eq!(
            std::env::current_dir()
                .expect("startup context cwd")
                .file_name()
                .expect("startup context cwd leaf")
                .as_bytes(),
            b"cwd-\xfd",
        );
        let mut stdin = [0_u8; 1];
        std::io::stdin()
            .read_exact(&mut stdin)
            .expect("read inherited startup stdin");
        assert_eq!(stdin, [0xa5]);
        write_startup_frame(libc::STDOUT_FILENO, b"\0KTSTR-STDOUT-\xff\0")
            .expect("write inherited startup stdout");
        write_startup_frame(libc::STDERR_FILENO, b"\0KTSTR-STDERR-\xfe\0")
            .expect("write inherited startup stderr");
    }

    fn startup_test_worker() {
        let pidfile = std::path::PathBuf::from(
            std::env::var_os(STARTUP_TEST_PIDFILE_ENV)
                .expect("startup supervision pidfile env"),
        );
        append_startup_test_pid(&pidfile, unsafe { libc::getpid() });
        let mode = std::env::var(STARTUP_TEST_MODE_ENV).expect("startup test mode");
        if mode == "pre-ready" {
            loop {
                std::thread::sleep(Duration::from_secs(60));
            }
        }
        if mode == "delayed-ready" {
            let release = pidfile.with_extension("release");
            while !release.exists() {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        if mode == "pgrp-mismatch" {
            assert_eq!(
                unsafe { libc::setpgid(0, 0) },
                0,
                "move nested worker into an unexpected process group",
            );
        }
        startup_worker_bootstrap().expect("bootstrap nested test worker");
        if mode == "context-fidelity" {
            assert_startup_context_fidelity();
            commit_startup_worker_exit(0);
            return;
        }
        if matches!(
            mode.as_str(),
            "clean-fork"
                | "sigchld-ignore-clean"
                | "owner-death-pre-ack"
                | "owner-death-post-ack"
                | "clean-nonzero"
        ) {
            // The fork-only child deliberately retains the CLOEXEC
            // worker-control fd without an exec.
            let _child = fork_startup_test_pause_child(&pidfile);
            let code = if mode == "clean-nonzero" { 23 } else { 0 };
            commit_startup_worker_exit(code);
            if code != 0 {
                unsafe { libc::_exit(i32::from(code)) }
            }
            return;
        }
        if mode == "malformed-worker-control" || mode == "truncated-worker-control" {
            // This child execs, so it cannot pin the CLOEXEC control writer.
            let _child = spawn_startup_test_exec_child(&pidfile);
            let phase = if mode == "malformed-worker-control" {
                "malformed-control"
            } else {
                "truncated-control"
            };
            startup_test_checkpoint_if(&mode, phase, true)
                .expect("release startup worker control fault");
            let control = take_startup_test_worker_control();
            if mode == "malformed-worker-control" {
                write_startup_frame(control.as_raw_fd(), &[b'X'; 9])
                    .expect("write malformed startup worker frame");
            } else {
                write_startup_frame(
                    control.as_raw_fd(),
                    &STARTUP_WORKER_CLEAN_MAGIC[..4],
                )
                .expect("write truncated startup worker frame");
            }
            drop(control);
            loop {
                std::thread::sleep(Duration::from_secs(60));
            }
        }

        let pidfile_text = pidfile.to_string_lossy().into_owned();
        let mut same_group = Command::new("/bin/sh");
        same_group
            .arg("-c")
            .arg(
                "printf '%s\n' \"$$\" >> \"$KTSTR_STARTUP_PIDFILE\"; \
                 exec /bin/sleep 300",
            )
            .env("KTSTR_STARTUP_PIDFILE", &pidfile_text);
        let _same_group = same_group.spawn().expect("spawn same-group descendant");

        let mut separate_group = Command::new("/bin/sh");
        separate_group
            .arg("-c")
            .arg(
                "printf '%s\n' \"$$\" >> \"$KTSTR_STARTUP_PIDFILE\"; \
                 exec /bin/sleep 300",
            )
            .env("KTSTR_STARTUP_PIDFILE", &pidfile_text)
            .process_group(0);
        let _separate_group = separate_group
            .spawn()
            .expect("spawn explicit-setpgid descendant");

        let mut new_session = Command::new("/usr/bin/setsid");
        new_session
            .arg("/usr/bin/env")
            .arg("-i")
            .arg("/bin/sh")
            .arg("-c")
            .arg("printf '%s\n' \"$$\" >> \"$1\"; exec /bin/sleep 300")
            .arg("ktstr-startup-test")
            .arg(&pidfile_text);
        let _new_session = new_session
            .spawn()
            .expect("spawn setsid env-i descendant");

        let mut double_fork = Command::new("/bin/sh");
        double_fork
            .arg("-c")
            .arg(
                "( /bin/sh -c \
                    'printf \"%s\\n\" \"$$\" >> \"$KTSTR_STARTUP_PIDFILE\"; \
                     exec /bin/sleep 300' & ); \
                 exec /bin/sleep 300",
            )
            .env("KTSTR_STARTUP_PIDFILE", &pidfile_text);
        let _double_fork = double_fork.spawn().expect("spawn double-fork chain");

        loop {
            std::thread::sleep(Duration::from_secs(60));
        }
    }

    fn startup_test_role_if_requested() -> bool {
        match std::env::var(STARTUP_ROLE_ENV).as_deref() {
            Ok(STARTUP_ROLE_SUPERVISOR) => startup_supervisor_main(),
            Ok(STARTUP_ROLE_WORKER) => {
                startup_test_worker();
                true
            }
            _ if std::env::var_os(STARTUP_TEST_OWNER_ENV).is_some() => {
                if std::env::var(STARTUP_TEST_MODE_ENV).as_deref()
                    == Ok("sigchld-ignore-clean")
                {
                    unsafe {
                        let mut ignore: libc::sigaction = std::mem::zeroed();
                        ignore.sa_sigaction = libc::SIG_IGN;
                        libc::sigemptyset(&mut ignore.sa_mask);
                        assert_eq!(
                            libc::sigaction(
                                libc::SIGCHLD,
                                &ignore,
                                std::ptr::null_mut(),
                            ),
                            0,
                            "install inherited SIGCHLD ignore",
                        );
                    }
                }
                startup_launcher_main()
            }
            _ => false,
        }
    }

    fn spawn_startup_test_owner(
        test_name: &str,
        mode: &str,
        pidfile: &std::path::Path,
    ) -> Child {
        let mut command = Command::new(std::env::current_exe().expect("current test executable"));
        command
            .arg("--exact")
            .arg(test_name)
            .arg("--nocapture")
            .env(STARTUP_TEST_OWNER_ENV, "1")
            .env(STARTUP_TEST_MODE_ENV, mode)
            .env(STARTUP_TEST_PIDFILE_ENV, pidfile)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        command.spawn().expect("spawn nested startup-test owner")
    }

    fn startup_test_phase_path(
        pidfile: &std::path::Path,
        phase: &str,
        suffix: &str,
    ) -> std::path::PathBuf {
        let mut path = pidfile.to_owned();
        path.set_extension(format!("{phase}.{suffix}"));
        path
    }

    fn startup_test_checkpoint_pid(
        pidfile: &std::path::Path,
        phase: &str,
    ) -> libc::pid_t {
        let marker = startup_test_phase_path(pidfile, phase, "ready");
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || marker.exists()),
            "startup test checkpoint {phase} was not reached",
        );
        std::fs::read_to_string(marker)
            .expect("read startup test checkpoint")
            .trim()
            .parse()
            .expect("parse startup test checkpoint pid")
    }

    fn release_startup_test_checkpoint(pidfile: &std::path::Path, phase: &str) {
        std::fs::write(
            startup_test_phase_path(pidfile, phase, "release"),
            b"release",
        )
        .expect("release startup test checkpoint");
    }

    fn wait_startup_test_owner(owner: &mut Child, label: &str) -> ExitStatus {
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                owner.try_wait().expect("query nested startup owner").is_some()
            }),
            "{label} startup owner did not exit within its bounded deadline",
        );
        owner.wait().expect("reap nested startup owner")
    }

    fn assert_startup_fail_closed(status: &ExitStatus) {
        assert_eq!(
            status.code(),
            Some(CHILD_OWNERSHIP_FAIL_CLOSED_EXIT_CODE),
            "startup protocol fault must use the fail-closed exit status",
        );
    }

    fn startup_test_pids(
        pidfile: &std::path::Path,
        count: usize,
    ) -> Vec<libc::pid_t> {
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                std::fs::read_to_string(pidfile)
                    .ok()
                    .is_some_and(|contents| contents.lines().count() >= count)
            }),
            "nested startup worker did not publish {count} pids",
        );
        std::fs::read_to_string(pidfile)
            .expect("read nested startup pids")
            .lines()
            .map(|line| line.parse().expect("parse nested startup pid"))
            .collect()
    }

    fn assert_startup_pids_exit(
        pids: &[libc::pid_t],
        pidfds: &[OwnedFd],
    ) {
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                pidfds.iter().all(test_pidfd_exited)
            }),
            "startup supervisor left at least one exact descendant alive: {pids:?}",
        );
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                pids.iter().all(|pid| {
                    // SAFETY: signal zero only probes the recorded pid. The
                    // still-open pidfd above prevents an alias from satisfying
                    // the exact-exit assertion.
                    (unsafe { libc::kill(*pid, 0) }) != 0
                })
            }),
            "startup supervisor did not reap every recorded descendant: {pids:?}",
        );
    }

    #[derive(Default)]
    struct ObservedState {
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        heartbeats: usize,
        finished: Vec<Option<i32>>,
        failures: Vec<String>,
    }

    struct RecordingObserver {
        state: Arc<Mutex<ObservedState>>,
        interval: Duration,
        next_heartbeat: Instant,
    }

    impl RecordingObserver {
        fn new(interval: Duration) -> (Self, Arc<Mutex<ObservedState>>) {
            let state = Arc::new(Mutex::new(ObservedState::default()));
            (
                Self {
                    state: Arc::clone(&state),
                    interval,
                    next_heartbeat: Instant::now() + interval,
                },
                state,
            )
        }
    }

    impl StdoutObserver for RecordingObserver {
        fn observe_stdout(&mut self, bytes: &[u8]) {
            self.state
                .lock()
                .expect("observer state")
                .stdout
                .extend_from_slice(bytes);
        }

        fn observe_stderr(&mut self, bytes: &[u8]) {
            self.state
                .lock()
                .expect("observer state")
                .stderr
                .extend_from_slice(bytes);
        }

        fn tick(&mut self) {
            let now = Instant::now();
            if now >= self.next_heartbeat {
                self.state.lock().expect("observer state").heartbeats += 1;
                self.next_heartbeat = now + self.interval;
            }
        }

        fn next_tick_in(&self) -> Duration {
            self.next_heartbeat
                .saturating_duration_since(Instant::now())
        }

        fn finished(&mut self, status: &ExitStatus) {
            self.state
                .lock()
                .expect("observer state")
                .finished
                .push(status.code());
        }

        fn failed(&mut self, error: &io::Error) {
            self.state
                .lock()
                .expect("observer state")
                .failures
                .push(error.to_string());
        }
    }

    struct PanicOnOutputObserver;

    impl StdoutObserver for PanicOnOutputObserver {
        fn observe_stdout(&mut self, _bytes: &[u8]) {
            panic!("intentional stdout observer panic");
        }

        fn tick(&mut self) {}

        fn next_tick_in(&self) -> Duration {
            Duration::from_millis(20)
        }

        fn finished(&mut self, _status: &ExitStatus) {}

        fn failed(&mut self, _error: &io::Error) {}
    }

    struct PanicOnStderrObserver;

    impl StdoutObserver for PanicOnStderrObserver {
        fn observe_stdout(&mut self, _bytes: &[u8]) {}

        fn observe_stderr(&mut self, _bytes: &[u8]) {
            panic!("intentional stderr observer panic");
        }

        fn tick(&mut self) {}

        fn next_tick_in(&self) -> Duration {
            Duration::from_millis(20)
        }

        fn finished(&mut self, _status: &ExitStatus) {}

        fn failed(&mut self, _error: &io::Error) {}
    }

    fn current(sig: libc::c_int) -> usize {
        // SAFETY: querying with a null action and a valid out-pointer.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            libc::sigaction(sig, std::ptr::null(), &mut action);
            action.sa_sigaction
        }
    }

    fn watchdog() -> (mpsc::Sender<()>, JoinHandle<()>) {
        let (cancel, receiver) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            if receiver.recv_timeout(Duration::from_secs(5)).is_err() {
                let pgid = active_group_for_test();
                if pgid > 0 {
                    signal_group(pgid, libc::SIGKILL);
                }
            }
        });
        (cancel, handle)
    }

    fn cancel_watchdog(cancel: mpsc::Sender<()>, handle: JoinHandle<()>) {
        let _ = cancel.send(());
        handle.join().expect("watchdog joins");
    }

    fn wait_until(timeout: Duration, predicate: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if predicate() {
                return true;
            }
            std::thread::yield_now();
        }
        predicate()
    }

    fn install_cleanup_guard() -> InterruptGuard {
        let guard = InterruptGuard::install();
        assert!(
            !ktstr::fetch::git_operation_interrupted(),
            "a new interrupt guard resets the gix cancellation epoch",
        );
        enter_cleanup_phase().expect("test enters cleanup phase");
        guard
    }

    #[test]
    fn startup_owner_sigkill_drains_all_descendant_shapes() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        if !std::path::Path::new("/usr/bin/setsid").exists() {
            return;
        }
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("owner-kill.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner = spawn_startup_test_owner(&test_name, "descendants", &pidfile);
        let owner_pidfd = test_pidfd_open(owner.id() as libc::pid_t);
        let pids = startup_test_pids(&pidfile, 5);
        let pidfds: Vec<_> = pids.iter().copied().map(test_pidfd_open).collect();

        test_pidfd_kill(&owner_pidfd);
        let owner_status = owner.wait().expect("reap SIGKILLed startup owner");
        assert_eq!(owner_status.signal(), Some(libc::SIGKILL));
        assert_startup_pids_exit(&pids, &pidfds);
    }

    #[test]
    fn startup_worker_sigkill_drains_all_descendant_shapes() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        if !std::path::Path::new("/usr/bin/setsid").exists() {
            return;
        }
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("worker-kill.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner = spawn_startup_test_owner(&test_name, "descendants", &pidfile);
        let pids = startup_test_pids(&pidfile, 5);
        let pidfds: Vec<_> = pids.iter().copied().map(test_pidfd_open).collect();

        test_pidfd_kill(&pidfds[0]);
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                owner.try_wait().expect("query nested startup owner").is_some()
            }),
            "startup launcher did not relay the killed worker status",
        );
        let owner_status = owner
            .wait()
            .expect("read already-reaped nested owner status");
        assert_eq!(owner_status.signal(), Some(libc::SIGKILL));
        assert_startup_pids_exit(&pids, &pidfds);
    }

    #[test]
    fn startup_owner_death_is_observed_while_worker_is_stalled_before_ready() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("pre-ready-owner-kill.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner = spawn_startup_test_owner(&test_name, "pre-ready", &pidfile);
        let owner_pidfd = test_pidfd_open(owner.id() as libc::pid_t);
        let pids = startup_test_pids(&pidfile, 1);
        let pidfds: Vec<_> = pids.iter().copied().map(test_pidfd_open).collect();

        test_pidfd_kill(&owner_pidfd);
        let owner_status = owner.wait().expect("reap pre-ready startup owner");
        assert_eq!(owner_status.signal(), Some(libc::SIGKILL));
        assert_startup_pids_exit(&pids, &pidfds);
    }

    #[test]
    fn startup_pre_ready_signal_cancels_wedged_worker_promptly() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("pre-ready-signal.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner = spawn_startup_test_owner(&test_name, "pre-ready", &pidfile);
        let owner_pidfd = test_pidfd_open(owner.id() as libc::pid_t);
        let pids = startup_test_pids(&pidfile, 1);
        let pidfds: Vec<_> = pids.iter().copied().map(test_pidfd_open).collect();

        let signalled = Instant::now();
        test_pidfd_signal(&owner_pidfd, libc::SIGTERM);
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                owner
                    .try_wait()
                    .expect("query pre-ready signalled startup owner")
                    .is_some()
            }),
            "startup launcher waited out its readiness backstop after SIGTERM",
        );
        let owner_status = owner
            .wait()
            .expect("read pre-ready signalled startup owner status");
        assert_eq!(owner_status.signal(), Some(libc::SIGTERM));
        assert!(
            signalled.elapsed() < STARTUP_TEST_PROMPT_BACKSTOP,
            "pre-ready signal cancellation waited for its protocol backstop",
        );
        assert_startup_pids_exit(&pids, &pidfds);
    }

    #[test]
    fn startup_replays_pre_ready_signal_after_worker_barrier() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("delayed-ready.pids");
        let release = pidfile.with_extension("release");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner = spawn_startup_test_owner(&test_name, "delayed-ready", &pidfile);
        let owner_pidfd = test_pidfd_open(owner.id() as libc::pid_t);
        let pids = startup_test_pids(&pidfile, 1);
        let pidfds: Vec<_> = pids.iter().copied().map(test_pidfd_open).collect();

        test_pidfd_signal(&owner_pidfd, libc::SIGTERM);
        std::fs::write(release, b"release").expect("release worker readiness barrier");
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                owner.try_wait().expect("query signalled startup owner").is_some()
            }),
            "startup launcher did not replay its queued pre-ready signal",
        );
        let owner_status = owner.wait().expect("read signalled owner status");
        assert_eq!(owner_status.signal(), Some(libc::SIGTERM));
        assert_startup_pids_exit(&pids, &pidfds);
    }

    #[test]
    fn startup_clean_exit_does_not_wait_for_fork_only_fd_holder() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("clean-fork.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let started = Instant::now();
        let mut owner = spawn_startup_test_owner(&test_name, "clean-fork", &pidfile);
        let pids = startup_test_pids(&pidfile, 2);
        let detached_pidfd = test_pidfd_open(pids[1]);
        let status = wait_startup_test_owner(&mut owner, "clean fork-only fd holder");

        assert!(status.success(), "clean worker status is relayed exactly");
        assert!(
            started.elapsed() < STARTUP_TEST_PROMPT_BACKSTOP,
            "fork-only control-fd holder delayed clean commit",
        );
        assert!(
            !test_pidfd_exited(&detached_pidfd),
            "explicit clean commit must release a live detached child",
        );
        test_pidfd_kill(&detached_pidfd);
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                test_pidfd_exited(&detached_pidfd)
            }),
            "clean detached test child did not accept cleanup SIGKILL",
        );
    }

    #[test]
    fn startup_normalizes_inherited_sigchld_ignore() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("sigchld-ignore.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner =
            spawn_startup_test_owner(&test_name, "sigchld-ignore-clean", &pidfile);
        let pids = startup_test_pids(&pidfile, 2);
        let detached_pidfd = test_pidfd_open(pids[1]);
        let status = wait_startup_test_owner(&mut owner, "SIGCHLD-normalized owner");

        assert!(
            status.success(),
            "inherited SIGCHLD=SIG_IGN must not auto-reap S or W",
        );
        assert!(!test_pidfd_exited(&detached_pidfd));
        test_pidfd_kill(&detached_pidfd);
        assert!(wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
            test_pidfd_exited(&detached_pidfd)
        }));
    }

    #[test]
    fn startup_rejects_worker_process_group_mismatch_before_ready() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("pgrp-mismatch.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner = spawn_startup_test_owner(&test_name, "pgrp-mismatch", &pidfile);
        let pids = startup_test_pids(&pidfile, 1);
        let worker_pidfd = test_pidfd_open(pids[0]);
        let status = wait_startup_test_owner(&mut owner, "pgrp-mismatch owner");

        assert_eq!(
            status.code(),
            Some(CHILD_OWNERSHIP_FAIL_CLOSED_EXIT_CODE),
        );
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                test_pidfd_exited(&worker_pidfd)
            }),
            "pgrp-mismatched worker survived fail-closed startup",
        );
    }

    #[test]
    fn startup_owner_death_before_ack_dirty_drains_detached_child() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("owner-death-pre-ack.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner =
            spawn_startup_test_owner(&test_name, "owner-death-pre-ack", &pidfile);
        let owner_pidfd = test_pidfd_open(owner.id() as libc::pid_t);
        let checkpoint_pid = startup_test_checkpoint_pid(&pidfile, "pre-ack");
        assert_eq!(checkpoint_pid, owner.id() as libc::pid_t);
        let pids = startup_test_pids(&pidfile, 2);
        let detached_pidfd = test_pidfd_open(pids[1]);

        test_pidfd_kill(&owner_pidfd);
        let owner_status = owner.wait().expect("reap pre-ACK killed owner");
        assert_eq!(owner_status.signal(), Some(libc::SIGKILL));
        assert_startup_pids_exit(&[pids[1]], std::slice::from_ref(&detached_pidfd));
    }

    #[test]
    fn startup_owner_death_after_consumed_ack_preserves_clean_detached_child() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("owner-death-post-ack.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner =
            spawn_startup_test_owner(&test_name, "owner-death-post-ack", &pidfile);
        let owner_pidfd = test_pidfd_open(owner.id() as libc::pid_t);
        let supervisor_pid = startup_test_checkpoint_pid(&pidfile, "post-ack");
        let supervisor_pidfd = test_pidfd_open(supervisor_pid);
        let pids = startup_test_pids(&pidfile, 2);
        let detached_pidfd = test_pidfd_open(pids[1]);

        test_pidfd_kill(&owner_pidfd);
        let owner_status = owner.wait().expect("reap post-ACK killed owner");
        assert_eq!(owner_status.signal(), Some(libc::SIGKILL));
        release_startup_test_checkpoint(&pidfile, "post-ack");
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                test_pidfd_exited(&supervisor_pidfd)
            }),
            "released startup supervisor did not exit",
        );
        assert!(
            !test_pidfd_exited(&detached_pidfd),
            "consumed ACK must commit clean detached-child survival",
        );
        test_pidfd_kill(&detached_pidfd);
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                test_pidfd_exited(&detached_pidfd)
            }),
            "post-ACK detached child did not accept cleanup SIGKILL",
        );
    }

    #[test]
    fn startup_clean_nonzero_status_is_exact_and_releases_detached_child() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("clean-nonzero.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner = spawn_startup_test_owner(&test_name, "clean-nonzero", &pidfile);
        let pids = startup_test_pids(&pidfile, 2);
        let detached_pidfd = test_pidfd_open(pids[1]);
        let status = wait_startup_test_owner(&mut owner, "clean nonzero");

        assert_eq!(status.code(), Some(23));
        assert_eq!(status.signal(), None);
        assert!(
            !test_pidfd_exited(&detached_pidfd),
            "matching nonzero clean status must release detached children",
        );
        test_pidfd_kill(&detached_pidfd);
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                test_pidfd_exited(&detached_pidfd)
            }),
            "clean nonzero detached child did not accept cleanup SIGKILL",
        );
    }

    fn assert_startup_protocol_fault_drains(
        mode: &str,
        phase: &str,
        pid_count: usize,
        relayed_signal: Option<libc::c_int>,
    ) {
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join(format!("{mode}.pids"));
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner = spawn_startup_test_owner(&test_name, mode, &pidfile);
        let pids = startup_test_pids(&pidfile, pid_count);
        let pidfds: Vec<_> = pids.iter().copied().map(test_pidfd_open).collect();
        let _checkpoint_pid = startup_test_checkpoint_pid(&pidfile, phase);
        release_startup_test_checkpoint(&pidfile, phase);

        let status = wait_startup_test_owner(&mut owner, mode);
        if let Some(signal) = relayed_signal {
            assert_eq!(status.signal(), Some(signal));
        } else {
            assert_startup_fail_closed(&status);
        }
        assert_startup_pids_exit(&pids, &pidfds);
    }

    #[test]
    fn startup_malformed_worker_control_fails_closed() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        assert_startup_protocol_fault_drains(
            "malformed-worker-control",
            "malformed-control",
            2,
            None,
        );
    }

    #[test]
    fn startup_truncated_worker_control_fails_closed() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        assert_startup_protocol_fault_drains(
            "truncated-worker-control",
            "truncated-control",
            2,
            Some(libc::SIGKILL),
        );
    }

    #[test]
    fn startup_malformed_report_fails_closed() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        if !std::path::Path::new("/usr/bin/setsid").exists() {
            return;
        }
        assert_startup_protocol_fault_drains(
            "malformed-report",
            "report-fault",
            5,
            None,
        );
    }

    #[test]
    fn startup_truncated_report_fails_closed() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        if !std::path::Path::new("/usr/bin/setsid").exists() {
            return;
        }
        assert_startup_protocol_fault_drains(
            "truncated-report",
            "report-fault",
            5,
            None,
        );
    }

    #[test]
    fn startup_supervisor_exec_failure_is_prompt() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("supervisor-exec-failure.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let started = Instant::now();
        let mut owner =
            spawn_startup_test_owner(&test_name, "supervisor-exec-failure", &pidfile);
        let status = wait_startup_test_owner(&mut owner, "supervisor exec failure");

        assert_startup_fail_closed(&status);
        assert!(
            started.elapsed() < STARTUP_TEST_PROMPT_BACKSTOP,
            "supervisor exec failure waited for a protocol backstop",
        );
    }

    #[test]
    fn startup_worker_exec_failure_is_prompt_and_reaped() {
        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup supervision tempdir");
        let pidfile = root.path().join("worker-exec-failure.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut owner =
            spawn_startup_test_owner(&test_name, "worker-exec-failure", &pidfile);
        let supervisor_pid = startup_test_checkpoint_pid(&pidfile, "worker-exec");
        let supervisor_pidfd = test_pidfd_open(supervisor_pid);
        let started = Instant::now();
        release_startup_test_checkpoint(&pidfile, "worker-exec");
        let status = wait_startup_test_owner(&mut owner, "worker exec failure");

        assert_startup_fail_closed(&status);
        assert!(
            started.elapsed() < STARTUP_TEST_PROMPT_BACKSTOP,
            "worker exec failure waited for a protocol backstop",
        );
        assert!(
            test_pidfd_exited(&supervisor_pidfd),
            "launcher did not reap the failed worker's supervisor",
        );
    }

    #[test]
    fn startup_command_builder_preserves_non_utf8_argv() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let command = startup_command_for_self_with_args(
            vec![
                std::ffi::OsString::from_vec(b"argv0-\xff".to_vec()),
                std::ffi::OsString::from_vec(b"argument-\xfe".to_vec()),
                std::ffi::OsString::from("--exact"),
            ]
            .into_iter(),
        )
        .expect("build exact startup re-exec command");
        let args: Vec<_> = command
            .get_args()
            .map(|argument| argument.as_bytes())
            .collect();

        assert_eq!(command.get_program().as_bytes(), b"/proc/self/exe");
        assert_eq!(args, vec![b"argument-\xfe".as_slice(), b"--exact".as_slice()]);
    }

    fn byte_occurrences(haystack: &[u8], needle: &[u8]) -> usize {
        haystack
            .windows(needle.len())
            .filter(|window| *window == needle)
            .count()
    }

    #[test]
    fn startup_reexec_preserves_arg0_env_cwd_and_stdio() {
        use std::os::unix::ffi::OsStringExt;

        if startup_test_role_if_requested() {
            return;
        }
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("startup context tempdir");
        let cwd = root
            .path()
            .join(std::ffi::OsString::from_vec(b"cwd-\xfd".to_vec()));
        std::fs::create_dir(&cwd).expect("create non-UTF8 startup cwd");
        let pidfile = root.path().join("context-fidelity.pids");
        let test_name = std::thread::current()
            .name()
            .expect("libtest supplies the current test name")
            .to_owned();
        let mut command = Command::new(std::env::current_exe().expect("current test executable"));
        command
            .arg0("ktstr-context-argv0")
            .arg("--exact")
            .arg(&test_name)
            .arg("--nocapture")
            .env(STARTUP_TEST_OWNER_ENV, "1")
            .env(STARTUP_TEST_MODE_ENV, "context-fidelity")
            .env(STARTUP_TEST_PIDFILE_ENV, &pidfile)
            .env(
                std::ffi::OsString::from_vec(b"KTSTR_CONTEXT_\xff".to_vec()),
                std::ffi::OsString::from_vec(b"value_\xfe".to_vec()),
            )
            .current_dir(&cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut owner = command.spawn().expect("spawn startup context owner");
        owner
            .stdin
            .take()
            .expect("startup context stdin pipe")
            .write_all(&[0xa5])
            .expect("write startup context stdin");
        assert!(
            wait_until(STARTUP_TEST_OBSERVATION_BACKSTOP, || {
                owner.try_wait().expect("query startup context owner").is_some()
            }),
            "startup context owner did not exit",
        );
        let output = owner
            .wait_with_output()
            .expect("collect startup context output");

        assert!(output.status.success(), "startup context re-exec succeeded");
        assert_eq!(
            byte_occurrences(&output.stdout, b"\0KTSTR-STDOUT-\xff\0"),
            1,
        );
        assert_eq!(
            byte_occurrences(&output.stderr, b"\0KTSTR-STDERR-\xfe\0"),
            1,
        );
    }

    #[test]
    fn install_records_first_signal_and_restores_dispositions() {
        let _serial = test_serial_guard();
        // SAFETY: SIG_IGN is a valid sentinel disposition.
        let (mut pre_int, mut pre_term) = unsafe {
            (
                std::mem::zeroed::<libc::sigaction>(),
                std::mem::zeroed::<libc::sigaction>(),
            )
        };
        unsafe {
            let mut ignore: libc::sigaction = std::mem::zeroed();
            ignore.sa_sigaction = libc::SIG_IGN;
            libc::sigemptyset(&mut ignore.sa_mask);
            libc::sigaction(libc::SIGINT, &ignore, &mut pre_int);
            libc::sigaction(libc::SIGTERM, &ignore, &mut pre_term);
        }

        let guard = InterruptGuard::install();
        let wanted = handler as *const () as usize;
        assert_eq!(current(libc::SIGINT), wanted);
        assert_eq!(current(libc::SIGTERM), wanted);
        record_for_test(libc::SIGTERM);
        record_for_test(libc::SIGINT);
        assert_eq!(guard.interrupted(), Some(libc::SIGTERM));
        assert!(INTERRUPTED.load(Ordering::SeqCst));
        assert!(
            ktstr::fetch::git_operation_interrupted(),
            "recording a process signal cancels in-process gix work",
        );
        assert_eq!(restore_and_caught(guard), Some(libc::SIGTERM));
        assert_eq!(current(libc::SIGINT), libc::SIG_IGN);
        assert_eq!(current(libc::SIGTERM), libc::SIG_IGN);

        // SAFETY: hand the process back to the test runner's dispositions.
        unsafe {
            libc::sigaction(libc::SIGINT, &pre_int, std::ptr::null_mut());
            libc::sigaction(libc::SIGTERM, &pre_term, std::ptr::null_mut());
        }
    }

    #[test]
    fn anchor_signal_setup_propagates_partial_install_and_mask_failures() {
        let installed = std::cell::RefCell::new(Vec::new());
        let unblocked = std::cell::Cell::new(false);
        let error = initialize_anchor_signal_state_with(
            |signal| {
                installed.borrow_mut().push(signal);
                if signal == libc::SIGTERM {
                    Err(io::Error::other("injected SIGTERM install failure"))
                } else {
                    Ok(())
                }
            },
            || {
                unblocked.set(true);
                Ok(())
            },
        )
        .expect_err("a partial anchor disposition install must fail");
        assert!(error.to_string().contains("SIGTERM install failure"));
        assert_eq!(*installed.borrow(), [libc::SIGINT, libc::SIGTERM]);
        assert!(
            !unblocked.get(),
            "anchor signals cannot be unblocked after a partial disposition install",
        );

        let error = initialize_anchor_signal_state_with(
            |_| Ok(()),
            || Err(io::Error::other("injected mask failure")),
        )
        .expect_err("an anchor mask failure must prevent readiness");
        assert!(error.to_string().contains("mask failure"));
    }

    #[test]
    fn signal_pair_install_restores_first_disposition_if_second_fails() {
        let _serial = test_serial_guard();
        // SAFETY: all sigaction pointers name initialized storage. SIGUSR1 is
        // restored before this test returns; -1 deterministically makes the
        // second install fail with EINVAL.
        unsafe {
            let mut before: libc::sigaction = std::mem::zeroed();
            assert_eq!(
                libc::sigaction(libc::SIGUSR1, std::ptr::null(), &mut before),
                0,
            );
            let mut ignore: libc::sigaction = std::mem::zeroed();
            ignore.sa_sigaction = libc::SIG_IGN;
            libc::sigemptyset(&mut ignore.sa_mask);

            let error = match install_signal_pair(libc::SIGUSR1, -1, &ignore) {
                Err(error) => error,
                Ok(_) => panic!("invalid second signal must fail"),
            };
            assert_eq!(error.raw_os_error(), Some(libc::EINVAL));

            let mut after: libc::sigaction = std::mem::zeroed();
            assert_eq!(
                libc::sigaction(libc::SIGUSR1, std::ptr::null(), &mut after),
                0,
            );
            assert_eq!(
                after.sa_sigaction, before.sa_sigaction,
                "the first disposition is restored transactionally",
            );
            assert_eq!(
                libc::sigaction(libc::SIGUSR1, &before, std::ptr::null_mut()),
                0,
            );
        }
    }

    #[test]
    fn restore_waits_for_handler_to_publish_first_signal() {
        let _serial = test_serial_guard();
        let guard = install_cleanup_guard();
        let installed = handler as *const () as usize;
        TEST_PAUSE_HANDLER_BEFORE_RECORD.store(true, Ordering::SeqCst);

        let sender = std::thread::spawn(|| {
            // Target this peer thread so the test thread remains available to
            // drive the deterministic restore/read race.
            // SAFETY: pthread_self identifies this live thread.
            unsafe { libc::pthread_kill(libc::pthread_self(), libc::SIGTERM) }
        });
        let entered = wait_until(Duration::from_secs(3), || {
            TEST_HANDLER_BEFORE_RECORD.load(Ordering::SeqCst)
        });

        let (result_tx, result_rx) = mpsc::channel();
        let restorer = std::thread::spawn(move || {
            let _ = result_tx.send(restore_and_caught(guard));
        });
        let dispositions_restored = wait_until(Duration::from_secs(3), || {
            current(libc::SIGINT) != installed && current(libc::SIGTERM) != installed
        });
        // Once both saved dispositions are visible, Drop must still be parked
        // on the handler that has incremented HANDLERS_IN_FLIGHT but has not
        // yet published CAUGHT_SIGNAL.
        let completed_while_paused = if dispositions_restored {
            result_rx.recv_timeout(Duration::from_millis(100)).ok()
        } else {
            None
        };

        // Always release global handler state before asserting.
        TEST_RELEASE_HANDLER_BEFORE_RECORD.store(true, Ordering::SeqCst);
        let sender_rc = sender.join().expect("signal sender joins");
        let restored_result = completed_while_paused.unwrap_or_else(|| {
            result_rx
                .recv_timeout(Duration::from_secs(3))
                .expect("restore completes after delayed handler publishes")
        });
        restorer.join().expect("restorer joins");
        TEST_PAUSE_HANDLER_BEFORE_RECORD.store(false, Ordering::SeqCst);

        assert_eq!(sender_rc, 0, "peer-thread pthread_kill succeeds");
        assert!(entered, "handler reached the deterministic pre-record seam");
        assert!(
            dispositions_restored,
            "guard restored both saved dispositions before draining",
        );
        assert!(
            completed_while_paused.is_none(),
            "restore_and_caught returned before an entered handler published",
        );
        assert_eq!(restored_result, Some(libc::SIGTERM));
    }

    #[test]
    fn deferred_exit_is_one_shot_and_waits_for_top_level_consumption() {
        let _serial = test_serial_guard();
        defer_exit_code(23);
        defer_exit_code(23);
        assert_eq!(take_deferred_exit_code(), Some(23));
        assert_eq!(take_deferred_exit_code(), None);
    }

    #[test]
    fn pre_caught_signal_prevents_spawn() {
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("tempdir");
        let marker = root.path().join("spawned");
        let guard = install_cleanup_guard();
        // SAFETY: the installed handler returns normally.
        assert_eq!(unsafe { libc::raise(libc::SIGINT) }, 0);

        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg("printf spawned > \"$KTSTR_SIGNAL_MARKER\"")
            .env("KTSTR_SIGNAL_MARKER", &marker);
        let error = run_status(command).expect_err("pre-caught signal must suppress spawn");
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert!(!marker.exists(), "suppressed child left no marker");
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn cleanup_transition_exposes_an_already_recorded_interrupt() {
        let _serial = test_serial_guard();
        let guard = InterruptGuard::install();
        // Stand in for a handler that loaded TRANSITION and recorded before
        // enter_cleanup_phase's post-drain check.
        record_for_test(libc::SIGINT);
        let error = enter_cleanup_phase().expect_err("transition must expose the caught interrupt");
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert!(
            cleanup_phase_active(),
            "cleanup ownership is still established"
        );
        drop(guard);
    }

    #[test]
    fn signal_during_spawn_handoff_reaches_real_child() {
        let _serial = test_serial_guard();
        let guard = install_cleanup_guard();
        let (cancel, watchdog) = watchdog();
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg("while :; do :; done");
        let mut raise_rc = None;
        let status = run_status_with_handoff(command, || {
            // SAFETY: the installed handler returns and records the handoff
            // signal while ACTIVE_CHILD_GROUP is still SPAWNING.
            raise_rc = Some(unsafe { libc::raise(libc::SIGTERM) });
        })
        .expect("handoff child is reaped");
        cancel_watchdog(cancel, watchdog);

        assert_eq!(raise_rc, Some(0), "handoff signal delivery succeeds");
        assert_eq!(caught(), Some(libc::SIGTERM));
        assert_eq!(
            status.signal(),
            Some(libc::SIGTERM),
            "real child, not only the signal-ignoring anchor, receives replay",
        );
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn signal_on_peer_thread_reaches_child_descendant_group() {
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("tempdir");
        let pidfile = root.path().join("descendant.pid");
        let guard = install_cleanup_guard();
        let signal_pidfile = pidfile.clone();
        let sender = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            while !signal_pidfile.exists() && Instant::now() < deadline {
                std::thread::yield_now();
            }
            assert!(signal_pidfile.exists(), "descendant published its pid");
            // Deliver specifically on this peer thread. The global handoff
            // state, not a spawning-thread signal mask, must find the child.
            // SAFETY: pthread_self identifies this live thread.
            let rc = unsafe { libc::pthread_kill(libc::pthread_self(), libc::SIGTERM) };
            assert_eq!(rc, 0);
        });
        let (cancel, watchdog) = watchdog();

        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "child=; trap 'wait \"$child\"; exit 0' TERM; \
                 sleep 300 & child=$!; printf '%s' \"$child\" > \"$KTSTR_DESC_PID\"; \
                 wait \"$child\"",
            )
            .env("KTSTR_DESC_PID", &pidfile);
        let _ = run_status(command).expect("group-signalled shell is reaped");
        sender.join().expect("signal sender joins");
        cancel_watchdog(cancel, watchdog);

        let descendant: libc::pid_t = std::fs::read_to_string(&pidfile)
            .expect("read descendant pid")
            .parse()
            .expect("parse descendant pid");
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            // SAFETY: signal 0 only probes whether this pid remains.
            let alive = unsafe { libc::kill(descendant, 0) } == 0;
            if !alive {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "descendant {descendant} survived process-group forwarding",
            );
            std::thread::yield_now();
        }
        assert_eq!(caught(), Some(libc::SIGTERM));
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn runner_waits_for_same_group_descendant_after_leader_exit() {
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("tempdir");
        let descendant_ready = root.path().join("descendant-ready");
        let leader_exited = root.path().join("leader-exited");
        let release = root.path().join("release");
        let descendant_done = root.path().join("descendant-done");
        let guard = install_cleanup_guard();

        let ready_env = descendant_ready.clone();
        let leader_env = leader_exited.clone();
        let release_env = release.clone();
        let done_env = descendant_done.clone();
        let (result_tx, result_rx) = mpsc::channel();
        let runner = std::thread::spawn(move || {
            let mut command = Command::new("/bin/sh");
            command
                .arg("-c")
                .arg(
                    "( printf ready > \"$KTSTR_DESC_READY\"; \
                       while [ ! -e \"$KTSTR_DESC_RELEASE\" ]; do :; done; \
                       printf done > \"$KTSTR_DESC_DONE\" ) & \
                     printf exited > \"$KTSTR_LEADER_EXITED\"; exit 0",
                )
                .env("KTSTR_DESC_READY", ready_env)
                .env("KTSTR_DESC_RELEASE", release_env)
                .env("KTSTR_DESC_DONE", done_env)
                .env("KTSTR_LEADER_EXITED", leader_env);
            let _ = result_tx.send(run_status(command));
        });

        assert!(
            wait_until(Duration::from_secs(3), || {
                descendant_ready.exists() && leader_exited.exists()
            }),
            "leader exited while its same-group descendant remained alive",
        );
        assert!(
            result_rx.try_recv().is_err(),
            "runner must not return with a live same-group descendant",
        );
        std::fs::write(&release, b"go").expect("release descendant");
        let status = result_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("runner returns after descendant exit")
            .expect("group wait succeeds");
        runner.join().expect("runner joins");

        assert!(status.success(), "leader status is preserved");
        assert!(
            descendant_done.exists(),
            "same-group descendant completed before the runner returned",
        );
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn runner_does_not_wait_for_a_helper_that_escapes_the_group() {
        let _serial = test_serial_guard();
        if !std::path::Path::new("/usr/bin/setsid").exists() {
            return;
        }
        let root = tempfile::tempdir().expect("tempdir");
        let escaped_ready = root.path().join("escaped-ready");
        let escaped_done = root.path().join("escaped-done");
        let guard = install_cleanup_guard();

        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "/usr/bin/setsid /bin/sh -c \
                   'printf ready > \"$KTSTR_ESCAPED_READY\"; \
                    sleep 1; printf done > \"$KTSTR_ESCAPED_DONE\"' \
                   </dev/null >/dev/null 2>&1 & \
                 while [ ! -e \"$KTSTR_ESCAPED_READY\" ]; do :; done; \
                 exit 0",
            )
            .env("KTSTR_ESCAPED_READY", &escaped_ready)
            .env("KTSTR_ESCAPED_DONE", &escaped_done);
        let started = Instant::now();
        let status = run_status(command).expect("leader and escaped helper spawn");
        let elapsed = started.elapsed();

        assert!(status.success(), "leader exits successfully");
        assert!(
            elapsed < Duration::from_millis(500),
            "escaped helper must not delay runner completion: {elapsed:?}",
        );
        assert!(
            wait_until(Duration::from_secs(3), || escaped_done.exists()),
            "escaped helper finishes independently",
        );
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn stopped_anchor_is_continued_and_reaped_without_hanging() {
        let _serial = test_serial_guard();
        let ps = if std::path::Path::new("/bin/ps").exists() {
            "/bin/ps"
        } else if std::path::Path::new("/usr/bin/ps").exists() {
            "/usr/bin/ps"
        } else {
            return;
        };
        let guard = install_cleanup_guard();
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "anchor=\"$($KTSTR_PS -o pgid= -p $$)\"; \
                 kill -STOP \"$anchor\"; exit 0",
            )
            .env("KTSTR_PS", ps);
        let started = Instant::now();
        let status = run_status(command).expect("stopped anchor is resumed and reaped");
        let elapsed = started.elapsed();

        assert!(status.success());
        assert!(
            elapsed < ANCHOR_REAP_BACKSTOP,
            "stopped anchor exceeded its bounded reap: {elapsed:?}",
        );
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn reap_waits_for_handler_paused_after_loading_pgid() {
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("tempdir");
        let exit_marker = root.path().join("exit-now");
        let guard = install_cleanup_guard();
        TEST_PAUSE_HANDLER_AFTER_LOAD.store(true, Ordering::SeqCst);

        let runner_exit = exit_marker.clone();
        let (result_tx, result_rx) = mpsc::channel();
        let runner = std::thread::spawn(move || {
            let mut command = Command::new("/bin/sh");
            command
                .arg("-c")
                .arg("while [ ! -e \"$KTSTR_EXIT_MARKER\" ]; do :; done; exit 0")
                .env("KTSTR_EXIT_MARKER", &runner_exit);
            let _ = result_tx.send(run_status(command));
        });

        let published = wait_until(Duration::from_secs(3), || active_group_for_test() > 0);
        let sender = std::thread::spawn(|| {
            // Deliver on this peer thread so the runner/reaper thread remains
            // free to reach REAPING while this handler is deliberately paused.
            // SAFETY: pthread_self identifies this live thread.
            unsafe { libc::pthread_kill(libc::pthread_self(), libc::SIGTERM) }
        });
        let loaded = wait_until(Duration::from_secs(3), || {
            TEST_HANDLER_LOADED.load(Ordering::SeqCst)
        });

        // Let the child exit independently while the handler is suspended
        // between its pgid load and kill. waitid(WNOWAIT) can now observe it,
        // but the runner must stop in REAPING until the forwarder drains.
        std::fs::write(&exit_marker, b"exit").expect("release child");
        let reached_reaping = wait_until(Duration::from_secs(3), || {
            active_group_for_test() == REAPING
        });
        let completed_while_paused = result_rx.try_recv().ok();

        // Always release the handler before asserting, so a failed invariant
        // cannot strand the process-global signal handler or child.
        TEST_RELEASE_HANDLER.store(true, Ordering::SeqCst);
        let sender_rc = sender.join().expect("signal sender joins");
        let result = result_rx.recv_timeout(Duration::from_secs(5));
        runner.join().expect("runner joins");
        TEST_PAUSE_HANDLER_AFTER_LOAD.store(false, Ordering::SeqCst);

        assert_eq!(sender_rc, 0, "peer-thread pthread_kill succeeds");
        assert!(published, "runner published a real child pgid");
        assert!(loaded, "handler reached the deterministic post-load seam");
        assert!(
            reached_reaping,
            "exited child entered REAPING while the old-pgid reader was paused",
        );
        assert!(
            completed_while_paused.is_none(),
            "real reap must not complete before the old-pgid handler drains",
        );
        assert!(
            result
                .expect("runner completes after handler release")
                .expect("runner wait succeeds")
                .success(),
            "child exited cleanly before the delayed group signal",
        );
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn child_boundary_restores_signal_disposition_and_mask() {
        let _serial = test_serial_guard();
        let guard = install_cleanup_guard();
        let worker = std::thread::spawn(|| {
            // The parent has cargo-ktstr's caught SIGTERM disposition. Block
            // SIGTERM only on this spawning thread, as can happen in a
            // multithreaded caller, and prove the runner temporarily unblocks
            // it for spawn without changing the caller's lasting mask.
            let (mut blocked, mut previous) = unsafe {
                (
                    std::mem::zeroed::<libc::sigset_t>(),
                    std::mem::zeroed::<libc::sigset_t>(),
                )
            };
            // SAFETY: both signal-set pointers are valid for pthread_sigmask.
            let block_rc = unsafe {
                libc::sigemptyset(&mut blocked);
                libc::sigaddset(&mut blocked, libc::SIGTERM);
                libc::pthread_sigmask(libc::SIG_BLOCK, &blocked, &mut previous)
            };
            assert_eq!(block_rc, 0, "worker blocks SIGTERM");

            let mut command = Command::new("/bin/sh");
            command.arg("-c").arg("kill -TERM $$; exit 99");
            let status = run_status(command).expect("self-signalling child runs");

            let mut after = unsafe { std::mem::zeroed::<libc::sigset_t>() };
            // SAFETY: a null set queries this thread's current mask.
            let query_rc =
                unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, std::ptr::null(), &mut after) };
            assert_eq!(query_rc, 0, "worker queries its restored mask");
            // SAFETY: `after` was initialized by pthread_sigmask.
            let remained_blocked = unsafe { libc::sigismember(&after, libc::SIGTERM) } == 1;
            // SAFETY: restore the worker's original mask before it exits.
            let restore_rc = unsafe {
                libc::pthread_sigmask(libc::SIG_SETMASK, &previous, std::ptr::null_mut())
            };
            assert_eq!(restore_rc, 0, "worker restores its original mask");
            (status, remained_blocked)
        });
        let (status, remained_blocked) = worker.join().expect("spawn worker joins");
        assert_eq!(
            status.signal(),
            Some(libc::SIGTERM),
            "exec resets the parent's caught disposition and the runner \
             unblocks SIGTERM in the spawned child",
        );
        assert!(
            remained_blocked,
            "runner restores the spawning thread's mask"
        );
        assert_eq!(caught(), None, "child signal never targets the parent");
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn spawn_error_and_reap_both_clear_active_state() {
        let _serial = test_serial_guard();
        let guard = install_cleanup_guard();
        let missing = Command::new("/definitely/not/a/ktstr-command");
        assert!(run_status(missing).is_err());
        assert_eq!(active_group_for_test(), IDLE, "spawn-error slot cleanup");

        let mut success = Command::new("/bin/sh");
        success.arg("-c").arg("exit 0");
        assert!(run_status(success).expect("second spawn works").success());
        assert_eq!(active_group_for_test(), IDLE, "post-reap slot cleanup");
        drop(guard);
    }

    #[test]
    fn first_signal_dominates_during_handoff() {
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("tempdir");
        let ready = root.path().join("ready");
        let term_seen = root.path().join("term-seen");
        let guard = install_cleanup_guard();
        let (cancel, watchdog) = watchdog();
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "trap ':' INT; \
                 trap 'printf term > \"$KTSTR_TERM_SEEN\"; exit 0' TERM; \
                 printf ready > \"$KTSTR_SIGNAL_READY\"; \
                 while :; do :; done",
            )
            .env("KTSTR_SIGNAL_READY", &ready)
            .env("KTSTR_TERM_SEEN", &term_seen);
        let mut child_was_ready = false;
        let mut raise_results = Vec::new();
        let _ = run_status_with_handoff(command, || {
            child_was_ready = wait_until(Duration::from_secs(3), || ready.exists());
            // SAFETY: both deliveries run the installed handler synchronously.
            raise_results.push(unsafe { libc::raise(libc::SIGINT) });
            raise_results.push(unsafe { libc::raise(libc::SIGINT) });
            raise_results.push(unsafe { libc::raise(libc::SIGTERM) });
        })
        .expect("signalled child is reaped");
        cancel_watchdog(cancel, watchdog);
        assert!(
            child_was_ready,
            "child installed both signal traps before handoff delivery",
        );
        assert_eq!(raise_results, [0, 0, 0], "all handoff deliveries succeed");
        assert_eq!(
            caught(),
            Some(libc::SIGINT),
            "first signal remains authoritative for the parent re-raise",
        );
        assert!(
            term_seen.exists(),
            "the distinct second signal is also replayed for child escalation",
        );
        assert_eq!(
            TEST_FORWARDED_SIGINT.load(Ordering::SeqCst),
            2,
            "both handler-observed Ctrl-C deliveries are forwarded",
        );
        assert_eq!(
            TEST_FORWARDED_SIGTERM.load(Ordering::SeqCst),
            1,
            "the distinct escalation signal is forwarded",
        );
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn captured_output_uses_the_same_group_runner() {
        let _serial = test_serial_guard();
        let guard = install_cleanup_guard();
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg("printf stdout-value; printf stderr-value >&2");
        let output = run_output(command).expect("capture succeeds");
        assert!(output.status.success());
        assert_eq!(output.stdout, b"stdout-value");
        assert_eq!(output.stderr, b"stderr-value");
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn capture_drains_chatty_stdout_and_stderr_without_pipe_deadlock() {
        let _serial = test_serial_guard();
        if !std::path::Path::new("/usr/bin/head").exists() {
            return;
        }
        let guard = install_cleanup_guard();
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg(
            "/usr/bin/head -c 262144 /dev/zero; \
             /usr/bin/head -c 262144 /dev/zero >&2",
        );
        let output = run_output(command).expect("dual-pipe capture succeeds");
        assert!(output.status.success());
        assert_eq!(output.stdout, vec![0_u8; 262_144]);
        assert_eq!(output.stderr, vec![0_u8; 262_144]);
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn observer_panic_drops_an_armed_group_and_reaps_its_descendant() {
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("tempdir");
        let pidfile = root.path().join("observer-child.pid");
        let guard = install_cleanup_guard();
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "printf '%s' \"$$\" > \"$KTSTR_PANIC_CHILD_PID\"; \
                 printf trigger; sleep 300",
            )
            .env("KTSTR_PANIC_CHILD_PID", &pidfile);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = run_stdout_observed(command, PanicOnOutputObserver);
        }));

        assert!(panic.is_err(), "observer panic propagates after cleanup");
        let pid: libc::pid_t = std::fs::read_to_string(&pidfile)
            .expect("read observer child pid")
            .parse()
            .expect("parse observer child pid");
        assert!(
            wait_until(Duration::from_secs(2), || {
                // SAFETY: signal zero only probes process existence.
                (unsafe { libc::kill(pid, 0) }) != 0
            }),
            "observer panic left its child alive",
        );
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn direct_observer_panic_reaps_its_armed_child() {
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("tempdir");
        let pidfile = root.path().join("direct-observer-child.pid");
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "printf '%s' \"$$\" > \"$KTSTR_DIRECT_PANIC_CHILD_PID\"; \
                 printf trigger; sleep 300",
            )
            .env("KTSTR_DIRECT_PANIC_CHILD_PID", &pidfile);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = run_stdout_observed(command, PanicOnOutputObserver);
        }));

        assert!(
            panic.is_err(),
            "direct observer panic propagates after cleanup"
        );
        let pid: libc::pid_t = std::fs::read_to_string(&pidfile)
            .expect("read direct observer child pid")
            .parse()
            .expect("parse direct observer child pid");
        assert!(
            wait_until(Duration::from_secs(2), || {
                // SAFETY: signal zero only probes process existence.
                (unsafe { libc::kill(pid, 0) }) != 0
            }),
            "direct observer panic detached its child",
        );
        assert_eq!(active_group_for_test(), IDLE);
    }

    #[test]
    fn direct_stderr_observer_panic_reaps_its_armed_child() {
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("tempdir");
        let pidfile = root.path().join("direct-stderr-observer-child.pid");
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "printf '%s' \"$$\" > \"$KTSTR_DIRECT_STDERR_PANIC_CHILD_PID\"; \
                 printf trigger >&2; sleep 300",
            )
            .env("KTSTR_DIRECT_STDERR_PANIC_CHILD_PID", &pidfile);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = run_output_observed(command, PanicOnStderrObserver);
        }));

        assert!(
            panic.is_err(),
            "direct stderr observer panic propagates after cleanup"
        );
        let pid: libc::pid_t = std::fs::read_to_string(&pidfile)
            .expect("read direct stderr observer child pid")
            .parse()
            .expect("parse direct stderr observer child pid");
        assert!(
            wait_until(Duration::from_secs(2), || {
                // SAFETY: signal zero only probes process existence.
                (unsafe { libc::kill(pid, 0) }) != 0
            }),
            "direct stderr observer panic detached its child",
        );
        assert_eq!(active_group_for_test(), IDLE);
    }

    #[test]
    fn proc_permission_filter_skips_only_unknown_foreign_owners() {
        let known = HashSet::from([41]);
        assert!(
            proc_owner_is_unrelated(42, &known, 2000, 1000),
            "an unknown foreign process cannot poison a same-euid group scan",
        );
        assert!(
            !proc_owner_is_unrelated(42, &known, 1000, 1000),
            "same-euid permission denial remains a hard scan error",
        );
        assert!(
            !proc_owner_is_unrelated(41, &known, 2000, 1000),
            "a known member cannot disappear by changing its effective uid",
        );
    }

    #[test]
    fn proc_exit_races_accept_only_enoent_and_esrch_as_gone() {
        assert!(process_gone(&io::Error::from_raw_os_error(libc::ENOENT)));
        assert!(process_gone(&io::Error::from_raw_os_error(libc::ESRCH)));
        assert!(
            !process_gone(&io::Error::from_raw_os_error(libc::EACCES)),
            "permission failures must remain visible",
        );
        assert!(
            !process_gone(&io::Error::from_raw_os_error(libc::EIO)),
            "procfs I/O failures must remain visible",
        );
    }

    #[test]
    fn startup_worker_status_is_immutable_after_first_exact_reap() {
        let mut status = None;
        record_startup_worker_status(41, 41, 0x0100, &mut status);
        assert_eq!(status, Some(0x0100));
        record_startup_worker_status(41, 41, 0x0900, &mut status);
        assert_eq!(
            status,
            Some(0x0100),
            "a later descendant reusing W's numeric pid cannot replace W's status",
        );
        record_startup_worker_status(41, 42, 0x0200, &mut status);
        assert_eq!(status, Some(0x0100));
    }

    #[test]
    fn proc_stat_parser_accepts_non_utf8_and_parentheses_in_comm() {
        let stat = b"123 (\xff) arbitrary ) bytes) R 1 77 1 0 0 0 0 0 0 0 5 7 0 0 0 0 0 0 99\n";
        let member = parse_process_stat(123, stat).expect("parse byte-oriented proc stat");
        assert_eq!(
            member.identity,
            ProcessIdentity {
                pid: 123,
                starttime_ticks: 99,
            },
        );
        assert_eq!(member.pgrp, 77);
        assert_eq!(member.cpu_ticks, 12);
        assert_eq!(member.state, b'R');
    }

    #[test]
    fn wedged_handler_counter_hits_the_drain_bound() {
        let _serial = test_serial_guard();
        HANDLERS_IN_FLIGHT.store(1, Ordering::SeqCst);
        let started = Instant::now();
        let drained = handlers_in_flight_drain_within(Duration::from_millis(20));
        let elapsed = started.elapsed();
        HANDLERS_IN_FLIGHT.store(0, Ordering::SeqCst);

        assert!(!drained, "a wedged handler counter cannot report success");
        assert!(
            elapsed >= Duration::from_millis(20),
            "the helper returned before its explicit deadline: {elapsed:?}",
        );
        assert!(
            handlers_in_flight_drain_within(Duration::from_millis(20)),
            "the helper completes immediately once ownership drains",
        );
    }

    #[test]
    fn handoff_panic_is_owned_before_pgid_publication() {
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("tempdir");
        let pidfile = root.path().join("handoff-child.pid");
        let guard = install_cleanup_guard();
        let child_pidfile = pidfile.clone();
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "printf '%s' \"$$\" > \"$KTSTR_HANDOFF_CHILD_PID\"; \
                 sleep 300",
            )
            .env("KTSTR_HANDOFF_CHILD_PID", &pidfile);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = run_status_with_handoff(command, || {
                assert!(
                    wait_until(Duration::from_secs(3), || child_pidfile.exists()),
                    "handoff child publishes pid before panic",
                );
                panic!("intentional handoff panic");
            });
        }));

        assert!(panic.is_err(), "handoff panic propagates after cleanup");
        let pid: libc::pid_t = std::fs::read_to_string(&pidfile)
            .expect("read handoff child pid")
            .parse()
            .expect("parse handoff child pid");
        assert!(
            wait_until(Duration::from_secs(2), || {
                // SAFETY: signal zero only probes process existence.
                (unsafe { libc::kill(pid, 0) }) != 0
            }),
            "handoff panic left its pre-publication child alive",
        );
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn capture_retains_output_from_a_normal_delayed_group_writer() {
        let _serial = test_serial_guard();
        let guard = install_cleanup_guard();
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg(
            "( sleep 0.12; printf late-stdout; printf late-stderr >&2 ) & \
             printf early-stdout; printf early-stderr >&2; exit 0",
        );
        let output = run_output(command).expect("delayed same-group writer is drained");
        assert!(output.status.success());
        assert_eq!(output.stdout, b"early-stdoutlate-stdout");
        assert_eq!(output.stderr, b"early-stderrlate-stderr");
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn escaped_pipe_holder_cannot_extend_capture_lifetime() {
        let _serial = test_serial_guard();
        if !std::path::Path::new("/usr/bin/setsid").exists() {
            return;
        }
        let root = tempfile::tempdir().expect("tempdir");
        let helper_pid = root.path().join("helper.pid");
        let helper_done = root.path().join("helper.done");
        let helper_release = root.path().join("helper.release");
        let guard = install_cleanup_guard();
        let watchdog_release = helper_release.clone();
        let (cancel_release, release_waiter) = mpsc::channel();
        let release_watchdog = std::thread::spawn(move || {
            if release_waiter.recv_timeout(Duration::from_secs(5)).is_err() {
                std::fs::write(watchdog_release, b"release")
                    .expect("watchdog releases escaped helper");
            }
        });
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "printf owned-stdout; printf owned-stderr >&2; \
                 /usr/bin/setsid /bin/sh -c \
                   'while [ ! -e \"$KTSTR_ESCAPED_RELEASE\" ]; do sleep 0.02; done; \
                    printf done > \"$KTSTR_ESCAPED_DONE\"' & \
                 printf '%s' \"$!\" > \"$KTSTR_ESCAPED_PID\"; exit 0",
            )
            .env("KTSTR_ESCAPED_PID", &helper_pid)
            .env("KTSTR_ESCAPED_DONE", &helper_done)
            .env("KTSTR_ESCAPED_RELEASE", &helper_release);
        let started = Instant::now();
        let output = run_output(command).expect("escaped pipe holder is detached");
        let elapsed = started.elapsed();

        assert!(output.status.success());
        assert_eq!(output.stdout, b"owned-stdout");
        assert_eq!(output.stderr, b"owned-stderr");
        assert!(
            elapsed < Duration::from_secs(3),
            "setsid pipe holder delayed capture by {elapsed:?}",
        );
        let pid: libc::pid_t = std::fs::read_to_string(&helper_pid)
            .expect("read escaped helper pid")
            .parse()
            .expect("parse escaped helper pid");
        // SAFETY: signal zero only probes process existence. The helper is
        // intentionally independent once it leaves the anchored group.
        assert_eq!(unsafe { libc::kill(pid, 0) }, 0);
        std::fs::write(&helper_release, b"release").expect("release escaped helper");
        let _ = cancel_release.send(());
        release_watchdog.join().expect("release watchdog joins");
        assert!(
            wait_until(Duration::from_secs(3), || helper_done.exists()),
            "escaped helper finishes independently after capture returns",
        );
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn cpu_burning_post_leader_group_is_killed_by_service_budget() {
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("tempdir");
        let child_pid = root.path().join("child.pid");
        let guard = install_cleanup_guard();
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "/bin/sh -c 'while :; do :; done' & child=$!; \
                 printf '%s' \"$child\" > \"$KTSTR_IMMORTAL_PID\"; exit 0",
            )
            .env("KTSTR_IMMORTAL_PID", &child_pid);
        let started = Instant::now();
        let error = run_output(command).expect_err("immortal group tail must fail");
        let elapsed = started.elapsed();

        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(
            error.to_string().contains("CPU-service budget")
                || error.to_string().contains("wall backstop"),
            "timeout identifies the exhausted bound: {error}",
        );
        assert!(
            elapsed < Duration::from_secs(7),
            "immortal group teardown remained bounded: {elapsed:?}",
        );
        let pid: libc::pid_t = std::fs::read_to_string(&child_pid)
            .expect("read child pid")
            .parse()
            .expect("parse child pid");
        assert!(
            wait_until(Duration::from_secs(2), || {
                // SAFETY: signal zero only probes process existence.
                (unsafe { libc::kill(pid, 0) }) != 0
            }),
            "service-budget teardown reaps the residual child",
        );
        assert_eq!(active_group_for_test(), IDLE);
        drop(guard);
    }

    #[test]
    fn observed_stdout_is_exact_and_heartbeats_while_the_child_is_silent() {
        let _serial = test_serial_guard();
        let guard = install_cleanup_guard();
        let (observer, state) = RecordingObserver::new(Duration::from_millis(15));
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg("printf 'first\\n'; sleep 0.08; printf 'last-without-newline'");

        let output = run_stdout_observed(command, observer).expect("observed command");
        assert!(output.status.success());
        assert_eq!(output.stdout, b"first\nlast-without-newline");
        assert!(output.stderr.is_empty(), "stderr remains inherited");

        let state = state.lock().expect("observer state");
        assert_eq!(
            state.stdout, output.stdout,
            "observer receives the same byte sequence returned to the JSON parser",
        );
        assert!(
            state.heartbeats >= 1,
            "an eighty-millisecond silent interval must trigger periodic ticks: {}",
            state.heartbeats,
        );
        assert_eq!(state.finished, [Some(0)]);
        assert!(state.failures.is_empty());
        assert_eq!(active_group_for_test(), IDLE);
        drop(state);
        drop(guard);
    }

    #[test]
    fn observed_dual_streams_are_exact_and_delivered_to_the_observer() {
        let _serial = test_serial_guard();
        let guard = install_cleanup_guard();
        let (observer, state) = RecordingObserver::new(Duration::from_millis(15));
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg(
            "printf stdout-first; printf stderr-first >&2; \
             sleep 0.04; printf stdout-last; printf stderr-last >&2",
        );

        let output = run_output_observed(command, observer).expect("dual-stream observed command");
        assert!(output.status.success());
        assert_eq!(output.stdout, b"stdout-firststdout-last");
        assert_eq!(output.stderr, b"stderr-firststderr-last");

        let state = state.lock().expect("observer state");
        assert_eq!(state.stdout, output.stdout);
        assert_eq!(state.stderr, output.stderr);
        assert_eq!(state.finished, [Some(0)]);
        assert!(state.failures.is_empty());
        assert_eq!(active_group_for_test(), IDLE);
        drop(state);
        drop(guard);
    }

    #[test]
    fn observed_nonzero_and_spawn_failure_both_finish_ownership() {
        let _serial = test_serial_guard();
        let guard = install_cleanup_guard();

        let (nonzero_observer, nonzero_state) = RecordingObserver::new(Duration::from_millis(20));
        let mut nonzero = Command::new("/bin/sh");
        nonzero.arg("-c").arg("printf cargo-json; exit 7");
        let output =
            run_stdout_observed(nonzero, nonzero_observer).expect("nonzero output is returned");
        assert_eq!(output.status.code(), Some(7));
        assert_eq!(output.stdout, b"cargo-json");
        assert_eq!(
            nonzero_state.lock().expect("nonzero state").finished,
            [Some(7)],
        );
        assert_eq!(active_group_for_test(), IDLE);

        let (missing_observer, missing_state) = RecordingObserver::new(Duration::from_millis(20));
        let missing = Command::new("/definitely/not/a/ktstr-observed-command");
        let error =
            run_stdout_observed(missing, missing_observer).expect_err("missing command fails");
        let state = missing_state.lock().expect("missing state");
        assert!(state.finished.is_empty());
        assert_eq!(state.failures, [error.to_string()]);
        assert_eq!(active_group_for_test(), IDLE);
        drop(state);
        drop(guard);
    }

    #[test]
    fn observed_signal_reaps_the_entire_group_without_stranding_progress() {
        let _serial = test_serial_guard();
        let root = tempfile::tempdir().expect("tempdir");
        let pidfile = root.path().join("descendant.pid");
        let guard = install_cleanup_guard();
        let signal_pidfile = pidfile.clone();
        let sender = std::thread::spawn(move || {
            assert!(
                wait_until(Duration::from_secs(3), || signal_pidfile.exists()),
                "child published descendant pid",
            );
            // SAFETY: pthread_self identifies this live peer thread and the
            // installed handler forwards SIGTERM to the published group.
            unsafe { libc::pthread_kill(libc::pthread_self(), libc::SIGTERM) }
        });
        let (cancel, watchdog) = watchdog();
        let (observer, state) = RecordingObserver::new(Duration::from_millis(20));
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "child=; \
                 trap 'kill -TERM \"$child\" 2>/dev/null; wait \"$child\"; exit 143' TERM; \
                 sleep 300 & child=$!; \
                 printf '%s' \"$child\" > \"$KTSTR_OBSERVED_DESC_PID\"; \
                 wait \"$child\"",
            )
            .env("KTSTR_OBSERVED_DESC_PID", &pidfile);

        let started = Instant::now();
        let output = run_stdout_observed(command, observer).expect("signalled group is reaped");
        let elapsed = started.elapsed();
        let sender_rc = sender.join().expect("signal sender joins");
        cancel_watchdog(cancel, watchdog);

        assert_eq!(sender_rc, 0, "peer-thread signal delivery succeeds");
        assert!(
            !output.status.success(),
            "SIGTERM is visible at command exit"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "signal cleanup remains bounded: {elapsed:?}",
        );
        let descendant: libc::pid_t = std::fs::read_to_string(&pidfile)
            .expect("read descendant pid")
            .parse()
            .expect("parse descendant pid");
        assert!(
            wait_until(Duration::from_secs(3), || {
                // SAFETY: signal zero only probes process existence.
                (unsafe { libc::kill(descendant, 0) }) != 0
            }),
            "observed runner left no same-group descendant",
        );
        let state = state.lock().expect("observer state");
        assert_eq!(state.finished.len(), 1);
        assert!(state.failures.is_empty());
        assert_eq!(caught(), Some(libc::SIGTERM));
        assert_eq!(active_group_for_test(), IDLE);
        drop(state);
        drop(guard);
    }
}
