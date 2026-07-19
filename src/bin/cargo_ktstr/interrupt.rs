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
//! Cleanup-phase children run in a group pinned by a tiny anchor process.
//! Spawning uses [`CommandExt::process_group`] and temporarily unblocks
//! SIGINT/SIGTERM in the calling thread, retaining the standard library's
//! `posix_spawn` path. No `pre_exec` closure forces unsafe fork+exec in this
//! multithreaded CLI. The anchor ignores terminal signals and lives until its
//! control pipe closes, so the pgid remains valid after a wrapper `cargo`
//! leader exits while same-group descendants are still tearing down.
//!
//! The active handoff is `IDLE -> SPAWNING -> anchor-pgid -> REAPING -> IDLE`.
//! Signals handled during SPAWNING are counted by kind and replayed after
//! publication (including repeated Ctrl-C). At teardown, REAPING prevents new
//! readers, in-flight handlers drain while the anchor still pins its pgid, and
//! only then is the anchor released/reaped. A handler therefore cannot resume
//! a stale `kill(-pgid)` after reuse.

use std::collections::{HashMap, HashSet};
use std::io::{self, Read};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::MetadataExt;
use std::os::unix::process::CommandExt;
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

fn read_process_stat(pid: libc::pid_t) -> io::Result<Option<GroupMember>> {
    let stat = match std::fs::read(format!("/proc/{pid}/stat")) {
        Ok(stat) => stat,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
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
                unsafe { libc::kill(pid, 0) } != 0
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
                unsafe { libc::kill(pid, 0) } != 0
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
                unsafe { libc::kill(pid, 0) } != 0
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
                unsafe { libc::kill(pid, 0) } != 0
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
                unsafe { libc::kill(pid, 0) } != 0
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
                unsafe { libc::kill(descendant, 0) } != 0
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
