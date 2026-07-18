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

use std::io::{self, Read};
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdin, Command, ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicUsize, Ordering};
use std::thread::JoinHandle;

const IDLE: libc::pid_t = 0;
const SPAWNING: libc::pid_t = -1;
const REAPING: libc::pid_t = -2;

const EARLY: u8 = 0;
const TRANSITION: u8 = 1;
const CLEANUP: u8 = 2;

const ANCHOR_MODE_ENV: &str = "__KTSTR_PROCESS_GROUP_ANCHOR";

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
        let was_active = GUARD_ACTIVE.swap(true, Ordering::SeqCst);
        assert!(!was_active, "only one InterruptGuard may be active",);
        debug_assert_eq!(
            ACTIVE_CHILD_GROUP.load(Ordering::SeqCst),
            IDLE,
            "cannot install InterruptGuard while a child runner is active",
        );
        CAUGHT_SIGNAL.store(0, Ordering::SeqCst);
        DEFERRED_EXIT_CODE.store(0, Ordering::SeqCst);
        INTERRUPTED.store(false, Ordering::SeqCst);
        ktstr::fetch::set_git_operation_interrupted(false);
        SIGNAL_PHASE.store(EARLY, Ordering::SeqCst);

        // SAFETY: `handler` has the signal-handler ABI and performs only
        // async-signal-safe operations. The out-pointers are valid.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = handler as *const () as usize;
            libc::sigemptyset(&mut action.sa_mask);
            action.sa_flags = 0;

            let mut prev_sigint: libc::sigaction = std::mem::zeroed();
            let mut prev_sigterm: libc::sigaction = std::mem::zeroed();
            let int_rc = libc::sigaction(libc::SIGINT, &action, &mut prev_sigint);
            let term_rc = libc::sigaction(libc::SIGTERM, &action, &mut prev_sigterm);
            debug_assert_eq!(int_rc, 0, "install SIGINT handler");
            debug_assert_eq!(term_rc, 0, "install SIGTERM handler");
            Self {
                prev_sigint,
                prev_sigterm,
            }
        }
    }

    /// The first signal caught during this guard's lifetime.
    #[cfg(test)]
    pub(crate) fn interrupted(&self) -> Option<libc::c_int> {
        caught()
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
            while HANDLERS_IN_FLIGHT.load(Ordering::SeqCst) != 0 {
                std::hint::spin_loop();
            }
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
        // No cleanup-owned waiter remains when the top-level guard drops.
        // Hide and join the broker while our handler is still installed, so a
        // late handler can observe only a closed broker epoch.
        ktstr::flock::stop_interruptible_flock_broker();
        debug_assert_eq!(
            ACTIVE_CHILD_GROUP.load(Ordering::SeqCst),
            IDLE,
            "InterruptGuard dropped while a child group is still published",
        );

        // SAFETY: these are the dispositions returned by `sigaction` during
        // installation and both pointers remain valid for each call.
        unsafe {
            libc::sigaction(libc::SIGINT, &self.prev_sigint, std::ptr::null_mut());
            libc::sigaction(libc::SIGTERM, &self.prev_sigterm, std::ptr::null_mut());
        }

        // Restoring both dispositions prevents new entries into `handler`.
        // Drain an entry that began just before restoration only afterwards:
        // it may not have published CAUGHT_SIGNAL yet. This closes the final
        // restore/read race for `restore_and_caught`.
        while HANDLERS_IN_FLIGHT.load(Ordering::SeqCst) != 0 {
            std::hint::spin_loop();
        }
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

    // SAFETY: the anchor is a newly exec'd, single-threaded process. Ignoring
    // and unblocking these signals before the ready byte makes the group pin
    // stable before its parent publishes the pgid.
    unsafe {
        let mut ignore: libc::sigaction = std::mem::zeroed();
        ignore.sa_sigaction = libc::SIG_IGN;
        libc::sigemptyset(&mut ignore.sa_mask);
        libc::sigaction(libc::SIGINT, &ignore, std::ptr::null_mut());
        libc::sigaction(libc::SIGTERM, &ignore, std::ptr::null_mut());

        let mut unblock: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut unblock);
        libc::sigaddset(&mut unblock, libc::SIGINT);
        libc::sigaddset(&mut unblock, libc::SIGTERM);
        libc::sigprocmask(libc::SIG_UNBLOCK, &unblock, std::ptr::null_mut());
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
    let mut child = spawn_with_unblocked_signals(&mut command)?;
    let pgid = libc::pid_t::try_from(child.id())
        .ok()
        .filter(|pid| *pid > 0)
        .ok_or_else(|| io::Error::other("anchor child id is not a valid process-group id"))?;
    let control = child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("anchor stdin control pipe missing"))?;
    let mut ready = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("anchor stdout ready pipe missing"))?;
    let mut byte = [0_u8; 1];
    if let Err(error) = ready.read_exact(&mut byte) {
        drop(control);
        let _ = child.wait();
        return Err(error);
    }
    if byte != *b"R" {
        drop(control);
        let _ = child.wait();
        return Err(io::Error::other("anchor emitted an invalid ready byte"));
    }
    // SAFETY: `pgid` is the live anchor's positive child pid. Readiness means
    // it is already executing its blocking anchor mode.
    let actual_pgid = unsafe { libc::getpgid(pgid) };
    if actual_pgid != pgid {
        drop(control);
        let _ = child.wait();
        return Err(io::Error::other(format!(
            "anchor process group was not established (pid {pgid}, pgid {actual_pgid})",
        )));
    }
    Ok(GroupAnchor {
        child,
        control: Some(control),
        pgid,
    })
}

fn release_anchor(anchor: &mut GroupAnchor) -> io::Result<ExitStatus> {
    drop(anchor.control.take());
    anchor.child.wait()
}

fn abandon_spawning() {
    let transitioned =
        ACTIVE_CHILD_GROUP.compare_exchange(SPAWNING, REAPING, Ordering::SeqCst, Ordering::SeqCst);
    debug_assert_eq!(
        transitioned,
        Ok(SPAWNING),
        "spawn-abort slot changed before pending handlers drained",
    );
    while HANDLERS_IN_FLIGHT.load(Ordering::SeqCst) != 0 {
        std::hint::spin_loop();
    }
    clear_pending_handoff_signals();
    ACTIVE_CHILD_GROUP.store(IDLE, Ordering::SeqCst);
}

fn abandon_spawning_anchor(anchor: &mut GroupAnchor) {
    let transitioned =
        ACTIVE_CHILD_GROUP.compare_exchange(SPAWNING, REAPING, Ordering::SeqCst, Ordering::SeqCst);
    debug_assert_eq!(
        transitioned,
        Ok(SPAWNING),
        "spawn-abort slot changed before pending handlers drained",
    );
    while HANDLERS_IN_FLIGHT.load(Ordering::SeqCst) != 0 {
        std::hint::spin_loop();
    }
    clear_pending_handoff_signals();
    let _ = release_anchor(anchor);
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

    // Deterministic test seam: real child exists in the anchor group, but the
    // signal handler still observes SPAWNING.
    handoff();

    let prior = ACTIVE_CHILD_GROUP.compare_exchange(
        SPAWNING,
        anchor.pgid,
        Ordering::SeqCst,
        Ordering::SeqCst,
    );
    debug_assert_eq!(prior, Ok(SPAWNING), "spawn slot changed during handoff");

    // A handler that loaded SPAWNING is counted until it publishes its pending
    // signal. New handlers see the positive anchor pgid and forward directly.
    while HANDLERS_IN_FLIGHT.load(Ordering::SeqCst) != 0 {
        std::hint::spin_loop();
    }
    replay_handoff_signals(anchor.pgid);
    Ok(GroupChild { child, anchor })
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

fn process_group_has_live_members(pgid: libc::pid_t) -> io::Result<bool> {
    let entries = std::fs::read_dir("/proc")?;
    for entry in entries.flatten() {
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
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        let Some(after_comm) = stat.rsplit_once(')').map(|(_, rest)| rest.trim_start()) else {
            continue;
        };
        let mut fields = after_comm.split_whitespace();
        let Some(state) = fields.next() else {
            continue;
        };
        let _ppid = fields.next();
        let Some(member_pgid) = fields
            .next()
            .and_then(|field| field.parse::<libc::pid_t>().ok())
        else {
            continue;
        };
        if member_pgid == pgid && state != "Z" && state != "X" {
            return Ok(true);
        }
    }
    Ok(false)
}

fn wait_group_quiescent(pgid: libc::pid_t) -> io::Result<()> {
    // A member can fork after its `/proc` entry was visited and then exit
    // before the new member's earlier-sorted entry would be observed. Require
    // two complete empty snapshots, separated in time, before releasing the
    // anchor that pins the pgid.
    let mut empty_snapshots = 0;
    loop {
        if process_group_has_live_members(pgid)? {
            empty_snapshots = 0;
        } else {
            empty_snapshots += 1;
            if empty_snapshots == 2 {
                return Ok(());
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
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
    debug_assert_eq!(
        transitioned,
        Ok(anchor.pgid),
        "active child slot lost its anchor pgid",
    );
    while HANDLERS_IN_FLIGHT.load(Ordering::SeqCst) != 0 {
        std::hint::spin_loop();
    }
    let result = release_anchor(anchor).map(|_| ());
    ACTIVE_CHILD_GROUP.store(IDLE, Ordering::SeqCst);
    result
}

fn terminate_group(group: &mut GroupChild) {
    signal_group(group.anchor.pgid, libc::SIGKILL);
    let _ = group.child.wait();
    let _ = wait_group_quiescent(group.anchor.pgid);
    let _ = finish_anchor(&mut group.anchor);
}

fn wait_leader_and_group(group: &mut GroupChild) -> io::Result<ExitStatus> {
    let status = group.child.wait()?;
    wait_group_quiescent(group.anchor.pgid)?;
    Ok(status)
}

fn run_status_with_handoff<F>(command: Command, handoff: F) -> io::Result<ExitStatus>
where
    F: FnOnce(),
{
    let mut group = spawn_group(command, handoff)?;
    let status = match wait_leader_and_group(&mut group) {
        Ok(status) => status,
        Err(error) => {
            terminate_group(&mut group);
            return Err(error);
        }
    };
    finish_anchor(&mut group.anchor)?;
    Ok(status)
}

fn spawn_reader<R>(reader: R, name: &'static str) -> io::Result<JoinHandle<io::Result<Vec<u8>>>>
where
    R: Read + Send + 'static,
{
    std::thread::Builder::new()
        .name(format!("ktstr-{name}-drain"))
        .spawn(move || {
            let mut reader = reader;
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes)?;
            Ok(bytes)
        })
}

fn join_reader(handle: JoinHandle<io::Result<Vec<u8>>>, name: &'static str) -> io::Result<Vec<u8>> {
    handle.join().map_err(|_| {
        io::Error::other(format!(
            "cargo-ktstr {name} drain thread panicked while collecting child output",
        ))
    })?
}

fn run_capture(mut command: Command, capture_stderr: bool) -> io::Result<Output> {
    command.stdout(Stdio::piped());
    if capture_stderr {
        command.stderr(Stdio::piped());
    }

    let mut group = spawn_group(command, || {})?;
    let stdout =
        group.child.stdout.take().ok_or_else(|| {
            io::Error::other("cargo-ktstr child stdout was not configured as a pipe")
        });
    let stdout = match stdout.and_then(|pipe| spawn_reader(pipe, "stdout")) {
        Ok(handle) => handle,
        Err(error) => {
            terminate_group(&mut group);
            return Err(error);
        }
    };

    let stderr = if capture_stderr {
        let pipe = match group.child.stderr.take() {
            Some(pipe) => pipe,
            None => {
                terminate_group(&mut group);
                let _ = stdout.join();
                return Err(io::Error::other(
                    "cargo-ktstr child stderr was not configured as a pipe",
                ));
            }
        };
        match spawn_reader(pipe, "stderr") {
            Ok(handle) => Some(handle),
            Err(error) => {
                terminate_group(&mut group);
                let _ = stdout.join();
                return Err(error);
            }
        }
    } else {
        None
    };

    let status = match wait_leader_and_group(&mut group) {
        Ok(status) => status,
        Err(error) => {
            terminate_group(&mut group);
            let _ = stdout.join();
            if let Some(stderr) = stderr {
                let _ = stderr.join();
            }
            return Err(error);
        }
    };

    // Keep the anchor pgid published while pipe drains finish. A descendant
    // that inherited stdout/stderr remains reachable by later interrupts.
    let stdout_result = join_reader(stdout, "stdout");
    let stderr_result = stderr
        .map(|handle| join_reader(handle, "stderr"))
        .unwrap_or_else(|| Ok(Vec::new()));
    let anchor_result = finish_anchor(&mut group.anchor);

    Ok(Output {
        status,
        stdout: stdout_result?,
        stderr: stderr_result?,
    })
    .and_then(|output| {
        anchor_result?;
        Ok(output)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

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
}
