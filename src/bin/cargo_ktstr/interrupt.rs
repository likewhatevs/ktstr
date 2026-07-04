//! Interrupt guard for the child-run CLI paths (`cargo ktstr test`,
//! `perf-delta`).
//!
//! The parent (cargo-ktstr) shells `cargo nextest run` and blocks in
//! [`std::process::Command::status`] at the default SIGINT/SIGTERM
//! disposition, so a terminal Ctrl-C (or a `kill`) killed the parent
//! BEFORE it could run its post-run cleanup (the `/dev/shm` sweep, the
//! artifact footer, the perf-delta worktree removal).
//!
//! nextest itself, being in the same foreground process group, receives
//! its own SIGINT and tears down every per-test child (its own handler,
//! one process group per test) independently of the parent's
//! disposition. So the parent can safely record-and-survive the signal,
//! let `status()` return once nextest has finished its teardown, run its
//! cleanup in normal context, and then re-raise so the process still
//! exits with the conventional 128+signal status.
//!
//! The handler is async-signal-safe: it only stores the signal number and
//! sets an interrupted flag, both plain atomics. All cleanup runs later in
//! normal context, never in the handler.

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

/// Signal number recorded by [`handler`] on the first SIGINT/SIGTERM
/// delivery while a guard is installed; `0` means none seen. Read via
/// [`InterruptGuard::interrupted`]. Exactly one guard is live at a time
/// (the child-run paths are sequential), so one global suffices.
static CAUGHT_SIGNAL: AtomicI32 = AtomicI32::new(0);

/// Set true by [`handler`] on any SIGINT/SIGTERM while a guard is installed;
/// cleared by [`InterruptGuard::install`]. A `&AtomicBool` is exactly what
/// `gix`'s parallel checkout polls to abort mid-materialize, so perf-delta's
/// baseline checkout passes `&INTERRUPTED` to let a Ctrl-C abort the checkout
/// promptly instead of running it to completion.
pub(crate) static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// SIGINT/SIGTERM handler: record the signal and return so the blocked
/// `Command::status()` resumes (its `waitpid` is EINTR-retried by std)
/// once the child finishes its own teardown.
///
/// Async-signal-safe: the only operations are two lock-free atomic ops (a
/// compare-exchange on the signal number and a store on the interrupted
/// flag), both on the async-signal-safe list.
extern "C" fn handler(sig: libc::c_int) {
    // Keep the FIRST signal seen so SIGINT-then-SIGTERM reports SIGINT.
    let _ = CAUGHT_SIGNAL.compare_exchange(0, sig, Ordering::SeqCst, Ordering::SeqCst);
    // Abort any in-progress gix checkout polling this flag.
    INTERRUPTED.store(true, Ordering::SeqCst);
}

/// RAII guard that routes SIGINT + SIGTERM to [`handler`] for its
/// lifetime and restores the previous dispositions on drop. Install it
/// around a blocking child `Command::status()` so the parent survives the
/// signal, then check [`interrupted`](Self::interrupted) after the child
/// returns and call [`reraise`] once cleanup is done.
pub(crate) struct InterruptGuard {
    prev_sigint: libc::sigaction,
    prev_sigterm: libc::sigaction,
}

impl InterruptGuard {
    /// Install the SIGINT + SIGTERM handler, clearing any signal recorded
    /// by a previous guard. Only one guard should be live at a time.
    ///
    /// Relies on the caller blocking in std's default `Command::status()`
    /// (waitpid), which retries `EINTR` — so a signal delivered mid-wait
    /// resumes the wait rather than erroring. A child spawned with
    /// `create_pidfd(true)` would wait via `waitid` (NOT `EINTR`-retried in
    /// std) and would need re-checking.
    pub(crate) fn install() -> Self {
        CAUGHT_SIGNAL.store(0, Ordering::SeqCst);
        INTERRUPTED.store(false, Ordering::SeqCst);
        // SAFETY: `handler` is an `extern "C" fn` matching the sa_handler
        // ABI and touches only an atomic (async-signal-safe). `sigaction`
        // with valid pointers cannot fail for SIGINT/SIGTERM. `prev_*` are
        // zeroed sigactions, valid to overwrite. sa_flags = 0 (no
        // SA_RESETHAND: keep catching until Drop restores the prior
        // handler; std's `wait` retries the EINTR'd `waitpid`, so no
        // SA_RESTART is needed).
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = handler as *const () as usize;
            libc::sigemptyset(&mut sa.sa_mask);
            sa.sa_flags = 0;
            let mut prev_sigint: libc::sigaction = std::mem::zeroed();
            let mut prev_sigterm: libc::sigaction = std::mem::zeroed();
            libc::sigaction(libc::SIGINT, &sa, &mut prev_sigint);
            libc::sigaction(libc::SIGTERM, &sa, &mut prev_sigterm);
            Self {
                prev_sigint,
                prev_sigterm,
            }
        }
    }

    /// The signal that interrupted the guarded region, if any.
    pub(crate) fn interrupted(&self) -> Option<libc::c_int> {
        caught()
    }
}

/// The signal recorded by a live [`InterruptGuard`], if any — a
/// free-standing read for code running under a guard a caller installed
/// (e.g. perf-delta's `noise_dual_run` short-circuiting between child runs).
pub(crate) fn caught() -> Option<libc::c_int> {
    match CAUGHT_SIGNAL.load(Ordering::SeqCst) {
        0 => None,
        sig => Some(sig),
    }
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        // SAFETY: restoring the previously-saved sigactions captured in
        // install(); the pointers are valid for the call's duration.
        unsafe {
            libc::sigaction(libc::SIGINT, &self.prev_sigint, std::ptr::null_mut());
            libc::sigaction(libc::SIGTERM, &self.prev_sigterm, std::ptr::null_mut());
        }
    }
}

/// After a guarded region reported an interrupt AND the parent has
/// finished its cleanup, terminate with the conventional 128+signal
/// status (130 for SIGINT, 143 for SIGTERM) by resetting the signal to
/// its default disposition and re-raising it — matching the shell /
/// `WIFSIGNALED` convention. Drop the [`InterruptGuard`] before calling
/// so the handler is already uninstalled. Never returns.
pub(crate) fn reraise(sig: libc::c_int) -> ! {
    // SAFETY: SIG_DFL is always a valid disposition; `raise` on
    // SIGINT/SIGTERM under SIG_DFL terminates the process. Both are libc
    // FFI calls with no Rust-state effects (here run in normal context).
    unsafe {
        let mut dfl: libc::sigaction = std::mem::zeroed();
        dfl.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut dfl.sa_mask);
        libc::sigaction(sig, &dfl, std::ptr::null_mut());
        libc::raise(sig);
    }
    // A terminating signal under SIG_DFL should not return; if it somehow
    // does, fall back to the conventional exit code.
    std::process::exit(128 + sig)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read the current disposition of `sig`.
    fn current(sig: libc::c_int) -> usize {
        // SAFETY: querying a signal's disposition with a null `act` and a
        // valid `oldact` out-param is a well-defined sigaction(2) read.
        unsafe {
            let mut cur: libc::sigaction = std::mem::zeroed();
            libc::sigaction(sig, std::ptr::null(), &mut cur);
            cur.sa_sigaction
        }
    }

    /// install() replaces SIGINT/SIGTERM with the handler; a synthetic
    /// handler call records the signal; drop() restores the previous
    /// (sentinel) disposition verbatim. Each nextest test runs in its own
    /// process, so mutating global signal state here is test-isolated.
    #[test]
    fn install_records_and_restores() {
        // Install a SIG_IGN sentinel we can prove was restored.
        // SAFETY: SIG_IGN is a valid disposition; out-params are zeroed.
        let (mut pre_int, mut pre_term) = unsafe {
            (
                std::mem::zeroed::<libc::sigaction>(),
                std::mem::zeroed::<libc::sigaction>(),
            )
        };
        unsafe {
            let mut ign: libc::sigaction = std::mem::zeroed();
            ign.sa_sigaction = libc::SIG_IGN;
            libc::sigemptyset(&mut ign.sa_mask);
            libc::sigaction(libc::SIGINT, &ign, &mut pre_int);
            libc::sigaction(libc::SIGTERM, &ign, &mut pre_term);
        }
        assert_eq!(current(libc::SIGINT), libc::SIG_IGN, "sentinel in place");

        let guard = InterruptGuard::install();
        assert_eq!(guard.interrupted(), None, "no signal before delivery");
        assert!(
            !INTERRUPTED.load(Ordering::SeqCst),
            "install clears the interrupted flag"
        );
        let want = handler as *const () as usize;
        assert_eq!(
            current(libc::SIGINT),
            want,
            "install points SIGINT at handler"
        );
        assert_eq!(
            current(libc::SIGTERM),
            want,
            "install points SIGTERM at handler"
        );

        // Synthetic delivery — never `raise`, which would kill the test.
        handler(libc::SIGINT);
        assert_eq!(guard.interrupted(), Some(libc::SIGINT));
        assert!(
            INTERRUPTED.load(Ordering::SeqCst),
            "handler sets the interrupted flag so a gix checkout can abort"
        );
        // First signal wins.
        handler(libc::SIGTERM);
        assert_eq!(guard.interrupted(), Some(libc::SIGINT));

        drop(guard);
        assert_eq!(current(libc::SIGINT), libc::SIG_IGN, "drop restores SIGINT");
        assert_eq!(
            current(libc::SIGTERM),
            libc::SIG_IGN,
            "drop restores SIGTERM"
        );

        // Hand the process back to whatever ran before the test.
        unsafe {
            libc::sigaction(libc::SIGINT, &pre_int, std::ptr::null_mut());
            libc::sigaction(libc::SIGTERM, &pre_term, std::ptr::null_mut());
        }
    }
}
