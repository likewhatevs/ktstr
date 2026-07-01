//! schbench's message-thread <-> worker futex handshake, ported faithfully
//! from `schbench.c` (`fpost` :821, `fwait` :842, the wrappers + the
//! `FUTEX_BLOCKED`/`FUTEX_RUNNING` constants :808-815). A [`Handshake`] wraps
//! one futex word that ping-pongs between BLOCKED (0) and RUNNING (1):
//! [`Handshake::post`] hands the word to a waiter and wakes it; [`wait`] blocks
//! until posted (or times out), and [`wait_forever`] blocks with no deadline
//! (schbench's `fwait` with a NULL timeout, used by the message/worker loops).
//!
//! [`wait`]: Handshake::wait
//! [`wait_forever`]: Handshake::wait_forever
//! [`post`]: Handshake::post
//!
//! # Fidelity
//!
//! - PRIVATE futex (`FUTEX_WAIT_PRIVATE`/`FUTEX_WAKE_PRIVATE`), matching
//!   schbench (`schbench.c:827,852`). schbench is a single multi-threaded
//!   process and schbench_rs re-expresses that topology with threads
//!   (`CLONE_VM`, one address space), so a PRIVATE futex -- keyed on
//!   `(mm, address)` -- is correct and faster than a shared futex. (A SHARED
//!   futex would be required only for *forked processes*; ktstr's
//!   `worker::futex_wake`/`futex_wait` use the non-private op for exactly that
//!   reason. This handshake is thread-local to one schbench process, so it
//!   diverges to PRIVATE deliberately.)
//!
//! - The CAS uses sequentially-consistent ordering, matching schbench's
//!   `__sync_bool_compare_and_swap` (a full barrier).
//!
//! - No lost-wakeup window (kernel-source-verified). The kernel takes the
//!   futex-bucket lock, reads `*uaddr`, compares it to the expected value, and
//!   only then enqueues the waiter -- all atomic w.r.t. the bucket. So a
//!   `post` that lands between [`wait`]'s failed CAS and its `FUTEX_WAIT`
//!   entry makes the kernel see RUNNING != BLOCKED and return `EAGAIN` without
//!   blocking; [`wait`] loops and the CAS then consumes the RUNNING token. The
//!   stray `FUTEX_WAKE` from that `post` wakes zero waiters -- harmless.
//!
//! - Error policy diverges from schbench's `perror`+`exit(1)`
//!   (`schbench.c:828-831`, `:853-858`) in two ways. (1) [`wait`] loops on
//!   `EINTR` -- the standard correct behavior -- where schbench's
//!   `errno != EAGAIN` check (`schbench.c:853`) treats a signal as fatal and
//!   exits. (2) An unexpected errno panics instead of exiting. So [`wait`]
//!   loops on the transient set (`EAGAIN`/`EWOULDBLOCK` value-mismatch,
//!   `EINTR`) and returns [`WaitOutcome::TimedOut`] on `ETIMEDOUT`; any other
//!   errno (e.g. `EINVAL` from a caller's malformed `timespec`) panics rather
//!   than spinning at 100% CPU forever. [`post`] asserts its `FUTEX_WAKE`
//!   return is non-negative. With statically valid args the panic paths are
//!   unreachable -- they exist so a future caller bug fails loud, not silently.

use core::ffi::c_int;
use core::sync::atomic::{AtomicU32, Ordering};

/// Futex word value: no token pending, a waiter must block (`schbench.c:808`).
const FUTEX_BLOCKED: u32 = 0;
/// Futex word value: a token is posted, a waiter may proceed (`schbench.c:809`).
const FUTEX_RUNNING: u32 = 1;

/// `FUTEX_WAIT` with the private flag (`schbench.c:852`). Keyed on
/// `(current->mm, address)`, valid only within one address space.
const FUTEX_WAIT_PRIVATE: c_int = libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG;
/// `FUTEX_WAKE` with the private flag (`schbench.c:827`).
const FUTEX_WAKE_PRIVATE: c_int = libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG;

/// Outcome of [`Handshake::wait`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WaitOutcome {
    /// A [`post`](Handshake::post) was observed and consumed.
    Posted,
    /// The timeout elapsed with no post (`schbench.c` `fwait` returns
    /// `-ETIMEDOUT`). The word may still hold a racing post, which the next
    /// [`wait`](Handshake::wait) consumes.
    TimedOut,
}

/// One futex handshake word. Shared across threads (the futex itself provides
/// the synchronization; [`AtomicU32`] is [`Sync`]).
pub(crate) struct Handshake {
    word: AtomicU32,
}

impl Default for Handshake {
    fn default() -> Self {
        Self::new()
    }
}

impl Handshake {
    /// A fresh handshake in the BLOCKED state -- schbench sets
    /// `td->futex = FUTEX_BLOCKED` before a worker waits (`schbench.c:1007`).
    pub(crate) fn new() -> Self {
        Self {
            word: AtomicU32::new(FUTEX_BLOCKED),
        }
    }

    /// Post a token and wake a waiter. Faithful port of `fpost`
    /// (`schbench.c:821`): CAS BLOCKED -> RUNNING, and issue `FUTEX_WAKE`
    /// *only if* the CAS won (the waiter was actually blocked). If the word was
    /// already RUNNING (a prior post not yet consumed), skip the syscall.
    pub(crate) fn post(&self) {
        if self
            .word
            .compare_exchange(
                FUTEX_BLOCKED,
                FUTEX_RUNNING,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
        {
            // SAFETY: `self.word.as_ptr()` is a live `u32` reachable by every
            // thread sharing this `Handshake`; FUTEX_WAKE_PRIVATE wakes at most
            // one waiter and ignores the timeout/uaddr2/val3 args.
            let rc = unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    self.word.as_ptr(),
                    FUTEX_WAKE_PRIVATE,
                    1i32,
                    core::ptr::null::<libc::timespec>(),
                    core::ptr::null::<u32>(),
                    0u32,
                )
            };
            // FUTEX_WAKE returns the count woken (>= 0); -1 means a bug in the
            // (statically valid) args. Fail loud, matching schbench's
            // perror+exit (`schbench.c:828-831`), rather than swallow it.
            assert!(
                rc != -1,
                "FUTEX_WAKE_PRIVATE failed: {}",
                std::io::Error::last_os_error()
            );
        }
    }

    /// Wait for a posted token, blocking up to `timeout`. Faithful port of
    /// `fwait` (`schbench.c:842`): loop { CAS RUNNING -> BLOCKED to consume a
    /// token; else `FUTEX_WAIT` expecting BLOCKED }. Returns
    /// [`WaitOutcome::Posted`] once a token is consumed, or
    /// [`WaitOutcome::TimedOut`] if `timeout` elapses first. Spurious wakeups
    /// and `EAGAIN` (the word changed before the kernel enqueued us) just loop
    /// -- the CAS re-checks the state, so neither loses a post.
    ///
    /// Production message/worker loops wait with a NULL timeout via
    /// [`wait_forever`](Self::wait_forever); this timed entry is the faithful
    /// `fwait`-with-timeout half of the handshake and is exercised by the
    /// timeout tests until a timed-wait caller (e.g. RPS pacing) lands.
    #[allow(dead_code)]
    pub(crate) fn wait(&self, timeout: &libc::timespec) -> WaitOutcome {
        self.wait_inner(Some(timeout))
    }

    /// Wait indefinitely for a posted token -- schbench's `fwait` with a NULL
    /// timeout (`schbench.c:1032`, `:1176`), used by the message/worker loops
    /// where there is no deadline. The `FUTEX_WAIT` cannot time out, so this
    /// always returns once posted.
    pub(crate) fn wait_forever(&self) {
        let outcome = self.wait_inner(None);
        debug_assert_eq!(
            outcome,
            WaitOutcome::Posted,
            "a NULL-timeout wait cannot time out"
        );
    }

    /// Shared CAS-consume / `FUTEX_WAIT` loop. `None` waits indefinitely (NULL
    /// timeout, never returns [`WaitOutcome::TimedOut`]); `Some(ts)` returns
    /// `TimedOut` once `ts` elapses.
    fn wait_inner(&self, timeout: Option<&libc::timespec>) -> WaitOutcome {
        loop {
            if self
                .word
                .compare_exchange(
                    FUTEX_RUNNING,
                    FUTEX_BLOCKED,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                return WaitOutcome::Posted;
            }
            let ts_ptr = match timeout {
                Some(ts) => ts as *const libc::timespec,
                None => core::ptr::null::<libc::timespec>(),
            };
            // SAFETY: `self.word.as_ptr()` is a live `u32` reachable by every
            // thread that might post this word; the timespec (when `Some`)
            // outlives the call, and a NULL pointer requests an indefinite wait.
            let rc = unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    self.word.as_ptr(),
                    FUTEX_WAIT_PRIVATE,
                    FUTEX_BLOCKED,
                    ts_ptr,
                    core::ptr::null::<u32>(),
                    0u32,
                )
            };
            // rc == 0 (woken) falls through to re-CAS. On error, classify the
            // errno: a genuine timeout ends the wait, transient errors loop,
            // and anything else is a bug that fails loud (vs spinning forever).
            if rc == -1 {
                match std::io::Error::last_os_error().raw_os_error() {
                    // schbench returns -ETIMEDOUT here (`schbench.c:854-855`).
                    // Unreachable when `timeout` is `None` (NULL timeout).
                    Some(libc::ETIMEDOUT) => return WaitOutcome::TimedOut,
                    // EAGAIN (== EWOULDBLOCK on Linux): the word changed before
                    // the kernel enqueued us (a post raced in). EINTR: a signal.
                    // Both loop back to the CAS, which observes RUNNING iff a
                    // post actually landed -- no post is lost.
                    Some(libc::EAGAIN) | Some(libc::EINTR) => {}
                    // Any other errno (e.g. EINVAL from a malformed `timeout`)
                    // is a programming bug, not a transient condition. Fail loud
                    // instead of spinning at 100% CPU forever -- schbench
                    // perror+exit(1)s on it (`schbench.c:853-858`).
                    other => panic!("FUTEX_WAIT_PRIVATE failed: errno {other:?}"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timespec(secs: i64, nanos: i64) -> libc::timespec {
        // SAFETY: timespec is a plain POD struct; zeroing then setting the two
        // fields is valid. (Other libc timespec fields, if any, are padding.)
        let mut ts: libc::timespec = unsafe { core::mem::zeroed() };
        ts.tv_sec = secs as libc::time_t;
        ts.tv_nsec = nanos as _;
        ts
    }

    #[test]
    fn post_then_wait_consumes_via_fast_path() {
        // post sets RUNNING; wait's CAS consumes it immediately without
        // entering FUTEX_WAIT. Generous timeout is just a deadlock guard.
        let h = Handshake::new();
        h.post();
        assert_eq!(h.wait(&timespec(5, 0)), WaitOutcome::Posted);
        // Token consumed: a second wait with a short timeout must time out.
        assert_eq!(h.wait(&timespec(0, 50_000_000)), WaitOutcome::TimedOut);
    }

    #[test]
    fn wait_times_out_without_post() {
        let h = Handshake::new();
        assert_eq!(h.wait(&timespec(0, 100_000_000)), WaitOutcome::TimedOut);
    }

    #[test]
    fn cross_thread_post_wakes_waiter() {
        // One thread blocks in wait(); another posts. The waiter must observe
        // Posted. The 5s timeout is a deadlock guard, not the thing tested: a
        // working handshake returns in microseconds; only a broken wake path
        // would let it elapse and fail the assertion.
        let h = Handshake::new();
        std::thread::scope(|s| {
            let waiter = s.spawn(|| h.wait(&timespec(5, 0)));
            // The poster races the waiter; whether the waiter blocks first
            // (FUTEX_WAIT then FUTEX_WAKE) or the post lands first (CAS
            // fast-path), the outcome must be Posted.
            h.post();
            assert_eq!(waiter.join().unwrap(), WaitOutcome::Posted);
        });
    }

    #[test]
    fn wait_forever_consumes_posted_token() {
        // post sets RUNNING; wait_forever's CAS consumes it without entering
        // FUTEX_WAIT. Only the fast path is exercised here (there is no timeout
        // to guard a hang); the blocking + wake path shares `wait_inner` with
        // the timeout-guarded cross_thread test above.
        let h = Handshake::new();
        h.post();
        h.wait_forever(); // returns immediately; would hang if the token were missed
    }
}
