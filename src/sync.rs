//! Generic synchronization primitives shared across ktstr subsystems.
//!
//! Keeps small, reusable blocking primitives out of feature-specific
//! modules. Callers compose these — they do not carry domain
//! semantics like "probe readiness" or "phase-B attach" in their
//! type or method names.

use std::sync::{Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Adds [`Self::lock_unpoisoned`] to [`Mutex`] for the recover-on-
/// poison policy every in-tree mutex caller follows. Equivalent to
/// `self.lock().unwrap_or_else(|e| e.into_inner())`, consolidated
/// in one named method so the policy ("we tolerate poison;
/// recovery is structurally safe because the protected state is
/// observation-only or already tombstoned") lives in one place
/// instead of ~70 hand-rolled call sites.
pub(crate) trait MutexExt<T> {
    /// Acquire the mutex, returning the inner guard regardless of
    /// poison state. A poisoned mutex still yields its protected
    /// data — the caller is responsible for treating the value as
    /// "possibly stale" if a previous thread panicked while
    /// holding the lock.
    fn lock_unpoisoned(&self) -> MutexGuard<'_, T>;
    /// Consume the mutex, returning the protected value regardless
    /// of poison state. Equivalent to
    /// `self.into_inner().unwrap_or_else(|e| e.into_inner())` —
    /// keeps the recover-on-poison policy in one place for the
    /// consuming variant alongside [`Self::lock_unpoisoned`].
    fn into_inner_unpoisoned(self) -> T
    where
        Self: Sized;
}

impl<T> MutexExt<T> for Mutex<T> {
    fn lock_unpoisoned(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn into_inner_unpoisoned(self) -> T {
        self.into_inner().unwrap_or_else(|e| e.into_inner())
    }
}

/// Adds [`Self::read_unpoisoned`] and [`Self::write_unpoisoned`] to
/// [`RwLock`] for the recover-on-poison policy. Mirrors
/// [`MutexExt::lock_unpoisoned`] — each method returns the inner
/// guard regardless of poison state.
pub(crate) trait RwLockExt<T> {
    /// Acquire a shared read guard, returning the inner guard
    /// regardless of poison state.
    fn read_unpoisoned(&self) -> RwLockReadGuard<'_, T>;
    /// Acquire an exclusive write guard, returning the inner guard
    /// regardless of poison state.
    fn write_unpoisoned(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T> RwLockExt<T> for RwLock<T> {
    fn read_unpoisoned(&self) -> RwLockReadGuard<'_, T> {
        self.read().unwrap_or_else(|e| e.into_inner())
    }
    fn write_unpoisoned(&self) -> RwLockWriteGuard<'_, T> {
        self.write().unwrap_or_else(|e| e.into_inner())
    }
}

/// One-shot signal from a producer thread to one or more waiters.
///
/// `set` flips the state and wakes every waiter currently blocked in
/// `wait`; subsequent waiters return immediately. Uses
/// `Mutex<bool> + Condvar` under the hood so waiters block in the
/// kernel instead of spinning. Replaces the `Arc<AtomicBool>` +
/// `while !flag { thread::sleep(10ms) }` pattern callers previously
/// used to hand off readiness between producer and consumer threads.
#[derive(Default)]
pub struct Latch {
    set: Mutex<bool>,
    cv: Condvar,
}

impl Latch {
    /// Create a new latch in the unset state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the latch and wake every waiter. Idempotent: a second call
    /// is a no-op beyond re-notifying, matching the previous
    /// `AtomicBool::store(true, Release)` semantics.
    pub fn set(&self) {
        let mut guard = self.set.lock().unwrap();
        *guard = true;
        self.cv.notify_all();
    }

    /// Block until `set` is called. Returns immediately if already set.
    pub fn wait(&self) {
        let mut guard = self.set.lock().unwrap();
        while !*guard {
            guard = self.cv.wait(guard).unwrap();
        }
    }

    /// Non-blocking check: return `true` iff the latch is currently
    /// set. The check briefly acquires the underlying mutex; callers
    /// on a hot path should prefer `wait` / `wait_timeout` (which
    /// block on the condvar) over polling this in a busy loop. The
    /// non-blocking variant exists for places that need to "branch
    /// on already-emitted" without committing to a blocking wait —
    /// e.g. a tail handler that re-emits only when the producer
    /// thread didn't already do it.
    pub fn is_set(&self) -> bool {
        *self.set.lock().unwrap()
    }

    /// Block until `set` is called or `timeout` elapses. Returns
    /// `true` if the latch was set within the deadline, `false` on
    /// timeout. `Condvar::wait_timeout` may return spuriously, so the
    /// loop re-checks the flag and recomputes the remaining duration
    /// against an absolute deadline.
    pub fn wait_timeout(&self, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let mut guard = self.set.lock().unwrap();
        while !*guard {
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }
            let remaining = deadline - now;
            let (g, res) = self.cv.wait_timeout(guard, remaining).unwrap();
            guard = g;
            if res.timed_out() && !*guard {
                return false;
            }
        }
        true
    }
}

/// Outcome of [`pidfd_poll_exited`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PidfdWait {
    /// The pidfd became readable — the process reached `EXIT_ZOMBIE`. The
    /// caller still reaps it (`Child::wait` / `waitpid`).
    Exited,
    /// `timeout` elapsed (or `poll(2)` errored) with the process still
    /// alive — the caller gives up (e.g. leaves it for a VM reboot).
    TimedOut,
    /// `pidfd_open(2)` failed (`ESRCH` — already reaped/gone — or an env
    /// defect). The caller should fall back to a non-blocking reap.
    NoPidfd,
}

/// Wait up to `timeout` for process `pid` to exit, evented via
/// `pidfd_open(2)` + `poll(2)` — no busy-poll, no signal sent, no reap.
/// The shared core of ktstr's bounded teardown reaps (the scheduler
/// reap and the worker reap): a SIGKILLed-but-wedged process — e.g. one
/// stuck uninterruptible in a syscall (a worker mid-`affine_move_task`
/// migration during a crash/bypass window) — cannot take its pending
/// signal until the syscall returns, so an unbounded
/// `wait`/`waitpid` would stall teardown for that whole window; this
/// caps the wait and lets the caller give up. The pidfd becomes readable
/// ONLY at `EXIT_ZOMBIE`, so [`PidfdWait::TimedOut`] means
/// genuinely-still-alive (not merely signal-pending).
pub(crate) fn pidfd_poll_exited(pid: libc::pid_t, timeout: std::time::Duration) -> PidfdWait {
    // SAFETY: pidfd_open(2) on `pid`, flags 0; returns an owned fd (>=0)
    // or -1 with errno set.
    let pidfd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0u32) as libc::c_int };
    if pidfd < 0 {
        return PidfdWait::NoPidfd;
    }
    let mut pfd = libc::pollfd {
        fd: pidfd,
        events: libc::POLLIN,
        revents: 0,
    };
    let ms = timeout.as_millis().min(i32::MAX as u128) as libc::c_int;
    // SAFETY: one valid pollfd; the pidfd becomes readable when `pid`
    // transitions to EXIT_ZOMBIE.
    let ret = unsafe { libc::poll(&mut pfd, 1, ms) };
    // SAFETY: closing the pidfd we just opened.
    unsafe {
        libc::close(pidfd);
    }
    if ret > 0 {
        PidfdWait::Exited
    } else {
        PidfdWait::TimedOut
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    /// An unset latch blocks `wait` until a producer thread calls
    /// `set`; the waiter must observe `set` before returning.
    #[test]
    fn latch_blocks_until_set_from_producer() {
        let latch = Arc::new(Latch::new());
        let l2 = latch.clone();
        let waiter = std::thread::spawn(move || {
            l2.wait();
        });
        // Give the waiter a chance to reach `cv.wait`.
        std::thread::sleep(Duration::from_millis(20));
        latch.set();
        waiter.join().unwrap();
    }

    /// A latch already in the set state returns from `wait`
    /// immediately — the mutex guards against the condvar missing the
    /// prior `notify_all`.
    #[test]
    fn latch_returns_immediately_when_already_set() {
        let latch = Latch::new();
        latch.set();
        let start = std::time::Instant::now();
        latch.wait();
        assert!(start.elapsed() < Duration::from_millis(50));
    }

    /// Two threads blocked in `wait` must both return after a single
    /// `set` — `notify_all` wakes every waiter in one call.
    #[test]
    fn set_wakes_every_waiter() {
        let latch = Arc::new(Latch::new());
        let a = latch.clone();
        let b = latch.clone();
        let wa = std::thread::spawn(move || a.wait());
        let wb = std::thread::spawn(move || b.wait());
        std::thread::sleep(Duration::from_millis(20));
        latch.set();
        wa.join().unwrap();
        wb.join().unwrap();
    }

    /// Calling `set` twice is idempotent — subsequent `wait` calls
    /// return immediately as they would after a single set.
    #[test]
    fn set_twice_is_idempotent() {
        let latch = Latch::new();
        latch.set();
        latch.set();
        latch.wait();
    }

    /// `lock_unpoisoned` on an unpoisoned mutex matches plain
    /// `.lock().unwrap()`.
    #[test]
    fn lock_unpoisoned_unpoisoned() {
        let m = Mutex::new(42);
        assert_eq!(*m.lock_unpoisoned(), 42);
    }

    /// Pins the recover-on-poison policy: after another thread
    /// panics while holding the mutex, `lock_unpoisoned()` still
    /// returns the inner guard with the protected state intact.
    #[test]
    fn lock_unpoisoned_recovers_from_poison() {
        let m = Arc::new(Mutex::new(99));
        let m_inner = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let _g = m_inner.lock().unwrap();
            panic!("poison the mutex");
        })
        .join();
        assert!(m.is_poisoned());
        assert_eq!(*m.lock_unpoisoned(), 99);
    }

    /// `into_inner_unpoisoned` on an unpoisoned mutex matches plain
    /// `.into_inner().unwrap()`.
    #[test]
    fn into_inner_unpoisoned_unpoisoned() {
        let m = Mutex::new(42);
        assert_eq!(m.into_inner_unpoisoned(), 42);
    }

    /// Pins the recover-on-poison policy for the consuming variant:
    /// after another thread panics while holding the mutex,
    /// `into_inner_unpoisoned()` still returns the inner value with
    /// the protected state intact.
    #[test]
    fn into_inner_unpoisoned_recovers_from_poison() {
        let m = Arc::new(Mutex::new(99));
        let m_inner = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let _g = m_inner.lock().unwrap();
            panic!("poison the mutex");
        })
        .join();
        assert!(m.is_poisoned());
        let mutex = Arc::try_unwrap(m).expect("only one Arc remains");
        assert_eq!(mutex.into_inner_unpoisoned(), 99);
    }

    /// `read_unpoisoned` / `write_unpoisoned` on an unpoisoned
    /// RwLock match plain `.read()` / `.write()`.
    #[test]
    fn rwlock_unpoisoned_unpoisoned() {
        let l = RwLock::new(7);
        assert_eq!(*l.read_unpoisoned(), 7);
        *l.write_unpoisoned() = 8;
        assert_eq!(*l.read_unpoisoned(), 8);
    }

    /// Pins recover-on-poison for both read and write paths on
    /// RwLock — mirrors `lock_unpoisoned_recovers_from_poison` for
    /// the rwlock case.
    #[test]
    fn rwlock_unpoisoned_recovers_from_poison() {
        let l = Arc::new(RwLock::new(123));
        let l_inner = Arc::clone(&l);
        let _ = std::thread::spawn(move || {
            let mut _w = l_inner.write().unwrap();
            panic!("poison the rwlock");
        })
        .join();
        assert!(l.is_poisoned());
        assert_eq!(*l.read_unpoisoned(), 123);
        *l.write_unpoisoned() = 456;
        assert_eq!(*l.read_unpoisoned(), 456);
    }

    /// `pidfd_poll_exited` reports `Exited` once a short-lived child
    /// reaches `EXIT_ZOMBIE`, and `TimedOut` (the pidfd never became
    /// readable) while a long-lived child is still alive — the
    /// bounded-wait contract the teardown reaps depend on.
    #[test]
    fn pidfd_poll_exited_distinguishes_exited_from_live() {
        // A child that exits well within the wait window → Exited.
        let mut quick = std::process::Command::new("sleep")
            .arg("0.1")
            .spawn()
            .expect("spawn sleep 0.1");
        assert_eq!(
            pidfd_poll_exited(quick.id() as libc::pid_t, Duration::from_secs(10)),
            PidfdWait::Exited,
        );
        quick.wait().unwrap(); // reap the zombie pidfd_poll_exited left

        // A child still alive when the (short) bound elapses → TimedOut.
        let mut sleeper = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep 30");
        assert_eq!(
            pidfd_poll_exited(sleeper.id() as libc::pid_t, Duration::from_millis(200)),
            PidfdWait::TimedOut,
        );
        sleeper.kill().unwrap();
        sleeper.wait().unwrap();
    }
}
