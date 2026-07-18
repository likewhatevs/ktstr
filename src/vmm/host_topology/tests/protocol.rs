//! Acquisition-protocol tests: inotify wake mechanics, claim
//! visibility/subtraction, queue lifecycle ownership, re-plan-on-wake,
//! and work conservation. Everything runs against real flocks in a
//! per-test tempdir (the lock-prefix override guards).

use super::super::protocol;
use super::super::*;
use super::*;

static INTERRUPTIBLE_FLOCK_BROKER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct InterruptibleFlockBrokerGuard {
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl InterruptibleFlockBrokerGuard {
    fn start() -> Self {
        let serial = INTERRUPTIBLE_FLOCK_BROKER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        crate::flock::start_interruptible_flock_broker().expect("start interruptible flock broker");
        Self { _serial: serial }
    }
}

impl Drop for InterruptibleFlockBrokerGuard {
    fn drop(&mut self) {
        crate::flock::stop_interruptible_flock_broker();
    }
}

fn wait_for_broker_signal_after(previous: usize) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while crate::flock::primitives::interruptible_flock_broker_signal_count() <= previous
        && std::time::Instant::now() < deadline
    {
        std::thread::yield_now();
    }
    assert!(
        crate::flock::primitives::interruptible_flock_broker_signal_count() > previous,
        "eventfd broker must deliver a targeted RT wake",
    );
}

/// True when the machine's 1-minute loadavg exceeds 60% of its core
/// count. The CI runners are colocated and routinely saturated, which
/// dilates the wall time of a promptness assertion past any fixed bound
/// even when the code under test did nothing slow — the delay is in the
/// scheduler, not the acquisition path. Read `/proc/loadavg` field 1
/// against the `/proc/stat` per-cpu line count (machine-wide, ignoring
/// any cpuset the test process runs under).
fn host_appears_loaded() -> bool {
    let machine_cpus = std::fs::read_to_string("/proc/stat")
        .ok()
        .map(|s| {
            s.lines()
                .filter(|l| {
                    l.starts_with("cpu") && l.as_bytes().get(3).is_some_and(u8::is_ascii_digit)
                })
                .count()
        })
        .filter(|&n| n > 0)
        .unwrap_or(1);
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok())
        .is_some_and(|load1| load1 > machine_cpus as f64 * 0.6)
}

/// EMPIRICAL verification of the inotify wake contract the head
/// engine sleeps on: releasing a flock (dropping the fd — the only
/// release path in this codebase) closes an `O_RDWR` fd and must fire
/// `IN_CLOSE_WRITE` on the lockfile's name in the watched directory.
/// The protocol watches both `IN_CLOSE_WRITE` and `IN_CLOSE_NOWRITE`
/// defensively; this test pins that the release edge is observable at
/// all — if a kernel/library change stopped these events, heads would
/// degrade to the 500 ms fallback tick (correct but slower), and this
/// test would catch the regression loudly instead.
#[test]
fn flock_release_fires_in_close_event() {
    use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let ino = Inotify::init(InitFlags::IN_NONBLOCK).expect("inotify init");
    ino.add_watch(
        tmp.path(),
        AddWatchFlags::IN_CLOSE_WRITE | AddWatchFlags::IN_CLOSE_NOWRITE,
    )
    .expect("add watch");

    let lockfile = tmp.path().join("release-edge.lock");
    let fd = crate::flock::try_flock(&lockfile, crate::flock::FlockMode::Exclusive)
        .expect("open")
        .expect("EX on fresh file");
    // Drain the open/create noise, then release and observe.
    std::thread::sleep(std::time::Duration::from_millis(50));
    let _ = ino.read_events();
    drop(fd);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let mut saw_close = false;
    while std::time::Instant::now() < deadline && !saw_close {
        if let Ok(events) = ino.read_events() {
            for ev in events {
                if ev.name.as_deref() == Some(std::ffi::OsStr::new("release-edge.lock")) {
                    saw_close = true;
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        saw_close,
        "dropping a flock fd must fire an IN_CLOSE_* event for the \
         lockfile — the head engine's wake mechanism depends on it",
    );
}

/// Claim liveness is the marker flock, not the manifest content: a
/// manifest with no live marker holder reads as NO claim (crashed or
/// finished head); the same manifest with the marker held reads back
/// verbatim; releasing the marker kills it again. This is the
/// crash-safety story — claims can never outlive their head.
#[test]
fn live_claim_requires_marker_holder() {
    let _prefixes = LockPrefixesGuard::new();
    let claim_json = serde_json::json!({"llcs": [3, 5], "cpus": [12]}).to_string();
    std::fs::write(protocol::head_claim_path(), &claim_json).expect("write manifest");

    assert!(
        protocol::read_live_claim().is_empty(),
        "manifest without a live marker holder must read as no claim",
    );

    let marker = crate::flock::try_flock(
        protocol::head_marker_path(),
        crate::flock::FlockMode::Exclusive,
    )
    .expect("open marker")
    .expect("EX on fresh marker");
    let live = protocol::read_live_claim();
    assert!(live.llcs.contains(&3) && live.llcs.contains(&5) && live.cpus.contains(&12));

    drop(marker);
    // A concurrent test thread can fork while the marker fd is live. CLOEXEC
    // closes the inherited copy at exec, but the raw-fork queue test retains
    // it until `_exit`, so the marker may remain live briefly after our local
    // close. That is an eventual-close contract, not a leaked claim: poll
    // through the bounded fork-child window and still fail diagnostically if
    // the marker persists.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let final_claim = loop {
        let claim = protocol::read_live_claim();
        if claim.is_empty() || std::time::Instant::now() >= deadline {
            break claim;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    assert!(
        final_claim.is_empty(),
        "claim must die with the marker flock (crashed-head safety); \
         still live after fork-child grace period: {final_claim:?}",
    );
}

/// Protocol rule 2 pinned end-to-end at the fast-path seam: a head
/// claiming LLCs {A, B} (holding A) fences BOTH from fast-path
/// callers — a caller wanting {B, C} must bounce WITHOUT taking B
/// even though B's lockfile is free — while a caller wanting {C, D}
/// (disjoint from the claim) proceeds. This is what stops disjoint
/// invocations from sniping the slots the head is accumulating.
#[test]
fn fast_path_subtracts_live_claim_but_takes_disjoint_capacity() {
    let _prefixes = LockPrefixesGuard::new();
    // Simulate a live head: marker flock held + published claim
    // {llcs: A=90900, B=90901}. The head "holds" A's lock; B is FREE
    // — the claim alone must fence it.
    let _marker = crate::flock::try_flock(
        protocol::head_marker_path(),
        crate::flock::FlockMode::Exclusive,
    )
    .expect("open marker")
    .expect("EX marker");
    std::fs::write(
        protocol::head_claim_path(),
        serde_json::json!({"llcs": [90900usize, 90901], "cpus": []}).to_string(),
    )
    .expect("write claim");
    let _head_holds_a =
        crate::flock::try_flock(llc_lock_path(90900), crate::flock::FlockMode::Shared)
            .expect("open A")
            .expect("SH on A");

    // Caller wanting {B, C}: must bounce on the CLAIM (B fenced), and
    // must not have taken B's lock in passing.
    let plan_bc = PinningPlan {
        assignments: vec![(0, 91000)],
        service_cpu: None,
        llc_indices: vec![90901, 90902],
        locks: Vec::new(),
    };
    let outcome =
        acquire_resource_locks(&plan_bc, &plan_bc.llc_indices, LlcLockMode::Shared).unwrap();
    let reason = expect_unavailable(outcome, Some("claimed LLC must fence the fast path"));
    assert!(
        reason.contains("claimed by the queue head"),
        "bounce must be attributed to the claim, not a flock race: {reason}",
    );
    // B's lockfile stayed untouched-free: an EX grab succeeds.
    let b_free = crate::flock::try_flock(llc_lock_path(90901), crate::flock::FlockMode::Exclusive)
        .expect("open B")
        .expect("B must still be free — the bounced caller must not have taken it");
    drop(b_free);

    // Caller wanting {C, D} (disjoint): proceeds immediately —
    // work-conserving fast path around the head's claim.
    let plan_cd = PinningPlan {
        assignments: vec![(0, 91001)],
        service_cpu: None,
        llc_indices: vec![90902, 90903],
        locks: Vec::new(),
    };
    let outcome =
        acquire_resource_locks(&plan_cd, &plan_cd.llc_indices, LlcLockMode::Shared).unwrap();
    let (_, locks) = unwrap_acquired(outcome, Some("disjoint capacity stays available"));
    assert_eq!(locks.len(), 3, "two LLC SH + one CPU EX");
}

/// A waiter keeps one kernel ticket while a chain of queue holders
/// advances, then acquires after the last release. Five successive
/// holders make the wait long enough to expose reopen/poll schemes
/// that lose their queue position.
#[test]
fn queue_ticket_survives_multiple_holder_turns() {
    let _prefixes = LockPrefixesGuard::new();
    let qpath = protocol::queue_lock_path();

    // Handover chain: five successive holder GENERATIONS, each a
    // forked child (distinct pid — the progress observation keys on
    // holder identity from /proc/locks) that opens its own fd, takes
    // LOCK_EX, holds ~220 ms, and exits. Forked children run only
    // async-signal-safe syscalls (open/flock/nanosleep/_exit); the
    // path CString is prepared pre-fork. In-process fork instead of
    // shelling out to flock(1) keeps the test host-tool-free.
    let qpath_c = std::ffi::CString::new(qpath.clone()).expect("no NUL in path");
    let mut children: Vec<libc::pid_t> = Vec::new();
    for _ in 0..5 {
        // SAFETY: child branch only calls async-signal-safe syscalls
        // and exits via _exit; parent branch just records the pid.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            unsafe {
                let fd = libc::open(qpath_c.as_ptr(), libc::O_RDWR | libc::O_CREAT, 0o666);
                if fd >= 0 {
                    libc::flock(fd, libc::LOCK_EX);
                    let ts = libc::timespec {
                        tv_sec: 0,
                        tv_nsec: 220_000_000,
                    };
                    libc::nanosleep(&ts, std::ptr::null_mut());
                }
                libc::_exit(0);
            }
        }
        children.push(pid);
        // Stagger so the chain overlaps: each child queues behind the
        // previous one inside the kernel.
        std::thread::sleep(std::time::Duration::from_millis(30));
    }
    let start = std::time::Instant::now();
    let _ticket = protocol::wait_for_queue_turn().expect("queue wait must not error");
    let elapsed = start.elapsed();
    for pid in children {
        // SAFETY: reaping our own forked children.
        unsafe {
            libc::waitpid(pid, std::ptr::null_mut(), 0);
        }
    }
    assert!(
        elapsed >= std::time::Duration::from_millis(700),
        "the waiter must actually have waited through multiple holders; \
         elapsed={elapsed:?}",
    );
}

/// A production [`protocol::QueueTurn`] release grants the next
/// blocking flock waiter directly.
#[test]
fn queue_turn_release_grants_waiter() {
    let _prefixes = LockPrefixesGuard::new();
    let holder = protocol::wait_for_queue_turn().expect("initial queue acquire");

    // Lock-prefix overrides are thread-local. Copy them so the helper joins
    // this test's isolated queue instead of the production one.
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
    let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
    let waiter = std::thread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = cpu_prefix);
        started_tx.send(()).expect("announce waiter start");
        let start = std::time::Instant::now();
        protocol::wait_for_queue_turn().expect("queue wait must not error");
        done_tx.send(start.elapsed()).expect("report queue acquire");
    });

    started_rx.recv().expect("waiter start");
    std::thread::sleep(std::time::Duration::from_millis(100));
    drop(holder);

    let elapsed = done_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("queue release must grant the blocking waiter");
    waiter.join().expect("queue waiter thread");
    assert!(
        elapsed >= std::time::Duration::from_millis(80),
        "the waiter must actually block behind the initial turn; elapsed={elapsed:?}",
    );
}

/// Cancellation preserves the ordinary queue's one-ticket FIFO behavior while
/// still closing the last-check/enter-flock race. The same registration stays
/// live after becoming head so its inotify sleep is interruptible, and the
/// published claim is removed before marker liveness is released.
#[test]
fn interrupt_wakes_queue_gap_and_head_poll_with_raii_claim_cleanup() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};

    // Declared first so broker shutdown/join is the final test cleanup action.
    let _broker = InterruptibleFlockBrokerGuard::start();
    let _prefixes = LockPrefixesGuard::new();
    let holder = protocol::wait_for_queue_turn().expect("initial queue acquire");

    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cancelled = Arc::new(AtomicBool::new(false));
    let waiter_cancelled = Arc::clone(&cancelled);
    let (gap_tx, gap_rx) = mpsc::sync_channel(1);
    let (continue_tx, continue_rx) = mpsc::sync_channel(1);
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    let waiter = std::thread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = cpu_prefix);
        let result =
            protocol::wait_for_queue_turn_interruptible_with_handoff(&waiter_cancelled, || {
                gap_tx.send(()).expect("announce final-check gap");
                continue_rx.recv().expect("release final-check gap");
            })
            .map(|_| ());
        result_tx
            .send(result)
            .expect("report interrupted queue wait");
    });

    if let Err(error) = gap_rx.recv_timeout(std::time::Duration::from_secs(2)) {
        cancelled.store(true, Ordering::Release);
        let waiter_id = crate::flock::interruptible_flock_waiter_id();
        crate::flock::wake_interruptible_flock_waiter(waiter_id);
        // The helper may already be parked inside the handoff closure. This
        // channel has capacity one, so sending is also safe if it has not
        // reached `recv` yet; ignore disconnect when it failed earlier.
        let _ = continue_tx.send(());
        drop(holder);
        waiter.join().expect("queue waiter thread");
        panic!("waiter did not reach the final-check/enter-flock gap: {error}");
    }
    let waiter_id = crate::flock::interruptible_flock_waiter_id();
    if waiter_id == 0 {
        cancelled.store(true, Ordering::Release);
        continue_tx.send(()).expect("release waiter into flock");
        drop(holder);
        waiter.join().expect("queue waiter thread");
        panic!("queue waiter must publish its registration generation");
    }
    cancelled.store(true, Ordering::Release);
    crate::flock::wake_interruptible_flock_waiter(waiter_id);
    continue_tx.send(()).expect("release waiter into flock");

    let queue_result = match result_rx.recv_timeout(std::time::Duration::from_secs(2)) {
        Ok(result) => result,
        Err(error) => {
            // Avoid stranding the helper if the assertion fails: releasing the
            // authoritative holder lets it leave the flock even if the wake
            // mechanism regressed.
            drop(holder);
            waiter.join().expect("queue waiter thread");
            panic!("cancelled waiter remained blocked behind a live holder: {error}");
        }
    };
    assert!(
        queue_result
            .expect_err("cancelled queue wait must not acquire")
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::Interrupted),
        "queue cancellation must surface ErrorKind::Interrupted",
    );
    waiter.join().expect("queue waiter thread");
    // The holder was deliberately still live when the waiter returned: the
    // wake, not an authoritative flock release, ended its wait.
    drop(holder);
    assert_eq!(
        crate::flock::interruptible_flock_waiter_id(),
        0,
        "queue registration must be torn down on error",
    );

    // Exercise the same registration across the queue-to-head transition.
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let head_cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&head_cancelled);
    let (head_result_tx, head_result_rx) = mpsc::sync_channel(1);
    let head = std::thread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = cpu_prefix);
        let result = (|| {
            let _turn = protocol::wait_for_queue_turn_interruptible(&worker_cancelled)?;
            protocol::acquire_as_head_interruptible(&worker_cancelled, |_| {
                let mut claim = protocol::ClaimSet::default();
                claim.llcs.insert(91991);
                Ok::<_, anyhow::Error>(protocol::HeadStep::<()>::Waiting { claim })
            })
        })();
        head_result_tx
            .send(result.map(|_| ()))
            .expect("report interrupted head wait");
    });

    let claim_deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let claim_path = protocol::head_claim_path();
    while !std::path::Path::new(&claim_path).exists() && std::time::Instant::now() < claim_deadline
    {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let claim_was_published = std::path::Path::new(&claim_path).exists();
    let head_waiter_id = crate::flock::interruptible_flock_waiter_id();
    head_cancelled.store(true, Ordering::Release);
    crate::flock::wake_interruptible_flock_waiter(head_waiter_id);

    let head_wake_start = std::time::Instant::now();
    let head_result = head_result_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("cancelled head poll must wake without its fallback delay");
    let head_wake_elapsed = head_wake_start.elapsed();
    assert!(
        claim_was_published,
        "head must publish its claim before sleeping",
    );
    assert!(
        head_waiter_id > 0,
        "queue head must retain its registered generation",
    );
    assert!(
        head_result
            .expect_err("cancelled head must not complete")
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::Interrupted),
        "head cancellation must surface ErrorKind::Interrupted",
    );
    if !host_appears_loaded() {
        assert!(
            head_wake_elapsed < std::time::Duration::from_millis(400),
            "RT cancellation must interrupt the head poll before its 500 ms \
             fallback; elapsed={head_wake_elapsed:?}",
        );
    }
    head.join().expect("queue head thread");
    assert!(
        !std::path::Path::new(&claim_path).exists(),
        "claim cleanup must run on the interrupt error path",
    );
    let _marker = crate::flock::try_flock(
        protocol::head_marker_path(),
        crate::flock::FlockMode::Exclusive,
    )
    .expect("open marker after cancellation")
    .expect("head marker must release after claim cleanup");
}

/// Queue an old generation in the broker, then tear it down and register a new
/// generation on the SAME Linux thread with the private RT signal originally
/// blocked. Drop must drain through EINTR, restore the blocked mask, and stop
/// the old retry before the new generation's poll.
#[test]
fn stale_generation_cannot_wake_same_tid_after_blocked_mask_restore() {
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, mpsc};

    let _broker = InterruptibleFlockBrokerGuard::start();
    let _prefixes = LockPrefixesGuard::new();
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let stale_cancelled = Arc::new(AtomicBool::new(false));
    let stale_worker_cancelled = Arc::clone(&stale_cancelled);
    let (first_id_tx, first_id_rx) = mpsc::sync_channel(1);
    let (advance_tx, advance_rx) = mpsc::sync_channel(1);
    let (second_id_tx, second_id_rx) = mpsc::sync_channel(1);
    let (stale_result_tx, stale_result_rx) = mpsc::sync_channel(1);
    let stale_worker = std::thread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = cpu_prefix);

        // Exercise the harder mask-restoration case: each registration
        // temporarily unblocks the private RT signal, while Drop must drain
        // every old wake and restore it to blocked before the next generation
        // on this same thread.
        let wake_signal = libc::SIGRTMIN() + 4;
        let mut wake_set: libc::sigset_t = unsafe { std::mem::zeroed() };
        unsafe {
            libc::sigemptyset(&mut wake_set);
            libc::sigaddset(&mut wake_set, wake_signal);
            assert_eq!(
                libc::pthread_sigmask(libc::SIG_BLOCK, &wake_set, std::ptr::null_mut(),),
                0,
                "block private RT wake before first registration",
            );
        }

        let first = protocol::wait_for_queue_turn_interruptible(&stale_worker_cancelled)
            .expect("first same-thread registration");
        first_id_tx
            .send(crate::flock::interruptible_flock_waiter_id())
            .expect("report first generation");
        advance_rx.recv().expect("advance to next generation");
        drop(first);

        let mut restored_mask: libc::sigset_t = unsafe { std::mem::zeroed() };
        unsafe {
            assert_eq!(
                libc::pthread_sigmask(libc::SIG_SETMASK, std::ptr::null(), &mut restored_mask,),
                0,
                "read mask restored by first registration",
            );
            assert_eq!(
                libc::sigismember(&restored_mask, wake_signal),
                1,
                "first registration must restore the originally blocked RT wake",
            );
        }

        let second = protocol::wait_for_queue_turn_interruptible(&stale_worker_cancelled)
            .expect("second same-thread registration");
        second_id_tx
            .send(crate::flock::interruptible_flock_waiter_id())
            .expect("report second generation");
        let result = protocol::LockDirWatch::new()
            .and_then(|watch| watch.wait(std::time::Duration::from_millis(150)));
        stale_result_tx
            .send(result)
            .expect("report stale-generation poll result");
        drop(second);
    });

    let first_id = first_id_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("first registration generation");
    let signal_count = crate::flock::primitives::interruptible_flock_broker_signal_count();
    crate::flock::wake_interruptible_flock_waiter(first_id);
    wait_for_broker_signal_after(signal_count);
    advance_tx.send(()).expect("advance same-thread waiter");
    let second_id = second_id_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("second registration generation");
    assert_ne!(
        first_id, second_id,
        "successive registrations need distinct generations",
    );
    let stale_result = stale_result_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("new generation's bounded poll");
    assert!(
        stale_result.is_ok(),
        "an old broker retry must not interrupt a later same-thread \
         registration: {stale_result:?}",
    );
    stale_worker.join().expect("same-thread generation worker");
}

/// Stop while the broker is actively retrying a live generation. Shutdown
/// hides the eventfd, drains handler writers, wakes through the owned fd, joins
/// the reader, and can then be restarted after the registration drops.
#[test]
fn interruptible_broker_stops_during_retry_and_restarts() {
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, mpsc};

    let _broker = InterruptibleFlockBrokerGuard::start();
    let _prefixes = LockPrefixesGuard::new();
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let shutdown_cancelled = Arc::new(AtomicBool::new(false));
    let shutdown_worker_cancelled = Arc::clone(&shutdown_cancelled);
    let (shutdown_id_tx, shutdown_id_rx) = mpsc::sync_channel(1);
    let (shutdown_release_tx, shutdown_release_rx) = mpsc::sync_channel(1);
    let shutdown_worker = std::thread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = cpu_prefix);
        let turn = protocol::wait_for_queue_turn_interruptible(&shutdown_worker_cancelled)
            .expect("shutdown-drain registration");
        shutdown_id_tx
            .send(crate::flock::interruptible_flock_waiter_id())
            .expect("report shutdown generation");
        shutdown_release_rx
            .recv()
            .expect("release shutdown registration");
        drop(turn);
    });
    let shutdown_id = shutdown_id_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("shutdown registration generation");
    let signal_count = crate::flock::primitives::interruptible_flock_broker_signal_count();
    crate::flock::wake_interruptible_flock_waiter(shutdown_id);
    wait_for_broker_signal_after(signal_count);
    crate::flock::stop_interruptible_flock_broker();
    shutdown_release_tx
        .send(())
        .expect("release registration after broker join");
    shutdown_worker
        .join()
        .expect("broker-shutdown registration worker");

    crate::flock::start_interruptible_flock_broker()
        .expect("broker must restart after a complete stop");
    crate::flock::stop_interruptible_flock_broker();
}

/// The queue does not diagnose a live holder from elapsed wall time.
/// It remains blocked until that holder's authoritative flock release;
/// holder crash cleanup and the external process rail own the wedged
/// case.
#[test]
fn queue_wait_defers_to_holder_lifecycle() {
    let _prefixes = LockPrefixesGuard::new();
    let qpath = protocol::queue_lock_path();
    let holder = crate::flock::try_flock(&qpath, crate::flock::FlockMode::Exclusive)
        .expect("open queue")
        .expect("EX on fresh queue");
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(800));
        drop(holder);
    });
    let start = std::time::Instant::now();
    let _ticket = protocol::wait_for_queue_turn().expect("queue wait must not error");
    let elapsed = start.elapsed();
    releaser.join().expect("holder release thread");
    assert!(
        elapsed >= std::time::Duration::from_millis(700),
        "the queue must wait for the holder's release, not invent an \
         earlier lifecycle verdict; elapsed={elapsed:?}",
    );
    if !host_appears_loaded() {
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "the waiter must acquire promptly after release; elapsed={elapsed:?}",
        );
    }
}

/// RE-PLAN-ON-WAKE, the falsifiable form (a REQUIREMENT of the
/// regime, not an implementation detail): the head enters the wait
/// wanting resource set X (all candidates busy); a DIFFERENT
/// sufficient set Y frees; the head must acquire Y promptly — never
/// keep waiting on X. Driven through the PRODUCTION default-path
/// acquisition (`KtstrVm::acquire_default_run_locks`) on a synthetic
/// two-LLC host: candidate X (LLC 0 / CPU 0) stays wedged for the
/// whole test; candidate Y (LLC 1 / CPU 1) frees after ~400 ms; the
/// returned plan must be Y's.
#[test]
fn head_replans_to_freed_alternative_candidate() {
    let _prefixes = LockPrefixesGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1]);
    let host = HostTopology::new_for_tests(&[(vec![0], 0), (vec![1], 0)]);
    let topo = crate::vmm::Topology::new(1, 1, 1, 1);

    // Wedge BOTH candidates' CPU locks (the per-CPU LOCK_EX is what
    // serializes 1:1 default pins); free Y's after 400 ms.
    let hold_x = crate::flock::try_flock(cpu_lock_path(0), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .expect("EX cpu0");
    let hold_y = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .expect("EX cpu1");
    // Copy the per-thread lock-prefix overrides into the releaser so
    // its paths (none needed — it only drops fds) stay coherent.
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(400));
        drop(hold_y);
    });

    let start = std::time::Instant::now();
    let rl = crate::vmm::KtstrVm::acquire_default_run_locks(Some(&host), &topo, true)
        .expect("acquisition must complete via the freed alternative");
    let elapsed = start.elapsed();
    releaser.join().expect("releaser thread");
    drop(hold_x);

    let plan = rl
        .pinning_plan
        .as_ref()
        .expect("1:1 pin, not overcommit — both candidates map");
    assert_eq!(
        plan.assignments,
        vec![(0, 1)],
        "the head must have re-planned onto the FREED candidate \
         (LLC 1 / CPU 1), not kept waiting on the wedged one",
    );
    assert!(
        elapsed >= std::time::Duration::from_millis(350),
        "must actually have waited for the release; elapsed={elapsed:?}",
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "the freed alternative must be taken PROMPTLY (inotify wake + \
         re-plan), not at some distant fallback; elapsed={elapsed:?}",
    );
}

/// WORK CONSERVATION: while a big `LOCK_EX` request is queued as head
/// and hungering for a claimed LLC, a concurrent small `LOCK_SH` cell
/// wanting DIFFERENT capacity completes immediately — it never waits
/// behind the head (no strict-FIFO head-of-line blocking for
/// satisfiable-now work). The head's own target stays fenced by its
/// claim (pinned separately above).
#[test]
fn small_shared_cell_proceeds_while_head_hungers() {
    let _prefixes = LockPrefixesGuard::new();
    // Peer wedges LLC 91100 with LOCK_SH so an EX head must wait.
    let peer_sh = crate::flock::try_flock(llc_lock_path(91100), crate::flock::FlockMode::Shared)
        .unwrap()
        .expect("peer SH");

    // Head: EX request for LLC 91100, queued in a helper thread. The
    // lock-prefix overrides are thread-local, so the helper re-installs
    // the same prefixes before acquiring.
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let head = std::thread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = cpu_prefix);
        let plan = PinningPlan {
            assignments: vec![(0, 91190)],
            service_cpu: None,
            llc_indices: vec![91100],
            locks: Vec::new(),
        };
        acquire_resource_locks_waiting(&plan, &[91100usize], LlcLockMode::Exclusive, true)
    });
    // Give the head time to queue + publish its claim.
    std::thread::sleep(std::time::Duration::from_millis(400));

    // Small SH cell on a DIFFERENT LLC/CPU: must complete promptly.
    let plan_small = PinningPlan {
        assignments: vec![(0, 91191)],
        service_cpu: None,
        llc_indices: vec![91101],
        locks: Vec::new(),
    };
    let start = std::time::Instant::now();
    let outcome =
        acquire_resource_locks(&plan_small, &plan_small.llc_indices, LlcLockMode::Shared).unwrap();
    let elapsed = start.elapsed();
    let (_, locks) = unwrap_acquired(outcome, Some("disjoint SH cell while the head hungers"));
    assert_eq!(locks.len(), 2);
    // The `Acquired` outcome above already proves work conservation
    // happened at all: a FIFO-strict queue would have forced this
    // disjoint-capacity cell to wait behind the head and return
    // `Unavailable`, which `unwrap_acquired` would have panicked on. The
    // wall-time bound below additionally proves the fast path incurred no
    // queue delay — but that latency is only measurable on a
    // quiet host. On a saturated CI runner the elapsed time is dominated
    // by thread-scheduling latency, so enforce the bound only when the
    // host is not loaded (the semantic proof holds unconditionally).
    if !host_appears_loaded() {
        assert!(
            elapsed < std::time::Duration::from_millis(200),
            "work conservation: the small cell must not wait behind the \
             hungering head; elapsed={elapsed:?}",
        );
    }

    // Release the peer; the head must now complete.
    drop(peer_sh);
    let head_outcome = head.join().expect("head thread").expect("head acquire");
    match head_outcome {
        LockOutcome::Acquired { locks, .. } => assert_eq!(locks.len(), 1),
        LockOutcome::Unavailable(r) => panic!("head must complete after the release: {r}"),
    }
}
