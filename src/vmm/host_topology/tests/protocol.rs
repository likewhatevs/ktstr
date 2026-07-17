//! Acquisition-protocol tests: inotify wake mechanics, claim
//! visibility/subtraction, queue lifecycle ownership, re-plan-on-wake,
//! and work conservation. Everything runs against real flocks in a
//! per-test tempdir (the lock-prefix override guards).

use super::super::protocol;
use super::super::*;
use super::*;

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
    assert!(
        protocol::read_live_claim().is_empty(),
        "claim must die with the marker flock (crashed-head safety)",
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
    let rl = crate::vmm::KtstrVm::acquire_default_run_locks(Some(&host), &topo, None, true)
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
