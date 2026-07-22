//! Acquisition-protocol tests: inotify wake mechanics, claim
//! visibility/subtraction, queue lifecycle ownership, re-plan-on-wake,
//! and work conservation. Everything runs against real flocks in a
//! per-test tempdir (the lock-prefix override guards).

use super::super::protocol;
use super::super::*;
use super::*;

static INTERRUPTIBLE_FLOCK_BROKER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn sustained_wait_diagnostics_use_a_bounded_cross_process_ring() {
    let _prefixes = LockPrefixesGuard::new();
    let diagnostics = tempfile::TempDir::new().expect("queue diagnostics tempdir");
    let mut pending = Vec::new();
    for llc in 0usize..(64 + 6) {
        pending.push(
            protocol::register_pending_claim_for_tests(protocol::ClaimSet::with_modes(
                [llc],
                std::iter::empty(),
                crate::flock::FlockMode::Shared,
                crate::flock::FlockMode::Shared,
            ))
            .expect("publish disjoint diagnostic fixture"),
        );
    }

    for bucket in 100u64..112 {
        protocol::persist_wait_diagnostic_for_tests(diagnostics.path(), bucket)
            .expect("persist bounded queue diagnostic");
    }

    let mut text_files = Vec::new();
    let mut lock_files = Vec::new();
    for entry in std::fs::read_dir(diagnostics.path()).expect("list queue diagnostics") {
        let path = entry.expect("queue diagnostic entry").path();
        match path.extension().and_then(std::ffi::OsStr::to_str) {
            Some("txt") => text_files.push(path),
            Some("lock") => lock_files.push(path),
            other => panic!("unexpected queue diagnostic artifact extension: {other:?}"),
        }
    }
    assert_eq!(
        text_files.len(),
        8,
        "the snapshot ring must stay fixed-size"
    );
    assert_eq!(
        lock_files.len(),
        8,
        "the writer-lock ring must stay fixed-size"
    );
    let newest = diagnostics.path().join("queue-wait-07.txt");
    let rendered = std::fs::read_to_string(&newest).expect("read newest queue diagnostic");
    assert!(rendered.starts_with("bucket=111\n"));
    assert!(rendered.contains("active_records=70"));
    assert!(rendered.contains("records_rendered=64 records_truncated=true"));
    assert!(rendered.len() <= 128 * 1024 + "\ntruncated_bytes=true\n".len());

    drop(pending);
}

#[test]
fn sustained_wait_diagnostics_keep_the_largest_live_queue_in_each_bucket() {
    let _prefixes = LockPrefixesGuard::new();
    let diagnostics = tempfile::TempDir::new().expect("queue diagnostics tempdir");
    let bucket = 200;
    let output = diagnostics.path().join("queue-wait-00.txt");
    let mut pending = vec![
        protocol::register_pending_claim_for_tests(protocol::ClaimSet::with_modes(
            [0usize],
            std::iter::empty(),
            crate::flock::FlockMode::Shared,
            crate::flock::FlockMode::Shared,
        ))
        .expect("publish small diagnostic fixture"),
    ];

    protocol::persist_wait_diagnostic_for_tests(diagnostics.path(), bucket)
        .expect("persist small queue diagnostic");
    assert!(
        std::fs::read_to_string(&output)
            .expect("read small queue diagnostic")
            .contains("active_records=1")
    );

    for llc in 1usize..12 {
        pending.push(
            protocol::register_pending_claim_for_tests(protocol::ClaimSet::with_modes(
                [llc],
                std::iter::empty(),
                crate::flock::FlockMode::Shared,
                crate::flock::FlockMode::Shared,
            ))
            .expect("grow diagnostic fixture"),
        );
    }
    protocol::persist_wait_diagnostic_for_tests(diagnostics.path(), bucket)
        .expect("replace bucket with larger queue diagnostic");
    let largest = std::fs::read_to_string(&output).expect("read larger queue diagnostic");
    assert!(largest.contains("active_records=12"));

    pending.truncate(1);
    protocol::persist_wait_diagnostic_for_tests(diagnostics.path(), bucket)
        .expect("retain larger queue diagnostic after smaller observation");
    assert_eq!(
        std::fs::read_to_string(&output).expect("read retained queue diagnostic"),
        largest,
        "a later small fixture must not overwrite the useful large-queue snapshot",
    );
}

#[test]
fn pending_activation_republishes_an_overlapping_watch_observation() {
    let _prefixes = LockPrefixesGuard::new();
    let (watched, candidate, pending) =
        protocol::exercise_pending_activation_overlap_watch_for_tests()
            .expect("exercise overlapping PENDING watch replacement");
    assert!(
        watched,
        "the exact record must retain the overlapping watch"
    );
    assert!(
        candidate,
        "the overlapping shared mode must remain observable"
    );
    assert!(
        pending,
        "dropping the old sole watch must request a fresh observation before scheduling",
    );
}

#[test]
fn live_pending_head_fences_snapshot_and_probe_without_ex_recovery() {
    let _prefixes = LockPrefixesGuard::new();
    let claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
        crate::flock::FlockMode::Exclusive,
    );
    let _pending = protocol::register_pending_claim_for_tests(claim.clone())
        .expect("publish a live coordinatorless PENDING head");
    let ex_before = protocol::registry_ex_acquisition_count_for_tests();

    let snapshot = protocol::registered_claim_snapshot(&claim)
        .expect("read the live PENDING claim without recovery");
    assert!(
        snapshot.conflicts(&claim).expect("pending conflict"),
        "the PENDING preparation claim must remain visible to snapshot planning",
    );
    assert_eq!(
        protocol::registry_ex_acquisition_count_for_tests(),
        ex_before,
        "a live PENDING head must not send aggregate snapshot through EX recovery",
    );

    let ran = std::cell::Cell::new(false);
    let outcome = protocol::with_registry_fence(&claim, || {
        ran.set(true);
        Ok::<_, anyhow::Error>(())
    })
    .expect("fence the live PENDING claim without recovery");
    assert!(matches!(outcome, protocol::RegistryFence::Fenced));
    assert!(!ran.get(), "a conflicting fast probe must remain fenced");
    assert_eq!(
        protocol::registry_ex_acquisition_count_for_tests(),
        ex_before,
        "a live PENDING head must not send the fast fence through EX recovery",
    );
}

#[test]
fn completed_preparation_retains_full_pending_claim_and_physical_ownership() {
    let _prefixes = LockPrefixesGuard::new();
    let protocol::PreparationContinuityForTests {
        pending,
        ticket,
        affinity_cpu,
        cpu_permits,
        memory_permits,
        token_permit,
        pending_claim,
    } = protocol::exercise_preparation_continuity_for_tests()
        .expect("complete preparation without weakening PENDING ownership");

    assert_eq!(cpu_permits.len(), PREPARATION_CPU_PERMITS);
    assert!(pending_claim.cpus.contains(&affinity_cpu));
    assert!(pending_claim.llcs.is_empty());
    assert!(pending_claim.permits.contains(&token_permit));
    assert_eq!(
        pending_claim
            .permits
            .intersection(&memory_permits.iter().copied().collect())
            .count(),
        memory_permits.len(),
    );
    assert_eq!(
        pending_claim
            .permits
            .intersection(&cpu_permits.iter().copied().collect())
            .count(),
        cpu_permits.len(),
        "completed preparation must retain every cooperative CPU permit",
    );

    let snapshot =
        protocol::ticket_registry_snapshot_for_tests().expect("snapshot completed PENDING record");
    assert_eq!(snapshot.len(), 1);
    assert_eq!(
        snapshot[0].0, ticket,
        "preparation completion must retain ticket identity"
    );
    assert_eq!(snapshot[0].2, pending_claim);

    assert!(
        crate::flock::try_flock(
            cpu_lock_path(affinity_cpu),
            crate::flock::FlockMode::Exclusive,
        )
        .expect("probe retained preparation affinity CPU")
        .is_none(),
        "completed preparation released its affinity CPU before exact activation",
    );
    for permit in &cpu_permits {
        assert!(
            crate::flock::try_flock(
                permit_lock_path(*permit),
                crate::flock::FlockMode::Exclusive,
            )
            .expect("probe retained preparation CPU permit")
            .is_none(),
            "completed preparation released CPU permit {permit} before exact activation",
        );
    }
    for permit in memory_permits.iter().chain(std::iter::once(&token_permit)) {
        assert!(
            crate::flock::try_flock(
                permit_lock_path(*permit),
                crate::flock::FlockMode::Exclusive,
            )
            .expect("probe retained preparation permit")
            .is_none(),
            "completed preparation released permit {permit} before exact activation",
        );
    }

    let perf_probe = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [affinity_cpu],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Exclusive,
    );
    assert!(
        protocol::registered_claim_conflicts(&perf_probe)
            .expect("probe perf admission after preparation completion"),
        "completed preparation must retain its CPU fence until exact activation",
    );

    drop(pending);
    assert!(
        protocol::ticket_registry_snapshot_for_tests()
            .expect("snapshot released completed preparation")
            .is_empty(),
    );
}

#[test]
fn synchronous_pending_retirement_cannot_self_fence_a_fresh_probe() {
    let _prefixes = LockPrefixesGuard::new();
    let retiring_claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [0usize],
        crate::flock::FlockMode::Exclusive,
        crate::flock::FlockMode::Exclusive,
    );
    let live_disjoint_claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
        crate::flock::FlockMode::Exclusive,
    );
    let retiring = protocol::register_pending_claim_for_tests(retiring_claim.clone())
        .expect("publish the claim which will be retired");
    let _live_disjoint = protocol::register_pending_claim_for_tests(live_disjoint_claim)
        .expect("keep an unrelated live PENDING record in the aggregate");

    retiring
        .retire_synchronously()
        .expect("synchronously retire the caller's own PENDING claim");

    let snapshot = protocol::registered_claim_snapshot(&retiring_claim)
        .expect("read aggregate after synchronous retirement");
    assert!(
        !snapshot
            .conflicts(&retiring_claim)
            .expect("retired-claim conflict query"),
        "an unrelated live PENDING record must not preserve the retired self-claim",
    );
    let ran = std::cell::Cell::new(false);
    let outcome = protocol::with_registry_fence(&retiring_claim, || {
        ran.set(true);
        Ok::<_, anyhow::Error>(())
    })
    .expect("fresh probe after synchronous PENDING retirement");
    assert!(matches!(outcome, protocol::RegistryFence::Ran { .. }));
    assert!(
        ran.get(),
        "the retired claim must not fence its owner's fresh probe"
    );
}

#[test]
fn pending_one_shot_promotes_the_same_ticket_without_waiting() {
    let _prefixes = LockPrefixesGuard::new();
    let preparation = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [0usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    let exact = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [0usize, 1usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    let pending = protocol::register_pending_claim_for_tests(preparation)
        .expect("publish one-shot PENDING claim");
    let ticket = protocol::ticket_registry_snapshot_for_tests()
        .expect("snapshot one-shot PENDING claim")[0]
        .0;
    let probes = std::cell::Cell::new(0usize);
    let waits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let wait_hook = waits.clone();
    let acquired = with_reservation_wait_progress(
        move || {
            wait_hook.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        },
        || {
            pending.try_activate_once(exact.clone(), || {
                probes.set(probes.get() + 1);
                Ok(Some(()))
            })
        },
    )
    .expect("atomically promote PENDING to HELD")
    .expect("one-shot candidate must acquire");
    assert_eq!(probes.get(), 1, "the physical callback runs exactly once");
    assert_eq!(
        waits.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "one-shot activation must not enter reservation waiting",
    );
    let snapshot =
        protocol::ticket_registry_snapshot_for_tests().expect("snapshot one-shot HELD claim");
    assert_eq!(snapshot.len(), 1);
    assert_eq!(
        snapshot[0].0, ticket,
        "activation must retain ticket identity"
    );
    assert_eq!(
        snapshot[0].2, exact,
        "the same slot must publish exact HELD"
    );
    assert_eq!(
        protocol::registered_claim_snapshot(&exact)
            .expect("snapshot one-shot HELD aggregate")
            .cpu_holder_count(1)
            .expect("exact CPU holder count"),
        1,
    );
    drop(acquired);
    assert!(
        protocol::ticket_registry_snapshot_for_tests()
            .expect("snapshot released one-shot claim")
            .is_empty(),
    );
}

#[test]
fn pending_one_shot_external_ex_fence_never_probes_or_queues() {
    let _prefixes = LockPrefixesGuard::new();
    let preparation = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [0usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    let exact = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [0usize, 1usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    let blocker_claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
        crate::flock::FlockMode::Exclusive,
    );
    let blocker = protocol::publish_acquired(&blocker_claim, ())
        .expect("publish external performance-style blocker");
    let pending = protocol::register_pending_claim_for_tests(preparation)
        .expect("publish PENDING claim beside blocker");
    let probes = std::cell::Cell::new(0usize);
    let waits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let wait_hook = waits.clone();
    let acquired = with_reservation_wait_progress(
        move || {
            wait_hook.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        },
        || {
            pending.try_activate_once(exact, || {
                probes.set(probes.get() + 1);
                Ok(Some(()))
            })
        },
    )
    .expect("reject externally fenced one-shot candidate");
    assert!(acquired.is_none());
    assert_eq!(probes.get(), 0, "registry fencing must precede the probe");
    assert_eq!(
        waits.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "fenced one-shot activation must not enter reservation waiting",
    );
    let snapshot = protocol::ticket_registry_snapshot_for_tests()
        .expect("snapshot after fenced one-shot retirement");
    assert_eq!(snapshot.len(), 1, "only the external HELD record remains");
    assert_eq!(snapshot[0].2, blocker_claim);
    drop(blocker);
}

#[test]
fn pending_one_shot_physical_miss_probes_once_and_retires() {
    let _prefixes = LockPrefixesGuard::new();
    let preparation = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [0usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    let exact = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [0usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    let pending = protocol::register_pending_claim_for_tests(preparation)
        .expect("publish physical-miss PENDING claim");
    let probes = std::cell::Cell::new(0usize);
    let acquired = pending
        .try_activate_once(exact, || {
            probes.set(probes.get() + 1);
            Ok::<_, anyhow::Error>(None::<()>)
        })
        .expect("complete physical-miss one-shot activation");
    assert!(acquired.is_none());
    assert_eq!(probes.get(), 1, "physical miss must not be retried");
    assert!(
        protocol::ticket_registry_snapshot_for_tests()
            .expect("snapshot retired physical-miss claim")
            .is_empty(),
        "a physical miss must synchronously remove PENDING",
    );
}

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

fn interruptible_flock_broker_service() -> TestTaskService {
    let broker_tid = loop {
        let tid = crate::flock::primitives::interruptible_flock_broker_tid();
        if tid != 0 {
            break tid;
        }
        // An unpublished TID means the newly spawned broker has not executed
        // its first instruction yet, so no delivered service can be charged.
        std::thread::yield_now();
    };
    TestTaskService::thread(std::process::id(), broker_tid)
}

fn wait_for_broker_signal_after(previous: usize) {
    let service = interruptible_flock_broker_service();
    wait_with_task_service(
        "eventfd broker targeted RT wake",
        std::slice::from_ref(&service),
        || {
            Ok(
                (crate::flock::primitives::interruptible_flock_broker_signal_count() > previous)
                    .then_some(()),
            )
        },
    )
    .expect("eventfd broker must deliver a targeted RT wake");
}

enum BrokerWakeOrCompletion<T> {
    BrokerSignalled,
    WaiterCompleted(T),
}

fn wait_for_broker_signal_or_waiter_completion<T>(
    previous: usize,
    receiver: &std::sync::mpsc::Receiver<T>,
    waiter: &TestTaskService,
) -> anyhow::Result<BrokerWakeOrCompletion<T>> {
    let broker = interruptible_flock_broker_service();
    let sources = [broker, waiter.clone()];
    wait_with_task_service(
        "eventfd broker targeted RT wake or waiter cancellation",
        &sources,
        || match receiver.try_recv() {
            Ok(result) => Ok(Some(BrokerWakeOrCompletion::WaiterCompleted(result))),
            Err(std::sync::mpsc::TryRecvError::Empty)
                if crate::flock::primitives::interruptible_flock_broker_signal_count()
                    > previous =>
            {
                Ok(Some(BrokerWakeOrCompletion::BrokerSignalled))
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => Ok(None),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                anyhow::bail!("registry waiter channel disconnected before cancellation completed")
            }
        },
    )
}

fn cancel_registry_worker(cancelled: &std::sync::atomic::AtomicBool) {
    let waiter = crate::flock::interruptible_flock_waiter_id();
    // Capture the generation before publishing cancellation so a waiter that
    // remains blocked can receive the broker wake. The waiter may instead
    // consume the flag and unregister before either this call or the broker
    // rechecks the generation; its completed Interrupted result is then the
    // authoritative cancellation edge and no targeted signal is necessary.
    cancelled.store(true, std::sync::atomic::Ordering::Release);
    if waiter != 0 {
        crate::flock::wake_interruptible_flock_waiter(waiter);
    }
}

const TEST_TRANSITION_SERVICE_BUDGET_NS: u64 = 1_000_000_000;
// Cross-process protocol transitions may deliberately wait through the
// 3-second waiter-crash recovery period plus its sub-second PID stagger.
// Charge the observer only after every live helper is blocked, and allow that
// one real semantic timer to expire before diagnosing an all-actor deadlock.
const TEST_EXTERNAL_BLOCKED_SERVICE_BUDGET_NS: u64 = 5_000_000_000;

fn protocol_test_thread_cpu_time_ns() -> anyhow::Result<u64> {
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `timestamp` is a valid out-pointer and `clock_gettime` writes
    // exactly one `timespec`.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut timestamp) } != 0 {
        anyhow::bail!(
            "read protocol-test thread CPU clock: {}",
            std::io::Error::last_os_error(),
        );
    }
    anyhow::ensure!(
        timestamp.tv_sec >= 0 && timestamp.tv_nsec >= 0,
        "protocol-test thread CPU clock returned a negative timestamp",
    );
    Ok((timestamp.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(timestamp.tv_nsec as u64))
}

#[derive(Clone, Debug)]
enum TestTaskService {
    Process {
        pid: u32,
    },
    HelperThread {
        pid: u32,
        tid_path: std::path::PathBuf,
    },
    Thread {
        pid: u32,
        tid: std::sync::Arc<std::sync::atomic::AtomicU32>,
    },
}

impl TestTaskService {
    fn process(pid: u32) -> Self {
        Self::Process { pid }
    }

    fn helper_thread(pid: u32, tid_path: std::path::PathBuf) -> Self {
        Self::HelperThread { pid, tid_path }
    }

    fn thread(pid: u32, tid: u32) -> Self {
        Self::Thread {
            pid,
            tid: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(tid)),
        }
    }

    fn published_thread() -> Self {
        Self::Thread {
            pid: std::process::id(),
            tid: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    fn publish_current_thread(&self) {
        // SAFETY: `SYS_gettid` has no pointer arguments and cannot violate
        // memory safety.
        let tid = unsafe { libc::syscall(libc::SYS_gettid) };
        assert!(tid > 0 && tid <= u32::MAX as libc::c_long);
        let Self::Thread { tid: published, .. } = self else {
            panic!("only a thread service source can publish its current TID");
        };
        published.store(tid as u32, std::sync::atomic::Ordering::Release);
    }
}

struct TestServiceThread<T> {
    handle: std::thread::JoinHandle<T>,
    service: TestTaskService,
}

impl<T: Send + 'static> TestServiceThread<T> {
    fn spawn(run: impl FnOnce() -> T + Send + 'static) -> Self {
        let service = TestTaskService::published_thread();
        let worker_service = service.clone();
        let handle = std::thread::spawn(move || {
            worker_service.publish_current_thread();
            run()
        });
        Self { handle, service }
    }

    fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    fn join(self) -> std::thread::Result<T> {
        self.handle.join()
    }
}

#[derive(Clone, Copy, Debug)]
struct TestTaskServiceSample {
    cpu_ns: u64,
    run_delay_ns: u64,
    state: u8,
}

fn test_task_sample(pid: u32, tid: u32) -> Option<TestTaskServiceSample> {
    let task = format!("/proc/{pid}/task/{tid}");
    let schedstat = std::fs::read_to_string(format!("{task}/schedstat")).ok()?;
    let mut fields = schedstat.split_whitespace();
    let cpu_ns = fields.next()?.parse().ok()?;
    let run_delay_ns = fields.next()?.parse().ok()?;
    let stat = std::fs::read_to_string(format!("{task}/stat")).ok()?;
    let state = stat
        .rfind(") ")
        .and_then(|close| stat.as_bytes().get(close + 2))
        .copied()?;
    Some(TestTaskServiceSample {
        cpu_ns,
        run_delay_ns,
        state,
    })
}

fn process_has_schedulable_task(pid: u32) -> bool {
    let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
        return false;
    };
    tasks.flatten().any(|task| {
        let Ok(tid) = task.file_name().to_string_lossy().parse::<u32>() else {
            return false;
        };
        test_task_sample(pid, tid).is_some_and(|sample| matches!(sample.state, b'R' | b'D'))
    })
}

#[derive(Default)]
struct TestTaskServiceSnapshot {
    pending: bool,
    tasks: std::collections::BTreeMap<(u32, u32), TestTaskServiceSample>,
}

impl TestTaskService {
    fn snapshot(&self) -> TestTaskServiceSnapshot {
        match self {
            Self::Process { pid } => {
                let mut snapshot = TestTaskServiceSnapshot::default();
                let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
                    return snapshot;
                };
                for task in tasks.flatten() {
                    let Ok(tid) = task.file_name().to_string_lossy().parse::<u32>() else {
                        continue;
                    };
                    if let Some(sample) = test_task_sample(*pid, tid) {
                        snapshot.tasks.insert((*pid, tid), sample);
                    }
                }
                snapshot
            }
            Self::HelperThread { pid, tid_path } => {
                let Some(tid) = std::fs::read_to_string(tid_path)
                    .ok()
                    .and_then(|tid| tid.trim().parse::<u32>().ok())
                else {
                    // The libtest worker has not entered the ignored helper
                    // test yet. Track whether any startup task is actually
                    // schedulable, but charge none of the harness process's
                    // unrelated startup service. If startup itself deadlocks,
                    // the observer budget still diagnoses it.
                    return TestTaskServiceSnapshot {
                        pending: process_has_schedulable_task(*pid),
                        ..TestTaskServiceSnapshot::default()
                    };
                };
                let mut snapshot = TestTaskServiceSnapshot::default();
                if let Some(sample) = test_task_sample(*pid, tid) {
                    snapshot.tasks.insert((*pid, tid), sample);
                }
                snapshot
            }
            Self::Thread { pid, tid } => {
                let tid = tid.load(std::sync::atomic::Ordering::Acquire);
                if tid == 0 {
                    return TestTaskServiceSnapshot {
                        pending: true,
                        ..TestTaskServiceSnapshot::default()
                    };
                }
                let mut snapshot = TestTaskServiceSnapshot::default();
                if let Some(sample) = test_task_sample(*pid, tid) {
                    snapshot.tasks.insert((*pid, tid), sample);
                }
                snapshot
            }
        }
    }
}

struct TestTaskServiceWatchdog {
    sources: Vec<TestTaskService>,
    previous: Vec<TestTaskServiceSnapshot>,
    charged_cpu_ns: Vec<u64>,
    blocked_observer_started_ns: u64,
    blocked_observer_budget_ns: u64,
}

impl TestTaskServiceWatchdog {
    fn new(sources: &[TestTaskService], blocked_observer_budget_ns: u64) -> anyhow::Result<Self> {
        let sources = sources.to_vec();
        let previous = sources.iter().map(TestTaskService::snapshot).collect();
        let source_count = sources.len();
        Ok(Self {
            sources,
            previous,
            charged_cpu_ns: vec![0; source_count],
            blocked_observer_started_ns: protocol_test_thread_cpu_time_ns()?,
            blocked_observer_budget_ns,
        })
    }

    fn poll(&mut self, context: &str) -> anyhow::Result<()> {
        let observer_now = protocol_test_thread_cpu_time_ns()?;
        let mut any_runnable = false;
        let mut made_progress = false;

        for (index, source) in self.sources.iter().enumerate() {
            let current = source.snapshot();
            let mut delivered_cpu_ns = 0u64;
            // A task in uninterruptible I/O sleep is still an active producer:
            // it is waiting for kernel/device service, not for another test
            // actor. Treat D consistently with the pre-publication process
            // probe above so an I/O-heavy storm cannot make the observer burn
            // its "every producer is blocked" CPU budget spuriously.
            any_runnable |= current.pending
                || current
                    .tasks
                    .values()
                    .any(|sample| matches!(sample.state, b'R' | b'D'));
            made_progress |= current.pending != self.previous[index].pending
                || !current.tasks.keys().eq(self.previous[index].tasks.keys());
            for (task, sample) in &current.tasks {
                if let Some(previous) = self.previous[index].tasks.get(task) {
                    delivered_cpu_ns = delivered_cpu_ns
                        .saturating_add(sample.cpu_ns.saturating_sub(previous.cpu_ns));
                    made_progress |= sample.cpu_ns != previous.cpu_ns
                        || sample.run_delay_ns != previous.run_delay_ns;
                } else {
                    made_progress = true;
                }
            }
            self.charged_cpu_ns[index] =
                self.charged_cpu_ns[index].saturating_add(delivered_cpu_ns);
            anyhow::ensure!(
                self.charged_cpu_ns[index] <= TEST_TRANSITION_SERVICE_BUDGET_NS,
                "{context} did not complete after awaited producer {index} ({source:?}) \
                 received {}ns of CPU service",
                self.charged_cpu_ns[index],
            );
            self.previous[index] = current;
        }

        if any_runnable || made_progress {
            self.blocked_observer_started_ns = observer_now;
        } else {
            let blocked_observer_service =
                observer_now.saturating_sub(self.blocked_observer_started_ns);
            anyhow::ensure!(
                blocked_observer_service <= self.blocked_observer_budget_ns,
                "{context} made no producer-task progress after every producer \
                 blocked and the observer received {blocked_observer_service}ns \
                 of CPU service",
            );
        }
        Ok(())
    }
}

/// Wait for one deterministic test handshake without charging host time in
/// which the task responsible for the transition did not run.
///
/// A runnable-but-starved producer consumes neither its CPU-service budget nor
/// the observer's blocked-state budget. Once every producer is blocked, the
/// observer's own delivered service bounds a genuinely missing transition.
fn wait_with_task_service<T>(
    context: &str,
    sources: &[TestTaskService],
    observe: impl FnMut() -> anyhow::Result<Option<T>>,
) -> anyhow::Result<T> {
    wait_with_task_service_config(context, sources, TEST_TRANSITION_SERVICE_BUDGET_NS, observe)
}

fn wait_with_external_task_service<T>(
    context: &str,
    sources: &[TestTaskService],
    observe: impl FnMut() -> anyhow::Result<Option<T>>,
) -> anyhow::Result<T> {
    wait_with_task_service_config(
        context,
        sources,
        TEST_EXTERNAL_BLOCKED_SERVICE_BUDGET_NS,
        observe,
    )
}

fn wait_with_task_service_config<T>(
    context: &str,
    sources: &[TestTaskService],
    blocked_observer_budget_ns: u64,
    mut observe: impl FnMut() -> anyhow::Result<Option<T>>,
) -> anyhow::Result<T> {
    let mut watchdog = TestTaskServiceWatchdog::new(sources, blocked_observer_budget_ns)?;
    let mut polls = 0usize;
    loop {
        if let Some(value) = observe()? {
            return Ok(value);
        }
        std::thread::yield_now();
        polls = polls.wrapping_add(1);
        if polls.is_multiple_of(64) {
            watchdog.poll(context)?;
        }
    }
}

fn wait_with_delivered_service<T>(
    context: &str,
    observe: impl FnMut() -> anyhow::Result<Option<T>>,
) -> anyhow::Result<T> {
    wait_with_task_service(context, &[], observe)
}

fn recv_with_task_service<T>(
    receiver: &std::sync::mpsc::Receiver<T>,
    context: &str,
    sources: &[TestTaskService],
) -> anyhow::Result<T> {
    wait_with_task_service(context, sources, || match receiver.try_recv() {
        Ok(value) => Ok(Some(value)),
        Err(std::sync::mpsc::TryRecvError::Empty) => Ok(None),
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
            anyhow::bail!("{context} channel disconnected before publishing")
        }
    })
}

fn recv_from_service_thread<T, U>(
    receiver: &std::sync::mpsc::Receiver<T>,
    context: &str,
    worker: &TestServiceThread<U>,
) -> anyhow::Result<T> {
    recv_with_task_service(receiver, context, std::slice::from_ref(&worker.service))
}

/// EMPIRICAL verification of the inotify wake contract the coordinator
/// engine sleeps on: releasing a flock (dropping the fd — the only
/// release path in this codebase) closes an `O_RDWR` fd and must fire
/// `IN_CLOSE_WRITE` on the lockfile's name in the watched directory.
/// The protocol intentionally watches that writable-close edge only, so
/// read-only liveness probes never enter its queue. This test pins that
/// holder release remains observable — if a kernel/library change stopped it,
/// coordinators would degrade to the 30 s missed-event maintenance wake
/// (correct but slower),
/// and this test would catch the regression loudly instead.
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
         lockfile — the coordinator's wake mechanism depends on it",
    );
}

#[test]
fn coordinator_watch_filters_registry_self_closes_without_spinning() {
    let _prefixes = LockPrefixesGuard::new();
    let claim = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    let coordinator =
        match protocol::register_ticket_or_acquire(claim.clone(), claim.clone(), None, |_| {
            Ok::<Option<()>, anyhow::Error>(None)
        })
        .expect("initialize registry")
        {
            protocol::TicketWork::Coordinator(coordinator) => coordinator,
            protocol::TicketWork::Acquired(_) => panic!("fresh registry must elect a coordinator"),
        };
    drop(coordinator);

    let watch = protocol::LockDirWatch::new_real_for_tests().expect("coordinator watch");
    watch
        .drain(&protocol::ClaimSet::default())
        .expect("drain coordinator watch");
    assert!(
        !protocol::registered_claim_conflicts(&claim).expect("read registry aggregate"),
        "empty registry must not fence a candidate",
    );
    let started = std::time::Instant::now();
    assert!(
        watch
            .wait(
                std::time::Duration::from_millis(120),
                &protocol::ClaimSet::default(),
            )
            .expect("bounded coordinator wait")
            .is_none(),
        "registry lock/header close events must not wake the coordinator",
    );
    assert!(
        started.elapsed() >= std::time::Duration::from_millis(80),
        "protocol-internal closes must not turn the bounded wait into an immediate self-wake",
    );
}

#[test]
fn coordinator_watch_classifies_close_events_by_watched_directory() {
    let _prefixes = LockPrefixesGuard::new();
    let watched = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    let watch = protocol::LockDirWatch::new_real_for_tests().expect("coordinator watch");
    watch.drain(&watched).expect("drain coordinator watch");

    let resource_dir = std::path::Path::new(&cpu_lock_path(1))
        .parent()
        .expect("CPU resource parent")
        .to_path_buf();
    let fake_notify = resource_dir.join("notify");
    drop(
        std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&fake_notify)
            .expect("resource-directory notify collision"),
    );

    let event_resource = protocol::registry_event_dir_for_tests().join(
        std::path::Path::new(&cpu_lock_path(1))
            .file_name()
            .expect("CPU resource basename"),
    );
    drop(
        std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&event_resource)
            .expect("event-directory resource collision"),
    );

    assert!(
        watch
            .wait(std::time::Duration::from_millis(120), &watched)
            .expect("bounded collision wait")
            .is_none(),
        "a basename collision on the wrong watch descriptor must not become a registry or resource event",
    );
}

#[test]
fn uncontended_fast_fence_does_not_create_registry_metadata() {
    let _prefixes = LockPrefixesGuard::new();
    let cpu_path = cpu_lock_path(1);
    let protocol_dir = std::path::Path::new(&cpu_path)
        .parent()
        .expect("resource lock parent");
    let registry_dir = protocol_dir.join("ktstr-acquire-registry-v20");
    let event_dir = protocol_dir.join("ktstr-acquire-events-v20");
    assert!(!registry_dir.exists());
    assert!(!event_dir.exists());

    assert!(
        protocol::ticket_registry_snapshot_for_tests()
            .expect("snapshot absent registry")
            .is_empty(),
        "never-created registry metadata is the authoritative empty state",
    );
    assert!(
        !registry_dir.exists() && !event_dir.exists(),
        "observing the empty registry must preserve the metadata-free fast path",
    );

    let claim = ticket_claim(&[1]);
    let outcome = protocol::with_registry_fence(&claim, || Ok::<_, anyhow::Error>("uncontended"))
        .expect("run absent-registry fence");
    assert!(matches!(
        outcome,
        protocol::RegistryFence::Ran {
            value: "uncontended",
            watched: false,
        }
    ));
    assert!(
        !registry_dir.exists() && !event_dir.exists(),
        "an uncontended fast probe must not create registry directories or lockfiles",
    );
}

#[test]
fn held_counts_track_shared_publications_but_not_queued_claims() {
    let _prefixes = LockPrefixesGuard::new();
    let claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );

    let queued =
        match protocol::register_ticket_or_acquire(claim.clone(), claim.clone(), None, |_| {
            Ok::<Option<()>, anyhow::Error>(None)
        })
        .expect("register queued claim")
        {
            protocol::TicketWork::Coordinator(coordinator) => coordinator,
            protocol::TicketWork::Acquired(_) => panic!("probe deliberately returned unavailable"),
        };
    let incompatible = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        FlockMode::Shared,
        FlockMode::Exclusive,
    );
    let queued_snapshot =
        protocol::registered_claim_snapshot(&incompatible).expect("read queued aggregate");
    assert!(
        queued_snapshot
            .conflicts(&incompatible)
            .expect("queued conflict"),
        "an exact queued claim must still fence an incompatible fast probe",
    );
    assert_eq!(
        queued_snapshot.cpu_holder_count(1).expect("queued count"),
        0,
        "a queue reservation is not a physical holder",
    );
    drop(queued);

    let first = crate::flock::try_flock(cpu_lock_path(1), FlockMode::Shared)
        .expect("first SH probe")
        .expect("first SH");
    let first = protocol::publish_acquired(&claim, first).expect("publish first SH");
    let second = crate::flock::try_flock(cpu_lock_path(1), FlockMode::Shared)
        .expect("second SH probe")
        .expect("second SH");
    let second = protocol::publish_acquired(&claim, second).expect("publish second SH");
    let both = protocol::registered_claim_snapshot(&claim).expect("read two SH holders");
    assert_eq!(both.cpu_holder_count(1).expect("two-holder count"), 2);
    assert!(
        !both.cpu_exclusive_held(1).expect("two-holder mode"),
        "two shared publications must not become an exclusive holder",
    );

    drop(first);
    assert_eq!(
        protocol::registered_claim_snapshot(&claim)
            .expect("read one SH holder")
            .cpu_holder_count(1)
            .expect("one-holder count"),
        1,
    );
    drop(second);
    assert_eq!(
        protocol::registered_claim_snapshot(&claim)
            .expect("read released holders")
            .cpu_holder_count(1)
            .expect("released count"),
        0,
    );
}

#[test]
fn acquired_drop_releases_physical_flock_before_held_record() {
    let _prefixes = LockPrefixesGuard::new();
    let path = cpu_lock_path(2);
    let claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [2usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let physical = crate::flock::try_flock(&path, FlockMode::Shared)
        .expect("physical SH probe")
        .expect("fresh physical SH");
    let acquired = protocol::publish_acquired(&claim, physical).expect("publish HELD claim");

    let observed = std::rc::Rc::new(std::cell::Cell::new(None));
    let hook_observed = std::rc::Rc::clone(&observed);
    let hook_claim = claim.clone();
    protocol::set_held_drop_hook_for_tests(move || {
        let physical_is_free = crate::flock::try_flock(&path, FlockMode::Exclusive)
            .expect("probe physical release inside HELD drop")
            .is_some();
        let published_count = protocol::registered_claim_snapshot(&hook_claim)
            .expect("read HELD record before removal")
            .cpu_holder_count(2)
            .expect("HELD count before removal");
        hook_observed.set(Some((physical_is_free, published_count)));
    });
    drop(acquired);

    assert_eq!(
        observed.get(),
        Some((true, 1)),
        "physical release must precede HELD removal, leaving only a conservative false-busy window",
    );
    assert_eq!(
        protocol::registered_claim_snapshot(&claim)
            .expect("read final holder count")
            .cpu_holder_count(2)
            .expect("final holder count"),
        0,
    );
}

#[test]
fn tracked_acquired_drop_keeps_its_registry_namespace_across_threads() {
    let _prefixes = LockPrefixesGuard::new();
    let claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [2usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let acquired = protocol::publish_acquired(&claim, ()).expect("publish isolated HELD claim");
    assert_eq!(
        protocol::ticket_registry_snapshot_for_tests()
            .expect("read isolated HELD record")
            .len(),
        1,
    );

    // Give the dropping thread a different valid protocol namespace and hold
    // that registry's EX lock. A teardown path which re-resolves thread-local
    // prefixes will block here; a captured owner must remove its record from
    // the original namespace without touching this lock.
    let wrong = tempfile::TempDir::new().expect("wrong-namespace tempdir");
    let wrong_llc_prefix = format!("{}/llc-", wrong.path().display());
    let wrong_cpu_prefix = format!("{}/cpu-", wrong.path().display());
    let wrong_registry = wrong.path().join("ktstr-acquire-registry-v20");
    std::fs::create_dir_all(&wrong_registry).expect("create wrong registry directory");
    let wrong_registry_lock =
        crate::flock::try_flock(wrong_registry.join("registry.lock"), FlockMode::Exclusive)
            .expect("open wrong registry lock")
            .expect("hold wrong registry EX");

    let (dropped_tx, dropped_rx) = std::sync::mpsc::sync_channel(1);
    let dropper = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(wrong_llc_prefix));
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(wrong_cpu_prefix));
        drop(acquired);
        dropped_tx.send(()).expect("publish cross-thread drop");
    });
    let completed = recv_from_service_thread(
        &dropped_rx,
        "captured-namespace HELD cleanup while the current namespace is locked",
        &dropper,
    );
    drop(wrong_registry_lock);
    dropper.join().expect("cross-thread HELD drop");
    completed.expect("HELD cleanup must not enter the dropping thread's registry");

    assert!(
        protocol::ticket_registry_snapshot_for_tests()
            .expect("read isolated registry after cross-thread drop")
            .is_empty(),
        "cross-thread Drop must remove the exact original HELD publication",
    );
}

#[test]
fn dead_held_publication_is_pruned_by_the_next_conflicting_snapshot() {
    let _prefixes = LockPrefixesGuard::new();
    let claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [3usize],
        FlockMode::Shared,
        FlockMode::Exclusive,
    );
    let physical = crate::flock::try_flock(cpu_lock_path(3), FlockMode::Exclusive)
        .expect("physical EX probe")
        .expect("fresh physical EX");
    let acquired = protocol::publish_acquired(&claim, physical).expect("publish HELD EX");
    assert_eq!(
        protocol::registered_claim_snapshot(&claim)
            .expect("read live HELD record")
            .cpu_holder_count(3)
            .expect("live HELD count"),
        1,
    );

    acquired.abandon_publication_for_tests();
    let recovered =
        protocol::registered_claim_snapshot(&claim).expect("recover dead HELD publication");
    assert_eq!(recovered.cpu_holder_count(3).expect("recovered count"), 0);
    assert!(
        !recovered.conflicts(&claim).expect("recovered conflict"),
        "a dead liveness owner must not leave a permanent reservation",
    );
}

#[test]
fn concurrent_held_publication_counts_are_exact() {
    const HOLDERS: usize = 32;

    let _prefixes = LockPrefixesGuard::new();
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE
        .with(|slot| slot.borrow().clone())
        .expect("LLC test prefix");
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE
        .with(|slot| slot.borrow().clone())
        .expect("CPU test prefix");
    let claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [4usize],
        FlockMode::Shared,
        FlockMode::Shared,
    );
    let published = std::sync::Arc::new(std::sync::Barrier::new(HOLDERS + 1));
    let release = std::sync::Arc::new(std::sync::Barrier::new(HOLDERS + 1));
    let mut threads = Vec::with_capacity(HOLDERS);
    for _ in 0..HOLDERS {
        let llc_prefix = llc_prefix.clone();
        let cpu_prefix = cpu_prefix.clone();
        let claim = claim.clone();
        let published = std::sync::Arc::clone(&published);
        let release = std::sync::Arc::clone(&release);
        threads.push(std::thread::spawn(move || {
            LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(llc_prefix));
            CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(cpu_prefix));
            let physical = crate::flock::try_flock(cpu_lock_path(4), FlockMode::Shared)
                .expect("concurrent SH probe")
                .expect("concurrent SH");
            let acquired =
                protocol::publish_acquired(&claim, physical).expect("publish concurrent HELD");
            published.wait();
            release.wait();
            drop(acquired);
        }));
    }

    published.wait();
    assert_eq!(
        protocol::registered_claim_snapshot(&claim)
            .expect("read concurrent holders")
            .cpu_holder_count(4)
            .expect("concurrent holder count"),
        HOLDERS,
    );
    release.wait();
    for thread in threads {
        thread.join().expect("concurrent publisher");
    }
    assert_eq!(
        protocol::registered_claim_snapshot(&claim)
            .expect("read released concurrent holders")
            .cpu_holder_count(4)
            .expect("released concurrent holder count"),
        0,
    );
}

#[test]
fn failed_complete_probe_releases_its_fresh_prefix_immediately() {
    let _prefixes = LockPrefixesGuard::new();
    let blocker = crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
        .expect("open CPU 2 blocker")
        .expect("lock CPU 2");
    let blocked_candidate =
        protocol::canonical_lock_order(&[], crate::flock::FlockMode::Shared, &[1, 2]);
    let reusable_prefix =
        protocol::canonical_lock_order(&[], crate::flock::FlockMode::Shared, &[1]);
    let mut held = protocol::HeldLocks::default();

    for _ in 0..256 {
        assert!(
            held.probe_complete(&blocked_candidate)
                .expect("probe blocked candidate")
                .is_none(),
            "CPU 2 must keep the larger candidate blocked",
        );
        let locks = held
            .probe_complete(&reusable_prefix)
            .expect("probe reusable prefix")
            .expect("the failed candidate's fresh CPU 1 fd must already be dropped");
        drop(locks);
    }
    drop(blocker);
}

#[test]
fn coordinator_probe_rejects_a_target_that_does_not_exactly_match_its_claim() {
    let _prefixes = LockPrefixesGuard::new();
    let claim = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    let wrong_resource = protocol::canonical_lock_order(&[], crate::flock::FlockMode::Shared, &[2]);
    let mut held = protocol::HeldLocks::default();
    let resource_error = held
        .probe_complete_if_ready(&claim, &wrong_resource)
        .expect_err("a physical target for CPU 2 cannot publish a CPU 1 claim");
    assert!(
        resource_error
            .to_string()
            .contains("does not exactly match"),
        "resource mismatch must fail diagnostically: {resource_error:#}",
    );

    let shared_claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    let exclusive_target =
        protocol::canonical_lock_order(&[], crate::flock::FlockMode::Shared, &[1]);
    let mode_error = held
        .probe_complete_if_ready(&shared_claim, &exclusive_target)
        .expect_err("an EX physical target cannot publish a SH CPU claim");
    assert!(
        mode_error.to_string().contains("does not match claim mode"),
        "mode mismatch must fail diagnostically: {mode_error:#}",
    );
}

#[test]
fn failed_complete_probe_does_not_sequester_its_free_prefix() {
    let _prefixes = LockPrefixesGuard::new();
    let blocked_cpu = 4;
    let blocker = crate::flock::try_flock(
        cpu_lock_path(blocked_cpu),
        crate::flock::FlockMode::Exclusive,
    )
    .expect("open final CPU blocker")
    .expect("lock final CPU");
    let target = protocol::canonical_lock_order(
        &[],
        crate::flock::FlockMode::Shared,
        &[1, 2, 3, blocked_cpu],
    );
    let mut held = protocol::HeldLocks::default();

    assert!(
        held.probe_complete(&target)
            .expect("probe four-CPU target")
            .is_none(),
        "a target missing its final resource must not report partial progress",
    );
    assert_eq!(
        held.contention_markers_for_tests(),
        vec![protocol::ContentionMarker {
            blocker: protocol::ResourceKey::Cpu(blocked_cpu),
            mode: crate::flock::FlockMode::Exclusive,
        }],
        "the failed all-or-nothing attempt must retain exact blocker evidence",
    );

    let mut reusable = Vec::new();
    for cpu in [1, 2, 3] {
        reusable.push(
            crate::flock::try_flock(cpu_lock_path(cpu), crate::flock::FlockMode::Exclusive)
                .expect("probe free prefix CPU")
                .expect("the failed coordinator attempt must release every free prefix CPU"),
        );
    }
    drop(reusable);
    drop(blocker);
}

#[test]
fn cpu_only_exact_claim_keeps_independent_canonical_modes() {
    let _prefixes = LockPrefixesGuard::new();
    let initial = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Shared,
    );
    let watch = protocol::ClaimSet::new(
        [9usize],
        [1usize, 2usize],
        crate::flock::FlockMode::Exclusive,
    );
    let coordinator = match protocol::register_ticket_or_acquire(initial, watch, None, |_| {
        Ok::<Option<()>, anyhow::Error>(None)
    })
    .expect("register CPU-only exact claim")
    {
        protocol::TicketWork::Coordinator(coordinator) => coordinator,
        protocol::TicketWork::Acquired(_) => panic!("fresh registry must elect a coordinator"),
    };
    let snapshot =
        protocol::ticket_registry_snapshot_for_tests().expect("read normalized exact claim");
    assert_eq!(snapshot.len(), 1);
    assert_eq!(
        snapshot[0].2.llc_mode,
        protocol::ClaimMode::Exclusive,
        "an absent LLC class must retain its canonical EX mode",
    );
    assert_eq!(
        snapshot[0].2.cpu_mode,
        protocol::ClaimMode::Exclusive,
        "the CPU exact mode must be encoded independently of the LLC class",
    );
    drop(coordinator);
}

#[test]
fn coordinator_watch_wakes_for_registry_notification() {
    let _prefixes = LockPrefixesGuard::new();
    let watch = protocol::LockDirWatch::new_real_for_tests().expect("coordinator watch");
    watch
        .drain(&protocol::ClaimSet::default())
        .expect("drain coordinator watch");
    let claim = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    let coordinator = match protocol::register_ticket_or_acquire(claim.clone(), claim, None, |_| {
        Ok::<Option<()>, anyhow::Error>(None)
    })
    .expect("register ticket")
    {
        protocol::TicketWork::Coordinator(coordinator) => coordinator,
        protocol::TicketWork::Acquired(_) => panic!("fresh registry must elect a coordinator"),
    };
    assert!(
        watch
            .wait(
                std::time::Duration::from_secs(1),
                &protocol::ClaimSet::default(),
            )
            .expect("wait for registry notification")
            .is_some(),
        "ticket publication must wake the coordinator watch",
    );
    drop(coordinator);
}

#[test]
fn stalled_takeover_wakes_and_parks_the_displaced_coordinator() {
    let _prefixes = LockPrefixesGuard::new();
    let watch = protocol::LockDirWatch::new_real_for_tests().expect("coordinator watch");
    let (slot_woken, coordinator_parked, successor_promoted, inotify_notified) =
        protocol::exercise_stalled_takeover_notification_for_tests(&watch)
            .expect("exercise stalled coordinator takeover");
    assert!(
        slot_woken,
        "stalled takeover must wake the displaced coordinator's slot futex",
    );
    assert!(
        coordinator_parked && successor_promoted,
        "stalled takeover must atomically park the old coordinator and promote its successor",
    );
    assert!(
        inotify_notified,
        "stalled takeover must also wake an old coordinator already blocked in real inotify",
    );
}

#[test]
fn dirty_repair_notifies_real_coordinator_watch_after_publishing_clean_state() {
    let _prefixes = LockPrefixesGuard::new();
    let watch = protocol::LockDirWatch::new_real_for_tests().expect("coordinator watch");
    let (repair_clean, coordinator_restored, slot_woken, inotify_notified) =
        protocol::exercise_dirty_repair_notification_for_tests(&watch)
            .expect("exercise dirty coordinator repair");
    assert!(
        repair_clean && coordinator_restored,
        "dirty repair must publish a clean image with one coherent coordinator",
    );
    assert!(
        slot_woken,
        "dirty repair must retain its targeted coordinator futex wake",
    );
    assert!(
        inotify_notified,
        "dirty repair must wake a real coordinator watch after clearing the torn image",
    );
}

#[test]
fn coordinator_watch_observes_owner_liveness_close_but_not_readonly_probes() {
    let _prefixes = LockPrefixesGuard::new();
    let claim = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    let coordinator = match protocol::register_ticket_or_acquire(claim.clone(), claim, None, |_| {
        Ok::<Option<()>, anyhow::Error>(None)
    })
    .expect("register coordinator")
    {
        protocol::TicketWork::Coordinator(coordinator) => coordinator,
        protocol::TicketWork::Acquired(_) => panic!("fresh registry must elect a coordinator"),
    };
    let watch = protocol::LockDirWatch::new_real_for_tests().expect("coordinator watch");
    watch
        .drain(&protocol::ClaimSet::default())
        .expect("drain registration events");

    let (identity, live) =
        protocol::coordinator_liveness_probe_for_tests().expect("probe coordinator liveness");
    assert!(live, "coordinator liveness fd must be locked");
    assert!(
        protocol::missing_liveness_probe_does_not_create_for_tests()
            .expect("probe absent liveness inode"),
        "an O(1) liveness check must never create a missing owner inode",
    );
    assert!(
        watch
            .wait(
                std::time::Duration::from_millis(120),
                &protocol::ClaimSet::default(),
            )
            .expect("bounded liveness-probe wait")
            .is_none(),
        "O_RDONLY liveness probes and registry mappings must not enqueue actionable closes",
    );

    drop(coordinator);
    let events = watch
        .wait(
            std::time::Duration::from_secs(1),
            &protocol::ClaimSet::default(),
        )
        .expect("wait for liveness owner close")
        .expect("owner close must be actionable");
    assert!(
        events.contains_liveness(identity),
        "dropping the O_RDWR liveness owner must report its exact slot+ticket identity",
    );
}

#[test]
fn coordinator_watch_wakes_for_resource_release() {
    let _prefixes = LockPrefixesGuard::new();
    let held = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .expect("open resource lock")
        .expect("take fresh resource lock");
    let watch = protocol::LockDirWatch::new_real_for_tests().expect("coordinator watch");
    let watched = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    watch.drain(&watched).expect("drain coordinator watch");
    drop(held);
    assert!(
        watch
            .wait(std::time::Duration::from_secs(1), &watched)
            .expect("wait for resource release")
            .is_some(),
        "resource-lock close must wake the coordinator watch",
    );
}

#[test]
fn coordinator_watch_yields_between_bounded_close_storm_chunks() {
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let watch = protocol::LockDirWatch::new_real_for_tests().expect("coordinator watch");
    let cpu_path = cpu_lock_path(0);
    let resource_dir = std::path::Path::new(&cpu_path)
        .parent()
        .expect("CPU resource parent");
    for index in 0..2_048usize {
        drop(
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(resource_dir.join(format!("unwatched-storm-{index}")))
                .expect("create unwatched close"),
        );
    }
    drop(
        std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(protocol::registry_event_dir_for_tests().join("notify"))
            .expect("create trailing registry notification"),
    );

    let watched = protocol::ClaimSet::default();
    let first = watch.drain(&watched).expect("drain first bounded turn");
    assert!(
        first.has_backlog(),
        "a close storm larger than one coordinator budget must yield with backlog",
    );
    assert!(
        !first.overflowed(),
        "a user-space fairness budget must not masquerade as kernel queue overflow",
    );

    let mut saw_notify = first.contains_registry_notify();
    for _ in 0..32 {
        if saw_notify {
            break;
        }
        let events = watch.drain(&watched).expect("drain next bounded turn");
        assert!(
            !events.overflowed(),
            "the bounded test storm must remain below the kernel overflow limit",
        );
        saw_notify |= events.contains_registry_notify();
    }
    assert!(
        saw_notify,
        "bounded turns must eventually reach a trailing actionable notification",
    );
}

#[test]
fn claim_llc_compatibility_matches_flock_matrix() {
    let shared = protocol::ClaimSet::new(
        [7usize],
        std::iter::empty(),
        crate::flock::FlockMode::Shared,
    );
    assert!(
        !shared.conflicts_with_llc(7, crate::flock::FlockMode::Shared),
        "SH request is compatible with a SH claim",
    );
    assert!(
        shared.conflicts_with_llc(7, crate::flock::FlockMode::Exclusive),
        "EX request conflicts with a SH claim",
    );

    let exclusive = protocol::ClaimSet::new(
        [7usize],
        std::iter::empty(),
        crate::flock::FlockMode::Exclusive,
    );
    assert!(
        exclusive.conflicts_with_llc(7, crate::flock::FlockMode::Shared),
        "SH request conflicts with an EX claim",
    );
    assert!(
        exclusive.conflicts_with_llc(7, crate::flock::FlockMode::Exclusive),
        "EX request conflicts with an EX claim",
    );
    assert!(
        !exclusive.conflicts_with_llc(8, crate::flock::FlockMode::Exclusive),
        "an unrelated LLC never conflicts",
    );
}

#[test]
fn complete_claim_compatibility_matches_resource_lock_matrix() {
    let shared_a = protocol::ClaimSet::new([1usize], [10usize], crate::flock::FlockMode::Shared);
    let shared_disjoint_cpu =
        protocol::ClaimSet::new([1usize], [11usize], crate::flock::FlockMode::Shared);
    let shared_same_cpu =
        protocol::ClaimSet::new([2usize], [10usize], crate::flock::FlockMode::Shared);
    let exclusive_same_llc = protocol::ClaimSet::new(
        [1usize],
        std::iter::empty(),
        crate::flock::FlockMode::Exclusive,
    );

    assert!(
        !shared_a.conflicts_with(&shared_disjoint_cpu),
        "SH/SH LLC overlap with disjoint exclusive CPU locks is compatible",
    );
    assert!(
        shared_a.conflicts_with(&shared_same_cpu),
        "an exact CPU overlap always conflicts",
    );
    assert!(
        shared_a.conflicts_with(&exclusive_same_llc),
        "SH/EX overlap on one LLC conflicts",
    );

    let cpu_shared_a = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [10usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    let cpu_shared_b = cpu_shared_a.clone();
    let cpu_exclusive = protocol::ClaimSet::new(
        std::iter::empty(),
        [10usize],
        crate::flock::FlockMode::Exclusive,
    );
    assert!(
        !cpu_shared_a.conflicts_with(&cpu_shared_b),
        "CPU SH/SH overlap is compatible",
    );
    assert!(
        cpu_shared_a.conflicts_with(&cpu_exclusive) && cpu_exclusive.conflicts_with(&cpu_shared_a),
        "CPU SH/EX overlap conflicts symmetrically",
    );
}

#[test]
fn empty_resource_modes_are_canonical_across_construction_union_and_record_round_trip() {
    let llc_new = protocol::ClaimSet::new(
        [7usize],
        std::iter::empty(),
        crate::flock::FlockMode::Shared,
    );
    let llc_with_modes = protocol::ClaimSet::with_modes(
        [7usize],
        std::iter::empty(),
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    assert_eq!(llc_new, llc_with_modes);

    let cpu_shared = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [9usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    let empty = protocol::ClaimSet::default();
    assert_eq!(
        protocol::union_claims_for_tests(&llc_new, &empty),
        llc_new,
        "an absent CPU/LLC class must not upgrade a present SH mode",
    );
    assert_eq!(
        protocol::union_claims_for_tests(&empty, &cpu_shared),
        cpu_shared,
        "union must ignore the canonical EX mode of an absent operand class",
    );
    assert_eq!(
        protocol::union_claims_for_tests(&empty, &empty),
        empty,
        "the both-empty union remains canonical",
    );
    assert_eq!(
        [llc_new.clone()]
            .into_iter()
            .reduce(|envelope, claim| envelope.union_envelope(&claim))
            .expect("one candidate"),
        llc_new,
        "a one-candidate LLC-SH envelope is exactly its SH claim",
    );

    let (round_trip_claim, round_trip_watch) =
        protocol::round_trip_claim_modes_for_tests(&llc_new, &cpu_shared)
            .expect("round-trip independently-modeled claim and watch");
    assert_eq!(round_trip_claim, llc_new);
    assert_eq!(round_trip_watch, cpu_shared);
}

#[test]
fn registry_cpu_aggregate_uses_the_shared_exclusive_compatibility_matrix() {
    let _prefixes = LockPrefixesGuard::new();
    let shared = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [31usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    let coordinator =
        match protocol::register_ticket_or_acquire(shared.clone(), shared.clone(), None, |_| {
            Ok::<Option<()>, anyhow::Error>(None)
        })
        .expect("register CPU SH aggregate owner")
        {
            protocol::TicketWork::Coordinator(coordinator) => coordinator,
            protocol::TicketWork::Acquired(_) => panic!("fresh registry must elect a coordinator"),
        };
    assert!(
        !protocol::registered_claim_conflicts(&shared).expect("query CPU SH aggregate"),
        "an active CPU SH claim must permit another CPU SH fast probe",
    );
    let exclusive = protocol::ClaimSet::new(
        std::iter::empty(),
        [31usize],
        crate::flock::FlockMode::Exclusive,
    );
    assert!(
        protocol::registered_claim_conflicts(&exclusive).expect("query CPU EX aggregate"),
        "an active CPU SH claim must fence an overlapping CPU EX fast probe",
    );
    drop(coordinator);
}

#[test]
fn live_build_and_default_borrower_may_share_but_perf_ex_remains_a_hard_fence() {
    let _prefixes = LockPrefixesGuard::new();
    let build = protocol::ClaimSet::with_permits(
        std::iter::empty(),
        [41usize],
        [8usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Exclusive,
    )
    .with_admission_class(protocol::AdmissionClass::Build);
    let coordinator =
        match protocol::register_ticket_or_acquire(build.clone(), build.clone(), None, |_| {
            Ok::<Option<()>, anyhow::Error>(None)
        })
        .expect("register live build")
        {
            protocol::TicketWork::Coordinator(coordinator) => coordinator,
            protocol::TicketWork::Acquired(_) => panic!("fresh registry must elect a coordinator"),
        };

    let default = protocol::ClaimSet::with_permits(
        std::iter::empty(),
        [41usize],
        [7usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Exclusive,
    )
    .with_admission_class(protocol::AdmissionClass::DefaultBorrow);
    assert!(
        !protocol::registered_claim_conflicts(&default)
            .expect("query compatible default-borrow claim"),
        "build demand is a placement preference, not a hard fence for default CPU-SH work",
    );
    let default_held = protocol::publish_acquired(&default, Vec::<std::os::fd::OwnedFd>::new())
        .expect("publish compatible default borrower");
    let snapshot =
        protocol::registered_claim_snapshot(&build).expect("read aggregate with live build claim");
    assert!(
        snapshot
            .cpu_build_claimed(41)
            .expect("query build CPU count"),
        "preparation placement must observe CPUs used by live build claims",
    );

    let performance = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [41usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Exclusive,
    );
    assert!(
        protocol::registered_claim_conflicts(&performance).expect("query performance CPU-EX claim"),
        "performance CPU-EX must remain incompatible with cooperative/build CPU-SH",
    );
    drop(default_held);
    drop(coordinator);
}

#[test]
fn fixed_registry_handles_five_hundred_waiters_without_per_waiter_watchers() {
    let _prefixes = LockPrefixesGuard::new();
    assert_eq!(
        protocol::exercise_registry_high_water_for_tests(500)
            .expect("exercise fixed-record registry"),
        500,
        "all records must remain independently addressable across eight chunks",
    );
    assert!(
        protocol::ticket_registry_snapshot_for_tests()
            .expect("empty registry after stress")
            .is_empty(),
        "normal teardown must recycle every high-water slot",
    );
}

#[test]
fn draining_granted_callbacks_does_not_rescan_for_a_nonexistent_waiter() {
    let _prefixes = LockPrefixesGuard::new();
    let election_reads = protocol::exercise_granted_only_drain_election_reads_for_tests(500)
        .expect("drain a callback-only active list");
    assert_eq!(
        election_reads, 0,
        "removing GRANTED callbacks cannot create WAITING work and must not rescan the shrinking list",
    );
}

#[test]
fn candidate_beyond_registry_capacity_never_enters_fast_probe() {
    let _prefixes = LockPrefixesGuard::new();
    protocol::exercise_registry_high_water_for_tests(1).expect("initialize registry layout");
    let candidate = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize << 19],
        crate::flock::FlockMode::Exclusive,
    );
    let ran = std::cell::Cell::new(false);
    let error = protocol::with_registry_fence(&candidate, || {
        ran.set(true);
        Ok(())
    })
    .err()
    .expect("unrepresentable candidate must fail closed");
    assert!(
        error.to_string().contains("capacity"),
        "capacity failure must be diagnostic: {error:#}",
    );
    assert!(
        !ran.get(),
        "an unrepresentable candidate must never enter the fast-path closure",
    );
}

#[test]
fn oversized_first_candidate_never_enters_fast_probe() {
    let _prefixes = LockPrefixesGuard::new();
    let candidate = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize << 20],
        crate::flock::FlockMode::Exclusive,
    );
    let ran = std::cell::Cell::new(false);
    let error = protocol::with_registry_fence(&candidate, || {
        ran.set(true);
        Ok(())
    })
    .err()
    .expect("fresh unrepresentable candidate must fail closed");
    assert!(
        error.to_string().contains("capacity"),
        "capacity failure must be diagnostic: {error:#}",
    );
    assert!(
        !ran.get(),
        "a fresh unrepresentable candidate must never enter the fast-path closure",
    );
}

#[test]
fn active_record_cannot_be_reused_through_a_malformed_free_head() {
    let _prefixes = LockPrefixesGuard::new();
    let claim = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    let coordinator = match protocol::register_ticket_or_acquire(claim.clone(), claim, None, |_| {
        Ok::<Option<()>, anyhow::Error>(None)
    })
    .expect("register active ticket")
    {
        protocol::TicketWork::Coordinator(coordinator) => coordinator,
        protocol::TicketWork::Acquired(_) => panic!("fresh registry must elect a coordinator"),
    };
    protocol::active_free_head_is_rejected_for_tests()
        .expect("active free-list head must fail closed without overwriting its record");
    assert_eq!(
        protocol::ticket_registry_snapshot_for_tests()
            .expect("registry after malformed free-list probe")
            .len(),
        1,
        "the active record must remain intact",
    );
    drop(coordinator);
}

#[test]
fn one_aggregate_snapshot_filters_many_candidates_with_one_registry_read() {
    let _prefixes = LockPrefixesGuard::new();
    let claim = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    let coordinator = match protocol::register_ticket_or_acquire(claim.clone(), claim, None, |_| {
        Ok::<Option<()>, anyhow::Error>(None)
    })
    .expect("register aggregate owner")
    {
        protocol::TicketWork::Coordinator(coordinator) => coordinator,
        protocol::TicketWork::Acquired(_) => panic!("fresh registry must elect a coordinator"),
    };
    let required = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize, 2, 3, 4],
        crate::flock::FlockMode::Exclusive,
    );
    let before = protocol::aggregate_snapshot_read_count_for_tests();
    let snapshot =
        protocol::registered_claim_snapshot(&required).expect("copy aggregate snapshot once");
    for cpu in 1..=4 {
        let candidate = protocol::ClaimSet::new(
            std::iter::empty(),
            [cpu],
            crate::flock::FlockMode::Exclusive,
        );
        assert_eq!(
            snapshot
                .conflicts(&candidate)
                .expect("filter snapshot candidate"),
            cpu == 1,
        );
    }
    assert_eq!(
        protocol::aggregate_snapshot_read_count_for_tests() - before,
        1,
        "N candidate filters must reuse one copied aggregate mapping",
    );
    drop(coordinator);
}

#[test]
fn ordinary_state_reads_overlap_registry_shared_readers() {
    use std::sync::mpsc;

    let _prefixes = LockPrefixesGuard::new();
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let (go_tx, go_rx) = mpsc::sync_channel(1);
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    let reader = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        let claim = protocol::ClaimSet::new(
            std::iter::empty(),
            [1usize],
            crate::flock::FlockMode::Exclusive,
        );
        let coordinator =
            match protocol::register_ticket_or_acquire(claim.clone(), claim, None, |_| {
                Ok::<Option<()>, anyhow::Error>(None)
            })
            .expect("register coordinator")
            {
                protocol::TicketWork::Coordinator(coordinator) => coordinator,
                protocol::TicketWork::Acquired(_) => {
                    panic!("fresh registry must elect a coordinator")
                }
            };
        ready_tx.send(()).expect("report coordinator registration");
        go_rx.recv().expect("receive overlapping-read start");
        let before = protocol::shared_state_read_count_for_tests();
        let result = coordinator.read_state_shared_for_tests();
        let delta = protocol::shared_state_read_count_for_tests() - before;
        result_tx
            .send((result, delta))
            .expect("report overlapping state read");
        drop(coordinator);
    });
    recv_from_service_thread(&ready_rx, "coordinator registration", &reader)
        .expect("coordinator registration must complete");
    let held_reader =
        protocol::hold_registry_shared_for_tests().expect("hold first registry SH reader");
    go_tx.send(()).expect("start overlapping state read");
    let (result, delta) =
        recv_from_service_thread(&result_rx, "overlapping SH registry state read", &reader)
            .expect("a second SH state read must not serialize behind the first");
    result.expect("read coordinator state through readonly mapping");
    assert_eq!(
        delta, 1,
        "the ordinary state path must consume one shared registry read",
    );
    drop(held_reader);
    reader.join().expect("shared state reader");
}

#[test]
fn state_or_wait_reuses_one_retained_shared_mapping() {
    let _prefixes = LockPrefixesGuard::new();
    let claim = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    let mappings_before = protocol::ticket_shared_mapping_build_count_for_tests();
    let coordinator = match protocol::register_ticket_or_acquire(claim.clone(), claim, None, |_| {
        Ok::<Option<()>, anyhow::Error>(None)
    })
    .expect("register retained-map coordinator")
    {
        protocol::TicketWork::Coordinator(coordinator) => coordinator,
        protocol::TicketWork::Acquired(_) => panic!("fresh registry must elect a coordinator"),
    };
    let mappings_after_registration = protocol::ticket_shared_mapping_build_count_for_tests();
    assert_eq!(
        mappings_after_registration - mappings_before,
        1,
        "one live ticket must construct exactly one retained header+chunk view",
    );
    for _ in 0..128 {
        coordinator
            .state_or_wait_for_tests()
            .expect("read coordinator state through retained mappings");
    }
    assert_eq!(
        protocol::ticket_shared_mapping_build_count_for_tests(),
        mappings_after_registration,
        "repeated state_or_wait reads must not reopen or remap registry files",
    );
    drop(coordinator);
}

#[test]
fn writer_publication_is_visible_through_retained_readonly_shared_mapping() {
    let _prefixes = LockPrefixesGuard::new();
    let (state_visible, futex_visible, mapping_reused) =
        protocol::exercise_retained_shared_publication_for_tests()
            .expect("publish through retained shared mappings");
    assert!(
        state_visible,
        "MAP_SHARED state publication was not visible"
    );
    assert!(
        futex_visible,
        "MAP_SHARED futex publication was not visible"
    );
    assert!(
        mapping_reused,
        "observing writer publication unexpectedly rebuilt the ticket mapping",
    );
}

#[test]
fn reused_slot_rejects_an_old_retained_mapping_identity() {
    let _prefixes = LockPrefixesGuard::new();
    let (slot_reused, stale_mapping_rejected) =
        protocol::exercise_retained_mapping_slot_reuse_for_tests()
            .expect("exercise retained mapping across slot reuse");
    assert!(slot_reused, "test did not recycle the retired slot");
    assert!(
        stale_mapping_rejected,
        "old retained mapping aliased the replacement ticket",
    );
}

#[test]
fn targeted_broker_wake_cancels_one_registry_waiter() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};

    let _broker = InterruptibleFlockBrokerGuard::start();
    let _prefixes = LockPrefixesGuard::new();
    let claim = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    let coordinator =
        match protocol::register_ticket_or_acquire(claim.clone(), claim.clone(), None, |_| {
            Ok::<Option<()>, anyhow::Error>(None)
        })
        .expect("register coordinator")
        {
            protocol::TicketWork::Coordinator(coordinator) => coordinator,
            protocol::TicketWork::Acquired(_) => panic!("fresh registry must elect a coordinator"),
        };

    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let worker_claim = claim.clone();
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    let waiter = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = cpu_prefix);
        let result = protocol::register_ticket_or_acquire(
            worker_claim.clone(),
            worker_claim,
            Some(&worker_cancelled),
            |_| Ok::<Option<()>, anyhow::Error>(None),
        )
        .map(|_| ());
        result_tx.send(result).expect("report waiter result");
    });

    let waiter_id = wait_with_task_service(
        "registry futex cancellation-generation publication",
        std::slice::from_ref(&waiter.service),
        || {
            let id = crate::flock::interruptible_flock_waiter_id();
            Ok((id != 0).then_some(id))
        },
    )
    .expect("registry futex waiter must publish its cancellation generation");
    cancelled.store(true, Ordering::Release);
    let signal_count = crate::flock::primitives::interruptible_flock_broker_signal_count();
    crate::flock::wake_interruptible_flock_waiter(waiter_id);
    let result = match wait_for_broker_signal_or_waiter_completion(
        signal_count,
        &result_rx,
        &waiter.service,
    )
    .expect("targeted wake or the cancellation it races must make progress")
    {
        BrokerWakeOrCompletion::BrokerSignalled => {
            recv_from_service_thread(&result_rx, "targeted registry-wait cancellation", &waiter)
                .expect("targeted wake must cancel the registry wait")
        }
        BrokerWakeOrCompletion::WaiterCompleted(result) => result,
    };
    assert!(
        result
            .expect_err("cancelled registry wait must not acquire")
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::Interrupted),
        "registry cancellation must surface ErrorKind::Interrupted",
    );
    waiter.join().expect("registry futex waiter");
    drop(coordinator);
    assert_eq!(
        crate::flock::interruptible_flock_waiter_id(),
        0,
        "registry waiter generation must be torn down on cancellation",
    );
}

#[test]
fn cancelled_waiter_can_unregister_before_broker_handoff() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};

    let _broker = InterruptibleFlockBrokerGuard::start();
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let (registered_tx, registered_rx) = mpsc::sync_channel(1);
    let (observe_tx, observe_rx) = mpsc::sync_channel(1);
    let (completed_tx, completed_rx) = mpsc::sync_channel(1);
    let worker = TestServiceThread::spawn(move || {
        let registration =
            crate::flock::InterruptibleFlockWaiter::register().expect("register waiter");
        let waiter_id = crate::flock::interruptible_flock_waiter_id();
        assert_ne!(waiter_id, 0, "registration must publish its generation");
        registered_tx
            .send(waiter_id)
            .expect("publish waiter generation");
        observe_rx
            .recv()
            .expect("observe cancellation before unregistering");
        assert!(
            worker_cancelled.load(Ordering::Acquire),
            "worker must observe the published cancellation",
        );
        drop(registration);
        completed_tx
            .send(())
            .expect("publish cancellation completion");
    });

    let waiter_id =
        recv_from_service_thread(&registered_rx, "waiter generation publication", &worker)
            .expect("waiter generation");
    let signal_count = crate::flock::primitives::interruptible_flock_broker_signal_count();
    cancelled.store(true, Ordering::Release);
    observe_tx.send(()).expect("release cancelled waiter");
    recv_from_service_thread(
        &completed_rx,
        "cancellation completion before broker handoff",
        &worker,
    )
    .expect("cancelled waiter must unregister");
    assert_eq!(
        crate::flock::interruptible_flock_waiter_id(),
        0,
        "completed cancellation must tear down its waiter generation",
    );

    crate::flock::wake_interruptible_flock_waiter(waiter_id);
    assert_eq!(
        crate::flock::primitives::interruptible_flock_broker_signal_count(),
        signal_count,
        "a wake requested after valid cancellation completion must not signal \
         a dead waiter generation",
    );
    worker.join().expect("cancelled waiter worker");
}

#[test]
fn stale_broker_generation_cannot_interrupt_the_next_registration() {
    use std::sync::mpsc;

    let _broker = InterruptibleFlockBrokerGuard::start();
    let _prefixes = LockPrefixesGuard::new();
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let (first_id_tx, first_id_rx) = mpsc::sync_channel(1);
    let (advance_tx, advance_rx) = mpsc::sync_channel(1);
    let (second_id_tx, second_id_rx) = mpsc::sync_channel(1);
    let (poll_tx, poll_rx) = mpsc::sync_channel(1);
    let worker = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = cpu_prefix);

        let wake_signal = libc::SIGRTMIN() + 4;
        let mut wake_set: libc::sigset_t = unsafe { std::mem::zeroed() };
        unsafe {
            libc::sigemptyset(&mut wake_set);
            libc::sigaddset(&mut wake_set, wake_signal);
            assert_eq!(
                libc::pthread_sigmask(libc::SIG_BLOCK, &wake_set, std::ptr::null_mut()),
                0,
                "block private RT wake before first registration",
            );
        }

        let first = crate::flock::InterruptibleFlockWaiter::register().expect("first registration");
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
                "read restored signal mask",
            );
            assert_eq!(
                libc::sigismember(&restored_mask, wake_signal),
                1,
                "registration drop must restore the originally blocked RT wake",
            );
        }

        let second =
            crate::flock::InterruptibleFlockWaiter::register().expect("second registration");
        second_id_tx
            .send(crate::flock::interruptible_flock_waiter_id())
            .expect("report second generation");
        let result = protocol::LockDirWatch::new_real_for_tests().and_then(|watch| {
            watch.wait(
                std::time::Duration::from_millis(150),
                &protocol::ClaimSet::default(),
            )
        });
        poll_tx.send(result).expect("report bounded poll");
        drop(second);
    });

    let first_id = recv_from_service_thread(&first_id_rx, "first waiter generation", &worker)
        .expect("first generation");
    let signal_count = crate::flock::primitives::interruptible_flock_broker_signal_count();
    crate::flock::wake_interruptible_flock_waiter(first_id);
    wait_for_broker_signal_after(signal_count);
    advance_tx.send(()).expect("advance worker");
    let second_id = recv_from_service_thread(&second_id_rx, "second waiter generation", &worker)
        .expect("second generation");
    assert_ne!(
        first_id, second_id,
        "successive registrations need distinct generations"
    );
    assert!(
        recv_from_service_thread(&poll_rx, "second-generation bounded poll", &worker)
            .expect("second generation poll")
            .is_ok(),
        "an old broker retry must not interrupt a later registration",
    );
    worker.join().expect("same-thread generation worker");
}

#[test]
fn interruptible_broker_stops_during_retry_and_restarts() {
    use std::sync::mpsc;

    let _broker = InterruptibleFlockBrokerGuard::start();
    let (id_tx, id_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let worker = TestServiceThread::spawn(move || {
        let registration =
            crate::flock::InterruptibleFlockWaiter::register().expect("live registration");
        id_tx
            .send(crate::flock::interruptible_flock_waiter_id())
            .expect("report generation");
        release_rx.recv().expect("release registration");
        drop(registration);
    });
    let waiter_id = recv_from_service_thread(&id_rx, "live waiter generation", &worker)
        .expect("live registration generation");
    let signal_count = crate::flock::primitives::interruptible_flock_broker_signal_count();
    crate::flock::wake_interruptible_flock_waiter(waiter_id);
    wait_for_broker_signal_after(signal_count);
    crate::flock::stop_interruptible_flock_broker();
    release_tx.send(()).expect("release registration");
    worker.join().expect("registration worker");
    crate::flock::start_interruptible_flock_broker()
        .expect("broker must restart after a complete stop");
    crate::flock::stop_interruptible_flock_broker();
}

/// Exact default placement is opportunistic. Shared CPU holders defeat every
/// CPU-EX candidate, but the run must immediately take the same CPU-SH pool;
/// multiple default fallbacks may overlap it. Once peers leave, free admission
/// returns to exact 1:1 pinning.
#[test]
fn default_exact_is_best_effort_and_shared_fallbacks_overlap() {
    let _prefixes = LockPrefixesGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1]);
    let host = HostTopology::new_for_tests(&[(vec![0, 1], 0)]);
    let topo = crate::vmm::Topology::new(1, 1, 1, 1);

    let peer0 = crate::flock::try_flock(cpu_lock_path(0), crate::flock::FlockMode::Shared)
        .unwrap()
        .expect("SH cpu0");
    let peer1 = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Shared)
        .unwrap()
        .expect("SH cpu1");
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let first_host = host.clone();
    let (first_tx, first_rx) = std::sync::mpsc::sync_channel(1);
    let first_worker = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        ALLOWED_CPUS_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(vec![0, 1]));
        let result = crate::vmm::KtstrVm::acquire_default_preferred_run_locks(
            Some(&first_host),
            &topo,
            false,
            None,
            None,
            256,
        );
        let _ = first_tx.send(result);
    });
    let first = recv_from_service_thread(
        &first_rx,
        "non-waiting default shared fallback",
        &first_worker,
    )
    .expect("default fallback worker must publish")
    .expect("SH peers permit immediate shared fallback");
    first_worker.join().expect("default fallback worker");
    let mut first_mask = first
        .shared_cpu_mask
        .clone()
        .expect("fallback retains its admitted CPU set");
    first_mask.sort_unstable();
    assert_eq!(first_mask, vec![0, 1]);
    assert!(first.pinning_plan.is_none());

    let second = crate::vmm::KtstrVm::acquire_default_preferred_run_locks(
        Some(&host),
        &topo,
        false,
        None,
        None,
        256,
    )
    .expect("a second default fallback may overlap the first");
    let mut second_mask = second
        .shared_cpu_mask
        .clone()
        .expect("second fallback retains its admitted CPU set");
    second_mask.sort_unstable();
    assert_eq!(second_mask, first_mask);
    assert!(second.pinning_plan.is_none());

    drop(second);
    drop(first);
    drop(peer1);
    drop(peer0);
    let exact = crate::vmm::KtstrVm::acquire_default_preferred_run_locks(
        Some(&host),
        &topo,
        false,
        None,
        None,
        256,
    )
    .expect("free default placement");
    assert!(exact.shared_cpu_mask.is_none());
    assert!(
        exact.pinning_plan.is_some(),
        "free admission prefers exact 1:1"
    );

    let fallback_peer = crate::vmm::KtstrVm::acquire_default_preferred_run_locks(
        Some(&host),
        &topo,
        false,
        None,
        None,
        256,
    )
    .expect("a second default may share the first exact run's service footprint");
    assert!(
        fallback_peer.pinning_plan.is_none(),
        "the first exact run owns CPU-SH service headroom, so another default must fall back",
    );
    let mut fallback_mask = fallback_peer
        .shared_cpu_mask
        .clone()
        .expect("fallback retains its admitted CPU set");
    fallback_mask.sort_unstable();
    assert_eq!(fallback_mask, vec![0, 1]);

    let overlapping = crate::vmm::KtstrVm::acquire_default_preferred_run_locks(
        Some(&host),
        &topo,
        false,
        None,
        None,
        256,
    )
    .expect("exact defaults retain SH and permit a shared default peer");
    assert!(
        overlapping.pinning_plan.is_none(),
        "once exact capacity is occupied, default must fall back without waiting",
    );
    let mut overlapping_mask = overlapping
        .shared_cpu_mask
        .clone()
        .expect("overlapping fallback retains its admitted CPU set");
    overlapping_mask.sort_unstable();
    assert_eq!(overlapping_mask, vec![0, 1]);
    drop(overlapping);

    let exact_cpu = exact
        .pinning_plan
        .as_ref()
        .and_then(|plan| plan.assignments.first())
        .map(|(_, cpu)| *cpu)
        .expect("exact plan has one vCPU assignment");
    let build = try_acquire_resources(&[0], LlcLockMode::Shared, &[exact_cpu], FlockMode::Shared)
        .expect("build-style SH probe against exact default");
    assert!(
        matches!(&build, TryAcquireAll::Acquired(_)),
        "an exact-pinned default must not block overlapping build/no-perf SH",
    );
    drop(build);
    assert!(matches!(
        try_acquire_resources(
            &[0],
            LlcLockMode::Shared,
            &[exact_cpu],
            FlockMode::Exclusive,
        )
        .expect("perf-grain probe against exact default"),
        TryAcquireAll::Contended { .. },
    ));
}

#[test]
fn default_shared_fallback_uses_disjoint_capacity_but_never_perf_ex() {
    let _prefixes = LockPrefixesGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3, 4, 5]);
    let host = HostTopology::new_for_tests(&[(vec![0, 1], 0), (vec![2, 3], 0), (vec![4, 5], 0)]);
    let topo = crate::vmm::Topology::new(1, 1, 1, 1);
    let perf = try_acquire_resources(&[0], LlcLockMode::Exclusive, &[0, 1], FlockMode::Exclusive)
        .expect("acquire perf EX");
    let perf = match perf {
        TryAcquireAll::Acquired(locks) => locks,
        TryAcquireAll::Contended { reason, .. } => panic!("fresh perf EX contended: {reason}"),
    };
    let shared_peers = [2usize, 3, 4, 5]
        .into_iter()
        .map(|cpu| {
            crate::flock::try_flock(cpu_lock_path(cpu), crate::flock::FlockMode::Shared)
                .unwrap()
                .expect("SH exact-pin blocker")
        })
        .collect::<Vec<_>>();

    let admitted = crate::vmm::KtstrVm::acquire_default_preferred_run_locks(
        Some(&host),
        &topo,
        false,
        None,
        None,
        256,
    )
    .expect("disjoint shared capacity must remain usable");
    let mask = admitted
        .shared_cpu_mask
        .as_deref()
        .expect("SH peers force default fallback");
    assert!(mask.iter().all(|cpu| ![0, 1].contains(cpu)));
    assert_eq!(mask.len(), 2);
    drop(admitted);
    drop(shared_peers);
    drop(perf);
}

#[test]
fn default_hard_perf_contention_is_no_wait_and_cancellable() {
    use std::sync::mpsc;
    use std::sync::{Arc, atomic::AtomicBool};

    let _broker = InterruptibleFlockBrokerGuard::start();
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let _allowed = AllowedCpusGuard::new(vec![0, 1]);
    let host = HostTopology::new_for_tests(&[(vec![0, 1], 0)]);
    let topo = crate::vmm::Topology::new(1, 1, 1, 1);
    let perf = try_acquire_resources(&[0], LlcLockMode::Exclusive, &[0, 1], FlockMode::Exclusive)
        .expect("acquire perf EX");
    let perf = match perf {
        TryAcquireAll::Acquired(locks) => locks,
        TryAcquireAll::Contended { reason, .. } => panic!("fresh perf EX contended: {reason}"),
    };

    let error = match crate::vmm::KtstrVm::acquire_default_preferred_run_locks(
        Some(&host),
        &topo,
        false,
        None,
        None,
        256,
    ) {
        Ok(_) => panic!("interactive/no-wait default must surface hard perf contention"),
        Err(error) => error,
    };
    assert!(error.downcast_ref::<ResourceContention>().is_some());

    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    let worker = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        ALLOWED_CPUS_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(vec![0, 1]));
        let result = crate::vmm::KtstrVm::acquire_default_preferred_run_locks(
            Some(&host),
            &topo,
            true,
            Some(&worker_cancelled),
            None,
            256,
        );
        ALLOWED_CPUS_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        let _ = result_tx.send(result);
    });
    wait_with_task_service(
        "default waiter cancellation-generation publication",
        std::slice::from_ref(&worker.service),
        || Ok((crate::flock::interruptible_flock_waiter_id() != 0).then_some(())),
    )
    .expect("default waiter must publish its cancellation generation");
    let broker_service = interruptible_flock_broker_service();
    cancel_registry_worker(&cancelled);
    // Cancellation completion is authoritative. If the worker observes the
    // flag before entering its blocking wait, it can unregister before the
    // broker sees a live generation and no targeted signal is expected. The
    // dedicated broker test above separately requires signal delivery while a
    // waiter is known to remain blocked. Either the worker consumes the flag
    // directly or the broker wakes it; account both causal producers
    // independently so their service does not amplify one shared budget.
    let result = recv_with_task_service(
        &result_rx,
        "cancelled default waiter unwind",
        &[worker.service.clone(), broker_service],
    )
    .expect("cancelled default waiter must unwind");
    worker.join().expect("cancelled default waiter");
    let error = match result {
        Ok(_) => panic!("cancelled admission must not acquire through perf EX"),
        Err(error) => error,
    };
    assert!(
        error
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::Interrupted),
        "unexpected cancellation error: {error:#}",
    );
    drop(perf);
}

#[test]
fn bare_interactive_preparation_saturation_is_immediate_resource_contention() {
    let _prefixes = LockPrefixesGuard::new();
    let token_locks = preparation_token_range()
        .expect("resolve preparation token namespace")
        .map(|permit| {
            crate::flock::try_flock(permit_lock_path(permit), crate::flock::FlockMode::Exclusive)
                .expect("probe preparation token")
                .expect("hold every preparation token")
        })
        .collect::<Vec<_>>();
    assert!(!token_locks.is_empty());

    // Lock-path overrides are thread-local. Point the worker at the saturated
    // pool, then require its no-wait operation to complete within delivered
    // CPU service rather than sleeping until one of these tokens is released.
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
    let worker = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        let result = crate::vmm::KtstrVm::register_interactive_pending_admission(false);
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        let observation = match result {
            Ok(pending) => {
                drop(pending);
                (
                    false,
                    "interactive preparation unexpectedly acquired".into(),
                )
            }
            Err(error) => (
                error.downcast_ref::<ResourceContention>().is_some(),
                error.to_string(),
            ),
        };
        let _ = result_tx.send(observation);
    });
    let observed = recv_from_service_thread(
        &result_rx,
        "nonblocking interactive preparation saturation",
        &worker,
    );

    // Always unblock and join a regressed blocking implementation before an
    // assertion can unwind the isolated lock-directory guard.
    drop(token_locks);
    worker.join().expect("interactive preparation worker");
    let (is_contention, diagnostic) =
        observed.expect("interactive preparation must not sleep behind saturated tokens");
    assert!(
        is_contention,
        "saturation must surface ResourceContention: {diagnostic}",
    );
    assert!(
        diagnostic.contains("CPU/memory preparation admission is busy"),
        "unexpected interactive contention diagnostic: {diagnostic}",
    );
}

#[test]
fn bare_interactive_preparation_does_not_wait_for_registry_writer() {
    let _prefixes = LockPrefixesGuard::new();
    let registry = protocol::hold_registry_exclusive_for_tests()
        .expect("hold the admission registry writer lock");
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
    let worker = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        let result = crate::vmm::KtstrVm::register_interactive_pending_admission(false);
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        let _ = result_tx.send(match result {
            Ok(pending) => {
                drop(pending);
                false
            }
            Err(error) => error.downcast_ref::<ResourceContention>().is_some(),
        });
    });
    let observed = recv_from_service_thread(
        &result_rx,
        "nonblocking interactive registry publication",
        &worker,
    );
    drop(registry);
    worker.join().expect("interactive registry worker");
    assert!(
        observed.expect("interactive registry contention must not wait"),
        "busy registry writer must surface immediate ResourceContention",
    );
}

#[test]
fn interactive_exec_preparation_waits_for_capacity_then_acquires() {
    let _prefixes = LockPrefixesGuard::new();
    let token_locks = preparation_token_range()
        .expect("resolve preparation token namespace")
        .map(|permit| {
            crate::flock::try_flock(permit_lock_path(permit), crate::flock::FlockMode::Exclusive)
                .expect("probe preparation token")
                .expect("hold every preparation token")
        })
        .collect::<Vec<_>>();
    assert!(!token_locks.is_empty());

    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
    let worker = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        let result =
            crate::vmm::KtstrVm::register_interactive_pending_admission(true).map(|pending| {
                // Pending admission is intentionally thread-affine. Retire it
                // on its owner and send only the semantic result to the
                // observing test thread.
                drop(pending);
            });
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        let _ = result_tx.send(result);
    });

    wait_with_task_service(
        "interactive exec parks on preparation capacity",
        std::slice::from_ref(&worker.service),
        || {
            let snapshot = worker.service.snapshot();
            let parked =
                !snapshot.pending && snapshot.tasks.values().any(|sample| sample.state == b'S');
            match result_rx.try_recv() {
                Ok(_) => {
                    anyhow::bail!(
                        "interactive exec returned while every preparation token was held"
                    )
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    anyhow::bail!("interactive exec worker exited before releasing capacity")
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            Ok(parked.then_some(()))
        },
    )
    .expect("observe interactive exec waiting in the kernel");

    drop(token_locks);
    recv_from_service_thread(
        &result_rx,
        "interactive exec acquires released preparation capacity",
        &worker,
    )
    .expect("interactive exec must wake after capacity release")
    .expect("interactive exec preparation admission");
    worker.join().expect("interactive exec preparation worker");
}

/// The heavyweight immutable-image phase must not hide a cell's eventual run
/// footprint. Publish that footprint first, park while every preparation token
/// is occupied, then require the very same ticket to become PENDING once one
/// tuple can be acquired. PENDING must claim only that physical tuple: ticket
/// order retains the selected intent without idling its other final CPUs during
/// cold immutable preparation.
#[test]
fn early_final_intent_is_visible_before_preparation_and_transitions_same_ticket_to_pending() {
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let allowed = host_allowed_cpus();
    let selected_cpus = allowed.iter().copied().take(2).collect::<Vec<_>>();
    assert_eq!(
        selected_cpus.len(),
        2,
        "selected-intent preparation test needs two allowed CPUs",
    );
    let final_claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        selected_cpus.iter().copied(),
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Exclusive,
    );
    // A disjoint live predecessor gives the selected ticket a real predecessor
    // prefix throughout the transition. Its LLC-only footprint cannot
    // interfere with either the selected CPU or preparation permits.
    let _anchor = protocol::register_pending_claim_for_tests(protocol::ClaimSet::with_modes(
        [0usize],
        std::iter::empty(),
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    ))
    .expect("publish a disjoint predecessor");
    let preparation_tokens = preparation_token_range()
        .expect("resolve preparation token namespace")
        .collect::<Vec<_>>();
    let token_locks = preparation_tokens
        .iter()
        .map(|&permit| {
            crate::flock::try_flock(permit_lock_path(permit), crate::flock::FlockMode::Exclusive)
                .expect("probe preparation token")
                .expect("hold every preparation token")
        })
        .collect::<Vec<_>>();
    assert!(!token_locks.is_empty());

    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let planner_steps = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let worker_steps = std::sync::Arc::clone(&planner_steps);
    let worker_claim = final_claim.clone();
    let (prepared_tx, prepared_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
    let worker = TestServiceThread::spawn(move || -> anyhow::Result<()> {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        let granted_claim = worker_claim.clone();
        let coordinator_claim = worker_claim.clone();
        let granted_steps = std::sync::Arc::clone(&worker_steps);
        let coordinator_steps = worker_steps;
        let mut pending = protocol::register_intent_for_preparation(
            worker_claim.clone(),
            worker_claim,
            move |_| {
                granted_steps.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(Some(granted_claim.clone()))
            },
            move |_| {
                coordinator_steps.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(Some(coordinator_claim.clone()))
            },
        )?;
        let ticket = pending.exec_handoff_parts()?.1;
        let (prepared_cpu, _, _) = pending.preparation_affinity_handoff_parts()?;
        let permits = pending
            .preparation_handoff_parts()?
            .1
            .into_iter()
            .map(|(permit, _)| permit)
            .collect::<Vec<_>>();
        let (published_claim, published_watch) = pending.pending_claim_watch_for_tests()?;
        prepared_tx
            .send((
                ticket,
                prepared_cpu,
                permits,
                published_claim,
                published_watch,
            ))
            .map_err(|_| anyhow::anyhow!("prepared-observation receiver disappeared"))?;
        release_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("pending-release sender disappeared"))?;
        pending.restore_preparation_affinity()?;
        drop(pending);
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        Ok(())
    });

    let early_ticket = wait_with_task_service(
        "early final intent publication before preparation",
        std::slice::from_ref(&worker.service),
        || {
            let snapshot = protocol::ticket_registry_snapshot_for_tests()?;
            Ok(snapshot
                .into_iter()
                .find(|(_, pid, claim)| *pid == std::process::id() && *claim == final_claim)
                .map(|(ticket, _, _)| ticket))
        },
    )
    .expect("the final run intent must be visible while preparation is saturated");
    wait_with_task_service(
        "early intent parks after one preparation-capacity miss",
        std::slice::from_ref(&worker.service),
        || {
            let sample = worker.service.snapshot();
            let parked = planner_steps.load(std::sync::atomic::Ordering::Relaxed) > 0
                && !sample.pending
                && sample.tasks.values().any(|sample| sample.state == b'S');
            Ok(parked.then_some(()))
        },
    )
    .expect("a preparation miss must park instead of hot-regranting the intent");

    // Publish a conflicting *later* ticket after the selected intent has
    // parked.  The final->PENDING commit is ordered only by the selected
    // ticket's predecessor prefix; consulting the full aggregate here lets a
    // later arrival veto an already-selected ticket and collapses a storm.
    let later_preparation_claim = protocol::ClaimSet::with_permits(
        std::iter::empty(),
        std::iter::empty(),
        preparation_tokens,
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Exclusive,
    );
    let _later = protocol::register_pending_claim_for_tests(later_preparation_claim)
        .expect("publish a later conflicting intent");

    drop(token_locks);
    let (pending_ticket, prepared_cpu, preparation_permits, published_claim, published_watch) =
        recv_from_service_thread(
            &prepared_rx,
            "same-ticket transition into PENDING preparation",
            &worker,
        )
        .expect("released preparation capacity must wake the selected intent");
    assert_eq!(
        pending_ticket, early_ticket,
        "PENDING must reuse the intent ticket"
    );
    assert!(selected_cpus.contains(&prepared_cpu));
    let snapshot = protocol::ticket_registry_snapshot_for_tests()
        .expect("snapshot same-ticket PENDING transition");
    let pending_claim = snapshot
        .iter()
        .find(|(ticket, _, _)| *ticket == early_ticket)
        .map(|(_, _, claim)| claim)
        .expect("same ticket must remain published after preparation acquisition");
    assert_eq!(pending_claim, &published_claim);
    assert_eq!(pending_claim.cpus, [prepared_cpu].into_iter().collect());
    assert_eq!(pending_claim.cpu_mode, protocol::ClaimMode::Shared);
    assert!(
        preparation_permits
            .iter()
            .all(|permit| pending_claim.permits.contains(permit)),
        "PENDING claim must publish every physically held preparation permit",
    );
    assert!(final_claim.cpus.is_subset(&published_watch.cpus));
    assert_eq!(published_watch.cpu_mode, protocol::ClaimMode::Exclusive);
    assert!(pending_claim.cpus.is_subset(&published_watch.cpus));
    assert!(pending_claim.permits.is_subset(&published_watch.permits));
    let unprepared_cpu = selected_cpus
        .iter()
        .copied()
        .find(|cpu| *cpu != prepared_cpu)
        .expect("two-CPU selection has one non-preparation CPU");
    let unprepared_final = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [unprepared_cpu],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Exclusive,
    );
    assert!(
        !protocol::registered_claim_conflicts(&unprepared_final)
            .expect("probe unowned selected-final CPU during PENDING"),
        "PENDING must not sequester selected final resources without physical owners",
    );

    release_tx
        .send(())
        .expect("release the worker-owned pending admission");
    worker
        .join()
        .expect("early-intent preparation worker")
        .expect("complete early-intent preparation worker");
}

/// Both performance's CPU-EX intent and default's cooperative CPU-SH intent
/// must reach the same selected-preparation path.  In particular, an EX intent
/// cannot reject its own CPU when the preparation phase asks for a temporary
/// SH affinity owner.
#[test]
fn selected_performance_and_default_intents_share_preparation_affinity_path() {
    let _prefixes = LockPrefixesGuard::new();
    let allowed = host_allowed_cpus();
    let affinity_cpu = *allowed
        .first()
        .expect("test process must have an allowed host CPU");
    let disjoint_cpu = *allowed
        .iter()
        .find(|&&cpu| cpu != affinity_cpu)
        .expect("selected-preparation throughput test needs two allowed CPUs");
    let _anchor = protocol::register_pending_claim_for_tests(protocol::ClaimSet::with_modes(
        [0usize],
        std::iter::empty(),
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    ))
    .expect("publish a disjoint predecessor");

    for cpu_mode in [
        crate::flock::FlockMode::Exclusive,
        crate::flock::FlockMode::Shared,
    ] {
        let admission_class = if cpu_mode == crate::flock::FlockMode::Shared {
            protocol::AdmissionClass::DefaultBorrow
        } else {
            protocol::AdmissionClass::Ordinary
        };
        let claim = protocol::ClaimSet::with_modes(
            std::iter::empty(),
            [affinity_cpu],
            crate::flock::FlockMode::Shared,
            cpu_mode,
        )
        .with_admission_class(admission_class);
        let granted_claim = claim.clone();
        let coordinator_claim = claim.clone();
        let mut pending = protocol::register_intent_for_preparation(
            claim.clone(),
            claim.clone(),
            move |_| Ok(Some(granted_claim.clone())),
            move |_| Ok(Some(coordinator_claim.clone())),
        )
        .unwrap_or_else(|error| {
            panic!("{cpu_mode:?} early intent failed selected preparation: {error:#}")
        });
        assert_eq!(
            pending
                .preparation_affinity_handoff_parts()
                .expect("inspect selected preparation affinity")
                .0,
            affinity_cpu,
        );
        let ticket = pending
            .exec_handoff_parts()
            .expect("inspect selected preparation ticket")
            .1;
        let (_, pending_watch) = pending
            .pending_claim_watch_for_tests()
            .expect("inspect selected intent watch");
        let pending_claim = protocol::ticket_registry_snapshot_for_tests()
            .expect("snapshot selected preparation")
            .into_iter()
            .find(|(candidate, _, _)| *candidate == ticket)
            .map(|(_, _, claim)| claim)
            .expect("selected preparation remains published");
        assert_eq!(pending_claim.cpu_mode, protocol::ClaimMode::Shared);
        assert_eq!(pending_claim.cpus, [affinity_cpu].into_iter().collect());
        assert!(claim.cpus.is_subset(&pending_watch.cpus));
        assert_eq!(pending_watch.cpu_mode, cpu_mode.into());
        assert_eq!(pending_watch.admission_class, admission_class);
        assert_eq!(
            protocol::registered_claim_conflicts(&claim)
                .expect("probe selected-final sharing semantics"),
            cpu_mode == crate::flock::FlockMode::Exclusive,
            "the physical preparation CPU-SH owner must fence performance EX but permit default SH",
        );
        let disjoint_ex = protocol::ClaimSet::with_modes(
            std::iter::empty(),
            [disjoint_cpu],
            crate::flock::FlockMode::Shared,
            crate::flock::FlockMode::Exclusive,
        );
        let ran = std::cell::Cell::new(false);
        assert!(matches!(
            protocol::with_registry_fence(&disjoint_ex, || {
                ran.set(true);
                Ok::<_, anyhow::Error>(())
            })
            .expect("fence a disjoint EX probe during selected preparation"),
            protocol::RegistryFence::Ran { .. },
        ));
        assert!(
            ran.get(),
            "PENDING preparation must not blanket-block disjoint CPU-EX throughput",
        );
        pending
            .restore_preparation_affinity()
            .expect("restore selected preparation affinity");
        drop(pending);
        assert_eq!(
            protocol::ticket_registry_snapshot_for_tests()
                .expect("snapshot retired selected preparation")
                .len(),
            1,
            "retiring one selected preparation must leave only the anchor",
        );
    }
}

#[test]
fn selected_preparation_affinity_intersects_process_cpuset_without_shrinking_final_claim() {
    let selected = [2usize, 3, 8, 9]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        preparation_affinity_candidates(&selected, &[3, 7, 9]),
        vec![3, 9]
    );
    assert_eq!(
        selected,
        [2usize, 3, 8, 9]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>(),
        "preparation affinity filtering must not mutate the selected final footprint",
    );
    assert!(preparation_affinity_candidates(&selected, &[0, 1]).is_empty());
}

#[test]
fn interactive_preparation_tries_a_second_tuple_after_candidate_rejection() {
    let _prefixes = LockPrefixesGuard::new();
    assert!(
        preparation_token_range()
            .expect("resolve preparation tokens")
            .len()
            >= 2,
        "test host must fund two preparation candidates",
    );
    let mut attempted_tokens = Vec::new();
    let allowed = host_allowed_cpus();
    let selected = try_preparation_candidates_once(0, &allowed, |preparation, _claim| {
        attempted_tokens.push(preparation.token_permit);
        if attempted_tokens.len() == 1 {
            drop(preparation);
            Ok(PreparationCandidateDecision::Retry)
        } else {
            Ok(PreparationCandidateDecision::Accepted(
                preparation.token_permit,
            ))
        }
    })
    .expect("scan preparation candidates");
    let PreparationProbe::Acquired(selected) = selected else {
        panic!("second preparation candidate was not acquired");
    };
    assert_eq!(attempted_tokens.len(), 2);
    assert_ne!(attempted_tokens[0], selected);
}

#[test]
fn preparation_sweep_stops_after_one_global_cpu_shortage() {
    let _prefixes = LockPrefixesGuard::new();
    let pool = AdmissionPermitPool::for_host(possible_cpu_width());
    let cpu_permits = pool
        .all()
        .map(|permit| {
            crate::flock::try_flock(permit_lock_path(permit), crate::flock::FlockMode::Exclusive)
                .expect("probe CPU admission permit")
                .expect("hold CPU admission permit")
        })
        .collect::<Vec<_>>();
    assert!(!cpu_permits.is_empty());
    reset_preparation_resource_probe_count_for_tests();
    let allowed = host_allowed_cpus();
    let selected = try_preparation_candidates_once::<()>(0, &allowed, |_preparation, _claim| {
        panic!("global CPU shortage must prevent a complete preparation candidate")
    })
    .expect("probe globally saturated CPU admission");
    assert!(
        !matches!(selected, PreparationProbe::Acquired(())),
        "global CPU shortage unexpectedly acquired preparation",
    );
    assert_eq!(
        preparation_resource_probe_count_for_tests(),
        1,
        "a global resource miss must not be repeated for every free token",
    );
}

/// WORK CONSERVATION: while a big `LOCK_EX` request is the coordinator
/// and hungering for a claimed LLC, a concurrent small `LOCK_SH` cell
/// wanting DIFFERENT capacity completes immediately — it never waits
/// behind the coordinator (no head-of-line blocking for satisfiable-now
/// work). The coordinator's own target stays fenced by its
/// claim (pinned separately above).
#[test]
fn small_shared_cell_proceeds_while_coordinator_hungers() {
    use std::sync::mpsc;
    use std::sync::{Arc, atomic::AtomicBool};

    let _broker = InterruptibleFlockBrokerGuard::start();
    let _prefixes = LockPrefixesGuard::new();
    // Peer wedges LLC 10 with LOCK_SH so an EX coordinator must wait.
    let peer_sh = crate::flock::try_flock(llc_lock_path(10), crate::flock::FlockMode::Shared)
        .unwrap()
        .expect("peer SH");

    // Coordinator: EX request for LLC 10, registered in a helper thread. The
    // lock-prefix overrides are thread-local, so the helper re-installs
    // the same prefixes before acquiring.
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let (coordinator_tx, coordinator_rx) = mpsc::sync_channel(1);
    let coordinator_thread = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = cpu_prefix);
        let result = acquire_resource_locks_waiting_impl(
            &[10usize],
            LlcLockMode::Exclusive,
            &[50],
            FlockMode::Exclusive,
            true,
            Some(&worker_cancelled),
        );
        let _ = coordinator_tx.send(result);
    });
    // The high-level acquisition API deliberately has no test callback in its
    // registration path, so observe the registry state while charging service
    // delivered to the coordinator which must publish it.
    let ready = wait_with_task_service(
        "coordinator exact-claim publication",
        std::slice::from_ref(&coordinator_thread.service),
        || {
            if !protocol::ticket_registry_snapshot_for_tests()?.is_empty() {
                Ok(Some(()))
            } else if coordinator_thread.is_finished() {
                anyhow::bail!("coordinator returned before publishing its blocked claim")
            } else {
                Ok(None)
            }
        },
    );
    if let Err(error) = ready {
        cancel_registry_worker(&cancelled);
        drop(peer_sh);
        let result = recv_from_service_thread(
            &coordinator_rx,
            "cancelled fixed-set coordinator unwind",
            &coordinator_thread,
        )
        .expect("cancelled fixed-set coordinator must unwind");
        coordinator_thread
            .join()
            .expect("cancelled fixed-set coordinator thread");
        panic!("coordinator did not publish its claim: {error:#}; worker={result:?}");
    }

    // Small SH cell on a DIFFERENT LLC/CPU: must take disjoint capacity
    // without entering the coordinator's queue.
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let (small_tx, small_rx) = std::sync::mpsc::sync_channel(1);
    let small_thread = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        let result = acquire_resource_locks_waiting_impl(
            &[11],
            LlcLockMode::Shared,
            &[51],
            FlockMode::Exclusive,
            false,
            None,
        );
        let _ = small_tx.send(result);
    });
    let producer_services = [
        small_thread.service.clone(),
        coordinator_thread.service.clone(),
    ];
    let outcome = recv_with_task_service(
        &small_rx,
        "disjoint non-waiting shared-cell admission",
        &producer_services,
    )
    .expect("small-cell worker must publish")
    .expect("small-cell acquisition");
    small_thread.join().expect("small-cell worker");
    let (_, locks) = unwrap_acquired(
        outcome,
        Some("disjoint SH cell while the coordinator hungers"),
    );
    assert_eq!(locks.len(), 2);
    // `wait=false` plus the delivered-service-bounded worker proves this
    // disjoint request did not enter a queue behind the coordinator. Strict
    // arrival-order admission would either return `Unavailable` (caught above)
    // or consume the service budget while waiting.

    // Release the peer; the coordinator must now complete.
    drop(peer_sh);
    let coordinator_outcome = match recv_from_service_thread(
        &coordinator_rx,
        "coordinator post-release completion",
        &coordinator_thread,
    ) {
        Ok(result) => result.expect("coordinator acquire"),
        Err(error) => {
            cancel_registry_worker(&cancelled);
            let _ = recv_from_service_thread(
                &coordinator_rx,
                "cancelled fixed-set coordinator unwind",
                &coordinator_thread,
            )
            .expect("cancelled fixed-set coordinator must unwind");
            coordinator_thread
                .join()
                .expect("cancelled fixed-set coordinator thread");
            panic!("coordinator did not complete after release: {error:#}");
        }
    };
    coordinator_thread
        .join()
        .expect("completed coordinator thread");
    match coordinator_outcome {
        LockOutcome::Acquired { locks, .. } => assert_eq!(
            locks.len(),
            2,
            "independent LLC and CPU admission must retain both exclusive fds",
        ),
        LockOutcome::Unavailable(r) => {
            panic!("coordinator must complete after the release: {r}")
        }
    }
}

#[test]
fn coordinator_reprobes_when_a_release_races_the_initial_watch_transition() {
    use std::sync::mpsc;
    use std::sync::{Arc, atomic::AtomicBool};

    let _broker = InterruptibleFlockBrokerGuard::start();
    let _prefixes = LockPrefixesGuard::new();
    let claim = ticket_claim(&[1]);
    let blocker = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let (first_step_tx, first_step_rx) = mpsc::sync_channel(1);
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    let worker = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        let result = (|| {
            let coordinator = match protocol::register_ticket_or_acquire(
                claim.clone(),
                claim.clone(),
                Some(&worker_cancelled),
                |_| Ok::<Option<()>, anyhow::Error>(None),
            )? {
                protocol::TicketWork::Coordinator(coordinator) => coordinator,
                protocol::TicketWork::Acquired(_) => {
                    anyhow::bail!("fresh registry must elect a coordinator")
                }
            };
            let mut steps = 0usize;
            let outcome = protocol::acquire_as_coordinator_interruptible(
                coordinator,
                &worker_cancelled,
                |_| {
                    steps += 1;
                    if steps == 1 {
                        // Ask the parent to emit the close after the first old-watch
                        // observation, while this callback is publishing its initial
                        // waiting claim.
                        first_step_tx.send(()).ok();
                        return Ok(protocol::CoordinatorStep::Waiting {
                            claim: claim.clone(),
                        });
                    }
                    let lock = crate::flock::try_flock(
                        cpu_lock_path(1),
                        crate::flock::FlockMode::Exclusive,
                    )?
                    .expect("transition release must make CPU 1 available");
                    Ok(protocol::CoordinatorStep::Complete {
                        claim: claim.clone(),
                        value: lock,
                    })
                },
            )?;
            match outcome {
                protocol::CoordinatorOutcome::Acquired(lock) => drop(lock),
                protocol::CoordinatorOutcome::Prepared(_) => {
                    anyhow::bail!("transition-race coordinator prepared a VM intent")
                }
                protocol::CoordinatorOutcome::Aborted { reason } => {
                    anyhow::bail!("transition-race coordinator aborted: {reason}")
                }
            }
            Ok::<usize, anyhow::Error>(steps)
        })();
        let _ = result_tx.send(result);
    });
    if let Err(error) =
        recv_from_service_thread(&first_step_rx, "initial coordinator planning step", &worker)
    {
        cancel_registry_worker(&cancelled);
        drop(blocker);
        let result = recv_from_service_thread(
            &result_rx,
            "cancelled transition-race worker unwind",
            &worker,
        )
        .expect("cancelled transition-race worker must unwind");
        worker
            .join()
            .expect("cancelled transition-race coordinator worker");
        panic!(
            "coordinator did not reach its initial planning step: {error:#}; \
             worker={result:?}"
        );
    }
    drop(blocker);
    let steps = match recv_from_service_thread(
        &result_rx,
        "transition-race coordinator post-release completion",
        &worker,
    ) {
        Ok(result) => result.expect("coordinator transition-race acquire"),
        Err(error) => {
            cancel_registry_worker(&cancelled);
            let _ = recv_from_service_thread(
                &result_rx,
                "cancelled transition-race worker unwind",
                &worker,
            )
            .expect("cancelled transition-race worker must unwind");
            worker
                .join()
                .expect("cancelled transition-race coordinator worker");
            panic!("transition-race coordinator did not complete after release: {error:#}");
        }
    };
    worker
        .join()
        .expect("completed transition-race coordinator worker");
    assert_eq!(
        steps, 2,
        "a watch-set transition must force one fresh holder observation and re-probe",
    );
}

#[test]
fn known_free_close_storm_does_no_observation_scan_or_planner_work() {
    let _prefixes = LockPrefixesGuard::new();
    let (observations, scans, planner_steps, generation_changes, ex_acquisitions) =
        protocol::exercise_known_free_close_storm_for_tests(2_000)
            .expect("exercise known-free close storm");
    assert_eq!(
        (observations, scans, planner_steps, generation_changes,),
        (0, 0, 0, 0),
        "writable closes on already-free watched resources must be discarded \
         before any registry mutation, procfs observation, grant scan, or \
         planner execution",
    );
    // The normal live-heartbeat path stays SH-only. A host that denies this
    // process service for the full coordinator lease may legitimately take
    // the slow EX path solely to refresh liveness; the forced-stale regression
    // below covers that path without making this stress test wall-clock flaky.
    assert!(
        ex_acquisitions <= 1,
        "known-free closes repeatedly fell out of the SH fast path",
    );
}

#[test]
fn stale_heartbeat_release_fallback_ignores_unwatched_out_of_range_indices() {
    let _prefixes = LockPrefixesGuard::new();
    let (observations, scans, planner_steps, generation_changes, ex_acquisitions) =
        protocol::exercise_stale_heartbeat_known_free_close_for_tests()
            .expect("exercise stale-heartbeat known-free release fallback");
    assert_eq!(
        (observations, scans, planner_steps, generation_changes),
        (0, 0, 0, 0),
        "the stale-heartbeat fallback must preserve known-free fast-path semantics",
    );
    assert!(
        ex_acquisitions > 0,
        "test setup did not force the slow registry path",
    );
}

#[test]
fn busy_to_free_close_produces_exactly_one_improvement() {
    let _prefixes = LockPrefixesGuard::new();
    let (observations, scans, planner_steps) =
        protocol::exercise_busy_to_free_close_for_tests().expect("exercise busy-to-free close");
    assert_eq!(
        (observations, scans, planner_steps),
        (1, 1, 1),
        "a real busy-to-free transition must produce one observation, one \
         grant scan, and one coordinator planner wake",
    );
}

#[test]
fn shared_holder_close_does_not_replan_an_llc_sh_only_watch() {
    let _prefixes = LockPrefixesGuard::new();
    let (observation_requested, planner_step, scans, generations) =
        protocol::exercise_llc_sh_only_shared_to_free_close_for_tests()
            .expect("exercise SH-only shared-to-free close");
    assert!(
        !observation_requested && !planner_step,
        "SharedHeld→Free improves only EX compatibility and is irrelevant to an SH-only watch",
    );
    assert_eq!(
        (scans, generations),
        (0, 0),
        "an irrelevant close must not scan grants or publish a replan wake",
    );
}

#[test]
fn failed_llc_ex_probe_releases_compatible_shared_waiter_without_fallback() {
    let _prefixes = LockPrefixesGuard::new();
    let (scans, shared_granted, exclusive_waiting, coordinator_did_not_replan) =
        protocol::exercise_llc_ex_contention_shared_wake_for_tests()
            .expect("exercise LLC mode-compatible wake");
    assert_eq!(
        scans, 1,
        "the shared-held observation must trigger exactly one grant scan",
    );
    assert!(
        shared_granted,
        "SH compatibility restored by a shared-held observation must grant the queued SH ticket",
    );
    assert!(
        exclusive_waiting,
        "the same shared-held observation must keep the incompatible EX ticket blocked",
    );
    assert!(
        coordinator_did_not_replan,
        "an SH compatibility improvement must not spin the incompatible EX coordinator",
    );
}

#[test]
fn failed_cpu_ex_probe_wakes_only_the_compatible_shared_waiter() {
    let _prefixes = LockPrefixesGuard::new();
    let protocol::CpuExContentionSharedWake {
        scans,
        shared_granted,
        exclusive_waiting,
        sh_serial_advanced,
        ex_serial_unchanged,
        shared_woke,
        exclusive_not_woken,
        coordinator_did_not_replan,
    } = protocol::exercise_cpu_ex_contention_shared_wake_for_tests()
        .expect("exercise CPU mode-compatible wake");
    assert_eq!(
        scans, 1,
        "the CPU shared-held observation must trigger exactly one grant scan",
    );
    assert!(shared_granted, "CPU SH waiter must be granted");
    assert!(exclusive_waiting, "CPU EX waiter must remain blocked");
    assert!(
        sh_serial_advanced && ex_serial_unchanged,
        "SharedHeld must advance only the CPU SH compatibility serial",
    );
    assert!(
        shared_woke && exclusive_not_woken,
        "the targeted wake belongs only to the newly compatible CPU SH ticket",
    );
    assert!(
        coordinator_did_not_replan,
        "an SH compatibility improvement must not spin the incompatible EX coordinator",
    );
}

#[test]
fn exclusive_coordinator_waits_for_shared_holder_without_reprobe_spin() {
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let coordinator_waiting = markers.path().join("exclusive-coordinator.waiting");
    let shared_holder = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Shared)
        .expect("open shared CPU holder")
        .expect("hold CPU in shared mode");
    let coordinator = TicketChild::spawn_with_options(
        markers.path(),
        "exclusive-coordinator",
        "1",
        TicketSpawnOptions {
            coordinator_waiting: Some(&coordinator_waiting),
            ..TicketSpawnOptions::default()
        },
    );
    coordinator.wait_for_probe();
    coordinator.wait_for_path(&coordinator_waiting, "coordinator waiting marker");
    assert_eq!(
        coordinator.probe_count(),
        1,
        "resolving EX contention as SharedHeld must not synchronously reprobe EX",
    );

    drop(shared_holder);
    coordinator.wait_for_probe_count(2);
    coordinator.wait_for_acquired();
    coordinator.release_and_wait();
}

#[test]
fn coordinator_turnover_does_one_global_liveness_sweep_and_skips_known_free_modes() {
    let _prefixes = LockPrefixesGuard::new();
    let coordinators = 96usize;
    let (sweeps, probes, observation_requests, reconcile_deadline_coalesced) =
        protocol::exercise_coordinator_turnover_for_tests(coordinators)
            .expect("exercise rapid coordinator turnover");
    assert_eq!(
        sweeps, 1,
        "liveness maintenance is host-registry global, not once per coordinator",
    );
    assert_eq!(
        probes, coordinators,
        "one global sweep probes the initial active set once instead of producing an O(N²) tail",
    );
    assert!(
        reconcile_deadline_coalesced,
        "post-watch liveness reconciliation must keep the first shared deadline across rapid \
         coordinator handoffs instead of postponing it or sweeping once per handoff",
    );
    assert_eq!(
        observation_requests, 1,
        "only the known-busy CPU needs one durable observation request; the known-free CPU must \
         not be re-observed at each handoff",
    );
}

#[test]
fn exact_acquired_commits_do_not_force_full_grant_scans() {
    let _prefixes = LockPrefixesGuard::new();
    let scans = protocol::exercise_exact_commit_scan_elision_for_tests(64)
        .expect("exercise exact acquired commits");
    assert_eq!(
        scans, 0,
        "publishing an unchanged exact claim busy leaves every later grant valid; overlap waits \
         for the real release instead of scanning the whole registry at commit",
    );
}

#[test]
fn backfill_is_weighted_by_cooperative_capacity_instead_of_callback_count() {
    let _prefixes = LockPrefixesGuard::new();
    let (wave_credit, light_cost, heavy_cost, non_cooperative_cost) =
        protocol::exercise_resource_weighted_backfill_accounting_for_tests();
    assert!(
        wave_credit >= 4,
        "one backfill wave must expose at least the four cooperative lanes of one possible CPU",
    );
    assert_eq!(
        light_cost, 1,
        "one cooperative CPU permit costs one wave unit"
    );
    assert_eq!(
        heavy_cost, 4,
        "one callback consuming four CPU permits must cost four units, not one callback",
    );
    assert_eq!(
        non_cooperative_cost, 2,
        "claims outside the cooperative permit namespace fall back to physical width",
    );
}

#[test]
fn unavailable_wide_head_refills_live_backfill_capacity_until_bounded_age_then_drains() {
    let _prefixes = LockPrefixesGuard::new();
    let outcome = protocol::exercise_work_conserving_backfill_for_tests()
        .expect("exercise work-conserving bounded backfill");
    assert_eq!(
        outcome.conflicting_grants, 3,
        "the blocked wide head must admit exactly its configured outstanding resource capacity",
    );
    assert_eq!(
        outcome.conflicting_waiters, 2,
        "new conflicting work must wait while the head's full live capacity is occupied",
    );
    assert_eq!(
        outcome.disjoint_grants, 8,
        "disjoint suffix work must remain unbounded by the fairness capacity",
    );
    assert!(
        outcome.refilled_after_completion,
        "completed bypass work must immediately open replacement capacity instead of creating a low-utilization drain",
    );
    assert!(
        outcome.expired_head_stops_refill,
        "an aged head must stop admitting replacement conflicts so exclusive work cannot starve",
    );
    assert!(
        outcome.wide_wins,
        "once the admitted burst releases, the now-viable wide head must receive the next exact grant",
    );
    assert!(
        outcome.racer_revoked_without_placement_damage,
        "the viability scan must revoke a stale conflicting callback while preserving disjoint grants",
    );
    assert!(
        outcome.stale_callback_suppressed,
        "a callback revoked by the wide-head scan must not execute planner or acquisition code",
    );
}

#[test]
fn commit_that_omits_old_claim_resources_still_rescans_the_unblocked_suffix() {
    let _prefixes = LockPrefixesGuard::new();
    let (scans, later_granted) = protocol::exercise_mismatched_commit_rescan_for_tests()
        .expect("exercise mismatched coordinator commit");
    assert_eq!(
        scans, 1,
        "removing an old claim not covered by the committed exact hold needs one suffix scan",
    );
    assert!(
        later_granted,
        "the scan must grant later work on the omitted, already-free old resource",
    );
}

#[test]
fn commit_that_adds_resources_rescans_and_revokes_stale_grants() {
    let _prefixes = LockPrefixesGuard::new();
    let (scans, later_revoked) = protocol::exercise_superset_commit_rescan_for_tests()
        .expect("exercise strict-superset coordinator commit");
    assert_eq!(
        scans, 1,
        "adding a newly-busy resource to the committed exact hold needs one suffix scan",
    );
    assert!(
        later_revoked,
        "a later grant on the newly-busy resource must be revoked before its real flock probe",
    );
}

#[test]
fn shared_commit_rescans_when_real_hold_improves_shared_compatibility() {
    let _prefixes = LockPrefixesGuard::new();
    let (scans, later_granted) = protocol::exercise_shared_commit_improvement_for_tests()
        .expect("exercise shared compatibility improvement");
    assert_eq!(
        scans, 1,
        "publishing a real SH hold over unknown or EX-held state needs one compatibility scan",
    );
    assert!(
        later_granted,
        "a later SH waiter must be granted when the committed real SH hold proves compatibility",
    );
}

#[test]
fn cpu_shared_commit_rescans_when_real_hold_improves_shared_compatibility() {
    let _prefixes = LockPrefixesGuard::new();
    let (scans, later_granted) = protocol::exercise_cpu_shared_commit_improvement_for_tests()
        .expect("exercise CPU shared compatibility improvement");
    assert_eq!(
        scans, 1,
        "publishing a real CPU SH hold over unknown or EX-held state needs one compatibility scan",
    );
    assert!(
        later_granted,
        "a later CPU SH waiter must be granted when the committed real SH hold proves compatibility",
    );
}

#[test]
fn dirty_repair_preserves_exact_and_watch_cpu_modes() {
    let _prefixes = LockPrefixesGuard::new();
    let (flexible_preserved, flexible_still_flexible, fixed_preserved, fixed_still_fixed) =
        protocol::exercise_cpu_mode_repair_for_tests().expect("exercise dirty CPU-mode repair");
    assert!(
        flexible_preserved && flexible_still_flexible,
        "repair must retain a CPU SH exact claim under its CPU EX watch as an invalidated WAITING ticket",
    );
    assert!(
        fixed_preserved && fixed_still_fixed,
        "canonical empty-class modes must keep an exact CPU SH claim fixed after repair",
    );
}

#[test]
fn common_watch_replan_wave_is_work_conserving_and_finite() {
    let _prefixes = LockPrefixesGuard::new();
    let waiters = 1_000usize;
    let outcome = protocol::exercise_replan_token_wave_for_tests(waiters)
        .expect("exercise finite work-conserving REPLAN waves");
    let expected_exact_grants = (0..waiters).step_by(16).count();
    let expected_replans = waiters - expected_exact_grants;
    assert!(
        outcome.registration_waiting,
        "flexible registration must not publish speculative callbacks directly",
    );
    assert_eq!(
        outcome.exact_grants, expected_exact_grants,
        "one scan must still drain every exact viable disjoint grant",
    );
    assert_eq!(
        outcome.initial_replans, expected_replans,
        "one finite scan must publish every eligible flexible callback",
    );
    assert_eq!(
        outcome.initial_wakes, waiters,
        "the first scan must wake every exact grant and every eligible REPLAN callback",
    );
    assert_eq!(
        outcome.initial_prefix_comparisons, 1,
        "the scan may validate its coordinator publication, but must not reread the 1,000 flexible WAITING prefixes",
    );
    assert_eq!(
        outcome.initial_full_watch_materializations, 0,
        "the authoritative scan must not expand any ticket's encoded alternative watch into BTree nodes",
    );
    assert_eq!(
        outcome.initial_encoded_watch_serial_walks, expected_replans,
        "the global serial filter must walk each eligible callback's encoded watch exactly once",
    );
    assert_eq!(
        outcome.initial_full_prefix_snapshot_publishes, 0,
        "the scan must publish predecessor words directly without allocating unused host-sized holder vectors",
    );
    assert_eq!(
        (outcome.repeated_replans, outcome.repeated_wakes),
        (expected_replans, 0),
        "repeating the scan must preserve every live callback without duplicate wakes",
    );
    assert!(
        outcome.fixed_waiter_granted && outcome.fixed_waiter_woken,
        "new exact capacity must grant and wake independently while a REPLAN wave is live",
    );
    assert_eq!(
        (outcome.fixed_scan_replans, outcome.fixed_scan_replan_wakes,),
        (expected_replans, 0),
        "an exact grant scan must preserve the live REPLAN wave without re-waking it",
    );
    assert!(
        outcome.callback_requeued,
        "an unchanged speculative callback must return to WAITING",
    );
    assert_eq!(
        outcome.callback_prefix_reads, 1,
        "each callback must copy only its own cached prefix record",
    );
    assert_eq!(
        outcome.callback_active_reads, 0,
        "callback admission must remain independent of active queue depth",
    );
    assert!(
        !outcome.mixed_age_old_replanned
            && !outcome.mixed_age_old_woken
            && !outcome.mixed_age_late_replanned
            && !outcome.mixed_age_late_woken,
        "a live finite wave must leave both an older returned callback and a later arrival for the next batch",
    );
    assert_eq!(
        (
            outcome.mixed_age_repeated_replans,
            outcome.mixed_age_repeated_wakes,
        ),
        (0, 0),
        "repeated scans must not extend or wake a live finite wave",
    );
    assert!(
        outcome.next_wave_edge_published,
        "the final original callback must publish one deferred next-wave rescan without running it",
    );
    assert_eq!(
        outcome.next_wave_scan_delta, 1,
        "one authoritative scan must consume the completed finite-wave edge",
    );
    assert!(
        outcome.next_wave_old_replanned
            && outcome.next_wave_old_woken
            && outcome.next_wave_late_replanned
            && outcome.next_wave_late_woken,
        "the next wave must publish and wake both the older returned callback and post-horizon arrival exactly once",
    );
    assert_eq!(
        (
            outcome.next_wave_repeated_replans,
            outcome.next_wave_repeated_wakes,
        ),
        (2, 0),
        "a repeated scan must preserve both next-wave callbacks without duplicate wakes",
    );
}

#[test]
fn changed_replan_wave_defers_one_authoritative_scan_until_every_callback_returns() {
    let _prefixes = LockPrefixesGuard::new();
    let callbacks = 1_000usize;
    let outcome = protocol::exercise_changed_replan_batch_for_tests(callbacks)
        .expect("exercise changed speculative callback batch");
    assert_eq!(outcome.callbacks, callbacks);
    assert_eq!(
        (
            outcome.intermediate_scan_delta,
            outcome.intermediate_generation_wake_delta,
        ),
        (0, 0),
        "the first N-1 changed callbacks must do O(1) publication work without scanning or waking the queue",
    );
    assert!(
        outcome.intermediate_batch_only,
        "the unfinished wave must retain one batch-only dirty edge and one outstanding callback",
    );
    assert_eq!(
        outcome.final_scan_delta_before_authoritative, 0,
        "the final callback must publish, but must not execute, the authoritative scan",
    );
    assert_eq!(
        outcome.final_generation_wake_delta, 1,
        "the complete changed wave must publish exactly one queue wake",
    );
    assert!(
        outcome.final_rescan_edge,
        "the final callback must atomically promote the batch to one pending rescan",
    );
    assert_eq!(
        outcome.authoritative_scan_delta, 1,
        "one coordinator scan must consume the complete changed batch",
    );
    assert!(
        outcome.authoritative_flags_clear && outcome.replacements_preserved,
        "the authoritative scan must clear both dirty flags without losing any callback replacement",
    );
}

#[test]
fn changed_replan_batch_parks_later_grants_and_coordinator_commits_without_scanning() {
    let _prefixes = LockPrefixesGuard::new();
    let outcome = protocol::exercise_replan_batch_barriers_for_tests()
        .expect("exercise changed speculative batch priority barriers");
    assert!(
        outcome.granted_entry_callback_suppressed,
        "a later grant must park before entering its physical callback",
    );
    assert!(
        outcome.granted_completion_payload_dropped
            && outcome.granted_completion_payload_dropped_unlocked,
        "a later physical success must be released outside the registry lock when an older choice changed",
    );
    assert!(
        outcome.granted_contention_serials_preserved
            && outcome.granted_contention_unknown_before_witness_drop,
        "a deferred negative probe must retain its blocker/issue serial and publish UNKNOWN before releasing its witness",
    );
    assert!(
        outcome.granted_records_parked,
        "both later grants must return to WAITING behind the unfinished speculative batch",
    );
    assert_eq!(
        outcome.granted_scan_delta, 0,
        "entry and completion barriers must remain O(1)",
    );
    assert!(
        outcome.coordinator_acquire_deferred && outcome.coordinator_preparation_deferred,
        "both coordinator commit shapes must defer behind an older changed choice",
    );
    assert!(
        outcome.coordinator_acquire_evidence_unknown
            && outcome.coordinator_preparation_evidence_unknown,
        "coordinator deferral must publish the exact acquire, accumulated blockers, and distinct preparation footprint as UNKNOWN",
    );
    assert!(
        outcome.coordinator_preserved,
        "batch deferral must retain the coordinator license until the final rescan edge",
    );
    assert_eq!(
        outcome.coordinator_scan_delta, 0,
        "coordinator batch deferral must not reconcile the queue eagerly",
    );
}

#[test]
fn replan_wave_crash_repair_recovers_every_eligible_callback_once() {
    let _prefixes = LockPrefixesGuard::new();
    let outcome = protocol::exercise_replan_crash_repair_for_tests()
        .expect("repair torn REPLAN cursor publication");
    assert!(
        outcome.dirty_repair_completed && outcome.torn_callbacks_demoted,
        "dirty recovery must demote ambiguously delivered REPLAN publications to invalidated WAITING",
    );
    assert!(
        outcome.repair_generation_advanced && outcome.repair_generation_woke,
        "dirty repair must publish a generation change and wake every pending registrant",
    );
    assert!(
        outcome.cursor_preserved && outcome.horizon_preserved,
        "recovery must retain a coherent finite-round cursor and horizon",
    );
    assert!(
        outcome.all_eligible_recovered,
        "the repaired scan must republish every eligible callback",
    );
    assert_eq!(
        (outcome.recovered_replans, outcome.recovered_wakes),
        (2, 2),
        "recovery must publish and wake both eligible REPLAN callbacks",
    );
    assert_eq!(
        (outcome.repeated_replans, outcome.repeated_wakes),
        (2, 0),
        "a repeated scan must preserve the recovered wave without duplicate wakes",
    );
}

#[test]
fn intrascan_fence_activation_refreshes_live_replan_epoch() {
    let _prefixes = LockPrefixesGuard::new();
    let (earlier_granted, publication_changed, completion_accepted_for_revalidation) =
        protocol::exercise_intrascan_fence_epoch_for_tests()
            .expect("exercise intra-scan predecessor-fence activation");
    assert!(
        earlier_granted,
        "the earlier fixed waiter must become GRANTED"
    );
    assert!(
        publication_changed,
        "activating its fence must refresh the later live REPLAN with a fresh epoch",
    );
    assert!(
        completion_accepted_for_revalidation,
        "the same live non-acquiring callback must publish WAITING once and defer exact validation to the batch scan",
    );
}

#[test]
fn scanner_death_after_earlier_grant_demotes_stale_later_grant() {
    let _prefixes = LockPrefixesGuard::new();
    let (stale_later_rejected, later_demoted, earlier_regranted) =
        protocol::exercise_grant_scan_crash_fence_for_tests()
            .expect("repair scanner death between an earlier grant and later revocation");
    assert!(
        stale_later_rejected,
        "a later conflicting GRANTED publication must not enter its stale callback after scanner death",
    );
    assert!(
        later_demoted,
        "dirty repair must demote the unvisited later grant before exposing a clean registry",
    );
    assert!(
        earlier_regranted,
        "a fresh authoritative scan must restore forward progress for the earlier ticket",
    );
}

#[test]
fn callback_tokens_change_only_when_their_exact_prefix_or_watch_changes() {
    let _prefixes = LockPrefixesGuard::new();
    let outcome = protocol::exercise_granular_prefix_invalidation_for_tests()
        .expect("exercise granular callback invalidation");
    assert!(
        outcome.coordinator_preserved,
        "later claim epochs must not rewrite the coordinator's unchanged empty prefix",
    );
    assert!(
        outcome.granted_preserved && outcome.replan_preserved && outcome.waiting_preserved,
        "duplicate predecessor churn with identical aggregate prefix, watch serial, and blocker \
         viability must preserve GRANTED, REPLAN, and WAITING publications",
    );
    assert!(
        outcome.granted_refreshed && outcome.replan_refreshed && outcome.waiting_replanned,
        "a real newly-added predecessor bit must refresh or replan every affected callback state",
    );
    assert!(
        outcome.entry_unchanged_deferred && outcome.entry_changed_deferred,
        "a coherent non-acquiring callback must keep its one planner turn and defer both equivalent and changed predecessor reconciliation",
    );
    assert!(
        outcome.completion_unchanged_kept && outcome.completion_changed_deferred,
        "callback completion must retain its one speculative result while exact validation remains deferred to the authoritative scan",
    );
    assert!(
        outcome.coordinator_completion_unchanged_kept
            && outcome.coordinator_completion_changed_rejected,
        "coordinator commit must likewise retain physical success across aggregate-equivalent \
         predecessor churn and reject it after a real prefix change",
    );
}

#[test]
fn live_callbacks_ignore_resource_serial_churn_until_waiting_again() {
    let _prefixes = LockPrefixesGuard::new();
    let (alternatives_kept, designated_kept, physical_commit_kept, herd_kept) =
        protocol::exercise_granted_serial_scope_for_tests()
            .expect("exercise exact grant resource-serial scope");
    assert!(
        alternatives_kept && designated_kept,
        "resource-improvement serials must not rewrite a live exact grant; its physical probe is \
         the authoritative availability result",
    );
    assert!(
        physical_commit_kept,
        "an unrelated alternative improvement during an exact physical acquisition must not \
         discard the acquired payload",
    );
    assert!(
        herd_kept,
        "repeated unrelated improvements must preserve every live GRANTED/REPLAN publication in \
         a herd",
    );
}

#[test]
fn revoked_grants_keep_their_fence_until_the_callback_acknowledges() {
    let _prefixes = LockPrefixesGuard::new();
    let outcome = protocol::exercise_revocation_ack_for_tests()
        .expect("exercise revoked grant acknowledgement");
    assert!(
        outcome.before_entry_acked && outcome.during_callback_acked,
        "revocation must be acknowledged both before callback entry and after an in-flight \
         callback drops its result",
    );
    assert!(
        outcome.later_publication_preserved,
        "the revocation scan must retain the old exact predecessor fence and must not mutate a \
         later callback publication",
    );
    assert!(
        outcome.successor_rescan_published,
        "REVOKED acknowledgement must atomically publish WAITING plus the successor rescan",
    );
    assert!(
        outcome.flexible_replanned_without_serial_churn,
        "a flexible revoked grant must reconsider an already-free alternative after ACK without \
         waiting for unrelated resource-serial churn",
    );
}

#[test]
fn revoked_owner_death_retires_fence_and_wakes_successor() {
    let _prefixes = LockPrefixesGuard::new();
    let (live_fence_preserved, dead_record_removed, successor_granted, successor_woken) =
        protocol::exercise_revoked_owner_death_for_tests()
            .expect("exercise REVOKED owner liveness pruning");
    assert!(
        live_fence_preserved,
        "a live REVOKED owner must retain its predecessor fence even after capacity returns",
    );
    assert!(
        dead_record_removed,
        "the same fence must be removed once its authoritative liveness owner dies",
    );
    assert!(
        successor_granted && successor_woken,
        "pruning the dead REVOKED fence must immediately grant and wake its successor",
    );
}

#[test]
fn revoke_before_wake_crash_repair_wakes_ack_and_successor() {
    let _prefixes = LockPrefixesGuard::new();
    let (preserved, owner_woken, acked_without_callback, successor_progressed) =
        protocol::exercise_revoke_crash_repair_for_tests()
            .expect("repair a torn GRANTED-to-REVOKED publication");
    assert!(
        preserved,
        "dirty repair must preserve the ambiguous REVOKED predecessor fence",
    );
    assert!(
        owner_woken,
        "repair must replay the targeted wake which the crashed scanner may have missed",
    );
    assert!(
        acked_without_callback,
        "the woken owner must acknowledge REVOKED without entering stale planner code",
    );
    assert!(
        successor_progressed,
        "retiring the acknowledged owner must let its successor advance",
    );
}

#[test]
fn free_observation_wakes_clean_waiting_queue_without_an_initial_scan() {
    let _prefixes = LockPrefixesGuard::new();
    let (waiting_without_scan, observation_scheduled, granted, futex_woken) =
        protocol::exercise_waiting_release_wake_for_tests()
            .expect("exercise no-initial-scan WAITING wake");
    assert!(
        waiting_without_scan,
        "the appended ticket must begin WAITING before any compatibility scan has run",
    );
    assert!(
        observation_scheduled,
        "a free observation must durably request the WAITING ticket's first compatibility scan",
    );
    assert!(
        granted && futex_woken,
        "that scan must grant and directly wake the clean WAITING ticket",
    );
}

#[test]
fn replan_publishes_one_replacement_then_returns_to_coordinator_admission() {
    let _prefixes = LockPrefixesGuard::new();
    let (callbacks, requeued_without_acquire, waiting, replaced, rescan_pending, active_reads) =
        protocol::exercise_one_shot_replacement_for_tests().expect("exercise one-shot replacement");
    assert_eq!(callbacks, 1, "one wake must invoke the planner once");
    assert!(
        requeued_without_acquire,
        "REPLAN must publish a replacement without acquiring it",
    );
    assert!(
        waiting && replaced && rescan_pending,
        "the replacement must be WAITING behind a durable coordinator rescan",
    );
    assert_eq!(
        active_reads, 0,
        "replacement publication must not scan the active queue",
    );
}

#[test]
fn torn_and_stale_prefix_epochs_fail_closed_before_callback() {
    let _prefixes = LockPrefixesGuard::new();
    let (callbacks, torn_rejected, stale_rejected) =
        protocol::exercise_prefix_epoch_validation_for_tests()
            .expect("exercise prefix epoch validation");
    assert_eq!(
        callbacks, 0,
        "neither a torn nor stale predecessor snapshot may reach planner code",
    );
    assert!(
        torn_rejected && stale_rejected,
        "invalid snapshots must demote the ticket to WAITING for a fresh scan",
    );
}

#[test]
fn stale_negative_probe_commits_current_blocker_and_discards_stale_alternative() {
    let _prefixes = LockPrefixesGuard::new();
    let (requeued, exact_preserved, blocked_current, stayed_waiting) =
        protocol::exercise_stale_contention_commit_for_tests()
            .expect("exercise stale negative probe commit");
    assert!(
        requeued,
        "stale negative evidence must requeue the live grant"
    );
    assert!(
        exact_preserved,
        "an alternative selected from the stale snapshot must be discarded",
    );
    assert!(
        blocked_current,
        "the real contention must be recorded at the current resource serial",
    );
    assert!(
        stayed_waiting,
        "the same serial must not immediately regrant the failed exact probe",
    );
}

#[test]
fn stale_positive_probe_releases_payload_before_coordinator_notification() {
    let _prefixes = LockPrefixesGuard::new();
    let (
        lost_grant,
        regrant_revoked,
        payload_dropped,
        registry_unlocked_at_drop,
        dropped_at_notify,
        observation_requested,
    ) = protocol::exercise_stale_acquired_release_order_for_tests()
        .expect("exercise stale acquired-payload release ordering");
    assert!(
        lost_grant,
        "an acquired payload from an invalidated callback must lose its stale grant",
    );
    assert!(
        regrant_revoked,
        "a same-claim regrant issued while the stale payload was held must be revoked",
    );
    assert!(
        payload_dropped,
        "the physical payload must be released on the lost-grant path",
    );
    assert!(
        registry_unlocked_at_drop,
        "caller-owned payload destruction must not run under the registry fence",
    );
    assert!(
        dropped_at_notify,
        "the physical payload must be released before notifying the coordinator",
    );
    assert!(
        observation_requested,
        "discarding a stale positive probe must invalidate optimistic availability",
    );
}

#[test]
fn predecessor_prefixes_preserve_modes_order_and_dirty_repair() {
    let _prefixes = LockPrefixesGuard::new();
    let (initial_modes, successor_excluded, repaired_modes, repaired_order) =
        protocol::exercise_prefix_order_and_repair_for_tests()
            .expect("exercise prefix order and dirty repair");
    assert!(
        initial_modes && successor_excluded,
        "a newly appended prefix must include mode-correct predecessors and exclude successors",
    );
    assert!(
        repaired_modes && repaired_order,
        "dirty recovery must rebuild mode-correct prefixes in ticket order",
    );
}

#[test]
fn predecessor_release_refreshes_an_already_runnable_replan_prefix() {
    let _prefixes = LockPrefixesGuard::new();
    let (prefix_refreshed, publication_refreshed, candidate_ready, replacement_committed) =
        protocol::exercise_prefix_refresh_after_predecessor_release_for_tests()
            .expect("exercise acquired-predecessor prefix refresh");
    assert!(
        prefix_refreshed && publication_refreshed,
        "a holder release must publish one coherent refreshed predecessor snapshot",
    );
    assert!(
        candidate_ready && replacement_committed,
        "the release must become a usable one-shot replacement without waiting for another event",
    );
}

#[test]
fn waiting_publication_preserves_consumed_predecessor_release_progress() {
    let _prefixes = LockPrefixesGuard::new();
    let (stale_prefix, release_published, prefix_refreshed, immediate_step) =
        protocol::exercise_waiting_publication_release_progress_for_tests()
            .expect("exercise release during coordinator WAITING publication");
    assert!(
        stale_prefix && release_published,
        "HELD removal must durably publish a prefix rescan even when availability was already free",
    );
    assert!(
        prefix_refreshed && immediate_step,
        "the WAITING publication must return refreshed predecessors as immediate planner progress without relying on another close event",
    );
    assert!(
        protocol::waiting_publication_requires_immediate_turn_for_tests(true, false),
        "consuming the one-shot prefix progress signal must skip the coordinator watch wait",
    );
    assert!(
        !protocol::waiting_publication_requires_immediate_turn_for_tests(false, false),
        "a quiescent WAITING publication must still sleep instead of spinning",
    );
}

#[test]
fn callback_cannot_consume_an_improvement_it_did_not_observe() {
    let _prefixes = LockPrefixesGuard::new();
    let (stale_published_once, fresh_seen, replacement_revalidated, serial_consumed_by_fresh) =
        protocol::exercise_issue_serial_race_for_tests()
            .expect("exercise callback issue-serial race");
    assert!(
        stale_published_once,
        "a REPLAN callback may publish its old choice once, but WAITING must retain the unseen \
         improvement as immediate replan work",
    );
    assert!(
        fresh_seen && replacement_revalidated && serial_consumed_by_fresh,
        "the already-runnable ticket must use the refreshed snapshot once, then keep a selected \
         claim WAITING when the successor scan finds it unavailable",
    );
}

#[test]
fn candidate_readiness_uses_prefix_availability_contention_and_exact_license() {
    protocol::exercise_candidate_ready_matrix_for_tests()
        .expect("exercise candidate readiness matrix");
}

#[test]
fn coordinator_commit_is_terminal_even_if_cancellation_arrives_afterward() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let _broker = InterruptibleFlockBrokerGuard::start();
    let _prefixes = LockPrefixesGuard::new();
    let cancelled = AtomicBool::new(false);
    let claim = ticket_claim(&[1]);
    let coordinator = match protocol::register_ticket_or_acquire(
        claim.clone(),
        claim.clone(),
        Some(&cancelled),
        |_| Ok::<Option<Vec<std::os::fd::OwnedFd>>, anyhow::Error>(None),
    )
    .expect("register terminal-commit coordinator")
    {
        protocol::TicketWork::Coordinator(coordinator) => coordinator,
        protocol::TicketWork::Acquired(_) => panic!("fresh registry must elect a coordinator"),
    };
    protocol::cancel_coordinator_after_commit_for_tests();
    let outcome = protocol::acquire_as_coordinator_interruptible(coordinator, &cancelled, |_| {
        let locks = match try_ticket_candidate(&[1])? {
            protocol::ProbeOutcome::Acquired(locks) => locks,
            protocol::ProbeOutcome::Contended(_) | protocol::ProbeOutcome::Unavailable => {
                anyhow::bail!("fresh coordinator resource unexpectedly contended")
            }
        };
        Ok(protocol::CoordinatorStep::Complete {
            claim: claim.clone(),
            value: locks,
        })
    })
    .expect("post-commit cancellation must not replace coordinator success");
    let locks = match outcome {
        protocol::CoordinatorOutcome::Acquired(locks) => locks,
        protocol::CoordinatorOutcome::Prepared(_) => {
            panic!("terminal-commit coordinator prepared a VM intent")
        }
        protocol::CoordinatorOutcome::Aborted { reason } => {
            panic!("terminal-commit coordinator aborted: {reason}")
        }
    };
    assert!(
        cancelled.load(Ordering::Acquire),
        "deterministic hook must deliver cancellation after registry commit",
    );
    assert!(
        crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
            .expect("probe committed coordinator ownership")
            .is_none(),
        "the successful return must retain the committed real flock",
    );
    drop(locks);
    drop(
        crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
            .expect("probe released coordinator ownership")
            .expect("the next acquire must succeed immediately after ownership drops"),
    );
}

#[test]
fn granted_commit_is_terminal_even_if_cancellation_arrives_afterward() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};

    let _broker = InterruptibleFlockBrokerGuard::start();
    let _prefixes = LockPrefixesGuard::new();
    let blocker = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .expect("open coordinator blocker")
        .expect("lock coordinator CPU");
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let coordinator_cancelled = Arc::new(AtomicBool::new(false));
    let worker_coordinator_cancelled = Arc::clone(&coordinator_cancelled);
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let (coordinator_tx, coordinator_rx) = mpsc::sync_channel(1);
    let coordinator_worker = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        let result = (|| {
            let claim = ticket_claim(&[1]);
            let coordinator = match protocol::register_ticket_or_acquire(
                claim.clone(),
                claim.clone(),
                None,
                |_| Ok::<Option<Vec<std::os::fd::OwnedFd>>, anyhow::Error>(None),
            )? {
                protocol::TicketWork::Coordinator(coordinator) => coordinator,
                protocol::TicketWork::Acquired(_) => {
                    anyhow::bail!("fresh registry must elect a coordinator")
                }
            };
            ready_tx.send(()).ok();
            let target = protocol::canonical_lock_order(&[], crate::flock::FlockMode::Shared, &[1]);
            let outcome = protocol::acquire_as_coordinator_interruptible(
                coordinator,
                &worker_coordinator_cancelled,
                |held| {
                    if let Some(locks) = held.probe_complete_if_ready(&claim, &target)? {
                        Ok(protocol::CoordinatorStep::Complete {
                            claim: claim.clone(),
                            value: locks,
                        })
                    } else {
                        Ok(protocol::CoordinatorStep::Waiting {
                            claim: claim.clone(),
                        })
                    }
                },
            )?;
            match outcome {
                protocol::CoordinatorOutcome::Acquired(locks) => drop(locks),
                protocol::CoordinatorOutcome::Prepared(_) => {
                    anyhow::bail!("granted-path coordinator prepared a VM intent")
                }
                protocol::CoordinatorOutcome::Aborted { reason } => {
                    anyhow::bail!("granted-path coordinator aborted: {reason}")
                }
            }
            Ok::<(), anyhow::Error>(())
        })();
        let _ = coordinator_tx.send(result);
    });
    if let Err(error) = recv_from_service_thread(
        &ready_rx,
        "granted-path coordinator publication",
        &coordinator_worker,
    ) {
        coordinator_cancelled.store(true, Ordering::Release);
        drop(blocker);
        let result = recv_from_service_thread(
            &coordinator_rx,
            "cancelled granted-path coordinator unwind",
            &coordinator_worker,
        )
        .expect("cancelled granted-path coordinator must unwind");
        coordinator_worker
            .join()
            .expect("cancelled granted-path coordinator worker");
        panic!(
            "coordinator did not publish before granted waiter starts: {error:#}; \
             worker={result:?}"
        );
    }

    let waiter_cancelled = Arc::new(AtomicBool::new(false));
    let worker_waiter_cancelled = Arc::clone(&waiter_cancelled);
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let (waiter_tx, waiter_rx) = mpsc::sync_channel(1);
    let waiter_worker = TestServiceThread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        protocol::cancel_granted_after_commit_for_tests();
        let claim = ticket_claim(&[2]);
        let result = protocol::register_ticket_or_acquire(
            claim.clone(),
            claim.clone(),
            Some(&worker_waiter_cancelled),
            |probe| probe.try_acquire(&claim, || try_ticket_candidate(&[2])),
        )
        .and_then(|work| match work {
            protocol::TicketWork::Acquired(locks) => Ok(locks),
            protocol::TicketWork::Coordinator(_) => {
                anyhow::bail!("later disjoint waiter unexpectedly became coordinator")
            }
        });
        let _ = waiter_tx.send(result);
    });
    let locks = match recv_with_task_service(
        &waiter_rx,
        "disjoint granted waiter completion",
        &[
            waiter_worker.service.clone(),
            coordinator_worker.service.clone(),
        ],
    ) {
        Ok(result) => result.expect("granted waiter must return committed success"),
        Err(error) => {
            cancel_registry_worker(&waiter_cancelled);
            drop(blocker);
            let _ = recv_from_service_thread(
                &waiter_rx,
                "cancelled granted waiter unwind",
                &waiter_worker,
            )
            .expect("cancelled granted waiter must unwind");
            waiter_worker.join().expect("cancelled granted waiter");
            let _ = recv_from_service_thread(
                &coordinator_rx,
                "released granted-path coordinator completion",
                &coordinator_worker,
            )
            .expect("released coordinator must complete");
            coordinator_worker
                .join()
                .expect("released coordinator worker");
            panic!("granted waiter did not complete: {error:#}");
        }
    };
    waiter_worker.join().expect("completed granted waiter");
    assert!(
        waiter_cancelled.load(Ordering::Acquire),
        "deterministic hook must cancel after the granted registry commit",
    );
    assert!(
        crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
            .expect("probe committed granted ownership")
            .is_none(),
        "the granted return must retain its committed real flock",
    );
    drop(locks);
    drop(
        crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
            .expect("probe released granted ownership")
            .expect("the next acquire must succeed immediately after granted ownership drops"),
    );

    drop(blocker);
    match recv_from_service_thread(
        &coordinator_rx,
        "granted-path coordinator post-release completion",
        &coordinator_worker,
    ) {
        Ok(result) => result.expect("coordinator completes after blocker release"),
        Err(error) => {
            coordinator_cancelled.store(true, Ordering::Release);
            drop(
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(cpu_lock_path(1))
                    .expect("wake cancelled coordinator"),
            );
            let _ = recv_from_service_thread(
                &coordinator_rx,
                "cancelled granted-path coordinator unwind",
                &coordinator_worker,
            )
            .expect("cancelled coordinator must unwind");
            coordinator_worker
                .join()
                .expect("cancelled coordinator worker");
            panic!("coordinator did not complete after blocker release: {error:#}");
        }
    }
    coordinator_worker
        .join()
        .expect("completed granted-path coordinator");
}

const TICKET_HELPER_LLC_PREFIX: &str = "KTSTR_TEST_TICKET_LLC_PREFIX";
const TICKET_HELPER_CPU_PREFIX: &str = "KTSTR_TEST_TICKET_CPU_PREFIX";
const TICKET_HELPER_CANDIDATES: &str = "KTSTR_TEST_TICKET_CANDIDATES";
const TICKET_HELPER_ACQUIRED: &str = "KTSTR_TEST_TICKET_ACQUIRED";
const TICKET_HELPER_PROBED: &str = "KTSTR_TEST_TICKET_PROBED";
const TICKET_HELPER_RELEASE: &str = "KTSTR_TEST_TICKET_RELEASE";
const TICKET_HELPER_SERVICE_TID: &str = "KTSTR_TEST_TICKET_SERVICE_TID";
const TICKET_HELPER_DISABLE_BYPASS: &str = "KTSTR_TEST_TICKET_DISABLE_BYPASS";
const TICKET_HELPER_FORCE_OBSERVER_NONE: &str = "KTSTR_TEST_TICKET_FORCE_OBSERVER_NONE";
const TICKET_HELPER_PROBE_BARRIER_DIR: &str = "KTSTR_TEST_TICKET_PROBE_BARRIER_DIR";
const TICKET_HELPER_PROBE_BARRIER_COUNT: &str = "KTSTR_TEST_TICKET_PROBE_BARRIER_COUNT";
const TICKET_HELPER_BEFORE_PROBE_GATE: &str = "KTSTR_TEST_TICKET_BEFORE_PROBE_GATE";
const TICKET_HELPER_RETAIN_FINAL_CANDIDATE: &str = "KTSTR_TEST_TICKET_RETAIN_FINAL_CANDIDATE";
const TICKET_HELPER_AFTER_ACQUIRE_GATE: &str = "KTSTR_TEST_TICKET_AFTER_ACQUIRE_GATE";
const TICKET_HELPER_AFTER_ACQUIRE_ENTERED: &str = "KTSTR_TEST_TICKET_AFTER_ACQUIRE_ENTERED";
const TICKET_HELPER_BEFORE_COORDINATOR_GATE: &str = "KTSTR_TEST_TICKET_BEFORE_COORDINATOR_GATE";
const TICKET_HELPER_COORDINATOR_ENTERED: &str = "KTSTR_TEST_TICKET_COORDINATOR_ENTERED";
const TICKET_HELPER_COORDINATOR_WAITING: &str = "KTSTR_TEST_TICKET_COORDINATOR_WAITING";
const TICKET_HELPER_MAPPING_COUNT: &str = "KTSTR_TEST_TICKET_MAPPING_COUNT";
const RETAINED_FUTEX_WAIT_MARKER: &str = "KTSTR_TEST_RETAINED_FUTEX_WAIT_MARKER";

const PENDING_V3_LLC_PREFIX: &str = "KTSTR_TEST_PENDING_V3_LLC_PREFIX";
const PENDING_V3_CPU_PREFIX: &str = "KTSTR_TEST_PENDING_V3_CPU_PREFIX";
const PENDING_V3_MARKER: &str = "KTSTR_TEST_PENDING_V3_MARKER";

fn install_pending_v3_prefixes() -> bool {
    let Some(llc) = std::env::var_os(PENDING_V3_LLC_PREFIX) else {
        return false;
    };
    let cpu = std::env::var(PENDING_V3_CPU_PREFIX).expect("pending v3 CPU prefix");
    LLC_LOCK_PREFIX_OVERRIDE
        .with(|slot| *slot.borrow_mut() = Some(llc.to_string_lossy().into_owned()));
    CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(cpu));
    true
}

#[test]
#[ignore]
fn pending_exec_v3_export_process_helper() {
    if !install_pending_v3_prefixes() {
        return;
    }
    let affinity_cpu = *host_allowed_cpus()
        .first()
        .expect("pending v3 exporter has an allowed CPU");
    let final_claim = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [affinity_cpu],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Exclusive,
    );
    let granted_claim = final_claim.clone();
    let coordinator_claim = final_claim.clone();
    let pending = protocol::register_intent_for_preparation(
        final_claim.clone(),
        final_claim,
        move |_| Ok(Some(granted_claim.clone())),
        move |_| Ok(Some(coordinator_claim.clone())),
    )
    .expect("register selected-final pending v3 admission");
    let (prepared_cpu, _, original_affinity) = pending
        .preparation_affinity_handoff_parts()
        .expect("read pending v3 affinity");
    assert_eq!(prepared_cpu, affinity_cpu);
    let metadata = format!(
        "{prepared_cpu}|{}",
        original_affinity
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
    );
    let handoff = protocol::prepare_pending_exec_handoff(&pending, metadata.as_bytes())
        .expect("prepare pending v3 exec handoff");
    let seals = unsafe { libc::fcntl(handoff.descriptor_fd_for_tests(), libc::F_GET_SEALS) };
    assert_eq!(
        seals & (libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE),
        libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE,
    );

    let mut command =
        std::process::Command::new(std::env::current_exe().expect("current test executable"));
    command.args([
        "--ignored",
        "--exact",
        "vmm::host_topology::tests::protocol::pending_exec_v3_import_process_helper",
        "--nocapture",
        "--test-threads=1",
    ]);
    handoff.configure_exec(&mut command);
    use std::os::unix::process::CommandExt;
    let error = command.exec();
    panic!("pending v3 exec consumer failed: {error}");
}

#[test]
#[ignore]
fn pending_exec_v3_import_process_helper() {
    if !install_pending_v3_prefixes() {
        return;
    }
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let reader_barrier = barrier.clone();
    let reader = std::thread::spawn(move || {
        reader_barrier.wait();
        for _ in 0..1_000 {
            assert!(std::env::var_os(protocol::EXEC_HANDOFF_ENV).is_some());
        }
    });
    barrier.wait();
    let mappings_before_import = protocol::ticket_shared_mapping_build_count_for_tests();
    let imported = protocol::take_pending_exec_handoff()
        .expect("validate pending v3 handoff while another thread reads the environment")
        .expect("wrapper supplied pending v3 handoff");
    let mappings_after_import = protocol::ticket_shared_mapping_build_count_for_tests();
    assert_eq!(
        mappings_after_import - mappings_before_import,
        1,
        "exec import must reconstruct exactly one validated retained mapping",
    );
    reader.join().expect("environment reader thread");
    assert!(
        protocol::take_pending_exec_handoff()
            .expect("second pending v3 read")
            .is_none(),
        "one inherited descriptor must be consumed exactly once",
    );

    let metadata = String::from_utf8(imported.metadata).expect("pending v3 metadata is UTF-8");
    let (affinity_cpu, original) = metadata.split_once('|').expect("pending v3 metadata shape");
    let affinity_cpu: usize = affinity_cpu.parse().expect("pending v3 affinity CPU");
    let original = original
        .split(',')
        .map(|cpu| cpu.parse().expect("pending v3 original affinity CPU"))
        .collect::<Vec<usize>>();
    assert_eq!(host_allowed_cpus(), vec![affinity_cpu]);

    let mut pending = imported.pending;
    let (_, imported_watch) = pending
        .pending_claim_watch_for_tests()
        .expect("read imported selected-intent watch");
    assert_eq!(
        protocol::ticket_shared_mapping_build_count_for_tests(),
        mappings_after_import,
        "validated exec-import mapping must remain retained by the pending owner",
    );
    let (_, preparation_fds) = pending
        .preparation_handoff_parts()
        .expect("imported preparation descriptors");
    let permits = preparation_fds
        .iter()
        .map(|(permit, _)| *permit)
        .collect::<Vec<_>>();
    assert!(
        permits.len() > 1,
        "CPU, memory, and token permits must cross exec"
    );
    let snapshot = protocol::ticket_registry_snapshot_for_tests()
        .expect("read imported PENDING registry record");
    let imported_claim = snapshot
        .iter()
        .map(|(_, _, claim)| claim)
        .find(|claim| claim.cpus.contains(&affinity_cpu) && claim.permits.len() == permits.len())
        .expect("imported PENDING claim remains published");
    assert_eq!(
        imported_claim.permits.iter().copied().collect::<Vec<_>>(),
        permits,
    );
    assert_eq!(
        imported_claim.cpu_mode,
        protocol::ClaimMode::Shared,
        "same-exec handoff must recover the physical CPU-SH preparation owner",
    );
    assert!(imported_watch.cpus.contains(&affinity_cpu));
    assert_eq!(imported_watch.cpu_mode, protocol::ClaimMode::Exclusive);
    assert_eq!(
        imported_watch.admission_class,
        protocol::AdmissionClass::Ordinary,
    );
    assert!(
        crate::flock::try_flock(
            cpu_lock_path(affinity_cpu),
            crate::flock::FlockMode::Exclusive
        )
        .expect("probe inherited affinity flock")
        .is_none(),
    );
    for permit in &permits {
        assert!(
            crate::flock::try_flock(
                permit_lock_path(*permit),
                crate::flock::FlockMode::Exclusive
            )
            .expect("probe inherited preparation permit")
            .is_none(),
            "preparation permit {permit} did not survive exec",
        );
    }

    pending
        .restore_preparation_affinity()
        .expect("restore pre-exec affinity after import");
    assert_eq!(host_allowed_cpus(), original);
    let exact = resource_claim_with_modes(
        &[],
        LlcLockMode::Shared,
        &[affinity_cpu],
        crate::flock::FlockMode::Shared,
    );
    let target = protocol::canonical_lock_order_with_modes(
        &[],
        crate::flock::FlockMode::Shared,
        &[affinity_cpu],
        crate::flock::FlockMode::Shared,
    );
    let work =
        protocol::activate_pending_ticket(pending, exact.clone(), exact.clone(), None, |probe| {
            let designated = probe.designated().clone();
            let reusable = probe.clone_reusable_permits()?;
            probe.try_acquire(&designated, || {
                acquire_resources_with_permits_granted_reusing(
                    &[],
                    LlcLockMode::Shared,
                    &[affinity_cpu],
                    crate::flock::FlockMode::Shared,
                    &[],
                    &reusable,
                )
            })
        })
        .expect("activate imported PENDING admission");
    let acquired = match work {
        protocol::TicketWork::Acquired(acquired) => acquired,
        protocol::TicketWork::Coordinator(coordinator) => {
            match protocol::acquire_as_coordinator(coordinator, |held| {
                if let Some(locks) = held.probe_complete_if_ready(&exact, &target)? {
                    Ok(protocol::CoordinatorStep::Complete {
                        claim: exact.clone(),
                        value: locks,
                    })
                } else {
                    Ok(protocol::CoordinatorStep::Waiting {
                        claim: exact.clone(),
                    })
                }
            })
            .expect("coordinate imported exact activation")
            {
                protocol::CoordinatorOutcome::Acquired(acquired) => acquired,
                protocol::CoordinatorOutcome::Prepared(_) => {
                    panic!("imported exact activation prepared a VM intent")
                }
                protocol::CoordinatorOutcome::Aborted { reason } => {
                    panic!("imported exact activation aborted: {reason}")
                }
            }
        }
    };
    assert_eq!(
        protocol::registered_claim_snapshot(&exact)
            .expect("read activated exact claim")
            .cpu_holder_count(affinity_cpu)
            .expect("activated CPU holder count"),
        1,
    );
    drop(acquired);
    drop(
        crate::flock::try_flock(
            cpu_lock_path(affinity_cpu),
            crate::flock::FlockMode::Exclusive,
        )
        .expect("probe released exact CPU")
        .expect("exact CPU flock must release with the imported owner"),
    );
    assert_eq!(
        protocol::registered_claim_snapshot(&exact)
            .expect("read released exact claim")
            .cpu_holder_count(affinity_cpu)
            .expect("released CPU holder count"),
        0,
    );
    std::fs::write(
        std::env::var_os(PENDING_V3_MARKER).expect("pending v3 completion marker"),
        b"ok",
    )
    .expect("publish pending v3 completion");
}

#[test]
fn pending_exec_v3_preserves_preparation_and_exact_lifecycle_across_real_exec() {
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE
        .with(|slot| slot.borrow().clone())
        .expect("parent LLC prefix");
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE
        .with(|slot| slot.borrow().clone())
        .expect("parent CPU prefix");
    let temp = tempfile::tempdir().expect("pending v3 test directory");
    let marker = temp.path().join("complete");
    let output =
        std::process::Command::new(std::env::current_exe().expect("current test executable"))
            .args([
                "--ignored",
                "--exact",
                "vmm::host_topology::tests::protocol::pending_exec_v3_export_process_helper",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(PENDING_V3_LLC_PREFIX, llc_prefix)
            .env(PENDING_V3_CPU_PREFIX, cpu_prefix)
            .env(PENDING_V3_MARKER, &marker)
            .output()
            .expect("run pending v3 exec process");
    assert!(
        output.status.success(),
        "pending-v3 process failed: status={} stdout={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(
        std::fs::read(marker).expect("read pending v3 marker"),
        b"ok"
    );
}

thread_local! {
    static TICKET_HELPER_LOGS:
        std::cell::RefCell<std::collections::BTreeMap<u32, (
            String,
            std::path::PathBuf,
            std::path::PathBuf,
            std::path::PathBuf,
        )>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}

fn wait_for_ticket_fs_state<T>(
    watch_dir: &std::path::Path,
    context: &str,
    mut observe: impl FnMut() -> Option<T>,
) -> T {
    use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
    use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
    use std::os::fd::AsFd;

    let inotify = Inotify::init(InitFlags::IN_CLOEXEC | InitFlags::IN_NONBLOCK)
        .unwrap_or_else(|error| panic!("{context}: initialize inotify: {error}"));
    inotify
        .add_watch(
            watch_dir,
            AddWatchFlags::IN_CREATE
                | AddWatchFlags::IN_MOVED_TO
                | AddWatchFlags::IN_CLOSE_WRITE
                | AddWatchFlags::IN_MODIFY
                | AddWatchFlags::IN_DELETE,
        )
        .unwrap_or_else(|error| {
            panic!(
                "{context}: watch helper synchronization directory {}: {error}",
                watch_dir.display(),
            )
        });
    loop {
        if let Some(value) = observe() {
            return value;
        }
        let mut fds = [PollFd::new(inotify.as_fd(), PollFlags::POLLIN)];
        match poll(&mut fds, PollTimeout::NONE) {
            Ok(_) => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(error) => panic!("{context}: wait for helper filesystem event: {error}"),
        }
        match inotify.read_events() {
            Ok(_) | Err(nix::errno::Errno::EAGAIN) | Err(nix::errno::Errno::EINTR) => {}
            Err(error) => panic!("{context}: drain helper filesystem events: {error}"),
        }
    }
}

fn ticket_claim(cpus: &[usize]) -> protocol::ClaimSet {
    protocol::ClaimSet::new(
        std::iter::empty(),
        cpus.iter().copied(),
        crate::flock::FlockMode::Shared,
    )
}

fn try_ticket_candidate(
    cpus: &[usize],
) -> anyhow::Result<protocol::ProbeOutcome<Vec<std::os::fd::OwnedFd>>> {
    let mut locks = Vec::with_capacity(cpus.len());
    for &cpu in cpus {
        match crate::flock::try_flock_with_witness(
            cpu_lock_path(cpu),
            crate::flock::FlockMode::Exclusive,
        )? {
            crate::flock::TryFlockOutcome::Acquired(lock) => locks.push(lock),
            crate::flock::TryFlockOutcome::Contended(witness) => {
                drop(locks);
                return Ok(protocol::ProbeOutcome::Contended(
                    protocol::ContentionEvidence {
                        blocker: protocol::ResourceKey::Cpu(cpu),
                        mode: crate::flock::FlockMode::Exclusive,
                        _witness: witness,
                    },
                ));
            }
        }
    }
    Ok(protocol::ProbeOutcome::Acquired(locks))
}

/// Self-exec helper for the registry tests below. Each process publishes one
/// monotonic ticket and either bypasses on an exact compatible candidate or,
/// as coordinator, probes only its primary exact candidate.
#[test]
#[ignore]
fn ticket_registry_process_helper() {
    let Some(candidate_text) = std::env::var_os(TICKET_HELPER_CANDIDATES) else {
        return;
    };
    let service_tid_path =
        std::path::PathBuf::from(std::env::var_os(TICKET_HELPER_SERVICE_TID).unwrap());
    // SAFETY: `SYS_gettid` has no pointer arguments.
    let service_tid = unsafe { libc::syscall(libc::SYS_gettid) };
    assert!(service_tid > 0 && service_tid <= u32::MAX as libc::c_long);
    std::fs::write(&service_tid_path, service_tid.to_string())
        .expect("publish helper test-worker TID");
    let candidates: Vec<Vec<usize>> = candidate_text
        .to_string_lossy()
        .split(';')
        .map(|candidate| {
            candidate
                .split(',')
                .filter(|value| !value.is_empty())
                .map(|value| value.parse().expect("numeric helper CPU"))
                .collect()
        })
        .collect();
    assert!(
        !candidates.is_empty(),
        "helper needs at least one candidate"
    );
    let llc_prefix = std::env::var(TICKET_HELPER_LLC_PREFIX).expect("helper LLC prefix");
    let cpu_prefix = std::env::var(TICKET_HELPER_CPU_PREFIX).expect("helper CPU prefix");
    LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(llc_prefix));
    CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(cpu_prefix));
    if std::env::var_os(TICKET_HELPER_FORCE_OBSERVER_NONE).is_some() {
        protocol::force_holder_observer_unavailable_for_tests();
    }

    let acquired_path = std::path::PathBuf::from(std::env::var_os(TICKET_HELPER_ACQUIRED).unwrap());
    let probed_path = std::path::PathBuf::from(std::env::var_os(TICKET_HELPER_PROBED).unwrap());
    let release_path = std::path::PathBuf::from(std::env::var_os(TICKET_HELPER_RELEASE).unwrap());
    let disable_bypass = std::env::var_os(TICKET_HELPER_DISABLE_BYPASS).is_some();
    let probe_barrier = std::env::var_os(TICKET_HELPER_PROBE_BARRIER_DIR).map(|path| {
        (
            std::path::PathBuf::from(path),
            std::env::var(TICKET_HELPER_PROBE_BARRIER_COUNT)
                .expect("probe barrier count")
                .parse::<usize>()
                .expect("numeric probe barrier count"),
        )
    });
    let before_probe_gate =
        std::env::var_os(TICKET_HELPER_BEFORE_PROBE_GATE).map(std::path::PathBuf::from);
    let retain_final_candidate = std::env::var_os(TICKET_HELPER_RETAIN_FINAL_CANDIDATE).is_some();
    let after_acquire_gate =
        std::env::var_os(TICKET_HELPER_AFTER_ACQUIRE_GATE).map(std::path::PathBuf::from);
    let after_acquire_entered =
        std::env::var_os(TICKET_HELPER_AFTER_ACQUIRE_ENTERED).map(std::path::PathBuf::from);
    let before_coordinator_gate =
        std::env::var_os(TICKET_HELPER_BEFORE_COORDINATOR_GATE).map(std::path::PathBuf::from);
    let coordinator_entered =
        std::env::var_os(TICKET_HELPER_COORDINATOR_ENTERED).map(std::path::PathBuf::from);
    let coordinator_waiting =
        std::env::var_os(TICKET_HELPER_COORDINATOR_WAITING).map(std::path::PathBuf::from);
    let claims: Vec<_> = candidates
        .iter()
        .map(|candidate| ticket_claim(candidate))
        .collect();
    let watch_claim = ticket_claim(&candidates.iter().flatten().copied().collect::<Vec<_>>());

    let queue =
        protocol::register_ticket_or_acquire(claims[0].clone(), watch_claim, None, |probe| {
            let mut probe_log = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&probed_path)
                .expect("open helper probe log");
            std::io::Write::write_all(&mut probe_log, b"probe\n").expect("publish helper probe");
            if let Some((barrier_dir, expected)) = &probe_barrier {
                std::fs::create_dir_all(barrier_dir).expect("create probe barrier");
                std::fs::write(barrier_dir.join(std::process::id().to_string()), b"entered")
                    .expect("enter probe barrier");
                wait_for_ticket_fs_state(barrier_dir, "probe barrier", || {
                    let entered = std::fs::read_dir(barrier_dir)
                        .expect("read probe barrier")
                        .count();
                    (entered >= *expected).then_some(())
                });
            }
            if let Some(gate) = &before_probe_gate {
                wait_for_ticket_path(gate);
            }
            if disable_bypass {
                return Ok(None);
            }
            let index = claims
                .iter()
                .position(|claim| claim == probe.designated())
                .expect("helper designation must be one declared candidate");
            if let Some(locks) =
                probe.try_acquire(&claims[index], || try_ticket_candidate(&candidates[index]))?
            {
                if let Some(entered) = &after_acquire_entered {
                    std::fs::write(entered, b"entered").expect("publish acquired-probe entry");
                }
                if let Some(gate) = &after_acquire_gate {
                    wait_for_ticket_path(gate);
                }
                return Ok(Some((index, locks)));
            }
            let next = if retain_final_candidate && index + 1 == claims.len() {
                index
            } else {
                (index + 1) % claims.len()
            };
            probe.reserve(&claims[next])?;
            Ok(None)
        })
        .expect("helper queue");
    if let Some(path) = std::env::var_os(TICKET_HELPER_MAPPING_COUNT) {
        std::fs::write(
            path,
            protocol::ticket_shared_mapping_build_count_for_tests().to_string(),
        )
        .expect("publish helper retained mapping count");
    }

    let (index, locks) = match queue {
        protocol::TicketWork::Acquired(acquired) => {
            acquired.split_map(|(index, locks)| (index, locks))
        }
        protocol::TicketWork::Coordinator(coordinator) => {
            if let Some(entered) = &coordinator_entered {
                std::fs::write(entered, b"entered").expect("publish pre-watch coordinator entry");
            }
            if let Some(gate) = &before_coordinator_gate {
                wait_for_ticket_path(gate);
            }
            let target = protocol::canonical_lock_order(
                &[],
                crate::flock::FlockMode::Shared,
                &candidates[0],
            );
            let claim = claims[0].clone();
            let outcome = protocol::acquire_as_coordinator(coordinator, |held| {
                let mut probe_log = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&probed_path)
                    .expect("open coordinator probe log");
                std::io::Write::write_all(&mut probe_log, b"probe\n")
                    .expect("publish coordinator probe");
                if let Some(locks) = held.probe_complete_if_ready(&claim, &target)? {
                    if let Some(entered) = &after_acquire_entered {
                        std::fs::write(entered, b"entered")
                            .expect("publish coordinator acquired-probe entry");
                    }
                    if let Some(gate) = &after_acquire_gate {
                        wait_for_ticket_path(gate);
                    }
                    Ok(protocol::CoordinatorStep::Complete {
                        claim: claim.clone(),
                        value: locks,
                    })
                } else {
                    if let Some(waiting) = &coordinator_waiting {
                        std::fs::write(waiting, b"waiting")
                            .expect("publish coordinator waiting entry");
                    }
                    Ok(protocol::CoordinatorStep::Waiting {
                        claim: claim.clone(),
                    })
                }
            })
            .expect("helper coordinator acquire");
            match outcome {
                protocol::CoordinatorOutcome::Acquired(locks) => (0, locks),
                protocol::CoordinatorOutcome::Prepared(_) => {
                    panic!("helper coordinator prepared a VM intent")
                }
                protocol::CoordinatorOutcome::Aborted { reason } => {
                    panic!("helper coordinator aborted: {reason}")
                }
            }
        }
    };
    std::fs::write(&acquired_path, index.to_string()).expect("publish helper acquisition");
    wait_for_ticket_path(&release_path);
    drop(locks);
}

struct TicketChild {
    child: std::cell::RefCell<Option<std::process::Child>>,
    label: String,
    pid: u32,
    acquired: std::path::PathBuf,
    probed: std::path::PathBuf,
    release: std::path::PathBuf,
    stdout: std::path::PathBuf,
    stderr: std::path::PathBuf,
    mapping_count: Option<std::path::PathBuf>,
}

#[derive(Default)]
struct TicketSpawnOptions<'a> {
    disable_bypass: bool,
    crash_point: Option<&'a str>,
    force_observer_none: bool,
    probe_barrier: Option<(&'a std::path::Path, usize)>,
    before_probe_gate: Option<&'a std::path::Path>,
    retain_final_candidate: bool,
    after_acquire_gate: Option<(&'a std::path::Path, &'a std::path::Path)>,
    before_coordinator_gate: Option<(&'a std::path::Path, &'a std::path::Path)>,
    coordinator_waiting: Option<&'a std::path::Path>,
    record_mapping_count: bool,
    retained_futex_wait_marker: Option<&'a std::path::Path>,
}

impl TicketChild {
    fn spawn(
        marker_dir: &std::path::Path,
        label: &str,
        candidates: &str,
        disable_bypass: bool,
    ) -> Self {
        Self::spawn_with_options(
            marker_dir,
            label,
            candidates,
            TicketSpawnOptions {
                disable_bypass,
                ..TicketSpawnOptions::default()
            },
        )
    }

    fn spawn_crashing(
        marker_dir: &std::path::Path,
        label: &str,
        candidates: &str,
        crash_point: &str,
    ) -> Self {
        Self::spawn_with_options(
            marker_dir,
            label,
            candidates,
            TicketSpawnOptions {
                crash_point: Some(crash_point),
                ..TicketSpawnOptions::default()
            },
        )
    }

    fn spawn_probe_barrier(
        marker_dir: &std::path::Path,
        label: &str,
        candidates: &str,
        barrier_dir: &std::path::Path,
        barrier_count: usize,
    ) -> Self {
        Self::spawn_with_options(
            marker_dir,
            label,
            candidates,
            TicketSpawnOptions {
                probe_barrier: Some((barrier_dir, barrier_count)),
                ..TicketSpawnOptions::default()
            },
        )
    }

    fn spawn_before_probe_gate(
        marker_dir: &std::path::Path,
        label: &str,
        candidates: &str,
        gate: &std::path::Path,
    ) -> Self {
        Self::spawn_with_options(
            marker_dir,
            label,
            candidates,
            TicketSpawnOptions {
                before_probe_gate: Some(gate),
                ..TicketSpawnOptions::default()
            },
        )
    }

    fn spawn_before_probe_gate_retain_final(
        marker_dir: &std::path::Path,
        label: &str,
        candidates: &str,
        gate: &std::path::Path,
    ) -> Self {
        Self::spawn_with_options(
            marker_dir,
            label,
            candidates,
            TicketSpawnOptions {
                before_probe_gate: Some(gate),
                retain_final_candidate: true,
                ..TicketSpawnOptions::default()
            },
        )
    }

    fn spawn_after_acquire_gate(
        marker_dir: &std::path::Path,
        label: &str,
        candidates: &str,
        gate: &std::path::Path,
        entered: &std::path::Path,
    ) -> Self {
        Self::spawn_with_options(
            marker_dir,
            label,
            candidates,
            TicketSpawnOptions {
                after_acquire_gate: Some((gate, entered)),
                ..TicketSpawnOptions::default()
            },
        )
    }

    fn spawn_before_coordinator_gate(
        marker_dir: &std::path::Path,
        label: &str,
        candidates: &str,
        gate: &std::path::Path,
        entered: &std::path::Path,
    ) -> Self {
        Self::spawn_with_options(
            marker_dir,
            label,
            candidates,
            TicketSpawnOptions {
                before_coordinator_gate: Some((gate, entered)),
                ..TicketSpawnOptions::default()
            },
        )
    }

    fn spawn_recording_retained_wait(
        marker_dir: &std::path::Path,
        label: &str,
        candidates: &str,
        futex_wait_marker: &std::path::Path,
    ) -> Self {
        Self::spawn_with_options(
            marker_dir,
            label,
            candidates,
            TicketSpawnOptions {
                record_mapping_count: true,
                retained_futex_wait_marker: Some(futex_wait_marker),
                ..TicketSpawnOptions::default()
            },
        )
    }

    fn spawn_with_options(
        marker_dir: &std::path::Path,
        label: &str,
        candidates: &str,
        options: TicketSpawnOptions<'_>,
    ) -> Self {
        let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE
            .with(|slot| slot.borrow().clone())
            .expect("parent LLC prefix");
        let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE
            .with(|slot| slot.borrow().clone())
            .expect("parent CPU prefix");
        let acquired = marker_dir.join(format!("{label}.acquired"));
        let probed = marker_dir.join(format!("{label}.probed"));
        let release = marker_dir.join(format!("{label}.release"));
        let service_tid = marker_dir.join(format!("{label}.service-tid"));
        let stdout = marker_dir.join(format!("{label}.stdout"));
        let stderr = marker_dir.join(format!("{label}.stderr"));
        let mapping_count = options
            .record_mapping_count
            .then(|| marker_dir.join(format!("{label}.mapping-count")));
        let mut command =
            std::process::Command::new(std::env::current_exe().expect("current test executable"));
        command
            .args([
                "--ignored",
                "--exact",
                "vmm::host_topology::tests::protocol::ticket_registry_process_helper",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(TICKET_HELPER_LLC_PREFIX, llc_prefix)
            .env(TICKET_HELPER_CPU_PREFIX, cpu_prefix)
            .env(TICKET_HELPER_CANDIDATES, candidates)
            .env(TICKET_HELPER_ACQUIRED, &acquired)
            .env(TICKET_HELPER_PROBED, &probed)
            .env(TICKET_HELPER_RELEASE, &release)
            .env(TICKET_HELPER_SERVICE_TID, &service_tid)
            .stdout(std::process::Stdio::from(
                std::fs::File::create(&stdout).expect("create helper stdout log"),
            ))
            .stderr(std::process::Stdio::from(
                std::fs::File::create(&stderr).expect("create helper stderr log"),
            ));
        if options.disable_bypass {
            command.env(TICKET_HELPER_DISABLE_BYPASS, "1");
        }
        if let Some(crash_point) = options.crash_point {
            command.env("KTSTR_TEST_REGISTRY_CRASH_POINT", crash_point);
        }
        if options.force_observer_none {
            command.env(TICKET_HELPER_FORCE_OBSERVER_NONE, "1");
        }
        if let Some((barrier_dir, count)) = options.probe_barrier {
            command
                .env(TICKET_HELPER_PROBE_BARRIER_DIR, barrier_dir)
                .env(TICKET_HELPER_PROBE_BARRIER_COUNT, count.to_string());
        }
        if let Some(gate) = options.before_probe_gate {
            command.env(TICKET_HELPER_BEFORE_PROBE_GATE, gate);
        }
        if options.retain_final_candidate {
            command.env(TICKET_HELPER_RETAIN_FINAL_CANDIDATE, "1");
        }
        if let Some((gate, entered)) = options.after_acquire_gate {
            command
                .env(TICKET_HELPER_AFTER_ACQUIRE_GATE, gate)
                .env(TICKET_HELPER_AFTER_ACQUIRE_ENTERED, entered);
        }
        if let Some((gate, entered)) = options.before_coordinator_gate {
            command
                .env(TICKET_HELPER_BEFORE_COORDINATOR_GATE, gate)
                .env(TICKET_HELPER_COORDINATOR_ENTERED, entered);
        }
        if let Some(waiting) = options.coordinator_waiting {
            command.env(TICKET_HELPER_COORDINATOR_WAITING, waiting);
        }
        if let Some(mapping_count) = &mapping_count {
            command.env(TICKET_HELPER_MAPPING_COUNT, mapping_count);
        }
        if let Some(marker) = options.retained_futex_wait_marker {
            command.env(RETAINED_FUTEX_WAIT_MARKER, marker);
        }
        let child = command.spawn().expect("spawn ticket helper");
        let pid = child.id();
        TICKET_HELPER_LOGS.with(|logs| {
            logs.borrow_mut().insert(
                pid,
                (
                    label.to_owned(),
                    stdout.clone(),
                    stderr.clone(),
                    service_tid.clone(),
                ),
            );
        });
        Self {
            child: std::cell::RefCell::new(Some(child)),
            label: label.to_owned(),
            pid,
            acquired,
            probed,
            release,
            stdout,
            stderr,
            mapping_count,
        }
    }

    fn wait_for_probe(&self) {
        self.wait_for_path(&self.probed, "probe marker");
    }

    fn probe_count(&self) -> usize {
        std::fs::read_to_string(&self.probed)
            .map(|contents| contents.lines().count())
            .unwrap_or(0)
    }

    fn wait_for_probe_count(&self, expected: usize) {
        self.wait_for_observation("probe-count marker", || {
            (self.probe_count() >= expected).then_some(())
        });
    }

    fn wait_for_acquired(&self) {
        self.wait_for_path(&self.acquired, "acquired marker");
    }

    fn wait_for_mapping_count(&self, expected: usize) {
        let path = self
            .mapping_count
            .as_ref()
            .expect("ticket helper did not record retained mapping count");
        self.wait_for_observation("retained mapping count", || {
            let count = std::fs::read_to_string(path).ok()?.parse::<usize>().ok()?;
            (count == expected).then_some(())
        });
    }

    fn release_and_wait(self) {
        std::fs::write(&self.release, b"release").expect("release ticket helper");
        // Releasing the protocol worker only finishes the ignored helper test.
        // Libtest still has to hand the result back to its harness thread and
        // terminate the process, so account service from the whole helper
        // process while waiting for Child::try_wait rather than watching the
        // worker TID after its work is already complete.
        let services = [TestTaskService::process(self.pid)];
        let status =
            wait_with_external_task_service("released ticket-helper exit", &services, || {
                Ok(self.try_status())
            })
            .unwrap_or_else(|error| {
                self.terminate_bounded();
                panic!(
                    "ticket helper {} did not exit after release: {error:#}; {}",
                    self.pid,
                    self.diagnostics(),
                );
            });
        self.child.borrow_mut().take();
        assert!(
            status.success(),
            "ticket helper {} failed with {status}; {}",
            self.pid,
            self.diagnostics(),
        );
    }

    fn diagnostics(&self) -> String {
        let stdout = std::fs::read_to_string(&self.stdout)
            .unwrap_or_else(|error| format!("<read {}: {error}>", self.stdout.display()));
        let stderr = std::fs::read_to_string(&self.stderr)
            .unwrap_or_else(|error| format!("<read {}: {error}>", self.stderr.display()));
        format!(
            "label={:?} pid={} stdout={stdout:?} stderr={stderr:?}",
            self.label, self.pid
        )
    }

    fn try_status(&self) -> Option<std::process::ExitStatus> {
        self.child
            .borrow_mut()
            .as_mut()
            .and_then(|child| child.try_wait().expect("poll ticket helper"))
    }

    fn assert_running(&self, waiting_for: &str) {
        if let Some(status) = self.try_status() {
            panic!(
                "ticket helper {} exited with {status} while waiting for {waiting_for}; {}",
                self.pid,
                self.diagnostics(),
            );
        }
    }

    fn wait_for_observation<T>(&self, context: &str, mut observe: impl FnMut() -> Option<T>) -> T {
        // A queued child's transition may require service from its
        // coordinator or a predecessor before this endpoint can publish it.
        // Track every live helper, but charge each independently so adding
        // participants cannot amplify one transition's CPU-service budget.
        let services = live_ticket_helper_services();
        wait_with_external_task_service(context, &services, || {
            if let Some(value) = observe() {
                return Ok(Some(value));
            }
            if let Some(status) = self.try_status() {
                anyhow::bail!(
                    "ticket helper {} exited with {status} while waiting for {context}; {}",
                    self.pid,
                    self.diagnostics(),
                );
            }
            Ok(None)
        })
        .unwrap_or_else(|error| {
            let pids = TICKET_HELPER_LOGS.with(|logs| {
                logs.borrow()
                    .keys()
                    .copied()
                    .filter(|pid| std::path::Path::new(&format!("/proc/{pid}")).exists())
                    .collect::<Vec<_>>()
            });
            let helpers = ticket_helper_diagnostics(&pids);
            let registry = protocol::ticket_registry_diagnostics_for_tests()
                .unwrap_or_else(|error| format!("<unavailable: {error:#}>"));
            panic!(
                "ticket helper {:?} ({}) failed to reach {context}: {error:#}; {}; \
                 all helpers: {helpers}; registry: {registry}",
                self.label,
                self.pid,
                self.diagnostics()
            )
        })
    }

    fn wait_for_path(&self, path: &std::path::Path, marker: &str) {
        self.wait_for_observation(marker, || path.exists().then_some(()));
    }

    fn terminate_bounded(&self) {
        {
            let mut child = self.child.borrow_mut();
            let Some(child) = child.as_mut() else {
                // A successful wait already reaped the helper and cleared the
                // slot. In particular, Drop after release_and_wait must not
                // pay the two-second kill fallback a second time.
                return;
            };
            let _ = child.kill();
        }
        let service = TestTaskService::process(self.pid);
        let _ = wait_with_external_task_service(
            "killed ticket-helper reap",
            std::slice::from_ref(&service),
            || Ok(self.try_status()),
        );
        // `kill(2)` has already made forward progress independent of the
        // helper. Never turn test cleanup failure into a second panic.
        self.child.borrow_mut().take();
    }

    fn kill_and_wait(&self) {
        self.terminate_bounded();
    }

    fn wait_for_injected_crash(&self) {
        let services = [TestTaskService::process(self.pid)];
        let status = wait_with_external_task_service("injected registry crash", &services, || {
            Ok(self.try_status())
        })
        .unwrap_or_else(|error| {
            self.terminate_bounded();
            panic!(
                "ticket helper {} did not reach its injected registry crash: {error:#}; {}",
                self.pid,
                self.diagnostics(),
            )
        });
        assert_eq!(
            status.code(),
            Some(86),
            "helper must stop at the requested registry crash point: {status}; {}",
            self.diagnostics(),
        );
        self.child.borrow_mut().take();
    }
}

impl Drop for TicketChild {
    fn drop(&mut self) {
        self.kill_and_wait();
        // Worker threads are reused across libtest cases. Keeping a reaped
        // helper keyed only by PID lets rapid process churn alias a later,
        // unrelated process and poisons live_ticket_helper_services().
        TICKET_HELPER_LOGS.with(|logs| {
            logs.borrow_mut().remove(&self.pid);
        });
    }
}

fn wait_for_ticket_path(path: &std::path::Path) {
    let parent = path.parent().expect("ticket helper marker parent");
    wait_for_ticket_fs_state(parent, "ticket helper marker", || {
        path.exists().then_some(())
    });
}

fn ticket_helper_diagnostics(pids: &[u32]) -> String {
    TICKET_HELPER_LOGS.with(|logs| {
        let logs = logs.borrow();
        pids.iter()
            .map(|pid| {
                logs.get(pid)
                    .map(|(label, stdout, stderr, _)| {
                        let stdout = std::fs::read_to_string(stdout).unwrap_or_else(|error| {
                            format!("<read {}: {error}>", stdout.display())
                        });
                        let stderr = std::fs::read_to_string(stderr).unwrap_or_else(|error| {
                            format!("<read {}: {error}>", stderr.display())
                        });
                        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))
                            .unwrap_or_else(|error| format!("<unavailable: {error}>"));
                        format!(
                            "label={label:?} pid={pid} stat={stat:?} \
                             stdout={stdout:?} stderr={stderr:?}"
                        )
                    })
                    .unwrap_or_else(|| format!("pid={pid} helper logs unavailable"))
            })
            .collect::<Vec<_>>()
            .join("; ")
    })
}

fn ticket_helper_services(pids: &[u32]) -> Vec<TestTaskService> {
    TICKET_HELPER_LOGS.with(|logs| {
        let logs = logs.borrow();
        pids.iter()
            .filter(|pid| std::path::Path::new(&format!("/proc/{pid}")).exists())
            .filter_map(|pid| {
                logs.get(pid).map(|(_, _, _, service_tid)| {
                    TestTaskService::helper_thread(*pid, service_tid.clone())
                })
            })
            .collect()
    })
}

fn live_ticket_helper_services() -> Vec<TestTaskService> {
    TICKET_HELPER_LOGS.with(|logs| {
        logs.borrow()
            .iter()
            .filter(|(pid, _)| std::path::Path::new(&format!("/proc/{pid}")).exists())
            .map(|(pid, (_, _, _, service_tid))| {
                TestTaskService::helper_thread(*pid, service_tid.clone())
            })
            .collect()
    })
}

fn wait_for_ticket_pids(expected: &[u32]) {
    for &awaited_pid in expected {
        let already_published = protocol::ticket_registry_snapshot_for_tests()
            .expect("ticket registry snapshot")
            .iter()
            .any(|(_, pid, _)| *pid == awaited_pid);
        if already_published {
            continue;
        }
        let services = live_ticket_helper_services();
        wait_with_external_task_service("ticket registry publication", &services, || {
            let pid = awaited_pid;
            let state = std::fs::read_to_string(format!("/proc/{pid}/stat"))
                .ok()
                .and_then(|stat| {
                    stat.rsplit_once(')')
                        .and_then(|(_, tail)| tail.trim_start().as_bytes().first().copied())
                });
            if state.is_none_or(|state| matches!(state, b'Z' | b'X')) {
                let diagnostics = ticket_helper_diagnostics(&[pid]);
                panic!(
                    "ticket helper {pid} exited before publishing the expected registry state; \
                     {diagnostics}"
                );
            }
            let published = protocol::ticket_registry_snapshot_for_tests()
                .expect("ticket registry snapshot")
                .iter()
                .any(|(_, pid, _)| *pid == awaited_pid);
            Ok(published.then_some(()))
        })
        .unwrap_or_else(|error| {
            panic!(
                "ticket {awaited_pid} was not published: {error:#}; {}",
                ticket_helper_diagnostics(&[awaited_pid]),
            )
        });
    }
    let actual: Vec<u32> = protocol::ticket_registry_snapshot_for_tests()
        .expect("ticket registry snapshot")
        .into_iter()
        .map(|(_, pid, _)| pid)
        .collect();
    assert_eq!(
        actual,
        expected,
        "ticket order mismatch; {}",
        ticket_helper_diagnostics(expected),
    );
}

fn wait_for_ticket_claim(child: &TicketChild, expected: &protocol::ClaimSet) {
    child.wait_for_observation("expected registry claim", || {
        let claim = protocol::ticket_registry_snapshot_for_tests()
            .expect("ticket registry snapshot")
            .into_iter()
            .find_map(|(_, pid, claim)| (pid == child.pid).then_some(claim));
        if claim.as_ref() == Some(expected) {
            return Some(());
        }
        None
    });
}

#[test]
fn disjoint_ticket_bypasses_a_blocked_coordinator() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);

    let disjoint = TicketChild::spawn(markers.path(), "disjoint", "2", false);
    disjoint.wait_for_probe();
    disjoint.wait_for_acquired();
    assert!(
        !coordinator.acquired.exists(),
        "blocked coordinator must still be waiting"
    );
    disjoint.release_and_wait();

    drop(blocker);
    coordinator.wait_for_acquired();
    coordinator.release_and_wait();
}

#[test]
fn retained_shared_mapping_wakes_waiter_across_processes() {
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let coordinator_blocker =
        crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
            .expect("open coordinator blocker")
            .expect("hold coordinator blocker");
    let waiter_blocker =
        crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
            .expect("open waiter blocker")
            .expect("hold waiter blocker");
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    coordinator.wait_for_probe();

    let futex_wait_marker = markers.path().join("waiter.futex-wait");
    let waiter = TicketChild::spawn_recording_retained_wait(
        markers.path(),
        "retained-map-waiter",
        "2",
        &futex_wait_marker,
    );
    waiter.wait_for_path(&futex_wait_marker, "retained FUTEX_WAIT entry");
    waiter.wait_for_observation("retained-map waiter publication", || {
        protocol::ticket_is_waiting_for_tests(waiter.pid)
            .expect("read retained-map waiter state")
            .then_some(())
    });

    // This close wakes the coordinator process, which publishes GRANTED and
    // FUTEX_WAKEs the different process already sleeping on its retained
    // read-only MAP_SHARED record view.
    drop(waiter_blocker);
    waiter.wait_for_acquired();
    waiter.wait_for_observation("successful retained FUTEX_WAKE", || {
        (std::fs::read(&futex_wait_marker).ok()?.as_slice() == b"woken").then_some(())
    });
    waiter.wait_for_mapping_count(1);
    waiter.release_and_wait();

    drop(coordinator_blocker);
    coordinator.wait_for_acquired();
    coordinator.release_and_wait();
}

#[test]
fn live_parked_coordinator_lease_transfers_and_successors_drain() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let coordinator_gate = markers.path().join("resume-original-coordinator");
    let coordinator_entered = markers.path().join("original-coordinator-parked");
    let coordinator = TicketChild::spawn_before_coordinator_gate(
        markers.path(),
        "parked-coordinator",
        "1",
        &coordinator_gate,
        &coordinator_entered,
    );
    coordinator.wait_for_path(&coordinator_entered, "parked coordinator marker");

    // The first successor deliberately targets the parked ticket's exact CPU:
    // lease transfer must suspend the stale logical prefix while the physical
    // flock continues to enforce the requested mode.
    let first = TicketChild::spawn(markers.path(), "first-successor", "1", false);
    wait_for_ticket_pids(&[coordinator.pid, first.pid]);
    let second = TicketChild::spawn(markers.path(), "second-successor", "2", false);
    wait_for_ticket_pids(&[coordinator.pid, first.pid, second.pid]);
    protocol::expire_coordinator_lease_for_tests().expect("expire coordinator lease");
    protocol::churn_registry_generation_for_tests(256)
        .expect("simulate registration/cancellation generation churn");

    first.wait_for_acquired();
    second.wait_for_acquired();
    coordinator.assert_running("successor drain under transferred coordinator lease");
    assert!(
        !coordinator.acquired.exists(),
        "the parked original coordinator must remain alive and unacquired while successors drain"
    );
    first.release_and_wait();
    second.release_and_wait();

    std::fs::write(&coordinator_gate, b"resume").expect("resume original coordinator");
    coordinator.wait_for_acquired();
    coordinator.release_and_wait();
}

#[test]
fn coordinator_election_skips_a_live_granted_head_for_disjoint_waiting_work() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let first_blocker =
        crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
            .expect("open first coordinator blocker")
            .expect("hold first coordinator blocker");
    let third_blocker =
        crate::flock::try_flock(cpu_lock_path(3), crate::flock::FlockMode::Exclusive)
            .expect("open third-ticket blocker")
            .expect("hold third-ticket blocker");

    let first = TicketChild::spawn(markers.path(), "first", "1", false);
    wait_for_ticket_pids(&[first.pid]);

    let granted_gate = markers.path().join("commit-granted-head");
    let granted_entered = markers.path().join("granted-head-acquired-lock");
    let granted = TicketChild::spawn_after_acquire_gate(
        markers.path(),
        "granted-head",
        "2",
        &granted_gate,
        &granted_entered,
    );
    granted.wait_for_path(&granted_entered, "granted-head physical acquisition");

    let successor_gate = markers.path().join("start-successor-coordinator");
    let successor_entered = markers.path().join("successor-became-coordinator");
    let successor = TicketChild::spawn_before_coordinator_gate(
        markers.path(),
        "waiting-successor",
        "3",
        &successor_gate,
        &successor_entered,
    );
    wait_for_ticket_pids(&[first.pid, granted.pid, successor.pid]);

    drop(first_blocker);
    first.wait_for_acquired();
    successor.wait_for_path(
        &successor_entered,
        "coordinator election behind a live granted head",
    );
    assert!(
        !granted.acquired.exists(),
        "the earlier granted callback must remain live and uncommitted during successor election",
    );

    drop(third_blocker);
    std::fs::write(&successor_gate, b"run").expect("release successor coordinator gate");
    successor.wait_for_acquired();
    assert!(
        !granted.acquired.exists(),
        "disjoint coordinator work must complete without waiting for the earlier granted callback",
    );
    successor.release_and_wait();

    std::fs::write(&granted_gate, b"commit").expect("release granted-head commit gate");
    granted.wait_for_acquired();
    granted.release_and_wait();
    first.release_and_wait();
}

#[test]
fn coordinator_behind_a_granted_predecessor_does_not_probe_its_conflicting_claim() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let first_blocker =
        crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
            .expect("open first coordinator blocker")
            .expect("hold first coordinator blocker");

    let first = TicketChild::spawn(markers.path(), "first", "1", false);
    wait_for_ticket_pids(&[first.pid]);

    let granted_probe_gate = markers.path().join("run-granted-head-probe");
    let granted = TicketChild::spawn_before_probe_gate(
        markers.path(),
        "granted-head",
        "2",
        &granted_probe_gate,
    );
    granted.wait_for_probe();

    let successor_gate = markers.path().join("start-conflicting-coordinator");
    let successor_entered = markers
        .path()
        .join("conflicting-successor-became-coordinator");
    let successor_waiting = markers
        .path()
        .join("conflicting-successor-remained-waiting");
    let successor = TicketChild::spawn_with_options(
        markers.path(),
        "conflicting-successor",
        "2",
        TicketSpawnOptions {
            before_coordinator_gate: Some((&successor_gate, &successor_entered)),
            coordinator_waiting: Some(&successor_waiting),
            ..TicketSpawnOptions::default()
        },
    );
    wait_for_ticket_pids(&[first.pid, granted.pid, successor.pid]);

    drop(first_blocker);
    first.wait_for_acquired();
    successor.wait_for_path(
        &successor_entered,
        "conflicting coordinator election behind a granted head",
    );
    std::fs::write(&successor_gate, b"run").expect("release conflicting coordinator gate");
    successor.wait_for_path(
        &successor_waiting,
        "mode-aware predecessor rejection before physical probing",
    );
    assert!(
        !successor.acquired.exists(),
        "the later coordinator must not acquire a claim reserved by its granted predecessor",
    );

    std::fs::write(&granted_probe_gate, b"probe").expect("release granted-head physical probe");
    granted.wait_for_acquired();
    assert!(
        !successor.acquired.exists(),
        "the earlier granted callback must retain admission priority",
    );
    granted.release_and_wait();

    successor.wait_for_acquired();
    successor.release_and_wait();
    first.release_and_wait();
}

#[test]
fn coordinator_retries_physical_success_if_an_earlier_replan_claims_its_target() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let first_blocker =
        crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
            .expect("open first coordinator blocker")
            .expect("hold first coordinator blocker");
    let target_blocker =
        crate::flock::try_flock(cpu_lock_path(3), crate::flock::FlockMode::Exclusive)
            .expect("open later coordinator target blocker")
            .expect("hold later coordinator target blocker");

    let first = TicketChild::spawn(markers.path(), "first", "1", false);
    wait_for_ticket_pids(&[first.pid]);

    let replan_gate = markers.path().join("publish-earlier-replan");
    let earlier =
        TicketChild::spawn_before_probe_gate(markers.path(), "earlier-replan", "2;3", &replan_gate);
    earlier.wait_for_probe();
    wait_for_ticket_claim(&earlier, &ticket_claim(&[2]));

    let coordinator_gate = markers.path().join("run-later-coordinator");
    let coordinator_entered = markers.path().join("later-became-coordinator");
    let stale_commit_gate = markers.path().join("commit-later-physical-success");
    let physical_success = markers.path().join("later-physical-success");
    let later = TicketChild::spawn_with_options(
        markers.path(),
        "later-coordinator",
        "3",
        TicketSpawnOptions {
            after_acquire_gate: Some((&stale_commit_gate, &physical_success)),
            before_coordinator_gate: Some((&coordinator_gate, &coordinator_entered)),
            ..TicketSpawnOptions::default()
        },
    );
    wait_for_ticket_pids(&[first.pid, earlier.pid, later.pid]);

    drop(first_blocker);
    first.wait_for_acquired();
    later.wait_for_path(
        &coordinator_entered,
        "later coordinator election behind a live REPLAN callback",
    );

    drop(target_blocker);
    std::fs::write(&coordinator_gate, b"run").expect("release later coordinator gate");
    later.wait_for_path(
        &physical_success,
        "later coordinator physical success before registry commit",
    );

    std::fs::write(&replan_gate, b"replan").expect("release earlier REPLAN callback");
    let target_claim = ticket_claim(&[3]);
    wait_for_ticket_claim(&earlier, &target_claim);

    std::fs::write(&stale_commit_gate, b"commit").expect("release stale later-coordinator commit");
    earlier.wait_for_acquired();
    assert!(
        !later.acquired.exists(),
        "the later coordinator must drop physical success from a stale predecessor snapshot",
    );

    earlier.release_and_wait();
    later.wait_for_acquired();
    later.release_and_wait();
    first.release_and_wait();
}

#[test]
fn release_before_successor_installs_watch_is_sampled_on_its_first_schedule() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker_x = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .expect("open pre-watch blocker")
        .expect("hold pre-watch blocker");
    let blocker_y = crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
        .expect("open successor blocker")
        .expect("hold successor blocker");
    let first_gate = markers.path().join("commit-first-acquire");
    let first_acquired = markers.path().join("first-real-lock-acquired");
    let first = TicketChild::spawn_after_acquire_gate(
        markers.path(),
        "first-coordinator",
        "1",
        &first_gate,
        &first_acquired,
    );
    wait_for_ticket_pids(&[first.pid]);
    first.wait_for_probe();

    let successor_gate = markers.path().join("install-successor-watch");
    let successor_entered = markers.path().join("successor-elected-before-watch");
    let successor = TicketChild::spawn_before_coordinator_gate(
        markers.path(),
        "successor",
        "2",
        &successor_gate,
        &successor_entered,
    );
    wait_for_ticket_pids(&[first.pid, successor.pid]);
    let compatible = TicketChild::spawn(markers.path(), "compatible", "1", false);
    wait_for_ticket_pids(&[first.pid, successor.pid, compatible.pid]);

    drop(blocker_x);
    first.wait_for_path(&first_acquired, "first coordinator real-lock marker");
    std::fs::write(&first_gate, b"commit").expect("commit first coordinator acquisition");
    first.wait_for_acquired();
    successor.wait_for_path(&successor_entered, "pre-watch coordinator marker");

    // The first coordinator committed X and published it busy while removing
    // its ticket. Its returned X fd now closes after the Y coordinator is
    // elected but before that process calls LockDirWatch::new(). B's mandatory
    // first planner step cannot mask this race because Y remains externally
    // wedged; only the first post-watch aggregate observation can mark X free
    // and grant the compatible C ticket.
    first.release_and_wait();
    std::fs::write(&successor_gate, b"install watch").expect("release successor pre-watch gate");
    compatible.wait_for_acquired();
    assert!(
        !successor.acquired.exists(),
        "the successor must remain blocked on Y while post-watch observation grants C on X",
    );
    compatible.release_and_wait();
    drop(blocker_y);
    successor.wait_for_acquired();
    successor.release_and_wait();
}

#[test]
fn ticket_death_before_successor_installs_watch_is_reconciled_promptly() {
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker_x = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .expect("open first-coordinator blocker")
        .expect("hold first-coordinator blocker");
    let blocker_y = crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
        .expect("open successor-coordinator blocker")
        .expect("hold successor-coordinator blocker");
    let blocker_z = crate::flock::try_flock(cpu_lock_path(3), crate::flock::FlockMode::Exclusive)
        .expect("open fenced-resource blocker")
        .expect("hold fenced-resource blocker");

    let first_gate = markers.path().join("commit-first-acquire");
    let first_acquired = markers.path().join("first-real-lock-acquired");
    let first = TicketChild::spawn_after_acquire_gate(
        markers.path(),
        "first-coordinator",
        "1",
        &first_gate,
        &first_acquired,
    );
    wait_for_ticket_pids(&[first.pid]);
    first.wait_for_probe();

    let successor_gate = markers.path().join("install-successor-watch");
    let successor_entered = markers.path().join("successor-elected-before-watch");
    let successor = TicketChild::spawn_before_coordinator_gate(
        markers.path(),
        "successor-coordinator",
        "2",
        &successor_gate,
        &successor_entered,
    );
    wait_for_ticket_pids(&[first.pid, successor.pid]);

    // C publishes an exact Z claim. The coordinator's availability snapshot
    // already proves Z busy, so no speculative granted callback is needed.
    // D is later and conflicts with C, so C's live claim must fence D.
    let dead_predecessor = TicketChild::spawn(markers.path(), "dead-predecessor", "3", false);
    wait_for_ticket_pids(&[first.pid, successor.pid, dead_predecessor.pid]);
    let fenced_successor = TicketChild::spawn(markers.path(), "fenced-successor", "3", false);
    wait_for_ticket_pids(&[
        first.pid,
        successor.pid,
        dead_predecessor.pid,
        fenced_successor.pid,
    ]);
    assert!(
        !fenced_successor.probed.exists() && !fenced_successor.acquired.exists(),
        "the earlier live Z claim must initially fence its conflicting successor",
    );

    drop(blocker_x);
    first.wait_for_path(&first_acquired, "first coordinator real-lock marker");
    std::fs::write(&first_gate, b"commit").expect("commit first coordinator acquisition");
    first.wait_for_acquired();
    successor.wait_for_path(&successor_entered, "pre-watch coordinator marker");

    // B is now the elected coordinator but has not called LockDirWatch::new().
    // Kill the non-coordinator C in that exact gap, then make Z physically
    // available. Neither close can be present in B's future inotify queue.
    let dead_pid = dead_predecessor.pid;
    dead_predecessor.kill_and_wait();
    drop(blocker_z);
    first.release_and_wait();
    protocol::defer_liveness_maintenance_for_tests()
        .expect("keep the 30-second periodic sweep out of the regression");
    std::fs::write(&successor_gate, b"install watch").expect("release successor pre-watch gate");

    // B's first post-watch pass re-observes Z as free. C's stale claim still
    // fences D until the shared short reconciliation deadline prunes C; the
    // periodic 30-second maintenance sweep is deliberately deferred above,
    // while the helpers below are bounded by service delivered to every
    // process that can produce their next transition.
    fenced_successor.wait_for_acquired();
    assert!(
        !successor.acquired.exists(),
        "the successor coordinator must remain blocked on Y while D acquires Z",
    );
    assert!(
        protocol::ticket_registry_snapshot_for_tests()
            .expect("ticket registry snapshot after reconciliation")
            .iter()
            .all(|(_, pid, _)| *pid != dead_pid),
        "the pre-watch dead ticket must be removed by the short shared reconciliation",
    );
    fenced_successor.release_and_wait();

    drop(blocker_y);
    successor.wait_for_acquired();
    successor.release_and_wait();
}

#[test]
fn disjoint_granted_callbacks_probe_concurrently_without_registry_ex_convoy() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker_one = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);

    let barrier = markers.path().join("concurrent-probes");
    let second = TicketChild::spawn_probe_barrier(markers.path(), "second", "2", &barrier, 2);
    let third = TicketChild::spawn_probe_barrier(markers.path(), "third", "3", &barrier, 2);
    for child in [&second, &third] {
        let entered = barrier.join(child.pid.to_string());
        child.wait_for_observation("concurrent helper probe barrier", || {
            entered.exists().then_some(())
        });
    }
    assert_eq!(
        std::fs::read_dir(&barrier)
            .expect("read concurrent probe barrier")
            .count(),
        2,
        "both disjoint granted callbacks must overlap at the probe barrier",
    );

    second.wait_for_acquired();
    third.wait_for_acquired();
    second.release_and_wait();
    third.release_and_wait();
    drop(blocker_one);
    coordinator.wait_for_acquired();
    coordinator.release_and_wait();
}

#[test]
fn invalidated_inflight_grant_drops_its_acquired_payload_before_commit() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker_one = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let blocker_two = crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);

    let earlier_gate = markers.path().join("release-earlier-probe");
    // Rotate once from blocked CPU 2 onto CPU 3, then retain CPU 3 if its
    // first probe races the still-gated stale payload. The stable exact claim
    // is the FIFO invariant this test needs while that payload is released.
    let earlier = TicketChild::spawn_before_probe_gate_retain_final(
        markers.path(),
        "earlier",
        "2;3",
        &earlier_gate,
    );
    earlier.wait_for_probe();

    let later_gate = markers.path().join("release-later-acquired-probe");
    let later_entered = markers.path().join("later-acquired-probe-entered");
    let later = TicketChild::spawn_after_acquire_gate(
        markers.path(),
        "later",
        "3",
        &later_gate,
        &later_entered,
    );
    later.wait_for_path(&later_entered, "post-acquire gate marker");

    std::fs::write(&earlier_gate, b"release").expect("release earlier alternative probe");
    let revocation_services = ticket_helper_services(&[earlier.pid, coordinator.pid]);
    wait_with_external_task_service(
        "later in-flight grant revocation",
        &revocation_services,
        || {
            coordinator.assert_running("later in-flight grant revocation");
            earlier.assert_running("later in-flight grant revocation");
            later.assert_running("later in-flight grant revocation");
            Ok(protocol::ticket_is_revoked_for_tests(later.pid)
                .expect("read later ticket state")
                .then_some(()))
        },
    )
    .unwrap_or_else(|error| {
        let registry = protocol::ticket_registry_diagnostics_for_tests()
            .unwrap_or_else(|error| format!("<unavailable: {error:#}>"));
        panic!(
            "later in-flight grant was not revoked: {error:#}; {}; registry: {registry}",
            ticket_helper_diagnostics(&[coordinator.pid, earlier.pid, later.pid]),
        )
    });
    std::fs::write(&later_gate, b"release").expect("release stale acquired probe");

    earlier.wait_for_acquired();
    assert!(
        !later.acquired.exists(),
        "a payload acquired under a revoked later grant must be dropped, not committed",
    );
    earlier.release_and_wait();
    later.wait_for_acquired();
    later.release_and_wait();
    drop(blocker_two);
    drop(blocker_one);
    coordinator.wait_for_acquired();
    coordinator.release_and_wait();
}

#[test]
fn failed_inflight_probe_blocks_at_the_current_resource_epoch() {
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker_one = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);
    let gate = markers.path().join("release-stale-epoch-probe");
    let waiter = TicketChild::spawn_before_probe_gate(markers.path(), "waiter", "2", &gate);
    waiter.wait_for_probe();

    let claim = ticket_claim(&[2]);
    let blocker_two = crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .expect("block the in-flight waiter probe");
    let blocker_two =
        protocol::publish_acquired(&claim, blocker_two).expect("publish first external hold");
    let before = protocol::resource_epoch_for_tests().expect("resource epoch before release");
    drop(blocker_two);
    wait_with_delivered_service("resource epoch advance after real holder release", || {
        Ok((protocol::resource_epoch_for_tests()? != before).then_some(()))
    })
    .expect("coordinator must advance the resource epoch for a real holder release");
    let blocker_two = crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .expect("re-block waiter after epoch transition");
    // Leave the replacement as an external, unregistered flock. A current-v20
    // HELD publication would authoritatively revoke the in-flight grant before
    // its callback returned, bypassing the stale-negative-evidence path this
    // test is meant to pin.
    std::fs::write(&gate, b"release").expect("release stale-epoch probe");
    let blocked = wait_with_delivered_service("current blocked-serial publication", || {
        waiter.assert_running("current blocked-serial publication");
        Ok(protocol::ticket_blocked_at_current_serial_for_tests(waiter.pid)?.then_some(()))
    });
    if let Err(error) = blocked {
        panic!(
            "failed in-flight probe did not publish its blocker at the current serial: \
             {error:#}; {}",
            waiter.diagnostics(),
        );
    }
    assert_eq!(
        waiter.probe_count(),
        1,
        "a failed probe must commit blocked_epoch=current and avoid an immediate same-claim regrant",
    );

    drop(blocker_two);
    waiter.wait_for_probe_count(2);
    waiter.wait_for_acquired();
    waiter.release_and_wait();
    drop(blocker_one);
    coordinator.wait_for_acquired();
    coordinator.release_and_wait();
}

#[test]
fn fast_probe_prunes_a_killed_sole_ticket_before_fencing() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);
    coordinator.kill_and_wait();
    drop(blocker);

    let claim = ticket_claim(&[1]);
    let outcome = protocol::with_registry_fence(&claim, || try_ticket_candidate(&[1]))
        .expect("stale aggregate recovery");
    match outcome {
        protocol::RegistryFence::Ran {
            value: protocol::ProbeOutcome::Acquired(locks),
            ..
        } => drop(locks),
        protocol::RegistryFence::Ran {
            value: protocol::ProbeOutcome::Contended(_),
            ..
        }
        | protocol::RegistryFence::Ran {
            value: protocol::ProbeOutcome::Unavailable,
            ..
        } => panic!("resource was released, so the nonwaiting probe must acquire"),
        protocol::RegistryFence::Fenced => {
            panic!("killed sole ticket must be pruned before the fast path is fenced")
        }
    }
}

#[test]
fn observer_winning_first_registry_lock_leaves_layout_choice_to_the_registrant() {
    let _prefixes = LockPrefixesGuard::new();
    assert!(
        protocol::observer_preserves_uninitialized_header_for_tests()
            .expect("observe missing, zero-length, and zeroed registry headers"),
        "a non-authoritative observer must leave every unpublished header representation untouched",
    );

    let sparse_llc = 8191usize;
    let claim = protocol::ClaimSet::new(
        [sparse_llc],
        std::iter::empty(),
        crate::flock::FlockMode::Exclusive,
    );
    let coordinator =
        match protocol::register_ticket_or_acquire(claim.clone(), claim.clone(), None, |_| {
            Ok::<Option<()>, anyhow::Error>(None)
        })
        .expect("let the authoritative sparse-LLC registrant choose the host layout")
        {
            protocol::TicketWork::Coordinator(coordinator) => coordinator,
            protocol::TicketWork::Acquired(_) => {
                panic!("first sparse-LLC registrant must become coordinator")
            }
        };
    let snapshot =
        protocol::ticket_registry_snapshot_for_tests().expect("read sparse-LLC registration");
    assert_eq!(
        snapshot.len(),
        1,
        "sparse-LLC registration must publish exactly one ticket",
    );
    assert_eq!(
        snapshot[0].2, claim,
        "the observer must not freeze the registry below the sparse LLC id",
    );
    drop(coordinator);
}

fn register_sparse_first_ticket_after_observation() {
    let sparse_llc = 8191usize;
    let claim = protocol::ClaimSet::new(
        [sparse_llc],
        std::iter::empty(),
        crate::flock::FlockMode::Exclusive,
    );
    let coordinator =
        match protocol::register_ticket_or_acquire(claim.clone(), claim.clone(), None, |_| {
            Ok::<Option<()>, anyhow::Error>(None)
        })
        .expect("let the authoritative sparse-LLC registrant choose the host layout")
        {
            protocol::TicketWork::Coordinator(coordinator) => coordinator,
            protocol::TicketWork::Acquired(_) => {
                panic!("first sparse-LLC registrant must become coordinator")
            }
        };
    let snapshot =
        protocol::ticket_registry_snapshot_for_tests().expect("read sparse-LLC registration");
    assert_eq!(snapshot.len(), 1);
    assert_eq!(
        snapshot[0].2, claim,
        "an observer must not freeze the registry below the first registrant's sparse LLC id",
    );
    drop(coordinator);
}

#[test]
fn aggregate_snapshot_of_zeroed_header_leaves_sparse_layout_to_first_registrant() {
    let _prefixes = LockPrefixesGuard::new();
    protocol::prepare_zeroed_uninitialized_header_for_tests()
        .expect("prepare interrupted zeroed header");
    let low = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    let snapshot =
        protocol::registered_claim_snapshot(&low).expect("observe zeroed aggregate header");
    assert!(
        !snapshot.conflicts(&low).expect("query empty aggregate"),
        "an unpublished zeroed header must read as an empty registry",
    );
    register_sparse_first_ticket_after_observation();
}

#[test]
fn fast_fence_of_zeroed_header_leaves_sparse_layout_to_first_registrant() {
    let _prefixes = LockPrefixesGuard::new();
    protocol::prepare_zeroed_uninitialized_header_for_tests()
        .expect("prepare interrupted zeroed header");
    let low = protocol::ClaimSet::new(
        std::iter::empty(),
        [1usize],
        crate::flock::FlockMode::Exclusive,
    );
    match protocol::with_registry_fence(&low, || Ok::<_, anyhow::Error>(17usize))
        .expect("run fast probe through zeroed header")
    {
        protocol::RegistryFence::Ran { value, watched } => {
            assert_eq!(value, 17);
            assert!(
                !watched,
                "an unpublished zeroed header cannot contain a watched resource",
            );
        }
        protocol::RegistryFence::Fenced => {
            panic!("an unpublished zeroed header must be an empty fence")
        }
    }
    register_sparse_first_ticket_after_observation();
}

#[test]
fn interrupted_initializer_is_rebuilt_from_the_zero_length_header() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let creator = TicketChild::spawn_crashing(
        markers.path(),
        "initializer",
        "1",
        "initialize_before_publish",
    );
    creator.wait_for_injected_crash();

    let replacement = TicketChild::spawn(markers.path(), "replacement", "1", false);
    replacement.wait_for_acquired();
    replacement.release_and_wait();
    assert_eq!(
        protocol::registry_initializer_temp_count_for_tests()
            .expect("count stale registry initializers"),
        0,
        "the replacement opener must reclaim the killed creator's unpublished tempfile",
    );
}

#[test]
fn register_crash_after_record_before_counts_is_repaired() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let registering = TicketChild::spawn_crashing(
        markers.path(),
        "registering",
        "1",
        "register_record_before_counts",
    );
    registering.wait_for_injected_crash();

    let replacement = TicketChild::spawn(markers.path(), "replacement", "1", false);
    replacement.wait_for_acquired();
    replacement.release_and_wait();
}

#[test]
fn register_crash_before_state_publish_leaves_no_partial_active_ticket() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let registering = TicketChild::spawn_crashing(
        markers.path(),
        "registering-state",
        "1",
        "register_record_before_state_publish",
    );
    registering.wait_for_injected_crash();

    let replacement = TicketChild::spawn(markers.path(), "replacement", "1", false);
    replacement.wait_for_acquired();
    replacement.release_and_wait();
    assert!(
        protocol::ticket_registry_snapshot_for_tests()
            .expect("snapshot after partial registration recovery")
            .is_empty(),
        "dirty repair must discard a record killed before active-state publication",
    );
}

#[test]
fn remove_crash_after_counts_before_free_is_repaired() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let removing =
        TicketChild::spawn_crashing(markers.path(), "removing", "1", "remove_counts_before_free");
    // The v20 HELD lifecycle removes its registry record only after the
    // physical reservation is released. Let the helper pass its normal
    // release barrier so the injected crash observes that production ordering.
    std::fs::write(&removing.release, b"release").expect("release crash-test reservation");
    removing.wait_for_injected_crash();

    let replacement = TicketChild::spawn(markers.path(), "replacement", "1", false);
    replacement.wait_for_acquired();
    replacement.release_and_wait();
}

#[test]
fn replace_crash_after_counts_before_record_is_repaired() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker_one = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let blocker_two = crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);
    let replacing = TicketChild::spawn_crashing(
        markers.path(),
        "replacing",
        "2;3",
        "replace_counts_before_record",
    );
    replacing.wait_for_injected_crash();

    let replacement = TicketChild::spawn(markers.path(), "replacement", "3", false);
    replacement.wait_for_acquired();
    replacement.release_and_wait();
    coordinator.kill_and_wait();
    drop(blocker_two);
    drop(blocker_one);
}

#[test]
fn replace_crash_before_state_publish_cannot_reserve_a_partial_claim() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker_one = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let blocker_two = crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);
    let replacing = TicketChild::spawn_crashing(
        markers.path(),
        "replacing-state",
        "2;3",
        "replace_record_before_state_publish",
    );
    replacing.wait_for_injected_crash();

    let replacement = TicketChild::spawn(markers.path(), "replacement", "3", false);
    replacement.wait_for_acquired();
    replacement.release_and_wait();
    coordinator.kill_and_wait();
    drop(blocker_two);
    drop(blocker_one);
}

#[test]
fn grant_crash_after_state_before_wake_still_makes_progress() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn_crashing(
        markers.path(),
        "coordinator",
        "1",
        "grant_state_before_wake",
    );
    wait_for_ticket_pids(&[coordinator.pid]);
    let waiter = TicketChild::spawn(markers.path(), "waiter", "2", false);
    coordinator.wait_for_injected_crash();
    waiter.wait_for_acquired();
    waiter.release_and_wait();
    drop(blocker);
}

#[test]
fn replan_cursor_crash_before_wake_recovers_across_processes() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker_one = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let blocker_two = crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let blocker_three =
        crate::flock::try_flock(cpu_lock_path(3), crate::flock::FlockMode::Exclusive)
            .unwrap()
            .unwrap();
    let coordinator = TicketChild::spawn_crashing(
        markers.path(),
        "replan-crashing-coordinator",
        "1",
        "replan_state_and_cursor_before_wake",
    );
    wait_for_ticket_pids(&[coordinator.pid]);
    let waiter = TicketChild::spawn(markers.path(), "replan-waiter", "2;3", false);
    coordinator.wait_for_injected_crash();

    // A new registrant acquires the abandoned EX flock, repairs the torn
    // cursor/state transaction, and proves disjoint work still progresses.
    let recovery = TicketChild::spawn(markers.path(), "replan-recovery", "4", false);
    recovery.wait_for_acquired();
    recovery.release_and_wait();

    drop(blocker_three);
    drop(blocker_two);
    waiter.wait_for_acquired();
    waiter.release_and_wait();
    drop(blocker_one);
}

#[test]
fn granted_acquirer_crash_before_record_clear_is_pruned() {
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);
    let acquired = TicketChild::spawn_crashing(
        markers.path(),
        "acquired",
        "2",
        "granted_acquired_before_clear",
    );
    acquired.wait_for_injected_crash();

    let replacement = TicketChild::spawn(markers.path(), "replacement", "2", false);
    replacement.wait_for_acquired();
    replacement.release_and_wait();
    coordinator.kill_and_wait();
    drop(blocker);
}

#[test]
fn election_crash_after_header_publish_is_repaired_by_the_next_waiter() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);
    let crashing = TicketChild::spawn_crashing(
        markers.path(),
        "crashing-election",
        "1",
        "elect_header_before_state",
    );
    wait_for_ticket_pids(&[coordinator.pid, crashing.pid]);

    coordinator.kill_and_wait();
    crashing.wait_for_injected_crash();

    let replacement = TicketChild::spawn(markers.path(), "replacement", "2", false);
    replacement.wait_for_acquired();
    replacement.release_and_wait();
    drop(blocker);
    assert!(
        protocol::ticket_registry_snapshot_for_tests()
            .expect("registry after election recovery")
            .is_empty(),
        "recovery must prune both crashed tickets and recycle their slots",
    );
}

#[test]
fn clean_coordinator_header_mismatch_is_repaired_by_the_next_registrant() {
    let _prefixes = LockPrefixesGuard::new();
    protocol::exercise_clean_coordinator_mismatch_recovery_for_tests()
        .expect("repair a clean coordinator header/record mismatch");
}

#[test]
fn conflicting_ticket_cannot_bypass_an_earlier_live_claim() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);
    let earlier = TicketChild::spawn(markers.path(), "earlier", "2", true);
    wait_for_ticket_pids(&[coordinator.pid, earlier.pid]);
    earlier.wait_for_probe();
    let conflicting = TicketChild::spawn(markers.path(), "conflicting", "2", false);
    wait_for_ticket_pids(&[coordinator.pid, earlier.pid, conflicting.pid]);
    conflicting.wait_for_observation("conflicting ticket entered the waiting state", || {
        protocol::ticket_is_waiting_for_tests(conflicting.pid)
            .expect("read conflicting ticket state")
            .then_some(())
    });
    assert!(
        !conflicting.probed.exists() && !conflicting.acquired.exists(),
        "a conflicting successor must not even receive a probe grant",
    );

    conflicting.kill_and_wait();
    earlier.kill_and_wait();
    drop(blocker);
    coordinator.wait_for_acquired();
    coordinator.release_and_wait();
}

#[test]
fn every_predecessor_claim_is_respected_while_disjoint_work_passes() {
    let _prefixes = LockPrefixesGuard::new();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);
    let second = TicketChild::spawn(markers.path(), "second", "2", true);
    wait_for_ticket_pids(&[coordinator.pid, second.pid]);
    let third = TicketChild::spawn(markers.path(), "third", "3", true);
    wait_for_ticket_pids(&[coordinator.pid, second.pid, third.pid]);

    let conflicts_third = TicketChild::spawn(markers.path(), "conflict-third", "3", false);
    wait_for_ticket_pids(&[coordinator.pid, second.pid, third.pid, conflicts_third.pid]);

    let disjoint = TicketChild::spawn(markers.path(), "fully-disjoint", "4", false);
    disjoint.wait_for_probe();
    disjoint.wait_for_acquired();
    disjoint.release_and_wait();
    assert!(
        !conflicts_third.probed.exists() && !conflicts_third.acquired.exists(),
        "the second non-coordinator predecessor claim must fence later work",
    );

    conflicts_third.kill_and_wait();
    third.kill_and_wait();
    second.kill_and_wait();
    drop(blocker);
    coordinator.wait_for_acquired();
    coordinator.release_and_wait();
}

#[test]
fn crashed_ticket_is_pruned_before_a_later_compatible_probe() {
    let _prefixes = LockPrefixesGuard::new_real_wake();
    let markers = tempfile::TempDir::new().expect("marker dir");
    let blocker = crate::flock::try_flock(cpu_lock_path(1), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);
    let crashed = TicketChild::spawn(markers.path(), "crashed", "2", true);
    wait_for_ticket_pids(&[coordinator.pid, crashed.pid]);
    crashed.wait_for_probe();
    let crashed_pid = crashed.pid;
    crashed.kill_and_wait();

    let replacement = TicketChild::spawn(markers.path(), "replacement", "2", false);
    replacement.wait_for_probe();
    replacement.wait_for_acquired();
    assert!(
        protocol::ticket_registry_snapshot_for_tests()
            .unwrap()
            .iter()
            .all(|(_, pid, _)| *pid != crashed_pid),
        "the killed ticket's stale registry record must be pruned",
    );
    replacement.release_and_wait();

    drop(blocker);
    coordinator.wait_for_acquired();
    coordinator.release_and_wait();
}
