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

fn cancel_registry_worker(cancelled: &std::sync::atomic::AtomicBool) {
    cancelled.store(true, std::sync::atomic::Ordering::Release);
    let waiter = crate::flock::interruptible_flock_waiter_id();
    if waiter != 0 {
        crate::flock::wake_interruptible_flock_waiter(waiter);
    }
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
            protocol::TicketWork::Acquired(()) => panic!("fresh registry must elect a coordinator"),
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
    let registry_dir = protocol_dir.join("ktstr-acquire-registry-v6");
    let event_dir = protocol_dir.join("ktstr-acquire-events-v6");
    assert!(!registry_dir.exists());
    assert!(!event_dir.exists());

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
        protocol::TicketWork::Acquired(()) => panic!("fresh registry must elect a coordinator"),
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
        protocol::TicketWork::Acquired(()) => panic!("fresh registry must elect a coordinator"),
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
        protocol::TicketWork::Acquired(()) => panic!("fresh registry must elect a coordinator"),
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
        cpu_shared_a.conflicts_with(&cpu_exclusive)
            && cpu_exclusive.conflicts_with(&cpu_shared_a),
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
    let llc_reconstructed = protocol::claim_from_resource_modes_for_tests([(
        protocol::ResourceKey::Llc(7),
        crate::flock::FlockMode::Shared,
    )])
    .expect("reconstruct LLC-only held claim");
    assert_eq!(llc_new, llc_with_modes);
    assert_eq!(llc_new, llc_reconstructed);

    let cpu_shared = protocol::ClaimSet::with_modes(
        std::iter::empty(),
        [9usize],
        crate::flock::FlockMode::Shared,
        crate::flock::FlockMode::Shared,
    );
    let cpu_reconstructed = protocol::claim_from_resource_modes_for_tests([(
        protocol::ResourceKey::Cpu(9),
        crate::flock::FlockMode::Shared,
    )])
    .expect("reconstruct CPU-only held claim");
    assert_eq!(cpu_shared, cpu_reconstructed);

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
fn reconstructed_cpu_holds_accept_uniform_shared_mode_and_reject_mixed_modes() {
    let shared = protocol::claim_from_resource_modes_for_tests([
        (
            protocol::ResourceKey::Cpu(1),
            crate::flock::FlockMode::Shared,
        ),
        (
            protocol::ResourceKey::Cpu(2),
            crate::flock::FlockMode::Shared,
        ),
    ])
    .expect("uniform CPU SH holds form one claim");
    assert_eq!(
        shared,
        protocol::ClaimSet::with_modes(
            std::iter::empty(),
            [1usize, 2usize],
            crate::flock::FlockMode::Shared,
            crate::flock::FlockMode::Shared,
        ),
    );
    let error = protocol::claim_from_resource_modes_for_tests([
        (
            protocol::ResourceKey::Cpu(1),
            crate::flock::FlockMode::Shared,
        ),
        (
            protocol::ResourceKey::Cpu(2),
            crate::flock::FlockMode::Exclusive,
        ),
    ])
    .expect_err("mixed CPU lock modes cannot be represented by one exact claim");
    assert!(error.to_string().contains("mixed CPU lock modes"));
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
    let coordinator = match protocol::register_ticket_or_acquire(
        shared.clone(),
        shared.clone(),
        None,
        |_| Ok::<Option<()>, anyhow::Error>(None),
    )
    .expect("register CPU SH aggregate owner")
    {
        protocol::TicketWork::Coordinator(coordinator) => coordinator,
        protocol::TicketWork::Acquired(()) => panic!("fresh registry must elect a coordinator"),
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
        protocol::TicketWork::Acquired(()) => panic!("fresh registry must elect a coordinator"),
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
        protocol::TicketWork::Acquired(()) => panic!("fresh registry must elect a coordinator"),
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
    let reader = std::thread::spawn(move || {
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
                protocol::TicketWork::Acquired(()) => {
                    panic!("fresh registry must elect a coordinator")
                }
            };
        ready_tx
            .send(())
            .expect("report coordinator registration");
        go_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("receive overlapping-read start");
        let before = protocol::shared_state_read_count_for_tests();
        let result = coordinator.read_state_shared_for_tests();
        let delta = protocol::shared_state_read_count_for_tests() - before;
        result_tx
            .send((result, delta))
            .expect("report overlapping state read");
        drop(coordinator);
    });
    ready_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("coordinator registration must complete");
    let held_reader =
        protocol::hold_registry_shared_for_tests().expect("hold first registry SH reader");
    go_tx.send(()).expect("start overlapping state read");
    let (result, delta) = result_rx
        .recv_timeout(std::time::Duration::from_secs(1))
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
fn targeted_broker_wake_interrupts_one_registry_futex_waiter() {
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
            protocol::TicketWork::Acquired(()) => panic!("fresh registry must elect a coordinator"),
        };

    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|p| p.borrow().clone());
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let worker_claim = claim.clone();
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    let waiter = std::thread::spawn(move || {
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

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let waiter_id = loop {
        let id = crate::flock::interruptible_flock_waiter_id();
        if id != 0 {
            break id;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "registry futex waiter did not publish its cancellation generation"
        );
        std::thread::yield_now();
    };
    let signal_count = crate::flock::primitives::interruptible_flock_broker_signal_count();
    cancelled.store(true, Ordering::Release);
    crate::flock::wake_interruptible_flock_waiter(waiter_id);
    wait_for_broker_signal_after(signal_count);
    let result = result_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("targeted wake must interrupt the registry futex");
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
    let worker = std::thread::spawn(move || {
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

    let first_id = first_id_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("first generation");
    let signal_count = crate::flock::primitives::interruptible_flock_broker_signal_count();
    crate::flock::wake_interruptible_flock_waiter(first_id);
    wait_for_broker_signal_after(signal_count);
    advance_tx.send(()).expect("advance worker");
    let second_id = second_id_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("second generation");
    assert_ne!(
        first_id, second_id,
        "successive registrations need distinct generations"
    );
    assert!(
        poll_rx
            .recv_timeout(std::time::Duration::from_secs(2))
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
    let worker = std::thread::spawn(move || {
        let registration =
            crate::flock::InterruptibleFlockWaiter::register().expect("live registration");
        id_tx
            .send(crate::flock::interruptible_flock_waiter_id())
            .expect("report generation");
        release_rx.recv().expect("release registration");
        drop(registration);
    });
    let waiter_id = id_rx
        .recv_timeout(std::time::Duration::from_secs(2))
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

/// RE-PLAN-ON-WAKE, the falsifiable form (a REQUIREMENT of the
/// regime, not an implementation detail): the coordinator enters the wait
/// wanting resource set X (all candidates busy); a DIFFERENT
/// sufficient set Y frees; the coordinator must acquire Y promptly — never
/// keep waiting on X. Driven through the PRODUCTION default-path
/// acquisition (`KtstrVm::acquire_default_run_locks`) on a synthetic
/// two-LLC host: candidate X (LLC 0 / CPU 0) stays wedged for the
/// whole test; candidate Y (LLC 1 / CPU 1) frees after ~400 ms; the
/// returned plan must be Y's.
#[test]
fn coordinator_replans_to_freed_alternative_candidate() {
    use std::sync::mpsc;
    use std::sync::{Arc, atomic::AtomicBool};

    let _broker = InterruptibleFlockBrokerGuard::start();
    let _prefixes = LockPrefixesGuard::new_real_wake();
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

    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    let start = std::time::Instant::now();
    let worker = std::thread::spawn(move || {
        LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = llc_prefix);
        CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = cpu_prefix);
        let result = crate::vmm::KtstrVm::acquire_default_run_locks_interruptible(
            Some(&host),
            &topo,
            true,
            &worker_cancelled,
        );
        let _ = result_tx.send(result);
    });
    let result = match result_rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(result) => result,
        Err(error) => {
            cancel_registry_worker(&cancelled);
            drop(hold_x);
            let _ = result_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("cancelled default-path worker must unwind");
            worker
                .join()
                .expect("cancelled default-path coordinator worker");
            releaser.join().expect("bounded releaser thread");
            panic!("default-path coordinator did not complete within 10 s: {error}");
        }
    };
    let elapsed = start.elapsed();
    worker
        .join()
        .expect("completed default-path coordinator worker");
    releaser.join().expect("releaser thread");
    drop(hold_x);
    let rl = result.expect("acquisition must complete via the freed alternative");

    let plan = rl
        .pinning_plan
        .as_ref()
        .expect("1:1 pin, not overcommit — both candidates map");
    assert_eq!(
        plan.assignments,
        vec![(0, 1)],
        "the coordinator must have re-planned onto the FREED candidate \
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
    let coordinator_thread = std::thread::spawn(move || {
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
    // Wait boundedly for the coordinator to publish its exact claim. Surface
    // an early worker return immediately instead of sleeping and later
    // misdiagnosing a missing registry record.
    let ready_deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        if !protocol::ticket_registry_snapshot_for_tests()
            .expect("poll coordinator ticket")
            .is_empty()
        {
            break;
        }
        if let Ok(result) = coordinator_rx.try_recv() {
            coordinator_thread
                .join()
                .expect("early coordinator worker return");
            panic!("coordinator returned before blocking: {result:?}");
        }
        if std::time::Instant::now() >= ready_deadline {
            cancel_registry_worker(&cancelled);
            drop(peer_sh);
            let _ = coordinator_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("cancelled fixed-set coordinator must unwind");
            coordinator_thread
                .join()
                .expect("cancelled fixed-set coordinator thread");
            panic!("coordinator did not publish its claim within 2 s");
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    // Small SH cell on a DIFFERENT LLC/CPU: must complete promptly.
    let plan_small = PinningPlan {
        assignments: vec![(0, 51)],
        service_cpu: None,
        llc_indices: vec![11],
        locks: Vec::new(),
    };
    let start = std::time::Instant::now();
    let outcome = acquire_resource_locks_waiting_impl(
        &plan_small.llc_indices,
        LlcLockMode::Shared,
        &[51],
        FlockMode::Exclusive,
        false,
        None,
    )
    .unwrap();
    let elapsed = start.elapsed();
    let (_, locks) = unwrap_acquired(
        outcome,
        Some("disjoint SH cell while the coordinator hungers"),
    );
    assert_eq!(locks.len(), 2);
    // The `Acquired` outcome above already proves work conservation
    // happened at all: strict arrival-order admission would have forced this
    // disjoint-capacity cell to wait behind the coordinator and return
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
             hungering coordinator; elapsed={elapsed:?}",
        );
    }

    // Release the peer; the coordinator must now complete.
    drop(peer_sh);
    let coordinator_outcome = match coordinator_rx.recv_timeout(std::time::Duration::from_secs(10))
    {
        Ok(result) => result.expect("coordinator acquire"),
        Err(error) => {
            cancel_registry_worker(&cancelled);
            let _ = coordinator_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("cancelled fixed-set coordinator must unwind");
            coordinator_thread
                .join()
                .expect("cancelled fixed-set coordinator thread");
            panic!("coordinator did not complete within 10 s of release: {error}");
        }
    };
    coordinator_thread
        .join()
        .expect("completed coordinator thread");
    match coordinator_outcome {
        LockOutcome::Acquired { locks, .. } => assert_eq!(locks.len(), 1),
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
    let worker = std::thread::spawn(move || {
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
                protocol::TicketWork::Acquired(()) => {
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
                protocol::CoordinatorOutcome::Aborted { reason } => {
                    anyhow::bail!("transition-race coordinator aborted: {reason}")
                }
            }
            Ok::<usize, anyhow::Error>(steps)
        })();
        let _ = result_tx.send(result);
    });
    let first_step = first_step_rx.recv_timeout(std::time::Duration::from_secs(2));
    if let Err(error) = first_step {
        cancel_registry_worker(&cancelled);
        drop(blocker);
        let _ = result_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("cancelled transition-race worker must unwind");
        worker
            .join()
            .expect("cancelled transition-race coordinator worker");
        panic!("coordinator did not reach its initial planning step: {error}");
    }
    drop(blocker);
    let steps = match result_rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(result) => result.expect("coordinator transition-race acquire"),
        Err(error) => {
            cancel_registry_worker(&cancelled);
            let _ = result_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("cancelled transition-race worker must unwind");
            worker
                .join()
                .expect("cancelled transition-race coordinator worker");
            panic!("transition-race coordinator did not complete within 10 s: {error}");
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
        (
            observations,
            scans,
            planner_steps,
            generation_changes,
            ex_acquisitions,
        ),
        (0, 0, 0, 0, 0),
        "writable closes on already-free watched resources must be discarded \
         under SH before any registry mutation/EX acquisition, procfs \
         observation, grant scan, or planner execution",
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
    let (scans, shared_granted, exclusive_waiting) =
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
}

#[test]
fn failed_cpu_ex_probe_wakes_only_the_compatible_shared_waiter() {
    let _prefixes = LockPrefixesGuard::new();
    let (
        scans,
        shared_granted,
        exclusive_waiting,
        sh_serial_advanced,
        ex_serial_unchanged,
        shared_woke,
        exclusive_not_woken,
    ) = protocol::exercise_cpu_ex_contention_shared_wake_for_tests()
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
    let (scans, later_granted) =
        protocol::exercise_cpu_shared_commit_improvement_for_tests()
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
        protocol::exercise_cpu_mode_repair_for_tests()
            .expect("exercise dirty CPU-mode repair");
    assert!(
        flexible_preserved && flexible_still_flexible,
        "repair must retain a CPU SH exact claim under its CPU EX watch and mark it REPLAN",
    );
    assert!(
        fixed_preserved && fixed_still_fixed,
        "canonical empty-class modes must keep an exact CPU SH claim fixed after repair",
    );
}

#[test]
fn granted_callbacks_read_one_cached_prefix_without_walking_the_queue() {
    let _prefixes = LockPrefixesGuard::new();
    let waiters = 128usize;
    let (callbacks, prefix_reads, active_list_reads) =
        protocol::exercise_prefix_callback_scaling_for_tests(waiters)
            .expect("exercise cached predecessor prefixes");
    assert_eq!(callbacks, waiters, "each REPLAN ticket must run once");
    assert_eq!(
        prefix_reads, waiters,
        "each callback must copy only its own cached prefix record",
    );
    assert_eq!(
        active_list_reads, 0,
        "callback admission must be independent of active queue depth",
    );
}

#[test]
fn replan_publishes_one_replacement_then_returns_to_coordinator_admission() {
    let _prefixes = LockPrefixesGuard::new();
    let (callbacks, requeued_without_acquire, waiting, replaced, rescan_pending, active_reads) =
        protocol::exercise_one_shot_replacement_for_tests()
            .expect("exercise one-shot replacement");
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
    let (prefix_refreshed, serial_refreshed, candidate_ready, replacement_committed) =
        protocol::exercise_prefix_refresh_after_predecessor_release_for_tests()
            .expect("exercise acquired-predecessor prefix refresh");
    assert!(
        prefix_refreshed && serial_refreshed,
        "a holder-release improvement must refresh both the cached predecessor prefix and its issue serial",
    );
    assert!(
        candidate_ready && replacement_committed,
        "the release must become a usable one-shot replacement without waiting for another event",
    );
}

#[test]
fn callback_cannot_consume_an_improvement_it_did_not_observe() {
    let _prefixes = LockPrefixesGuard::new();
    let (stale_rejected, fresh_seen, replacement_committed, serial_consumed_by_fresh) =
        protocol::exercise_issue_serial_race_for_tests()
            .expect("exercise callback issue-serial race");
    assert!(
        stale_rejected,
        "a callback whose availability snapshot predates an improvement must lose its issuance",
    );
    assert!(
        fresh_seen && replacement_committed && serial_consumed_by_fresh,
        "the already-runnable ticket must immediately use a refreshed snapshot before consuming the improvement serial",
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
    let coordinator_worker = std::thread::spawn(move || {
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
                    held.sweep(&target)?;
                    if held.covers(&target) {
                        Ok(protocol::CoordinatorStep::Complete {
                            claim: claim.clone(),
                            value: held.take(&target),
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
                protocol::CoordinatorOutcome::Aborted { reason } => {
                    anyhow::bail!("granted-path coordinator aborted: {reason}")
                }
            }
            Ok::<(), anyhow::Error>(())
        })();
        let _ = coordinator_tx.send(result);
    });
    ready_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("coordinator must publish before granted waiter starts");

    let waiter_cancelled = Arc::new(AtomicBool::new(false));
    let worker_waiter_cancelled = Arc::clone(&waiter_cancelled);
    let llc_prefix = LLC_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let cpu_prefix = CPU_LOCK_PREFIX_OVERRIDE.with(|slot| slot.borrow().clone());
    let (waiter_tx, waiter_rx) = mpsc::sync_channel(1);
    let waiter_worker = std::thread::spawn(move || {
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
    let locks = match waiter_rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(result) => result.expect("granted waiter must return committed success"),
        Err(error) => {
            cancel_registry_worker(&waiter_cancelled);
            drop(blocker);
            let _ = waiter_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("cancelled granted waiter must unwind");
            waiter_worker.join().expect("cancelled granted waiter");
            let _ = coordinator_rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("released coordinator must complete");
            coordinator_worker
                .join()
                .expect("released coordinator worker");
            panic!("granted waiter did not complete within 10 s: {error}");
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
    match coordinator_rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(result) => result.expect("coordinator completes after blocker release"),
        Err(error) => {
            coordinator_cancelled.store(true, Ordering::Release);
            drop(
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(cpu_lock_path(1))
                    .expect("wake cancelled coordinator"),
            );
            let _ = coordinator_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("cancelled coordinator must unwind");
            coordinator_worker
                .join()
                .expect("cancelled coordinator worker");
            panic!("coordinator did not complete after blocker release: {error}");
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
const TICKET_HELPER_DISABLE_BYPASS: &str = "KTSTR_TEST_TICKET_DISABLE_BYPASS";
const TICKET_HELPER_FORCE_OBSERVER_NONE: &str = "KTSTR_TEST_TICKET_FORCE_OBSERVER_NONE";
const TICKET_HELPER_PROBE_BARRIER_DIR: &str = "KTSTR_TEST_TICKET_PROBE_BARRIER_DIR";
const TICKET_HELPER_PROBE_BARRIER_COUNT: &str = "KTSTR_TEST_TICKET_PROBE_BARRIER_COUNT";
const TICKET_HELPER_BEFORE_PROBE_GATE: &str = "KTSTR_TEST_TICKET_BEFORE_PROBE_GATE";
const TICKET_HELPER_AFTER_ACQUIRE_GATE: &str = "KTSTR_TEST_TICKET_AFTER_ACQUIRE_GATE";
const TICKET_HELPER_AFTER_ACQUIRE_ENTERED: &str = "KTSTR_TEST_TICKET_AFTER_ACQUIRE_ENTERED";
const TICKET_HELPER_BEFORE_COORDINATOR_GATE: &str = "KTSTR_TEST_TICKET_BEFORE_COORDINATOR_GATE";
const TICKET_HELPER_COORDINATOR_ENTERED: &str = "KTSTR_TEST_TICKET_COORDINATOR_ENTERED";

thread_local! {
    static TICKET_HELPER_LOGS:
        std::cell::RefCell<std::collections::BTreeMap<u32, (std::path::PathBuf, std::path::PathBuf)>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
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
/// as coordinator, accumulates only its primary exact candidate.
#[test]
#[ignore]
fn ticket_registry_process_helper() {
    let Some(candidate_text) = std::env::var_os(TICKET_HELPER_CANDIDATES) else {
        return;
    };
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
    let after_acquire_gate =
        std::env::var_os(TICKET_HELPER_AFTER_ACQUIRE_GATE).map(std::path::PathBuf::from);
    let after_acquire_entered =
        std::env::var_os(TICKET_HELPER_AFTER_ACQUIRE_ENTERED).map(std::path::PathBuf::from);
    let before_coordinator_gate =
        std::env::var_os(TICKET_HELPER_BEFORE_COORDINATOR_GATE).map(std::path::PathBuf::from);
    let coordinator_entered =
        std::env::var_os(TICKET_HELPER_COORDINATOR_ENTERED).map(std::path::PathBuf::from);
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
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                loop {
                    let entered = std::fs::read_dir(barrier_dir)
                        .expect("read probe barrier")
                        .count();
                    if entered >= *expected {
                        break;
                    }
                    assert!(
                        std::time::Instant::now() < deadline,
                        "probe barrier timed out with {entered}/{expected} callbacks"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
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
            probe.reserve(&claims[(index + 1) % claims.len()])?;
            Ok(None)
        })
        .expect("helper queue");

    let (index, locks) = match queue {
        protocol::TicketWork::Acquired(acquired) => acquired,
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
                held.sweep(&target)?;
                if held.covers(&target) {
                    if let Some(entered) = &after_acquire_entered {
                        std::fs::write(entered, b"entered")
                            .expect("publish coordinator acquired-probe entry");
                    }
                    if let Some(gate) = &after_acquire_gate {
                        wait_for_ticket_path(gate);
                    }
                    Ok(protocol::CoordinatorStep::Complete {
                        claim: claim.clone(),
                        value: held.take(&target),
                    })
                } else {
                    Ok(protocol::CoordinatorStep::Waiting {
                        claim: claim.clone(),
                    })
                }
            })
            .expect("helper coordinator acquire");
            match outcome {
                protocol::CoordinatorOutcome::Acquired(locks) => (0, locks),
                protocol::CoordinatorOutcome::Aborted { reason } => {
                    panic!("helper coordinator aborted: {reason}")
                }
            }
        }
    };
    std::fs::write(&acquired_path, index.to_string()).expect("publish helper acquisition");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !release_path.exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "helper release barrier timed out"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    drop(locks);
}

struct TicketChild {
    child: std::cell::RefCell<Option<std::process::Child>>,
    pid: u32,
    acquired: std::path::PathBuf,
    probed: std::path::PathBuf,
    release: std::path::PathBuf,
    stdout: std::path::PathBuf,
    stderr: std::path::PathBuf,
}

#[derive(Default)]
struct TicketSpawnOptions<'a> {
    disable_bypass: bool,
    crash_point: Option<&'a str>,
    force_observer_none: bool,
    probe_barrier: Option<(&'a std::path::Path, usize)>,
    before_probe_gate: Option<&'a std::path::Path>,
    after_acquire_gate: Option<(&'a std::path::Path, &'a std::path::Path)>,
    before_coordinator_gate: Option<(&'a std::path::Path, &'a std::path::Path)>,
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

    fn spawn_observerless(marker_dir: &std::path::Path, label: &str, candidates: &str) -> Self {
        Self::spawn_with_options(
            marker_dir,
            label,
            candidates,
            TicketSpawnOptions {
                force_observer_none: true,
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
        let stdout = marker_dir.join(format!("{label}.stdout"));
        let stderr = marker_dir.join(format!("{label}.stderr"));
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
        let child = command.spawn().expect("spawn ticket helper");
        let pid = child.id();
        TICKET_HELPER_LOGS.with(|logs| {
            logs.borrow_mut()
                .insert(pid, (stdout.clone(), stderr.clone()));
        });
        Self {
            child: std::cell::RefCell::new(Some(child)),
            pid,
            acquired,
            probed,
            release,
            stdout,
            stderr,
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
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while self.probe_count() < expected {
            self.assert_running("probe-count marker");
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for helper {} to reach {expected} probes; observed {}; {}",
                self.pid,
                self.probe_count(),
                self.diagnostics(),
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn wait_for_acquired(&self) {
        self.wait_for_path(&self.acquired, "acquired marker");
    }

    fn release_and_wait(self) {
        std::fs::write(&self.release, b"release").expect("release ticket helper");
        let status = self.wait_for_status(std::time::Duration::from_secs(10));
        let status = status.unwrap_or_else(|| {
            self.terminate_bounded();
            panic!(
                "timed out waiting for ticket helper {} after release; {}",
                self.pid,
                self.diagnostics(),
            );
        });
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
        format!("stdout={stdout:?} stderr={stderr:?}")
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

    fn wait_for_path(&self, path: &std::path::Path, marker: &str) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !path.exists() {
            self.assert_running(marker);
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for ticket helper {} {marker} {}; {}",
                self.pid,
                path.display(),
                self.diagnostics(),
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn wait_for_status(&self, timeout: std::time::Duration) -> Option<std::process::ExitStatus> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(status) = self.try_status() {
                self.child.borrow_mut().take();
                return Some(status);
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn terminate_bounded(&self) {
        if let Some(child) = self.child.borrow_mut().as_mut() {
            let _ = child.kill();
        }
        let _ = self.wait_for_status(std::time::Duration::from_secs(2));
        // `kill(2)` has already made forward progress independent of the
        // helper. Never turn test cleanup into another unbounded wait.
        self.child.borrow_mut().take();
    }

    fn kill_and_wait(&self) {
        self.terminate_bounded();
    }

    fn wait_for_injected_crash(&self) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Some(status) = self.try_status() {
                assert_eq!(
                    status.code(),
                    Some(86),
                    "helper must stop at the requested registry crash point: {status}; {}",
                    self.diagnostics(),
                );
                self.child.borrow_mut().take();
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for injected registry crash; {}",
                self.diagnostics(),
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}

impl Drop for TicketChild {
    fn drop(&mut self) {
        self.kill_and_wait();
    }
}

fn wait_for_ticket_path(path: &std::path::Path) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !path.exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for ticket helper marker {}",
            path.display(),
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

fn ticket_helper_diagnostics(pids: &[u32]) -> String {
    TICKET_HELPER_LOGS.with(|logs| {
        let logs = logs.borrow();
        pids.iter()
            .map(|pid| {
                logs.get(pid)
                    .map(|(stdout, stderr)| {
                        let stdout = std::fs::read_to_string(stdout).unwrap_or_else(|error| {
                            format!("<read {}: {error}>", stdout.display())
                        });
                        let stderr = std::fs::read_to_string(stderr).unwrap_or_else(|error| {
                            format!("<read {}: {error}>", stderr.display())
                        });
                        format!("pid={pid} stdout={stdout:?} stderr={stderr:?}")
                    })
                    .unwrap_or_else(|| format!("pid={pid} helper logs unavailable"))
            })
            .collect::<Vec<_>>()
            .join("; ")
    })
}

fn wait_for_ticket_pids(expected: &[u32]) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        for &pid in expected {
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
        }
        let actual: Vec<u32> = protocol::ticket_registry_snapshot_for_tests()
            .expect("ticket registry snapshot")
            .into_iter()
            .map(|(_, pid, _)| pid)
            .collect();
        if actual == expected {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "ticket order mismatch: expected={expected:?} actual={actual:?}; {}",
            ticket_helper_diagnostics(expected),
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
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
    let _prefixes = LockPrefixesGuard::new();
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

    // C first receives a disjoint grant, proves Z busy, and requeues its exact
    // Z claim. D is later and conflicts with C, so C's live claim must fence D.
    let dead_predecessor =
        TicketChild::spawn(markers.path(), "dead-predecessor", "3", false);
    wait_for_ticket_pids(&[first.pid, successor.pid, dead_predecessor.pid]);
    dead_predecessor.wait_for_probe();
    let fenced_successor =
        TicketChild::spawn(markers.path(), "fenced-successor", "3", false);
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
    std::fs::write(&successor_gate, b"install watch")
        .expect("release successor pre-watch gate");

    // B's first post-watch pass re-observes Z as free. C's stale claim still
    // fences D until the shared short reconciliation deadline prunes C; the
    // periodic 30-second maintenance sweep is deliberately deferred above,
    // while every helper wait below has its own ten-second deadline.
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
    let blocker_two = crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .unwrap();
    let blocker_three =
        crate::flock::try_flock(cpu_lock_path(3), crate::flock::FlockMode::Exclusive)
            .unwrap()
            .unwrap();
    let coordinator = TicketChild::spawn(markers.path(), "coordinator", "1", false);
    wait_for_ticket_pids(&[coordinator.pid]);

    let barrier = markers.path().join("concurrent-probes");
    let second = TicketChild::spawn_probe_barrier(markers.path(), "second", "2", &barrier, 2);
    let third = TicketChild::spawn_probe_barrier(markers.path(), "third", "3", &barrier, 2);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let entered = std::fs::read_dir(&barrier)
            .map(|entries| entries.count())
            .unwrap_or(0);
        if entered == 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "disjoint granted callbacks did not overlap; only {entered}/2 entered the barrier"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    drop(blocker_two);
    drop(blocker_three);
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
    let earlier =
        TicketChild::spawn_before_probe_gate(markers.path(), "earlier", "2;3", &earlier_gate);
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
    earlier.wait_for_probe_count(2);
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
    let gate = markers.path().join("release-stale-epoch-probe");
    let waiter = TicketChild::spawn_before_probe_gate(markers.path(), "waiter", "2", &gate);
    waiter.wait_for_probe();

    let before = protocol::resource_epoch_for_tests().expect("resource epoch before release");
    drop(blocker_two);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while protocol::resource_epoch_for_tests().expect("poll resource epoch") == before {
        assert!(
            std::time::Instant::now() < deadline,
            "coordinator did not advance the resource epoch for a real holder release"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let blocker_two = crate::flock::try_flock(cpu_lock_path(2), crate::flock::FlockMode::Exclusive)
        .unwrap()
        .expect("re-block waiter after epoch transition");
    std::fs::write(&gate, b"release").expect("release stale-epoch probe");
    std::thread::sleep(std::time::Duration::from_millis(150));
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
fn granted_acquirer_crash_before_record_clear_is_pruned() {
    let _prefixes = LockPrefixesGuard::new();
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
    std::thread::sleep(std::time::Duration::from_millis(100));
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
    let _prefixes = LockPrefixesGuard::new();
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
