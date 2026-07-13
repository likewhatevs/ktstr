use super::super::*;
use super::*;

#[test]
fn resource_lock_shared_acquires() {
    let _tempfile_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-flock-shared-acquires-")
        .suffix(".lock")
        .tempfile()
        .unwrap();
    let path = _tempfile_keep_alive.path().to_str().unwrap();
    let fd = try_flock(path, FlockMode::Shared).expect("open should succeed");
    assert!(fd.is_some(), "shared lock on fresh file should succeed");
}

#[test]
fn resource_lock_exclusive_contention() {
    let _tempfile_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-flock-excl-contention-")
        .suffix(".lock")
        .tempfile()
        .unwrap();
    let path = _tempfile_keep_alive.path().to_str().unwrap();
    let holder = try_flock(path, FlockMode::Exclusive)
        .expect("open should succeed")
        .expect("first lock should succeed");
    let second = try_flock(path, FlockMode::Exclusive).expect("open should succeed");
    assert!(
        second.is_none(),
        "second exclusive lock while held should return None",
    );
    drop(holder);
}

#[test]
fn resource_lock_shared_coexist() {
    let _tempfile_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-flock-shared-coexist-")
        .suffix(".lock")
        .tempfile()
        .unwrap();
    let path = _tempfile_keep_alive.path().to_str().unwrap();
    let h1 = try_flock(path, FlockMode::Shared)
        .expect("open should succeed")
        .expect("first shared lock should succeed");
    let h2 = try_flock(path, FlockMode::Shared)
        .expect("open should succeed")
        .expect("second shared lock should succeed");
    // Both held simultaneously.
    drop(h1);
    drop(h2);
}

#[test]
fn resource_lock_exclusive_blocks_shared() {
    let _tempfile_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-flock-excl-blocks-sh-")
        .suffix(".lock")
        .tempfile()
        .unwrap();
    let path = _tempfile_keep_alive.path().to_str().unwrap();
    let holder = try_flock(path, FlockMode::Exclusive)
        .expect("open should succeed")
        .expect("exclusive lock should succeed");
    let shared = try_flock(path, FlockMode::Shared).expect("open should succeed");
    assert!(
        shared.is_none(),
        "shared lock should fail while exclusive is held",
    );
    drop(holder);
}

#[test]
fn resource_lock_shared_blocks_exclusive() {
    let _tempfile_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-flock-sh-blocks-excl-")
        .suffix(".lock")
        .tempfile()
        .unwrap();
    let path = _tempfile_keep_alive.path().to_str().unwrap();
    let holder = try_flock(path, FlockMode::Shared)
        .expect("open should succeed")
        .expect("shared lock should succeed");
    let excl = try_flock(path, FlockMode::Exclusive).expect("open should succeed");
    assert!(
        excl.is_none(),
        "exclusive lock should fail while shared is held",
    );
    drop(holder);
}

#[test]
fn resource_lock_release_on_drop() {
    let _tempfile_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-flock-release-drop-")
        .suffix(".lock")
        .tempfile()
        .unwrap();
    let path = _tempfile_keep_alive.path().to_str().unwrap();
    {
        let _holder = try_flock(path, FlockMode::Exclusive)
            .expect("open should succeed")
            .expect("lock should succeed");
    }
    // After drop, the lock should be available again.
    let fd = try_flock(path, FlockMode::Exclusive)
        .expect("open should succeed")
        .expect("lock should be available after drop");
    drop(fd);
}

#[test]
fn resource_lock_exclusive_success() {
    let _prefixes = LockPrefixesGuard::new();
    // Use high LLC indices to avoid collision with real locks.
    let plan = PinningPlan {
        assignments: vec![(0, 90100), (1, 90101)],
        service_cpu: None,
        llc_indices: vec![90100],
        locks: Vec::new(),
    };
    let llc_indices = &[90100usize];
    let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Exclusive).unwrap();
    let (llc_offset, locks) = unwrap_acquired(outcome, None);
    assert_eq!(llc_offset, 90100);
    // Exclusive mode: only LLC locks, no per-CPU locks.
    assert_eq!(locks.len(), 1);
}

#[test]
fn resource_lock_shared_includes_cpu_locks() {
    let _prefixes = LockPrefixesGuard::new();
    let plan = PinningPlan {
        assignments: vec![(0, 90200), (1, 90201)],
        service_cpu: None,
        llc_indices: vec![90200],
        locks: Vec::new(),
    };
    let llc_indices = &[90200usize];

    let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Shared).unwrap();
    let (_, locks) = unwrap_acquired(outcome, None);
    // Shared mode: 1 LLC lock + 2 CPU locks = 3 total.
    assert_eq!(locks.len(), 3);
}

#[test]
fn resource_lock_shared_with_service_cpu() {
    let _prefixes = LockPrefixesGuard::new();
    let plan = PinningPlan {
        assignments: vec![(0, 90300)],
        service_cpu: Some(90301),
        llc_indices: vec![90300],
        locks: Vec::new(),
    };
    let llc_indices = &[90300usize];

    let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Shared).unwrap();
    let (_, locks) = unwrap_acquired(outcome, None);
    // 1 LLC lock + 1 assignment CPU lock + 1 service CPU lock = 3.
    assert_eq!(locks.len(), 3);
}

#[test]
fn resource_lock_exclusive_skips_cpu_locks() {
    let _prefixes = LockPrefixesGuard::new();
    // Exclusive LLC mode should NOT acquire per-CPU locks.
    let plan = PinningPlan {
        assignments: vec![(0, 90400), (1, 90401)],
        service_cpu: Some(90402),
        llc_indices: vec![90400],
        locks: Vec::new(),
    };
    let llc_indices = &[90400usize];

    let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Exclusive).unwrap();
    let (_, locks) = unwrap_acquired(outcome, None);
    // Exclusive: only 1 LLC lock, no CPU locks.
    assert_eq!(locks.len(), 1);
}

#[test]
fn resource_lock_contention_returns_unavailable() {
    let _prefixes = LockPrefixesGuard::new();
    // Hold an exclusive lock, then try to acquire the same LLC.
    let plan = PinningPlan {
        assignments: vec![(0, 90500)],
        service_cpu: None,
        llc_indices: vec![90500],
        locks: Vec::new(),
    };
    let llc_indices = &[90500usize];
    let lock_path = llc_lock_path(90500);

    let holder = try_flock(&lock_path, FlockMode::Exclusive)
        .unwrap()
        .unwrap();

    let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Exclusive).unwrap();
    let reason = expect_unavailable(outcome, Some("while lock is held"));
    assert!(
        reason.contains("90500"),
        "reason should identify the busy LLC: {reason}",
    );
    drop(holder);
}

#[test]
fn resource_lock_all_or_nothing() {
    let _prefixes = LockPrefixesGuard::new();
    // Two LLC indices: hold the second one, verify the first is
    // released when the second fails (all-or-nothing semantics).
    let plan = PinningPlan {
        assignments: vec![(0, 90600), (1, 90601)],
        service_cpu: None,
        llc_indices: vec![90600, 90601],
        locks: Vec::new(),
    };
    let llc_indices = &[90600usize, 90601];
    let llc_600 = llc_lock_path(90600);
    let llc_601 = llc_lock_path(90601);

    let holder = try_flock(&llc_601, FlockMode::Exclusive).unwrap().unwrap();

    let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Exclusive).unwrap();
    assert!(
        matches!(outcome, LockOutcome::Unavailable(_)),
        "should fail when second LLC is busy",
    );

    // LLC 90600 should be released (all-or-nothing). Verify by
    // acquiring it successfully.
    let reacquire = try_flock(&llc_600, FlockMode::Exclusive)
        .unwrap()
        .expect("LLC 90600 should be released after all-or-nothing failure");
    drop(reacquire);
    drop(holder);
}

#[test]
fn resource_lock_shared_cpu_contention() {
    let _prefixes = LockPrefixesGuard::new();
    // Shared LLC mode: hold a CPU lock, verify acquire fails.
    let plan = PinningPlan {
        assignments: vec![(0, 90700)],
        service_cpu: None,
        llc_indices: vec![90700],
        locks: Vec::new(),
    };
    let llc_indices = &[90700usize];
    let llc_path = llc_lock_path(90700);
    let cpu_path = cpu_lock_path(90700);

    let holder = try_flock(&cpu_path, FlockMode::Exclusive).unwrap().unwrap();

    let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Shared).unwrap();
    assert!(
        matches!(outcome, LockOutcome::Unavailable(_)),
        "should fail when CPU lock is held",
    );

    // LLC lock should be released (all-or-nothing).
    let reacquire = try_flock(&llc_path, FlockMode::Shared)
        .unwrap()
        .expect("LLC 90700 should be released after CPU contention");
    drop(reacquire);
    drop(holder);
}

#[test]
fn resource_lock_empty_llc_indices() {
    // Empty llc_indices: LLC lock loop iterates zero times.
    // Exclusive mode skips CPU locks. Result: Acquired with
    // llc_offset 0 and empty locks vec.
    let plan = PinningPlan {
        assignments: vec![(0, 90800)],
        service_cpu: None,
        llc_indices: vec![],
        locks: Vec::new(),
    };
    let outcome = acquire_resource_locks(&plan, &[], LlcLockMode::Exclusive).unwrap();
    let (llc_offset, locks) = unwrap_acquired(outcome, None);
    assert_eq!(llc_offset, 0);
    assert!(locks.is_empty());
}

#[test]
fn resource_lock_service_cpu_contention() {
    let _prefixes = LockPrefixesGuard::new();
    // Shared mode: LLC and assignment CPU locks succeed, but
    // service CPU is held → Unavailable. All prior locks released.
    let plan = PinningPlan {
        assignments: vec![(0, 90900)],
        service_cpu: Some(90901),
        llc_indices: vec![90850],
        locks: Vec::new(),
    };
    let llc_indices = &[90850usize];
    let llc_path = llc_lock_path(90850);
    let cpu_900 = cpu_lock_path(90900);
    let cpu_901 = cpu_lock_path(90901);

    // Hold the service CPU lock.
    let holder = try_flock(&cpu_901, FlockMode::Exclusive).unwrap().unwrap();

    let outcome = acquire_resource_locks(&plan, llc_indices, LlcLockMode::Shared).unwrap();
    let reason = expect_unavailable(outcome, Some("when service CPU is held"));
    // The canonical-order acquire reports the busy LOCKFILE PATH
    // (service CPU and assignment CPUs share the ktstr-cpu-* family);
    // the service CPU's index in the path is what identifies it.
    assert!(
        reason.contains("90901") && reason.contains("busy"),
        "reason should name the busy service-CPU lockfile 90901: {reason}",
    );

    // All prior locks should be released (all-or-nothing).
    let reacquire_llc = try_flock(&llc_path, FlockMode::Shared)
        .unwrap()
        .expect("LLC 90850 should be released after service CPU contention");
    let reacquire_cpu = try_flock(&cpu_900, FlockMode::Exclusive)
        .unwrap()
        .expect("CPU 90900 should be released after service CPU contention");
    drop(reacquire_llc);
    drop(reacquire_cpu);
    drop(holder);
}

/// `pid_window_offset` must diffuse adjacent pids across the
/// offset space so a batch-spawn (e.g. nextest forking N test
/// processes back-to-back, common pid range like
/// 100000..100100) doesn't pile every peer onto the same
/// starting window.
///
/// Pin the spread shape across a window of 100 consecutive pids:
///   * every pid's offset must fit in `[0, max_start)`;
///   * the unique-offset count must reach most of the
///     33-offset space (assert >= 25), so adjacent pids
///     don't all collapse to the same offset;
///   * the average gap between consecutive pids' offsets
///     must exceed 5 (the trivial `pid % max_start` baseline
///     produces a gap of 1).
///
/// `max_start = 33` is chosen so the ideal uniform spread
/// would visit every offset 100/33 ≈ 3 times; the unique
/// count must approach `max_start`.
#[test]
fn pid_window_offset_spreads_adjacent_pids() {
    let max_start = 33usize;
    let pids: Vec<u32> = (100_000..100_100).collect();
    let offsets: Vec<usize> = pids
        .iter()
        .map(|&p| pid_window_offset(p, max_start))
        .collect();

    // Every offset must fit in the target range.
    for (pid, off) in pids.iter().zip(offsets.iter()) {
        assert!(
            *off < max_start,
            "pid_window_offset({pid}, {max_start}) = {off}, exceeds max_start",
        );
    }

    // Unique-offset count: AHasher's avalanche makes
    // adjacent pid landings independent of each other, so
    // 100 pids over 33 offsets should hit roughly all 33
    // offsets. Pin >= 25 to absorb the natural birthday-
    // paradox compression while still catching a regression
    // that flattens adjacent pids onto the same offset.
    //
    // The bare `pid % 33` baseline ALSO produces 33 unique
    // offsets, so the unique-count assertion alone does not
    // distinguish the avalanching hash from the trivial
    // modulo. The "adjacent-pid landings differ by 1"
    // assertion below catches that case: bare modulo gives
    // |offset[i+1] - offset[i]| == 1 always (the cyclic
    // step), whereas AHasher produces a fully randomized
    // step distribution.
    let unique: std::collections::HashSet<_> = offsets.iter().copied().collect();
    assert!(
        unique.len() >= 25,
        "100 adjacent pids spread to only {} unique offsets (max_start={max_start}); \
         hash mixer is losing entropy. offsets: {offsets:?}",
        unique.len(),
    );

    // Distinguish from the bare `pid % max_start` baseline:
    // bare modulo on consecutive pids gives consecutive
    // offsets (gap of exactly 1 modulo max_start). Count how
    // many adjacent pid pairs land at exactly +/-1 offset
    // (handling the wrap at the boundary). For AHasher,
    // each adjacent pair has a 2/max_start ≈ 6% probability
    // of landing at +/-1 by chance, so over 99 pairs the
    // expected count is ~6 with stddev ~2.4. The bare
    // modulo baseline produces 99/99. Pin <= 30 (well above
    // the random-noise expected value, well below the bare-
    // modulo signature) so a regression that drops the
    // AHasher mixer trips this without flaking on the
    // hashed distribution.
    let adjacent_step_count = offsets
        .windows(2)
        .filter(|w| {
            let d = (w[0] as i64 - w[1] as i64).unsigned_abs() as usize;
            d == 1 || d == max_start - 1
        })
        .count();
    assert!(
        adjacent_step_count <= 30,
        "{adjacent_step_count} of {} adjacent pid pairs landed at +/-1 offset \
         (max_start={max_start}); the bare `pid % {max_start}` baseline produces \
         99/99 such pairs. AHasher avalanche should give ~6. offsets: {offsets:?}",
        offsets.len() - 1,
    );

    // Average gap between consecutive pids' offsets — the
    // signal that distinguishes "diffused" (gap >> 1) from
    // "adjacent collapse" (gap == 1, the bare `pid % 33`
    // shape that this fix replaces). Compute mean absolute
    // difference; AHasher avalanche should give average
    // gap near `max_start / 3` (uniform random walk on a
    // circular space of size N has expected step `N/3`).
    let gaps: Vec<usize> = offsets
        .windows(2)
        .map(|w| {
            let a = w[0] as i64;
            let b = w[1] as i64;
            (a - b).unsigned_abs() as usize
        })
        .collect();
    let mean_gap: f64 = gaps.iter().sum::<usize>() as f64 / gaps.len() as f64;
    assert!(
        mean_gap > 5.0,
        "mean offset gap between adjacent pids = {mean_gap:.2}, expected > 5 \
         (the bare `pid % {max_start}` baseline produces gap = 1; AHasher \
         avalanche should produce >> 5). offsets: {offsets:?}",
    );
}

/// `pid_window_offset` is deterministic: same (pid, max_start)
/// always produces the same offset. Pin against a regression
/// that introduces randomness or per-run state.
#[test]
fn pid_window_offset_deterministic() {
    for &pid in &[1u32, 100, 12345, 999_999, u32::MAX] {
        for &max_start in &[1usize, 3, 33, 1024, usize::MAX] {
            assert_eq!(
                pid_window_offset(pid, max_start),
                pid_window_offset(pid, max_start),
                "non-deterministic offset for pid={pid}, max_start={max_start}",
            );
        }
    }
}

/// `pid_window_offset` with `max_start == 1` always returns 0
/// (only valid offset). Pin the trivial-domain edge.
#[test]
fn pid_window_offset_max_start_one() {
    for &pid in &[0u32, 1, 100, u32::MAX] {
        assert_eq!(pid_window_offset(pid, 1), 0);
    }
}

/// `acquire_llc_plan` with `cpu_cap: None` reserves exactly 30%
/// of the allowed-CPU set (ceiling), walking whole LLCs and
/// partial-taking the last LLC's CPUs when the budget falls
/// mid-LLC. On a 10-CPU host split across 5 LLCs (2 CPUs each)
/// where every CPU is in the allowed set: ceil(10 * 0.30) = 3
/// CPUs → flock 2 LLCs (the first LLC's 2 CPUs + 1 CPU from
/// the second), `plan.cpus` holds exactly 3 CPUs.
///
/// Uses a per-test lockfile prefix via [`LlcLockPrefixGuard`] so
/// the `LlcGroup` vector can be a small 5-entry topology rather
/// than padding to host-LLC-count slots. Production path runs
/// through [`llc_lock_path`] which honors the test-only override.
/// Uses [`AllowedCpusGuard`] to pin the allowed-CPU set so the
/// 30%-default math is deterministic regardless of the CI
/// runner's real sched_getaffinity.
#[test]
fn acquire_llc_plan_none_cap_reserves_thirty_percent_cpus() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let topo = HostTopology::new_for_tests(&[
        (vec![0, 1], 0),
        (vec![2, 3], 0),
        (vec![4, 5], 0),
        (vec![6, 7], 0),
        (vec![8, 9], 0),
    ]);

    // synthetic() requires num_cpus >= num_llcs > 0. All 5
    // synthetic LLCs sit on NUMA node 0, so the seed-node walk in
    // plan_from_snapshots fills before any cross-node spill and the
    // distance closure's shape can't change the selection — the
    // TestTopology only needs to be a valid synthetic().
    let test_topo = crate::topology::TestTopology::synthetic(4, 1);

    let plan = acquire_llc_plan(&topo, &test_topo, None, PlacementPolicy::Consolidate, false)
        .expect("clean pool must allow SH on every selected LLC");
    // 30% of 10 CPUs = ceil(3.0) = 3 CPUs. 2-CPU LLCs: LLC 0
    // contributes 2, LLC 1 contributes 1 (partial-take), total
    // exactly 3.
    assert_eq!(
        plan.locked_llcs.len(),
        2,
        "budget of 3 CPUs flocks 2 LLCs (2 CPUs + 1 partial): {:?}",
        plan.locked_llcs,
    );
    assert_eq!(
        plan.cpus.len(),
        3,
        "plan.cpus is truncated to exactly the budget: {:?}",
        plan.cpus,
    );
    assert_eq!(plan.locks.len(), 2, "one fd per selected LLC");
}

/// `acquire_llc_plan` bails with `ResourceContention` when ANY
/// target LLC is held `LOCK_EX` by a peer (the path a perf-mode
/// VM takes via `acquire_resource_locks`). The error chain must
/// name the busy LLC index so an operator running `fuser
/// {lockfile}` can trace the holder.
///
/// Scope: pins ONE attempt of the EX-blocks-SH invariant — a
/// single DISCOVER round-trip. The full Tier-1 / Tier-2
/// coordination contract (retry budget, TOCTOU recovery,
/// holder-diagnostic freshness) is covered piecewise by the
/// retry-budget pin and the coexistence test; this test's
/// narrow claim is: when EX is held, SH fails fast with an
/// actionable error that names the LLC.
#[test]
fn acquire_llc_plan_bails_on_exclusive_peer() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0]);
    let topo = HostTopology::new_for_tests(&[(vec![0], 0)]);

    // Peer holds EX on LLC 0's lockfile through the overridden
    // prefix. Acquire the path after setting the prefix so the
    // held lock matches what the planner will try to acquire.
    let busy_path = llc_lock_path(0);
    let _peer_ex = try_flock(&busy_path, FlockMode::Exclusive)
        .unwrap()
        .expect("peer EX must acquire on clean pool");

    let test_topo = crate::topology::TestTopology::synthetic(4, 1);
    let err = acquire_llc_plan(&topo, &test_topo, None, PlacementPolicy::Consolidate, false)
        .expect_err("EX peer must block SH acquisition of the only LLC");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("LLC 0"),
        "error must name the busy LLC index so fuser can trace: {rendered}",
    );
    // Verify the ResourceContention tag survives so callers
    // pattern-matching on the error type (for nextest-retry
    // routing) see it correctly.
    assert!(
        err.downcast_ref::<ResourceContention>().is_some(),
        "error must downcast to ResourceContention for retry routing: {rendered}",
    );

    drop(_peer_ex);
}

/// Two no-perf-mode peers coexist: both acquire `acquire_llc_plan`
/// successfully because `LOCK_SH` is reentrant. The contract says
/// "shared holders coexist; exclusive blocks" — this pins the
/// shared-coexistence half, complementing the EX-blocks-SH test
/// above.
#[test]
fn acquire_llc_plan_coexists_with_shared_peer() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0]);
    let topo = HostTopology::new_for_tests(&[(vec![0], 0)]);
    let shared_path = llc_lock_path(0);

    // First peer: SH. Simulates an already-running no-perf-mode VM.
    let _peer_sh = try_flock(&shared_path, FlockMode::Shared)
        .unwrap()
        .expect("peer SH must acquire on clean pool");

    let test_topo = crate::topology::TestTopology::synthetic(4, 1);
    let plan = acquire_llc_plan(&topo, &test_topo, None, PlacementPolicy::Consolidate, false)
        .expect("second SH caller must coexist with the first");
    assert_eq!(
        plan.locks.len(),
        topo.llc_groups.len(),
        "second SH caller must acquire one fd per LLC group",
    );
}

/// The bounded-WAIT acquire completes once the holder releases: a peer
/// holds `LOCK_EX` on the target LLC, a releaser thread drops it after
/// ~300 ms, and `acquire_resource_locks_waiting` with a generous
/// deadline returns `Acquired` — it must PARK and pick the lock up,
/// not bail `Unavailable` on the first busy poll (the pre-wait
/// behaviour that became a host-skip). The elapsed floor proves it
/// actually waited through the hold window rather than racing the
/// release.
#[test]
fn resource_lock_wait_acquires_after_peer_release() {
    let _prefixes = LockPrefixesGuard::new();
    let plan = PinningPlan {
        assignments: vec![(0, 90700)],
        service_cpu: None,
        llc_indices: vec![90700],
        locks: Vec::new(),
    };
    let lock_path = llc_lock_path(90700);
    let holder = try_flock(&lock_path, FlockMode::Exclusive)
        .unwrap()
        .expect("peer EX must acquire on clean pool");
    // Releaser: drop the peer's fd after the acquirer has begun
    // polling. The fd moves into the thread; the thread-local lock
    // prefix stays on the test thread where the acquire runs.
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        drop(holder);
    });
    let start = std::time::Instant::now();
    let outcome =
        acquire_resource_locks_waiting(&plan, &[90700usize], LlcLockMode::Exclusive, true).unwrap();
    let elapsed = start.elapsed();
    let (_, locks) = unwrap_acquired(outcome, Some("after the peer's timed release"));
    assert_eq!(locks.len(), 1);
    assert!(
        elapsed >= std::time::Duration::from_millis(250),
        "acquire must have parked through the peer's ~300 ms hold; elapsed={elapsed:?}",
    );
    releaser.join().expect("releaser thread must not panic");
}

/// Progress-based patience is honoured: with the peer's `LOCK_EX`
/// never released (a "wedged holder"), the head gains nothing, its
/// no-progress patience (overridden short for the test) expires, and
/// the acquire returns `Unavailable` (which the run path surfaces as
/// retryable contention) — it neither bails early nor parks forever.
/// Timing floor pins that it actually waited through the window.
#[test]
fn resource_lock_wait_patience_expiry_returns_unavailable() {
    let _prefixes = LockPrefixesGuard::new();
    super::super::protocol::PATIENCE_OVERRIDE.with(|p| {
        *p.borrow_mut() = Some(std::time::Duration::from_millis(400));
    });
    let plan = PinningPlan {
        assignments: vec![(0, 90800)],
        service_cpu: None,
        llc_indices: vec![90800],
        locks: Vec::new(),
    };
    let lock_path = llc_lock_path(90800);
    let _holder = try_flock(&lock_path, FlockMode::Exclusive)
        .unwrap()
        .expect("peer EX must acquire on clean pool");
    let start = std::time::Instant::now();
    let outcome =
        acquire_resource_locks_waiting(&plan, &[90800usize], LlcLockMode::Exclusive, true).unwrap();
    let elapsed = start.elapsed();
    super::super::protocol::PATIENCE_OVERRIDE.with(|p| *p.borrow_mut() = None);
    let reason = expect_unavailable(outcome, Some("with the peer never releasing"));
    assert!(
        reason.contains("90800"),
        "reason should identify the stalled lock: {reason}",
    );
    assert!(
        reason.contains("no acquisition progress"),
        "reason must frame the bail as a no-progress verdict, not a \
         wall deadline: {reason}",
    );
    assert!(
        elapsed >= std::time::Duration::from_millis(350),
        "acquire must wait out the patience window, not bail early; elapsed={elapsed:?}",
    );
}

/// The full `acquire_llc_plan` pipeline waits out a REAL `LOCK_EX`
/// holder when given a wait deadline: with the only LLC held EX (the
/// exact shape of a colocated peer runner's perf-mode reservation) and
/// a releaser dropping it after ~400 ms — past the entire ~260 ms
/// TOCTOU backoff budget that used to be the hard cutoff — the plan
/// acquisition COMPLETES instead of bailing `ResourceContention`.
/// This is the end-to-end pin for the CI regression where 205 cells
/// were host-skipped while peers' perf-mode fixtures held their LLCs.
#[test]
fn acquire_llc_plan_waits_out_real_exclusive_peer() {
    let _llc_prefix = LlcLockPrefixGuard::new();
    let _allowed = AllowedCpusGuard::new(vec![0]);
    let topo = HostTopology::new_for_tests(&[(vec![0], 0)]);

    let busy_path = llc_lock_path(0);
    let peer_ex = try_flock(&busy_path, FlockMode::Exclusive)
        .unwrap()
        .expect("peer EX must acquire on clean pool");
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(400));
        drop(peer_ex);
    });

    let test_topo = crate::topology::TestTopology::synthetic(4, 1);
    let plan = acquire_llc_plan(&topo, &test_topo, None, PlacementPolicy::Consolidate, true)
        .expect("wait-enabled acquisition must complete after the peer releases");
    assert_eq!(plan.locks.len(), 1, "one fd for the single released LLC");
    releaser.join().expect("releaser thread must not panic");
}

/// `CpuCap::new(0)` must reject with the "≥ 1 (got 0)" message.
/// Zero is a scripting-mistake sentinel — silent acceptance would
/// disable the resource contract.
#[test]
fn cpu_cap_new_rejects_zero() {
    let err = CpuCap::new(0).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("≥ 1"), "msg={msg}");
    assert!(msg.contains("got 0"), "msg={msg}");
}
