use super::*;

#[test]
fn interactive_run_guard_drop_signals_wakes_and_joins_every_helper() {
    use std::sync::atomic::AtomicUsize;

    let kill = Arc::new(AtomicBool::new(false));
    let kill_evt = Arc::new(
        vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK)
            .expect("interactive guard kill eventfd"),
    );
    let freeze = Arc::new(AtomicBool::new(true));
    let bsp_done = Arc::new(AtomicBool::new(false));
    let (stdin_read, stdin_write) = nix::unistd::pipe().expect("stdin wake pipe");
    let dmesg_wakeup =
        Arc::new(vmm_sys_util::eventfd::EventFd::new(0).expect("dmesg wake eventfd"));
    let joined = Arc::new(AtomicUsize::new(0));

    let exec_kill = Arc::clone(&kill);
    let exec_joined = Arc::clone(&joined);
    let deadline_publisher = std::thread::spawn(move || {
        while !exec_kill.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        exec_joined.fetch_or(1 << 0, Ordering::AcqRel);
    });

    let stdin_joined = Arc::clone(&joined);
    let stdin = std::thread::spawn(move || {
        let mut byte = [0u8; 1];
        assert_eq!(
            nix::unistd::read(&stdin_read, &mut byte).expect("stdin wake read"),
            0,
            "guard closes the writer so poll/read observes EOF",
        );
        stdin_joined.fetch_or(1 << 1, Ordering::AcqRel);
    });

    let stdout_kill = Arc::clone(&kill);
    let stdout_joined = Arc::clone(&joined);
    let stdout = std::thread::spawn(move || {
        while !stdout_kill.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        stdout_joined.fetch_or(1 << 2, Ordering::AcqRel);
        true
    });

    let dmesg_evt = Arc::clone(&dmesg_wakeup);
    let dmesg_joined = Arc::clone(&joined);
    let dmesg = std::thread::spawn(move || {
        dmesg_evt.read().expect("dmesg shutdown wake");
        dmesg_joined.fetch_or(1 << 3, Ordering::AcqRel);
    });

    let mut guard = InteractiveRunGuard::new(
        Vec::new(),
        stdin_write,
        Arc::clone(&kill),
        Arc::clone(&kill_evt),
        Arc::clone(&bsp_done),
        Arc::clone(&freeze),
    );
    guard.deadline_publisher = Some(deadline_publisher);
    guard.stdin = Some(stdin);
    guard.stdout = Some(stdout);
    guard.dmesg = Some(dmesg);
    guard.dmesg_wakeup = Some(dmesg_wakeup);

    drop(guard);

    assert!(kill.load(Ordering::Acquire));
    assert!(bsp_done.load(Ordering::Acquire));
    assert!(!freeze.load(Ordering::Acquire));
    assert_eq!(
        kill_evt
            .read()
            .expect("done and kill eventfd edges were signaled"),
        2,
    );
    assert_eq!(
        joined.load(Ordering::Acquire),
        0b1111,
        "Drop must join every owned helper before returning",
    );
}

const INTERACTIVE_RELAY_CHILD_ENV: &str = "KTSTR_INTERACTIVE_RELAY_CHILD";
const INTERACTIVE_RELAY_DEADLINE_MODE: &str = "deadline";
const INTERACTIVE_RELAY_STDOUT_MODE: &str = "stdout-failure";
const INTERACTIVE_RELAY_AP_FATAL_MODE: &str = "ap-fatal";
const INTERACTIVE_RELAY_SETUP_FAILURE_MODE: &str = "setup-failure";
const INTERACTIVE_RELAY_READY: &str = "KTSTR_INTERACTIVE_RELAY_READY";
const INTERACTIVE_RELAY_CHILD_FAILURE: i32 = 75;

fn interactive_runtime_threads() -> std::collections::BTreeMap<u32, String> {
    std::fs::read_dir("/proc/self/task")
        .expect("enumerate process tasks")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let tid = entry.file_name().to_string_lossy().parse().ok()?;
            let comm = std::fs::read_to_string(entry.path().join("comm")).ok()?;
            Some((tid, comm.trim().to_owned()))
        })
        .collect()
}

/// Subprocess-only real-KVM driver for the interactive relay regressions below.
///
/// `_exit` is deliberate: the stdout-failure case closes the test process's
/// stdout while the VM runs. Returning through libtest would make the harness's
/// own trailing status write—not the VMM—fail on the closed pipe.
#[test]
fn interactive_kill_relay_subprocess_helper() {
    let Ok(mode) = std::env::var(INTERACTIVE_RELAY_CHILD_ENV) else {
        return;
    };

    INTERACTIVE_BSP_ENTERED.store(false, Ordering::Release);
    INJECT_INTERACTIVE_AP_FATAL.store(mode == INTERACTIVE_RELAY_AP_FATAL_MODE, Ordering::Release);
    FAIL_INTERACTIVE_AFTER_AP_SPAWN.store(
        mode == INTERACTIVE_RELAY_SETUP_FAILURE_MODE,
        Ordering::Release,
    );
    let baseline_threads =
        (mode == INTERACTIVE_RELAY_SETUP_FAILURE_MODE).then(interactive_runtime_threads);

    let kernel = crate::test_support::require_kernel();
    let payload = crate::resolve_current_exe().expect("resolve test binary for shell init");
    let busybox = blobs::load_busybox_bytes().expect("load busybox for shell initramfs");
    let topology = if matches!(
        mode.as_str(),
        INTERACTIVE_RELAY_AP_FATAL_MODE | INTERACTIVE_RELAY_SETUP_FAILURE_MODE
    ) {
        Topology::new(1, 1, 2, 1)
    } else {
        Topology::new(1, 1, 1, 1)
    };
    let exec_cmd = match mode.as_str() {
        INTERACTIVE_RELAY_DEADLINE_MODE | INTERACTIVE_RELAY_AP_FATAL_MODE => "sleep 60",
        INTERACTIVE_RELAY_SETUP_FAILURE_MODE => "true",
        INTERACTIVE_RELAY_STDOUT_MODE => {
            "echo KTSTR_INTERACTIVE_RELAY_READY; while :; do echo KTSTR_INTERACTIVE_RELAY_SPAM; done"
        }
        other => {
            eprintln!("unknown interactive relay child mode: {other}");
            // SAFETY: this is an isolated subprocess test driver.
            unsafe { libc::_exit(INTERACTIVE_RELAY_CHILD_FAILURE) };
        }
    };
    let exec_timeout = if mode == INTERACTIVE_RELAY_DEADLINE_MODE {
        Duration::from_secs(1)
    } else {
        Duration::from_secs(60)
    };
    let topo_arg = if matches!(
        mode.as_str(),
        INTERACTIVE_RELAY_AP_FATAL_MODE | INTERACTIVE_RELAY_SETUP_FAILURE_MODE
    ) {
        "KTSTR_MODE=shell KTSTR_TOPO=1,1,2,1"
    } else {
        "KTSTR_MODE=shell KTSTR_TOPO=1,1,1,1"
    };

    let vm = match KtstrVm::builder()
        .kernel(&kernel)
        .init_binary(payload)
        .busybox(Some(busybox))
        .topology(topology)
        .memory_deferred_min(128)
        .cmdline(topo_arg)
        .exec_cmd(exec_cmd)
        .exec_timeout(exec_timeout)
        .no_perf_mode(true)
        .build()
    {
        Ok(vm) => vm,
        Err(error) => {
            eprintln!("interactive relay child VM build failed: {error:#}");
            // SAFETY: this is an isolated subprocess test driver.
            unsafe { libc::_exit(INTERACTIVE_RELAY_CHILD_FAILURE) };
        }
    };

    let started = Instant::now();
    let outcome = vm.run_interactive();
    let no_setup_leaks = baseline_threads.as_ref().is_none_or(|baseline| {
        interactive_runtime_threads()
            .into_iter()
            .all(|(tid, name)| {
                baseline.contains_key(&tid)
                    || !(name.starts_with("vcpu-")
                        || name.starts_with("interactive-")
                        || name.starts_with("ktstr-exec-"))
            })
    });
    if !no_setup_leaks {
        eprintln!("interactive setup failure returned with live vCPU/helper tasks");
    }
    let expected = match mode.as_str() {
        INTERACTIVE_RELAY_DEADLINE_MODE => outcome
            .as_ref()
            .is_err_and(|error| error.to_string().contains("exec-timeout")),
        INTERACTIVE_RELAY_STDOUT_MODE | INTERACTIVE_RELAY_AP_FATAL_MODE => outcome.is_err(),
        INTERACTIVE_RELAY_SETUP_FAILURE_MODE => {
            outcome.as_ref().is_err_and(|error| {
                error
                    .to_string()
                    .contains("injected interactive setup failure after AP spawn")
            }) && no_setup_leaks
        }
        _ => false,
    };
    if !expected {
        eprintln!(
            "interactive relay child returned an unexpected result after {:?}: {outcome:?}",
            started.elapsed(),
        );
    }

    INJECT_INTERACTIVE_AP_FATAL.store(false, Ordering::Release);
    FAIL_INTERACTIVE_AFTER_AP_SPAWN.store(false, Ordering::Release);
    INTERACTIVE_BSP_ENTERED.store(false, Ordering::Release);
    // SAFETY: skip libtest teardown because stdout is intentionally invalid in
    // one mode and encode the semantic result directly in the child status.
    unsafe {
        libc::_exit(if expected {
            0
        } else {
            INTERACTIVE_RELAY_CHILD_FAILURE
        })
    };
}

fn spawn_interactive_relay_child(
    mode: &str,
    stdout: std::process::Stdio,
) -> (std::process::Child, std::thread::JoinHandle<Vec<u8>>) {
    use std::io::Read;

    let exact = "vmm::tests::interactive_kill_relay_subprocess_helper";
    let mut child = std::process::Command::new(
        std::env::current_exe().expect("resolve current unit-test executable"),
    )
    .arg("--exact")
    .arg(exact)
    .arg("--nocapture")
    .env(INTERACTIVE_RELAY_CHILD_ENV, mode)
    .stdin(std::process::Stdio::null())
    .stdout(stdout)
    .stderr(std::process::Stdio::piped())
    .spawn()
    .expect("spawn interactive relay subprocess");
    let mut stderr = child.stderr.take().expect("capture relay child stderr");
    let stderr_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr.read_to_end(&mut bytes);
        bytes
    });
    (child, stderr_reader)
}

fn wait_interactive_relay_child(
    child: &mut std::process::Child,
    timeout: Duration,
) -> std::process::ExitStatus {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().expect("poll interactive relay child") {
            return status;
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let reap_deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < reap_deadline {
                if child
                    .try_wait()
                    .expect("poll killed interactive relay child")
                    .is_some()
                {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            panic!(
                "interactive relay subprocess exceeded its {:?} hard timeout",
                timeout,
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn assert_interactive_relay_child(mode: &str) {
    let (mut child, stderr_reader) =
        spawn_interactive_relay_child(mode, std::process::Stdio::null());
    let status = wait_interactive_relay_child(&mut child, Duration::from_secs(20));
    let stderr = stderr_reader.join().unwrap_or_default();
    assert!(
        status.success(),
        "interactive relay {mode} child failed with {status}; stderr={}",
        String::from_utf8_lossy(&stderr),
    );
}

#[test]
fn boot_kernel_interactive_deadline_relay_exits_blocked_bsp() {
    if !crate::test_support::cargo_ktstr_orchestrated() {
        skip!("real-KVM interactive deadline relay regression requires cargo-ktstr orchestration");
    }
    assert_interactive_relay_child(INTERACTIVE_RELAY_DEADLINE_MODE);
}

#[test]
fn boot_kernel_interactive_ap_fatal_relay_exits_blocked_bsp() {
    if !crate::test_support::cargo_ktstr_orchestrated() {
        skip!("real-KVM interactive AP-fatal relay regression requires cargo-ktstr orchestration");
    }
    assert_interactive_relay_child(INTERACTIVE_RELAY_AP_FATAL_MODE);
}

#[test]
fn boot_kernel_interactive_stdout_failure_relay_exits_blocked_bsp() {
    use std::io::BufRead;

    if !crate::test_support::cargo_ktstr_orchestrated() {
        skip!(
            "real-KVM interactive stdout-failure relay regression requires cargo-ktstr orchestration"
        );
    }

    let (mut child, stderr_reader) =
        spawn_interactive_relay_child(INTERACTIVE_RELAY_STDOUT_MODE, std::process::Stdio::piped());
    let stdout = child.stdout.take().expect("capture relay child stdout");
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let stdout_reader = std::thread::spawn(move || {
        let mut reader = std::io::BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    let _ = ready_tx.send(false);
                    return;
                }
                Ok(_) if line.contains(INTERACTIVE_RELAY_READY) => {
                    let _ = ready_tx.send(true);
                    // Dropping the only read end makes the VMM's next guest
                    // output write observe EPIPE/BrokenPipe.
                    return;
                }
                Ok(_) => {}
                Err(_) => {
                    let _ = ready_tx.send(false);
                    return;
                }
            }
        }
    });

    let ready = ready_rx
        .recv_timeout(Duration::from_secs(15))
        .unwrap_or(false);
    if !ready {
        let _ = child.kill();
        let _ = wait_interactive_relay_child(&mut child, Duration::from_secs(2));
    }
    assert!(
        ready,
        "guest never produced the stdout relay readiness marker"
    );

    let status = wait_interactive_relay_child(&mut child, Duration::from_secs(20));
    let _ = stdout_reader.join();
    let stderr = stderr_reader.join().unwrap_or_default();
    assert!(
        status.success(),
        "interactive stdout-failure relay child failed with {status}; stderr={}",
        String::from_utf8_lossy(&stderr),
    );
}

/// Whether a timing-sensitive boot-race VM test ran in an environment
/// too loaded for its wall-anchored expectation to hold — so the test
/// SKIPs (environmental non-verdict) instead of false-failing. Two
/// independent signals, either sufficient:
///
///   - the guest's vCPU dilation `D` (`1 + Σrun_delay/Σon_cpu`) above
///     1.1 — the host stole CPU from a guest that WAS trying to run;
///   - the machine's 1-minute loadavg above 60% of its core count — the
///     load-bearing signal for IDLE-boot VMs, whose vCPU threads mostly
///     wait and so accrue little run_delay (`D ≈ 1`) even while a
///     saturated host drags their boot out. `D` alone is blind to that;
///     machine loadavg sees it directly. The machine core count is read
///     from `/proc/stat` (all cores, ignoring any cpuset the test process
///     runs under) so the ratio compares like with like.
///
/// Both false ⇒ a genuinely quiet host ⇒ the assertion is enforced (dev
/// boxes and lightly-loaded x86 CI keep full coverage of the real bug).
fn run_env_was_loaded(result: &VmResult) -> bool {
    let dilated = result
        .host_vcpu_schedstat
        .as_ref()
        .and_then(|s| s.dilation())
        .is_some_and(|d| d > 1.1);
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
    let host_busy = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok())
        .is_some_and(|load1| load1 > machine_cpus as f64 * 0.6);
    dilated || host_busy
}

#[cfg(target_arch = "x86_64")]
#[test]
fn routing_failure_summary_none_when_zero_else_counts() {
    assert!(
        routing_failure_summary(0).is_none(),
        "no routing failures → no summary"
    );
    let msg = routing_failure_summary(3).expect("n>0 → summary");
    assert!(
        msg.contains("3 device-IRQ routing failure"),
        "summary names the count: {msg:?}"
    );
}

/// build_overcommit_run_locks uses the resolved allowed cpuset as the default
/// mask (no locks, no pinning plan). The mask is the run-time budget source:
/// run() stamps result.cpu_budget = default_cpu_mask.len().max(1), so the
/// sidecar reflects the overcommit mask, not the build-time vCPU count. vcpus
/// equals the mask size here so the overcommit_warning side-channel (which fires
/// only when the mask is narrower than vcpus) stays out of this mask-focused
/// test.
#[test]
fn build_overcommit_run_locks_uses_allowed_as_mask() {
    let rl = KtstrVm::build_overcommit_run_locks(vec![0, 1, 2, 3], 4);
    assert_eq!(rl.default_cpu_mask, Some(vec![0, 1, 2, 3]));
    assert!(rl.locks.is_empty());
    assert!(rl.pinning_plan.is_none());
    // The overcommit path is not the no-perf replan path — it must not
    // populate `no_perf_cpus` (only `acquire_run_locks`' no-perf arm
    // does, from its fresh LLC plan).
    assert!(rl.no_perf_cpus.is_none());
}

/// acquire_default_run_locks overcommits (default_cpu_mask = the allowed cpuset)
/// when there is no cached host topology — nothing to plan a 1:1 pin against.
#[test]
fn acquire_default_run_locks_overcommits_with_no_host_topo() {
    // Reset BEFORE the asserts so an assert-fail cannot leak the thread-local
    // into a sibling; the call itself has no panic path. (host_topology's RAII
    // AllowedCpusGuard is module-private; the override is thread-local +
    // per-test isolated, so a leak could not cross tests regardless.)
    host_topology::ALLOWED_CPUS_OVERRIDE.with(|p| *p.borrow_mut() = Some(vec![0, 1]));
    let rl = KtstrVm::acquire_default_run_locks(None, &Topology::new(1, 1, 1, 1), false);
    host_topology::ALLOWED_CPUS_OVERRIDE.with(|p| *p.borrow_mut() = None);
    let rl = rl.expect("no-host overcommit is Ok, not an error");
    assert_eq!(
        rl.default_cpu_mask,
        Some(vec![0, 1]),
        "overcommit uses the allowed cpuset as the mask",
    );
    assert!(rl.pinning_plan.is_none());
    assert!(rl.no_perf_cpus.is_none());
}

/// acquire_default_run_locks overcommits when the host is too small for a 1:1
/// pin (compute_pinning fails for every offset, produced_candidate stays false)
/// — the make-it-work-overcommit-if-topologically-impossible host-gate policy.
/// Host has 1 LLC; the topology requests 2, so no offset can map it.
#[test]
fn acquire_default_run_locks_overcommits_when_host_too_small() {
    let host = host_topology::HostTopology::new_for_tests(&[(vec![0, 1], 0)]);
    let topo = Topology::new(1, 2, 1, 1);
    // Reset BEFORE the asserts so an assert-fail cannot leak the thread-local
    // into a sibling; the call itself has no panic path. (host_topology's RAII
    // AllowedCpusGuard is module-private; the override is thread-local +
    // per-test isolated, so a leak could not cross tests regardless.)
    host_topology::ALLOWED_CPUS_OVERRIDE.with(|p| *p.borrow_mut() = Some(vec![0, 1]));
    let rl = KtstrVm::acquire_default_run_locks(Some(&host), &topo, false);
    host_topology::ALLOWED_CPUS_OVERRIDE.with(|p| *p.borrow_mut() = None);
    let rl = rl.expect("a too-small host overcommits, it does not error or skip");
    assert_eq!(
        rl.default_cpu_mask,
        Some(vec![0, 1]),
        "host too small for a 1:1 pin overcommits to the allowed cpuset",
    );
    assert!(rl.pinning_plan.is_none());
    assert!(rl.no_perf_cpus.is_none());
}

#[test]
fn default_candidate_scan_reuses_one_aggregate_snapshot_and_fences_survivors() {
    struct ResetOverrides;
    impl Drop for ResetOverrides {
        fn drop(&mut self) {
            host_topology::ALLOWED_CPUS_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
            host_topology::LLC_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
            host_topology::CPU_LOCK_PREFIX_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        }
    }

    let temp = tempfile::TempDir::new().expect("default-scan tempdir");
    host_topology::ALLOWED_CPUS_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(vec![0, 1, 2, 3]));
    host_topology::LLC_LOCK_PREFIX_OVERRIDE.with(|slot| {
        *slot.borrow_mut() = Some(format!("{}/llc-", temp.path().display()));
    });
    host_topology::CPU_LOCK_PREFIX_OVERRIDE.with(|slot| {
        *slot.borrow_mut() = Some(format!("{}/cpu-", temp.path().display()));
    });
    let _reset = ResetOverrides;

    let claim = host_topology::protocol::ClaimSet::new(
        std::iter::empty(),
        0usize..4,
        crate::flock::FlockMode::Exclusive,
    );
    let coordinator = match host_topology::protocol::register_ticket_or_acquire(
        claim.clone(),
        claim,
        None,
        |_| Ok::<Option<()>, anyhow::Error>(None),
    )
    .expect("register union CPU claim")
    {
        host_topology::protocol::TicketWork::Coordinator(coordinator) => coordinator,
        host_topology::protocol::TicketWork::Acquired(()) => {
            panic!("fresh registry must elect a coordinator")
        }
    };
    let host = host_topology::HostTopology::new_for_tests(&[(vec![0, 1, 2, 3], 0)]);
    let topo = Topology::new(1, 1, 1, 1);
    let snapshot_before = host_topology::protocol::aggregate_snapshot_read_count_for_tests();
    let probe_before = host_topology::authoritative_registry_probe_count_for_tests();
    let error = match KtstrVm::acquire_default_run_locks(Some(&host), &topo, false) {
        Ok(_) => panic!("the union claim must fence every default CPU window"),
        Err(error) => error,
    };
    assert!(
        error
            .downcast_ref::<host_topology::ResourceContention>()
            .is_some(),
        "all locally fenced candidates must report contention: {error:#}",
    );
    assert_eq!(
        host_topology::protocol::aggregate_snapshot_read_count_for_tests() - snapshot_before,
        1,
        "the full default candidate scan must copy aggregates once",
    );
    assert_eq!(
        host_topology::authoritative_registry_probe_count_for_tests() - probe_before,
        0,
        "locally fenced windows must not reopen the authoritative registry fence",
    );

    drop(coordinator);
    let probe_before = host_topology::authoritative_registry_probe_count_for_tests();
    let acquired = KtstrVm::acquire_default_run_locks(Some(&host), &topo, false)
        .expect("an unfenced default candidate must acquire");
    assert_eq!(
        host_topology::authoritative_registry_probe_count_for_tests() - probe_before,
        1,
        "the selected unfenced window must still use the authoritative fence",
    );
    drop(acquired);
}

/// degrade_contention_to_overcommit converts a transient ResourceContention
/// into a lock-free best-effort RunLocks: the one-shot shell deliberately
/// does not park on a peer's LOCK_EX (no wait deadline, no nextest retry),
/// so it boots overcommitted rather than aborting. No locks, no mask, no
/// pinning plan.
#[test]
fn degrade_contention_to_overcommit_maps_contention_to_lockfree() {
    let contended = Err(anyhow::Error::new(host_topology::ResourceContention {
        reason: "all 3 LLC slots busy (LOCK_SH)".into(),
    }));
    let rl = KtstrVm::degrade_contention_to_overcommit(contended)
        .expect("contention degrades to Ok, not an error");
    assert!(
        rl.locks.is_empty(),
        "best-effort boot holds no resource locks"
    );
    assert!(rl.default_cpu_mask.is_none());
    assert!(rl.pinning_plan.is_none());
}

/// degrade_contention_to_overcommit passes a successful acquire through
/// unchanged — the degrade only fires on ResourceContention.
#[test]
fn degrade_contention_to_overcommit_passes_success_through() {
    let ok = Ok(RunLocks {
        locks: Vec::new(),
        default_cpu_mask: Some(vec![0, 1]),
        pinning_plan: None,
        no_perf_cpus: None,
    });
    let rl = KtstrVm::degrade_contention_to_overcommit(ok).expect("Ok passes through");
    assert_eq!(rl.default_cpu_mask, Some(vec![0, 1]));
}

/// degrade_contention_to_overcommit propagates a non-contention error rather
/// than masking a genuine failure as an overcommit boot.
#[test]
fn degrade_contention_to_overcommit_propagates_other_errors() {
    let boom = Err::<RunLocks, _>(anyhow::anyhow!("disk gone"));
    let err = match KtstrVm::degrade_contention_to_overcommit(boom) {
        Ok(_) => panic!("non-contention error must propagate, not degrade"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("disk gone"));
    assert!(
        err.downcast_ref::<host_topology::ResourceContention>()
            .is_none()
    );
}

#[test]
fn detect_guest_failure_surfaces_alloc_oom_panic_and_generic() {
    // Rust alloc-error on COM2 → actionable OOM cause, echoing the line.
    let c = KtstrVm::detect_guest_failure(
        "[    0.000000] Booting Linux\n",
        "memory allocation of 24 bytes failed\n",
    );
    assert!(c.contains("failed allocation"), "alloc cause: {c:?}");
    assert!(
        c.contains("memory allocation of 24 bytes failed"),
        "echoes the line: {c:?}"
    );
    // Kernel panic on COM1 → panic cause.
    let c =
        KtstrVm::detect_guest_failure("Kernel panic - not syncing: Attempted to kill init!\n", "");
    assert!(c.contains("Guest kernel panic"), "panic cause: {c:?}");
    // No marker → generic hint (preserves the original wording so the
    // error reads identically when the cause is unknown).
    let c = KtstrVm::detect_guest_failure("nothing here\n", "benign output\n");
    assert!(
        c.contains("may have panicked or rebooted"),
        "generic: {c:?}"
    );
    // Alloc (COM2) wins over a co-occurring panic (COM1): the failed
    // allocation is the root cause; the "Attempted to kill init" panic
    // is its downstream consequence.
    let c = KtstrVm::detect_guest_failure(
        "Kernel panic - not syncing: Attempted to kill init!\n",
        "memory allocation of 8 bytes failed\n",
    );
    assert!(c.contains("failed allocation"), "alloc-priority: {c:?}");
}

#[test]
fn exec_exit_from_entries_decodes_last_crc_valid_frame() {
    use crate::vmm::wire::{MSG_TYPE_EXEC_EXIT, ShmEntry};
    let mk = |msg_type, payload: Vec<u8>, crc_ok| ShmEntry {
        msg_type,
        payload,
        crc_ok,
    };
    // CRC-valid 4-byte ExecExit → decoded little-endian i32.
    assert_eq!(
        KtstrVm::exec_exit_from_entries(&[mk(
            MSG_TYPE_EXEC_EXIT,
            17i32.to_le_bytes().to_vec(),
            true
        )]),
        Some(17),
    );
    // Negative codes round-trip through the LE decode.
    assert_eq!(
        KtstrVm::exec_exit_from_entries(&[mk(
            MSG_TYPE_EXEC_EXIT,
            (-1i32).to_le_bytes().to_vec(),
            true
        )]),
        Some(-1),
    );
    // CRC-failed frame is skipped — a torn frame must never promote
    // into a bogus exit code.
    assert_eq!(
        KtstrVm::exec_exit_from_entries(&[mk(
            MSG_TYPE_EXEC_EXIT,
            17i32.to_le_bytes().to_vec(),
            false
        )]),
        None,
    );
    // Wrong payload length is skipped.
    assert_eq!(
        KtstrVm::exec_exit_from_entries(&[mk(MSG_TYPE_EXEC_EXIT, vec![1, 2, 3], true)]),
        None,
    );
    // No ExecExit frame among other types → None.
    assert_eq!(
        KtstrVm::exec_exit_from_entries(&[mk(0xDEAD_BEEF, 0i32.to_le_bytes().to_vec(), true)]),
        None,
    );
    // Multiple ExecExit frames → last (reverse-find) wins.
    assert_eq!(
        KtstrVm::exec_exit_from_entries(&[
            mk(MSG_TYPE_EXEC_EXIT, 1i32.to_le_bytes().to_vec(), true),
            mk(MSG_TYPE_EXEC_EXIT, 2i32.to_le_bytes().to_vec(), true),
        ]),
        Some(2),
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn ap_mp_state_set_correctly() {
    let topo = Topology {
        llcs: 2,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
        llc_cores: None,
    };
    let vm = kvm::KtstrKvm::new(topo, 128, false).unwrap();
    for vcpu in &vm.vcpus[1..] {
        let state = vcpu.get_mp_state().unwrap();
        assert_eq!(
            state.mp_state,
            kvm_bindings::KVM_MP_STATE_UNINITIALIZED,
            "AP should default to UNINITIALIZED"
        );
    }
}
/// Boot a real kernel and verify it produces console output.
/// No initramfs — the kernel boots to panic, which is enough to
/// confirm KVM, kernel loading, and serial console all work.
#[test]
fn boot_kernel_produces_output() {
    let kernel = crate::test_support::require_kernel();

    let vm = skip_on_contention!(
        KtstrVm::builder()
            .kernel(&kernel)
            .topology(Topology::new(1, 1, 1, 1))
            .memory_mib(256)
            .timeout(Duration::from_secs(10))
            .cmdline("loglevel=7")
            .build()
    );
    let result = skip_on_contention!(vm.run());
    assert!(
        result.stderr.contains("Linux") || result.stderr.contains("Booting"),
        "kernel console should contain boot messages"
    );
}

/// Inject a fallible helper-setup edge immediately after the interactive path
/// has spawned its AP. Returning the injected error must run
/// `InteractiveRunGuard::drop`, drain the live real-KVM AP through the shared
/// bounded helper, and reach the caller instead of detaching or hanging.
#[test]
fn boot_kernel_interactive_post_ap_setup_failure_drains_threads() {
    if !crate::test_support::cargo_ktstr_orchestrated() {
        skip!(
            "real-KVM interactive teardown regression requires cargo-ktstr orchestration; \
             set {}=1 for an intentional local run",
            crate::KTSTR_ORCHESTRATED_ENV,
        );
    }
    assert_interactive_relay_child(INTERACTIVE_RELAY_SETUP_FAILURE_MODE);
}

/// Boot with SMP topology and verify kernel detects multiple CPUs.
#[test]
fn boot_kernel_smp_topology() {
    let kernel = crate::test_support::require_kernel();

    let vm = skip_on_contention!(
        KtstrVm::builder()
            .kernel(&kernel)
            .topology(Topology::new(1, 2, 2, 1)) // 4 CPUs
            .memory_mib(256)
            .timeout(Duration::from_secs(10))
            .cmdline("loglevel=7")
            .build()
    );
    let result = skip_on_contention!(vm.run());
    assert!(!result.stderr.is_empty(), "no console output from SMP boot");
}

/// Measure VM boot time to kernel panic (no init = fastest path) AND
/// assert the boot path actually worked. The kernel boots, finds no
/// initramfs, panics; the panic timestamp IS the boot time. With
/// `panic=-1`, the kernel calls `emergency_restart()` which triggers
/// an I8042 reset (port 0x64, 0xFE via `reboot=k`), returning to
/// userspace.
///
/// Each non-skipped topology must: (a) reboot via the panic path
/// within the 10s timeout (`!timed_out`), and (b) emit a parseable
/// `[<secs>] Kernel panic` line whose timestamp yields a non-zero
/// `boot_ms` under a sane ceiling. A boot that produced no panic line,
/// a zero/garbage timestamp, or hit the timeout fails the test — so a
/// regression in the boot path or the timestamp-extraction logic
/// surfaces here instead of passing silently.
#[test]
fn bench_boot_time() {
    let kernel = crate::test_support::require_kernel();

    // A no-initramfs guest boots to panic in well under a second on an
    // idle host; allow generous slack for cold-cache / contended runs
    // while still rejecting a parse that landed on a wall-clock-scale
    // garbage value.
    const BOOT_MS_CEILING: u64 = 10_000;

    let mut ran_any = false;
    for (label, llcs, cores, threads, mem) in [("1cpu", 1, 1, 1, 256), ("4cpu", 2, 2, 1, 512)] {
        let start = Instant::now();
        let vm = match KtstrVm::builder()
            .kernel(&kernel)
            .topology(Topology::new(1, llcs, cores, threads))
            .memory_mib(mem)
            .timeout(Duration::from_secs(10))
            .build()
        {
            Ok(vm) => vm,
            // Bespoke, not skip_on_contention!: this is a per-config LOOP
            // that `continue`s to the next topology on contention, and the
            // SKIP banner carries the current config's `{label}`.
            // skip_on_contention! would `return` from the whole test and
            // cannot inject the label, so its semantics don't fit here.
            Err(e)
                if e.downcast_ref::<host_topology::ResourceContention>()
                    .is_some() =>
            {
                crate::report::test_skip(format_args!("{label}: resource contention: {e}"));
                continue;
            }
            Err(e) => panic!("{e:#}"),
        };
        let setup = start.elapsed();
        let result = skip_on_contention!(vm.run());
        ran_any = true;
        // Extract kernel timestamp from last line (e.g. "[    0.189300] Kernel panic").
        // Keep it as Option so a parse miss is a test failure, not a
        // silently-swallowed 0 (the old `unwrap_or(0)` masked exactly
        // that: a broken boot or extraction regression read as "0ms").
        let boot_ms = result
            .stderr
            .lines()
            .rev()
            .find(|l| l.contains("Kernel panic") || l.contains("end Kernel panic"))
            .and_then(|l| {
                l.trim()
                    .strip_prefix('[')
                    .and_then(|s| s.split(']').next())
                    .and_then(|s| s.trim().parse::<f64>().ok())
            })
            .map(|s| (s * 1000.0) as u64);
        eprintln!(
            "BENCH {label}: setup={:.0}ms kernel_boot={}ms wall={:.0}ms timed_out={}",
            setup.as_millis(),
            boot_ms.map_or_else(|| "?".to_string(), |ms| ms.to_string()),
            result.duration.as_millis(),
            result.timed_out,
        );
        assert!(
            !result.timed_out,
            "{label}: guest did not panic/reboot within the 10s timeout — \
             the no-initramfs fast boot path is broken. stderr tail: {:?}",
            result.stderr.lines().rev().take(5).collect::<Vec<_>>(),
        );
        // `!result.timed_out` above is the boot-success contract: a
        // broken boot never panics/reboots and trips the timeout. The
        // kernel-timestamp line is a benchmark MEASUREMENT layered on
        // top — the 1-CPU fast path occasionally emits garbled/truncated
        // serial on an otherwise-successful boot, so a parse miss here is
        // not a correctness failure once the boot has completed. Pin the
        // value to a sane range only WHEN it parses, so a regression that
        // lands on zero or garbage is still caught without flaking on a
        // benign serial-capture hiccup.
        match boot_ms {
            Some(ms) => assert!(
                ms > 0 && ms < BOOT_MS_CEILING,
                "{label}: parsed kernel boot time {ms}ms is out of the \
                 sane (0, {BOOT_MS_CEILING}) range — timestamp parse landed \
                 on a zero or garbage value",
            ),
            None => eprintln!(
                "BENCH {label}: boot completed (no timeout) but the kernel-panic \
                 timestamp line was not parseable from the serial tail — \
                 measurement skipped"
            ),
        }
    }
    if !ran_any {
        crate::report::test_skip(format_args!(
            "every topology skipped on resource contention; no boot measured"
        ));
    }
}

/// End-to-end AP-bring-up-gap boot retry: exercises the REAL pipeline
/// (guest PID-1 PANIC → serial → `extract_panic_message` →
/// `VmResult::crash_message` → `run_vm_with_ap_gap_retry` → clean second
/// boot) that the `boot_retry` unit tests can only simulate with fake
/// `VmResult`s.
///
/// The gap is injected deterministically, not raced: the guest honors a
/// `KTSTR_FAULT_AP_GAP` cmdline token (see
/// `rust_init::init::ap_gap_check_with_fault_injection`) and fabricates
/// the AP-gap PANIC on an all-online guest. The closure passed to the
/// retry helper builds a fresh init-running VM each attempt — WITH the
/// fault token on attempt 1, WITHOUT it on attempt ≥2 — so the first
/// boot PANICs with the marker and the second boots clean, proving the
/// helper both detects the marker in a genuine `crash_message` and stops
/// once a clean boot arrives.
///
/// Named `boot_kernel_*` so the nextest slow-timeout override
/// (`test(boot_kernel) | test(bench_boot)`) covers its two cold boots.
/// Follows the sibling boot tests' `skip_on_contention!` convention for
/// the /dev/kvm + host-resource gate.
#[test]
fn boot_kernel_ap_gap_retry_e2e() {
    let kernel = crate::test_support::require_kernel();
    let exe = crate::resolve_current_exe().unwrap();

    // Attempt counter and attempt-1 crash message, mutated from inside
    // the FnMut closure via interior mutability.
    let attempts = std::cell::Cell::new(0u32);
    let first_crash: std::cell::Cell<Option<String>> = std::cell::Cell::new(None);

    let outcome = crate::test_support::run_vm_with_ap_gap_retry(|| {
        let n = attempts.get() + 1;
        attempts.set(n);
        let mut builder = KtstrVm::builder()
            .kernel(&kernel)
            .init_binary(&exe)
            .topology(Topology::new(1, 1, 1, 1))
            .memory_deferred()
            .timeout(Duration::from_secs(30));
        // Inject the AP-gap fault ONLY on the first attempt; the retry
        // must then re-boot WITHOUT it and come up clean.
        if n == 1 {
            builder = builder.cmdline("KTSTR_FAULT_AP_GAP=1");
        }
        let vm = builder.build()?;
        let result = vm.run()?;
        if n == 1 {
            first_crash.set(result.crash_message.clone());
        }
        Ok(result)
    });

    // A build/run host-insufficiency error (no kernel resolvable,
    // resource contention) skips rather than fails — the retry helper
    // propagates such an Err immediately (it is not a marker result).
    let outcome = skip_on_contention!(outcome);

    // A cold / contended host can leave the fault-injection boot short of
    // the guest AP-gap PANIC when it overruns the per-boot timeout: the
    // result then carries no `crash_message`, the helper sees no marker
    // and returns after a single attempt. That is inconclusive infra
    // slowness, not a retry regression — skip, mirroring
    // `boot_kernel_with_monitor`'s slow-cold-boot skip. A non-timed-out
    // single attempt IS a real regression and trips the assert below.
    if attempts.get() == 1 && outcome.timed_out {
        skip!(
            "attempt 1's fault-injection boot overran the per-boot timeout \
             before the guest AP-gap PANIC (slow/cold host); e2e inconclusive"
        );
    }

    assert_eq!(
        attempts.get(),
        2,
        "expected exactly one fault-injected boot then one clean retry; \
         attempt 1 should PANIC with the AP-gap marker and attempt 2 \
         (no fault token) should boot clean",
    );
    assert!(
        outcome.crash_message.is_none(),
        "the retried clean boot must carry no crash_message; got {:?}",
        outcome.crash_message,
    );
    let attempt1 = first_crash
        .take()
        .expect("attempt 1 must have produced a crash_message");
    assert!(
        attempt1.contains(crate::test_support::AP_BRINGUP_GAP_MARKER),
        "attempt 1's crash_message must be the AP-bring-up-gap marker \
         routed through the guest PANIC → serial → extract_panic_message \
         path; got {attempt1:?}",
    );
}

#[test]
fn kvm_has_immediate_exit_cap() {
    let topo = Topology {
        llcs: 1,
        cores_per_llc: 1,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
        llc_cores: None,
    };
    let vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
    // KVM_CAP_IMMEDIATE_EXIT has been available since Linux 4.12.
    assert!(
        vm.has_immediate_exit,
        "KVM_CAP_IMMEDIATE_EXIT should be available on modern kernels"
    );
}
/// Boot a kernel with vmlinux available and verify the monitor
/// produces samples with meaningful runqueue data and degrades
/// gracefully for scx_root-gated paths.
///
/// No scheduler is loaded. Event counters (gated on scx_root)
/// must be None. Watchdog observation may be Some on kernels
/// with a static watchdog_timeout symbol (pre-7.1); if present,
/// the write/read roundtrip must match.
///
#[test]
fn boot_kernel_with_monitor() {
    if !crate::test_support::cargo_ktstr_orchestrated() {
        skip!(
            "test boots a real KVM VM and depends on cargo-ktstr's VM-test \
             concurrency cap to keep KVM page allocation, vCPU thread scheduling, \
             and freeze rendezvous timing within budget. Raw `cargo nextest run` \
             / `cargo test` fans 7000+ tests at full host parallelism and \
             produces a misleading `kill set by AP` failure ~5 s after VM start \
             that masks the real cause (resource starvation, not a real bug). \
             Run via `cargo ktstr test --kernel ../linux` instead, which sets \
             KTSTR_ORCHESTRATED and constrains the per-VM resource budgets."
        );
    }
    let kernel = crate::test_support::require_kernel();
    let _vmlinux = crate::test_support::require_vmlinux(&kernel);
    let exe = crate::resolve_current_exe().unwrap();

    let vm = skip_on_contention!(
        KtstrVm::builder()
            .kernel(&kernel)
            .init_binary(&exe)
            .topology(Topology::new(1, 1, 2, 1))
            .memory_deferred()
            .timeout(Duration::from_secs(15))
            .build()
    );
    let result = skip_on_contention!(vm.run());
    let Some(ref report) = result.monitor else {
        return;
    };
    // Skip (not fail) when the boot wait did not observe a sys_rdy wake
    // (boot_wait_outcome != Fired): a slow cold-cache guest boot or a
    // kill-evt race kills the monitor-setup closure before its sample
    // loop runs, yielding zero samples — inconclusive, not a monitor-data
    // regression. Mirrors the sys_rdy_releases_monitor_before_5s_timeout
    // sibling; with Fired confirmed the assertions below pin the real path.
    if report.boot_wait_outcome != crate::monitor::BootWaitOutcome::Fired {
        skip!(
            "boot wait did not observe a sys_rdy wake (boot_wait_outcome={:?}) \
             — inconclusive (slow guest boot or kill-evt race); not a \
             monitor-data regression. Total samples: {}, run duration: {:?}.",
            report.boot_wait_outcome,
            report.summary.total_samples,
            result.duration,
        );
    }
    assert!(
        report.summary.total_samples > 0,
        "monitor should have collected at least one sample"
    );

    // Scan samples in reverse for the first one where ANY CPU
    // reports rq_clock past the early-boot noise floor.
    let populated = report
        .samples
        .iter()
        .rev()
        .find(|s| s.cpus.iter().any(|c| c.rq_clock > 1_000_000))
        .expect(
            "no monitor sample showed populated runqueue data — every sample \
             had all CPUs at rq_clock <= 1ms, \
             or the monitor is reading the wrong rq offsets",
        );
    assert_eq!(
        populated.cpus.len(),
        2,
        "topology requested 2 CPUs but monitor saw {}",
        populated.cpus.len()
    );
    for (i, cpu) in populated.cpus.iter().enumerate() {
        if cpu.rq_clock <= 1_000_000 {
            continue;
        }
        assert!(
            cpu.rq_clock < 300_000_000_000,
            "cpu {i}: rq_clock must be < 300s (ns), got {}",
            cpu.rq_clock
        );
    }
    if let Some(ref obs) = report.watchdog_observation {
        assert_eq!(
            obs.expected_jiffies, obs.observed_jiffies,
            "watchdog write/read roundtrip mismatch: expected={} observed={}",
            obs.expected_jiffies, obs.observed_jiffies
        );
    }
    for (i, cpu) in populated.cpus.iter().enumerate() {
        assert!(
            cpu.event_counters.is_none(),
            "cpu {i}: event_counters must be None when no scheduler is loaded"
        );
    }
}

/// Asserts the monitor's `DATA_VALID` latch fires before the run
/// ends and records the live KASLR-randomized `page_offset`. The
/// per-iteration refresh in `monitor_loop` reads
/// `page_offset_base` from guest memory once the guest BSP has
/// completed `setup_per_cpu_areas` and KASLR randomization, then
/// latches `page_offset` for every subsequent KVA→PA translation.
/// This test fails if the latch never fires (`page_offset == 0`),
/// proving the boot signal + refresh pipeline reaches the
/// fully-populated `__per_cpu_offset[]` (every slot has bit 63
/// set, not just `[0]`) && `page_offset_resolved` AND condition
/// before the run closes.
///
/// Rationale: the same wrong `page_offset` would make every
/// `kva_to_pa` translation off by the KASLR delta and zero out
/// every monitor read. `boot_kernel_with_monitor`'s
/// `rq_clock > 1ms` assertion only fires when the read landed in
/// DRAM — but the test does not distinguish "latch never fired"
/// (page_offset stays at 0 here) from "latch fired but data still
/// pre-boot." Probing the latched value directly closes that gap.
#[test]
fn monitor_data_valid_latch_records_live_page_offset() {
    if !crate::test_support::cargo_ktstr_orchestrated() {
        skip!("{}", crate::test_support::SKIP_NOT_ORCHESTRATED_MSG);
    }
    let kernel = crate::test_support::require_kernel();
    let _vmlinux = crate::test_support::require_vmlinux(&kernel);
    let exe = crate::resolve_current_exe().unwrap();

    let vm = skip_on_contention!(
        KtstrVm::builder()
            .kernel(&kernel)
            .init_binary(&exe)
            .topology(Topology::new(1, 1, 2, 1))
            .memory_deferred()
            .timeout(Duration::from_secs(15))
            .build()
    );
    let result = skip_on_contention!(vm.run());
    let Some(ref report) = result.monitor else {
        return;
    };
    // Skip (not fail) when the boot wait did not observe a sys_rdy wake: a slow
    // cold-cache guest boot (the initramfs build alone can take ~4s) or a
    // kill-evt race kills the monitor-setup closure before its sample loop runs,
    // yielding zero samples — inconclusive, not a DATA_VALID latch regression.
    // Mirrors the `boot_kernel_with_monitor` sibling (whose 15s timeout this test
    // now also uses); the prior 5s/2s budget timed the VM out before sampling on
    // a slow host (the failure this guard + timeout eliminate).
    if report.boot_wait_outcome != crate::monitor::BootWaitOutcome::Fired {
        skip!(
            "boot wait did not observe a sys_rdy wake (boot_wait_outcome={:?}) \
             — inconclusive (slow guest boot or kill-evt race), not a \
             DATA_VALID latch regression. Total samples: {}, run duration: {:?}.",
            report.boot_wait_outcome,
            report.summary.total_samples,
            result.duration,
        );
    }
    assert!(
        report.summary.total_samples > 0,
        "monitor produced no samples — DATA_VALID latch \
         observability cannot be evaluated"
    );

    // x86_64: DATA_VALID requires page_offset_resolved (bit 63 +
    // 4 KiB alignment + stability gate) AND every
    // `__per_cpu_offset[]` slot populated (every entry with
    // bit 63 set, not just `[0]`). A non-zero `report.page_offset`
    // proves the full gate (page_offset_resolved + non-empty
    // slice + every slot kernel-half) held during at least one
    // iteration.
    assert_ne!(
        report.page_offset, 0,
        "DATA_VALID latch never fired during the run — \
         monitor.page_offset stayed at the initial 0 sentinel. \
         page_offset_base was never resolved or \
         __per_cpu_offset[0] never became non-zero before the \
         run closed",
    );

    // Bit 63 set: kernel half on x86_64 (canonical addresses
    // with VA_BITS=48 occupy 0xffff_8000_0000_0000 and above).
    // The latch's own gate enforces this same bit, so any
    // value here that lacks bit 63 means the assertion suite
    // is reading garbage rather than a live latch capture.
    assert!(
        report.page_offset & (1u64 << 63) != 0,
        "monitor.page_offset {:#x} is not in the canonical \
         upper half — page_offset_resolved gate accepted a \
         user-space address",
        report.page_offset,
    );

    // 4 KiB page alignment: kernel PAGE_OFFSET is page-aligned
    // by construction. The latch gate also enforces this; a
    // misaligned value here would be a regression in either
    // the gate or the field plumbing.
    assert_eq!(
        report.page_offset & 0xFFF,
        0,
        "monitor.page_offset {:#x} is not 4 KiB aligned",
        report.page_offset,
    );
}

/// End-to-end check that the SYS_RDY eventfd actually wakes the
/// freeze coordinator's pre-resolution boot wait. With sys_rdy
/// wired correctly the guest publishes
/// [`crate::vmm::wire::MSG_TYPE_SYS_RDY`] after
/// `mount_filesystems()`; the host stamps
/// `MonitorReport::boot_wait_outcome`
/// (`Fired`/`TimedOut`/`NotConfigured`) based on whether that
/// wake reached the boot epoll before the 5 s ceiling.
///
/// Two-stage assertion keyed on `boot_wait_outcome`:
/// - `!= Fired`: skip (inconclusive). The boot was too slow to
///   emit sys_rdy within the ceiling, a kill raced the wake, or
///   the wait did not run. The kill_evt fall-through is covered
///   by `monitor_exits_cleanly_when_guest_panics_before_sys_rdy`.
/// - `== Fired`: assert the wake propagated into the sample loop
///   (`total_samples > 0`) and the first sample landed within
///   8 s of `run_start` — pins the post-wake path (phys_base
///   poll, page_offset resolve, first iteration) against
///   pathological regressions.
///
/// Returns silently (test-skip-equivalent) when the host has
/// no kernel / no vmlinux / no scx_root etc.; the assertions
/// only fire on a real run that produced a `MonitorReport`.
#[test]
fn sys_rdy_releases_monitor_before_5s_timeout() {
    if !crate::test_support::cargo_ktstr_orchestrated() {
        skip!("{}", crate::test_support::SKIP_NOT_ORCHESTRATED_MSG);
    }
    let kernel = crate::test_support::require_kernel();
    let _vmlinux = crate::test_support::require_vmlinux(&kernel);
    let exe = crate::resolve_current_exe().unwrap();

    let vm = skip_on_contention!(
        KtstrVm::builder()
            .kernel(&kernel)
            .init_binary(&exe)
            .topology(Topology::new(1, 1, 2, 1))
            .memory_deferred()
            .timeout(Duration::from_secs(15))
            .build()
    );
    let result = skip_on_contention!(vm.run());
    let Some(ref report) = result.monitor else {
        return;
    };
    // Skip (not fail) when the boot wait did not observe a sys_rdy
    // wake (boot_wait_outcome != Fired): a slow guest boot, a
    // kill-evt race, or the wait not running — all inconclusive
    // for the sys_rdy-delivery regression this test pins, which
    // requires a confirmed wake. boot_wait_outcome distinguishes
    // them from a real "fired but the monitor never woke" defect.
    if report.boot_wait_outcome != crate::monitor::BootWaitOutcome::Fired {
        skip!(
            "boot wait did not observe a sys_rdy wake before the host's \
             5 s ceiling (boot_wait_outcome={:?}) — inconclusive (slow \
             guest boot, kill-evt race, or wait not run); not the \
             sys_rdy → monitor-wake regression this test pins. Total \
             samples: {}, run duration: {:?}. (The kill_evt fall-through \
             is covered by \
             monitor_exits_cleanly_when_guest_panics_before_sys_rdy.)",
            report.boot_wait_outcome,
            report.summary.total_samples,
            result.duration,
        );
    }
    // sys_rdy fired on the host — the monitor MUST have woken on it
    // and produced samples.
    assert!(
        report.summary.total_samples > 0,
        "sys_rdy fired but the monitor produced no samples — the wake \
         reached the boot epoll but never reached the sample loop. Run \
         wall time: {:?}",
        result.duration,
    );
    let first = report
        .samples
        .first()
        .expect("total_samples > 0 but samples list empty");
    assert!(
        first.elapsed_ms < 8_000,
        "sys_rdy fired but the first monitor sample landed at {} ms — \
         past the 8 s budget. The post-wake path (phys_base poll / \
         page_offset resolve / first iteration) is broken or \
         pathologically slow. Total samples: {}, run duration: {:?}",
        first.elapsed_ms,
        report.summary.total_samples,
        result.duration,
    );
}

/// Pins the monitor's clean-exit path when the guest never
/// reaches `send_sys_rdy`. With `init=/nonexistent` and
/// `panic=-1`, the kernel panics on its `run_init_process`
/// failure, the guest reboots immediately, and the host VM
/// loop sees the reboot and shuts down. The monitor's
/// pre-sample boot wait MUST observe the kill eventfd and
/// fall through — not block until the 5 s sys_rdy ceiling.
///
/// Wallclock budget: 12 s. The path to a kill_evt-driven
/// monitor wakeup is "kernel panic → reboot exit → BSP loop
/// sets kill → freeze coordinator writes kill_evt → monitor
/// boot wait wakes". A regression that left the monitor
/// blocked on sys_rdy alone (no kill_evt registration) would
/// hold the VM open for the full 5 s ceiling — still under
/// the 12 s budget, but a kill_evt regression that blocks
/// indefinitely on a different fd would still surface here.
///
/// `init=/nonexistent` is supplied via the builder cmdline
/// (this test sets no `init_binary`, so no `rdinit=/init`
/// token is emitted); the kernel's
/// `init/main.c::run_init_process` tries every `init=` path
/// in order and panics when none succeeds.
/// `panic=-1` is the existing default in
/// `KtstrVm::setup_memory`'s cmdline composition; setting it
/// again via `cmdline_extra` is a no-op for the kernel parser
/// (last token wins, and both tokens specify the same value).
#[test]
fn monitor_exits_cleanly_when_guest_panics_before_sys_rdy() {
    let kernel = crate::test_support::require_kernel();
    let _vmlinux = crate::test_support::require_vmlinux(&kernel);

    let vm = skip_on_contention!(
        KtstrVm::builder()
            .kernel(&kernel)
            .topology(Topology::new(1, 1, 2, 1))
            .memory_mib(256)
            .timeout(Duration::from_secs(15))
            .cmdline("init=/nonexistent panic=-1")
            .build()
    );
    let result = skip_on_contention!(vm.run());
    // The VM loop must shut down via the kernel's reboot exit
    // path, not via the builder's 15 s timeout.
    assert!(
        !result.timed_out,
        "guest never panicked / rebooted within 15 s — the test's \
         premise (panic-before-sys_rdy → kernel reboot → VM exit) \
         is not holding. Stderr tail: {:?}",
        result.stderr.lines().rev().take(5).collect::<Vec<_>>(),
    );
    // Wallclock budget: 12 s. The monitor's 5 s sys_rdy ceiling
    // plus VM setup + guest panic + reboot + teardown nominally
    // finishes in 3-5 s on an idle host. The 12 s budget absorbs
    // host contention (observed runs at 8-9 s under load) while
    // still catching a regression that blocks the boot wait
    // indefinitely (e.g. kill_evt unregistered, sys_rdy not
    // promoted to the eventfd) — that path would either hit the
    // builder's 15 s timeout (caught above) or sit on the 5 s
    // ceiling under heavy overhead (well past 12 s).
    // The 12 s budget is an IDLE-host expectation: setup + panic +
    // reboot + teardown nominally finishes in 3-5 s. Under host dilation
    // the wall inflates without the monitor misbehaving, so the budget is
    // enforced only on a quiet host (D <= 1.1). The test's real invariant
    // — the monitor woke on kill_evt rather than hanging on the sys_rdy
    // ceiling — is already proven by the guest-rebooted check above (a
    // hung monitor would have hit the builder's 15 s timeout, failing
    // that assert). Under dilation the budget is downgraded to a note.
    let d = result.host_vcpu_schedstat.and_then(|s| s.dilation());
    if d.is_none_or(|d| d <= 1.1) {
        assert!(
            result.duration < Duration::from_secs(12),
            "VM ran for {:?} on a quiet host (D={}) — past the 12 s budget. \
             The monitor's boot wait did not wake on kill_evt; the loop sat on \
             the sys_rdy ceiling instead. timed_out={}, exit_code={}",
            result.duration,
            d.map_or_else(|| "n/a".to_string(), |d| format!("{d:.2}x")),
            result.timed_out,
            result.exit_code,
        );
    } else if result.duration >= Duration::from_secs(12) {
        eprintln!(
            "ktstr: monitor_exits_cleanly ran {:?} (> 12 s budget) under host \
             dilation D={:.2}x — budget not enforced (the no-hang invariant is \
             covered by the guest-rebooted check); wall is a host artifact",
            result.duration,
            d.unwrap_or(0.0),
        );
    }
}

/// Asserts at least one of the first 5 monitor samples (no
/// reverse scan) has `rq_clock > 1ms` on at least one CPU.
/// This pins the SYS_RDY
/// → DATA_VALID pipeline's load-bearing semantics: when
/// `send_sys_rdy` fires, the guest BSP has already completed
/// `setup_per_cpu_areas` AND KASLR randomization AND
/// `mount_filesystems()`, so the early per-iteration refreshes in
/// `monitor_loop` produce in-DRAM PAs and `read_rq_stats`
/// returns live counters — no zero-pad sentinel period and no
/// reverse scan needed to find a populated sample.
///
/// Distinct from `boot_kernel_with_monitor`'s reverse-scan
/// assertion: that test passes if ANY sample (even the last
/// one, after seconds of pre-boot zeros) is populated. This
/// test fails if none of the first 5 samples is populated —
/// which would indicate the monitor started sampling before
/// the guest had the rq fields written, defeating the whole
/// point of the SYS_RDY gate.
///
#[test]
fn first_sample_has_valid_rq_clock_thanks_to_sys_rdy() {
    if !crate::test_support::cargo_ktstr_orchestrated() {
        skip!("{}", crate::test_support::SKIP_NOT_ORCHESTRATED_MSG);
    }
    let kernel = crate::test_support::require_kernel();
    let _vmlinux = crate::test_support::require_vmlinux(&kernel);
    let exe = crate::resolve_current_exe().unwrap();

    let vm = skip_on_contention!(
        KtstrVm::builder()
            .kernel(&kernel)
            .init_binary(&exe)
            .topology(Topology::new(1, 1, 2, 1))
            .memory_deferred()
            .timeout(Duration::from_secs(15))
            .build()
    );
    let result = skip_on_contention!(vm.run());
    let Some(ref report) = result.monitor else {
        return;
    };
    // Skip (not fail) when the boot wait did not observe a sys_rdy
    // wake (boot_wait_outcome != Fired): a slow guest boot that
    // emitted sys_rdy past the host's 5 s ceiling, a kill-evt race,
    // or the wait not running — all inconclusive for the FIRST-sample
    // rq_clock contract this test pins, which requires a confirmed
    // wake. Mirrors sys_rdy_releases_monitor_before_5s_timeout; the
    // boot_wait_outcome discriminator (monitor::BootWaitOutcome)
    // exists for exactly this distinction. Without it, a debug-init
    // boot slower than the 5 s ceiling produces zero samples and
    // looks like a regression (the original "intermittent
    // no-samples"), when it is just inconclusive.
    if report.boot_wait_outcome != crate::monitor::BootWaitOutcome::Fired {
        skip!(
            "boot wait did not observe a sys_rdy wake before the host's \
             5 s ceiling (boot_wait_outcome={:?}) — inconclusive (slow \
             guest boot / kill-evt race), not the FIRST-sample rq_clock \
             contract this test pins. total_samples={}",
            report.boot_wait_outcome,
            report.summary.total_samples,
        );
    }
    // The FIRST-sample contract is evaluated over the first 5 samples. Under
    // host oversubscription the guest can run several times slower, so the
    // monitor (~100ms cadence) collects only a handful of samples in the run
    // window — and the few it gets can predate the guest's rq_clock
    // population even though SYS_RDY fired. That is inconclusive for this
    // contract, NOT a sys_rdy regression. Skip when fewer than the 5-sample
    // evaluation window were collected (a sample-starved run). A real
    // regression — the monitor starting before the rq fields are populated —
    // still FAILS on a normally-loaded host, which collects tens of samples
    // (~100ms cadence over seconds) so the assertion below runs with a full
    // window. Whether the monitor samples AT ALL is pinned by
    // `boot_kernel_with_monitor` / `sys_rdy_releases_monitor_before_5s_timeout`,
    // not here.
    if report.summary.total_samples < 5 {
        skip!(
            "monitor collected only {} sample(s), fewer than the 5-sample \
             first-window — sample-starved run (slow guest under host load), \
             inconclusive for the FIRST-sample rq_clock contract, not a \
             sys_rdy regression",
            report.summary.total_samples,
        );
    }
    let early_populated = report
        .samples
        .iter()
        .take(5)
        .any(|s| s.cpus.iter().any(|c| c.rq_clock > 1_000_000));
    assert!(
        early_populated,
        "none of the first 5 monitor samples had any CPU with \
         rq_clock > 1ms — SYS_RDY did not wait for the guest's \
         runqueue fields to be populated. \
         total_samples: {}, run duration: {:?}",
        report.summary.total_samples, result.duration,
    );
}

/// Regression guard for the `scx_sched.watchdog_timeout` host-write
/// mechanism. Boots a VM with scx-ktstr loaded plus a distinctive
/// 2-second watchdog override, then asserts the monitor loop
/// observed the expected jiffies value in guest memory.
///
/// Skips gracefully when: no host kernel available, no vmlinux for
/// BTF, `scx_root` symbol or `scx_sched.watchdog_timeout` BTF field
/// missing, or the scheduler failed to attach. Real failure
/// requires the override path to silently stop writing — which is
/// exactly what we want to catch.
#[test]
fn watchdog_timeout_override_lands_in_guest_memory() {
    let kernel = crate::test_support::require_kernel();
    let vmlinux = crate::test_support::require_vmlinux(&kernel);

    // Version-dependent skips, in order of check cost. scx_root
    // is a 6.16+ symbol; its absence means either the kernel
    // predates the 6.16 scx_sched refactor (sched_ext still
    // present via the older scx_ops API, e.g. 6.14) or sched_ext
    // was not compiled in. Either way this test has nothing to
    // verify — skip. watchdog_offsets depends on BTF field layout
    // that only exists on 7.1+ kernels where
    // `scx_sched.watchdog_timeout` is a struct field.
    let syms = crate::test_support::require_kernel_symbols(&vmlinux);
    if syms.scx_root.is_none() {
        skip!("scx_root not present (needs Linux 6.16+ with sched_ext enabled)");
    }
    let offsets = crate::test_support::require_kernel_offsets(&vmlinux);
    if offsets.watchdog_offsets.is_none() {
        skip!(
            "scx_sched.watchdog_timeout field not in BTF \
             (needs Linux 7.1+; pre-7.1 exposes watchdog timeout as a file-scope \
             scx_watchdog_timeout symbol handled separately)"
        );
    }

    const TIMEOUT_SECS: u64 = 2;
    let hz = crate::monitor::guest_kernel_hz(Some(&kernel));
    let expected_jiffies = TIMEOUT_SECS * hz;

    let sched_bin = crate::test_support::require_binary("scx-ktstr");

    let vm = skip_on_contention!(
        KtstrVm::builder()
            .kernel(&kernel)
            .topology(Topology::new(1, 1, 1, 1))
            .memory_mib(256)
            .timeout(Duration::from_secs(5))
            .scheduler_binary(&sched_bin)
            .watchdog_timeout(Duration::from_secs(TIMEOUT_SECS))
            .build()
    );
    let result = skip_on_contention!(vm.run());
    let report = result.monitor.as_ref().expect(
        "ktstr: monitor report missing — require_kernel_offsets, scx_root, and \
         watchdog_offsets all resolved at setup, so monitor initialization must \
         have succeeded. A None report here is a bug in monitor startup",
    );
    let Some(obs) = &report.watchdog_observation else {
        // scx_root remained null for the whole run — the scheduler
        // never attached. Not a watchdog regression — skip.
        skip!(
            "watchdog observation missing — the scheduler did not attach \
             (scx_root remained null throughout the run)"
        );
    };
    assert_eq!(
        obs.expected_jiffies, expected_jiffies,
        "expected_jiffies recorded by monitor ({}) does not match {} * HZ {} = {}",
        obs.expected_jiffies, TIMEOUT_SECS, hz, expected_jiffies,
    );
    assert_eq!(
        obs.observed_jiffies, obs.expected_jiffies,
        "host wrote {} jiffies to scx_sched.watchdog_timeout but guest memory holds {} — host-write mechanism broken",
        obs.expected_jiffies, obs.observed_jiffies,
    );
}

/// Prove the kernel uses the host-written watchdog timeout.
///
/// Sets a 300-second watchdog and runs the scheduler for 30s.
/// If the host write is effective, the kernel's watchdog timer
/// uses 300s and no stall exit occurs. If the write were
/// ineffective (kernel ignoring the value), the default timeout
/// would apply and could spuriously fire on a slow guest.
#[test]
fn watchdog_override_prevents_stall_exit() {
    let kernel = crate::test_support::require_kernel();
    let _vmlinux = crate::test_support::require_vmlinux(&kernel);

    let sched_bin = crate::test_support::require_binary("scx-ktstr");

    let vm = skip_on_contention!(
        KtstrVm::builder()
            .kernel(&kernel)
            .topology(Topology::new(1, 1, 2, 1))
            .memory_mib(256)
            .timeout(Duration::from_secs(30))
            .scheduler_binary(&sched_bin)
            .watchdog_timeout(Duration::from_secs(300))
            .build()
    );
    let result = skip_on_contention!(vm.run());
    // Prior versions asserted `result.success` here. That's the
    // conjunction `!timed_out && exit_code == 0`, which depends
    // on init writing MSG_TYPE_EXIT to SHM before the AP-triggered
    // reboot propagates through the watchdog-kicks-BSP path. When
    // init is slightly slow (cold host cache, contended CPU),
    // exit_code lands at -1 (BSP run-loop default) and the
    // assertion fires even though the thing under test — scx
    // stall-exit behavior — is unaffected. Assert the actual
    // invariants instead: no guest crash, no scheduler
    // stall-exit markers in guest output. These are what would
    // change if the 300s watchdog override had failed.
    assert!(
        result.crash_message.is_none(),
        "no crash expected with 300s watchdog: {:?}",
        result.crash_message
    );
    // SchedulerDied / SchedulerNotAttached lifecycle frames are
    // written by start_scheduler in rust_init on attach failure
    // or scheduler exit (now via `send_lifecycle` on the bulk
    // data port — pre-bulk-port-migration these were COM2
    // sentinel strings). "sched_ext: disabled" is the kernel's
    // own disable message when scx tears down a scheduler (e.g.
    // on watchdog stall). Any of these appearing proves the
    // watchdog either fired or the scheduler exited for another
    // reason — either way the test's "no stall exit" invariant
    // is broken.
    let output = &result.output;
    let stderr = &result.stderr;
    let lifecycle_phase_seen = |phase: crate::vmm::wire::LifecyclePhase| -> bool {
        let Some(ref drain) = result.guest_messages else {
            return false;
        };
        drain.entries.iter().any(|e| {
            e.msg_type == crate::vmm::wire::MSG_TYPE_LIFECYCLE
                && e.crc_ok
                && !e.payload.is_empty()
                && crate::vmm::wire::LifecyclePhase::from_wire(e.payload[0]) == Some(phase)
        })
    };
    assert!(
        !lifecycle_phase_seen(crate::vmm::wire::LifecyclePhase::SchedulerDied),
        "scheduler no longer running after 30s — either the watchdog fired or the \
         scheduler exited for another reason. output: {output:?}, stderr: {stderr:?}",
    );
    assert!(
        !lifecycle_phase_seen(crate::vmm::wire::LifecyclePhase::SchedulerNotAttached),
        "scheduler did not attach — no watchdog override to evaluate. \
         output: {output:?}, stderr: {stderr:?}",
    );
    assert!(
        !output.contains("sched_ext: disabled") && !stderr.contains("sched_ext: disabled"),
        "kernel disabled sched_ext during run — a watchdog stall or ops \
         error fired. output: {output:?}, stderr: {stderr:?}",
    );
    if let Some(ref report) = result.monitor
        && let Some(ref obs) = report.watchdog_observation
    {
        let hz = crate::monitor::guest_kernel_hz(Some(&kernel));
        let expected_jiffies = 300 * hz;
        assert_eq!(
            obs.expected_jiffies, expected_jiffies,
            "watchdog override should be 300s * HZ={hz}"
        );
        assert_eq!(
            obs.observed_jiffies, obs.expected_jiffies,
            "write/read roundtrip mismatch"
        );
    }
}

/// Validate that sched_domain data is populated when BTF offsets
/// resolve. Domains are kernel-built at boot and do not require a
/// scheduler.
///
/// Gates on sched_domain_offsets BTF availability. Uses a 2-CPU
/// topology so the domain tree spans multiple CPUs.
///
#[test]
fn sched_domain_data_populated() {
    if !crate::test_support::cargo_ktstr_orchestrated() {
        skip!("{}", crate::test_support::SKIP_NOT_ORCHESTRATED_MSG);
    }
    let kernel = crate::test_support::require_kernel();
    let vmlinux = crate::test_support::require_vmlinux(&kernel);

    let offsets = crate::test_support::require_kernel_offsets(&vmlinux);
    if offsets.sched_domain_offsets.is_none() {
        skip!(
            "sched_domain BTF fields not found (likely CONFIG_SMP=n; \
             struct sched_domain is absent or incomplete in BTF on UP kernels, \
             and on pre-6.17 kernels the rq.sd field is also compiled out)"
        );
    }

    let exe = crate::resolve_current_exe().unwrap();

    let vm = skip_on_contention!(
        KtstrVm::builder()
            .kernel(&kernel)
            .init_binary(&exe)
            .topology(Topology::new(1, 1, 2, 1))
            .memory_deferred()
            // 15s window (was 5s): the monitor must catch at least one
            // GENEROUS cap: the VM exits the instant the monitor has a
            // sample carrying populated domains, so an idle host still
            // finishes in ~15 s — the cap only gives a dilated boot
            // (contended CI runner) room to reach post-SMP domain-tree
            // construction before the run is killed. A tight cap made a
            // slow-but-functional boot read as a domain-resolution bug.
            //
            // Original note: the monitor must catch at least one
            // sample after the kernel builds the sched_domain tree,
            // which lands late in boot (post-SMP-bringup). A 5s window
            // flaked on slow hosts where boot consumed it before rq.sd
            // populated. watchdog_timeout is the guest scx stall
            // detector, inert here (no scheduler), so only this timeout
            // bounds the run.
            .timeout(Duration::from_secs(60))
            .watchdog_timeout(Duration::from_secs(2))
            .build()
    );
    let result = skip_on_contention!(vm.run());
    let report = result.monitor.as_ref().expect(
        "ktstr: monitor report missing — require_kernel_offsets and \
         sched_domain_offsets resolved at setup, so monitor initialization \
         must have succeeded. A None report here is a bug in monitor startup",
    );

    // Zero samples means the monitor never latched its evidence channels
    // — the guest never booted far enough to be observed. On a QUIET host
    // that is a real monitor-startup bug; on a loaded CI runner it is the
    // environmental boot starvation (a 2-vCPU guest that cannot boot to a
    // samplable state inside the window). `run_env_was_loaded` is the
    // discriminator: it uses machine loadavg, NOT the guest's vCPU
    // dilation, because this idle-boot guest's threads barely run so their
    // dilation reads ~1 even on a saturated host — only host load captures
    // the contention that slowed the boot.
    if report.summary.total_samples == 0 {
        if run_env_was_loaded(&result) {
            skip!(
                "monitor collected 0 samples under host load — the guest could \
                 not boot to a samplable state in the window (environmental; \
                 the offset/monitor path is exercised on quiet hosts)"
            );
        }
        panic!(
            "monitor collected 0 samples on a quiet host — a monitor-startup \
             or evidence-channel bug, not host load"
        );
    }

    // Scan samples in reverse chronological order for the first
    // one where at least one CPU reports a non-empty sched_domains
    // list. `.last()` alone flaked on slow hosts where the final
    // sample was captured before the kernel finished building the
    // domain tree — sched_domains is populated via kernel threads
    // at boot, and the per-CPU `rq.sd` pointer lags the first rq
    // samples. Reverse-searching guards against that boot race:
    // if ANY sample in the run carries populated domains, the
    // kernel path works and the assertion passes.
    let populated_sample = report.samples.iter().rev().find(|s| {
        s.cpus.iter().any(|c| {
            c.sched_domains
                .as_ref()
                .is_some_and(|doms| !doms.is_empty())
        })
    });
    // No sample carried populated domains. On a QUIET host this is a real
    // defect (offsets wrong, or the tree never built); under host dilation
    // it is the environmental boot race — a starved guest that never
    // reached post-SMP domain construction inside the (already generous)
    // run cap. Discriminate with the host dilation D (1 + Σrun_delay/Σon_cpu
    // over the vCPU threads): a dilated run SKIPs (non-verdict), a quiet run
    // FAILs. Absent D (schedstats off) is treated as "cannot attribute" →
    // skip, so a host we cannot measure never produces a false failure.
    // Handled in the function body (not a closure) so `skip!` returns from
    // the test.
    let Some(populated) = populated_sample else {
        // Same environmental discriminator as the zero-samples case: a
        // loaded host slows the boot-time kernel thread that builds the
        // domain tree past the run window, so no sample carries domains.
        if run_env_was_loaded(&result) {
            skip!(
                "sched_domains never populated across {} samples under host \
                 load — the guest was too starved to reach post-SMP domain-tree \
                 construction in the run window (environmental, not a resolution \
                 bug; reproduces only on a contended host)",
                report.samples.len(),
            );
        }
        panic!(
            "no sample had any CPU with non-empty sched_domains across {} \
             collected samples on a quiet host — monitor samples may be racing \
             the boot-time kernel thread that builds the domain tree, or \
             `rq.sd` offsets are wrong",
            report.samples.len(),
        );
    };

    for cpu in &populated.cpus {
        if let Some(ref doms) = cpu.sched_domains {
            if doms.is_empty() {
                continue;
            }
            for w in doms.windows(2) {
                assert!(
                    w[1].level > w[0].level,
                    "domain levels must be strictly increasing: {} -> {}",
                    w[0].level,
                    w[1].level
                );
            }
            assert!(
                doms[0].span_weight >= 2,
                "lowest domain span_weight must be >= 2 for a 2-CPU topology, got {}",
                doms[0].span_weight
            );
            for dom in doms {
                assert!(
                    dom.span_weight > 0,
                    "domain level {} span_weight must be > 0",
                    dom.level
                );
            }
        }
    }
}
#[test]
fn builder_performance_mode_false_no_validation() {
    // performance_mode=false should not trigger validation, even with
    // a topology that exceeds host capacity.
    let exe = crate::resolve_current_exe().unwrap();
    let result = KtstrVmBuilder::default()
        .kernel(&exe)
        .topology(Topology::new(1, 1, 1, 1))
        .performance_mode(false)
        .build();
    // Bespoke, not skip_on_contention!: the `Err(e) => panic!` carries this
    // test's NEGATIVE assertion — "performance_mode=false should not
    // validate host topology" — which skip_on_contention!'s generic
    // panic!("{e:#}") would erase. Only RC skips here; the trivial 1×1×1×1
    // topology never reaches the TI / perf-mode classes.
    match result {
        Ok(_) => {}
        Err(e)
            if e.downcast_ref::<host_topology::ResourceContention>()
                .is_some() =>
        {
            skip!("resource contention: {e}");
        }
        Err(e) => panic!("performance_mode=false should not validate host topology: {e:#}",),
    }
}

#[test]
fn builder_performance_mode_oversubscribed_fails() {
    let exe = crate::resolve_current_exe().unwrap();
    let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
    let too_many = host_topo.total_cpus() as u32 + 1;
    let result = KtstrVmBuilder::default()
        .kernel(&exe)
        .topology(Topology::new(1, 1, too_many, 1))
        .performance_mode(true)
        .build();
    match result {
        Ok(_) => panic!("oversubscribed topology should fail"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("performance_mode"),
                "error should mention performance_mode: {msg}",
            );
        }
    }
}

#[test]
fn builder_performance_mode_too_many_llcs_fails() {
    let exe = crate::resolve_current_exe().unwrap();
    let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
    let too_many_llcs = host_topo.llc_groups.len() as u32 + 1;
    // Need total vCPUs + 1 service CPU to fit without oversubscription.
    if (too_many_llcs as usize + 1) <= host_topo.total_cpus() {
        let result = KtstrVmBuilder::default()
            .kernel(&exe)
            .topology(Topology::new(1, too_many_llcs, 1, 1))
            .performance_mode(true)
            .build();
        assert!(
            result.is_err(),
            "more virtual LLCs than host LLCs should fail",
        );
    }
}

#[test]
fn builder_performance_mode_valid_succeeds() {
    let exe = crate::resolve_current_exe().unwrap();
    let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
    if host_topo.total_cpus() < 3 {
        skip!("need >= 3 host CPUs for performance_mode test");
    }
    let result = KtstrVmBuilder::default()
        .kernel(&exe)
        .topology(Topology::new(1, 1, 2, 1))
        .performance_mode(true)
        .build();
    match result {
        Ok(_) => {}
        Err(e)
            if e.downcast_ref::<host_topology::ResourceContention>()
                .is_some() =>
        {
            skip!("resource contention: {e}");
        }
        Err(e)
            if e.downcast_ref::<host_topology::PerfModeUnavailable>()
                .is_some() =>
        {
            // The host fundamentally cannot honor perf-mode (too few CPUs
            // for an exclusive LLC + a service CPU — e.g. a single-LLC
            // host whose LLC spans every CPU). Skip rather than panic: the
            // "valid perf-mode topology builds" invariant cannot be
            // exercised on a host that cannot do perf-mode at all.
            skip!("performance mode unavailable: {e}");
        }
        Err(e) => panic!("valid topology with performance_mode should build: {e:#}",),
    }
}

#[test]
fn builder_performance_mode_preserves_in_vm() {
    let exe = crate::resolve_current_exe().unwrap();
    let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
    if host_topo.total_cpus() < 3 {
        skip!("need >= 3 host CPUs for performance_mode test");
    }
    let vm = skip_on_contention!(
        KtstrVmBuilder::default()
            .kernel(&exe)
            .topology(Topology::new(1, 1, 2, 1))
            .performance_mode(true)
            .build()
    );
    assert!(vm.performance_mode);
}

#[test]
fn builder_performance_mode_false_preserves_in_vm() {
    let exe = crate::resolve_current_exe().unwrap();
    let vm = skip_on_contention!(
        KtstrVmBuilder::default()
            .kernel(&exe)
            .topology(Topology::new(1, 1, 1, 1))
            .performance_mode(false)
            .build()
    );
    assert!(!vm.performance_mode);
}

#[test]
fn builder_performance_mode_mbind_nodes_populated() {
    let exe = crate::resolve_current_exe().unwrap();
    let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
    if host_topo.total_cpus() < 3 {
        skip!("need >= 3 host CPUs for performance_mode test");
    }
    let vm = KtstrVmBuilder::default()
        .kernel(&exe)
        .topology(Topology::new(1, 1, 2, 1))
        .performance_mode(true)
        .build();
    if let Ok(vm) = vm {
        assert!(
            !vm.mbind_node_map.is_empty(),
            "mbind_node_map should be populated for performance_mode",
        );
    }
}
