use super::*;
use crate::sync::MutexExt;

#[test]
fn mkdir_p_creates_nested() {
    let _tempdir_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-rust-init-test-mkdir-")
        .tempdir()
        .unwrap();
    let base = _tempdir_keep_alive.path();
    let nested = base.join("a/b/c");
    mkdir_p(nested.to_str().unwrap());
    assert!(nested.exists());
}

/// `mkdir_p` against an already-existing directory must leave
/// it present (the existing-dir branch is a no-op, NOT a
/// remove-and-recreate). Uses a fresh `tempfile::TempDir` so the
/// directory's identity is observable: capture the dir's inode
/// before the no-op call and assert it is byte-identical after,
/// proving `mkdir_p` did not unlink+recreate (which would mint a
/// new inode) — a stronger guard than mere `is_dir()`, which a
/// remove-and-recreate would still satisfy.
#[test]
fn mkdir_p_existing_is_noop() {
    use std::os::unix::fs::MetadataExt;
    // Create the directory first so the second call hits the
    // existing-dir (recursive-create-is-idempotent) branch.
    let tempdir = tempfile::Builder::new()
        .prefix("ktstr-rust-init-test-mkdir-noop-")
        .tempdir()
        .unwrap();
    let path = tempdir.path();
    assert!(path.is_dir(), "tempdir setup must create the directory");
    let ino_before = fs::metadata(path).unwrap().ino();

    // The no-op call: the directory already exists.
    mkdir_p(path.to_str().unwrap());

    // Postcondition: still a directory, and the SAME directory
    // (same inode) — not unlinked and recreated.
    assert!(
        path.is_dir(),
        "mkdir_p on an existing dir must leave it present"
    );
    let ino_after = fs::metadata(path).unwrap().ino();
    assert_eq!(
        ino_before, ino_after,
        "mkdir_p on an existing dir must be a true no-op \
         (inode {ino_before} != {ino_after} means it removed+recreated)",
    );
}

#[test]
fn exec_shell_line_echo_redirect() {
    let _tempfile_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-rust-init-echo-test-")
        .tempfile()
        .unwrap();
    let path = _tempfile_keep_alive.path().to_str().unwrap();
    assert!(exec_shell_line(&format!("echo 42 > {path}")).is_ok());
    let content = fs::read_to_string(_tempfile_keep_alive.path()).unwrap();
    assert_eq!(content, "42\n");
}

#[test]
fn exec_shell_line_unsupported_input_returns_err() {
    // Comments are filtered upstream in exec_shell_script;
    // a bare "# comment" reaching exec_shell_line is an
    // unsupported command. Pinning the Err signal so the
    // partial-apply counter in exec_shell_script catches
    // typo'd lines instead of silently skipping them.
    assert!(exec_shell_line("# this is a comment").is_err());
}

/// `exec_shell_script` emits an error-level summary instead of
/// silently partial-applying. A script with mixed-success lines
/// must surface the failure count to the operator — the
/// prior implementation only logged per-line errors with no
/// roll-up, so an operator scanning init-log for the
/// sched_enable result couldn't easily count failures.
#[test]
#[tracing_test::traced_test]
fn exec_shell_script_counts_per_line_failures() {
    // Build a script with one valid echo + one unsupported
    // command. The valid line writes a sentinel value to a
    // tempfile so the test asserts the partial-apply did
    // produce the expected side effect — proving the function
    // didn't short-circuit on first failure.
    let _payload_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-tax-payload-")
        .tempfile()
        .unwrap();
    let payload_path = _payload_keep_alive.path().to_str().unwrap();
    let mut script = tempfile::Builder::new()
        .prefix("ktstr-tax-script-")
        .tempfile()
        .unwrap();
    use std::io::Write;
    writeln!(script, "echo 7 > {payload_path}").unwrap();
    writeln!(script, "not_a_supported_command").unwrap();
    script.flush().unwrap();
    exec_shell_script(script.path().to_str().unwrap());
    let payload = fs::read_to_string(payload_path).unwrap();
    assert_eq!(payload, "7\n", "valid line must still apply");

    // The headline behavior is the roll-up summary: a single
    // error-level emit reporting the failure count. Pin both the
    // message body and the structured count fields so a
    // regression that drops the summary, miscounts, or downgrades
    // the level trips here (the side-effect assert above alone
    // would still pass under any of those regressions).
    assert!(
        logs_contain("1 line(s) failed, 1 line(s) ok"),
        "partial-apply summary must report the per-line counts in its message",
    );
    assert!(
        logs_contain("fail_count=1"),
        "summary must carry the structured fail_count field",
    );
    assert!(
        logs_contain("ok_count=1"),
        "summary must carry the structured ok_count field",
    );
}

/// File-not-found returns silently (legitimate "no script"
/// case for the optional sched_enable/sched_disable hooks).
/// Pins the debug-level skip so a future refactor that flipped
/// the missing-file path to error-level — or to a partial-apply
/// summary — would surface here. `traced_test` installs a
/// capturing subscriber with a `<crate>=trace` filter, so the
/// DEBUG skip line is captured and observable.
#[test]
#[tracing_test::traced_test]
fn exec_shell_script_missing_file_returns_silently() {
    exec_shell_script("/tmp/ktstr-tax-nonexistent-script-path");
    // The NotFound arm must log the skip at DEBUG and return —
    // observe the concrete no-op: the debug skip line is present.
    assert!(
        logs_contain("no script (skipping)"),
        "missing-file must take the DEBUG skip branch",
    );
    // ...and must NOT escalate to the error arm ("read failed"),
    // nor emit a partial-apply summary ("line(s) failed"): a
    // regression flipping NotFound to error-level or running the
    // line loop on absent content would trip one of these.
    assert!(
        !logs_contain("read failed"),
        "missing-file must NOT hit the error-level read-failed arm",
    );
    assert!(
        !logs_contain("line(s) failed"),
        "missing-file must NOT emit a partial-apply failure summary",
    );
}

#[test]
fn shell_mode_not_requested_in_test() {
    // /proc/cmdline exists on the host but won't contain KTSTR_MODE=shell.
    assert!(!shell_mode_requested());
}

#[test]
fn disk_template_mode_not_requested_in_test() {
    // /proc/cmdline on the host won't contain KTSTR_MODE=disk_template.
    assert!(!disk_template_mode_requested());
}

#[test]
fn disk_template_dispatch_precedes_shell_when_both_present() {
    // The dispatch order in `ktstr_guest_init` is:
    //   1. disk_template_mode_requested → run mkfs + reboot, never returns
    //   2. shell_mode_requested → drop into busybox shell
    //   3. test dispatch
    //
    // If both KTSTR_MODE entries appear in /proc/cmdline (e.g.
    // operator typo, host-side cmdline-construction bug), the
    // disk_template path MUST win — running shell mode against
    // a disk that the operator intended to format would skip
    // the formatting step silently. Pin the token-parser
    // semantics so a future refactor that changes the matching
    // logic (regex, prefix-only, or per-token last-wins) does
    // not silently invert the precedence.
    let cmdline = "ro KTSTR_MODE=disk_template KTSTR_MODE=shell console=ttyS0";
    // Both checks see their token in the cmdline.
    assert!(cmdline_contains_token(cmdline, "KTSTR_MODE=disk_template"));
    assert!(cmdline_contains_token(cmdline, "KTSTR_MODE=shell"));
    // The dispatch order in ktstr_guest_init runs the
    // disk_template check FIRST, so the disk_template path is
    // taken and the shell branch is never reached. This test
    // pins the token-parser invariant; the dispatch-order
    // invariant lives in the code at ktstr_guest_init's
    // disk-template-mode block.
    //
    // Reverse-token order produces the same result — the
    // checks are commutative and dispatch-order is the only
    // disambiguator.
    let cmdline_reversed = "ro KTSTR_MODE=shell KTSTR_MODE=disk_template console=ttyS0";
    assert!(cmdline_contains_token(
        cmdline_reversed,
        "KTSTR_MODE=disk_template"
    ));
    assert!(cmdline_contains_token(cmdline_reversed, "KTSTR_MODE=shell"));
}

#[test]
fn cmdline_contains_token_exact_match_not_prefix() {
    // Matching is whole-token, not prefix. A future kernel
    // cmdline that introduces e.g. `KTSTR_MODE=shell_extended`
    // must not accidentally trip the shell-mode dispatch.
    assert!(cmdline_contains_token(
        "KTSTR_MODE=shell",
        "KTSTR_MODE=shell"
    ));
    assert!(!cmdline_contains_token(
        "KTSTR_MODE=shell_extended",
        "KTSTR_MODE=shell"
    ));
    assert!(!cmdline_contains_token(
        "prefix_KTSTR_MODE=shell",
        "KTSTR_MODE=shell"
    ));
    assert!(!cmdline_contains_token("", "KTSTR_MODE=shell"));
}

#[test]
fn count_online_cpus_returns_some() {
    // On any Linux host, /sys/devices/system/cpu/online exists.
    let count = count_online_cpus();
    assert!(count.is_some());
    assert!(count.unwrap() >= 1);
}

#[test]
fn parse_online_cpus_single_index() {
    assert_eq!(parse_online_cpus("0"), Some(1));
    assert_eq!(parse_online_cpus("7"), Some(1));
}

#[test]
fn parse_online_cpus_simple_range() {
    assert_eq!(parse_online_cpus("0-3"), Some(4));
    assert_eq!(parse_online_cpus("4-7"), Some(4));
}

#[test]
fn parse_online_cpus_mixed_ranges_and_singles() {
    assert_eq!(parse_online_cpus("0,2,4"), Some(3));
    assert_eq!(parse_online_cpus("0-1,4-7"), Some(6));
    assert_eq!(parse_online_cpus("0-2,4,6-7"), Some(6));
}

#[test]
fn parse_online_cpus_strips_trailing_newline() {
    // /sys/devices/system/cpu/online emits a trailing '\n'.
    assert_eq!(parse_online_cpus("0-3\n"), Some(4));
}

#[test]
fn parse_online_cpus_single_cpu_zero() {
    assert_eq!(parse_online_cpus("0-0"), Some(1));
}

#[test]
fn parse_online_cpus_empty_content_is_none() {
    assert_eq!(parse_online_cpus(""), None);
    assert_eq!(parse_online_cpus("   "), None);
    assert_eq!(parse_online_cpus("\n"), None);
}

#[test]
fn parse_online_cpus_non_numeric_is_none() {
    assert_eq!(parse_online_cpus("abc"), None);
    assert_eq!(parse_online_cpus("0-abc"), None);
    assert_eq!(parse_online_cpus("a-3"), None);
    assert_eq!(parse_online_cpus("0,abc,3"), None);
    // Empty tokens from malformed list shapes — the kernel never
    // produces these but the parser must reject loudly rather
    // than silently skip.
    assert_eq!(parse_online_cpus("0,"), None); // trailing comma
    assert_eq!(parse_online_cpus(",0"), None); // leading comma
    assert_eq!(parse_online_cpus("-3"), None); // leading dash → empty range start
}

#[test]
fn parse_online_cpus_inverted_range_is_none() {
    // Defensive: an inverted range "10-3" would previously
    // panic in debug (overflow) or wrap in release. checked_sub
    // returns None instead.
    assert_eq!(parse_online_cpus("10-3"), None);
}

#[test]
fn parse_online_cpus_extreme_range_does_not_overflow() {
    // u32::MAX - 0 + 1 overflows u32; checked_add returns None.
    assert_eq!(parse_online_cpus(&format!("0-{}", u32::MAX)), None);
}

#[test]
fn parse_online_cpus_large_topology() {
    // 256 vCPUs as a single range.
    assert_eq!(parse_online_cpus("0-255"), Some(256));
}

#[test]
fn parse_cpu_list_enumerates_ids() {
    assert_eq!(parse_cpu_list("0"), Some(vec![0]));
    assert_eq!(parse_cpu_list("0-3"), Some(vec![0, 1, 2, 3]));
    assert_eq!(parse_cpu_list("0-1,4-7"), Some(vec![0, 1, 4, 5, 6, 7]));
    assert_eq!(parse_cpu_list("0-2,4,6-7"), Some(vec![0, 1, 2, 4, 6, 7]));
    assert_eq!(parse_cpu_list("0-3\n"), Some(vec![0, 1, 2, 3]));
}

#[test]
fn parse_cpu_list_rejects_malformed() {
    assert_eq!(parse_cpu_list(""), None);
    assert_eq!(parse_cpu_list("   "), None);
    assert_eq!(parse_cpu_list("abc"), None);
    assert_eq!(parse_cpu_list("0,"), None); // trailing comma
    assert_eq!(parse_cpu_list("10-3"), None); // inverted range
    // Absurd range would balloon the id vec — rejected at the ceiling,
    // parallel to parse_online_cpus's overflow guard.
    assert_eq!(parse_cpu_list(&format!("0-{}", u32::MAX)), None);
}

#[test]
fn offline_possible_cpus_all_online_reports_no_gap() {
    // Healthy 128-vCPU boot: possible == online, no missing CPUs.
    let report = offline_possible_cpus("0-127", "0-127").unwrap();
    assert!(report.missing.is_empty());
    assert_eq!(report.online, 128);
    assert_eq!(report.possible, 128);
}

#[test]
fn offline_possible_cpus_names_missing_ids() {
    // 128 possible, CPUs 4 and 6 failed AP bring-up → online has a hole.
    let online = "0-3,5,7-127";
    let report = offline_possible_cpus("0-127", online).unwrap();
    assert_eq!(report.missing, vec![4, 6]);
    assert_eq!(report.online, 126);
    assert_eq!(report.possible, 128);
    // The message the operator sees, built from this report through the
    // SAME formatter the guest PANICs with. Pins the exact wording AND
    // that it still carries the host-side retry marker — a reword that
    // dropped the marker would silently disable the boot retry.
    let msg = format_ap_gap_message(&report.missing, report.online, report.possible);
    assert_eq!(
        msg,
        "CPUs [4, 6] failed to come online (AP bring-up failed; 126/128 online)"
    );
    assert!(
        msg.contains(crate::test_support::AP_BRINGUP_GAP_MARKER),
        "guest AP-gap message must contain the host retry marker: {msg}",
    );
}

#[test]
fn offline_possible_cpus_unparseable_list_is_none() {
    // A procfs hiccup on either list degrades to "skip the check".
    assert!(offline_possible_cpus("garbage", "0-127").is_none());
    assert!(offline_possible_cpus("0-127", "").is_none());
}

/// Zero budget: loop exits within one sleep step and emits the
/// WARN with the expected diagnostic fields. The `traced_test`
/// attribute installs a capturing subscriber so `logs_contain`
/// can verify the message body and each structured field
/// rendered into the log line.
///
/// Pins both the time bound AND the WARN content — a regression
/// that silently dropped the structured fields, or moved the
/// emit to a lower log level, would trip the logs_contain
/// assertions.
#[test]
#[tracing_test::traced_test]
fn send_sys_rdy_retry_exits_when_budget_exhausted() {
    let budget = std::time::Duration::from_millis(0);
    let addrs = crate::vmm::wire::KernAddrs::new(0, 0, None);
    // Use a path that won't exist on the host so the loop
    // takes the port_exists=false branch — no real device
    // interaction in unit tests.
    let port_path = std::path::Path::new("/tmp/ktstr-test-nonexistent-port-please-do-not-create");
    let t0 = std::time::Instant::now();
    send_sys_rdy_with_retry(budget, 1, &addrs, Some(port_path));
    let elapsed = t0.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "send_sys_rdy_with_retry with zero budget took {elapsed:?}; \
         must exit within one sleep step (with slack for CI load)",
    );
    // Verify the WARN content. tracing-test's logs_contain does
    // a substring match against captured log lines.
    assert!(
        logs_contain("send_sys_rdy failed within boot budget"),
        "WARN message must be emitted on budget exhaustion",
    );
    for field in [
        "budget_ms=0",
        "vcpus=1",
        "elapsed_ms=",
        "port_exists=false",
        "kern_addrs_sent=false",
    ] {
        assert!(
            logs_contain(field),
            "WARN must include structured field `{field}`",
        );
    }
    assert!(
        logs_contain("send_sys_rdy-timeout"),
        "WARN must include the docs anchor pointer",
    );
}

/// Wall-time floor invariant: the loop must wait at least
/// `budget` wall-clock time before emitting the WARN (with the
/// port absent). Parameterized over several (budget, vcpus)
/// combinations to pin the invariant across the production
/// budget range — replaces the deleted count-based formula
/// coupling test.
#[test]
fn send_sys_rdy_retry_respects_budget_across_sizes() {
    let port_path = std::path::Path::new("/tmp/ktstr-test-nonexistent-port-please-do-not-create");
    let addrs = crate::vmm::wire::KernAddrs::new(0, 0, None);
    for &(budget_ms, vcpus) in &[(50u64, 1u32), (150, 2), (250, 8), (500, 32)] {
        let budget = std::time::Duration::from_millis(budget_ms);
        let t0 = std::time::Instant::now();
        send_sys_rdy_with_retry(budget, vcpus, &addrs, Some(port_path));
        let elapsed = t0.elapsed();
        assert!(
            elapsed >= budget,
            "(budget={budget_ms}ms, vcpus={vcpus}): elapsed {elapsed:?} \
             < budget; the loop must wait at least the budget before \
             the WARN fires",
        );
        // Generous upper bound — CI runners under load can
        // stretch std::thread::sleep significantly. Keep wide
        // enough to never flake while still catching a runaway
        // (e.g. count-based loop ignoring the budget).
        let cap = budget + std::time::Duration::from_secs(2);
        assert!(
            elapsed < cap,
            "(budget={budget_ms}ms, vcpus={vcpus}): elapsed {elapsed:?} \
             exceeded {cap:?}; the loop should not overshoot by more \
             than ~2s of slack",
        );
    }
}

/// Port-exists branch coverage: when the device-node path
/// resolves, the loop takes the `if port_path.exists()` arm
/// and calls `send_kern_addrs` / `send_sys_rdy`. In host
/// context those calls no-op via `assert_guest_context` (and
/// the production named-port resolver cannot use the injected
/// tempfile), so kern_addrs_sent stays false and the loop
/// exhausts the budget. The WARN must report
/// `port_exists=true, kern_addrs_sent=false` — the
/// diagnostic combination the troubleshooting doc explains as
/// "the port device exists but writes failed".
#[test]
#[tracing_test::traced_test]
fn send_sys_rdy_retry_reports_port_exists_when_path_resolves() {
    let tmpfile =
        tempfile::NamedTempFile::new().expect("create tempfile to stand in for the bulk port");
    let budget = std::time::Duration::from_millis(150);
    let addrs = crate::vmm::wire::KernAddrs::new(0, 0, None);
    send_sys_rdy_with_retry(budget, 4, &addrs, Some(tmpfile.path()));
    assert!(
        logs_contain("port_exists=true"),
        "WARN must report port_exists=true when the path resolves",
    );
    assert!(
        logs_contain("kern_addrs_sent=false"),
        "WARN must report kern_addrs_sent=false when host-context \
         writes no-op via assert_guest_context",
    );
    assert!(logs_contain("vcpus=4"), "WARN must include the vcpus value",);
}

/// Regression pin for the fast-fail retry THROTTLE in
/// `send_sys_rdy_with_retry`: when the port node exists but every send
/// fails (host context), the loop must retry at a BOUNDED rate, not
/// hot-spin the guest init (PID 1) thread. The 100 ms guard-rail sleep
/// is the sole hot-spin guard on that path — without it the loop
/// re-enters Stage 1 (fast-path Done, since the node exists) and
/// re-fails with zero delay. Elapsed wall-clock cannot distinguish
/// throttle-present from -absent (the deadline gate bounds both to
/// ~budget), so this pins the ITERATION COUNT: with the ~100 ms throttle
/// a 400 ms budget yields only a handful of iterations, whereas a
/// busy-spin would yield thousands.
#[test]
fn send_sys_rdy_retry_throttles_fast_fail_does_not_hot_spin() {
    use std::sync::atomic::Ordering;
    let tmpfile =
        tempfile::NamedTempFile::new().expect("create tempfile to stand in for the bulk port");
    let budget = std::time::Duration::from_millis(400);
    let addrs = crate::vmm::wire::KernAddrs::new(0, 0, None);
    SEND_SYS_RDY_RETRY_ITERS.store(0, Ordering::Relaxed);
    send_sys_rdy_with_retry(budget, 4, &addrs, Some(tmpfile.path()));
    let iters = SEND_SYS_RDY_RETRY_ITERS.load(Ordering::Relaxed);
    // ~budget/100ms ≈ 4-5 throttled iterations; allow generous slack for
    // scheduling jitter. A dropped throttle would spin into the
    // thousands, so the upper bound is the regression guard.
    assert!(
        (2..=12).contains(&iters),
        "port-exists + always-failing-send must retry at the ~100 ms \
         throttled rate (≈ budget/100ms iterations), not hot-spin: got \
         {iters} iterations for a 400 ms budget",
    );
}

#[test]
fn parse_topo_from_cmdline_not_present_on_host() {
    // Host /proc/cmdline won't contain KTSTR_TOPO.
    assert!(parse_topo_from_cmdline().is_none());
}

/// A child that exits immediately must be observed as `Died`
/// well before the poll timeout. This is the regression gate
/// for the old unconditional `sleep(1s)` — we don't want to
/// wait a full second to notice an instant crash.
#[test]
fn poll_startup_detects_early_death_quickly() {
    let mut child = std::process::Command::new("/bin/true")
        .spawn()
        .expect("spawn /bin/true");
    let start = std::time::Instant::now();
    let status = poll_startup(
        &mut child,
        std::time::Duration::from_millis(10),
        std::time::Duration::from_secs(1),
    );
    let elapsed = start.elapsed();
    assert!(
        matches!(status, StartupStatus::Died),
        "expected Died, got {status:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_millis(500),
        "early death must be detected fast, took {elapsed:?}"
    );
}

/// A child that stays alive past the poll window must be
/// observed as `Alive` within ~timeout — the caller accepts
/// this as "scheduler ready" without any longer wait.
#[test]
fn poll_startup_reports_alive_after_timeout() {
    let mut child = std::process::Command::new("/bin/sleep")
        .arg("5")
        .spawn()
        .expect("spawn /bin/sleep");
    let start = std::time::Instant::now();
    let status = poll_startup(
        &mut child,
        std::time::Duration::from_millis(20),
        std::time::Duration::from_millis(100),
    );
    let elapsed = start.elapsed();
    let _ = child.kill();
    let _ = child.wait();
    assert!(
        matches!(status, StartupStatus::Alive),
        "expected Alive, got {status:?}"
    );
    assert!(
        elapsed >= std::time::Duration::from_millis(100),
        "Alive must wait the full timeout, took only {elapsed:?}"
    );
    // Poll is allowed one extra interval of slack.
    assert!(
        elapsed < std::time::Duration::from_millis(300),
        "Alive should not overshoot timeout significantly, took {elapsed:?}"
    );
}

// -- kill_scheduler_process tests --
//
// The kill helper is the building block for Op::DetachScheduler /
// Op::RestartScheduler / Op::ReplaceScheduler dispatch (follow-up
// work). Tests pin the three outcome variants
// (AlreadyExited / ExitedAfterSigterm / EscalatedToSigkill) plus
// the InvalidPid error path. The escalation test deliberately
// installs SIGTERM-ignoring trap to force the SIGKILL branch —
// matches the scx-scheduler-without-handler scenario the
// EscalatedToSigkill variant is named for.

/// `pid` <= 0 must surface InvalidPid immediately without
/// touching the kernel. POSIX kill(2) reserves 0 (caller's pgrp)
/// and negative values (signal pgrp), neither of which the
/// scheduler-lifecycle call site ever wants. The check is a
/// programming-error guard for callers that fail to validate a pid.
#[test]
fn kill_scheduler_process_invalid_pid_returns_err() {
    assert_eq!(
        kill_scheduler_process(0, std::time::Duration::from_millis(50)),
        Err(KillSchedulerError::InvalidPid),
    );
    assert_eq!(
        kill_scheduler_process(-1, std::time::Duration::from_millis(50)),
        Err(KillSchedulerError::InvalidPid),
    );
}

/// A pid that was never alive (or was reaped before the call)
/// surfaces as AlreadyExited — the idempotent-detach case that
/// lifecycle Op semantics rely on (detaching nothing is success,
/// not error).
#[test]
fn kill_scheduler_process_already_exited_pid_yields_already_exited() {
    // Spawn /bin/true and let it exit + reap before kill_scheduler_process
    // is called. /bin/true exits ~immediately.
    let mut child = std::process::Command::new("/bin/true")
        .spawn()
        .expect("spawn /bin/true");
    let pid = child.id() as libc::pid_t;
    let _ = child.wait();
    // After wait, /proc/{pid} has been released. Poll briefly
    // to ensure procfs cleanup has propagated.
    let mut waits = 0u32;
    while proc_pid_alive(pid as u32) && waits < 50 {
        std::thread::sleep(std::time::Duration::from_millis(10));
        waits += 1;
    }
    assert!(
        !proc_pid_alive(pid as u32),
        "procfs should have released the pid after wait"
    );
    assert_eq!(
        kill_scheduler_process(pid, std::time::Duration::from_millis(50)),
        Ok(KillSchedulerOutcome::AlreadyExited),
    );
}

/// A responsive child (one that catches SIGTERM and exits)
/// produces ExitedAfterSigterm. /bin/sleep installs the default
/// SIGTERM handler (terminate-on-signal — kernel-side action,
/// no userspace handler, but the kernel exit completes well
/// inside the grace window).
///
/// Installs SIGCHLD=SIG_IGN for the test duration — matches the
/// production guest-init disposition, where the kernel
/// auto-reaps children so `/proc/{pid}` disappears at exit
/// without an explicit `waitpid`. Without this the test would
/// race with the standard SIGCHLD=SIG_DFL test environment that
/// keeps the exited child as a zombie (procfs entry persists)
/// until the explicit Child::wait, breaking the poll_pid_gone
/// observation that kill_scheduler_process relies on.
#[test]
fn kill_scheduler_process_responsive_child_yields_exited_after_sigterm() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_IGN);

    let mut child = std::process::Command::new("/bin/sleep")
        .arg("60")
        .spawn()
        .expect("spawn /bin/sleep");
    let pid = child.id() as libc::pid_t;
    let outcome = kill_scheduler_process(pid, std::time::Duration::from_millis(500));
    // Best-effort reap. Under SIG_IGN the kernel auto-reaps so
    // Child::wait returns ECHILD; the call is harmless either
    // way. SigchldGuard's Drop restores the previous disposition
    // before the test exits so subsequent tests aren't poisoned.
    let _ = child.wait();
    assert_eq!(outcome, Ok(KillSchedulerOutcome::ExitedAfterSigterm));
}

/// A child that ignores SIGTERM must produce
/// EscalatedToSigkill. /bin/sh -c 'trap "" TERM; sleep 30'
/// installs an empty SIGTERM trap, so SIGTERM is no-op'd and
/// the SIGKILL fallback is the only way to terminate. Pins the
/// escalation branch against a regression that drops the
/// SIGKILL step or treats SIGTERM-grace-exhausted as success.
///
/// SIGCHLD=SIG_IGN for the same reason as the
/// `_responsive_child_` sibling test — see that test's docs.
///
/// Synchronizes via filesystem marker rather than a timing-based
/// settle delay so the test is immune to CI scheduling jitter.
/// The shell does `trap '' TERM; touch <marker>; sleep 30`, the
/// test polls for marker existence with a generous 5s deadline,
/// THEN sends SIGTERM. This eliminates the race where the kill
/// can land before the shell has installed its trap — the marker
/// existence is a kernel-observable HAPPENS-AFTER signal proving
/// the trap installation already returned. Marker filename uses
/// a fixed path because SIGCHLD_TEST_LOCK serializes the tests
/// that write SIGCHLD disposition, so concurrent writers cannot
/// collide.
#[test]
fn kill_scheduler_process_ignoring_sigterm_child_escalates_to_sigkill() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_IGN);

    let marker = "/tmp/ktstr_kill_test_trap_ready";
    // Clear any stale marker from a prior aborted run.
    let _ = std::fs::remove_file(marker);

    let mut child = std::process::Command::new("/bin/sh")
        .arg("-c")
        // `exec sleep 30` forces sleep to replace sh in-place
        // so SIGKILL on the sh pid kills the sleep too. Without
        // `exec`, sh runs `touch` first (which forces sh to stay
        // around as a process), then forks for `sleep` — and
        // SIGKILL on the sh pid leaves the orphaned sleep
        // re-parented to init, surfaced by nextest's leak
        // detector as a "leaky" test.
        .arg(format!("trap '' TERM; touch {marker}; exec sleep 30"))
        .spawn()
        .expect("spawn /bin/sh");
    let pid = child.id() as libc::pid_t;

    // Wait for the marker — proves the trap is installed.
    let marker_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !std::path::Path::new(marker).exists() {
        if std::time::Instant::now() >= marker_deadline {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(marker);
            panic!(
                "shell did not create trap-ready marker within 5s — \
                 /bin/sh failed to start or filesystem is too slow"
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Tight SIGTERM grace (200ms) so the test doesn't burn a
    // full second on the polite-shutdown timeout. The trap is
    // confirmed installed via the marker so the shell will
    // ignore SIGTERM and force the SIGKILL escalation.
    let outcome = kill_scheduler_process(pid, std::time::Duration::from_millis(200));
    let _ = child.wait();
    let _ = std::fs::remove_file(marker);
    assert_eq!(outcome, Ok(KillSchedulerOutcome::EscalatedToSigkill));
}

/// SIGCHLD signal disposition is process-wide, so the
/// `with_sigchld_default_*`, `poll_startup_*_under_sigchld_ignore`,
/// `kill_scheduler_process_*`, and `sched_pid_*` regression tests
/// must serialize. Without this lock, two
/// concurrent `libc::signal(SIGCHLD, ...)` calls from different
/// test threads could leave SIGCHLD in an unexpected state when
/// either test inspects or restores it. Acquired via
/// [`crate::sync::MutexExt::lock_unpoisoned`] so a panic in one
/// signal-aware test does not poison every other one.
static SIGCHLD_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII guard that snapshots the current SIGCHLD disposition on
/// construction and restores it on drop. Tests that flip
/// `SIGCHLD` to `SIG_IGN` to reproduce the PID-1 environment
/// must not bleed that disposition into the rest of the test
/// run — the cargo nextest binary runs every test in a single
/// process under threads, so a leaked `SIG_IGN` would make
/// every subsequent `Child::wait` (in unrelated tests) return
/// ECHILD. `signal(2)` returns the previous handler; we restore
/// it verbatim via a second `signal` call.
struct SigchldGuard {
    prev: libc::sighandler_t,
}

impl SigchldGuard {
    fn install(handler: libc::sighandler_t) -> Self {
        // SAFETY: `libc::signal` accepts any process-wide signal
        // disposition; the returned value is the previous
        // handler, captured here for restoration in `Drop`.
        let prev = unsafe { libc::signal(libc::SIGCHLD, handler) };
        Self { prev }
    }
}

impl Drop for SigchldGuard {
    fn drop(&mut self) {
        // SAFETY: `self.prev` was returned by an earlier
        // `libc::signal` call on the same signal number;
        // re-installing it is the documented restore pattern.
        unsafe {
            libc::signal(libc::SIGCHLD, self.prev);
        }
    }
}

/// Regression for the verifier false-green race: under SIGCHLD=SIG_DFL an
/// exited scheduler is a zombie, so its pidfd is readable while `/proc/<pid>`
/// still exists. The scheduler-exit monitor must treat the pidfd edge as exit,
/// and cleanup must reject the zombie even though kill(2) accepts SIGKILL for
/// it.
#[test]
fn verifier_cleanup_rejects_pidfd_ready_zombie() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_DFL);

    let mut child = std::process::Command::new("/bin/sh")
        .args(["-c", "exit 7"])
        .spawn()
        .expect("spawn exiting child");
    let pid = child.id() as libc::pid_t;
    assert_eq!(
        crate::sync::pidfd_poll_exited(pid, std::time::Duration::from_secs(5)),
        crate::sync::PidfdWait::Exited,
    );
    assert!(
        Path::new(&format!("/proc/{pid}")).exists(),
        "SIGCHLD=SIG_DFL keeps the exited child as a proc-visible zombie",
    );
    assert!(
        sched_exit_observed(libc::POLLIN).expect("valid pidfd event"),
        "pidfd POLLIN must outrank a still-present zombie proc entry",
    );
    assert!(
        !sched_exit_observed(0).expect("empty pidfd event"),
        "numeric pid/proc disappearance is not scheduler identity evidence",
    );
    assert!(sched_exit_observed(libc::POLLNVAL).is_err());

    // Linux accepts a signal directed at a zombie. That syscall success must
    // not be mistaken for proof that cleanup killed a live scheduler.
    let kill_sent = unsafe { libc::kill(pid, libc::SIGKILL) } == 0;
    assert!(
        kill_sent,
        "Linux should accept SIGKILL for the still-unreaped zombie"
    );
    let status = child.wait().expect("reap zombie");
    assert_eq!(status.code(), Some(7));
    assert!(
        !verifier_cleanup_kill_confirmed(kill_sent, Some(&status)),
        "natural exit status must fail even when kill(2) returned success",
    );
}

fn open_pidfd_for_sched_exit_test(pid: u32) -> OwnedFd {
    // SAFETY: pidfd_open on a child owned by this test with flags zero. A
    // non-negative return is a new descriptor transferred into OwnedFd.
    let raw =
        unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0u32) as libc::c_int };
    assert!(
        raw >= 0,
        "pidfd_open({pid}) failed: {}",
        std::io::Error::last_os_error(),
    );
    unsafe { OwnedFd::from_raw_fd(raw) }
}

static OWNER_TEST_SPEC_A: crate::test_support::SchedulerSpec =
    crate::test_support::SchedulerSpec::Discover("owner-test-a");
static OWNER_TEST_SPEC_B: crate::test_support::SchedulerSpec =
    crate::test_support::SchedulerSpec::Discover("owner-test-b");

#[test]
fn restart_identity_comes_from_declared_boot_spec() {
    install_boot_scheduler(Some(&OWNER_TEST_SPEC_A));
    let restart = boot_scheduler().expect("declared boot scheduler");
    assert!(std::ptr::eq(restart, &OWNER_TEST_SPEC_A));
    assert!(
        !std::ptr::eq(restart, &OWNER_TEST_SPEC_B),
        "Restart must not inherit the staged current scheduler identity"
    );
}

fn scheduler_owner_test_process(
    generation: u64,
    log_path: &str,
    scheduler: &'static crate::test_support::SchedulerSpec,
) -> CurrentSchedulerProcess {
    let child = std::process::Command::new("/bin/sleep")
        .arg("30")
        .spawn()
        .expect("spawn scheduler owner stand-in");
    let pidfd = open_pidfd_for_sched_exit_test(child.id());
    let monitor_pidfd = pidfd.try_clone().expect("clone exact monitor pidfd");
    let monitor = start_sched_exit_monitor(
        child.id(),
        monitor_pidfd,
        None,
        Arc::new(AtomicBool::new(false)),
        None,
    )
    .expect("start scheduler owner monitor");
    CurrentSchedulerProcess {
        generation,
        child,
        pidfd,
        log_path: log_path.to_string(),
        scheduler: Some(scheduler),
        monitor: Some(monitor),
        drop_reap_exhausted: false,
    }
}

#[test]
fn scheduler_owner_install_take_keeps_one_coherent_record() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_DFL);
    let slot = std::sync::Mutex::new(None);
    let mut owner = scheduler_process_owner_for_test(&slot);
    let process = scheduler_owner_test_process(41, "/tmp/owner-a.log", &OWNER_TEST_SPEC_A);
    let pid = process.pid();

    assert!(owner.install(process).is_ok(), "install empty local owner");
    let current = owner.current().expect("owner published");
    assert_eq!(current.pid(), pid);
    assert_eq!(current.generation, 41);
    assert_eq!(current.log_path, "/tmp/owner-a.log");
    assert!(std::ptr::eq(
        current.scheduler.expect("scheduler spec"),
        &OWNER_TEST_SPEC_A
    ));
    assert!(current.monitor.is_some());

    let mut taken = owner.take().expect("take coherent owner");
    assert!(owner.current().is_none());
    assert_eq!(taken.pid(), pid);
    assert_eq!(taken.generation, 41);
    assert_eq!(taken.log_path, "/tmp/owner-a.log");
    let _ = taken.stop_monitor();
    taken.terminate_exact().expect("exact cleanup");
}

#[test]
fn exact_owner_cleanup_leaves_unrelated_process_alive() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_DFL);
    let mut unrelated = std::process::Command::new("/bin/sleep")
        .arg("30")
        .spawn()
        .expect("spawn unrelated process");
    let mut owned = scheduler_owner_test_process(42, "/tmp/owner-b.log", &OWNER_TEST_SPEC_B);
    let _ = owned.stop_monitor();
    owned.terminate_exact().expect("exact owner cleanup");
    assert!(
        matches!(unrelated.try_wait(), Ok(None)),
        "pidfd cleanup of owner A must not touch unrelated process B"
    );
    unrelated.kill().expect("kill unrelated process");
    unrelated.wait().expect("reap unrelated process");
}

#[test]
fn pidfd_liveness_does_not_consume_child_wait_status() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_DFL);
    let mut owned = scheduler_owner_test_process(43, "/tmp/owner-c.log", &OWNER_TEST_SPEC_A);
    let _ = owned.stop_monitor();
    assert_eq!(
        owned.send_signal(libc::SIGKILL),
        Ok(PidfdSignalOutcome::Delivered)
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while owned.is_alive().expect("pidfd liveness") {
        assert!(
            std::time::Instant::now() < deadline,
            "pidfd did not report scheduler exit"
        );
        std::thread::yield_now();
    }
    let status = owned
        .child
        .wait()
        .expect("pidfd liveness must leave Child status unconsumed");
    assert_eq!(status.signal(), Some(libc::SIGKILL));
    assert_eq!(
        owned.send_signal(libc::SIGKILL),
        Ok(PidfdSignalOutcome::AlreadyExited),
        "an already-readable/reaped pidfd must not claim signal delivery"
    );
}

#[test]
fn sigchld_ignored_terminal_reap_is_not_reported_as_timeout() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_IGN);
    let mut owned = scheduler_owner_test_process(44, "/tmp/owner-sigign.log", &OWNER_TEST_SPEC_A);
    let _ = owned.stop_monitor();
    assert_eq!(
        owned.send_signal(libc::SIGKILL),
        Ok(PidfdSignalOutcome::Delivered)
    );

    let outcome = owned.reap_bounded_status(std::time::Duration::from_secs(5));
    assert!(
        matches!(outcome, SchedulerReapOutcome::TerminalWithoutStatus),
        "pidfd terminal readiness under SIGCHLD=SIG_IGN must be distinguished \
         from a live-process timeout (got {outcome:?})"
    );
    assert!(
        !owned.drop_reap_exhausted,
        "an auto-reaped terminal process must not consume Drop's reap budget"
    );
}

#[test]
fn sched_exit_monitor_reports_an_unreaped_zombie() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_DFL);

    let mut child = std::process::Command::new("/bin/sh")
        .args(["-c", "exit 9"])
        .spawn()
        .expect("spawn exiting scheduler stand-in");
    let pid = child.id() as libc::pid_t;
    assert_eq!(
        crate::sync::pidfd_poll_exited(pid, std::time::Duration::from_secs(5)),
        crate::sync::PidfdWait::Exited,
    );
    assert!(
        Path::new(&format!("/proc/{pid}")).exists(),
        "the child must remain an unreaped proc-visible zombie",
    );

    let pidfd = open_pidfd_for_sched_exit_test(pid as u32);
    let stop = start_pending_sched_exit_monitor(
        pid as u32,
        pidfd,
        None,
        Arc::new(AtomicBool::new(false)),
        None,
    )
    .expect("start pending monitor on zombie");
    assert_eq!(
        stop.commit(),
        Err(SchedExitTerminal::Exited),
        "an unreaped zombie cannot be committed as the current scheduler"
    );
    assert!(
        stop.stop_and_join(),
        "pidfd readiness must propagate through the actual monitor join result",
    );

    let status = child.wait().expect("reap scheduler stand-in");
    assert_eq!(status.code(), Some(9));
}

#[test]
fn sched_exit_monitor_accepts_an_already_readable_handed_off_pidfd() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_DFL);

    let mut child = std::process::Command::new("/bin/sleep")
        .arg("30")
        .spawn()
        .expect("spawn scheduler stand-in");
    let pid = child.id();
    let pidfd = open_pidfd_for_sched_exit_test(pid);
    child.kill().expect("kill scheduler stand-in");
    child.wait().expect("reap scheduler stand-in");
    assert!(
        !Path::new(&format!("/proc/{pid}")).exists(),
        "the numeric pid must be gone before monitor installation",
    );

    let stop =
        start_pending_sched_exit_monitor(pid, pidfd, None, Arc::new(AtomicBool::new(false)), None)
            .expect("install pending monitor from the handed-off readable pidfd");
    assert_eq!(
        stop.commit(),
        Err(SchedExitTerminal::Exited),
        "an already-readable pidfd must reject owner publication"
    );
    assert!(
        stop.stop_and_join(),
        "an already-readable pidfd must be observed without reopening the numeric pid",
    );
}

#[test]
fn sched_exit_monitor_thread_spawn_failure_is_synchronous() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_DFL);

    let mut child = std::process::Command::new("/bin/sleep")
        .arg("30")
        .spawn()
        .expect("spawn scheduler stand-in");
    let pid = child.id();
    let pidfd = open_pidfd_for_sched_exit_test(pid);
    let error = match start_sched_exit_monitor_with_spawn_failure_for_test(pid, pidfd) {
        Ok(_) => panic!("injected thread spawn failure unexpectedly installed a monitor"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert!(
        error
            .to_string()
            .contains("injected sched-exit monitor thread spawn failure"),
        "unexpected spawn error: {error}",
    );

    child.kill().expect("kill scheduler stand-in");
    child.wait().expect("reap scheduler stand-in");
}

#[test]
fn sched_exit_monitor_only_waits_forever_with_a_wake_writer() {
    assert_eq!(sched_exit_poll_timeout(7, true), -1);
    assert_eq!(sched_exit_poll_timeout(7, false), 250);
    assert_eq!(sched_exit_poll_timeout(-1, true), 250);
    assert_eq!(sched_exit_poll_timeout(-1, false), 250);
}

#[test]
fn sched_exit_monitor_without_wake_writer_stops_on_finite_poll() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_DFL);

    let mut child = std::process::Command::new("/bin/sleep")
        .arg("30")
        .spawn()
        .expect("spawn live child");
    let pidfd = open_pidfd_for_sched_exit_test(child.id());
    let stop = start_sched_exit_monitor_without_wake_writer_for_test(child.id(), pidfd)
        .expect("start scheduler-exit monitor");
    let started = std::time::Instant::now();
    assert!(
        !stop.stop_and_join(),
        "a live child stopped by the caller is not an observed scheduler exit",
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "missing wake writer must fall back to a finite poll instead of deadlocking",
    );
    child.kill().expect("kill test child");
    child.wait().expect("reap test child");
}

#[test]
fn verifier_cleanup_requires_enabled_state() {
    use crate::scenario::ops::ScxState;

    assert!(verifier_cleanup_state_confirmed(Some(ScxState::Enabled)));
    for state in [
        None,
        Some(ScxState::Enabling),
        Some(ScxState::Disabling),
        Some(ScxState::Disabled),
    ] {
        assert!(
            !verifier_cleanup_state_confirmed(state),
            "cleanup must fail closed for {state:?}",
        );
    }
}

/// Verifier probes do not run the scenario driver which normally publishes
/// ScenarioEnd. Their init-owned completion boundary must therefore close
/// Body before Phase 6, while ordinary tests retain the pause used to exclude
/// cleanup from scenario timing.
#[test]
fn verifier_completion_closes_body_before_cleanup() {
    assert_eq!(
        post_workload_boundary(true),
        PostWorkloadBoundary::ScenarioEndThenPause
    );
    assert_eq!(
        post_workload_boundary(false),
        PostWorkloadBoundary::ScenarioPause
    );
}

#[test]
fn verifier_cleanup_accepts_its_own_sigkill_status() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_DFL);

    let mut child = std::process::Command::new("/bin/sleep")
        .arg("30")
        .spawn()
        .expect("spawn live child");
    child.kill().expect("send intentional SIGKILL");
    let status = child.wait().expect("reap SIGKILLed child");
    assert!(
        verifier_cleanup_kill_confirmed(true, Some(&status)),
        "wait status from ktstr's SIGKILL is the positive cleanup proof",
    );
}

/// Regression: with SIGCHLD set to `SIG_IGN`, a bare
/// `Command::status()` returns `Err(ECHILD)` because the kernel
/// auto-reaps the child before `waitpid` can observe it.
/// `with_sigchld_default` must restore `SIG_DFL` for the
/// closure's lifetime so `waitpid` reaps and reports a real
/// status. After the closure returns, `SIG_IGN` must be
/// restored.
#[test]
fn with_sigchld_default_captures_real_exit_status() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_IGN);

    // Sanity: under SIG_IGN, plain Command::status() returns
    // Err(ECHILD) — proves the ambient state matches PID 1.
    let bare = Command::new("/bin/true").status();
    assert!(
        bare.is_err(),
        "under SIG_IGN, Command::status must fail with ECHILD; got {bare:?}",
    );

    // Helper restores SIG_DFL for the closure body, so the same
    // Command::status() succeeds and reports exit code 0.
    let wrapped = with_sigchld_default(|| Command::new("/bin/true").status());
    let status = wrapped.expect("with_sigchld_default must capture status");
    assert_eq!(
        status.code(),
        Some(0),
        "/bin/true must exit 0 under helper; got {status:?}",
    );

    // After the closure returns, SIG_IGN must be back in place
    // so subsequent guest children continue to be auto-reaped.
    // SAFETY: signal(SIG_IGN) reads the previous disposition
    // and re-installs SIG_IGN; we compare the previous value to
    // SIG_IGN to assert nothing changed it underneath us.
    let after = unsafe { libc::signal(libc::SIGCHLD, libc::SIG_IGN) };
    assert_eq!(
        after,
        libc::SIG_IGN,
        "with_sigchld_default must restore SIG_IGN after closure returns",
    );
}

/// Regression (non-zero exit propagation): the helper
/// must surface the child's real non-zero exit code, not the
/// previous-implementation `Err(_) => 1` mapping that swallowed
/// every status under SIG_IGN.
#[test]
fn with_sigchld_default_captures_nonzero_exit_status() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_IGN);

    let wrapped = with_sigchld_default(|| Command::new("/bin/false").status());
    let status = wrapped.expect("with_sigchld_default must capture status");
    // /bin/false on every supported Unix exits with code 1.
    assert_eq!(
        status.code(),
        Some(1),
        "/bin/false must surface non-zero code under helper; got {status:?}",
    );
}

/// Regression: under `SIGCHLD = SIG_IGN`, a child that
/// exits before the poll window closes MUST be observed as
/// `Died`. The previous implementation called `Child::try_wait`
/// which internally calls `waitpid(pid, ..., WNOHANG)`; under
/// SIG_IGN that returns `ECHILD` and the old code mapped it to
/// `WaitError`, which the caller in `start_scheduler` then
/// treated as alive — leaving a crashed scheduler undetected.
/// The fix uses `proc_pid_alive` and pidfd POLLIN, both of
/// which are signal-disposition independent.
#[test]
fn poll_startup_detects_death_under_sigchld_ignore() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_IGN);

    let mut child = std::process::Command::new("/bin/true")
        .spawn()
        .expect("spawn /bin/true");
    let status = poll_startup(
        &mut child,
        std::time::Duration::from_millis(10),
        std::time::Duration::from_secs(1),
    );
    assert!(
        matches!(status, StartupStatus::Died),
        "under SIG_IGN, an exited child must be observed as Died (was {status:?})",
    );
}

/// Regression (Alive arm under SIG_IGN): a child that
/// is still running when the timeout elapses must be observed
/// as `Alive` even when SIGCHLD is `SIG_IGN`. This guards the
/// post-timeout `proc_pid_alive` re-check that replaced the
/// old `try_wait` call (which would have returned ECHILD-as-
/// `WaitError` and the caller would have reported alive
/// anyway, but the new path must not regress that branch).
#[test]
fn poll_startup_reports_alive_under_sigchld_ignore() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_IGN);

    let mut child = std::process::Command::new("/bin/sleep")
        .arg("5")
        .spawn()
        .expect("spawn /bin/sleep");
    let status = poll_startup(
        &mut child,
        std::time::Duration::from_millis(20),
        std::time::Duration::from_millis(100),
    );
    // Reap the still-running child via SIGKILL + waitpid. We
    // need to drop SIG_IGN before waiting or `child.wait()`
    // would itself return ECHILD; the SigchldGuard's Drop
    // restores at the end of the test, so flip to SIG_DFL for
    // the cleanup. SAFETY: signal disposition is process-wide
    // but this test holds SIGCHLD_TEST_LOCK, so no other
    // signal-aware test runs concurrently.
    let _ = child.kill();
    unsafe {
        libc::signal(libc::SIGCHLD, libc::SIG_DFL);
    }
    let _ = child.wait();
    assert!(
        matches!(status, StartupStatus::Alive),
        "under SIG_IGN, a running child must be observed as Alive (was {status:?})",
    );
}

/// T2 regression: the trace_pipe→COM1 reader's dump-marker scanner
/// fires the started + complete latches, and matches the end-marker
/// even when it is split across a read boundary (the rolling-tail
/// seam — `SCAN_TAIL_KEEP` must exceed the longest marker so the split
/// prefix survives into the next chunk).
#[test]
fn scan_dump_markers_fires_latches_across_chunk_seam() {
    let mut tail: Vec<u8> = Vec::new();
    assert!(!scx_dump_started_latch().is_set());
    assert!(!scx_dump_complete_latch().is_set());

    // First dump line fires the started latch.
    scan_dump_markers(
        b"  init-1 [000] d.h1. 1.0: sched_ext_dump: init[1] triggered exit kind 1024:\n",
        &mut tail,
    );
    assert!(
        scx_dump_started_latch().is_set(),
        "started latch fires on the first `sched_ext_dump:` line"
    );
    assert!(
        !scx_dump_complete_latch().is_set(),
        "complete latch unset before the end-marker"
    );

    // End-marker split across two reads — the rolling tail must match.
    scan_dump_markers(b"  ...event counters... SCX_EV_SUB_BYPASS", &mut tail);
    assert!(
        !scx_dump_complete_latch().is_set(),
        "a partial end-marker must not fire the complete latch"
    );
    scan_dump_markers(b"_DISPATCH: 0\n", &mut tail);
    assert!(
        scx_dump_complete_latch().is_set(),
        "the seam-split end-marker matches via the rolling tail"
    );
}

/// T3 regression: `reap_child_bounded_status` reaps a child that exits within
/// the bound, and gives up (false) on a still-live child once the
/// bound elapses — so a process that can't take its pending SIGKILL
/// promptly (the defensive case `SCHED_REAP_TIMEOUT` caps) cannot stall
/// teardown.
#[test]
fn reap_child_bounded_status_reaps_quick_and_times_out_on_live() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_DFL);

    let mut quick = std::process::Command::new("sleep")
        .arg("0.1")
        .spawn()
        .expect("spawn sleep 0.1");
    assert!(
        reap_child_bounded_status(&mut quick, std::time::Duration::from_secs(10)).is_some(),
        "a child that exits within the bound is reaped"
    );

    let mut live = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .expect("spawn sleep 30");
    assert!(
        reap_child_bounded_status(&mut live, std::time::Duration::from_millis(200)).is_none(),
        "a still-live child is not reaped within the bound"
    );
    live.kill().unwrap();
    live.wait().unwrap();
}
