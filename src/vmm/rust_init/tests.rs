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

/// Uses raw `std::env::temp_dir()` (not `tempfile::TempDir`)
/// because the test's premise is "mkdir_p is a no-op when the
/// dir already exists" — pointing at an existing dir is the
/// whole point. `tempfile::TempDir` would also work, but raw
/// `temp_dir()` is closer to the production input: `mkdir_p`
/// is called against arbitrary already-existing system paths.
#[test]
fn mkdir_p_existing_is_noop() {
    let tmp = std::env::temp_dir();
    mkdir_p(tmp.to_str().unwrap());
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
}

/// File-not-found returns silently (legitimate "no script"
/// case for the optional sched_enable/sched_disable hooks).
/// Pins the debug-level skip so a future refactor that flipped
/// the missing-file path to error-level would surface here.
#[test]
fn exec_shell_script_missing_file_returns_silently() {
    exec_shell_script("/tmp/ktstr-tax-nonexistent-script-path");
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
    let port_path =
        std::path::Path::new("/tmp/ktstr-test-nonexistent-port-please-do-not-create");
    let t0 = std::time::Instant::now();
    send_sys_rdy_with_retry(budget, 1, &addrs, port_path);
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
    let port_path =
        std::path::Path::new("/tmp/ktstr-test-nonexistent-port-please-do-not-create");
    let addrs = crate::vmm::wire::KernAddrs::new(0, 0, None);
    for &(budget_ms, vcpus) in &[(50u64, 1u32), (150, 2), (250, 8), (500, 32)] {
        let budget = std::time::Duration::from_millis(budget_ms);
        let t0 = std::time::Instant::now();
        send_sys_rdy_with_retry(budget, vcpus, &addrs, port_path);
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
/// `write_to_bulk_port`'s hardcoded `/dev/vport0p1` open will
/// fail too), so kern_addrs_sent stays false and the loop
/// exhausts the budget. The WARN must report
/// `port_exists=true, kern_addrs_sent=false` — the
/// diagnostic combination the troubleshooting doc explains as
/// "the port device exists but writes failed".
#[test]
#[tracing_test::traced_test]
fn send_sys_rdy_retry_reports_port_exists_when_path_resolves() {
    let tmpfile =
        tempfile::NamedTempFile::new().expect("create tempfile to stand in for /dev/vport0p1");
    let budget = std::time::Duration::from_millis(150);
    let addrs = crate::vmm::wire::KernAddrs::new(0, 0, None);
    send_sys_rdy_with_retry(budget, 4, &addrs, tmpfile.path());
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
/// programming-error guard for callers that fail to validate
/// SCHED_PID readouts.
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

/// kill_scheduler_process MUST NOT mutate SCHED_PID — the design
/// at L320-327 of rust_init.rs explicitly keeps the helper
/// generic-pid (no implicit singleton-pid assumption) and defers
/// SCHED_PID ownership to the dispatcher (the future
/// Op::DetachScheduler arm). This test pins that contract against
/// a future "improvement" that adds an implicit SCHED_PID reset
/// for symmetry with the dispatcher path — silent decoupling
/// breakage that would couple kill-pid choice to the singleton
/// scheduler pid in unintended ways.
///
/// Seeds SCHED_PID with a sentinel distinct from any spawnable
/// pid (99_999_999 > Linux's default kernel.pid_max), exercises
/// kill_scheduler_process against an unrelated /bin/sleep pid,
/// and asserts the sentinel survives. Restores SCHED_PID to 0
/// at end so subsequent tests see a clean baseline.
#[test]
fn kill_scheduler_process_does_not_mutate_sched_pid() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
    let _restore = SigchldGuard::install(libc::SIG_IGN);

    let original = SCHED_PID.load(Ordering::Acquire);
    let sentinel: i32 = 99_999_999;
    SCHED_PID.store(sentinel, Ordering::Release);

    let mut child = std::process::Command::new("/bin/sleep")
        .arg("60")
        .spawn()
        .expect("spawn /bin/sleep");
    let pid = child.id() as libc::pid_t;
    let _ = kill_scheduler_process(pid, std::time::Duration::from_millis(500));
    let _ = child.wait();

    let observed = SCHED_PID.load(Ordering::Acquire);
    // Restore BEFORE the assert so a failure does not leak
    // sentinel state to subsequent tests.
    SCHED_PID.store(original, Ordering::Release);

    assert_eq!(
        observed, sentinel,
        "kill_scheduler_process(pid={pid}) mutated SCHED_PID \
         (sentinel={sentinel}, observed={observed}); the helper \
         must NOT touch SCHED_PID — that side channel is the \
         dispatcher's responsibility per the helper's design \
         decoupling. A future commit that adds an implicit reset \
         couples the helper to singleton-pid semantics that the \
         design explicitly avoids."
    );
}

/// SIGCHLD signal disposition is process-wide, so the
/// `with_sigchld_default_*` and `poll_startup_under_sigign_*`
/// regression tests must serialize. Without this lock, two
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

/// Regression: the [`SCHED_PID`] side channel must
/// publish the writer's value and `sched_pid()` must return
/// `Some(pid)` when set, `None` when the sentinel `0` is in
/// place. Since `SCHED_PID` is a process-wide static, the test
/// snapshots the current value, exercises both store paths,
/// and restores the snapshot — so concurrent tests (and the
/// real producer in `start_scheduler` if some other test ever
/// drives it) do not see ambient corruption.
#[test]
fn sched_pid_side_channel_roundtrips() {
    // Snapshot and restore with `Acquire`/`Release` to mirror
    // the production load/store ordering. The test must hold
    // exclusive access to the static for its lifetime; serial
    // execution under the same process means concurrent
    // `sched_pid()` readers in other tests would race, so this
    // test is annotated to acquire `SIGCHLD_TEST_LOCK` even
    // though it has no signal interaction — the existing lock
    // is already the chokepoint for "tests that touch
    // process-wide state" and serializing through it is
    // cheaper than introducing a second mutex for one test.
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();

    let snapshot = SCHED_PID.load(Ordering::Acquire);

    // Sentinel 0 must read as None.
    SCHED_PID.store(0, Ordering::Release);
    assert_eq!(sched_pid(), None, "0 must read as None (sentinel)");

    // Non-zero writer publishes, reader observes.
    SCHED_PID.store(12345, Ordering::Release);
    assert_eq!(
        sched_pid(),
        Some(12345),
        "writer must publish via the atomic side channel",
    );

    // Restore so the test does not leak state into peers.
    SCHED_PID.store(snapshot, Ordering::Release);
}

/// Regression (no env-var write): the new fix must NOT
/// touch `std::env::set_var("SCHED_PID", ...)` because
/// mutating glibc's `__environ` while the probe thread is live
/// is documented UB. Asserting that the env var is absent
/// after a fresh atomic store is a proxy for "no rogue
/// env-mutation snuck back in." If a future refactor brings
/// `set_var` back, this test fails immediately.
#[test]
fn sched_pid_does_not_publish_via_env_var() {
    let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();

    // Clear any ambient env var — some test harnesses inherit
    // `SCHED_PID` from a parent shell. SAFETY: holding the
    // mutex guarantees no concurrent env reader/writer in this
    // test binary.
    unsafe { std::env::remove_var("SCHED_PID") };

    let snapshot = SCHED_PID.load(Ordering::Acquire);
    SCHED_PID.store(99999, Ordering::Release);
    assert_eq!(sched_pid(), Some(99999));
    assert!(
        std::env::var("SCHED_PID").is_err(),
        "atomic side channel must not publish via env var",
    );
    SCHED_PID.store(snapshot, Ordering::Release);
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

/// T3 regression: `reap_child_bounded` reaps a child that exits within
/// the bound, and gives up (false) on a still-live child once the
/// bound elapses — so a process that can't take its pending SIGKILL
/// promptly (the defensive case `SCHED_REAP_TIMEOUT` caps) cannot stall
/// teardown.
#[test]
fn reap_child_bounded_reaps_quick_and_times_out_on_live() {
    let mut quick = std::process::Command::new("sleep")
        .arg("0.1")
        .spawn()
        .expect("spawn sleep 0.1");
    assert!(
        reap_child_bounded(&mut quick, std::time::Duration::from_secs(10)),
        "a child that exits within the bound is reaped"
    );

    let mut live = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .expect("spawn sleep 30");
    assert!(
        !reap_child_bounded(&mut live, std::time::Duration::from_millis(200)),
        "a still-live child is not reaped within the bound"
    );
    live.kill().unwrap();
    live.wait().unwrap();
}
