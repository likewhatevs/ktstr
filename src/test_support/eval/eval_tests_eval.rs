//! Part of the eval module's unit-test suite, split across sibling
//! `eval_tests*.rs` files to keep each under the size ceiling. Child of
//! `eval`: reaches the production core via `super::` / `super::super::`.
use super::super::output::{
    STAGE_INIT_NOT_STARTED, STAGE_INIT_STARTED_NO_PAYLOAD, STAGE_PAYLOAD_STARTED_NO_RESULT,
};
use super::super::test_helpers::{
    EVAL_TOPO, EnvVarGuard, build_assert_result, eevdf_entry, isolated_cache_dir,
    isolated_sidecar_dir, lifecycle_drain, lock_env, make_vm_result, make_vm_result_with_assert,
    no_repro, sched_entry,
};
use super::*;
use crate::assert::{AssertDetail, DetailKind};
use crate::verifier::SCHED_OUTPUT_END;
use tempfile::TempDir;

// -- evaluate_vm_result error path tests --

#[test]
fn eval_eevdf_no_com2_output() {
    let _lock = lock_env();
    let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
    let entry = eevdf_entry("__eval_eevdf_no_out__");
    let result = make_vm_result("", "boot log line\nKernel panic", 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(ERR_NO_TEST_FUNCTION_OUTPUT),
        "EEVDF with no COM2 output should say {ERR_NO_TEST_FUNCTION_OUTPUT:?}, got: {msg}",
    );
    assert!(
        !msg.contains("no test result received from guest"),
        "EEVDF error should not use the scheduler-path wording, got: {msg}",
    );
    assert!(
        msg.contains("exit_code=1"),
        "should include exit code, got: {msg}"
    );
    assert!(
        msg.contains("Kernel panic"),
        "should include console output, got: {msg}"
    );
}

#[test]
fn eval_sched_exits_no_com2_output() {
    let entry = sched_entry("__eval_sched_exits__");
    let result = make_vm_result("", "boot ok", 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(ERR_NO_TEST_RESULT_FROM_GUEST),
        "scheduler present with no output should take the scheduler-path fallback, got: {msg}",
    );
    assert!(
        !msg.contains("test function produced no output"),
        "should not say 'test function produced no output' when scheduler is set, got: {msg}",
    );
}

#[test]
fn eval_sched_exits_with_sched_log() {
    let _lock = lock_env();
    let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
    let sched_log = format!(
        "noise\n{SCHED_OUTPUT_START}\ndo_enqueue_task+0x1a0\nbalance_one+0x50\n{SCHED_OUTPUT_END}\nmore",
    );
    let entry = sched_entry("__eval_sched_log__");
    let result = make_vm_result(&sched_log, "", -1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(ERR_NO_TEST_RESULT_FROM_GUEST),
        "should take the scheduler-path fallback, got: {msg}",
    );
    assert!(
        msg.contains("--- scheduler log ---"),
        "should include scheduler log section, got: {msg}",
    );
    assert!(
        msg.contains("do_enqueue_task"),
        "should include scheduler log content, got: {msg}",
    );
}

#[test]
fn eval_sched_mid_test_exit_triggers_repro() {
    // Scheduler exits mid-test: sched_exit_monitor dumps log to COM2
    // but does NOT write "SCHEDULER_DIED". Auto-repro should still
    // trigger because has_active_scheduling() is true and no
    // AssertResult was produced.
    let sched_log = format!("{SCHED_OUTPUT_START}\nError: BPF program error\n{SCHED_OUTPUT_END}",);
    let entry = sched_entry("__eval_mid_exit_repro__");
    let result = make_vm_result(&sched_log, "", 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let repro_called = std::sync::atomic::AtomicBool::new(false);
    let repro_fn = |_output: &str| -> Option<String> {
        repro_called.store(true, std::sync::atomic::Ordering::Relaxed);
        Some("repro data".to_string())
    };
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &repro_fn,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        repro_called.load(std::sync::atomic::Ordering::Relaxed),
        "repro_fn should be called for mid-test scheduler exit without SCHEDULER_DIED marker",
    );
    assert!(
        msg.contains("--- auto-repro ---"),
        "error should include auto-repro section, got: {msg}",
    );
    assert!(
        msg.contains("repro data"),
        "error should include repro output, got: {msg}",
    );
}

#[test]
fn eval_sched_repro_no_data_shows_diagnostic() {
    // When repro_fn returns the fallback diagnostic, the error
    // output should include it so the user knows auto-repro was
    // tried and why it produced nothing.
    let entry = sched_entry("__eval_repro_no_data__");
    let result = make_vm_result("", "", 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let repro_fn = |_output: &str| -> Option<String> {
        Some(
            "auto-repro: no probe data — scheduler may have exited before \
                 probes could attach. Check the sched_ext dump and scheduler \
                 log sections above for crash details."
                .to_string(),
        )
    };
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &repro_fn,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("--- auto-repro ---"),
        "should include auto-repro section, got: {msg}",
    );
    assert!(
        msg.contains("no probe data"),
        "should include diagnostic message, got: {msg}",
    );
    assert!(
        msg.contains("sched_ext dump"),
        "should direct user to dump section, got: {msg}",
    );
}

#[test]
fn eval_timeout_no_result() {
    let _lock = lock_env();
    let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
    let entry = eevdf_entry("__eval_timeout__");
    let result = make_vm_result("", "booting...\nstill booting...", 0, true);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(ERR_TIMED_OUT_NO_RESULT),
        "should contain full timed-out reason {ERR_TIMED_OUT_NO_RESULT:?}, got: {msg}",
    );
    assert!(
        msg.contains("booting"),
        "should include console output, got: {msg}",
    );
    assert!(
        msg.contains("[topo="),
        "error should include topology, got: {msg}",
    );
}

#[test]
fn eval_payload_exits_no_check_result() {
    // Payload wrote something to COM2 but not a valid AssertResult.
    let entry = eevdf_entry("__eval_no_check__");
    let result = make_vm_result(
        "some output but no delimiters",
        "Linux version 6.14.0\nboot complete",
        0,
        false,
    );
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(ERR_NO_TEST_FUNCTION_OUTPUT),
        "non-parseable COM2 with EEVDF should say {ERR_NO_TEST_FUNCTION_OUTPUT:?}, got: {msg}",
    );
    assert!(
        !msg.contains("no test result received from guest"),
        "EEVDF should not use the scheduler-path wording, got: {msg}",
    );
}

#[test]
fn eval_sched_ext_dump_included() {
    let dump_line = "ktstr-0 [001] 0.5: sched_ext_dump: Debug dump line";
    let entry = sched_entry("__eval_dump__");
    let result = make_vm_result("", dump_line, -1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("--- sched_ext dump ---"),
        "should include dump section, got: {msg}",
    );
    assert!(
        msg.contains("sched_ext_dump: Debug dump"),
        "should include dump content, got: {msg}",
    );
}

#[test]
fn eval_check_result_passed_returns_ok() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(true, vec![]);
    let entry = eevdf_entry("__eval_pass__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    assert!(
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .is_ok(),
        "passing AssertResult should return Ok",
    );
}

#[test]
fn eval_check_result_skip_returns_ok() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    // Regression: an in-VM scenario skip (AssertResult::skip — e.g. the
    // booted topology is below the scenario's CPU/LLC floor) must
    // project to Ok so the exit-code path maps it to EXIT_PASS, NOT be
    // rendered through the failure path as a test failure. Before the
    // is_skip guard in evaluate_vm_result a skip-only result fell into
    // `!is_pass()` (a skip is not is_pass) and returned Err (exit FAIL).
    let assert = crate::assert::AssertResult::skip("topology below scenario floor");
    let entry = eevdf_entry("__eval_skip__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let check_result = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .expect("in-VM skip-only AssertResult must return Ok (EXIT_PASS), not render as a failure");
    assert!(
        check_result.is_skip(),
        "the returned AssertResult must remain skip-only, not be flipped",
    );
}

#[test]
fn eval_check_result_failed_includes_details() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(
        false,
        vec![
            AssertDetail::new(DetailKind::Stuck, "stuck 3000ms"),
            AssertDetail::new(DetailKind::Unfair, "spread 45%"),
        ],
    );
    let entry = eevdf_entry("__eval_fail_details__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .unwrap_err()
    );
    assert!(msg.contains("failed:"), "got: {msg}");
    assert!(msg.contains("stuck 3000ms"), "got: {msg}");
    assert!(msg.contains("spread 45%"), "got: {msg}");
}

/// Cleanup-budget enforcement: when the entry's `cleanup_budget`
/// is set and the run's measured `cleanup_duration` exceeds it,
/// `evaluate_vm_result` folds a failing `AssertDetail` (kind
/// `Other`) carrying the "vm cleanup overran budget" message into
/// the test verdict. The guest body returned a passing
/// `AssertResult` (so the parse-success arm is taken — the only
/// arm where this check fires, see the contract paragraph at
/// `evaluate_vm_result`'s budget block); the budget overshoot
/// flips the merged verdict to a failure, which propagates as a
/// `bail!` error string downstream.
#[test]
fn eval_cleanup_budget_overshoot_folds_failing_detail() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(true, vec![]);
    let mut entry = eevdf_entry("__eval_cleanup_overshoot__");
    entry.cleanup_budget = Some(std::time::Duration::from_secs(1));
    let mut result = make_vm_result_with_assert("", "", 0, false, &assert);
    result.cleanup_duration = Some(std::time::Duration::from_secs(10));
    // Attest a QUIET host (whole-run D = 1.05 ≤ the 1.1 enforcement
    // bar): the re-scoped gate enforces the budget only when the
    // witness proves the box was idle enough for wall attribution.
    result.host_vcpu_schedstat = Some(crate::vmm::HostVcpuSchedstat {
        total_on_cpu_ns: 100,
        total_run_delay_ns: 5,
        sampled_vcpus: 1,
    });
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .unwrap_err()
    );
    assert!(
        msg.contains("vm cleanup overran budget"),
        "budget-overshoot detail must surface in the error string, got: {msg}",
    );
    assert!(
        msg.contains("measured 10.000s"),
        "measured duration must be rendered, got: {msg}",
    );
    assert!(
        msg.contains("budget 1.000s"),
        "budget must be rendered, got: {msg}",
    );
}

/// Cleanup-budget contention demotion: the SAME overshoot as
/// `eval_cleanup_budget_overshoot_folds_failing_detail`, but with a
/// whole-run witness proving the host was NOT quiet (D = 3.0 > the
/// 1.1 enforcement bar) — the overrun demotes to a non-blocking
/// stderr warning and the guest's passing verdict survives. The
/// join/drain teardown dilates with host load by design; only a
/// quiet-host overrun is a teardown regression.
#[test]
fn eval_cleanup_budget_overshoot_demotes_under_witnessed_contention() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(true, vec![]);
    let mut entry = eevdf_entry("__eval_cleanup_overshoot_contended__");
    entry.cleanup_budget = Some(std::time::Duration::from_secs(1));
    let mut result = make_vm_result_with_assert("", "", 0, false, &assert);
    result.cleanup_duration = Some(std::time::Duration::from_secs(10));
    // NON-quiet whole-run witness (D = 3.0 > the 1.1 enforcement bar):
    // the re-scoped gate demotes — join/drain wall on a loaded box is
    // the joined threads' exit-path starvation, unattributable from
    // the host side (the joiner-window instrument was field-falsified:
    // it read D_cleanup = 1.00 on starved runs because the joiner
    // SLEEPS while the joined threads starve).
    result.host_vcpu_schedstat = Some(crate::vmm::HostVcpuSchedstat {
        total_on_cpu_ns: 10,
        total_run_delay_ns: 20,
        sampled_vcpus: 1,
    });
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let res = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    );
    assert!(
        res.is_ok(),
        "a witnessed-contended overrun must not fail the verdict, got: {:?}",
        res.err(),
    );
}

/// Cleanup-budget no-fire: when the run's `cleanup_duration` is
/// strictly under the entry's `cleanup_budget`, the guest's
/// passing `AssertResult` survives the merge and
/// `evaluate_vm_result` returns `Ok`. Verifies that
/// `measured < budget` passes without folding a fail; the exact
/// `measured == budget` boundary is covered separately by
/// [`eval_cleanup_budget_equal_passes`].
#[test]
fn eval_cleanup_budget_under_passes() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(true, vec![]);
    let mut entry = eevdf_entry("__eval_cleanup_under__");
    entry.cleanup_budget = Some(std::time::Duration::from_secs(5));
    let mut result = make_vm_result_with_assert("", "", 0, false, &assert);
    result.cleanup_duration = Some(std::time::Duration::from_millis(500));
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    assert!(
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .is_ok(),
        "cleanup_duration under budget must keep the verdict Ok",
    );
}

/// Cleanup-budget boundary pin: `measured == budget` must NOT
/// fold a fail because the enforcement at
/// `evaluate_vm_result`'s budget block uses strict `>`. A future
/// regression that flips the comparator to `>=` (or to `<` on the
/// pass-side) flips the verdict here, surfacing the bug. Together
/// with [`eval_cleanup_budget_overshoot_folds_failing_detail`] and
/// [`eval_cleanup_budget_under_passes`] this test pins the full
/// {<, ==, >} comparator triplet.
#[test]
fn eval_cleanup_budget_equal_passes() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(true, vec![]);
    let mut entry = eevdf_entry("__eval_cleanup_equal__");
    entry.cleanup_budget = Some(std::time::Duration::from_secs(5));
    let mut result = make_vm_result_with_assert("", "", 0, false, &assert);
    result.cleanup_duration = Some(std::time::Duration::from_secs(5));
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    assert!(
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .is_ok(),
        "cleanup_duration EQUAL to budget must keep the verdict Ok \
             (strict `>` comparator); a `>=` regression lands here",
    );
}

#[test]
fn eval_assert_failure_includes_sched_log() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(
        false,
        vec![AssertDetail::new(
            DetailKind::Stuck,
            "worker 0 stuck 5000ms",
        )],
    );
    // Sched log section still travels via COM2 in this fixture
    // — it's the host's `parse_sched_output` that the assert
    // failure renderer reads, and the bulk-port migration of
    // SCHED_OUTPUT happens in a sibling task. The assert verdict
    // is the part that moved to postcard-over-bulk-port.
    let output = format!("{SCHED_OUTPUT_START}\nscheduler noise line\n{SCHED_OUTPUT_END}",);
    let entry = sched_entry("__eval_fail_sched_log__");
    let result = make_vm_result_with_assert(&output, "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .unwrap_err()
    );
    assert!(msg.contains("worker 0 stuck 5000ms"), "got: {msg}");
    assert!(msg.contains("scheduler noise"), "got: {msg}");
    assert!(msg.contains("--- scheduler log ---"), "got: {msg}");
}

#[test]
fn eval_assert_failure_has_fingerprint() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(
        false,
        vec![AssertDetail::new(DetailKind::Stuck, "stuck 3000ms")],
    );
    let error_line = "Error: apply_cell_config BPF program returned error -2";
    let output = format!("{SCHED_OUTPUT_START}\nstarting\n{error_line}\n{SCHED_OUTPUT_END}",);
    let entry = sched_entry("__eval_fingerprint__");
    let result = make_vm_result_with_assert(&output, "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .unwrap_err()
    );
    assert!(msg.contains(error_line), "got: {msg}");
    let fp_pos = msg.find(error_line).unwrap();
    let name_pos = msg.find("ktstr_test").unwrap();
    assert!(fp_pos < name_pos, "got: {msg}");
}

#[test]
fn eval_timeout_has_fingerprint() {
    let error_line = "Error: scheduler panicked";
    let output = format!("{SCHED_OUTPUT_START}\n{error_line}\n{SCHED_OUTPUT_END}",);
    let entry = sched_entry("__eval_timeout_fp__");
    let result = make_vm_result(&output, "", 0, true);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(error_line),
        "timeout should contain fingerprint, got: {msg}",
    );
    let fp_pos = msg.find(error_line).unwrap();
    let name_pos = msg.find("ktstr_test").unwrap();
    assert!(
        fp_pos < name_pos,
        "fingerprint should appear before ktstr_test line, got: {msg}",
    );
}

#[test]
fn eval_no_result_has_fingerprint() {
    let error_line = "Error: fatal scheduler crash";
    let output = format!("{SCHED_OUTPUT_START}\nstartup log\n{error_line}\n{SCHED_OUTPUT_END}",);
    let entry = sched_entry("__eval_no_result_fp__");
    let result = make_vm_result(&output, "", 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(error_line),
        "no-result failure should contain fingerprint, got: {msg}",
    );
    let fp_pos = msg.find(error_line).unwrap();
    let name_pos = msg.find("ktstr_test").unwrap();
    assert!(
        fp_pos < name_pos,
        "fingerprint should appear before ktstr_test line, got: {msg}",
    );
}

#[test]
fn eval_no_sched_output_no_fingerprint() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(false, vec![AssertDetail::new(DetailKind::Stuck, "stuck")]);
    let entry = eevdf_entry("__eval_no_fp__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .unwrap_err()
    );
    assert!(msg.starts_with("ktstr_test"), "got: {msg}");
}

#[test]
fn eval_monitor_fail_has_fingerprint() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let pass_assert = build_assert_result(true, vec![]);
    let error_line = "Error: imbalance detected internally";
    let output = format!("{SCHED_OUTPUT_START}\nstarting\n{error_line}\n{SCHED_OUTPUT_END}",);
    let entry = sched_entry("__eval_monitor_fp__");
    let imbalance_samples: Vec<crate::monitor::MonitorSample> = (0..30)
        .map(|i| {
            crate::monitor::MonitorSample::new(
                (i * 100) as u64,
                vec![
                    crate::monitor::CpuSnapshot {
                        nr_running: 10,
                        scx_nr_running: 10,
                        local_dsq_depth: 0,
                        rq_clock: 1000 + (i as u64 * 100),
                        scx_flags: 0,
                        event_counters: None,
                        schedstat: None,
                        vcpu_cpu_time_ns: None,
                        vcpu_perf: None,
                        avg_irq_util: None,
                        sched_domains: None,
                    },
                    crate::monitor::CpuSnapshot {
                        nr_running: 1,
                        scx_nr_running: 1,
                        local_dsq_depth: 0,
                        rq_clock: 2000 + (i as u64 * 100),
                        scx_flags: 0,
                        event_counters: None,
                        schedstat: None,
                        vcpu_cpu_time_ns: None,
                        vcpu_perf: None,
                        avg_irq_util: None,
                        sched_domains: None,
                    },
                ],
            )
        })
        .collect();
    let summary =
        crate::monitor::MonitorSummary::from_samples_with_threshold(&imbalance_samples, 0);
    let result = crate::vmm::VmResult {
        success: true,
        vcpus: 1,
        cpu_budget: 1,
        resolve_source: None,
        expect_auto_repro_satisfied: false,
        exit_code: 0,
        duration: std::time::Duration::from_secs(1),
        timed_out: false,
        watchdog_kill_reason: None,
        final_guest_phase: crate::vmm::GuestLifecyclePhase::Boot,
        final_progress_epoch: 0,
        bpf_map_writes_delivered: None,
        periodic_prereqs_ready: None,
        periodic_window_end: None,
        output,
        stderr: String::new(),
        monitor: Some(crate::monitor::MonitorReport {
            samples: imbalance_samples,
            summary,
            preemption_threshold_ns: 0,
            watchdog_observation: None,
            page_offset: 0,
            boot_wait_outcome: crate::monitor::BootWaitOutcome::NotConfigured,
            scx_event_counters_supported: false,
        }),
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &pass_assert,
            )],
        }),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        crash_message: None,
        cleanup_duration: None,
        cleanup_sched_delta: None,
        virtio_blk_counters: None,
        virtio_net_counters: None,
        snapshot_bridge: {
            let cb: crate::scenario::snapshot::CaptureCallback = std::sync::Arc::new(|_| None);
            crate::scenario::snapshot::SnapshotBridge::new(cb)
        },
        stats_client: None,
        periodic_fired: 0,
        periodic_real: 0,
        periodic_target: 0,
        kern_kaslr_offset: 0,
        entry_name: None,
        variant_hash: 0,
        host_vcpu_schedstat: None,
        contention_witness: None,
        periodic_series_cache: std::sync::OnceLock::new(),
    };
    let assertions = crate::assert::Assert::NO_OVERRIDES
        .max_imbalance_ratio(4.0)
        .fail_on_rq_clock_stuck(true)
        .with_monitor_defaults();
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .unwrap_err()
    );
    assert!(
        msg.contains(ERR_MONITOR_FAILED_AFTER_SCENARIO),
        "got: {msg}"
    );
    assert!(msg.contains(error_line), "got: {msg}");
    let fp_pos = msg.find(error_line).unwrap();
    let name_pos = msg.find("ktstr_test").unwrap();
    assert!(fp_pos < name_pos, "got: {msg}");
}

#[test]
fn eval_timeout_with_sched_includes_diagnostics() {
    let _lock = lock_env();
    let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
    let entry = sched_entry("__eval_timeout_sched__");
    let result = make_vm_result("", "Linux version 6.14.0\nkernel panic here", -1, true);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(ERR_TIMED_OUT_NO_RESULT),
        "should contain {ERR_TIMED_OUT_NO_RESULT:?}, got: {msg}"
    );
    assert!(
        msg.contains("[sched=test_sched_bin]"),
        "should include scheduler label, got: {msg}"
    );
    assert!(
        msg.contains("--- diagnostics ---"),
        "should include diagnostics, got: {msg}"
    );
    assert!(
        msg.contains("kernel panic here"),
        "should include console tail, got: {msg}"
    );
}

// -- sentinel integration in evaluate_vm_result --

#[test]
fn eval_no_sentinels_shows_initramfs_failure() {
    let _lock = lock_env();
    let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
    let entry = eevdf_entry("__eval_no_sentinel__");
    let result = make_vm_result("", "Kernel panic", 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(STAGE_INIT_NOT_STARTED),
        "no sentinels should indicate kernel/mount failure, got: {msg}",
    );
}

#[test]
fn eval_init_started_but_no_payload() {
    let _lock = lock_env();
    let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
    let entry = eevdf_entry("__eval_init_only__");
    // `classify_init_stage` walks `MSG_TYPE_LIFECYCLE` entries
    // from the bulk drain (the COM2 sentinel-string path is
    // gone), so the test must publish the lifecycle phase
    // through `guest_messages` rather than seed it via stdout.
    // The `output` argument still flows to the sched-log /
    // panic scrapers downstream of this classification.
    let mut result = make_vm_result("KTSTR_INIT_STARTED\n", "boot log", 1, false);
    result.guest_messages = Some(lifecycle_drain(&[
        crate::vmm::wire::LifecyclePhase::InitStarted,
    ]));
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(STAGE_INIT_STARTED_NO_PAYLOAD),
        "init lifecycle phase only should indicate cgroup/scheduler setup failure, got: {msg}",
    );
}

#[test]
fn eval_payload_started_no_result() {
    let _lock = lock_env();
    let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
    let entry = eevdf_entry("__eval_payload_start__");
    // Same migration as `eval_init_started_but_no_payload`:
    // `classify_init_stage` reads `MSG_TYPE_LIFECYCLE` entries
    // from `guest_messages`, not the COM2 sentinel strings the
    // legacy fixture seeded via stdout. Publish both
    // `InitStarted` and `PayloadStarting` so the classifier
    // resolves to the deepest reached stage.
    let output = "KTSTR_INIT_STARTED\nKTSTR_PAYLOAD_STARTING\ngarbage";
    let mut result = make_vm_result(output, "", 1, false);
    result.guest_messages = Some(lifecycle_drain(&[
        crate::vmm::wire::LifecyclePhase::InitStarted,
        crate::vmm::wire::LifecyclePhase::PayloadStarting,
    ]));
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(STAGE_PAYLOAD_STARTED_NO_RESULT),
        "both lifecycle phases should indicate payload ran but failed, got: {msg}",
    );
}

// -- guest panic detection tests --

#[test]
fn eval_crash_in_output_says_guest_crashed() {
    let entry = sched_entry("__eval_crash_detect__");
    let output = "KTSTR_INIT_STARTED\nPANIC: panicked at src/foo.rs:42: assertion failed";
    let result = make_vm_result(output, "", 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains(ERR_GUEST_CRASHED_PREFIX), "got: {msg}");
    assert!(msg.contains("assertion failed"), "got: {msg}");
}

#[test]
fn eval_crash_eevdf_says_guest_crashed() {
    let entry = eevdf_entry("__eval_crash_eevdf__");
    let output = "PANIC: panicked at src/bar.rs:10: index out of bounds";
    let result = make_vm_result(output, "", 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains(ERR_GUEST_CRASHED_PREFIX), "got: {msg}");
    assert!(msg.contains("index out of bounds"), "got: {msg}");
}

#[test]
fn eval_crash_message_from_field() {
    // `result.crash_message` (the structured-field path)
    // carries the multiline `PANIC: ... \n   0: <frame>\n`
    // backtrace populated by `freeze_coord::collect_results`
    // from COM2's `extract_panic_message`. The eval path uses
    // the structured field when set, falling back to a fresh
    // `extract_panic_message(output)` call only when the field
    // is `None`. The structured-field path renders the multiline
    // form (`guest crashed:\n{crash}`) so the full backtrace is
    // visible in the test failure.
    let entry = sched_entry("__eval_crash_field__");
    let crash = "PANIC: panicked at src/test.rs:42: assertion failed\n   \
                          0: ktstr::vmm::rust_init::ktstr_guest_init\n";
    // COM2 also has a PANIC: line (serial). The structured
    // field must take priority and render the multiline form.
    let output = "PANIC: panicked at src/test.rs:42: assertion failed";
    let mut result = make_vm_result(output, "", 1, false);
    result.crash_message = Some(crash.to_string());
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(ERR_GUEST_CRASHED_PREFIX),
        "should say {ERR_GUEST_CRASHED_PREFIX:?}, got: {msg}",
    );
    assert!(
        msg.contains("ktstr_guest_init"),
        "backtrace content should be present, got: {msg}",
    );
    // Structured-field path uses "guest crashed:\n{crash}"
    // (multiline); the bare-output fallback uses "guest
    // crashed: {msg}" (single line). The backtrace frame proves
    // the structured field was used, not the fallback.
    assert!(
        msg.contains("0: ktstr::vmm::rust_init::ktstr_guest_init"),
        "full backtrace from structured field should appear, got: {msg}",
    );
}

// -- diagnostic section tests --

#[test]
fn eval_sched_exit_includes_console() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(
        false,
        vec![AssertDetail::new(
            DetailKind::SchedulerCrashed,
            "scheduler process died unexpectedly after completing step 1 of 2 (0.5s into test)",
        )],
    );
    let entry = sched_entry("__eval_sched_exit_console__");
    let result =
        make_vm_result_with_assert("", "kernel panic\nsched_ext: disabled", 1, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .unwrap_err()
    );
    assert!(msg.contains("--- diagnostics ---"), "got: {msg}");
    assert!(msg.contains("kernel panic"), "got: {msg}");
}

#[test]
fn eval_sched_exit_includes_monitor() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(
        false,
        vec![AssertDetail::new(
            DetailKind::SchedulerCrashed,
            "scheduler process died unexpectedly during workload (2.0s into test)",
        )],
    );
    let entry = sched_entry("__eval_sched_exit_monitor__");
    let result = crate::vmm::VmResult {
        success: false,
        vcpus: 1,
        cpu_budget: 1,
        resolve_source: None,
        expect_auto_repro_satisfied: false,
        exit_code: 1,
        duration: std::time::Duration::from_secs(1),
        timed_out: false,
        watchdog_kill_reason: None,
        final_guest_phase: crate::vmm::GuestLifecyclePhase::Boot,
        final_progress_epoch: 0,
        bpf_map_writes_delivered: None,
        periodic_prereqs_ready: None,
        periodic_window_end: None,
        output: String::new(),
        stderr: String::new(),
        monitor: Some(crate::monitor::MonitorReport {
            samples: vec![],
            summary: crate::monitor::MonitorSummary {
                total_samples: 5,
                max_imbalance_ratio: 3.0,
                max_local_dsq_depth: 2,
                stuck_count: 0,
                event_deltas: None,
                schedstat_deltas: None,
                prog_stats_deltas: None,
                ..Default::default()
            },
            preemption_threshold_ns: 0,
            watchdog_observation: None,
            page_offset: 0,
            boot_wait_outcome: crate::monitor::BootWaitOutcome::NotConfigured,
            scx_event_counters_supported: false,
        }),
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &assert,
            )],
        }),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        crash_message: None,
        cleanup_duration: None,
        cleanup_sched_delta: None,
        virtio_blk_counters: None,
        virtio_net_counters: None,
        snapshot_bridge: {
            let cb: crate::scenario::snapshot::CaptureCallback = std::sync::Arc::new(|_| None);
            crate::scenario::snapshot::SnapshotBridge::new(cb)
        },
        stats_client: None,
        periodic_fired: 0,
        periodic_real: 0,
        periodic_target: 0,
        kern_kaslr_offset: 0,
        entry_name: None,
        variant_hash: 0,
        host_vcpu_schedstat: None,
        contention_witness: None,
        periodic_series_cache: std::sync::OnceLock::new(),
    };
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .unwrap_err()
    );
    assert!(msg.contains("--- monitor ---"), "got: {msg}");
    assert!(msg.contains("max_imbalance"), "got: {msg}");
}

#[test]
fn eval_monitor_fail_includes_sched_log() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let pass_assert = build_assert_result(true, vec![]);
    let output = format!("{SCHED_OUTPUT_START}\nscheduler debug output here\n{SCHED_OUTPUT_END}",);
    let entry = sched_entry("__eval_monitor_fail_sched__");
    // Imbalance ratio 10.0 exceeds default threshold of 4.0,
    // sustained for 5+ samples past the 20-sample warmup window.
    let imbalance_samples: Vec<crate::monitor::MonitorSample> = (0..30)
        .map(|i| {
            crate::monitor::MonitorSample::new(
                (i * 100) as u64,
                vec![
                    crate::monitor::CpuSnapshot {
                        nr_running: 10,
                        scx_nr_running: 10,
                        local_dsq_depth: 0,
                        rq_clock: 1000 + (i as u64 * 100),
                        scx_flags: 0,
                        event_counters: None,
                        schedstat: None,
                        vcpu_cpu_time_ns: None,
                        vcpu_perf: None,
                        avg_irq_util: None,
                        sched_domains: None,
                    },
                    crate::monitor::CpuSnapshot {
                        nr_running: 1,
                        scx_nr_running: 1,
                        local_dsq_depth: 0,
                        rq_clock: 2000 + (i as u64 * 100),
                        scx_flags: 0,
                        event_counters: None,
                        schedstat: None,
                        vcpu_cpu_time_ns: None,
                        vcpu_perf: None,
                        avg_irq_util: None,
                        sched_domains: None,
                    },
                ],
            )
        })
        .collect();
    let summary =
        crate::monitor::MonitorSummary::from_samples_with_threshold(&imbalance_samples, 0);
    let result = crate::vmm::VmResult {
        success: true,
        vcpus: 1,
        cpu_budget: 1,
        resolve_source: None,
        expect_auto_repro_satisfied: false,
        exit_code: 0,
        duration: std::time::Duration::from_secs(1),
        timed_out: false,
        watchdog_kill_reason: None,
        final_guest_phase: crate::vmm::GuestLifecyclePhase::Boot,
        final_progress_epoch: 0,
        bpf_map_writes_delivered: None,
        periodic_prereqs_ready: None,
        periodic_window_end: None,
        output,
        stderr: String::new(),
        monitor: Some(crate::monitor::MonitorReport {
            samples: imbalance_samples,
            summary,
            preemption_threshold_ns: 0,
            watchdog_observation: None,
            page_offset: 0,
            boot_wait_outcome: crate::monitor::BootWaitOutcome::NotConfigured,
            scx_event_counters_supported: false,
        }),
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &pass_assert,
            )],
        }),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        crash_message: None,
        cleanup_duration: None,
        cleanup_sched_delta: None,
        virtio_blk_counters: None,
        virtio_net_counters: None,
        snapshot_bridge: {
            let cb: crate::scenario::snapshot::CaptureCallback = std::sync::Arc::new(|_| None);
            crate::scenario::snapshot::SnapshotBridge::new(cb)
        },
        stats_client: None,
        periodic_fired: 0,
        periodic_real: 0,
        periodic_target: 0,
        kern_kaslr_offset: 0,
        entry_name: None,
        variant_hash: 0,
        host_vcpu_schedstat: None,
        contention_witness: None,
        periodic_series_cache: std::sync::OnceLock::new(),
    };
    let assertions = crate::assert::Assert::NO_OVERRIDES
        .max_imbalance_ratio(4.0)
        .fail_on_rq_clock_stuck(true)
        .with_monitor_defaults();
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &no_repro,
            None,
        )
        .unwrap_err()
    );
    assert!(
        msg.contains(ERR_MONITOR_FAILED_AFTER_SCENARIO),
        "got: {msg}"
    );
    assert!(msg.contains("--- scheduler log ---"), "got: {msg}");
}

/// MANDATORY guard for the drain-once de-confliction:
///
/// 1. **No starvation.** A `post_vm`-style read of the series
///    (`phase_buckets()`, which routes through `captures_series()`)
///    runs BEFORE `evaluate_vm_result` — exactly the production
///    ordering. The framework's later `stats.phases` build must still
///    be non-empty: before the `captures_series()` cache, the post_vm
///    read drained the bridge and `stats.phases` came up silently
///    empty.
/// 2. **Single source.** `stats.phases` and `VmResult::phase_buckets()`
///    must carry IDENTICAL content. Both fold the same
///    `captures_series()` cache through
///    `build_phase_buckets_with_stimulus`; the only input that differs
///    is the stimulus arg, and the production caller passes
///    `result.stimulus_timeline()` — the same source `phase_buckets()`
///    uses internally. This test pins that equality so a future drift
///    of `evaluate`'s stimulus arg from `stimulus_timeline()` fails
///    loudly.
#[test]
fn phase_buckets_equals_stats_phases_and_post_vm_read_does_not_starve() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let pass_assert = build_assert_result(true, vec![]);
    let entry = sched_entry("__eval_phase_buckets_eq__");
    let result = crate::vmm::VmResult {
        success: true,
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &pass_assert,
            )],
        }),
        periodic_fired: 3,
        periodic_target: 3,
        ..crate::vmm::VmResult::test_fixture()
    };
    // Populate the snapshot bridge with periodic captures stamped into
    // Step[0] (step_index = 1). No Stimulus frames are attached, so the
    // bucketer falls back to each capture's stamped step_index on both
    // the phase_buckets() and the evaluate path.
    for i in 0..3 {
        result.snapshot_bridge.store_with_stats_and_step(
            &format!("periodic_{i}"),
            crate::monitor::dump::FailureDumpReport::default(),
            None,
            Some(i as u64 * 100),
            None,
            1,
        );
    }
    // post_vm runs BEFORE evaluate: read the series first (the
    // pre-bug double-drain trigger).
    let post_vm_buckets = result.phase_buckets();
    assert!(
        !post_vm_buckets.is_empty(),
        "fixture with 3 captures must yield buckets"
    );
    // Framework builds stats.phases AFTER post_vm. Pass the SAME
    // stimulus source the production caller passes (the
    // `stimulus_events = result.stimulus_timeline()` binding in
    // `run_ktstr_test_inner_impl`).
    let stimulus = result.stimulus_timeline();
    let ar = evaluate_vm_result(
        &entry,
        &result,
        &crate::assert::Assert::NO_OVERRIDES,
        &stimulus,
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .expect("pass_assert on the success arm must return Ok");
    assert!(
        !ar.stats.phases.is_empty(),
        "a post_vm series read must NOT starve stats.phases — the \
         latent drain-once bug this task fixes",
    );
    assert_eq!(
        ar.stats.phases, post_vm_buckets,
        "stats.phases must equal VmResult::phase_buckets() (single \
         source: shared captures_series() cache + same builder + \
         stimulus_timeline)",
    );
}

/// Sibling of the no-carrier equality pin above, WITH guest per-cgroup
/// carriers: `VmResult::phase_buckets()` folds the guest carriers exactly as
/// production `evaluate_vm_result` does, so the two stay equal AND both expose
/// `per_cgroup`. Pins that the fold is applied on both paths — without it,
/// `phase_buckets()` returned host buckets with EMPTY per_cgroup (a strict
/// subset of `stats.phases`), so this `assert_eq!` would have failed.
#[test]
fn phase_buckets_equals_stats_phases_with_guest_per_cgroup_carriers() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    // Passing guest AssertResult carrying a per-cgroup carrier at
    // step_index=1; the captures below stamp the same step_index, so the
    // carrier takes the matched arm and unions its per_cgroup into that bucket.
    let mut guest_assert = build_assert_result(true, vec![]);
    let mut pc = std::collections::BTreeMap::new();
    pc.insert(
        "cellA".to_string(),
        crate::assert::PhaseCgroupStats {
            total_migrations: 7,
            total_iterations: 11,
            ..Default::default()
        },
    );
    guest_assert.stats.phases = vec![crate::assert::PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        // Merge-neutral window required by fold_guest_per_cgroup_into_host_buckets.
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: std::collections::BTreeMap::new(),
        per_cgroup: pc,
    }];
    let entry = sched_entry("__eval_phase_buckets_eq_carriers__");
    let result = crate::vmm::VmResult {
        success: true,
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &guest_assert,
            )],
        }),
        periodic_fired: 3,
        periodic_target: 3,
        ..crate::vmm::VmResult::test_fixture()
    };
    for i in 0..3 {
        result.snapshot_bridge.store_with_stats_and_step(
            &format!("periodic_{i}"),
            crate::monitor::dump::FailureDumpReport::default(),
            None,
            Some(i as u64 * 100),
            None,
            1,
        );
    }
    // post_vm reads first; the fold must EXPOSE per_cgroup on this view.
    let post_vm_buckets = result.phase_buckets();
    let step0 = post_vm_buckets
        .iter()
        .find(|b| b.step_index == 1)
        .expect("a Step[0] bucket from the stamped captures");
    assert_eq!(
        step0.per_cgroup.get("cellA").map(|c| c.total_migrations),
        Some(7),
        "phase_buckets() must fold the guest per_cgroup carrier",
    );
    let stimulus = result.stimulus_timeline();
    let ar = evaluate_vm_result(
        &entry,
        &result,
        &crate::assert::Assert::NO_OVERRIDES,
        &stimulus,
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .expect("pass_assert on the success arm must return Ok");
    assert_eq!(
        ar.stats.phases, post_vm_buckets,
        "stats.phases must equal VmResult::phase_buckets() WITH per_cgroup \
         carriers folded — identical two-source fold on both paths",
    );
}

/// A guest `AssertResult` carrying ONE measured cgroup (counters + wake/run-delay
/// reductions), TLV-encoded into a [`crate::vmm::VmResult`] with three stamped
/// periodic captures. The shared fixture for the `run_metric` parity / boundary
/// tests below: its `stats.cgroups` drives the pooled `iteration_rate`
/// (family 4) and the `WorstLowest` / Distribution re-pools (family 5) — the
/// families `VmResult::run_metric` must reconstruct from the guest cgroups.
#[cfg(test)]
fn run_metric_fixture(cg: crate::assert::CgroupStats) -> crate::vmm::VmResult {
    let mut guest_assert = build_assert_result(true, vec![]);
    guest_assert.stats.cgroups = vec![cg];
    let result = crate::vmm::VmResult {
        success: true,
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &guest_assert,
            )],
        }),
        periodic_fired: 3,
        periodic_target: 3,
        ..crate::vmm::VmResult::test_fixture()
    };
    for i in 0..3 {
        result.snapshot_bridge.store_with_stats_and_step(
            &format!("periodic_{i}"),
            crate::monitor::dump::FailureDumpReport::default(),
            None,
            Some(i as u64 * 100),
            None,
            1,
        );
    }
    result
}

/// LOAD-BEARING PARITY: `VmResult::run_metric` self-computes the SAME
/// run-level `ext_metrics` `evaluate_vm_result` writes — for every key the eval
/// path produces, `run_metric` resolves the identical value. Exercises the
/// families that need the guest per-cgroup roll-up (pooled
/// `iteration_rate` + the `WorstLowest` / Distribution re-pools), the
/// reconstruction that makes `run_metric` possible: `check_result.stats.cgroups`
/// equals `guest_assert_result().stats.cgroups` (the host adds no cgroups —
/// `evaluate_verdict_folds` merges only empty-cgroup `fail()`s, and
/// `populate_run_stats_and_folded_timeline` writes `phases`/`ext_metrics`, never
/// `cgroups`), so `run_metric` replays the eval sequence over the guest cgroups +
/// pre-derive phase fold to the byte-identical map.
#[test]
fn run_metric_equals_evaluate_run_level_ext() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let result = run_metric_fixture(crate::assert::CgroupStats {
        cgroup_name: "cellA".to_string(),
        num_workers: 4,
        total_iterations: 1000,
        total_cpu_time_ns: 2_000_000_000,
        p99_wake_latency_us: 80.0,
        median_wake_latency_us: 20.0,
        wake_latency_cv: 0.5,
        mean_run_delay_us: 10.0,
        worst_run_delay_us: 40.0,
        ..Default::default()
    });
    let entry = sched_entry("__eval_run_metric_parity__");
    let stimulus = result.stimulus_timeline();
    let ar = evaluate_vm_result(
        &entry,
        &result,
        &crate::assert::Assert::NO_OVERRIDES,
        &stimulus,
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .expect("pass guest verdict on the success arm returns Ok");
    assert!(
        !ar.stats.ext_metrics.is_empty(),
        "the measured-cgroup fixture must produce run-level ext keys",
    );
    // Family-4 anchor: the pooled rate is a real ratio in the eval map.
    assert_eq!(
        ar.stats.ext_metrics.get("iteration_rate").copied(),
        Some(500.0),
        "1000 iters / 2.0 cpu-sec = 500 — the pooled rate must be in the eval map",
    );
    // FULL-MAP forward parity: every key evaluate produced resolves via
    // run_metric to the identical value (self-computed pre-merge == stored
    // post-merge).
    for (k, v) in &ar.stats.ext_metrics {
        assert_eq!(
            result.run_metric(k.as_str()),
            Some(*v),
            "run_metric({k}) must equal the eval-layer ext value",
        );
    }
    // The typed Into<MetricId> path resolves the same value as the &str path.
    assert_eq!(
        result.run_metric(crate::stats::BuiltinMetric::IterationRate),
        Some(500.0),
    );
    // A registered ext metric this run did not produce -> None (loud-absent,
    // not a false 0.0).
    assert_eq!(result.run_metric("total_hardirqs"), None);
}

/// NON-DESTRUCTIVE: `run_metric` composes with `phase_metric` /
/// `phase_buckets` in one `post_vm` — the memoized snapshot-bridge drain means a
/// `run_metric` read does not starve the others (the latent double-drain class),
/// and `run_metric` is idempotent across interleaved bridge-draining accessors.
#[test]
fn run_metric_is_non_destructive_alongside_phase_reads() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let result = run_metric_fixture(crate::assert::CgroupStats {
        cgroup_name: "cellA".to_string(),
        num_workers: 2,
        total_iterations: 600,
        total_cpu_time_ns: 3_000_000_000,
        ..Default::default()
    });
    let first = result.run_metric("iteration_rate");
    // Interleave the other bridge-draining accessors.
    let _pm = result.phase_metric(crate::assert::Phase::step(0), "system_time_ns");
    let _buckets = result.phase_buckets();
    let second = result.run_metric("iteration_rate");
    assert_eq!(first, Some(200.0), "600 iters / 3.0 cpu-sec = 200");
    assert_eq!(
        first, second,
        "run_metric must be idempotent across interleaved drains (memoized, no starve)",
    );
}

/// LOUD-ABSENT: `None` means absent (an unregistered key, or a registered
/// key this run did not produce); `Some(0.0)` means a real measured zero — never
/// conflated. A cgroup with zero iterations over positive on-CPU time makes
/// `iteration_rate` a measured `Some(0.0)`.
#[test]
fn run_metric_loud_absent_distinct_from_measured_zero() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let result = run_metric_fixture(crate::assert::CgroupStats {
        cgroup_name: "idle".to_string(),
        num_workers: 1,
        total_iterations: 0,
        total_cpu_time_ns: 1_000_000_000,
        ..Default::default()
    });
    // measured zero (0 iters over 1.0 cpu-sec) -> Some(0.0)
    assert_eq!(result.run_metric("iteration_rate"), Some(0.0));
    // unregistered dynamic key -> None
    assert_eq!(result.run_metric("scx_totally_made_up_key"), None);
    // registered ext metric this run did not produce -> None
    assert_eq!(result.run_metric("total_hardirqs"), None);
}

/// `run_metric` boundary: the typed cross-cgroup fields now RESOLVE None-aware
/// (re-derived from the carriers, distinguishing measured from never-measured),
/// the ext-sourced pooled rate resolves, and the monitor-sourced run-level
/// metrics remain per-phase-only (`None` here) — matching
/// `ScenarioStats::run_metric`'s boundary.
#[test]
fn run_metric_resolves_typed_excludes_monitor_metrics() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let result = run_metric_fixture(crate::assert::CgroupStats {
        cgroup_name: "cellA".to_string(),
        num_workers: 4,
        total_iterations: 1000,
        total_cpu_time_ns: 2_000_000_000,
        total_migrations: 50,
        ..Default::default()
    });
    // typed cross-cgroup fields now resolve None-aware: total_migrations is a
    // cross-cgroup SUM (one cgroup, 50) — a measured value, not absent.
    assert_eq!(result.run_metric("total_migrations"), Some(50.0));
    // worst_spread is None-aware ABSENT: this fixture's cgroup set no spread
    // (CgroupStats::spread == None — no worker had measurable wall time), so the
    // re-derive reports not-measured, distinct from a measured 0.0.
    assert_eq!(result.run_metric("worst_spread"), None);
    // monitor-sourced run-level — per-phase only, not in this accessor.
    assert_eq!(result.run_metric("max_imbalance_ratio"), None);
    assert_eq!(result.run_metric("stuck_count"), None);
    // the ext-sourced pooled rate IS resolved (1000 iters / 2 s).
    assert_eq!(result.run_metric("iteration_rate"), Some(500.0));
}

/// HOST-ONLY / EMPTY: a `VmResult` with no guest verdict and no captures
/// (`guest_assert_result` `Err`, empty series) yields `None` for every metric,
/// no panic — the degraded path AssertResult::run_metric also documents.
#[test]
fn run_metric_host_only_run_yields_none() {
    let result = crate::vmm::VmResult::test_fixture();
    assert_eq!(result.run_metric("iteration_rate"), None);
    assert_eq!(result.run_metric("total_hardirqs"), None);
    assert_eq!(result.run_metric("worst_p99_wake_latency_us"), None);
    assert_eq!(result.run_metric("scx_made_up"), None);
}

/// ABSENT RATE: a pooled rate whose denominator is absent (zero on-CPU time
/// across cgroups) is NOT produced — `run_metric` returns `None`, never an `inf`
/// or `NaN` (the summed-zero early return + both-or-neither component insert).
#[test]
fn run_metric_absent_rate_is_none_not_inf() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let result = run_metric_fixture(crate::assert::CgroupStats {
        cgroup_name: "nocpu".to_string(),
        num_workers: 1,
        total_iterations: 500,
        total_cpu_time_ns: 0,
        ..Default::default()
    });
    let v = result.run_metric("iteration_rate");
    assert_eq!(v, None, "zero-denominator rate must be absent, not inf/NaN");
}

/// Production-hook pin: `VmResult::phase_buckets()` — the method a per-phase A/B
/// claim reads via `phase_metric` — must DERIVE the schbench per-phase scalars
/// into `PhaseBucket.metrics`. A guest carrier with a `SchbenchPhaseStats` at
/// step_index=1 must surface e.g. `wakeup_p99_latency_us` in that bucket's
/// metrics. Guards the WIRING (a refactor that dropped the
/// `derive_phase_metrics` call at the `phase_buckets()` site would keep
/// every direct-derive unit test green while production silently stopped
/// emitting the metrics — the mirror-tests-pin-nothing class). Also exercises
/// the schbench carrier's postcard roundtrip guest→host (the TLV entry below).
#[test]
fn phase_buckets_derives_schbench_perphase_metrics() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    // A schbench carrier: 100-sample wakeup histogram (99×10µs + 1×10000µs, so
    // p99 = 10µs), msg run-delay 50000ns over 5 schedules, loop_count 42.
    let mut wakeup = crate::workload::schbench::plat::PlatStats::default();
    for _ in 0..99 {
        wakeup.add_lat(10);
    }
    wakeup.add_lat(10_000);
    let schbench = crate::workload::schbench::run::SchbenchPhaseStats {
        wakeup,
        request: crate::workload::schbench::plat::PlatStats::default(),
        rps: crate::workload::schbench::plat::PlatStats::default(),
        msg_run_delay_ns: 50_000,
        msg_pcount: 5,
        worker_run_delay_ns: 0,
        worker_pcount: 0,
        loop_count: 42,
        worker_cpu_ns: 0,
    };
    let mut guest_assert = build_assert_result(true, vec![]);
    let mut pc = std::collections::BTreeMap::new();
    pc.insert(
        "cg".to_string(),
        crate::assert::PhaseCgroupStats {
            schbench: Some(schbench),
            ..Default::default()
        },
    );
    guest_assert.stats.phases = vec![crate::assert::PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: std::collections::BTreeMap::new(),
        per_cgroup: pc,
    }];
    let result = crate::vmm::VmResult {
        success: true,
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &guest_assert,
            )],
        }),
        periodic_fired: 3,
        periodic_target: 3,
        ..crate::vmm::VmResult::test_fixture()
    };
    for i in 0..3 {
        result.snapshot_bridge.store_with_stats_and_step(
            &format!("periodic_{i}"),
            crate::monitor::dump::FailureDumpReport::default(),
            None,
            Some(i as u64 * 100),
            None,
            1,
        );
    }
    let buckets = result.phase_buckets();
    let step0 = buckets
        .iter()
        .find(|b| b.step_index == 1)
        .expect("a Step[0] bucket from the stamped captures");
    // The production hook derived the schbench per-phase scalars (and the carrier
    // survived the postcard TLV roundtrip).
    assert_eq!(
        step0.metrics.get("wakeup_p99_latency_us"),
        Some(&10.0),
        "phase_buckets() must derive the schbench per-phase metrics (production hook)",
    );
    assert_eq!(step0.metrics.get("schbench_loop_count"), Some(&42.0));
    assert_eq!(step0.metrics.get("sched_delay_msg_us"), Some(&10.0));
    // worker pcount 0 -> ABSENT, request empty -> ABSENT (no false zeros).
    assert!(!step0.metrics.contains_key("sched_delay_worker_us"));
    assert!(!step0.metrics.contains_key("request_p99_latency_us"));
}

/// Wiring pin for VmResult::better_across_phases — the per-phase A/B
/// primitive. Two phases carry schbench wakeup p99 = 10µs (scx, step_index 1)
/// vs 100µs (EEVDF, step_index 2); the comparator must orient "better" from the
/// LowerBetter polarity (registry) so scx (candidate) beats EEVDF (baseline)
/// and the reverse framing fails, and a metric absent in a phase is
/// Inconclusive→Err (no silent pass). Guards that better_across_phases resolves
/// both phases via phase_metric + the polarity via metric_def and records the
/// right verdict — a swapped baseline/candidate or wrong polarity source fails
/// here, not in the pure better_outcome unit tests.
#[test]
fn better_across_phases_orients_by_polarity_end_to_end() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    // A schbench carrier whose wakeup p99 == `us` (100 identical samples) and
    // whose loop_count == `loops` (the HigherBetter throughput key).
    let carrier = |us: u32, loops: u64| {
        let mut wakeup = crate::workload::schbench::plat::PlatStats::default();
        for _ in 0..100 {
            wakeup.add_lat(us);
        }
        crate::assert::PhaseCgroupStats {
            schbench: Some(crate::workload::schbench::run::SchbenchPhaseStats {
                wakeup,
                request: crate::workload::schbench::plat::PlatStats::default(),
                rps: crate::workload::schbench::plat::PlatStats::default(),
                msg_run_delay_ns: 0,
                msg_pcount: 0,
                worker_run_delay_ns: 0,
                worker_pcount: 0,
                loop_count: loops,
                worker_cpu_ns: 1_000_000_000,
            }),
            ..Default::default()
        }
    };
    let phase_bucket = |idx: u16, us: u32, loops: u64| {
        let mut pc = std::collections::BTreeMap::new();
        pc.insert("cg".to_string(), carrier(us, loops));
        crate::assert::PhaseBucket {
            step_index: idx,
            label: format!("Step[{}]", idx - 1),
            start_ms: u64::MAX,
            end_ms: 0,
            sample_count: 0,
            metrics: std::collections::BTreeMap::new(),
            per_cgroup: pc,
        }
    };
    let mut guest_assert = build_assert_result(true, vec![]);
    // scx (step_index 1): p99 10µs, loop_count 200; EEVDF (step_index 2): p99
    // 100µs, loop_count 50. scx wins on BOTH a LowerBetter (latency) and a
    // HigherBetter (throughput) key.
    guest_assert.stats.phases = vec![phase_bucket(1, 10, 200), phase_bucket(2, 100, 50)];
    let result = crate::vmm::VmResult {
        success: true,
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &guest_assert,
            )],
        }),
        periodic_fired: 4,
        periodic_target: 4,
        ..crate::vmm::VmResult::test_fixture()
    };
    // Stamp captures in BOTH steps so phase_buckets() builds step_index 1 and 2.
    for &(i, step) in &[(0u64, 1u16), (1, 1), (2, 2), (3, 2)] {
        result.snapshot_bridge.store_with_stats_and_step(
            &format!("periodic_{i}"),
            crate::monitor::dump::FailureDumpReport::default(),
            None,
            Some(i * 100),
            None,
            step,
        );
    }
    let scx = crate::assert::Phase::step(0); // step_index 1
    let eevdf = crate::assert::Phase::step(1); // step_index 2
    // scx (candidate) p99 10µs is strictly better than EEVDF (baseline) 100µs.
    let mut v = crate::assert::Verdict::new();
    result
        .better_across_phases(&mut v, eevdf, scx, "wakeup_p99_latency_us")
        .better_than();
    assert!(
        v.into_anyhow_or_log().is_ok(),
        "scx p99 10µs is strictly better than EEVDF 100µs (LowerBetter)"
    );
    // Reverse framing: EEVDF (candidate) is NOT better than scx (baseline) -> Err.
    let mut v2 = crate::assert::Verdict::new();
    result
        .better_across_phases(&mut v2, scx, eevdf, "wakeup_p99_latency_us")
        .better_than();
    assert!(
        v2.into_anyhow_or_log().is_err(),
        "EEVDF p99 100µs is not better than scx 10µs"
    );
    // A metric absent in the phases (empty request histogram -> request keys
    // never derived) is Inconclusive -> Err, never a silent pass.
    let mut v3 = crate::assert::Verdict::new();
    result
        .better_across_phases(&mut v3, eevdf, scx, "request_p99_latency_us")
        .better_than();
    assert!(
        v3.into_anyhow_or_log().is_err(),
        "absent metric -> inconclusive -> Err (no silent pass)"
    );
    // HigherBetter through production: CPU-second throughput is directional,
    // while the raw loop count is informational. With one worker CPU-second on
    // each side, the derived values equal the loop counts.
    let mut v4 = crate::assert::Verdict::new();
    result
        .better_across_phases(&mut v4, eevdf, scx, "schbench_loops_per_cpu_sec")
        .better_than();
    assert!(
        v4.into_anyhow_or_log().is_ok(),
        "scx loop_count 200 > EEVDF 50 (HigherBetter)"
    );
    let mut v5 = crate::assert::Verdict::new();
    result
        .better_across_phases(&mut v5, scx, eevdf, "schbench_loops_per_cpu_sec")
        .better_than();
    assert!(
        v5.into_anyhow_or_log().is_err(),
        "EEVDF loop_count 50 is not > scx 200"
    );
    // by_at_least(margin) through production: scx p99 10µs is a 90% improvement
    // over EEVDF 100µs — clears a 50% margin, falls short of a 95% margin.
    let mut v6 = crate::assert::Verdict::new();
    result
        .better_across_phases(&mut v6, eevdf, scx, "wakeup_p99_latency_us")
        .by_at_least(0.5);
    assert!(
        v6.into_anyhow_or_log().is_ok(),
        "90% improvement clears a 50% margin"
    );
    let mut v7 = crate::assert::Verdict::new();
    result
        .better_across_phases(&mut v7, eevdf, scx, "wakeup_p99_latency_us")
        .by_at_least(0.95);
    assert!(
        v7.into_anyhow_or_log().is_err(),
        "90% improvement is short of a 95% margin"
    );
}

/// Per-cgroup query API end-to-end through the production VmResult path:
/// VmResult::phase_cgroup_metric reads the NAMED cgroup (distinct per cgroup, not
/// the pool), the counter fallback exposes total_migrations, the None taxonomy
/// holds, and better_across_phases_cgroup orients per-cgroup (opposite outcomes for
/// two cgroups proves it reads the named one, not the aggregate).
#[test]
fn phase_cgroup_metric_and_better_across_phases_cgroup_end_to_end() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let carrier = |us: u32, loops: u64, migs: u64| {
        let mut wakeup = crate::workload::schbench::plat::PlatStats::default();
        for _ in 0..100 {
            wakeup.add_lat(us);
        }
        crate::assert::PhaseCgroupStats {
            schbench: Some(crate::workload::schbench::run::SchbenchPhaseStats {
                wakeup,
                request: crate::workload::schbench::plat::PlatStats::default(),
                rps: crate::workload::schbench::plat::PlatStats::default(),
                msg_run_delay_ns: 0,
                msg_pcount: 0,
                worker_run_delay_ns: 0,
                worker_pcount: 0,
                loop_count: loops,
                worker_cpu_ns: 1_000_000_000,
            }),
            total_migrations: migs,
            ..Default::default()
        }
    };
    // Two cgroups per phase with DISTINCT values, so a per-cgroup lookup is
    // distinguishable from the pooled aggregate.
    let phase_bucket = |idx: u16, a: (u32, u64, u64), b: (u32, u64, u64)| {
        let mut pc = std::collections::BTreeMap::new();
        pc.insert("cg_a".to_string(), carrier(a.0, a.1, a.2));
        pc.insert("cg_b".to_string(), carrier(b.0, b.1, b.2));
        crate::assert::PhaseBucket {
            step_index: idx,
            label: format!("Step[{}]", idx - 1),
            start_ms: u64::MAX,
            end_ms: 0,
            sample_count: 0,
            metrics: std::collections::BTreeMap::new(),
            per_cgroup: pc,
        }
    };
    let mut guest_assert = build_assert_result(true, vec![]);
    // step 1 (scx): cg_a loop 200 / cg_b loop 80; step 2 (eevdf): cg_a loop 50 / cg_b loop 90.
    guest_assert.stats.phases = vec![
        phase_bucket(1, (10, 200, 7), (50, 80, 3)),
        phase_bucket(2, (100, 50, 4), (40, 90, 9)),
    ];
    let result = crate::vmm::VmResult {
        success: true,
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &guest_assert,
            )],
        }),
        periodic_fired: 4,
        periodic_target: 4,
        ..crate::vmm::VmResult::test_fixture()
    };
    for &(i, step) in &[(0u64, 1u16), (1, 1), (2, 2), (3, 2)] {
        result.snapshot_bridge.store_with_stats_and_step(
            &format!("periodic_{i}"),
            crate::monitor::dump::FailureDumpReport::default(),
            None,
            Some(i * 100),
            None,
            step,
        );
    }
    let scx = crate::assert::Phase::step(0); // step_index 1
    let eevdf = crate::assert::Phase::step(1); // step_index 2
    // phase_cgroup_metric reads the NAMED cgroup, distinct per cgroup (not pooled).
    assert_eq!(
        result.phase_cgroup_metric(scx, "cg_a", "schbench_loops_per_cpu_sec"),
        Some(200.0)
    );
    assert_eq!(
        result.phase_cgroup_metric(scx, "cg_b", "schbench_loops_per_cpu_sec"),
        Some(80.0)
    );
    // Counter fallback (carrier field, not a derived metric): cg_a step-1 migrations 7.
    assert_eq!(
        result.phase_cgroup_metric(scx, "cg_a", "total_migrations"),
        Some(7.0)
    );
    // None taxonomy: missing cgroup, typo metric.
    assert_eq!(
        result.phase_cgroup_metric(scx, "missing", "schbench_loops_per_cpu_sec"),
        None
    );
    assert_eq!(
        result.phase_cgroup_metric(scx, "cg_a", "not_a_metric"),
        None
    );
    // better_across_phases_cgroup orients the named cgroup's CPU-second
    // throughput; the reverse framing must fail.
    let mut v = crate::assert::Verdict::new();
    result
        .better_across_phases_cgroup(&mut v, eevdf, scx, "cg_a", "schbench_loops_per_cpu_sec")
        .better_than();
    assert!(
        v.into_anyhow_or_log().is_ok(),
        "cg_a scx loop 200 > eevdf 50 (HigherBetter)"
    );
    let mut v2 = crate::assert::Verdict::new();
    result
        .better_across_phases_cgroup(&mut v2, scx, eevdf, "cg_a", "schbench_loops_per_cpu_sec")
        .better_than();
    assert!(
        v2.into_anyhow_or_log().is_err(),
        "cg_a eevdf loop 50 is not > scx 200"
    );
    // The per-cgroup orientation for cg_b is OPPOSITE (scx loop 80 < eevdf 90), so
    // scx is NOT better for cg_b — proving the lookup reads cg_b, not the pool.
    let mut v3 = crate::assert::Verdict::new();
    result
        .better_across_phases_cgroup(&mut v3, eevdf, scx, "cg_b", "schbench_loops_per_cpu_sec")
        .better_than();
    assert!(
        v3.into_anyhow_or_log().is_err(),
        "cg_b scx loop 80 is not > eevdf 90 (distinct from cg_a)"
    );
}

/// ScenarioStats per-cgroup lookups (the AssertResult-holding path):
/// phase_cgroup_metric reads a named cgroup's derived metric for a Phase, falls
/// back to the carrier Counters, and returns None for a missing cgroup / typo
/// metric / missing phase.
#[test]
fn scenario_stats_phase_cgroup_metric_and_counter_fallback() {
    let _lock = lock_env();
    let mut ar = build_assert_result(true, vec![]);
    let mut metrics = std::collections::BTreeMap::new();
    metrics.insert("schbench_loop_count".to_string(), 10.0);
    let mut pc = std::collections::BTreeMap::new();
    pc.insert(
        "cg_a".to_string(),
        crate::assert::PhaseCgroupStats {
            metrics,
            total_migrations: 7,
            ..Default::default()
        },
    );
    ar.stats.phases = vec![crate::assert::PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: std::collections::BTreeMap::new(),
        per_cgroup: pc,
    }];
    // phase_cgroup_metric via Phase::step(0) (the scenario's first Step); reads the derived map.
    assert_eq!(
        ar.stats
            .phase_cgroup_metric(crate::assert::Phase::step(0), "cg_a", "schbench_loop_count"),
        Some(10.0)
    );
    // Counter fallback (carrier field, not derived).
    assert_eq!(
        ar.stats
            .phase_cgroup_metric(crate::assert::Phase::step(0), "cg_a", "total_migrations"),
        Some(7.0)
    );
    // None: missing cgroup / typo metric / missing phase.
    assert_eq!(
        ar.stats.phase_cgroup_metric(
            crate::assert::Phase::step(0),
            "missing",
            "schbench_loop_count"
        ),
        None
    );
    assert_eq!(
        ar.stats
            .phase_cgroup_metric(crate::assert::Phase::step(0), "cg_a", "not_a_metric"),
        None
    );
    assert_eq!(
        ar.stats
            .phase_cgroup_metric(crate::assert::Phase::step(8), "cg_a", "schbench_loop_count"),
        None
    );
}

/// Pooled-path counter symmetry (the AssertResult-holding path): ScenarioStats
/// phase_metric resolves the per-cgroup Counters total_migrations /
/// total_iterations as the cross-cgroup SUM via PhaseBucket::cgroup_counter_total,
/// symmetric with the per-cgroup phase_cgroup_metric (which surfaces each cgroup's
/// own value) and with VmResult::phase_metric. These two Counters have read_sample
/// == None, so they live ONLY in the per-cgroup carriers (never in
/// PhaseBucket.metrics); without the fallback the pooled lookup returned a silent
/// None while the per-cgroup sibling surfaced the value.
#[test]
fn scenario_stats_phase_metric_pools_per_cgroup_counters() {
    let _lock = lock_env();
    let mut ar = build_assert_result(true, vec![]);
    let mut pc = std::collections::BTreeMap::new();
    pc.insert(
        "cg_a".to_string(),
        crate::assert::PhaseCgroupStats {
            total_migrations: 7,
            total_iterations: 100,
            ..Default::default()
        },
    );
    pc.insert(
        "cg_b".to_string(),
        crate::assert::PhaseCgroupStats {
            total_migrations: 4,
            total_iterations: 50,
            ..Default::default()
        },
    );
    ar.stats.phases = vec![crate::assert::PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: std::collections::BTreeMap::new(),
        per_cgroup: pc,
    }];
    // Pooled lookup resolves the carrier-only Counters as the cross-cgroup SUM.
    assert_eq!(
        ar.stats
            .phase_metric(crate::assert::Phase::step(0), "total_migrations"),
        Some(11.0)
    ); // 7 + 4
    assert_eq!(
        ar.stats
            .phase_metric(crate::assert::Phase::step(0), "total_iterations"),
        Some(150.0)
    ); // 100 + 50
    // Per-cgroup sibling surfaces each cgroup's own value (the symmetry the fix restores).
    assert_eq!(
        ar.stats
            .phase_cgroup_metric(crate::assert::Phase::step(0), "cg_a", "total_migrations"),
        Some(7.0)
    );
    assert_eq!(
        ar.stats
            .phase_cgroup_metric(crate::assert::Phase::step(0), "cg_b", "total_migrations"),
        Some(4.0)
    );
    // typo metric / missing phase -> None (sentinel-free).
    assert_eq!(
        ar.stats
            .phase_metric(crate::assert::Phase::step(0), "not_a_metric"),
        None
    );
    assert_eq!(
        ar.stats
            .phase_metric(crate::assert::Phase::step(8), "total_migrations"),
        None
    );
}

/// Eval REORDER wiring: on the GUEST-FAIL path the failure message's
/// timeline is built from the POST-fold `check_result.stats.phases`
/// (folded_timeline), so the per-cgroup sub-block AND orphan not-measured
/// markers reach operator-facing output — not the pre-fold `early_timeline`
/// (empty per_cgroup, orphans excluded). The timeline.rs unit tests prove
/// from_phase_buckets renders these; the eval PASS-arm fold test proves the
/// fold populates stats.phases.per_cgroup; this pins that the two are WIRED on
/// the failure path. A revert of the call site to early_timeline would silently
/// drop the per-cgroup detail + orphan markers from failures with the suite
/// otherwise green.
#[test]
fn evaluate_failure_message_renders_per_cgroup_via_folded_timeline() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    use crate::timeline::StimulusEvent;
    // A FAILING guest AssertResult -> evaluate_vm_result takes the failure arm
    // (returns Err with the rendered message).
    let mut guest_assert = build_assert_result(
        false,
        vec![crate::assert::AssertDetail::new(
            crate::assert::DetailKind::NoProgress,
            "deliberate failure for the render test".to_string(),
        )],
    );
    let carrier = |step: u16, name: &str, off_cpu: f64, iters: u64| {
        let mut pc = std::collections::BTreeMap::new();
        pc.insert(
            name.to_string(),
            crate::assert::PhaseCgroupStats {
                num_workers: 1,
                off_cpu_pcts: vec![off_cpu],
                total_iterations: iters,
                ..Default::default()
            },
        );
        crate::assert::PhaseBucket {
            step_index: step,
            label: format!("Step[{}]", step - 1),
            start_ms: u64::MAX,
            end_ms: 0,
            sample_count: 0,
            metrics: std::collections::BTreeMap::new(),
            per_cgroup: pc,
        }
    };
    // step 1 carrier MATCHES a real host bucket (per-cgroup sub-block renders on
    // a measured phase); step 2 carrier has NO host bucket -> orphan arm
    // (renders "window not measured").
    guest_assert.stats.phases = vec![
        carrier(1, "cgHog", 75.0, 900),
        carrier(2, "cgOrphan", 10.0, 5),
    ];
    let entry = sched_entry("__eval_fail_render_per_cgroup__");
    let result = crate::vmm::VmResult {
        success: true,
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &guest_assert,
            )],
        }),
        periodic_fired: 1,
        periodic_target: 1,
        ..crate::vmm::VmResult::test_fixture()
    };
    // Real host capture for step 1 ONLY -> step 1 folds via the matched arm;
    // step 2 has no host bucket -> orphan arm (same setup the fold PASS-arm test
    // uses for step 1).
    result.snapshot_bridge.store_with_stats_and_step(
        "periodic_000",
        crate::monitor::dump::FailureDumpReport::default(),
        None,
        Some(1500),
        None,
        1,
    );
    let start = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    // ONLY StepStart[1] — a StepStart[2] would make build_phase_buckets_with_stimulus
    // SYNTHESIZE a host bucket at step 2, matching carrier2 (not the orphan arm).
    // With no step-2 host bucket, carrier2 stays an orphan (the not-measured case).
    let stimulus = vec![start(1000, 1, 0)];
    let err = evaluate_vm_result(
        &entry,
        &result,
        &crate::assert::Assert::NO_OVERRIDES,
        &stimulus,
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("per-cgroup:"),
        "failure message must carry the per-cgroup sub-block via folded_timeline; got:\n{msg}",
    );
    assert!(
        msg.contains("cgHog: off-cpu avg=75.0%"),
        "the matched-arm carrier's reduced line must render; got:\n{msg}",
    );
    assert!(
        msg.contains("window not measured"),
        "the orphan carrier must render its not-measured window; got:\n{msg}",
    );
    assert!(
        msg.contains("cgOrphan:"),
        "the orphan carrier's per-cgroup line must render; got:\n{msg}",
    );
}

/// Through the production eval path: with stimulus StepStarts
/// spanning steps 1..3 but periodic captures landing only in step 1,
/// evaluate_vm_result's stats.phases must contain a synthesized bucket
/// (sample_count==0) for the uncaptured steps, without fabricating an
/// iteration rate from stimulus wall time. This is the --cell-parent-cgroup
/// short-interior-step scenario, pinned through
/// evaluate_vm_result (not just build_phase_buckets_with_stimulus): the
/// non-empty synthesized buckets also flip timeline selection onto the
/// from_phase_buckets path.
#[test]
fn evaluate_synthesizes_phase_buckets_for_uncaptured_steps() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    use crate::timeline::StimulusEvent;
    let pass_assert = build_assert_result(true, vec![]);
    let entry = sched_entry("__eval_synthesize_uncaptured__");
    let result = crate::vmm::VmResult {
        success: true,
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &pass_assert,
            )],
        }),
        periodic_fired: 1,
        periodic_target: 1,
        ..crate::vmm::VmResult::test_fixture()
    };
    // One capture stamped into step 1 (boundary_offset None -> stamped
    // step_index fallback in by_stimulus_phase). Steps 2 and 3 capture
    // nothing.
    result.snapshot_bridge.store_with_stats_and_step(
        "periodic_000",
        crate::monitor::dump::FailureDumpReport::default(),
        None,
        Some(100),
        None,
        1,
    );
    // Hand-built stimulus (the evaluate_vm_result param): a StepStart per
    // step with cumulative iterations. Steps 2/3 have starts but no
    // captures, so build_phase_buckets_with_stimulus must synthesize them.
    let start = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![
        start(1000, 1, 0),
        start(2000, 2, 1000),
        start(3000, 3, 2000),
    ];
    let ar = evaluate_vm_result(
        &entry,
        &result,
        &crate::assert::Assert::NO_OVERRIDES,
        &stimulus,
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .expect("pass_assert on the success arm must return Ok");
    // Step 2 captured nothing but must appear in stats.phases, synthesized.
    let step2 = ar
        .stats
        .phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("uncaptured step 2 must appear as a synthesized bucket in stats.phases");
    assert_eq!(step2.sample_count, 0, "synthesized bucket is capture-free");
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        None,
        "a synthesized wall-only step must not fabricate iteration_rate",
    );
}

/// End-to-end through the host eval path: a guest AssertResult
/// carrying a per-phase per_cgroup carrier (step_index 1) survives the TLV
/// serialize/parse roundtrip and is FOLDED into the host-rebuilt bucket of the
/// same step_index — not clobbered. Asserts the host window/metrics survive
/// (proving fold, not overwrite) AND the guest per_cgroup is carried through to
/// check_result.stats.phases (the durable sidecar telemetry).
#[test]
fn evaluate_folds_guest_per_cgroup_into_host_phase_buckets() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    use crate::timeline::StimulusEvent;
    let mut guest_assert = build_assert_result(true, vec![]);
    let mut per_cgroup = std::collections::BTreeMap::new();
    per_cgroup.insert(
        "cgTest".to_string(),
        crate::assert::PhaseCgroupStats {
            num_workers: 2,
            total_iterations: 99,
            total_cpu_time_ns: 4242,
            ..Default::default()
        },
    );
    // The guest carrier: step_index 1, merge-neutral window, empty metrics,
    // per_cgroup payload — exactly what step_per_cgroup_bucket emits in the guest.
    guest_assert.stats.phases = vec![crate::assert::PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: std::collections::BTreeMap::new(),
        per_cgroup,
    }];
    let entry = sched_entry("__eval_fold_per_cgroup__");
    let result = crate::vmm::VmResult {
        success: true,
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &guest_assert,
            )],
        }),
        periodic_fired: 1,
        periodic_target: 1,
        ..crate::vmm::VmResult::test_fixture()
    };
    // One real host capture for step 1: elapsed_ms=1500, boundary_offset_ms=None,
    // step_index=1. by_stimulus_phase keys it by the STAMPED step_index (1)
    // (boundary_offset_ms is None, so there is no offset remap), and the bucket
    // window comes from elapsed_ms — start_ms==end_ms==1500. So
    // build_phase_buckets_with_stimulus produces a REAL host bucket at step_index
    // 1 (sample_count 1, a non-sentinel window), which makes the fold take the
    // MATCHED arm (host bucket + guest carrier at the same step_index merged via
    // merge_matched_phase_buckets) — the path under test. Without a real host
    // bucket at step_index 1 the guest carrier would be an ORPHAN whose
    // assertions pass trivially.
    result.snapshot_bridge.store_with_stats_and_step(
        "periodic_000",
        crate::monitor::dump::FailureDumpReport::default(),
        None,
        Some(1500),
        None,
        1,
    );
    // StepStart[1] -> StepStart[2] supplies step 1's iteration_rate (0 -> 1000
    // iters over 1s), matched to the bucket by step_index; it does NOT set the
    // bucket window (that comes from the capture's elapsed_ms above).
    let start = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![start(1000, 1, 0), start(2000, 2, 1000)];
    let ar = evaluate_vm_result(
        &entry,
        &result,
        &crate::assert::Assert::NO_OVERRIDES,
        &stimulus,
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .expect("pass_assert on the success arm must return Ok");
    let b = ar
        .stats
        .phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("host bucket at step_index 1 must survive the fold");
    // MATCHED arm, not orphan: the host bucket's capture (sample_count 1) and
    // its window/metrics survived the merge with the guest carrier. An orphan
    // (the bug this test guards against) would carry the guest carrier verbatim:
    // sample_count 0, a normalized (0,0) window, and no metrics.
    assert_eq!(
        b.sample_count, 1,
        "host capture merged — matched arm, not an orphan"
    );
    assert_ne!(
        b.start_ms,
        u64::MAX,
        "host window survived (min vs the carrier's MAX sentinel)",
    );
    assert_ne!(
        b.start_ms, 0,
        "real host window start, not the orphan's normalized 0"
    );
    assert!(
        b.metrics.contains_key("iteration_rate"),
        "host metric (iteration_rate) survived the matched merge — a clobber by \
         the carrier's empty metrics would drop it",
    );
    // The guest per_cgroup folded into the matched host bucket through the TLV
    // roundtrip + parse + fold.
    let pc = b
        .per_cgroup
        .get("cgTest")
        .expect("guest per_cgroup must fold into the host bucket, not be clobbered");
    assert_eq!(pc.total_iterations, 99);
    assert_eq!(pc.num_workers, 2);
    assert_eq!(pc.total_cpu_time_ns, 4242);
}

/// `acquire_test_kernel_lock_if_cached` returns `Some(guard)`
/// when `kernel_path` is shaped like a real cache entry:
/// `{cache_root}/{cache_key}/{image_name}`. Exercises the
/// canonicalize + candidate-root-equality branch.
///
/// Uses [`isolated_cache_dir`] so the tempdir is both pointed
/// at by `KTSTR_CACHE_DIR` AND cleaned up on drop. Holds
/// [`lock_env`] throughout so parallel tests don't race the
/// env var.
#[test]
fn acquire_test_kernel_lock_if_cached_returns_guard_on_cache_entry() {
    let _env_lock = lock_env();
    let cache = isolated_cache_dir();
    // Fake cache entry: {cache_root}/my-kernel-key/bzImage.
    let entry_dir = cache.path().join("my-kernel-key");
    std::fs::create_dir_all(&entry_dir).expect("create entry dir");
    let image_path = entry_dir.join("bzImage");
    std::fs::write(&image_path, b"fake kernel image").expect("plant image");

    let guard = super::acquire_test_kernel_lock_if_cached(&image_path)
        .expect("lock acquire must not error on valid cache entry");
    assert!(
        guard.is_some(),
        "cache-entry path must produce a SharedLockGuard",
    );
    // Confirm the .locks/ subdir materialized as a side effect
    // of the acquire — pins the integration with
    // `CacheDir::acquire_shared_lock`'s ensure_lock_dir path.
    assert!(
        cache.path().join(".locks").is_dir(),
        ".locks/ must materialize under the cache root",
    );
}

/// `acquire_test_kernel_lock_if_cached` returns `Ok(None)`
/// when `kernel_path` is NOT under the resolved cache root —
/// e.g. a `/lib/modules/…/vmlinuz` bootloader image or an
/// operator-supplied raw path. The function silently skips
/// locking rather than erroring, matching the doc contract:
/// "Such paths do not need coordination because the build
/// pipeline never touches them."
#[test]
fn acquire_test_kernel_lock_if_cached_returns_none_outside_cache() {
    let _env_lock = lock_env();
    let cache = isolated_cache_dir();
    // Path under a DIFFERENT tempdir, not the cache root.
    let outside = TempDir::new().expect("tempdir outside cache");
    let entry_dir = outside.path().join("raw-kernel-key");
    std::fs::create_dir_all(&entry_dir).expect("create entry dir");
    let image_path = entry_dir.join("bzImage");
    std::fs::write(&image_path, b"fake kernel image").expect("plant image");

    let guard = super::acquire_test_kernel_lock_if_cached(&image_path)
        .expect("non-cache path must not error");
    assert!(
        guard.is_none(),
        "path outside {} must skip locking, got guard",
        cache.path().display(),
    );
}

/// `acquire_test_kernel_lock_if_cached`'s detection seam matches a
/// flock-timeout-shaped error string iff it contains BOTH the
/// substrings `"timed out after"` and `"flock LOCK_"`. Pin the
/// substring contract so a rewording in
/// `crate::flock`'s bail message that drops either substring is
/// caught here rather than silently degrading flock-timeout
/// classification (a typed, retryable `ResourceContention`) into an
/// unclassified plain anyhow.
///
/// The test feeds the seam a representative shared-lock-timeout
/// rendering (matching the literal format produced at
/// `flock/acquire.rs::acquire_flock_with_timeout` — `"flock LOCK_SH on
/// {context} timed out after {timeout:?}"`) and the
/// exclusive-lock equivalent. A negative-control string lacking
/// the `"flock LOCK_"` marker must NOT match — that protects
/// against a future seam rewrite that overfits the timeout
/// substring and accepts unrelated timeouts.
#[test]
fn flock_timeout_substring_classification_pins_seam() {
    let shared_rendering = "flock LOCK_SH on /tmp/cache/.locks/key.lock \
                                timed out after 30s (lockfile \
                                /tmp/cache/.locks/key.lock, holders: pid=42)";
    assert!(
        super::kernel::is_flock_timeout_message(shared_rendering),
        "shared-lock timeout rendering must classify as flock timeout: {shared_rendering}",
    );

    let exclusive_rendering = "flock LOCK_EX on /tmp/cache/.locks/key.lock \
                                   timed out after 30s (lockfile \
                                   /tmp/cache/.locks/key.lock, holders: pid=99)";
    assert!(
        super::kernel::is_flock_timeout_message(exclusive_rendering),
        "exclusive-lock timeout rendering must classify as flock timeout: \
             {exclusive_rendering}",
    );

    // Negative control: a different timeout (e.g. cgroup write)
    // contains "timed out after" but not "flock LOCK_". The seam
    // must reject it so non-flock timeouts are not laundered as
    // ResourceContention.
    let unrelated_timeout = "cgroup write to /sys/fs/cgroup/foo timed out after 5000ms";
    assert!(
        !super::kernel::is_flock_timeout_message(unrelated_timeout),
        "non-flock timeout must NOT classify as flock timeout: {unrelated_timeout}",
    );

    // Negative control: a flock error that is NOT a timeout
    // (e.g. an EBADF on the descriptor) lacks "timed out after"
    // and must reject so non-timeout flock errors fall through to
    // the hard-error arm rather than being SKIP-classified.
    let flock_non_timeout =
        "flock LOCK_SH on /tmp/cache/.locks/key.lock failed: Bad file descriptor (os error 9)";
    assert!(
        !super::kernel::is_flock_timeout_message(flock_non_timeout),
        "flock non-timeout error must NOT classify as flock timeout: {flock_non_timeout}",
    );
}

// -- timed-out arm: scheduler-exited reason override (the `timeout_reason` block in `render_no_result_message`'s `if result.timed_out` arm, reached via evaluate_vm_result) --

/// Timed-out run whose stderr carries an scx-disable kmsg anchor with
/// a NON-EMPTY parenthesized body: `parse_kmsg_window` parses the
/// anchor and the timeout reason becomes
/// `timed out (scheduler exited: <message>)`, OVERRIDING the default
/// `ERR_TIMED_OUT_NO_RESULT`. Pins the `timeout_reason` block's
/// `if let Some(ev) = scx_exits.last()` non-empty-message
/// (`!ev.message.is_empty()`) sub-arm. The `--- watchdog ---` block still renders.
#[test]
fn eval_timeout_sched_exited_reason_override() {
    let _lock = lock_env();
    let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
    let entry = sched_entry("__eval_timeout_sched_exited__");
    // Single anchor line, body "(runnable task stall)" -> message
    // "runnable task stall" (trimmed inside the parens by
    // parse_kmsg_window). No follow-on lines append to the message.
    let stderr = "[1.0] sched_ext: BPF scheduler \"scx_test\" disabled (runnable task stall)\n";
    let result = make_vm_result("", stderr, -1, true);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("timed out (scheduler exited: runnable task stall)"),
        "non-empty scx-exit message must override the default timeout reason, got: {msg}",
    );
    assert!(
        !msg.contains(ERR_TIMED_OUT_NO_RESULT),
        "the override must replace the default ERR_TIMED_OUT_NO_RESULT body, got: {msg}",
    );
    assert!(
        msg.contains("--- ktstr-watchdog ---"),
        "watchdog diagnostic block must render on the timed-out arm, got: {msg}",
    );
}

/// Timed-out run whose scx-disable anchor has an EMPTY parenthesized
/// body `()`: `parse_kmsg_window` yields an event with an empty
/// `message`, so the timeout reason takes the `timeout_reason` block's
/// `if ev.message.is_empty()` empty-message sub-arm —
/// `timed out (scheduler <name> exited)` formatted
/// from `ev.scheduler_name` (parsed as `scx_test` from the anchor).
#[test]
fn eval_timeout_sched_exited_empty_message() {
    let _lock = lock_env();
    let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
    let entry = sched_entry("__eval_timeout_sched_exited_empty__");
    // Parenthesized body empty -> message_body trims to "" and no
    // follow-on line appends, so ev.message.is_empty() is true.
    let stderr = "[1.0] sched_ext: BPF scheduler \"scx_test\" disabled ()\n";
    let result = make_vm_result("", stderr, -1, true);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("timed out (scheduler scx_test exited)"),
        "empty scx-exit message must render the scheduler-name-only form, got: {msg}",
    );
    assert!(
        !msg.contains(ERR_TIMED_OUT_NO_RESULT),
        "the override must replace the default ERR_TIMED_OUT_NO_RESULT body, got: {msg}",
    );
    assert!(
        msg.contains("--- ktstr-watchdog ---"),
        "watchdog diagnostic block must render, got: {msg}",
    );
}

#[test]
fn eval_timeout_duration_hint_only_for_body_tier3() {
    let entry = sched_entry("__eval_timeout_duration_hint__");
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    for (reason, phase, expect_hint) in [
        (
            crate::vmm::WatchdogKillReason::Tier1CpuBudget,
            crate::vmm::GuestLifecyclePhase::Boot,
            false,
        ),
        (
            crate::vmm::WatchdogKillReason::Tier3Deadman,
            crate::vmm::GuestLifecyclePhase::Boot,
            false,
        ),
        (
            crate::vmm::WatchdogKillReason::Tier1CpuBudget,
            crate::vmm::GuestLifecyclePhase::Body,
            false,
        ),
        (
            crate::vmm::WatchdogKillReason::Tier3Deadman,
            crate::vmm::GuestLifecyclePhase::Body,
            true,
        ),
    ] {
        let mut result = make_vm_result("", "guest stalled...", -1, true);
        result.watchdog_kill_reason = Some(reason);
        result.final_guest_phase = phase;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &EVAL_TOPO,
                &no_repro,
                None,
            )
            .unwrap_err()
        );
        assert_eq!(
            msg.contains("if the test body needs more wall time"),
            expect_hint,
            "duration hint polarity for {reason:?}/{phase:?}: {msg}",
        );
    }
}

// -- timed-out arm: crash_section dual-fire (the `crash_section` binding in evaluate_vm_result's `if result.timed_out` arm) --

/// Timed-out run that ALSO carries a structured `crash_message`: both
/// the timeout reason AND the guest crash backtrace render. The
/// `crash_section` `if let Some(ref guest_crash) = result.crash_message`
/// true-branch fires only when
/// `timed_out && crash_message.is_some()`. Timeout stays the primary
/// classification (the host watchdog halted the run); the crash
/// backtrace appends after it, so the ordering
/// `ERR_TIMED_OUT_NO_RESULT` before the backtrace frame holds.
#[test]
fn eval_timeout_with_crash_renders_both() {
    let entry = eevdf_entry("__eval_timeout_with_crash__");
    let mut result = make_vm_result("", "booting...", 0, true);
    result.crash_message = Some("PANIC: panicked at src/x.rs:7: boom\n   0: frame_one".to_string());
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(ERR_TIMED_OUT_NO_RESULT),
        "timeout stays the primary classification, got: {msg}",
    );
    assert!(
        msg.contains(ERR_GUEST_CRASHED_PREFIX),
        "the crash section must render its `guest crashed:` prefix, got: {msg}",
    );
    assert!(
        msg.contains("frame_one"),
        "the crash backtrace must append, not be dropped, got: {msg}",
    );
    let timeout_pos = msg.find(ERR_TIMED_OUT_NO_RESULT).unwrap();
    let crash_pos = msg.find("frame_one").unwrap();
    assert!(
        timeout_pos < crash_pos,
        "timeout reason must precede the appended crash section, got: {msg}",
    );
}

// -- no-result arm: scheduler-exited reason from kmsg (the `reason` ladder's `has_active_scheduling()` branch in evaluate_vm_result) --

/// No parseable result, active scheduler, stderr carrying a NON-EMPTY
/// scx-disable anchor body: the `reason` ladder's
/// `else if entry.scheduler.has_active_scheduling()` branch takes
/// the `if let Some(ev) = scx_exits.last()` non-empty
/// (`else` of `ev.message.is_empty()`) sub-arm and
/// renders `scheduler exited: <message>`, overriding the
/// `ERR_NO_TEST_RESULT_FROM_GUEST` fallback. crash_message is None and
/// output is empty so the earlier crash/panic rungs are not taken.
#[test]
fn eval_noresult_sched_exited_from_kmsg() {
    let entry = sched_entry("__eval_noresult_sched_exited__");
    let stderr = "[1.0] sched_ext: BPF scheduler \"scx_test\" disabled (BPF runtime error)\n";
    let result = make_vm_result("", stderr, 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("scheduler exited: BPF runtime error"),
        "non-empty kmsg message must drive the no-result reason, got: {msg}",
    );
    assert!(
        !msg.contains(ERR_NO_TEST_RESULT_FROM_GUEST),
        "the kmsg override must replace the default no-result fallback, got: {msg}",
    );
}

/// No parseable result, active scheduler, scx-disable anchor with an
/// EMPTY parenthesized body: the `reason` ladder takes the
/// `if ev.message.is_empty()` empty-message sub-arm
/// and renders `scheduler exited (<name>)` from
/// `ev.scheduler_name` (parsed `scx_test`).
#[test]
fn eval_noresult_sched_exited_empty_message() {
    let entry = sched_entry("__eval_noresult_sched_exited_empty__");
    let stderr = "[1.0] sched_ext: BPF scheduler \"scx_test\" disabled ()\n";
    let result = make_vm_result("", stderr, 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("scheduler exited (scx_test)"),
        "empty kmsg message must render the scheduler-name-only form, got: {msg}",
    );
    assert!(
        !msg.contains(ERR_NO_TEST_RESULT_FROM_GUEST),
        "the kmsg override must replace the default no-result fallback, got: {msg}",
    );
}

// -- no-result arm: extract_exit_from_dump_trace fallback (the `reason` ladder's `extract_exit_from_dump_trace` rung in evaluate_vm_result) --

/// No parseable result, active scheduler, stderr that has NO
/// `sched_ext: BPF scheduler "` kmsg anchor (so `parse_kmsg_window` is
/// empty) BUT carries a `sched_ext_dump:` trace with a
/// `triggered exit kind` anchor plus a same-CPU body line: the `reason`
/// ladder falls through to the
/// `else if let Some(reason) = extract_exit_from_dump_trace(...)` rung
/// and renders `scheduler exited: <reason>` with the
/// exact body the parser surfaces. Canonical input shape mirrors
/// `output.rs::extract_exit_from_dump_trace_canonical`.
#[test]
fn eval_noresult_exit_from_dump_trace_fallback() {
    let entry = sched_entry("__eval_noresult_dumptrace__");
    // trace_pipe shape: anchor line + same-CPU body line, both carrying
    // the `sched_ext_dump:` prefix; NO kmsg disable anchor. The body
    // after `sched_ext_dump:` trims to "apply_cell_config returned -EINVAL".
    let stderr = "\
ktstr-1 [001] 0.500: sched_ext_dump: scheduler[1] triggered exit kind 5:
ktstr-1 [001] 0.501: sched_ext_dump:   apply_cell_config returned -EINVAL
";
    let result = make_vm_result("", stderr, 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("scheduler exited: apply_cell_config returned -EINVAL"),
        "dump-trace fallback must surface the extracted exit reason, got: {msg}",
    );
    assert!(
        !msg.contains(ERR_NO_TEST_RESULT_FROM_GUEST),
        "the dump-trace rung must take precedence over the default fallback \
         (no kmsg anchor was present), got: {msg}",
    );
}

// -- failure-path info_notes section (the `info_section` binding in evaluate_vm_result's guest-fail block) --

/// A FAILING guest AssertResult that ALSO carries an `info_notes`
/// entry: the `--- info ---` section renders the note with its
/// two-space indent, AFTER the failure-details block. Pins the
/// `info_section` `if check_result.info_notes.is_empty()` non-empty
/// (`else`) arm — every existing eval
/// fixture leaves `info_notes` empty.
#[test]
fn eval_failure_renders_info_notes_section() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let mut assert = build_assert_result(
        false,
        vec![AssertDetail::new(DetailKind::Stuck, "stuck 9000ms")],
    );
    assert.note("context: ran under cgroup cgA");
    let entry = eevdf_entry("__eval_info_section__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("--- info ---"),
        "info section must render when info_notes is non-empty, got: {msg}",
    );
    assert!(
        msg.contains("context: ran under cgroup cgA"),
        "the note text must render (survives the TLV postcard roundtrip), got: {msg}",
    );
    assert!(
        msg.contains("stuck 9000ms"),
        "the failure detail must still render in the details block, got: {msg}",
    );
    let detail_pos = msg.find("stuck 9000ms").unwrap();
    let info_pos = msg.find("--- info ---").unwrap();
    assert!(
        detail_pos < info_pos,
        "the info section must follow the failures block (details-vs-info split), got: {msg}",
    );
}

// -- failure-path stats section + cgroup spread n/a (the `stats_section` binding in evaluate_vm_result's guest-fail block) --

/// A FAILING guest AssertResult whose `stats.cgroups` is non-empty:
/// the `--- stats ---` block renders, exercising BOTH the
/// `Some(spread)` and the `None` ("n/a") arms of the per-cgroup
/// `cg.spread.map_or_else(|| "n/a".to_string(), ...)` call. The header line renders
/// the run-level scalars; each per-cg line carries its distinct `iter=`
/// value so the index loop is proven to run for both cgroups.
#[test]
fn eval_failure_renders_stats_section_with_spread_na() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let mut assert = build_assert_result(
        false,
        vec![AssertDetail::new(DetailKind::Unfair, "spread too wide")],
    );
    assert.stats.total_workers = 6;
    assert.stats.total_cpus = 4;
    assert.stats.total_migrations = 11;
    assert.stats.worst_spread = 12.5;
    assert.stats.worst_gap_ms = 33;
    assert.stats.cgroups = vec![
        crate::assert::CgroupStats {
            num_workers: 2,
            num_cpus: 2,
            spread: Some(12.5),
            max_gap_ms: 33,
            total_migrations: 7,
            total_iterations: 900,
            ..Default::default()
        },
        crate::assert::CgroupStats {
            num_workers: 1,
            num_cpus: 1,
            spread: None,
            max_gap_ms: 5,
            total_migrations: 4,
            total_iterations: 42,
            ..Default::default()
        },
    ];
    let entry = eevdf_entry("__eval_stats_section__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("--- stats ---"),
        "stats section must render, got: {msg}"
    );
    assert!(
        msg.contains("6 workers, 4 cpus, 11 migrations, worst_spread=12.5%, worst_gap=33ms"),
        "the run-level header must render the scalars exactly, got: {msg}",
    );
    assert!(
        msg.contains("spread=12.5%"),
        "the Some(spread) arm must render the percentage, got: {msg}",
    );
    assert!(
        msg.contains("spread=n/a"),
        "the None spread arm must render `n/a`, not a fake 0%, got: {msg}",
    );
    assert!(
        msg.contains("iter=900"),
        "cgroup 0's distinct iteration count must render (loop ran), got: {msg}",
    );
    assert!(
        msg.contains("iter=42"),
        "cgroup 1's distinct iteration count must render (loop ran), got: {msg}",
    );
}

// -- failure-path repro section on guest-fail (the `repro` / `repro_section` bindings in evaluate_vm_result's guest-fail block) --

/// A FAILING guest AssertResult on the active-scheduler path: the
/// `repro_section`'s `repro.map(...)` chain (built from the
/// `if entry.scheduler.has_active_scheduling() { repro_fn(output) }`
/// binding) fires because
/// `entry.scheduler.has_active_scheduling()` is true (sched_entry) AND
/// `repro_fn` returns Some, rendering the `--- auto-repro ---` section
/// with the payload. This is the GUEST-AssertResult-fail arm —
/// distinct from `eval_sched_mid_test_exit_triggers_repro` which drives
/// the no-parseable-result arm.
#[test]
fn eval_failure_repro_section_on_guest_fail() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(
        false,
        vec![AssertDetail::new(DetailKind::Stuck, "worker 0 stuck")],
    );
    let entry = sched_entry("__eval_fail_repro_section__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let repro_fn = |_o: &str| Some("REPRO-PAYLOAD-X".to_string());
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &repro_fn,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("--- auto-repro ---"),
        "active-scheduler guest-fail must render the auto-repro section, got: {msg}",
    );
    assert!(
        msg.contains("REPRO-PAYLOAD-X"),
        "the repro closure's payload must render in the section, got: {msg}",
    );
}

/// Contrast control for `eval_failure_repro_section_on_guest_fail`:
/// with `eevdf_entry` (`has_active_scheduling() == false`) the repro
/// gate (the `if entry.scheduler.has_active_scheduling()` guard on the
/// `repro` binding) returns None even though `repro_fn` would have
/// returned Some — so NO `--- auto-repro ---` section renders.
#[test]
fn eval_failure_no_repro_section_without_active_scheduling() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(
        false,
        vec![AssertDetail::new(DetailKind::Stuck, "worker 0 stuck")],
    );
    let entry = eevdf_entry("__eval_fail_no_repro_section__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let repro_fn = |_o: &str| Some("REPRO-PAYLOAD-X".to_string());
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &repro_fn,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        !msg.contains("--- auto-repro ---"),
        "EEVDF (no active scheduling) must NOT render the auto-repro section \
         even when repro_fn returns Some, got: {msg}",
    );
}

// -- inconclusive monitor verdict fold (the `else if verdict.is_inconclusive()` arm of the monitor-threshold block in evaluate_vm_result) --

/// A PASSING guest AssertResult plus monitor data that is UNINITIALIZED
/// (constant `rq_clock` across every CPU and sample): the monitor
/// evaluator returns an INCONCLUSIVE verdict
/// (`MonitorThresholds::evaluate`'s `if !Self::data_looks_valid(...)`
/// arm -> summary "monitor data not yet
/// initialized"). The `else if verdict.is_inconclusive()` arm of the
/// monitor-threshold block folds a `DetailKind::Monitor` Inconclusive outcome
/// into `check_result` instead of bailing, so `evaluate` returns Ok and
/// the merged verdict is inconclusive.
#[test]
fn eval_monitor_inconclusive_folds_into_verdict() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let pass_assert = build_assert_result(true, vec![]);
    let entry = sched_entry("__eval_monitor_inconclusive__");
    // 30 samples, 2 CPUs each, ALL with rq_clock == 1000 -> after the
    // 20-sample warmup trim, 10 samples * 2 readings = 20 readings, all
    // identical -> data_looks_valid() == false -> inconclusive verdict.
    let constant_clock_samples: Vec<crate::monitor::MonitorSample> = (0..30)
        .map(|i| {
            crate::monitor::MonitorSample::new(
                (i * 100) as u64,
                vec![
                    crate::monitor::CpuSnapshot {
                        nr_running: 1,
                        scx_nr_running: 1,
                        local_dsq_depth: 0,
                        rq_clock: 1000,
                        scx_flags: 0,
                        event_counters: None,
                        schedstat: None,
                        vcpu_cpu_time_ns: None,
                        vcpu_perf: None,
                        avg_irq_util: None,
                        sched_domains: None,
                    },
                    crate::monitor::CpuSnapshot {
                        nr_running: 1,
                        scx_nr_running: 1,
                        local_dsq_depth: 0,
                        rq_clock: 1000,
                        scx_flags: 0,
                        event_counters: None,
                        schedstat: None,
                        vcpu_cpu_time_ns: None,
                        vcpu_perf: None,
                        avg_irq_util: None,
                        sched_domains: None,
                    },
                ],
            )
        })
        .collect();
    let summary =
        crate::monitor::MonitorSummary::from_samples_with_threshold(&constant_clock_samples, 0);
    let result = crate::vmm::VmResult {
        success: true,
        vcpus: 1,
        cpu_budget: 1,
        resolve_source: None,
        expect_auto_repro_satisfied: false,
        exit_code: 0,
        duration: std::time::Duration::from_secs(1),
        timed_out: false,
        watchdog_kill_reason: None,
        final_guest_phase: crate::vmm::GuestLifecyclePhase::Boot,
        final_progress_epoch: 0,
        bpf_map_writes_delivered: None,
        periodic_prereqs_ready: None,
        periodic_window_end: None,
        output: String::new(),
        stderr: String::new(),
        monitor: Some(crate::monitor::MonitorReport {
            samples: constant_clock_samples,
            summary,
            preemption_threshold_ns: 0,
            watchdog_observation: None,
            page_offset: 0,
            boot_wait_outcome: crate::monitor::BootWaitOutcome::NotConfigured,
            scx_event_counters_supported: false,
        }),
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &pass_assert,
            )],
        }),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        crash_message: None,
        cleanup_duration: None,
        cleanup_sched_delta: None,
        virtio_blk_counters: None,
        virtio_net_counters: None,
        snapshot_bridge: {
            let cb: crate::scenario::snapshot::CaptureCallback = std::sync::Arc::new(|_| None);
            crate::scenario::snapshot::SnapshotBridge::new(cb)
        },
        stats_client: None,
        periodic_fired: 0,
        periodic_real: 0,
        periodic_target: 0,
        kern_kaslr_offset: 0,
        entry_name: None,
        variant_hash: 0,
        host_vcpu_schedstat: None,
        contention_witness: None,
        periodic_series_cache: std::sync::OnceLock::new(),
    };
    let assertions = crate::assert::Assert::NO_OVERRIDES
        .max_imbalance_ratio(4.0)
        .fail_on_rq_clock_stuck(true)
        .with_monitor_defaults();
    let ar = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .expect("the inconclusive monitor arm must NOT bail — it merges and returns Ok");
    assert!(
        ar.is_inconclusive(),
        "the merged Inconclusive outcome must flip the verdict lattice to inconclusive",
    );
    let monitor_detail = ar
        .inconclusive_details()
        .find(|d| d.kind == DetailKind::Monitor)
        .expect("a DetailKind::Monitor inconclusive detail must be folded in");
    assert!(
        monitor_detail
            .message
            .starts_with("monitor evaluation inconclusive:"),
        "the folded detail must carry the inconclusive narrative, got: {}",
        monitor_detail.message,
    );
    assert!(
        monitor_detail
            .message
            .contains("monitor data not yet initialized"),
        "the narrative must carry the evaluator's uninitialized-data summary, got: {}",
        monitor_detail.message,
    );
}

// -- verdict_word = "inconclusive" in failure header (the `verdict_word` binding in evaluate_vm_result's guest-fail block) --

/// A check_result that is INCONCLUSIVE (not pass, not fail): the
/// failure-message header uses the `verdict_word` binding's
/// `if check_result.is_inconclusive() { "inconclusive" }` arm,
/// not "failed". Built by merging an
/// `AssertResult::inconclusive(...)` onto a passing base — the
/// resulting lattice is `is_fail=false / is_inconclusive=true /
/// is_pass=false`, so the failure-render block runs.
#[test]
fn eval_inconclusive_verdict_word_in_header() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let mut assert = build_assert_result(true, vec![]);
    assert.merge(crate::assert::AssertResult::inconclusive(
        AssertDetail::new(DetailKind::Other, "zero-denominator metric"),
    ));
    let entry = eevdf_entry("__eval_inconclusive_word__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("] inconclusive:"),
        "the header verdict word must be `inconclusive`, got: {msg}",
    );
    assert!(
        !msg.contains("] failed:"),
        "an inconclusive verdict must NOT render the `failed` header word, got: {msg}",
    );
    assert!(
        msg.contains("zero-denominator metric"),
        "the inconclusive detail must render (details block chains \
         inconclusive_details), got: {msg}",
    );
}

// -- scx_bpf_error matcher fold + ScxBpfErrorMatcherMismatch context --
// (the `matcher_details` / `matcher_mismatch` bindings and the
// `if matcher_mismatch { err.context(...) }` return in evaluate_vm_result)

/// A configured `expect_scx_bpf_error_contains` matcher whose needle is
/// ABSENT from the captured corpus folds a mismatch `AssertDetail` into
/// `check_result` (via the `matcher_details` `evaluate_scx_bpf_error_match`
/// fold under the `matcher_configured` gate) and wraps the failure `Err`
/// with the [`ScxBpfErrorMatcherMismatch`] context (the
/// `return Err(if matcher_mismatch { err.context(ScxBpfErrorMatcherMismatch) ... })`).
///
/// `entry.expect_err` is set to `true` so the matcher takes the
/// "substring not found" diagnostic path (with `expect_err = false`
/// `evaluate_scx_bpf_error_match` emits the MISUSE reminder instead —
/// the mismatch + context still fire, but the diagnostic text differs;
/// setting `expect_err = true` pins the substring-not-found text).
#[test]
fn eval_scx_bpf_error_matcher_mismatch_wraps_context() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(true, vec![]);
    let mut entry = sched_entry("__eval_scx_matcher_mismatch__");
    entry.expect_err = true;
    // Corpus (output -> sched_log_input) lacks the needle.
    let result = make_vm_result_with_assert("benign scheduler log line", "", 0, false, &assert);
    let assertions =
        crate::assert::Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("EXPECTED-NEEDLE");
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    assert!(
        err.downcast_ref::<ScxBpfErrorMatcherMismatch>().is_some(),
        "the matcher mismatch must attach the ScxBpfErrorMatcherMismatch context \
         (anyhow context-aware downcast), got: {err:#}",
    );
    let msg = format!("{err:#}");
    assert!(
        msg.contains("substring not found in the scheduler log + sched_ext dump corpus"),
        "the substring-not-found diagnostic must render (expect_err=true path), got: {msg}",
    );
}

/// Negative control for `eval_scx_bpf_error_matcher_mismatch_wraps_context`:
/// with NO matcher configured (`matcher_configured == false`), an
/// independently-failing check_result still produces an `Err`, but it
/// is NOT wrapped with the [`ScxBpfErrorMatcherMismatch`] context.
///
/// The failure must come from a SEPARATE source
/// (`build_assert_result(false, ...)`) — a passing
/// guest result with no matcher / monitor / host failure returns
/// `Ok`, so there would be no `Err` to inspect.
#[test]
fn eval_no_matcher_no_mismatch_context() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(
        false,
        vec![AssertDetail::new(DetailKind::Stuck, "independent failure")],
    );
    let entry = sched_entry("__eval_no_matcher_context__");
    let result = make_vm_result_with_assert("benign scheduler log line", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    assert!(
        err.downcast_ref::<ScxBpfErrorMatcherMismatch>().is_none(),
        "with no matcher configured the mismatch context must NOT be attached, got: {err:#}",
    );
    assert!(
        format!("{err}").contains("independent failure"),
        "the independent failure detail must still render, got: {err}",
    );
}

// -- post_vm_err fold on guest-AssertResult path (the `if let Some(err) = post_vm_err` block in evaluate_vm_result's parse-success arm) --

/// A host-side `post_vm` callback `Err` folds a `DetailKind::Other`
/// failure into an otherwise-PASSING guest `check_result`
/// (the `if let Some(err) = post_vm_err { check_result.merge(...) }`
/// block), flipping the verdict to a hard failure. The
/// folded detail renders the exact `post_vm callback returned Err: ...`
/// text from that block's `format!("post_vm callback returned Err: {err:#}")`.
#[test]
fn eval_post_vm_err_folds_into_guest_pass() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(true, vec![]);
    let entry = eevdf_entry("__eval_post_vm_fold__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let pv = anyhow::anyhow!("snapshot bridge captured nothing");
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        Some(&pv),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("post_vm callback returned Err: snapshot bridge captured nothing"),
        "the post_vm Err must fold in with its exact rendered text, got: {msg}",
    );
    assert!(
        msg.contains("] failed:"),
        "the folded Other detail is a hard Fail -> verdict word `failed`, got: {msg}",
    );
}

// -- scheduler-log >200-line truncation (the `tail` binding in evaluate_vm_result's `sched_log_section` builder) --

/// A no-result run whose scheduler log carries >200 NON-verifier lines
/// triggers the tail-truncation branch in the `sched_log_section`
/// builder's `tail` binding: the
/// `if !is_verifier && lines.len() > 200` arm renders
/// `[N lines truncated]` followed by the last 200 lines.
///
/// The body is 250 DISTINCT `frame_<i>+0x10` lines so `collapse_cycles`
/// finds no repeating cycle (each line is unique -> no anchor repeats
/// >= 3 times) and leaves all 250 intact; `is_verifier` is false (the
/// > lines contain neither "processed" nor "insns"). With exactly 250
/// > post-collapse lines the skip count is `250 - 200 = 50`, the last
/// > line (`frame_249+0x10`) survives in the kept tail, and an early line
/// > (`frame_0+0x10`) is truncated.
#[test]
fn eval_sched_log_truncates_over_200_lines() {
    let body = (0..250)
        .map(|i| format!("frame_{i}+0x10"))
        .collect::<Vec<_>>()
        .join("\n");
    let log = format!("{SCHED_OUTPUT_START}\n{body}\n{SCHED_OUTPUT_END}");
    let entry = sched_entry("__eval_sched_log_trunc__");
    let result = make_vm_result(&log, "", 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("[50 lines truncated]"),
        "250 distinct non-verifier lines must truncate to the last 200 (skip 50), got: {msg}",
    );
    assert!(
        msg.contains("frame_249+0x10"),
        "the last log line must survive in the kept tail, got: {msg}",
    );
    assert!(
        !msg.contains("frame_0+0x10"),
        "an early log line must be truncated out of the kept tail, got: {msg}",
    );
}

// -- guest-fail block bug_summary_line() closure Some-arm (the `bug_summary_line` closure in evaluate_vm_result) --

/// On the guest-AssertResult-fail path the `bug_summary_line()` closure
/// (the `|| -> String { match ...extract_bug_summary(...) }` closure)
/// extracts a `scx_bpf_error`-class line from the
/// scheduler-log corpus and prepends a `BUG SUMMARY: <text>\n` line to
/// the rendered failure message (concatenated as the `bug_summary_line()`
/// arg of the failure `format!`, ahead of
/// the `ktstr_test` header). `extract_bug_summary` falls through its
/// dump scan to the `for line in sched_clean.lines() { if line.contains("scx_bpf_error")` loop,
/// returning the
/// trimmed `scx_bpf_error: cell config invalid` line. Pins the Some-arm
/// rendering through `evaluate_vm_result`: every existing eval failure
/// fixture either has no `scx_bpf_error` substring (so the closure
/// returns `None`) or exercises `write_placeholder` rather than this
/// eval closure.
///
/// `stderr_color()` is `false` under the captured-stderr test harness
/// (`cli::util::stderr_color` reads `std::io::stderr().is_terminal()`,
/// cached in a `OnceLock`), so the plain
/// `BUG SUMMARY: ` form renders — the combined substring asserted below
/// exists only in that plain form, not the `\x1b`-wrapped one. The
/// `output` carries the line bracketed by `SCHED_OUTPUT_START` /
/// `SCHED_OUTPUT_END`; `guest_messages` holds only the TEST_RESULT TLV,
/// so `concat_sched_log_chunks` is empty and `sched_log_input` falls
/// back to `output` (the `if !sched_log_merged.is_empty() { ... } else { output }`
/// binding of `sched_log_input`), the corpus the closure scans.
#[test]
fn eval_failure_renders_bug_summary_line_via_closure() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(false, vec![AssertDetail::new(DetailKind::Stuck, "stuck")]);
    let output =
        format!("{SCHED_OUTPUT_START}\nscx_bpf_error: cell config invalid\n{SCHED_OUTPUT_END}",);
    let entry = sched_entry("__eval_bug_summary_closure__");
    let result = make_vm_result_with_assert(&output, "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("BUG SUMMARY: scx_bpf_error: cell config invalid"),
        "the closure's Some-arm must render the plain `BUG SUMMARY: <text>` line, got: {msg}",
    );
    let summary_pos = msg.find("BUG SUMMARY:").unwrap();
    let name_pos = msg.find("ktstr_test").unwrap();
    assert!(
        summary_pos < name_pos,
        "the BUG SUMMARY line must precede the ktstr_test header, got: {msg}",
    );
}

// -- guest-fail block periodic-samples section wiring (the `periodic_section` binding + its `periodic_section` arg in evaluate_vm_result's failure format!) --

/// On the guest-fail path the periodic-samples section
/// (the `periodic_section` binding's
/// `format_periodic_samples_section(result)` call)
/// renders into the failure message (its `periodic_section` arg of the
/// failure `format!`) when
/// `result.periodic_target > 0`. Every existing fail-arm eval fixture
/// leaves `periodic_target == 0`, so the section returns `""` and never
/// appears in an asserted failure message; the non-zero-target
/// render-into-message wiring is otherwise untested. `make_vm_result_*`
/// helpers can't set the periodic fields, so the `VmResult` is built via
/// the `test_fixture()` struct-update idiom with the fields overridden.
/// Exact strings per `format_periodic_samples_section`
/// (its `if real < fired` / `if fired < target` line gates): with
/// `fired=2 real=2 target=4` the degraded-
/// placeholder line is skipped (`real < fired` false) and the
/// missing-samples line renders (`fired < target` true).
#[test]
fn eval_failure_renders_periodic_samples_section() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(false, vec![AssertDetail::new(DetailKind::Stuck, "stuck")]);
    let entry = eevdf_entry("__eval_periodic_section__");
    let result = crate::vmm::VmResult {
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &assert,
            )],
        }),
        periodic_fired: 2,
        periodic_real: 2,
        periodic_target: 4,
        ..crate::vmm::VmResult::test_fixture()
    };
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("--- periodic samples ---"),
        "periodic_target>0 must render the periodic section into the failure message, got: {msg}",
    );
    assert!(
        msg.contains("fired 2/4 periodic snapshots (50% coverage)"),
        "the fired/target coverage line must render exactly, got: {msg}",
    );
    assert!(
        msg.contains("missing 2 sample(s)"),
        "the missing-samples line must render (fired < target), got: {msg}",
    );
}

// -- guest-fail block temporal-assertions section wiring (the `temporal_section` binding + its `temporal_section` arg in evaluate_vm_result's failure format!) --

/// On the guest-fail path the temporal-assertions section
/// (the `temporal_section` binding's
/// `format_temporal_assertions_section(&check_result)` call)
/// renders into the failure message (its `temporal_section` arg of the
/// failure `format!`)
/// when `check_result` carries a `DetailKind::Temporal` detail.
/// `format_temporal_assertions_section` is unit-tested directly in
/// output.rs, but no eval test feeds a Temporal-tagged detail through
/// `evaluate_vm_result`, so the boundary wiring — that a post-TLV-
/// roundtrip Temporal detail reaches the section — is otherwise
/// unverified. The Temporal detail survives the postcard TLV roundtrip
/// (`DetailKind` is a plain serde enum). Exact header per
/// `format_temporal_assertions_section`'s
/// `"{n} temporal assertion entry(ies):"` push.
#[test]
fn eval_failure_renders_temporal_assertions_section() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(
        false,
        vec![AssertDetail::new(
            DetailKind::Temporal,
            "sample tag p3 violated rate_within",
        )],
    );
    let entry = eevdf_entry("__eval_temporal_section__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("--- temporal assertions ---"),
        "a Temporal detail must render the temporal section into the failure message, got: {msg}",
    );
    assert!(
        msg.contains("1 temporal assertion entry(ies):"),
        "the temporal section header must render the entry count exactly, got: {msg}",
    );
    assert!(
        msg.contains("sample tag p3 violated rate_within"),
        "the Temporal detail message must render (survives the TLV roundtrip), got: {msg}",
    );
}

// -- no-result console-suppression else arm (the no-result `console_section` binding in evaluate_vm_result) --

/// The no-parseable-result `console_section` binding's
/// `else { String::new() }` arm is reached only when
/// `has_sched_output == true` AND `!verbose()` AND
/// `!entry.scheduler.has_active_scheduling()` (the negation of its
/// `if !has_sched_output || verbose() || entry.scheduler.has_active_scheduling()`
/// guard). Every existing EEVDF
/// no-result fixture lacks `SCHED_OUTPUT_START`, so `has_sched_output`
/// is false and the diagnostics section always renders; the suppression
/// branch is otherwise unexercised. EEVDF (`has_active_scheduling()
/// == false`) plus `SCHED_OUTPUT_START` in `output` (sets
/// `has_sched_output` via the `output.contains(SCHED_OUTPUT_START) || ...`
/// binding) plus `verbose()` false drives
/// the else arm, so no `--- diagnostics ---` appears. `verbose()` reads
/// `RUST_BACKTRACE` (`test_support::runtime::verbose`), removed here under
/// `lock_env()`. The reason stays `ERR_NO_TEST_FUNCTION_OUTPUT` (EEVDF,
/// no crash/panic, the final `else` rung of the `reason` ladder).
#[test]
fn eval_noresult_eevdf_with_sched_output_suppresses_console_section() {
    let _lock = lock_env();
    let _bt = EnvVarGuard::remove("RUST_BACKTRACE");
    let entry = eevdf_entry("__eval_console_suppress__");
    let output = format!("{SCHED_OUTPUT_START}\nnoise\n{SCHED_OUTPUT_END}",);
    let result = make_vm_result(&output, "Kernel panic", 1, false);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        !msg.contains("--- diagnostics ---"),
        "EEVDF + SCHED_OUTPUT present + non-verbose must suppress the diagnostics section, got: {msg}",
    );
    assert!(
        msg.contains(ERR_NO_TEST_FUNCTION_OUTPUT),
        "the EEVDF no-result reason must still render, got: {msg}",
    );
}

// -- guest-fail block build_monitor_section() empty else arm (the `build_monitor_section` closure in evaluate_vm_result) --

/// `build_monitor_section()`'s `String::new()` else arm:
/// `entry.scheduler.has_active_scheduling()` is true
/// (sched_entry) but `result.monitor` is `None`, so the closure's
/// `if entry.scheduler.has_active_scheduling() && let Some(ref monitor) = result.monitor`
/// guard fails and the closure
/// returns empty — no `--- monitor ---` section despite an active
/// scheduler. `eval_sched_exit_includes_monitor` covers the
/// monitor=Some arm; the `eval_eevdf_*` fixtures take the
/// `has_active_scheduling()==false` short-circuit. `make_vm_result*`
/// sets `monitor: None`, so the guest-fail block's
/// `let monitor_section = build_monitor_section();` call takes the empty else and
/// the rendered failure message carries no monitor section.
#[test]
fn eval_sched_fail_with_no_monitor_omits_monitor_section() {
    let _lock = lock_env();
    let _sd = isolated_sidecar_dir();
    let assert = build_assert_result(
        false,
        vec![AssertDetail::new(DetailKind::Stuck, "worker 0 stuck")],
    );
    let entry = sched_entry("__eval_no_monitor_section__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        !msg.contains("--- monitor ---"),
        "active scheduler with monitor=None must omit the monitor section, got: {msg}",
    );
    assert!(
        msg.contains("worker 0 stuck"),
        "the failure detail must still render (the guest-fail path ran), got: {msg}",
    );
}

// -- contention_not_attached_skip_reason: skip vs fail --

/// Build a bulk drain with a single `SchedulerNotAttached` lifecycle
/// frame carrying `reason` as its UTF-8 suffix (the shape the guest's
/// `send_lifecycle` emits). `lifecycle_drain` drops the suffix, so the
/// reason-bearing frame is built here.
fn not_attached_drain(reason: &str) -> crate::vmm::host_comms::BulkDrainResult {
    use crate::vmm::wire::{LifecyclePhase, MSG_TYPE_LIFECYCLE, ShmEntry};
    let mut payload = vec![LifecyclePhase::SchedulerNotAttached.wire_value()];
    payload.extend_from_slice(reason.as_bytes());
    crate::vmm::host_comms::BulkDrainResult {
        entries: vec![ShmEntry {
            msg_type: MSG_TYPE_LIFECYCLE,
            payload,
            crc_ok: true,
        }],
    }
}

/// A still-in-flight enable (state=enabling) under host oversubscription
/// (64 vCPUs on 8 allowed CPUs = 8x) is a contention SKIP.
#[test]
fn contention_skip_enabling_oversubscribed_skips() {
    let drain = not_attached_drain("timeout: state=enabling");
    let reason = contention_not_attached_skip_reason(Some(&drain), 64, 8, None);
    assert!(
        reason
            .as_deref()
            .is_some_and(|r| r.contains("oversubscription") && r.contains("enabling")),
        "state=enabling + 8x oversubscription must classify as a contention skip, got {reason:?}",
    );
}

/// The SAME still-in-flight enable on a FITTING host (8 vCPUs on 8 CPUs
/// = 1.0x) is NOT a skip — a startup stall with no contention to blame
/// is a real defect that must FAIL.
#[test]
fn contention_skip_enabling_fitting_host_fails() {
    let drain = not_attached_drain("timeout: state=enabling");
    assert_eq!(
        contention_not_attached_skip_reason(Some(&drain), 8, 8, None),
        None,
        "state=enabling on a fitting host (1.0x) is a real defect, not a skip",
    );
}

/// A REJECTED enable (state=disabling) is a real defect and must FAIL
/// even under heavy oversubscription.
#[test]
fn contention_skip_disabling_never_skips() {
    let drain = not_attached_drain("timeout: state=disabling");
    assert_eq!(
        contention_not_attached_skip_reason(Some(&drain), 64, 8, None),
        None,
        "a rejected enable (state=disabling) must FAIL, never skip, even oversubscribed",
    );
}

/// state=disabled (a rejected enable, further along the disable path)
/// also never skips.
#[test]
fn contention_skip_disabled_never_skips() {
    let drain = not_attached_drain("timeout: state=disabled");
    assert_eq!(
        contention_not_attached_skip_reason(Some(&drain), 64, 8, None),
        None,
    );
}

/// A missing sched_ext sysfs (a config fault, not contention) never
/// skips.
#[test]
fn contention_skip_sysfs_absent_never_skips() {
    let drain = not_attached_drain("sched_ext sysfs absent");
    assert_eq!(
        contention_not_attached_skip_reason(Some(&drain), 64, 8, None),
        None,
    );
}

/// A `SchedulerDied` frame (the process exited during load — a verifier
/// reject / crash) never skips, even oversubscribed.
#[test]
fn contention_skip_scheduler_died_never_skips() {
    let drain = lifecycle_drain(&[crate::vmm::wire::LifecyclePhase::SchedulerDied]);
    assert_eq!(
        contention_not_attached_skip_reason(Some(&drain), 64, 8, None),
        None,
        "SchedulerDied is a real defect and must FAIL, never skip",
    );
}

/// No lifecycle frame at all (a generic no-result run) never skips.
#[test]
fn contention_skip_no_frame_never_skips() {
    assert_eq!(contention_not_attached_skip_reason(None, 64, 8, None), None);
}
