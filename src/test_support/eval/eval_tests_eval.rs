//! Part of the eval module's unit-test suite, split across sibling
//! `eval_tests*.rs` files to keep each under the size ceiling. Child of
//! `eval`: reaches the production core via `super::` / `super::super::`.
use super::super::output::{
    STAGE_INIT_NOT_STARTED, STAGE_INIT_STARTED_NO_PAYLOAD, STAGE_PAYLOAD_STARTED_NO_RESULT,
};
use super::super::test_helpers::{
    EVAL_TOPO, EnvVarGuard, build_assert_result, eevdf_entry, isolated_cache_dir, lifecycle_drain,
    lock_env, make_vm_result, make_vm_result_with_assert, no_repro, sched_entry,
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
fn eval_check_result_failed_includes_details() {
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
    let assert = build_assert_result(true, vec![]);
    let mut entry = eevdf_entry("__eval_cleanup_overshoot__");
    entry.cleanup_budget = Some(std::time::Duration::from_secs(1));
    let mut result = make_vm_result_with_assert("", "", 0, false, &assert);
    result.cleanup_duration = Some(std::time::Duration::from_secs(10));
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
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

/// Cleanup-budget no-fire: when the run's `cleanup_duration` is
/// strictly under the entry's `cleanup_budget`, the guest's
/// passing `AssertResult` survives the merge and
/// `evaluate_vm_result` returns `Ok`. Verifies that
/// `measured < budget` passes without folding a fail; the exact
/// `measured == budget` boundary is covered separately by
/// [`eval_cleanup_budget_equal_passes`].
#[test]
fn eval_cleanup_budget_under_passes() {
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
        expect_auto_repro_satisfied: false,
        exit_code: 0,
        duration: std::time::Duration::from_secs(1),
        timed_out: false,
        output,
        stderr: String::new(),
        monitor: Some(crate::monitor::MonitorReport {
            samples: imbalance_samples,
            summary,
            preemption_threshold_ns: 0,
            watchdog_observation: None,
            page_offset: 0,
            boot_wait_outcome: crate::monitor::BootWaitOutcome::NotConfigured,
        }),
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &pass_assert,
            )],
        }),
        stimulus_events: Vec::new(),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        crash_message: None,
        cleanup_duration: None,
        virtio_blk_counters: None,
        virtio_net_counters: None,
        snapshot_bridge: {
            let cb: crate::scenario::snapshot::CaptureCallback = std::sync::Arc::new(|_| None);
            crate::scenario::snapshot::SnapshotBridge::new(cb)
        },
        stats_client: None,
        periodic_fired: 0,
        periodic_target: 0,
        kern_kaslr_offset: 0,
        entry_name: None,
    };
    let assertions = crate::assert::Assert::NO_OVERRIDES
        .max_imbalance_ratio(4.0)
        .fail_on_stall(true)
        .with_monitor_defaults();
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
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
        expect_auto_repro_satisfied: false,
        exit_code: 1,
        duration: std::time::Duration::from_secs(1),
        timed_out: false,
        output: String::new(),
        stderr: String::new(),
        monitor: Some(crate::monitor::MonitorReport {
            samples: vec![],
            summary: crate::monitor::MonitorSummary {
                total_samples: 5,
                max_imbalance_ratio: 3.0,
                max_local_dsq_depth: 2,
                stuck_detected: false,
                event_deltas: None,
                schedstat_deltas: None,
                prog_stats_deltas: None,
                ..Default::default()
            },
            preemption_threshold_ns: 0,
            watchdog_observation: None,
            page_offset: 0,
            boot_wait_outcome: crate::monitor::BootWaitOutcome::NotConfigured,
        }),
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &assert,
            )],
        }),
        stimulus_events: Vec::new(),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        crash_message: None,
        cleanup_duration: None,
        virtio_blk_counters: None,
        virtio_net_counters: None,
        snapshot_bridge: {
            let cb: crate::scenario::snapshot::CaptureCallback = std::sync::Arc::new(|_| None);
            crate::scenario::snapshot::SnapshotBridge::new(cb)
        },
        stats_client: None,
        periodic_fired: 0,
        periodic_target: 0,
        kern_kaslr_offset: 0,
        entry_name: None,
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
        expect_auto_repro_satisfied: false,
        exit_code: 0,
        duration: std::time::Duration::from_secs(1),
        timed_out: false,
        output,
        stderr: String::new(),
        monitor: Some(crate::monitor::MonitorReport {
            samples: imbalance_samples,
            summary,
            preemption_threshold_ns: 0,
            watchdog_observation: None,
            page_offset: 0,
            boot_wait_outcome: crate::monitor::BootWaitOutcome::NotConfigured,
        }),
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult {
            entries: vec![crate::test_support::test_helpers::assert_result_tlv_entry(
                &pass_assert,
            )],
        }),
        stimulus_events: Vec::new(),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        crash_message: None,
        cleanup_duration: None,
        virtio_blk_counters: None,
        virtio_net_counters: None,
        snapshot_bridge: {
            let cb: crate::scenario::snapshot::CaptureCallback = std::sync::Arc::new(|_| None);
            crate::scenario::snapshot::SnapshotBridge::new(cb)
        },
        stats_client: None,
        periodic_fired: 0,
        periodic_target: 0,
        kern_kaslr_offset: 0,
        entry_name: None,
    };
    let assertions = crate::assert::Assert::NO_OVERRIDES
        .max_imbalance_ratio(4.0)
        .fail_on_stall(true)
        .with_monitor_defaults();
    let msg = format!(
        "{}",
        evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
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
/// classification (a SKIP-able `ResourceContention`) into a
/// hard-error plain anyhow.
///
/// The test feeds the seam a representative shared-lock-timeout
/// rendering (matching the literal format produced at
/// `flock.rs::try_flock_with_deadline` — `"flock LOCK_SH on
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
