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
        vcpus: 1,
        cpu_budget: 1,
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
        periodic_real: 0,
        periodic_target: 0,
        kern_kaslr_offset: 0,
        entry_name: None,
        periodic_series_cache: std::sync::OnceLock::new(),
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
        vcpus: 1,
        cpu_budget: 1,
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
        vcpus: 1,
        cpu_budget: 1,
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
        periodic_real: 0,
        periodic_target: 0,
        kern_kaslr_offset: 0,
        entry_name: None,
        periodic_series_cache: std::sync::OnceLock::new(),
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
    use crate::timeline::StimulusEvent;
    // A FAILING guest AssertResult -> evaluate_vm_result takes the failure arm
    // (returns Err with the rendered message).
    let mut guest_assert = build_assert_result(
        false,
        vec![crate::assert::AssertDetail::new(
            crate::assert::DetailKind::Starved,
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
/// evaluate_vm_result's stats.phases must contain a SYNTHESIZED bucket
/// (sample_count==0) for the uncaptured steps carrying their
/// stimulus-derived iteration_rate. This is the --cell-parent-cgroup
/// short-interior-step scenario, pinned through
/// evaluate_vm_result (not just build_phase_buckets_with_stimulus): the
/// non-empty synthesized buckets also flip timeline selection onto the
/// from_phase_buckets path.
#[test]
fn evaluate_synthesizes_phase_buckets_for_uncaptured_steps() {
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
        Some(1000.0),
        "synthesized step 2 carries its stimulus-derived rate \
         (StepStart[2]=1000 -> StepStart[3]=2000 over 1s) through evaluate",
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

// -- timed-out arm: scheduler-exited reason override (the `timeout_reason` block in evaluate_vm_result's `if result.timed_out` arm) --

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
        msg.contains("--- watchdog ---"),
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
        msg.contains("--- watchdog ---"),
        "watchdog diagnostic block must render, got: {msg}",
    );
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
        expect_auto_repro_satisfied: false,
        exit_code: 0,
        duration: std::time::Duration::from_secs(1),
        timed_out: false,
        output: String::new(),
        stderr: String::new(),
        monitor: Some(crate::monitor::MonitorReport {
            samples: constant_clock_samples,
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
        periodic_real: 0,
        periodic_target: 0,
        kern_kaslr_offset: 0,
        entry_name: None,
        periodic_series_cache: std::sync::OnceLock::new(),
    };
    let assertions = crate::assert::Assert::NO_OVERRIDES
        .max_imbalance_ratio(4.0)
        .fail_on_stall(true)
        .with_monitor_defaults();
    let ar = evaluate_vm_result(
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

// -- host_extract_failures fold (the `for detail in host_extract_failures` loop in evaluate_vm_result's parse-success arm) --

/// A non-empty `host_extract_failures` slice (the 6th param) folds each
/// detail into an otherwise-PASSING guest `check_result` via
/// the `for detail in host_extract_failures {
/// check_result.merge(AssertResult::fail(detail.clone())) }` loop,
/// flipping the verdict to failed and rendering the
/// host-extract detail in the details block.
#[test]
fn eval_host_extract_failures_fold_into_guest_pass() {
    let assert = build_assert_result(true, vec![]);
    let entry = eevdf_entry("__eval_host_extract_fold__");
    let result = make_vm_result_with_assert("", "", 0, false, &assert);
    let assertions = crate::assert::Assert::NO_OVERRIDES;
    let host_fails = vec![AssertDetail::new(
        DetailKind::Other,
        "llm model unavailable",
    )];
    let err = evaluate_vm_result(
        &entry,
        &result,
        &assertions,
        &[],
        &[],
        &host_fails,
        &EVAL_TOPO,
        &no_repro,
        None,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("llm model unavailable"),
        "the host-extract failure detail must render in the details block, got: {msg}",
    );
    assert!(
        msg.contains("] failed:"),
        "the folded host-extract failure must flip the verdict to failed, got: {msg}",
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
/// lines contain neither "processed" nor "insns"). With exactly 250
/// post-collapse lines the skip count is `250 - 200 = 50`, the last
/// line (`frame_249+0x10`) survives in the kept tail, and an early line
/// (`frame_0+0x10`) is truncated.
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
