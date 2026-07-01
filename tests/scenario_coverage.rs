use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::test_support::{BpfMapWrite, KtstrTestEntry, Scheduler, SchedulerSpec, Topology};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

const TOPO_1L_4C_1T: Topology = Topology {
    llcs: 1,
    cores_per_llc: 4,
    threads_per_core: 1,
    numa_nodes: 1,
    nodes: None,
    distances: None,
};

// -- basic --

#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 2048,
    max_spread_pct = 80.0
)]
fn cover_cgroup_pipe_io(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_cgroup_pipe_io(ctx)
}

// Test exercises workload generation (Normal/Batch/Idle/FIFO mix),
// not scheduler correctness. Fairness rate ceilings (keep_last,
// fallback) that activate when ANY monitor threshold is set are
// raised here to skip those checks while keeping sustained_samples
// for the stall-pattern coverage.
#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 4, threads = 1, memory_mib = 2048, sustained_samples = 25, max_keep_last_rate = 1000000000.0, max_fallback_rate = 1000000000.0)]
fn cover_sched_mixed(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_sched_mixed(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_host_cgroup_contention(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_host_cgroup_contention(ctx)
}

// -- affinity --

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_affinity_change(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::affinity::custom_cgroup_affinity_change(ctx)
}

#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 2048,
    max_spread_pct = 80.0
)]
fn cover_cgroup_multicpu_pin(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::affinity::custom_cgroup_multicpu_pin(ctx)
}

#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 2048,
    max_spread_pct = 80.0
)]
fn cover_cgroup_cpuset_multicpu_pin(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::affinity::custom_cgroup_cpuset_multicpu_pin(ctx)
}

// -- cpuset --

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_cpuset_apply_midrun(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::cpuset::custom_cgroup_cpuset_apply_midrun(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_cpuset_clear_midrun(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::cpuset::custom_cgroup_cpuset_clear_midrun(ctx)
}

#[ktstr_test(llcs = 1, cores = 8, threads = 1, memory_mib = 2048)]
fn cover_cgroup_cpuset_resize(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::cpuset::custom_cgroup_cpuset_resize(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 2, memory_mib = 2048)]
fn cover_cgroup_cpuset_swap_disjoint(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::cpuset::custom_cgroup_cpuset_swap_disjoint(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_cpuset_workload_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::cpuset::custom_cgroup_cpuset_workload_imbalance(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_cpuset_change_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::cpuset::custom_cgroup_cpuset_change_imbalance(ctx)
}

// scenario-coverage test for the cpuset-load-shift scenario under the
// scx-ktstr toy scheduler. Fairness rate ceilings (keep_last, fallback)
// are raised to skip those checks — see cover_sched_mixed for the
// design rationale (toy scheduler trips the defaults whenever any
// monitor threshold is set).
#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 4, threads = 1, memory_mib = 2048, max_imbalance_ratio = 20.0, sustained_samples = 15, watchdog_timeout_s = 15, max_keep_last_rate = 1000000000.0, max_fallback_rate = 1000000000.0)]
fn cover_cgroup_cpuset_load_shift(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::cpuset::custom_cgroup_cpuset_load_shift(ctx)
}

// -- dynamic --

// Fairness rate ceilings raised — see cover_sched_mixed for rationale.
#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 4, threads = 1, memory_mib = 2048, sustained_samples = 25, watchdog_timeout_s = 15, max_keep_last_rate = 1000000000.0, max_fallback_rate = 1000000000.0)]
fn cover_cgroup_add_midrun(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::dynamic::custom_cgroup_add_midrun(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_remove_midrun(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::dynamic::custom_cgroup_remove_midrun(ctx)
}

#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 2048,
    max_spread_pct = 80.0,
    max_gap_ms = 6000
)]
fn cover_cgroup_rapid_churn(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::dynamic::custom_cgroup_rapid_churn(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_cpuset_add_remove(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::dynamic::custom_cgroup_cpuset_add_remove(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_add_during_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::dynamic::custom_cgroup_add_during_imbalance(ctx)
}

// -- interaction --

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_imbalance_mixed_workload(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_imbalance_mixed_workload(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048, max_gap_ms = 6000)]
fn cover_cgroup_add_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_add_load_imbalance(ctx)
}

// Test exercises workload generation (heavy/light load oscillation
// across phases), not scheduler correctness. See cover_sched_mixed
// for rate-ceiling rationale.
#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 4, threads = 1, memory_mib = 2048, sustained_samples = 25, max_keep_last_rate = 1000000000.0, max_fallback_rate = 1000000000.0)]
fn cover_cgroup_load_oscillation(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_load_oscillation(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 2, memory_mib = 2048)]
fn cover_cgroup_4way_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_4way_load_imbalance(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_cpuset_imbalance_combined(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_cpuset_imbalance_combined(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_cpuset_overlap_imbalance_combined(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_cpuset_overlap_imbalance_combined(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_no_ctrl_task_migration(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_no_ctrl_task_migration(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_no_ctrl_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_no_ctrl_imbalance(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_no_ctrl_cpuset_change(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_no_ctrl_cpuset_change(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048, max_gap_ms = 8000)]
fn cover_cgroup_no_ctrl_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_no_ctrl_load_imbalance(ctx)
}

#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 4, threads = 1, memory_mib = 2048, sustained_samples = 25, watchdog_timeout_s = 15)]
fn cover_cgroup_io_compute_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_io_compute_imbalance(ctx)
}

// -- nested --

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_nested_cgroup_steady(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::nested::custom_nested_cgroup_steady(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_nested_cgroup_task_move(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::nested::custom_nested_cgroup_task_move(ctx)
}

#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 2048,
    max_spread_pct = 80.0,
    max_gap_ms = 6000
)]
fn cover_nested_cgroup_rapid_churn(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::nested::custom_nested_cgroup_rapid_churn(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_nested_cgroup_cpuset(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::nested::custom_nested_cgroup_cpuset(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_nested_cgroup_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::nested::custom_nested_cgroup_imbalance(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_nested_cgroup_no_ctrl(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::nested::custom_nested_cgroup_no_ctrl(ctx)
}

// -- stress --

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_per_cpu(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::stress::custom_cgroup_per_cpu(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_exhaust_reuse(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::stress::custom_cgroup_exhaust_reuse(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_dsq_contention(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::stress::custom_cgroup_dsq_contention(ctx)
}

#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 2,
    memory_mib = 2048,
    max_spread_pct = 80.0
)]
fn cover_cgroup_workload_variety(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::stress::custom_cgroup_workload_variety(ctx)
}

#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 2,
    memory_mib = 2048,
    max_spread_pct = 80.0,
    duration_s = 10
)]
fn cover_cgroup_cpuset_workload_variety(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::stress::custom_cgroup_cpuset_workload_variety(ctx)
}

#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 2,
    memory_mib = 2048,
    max_spread_pct = 80.0
)]
fn cover_cgroup_dynamic_workload_variety(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::stress::custom_cgroup_dynamic_workload_variety(ctx)
}

#[ktstr_test(llcs = 2, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cgroup_cpuset_cross_llc_race(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::stress::custom_cgroup_cpuset_cross_llc_race(ctx)
}

// -- performance --

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cache_pressure_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::performance::custom_cache_pressure_imbalance(ctx)
}

// performance_mode is REQUIRED here, unlike the sibling perf-coverage
// scenarios: this one binds each cgroup to a SEPARATE guest LLC
// (CpusetSpec::llc(0)/llc(1)) to make cross-LLC wake-affine placement
// observable, then asserts a tight wake-latency CV. Without perf-mode's
// host-LLC pinning, the two guest LLCs float on shared host CPUs; under
// host oversubscription (CI runs `-j $(nproc)`) one LLC's vCPUs starve, so
// that cgroup's worker does zero work (exit=TimedOut) and the CV/throughput
// asserts measure noise rather than scheduler behavior. Pinning gives each
// guest LLC dedicated host CPUs; on a contended host the test
// ResourceContention-skips instead of hard-failing.
#[ktstr_test(
    llcs = 2,
    cores = 4,
    threads = 1,
    memory_mib = 2048,
    performance_mode = true
)]
fn cover_cache_yield_wake_affine(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::performance::custom_cache_yield_wake_affine(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_cache_pipe_io_compute_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::performance::custom_cache_pipe_io_compute_imbalance(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_fan_out_wake(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::performance::custom_fan_out_wake(ctx)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_fan_out_compute(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::performance::custom_fan_out_compute(ctx)
}

// -- watchdog timeout overwrite --

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1, cores = 4, threads = 1, memory_mib = 2048,
    watchdog_timeout_s = 60,
    duration_s = 30,
    max_imbalance_ratio = 10.0,
    fail_on_stall = false,
)]
fn cover_watchdog_long_timeout_survives(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_sched_mixed(ctx)
}

// -- watchdog forced stall (expects scheduler death) --

fn scenario_forced_failure(ctx: &Ctx) -> Result<AssertResult> {
    let mut r = ktstr::scenario::basic::custom_sched_mixed(ctx)?;

    r.record_fail("forced failure for auto-repro test".into());
    Ok(r)
}

fn scenario_yield_heavy(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
    use ktstr::workload::WorkType;
    use std::time::Duration;
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("demo_workers")
                .work_type(WorkType::YieldHeavy)
                .workers(4),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::fixed(Duration::from_secs(8)),
    }];
    execute_steps(ctx, steps)
}

static BPF_CRASH: BpfMapWrite = BpfMapWrite::new(".bss", "crash", 1);

/// scx-error matcher for the two plain-`expect_err` BPF_CRASH negative
/// tests (`neg_crash_after_auto_repro`, `neg_host_crash_auto_repro`).
/// They write `crash=1` via `BPF_CRASH`, which makes `ktstr_dispatch`
/// call `scx_bpf_error("ktstr: host-triggered crash")` — a BPF-side
/// abort (`SCX_EXIT_ERROR_BPF`). Plain `expect_err` alone only proves
/// "the run failed"; pinning the scx-error message proves the failure
/// is a BPF crash, not a hang/panic/OOM/different exit class.
///
/// The dump's `scx_sched_state.exit_kind` is NOT usable here: a crashed
/// scheduler tears down (scx_disable on the abort) before the freeze
/// coordinator walks `*scx_root`, so the dump renders with
/// `scx_walker_unavailable` and `scx_sched_state: null`. The
/// dmesg-backed scx-error matcher is the reliable surface — the same
/// one `neg_expect_scx_bpf_error_contains_e2e` uses. `expect_err: true`
/// is required alongside it (validated in `KtstrTestEntry`).
const CRASH_ASSERT: ktstr::assert::Assert = ktstr::assert::Assert {
    expect_scx_bpf_error_contains: Some("ktstr: host-triggered crash"),
    ..ktstr::assert::Assert::NO_OVERRIDES
};

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_FORCED_STALL: KtstrTestEntry = KtstrTestEntry {
    name: "cover_watchdog_forced_stall",
    func: ktstr::scenario::basic::custom_crash_light,
    topology: TOPO_1L_4C_1T,
    scheduler: &KTSTR_SCHED,
    extra_sched_args: &["--stall-after", "1"],
    watchdog_timeout: std::time::Duration::from_secs(2),
    performance_mode: true,
    expect_err: true,
    ..KtstrTestEntry::DEFAULT
};

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_STALL_DETECT: KtstrTestEntry = KtstrTestEntry {
    name: "neg_stall_detection_scx_exit",
    func: ktstr::scenario::basic::custom_crash_light,
    topology: TOPO_1L_4C_1T,
    scheduler: &KTSTR_SCHED,
    auto_repro: false,
    extra_sched_args: &["--stall-after", "1"],
    watchdog_timeout: std::time::Duration::from_secs(3),
    expect_err: true,
    ..KtstrTestEntry::DEFAULT
};

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_SCHED_DEATH: KtstrTestEntry = KtstrTestEntry {
    name: "neg_sched_death_no_check_result",
    func: ktstr::scenario::basic::custom_crash_light,
    topology: TOPO_1L_4C_1T,
    scheduler: &KTSTR_SCHED,
    extra_sched_args: &["--stall-after", "1"],
    watchdog_timeout: std::time::Duration::from_secs(3),
    duration: std::time::Duration::from_secs(10),
    expect_err: true,
    ..KtstrTestEntry::DEFAULT
};

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_AUTO_REPRO_CHECK: KtstrTestEntry = KtstrTestEntry {
    name: "neg_auto_repro_on_check_failure",
    func: scenario_forced_failure,
    topology: TOPO_1L_4C_1T,
    scheduler: &KTSTR_SCHED,
    expect_err: true,
    ..KtstrTestEntry::DEFAULT
};

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_CRASH_AFTER: KtstrTestEntry = KtstrTestEntry {
    name: "neg_crash_after_auto_repro",
    func: ktstr::scenario::basic::custom_crash_light,
    topology: TOPO_1L_4C_1T,
    scheduler: &KTSTR_SCHED,
    bpf_map_write: &[&BPF_CRASH],
    expect_err: true,
    assert: CRASH_ASSERT,
    ..KtstrTestEntry::DEFAULT
};

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 2048,
    bpf_map_write = BPF_CRASH,
    expect_err = true,
    expect_scx_bpf_error_matches = r"ktstr:\s+host-triggered\s+crash",
)]
fn neg_expect_scx_bpf_error_matches_e2e(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_crash_light(ctx)
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 2048,
    bpf_map_write = BPF_CRASH,
    expect_err = true,
    expect_scx_bpf_error_contains = "ktstr: host-triggered crash",
)]
fn neg_expect_scx_bpf_error_contains_e2e(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_crash_light(ctx)
}

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_HOST_CRASH: KtstrTestEntry = KtstrTestEntry {
    name: "neg_host_crash_auto_repro",
    func: scenario_yield_heavy,
    topology: TOPO_1L_4C_1T,
    scheduler: &KTSTR_SCHED,
    bpf_map_write: &[&BPF_CRASH],
    expect_err: true,
    assert: CRASH_ASSERT,
    ..KtstrTestEntry::DEFAULT
};

// A real host-injected crash drives the auto-repro chain end to end: the
// host write sets `crash=1`, the scheduler's dispatcher calls
// `scx_bpf_error`, the scheduler tears down (scx_disable), and the
// auto-repro VM boots, replays, and lands a shape-valid `.repro.wprof.pb`
// — inverting the verdict to PASS via `expect_auto_repro`. Unlike the
// `pos_expect_auto_repro_*` tests (which force a body `record_fail` while
// the scheduler stays LIVE), this exercises the auto-repro path from an
// actual `scx_bpf_error` scheduler teardown — the chain the deleted phantom
// `demo_*_crash_auto_repro` fixtures never reached (they were `is_ignored`
// and hit `.bss` padding, so their crash never fired). Requires `wprof`:
// the `.repro.wprof.pb` artifact whose presence satisfies `expect_auto_repro`
// is written by the wprof binary in the auto-repro VM.
#[cfg(feature = "wprof")]
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 4,
    threads = 1,
    duration_s = 3,
    watchdog_timeout_s = 60,
    wprof,
    auto_repro = true,
    expect_auto_repro,
    bpf_map_write = BPF_CRASH,
)]
fn bpf_crash_auto_repro_e2e(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_crash_light(ctx)
}

// -- monitor evaluation path with default thresholds --

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1, cores = 4, threads = 1, memory_mib = 2048,
    watchdog_timeout_s = 60,
    max_imbalance_ratio = 20.0,
    sustained_samples = 15,
)]
fn cover_monitor_evaluation(ctx: &Ctx) -> Result<AssertResult> {
    // Exercises the host-side monitor threshold evaluation path in
    // run_ktstr_test_inner. The scenario passes, then the host evaluates
    // monitor samples against merged thresholds.
    ktstr::scenario::basic::custom_sched_mixed(ctx)
}

// -- ops coverage --

#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 2048,
    max_spread_pct = 80.0
)]
fn cover_op_move_all_tasks(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_src").workers(2),
                CgroupDef::named("cg_dst"),
            ],
            HoldSpec::fixed(std::time::Duration::from_secs(2)),
        ),
        Step::new(
            vec![Op::move_all_tasks("cg_src", "cg_dst")],
            HoldSpec::fixed(std::time::Duration::from_secs(3)),
        ),
    ];
    execute_steps(ctx, steps)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_op_spawn_host(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::{CgroupDef, Op, Step, execute_steps};
    use ktstr::workload::{WorkSpec, WorkType};
    let steps = vec![
        Step::with_defs(
            vec![CgroupDef::named("cg_0").workers(2)],
            ctx.settled_hold(1.0),
        )
        .set_ops(vec![Op::spawn_host(
            WorkSpec::default().workers(2).work_type(WorkType::SpinWait),
        )]),
    ];
    execute_steps(ctx, steps)
}

// -- workload coverage --

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_work_type_sequence(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::{CgroupDef, Step, execute_steps};
    use ktstr::workload::{WorkPhase, WorkType};
    use std::time::Duration;
    let seq = WorkType::Sequence {
        first: WorkPhase::Spin(Duration::from_millis(50)),
        rest: vec![
            WorkPhase::Yield(Duration::from_millis(20)),
            WorkPhase::Sleep(Duration::from_millis(30)),
        ],
    };
    let steps = vec![Step::with_defs(
        vec![CgroupDef::named("cg_seq").workers(2).work_type(seq)],
        ctx.settled_hold(1.0),
    )];
    execute_steps(ctx, steps)
}

#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn cover_execute_defs_two_cgroups(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::{CgroupDef, execute_defs};
    use ktstr::workload::WorkType;
    execute_defs(
        ctx,
        vec![
            CgroupDef::named("cg_0")
                .workers(2)
                .work_type(WorkType::SpinWait),
            CgroupDef::named("cg_1")
                .workers(2)
                .work_type(WorkType::SpinWait),
        ],
    )
}
