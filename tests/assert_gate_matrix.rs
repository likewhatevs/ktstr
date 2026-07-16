use anyhow::Result;
use ktstr::assert::{Assert, AssertResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps_with};
use ktstr::test_support::{KtstrTestEntry, Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

fn scenario_with_checks(ctx: &Ctx, checks: &Assert) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(checks))
}

// Macro emits a module-scope distributed_slice entry. Each test gets a
// scenario function that captures its Assert checks, and a static
// KtstrTestEntry registered in KTSTR_TESTS.
//
// perf: performance_mode value (true/false)
// negative: when true, passes --degrade and expects failure

macro_rules! gate_test {
    ($name:ident, perf: $perf:expr, negative: $neg:expr, $checks:expr) => {
        mod $name {
            use super::*;
            pub(super) fn scenario(ctx: &Ctx) -> Result<AssertResult> {
                let checks = $checks;
                scenario_with_checks(ctx, &checks)
            }
        }

        #[allow(non_upper_case_globals)]
        #[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
        #[linkme(crate = ktstr::linkme)]
        static $name: KtstrTestEntry = KtstrTestEntry {
            name: stringify!($name),
            func: $name::scenario,
            scheduler: &KTSTR_SCHED,
            auto_repro: false,
            performance_mode: $perf,
            extra_sched_args: if $neg { &["--degrade"] } else { &[] },
            expect_err: $neg,
            duration: std::time::Duration::from_secs(5),
            ..KtstrTestEntry::DEFAULT
        };
    };
}

// ===========================================================================
// max_p99_wake_latency_ns
// ===========================================================================

gate_test!(demo_gate_p99_wake_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_p99_wake_latency_ns(100_000_000));
gate_test!(demo_gate_p99_wake_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_p99_wake_latency_ns(1));
gate_test!(demo_gate_p99_wake_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_p99_wake_latency_ns(100_000_000));
gate_test!(demo_gate_p99_wake_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_p99_wake_latency_ns(1));

// ===========================================================================
// max_wake_latency_cv
// ===========================================================================

gate_test!(demo_gate_wake_cv_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_wake_latency_cv(100.0));
gate_test!(demo_gate_wake_cv_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_wake_latency_cv(0.0001));
gate_test!(demo_gate_wake_cv_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_wake_latency_cv(100.0));
gate_test!(demo_gate_wake_cv_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_wake_latency_cv(0.0001));

// ===========================================================================
// min_iteration_rate
// ===========================================================================

gate_test!(demo_gate_iter_rate_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().min_iteration_rate(1.0));
gate_test!(demo_gate_iter_rate_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().min_iteration_rate(1_000_000_000.0));
gate_test!(demo_gate_iter_rate_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().min_iteration_rate(1.0));
gate_test!(demo_gate_iter_rate_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().min_iteration_rate(1_000_000_000.0));

// ===========================================================================
// max_gap_ms
//
// The gap is worker CPU-TIME between progress checkpoints
// (compute-without-progress — see `WorkerReport::max_gap_ms`), so a DEGRADED
// SCHEDULER cannot trip it: scheduler degradation starves wall time, and a
// starved worker accrues no CPU (blocked/starved-stuck is the cell
// watchdog's domain; the wall gap survives as `max_gap_wall_ms` evidence).
// The positive arms keep the standard workload (its per-checkpoint CPU is
// microseconds — far under any sane ceiling). The negative arms run a
// Custom worker that genuinely burns ~200ms of measured thread CPU between
// its progress checkpoints and reports the measured gap — a real
// compute-without-progress pathology (no scheduler flag can produce one),
// proving the in-VM measure→report→gate→verdict chain end to end.
// ===========================================================================

gate_test!(demo_gate_gap_ms_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_gap_ms(10_000));
gate_test!(demo_gate_gap_ms_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_gap_ms(10_000));

/// Custom worker for the gap-negative arms: burns ~200ms of REAL thread CPU
/// (`CLOCK_THREAD_CPUTIME_ID`-measured spin) between two of its own progress
/// checkpoints and reports that measured CPU gap — the compute-without-
/// progress shape the CPU-denominated Stuck gate exists to catch.
fn cpu_burn_gap_worker(ctx: &ktstr::workload::WorkerCtx) -> ktstr::workload::WorkerReport {
    fn thread_cpu_ns() -> u64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: clock_gettime writes a timespec through the out-pointer.
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        assert_eq!(rc, 0, "clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
        (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
    }
    let wall_start = std::time::Instant::now();
    let cpu_start = thread_cpu_ns();
    // Checkpoint 1 (work recorded), then burn 200ms of CPU with NO further
    // progress, then checkpoint 2. The burn is real spin — under host/guest
    // preemption it takes longer in wall time but the CPU gap is invariant.
    let mut sink = 0u64;
    let mut cpu_gap_ns;
    loop {
        for _ in 0..4096 {
            sink = std::hint::black_box(sink.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
        }
        cpu_gap_ns = thread_cpu_ns().saturating_sub(cpu_start);
        if cpu_gap_ns >= 200_000_000 || ctx.stop().load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
    }
    let cpu_end = thread_cpu_ns();
    let wall_ns = wall_start.elapsed().as_nanos() as u64;
    ktstr::workload::WorkerReport {
        tid: unsafe { libc::syscall(libc::SYS_gettid) as i32 },
        work_units: 2048,
        iterations: 2,
        cpu_time_ns: cpu_end.saturating_sub(cpu_start),
        wall_time_ns: wall_ns,
        // The genuinely measured CPU-time gap between the two checkpoints,
        // plus the same-window wall gap as evidence.
        max_gap_ms: cpu_gap_ns / 1_000_000,
        max_gap_wall_ms: wall_ns / 1_000_000,
        completed: true,
        ..Default::default()
    }
}

macro_rules! gap_negative_test {
    ($name:ident, perf: $perf:expr) => {
        mod $name {
            use super::*;
            pub(super) fn scenario(ctx: &Ctx) -> Result<AssertResult> {
                let mut handle =
                    ktstr::workload::WorkloadHandle::spawn(&ktstr::workload::WorkloadConfig {
                        num_workers: 1,
                        work_type: ktstr::workload::WorkType::custom(
                            "cpu_burn_gap",
                            cpu_burn_gap_worker,
                        ),
                        ..Default::default()
                    })?;
                handle.start();
                std::thread::sleep(ctx.duration);
                let reports = handle.stop_and_collect();
                Ok(Assert::default_checks()
                    .max_gap_ms(50)
                    .assert_cgroup(&reports, None))
            }
        }

        #[allow(non_upper_case_globals)]
        #[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
        #[linkme(crate = ktstr::linkme)]
        static $name: KtstrTestEntry = KtstrTestEntry {
            name: stringify!($name),
            func: $name::scenario,
            scheduler: &KTSTR_SCHED,
            auto_repro: false,
            performance_mode: $perf,
            expect_err: true,
            duration: std::time::Duration::from_secs(5),
            ..KtstrTestEntry::DEFAULT
        };
    };
}

gap_negative_test!(demo_gate_gap_ms_perf_on_negative, perf: true);
gap_negative_test!(demo_gate_gap_ms_perf_off_negative, perf: false);

// ===========================================================================
// max_spread_pct
// ===========================================================================

gate_test!(demo_gate_spread_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_spread_pct(99.0));
gate_test!(demo_gate_spread_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_spread_pct(0.01));
gate_test!(demo_gate_spread_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_spread_pct(99.0));
gate_test!(demo_gate_spread_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_spread_pct(0.01));

// ===========================================================================
// max_throughput_cv
// ===========================================================================

gate_test!(demo_gate_throughput_cv_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_throughput_cv(100.0));
gate_test!(demo_gate_throughput_cv_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_throughput_cv(0.0001));
gate_test!(demo_gate_throughput_cv_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_throughput_cv(100.0));
gate_test!(demo_gate_throughput_cv_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_throughput_cv(0.0001));

// ===========================================================================
// min_work_rate
// ===========================================================================

gate_test!(demo_gate_work_rate_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().min_work_rate(1.0));
gate_test!(demo_gate_work_rate_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().min_work_rate(1_000_000_000_000.0));
gate_test!(demo_gate_work_rate_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().min_work_rate(1.0));
gate_test!(demo_gate_work_rate_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().min_work_rate(1_000_000_000_000.0));

// ===========================================================================
// max_migration_ratio
// ===========================================================================

gate_test!(demo_gate_migration_ratio_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_migration_ratio(100.0));
gate_test!(demo_gate_migration_ratio_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_migration_ratio(0.0));
gate_test!(demo_gate_migration_ratio_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_migration_ratio(100.0));
gate_test!(demo_gate_migration_ratio_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_migration_ratio(0.0));

// ===========================================================================
// max_cross_node_migration_ratio
// ===========================================================================

gate_test!(demo_gate_xnode_migration_ratio_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_cross_node_migration_ratio(1.0));
gate_test!(demo_gate_xnode_migration_ratio_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_cross_node_migration_ratio(0.0));
gate_test!(demo_gate_xnode_migration_ratio_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_cross_node_migration_ratio(1.0));
gate_test!(demo_gate_xnode_migration_ratio_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_cross_node_migration_ratio(0.0));

// ===========================================================================
// max_slow_tier_ratio
// ===========================================================================

gate_test!(demo_gate_slow_tier_ratio_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_slow_tier_ratio(1.0));
gate_test!(demo_gate_slow_tier_ratio_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_slow_tier_ratio(0.0));
gate_test!(demo_gate_slow_tier_ratio_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_slow_tier_ratio(1.0));
gate_test!(demo_gate_slow_tier_ratio_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_slow_tier_ratio(0.0));
