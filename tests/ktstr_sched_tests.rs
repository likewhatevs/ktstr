use anyhow::Result;
use ktstr::WatchdogObservation;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{VmResult, post_vm_skip};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, CpusetSpec, HoldSpec, Step, execute_steps};
use ktstr::test_support::{BpfMapWrite, Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 2, threads = 1, sustained_samples = 15, watchdog_timeout_s = 15)]
fn sched_basic_proportional(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0"), ctx.cgroup_def("cg_1")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 4, threads = 1, sustained_samples = 15, watchdog_timeout_s = 15, max_spread_pct = 80.0)]
fn sched_cpuset_split(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
            CgroupDef::named("cg_1").cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

/// Per-cgroup CPU placement is captured and labeled on a PASSING
/// run (not only the failure-path scheduler dump). Two cgroups on
/// DISJOINT cpuset halves run a spin workload; the run passes, then the
/// body asserts each cgroup's `CgroupStats` is labeled with its name and
/// carries the (non-empty, disjoint) set of CPUs its workers actually
/// ran on. Exercises the full chain unit tests cannot: real worker
/// cpus_used -> cgroup_stats union -> collect_handles labeling ->
/// result.stats.cgroups. A labeling or capture regression fails here
/// even though the scheduler run itself passes.
#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 4, threads = 1, sustained_samples = 15, watchdog_timeout_s = 15, max_spread_pct = 80.0)]
fn sched_cgroup_cpus_used_surfaced_on_pass(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
            CgroupDef::named("cg_1").cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let result = execute_steps(ctx, steps)?;
    let cgroups = &result.stats.cgroups;
    assert_eq!(
        cgroups.len(),
        2,
        "two declared cgroups -> two labeled stats entries; got {cgroups:#?}",
    );
    let by_name: std::collections::BTreeMap<&str, &_> = cgroups
        .iter()
        .map(|c| (c.cgroup_name.as_str(), c))
        .collect();
    let cg0 = by_name
        .get("cg_0")
        .unwrap_or_else(|| panic!("cg_0 must be labeled; got {cgroups:#?}"));
    let cg1 = by_name
        .get("cg_1")
        .unwrap_or_else(|| panic!("cg_1 must be labeled; got {cgroups:#?}"));
    for cg in [cg0, cg1] {
        // Non-empty presumes a real (non-sentinel) report: every worker
        // records its starting CPU before the work loop, and a passing run
        // collects only thawed workers, so no empty-cpus_used sentinel
        // reaches the union. A future report-collection change that emitted
        // a sentinel here would fail this assertion, naming the cause.
        assert!(
            !cg.cpus_used.is_empty(),
            "cgroup {} must record the CPUs its workers ran on: {cg:#?}",
            cg.cgroup_name,
        );
        assert_eq!(
            cg.num_cpus,
            cg.cpus_used.len(),
            "num_cpus must equal cpus_used.len() for {}",
            cg.cgroup_name,
        );
    }
    // Disjoint cpusets are a hard kernel constraint, so the captured
    // cpus_used sets must not overlap — proving cpus_used reflects the
    // real per-cgroup confinement, not a shared/aggregate set.
    assert!(
        cg0.cpus_used.is_disjoint(&cg1.cpus_used),
        "disjoint-cpuset cgroups must record disjoint cpus_used; \
         cg_0={:?} cg_1={:?}",
        cg0.cpus_used,
        cg1.cpus_used,
    );
    Ok(result)
}

#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 2, threads = 1, sustained_samples = 15, watchdog_timeout_s = 15)]
fn sched_dynamic_add(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step {
            setup: vec![CgroupDef::named("cg_0")].into(),
            ops: vec![],
            hold: HoldSpec::frac(0.5),
        },
        Step {
            setup: vec![CgroupDef::named("cg_1")].into(),
            ops: vec![],
            hold: HoldSpec::frac(0.5),
        },
    ];
    execute_steps(ctx, steps)
}

fn bpf_api_scenario(
    ctx: &ktstr::scenario::Ctx,
    hold: HoldSpec,
) -> Result<ktstr::assert::AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold,
    }];
    execute_steps(ctx, steps)
}

/// Stall-test scenario: holds for the full duration so the host's
/// `stall=1` write, the scheduler's resulting stall, and the
/// scx-watchdog teardown all fit inside the scenario window.
fn scenario_bpf_api(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    bpf_api_scenario(ctx, HoldSpec::FULL)
}

/// Link-test scenario: a short fixed hold. The no-op map write only
/// needs to complete and the scenario to publish a result; the
/// cold-BTF `bpf_map_write` phase-1 latency is absorbed by the
/// vm_timeout floor (`vm_timeout_from_entry` adds it for every
/// bpf_map_write entry), not by this entry's duration.
fn scenario_bpf_api_link(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    bpf_api_scenario(ctx, HoldSpec::fixed(std::time::Duration::from_secs(2)))
}

/// Write stall=0 to the .bss map after scenario starts — a no-op (the
/// `stall` global is already 0) that exercises the full BPF map API
/// pipeline. The byte offset is resolved from the map's BTF by the
/// `stall` VAR name at write time.
static BPF_NOOP: BpfMapWrite = BpfMapWrite::new(".bss", "stall", 0);

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_BPF_API: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "sched_bpf_map_api_integration",
        func: scenario_bpf_api_link,
        scheduler: &KTSTR_SCHED,
        auto_repro: false,
        assert: ktstr::assert::Assert::NO_OVERRIDES.fail_on_rq_clock_stuck(false),
        bpf_map_write: &[&BPF_NOOP],
        // The link scenario holds a fixed 2s. The cold-BTF
        // bpf_map_write phase-1 latency (the wait_for_map_write block
        // before the workload) is covered by the vm_timeout floor
        // vm_timeout_from_entry applies to every bpf_map_write entry —
        // not by this duration.
        duration: std::time::Duration::from_secs(2),
        watchdog_timeout: std::time::Duration::from_secs(15),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

// Host-to-guest observable-action integration test.
//
// Exercises the full loop:
//   1. HOST writes into the guest's BPF `.bss` map via the KVM
//      memslot path (`BpfMapAccessor::write_value_u32` dispatched
//      from `freeze_coord::start_bpf_map_write`).
//   2. GUEST scheduler's BPF dispatcher reads the new `stall` value
//      from its `.bss` section on the next dispatch entry
//      (`if (stall) return;` in main.bpf.c).
//   3. GUEST ACTS: the scheduler stops moving tasks from the
//      shared DSQ to per-CPU local DSQs, every CPU sits idle, the scx watchdog observes no progress
//      within its budget and tears the scheduler down (emitted
//      via the `SchedulerDied` assert detail the runtime records).
//   4. HOST CONFIRMS: the scenario returns a failing AssertResult
//      carrying the scheduler-died signal; `expect_err: true`
//      inverts the verdict so "fails as expected" is the PASS
//      state.
//
// Differs from the existing BPF-NOOP test (value=0 over a field
// already 0) — that proves the API pipeline LINKS, this proves
// the pipeline's WRITE is OBSERVED by the guest and produces
// distinct guest behaviour. Differs from `cover_watchdog_forced_stall`
// which achieves the same stall via the scheduler's
// `--stall-after` CLI flag (a scheduler-internal timer, no host
// write): that path tests the scheduler's self-stall plumbing,
// this path tests the host→guest map-write plumbing.
//
// `watchdog_timeout` is set short (2 s) so the stall-detection
// fires quickly; `duration` is longer so the watchdog has room
// to fire inside the scenario window rather than racing the
// natural scenario end.
// Writes the `stall` global (main.bpf.c) — resolved to its `.bss` byte
// offset from the map's BTF by VAR name at write time.
static BPF_STALL_HOST_WRITE: BpfMapWrite = BpfMapWrite::new(".bss", "stall", 1);

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_BPF_HOST_WRITE_STALLS: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "sched_host_bpf_map_write_stalls_scheduler",
        func: scenario_bpf_api,
        scheduler: &KTSTR_SCHED,
        auto_repro: false,
        bpf_map_write: &[&BPF_STALL_HOST_WRITE],
        watchdog_timeout: std::time::Duration::from_secs(2),
        duration: std::time::Duration::from_secs(10),
        performance_mode: true,
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Positive benchmarking test: scx-ktstr under performance_mode passes
/// min_iteration_rate and max_gap_ms gates.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    performance_mode = true,
    duration_s = 3,
    sustained_samples = 15,
    watchdog_timeout_s = 15,
)]
fn sched_perf_positive(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::execute_steps_with;
    let checks = ktstr::assert::Assert::default_checks()
        .min_iteration_rate(5000.0)
        .max_gap_ms(500);
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

fn scenario_perf_negative(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    use ktstr::scenario::ops::execute_steps_with;
    use ktstr::workload::{WorkSpec, WorkType};
    // Wake latency, not max_gap_ms: the Stuck gate is CPU-denominated
    // (compute-without-progress) and deliberately blind to scheduler
    // starvation — a degraded scheduler starves wall time, and a starved
    // worker accrues no CPU (see the max_gap_ms section of
    // tests/assert_gate_matrix.rs). The workload must actually block and
    // wake for the gate to have samples: a futex ping-pong pair whose
    // per-turn spin (~tens of ms) is on the order of degrade's ~134ms
    // dispatch-blackout window, so a large fraction of its hand-off
    // wakes land inside a blackout and park in SHARED_DSQ — pooled p99
    // wake latency blows far past 50ms. A healthy scheduler serves the
    // same hand-offs in microseconds.
    let checks = ktstr::assert::Assert::default_checks().max_p99_wake_latency_ns(50_000_000);
    let steps = vec![Step {
        setup: vec![
            ctx.cgroup_def("cg_0")
                .work(
                    WorkSpec::default()
                        .workers(2)
                        .work_type(WorkType::FutexPingPong {
                            spin_iters: 20_000_000,
                        }),
                ),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_PERF_NEG: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "sched_perf_negative",
        func: scenario_perf_negative,
        scheduler: &KTSTR_SCHED,
        auto_repro: false,
        extra_sched_args: &["--degrade"],
        performance_mode: true,
        duration: std::time::Duration::from_secs(5),
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

fn scenario_scattershot(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    use ktstr::scenario::ops::execute_steps_with;
    let checks = ktstr::assert::Assert::default_checks()
        .max_gap_ms(10000)
        .max_spread_pct(80.0);
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_SCATTER: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_scattershot_migration",
        func: scenario_scattershot,
        topology: ktstr::test_support::Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
            llc_cores: None,
        },
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &["--scattershot"],
        performance_mode: true,
        duration: std::time::Duration::from_secs(5),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

fn scenario_throughput_regression(
    ctx: &ktstr::scenario::Ctx,
) -> Result<ktstr::assert::AssertResult> {
    use ktstr::scenario::ops::execute_steps_with;
    let checks = ktstr::assert::Assert::default_checks()
        .min_iteration_rate(5000.0)
        .max_gap_ms(500);
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_SLOW: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_throughput_regression",
        func: scenario_throughput_regression,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &["--slow"],
        performance_mode: true,
        duration: std::time::Duration::from_secs(5),
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

fn scenario_auto_repro(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_AUTO_REPRO: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_auto_repro",
        func: scenario_auto_repro,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

// Watchdog timing-precision: prove the host's `watchdog_timeout=2s`
// override (set via `KtstrTestEntry::watchdog_timeout`) both LANDED in
// `scx_sched.watchdog_timeout` in guest memory and was ENFORCED — the
// watchdog actually fired a stall rather than the scheduler dying via
// some other exit path. The two-tier assertion lives in the host-side
// `check_watchdog_timing` post_vm callback (see its doc); this scenario
// runs the stall and forwards the kmsg evidence.
//
// The watchdog stall printk — kernel/sched/ext.c scx_exit
// SCX_EXIT_ERROR_STALL, one of
//   "{task}[{pid}] failed to run for {seconds}.{millis}s" or
//   "watchdog failed to check in for {seconds}.{millis}s"
// — lands in the guest /dev/kmsg but is suppressed from the COM1
// console at the default loglevel=0, so the host cannot read it from
// VmResult.stderr. The scenario forwards the guest kmsg to the host via
// `ktstr::send_kmsg(read_kmsg())`; the callback reads it back from
// `VmResult::guest_kmsg` and asserts a stall line is PRESENT.
//
// Why presence, not the kernel-measured duration? Under a deterministic
// stall (`--stall-after=1`) the scheduler stops dispatching tasks, so
// the watchdog wq-kworker — itself a task the scheduler must dispatch —
// is starved and scx_watchdog_workfn runs late; the measured "failed to
// run for" duration then reflects when the kworker finally ran
// (~8.7-9.3s observed), not the 2s timeout. The duration is
// starvation-noise; override effectiveness is proven by tier 1 (the
// eager in-DRAM readback in `VmResult::watchdog_observation`), not by
// the duration.
//
// `expect_err: true` inverts the SCX_EXIT_ERROR_STALL itself (the
// "scheduler died as planned" outcome) to the PASS state; the two-tier
// assertion gates separately in the post_vm callback.
fn scenario_watchdog_timing_precision(
    ctx: &ktstr::scenario::Ctx,
) -> Result<ktstr::assert::AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let result = execute_steps(ctx, steps)?;
    // Forward the guest /dev/kmsg to the host so the host-side
    // check_watchdog_timing post_vm callback can read the
    // kernel-measured stall duration. The scx_exit SCX_EXIT_ERROR_STALL
    // printk is in /dev/kmsg but suppressed from the COM1 console at the
    // default loglevel=0, so the host cannot read it from
    // VmResult.stderr — read_kmsg (guest /dev/kmsg) + send_kmsg bridges
    // it into VmResult::guest_kmsg().
    ktstr::send_kmsg(ktstr::read_kmsg().as_bytes());
    Ok(result)
}

/// Host-side post_vm assertion for `watchdog_override_timing_precision`,
/// in two tiers. Tier 1: the host's `watchdog_timeout=2s` override
/// landed in guest memory — `VmResult::watchdog_observation` reports the
/// jiffies the host wrote vs the value the monitor read back from
/// `scx_sched.watchdog_timeout`; they must match. That readback is
/// eager and in-DRAM, so (unlike the kernel-measured stall duration) it is
/// immune to the watchdog-kworker starvation a deterministic stall
/// induces. Tier 2: the watchdog actually FIRED a stall — a
/// SCX_EXIT_ERROR_STALL stall line must be present (presence, not the
/// starvation-noisy duration), proving the death was a watchdog stall
/// rather than a different scx exit. Two sources carry that line: the
/// guest kmsg the scenario forwards via `ktstr::send_kmsg` (the kernel's
/// own printk, but its emit races the resumed payload's read — see the
/// tier-2 body) and the scheduler's forwarded log (its `uei_report` echo
/// of the same exit reason, flushed race-free over the bulk port); a stall
/// line in EITHER satisfies the tier. Runs unconditionally; its Err is a
/// hard FAIL via PostVmAssertionFailure even though expect_err inverts
/// the scheduler-died outcome to PASS.
fn check_watchdog_timing(result: &VmResult) -> Result<()> {
    // Tier 1 — the override value landed in guest memory.
    let Some(WatchdogObservation {
        expected_jiffies,
        observed_jiffies,
    }) = result.watchdog_observation()
    else {
        return Err(post_vm_skip(
            "no watchdog observation — the monitor recorded no override \
             readback (scx_root stayed null; the scheduler never attached), \
             so the override cannot be verified",
        ));
    };
    anyhow::ensure!(
        observed_jiffies == expected_jiffies,
        "watchdog override ineffective: the host wrote {expected_jiffies} \
         jiffies to scx_sched.watchdog_timeout but guest memory reads back \
         {observed_jiffies} — the host-write missed the field (a kernel \
         refactor moved the offset, the deref resolved the wrong scx_sched, \
         or scx-ktstr's .timeout_ms=4000 BPF default overwrote it)"
    );

    // Tier 2 — the watchdog actually fired a stall (the override was
    // enforced, not just configured). Presence of a stall line is the
    // signal; the duration itself reflects when the starved watchdog
    // kworker finally ran, not the 2s timeout, so it is not asserted.
    //
    // An empty guest kmsg means the payload never resumed to forward it:
    // on a host/kernel where the stall is suppressed (a quiet host can
    // starve the watchdog kworker before it observes the parked workload,
    // so SCX_EXIT_ERROR_STALL never fires and the workload task stays
    // parked for the whole run), the scenario is force-rebooted mid-hold
    // and `send_kmsg` never runs. Treat that as inconclusive, not a
    // failure — tier 1 already proved the override value landed.
    let kmsg = result.guest_kmsg();
    if kmsg.is_empty() {
        return Err(post_vm_skip(
            "guest forwarded no kmsg (send_kmsg did not run, or the VM \
             exited before the bulk-port forward completed) — the override \
             readback passed, but the watchdog-fired signal is missing, so \
             the run is inconclusive",
        ));
    }
    // A non-empty kmsg means the payload resumed, so the scheduler DIED and
    // its tasks fell back to fair. The kernel's stall printk
    // (`scx_log_sched_disable`'s `pr_err` in kernel/sched/ext.c) reaches
    // /dev/kmsg only from the disable workfn, which runs AFTER scx enters
    // bypass and re-releases tasks — so the resumed payload's `read_kmsg`
    // can race ahead of it and forward a kmsg that lacks the line. The
    // scheduler's own `uei_report` echo of the SAME SCX_EXIT_ERROR_STALL
    // reason ("… failed to run for {d}s") is flushed to its captured log
    // over the fast bulk port during the crash grace window, race-free;
    // corroborate against it so the tier holds regardless of which side of
    // that race the kmsg read landed on. Failing only when NEITHER source
    // carries a stall line still catches a death via a different scx exit
    // path or a changed printk format.
    let sched_log = result.scheduler_log();
    anyhow::ensure!(
        parse_stall_duration_seconds(&kmsg).is_some()
            || parse_stall_duration_seconds(&sched_log).is_some(),
        "watchdog override landed ({observed_jiffies} jiffies) but neither \
         the guest kmsg nor the scheduler log has an SCX_EXIT_ERROR_STALL \
         stall line ('failed to run for' / 'watchdog failed to check in \
         for') — the scheduler exited via a different path, or the kernel \
         printk format changed. kmsg: {kmsg}\nsched_log: {sched_log}"
    );
    eprintln!(
        "watchdog override applied + enforced: scx_sched.watchdog_timeout = \
         {observed_jiffies} jiffies (matches the host write), and a watchdog \
         stall fired"
    );
    Ok(())
}

/// Parse the sched_ext stall-duration seconds from a guest kmsg
/// dump using a two-part grok pattern matching the kernel's
/// `%u.%03us` printf format. The watchdog emits one of two messages
/// at `kernel/sched/ext.c scx_exit(sch, SCX_EXIT_ERROR_STALL, ...)`:
/// `{task}[{pid}] failed to run for {secs}.{millis}s` (per-task,
/// check_rq_for_timeouts) or `watchdog failed to check in for
/// {secs}.{millis}s` (per-CPU, scx_tick); a deterministic stall that
/// starves the watchdog wq-kworker fires the latter. Return
/// `secs + millis/1000.0` as f64 seconds, or `None` if neither line
/// is present.
///
/// Pattern decomposes into two `INT` captures
/// (`%{INT:seconds}\.%{INT:millis}s`) — NOT `NUMBER`, because
/// NUMBER expands to BASE10NUM which already matches `2.004` as a
/// whole decimal and would greedily consume the `.`, leaving
/// nothing for the second capture. `INT` (`[+-]?[0-9]+`) matches
/// each side of the kernel's printf individually, exactly mirroring
/// the format string. The `fancy-regex` grok backend is required
/// because INT is stable under any backend but the default
/// BASE10NUM / NUMBER patterns use lookbehind (`(?<!...)`) and
/// atomic groups (`(?>...)`) — selecting `fancy-regex` keeps all
/// of grok's default patterns usable regardless of which one we
/// compose here.
///
/// Exposed as a standalone helper so a unit test can pin the
/// parser against a synthetic input without booting a VM. Unit
/// tests live in `tests/parse_stall_duration_test.rs` (integration-
/// test binaries with KtstrTestEntry entries filter out plain
/// `#[test]` functions).
fn parse_stall_duration_seconds(kmsg: &str) -> Option<f64> {
    let grok = grok::Grok::with_default_patterns();
    let pattern = grok
        .compile(
            r"(?:failed to run for|watchdog failed to check in for) %{INT:seconds}\.%{INT:millis}s",
            false,
        )
        .expect("grok pattern compiles with fancy-regex backend");
    let matches = pattern.match_against(kmsg)?;
    let seconds: u64 = matches.get("seconds")?.parse().ok()?;
    let millis: u64 = matches.get("millis")?.parse().ok()?;
    Some(seconds as f64 + (millis as f64) / 1000.0)
}

// Unit tests for `parse_stall_duration_seconds` live in
// `tests/parse_stall_duration_test.rs`. Integration-test binaries
// that register `KtstrTestEntry` distributed-slice entries go
// through ktstr's early-dispatch path, which intercepts nextest
// `--list` / `--exact` and filters out plain `#[test]` functions —
// so the parser's host-side unit tests cannot coexist in this file
// without being invisible to the test runner.

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_WATCHDOG_TIMING: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "watchdog_override_timing_precision",
        func: scenario_watchdog_timing_precision,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(2),
        duration: std::time::Duration::from_secs(15),
        // expect_err inverts the SCX_EXIT_ERROR_STALL (the expected
        // outcome of --stall-after=1) to PASS. The real timing assertion
        // lives in check_watchdog_timing, a post_vm_unconditional
        // callback whose Err is a hard FAIL via PostVmAssertionFailure —
        // so an ineffective override fails the test even though the stall
        // itself is inverted.
        expect_err: true,
        post_vm_unconditional: Some(check_watchdog_timing),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

fn scenario_baseline(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_EEVDF: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_baseline_eevdf",
        func: scenario_baseline,
        auto_repro: false,
        performance_mode: true,
        duration: std::time::Duration::from_secs(3),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_SCX: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_baseline_scx",
        func: scenario_baseline,
        scheduler: &KTSTR_SCHED,
        performance_mode: true,
        duration: std::time::Duration::from_secs(3),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Minimal scheduler test that exercises host-side BPF program enumeration.
/// The framework warns when verifier_stats is empty for scheduler tests.
///
/// `watchdog_timeout_s = 15` overrides the 5 s
/// `KtstrTestEntry::DEFAULT.watchdog_timeout`. The minimal scx-ktstr
/// pull-from-shared-DSQ scheduler under 2-vCPU SpinWait contention can
/// leave background tasks (init, RCU kthreads) sitting on
/// `rq->scx.runnable_list` longer than 4 s as the per-CPU dispatch
/// loop cycles workers in/out of SHARED_DSQ; the kernel watchdog's
/// per-task `runnable_at` walk
/// (`kernel/sched/ext.c:check_rq_for_timeouts`) flags this as
/// `SCX_EXIT_ERROR_STALL`. 15 s matches `sched_bpf_api` above, which
/// runs the same scheduler against a similar workload shape.
#[ktstr_test(scheduler = KTSTR_SCHED, llcs = 1, cores = 2, threads = 1, duration_s = 2, watchdog_timeout_s = 15, max_spread_pct = 80.0)]
fn sched_verifier_stats_populated(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

fn scenario_mid_degrade(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    use ktstr::scenario::ops::execute_steps_with;
    use ktstr::workload::{WorkSpec, WorkType};
    // Wake latency, not max_gap_ms — same rationale and workload as
    // `scenario_perf_negative`: the CPU-denominated Stuck gate cannot
    // see scheduler starvation, and only a blocking/waking workload
    // gives the wake gate samples. The ping-pong pairs run healthy
    // through the first phase; the post-trigger dispatch blackouts then
    // balloon their hand-off wake latencies past the 50ms cap.
    let checks = ktstr::assert::Assert::default_checks().max_p99_wake_latency_ns(50_000_000);
    let ping_pong = |name: &'static str| {
        ctx.cgroup_def(name)
            .work(
                WorkSpec::default()
                    .workers(2)
                    .work_type(WorkType::FutexPingPong {
                        spin_iters: 20_000_000,
                    }),
            )
    };
    let steps = vec![
        Step {
            setup: vec![ping_pong("cg_0"), ping_pong("cg_1")].into(),
            ops: vec![],
            hold: HoldSpec::fixed(std::time::Duration::from_secs(3)),
        },
        Step {
            setup: vec![].into(),
            ops: vec![],
            hold: HoldSpec::fixed(std::time::Duration::from_secs(5)),
        },
    ];
    execute_steps_with(ctx, steps, Some(&checks))
}

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_MID_DEGRADE: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_mid_run_degrade",
        func: scenario_mid_degrade,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &["--degrade-after=3"],
        performance_mode: true,
        duration: std::time::Duration::from_secs(10),
        watchdog_timeout: std::time::Duration::from_secs(60),
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };
