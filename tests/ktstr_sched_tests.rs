use anyhow::Result;
use ktstr::WatchdogObservation;
use ktstr::assert::AssertResult;
use ktstr::prelude::{VmResult, post_vm_skip};
use ktstr::scenario::ops::{CgroupDef, CpusetSpec, HoldSpec, Step, execute_steps};
use ktstr::scenario::{Ctx, ScenarioDef};
use ktstr::test_support::{BpfMapWrite, Scheduler, SchedulerSpec};
use ktstr::{ktstr_scenario, ktstr_test};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// The two-cgroup workload, in one place.
///
/// Shared by `sched_basic_proportional` and its wprof-capturing sibling so the
/// traced run and the calibrated run cannot drift apart.
fn sched_basic_proportional_scenario() -> ScenarioDef {
    ScenarioDef::with_defs(vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")])
}

#[ktstr_scenario(scheduler = KTSTR_SCHED, llcs = 1, cores = 2, threads = 1, sustained_samples = 15, watchdog_timeout_s = 15)]
fn sched_basic_proportional() -> ScenarioDef {
    sched_basic_proportional_scenario()
}

/// `sched_basic_proportional` with wprof capture attached.
///
/// Same `ScenarioDef` as the test above, deliberately built by calling it
/// rather than restating it — a copy would drift, and the whole point is that
/// the traced workload IS the calibrated one.
///
/// # Why this is a separate test rather than `wprof` on the original
///
/// `wprof` requires the `wprof` cargo feature: the macro rejects the attribute
/// at parse time when the feature is off, so adding it to
/// `sched_basic_proportional` would make the default build of this test file
/// fail to compile. A `cfg`-gated sibling keeps `cargo test` working for
/// everyone who does not have the feature (and does not want a build that
/// clones and compiles wprof from GitHub) while still producing the trace.
///
/// Capture is not failure-only: a PASSING run writes
/// `{sidecar_dir}/{test_name}-{variant_hash:016x}.wprof.pb`, next to the stats
/// JSON that the scx-sim calibration already consumes.
#[cfg(feature = "wprof")]
#[ktstr_scenario(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    sustained_samples = 15,
    watchdog_timeout_s = 15,
    wprof,
    // The DEFAULT capture is `-d 500`, i.e. 500 ms, and guest init spawns the
    // tracer at boot — so a default-args run captures the guest booting and
    // stops ~11.5 s before the workload starts. Measured: a default run
    // produced a 0.495 s trace containing init/swapper/rcu/kworker and NOT ONE
    // workload task. Diffing that against a 12 s simulation would have
    // "worked" and been meaningless.
    //
    // 15 s spans boot plus the whole 12 s hold with margin, while still
    // finishing inside the ~19 s test so the trace is shipped host-side before
    // teardown. wprof_args REPLACES the defaults rather than appending, so the
    // ringbuf flags are restated at their default values
    // (WPROF_DEFAULT_RINGBUF_SIZE_KB = 16384, WPROF_DEFAULT_RINGBUF_CNT = 1).
    // Sizing check: 0.495 s of boot — the busiest phase — produced 36 KB, so
    // 15 s of mostly-steady-state spinning stays far inside the 16 MiB arena.
    //
    // DO NOT ADD `--kthread --idle`. Their help text reads "Allow kernel
    // tasks" / "Allow idle tasks", which sounds purely additive. It is not:
    // measured on this exact scenario, adding them DROPPED THE USERSPACE
    // WORKLOAD ENTIRELY — 24.0 s of `init` on-CPU time vanished, the trace
    // went 587 KB -> 189 KB and 10777 -> 3034 packets, and kthread time went
    // DOWN (4.080 ms -> 2.745 ms) rather than up. Whatever the mechanism, the
    // flags change which tasks are traced rather than widening the set, and
    // the configuration below is the one that contains the workload.
    wprof_args = "-d 15000 -e sched --ringbuf-size=16384 --ringbuf-cnt=1"
)]
fn sched_basic_proportional_wprof() -> ScenarioDef {
    sched_basic_proportional_scenario()
}

#[ktstr_scenario(scheduler = KTSTR_SCHED, llcs = 1, cores = 4, threads = 1, sustained_samples = 15, watchdog_timeout_s = 15, max_spread_pct = 80.0)]
fn sched_cpuset_split() -> ScenarioDef {
    ScenarioDef::with_defs(vec![
        CgroupDef::named("cg_0").cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
        CgroupDef::named("cg_1").cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
    ])
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

#[ktstr_scenario(scheduler = KTSTR_SCHED, llcs = 1, cores = 2, threads = 1, sustained_samples = 15, watchdog_timeout_s = 15)]
fn sched_dynamic_add() -> ScenarioDef {
    ScenarioDef::new(vec![
        Step::with_defs(vec![CgroupDef::named("cg_0")], HoldSpec::frac(0.5)),
        Step::with_defs(vec![CgroupDef::named("cg_1")], HoldSpec::frac(0.5)),
    ])
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
#[ktstr_scenario(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    performance_mode = true,
    duration_s = 3,
    sustained_samples = 15,
    watchdog_timeout_s = 15,
)]
fn sched_perf_positive() -> ScenarioDef {
    let checks = ktstr::assert::Assert::default_checks()
        .min_iteration_rate(5000.0)
        .max_gap_ms(500);
    ScenarioDef::with_defs(vec![CgroupDef::named("cg_0")]).set_checks(checks)
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
/// tests live in `tests/parse_stall_duration_test.rs`.
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
// `tests/parse_stall_duration_test.rs`, for file organisation only.
//
// This note used to claim they COULD NOT live here, because a
// binary registering `KtstrTestEntry` entries filters plain `#[test]`
// functions out of the runner's view. That is no longer true:
// `ktstr_main`'s `--list` interception falls through to libtest and
// `list_plain_tests` re-emits the plain tests (dropping only the
// per-entry wrappers, matched by name). Verified against this
// binary's own `NEXTEST=1 --list` output — the scenario-extraction
// tests at the end of this file are plain `#[test]` functions and
// they do run.

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
#[ktstr_scenario(scheduler = KTSTR_SCHED, llcs = 1, cores = 2, threads = 1, duration_s = 2, watchdog_timeout_s = 15, max_spread_pct = 80.0)]
fn sched_verifier_stats_populated() -> ScenarioDef {
    ScenarioDef::with_defs(vec![CgroupDef::named("cg_0")])
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

// ---------------------------------------------------------------------
// Scenario-extraction checks for the `#[ktstr_scenario]` tests above.
//
// These are plain `#[test]` functions, and they DO run: `ktstr_main`
// intercepts `--list` only to add its own ktstr/gauntlet names, then
// falls through so libtest lists these too (`list_plain_tests`, which
// filters out the per-entry wrappers by name). Verified against this
// binary's nextest listing.
//
// They belong in this file rather than a fixture binary of their own
// because linkme slices are per-binary: the only way to assert on what
// the ported tests registered is to be linked alongside them. They
// need no VM — that is the property under test.
// ---------------------------------------------------------------------

/// The names ported to `#[ktstr_scenario]` in this file. Listed
/// explicitly rather than derived from the registry so that deleting a
/// port (or silently losing its registration) fails here instead of
/// shrinking the checked population to nothing.
const PORTED_SCENARIOS: &[&str] = &[
    "sched_basic_proportional",
    "sched_cpuset_split",
    "sched_dynamic_add",
    "sched_verifier_stats_populated",
    "sched_perf_positive",
];

/// Every ported test registers an extractable scenario whose builder
/// runs on the host. This is the property the port exists to create;
/// without it the DSL is just a different way to spell the same
/// opaque body.
#[test]
fn ported_tests_register_extractable_scenarios() {
    for name in PORTED_SCENARIOS {
        let entry = ktstr::test_support::find_scenario(name).unwrap_or_else(|| {
            let all: Vec<&str> = ktstr::test_support::KTSTR_SCENARIOS
                .iter()
                .map(|e| e.name)
                .collect();
            panic!("{name} did not register a scenario; registered: {all:?}")
        });
        let def = (entry.build)();
        assert!(
            !def.steps().is_empty(),
            "{name} extracted an empty scenario",
        );
        assert!(
            def.is_declarative(),
            "{name} is not statically inspectable — it contains a \
             Setup::Factory step",
        );
    }
}

/// Each scenario must also be a registered test, and every registered
/// scenario in this binary must be one of the ports above. The second
/// direction is what catches a future port that lands without being
/// added to `PORTED_SCENARIOS` and therefore never gets its shape
/// checked below.
#[test]
fn scenario_registry_matches_the_ported_set() {
    let mut registered: Vec<&str> = ktstr::test_support::KTSTR_SCENARIOS
        .iter()
        .map(|e| e.name)
        .collect();
    registered.sort_unstable();
    let mut expected: Vec<&str> = PORTED_SCENARIOS.to_vec();
    expected.sort_unstable();
    assert_eq!(
        registered, expected,
        "the scenario registry must hold exactly the ported tests",
    );

    let test_names: Vec<&str> = ktstr::test_support::KTSTR_TESTS
        .iter()
        .map(|e| e.name)
        .collect();
    for name in &registered {
        assert!(
            test_names.contains(name),
            "scenario {name:?} has no KtstrTestEntry — the expansion \
             registered one slice but not the other",
        );
    }
}

/// The extracted values must be the workloads the bodies declared,
/// asserted against independently written expectations rather than
/// against the bodies themselves. A `ScenarioDef` constructor that
/// quietly dropped a cpuset, collapsed two steps into one, or lost a
/// per-step hold would pass a round-trip check and fail this one.
#[test]
fn extracted_scenarios_have_the_declared_shape() {
    let build = |name: &str| {
        (ktstr::test_support::find_scenario(name)
            .unwrap_or_else(|| panic!("{name} is registered"))
            .build)()
    };
    let cgroup_names = |def: &ktstr::scenario::ScenarioDef| -> Vec<Vec<String>> {
        def.steps()
            .iter()
            .map(|step| match &step.setup {
                ktstr::scenario::ops::Setup::Defs(defs) => {
                    defs.iter().map(|d| d.name.to_string()).collect()
                }
                ktstr::scenario::ops::Setup::Factory(_) => {
                    panic!("ported scenarios use static def lists")
                }
            })
            .collect()
    };

    // Two cgroups, one step, whole run.
    let basic = build("sched_basic_proportional");
    assert_eq!(cgroup_names(&basic), vec![vec!["cg_0", "cg_1"]]);
    assert!(basic.checks().is_none());

    // Disjoint cpuset halves survive extraction.
    let split = build("sched_cpuset_split");
    assert_eq!(cgroup_names(&split), vec![vec!["cg_0", "cg_1"]]);
    let ktstr::scenario::ops::Setup::Defs(defs) = &split.steps()[0].setup else {
        unreachable!("checked above")
    };
    let cpusets: Vec<String> = defs.iter().map(|d| format!("{:?}", d.cpuset)).collect();
    assert!(
        cpusets[0].contains("index: 0") && cpusets[1].contains("index: 1"),
        "each cgroup keeps its own disjoint half: {cpusets:?}",
    );

    // Two steps, added in order, each holding half the run.
    let dynamic = build("sched_dynamic_add");
    assert_eq!(cgroup_names(&dynamic), vec![vec!["cg_0"], vec!["cg_1"]]);
    for step in dynamic.steps() {
        assert!(
            matches!(step.hold, HoldSpec::Frac(f) if (f - 0.5).abs() < f64::EPSILON),
            "each step holds half the run, got {:?}",
            step.hold,
        );
    }

    // The one port that overrides ctx.assert: its gates must survive
    // into the extracted value, or the ported test silently runs
    // ungated.
    let perf = build("sched_perf_positive");
    let checks = perf
        .checks()
        .expect("sched_perf_positive declares an Assert override");
    assert_eq!(checks.min_iteration_rate, Some(5000.0));
    assert_eq!(checks.max_gap_ms, Some(500));
}

/// Dump every registered scenario as a `SourceScenario`-shaped JSON record,
/// for the simulator backend to consume.
///
/// Writes to `$KTSTR_SCENARIO_EXPORT_DIR` and does nothing when that is unset,
/// so a normal test run is unaffected. The records are derived from the REAL
/// registry — `KTSTR_SCENARIOS` for the workload, the paired `KtstrTestEntry`
/// for the topology and duration — not hand-written, which is the only reason
/// a simulator run from one of them can be called "the same scenario".
///
/// `workers_per_cgroup` is 1: `Ctx::builder` defaults it to 1 and nothing on
/// the ktstr-test path overrides it, matching the `num_workers: 1` per cgroup
/// observed in a real VM run (`sched_basic_proportional` on 6.14.11).
///
/// THIS IS NOT CHECKED, AND AN EARLIER VERSION OF THIS COMMENT CLAIMED IT WAS.
/// It said the value "is asserted below against the value observed in a real VM
/// run". No such assertion existed or exists: the only one here is the export
/// count, and this body never mentions `workers_per_cgroup`. A comment
/// describing a check nobody wrote is worse than no comment, because it stops
/// the next reader from looking.
///
/// WHY IT MATTERS. Once a scenario uses `CgroupDef::named` rather than
/// `ctx.cgroup_def` — which is what the hand conversion at 85c72e11 did — the
/// two backends resolve the worker count from DIFFERENT PLACES. The VM resolves
/// an unset `num_workers` through
/// `resolve_num_workers(work, ctx.workers_per_cgroup, ..)`, following `Ctx`.
/// The record carries `workers: null`, and the simulator binds it to this
/// literal. They agree at 1 and diverge at anything else, silently — the
/// cross-backend check compares CPU *shares*, so two runs at different worker
/// counts can still agree.
///
/// WHY THERE IS STILL NO ASSERTION HERE. Reading `Ctx`'s default requires
/// constructing one, and `TestTopology::synthetic` is `#[cfg(test)]`, so it is
/// unreachable from an integration test. The honest fix is to stop hardcoding:
/// have the exporter carry the resolved count instead of inheriting a literal.
/// Tracked in the dev-harness audit
/// `ai_docs/KTSTR_CONVERSION_AUDIT_20260813.md`; deliberately not bodged here.
#[test]
fn export_registered_scenarios() {
    // One shared constant rather than a copy per binary -- see its doc comment
    // for the cross-backend coupling that makes duplicating it a hazard.
    let workers = ktstr::test_support::DEFAULT_WORKERS_PER_CGROUP;

    let Some(out) = ktstr::test_support::export_registered_scenarios(workers) else {
        return; // KTSTR_SCENARIO_EXPORT_DIR unset: a normal run writes nothing.
    };
    assert_eq!(
        out.written.len(),
        PORTED_SCENARIOS.len(),
        "every ported scenario in THIS binary must export; wrote {:?}",
        out.written,
    );
}

/// No scenario may be declared in a test binary that never exports.
///
/// THIS IS THE GUARD FOR THE DEFECT, and it is worth more than the refactor it
/// accompanies. `KTSTR_SCENARIOS` is a linkme distributed slice, which is
/// per-link-unit: each `tests/*.rs` is its own binary with its own slice. The
/// exporter used to live in this file and iterate this binary's slice, so a
/// scenario declared in any OTHER test file was invisible to it — the
/// conversion compiled, its own tests passed, and no record was ever written.
///
/// It refused SILENTLY. That is the failure mode: not a wrong record, an absent
/// one, with nothing anywhere reporting the absence. Eight conversions were
/// planned against candidates that all live in other binaries before anyone
/// noticed.
///
/// This check reads the test sources rather than the registry, and it has to.
/// The property is "this binary calls the exporter", and a binary that does not
/// call it cannot report that — the missing call is exactly the code that isn't
/// there to run. The source tree is the only place the absence is visible.
#[test]
fn every_scenario_binary_exports() {
    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let missing = ktstr::test_support::scenario_binaries_missing_export(&tests_dir);
    assert!(
        missing.is_empty(),
        "these test binaries declare a #[ktstr_scenario] but never call \
         ktstr::test_support::export_registered_scenarios, so their scenarios \
         are silently absent from every export and never reach a second \
         backend: {missing:?}. Add an export test to each -- see \
         export_registered_scenarios in this file for the three-line shape.",
    );
}
