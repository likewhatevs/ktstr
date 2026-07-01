//! Fork-mode WorkType coverage gauntlet.
//!
//! Boots one small VM with no sched_ext scheduler attached (EEVDF is
//! the in-kernel default) and drives every `CloneMode::Fork`-compatible
//! [`WorkType`] variant sequentially through the production
//! [`WorkloadHandle`] spawn/start/stop pipeline. Each arm spawns the
//! variant, lets it run a short window, collects the per-worker
//! [`WorkerReport`]s, and asserts liveness: the spawn succeeded, at
//! least one report came back, and (for variants that do measurable
//! per-iteration work) the workers recorded non-zero
//! `work_units + iterations`.
//!
//! This is the coverage lever for `src/workload/worker/mod.rs`: the
//! per-WorkType dispatch arms there are only exercised when a scenario
//! actually selects the variant, and most variants had no driving test.
//! Driving every arm here pins that each variant's worker body
//! dispatches and produces work rather than silently no-op'ing or
//! panicking. Liveness — not a pathology-specific threshold — is the
//! meaningful cross-variant invariant: "the arm ran and did work."
//!
//! Variants requiring same-CPU pinning to manifest their pathology
//! ([`WorkType::PriorityInversion`], [`WorkType::RtStarvation`],
//! [`WorkType::PreemptStorm`]) use `AffinityIntent::exact([0])` rather
//! than `AffinityIntent::SingleCpu`: `WorkloadHandle::spawn` rejects the
//! topology-aware `SingleCpu` intent (it requires scenario context),
//! while `Exact` resolves to a fixed CPU set at spawn time and produces
//! the same single-CPU placement.
//!
//! Every arm is asserted live: each does
//! unconditional per-iteration work (a `spin_burst` and/or an
//! `iterations` bump — TimerLatency bumps only `iterations`)
//! regardless of host topology — including
//! [`WorkType::CrossAffinityChurn`], whose cgroup-sibling affinity
//! toggle no-ops on this direct-spawn path (no dedicated `CgroupDef`)
//! but is a side effect, not a liveness gate.
//!
//! Excluded variants (covered elsewhere or incompatible with this
//! Fork direct-spawn path):
//!   - [`WorkType::Custom`] — bypasses built-in instrumentation by
//!     contract; covered by the closure-driving tests in
//!     `src/workload/spawn/`.
//!   - [`WorkType::WakeChain`] — covered by
//!     `tests/worktype_eevdf_validation.rs`.
//!   - [`WorkType::EpollStorm`] / [`WorkType::CgroupChurn`] — Thread-only
//!     and Fork-only respectively; driven by the sibling
//!     `tests/worktype_coverage_thread_gauntlet_e2e.rs`.
//!   - [`WorkType::CgroupAttachStorm`] — Fork-compatible, but needs a
//!     sibling `dest` cgroup (created via `Op::add_cgroup`) that this
//!     direct-spawn path does not provide; here it would no-op (the
//!     dest is unresolvable). Driven by the dedicated
//!     `tests/cgroup_attach_storm_e2e.rs`.
//!   - [`WorkType::NetTraffic`] — Fork-compatible, but needs an attached
//!     NIC (`#[ktstr_test(networks = [...])]`) that this direct-spawn path
//!     does not provide; here it would no-op (no non-loopback interface).
//!     Driven by the dedicated `tests/net_traffic_e2e.rs`.
//!   - [`WorkType::IrqWake`] — like `NetTraffic`, needs an attached NIC the
//!     direct-spawn path does not provide (here it would no-op); also a
//!     paired sender/receiver (group of 2). Driven by the dedicated
//!     `tests/irq_wake_e2e.rs`.
//!   - [`WorkType::Schbench`] — driven by its own schbench engine (a
//!     message-thread + worker-thread pool, not the generic
//!     per-iteration worker body); covered by
//!     `tests/performance_mode_e2e.rs`.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::workload::{
    AffinityIntent, AluWidth, FutexLockMode, SchedClass, WorkPhase, WorkType, WorkloadConfig,
    WorkloadHandle,
};
use std::time::Duration;

/// Spawn `cfg`, run it briefly, collect reports, and record liveness
/// failures into `result`: the spawn must succeed, at least one report
/// must come back, and the workers must record non-zero
/// `work_units + iterations` (every gauntlet arm does measurable
/// per-iteration work).
fn run_arm(label: &str, cfg: WorkloadConfig, result: &mut AssertResult) {
    let mut handle = match WorkloadHandle::spawn(&cfg) {
        Ok(h) => h,
        Err(e) => {
            result.record_fail(AssertDetail::new(
                DetailKind::Other,
                format!("{label}: spawn failed: {e:#}"),
            ));
            return;
        }
    };
    handle.start();
    std::thread::sleep(Duration::from_millis(500));
    let reports = handle.stop_and_collect();

    if reports.is_empty() {
        result.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!("{label}: zero reports — spawn or collection broken"),
        ));
        return;
    }

    let units: u64 = reports.iter().map(|r| r.work_units + r.iterations).sum();
    if units == 0 {
        result.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "{label}: no work_units/iterations across {} workers — \
                 the dispatch arm produced no measurable work",
                reports.len(),
            ),
        ));
    }
}

#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 1024,
    max_spread_pct = 80.0,
    duration_s = 30,
    watchdog_timeout_s = 90
)]
fn worktype_fork_gauntlet_covers_all_arms(_ctx: &Ctx) -> Result<AssertResult> {
    let mut result = AssertResult::pass();

    // Helper: a config with the given work type, worker count, and
    // (default-Inherit) affinity, with everything else defaulted.
    let cfg = |wt: WorkType, nw: usize| WorkloadConfig {
        num_workers: nw,
        work_type: wt,
        ..Default::default()
    };

    run_arm("SpinWait", cfg(WorkType::SpinWait, 2), &mut result);
    run_arm("YieldHeavy", cfg(WorkType::YieldHeavy, 2), &mut result);
    run_arm("Mixed", cfg(WorkType::Mixed, 2), &mut result);
    run_arm(
        "Bursty",
        cfg(
            WorkType::Bursty {
                burst_duration: Duration::from_millis(20),
                sleep_duration: Duration::from_millis(20),
            },
            2,
        ),
        &mut result,
    );
    run_arm("IoSyncWrite", cfg(WorkType::IoSyncWrite, 2), &mut result);
    run_arm("IoRandRead", cfg(WorkType::IoRandRead, 2), &mut result);
    run_arm("IoConvoy", cfg(WorkType::IoConvoy, 2), &mut result);
    run_arm(
        "PipeIo",
        cfg(WorkType::PipeIo { burst_iters: 64 }, 2),
        &mut result,
    );
    run_arm(
        "FutexPingPong",
        cfg(WorkType::FutexPingPong { spin_iters: 256 }, 2),
        &mut result,
    );
    run_arm(
        "CachePressure",
        cfg(
            WorkType::CachePressure {
                size_kib: 256,
                stride: 64,
            },
            2,
        ),
        &mut result,
    );
    run_arm(
        "CacheYield",
        cfg(
            WorkType::CacheYield {
                size_kib: 256,
                stride: 64,
            },
            2,
        ),
        &mut result,
    );
    run_arm(
        "CachePipe",
        cfg(
            WorkType::CachePipe {
                size_kib: 256,
                burst_iters: 64,
            },
            2,
        ),
        &mut result,
    );
    run_arm(
        "FutexFanOut",
        cfg(
            WorkType::FutexFanOut {
                fan_out: 3,
                spin_iters: 128,
            },
            4,
        ),
        &mut result,
    );
    run_arm(
        "Sequence",
        cfg(
            WorkType::Sequence {
                first: WorkPhase::Spin(Duration::from_millis(10)),
                rest: vec![
                    WorkPhase::Sleep(Duration::from_millis(10)),
                    WorkPhase::Yield(Duration::from_millis(10)),
                    WorkPhase::Io(Duration::from_millis(10)),
                    WorkPhase::alu_hot(AluWidth::Widest, Duration::from_millis(10)),
                ],
            },
            2,
        ),
        &mut result,
    );
    run_arm("ForkExit", cfg(WorkType::ForkExit, 2), &mut result);
    run_arm("NiceSweep", cfg(WorkType::NiceSweep, 2), &mut result);
    run_arm(
        "AffinityChurn",
        cfg(WorkType::AffinityChurn { spin_iters: 128 }, 2),
        &mut result,
    );
    run_arm(
        "PolicyChurn",
        cfg(WorkType::PolicyChurn { spin_iters: 128 }, 2),
        &mut result,
    );
    run_arm(
        "FanOutCompute",
        cfg(
            WorkType::FanOutCompute {
                fan_out: 3,
                cache_footprint_kib: 256,
                operations: 8,
                sleep_usec: 50,
            },
            4,
        ),
        &mut result,
    );
    run_arm(
        "PageFaultChurn",
        cfg(
            WorkType::PageFaultChurn {
                region_kib: 1024,
                touches_per_cycle: 64,
                spin_iters: 64,
            },
            2,
        ),
        &mut result,
    );
    run_arm(
        "MutexContention",
        cfg(
            WorkType::MutexContention {
                contenders: 4,
                hold_iters: 128,
                work_iters: 128,
            },
            4,
        ),
        &mut result,
    );
    run_arm(
        "ThunderingHerd",
        cfg(
            WorkType::ThunderingHerd {
                waiters: 3,
                batches: 50,
                inter_batch_ms: 5,
            },
            4,
        ),
        &mut result,
    );
    run_arm(
        "PriorityInversion",
        WorkloadConfig {
            num_workers: 3,
            work_type: WorkType::PriorityInversion {
                high_count: 1,
                medium_count: 1,
                low_count: 1,
                hold_iters: 128,
                work_iters: 128,
                pi_mode: FutexLockMode::Pi,
            },
            affinity: AffinityIntent::exact([0]),
            ..Default::default()
        },
        &mut result,
    );
    run_arm(
        "ProducerConsumerImbalance",
        cfg(
            WorkType::ProducerConsumerImbalance {
                producers: 2,
                consumers: 2,
                produce_rate_hz: 1000,
                consume_iters: 256,
                queue_depth_target: 64,
            },
            4,
        ),
        &mut result,
    );
    run_arm(
        "RtStarvation",
        WorkloadConfig {
            num_workers: 2,
            work_type: WorkType::RtStarvation {
                rt_workers: 1,
                cfs_workers: 1,
                rt_priority: 1,
                burst_iters: 128,
            },
            affinity: AffinityIntent::exact([0]),
            ..Default::default()
        },
        &mut result,
    );
    run_arm(
        "AsymmetricWaker",
        cfg(
            WorkType::AsymmetricWaker {
                waker_class: SchedClass::Cfs,
                wakee_class: SchedClass::Cfs,
                burst_iters: 128,
            },
            2,
        ),
        &mut result,
    );
    run_arm(
        "SignalStorm",
        cfg(
            WorkType::SignalStorm {
                signals_per_iter: 8,
                work_iters: 128,
            },
            2,
        ),
        &mut result,
    );
    run_arm(
        "PreemptStorm",
        WorkloadConfig {
            num_workers: 2,
            work_type: WorkType::PreemptStorm {
                cfs_workers: 1,
                rt_burst_iters: 128,
                rt_sleep_us: 1000,
            },
            affinity: AffinityIntent::exact([0]),
            ..Default::default()
        },
        &mut result,
    );
    run_arm(
        "NumaWorkingSetSweep",
        cfg(
            WorkType::NumaWorkingSetSweep {
                region_kib: 512,
                sweep_period_ms: 20,
                target_nodes: vec![0],
            },
            2,
        ),
        &mut result,
    );
    run_arm(
        "NumaMigrationChurn",
        cfg(WorkType::NumaMigrationChurn { period_ms: 20 }, 2),
        &mut result,
    );
    run_arm(
        "IdleChurn",
        cfg(
            WorkType::IdleChurn {
                burst_duration: Duration::from_millis(2),
                sleep_duration: Duration::from_millis(5),
                precise_timing: true,
            },
            2,
        ),
        &mut result,
    );
    run_arm(
        "TimerLatency",
        cfg(WorkType::TimerLatency { interval_us: 1000 }, 2),
        &mut result,
    );
    run_arm(
        "AluHot",
        cfg(
            WorkType::AluHot {
                width: AluWidth::Widest,
            },
            2,
        ),
        &mut result,
    );
    run_arm(
        "SmtSiblingSpin",
        cfg(WorkType::SmtSiblingSpin, 2),
        &mut result,
    );
    run_arm(
        "IpcVariance",
        cfg(
            WorkType::IpcVariance {
                hot_iters: 128,
                cold_iters: 64,
                period_iters: 8,
            },
            2,
        ),
        &mut result,
    );
    // CrossAffinityChurn does an unconditional spin_burst (work_units)
    // plus an iterations bump every loop (worker/mod.rs:1376,1405); the
    // cgroup-sibling affinity toggle is a side effect that no-ops here
    // (no dedicated CgroupDef on this direct-spawn path), not a liveness
    // gate — so it records work regardless.
    run_arm(
        "CrossAffinityChurn",
        cfg(WorkType::CrossAffinityChurn { spin_iters: 128 }, 2),
        &mut result,
    );

    Ok(result)
}
