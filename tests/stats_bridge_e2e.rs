//! End-to-end coverage of the stats-bridge round-trip path.
//!
//! Boots a real guest VM under `scx-ktstr` for 5 s with a single
//! periodic capture so the freeze coordinator's periodic-capture
//! loop fires exactly one boundary. The cgroup holds workers across
//! the entire duration so scx-ktstr's enqueue/dispatch callbacks
//! advance `nr_dispatched` (.bss + scx_stats `KtstrStats` envelope)
//! before the boundary fires.
//!
//! The `post_vm` callback runs on the host after `vm.run()` returns
//! and exercises the full stats-axis path:
//!
//! 1. Periodic boundary fires → freeze coordinator issues a
//!    scx_stats request over the port-2 dedicated channel.
//! 2. scx-ktstr's `Stats` derive answers with a `KtstrStats` JSON
//!    envelope carrying the BSS counter `nr_dispatched`.
//! 3. The relay routes the response back to the host bridge,
//!    coupled with the BPF capture into a single periodic sample.
//! 4. `SampleSeries::stats(...)` projects the JSON axis and the
//!    test asserts `nr_dispatched > 0` at the lone boundary.
//!
//! A non-zero observation proves every leg of the pipeline ran: the
//! relay landed a real envelope on the bridge, the JSON parsed into
//! `serde_json::Value`, the path projection resolved the field, and
//! the scheduler's dispatch path actually advanced.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{SampleSeries, VmResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Drain the bridge's periodic captures and assert the
/// scheduler-stats path delivered a non-zero `nr_dispatched` count
/// at the lone interior boundary (num_snapshots = 1 → midpoint).
/// Proves the port-2 stats relay landed a real `scx_stats` JSON
/// envelope on the host bridge — the response carries the BSS
/// counter that the scheduler advertises via its `Stats` derive.
fn assert_stats_round_trip(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.periodic_target == 1,
        "periodic_target must mirror num_snapshots = 1, got {}",
        result.periodic_target,
    );
    anyhow::ensure!(
        result.periodic_fired >= 1,
        "the lone midpoint capture must have fired at least once \
         under a 5 s workload — periodic_fired = {} of {}",
        result.periodic_fired,
        result.periodic_target,
    );

    let drained = result.snapshot_bridge.drain_ordered_with_stats();
    anyhow::ensure!(
        !drained.is_empty(),
        "drain_ordered_with_stats returned an empty bundle despite \
         periodic_fired = {}",
        result.periodic_fired,
    );
    let series = SampleSeries::from_drained_typed(drained, result.monitor.clone()).periodic_only();
    anyhow::ensure!(
        !series.is_empty(),
        "no periodic-tagged entries on the bridge after the run"
    );

    // .stats() projects the scheduler-stats JSON axis: every sample's
    // stats slot must be present (None would surface as
    // SnapshotError::MissingStats; absence here means the port-2
    // relay never delivered an envelope). The `series.is_empty()`
    // guard above ensures iter_full() yields at least one entry —
    // any Err slot bails immediately, so reaching the post-loop
    // assertion proves every slot was Ok and at least one sample
    // existed.
    let nr_dispatched = series.stats("nr_dispatched", |sv| sv.get("nr_dispatched").as_u64());
    let mut any_progress = false;
    for (tag, _elapsed_ms, slot) in nr_dispatched.iter_full() {
        match slot {
            Ok(v) => {
                if *v > 0 {
                    any_progress = true;
                }
            }
            Err(e) => anyhow::bail!(
                "stats projection for `nr_dispatched` failed at \
                 sample {tag}: {e}"
            ),
        }
    }
    anyhow::ensure!(
        any_progress,
        "scheduler reported nr_dispatched = 0 across every periodic \
         sample — the dispatch path never advanced under the 5 s \
         workload (was scx-ktstr loaded?)"
    );

    // Exact-value fidelity: scx-ktstr's `ktstr_init` stamps the BSS
    // field `stats_magic` with a fixed sentinel (KTSTR_STATS_MAGIC in
    // scx-ktstr/src/bpf/main.bpf.c). Assert the host receives that
    // EXACT value over the relay. The `nr_dispatched > 0` check above
    // would pass on any stray non-zero cardinal; this pins a known
    // 64-bit value end-to-end and fails if the bridge drops,
    // truncates, re-types, or zero-fills the field. The constant
    // mirrors the C #define — they must stay in sync.
    const KTSTR_STATS_MAGIC: u64 = 0x5354_4154_7374_6174; // "STATstat"
    let magic = series.stats("stats_magic", |sv| sv.get("stats_magic").as_u64());
    let mut saw_magic = false;
    for (tag, _elapsed_ms, slot) in magic.iter_full() {
        match slot {
            Ok(v) => {
                anyhow::ensure!(
                    *v == KTSTR_STATS_MAGIC,
                    "stats_magic mismatch at sample {tag}: got {v:#018x}, \
                     expected {KTSTR_STATS_MAGIC:#018x} — the scx_stats \
                     relay did not deliver the emitted value byte-exact"
                );
                saw_magic = true;
            }
            Err(e) => anyhow::bail!(
                "stats projection for `stats_magic` failed at sample \
                 {tag}: {e}"
            ),
        }
    }
    anyhow::ensure!(
        saw_magic,
        "no stats_magic value observed across periodic samples — the \
         relay never delivered the sentinel field"
    );

    // series.monitor() consume-path assertion: the host-side
    // MonitorReport must wire through SampleSeries (via
    // `result.monitor.clone()` at the from_drained call site) and
    // be reachable as a typed view. On a 5 s VM run with scx-ktstr
    // loaded, the monitor loop fires repeatedly so
    // `summary.total_samples` must be non-zero. Catches a
    // regression where the from_drained wire-through becomes a
    // no-op (e.g. monitor field accidentally not propagated, or
    // VmResult.monitor never populated by the freeze coordinator).
    let monitor_view = series.monitor().ok_or_else(|| {
        anyhow::anyhow!(
            "series.monitor() returned None despite VmResult.monitor \
             being populated for this run — the monitor pipeline did \
             not wire through SampleSeries"
        )
    })?;
    anyhow::ensure!(
        monitor_view.summary().total_samples > 0,
        "monitor summary reported total_samples = 0 across the 5 s \
         run — the monitor loop never sampled, or summary aggregation \
         failed"
    );

    // series.host() consume-path assertion: the per-sample
    // per-CPU host snapshot data (FailureDumpReport.per_cpu_time)
    // must wire through SampleSeries and surface via the
    // HostView projector. With num_snapshots = 1 the bridge
    // captures at least one periodic boundary that includes the
    // per-CPU time stats; HostView.cpus() must therefore report
    // at least one captured CPU. Catches the regression where
    // the per_cpu_time slice is unpopulated or the host accessor
    // dropped the rows.
    let host_view = series.host().ok_or_else(|| {
        anyhow::anyhow!(
            "series.host() returned None despite a periodic capture \
             having fired — the per-sample per_cpu_time rows did \
             not wire through SampleSeries"
        )
    })?;
    let captured_cpus = host_view.cpus();
    anyhow::ensure!(
        !captured_cpus.is_empty(),
        "host view reported zero captured CPUs across the 5 s run \
         — the per_cpu_time pipeline never populated rows, or the \
         HostView::cpus discovery dropped them"
    );

    // Closure projector consume-path: exercise
    // per_cpu_field_u64 against EVERY captured CPU. With a 5 s
    // workload holding a cgroup full of busy workers, at least one
    // CPU's cpustat_user_ns must have advanced — at least one
    // sample's Ok-slot value across any captured CPU must be > 0.
    // Catches regressions in the closure-projector loop body
    // (wrong cpu match, wrong tag/elapsed thread-through, wrong
    // slot type) that the construction-only assertion above would
    // miss. Iterating every CPU avoids the brittle-on-1-worker-2-vCPU
    // shape where the worker happens to land entirely on cpu N != 0:
    // any captured CPU advancing proves the pipeline works.
    let advanced_on_any_cpu = captured_cpus.iter().any(|&cpu| {
        host_view
            .per_cpu_field_u64(cpu, "user_ns", |stats| stats.cpustat_user_ns)
            .values_iter()
            .filter_map(|slot| slot.as_ref().ok())
            .any(|v| *v > 0)
    });
    anyhow::ensure!(
        advanced_on_any_cpu,
        "per_cpu_field_u64 cpustat_user_ns reported 0 on every \
         periodic sample across every captured CPU \
         ({captured_cpus:?}) — the workload never ran in user mode \
         on any captured CPU, or the projector dropped the per-sample \
         field"
    );
    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    num_snapshots = 1,
    duration_s = 5,
    watchdog_timeout_s = 15,
    auto_repro = false,
    post_vm = assert_stats_round_trip,
)]
fn stats_bridge_round_trip(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
