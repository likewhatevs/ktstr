# Compare a Scheduler vs EEVDF

A standard regression guard for a sched_ext scheduler: does it match (or
beat) the kernel default (EEVDF) on the same workload — not just for
throughput, but for latency and CPU overhead too? Run the workload under
the scheduler in one phase, [detach the scheduler](../concepts/ops.md)
mid-run so the kernel default takes over for a second phase, then compare
the two phases metric by metric.

The workload must **persist** across the detach — a `Backdrop` population,
not per-step workers — so its cumulative counters span both phases. That
shared, continuous measurement is what makes a per-phase delta meaningful
(per-step workers reset each phase and read ~0).

Two readers cover the comparison, both on the `&VmResult` a `post_vm`
callback receives (the host-side hook that runs after the VM exits):

- `VmResult::throughput_ratio(a, b)` — iterations/sec from the stimulus
  timeline. The timeline carries per-step boundaries independent of the
  periodic capture pipeline, so throughput works even for
  `--cell-parent-cgroup` schedulers.
- `VmResult::phase_metric(phase, name)` — any other per-phase metric by
  its registry name (see [Checking](../concepts/checking.md)): CPU
  overhead (`system_time_ns`, `user_time_ns`) and scheduling quality
  (`avg_imbalance_ratio`, `avg_dsq_depth`). Wake-latency and run-delay
  distributions are run-level — pooled across cgroups into one whole-run
  value — so they cannot be split into the scheduler phase vs the EEVDF
  phase; to compare them, run the scheduler and EEVDF as two separate
  tests and read each run's run-level metric. (The built-in
  [`Schbench` workload](../concepts/work-types.md#schbench) is the
  exception: it measures its own wakeup latency internally and emits
  `wakeup_p99_latency_us` per phase.) Everything else flows
  through the one per-phase bucket pipeline, so a new metric becomes
  comparable here the moment it lands in that pipeline.

```rust,ignore
use anyhow::{ensure, Result};
use ktstr::assert::{AssertResult, Phase};
use ktstr::ktstr_test;
use ktstr::prelude::{Backdrop, VmResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{execute_scenario, CgroupDef, HoldSpec, Op, Step};
use ktstr::test_support::{Scheduler, SchedulerSpec};

// Built directly rather than via declare_scheduler! so this comparison
// harness stays out of the verifier sweep (manual consts are not
// registered for sweeping). Use declare_scheduler! for the scheduler
// definition you ship.
const MY_SCHED: Scheduler =
    Scheduler::named("my_sched").binary(SchedulerSpec::Discover("scx_my_sched"));

// Runs on the host after the VM exits; the &VmResult carries the stimulus
// timeline and the per-phase metric buckets the comparison reads.
fn compare_vs_eevdf(result: &VmResult) -> Result<()> {
    let sched = Phase::step(0); // first Step ran under the scheduler under test
    let eevdf = Phase::step(1); // second Step ran under EEVDF, after the detach

    // Throughput: > 1.0 means the scheduler out-throughputs EEVDF; < 1.0
    // is a regression.
    let throughput = result
        .throughput_ratio(sched, eevdf)
        .ok_or_else(|| anyhow::anyhow!("no per-phase throughput — did both phases run?"))?;
    ensure!(
        throughput >= 0.8,
        "my_sched throughput is {throughput:.2}x EEVDF (below the 0.8x floor)"
    );

    // Scheduling quality: any per-phase metric compares the same way via
    // phase_metric. Skip the gate when a phase has no reading (None)
    // rather than failing. (Wake-latency / run-delay distributions are
    // run-level and not readable here — see the reader list above.)
    if let (Some(s), Some(e)) = (
        result.phase_metric(sched, "avg_imbalance_ratio"),
        result.phase_metric(eevdf, "avg_imbalance_ratio"),
    ) {
        ensure!(s <= e * 1.5, "my_sched imbalance {s:.2} is >1.5x EEVDF {e:.2}");
    }

    // CPU overhead: per-phase kernel (system) CPU time.
    if let (Some(s), Some(e)) = (
        result.phase_metric(sched, "system_time_ns"),
        result.phase_metric(eevdf, "system_time_ns"),
    ) {
        ensure!(s <= e * 2.0, "my_sched system time {s:.0}ns is >2x EEVDF {e:.0}ns");
    }

    Ok(())
}

#[ktstr_test(
    scheduler = MY_SCHED,
    duration_s = 10,
    watchdog_timeout_s = 10,
    post_vm = compare_vs_eevdf,
)]
fn scheduler_vs_eevdf(ctx: &Ctx) -> Result<AssertResult> {
    // Persistent Backdrop population: runs across both phases so its
    // cumulative counters span the detach.
    let backdrop = Backdrop::new().push_cgroup(CgroupDef::named("cg").workers(4));
    let steps = vec![
        // Phase A: workload under the scheduler under test.
        Step::new(vec![], HoldSpec::frac(0.5)),
        // Phase B: detach -> the kernel default (EEVDF) takes over.
        Step::new(vec![Op::detach_scheduler()], HoldSpec::frac(0.5)),
    ];
    execute_scenario(ctx, backdrop, steps)
}
```

The `0.8x` / `1.5x` / `2.0x` bounds above are illustrative, not
recommendations. Calibrate yours: run the test a few times with
generous bounds, note the observed ratios (each `ensure!` message
prints them; a run's failure output leads with whichever message
tripped), and set each floor just outside the observed noise band.
A gate inside the noise band fails honest runs; one far outside it
never fails at all.

Notes:

- `Op::detach_scheduler()` cleanly hands the workload to the kernel default.
  Each step emits its own boundary, so no trailing closer step is needed,
  and the intentional detach is not promoted to a scheduler-died failure.
- Phases are keyed by `Phase`: `Phase::step(0)` is the first scenario Step,
  `Phase::step(1)` the second. `Phase::BASELINE` is the pre-Step settle
  window. Use `Phase` rather than the raw stimulus `step_index`.
- `phase_metric` returns `None` when a phase has no reading for a metric,
  so gate inside `if let (Some(..), Some(..))` rather than unwrapping —
  a metric that did not populate skips its gate instead of failing the run.
- For cross-cell **balance** rather than a phase-vs-phase comparison, read
  `result.stats.cgroup_balance_ratio()` in the test body (the test body's
  `AssertResult` carries `stats`).

This test gates scheduler-vs-EEVDF *within one run*. To gate your
scheduler against its own past self *across commits*, use
[`cargo ktstr perf-delta`](ab-compare.md) — the two nets catch
different regressions, and CI wants both.

<div class="kt-figure"><svg width="700" height="150" viewBox="0 0 700 150" role="img" aria-label="Two-phase timeline: the same workers run under the scheduler in phase 0, the scheduler detaches mid-run, EEVDF takes over for phase 1, and per-phase metrics compare across the boundary">
  <path d="M20 96 L 688 96" stroke="var(--fg)" stroke-width="1.4"/>
  <path d="M681 92 L 688 96 L 681 100" fill="none" stroke="var(--fg)" stroke-width="1.4"/>
  <rect x="40" y="30" width="290" height="44" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.3"/>
  <text x="60" y="57" font-size="12.5" font-weight="700" fill="var(--kt-accent)">Phase 0 · your scheduler</text>
  <rect x="366" y="30" width="290" height="44" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.3"/>
  <text x="386" y="57" font-size="12.5" font-weight="700" fill="var(--fg)" opacity=".8">Phase 1 · EEVDF</text>
  <path d="M348 24 L 348 104" stroke="var(--fg)" stroke-width="1.2" stroke-dasharray="4 4"/>
  <text x="248" y="20" font-size="10.5" fill="var(--fg)" font-family="var(--mono-font)">Op::detach_scheduler()</text>
  <g font-size="10" fill="var(--fg)" opacity=".75">
    <text x="40" y="120">same cumulative workers across the boundary — counters never reset,</text>
    <text x="40" y="136">so phase_metric(0) vs phase_metric(1) is a like-for-like delta</text>
  </g>
</svg></div>
