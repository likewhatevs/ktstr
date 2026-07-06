# Periodic Capture

A single snapshot proves state was right once; scheduler bugs are
usually about how state *evolves* — a counter that stops advancing,
utilization that drifts after warmup. **Periodic capture** samples
guest BPF state on a cadence across the workload window, driven
entirely by the host: no scenario-code changes, no capture calls in
the test body. The result is a time-ordered series of samples that
feeds the [temporal assertion](temporal-assertions.md) patterns.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Cadence</strong><p><code>num_snapshots</code> places samples inside the stable middle of the workload.</p></div>
<div class="kt-doc-card"><strong>Bridge</strong><p>The host captures BPF state and stats without adding guest-side capture code.</p></div>
<div class="kt-doc-card"><strong>Series</strong><p>Post-VM code drains samples into <code>SampleSeries</code> for temporal checks.</p></div>
</div>

## Enabling it

Set `num_snapshots = N` on the test; `0` (the default) disables
periodic capture entirely.

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(num_snapshots = 3, duration_s = 10)]
fn paced_capture(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("workers").workers(2).work_type(WorkType::SpinWait),
    ])
}
```

## When boundaries fire

The window is the **10%–90% slice** of the workload duration,
anchored at the moment the scenario actually starts — VM boot and
BPF verifier time do not eat the budget. The 10% buffers at each
end keep samples off ramp-up and ramp-down transients. The
remaining 80% divides into `N + 1` equal intervals, yielding `N`
interior boundaries at `0.1·d + (i+1)·0.8·d/(N+1)`. For a 10 s
workload, `num_snapshots = 3` captures at scenario start +
{3 s, 5 s, 7 s}.

The boundary clock is workload time, not wall-clock: a scenario
pause shifts every un-fired boundary by the pause duration.

Two validation rules, enforced when the entry is built:

- **Minimum spacing** — `0.8 · duration / (N + 1) >= 100 ms`.
  Boundaries closer than that would fire back-to-back with no
  workload progress between them. Reduce `num_snapshots` or extend
  `duration_s`.
- **Bridge cap** — `num_snapshots` cannot exceed 64
  (`MAX_STORED_SNAPSHOTS`). Validation rejects higher values rather
  than silently evicting the earliest samples.

## What a capture costs

Each boundary runs the same pipeline as an on-demand
`Op::capture_snapshot`: every vCPU is parked, the BPF maps are
walked, the report is stored. On a healthy guest the freeze is tens
of milliseconds (10–100 ms steady state; cold-cache or large
guest-memory walks push higher). The host watchdog deadline is
extended by each freeze's duration, so periodic captures do not eat
the workload's wall-clock budget — but they do briefly stop the
guest, which is why the spacing floor exists.

## Tags and best-effort delivery

Each capture lands on the host `SnapshotBridge` under
`periodic_NNN` (`periodic_000`, `periodic_001`, …), coexisting with
on-demand and watchpoint tags on the same bridge — filter with
`SampleSeries::periodic_only()` before asserting.

Delivery is best-effort: an early VM exit, rendezvous timeout, or
watchdog deadline can cut the sequence short, and the run loop
abandons the remainder after 2 consecutive rendezvous timeouts so a
sustained host overload does not pile up placeholder samples. Under
KASLR (the default), a boundary that would fire before the guest's
address slide is published is deferred, not dropped — it fires on
the next loop iteration. Assert a lower bound on coverage, not
equality:

```rust,ignore
fn check_coverage(result: &VmResult) -> Result<()> {
    anyhow::ensure!(result.periodic_target == 3);
    anyhow::ensure!(
        result.periodic_fired >= 2,
        "too few periodic samples ({}/{})",
        result.periodic_fired,
        result.periodic_target,
    );
    Ok(())
}
```

`periodic_target` mirrors the configured `num_snapshots`;
`periodic_fired` counts boundaries actually serviced (including
rendezvous-timeout placeholders). When `post_vm` is omitted on a
periodic-configured test, the macro installs a default callback
asserting at least one boundary fired with real BPF state.

## Draining the bridge

The assertion pipeline runs on the host after `vm.run()` returns —
inside a `post_vm` callback. The recommended path is
`drain_ordered_with_stats` fed into
`SampleSeries::from_drained_typed`, which preserves insertion order,
per-sample stats results, and timestamps:

```rust,ignore
use ktstr::prelude::*;

fn post_vm(result: &VmResult) -> Result<()> {
    let series = SampleSeries::from_drained_typed(
        result.snapshot_bridge.drain_ordered_with_stats(),
        result.monitor.clone(),
    )
    .periodic_only();

    anyhow::ensure!(
        !series.is_empty(),
        "no periodic samples — coordinator never fired",
    );

    // ... project a field and feed a temporal pattern ...
    Ok(())
}

#[ktstr_test(num_snapshots = 3, duration_s = 10, post_vm = post_vm)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("workers").workers(2).work_type(WorkType::SpinWait),
    ])
}
```

Each drained entry carries the tag, the captured report, the typed
per-sample stats result (`Err(MissingStatsReason)` when the stats
request failed or no scheduler stats client was wired), a
pause-adjusted `elapsed_ms` timestamp, the scheduled
`boundary_offset_ms`, and the scenario phase stamp (`step_index`).
The other drain variants drop metadata the temporal pipeline needs —
see the
[`SnapshotBridge` rustdoc](https://ktstr.dev/rustdoc/ktstr/scenario/snapshot/struct.SnapshotBridge.html)
if you need them.

[Temporal Assertions](temporal-assertions.md) owns the sample
anatomy and projection surface;
[Snapshots](snapshots.md#errors-carry-the-fix) owns the per-sample
error routing (`PlaceholderSample`, `MissingStats`).

## What to assert

Two stages: compose the series (drain, `periodic_only()`), then
project a column and pick a pattern. For monotonic counters,
`nondecreasing` is the canonical choice; for utilization-style
metrics that should hold once warmup ends, `steady_within`; for
"stabilizes near a target by a deadline", `converges_to`. The full
pattern surface, projection helpers, and failure rendering live in
[Temporal Assertions](temporal-assertions.md).
