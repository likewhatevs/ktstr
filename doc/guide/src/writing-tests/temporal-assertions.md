# Projections and Temporal Assertions

The built-in [checks](../concepts/checking.md) assert on what ktstr
measures for you. **Projections** assert on what *your scheduler*
measures: any per-sample value the harness can see — a field in the
scheduler's own scx_stats output, any global or map entry in the
frozen BPF snapshots, per-CPU host time — projects into a typed
column, and every pattern and claim on this page applies to it. A
scheduler's internal counters become first-class test signals
without ktstr knowing their names.

<div class="kt-steps">
<div class="kt-step" data-step="1"><strong>Capture</strong><p>Enable periodic snapshots so ktstr records comparable samples during the run.</p></div>
<div class="kt-step" data-step="2"><strong>Project</strong><p>Turn a scx_stats path, BPF value, or host-CPU field into one typed <code>SeriesField&lt;T&gt;</code>.</p></div>
<div class="kt-step" data-step="3"><strong>Assert</strong><p>Apply monotonicity, rate, convergence, ratio, or scalar checks to one verdict.</p></div>
</div>

The shape is two-stage: build a [`SampleSeries`](#sampleseries) from
the drained periodic captures, then **project** a
[`SeriesField<T>`](#seriesfield) — one column of `T`-typed values
across every sample — and feed it through a pattern. Pick by the
question:

| Pattern | Question it answers | Type | On a projection error |
|---|---|---|---|
| [`nondecreasing`](#nondecreasing--strictly_increasing) | does this counter only go up? | any ordered | skip the pair, note the gap |
| [`strictly_increasing`](#nondecreasing--strictly_increasing) | does it advance every period? | any ordered | skip the pair, note the gap |
| [`rate_within(lo, hi)`](#rate_withinlo-hi-f64-only) | does it advance at the right speed? | f64 | gap — no rate across it, note |
| [`steady_within(warmup, tol)`](#steady_withinwarmup_ms-tolerance-f64-only) | does it hold near its mean after warmup? | f64 | skip the sample, note |
| [`converges_to(target, tol, deadline)`](#converges_totarget-tolerance-deadline_ms-f64-only) | does it stabilize near a target in time? | f64 | interrupts the witness run, note |
| [`always_true`](#always_true-bool-only) | does this invariant hold at every sample? | bool | fail — strict |
| [`ratio_within(other, lo, hi)`](#ratio_withinother-lo-hi-f64-only) | do two series stay in proportion? | f64 | skip the index, note |
| [`each(...)`](#per-sample-scalar-checks-each) | is every sample inside a scalar bound? | any | fail — strict |

Every pattern takes `&mut Verdict` and returns it, so assertions
chain onto one accumulator; each failure records a
`DetailKind::Temporal` detail, and coverage gaps record `Note`s.
For enabling capture and draining the bridge, see
[Periodic capture](snapshots.md#periodic-capture) — this page covers
projection and assertion.

## SampleSeries

`SampleSeries` is the ordered sample sequence drained from the
bridge after the VM exits:

```rust,ignore
use ktstr::prelude::*;

let drained = vm_result.snapshot_bridge.drain_ordered_with_stats();
let series = SampleSeries::from_drained_typed(drained, monitor).periodic_only();
```

`periodic_only()` filters to tags beginning with `"periodic_"`,
stripping on-demand captures and watchpoint fires that share the
bridge; `periodic_ref()` is the borrowed-iterator equivalent when
one test needs both views.

`SampleSeries` exposes:

- `len()`, `is_empty()` — sample count.
- `iter_samples()` — borrowed `Sample<'_>` views. Each sample
  carries `tag`, a pause-adjusted `elapsed_ms`, a
  `Snapshot<'_>` over the captured BPF state, a
  `step_index: Option<u16>` phase stamp, and
  `stats: Result<&Value, &MissingStatsReason>` — the per-sample
  scx_stats JSON, or the typed reason the stats request failed.
- `bpf(label, |snap| …)` / `stats(label, |sv| …)` — closure
  projection along the BPF or stats axis.
- `bpf_live_u64(name)` / `bpf_live_i64` / `bpf_live_f64` — terse
  BPF-axis shorthand that resolves `name` via the
  auto-disambiguating `Snapshot::live_var` accessor (no closure);
  mirrored on the stats axis as `stats_live_u64(path)` / `_i64` /
  `_f64`.
- `bpf_map(map_name)` / `stats_path(path)` — typed auto-projection
  (see [Auto-projection](#auto-projection)).
- `host()` / `monitor()` — the two host-side data sources: a
  per-sample per-CPU time timeline, and the run-level monitor
  aggregate (see [The host and monitor axes](#the-host-and-monitor-axes)).
- `by_stamped_phase()` — group samples by the bridge-stamped
  scenario phase (`BTreeMap<u16, Vec<Sample>>`; 0 = baseline,
  1..=N = step ordinals). Prefer
  `by_stimulus_phase(stimulus_events)` when a stimulus timeline is
  available — it re-derives the phase from each sample's
  `boundary_offset_ms` and is immune to deferred-fire bursts that
  collapse stamped phases.

## SeriesField

A `SeriesField<T>` is one per-sample column. Each slot is a
`SnapshotResult<T>`, so a missing field, type mismatch, or
placeholder report on one sample does not abort the projection — it
surfaces at the assertion site as a per-sample error the pattern
decides how to handle (see the table above). The field carries each
sample's tag and timestamp alongside the value, so failure messages
name the offending sample without re-threading the series.

### Projecting from BPF state

The `bpf` closure receives each sample's `Snapshot<'_>`; the body is
a normal [Snapshot accessor expression](snapshots.md):

```rust,ignore
let nr_dispatched: SeriesField<u64> = series.bpf(
    "nr_dispatched",
    |snap| snap.var("nr_dispatched").as_u64(),
);
```

### Projecting from scx_stats JSON

The `stats` closure receives a `StatsValue<'_>` wrapper over the
per-sample stats JSON:

```rust,ignore
let busy: SeriesField<f64> = series.stats(
    "busy",
    |sv| sv.get("busy").as_f64(),
);
```

A sample whose stats slot is `Err` (the stats request failed, or no
scheduler stats client was wired) yields a
`SnapshotError::MissingStats { tag, reason }` slot — distinct from
an in-JSON path miss (`FieldNotFound` / `TypeMismatch`) so coverage
gaps and data errors stay distinguishable.

Each sample's stats slot is the scheduler's response stored
**verbatim** — ktstr issues a fresh scx_stats request just before
each freeze and never diffs consecutive responses. Whether a field
is cumulative-since-attach or per-interval is the *scheduler's*
convention, and it decides which assertion is sound: a cumulative
counter fits `nondecreasing` and per-phase
`counter_delta_per_phase()`, while a per-interval value fed to a
delta reducer double-subtracts and asserts on noise.

### Auto-projection

The typed auto-projectors emit ready-to-feed `SeriesField`s without
a closure:

```rust,ignore
// Top-level scalar member of a BPF map's first entry.
let dispatched = series
    .bpf_map("scx_obj.bss")
    .at(0)
    .field_u64("nr_dispatched");

// Stats path drilling into nested layer/cgroup keys.
let layer_util = series
    .stats_path("layers")
    .key("batch")
    .field_f64("util");
```

Bulk discovery: `member_names()` / `u64_fields()` / `f64_fields()`
on the BPF projector (`key_names()` on the stats projector) project
every member that yields at least one `Ok` across the series —
useful for blanket "every counter must be nondecreasing" sweeps.
The typed `field_*` helpers reach top-level scalars only; nested
members (`"ctx.weight"`) need the closure path. Per-CPU maps use
the projector's cross-CPU reductions (`field_cpu_sum_*` /
`field_cpu_max_*` / `field_cpu_min_*`) or `.cpu(n).field_*`. When
several counters must come from the *same* scheduler section,
`live_bpf_vars_via([names; N], picker)` resolves them per-sample
through one shared picker call and returns N parallel fields.

## The host and monitor axes

Two more data sources ride the same series, both captured host-side:

**`series.host()`** returns a per-sample **per-CPU time timeline**
(`None` when no sample carried per-CPU data). `cpus()` lists every
captured CPU; `per_cpu_field_u64(cpu, label, |t| …)` projects one
field of the per-CPU kernel cpustat readings (`cpustat_user_ns`,
`cpustat_system_ns`, `cpustat_irq_ns`, `cpustat_idle_ns`, …) into a
`SeriesField<u64>` — the same shape as `series.bpf`, so the identical
patterns apply:

```rust,ignore
if let Some(host) = series.host() {
    for cpu in host.cpus() {
        host.per_cpu_field_u64(cpu, format!("cpu{cpu}_system_ns"),
            |t| t.cpustat_system_ns)
            .nondecreasing(&mut v);
    }
}
```

A sample that missed a CPU surfaces as a per-sample
`HostFieldUnavailable` slot, routed like any other projection error.

**`series.monitor()`** is not a series — it is the run-level
**aggregate** view over the monitor thread's own sampling cadence
(`None` when no monitor report was attached). `summary()` exposes
imbalance ratios, nr_running averages, DSQ depths, and schedstat
deltas; `scx_events()` the SCX event-counter deltas. These are
scalars: feed them to [`claim!` and the Verdict
comparators](../concepts/checking.md#verdict-the-claim-accumulator),
not the trajectory patterns.

## The patterns

### `nondecreasing` / `strictly_increasing`

Pass when every consecutive pair satisfies
`values[i] <= values[i+1]` (or `<` for the strict variant). The
shape for kernel counters whose only legal direction is up.

```rust,ignore
let mut v = Verdict::new();
nr_dispatched.nondecreasing(&mut v);
nr_dispatched.strictly_increasing(&mut v); // require advance every period
```

Projection errors are skipped — the affected pair is dropped, the
skip is logged as a `Note`, and the verdict is not flipped on
missing data; adjacent samples on either side of a gap are still
checked. Fewer than 2 samples records a "vacuously holds" `Note`
and passes.

### `rate_within(lo, hi)` (f64 only)

Pass when every consecutive `(delta_value / delta_ms)` lies in
`[lo, hi]`, computed from the per-sample timestamps — a counter that
should advance at ~1 unit/ms reads as `rate_within(0.5, 2.0)`.

```rust,ignore
let ticks: SeriesField<f64> = series.bpf("ticks",
    |snap| snap.var("ticks").as_f64());
ticks.rate_within(&mut v, 0.5, 2.0);
```

A zero-time delta records an inconclusive detail (zero denominator)
naming the pair; a non-finite rate records its own detail rather
than slipping past the band; `lo > hi` is a single caller-error
detail. Projection errors are gaps — no rate is computed across
them.

### `steady_within(warmup_ms, tolerance)` (f64 only)

Pass when every post-warmup sample (`elapsed_ms >= warmup_ms`) lies
inside `[mean·(1-tolerance), mean·(1+tolerance)]`. The mean is
computed over post-warmup samples only, so ramp-up does not bias
the baseline. `tolerance` is a fraction (`0.10` = ±10%).

```rust,ignore
let util: SeriesField<f64> = series.stats("busy",
    |sv| sv.get("busy").as_f64());
util.steady_within(&mut v, /*warmup_ms=*/ 1000, /*tolerance=*/ 0.10);
```

Projection errors are skipped with a `Note`. When warmup absorbs
every sample, the pattern notes "no samples beyond warmup" and
passes vacuously.

### `converges_to(target, tolerance, deadline_ms)` (f64 only)

Pass when three consecutive samples land inside
`[target - tolerance, target + tolerance]` at or before
`deadline_ms` — the convergence-witness shape for "the system
stabilizes near `target` by the deadline".

```rust,ignore
load.converges_to(&mut v, /*target=*/ 1.0, /*tol=*/ 0.5, /*deadline_ms=*/ 5_000);
```

Distinct outcomes: witness found — pass. No witness before the
deadline — a temporal failure naming the sample count (and any
errored samples that interrupted in-progress runs). Fewer than 3
successfully-projected samples in the window — a `Note`, not a
failure: absence of data is a coverage gap, not a negative finding,
and the note distinguishes "did not collect enough" from "collected
enough but never converged".

### `always_true` (bool only)

Pass when every sample's value is `true`. Projection errors fail
the assertion — this is a strict pattern; a missing boolean is a
coverage gap that must surface.

```rust,ignore
let alive: SeriesField<bool> = series.bpf("scheduler_alive",
    |snap| snap.var("scheduler_alive").as_bool());
alive.always_true(&mut v);
```

### `ratio_within(other, lo, hi)` (f64 only)

Pass when every per-index `self[i] / other[i]` lies in `[lo, hi]` —
two same-length series walked in lock-step.

```rust,ignore
util.ratio_within(&mut v, &runtime, 0.4, 0.6);
```

A length mismatch fires one caller-error detail and aborts. A zero
denominator records an inconclusive detail naming the sample;
out-of-band ratios record the lhs/rhs values. Projection errors on
either side are skipped with a `Note` naming each gap and which
side errored.

## Per-sample scalar checks: `each`

For per-sample bounds, bypass the trajectory patterns via
`SeriesField::each`:

```rust,ignore
nr_dispatched.each(&mut v).at_least(1u64);
util.each(&mut v).between(0.0_f64, 100.0_f64);
ticks.each(&mut v).at_most(10_000.0_f64);
```

`each` runs the comparator on every successfully-projected sample;
the first failure records a detail and subsequent failures pile on,
so the timeline shows every offending sample. Projection errors
flip the verdict (`each` is strict, matching `always_true`). NaN
samples report an `incomparable` failure by name — without that
branch, IEEE-754 comparisons against NaN are always false, and a
NaN would silently pass `value < floor` checks.

## Phase-bucketed comparisons

Steps stamp each capture with a scenario phase
(`Phase::BASELINE`, then `Phase::step(0)`, `Phase::step(1)`, … in
step order). Per-phase reducers on a projected field —
`counter_delta_per_phase()`, `first_per_phase()`,
`last_per_phase()`, `value_at_phase(phase)` — reduce the series to
one value per phase, and `ratio_across_phases` pins a later phase
against an earlier one. The swap-A/B shape (step 0 runs scheduler
A, step 1 swaps in B via `Op::ReplaceScheduler`):

```rust,ignore
use ktstr::assert::Phase;

let dispatched = series.bpf_live_f64("nr_dispatched");
dispatched
    .ratio_across_phases(&mut v, Phase::step(0), Phase::step(1))
    .at_most(1.5); // B may cost at most 1.5x A on this counter
```

`at_most` records the computed ratio and both phase values in the
verdict — pass or fail — so the margin is visible without extra
printing. A phase with no Ok-samples or a zero baseline records an
inconclusive detail rather than a fake ratio.
`PhaseMapExt::ratio_across_phases` does the same on a pre-reduced
`BTreeMap<Phase, _>` for caller-derived per-phase values.

## Failure rendering

Every temporal failure carries the field's label, the pattern name,
and the offending sample's tag and timestamp. A nondecreasing
regression renders as (shape pinned by the library's format
strings):

```text
nr_dispatched (nondecreasing): regression at sample periodic_004 (+850ms): \
    value 100 after prior value 200 at sample periodic_003 (+700ms)
```

Coverage `Note`s render with the per-sample error variant, so
`PlaceholderSample` (rendezvous timeout), `MissingStats` (stats
request failed), `FieldNotFound` (typo / wrong map), and
`TypeMismatch` are distinguishable without a debugger:

```text
nr_dispatched (nondecreasing): skipped 1 sample(s) with projection errors: \
    periodic_002(+500ms): snapshot has no global variable 'nrdispatch' \
    in any *.bss/*.data/*.rodata map (available globals: ["nr_dispatched", \
    "nr_enqueued"])
```

## Worked example

The pipeline runs on the host: `post_vm` receives the `VmResult`
after `vm.run()` returns, drains the bridge, and walks the series:

```rust,ignore
use ktstr::prelude::*;

fn assert_temporal_patterns(result: &VmResult) -> Result<()> {
    let series = SampleSeries::from_drained_typed(
        result.snapshot_bridge.drain_ordered_with_stats(),
        result.monitor.clone(),
    )
    .periodic_only();

    let mut v = Verdict::new();

    // BPF axis: counter must never regress.
    let nr_dispatched: SeriesField<u64> = series.bpf(
        "nr_dispatched",
        |snap| snap.var("nr_dispatched").as_u64(),
    );
    nr_dispatched.nondecreasing(&mut v);

    // Stats axis: stay under a generous ceiling.
    let stats_dispatched: SeriesField<u64> = series.stats(
        "nr_dispatched",
        |sv| sv.get("nr_dispatched").as_u64(),
    );
    stats_dispatched.each(&mut v).at_most(1_000_000_000u64);

    v.into_anyhow_or_log()
}

#[ktstr_test(num_snapshots = 3, duration_s = 10, post_vm = assert_temporal_patterns)]
fn dispatch_counter_advances(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("workers").workers(2).work_type(WorkType::SpinWait),
    ])
}
```

## Where projected values travel

A projected value has exactly two sinks: the temporal patterns and
the `Verdict` claim comparators. Both terminate in the test's
`AssertResult` — its pass/fail verdict and failure details.

Projected values do **not** enter the stats sidecar, so they are not
visible to `cargo ktstr stats` or gateable by `perf-delta` /
`PerfDeltaAssertion`. That pipeline is fed exclusively by the
[metric registry](../reference/assertable-metrics.md)'s fixed
populators; scx_stats JSON never reaches it at all. The two levers
are complementary, not interchangeable:

- **Cross-run regression gating** on a scheduler-specific number:
  emit it from a workload payload with JSON output
  ([Payloads](payloads.md)), whose extracted keys *do* land in the
  sidecar.
- **Within-run assertions** on scheduler internals — trajectory,
  bounds, phase A/B: project it here. No registration, no payload,
  any name your scheduler exports.

For capture wiring and `num_snapshots` semantics, see
[Periodic capture](snapshots.md#periodic-capture); for the `Snapshot`
accessors the projection closures call into, see
[Snapshots and Live Capture](snapshots.md).
