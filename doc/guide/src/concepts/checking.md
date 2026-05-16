# Checking

ktstr checks scheduler behavior through two channels: worker-side
telemetry and host-side monitoring.

## Worker checks

After each scenario, ktstr collects
[`WorkerReport`](../architecture/workers.md#telemetry) from every worker
process. Several checks run against these reports:

**Starvation** -- any worker with `work_units == 0` fails the test.

**Fairness** -- workers in the same cgroup should get similar CPU time.
The "spread" (max off-CPU% - min off-CPU%) must be below a threshold
(15% in release builds, 35% in debug). Violations report the spread
and per-cgroup statistics.

**Scheduling gaps** -- the longest wall-clock gap observed at
work-unit checkpoints. Gaps above a threshold (2000ms release, 3000ms
debug) indicate the scheduler dropped a task. Reports include the gap
duration, CPU, and timing.

**Cpuset isolation** -- workers must only run on CPUs in their assigned
cpuset. Any execution on an unexpected CPU fails the test. Opt-in via
`isolation = true` on the `#[ktstr_test]` attribute or via
`Assert::check_isolation()`; `Assert::default_checks()` leaves this
`None`, so the runtime merge resolves to `false` and the check is
skipped unless explicitly enabled.

**Throughput parity** -- `assert_throughput_parity()` checks that
workers produce similar throughput (work_units per CPU-second). Two
thresholds:
- `max_throughput_cv`: coefficient of variation across workers. High
  CV means the scheduler gives some workers disproportionately less
  effective CPU. Requires at least 2 workers with nonzero CPU time.
- `min_work_rate`: minimum work_units per CPU-second per worker.
  Catches cases where all workers are equally slow (CV passes but
  absolute throughput is too low).

Neither threshold is set by default; enable via `Assert` setters or
`#[ktstr_test]` attributes.

**Benchmarking** -- `assert_benchmarks()` checks per-wakeup latency
and iteration throughput. Three thresholds:
- `max_p99_wake_latency_ns`: p99 of all `resume_latencies_ns` samples
  across workers in a cgroup. Populated only for work types that
  record wake-to-run latency: `IoSyncWrite`, `IoRandRead`, `IoConvoy`,
  `Bursty`, `PipeIo`,
  `FutexPingPong`, `CacheYield`, `CachePipe`, `FutexFanOut`
  (receivers), `Sequence` (Sleep / Yield / Io phases),
  `ForkExit`, `NiceSweep`, `AffinityChurn`, `PolicyChurn`,
  `FanOutCompute`, `MutexContention`. Pure-CPU work types
  (`SpinWait`, `Mixed`, `CachePressure`, `PageFaultChurn`) do not
  record samples.
- `max_wake_latency_cv`: coefficient of variation of wake latency
  samples. High CV means inconsistent scheduling latency.
- `min_iteration_rate`: minimum outer-loop iterations per wall-clock
  second per worker.

None are set by default. Set via `Assert` setters or `#[ktstr_test]`
attributes.

## Monitor checks

The [host-side monitor](../architecture/monitor.md) reads guest VM
memory (per-CPU runqueue structs via BTF offsets) and evaluates:

- **Imbalance ratio**: `max(nr_running) / max(1, min(nr_running))`
  across CPUs. The denominator is clamped to 1 so an all-idle sample
  does not divide by zero.
- **Local DSQ depth**: per-CPU dispatch queue depth.
- **Stall detection**: `rq_clock` not advancing on a CPU with
  runnable tasks. Idle CPUs and preempted vCPUs are exempt. See
  [Monitor: Stall detection](../architecture/monitor.md#stall-detection)
  for exemption details.
- **Event rates**: scx fallback and keep-last event counters.

Monitor thresholds use a sustained sample window (default: 5 samples).
A violation must persist for N consecutive samples before failing.

## NUMA checks

When workers use a [`MemPolicy`](mem-policy.md), ktstr collects NUMA
page placement data and checks it against thresholds:

**Page locality** -- `assert_page_locality()` checks the fraction of
pages residing on the expected NUMA node(s). Expected nodes are derived
from the worker's `MemPolicy::node_set()` at evaluation time. Page
counts come from `WorkerReport::numa_pages` (parsed from
`/proc/self/numa_maps`). Returns 0.0 when no pages are observed -- a
zero-allocation workload is treated as zero-locality (not vacuously
local) so `min_page_locality` thresholds surface broken runs that
produced no NUMA signal. Fails if the observed fraction falls below
`min_page_locality`.

**Cross-node migration** -- `assert_cross_node_migration()` checks
the ratio of migrated pages to total allocated pages.
`WorkerReport::vmstat_numa_pages_migrated` provides the delta of the
`numa_pages_migrated` counter from `/proc/vmstat` over the work loop.
Fails if the ratio exceeds `max_cross_node_migration_ratio`.

**Slow-tier ratio** -- `max_slow_tier_ratio` checks the fraction of
pages on memory-only NUMA nodes (CXL tiers). Fails if more than the
specified fraction of pages land on memory-only nodes.

None of these thresholds are set by default. Set via `Assert` setters
or `#[ktstr_test]` attributes.

## Assert struct

`Assert` is a composable configuration that carries both worker checks
and monitor thresholds:

```rust,ignore
pub struct Assert {
    // Worker checks
    pub not_starved: Option<bool>,
    pub isolation: Option<bool>,
    pub max_gap_ms: Option<u64>,
    pub max_spread_pct: Option<f64>,

    // Throughput checks
    pub max_throughput_cv: Option<f64>,
    pub min_work_rate: Option<f64>,

    // Benchmarking checks
    pub max_p99_wake_latency_ns: Option<u64>,
    pub max_wake_latency_cv: Option<f64>,
    pub min_iteration_rate: Option<f64>,
    pub max_migration_ratio: Option<f64>,

    // Monitor checks
    pub max_imbalance_ratio: Option<f64>,
    pub max_local_dsq_depth: Option<u32>,
    pub fail_on_stall: Option<bool>,
    pub sustained_samples: Option<usize>,
    pub max_fallback_rate: Option<f64>,
    pub max_keep_last_rate: Option<f64>,

    // NUMA checks
    pub min_page_locality: Option<f64>,
    pub max_cross_node_migration_ratio: Option<f64>,
    pub max_slow_tier_ratio: Option<f64>,

    // Monitor-merge policy + scx_bpf_error matchers
    pub enforce_monitor_thresholds: bool,
    pub expect_scx_bpf_error_contains: Option<&'static str>,
    pub expect_scx_bpf_error_matches: Option<&'static str>,
}
```

Every threshold field is `Option`; `None` means "inherit from parent
layer." `enforce_monitor_thresholds` is the only non-`Option` field
because it controls the sticky-`||` merge policy (any layer setting
`true` keeps it `true`). The two `expect_scx_bpf_error_*` fields pin
a regex / substring against the SCX exit-message stream and are
documented per-attribute in the
[`#[ktstr_test]` macro reference](../writing-tests/ktstr-test-macro.md).

## Merge layers

Checking uses a three-layer merge:

1. `Assert::default_checks()` -- currently aliases `NO_OVERRIDES`;
   every check is `None`. The fn-name is a hook for a future
   baseline policy; today it is a synonym. Tests opt in to
   assertions explicitly via scheduler-level or per-test overrides,
   or by calling `.with_monitor_defaults()` to populate the
   monitor-threshold bundle from `MonitorThresholds::DEFAULT`.
2. `Scheduler.assert` -- scheduler-level overrides.
3. Per-test `assert` -- test-specific overrides via `#[ktstr_test]`
   attributes.

All threshold fields use last-`Some`-wins semantics. A `Some(false)`
in a higher layer can disable a check that a lower layer enabled.
`enforce_monitor_thresholds` uses sticky-`||`: once any layer sets it
`true` the merged result stays `true`.

```rust,ignore
let test_assert = Assert::NO_OVERRIDES.max_gap_ms(5000);
let final_assert = Assert::default_checks()
    .merge(&scheduler.assert)
    .merge(&test_assert);
```

## Default thresholds

### Worker checks

| Check | Default (release) | Default (debug) |
|---|---|---|
| Scheduling gap | 2000 ms | 3000 ms |
| Fairness spread | 15% | 35% |

Debug builds run in small VMs with higher scheduling overhead, so
thresholds are relaxed. Coverage-instrumented builds collect profraw
data for code coverage analysis; all assertion and monitor threshold
checks run normally.

### Monitor threshold values applied when `with_monitor_defaults()` is called

These thresholds activate only when a test (or its scheduler) calls
`.with_monitor_defaults()` on its `Assert`; otherwise the
corresponding fields stay `None` and the monitor's violations land
in `details` without flipping `passed`.

| Threshold | Default | Rationale |
|---|---|---|
| `max_imbalance_ratio` | 4.0 | `max(nr_running) / max(1, min(nr_running))` across CPUs (denominator clamped to 1 so an all-idle sample does not divide by zero). Lower values (2-3) false-positive during cpuset transitions. |
| `max_local_dsq_depth` | 50 | Per-CPU dispatch queue overflow. Sustained depth above this means the scheduler is not consuming dispatched tasks. |
| `fail_on_stall` | true | Fail when `rq_clock` does not advance on a CPU with runnable tasks. Idle CPUs (NOHZ) and preempted vCPUs are exempt. |
| `sustained_samples` | 5 | At ~100ms sample interval, requires ~500ms of sustained violation. Filters transient spikes from cpuset reconfiguration. |
| `max_fallback_rate` | 200.0/s | `select_cpu_fallback` events per second across all CPUs. Sustained rate indicates systematic `select_cpu` failure. |
| `max_keep_last_rate` | 100.0/s | `dispatch_keep_last` events per second across all CPUs. Sustained rate indicates dispatch starvation. |

All monitor thresholds use the `sustained_samples` window -- a
violation must persist for N consecutive samples before failing.

## Worker checks via Assert

`Assert` provides `assert_cgroup()` for running worker-side checks
directly against collected reports:

```rust,ignore
let a = Assert::default_checks().max_gap_ms(5000);
let result = a.assert_cgroup(&reports, Some(&cpuset));
```

Use `Assert` for both the merge chain (`#[ktstr_test]` attributes,
`Scheduler.assert`, `execute_steps_with`) and direct report checking.

For NUMA-aware tests, use `assert_cgroup_with_numa()` to pass the
expected NUMA node set explicitly:

```rust,ignore
let result = a.assert_cgroup_with_numa(
    &reports,
    Some(&cpuset),
    Some(&numa_nodes),  // e.g. derived via TestTopology::numa_nodes_for_cpuset
);
```

The bare `assert_cgroup` passes `None` for `numa_nodes`, which skips
`page_locality` and `cross_node_migration` checks. Tests that drive
NUMA assertions must use the `_with_numa` variant.

## Preset baselines: `SchedulerBaseline`

`SchedulerBaseline` is a flat threshold preset designed for direct
invocation in test bodies, distinct from the merge-tree threshold
config carried by `Assert`. Use when a test wants a one-call
multi-field check without engaging the `default_checks → scheduler →
test` merge chain.

```rust,ignore
use ktstr::assert::{SchedulerBaseline, assert_baseline};

// Sane-default preset: p99 wake under 10ms, p99 iteration cost
// under 1ms, total migrations under 1000, each worker >= 1 work unit.
let r = assert_baseline(&reports, &SchedulerBaseline::strict());

// Or build piecewise with explicit thresholds.
let baseline = SchedulerBaseline::EMPTY
    .max_p99_wake_latency_ns(5_000_000)
    .min_work_units(100);
let r = assert_baseline(&reports, &baseline);
```

Each field is independent — `None` skips that check. The four fields:

- `max_p99_wake_latency_ns` -- pooled p99 across every worker's
  `resume_latencies_ns`. Same semantics as `Assert::max_p99_wake_latency_ns`.
- `max_iteration_cost_p99_ns` -- pooled p99 across every worker's
  `iteration_costs_ns`. Only meaningful for compute work types
  (`AluHot`, `SmtSiblingSpin`, `IpcVariance`); blocking variants
  report empty reservoirs and the check is a no-op.
- `max_migrations` -- absolute sum of `migration_count` across
  workers. Distinct from `Assert::max_migration_ratio` (per-iteration
  rate); useful when the test pins a known workload size.
- `min_work_units` -- per-worker floor. One starved worker fails.
  Distinct from `assert_not_starved`'s zero-floor — accepts a
  non-zero threshold so tests can reject "barely made progress" runs.

`assert_baseline` returns a skip when `reports` is empty (a baseline
against zero samples would silently green-light a broken run that
produced no signal).

The preset composes with the merge-chain path: a test can run
`assert_baseline` against a worker-report slice AND merge the
`Assert`-derived result into the same accumulator via
`AssertResult::merge`.

## SCX event checks

`assert_scx_events_clean(events, max_count)` checks SCX scheduler
event counters (BPF-side `scx_event_stats`) against a bound. Useful
for pinning "no fallbacks fired" or "no error-class events occurred"
in tests that drive a specific scheduler path.

```rust,ignore
use ktstr::assert::assert_scx_events_clean;

// Strict: every counter must be exactly zero.
let r = assert_scx_events_clean(
    &[("select_cpu_fallback", 0), ("dispatch_keep_last", 0)],
    None,
);

// Tolerant: small counts allowed up to a caller-supplied bound.
let r = assert_scx_events_clean(
    &[("dispatch_keep_last", 3)],
    Some(10),
);
```

Negative counts (corrupted source data — wraparound, signed
conversion, JSON bit-loss) are treated as failures regardless of
bound. Failures are tagged `DetailKind::SchedulerEvent`.

## Verdict: the claim accumulator

`Verdict` is the per-test claim accumulator. `Assert` holds threshold
config and stays `Copy`; `Verdict` carries the per-test claim records
(which include `Vec`/`String` allocations) and is built via
`Assert::defaults().verdict()` or `Verdict::new()`.

Test authors reach for one of two compile-mechanical labelers:

1. **Typed field accessors** generated by `#[derive(Claim)]` on stats
   structs (where `stats` is a `CgroupStats` value collected from your
   worker reports):

   ```rust,ignore
   use ktstr::assert::{Assert, Verdict};
   let mut v = Assert::defaults().verdict();
   stats.claim_max_gap_ms(&mut v).at_most(100);
   stats.claim_total_iterations(&mut v).at_least(1000);
   let result = v.into_result();
   ```

   The label (`"max_gap_ms"`) comes from `stringify!(max_gap_ms)` in
   the generated method body — renaming the field updates both the
   method name AND the rendered label.

2. **The `claim!` macro** on a local binding or expression:

   ```rust,ignore
   use ktstr::claim;
   let mut v = Verdict::new();
   let iter_delta = compute_delta(&reports);
   claim!(v, iter_delta).at_least(100);
   let result = v.into_result();
   ```

   The label comes from `stringify!(<token tree>)` over the
   expression tokens.

There is no recommended third "manual string" path. `Verdict` does
expose `claim`, `claim_set`, and `claim_seq` `pub` methods (all marked
`#[doc(hidden)]`) that the derive and the macro dispatch through, but
hand-typing them is disallowed by convention — a manual string can
drift from the value it labels (rename a field, leave the literal
stale), so labels must originate from `stringify!(field)` or
`stringify!(expr)` via the derive or the macro. The methods compile if
invoked directly, but a code reviewer should treat hand-typed
`claim` / `claim_set` / `claim_seq` calls as a violation of the
intended API surface.

### Comparator surface

For scalar `ClaimBuilder<T>`:

- `T: PartialOrd + Display` → `at_least`, `at_most`, `lt`, `gt`, `between`
- `T: PartialEq + Display` → `eq`, `ne`
- `T = f64` → `is_finite`, `near`

For container claims (set / sequence), comparators bypass scalars and
offer `empty` / `nonempty` / `contains` / `len_eq` / `len_at_most` /
`len_at_least` / `subset_of` / `disjoint_from`.

### Finishing the verdict

`Verdict::into_result()` consumes the accumulator and returns an
`AssertResult` with the same `passed` / `details` / `stats` shape as
the direct-invocation paths. Compose via `AssertResult::merge` to
combine claim outcomes with `assert_cgroup` / `assert_baseline` /
`assert_scx_events_clean` results in the same scenario.

## Constants

- `Assert::NO_OVERRIDES` -- identity for `merge`; every field is `None`,
  so it overrides nothing. This is not "no checks" -- when used as a
  per-test or per-scheduler `assert`, the runtime chain still applies
  the merge of `default_checks() -> scheduler -> test`.
- `Assert::default_checks()` -- currently aliases `NO_OVERRIDES` (every
  check is `None`). Reserved as a hook for a future baseline policy.
- `Assert::empty()` and `Assert::defaults()` -- method-style aliases
  for the two constants above. Pair naturally with `.verdict()` when
  building a `Verdict` from a fresh `Assert` in claim-style code.
- `.with_monitor_defaults()` -- populates the monitor-threshold
  bundle (`max_imbalance_ratio`, `max_local_dsq_depth`,
  `fail_on_stall`, `sustained_samples`, `max_fallback_rate`,
  `max_keep_last_rate`) from `MonitorThresholds::DEFAULT`. Tests that
  want stall + imbalance protection must opt in via this method or
  set the fields directly.

## AssertResult

`AssertResult` carries pass/fail status, diagnostic messages, and
aggregated statistics from a scenario run.

### Construction

- `AssertResult::pass()` -- creates a passing result with empty
  details and default stats.
- `AssertResult::skip(reason)` -- creates a passing result with a
  skip reason in `details` and `skipped = true`. Used when a
  scenario cannot run under the current topology or flag
  combination but is not a failure.
- `AssertResult::fail(detail)` -- failing result carrying a single
  `AssertDetail`. Mirrors `pass` / `skip` for the failure axis.
- `AssertResult::fail_msg(msg)` -- shortcut for the common case
  where the failure is a plain diagnostic message tagged
  `DetailKind::Other`.

### Mutation and inspection

- `result.note(msg)` -- append an informational annotation tagged
  `DetailKind::Note`. Does NOT flip `passed` or `skipped` — a
  note is context, not a verdict. Returns `&mut Self` so calls
  chain.
- `result.with_note(msg)` -- builder-style sibling of `note` that
  consumes and returns `self`. Use at the return site to chain a
  context annotation onto a fresh result without an intermediate
  `let mut`.
- `result.note_value(key, value)` -- insert a typed measurement
  into `measurements` under `key`. Use for any value a downstream
  comparator should lift programmatically (latency p99, throughput
  per worker, scheduler-specific counter). Returns `&mut Self`.
- `result.with_note_value(key, value)` -- builder-style sibling of
  `note_value` that consumes and returns `self`. Pairs naturally
  with `pass()` / `fail_msg(msg)` at the return site.
- `result.is_skipped()` -- convenience accessor returning
  `skipped`. Stats tooling uses this to subtract non-executions
  from pass counts.
- `result.is_failed()` -- convenience accessor returning
  `!passed`. Mirrors `is_skipped` so branches reading "did this
  claim fail?" don't negate `.passed` inline.

### Composing results: `any_of` and `all_of`

When several sibling assertions form a logical AND or OR,
`AssertResult::all_of([...])` and `AssertResult::any_of([...])`
fold a slice of results into one. `all_of` passes only when every
input passes; details are concatenated. `any_of` passes if any
input passes (the first passing branch is chosen and its details
returned); on a full failure the failed-branch details are
concatenated with an `any_of[N]:` prefix per branch so the
operator can see why every alternative was rejected.

```rust,ignore
let combined = AssertResult::any_of([
    cpu_quota_satisfied,
    fair_under_contention,
]);
```

Use these to express "either this OR that" without writing the
fold by hand. `merge` remains the right tool when results
accumulate in a loop body.

### Fields

- `passed: bool` -- whether all checks passed.
- `skipped: bool` -- distinguishes a passing result that ran every
  check from one that skipped execution (topology / flag mismatch,
  prerequisite absent). `AssertResult::skip` sets this; `pass` /
  `fail` / `fail_msg` leave it `false`.
- `details: Vec<AssertDetail>` -- structured diagnostic entries; each
  carries a `kind: DetailKind` (`Other`, `Note`, `Skip`, `Temporal`,
  …) plus a human-readable `message: String`. Consumers filter by
  `kind` for routing (failure vs informational note) and read
  `message` for display.
- `stats: ScenarioStats` -- aggregated worker telemetry across all
  cgroups (spread, gaps, migrations, wake latency, iterations).
- `measurements: BTreeMap<String, NoteValue>` -- structured
  per-test measurements keyed by name. Sidecar consumers and
  comparison tooling read this map directly without parsing
  `details` strings, so populate it (via `Verdict::note_value`
  during claim evaluation) for any value a downstream comparison
  needs to lift programmatically.

### Merging

`result.merge(other)` combines two results. If `other.passed` is
false, the merged result is also false. Details and stats are
accumulated:

```rust,ignore
let mut combined = AssertResult::pass();
combined.merge(cgroup_0_result);
combined.merge(cgroup_1_result);
// combined.passed is false if either cgroup failed
// combined.details contains messages from both
```

Stats merging takes worst values across cgroups for spread, gap, wake
latency, and migration ratio. Counters (`total_workers`, `total_cpus`,
`total_migrations`, `total_iterations`) are summed.

For examples of overriding thresholds at the scheduler and per-test
level, see [Customize Checking](../recipes/custom-checking.md).
