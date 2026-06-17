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
- `max_p99_wake_latency_ns`: p99 of all `wake_latencies_ns` samples
  across workers in a cgroup. Populated by work types that block or
  yield and measure wake-to-run latency — most I/O, futex, pipe,
  sleep, and yield-churn variants, including `IoSyncWrite`,
  `IoRandRead`, `IoConvoy`, `Bursty`, `PipeIo`, `FutexPingPong`,
  `CacheYield`, `CachePipe`, `FutexFanOut` (receivers), `Sequence`
  (Sleep / Yield / Io phases), `ForkExit`, `NiceSweep`,
  `AffinityChurn`, `CrossAffinityChurn`, `PolicyChurn`,
  `FanOutCompute`, `MutexContention`, `ThunderingHerd`, `WakeChain`,
  `AsymmetricWaker`, `PriorityInversion`, `ProducerConsumerImbalance`,
  `NumaWorkingSetSweep`, `EpollStorm`, `IdleChurn`. Pure-CPU spin
  variants (`SpinWait`, `Mixed`, `CachePressure`, `PageFaultChurn`,
  `YieldHeavy`, `AluHot`, `SmtSiblingSpin`, `IpcVariance`, and the
  like) record no wake-latency samples — the compute variants sample
  iteration cost instead.
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
   monitor-threshold bundle from `MonitorThresholds::new()`.
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

## Preset thresholds: `AbsoluteThresholds`

`AbsoluteThresholds` is a flat absolute-threshold preset designed for
direct invocation in test bodies, distinct from the merge-tree
threshold config carried by `Assert`. (It is NOT an A/B comparison
against a reference scheduler — every check is an absolute per-run
bound.) Use when a test wants a one-call multi-field check without
engaging the `default_checks → scheduler → test` merge chain.

```rust,ignore
use ktstr::assert::{AbsoluteThresholds, assert_thresholds};

// Sane-default preset: p99 wake under 10ms, p99 iteration cost
// under 1ms, total migrations under 1000, each worker >= 1 work unit.
let r = assert_thresholds(&reports, &AbsoluteThresholds::strict());

// Or build piecewise with explicit thresholds.
let thresholds = AbsoluteThresholds::default()
    .max_p99_wake_latency_ns(5_000_000)
    .min_work_units(100);
let r = assert_thresholds(&reports, &thresholds);
```

Each field is independent — `None` skips that check. The four fields:

- `max_p99_wake_latency_ns` -- pooled p99 across every worker's
  `wake_latencies_ns`. Same semantics as `Assert::max_p99_wake_latency_ns`.
- `max_iteration_cost_p99_ns` -- pooled p99 across every worker's
  `iteration_costs_ns`. Meaningful for compute work types that sample
  iteration cost — `AluHot`, `SmtSiblingSpin`, `IpcVariance`, and a
  `Sequence` whose phases include `WorkPhase::AluHot`; purely blocking
  variants report empty reservoirs and the check is a no-op.
- `max_migrations` -- absolute sum of `migration_count` across
  workers. Distinct from `Assert::max_migration_ratio` (per-iteration
  rate); useful when the test pins a known workload size.
- `min_work_units` -- per-worker floor. One starved worker fails.
  Distinct from `assert_not_starved`'s zero-floor — accepts a
  non-zero threshold so tests can reject "barely made progress" runs.

`assert_thresholds` returns a skip when `reports` is empty (thresholds
against zero samples would silently green-light a broken run that
produced no signal).

The preset composes with the merge-chain path: a test can run
`assert_thresholds` against a worker-report slice AND merge the
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
`Assert::default_checks().verdict()` or `Verdict::new()`.

Test authors reach for one of two compile-mechanical labelers:

1. **Typed field accessors** generated by `#[derive(Claim)]` on stats
   structs (where `stats` is a `CgroupStats` value collected from your
   worker reports):

   ```rust,ignore
   use ktstr::assert::{Assert, Verdict};
   let mut v = Assert::default_checks().verdict();
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

For container claims, comparators bypass scalars. Set claims offer
`empty` / `nonempty` / `contains` / `len_eq` / `len_at_most` /
`len_at_least` / `subset_of` / `disjoint_from`; sequence claims offer
the same set minus `subset_of` / `disjoint_from`.

### Finishing the verdict

`Verdict::into_result()` consumes the accumulator and returns an
`AssertResult` carrying `outcomes` / `passes` / `info_notes` /
`stats` / `measurements`. The terminal verdict is the fold of the
`outcomes: Vec<Outcome>` slot per the four-state lattice (see the
"Verdict outcomes" section below). Compose via `AssertResult::merge`
to combine claim outcomes with `assert_cgroup` / `assert_thresholds` /
`assert_scx_events_clean` results in the same scenario.

## Verdict outcomes

Every assertion produces an `Outcome` — one of four mutually
exclusive variants — and `AssertResult` carries the sequence of
recorded outcomes in `outcomes: Vec<Outcome>`. The terminal
verdict is the fold over that vec per the lattice
**`Fail > Inconclusive > Pass > Skip`**.

| Variant | Meaning | When |
|---|---|---|
| `Pass` | a real check succeeded | the assertion ran and the value satisfied the threshold |
| `Skip(d)` | scenario couldn't run | precondition unmet (topology mismatch, missing resource, in-VM `AssertResult::skip` return) |
| `Inconclusive(d)` | ran but no signal | a ratio gate's denominator legitimately reached zero (zero iterations across all workers under `max_migration_ratio`, zero NUMA pages under `max_slow_tier_ratio`, etc.) so the threshold can't be evaluated |
| `Fail(d)` | a real check failed | the assertion ran and the value violated the threshold |

`Inconclusive` exists for INSTRUMENT-derived denominators
(iteration count, sample count, wall-clock interval) that reached
zero because the workload produced no signal. POLICY-derived
denominators (e.g. NUMA pages under `MemPolicy::Bind`, where the
policy specifies that pages will exist) stay `Fail` on zero — the
policy implies the value should exist, so its absence is a defect
signal, not "couldn't measure."

### Recording outcomes

Producers append to the `outcomes` vec via atomic mutators that
return `&mut Self` for chaining:

- `r.record_pass()` — append `Outcome::Pass`.
- `r.record_skip(reason)` — append `Outcome::Skip(reason)`. Use
  when the scenario can't run.
- `r.record_inconclusive(detail)` — append
  `Outcome::Inconclusive(detail)`. Use when the assertion ran but
  the denominator (or other input) is zero so the threshold can't
  be evaluated.
- `r.record_fail(detail)` — append `Outcome::Fail(detail)`. Use
  when the assertion ran and the value violated the threshold.
- `r.record_outcome(o)` — escape hatch for callers that already
  hold a pre-folded `Outcome`.

Constructors `AssertResult::pass()` / `::skip(reason)` /
`::fail(detail)` seed the vec with the corresponding variant.
`pass()` is zero-allocation (empty vec; the merge identity).

### Reading the verdict

- `r.outcome()` — folds the vec into a single `Outcome` per the
  lattice. Use when matching on the terminal verdict.
- `r.outcome_ref()` — same fold returning `OutcomeRef<'_>` that
  borrows the payload in place.
- `r.is_pass()` — true iff no Fail / Inconclusive was recorded
  and the stream is not all-Skip. An empty stream (the `pass()`
  zero-state, which is the merge identity element) returns
  **true**; a stream containing at least one real Pass marker
  alongside no Fail / Inconclusive also returns **true**;
  Inconclusive (a zero-denominator gate didn't pass — it
  couldn't evaluate) and all-Skip (the scenario didn't run) both
  return **false**.
- `r.is_fail()` — true iff at least one Fail was recorded.
- `r.is_inconclusive()` — true iff at least one Inconclusive was
  recorded and no Fail was recorded (Fail dominates).
- `r.is_skip()` — true iff the stream is non-empty and every
  recorded outcome is Skip.
- Per-variant payload iterators: `r.failure_details()`,
  `r.inconclusive_details()`, `r.skip_details()`.

### Merge precedence

`AssertResult::merge` concatenates the `outcomes` vecs. The
terminal-verdict semantics fall out of the per-variant fold:

- any `Fail` in either operand → merged result is `Fail` (Fail
  dominates).
- absent `Fail`, any `Inconclusive` → merged is `Inconclusive`.
- absent both, at least one `Pass` → merged is `Pass`.
- all-Skip on both → merged is `Skip`.

Same-discriminant ties (e.g. two `Fail` outcomes from different
checks) preserve the LEFT operand's payload, so the first recorded
diagnostic surfaces in the terminal verdict.

### CI gate patterns

```rust,ignore
// "Real pass" — the assertion ran and succeeded.
if r.is_pass() {
    // ship
}

// "Real fail" — the assertion ran and violated a threshold.
if r.is_fail() {
    // block release; surface r.failure_details()
}

// "Couldn't evaluate" — scenario skipped (precondition unmet) or
// the assertion ran without enough signal (zero-denominator).
if r.is_skip() || r.is_inconclusive() {
    // treat as "no verdict"; review the run inputs
}
```

Match-on-outcome is also fine:

```rust,ignore
match r.outcome() {
    Outcome::Pass => { /* ship */ }
    Outcome::Fail(d) => { /* block; surface d */ }
    Outcome::Inconclusive(d) => { /* operator triages */ }
    Outcome::Skip(d) => { /* not a verdict; record for stats */ }
}
```

### `any_of` summary

`AssertResult::any_of(...)` (disjunction) synthesizes a terminal
verdict from N branches per the same lattice. The synthesis
order is: any Pass branch wins (the first passing branch's result
is returned); absent any Pass, if any branch failed the result is
`Fail`; absent Pass and Fail, if any branch is `Inconclusive` the
result is `Inconclusive`; otherwise (all-Skip) the result is
`Skip`. The failure-path summary line reports
`X failed, Y inconclusive, Z skipped of N branches` so the
operator sees the disposition mix without re-walking the per-branch
details.

## Constants

- `Assert::NO_OVERRIDES` -- identity for `merge`; every field is `None`,
  so it overrides nothing. Use this const for spread-into-struct-literal
  composition (e.g. `Assert { not_starved: Some(true), ..Assert::NO_OVERRIDES }`).
  This is not "no checks" -- when used as a per-test or per-scheduler
  `assert`, the runtime chain still applies the merge of
  `default_checks() -> scheduler -> test`.
- `Assert::default_checks()` -- `const fn` returning the same value as
  `NO_OVERRIDES`. The method-style entry point that pairs with
  `.verdict()` and builder setters (e.g.
  `Assert::default_checks().check_not_starved().max_gap_ms(100).verdict()`).
- `.with_monitor_defaults()` -- populates the monitor-threshold
  bundle (`max_imbalance_ratio`, `max_local_dsq_depth`,
  `fail_on_stall`, `sustained_samples`, `max_fallback_rate`,
  `max_keep_last_rate`) from `MonitorThresholds::new()`. Tests that
  want stall + imbalance protection must opt in via this method or
  set the fields directly.

## AssertResult

`AssertResult` carries pass/fail status, diagnostic messages, and
aggregated statistics from a scenario run.

### Construction

- `AssertResult::pass()` -- creates a passing result with an empty
  `outcomes` vec and default stats. The empty vec is the Pass
  identity; merging anything into it yields the same lattice fold
  as recording outcomes directly.
- `AssertResult::skip(reason)` -- seeds `outcomes` with one
  `Outcome::Skip(detail)` carrying the reason. Use when a scenario
  cannot run under the current topology or flag combination but is
  not a failure. `is_pass()` reads false on the result (Skip is
  not Pass — the scenario didn't run).
- `AssertResult::inconclusive(detail)` -- seeds `outcomes` with one
  `Outcome::Inconclusive(detail)`. Use when the assertion ran but
  the denominator was zero so the threshold can't be evaluated.
  `is_pass()` reads false; `is_inconclusive()` reads true.
- `AssertResult::fail(detail)` -- failing result carrying a single
  `AssertDetail`. Mirrors `pass` / `skip` / `inconclusive` for the
  failure axis.
- `AssertResult::fail_msg(msg)` -- shortcut for the common case
  where the failure is a plain diagnostic message tagged
  `DetailKind::Other`.

### Mutation and inspection

- `result.note(msg)` -- append an informational annotation to
  `AssertResult::info_notes` (the structurally-separate context
  stream, distinct from failure-carrying `outcomes`). Does NOT
  alter the terminal verdict — a note is context, not a verdict.
  Returns `&mut Self` so calls chain.
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
- `result.is_pass()` / `is_fail()` / `is_inconclusive()` /
  `is_skip()` -- four-state verdict accessors over the
  `Fail > Inconclusive > Pass > Skip` lattice. See
  [Verdict outcomes](#verdict-outcomes) for the full table and
  CI-gate patterns.

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

- `outcomes: Vec<Outcome>` -- per-claim outcome entries; each
  carries the claim shape, comparator, and pass/fail/skip status.
  Verdict is COMPUTED via `is_pass()` / `is_fail()` /
  `is_inconclusive()` / `is_skip()` from this vec (no separate
  `passed: bool` / `skipped: bool` field exists).
- `passes: Vec<PassDetail>` -- recorded pass details (capped at
  `MAX_RECORDED_PASSES`). Consumed by the auto-repro renderer (which
  iterates both `outcomes` and `passes`) to surface passing context
  alongside failing assertions.
- `stats: ScenarioStats` -- aggregated worker telemetry across all
  cgroups (spread, gaps, migrations, wake latency, iterations).
- `measurements: BTreeMap<String, NoteValue>` -- structured
  per-test measurements keyed by name. Sidecar consumers and
  comparison tooling read this map directly without parsing
  failure-message strings, so populate it (via `Verdict::note_value`
  during claim evaluation) for any value a downstream comparison
  needs to lift programmatically.
- `info_notes: Vec<InfoNote>` -- informational notes (distinct
  from outcomes/passes) used by `verdict.note(...)`.

### Merging

`result.merge(other)` combines two results. Outcomes, passes,
info notes, and stats are accumulated; the terminal verdict
follows the `Fail > Inconclusive > Pass > Skip` lattice (see
[Verdict outcomes: Merge precedence](#merge-precedence)):

```rust,ignore
let mut combined = AssertResult::pass();
combined.merge(cgroup_0_result);
combined.merge(cgroup_1_result);
// combined.is_fail() returns true if any cgroup failed
// combined.is_inconclusive() returns true if any cgroup was
//   inconclusive AND none failed (Fail dominates Inconclusive)
// combined.failure_details() iterates concatenated failure notes
// combined.inconclusive_details() iterates inconclusive payloads
```

Stats merging takes worst values across cgroups for spread, gap, wake
latency, and migration ratio. Counters (`total_workers`, `total_cpus`,
`total_migrations`, `total_iterations`) are summed.

For examples of overriding thresholds at the scheduler and per-test
level, see [Customize Checking](../recipes/custom-checking.md).

## Phase-aware checks

A scenario built from a `vec![Step, Step, ...]` runs each Step in
sequence with the framework publishing the active phase as the
test progresses. Captures fired during a Step (periodic samples,
`Op::WatchSnapshot` trips, on-demand `Op::CaptureSnapshot` calls)
stamp with the phase active at capture time, and assertions
constructed under a Step's hold auto-stamp with the matching
phase label.

### Encoding

Phases use a 1-indexed convention so the pre-first-Step settle
window owns the unambiguous `0` slot:

- `Phase::BASELINE` (inner `u16` = `0`) — settle window before
  Step 0 starts. Captures fired during this window (boot, scheduler
  attach, pre-Step warmup) land here.
- `Phase::step(k)` — the `k`-th scenario Step (0-indexed in the
  scenario `vec`, 1-indexed in the inner `u16`). `Phase::step(0)`
  is the first Step, `Phase::step(1)` is the second, etc.

Labels render as `"BASELINE"` and `"Step[k]"` (the 0-indexed Step
number embedded in brackets) across the structured sidecar JSON,
the formatted timeline diagnostic, and the per-assertion
`detail.phase` field, so the same identifier appears wherever the
operator looks.

### Looking up phase metrics on `ScenarioStats`

```rust,ignore
// Phase-aware accessor by 1-indexed encoded value.
let baseline = r.stats.phase(0).expect("BASELINE always populated");

// Scenario-side 0-indexed Step accessor (preferred for the
// "I want metrics for the k-th Step I wrote" case).
let step_0 = r.stats.step(0).expect("Step 0 ran");
let step_1 = r.stats.step(1).expect("Step 1 ran");

// Single-metric shortcut.
let throughput = r.stats.step_metric(0, "throughput");

// Gate on "scenario advanced past BASELINE" before assuming any
// Step-phase bucket exists — a scenario that bailed in setup or
// declared zero Steps returns None from every step()/step_metric()
// lookup and the test either panics on .expect(...) or passes
// vacuously.
anyhow::ensure!(
    r.stats.has_steps(),
    "scenario produced no Step phases — declare a Step or use \
     stats.phase(0) for BASELINE",
);
```

### `bucket.expect_metric` for actionable failures

`PhaseBucket::expect_metric` panics with a diagnostic naming the
bucket's `step_index`, `label`, `sample_count`, and the set of
metric keys actually present — the operator can distinguish
"phase carried 0 samples" (`sample_count == 0`) from "metric name
typo" (positive `sample_count` but the key isn't in `metrics`)
without re-reading the bucket by hand:

```rust,ignore
let bucket = r.stats.step(0).expect("Step 0 phase bucket");
let throughput = bucket.expect_metric("throughput");
```

### Auto-stamped assertion phase

Every `AssertDetail` / `PassDetail` / `InfoNote` constructed
inside a Step's hold auto-stamps its `phase` field with that
Step's label. The structured sidecar entry the operator inspects
post-run reads `detail.phase: "Step[k]"` for failures inside
Step `k`, `detail.phase: "BASELINE"` for failures during the
settle window. Explicit overrides chain via the existing
`.with_phase("custom_label")` builder when the auto-stamp is
not appropriate (typically only in synthetic test fixtures).

The wire format on `step_index` u16 fields is unchanged across
this surface — `Phase` is `#[serde(transparent)]` over `u16`, so
sidecar JSON / typeshare consumers see the same scalar field
they saw before the typed wrapper landed.
