# Checking

ktstr judges scheduler behavior through two channels: worker-side
telemetry (every worker process reports what happened to it) and
host-side monitoring (the [monitor](../architecture/monitor.md) reads
guest kernel state from outside). Both channels always *measure*;
nothing *asserts* until the test opts in — a test with no checking
attributes passes as long as the VM boots and the scenario completes.

Which API to reach for:

- **`#[ktstr_test]` attributes** — cover most tests: `not_starved`,
  `max_gap_ms`, `max_spread_pct`, `min_iteration_rate`, and every
  other threshold below has an attribute (see the
  [macro reference](../writing-tests/ktstr-test-macro.md)).
- **`Verdict` + `claim!`** — labeled assertions on values you compute
  inside a custom scenario body.
- **`AbsoluteThresholds`** — a one-call multi-field bound check
  against collected reports, bypassing the config merge.
- **`assert_scx_events_clean`** — bounds on SCX event counters
  ("no fallbacks fired").

## Worker checks

After each scenario, ktstr collects a
[`WorkerReport`](../architecture/workers.md) from every worker and
runs the opted-in checks against them:

- **Starvation** (`not_starved`) — any worker with zero work units
  fails: `tid N starved (0 work units)`.
- **Scheduling gaps** (`max_gap_ms`) — the longest wall-clock gap
  observed at work-unit checkpoints. A violation renders as
  `tid N stuck Xms on cpuY at +Zms (threshold Nms)`.
- **Fairness** (`max_spread_pct`) — workers in one cgroup should get
  similar CPU time; the spread (max off-CPU% − min off-CPU%) must
  stay below the bound.
- **Cpuset isolation** (`isolation`) — workers may only run on CPUs
  in their assigned cpuset; any excursion fails.
- **Throughput** — `max_throughput_cv` bounds the coefficient of
  variation of per-worker work rate (some workers quietly slower);
  `min_work_rate` sets an absolute floor (all workers equally slow).
- **Benchmarking** — `max_p99_wake_latency_ns` and
  `max_wake_latency_cv` bound wake-to-run latency for work types
  that block and measure it (see [Work Types](work-types.md) for
  which do); `min_iteration_rate` floors outer-loop iterations per
  second per worker.

## The loop, end to end

A test sets a threshold, the run violates it, the failure output
names the check, the value, and the bound:

```rust,ignore
#[ktstr_test(
    scheduler = MY_SCHED,
    llcs = 1, cores = 2, threads = 1,
    min_iteration_rate = 50_000_000.0,  // deliberately unreachable floor
)]
fn throughput_gate(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_a").workers(1).cpuset(CpusetSpec::disjoint(0, 2)),
        CgroupDef::named("cg_b").workers(1).cpuset(CpusetSpec::disjoint(1, 2)),
    ])
}
```

<!-- captured: cargo ktstr test --kernel 7.0 (throughput_gate demo test) | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr test --kernel 7.0</span></div>

<pre><span class="t-b">ktstr_test 'throughput_gate' [sched=scx-ktstr] [topo=1n1l2c1t] failed:</span>
<span class="t-red">  worker 71 iteration rate 41903.3/s below floor 50000000.0/s</span>
<span class="t-red">  worker 73 iteration rate 37834.5/s below floor 50000000.0/s</span>

--- stats ---
2 workers, 4 cpus, 2 migrations, worst_spread=0.0%, worst_gap=21ms
  cg0: workers=1 cpus=2 spread=0.0% gap=10ms migrations=1 iter=209600
  cg1: workers=1 cpus=2 spread=0.0% gap=21ms migrations=1 iter=189252
...
--- monitor ---
samples=41 max_imbalance=2.00 max_dsq_depth=0 stuck=0
avg: imbalance=1.32 nr_running/cpu=1.2 dsq/cpu=0.0
events: fallback=0 (0.0/s) keep_last=210 (52.5/s) offline=0
...
<span class="t-grn">verdict: monitor OK</span></pre></div>

Both channels report: the worker check that tripped, and the monitor
verdict that did not. The full failure anatomy — timeline, scheduler
log, dump sections — is in
[Reading Failure Output](../running-tests/failures.md).

## Monitor checks

The host-side monitor samples guest per-CPU runqueue state (via BTF
offsets, no guest instrumentation) roughly every 100ms and evaluates:

- **Imbalance ratio** — `max(nr_running) / max(1, min(nr_running))`
  across CPUs.
- **Local DSQ depth** — per-CPU dispatch queue depth.
- **Stall detection** — `rq_clock` not advancing on a CPU with
  runnable tasks; idle CPUs and preempted vCPUs are exempt.
- **Event rates** — `select_cpu_fallback` and `dispatch_keep_last`
  counters per second.

Monitor violations always land in the failure report's `--- monitor
---` section, but they flip the test result only when the test
enforces them — set the corresponding attributes, call
`.with_monitor_defaults()` on an `Assert`, or set
`enforce_monitor_thresholds`. A monitor that produced no usable
signal (empty samples, uninitialized guest memory) reports
*inconclusive*, never a silent pass — a CI gate can always tell
"verified OK" from "never measured".

The defaults `with_monitor_defaults()` applies:

| Threshold | Default | Rationale |
|---|---|---|
| `max_imbalance_ratio` | 4.0 | `max(nr_running) / max(1, min(nr_running))` across CPUs (denominator clamped so an all-idle sample does not divide by zero). Lower values (2-3) false-positive during cpuset transitions. |
| `max_local_dsq_depth` | 50 | Per-CPU dispatch queue overflow. Sustained depth above this means the scheduler is not consuming dispatched tasks. |
| `fail_on_stall` | true | Fail when `rq_clock` does not advance on a CPU with runnable tasks. Idle CPUs (NOHZ) and preempted vCPUs are exempt. |
| `sustained_samples` | 5 | At ~100ms sample interval, requires ~500ms of sustained violation. Filters transient spikes from cpuset reconfiguration. |
| `max_fallback_rate` | 200.0/s | `select_cpu_fallback` events per second across all CPUs. Sustained rate indicates systematic `select_cpu` failure. |
| `max_keep_last_rate` | 100.0/s | `dispatch_keep_last` events per second across all CPUs. Sustained rate indicates dispatch starvation. |

Every monitor threshold uses the `sustained_samples` window — a
violation must persist for N consecutive samples before it counts.

## NUMA checks

For workers with a [`MemPolicy`](mem-policy.md), three thresholds
gate page placement:

- **`min_page_locality`** — minimum fraction of pages on the
  expected NUMA nodes (the cgroup's cpuset nodes, derived at
  evaluation time). Zero observed pages counts as zero locality, not
  a vacuous pass.
- **`max_cross_node_migration_ratio`** — bound on migrated pages
  relative to allocated pages (from `/proc/vmstat` deltas).
- **`max_slow_tier_ratio`** — bound on the fraction of pages landing
  on memory-only (CXL-tier) nodes.

## Default thresholds

`not_starved = true` also enables the built-in fairness and gap
checks at these defaults:

| Check | Release | Debug |
|---|---|---|
| Scheduling gap | 2000 ms | 3000 ms |
| Fairness spread | 15% | 35% |

Debug builds run with higher scheduling overhead, so thresholds are
relaxed.

## How configuration merges

`Assert` is the threshold-config struct; every field is an `Option`
where `None` means "inherit". Three layers merge, last-`Some` wins:
the baseline (all `None`), then the scheduler's `assert`, then the
per-test attributes — so a scheduler-wide bound applies to every
test and any single test can override or disable it.
`enforce_monitor_thresholds` is the one sticky field: once any layer
sets it, it stays set. Worked override recipes live in
[Customize Checking](../recipes/custom-checking.md).

`execute_steps_with(ctx, steps, Some(&assert))` bypasses the merged
config with an explicit `Assert` for that scenario's worker checks.

## Verdicts and outcomes

Every assertion produces one of four outcomes, and a result's
terminal verdict is the fold over all of them, most severe first:
**`Fail > Inconclusive > Pass > Skip`**.

| Outcome | Meaning |
|---|---|
| `Pass` | the assertion ran and the value satisfied the bound |
| `Fail` | the assertion ran and the value violated the bound |
| `Inconclusive` | the assertion ran but had no signal to evaluate |
| `Skip` | the scenario couldn't run (unmet precondition) |

`Inconclusive` exists for instrument-derived denominators — a ratio
whose denominator (iterations, samples, wall-clock interval)
legitimately reached zero because the workload produced no signal.
Policy-derived denominators stay `Fail` on zero: under
`MemPolicy::Bind` the policy says pages will exist, so their absence
is a defect, not "couldn't measure".

CI gates read the verdict through four accessors:

```rust,ignore
if r.is_pass() { /* ship */ }
if r.is_fail() { /* block; surface r.failure_details() */ }
if r.is_skip() || r.is_inconclusive() { /* no verdict — triage */ }
```

`is_pass()` is deliberately strict: inconclusive and all-skip both
read `false`.

## Beyond attributes {#verdict-the-claim-accumulator}

- **`Verdict` + `claim!`** — the claim accumulator for custom
  scenario bodies. Labels come from the code itself
  (`stringify!`-derived), so they cannot drift from the value they
  describe:

  ```rust,ignore
  let mut v = Assert::default_checks().verdict();
  stats.claim_max_gap_ms(&mut v).at_most(100);
  claim!(v, iter_delta).at_least(1000);
  let result = v.into_result();
  ```

- **`AbsoluteThresholds`** — flat per-run bounds
  (`max_p99_wake_latency_ns`, `max_iteration_cost_p99_ns`,
  `max_migrations`, `min_work_units`) checked in one call:
  `assert_thresholds(&reports, &AbsoluteThresholds::strict())`.
  Empty report slices return a skip rather than a vacuous pass.
- **`assert_scx_events_clean(events, bound)`** — SCX event counters
  under a cap (`None` = exactly zero); negative counts always fail.
- **Composition** — `AssertResult::merge` accumulates results in a
  loop; `all_of` / `any_of` fold sibling results as AND / OR.

Signatures, comparators, and construction details are in the
[`ktstr::assert` rustdoc](https://ktstr.dev/rustdoc/ktstr/assert/index.html).
For phase-scoped checks over a stepped scenario, see
[Phases](ops.md#phases) and
[Temporal Assertions](../writing-tests/temporal-assertions.md).
