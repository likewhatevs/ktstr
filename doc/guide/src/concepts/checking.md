# Checking

ktstr judges scheduler behavior through two channels: worker-side
telemetry (every worker process reports what happened to it) and
host-side monitoring (the [monitor](../architecture/monitor.md) reads
guest kernel state from outside). Both channels always *measure*;
nothing *asserts* until the test opts in — a test with no checking
attributes passes as long as the VM boots and the scenario completes.

Which API to reach for:

- **`#[ktstr_test]` attributes** — cover most tests: `not_stuck`,
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

- **Zero work units** (`not_stuck`) — any worker with no measured
  work fails: `tid N made no progress (0 work units)`.
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
<!-- captured: cargo ktstr test --kernel 7.0 (throughput_gate demo test) | ktstr 0.23.0 | kernel 7.0.14 -->
```

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
- **Stuck-task detection** — `rq_clock` not advancing on a CPU with
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
| `fail_on_rq_clock_stuck` | true | Fail when `rq_clock` does not advance on a CPU with runnable tasks. Idle CPUs (NOHZ) and preempted vCPUs are exempt. |
| `sustained_samples` | 5 | At ~100ms sample interval, requires ~500ms of sustained violation. Filters transient spikes from cpuset reconfiguration. |
| `max_fallback_rate` | 200.0/s | `select_cpu_fallback` events per second across all CPUs. Sustained rate indicates systematic `select_cpu` failure. |
| `max_keep_last_rate` | 100.0/s | `dispatch_keep_last` events per second across all CPUs. Sustained rate indicates the scheduler keeps reusing the previous dispatch target instead of making progress through the normal path. |

Every monitor threshold uses the `sustained_samples` window — a
violation must persist for N consecutive samples before it counts.

## NUMA checks

For workers with a [`MemPolicy`](topology.md#memory-policy), three thresholds
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

`not_stuck = true` also enables the built-in fairness and stuck-gap
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

<div class="kt-figure"><svg width="700" height="250" viewBox="0 0 700 250" role="img" aria-label="Configuration merge cascade: three Assert layers merge last-Some-wins — baseline with all fields None, then the scheduler's assert, then per-test #[ktstr_test] attributes — yielding the merged Assert used for worker checks; execute_steps_with bypasses the merge with an explicit Assert">
  <defs><marker id="kt-arr6" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg)"/></marker></defs>
  <g opacity=".7">
    <rect x="40" y="14" width="280" height="42" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="56" y="34" font-size="12" font-weight="700" fill="var(--fg)">baseline</text>
    <text x="56" y="50" font-size="9.5" fill="var(--fg)" opacity=".8">every field None — inherit</text>
  </g>
  <path d="M180 56 L 180 72" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#kt-arr6)"/>
  <text x="190" y="70" font-size="9" fill="var(--fg)" opacity=".55">override</text>
  <rect x="40" y="74" width="280" height="42" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
  <text x="56" y="94" font-size="12" font-weight="700" fill="var(--fg)">scheduler's assert</text>
  <text x="56" y="110" font-size="9.5" fill="var(--fg)" opacity=".75">scheduler-wide bound</text>
  <path d="M180 116 L 180 132" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#kt-arr6)"/>
  <text x="190" y="130" font-size="9" fill="var(--fg)" opacity=".55">override</text>
  <rect x="40" y="134" width="280" height="42" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.4"/>
  <text x="56" y="154" font-size="12" font-weight="700" fill="var(--kt-accent)">per-test #[ktstr_test] attributes</text>
  <text x="56" y="170" font-size="9.5" fill="var(--fg)" opacity=".8">max_gap_ms, max_spread_pct, …</text>
  <path d="M180 176 L 180 192" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#kt-arr6)"/>
  <rect x="40" y="194" width="280" height="42" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.6"/>
  <text x="56" y="214" font-size="12" font-weight="700" fill="var(--kt-accent)">merged Assert → worker checks</text>
  <text x="56" y="230" font-size="9.5" fill="var(--fg)" opacity=".8">last Some wins</text>
  <rect x="430" y="74" width="240" height="64" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.3" stroke-dasharray="5 4"/>
  <text x="446" y="98" font-size="11" font-weight="700" fill="var(--fg)">execute_steps_with</text>
  <text x="446" y="115" font-size="9.5" fill="var(--fg)" opacity=".75">(ctx, steps, Some(&amp;assert))</text>
  <text x="446" y="130" font-size="9.5" fill="var(--fg)" opacity=".75">explicit Assert</text>
  <path d="M430 122 C 384 154, 362 196, 326 214" stroke="var(--fg)" stroke-width="1.3" fill="none" marker-end="url(#kt-arr6)"/>
  <text x="392" y="200" font-size="9.5" fill="var(--fg)" opacity=".7">bypasses the merge</text>
  <text x="40" y="248" font-size="9.5" fill="var(--fg)" opacity=".6">enforce_monitor_thresholds is sticky: once any layer sets it, it stays set.</text>
</svg></div>

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

<div class="kt-figure"><svg width="700" height="216" viewBox="0 0 700 216" role="img" aria-label="Verdict lattice and its projection to process exit codes. A result's terminal verdict is the fold over all outcomes, most severe first: Fail, then Inconclusive, then Pass, then Skip. Default projection: Fail exits 1, Inconclusive exits 2, and Pass and Skip exit 0. Two modifiers bend the mapping: expect_err makes a clean Pass exit 1, and allow_inconclusive makes an Inconclusive exit 0; under --no-skip-mode a Skip exits 1 instead of 0.">
  <defs><marker id="kt-arr9" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg)"/></marker></defs>
  <text x="28" y="16" font-size="11" font-weight="700" fill="var(--kt-accent)">terminal verdict = fold over all outcomes (most severe wins)</text>
  <text x="28" y="32" font-size="10" fill="var(--fg)" opacity=".8">Fail &gt; Inconclusive &gt; Pass &gt; Skip</text>
  <g fill="var(--fg)">
    <rect x="28" y="42" width="214" height="30" rx="8" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.6"/>
    <text x="44" y="61" font-size="11.5" font-weight="700" fill="var(--kt-accent)">Fail</text>
    <text x="230" y="61" font-size="9" text-anchor="end" opacity=".75">ran · violated bound</text>
    <rect x="28" y="80" width="214" height="30" rx="8" fill="none" stroke="var(--kt-rule)" stroke-width="1.3"/>
    <text x="44" y="99" font-size="11.5" font-weight="700">Inconclusive</text>
    <text x="230" y="99" font-size="9" text-anchor="end" opacity=".75">ran · no signal</text>
    <rect x="28" y="118" width="214" height="30" rx="8" fill="none" stroke="var(--kt-rule)" stroke-width="1.3"/>
    <text x="44" y="137" font-size="11.5" font-weight="700">Pass</text>
    <text x="230" y="137" font-size="9" text-anchor="end" opacity=".75">ran · satisfied bound</text>
    <rect x="28" y="156" width="214" height="30" rx="8" fill="none" stroke="var(--kt-rule)" stroke-width="1.3" opacity=".8"/>
    <text x="44" y="175" font-size="11.5" font-weight="700" opacity=".85">Skip</text>
    <text x="230" y="175" font-size="9" text-anchor="end" opacity=".7">couldn't run · precondition</text>
  </g>
  <path d="M242 57 L 458 46" stroke="var(--kt-accent)" stroke-width="1.4" fill="none" marker-end="url(#kt-arr9)"/>
  <path d="M242 95 L 458 102" stroke="var(--fg)" stroke-width="1.3" fill="none" marker-end="url(#kt-arr9)"/>
  <path d="M242 133 L 458 158" stroke="var(--fg)" stroke-width="1.3" fill="none" marker-end="url(#kt-arr9)"/>
  <path d="M242 171 L 458 162" stroke="var(--fg)" stroke-width="1.3" fill="none" marker-end="url(#kt-arr9)"/>
  <path d="M242 128 Q 360 74 458 50" stroke="var(--kt-accent)" stroke-width="1.2" fill="none" stroke-dasharray="5 4" opacity=".85" marker-end="url(#kt-arr9)"/>
  <text x="300" y="70" font-size="9" fill="var(--kt-accent)" opacity=".9">expect_err · clean pass → 1</text>
  <path d="M242 100 Q 365 150 458 160" stroke="var(--fg)" stroke-width="1.2" fill="none" stroke-dasharray="5 4" opacity=".75" marker-end="url(#kt-arr9)"/>
  <text x="300" y="140" font-size="9" fill="var(--fg)" opacity=".8">allow_inconclusive → 0</text>
  <g>
    <rect x="460" y="30" width="180" height="34" rx="8" fill="none" stroke="var(--kt-rule)" stroke-width="1.4"/>
    <text x="476" y="52" font-size="11" font-weight="700" fill="var(--fg)">exit 1 · Fail</text>
    <rect x="460" y="86" width="180" height="34" rx="8" fill="none" stroke="var(--kt-rule)" stroke-width="1.4"/>
    <text x="476" y="108" font-size="11" font-weight="700" fill="var(--fg)">exit 2 · Inconclusive</text>
    <rect x="460" y="142" width="180" height="34" rx="8" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)" stroke-width="1.4"/>
    <text x="476" y="164" font-size="11" font-weight="700" fill="var(--fg)">exit 0 · Pass / Skip</text>
  </g>
  <text x="460" y="196" font-size="9" fill="var(--fg)" opacity=".65">--no-skip-mode: Skip → exit 1</text>
</svg></div>

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

- **`claim_better`** — the comparison primitive for "candidate beats
  baseline" gates (a scheduler vs EEVDF, this run vs a recorded
  number). It looks up the metric's polarity in the registry, so you
  never write the wrong direction:

  ```rust,ignore
  // wakeup latency is lower-better; 60 vs 50 fails, correctly
  v.claim_better(BuiltinMetric::WakeupP99LatencyUs, cand).than(base);
  // require a 10% margin, not just any improvement
  v.claim_better(BuiltinMetric::TaobenchTotalQps, cand).than_by(base, 0.10);
  ```

  An unregistered metric yields Inconclusive, never a silent pass.
- **Claim variants and measurements** — `claim_present!(v, opt)`
  fails loudly on `None` instead of vacuously passing; `claim_set` /
  `claim_seq` assert membership, length, and subset bounds over
  collections;
  `note(msg)` records free-text context and `note_value(key, val)`
  a typed measurement into `AssertResult::measurements` — triage
  payload attached to the verdict, not a metric (nothing recorded
  through claims or notes reaches the stats sidecar or `perf-delta`;
  see [Assertable Metrics](../reference/assertable-metrics.md)).
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
[Phases](ops.md#phases). To assert on your scheduler's *own*
counters — scx_stats fields, BPF globals and maps — see
[Projections and Temporal
Assertions](../writing-tests/temporal-assertions.md).
