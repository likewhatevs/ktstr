# Assertable Metrics

Every regression comparison — `cargo ktstr perf-delta` and the
per-test [`PerfDeltaAssertion`](#perfdeltaassertion-how-to) gate — is
driven by the **metric registry**: the static `ktstr::stats::METRICS`
table. Each entry carries a metric's name, its regression
**polarity**, its aggregation **kind**, the **dual-gate**
significance thresholds, and a display unit. This chapter explains
those fields, how to enumerate the live catalog, which workloads emit
which metric families, and how to pin a per-test regression gate.

## The catalog: `stats list-metrics`

The authoritative, always-current catalog is the command output — it
enumerates the registry directly, so it never drifts from the code:

```sh
cargo ktstr stats list-metrics          # text table
cargo ktstr stats list-metrics --json   # machine-readable (includes kind + every field)
```

<!-- captured: cargo ktstr stats list-metrics | ktstr 0.23.0 | host-side (no VM) -->
```text
 NAME                                    POLARITY       DEFAULT_ABS  DEFAULT_REL  UNIT
 worst_spread                            lower          5            0.25         %
 worst_gap_ms                            lower          500          0.5          ms
 total_migrations                        lower          2            0.3
 worst_migration_ratio                   lower          0.05         0.2
 max_imbalance_ratio                     lower          1            0.25         x
...
 worst_p99_wake_latency_us               lower          50           0.25         µs
 worst_median_wake_latency_us            lower          20           0.25         µs
...
 iteration_rate                          higher         1            0.3          iter/s
 total_iterations                        higher         2            0.1
```

`list-metrics` reads only the static registry; it needs no sidecar
pool. Which of these metrics a *particular* run actually carries
depends on the emitting workload — see
[Workload → emitted metrics](#workload--emitted-metrics).
(`cargo ktstr stats list-values` enumerates the pool's filter
dimensions — kernels, commits, schedulers, topologies, work types —
not its metric keys, so it cannot answer which metrics are present.)

## Registry fields

- **name** — the metric key (e.g. `worst_spread`, `worst_gap_ms`,
  `sched_count_per_sec`). This is the string a `PerfDeltaAssertion`
  names and the key `perf-delta` reports on.
- **polarity** — the regression direction:
  - `LowerBetter` — an increase is a regression (latency, spread).
  - `HigherBetter` — a decrease is a regression (throughput,
    iterations).
  - `Informational` — directionless: a change is shown but never
    counted as a regression or improvement and never gates the exit.
  - `TargetValue(t)` / `Unknown` — also exist (rendered `target(t)` /
    `unknown` by `list-metrics`) but no registered metric uses them
    today.
- **kind** — how per-sample readings fold into the run-level value:
  `Counter` (sum), `Peak` (max-of-max), `Gauge` (average, last, or
  max, per metric), `Rate` (re-derived ratio), plus phase-aware kinds
  such as `DeltaSum` that fold pre-deltaed per-phase readings. The
  kind decides whether the cross-run fold is a mean, a max, or a
  re-derived ratio.
- **default_abs** / **default_rel** — the **dual gate**. A move counts
  as a confident regression only when it clears both the absolute
  floor (`default_abs`, in the metric's units) and the relative
  threshold (`default_rel`, a fraction). The absolute floor's role
  depends on the metric's dynamic range:
  - **Scale-bounded** metrics (fractions, ratios, `%` spread, `ms`/`µs`
    latencies) use `default_abs` as a fixed unit-scale noise floor — a
    sub-unit move is immaterial regardless of its relative size.
  - **Scale-varying** metrics (`*_per_sec` rates, `ops/s`, `req/s`, raw
    counts) can span orders of magnitude across workloads, so a fixed
    floor would mask a large relative regression on a low-throughput
    workload. For these, `default_abs` is only a near-idle activity
    guard and `default_rel` carries materiality — a 40 % drop is
    flagged whether the baseline is 50/s or 50000/s.

  `perf-delta --threshold PCT` / `--policy FILE` override the relative
  gate; the absolute gate is per-metric.
- **display_unit** — the unit rendered in tables (`ms`, `/s`, `ns`, …).

## Workload → emitted metrics

A metric only appears in a comparison if the run actually emitted it.

| Family | Example metrics | Emitted by | Present when |
|---|---|---|---|
| Spread / gap | `worst_spread`, `worst_gap_ms` | every scenario (scheduling-latency capture) | always |
| Iteration throughput | `total_iterations`, `worst_iterations_per_cpu_sec` | compute / spin workloads | the workload iterates; the `*_per_cpu_sec` form is overcommit-invariant |
| schedstat counters / rates | `total_run_delay_ns_per_sched`, `total_ttwu_count`, `sched_count_per_sec` | schedstat sampling over the run | schedstat capture enabled |
| IRQ / pressure | `avg_irq_util`, `total_irq_pressure_us`, `max_cgroup_psi_irq_avg10` | IRQ-heavy scenarios, periodic host-pressure capture | those captures ran |
| NUMA locality | `worst_page_locality`, `worst_cross_node_migration_ratio` | NUMA-aware scenarios | multi-node topology |
| Built-in benchmark workloads | `sched_delay_msg_us`, `taobench_total_qps`, `schbench_loop_count` | the in-process `Schbench` / `Taobench` work types | a built-in `Schbench` / `Taobench` workload ran (not the `schbench`/`fio` binary payloads, which emit their own JSON keys) |

Not every registry name can back a gate: `perf-delta --must-fail`
rejects unknown names, internal rate components, per-phase-only
metrics, and — without `--noise-adjust` — whole-run distribution
metrics and informational metrics, up front rather than silently
never firing.

## PerfDeltaAssertion how-to

A `PerfDeltaAssertion` is a per-test performance-regression gate. It
is **inert during a normal `cargo ktstr test` run** (the in-VM verdict
never consults it) and **active only under `cargo ktstr perf-delta
--noise-adjust`**, which serializes the declaration into the sidecar
and enforces it host-side. Plain (scalar) `perf-delta` does not
evaluate declared gates — gating on a single run would flip CI on
noise, so only the multi-run `--noise-adjust` path (Welch / disjoint-
band separation) is a sound basis. Declaring a gate **requires
`performance_mode`** (checked by the macro at compile time and by test
discovery at run time).

A declaration names a registry metric and overrides, for this test,
the gate that decides a confident regression on it. It layers on top
of the `--noise-adjust` all-metrics regression net (which still runs
to catch unknown-unknown regressions) — it is an explicit contract
check, not a whitelist.

Bind each gate to a `const` and list it on the macro:

```rust,ignore
use ktstr::prelude::*;

// Name any metric from `cargo ktstr stats list-metrics`.
const SPREAD_GATE: PerfDeltaAssertion =
    PerfDeltaAssertion::new("worst_spread").with_max_regression_pct(5.0);

#[ktstr_test(performance_mode = true, perf_delta_assertions = [SPREAD_GATE])]
fn schbench_steady() -> Scenario {
    // ... a degenerate / steady-state scenario whose worst_spread
    // must not regress more than 5% against the baseline commit.
}
```

Builders (all `const fn`, chainable):

- `.with_max_regression_pct(pct)` — relative gate: a worsening move
  larger than `pct`% of the baseline gates. Unset → registry
  `default_rel`.
- `.with_min_abs(min)` — absolute-materiality floor: a move smaller
  than `min` (in the metric's units) never gates. Unset → registry
  `default_abs`.
- `.with_direction(polarity)` — pin the regression direction instead
  of inheriting the registry polarity (e.g. treat an `Informational`
  metric as `LowerBetter` for this test).
- `.with_phase(step_index)` — scope the gate to one phase
  (`0` = BASELINE, `1..=N` = scenario Step ordinals) instead of the
  whole-run value.

Then gate CI with the noise-adjusted compare:

```sh
cargo ktstr perf-delta --noise-adjust 5 --kernel 7.0 \
    -E 'test(schbench_steady)'
```

This runs `schbench_steady` five times at HEAD and five at the
baseline, and fails when `worst_spread` regresses past the declared
5% gate with statistical confidence. See
[Runs and Regression Gates](../running-tests/runs.md) for the full
perf-delta workflow and the [CI chapter](../ci.md) for wiring the
gate into a pull-request job.
