# Assertable Metrics

Every regression comparison — `cargo ktstr perf-delta` and the
per-test [`PerfDeltaAssertion`](#perfdeltaassertion-how-to) gate — is
driven by the **metric registry**: the static `ktstr::stats::METRICS`
table. Each entry (`MetricDef`) carries a metric's name, its
regression **polarity**, its aggregation **kind**, the **dual-gate**
significance thresholds, and a display unit. This chapter explains the
registry fields, how to enumerate the live catalog, which workloads
emit which metric families, and how to pin a per-test regression gate.

## The catalog: `stats list-metrics`

The authoritative, always-current catalog is the command output — it
enumerates `METRICS` directly, so it never drifts from the code:

```sh
cargo ktstr stats list-metrics          # text table: NAME / POLARITY / DEFAULT_ABS / DEFAULT_REL / UNIT
cargo ktstr stats list-metrics --json   # machine-readable (includes kind + every field) for dashboards / tooling
```

`list-metrics` reads only the static registry; it needs no sidecar
pool. To see which of these metrics a *particular* run pool actually
carries, use [`cargo ktstr stats list-values`](../running-tests/cargo-ktstr.md#list-values).

## Registry fields

Each `MetricDef` row means:

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
    `unknown` by `list-metrics`) but no registered metric uses them today.
- **kind** — how per-sample readings fold into the run-level value
  (`Counter`, `Peak`, `Gauge`, `Rate`, and the phase-aware
  reductions). The kind decides whether the cross-run fold is a mean,
  a max, or a re-derived ratio.
- **default_abs** / **default_rel** — the **dual gate**. A move counts
  as a confident regression only when it clears BOTH the absolute
  floor (`default_abs`, in the metric's units) AND the relative
  threshold (`default_rel`, a fraction). A tiny absolute move on a
  huge relative base — or a large relative move below the absolute
  floor — is not material. `perf-delta --threshold PCT` / `--policy
  FILE` override the relative gate; the absolute gate is per-metric.
- **display_unit** — the unit rendered in tables (`ms`, `/s`, `ns`, …).

## Workload → emitted metrics

A metric only appears in a comparison if the run actually emitted it.
The metric FAMILIES map to workload categories roughly as:

- **Spread / gap** (`worst_spread`, `worst_gap_ms`) —
  emitted by every scenario from the scheduling-latency capture; the
  primary fairness/latency signal.
- **Iteration throughput** (`total_iterations`,
  `worst_iterations_per_cpu_sec`) — emitted by compute/spin workloads;
  the overcommit-invariant `*_per_cpu_sec` form is the one to compare
  when host-CPU budgets differ between runs.
- **schedstat counters / rates** (`total_run_delay_ns_per_sched`,
  `*_per_sec` families, `total_sched_count`, `total_ttwu_count`) —
  derived from `/proc` schedstat over the run; present whenever
  schedstat capture is enabled.
- **cgroup / pressure** — emitted by cgroup-exercising scenarios and
  the periodic host-pressure capture.

Because emission is data-dependent, treat `stats list-values` (what a
pool carries) as ground truth for "will this metric be there?", and
`list-metrics` as the registry of everything comparable.

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

A declaration names a registry metric and OVERRIDES, for this test,
the gate that decides a confident regression on it. It LAYERS ON TOP
of the `--noise-adjust` all-metrics regression net (which still runs
to catch unknown-unknown regressions) — it is an explicit contract
check, not a whitelist.

Bind each gate to a `const` and list it on the macro:

```rust
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
cargo ktstr perf-delta --noise-adjust 5 --kernel ../linux \
    -E 'test(schbench_steady)'
```

This runs `schbench_steady` five times at HEAD and five at the
baseline, and fails when `worst_spread` regresses past the declared
5% gate with statistical confidence. See
[perf-delta](../running-tests/cargo-ktstr.md#perf-delta) for the full
comparison flag set.
