# Scenarios

Scenarios define the scheduling conditions a test creates. Each
scenario sets up cgroups, workers, and cpusets to produce a specific
condition. The `#[ktstr_test]` harness invokes the scenario, captures
the result, and applies whichever checks the test author opted in
to (see [Checking](checking.md)).

A scenario is one of:

- A **canned** function from `scenarios::*` (the table below) —
  picked up by name and called as `scenarios::steady(ctx)`.
- A **custom** function with signature
  `fn(&Ctx) -> Result<AssertResult>` ([Custom Scenarios](../writing-tests/custom-scenarios.md)).
- A **step composition** built from `Step` + `Op` and run via
  `execute_steps(ctx, vec![...])` or `execute_scenario(ctx,
  Backdrop::from_cgroups(...), vec![...])` for scenarios that need
  cross-step state (a long-lived `Backdrop` cgroup set, payload
  handles, kernel-write seeds — see [Ops and Steps](ops.md)).

## Canned scenarios (`scenarios::*`)

`ktstr::scenario::scenarios` provides curated scenario functions that
can be called directly from `#[ktstr_test]`:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

| Function | Condition tested | Setup |
|---|---|---|
| `steady` | Baseline fairness | 2 cgroups, no cpusets, equal CPU-spin load |
| `steady_llc` | LLC-boundary scheduling | 2 cgroups with LLC-aligned cpusets |
| `oversubscribed` | Dispatch under oversubscription | 2 cgroups, 32 mixed workers each |
| `cpuset_apply` | Cpuset assignment on running tasks | Disjoint cpusets applied mid-run |
| `cpuset_clear` | Cpuset removal on confined tasks | Cpusets cleared mid-run |
| `cpuset_resize` | Cpuset resizing adaptation | Cpusets shrink then grow |
| `cgroup_add` | New cgroup appearance | Cgroups added mid-run |
| `cgroup_remove` | Cgroup removal while others run | Cgroups removed mid-run |
| `affinity_change` | Affinity mask changes | Worker affinities randomized mid-run |
| `affinity_pinned` | Narrow-affinity contention | Workers pinned to 2-CPU subset |
| `host_contention` | Fairness between cgroup and host tasks | Host workers vs cgroup workers |
| `mixed_workloads` | Mixed workload fairness | Heavy + bursty + IO cgroups |
| `nested_steady` | Nested cgroup hierarchy | Workers in nested sub-cgroups |
| `nested_task_move` | Cross-level task migration | Tasks moved between nested cgroups |

Additional `custom_*` functions are available in
`ktstr::scenario::{affinity, basic, cpuset, dynamic, interaction,
nested, performance, stress}`. See the
[API docs](https://ktstr.dev/rustdoc/ktstr/scenario/index.html)
for the full list.

Most tests use these canned functions or build custom scenarios with
`CgroupDef` and the executors in `ktstr::scenario::ops`. The five
executor entry points (all in the prelude):

- `execute_defs(ctx, Vec<CgroupDef>)` — one-shot cgroup setup, run
  for the full duration, collect reports.
- `execute_steps(ctx, Vec<Step>)` / `execute_steps_with(ctx,
  Vec<Step>, Option<&Assert>)` — multi-step composition without a
  long-lived backdrop.
- `execute_scenario(ctx, Backdrop, Vec<Step>)` /
  `execute_scenario_with(ctx, Backdrop, Vec<Step>, Option<&Assert>)`
  — full composition. Use when a `Backdrop` (long-lived cgroups,
  persistent payloads, kernel-write seeds) must coexist with
  per-step ops; see the [Backdrop](ops.md) guide.

Custom scenarios receive a `Ctx` reference; see
[Custom Scenarios](../writing-tests/custom-scenarios.md) for the
`Ctx` struct and helper functions.
