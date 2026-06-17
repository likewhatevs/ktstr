# Core Concepts

ktstr tests compose from three layers:

1. **[Scenarios](concepts/scenarios.md)** -- what to test: cgroup
   layout, CPU partitioning, workloads, custom logic. Composed from
   `Step`s; each `Step` carries a list of [`Op`s](concepts/ops.md)
   plus a hold spec.

2. **[WorkType](concepts/work-types.md)** -- what each worker
   process does: CPU spin, yield, I/O, bursty patterns, pipe-based
   IPC, signal storms, lock contention, page-fault churn, etc.

3. **[Checking](concepts/checking.md)** -- how to evaluate results:
   starvation, fairness, isolation, scheduling gaps, monitor
   thresholds, temporal patterns.

These compose orthogonally. A scenario runs with every valid
topology and checks apply uniformly across all runs.

Five supporting concepts complete the picture:

- [Ops and Steps](concepts/ops.md) — the primary API for defining
  scenarios. Most tests use `CgroupDef` + `execute_defs`; tests that
  need mid-scenario state changes (snapshot, replace scheduler,
  read/write kernel memory) compose `Op`s into `Step`s and feed
  them to `execute_steps` or `execute_scenario`.
- [TestTopology](concepts/topology.md) — CPU and LLC layout for
  cpuset partitioning.
- [MemPolicy](concepts/mem-policy.md) — per-cgroup NUMA memory
  policy (Bind, Interleave, WeightedInterleave, and more) for
  tests that exercise the memory policy subsystem.
- [Performance Mode](concepts/performance-mode.md) — host-side
  isolation for noise-sensitive measurements.
- [Resource Budget](concepts/resource-budget.md) — the `--cpu-cap`
  tier that coordinates concurrent no-perf-mode VMs and kernel
  builds via LLC flocks and cgroup v2 cpuset sandboxes.
