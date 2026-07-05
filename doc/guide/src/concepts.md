# Core Concepts

ktstr tests compose from three layers:

1. **[Scenarios](concepts/scenarios.md)** — the scheduling condition
   the test creates: cgroup layout, CPU partitioning, workloads,
   mid-run changes.
2. **[Work types](concepts/work-types.md)** — what each worker
   process does, each variant targeting a specific kernel scheduling
   path.
3. **[Checking](concepts/checking.md)** — how results are judged:
   starvation, fairness, gaps, monitor thresholds, temporal patterns.

One test, all three layers visible:

```rust,ignore
#[ktstr_test(
    scheduler = MY_SCHED,             // scheduler under test
    llcs = 2, cores = 4, threads = 1, // topology the VM boots with
    not_starved = true,               // checking: every worker progressed
    max_spread_pct = 20.0,            // checking: fairness bound
)]
fn steady_two_cells(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)  // scenario: 2 cgroups of CPU-spin workers
                            // (work type: the default SpinWait)
}
```

The layers compose orthogonally: the same scenario body runs across
every topology a [gauntlet](running-tests/gauntlet.md) sweep
declares, and the checks apply uniformly to every variant.

Five more concepts round out the picture:

- **[Ops, Steps, and Backdrop](concepts/ops.md)** — the API scenarios
  are built from. Most tests declare cgroups with `CgroupDef`; tests
  that change state mid-run compose `Op`s into `Step`s.
- **[Topology](concepts/topology.md)** — the NUMA/LLC/core/thread
  layout a test declares and the VM actually boots with.
- **[MemPolicy](concepts/mem-policy.md)** — per-worker NUMA memory
  placement, for tests that measure memory locality.
- **[Performance Mode](concepts/performance-mode.md)** — host-side
  isolation for noise-sensitive measurements.
- **[Resource Budget](concepts/resource-budget.md)** — how concurrent
  VMs and kernel builds share host CPUs safely.

Read [Scenarios](concepts/scenarios.md), [Work
types](concepts/work-types.md), and [Checking](concepts/checking.md)
first — every test touches all three. [Ops](concepts/ops.md) matters
once a canned scenario stops being enough, and
[Topology](concepts/topology.md) once placement behavior is under
test. Performance Mode and Resource Budget are operational: read them
when measurements get noisy or hosts get shared.
