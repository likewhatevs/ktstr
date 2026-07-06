# Core Concepts

ktstr tests compose from three layers:

<div class="kt-steps">
<div class="kt-step" data-step="1"><strong><a href="writing-tests.html#scenarios">Scenarios</a></strong><p>The condition the test creates: cgroups, CPU partitioning, workloads, and mid-run changes.</p></div>
<div class="kt-step" data-step="2"><strong><a href="concepts/work-types.html">Work types</a></strong><p>What each worker process does, with variants aimed at specific scheduler paths.</p></div>
<div class="kt-step" data-step="3"><strong><a href="concepts/checking.html">Checking</a></strong><p>How results are judged: worker progress, spread, stuck gaps, monitor thresholds, and temporal patterns.</p></div>
</div>

One test, all three layers visible:

```rust,ignore
#[ktstr_test(
    scheduler = MY_SCHED,             // scheduler under test
    llcs = 2, cores = 4, threads = 1, // topology the VM boots with
    not_stuck = true,               // checking: every worker progressed
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

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong><a href="concepts/ops.html">Ops, Steps, and Backdrop</a></strong><p>The lower-level API behind scenarios, used when state changes mid-run.</p></div>
<div class="kt-doc-card"><strong><a href="concepts/topology.html">Topology</a></strong><p>The NUMA/LLC/core/thread layout a test declares and the VM actually boots.</p></div>
<div class="kt-doc-card"><strong><a href="concepts/topology.html#memory-policy">MemPolicy</a></strong><p>Per-worker NUMA memory placement for locality-sensitive tests.</p></div>
<div class="kt-doc-card"><strong><a href="concepts/performance-mode.html">Performance Mode</a></strong><p>Host-side isolation for noise-sensitive measurements.</p></div>
<div class="kt-doc-card"><strong><a href="concepts/resource-budget.html">Resource Budget</a></strong><p>How concurrent VMs and kernel builds share host CPUs safely.</p></div>
</div>

Read [Scenarios](writing-tests.md#scenarios), [Work
types](concepts/work-types.md), and [Checking](concepts/checking.md)
first — every test touches all three. [Ops](concepts/ops.md) matters
once a canned scenario stops being enough, and
[Topology](concepts/topology.md) once placement behavior is under
test. Performance Mode and Resource Budget are operational: read them
when measurements get noisy or hosts get shared.
