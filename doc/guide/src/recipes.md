# Recipes

Task-oriented walkthroughs. Each recipe is self-contained: pick the
one that matches your problem and follow it top to bottom. For the
model behind the commands, read [Core Concepts](concepts.md); for
flag-by-flag detail, the [Running Tests](running-tests.md) chapters.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong><a href="recipes/test-new-scheduler.html">First scheduler test</a></strong><p>Define a scheduler, run a smoke scenario, then add checks.</p></div>
<div class="kt-doc-card"><strong><a href="recipes/investigate-crash.html">Crash investigation</a></strong><p>Use the failure dump and auto-repro trail to turn a crash into a small regression test.</p></div>
<div class="kt-doc-card"><strong><a href="recipes/diagnose-slow-scheduler.html">Performance regression</a></strong><p>Capture host scheduler state with <code>ctprof</code>, compare runs, and find the changed workload.</p></div>
</div>

> [!NOTE]
> Two binaries appear below. `cargo ktstr <subcommand>` is the
> host-side cargo wrapper for test workflows; bare `ktstr` is the
> guest-init binary that doubles as a host CLI for a few tools
> (`ctprof`, `topo`, `locks`). Both install with `cargo install
> ktstr`. See [cargo ktstr](running-tests/cargo-ktstr.md) and
> [the standalone ktstr binary](running-tests/cargo-ktstr.md#the-standalone-ktstr-binary).

## Which recipe do I want?

| Symptom | Recipe |
|---|---|
| I have a scheduler binary and no tests | [Test a New Scheduler](recipes/test-new-scheduler.md) |
| A test failed and the scheduler died | [Investigate a Crash](recipes/investigate-crash.md) |
| Default checks don't fit my scheduler — or nothing is checked at all | [Customize Checking](recipes/custom-checking.md) |
| I want gates that catch performance regressions — and proof they fire | [Benchmark Gates and Negative Tests](recipes/benchmarking-tests.md) |
| Is my scheduler at least as good as the kernel default? | [Compare a Scheduler vs EEVDF](recipes/scheduler-vs-eevdf.md) |

Three recipes compare two runs. They answer different questions:

| Two runs differ because… | Recipe |
|---|---|
| …the scheduler source changed (branch vs baseline commit) | [A/B Compare Branches](recipes/ab-compare.md) |
| …a workload got slower even though tests still pass | [Diagnose a Slow Scheduler with ctprof](recipes/diagnose-slow-scheduler.md) |
| …the host changed (machine, reboot, sysctl drift) | [Capture and Compare Host State](recipes/host-state.md) |

## All recipes

In rough lifecycle order:

- [Test a New Scheduler](recipes/test-new-scheduler.md) — define the
  scheduler, write tests, sweep the BPF verifier, host the tests in
  your own crate
- [Investigate a Crash](recipes/investigate-crash.md) — read the
  crash report, use auto-repro, pin the bug as a regression test
- [A/B Compare Branches](recipes/ab-compare.md) — `cargo ktstr
  perf-delta` between HEAD and a baseline commit
- [Capture and Compare Host State](recipes/host-state.md) —
  `cargo ktstr show-host` snapshots and the perf-delta host-delta
  section
- [Diagnose a Slow Scheduler with ctprof](recipes/diagnose-slow-scheduler.md) —
  per-thread off-CPU diff between two `ktstr ctprof` snapshots
- [Customize Checking](recipes/custom-checking.md) — scheduler-level
  thresholds, per-test overrides, merge order
- [Benchmark Gates and Negative Tests](recipes/benchmarking-tests.md) —
  performance gates plus the negative tests that prove they fire
- [Compare a Scheduler vs EEVDF](recipes/scheduler-vs-eevdf.md) —
  detach the scheduler mid-run and compare phases within one test
